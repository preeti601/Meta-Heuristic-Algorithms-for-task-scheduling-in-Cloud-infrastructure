"""Run comparable scalability experiments for all eight schedulers.

Run from the project root:
    py run_scalability_all.py --tasks 50 100 500 1000 --vms 3 10 25 --trials 5

Outputs:
    results/scalability_results_all.csv
    results/scalability_summary_all.csv

Energy is a simplified model-based estimate, not a real cloud measurement.
"""
import argparse
import random
import statistics
import time
from pathlib import Path

import pandas as pd

from main import Task, VM, schedule_tasks, calculate_metrics
from scheduler.ga import genetic_algorithm
from scheduler.pso import particle_swarm_optimization
from scheduler.laas import load_aware_scheduler
from scheduler.ecotaskopt import eco_task_opt
from hera import hera_schedule

ALGORITHMS = [
    "FCFS", "ROUND_ROBIN", "MIN_MIN", "LAAS_LPT", "GA", "PSO",
    "ECOTASKOPT", "HERA",
]


def make_vms(count, profile="standard"):
    """Create deterministic VM pools for standard and stress experiments."""
    if profile == "standard":
        templates = [
            (2000, 0.05, 70, 140),
            (3000, 0.08, 90, 180),
            (1500, 0.03, 50, 100),
            (2500, 0.06, 80, 160),
            (3500, 0.10, 100, 200),
        ]
    elif profile == "heterogeneous":
        # A wider capability and power range exposes poor placement decisions.
        templates = [
            (800, 0.015, 35, 75),
            (1800, 0.045, 65, 135),
            (4000, 0.16, 130, 290),
            (7000, 0.32, 210, 480),
            (12000, 0.75, 360, 850),
        ]
    else:
        raise ValueError(f"Unknown VM profile: {profile}")
    return [VM(i + 1, *templates[i % len(templates)]) for i in range(count)]


def make_tasks(count, seed, distribution="uniform"):
    """Create repeatable synthetic workloads with different task-size shapes."""
    rng = random.Random(seed)
    lengths = []
    for _ in range(count):
        if distribution == "uniform":
            value = rng.randint(5000, 30000)
        elif distribution == "bimodal":
            # Mixture of short and long tasks, common in mixed batch workloads.
            value = rng.randint(2000, 8000) if rng.random() < 0.70 else rng.randint(25000, 60000)
        elif distribution == "heavy_tail":
            # Log-normal-like positive skew, bounded to keep runs numerically stable.
            value = int(min(120000, max(1000, rng.lognormvariate(9.0, 0.85))))
        elif distribution == "straggler":
            # Mostly short jobs with a reproducible minority of very large jobs.
            # This stresses algorithms that assign by position or VM rotation.
            if rng.random() < 0.15:
                value = rng.randint(150000, 300000)
            else:
                value = rng.randint(2000, 12000)
        else:
            raise ValueError(f"Unknown task distribution: {distribution}")
        lengths.append(value)
    return [Task(i + 1, length) for i, length in enumerate(lengths)]


def run_algorithm(name, tasks, vms):
    if name == "FCFS":
        return schedule_tasks(tasks, vms, "fcfs")
    if name == "ROUND_ROBIN":
        return schedule_tasks(tasks, vms, "round_robin")
    if name == "MIN_MIN":
        return schedule_tasks(tasks, vms, "min_min")
    if name == "LAAS_LPT":
        return load_aware_scheduler(tasks, vms)
    if name == "GA":
        return genetic_algorithm(tasks, vms)
    if name == "PSO":
        return particle_swarm_optimization(tasks, vms)
    if name == "ECOTASKOPT":
        return eco_task_opt(tasks, vms)
    if name == "HERA":
        hera_tasks = [
            {"id": task.task_id, "length": task.length_mi}
            for task in tasks
        ]
        hera_vms = [
            {"id": vm.vm_id, "speed": vm.mips, "power": vm.max_watts,
             "cost_rate": vm.cost_per_second}
            for vm in vms
        ]
        n = len(tasks)
        result = hera_schedule(
            hera_tasks, hera_vms,
            iterations=4 if n <= 100 else 1,
            max_moves=20 if n <= 100 else 5,
            max_swap_checks=100 if n <= 100 else 20,
        )
        vm_lookup = {vm.vm_id: vm for vm in vms}
        task_lookup = {task.task_id: task for task in tasks}
        availability = {vm.vm_id: 0.0 for vm in vms}
        records = []
        for task_id, vm_id in result["assignment"].items():
            task = task_lookup[task_id]
            vm = vm_lookup[vm_id]
            start = availability[vm_id]
            duration = task.length_mi / vm.mips
            finish = start + duration
            availability[vm_id] = finish
            records.append({
                "task_id": task_id, "vm_id": vm_id,
                "start_time": start, "execution_time": duration,
                "finish_time": finish,
                "cost": duration * vm.cost_per_second,
            })
        return pd.DataFrame(records)
    raise ValueError(f"Unknown algorithm: {name}")


def main():
    parser = argparse.ArgumentParser(description="Eight-algorithm scalability experiment")
    parser.add_argument("--tasks", nargs="+", type=int, default=[50, 100, 500, 1000])
    parser.add_argument("--vms", nargs="+", type=int, default=[3, 10, 25])
    parser.add_argument("--task-distributions", nargs="+",
                        choices=["uniform", "bimodal", "heavy_tail", "straggler"],
                        default=["uniform", "bimodal", "heavy_tail"],
                        help="Task-length distributions to evaluate")
    parser.add_argument("--vm-profile", choices=["standard", "heterogeneous"],
                        default="standard",
                        help="VM capability/power profile for the experiment")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--population", type=int, default=30,
                        help="GA/PSO population size (lower this to speed up tests)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="GA generations, PSO iterations, and EcoTaskOpt iterations")
    parser.add_argument("--ants", type=int, default=10,
                        help="Number of ACO ants used by EcoTaskOpt")
    parser.add_argument("--objective-weights", nargs=3, type=float, default=[0.55, 0.30, 0.05],
                        metavar=("MAKESPAN", "ENERGY", "COST"),
                        help="Relative objective weights; normalized internally. Default: 0.55 0.30 0.05")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if any(n <= 0 for n in args.tasks + args.vms) or args.trials <= 0:
        parser.error("Task counts, VM counts, and trials must all be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.tasks) * len(args.vms) * len(args.task_distributions) * args.trials * len(ALGORITHMS)
    completed = 0

    for task_count in args.tasks:
      for vm_count in args.vms:
        vms = make_vms(vm_count, args.vm_profile)
        for distribution_index, distribution in enumerate(args.task_distributions):
          for trial in range(1, args.trials + 1):
                workload_seed = (2026 + trial - 1 + task_count * 1000 + vm_count * 100
                                 + distribution_index * 10_000_000)
                tasks = make_tasks(task_count, workload_seed, distribution)
                for algorithm_index, algorithm in enumerate(ALGORITHMS):
                    algorithm_seed = workload_seed + 100_000 + algorithm_index
                    random.seed(algorithm_seed)
                    start = time.perf_counter()
                    if algorithm == "GA":
                        schedule = genetic_algorithm(
                            tasks, vms, population_size=args.population,
                            generations=args.iterations)
                    elif algorithm == "PSO":
                        schedule = particle_swarm_optimization(
                            tasks, vms, population_size=args.population,
                            iterations=args.iterations)
                    elif algorithm == "ECOTASKOPT":
                        schedule = eco_task_opt(
                            tasks, vms, ants=args.ants, particles=args.population,
                            iterations=args.iterations, seed=algorithm_seed,
                            objective_weights=tuple(args.objective_weights))
                    else:
                        schedule = run_algorithm(algorithm, tasks, vms)
                    runtime = time.perf_counter() - start
                    metrics = calculate_metrics(schedule, vms)
                    completed += 1
                    rows.append({
                        "tasks": task_count,
                        "vms": vm_count,
                        "task_distribution": distribution,
                        "trial": trial,
                        "seed": workload_seed,
                        "workload_seed": workload_seed,
                        "algorithm_seed": algorithm_seed,
                        "dynamic_arrivals": False,
                        "vm_profile": args.vm_profile,
                        "algorithm": algorithm,
                        "algorithm_runtime_sec": runtime,
                        "makespan": metrics["Makespan (seconds)"],
                        "avg_execution_time": metrics["Average task execution time (seconds)"],
                        "cost": metrics["Execution cost"],
                        "energy_joules_est": metrics["Estimated energy (joules)"],
                        "utilization_pct": metrics["Aggregate CPU utilization (%)"],
                        "ecotaskopt_fallback_used": (schedule.attrs.get("fallback_used")
                                                      if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_selection_source": (schedule.attrs.get("selection_source")
                                                         if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_tasks_changed_from_lpt": (schedule.attrs.get("tasks_changed_from_lpt")
                                                               if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_lpt_local_search_checks": (schedule.attrs.get("lpt_local_search_checks")
                                                               if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_lpt_local_search_improvements": (schedule.attrs.get("lpt_local_search_improvements")
                                                                     if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_heavy_tail_detected": (schedule.attrs.get("heavy_tail_detected")
                                                            if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_effective_iterations": (schedule.attrs.get("effective_iterations")
                                                             if algorithm == "ECOTASKOPT" else None),
                        "ecotaskopt_early_stopped": (schedule.attrs.get("early_stopped")
                                                      if algorithm == "ECOTASKOPT" else None),
                        "objective_weight_makespan": (schedule.attrs.get("objective_weight_makespan")
                                                       if algorithm == "ECOTASKOPT" else None),
                        "objective_weight_energy": (schedule.attrs.get("objective_weight_energy")
                                                     if algorithm == "ECOTASKOPT" else None),
                        "objective_weight_cost": (schedule.attrs.get("objective_weight_cost")
                                                   if algorithm == "ECOTASKOPT" else None),
                    })
                    print(f"[{completed}/{total}] tasks={task_count}, vms={vm_count}, "
                          f"trial={trial}, {algorithm}: makespan="
                          f"{metrics['Makespan (seconds)']:.3f}s, runtime={runtime:.4f}s")

    results = pd.DataFrame(rows)
    results_path = out_dir / "scalability_results_all.csv"
    results.to_csv(results_path, index=False, float_format="%.10f")

    metric_cols = ["makespan", "algorithm_runtime_sec", "avg_execution_time",
                   "cost", "energy_joules_est", "utilization_pct"]
    summary = results.groupby(["tasks", "vms", "task_distribution", "algorithm"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = out_dir / "scalability_summary_all.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")

    print("\nExperiment complete.")
    print(f"Detailed results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(results)}; algorithms: {', '.join(ALGORITHMS)}")
    eco_rows = results[results["algorithm"] == "ECOTASKOPT"].copy()
    if not eco_rows.empty:
        fallback_rate = 100.0 * eco_rows["ecotaskopt_fallback_used"].astype(bool).mean()
        print(f"EcoTaskOpt exact-LPT fallback rate: {fallback_rate:.1f}%")
        print(f"Task distributions: {', '.join(args.task_distributions)}")
        diagnostics = eco_rows.groupby(["tasks", "vms", "task_distribution"]).agg(
            trials=("trial", "count"),
            fallback_rate_pct=("ecotaskopt_fallback_used", lambda x: 100.0 * x.astype(bool).mean()),
            mean_tasks_changed_from_lpt=("ecotaskopt_tasks_changed_from_lpt", "mean"),
            mean_local_search_checks=("ecotaskopt_lpt_local_search_checks", "mean"),
            mean_local_search_improvements=("ecotaskopt_lpt_local_search_improvements", "mean"),
            mean_runtime_sec=("algorithm_runtime_sec", "mean"),
        ).reset_index()
        diagnostics.to_csv(out_dir / "ecotaskopt_diagnostics.csv", index=False, float_format="%.10f")

        # Paired comparisons use the exact same workload seed/configuration for
        # EcoTaskOpt and LAAS-LPT; positive delta means EcoTaskOpt is worse.
        paired = results[results["algorithm"].isin(["ECOTASKOPT", "LAAS_LPT"])].pivot(
            index=["tasks", "vms", "task_distribution", "trial", "workload_seed"],
            columns="algorithm", values=metric_cols)
        comparison_rows = []
        for metric in metric_cols:
            if (metric, "ECOTASKOPT") not in paired.columns or (metric, "LAAS_LPT") not in paired.columns:
                continue
            delta = paired[(metric, "ECOTASKOPT")] - paired[(metric, "LAAS_LPT")]
            temp = delta.rename("delta").reset_index()
            grouped = temp.groupby(["tasks", "vms", "task_distribution"])["delta"]
            for keys, values in grouped:
                comparison_rows.append({
                    "tasks": keys[0], "vms": keys[1], "task_distribution": keys[2],
                    "metric": metric, "mean_eco_minus_laas": float(values.mean()),
                    "std_paired_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "eco_better_count": int((values < -1e-9).sum()) if metric != "utilization_pct" else int((values > 1e-9).sum()),
                    "ties_count": int((values.abs() <= 1e-9).sum()),
                    "eco_worse_count": int((values > 1e-9).sum()) if metric != "utilization_pct" else int((values < -1e-9).sum()),
                })
        pd.DataFrame(comparison_rows).to_csv(out_dir / "paired_comparisons_ecotaskopt_vs_laas.csv", index=False, float_format="%.10f")
    print(f"VM profile: {args.vm_profile}; energy is a model-based estimate; "
          "workloads are synthetic and static.")
    print("ECOTASKOPT is an EcoTaskOpt-inspired ACO-PSO adaptation, not a CloudSim reproduction.")
    print("Next run: py analyze_paired_stats.py --input-dir " + str(out_dir))


if __name__ == "__main__":
    main()
