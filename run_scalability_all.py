"""Run comparable scalability experiments for all six schedulers.

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

ALGORITHMS = ["FCFS", "ROUND_ROBIN", "MIN_MIN", "LAAS_LPT", "GA", "PSO", "ECOTASKOPT"]


def make_vms(count):
    # Deterministic heterogeneous VM pool, cycled to support arbitrary counts.
    templates = [
        (2000, 0.05, 70, 140),
        (3000, 0.08, 90, 180),
        (1500, 0.03, 50, 100),
        (2500, 0.06, 80, 160),
        (3500, 0.10, 100, 200),
    ]
    return [VM(i + 1, *templates[i % len(templates)]) for i in range(count)]


def make_tasks(count, seed):
    rng = random.Random(seed)
    return [Task(i + 1, rng.randint(5000, 30000)) for i in range(count)]


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
    raise ValueError(f"Unknown algorithm: {name}")


def main():
    parser = argparse.ArgumentParser(description="Six-algorithm scalability experiment")
    parser.add_argument("--tasks", nargs="+", type=int, default=[50, 100, 500, 1000])
    parser.add_argument("--vms", nargs="+", type=int, default=[3, 10, 25])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--population", type=int, default=30,
                        help="GA/PSO population size (lower this to speed up tests)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="GA generations, PSO iterations, and EcoTaskOpt iterations")
    parser.add_argument("--ants", type=int, default=10,
                        help="Number of ACO ants used by EcoTaskOpt")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if any(n <= 0 for n in args.tasks + args.vms) or args.trials <= 0:
        parser.error("Task counts, VM counts, and trials must all be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.tasks) * len(args.vms) * args.trials * len(ALGORITHMS)
    completed = 0

    for task_count in args.tasks:
        for vm_count in args.vms:
            vms = make_vms(vm_count)
            for trial in range(1, args.trials + 1):
                workload_seed = 2026 + trial - 1 + task_count * 1000 + vm_count * 100
                tasks = make_tasks(task_count, workload_seed)
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
                            iterations=args.iterations, seed=algorithm_seed)
                    else:
                        schedule = run_algorithm(algorithm, tasks, vms)
                    runtime = time.perf_counter() - start
                    metrics = calculate_metrics(schedule, vms)
                    completed += 1
                    rows.append({
                        "tasks": task_count,
                        "vms": vm_count,
                        "trial": trial,
                        "seed": workload_seed,
                        "workload_seed": workload_seed,
                        "algorithm_seed": algorithm_seed,
                        "dynamic_arrivals": False,
                        "algorithm": algorithm,
                        "algorithm_runtime_sec": runtime,
                        "makespan": metrics["Makespan (seconds)"],
                        "avg_execution_time": metrics["Average task execution time (seconds)"],
                        "cost": metrics["Execution cost"],
                        "energy_joules_est": metrics["Estimated energy (joules)"],
                        "utilization_pct": metrics["Aggregate CPU utilization (%)"],
                    })
                    print(f"[{completed}/{total}] tasks={task_count}, vms={vm_count}, "
                          f"trial={trial}, {algorithm}: makespan="
                          f"{metrics['Makespan (seconds)']:.3f}s, runtime={runtime:.4f}s")

    results = pd.DataFrame(rows)
    results_path = out_dir / "scalability_results_all.csv"
    results.to_csv(results_path, index=False, float_format="%.10f")

    metric_cols = ["makespan", "algorithm_runtime_sec", "avg_execution_time",
                   "cost", "energy_joules_est", "utilization_pct"]
    summary = results.groupby(["tasks", "vms", "algorithm"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = out_dir / "scalability_summary_all.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")

    print("\nExperiment complete.")
    print(f"Detailed results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(results)}; algorithms: {', '.join(ALGORITHMS)}")
    print("Note: energy is a model-based estimate; workloads are synthetic and static.")
    print("ECOTASKOPT is an EcoTaskOpt-inspired ACO-PSO adaptation, not a CloudSim reproduction.")


if __name__ == "__main__":
    main()
