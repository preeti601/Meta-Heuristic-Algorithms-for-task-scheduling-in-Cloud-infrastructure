import random
import statistics
import time
import pandas as pd

from main import Task, VM, schedule_tasks, calculate_metrics
from scheduler.ga import genetic_algorithm
from scheduler.pso import particle_swarm_optimization
from scheduler.laas import load_aware_scheduler
from scheduler.ecotaskopt import eco_task_opt
from hera import hera_schedule

WORKLOAD_SIZES = [50, 100, 500]
TRIALS = 30
RUNTIME_REPEATS = 1  # median of repeated timings reduces timer noise
VMS = [
    VM(1, 2000, 0.05, 70, 140),
    VM(2, 3000, 0.08, 90, 180),
    VM(3, 1500, 0.03, 50, 100),
]
ALGORITHMS = [
    "FCFS", "ROUND_ROBIN", "MIN_MIN", "GENETIC ALGORITHM", "PSO", "LAAS",
    "ECOTASKOPT", "HERA",
]


def generate_tasks(number_of_tasks, seed):
    rng = random.Random(seed)
    return [Task(i + 1, rng.randint(5000, 30000)) for i in range(number_of_tasks)]


def run_algorithm(name, tasks, seed=None):
    if seed is not None:
        random.seed(seed)
    if name in ("FCFS", "ROUND_ROBIN", "MIN_MIN"):
        return schedule_tasks(tasks, VMS, name.lower())
    if name == "GENETIC ALGORITHM":
        return genetic_algorithm(tasks, VMS)
    if name == "PSO":
        return particle_swarm_optimization(tasks, VMS)
    if name == "LAAS":
        return load_aware_scheduler(tasks, VMS)
    if name == "ECOTASKOPT":
        return eco_task_opt(tasks, VMS, seed=seed, iterations=30)
    if name == "HERA":
        # HERA returns task->VM assignments; reconstruct the same schedule
        # format used by the other algorithms so all reported metrics use
        # main.calculate_metrics().
        hera_tasks = [
            {"id": task.task_id, "length": task.length_mi}
            for task in tasks
        ]
        hera_vms = [
            {"id": vm.vm_id, "speed": vm.mips,
             "power": vm.max_watts, "cost_rate": vm.cost_per_second}
            for vm in VMS
        ]
        # Keep HERA tractable for the 500-task workload. These settings are
        # recorded in the report and should be held fixed for reproducibility.
        n = len(tasks)
        iterations = 4 if n <= 100 else 1
        max_moves = 20 if n <= 100 else 5
        max_swap_checks = 100 if n <= 100 else 20
        result = hera_schedule(
            hera_tasks, hera_vms, seed=seed,
            iterations=iterations, max_moves=max_moves,
            max_swap_checks=max_swap_checks
        )
        vm_lookup = {vm.vm_id: vm for vm in VMS}
        task_lookup = {task.task_id: task for task in tasks}
        availability = {vm.vm_id: 0.0 for vm in VMS}
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
                "cost": duration * vm.cost_per_second
            })
        return pd.DataFrame(records)
    raise ValueError(f"Unknown algorithm: {name}")


def measure_runtime(name, tasks, seed):
    durations = []
    for repeat in range(RUNTIME_REPEATS):
        start = time.perf_counter()
        run_algorithm(name, tasks, seed + repeat * 100_003)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def main():
    rows = []
    total = len(WORKLOAD_SIZES) * TRIALS * len(ALGORITHMS)
    completed = 0

    for size in WORKLOAD_SIZES:
        for trial in range(1, TRIALS + 1):
            tasks = generate_tasks(size, seed=size * 1000 + trial)
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                completed += 1
                seed = size * 1_000_000 + trial * 100 + algorithm_index
                schedule = run_algorithm(algorithm, tasks, seed)
                metrics = calculate_metrics(schedule, VMS)
                runtime = measure_runtime(algorithm, tasks, seed + 50_000_000)
                rows.append({
                    "Tasks": size,
                    "Trial": trial,
                    "Algorithm": algorithm,
                    **metrics,
                    "Algorithm Runtime (seconds)": runtime,
                })
                print(
                    f"[{completed}/{total}] tasks={size}, trial={trial:02d}, "
                    f"{algorithm}: makespan={metrics['Makespan (seconds)']:.4f}s, "
                    f"median_runtime={runtime:.6f}s"
                )

    results = pd.DataFrame(rows)
    results.to_csv("experiment_results.csv", index=False, float_format="%.10f")

    metric_columns = [
        "Makespan (seconds)", "Average task execution time (seconds)",
        "Execution cost", "Estimated energy (joules)",
        "Aggregate CPU utilization (%)", "Algorithm Runtime (seconds)",
    ]
    summary = results.groupby(["Tasks", "Algorithm"])[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric} - {stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv("experiment_summary.csv", index=False, float_format="%.10f")

    print("\n===== AVERAGE RESULTS ACROSS TRIALS =====")
    display = summary.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(4)
    print(display.to_string(index=False))
    print("\nSaved full-precision detailed results to experiment_results.csv")
    print("Saved mean and sample standard deviation to experiment_summary.csv")


if __name__ == "__main__":
    main()
