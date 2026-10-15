import random
import statistics
import time
import pandas as pd

from main import Task, VM, schedule_tasks, calculate_metrics
from scheduler.ga import genetic_algorithm
from scheduler.pso import particle_swarm_optimization
from scheduler.laas import load_aware_scheduler

WORKLOAD_SIZES = [50, 100, 500]
TRIALS = 30
RUNTIME_REPEATS = 1  # median of repeated timings reduces timer noise
VMS = [
    VM(1, 2000, 0.05, 70, 140),
    VM(2, 3000, 0.08, 90, 180),
    VM(3, 1500, 0.03, 50, 100),
]
ALGORITHMS = ["FCFS", "ROUND_ROBIN", "MIN_MIN", "GENETIC ALGORITHM", "PSO", "LAAS"]


def generate_tasks(number_of_tasks, seed):
    rng = random.Random(seed)
    return [Task(i + 1, rng.randint(5000, 30000)) for i in range(number_of_tasks)]


def run_algorithm(name, tasks):
    if name in ("FCFS", "ROUND_ROBIN", "MIN_MIN"):
        return schedule_tasks(tasks, VMS, name.lower())
    if name == "GENETIC ALGORITHM":
        return genetic_algorithm(tasks, VMS)
    if name == "PSO":
        return particle_swarm_optimization(tasks, VMS)
    if name == "LAAS":
        return load_aware_scheduler(tasks, VMS)
    raise ValueError(f"Unknown algorithm: {name}")


def measure_runtime(name, tasks, seed):
    durations = []
    for repeat in range(RUNTIME_REPEATS):
        # A distinct, reproducible seed for each algorithm/repetition.
        random.seed(seed + repeat * 100_003)
        start = time.perf_counter()
        run_algorithm(name, tasks)
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
                random.seed(seed)
                schedule = run_algorithm(algorithm, tasks)
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
