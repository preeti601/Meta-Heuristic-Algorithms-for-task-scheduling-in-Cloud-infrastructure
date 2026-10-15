"""
Create comparison diagrams from the scalability experiment CSV.
Run from the project root:
    py plot_eco_results.py --input results_eco_local/scalability_results_all.csv
Charts are saved under results_eco_local/plots/.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_eco_local/scalability_results_all.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path}. Run the experiment first, then pass the CSV path with --input."
        )

    df = pd.read_csv(csv_path)
    required = {"tasks", "vms", "algorithm", "makespan", "algorithm_runtime_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One chart per task/VM configuration, comparing makespan.
    for (task_count, vm_count), group in df.groupby(["tasks", "vms"], sort=True):
        means = group.groupby("algorithm", sort=False)["makespan"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        means.plot(kind="bar", ax=ax)
        ax.set_title(f"Makespan comparison ({int(task_count)} tasks, {int(vm_count)} VMs)")
        ax.set_xlabel("Scheduling algorithm")
        ax.set_ylabel("Makespan (seconds)")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"makespan_tasks_{int(task_count)}_vms_{int(vm_count)}.png", dpi=160)
        plt.close(fig)

        runtimes = group.groupby("algorithm", sort=False)["algorithm_runtime_sec"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        runtimes.plot(kind="bar", ax=ax)
        ax.set_title(f"Scheduler runtime ({int(task_count)} tasks, {int(vm_count)} VMs)")
        ax.set_xlabel("Scheduling algorithm")
        ax.set_ylabel("Runtime (seconds)")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"runtime_tasks_{int(task_count)}_vms_{int(vm_count)}.png", dpi=160)
        plt.close(fig)

    # Optional metrics, if included in this particular CSV.
    optional_metrics = [
        ("energy_joules_est", "Estimated energy (J)", "energy"),
        ("utilization_pct", "Utilization (%)", "utilization"),
        ("cost", "Estimated cost", "cost"),
    ]
    for column, ylabel, filename in optional_metrics:
        if column not in df.columns:
            continue
        for (task_count, vm_count), group in df.groupby(["tasks", "vms"], sort=True):
            values = group.groupby("algorithm", sort=False)[column].mean().sort_values()
            fig, ax = plt.subplots(figsize=(10, 5.5))
            values.plot(kind="bar", ax=ax)
            ax.set_title(f"{ylabel} comparison ({int(task_count)} tasks, {int(vm_count)} VMs)")
            ax.set_xlabel("Scheduling algorithm")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=35)
            fig.tight_layout()
            fig.savefig(output_dir / f"{filename}_tasks_{int(task_count)}_vms_{int(vm_count)}.png", dpi=160)
            plt.close(fig)

    # A summary table is also written for convenient reporting.
    summary = df.groupby(["tasks", "vms", "algorithm"], as_index=False).agg(
        makespan_mean=("makespan", "mean"),
        makespan_std=("makespan", "std"),
        runtime_mean_sec=("algorithm_runtime_sec", "mean"),
    )
    summary.to_csv(output_dir / "plot_summary.csv", index=False)

    print(f"Plots saved to: {output_dir.resolve()}")
    print(f"Summary saved to: {(output_dir / 'plot_summary.csv').resolve()}")
    print("Open the PNG files to view the diagrams.")

if __name__ == "__main__":
    main()
