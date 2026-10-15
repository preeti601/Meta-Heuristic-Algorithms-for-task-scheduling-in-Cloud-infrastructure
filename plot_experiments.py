import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = Path("experiment_results.csv")
SUMMARY_FILE = Path("experiment_summary.csv")
PLOT_DIR = Path("experiment_plots")
PLOT_DIR.mkdir(exist_ok=True)

if not RESULTS_FILE.exists() or not SUMMARY_FILE.exists():
    raise FileNotFoundError("Run python run_experiments.py before plotting.")

summary = pd.read_csv(SUMMARY_FILE)
metrics = [
    ("Makespan (seconds)", "Mean makespan by workload", "Seconds", "makespan.png"),
    ("Execution cost", "Mean execution cost by workload", "Cost units", "execution_cost.png"),
    ("Estimated energy (joules)", "Mean estimated energy by workload", "Joules (estimated)", "energy.png"),
    ("Aggregate CPU utilization (%)", "Mean aggregate CPU utilization by workload", "Utilization (%)", "cpu_utilization.png"),
    ("Algorithm Runtime (seconds)", "Median-repeated mean algorithm runtime", "Seconds (log scale)", "runtime.png"),
]

for metric, title, ylabel, filename in metrics:
    mean_col = f"{metric} - mean"
    std_col = f"{metric} - std"
    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm, group in summary.groupby("Algorithm"):
        group = group.sort_values("Tasks")
        yerr = group[std_col].fillna(0) if std_col in group else None
        ax.errorbar(group["Tasks"], group[mean_col], yerr=yerr,
                    marker="o", capsize=3, linewidth=1.8, label=algorithm)
    ax.set_title(title)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["Tasks"].unique()))
    ax.grid(True, alpha=0.3)
    if metric == "Algorithm Runtime (seconds)":
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=200)
    plt.close(fig)

print(f"Saved five comparison graphs in: {PLOT_DIR.resolve()}")
