"""Paired statistical analysis for experiment_results.csv.

Run from the project directory:
    python analyze_paired_stats.py

For each task count, compares algorithms on matching Trial IDs using paired
Wilcoxon signed-rank tests. Holm-adjusted p-values are calculated separately
for each workload and metric. Negative paired deltas (algorithm - reference)
mean the algorithm has a lower metric than the reference.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

INPUT = Path("experiment_results.csv")
OUTPUT = Path("paired_statistical_results.csv")
METRICS = [
    "Makespan (seconds)",
    "Execution cost",
    "Estimated energy (joules)",
    "Aggregate CPU utilization (%)",
    "Algorithm Runtime (seconds)",
]
REFERENCES = ["ECOTASKOPT", "HERA"]

def paired_test(a, b):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return len(delta), np.nan, np.nan, np.nan, np.nan
    mean_delta = float(delta.mean())
    if len(delta) < 2 or np.allclose(delta, 0):
        p = 1.0 if len(delta) >= 2 else np.nan
    else:
        try:
            p = float(wilcoxon(delta, alternative="two-sided",
                               zero_method="wilcox").pvalue)
        except ValueError:
            p = 1.0
    return len(delta), mean_delta, float(np.median(delta)), float(np.quantile(delta, .025)), float(np.quantile(delta, .975)), p

def main():
    if not INPUT.exists():
        raise SystemExit(f"Cannot find {INPUT.resolve()}; run python run_experiments.py first.")
    df = pd.read_csv(INPUT)
    required = {"Tasks", "Trial", "Algorithm", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if "HERA" not in set(df["Algorithm"]):
        raise SystemExit("HERA results are missing. Run python run_experiments.py first.")

    rows = []
    for tasks in sorted(df["Tasks"].unique()):
        sub = df[df.Tasks == tasks]
        wide = sub.pivot(index="Trial", columns="Algorithm", values=METRICS)
        for metric in METRICS:
            if (metric, "ECOTASKOPT") not in wide.columns or (metric, "HERA") not in wide.columns:
                continue
            for reference in REFERENCES:
                for algorithm in sorted(sub.Algorithm.unique()):
                    if algorithm == reference:
                        continue
                    if (metric, algorithm) not in wide.columns:
                        continue
                    pair = wide[[ (metric, algorithm), (metric, reference) ]].dropna()
                    # paired delta: candidate minus reference
                    result = paired_test(pair[(metric, algorithm)], pair[(metric, reference)])
                    if len(result) == 5:
                        n, mean_delta, median_delta, ci_low, ci_high = result
                        p_value = np.nan
                    else:
                        n, mean_delta, median_delta, ci_low, ci_high, p_value = result
                    rows.append({
                        "Tasks": int(tasks), "Metric": metric,
                        "Reference": reference, "Compared algorithm": algorithm,
                        "Paired trials": n, "Mean delta (algorithm - reference)": mean_delta,
                        "Median delta": median_delta, "Delta 2.5% quantile": ci_low,
                        "Delta 97.5% quantile": ci_high, "Wilcoxon p-value": p_value,
                    })
    out = pd.DataFrame(rows)
    out["Holm-adjusted p-value"] = np.nan
    out["Significant after Holm (alpha=0.05)"] = False
    for _, idx in out.groupby(["Tasks", "Metric", "Reference"]).groups.items():
        vals = out.loc[idx, "Wilcoxon p-value"].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.any():
            positions = np.array(list(idx))[valid]
            raw = vals[valid]
            order = np.argsort(raw)
            ranked = raw[order]
            m = len(ranked)
            adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked)
            adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
            adjusted = np.empty(m, dtype=float)
            adjusted[order] = adjusted_sorted
            out.loc[positions, "Holm-adjusted p-value"] = adjusted
    out["Significant after Holm (alpha=0.05)"] = out["Holm-adjusted p-value"] < 0.05
    out.to_csv(OUTPUT, index=False, float_format="%.10g")
    print(f"Saved {OUTPUT.resolve()}")
    print("Negative delta means the compared algorithm has a lower value than the reference.")
    print(out[(out.Metric == "Makespan (seconds)") &
              (out.Reference.isin(["ECOTASKOPT", "HERA"]))].to_string(index=False))

if __name__ == "__main__":
    main()
