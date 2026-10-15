"""Statistical summary and paired algorithm comparison for CSV trial results."""
import argparse
from pathlib import Path
import math
import pandas as pd

def mean_ci95(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    # Normal approximation; for n>=30 this is reasonable. For small n, use as a screening CI only.
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return mean, sd, mean-half if n > 1 else float("nan"), mean+half if n > 1 else float("nan"), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to experiment results CSV")
    ap.add_argument("--out", default="results/research_gap")
    ap.add_argument("--metric", default="makespan")
    ap.add_argument("--baseline", default="FCFS")
    args = ap.parse_args()
    src = Path(args.csv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    required = {"algorithm", args.metric}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {sorted(required)}")
    group_cols = [c for c in ["tasks", "vms", "algorithm"] if c in df.columns]
    summary_rows = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        mean, sd, low, high, n = mean_ci95(g[args.metric])
        row = dict(zip(group_cols, key))
        row.update(metric=args.metric, mean=mean, std=sd, ci95_low=low, ci95_high=high, n=n)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out/"confidence_intervals.csv", index=False)

    # Pairing is by task/vm/trial and only valid if rows share the same workload/trial identifiers.
    pair_cols = [c for c in ["tasks", "vms", "trial", "seed"] if c in df.columns]
    comparisons = []
    if pair_cols:
        base = df[df.algorithm.astype(str).str.upper() == args.baseline.upper()]
        for alg in sorted(set(df.algorithm.astype(str)) - {args.baseline}):
            other = df[df.algorithm.astype(str) == alg]
            merged = base[pair_cols+[args.metric]].merge(
                other[pair_cols+[args.metric]], on=pair_cols, suffixes=("_base", "_candidate"))
            if merged.empty:
                continue
            delta = merged[f"{args.metric}_candidate"] - merged[f"{args.metric}_base"]
            row = {"baseline": args.baseline, "algorithm": alg, "metric": args.metric,
                   "paired_n": len(delta), "mean_delta_candidate_minus_baseline": float(delta.mean())}
            try:
                from scipy.stats import wilcoxon
                stat, p = wilcoxon(delta, alternative="two-sided", method="approx")
                row.update(wilcoxon_statistic=float(stat), p_value=float(p))
            except ImportError:
                row.update(wilcoxon_statistic=float("nan"), p_value=float("nan"),
                           note="Install scipy for Wilcoxon test")
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv", index=False)
    print("Wrote", out/"confidence_intervals.csv")
    print("Wrote", out/"paired_comparisons.csv")
    print("Interpretation: paired tests require identical workloads and trial IDs across algorithms.")

if __name__ == "__main__":
    main()
