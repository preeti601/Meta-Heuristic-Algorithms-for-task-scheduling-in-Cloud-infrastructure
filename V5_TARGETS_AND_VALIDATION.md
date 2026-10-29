# V5: Optimization Targets and Validation Protocol

## Important: 50% is a target, not a guaranteed result
This update does not hard-code or fabricate a 50% improvement. A 50% reduction in makespan and energy, plus a 50% relative increase in utilization, may be physically infeasible for some workload/VM configurations. It must be demonstrated against a defined baseline.

## V5 change in this package
- The objective-sweep runner now records `cpu_utilization_pct` for each schedule using the simulator's existing `Aggregate CPU utilization (%)` metric.
- Utilization is reported separately. It is NOT mistakenly minimized as if lower utilization were better.
- The current optimizer still minimizes makespan, model-estimated energy and cost; the utilization field is for analysis until a correctly formulated maximize-utilization/constrained objective is implemented.
- Existing model energy is an estimate, not measured cloud-server power.

## Correct percentage definitions
For metrics where lower is better:
`reduction_pct = (baseline - proposed) / baseline * 100`

For utilization where higher is better:
`relative_improvement_pct = (proposed - baseline) / baseline * 100`
Also report absolute percentage-point change: `proposed_utilization - baseline_utilization`.

A "50% utilization improvement" is ambiguous: 50% relative improvement is not the same as +50 percentage points. Utilization is capped at 100%, so a relative 50% increase is impossible if baseline utilization exceeds 66.67%.

## Run in PowerShell
From this project folder:
```powershell
py -m pip install -r requirements.txt
py run_objective_sweep.py --tasks 100 500 --vms 3 10 --trials 10 --iterations 30 --output-dir objective_sweep_v5 --candidate-seeds-per-profile 3 --makespan-slack 0.25 --bootstrap 5000
```
If `requirements.txt` is absent in this project, install the dependencies listed in `README.txt` and the script imports (at minimum pandas, numpy, scipy, matplotlib).

Inspect utilization columns:
```powershell
py -c "import pandas as pd; d=pd.read_csv('objective_sweep_v5/objective_sweep_trials.csv'); print(d[['algorithm','cpu_utilization_pct']].groupby('algorithm').agg(['mean','std']).to_string())"
```
The CSV has LAAS-LPT baseline rows and ECOTASKOPT rows. Compare matching workload seeds and trials, not unmatched averages.

## Acceptance criteria for claiming the 50% target
For each metric, report paired per-workload differences and a confidence interval. Do not claim the target unless:
1. makespan reduction is at least 50% on the predeclared aggregation and the comparison is against the same baseline/workloads;
2. model-estimated energy reduction is at least 50%, and the model is documented;
3. utilization improvement is defined in advance (relative percent or percentage points), measured from the same runs, and reaches the target;
4. improvements are robust across workload distributions and repeated seeds;
5. all baselines and fallback rows are included transparently.

If the run does not meet the target, report the measured result and continue optimization rather than changing the metric or baseline after seeing results.

## Stress-test profile

The scalability runner also supports a predeclared stress profile without
changing the standard experiment:

```powershell
py run_scalability_all.py --tasks 100 500 --vms 3 10 `
  --task-distributions heavy_tail straggler --vm-profile heterogeneous `
  --trials 5 --iterations 30 --output-dir results_stress_validation
```

The `heterogeneous` VM profile widens the processing-speed, power, and cost
range. The `straggler` workload contains a reproducible minority of very large
tasks. Results must still be compared on matched workload seeds, and the
simple Round Robin baseline must not be presented as a substitute for the
strong LAAS-LPT baseline.
