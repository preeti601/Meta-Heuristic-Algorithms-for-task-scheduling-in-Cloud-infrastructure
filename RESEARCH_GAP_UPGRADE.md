# Research-gap upgrade for ScaleSched

This is an additive extension. It does not overwrite the existing project.

## Research gaps taken from the EcoTaskOpt paper
The paper's conclusion (PDF pp. 32–33) explicitly proposes:
1. Reinforcement learning or continual learning for better adaptation to workload changes.
2. Statistical testing on containerized cloud-native platforms such as Kubernetes.
3. Multi-tenant fairness.
4. Network-aware scheduling.
5. Carbon-footprint computation.

The paper also describes metaheuristic overhead as a scalability/real-time concern (PDF p. 31). This extension addresses these as staged research objectives rather than claiming all are solved.

## What this extension implements now
- Dynamic task arrivals with per-task deadlines, tenant, data size, memory demand and I/O intensity.
- Additional metrics: response time, deadline-violation rate, tenant fairness (Jain's index), network-transfer energy estimate and carbon estimate using a configurable carbon-intensity trace.
- Statistical analysis for paired trials: mean, standard deviation, 95% confidence interval, Wilcoxon signed-rank test (when SciPy is installed), and paired effect size.
- A reproducible experiment schema and a gap-to-evidence checklist.

## What is still future work
- Actual Kubernetes/container deployment and live measurements.
- Trace-driven experiments using a legally obtained Google Cluster Data subset.
- A trained RL/continual-learning controller.
- Calibrated network/thermal/energy coefficients from real infrastructure.
- Multi-tenant fairness must be measured using actual per-tenant completion/service outcomes.

Do not claim the extension reproduces EcoTaskOpt or proves superiority over it. Compare algorithms on the same workload seeds and report limitations.
