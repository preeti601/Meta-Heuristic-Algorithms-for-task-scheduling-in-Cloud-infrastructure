# EcoTaskOpt-inspired upgrade notes

This upgrade adds `scheduler/ecotaskopt.py`, a hybrid Ant Colony Optimization + Particle Swarm Optimization scheduler inspired by:

Sanaj et al., “Energy-Efficient Task Scheduling in Cloud Computing with EcoTaskOpt: A Hybrid Ant Colony and Particle Swarm Optimization Approach,” *International Journal of Computational Intelligence Systems* (2026), DOI: 10.1007/s44196-025-01095-w.

## Paper ideas represented in this project

- ACO pheromone-based exploration of task-to-VM assignments.
- Discrete PSO-style assignment updates and information exchange between ACO and PSO.
- An adaptive search schedule across iterations.
- A multi-objective fitness function combining estimated energy, makespan, and utilization imbalance.
- A nonlinear utilization-to-power relationship inspired by the paper's power equation.
- The same generated workload is used for each algorithm within a given task/VM/trial configuration. `workload_seed` and `algorithm_seed` are recorded separately.

## Important scope limitations

This is an educational Python adaptation, **not a reproduction of the paper's EcoTaskOpt implementation**. It does not use CloudSim, Google Cluster Data 2019, DAG workflows, physical-host aggregation, network/migration energy, deadlines, live resource monitoring, cooling/PUE, or the paper's exact adaptive equations. VM-level power is a proxy. Results must not be claimed as directly comparable to the paper's reported percentages.

The paper reports 30 runs per setting and uses confidence intervals and statistical tests. The default experiment here is only 5 trials; use at least 30 trials for final research claims if runtime permits. Compare algorithms under identical workloads and report mean ± standard deviation or confidence intervals.

## Run a quick check

```powershell
py run_scalability_all.py --tasks 50 --vms 3 --trials 1 --population 8 --iterations 3 --ants 4
```

## Run the full experiment

```powershell
py run_scalability_all.py --tasks 50 100 500 1000 --vms 3 10 25 --trials 5 --population 20 --iterations 20 --ants 8
```

The full run produces seven algorithms (FCFS, Round Robin, Min-Min, LAAS-LPT, GA, PSO, ECOTASKOPT), so 4 × 3 × 5 × 7 = 420 detailed result rows.

Outputs:
- `results/scalability_results_all.csv`
- `results/scalability_summary_all.csv`

If pandas is missing: `py -m pip install pandas`.
