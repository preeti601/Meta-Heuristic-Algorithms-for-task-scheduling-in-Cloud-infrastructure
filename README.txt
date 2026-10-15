Cloud Task Scheduling Capstone - Updated Experiment Version

Files:
  main.py                      Core task/VM models and full-precision metrics
  run_experiments.py           30 trials for 50, 100, and 500 tasks
  plot_experiments.py          Creates five comparison graphs with error bars
  scheduler/ga.py              Genetic algorithm
  scheduler/pso.py             Discrete PSO-inspired assignment heuristic
  scheduler/laas.py            Updated global-load-aware greedy heuristic

Run from this folder in VS Code terminal:
  python -m pip install pandas matplotlib
  python main.py
  python run_experiments.py
  python plot_experiments.py

The experiment script performs 540 algorithm/trial evaluations and separately
measures runtime using the median of three repeated runs. This may take several
minutes, especially for PSO on 500-task workloads.

Important interpretation notes:
- CSV metrics are saved at full precision; only terminal output is rounded.
- Runtime is the median of three repeated executions for the same workload and
  trial; stochastic algorithms are re-seeded for reproducibility.
- Energy is a simplified estimate based on configured idle/max VM power, not
  measured physical cloud energy.
- LAAS is a greedy heuristic, not guaranteed to be globally optimal.
- The PSO implementation is a discrete PSO-inspired adaptation, not canonical
  continuous PSO.
