from dataclasses import dataclass
from typing import List
import pandas as pd

from scheduler.laas import load_aware_scheduler
from scheduler.pso import particle_swarm_optimization
from scheduler.ga import genetic_algorithm


@dataclass
class Task:
    task_id: int
    length_mi: float


@dataclass
class VM:
    vm_id: int
    mips: float
    cost_per_second: float
    idle_watts: float
    max_watts: float


TASKS = [
    Task(1, 12000), Task(2, 8000), Task(3, 20000), Task(4, 6000),
    Task(5, 15000), Task(6, 10000), Task(7, 18000), Task(8, 5000),
]

VMS = [
    VM(1, 2000, 0.05, 70, 140),
    VM(2, 3000, 0.08, 90, 180),
    VM(3, 1500, 0.03, 50, 100),
]


def schedule_tasks(tasks: List[Task], vms: List[VM], method: str):
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")
    if any(vm.mips <= 0 for vm in vms):
        raise ValueError("VM processing speed (mips) must be positive.")

    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []
    remaining = list(tasks)
    rr_index = 0

    while remaining:
        if method == "min_min":
            # Explicit key avoids comparing Task/VM objects when finish times tie.
            choices = []
            for task in remaining:
                for vm in vms:
                    duration = task.length_mi / vm.mips
                    finish = availability[vm.vm_id] + duration
                    choices.append((finish, task.task_id, vm.vm_id, task, vm))
            finish, _, _, task, vm = min(choices, key=lambda x: x[:3])
            remaining.remove(task)
        else:
            task = remaining.pop(0)
            if method == "round_robin":
                vm = vms[rr_index % len(vms)]
                rr_index += 1
            elif method == "fcfs":
                vm = min(vms, key=lambda x: (availability[x.vm_id], x.vm_id))
            else:
                raise ValueError(f"Unknown scheduling method: {method}")
            duration = task.length_mi / vm.mips
            finish = availability[vm.vm_id] + duration

        start = availability[vm.vm_id]
        duration = task.length_mi / vm.mips
        finish = start + duration
        availability[vm.vm_id] = finish
        records.append({
            "task_id": task.task_id, "vm_id": vm.vm_id,
            "start_time": start, "execution_time": duration,
            "finish_time": finish, "cost": duration * vm.cost_per_second,
        })

    return pd.DataFrame(records)


def calculate_metrics(schedule: pd.DataFrame, vms: List[VM]):
    """Return full-precision metrics; round only when displaying results."""
    if schedule.empty:
        raise ValueError("Cannot calculate metrics for an empty schedule.")

    makespan = float(schedule["finish_time"].max())
    total_cost = float(schedule["cost"].sum())
    average_task_time = float(schedule["execution_time"].mean())

    # Simplified estimate: each VM's average power is multiplied by the
    # overall makespan. This is a model-based estimate, not measured energy.
    energy_joules = 0.0
    for vm in vms:
        rows = schedule[schedule["vm_id"] == vm.vm_id]
        busy_time = float(rows["execution_time"].sum())
        if makespan > 0:
            vm_utilization = min(busy_time / makespan, 1.0)
            average_power = vm.idle_watts + (
                vm.max_watts - vm.idle_watts
            ) * vm_utilization
            energy_joules += average_power * makespan

    total_capacity = sum(vm.mips for vm in vms)
    busy_capacity_time = sum(
        float(schedule.loc[schedule["vm_id"] == vm.vm_id,
                           "execution_time"].sum()) * vm.mips
        for vm in vms
    )
    utilization = 100.0 * busy_capacity_time / (makespan * total_capacity) if makespan > 0 else 0.0

    return {
        "Makespan (seconds)": makespan,
        "Average task execution time (seconds)": average_task_time,
        "Execution cost": total_cost,
        "Estimated energy (joules)": energy_joules,
        "Aggregate CPU utilization (%)": utilization,
        "Total task count": int(len(schedule)),
    }


def main():
    import random
    results = {}
    for method in ["fcfs", "round_robin", "min_min"]:
        results[method.upper()] = calculate_metrics(schedule_tasks(TASKS, VMS, method), VMS)
    random.seed(42)
    results["LAAS"] = calculate_metrics(load_aware_scheduler(TASKS, VMS), VMS)
    random.seed(42)
    results["GENETIC ALGORITHM"] = calculate_metrics(genetic_algorithm(TASKS, VMS), VMS)
    random.seed(42)
    results["PARTICLE SWARM OPTIMIZATION"] = calculate_metrics(particle_swarm_optimization(TASKS, VMS), VMS)

    print("\n===== ALGORITHM COMPARISON =====")
    print(pd.DataFrame(results).T.round(3).to_string())

    print("\n===== GENETIC ALGORITHM TASK SCHEDULE =====")
    print(genetic_algorithm(TASKS, VMS).round(4).to_string(index=False))
    print("\n===== PSO TASK SCHEDULE =====")
    print(particle_swarm_optimization(TASKS, VMS).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
