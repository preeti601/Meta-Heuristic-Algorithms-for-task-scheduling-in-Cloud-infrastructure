from dataclasses import dataclass
from typing import List
import random
import pandas as pd

from scheduler.laas import load_aware_scheduler
from scheduler.pso import particle_swarm_optimization
from scheduler.ga import genetic_algorithm
from scheduler.ecotaskopt import eco_task_opt
from hera import hera_schedule


# ============================================================
# DATA CLASSES
# ============================================================

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


# ============================================================
# EXPERIMENT DATA
# ============================================================

TASKS = [
    Task(1, 12000),
    Task(2, 8000),
    Task(3, 20000),
    Task(4, 6000),
    Task(5, 15000),
    Task(6, 10000),
    Task(7, 18000),
    Task(8, 5000),
]

VMS = [
    VM(1, 2000, 0.05, 70, 140),
    VM(2, 3000, 0.08, 90, 180),
    VM(3, 1500, 0.03, 50, 100),
]


# ============================================================
# TRADITIONAL SCHEDULING ALGORITHMS
# ============================================================

def schedule_tasks(
    tasks: List[Task],
    vms: List[VM],
    method: str
):
    """Run FCFS, Round Robin, or MIN-MIN scheduling."""

    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")

    if any(vm.mips <= 0 for vm in vms):
        raise ValueError("VM processing speed must be positive.")

    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []
    remaining = list(tasks)
    rr_index = 0

    while remaining:

        if method == "min_min":
            choices = []

            for task in remaining:
                for vm in vms:
                    duration = task.length_mi / vm.mips
                    finish = availability[vm.vm_id] + duration

                    choices.append(
                        (finish, task.task_id, vm.vm_id, task, vm)
                    )

            _, _, _, task, vm = min(
                choices, key=lambda item: item[:3]
            )

            remaining.remove(task)

        else:
            task = remaining.pop(0)

            if method == "round_robin":
                vm = vms[rr_index % len(vms)]
                rr_index += 1

            elif method == "fcfs":
                vm = min(
                    vms,
                    key=lambda item: (
                        availability[item.vm_id],
                        item.vm_id
                    )
                )

            else:
                raise ValueError(
                    f"Unknown scheduling method: {method}"
                )

        start = availability[vm.vm_id]
        duration = task.length_mi / vm.mips
        finish = start + duration

        availability[vm.vm_id] = finish

        records.append({
            "task_id": task.task_id,
            "vm_id": vm.vm_id,
            "start_time": start,
            "execution_time": duration,
            "finish_time": finish,
            "cost": duration * vm.cost_per_second,
        })

    return pd.DataFrame(records)


# ============================================================
# COMMON METRIC CALCULATION
# ============================================================

def calculate_metrics(
    schedule: pd.DataFrame,
    vms: List[VM]
):
    """
    Calculate common reporting metrics.

    Energy is a simplified model-based estimate, not a
    measurement of physical server energy consumption.
    """

    if schedule.empty:
        raise ValueError(
            "Cannot calculate metrics for an empty schedule."
        )

    makespan = float(schedule["finish_time"].max())
    total_cost = float(schedule["cost"].sum())

    average_task_time = float(
        schedule["execution_time"].mean()
    )

    # Simplified energy model using idle and maximum VM power.
    energy_joules = 0.0

    for vm in vms:
        rows = schedule[
            schedule["vm_id"] == vm.vm_id
        ]

        busy_time = float(
            rows["execution_time"].sum()
        )

        if makespan > 0:
            vm_utilization = min(
                busy_time / makespan, 1.0
            )

            average_power = vm.idle_watts + (
                vm.max_watts - vm.idle_watts
            ) * vm_utilization

            energy_joules += average_power * makespan

    total_capacity = sum(
        vm.mips for vm in vms
    )

    busy_capacity_time = sum(
        float(
            schedule.loc[
                schedule["vm_id"] == vm.vm_id,
                "execution_time"
            ].sum()
        ) * vm.mips
        for vm in vms
    )

    utilization = (
        100.0 * busy_capacity_time
        / (makespan * total_capacity)
        if makespan > 0 and total_capacity > 0
        else 0.0
    )

    return {
        "Makespan (seconds)": makespan,
        "Average task execution time (seconds)": average_task_time,
        "Execution cost": total_cost,
        "Estimated energy (joules)": energy_joules,
        "Aggregate CPU utilization (%)": utilization,
        "Total task count": int(len(schedule)),
    }


# ============================================================
# HERA ADAPTER
# ============================================================

def run_hera(
    tasks: List[Task],
    vms: List[VM]
):
    """
    Convert Task/VM dataclasses to HERA dictionaries,
    execute HERA, and convert its assignment to a DataFrame.

    Returns:
        schedule: DataFrame in the common format.
        result: HERA's internal metrics, objective, and runtime.
    """

    hera_tasks = [
        {
            "id": task.task_id,
            "length": task.length_mi
        }
        for task in tasks
    ]

    hera_vms = [
        {
            "id": vm.vm_id,
            "speed": vm.mips,
            "power": vm.max_watts,
            "cost_rate": vm.cost_per_second
        }
        for vm in vms
    ]

    result = hera_schedule(
        hera_tasks,
        hera_vms,
        iterations=20,
        seed=42
    )

    vm_lookup = {
        vm.vm_id: vm for vm in vms
    }

    task_lookup = {
        task.task_id: task for task in tasks
    }

    availability = {
        vm.vm_id: 0.0 for vm in vms
    }

    records = []

    # HERA assigns each task to a VM.
    # Task start/finish times are reconstructed here.
    for task_id, vm_id in result["assignment"].items():

        task = task_lookup[task_id]
        vm = vm_lookup[vm_id]

        start = availability[vm_id]
        duration = task.length_mi / vm.mips
        finish = start + duration

        availability[vm_id] = finish

        records.append({
            "task_id": task_id,
            "vm_id": vm_id,
            "start_time": start,
            "execution_time": duration,
            "finish_time": finish,
            "cost": duration * vm.cost_per_second
        })

    schedule = pd.DataFrame(records)

    schedule = schedule.sort_values(
        "task_id"
    ).reset_index(drop=True)

    return schedule, result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    results = {}
    schedules = {}

    # --------------------------------------------------------
    # 1. FCFS, ROUND ROBIN, MIN-MIN
    # --------------------------------------------------------

    for method in [
        "fcfs",
        "round_robin",
        "min_min"
    ]:

        name = method.upper()

        schedules[name] = schedule_tasks(
            TASKS, VMS, method
        )

        results[name] = calculate_metrics(
            schedules[name], VMS
        )

    # --------------------------------------------------------
    # 2. LAAS
    # --------------------------------------------------------

    random.seed(42)

    schedules["LAAS"] = load_aware_scheduler(
        TASKS, VMS
    )

    results["LAAS"] = calculate_metrics(
        schedules["LAAS"], VMS
    )

    # --------------------------------------------------------
    # 3. GENETIC ALGORITHM
    # --------------------------------------------------------

    random.seed(42)

    schedules["GENETIC ALGORITHM"] = genetic_algorithm(
        TASKS, VMS
    )

    results["GENETIC ALGORITHM"] = calculate_metrics(
        schedules["GENETIC ALGORITHM"], VMS
    )

    # --------------------------------------------------------
    # 4. PARTICLE SWARM OPTIMIZATION
    # --------------------------------------------------------

    random.seed(42)

    schedules["PARTICLE SWARM OPTIMIZATION"] = (
        particle_swarm_optimization(TASKS, VMS)
    )

    results["PARTICLE SWARM OPTIMIZATION"] = (
        calculate_metrics(
            schedules["PARTICLE SWARM OPTIMIZATION"],
            VMS
        )
    )

    # --------------------------------------------------------
    # 5. ECOTASKOPT
    # --------------------------------------------------------

    schedules["ECOTASKOPT"] = eco_task_opt(
        TASKS, VMS, seed=42
    )

    results["ECOTASKOPT"] = calculate_metrics(
        schedules["ECOTASKOPT"], VMS
    )

    # --------------------------------------------------------
    # 6. HERA
    # --------------------------------------------------------

    schedules["HERA"], hera_result = run_hera(
        TASKS, VMS
    )

    results["HERA"] = calculate_metrics(
        schedules["HERA"], VMS
    )

    # --------------------------------------------------------
    # 7. ALGORITHM COMPARISON
    # --------------------------------------------------------

    print("\n===== ALGORITHM COMPARISON =====")

    comparison = pd.DataFrame(results).T

    print(comparison.round(3).to_string())

    # --------------------------------------------------------
    # 8. TASK SCHEDULES
    # --------------------------------------------------------

    schedules_to_print = [
        "GENETIC ALGORITHM",
        "PARTICLE SWARM OPTIMIZATION",
        "ECOTASKOPT",
        "HERA"
    ]

    for name in schedules_to_print:

        print(f"\n===== {name} TASK SCHEDULE =====")

        print(
            schedules[name]
            .round(4)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 9. HERA INTERNAL RESULTS
    # --------------------------------------------------------

    print("\n===== HERA INTERNAL RESULTS =====")

    print(
        "Objective:",
        round(hera_result["objective"], 6)
    )

    print(
        "Runtime (seconds):",
        round(hera_result["runtime_seconds"], 6)
    )


if __name__ == "__main__":
    main()
