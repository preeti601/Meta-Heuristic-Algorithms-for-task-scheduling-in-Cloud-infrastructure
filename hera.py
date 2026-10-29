"""
HERA: Hybrid Energy-Aware Reassignment Algorithm

Features:
1. LPT initialization
2. Energy-aware VM assignment
3. Bottleneck-aware task reassignment
4. Pairwise task-swap local search
5. Randomized candidate exploration
6. Best-solution retention

No third-party dependencies.

Input:
    tasks = [
        {"id": 0, "length": 20},
        {"id": 1, "length": 10},
    ]

    vms = [
        {
            "id": 0,
            "speed": 2.0,
            "power": 100.0,
            "cost_rate": 0.05,
        },
        ...
    ]

Energy model:
    energy = power * execution_time

Cost model:
    cost = cost_rate * execution_time

These are simplified models, not physical measurements.
"""

import random
import time
from copy import deepcopy


def evaluate_schedule(tasks, vms, assignment, weights=None):
    """Calculate scheduling metrics and the weighted objective."""

    if weights is None:
        weights = {
            "makespan": 0.50,
            "energy": 0.30,
            "cost": 0.20,
        }

    vm_finish = {vm["id"]: 0.0 for vm in vms}
    vm_energy = {vm["id"]: 0.0 for vm in vms}
    vm_cost = {vm["id"]: 0.0 for vm in vms}
    vm_task_count = {vm["id"]: 0 for vm in vms}

    vm_lookup = {vm["id"]: vm for vm in vms}
    task_lookup = {task["id"]: task for task in tasks}

    for task_id, vm_id in assignment.items():
        task = task_lookup[task_id]
        vm = vm_lookup[vm_id]

        execution_time = task["length"] / vm["speed"]

        vm_finish[vm_id] += execution_time
        vm_energy[vm_id] += vm["power"] * execution_time
        vm_cost[vm_id] += vm["cost_rate"] * execution_time
        vm_task_count[vm_id] += 1

    makespan = max(vm_finish.values(), default=0.0)
    energy = sum(vm_energy.values())
    cost = sum(vm_cost.values())

    total_execution = sum(vm_finish.values())
    total_capacity = sum(vm["speed"] for vm in vms)

    utilization = (
        total_execution / (makespan * len(vms)) * 100.0
        if makespan > 0 and vms
        else 0.0
    )

    # Normalize objective components against the current problem.
    # These scales are fixed for a run so candidate scores are comparable.
    total_work = sum(task["length"] for task in tasks)
    max_speed = max((vm["speed"] for vm in vms), default=1.0)

    makespan_scale = max(total_work / max_speed, 1e-12)
    energy_scale = max(
        sum(
            vm["power"] * task["length"] / vm["speed"]
            for task in tasks
            for vm in vms
        ) / max(len(vms), 1),
        1e-12,
    )
    cost_scale = max(
        sum(
            vm["cost_rate"] * task["length"] / vm["speed"]
            for task in tasks
            for vm in vms
        ) / max(len(vms), 1),
        1e-12,
    )

    score = (
        weights["makespan"] * makespan / makespan_scale
        + weights["energy"] * energy / energy_scale
        + weights["cost"] * cost / cost_scale
    )

    return {
        "makespan": makespan,
        "energy": energy,
        "cost": cost,
        "cpu_utilization": utilization,
        "objective": score,
        "vm_finish_times": vm_finish,
        "vm_energy": vm_energy,
        "vm_cost": vm_cost,
        "vm_task_count": vm_task_count,
    }


def lpt_initialization(tasks, vms):
    """Assign longest tasks first to the VM with the earliest finish time."""

    assignment = {}
    finish = {vm["id"]: 0.0 for vm in vms}
    vm_lookup = {vm["id"]: vm for vm in vms}

    for task in sorted(
        tasks, key=lambda t: t["length"], reverse=True
    ):
        best_vm = min(
            vms,
            key=lambda vm: (
                finish[vm["id"]]
                + task["length"] / vm["speed"]
            ),
        )

        vm_id = best_vm["id"]
        assignment[task["id"]] = vm_id
        finish[vm_id] += task["length"] / vm_lookup[vm_id]["speed"]

    return assignment


def energy_aware_initialization(tasks, vms, weights=None):
    """Greedily choose a VM using projected completion, energy and cost."""

    if weights is None:
        weights = {
            "makespan": 0.50,
            "energy": 0.30,
            "cost": 0.20,
        }

    assignment = {}
    finish = {vm["id"]: 0.0 for vm in vms}

    max_speed = max(vm["speed"] for vm in vms)
    makespan_scale = max(
        sum(t["length"] for t in tasks) / max_speed, 1e-12
    )
    energy_scale = max(
        max(vm["power"] for vm in vms) * makespan_scale, 1e-12
    )
    cost_scale = max(
        max(vm["cost_rate"] for vm in vms) * makespan_scale, 1e-12
    )

    for task in sorted(
        tasks, key=lambda t: t["length"], reverse=True
    ):
        best_vm = None
        best_score = float("inf")

        for vm in vms:
            duration = task["length"] / vm["speed"]
            projected_finish = finish[vm["id"]] + duration

            energy = vm["power"] * duration
            cost = vm["cost_rate"] * duration

            score = (
                weights["makespan"] * projected_finish / makespan_scale
                + weights["energy"] * energy / energy_scale
                + weights["cost"] * cost / cost_scale
            )

            if score < best_score:
                best_score = score
                best_vm = vm

        assignment[task["id"]] = best_vm["id"]
        finish[best_vm["id"]] += (
            task["length"] / best_vm["speed"]
        )

    return assignment


def improve_by_reassignment(
    tasks, vms, assignment, weights=None, max_moves=100
):
    """Move tasks from the bottleneck VM when the objective improves."""

    current = deepcopy(assignment)
    current_metrics = evaluate_schedule(
        tasks, vms, current, weights
    )

    vm_lookup = {vm["id"]: vm for vm in vms}
    task_lookup = {task["id"]: task for task in tasks}

    for _ in range(max_moves):
        finish = current_metrics["vm_finish_times"]
        bottleneck_vm = max(finish, key=finish.get)

        bottleneck_tasks = [
            task for task in tasks
            if current[task["id"]] == bottleneck_vm
        ]

        # Try longer tasks first to seek useful load transfers.
        bottleneck_tasks.sort(
            key=lambda task: task["length"], reverse=True
        )

        best_assignment = None
        best_metrics = current_metrics

        for task in bottleneck_tasks:
            original_vm = current[task["id"]]

            for target_vm in vms:
                target_id = target_vm["id"]

                if target_id == original_vm:
                    continue

                candidate = deepcopy(current)
                candidate[task["id"]] = target_id

                metrics = evaluate_schedule(
                    tasks, vms, candidate, weights
                )

                if metrics["objective"] < best_metrics["objective"] - 1e-12:
                    best_assignment = candidate
                    best_metrics = metrics

        if best_assignment is None:
            break

        current = best_assignment
        current_metrics = best_metrics

    return current


def pairwise_swap_search(
    tasks, vms, assignment, weights=None, max_checks=500
):
    """Try swapping VM assignments for pairs of tasks."""

    current = deepcopy(assignment)
    current_metrics = evaluate_schedule(
        tasks, vms, current, weights
    )

    task_ids = [task["id"] for task in tasks]
    checks = 0

    while checks < max_checks:
        best_assignment = None
        best_metrics = current_metrics

        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                if checks >= max_checks:
                    break

                checks += 1
                task_a = task_ids[i]
                task_b = task_ids[j]

                vm_a = current[task_a]
                vm_b = current[task_b]

                if vm_a == vm_b:
                    continue

                candidate = deepcopy(current)
                candidate[task_a] = vm_b
                candidate[task_b] = vm_a

                metrics = evaluate_schedule(
                    tasks, vms, candidate, weights
                )

                if (
                    metrics["objective"]
                    < best_metrics["objective"] - 1e-12
                ):
                    best_assignment = candidate
                    best_metrics = metrics

            if checks >= max_checks:
                break

        if best_assignment is None:
            break

        current = best_assignment
        current_metrics = best_metrics

    return current


def hera_schedule(
    tasks,
    vms,
    weights=None,
    iterations=20,
    max_moves=100,
    max_swap_checks=500,
    seed=42,
):
    """
    Run HERA and return the best schedule and metrics.

    Each task must have a unique 'id' and positive 'length'.
    Each VM must have a unique 'id', positive 'speed',
    non-negative 'power', and non-negative 'cost_rate'.
    """

    start_time = time.perf_counter()

    if not tasks:
        raise ValueError("tasks must not be empty")

    if not vms:
        raise ValueError("vms must not be empty")

    task_ids = [task["id"] for task in tasks]
    vm_ids = [vm["id"] for vm in vms]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs must be unique")

    if len(vm_ids) != len(set(vm_ids)):
        raise ValueError("VM IDs must be unique")

    for task in tasks:
        if task["length"] <= 0:
            raise ValueError("Task lengths must be positive")

    for vm in vms:
        if vm["speed"] <= 0:
            raise ValueError("VM speeds must be positive")
        if vm["power"] < 0 or vm["cost_rate"] < 0:
            raise ValueError("Power and cost_rate cannot be negative")

    if weights is None:
        weights = {
            "makespan": 0.50,
            "energy": 0.30,
            "cost": 0.20,
        }

    rng = random.Random(seed)

    # Candidate 1: LPT
    candidates = [
        lpt_initialization(tasks, vms),
        energy_aware_initialization(tasks, vms, weights),
    ]

    best_assignment = min(
        candidates,
        key=lambda a: evaluate_schedule(
            tasks, vms, a, weights
        )["objective"],
    )

    best_metrics = evaluate_schedule(
        tasks, vms, best_assignment, weights
    )

    # Explore additional randomized task orders.
    for _ in range(max(0, iterations)):
        randomized_tasks = tasks[:]
        rng.shuffle(randomized_tasks)

        candidate = energy_aware_initialization(
            randomized_tasks, vms, weights
        )

        candidate = improve_by_reassignment(
            tasks, vms, candidate, weights, max_moves
        )

        candidate = pairwise_swap_search(
            tasks, vms, candidate, weights, max_swap_checks
        )

        metrics = evaluate_schedule(
            tasks, vms, candidate, weights
        )

        if metrics["objective"] < best_metrics["objective"]:
            best_assignment = candidate
            best_metrics = metrics

    # Refine the best candidate, then retain it only if it improves.
    refined = improve_by_reassignment(
        tasks, vms, best_assignment, weights, max_moves
    )
    refined = pairwise_swap_search(
        tasks, vms, refined, weights, max_swap_checks
    )

    refined_metrics = evaluate_schedule(
        tasks, vms, refined, weights
    )

    if refined_metrics["objective"] < best_metrics["objective"]:
        best_assignment = refined
        best_metrics = refined_metrics

    runtime = time.perf_counter() - start_time

    return {
        "algorithm": "HERA",
        "assignment": best_assignment,
        "makespan": best_metrics["makespan"],
        "estimated_energy": best_metrics["energy"],
        "cost": best_metrics["cost"],
        "cpu_utilization": best_metrics["cpu_utilization"],
        "objective": best_metrics["objective"],
        "runtime_seconds": runtime,
    }


if __name__ == "__main__":
    tasks = [
        {"id": 0, "length": 20},
        {"id": 1, "length": 10},
        {"id": 2, "length": 35},
        {"id": 3, "length": 15},
        {"id": 4, "length": 25},
        {"id": 5, "length": 12},
    ]

    vms = [
        {"id": 0, "speed": 1.0, "power": 80, "cost_rate": 0.04},
        {"id": 1, "speed": 2.0, "power": 120, "cost_rate": 0.07},
        {"id": 2, "speed": 1.5, "power": 95, "cost_rate": 0.05},
    ]

    result = hera_schedule(tasks, vms)

    print("\n===== HERA RESULTS =====")
    print("Algorithm:", result["algorithm"])
    print("Assignment:", result["assignment"])
    print("Makespan:", round(result["makespan"], 4))
    print("Estimated energy:", round(result["estimated_energy"], 4))
    print("Cost:", round(result["cost"], 4))
    print("CPU utilization:", round(result["cpu_utilization"], 2), "%")
    print("Objective:", round(result["objective"], 6))
    print("Runtime:", round(result["runtime_seconds"], 6), "seconds")