import pandas as pd


def load_aware_scheduler(tasks, vms):
    """Load-aware scheduling using LPT initialization and local refinement.

    1. Sort tasks longest-first and assign each to the VM with the earliest
       projected completion time.
    2. Apply best-improvement single-task migrations while a migration reduces
       the overall makespan. Ties prefer lower incremental execution cost.

    This heuristic can improve a greedy starting assignment, but does not
    guarantee a globally optimal schedule.
    """
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")
    if any(vm.mips <= 0 for vm in vms):
        raise ValueError("VM processing speed (mips) must be positive.")

    ordered_tasks = sorted(tasks, key=lambda task: (-task.length_mi, task.task_id))
    vm_by_id = {vm.vm_id: vm for vm in vms}
    assignments = {}
    loads = {vm.vm_id: 0.0 for vm in vms}

    # LPT + earliest projected finish initialization.
    for task in ordered_tasks:
        vm = min(vms, key=lambda candidate: (
            loads[candidate.vm_id] + task.length_mi / candidate.mips,
            candidate.vm_id,
        ))
        assignments[task.task_id] = vm.vm_id
        loads[vm.vm_id] += task.length_mi / vm.mips

    # Best-improvement local search: move one task to another VM if it lowers
    # the maximum accumulated processing time. Each pass applies one best move.
    max_passes = max(1, min(5, len(tasks)))
    for _ in range(max_passes):
        current_makespan = max(loads.values())
        best_move = None
        best_key = (current_makespan, float("inf"), float("inf"))

        for task in ordered_tasks:
            source_id = assignments[task.task_id]
            source_vm = vm_by_id[source_id]
            source_duration = task.length_mi / source_vm.mips
            for target_vm in vms:
                if target_vm.vm_id == source_id:
                    continue
                target_duration = task.length_mi / target_vm.mips
                projected = dict(loads)
                projected[source_id] -= source_duration
                projected[target_vm.vm_id] += target_duration
                projected_makespan = max(projected.values())
                # Only accept strict makespan improvement; tie-break on the
                # incremental cost change and then target VM id.
                cost_delta = (target_duration * target_vm.cost_per_second
                              - source_duration * source_vm.cost_per_second)
                key = (projected_makespan, cost_delta, target_vm.vm_id)
                if projected_makespan < current_makespan - 1e-12 and key < best_key:
                    best_key = key
                    best_move = (task, source_id, target_vm.vm_id, projected)

        if best_move is None:
            break
        task, _, target_id, loads = best_move
        assignments[task.task_id] = target_id

    # Rebuild execution timeline per VM in deterministic LPT order.
    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []
    for task in ordered_tasks:
        vm = vm_by_id[assignments[task.task_id]]
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
