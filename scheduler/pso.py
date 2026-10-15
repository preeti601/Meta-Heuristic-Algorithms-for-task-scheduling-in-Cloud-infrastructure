
import random
import pandas as pd


def particle_swarm_optimization(
    tasks, vms, population_size=30, iterations=100
):
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")

    task_count = len(tasks)
    vm_count = len(vms)

    # Each particle represents a VM assignment for every task.
    def evaluate(position):
        vm_times = [0.0] * vm_count

        for i, vm_index in enumerate(position):
            vm_times[vm_index] += (
                tasks[i].length_mi / vms[vm_index].mips
            )

        return max(vm_times)

    particles = [
        [random.randrange(vm_count) for _ in range(task_count)]
        for _ in range(population_size)
    ]

    velocities = [
        [0.0] * task_count for _ in range(population_size)
    ]

    personal_best = [p[:] for p in particles]
    personal_scores = [evaluate(p) for p in particles]

    best_index = min(
        range(population_size),
        key=lambda i: personal_scores[i]
    )
    global_best = personal_best[best_index][:]
    global_score = personal_scores[best_index]

    for _ in range(iterations):
        for i, particle in enumerate(particles):
            for j in range(task_count):
                r1 = random.random()
                r2 = random.random()

                velocities[i][j] = (
                    0.5 * velocities[i][j]
                    + 1.5 * r1
                    * (personal_best[i][j] - particle[j])
                    + 1.5 * r2
                    * (global_best[j] - particle[j])
                )

                if random.random() < min(
                    0.5, abs(velocities[i][j]) / max(vm_count, 1)
                ):
                    particle[j] = random.randrange(vm_count)

            score = evaluate(particle)

            if score < personal_scores[i]:
                personal_best[i] = particle[:]
                personal_scores[i] = score

            if score < global_score:
                global_best = particle[:]
                global_score = score

    # Convert the best assignment to the standard schedule format.
    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []

    for task, vm_index in zip(tasks, global_best):
        vm = vms[vm_index]
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
