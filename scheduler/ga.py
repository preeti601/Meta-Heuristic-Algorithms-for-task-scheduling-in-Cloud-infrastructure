
import random
import pandas as pd


def genetic_algorithm(tasks, vms, population_size=30, generations=100):
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")

    task_count = len(tasks)
    vm_count = len(vms)

    def evaluate(chromosome):
        vm_times = [0.0] * vm_count

        for i, vm_index in enumerate(chromosome):
            vm_times[vm_index] += tasks[i].length_mi / vms[vm_index].mips

        return max(vm_times)

    population = [
        [random.randrange(vm_count) for _ in range(task_count)]
        for _ in range(population_size)
    ]

    for _ in range(generations):
        population.sort(key=evaluate)
        next_population = population[:2]

        while len(next_population) < population_size:
            parent1, parent2 = random.choices(
                population[:max(2, population_size // 2)], k=2
            )

            split = random.randrange(1, task_count) if task_count > 1 else 0
            child = parent1[:split] + parent2[split:]

            if random.random() < 0.1:
                gene = random.randrange(task_count)
                child[gene] = random.randrange(vm_count)

            next_population.append(child)

        population = next_population

    best = min(population, key=evaluate)
    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []

    for task, vm_index in zip(tasks, best):
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
