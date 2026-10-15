"""Reproducible dynamic workload generator for scheduling research."""
from dataclasses import dataclass
import random
from typing import List

@dataclass(frozen=True)
class DynamicTask:
    task_id: int
    length_mi: float
    arrival_time_s: float
    deadline_s: float
    tenant_id: str
    memory_mb: int
    data_mb: float
    io_intensity: float
    priority: int = 1

def generate_dynamic_tasks(count: int, seed: int = 2026, horizon_s: float = 3600.0,
                           tenant_count: int = 5) -> List[DynamicTask]:
    """Generate deterministic, burst-capable arrivals; synthetic, not a real trace."""
    if count < 1 or horizon_s <= 0 or tenant_count < 1:
        raise ValueError("count, horizon_s and tenant_count must be positive")
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        # Mix uniform arrivals with clustered bursts to create changing load.
        if rng.random() < 0.35:
            center = rng.choice([0.2, 0.5, 0.8]) * horizon_s
            arrival = max(0.0, min(horizon_s, rng.gauss(center, horizon_s * 0.025)))
        else:
            arrival = rng.uniform(0, horizon_s)
        length = rng.uniform(1_000, 50_000)
        # Deadline relative to arrival, based on an optimistic execution estimate.
        deadline = arrival + rng.uniform(1.5, 3.0) * (length / 3000.0)
        tasks.append(DynamicTask(
            task_id=i + 1, length_mi=length, arrival_time_s=arrival,
            deadline_s=deadline, tenant_id=f"tenant_{rng.randint(1, tenant_count)}",
            memory_mb=rng.choice([128, 256, 512, 1024, 2048, 4096]),
            data_mb=round(rng.uniform(10, 5000), 2),
            io_intensity=round(rng.uniform(0.1, 10.0), 3),
            priority=rng.randint(1, 5)
        ))
    return sorted(tasks, key=lambda t: (t.arrival_time_s, t.task_id))
