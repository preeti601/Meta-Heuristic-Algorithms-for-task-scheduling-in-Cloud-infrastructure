"""Efficient EcoTaskOpt-inspired hybrid ACO/PSO task scheduler.

This is an educational adaptation, not a reproduction of the published
CloudSim EcoTaskOpt system. It uses an LPT incumbent, bounded ACO-style
assignment sampling, PSO-inspired perturbations, and bounded local search.

The returned schedule and objective use the same model: each VM executes its
assigned tasks sequentially, with all tasks available at time zero.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np
import pandas as pd


def _build_evaluator(tasks, vms, weights=(0.50, 0.30, 0.20)):
    lengths = np.asarray([float(t.length_mi) for t in tasks], dtype=float)
    speeds = np.asarray([float(v.mips) for v in vms], dtype=float)
    idle = np.asarray([float(v.idle_watts) for v in vms], dtype=float)
    maximum = np.asarray([float(v.max_watts) for v in vms], dtype=float)
    costs = np.asarray([float(v.cost_per_second) for v in vms], dtype=float)
    n, k = len(lengths), len(speeds)
    total_work = float(lengths.sum())
    makespan_scale = max(total_work / float(speeds.sum()), 1e-12)
    energy_scale = max(float(maximum.sum()) * makespan_scale, 1e-12)
    max_cost = max(float(costs.max(initial=0.0)), 1e-12)
    cost_scale = max(total_work / float(speeds.min()) * max_cost, 1e-12)

    @lru_cache(maxsize=20000)
    def evaluate(key):
        assignment = np.fromiter(key, dtype=np.int32, count=n)
        durations = lengths / speeds[assignment]
        loads = np.bincount(assignment, weights=durations, minlength=k)
        makespan = float(loads.max(initial=0.0))
        util = loads / makespan if makespan > 0 else np.zeros(k)
        power = idle + (maximum - idle) * np.power(np.clip(util, 0.0, 1.0), 1.4)
        energy = float(np.sum(power) * makespan)
        imbalance = float(np.std(util))
        cost = float(np.sum(durations * costs[assignment]))
        score = (
            weights[0] * energy / energy_scale
            + weights[1] * makespan / makespan_scale
            + weights[2] * imbalance
        )
        return float(score), makespan, energy, cost, imbalance, loads

    return evaluate


def eco_task_opt(
    tasks, vms, ants=8, particles=20, iterations=20, seed=None,
    patience=5, min_iterations=5
):
    """Return a DataFrame schedule compatible with the project's runner.

    `ants`, `particles`, and `iterations` control bounded search effort. A
    strong LPT schedule is always retained, so the result cannot be worse than
    that initial schedule under this function's own fitness score.
    """
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")
    if any(float(vm.mips) <= 0 for vm in vms):
        raise ValueError("VM processing speed (mips) must be positive.")
    if ants < 1 or particles < 2 or iterations < 1 or patience < 1:
        raise ValueError("ants >= 1, particles >= 2, iterations >= 1, patience >= 1 required")

    rng = random.Random(seed)
    n, k = len(tasks), len(vms)
    lengths = np.asarray([float(t.length_mi) for t in tasks], dtype=float)
    speeds = np.asarray([float(v.mips) for v in vms], dtype=float)
    evaluate = _build_evaluator(tasks, vms)

    # LPT / earliest-finish initialization: O(n log n + n*k).
    order = sorted(range(n), key=lambda i: (-lengths[i], getattr(tasks[i], "task_id", i)))
    lpt = [0] * n
    loads = np.zeros(k, dtype=float)
    for ti in order:
        durations = lengths[ti] / speeds
        vi = int(np.argmin(loads + durations))
        lpt[ti] = vi
        loads[vi] += durations[vi]

    best = tuple(lpt)
    best_score = evaluate(best)[0]
    tau = np.ones((n, k), dtype=float)
    eta = (speeds / speeds.max())[None, :] / (
        0.75 + 0.25 * (np.asarray([float(v.max_watts) for v in vms]) /
                       max(max(float(v.max_watts) for v in vms), 1e-12))
    )[None, :]
    eta = np.repeat(eta, n, axis=0)
    no_improve = 0
    # Limit actual work for large inputs; iteration count scales down with n.
    effective_iterations = min(int(iterations), max(3, int(12000 / max(n, 1))))
    effective_ants = min(int(ants), 4 if n >= 500 else 8)
    effective_particles = min(int(particles), 6 if n >= 500 else 12)

    for it in range(effective_iterations):
        old_best = best_score
        tau_power = np.power(np.clip(tau, 0.05, 10.0), 1.0)
        candidates = []

        # ACO-style construction in LPT order. Sample from a small heuristic
        # distribution using projected per-VM load.
        for _ in range(effective_ants):
            assignment = [0] * n
            current_loads = np.zeros(k, dtype=float)
            for ti in order:
                projected = current_loads + lengths[ti] / speeds
                desirability = tau_power[ti] * eta[0] / (
                    1.0 + projected / max(float(lengths.sum() / speeds.sum()), 1e-12)
                )
                desirability = np.maximum(desirability, 1e-15)
                cumulative = np.cumsum(desirability)
                pick = rng.random() * float(cumulative[-1])
                vi = min(k - 1, int(np.searchsorted(cumulative, pick, side="left")))
                assignment[ti] = vi
                current_loads[vi] += lengths[ti] / speeds[vi]
            candidates.append(tuple(assignment))

        # PSO-inspired discrete perturbations of the incumbent and LPT seed.
        for pi in range(effective_particles):
            base = best if pi % 2 == 0 else tuple(lpt)
            candidate = list(base)
            changes = max(1, min(n, int(math.sqrt(n)) + it % 3))
            for ti in rng.sample(range(n), changes):
                if rng.random() < 0.65:
                    # Bias toward a faster VM, with occasional random exploration.
                    if rng.random() < 0.8:
                        candidate[ti] = int(np.argmax(speeds))
                    else:
                        candidate[ti] = rng.randrange(k)
                else:
                    candidate[ti] = rng.randrange(k)
            candidates.append(tuple(candidate))

        scored = sorted(((evaluate(c)[0], c) for c in candidates), key=lambda x: x[0])
        for score, candidate in scored:
            if score < best_score:
                best, best_score = candidate, score

        # Small, bounded coordinate-descent pass on a subset of tasks.
        # This is usually the most useful refinement and avoids exhaustive O(n*k)
        # full reevaluations for every possible task/VM move.
        if k > 1 and n:
            current = list(best)
            indices = order[:min(n, max(20, int(3000 / max(n, 1))))]
            changed = False
            for ti in indices:
                original_vm = current[ti]
                local_best_vm = original_vm
                local_best_score = best_score
                for vi in range(k):
                    if vi == original_vm:
                        continue
                    trial = current.copy()
                    trial[ti] = vi
                    score = evaluate(tuple(trial))[0]
                    if score < local_best_score:
                        local_best_score, local_best_vm = score, vi
                if local_best_vm != original_vm:
                    current[ti] = local_best_vm
                    best_score = local_best_score
                    changed = True
            if changed:
                best = tuple(current)
                best_score = evaluate(best)[0]

        tau *= 0.88
        for score, candidate in scored[:max(1, effective_ants // 2)]:
            deposit = min(1.0 / max(score, 1e-9), 2.0)
            for ti, vi in enumerate(candidate):
                tau[ti, vi] += deposit
        for ti, vi in enumerate(best):
            tau[ti, vi] += 0.2
        np.clip(tau, 0.05, 10.0, out=tau)

        if best_score < old_best - 1e-12:
            no_improve = 0
        else:
            no_improve += 1
        if it + 1 >= min_iterations and no_improve >= patience:
            break

    # Generate the exact sequential per-VM timeline used by the evaluator.
    availability = {vm.vm_id: 0.0 for vm in vms}
    records = []
    for task, vm_idx in zip(tasks, best):
        vm = vms[vm_idx]
        start = availability[vm.vm_id]
        duration = float(task.length_mi) / float(vm.mips)
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
