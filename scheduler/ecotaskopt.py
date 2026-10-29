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


def _build_evaluator(tasks, vms, weights=(0.55, 0.30, 0.10, 0.05)):
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
            weights[0] * makespan / makespan_scale
            + weights[1] * energy / energy_scale
            + weights[2] * imbalance
            + weights[3] * cost / cost_scale
        )
        return float(score), makespan, energy, cost, imbalance, loads

    return evaluate


def eco_task_opt(
    tasks, vms, ants=8, particles=20, iterations=20, seed=None,
    patience=2, min_iterations=2, objective_weights=(0.55, 0.30, 0.05),
    makespan_slack=0.0
):
    """Return a DataFrame schedule compatible with the project's runner.

    `ants`, `particles`, and `iterations` control bounded search effort. A
    strong LPT schedule is retained as a fallback. By default, the returned
    candidate cannot exceed the LPT makespan; `makespan_slack` optionally allows
    a bounded makespan trade-off while minimizing the combined fitness.
    """
    if not tasks or not vms:
        raise ValueError("Tasks and VMs must not be empty.")
    if any(float(vm.mips) <= 0 for vm in vms):
        raise ValueError("VM processing speed (mips) must be positive.")
    if ants < 1 or particles < 2 or iterations < 1 or patience < 1:
        raise ValueError("ants >= 1, particles >= 2, iterations >= 1, patience >= 1 required")

    if len(objective_weights) != 3 or any(float(w) < 0 for w in objective_weights):
        raise ValueError("objective_weights must contain non-negative makespan, energy, and cost weights")
    weight_total = sum(float(w) for w in objective_weights)
    if weight_total <= 0:
        raise ValueError("At least one objective weight must be positive")
    if float(makespan_slack) < 0:
        raise ValueError("makespan_slack must be non-negative")
    # Normalize the three user-facing objective weights. Load imbalance remains
    # a small fixed tie-breaker rather than silently replacing any objective.
    w_makespan, w_energy, w_cost = [float(w) / weight_total for w in objective_weights]
    imbalance_weight = 0.05
    objective_scale = 1.0 - imbalance_weight
    w_makespan *= objective_scale
    w_energy *= objective_scale
    w_cost *= objective_scale

    rng = random.Random(seed)
    n, k = len(tasks), len(vms)
    lengths = np.asarray([float(t.length_mi) for t in tasks], dtype=float)
    speeds = np.asarray([float(v.mips) for v in vms], dtype=float)
    evaluate = _build_evaluator(tasks, vms, weights=(w_makespan, w_energy, imbalance_weight, w_cost))
    idle_watts = np.asarray([float(v.idle_watts) for v in vms], dtype=float)
    max_watts = np.asarray([float(v.max_watts) for v in vms], dtype=float)
    vm_costs = np.asarray([float(v.cost_per_second) for v in vms], dtype=float)
    total_work = float(lengths.sum())
    makespan_scale = max(total_work / float(speeds.sum()), 1e-12)
    energy_scale = max(float(max_watts.sum()) * makespan_scale, 1e-12)
    cost_scale = max(total_work / float(speeds.min()) * max(float(vm_costs.max(initial=0.0)), 1e-12), 1e-12)

    def score_from_loads(load_vector, total_cost):
        """Score a schedule from VM loads in O(number of VMs), not O(tasks)."""
        span = float(np.max(load_vector))
        utilization = load_vector / span if span > 0 else np.zeros(k, dtype=float)
        power = idle_watts + (max_watts - idle_watts) * np.power(np.clip(utilization, 0.0, 1.0), 1.4)
        energy_value = float(np.sum(power) * span)
        imbalance_value = float(np.std(utilization))
        return (w_makespan * span / makespan_scale
                + w_energy * energy_value / energy_scale
                + imbalance_weight * imbalance_value
                + w_cost * total_cost / cost_scale)

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

    # Keep a separate best candidate within the configured makespan ceiling.
    # A zero slack preserves the strict LPT makespan guard; positive slack allows
    # bounded trade-offs for energy/cost objectives.
    lpt_makespan_reference = float(evaluate(tuple(lpt))[1])
    makespan_ceiling = lpt_makespan_reference * (1.0 + float(makespan_slack))
    safe_best = tuple(lpt)
    safe_best_score = best_score

    # LPT-seeded constrained local search. Explore targeted single-task moves
    # and cross-VM swaps directly from the strong baseline before stochastic
    # ACO/PSO search. Accept only candidates that preserve the LPT makespan
    # ceiling and improve the same combined objective used for final selection.
    # The candidate budget is bounded so large workloads remain practical.
    local_search_checks = 0
    local_search_improvements = 0
    if k > 1 and n > 1:
        seed_assignment = list(lpt)
        length_cv = float(np.std(lengths) / max(np.mean(lengths), 1e-12))
        heavy_tail = length_cv >= 0.85 or float(np.max(lengths) / max(np.median(lengths), 1e-12)) >= 4.0
        max_rounds = (3 if n < 500 else 2) + (1 if heavy_tail and n < 500 else 0)
        task_budget = min(n, (64 if n < 500 else 32) if heavy_tail else (24 if n >= 500 else 48))
        swap_budget = min(220 if heavy_tail else 120, task_budget * max(1, k - 1))
        for _round in range(max_rounds):
            base_tuple = tuple(seed_assignment)
            base_eval = evaluate(base_tuple)
            base_score, base_span = base_eval[0], base_eval[1]
            base_loads = np.asarray(base_eval[5], dtype=float)
            bottleneck = int(np.argmax(base_loads))
            bottleneck_tasks = [i for i, vm_idx in enumerate(seed_assignment)
                                if vm_idx == bottleneck]
            bottleneck_tasks.sort(
                key=lambda i: lengths[i] / speeds[bottleneck], reverse=True
            )
            candidate_tasks = list(bottleneck_tasks[:task_budget])
            # Heavy-tail neighborhood: combine bottleneck jobs, globally longest
            # jobs, and jobs on the least-loaded VMs. This targets the long-job
            # placement failures common in skewed workloads.
            if heavy_tail:
                global_long = sorted(range(n), key=lambda i: lengths[i], reverse=True)
                light_vms = set(np.argsort(base_loads)[:max(1, min(k - 1, 3))].tolist())
                light_vm_tasks = [i for i, assigned in enumerate(seed_assignment) if assigned in light_vms]
                targeted = global_long[:min(task_budget // 2 + 1, n)] + light_vm_tasks[:task_budget // 3 + 1]
                seen_tasks = set(candidate_tasks)
                for ti in targeted:
                    if ti not in seen_tasks and len(candidate_tasks) < task_budget:
                        candidate_tasks.append(ti)
                        seen_tasks.add(ti)
            # Fill remaining slots deterministically; never repeatedly scan the
            # same task in a round.
            if len(candidate_tasks) < task_budget:
                seen_tasks = set(candidate_tasks)
                candidate_tasks.extend(i for i in range(n) if i not in seen_tasks)  # sliced below
                candidate_tasks = candidate_tasks[:task_budget]

            round_best = tuple(seed_assignment)
            round_best_score = base_score
            seen_assignments = {tuple(seed_assignment)}
            # Evaluate only promising destination VMs for heavy-tail jobs: fastest
            # VM, cheapest VM, lowest-power VM, and currently least-loaded VMs.
            for ti in candidate_tasks:
                old_vm = seed_assignment[ti]
                if heavy_tail:
                    destination_order = [int(np.argmax(speeds)), int(np.argmin(vm_costs)),
                                         int(np.argmin(max_watts))]
                    destination_order.extend(int(x) for x in np.argsort(base_loads)[:min(k, 3)])
                    destinations = list(dict.fromkeys(v for v in destination_order if v != old_vm))
                else:
                    destinations = [v for v in range(k) if v != old_vm]
                for new_vm in destinations:
                    trial = list(seed_assignment)
                    trial[ti] = new_vm
                    trial_tuple = tuple(trial)
                    if trial_tuple in seen_assignments:
                        continue
                    seen_assignments.add(trial_tuple)
                    trial_eval = evaluate(trial_tuple)
                    local_search_checks += 1
                    if (trial_eval[1] <= makespan_ceiling + 1e-9
                            and trial_eval[0] < round_best_score - 1e-12):
                        round_best, round_best_score = trial_tuple, trial_eval[0]

            # Targeted swaps: heavy-tail tasks are paired with short tasks on
            # lighter VMs first, then the remaining candidate pool. Duplicate
            # assignment tuples are skipped to avoid wasted objective calls.
            tried_swaps = 0
            swap_tasks = list(candidate_tasks)
            if heavy_tail:
                swap_tasks.sort(key=lambda i: lengths[i], reverse=True)
            for ti in swap_tasks:
                vm_i = seed_assignment[ti]
                if heavy_tail:
                    partners = [j for j in range(n) if j != ti and seed_assignment[j] != vm_i]
                    partners.sort(key=lambda j: (lengths[j], base_loads[seed_assignment[j]]))
                    partner_iter = partners[:max(8, min(32, swap_budget))]
                else:
                    partner_iter = range(n)
                for tj in partner_iter:
                    if seed_assignment[tj] == vm_i or ti == tj:
                        continue
                    trial = list(seed_assignment)
                    trial[ti], trial[tj] = trial[tj], trial[ti]
                    trial_tuple = tuple(trial)
                    if trial_tuple in seen_assignments:
                        continue
                    seen_assignments.add(trial_tuple)
                    trial_eval = evaluate(trial_tuple)
                    local_search_checks += 1
                    tried_swaps += 1
                    if (trial_eval[1] <= makespan_ceiling + 1e-9
                            and trial_eval[0] < round_best_score - 1e-12):
                        round_best, round_best_score = trial_tuple, trial_eval[0]
                    if tried_swaps >= swap_budget:
                        break
                if tried_swaps >= swap_budget:
                    break

            if round_best_score < safe_best_score - 1e-12:
                safe_best, safe_best_score = round_best, round_best_score
                local_search_improvements += 1
            if round_best == tuple(seed_assignment):
                break
            seed_assignment = list(round_best)

    tau = np.ones((n, k), dtype=float)
    eta = (speeds / speeds.max())[None, :] / (
        0.75 + 0.25 * (np.asarray([float(v.max_watts) for v in vms]) /
                       max(max(float(v.max_watts) for v in vms), 1e-12))
    )[None, :]
    eta = np.repeat(eta, n, axis=0)
    no_improve = 0
    duplicate_candidate_count = 0
    # Limit actual work for large inputs; iteration count scales down with n.
    effective_iterations = min(int(iterations), max(3, int(12000 / max(n, 1))))
    effective_ants = min(int(ants), 2 if n >= 500 else 4)
    effective_particles = min(int(particles), 3 if n >= 500 else 6)

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

        # Diverse discrete PSO-inspired candidates. Use multiple mutation
        # strengths and several different base schedules instead of repeatedly
        # making tiny edits to the same LPT incumbent.
        mutation_rates = (0.02, 0.08, 0.20, 0.40)
        for pi in range(effective_particles):
            mode = pi % 4
            if mode == 0:
                base = list(best)
            elif mode == 1:
                base = list(lpt)
            elif mode == 2:
                # Randomized greedy construction: shuffle task order, then put
                # each task on the VM with the earliest projected finish.
                randomized_order = list(range(n))
                rng.shuffle(randomized_order)
                base = [0] * n
                base_loads = np.zeros(k, dtype=float)
                for ti in randomized_order:
                    vi = int(np.argmin(base_loads + lengths[ti] / speeds))
                    base[ti] = vi
                    base_loads[vi] += lengths[ti] / speeds[vi]
            else:
                base = [rng.randrange(k) for _ in range(n)]
            candidate = list(base)
            rate = mutation_rates[(it + pi) % len(mutation_rates)]
            changes = max(1, min(n, int(math.ceil(n * rate))))
            for ti in rng.sample(range(n), changes):
                # Explore fastest, cheapest, lowest-power and random choices.
                choice = (it + pi + ti) % 4
                if choice == 0:
                    candidate[ti] = int(np.argmax(speeds))
                elif choice == 1:
                    candidate[ti] = int(np.argmin(vm_costs))
                elif choice == 2:
                    candidate[ti] = int(np.argmin(max_watts))
                else:
                    candidate[ti] = rng.randrange(k)
            candidates.append(tuple(candidate))

        # Candidate generation can produce the same assignment from multiple
        # ants/particles. Score each distinct assignment only once per iteration.
        unique_candidates = list(dict.fromkeys(tuple(c) for c in candidates))
        duplicate_candidate_count += len(candidates) - len(unique_candidates)
        scored = sorted(((evaluate(c)[0], c) for c in unique_candidates), key=lambda x: x[0])
        for score, candidate in scored:
            if score < best_score:
                best, best_score = candidate, score
            candidate_makespan = evaluate(candidate)[1]
            if (candidate_makespan <= makespan_ceiling + 1e-9
                    and score < safe_best_score - 1e-12):
                safe_best, safe_best_score = candidate, score

        # Incremental bottleneck local search. Candidate moves update only the
        # affected VM loads and cost; no full n-task fitness recomputation per move.
        if k > 1 and n:
            current = list(best)
            current_loads = np.bincount(
                np.asarray(current, dtype=np.int32),
                weights=lengths / speeds[np.asarray(current, dtype=np.int32)],
                minlength=k,
            ).astype(float)
            current_cost = float(np.sum((lengths / speeds[np.asarray(current, dtype=np.int32)])
                                        * vm_costs[np.asarray(current, dtype=np.int32)]))
            current_score = score_from_loads(current_loads, current_cost)
            bottleneck_vm = int(np.argmax(current_loads))
            bottleneck_tasks = [ti for ti, assigned_vm in enumerate(current) if assigned_vm == bottleneck_vm]
            bottleneck_tasks.sort(key=lambda ti: lengths[ti] / speeds[bottleneck_vm], reverse=True)
            inspect_count = min(len(bottleneck_tasks), max(4, min(8, int(400 / max(n, 1)))))
            search_tasks = list(bottleneck_tasks[:inspect_count])
            length_cv_iter = float(np.std(lengths) / max(np.mean(lengths), 1e-12))
            is_heavy_iter = length_cv_iter >= 0.85 or float(np.max(lengths) / max(np.median(lengths), 1e-12)) >= 4.0
            if is_heavy_iter:
                # Include globally long tasks even when LPT placed them outside
                # the current bottleneck VM; these are the highest-value heavy-tail
                # candidates for cross-VM moves.
                for ti in np.argsort(lengths)[::-1][:min(8, n)]:
                    ti = int(ti)
                    if ti not in search_tasks:
                        search_tasks.append(ti)
            for ti in search_tasks:
                old_vm = current[ti]
                old_duration = lengths[ti] / speeds[old_vm]
                old_cost = old_duration * vm_costs[old_vm]
                best_vm, best_move_score = old_vm, current_score
                for new_vm in range(k):
                    if new_vm == old_vm:
                        continue
                    new_duration = lengths[ti] / speeds[new_vm]
                    trial_loads = current_loads.copy()
                    trial_loads[old_vm] -= old_duration
                    trial_loads[new_vm] += new_duration
                    trial_cost = current_cost - old_cost + new_duration * vm_costs[new_vm]
                    trial_score = score_from_loads(trial_loads, trial_cost)
                    if trial_score < best_move_score - 1e-12:
                        best_vm, best_move_score = new_vm, trial_score
                if best_vm != old_vm:
                    new_duration = lengths[ti] / speeds[best_vm]
                    current_loads[old_vm] -= old_duration
                    current_loads[best_vm] += new_duration
                    current_cost += new_duration * vm_costs[best_vm] - old_cost
                    current[ti] = best_vm
                    current_score = best_move_score

            # Bounded pairwise swaps; evaluate each in O(number of VMs).
            if n <= 200 and inspect_count:
                swap_budget = min(60 if is_heavy_iter else 40, max(inspect_count, len(search_tasks)) * max(1, k - 1))
                tried = 0
                for ti in search_tasks[:min(len(search_tasks), 10 if is_heavy_iter else 8)]:
                    vm_i = current[ti]
                    dur_i = lengths[ti] / speeds[vm_i]
                    for tj in range(n):
                        vm_j = current[tj]
                        if vm_j == vm_i or vm_j == bottleneck_vm:
                            continue
                        dur_j = lengths[tj] / speeds[vm_j]
                        trial_loads = current_loads.copy()
                        trial_loads[vm_i] += lengths[tj] / speeds[vm_i] - dur_i
                        trial_loads[vm_j] += lengths[ti] / speeds[vm_j] - dur_j
                        trial_cost = current_cost - dur_i * vm_costs[vm_i] - dur_j * vm_costs[vm_j] \
                                     + (lengths[tj] / speeds[vm_i]) * vm_costs[vm_i] \
                                     + (lengths[ti] / speeds[vm_j]) * vm_costs[vm_j]
                        trial_score = score_from_loads(trial_loads, trial_cost)
                        if trial_score < current_score - 1e-12:
                            current[ti], current[tj] = vm_j, vm_i
                            current_loads, current_cost, current_score = trial_loads, trial_cost, trial_score
                            vm_i = current[ti]
                            dur_i = lengths[ti] / speeds[vm_i]
                        tried += 1
                        if tried >= swap_budget:
                            break
                    if tried >= swap_budget:
                        break

            current_tuple = tuple(current)
            candidate_score = evaluate(current_tuple)[0]
            candidate_makespan = evaluate(current_tuple)[1]
            if candidate_score < best_score - 1e-12:
                best, best_score = current_tuple, candidate_score
            if (candidate_makespan <= makespan_ceiling + 1e-9
                    and candidate_score < safe_best_score - 1e-12):
                safe_best, safe_best_score = current_tuple, candidate_score

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

    # Deterministic, bounded heavy-tail neighborhood search.
    # Targets globally long tasks, low-load VMs, and cross-VM long/short swaps.
    heavy_tail_cv = float(np.std(lengths) / max(np.mean(lengths), 1e-12))
    heavy_tail_ratio = float(np.max(lengths) / max(np.median(lengths), 1e-12))
    heavy_tail_detected = bool(heavy_tail_cv >= 0.85 or heavy_tail_ratio >= 4.0)
    targeted_checks = 0
    targeted_improvements = 0
    current_seed = tuple(safe_best)
    if k > 1 and n > 1:
        long_count = min(n, 48 if n < 500 else 24)
        long_tasks = sorted(range(n), key=lambda i: (-lengths[i], i))[:long_count]
        short_tasks = sorted(range(n), key=lambda i: (lengths[i], i))[:min(n, 64 if n < 500 else 24)]
        for _round in range(2 if n < 500 else 1):
            current_eval = evaluate(current_seed)
            current_loads = np.asarray(current_eval[5], dtype=float)
            light_vms = [int(x) for x in np.argsort(current_loads)]
            destinations = list(dict.fromkeys(
                [int(np.argmax(speeds)), int(np.argmin(vm_costs)),
                 int(np.argmin(max_watts))] + light_vms[:min(3, k)]
            ))
            best_assignment = current_seed
            best_score = current_eval[0]
            for ti in long_tasks:
                old_vm = current_seed[ti]
                dests = destinations if heavy_tail_detected else list(range(k))
                for new_vm in dests:
                    if new_vm == old_vm:
                        continue
                    trial = list(current_seed)
                    trial[ti] = new_vm
                    trial = tuple(trial)
                    trial_eval = evaluate(trial)
                    targeted_checks += 1
                    if (trial_eval[1] <= makespan_ceiling + 1e-9
                            and trial_eval[0] < best_score - 1e-12):
                        best_assignment, best_score = trial, trial_eval[0]
            swap_budget = 160 if n < 500 else 48
            swaps = 0
            for ti in long_tasks[:min(len(long_tasks), 24)]:
                for tj in short_tasks:
                    if ti == tj or current_seed[ti] == current_seed[tj]:
                        continue
                    trial = list(current_seed)
                    trial[ti], trial[tj] = trial[tj], trial[ti]
                    trial = tuple(trial)
                    trial_eval = evaluate(trial)
                    targeted_checks += 1
                    swaps += 1
                    if (trial_eval[1] <= makespan_ceiling + 1e-9
                            and trial_eval[0] < best_score - 1e-12):
                        best_assignment, best_score = trial, trial_eval[0]
                    if swaps >= swap_budget:
                        break
                if swaps >= swap_budget:
                    break
            if best_assignment == current_seed:
                break
            current_seed = best_assignment
            targeted_improvements += 1
        if (evaluate(current_seed)[1] <= makespan_ceiling + 1e-9
                and evaluate(current_seed)[0] < evaluate(safe_best)[0] - 1e-12):
            safe_best = current_seed
            safe_best_score = evaluate(current_seed)[0]

    # Makespan safety guard: the stochastic search must not return a schedule
    # with a worse makespan than its deterministic LPT incumbent. Energy/cost
    # trade-offs are reported separately and must not obscure this guarantee.
    def assignment_makespan(assignment):
        load_vector = np.zeros(k, dtype=float)
        for ti, vi in enumerate(assignment):
            load_vector[vi] += lengths[ti] / speeds[vi]
        return float(load_vector.max(initial=0.0))

    lpt_makespan = assignment_makespan(lpt)

    # Select across both the stochastic incumbent and the LPT-seeded local
    # search result. Earlier versions could accidentally discard safe_best
    # whenever `best` was a valid-but-worse LPT-equivalent candidate.
    eligible = []
    for candidate in (tuple(best), tuple(safe_best), tuple(lpt)):
        if assignment_makespan(candidate) <= makespan_ceiling + 1e-9:
            eligible.append((evaluate(candidate)[0], candidate))
    if not eligible:
        final_candidate = tuple(lpt)
        final_score = evaluate(final_candidate)[0]
    else:
        final_score, final_candidate = min(eligible, key=lambda item: item[0])

    # The final candidate must always meet the configured makespan ceiling.
    if assignment_makespan(final_candidate) > makespan_ceiling + 1e-9:
        final_candidate = tuple(lpt)
        final_score = evaluate(final_candidate)[0]

    best = final_candidate
    best_score = final_score
    selected_lpt_fallback = tuple(best) == tuple(lpt)

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
    result = pd.DataFrame(records)
    result.attrs["algorithm"] = "ECOTASKOPT"
    result.attrs["fitness_score"] = float(best_score)
    result.attrs["makespan_slack"] = float(makespan_slack)
    result.attrs["makespan_ceiling"] = float(makespan_ceiling)
    result.attrs["lpt_makespan"] = float(max(
        sum(float(task.length_mi) / float(vm.mips)
            for task, assigned_vm in zip(tasks, lpt) if assigned_vm == vm.vm_id)
        for vm in vms
    ))
    result.attrs["final_makespan"] = float(max(
        sum(float(task.length_mi) / float(vm.mips)
            for task, assigned_vm in zip(tasks, best) if assigned_vm == vm.vm_id)
        for vm in vms
    ))
    result.attrs["tasks_changed_from_lpt"] = int(sum(
        int(a != b) for a, b in zip(best, lpt)
    ))
    result.attrs["fallback_used"] = bool(selected_lpt_fallback)
    result.attrs["selection_source"] = "lpt_fallback" if selected_lpt_fallback else "hybrid_candidate"
    result.attrs["heavy_tail_detected"] = bool(heavy_tail_detected)
    result.attrs["targeted_search_checks"] = int(targeted_checks)
    result.attrs["targeted_search_improvements"] = int(targeted_improvements)
    result.attrs["lpt_local_search_checks"] = int(local_search_checks)
    result.attrs["lpt_local_search_improvements"] = int(local_search_improvements)
    result.attrs["duplicate_candidates_skipped"] = int(duplicate_candidate_count)
    result.attrs["heavy_tail_detected"] = bool(float(np.std(lengths) / max(np.mean(lengths), 1e-12)) >= 0.85 or float(np.max(lengths) / max(np.median(lengths), 1e-12)) >= 4.0)
    result.attrs["objective_weight_makespan"] = float(w_makespan)
    result.attrs["objective_weight_energy"] = float(w_energy)
    result.attrs["objective_weight_cost"] = float(w_cost)
    result.attrs["objective_weight_imbalance"] = float(imbalance_weight)
    result.attrs["effective_iterations"] = int(it + 1 if 'it' in locals() else 0)
    result.attrs["early_stopped"] = bool(no_improve >= patience and effective_iterations > min_iterations)
    return result
