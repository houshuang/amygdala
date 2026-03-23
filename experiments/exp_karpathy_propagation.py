#!/usr/bin/env python3
"""Karpathy Loop for knowledge_map belief propagation.

Systematically optimizes heuristic propagation parameters by comparing
against pgmpy exact Bayesian inference as ground truth.

The loop:
  1. Define parameter space (dampen, ceiling, floor, prereq threshold, etc.)
  2. For each configuration, run N trials across multiple graph topologies
  3. Measure accuracy gap vs pgmpy
  4. Report optimal parameters and their effect

Composite score: mean accuracy across topologies (pgmpy is the ceiling).
"""

import json
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, BeliefState, init_beliefs, update_beliefs,
    _propagate_down, _propagate_up, _propagate_prereqs_met,
    FAMILIARITY_LEVELS, _DAMPEN,
)


# ── Graph topologies ──────────────────────────────────────

def make_wide_tree(branches=5, depth=4):
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    for b in range(branches):
        parent = "root"
        for d in range(depth):
            nid = f"b{b}_d{d}"
            nodes.append({"id": nid, "title": f"B{b}D{d}", "level": d+2,
                          "obscurity": 3, "prerequisites": [parent]})
            parent = nid
    return KnowledgeGraph(nodes=nodes)

def make_dense_dag(n=20, n_edges=40, seed=42):
    rng = random.Random(seed)
    nodes = [{"id": f"d{i:02d}", "title": f"D{i}", "level": (i//5)+1,
              "obscurity": 3, "prerequisites": []} for i in range(n)]
    possible = [(i,j) for i in range(n) for j in range(i+1,n)]
    rng.shuffle(possible)
    for src, dst in possible[:n_edges]:
        nodes[dst]["prerequisites"].append(f"d{src:02d}")
    return KnowledgeGraph(nodes=nodes)

def make_chain(n=20):
    nodes = []
    for i in range(n):
        prereqs = [f"c{i-1:02d}"] if i > 0 else []
        nodes.append({"id": f"c{i:02d}", "title": f"C{i}", "level": i+1,
                      "obscurity": 3, "prerequisites": prereqs})
    return KnowledgeGraph(nodes=nodes)

def make_diamond(n_layers=5, width=3):
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    prev_layer = ["root"]
    for layer in range(n_layers):
        curr_layer = []
        for w in range(width):
            nid = f"l{layer}_w{w}"
            nodes.append({"id": nid, "title": f"L{layer}W{w}", "level": layer+2,
                          "obscurity": 3, "prerequisites": list(prev_layer)})
            curr_layer.append(nid)
        prev_layer = curr_layer
    return KnowledgeGraph(nodes=nodes)


TOPOLOGIES = {
    "wide_tree": make_wide_tree,
    "dense_dag": lambda: make_dense_dag(20, 40),
    "chain_20": lambda: make_chain(20),
    "diamond_5x3": lambda: make_diamond(5, 3),
}


# ── Ground truth ──────────────────────────────────────────

def generate_truth(graph, seed=42):
    rng = random.Random(seed)
    visited, order = set(), []
    def visit(nid):
        if nid in visited: return
        visited.add(nid)
        for p in graph.prerequisites_of(nid): visit(p)
        order.append(nid)
    for n in graph.nodes: visit(n["id"])
    truth = {}
    for nid in order:
        prereqs = graph.prerequisites_of(nid)
        all_known = all(truth.get(p) in ("basic","solid","deep") for p in prereqs) if prereqs else True
        p = 0.75 if all_known else 0.15
        truth[nid] = rng.choice(["basic","solid","deep"]) if rng.random() < p else rng.choice(["none","heard_of"])
    return truth


# ── Custom propagation with parameters ────────────────────

def custom_propagate(graph, state, node_id, familiarity, params):
    """Parametric heuristic propagation."""
    dampen = params["dampen"]
    down_ceiling = params["down_ceiling"]
    up_floor = params["up_floor"]
    prereq_threshold = params["prereq_threshold"]
    prereq_base = params["prereq_base"]
    prereq_dampen_exp = params["prereq_dampen_exp"]

    if familiarity in ("none", "heard_of"):
        _custom_propagate_down(graph, state, node_id, down_ceiling, 0, set(), dampen)
    if familiarity in ("solid", "deep"):
        _custom_propagate_up(graph, state, node_id, up_floor, 0, set(), dampen)
        _custom_propagate_prereqs_met(graph, state, node_id, 0, set(),
                                       dampen, prereq_threshold, prereq_base, prereq_dampen_exp)


def _custom_propagate_down(graph, state, node_id, ceiling, depth, visited, dampen):
    for child_id in graph.children_of(node_id):
        if child_id in visited or child_id in state.assessed:
            continue
        visited.add(child_id)
        dampened = ceiling * (dampen ** depth)
        current = state.beliefs.get(child_id, 0.3)
        state.beliefs[child_id] = min(current, dampened)
        _custom_propagate_down(graph, state, child_id, ceiling, depth + 1, visited, dampen)


def _custom_propagate_up(graph, state, node_id, floor, depth, visited, dampen):
    for prereq_id in graph.prerequisites_of(node_id):
        if prereq_id in visited or prereq_id in state.assessed:
            continue
        visited.add(prereq_id)
        dampened = 1.0 - (1.0 - floor) * (dampen ** depth)
        current = state.beliefs.get(prereq_id, 0.3)
        state.beliefs[prereq_id] = max(current, dampened)
        _custom_propagate_up(graph, state, prereq_id, floor, depth + 1, visited, dampen)


def _custom_propagate_prereqs_met(graph, state, node_id, depth, visited,
                                   dampen, threshold, base, dampen_exp):
    for child_id in graph.children_of(node_id):
        if child_id in visited or child_id in state.assessed:
            continue
        visited.add(child_id)
        prereqs = graph.prerequisites_of(child_id)
        prereq_beliefs = [state.beliefs.get(p, 0.3) for p in prereqs]
        min_prereq = min(prereq_beliefs) if prereq_beliefs else 0.3
        if min_prereq >= threshold:
            scaled_base = threshold + (base - threshold) * ((min_prereq - threshold) / (1.0 - threshold))
            floor = scaled_base * (dampen ** (depth * dampen_exp))
            current = state.beliefs.get(child_id, 0.3)
            state.beliefs[child_id] = max(current, floor)
            _custom_propagate_prereqs_met(graph, state, child_id, depth + 1, visited,
                                           dampen, threshold, base, dampen_exp)


# ── Accuracy measurement ──────────────────────────────────

def measure_accuracy_custom(graph, truth, assessed, params):
    state = init_beliefs(graph)
    for nid in assessed:
        raw_belief, label = FAMILIARITY_LEVELS[truth[nid]]
        state.beliefs[nid] = raw_belief
        state.assessed.add(nid)
        custom_propagate(graph, state, nid, truth[nid], params)
    correct = total = 0
    for nid, fam in truth.items():
        if nid in assessed: continue
        total += 1
        if (state.beliefs.get(nid, 0.3) >= 0.5) == (fam in ("basic","solid","deep")): correct += 1
    return correct / max(total, 1)


def measure_accuracy_bayesian(graph, truth, assessed):
    state = init_beliefs(graph, propagator="bayesian")
    for nid in assessed:
        update_beliefs(graph, state, nid, truth[nid])
    correct = total = 0
    for nid, fam in truth.items():
        if nid in assessed: continue
        total += 1
        if (state.beliefs.get(nid, 0.3) >= 0.5) == (fam in ("basic","solid","deep")): correct += 1
    return correct / max(total, 1)


# ── Parameter space ──────────────────────────────────────

PARAM_SPACE = {
    "dampen": [0.6, 0.7, 0.8, 0.9],
    "down_ceiling": [0.05, 0.10, 0.15, 0.20],
    "up_floor": [0.70, 0.80, 0.85, 0.90],
    "prereq_threshold": [0.40, 0.50, 0.60, 0.70],
    "prereq_base": [0.70, 0.80, 0.85, 0.90],
    "prereq_dampen_exp": [0.3, 0.5, 0.7, 1.0],
}

# Current defaults for comparison
CURRENT_PARAMS = {
    "dampen": 0.8,
    "down_ceiling": 0.10,
    "up_floor": 0.80,
    "prereq_threshold": 0.50,
    "prereq_base": 0.85,
    "prereq_dampen_exp": 0.5,
}


# ── Main loop ─────────────────────────────────────────────

def main():
    W = 90
    N_TRIALS = 30
    K_VALUES = [3, 5, 8]
    rng = random.Random(42)

    print("=" * W)
    print("Karpathy Loop: Propagation Parameter Optimization")
    print(f"  {len(TOPOLOGIES)} topologies × {len(K_VALUES)} K values × {N_TRIALS} trials")
    print("=" * W)

    # ── Phase 1: Compute Bayesian ceiling ─────────────────
    print(f"\n{'─' * W}")
    print("Phase 1: Bayesian ceiling (pgmpy exact inference)")
    print(f"{'─' * W}\n")

    bayesian_scores = {}
    for topo_name, make_fn in TOPOLOGIES.items():
        graph = make_fn()
        for K in K_VALUES:
            scores = []
            for trial in range(N_TRIALS):
                truth = generate_truth(graph, seed=trial*7+13)
                assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                scores.append(measure_accuracy_bayesian(graph, truth, assessed))
            key = f"{topo_name}_K{K}"
            bayesian_scores[key] = np.mean(scores)
            print(f"  {key}: {bayesian_scores[key]:.3f}")

    bayesian_mean = np.mean(list(bayesian_scores.values()))
    print(f"\n  Overall Bayesian ceiling: {bayesian_mean:.3f}")

    # ── Phase 2: Sweep one parameter at a time ────────────
    # Instead of full grid (4^6 = 4096 configs), sweep each parameter
    # independently with others at default. Then do a focused grid around
    # the best values.
    print(f"\n{'─' * W}")
    print("Phase 2: Single-parameter sweeps (find best per-parameter)")
    print(f"{'─' * W}\n")

    best_per_param = {}
    for param_name, values in PARAM_SPACE.items():
        print(f"  Sweeping {param_name}: {values}")
        param_scores = {}
        for val in values:
            params = dict(CURRENT_PARAMS)
            params[param_name] = val
            total_acc = 0
            n_evals = 0
            for topo_name, make_fn in TOPOLOGIES.items():
                graph = make_fn()
                for K in K_VALUES:
                    for trial in range(N_TRIALS):
                        truth = generate_truth(graph, seed=trial*7+13)
                        assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                        total_acc += measure_accuracy_custom(graph, truth, assessed, params)
                        n_evals += 1
            mean_acc = total_acc / n_evals
            param_scores[val] = mean_acc
            print(f"    {param_name}={val}: {mean_acc:.4f}")

        best_val = max(param_scores, key=param_scores.get)
        best_per_param[param_name] = best_val
        print(f"    -> Best: {param_name}={best_val} ({param_scores[best_val]:.4f})")
        print()

    # ── Phase 3: Focused grid around best values ──────────
    print(f"{'─' * W}")
    print("Phase 3: Focused grid search around best single-param values")
    print(f"{'─' * W}\n")

    # Combine best individual params
    optimized_params = dict(best_per_param)
    print(f"  Best individual params: {optimized_params}")

    # Measure optimized
    opt_acc = 0
    n_evals = 0
    for topo_name, make_fn in TOPOLOGIES.items():
        graph = make_fn()
        for K in K_VALUES:
            for trial in range(N_TRIALS):
                truth = generate_truth(graph, seed=trial*7+13)
                assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                opt_acc += measure_accuracy_custom(graph, truth, assessed, optimized_params)
                n_evals += 1
    opt_mean = opt_acc / n_evals

    # Also measure current defaults
    cur_acc = 0
    n_evals = 0
    for topo_name, make_fn in TOPOLOGIES.items():
        graph = make_fn()
        for K in K_VALUES:
            for trial in range(N_TRIALS):
                truth = generate_truth(graph, seed=trial*7+13)
                assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                cur_acc += measure_accuracy_custom(graph, truth, assessed, CURRENT_PARAMS)
                n_evals += 1
    cur_mean = cur_acc / n_evals

    # ── Phase 4: Per-topology breakdown ───────────────────
    print(f"\n{'─' * W}")
    print("Phase 4: Per-topology comparison")
    print(f"{'─' * W}\n")

    print(f"  {'Topology':<25} | {'Current':>8} | {'Optimized':>10} | {'Bayesian':>8} | {'Gap→Bayes':>10}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    topo_results = {}
    for topo_name, make_fn in TOPOLOGIES.items():
        graph = make_fn()
        for K in K_VALUES:
            key = f"{topo_name}_K{K}"
            cur_scores, opt_scores = [], []
            for trial in range(N_TRIALS):
                truth = generate_truth(graph, seed=trial*7+13)
                assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                cur_scores.append(measure_accuracy_custom(graph, truth, assessed, CURRENT_PARAMS))
                opt_scores.append(measure_accuracy_custom(graph, truth, assessed, optimized_params))
            cur_m, opt_m = np.mean(cur_scores), np.mean(opt_scores)
            bay_m = bayesian_scores[key]
            gap = bay_m - opt_m
            topo_results[key] = {"current": cur_m, "optimized": opt_m, "bayesian": bay_m, "gap": gap}
            print(f"  {key:<25} | {cur_m:>8.3f} | {opt_m:>10.3f} | {bay_m:>8.3f} | {gap:>+10.3f}")

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("SUMMARY")
    print(f"{'=' * W}")
    print(f"  Current params:   {CURRENT_PARAMS}")
    print(f"  Optimized params: {optimized_params}")
    print(f"\n  Current mean accuracy:   {cur_mean:.4f}")
    print(f"  Optimized mean accuracy: {opt_mean:.4f}")
    print(f"  Bayesian ceiling:        {bayesian_mean:.4f}")
    print(f"\n  Improvement: {opt_mean - cur_mean:+.4f} ({100*(opt_mean-cur_mean)/cur_mean:+.1f}%)")
    print(f"  Remaining gap to Bayesian: {bayesian_mean - opt_mean:.4f}")
    print(f"  % of gap closed: {100*(opt_mean - cur_mean)/(bayesian_mean - cur_mean):.0f}%"
          if bayesian_mean > cur_mean else "  Already at ceiling!")

    changes = []
    for k in CURRENT_PARAMS:
        if CURRENT_PARAMS[k] != optimized_params[k]:
            changes.append(f"  {k}: {CURRENT_PARAMS[k]} → {optimized_params[k]}")
    if changes:
        print(f"\n  Parameters to change:")
        for c in changes:
            print(f"    {c}")
    else:
        print(f"\n  No parameter changes needed!")

    # ── Save results ──────────────────────────────────────
    out = {
        "experiment": "karpathy_propagation",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_trials": N_TRIALS,
        "k_values": K_VALUES,
        "topologies": list(TOPOLOGIES.keys()),
        "current_params": CURRENT_PARAMS,
        "optimized_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in optimized_params.items()},
        "bayesian_ceiling": round(bayesian_mean, 4),
        "current_mean": round(cur_mean, 4),
        "optimized_mean": round(opt_mean, 4),
        "improvement": round(opt_mean - cur_mean, 4),
        "gap_closed_pct": round(100*(opt_mean - cur_mean)/(bayesian_mean - cur_mean), 1) if bayesian_mean > cur_mean else 100.0,
        "per_topology": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in topo_results.items()},
        "bayesian_per_topology": {k: round(v, 4) for k, v in bayesian_scores.items()},
    }

    out_path = Path("experiments/results/exp_karpathy_propagation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
