#!/usr/bin/env python3
"""Comprehensive knowledge_map experiment: cross-test all feature combinations.

Tests the full matrix of:
  - Propagators: heuristic, bayesian
  - Strategies: eig, entropy, random, level_order, most_connected
  - Observation modes: direct, noisy (overclaim rates 0.1, 0.2, 0.4)
  - Calibration: none, foil-based (with overclaiming users)
  - Topologies: chain, wide_tree, dense_dag, diamond, flat
  - Batch selection: sequential vs batch(3) vs batch(5)

Usage:
    python3 experiments/exp_knowledge_map_matrix.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbic.amygdala.knowledge_map import (
    BeliefState,
    KnowledgeGraph,
    FAMILIARITY_LEVELS,
    calibrate_beliefs,
    adjust_for_calibration,
    init_beliefs,
    is_converged,
    knowledge_fringes,
    next_probe,
    next_probe_batch,
    update_beliefs,
)


SEED = 42
N_TRIALS = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# ── Graph topologies ─────────────────────────────────────


def make_chain(n: int = 20) -> KnowledgeGraph:
    nodes = []
    for i in range(n):
        prereqs = [f"c{i-1:02d}"] if i > 0 else []
        nodes.append({"id": f"c{i:02d}", "title": f"Chain {i}", "level": i + 1,
                       "obscurity": 3, "prerequisites": prereqs})
    return KnowledgeGraph(nodes=nodes)


def make_wide_tree(branches: int = 5, depth: int = 4) -> KnowledgeGraph:
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    for b in range(branches):
        parent = "root"
        for d in range(depth):
            nid = f"b{b}_d{d}"
            nodes.append({"id": nid, "title": f"B{b}D{d}", "level": d + 2,
                           "obscurity": 3, "prerequisites": [parent]})
            parent = nid
    return KnowledgeGraph(nodes=nodes)


def make_dense_dag(n: int = 20, n_edges: int = 40, seed: int = SEED) -> KnowledgeGraph:
    rng = random.Random(seed)
    nodes = [{"id": f"d{i:02d}", "title": f"D{i}", "level": (i // 5) + 1,
              "obscurity": 3, "prerequisites": []} for i in range(n)]
    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(possible)
    for src, dst in possible[:n_edges]:
        nodes[dst]["prerequisites"].append(f"d{src:02d}")
    return KnowledgeGraph(nodes=nodes)


def make_diamond(n_layers: int = 5, width: int = 3) -> KnowledgeGraph:
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    prev_layer = ["root"]
    for layer in range(n_layers):
        curr_layer = []
        for w in range(width):
            nid = f"l{layer}_w{w}"
            nodes.append({"id": nid, "title": f"L{layer}W{w}", "level": layer + 2,
                           "obscurity": 3, "prerequisites": list(prev_layer)})
            curr_layer.append(nid)
        prev_layer = curr_layer
    return KnowledgeGraph(nodes=nodes)


def make_flat(n: int = 20) -> KnowledgeGraph:
    nodes = [{"id": f"f{i:02d}", "title": f"Flat {i}", "level": 1,
              "obscurity": 3, "prerequisites": []} for i in range(n)]
    return KnowledgeGraph(nodes=nodes)


ALL_TOPOLOGIES = {
    "chain_20": lambda: make_chain(20),
    "wide_tree_5x4": lambda: make_wide_tree(5, 4),
    "dense_dag_20_40": lambda: make_dense_dag(20, 40),
    "diamond_5x3": lambda: make_diamond(5, 3),
    "flat_20": lambda: make_flat(20),
}


# ── Ground truth generation ──────────────────────────────


def _topo_sort(graph: KnowledgeGraph) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def visit(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        for prereq in graph.prerequisites_of(nid):
            visit(prereq)
        order.append(nid)

    for n in graph.nodes:
        visit(n["id"])
    return order


def generate_truth(graph: KnowledgeGraph, seed: int = SEED) -> dict[str, str]:
    """Generate ground truth respecting prerequisite structure."""
    rng = random.Random(seed)
    order = _topo_sort(graph)
    truth: dict[str, str] = {}
    for nid in order:
        prereqs = graph.prerequisites_of(nid)
        all_known = all(truth.get(p) in ("basic", "solid", "deep") for p in prereqs) if prereqs else True
        p = 0.75 if all_known else 0.15
        truth[nid] = rng.choice(["basic", "solid", "deep"]) if rng.random() < p else rng.choice(["none", "heard_of"])
    return truth


def truth_to_binary(truth: dict[str, str]) -> dict[str, bool]:
    return {nid: fam in ("basic", "solid", "deep") for nid, fam in truth.items()}


# ── Simulated user (with optional overclaiming) ──────────


def simulate_response(
    truth_fam: str, overclaim_rate: float = 0.0, rng: random.Random | None = None,
) -> str:
    """Return a possibly-noisy familiarity self-report.

    With probability overclaim_rate, an "unknown" user inflates their report.
    """
    if rng is None:
        rng = random.Random()
    is_known = truth_fam in ("basic", "solid", "deep")
    if not is_known and rng.random() < overclaim_rate:
        return rng.choice(["basic", "solid"])  # overclaim
    return truth_fam


# ── Strategy selectors ───────────────────────────────────


def pick_eig(graph, state, rng):
    probe = next_probe(graph, state, strategy="eig")
    if probe is None:
        unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
        return unassessed[0] if unassessed else None
    return probe["node_id"]


def pick_entropy(graph, state, rng):
    probe = next_probe(graph, state, strategy="entropy")
    if probe is None:
        unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
        return unassessed[0] if unassessed else None
    return probe["node_id"]


def pick_random(graph, state, rng):
    unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
    return rng.choice(unassessed) if unassessed else None


def pick_level_order(graph, state, rng):
    unassessed = [n for n in graph.nodes if n["id"] not in state.assessed]
    if not unassessed:
        return None
    unassessed.sort(key=lambda n: (n.get("level", 1), n["id"]))
    return unassessed[0]["id"]


def pick_most_connected(graph, state, rng):
    unassessed = [n for n in graph.nodes if n["id"] not in state.assessed]
    if not unassessed:
        return None
    unassessed.sort(
        key=lambda n: len(n.get("prerequisites", [])) + len(graph.children_of(n["id"])),
        reverse=True,
    )
    return unassessed[0]["id"]


STRATEGIES = {
    "eig": pick_eig,
    "entropy": pick_entropy,
    "random": pick_random,
    "level_order": pick_level_order,
    "most_connected": pick_most_connected,
}


# ── Accuracy measurement ─────────────────────────────────


def measure_accuracy(state: BeliefState, truth_binary: dict[str, bool],
                     exclude: set[str] | None = None) -> float:
    exclude = exclude or set()
    correct = total = 0
    for nid, is_known in truth_binary.items():
        if nid in exclude:
            continue
        total += 1
        if (state.beliefs.get(nid, 0.3) >= 0.5) == is_known:
            correct += 1
    return correct / max(total, 1)


# ── Experiment A: Propagator × Strategy convergence ──────


def run_exp_a(topologies: dict[str, KnowledgeGraph]) -> dict:
    """Compare propagator × strategy convergence on each topology."""
    print("\n" + "=" * 80)
    print("EXPERIMENT A: Propagator × Strategy Convergence")
    print("=" * 80)

    propagators = ["heuristic", "bayesian"]
    results = {}

    for topo_name, graph in topologies.items():
        n_nodes = len(graph.nodes)
        n_edges = sum(len(n.get("prerequisites", [])) for n in graph.nodes)
        print(f"\n  {topo_name} ({n_nodes}n, {n_edges}e)")

        for propagator in propagators:
            for strat_name, pick_fn in STRATEGIES.items():
                key = f"{topo_name}/{propagator}/{strat_name}"

                if strat_name == "random":
                    # Average over N_TRIALS
                    all_accs = []
                    for trial in range(N_TRIALS):
                        rng = random.Random(SEED + trial)
                        truth = generate_truth(graph, seed=SEED + trial)
                        truth_bin = truth_to_binary(truth)
                        state = init_beliefs(graph, propagator=propagator)
                        accs = [measure_accuracy(state, truth_bin)]
                        for _ in range(n_nodes):
                            nid = pick_fn(graph, state, rng)
                            if nid is None:
                                accs.extend([accs[-1]] * (n_nodes - len(accs) + 1))
                                break
                            update_beliefs(graph, state, nid, truth[nid])
                            accs.append(measure_accuracy(state, truth_bin))
                        all_accs.append(accs[:n_nodes + 1])
                    # Pad to same length and average
                    max_len = max(len(a) for a in all_accs)
                    for a in all_accs:
                        while len(a) < max_len:
                            a.append(a[-1])
                    avg_accs = np.mean(all_accs, axis=0).tolist()
                else:
                    # Single deterministic run (average over multiple ground truths)
                    all_accs = []
                    for trial in range(N_TRIALS):
                        rng = random.Random(SEED + trial)
                        truth = generate_truth(graph, seed=SEED + trial)
                        truth_bin = truth_to_binary(truth)
                        state = init_beliefs(graph, propagator=propagator)
                        accs = [measure_accuracy(state, truth_bin)]
                        for _ in range(n_nodes):
                            nid = pick_fn(graph, state, rng)
                            if nid is None:
                                accs.extend([accs[-1]] * (n_nodes - len(accs) + 1))
                                break
                            update_beliefs(graph, state, nid, truth[nid])
                            accs.append(measure_accuracy(state, truth_bin))
                        all_accs.append(accs[:n_nodes + 1])
                    max_len = max(len(a) for a in all_accs)
                    for a in all_accs:
                        while len(a) < max_len:
                            a.append(a[-1])
                    avg_accs = np.mean(all_accs, axis=0).tolist()

                q80 = next((i for i, a in enumerate(avg_accs) if a >= 0.80), None)
                q90 = next((i for i, a in enumerate(avg_accs) if a >= 0.90), None)

                results[key] = {
                    "convergence": avg_accs,
                    "q80": q80,
                    "q90": q90,
                    "final": avg_accs[-1] if avg_accs else 0,
                }

        # Print summary for this topology
        print(f"    {'':16s}  {'heuristic':>28s}  {'bayesian':>28s}")
        print(f"    {'Strategy':16s}  {'Q→80%':>6s} {'Q→90%':>6s} {'Final':>6s}    {'Q→80%':>6s} {'Q→90%':>6s} {'Final':>6s}")
        print(f"    {'─' * 70}")
        for strat_name in STRATEGIES:
            row = f"    {strat_name:16s}"
            for prop in propagators:
                key = f"{topo_name}/{prop}/{strat_name}"
                r = results[key]
                q80 = str(r["q80"]) if r["q80"] is not None else ">all"
                q90 = str(r["q90"]) if r["q90"] is not None else ">all"
                row += f"  {q80:>6s} {q90:>6s} {r['final']:>6.1%}"
                row += "  "
            print(row)

    return results


# ── Experiment B: Noisy observation robustness ───────────


def run_exp_b(topologies: dict[str, KnowledgeGraph]) -> dict:
    """Test how noisy self-reports + overclaiming affect accuracy."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B: Noisy Observation Robustness")
    print("=" * 80)

    overclaim_rates = [0.0, 0.1, 0.2, 0.4]
    observation_modes = ["direct", "noisy"]
    results = {}

    # Use one topology (wide_tree) for this experiment
    graph = topologies["wide_tree_5x4"]
    n_nodes = len(graph.nodes)
    max_q = min(15, n_nodes)  # assess up to 15 nodes, measure accuracy on rest

    print(f"\n  Topology: wide_tree_5x4 ({n_nodes} nodes)")
    print(f"  Questions: {max_q}, Trials: {N_TRIALS}")
    print(f"  Strategy: eig, Propagator: bayesian")

    for overclaim in overclaim_rates:
        for obs_mode in observation_modes:
            key = f"overclaim_{overclaim}/{obs_mode}"
            trial_accs = []

            for trial in range(N_TRIALS):
                rng = random.Random(SEED + trial)
                truth = generate_truth(graph, seed=SEED + trial)
                truth_bin = truth_to_binary(truth)
                state = init_beliefs(graph, propagator="bayesian")

                for q in range(max_q):
                    probe = next_probe(graph, state, strategy="eig")
                    if probe is None:
                        break
                    nid = probe["node_id"]
                    reported = simulate_response(truth[nid], overclaim, rng)
                    update_beliefs(
                        graph, state, nid, reported,
                        noisy=(obs_mode == "noisy"),
                        overclaim_rate=overclaim if obs_mode == "noisy" else 0.15,
                    )

                acc = measure_accuracy(state, truth_bin, exclude=state.assessed)
                trial_accs.append(acc)

            results[key] = {
                "mean": float(np.mean(trial_accs)),
                "std": float(np.std(trial_accs)),
            }

    # Print results
    print(f"\n    {'Overclaim':>10s}  {'Direct':>14s}  {'Noisy':>14s}  {'Delta':>8s}")
    print(f"    {'─' * 52}")
    for overclaim in overclaim_rates:
        d = results[f"overclaim_{overclaim}/direct"]
        n = results[f"overclaim_{overclaim}/noisy"]
        delta = n["mean"] - d["mean"]
        print(f"    {overclaim:>10.0%}  {d['mean']:>6.1%} ±{d['std']:.2f}  "
              f"{n['mean']:>6.1%} ±{n['std']:.2f}  {delta:>+6.1%}")

    return results


# ── Experiment C: Foil calibration impact ────────────────


def run_exp_c(topologies: dict[str, KnowledgeGraph]) -> dict:
    """Test whether foil-based calibration improves accuracy for overclaiming users."""
    print("\n" + "=" * 80)
    print("EXPERIMENT C: Foil Calibration Impact")
    print("=" * 80)

    graph = topologies["wide_tree_5x4"]
    n_nodes = len(graph.nodes)
    max_q = min(12, n_nodes)
    n_foils = 3
    overclaim_rates = [0.0, 0.15, 0.3, 0.5]
    results = {}

    print(f"\n  Topology: wide_tree_5x4 ({n_nodes} nodes)")
    print(f"  Questions: {max_q}, Foils: {n_foils}, Trials: {N_TRIALS}")

    for overclaim in overclaim_rates:
        for use_calibration in [False, True]:
            key = f"overclaim_{overclaim}/{'calibrated' if use_calibration else 'uncalibrated'}"
            trial_accs = []

            for trial in range(N_TRIALS):
                rng = random.Random(SEED + trial)
                truth = generate_truth(graph, seed=SEED + trial)
                truth_bin = truth_to_binary(truth)
                state = init_beliefs(graph, propagator="bayesian")

                # Phase 1: regular assessment with EIG
                for q in range(max_q):
                    probe = next_probe(graph, state, strategy="eig")
                    if probe is None:
                        break
                    nid = probe["node_id"]
                    reported = simulate_response(truth[nid], overclaim, rng)
                    update_beliefs(graph, state, nid, reported)

                # Phase 2: foil assessment (if calibration enabled)
                if use_calibration:
                    foil_responses = []
                    for f in range(n_foils):
                        # Simulate foil responses — overclaiming user may claim to know fake concepts
                        if rng.random() < overclaim:
                            foil_responses.append({"node_id": f"foil_{f}", "familiarity": rng.choice(["basic", "solid"])})
                        else:
                            foil_responses.append({"node_id": f"foil_{f}", "familiarity": "none"})

                    cal_factor = calibrate_beliefs(state, foil_responses)
                    adjust_for_calibration(state, cal_factor, graph)

                acc = measure_accuracy(state, truth_bin, exclude=state.assessed)
                trial_accs.append(acc)

            results[key] = {
                "mean": float(np.mean(trial_accs)),
                "std": float(np.std(trial_accs)),
            }

    # Phase 3: test noisy mode with foil-informed overclaim_rate
    # This is the "right" architecture: foils calibrate the noise model
    for overclaim in overclaim_rates:
        key = f"overclaim_{overclaim}/noisy_calibrated"
        trial_accs = []

        for trial in range(N_TRIALS):
            rng = random.Random(SEED + trial)
            truth = generate_truth(graph, seed=SEED + trial)
            truth_bin = truth_to_binary(truth)
            state = init_beliefs(graph, propagator="bayesian")

            # Phase 1: present foils first to estimate overclaim_rate
            foil_responses = []
            for f in range(n_foils):
                if rng.random() < overclaim:
                    foil_responses.append({"node_id": f"foil_{f}", "familiarity": rng.choice(["basic", "solid"])})
                else:
                    foil_responses.append({"node_id": f"foil_{f}", "familiarity": "none"})
            cal_factor = calibrate_beliefs(state, foil_responses)
            estimated_overclaim = 1.0 - cal_factor  # Use foil signal as overclaim_rate

            # Phase 2: assess using noisy mode with estimated overclaim_rate
            for q in range(max_q):
                probe = next_probe(graph, state, strategy="eig")
                if probe is None:
                    break
                nid = probe["node_id"]
                reported = simulate_response(truth[nid], overclaim, rng)
                update_beliefs(
                    graph, state, nid, reported,
                    noisy=True,
                    overclaim_rate=max(0.05, estimated_overclaim),
                )

            acc = measure_accuracy(state, truth_bin, exclude=state.assessed)
            trial_accs.append(acc)

        results[key] = {
            "mean": float(np.mean(trial_accs)),
            "std": float(np.std(trial_accs)),
        }

    # Print results
    print(f"\n    {'Overclaim':>10s}  {'Direct':>14s}  {'Post-hoc cal':>14s}  {'Noisy+cal':>14s}")
    print(f"    {'─' * 60}")
    for overclaim in overclaim_rates:
        u = results[f"overclaim_{overclaim}/uncalibrated"]
        c = results[f"overclaim_{overclaim}/calibrated"]
        nc = results[f"overclaim_{overclaim}/noisy_calibrated"]
        print(f"    {overclaim:>10.0%}  {u['mean']:>6.1%} ±{u['std']:.2f}  "
              f"{c['mean']:>6.1%} ±{c['std']:.2f}  "
              f"{nc['mean']:>6.1%} ±{nc['std']:.2f}")

    return results


# ── Experiment D: Batch selection efficiency ─────────────


def run_exp_d(topologies: dict[str, KnowledgeGraph]) -> dict:
    """Compare sequential probing vs batch selection."""
    print("\n" + "=" * 80)
    print("EXPERIMENT D: Batch Selection Efficiency")
    print("=" * 80)

    graph = topologies["wide_tree_5x4"]
    n_nodes = len(graph.nodes)
    batch_sizes = [1, 3, 5]
    results = {}

    print(f"\n  Topology: wide_tree_5x4 ({n_nodes} nodes)")
    print(f"  Propagator: bayesian, Trials: {N_TRIALS}")

    for batch_size in batch_sizes:
        key = f"batch_{batch_size}"
        all_accs = []

        for trial in range(N_TRIALS):
            truth = generate_truth(graph, seed=SEED + trial)
            truth_bin = truth_to_binary(truth)
            state = init_beliefs(graph, propagator="bayesian")
            accs = [measure_accuracy(state, truth_bin)]
            total_questions = 0

            while total_questions < n_nodes:
                if batch_size == 1:
                    probe = next_probe(graph, state, strategy="eig")
                    if probe is None:
                        break
                    nids = [probe["node_id"]]
                else:
                    probes = next_probe_batch(graph, state, n=batch_size, strategy="eig")
                    if not probes:
                        break
                    nids = [p["node_id"] for p in probes]

                for nid in nids:
                    update_beliefs(graph, state, nid, truth[nid])
                    total_questions += 1
                accs.append(measure_accuracy(state, truth_bin))

            all_accs.append(accs)

        # Normalize to "questions asked" (not rounds)
        # For batch, each round asks batch_size questions
        max_len = max(len(a) for a in all_accs)
        for a in all_accs:
            while len(a) < max_len:
                a.append(a[-1])

        avg_accs = np.mean(all_accs, axis=0).tolist()
        q80 = next((i for i, a in enumerate(avg_accs) if a >= 0.80), None)
        q90 = next((i for i, a in enumerate(avg_accs) if a >= 0.90), None)

        results[key] = {
            "convergence": avg_accs,
            "q80": q80,
            "q90": q90,
            "n_rounds_to_80": q80,  # rounds, not individual questions
        }

    # Print results
    print(f"\n    {'Batch':>8s}  {'Rounds→80%':>12s}  {'Rounds→90%':>12s}  {'Qs→80%':>10s}")
    print(f"    {'─' * 48}")
    for bs in batch_sizes:
        r = results[f"batch_{bs}"]
        q80 = str(r["q80"]) if r["q80"] is not None else ">all"
        q90 = str(r["q90"]) if r["q90"] is not None else ">all"
        qs80 = str(r["q80"] * bs) if r["q80"] is not None else ">all"
        print(f"    {bs:>8d}  {q80:>12s}  {q90:>12s}  {qs80:>10s}")

    return results


# ── Experiment E: Fringe quality ─────────────────────────


def run_exp_e(topologies: dict[str, KnowledgeGraph]) -> dict:
    """Test how accurately fringes identify learning boundaries."""
    print("\n" + "=" * 80)
    print("EXPERIMENT E: Knowledge Fringe Accuracy")
    print("=" * 80)

    graph = topologies["wide_tree_5x4"]
    n_nodes = len(graph.nodes)
    results = {}

    print(f"\n  Topology: wide_tree_5x4 ({n_nodes} nodes)")
    print(f"  Strategy: eig, Propagator: bayesian, Trials: {N_TRIALS}")

    # After N questions, how well do the fringes match ground truth?
    question_counts = [3, 5, 8, 12]

    for n_q in question_counts:
        outer_precision_trials = []
        outer_recall_trials = []
        inner_precision_trials = []

        for trial in range(N_TRIALS):
            truth = generate_truth(graph, seed=SEED + trial)
            truth_bin = truth_to_binary(truth)
            state = init_beliefs(graph, propagator="bayesian")

            for q in range(n_q):
                probe = next_probe(graph, state, strategy="eig")
                if probe is None:
                    break
                update_beliefs(graph, state, probe["node_id"], truth[probe["node_id"]])

            fringes = knowledge_fringes(graph, state)

            # Ground truth fringes
            true_known = {nid for nid, k in truth_bin.items() if k}
            true_unknown = {nid for nid, k in truth_bin.items() if not k}
            true_outer = {
                nid for nid in true_unknown
                if all(p in true_known for p in graph.prerequisites_of(nid))
            }
            true_inner = {
                nid for nid in true_known
                if any(c in true_unknown for c in graph.children_of(nid))
            }

            # Outer fringe precision/recall
            pred_outer = set(fringes["outer_fringe"])
            if pred_outer:
                outer_precision_trials.append(len(pred_outer & true_outer) / len(pred_outer))
            if true_outer:
                outer_recall_trials.append(len(pred_outer & true_outer) / len(true_outer))

            # Inner fringe precision
            pred_inner = set(fringes["inner_fringe"])
            if pred_inner:
                inner_precision_trials.append(len(pred_inner & true_inner) / len(pred_inner))

        results[f"q{n_q}"] = {
            "outer_precision": float(np.mean(outer_precision_trials)) if outer_precision_trials else 0,
            "outer_recall": float(np.mean(outer_recall_trials)) if outer_recall_trials else 0,
            "inner_precision": float(np.mean(inner_precision_trials)) if inner_precision_trials else 0,
        }

    # Print
    print(f"\n    {'Questions':>10s}  {'Outer Prec':>12s}  {'Outer Recall':>14s}  {'Inner Prec':>12s}")
    print(f"    {'─' * 52}")
    for n_q in question_counts:
        r = results[f"q{n_q}"]
        print(f"    {n_q:>10d}  {r['outer_precision']:>12.1%}  {r['outer_recall']:>14.1%}  {r['inner_precision']:>12.1%}")

    return results


# ── Summary ──────────────────────────────────────────────


def print_synthesis(results_a: dict, results_b: dict, results_c: dict):
    """Print key findings."""
    print("\n" + "=" * 80)
    print("SYNTHESIS: Key Findings")
    print("=" * 80)

    # Best strategy × propagator combo
    print("\n  1. Best strategy × propagator (avg Q→80% across topologies):")
    combos: dict[str, list[int]] = {}
    for key, val in results_a.items():
        parts = key.split("/")
        combo = f"{parts[1]}/{parts[2]}"
        if val["q80"] is not None:
            combos.setdefault(combo, []).append(val["q80"])
    for combo in sorted(combos, key=lambda c: np.mean(combos[c])):
        avg = np.mean(combos[combo])
        print(f"    {combo:30s}  avg Q→80% = {avg:.1f}")

    # Noisy observation benefit
    print("\n  2. Noisy observation mode benefit (overclaim=20%):")
    if "overclaim_0.2/direct" in results_b and "overclaim_0.2/noisy" in results_b:
        d = results_b["overclaim_0.2/direct"]["mean"]
        n = results_b["overclaim_0.2/noisy"]["mean"]
        print(f"    Direct: {d:.1%}  Noisy: {n:.1%}  Delta: {n-d:+.1%}")

    # Calibration benefit
    print("\n  3. Foil calibration benefit (overclaim=30%):")
    if "overclaim_0.3/uncalibrated" in results_c and "overclaim_0.3/calibrated" in results_c:
        u = results_c["overclaim_0.3/uncalibrated"]["mean"]
        c = results_c["overclaim_0.3/calibrated"]["mean"]
        print(f"    Uncalibrated: {u:.1%}  Calibrated: {c:.1%}  Delta: {c-u:+.1%}")

    # EIG vs entropy
    print("\n  4. EIG vs Entropy advantage (avg Q→80%, bayesian propagator):")
    eig_qs = []
    ent_qs = []
    for key, val in results_a.items():
        if "/bayesian/eig" in key and val["q80"] is not None:
            eig_qs.append(val["q80"])
        elif "/bayesian/entropy" in key and val["q80"] is not None:
            ent_qs.append(val["q80"])
    if eig_qs and ent_qs:
        print(f"    EIG: {np.mean(eig_qs):.1f} questions  Entropy: {np.mean(ent_qs):.1f} questions  "
              f"Saved: {np.mean(ent_qs) - np.mean(eig_qs):.1f}")


# ── Main ─────────────────────────────────────────────────


def main():
    t0 = time.time()
    print("=" * 80)
    print("Knowledge Map: Comprehensive Feature Matrix Experiment")
    print(f"  {N_TRIALS} trials, seed={SEED}")
    print("=" * 80)

    # Build all topologies
    topologies = {name: fn() for name, fn in ALL_TOPOLOGIES.items()}
    for name, g in topologies.items():
        n_e = sum(len(n.get("prerequisites", [])) for n in g.nodes)
        print(f"  {name}: {len(g.nodes)} nodes, {n_e} edges")

    # Run experiments
    results_a = run_exp_a(topologies)
    results_b = run_exp_b(topologies)
    results_c = run_exp_c(topologies)
    results_d = run_exp_d(topologies)
    results_e = run_exp_e(topologies)

    # Synthesis
    print_synthesis(results_a, results_b, results_c)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "exp_knowledge_map_matrix.json"
    all_results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "n_trials": N_TRIALS,
            "seed": SEED,
            "topologies": {n: {"nodes": len(g.nodes), "edges": sum(len(nd.get("prerequisites", [])) for nd in g.nodes)}
                           for n, g in topologies.items()},
        },
        "exp_a_convergence": results_a,
        "exp_b_noisy": results_b,
        "exp_c_calibration": results_c,
        "exp_d_batch": results_d,
        "exp_e_fringes": results_e,
    }
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
