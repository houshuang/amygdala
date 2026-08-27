"""Tests for limbic.amygdala.serendipity."""
import numpy as np
import pytest
from limbic.amygdala import serendipity as ser


def test_inverted_u_peaks_at_center():
    assert ser.inverted_u(0.65, 0.5, 0.8) == pytest.approx(1.0)   # centre
    assert ser.inverted_u(0.5, 0.5, 0.8) == pytest.approx(0.0)    # edge
    assert ser.inverted_u(0.9, 0.5, 0.8) == 0.0                    # outside
    assert ser.inverted_u(0.2, 0.5, 0.8) == 0.0


def _vec(*xs):
    return np.array(xs, dtype=float)


def test_serendipity_prefers_midband_over_near_duplicate():
    # a-b nearly identical (obvious); a-c moderate (surprising); a-d unrelated
    ids = ["a", "b", "c", "d"]
    emb = np.array([
        [1.0, 0.0, 0.0],          # a
        [0.99, 0.14, 0.0],        # b ~ a (cos ~0.99) -> too obvious
        [0.7, 0.7, 0.0],          # c (cos ~0.7 with a) -> sweet spot
        [0.0, 0.0, 1.0],          # d orthogonal -> unrelated
    ])
    out = ser.serendipity_pairs(ids, emb, band=(0.5, 0.85), top=10)
    pair_set = {frozenset((o["a"], o["b"])) for o in out}
    assert frozenset(("a", "c")) in pair_set        # surprising pair surfaced
    assert frozenset(("a", "b")) not in pair_set    # near-duplicate excluded (>hi)
    assert frozenset(("a", "d")) not in pair_set    # unrelated excluded (<lo)


def test_facet_bonus_boosts_cross_source_pairs():
    ids = ["a", "b", "c"]
    # a-b and a-c both ~0.7; b is same facet as a, c is a different facet
    emb = np.array([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.7, 0.0, 0.7]])
    metas = [{"src": "blog"}, {"src": "blog"}, {"src": "roam"}]
    out = ser.serendipity_pairs(ids, emb, metas=metas, facet_key=lambda m: m["src"],
                                band=(0.5, 0.85), facet_bonus=0.3, top=10)
    score = {frozenset((o["a"], o["b"])): o["score"] for o in out}
    assert score[frozenset(("a", "c"))] > score[frozenset(("a", "b"))]


def test_abc_bridge_finds_transitive_link():
    # A and C not similar; B bridges both
    ids = ["A", "B", "C"]
    emb = np.array([
        [1.0, 0.0, 0.0],                  # A
        [0.7071, 0.0, 0.7071],            # B ~0.71 to both A and C
        [0.0, 0.0, 1.0],                  # C orthogonal to A
    ])
    out = ser.abc_bridges(ids, emb, low=0.5, high=0.6, top=5)
    assert out and out[0]["bridge_b"] == "B"
    assert {out[0]["a"], out[0]["c"]} == {"A", "C"}
