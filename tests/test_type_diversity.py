"""Tests for limbic.amygdala.search.type_diversity_cap."""
from limbic.amygdala import type_diversity_cap, Result


def _r(rid, typ):
    return Result(id=rid, score=1.0, content=rid, metadata={"t": typ}, source="x")


def key(r):
    return r.metadata["t"]


def test_caps_dominant_type():
    # 8 of type A, 2 of type B; cap 0.6 over k=5 -> max 3 of A in top-5
    results = [_r(f"a{i}", "A") for i in range(8)] + [_r(f"b{i}", "B") for i in range(2)]
    out = type_diversity_cap(results, key, max_fraction=0.6, k=5)
    assert len(out) == 5
    assert sum(1 for r in out if key(r) == "A") <= 3
    assert any(key(r) == "B" for r in out)   # minority type makes it in


def test_backfills_when_short():
    # only one type available -> cap can't be met, must backfill to k
    results = [_r(f"a{i}", "A") for i in range(5)]
    out = type_diversity_cap(results, key, max_fraction=0.5, k=5)
    assert len(out) == 5


def test_preserves_order_within_kept():
    results = [_r("a0", "A"), _r("b0", "B"), _r("a1", "A"), _r("a2", "A")]
    out = type_diversity_cap(results, key, max_fraction=0.5, k=4)
    # a0 before b0 before remaining A's; relative order preserved
    assert [r.id for r in out][:2] == ["a0", "b0"]
