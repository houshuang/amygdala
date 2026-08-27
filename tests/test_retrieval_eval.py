"""Tests for limbic.amygdala.retrieval_eval — IR metrics, pooling, scoring."""
import math
import pytest
from limbic.amygdala import retrieval_eval as rev


def test_dcg_and_ndcg_perfect_vs_reversed():
    rel = {"a": 3, "b": 2, "c": 1}
    perfect = ["a", "b", "c"]
    reversed_ = ["c", "b", "a"]
    assert rev.ndcg(perfect, rel, k=3) == pytest.approx(1.0)
    assert rev.ndcg(reversed_, rel, k=3) < rev.ndcg(perfect, rel, k=3)


def test_ndcg_known_value():
    rel = {"a": 3, "b": 0, "c": 1}
    # dcg = 7/1 + 0 + 1/log2(4) = 7.5 ; idcg(3,1,0) = 7 + 1/log2(3)
    idcg = 7 + 1 / math.log2(3)
    assert rev.ndcg(["a", "b", "c"], rel, k=3) == pytest.approx(7.5 / idcg)


def test_ndcg_zero_when_no_grades():
    assert rev.ndcg(["a", "b"], {}, k=10) == 0.0


def test_recall_threshold_and_none():
    rel = {"a": 3, "b": 2, "c": 1}  # 2 docs at grade>=2
    assert rev.recall(["a", "x", "y"], rel, k=3, rel_threshold=2) == pytest.approx(0.5)
    assert rev.recall(["a", "b"], rel, k=3, rel_threshold=2) == pytest.approx(1.0)
    assert rev.recall(["a"], {"a": 1}, k=3, rel_threshold=2) is None  # no relevant


def test_mrr():
    rel = {"x": 2}
    assert rev.mrr(["a", "b", "x"], rel, k=10, rel_threshold=2) == pytest.approx(1 / 3)
    assert rev.mrr(["a", "b"], rel, k=10, rel_threshold=2) == 0.0


def test_average_precision_known_value():
    rel = {"d1": 2, "d3": 2}
    ap = rev.average_precision(["d1", "d2", "d3"], rel, rel_threshold=2)
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)


def test_pool_union_depth():
    runs = {"q1": {"m1": ["a", "b", "c"], "m2": ["b", "d", "e"]}}
    assert rev.pool(runs, depth=2) == {"q1": {"a", "b", "d"}}


def test_judge_pool_incremental():
    pooled = {"q1": {"a", "b"}}
    queries = {"q1": "find a"}
    calls = []

    def judge_fn(q, d):
        calls.append(d)
        return 3 if "DOC-a" in d else 0

    qrels = rev.judge_pool(pooled, queries, lambda d: f"DOC-{d}", judge_fn)
    assert qrels["q1"]["a"] == 3 and qrels["q1"]["b"] == 0
    # incremental: existing pair 'a' is skipped, only 'b' re-judged-as-new would run
    calls.clear()
    pooled2 = {"q1": {"a", "b", "c"}}
    rev.judge_pool(pooled2, queries, lambda d: f"DOC-{d}", judge_fn, existing=qrels)
    assert calls == ["DOC-c"]   # only the new doc 'c' is judged (text DOC-c)


def test_score_ranks_better_method_higher():
    runs = {
        "q1": {"good": ["a", "b"], "bad": ["b", "a"]},
        "q2": {"good": ["c", "d"], "bad": ["d", "c"]},
    }
    qrels = {"q1": {"a": 3, "b": 0}, "q2": {"c": 3, "d": 0}}
    out = rev.score(runs, qrels, rel_threshold=2, strata={"q1": "x", "q2": "y"})
    assert out["methods"]["good"]["ndcg"] > out["methods"]["bad"]["ndcg"]
    assert out["methods"]["good"]["n"] == 2
    assert "by_stratum" in out
    assert rev.format_report(out)  # renders without error


def test_score_skips_queries_with_no_relevant():
    runs = {"q1": {"m": ["a"]}}
    qrels = {"q1": {"a": 1}}  # below threshold 2
    out = rev.score(runs, qrels, rel_threshold=2)
    assert out["skipped"] == ["q1"]
    assert out["methods"]["m"]["n"] == 0
