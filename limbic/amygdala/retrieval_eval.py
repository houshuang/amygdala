"""Retrieval evaluation — pooled qrels + IR metrics for search quality.

The measurement tool amygdala was missing: `calibrate` validates an LLM *judge*
against humans, but nothing scored *retrieval*. This adds the standard IR loop —
pool candidates from competing methods, judge graded relevance, score with
nDCG / recall / MRR / MAP — so any limbic corpus can answer "is my search good,
and which method/knob is best?" objectively.

Data shapes (plain dicts, JSON-friendly):
    runs  : {query_id: {method_name: [doc_id, ...ranked best-first]}}
    qrels : {query_id: {doc_id: grade}}        # graded relevance, 0..N
    strata: {query_id: label}                  # optional per-category breakdown

Pooled judging avoids privileging any method: the judged set is the union of
every method's top-`depth` results, so a method is never penalised for surfacing
a good document the others missed (which would otherwise be scored as irrelevant
simply because no one judged it).

Typical use::

    from limbic.amygdala import retrieval_eval as rev
    qrels = rev.judge_pool(rev.pool(runs), queries, doc_text, judge_fn)
    scores = rev.score(runs, qrels, strata=categories)
    print(rev.format_report(scores))
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------- #
# Metrics (graded relevance; `rel_threshold` binarises for recall/MRR/MAP)
# --------------------------------------------------------------------------- #


def dcg(gains: list[float]) -> float:
    """Discounted cumulative gain with the 2^g - 1 gain formulation."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranking: list[str], rel: dict[str, int], k: int = 10) -> float:
    """Normalised DCG@k. 0.0 if no graded docs exist for the query."""
    gains = [rel.get(d, 0) for d in ranking[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall(ranking: list[str], rel: dict[str, int], k: int = 20,
           rel_threshold: int = 1) -> float | None:
    """Recall@k over docs with grade >= rel_threshold. None if no relevant docs."""
    total = sum(1 for g in rel.values() if g >= rel_threshold)
    if not total:
        return None
    hit = sum(1 for d in ranking[:k] if rel.get(d, 0) >= rel_threshold)
    return hit / total


def mrr(ranking: list[str], rel: dict[str, int], k: int = 10,
        rel_threshold: int = 1) -> float:
    """Reciprocal rank of the first relevant doc in top-k (0.0 if none)."""
    for i, d in enumerate(ranking[:k]):
        if rel.get(d, 0) >= rel_threshold:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranking: list[str], rel: dict[str, int], k: int | None = None,
                      rel_threshold: int = 1) -> float | None:
    """Average precision@k. None if the query has no relevant docs."""
    total = sum(1 for g in rel.values() if g >= rel_threshold)
    if not total:
        return None
    cut = ranking[:k] if k else ranking
    hits = 0
    acc = 0.0
    for i, d in enumerate(cut):
        if rel.get(d, 0) >= rel_threshold:
            hits += 1
            acc += hits / (i + 1)
    return acc / min(total, len(cut)) if hits else 0.0


# --------------------------------------------------------------------------- #
# Pooling + judging
# --------------------------------------------------------------------------- #


def pool(runs: dict[str, dict[str, list[str]]], depth: int = 10) -> dict[str, set[str]]:
    """Union of every method's top-`depth` results, per query."""
    out: dict[str, set[str]] = {}
    for qid, methods in runs.items():
        s: set[str] = set()
        for ranking in methods.values():
            s.update(ranking[:depth])
        out[qid] = s
    return out


def judge_pool(pooled: dict[str, set[str]], queries: dict[str, str],
               doc_text, judge_fn, *, max_workers: int = 8,
               existing: dict[str, dict[str, int]] | None = None,
               progress=None) -> dict[str, dict[str, int]]:
    """Judge every (query, doc) in the pool, returning qrels.

    Args:
        pooled: {query_id: set(doc_id)} from `pool()`.
        queries: {query_id: query_text}.
        doc_text: callable(doc_id) -> str (document content for the judge).
        judge_fn: callable(query_text, doc_text) -> int grade. Inject any backend
            (LLM, human, heuristic). See `make_llm_judge` for a default.
        existing: prior qrels to extend; only unjudged pairs are (re)judged.
        progress: optional callable(done, total).

    Incremental: pairs already present in `existing` are skipped, so judging is
    resumable and cheap to extend when new methods enlarge the pool.
    """
    qrels: dict[str, dict[str, int]] = {q: dict(v) for q, v in (existing or {}).items()}
    work = [(qid, did) for qid, docs in pooled.items() for did in docs
            if did not in qrels.get(qid, {})]
    total = len(work)
    if not total:
        return qrels

    def _one(pair):
        qid, did = pair
        try:
            g = int(judge_fn(queries[qid], doc_text(did)))
        except Exception:
            return qid, did, None
        return qid, did, g

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for qid, did, g in ex.map(_one, work):
            done += 1
            if g is not None:
                qrels.setdefault(qid, {})[did] = g
            if progress:
                progress(done, total)
    return qrels


def make_llm_judge(*, model: str = "gemini3-flash", max_grade: int = 3,
                   system: str | None = None):
    """Default graded-relevance judge backed by `limbic.amygdala.llm`.

    Returns a judge_fn(query, doc) -> int in [0, max_grade]. Requires the `[llm]`
    extra. Callers wanting a different backend (Codex, human) just pass their own
    judge_fn to `judge_pool` instead.
    """
    from .llm import generate_structured_sync

    sys = system or (
        "You are a strict relevance judge for a search engine. Given a QUERY and "
        f"a DOCUMENT, grade relevance 0..{max_grade} where {max_grade} = directly "
        "on-topic/primary source and 0 = not relevant. Judge the document's actual "
        "content, not keyword overlap.")
    schema = {"type": "object",
              "properties": {"grade": {"type": "integer"}, "reason": {"type": "string"}},
              "required": ["grade"]}

    def judge_fn(query: str, doc: str) -> int:
        res, _ = generate_structured_sync(
            prompt=f"QUERY: {query}\n\nDOCUMENT:\n{doc}", schema=schema,
            system_prompt=sys, model=model)
        return max(0, min(max_grade, int(res.get("grade", 0))))

    return judge_fn


# --------------------------------------------------------------------------- #
# Scoring + reporting
# --------------------------------------------------------------------------- #


def score(runs: dict[str, dict[str, list[str]]], qrels: dict[str, dict[str, int]], *,
          k_ndcg: int = 10, k_recall: int = 20, rel_threshold: int = 2,
          strata: dict[str, str] | None = None) -> dict:
    """Score every method over all judged queries.

    Returns {"methods": {method: {ndcg, recall, mrr, map, n}},
             "by_stratum": {method: {stratum: ndcg}},   # if strata given
             "skipped": [query_ids with no relevant docs in the pool]}.
    """
    methods = sorted({m for q in runs.values() for m in q})
    acc = {m: {"ndcg": [], "recall": [], "mrr": [], "map": []} for m in methods}
    bystrat = {m: {} for m in methods}
    skipped = []

    for qid, mrankings in runs.items():
        rel = qrels.get(qid, {})
        if not any(g >= rel_threshold for g in rel.values()):
            skipped.append(qid)
            continue
        for m in methods:
            r = mrankings.get(m, [])
            nd = ndcg(r, rel, k_ndcg)
            acc[m]["ndcg"].append(nd)
            rc = recall(r, rel, k_recall, rel_threshold)
            if rc is not None:
                acc[m]["recall"].append(rc)
            acc[m]["mrr"].append(mrr(r, rel, k_ndcg, rel_threshold))
            ap = average_precision(r, rel, k_recall, rel_threshold)
            if ap is not None:
                acc[m]["map"].append(ap)
            if strata and qid in strata:
                bystrat[m].setdefault(strata[qid], []).append(nd)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out = {"methods": {m: {k: mean(v) for k, v in acc[m].items()} for m in methods},
           "skipped": skipped}
    for m in methods:
        out["methods"][m]["n"] = len(acc[m]["ndcg"])
    if strata:
        out["by_stratum"] = {m: {s: mean(v) for s, v in bystrat[m].items()}
                             for m in methods}
    return out


def format_report(scores: dict, *, sort_by: str = "ndcg") -> str:
    """Render `score()` output as an aligned text table."""
    methods = scores["methods"]
    order = sorted(methods, key=lambda m: methods[m].get(sort_by, 0), reverse=True)
    lines = [f"{'method':18} {'nDCG':>7} {'Recall':>7} {'MRR':>7} {'MAP':>7} {'n':>4}",
             "-" * 54]
    for m in order:
        s = methods[m]
        lines.append(f"{m:18} {s['ndcg']:7.3f} {s['recall']:7.3f} {s['mrr']:7.3f} "
                     f"{s['map']:7.3f} {s['n']:4d}")
    if "by_stratum" in scores:
        strata = sorted({s for m in scores["by_stratum"].values() for s in m})
        lines += ["", f"{'method (nDCG by stratum)':18} " +
                  " ".join(f"{s[:7]:>7}" for s in strata)]
        for m in order:
            row = scores["by_stratum"].get(m, {})
            lines.append(f"{m:18} " + " ".join(f"{row.get(s, 0):7.3f}" for s in strata))
    if scores.get("skipped"):
        lines.append(f"\nskipped (0 relevant in pool): {', '.join(scores['skipped'])}")
    return "\n".join(lines)
