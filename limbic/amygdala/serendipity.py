"""Serendipity — surface useful but *non-obvious* links between documents.

Relevance-ranked retrieval optimises precision; this optimises **surprise**:
pairs related enough to be meaningful yet far enough apart to be non-obvious
(the inverted-U "sweet spot"), boosted when they cross a facet (source / era /
domain) you'd never manually connect. Plus Swanson **ABC bridging** for
transitive A–C links via a shared intermediate B.

Similarity bands are embedding-space dependent — calibrate ``band`` to your model
(whitening spreads the unrelated floor; raw multilingual encoders compress high).
See ``amygdala.embed`` whitening and ``amygdala.cluster.pairwise_cosine``.
"""
from __future__ import annotations

import numpy as np


def _normalize(embeddings) -> np.ndarray:
    E = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(norms, 1e-9, None)


def inverted_u(sim: float, lo: float, hi: float) -> float:
    """Weight peaking at the band centre, falling to 0 at the edges, 0 outside.

    Encodes "related but not obvious": neither near-duplicate (sim→hi) nor
    unrelated (sim→lo) scores well; the surprising middle does.
    """
    if sim < lo or sim > hi:
        return 0.0
    center = (lo + hi) / 2
    half = (hi - lo) / 2 or 1e-9
    return 1.0 - abs(sim - center) / half


def serendipity_pairs(ids, embeddings, *, metas=None, facet_key=None,
                      band: tuple[float, float] = (0.55, 0.82),
                      facet_bonus: float = 0.2, top: int = 50,
                      neighbors: int = 20) -> list[dict]:
    """Rank surprising-yet-related document pairs.

    Args:
        ids: doc ids (len N).
        embeddings: (N, D) array.
        metas: optional list of per-doc metadata (len N), for ``facet_key``.
        facet_key: callable(meta) -> hashable facet (e.g. source type / decade).
            Pairs that *cross* the facet get ``facet_bonus`` added (more surprising).
        band: (lo, hi) cosine sweet spot. Pairs outside are ignored.
        facet_bonus: added to the inverted-U weight for cross-facet pairs.
        top: number of pairs to return.
        neighbors: per-doc cap on in-band neighbours considered (keeps it cheap).

    Returns: list of {a, b, sim, score} sorted by score desc.
    """
    E = _normalize(embeddings)
    sims = E @ E.T
    lo, hi = band
    pairs: list[tuple] = []
    n = len(ids)
    for i in range(n):
        order = np.argsort(-sims[i])
        cnt = 0
        for j in order:
            if j <= i:
                continue
            s = float(sims[i, j])
            if s > hi:
                continue
            if s < lo:
                break  # sorted desc: nothing below lo remains useful
            w = inverted_u(s, lo, hi)
            if facet_key and metas is not None and facet_key(metas[i]) != facet_key(metas[j]):
                w += facet_bonus
            pairs.append((w, ids[i], ids[j], s))
            cnt += 1
            if cnt >= neighbors:
                break
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [{"a": a, "b": b, "sim": s, "score": w} for w, a, b, s in pairs[:top]]


def abc_bridges(ids, embeddings, *, low: float = 0.4, high: float = 0.7,
                top: int = 30) -> list[dict]:
    """Swanson ABC bridging: A and C aren't directly similar (< ``low``) but both
    strongly relate (> ``high``) to some intermediate B — a transitive,
    literature-based-discovery style hidden link.

    Returns: list of {a, c, bridge_b, ac_sim, bridge_strength} sorted by strength.
    O(N^2); for large N restrict ``ids`` to a candidate subset first.
    """
    E = _normalize(embeddings)
    sims = E @ E.T
    n = len(ids)
    out: list[tuple] = []
    for i in range(n):
        for k in range(i + 1, n):
            if sims[i, k] >= low:
                continue
            both = np.minimum(sims[i], sims[k])
            both[i] = both[k] = -1.0
            b = int(np.argmax(both))
            if both[b] >= high:
                out.append((float(both[b]), ids[i], ids[k], ids[b], float(sims[i, k])))
    out.sort(key=lambda p: p[0], reverse=True)
    return [{"a": a, "c": c, "bridge_b": bb, "ac_sim": ac, "bridge_strength": bs}
            for bs, a, c, bb, ac in out[:top]]
