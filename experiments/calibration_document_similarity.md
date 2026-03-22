# Experiment 22: Document-Level Similarity

**Date**: 2026-03-21
**Status**: Complete — module implemented and tested

## Hypothesis

Embedding article summaries with MiniLM can detect thematic overlap between documents well enough to power a "similar articles" feature, and combining multiple text fields improves over single-field embedding.

## Background

Petrarca's claim-level similarity (cosine on atomic claims) had a 70% false positive rate for detecting article-level thematic overlap — claims from different articles in the same domain (e.g., AI agents) shared vocabulary without sharing meaning. We needed a document-level signal.

## Method

### Ground truth collection
1. **18 human-rated pairs** — user rated article pairs as high/some/none overlap using a custom calibration tool (`claim-calibration.html`)
2. **300 LLM-rated pairs** — stratified sample across cosine bands, rated by Gemini 3.1 Flash Lite on 0-4 scale
3. **50 synthetic pairs** — controlled overlap levels (near-duplicate, shared theme, same field, cross-domain, unrelated)

### Strategies tested (11 total)
1. full_summary embedding only
2. one_line_summary only
3. title + one_line_summary
4. full_summary + key_claims concatenated
5. key_claims only
6. interest_topics as text
7. title + summary + topics
8. **Weighted 0.5×summary + 0.5×claims** (separate embeddings, weighted combination)
9. Max-sim of key_claims (top-1, top-3, top-5)

Also tested:
- LLM direct judgment (Gemini Flash Lite, 0-4 scale)
- Two-stage: embed filter → LLM judge
- Ensemble: cosine + topic Jaccard, cosine + content_type bonus, cosine + entity overlap

## Results

### Embedding strategies (18 human-rated pairs)

| Strategy | Accuracy | Spearman ρ | p-value |
|---|---|---|---|
| **8. Weighted (0.5 summary + 0.5 claims)** | **94.4%** | **0.818** | **<0.001** |
| 4. full_summary + claims concat | 88.9% | 0.695 | 0.001 |
| 1. full_summary only | 88.9% | 0.654 | 0.003 |
| 5. key_claims only | 88.9% | 0.629 | 0.005 |
| 9. max-sim claims top1 | 77.8% | 0.503 | 0.034 |
| 7. title + summary + topics | 72.2% | 0.450 | 0.061 |
| 2. one_line_summary only | 72.2% | 0.347 | 0.158 |
| 3. title + one_line | 72.2% | 0.327 | 0.185 |
| 6. topics as text | 72.2% | -0.020 | 0.936 |

### Scale validation (300 LLM-rated pairs)

Using full_summary embedding (the production baseline before weighted was discovered):
- AUROC = 0.930
- Spearman ρ = 0.762
- Best F1 = 0.831 at threshold 0.64
- Pearson r = 0.822

### Synthetic benchmark (50 pairs)

| Category | Expected score | Mean cosine | Spearman ρ |
|---|---|---|---|
| near_duplicate | 4.0 | 0.686 | — |
| shared_theme | 2.5 | 0.581 | — |
| cross_domain | 1.5 | 0.390 | — |
| same_field | 1.0 | 0.339 | — |
| unrelated | 0.0 | 0.128 | — |
| **Overall** | — | — | **0.895** |

### What didn't work

- **LLM judge**: 78% accuracy (worse than embedding). Systematically over-rates within-domain similarity.
- **Two-stage pipeline**: doesn't beat embedding alone. LLM re-ranking hurts.
- **Topic Jaccard**: 50% accuracy — no better than random.
- **Topics as embeddings**: ρ = -0.020 — anti-correlated.

### Corpus analysis (257 articles)

- 26 natural clusters (HDBSCAN), 119 singletons (46%)
- 70% of articles have 10+ neighbors at threshold 0.47
- Dense overlap in domain-focused clusters (Sicily: 0.75-0.88 cohesion)
- Isolated tech articles form smaller clusters with higher novelty

## Key insights

1. **Weighted > concatenated > single-field.** Embedding summary and claims separately then combining preserves distinct signal geometry. Concatenation lets the longer text dominate the embedding.

2. **The 0.5/0.5 weight is robust.** Any split from 35/65 to 65/35 achieves 94.4% accuracy.

3. **Cosine similarity is a vocabulary matcher at claim level but a theme matcher at summary level.** The same MiniLM model that fails at claim-level thematic matching (70% false positive) works well at summary level (94% accuracy). Summaries are the right abstraction.

4. **LLMs over-estimate similarity.** The LLM sees "both about AI agents" as overlap; humans distinguish between different kinds of agent articles. Embedding is more discriminating within a domain.

## Implementation

Module: `amygdala/document_similarity.py`
- `Document` dataclass with `texts: dict[str, str]`
- `find_similar_documents()` — weighted multi-field embedding + pairwise cosine
- `embed_documents()` — returns embeddings for custom use
- `document_similarity_matrix()` — full NxN matrix

Tests: `tests/test_document_similarity.py` — 11 tests

Ground truth datasets (in Petrarca `scripts/ground-truth/`):
- `article_similarity_ratings.json` — 300 LLM-rated pairs
- `synthetic_benchmark.json` — 50 controlled pairs
- `embedding_strategy_comparison.json` — all strategy results
- `corpus_analysis.json` — cluster analysis of 257 articles
- `threshold_config.json` — calibrated thresholds
