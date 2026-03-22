# Document Similarity Module — Design Document

**Date**: 2026-03-21
**Status**: Proposal — no code written yet
**Author**: Research session, Petrarca project

## Motivation

Amygdala has strong primitives for **claim-level** similarity (pairwise_cosine, extract_pairs, classify_pairs) and **item-level** novelty scoring (novelty_score). But there is no module for **document-level** similarity — the task of finding which documents in a corpus are thematically related to each other.

Petrarca discovered through calibration (18 human-rated article pairs, 2026-03-21) that:

- Embedding `full_summary` text with MiniLM achieves **83% accuracy** for detecting article-level thematic overlap (Spearman rho=0.654)
- Individual claim-level similarity has **70% false positive rate** for thematic matching — claims share vocabulary without implying documents overlap
- LLM-as-judge achieves the same 83% accuracy but costs an API call per pair
- Interest topic Jaccard overlap is useless (50% accuracy)
- A **two-stage approach** (embed summaries for candidates, LLM-judge top pairs) could combine the best of both

This pattern — "find similar documents in a corpus using text representation embeddings, optionally refined by an LLM judge" — is useful beyond Petrarca. Any project with a document corpus (articles, learning materials, research papers, book chapters) could benefit.

## Where It Fits in Amygdala

Current module map:

| Module | Granularity | Purpose |
|--------|-------------|---------|
| `embed.py` | Text → vector | Embedding with caching, whitening, genericization |
| `search.py` | Query → ranked results | VectorIndex, FTS5, hybrid search, reranking |
| `novelty.py` | Item vs corpus | Novelty scoring, NLI classification |
| `cluster.py` | Items → groups | Clustering, pairwise cosine, pair extraction |
| `knowledge_map.py` | Graph → probes | Bayesian knowledge mapping |
| `llm.py` | Prompt → text/JSON | Multi-provider LLM client |
| `cache.py` | Text → cached embedding | Persistent SQLite embedding cache |
| `index.py` | Documents → SQLite | Document/chunk storage with search |
| **`document_similarity.py`** | **Documents → similar pairs** | **NEW: document-level similarity with two-stage pipeline** |

The new module sits between `cluster.py` (which works on raw embeddings) and project-specific code (which currently has to orchestrate embedding, similarity computation, and optional LLM refinement itself).

### Relationship to Existing Modules

- **Uses** `embed.py` (EmbeddingModel) for text embedding
- **Uses** `cluster.py` (pairwise_cosine, extract_pairs) for similarity computation
- **Optionally uses** `llm.py` (generate_structured) for LLM re-ranking
- **Does NOT overlap with** `novelty.py` (novelty scores an item against a corpus; document_similarity finds pairs within a corpus)
- **Does NOT overlap with** `search.py` (search is query-driven; document_similarity is corpus-wide pairwise)

## Proposed API

### Core Types

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    """A document with an ID and text fields to embed."""
    id: str
    texts: dict[str, str]  # field_name -> text content
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SimilarityPair:
    """A pair of documents with similarity score and optional judgment."""
    id_a: str
    id_b: str
    score: float               # cosine similarity of primary text field
    field_scores: dict[str, float] = field(default_factory=dict)  # per-field scores
    llm_judgment: str | None = None    # e.g. "related", "unrelated", "duplicate"
    llm_confidence: float | None = None
```

### Main Function

```python
def find_similar_documents(
    documents: list[Document],
    *,
    # Text representation
    text_field: str = "summary",          # which field in Document.texts to embed
    model: EmbeddingModel | None = None,  # reuse an existing model, or create default

    # Thresholds
    threshold: float = 0.40,              # minimum cosine to return

    # Optional LLM re-ranking (stage 2)
    llm_rerank: bool = False,
    llm_model: str = "gemini3-flash",     # amygdala.llm model key
    llm_threshold: float = 0.50,          # only send pairs above this to LLM
    llm_max_pairs: int = 100,             # cap on LLM calls
    llm_prompt: str | None = None,        # custom prompt template (or use default)

    # Output control
    max_pairs: int | None = None,         # limit returned pairs
) -> list[SimilarityPair]:
    """Find similar document pairs in a corpus using embedding similarity.

    Stage 1: Embed the specified text field for each document, compute
    pairwise cosine similarity, return pairs above threshold.

    Stage 2 (optional): For top pairs above llm_threshold, ask an LLM
    to judge whether the documents are truly thematically related.
    Pairs are returned with llm_judgment and llm_confidence populated.

    Returns pairs sorted by score descending.
    """
```

### Embedding-Only Helper

For cases where the caller wants embeddings back (e.g., to store them, to use them in other computations):

```python
def embed_documents(
    documents: list[Document],
    *,
    text_field: str = "summary",
    model: EmbeddingModel | None = None,
) -> tuple[list[str], np.ndarray]:
    """Embed a text field from each document.

    Returns:
        (ids, embeddings) where ids[i] corresponds to embeddings[i].
        Documents missing the specified text_field are skipped.
    """
```

### Similarity Matrix Helper

For cases where the caller needs the full matrix (e.g., for clustering or visualization):

```python
def document_similarity_matrix(
    documents: list[Document],
    *,
    text_field: str = "summary",
    model: EmbeddingModel | None = None,
) -> tuple[list[str], np.ndarray]:
    """Compute full pairwise similarity matrix for documents.

    Returns:
        (ids, sim_matrix) where sim_matrix[i][j] is the cosine
        similarity between documents ids[i] and ids[j].
    """
```

### Default LLM Prompt

```python
DEFAULT_JUDGE_PROMPT = """Given two document summaries, judge their thematic relationship.

Document A: {summary_a}
Document B: {summary_b}

Rate the thematic overlap:
- "strong": These documents cover substantially the same topic or theme
- "moderate": These documents share a significant subtopic or theme
- "weak": These documents are only loosely related
- "none": These documents are about different things

Respond with JSON: {{"judgment": "strong|moderate|weak|none", "confidence": 0.0-1.0}}"""
```

The prompt template uses `{summary_a}` and `{summary_b}` placeholders which are filled from the document's `text_field`. Callers can override the entire prompt for domain-specific judgment criteria.

## Configuration Patterns

### Minimal Usage (Petrarca)

```python
from amygdala.document_similarity import find_similar_documents, Document

docs = [
    Document(id=a["id"], texts={"summary": a["full_summary"]})
    for a in articles if a.get("full_summary")
]
pairs = find_similar_documents(docs, threshold=0.40)
```

This replaces the 50-line `compute_article_summary_similarities()` function currently in Petrarca's `build_knowledge_index.py`.

### With LLM Re-ranking (Petrarca, future)

```python
pairs = find_similar_documents(
    docs,
    threshold=0.40,
    llm_rerank=True,
    llm_threshold=0.55,   # only judge the top candidates
    llm_max_pairs=50,     # budget cap
)
# pairs now have llm_judgment populated for top-scoring entries
strong_pairs = [p for p in pairs if p.llm_judgment in ("strong", "moderate")]
```

### Multi-Field Comparison (Alif, hypothetical)

```python
docs = [
    Document(
        id=lesson["id"],
        texts={
            "summary": lesson["description"],
            "objectives": " ".join(lesson["learning_objectives"]),
        }
    )
    for lesson in lessons
]
# Compare by learning objectives instead of summary
pairs = find_similar_documents(docs, text_field="objectives")
```

### Custom LLM Judgment (research use)

```python
custom_prompt = """Are these two articles about the same historical event or period?
Article A: {summary_a}
Article B: {summary_b}
Respond with JSON: {{"judgment": "same_event|same_period|different", "confidence": 0.0-1.0}}"""

pairs = find_similar_documents(
    docs,
    llm_rerank=True,
    llm_prompt=custom_prompt,
    llm_model="sonnet",  # use Claude for nuanced judgment
)
```

## Design Decisions

### Why `Document` Instead of Raw Dicts

A dataclass provides:
1. Clear contract — callers know exactly what to provide
2. IDE autocomplete and type checking
3. The `texts` dict allows multiple text fields without a rigid schema
4. The `metadata` dict carries through to results without the module needing to understand it

### Why a Single `text_field` Parameter Instead of Multi-Field Fusion

The calibration data shows that summary embedding alone achieves 83% accuracy. Combining multiple fields (summary + claims + topics) is a research question, not an established technique. The module should do one thing well first. Multi-field fusion (weighted average of per-field embeddings, or concatenated text) can be added later as a `text_fields` parameter with weights.

If callers want multi-field scoring, they can:
1. Call `find_similar_documents` multiple times with different `text_field` values
2. Use `field_scores` in the result to compare
3. Concatenate fields into a single text before passing

### Why Optional LLM Re-ranking Instead of Always-On

LLM calls are expensive (time and money). The two-stage pattern lets callers choose their accuracy/cost tradeoff:
- Stage 1 only: free, fast, 83% accurate (for Petrarca's use case)
- Stage 1 + 2: better accuracy, but costs API calls proportional to pairs above `llm_threshold`

The `llm_max_pairs` cap prevents runaway costs on large corpora.

### Why Not Build on `classify_pairs` / NLI

`classify_pairs` uses NLI cross-encoders to distinguish KNOWN/EXTENDS/NEW at the **claim level**. Document-level similarity is a different task: we want to know "are these documents about the same topic?" not "does document A entail document B?" NLI is the wrong frame for document-level thematic matching.

The LLM judge approach is more flexible — the prompt can be customized for the specific judgment task (thematic overlap, topical similarity, coverage overlap, etc.).

### Threshold Semantics

The `threshold` parameter has different meanings depending on corpus and text representation. The module stores the raw cosine score, making no claim about what constitutes "similar" — that judgment lives in the caller.

From Petrarca's calibration:
- Summary cosine >= 0.55: likely thematically related (precision ~80%)
- Summary cosine >= 0.47: optimal accuracy/recall balance (83% accuracy)
- Summary cosine >= 0.40: cast a wide net, accept more noise

These numbers are specific to Petrarca's article summaries (50-200 word LLM-generated summaries of news/research articles). Other corpora will have different distributions.

## Calibration Data Format

To enable systematic threshold tuning across projects, the module should define a standard format for ground-truth calibration data. This allows reuse of Petrarca's labeled pairs and makes it easy for other projects to collect their own.

### Proposed Format

```json
{
  "metadata": {
    "project": "petrarca",
    "date": "2026-03-21",
    "corpus_description": "257 articles (Sicilian history, AI/tech, mixed)",
    "text_field": "full_summary",
    "n_documents": 257,
    "n_pairs_rated": 18,
    "annotator": "stian"
  },
  "pairs": [
    {
      "id_a": "article_id_1",
      "id_b": "article_id_2",
      "title_a": "Article Title 1",
      "title_b": "Article Title 2",
      "cosine_score": 0.623,
      "human_rating": 4,
      "human_label": "strong",
      "notes": "Both about Norman conquest of Sicily"
    }
  ],
  "rating_scale": {
    "description": "1-5 thematic overlap",
    "1": "Unrelated topics",
    "2": "Same broad domain only",
    "3": "Share a significant subtopic",
    "4": "Cover substantially overlapping themes",
    "5": "Near-duplicate topic coverage"
  }
}
```

### Calibration Script

A companion calibration script (not part of the library, but in `experiments/`) would:
1. Take a corpus and run `find_similar_documents`
2. Sample pairs from different score bands (stratified)
3. Present them to a human for rating (simple HTML UI, like Petrarca's `feedback-calibration.html`)
4. Compute accuracy metrics (Spearman correlation, accuracy at threshold, precision/recall curves)
5. Output the calibration data file

This follows the proven methodology from `calibration_petrarca_thresholds.md`: sample ~15 pairs per cosine band, collect human labels, find the accuracy cliff.

## Testing Strategy

### Unit Tests (`tests/test_document_similarity.py`)

1. **Basic pair finding**: 5 documents with known relationships, verify correct pairs returned
2. **Threshold filtering**: Verify pairs below threshold are excluded
3. **Empty/single document**: Edge cases return empty list
4. **Missing text field**: Documents without the specified field are silently skipped
5. **Score ordering**: Results are sorted by score descending
6. **max_pairs**: Verify output truncation
7. **embed_documents**: Verify returned IDs match embeddings shape
8. **document_similarity_matrix**: Verify matrix is symmetric, diagonal is ~1.0

### Integration Tests (require model loading)

Use the same `scope="module"` fixture pattern as existing tests to load the model once:

```python
@pytest.fixture(scope="module")
def history_docs():
    return [
        Document(id="rome", texts={"summary": "The Roman Empire conquered the Mediterranean..."}),
        Document(id="greece", texts={"summary": "Ancient Greek city-states developed democracy..."}),
        Document(id="sicily", texts={"summary": "Sicily was conquered by Romans, Greeks, and Normans..."}),
        Document(id="python", texts={"summary": "Python is a programming language for data science..."}),
    ]
```

Expected: rome-sicily and greece-sicily should have higher scores than any pair involving python.

### LLM Re-ranking Tests

Mock the LLM call to avoid API dependencies in CI:

```python
def test_llm_rerank_populates_judgment(monkeypatch):
    # Mock amygdala.llm.generate_structured to return a canned response
    ...
```

## What NOT to Include

To keep the module focused, these are explicitly out of scope:

1. **Claim-level similarity** — already handled by `classify_pairs` in `novelty.py`
2. **Document clustering** — use `cluster.py` on the embeddings from `embed_documents`
3. **Incremental updates** — the module is stateless; caller manages what documents to include
4. **Persistence** — no database; caller stores results however they want
5. **Multi-field embedding fusion** — single field only in v1; caller concatenates if needed
6. **Visualization** — no graph/chart output; caller can use the matrix/pairs for visualization
7. **NLI cross-encoder for documents** — wrong abstraction for document-level thematic matching
8. **Topic extraction** — out of scope; caller provides whatever text fields they want

## Migration Path for Petrarca

After the module is implemented:

1. **Replace** `compute_article_summary_similarities()` in `build_knowledge_index.py` with a call to `find_similar_documents`
2. **Remove** the inline EmbeddingModel instantiation, normalization, and manual pairwise loop (lines 260-309)
3. **Adapt** the output format: `SimilarityPair` has `id_a`/`id_b`/`score` which maps directly to Petrarca's `{"a": ..., "b": ..., "score": ...}` format
4. **Optionally** enable LLM re-ranking to improve accuracy beyond 83% for the "related articles" feature

The existing claim-level pipeline (`pairwise_cosine` + `extract_pairs` + `classify_pairs`) remains unchanged — this module is additive.

## Open Questions

1. **Should `Document` be a protocol instead of a dataclass?** A Protocol with `id: str` and `texts: dict[str, str]` would let callers pass their own objects without wrapping. But dataclasses are simpler and more explicit.

2. **Should the LLM judge be async?** `amygdala.llm.generate_structured` is async. The document_similarity module could offer both `find_similar_documents` (sync, using `asyncio.run`) and `find_similar_documents_async`. Or it could just be sync and call `generate_structured_sync` internally.

3. **Should we support pre-computed embeddings?** If a caller already has embeddings (like Petrarca's claim_embeddings.npz), they could pass them directly instead of re-embedding. This would be a separate function or an `embeddings` parameter on `find_similar_documents`. Probably not needed in v1 — the embedding cache handles repeat calls efficiently.

4. **Batch size for LLM re-ranking**: Should pairs be sent to the LLM one at a time or in batches? Single-pair prompts are simpler and more reliable. Batch prompts save API calls but risk confusion and lower accuracy. Start with single-pair.
