"""Tests for amygdala.document_similarity — document-level similarity."""

import numpy as np
import pytest

from amygdala import EmbeddingModel
from amygdala.document_similarity import (
    Document,
    SimilarityPair,
    find_similar_documents,
    embed_documents,
    document_similarity_matrix,
)


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def sample_docs():
    """Documents with known overlap structure."""
    return [
        Document(
            id="sicily_history",
            texts={
                "summary": "Sicily's history spans from prehistoric settlements through Greek colonization, Roman rule, and Byzantine governance.",
                "claims": "Greeks founded Syracuse in 734 BC. Romans made Sicily their first province in 241 BC.",
            },
        ),
        Document(
            id="sicily_culture",
            texts={
                "summary": "Sicilian culture encompasses Baroque architecture, the invention of the sonnet, and a distinct bilingual identity.",
                "claims": "Giacomo da Lentini invented the sonnet at Frederick II's court. Sicilian Baroque is a UNESCO World Heritage style.",
            },
        ),
        Document(
            id="sicily_overview",
            texts={
                "summary": "Sicily is the largest Mediterranean island with a rich history spanning Greek, Roman, Norman, and modern Italian periods.",
                "claims": "Sicily has 5 million inhabitants. It became an autonomous Italian region after World War II.",
            },
        ),
        Document(
            id="agent_patterns",
            texts={
                "summary": "Effective agentic systems prioritize simple, composable patterns over complex frameworks for building LLM agents.",
                "claims": "Simple patterns outperform complex frameworks. Agents trade latency for performance.",
            },
        ),
        Document(
            id="agent_tools",
            texts={
                "summary": "AI coding agents can effectively work with unfamiliar tools by reading documentation at runtime via large context windows.",
                "claims": "Large context windows enable agents to use niche libraries. Skills mechanisms facilitate agent-tool integration.",
            },
        ),
        Document(
            id="python_async",
            texts={
                "summary": "Python's asyncio provides concurrent execution for I/O-bound tasks using event loops and coroutines.",
                "claims": "Asyncio uses cooperative multitasking. Event loops manage coroutine scheduling.",
            },
        ),
    ]


class TestFindSimilarDocuments:
    def test_basic_similarity(self, sample_docs, model):
        pairs = find_similar_documents(
            sample_docs, text_fields="summary", model=model, threshold=0.3,
        )
        assert len(pairs) > 0
        assert all(isinstance(p, SimilarityPair) for p in pairs)
        assert pairs[0].score >= pairs[-1].score  # sorted descending

    def test_sicily_cluster(self, sample_docs, model):
        """Sicily articles should be more similar to each other than to agent articles."""
        pairs = find_similar_documents(
            sample_docs, text_fields="summary", model=model, threshold=0.0,
        )
        pair_map = {}
        for p in pairs:
            pair_map[(p.id_a, p.id_b)] = p.score
            pair_map[(p.id_b, p.id_a)] = p.score

        sicily_sim = pair_map.get(("sicily_history", "sicily_overview"), 0)
        cross_sim = pair_map.get(("sicily_history", "python_async"), 0)
        assert sicily_sim > cross_sim + 0.1, \
            f"Sicily pair ({sicily_sim:.3f}) should be much more similar than cross-domain ({cross_sim:.3f})"

    def test_weighted_multi_field(self, sample_docs, model):
        """Weighted combination should work without errors."""
        pairs = find_similar_documents(
            sample_docs,
            text_fields={"summary": 0.5, "claims": 0.5},
            model=model,
            threshold=0.3,
        )
        assert len(pairs) > 0
        # Should have field_scores when multiple fields
        for p in pairs:
            assert "summary" in p.field_scores
            assert "claims" in p.field_scores

    def test_threshold_filtering(self, sample_docs, model):
        low = find_similar_documents(
            sample_docs, text_fields="summary", model=model, threshold=0.2,
        )
        high = find_similar_documents(
            sample_docs, text_fields="summary", model=model, threshold=0.6,
        )
        assert len(low) >= len(high)
        assert all(p.score >= 0.6 for p in high)

    def test_max_pairs(self, sample_docs, model):
        pairs = find_similar_documents(
            sample_docs, text_fields="summary", model=model,
            threshold=0.0, max_pairs=3,
        )
        assert len(pairs) <= 3

    def test_empty_input(self, model):
        assert find_similar_documents([], model=model) == []
        assert find_similar_documents(
            [Document(id="a", texts={"summary": "hello"})], model=model,
        ) == []

    def test_missing_field_fallback(self, model):
        """Documents missing the target field should fall back to first available."""
        docs = [
            Document(id="a", texts={"summary": "Sicily history", "claims": "Greeks founded Syracuse"}),
            Document(id="b", texts={"other": "Sicily culture and architecture"}),  # no "summary"
        ]
        pairs = find_similar_documents(
            docs, text_fields="summary", model=model, threshold=0.0,
        )
        assert len(pairs) == 1  # should still compare


class TestEmbedDocuments:
    def test_basic_embedding(self, sample_docs, model):
        ids, embs = embed_documents(
            sample_docs, text_fields="summary", model=model,
        )
        assert len(ids) == len(sample_docs)
        assert embs.shape == (len(sample_docs), model.dim)
        # Should be normalized
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_weighted_embedding(self, sample_docs, model):
        ids, embs = embed_documents(
            sample_docs, text_fields={"summary": 0.5, "claims": 0.5}, model=model,
        )
        assert embs.shape == (len(sample_docs), model.dim)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_weighted_differs_from_single(self, sample_docs, model):
        """Weighted combo should produce different vectors than single field."""
        _, embs_single = embed_documents(
            sample_docs, text_fields="summary", model=model,
        )
        _, embs_weighted = embed_documents(
            sample_docs, text_fields={"summary": 0.5, "claims": 0.5}, model=model,
        )
        # They should be different (not identical)
        diff = np.abs(embs_single - embs_weighted).max()
        assert diff > 0.01, "Weighted should differ from single-field"


class TestDocumentSimilarityMatrix:
    def test_matrix_shape(self, sample_docs, model):
        ids, matrix = document_similarity_matrix(
            sample_docs, text_fields="summary", model=model,
        )
        n = len(sample_docs)
        assert len(ids) == n
        assert matrix.shape == (n, n)
        # Diagonal should be ~1.0
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)
        # Should be symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)
