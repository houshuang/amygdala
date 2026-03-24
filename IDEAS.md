# Limbic - Ideas and Future Directions

## Active Research

- Document similarity calibration: threshold tuning across more domains (currently calibrated on news articles)
- Knowledge map: test on real educational graphs (AL-CPL 586 concepts, Metacademy 141 concepts — see research/tutorial_platform_graphs.md)
- Calibration metrics: validating LLM judges against human gold labels at scale

## Recently Completed

- **Knowledge map comprehensive experiment** (2026-03-23): Full matrix of propagator × strategy × topology × noise. Key findings: Bayesian propagator 42% faster convergence than heuristic on chains; EIG and entropy tie at aggregate; batch probing works efficiently; post-hoc foil calibration doesn't help (Bayesian constraint propagation is the primary overclaiming defense). Default propagator switched to "bayesian". Added `next_probe_batch(n)`.
- **Knowledge map: Bayesian belief propagation** — replaced pgmpy with from-scratch implementation, optimized CPD parameters via 180-config grid sweep
- **Knowledge map: EIG probe selection, foil calibration, KST fringes, noisy observation mode** — all implemented and tested

## Hippocampus Ideas

- **Proposal UI**: Web-based review interface for pending proposals. Could be extracted from kulturperler's `/dev/proposals` into a reusable component.
- **Semantic dedup**: Use amygdala embeddings as a VetoGate. Embed entity names, compute cosine similarity, reject pairs below threshold. Would complement the existing string-distance approach.
- **Schema-aware validation**: Auto-generate Rule sets from JSON Schema definitions, dataclass annotations, or YAML schema files. Reduces boilerplate when onboarding new entity types.
- **Multi-file atomic transactions**: Apply multiple proposals as one atomic operation with rollback on failure. Currently each proposal applies independently.
- **Conflict detection**: Warn when two pending proposals touch the same entity or would produce contradictory states.
- **Proposal templates**: Pre-built proposal patterns for common operations (bulk rename, field migration, schema upgrade).
- **Undo log**: Record applied proposals in a format that supports reversal, beyond the current backup mechanism.

## Cerebellum Ideas

- **Cost model library**: Pre-calibrated cost estimates for common LLM providers (Gemini Flash, Claude Sonnet, GPT-4o, etc.) so budget planning works out of the box.
- **Parallel tier execution**: Run tier 1 on the current batch while tier 2 processes the previous batch's escalated items. Currently tiers run sequentially.
- **Webhook notifications**: Notify via HTTP callback on budget warnings, batch completion, or error thresholds.
- **Dashboard**: Real-time status page for running audits. Could be a simple terminal UI or a lightweight web page.
- **Retry strategies**: Exponential backoff and circuit breaker patterns for LLM API errors. Currently a batch-level try/catch marks all items as errors.
- **Dry-run mode**: Simulate a full audit run without calling LLM APIs, using cached or mocked responses. Useful for testing orchestrator configuration.
- **Priority queues**: Process high-value or recently-changed items first within a batch.

## Amygdala Ideas

- **Embedding model zoo**: Pre-tested configurations for common domains -- medical (PubMedBERT), legal, Nordic languages (NbAiLab models). Include recommended whitening settings and threshold calibrations.
- **Streaming novelty**: Process items as they arrive without rebuilding the full VectorIndex. The IncrementalCentroidCluster already supports this pattern; novelty scoring could too.
- **GPU acceleration**: Optional FAISS backend for corpora beyond 100K vectors, where brute-force numpy starts to lag. Keep numpy as the default for simplicity.
- **Cross-encoder fine-tuning**: Domain-specific reranker training using the eval harness and calibration data already in the experiments directory.
- **Embedding drift detection**: Track corpus centroid shift over time and alert when the distribution changes significantly (new topic cluster appearing, data quality regression).
- **Whitening auto-calibration**: Detect whether a corpus is domain-homogeneous or diverse and automatically choose whitening mode. Currently requires manual judgment.
- **Matryoshka cascading search**: Use low-dim embeddings for fast first-pass retrieval, then full-dim for reranking. Trades storage for latency.

## Cross-Package Integration Ideas

- **amygdala.dedup + hippocampus.dedup**: Use embedding similarity as a VetoGate input. Candidate pairs from fuzzy string matching get an embedding cosine check before proceeding to merge proposal creation.
- **cerebellum audit -> hippocampus proposals**: Audit findings (flagged items, suggested corrections) automatically create pending proposals for human review. Closes the loop between verification and action.
- **End-to-end pipeline template**: Import -> embed -> dedup -> propose -> review -> apply -> audit. A reference implementation showing how all three packages compose for a complete data curation workflow.
- **Shared eval harness**: Use amygdala's calibrate module (cohens_kappa, validate_llm_judge) to evaluate cerebellum's LLM audit accuracy, creating a feedback loop for tier configuration.
- **Knowledge map for audit prioritization**: Use knowledge_map's entropy-maximizing probe selection to decide which entities to audit next, instead of simple sequential ordering.
