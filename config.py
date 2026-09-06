"""SCRAG configuration — the single source of every tunable in the project.

No magic numbers anywhere else in the codebase. Later phases add their
thresholds here rather than inline; that is what lets the Phase 6 ablation
runner vary the pipeline without rewriting module code.

Secrets are read from the environment only (`.env` locally, GitHub Secrets in
CI, host env vars in deploy). Nothing in this file ever holds a key value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op when .env is absent, e.g. in CI or a container

REPO_ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    """Read an override from the environment, treating empty as unset."""
    value = os.getenv(name, "").strip()
    return value or default


# --------------------------------------------------------------------------
# Module A — ingest, chunking, embedding, FAISS index
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = _env("SCRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = 384
    # Chunk sizes are measured in characters, not tokens: the splitter runs
    # before the embedding tokenizer and must not depend on it.
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    # Candidates pulled before any reranking or corrective retrieval.
    fetch_k: int = 20
    index_path: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "corpus.faiss")
    metadata_path: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "corpus.meta.json")
    normalize_embeddings: bool = True


# --------------------------------------------------------------------------
# Module B — retrieval evaluator (the contribution)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatorConfig:
    base_model: str = "microsoft/deberta-v3-small"
    # Filled in by Phase 2 once the fine-tune exists; None means fall back to
    # the untrained base model and say so loudly.
    checkpoint_path: Path | None = None
    labels: tuple[str, ...] = ("correct", "ambiguous", "wrong")
    max_length: int = 512
    # PLACEHOLDER thresholds — Phase 2 replaces these with measured values.
    correct_threshold: float = 0.70
    wrong_threshold: float = 0.30
    # Corrective retrieval fires when the top chunk grades below this.
    corrective_trigger_threshold: float = 0.50
    max_corrective_rounds: int = 1


# --------------------------------------------------------------------------
# Module C — citation-forced generation (the one paid component)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneratorConfig:
    # Exact ID string, never a date suffix. See tech-stack.md for pricing.
    model: str = _env("SCRAG_GENERATOR_MODEL", "claude-opus-5")
    max_tokens: int = 1024
    temperature: float = 0.0
    # Opus 5 takes adaptive thinking; budget_tokens is removed and 400s.
    thinking: str | None = "adaptive"
    # Cost controls (tech-stack.md). All four are on by default.
    use_prompt_caching: bool = True
    use_batch_api_for_eval: bool = True
    cache_responses_on_disk: bool = True
    response_cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "eval" / "results" / ".api_cache")
    # Citation format the Phase 3 parser must accept, e.g. "[1]".
    citation_pattern: str = r"\[(\d+)\]"
    request_timeout_s: float = 120.0
    max_retries: int = 3
    effort: str = "high"          # low | medium | high | xhigh | max
    # Opus 5 list price, USD per 1M tokens. Used to price a run from measured
    # usage rather than from an assumption.
    price_input_per_mtok: float = 5.00
    price_output_per_mtok: float = 25.00
    price_cache_write_per_mtok: float = 6.25   # 1.25x input, 5-minute TTL
    price_cache_read_per_mtok: float = 0.50    # 0.1x input
    # Opus 5 will not create a cache entry below this prefix length. Shorter
    # prefixes fail silently, so the runner reports the measured value.
    min_cacheable_prefix_tokens: int = 512


# --------------------------------------------------------------------------
# Module D — NLI citation verifier
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifierConfig:
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    max_length: int = 512
    # PLACEHOLDER — Phase 4 sets this from a measured P/R curve.
    entailment_threshold: float = 0.50
    batch_size: int = 16


# --------------------------------------------------------------------------
# Module E — repair and abstention
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RepairConfig:
    max_repair_attempts: int = 1
    # PLACEHOLDER — Phase 5 tunes these against coverage/accuracy tradeoff.
    # Abstain outright when this fraction of sentences fails verification.
    abstain_if_unsupported_fraction_above: float = 0.50
    # Abstain when no retrieved chunk clears the evaluator at all.
    abstain_on_no_correct_chunk: bool = True
    abstention_message: str = (
        "I cannot answer this from the provided sources."
    )


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    title: str = "SCRAG"
    version: str = "0.1.0"


DATA_DIR: Path = Path(_env("SCRAG_DATA_DIR", str(REPO_ROOT / "data")))

RETRIEVAL = RetrievalConfig()
EVALUATOR = EvaluatorConfig()
GENERATOR = GeneratorConfig()
VERIFIER = VerifierConfig()
REPAIR = RepairConfig()
APP = AppConfig()

# Random seed. Phases 1-6 must share one seeded split for the ablation table
# to be comparable across variants.
SEED: int = 42


def anthropic_api_key() -> str | None:
    """The Claude API key, or None if unset.

    Never logged, never written to config, never returned in an API response.
    Callers that need it should fail loudly rather than silently degrading.
    """
    return os.getenv("ANTHROPIC_API_KEY") or None


def has_api_key() -> bool:
    return anthropic_api_key() is not None


def summary() -> dict[str, object]:
    """Non-secret config summary, safe to expose on /health."""
    return {
        "embedding_model": RETRIEVAL.embedding_model,
        "generator_model": GENERATOR.model,
        "top_k": RETRIEVAL.top_k,
    }


# --------------------------------------------------------------------------
# Corpus construction (Phase 1)
#
# PopQA ships questions and answers but no document corpus, so Module A indexes
# Wikipedia lead sections for the entities the questions ask about, plus a pool
# of unrelated entities as distractors. Without distractors retrieval is
# near-trivial and the baseline flatters itself.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusConfig:
    wikipedia_api: str = "https://en.wikipedia.org/w/api.php"
    # Wikipedia asks for a descriptive UA with contact info on API clients.
    user_agent: str = "SCRAG-research/0.1 (academic RAG project)"
    titles_per_request: int = 20
    max_chars_per_doc: int = 6000
    request_timeout_s: float = 30.0
    retry_attempts: int = 5
    # Wikipedia returns 429 readily for anonymous clients; pace politely.
    backoff_base_s: float = 5.0
    delay_between_batches_s: float = 1.0
    # Entities from questions outside the eval split, indexed purely as noise.
    distractor_pool_size: int = 800
    corpus_path: Path = field(default_factory=lambda: DATA_DIR / "corpus" / "wikipedia.jsonl")


# --------------------------------------------------------------------------
# Datasets and eval splits (Phase 1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    popqa_hf_id: str = "akariasai/PopQA"
    popqa_hf_split: str = "test"      # PopQA ships one split upstream
    # Our own seeded carve-up of it. Every phase from here on uses these exact
    # ids; resampling invalidates the whole ablation.
    dev_size: int = 500
    test_size: int = 1000
    splits_dir: Path = field(default_factory=lambda: DATA_DIR / "splits")
    embedding_cache_path: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "embeddings.npy")


# --------------------------------------------------------------------------
# Evaluation harness (Phase 1 baseline; extended in Phase 6)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "eval" / "results")
    default_limit: int = 200
    # Faithfulness is measured with the RAGAS definition (claims entailed by
    # retrieved context / total claims) but our own implementation, calling
    # Claude directly. RAGAS itself hard-depends on openai + langchain, which
    # this project does not take. Report it as such - it is not RAGAS output.
    faithfulness_model: str = "claude-opus-5"
    faithfulness_max_claims: int = 12
    faithfulness_effort: str = "medium"


CORPUS = CorpusConfig()
DATASET = DatasetConfig()
EVAL = EvalConfig()
