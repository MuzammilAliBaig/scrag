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
