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
    # Vectors are persisted beside the index so replacing a document never
    # re-embeds the documents that survived. Re-embedding survivors is O(corpus)
    # per upload and dominates everything else at scale.
    vectors_path: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "corpus.vectors.npy")

    # -- bulk ingest ----------------------------------------------------
    # Embedding is the bottleneck: measured ~41-52 chunks/s for bge-small on a
    # 6-core CPU, and neither batch size, sequence length nor int8 quantisation
    # moved it (sentence-transformers pads to the longest item in the batch, so
    # a lower max_seq_length buys nothing when chunks are already ~200 tokens).
    # The only two levers that work are fewer chunks and more processes.
    # "auto" uses CUDA when available and falls back to CPU. Everything in this
    # project was MEASURED on CPU; the device is configurable because embedding
    # throughput is the entire ingest bottleneck and it is the one thing a GPU
    # changes by an order of magnitude.
    device: str = "auto"
    embed_batch_size: int = 64
    # Worker processes for embedding. 1 = single process (the DEFAULT, and the
    # fastest configuration measured here); 0 = auto (physical cores, capped
    # at 8); N = that many.
    #
    # MEASURED on a 6-physical-core CPU: single process 45 chunks/s, six worker
    # processes 22 chunks/s. Process parallelism is a PESSIMISATION here - the
    # workers contend for the same cores that torch already saturates via BLAS
    # threads, and pool startup costs ~30 s on top. The path is kept because it
    # can win on a machine with many more cores, but it is off by default and
    # the number above is why. Re-measure with:
    #   python -m eval.bench_ingest --docs 25 --pages 500 --workers 6
    embed_workers: int = 1
    bulk_worker_threshold: int = 2000
    # Coarser chunking for large uploads: 26,557 chunks instead of 69,335 on a
    # 25 x 500-page corpus.
    #
    # MEASURED, and the result is not what it looks like: coarse chunks embed at
    # 21 chunks/s against 45 chunks/s for default chunks, because embedding cost
    # scales with TOKENS, not chunks. Net saving is about 1.2x (21.2 min vs
    # 25.7 min), almost all of it from the slightly lower overlap ratio
    # (240/2000 = 12% vs 120/800 = 15%), not from chunk size.
    #
    # It is still worth having - a 2.6x smaller index costs less memory and less
    # search time - but it is not the throughput lever it appears to be.
    bulk_chunk_size: int = 2000
    bulk_chunk_overlap: int = 240


# --------------------------------------------------------------------------
# Module B — retrieval evaluator (the contribution)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatorConfig:
    base_model: str = "microsoft/deberta-v3-small"
    # Filled in by Phase 2 once the fine-tune exists; None means fall back to
    # the untrained base model and say so loudly.
    checkpoint_path: Path | None = field(
        default_factory=lambda: DATA_DIR / "models" / "evaluator"
    )
    labels: tuple[str, ...] = ("correct", "ambiguous", "wrong")
    # Cross-encoder input is question + chunk. Chunks are 800 chars (~200
    # tokens), so 320 fits both with headroom. Train and inference must use the
    # same value or the checkpoint sees different truncation than it trained on.
    max_length: int = 320
    # Tuned on the VALIDATION split by eval/tune_thresholds.py, never on test.
    # Selected by per-chunk macro-F1 (0.8591 on val), not by action accuracy:
    # the action distribution on this split is ~89% Correct, so maximising
    # action accuracy drifts toward a trivial always-Correct predictor.
    correct_threshold: float = 0.50
    wrong_threshold: float = 0.05
    # Corrective retrieval fires when the top chunk grades below this.
    corrective_trigger_threshold: float = 0.50
    # Bounded, per the phase design note: an unbounded re-query loop burns time
    # on a hard query and can never terminate.
    max_corrective_rounds: int = 1
    # Chunks fetched per corrective round, wider than top_k to surface
    # something the first pass missed.
    corrective_fetch_k: int = 15
    batch_size: int = 16


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
    # Citations are 1-based passage numbers, the ALCE convention. Chunk ids
    # like "Ada_Lovelace::3" are error-prone for a model to reproduce, and a
    # single wrong character would read as an invented source.
    citation_pattern: str = r"\[(\d+)\]"
    # Structured outputs make the answer a schema-validated object instead of
    # free text to regex. The text parser stays as a measured fallback: a
    # schema guarantees well-formed citations, never correct ones.
    use_structured_outputs: bool = True
    # Bounded retry when output comes back malformed or uncited.
    max_citation_retries: int = 2
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
    # Trained on MNLI + FEVER + ANLI. FEVER is fact verification against
    # Wikipedia evidence, which is exactly this task: does this passage support
    # this claim. Plain MNLI cross-encoders are weaker on evidence-style pairs.
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    # Label order differs between NLI checkpoints - this one is
    # (entailment, neutral, contradiction) while the cross-encoder/nli-* family
    # is (contradiction, entailment, neutral). Never hardcode an index; the
    # verifier reads id2label off the loaded model.
    max_length: int = 512
    # CALIBRATED by eval/calibrate_verifier.py on a 600-pair labeled slice
    # (203 lexically-verified positives, 397 different-question negatives).
    # 0.20 is the best-F1 operating point: precision 0.9310, recall 0.7980,
    # F1 0.8594, with 12 unsupported pairs passing and 41 supported pairs
    # flagged. The full sweep is in eval/results/verifier_calibration.json;
    # Phase 5 may revisit it once the coverage cost of false flags is known.
    entailment_threshold: float = 0.20
    threshold_is_calibrated: bool = True
    batch_size: int = 16
    # Multi-citation policy. "concat" joins every cited chunk into one premise
    # and asks whether they JOINTLY support the sentence, matching ALCE
    # citation recall. "any" passes if a single chunk entails it, which is more
    # permissive and lets a model pad citations for free.
    multi_citation_policy: str = "concat"
    # Cap on concatenated premise length, so a 5-chunk concat cannot silently
    # truncate away the chunk that actually mattered.
    max_premise_chars: int = 4000
    verdict_log_path: Path = field(
        default_factory=lambda: REPO_ROOT / "eval" / "results" / "verdicts.jsonl"
    )


# --------------------------------------------------------------------------
# Module E — repair and abstention
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RepairConfig:
    # One regeneration attempt per failed sentence, then drop. An unbounded
    # loop burns API budget and can oscillate between two equally unsupported
    # phrasings of the same claim.
    max_repair_attempts: int = 1
    # Trigger (a). Abstain when more than this fraction of the drafted answer
    # fails verification. Evaluated BEFORE repair: a draft that was mostly
    # unsupported is not rescued by patching a sentence or two.
    # Interacts with VerifierConfig.entailment_threshold - a stricter verifier
    # raises the abstention rate - so Phase 6 varies both together.
    abstain_if_unsupported_fraction_above: float = 0.50
    # Trigger (b). Abstain before generation when Module B graded every
    # retrieved chunk wrong. This fires before the one paid component is
    # called, so such an abstention costs nothing.
    abstain_on_no_correct_chunk: bool = True
    # Minimum surviving-answer condition. A single disconnected clause left
    # behind by dropping is worse than a clean refusal.
    min_surviving_sentences: int = 1
    min_surviving_chars: int = 20
    abstention_message: str = "I cannot answer this from the provided sources."


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
