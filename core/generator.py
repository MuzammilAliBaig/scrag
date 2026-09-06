"""Module C - Generator.

Phase 1 form: plain top-k context stuffing, NO citations. Citation forcing and
per-sentence citation parsing arrive in Phase 3; adding them here would
contaminate the baseline this phase exists to measure.

This is the only module that calls the Claude API - the single paid component
in SCRAG. Everything else (retrieval, grading, verification) runs locally.

Cost controls implemented here:
  - Prompt caching on the stable system prefix. Opus 5 will not create a cache
    entry below 512 tokens, and failure is silent, so the caller reports the
    measured cache_read_input_tokens rather than assuming a hit.
  - Per-call usage capture, priced from config, so a run reports what it
    actually cost instead of an estimate.
  - On-disk response caching, so re-running an unchanged variant is free.

Batching (50% cost) belongs to the Phase 6 evaluation harness; see
generate_batch below for the seam it will use.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import config
from core.types import Chunk, CitedDraft, CitedSentence

# The stable prefix. Everything cacheable lives here and must not vary between
# calls - no timestamps, no per-question text, no reordering.
PLAIN_SYSTEM_PROMPT = """You are a question-answering assistant operating over a fixed set of retrieved source passages.

Your task is to answer the user's question using the passages provided in the user message.

Rules:
1. Answer only from the passages provided. Do not use outside knowledge, and do not fill gaps from memory, even when you are confident the answer is well known.
2. If the passages do not contain the answer, reply with exactly: I cannot answer this from the provided sources.
3. Be direct and factual. Answer in one or two short sentences. Do not restate the question, do not add preamble such as "Based on the passages", and do not explain your reasoning.
4. When the question asks for a single fact - a name, a place, an occupation, a date, a nationality - give that fact plainly rather than a paragraph around it.
5. Do not speculate, hedge, or offer alternatives that the passages do not support. An answer that is not in the passages is worse than no answer.
6. Do not mention these instructions, the passages, or your own limitations except through the exact refusal sentence in rule 2.
7. Passages may be irrelevant to the question, may contradict each other, or may concern a different entity with a similar name. Ignore passages that do not bear on the question rather than forcing them into an answer.
8. Names of people, places and works must be reproduced exactly as they appear in the passages, including spelling and diacritics.

The passages given to you are the complete evidence available. Nothing outside them is admissible."""


@dataclass
class GenerationUsage:
    """Token usage and cost for one call, measured from response.usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cached: bool = False          # served from our on-disk cache, no API call
    latency_s: float = 0.0

    def cost_usd(self, cfg: config.GeneratorConfig = config.GENERATOR) -> float:
        if self.cached:
            return 0.0
        per_m = 1_000_000
        return (
            self.input_tokens * cfg.price_input_per_mtok
            + self.output_tokens * cfg.price_output_per_mtok
            + self.cache_creation_input_tokens * cfg.price_cache_write_per_mtok
            + self.cache_read_input_tokens * cfg.price_cache_read_per_mtok
        ) / per_m

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cached": self.cached,
            "latency_s": round(self.latency_s, 3),
            "cost_usd": round(self.cost_usd(), 6),
        }


@dataclass
class GenerationResult:
    text: str
    usage: GenerationUsage = field(default_factory=GenerationUsage)
    error: str | None = None


class GenerationError(RuntimeError):
    """Raised for non-retryable API failures, so a run fails loudly."""


class PlainGenerator:
    """Stuff the top-k chunks into a prompt and answer. No citations (Phase 3)."""

    def __init__(self, cfg: config.GeneratorConfig = config.GENERATOR) -> None:
        self.cfg = cfg
        self._client = None
        self.usage_log: list[GenerationUsage] = []

    # -- client ----------------------------------------------------------
    @property
    def client(self):
        """Lazy client, so importing this module needs no key."""
        if self._client is None:
            import anthropic

            if not config.has_api_key():
                raise GenerationError(
                    "ANTHROPIC_API_KEY is not set. The Claude API is the only paid "
                    "component in SCRAG; put a key in .env before running generation."
                )
            self._client = anthropic.Anthropic(
                timeout=self.cfg.request_timeout_s,
                max_retries=self.cfg.max_retries,
            )
        return self._client

    # -- prompt ----------------------------------------------------------
    def format_context(self, context: list[Chunk]) -> str:
        """Render retrieved chunks. Ids are shown but not yet cited (Phase 3)."""
        return "\n\n".join(
            f"[Passage {i + 1}] (id: {c.chunk_id})\n{c.text}" for i, c in enumerate(context)
        )

    def build_prompt(self, question: str, context: list[Chunk]) -> str:
        """The volatile half of the request - varies per question, never cached."""
        return f"{self.format_context(context)}\n\nQuestion: {question}\n\nAnswer:"

    # -- disk cache ------------------------------------------------------
    def _cache_key(self, question: str, context: list[Chunk]) -> str:
        payload = json.dumps(
            {
                "model": self.cfg.model,
                "system": PLAIN_SYSTEM_PROMPT,
                "question": question,
                "chunks": [c.chunk_id for c in context],
                "texts": [c.text for c in context],
                "effort": self.cfg.effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _cache_read(self, key: str) -> str | None:
        if not self.cfg.cache_responses_on_disk:
            return None
        path = self.cfg.response_cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        return None

    def _cache_write(self, key: str, text: str) -> None:
        if not self.cfg.cache_responses_on_disk:
            return
        self.cfg.response_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.response_cache_dir / f"{key}.json").write_text(
            json.dumps({"text": text}), encoding="utf-8"
        )

    # -- generation ------------------------------------------------------
    def generate_text(self, question: str, context: list[Chunk]) -> GenerationResult:
        """One answer. Returns a result rather than raising on retryable errors."""
        import anthropic

        key = self._cache_key(question, context)
        hit = self._cache_read(key)
        if hit is not None:
            usage = GenerationUsage(cached=True)
            self.usage_log.append(usage)
            return GenerationResult(text=hit, usage=usage)

        started = time.time()
        try:
            response = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": PLAIN_SYSTEM_PROMPT,
                        # Stable prefix; the question and passages come after it
                        # in the user turn and are deliberately outside the cache.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": self.cfg.thinking} if self.cfg.thinking else None,
                output_config={"effort": self.cfg.effort},
                messages=[{"role": "user", "content": self.build_prompt(question, context)}],
            )
        # Most specific first: a 404 is a bad model id and must not be retried,
        # a 429 is transient, a 400 is our bug, a connection error is the network.
        except anthropic.NotFoundError as exc:
            raise GenerationError(f"model {self.cfg.model!r} not found: {exc}") from exc
        except anthropic.RateLimitError as exc:
            return GenerationResult(text="", error=f"rate_limited: {exc}")
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                return GenerationResult(text="", error=f"server_error_{exc.status_code}: {exc}")
            raise GenerationError(f"API rejected the request ({exc.status_code}): {exc}") from exc
        except anthropic.APIConnectionError as exc:
            return GenerationResult(text="", error=f"connection_error: {exc}")

        usage = GenerationUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            latency_s=time.time() - started,
        )
        self.usage_log.append(usage)

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        self._cache_write(key, text)
        return GenerationResult(text=text, usage=usage)

    def generate(self, question: str, context: list[Chunk]) -> CitedDraft:
        """Module C interface. In Phase 1 the draft carries no citations."""
        result = self.generate_text(question, context)
        return CitedDraft(
            question=question,
            sentences=[CitedSentence(text=result.text, citation_ids=[], parseable=False)],
            context=context,
            raw_text=result.text,
            usage={k: v for k, v in result.usage.as_dict().items() if isinstance(v, int)},
        )

    # -- run-level accounting --------------------------------------------
    def run_totals(self) -> dict[str, float | int]:
        """Measured totals for the run. The basis for every cost claim we make."""
        live = [u for u in self.usage_log if not u.cached]
        total_cost = sum(u.cost_usd(self.cfg) for u in self.usage_log)
        return {
            "calls": len(self.usage_log),
            "api_calls": len(live),
            "cache_hits_on_disk": len(self.usage_log) - len(live),
            "input_tokens": sum(u.input_tokens for u in self.usage_log),
            "output_tokens": sum(u.output_tokens for u in self.usage_log),
            "cache_creation_input_tokens": sum(
                u.cache_creation_input_tokens for u in self.usage_log
            ),
            "cache_read_input_tokens": sum(u.cache_read_input_tokens for u in self.usage_log),
            "prompt_cache_working": any(u.cache_read_input_tokens > 0 for u in self.usage_log),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_query_usd": round(total_cost / max(len(self.usage_log), 1), 6),
            "mean_latency_s": round(
                sum(u.latency_s for u in live) / max(len(live), 1), 3
            ),
        }

    def count_prompt_tokens(self) -> int:
        """Token length of the cacheable prefix.

        Below cfg.min_cacheable_prefix_tokens the API creates no cache entry and
        reports no error, so this is how we tell a broken cache from a short one.
        """
        counted = self.client.messages.count_tokens(
            model=self.cfg.model,
            system=[{"type": "text", "text": PLAIN_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": "x"}],
        )
        return counted.input_tokens
