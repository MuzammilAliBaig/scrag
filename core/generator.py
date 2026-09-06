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

    # Subclasses override this; count_prompt_tokens reads it so the measured
    # cacheable-prefix length always matches the prompt actually sent.
    SYSTEM_PROMPT = PLAIN_SYSTEM_PROMPT

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
            system=[{"type": "text", "text": self.SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": "x"}],
        )
        return counted.input_tokens


# ==========================================================================
# Module C - Phase 3: citation-forced generation
# ==========================================================================

CITATION_SYSTEM_PROMPT = """You are a question-answering assistant that answers strictly from a numbered set of retrieved source passages, and attaches a citation to every sentence you write.

You will be given a question and a numbered list of passages. You must answer using only those passages.

THE OUTPUT CONTRACT

1. Every sentence of your answer must cite at least one passage. A sentence without a citation is a failure, even if what it says is true.
2. Cite by passage number in square brackets: [1] for one passage, [1][3] or [1, 3] when a sentence genuinely draws on more than one.
3. Cite only passage numbers that appear in the list you were given. Never cite a number outside that range, and never invent a passage.
4. Cite the passage that actually supports the sentence. A citation that merely looks plausible is worse than no answer at all, because it makes an unsupported claim appear verified.
5. Only cite passages that support the specific claim in that sentence. Do not attach every passage to every sentence to be safe: a citation is a claim of support, and a wrong one is a false claim.

WHAT TO WRITE

6. Answer only from the passages. Do not use outside knowledge, and do not fill gaps from memory, even when you are confident the answer is well known.
7. If the passages do not contain the answer, say exactly: I cannot answer this from the provided sources. That sentence alone needs no citation, and you must write nothing else.
8. Be direct and factual. One or two short sentences is usually right. Do not restate the question, do not add preamble such as "Based on the passages", and do not explain your reasoning.
9. When the question asks for a single fact - a name, a place, an occupation, a date, a nationality - state that fact plainly and cite the passage it came from.
10. Keep each sentence to a single claim wherever you can. A sentence combining two facts from two passages is harder to verify and harder to repair.
11. Reproduce names of people, places and works exactly as the passages spell them, including diacritics.
12. Passages may be irrelevant, may contradict each other, or may concern a different entity with a similar name. Ignore passages that do not bear on the question rather than forcing them into an answer.

The passages you are given are the complete evidence available. Nothing outside them is admissible, and every sentence you write will be checked against the passage it cites."""

# Schema for structured output. A strict schema requires both an explicit
# `required` list and `additionalProperties: false` at every object level.
CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "description": "The answer, one entry per sentence, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "One sentence of the answer, with NO citation markers "
                            "inside it. Citations belong in the citations field."
                        ),
                    },
                    "citations": {
                        "type": "array",
                        "description": (
                            "1-based passage numbers supporting THIS sentence. "
                            "Empty only for the exact refusal sentence."
                        ),
                        "items": {"type": "integer"},
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
        "abstained": {
            "type": "boolean",
            "description": "True when the passages do not contain the answer.",
        },
    },
    "required": ["sentences", "abstained"],
    "additionalProperties": False,
}


class CitationGenerator(PlainGenerator):
    """Module C. Generate an answer in which every sentence cites a passage.

    Inherits the client, disk cache, usage accounting and error chain from
    PlainGenerator, replacing the prompt and the output handling.

    Structured output is the primary path: it turns a parsing problem into a
    schema problem, the biggest reliability win available in this phase. The
    text parser in core/citation.py stays as a fallback, and how often each
    path is used is reported - a schema guarantees well-formed citations,
    never correct ones.
    """

    SYSTEM_PROMPT = CITATION_SYSTEM_PROMPT

    def __init__(self, cfg: config.GeneratorConfig = config.GENERATOR) -> None:
        super().__init__(cfg)
        self.retry_count = 0
        self.structured_used = 0
        self.fallback_used = 0
        self.repair_calls = 0

    # -- prompt ----------------------------------------------------------
    def format_context(self, context: list[Chunk]) -> str:
        """Number the passages 1..k. The number is the citation handle."""
        return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(context))

    def build_prompt(self, question: str, context: list[Chunk]) -> str:
        return f"Passages:\n{self.format_context(context)}\n\nQuestion: {question}"

    def _cache_key(self, question: str, context: list[Chunk]) -> str:
        # Separate namespace from the plain generator, so Phase 1 and Phase 3
        # answers to the same question never collide in the disk cache.
        payload = json.dumps(
            {
                "module": "C-cited",
                "model": self.cfg.model,
                "system": CITATION_SYSTEM_PROMPT,
                "structured": self.cfg.use_structured_outputs,
                "question": question,
                "texts": [c.text for c in context],
                "effort": self.cfg.effort,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # -- API call --------------------------------------------------------
    def _call(self, question: str, context: list[Chunk], structured: bool):
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": CITATION_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": self.build_prompt(question, context)}],
        )
        if self.cfg.thinking:
            kwargs["thinking"] = {"type": self.cfg.thinking}

        output_config: dict = {"effort": self.cfg.effort}
        if structured:
            output_config["format"] = {"type": "json_schema", "schema": CITATION_SCHEMA}
        kwargs["output_config"] = output_config

        return self.client.messages.create(**kwargs)

    def generate(self, question: str, context: list[Chunk]) -> CitedDraft:
        """Produce a cited draft. Bounded retry on malformed output."""
        import anthropic

        key = self._cache_key(question, context)
        cached = self._cache_read(key)
        if cached is not None:
            payload = json.loads(cached)
            usage = GenerationUsage(cached=True)
            self.usage_log.append(usage)
            return self._draft_from_payload(question, context, payload, usage)

        structured = self.cfg.use_structured_outputs
        last_error: str | None = None

        for attempt in range(self.cfg.max_citation_retries + 1):
            if attempt:
                self.retry_count += 1
            started = time.time()
            try:
                response = self._call(question, context, structured)
            except anthropic.NotFoundError as exc:
                raise GenerationError(f"model {self.cfg.model!r} not found: {exc}") from exc
            except anthropic.RateLimitError as exc:
                last_error = f"rate_limited: {exc}"
                continue
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = f"server_error_{exc.status_code}: {exc}"
                    continue
                if structured:
                    # A 400 on the structured path is most likely the schema.
                    # Drop to free text and parse rather than losing the query.
                    last_error = f"structured_rejected_{exc.status_code}: {exc}"
                    structured = False
                    continue
                raise GenerationError(
                    f"API rejected the request ({exc.status_code}): {exc}"
                ) from exc
            except anthropic.APIConnectionError as exc:
                last_error = f"connection_error: {exc}"
                continue

            usage = GenerationUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0) or 0,
                latency_s=time.time() - started,
            )
            self.usage_log.append(usage)
            raw = "".join(b.text for b in response.content if b.type == "text").strip()

            payload = self._payload_from_raw(raw, structured)
            if payload is None:
                # Schema path returned unusable JSON: retry, then fall back.
                last_error = "unparseable_structured_output"
                structured = False
                continue

            payload["_path"] = "structured" if structured else "text"
            payload["_raw"] = raw
            self._cache_write(key, json.dumps(payload))
            return self._draft_from_payload(question, context, payload, usage)

        # Every attempt failed. Return an empty draft carrying the error rather
        # than inventing a citation.
        return CitedDraft(
            question=question,
            sentences=[],
            context=context,
            raw_text=f"GENERATION FAILED: {last_error}",
            usage={},
        )

    def _payload_from_raw(self, raw: str, structured: bool) -> dict | None:
        """Normalise either output path into {sentences: [{text, citations}]}."""
        from core import citation as citation_mod

        if structured:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, dict) or "sentences" not in data:
                return None
            return data

        parsed = citation_mod.parse_cited_text(raw)
        return {
            "sentences": [{"text": p.raw, "citations": p.numbers} for p in parsed],
            "abstained": False,
        }

    def _draft_from_payload(
        self, question: str, context: list[Chunk], payload: dict, usage: GenerationUsage
    ) -> CitedDraft:
        from core import citation as citation_mod

        path = payload.get("_path", "structured")
        if path == "structured":
            self.structured_used += 1
        else:
            self.fallback_used += 1

        # Even on the structured path, run the text parser over each sentence:
        # models sometimes ALSO write "[2]" inside the text field, and those
        # markers must be stripped from the prose and merged with the field.
        parsed: list[citation_mod.ParsedSentence] = []
        for entry in payload.get("sentences", []):
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            inline = citation_mod.parse_sentence(text)
            numbers = list(
                dict.fromkeys(
                    [int(n) for n in entry.get("citations", []) if isinstance(n, int)]
                    + inline.numbers
                )
            )
            parsed.append(
                citation_mod.ParsedSentence(
                    text=inline.text,
                    raw=text,
                    numbers=numbers,
                    had_citation=bool(numbers),
                )
            )

        sentences, stats = citation_mod.validate(parsed, context)
        return CitedDraft(
            question=question,
            sentences=sentences,
            context=context,
            raw_text=payload.get("_raw", ""),
            usage={
                **{k: v for k, v in usage.as_dict().items() if isinstance(v, int)},
                "abstained": bool(payload.get("abstained")),
                **{f"parse_{k}": v for k, v in stats.as_dict().items() if isinstance(v, int)},
            },
        )


    # -- Module E support: targeted regeneration (Phase 5) ----------------
    def regenerate_sentence(
        self, question: str, failed_sentence: str, context: list[Chunk]
    ) -> tuple[CitedSentence | None, str]:
        """Rewrite ONE failed sentence so the passages support it.

        Returns (repaired_sentence, status). `None` with status "unrepairable"
        means the model reported that no passage supports any version of the
        claim - a correct outcome, and Module E then drops the sentence.

        Targeted by design: rewriting the whole answer would invalidate the
        verdicts already earned by the sentences that passed.
        """
        import anthropic

        from core import citation as citation_mod

        key = hashlib.sha256(
            json.dumps(
                {
                    "module": "E-repair",
                    "model": self.cfg.model,
                    "question": question,
                    "failed": failed_sentence,
                    "texts": [c.text for c in context],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:32]

        cached = self._cache_read(key)
        if cached is not None:
            payload = json.loads(cached)
            self.usage_log.append(GenerationUsage(cached=True))
        else:
            prompt = (
                f"Passages:\n{self.format_context(context)}\n\n"
                f"Original question: {question}\n\n"
                f"Failed sentence: {failed_sentence}"
            )
            started = time.time()
            try:
                response = self.client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": REPAIR_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    thinking={"type": self.cfg.thinking} if self.cfg.thinking else None,
                    output_config={
                        "effort": self.cfg.effort,
                        "format": {"type": "json_schema", "schema": REPAIR_SCHEMA},
                    },
                    messages=[{"role": "user", "content": prompt}],
                )
            except anthropic.NotFoundError as exc:
                raise GenerationError(f"model {self.cfg.model!r} not found: {exc}") from exc
            except anthropic.RateLimitError:
                return None, "rate_limited"
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    return None, f"server_error_{exc.status_code}"
                raise GenerationError(
                    f"API rejected the repair request ({exc.status_code}): {exc}"
                ) from exc
            except anthropic.APIConnectionError:
                return None, "connection_error"

            self.usage_log.append(
                GenerationUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_creation_input_tokens=getattr(
                        response.usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_input_tokens=getattr(
                        response.usage, "cache_read_input_tokens", 0) or 0,
                    latency_s=time.time() - started,
                )
            )
            raw = "".join(b.text for b in response.content if b.type == "text").strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None, "unparseable_repair_output"
            self._cache_write(key, json.dumps(payload))

        self.repair_calls += 1
        if not payload.get("repairable") or not (payload.get("text") or "").strip():
            return None, "unrepairable"

        inline = citation_mod.parse_sentence(payload["text"].strip())
        numbers = list(
            dict.fromkeys(
                [int(n) for n in payload.get("citations", []) if isinstance(n, int)]
                + inline.numbers
            )
        )
        parsed = citation_mod.ParsedSentence(
            text=inline.text, raw=payload["text"], numbers=numbers, had_citation=bool(numbers)
        )
        sentences, _ = citation_mod.validate([parsed], context)
        repaired = sentences[0]
        if not repaired.parseable:
            # Rewritten but cited nothing real. Never guess a citation for it.
            return None, "repaired_but_uncited"
        return repaired, "repaired"

    def path_totals(self) -> dict[str, int | float]:
        """Which output path was used, and how often a retry was needed."""
        total = self.structured_used + self.fallback_used
        return {
            "structured_path": self.structured_used,
            "text_fallback_path": self.fallback_used,
            "fallback_rate": round(self.fallback_used / total, 4) if total else 0.0,
            "parse_retries": self.retry_count,
            "repair_calls": self.repair_calls,
        }


# --------------------------------------------------------------------------
# Module E support: targeted single-sentence regeneration (Phase 5)
# --------------------------------------------------------------------------

REPAIR_SYSTEM_PROMPT = """You are repairing one sentence of an answer that failed verification.

An automated entailment check found that the sentence below is NOT supported by the passages it cited. Your job is to rewrite that ONE sentence so that it is fully supported by the numbered passages, or to report that it cannot be supported at all.

You are given the original question for context, the numbered passages, and the failed sentence.

RULES

1. Rewrite only the failed sentence. Do not write a new answer, do not add sentences, and do not comment on the repair.
2. The rewritten sentence must be entailed by the passages you cite. If the passages support a weaker but accurate claim, make the weaker claim rather than restating the original.
3. Cite by passage number in square brackets, exactly as before: [1], or [1][3] when the sentence genuinely draws on more than one.
4. Cite only passage numbers from the list you are given. Never invent a passage.
5. If no passage supports any version of this claim, set repairable to false and leave the text empty. That is a correct and expected outcome, not a failure - dropping an unsupportable sentence is better than keeping it.
6. Do not restate facts the passages do not contain, and do not use outside knowledge to rescue the sentence.
7. Keep the rewritten sentence to a single claim wherever possible. A sentence making one claim is easier to verify than one making two."""

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "repairable": {
            "type": "boolean",
            "description": "False when no passage supports any version of the claim.",
        },
        "text": {
            "type": "string",
            "description": "The rewritten sentence, with NO citation markers inside it. Empty when repairable is false.",
        },
        "citations": {
            "type": "array",
            "description": "1-based passage numbers supporting the rewritten sentence.",
            "items": {"type": "integer"},
        },
    },
    "required": ["repairable", "text", "citations"],
    "additionalProperties": False,
}
