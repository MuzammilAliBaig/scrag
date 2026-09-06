"""Evaluation metrics for the SCRAG baseline and ablation.

Two metrics live here, and they measure different things:

* **Answer accuracy** - does the answer contain a gold surface form? This is
  PopQA's own metric, computed locally at no cost.
* **Faithfulness** - of the claims the answer makes, how many are supported by
  the retrieved context? This uses the RAGAS *definition* (claims entailed by
  context / total claims) with our own implementation calling Claude directly.
  RAGAS the library hard-requires openai + langchain + langchain_openai, which
  this project does not take on. **Report this as "RAGAS-definition
  faithfulness, own implementation", never as a RAGAS number.**

Accuracy and faithfulness can move in opposite directions, and that is the
point: an answer can be accurate because the model already knew the fact while
being unfaithful to the passages it was given. Closing that gap is what
Modules B-E exist to do.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import dataclass, field

import config
from core.types import Chunk, CitedSentence

# --------------------------------------------------------------------------
# Answer accuracy - local, free
# --------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents, articles, punctuation and extra whitespace.

    Standard open-domain QA normalization, so "The Politician." matches
    "politician".
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_matches(prediction: str, gold_answers: list[str]) -> bool:
    """PopQA accuracy: does the prediction contain any gold surface form?

    Substring containment rather than exact match, because the generator may
    answer in a sentence while gold answers are bare entities.
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred:
            return True
    return False


def is_abstention(prediction: str) -> bool:
    """Whether the model declined rather than answered.

    Counted separately from a wrong answer: an abstention is a correct outcome
    when the sources do not support an answer, and it lowers coverage rather
    than accuracy.
    """
    return "cannot answer this from the provided sources" in prediction.lower()


# --------------------------------------------------------------------------
# Faithfulness - RAGAS definition, own implementation, billed to the Claude API
# --------------------------------------------------------------------------
FAITHFULNESS_SYSTEM = """You are a strict evaluator measuring whether an answer is grounded in the passages it was given.

You will receive a question, a set of source passages, and an answer.

Work in two steps.

Step 1. Break the answer into atomic factual claims. An atomic claim states exactly one fact and can be judged true or false on its own. Split compound sentences. Ignore hedges, restatements of the question, and pure formatting. If the answer is a refusal to answer, produce zero claims.

Step 2. For each claim, decide whether the passages support it. A claim is supported only if the passages state it or directly entail it. A claim is not supported if it merely sounds plausible, if it is common knowledge absent from the passages, or if the passages discuss a different entity. Judge only against the passages, never against your own knowledge, no matter how certain you are that a claim is true in the world.

Return your judgement as JSON with this exact shape:

{"claims": [{"claim": "<the atomic claim>", "supported": true or false, "reason": "<short justification>"}]}

Return only the JSON object. No preamble, no code fences, no commentary. If the answer contains no factual claims, return {"claims": []}."""


@dataclass
class FaithfulnessResult:
    score: float | None            # None when it could not be measured
    n_claims: int = 0
    n_supported: int = 0
    claims: list[dict] = field(default_factory=list)
    error: str | None = None


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a response, tolerating stray prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class FaithfulnessJudge:
    """LLM judge implementing the RAGAS faithfulness definition."""

    def __init__(self, cfg: config.EvalConfig = config.EVAL) -> None:
        self.cfg = cfg
        self._client = None
        self.usage: list[dict] = []

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def score(self, question: str, answer: str, context: list[Chunk]) -> FaithfulnessResult:
        """Fraction of the answer's claims that the context supports."""
        import anthropic

        if is_abstention(answer) or not answer.strip():
            # An abstention asserts nothing, so it cannot be unfaithful. It is
            # excluded from the mean rather than scored 1.0, which would let a
            # system inflate faithfulness by refusing everything.
            return FaithfulnessResult(score=None, n_claims=0)

        passages = "\n\n".join(
            f"[Passage {i + 1}]\n{c.text}" for i, c in enumerate(context)
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Passages:\n{passages}\n\n"
            f"Answer:\n{answer}"
        )

        try:
            response = self.client.messages.create(
                model=self.cfg.faithfulness_model,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": FAITHFULNESS_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.faithfulness_effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.NotFoundError as exc:
            return FaithfulnessResult(score=None, error=f"model_not_found: {exc}")
        except anthropic.RateLimitError as exc:
            return FaithfulnessResult(score=None, error=f"rate_limited: {exc}")
        except anthropic.APIStatusError as exc:
            return FaithfulnessResult(score=None, error=f"api_error_{exc.status_code}: {exc}")
        except anthropic.APIConnectionError as exc:
            return FaithfulnessResult(score=None, error=f"connection_error: {exc}")

        self.usage.append(
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                ) or 0,
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ) or 0,
            }
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        parsed = _extract_json(text)
        if parsed is None or "claims" not in parsed:
            return FaithfulnessResult(score=None, error=f"unparseable_judge_output: {text[:200]}")

        claims = parsed["claims"][: self.cfg.faithfulness_max_claims]
        if not claims:
            return FaithfulnessResult(score=None, n_claims=0, claims=[])

        supported = sum(1 for c in claims if c.get("supported") is True)
        return FaithfulnessResult(
            score=supported / len(claims),
            n_claims=len(claims),
            n_supported=supported,
            claims=claims,
        )

    def cost_usd(self) -> float:
        cfg = config.GENERATOR
        per_m = 1_000_000
        return sum(
            (
                u["input_tokens"] * cfg.price_input_per_mtok
                + u["output_tokens"] * cfg.price_output_per_mtok
                + u["cache_creation_input_tokens"] * cfg.price_cache_write_per_mtok
                + u["cache_read_input_tokens"] * cfg.price_cache_read_per_mtok
            )
            / per_m
            for u in self.usage
        )


# --------------------------------------------------------------------------
# Retrieval quality - local, free
# --------------------------------------------------------------------------
def retrieval_hit(gold_answers: list[str], context: list[Chunk]) -> bool:
    """Whether any retrieved chunk actually contains a gold answer.

    An upper bound on what the generator could get right from the context
    alone. The gap between this and accuracy is how often the model answered
    from parametric memory rather than from the passages - exactly the failure
    SCRAG exists to close.
    """
    joined = normalize_answer(" ".join(c.text for c in context))
    return any(normalize_answer(g) in joined for g in gold_answers if g.strip())


# --------------------------------------------------------------------------
# Citation precision / recall - ALCE definitions, computed by entailment
# --------------------------------------------------------------------------
#
# Phase 3 could only report a word-overlap proxy, because entailment is Module
# D and did not exist yet. With Module D built these are the real ALCE-style
# quantities:
#
#   citation RECALL    - of the sentences that make a claim, how many are
#                        entailed by the set of passages they cite. A sentence
#                        whose citations do not support it counts against
#                        recall even if it cites something.
#
#   citation PRECISION - of the individual citations attached, how many
#                        actually contribute. ALCE calls a citation precise
#                        when the cited passage supports the sentence on its
#                        own OR is necessary to the joint support; here a
#                        citation counts as precise when it individually
#                        entails the sentence, which is the stricter reading
#                        and the one that penalises citation padding.
#
# These are OUR implementation of ALCE's definitions using a different
# entailment model (ALCE uses TRUE/T5-XXL). Report them as such, never as
# ALCE's published numbers.


@dataclass
class CitationScores:
    recall: float | None
    precision: float | None
    sentences_scored: int
    citations_scored: int
    supported_sentences: int
    precise_citations: int

    def as_dict(self) -> dict:
        return {
            "citation_recall": round(self.recall, 4) if self.recall is not None else None,
            "citation_precision": (
                round(self.precision, 4) if self.precision is not None else None
            ),
            "sentences_scored": self.sentences_scored,
            "citations_scored": self.citations_scored,
            "supported_sentences": self.supported_sentences,
            "precise_citations": self.precise_citations,
        }


def citation_precision_recall(
    drafts_and_reports: list[tuple[list[CitedSentence], list, list[Chunk]]],
    threshold: float,
) -> CitationScores:
    """Citation P/R over a set of (sentences, verdicts, context) triples.

    `verdicts` must come from Module D with per-citation scores populated, so
    this function does no model work of its own and can be applied to any
    already-verified run.
    """
    sentences_scored = 0
    supported = 0
    citations_scored = 0
    precise = 0

    for sentences, verdicts, _context in drafts_and_reports:
        for verdict in verdicts:
            # A refusal makes no claim, so it is neither supported nor
            # unsupported - excluded rather than scored zero.
            if verdict.uncited and not verdict.sentence.citation_ids:
                continue
            sentences_scored += 1
            if verdict.supported:
                supported += 1
            for _cid, score in (verdict.per_citation or {}).items():
                citations_scored += 1
                if score >= threshold:
                    precise += 1

    return CitationScores(
        recall=supported / sentences_scored if sentences_scored else None,
        precision=precise / citations_scored if citations_scored else None,
        sentences_scored=sentences_scored,
        citations_scored=citations_scored,
        supported_sentences=supported,
        precise_citations=precise,
    )


# --------------------------------------------------------------------------
# Hallucination rate
# --------------------------------------------------------------------------


def hallucination_rate_automatic(verdicts: list) -> dict:
    """Automatic proxy for hallucination rate, from Module D verdicts.

    A sentence counts as hallucinated when it asserts something no cited
    passage supports. Contradictions are counted separately: a passage that
    *refutes* the sentence is a stronger signal than one that merely fails to
    support it.

    This is a proxy. It measures disagreement between the generator and the NLI
    checkpoint, not ground truth, and it inherits every error the verifier
    makes. The hand-labeled set (eval/LABELING_HALLUCINATION.md) exists to
    bound that gap; until it has been labeled, report this number as
    "automatic, verifier-derived" and never as a human-verified hallucination
    rate.
    """
    scored = [v for v in verdicts if not (v.uncited and not v.sentence.citation_ids)]
    if not scored:
        return {
            "hallucination_rate_auto": None,
            "contradiction_rate_auto": None,
            "sentences": 0,
        }
    unsupported = sum(1 for v in scored if not v.supported)
    contradicted = sum(
        1 for v in scored if getattr(v.nli_label, "value", v.nli_label) == "contradiction"
    )
    return {
        "hallucination_rate_auto": round(unsupported / len(scored), 4),
        "contradiction_rate_auto": round(contradicted / len(scored), 4),
        "sentences": len(scored),
    }


def hallucination_rate_labeled(rows: list[dict]) -> dict:
    """Hallucination rate against the hand-labeled set.

    `rows` are entries from eval/labeled/*.jsonl carrying a human `label` of
    "supported" or "hallucinated". Returns None when the set is empty, rather
    than a zero that would read as a perfect score.
    """
    labeled = [r for r in rows if r.get("label") in {"supported", "hallucinated"}]
    if not labeled:
        return {
            "hallucination_rate_labeled": None,
            "n_labeled": 0,
            "note": "hand-labeled set is empty - see eval/LABELING_HALLUCINATION.md",
        }
    hallucinated = sum(1 for r in labeled if r["label"] == "hallucinated")
    return {
        "hallucination_rate_labeled": round(hallucinated / len(labeled), 4),
        "n_labeled": len(labeled),
    }
