"""Module D - NLI citation verifier.

draft -> pass/fail per sentence.

For each sentence, the chunks it cited become the **premise** and the sentence
becomes the **hypothesis**. An off-the-shelf NLI checkpoint decides whether the
premise entails the hypothesis. Runs locally on CPU: no API call, so verifying
costs nothing however many sentences are checked.

**Independence is the entire point of this module.** The generator wrote the
sentence *and* chose its citation, so asking the same model whether the
citation is good would be self-marking. Module D must be a different model, and
it is never the generator LLM.

**Neutral is not contradiction.** Neutral means the passage does not support the
sentence; contradiction means the passage says the opposite. Both fail the gate,
but the logs keep them apart, because neutral usually means retrieval failed
while contradiction means generation did.

**Multi-citation policy** is `concat` by default: every cited chunk is joined
into one premise and the question becomes whether they *jointly* support the
sentence. This matches ALCE's citation-recall definition. The alternative,
`any`, passes a sentence if a single chunk entails it - more permissive, and it
lets a model pad citations for free. Both are implemented and the per-chunk
scores are always recorded, so the choice is auditable and Phase 5 can drop the
weakest citation instead of the whole sentence.

Out of scope, deliberately: training or fine-tuning the NLI model, and dropping
or repairing sentences (Phase 5).
"""

from __future__ import annotations

import json


import config
from core.types import (
    Chunk,
    CitedDraft,
    CitedSentence,
    NLILabel,
    SentenceVerdict,
    VerificationReport,
)


class NLIVerifier:
    """Entailment-check every cited sentence against the chunks it cites."""

    def __init__(self, cfg: config.VerifierConfig = config.VERIFIER) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._label_index: dict[NLILabel, int] = {}
        self.verdict_log: list[dict] = []

    # -- checkpoint (4.1) ------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.nli_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.nli_model, dtype=torch.float32
        ).to(self._device)
        self._model.eval()

        # Read the label order off the checkpoint. Hardcoding indices is a real
        # bug source here: this checkpoint is (entailment, neutral,
        # contradiction) while the cross-encoder/nli-* family is
        # (contradiction, entailment, neutral). Getting it wrong silently
        # inverts every verdict.
        id2label = self._model.config.id2label
        self._label_index = {}
        for index, name in id2label.items():
            try:
                self._label_index[NLILabel(str(name).lower())] = int(index)
            except ValueError:
                continue
        missing = set(NLILabel) - set(self._label_index)
        if missing:
            raise RuntimeError(
                f"checkpoint {self.cfg.nli_model} does not expose "
                f"{[m.value for m in missing]} in id2label: {id2label}"
            )

    @property
    def checkpoint(self) -> str:
        return self.cfg.nli_model

    # -- scoring (4.2, 4.6) ----------------------------------------------
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Batched NLI over (premise, hypothesis) pairs. CPU-safe."""
        import torch

        self._load()
        if not pairs:
            return []

        out: list[dict[str, float]] = []
        for start in range(0, len(pairs), self.cfg.batch_size):
            batch = pairs[start : start + self.cfg.batch_size]
            encoded = self._tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                out.append(
                    {label.value: float(row[index]) for label, index in self._label_index.items()}
                )
        return out

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Probability that the premise entails the hypothesis, in [0, 1]."""
        return self.score_pairs([(premise, hypothesis)])[0][NLILabel.ENTAILMENT.value]

    # -- premise construction (4.5) --------------------------------------
    def _premise(self, chunks: list[Chunk]) -> str:
        """Join cited chunks into one premise, truncating from the end.

        Truncation is capped rather than left to the tokenizer alone so a
        five-chunk concatenation cannot silently cut away the chunk that
        actually supported the sentence.
        """
        joined = "\n\n".join(c.text for c in chunks)
        return joined[: self.cfg.max_premise_chars]

    # -- verdicts (4.3, 4.8) ---------------------------------------------
    def verify_sentence(
        self, sentence: CitedSentence, context: list[Chunk]
    ) -> SentenceVerdict:
        """Pass/fail one sentence against the chunks it cites."""
        by_id = {c.chunk_id: c for c in context}
        cited = [by_id[cid] for cid in sentence.citation_ids if cid in by_id]

        if not cited:
            # Nothing to verify. That is a Module C failure (no resolvable
            # citation), not evidence that the sentence is wrong - so it is
            # flagged unsupported but marked `uncited` so the two do not get
            # confused in the error analysis.
            return SentenceVerdict(
                sentence=sentence,
                supported=False,
                entailment_score=0.0,
                nli_label=NLILabel.NEUTRAL,
                policy=self.cfg.multi_citation_policy,
                uncited=True,
            )

        # Always score each citation individually: Phase 5 needs per-citation
        # scores to drop the weakest one rather than the whole sentence.
        per_chunk = self.score_pairs([(c.text, sentence.text) for c in cited])
        per_citation = {
            c.chunk_id: round(scores[NLILabel.ENTAILMENT.value], 4)
            for c, scores in zip(cited, per_chunk)
        }

        if self.cfg.multi_citation_policy == "any" or len(cited) == 1:
            best_index = max(
                range(len(cited)),
                key=lambda i: per_chunk[i][NLILabel.ENTAILMENT.value],
            )
            scores = per_chunk[best_index]
            supporting = cited[best_index].chunk_id
        else:
            scores = self.score_pairs([(self._premise(cited), sentence.text)])[0]
            supporting = max(per_citation, key=per_citation.get)

        entail = scores[NLILabel.ENTAILMENT.value]
        label = NLILabel(max(scores, key=scores.get))

        verdict = SentenceVerdict(
            sentence=sentence,
            supported=entail >= self.cfg.entailment_threshold,
            entailment_score=round(entail, 4),
            supporting_chunk_id=supporting,
            nli_label=label,
            per_citation=per_citation,
            policy=self.cfg.multi_citation_policy,
        )
        self.verdict_log.append(
            {
                "sentence": sentence.text,
                "citation_ids": list(sentence.citation_ids),
                "entailment": verdict.entailment_score,
                "neutral": round(scores[NLILabel.NEUTRAL.value], 4),
                "contradiction": round(scores[NLILabel.CONTRADICTION.value], 4),
                "nli_label": label.value,
                "supported": verdict.supported,
                "per_citation": per_citation,
                "policy": self.cfg.multi_citation_policy,
                "threshold": self.cfg.entailment_threshold,
            }
        )
        return verdict

    def verify(self, draft: CitedDraft) -> VerificationReport:
        """Verify every sentence in a draft. Every sentence gets a verdict."""
        return VerificationReport(
            verdicts=[self.verify_sentence(s, draft.context) for s in draft.sentences]
        )

    def flush_log(self, path=None) -> None:
        """Append per-sentence verdicts to disk for later error analysis."""
        path = path or self.cfg.verdict_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in self.verdict_log:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.verdict_log.clear()


# --------------------------------------------------------------------------
# Demo - python -m core.verifier --demo
# --------------------------------------------------------------------------

_DEMO_CHUNKS = [
    Chunk(
        chunk_id="Ada_Lovelace::0",
        text=(
            "Augusta Ada King, Countess of Lovelace, was an English mathematician "
            "and writer, chiefly known for her work on Charles Babbage's proposed "
            "mechanical general-purpose computer, the Analytical Engine."
        ),
        doc_id="Ada_Lovelace",
    ),
    Chunk(
        chunk_id="Ada_Lovelace::1",
        text=(
            "She was the only legitimate child of the poet Lord Byron and his wife "
            "Anne Isabella Milbanke. Byron left England forever a few months after "
            "she was born."
        ),
        doc_id="Ada_Lovelace",
    ),
    Chunk(
        chunk_id="Mount_Everest::0",
        text=(
            "Mount Everest is Earth's highest mountain above sea level, located in "
            "the Mahalangur Himal sub-range of the Himalayas."
        ),
        doc_id="Mount_Everest",
    ),
]

_DEMO_SENTENCES = [
    # Entailed by its citation.
    CitedSentence("Ada Lovelace was an English mathematician.", ["Ada_Lovelace::0"]),
    # On-topic but the cited chunk says nothing about it: neutral.
    CitedSentence("Ada Lovelace was born in 1815.", ["Ada_Lovelace::1"]),
    # The cited chunk refutes it: contradiction, a generation failure.
    CitedSentence("Ada Lovelace was Lord Byron's illegitimate child.", ["Ada_Lovelace::1"]),
    # Cited chunk is about a different subject entirely.
    CitedSentence("Ada Lovelace worked on the Analytical Engine.", ["Mount_Everest::0"]),
    # No resolvable citation at all: a Module C failure, flagged as uncited.
    CitedSentence("Ada Lovelace is regarded as the first programmer.", [], parseable=False),
]


def _demo() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    verifier = NLIVerifier()
    draft = CitedDraft(
        question="Who was Ada Lovelace?",
        sentences=_DEMO_SENTENCES,
        context=_DEMO_CHUNKS,
        raw_text="",
    )
    print(f"checkpoint : {verifier.checkpoint}")
    print(f"policy     : {verifier.cfg.multi_citation_policy}")
    print(f"threshold  : {verifier.cfg.entailment_threshold}"
          f"{'' if verifier.cfg.threshold_is_calibrated else '   (UNCALIBRATED placeholder)'}")
    print()

    report = verifier.verify(draft)
    for verdict in report.verdicts:
        mark = "PASS" if verdict.supported else "FAIL"
        note = " [uncited]" if verdict.uncited else ""
        print(f"  {mark}  entail={verdict.entailment_score:.3f}  "
              f"{verdict.nli_label.value:>13}{note}")
        print(f"        {verdict.sentence.text}")
        print(f"        cited: {verdict.sentence.citation_ids or 'none'}")
    print(f"\n  flag rate: {report.flag_rate:.3f}   "
          f"contradictions: {len(report.contradicted)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module D demo")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        parser.print_help()
