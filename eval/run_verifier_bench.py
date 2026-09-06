"""Module D on real answers: the flag rate Phase 5 will have to handle.

    python -m eval.run_verifier_bench --limit 200

Free and local. Re-verifies the Phase 3 answers already in the response cache,
so it costs nothing and adds no API calls.

The gate for Phase 4 is that **every sentence gets a pass/fail and unsupported
sentences are flagged**. This script checks that literally: no sentence may be
left without a verdict.

The flag rate reported here is the input to Phase 5. Every flagged sentence
becomes either a repair attempt or an abstention, so this number sets the
ceiling on how much work Module E has to do - and, through abstention, the
coverage cost of the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import config
from core.generator import CitationGenerator, GenerationUsage
from core.types import NLILabel
from core.verifier import NLIVerifier
from eval import alce as alce_mod


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Verify cached Phase 3 answers")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--policy", default=None, choices=["concat", "any"])
    parser.add_argument("--out", default="verifier_bench.json")
    args = parser.parse_args()

    cfg = config.VERIFIER
    if args.policy:
        from dataclasses import replace

        cfg = replace(cfg, multi_citation_policy=args.policy)

    generator = CitationGenerator()
    verifier = NLIVerifier(cfg)
    examples = alce_mod.load_asqa(limit=args.limit)

    print(f"checkpoint : {verifier.checkpoint}")
    print(f"threshold  : {cfg.entailment_threshold}"
          f"{'  (calibrated)' if cfg.threshold_is_calibrated else '  (UNCALIBRATED)'}")
    print(f"policy     : {cfg.multi_citation_policy}\n")

    started = time.time()
    total_sentences = 0
    verdicts_issued = 0
    supported = 0
    uncited = 0
    label_counts = Counter()
    multi_cite_sentences = 0
    per_question = []
    skipped = 0

    for i, ex in enumerate(examples, 1):
        context = ex.context(args.k)
        cached = generator._cache_read(generator._cache_key(ex.question, context))
        if cached is None:
            skipped += 1
            continue

        draft = generator._draft_from_payload(
            ex.question, context, json.loads(cached), GenerationUsage(cached=True)
        )
        if not draft.sentences:
            continue

        report = verifier.verify(draft)

        # The gate, checked literally: one verdict per sentence, no exceptions.
        assert len(report.verdicts) == len(draft.sentences), (
            f"{len(draft.sentences)} sentences but {len(report.verdicts)} verdicts "
            f"on question {ex.sample_id}"
        )

        total_sentences += len(draft.sentences)
        verdicts_issued += len(report.verdicts)
        supported += sum(1 for v in report.verdicts if v.supported)
        uncited += sum(1 for v in report.verdicts if v.uncited)
        multi_cite_sentences += sum(
            1 for v in report.verdicts if len(v.sentence.citation_ids) > 1
        )
        for v in report.verdicts:
            label_counts[v.nli_label.value] += 1

        per_question.append(
            {
                "sample_id": ex.sample_id,
                "question": ex.question,
                "sentences": len(draft.sentences),
                "supported": sum(1 for v in report.verdicts if v.supported),
                "flag_rate": round(report.flag_rate, 4),
                "contradictions": len(report.contradicted),
                "verdicts": [
                    {
                        "text": v.sentence.text,
                        "citations": list(v.sentence.citation_ids),
                        "entailment": v.entailment_score,
                        "label": v.nli_label.value,
                        "supported": v.supported,
                        "per_citation": v.per_citation,
                    }
                    for v in report.verdicts
                ],
            }
        )

        if i % 50 == 0:
            print(f"  verified {i}/{len(examples)} questions", flush=True)

    elapsed = time.time() - started
    verifier.flush_log()

    flagged = total_sentences - supported
    contradicted = label_counts[NLILabel.CONTRADICTION.value]

    print("\n" + "=" * 72)
    print("  MODULE D - VERIFICATION ON REAL ANSWERS (ALCE/ASQA)   MEASURED")
    print("=" * 72)
    print(f"  questions verified        {len(per_question)}  (skipped {skipped}, not cached)")
    print(f"  sentences                 {total_sentences}")
    print(f"  verdicts issued           {verdicts_issued}   "
          f"({'PASS - every sentence judged' if verdicts_issued == total_sentences else 'FAIL'})")
    print(f"  elapsed                   {elapsed:.1f}s on CPU")

    print(f"\n  --- OUTCOMES ---")
    print(f"  supported (pass)          {supported}  ({supported / max(total_sentences,1):.4f})")
    print(f"  FLAGGED (fail)            {flagged}  ({flagged / max(total_sentences,1):.4f})")
    print(f"  of which uncited          {uncited}   (Module C failure, nothing to verify)")

    print(f"\n  --- NLI LABELS (neutral is not contradiction) ---")
    for name in [l.value for l in NLILabel]:
        count = label_counts[name]
        print(f"  {name:>14}  {count:>5}  ({count / max(total_sentences,1):.4f})")
    print(f"  contradictions are generation failures: the cited passage refutes")
    print(f"  the sentence. Neutral usually means retrieval failed instead.")

    print(f"\n  --- MULTI-CITATION ---")
    print(f"  sentences citing >1 chunk {multi_cite_sentences}  "
          f"({multi_cite_sentences / max(total_sentences,1):.4f})")
    print(f"  policy applied            {cfg.multi_citation_policy}")

    print(f"\n  --- WHAT PHASE 5 INHERITS ---")
    print(f"  {flagged} flagged sentences must each be repaired, dropped, or abstained on.")
    print("=" * 72)

    payload = {
        "checkpoint": verifier.checkpoint,
        "threshold": cfg.entailment_threshold,
        "threshold_is_calibrated": cfg.threshold_is_calibrated,
        "policy": cfg.multi_citation_policy,
        "questions": len(per_question),
        "skipped_not_cached": skipped,
        "gate": {
            "sentences": total_sentences,
            "verdicts_issued": verdicts_issued,
            "every_sentence_judged": verdicts_issued == total_sentences,
        },
        "outcomes": {
            "supported": supported,
            "flagged": flagged,
            "flag_rate": round(flagged / max(total_sentences, 1), 4),
            "uncited": uncited,
        },
        "nli_labels": dict(label_counts),
        "multi_citation": {
            "sentences_citing_multiple": multi_cite_sentences,
            "policy": cfg.multi_citation_policy,
        },
        "elapsed_s": round(elapsed, 1),
        "examples": per_question,
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"per-sentence verdict log: {cfg.verdict_log_path}")


if __name__ == "__main__":
    main()
