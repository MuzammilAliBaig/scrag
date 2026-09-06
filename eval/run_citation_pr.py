"""Citation precision/recall on ALCE, computed by entailment. Free, local.

    python -m eval.run_citation_pr --limit 200

Phase 3 could only report a word-overlap proxy for citation quality, because
entailment is Module D and did not exist yet. This closes that gap using the
cached Phase 3 answers and the Module D checkpoint, so it makes no API calls
and costs nothing.

Definitions follow ALCE:

* **citation recall**    - fraction of claim-making sentences that the passages
  they cite actually entail.
* **citation precision** - fraction of individual citations that support the
  sentence on their own. This is the stricter reading, and the one that
  penalises citation padding: attaching a fifth passage that does not bear on
  the sentence costs precision.

Computed with our checkpoint (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`),
NOT with ALCE's TRUE/T5-XXL. Comparable in spirit, not in implementation, and
no ALCE baseline figure is recorded in this project - so these numbers are
reported unqualified rather than against an assumed bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config
from core.generator import CitationGenerator, GenerationUsage
from core.verifier import NLIVerifier
from eval import alce as alce_mod
from eval.metrics import citation_precision_recall, hallucination_rate_automatic


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Citation P/R by entailment (free)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="citation_pr.json")
    args = parser.parse_args()

    generator = CitationGenerator()
    verifier = NLIVerifier()
    examples = alce_mod.load_asqa(limit=args.limit)

    print(f"checkpoint : {verifier.checkpoint}")
    print(f"threshold  : {config.VERIFIER.entailment_threshold} "
          f"({'calibrated' if config.VERIFIER.threshold_is_calibrated else 'UNCALIBRATED'})")
    print(f"policy     : {config.VERIFIER.multi_citation_policy}")
    print("source     : cached Phase 3 answers - no API calls\n")

    triples = []
    all_verdicts = []
    skipped = 0
    started = time.time()

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
        triples.append((draft.sentences, report.verdicts, context))
        all_verdicts.extend(report.verdicts)
        if i % 50 == 0:
            print(f"  {i}/{len(examples)}", flush=True)

    if not triples:
        raise SystemExit(
            "no cached answers found. Run:\n"
            f"  python -m eval.run_citation_bench --dataset alce --limit {args.limit}"
        )

    scores = citation_precision_recall(triples, config.VERIFIER.entailment_threshold)
    auto = hallucination_rate_automatic(all_verdicts)
    elapsed = time.time() - started

    # Citations per sentence, to make padding visible.
    cited_counts = [
        len(v.sentence.citation_ids) for v in all_verdicts if v.sentence.citation_ids
    ]
    mean_citations = sum(cited_counts) / len(cited_counts) if cited_counts else 0.0

    print("\n" + "=" * 72)
    print("  CITATION PRECISION / RECALL (ALCE definitions, our checkpoint)  MEASURED")
    print("=" * 72)
    print(f"  questions scored          {len(triples)}  (skipped {skipped}, not cached)")
    print(f"  sentences scored          {scores.sentences_scored}")
    print(f"  citations scored          {scores.citations_scored}")
    print(f"  mean citations / sentence {mean_citations:.2f}")
    print()
    print(f"  CITATION RECALL           {scores.recall:.4f}  "
          f"({scores.supported_sentences}/{scores.sentences_scored} sentences entailed "
          f"by what they cite)")
    print(f"  CITATION PRECISION        {scores.precision:.4f}  "
          f"({scores.precise_citations}/{scores.citations_scored} individual citations "
          f"support the sentence)")
    print()
    print(f"  hallucination rate (auto) {auto['hallucination_rate_auto']:.4f}")
    print(f"  contradiction rate (auto) {auto['contradiction_rate_auto']:.4f}")
    print(f"  elapsed                   {elapsed:.1f}s on CPU, $0.00")
    print()
    print("  ALCE baseline: NOT AVAILABLE - the ALCE paper is not recorded in this")
    print("  project, so these are reported unqualified. They also use a different")
    print("  entailment model than ALCE's TRUE/T5-XXL, so they would not be a")
    print("  like-for-like comparison even if the figure were to hand.")
    print("=" * 72)

    payload = {
        "source": "cached Phase 3 answers (ALCE/ASQA)",
        "api_cost_usd": 0.0,
        "checkpoint": verifier.checkpoint,
        "threshold": config.VERIFIER.entailment_threshold,
        "policy": config.VERIFIER.multi_citation_policy,
        "questions_scored": len(triples),
        "skipped_not_cached": skipped,
        "mean_citations_per_sentence": round(mean_citations, 4),
        **scores.as_dict(),
        **auto,
        "alce_baseline": (
            "NOT AVAILABLE - not recorded in this project. Also a different entailment "
            "model than ALCE's TRUE/T5-XXL, so not like-for-like regardless."
        ),
        "elapsed_s": round(elapsed, 1),
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
