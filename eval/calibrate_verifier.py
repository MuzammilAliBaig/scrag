"""Calibrate Module D's entailment threshold on a labeled slice.

    python -m eval.calibrate_verifier --limit 200

Free and local - no API calls. It reuses the Phase 3 answers already on disk in
the response cache, so it costs nothing to re-run.

THE HARD PART IS THE LABELS, NOT THE SWEEP
------------------------------------------
The obvious calibration set - "the model's own citations are correct" - is
circular: whether those citations are correct is exactly what Module D exists
to decide. Assuming it would calibrate the verifier to agree with the generator.

So the slice is built from two controls that do not depend on the verifier or
on trusting the generator's judgement:

* **Negatives (reliable).** Each generated sentence is paired with a passage
  retrieved for a *different question*. Such a passage cannot support the
  sentence. Ground truth here is as close to certain as this project gets.

* **Positives (lexically verified, weaker).** A sentence is paired with its
  cited passage only when a gold ALCE short answer appears in BOTH the sentence
  and that passage. The passage demonstrably contains the fact the sentence
  states, which is evidence of support independent of the generator's opinion.

The asymmetry is real and is reported: negatives are near-certain, positives are
lexical. String containment is not entailment, so positive recall here is a
lower bound on true performance rather than an unbiased estimate.

The threshold trades false flags against missed hallucinations and drives the
Phase 5 abstention rate directly, so the whole curve is printed, not just the
chosen point.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

import config
from core.generator import CitationGenerator, GenerationUsage
from core.types import NLILabel
from core.verifier import NLIVerifier
from eval import alce as alce_mod
from eval.metrics import normalize_answer


@dataclass
class LabeledPair:
    premise: str
    hypothesis: str
    supported: bool          # ground truth
    kind: str                # "positive_lexical" | "negative_swap"
    question: str = ""


def build_slice(limit: int, k: int, seed: int = config.SEED) -> tuple[list[LabeledPair], dict]:
    """Build the labeled slice from cached Phase 3 answers. No API calls."""
    generator = CitationGenerator()
    examples = alce_mod.load_asqa(limit=limit)
    rng = random.Random(seed)

    # All passages, so a negative can be drawn from a different question.
    pool: list[tuple[str, str]] = []          # (sample_id, text)
    for ex in examples:
        for chunk in ex.context(k):
            pool.append((ex.sample_id, chunk.text))

    pairs: list[LabeledPair] = []
    stats = {"questions": 0, "from_cache": 0, "skipped_no_cache": 0, "sentences": 0}

    for ex in examples:
        context = ex.context(k)
        key = generator._cache_key(ex.question, context)
        cached = generator._cache_read(key)
        if cached is None:
            stats["skipped_no_cache"] += 1
            continue
        stats["from_cache"] += 1
        stats["questions"] += 1

        payload = json.loads(cached)
        draft = generator._draft_from_payload(
            ex.question, context, payload, GenerationUsage(cached=True)
        )
        by_id = {c.chunk_id: c for c in context}
        golds = [normalize_answer(a) for a in ex.all_short_answers if a.strip()]

        for sentence in draft.sentences:
            if not sentence.citation_ids:
                continue
            stats["sentences"] += 1
            sentence_norm = normalize_answer(sentence.text)

            for cid in sentence.citation_ids:
                chunk = by_id.get(cid)
                if chunk is None:
                    continue
                chunk_norm = normalize_answer(chunk.text)
                # Positive only when a gold answer is in BOTH the sentence and
                # the cited passage: the passage demonstrably contains the fact.
                shared = [g for g in golds if g and g in sentence_norm and g in chunk_norm]
                if shared:
                    pairs.append(
                        LabeledPair(chunk.text, sentence.text, True, "positive_lexical", ex.question)
                    )
                    break

            # One negative per sentence: a passage from a different question.
            for _ in range(20):
                other_id, other_text = rng.choice(pool)
                if other_id != ex.sample_id:
                    pairs.append(
                        LabeledPair(other_text, sentence.text, False, "negative_swap", ex.question)
                    )
                    break

    return pairs, stats


def sweep(scores: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows = []
    for threshold in [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]:
        predicted = scores >= threshold
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        tn = int((~predicted & ~truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                # The two error types this decision trades off.
                "missed_hallucinations": fp,   # unsupported, but passed
                "false_flags": fn,             # supported, but flagged
                "tp": tp, "tn": tn,
            }
        )
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Calibrate the Module D threshold")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="verifier_calibration.json")
    args = parser.parse_args()

    print("building labeled slice from cached Phase 3 answers (no API calls)...")
    pairs, stats = build_slice(args.limit, args.k)
    if not pairs:
        raise SystemExit(
            "no pairs built. Run `python -m eval.run_citation_bench --dataset alce "
            f"--limit {args.limit}` first so the response cache is populated."
        )

    positives = sum(1 for p in pairs if p.supported)
    print(f"  questions from cache : {stats['from_cache']} "
          f"(skipped {stats['skipped_no_cache']} with no cached answer)")
    print(f"  pairs                : {len(pairs)}  "
          f"({positives} positive_lexical, {len(pairs) - positives} negative_swap)")

    verifier = NLIVerifier()
    print(f"  checkpoint           : {verifier.checkpoint}")
    print(f"  policy               : single-premise pairs (calibration is per-citation)\n")

    started = time.time()
    scored = verifier.score_pairs([(p.premise, p.hypothesis) for p in pairs])
    elapsed = time.time() - started

    scores = np.array([s[NLILabel.ENTAILMENT.value] for s in scored])
    truth = np.array([p.supported for p in pairs])
    labels = [max(s, key=s.get) for s in scored]

    rows = sweep(scores, truth)
    print(f"  scored {len(pairs)} pairs in {elapsed:.1f}s on CPU\n")
    print(f"  {'thr':>6}{'precision':>11}{'recall':>9}{'F1':>8}"
          f"{'missed halluc.':>16}{'false flags':>13}")
    print("  " + "-" * 63)
    best = max(rows, key=lambda r: r["f1"])
    for row in rows:
        mark = "  <- best F1" if row is best else ""
        print(f"  {row['threshold']:>6.2f}{row['precision']:>11.4f}{row['recall']:>9.4f}"
              f"{row['f1']:>8.4f}{row['missed_hallucinations']:>16}"
              f"{row['false_flags']:>13}{mark}")

    # Separation between the two populations - a sanity check on the slice
    # itself. If positives and negatives score alike, the checkpoint is not
    # discriminating and no threshold will save it.
    pos_scores, neg_scores = scores[truth], scores[~truth]
    print(f"\n  positive mean entailment {pos_scores.mean():.4f}  "
          f"median {np.median(pos_scores):.4f}")
    print(f"  negative mean entailment {neg_scores.mean():.4f}  "
          f"median {np.median(neg_scores):.4f}")
    print(f"  separation               {pos_scores.mean() - neg_scores.mean():+.4f}")

    label_counts = {name: labels.count(name) for name in
                    [l.value for l in NLILabel]}
    print(f"\n  raw NLI label counts     {label_counts}")

    print(f"\n  RECOMMENDED OPERATING POINT: {best['threshold']:.2f}")
    print(f"    precision {best['precision']:.4f}  recall {best['recall']:.4f}  "
          f"F1 {best['f1']:.4f}")
    print(f"    at this point {best['missed_hallucinations']} unsupported pairs pass "
          f"and {best['false_flags']} supported pairs are flagged")
    print("\n  Caveat: positives are lexically verified, not human-labeled - a gold")
    print("  answer string appears in both sentence and passage. String containment")
    print("  is not entailment, so positive recall here is a LOWER BOUND. Negatives")
    print("  (passages from other questions) are near-certain ground truth.")
    print(f"\n  Set config.VERIFIER.entailment_threshold = {best['threshold']:.2f} "
          "and threshold_is_calibrated = True")

    payload = {
        "checkpoint": verifier.checkpoint,
        "slice": {
            "pairs": len(pairs),
            "positives_lexical": positives,
            "negatives_swap": len(pairs) - positives,
            "questions": stats["from_cache"],
            "label_provenance": {
                "positive_lexical": (
                    "gold ALCE short answer appears in BOTH the sentence and the cited "
                    "passage; lexical, not human-labeled; recall is a lower bound"
                ),
                "negative_swap": (
                    "passage retrieved for a different question; near-certain unsupported"
                ),
            },
            "not_used": (
                "the generator's own citation choices as positives - circular, since "
                "their correctness is what Module D exists to decide"
            ),
        },
        "sweep": rows,
        "recommended": best,
        "separation": {
            "positive_mean": round(float(pos_scores.mean()), 4),
            "negative_mean": round(float(neg_scores.mean()), 4),
            "difference": round(float(pos_scores.mean() - neg_scores.mean()), 4),
        },
        "nli_label_counts": label_counts,
        "elapsed_s": round(elapsed, 1),
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
