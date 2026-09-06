"""Tune Module B's decision thresholds on the VALIDATION split.

    python -m eval.tune_thresholds

The thresholds in config.py started as placeholders. This sweeps them and
reports the frontier, so the values that end up in config are chosen by
measurement rather than by guess.

**Validation only.** Tuning on test and then reporting test would be scoring
the exam you wrote. This script refuses to run on the test split. Re-run
eval/run_evaluator_bench.py --split test exactly once afterwards.

Two objectives are reported because they disagree, and the disagreement is
itself a finding:

* **action accuracy** - the CRAG-comparable number. Maximising it alone drifts
  toward always predicting Correct, since the action distribution is ~87%
  Correct. The majority baseline is printed beside every row so that collapse
  is visible rather than hidden.
* **per-chunk macro F1** - forces the `ambiguous` class to keep counting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import config
from core.evaluator import RetrievalEvaluator
from core.types import Chunk, ChunkLabel
from train.weak_labels import ACTION_TO_CRAG, derive_action

LABELS = [ChunkLabel.CORRECT.value, ChunkLabel.AMBIGUOUS.value, ChunkLabel.WRONG.value]


def macro_f1(true: list[str], pred: list[str]) -> float:
    t, p = np.array(true), np.array(pred)
    scores = []
    for name in LABELS:
        tp = int(((p == name) & (t == name)).sum())
        fp = int(((p == name) & (t != name)).sum())
        fn = int(((p != name) & (t == name)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return float(np.mean(scores))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Tune Module B thresholds on validation")
    parser.add_argument("--data", default="data/eval_labels")
    parser.add_argument("--out", default="threshold_sweep.json")
    args = parser.parse_args()

    path = Path(args.data) / "val.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["qid"], []).append(row)

    evaluator = RetrievalEvaluator()

    # Score every pair once; thresholds are applied to cached probabilities, so
    # the sweep costs one forward pass rather than one per grid point.
    print(f"scoring {len(rows)} val pairs once...")
    cache: list[tuple[str, list[dict], list[str]]] = []
    for qid, group in by_question.items():
        chunks = [
            Chunk(
                chunk_id=r["chunk_id"],
                text=r["chunk_text"],
                doc_id=r["chunk_id"].rpartition("::")[0],
            )
            for r in group
        ]
        probs = evaluator.score_pairs(group[0]["question"], chunks)
        cache.append((qid, probs, [r["label"] for r in group]))

    gold_actions = [
        ACTION_TO_CRAG[derive_action([ChunkLabel(l) for l in labels])]
        for _, _, labels in cache
    ]
    majority_class, majority_count = Counter(gold_actions).most_common(1)[0]
    majority = majority_count / len(gold_actions)

    print(f"\n  val majority-class baseline: {majority:.4f} (always '{majority_class}')")
    print(f"  current config: correct>={config.EVALUATOR.correct_threshold} "
          f"wrong<={config.EVALUATOR.wrong_threshold}\n")
    print(f"  {'correct_thr':>12}{'wrong_thr':>11}{'chunk_acc':>11}"
          f"{'macro_F1':>10}{'action_acc':>12}{'vs_major':>10}")
    print("  " + "-" * 66)

    results = []
    from dataclasses import replace

    for correct_thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for wrong_thr in [0.05, 0.10, 0.20, 0.30]:
            if wrong_thr >= correct_thr:
                continue
            cfg = replace(
                config.EVALUATOR, correct_threshold=correct_thr, wrong_threshold=wrong_thr
            )
            scorer = RetrievalEvaluator(cfg)

            chunk_true: list[str] = []
            chunk_pred: list[str] = []
            action_pred: list[str] = []
            for _, probs, labels in cache:
                predicted = [scorer.label_from_probs(p)[0] for p in probs]
                chunk_true += labels
                chunk_pred += [l.value for l in predicted]
                action_pred.append(ACTION_TO_CRAG[derive_action(predicted)])

            chunk_acc = float(np.mean(np.array(chunk_true) == np.array(chunk_pred)))
            f1 = macro_f1(chunk_true, chunk_pred)
            action_acc = float(np.mean(np.array(gold_actions) == np.array(action_pred)))
            delta = action_acc - majority

            results.append(
                {
                    "correct_threshold": correct_thr,
                    "wrong_threshold": wrong_thr,
                    "chunk_accuracy": round(chunk_acc, 4),
                    "macro_f1": round(f1, 4),
                    "action_accuracy": round(action_acc, 4),
                    "vs_majority": round(delta, 4),
                }
            )
            flag = " *" if delta > 0 else ""
            print(f"  {correct_thr:>12.2f}{wrong_thr:>11.2f}{chunk_acc:>11.4f}"
                  f"{f1:>10.4f}{action_acc:>12.4f}{delta:>+10.4f}{flag}")

    best_action = max(results, key=lambda r: r["action_accuracy"])
    best_f1 = max(results, key=lambda r: r["macro_f1"])

    print("\n  best by ACTION ACCURACY (CRAG-comparable):")
    print(f"    correct>={best_action['correct_threshold']} "
          f"wrong<={best_action['wrong_threshold']}  ->  "
          f"action {best_action['action_accuracy']:.4f} "
          f"({best_action['vs_majority']:+.4f} vs majority), "
          f"macro-F1 {best_action['macro_f1']:.4f}")
    print("  best by MACRO F1 (keeps `ambiguous` honest):")
    print(f"    correct>={best_f1['correct_threshold']} "
          f"wrong<={best_f1['wrong_threshold']}  ->  "
          f"macro-F1 {best_f1['macro_f1']:.4f}, "
          f"action {best_f1['action_accuracy']:.4f} "
          f"({best_f1['vs_majority']:+.4f} vs majority)")

    if best_action["vs_majority"] <= 0:
        print("\n  NOTE: no threshold setting beats the majority-class baseline on val.")
        print("  That is a real result about this split, not a tuning failure.")

    payload = {
        "split": "val",
        "majority_baseline": round(majority, 4),
        "majority_class": majority_class,
        "n_questions": len(cache),
        "n_pairs": len(rows),
        "grid": results,
        "best_by_action_accuracy": best_action,
        "best_by_macro_f1": best_f1,
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    out = config.EVAL.results_dir / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nPut the chosen values in config.py, then run the test benchmark ONCE.")


if __name__ == "__main__":
    main()
