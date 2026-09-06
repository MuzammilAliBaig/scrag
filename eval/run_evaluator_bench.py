"""Module B benchmark: grading accuracy, per-class P/R/F1, confusion, action accuracy.

    python -m eval.run_evaluator_bench --split test

Free and local - no API calls.

**On comparing to CRAG.** CRAG reports 84.3% (Yan et al. 2024, Table 4) for its
T5-large retrieval evaluator on PopQA. That number is the accuracy of the
*action* chosen for a whole retrieved set, not per-chunk grading accuracy
(section 5.5). So this script reports two different things and keeps them apart:

* **Per-chunk** accuracy and per-class F1 - what the pipeline actually uses.
  CRAG published no per-chunk figure, so there is nothing to compare against.
* **Query-level action** accuracy - the only quantity comparable to 84.3, via
  the frozen derivation rule in train/LABELING.md section 5.

Even the action comparison is not like-for-like, and the script prints the
reasons every run so they cannot be quietly dropped from the report. The most
important one is the **majority-class baseline**: our action distribution is
heavily skewed toward Correct, so a predictor that always says Correct already
scores near CRAG's number. Any claim to "beat CRAG" that does not also beat the
majority baseline by a clear margin is meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import config
from core.evaluator import RetrievalEvaluator
from core.types import Chunk, ChunkLabel
from train.weak_labels import ACTION_TO_CRAG, derive_action

LABELS = [ChunkLabel.CORRECT.value, ChunkLabel.AMBIGUOUS.value, ChunkLabel.WRONG.value]

# --- The published baseline, with everything needed to cite it -------------
CRAG_BASELINE = {
    "value": 84.3,
    "metric": "accuracy of the retrieval action chosen for a retrieved set",
    "paper": "Yan et al. 2024, Corrective Retrieval Augmented Generation",
    "arxiv": "2401.15884v3",
    "table": "Table 4",
    "section": "5.5",
    "dataset": "PopQA",
    "model": "T5-large (0.77B), fine-tuned",
    "test_size": 1399,
    "retriever": "Contriever, ~10 documents per question (Self-RAG's retrieval results)",
    "also_in_table": {
        "ChatGPT": 58.0,
        "ChatGPT-CoT": 62.4,
        "ChatGPT-few-shot": 64.7,
    },
}

NOT_LIKE_FOR_LIKE = [
    "Different corpus. CRAG retrieves over full Wikipedia via Contriever with ~10 "
    "documents per question; SCRAG retrieves over a 990-document corpus of Wikipedia "
    "lead sections with top-5. A smaller corpus makes grading easier.",
    "Different test set. CRAG uses Self-RAG's 1,399-question PopQA test split; SCRAG "
    "uses its own seeded split (seed 42) and only questions whose subject page is in "
    "the corpus.",
    "Different ground truth. CRAG never published a per-chunk labeling rubric; our "
    "labels come from train/LABELING.md, approximated by the weak rule in "
    "train/weak_labels.py. Both sides are 'accuracy against our own labels'.",
    "Weak supervision. Test labels here are rule-derived, not hand-labeled. The "
    "human-labeled slice (train/agreement.py) is what bounds how much that costs.",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} - run `python -m train.build_dataset` first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def per_class_metrics(true: list[str], pred: list[str]) -> dict:
    out = {}
    t, p = np.array(true), np.array(pred)
    for name in LABELS:
        tp = int(((p == name) & (t == name)).sum())
        fp = int(((p == name) & (t != name)).sum())
        fn = int(((p != name) & (t == name)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((t == name).sum()),
        }
    return out


def confusion(true: list[str], pred: list[str]) -> dict:
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    for a, b in zip(true, pred):
        matrix[a][b] += 1
    return matrix


def print_confusion(matrix: dict, title: str, classes: list[str]) -> None:
    print(f"\n  {title}  (rows = gold, cols = predicted)")
    print("               " + "".join(f"{c:>12}" for c in classes))
    for a in classes:
        print(f"  {a:>11}  " + "".join(f"{matrix[a][b]:>12}" for b in classes))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Benchmark Module B")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--data", default="data/eval_labels")
    parser.add_argument("--out", default="evaluator_bench.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.data) / f"{args.split}.jsonl")
    evaluator = RetrievalEvaluator()

    # Group by question so the query-level action can be derived.
    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["qid"], []).append(row)

    started = time.time()
    chunk_true: list[str] = []
    chunk_pred: list[str] = []
    action_true: list[str] = []
    action_pred: list[str] = []

    for i, (qid, group) in enumerate(by_question.items(), 1):
        question = group[0]["question"]
        chunks = [
            Chunk(chunk_id=r["chunk_id"], text=r["chunk_text"], doc_id=r["chunk_id"].rpartition("::")[0])
            for r in group
        ]
        result = evaluator.grade(question, chunks)

        gold_labels = [ChunkLabel(r["label"]) for r in group]
        pred_labels = [g.label for g in result.graded]
        chunk_true += [l.value for l in gold_labels]
        chunk_pred += [l.value for l in pred_labels]

        action_true.append(ACTION_TO_CRAG[derive_action(gold_labels)])
        action_pred.append(ACTION_TO_CRAG[result.action])

        if i % 25 == 0:
            print(f"  graded {i}/{len(by_question)} questions", flush=True)

    elapsed = time.time() - started

    # --- per-chunk ---------------------------------------------------------
    chunk_acc = float(np.mean(np.array(chunk_true) == np.array(chunk_pred)))
    chunk_metrics = per_class_metrics(chunk_true, chunk_pred)
    macro_f1 = float(np.mean([m["f1"] for m in chunk_metrics.values()]))

    # --- query-level action ------------------------------------------------
    action_classes = ["Correct", "Ambiguous", "Incorrect"]
    action_acc = float(np.mean(np.array(action_true) == np.array(action_pred)))
    action_matrix = {a: {b: 0 for b in action_classes} for a in action_classes}
    for a, b in zip(action_true, action_pred):
        action_matrix[a][b] += 1

    # The baseline that makes or breaks the comparison.
    majority_class, majority_count = Counter(action_true).most_common(1)[0]
    majority_baseline = majority_count / len(action_true)
    corrective_rate = sum(1 for a in action_pred if a == "Ambiguous") / len(action_pred)

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  MODULE B BENCHMARK - {args.split} split   (MEASURED)")
    print("=" * 72)
    print(f"  questions {len(by_question)}   pairs {len(rows)}   elapsed {elapsed:.1f}s")
    print(f"  checkpoint {config.EVALUATOR.checkpoint_path}")
    print(f"  thresholds  correct>={config.EVALUATOR.correct_threshold}  "
          f"wrong<={config.EVALUATOR.wrong_threshold}")

    print(f"\n  --- PER-CHUNK (what the pipeline uses; no published CRAG counterpart) ---")
    print(f"  grading accuracy   {chunk_acc:.4f}")
    print(f"  macro F1           {macro_f1:.4f}")
    for name, m in chunk_metrics.items():
        print(f"    {name:>10}  P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}  n={m['support']}")
    print_confusion(confusion(chunk_true, chunk_pred), "per-chunk confusion", LABELS)

    print(f"\n  --- QUERY-LEVEL ACTION (the only CRAG-comparable number) ---")
    print(f"  action accuracy         {action_acc:.4f}  ({action_acc * 100:.1f}%)")
    print(f"  majority-class baseline {majority_baseline:.4f}  "
          f"({majority_baseline * 100:.1f}%, always predicting '{majority_class}')")
    print(f"  CRAG reported           {CRAG_BASELINE['value'] / 100:.4f}  "
          f"({CRAG_BASELINE['value']}%)  [{CRAG_BASELINE['table']}, {CRAG_BASELINE['arxiv']}]")
    print(f"  corrective-loop trigger rate  {corrective_rate:.4f}")
    print_confusion(action_matrix, "action confusion", action_classes)

    print("\n  --- HOW TO READ THIS ---")
    if action_acc <= majority_baseline:
        print(f"  The evaluator does NOT beat the majority-class baseline "
              f"({action_acc:.3f} vs {majority_baseline:.3f}).")
        print("  Any comparison to CRAG is meaningless until it does.")
    elif majority_baseline >= CRAG_BASELINE["value"] / 100:
        print(f"  NOTE: the majority-class baseline ({majority_baseline * 100:.1f}%) already")
        print(f"  exceeds CRAG's {CRAG_BASELINE['value']}%. Our action distribution is more")
        print("  skewed than theirs, so 'beats CRAG' on this split is NOT a real claim.")
        print(f"  The honest comparison is against the majority baseline: "
              f"{(action_acc - majority_baseline) * 100:+.1f} points.")
    else:
        print(f"  Beats majority baseline by {(action_acc - majority_baseline) * 100:+.1f} points "
              f"and CRAG's figure by {(action_acc * 100 - CRAG_BASELINE['value']):+.1f} points.")

    print("\n  Not like-for-like with CRAG, for these reasons:")
    for reason in NOT_LIKE_FOR_LIKE:
        print(f"    - {reason}")
    print("=" * 72)

    payload = {
        "split": args.split,
        "questions": len(by_question),
        "pairs": len(rows),
        "elapsed_s": round(elapsed, 1),
        "per_chunk": {
            "grading_accuracy": round(chunk_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": chunk_metrics,
            "confusion": confusion(chunk_true, chunk_pred),
        },
        "query_level_action": {
            "accuracy": round(action_acc, 4),
            "majority_class_baseline": round(majority_baseline, 4),
            "majority_class": majority_class,
            "beats_majority_baseline": bool(action_acc > majority_baseline),
            "corrective_trigger_rate": round(corrective_rate, 4),
            "confusion": action_matrix,
        },
        "crag_baseline": CRAG_BASELINE,
        "not_like_for_like": NOT_LIKE_FOR_LIKE,
        "config": {
            "checkpoint": str(config.EVALUATOR.checkpoint_path),
            "base_model": config.EVALUATOR.base_model,
            "correct_threshold": config.EVALUATOR.correct_threshold,
            "wrong_threshold": config.EVALUATOR.wrong_threshold,
            "max_length": config.EVALUATOR.max_length,
            "seed": config.SEED,
        },
        "labels_are": "weak (rule-derived), not hand-labeled",
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
