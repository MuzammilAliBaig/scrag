"""Human labeling CLI for (question, chunk) pairs.

    python -m train.label --annotator alice --n 100
    python -m train.label --annotator bob   --n 100 --same-sample-as alice

Samples pairs from the Phase 1 index, shows one at a time, and records a label
per train/LABELING.md. Resumable: re-running the same annotator continues where
they stopped.

Two rules enforced by the tool rather than left to discipline:

* The gold answer is **hidden** until after a label is entered (rubric section
  4.3). Seeing it first makes weak chunks look sufficient, and that is the
  largest single source of label drift.
* Sampling is stratified across the weak labels, so the `ambiguous` boundary -
  where grading is actually hard, and which is only ~7% of pairs - is not
  swamped by easy `wrong` cases.

`--same-sample-as` reproduces another annotator's exact sample so the two can be
compared by train/agreement.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import config
from core.retriever import Retriever
from core.types import ChunkLabel
from eval.datasets import load_split
from train.weak_labels import weak_label

LABEL_KEYS = {"c": ChunkLabel.CORRECT, "a": ChunkLabel.AMBIGUOUS, "w": ChunkLabel.WRONG}
LABELS_DIR = config.DATA_DIR / "human_labels"


def sample_pairs(
    retriever: Retriever,
    n: int,
    seed: int,
    split: str = "test",
    per_class: bool = True,
) -> list[dict]:
    """Stratified sample of (question, chunk) pairs, with weak labels attached."""
    examples = load_split(split)
    rng = random.Random(seed)
    rng.shuffle(examples)

    buckets: dict[str, list[dict]] = {"correct": [], "ambiguous": [], "wrong": []}
    for ex in examples:
        if all(len(v) >= n for v in buckets.values()):
            break
        for chunk in retriever.retrieve(ex.question):
            label = weak_label(ex, chunk)
            buckets[label.value].append(
                {
                    "qid": ex.qid,
                    "question": ex.question,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "gold_answers": ex.answers,
                    "weak_label": label.value,
                }
            )

    if per_class:
        take = max(n // 3, 1)
        pool = [row for rows in buckets.values() for row in rows[:take]]
    else:
        pool = [row for rows in buckets.values() for row in rows]

    rng.shuffle(pool)
    return pool[:n]


def load_existing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[f"{row['qid']}|{row['chunk_id']}"] = row["label"]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Hand-label chunks per train/LABELING.md")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--same-sample-as",
        default=None,
        help="reproduce another annotator's sample, for the agreement slice",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"{args.annotator}.jsonl"
    sample_seed = args.seed
    if args.same_sample_as:
        # Same seed reproduces the same sample; the label file differs.
        meta = LABELS_DIR / f"{args.same_sample_as}.meta.json"
        if meta.exists():
            sample_seed = json.loads(meta.read_text(encoding="utf-8"))["seed"]
        print(f"reproducing {args.same_sample_as}'s sample (seed {sample_seed})")

    retriever = Retriever()
    retriever.load_index()
    pairs = sample_pairs(retriever, args.n, sample_seed, args.split)
    (LABELS_DIR / f"{args.annotator}.meta.json").write_text(
        json.dumps({"seed": sample_seed, "n": args.n, "split": args.split}), encoding="utf-8"
    )

    done = load_existing(out_path)
    todo = [p for p in pairs if f"{p['qid']}|{p['chunk_id']}" not in done]

    print(f"\n{len(done)} already labeled, {len(todo)} remaining.")
    print("Read train/LABELING.md before starting. Keys: [c]orrect [a]mbiguous [w]rong "
          "[s]kip-and-quit\n")

    with out_path.open("a", encoding="utf-8") as fh:
        for i, pair in enumerate(todo, 1):
            print("=" * 72)
            print(f"  {i}/{len(todo)}   {pair['question']}")
            print("-" * 72)
            print(f"  {pair['chunk_text']}")
            print("-" * 72)

            choice = ""
            while choice not in LABEL_KEYS and choice != "s":
                choice = input("  label [c/a/w] (s to stop): ").strip().lower()
            if choice == "s":
                print("stopped - progress saved.")
                break

            label = LABEL_KEYS[choice]
            # Gold shown only AFTER the judgement, per rubric section 4.3.
            print(f"  (gold answers were: {', '.join(pair['gold_answers'])})")
            print(f"  (weak rule said: {pair['weak_label']})\n")

            fh.write(
                json.dumps(
                    {
                        "qid": pair["qid"],
                        "chunk_id": pair["chunk_id"],
                        "question": pair["question"],
                        "chunk_text": pair["chunk_text"],
                        "label": label.value,
                        "weak_label": pair["weak_label"],
                        "annotator": args.annotator,
                        "source": "human",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
