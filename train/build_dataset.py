"""Assemble the Module B training set with a held-out test split.

    python -m train.build_dataset --out data/eval_labels

Free and local: retrieval over the Phase 1 FAISS index plus the weak labeling
rule. No API calls.

Leakage discipline, which matters more here than anywhere else in the project:

* The evaluator is trained on questions from the **test** split pool, whose
  subject pages are the distractor entities already in the corpus.
* The **dev** split is never touched. Phase 6 runs the A-E ablation on dev, so
  an evaluator trained on dev questions would inflate every downstream number.
* The evaluator's own held-out slice is carved from the training pool by
  question, never by pair, so no question appears in two splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import config
from core.retriever import Retriever
from eval.datasets import load_split
from train.weak_labels import LabeledPair, derive_action, weak_label


def build_alias_map(retriever: Retriever) -> dict[str, str]:
    """Map lowercased corpus titles to their canonical doc_id.

    Wikipedia resolves redirects on fetch, so a question's s_wiki_title may not
    equal the title the article was filed under.
    """
    aliases: dict[str, str] = {}
    for chunk in retriever._chunks:
        aliases.setdefault(chunk.doc_id.replace("_", " ").lower(), chunk.doc_id)
    return aliases


def build_pairs(
    retriever: Retriever,
    split: str,
    limit: int | None,
    k: int,
) -> tuple[list[LabeledPair], dict]:
    """Retrieve for each question and weak-label every returned chunk."""
    examples = load_split(split, limit=limit)
    corpus_titles = build_alias_map(retriever)

    pairs: list[LabeledPair] = []
    actions = Counter()
    covered = 0

    for ex in examples:
        # Only keep questions whose subject page is actually in the corpus.
        # Otherwise every chunk is trivially `wrong` and the model learns
        # "absent from corpus" rather than "irrelevant to the question".
        key = (ex.subject_title or "").lower()
        if key not in corpus_titles:
            continue
        covered += 1

        chunks = retriever.retrieve(ex.question, k=k)
        labels = [weak_label(ex, c) for c in chunks]
        actions[derive_action(labels).value] += 1
        pairs.extend(
            LabeledPair(
                qid=ex.qid,
                question=ex.question,
                chunk_id=c.chunk_id,
                chunk_text=c.text,
                label=label,
            )
            for c, label in zip(chunks, labels)
        )

    stats = {
        "questions_seen": len(examples),
        "questions_with_subject_in_corpus": covered,
        "pairs": len(pairs),
        "label_distribution": dict(Counter(p.label.value for p in pairs)),
        "action_distribution": dict(actions),
    }
    return pairs, stats


def split_by_question(
    pairs: list[LabeledPair],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, list[LabeledPair]]:
    """Split by question id, so no question straddles two splits."""
    qids = sorted({p.qid for p in pairs})
    random.Random(seed).shuffle(qids)

    n = len(qids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    assignment = {}
    for i, qid in enumerate(qids):
        if i < n_train:
            assignment[qid] = "train"
        elif i < n_train + n_val:
            assignment[qid] = "val"
        else:
            assignment[qid] = "test"

    out: dict[str, list[LabeledPair]] = {"train": [], "val": [], "test": []}
    for pair in pairs:
        out[assignment[pair.qid]].append(pair)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build the Module B training set")
    parser.add_argument("--out", default="data/eval_labels")
    parser.add_argument(
        "--split",
        default="test",
        help="PopQA split to draw TRAINING questions from. Never 'dev' - Phase 6 evaluates there.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=config.RETRIEVAL.top_k)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    if args.split == "dev":
        print(
            "REFUSING: 'dev' is the Phase 6 ablation split. Training Module B on it\n"
            "would leak into every downstream number. Use --split test.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    retriever = Retriever()
    retriever.load_index()
    print(f"index: {retriever.size} chunks")

    pairs, stats = build_pairs(retriever, args.split, args.limit, args.k)
    if not pairs:
        print("no pairs built - is the corpus covering this split?", file=sys.stderr)
        raise SystemExit(1)

    splits = split_by_question(pairs, config.SEED, args.train_frac, args.val_frac)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for pair in rows:
                fh.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")
        dist = Counter(p.label.value for p in rows)
        print(f"  {name}: {len(rows)} pairs, {len({p.qid for p in rows})} questions, {dict(dist)}")

    stats["splits"] = {k: len(v) for k, v in splits.items()}
    stats["seed"] = config.SEED
    stats["source_split"] = args.split
    stats["k"] = args.k
    stats["labeling"] = "weak (train/weak_labels.py), NOT hand-labeled"
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n{json.dumps(stats, indent=2)}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
