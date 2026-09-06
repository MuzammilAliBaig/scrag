"""One-shot corpus fetch + FAISS index build for Module A.

    python -m eval.build_index --limit 200

Costs nothing: Wikipedia is free and embedding runs locally on CPU. Safe to
re-run - the corpus fetch resumes and embeddings are cached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config
from core.retriever import Retriever
from eval import corpus as corpus_mod
from eval.datasets import build_splits, load_split


def main() -> None:
    # Article text is not ASCII; a cp1252 console would crash on printing it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build the SCRAG corpus and FAISS index")
    parser.add_argument("--split", default="dev", help="split whose subjects must be covered")
    parser.add_argument("--limit", type=int, default=None, help="cap questions from that split")
    parser.add_argument(
        "--distractors",
        type=int,
        default=config.CORPUS.distractor_pool_size,
        help="unrelated entities added as retrieval noise",
    )
    args = parser.parse_args()

    started = time.time()
    splits = build_splits()
    print(f"splits: dev={len(splits['dev'])} test={len(splits['test'])} (seed {config.SEED})")

    # Titles the eval questions are actually about.
    examples = load_split(args.split, limit=args.limit)
    titles = [ex.subject_title for ex in examples if ex.subject_title]
    print(f"{len(examples)} questions from split '{args.split}' -> {len(set(titles))} subject pages")

    # Distractor entities, drawn from questions never evaluated on.
    if args.distractors:
        distractor_ids = set(splits.get("distractor", []))
        pool: list[str] = []
        for ex in load_split("test"):
            if ex.qid in distractor_ids or len(pool) >= args.distractors:
                continue
            if ex.subject_title:
                pool.append(ex.subject_title)
            if len(pool) >= args.distractors:
                break
        titles.extend(pool)
        print(f"+ {len(set(pool))} distractor pages")

    docs = corpus_mod.build_corpus(titles)
    print(f"corpus: {len(docs)} documents at {config.CORPUS.corpus_path}")

    retriever = Retriever()
    print(f"embedding with {config.RETRIEVAL.embedding_model} on CPU...")
    retriever.build_index(docs)
    print(f"index: {retriever.size} chunks at {config.RETRIEVAL.index_path}")

    # Sanity check: a real question should retrieve something about its subject.
    probe = examples[0]
    hits = retriever.retrieve(probe.question)
    print(f"\nprobe: {probe.question}")
    for hit in hits[:3]:
        print(f"  {hit.score:.3f}  {hit.chunk_id}  {hit.text[:70]}...")

    stats = {
        "documents": len(docs),
        "chunks": retriever.size,
        "embedding_model": config.RETRIEVAL.embedding_model,
        "chunk_size": config.RETRIEVAL.chunk_size,
        "chunk_overlap": config.RETRIEVAL.chunk_overlap,
        "seed": config.SEED,
        "elapsed_s": round(time.time() - started, 1),
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    (config.EVAL.results_dir / "index_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
