"""Module A quality check. Local, free, no API key required.

    python -m eval.check_retrieval --split dev --limit 200

Reports retrieval hit rate: the fraction of questions whose gold answer appears
somewhere in the top-k retrieved chunks. This is the ceiling on what any
generator could answer from context alone, so it bounds the whole pipeline -
and the gap between it and the eventual baseline accuracy tells you how often
the model answered from memory instead of from the passages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config
from core.retriever import Retriever
from eval import metrics
from eval.datasets import load_split


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Retrieval-only evaluation (no API cost)")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=config.EVAL.default_limit)
    parser.add_argument("--k", type=int, default=config.RETRIEVAL.top_k)
    parser.add_argument("--out", default="retrieval_check.json")
    args = parser.parse_args()

    retriever = Retriever()
    retriever.load_index()
    examples = load_split(args.split, limit=args.limit)

    started = time.time()
    hits = 0
    top1_hits = 0
    rows = []
    for ex in examples:
        context = retriever.retrieve(ex.question, k=args.k)
        hit = metrics.retrieval_hit(ex.answers, context)
        top1 = metrics.retrieval_hit(ex.answers, context[:1])
        hits += hit
        top1_hits += top1
        rows.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "hit": hit,
                "top1_hit": top1,
                "top_chunk": context[0].chunk_id if context else None,
                "top_score": round(context[0].score or 0.0, 4) if context else None,
            }
        )

    n = len(examples)
    stats = {
        "split": args.split,
        "n": n,
        "k": args.k,
        "retrieval_hit_rate": round(hits / max(n, 1), 4),
        "top1_hit_rate": round(top1_hits / max(n, 1), 4),
        "corpus_chunks": retriever.size,
        "embedding_model": config.RETRIEVAL.embedding_model,
        "elapsed_s": round(time.time() - started, 1),
    }

    print(json.dumps(stats, indent=2))
    print(f"\n  hit rate @k={args.k}: {stats['retrieval_hit_rate']:.3f}"
          f"   (top-1: {stats['top1_hit_rate']:.3f})")
    print("  This is the ceiling on baseline accuracy from context alone.")

    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps({**stats, "examples": rows}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
