"""Measure ingest throughput and project it to a target workload.

    python -m eval.bench_ingest --docs 25 --pages 500
    python -m eval.bench_ingest --docs 25 --pages 500 --sample 3000

Free and local. Chunks a synthetic corpus of the requested size, embeds a
sample, and projects the full wall clock from the measured rate.

**Chunking is not the bottleneck and never was.** Splitting 47 MB of text into
70,000 chunks takes about 0.2 seconds. Embedding those chunks takes minutes.
Any claim about "ingest latency" that quotes the chunking number is measuring
the wrong thing, so this script reports the two separately and always shows the
embedding total.

Two levers actually move the wall clock on CPU:

  --bulk      coarser chunks (2000/240 instead of 800/120) - about 2.6x fewer
              chunks, so about 2.6x less embedding, at some cost in retrieval
              precision
  --workers   embed across processes; transformer inference on CPU scales
              better across processes than across BLAS threads

Measured and rejected: larger batch sizes, a lower max_seq_length, and int8
dynamic quantisation all left throughput unchanged at ~41-52 chunks/s.
sentence-transformers pads to the longest item in each batch, so lowering
max_seq_length buys nothing when chunks are already ~200 tokens.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace

import config
from core.retriever import Retriever

WORDS = (
    "policy student refund tuition library borrowing examination candidate "
    "accommodation deposit contract regulation university faculty deadline appeal "
    "committee evidence semester enrolment invigilator submission penalty campus"
).split()

CHARS_PER_PAGE = 2400


def synthetic_document(pages: int, seed: int) -> str:
    rng = random.Random(seed)
    words_per_page = CHARS_PER_PAGE // 6
    return "\n\n".join(
        " ".join(rng.choice(WORDS) for _ in range(words_per_page)) for _ in range(pages)
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Ingest throughput benchmark")
    parser.add_argument("--docs", type=int, default=25)
    parser.add_argument("--pages", type=int, default=500)
    parser.add_argument("--sample", type=int, default=1500,
                        help="chunks actually embedded to measure the rate")
    parser.add_argument("--bulk", action="store_true", help="use coarse bulk chunking")
    parser.add_argument("--workers", type=int, default=None,
                        help="embedding worker processes (default: config/auto)")
    parser.add_argument("--target-minutes", type=float, default=1.5)
    args = parser.parse_args()

    cfg = config.RETRIEVAL
    if args.workers is not None:
        cfg = replace(cfg, embed_workers=args.workers)
    retriever = Retriever(cfg)

    docs = {f"doc_{i:02d}": synthetic_document(args.pages, seed=i) for i in range(args.docs)}
    total_chars = sum(len(v) for v in docs.values())

    print("=" * 72)
    print("  INGEST BENCHMARK   (MEASURED)")
    print("=" * 72)
    print(f"  workload      {args.docs} documents x {args.pages} pages "
          f"= {total_chars/1e6:.1f} M characters")
    print(f"  profile       {'BULK 2000/240' if args.bulk else 'default 800/120'}")

    chunker = retriever.chunk_bulk if args.bulk else retriever.chunk
    started = time.time()
    chunks = [c for doc_id, text in docs.items() for c in chunker(text, doc_id)]
    chunk_seconds = time.time() - started
    print(f"\n  CHUNKING      {chunk_seconds:.2f} s -> {len(chunks):,} chunks "
          f"({len(chunks)/max(chunk_seconds,1e-6):,.0f} chunks/s)")

    sample = [c.text for c in chunks[: args.sample]]
    workers = retriever._worker_count(len(chunks))
    print(f"  EMBEDDING     measuring on {len(sample):,} chunks "
          f"with {workers} worker process(es)...")

    # Warm the model so the measurement excludes load time.
    retriever.embed(sample[:32])
    started = time.time()
    if workers > 1:
        retriever.embed_parallel(sample)
    else:
        retriever.embed(sample)
    embed_seconds = time.time() - started
    rate = len(sample) / embed_seconds

    projected = len(chunks) / rate
    total = projected + chunk_seconds
    print(f"                {embed_seconds:.1f} s for {len(sample):,} chunks "
          f"-> {rate:,.0f} chunks/s")

    print(f"\n  PROJECTED TOTAL FOR THE FULL WORKLOAD")
    print(f"    chunking     {chunk_seconds:6.2f} s")
    print(f"    embedding    {projected:6.1f} s   ({projected/60:.1f} min)")
    print(f"    TOTAL        {total:6.1f} s   ({total/60:.1f} min)")

    target = args.target_minutes * 60
    print(f"\n  TARGET        {args.target_minutes} min ({target:.0f} s)")
    if total <= target:
        print(f"  VERDICT       MET - {total:.0f} s is within target")
    else:
        needed = len(chunks) / max(target - chunk_seconds, 1)
        print(f"  VERDICT       NOT MET - {total/60:.1f} min against a "
              f"{args.target_minutes} min target")
        print(f"                would need {needed:,.0f} chunks/s "
              f"({needed/rate:.1f}x the measured rate)")
        print(f"                or at most {int(rate*(target-chunk_seconds)):,} chunks "
              f"(this workload has {len(chunks):,})")
    print("=" * 72)


if __name__ == "__main__":
    main()
