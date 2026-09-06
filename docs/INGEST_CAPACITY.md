# Ingest capacity and latency

All figures measured on the development machine: **6 physical cores (12 logical), no GPU**,
`BAAI/bge-small-en-v1.5` on CPU. Reproduce with:

```bash
python -m eval.bench_ingest --docs 25 --pages 500
```

---

## The headline

**Chunking is not the bottleneck and never was.** Splitting 47 MB of text into 69,335 chunks takes
**0.18 seconds** — about 380,000 chunks/second. Embedding those chunks takes **26 minutes**.

Any statement about "chunking latency" that quotes the fast number is measuring the wrong stage.
The numbers below are for the whole ingest: chunk → embed → index.

## Measured throughput

| Configuration | Chunks | Rate | 25 × 500 pages |
|---|---|---|---|
| Default (800/120), 1 process | 69,335 | 45 chunks/s | **25.7 min** |
| Bulk (2000/240), 1 process | 26,557 | 21 chunks/s | **21.2 min** |
| Bulk (2000/240), 6 processes | 26,557 | 22 chunks/s | 19.8 min |
| Default, 6 processes | 69,335 | 19 chunks/s | worse |

### Why coarser chunks barely help

2.6× fewer chunks gives only a 1.2× speedup, because **embedding cost scales with tokens, not
chunks**. A 2000-character chunk is ~500 tokens; an 800-character chunk is ~200. The corpus fixes
the total token count at roughly 13–14 M either way.

The only chunking-side lever that removes real work is the **overlap ratio** — 15% at 800/120
versus 12% at 2000/240. That difference is the entire 1.2×.

### Why multi-process embedding is off by default

Six worker processes measured **slower** than one (22 vs 45 chunks/s). They contend for the same six
physical cores that torch already saturates through BLAS threads, and the pool costs ~30 s to start.
The code path is kept (`embed_workers`) because it can win on a machine with many more cores, but
the default is `1` and this measurement is why.

### What was tried and rejected

| Change | Result |
|---|---|
| Batch size 64 → 256 | no change (46 vs 52 chunks/s) |
| `max_seq_length` 512 → 256 → 192 | **no change** |
| int8 dynamic quantisation | no change (0.9×) |
| `all-MiniLM-L6-v2` instead of bge-small | 1.9× faster, at a cost in retrieval quality |

`max_seq_length` buys nothing because sentence-transformers pads to the longest item **in each
batch**, not to the configured maximum, and chunks are already ~200 tokens.

---

## Capacity: what fits in a given time budget

At the measured **45 chunks/s** (default profile, ~9,000 tokens/s):

| Budget | Chunks | Roughly |
|---|---|---|
| **1.5 min** | ~4,000 | **2 documents × 500 pages** |
| 5 min | ~13,500 | 5 documents × 500 pages |
| 15 min | ~40,000 | 14 documents × 500 pages |
| 26 min | ~69,000 | **25 documents × 500 pages** |

Hard limits now enforced by the API (raised this change):

| Limit | Was | Now |
|---|---|---|
| Bytes per file | 10 MB | **150 MB** |
| Files per request | unlimited | **50** |
| Bytes per request | unlimited | **2 GB** |

The old 10 MB per-file cap silently **skipped** exactly the documents this is meant to handle — a
500-page PDF is typically 5–50 MB.

---

## The 1.5-minute target

**Not achievable for 25 × 500 pages on this CPU.** It needs ~295 chunks/s against a measured 21–45,
i.e. **6–14× more throughput**. No CPU-side configuration change closes a gap that size; the ones
that looked promising were measured and did not.

Three ways to actually get there:

**1. A GPU — the only one that reaches the target with this corpus.**
`config.RETRIEVAL.device` is now `"auto"` and selects CUDA when present. bge-small on a modest GPU
runs 1–2 orders of magnitude faster than 9,000 tokens/s, which puts 13–14 M tokens in the
low minutes or below. **This is the recommended path, and it is a one-line config change on a
CUDA box — no code change.** The figure is not quoted here because it has not been measured on this
hardware; run `eval/bench_ingest.py` on the target machine.

**2. Ingest asynchronously.** Accept the upload, return a job id, embed in the background, and let
`/ready` report progress. The *response* is then fast and the index warms behind it. This changes
the product rather than the throughput, and is the honest answer for a CPU deployment.

**3. Reduce the corpus.** 1.5 minutes buys ~4,000 chunks. If the real requirement is "25 documents",
and those documents are 40 pages rather than 500, it already fits.

---

## What did change

**Re-ingesting a document is now 57.8× faster.** Previously, replacing one document re-embedded
every surviving chunk in the index — O(corpus) per upload. At 70,000 chunks that is over 20 minutes
to correct a typo in one file. Vectors are now persisted alongside the index
(`corpus.vectors.npy`) and survivors are reused.

Measured on an 8-document index: initial ingest 15.7 s, replacing one document **0.3 s**.

This is the largest real improvement in this change, and it affects the operation users perform
most often.
