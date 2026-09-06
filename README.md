# SCRAG

**A Self-Correcting, Citation-Verified RAG Framework for Reliable Answers.**

A document-QA system that answers only from retrieved sources and refuses any claim it cannot back
with a verified citation.

Ordinary RAG fails in two specific ways, and the whole system exists to close both:

| Failure | Closed by |
|---|---|
| States facts that are not in the retrieved sources | Retrieval gating (Module B) + abstention (Module E) |
| Attaches citations that look valid but do not support the sentence | Citation forcing (Module C) + NLI verification (Module D) |

The measurable contribution is **Module B**, a fine-tuned retrieval evaluator replacing the CRAG
evaluator. Everything else reproduces published work.

---

## Modules

| Module | File | Responsibility | Status |
|---|---|---|---|
| A | `core/retriever.py` | Chunk, embed, store in FAISS | **built** (Phase 1) |
| B | `core/evaluator.py` | Grade chunks; trigger corrective retrieval | **built** (Phase 2) |
| C | `core/generator.py` | Generate answer, one citation per sentence | plain generator built; citations Phase 3 |
| D | `core/verifier.py` | NLI-check each cited chunk entails its sentence | stub (Phase 4) |
| E | `core/repair.py` | Regenerate, drop, or abstain | stub (Phase 5) |
| — | `core/orchestrator.py` | Wires A-E; flags drive the ablation variants | stub |

## Setup

Python **3.11** (the FAISS / torch / sentence-transformers wheels this project needs are reliable
there; 3.13+ is not).

```bash
py -3.11 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
cp .env.example .env            # then put a real ANTHROPIC_API_KEY in it
```

## Run

```bash
python -m app                   # http://127.0.0.1:8000
curl localhost:8000/health
pytest -q
```

`/health` returns status, version, embedding model, generator model, top-k, and whether an API key
is configured — never the key itself.

## Configuration

`config.py` is the single source of every tunable: model names, chunk size, top-k, and the
evaluator / verifier / abstention thresholds. **No magic numbers anywhere else.** Later phases add
their thresholds there rather than inline, which is what makes the Phase 6 ablation runner possible
without rewriting module code.

Thresholds currently marked `PLACEHOLDER` are set by measurement in Phases 2, 4 and 5.

## Secrets and cost

Every component is free except one: **the Claude API, billed per token with no free tier.** It is
the only thing in the pipeline that leaves your machine — retrieval, grading and verification all
run locally, so an abstention costs less than an answer.

- `ANTHROPIC_API_KEY` lives in `.env` locally (gitignored), GitHub Secrets in CI, host env vars in
  deploy. Never committed, never baked into a Docker image.
- Cost controls, all four required: prompt caching on the stable prefix, the Message Batches API
  (50% off) for every evaluation run, on-disk result caching, and `messages.count_tokens` before
  large sweeps.

## Repo layout

```
app/          FastAPI app (Streamlit UI lands in Phase 7)
core/         Modules A-E plus the orchestrator and shared types
train/        Evaluator labeling + fine-tuning scripts (Phase 2)
eval/         Benchmarks, ablation runner, metrics (Phase 6)
data/         Indexes and demo corpus (gitignored)
tests/        Unit tests per module
config.py     Every tunable
```

## Evaluation

PopQA ships questions but no corpus, so Module A indexes Wikipedia lead sections for each
question's subject entity plus a pool of unrelated entities as distractors. Without the
distractors retrieval is near-trivial and the baseline flatters itself.

```bash
python -m eval.build_index --split dev --limit 200   # free: Wikipedia + local CPU embedding
python -m eval.check_retrieval --split dev           # free: retrieval hit rate, no API key
python -m eval.run_baseline --split dev --limit 200  # PAID: calls the Claude API
```

The split is seeded (`SEED = 42`) and written to `data/splits/`. Every later phase evaluates
against these exact question ids; resampling invalidates the whole A-E ablation.

Faithfulness uses the **RAGAS definition** (claims entailed by context / total claims) with our
own implementation calling Claude directly. RAGAS the library hard-requires `openai`, `langchain`
and `langchain_openai`, which this project does not take on. Report it as
"RAGAS-definition faithfulness, own implementation" — it is not a RAGAS number.

## Module B — the contribution

Module B replaces CRAG's T5-large retrieval evaluator with a fine-tuned DeBERTa-v3-small
cross-encoder. It runs locally on CPU, so grading is free however many chunks are graded.

```bash
python -m train.build_dataset --out data/eval_labels   # free: weak labels from the index
python -m train.finetune_evaluator --epochs 3          # CPU ~90 min, free T4 ~3 min
python -m eval.run_evaluator_bench --split test        # free: the benchmark
```

**The rubric is frozen.** `train/LABELING.md` defines correct / ambiguous / wrong and was written
before any labeling. Revising it after seeing results would make the headline number
unfalsifiable; the fix for a genuine defect is a new dated version plus a full relabel.

**On the CRAG comparison.** CRAG reports **84.3%** (Yan et al. 2024, arXiv:2401.15884v3, Table 4,
PopQA, T5-large, 1,399-question test split). That figure is the accuracy of the *action* chosen for
a whole retrieved set (§5.5), **not** per-chunk grading accuracy. So the benchmark reports the two
separately, prints the majority-class baseline next to both, and lists every reason the comparison
is not like-for-like. Our action distribution is far more skewed than CRAG's, so beating 84.3 on
this split is not by itself a real claim.

**Labels are weak, not hand-labeled.** `train/weak_labels.py` derives them by rule, the same
approach CRAG used (PopQA's gold subject wiki title as the relevance signal). `train/label.py` and
`train/agreement.py` exist to measure how far that rule diverges from the rubric, via a
double-labeled human slice.

## Build status

**Phase 0 complete.** `python -m app` starts, `/health` returns 200.

**Phase 1 partially complete.** Module A is built and measured; generation is blocked on an API key.

Measured, from runs that actually executed:

| Measurement | Value |
|---|---|
| Corpus | 990 Wikipedia documents, 1,612 chunks |
| Retrieval hit rate @ k=5 | **0.870** |
| Top-1 hit rate | **0.800** |
| Index build | 340 s, CPU |
| Tests | 21 passed |

Not yet measured, because `ANTHROPIC_API_KEY` is unset: baseline answer accuracy, faithfulness,
and per-query cost. The Phase 1 gate is not met until those exist.

Phase prompts and the checklist live in `Major-Project/version 1/`.
