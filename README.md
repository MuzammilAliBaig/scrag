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
| A | `core/retriever.py` | Chunk, embed, store in FAISS | stub (Phase 1) |
| B | `core/evaluator.py` | Grade chunks; trigger corrective retrieval | stub (Phase 2) |
| C | `core/generator.py` | Generate answer, one citation per sentence | stub (Phase 3) |
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

## Build status

Phase 0 complete: `python -m app` starts, `pytest -q` passes 4/4, `/health` returns 200.
Next gate — **Phase 1**: an end-to-end answer on PopQA with a logged baseline (accuracy and RAGAS
faithfulness), using Module A plus a plain generator.

Phase prompts and the checklist live in `Major-Project/version 1/`.
