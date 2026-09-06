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
| C | `core/generator.py` | Generate answer, one citation per sentence | **built** (Phase 3) |
| D | `core/verifier.py` | NLI-check each cited chunk entails its sentence | **built** (Phase 4) |
| E | `core/repair.py` | Regenerate, drop, or abstain | **built** (Phase 5) |
| — | `core/orchestrator.py` | Wires A-E; flags drive the ablation variants | **complete** |

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

## Module C — citation-forced generation

Every sentence must cite a passage that was actually retrieved. Citations are 1-based **passage
numbers**, the ALCE convention: chunk ids like `Ada_Lovelace::3` are error-prone for a model to
reproduce, and one wrong character would read as an invented source rather than a typo.

```bash
python -m eval.run_citation_bench --dataset alce --limit 200 --dry-run  # price it first
python -m eval.run_citation_bench --dataset alce --limit 200            # PAID
```

Structured outputs (`output_config.format`) are the primary path, turning a parsing problem into a
schema problem. The text parser in `core/citation.py` remains as a fallback and the fallback rate
is reported — a schema guarantees *well-formed* citations, never *correct* ones.

**Three failure types, counted separately**, because merging them hides all three:

| Failure | Meaning | Measured in |
|---|---|---|
| Parse failure | no readable citation on the sentence | Phase 3 |
| Invalid id | citation names a passage never retrieved — an invented source | Phase 3 |
| Unsupported citation | citation is real but the passage does not support the sentence | **Phase 4** |

A sentence that cannot be attributed is **left flagged, never repaired by guessing** the nearest
chunk. Attaching a plausible source to an unsupported claim is the exact failure this project
exists to prevent.

## Module D — NLI citation verifier

The cited chunk is the **premise**, the sentence is the **hypothesis**. Runs locally on CPU, so
verification costs nothing however many sentences are checked.

```bash
python -m core.verifier --demo                  # per-sentence pass/fail on a worked example
python -m eval.calibrate_verifier --limit 200   # threshold sweep on a labeled slice
python -m eval.run_verifier_bench --limit 200   # flag rate on real answers
```

**Independence is the point.** The generator wrote the sentence *and* chose its citation, so
checking with the same model would be self-marking. Module D is a separate checkpoint —
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, chosen because FEVER is fact-verification against
Wikipedia evidence — and it is never the generator LLM.

**Neutral is not contradiction.** Neutral means the passage does not support the sentence;
contradiction means it says the opposite. Both fail the gate, but the logs keep them apart: neutral
usually means retrieval failed, contradiction means generation did.

**Label order is read from the checkpoint, never hardcoded.** This model is
(entailment, neutral, contradiction); the `cross-encoder/nli-*` family is
(contradiction, entailment, neutral). Hardcoding either silently inverts every verdict while
everything still appears to work.

**Multi-citation policy: `concat`** — all cited chunks are joined into one premise and must
*jointly* support the sentence, matching ALCE citation recall. Measured against the `any` policy the
difference was 0.2033 vs 0.2057 flag rate, i.e. immaterial on this data.

## Module E — repair and abstention

Three outcomes for a flagged sentence, in order: **repair** it (targeted regeneration of that one
sentence, then re-verification through Module D), **drop** it if the repair also fails, or
**abstain** on the whole answer.

```bash
python -m core.orchestrator --demo --query "What is Boulsa the capital of?"
python -m eval.run_abstention_bench --limit 150 --no-generation   # free: trigger (b) only
python -m eval.run_abstention_bench --limit 150                   # PAID: full risk-coverage
```

**Abstention is a visible product state, never an empty response.** The `FinalAnswer` carries
`abstained=True` plus a reason naming which of six triggers fired, so the API and UI render it as a
deliberate refusal.

**A repair never passes by fiat.** A regenerated sentence must pass Module D on the second look or
it is dropped. Bounded at one attempt — an unbounded loop burns budget and can oscillate between
two equally unsupported phrasings.

**The fragment guard.** If dropping sentences would leave a disconnected clause, the system abstains
instead. A stub is worse than a clean refusal.

**Trigger (b) costs nothing.** When Module B grades every retrieved chunk wrong, the pipeline stops
before the generator is called, so that abstention is free.

## The ablation — the report's key figure

```bash
python -m eval.run_ablation --all --split dev --limit 150 --dry-run  # price it first
python -m eval.run_ablation --all --split dev --limit 150            # PAID
python -m eval.plot_ablation
```

Variants are `PipelineFlags` over **one** `Orchestrator` (`eval/variants.py`). Five forked
pipelines drift, and drifted variants stop measuring what they claim to.

| Variant | Adds |
|---|---|
| A | Module A only — the Phase 1 baseline |
| B | + Module B grading and corrective retrieval |
| C | + Module C citation forcing |
| D | + Module D verification (**flags only — same text as C by construction**) |
| E | + Module E repair and abstention |

Generation goes through the **Message Batches API** at 50% pricing, keyed by `custom_id` because
batch results return in any order. `--dry-run` prices the sweep with `count_tokens` before anything
launches.

**Empty cells stay empty.** A baseline figure without a paper and a table behind it is left blank —
the ALCE citation cells are blank for exactly this reason, and a test asserts it.

## Running the app

```bash
uvicorn app.api:app --reload      # terminal 1
streamlit run app/ui.py           # terminal 2
```

Then click **Load demo corpus** in the sidebar, or upload your own PDF / .txt / .md.

| Endpoint | Does |
|---|---|
| `POST /ingest` | upload, chunk, embed, index — per-file status; one bad file never fails the batch |
| `POST /ingest/demo` | index the bundled demo corpus so the app is never empty |
| `POST /ask` | run Modules A–E, return the answer with per-sentence citations |
| `GET /health` | liveness plus index size and model readiness |

**An abstention is a 200 with `abstained: true`**, never an error code. Returning 4xx for a refusal
would train every client to treat the system working correctly as a fault.

**Quota exhaustion returns a readable 503**, not a stack trace — the failure most likely to happen
mid-demo. The message says what still works (retrieval, grading, verification) so a demo can carry
on.

**The app indexes into its own store** (`data/indexes/app.faiss`), separate from the evaluation
corpus. Uploading a document through the UI must not pollute the corpus every measured number was
computed against.

### A limitation worth knowing before you demo

Module B was fine-tuned on PopQA entity questions over Wikipedia lead sections. On documents unlike
those — policy text, contracts, uploaded PDFs — its grades are **out of distribution**. Measured on
the four demo documents it labels almost every chunk `ambiguous`, and returns `correct` for
"Who is the Vice-Chancellor?", which they never mention.

The consequence: the *free* pre-generation refusal (trigger b) does not fire reliably on
out-of-domain documents, so abstention falls to Modules C, D and E. The guarantee still holds — it
just costs a generation call instead of being free. On in-distribution questions trigger (b) fires
on 82.67% of unanswerable ones.

## Build status

**Phase 0 complete.** `python -m app` starts, `/health` returns 200.

**Phase 1 still open.** Module A is built and measured, but the baseline accuracy / faithfulness /
cost numbers have not been run.

**Phase 2 complete, gate NOT met.** Module B trained and benchmarked; see
`eval/results/phase2_summary.json`.

**Phase 3 complete, gate MET.**

Measured, from runs that actually executed:

| Measurement | Value |
|---|---|
| Corpus | 990 Wikipedia documents, 1,612 chunks |
| Retrieval hit rate @ k=5 | **0.870** |
| Top-1 hit rate | **0.800** |
| Index build | 340 s, CPU |
| Tests | 72 passed |
| **Module B** (test split, 152 questions) | |
| per-chunk grading accuracy | **0.9197** (macro-F1 0.8010) |
| query-level action accuracy | 0.8158 — below CRAG 0.843 and below the 0.8750 majority baseline |
| **Module C** (ALCE/ASQA, 200 questions, top-5) | |
| parseable-citation rate | **1.0000 — gate PASS** (397/397 sentences) |
| invalid-id rate | **0.0000** |
| structured-output path | 200/200, 0 fallbacks, 0 retries |
| abstention rate | 21/200 (10.5%) |
| ASQA str-EM | 0.4743 |
| measured cost | $2.18 for 200 questions ($0.0109 each) |
| **Module D** (same 200 answers, 418 sentences) | |
| verdicts issued | **418/418 — gate PASS** |
| calibrated threshold | **0.20** (P 0.9310 / R 0.7980 / F1 0.8594 on a 600-pair slice) |
| positive/negative separation | +0.6604 |
| flag rate on real answers | **0.2033** (85 sentences) |
| of which contradictions | 27 (6.5%) — generation failures |
| **Citation P/R** (entailment, 397 sentences, free) | |
| citation recall | **0.8388** |
| citation precision | **0.7409** |
| mean citations per sentence | 1.31 |
| hallucination rate (auto, verifier-derived) | 0.1612 |
| neutral | 83 (19.9%) — mostly retrieval failures |
| **Module E** (PopQA dev, 150+150, retrieval+grading only) | |
| trigger (b) on *unanswerable* questions | **0.8267** (124/150) — caught free, before generation |
| trigger (b) on *answerable* questions | 0.0267 (4/150) — not over-abstaining |
| corrective loop, answerable | 0.1733 |
| Tests | **152 passed** |

**Blocked on API credit.** The account ran out mid-Phase-5:
`400 invalid_request_error: Your credit balance is too low`. Still unmeasured as a result:

- the **Phase 1 baseline** (accuracy, faithfulness, per-query cost),
- Phase 5's **full risk-coverage curve** and answered-subset accuracy.

Everything that runs locally is measured. Add credit, then:
`python -m eval.run_baseline --split dev --limit 150` and
`python -m eval.run_abstention_bench --limit 150`.

**Prompt caching, measured:** the citation prompt prefix is 758 tokens and *does* cache
(227,810 cache-read tokens over the run). The Phase 1 plain prompt is 476 tokens — below Opus 5's
512-token minimum — so it does **not** cache. Failure is silent, which is why this is measured
rather than assumed.

Phase prompts and the checklist live in `Major-Project/version 1/`.
