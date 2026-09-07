# SCRAG

**A self-correcting, citation-verified RAG framework for reliable answers.**

---

## Description

SCRAG is a document question-answering system that answers only from the documents you give it, and
refuses when it can't.

That second half is the point. Most RAG systems will answer anything — and when the retrieved
passages don't contain the answer, they quietly make one up. Adding citations doesn't fix it,
because a citation that *looks* right is worse than none at all: it makes an invented claim look
checked.

So SCRAG treats those as two separate problems and measures them separately:

| The failure | What it looks like | How it's closed |
|---|---|---|
| **Ungrounded claim** | states something that isn't in the sources | grade the retrieval, then abstain |
| **Unsupported citation** | cites a passage that doesn't back the sentence | force a citation per sentence, then verify it by entailment |

The gap between "has a citation" and "has a *correct* citation" turned out to be large and
measurable. A word-overlap check scores our citations at **0.9963**. Actual entailment scores
citation precision at **0.7409** — about a quarter of individual citations don't support the
sentence they're attached to. Only the second number can see that.

Every sentence the system produces carries a citation you can click open, and every citation has
been checked by a separate model that never saw the sentence being written.

---

## Technologies

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | small enough to run on CPU, good enough to retrieve well |
| Vector store | FAISS | a local library, not a service — nothing to pay for or host |
| Retrieval evaluator | DeBERTa-v3-small, fine-tuned | the one component trained from scratch here |
| Verifier | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | trained on FEVER, which is fact-checking against Wikipedia — exactly this task |
| Generator | **Claude API** (`claude-opus-5`) | the only paid part of the whole project |
| Backend | FastAPI + Uvicorn | |
| Frontend | One static HTML file | no framework, no build step; FastAPI serves it itself |
| Packaging | Docker | multi-stage, CPU-only wheels, 3.6 GB |
| Frontend hosting | Vercel | static page only - 914 MB of Python cannot fit a serverless function |
| CI | GitHub Actions | with a regression gate that actually fails the build |
| Datasets | PopQA, ALCE (ASQA) | public, standard, and free |

Everything except generation runs on your own machine. That's a deliberate design property, not an
accident — it means verification is free, so there's never a reason to check less.

---

## Features

**Answers only from your documents.** Upload PDFs, text or markdown; ask questions; get answers
built solely from what you uploaded.

**A citation on every sentence.** Not one at the end of a paragraph — one per sentence, and you can
click it to see the exact passage it came from. Measured at **100% of sentences** carrying a
citation that resolves to a real retrieved passage, with **zero** invented sources across 397
sentences.

**Citations that are verified, not just present.** A separate NLI model checks whether each cited
passage actually entails the sentence. The generator wrote the sentence and picked the citation, so
letting it mark its own work would be meaningless.

**It refuses.** When the sources don't support an answer, it says so and tells you which rule
triggered the refusal. On deliberately unanswerable questions it refuses **82.67%** of the time
before the generator is ever called — those refusals cost nothing at all.

**It repairs itself.** A sentence that fails verification gets rewritten once, then re-checked. If
it fails again it's dropped. If too much of the answer fails, or dropping would leave a fragment,
the whole thing becomes a clean refusal rather than a stub.

**You can switch the modules off.** The page has a toggle per stage, so you can see what each one
actually contributes instead of taking my word for it.

**One page, no build step.** The whole frontend is a single static HTML file - no React, no
Tailwind, no bundler, no `node_modules`. FastAPI serves it at `/`, which is also why its upload
button posts to `/ingest` on the same origin and does real work rather than miming it.

**It knows what it costs.** Token usage and dollar cost are measured per run — **$0.01089 per
question**, measured, not estimated.

---

## Keyboard shortcuts

| Key | Does |
|---|---|
| `Enter` | send the question |
| `Shift` + `Enter` | new line inside the composer |
| `Esc` | close the mobile menu |

That is the honest list - three bindings, all in the composer. Everything else is a click, and the
heavy lifting happens on the command line ([Running the project](#running-the-project)).

---

## The process (how I built it)

I built this in eleven phases, and the rule I set myself at the start was the same rule the system
enforces on itself: **never report a number I hadn't actually measured.** That turned out to be
harder — and more useful — than the code.

**Phase 0–1: the foundation.** Repo skeleton, config with every tunable in one file, then Module A —
chunking, embedding, FAISS. PopQA ships questions but no documents, so I built the corpus by
fetching Wikipedia lead sections for each question's subject, plus 800 unrelated entities as
distractors. Without the distractors retrieval is nearly trivial and the whole thing flatters
itself.

**Phase 2: the contribution — and where it got uncomfortable.** I fine-tuned DeBERTa-v3-small to
grade retrieved chunks, wrote the labeling rubric *before* labeling anything, and froze it. Then I
read the CRAG paper properly and found the number I was supposed to beat: 84.3%, Table 4. I also
found that their metric is per-*query*, not per-chunk — so I had to emit both to compare honestly.

We got **0.8158**. CRAG reports **0.8430**. Worse: a predictor that just says "Correct" every time
scores **0.8750** on our split. So beating CRAG here wouldn't have meant anything either. That's in
the README because it's the truth, not because it's flattering.

**Phase 3–4: citations, then verification.** Structured outputs turned citation parsing from a regex
problem into a schema problem — 200/200 questions took the schema path with zero fallbacks. Then
Module D checks each citation by entailment, with the threshold calibrated on a labeled slice rather
than picked. Building the calibration set was the interesting part: the obvious approach ("assume
the model's own citations are right") is circular, since that's exactly what Module D exists to
decide.

**Phase 5: making it a guarantee.** Repair once, re-verify, drop, or abstain. The test I care most
about asserts that no unverified sentence can reach the output on *any* path.

**Phase 6–9: harness, app, Docker, CI.** The ablation runner treats variants as config flags over
one code path, because five forked pipelines drift and stop measuring what they claim to. Then the
FastAPI service, a Docker image, and a CI gate — which I fired five ways, including by genuinely
crippling the verifier and watching citation recall fall from 0.8388 to 0.4509 and the build go red.

**The frontend, twice.** I built it in Streamlit first because it was quick, then replaced it with a
single static HTML page. Streamlit dragged `pyarrow` and `pandas` into the Docker image for a UI
that was four widgets and a list, and it owned the page so I could not control the layout. One
hand-written file does the same job in 44 KB with no build step.

**Phase 10: writing it up.** Doing the error analysis is what caught my own mistake — see below.

---

## What I learned

**Measuring the wrong thing feels exactly like measuring the right thing.** In Phase 4 I compared
the two multi-citation policies by their aggregate flag rates — 0.2033 versus 0.2057 — and concluded
the choice didn't matter. It did. They fail *different sentences*. When I finally looked at which
ones, 11 of 23 multi-citation failures turned out to be dilution artifacts: one sentence scores
**0.998** against the passage that supports it and **0.075** once a second passage is appended to
the premise. The aggregate hid it completely.

**A number can be technically true and still misleading.** Our citations scored 0.9963 on word
overlap. The real figure, by entailment, is 0.7409. Same citations, same data — one metric just
couldn't see the failure.

**Most of what I flagged as hallucination wasn't.** Reading twenty real failures showed the majority
are *verifier* errors, not generator errors. One sentence was a near-verbatim copy of its source
passage and still scored 0.1332. So the hallucination rate of 0.1612 is an upper bound, and a loose
one — which I'd never have known from the number alone.

**Optimisation intuitions are usually wrong.** I was sure coarser chunks would speed up ingest 2.6×.
Measured: 1.2×. Embedding cost scales with *tokens*, not chunks, so bigger chunks just redistribute
the same work. Multi-process embedding, which should obviously have helped, made things *slower*.

**Fine-tuned models don't transfer.** Module B works well on PopQA-style questions and falls apart
on policy documents — it graded a passage "correct" for a question about a Vice-Chancellor those
documents never mention. Confidently wrong is worse than uncertain.

**Naming your own weak points is worth more than hiding them.** Four gates in this project are still
open, and one of them is open on the merits. Writing that down made the work stronger, not weaker.

---

## How to improve it

**Label the data by hand.** This is the big one. Three separate numbers currently rest on rules
rather than people: Module B's training labels are rule-derived, Module D's calibration positives
are only lexically verified, and the hallucination set is defined and frozen but unlabeled. The
tooling for all three is built and sitting unused. An afternoon of labeling would tighten the
project's strongest claims more than any code change.

**Fix the citation-dilution bug.** Module D should pass a sentence when the *best single* citation
entails it, and use the concatenation only to judge whether every cited passage was necessary. The
per-citation scores are already recorded, so this needs no new inference.

**Give Module B a harder corpus.** Its action distribution is 87% one class, which is why a trivial
predictor beats it. A larger, noisier corpus would make the task real. Alternatively, reframe the
contribution around per-chunk grading (0.9197), which is solid and has no published counterpart.

**Verify with more than one model.** Module D is a single checkpoint and the dominant error source.
An ensemble, or AlignScore alongside it, would bound that. Its 512-token premise window also
silently truncates long passages — sometimes cutting the very sentence that proves the claim.

**Ingest asynchronously.** 25 documents of 500 pages takes ~26 minutes on CPU because embedding runs
at 45 chunks/s. Returning a job id immediately and warming the index in the background would make
the app feel fast without pretending the work is faster. On a GPU it's a non-issue —
`config.RETRIEVAL.device` is already `"auto"`.

**Finish the measurements.** The A–E ablation table, the baseline faithfulness number, and the
risk-coverage curve all exist as code and need roughly $10–15 of API credit to produce.

---

## Running the project

You need **Python 3.11**. Not 3.13 or later — the FAISS and torch wheels aren't dependable there.

### Setup

```bash
git clone https://github.com/MuzammilAliBaig/scrag.git
cd scrag

py -3.11 -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # then put a real ANTHROPIC_API_KEY in it
```

The key is only needed for generating answers. Retrieval, grading and verification all run locally
and free — and the app tells you clearly when generation is unavailable instead of crashing.

### Run the app

One process. The API serves the page as well as the JSON endpoints:

```bash
uvicorn app.api:app --reload     # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>. Upload your own documents with the **+** button, or seed the bundled
demo corpus with `curl -X POST localhost:8000/ingest/demo`, then ask:

- *"How much of my tuition is refunded if I withdraw in week four?"* — answers, with citations
- *"How much does a parking permit cost?"* — refuses, and tells you why

Or with Docker:

```bash
docker compose up --build        # http://localhost:8000
```

### Run the evaluations

These are free and need no API key:

```bash
python -m eval.build_index --split dev --limit 200   # build the corpus (~6 min)
python -m eval.check_retrieval --split dev           # retrieval quality
python -m eval.run_evaluator_bench --split test      # Module B
python -m eval.run_citation_pr --limit 200           # citation precision / recall
python -m eval.check_regression                      # the CI gate
python -m pytest tests/ -q                           # 165 tests
```

These cost money — price them first with `--dry-run`:

```bash
python -m eval.run_citation_bench --dataset alce --limit 200 --dry-run
python -m eval.run_ablation --all --split dev --limit 150 --dry-run
```

### One question from the command line

```bash
python -m core.orchestrator --demo --query "What is Boulsa the capital of?"
```

---

## Where things actually stand

I'd rather say this plainly than let you find it out yourself.

**Working and measured:** retrieval (0.8700 hit rate), Module B grading (0.9197 per-chunk), citation
forcing (1.0000 parseable, 0 invented sources), verification (0.8388 recall / 0.7409 precision),
abstention (0.8267 on unanswerable questions), the app, the Docker image, and 165 passing tests.

**Not finished, and why:**

- **The A–E ablation table doesn't exist.** The harness runs from one command, but the sweep needs
  generation across five variants and the API credit ran out. No cell has been filled with a guess.
- **Faithfulness was never measured**, for the same reason.
- **Module B doesn't beat its baseline.** That's a finding, not a pending task.
- **No public URL yet** — the image builds and runs, deploying needs an account.
- **CI hasn't run on GitHub yet.** The gate is proven locally five ways; that GitHub invokes it
  isn't proven.

## Documentation

| Document | What's in it |
|---|---|
| [docs/architecture.md](docs/architecture.md) | the pipeline diagram, both corrective loops |
| [docs/error-analysis.md](docs/error-analysis.md) | twenty real failures, with commentary |
| [docs/INGEST_CAPACITY.md](docs/INGEST_CAPACITY.md) | ingest throughput, limits, what I tried and rejected |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | seeds, versions, hardware, every command |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | a rehearsed demo, with the refusal trigger named |
| [docs/tables/](docs/tables/) | every measured result, CSV and markdown |
| [DEPLOY.md](DEPLOY.md) | Docker, Hugging Face Spaces, Render |
| [CI.md](CI.md) | the regression gate and how to fire it |
