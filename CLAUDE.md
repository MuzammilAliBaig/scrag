# A Self-Correcting, Citation-Verified RAG Framework for Reliable Answers — Standing Agent Context

*Short handle used throughout: **SCRAG**.*

> **Copy this file into the root of the `scrag/` build repo, renamed to `CLAUDE.md`.**
> It loads automatically at the start of every session and keeps the agent on-stack across phases.
> Source of every fact below: `raw/briefs/trustrag-project-brief.pdf` (wiki page `S001-trustrag-project-brief`).

---

## 1. What is being built

**SCRAG** — *A Self-Correcting, Citation-Verified RAG Framework for Reliable Answers*. A document-QA system that answers only from retrieved sources and refuses any claim
it cannot back with a verified citation.

Ordinary RAG fails in two specific ways, and the whole system exists to close both:

| Failure | Closed by |
|---|---|
| States facts that are not in the retrieved sources | Retrieval gating (Module B) + abstention (Module E) |
| Attaches citations that look valid but do not support the sentence | Citation forcing (Module C) + NLI verification (Module D) |

**The measurable novelty is Module B**: a fine-tuned retrieval evaluator replacing the CRAG
evaluator. Everything else reproduces published work. Protect and highlight Module B.

Every claim in the final report must read: *"the paper reported X, we achieved Y."*

---

## 2. Architecture — five modules plus an app shell

Every function written must map to exactly one module. Keep boundaries clean.

| Module | Responsibility | Input → Output | Done when |
|---|---|---|---|
| **A** Ingest / Index | Chunk, embed, store in FAISS | `docs` → `vector index` | Query returns top-k relevant chunks |
| **B** Retrieval evaluator | Grade chunks correct/ambiguous/wrong; trigger corrective retrieval | `q + chunks` → `labels + action` | Beats CRAG-reported evaluator accuracy |
| **C** Citation generator | Generate answer, one citation per sentence | `q + good ctx` → `cited draft` | Every sentence has a parseable citation |
| **D** NLI verifier | Check each cited chunk entails its sentence | `draft` → `pass/fail per sentence` | Unsupported sentences are flagged |
| **E** Repair / abstain | Regenerate, drop, or abstain | `flagged draft` → `final / abstain` | No unverified sentence survives |

Bad retrieval triggers a corrective loop. Failed verification triggers regeneration or abstention.

---

## 3. Stack — pinned, all free

| Layer | Choice | Note |
|---|---|---|
| Embeddings | sentence-transformers (bge-small or all-MiniLM) | Runs locally, CPU-fine |
| Vector store | FAISS, local | No service, no cost |
| Generator LLM | **Claude API** — `claude-opus-5` (official `anthropic` Python SDK) | **The one paid component.** `ANTHROPIC_API_KEY` via env var, never hardcoded |
| Retrieval evaluator | DeBERTa-v3-small, fine-tuned | Train on free Kaggle/Colab GPU |
| NLI verifier | Off-the-shelf NLI / AlignScore checkpoint | No training needed |
| Backend | FastAPI + Uvicorn | Local / container |
| Frontend | Streamlit (default) or Next.js if the UI is a showpiece | Streamlit Cloud free |
| Eval | RAGAS + custom citation P/R + hand-labeled set | Local |
| Packaging | Docker | Free |
| Hosting | Hugging Face Spaces or Render free tier | Cold starts acceptable |
| CI | GitHub Actions | 2000 free min/month |

**Do not swap FAISS, DeBERTa-v3-small, FastAPI, or the Claude API without being asked.**

**Cost.** Every component above is free except the Claude API, which is billed per token with no
free tier. It is therefore a managed resource, like GPU time. Use prompt caching for the stable
prompt prefix, the Message Batches API (50% cost) for every evaluation run, and on-disk result
caching. Never use an OpenAI-compatible shim — use the official `anthropic` SDK. Full details and
cost estimates in `tech-stack.md`.

---

## 4. Target repository layout

```
scrag/
├── app/                    FastAPI app + Streamlit UI
├── core/
│   ├── retriever.py        Module A
│   ├── evaluator.py        Module B — the contribution
│   ├── generator.py        Module C
│   ├── verifier.py         Module D
│   ├── repair.py           Module E
│   └── orchestrator.py     wires A–E
├── train/                  evaluator labeling + fine-tuning scripts
├── eval/                   benchmarks, ablation runner, metrics
├── data/                   indexes, demo corpus
├── tests/                  unit tests per module
├── config.py
├── requirements.txt
├── Dockerfile
└── .github/workflows/eval.yml
```

---

## 5. Environment rules

1. Secrets (LLM API keys) only via environment variables / `.env`. Never committed.
2. Pin dependencies in `requirements.txt`. Pin versions that actually resolve at install time — do
   not invent version numbers.
3. Keep a `config.py` with model names, top-k, thresholds. **No magic numbers in code.**
4. Everything must run CPU-only as a fallback. GPU is used for exactly one job: the Phase 2 fine-tune.

---

## 6. Scope guardrails

**In scope:** retrieval, evaluator, citation-forced generation, NLI verification, repair/abstention,
evaluation harness, FastAPI + Streamlit app, Docker, free-tier deploy, GitHub Actions eval gate.

**Out of scope — do not add unless explicitly asked:** multi-agent frameworks, paid vector DBs,
fine-tuning the LLM itself, training the NLI model from scratch, auth/user accounts, multi-tenant
infra.

If something outside these guardrails seems necessary, **flag it rather than adding it.**

---

## 7. How to operate

1. **Confirm the phase first.** Never jump ahead of a "Done when" gate.
2. **Stay on the declared stack.** No paid services, no extra frameworks, no unrequested swaps.
3. **Ship runnable code.** Full files or clear diffs, with imports, a run command, and the repo path
   for every file.
4. **Map every function to a module A–E.**
5. **Benchmark, do not assert.** A phase with a metric needs the measurement script, not just the
   feature.
6. **Respect secrets and config.** Keys in env vars, tunables in `config.py`, pinned requirements.
7. **Keep the novelty central.** Module B is the contribution.
8. **Close each phase** by restating what changed and what the next "Done when" is.
9. **Small, testable steps.** A unit test with each module. CPU-only must work.
10. **Ask before scope creep.**

---

## 8. Reporting discipline

This project builds a system that refuses to state what it cannot cite. Hold the build to the same
standard:

- Never report a metric that was not produced by a run you actually executed.
- If a number comes from a paper rather than a run, say which paper.
- If a run failed or was skipped, say so. A missing number is a finding; a fabricated one is a
  broken project.
- When a phase produces measurements, record them so they can be filed back into the wiki.
