# CI and the regression gate

Two workflows:

| Workflow | Trigger | Cost |
|---|---|---|
| `.github/workflows/test.yml` | every push and PR | free |
| `.github/workflows/eval.yml` | every push and PR | **free by default** |

## Why the eval job is free by default

Every eval run that calls the Claude API spends real money on the only paid component in this
project. So the push-triggered job runs only metrics that need no generation:

- retrieval hit rate (local FAISS)
- Module B grading accuracy and macro-F1 (local checkpoint)
- citation precision/recall and the automatic hallucination rate, computed by Module D against
  **generations already in the response cache**

The paid subset — `run_citation_bench` and `run_baseline` — is behind a `workflow_dispatch` input
(`run_paid_eval`) and never runs on a push. It is capped at 50 questions.

## The gate

`python -m eval.check_regression` compares fresh results in `eval/results/` against the committed
`eval/baseline_metrics.json` and exits non-zero on a regression. That exit code is what turns the
CI check red.

Three rules it enforces:

1. **Thresholds sit below the measurements, with a stated margin.** A gate set exactly at today's
   number fires on noise, gets disabled within a week, and then protects nothing. Margins are sized
   by how noisy each metric is: small for deterministic local metrics, larger for anything
   downstream of the LLM.
2. **A missing measurement is not a pass.** If a required metric has no result file, the gate fails
   with `MISSING`. A gate that goes green when the evidence disappears is worse than no gate,
   because it looks like protection.
3. **An unmeasured metric is skipped, loudly, never invented.** `faithfulness` has never been
   measured — the Phase 1 baseline has not run — so it carries `value: null`, prints `SKIPPED` with
   the reason, and gets no threshold. Giving it a plausible number would make the metric this
   project cares about most unfalsifiable.

## Firing the gate — the acceptance test

**A gate that has never fired is not known to work, and the workflow file existing is not
evidence.** Two ways to fire it.

### Synthetic (fast, no measurement)

```bash
python -m eval.check_regression --simulate citation_recall=0.60
# expect: GATE FAILED, exit code 1
```

In CI: **Actions → eval gate → Run workflow**, and set `simulate_regression` to
`citation_recall=0.60`.

### Real regression (the honest one)

Weaken the pipeline for real and re-measure. Raising the entailment threshold cripples Module D
without touching any other module, and re-measuring is free:

```bash
# in config.py: entailment_threshold 0.20 -> 0.97
python -m eval.run_citation_pr --limit 200
python -m eval.check_regression
# then restore config.py
```

**Recorded result of this drill, 2026-09-07:**

| Metric | Baseline | Threshold | Regressed | Verdict |
|---|---|---|---|---|
| `citation_recall` | 0.8388 | 0.78 | **0.4509** | FAIL |
| `citation_precision` | 0.7409 | 0.68 | **0.4357** | FAIL |
| `hallucination_rate_auto` | 0.1612 | 0.22 | **0.5491** | FAIL |

Exit code **1**. After restoring `config.py`, the gate returned to exit code **0**.

The `MISSING` path was fired too: pointing `--results` at an empty directory failed the build on
all three required metrics rather than passing.

## Setting it up on GitHub

This repository has **no remote yet**, so none of the above has run on GitHub's runners. To enable:

```bash
git remote add origin git@github.com:<you>/scrag.git
git push -u origin main
```

Then **Settings → Secrets and variables → Actions → New repository secret**: `ANTHROPIC_API_KEY`.
It is only needed for the opt-in paid job; the default push-triggered eval does not use it.

Caching keeps the job inside the 2000 free minutes/month: pip by lockfile, HuggingFace weights by
`config.py` hash, the corpus and FAISS index by the hash of `core/retriever.py` and
`eval/corpus.py`, and the Claude response cache by run id with a restore-key fallback.
