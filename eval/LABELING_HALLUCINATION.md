# Hallucination Labeling Rubric — the ablation's ground truth

**Status: FROZEN as of 2026-09-07, before any labeling began.**

The hallucination-rate column of the ablation table is scored against labels defined here. As with
`train/LABELING.md`, this file is frozen: revising it after seeing which variant wins would make the
headline finding unfalsifiable. A genuine defect gets a new dated version and a full relabel, never
a quiet edit.

---

## 1. What is being labeled

One **sentence of a generated answer**, together with the passages that answer was given.

The label answers one question:

> **Does the passage set support this sentence?**

Not "is this sentence true in the world." A sentence can be perfectly true and still be a
hallucination *for this system*, because the system's contract is to answer only from its sources.
That distinction is the entire point of the project, and it is the single most common labeling
mistake.

---

## 2. The two classes

| Label | Test |
|---|---|
| `supported` | A careful reader could derive this sentence from the passages alone. |
| `hallucinated` | The passages do not support it — whether they are silent on it or contradict it. |

### 2.1 `supported`

The passages state the sentence, or directly entail it with no outside knowledge and no
inferential leap a careful reader would decline to make.

Paraphrase is fine. "He served as a Conservative politician" supports "His occupation was
politician."

### 2.2 `hallucinated`

Three distinct situations, all `hallucinated`:

1. **Unsupported.** The passages are silent. The sentence may well be true; it is not in evidence.
2. **Contradicted.** A passage says the opposite. This is the more serious kind and is sub-labeled.
3. **Wrong entity.** The sentence is about a different person, place or work than the passages
   describe — the name-collision case.

### 2.3 The sub-label

Every `hallucinated` sentence also takes one of `unsupported`, `contradicted`, `wrong_entity`.
This mirrors the neutral/contradiction distinction Module D already records, and it separates a
retrieval failure from a generation failure in the error analysis.

---

## 3. What is excluded

- **Abstentions.** "I cannot answer this from the provided sources" asserts nothing, so it is
  neither supported nor hallucinated. Excluded from the denominator, not scored as either. Scoring
  refusals as `supported` would let a system reach a zero hallucination rate by refusing
  everything.
- **Pure connectives** carrying no factual claim.

---

## 4. Procedural rules

1. **Judge against the passages, never against your own knowledge.** If you know the fact but the
   passages do not contain it, the sentence is `hallucinated`. This is the rule people break.
2. **Do not look at the citation before forming a judgement.** Read the sentence and the passages,
   decide, then look. Seeing a confident citation first makes weak support look sufficient.
3. **Do not look at which variant produced the sentence.** The set must be labeled blind, or the
   ablation measures the labeler's expectations. The labeling tool shuffles and strips variant
   identifiers for exactly this reason.
4. **Label every sampled sentence.** Skipping the hard ones biases the set toward the easy cases
   and deflates every hallucination rate in the table.

---

## 5. Set size and the double-labeled slice

| Parameter | Value |
|---|---|
| Target size | 200 sentences |
| Sampling | stratified across variants A–E, blind to variant |
| Double-labeled slice | 60 sentences (30%) |
| Agreement statistic | Cohen's kappa, reported overall and per class |

The double-labeled slice is not optional. Without an agreement figure the hallucination column is
one person's opinion, and the ablation's headline claim rests on it.

**Expected difficulty:** the `unsupported` / `supported` boundary is where annotators will diverge,
for the same reason `ambiguous` was the weak class in `train/LABELING.md` — deciding whether an
inferential step is small enough to count as entailment is a judgement call. Report per-class
agreement, and expect this pair to be the weakest.

---

## 6. Status

**The set has not been labeled.** `eval/labeled/` contains the sampling tool and an empty
`hallucination.jsonl`. Until it is populated:

- `hallucination_rate_labeled` returns `None`, never `0.0` — a zero would read as a perfect score.
- The ablation reports `hallucination_rate_auto` instead, derived from Module D's verdicts, and
  labels it as verifier-derived rather than human-verified.

The automatic rate measures disagreement between the generator and the NLI checkpoint. It inherits
every error Module D makes, and its calibration is bounded by the lexically-verified positives
described in `eval/results/verifier_calibration.json`. The hand-labeled set exists to bound that
gap, and until it exists the gap is unbounded and must be described that way in the report.
