# Demo script — rehearsed, with the triggers named

The most memorable thirty seconds of a viva is a system that **refuses to answer and explains why**.
That should be rehearsed, not discovered live. Every question below has been run; the outcome
column is what actually happened, not what should happen.

```bash
uvicorn app.api:app --reload     # terminal 1
streamlit run app/ui.py          # terminal 2
```

Sidebar → **Load demo corpus** (4 policy documents, 8 chunks).

---

## Part 1 — answers with clickable citations

Requires API credit. Each of these is covered by the demo corpus.

| # | Question | Expected |
|---|---|---|
| 1 | How much of my tuition is refunded if I withdraw in week four? | "50 per cent" — cites `refunds-policy` |
| 2 | What time does the library close on Saturday? | "18:00" — cites `library-access` |
| 3 | How many items can a postgraduate student borrow? | "25" — cites `library-access` |
| 4 | What happens if I arrive 40 minutes late to an examination? | "may not be admitted" — cites `exam-regulations` |

**What to point at:** click the citation under a sentence. The passage that opens is the exact chunk
Module D checked the sentence against. That is the difference between a citation and a *verified*
citation, and it is the whole argument of the project in one click.

---

## Part 2 — the refusal

### 2a. On the demo corpus (needs API credit)

| Question | Trigger that fires |
|---|---|
| How much does a parking permit cost? | **Module C**, output-contract rule 7 |
| What is the pass mark for a dissertation? | **Module C**, output-contract rule 7 |

**Be precise about which trigger fires, because it is not the one you would expect.**
Module B does **not** refuse these. Measured on all four demo documents:

| Question | Module B action |
|---|---|
| How much does a parking permit cost? | `corrective_retrieve` |
| Who is the Vice-Chancellor? | **`proceed`** |
| What is the pass mark for a dissertation? | `corrective_retrieve` |
| How do I apply for a student visa? | `corrective_retrieve` |

None abstains. Module B was fine-tuned on PopQA entity questions over Wikipedia lead sections, and
policy prose is out of distribution for it — on "Who is the Vice-Chancellor?" it graded a chunk
`correct` at confidence 0.52 for a fact the documents never mention.

So on this corpus the refusal comes from **Module C**, which is instructed to reply exactly *"I
cannot answer this from the provided sources"* when the passages do not contain the answer, and
that costs a generation call. **If asked "which module refused?", the honest answer is Module C,
not Module B.** Volunteering that limitation is stronger than being caught by it.

### 2b. Without API credit — a refusal that is genuinely free

Point the app at the evaluation index instead (`data/indexes/corpus.faiss`, 1,615 chunks of
Wikipedia). These questions abstain **before generation**, so they work with no credit at all:

```bash
python -m core.orchestrator --demo --query "What sport does Imbi Hoop play?"
```

| Question | Outcome |
|---|---|
| What sport does Imbi Hoop play? | abstains, `no_usable_chunk` |
| Who is the father of Æthelburh of Wilton? | abstains, `no_usable_chunk` |
| What sport does Hwang Byung-ju play? | abstains, `no_usable_chunk` |
| In what country is Pârâul Bogat? | abstains, `no_usable_chunk` |
| Who is the author of Miracle? | abstains, `no_usable_chunk` |

Here **trigger (b)** fires: Module B graded every retrieved chunk `wrong`, so the pipeline stopped
before the generator was called. The trace says `cost: zero - abstained before generation`.

**The line worth saying out loud:** on this split trigger (b) refuses **82.67%** of unanswerable
questions and only **2.67%** of answerable ones, and every one of those refusals is free — the paid
component is never reached.

---

## Part 3 — showing the machinery

Expand **"How this answer was produced"** under any answer. It shows the Module B action, whether
corrective retrieval fired, how many sentences were flagged, how many repairs succeeded, and how
many sentences were dropped.

A **dropped** sentence is the most persuasive thing on screen: it is a claim the system wrote,
failed to verify, tried once to repair, failed again, and removed rather than show you.

The sidebar toggles switch modules off. Turning **D — NLI verification** off and re-asking shows
the same answer with no verification badges: the clearest way to show what Module D contributes
without describing it.

---

## Questions you should expect, with honest answers

**"Does it beat the paper you compare against?"**
No. Module B's query-level action accuracy is **0.8158** against CRAG's reported **0.8430**
(Yan et al. 2024, Table 4). It is also below our own majority-class baseline of **0.8750**, which
means beating 0.8430 on this split would not have been a real claim either — the action
distribution is too skewed. The per-chunk accuracy of **0.9197** is the solid number, and CRAG
published no per-chunk figure to compare it to.

**"How do you know the citations are right and not just well-formed?"**
Those are measured separately and on purpose. Parseable-citation rate is **1.0000** — every sentence
carries a citation that resolves to a retrieved passage. Whether the passage *supports* the sentence
is entailment, measured at citation recall **0.8388** and precision **0.7409**. Phase 3's
word-overlap proxy said 0.9963; entailment says 26% of individual citations do not support their
sentence. The gap between those two numbers is the point.

**"What is the hallucination rate?"**
**0.1612**, and it is an **upper bound** derived from the verifier, not human judgement. The error
analysis (`docs/error-analysis.md`) shows most flagged sentences are verifier errors rather than
generator errors. The hand-labeled set that would tighten this is defined and frozen but not yet
labeled.

**"Where is the ablation table?"**
Not produced. The harness exists and runs one command, but the sweep needs generation across five
variants and the API credit ran out. No cell has been filled with an estimate.
