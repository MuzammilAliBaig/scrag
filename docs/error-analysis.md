# Error analysis — twenty real failures

Every case below is a real verdict from a real run. Sources: `eval/results/verifier_bench.json`
(200 ALCE/ASQA questions, 418 sentences) and `eval/results/verdicts.jsonl`. Nothing here is
constructed to make a point.

The headline: **most flagged sentences are verifier errors, not generation errors.** Of the 85
sentences Module D flagged, the sample below suggests the majority are cases where the cited passage
does support the claim and the NLI checkpoint failed to see it. That matters for how the numbers in
this project should be read — the automatic hallucination rate of 0.1612 is an upper bound on
generator failure, and a loose one.

---

## Finding 1 — the `concat` multi-citation policy dilutes entailment

**This corrects a claim made in Phase 4.** The Phase 4 sensitivity check compared aggregate flag
rates between the two policies — `concat` 0.2033, `any` 0.2057 — and concluded the choice was
immaterial. Aggregates were the wrong thing to look at. The two policies fail *different sentences*,
and one failure mode is systematic.

Under `concat`, every cited chunk is joined into one premise. When a sentence cites two chunks and
only one is relevant, the irrelevant text dilutes the premise and the entailment score collapses.

**Measured: 11 of the 23 multi-citation failures are dilution artifacts** — the best single citation
scores ≥ 0.20 (the pass threshold) while the concatenated premise fails.

| Sentence | concat | best single | citations |
|---|---|---|---|
| Bill Pertwee played ARP Warden Hodges in "Dad's Army". | **0.075** | **0.998** | 2 |
| It is popularly, though mistakenly, believed to be of his brother Eros… | **0.020** | **0.960** | 2 |
| The crash claimed the lives of Lexie Grey and Mark Sloan. | **0.001** | **0.850** | 2 |
| The video also featured the Eliminator girls… | **0.044** | **0.727** | 2 |
| The St. Louis Cardinals lead National League teams with 11 championships. | **0.013** | 0.483 | 2 |

A sentence scoring 0.998 against the passage that supports it, and 0.075 once a second passage is
appended, is not a marginal effect.

**Why it was not caught earlier:** the aggregate flag rates matched because `any` fails roughly as
many *other* sentences as `concat` rescues. Identical totals, different populations.

**What this implies:** `concat` matches ALCE's citation-recall definition and is defensible on
those grounds, but it penalises a generator for over-citing rather than for citing wrongly. Since
`per_citation` scores are already recorded for every verdict, a better policy is available without
new inference — pass when the best single citation entails the sentence, and use the concatenation
only to decide whether *all* cited passages were necessary. That is a change to Module D, so it is
recorded here rather than made silently.

---

## Category A — verifier false positives (the largest group)

The cited passage does support the sentence. Module D disagrees. These inflate the flag rate and
would, in variant E, cause an unnecessary repair or abstention.

**A1. Verbatim copy scored as neutral.**
Q: *Who wrote the book of the new testament?*
S: "The New Testament is a collection of writings by early Christians, believed to be mostly Jewish
disciples of Christ, written in first-century Koine Greek."
P: *"…The New Testament is a collection of writings by early Christians, believed to be mostly
Jewish disciples of Christ, written in first-c…"*
entail **0.1332**, label neutral.
The sentence is a near-verbatim copy of the passage. The most likely cause is premise truncation:
the passage is long, the supporting clause sits late in it, and the 512-token window cuts it. A
lexical-overlap check would have passed this instantly, which is precisely why the Phase 3 proxy
scored 0.9963 while entailment scores 0.8388.

**A2. Inca empire.**
Q: *What led to the downfall of the incan empire?*
S: "The empire fell to the Spanish in 1533 when the last Sapa Inca, Atahualpa, was captured and
executed on 29 August."
P: *"The Inca Empire eventually fell to the Spanish in 1533, when the last 'Sapa Inca'…"*
entail **0.0173**, label **contradiction**.
The passage states the same fact. Labelling it a contradiction is worse than labelling it neutral —
it would tell a reader the source *refutes* the sentence.

**A3. Gold standard.**
Q: *When did us currency leave the gold standard?*
S: "In a series of laws and executive orders in March and April 1933, the U.S. government suspended
the gold standard for United States currency."
P: passage on the 1933 gold recall and suspension.
entail **0.127**, label contradiction. The date and the action both match.

**A4. Pauline authorship.**
Q: *Who wrote the book of 1 and 2 thessalonians?*
S: "1 Thessalonians is one of the core group of epistles whose authorship by Paul the Apostle is
rarely contested…"
P: *"…traditionally attributed to Paul the Apostle, although many dispute the anonymous Epistle to
the Hebrews…"*
entail **0.0167**, label contradiction.
The model appears to read "many dispute" as contradicting "rarely contested", missing that the
dispute concerns Hebrews, not Thessalonians. A real reasoning failure, not truncation.

**A5. Minnehaha Academy.**
Q: *Where is the minnehaha academy located in minneapolis?*
S: "Minnehaha Academy is located in the Cooper and Hiawatha neighborhoods of Minneapolis, on West
River Parkway."
P: *"…a Christian private school in Minneapolis… There are two campuses, the South Campus…"*
entail **0.0312**, label contradiction.
"Two campuses" is treated as incompatible with two named neighbourhoods, when it is consistent
with them.

---

## Category B — genuine citation errors (Module D working correctly)

**B1. Wrong passage entirely.**
Q: *Who plays snowman in smokey and the bandit?*
S: "Jerry Reed plays Cledus 'Snowman' Snow, the Bandit's truck-driving partner…"
P: the **Jackie Gleason** article, about Buford T. Justice.
entail **0.1441**, neutral. The claim is true; the citation is wrong. Exactly the failure this
project exists to catch — a plausible-looking citation attached to a sentence it does not support.

**B2. Right topic, wrong fact in the passage.**
Q: *When did the last season of jersey shore air?*
S: "The last season (season six) was set in Seaside Heights, New Jersey."
P: the series overview, giving air dates but not the season-six setting.
entail **0.0196**. The passage supports neighbouring facts, not this one.

**B3. Quotation source confusion.**
Q: *Who said it is a tale told by an idiot full of sound and fury?*
S: "In the speech, Macbeth says life 'is a tale Told by an idiot…'"
P: **The Sound and the Fury** (the Faulkner novel), which quotes the same soliloquy.
entail **0.0416**. Genuinely ambiguous: the passage contains the quotation but attributes it in a
different frame. A human labeller might call this `ambiguous` rather than wrong.

**B4. Right person, wrong premise.**
Q: *Who was england's prime minister during ww1?*
S: "He was replaced in December 1916 by David Lloyd George, who became Prime Minister on 7 December
1916…"
P: Imperial War Cabinet, confirming the 7 December 1916 date but not the replacement of Asquith.
entail **0.0259**. Half the sentence is supported; the sentence as a whole is not. A
single-claim-per-sentence rule — already rule 10 of the Phase 3 output contract — would have made
this repairable.

---

## Category C — abstentions

21 of 200 questions produced a refusal. Module D flags these as unsupported because there is nothing
to verify; they are marked `uncited` so the two are not confused.

**C1–C4.** *Who killed the man in thelma and louise?* · *When did the voortrekkers arrive in south
africa?* · *Who is the chief minister of m.p?* · and 18 others produced exactly
*"I cannot answer this from the provided sources."*

**Were any of these abstentions wrong?** This is the question the phase brief specifically asks, and
it cannot be answered from the automatic data — deciding whether the top-5 ALCE passages *did*
contain the answer requires reading them. That is one of the jobs the unlabeled hallucination set
(`eval/LABELING_HALLUCINATION.md`) exists to do. **Reported as unknown rather than assumed correct.**

What can be said: the refusal rate of 10.5% on ALCE/ASQA is far below the 82.67% pre-generation
refusal rate on deliberately unanswerable PopQA questions, which is at least consistent with the
system refusing selectively rather than reflexively.

---

## Category D — Module B out of distribution

**D1. The demo corpus.**
Asked *"Who is the Vice-Chancellor?"* against four policy documents that never mention one, Module B
graded a chunk **`correct`** (confidence 0.52) and returned `proceed`.
Asked *"How much does a parking permit cost?"*, it graded all five chunks `ambiguous`.

Module B was fine-tuned on PopQA entity questions over Wikipedia lead sections. Policy prose is a
different distribution, and its grades there are close to uninformative. This is the sharpest
limitation in the project: the module presented as the contribution does not transfer off its
training distribution, and it fails **confidently** rather than by abstaining.

**D2. On PopQA it behaves.** Trigger (b) fires on 82.67% of unanswerable questions and only 2.67% of
answerable ones. The failure is domain shift, not a broken module.

---

## What this analysis changes

1. **The automatic hallucination rate (0.1612) is an upper bound**, and a loose one. Categories A
   and the dilution artifacts are verifier errors, not generator errors. The true generator failure
   rate is lower and cannot be quantified without the hand-labeled set.
2. **The `concat` policy needs revisiting** — 11 measured dilution artifacts, with a fix available
   from data already recorded.
3. **Premise truncation is a real failure mode** (A1). Long ALCE passages exceed the 512-token NLI
   window, and the supporting clause is sometimes the part that gets cut.
4. **Rule 10 of the citation contract earns its place** (B4). Sentences making two claims are harder
   to verify and harder to repair.
5. **Module B's domain narrowness should be stated in the report**, not discovered by an examiner
   asking what happens on an uploaded PDF.
