# SCRAG Chunk Labeling Rubric — Module B

**Status: FROZEN as of 2026-09-06, before any labeling began.**

This rubric defines the ground truth that Module B's grading accuracy is scored against. It is
frozen deliberately. Revising it after seeing results would make the headline number
unfalsifiable — we could always redefine the boundary until the model looked good. If a genuine
defect is found, the fix is a **new dated version of this file plus a full relabel**, never a quiet
edit.

---

## 1. What is being labeled

One **(question, chunk) pair** at a time.

- The **question** is a PopQA question, e.g. *"What is George Rankin's occupation?"*
- The **chunk** is a passage returned by Module A's FAISS retrieval — roughly 800 characters of a
  Wikipedia lead section.

The label answers exactly one question:

> **Does this chunk let a careful reader answer this question, using nothing else?**

It is a judgement about **evidential sufficiency**, not about topical similarity, writing quality,
or whether the chunk is interesting.

---

## 2. The three classes

| Label | One-line test |
|---|---|
| `correct` | The chunk states the answer. A careful reader could answer from this chunk alone. |
| `ambiguous` | The chunk is about the right subject and bears on the question, but does not settle it. |
| `wrong` | The chunk cannot support an answer — wrong subject, or right subject and nothing relevant. |

### 2.1 `correct`

Assign `correct` when the chunk **contains the answer**, explicitly or by direct paraphrase.

Requirements, all of which must hold:

1. The chunk concerns **the entity the question asks about** — the same person, place, work or
   organisation, not merely one with a similar name.
2. The chunk states the queried property.
3. A reader with no outside knowledge could produce the gold answer from this chunk.

Surface form does not matter. "He served as a Conservative Party politician" is `correct` for
*"What is Henry Feilden's occupation?"* even though the gold answer string is "politician".

### 2.2 `wrong`

Assign `wrong` when the chunk **cannot support an answer**. Two distinct situations, both `wrong`:

1. **Wrong subject.** The chunk is about a different entity. This includes the name-collision case,
   which is the single most common retrieval failure in this corpus: *Easy Living* the 1937 film
   retrieved for a question about *Easy Living* the song.
2. **Right subject, irrelevant content.** The chunk really is about the entity, but says nothing
   bearing on the queried property. A paragraph on Ada Lovelace's childhood is `wrong` for a
   question about her occupation if occupation is never touched on.

### 2.3 `ambiguous` — the boundary that matters

This is where grading is actually hard, and where the rubric has to be strictest with itself.

Assign `ambiguous` when the chunk is **on-subject and bears on the question, but does not settle
it**. The reader is better off with the chunk than without it, yet still cannot answer with
confidence.

Four recognised situations:

1. **Partial evidence.** The chunk narrows the answer without stating it.
2. **Answer implied but not stated.** Answering requires an inferential step a careful reader might
   decline to take.
3. **Under-determined subject.** The chunk plausibly concerns the right entity but does not confirm
   it — the name matches and the context is compatible, but the identity is not established.
4. **Conflicting or hedged evidence.** The chunk gestures at more than one answer, or hedges the one
   it gives.

**The tie-break rule.** If you find yourself arguing for `correct`, ask: *could a careful reader
state the answer from this chunk alone and defend it?* If the honest answer is "probably, if I
assume one more thing" — that is `ambiguous`, not `correct`.

If you are torn between `ambiguous` and `wrong`, ask: *does this chunk move me at all toward the
answer?* If genuinely not, it is `wrong`. Being about the right entity is **not** on its own
sufficient for `ambiguous`.

---

## 3. Worked examples

### 3.1 `correct`

**Example C1**
Q: *What is George Rankin's occupation?* (gold: politician)
Chunk: "George James Rankin was an Australian politician and soldier. He served in the House of
Representatives from 1937 to 1949."
→ **`correct`.** Right entity, occupation stated outright.

**Example C2**
Q: *Who was the producer of The Sea Wolf?* (gold: Henry Blanke)
Chunk: "The Sea Wolf is a 1941 American adventure drama film directed by Michael Curtiz and produced
by Henry Blanke, based on Jack London's 1904 novel."
→ **`correct`.** The property is stated, and the film is identified precisely enough to be sure it
is the right one.

**Example C3**
Q: *In what city was Aristotle born?* (gold: Stagira)
Chunk: "Aristotle was an Ancient Greek philosopher and polymath. Born in Stagira, Chalcidice, in
384 BC, he was a student of Plato."
→ **`correct`.** Paraphrase — "Born in Stagira" rather than "the city of Stagira" — is fine.

### 3.2 `ambiguous`

**Example A1 — partial evidence**
Q: *What is Kathryn Bigelow's occupation?* (gold: film director)
Chunk: "Kathryn Bigelow became the first woman to win the Academy Award for Best Director, for The
Hurt Locker in 2010."
→ **`ambiguous`.** Winning Best Director very strongly implies she is a director, but the chunk
never states her occupation. A reader must take one inferential step. Not `correct`, because the
property is not stated; not `wrong`, because the chunk is powerfully relevant.

**Example A2 — under-determined subject**
Q: *Who was the composer of Easy Living?* (gold: Ralph Rainger)
Chunk: "Easy Living is a 1937 American screwball comedy film directed by Mitchell Leisen, from a
screenplay by Preston Sturges."
→ **`ambiguous`.** The song *Easy Living* was written for a 1937 Paramount film, so this chunk may
concern the right work — but it names a director and screenwriter, never a composer, and never
confirms that a song is involved. It is not confidently the wrong entity, so `wrong` overstates;
it settles nothing, so `correct` is out of reach.

**Example A3 — implied but not stated**
Q: *In what country is Lake Bled?* (gold: Slovenia)
Chunk: "Lake Bled is a lake in the Julian Alps of the Upper Carniolan region, next to the town of
Bled. It is a popular tourist destination."
→ **`ambiguous`.** Upper Carniola is in Slovenia, but the chunk never says so. Answering requires
outside knowledge — precisely what SCRAG forbids. The chunk narrows the answer without stating it.

**Example A4 — conflicting evidence**
Q: *What is the genre of Blade Runner?* (gold: science fiction)
Chunk: "Blade Runner has been variously described as neo-noir, cyberpunk, and a tech-noir thriller,
and its genre classification remains debated among critics."
→ **`ambiguous`.** On-subject and on-property, but it offers several answers without settling on
the gold one.

### 3.3 `wrong`

**Example W1 — wrong subject, name collision**
Q: *Who was the screenwriter for Death of a Batman?* (gold: John Hopkins)
Chunk: "Batman is a 1989 American superhero film directed by Tim Burton. When Sam Hamm's script was
rewritten, ..."
→ **`wrong`.** Different work entirely. Lexical overlap on "Batman" is what retrieved it, and
lexical overlap is not evidence.

**Example W2 — right subject, irrelevant content**
Q: *What is Ada Lovelace's occupation?* (gold: mathematician)
Chunk: "Ada was the only legitimate child of the poet Lord Byron and his wife Anne Isabella Milbanke.
Byron separated from his wife a month after Ada was born and left England forever."
→ **`wrong`.** Genuinely about Ada Lovelace, but concerns her parentage and says nothing about what
she did. Being on-subject is not enough.

**Example W3 — a distractor with no connection**
Q: *What is the capital of Bhutan?* (gold: Thimphu)
Chunk: "All by Myself is a song by American singer-songwriter Eric Carmen, released in 1975."
→ **`wrong`.** No relationship of any kind. This is what most distractor retrievals look like.

---

## 4. Procedural rules

1. **Judge the chunk, not the corpus.** "Another chunk has the answer" is irrelevant. Each pair is
   labeled on its own.
2. **Do not use outside knowledge.** If you personally know the answer but the chunk does not state
   it, the chunk is not `correct`. This mirrors what the system is required to do.
3. **Do not look at the gold answer before forming a judgement**, then check it. Seeing the gold
   answer first makes weak chunks look sufficient — the single largest source of label drift.
4. **Truncated chunks are judged as given.** Chunks are windows, and a fact just past the boundary
   does not count.
5. **Label every sampled pair.** Skipping the hard ones biases the set toward the easy boundary and
   inflates every metric downstream.

---

## 5. Query-level actions, derived not labeled

CRAG's reported number is **not** per-chunk accuracy. It is the accuracy of the *action* chosen for
a whole retrieved set (§5.5, Table 4 of Yan et al. 2024). To compare like for like, SCRAG derives a
query-level action from the per-chunk labels by this fixed rule:

| Action | Condition on the retrieved set |
|---|---|
| `CORRECT` | at least one chunk labeled `correct` |
| `INCORRECT` | every chunk labeled `wrong` |
| `AMBIGUOUS` | otherwise — at least one `ambiguous`, no `correct` |

This mirrors CRAG's own definition, which assigns Correct when at least one document scores above
the upper threshold, Incorrect when all fall below the lower one, and Ambiguous otherwise.

**The derivation rule is part of the frozen rubric.** Changing it changes the headline number.

---

## 6. Known limits of this rubric

Stated up front so the report does not have to discover them later.

- **Per-chunk labels are ours, not CRAG's.** CRAG fine-tuned on weak relevance signals derived from
  PopQA's gold subject wiki title and randomly sampled negatives; it never published a per-chunk
  human rubric. Our per-chunk numbers therefore have no published counterpart, and only the derived
  query-level action is comparable.
- **The `ambiguous` class is intrinsically the least reliable.** Inter-annotator agreement should be
  reported per class, and `ambiguous` is expected to be the weakest. That is a finding, not a
  failure — CRAG introduced the class precisely because the boundary is soft (§4.3, Discussion).
- **Weak labels are not human labels.** Where the training set is built by rule rather than by hand,
  it must be described as weakly supervised, and the human-labeled slice exists to measure how far
  the rule diverges from this rubric.
