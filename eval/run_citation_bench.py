"""Phase 3 benchmark: citation parseability, invalid ids, and citation P/R on ALCE.

    python -m eval.run_citation_bench --dataset alce --limit 200
    python -m eval.run_citation_bench --dataset alce --limit 10 --dry-run

**The gate is the parseable rate: 100% of answer sentences must carry a
citation that resolves to a passage actually retrieved.**

Three failure types are counted separately, because merging them hides all
three:

1. **Parse failure** - the sentence carries no readable citation.
2. **Invalid id** - the citation is readable but names a passage outside the
   retrieved set. The model invented a source. This is a hard failure, not a
   formatting problem.
3. **Unsupported citation** - the citation is well-formed and real, but the
   passage does not actually support the sentence.

Only 1 and 2 are measured here. **Type 3 requires entailment, which is Module D
in Phase 4**, and this script does not attempt it. What it reports instead is a
clearly-labelled *lexical proxy*, which is weaker than ALCE's own metric and is
never to be reported as ALCE citation precision/recall.

ALCE's published citation numbers are computed with an NLI entailment model
(TRUE / T5-XXL). No ALCE baseline figure is recorded in this project's wiki, so
this script reports our numbers unqualified rather than against an assumed bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import config
from core.citation import ValidationStats
from core.generator import CitationGenerator, GenerationError
from core.types import Chunk, CitedDraft
from eval import alce as alce_mod
from eval.metrics import normalize_answer

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "was", "are", "were", "be", "been", "being", "it", "its",
    "he", "she", "they", "his", "her", "their", "this", "that", "these",
    "those", "with", "as", "by", "from", "has", "have", "had", "who", "which",
    "what", "when", "where", "not", "also", "there",
}


def content_words(text: str) -> set[str]:
    return {w for w in normalize_answer(text).split() if w and w not in STOPWORDS}


@dataclass
class ExampleResult:
    sample_id: str
    question: str
    sentences: int
    with_valid_citation: int
    sentences_with_invalid_id: int
    invalid_ids: int
    parseable_rate: float
    abstained: bool
    answer: str
    citation_support_proxy: float | None = None
    str_em: float = 0.0
    error: str | None = None
    uncited_sentences: list[str] = field(default_factory=list)


def lexical_support(sentence_text: str, cited: list[Chunk], threshold: float) -> bool:
    """Weak proxy for 'this passage supports this sentence'.

    Fraction of the sentence's content words that appear in the cited passage.
    This is NOT entailment and NOT ALCE's metric - a passage can share most of a
    sentence's vocabulary while contradicting it. Reported only as an interim
    signal until Module D lands in Phase 4.
    """
    words = content_words(sentence_text)
    if not words:
        return True
    for chunk in cited:
        overlap = len(words & content_words(chunk.text)) / len(words)
        if overlap >= threshold:
            return True
    return False


def str_em(answer: str, short_answer_sets: list[list[str]]) -> float:
    """ASQA's correctness metric: fraction of sub-answers the answer covers."""
    if not short_answer_sets:
        return 0.0
    normalized = normalize_answer(answer)
    hits = sum(
        1
        for group in short_answer_sets
        if any(normalize_answer(a) in normalized for a in group if a.strip())
    )
    return hits / len(short_answer_sets)


def run(
    examples: list[alce_mod.ALCEExample],
    k: int,
    support_threshold: float,
    verbose: bool = True,
) -> tuple[list[ExampleResult], CitationGenerator]:
    generator = CitationGenerator()
    results: list[ExampleResult] = []

    for i, ex in enumerate(examples, 1):
        context = ex.context(k)
        draft: CitedDraft = generator.generate(ex.question, context)

        if draft.raw_text.startswith("GENERATION FAILED"):
            results.append(
                ExampleResult(
                    sample_id=ex.sample_id, question=ex.question, sentences=0,
                    with_valid_citation=0, sentences_with_invalid_id=0, invalid_ids=0,
                    parseable_rate=0.0, abstained=False, answer="",
                    error=draft.raw_text,
                )
            )
            continue

        by_id = {c.chunk_id: c for c in context}
        supported = 0
        counted = 0
        uncited: list[str] = []
        for sentence in draft.sentences:
            if not sentence.parseable:
                uncited.append(sentence.text)
                continue
            counted += 1
            cited = [by_id[cid] for cid in sentence.citation_ids if cid in by_id]
            if lexical_support(sentence.text, cited, support_threshold):
                supported += 1

        usage = draft.usage
        n_sentences = usage.get("parse_sentences", len(draft.sentences))
        results.append(
            ExampleResult(
                sample_id=ex.sample_id,
                question=ex.question,
                sentences=n_sentences,
                with_valid_citation=usage.get("parse_with_valid_citation", 0),
                sentences_with_invalid_id=usage.get("parse_sentences_with_invalid_id", 0),
                invalid_ids=usage.get("parse_invalid_ids", 0),
                parseable_rate=(
                    usage.get("parse_with_valid_citation", 0) / n_sentences
                    if n_sentences else 0.0
                ),
                abstained=bool(usage.get("abstained")),
                answer=" ".join(s.text for s in draft.sentences),
                citation_support_proxy=(supported / counted) if counted else None,
                str_em=str_em(" ".join(s.text for s in draft.sentences), ex.short_answer_sets),
                uncited_sentences=uncited,
            )
        )

        if verbose:
            row = results[-1]
            mark = "OK" if row.parseable_rate == 1.0 else "!!"
            print(
                f"  [{i}/{len(examples)}] {mark} parseable {row.parseable_rate:.2f} "
                f"({row.with_valid_citation}/{row.sentences})  {ex.question[:44]}",
                flush=True,
            )

    return results, generator


def report(results: list[ExampleResult], generator: CitationGenerator, k: int, elapsed: float) -> dict:
    scored = [r for r in results if r.sentences > 0 and not r.error]
    # An abstention asserts nothing, so it cannot be an uncited claim: the
    # output contract explicitly exempts the refusal sentence from citation.
    # Excluding it from the gate denominator matches how Phase 1 treats
    # abstentions in faithfulness. Both rates are reported so the exclusion is
    # visible rather than a quiet way to pass the gate.
    answered = [r for r in scored if not r.abstained]
    abstentions = [r for r in scored if r.abstained]

    total_sentences = sum(r.sentences for r in answered)
    total_valid = sum(r.with_valid_citation for r in answered)
    all_sentences = sum(r.sentences for r in scored)
    all_valid = sum(r.with_valid_citation for r in scored)
    rate_including_abstentions = all_valid / all_sentences if all_sentences else 0.0
    total_invalid_ids = sum(r.invalid_ids for r in answered)
    sentences_with_invalid = sum(r.sentences_with_invalid_id for r in answered)

    parseable_rate = total_valid / total_sentences if total_sentences else 0.0
    invalid_id_rate = sentences_with_invalid / total_sentences if total_sentences else 0.0
    perfect = sum(1 for r in answered if r.parseable_rate == 1.0)

    proxies = [r.citation_support_proxy for r in answered if r.citation_support_proxy is not None]
    proxy_mean = sum(proxies) / len(proxies) if proxies else None
    mean_str_em = sum(r.str_em for r in answered) / len(answered) if answered else 0.0

    usage = generator.run_totals()
    paths = generator.path_totals()

    print("\n" + "=" * 72)
    print(f"  MODULE C - CITATION BENCHMARK (ALCE / ASQA, top-{k})   MEASURED")
    print("=" * 72)
    print(f"  questions {len(results)}   answered {len(answered)}   "
          f"abstained {len(abstentions)}   errors {sum(1 for r in results if r.error)}   "
          f"elapsed {elapsed:.1f}s")

    print("\n  --- THE GATE ---")
    print(f"  sentences                      {total_sentences}")
    print(f"  with a resolvable citation     {total_valid}")
    print(f"  PARSEABLE-CITATION RATE        {parseable_rate:.4f}   "
          f"({'PASS' if parseable_rate == 1.0 else 'FAIL - gate requires 1.0000'})")
    print(f"  questions fully cited          {perfect}/{len(answered)}")
    print(f"  abstentions excluded           {len(abstentions)}  "
          f"(refusals carry no claim, so no citation is owed)")
    print(f"  rate INCLUDING abstentions     {rate_including_abstentions:.4f}  "
          f"(shown so the exclusion is auditable)")

    print("\n  --- HARD FAILURES (invented sources) ---")
    print(f"  invalid citation ids           {total_invalid_ids}")
    print(f"  sentences with an invalid id   {sentences_with_invalid}  "
          f"(rate {invalid_id_rate:.4f})")

    print("\n  --- OUTPUT PATH ---")
    print(f"  structured-output path         {paths['structured_path']}")
    print(f"  text-parser fallback           {paths['text_fallback_path']}  "
          f"(rate {paths['fallback_rate']:.4f})")
    print(f"  parse retries                  {paths['parse_retries']}")

    print("\n  --- CITATION QUALITY (PROXY - NOT ALCE's METRIC) ---")
    if proxy_mean is None:
        print("  lexical support proxy          n/a")
    else:
        print(f"  lexical support proxy          {proxy_mean:.4f}")
    print("  ALCE computes citation precision/recall with an NLI entailment model.")
    print("  That is Module D, Phase 4. The number above is word-overlap only and")
    print("  must NOT be reported as ALCE citation precision or recall.")
    print("  No ALCE baseline figure is recorded in this project, so ours is")
    print("  reported unqualified rather than against an assumed bar.")

    print(f"\n  --- ANSWER CORRECTNESS ---")
    print(f"  ASQA str-EM                    {mean_str_em:.4f}")

    print("\n  --- MEASURED API USAGE ---")
    print(f"  api calls                      {usage['api_calls']} of {usage['calls']} "
          f"({usage['cache_hits_on_disk']} from disk cache)")
    print(f"  input / output tokens          {usage['input_tokens']:,} / {usage['output_tokens']:,}")
    print(f"  cache writes / reads           {usage['cache_creation_input_tokens']:,} / "
          f"{usage['cache_read_input_tokens']:,}")
    print(f"  prompt cache working           {usage['prompt_cache_working']}")
    print(f"  TOTAL cost                     ${usage['total_cost_usd']:.4f}")
    print(f"  cost per question              ${usage['cost_per_query_usd']:.5f}")
    print("=" * 72)

    return {
        "dataset": f"ALCE/ASQA top-{k}",
        "questions": len(results),
        "answered": len(answered),
        "errors": sum(1 for r in results if r.error),
        "abstentions": len(abstentions),
        "gate": {
            "parseable_citation_rate": round(parseable_rate, 4),
            "parseable_rate_including_abstentions": round(rate_including_abstentions, 4),
            "abstentions_excluded": len(abstentions),
            "why_excluded": (
                "A refusal makes no claim, and the output contract exempts it from "
                "citation. Both rates are reported."
            ),
            "passes": parseable_rate == 1.0,
            "sentences": total_sentences,
            "sentences_with_valid_citation": total_valid,
            "questions_fully_cited": perfect,
        },
        "hard_failures": {
            "invalid_citation_ids": total_invalid_ids,
            "sentences_with_invalid_id": sentences_with_invalid,
            "invalid_id_rate": round(invalid_id_rate, 4),
        },
        "output_path": paths,
        "citation_quality_proxy": {
            "lexical_support_proxy": round(proxy_mean, 4) if proxy_mean is not None else None,
            "WARNING": (
                "Word-overlap proxy only. NOT ALCE citation precision/recall, which "
                "requires NLI entailment (Module D, Phase 4). Do not report as ALCE."
            ),
        },
        "answer_correctness": {"asqa_str_em": round(mean_str_em, 4)},
        "alce_baseline": (
            "NOT AVAILABLE - the ALCE paper is not recorded in this project's wiki. "
            "Our numbers are reported unqualified."
        ),
        "api_usage": usage,
        "config": {
            "model": config.GENERATOR.model,
            "effort": config.GENERATOR.effort,
            "structured_outputs": config.GENERATOR.use_structured_outputs,
            "max_citation_retries": config.GENERATOR.max_citation_retries,
            "top_k": k,
        },
        "examples": [asdict(r) for r in results],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Module C citation benchmark")
    parser.add_argument("--dataset", default="alce", choices=["alce"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=5, help="passages per question")
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--out", default="citation_bench.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="price the run with count_tokens and exit without generating",
    )
    args = parser.parse_args()

    if not config.has_api_key():
        print("ANTHROPIC_API_KEY is not set. Module C calls the Claude API.", file=sys.stderr)
        raise SystemExit(2)

    examples = alce_mod.load_asqa(limit=args.limit)
    print(f"ALCE/ASQA: {len(examples)} questions, top-{args.k} passages")
    print(f"model={config.GENERATOR.model} effort={config.GENERATOR.effort} "
          f"structured={config.GENERATOR.use_structured_outputs}\n")

    if args.dry_run:
        generator = CitationGenerator()
        prefix = generator.count_prompt_tokens()
        sample = examples[0]
        counted = generator.client.messages.count_tokens(
            model=config.GENERATOR.model,
            system=[{"type": "text", "text": generator.SYSTEM_PROMPT}],
            messages=[{
                "role": "user",
                "content": generator.build_prompt(sample.question, sample.context(args.k)),
            }],
        )
        per_call_in = counted.input_tokens
        est_out = 300
        cfg = config.GENERATOR
        est = len(examples) * (
            per_call_in * cfg.price_input_per_mtok + est_out * cfg.price_output_per_mtok
        ) / 1_000_000
        print(f"  cacheable prefix       {prefix} tokens "
              f"(cache minimum {cfg.min_cacheable_prefix_tokens}) -> "
              f"{'WILL cache' if prefix >= cfg.min_cacheable_prefix_tokens else 'will NOT cache'}")
        print(f"  input tokens / call    {per_call_in}")
        print(f"  assumed output tokens  {est_out} (ESTIMATE, not measured)")
        print(f"  estimated cost         ${est:.2f} for {len(examples)} questions")
        print("\n  Estimate only - no generation ran. Caching and disk cache hits will")
        print("  reduce this. Re-run without --dry-run to measure it for real.")
        raise SystemExit(0)

    started = time.time()
    try:
        results, generator = run(examples, args.k, args.support_threshold, not args.quiet)
    except GenerationError as exc:
        print(f"\nRUN FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

    payload = report(results, generator, args.k, time.time() - started)
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
