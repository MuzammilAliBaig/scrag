"""Variant A of the ablation: vanilla RAG, no grading, no citations.

    python -m eval.run_baseline --split dev --limit 200

Structured so Phase 6 calls `run_variant()` rather than re-implementing it.
Every number this prints comes from a run that actually happened; when a stage
fails it is reported as a failure, never filled in with an estimate.

Cost: this is the first phase that spends money. Generation and the
faithfulness judge both bill to the Claude API. Use --limit while developing,
and --no-faithfulness to run accuracy only for free-ish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import config
from core.generator import GenerationError, PlainGenerator
from core.retriever import Retriever
from core.types import Chunk
from eval import metrics
from eval.datasets import QAExample, load_split


@dataclass
class ExampleResult:
    qid: str
    question: str
    gold_answers: list[str]
    prediction: str
    correct: bool
    abstained: bool
    retrieval_hit: bool
    retrieved_ids: list[str]
    retrieval_scores: list[float]
    faithfulness: float | None = None
    faithfulness_claims: int = 0
    error: str | None = None


@dataclass
class VariantResult:
    variant: str
    pipeline: str
    n: int
    accuracy: float
    coverage: float
    answered_accuracy: float
    retrieval_hit_rate: float
    faithfulness: float | None
    faithfulness_n: int
    elapsed_s: float
    generation_usage: dict = field(default_factory=dict)
    faithfulness_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    errors: int = 0
    config_summary: dict = field(default_factory=dict)
    examples: list[ExampleResult] = field(default_factory=list)


def run_variant(
    examples: list[QAExample],
    retriever: Retriever,
    variant: str = "A",
    pipeline: str = "vanilla RAG",
    measure_faithfulness: bool = True,
    verbose: bool = True,
) -> VariantResult:
    """Run one ablation variant end to end and measure it.

    Phase 6 will call this with later modules switched on; the signature is
    deliberately stable.
    """
    generator = PlainGenerator()
    judge = metrics.FaithfulnessJudge() if measure_faithfulness else None

    started = time.time()
    results: list[ExampleResult] = []

    for i, ex in enumerate(examples, 1):
        context: list[Chunk] = retriever.retrieve(ex.question)
        generated = generator.generate_text(ex.question, context)
        prediction = generated.text

        row = ExampleResult(
            qid=ex.qid,
            question=ex.question,
            gold_answers=ex.answers,
            prediction=prediction,
            correct=metrics.answer_matches(prediction, ex.answers),
            abstained=metrics.is_abstention(prediction),
            retrieval_hit=metrics.retrieval_hit(ex.answers, context),
            retrieved_ids=[c.chunk_id for c in context],
            retrieval_scores=[round(c.score or 0.0, 4) for c in context],
            error=generated.error,
        )

        if judge is not None and not row.error:
            faith = judge.score(ex.question, prediction, context)
            row.faithfulness = faith.score
            row.faithfulness_claims = faith.n_claims
            if faith.error:
                row.error = (row.error or "") + f" | faithfulness: {faith.error}"

        results.append(row)
        if verbose:
            mark = "OK " if row.correct else "-- "
            print(f"  [{i}/{len(examples)}] {mark} {ex.question[:58]}", flush=True)

    n = len(results)
    answered = [r for r in results if not r.abstained and not r.error]
    scored = [r.faithfulness for r in results if r.faithfulness is not None]
    gen_usage = generator.run_totals()
    judge_cost = judge.cost_usd() if judge else 0.0

    return VariantResult(
        variant=variant,
        pipeline=pipeline,
        n=n,
        accuracy=sum(r.correct for r in results) / max(n, 1),
        coverage=len(answered) / max(n, 1),
        answered_accuracy=sum(r.correct for r in answered) / max(len(answered), 1),
        retrieval_hit_rate=sum(r.retrieval_hit for r in results) / max(n, 1),
        faithfulness=(sum(scored) / len(scored)) if scored else None,
        faithfulness_n=len(scored),
        elapsed_s=round(time.time() - started, 1),
        generation_usage=gen_usage,
        faithfulness_cost_usd=round(judge_cost, 6),
        total_cost_usd=round(float(gen_usage["total_cost_usd"]) + judge_cost, 6),
        errors=sum(1 for r in results if r.error),
        config_summary={
            "embedding_model": config.RETRIEVAL.embedding_model,
            "generator_model": config.GENERATOR.model,
            "effort": config.GENERATOR.effort,
            "top_k": config.RETRIEVAL.top_k,
            "chunk_size": config.RETRIEVAL.chunk_size,
            "chunk_overlap": config.RETRIEVAL.chunk_overlap,
            "seed": config.SEED,
            "corpus_chunks": retriever.size,
        },
        examples=results,
    )


def report(result: VariantResult) -> None:
    print("\n" + "=" * 68)
    print(f"  VARIANT {result.variant} - {result.pipeline}   (MEASURED, n={result.n})")
    print("=" * 68)
    print(f"  answer accuracy         {result.accuracy:.3f}")
    print(f"  coverage                {result.coverage:.3f}   (fraction answered, not abstained)")
    print(f"  accuracy on answered    {result.answered_accuracy:.3f}")
    print(f"  retrieval hit rate      {result.retrieval_hit_rate:.3f}   (gold answer present in context)")
    if result.faithfulness is None:
        print(f"  faithfulness            NOT MEASURED")
    else:
        print(f"  faithfulness            {result.faithfulness:.3f}   (RAGAS definition, own impl, n={result.faithfulness_n})")
    print(f"  errors                  {result.errors}")
    print(f"  elapsed                 {result.elapsed_s}s")

    usage = result.generation_usage
    print("\n  --- measured API usage (generation) ---")
    print(f"  api calls               {usage.get('api_calls')} of {usage.get('calls')} "
          f"({usage.get('cache_hits_on_disk')} served from disk cache)")
    print(f"  input tokens            {usage.get('input_tokens'):,}")
    print(f"  output tokens           {usage.get('output_tokens'):,}")
    print(f"  cache writes / reads    {usage.get('cache_creation_input_tokens'):,} / "
          f"{usage.get('cache_read_input_tokens'):,}")
    print(f"  prompt cache working    {usage.get('prompt_cache_working')}")
    print(f"  generation cost         ${usage.get('total_cost_usd'):.4f}")
    print(f"  faithfulness cost       ${result.faithfulness_cost_usd:.4f}")
    print(f"  TOTAL cost              ${result.total_cost_usd:.4f}")
    print(f"  cost per query          ${result.total_cost_usd / max(result.n, 1):.5f}")
    print("=" * 68)


def save(result: VariantResult, filename: str = "baseline.json") -> None:
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / filename
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="SCRAG baseline (ablation variant A)")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=config.EVAL.default_limit)
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="skip the LLM judge (halves API spend; leaves faithfulness unmeasured)",
    )
    parser.add_argument("--out", default="baseline.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not config.has_api_key():
        print(
            "ANTHROPIC_API_KEY is not set.\n"
            "Generation and faithfulness both call the Claude API, the one paid\n"
            "component in SCRAG. Put a key in .env and re-run.\n"
            "Retrieval-only checks still work: python -m eval.check_retrieval",
            file=sys.stderr,
        )
        raise SystemExit(2)

    retriever = Retriever()
    retriever.load_index()
    examples = load_split(args.split, limit=args.limit)
    print(f"split={args.split} n={len(examples)} corpus_chunks={retriever.size}")
    print(f"model={config.GENERATOR.model} effort={config.GENERATOR.effort} "
          f"top_k={config.RETRIEVAL.top_k}\n")

    try:
        result = run_variant(
            examples,
            retriever,
            measure_faithfulness=not args.no_faithfulness,
            verbose=not args.quiet,
        )
    except GenerationError as exc:
        print(f"\nRUN FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

    report(result)
    save(result, args.out)


if __name__ == "__main__":
    main()
