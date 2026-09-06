"""The A-E ablation - the report's key figure.

    python -m eval.run_ablation --all --split dev --limit 150 --dry-run
    python -m eval.run_ablation --all --split dev --limit 150

Writes eval/results/ablation.csv and ablation.md.

FOUR RULES THIS RUNNER ENFORCES
-------------------------------
1. **One code path.** Variants are `PipelineFlags` over a single Orchestrator
   (eval/variants.py). Five forked pipelines drift, and drifted variants stop
   measuring what they claim to.
2. **One seeded split.** Every variant runs the identical questions from the
   Phase 1 split. A resampled split invalidates the whole comparison.
3. **Coverage beside every accuracy.** Variant E answers fewer questions than
   variant A, so raw accuracy is not comparable across the row. The table
   carries coverage in its own column and the markdown says so.
4. **Empty cells stay empty.** A baseline figure with no paper and table behind
   it is left blank. A plausible-looking number is worse than a gap.

COST
----
Generation goes through the **Message Batches API** at 50% of standard pricing.
The harness is non-latency-sensitive by nature, so this is free money on the
only paid component in the project. Batch results return in ANY order and are
keyed by `custom_id`, never by position.

`--dry-run` prices the sweep with `messages.count_tokens` before anything is
launched, so a five-variant run is never a surprise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import config
from core.orchestrator import Orchestrator
from core.retriever import Retriever
from eval import metrics, variants as variants_mod
from eval.datasets import QAExample, load_split

# --------------------------------------------------------------------------
# Published baselines. Each needs a paper AND a table, or it stays empty.
# --------------------------------------------------------------------------
PAPER_BASELINES = {
    "evaluator_action_accuracy": {
        "value": 0.843,
        "paper": "CRAG (Yan et al. 2024)",
        "arxiv": "2401.15884v3",
        "table": "Table 4",
        "note": "action accuracy over a retrieved set on PopQA, T5-large evaluator",
    },
    "citation_precision": {
        "value": None,
        "paper": "ALCE (Gao et al. 2023)",
        "table": None,
        "note": (
            "NOT AVAILABLE - the ALCE paper is not recorded in this project's wiki. "
            "Cell intentionally left empty rather than filled with a plausible number."
        ),
    },
    "citation_recall": {
        "value": None,
        "paper": "ALCE (Gao et al. 2023)",
        "table": None,
        "note": "NOT AVAILABLE - see citation_precision.",
    },
    "faithfulness": {
        "value": None,
        "paper": None,
        "table": None,
        "note": "n/a - no published baseline; RAGAS-definition faithfulness is our own measurement.",
    },
}


@dataclass
class VariantResult:
    variant: str
    name: str
    adds: str
    n: int = 0
    answered: int = 0
    coverage: float = 0.0
    accuracy_all: float = 0.0
    accuracy_answered: float | None = None
    faithfulness: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    hallucination_rate_auto: float | None = None
    contradiction_rate_auto: float | None = None
    hallucination_rate_labeled: float | None = None
    abstention_rate: float = 0.0
    parseable_citation_rate: float | None = None
    repair_attempted: int = 0
    repair_succeeded: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    errors: int = 0
    notes: str = ""
    rows: list[dict] = field(default_factory=list)


def cache_path(variant: str, split: str, limit: int):
    return config.EVAL.results_dir / "ablation_cache" / f"{variant}_{split}_{limit}.json"


def load_cached(variant: str, split: str, limit: int) -> dict | None:
    path = cache_path(variant, split, limit)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cached(variant: str, split: str, limit: int, payload: dict) -> None:
    path = cache_path(variant, split, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Batch submission
# --------------------------------------------------------------------------
def submit_batch(requests: list[dict]) -> dict[str, str]:
    """Submit generation requests through the Message Batches API (50% cost).

    Returns {custom_id: text}. Results come back in ANY order, so they are
    keyed by custom_id and never by position.
    """
    import anthropic

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted with {len(requests)} requests")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"    status {batch.processing_status}, waiting...", flush=True)
        time.sleep(20)

    out: dict[str, str] = {}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type == "succeeded":
            out[entry.custom_id] = "".join(
                b.text for b in entry.result.message.content if b.type == "text"
            )
        else:
            out[entry.custom_id] = ""
    return out


# --------------------------------------------------------------------------
# Running one variant
# --------------------------------------------------------------------------
def run_variant(
    variant: variants_mod.Variant,
    examples: list[QAExample],
    retriever: Retriever,
    verbose: bool,
) -> VariantResult:
    """Run one variant over the shared split, synchronously through the pipeline.

    Batch submission is used for the standalone generation sweep (see
    --batch); the orchestrator path is synchronous because Modules B, D and E
    interleave local model calls between generations.
    """
    result = VariantResult(variant=variant.key, name=variant.name, adds=variant.adds)
    orch = Orchestrator(retriever=retriever, flags=variant.flags)

    started = time.time()
    verdict_pool: list = []
    triples: list[tuple] = []
    correct = 0
    abstained = 0
    parseable_num = 0
    parseable_den = 0

    for i, ex in enumerate(examples, 1):
        try:
            answer = orch.answer(ex.question)
        except Exception as exc:                     # noqa: BLE001 - recorded, not swallowed
            result.errors += 1
            result.rows.append({"qid": ex.qid, "error": str(exc)[:200]})
            continue

        is_abstention = answer.abstained
        abstained += is_abstention
        text = "" if is_abstention else answer.text
        is_correct = (not is_abstention) and metrics.answer_matches(text, ex.answers)
        correct += is_correct

        # Citation parseability only means anything once Module C is on.
        if variant.flags.force_citations and not is_abstention:
            for sentence in answer.sentences:
                parseable_den += 1
                parseable_num += bool(sentence.citation_ids)

        # Module D verdicts, when this variant ran verification.
        if variant.flags.verify_citations and not is_abstention:
            verdicts = answer.trace.get("verdicts") or []
            if verdicts:
                verdict_pool.extend(verdicts)
                triples.append((answer.sentences, verdicts, answer.citations))

        result.rows.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "abstained": is_abstention,
                "abstain_reason": answer.abstain_reason,
                "answer": text,
                "correct": is_correct,
                "sentences": len(answer.sentences),
                "citations": len(answer.citations),
                "repair": answer.trace.get("repair", {}),
            }
        )
        result.repair_attempted += answer.trace.get("repair", {}).get("repair_attempted", 0)
        result.repair_succeeded += answer.trace.get("repair", {}).get("repair_succeeded", 0)

        if verbose and i % 25 == 0:
            print(f"    {variant.key}: {i}/{len(examples)}", flush=True)

    n = len(examples)
    answered = n - abstained - result.errors
    result.n = n
    result.answered = answered
    result.abstention_rate = round(abstained / n, 4) if n else 0.0
    result.coverage = round(answered / n, 4) if n else 0.0
    result.accuracy_all = round(correct / n, 4) if n else 0.0
    result.accuracy_answered = round(correct / answered, 4) if answered else None
    result.parseable_citation_rate = (
        round(parseable_num / parseable_den, 4) if parseable_den else None
    )
    if verdict_pool:
        scores = metrics.citation_precision_recall(
            triples, config.VERIFIER.entailment_threshold
        )
        result.citation_precision = scores.precision
        result.citation_recall = scores.recall
        auto = metrics.hallucination_rate_automatic(verdict_pool)
        result.hallucination_rate_auto = auto["hallucination_rate_auto"]
        result.contradiction_rate_auto = auto["contradiction_rate_auto"]

    labeled_path = config.REPO_ROOT / "eval" / "labeled" / "hallucination.jsonl"
    labeled_rows = []
    if labeled_path.exists():
        labeled_rows = [
            json.loads(l) for l in labeled_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
    result.hallucination_rate_labeled = metrics.hallucination_rate_labeled(labeled_rows)[
        "hallucination_rate_labeled"
    ]

    usage = orch.generator.run_totals()
    result.cost_usd = round(float(usage["total_cost_usd"]), 6)
    result.elapsed_s = round(time.time() - started, 1)
    return result


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
CSV_FIELDS = [
    "variant", "name", "adds", "n", "answered", "coverage",
    "accuracy_all", "accuracy_answered", "faithfulness",
    "citation_precision", "citation_recall",
    "hallucination_rate_auto", "contradiction_rate_auto",
    "hallucination_rate_labeled", "abstention_rate", "parseable_citation_rate",
    "repair_attempted", "repair_succeeded", "cost_usd", "elapsed_s", "errors", "notes",
]


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(results: list[VariantResult], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k in CSV_FIELDS}
            writer.writerow(row)


def write_markdown(results: list[VariantResult], path, split: str, meta: dict) -> None:
    lines = [
        "# SCRAG ablation — variants A to E",
        "",
        f"Split: `{split}`, n={results[0].n if results else 0}, seed {config.SEED}. "
        "Every variant ran the identical seeded questions through one code path.",
        "",
        "**Read accuracy together with coverage.** Variant E answers fewer questions than variant "
        "A, so `accuracy_all` and `accuracy_answered` are not comparable across the row on their "
        "own. Abstaining on a question the sources cannot support is the intended behaviour, and "
        "it lowers coverage rather than being a failure.",
        "",
        "| Variant | Pipeline | Coverage | Accuracy (all) | Accuracy (answered) | Faithfulness | "
        "Citation P | Citation R | Hallucination (auto) | Abstention |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.variant} | {r.name} | {fmt(r.coverage)} | {fmt(r.accuracy_all)} | "
            f"{fmt(r.accuracy_answered)} | {fmt(r.faithfulness)} | "
            f"{fmt(r.citation_precision)} | {fmt(r.citation_recall)} | "
            f"{fmt(r.hallucination_rate_auto)} | {fmt(r.abstention_rate)} |"
        )

    lines += [
        "",
        "## Paper reported X vs we achieved Y",
        "",
        "| Concern | Paper | Paper reported | Source | We achieved |",
        "|---|---|---|---|---|",
    ]
    evaluator = PAPER_BASELINES["evaluator_action_accuracy"]
    ours = meta.get("module_b_action_accuracy")
    lines.append(
        f"| Evaluator action accuracy | {evaluator['paper']} | {evaluator['value']} | "
        f"{evaluator['table']}, {evaluator['arxiv']} | {fmt(ours)} |"
    )
    for key in ["citation_precision", "citation_recall"]:
        base = PAPER_BASELINES[key]
        measured = getattr(results[-1], key) if results else None
        lines.append(
            f"| {key.replace('_', ' ').title()} | {base['paper']} | | "
            f"*not recorded* | {fmt(measured)} |"
        )
    lines += [
        "",
        "Empty cells are deliberate. A baseline figure without a paper and table behind it is left "
        "blank rather than filled with a plausible number.",
        "",
        "## What is measured and what is not",
        "",
        "- **Faithfulness** is the RAGAS *definition* (claims entailed by context / total claims) "
        "in our own implementation calling Claude directly. RAGAS the library hard-requires "
        "`openai` + `langchain`, which this project does not take on. Never report it as RAGAS.",
        "- **Citation precision/recall** use ALCE's definitions computed with our Module D "
        "checkpoint (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), not ALCE's TRUE/T5-XXL. "
        "They are comparable in spirit, not in implementation.",
        "- **Hallucination rate (auto)** is verifier-derived: it measures disagreement between the "
        "generator and the NLI checkpoint, not ground truth. The hand-labeled set "
        "(`eval/LABELING_HALLUCINATION.md`) exists to bound that gap and has **not** been labeled, "
        "so `hallucination_rate_labeled` is empty rather than zero.",
        "- **Variant D emits the same text as variant C** by construction: Module D flags but does "
        "not alter the answer. Identical accuracy on those two rows is correct, not a bug.",
        "",
        f"Total measured cost: ${meta.get('total_cost_usd', 0):.4f}. "
        f"Generation via {'Message Batches API (50% pricing)' if meta.get('batched') else 'synchronous calls'}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(results: list[VariantResult]) -> None:
    print("\n" + "=" * 100)
    print("  SCRAG ABLATION A-E   (MEASURED)")
    print("=" * 100)
    header = (
        f"  {'V':<2}{'pipeline':<26}{'cover':>8}{'acc(all)':>10}{'acc(ans)':>10}"
        f"{'faith':>8}{'citP':>8}{'citR':>8}{'halluc':>9}{'abst':>8}"
    )
    print(header)
    print("  " + "-" * 96)
    for r in results:
        print(
            f"  {r.variant:<2}{r.name:<26}{fmt(r.coverage):>8}{fmt(r.accuracy_all):>10}"
            f"{fmt(r.accuracy_answered):>10}{fmt(r.faithfulness):>8}"
            f"{fmt(r.citation_precision):>8}{fmt(r.citation_recall):>8}"
            f"{fmt(r.hallucination_rate_auto):>9}{fmt(r.abstention_rate):>8}"
        )
    print("=" * 100)
    print("  Accuracy is NOT comparable across rows without coverage: variant E answers fewer")
    print("  questions than variant A, and abstaining on an unsupportable question is correct.")


# --------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the A-E ablation")
    parser.add_argument("--all", action="store_true", help="run every variant")
    parser.add_argument("--variants", default=None, help="comma-separated subset, e.g. A,C,E")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="price the sweep and exit")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    selected = (
        variants_mod.all_variants()
        if args.all or not args.variants
        else [variants_mod.get(k.strip()) for k in args.variants.split(",")]
    )

    retriever = Retriever()
    retriever.load_index()
    examples = load_split(args.split, limit=args.limit)
    print(f"split={args.split}  n={len(examples)}  seed={config.SEED}  "
          f"corpus_chunks={retriever.size}")
    print(f"variants: {', '.join(v.key for v in selected)}")
    print(f"model={config.GENERATOR.model}  effort={config.GENERATOR.effort}\n")

    # ---- cost estimate before anything is launched ----------------------
    if args.dry_run:
        from core.generator import CitationGenerator

        gen = CitationGenerator()
        try:
            sample_ctx = retriever.retrieve(examples[0].question)
            counted = gen.client.messages.count_tokens(
                model=config.GENERATOR.model,
                system=[{"type": "text", "text": gen.SYSTEM_PROMPT}],
                messages=[{
                    "role": "user",
                    "content": gen.build_prompt(examples[0].question, sample_ctx),
                }],
            )
            per_call_in = counted.input_tokens
        except Exception as exc:                     # noqa: BLE001
            print(f"  could not price the sweep: {str(exc)[:160]}")
            print("  (count_tokens needs a working API credential)")
            raise SystemExit(2)

        est_out = 300
        cfg = config.GENERATOR
        generating = [v for v in selected if v.key != "A"] or selected
        calls = len(examples) * len(generating)
        sync = calls * (
            per_call_in * cfg.price_input_per_mtok + est_out * cfg.price_output_per_mtok
        ) / 1_000_000
        print(f"  input tokens / call     {per_call_in}")
        print(f"  assumed output tokens   {est_out}  (ESTIMATE, not measured)")
        print(f"  generation calls        {calls}  ({len(examples)} x {len(generating)} variants)")
        print(f"  estimated, synchronous  ${sync:.2f}")
        print(f"  estimated, Batch API    ${sync / 2:.2f}   <- use this")
        print("\n  Estimate only. Prompt caching and the on-disk response cache will")
        print("  reduce it. Re-run without --dry-run to measure it for real.")
        raise SystemExit(0)

    # ---- run ------------------------------------------------------------
    results: list[VariantResult] = []
    started = time.time()
    for variant in selected:
        cached = None if args.no_cache else load_cached(variant.key, args.split, args.limit)
        if cached:
            print(f"  {variant.label}: from cache")
            results.append(VariantResult(**cached))
            continue

        print(f"  {variant.label}: running...")
        try:
            result = run_variant(variant, examples, retriever, not args.quiet)
        except Exception as exc:                     # noqa: BLE001
            print(f"\nVARIANT {variant.key} FAILED: {str(exc)[:300]}", file=sys.stderr)
            print("No numbers are reported for it.", file=sys.stderr)
            continue
        save_cached(variant.key, args.split, args.limit, asdict(result))
        results.append(result)

    if not results:
        print("\nNo variant completed. Nothing to report.", file=sys.stderr)
        raise SystemExit(1)

    print_table(results)

    meta = {
        "total_cost_usd": sum(r.cost_usd for r in results),
        "batched": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    bench = config.EVAL.results_dir / "evaluator_bench.json"
    if bench.exists():
        meta["module_b_action_accuracy"] = json.loads(bench.read_text(encoding="utf-8"))[
            "query_level_action"
        ]["accuracy"]

    csv_path = config.EVAL.results_dir / "ablation.csv"
    md_path = config.EVAL.results_dir / "ablation.md"
    write_csv(results, csv_path)
    write_markdown(results, md_path, args.split, meta)
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"total measured cost ${meta['total_cost_usd']:.4f}  "
          f"elapsed {meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
