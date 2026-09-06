"""Phase 5 benchmark: risk-coverage on an answerable / unanswerable split.

    python -m eval.run_abstention_bench --limit 150

THE SUCCESS CRITERION IS COUNTER-INTUITIVE
------------------------------------------
Raw accuracy over all questions is the wrong measure here, and will usually
look *worse* than the Phase 1 baseline. That is expected. The system is
supposed to refuse questions it cannot support, and a refusal scores zero on
raw accuracy while being exactly the right behaviour.

What should improve is **accuracy on the questions still answered**. Abstaining
on the hard ones is a win traded against coverage, which is why this reports a
risk-coverage curve rather than a single number.

THE SPLIT
---------
PopQA dev questions divide cleanly by whether Module A's corpus contains the
subject entity's Wikipedia page:

* **answerable**   - subject page is in the corpus, so an answer is reachable
* **unanswerable** - subject page is absent, so no retrieved passage can
  support an answer, and the correct behaviour is to abstain every time

The unanswerable set is the honest test of abstention: on it, abstention rate
is the metric and any confident answer is a hallucination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import config
from core.orchestrator import Orchestrator, PipelineFlags
from core.retriever import Retriever
from eval import metrics
from eval.datasets import QAExample, load_split


@dataclass
class Row:
    qid: str
    question: str
    subset: str                    # "answerable" | "unanswerable"
    abstained: bool
    abstain_reason: str | None
    answer: str
    correct: bool
    confidence: float              # mean entailment over surviving sentences
    sentences: int
    citations: int
    repair: dict = field(default_factory=dict)
    module_b_action: str = ""
    corrective_fired: bool = False


def split_by_corpus(retriever: Retriever, limit: int) -> tuple[list[QAExample], list[QAExample]]:
    """Partition dev questions by whether their subject page is in the corpus."""
    titles = {c.doc_id.replace("_", " ").lower() for c in retriever._chunks}
    answerable: list[QAExample] = []
    unanswerable: list[QAExample] = []
    for ex in load_split("dev"):
        key = (ex.subject_title or "").lower()
        target = answerable if key in titles else unanswerable
        if len(target) < limit:
            target.append(ex)
        if len(answerable) >= limit and len(unanswerable) >= limit:
            break
    return answerable, unanswerable


def confidence_of(answer) -> float:
    """Mean entailment across surviving sentences, used to rank for risk-coverage.

    An abstention has no confidence; it is excluded from the curve rather than
    scored zero, because it is not a wrong answer.
    """
    scores = [
        v for v in (answer.trace.get("sentence_scores") or []) if isinstance(v, (int, float))
    ]
    return sum(scores) / len(scores) if scores else 1.0


def risk_coverage(rows: list[Row]) -> list[dict]:
    """Accuracy as a function of how many questions the system chooses to answer.

    Answers are ranked by confidence; the curve sweeps coverage from the most
    confident subset outward. A well-behaved selective system has accuracy
    falling as coverage rises.
    """
    answered = sorted(
        [r for r in rows if not r.abstained], key=lambda r: r.confidence, reverse=True
    )
    total = len(rows)
    curve = []
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        take = max(1, int(len(answered) * fraction)) if answered else 0
        subset = answered[:take]
        if not subset:
            continue
        accuracy = sum(r.correct for r in subset) / len(subset)
        curve.append(
            {
                "fraction_of_answered": fraction,
                "n": len(subset),
                "coverage_of_all": round(len(subset) / total, 4) if total else 0.0,
                "accuracy": round(accuracy, 4),
                "risk": round(1 - accuracy, 4),
            }
        )
    return curve


def run_subset(
    orch: Orchestrator, examples: list[QAExample], subset: str, verbose: bool
) -> list[Row]:
    rows: list[Row] = []
    for i, ex in enumerate(examples, 1):
        answer = orch.answer(ex.question)
        text = answer.text if not answer.abstained else ""
        rows.append(
            Row(
                qid=ex.qid,
                question=ex.question,
                subset=subset,
                abstained=answer.abstained,
                abstain_reason=answer.abstain_reason,
                answer=text,
                correct=(not answer.abstained) and metrics.answer_matches(text, ex.answers),
                confidence=confidence_of(answer),
                sentences=len(answer.sentences),
                citations=len(answer.citations),
                repair=answer.trace.get("repair", {}),
                module_b_action=answer.trace.get("retrieval", {}).get("action", ""),
                corrective_fired=answer.trace.get("retrieval", {}).get("corrective_fired", False),
            )
        )
        if verbose:
            row = rows[-1]
            mark = "ABST" if row.abstained else ("OK  " if row.correct else "WRONG")
            print(f"  [{subset} {i}/{len(examples)}] {mark}  {ex.question[:50]}", flush=True)
    return rows


def summarize(rows: list[Row], label: str) -> dict:
    n = len(rows)
    if not n:
        return {}
    abstained = [r for r in rows if r.abstained]
    answered = [r for r in rows if not r.abstained]
    correct = sum(r.correct for r in answered)
    return {
        "subset": label,
        "n": n,
        "abstention_rate": round(len(abstained) / n, 4),
        "coverage": round(len(answered) / n, 4),
        "raw_accuracy": round(correct / n, 4),
        "answered_subset_accuracy": round(correct / len(answered), 4) if answered else None,
        "abstain_reasons": {
            reason: sum(1 for r in abstained if r.abstain_reason == reason)
            for reason in sorted({r.abstain_reason for r in abstained if r.abstain_reason})
        },
    }


def _retrieval_only(args) -> None:
    """Measure abstention trigger (b) alone. No API calls, so no cost.

    Trigger (b) fires when Module B grades every retrieved chunk wrong, and it
    fires BEFORE the generator is called. On the unanswerable subset it is the
    only thing standing between the system and a hallucination, and it is
    measurable without spending anything.
    """
    retriever = Retriever()
    retriever.load_index()
    answerable, unanswerable = split_by_corpus(retriever, args.limit)
    orch = Orchestrator(retriever=retriever)

    print(f"answerable   : {len(answerable)} questions (subject page in corpus)")
    print(f"unanswerable : {len(unanswerable)} questions (subject page absent)")
    print("mode         : NO GENERATION - Module A + B only, zero API cost")
    print()

    out = {}
    started = time.time()
    for label, examples in [("answerable", answerable), ("unanswerable", unanswerable)]:
        pre_abstain = 0
        corrective = 0
        actions = {}
        for ex in examples:
            _, trace = orch.retrieve_and_grade(ex.question)
            actions[trace.action] = actions.get(trace.action, 0) + 1
            pre_abstain += bool(trace.abstained_before_generation)
            corrective += bool(trace.corrective_fired)
        n = len(examples)
        out[label] = {
            "n": n,
            "pre_generation_abstention_rate": round(pre_abstain / n, 4) if n else 0.0,
            "corrective_loop_rate": round(corrective / n, 4) if n else 0.0,
            "module_b_actions": actions,
        }
        print(f"  --- {label.upper()} (n={n}) ---")
        print(f"  trigger (b) fired          {pre_abstain}  "
              f"({out[label]['pre_generation_abstention_rate']:.4f})")
        print(f"  corrective loop fired      {corrective}  "
              f"({out[label]['corrective_loop_rate']:.4f})")
        print(f"  Module B actions           {actions}")
        print()

    print("  On the unanswerable subset, trigger (b) is the only thing preventing")
    print("  a hallucination, and every one it catches costs nothing - the paid")
    print("  component is never reached.")

    payload = {
        "mode": "retrieval_and_grading_only",
        "api_cost_usd": 0.0,
        "note": (
            "Partial Phase 5 measurement. Generation, repair and the full "
            "risk-coverage curve require API credit."
        ),
        "subsets": out,
        "elapsed_s": round(time.time() - started, 1),
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / "abstention_bench_retrieval_only.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phase 5 risk-coverage benchmark")
    parser.add_argument("--limit", type=int, default=150, help="questions per subset")
    parser.add_argument("--out", default="abstention_bench.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help=(
            "stop after Module B and report only the pre-generation abstention "
            "rate (trigger b). Free: makes no API calls."
        ),
    )
    args = parser.parse_args()

    if args.no_generation:
        _retrieval_only(args)
        return

    if not config.has_api_key():
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        raise SystemExit(2)

    retriever = Retriever()
    retriever.load_index()
    answerable, unanswerable = split_by_corpus(retriever, args.limit)
    print(f"answerable   : {len(answerable)} questions (subject page in corpus)")
    print(f"unanswerable : {len(unanswerable)} questions (subject page absent)")
    print(f"verifier threshold {config.VERIFIER.entailment_threshold}  "
          f"abstain above {config.REPAIR.abstain_if_unsupported_fraction_above}\n")

    orch = Orchestrator(retriever=retriever, flags=PipelineFlags())
    started = time.time()
    rows = run_subset(orch, answerable, "answerable", not args.quiet)
    rows += run_subset(orch, unanswerable, "unanswerable", not args.quiet)
    elapsed = time.time() - started

    overall = summarize(rows, "all")
    per_subset = [
        summarize([r for r in rows if r.subset == s], s)
        for s in ["answerable", "unanswerable"]
    ]
    curve = risk_coverage([r for r in rows if r.subset == "answerable"])

    usage = orch.generator.run_totals()
    paths = orch.generator.path_totals()
    repairs = [r.repair for r in rows if r.repair]
    repair_attempted = sum(r.get("repair_attempted", 0) for r in repairs)
    repair_succeeded = sum(r.get("repair_succeeded", 0) for r in repairs)
    dropped = sum(r.get("dropped", 0) for r in repairs)

    # Compare against the Phase 1 baseline if it has actually been run.
    baseline_path = config.EVAL.results_dir / "baseline.json"
    baseline = None
    if baseline_path.exists():
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = {
            "n": data.get("n"),
            "accuracy": data.get("accuracy"),
            "coverage": data.get("coverage"),
            "answered_accuracy": data.get("answered_accuracy"),
        }

    print("\n" + "=" * 72)
    print("  MODULE E - ABSTENTION & RISK-COVERAGE   MEASURED")
    print("=" * 72)
    for stats in per_subset:
        print(f"\n  --- {stats['subset'].upper()} (n={stats['n']}) ---")
        print(f"  abstention rate           {stats['abstention_rate']:.4f}")
        print(f"  coverage                  {stats['coverage']:.4f}")
        print(f"  raw accuracy (all)        {stats['raw_accuracy']:.4f}")
        acc = stats["answered_subset_accuracy"]
        print(f"  accuracy on ANSWERED      "
              f"{'n/a (abstained on everything)' if acc is None else f'{acc:.4f}'}")
        if stats["abstain_reasons"]:
            print(f"  abstain reasons           {stats['abstain_reasons']}")

    print(f"\n  --- PHASE 1 BASELINE COMPARISON ---")
    if baseline is None:
        print("  NOT AVAILABLE - eval/results/baseline.json does not exist.")
        print("  Run: python -m eval.run_baseline --split dev --limit 150")
        print("  Without it the 'accuracy went up on answered questions' claim")
        print("  cannot be checked, so it is NOT made here.")
    else:
        answerable_stats = per_subset[0]
        delta = (answerable_stats["answered_subset_accuracy"] or 0) - (baseline["accuracy"] or 0)
        print(f"  baseline accuracy (variant A, n={baseline['n']})   {baseline['accuracy']:.4f}")
        print(f"  full pipeline, answered subset             "
              f"{answerable_stats['answered_subset_accuracy']:.4f}")
        print(f"  delta                                      {delta:+.4f}")
        print(f"  coverage traded away                       "
              f"{1 - answerable_stats['coverage']:.4f}")

    print(f"\n  --- RISK-COVERAGE (answerable subset) ---")
    print(f"  {'coverage':>10}{'n':>6}{'accuracy':>11}{'risk':>9}")
    for point in curve:
        print(f"  {point['coverage_of_all']:>10.4f}{point['n']:>6}"
              f"{point['accuracy']:>11.4f}{point['risk']:>9.4f}")

    print(f"\n  --- REPAIR ---")
    print(f"  repairs attempted         {repair_attempted}")
    print(f"  repairs succeeded         {repair_succeeded}  "
          f"({repair_succeeded / repair_attempted:.4f})" if repair_attempted else
          "  repairs succeeded         0")
    print(f"  sentences dropped         {dropped}")

    print(f"\n  --- MEASURED API USAGE ---")
    print(f"  api calls                 {usage['api_calls']} of {usage['calls']} "
          f"({usage['cache_hits_on_disk']} from disk cache)")
    print(f"  repair calls              {paths['repair_calls']}")
    print(f"  TOTAL cost                ${usage['total_cost_usd']:.4f}")
    print(f"  cost per question         ${usage['total_cost_usd'] / max(len(rows),1):.5f}")
    print(f"  elapsed                   {elapsed:.1f}s")
    print("=" * 72)

    payload = {
        "overall": overall,
        "per_subset": per_subset,
        "risk_coverage_answerable": curve,
        "phase1_baseline": baseline,
        "repair": {
            "attempted": repair_attempted,
            "succeeded": repair_succeeded,
            "success_rate": round(repair_succeeded / repair_attempted, 4) if repair_attempted else None,
            "dropped": dropped,
        },
        "config": {
            "entailment_threshold": config.VERIFIER.entailment_threshold,
            "abstain_if_unsupported_fraction_above": config.REPAIR.abstain_if_unsupported_fraction_above,
            "max_repair_attempts": config.REPAIR.max_repair_attempts,
            "min_surviving_chars": config.REPAIR.min_surviving_chars,
            "generator_model": config.GENERATOR.model,
        },
        "api_usage": usage,
        "generator_paths": paths,
        "elapsed_s": round(elapsed, 1),
        "rows": [asdict(r) for r in rows],
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
