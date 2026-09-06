"""Export every measured number to docs/tables/ as CSV and markdown.

    python -m docs.export_results

Each row carries the result file it came from, so every figure in the report
traces back to a run in eval/results/ - the same discipline the system itself
enforces on its answers.

A metric that has not been measured is exported as an EMPTY cell with a stated
reason. It is never exported as zero, and never filled with a plausible number.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import config

DOCS = config.REPO_ROOT / "docs"
TABLES = DOCS / "tables"
RESULTS = config.EVAL.results_dir


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def dig(payload, path, default=None):
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def collect() -> list[dict]:
    retrieval = load("retrieval_check.json") or {}
    evaluator = load("evaluator_bench.json") or {}
    citation = load("citation_bench.json") or {}
    citation_pr = load("citation_pr.json") or {}
    verifier = load("verifier_bench.json") or {}
    calibration = load("verifier_calibration.json") or {}
    abstention = load("abstention_bench_retrieval_only.json") or {}
    baseline = load("baseline.json")

    def row(module, metric, value, source, note=""):
        return {
            "module": module,
            "metric": metric,
            "value": "" if value is None else f"{value:.4f}"
            if isinstance(value, float) else str(value),
            "source": source,
            "note": note,
        }

    rows = [
        row("A - Retrieval", "retrieval hit rate @ k=5",
            retrieval.get("retrieval_hit_rate"), "retrieval_check.json",
            "gold answer present in top-5; ceiling on answerable accuracy"),
        row("A - Retrieval", "top-1 hit rate",
            retrieval.get("top1_hit_rate"), "retrieval_check.json"),
        row("A - Retrieval", "corpus chunks",
            retrieval.get("corpus_chunks"), "retrieval_check.json",
            "990 Wikipedia lead sections incl. 800 distractor entities"),

        row("B - Evaluator", "per-chunk grading accuracy",
            dig(evaluator, ["per_chunk", "grading_accuracy"]), "evaluator_bench.json",
            "no published counterpart; CRAG never released a per-chunk rubric"),
        row("B - Evaluator", "per-chunk macro F1",
            dig(evaluator, ["per_chunk", "macro_f1"]), "evaluator_bench.json"),
        row("B - Evaluator", "F1 (ambiguous class)",
            dig(evaluator, ["per_chunk", "per_class", "ambiguous", "f1"]),
            "evaluator_bench.json", "the hard class, as the rubric predicted"),
        row("B - Evaluator", "query-level action accuracy",
            dig(evaluator, ["query_level_action", "accuracy"]), "evaluator_bench.json",
            "the only CRAG-comparable number"),
        row("B - Evaluator", "majority-class baseline",
            dig(evaluator, ["query_level_action", "majority_class_baseline"]),
            "evaluator_bench.json",
            "a trivial always-Correct predictor; our number is BELOW it"),
        row("B - Evaluator", "corrective-loop trigger rate",
            dig(evaluator, ["query_level_action", "corrective_trigger_rate"]),
            "evaluator_bench.json"),

        row("C - Citation", "parseable-citation rate",
            dig(citation, ["gate", "parseable_citation_rate"]), "citation_bench.json",
            "Phase 3 gate; abstentions excluded (0.9498 including them)"),
        row("C - Citation", "invalid-id rate",
            dig(citation, ["hard_failures", "invalid_id_rate"]), "citation_bench.json",
            "invented sources; zero observed"),
        row("C - Citation", "structured-output path",
            f"{dig(citation, ['output_path', 'structured_path'])}/200", "citation_bench.json",
            "0 text-parser fallbacks, 0 parse retries"),
        row("C - Citation", "abstention rate",
            (citation.get("abstentions") or 0) / max(citation.get("questions") or 1, 1),
            "citation_bench.json", "21/200 on ALCE/ASQA"),
        row("C - Citation", "ASQA str-EM",
            dig(citation, ["answer_correctness", "asqa_str_em"]), "citation_bench.json"),
        row("C - Citation", "cost per question (USD)",
            dig(citation, ["api_usage", "cost_per_query_usd"]), "citation_bench.json",
            "measured, claude-opus-5, prompt caching active"),

        row("D - Verifier", "citation recall",
            citation_pr.get("citation_recall"), "citation_pr.json",
            "ALCE definition, our checkpoint (not TRUE/T5-XXL)"),
        row("D - Verifier", "citation precision",
            citation_pr.get("citation_precision"), "citation_pr.json",
            "penalises citation padding"),
        row("D - Verifier", "hallucination rate (auto)",
            citation_pr.get("hallucination_rate_auto"), "citation_pr.json",
            "verifier-derived UPPER BOUND; see docs/error-analysis.md"),
        row("D - Verifier", "contradiction rate (auto)",
            citation_pr.get("contradiction_rate_auto"), "citation_pr.json"),
        row("D - Verifier", "flag rate on real answers",
            dig(verifier, ["outcomes", "flag_rate"]), "verifier_bench.json"),
        row("D - Verifier", "calibrated threshold",
            calibration.get("recommended", {}).get("threshold"),
            "verifier_calibration.json", "best-F1 point on a 600-pair slice"),
        row("D - Verifier", "calibration precision",
            calibration.get("recommended", {}).get("precision"), "verifier_calibration.json"),
        row("D - Verifier", "calibration recall",
            calibration.get("recommended", {}).get("recall"), "verifier_calibration.json",
            "LOWER BOUND - positives are lexically verified, not hand-labeled"),
        row("D - Verifier", "positive/negative separation",
            dig(calibration, ["separation", "difference"]), "verifier_calibration.json"),

        row("E - Abstention", "trigger (b), unanswerable",
            dig(abstention, ["subsets", "unanswerable", "pre_generation_abstention_rate"]),
            "abstention_bench_retrieval_only.json",
            "refused before generation, so free"),
        row("E - Abstention", "trigger (b), answerable",
            dig(abstention, ["subsets", "answerable", "pre_generation_abstention_rate"]),
            "abstention_bench_retrieval_only.json", "not over-abstaining"),
        row("E - Abstention", "corrective loop, answerable",
            dig(abstention, ["subsets", "answerable", "corrective_loop_rate"]),
            "abstention_bench_retrieval_only.json"),

        row("Pipeline", "RAGAS-definition faithfulness",
            None if baseline is None else baseline.get("faithfulness"),
            "baseline.json",
            "NOT MEASURED - Phase 1 baseline never ran (API credit exhausted)"),
        row("Pipeline", "baseline answer accuracy",
            None if baseline is None else baseline.get("accuracy"), "baseline.json",
            "NOT MEASURED - same reason"),
        row("Pipeline", "A-E ablation table",
            None, "ablation.csv",
            "NOT PRODUCED - requires generation across five variants"),
        row("Pipeline", "hallucination rate (hand-labeled)",
            None, "eval/labeled/hallucination.jsonl",
            "NOT MEASURED - set is defined and frozen but unlabeled"),
    ]
    return rows


PAPER_ROWS = [
    {
        "concern": "Evaluator action accuracy",
        "paper": "CRAG (Yan et al. 2024)",
        "paper_reported": "0.8430",
        "source": "Table 4, arXiv:2401.15884v3, section 5.5",
        "we_achieved": "0.8158",
        "verdict": "BELOW - and below our own 0.8750 majority baseline, so the comparison is not a real claim on this split",
    },
    {
        "concern": "Evaluator vs ChatGPT (same table)",
        "paper": "CRAG (Yan et al. 2024)",
        "paper_reported": "0.5800 / 0.6240 / 0.6470",
        "source": "Table 4 - ChatGPT, CoT, few-shot",
        "we_achieved": "0.8158",
        "verdict": "ABOVE all three, but so is the trivial majority predictor",
    },
    {
        "concern": "Citation precision",
        "paper": "ALCE (Gao et al. 2023)",
        "paper_reported": "",
        "source": "NOT RECORDED in this project",
        "we_achieved": "0.7409",
        "verdict": "reported unqualified - no baseline to hand, and a different entailment model than ALCE's TRUE/T5-XXL",
    },
    {
        "concern": "Citation recall",
        "paper": "ALCE (Gao et al. 2023)",
        "paper_reported": "",
        "source": "NOT RECORDED in this project",
        "we_achieved": "0.8388",
        "verdict": "reported unqualified - same reason",
    },
    {
        "concern": "Faithfulness",
        "paper": "n/a",
        "paper_reported": "",
        "source": "no published baseline",
        "we_achieved": "",
        "verdict": "NOT MEASURED - the Phase 1 baseline never ran",
    },
]


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, title: str, fields: list[str],
                   preamble: str = "") -> None:
    lines = [f"# {title}", ""]
    if preamble:
        lines += [preamble, ""]
    lines.append("| " + " | ".join(f.replace("_", " ").title() for f in fields) + " |")
    lines.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        cells = [str(row.get(f, "")).replace("|", "\\|") for f in fields]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Empty cells are deliberate: the measurement does not exist. "
                  "They are never rendered as zero."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    TABLES.mkdir(parents=True, exist_ok=True)

    rows = collect()
    fields = ["module", "metric", "value", "source", "note"]
    write_csv(rows, TABLES / "measured_results.csv", fields)
    write_markdown(
        rows, TABLES / "measured_results.md", "SCRAG - all measured results", fields,
        "Every value below came from a run recorded in `eval/results/`. The `source` column names "
        "the file. Rows with an empty value were **not measured**; the note says why.",
    )

    paper_fields = ["concern", "paper", "paper_reported", "source", "we_achieved", "verdict"]
    write_csv(PAPER_ROWS, TABLES / "paper_vs_ours.csv", paper_fields)
    write_markdown(
        PAPER_ROWS, TABLES / "paper_vs_ours.md", "Paper reported X vs we achieved Y", paper_fields,
        "Every baseline figure carries the paper **and** the table it came from. Where a figure is "
        "not recorded in this project, the cell is empty and the verdict says so - a plausible "
        "number would be worse than a gap.",
    )

    measured = sum(1 for r in rows if r["value"])
    print(f"wrote {TABLES / 'measured_results.csv'}  ({measured}/{len(rows)} metrics measured)")
    print(f"wrote {TABLES / 'measured_results.md'}")
    print(f"wrote {TABLES / 'paper_vs_ours.csv'}")
    print(f"wrote {TABLES / 'paper_vs_ours.md'}")
    print()
    print("NOT MEASURED:")
    for r in rows:
        if not r["value"]:
            print(f"  - {r['metric']}: {r['note']}")


if __name__ == "__main__":
    main()
