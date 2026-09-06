"""The CI gate: compare fresh results against the committed baseline.

    python -m eval.check_regression
    python -m eval.check_regression --simulate citation_recall=0.60   # fire it

Exit code 0 when every measured metric is within threshold, 1 when any has
regressed. That exit code is the whole point - it is what turns a red CI check
on.

DESIGN NOTES
------------
**A missing measurement is not a pass.** If a metric's result file is absent the
gate reports it as MISSING and, for a metric the baseline declares required,
fails. A gate that silently goes green when the evidence disappears is worse
than no gate, because it looks like protection.

**A null baseline is skipped, loudly.** `faithfulness` has never been measured,
so it carries `value: null`. The gate prints SKIPPED and explains why rather
than inventing a threshold for it. The alternative - picking a plausible number
- would make the one metric this project most cares about unfalsifiable.

**Thresholds live in eval/baseline_metrics.json, with margins.** They are not
in this file, and they are deliberately below the measured values. A gate set
exactly at today's number fires on noise and gets switched off within a week.

**--simulate exists to fire the gate.** A gate that has never fired is not known
to work, and the workflow file existing is not evidence. `--simulate` overrides
a metric with a worse value so the failure path can be exercised on demand,
including in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import config

BASELINE_PATH = config.REPO_ROOT / "eval" / "baseline_metrics.json"

# Where each metric's fresh value is read from in eval/results/.
RESULT_ACCESSORS = {
    "retrieval_hit_rate": ("retrieval_check.json", ["retrieval_hit_rate"]),
    "evaluator_chunk_accuracy": ("evaluator_bench.json", ["per_chunk", "grading_accuracy"]),
    "evaluator_macro_f1": ("evaluator_bench.json", ["per_chunk", "macro_f1"]),
    "parseable_citation_rate": ("citation_bench.json", ["gate", "parseable_citation_rate"]),
    "invalid_id_rate": ("citation_bench.json", ["hard_failures", "invalid_id_rate"]),
    "citation_recall": ("citation_pr.json", ["citation_recall"]),
    "citation_precision": ("citation_pr.json", ["citation_precision"]),
    "hallucination_rate_auto": ("citation_pr.json", ["hallucination_rate_auto"]),
    "faithfulness": ("baseline.json", ["faithfulness"]),
}


@dataclass
class Check:
    name: str
    status: str            # PASS | FAIL | SKIPPED | MISSING
    baseline: float | None
    threshold: float | None
    observed: float | None
    direction: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status in {"FAIL", "MISSING"}

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.observed is None:
            return None
        return self.observed - self.baseline


def dig(payload: dict, path: list[str]):
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def read_observed(metric: str, results_dir: Path) -> float | None:
    entry = RESULT_ACCESSORS.get(metric)
    if entry is None:
        return None
    filename, path = entry
    file = results_dir / filename
    if not file.exists():
        return None
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = dig(payload, path)
    return float(value) if isinstance(value, (int, float)) else None


def evaluate(baseline: dict, results_dir: Path, simulate: dict[str, float],
             required: set[str]) -> list[Check]:
    checks: list[Check] = []
    for name, spec in baseline["metrics"].items():
        direction = spec.get("direction", "higher_is_better")
        expected = spec.get("value")
        threshold = spec.get("min") if direction == "higher_is_better" else spec.get("max")

        if expected is None or threshold is None:
            checks.append(
                Check(name, "SKIPPED", None, None, None, direction,
                      spec.get("note", "no baseline recorded"))
            )
            continue

        observed = simulate.get(name)
        simulated = observed is not None
        if observed is None:
            observed = read_observed(name, results_dir)

        if observed is None:
            status = "MISSING" if name in required else "SKIPPED"
            checks.append(
                Check(name, status, expected, threshold, None, direction,
                      f"no result found; run: {spec.get('command', '?')}")
            )
            continue

        if direction == "higher_is_better":
            ok = observed >= threshold
        else:
            ok = observed <= threshold

        checks.append(
            Check(
                name, "PASS" if ok else "FAIL", expected, threshold, observed, direction,
                "simulated" if simulated else "",
            )
        )
    return checks


def render(checks: list[Check], required: set[str]) -> str:
    lines = [
        "| Metric | Status | Baseline | Threshold | Observed | Delta |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        def fmt(v):
            return "-" if v is None else f"{v:.4f}"
        delta = "-" if c.delta is None else f"{c.delta:+.4f}"
        mark = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIPPED": "skipped",
                "MISSING": "**MISSING**"}[c.status]
        lines.append(
            f"| `{c.name}` | {mark} | {fmt(c.baseline)} | {fmt(c.threshold)} | "
            f"{fmt(c.observed)} | {delta} |"
        )
    notes = [f"- `{c.name}`: {c.detail}" for c in checks if c.detail]
    if notes:
        lines += ["", "Notes:", *notes]
    lines += [
        "",
        f"Required metrics: {', '.join(sorted(required)) or 'none'}.",
        "A missing result for a required metric fails the build - a gate that goes "
        "green when the evidence disappears is not protection.",
    ]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="CI regression gate")
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--results", default=str(config.EVAL.results_dir))
    parser.add_argument(
        "--required",
        default="retrieval_hit_rate,evaluator_chunk_accuracy,evaluator_macro_f1",
        help="comma-separated metrics whose absence fails the build "
             "(default: the free, deterministic ones)",
    )
    parser.add_argument(
        "--simulate",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="override an observed value, to prove the gate fires",
    )
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    simulate: dict[str, float] = {}
    for item in args.simulate:
        key, _, value = item.partition("=")
        simulate[key.strip()] = float(value)

    required = {m.strip() for m in args.required.split(",") if m.strip()}
    checks = evaluate(baseline, Path(args.results), simulate, required)

    print("=" * 72)
    print("  SCRAG REGRESSION GATE")
    print("=" * 72)
    print(f"  baseline recorded {baseline.get('recorded')}  seed {baseline.get('seed')}")
    if simulate:
        print(f"  SIMULATED OVERRIDES: {simulate}")
    print()
    print(f"  {'metric':<28}{'status':>9}{'thresh':>10}{'observed':>11}{'delta':>10}")
    print("  " + "-" * 66)
    for c in checks:
        fmt = lambda v: "     -" if v is None else f"{v:>6.4f}"
        delta = "     -" if c.delta is None else f"{c.delta:>+6.4f}"
        print(f"  {c.name:<28}{c.status:>9}{fmt(c.threshold):>10}"
              f"{fmt(c.observed):>11}{delta:>10}")

    failures = [c for c in checks if c.failed]
    skipped = [c for c in checks if c.status == "SKIPPED"]

    print()
    for c in skipped:
        print(f"  SKIPPED {c.name}: {c.detail}")

    # GitHub Actions job summary, when running in CI.
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("## SCRAG regression gate\n\n")
            fh.write(render(checks, required))
            fh.write("\n\n")
            fh.write(
                "**RESULT: FAILED**\n" if failures else "**RESULT: passed**\n"
            )

    print()
    if failures:
        print("  GATE FAILED:")
        for c in failures:
            if c.status == "MISSING":
                print(f"    - {c.name}: no measurement produced ({c.detail})")
            else:
                print(f"    - {c.name}: {c.observed:.4f} breached threshold "
                      f"{c.threshold:.4f} (baseline {c.baseline:.4f})")
        print("=" * 72)
        raise SystemExit(1)

    print("  GATE PASSED - no metric regressed past its threshold.")
    print("=" * 72)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
