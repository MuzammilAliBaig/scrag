"""Tests for the CI regression gate.

The gate's whole job is its exit code, and the failure modes that matter are the
quiet ones: passing when evidence is missing, or inventing a threshold for a
metric nobody has measured. Both are tested here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.check_regression import BASELINE_PATH, evaluate, read_observed, render

BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
REQUIRED = {"retrieval_hit_rate", "evaluator_chunk_accuracy", "evaluator_macro_f1"}


def results(tmp_path: Path, **files) -> Path:
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# Thresholds sit below the measurements, with a margin
# --------------------------------------------------------------------------


def test_every_threshold_leaves_a_margin_below_the_measurement():
    """A gate set exactly at today's number fires on noise and gets disabled."""
    for name, spec in BASELINE["metrics"].items():
        if spec.get("value") is None:
            continue
        if spec["direction"] == "higher_is_better":
            assert spec["min"] <= spec["value"], f"{name} threshold exceeds its baseline"
        else:
            assert spec["max"] >= spec["value"], f"{name} threshold is below its baseline"


def test_parseable_citation_rate_is_a_contract_with_no_margin():
    """One uncited sentence is a defect, not noise."""
    spec = BASELINE["metrics"]["parseable_citation_rate"]
    assert spec["min"] == 1.0
    assert spec["margin"] == 0.0


def test_every_metric_records_where_its_number_came_from():
    for name, spec in BASELINE["metrics"].items():
        assert spec.get("source"), f"{name} has no source file recorded"
        assert spec.get("command"), f"{name} has no reproduction command"


# --------------------------------------------------------------------------
# Pass / fail behaviour
# --------------------------------------------------------------------------


def test_gate_passes_when_metrics_meet_thresholds(tmp_path):
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": 0.87},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
        },
    )
    checks = evaluate(BASELINE, directory, {}, REQUIRED)
    assert not [c for c in checks if c.failed]


def test_gate_fails_when_a_higher_is_better_metric_drops(tmp_path):
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": 0.50},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
        },
    )
    failures = [c for c in evaluate(BASELINE, directory, {}, REQUIRED) if c.failed]
    assert [c.name for c in failures] == ["retrieval_hit_rate"]


def test_gate_fails_when_a_lower_is_better_metric_rises(tmp_path):
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": 0.87},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
            "citation_pr.json": {
                "citation_recall": 0.84,
                "citation_precision": 0.74,
                "hallucination_rate_auto": 0.90,
            },
        },
    )
    failures = {c.name for c in evaluate(BASELINE, directory, {}, REQUIRED) if c.failed}
    assert "hallucination_rate_auto" in failures


def test_a_value_exactly_on_the_threshold_passes(tmp_path):
    spec = BASELINE["metrics"]["retrieval_hit_rate"]
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": spec["min"]},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
        },
    )
    checks = {c.name: c for c in evaluate(BASELINE, directory, {}, REQUIRED)}
    assert checks["retrieval_hit_rate"].status == "PASS"


# --------------------------------------------------------------------------
# The quiet failure modes
# --------------------------------------------------------------------------


def test_missing_required_metric_fails_rather_than_silently_passing(tmp_path):
    """A gate that goes green when the evidence disappears is not protection."""
    checks = {c.name: c for c in evaluate(BASELINE, tmp_path, {}, REQUIRED)}
    for name in REQUIRED:
        assert checks[name].status == "MISSING"
        assert checks[name].failed


def test_missing_optional_metric_is_skipped_not_failed(tmp_path):
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": 0.87},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
        },
    )
    checks = {c.name: c for c in evaluate(BASELINE, directory, {}, REQUIRED)}
    assert checks["citation_recall"].status == "SKIPPED"
    assert not checks["citation_recall"].failed


def test_unmeasured_faithfulness_is_skipped_never_invented(tmp_path):
    """The metric this project cares about most has no number yet.

    Giving it a plausible threshold would make the gate protecting it
    unfalsifiable.
    """
    spec = BASELINE["metrics"]["faithfulness"]
    assert spec["value"] is None and spec["min"] is None
    checks = {c.name: c for c in evaluate(BASELINE, tmp_path, {}, REQUIRED)}
    assert checks["faithfulness"].status == "SKIPPED"
    assert not checks["faithfulness"].failed
    assert "NOT YET MEASURED" in checks["faithfulness"].detail


def test_corrupt_result_file_is_treated_as_missing(tmp_path):
    (tmp_path / "retrieval_check.json").write_text("{not json", encoding="utf-8")
    assert read_observed("retrieval_hit_rate", tmp_path) is None


# --------------------------------------------------------------------------
# Simulation and reporting
# --------------------------------------------------------------------------


def test_simulate_overrides_an_observed_value_so_the_gate_can_be_fired(tmp_path):
    directory = results(
        tmp_path,
        **{
            "retrieval_check.json": {"retrieval_hit_rate": 0.87},
            "evaluator_bench.json": {"per_chunk": {"grading_accuracy": 0.92, "macro_f1": 0.80}},
        },
    )
    checks = {
        c.name: c
        for c in evaluate(BASELINE, directory, {"retrieval_hit_rate": 0.10}, REQUIRED)
    }
    assert checks["retrieval_hit_rate"].status == "FAIL"
    assert checks["retrieval_hit_rate"].detail == "simulated"


def test_summary_table_marks_failures_and_explains_missing_policy(tmp_path):
    checks = evaluate(BASELINE, tmp_path, {"retrieval_hit_rate": 0.1}, REQUIRED)
    table = render(checks, REQUIRED)
    assert "**FAIL**" in table
    assert "**MISSING**" in table
    assert "not protection" in table
