"""Phase 6 harness tests: variant definitions, table writing, and empty cells.

The property these protect is that the ablation measures what it claims to:
variants differ only by flags over one code path, and a missing measurement
stays visibly missing instead of being rendered as a zero.

No API key and no model weights needed.
"""

from __future__ import annotations

import csv

import pytest

import config
from core.orchestrator import Orchestrator, PipelineFlags
from eval import variants as variants_mod
from eval.metrics import (
    citation_precision_recall,
    hallucination_rate_automatic,
    hallucination_rate_labeled,
)
from eval.run_ablation import PAPER_BASELINES, VariantResult, fmt, write_csv, write_markdown
from core.types import Chunk, CitedSentence, NLILabel, SentenceVerdict


# --------------------------------------------------------------------------
# Variants are flags over one code path
# --------------------------------------------------------------------------


def test_all_five_variants_exist_in_order():
    assert variants_mod.ORDER == ["A", "B", "C", "D", "E"]
    assert [v.key for v in variants_mod.all_variants()] == ["A", "B", "C", "D", "E"]


def test_every_variant_has_a_distinct_flag_combination():
    seen = set()
    for variant in variants_mod.all_variants():
        combo = (
            variant.flags.use_evaluator,
            variant.flags.force_citations,
            variant.flags.verify_citations,
            variant.flags.repair_and_abstain,
        )
        assert combo not in seen, f"{variant.key} duplicates another variant"
        seen.add(combo)


def test_variants_are_strictly_cumulative():
    """Each variant turns on the next module and keeps the previous ones on."""
    order = ["use_evaluator", "force_citations", "verify_citations", "repair_and_abstain"]
    previous = [False, False, False, False]
    for variant in variants_mod.all_variants():
        current = [getattr(variant.flags, name) for name in order]
        for was_on, now_on in zip(previous, current):
            assert not (was_on and not now_on), f"{variant.key} turned a module back off"
        previous = current


def test_variant_a_is_the_phase_1_baseline():
    a = variants_mod.get("A")
    assert not a.flags.use_evaluator
    assert not a.flags.force_citations
    assert not a.flags.verify_citations
    assert not a.flags.repair_and_abstain


def test_variant_e_is_the_full_pipeline():
    e = variants_mod.get("E")
    assert all(
        [e.flags.use_evaluator, e.flags.force_citations,
         e.flags.verify_citations, e.flags.repair_and_abstain]
    )


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        variants_mod.get("Z")


def test_orchestrator_accepts_every_variant_without_forking():
    """One Orchestrator class, five flag sets - no variant-specific subclass."""
    for variant in variants_mod.all_variants():
        orch = Orchestrator(flags=variant.flags)
        assert isinstance(orch, Orchestrator)
        assert orch.flags is variant.flags


def test_variant_a_and_b_use_the_plain_generator_c_onward_the_citation_one():
    from core.generator import CitationGenerator, PlainGenerator

    for key in ["A", "B"]:
        orch = Orchestrator(flags=variants_mod.get(key).flags)
        assert type(orch.generator) is PlainGenerator
    for key in ["C", "D", "E"]:
        orch = Orchestrator(flags=variants_mod.get(key).flags)
        assert isinstance(orch.generator, CitationGenerator)


# --------------------------------------------------------------------------
# Missing measurements stay missing
# --------------------------------------------------------------------------


def test_fmt_renders_none_as_empty_not_zero():
    """A blank cell means 'not measured'. A 0.0000 would read as a real result."""
    assert fmt(None) == ""
    assert fmt(0.0) == "0.0000"


def test_unlabeled_hallucination_set_returns_none_not_zero():
    result = hallucination_rate_labeled([])
    assert result["hallucination_rate_labeled"] is None
    assert result["n_labeled"] == 0


def test_labeled_hallucination_rate_counts_only_labeled_rows():
    rows = [
        {"label": "hallucinated"},
        {"label": "supported"},
        {"label": "supported"},
        {"label": "skip"},          # not a valid label, must be ignored
    ]
    result = hallucination_rate_labeled(rows)
    assert result["n_labeled"] == 3
    # Metrics round to 4 dp, so compare with matching tolerance.
    assert result["hallucination_rate_labeled"] == pytest.approx(1 / 3, abs=1e-4)


def test_alce_baseline_cells_are_declared_unavailable_not_guessed():
    for key in ["citation_precision", "citation_recall"]:
        assert PAPER_BASELINES[key]["value"] is None
        assert "NOT AVAILABLE" in PAPER_BASELINES[key]["note"]


def test_crag_baseline_carries_a_paper_and_a_table():
    base = PAPER_BASELINES["evaluator_action_accuracy"]
    assert base["value"] == 0.843
    assert base["table"] == "Table 4"
    assert base["arxiv"] == "2401.15884v3"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

CONTEXT = [Chunk(chunk_id="D::0", text="premise text", doc_id="D")]


def verdict(supported: bool, per_citation: dict, uncited: bool = False, cites=None):
    return SentenceVerdict(
        sentence=CitedSentence("A claim.", cites if cites is not None else ["D::0"]),
        supported=supported,
        entailment_score=0.9 if supported else 0.01,
        per_citation=per_citation,
        nli_label=NLILabel.ENTAILMENT if supported else NLILabel.NEUTRAL,
        uncited=uncited,
    )


def test_citation_recall_counts_supported_sentences():
    verdicts = [verdict(True, {"D::0": 0.9}), verdict(False, {"D::0": 0.01})]
    scores = citation_precision_recall([([], verdicts, CONTEXT)], 0.2)
    assert scores.recall == pytest.approx(0.5)


def test_citation_precision_penalises_padding():
    """A second citation that does not support the sentence costs precision."""
    verdicts = [verdict(True, {"D::0": 0.9, "D::1": 0.01})]
    scores = citation_precision_recall([([], verdicts, CONTEXT)], 0.2)
    assert scores.recall == pytest.approx(1.0)
    assert scores.precision == pytest.approx(0.5)


def test_abstentions_are_excluded_from_citation_scores():
    """A refusal makes no claim - excluded, not scored zero."""
    verdicts = [verdict(True, {"D::0": 0.9}), verdict(False, {}, uncited=True, cites=[])]
    scores = citation_precision_recall([([], verdicts, CONTEXT)], 0.2)
    assert scores.sentences_scored == 1
    assert scores.recall == pytest.approx(1.0)


def test_hallucination_rate_separates_contradiction_from_unsupported():
    verdicts = [
        verdict(True, {"D::0": 0.9}),
        verdict(False, {"D::0": 0.01}),
        SentenceVerdict(
            sentence=CitedSentence("Refuted.", ["D::0"]),
            supported=False,
            entailment_score=0.0,
            nli_label=NLILabel.CONTRADICTION,
            per_citation={"D::0": 0.0},
        ),
    ]
    auto = hallucination_rate_automatic(verdicts)
    assert auto["hallucination_rate_auto"] == pytest.approx(2 / 3, abs=1e-4)
    assert auto["contradiction_rate_auto"] == pytest.approx(1 / 3, abs=1e-4)


def test_empty_verdicts_return_none_not_zero():
    auto = hallucination_rate_automatic([])
    assert auto["hallucination_rate_auto"] is None


# --------------------------------------------------------------------------
# Table output
# --------------------------------------------------------------------------


def sample_results() -> list[VariantResult]:
    return [
        VariantResult(
            variant="A", name="Vanilla RAG", adds="Module A only",
            n=10, answered=10, coverage=1.0, accuracy_all=0.5, accuracy_answered=0.5,
        ),
        VariantResult(
            variant="E", name="+ repair and abstention", adds="Module E",
            n=10, answered=6, coverage=0.6, accuracy_all=0.4, accuracy_answered=0.6667,
            citation_recall=0.8, citation_precision=0.7, abstention_rate=0.4,
        ),
    ]


def test_csv_has_a_row_per_variant_and_blank_cells_for_missing_metrics(tmp_path):
    path = tmp_path / "ablation.csv"
    write_csv(sample_results(), path)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["variant"] for r in rows] == ["A", "E"]
    # Faithfulness was never measured on this fixture, so it must be blank.
    assert rows[0]["faithfulness"] == ""
    assert rows[1]["citation_recall"] == "0.8"


def test_markdown_reports_coverage_and_warns_against_comparing_accuracy(tmp_path):
    path = tmp_path / "ablation.md"
    write_markdown(sample_results(), path, "dev", {"total_cost_usd": 1.23})
    text = path.read_text(encoding="utf-8")
    assert "Coverage" in text
    assert "not comparable across the row" in text
    assert "| A |" in text and "| E |" in text


def test_markdown_states_faithfulness_is_not_ragas(tmp_path):
    path = tmp_path / "ablation.md"
    write_markdown(sample_results(), path, "dev", {})
    text = path.read_text(encoding="utf-8")
    assert "Never report it as RAGAS" in text


def test_markdown_says_variant_d_matches_variant_c_by_construction(tmp_path):
    path = tmp_path / "ablation.md"
    write_markdown(sample_results(), path, "dev", {})
    assert "same text as variant C" in path.read_text(encoding="utf-8")


def test_markdown_leaves_the_alce_cells_empty(tmp_path):
    path = tmp_path / "ablation.md"
    write_markdown(sample_results(), path, "dev", {})
    text = path.read_text(encoding="utf-8")
    assert "*not recorded*" in text
    assert "Empty cells are deliberate" in text
