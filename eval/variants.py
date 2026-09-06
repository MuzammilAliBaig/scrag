"""The five ablation variants, as config flags over ONE code path.

Five forked pipelines drift, and the moment they drift the ablation stops
measuring what it claims to measure. So every variant here is the same
`Orchestrator` with different `PipelineFlags`; there is no variant-specific
branch anywhere in `core/`.

| Variant | Pipeline | Adds |
|---|---|---|
| A | Vanilla RAG | Module A only - the Phase 1 baseline |
| B | + retrieval evaluator | Module B grading and corrective retrieval |
| C | + citation forcing | Module C, one citation per sentence |
| D | + NLI verification | Module D flags unsupported sentences |
| E | + repair and abstention | Module E repairs, drops, or abstains |

A note on what D changes and what it does not. Module D **flags** but does not
alter the answer - only Module E acts on the flags. So variant D emits the same
text as variant C, and its faithfulness and accuracy will be identical by
construction. What D adds is measurement: the flag rate, and citation
precision/recall computed by entailment rather than by string overlap. If the
report shows D and C with identical accuracy, that is correct and expected, not
a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.orchestrator import PipelineFlags


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    flags: PipelineFlags
    adds: str
    # Whether this variant needs generation, and so API credit.
    needs_api: bool = True

    @property
    def label(self) -> str:
        return f"{self.key} - {self.name}"


VARIANTS: dict[str, Variant] = {
    "A": Variant(
        key="A",
        name="Vanilla RAG",
        flags=PipelineFlags(
            use_evaluator=False,
            force_citations=False,
            verify_citations=False,
            repair_and_abstain=False,
        ),
        adds="Module A only (the Phase 1 baseline)",
    ),
    "B": Variant(
        key="B",
        name="+ retrieval evaluator",
        flags=PipelineFlags(
            use_evaluator=True,
            force_citations=False,
            verify_citations=False,
            repair_and_abstain=False,
        ),
        adds="Module B grading + corrective retrieval",
    ),
    "C": Variant(
        key="C",
        name="+ citation forcing",
        flags=PipelineFlags(
            use_evaluator=True,
            force_citations=True,
            verify_citations=False,
            repair_and_abstain=False,
        ),
        adds="Module C, one citation per sentence",
    ),
    "D": Variant(
        key="D",
        name="+ NLI verification",
        flags=PipelineFlags(
            use_evaluator=True,
            force_citations=True,
            verify_citations=True,
            repair_and_abstain=False,
        ),
        adds="Module D flags unsupported sentences (measures, does not alter output)",
    ),
    "E": Variant(
        key="E",
        name="+ repair and abstention",
        flags=PipelineFlags(
            use_evaluator=True,
            force_citations=True,
            verify_citations=True,
            repair_and_abstain=True,
        ),
        adds="Module E repairs, drops, or abstains",
    ),
}

ORDER = ["A", "B", "C", "D", "E"]


def get(key: str) -> Variant:
    key = key.upper()
    if key not in VARIANTS:
        raise KeyError(f"unknown variant {key!r}; expected one of {ORDER}")
    return VARIANTS[key]


def all_variants() -> list[Variant]:
    return [VARIANTS[k] for k in ORDER]
