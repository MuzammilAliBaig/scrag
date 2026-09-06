"""Export report figures at print resolution to docs/figures/.

    python -m docs.make_figures

Two figures are produced from measured data:

  measured_metrics.png  - per-module headline metrics
  verifier_calibration.png - the threshold sweep that set Module D's operating point

The A-E ablation figure is NOT produced here. It requires eval/results/ablation.csv,
which does not exist because the sweep needs generation across five variants and the
API credit balance was exhausted. `eval/plot_ablation.py` builds it the moment that
file appears; it refuses to draw an empty chart in the meantime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import config

FIGURES = config.REPO_ROOT / "docs" / "figures"
RESULTS = config.EVAL.results_dir

# Muted, print-safe, and distinguishable in greyscale.
INK = "#1f2430"
MUTED = "#7c8493"
GOOD = "#3f6f4f"
WARN = "#b07d3a"
BAD = "#9c4a4a"
GRID = "#d8dce3"


def load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def figure_metrics(plt) -> Path | None:
    citation_pr = load("citation_pr.json")
    evaluator = load("evaluator_bench.json")
    retrieval = load("retrieval_check.json")
    citation = load("citation_bench.json")
    abstention = load("abstention_bench_retrieval_only.json")

    bars = [
        ("A  retrieval hit rate @5", retrieval.get("retrieval_hit_rate"), GOOD),
        ("B  per-chunk accuracy", (evaluator.get("per_chunk") or {}).get("grading_accuracy"), GOOD),
        ("B  macro F1", (evaluator.get("per_chunk") or {}).get("macro_f1"), GOOD),
        ("B  action accuracy", (evaluator.get("query_level_action") or {}).get("accuracy"), BAD),
        ("C  parseable citations", (citation.get("gate") or {}).get("parseable_citation_rate"), GOOD),
        ("D  citation recall", citation_pr.get("citation_recall"), GOOD),
        ("D  citation precision", citation_pr.get("citation_precision"), WARN),
        ("D  hallucination (auto)", citation_pr.get("hallucination_rate_auto"), WARN),
        ("E  refusal, unanswerable", ((abstention.get("subsets") or {}).get("unanswerable") or {})
            .get("pre_generation_abstention_rate"), GOOD),
    ]
    bars = [(label, value, colour) for label, value, colour in bars if value is not None]
    if not bars:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colours = [b[2] for b in bars]
    positions = range(len(bars))

    ax.barh(list(positions), values, color=colours, height=0.62, zorder=3)
    for pos, value in zip(positions, values):
        ax.text(value + 0.012, pos, f"{value:.3f}", va="center", fontsize=9, color=INK)

    # The majority-class baseline Module B fails to beat - the single most
    # important annotation on this chart.
    majority = (evaluator.get("query_level_action") or {}).get("majority_class_baseline")
    if majority is not None:
        ax.axvline(majority, color=BAD, linestyle="--", linewidth=1.2, zorder=4)
        # Annotated to the LEFT of the line and above the axis: to the right it
        # collides with the 1.0 x-tick label.
        ax.annotate(
            f"majority-class baseline {majority:.3f}",
            xy=(majority - 0.015, -0.72),
            fontsize=7.5, color=BAD, ha="right", va="center",
            annotation_clip=False,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score", fontsize=9)
    ax.set_title("SCRAG - measured metrics by module", fontsize=12, color=INK, pad=12)
    ax.grid(axis="x", color=GRID, linestyle=":", zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.text(0.01, 0.015,
             "Hallucination rate is verifier-derived and an upper bound (docs/error-analysis.md). "
             "Faithfulness and the A-E ablation are absent: not measured.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGURES / "measured_metrics.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def figure_calibration(plt) -> Path | None:
    calibration = load("verifier_calibration.json")
    sweep = calibration.get("sweep")
    if not sweep:
        return None

    thresholds = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds, [r["precision"] for r in sweep], marker="o", ms=3.5,
            color=GOOD, label="precision", zorder=3)
    ax.plot(thresholds, [r["recall"] for r in sweep], marker="s", ms=3.5,
            color=WARN, label="recall", zorder=3)
    ax.plot(thresholds, [r["f1"] for r in sweep], marker="^", ms=3.5,
            color=INK, label="F1", linewidth=2, zorder=4)

    chosen = calibration.get("recommended", {}).get("threshold")
    if chosen is not None:
        ax.axvline(chosen, color=BAD, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(chosen + 0.012, 0.05, f"operating point {chosen}",
                fontsize=8, color=BAD, rotation=90, va="bottom")

    ax.set_xlabel("entailment threshold", fontsize=9)
    ax.set_ylabel("score", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("Module D threshold calibration (600-pair labeled slice)",
                 fontsize=12, color=INK, pad=12)
    ax.grid(color=GRID, linestyle=":", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.text(0.01, 0.015,
             "Positives are lexically verified, not hand-labeled, so recall is a lower bound. "
             "Negatives (passages from other questions) are near-certain.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = FIGURES / "verifier_calibration.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })

    FIGURES.mkdir(parents=True, exist_ok=True)
    written = [figure_metrics(plt), figure_calibration(plt)]
    for path in written:
        if path:
            print(f"wrote {path} (+ .pdf) at 300 dpi")

    ablation = RESULTS / "ablation.csv"
    if ablation.exists():
        print(f"\n{ablation} exists - run `python -m eval.plot_ablation` for the ablation figure")
    else:
        print("\nNOT PRODUCED: the A-E ablation figure.")
        print("  eval/results/ablation.csv does not exist - the sweep needs generation")
        print("  across five variants. Run:")
        print("    python -m eval.run_ablation --all --split dev --limit 150")
        print("    python -m eval.plot_ablation")


if __name__ == "__main__":
    main()
