"""Plot the ablation: faithfulness (and friends) across variants A to E.

    python -m eval.plot_ablation

Reads eval/results/ablation.csv and writes eval/results/ablation.png.

The prediction under test is that faithfulness climbs at every step. **A flat or
falling step is a real result and is plotted as measured** - the axis is not
rescaled, the point is not dropped, and the caption says so. A curve massaged
until it cooperates would be worth nothing.

Coverage is plotted on a second axis, because accuracy without coverage is not
interpretable across this row: variant E answers fewer questions than variant A.
"""

from __future__ import annotations

import argparse
import csv
import sys

import config


def read_rows(path):
    if not path.exists():
        raise SystemExit(
            f"no ablation table at {path}\n"
            "Run: python -m eval.run_ablation --all --split dev --limit 150"
        )
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Plot the A-E ablation")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")            # headless: no display on a server or in CI
    import matplotlib.pyplot as plt

    csv_path = (
        config.EVAL.results_dir / "ablation.csv" if args.csv is None else __import__("pathlib").Path(args.csv)
    )
    out_path = (
        config.EVAL.results_dir / "ablation.png" if args.out is None else __import__("pathlib").Path(args.out)
    )

    rows = read_rows(csv_path)
    labels = [r["variant"] for r in rows]
    names = [r["name"] for r in rows]

    series = {
        "Faithfulness": [as_float(r.get("faithfulness")) for r in rows],
        "Citation recall": [as_float(r.get("citation_recall")) for r in rows],
        "Citation precision": [as_float(r.get("citation_precision")) for r in rows],
        "Accuracy (answered)": [as_float(r.get("accuracy_answered")) for r in rows],
    }
    coverage = [as_float(r.get("coverage")) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(labels))

    plotted_any = False
    for name, values in series.items():
        # Plot only the points that exist. A missing measurement leaves a gap
        # rather than being interpolated into a smooth line.
        xs = [i for i, v in enumerate(values) if v is not None]
        ys = [v for v in values if v is not None]
        if not ys:
            continue
        plotted_any = True
        ax.plot(xs, ys, marker="o", linewidth=2, label=name)

    if not plotted_any:
        raise SystemExit(
            "every metric column is empty - nothing to plot.\n"
            "The ablation has not produced measurements yet."
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n{n}" for l, n in zip(labels, names)], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("SCRAG ablation: does each module earn its place?", fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")

    ax2 = ax.twinx()
    cov_x = [i for i, v in enumerate(coverage) if v is not None]
    cov_y = [v for v in coverage if v is not None]
    if cov_y:
        ax2.bar(cov_x, cov_y, alpha=0.15, color="grey", width=0.5, label="coverage")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("coverage (bars)")

    handles, labs = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labs + l2, loc="lower left", fontsize=8)

    fig.text(
        0.01, 0.01,
        "Coverage falls by design: variant E abstains rather than answering unsupported questions. "
        "Flat or falling steps are plotted as measured.",
        fontsize=7, color="dimgrey",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
