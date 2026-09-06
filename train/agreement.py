"""Inter-annotator agreement on the double-labeled slice.

    python -m train.agreement --a alice --b bob
    python -m train.agreement --a alice --vs-weak

Reports raw agreement and Cohen's kappa, overall and per class. Two comparisons
matter for this project:

* **human vs human** - how reliable train/LABELING.md is as a rubric. If two
  people applying it disagree badly, the ground truth Module B is scored
  against is soft, and the headline number inherits that softness.
* **human vs weak rule** - how far train/weak_labels.py diverges from the
  rubric it approximates. The training set is weakly supervised, and this is
  the measurement that says how much that costs.

`ambiguous` is expected to be the weakest class in both comparisons. Report it
rather than hiding it in a macro average - LABELING.md section 6 says so up
front.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import config

LABELS_DIR = config.DATA_DIR / "human_labels"
CLASSES = ["correct", "ambiguous", "wrong"]


def load_labels(annotator: str) -> dict[str, dict]:
    path = LABELS_DIR / f"{annotator}.jsonl"
    if not path.exists():
        raise SystemExit(f"no labels for {annotator!r} at {path} - run train/label.py first")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[f"{row['qid']}|{row['chunk_id']}"] = row
    return out


def cohens_kappa(a: list[str], b: list[str], classes: list[str] = CLASSES) -> float:
    """Cohen's kappa: agreement corrected for what chance alone would give.

    Returns 0.0 when the annotators use a single class between them, where
    kappa is undefined rather than perfect.
    """
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in classes)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def confusion(a: list[str], b: list[str]) -> dict[str, dict[str, int]]:
    matrix = {x: {y: 0 for y in CLASSES} for x in CLASSES}
    for x, y in zip(a, b):
        matrix[x][y] += 1
    return matrix


def per_class_agreement(a: list[str], b: list[str]) -> dict[str, dict]:
    """One-vs-rest agreement for each class."""
    out = {}
    for cls in CLASSES:
        binary_a = [x == cls for x in a]
        binary_b = [y == cls for y in b]
        both = sum(x and y for x, y in zip(binary_a, binary_b))
        either = sum(x or y for x, y in zip(binary_a, binary_b))
        out[cls] = {
            "n_a": sum(binary_a),
            "n_b": sum(binary_b),
            # Jaccard: of the items either annotator put in this class, how many
            # did both. Harsher and more informative than raw agreement on a
            # rare class, where agreeing on all the negatives looks like success.
            "jaccard": round(both / either, 4) if either else None,
            "kappa": round(
                cohens_kappa(
                    [cls if x else "other" for x in binary_a],
                    [cls if y else "other" for y in binary_b],
                    [cls, "other"],
                ),
                4,
            ),
        }
    return out


def interpret(kappa: float) -> str:
    """Landis & Koch (1977) bands, the convention this literature reports in."""
    for threshold, name in [
        (0.81, "almost perfect"),
        (0.61, "substantial"),
        (0.41, "moderate"),
        (0.21, "fair"),
        (0.0, "slight"),
    ]:
        if kappa >= threshold:
            return name
    return "poor (worse than chance)"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Inter-annotator agreement")
    parser.add_argument("--a", required=True, help="first annotator")
    parser.add_argument("--b", default=None, help="second annotator")
    parser.add_argument(
        "--vs-weak",
        action="store_true",
        help="compare annotator --a against the weak labeling rule instead",
    )
    parser.add_argument("--out", default="agreement.json")
    args = parser.parse_args()

    labels_a = load_labels(args.a)

    if args.vs_weak:
        keys = [k for k, v in labels_a.items() if v.get("weak_label")]
        left = [labels_a[k]["label"] for k in keys]
        right = [labels_a[k]["weak_label"] for k in keys]
        name_a, name_b = args.a, "weak rule"
    else:
        if not args.b:
            raise SystemExit("give --b <annotator> or use --vs-weak")
        labels_b = load_labels(args.b)
        keys = sorted(set(labels_a) & set(labels_b))
        if not keys:
            raise SystemExit(
                "no overlapping pairs. Label the same sample with:\n"
                f"  python -m train.label --annotator {args.b} --same-sample-as {args.a}"
            )
        left = [labels_a[k]["label"] for k in keys]
        right = [labels_b[k]["label"] for k in keys]
        name_a, name_b = args.a, args.b

    n = len(keys)
    raw = sum(x == y for x, y in zip(left, right)) / n
    kappa = cohens_kappa(left, right)

    print(f"\n  {name_a} vs {name_b}   (n={n} overlapping pairs)")
    print(f"  raw agreement    {raw:.4f}")
    print(f"  Cohen's kappa    {kappa:.4f}   ({interpret(kappa)})")
    print("\n  confusion (rows = {}, cols = {})".format(name_a, name_b))
    matrix = confusion(left, right)
    print("               " + "".join(f"{c:>12}" for c in CLASSES))
    for cls in CLASSES:
        print(f"  {cls:>11}  " + "".join(f"{matrix[cls][c]:>12}" for c in CLASSES))

    per_class = per_class_agreement(left, right)
    print("\n  per class")
    for cls, stats in per_class.items():
        jac = f"{stats['jaccard']:.4f}" if stats["jaccard"] is not None else "n/a"
        print(f"    {cls:>10}  kappa {stats['kappa']:>7.4f}   jaccard {jac:>7}   "
              f"n({name_a})={stats['n_a']}  n({name_b})={stats['n_b']}")

    payload = {
        "comparison": f"{name_a} vs {name_b}",
        "n": n,
        "raw_agreement": round(raw, 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": interpret(kappa),
        "confusion": matrix,
        "per_class": per_class,
    }
    config.EVAL.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.EVAL.results_dir / args.out
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
