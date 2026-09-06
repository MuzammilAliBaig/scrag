"""PopQA loader with a fixed, seeded, on-disk eval split.

Every phase from here on evaluates against these exact question ids. Resampling
the split invalidates the whole A-E ablation, so the ids are written to disk on
first use and read back thereafter - the split is never regenerated silently.

PopQA upstream ships a single `test` split of 14,267 questions. The dev/test
carve-up below is ours.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass(frozen=True)
class QAExample:
    """One PopQA question with its accepted answer surface forms."""

    qid: str
    question: str
    answers: list[str]
    # The Wikipedia page for the entity the question is about. This is what
    # Module A indexes, since PopQA ships no corpus of its own.
    subject_title: str
    subject: str
    prop: str
    metadata: dict[str, str] = field(default_factory=dict)


def _parse_answers(raw: str | list) -> list[str]:
    """PopQA stores possible_answers as a JSON-encoded list of strings."""
    if isinstance(raw, list):
        return [str(a) for a in raw]
    try:
        return [str(a) for a in json.loads(raw)]
    except (json.JSONDecodeError, TypeError):
        return [str(raw)]


def _load_popqa_rows(cfg: config.DatasetConfig = config.DATASET) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg.popqa_hf_id, split=cfg.popqa_hf_split)
    return [dict(row) for row in ds]


def _to_example(row: dict) -> QAExample:
    return QAExample(
        qid=str(row["id"]),
        question=row["question"],
        answers=_parse_answers(row["possible_answers"]),
        subject_title=row.get("s_wiki_title") or row.get("subj") or "",
        subject=row.get("subj", ""),
        prop=row.get("prop", ""),
        metadata={"obj": str(row.get("obj", "")), "s_pop": str(row.get("s_pop", ""))},
    )


def _split_path(split: str, cfg: config.DatasetConfig = config.DATASET) -> Path:
    return cfg.splits_dir / f"popqa_{split}.json"


def build_splits(
    cfg: config.DatasetConfig = config.DATASET,
    seed: int = config.SEED,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Create the seeded dev/test id lists once and persist them.

    Refuses to overwrite an existing split unless asked, because a silently
    resampled split would invalidate every number measured against the old one.
    """
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    dev_path, test_path = _split_path("dev", cfg), _split_path("test", cfg)

    if dev_path.exists() and test_path.exists() and not overwrite:
        return {
            "dev": json.loads(dev_path.read_text(encoding="utf-8")),
            "test": json.loads(test_path.read_text(encoding="utf-8")),
        }

    rows = _load_popqa_rows(cfg)
    qids = [str(r["id"]) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(qids)

    dev = qids[: cfg.dev_size]
    test = qids[cfg.dev_size : cfg.dev_size + cfg.test_size]
    # Entities used only as retrieval noise - never evaluated on.
    distractor = qids[cfg.dev_size + cfg.test_size :][: config.CORPUS.distractor_pool_size]

    dev_path.write_text(json.dumps(dev, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test, indent=2), encoding="utf-8")
    _split_path("distractor", cfg).write_text(json.dumps(distractor, indent=2), encoding="utf-8")

    return {"dev": dev, "test": test, "distractor": distractor}


def load_split(
    split: str = "dev",
    limit: int | None = None,
    cfg: config.DatasetConfig = config.DATASET,
) -> list[QAExample]:
    """Load examples for a persisted split, in the split file's fixed order.

    `limit` truncates from the front, so a limited run is always a prefix of
    the full run and the two stay comparable.
    """
    path = _split_path(split, cfg)
    if not path.exists():
        build_splits(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no split file at {path}")

    wanted = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        wanted = wanted[:limit]
    wanted_set = set(wanted)

    by_id = {str(r["id"]): r for r in _load_popqa_rows(cfg) if str(r["id"]) in wanted_set}
    missing = wanted_set - by_id.keys()
    if missing:
        raise ValueError(f"{len(missing)} split ids not found upstream, e.g. {list(missing)[:3]}")

    return [_to_example(by_id[qid]) for qid in wanted]


def subject_titles(splits: list[str], cfg: config.DatasetConfig = config.DATASET) -> list[str]:
    """Deduplicated Wikipedia page titles for the entities in the given splits."""
    seen: dict[str, None] = {}
    for split in splits:
        for ex in load_split(split, cfg=cfg):
            if ex.subject_title:
                seen.setdefault(ex.subject_title, None)
    return list(seen)
