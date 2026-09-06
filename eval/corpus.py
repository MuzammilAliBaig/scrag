"""Corpus construction for Module A.

PopQA ships questions and answers but no documents, so there is nothing to
retrieve over out of the box. This module fetches the Wikipedia article for
each question's subject entity via the free MediaWiki API and writes them to a
JSONL corpus.

Two design points worth defending in the report:

1. The corpus includes a pool of *distractor* entities drawn from PopQA
   questions outside the eval split. Without them the corpus contains almost
   exactly the answers being asked about, retrieval becomes near-trivial, and
   the Phase 1 baseline flatters itself - leaving Module B nothing to correct.
2. Articles are truncated to a configured character budget, so corpus size
   stays bounded and CPU embedding stays in minutes rather than hours.

Free, no API key, no rate limit beyond politeness.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import config


def _fetch_batch(titles: list[str], cfg: config.CorpusConfig) -> dict[str, str]:
    """Fetch plain-text extracts for up to `titles_per_request` page titles."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        # MediaWiki returns an extract for exactly ONE title per request and
        # silently drops the rest unless BOTH of these are set: exlimit raises
        # the cap to 20, and the API only honours it for intro-only extracts.
        # Lead sections are what we want anyway - PopQA asks about properties
        # (occupation, birthplace, nationality) that live in the first
        # paragraphs - and they keep the corpus small enough to embed on CPU.
        "exlimit": "max",
        "exintro": "1",
        "redirects": "1",
        "titles": "|".join(titles),
        "formatversion": "2",
    }
    url = f"{cfg.wikipedia_api}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})

    last_error: Exception | None = None
    for attempt in range(cfg.retry_attempts):
        try:
            with urllib.request.urlopen(request, timeout=cfg.request_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            # Honour Retry-After when Wikipedia sends one, else back off hard.
            wait = float(exc.headers.get("Retry-After") or 0) or cfg.backoff_base_s * (3 ** attempt)
            print(f"\n  rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
        except Exception as exc:  # network flakiness, not a logic error
            last_error = exc
            time.sleep(cfg.backoff_base_s * (2 ** attempt))
    else:
        raise RuntimeError(
            f"Wikipedia fetch failed after {cfg.retry_attempts} attempts"
        ) from last_error

    out: dict[str, str] = {}
    for page in payload.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        extract = (page.get("extract") or "").strip()
        if extract:
            out[page["title"]] = extract[: cfg.max_chars_per_doc]
    return out


def build_corpus(
    titles: list[str],
    cfg: config.CorpusConfig = config.CORPUS,
    resume: bool = True,
) -> dict[str, str]:
    """Fetch every title and write the corpus to JSONL.

    Resumable: titles already on disk are not refetched, so an interrupted run
    picks up where it stopped.
    """
    path = cfg.corpus_path
    path.parent.mkdir(parents=True, exist_ok=True)

    docs: dict[str, str] = {}
    if resume and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                docs[row["title"]] = row["text"]

    pending = [t for t in dict.fromkeys(titles) if t not in docs]
    print(f"corpus: {len(docs)} cached, {len(pending)} to fetch")

    for start in range(0, len(pending), cfg.titles_per_request):
        batch = pending[start : start + cfg.titles_per_request]
        fetched = _fetch_batch(batch, cfg)
        docs.update(fetched)
        done = min(start + cfg.titles_per_request, len(pending))
        print(f"  {done}/{len(pending)} titles ({len(docs)} docs)", end="\r", flush=True)
        # Wikipedia redirects mean a requested title may land under another
        # name; both are kept, so the count can exceed the request count.
        _write_corpus(docs, path)
        time.sleep(cfg.delay_between_batches_s)  # politeness

    print()
    return docs


def _write_corpus(docs: dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for title, text in docs.items():
            fh.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")


def load_corpus(cfg: config.CorpusConfig = config.CORPUS) -> dict[str, str]:
    """Read the corpus back as a {doc_id: text} mapping for Module A."""
    if not cfg.corpus_path.exists():
        raise FileNotFoundError(
            f"no corpus at {cfg.corpus_path} - run `python -m eval.build_index` first"
        )
    docs: dict[str, str] = {}
    for line in cfg.corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            docs[row["title"]] = row["text"]
    return docs
