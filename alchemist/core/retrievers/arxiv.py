"""ArxivRetriever — thin wrapper around the `arxiv` library.

Provides year-filtered keyword search and abstract retrieval for use by
Alchemist's Research Agent (and occasionally Benchmark Agent). Results are
disk-cached per query to avoid hammering arxiv.org on every planning round.

Usage::

    ret = ArxivRetriever()
    papers = ret.search("CIFAR-100 SAM optimizer", years=[2023, 2024, 2025], top_k=5)
    for p in papers:
        print(p["title"], p["year"], p["url"])
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CACHE_DIR = Path(
    os.environ.get("ALCHEMIST_RETRIEVER_CACHE", Path.home() / ".cache" / "alchemist" / "retrievers")
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _cache_key(namespace: str, payload: dict[str, Any]) -> Path:
    blob = json.dumps(payload, sort_keys=True).encode()
    h = hashlib.sha1(blob).hexdigest()[:16]
    return _CACHE_DIR / f"{namespace}_{h}.json"


def _load_cache(path: Path, ttl: int) -> Any | None:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        log.warning("cache write failed (%s): %s", path, e)


class ArxivRetriever:
    """Search arXiv with optional year filter + disk cache."""

    def __init__(self, cache_ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self.cache_ttl = cache_ttl_seconds

    def search(
        self,
        query: str,
        years: list[int] | None = None,
        top_k: int = 5,
        sort_by: str = "relevance",
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` papers matching ``query`` (optionally within ``years``).

        Each paper dict has keys: arxiv_id, title, summary, authors, year, url, categories.
        """
        key = _cache_key("arxiv_search", {"q": query, "years": years, "k": top_k, "s": sort_by})
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            log.info("arxiv cache hit: %s", query)
            return cached

        try:
            import arxiv
        except ImportError:
            log.warning("arxiv library not installed")
            return []

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }

        client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=top_k * 3 if years else top_k,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        )

        out: list[dict[str, Any]] = []
        try:
            for result in client.results(search):
                yr = result.published.year if result.published else None
                if years and yr not in years:
                    continue
                out.append(
                    {
                        "arxiv_id": result.entry_id.rsplit("/", 1)[-1],
                        "title": result.title.strip(),
                        "summary": (result.summary or "").strip()[:1000],
                        "authors": [str(a) for a in result.authors][:5],
                        "year": yr,
                        "url": result.entry_id,
                        "categories": list(result.categories or []),
                        "primary_category": getattr(result, "primary_category", None),
                    }
                )
                if len(out) >= top_k:
                    break
        except Exception as e:
            log.warning("arxiv search failed for %r: %s", query, e)
            return []

        _save_cache(key, out)
        log.info("arxiv search '%s' → %d papers", query, len(out))
        return out

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch one paper by arxiv id (e.g. '2305.12345')."""
        key = _cache_key("arxiv_paper", {"id": arxiv_id})
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            return cached

        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            for r in client.results(search):
                data = {
                    "arxiv_id": arxiv_id,
                    "title": r.title.strip(),
                    "summary": (r.summary or "").strip()[:2000],
                    "authors": [str(a) for a in r.authors],
                    "year": r.published.year if r.published else None,
                    "url": r.entry_id,
                }
                _save_cache(key, data)
                return data
        except Exception as e:
            log.warning("arxiv get_paper(%s) failed: %s", arxiv_id, e)
        return None

    def summarize_for_llm(self, papers: list[dict[str, Any]], max_chars: int = 3000) -> str:
        """Render a compact list suitable for LLM prompt injection."""
        if not papers:
            return "(no arXiv results)"
        lines = []
        for i, p in enumerate(papers, 1):
            line = (
                f"[{i}] {p.get('title', '?')} ({p.get('year', '?')}) "
                f"arxiv:{p.get('arxiv_id', '?')}\n"
                f"    {(p.get('summary') or '')[:400].replace(chr(10), ' ')}"
            )
            lines.append(line)
        out = "\n".join(lines)
        return out[:max_chars]
