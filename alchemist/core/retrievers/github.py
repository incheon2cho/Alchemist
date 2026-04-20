"""GitHubRetriever — search GitHub for vision model repositories.

Discovers pretrained model implementations beyond timm/HuggingFace:
official paper repos, community reimplementations, torch.hub-compatible
models, etc. Results are cached to respect GitHub API rate limits.

Usage::

    ret = GitHubRetriever()
    repos = ret.search_model_repos("SwinV2 CIFAR-100 pytorch pretrained")
    for r in repos:
        print(r["full_name"], r["stars"], r["has_hubconf"])
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "alchemist" / "retrievers"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_TTL = 24 * 3600  # 1 day


class GitHubRetriever:
    """Search GitHub for vision model repositories with pretrained weights."""

    SEARCH_URL = "https://api.github.com/search/repositories"

    def __init__(self, cache_ttl: int = _DEFAULT_TTL) -> None:
        self.cache_ttl = cache_ttl

    def search_model_repos(
        self,
        query: str,
        language: str = "Python",
        min_stars: int = 10,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search GitHub for model repositories matching the query.

        Returns list of repo metadata dicts with:
          full_name, description, stars, url, has_hubconf, has_weights,
          topics, updated_at, license.
        """
        cache_key = f"gh_{query}_{language}_{min_stars}_{top_k}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        import urllib.request
        import urllib.parse

        q = f"{query} language:{language} stars:>={min_stars}"
        params = urllib.parse.urlencode({
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": min(top_k * 2, 30),  # fetch extra, filter later
        })
        url = f"{self.SEARCH_URL}?{params}"

        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Alchemist-Vision-Agent/1.0",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            log.warning("GitHub search failed: %s", e)
            return []

        results = []
        for item in data.get("items", [])[:top_k * 2]:
            repo = {
                "full_name": item.get("full_name", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stargazers_count", 0),
                "url": item.get("html_url", ""),
                "topics": item.get("topics", []),
                "updated_at": item.get("updated_at", ""),
                "license": (item.get("license") or {}).get("spdx_id", ""),
                "language": item.get("language", ""),
                "source": "github",
            }

            # Check for torch.hub compatibility and pretrained weights
            repo["has_hubconf"] = self._check_file_exists(
                repo["full_name"], "hubconf.py"
            )
            repo["has_weights"] = any(
                kw in (repo["description"] + " ".join(repo["topics"])).lower()
                for kw in ("pretrained", "weights", "checkpoint", "model zoo",
                           "pre-trained", "pretrain")
            )

            if repo["stars"] >= min_stars:
                results.append(repo)

            if len(results) >= top_k:
                break

        log.info("GitHub search '%s': %d repos found", query[:50], len(results))
        self._save_cache(cache_key, results)
        return results

    def search_vision_models(
        self,
        task_name: str = "",
        architecture: str = "",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Convenience method: search for vision model repos relevant to a task.

        Combines task-specific and architecture-specific queries.
        """
        queries = []
        if task_name:
            queries.append(f"{task_name} pytorch pretrained model")
        if architecture:
            queries.append(f"{architecture} pytorch pretrained")
        queries.append("vision model pytorch pretrained imagenet")

        all_repos: list[dict[str, Any]] = []
        seen = set()
        for q in queries:
            repos = self.search_model_repos(q, top_k=top_k)
            for r in repos:
                if r["full_name"] not in seen:
                    all_repos.append(r)
                    seen.add(r["full_name"])
            time.sleep(2)  # respect rate limit (10 req/min unauthenticated)

        # Sort by stars descending
        all_repos.sort(key=lambda r: r["stars"], reverse=True)
        return all_repos[:top_k]

    def summarize_for_llm(
        self,
        repos: list[dict[str, Any]],
        max_chars: int = 2000,
    ) -> str:
        """Format repo list for LLM prompt context."""
        if not repos:
            return "(no GitHub repos found)"
        lines = ["GitHub model repositories:"]
        for i, r in enumerate(repos, 1):
            hub = " [torch.hub]" if r.get("has_hubconf") else ""
            wt = " [pretrained]" if r.get("has_weights") else ""
            lines.append(
                f"  [{i}] {r['full_name']} (⭐{r['stars']}){hub}{wt}: "
                f"{r['description'][:100]}"
            )
        text = "\n".join(lines)
        return text[:max_chars]

    @staticmethod
    def _check_file_exists(full_name: str, filename: str) -> bool:
        """Check if a file exists in a GitHub repo's default branch."""
        import urllib.request
        url = f"https://api.github.com/repos/{full_name}/contents/{filename}"
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Alchemist-Vision-Agent/1.0",
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _cache_path(self, key: str) -> Path:
        import hashlib
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return _CACHE_DIR / f"github_{h}.json"

    def _load_cache(self, key: str) -> list | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.cache_ttl:
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _save_cache(self, key: str, data: list) -> None:
        try:
            self._cache_path(key).write_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass
