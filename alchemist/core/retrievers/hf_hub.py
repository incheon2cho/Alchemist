"""HFHubRetriever — search HuggingFace Hub for models, datasets, and papers.

Used by:
  - Benchmark Agent: dynamic candidate model scouting (replaces hard-coded list)
  - Research Agent: discover task-specific pretrained models and benchmark context

Also hosts the PwC-archive lookup (`search_pwc_leaderboard`) since PaperWithCode's
final data dumps live on HF at ``pwc-archive/*``.

All calls are cached to ``~/.cache/alchemist/retrievers/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_CACHE_DIR = Path(
    os.environ.get("ALCHEMIST_RETRIEVER_CACHE", Path.home() / ".cache" / "alchemist" / "retrievers")
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600


def _cache_key(namespace: str, payload: dict[str, Any]) -> Path:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    h = hashlib.sha1(blob).hexdigest()[:16]
    return _CACHE_DIR / f"{namespace}_{h}.json"


def _load_cache(path: Path, ttl: int) -> Any | None:
    if not path.exists() or time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        log.warning("cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Pretraining source detection (policy filter)
# ---------------------------------------------------------------------------
_IN1K_KEYWORDS = ["in1k", "imagenet-1k", "imagenet1k", "in_1k"]
_IN21K_KEYWORDS = ["in21k", "in22k", "imagenet-21k", "imagenet-22k", "imagenet21k", "imagenet22k"]
_BIG_DATA_KEYWORDS = ["jft", "laion", "ig-", "wit-", "datacomp", "mim_", "lvd-"]


def classify_pretrain_source(tags_or_name: list[str] | str) -> str:
    """Return 'imagenet-1k', 'imagenet-21k', 'large-extra', or 'unknown'."""
    if isinstance(tags_or_name, str):
        text = tags_or_name.lower()
    else:
        text = " ".join(str(t).lower() for t in tags_or_name)

    if any(k in text for k in _IN1K_KEYWORDS):
        return "imagenet-1k"
    if any(k in text for k in _IN21K_KEYWORDS):
        return "imagenet-21k"
    if any(k in text for k in _BIG_DATA_KEYWORDS):
        return "large-extra"
    return "unknown"


# ---------------------------------------------------------------------------
# HFHubRetriever
# ---------------------------------------------------------------------------
class HFHubRetriever:
    """Model & dataset scouting on HuggingFace Hub + PwC archive leaderboards."""

    def __init__(self, cache_ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self.cache_ttl = cache_ttl_seconds
        self._pwc_eval_df = None  # lazy-loaded

    # ------- Models -------------------------------------------------------
    def search_models(
        self,
        pipeline_tag: str | None = None,
        library: str | None = None,
        search_query: str | None = None,
        sort: str = "downloads",
        limit: int = 30,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List models with optional filters. Returns compact dicts.

        Sort is always descending (``direction=-1`` baked in; newer
        huggingface_hub releases no longer accept the keyword).
        """
        key = _cache_key(
            "hf_models",
            dict(pipeline_tag=pipeline_tag, library=library, q=search_query,
                 sort=sort, limit=limit, tags=filter_tags),
        )
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            return cached

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            kwargs: dict[str, Any] = {"sort": sort, "limit": limit}
            if pipeline_tag:
                kwargs["pipeline_tag"] = pipeline_tag
            if search_query:
                kwargs["search"] = search_query
            # `library` and `tags` moved into `filter` in newer huggingface_hub.
            filt: list[str] = []
            if library:
                filt.append(library)
            if filter_tags:
                filt.extend(filter_tags)
            if filt:
                kwargs["filter"] = filt
            models = list(api.list_models(**kwargs))
        except Exception as e:
            log.warning("hf search_models failed: %s", e)
            return []

        out: list[dict[str, Any]] = []
        for m in models:
            tags = list(getattr(m, "tags", []) or [])
            out.append({
                "id": m.id,
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "library_name": getattr(m, "library_name", None),
                "tags": tags,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "pretrain_source": classify_pretrain_source(tags + [m.id]),
            })
        _save_cache(key, out)
        log.info("hf search_models (pipeline=%s, lib=%s, q=%s) → %d",
                 pipeline_tag, library, search_query, len(out))
        return out

    def search_imagenet1k_models(
        self,
        limit: int = 30,
        library: str = "timm",
    ) -> list[dict[str, Any]]:
        """Shorthand for image-classification + ImageNet-1K pretrain only."""
        models = self.search_models(
            pipeline_tag="image-classification",
            library=library,
            sort="downloads", limit=limit * 3,
        )
        return [m for m in models if m["pretrain_source"] == "imagenet-1k"][:limit]

    _TIMM_CSV_URL = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/results-imagenet.csv"
    _TIMM_RESULTS: dict[str, dict[str, float]] | None = None

    @classmethod
    def _load_timm_imagenet_results(cls) -> dict[str, dict[str, float]]:
        """Load (and cache) timm's official ImageNet top-1 results CSV.

        Returns ``{model_id: {"top1": float, "top5": float, "param_count": float, "img_size": int}}``.
        """
        if cls._TIMM_RESULTS is not None:
            return cls._TIMM_RESULTS
        cache_path = _CACHE_DIR / "timm_imagenet_results.csv"
        text = None
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 30 * 24 * 3600:
            text = cache_path.read_text()
        if text is None:
            try:
                import urllib.request
                with urllib.request.urlopen(cls._TIMM_CSV_URL, timeout=10) as r:
                    text = r.read().decode()
                cache_path.write_text(text)
            except Exception as e:
                log.warning("timm CSV fetch failed (%s); using empty results", e)
                cls._TIMM_RESULTS = {}
                return cls._TIMM_RESULTS
        import csv, io
        out: dict[str, dict[str, float]] = {}
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            model_id = (row.get("model") or "").strip()
            if not model_id:
                continue
            try:
                out[model_id] = {
                    "top1": float(row.get("top1", 0) or 0),
                    "top5": float(row.get("top5", 0) or 0),
                    "param_count": float((row.get("param_count", "0") or "0").replace(",", "")),
                    "img_size": int(row.get("img_size", 224) or 224),
                }
            except ValueError:
                continue
        cls._TIMM_RESULTS = out
        log.info("Loaded timm ImageNet results: %d models", len(out))
        return out

    def timm_imagenet_top1(self, model_id: str) -> float | None:
        """Real ImageNet top-1 accuracy for a timm model id; None if unknown."""
        results = self._load_timm_imagenet_results()
        clean = model_id.replace("timm/", "", 1)
        row = results.get(clean) or results.get(clean.split(".")[0])
        return row["top1"] if row else None

    def get_model_meta(self, model_id: str) -> dict[str, Any]:
        """Fetch full model metadata (model card text, config)."""
        key = _cache_key("hf_model_meta", {"id": model_id})
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            return cached
        try:
            from huggingface_hub import HfApi, ModelCard
            api = HfApi()
            info = api.model_info(model_id)
            data: dict[str, Any] = {
                "id": info.id,
                "tags": list(info.tags or []),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "library_name": getattr(info, "library_name", None),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "config": getattr(info, "config", None) or {},
            }
            try:
                card = ModelCard.load(model_id)
                data["card"] = (card.content or "")[:4000]
            except Exception:
                data["card"] = ""
            data["pretrain_source"] = classify_pretrain_source(
                data["tags"] + [data["id"]] + [data.get("card", "")[:500]]
            )
        except Exception as e:
            log.warning("hf get_model_meta(%s) failed: %s", model_id, e)
            return {}
        _save_cache(key, data)
        return data

    # ------- Datasets -----------------------------------------------------
    def search_datasets(
        self, search_query: str | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        key = _cache_key("hf_datasets", {"q": search_query, "limit": limit})
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            return cached
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            ds = list(api.list_datasets(search=search_query, limit=limit))
        except Exception as e:
            log.warning("hf search_datasets failed: %s", e)
            return []
        out = [{"id": d.id, "downloads": getattr(d, "downloads", 0),
                "likes": getattr(d, "likes", 0),
                "tags": list(getattr(d, "tags", []) or [])} for d in ds]
        _save_cache(key, out)
        return out

    # ------- PwC archive (evaluation tables) ------------------------------
    def _load_pwc_eval(self):
        """Load (and cache on disk) the Papers-with-Code evaluation tables dump.

        Data comes from ``pwc-archive/evaluation-tables`` on HuggingFace — the
        final snapshot of PwC's SoTA leaderboards before shutdown. ~138 MB total.
        Cached as a pandas DataFrame in memory after first load; parquet files
        live under the HF cache (``~/.cache/huggingface/``).
        """
        if self._pwc_eval_df is not None:
            return self._pwc_eval_df
        try:
            from datasets import load_dataset
            ds = load_dataset("pwc-archive/evaluation-tables", split="train")
            self._pwc_eval_df = ds.to_pandas()
            log.info("loaded pwc evaluation-tables: %d rows", len(self._pwc_eval_df))
        except Exception as e:
            log.warning("pwc-archive load failed: %s", e)
            self._pwc_eval_df = False  # sentinel meaning "tried and failed"
        return self._pwc_eval_df

    def search_pwc_leaderboard(
        self,
        dataset_name: str,
        metric_keywords: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the top entries on a PwC benchmark matching ``dataset_name``.

        Args:
            dataset_name: e.g. "CIFAR-100", "ImageNet", "COCO"
            metric_keywords: keep only rows whose metric mentions one of these
                (e.g. ["accuracy", "top-1"]); None → all metrics.
            top_k: number of top entries per task.
        """
        key = _cache_key("pwc_lb", dict(d=dataset_name, m=metric_keywords, k=top_k))
        cached = _load_cache(key, self.cache_ttl)
        if cached is not None:
            return cached

        df = self._load_pwc_eval()
        if df is False or df is None or getattr(df, "empty", True):
            return []

        target = dataset_name.lower().replace("-", " ").replace("_", " ")
        hits: list[dict[str, Any]] = []

        # PwC dump structure: rows have nested dict 'datasets' with sota results.
        # Schema may vary — we iterate robustly.
        import pandas as pd  # noqa

        for _, row in df.iterrows():
            task = str(row.get("task", "")).lower()
            desc = str(row.get("description", "")).lower()
            datasets = row.get("datasets")
            if datasets is None:
                continue

            for d in datasets:
                dname = str(d.get("dataset", "")).lower().replace("-", " ").replace("_", " ")
                if target not in dname and target not in task and target not in desc:
                    continue
                sota = d.get("sota")
                if sota is None:
                    continue
                rows = sota.get("rows")
                if rows is None or (hasattr(rows, "__len__") and len(rows) == 0):
                    continue
                metrics_info = sota.get("metrics")
                for r in rows:
                    if r is None:
                        continue
                    metrics = r.get("metrics")
                    if metrics is None or (hasattr(metrics, "__len__") and len(metrics) == 0):
                        metrics = {}
                    # Optional metric keyword filter
                    if metric_keywords:
                        matched = False
                        for mk in metric_keywords:
                            if any(mk.lower() in k.lower() for k in metrics.keys()):
                                matched = True
                                break
                        if not matched:
                            continue
                    # Normalize numpy arrays to Python lists/dicts
                    code_links = r.get("code_links")
                    if code_links is None or (hasattr(code_links, "__len__") and len(code_links) == 0):
                        code_links = []
                    else:
                        code_links = list(code_links) if hasattr(code_links, "tolist") else code_links
                    pdate = r.get("paper_date")
                    hits.append({
                        "task": str(row.get("task") or ""),
                        "dataset": str(d.get("dataset") or ""),
                        "model": str(r.get("model_name") or ""),
                        "metrics": {str(k): str(v) for k, v in dict(metrics).items()},
                        "paper_title": str(r.get("paper_title") or ""),
                        "paper_url": str(r.get("paper_url") or ""),
                        "paper_date": str(pdate) if pdate is not None else None,
                        "code_links_count": len(code_links) if hasattr(code_links, "__len__") else 0,
                        "uses_additional_data": bool(r.get("uses_additional_data")) if r.get("uses_additional_data") is not None else False,
                    })

        # Rank: prefer entries with numeric metric values, sort by primary metric desc.
        def _primary_score(h):
            m = h.get("metrics", {})
            for k in ("Accuracy", "Top 1 Accuracy", "Percentage correct", "mAP", "F1", "AUC"):
                for mk, mv in m.items():
                    if k.lower() in mk.lower():
                        try:
                            v = float(str(mv).rstrip("% "))
                            return v
                        except Exception:
                            pass
            return -1

        hits.sort(key=_primary_score, reverse=True)
        out = hits[:top_k]
        _save_cache(key, out)
        log.info("pwc leaderboard '%s' → %d entries", dataset_name, len(out))
        return out

    # ------- LLM-friendly serializers ------------------------------------
    def summarize_models_for_llm(self, models: list[dict], max_chars: int = 3000) -> str:
        if not models:
            return "(no HuggingFace models found)"
        lines = []
        for i, m in enumerate(models[:20], 1):
            lines.append(
                f"[{i}] {m['id']}  "
                f"pipeline={m.get('pipeline_tag','?')}  lib={m.get('library_name','?')}  "
                f"pretrain={m.get('pretrain_source','?')}  "
                f"dl={m.get('downloads', 0)}"
            )
        return "\n".join(lines)[:max_chars]

    def summarize_leaderboard_for_llm(self, entries: list[dict], max_chars: int = 3000) -> str:
        if not entries:
            return "(no Papers-with-Code leaderboard entries found)"
        lines = []
        for i, e in enumerate(entries, 1):
            metrics = e.get("metrics") or {}
            metric_str = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:3])
            extra = " [uses extra data]" if e.get("uses_additional_data") else ""
            lines.append(
                f"[{i}] {e.get('dataset', '?')} | {e.get('task', '?')} | "
                f"{e.get('model', '?')} → {metric_str}{extra}  ({e.get('paper_date', '?')})"
            )
        return "\n".join(lines)[:max_chars]
