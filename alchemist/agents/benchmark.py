"""Benchmark Agent — 최신 모델 탐색 + 벤치마크 측정 + 순위화.

Responsibilities (AD-1):
- 최신 논문/모델 탐색 (arXiv, HuggingFace, timm)
- 후보 모델들을 다중 벤치마크에서 측정
- 벤치마크별 순위 + 종합 순위(median rank) 산출
- 사용자 태스크에 가장 적합한 모델 추천
"""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import asdict
from typing import Any

from alchemist.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageType,
    make_response,
)
from alchemist.core.llm import LLMClient, MockLLMClient, safe_llm_call
from alchemist.core.schemas import (
    Leaderboard,
    LeaderboardEntry,
    UserTask,
)
from alchemist.core.utils import safe_asdict

logger = logging.getLogger(__name__)


# Known models registry (expandable via Model Scout)
KNOWN_MODELS = [
    {"name": "resnet50", "backend": "timm", "model_id": "resnet50.a1_in1k", "params_m": 25.6, "img_size": 224},
    {"name": "dinov2_vitb14", "backend": "timm", "model_id": "vit_base_patch14_dinov2.lvd142m", "params_m": 86, "img_size": 224},
    {"name": "dinov2_vits14", "backend": "timm", "model_id": "vit_small_patch14_dinov2.lvd142m", "params_m": 22, "img_size": 224},
    {"name": "vit_b16_dino", "backend": "timm", "model_id": "vit_base_patch16_224.dino", "params_m": 86, "img_size": 224},
    {"name": "vit_s16_dino", "backend": "timm", "model_id": "vit_small_patch16_224.dino", "params_m": 22, "img_size": 224},
    {"name": "vit_s16_supervised", "backend": "timm", "model_id": "vit_small_patch16_224.augreg_in21k_ft_in1k", "params_m": 22, "img_size": 224},
    {"name": "vitamin_s_clip", "backend": "timm", "model_id": "vitamin_small_224.datacomp1b_clip", "params_m": 22, "img_size": 224},
    {"name": "mvitv2_tiny", "backend": "timm", "model_id": "mvitv2_tiny.fb_in1k", "params_m": 24, "img_size": 224},
]

# Detection-specific models — used when task is object detection
KNOWN_DETECTION_MODELS: list[dict[str, Any]] = [
    {"name": "yolov8n",  "backend": "ultralytics", "model_id": "yolov8n.pt",  "params_m": 3,   "img_size": 640},
    {"name": "yolov8s",  "backend": "ultralytics", "model_id": "yolov8s.pt",  "params_m": 11,  "img_size": 640},
    {"name": "yolov8m",  "backend": "ultralytics", "model_id": "yolov8m.pt",  "params_m": 26,  "img_size": 640},
    {"name": "yolov8l",  "backend": "ultralytics", "model_id": "yolov8l.pt",  "params_m": 44,  "img_size": 640},
    {"name": "yolov8x",  "backend": "ultralytics", "model_id": "yolov8x.pt",  "params_m": 68,  "img_size": 640},
    {"name": "yolo11n",  "backend": "ultralytics", "model_id": "yolo11n.pt",  "params_m": 3,   "img_size": 640},
    {"name": "yolo11s",  "backend": "ultralytics", "model_id": "yolo11s.pt",  "params_m": 10,  "img_size": 640},
    {"name": "yolo11m",  "backend": "ultralytics", "model_id": "yolo11m.pt",  "params_m": 20,  "img_size": 640},
    {"name": "yolo11l",  "backend": "ultralytics", "model_id": "yolo11l.pt",  "params_m": 26,  "img_size": 640},
    {"name": "yolo11x",  "backend": "ultralytics", "model_id": "yolo11x.pt",  "params_m": 57,  "img_size": 640},
    {"name": "rtdetr-l",  "backend": "ultralytics", "model_id": "rtdetr-l.pt",  "params_m": 32,  "img_size": 640},
    {"name": "rtdetr-x",  "backend": "ultralytics", "model_id": "rtdetr-x.pt",  "params_m": 65,  "img_size": 640},
]

# Published COCO val2017 mAP scores (pretrained weights)
PUBLISHED_DETECTION_SCORES: dict[str, dict[str, float]] = {
    "yolov8n":   {"mAP50": 52.6, "mAP50_95": 37.3},
    "yolov8s":   {"mAP50": 61.8, "mAP50_95": 44.9},
    "yolov8m":   {"mAP50": 67.2, "mAP50_95": 50.2},
    "yolov8l":   {"mAP50": 69.8, "mAP50_95": 52.9},
    "yolov8x":   {"mAP50": 71.0, "mAP50_95": 53.9},
    "yolo11n":   {"mAP50": 53.4, "mAP50_95": 39.5},
    "yolo11s":   {"mAP50": 62.5, "mAP50_95": 47.0},
    "yolo11m":   {"mAP50": 68.1, "mAP50_95": 51.5},
    "yolo11l":   {"mAP50": 69.4, "mAP50_95": 53.4},
    "yolo11x":   {"mAP50": 70.6, "mAP50_95": 54.7},
    "rtdetr-l":  {"mAP50": 71.4, "mAP50_95": 53.0},
    "rtdetr-x":  {"mAP50": 72.8, "mAP50_95": 54.8},
}

# NOTE: DINOv2 models default to 518px input but can run at 224px.
# When using timm, pass img_size=224 to timm.create_model() to override.
# e.g. timm.create_model(model_id, pretrained=True, img_size=224)

# Simulated published scores (real system queries VBench DB or runs benchmarks)
PUBLISHED_SCORES: dict[str, dict[str, float]] = {
    "resnet50":          {"linear_probe": 75.8, "knn": 68.5, "detection_ap": 38.0},
    "dinov2_vitb14":     {"linear_probe": 87.3, "knn": 82.1, "detection_ap": 47.2},
    "dinov2_vits14":     {"linear_probe": 82.2, "knn": 79.0, "detection_ap": 39.1},
    "vit_b16_dino":      {"linear_probe": 81.2, "knn": 76.1, "detection_ap": 42.3},
    "vit_s16_dino":      {"linear_probe": 78.0, "knn": 72.8, "detection_ap": 35.5},
    "vit_s16_supervised":{"linear_probe": 80.5, "knn": 74.2, "detection_ap": 40.1},
    "vitamin_s_clip":    {"linear_probe": 79.3, "knn": 75.5, "detection_ap": 38.8},
    "mvitv2_tiny":       {"linear_probe": 77.6, "knn": 71.3, "detection_ap": 34.2},
}


class BenchmarkAgent:
    """Agent ① — Model Scouting + Benchmarking + Ranking."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        hf_retriever=None,
        enable_retrieval: bool = True,
    ):
        self.llm = llm or MockLLMClient()
        self.name = AgentRole.BENCHMARK
        self.enable_retrieval = enable_retrieval
        self.hf = hf_retriever
        self.arxiv = None
        if enable_retrieval and self.hf is None:
            try:
                from alchemist.core.retrievers import HFHubRetriever
                self.hf = HFHubRetriever()
            except Exception as e:
                logger.warning("HFHubRetriever unavailable (%s); retrieval disabled", e)
                self.enable_retrieval = False
        if enable_retrieval:
            try:
                from alchemist.core.retrievers import ArxivRetriever
                self.arxiv = ArxivRetriever()
            except Exception as e:
                logger.warning("ArxivRetriever unavailable (%s); arXiv evidence disabled", e)
        self.github = None
        if enable_retrieval:
            try:
                from alchemist.core.retrievers import GitHubRetriever
                self.github = GitHubRetriever()
            except Exception as e:
                logger.warning("GitHubRetriever unavailable (%s); GitHub evidence disabled", e)

    def handle_directive(
        self,
        msg: AgentMessage,
        task: UserTask | None = None,
    ) -> AgentMessage:
        """Process directive: scout models, benchmark, rank, recommend."""
        payload = msg.payload
        benchmarks = payload.get("benchmarks", ["linear_probe", "knn"])

        # Use TaskRegistry to determine task-appropriate benchmarks
        from alchemist.core.task_registry import get_task_meta_for_name
        task_meta = get_task_meta_for_name(task.name) if task else None
        if task_meta and task_meta.benchmark_metrics:
            benchmarks = task_meta.benchmark_metrics

        # 1. Scout models (task-aware: seeds from registry)
        search_query = payload.get("search_query", "vision encoder")
        if task_meta and task_meta.task_type != "classification":
            search_query = f"{task_meta.task_type} {' '.join(m['name'] for m in task_meta.known_models[:3])}"
        models = self.scout_models(search_query, task=task)

        # 2. Run benchmarks (PwC actual scores when available, else estimated)
        scored_models = self.run_benchmarks(models, benchmarks, task=task)

        # 3. Build leaderboard with rankings
        leaderboard = self.build_leaderboard(scored_models, benchmarks)

        # 4. Look up SoTA standing (the number the Research Agent must beat)
        sota_standing = None
        if task:
            sota_standing = self.search_sota_standing(task, top_k=5)

        # 5. Recommend best model for user task
        if task:
            leaderboard = self.recommend(leaderboard, task)

        return make_response(
            self.name,
            {
                "leaderboard": safe_asdict(leaderboard),
                "model_count": len(leaderboard.entries),
                "benchmarks": benchmarks,
                "sota_standing": sota_standing,
            },
            episode=msg.episode,
            budget=msg.budget_remaining,
            trace_id=msg.trace_id,
        )

    def search_sota_standing(
        self,
        task: UserTask,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Look up the current SoTA standing for the user's benchmark.

        Returns::
            {
              "task": task.name,
              "entries": [
                {"model": ..., "metrics": {...}, "paper_title": ...,
                 "paper_date": ..., "uses_additional_data": bool}, ...
              ],
              "top_score_pct": float or None,    # best numeric metric found
              "top_model": str or None,
              "top_paper": str or None,
              "summary": "<LLM-friendly multi-line text>",
            }

        Data source: ``pwc-archive`` parquet dumps on HuggingFace (final snapshot
        before PapersWithCode shutdown in July 2025). Falls back to a free-form
        LLM answer if the dump is unavailable.
        """
        result: dict[str, Any] = {
            "task": task.name,
            "entries": [],
            "top_score_pct": None,
            "top_model": None,
            "top_paper": None,
            "summary": "",
        }

        # 1. PwC archive leaderboard lookup
        entries: list[dict[str, Any]] = []
        if self.enable_retrieval and self.hf is not None:
            try:
                entries = self.hf.search_pwc_leaderboard(
                    task.name,
                    metric_keywords=["accuracy", "top-1", "top 1", "f1", "map", "ap"],
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning("PwC SoTA lookup failed: %s", e)

        # 2. Try to extract a numeric top score
        import re
        best_score = -1.0
        for e in entries:
            for v in (e.get("metrics") or {}).values():
                s = str(v)
                m = re.search(r"(\d+(?:\.\d+)?)", s)
                if m:
                    try:
                        val = float(m.group(1))
                        # heuristic: percentages in [0, 100]
                        if 0 < val <= 100 and val > best_score:
                            best_score = val
                            result["top_score_pct"] = val
                            result["top_model"] = e.get("model")
                            result["top_paper"] = e.get("paper_title") or e.get("paper_url")
                    except ValueError:
                        pass

        result["entries"] = entries

        # 3. Always produce an LLM-friendly summary (even when empty)
        if entries:
            lines = [f"Current SoTA standing on {task.name} (from PwC archive snapshot):"]
            for i, e in enumerate(entries, 1):
                metric_s = ", ".join(
                    f"{k}={v}" for k, v in list((e.get('metrics') or {}).items())[:2]
                )
                extra = " [uses additional data]" if e.get("uses_additional_data") else ""
                lines.append(
                    f"  [{i}] {e.get('model', '?')} — {metric_s}{extra} "
                    f"({e.get('paper_date', '?')})"
                )
            result["summary"] = "\n".join(lines)
        else:
            # LLM fallback — ask for estimated SoTA (marks it as LLM-estimated)
            from alchemist.core.task_registry import get_task_meta_for_name as _get_meta2
            _meta2 = _get_meta2(task.name)
            _metric_name = _meta2.eval_metric if _meta2 else "accuracy"
            est = safe_llm_call(
                self.llm,
                (
                    f"What is the reported state-of-the-art {_metric_name} on "
                    f"the {task.name} benchmark (as of your training data)? "
                    f"The metric is {_metric_name} (percentage, 0-100). "
                    f"Return ONLY JSON with keys: top_model, top_score_pct, "
                    f"top_paper_title, year, note. "
                    f"Example: {{\"top_model\": \"Co-DETR\", \"top_score_pct\": 66.0, "
                    f"\"top_paper_title\": \"DETRs with Collaborative...\", \"year\": 2024, "
                    f"\"note\": \"with Swin-L backbone\"}}"
                ),
                fallback={},
            )
            if isinstance(est, dict) and est.get("top_score_pct"):
                result["top_score_pct"] = est.get("top_score_pct")
                result["top_model"] = est.get("top_model")
                result["top_paper"] = est.get("top_paper_title")
                result["summary"] = (
                    f"Estimated SoTA (LLM, no external retrieval): "
                    f"{est.get('top_model')} → {est.get('top_score_pct')}% "
                    f"({est.get('top_paper_title')}, {est.get('year')}). "
                    f"Note: {est.get('note', '')}"
                )
            else:
                result["summary"] = (
                    f"SoTA standing for {task.name} unavailable "
                    f"(no retrieval source + no LLM estimate)"
                )

        from alchemist.core.task_registry import get_task_meta_for_name as _get_meta
        _meta = _get_meta(task.name)
        _our_ceiling = max(_meta.model_ceilings.values()) if _meta and _meta.model_ceilings else 55.0
        _sota = result["top_score_pct"] or 0.0
        logger.info(
            "[REASONING] SoTA standing for %s:\n"
            "  Current SoTA: %.2f%% by %s\n"
            "  Source: %s\n"
            "  Implication: our best model ceiling is ~%.1f%% (gap: ~%.1f%%p from SoTA)",
            task.name, _sota, result["top_model"],
            "PwC archive" if result.get("entries") else "LLM estimation",
            _our_ceiling, _sota - _our_ceiling,
        )
        return result

    def scout_models(self, query: str = "", task: UserTask | None = None) -> list[dict[str, Any]]:
        """Scout candidate models.

        Sources (in order):
          1. PwC leaderboard for ``task.name`` — models with actual SoTA scores
             on this specific task. Tagged with ``pwc_metric`` + compliance flag.
          2. ``KNOWN_MODELS`` — curated seed set (stable baseline).
          3. HuggingFace Hub — live discovery of ImageNet-1K pretrained timm
             models (if retrieval enabled).
          4. LLM — model suggestions informed by the live list above.
        """
        models: list[dict[str, Any]] = []
        seen: set = set()

        # 1. PwC task-specific leaderboard → models with real metric values
        pwc_evidence = ""
        if task and self.enable_retrieval and self.hf is not None:
            dataset_aliases = self._task_to_pwc_aliases(task.name)
            pwc_hits: list[dict[str, Any]] = []
            for alias in dataset_aliases:
                try:
                    pwc_hits = self.hf.search_pwc_leaderboard(
                        alias,
                        metric_keywords=["accuracy", "top-1", "top 1"],
                        top_k=15,
                    )
                    if pwc_hits:
                        break
                except Exception as e:
                    logger.warning("PwC lookup for %s failed: %s", alias, e)
            if pwc_hits:
                for hit in pwc_hits:
                    model_name = hit.get("model", "").strip()
                    if not model_name or model_name in seen:
                        continue
                    score_pct = self._extract_primary_score(hit.get("metrics", {}))
                    models.append({
                        "name": model_name,
                        "source": "pwc",
                        "pwc_score_pct": score_pct,
                        "uses_additional_data": hit.get("uses_additional_data", False),
                        "paper_title": hit.get("paper_title", ""),
                        "paper_date": hit.get("paper_date"),
                    })
                    seen.add(model_name)
                logger.info("PwC scout for '%s': +%d SoTA entries", task.name, len(pwc_hits))
                pwc_evidence = self.hf.summarize_leaderboard_for_llm(pwc_hits, max_chars=1500)

        # 2. Curated seeds from TaskRegistry (task-type-aware)
        from alchemist.core.task_registry import get_task_meta_for_name
        task_meta = get_task_meta_for_name(task.name) if task else None
        seed_models = task_meta.known_models if task_meta else KNOWN_MODELS
        for m in seed_models:
            if m["name"] not in seen:
                models.append({**m, "source": m.get("source", "known")})
                seen.add(m["name"])
        # Attach published scores from registry
        if task_meta and task_meta.published_scores:
            for m in models:
                if m["name"] in task_meta.published_scores:
                    m["published_scores"] = task_meta.published_scores[m["name"]]

        # 3. HF Hub live discovery (pretrain-source-filtered + timm real top1 scoring)
        hub_evidence = "(retrieval disabled)"
        if self.enable_retrieval and self.hf is not None:
            try:
                hub_models = self.hf.search_imagenet1k_models(limit=40, library="timm")
                for m in hub_models:
                    short = m["id"].replace("timm/", "")
                    if short in seen:
                        continue
                    # Hard filter: skip models pretrained on truly external corpora
                    # (LAION/JFT/LVD-142M/CLIP). ImageNet-21K is now allowed.
                    low = short.lower()
                    if any(bad in low for bad in ("laion", "jft", "lvd142m", "dinov2", "clip", "datacomp")):
                        continue
                    top1 = self.hf.timm_imagenet_top1(short)
                    if top1 is None:
                        # Skip HF uploads without an official timm imagenet score —
                        # they're typically personal fine-tunes, not general backbones.
                        continue
                    row = self.hf._load_timm_imagenet_results().get(short) or {}
                    models.append({
                        "name": short,
                        "hf_id": m["id"],
                        "source": "huggingface",
                        "pretrain_source": m.get("pretrain_source", "imagenet-1k"),
                        "downloads": m.get("downloads", 0),
                        "timm_imagenet_top1": top1,
                        "params_m": row.get("param_count", m.get("params_m", 0)),
                        "img_size": row.get("img_size", 224),
                    })
                    seen.add(short)
                hub_evidence = self.hf.summarize_models_for_llm(hub_models, max_chars=2000)
                logger.info(
                    "HF Hub scout: +%d models (with timm top1 for %d)",
                    sum(1 for x in models if x.get("source") == "huggingface"),
                    sum(1 for x in models if x.get("source") == "huggingface" and x.get("timm_imagenet_top1")),
                )
            except Exception as e:
                logger.warning("HF Hub scout failed: %s", e)

        # 4. arXiv — recent SoTA paper architectures on this task
        arxiv_evidence = ""
        if task and self.arxiv is not None:
            try:
                arxiv_query = f"{task.name} {task_meta.task_type if task_meta else 'vision'} SoTA" if task_meta else f"{task.name} image classification SoTA"
                papers = self.arxiv.search(
                    arxiv_query,
                    years=[2023, 2024, 2025],
                    top_k=5,
                )
                if papers:
                    arxiv_evidence = self.arxiv.summarize_for_llm(papers, max_chars=1500)
                    logger.info("arXiv scout for '%s': +%d recent papers", task.name, len(papers))
            except Exception as e:
                logger.warning("arXiv scout failed: %s", e)

        # 5. GitHub — model repos with pretrained weights / torch.hub
        github_evidence = ""
        if task and self.github is not None:
            try:
                repos = self.github.search_vision_models(
                    task_name=task.name,
                    architecture=query,
                    top_k=5,
                )
                if repos:
                    for r in repos:
                        name = r["full_name"]
                        if name not in seen:
                            models.append({
                                "name": name,
                                "source": "github",
                                "stars": r.get("stars", 0),
                                "has_hubconf": r.get("has_hubconf", False),
                                "has_weights": r.get("has_weights", False),
                                "description": r.get("description", ""),
                                "url": r.get("url", ""),
                            })
                            seen.add(name)
                    github_evidence = self.github.summarize_for_llm(repos, max_chars=1500)
                    logger.info("GitHub scout for '%s': +%d repos", task.name, len(repos))
            except Exception as e:
                logger.warning("GitHub scout failed: %s", e)

        # 6. LLM suggestions grounded in all four evidence streams
        ctx_parts = []
        if pwc_evidence:
            ctx_parts.append(f"PwC SoTA for task:\n{pwc_evidence}")
        if hub_evidence:
            ctx_parts.append(f"HuggingFace Hub candidates:\n{hub_evidence}")
        if arxiv_evidence:
            ctx_parts.append(f"Recent arXiv papers on this task:\n{arxiv_evidence}")
        if github_evidence:
            ctx_parts.append(f"GitHub model repos:\n{github_evidence}")
        ctx = "\n\n".join(ctx_parts) or "(no retrieval evidence)"
        llm_result = safe_llm_call(
            self.llm,
            (
                f"Suggest latest vision encoder models for: {query}.\n"
                f"Evidence:\n{ctx}\n\n"
                f"Propose additional models NOT in the list above that may be "
                f"worth benchmarking. Restrict to ImageNet-1K-pretrained only.\n"
                f"Return JSON: {{\"new_models\": [{{\"name\": ..., \"backend\": ..., \"params_m\": ...}}]}}"
            ),
            fallback={"new_models": []},
        )
        if isinstance(llm_result, dict):
            for m in llm_result.get("new_models", []):
                if isinstance(m, dict) and m.get("name") and m["name"] not in seen:
                    m["source"] = m.get("source", "llm_suggestion")
                    models.append(m)
                    seen.add(m["name"])

        logger.info("Model Scout: %d candidates total", len(models))
        return models

    # PwC dataset name aliases per task (task.name → candidate leaderboard names)
    _PWC_ALIASES = {
        "cifar100":   ["CIFAR-100", "Image Classification on CIFAR-100"],
        "cifar10":    ["CIFAR-10", "Image Classification on CIFAR-10"],
        "imagenet":   ["ImageNet", "ImageNet-1k"],
        "butterfly":  ["Butterfly Image Classification", "Fine-Grained Image Classification"],
        "shopee_iet": [],  # not on PwC
    }

    def _task_to_pwc_aliases(self, task_name: str) -> list[str]:
        # First check TaskRegistry for dataset_aliases
        from alchemist.core.task_registry import get_task_meta_for_name
        task_meta = get_task_meta_for_name(task_name)
        if task_meta and task_meta.dataset_aliases:
            return task_meta.dataset_aliases

        key = (task_name or "").lower().replace("-", "").replace("_", "")
        for k, aliases in self._PWC_ALIASES.items():
            if k.replace("_", "") == key:
                return aliases
        return [task_name]  # try the raw name as last resort

    @staticmethod
    def _extract_primary_score(metrics: dict[str, Any]) -> float | None:
        import re
        priority = ("Accuracy", "Top 1", "Top-1", "Percentage correct", "mAP", "F1", "AUC")
        for key in priority:
            for mk, mv in metrics.items():
                if key.lower() in mk.lower():
                    m = re.search(r"(\d+(?:\.\d+)?)", str(mv))
                    if m:
                        try:
                            v = float(m.group(1))
                            if 0 < v <= 100:
                                return v
                        except ValueError:
                            pass
        return None

    @staticmethod
    def _hf_popularity_score(downloads: int) -> float:
        """Map HF monthly downloads to a 0-100 popularity score (log-scaled).

        1 DL → ~0, 100 → ~33, 10K → ~67, 1M → ~100.
        """
        import math
        if downloads <= 0:
            return 0.0
        return min(100.0, math.log10(downloads + 1) * 100.0 / 6.0)

    def run_benchmarks(
        self,
        models: list[dict[str, Any]],
        benchmarks: list[str],
        task: UserTask | None = None,
    ) -> list[dict[str, Any]]:
        """Score models. Priority (per user directive):
          1. HuggingFace Hub models: popularity score (from downloads); when a
             PwC task score also exists for the same model, use PwC accuracy
             instead (better signal than popularity).
          2. PwC leaderboard models: real task accuracy from PwC archive.
          3. PUBLISHED_SCORES seed models (backwards compat).
          4. Simulated score (last resort).

        Source priority for ranking is handled in build_leaderboard().
        """
        # Pre-build name → pwc_score lookup so HF Hub models benefit from PwC
        # accuracy if they appear under the same name in the PwC leaderboard.
        pwc_lookup = {m["name"]: m["pwc_score_pct"]
                      for m in models
                      if m.get("source") == "pwc" and m.get("pwc_score_pct") is not None}

        results = []
        for model in models:
            name = model["name"]
            source = model.get("source", "unknown")
            scores: dict[str, float] = {}

            if source == "huggingface":
                # 1. HF Hub model — prefer timm's official ImageNet top-1 (real
                # performance), else PwC task accuracy, else popularity proxy.
                top1 = model.get("timm_imagenet_top1")
                pwc_match = pwc_lookup.get(name)
                if top1 is not None:
                    base = float(top1)
                elif pwc_match is not None:
                    base = float(pwc_match)
                else:
                    base = self._hf_popularity_score(model.get("downloads", 0))
                for bench in benchmarks:
                    scores[bench] = base
            elif source == "pwc":
                # 2. PwC entry — real task accuracy
                pwc_score = model.get("pwc_score_pct")
                for bench in benchmarks:
                    scores[bench] = pwc_score if pwc_score is not None else self._simulate_score(bench)
            elif model.get("published_scores"):
                # 3a. Model with published scores from registry
                pub = model["published_scores"]
                for bench in benchmarks:
                    scores[bench] = pub.get(bench, self._simulate_score(bench))
            elif name in PUBLISHED_DETECTION_SCORES:
                # 3b. Detection model with published COCO mAP
                det_scores = PUBLISHED_DETECTION_SCORES[name]
                for bench in benchmarks:
                    scores[bench] = det_scores.get(bench, det_scores.get("mAP50_95", self._simulate_score(bench)))
            elif name in PUBLISHED_SCORES:
                # 3c. Seed model with hardcoded score
                published = PUBLISHED_SCORES[name]
                for bench in benchmarks:
                    scores[bench] = published.get(bench, self._simulate_score(bench))
            else:
                # 4. Unknown — simulated
                for bench in benchmarks:
                    scores[bench] = self._simulate_score(bench)

            results.append({**model, "scores": scores})

        return results

    def build_leaderboard(
        self,
        scored_models: list[dict[str, Any]],
        benchmarks: list[str],
    ) -> Leaderboard:
        """Build ranked leaderboard from scored models."""
        entries = []
        for model in scored_models:
            model_src = model.get("source", "")
            if model_src in ("huggingface", "pwc"):
                src = model_src
            elif model["name"] in PUBLISHED_DETECTION_SCORES or model["name"] in PUBLISHED_SCORES:
                src = "published"
            elif model_src == "known":
                src = "known"
            else:
                src = "estimated"
            entries.append(LeaderboardEntry(
                model_name=model["name"],
                backend=model.get("backend", "unknown"),
                params_m=model.get("params_m", 0),
                scores=model["scores"],
                source=src,
                uses_additional_data=bool(model.get("uses_additional_data", False)),
                paper_title=model.get("paper_title", "") or "",
                paper_date=model.get("paper_date"),
            ))

        # User directive: HuggingFace-first, PwC-second priority.
        # Lower tier number ranks higher.
        _source_tier = {"huggingface": 1, "pwc": 2, "published": 3, "known": 3, "estimated": 4}

        # Compute per-benchmark ranks respecting source tier
        for bench in benchmarks:
            sorted_by_bench = sorted(
                entries,
                key=lambda e: (_source_tier.get(e.source, 5), -e.scores.get(bench, 0)),
            )
            for rank, entry in enumerate(sorted_by_bench, 1):
                entry.ranks[bench] = rank

        # Compute overall rank (median of per-benchmark ranks)
        for entry in entries:
            if entry.ranks:
                entry.overall_rank = int(statistics.median(entry.ranks.values()))
            else:
                entry.overall_rank = len(entries)

        entries.sort(key=lambda e: (_source_tier.get(e.source, 5), e.overall_rank))

        from datetime import datetime, timezone
        return Leaderboard(
            entries=entries,
            benchmarks=benchmarks,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def recommend(
        self,
        leaderboard: Leaderboard,
        task: UserTask,
    ) -> Leaderboard:
        """Recommend the best model for user's task. (LLM-assisted)"""
        if not leaderboard.entries:
            return leaderboard

        # Filter by constraints
        candidates = leaderboard.entries
        max_params = task.constraints.get("max_params_m")
        if max_params:
            constrained = [e for e in candidates if e.params_m <= max_params]
            if constrained:
                candidates = constrained

        # Pretrain-data constraint — detect from task description.
        # Block only truly-external corpora (JFT, LAION, LVD-142M). ImageNet-21K
        # is now allowed per updated task constraints.
        desc_low = (task.description or "").lower()
        block_external = (
            "laion" in desc_low or "jft" in desc_low or "lvd-142m" in desc_low
            or "proprietary" in desc_low or "external" in desc_low
        )
        if block_external:
            def _is_external(entry) -> bool:
                low = entry.model_name.lower()
                # Keep in21k/in22k (fine-tuned on ImageNet), drop pure external corpora
                return any(bad in low for bad in ("laion", "jft", "lvd142m", "dinov2", "clip", "datacomp"))

            compliant = [e for e in candidates if not _is_external(e)]
            dropped = [e.model_name for e in candidates if _is_external(e)]
            if compliant:
                if dropped:
                    logger.info(
                        "Constraint filter (block external corpora) removed %d entries: %s",
                        len(dropped), dropped[:5],
                    )
                candidates = compliant

        # Default: recommend overall rank 1 among candidates
        best = min(candidates, key=lambda e: e.overall_rank)

        # LLM refinement with detailed context for downstream agents
        lb_detail = "\n".join(
            f"  #{e.overall_rank} {e.model_name} (backend={e.backend}, {e.params_m:.0f}M): "
            + ", ".join(f"{k}={v:.1f}" for k, v in e.scores.items())
            for e in candidates[:5]
        )
        prompt = (
            f"You are the Benchmark Agent recommending a model for the Research Agent.\n\n"
            f"Task: {task.description} ({task.num_classes} classes, metric={task.eval_metric})\n"
            f"Constraints: {task.constraints}\n\n"
            f"Candidates:\n{lb_detail}\n\n"
            f"Top pick: {best.model_name}\n\n"
            f"Provide:\n"
            f"1. Why this model is best for this specific task\n"
            f"2. Strengths and weaknesses to watch out for\n"
            f"3. Suggestions for the Research Agent on fine-tuning strategy "
            f"(e.g., should backbone be frozen? which adapter? learning rate range?)\n\n"
            f"Keep your answer concise (3-5 sentences)."
        )
        llm_reason = ""
        try:
            llm_reason = self.llm.generate(prompt)
        except Exception:
            llm_reason = f"Best overall rank: {best.model_name}"

        leaderboard.recommendation = best.model_name
        leaderboard.recommendation_reason = llm_reason

        # Top-K candidates for downstream Research Agent (compliant entries only).
        # Limited to the top-3 by overall_rank to keep the baseline pass fast.
        leaderboard.candidates = [e.model_name for e in candidates[:3]]

        logger.info(
            "[REASONING] Benchmark Agent model selection:\n"
            "  Task: %s (%s)\n"
            "  Candidates evaluated: %d models\n"
            "  Selected: %s (rank=%d, params=%.0fM)\n"
            "  Why: %s\n"
            "  Alternatives: %s",
            task.name, task.eval_metric,
            len(candidates), best.model_name, best.overall_rank, best.params_m,
            llm_reason[:300] if llm_reason else "highest ranked",
            ", ".join(e.model_name for e in candidates[1:4]),
        )
        logger.info(
            "Recommendation: %s (rank=%d, params=%.0fM); top-K candidates=%s",
            best.model_name, best.overall_rank, best.params_m,
            leaderboard.candidates,
        )
        return leaderboard

    def _simulate_score(self, benchmark: str) -> float:
        """Simulated benchmark score for testing."""
        base = {"linear_probe": 75.0, "knn": 70.0, "detection_ap": 35.0}
        return base.get(benchmark, 70.0) + random.uniform(-5, 5)
