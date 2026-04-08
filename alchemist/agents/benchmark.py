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

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or MockLLMClient()
        self.name = AgentRole.BENCHMARK

    def handle_directive(
        self,
        msg: AgentMessage,
        task: UserTask | None = None,
    ) -> AgentMessage:
        """Process directive: scout models, benchmark, rank, recommend."""
        payload = msg.payload
        benchmarks = payload.get("benchmarks", ["linear_probe", "knn"])

        # 1. Scout models
        models = self.scout_models(payload.get("search_query", "vision encoder"))

        # 2. Run benchmarks (or use published scores)
        scored_models = self.run_benchmarks(models, benchmarks)

        # 3. Build leaderboard with rankings
        leaderboard = self.build_leaderboard(scored_models, benchmarks)

        # 4. Recommend best model for user task
        if task:
            leaderboard = self.recommend(leaderboard, task)

        return make_response(
            self.name,
            {
                "leaderboard": safe_asdict(leaderboard),
                "model_count": len(leaderboard.entries),
                "benchmarks": benchmarks,
            },
            episode=msg.episode,
            budget=msg.budget_remaining,
            trace_id=msg.trace_id,
        )

    def scout_models(self, query: str = "") -> list[dict[str, Any]]:
        """Scout latest models from papers/hubs. (LLM + Web in production)"""
        # In production: arXiv search, HuggingFace API, PapersWithCode
        # For now: return known models + LLM suggestion for new ones
        models = list(KNOWN_MODELS)

        llm_result = safe_llm_call(
            self.llm,
            f"Suggest latest vision encoder models for: {query}. "
            f"Return JSON with 'new_models' list.",
            fallback={"new_models": []},
        )
        if isinstance(llm_result, dict):
            for m in llm_result.get("new_models", []):
                if isinstance(m, dict) and m.get("name"):
                    models.append(m)

        logger.info("Model Scout: found %d candidate models", len(models))
        return models

    def run_benchmarks(
        self,
        models: list[dict[str, Any]],
        benchmarks: list[str],
    ) -> list[dict[str, Any]]:
        """Run benchmarks on all models. Uses published scores or simulates."""
        results = []
        for model in models:
            name = model["name"]
            scores = {}

            # Use published scores if available
            if name in PUBLISHED_SCORES:
                published = PUBLISHED_SCORES[name]
                for bench in benchmarks:
                    if bench in published:
                        scores[bench] = published[bench]
                    else:
                        scores[bench] = self._simulate_score(bench)
            else:
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
            entries.append(LeaderboardEntry(
                model_name=model["name"],
                backend=model.get("backend", "unknown"),
                params_m=model.get("params_m", 0),
                scores=model["scores"],
                source="published" if model["name"] in PUBLISHED_SCORES else "estimated",
            ))

        # Compute per-benchmark ranks
        for bench in benchmarks:
            sorted_by_bench = sorted(
                entries,
                key=lambda e: e.scores.get(bench, 0),
                reverse=True,
            )
            for rank, entry in enumerate(sorted_by_bench, 1):
                entry.ranks[bench] = rank

        # Compute overall rank (median of per-benchmark ranks)
        for entry in entries:
            if entry.ranks:
                entry.overall_rank = int(statistics.median(entry.ranks.values()))
            else:
                entry.overall_rank = len(entries)

        entries.sort(key=lambda e: e.overall_rank)

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

        logger.info(
            "Recommendation: %s (rank=%d, params=%.0fM)",
            best.model_name, best.overall_rank, best.params_m,
        )
        return leaderboard

    def _simulate_score(self, benchmark: str) -> float:
        """Simulated benchmark score for testing."""
        base = {"linear_probe": 75.0, "knn": 70.0, "detection_ap": 35.0}
        return base.get(benchmark, 70.0) + random.uniform(-5, 5)
