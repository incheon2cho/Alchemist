"""Research Agent — 사용자 태스크 기반 오토 리서치.

Responsibilities (AD-2):
- Leaderboard 최고 모델을 base로 가져오기
- 사용자 태스크 데이터로 실험 설계 (LLM)
- HP 탐색 / 아키텍처 수정 / Fine-tuning
- 외부 지식 탐색: SoTA 검색 + 최신 기법 수집 (LLM)
- SoTA 대비 gap 분석 + 자동 기법 제안
- 실험 결과 자체 분석 → 다음 실험 방향 결정 (내부 루프)
- 전 과정 연구 로그 기록
- 최종 성능 리포트 생성
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alchemist.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageType,
    make_escalation,
    make_response,
)
from alchemist.core.executor import LocalExecutor, TrainingExecutor
from alchemist.core.llm import LLMClient, MockLLMClient, safe_llm_call
from alchemist.core.schemas import (
    ResearchResult,
    TrialConfig,
    TrialResult,
    UserTask,
)
from alchemist.core.utils import safe_asdict

logger = logging.getLogger(__name__)


# Default HP search space
DEFAULT_LR_CANDIDATES = [1e-4, 2e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_BATCH_SIZES = [16, 32, 64]
DEFAULT_FREEZE_OPTIONS = [True, False]
DEFAULT_ADAPTERS = ["none", "linear_head", "lora"]

# ---------------------------------------------------------------------------
# Vision Technique Catalog — maps SoTA technique names to concrete config
# overrides that train_worker.py understands. Used by suggest_techniques()
# to bridge the gap between "LLM knows the technique name" and "executor
# needs a typed config dict".
# ---------------------------------------------------------------------------
VISION_TECHNIQUE_CATALOG: dict[str, dict] = {
    # --- Optimizers ---
    "sam":                  {"optimizer": "sam", "sam_rho": 0.05},
    "sam_conservative":     {"optimizer": "sam", "sam_rho": 0.02},
    "sam_aggressive":       {"optimizer": "sam", "sam_rho": 0.1},
    # --- Augmentation ---
    "mixup_light":          {"mixup": True, "mixup_alpha": 0.2},
    "mixup_standard":       {"mixup": True, "mixup_alpha": 0.8},
    "cutmix":               {"cutmix": True, "cutmix_alpha": 1.0},
    "cutmix_light":         {"cutmix": True, "cutmix_alpha": 0.5},
    "randaugment":          {"randaugment": True},
    "random_erasing":       {"extra": {"random_erasing": True, "random_erasing_prob": 0.25}},
    # --- Regularization ---
    "stochastic_depth":     {"extra": {"drop_path_rate": 0.1}},
    "stochastic_depth_strong": {"extra": {"drop_path_rate": 0.3}},
    "label_smoothing":      {"label_smoothing": 0.1},
    "ema":                  {"ema": True, "ema_decay": 0.9999},
    "llrd":                 {"backbone_lr_scale": 0.7},
    "weight_decay_light":   {"weight_decay": 0.01},
    "weight_decay_heavy":   {"weight_decay": 0.05},
    # --- Schedule ---
    "cosine_restarts":      {"extra": {"lr_schedule": "cosine_restarts"}},
    "onecycle":             {"extra": {"lr_schedule": "onecycle"}},
    "longer_training_30ep": {"epochs": 30, "warmup_epochs": 3},
    "longer_training_50ep": {"epochs": 50, "warmup_epochs": 5},
    # --- Architecture (universal — works on any timm model) ---
    "se_attention":         {"extra": {"add_se": True}},
    "cbam_attention":       {"extra": {"add_cbam": True}},
    "self_attention_2d":    {"extra": {"add_self_attention": True, "self_attn_heads": 4}},
    "lora_attn":            {"extra": {"add_lora": True, "lora_rank": 8, "lora_targets": "attn"}},
    "lora_qkv":             {"extra": {"add_lora": True, "lora_rank": 4, "lora_targets": "qkv"}},
    "adapter_houlsby":      {"extra": {"add_adapter": True, "adapter_bottleneck": 64}},
    # --- Video-specific techniques ---
    "video_8frames":        {"extra": {"num_frames": 8, "frame_stride": 4}},
    "video_16frames":       {"extra": {"num_frames": 16, "frame_stride": 4}},
    "video_32frames":       {"extra": {"num_frames": 32, "frame_stride": 2}},
    "video_dense_sampling": {"extra": {"num_frames": 16, "frame_stride": 2}},
    "video_sparse_sampling":{"extra": {"num_frames": 8, "frame_stride": 8}},
}


# ---------------------------------------------------------------------------
# Detection Technique Catalog — YOLO/RT-DETR object detection
# ---------------------------------------------------------------------------

DETECTION_TECHNIQUE_CATALOG: dict[str, dict] = {
    # --- Model Architecture ---
    "yolov8n":              {"base_model": "yolov8n", "batch_size": 32},
    "yolov8s":              {"base_model": "yolov8s", "batch_size": 32},
    "yolov8m":              {"base_model": "yolov8m", "batch_size": 16},
    "yolov8l":              {"base_model": "yolov8l", "batch_size": 8},
    "yolov8x":              {"base_model": "yolov8x", "batch_size": 4},
    "yolo11n":              {"base_model": "yolo11n", "batch_size": 32},
    "yolo11s":              {"base_model": "yolo11s", "batch_size": 32},
    "yolo11m":              {"base_model": "yolo11m", "batch_size": 16},
    "yolo11l":              {"base_model": "yolo11l", "batch_size": 8},
    "rtdetr_l":             {"base_model": "rtdetr-l", "batch_size": 8},
    "rtdetr_x":             {"base_model": "rtdetr-x", "batch_size": 4},

    # --- Augmentation ---
    "mosaic_on":            {"extra": {"mosaic": 1.0}},
    "mosaic_off":           {"extra": {"mosaic": 0.0}},
    "mixup_det":            {"extra": {"mixup": 0.15}},
    "mixup_det_strong":     {"extra": {"mixup": 0.3}},
    "copy_paste":           {"extra": {"copy_paste": 0.1}},
    "copy_paste_strong":    {"extra": {"copy_paste": 0.3}},
    "multi_scale":          {"extra": {"multi_scale": 0.5}},
    "random_erasing_det":   {"extra": {"erasing": 0.4}},
    "hsv_augment":          {"extra": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}},
    "geometric_augment":    {"extra": {"degrees": 10.0, "translate": 0.1, "scale": 0.5, "fliplr": 0.5}},

    # --- Resolution ---
    "resolution_640":       {"img_size": 640},
    "resolution_800":       {"img_size": 800},
    "resolution_1280":      {"img_size": 1280},

    # --- Optimizer ---
    "sgd_optimizer":        {"extra": {"optimizer": "SGD"}},
    "adamw_optimizer":      {"extra": {"optimizer": "AdamW"}},
    "lr_high":              {"lr": 0.02},
    "lr_low":               {"lr": 0.005},
    "lr_very_low":          {"lr": 0.001},
    "weight_decay_det":     {"weight_decay": 0.0005},
    "weight_decay_strong":  {"weight_decay": 0.001},
    "warmup_long":          {"extra": {"warmup_epochs": 5.0}},
    "warmup_short":         {"extra": {"warmup_epochs": 1.0}},

    # --- Loss Tuning ---
    "box_loss_high":        {"extra": {"box": 10.0}},
    "box_loss_low":         {"extra": {"box": 5.0}},
    "cls_loss_high":        {"extra": {"cls": 1.0}},
    "cls_loss_low":         {"extra": {"cls": 0.3}},
    "dfl_loss_high":        {"extra": {"dfl": 2.0}},

    # --- NMS Tuning ---
    "nms_iou_tight":        {"extra": {"iou": 0.6}},
    "nms_iou_loose":        {"extra": {"iou": 0.8}},
    "nms_conf_low":         {"extra": {"conf": 0.001}},
    "nms_conf_high":        {"extra": {"conf": 0.01}},

    # --- Training Duration ---
    "short_training":       {"epochs": 30, "extra": {"patience": 10}},
    "standard_training":    {"epochs": 50, "extra": {"patience": 10}},
    "long_training":        {"epochs": 100, "extra": {"patience": 20}},
    "very_long_training":   {"epochs": 200, "extra": {"patience": 30}},

    # --- Backbone Freeze ---
    "freeze_backbone_10":   {"extra": {"freeze": 10}},
    "freeze_backbone_15":   {"extra": {"freeze": 15}},

    # --- Close Mosaic ---
    "close_mosaic_early":   {"extra": {"close_mosaic": 5}},
    "close_mosaic_late":    {"extra": {"close_mosaic": 15}},

    # --- Cosine LR ---
    "cosine_lr":            {"extra": {"cos_lr": True}},

    # --- Mosaic-9 (small object boost) ---
    "mosaic9_small_obj":    {"extra": {"mosaic": 1.0, "copy_paste": 0.2}, "img_size": 1280},

    # --- Bayesian-tuned loss (literature optimal) ---
    "bayesian_loss_tune":   {"extra": {"box": 18.28, "cls": 1.33, "dfl": 0.56}},

    # --- Combined best practices ---
    "coco_optimal_v8m":     {"base_model": "yolov8m", "epochs": 100, "batch_size": 16,
                             "img_size": 640, "lr": 0.01,
                             "extra": {"mosaic": 0.43, "patience": 20, "cos_lr": True}},
    "coco_optimal_v11m":    {"base_model": "yolo11m", "epochs": 100, "batch_size": 16,
                             "img_size": 640, "lr": 0.01,
                             "extra": {"mosaic": 1.0, "patience": 20, "cos_lr": True}},
    "coco_optimal_rtdetr":  {"base_model": "rtdetr-l", "epochs": 100, "batch_size": 8,
                             "img_size": 640, "lr": 0.001,
                             "extra": {"optimizer": "AdamW", "patience": 20}},
}


def get_technique_catalog(task_type: str = "classification") -> dict[str, dict]:
    """Return the appropriate technique catalog based on task type.

    Prefers TaskRegistry catalog; falls back to legacy module-level dicts.
    """
    from alchemist.core.task_registry import get_task_meta_for_name
    meta = get_task_meta_for_name(task_type)
    if meta.technique_catalog:
        return meta.technique_catalog
    # Legacy fallback
    if task_type in ("detection", "coco_detection", "object_detection"):
        return DETECTION_TECHNIQUE_CATALOG
    return VISION_TECHNIQUE_CATALOG


# ---------------------------------------------------------------------------
# Self-Evolution Engine — Research Agent learns from its own experiments
# ---------------------------------------------------------------------------

class SelfEvolutionEngine:
    """Tracks technique effectiveness and evolves search strategy over trials.

    Core capabilities:
    1. Effectiveness Tracking: records score delta for each technique applied
    2. Priority Reranking: reorders technique priority based on observed impact
    3. Config Mutation: generates new configs by mutating successful ones
    4. Pattern Discovery: learns technique combinations that work well together
    5. Persistent Memory: saves evolution state to disk across sessions
    """

    def __init__(self, store_path: str | Path | None = None):
        self.store_path = Path(store_path) if store_path else Path.home() / ".cache" / "alchemist" / "evolution.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # technique_name → {total_delta, count, avg_delta, best_score}
        self.technique_scores: dict[str, dict] = {}
        # Successful config combinations
        self.winning_configs: list[dict] = []
        # Technique co-occurrence patterns: (tech_a, tech_b) → avg_delta
        self.combo_scores: dict[str, float] = {}
        # Generation counter
        self.generation: int = 0

        self._load()

    def _load(self):
        """Load evolution state from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                self.technique_scores = data.get("technique_scores", {})
                self.winning_configs = data.get("winning_configs", [])
                self.combo_scores = data.get("combo_scores", {})
                self.generation = data.get("generation", 0)
                logger.info("[evolution] Loaded: gen=%d, %d techniques tracked, %d winners",
                            self.generation, len(self.technique_scores), len(self.winning_configs))
            except Exception as e:
                logger.warning("[evolution] Load failed: %s", e)

    def save(self):
        """Persist evolution state to disk."""
        try:
            with open(self.store_path, "w") as f:
                json.dump({
                    "technique_scores": self.technique_scores,
                    "winning_configs": self.winning_configs[-20:],  # keep last 20
                    "combo_scores": self.combo_scores,
                    "generation": self.generation,
                }, f, indent=2)
            logger.info("[evolution] Saved: gen=%d", self.generation)
        except Exception as e:
            logger.warning("[evolution] Save failed: %s", e)

    def record_trial(self, config: dict, score: float, baseline: float,
                     techniques_applied: list[str] | None = None):
        """Record a trial result and update technique effectiveness scores."""
        delta = score - baseline
        if techniques_applied is None:
            techniques_applied = self._extract_techniques(config)

        for tech in techniques_applied:
            if tech not in self.technique_scores:
                self.technique_scores[tech] = {
                    "total_delta": 0.0, "count": 0,
                    "avg_delta": 0.0, "best_score": 0.0,
                }
            ts = self.technique_scores[tech]
            ts["total_delta"] += delta
            ts["count"] += 1
            ts["avg_delta"] = ts["total_delta"] / ts["count"]
            ts["best_score"] = max(ts["best_score"], score)

        # Record winning config if above baseline
        if delta > 0:
            self.winning_configs.append({
                "config": config, "score": score, "delta": delta,
                "techniques": techniques_applied, "generation": self.generation,
            })

        # Record technique combinations
        if len(techniques_applied) >= 2 and delta > 0:
            for i, t1 in enumerate(techniques_applied):
                for t2 in techniques_applied[i+1:]:
                    key = "|".join(sorted([t1, t2]))
                    prev = self.combo_scores.get(key, 0.0)
                    self.combo_scores[key] = (prev + delta) / 2  # running avg

    def get_priority_ranking(self) -> list[tuple[str, float]]:
        """Return techniques ranked by observed effectiveness (descending)."""
        ranked = sorted(
            self.technique_scores.items(),
            key=lambda x: x[1]["avg_delta"],
            reverse=True,
        )
        return [(name, stats["avg_delta"]) for name, stats in ranked]

    def get_best_combos(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Return best-performing technique combinations."""
        ranked = sorted(self.combo_scores.items(), key=lambda x: -x[1])
        return [(combo, delta) for combo, delta in ranked[:top_k]]

    def mutate_config(self, config: dict, mutation_rate: float = 0.3) -> dict:
        """Generate a new config by mutating a successful config.

        Applies random perturbations to numerical values and swaps techniques.
        """
        mutated = dict(config)
        import random

        # Mutate numerical values
        for key in ["lr", "batch_size", "epochs", "img_size", "weight_decay"]:
            if key in mutated and random.random() < mutation_rate:
                val = mutated[key]
                if isinstance(val, float):
                    factor = random.choice([0.5, 0.7, 1.0, 1.5, 2.0])
                    mutated[key] = val * factor
                elif isinstance(val, int):
                    factor = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])
                    mutated[key] = max(1, int(val * factor))

        # Swap model variant with small probability
        if "base_model" in mutated and random.random() < mutation_rate * 0.5:
            model_families = {
                "yolov8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                "yolo11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                "rtdetr": ["rtdetr-l", "rtdetr-x"],
            }
            current = mutated["base_model"]
            for family, variants in model_families.items():
                if current in variants:
                    mutated["base_model"] = random.choice(variants)
                    break

        self.generation += 1
        return mutated

    def evolve_next_configs(self, catalog: dict, n_configs: int = 4) -> list[dict]:
        """Generate evolved configs based on accumulated experience.

        Strategy:
        1. Top-ranked techniques from effectiveness tracking
        2. Best config mutations from winning configs
        3. Promising untried combinations
        """
        configs = []

        # Strategy 1: Apply top-ranked techniques to best base config
        ranking = self.get_priority_ranking()
        if self.winning_configs:
            best_winner = max(self.winning_configs, key=lambda w: w["score"])
            base = dict(best_winner["config"])

            for tech_name, avg_delta in ranking[:3]:
                if avg_delta > 0 and len(configs) < n_configs:
                    cfg = dict(base)
                    if tech_name in catalog:
                        overrides = catalog[tech_name]
                        for k, v in overrides.items():
                            if k == "extra":
                                cfg.setdefault("extra", {}).update(v)
                            else:
                                cfg[k] = v
                    configs.append(cfg)
                    logger.info("[evolution] Config from top tech '%s' (avg_delta=+%.2f)",
                                tech_name, avg_delta)

        # Strategy 2: Mutate best winning configs
        for winner in sorted(self.winning_configs, key=lambda w: -w["score"])[:2]:
            if len(configs) < n_configs:
                mutated = self.mutate_config(winner["config"])
                configs.append(mutated)
                logger.info("[evolution] Mutated config from winner (score=%.2f)",
                            winner["score"])

        # Strategy 3: Try best untried combos
        best_combos = self.get_best_combos(3)
        for combo_key, delta in best_combos:
            if len(configs) < n_configs:
                techs = combo_key.split("|")
                cfg = {}
                for tech in techs:
                    if tech in catalog:
                        for k, v in catalog[tech].items():
                            if k == "extra":
                                cfg.setdefault("extra", {}).update(v)
                            else:
                                cfg[k] = v
                configs.append(cfg)
                logger.info("[evolution] Config from combo '%s' (avg_delta=+%.2f)",
                            combo_key, delta)

        return configs[:n_configs]

    def evolve_with_external_knowledge(
        self,
        llm: Any,
        task: Any,
        current_best: float,
        catalog: dict,
        trial_history: list[dict],
    ) -> list[dict]:
        """Evolve configs by combining internal experience + external SoTA knowledge.

        Uses LLM to:
        1. Analyze what worked/failed from trial history
        2. Search external knowledge for techniques not yet tried
        3. Generate new technique configs that bridge the gap to SoTA
        4. Auto-add discovered techniques to the catalog

        Returns: list of evolved config dicts
        """
        # Build internal experience summary
        ranking = self.get_priority_ranking()
        top_techs = ", ".join(f"{n}({d:+.1f})" for n, d in ranking[:5]) if ranking else "none yet"
        failed_techs = ", ".join(f"{n}({d:+.1f})" for n, d in ranking[-3:] if d < 0) if ranking else "none"

        trial_summary = ""
        for t in trial_history[-5:]:
            trial_summary += (
                f"  - {t.get('config', {}).get('base_model', '?')}: "
                f"mAP={t.get('map50_95', t.get('score', 0)):.1f}%, "
                f"techniques={t.get('applied_techniques', {})}\n"
            )

        prompt = (
            f"You are a detection research agent analyzing experiment results.\n\n"
            f"## Task: {task.name if hasattr(task, 'name') else task}\n"
            f"## Current best: {current_best:.2f}%\n"
            f"## SoTA reference: YOLOv8x=53.9%, YOLO11x=54.7%, RT-DETRv4-X=57.0%\n\n"
            f"## Internal experience:\n"
            f"Top effective techniques: {top_techs}\n"
            f"Techniques that hurt: {failed_techs}\n\n"
            f"## Recent trials:\n{trial_summary}\n"
            f"## Available catalog techniques (not yet fully explored):\n"
            f"{', '.join(list(catalog.keys())[:30])}\n\n"
            f"## Instructions:\n"
            f"1. Analyze the gap between current best ({current_best:.1f}%) and SoTA.\n"
            f"2. Identify 3-5 specific techniques from recent papers/literature that could help:\n"
            f"   - Consider: model architecture changes, augmentation strategies,\n"
            f"     optimizer tuning, loss function adjustments, training schedule.\n"
            f"3. For EACH technique, provide a concrete config override as JSON.\n"
            f"4. Prioritize techniques NOT yet tried in our experiments.\n\n"
            f"Respond with JSON array of objects:\n"
            f"[{{"
            f'"name": "technique_name", '
            f'"reason": "why this helps", '
            f'"config": {{"key": "value", ...}}'
            f"}}]\n"
            f"Be specific with numerical values. Use ultralytics config keys."
        )

        try:
            from alchemist.core.llm import safe_llm_call
            result = safe_llm_call(
                llm, prompt,
                fallback=[{
                    "name": "coco_optimal_v11m",
                    "reason": "YOLO11m has better small-object detection than v8m",
                    "config": catalog.get("coco_optimal_v11m", {}),
                }],
            )

            configs = []
            if isinstance(result, list):
                for item in result:
                    name = item.get("name", "")
                    config = item.get("config", {})
                    reason = item.get("reason", "")

                    if config:
                        configs.append(config)
                        logger.info("[evolution-external] LLM suggested: %s — %s", name, reason)

                        # Auto-add to catalog if new
                        if name and name not in catalog:
                            catalog[name] = config
                            logger.info("[evolution-external] New technique added to catalog: %s", name)

            return configs

        except Exception as e:
            logger.warning("[evolution-external] LLM call failed: %s, using fallback", e)
            return [catalog.get("coco_optimal_v11m", {})]

    def summarize(self) -> str:
        """Human-readable evolution summary."""
        lines = [f"=== Evolution Engine (Gen {self.generation}) ==="]
        lines.append(f"Techniques tracked: {len(self.technique_scores)}")
        lines.append(f"Winning configs: {len(self.winning_configs)}")

        ranking = self.get_priority_ranking()
        if ranking:
            lines.append("\nTop techniques by effectiveness:")
            for name, delta in ranking[:10]:
                count = self.technique_scores[name]["count"]
                lines.append(f"  {name:30s}: avg_delta={delta:+.2f}  (n={count})")

        combos = self.get_best_combos(5)
        if combos:
            lines.append("\nBest combinations:")
            for combo, delta in combos:
                lines.append(f"  {combo:40s}: avg_delta={delta:+.2f}")

        return "\n".join(lines)

    @staticmethod
    def _extract_techniques(config: dict) -> list[str]:
        """Infer technique names from a config dict.

        Extracts from both top-level fields and nested extra dict,
        supporting classification and detection/seg/pose configs.
        """
        techs = []
        extra = config.get("extra", {}) or {}

        # Model identity
        if config.get("base_model"):
            techs.append(config["base_model"])

        # Detection/seg/pose specific
        if extra.get("mosaic", 0) > 0:
            techs.append("mosaic_on")
        if extra.get("mixup", 0) > 0:
            techs.append("mixup_det")
        if extra.get("copy_paste", 0) > 0:
            techs.append("copy_paste")
        if extra.get("cos_lr"):
            techs.append("cos_lr")
        if extra.get("erasing", 0) > 0:
            techs.append("random_erasing_det")
        if extra.get("multi_scale", 0) > 0:
            techs.append("multi_scale")
        if extra.get("freeze") and extra["freeze"] > 0:
            techs.append(f"freeze_backbone_{extra['freeze']}")
        for loss_key in ("box", "cls", "dfl"):
            if extra.get(loss_key):
                techs.append(f"{loss_key}_loss={extra[loss_key]}")

        # Resolution
        img_size = config.get("img_size", 0) or extra.get("img_size", 0)
        if img_size > 640:
            techs.append(f"resolution_{img_size}")

        # Training duration
        if config.get("epochs", 50) > 50:
            techs.append("long_training")

        # Classification specific
        if config.get("optimizer") == "sam":
            techs.append(f"sam_rho={config.get('sam_rho', 0.05)}")
        if config.get("ema"):
            techs.append("ema")
        if config.get("mixup") is True:
            techs.append("mixup")
        if config.get("cutmix") is True:
            techs.append("cutmix")
        if config.get("randaugment"):
            techs.append("randaugment")
        if config.get("adapter") and config["adapter"] != "none":
            techs.append(f"adapter={config['adapter']}")

        return techs if techs else ["baseline"]


class ResearchLog:
    """Research Agent의 전체 연구 과정을 기록하는 로그.

    각 라운드별 의사결정 근거, 실험 결과, 분석 내용을 추적합니다.
    """

    def __init__(self, log_dir: Path | None = None):
        self._entries: list[dict[str, Any]] = []
        self._log_dir = log_dir

    def record(
        self,
        phase: str,
        action: str,
        detail: dict[str, Any] | str,
        round_num: int = 0,
    ) -> None:
        """연구 과정의 한 단계를 기록."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": round_num,
            "phase": phase,
            "action": action,
            "detail": detail,
        }
        self._entries.append(entry)
        logger.info("[ResearchLog] R%d/%s: %s", round_num, phase, action)

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def get_summary(self) -> str:
        """사람이 읽을 수 있는 연구 로그 요약."""
        lines = []
        for e in self._entries:
            detail = e["detail"]
            if isinstance(detail, dict):
                detail_str = json.dumps(detail, ensure_ascii=False, default=str)
                if len(detail_str) > 200:
                    detail_str = detail_str[:200] + "..."
            else:
                detail_str = str(detail)[:200]
            lines.append(
                f"[R{e['round']}] {e['phase']:>12} | {e['action']}: {detail_str}"
            )
        return "\n".join(lines)

    def save_to_disk(self, filepath: Path) -> None:
        """연구 로그를 JSON 파일로 저장."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Research log saved to %s (%d entries)", filepath, len(self._entries))


class ResearchAgent:
    """Agent ② — User-Task Auto-Research Optimization.

    Takes the recommended model from Benchmark Agent and optimizes it
    for the user's specific task via HP search, architecture modification,
    and iterative self-analysis loops.

    Key capabilities:
    - External knowledge: searches SoTA results and latest techniques via LLM
    - Gap analysis: compares current results against SoTA and identifies missing techniques
    - Auto technique suggestion: LLM proposes new optimizers, augmentations, pretrained weights
    - Self-analysis loop: autonomously analyzes results and refines experiment design
    - Research logging: records every decision, analysis, and SoTA insight
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        max_trials: int = 12,
        executor: TrainingExecutor | None = None,
        max_rounds: int = 3,
        log_dir: Path | None = None,
        arxiv_retriever=None,
        enable_retrieval: bool = True,
    ):
        self.llm = llm or MockLLMClient()
        self.name = AgentRole.RESEARCH
        self.max_trials = max_trials
        self.max_rounds = max_rounds
        self.executor = executor or LocalExecutor()
        self.research_log = ResearchLog(log_dir=log_dir)
        self.enable_retrieval = enable_retrieval
        self.arxiv = arxiv_retriever
        if enable_retrieval and self.arxiv is None:
            try:
                from alchemist.core.retrievers import ArxivRetriever
                self.arxiv = ArxivRetriever()
            except Exception as e:
                logger.warning("ArxivRetriever unavailable (%s)", e)
                self.enable_retrieval = False

        # Persistent cross-task experience memory. Retrieved at the start of
        # a run to prime the LLM with prior winning configs on similar tasks;
        # appended to at the end of a run so the next task benefits.
        try:
            from alchemist.core.experience_store import VisionExperienceStore
            self.experience = VisionExperienceStore()
        except Exception as e:
            logger.warning("VisionExperienceStore unavailable (%s)", e)
            self.experience = None

    def handle_directive(
        self,
        msg: AgentMessage,
    ) -> AgentMessage:
        """Process directive: iterative experiment design → run → analyze → refine."""
        payload = msg.payload
        base_model = payload.get("base_model", "")
        task_dict = payload.get("task", {})
        budget = payload.get("budget", 20.0)
        upstream_context = payload.get("upstream_context", "")

        task = UserTask(
            name=task_dict.get("name", ""),
            description=task_dict.get("description", ""),
            data_path=task_dict.get("data_path", ""),
            num_classes=task_dict.get("num_classes", 10),
            eval_metric=task_dict.get("eval_metric", "top1_accuracy"),
            constraints=task_dict.get("constraints", {}),
        )

        # Reset log for this run
        self.research_log = ResearchLog()
        self.research_log.record("init", "directive_received", {
            "base_model": base_model,
            "task": task.name,
            "budget_hours": budget,
            "max_trials": self.max_trials,
            "max_rounds": self.max_rounds,
            "upstream_context": upstream_context[:300] if upstream_context else "",
        })

        # 1. External knowledge: search SoTA and latest techniques
        sota_knowledge = self.search_sota(task)
        self.research_log.record("sota_search", "external_knowledge", {
            "task": task.name,
            "sota_summary": sota_knowledge[:500],
        })

        # 2. Baseline evaluation
        baseline = self.evaluate_baseline(base_model, task)
        self.research_log.record("baseline", "evaluated", {
            "base_model": base_model, "score": round(baseline, 2),
        })

        # 3. Iterative research loop: design → run → analyze → evolve → refine
        all_trials: list[TrialResult] = []
        total_budget_used = 0.0
        best_score = baseline
        best_trial: TrialResult | None = None
        analysis_history: list[str] = []

        # Initialize self-evolution engine
        evolution = SelfEvolutionEngine()
        task_type = task.name.lower()
        catalog = get_technique_catalog(task_type)

        # 3. Trial-by-trial adaptive loop
        #    No round boundaries — each trial is designed based on ALL previous results.
        #    Loop continues until budget is exhausted.
        from alchemist.core.task_registry import get_task_meta_for_name as _get_meta
        _task_meta = _get_meta(task.name)
        max_total_trials = self.max_trials * self.max_rounds  # total budget

        trial_num = 0
        while trial_num < max_total_trials:
            remaining_budget = budget - total_budget_used
            if remaining_budget <= 0.5:  # 30min minimum
                self.research_log.record("loop", "budget_exhausted", {
                    "remaining_budget": round(remaining_budget, 2),
                }, trial_num)
                break

            trial_num += 1

            # ── Design next trial config (Claude-driven) ──
            evolved_context = ""
            if all_trials:
                # Evolution: learn from all previous trials
                evolved_configs = evolution.evolve_next_configs(catalog, n_configs=1)
                trial_history = [
                    {"config": safe_asdict(t.config) if hasattr(t.config, '__dataclass_fields__') else {},
                     "score": t.score, "map50_95": t.score,
                     "applied_techniques": getattr(t, "applied_techniques", {})}
                    for t in all_trials
                ]
                # LLM-driven external knowledge (every 3rd trial to save LLM calls)
                external_configs = []
                if trial_num % 3 == 0:
                    external_configs = evolution.evolve_with_external_knowledge(
                        self.llm, task, best_score, catalog, trial_history,
                    )
                evolved_context = (
                    f"\n## Previous trials ({len(all_trials)} total, best={best_score:.1f}%):\n"
                    + "\n".join(
                        f"  Trial {t.trial_id}: model={t.config.base_model}, "
                        f"img={t.config.img_size}, lr={t.config.lr}, "
                        f"ep={t.config.epochs} → {t.score:.1f}%"
                        for t in all_trials[-5:]
                    )
                    + f"\n\nEvolution gen {evolution.generation}: "
                    + f"top techs={evolution.get_priority_ranking()[:3]}\n"
                )
                self.research_log.record("evolution", "trial_evolution", {
                    "trial_num": trial_num,
                    "evolved_configs": len(evolved_configs),
                    "external_configs": len(external_configs),
                    "best_so_far": best_score,
                    "top_techniques": evolution.get_priority_ranking()[:5],
                }, trial_num)

            # Design 1 config for this trial
            configs = self.design_experiment(
                base_model, task, upstream_context + evolved_context,
                prior_analysis=analysis_history,
                remaining_trials=1,  # one trial at a time
                sota_knowledge=sota_knowledge,
                round_num=trial_num,
            )

            if not configs:
                self.research_log.record("design", "no_config", {}, trial_num)
                break

            # Adapt config based on previous results (non-classification)
            if _task_meta.task_type != "classification" and all_trials:
                trial_dicts = []
                for t in all_trials:
                    td = safe_asdict(t.config) if hasattr(t.config, '__dataclass_fields__') else {}
                    td["map50_95"] = t.score
                    td["map50"] = getattr(t, "map50", t.score * 1.3)
                    td["precision"] = getattr(t, "precision", 70.0)
                    td["recall"] = getattr(t, "recall", 55.0)
                    trial_dicts.append(td)

                adapted_configs = []
                for cfg in configs:
                    cfg_dict = safe_asdict(cfg)
                    adapted = self._adapt_detection_from_results(cfg_dict, trial_dicts)
                    tc_fields = {f.name for f in TrialConfig.__dataclass_fields__.values()}
                    adapted_cfg = TrialConfig(**{k: v for k, v in adapted.items() if k in tc_fields})
                    adapted_configs.append(adapted_cfg)
                configs = adapted_configs

                logger.info(
                    "[REASONING] Trial %d adapted from %d previous results:\n"
                    "  Best so far: %.1f%% | Last: %.1f%%\n"
                    "  Config: model=%s, img=%d, lr=%s, ep=%d",
                    trial_num, len(all_trials), best_score,
                    all_trials[-1].score,
                    configs[0].base_model, configs[0].img_size,
                    configs[0].lr, configs[0].epochs,
                )

            self.research_log.record("design", "trial_designed", {
                "trial_num": trial_num,
                "config": {"model": configs[0].base_model, "lr": configs[0].lr,
                           "epochs": configs[0].epochs, "img_size": configs[0].img_size},
            }, trial_num)

            # ── Run single trial ──
            trials = self.run_trials(
                base_model, task, configs, remaining_budget,
                baseline_score=baseline,
            )
            all_trials.extend(trials)
            trial_time = sum(t.elapsed_s for t in trials)
            total_budget_used += trial_time / 3600

            self.research_log.record("execution", "trial_completed", {
                "trial_num": trial_num,
                "score": round(trials[0].score, 2) if trials else 0,
                "elapsed_s": round(trial_time, 1),
                "budget_used_hours": round(total_budget_used, 2),
                "budget_remaining_hours": round(remaining_budget - trial_time / 3600, 2),
            }, trial_num)

            if not trials:
                self.research_log.record("execution", "trial_failed", {}, trial_num)
                continue  # try next trial instead of breaking

            # ── Update best ──
            trial_result = trials[0]
            if trial_result.score > best_score:
                best_score = trial_result.score
                best_trial = trial_result
                logger.info("[RESULT] Trial %d: NEW BEST %.1f%% (+%.1f%%p over baseline)",
                            trial_num, best_score, best_score - baseline)
            else:
                logger.info("[RESULT] Trial %d: %.1f%% (best remains %.1f%%)",
                            trial_num, trial_result.score, best_score)

            # ── Record in evolution engine ──
            for trial in trials:
                trial_config = safe_asdict(trial.config) if hasattr(trial.config, '__dataclass_fields__') else {}
                evolution.record_trial(
                    config=trial_config,
                    score=trial.score,
                    baseline=baseline,
                )
            evolution.save()
            self.research_log.record("evolution", "trials_recorded", {
                "generation": evolution.generation,
                "summary": evolution.summarize()[:500],
            }, trial_num)

            # 2c. Self-analyze results + SoTA gap analysis (Claude-driven)
            analysis = self.analyze_results(
                base_model, task, baseline, all_trials, best_score, trial_num,
            )

            gap_analysis = self.analyze_sota_gap(
                task, best_score, sota_knowledge, all_trials, trial_num,
            )
            full_analysis = analysis + "\n\n" + gap_analysis
            analysis_history.append(full_analysis)

            self.research_log.record("analysis", "self_analysis", {
                "analysis": analysis,
                "current_best": round(best_score, 2),
                "improvement": round(best_score - baseline, 2),
            }, trial_num)

            # 2d. Decide: continue or stop?
            budget_after_trial = remaining_budget - (trial_time / 3600)
            should_continue = self.should_continue_research(
                analysis, best_score, baseline, trial_num, budget_after_trial,
                max_total_trials,
            )
            self.research_log.record("decision", "continue_or_stop", {
                "continue": should_continue,
                "trial": trial_num,
                "best_score": round(best_score, 2),
            }, trial_num)

            if not should_continue:
                break

        # 3. No successful trials at all
        if not all_trials:
            self.research_log.record("final", "no_successful_trials", {})
            return make_escalation(
                self.name,
                {"reason": "no_successful_trials", "base_model": base_model},
                episode=msg.episode,
                trace_id=msg.trace_id,
            )

        # 4. Final best
        if best_trial is None:
            best_trial = max(all_trials, key=lambda t: t.score)
        improvement = best_trial.score - baseline

        # 5. Generate final report (LLM)
        report = self.generate_report(
            base_model, task, best_trial, baseline, all_trials, analysis_history,
        )
        self.research_log.record("final", "report_generated", {
            "baseline": round(baseline, 2),
            "best_score": round(best_trial.score, 2),
            "improvement": round(improvement, 2),
            "total_trials": len(all_trials),
            "total_rounds": trial_num,
        })

        # Persist experience for future runs on similar tasks.
        if self.experience is not None:
            try:
                best_cfg_d = safe_asdict(best_trial.config)
                techniques = [
                    k for k in ("mixup", "cutmix", "randaugment", "ema")
                    if best_cfg_d.get(k)
                ]
                if best_cfg_d.get("optimizer") and best_cfg_d["optimizer"] != "adamw":
                    techniques.append(f"opt={best_cfg_d['optimizer']}")
                if best_cfg_d.get("backbone_lr_scale", 1.0) < 1.0:
                    techniques.append(f"llrd={best_cfg_d['backbone_lr_scale']}")
                self.experience.record(
                    task_name=task.name,
                    task_description=task.description,
                    num_classes=task.num_classes,
                    base_model=base_model,
                    baseline_score=baseline,
                    best_score=best_trial.score,
                    best_config=best_cfg_d,
                    techniques_tried=techniques,
                    summary=(
                        (report[:300] if isinstance(report, str) else "")
                        + (
                            " WARNING: mid-training collapse detected on long-epoch "
                            "trials (>10 ep) — keep epochs<=10 or increase warmup to 15%."
                            if any(t.score < 5.0 and t.config.epochs > 10 for t in all_trials
                                   if not t.config.freeze_backbone)
                            else ""
                        )
                    ),
                    rounds_run=trial_num,
                    total_trials=len(all_trials),
                )
            except Exception as e:
                logger.warning("experience recording failed: %s", e)

        result = ResearchResult(
            base_model=base_model,
            task=task,
            best_config=safe_asdict(best_trial.config),
            best_score=best_trial.score,
            baseline_score=baseline,
            improvement=improvement,
            trials=all_trials,
            checkpoint_path=f"/checkpoints/{task.name}_{base_model}_best.pt",
            report=report,
        )

        return make_response(
            self.name,
            {
                "research_result": safe_asdict(result),
                "best_score": best_trial.score,
                "baseline_score": baseline,
                "improvement": improvement,
                "trials_run": len(all_trials),
                "rounds_run": trial_num,
                "research_log": self.research_log.get_summary(),
            },
            episode=msg.episode,
            budget=msg.budget_remaining - total_budget_used,
            trace_id=msg.trace_id,
        )

    # ------------------------------------------------------------------
    # Experiment Design
    # ------------------------------------------------------------------

    def design_experiment(
        self,
        base_model: str,
        task: UserTask,
        upstream_context: str = "",
        prior_analysis: list[str] | None = None,
        remaining_trials: int | None = None,
        round_num: int = 1,
        sota_knowledge: str = "",
    ) -> list[TrialConfig]:
        """Design experiment search space. Uses LLM + prior analysis + SoTA knowledge."""
        if remaining_trials is None:
            remaining_trials = self.max_trials

        # Build context from prior rounds
        prior_block = ""
        if prior_analysis:
            prior_block = (
                "\n\nPrior round analyses:\n"
                + "\n---\n".join(
                    f"Round {i+1}: {a}" for i, a in enumerate(prior_analysis)
                )
                + "\n\nUse the above analysis to refine your experiment design. "
                "Avoid repeating configurations that didn't work. "
                "Focus on the most promising directions identified.\n"
            )

        context_block = ""
        if upstream_context:
            context_block = (
                f"\n\nContext from Benchmark & Controller Agents:\n{upstream_context}\n"
            )

        sota_block = ""
        if sota_knowledge:
            sota_block = (
                f"\n\nSoTA Knowledge (from external search):\n{sota_knowledge[:800]}\n"
                f"Use this to identify techniques that could improve performance.\n"
            )

        prompt = (
            f"You are a Research Agent (round {round_num}). "
            f"Design hyperparameter search for:\n"
            f"  Base model: {base_model}\n"
            f"  Task: {task.description}\n"
            f"  Classes: {task.num_classes}\n"
            f"  Metric: {task.eval_metric}\n"
            f"  Constraints: {task.constraints}\n"
            f"  Remaining trial budget: {remaining_trials}\n"
            f"{context_block}{sota_block}{prior_block}"
            f"Suggest configurations to try."
        )
        safe_llm_call(self.llm, prompt, fallback={"configs": []})

        # Build search space
        configs = []

        is_large_model = any(
            base_model in name and info.get("params_m", 0) > 50
            for name, info in [
                ("dinov2_vitb14", {"params_m": 86}),
                ("vit_b16_dino", {"params_m": 86}),
                ("vit_b16_mae", {"params_m": 86}),
                ("vjepa2", {"params_m": 86}),
            ]
        )
        freeze_options = [True] if is_large_model else DEFAULT_FREEZE_OPTIONS

        if round_num == 1:
            # Round 1: broad exploration — task-type-aware.
            from alchemist.core.task_registry import get_task_meta_for_name, select_model_for_gpu
            task_meta = get_task_meta_for_name(task.name)

            if task_meta.task_type != "classification":
                # ── Non-classification: LLM-driven single trial design ──
                # Query remote GPU memory
                remote_gpu_gb = None
                if hasattr(self, 'executor') and hasattr(self.executor, 'get_remote_gpu_gb'):
                    try:
                        remote_gpu_gb = self.executor.get_remote_gpu_gb()
                    except Exception:
                        pass
                gpu_model = select_model_for_gpu(task_meta, gpu_gb=remote_gpu_gb)

                # Epoch speed estimates
                _epoch_minutes = {
                    "yolov8n": 8, "yolov8s": 10, "yolov8m": 12,
                    "yolov8l": 18, "yolov8x": 25,
                    "yolo11n": 8, "yolo11s": 10, "yolo11m": 14,
                    "yolo11l": 20, "yolo11x": 28,
                    "rtdetr-l": 90, "rtdetr-x": 135,
                }

                # Build context for LLM
                models_info = "\n".join(
                    f"  {m['name']}: {task_meta.model_ceilings.get(m['name'], '?')}% ceiling, "
                    f"{m.get('params_m', '?')}M params, ~{_epoch_minutes.get(m['name'], '?')}min/epoch"
                    for m in task_meta.known_models
                )
                prior_results = ""
                if prior_analysis:
                    prior_results = "\n".join(prior_analysis[-3:])

                # Cross-task experience
                cross_task_exp = ""
                if self.experience is not None:
                    try:
                        past = self.experience.retrieve_similar(
                            task.name, task.description, task.num_classes, top_k=3,
                        )
                        if past:
                            cross_task_exp = (
                                "\n## Cross-task experience (prior successful optimizations):\n"
                                + self.experience.summarize_for_prompt(past, max_chars=1000)
                            )
                    except Exception:
                        pass

                prompt = (
                    f"You are a Research Agent designing the next training trial to MAXIMIZE {task.eval_metric}.\n\n"
                    f"## Task\n{task.description}\n"
                    f"GPU: {remote_gpu_gb or 46}GB | Time remaining: ~{8 - (round_num * 2)}h\n"
                    f"GPU-optimal model: {gpu_model}\n\n"
                    f"## Available models & published ceilings (pretrained COCO val)\n{models_info}\n\n"
                    f"## Available training techniques\n{list(task_meta.technique_catalog.keys())[:20]}\n\n"
                    f"{upstream_context}\n"
                    f"{cross_task_exp}\n"
                    f"## Previous trial results\n{prior_results or '(First trial — no previous results)'}\n\n"
                    f"## Instructions\n"
                    f"Respond with a structured JSON containing these fields:\n"
                    f"1. **analysis**: What patterns/issues do you observe from previous results? "
                    f"(e.g., 'mAP plateaued at 48% with yolov8m, near its 50.2% ceiling')\n"
                    f"2. **diagnosis**: What is the main bottleneck limiting performance? "
                    f"(e.g., 'model capacity ceiling', 'insufficient training', 'low recall on small objects')\n"
                    f"3. **prescription**: What specific change will address this bottleneck? "
                    f"(e.g., 'upgrade to yolov8x (ceiling 53.9%) to raise capacity ceiling')\n"
                    f"4. **expected_improvement**: What {task.eval_metric} improvement do you expect and why?\n"
                    f"5. **config**: The training config dict with: base_model, epochs, batch_size, "
                    f"img_size, lr, optimizer, patience, extra (dict of augmentation/training params)\n\n"
                    f"Return ONLY JSON:\n"
                    f"{{\n"
                    f"  \"analysis\": \"...\",\n"
                    f"  \"diagnosis\": \"...\",\n"
                    f"  \"prescription\": \"...\",\n"
                    f"  \"expected_improvement\": \"...\",\n"
                    f"  \"config\": {{\"base_model\": \"...\", \"epochs\": N, \"batch_size\": N, "
                    f"\"img_size\": N, \"lr\": N, \"optimizer\": \"...\", \"patience\": N, "
                    f"\"extra\": {{\"cos_lr\": true, \"mosaic\": 1.0, ...}}}}\n"
                    f"}}"
                )

                llm_response = safe_llm_call(self.llm, prompt, fallback={})

                # Parse structured response
                if isinstance(llm_response, dict) and llm_response.get("config"):
                    analysis = llm_response.get("analysis", "")
                    diagnosis = llm_response.get("diagnosis", "")
                    prescription = llm_response.get("prescription", "")
                    expected = llm_response.get("expected_improvement", "")
                    llm_config = llm_response["config"]

                    # Auto-determine epochs if LLM didn't account for speed
                    model_name = llm_config.get("base_model", gpu_model)
                    speed = _epoch_minutes.get(model_name, 20)
                    max_epochs = max(3, int(120 / speed))  # ~2h per trial
                    if llm_config.get("epochs", 50) > max_epochs:
                        llm_config["epochs"] = max_epochs

                    tc_fields = {f.name for f in TrialConfig.__dataclass_fields__.values()}
                    cfg_dict = {k: v for k, v in llm_config.items() if k in tc_fields}
                    configs.append(TrialConfig(**cfg_dict))

                    logger.info(
                        "[REASONING] Trial %d — Claude decision:\n"
                        "  Analysis: %s\n"
                        "  Diagnosis: %s\n"
                        "  Prescription: %s\n"
                        "  Expected: %s\n"
                        "  Config: model=%s, epochs=%d, img=%d, lr=%s, batch=%d",
                        round_num,
                        analysis[:200], diagnosis[:200], prescription[:200], expected[:200],
                        model_name, llm_config.get("epochs", 0),
                        llm_config.get("img_size", 640), llm_config.get("lr", 0.01),
                        llm_config.get("batch_size", 16),
                    )

                    # Save structured reasoning to research log
                    self.research_log.record("reasoning", "trial_decision", {
                        "trial_num": round_num,
                        "analysis": analysis,
                        "diagnosis": diagnosis,
                        "prescription": prescription,
                        "expected_improvement": expected,
                        "config": llm_config,
                    }, round_num)

                elif isinstance(llm_response, dict) and llm_response.get("base_model"):
                    # Flat config without structured reasoning (backward compat)
                    llm_config = llm_response
                    model_name = llm_config.get("base_model", gpu_model)
                    speed = _epoch_minutes.get(model_name, 20)
                    max_epochs = max(3, int(120 / speed))
                    if llm_config.get("epochs", 50) > max_epochs:
                        llm_config["epochs"] = max_epochs
                    tc_fields = {f.name for f in TrialConfig.__dataclass_fields__.values()}
                    cfg_dict = {k: v for k, v in llm_config.items() if k in tc_fields}
                    configs.append(TrialConfig(**cfg_dict))
                    logger.info("[REASONING] Trial %d (flat config): model=%s, epochs=%d",
                                round_num, model_name, llm_config.get("epochs", 0))
                else:
                    # LLM fallback: use GPU-optimal model with good defaults
                    base_cfg = dict(task_meta.default_config)
                    base_cfg["base_model"] = gpu_model
                    speed = _epoch_minutes.get(gpu_model, 20)
                    base_cfg["epochs"] = max(3, int(120 / speed))
                    base_cfg["extra"] = {"cos_lr": True, "close_mosaic": 10,
                        "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.1, "erasing": 0.4}
                    tc_fields = {f.name for f in TrialConfig.__dataclass_fields__.values()}
                    configs.append(TrialConfig(**{k: v for k, v in base_cfg.items() if k in tc_fields}))
                    logger.info("[REASONING] Trial %d config (fallback): %s %d epochs",
                                round_num, gpu_model, base_cfg["epochs"])

            else:
                # ── Classification R1: v019 SAM-based sweep (unchanged) ──
                def _default_batch(params_m: float, freeze: bool) -> int:
                    if freeze:
                        return 128
                    if params_m >= 50:
                        return 16
                    return 32
                params_m = 0.0
                for name, info in [
                    ("dinov2_vitb14", {"params_m": 86}),
                    ("vit_b16_dino", {"params_m": 86}),
                    ("convnextv2_base", {"params_m": 89}),
                    ("convnext_base", {"params_m": 88}),
                    ("convnext_small", {"params_m": 50}),
                    ("maxvit_tiny", {"params_m": 29}),
                    ("efficientnet", {"params_m": 20}),
                ]:
                    if name in base_model:
                        params_m = info["params_m"]
                        break
                v019_base = dict(
                    batch_size=_default_batch(params_m, False),
                    weight_decay=0.02, scheduler="cosine", augmentation="advanced",
                    freeze_backbone=False, adapter="linear_head", optimizer="sam",
                    mixup=True, mixup_alpha=0.3, cutmix=True, cutmix_alpha=0.5,
                    randaugment=True, label_smoothing=0.1, warmup_epochs=3,
                )
                trials = [
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": False, "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.02, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.1, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 1e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 3e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.3,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                    {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.7,
                     "epochs": 30, "ema": True, "ema_decay": 0.9999,
                     "extra": {"drop_path_rate": 0.2}},
                ]
                for t in trials[:remaining_trials]:
                    configs.append(TrialConfig(**{
                        k: v for k, v in t.items()
                        if k in {f.name for f in TrialConfig.__dataclass_fields__.values()}
                    }))
        else:
            # Round 2+: LLM-driven advanced-technique refinement.
            # Use suggest_techniques() to get configs with SAM/Mixup/CutMix/EMA
            # etc. populated, based on R1 analysis + SoTA gap.
            best_so_far = 0.0
            # Try to pull best score from prior analyses text (best-effort)
            for a in (prior_analysis or []):
                import re as _re
                m = _re.search(r"(\d+\.\d+)%", a)
                if m:
                    best_so_far = max(best_so_far, float(m.group(1)))

            llm_configs = self.suggest_techniques(
                task, best_so_far, sota_knowledge or "",
                prior_analysis or [],
            )

            if isinstance(llm_configs, list) and llm_configs:
                for cfg in llm_configs[:remaining_trials]:
                    if isinstance(cfg, dict):
                        configs.append(TrialConfig(
                            lr=float(cfg.get("lr", 3e-4)),
                            batch_size=int(cfg.get("batch_size", 64)),
                            epochs=int(cfg.get("epochs", 20)),
                            weight_decay=float(cfg.get("weight_decay", 0.01)),
                            scheduler=cfg.get("scheduler", "cosine"),
                            augmentation=cfg.get("augmentation", "advanced"),
                            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
                            adapter=cfg.get("adapter", "linear_head"),
                            optimizer=cfg.get("optimizer", "adamw"),
                            mixup=bool(cfg.get("mixup", True)),
                            mixup_alpha=float(cfg.get("mixup_alpha", 0.8)),
                            cutmix=bool(cfg.get("cutmix", True)),
                            cutmix_alpha=float(cfg.get("cutmix_alpha", 1.0)),
                            randaugment=bool(cfg.get("randaugment", True)),
                            label_smoothing=float(cfg.get("label_smoothing", 0.1)),
                            ema=bool(cfg.get("ema", True)),
                            ema_decay=float(cfg.get("ema_decay", 0.9999)),
                            warmup_epochs=int(cfg.get("warmup_epochs", 5)),
                            backbone_lr_scale=float(cfg.get("backbone_lr_scale", 0.7)),
                            sam_rho=float(cfg.get("sam_rho", 0.05)),
                            extra={k: v for k, v in cfg.items()
                                   if k not in {
                                       "lr", "batch_size", "epochs", "weight_decay",
                                       "scheduler", "augmentation", "freeze_backbone",
                                       "adapter", "optimizer", "mixup", "mixup_alpha",
                                       "cutmix", "cutmix_alpha", "randaugment",
                                       "label_smoothing", "ema", "ema_decay",
                                       "warmup_epochs", "backbone_lr_scale", "sam_rho",
                                   }},
                        ))

            # Fallback: if LLM didn't produce configs, add targeted variations
            if not configs:
                for lr in [1e-4, 3e-4]:
                    for adapter in ["lora", "linear_head"]:
                        configs.append(TrialConfig(
                            lr=lr, batch_size=32, epochs=10,
                            freeze_backbone=freeze_options[0],
                            adapter=adapter,
                        ))

        # Limit to remaining budget
        if len(configs) > remaining_trials:
            random.shuffle(configs)
            configs = configs[:remaining_trials]

        logger.info(
            "Experiment design (R%d): %d trial configs for %s on '%s'",
            round_num, len(configs), base_model, task.name,
        )
        return configs

    # ------------------------------------------------------------------
    # External Knowledge: SoTA Search + Gap Analysis + Technique Suggestion
    # ------------------------------------------------------------------

    def search_sota(
        self,
        task: UserTask,
        sota_standing: dict | None = None,
    ) -> str:
        """Synthesise an SoTA techniques summary for the given task.

        This now focuses on **HOW to improve** (techniques, optimizers,
        augmentation) rather than "what's the best number" — the latter is
        provided by Benchmark Agent via ``sota_standing`` and passed in here.

        Evidence sources:
          1. ``sota_standing`` (optional) — top entries from Benchmark Agent's
             PwC leaderboard lookup, telling the LLM which models already
             achieve top scores (so it can extract their shared techniques).
          2. arXiv retrieval — recent papers (≤ 3 years) about the specific
             benchmark or about promising technique keywords.
          3. The LLM's internal knowledge — used to synthesise and extrapolate.
        """
        # 1. arXiv evidence: dataset-name + generic technique vocabulary.
        arxiv_text = "(retrieval disabled)"
        if self.enable_retrieval and self.arxiv is not None:
            try:
                # Query 1: benchmark-specific
                papers_a = self.arxiv.search(
                    query=f"{task.name} image classification",
                    years=[2023, 2024, 2025],
                    top_k=5,
                    sort_by="relevance",
                )
                # Query 2: general training-recipe techniques
                papers_b = self.arxiv.search(
                    query="sharpness-aware minimization image classification",
                    years=[2022, 2023, 2024, 2025],
                    top_k=3,
                    sort_by="relevance",
                )
                combined = papers_a + [p for p in papers_b
                                        if p["arxiv_id"] not in {q["arxiv_id"] for q in papers_a}]
                arxiv_text = self.arxiv.summarize_for_llm(combined[:8], max_chars=2500)
            except Exception as e:
                logger.warning("arxiv retrieval failed: %s", e)
                arxiv_text = f"(arxiv unavailable: {type(e).__name__})"

        # 2. SoTA standing evidence (from Benchmark Agent, optional)
        standing_text = "(not provided by Benchmark Agent)"
        if sota_standing and sota_standing.get("summary"):
            standing_text = sota_standing["summary"]

        # 3. Cross-task experience (from prior completed runs)
        experience_text = "(no prior experience on similar tasks)"
        if self.experience is not None:
            try:
                past = self.experience.retrieve_similar(
                    task.name, task.description, task.num_classes, top_k=3,
                )
                if past:
                    experience_text = self.experience.summarize_for_prompt(past, max_chars=1800)
                    logger.info("Experience retrieved: %d prior similar tasks", len(past))
            except Exception as e:
                logger.warning("experience retrieval failed: %s", e)

        prompt = (
            f"You are a computer vision research assistant. Use the evidence "
            f"below to analyse effective techniques for the task.\n\n"
            f"## Task\n"
            f"- Dataset: {task.name} ({task.num_classes} classes)\n"
            f"- Description: {task.description}\n"
            f"- Metric: {task.eval_metric}\n\n"
            f"## Current SoTA standing (from Benchmark Agent)\n"
            f"{standing_text}\n\n"
            f"## Prior experience on similar vision tasks\n"
            f"{experience_text}\n\n"
            f"## Recent arXiv papers (2023–2025)\n"
            f"{arxiv_text}\n\n"
            f"## Instructions\n"
            f"Using the evidence above plus your own knowledge, summarise:\n"
            f"1. What TECHNIQUES drive the top performance on this benchmark? "
            f"Focus on: optimizer (SGD/AdamW/SAM/LAMB), LR schedule, "
            f"augmentation (Mixup/CutMix/RandAugment), regularization "
            f"(label smoothing, stochastic depth, EMA), architecture tweaks.\n"
            f"2. Which of these techniques are still under-explored / promising "
            f"to combine for improvement above the current SoTA?\n"
            f"3. What external-data shortcuts (JFT, LAION, ImageNet-21K) should "
            f"be FLAGGED as non-allowed for fair comparison?\n\n"
            f"Be concise. Prefer specific paper references from the arXiv list."
        )
        try:
            result = self.llm.generate(prompt)
            logger.info("SoTA techniques synthesis completed for '%s'", task.name)
            return result
        except Exception as e:
            logger.warning("SoTA techniques synthesis failed: %s", e)
            return (
                f"SoTA techniques for {task.name} (LLM knowledge only): "
                f"SAM optimizer, OneCycleLR / cosine warm restarts, "
                f"Mixup + CutMix + RandAugment, label smoothing 0.1, "
                f"stochastic depth 0.1–0.3, EMA, longer training (100–300 epochs)."
            )

    def analyze_sota_gap(
        self,
        task: UserTask,
        current_best: float,
        sota_knowledge: str,
        all_trials: list[TrialResult],
        round_num: int,
    ) -> str:
        """Analyze the gap between current results and SoTA, suggest specific improvements.

        Returns actionable suggestions for the next round of experiments.
        """
        tried_techniques = set()
        for t in all_trials:
            cfg = t.config
            tried_techniques.add(f"lr={cfg.lr}")
            tried_techniques.add(f"adapter={cfg.adapter}")
            tried_techniques.add(f"freeze={cfg.freeze_backbone}")
            tried_techniques.add(f"scheduler={cfg.scheduler}")

        prompt = (
            f"You are a Research Agent analyzing the gap to SoTA.\n\n"
            f"Task: {task.description} ({task.name}, {task.num_classes} classes)\n"
            f"Current best score: {current_best:.2f}%\n"
            f"Rounds completed: {round_num}\n"
            f"Trials run: {len(all_trials)}\n\n"
            f"Techniques already tried:\n{chr(10).join(sorted(tried_techniques))}\n\n"
            f"SoTA knowledge:\n{sota_knowledge[:1000]}\n\n"
            f"Analyze:\n"
            f"1. How far is our result from SoTA? Is the gap significant?\n"
            f"2. What specific techniques are we MISSING that SoTA uses?\n"
            f"3. Rank the top 3 most impactful techniques we should try next.\n"
            f"4. For each suggested technique, provide concrete config values.\n\n"
            f"Be specific and actionable. Output format:\n"
            f"GAP: X.X%\n"
            f"MISSING TECHNIQUE 1: [name] — [config suggestion]\n"
            f"MISSING TECHNIQUE 2: [name] — [config suggestion]\n"
            f"MISSING TECHNIQUE 3: [name] — [config suggestion]"
        )
        try:
            result = self.llm.generate(prompt)
            logger.info("SoTA gap analysis (R%d): current=%.2f%%", round_num, current_best)
            return f"[SoTA Gap Analysis]\n{result}"
        except Exception:
            gap = 96.0 - current_best
            return (
                f"[SoTA Gap Analysis]\n"
                f"GAP: {gap:.1f}% (current {current_best:.1f}% vs SoTA ~96%)\n"
                f"MISSING TECHNIQUE 1: SAM optimizer — use SAM(base_optimizer=AdamW, rho=0.05)\n"
                f"MISSING TECHNIQUE 2: Larger pretrained — try ViT-L/14 or ConvNeXt-Large with IN-22K\n"
                f"MISSING TECHNIQUE 3: Longer training — increase epochs to 50-100 with cosine decay"
            )

    def suggest_techniques(
        self,
        task: UserTask,
        current_best: float,
        sota_knowledge: str,
        analysis_history: list[str],
    ) -> list[dict[str, Any]]:
        """Suggest new technique configs based on SoTA gap + catalog.

        Two-path approach:
          1. **Catalog-driven** (deterministic): identify untried techniques
             from VISION_TECHNIQUE_CATALOG and compose configs.
          2. **LLM-driven** (creative): ask LLM for novel combinations with
             available catalog technique names as a menu.

        Returns list of config dicts ready for TrialConfig parsing.
        """
        # 1. Identify which catalog techniques were already tried
        tried_techniques = set()
        if analysis_history:
            combined = " ".join(analysis_history)
            for tech_name in VISION_TECHNIQUE_CATALOG:
                if tech_name.lower() in combined.lower():
                    tried_techniques.add(tech_name)

        # Select appropriate catalog based on task type
        task_type = task.name.lower() if task else ""
        is_detection = any(k in task_type for k in ("detection", "coco", "voc", "objects365"))
        catalog = DETECTION_TECHNIQUE_CATALOG if is_detection else VISION_TECHNIQUE_CATALOG

        # Untried techniques from catalog
        for tech_name in catalog:
            if tech_name.lower() in combined.lower():
                tried_techniques.add(tech_name)

        untried = {k: v for k, v in catalog.items()
                   if k not in tried_techniques}
        logger.info(
            "Technique catalog (%s): %d total, %d tried, %d untried",
            "detection" if is_detection else "classification",
            len(catalog), len(tried_techniques), len(untried),
        )

        # 2. Compose concrete configs from untried techniques
        #    Use TaskRegistry for task-appropriate defaults
        from alchemist.core.task_registry import get_task_meta_for_name, select_model_for_gpu
        task_meta = get_task_meta_for_name(task_type)

        if task_meta.task_type != "classification":
            # Non-classification: use registry defaults + GPU-optimal model
            base_config = dict(task_meta.default_config)
            remote_gpu_gb = None
            if hasattr(self, 'executor') and hasattr(self.executor, 'get_remote_gpu_gb'):
                try:
                    remote_gpu_gb = self.executor.get_remote_gpu_gb()
                except Exception:
                    pass
            gpu_model = select_model_for_gpu(task_meta, gpu_gb=remote_gpu_gb)
            if gpu_model:
                base_config["base_model"] = gpu_model
                # Adjust batch size for larger models
                for m in task_meta.known_models:
                    if m["name"] == gpu_model and m.get("params_m", 0) > 40:
                        base_config["batch_size"] = max(4, base_config.get("batch_size", 16) // 2)
                        base_config["epochs"] = max(base_config.get("epochs", 50), 80)
                        break
            priority_techs = list(task_meta.default_priority_techs)
        else:
            base_config = {
                "lr": 1e-4, "batch_size": 32, "epochs": 20,
                "freeze_backbone": False, "adapter": "linear_head",
                "optimizer": "sam", "sam_rho": 0.05,
                "mixup": True, "mixup_alpha": 0.3,
                "cutmix": True, "cutmix_alpha": 0.5,
                "randaugment": True, "label_smoothing": 0.1,
                "ema": True, "ema_decay": 0.9999,
                "warmup_epochs": 2, "backbone_lr_scale": 0.7,
                "weight_decay": 0.02,
            }
            priority_techs = [
                "stochastic_depth", "random_erasing", "sam_aggressive",
                "cosine_restarts", "longer_training_30ep",
                "stochastic_depth_strong", "se_attention",
            ]

        configs = []
        for tech_name in priority_techs:
            if tech_name in untried and len(configs) < 5:
                cfg = {**base_config}
                overrides = untried[tech_name]
                # Merge overrides (handle nested "extra" dict)
                extra = cfg.pop("extra", {})
                tech_extra = overrides.pop("extra", {}) if "extra" in overrides else {}
                cfg.update(overrides)
                cfg["extra"] = {**extra, **tech_extra}
                cfg["_technique"] = tech_name  # tag for logging
                configs.append(cfg)
                logger.info("[catalog] proposing untried technique: %s", tech_name)

        # 3. LLM-driven creative suggestions (with catalog as menu)
        catalog_menu = ", ".join(sorted(untried.keys()))
        history_summary = "\n---\n".join(analysis_history[-2:]) if analysis_history else "None"
        prompt = (
            f"You are a Research Agent. Suggest 2-3 NEW experiment configs.\n\n"
            f"Task: {task.name}, current best: {current_best:.2f}%\n"
            f"Available untried techniques: {catalog_menu}\n\n"
            f"Recent analysis:\n{history_summary[:500]}\n\n"
            f"Return ONLY a JSON array. Each config should have: "
            f"{'base_model, img_size, lr, batch_size, epochs, optimizer, patience, techniques' if is_detection else 'lr, batch_size, epochs, freeze_backbone, adapter, optimizer, sam_rho, mixup, cutmix, techniques'} "
            f"(techniques = list of technique names from above).\n"
            f"Focus on COMBINATIONS of untried techniques."
        )
        llm_configs = safe_llm_call(self.llm, prompt, fallback=[])
        if isinstance(llm_configs, list):
            for cfg in llm_configs[:3]:
                if isinstance(cfg, dict):
                    # Apply catalog overrides for any named techniques
                    for tech_name in cfg.get("techniques", []):
                        if tech_name in VISION_TECHNIQUE_CATALOG:
                            overrides = VISION_TECHNIQUE_CATALOG[tech_name].copy()
                            extra = overrides.pop("extra", {})
                            cfg.update(overrides)
                            cfg.setdefault("extra", {}).update(extra)
                    configs.append(cfg)

        logger.info("suggest_techniques: %d configs (%d catalog + %d LLM)",
                    len(configs),
                    sum(1 for c in configs if "_technique" in c),
                    sum(1 for c in configs if "_technique" not in c))
        return configs

    # ------------------------------------------------------------------
    # Self-Analysis
    # ------------------------------------------------------------------

    def analyze_results(
        self,
        base_model: str,
        task: UserTask,
        baseline: float,
        all_trials: list[TrialResult],
        current_best: float,
        round_num: int,
    ) -> str:
        """LLM-based self-analysis of experiment results.

        Returns analysis text that informs the next round's experiment design.
        """
        # Sort trials by score for analysis
        sorted_trials = sorted(all_trials, key=lambda t: t.score, reverse=True)

        # Task-type-aware analysis dimensions
        from alchemist.core.task_registry import get_task_meta_for_name
        task_meta = get_task_meta_for_name(task.name)
        is_cls = task_meta.task_type == "classification"

        if is_cls:
            trials_detail = "\n".join(
                f"  Trial {t.trial_id}: lr={t.config.lr}, bs={t.config.batch_size}, "
                f"freeze={t.config.freeze_backbone}, adapter={t.config.adapter}, "
                f"epochs={t.config.epochs} → score={t.score:.2f}%"
                for t in sorted_trials[:10]
            )
            dim_groups: dict[str, dict] = {
                "freeze": {}, "adapter": {}, "lr": {},
            }
            for t in all_trials:
                dim_groups["freeze"].setdefault(t.config.freeze_backbone, []).append(t.score)
                dim_groups["adapter"].setdefault(t.config.adapter, []).append(t.score)
                dim_groups["lr"].setdefault(t.config.lr, []).append(t.score)
            analysis_questions = (
                "1. Which hyperparameters contributed most to performance?\n"
                "2. What patterns do you see? (e.g., lower LR is better, frozen backbone helps)\n"
                "3. Are there unexplored promising directions?\n"
            )
        else:
            # Detection/segmentation/pose: analyze model, img_size, lr, epochs
            trials_detail = "\n".join(
                f"  Trial {t.trial_id}: model={t.config.base_model}, "
                f"img_size={t.config.img_size}, lr={t.config.lr}, "
                f"bs={t.config.batch_size}, epochs={t.config.epochs} → "
                f"score={t.score:.2f}%"
                for t in sorted_trials[:10]
            )
            dim_groups = {
                "base_model": {}, "img_size": {}, "lr": {}, "epochs": {},
            }
            for t in all_trials:
                dim_groups["base_model"].setdefault(t.config.base_model, []).append(t.score)
                dim_groups["img_size"].setdefault(t.config.img_size, []).append(t.score)
                dim_groups["lr"].setdefault(t.config.lr, []).append(t.score)
                dim_groups["epochs"].setdefault(t.config.epochs, []).append(t.score)
            analysis_questions = (
                "1. Which model architecture performed best? Should we try a larger model?\n"
                "2. Does higher resolution improve mAP? Is the GPU memory trade-off worth it?\n"
                "3. Are there augmentation or loss tuning opportunities unexplored?\n"
            )

        pattern_summary = "Score patterns:\n"
        for label, group in dim_groups.items():
            for k, scores in group.items():
                avg = sum(scores) / len(scores)
                pattern_summary += f"  {label}={k}: avg={avg:.2f}%, n={len(scores)}\n"

        prompt = (
            f"You are a Research Agent analyzing experiment results (round {round_num}).\n\n"
            f"Task type: {task_meta.task_type}\n"
            f"Base model: {base_model}\n"
            f"Task: {task.description} ({task.num_classes} classes)\n"
            f"Metric: {task_meta.eval_metric}\n"
            f"Baseline: {baseline:.2f}%\n"
            f"Current best: {current_best:.2f}% (+{current_best - baseline:.2f}%)\n"
            f"Total trials so far: {len(all_trials)}\n\n"
            f"All trial results (sorted by score):\n{trials_detail}\n\n"
            f"{pattern_summary}\n"
            f"Analyze:\n"
            f"{analysis_questions}"
            f"4. What should the next round focus on?\n"
            f"5. Is there still meaningful room for improvement, or are we near optimal?\n\n"
            f"Be concise and actionable (3-5 sentences)."
        )
        try:
            analysis = self.llm.generate(prompt)
        except Exception:
            # Deterministic fallback analysis
            best = sorted_trials[0] if sorted_trials else None
            analysis = (
                f"Round {round_num}: Best score {current_best:.1f}% "
                f"(+{current_best - baseline:.1f}% over baseline). "
            )
            if best:
                analysis += (
                    f"Best config: lr={best.config.lr}, "
                    f"freeze={best.config.freeze_backbone}, "
                    f"adapter={best.config.adapter}. "
                )
            tested_adapters = set(t.config.adapter for t in all_trials)
            untested = set(DEFAULT_ADAPTERS) - tested_adapters
            if untested:
                analysis += f"Untested adapters: {untested}. "

        return analysis

    def should_continue_research(
        self,
        analysis: str,
        best_score: float,
        baseline: float,
        round_num: int,
        remaining_budget: float,
        remaining_trials: int,
    ) -> bool:
        """LLM-based decision: should we do another round of experiments?"""
        if remaining_budget <= 0 or remaining_trials <= 0:
            return False

        if round_num >= self.max_rounds:
            return False

        prompt = (
            f"You are a Research Agent deciding whether to run another experiment round.\n\n"
            f"Your analysis from this round:\n{analysis}\n\n"
            f"Current best: {best_score:.2f}% (baseline: {baseline:.2f}%)\n"
            f"Improvement so far: +{best_score - baseline:.2f}%\n"
            f"Round: {round_num}/{self.max_rounds}\n"
            f"Remaining budget: {remaining_budget:.1f} GPU-hours\n"
            f"Remaining trials: {remaining_trials}\n\n"
            f"You should continue if ANY of these are true:\n"
            f"- Your analysis identified unexplored promising directions\n"
            f"- Key hyperparameter dimensions (adapter types, freeze options, LR ranges) remain untested\n"
            f"- The improvement trend suggests further gains are achievable\n"
            f"- You have remaining budget and trials to use\n\n"
            f"You should stop only if:\n"
            f"- All major directions have been explored\n"
            f"- Scores have plateaued across diverse configurations\n"
            f"- Budget or trial count is exhausted\n\n"
            f"Respond with JSON: {{\"continue\": true/false, \"reason\": \"...\"}}"
        )
        result = safe_llm_call(
            self.llm, prompt,
            fallback={"continue": round_num < self.max_rounds, "reason": "default"},
        )

        should_continue = result.get("continue", False)
        reason = result.get("reason", "")
        logger.info(
            "Continue decision (R%d): %s — %s", round_num, should_continue, reason,
        )
        return should_continue

    # ------------------------------------------------------------------
    # Baseline / Trials / Report
    # ------------------------------------------------------------------

    def evaluate_baseline(self, base_model: str, task: UserTask) -> float:
        """Evaluate base model without any modification (baseline score)."""
        score = self.executor.evaluate_baseline(base_model, task)
        logger.info("Baseline %s on '%s': %.1f%%", base_model, task.name, score)
        return score

    def _adapt_config_from_failures(
        self,
        config: TrialConfig,
        failures: list[tuple[str, TrialConfig]],
        recent_successes: int = 0,
    ) -> TrialConfig:
        """Modify an upcoming trial's config based on prior-trial failure
        reasons. Rule-based adaptation. Environmental failures (OOM) are
        forgiven after ``recent_successes >= 2`` consecutive clean trials
        (e.g., when the OOM was caused by transient concurrent GPU use).
        """
        from dataclasses import replace
        if not failures:
            return config
        adapted = config
        # Recent-successes gate: if we've had >=2 clean trials in a row since
        # the last OOM, stop treating OOM as a persistent constraint.
        oom_active = True
        if recent_successes >= 2:
            oom_active = False
            logger.info(
                "[adapt] OOM constraint lifted after %d consecutive successes",
                recent_successes,
            )
        # Build a set of observed failure modes
        modes = {mode for mode, _ in failures}
        # Catastrophic forgetting → cap lr on unfreeze trials
        if "catastrophic" in "|".join(modes) and not adapted.freeze_backbone:
            if adapted.lr > 1e-3:
                logger.info(
                    "[adapt] lowering lr %.1e → %.1e (prior catastrophic forgetting)",
                    adapted.lr, 1e-3,
                )
                adapted = replace(adapted, lr=1e-3)
        # Optimizer divergence → reduce augmentation strength
        if "divergence" in "|".join(modes):
            if adapted.mixup and adapted.mixup_alpha > 0.4:
                adapted = replace(adapted, mixup_alpha=0.4)
            if adapted.cutmix and adapted.cutmix_alpha > 0.5:
                adapted = replace(adapted, cutmix_alpha=0.5)
            if adapted.lr > 3e-4:
                logger.info("[adapt] lowering lr → 3e-4 (prior divergence)")
                adapted = replace(adapted, lr=3e-4)
        # OOM → halve batch (only if constraint still active)
        if oom_active and "oom" in "|".join(modes).lower() and adapted.batch_size > 32:
            new_bs = max(32, adapted.batch_size // 2)
            logger.info("[adapt] batch_size %d → %d (prior OOM)",
                        adapted.batch_size, new_bs)
            adapted = replace(adapted, batch_size=new_bs)

        # Mid-training collapse → shorten epochs + increase warmup ratio +
        # ensure batch >= 64 (gradient stability)
        if "collapse" in "|".join(modes) and not adapted.freeze_backbone:
            max_safe_epochs = 10  # proven stable in R1
            if adapted.epochs > max_safe_epochs:
                logger.info(
                    "[adapt] epochs %d → %d (prior mid-training collapse)",
                    adapted.epochs, max_safe_epochs,
                )
                adapted = replace(adapted, epochs=max_safe_epochs)
            # Warmup at least 10% of epochs
            min_warmup = max(2, int(adapted.epochs * 0.15))
            if adapted.warmup_epochs < min_warmup:
                logger.info(
                    "[adapt] warmup_epochs %d → %d (15%% of epochs)",
                    adapted.warmup_epochs, min_warmup,
                )
                adapted = replace(adapted, warmup_epochs=min_warmup)
            # Batch size floor for gradient stability
            if adapted.batch_size < 64:
                logger.info("[adapt] batch_size %d → 64 (gradient stability for long FT)",
                            adapted.batch_size)
                adapted = replace(adapted, batch_size=64)
        return adapted

    def _adapt_detection_from_results(
        self,
        config: dict,
        trial_results: list[dict],
    ) -> dict:
        """Detection-specific self-refinement based on trial results.

        Analyzes mAP, precision, recall patterns and adjusts config accordingly.
        """
        if not trial_results:
            return config

        adapted = dict(config)
        last = trial_results[-1]
        best = max(trial_results, key=lambda r: r.get("map50_95", 0))

        map50_95 = last.get("map50_95", 0)
        map50 = last.get("map50", 0)
        precision = last.get("precision", 0)
        recall = last.get("recall", 0)

        logger.info("[det-refine] Last: mAP50-95=%.1f, mAP50=%.1f, P=%.1f, R=%.1f",
                    map50_95, map50, precision, recall)

        # Low recall + high precision → model is too conservative
        if recall < 60 and precision > 70:
            logger.info("[det-refine] Low recall (%.1f%%), high precision → increase recall", recall)
            adapted["extra"] = adapted.get("extra", {})
            adapted["extra"]["conf"] = 0.15  # lower confidence threshold
            adapted["extra"]["cls"] = max(0.3, adapted["extra"].get("cls", 0.5) - 0.2)
            adapted["extra"]["mixup"] = 0.2  # add diversity

        # High recall + low precision → too many false positives
        elif precision < 50 and recall > 60:
            logger.info("[det-refine] High recall (%.1f%%), low precision → reduce FP", precision)
            adapted["extra"] = adapted.get("extra", {})
            adapted["extra"]["conf"] = 0.35
            adapted["extra"]["cls"] = min(2.0, adapted["extra"].get("cls", 0.5) + 0.5)

        # mAP below model's ceiling → try larger model for higher ceiling
        # Published COCO pretrained mAP50-95 ceilings:
        #   yolov8m=50.2, yolov8l=52.9, yolov8x=53.9, rtdetr-l=53.0, rtdetr-x=54.8
        model_ceiling = {
            "yolov8n": 37.3, "yolov8s": 44.9, "yolov8m": 50.2,
            "yolov8l": 52.9, "yolov8x": 53.9,
            "yolo11m": 51.5, "yolo11l": 53.4, "yolo11x": 54.7,
            "rtdetr-l": 53.0, "rtdetr-x": 54.8,
        }
        current_model = adapted.get("base_model", "yolov8m")
        ceiling = model_ceiling.get(current_model, 50.0)
        # Upgrade if we're within 5% of model ceiling (diminishing returns)
        if map50_95 > ceiling * 0.90:
            logger.info("[det-refine] mAP %.1f%% near ceiling %.1f%% for %s → upgrade model",
                        map50_95, ceiling, current_model)
            # Try larger model
            model_upgrade = {
                "yolov8n": "yolov8s", "yolov8s": "yolov8m",
                "yolov8m": "yolov8l", "yolov8l": "yolov8x",
                "yolo11n": "yolo11s", "yolo11s": "yolo11m",
                "yolo11m": "yolo11l", "yolo11l": "yolo11x",
                # Cross-architecture: YOLO top → RT-DETR (higher mAP ceiling)
                "yolov8x": "rtdetr-l", "yolo11x": "rtdetr-l",
                # RT-DETR internal upgrades
                "rtdetr-l": "rtdetr-x",
            }
            current_model = adapted.get("base_model", "yolov8m")
            if current_model in model_upgrade:
                adapted["base_model"] = model_upgrade[current_model]
                logger.info("[det-refine] Model upgrade: %s → %s",
                            current_model, adapted["base_model"])

        # Large gap between mAP50 and mAP50-95 → poor bbox precision
        if map50 > 0 and map50_95 > 0 and (map50 - map50_95) > 20:
            logger.info("[det-refine] Large mAP50-mAP95 gap (%.1f) → improve bbox precision",
                        map50 - map50_95)
            adapted["extra"] = adapted.get("extra", {})
            adapted["extra"]["dfl"] = 2.5  # increase DFL loss
            adapted["extra"]["box"] = 10.0  # increase box loss

        # Good performance → try refinement techniques
        if map50_95 > 45:
            logger.info("[det-refine] Good mAP (%.1f%%) → try fine-tuning techniques", map50_95)
            adapted["extra"] = adapted.get("extra", {})
            adapted["extra"]["copy_paste"] = 0.1
            adapted["epochs"] = max(adapted.get("epochs", 50), 100)
            adapted["extra"]["patience"] = 20

        # Start from best config
        if best.get("map50_95", 0) > map50_95:
            best_config = best.get("config", {})
            adapted["lr"] = best_config.get("lr", adapted.get("lr", 0.01))
            adapted["batch_size"] = best_config.get("batch_size", adapted.get("batch_size", 16))
            logger.info("[det-refine] Using best trial's base config (mAP=%.1f%%)",
                        best.get("map50_95", 0))

        return adapted

    def run_trials(
        self,
        base_model: str,
        task: UserTask,
        configs: list[TrialConfig],
        budget_hours: float,
        baseline_score: float = 0.0,
    ) -> list[TrialResult]:
        """Run all trial configurations and return results."""
        trials = []
        total_time = 0.0
        best_so_far = baseline_score
        failures: list[tuple[str, TrialConfig]] = []  # for adaptive tuning
        consecutive_clean = 0  # successes in a row since last OOM

        for i, config in enumerate(configs):
            # Adapt config based on prior failures within this round
            config = self._adapt_config_from_failures(
                config, failures, recent_successes=consecutive_clean,
            )
            if total_time / 3600 > budget_hours:
                logger.warning("Budget exceeded after %d trials", i)
                break

            try:
                result = self._run_single_trial(
                    i + 1, base_model, task, config,
                    baseline_score=baseline_score,
                    best_so_far=best_so_far,
                )
                trials.append(result)
                total_time += result.elapsed_s
                if result.score > best_so_far:
                    best_so_far = result.score
                # Count clean success (no OOM) towards "OOM forgiveness".
                if result.score >= baseline_score * 0.5:
                    consecutive_clean += 1

                # Closed-loop verification: check what train_worker ACTUALLY applied
                applied = getattr(result, '_applied_techniques', None)
                if applied is None and hasattr(result, 'config'):
                    # Try to read from result dict if returned by executor
                    applied = getattr(result, 'applied_techniques', None)
                proposed_opt = config.optimizer
                actual_opt = applied.get("optimizer", "unknown") if isinstance(applied, dict) else "unknown"
                if isinstance(applied, dict) and proposed_opt != actual_opt:
                    logger.warning(
                        "[closed-loop] MISMATCH: proposed optimizer=%s but train_worker used %s",
                        proposed_opt, actual_opt,
                    )
                else:
                    logger.info(
                        "[closed-loop] verified: optimizer=%s applied correctly",
                        proposed_opt,
                    )

                logger.info(
                    "Trial %d/%d: lr=%.4f freeze=%s opt=%s → %.1f%% (%.0fs)",
                    i + 1, len(configs),
                    config.lr, config.freeze_backbone, config.optimizer,
                    result.score, result.elapsed_s,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "Trial %d OOM (freeze=%s, batch=%d) — skipping unfreeze configs",
                        i + 1, config.freeze_backbone, config.batch_size,
                    )
                    failures.append(("oom", config))
                    consecutive_clean = 0  # reset on new OOM
                    import gc
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    if not config.freeze_backbone:
                        configs = [c for c in configs[i+1:] if c.freeze_backbone]
                else:
                    logger.warning("Trial %d failed: %s", i + 1, e)
                    failures.append(("error", config))
                continue
            except Exception as e:
                logger.warning("Trial %d failed: %s", i + 1, e)
                failures.append(("error", config))
                continue

            # Classify outcome as failure/success for adaptive tuning.
            if result.score < baseline_score - 5.0:
                # Score well below baseline — treat as failure mode to adapt
                if not config.freeze_backbone and config.lr >= 1e-3:
                    failures.append(("catastrophic", config))
                elif (
                    getattr(result, "train_loss", 0.0) > 3.0
                    or result.score < baseline_score * 0.7
                ):
                    failures.append(("divergence", config))

                # Mid-training collapse detection: if a long trial (epochs>10)
                # scores < 5% (near-random), the model likely collapsed
                # after initial convergence due to LR/numerical instability.
                if config.epochs > 10 and result.score < 5.0:
                    failures.append(("collapse", config))
                    logger.warning(
                        "[adapt] mid-training collapse detected (score=%.1f%% with "
                        "%d epochs) — will shorten epochs + increase warmup ratio "
                        "for future long-training trials",
                        result.score, config.epochs,
                    )

        logger.info(
            "Completed %d/%d trials in %.0fs",
            len(trials), len(configs), total_time,
        )
        return trials

    def generate_report(
        self,
        base_model: str,
        task: UserTask,
        best_trial: TrialResult,
        baseline: float,
        all_trials: list[TrialResult],
        analysis_history: list[str] | None = None,
    ) -> str:
        """Generate natural language report including research process. (LLM)"""
        analyses_block = ""
        if analysis_history:
            analyses_block = (
                "\n\nResearch process (per-round analyses):\n"
                + "\n---\n".join(
                    f"Round {i+1}: {a}" for i, a in enumerate(analysis_history)
                )
            )

        prompt = (
            f"Generate a concise research report:\n"
            f"  Base model: {base_model}\n"
            f"  Task: {task.description}\n"
            f"  Baseline score: {baseline:.1f}%\n"
            f"  Best score: {best_trial.score:.1f}% "
            f"(+{best_trial.score - baseline:.1f}%)\n"
            f"  Best config: lr={best_trial.config.lr}, "
            f"freeze={best_trial.config.freeze_backbone}, "
            f"adapter={best_trial.config.adapter}\n"
            f"  Total trials: {len(all_trials)}\n"
            f"{analyses_block}\n\n"
            f"Include: key findings, what worked/didn't, and recommendation."
        )
        try:
            return self.llm.generate(prompt)
        except Exception:
            return (
                f"Research Report: {base_model} on {task.name}\n"
                f"Baseline: {baseline:.1f}% → Best: {best_trial.score:.1f}% "
                f"(+{best_trial.score - baseline:.1f}%)\n"
                f"Best config: lr={best_trial.config.lr}, "
                f"freeze={best_trial.config.freeze_backbone}, "
                f"adapter={best_trial.config.adapter}\n"
                f"Trials: {len(all_trials)}"
            )

    def _run_single_trial(
        self,
        trial_id: int,
        base_model: str,
        task: UserTask,
        config: TrialConfig,
        baseline_score: float = 0.0,
        best_so_far: float = 0.0,
    ) -> TrialResult:
        """Run a single trial via the configured executor with Controller-based
        mid-trial early stopping.
        """
        # Vision-aware early-stop: let Controller judge each epoch snapshot.
        controller = getattr(self, "controller", None)
        early_stop_fn = None
        if controller is not None and hasattr(controller, "evaluate_trial_progress"):
            def _fn(progress):
                return controller.evaluate_trial_progress(
                    progress, baseline_score, best_so_far,
                )
            early_stop_fn = _fn
        return self.executor.run_trial(
            trial_id, base_model, task, config, early_stop_fn=early_stop_fn,
        )
