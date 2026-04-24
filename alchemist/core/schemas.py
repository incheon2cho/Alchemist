"""Core schemas for Alchemist 3-Agent Harness v2.

Agent roles:
  - Benchmark Agent: model scouting + benchmarking + ranking
  - Research Agent: user-task-specific auto-research optimization
  - Controller Agent: orchestration + safety + registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# User Task — Research Agent의 입력
# ---------------------------------------------------------------------------

@dataclass
class UserTask:
    """User-provided task definition for Research Agent."""
    name: str = ""                        # "pet_action_recognition"
    description: str = ""                 # "실내 반려동물 행동 인식"
    data_path: str = ""                   # "/data/pet_actions/"
    num_classes: int = 10
    eval_metric: str = "top1_accuracy"    # 평가 메트릭
    constraints: dict[str, Any] = field(default_factory=dict)  # {"max_params_m": 10}
    base_model: str | None = None         # None이면 Benchmark Agent 추천 사용


# ---------------------------------------------------------------------------
# Leaderboard — Benchmark Agent의 산출물
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    model_name: str = ""
    backend: str = ""                     # "timm" | "huggingface" | "vjepa2"
    params_m: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)   # benchmark → score
    ranks: dict[str, int] = field(default_factory=dict)      # benchmark → rank
    overall_rank: int = 0                 # median rank across benchmarks
    source: str = "measured"              # "measured" | "published" | "pwc"
    uses_additional_data: bool = False    # True = pretrained on corpora beyond ImageNet-1K
    paper_title: str = ""
    paper_date: str | None = None


@dataclass
class Leaderboard:
    entries: list[LeaderboardEntry] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    updated_at: str = ""
    recommendation: str = ""              # 추천 모델 이름 (top-1)
    recommendation_reason: str = ""
    candidates: list[str] = field(default_factory=list)  # top-K compliant timm-resolvable IDs


# ---------------------------------------------------------------------------
# Trial / Research Result — Research Agent의 산출물
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    augmentation: str = "basic"
    freeze_backbone: bool = True
    adapter: str = "none"                 # "none" | "lora" | "linear_head"
    # Advanced techniques — propagated to train_worker.py
    optimizer: str = "adamw"              # "adamw" | "sgd" | "sam"
    mixup: bool = False
    mixup_alpha: float = 0.2
    cutmix: bool = False
    cutmix_alpha: float = 1.0
    randaugment: bool = False
    label_smoothing: float = 0.0
    ema: bool = False
    ema_decay: float = 0.9999
    warmup_epochs: int = 0
    backbone_lr_scale: float = 1.0        # LLRD (layer-wise LR decay)
    sam_rho: float = 0.05
    # Universal fields (used by detection/segmentation/pose workers)
    img_size: int = 224
    patience: int = 10
    base_model: str = ""                  # model name for detection/seg/pose
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_id: int = 0
    config: TrialConfig = field(default_factory=TrialConfig)
    score: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class ResearchResult:
    base_model: str = ""
    task: UserTask = field(default_factory=UserTask)
    best_config: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0              # best - baseline (%)
    trials: list[TrialResult] = field(default_factory=list)
    checkpoint_path: str = ""
    report: str = ""


# ---------------------------------------------------------------------------
# Experiment State
# ---------------------------------------------------------------------------

@dataclass
class ExperimentState:
    phase: Literal["idle", "benchmarking", "researching", "completed", "halted"] = "idle"
    budget_total: float = 100.0           # GPU-hours
    budget_used: float = 0.0
    leaderboard: Leaderboard | None = None
    research_result: ResearchResult | None = None
    task: UserTask | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_total - self.budget_used)


# ---------------------------------------------------------------------------
# Legacy compatibility (used by protocol/controller)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    BENCHMARK = "benchmark"
    RESEARCH = "research"
    SHIP = "ship"
    HALT = "halt"


@dataclass
class Action:
    type: ActionType
    target: str = ""
    reason: str = ""
    priority: int = 0
