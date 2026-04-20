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
}


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

        # 3. Iterative research loop: design → run → analyze (with SoTA gap) → refine
        all_trials: list[TrialResult] = []
        total_budget_used = 0.0
        best_score = baseline
        best_trial: TrialResult | None = None
        analysis_history: list[str] = []

        for round_num in range(1, self.max_rounds + 1):
            remaining_budget = budget - total_budget_used

            if remaining_budget <= 0:
                self.research_log.record(
                    "loop", "budget_exhausted", {
                        "remaining_budget": round(remaining_budget, 2),
                    }, round_num,
                )
                break

            # 2a. Design experiments (informed by prior analysis)
            # Each round gets up to max_trials trials
            configs = self.design_experiment(
                base_model, task, upstream_context,
                prior_analysis=analysis_history,
                remaining_trials=self.max_trials,
                sota_knowledge=sota_knowledge,
                round_num=round_num,
            )
            self.research_log.record("design", "experiment_designed", {
                "num_configs": len(configs),
                "configs": [
                    {"lr": c.lr, "freeze": c.freeze_backbone, "adapter": c.adapter}
                    for c in configs
                ],
            }, round_num)

            if not configs:
                self.research_log.record("design", "no_configs_generated", {}, round_num)
                break

            # 2b. Run trials
            trials = self.run_trials(
                base_model, task, configs, remaining_budget,
                baseline_score=baseline,
            )
            all_trials.extend(trials)
            round_time = sum(t.elapsed_s for t in trials)
            total_budget_used += round_time / 3600

            self.research_log.record("execution", "trials_completed", {
                "num_trials": len(trials),
                "scores": [round(t.score, 2) for t in trials],
                "best_this_round": round(max(t.score for t in trials), 2) if trials else 0,
                "elapsed_s": round(round_time, 1),
                "budget_used_hours": round(total_budget_used, 2),
            }, round_num)

            if not trials:
                self.research_log.record("execution", "all_trials_failed", {}, round_num)
                break

            # Update best
            round_best = max(trials, key=lambda t: t.score)
            if round_best.score > best_score:
                best_score = round_best.score
                best_trial = round_best

            # 2c. Self-analyze results + SoTA gap analysis
            analysis = self.analyze_results(
                base_model, task, baseline, all_trials, best_score, round_num,
            )

            # 2c-1. SoTA gap analysis: compare with external knowledge
            gap_analysis = self.analyze_sota_gap(
                task, best_score, sota_knowledge, all_trials, round_num,
            )
            full_analysis = analysis + "\n\n" + gap_analysis
            analysis_history.append(full_analysis)

            self.research_log.record("analysis", "self_analysis", {
                "analysis": analysis,
                "current_best": round(best_score, 2),
                "improvement": round(best_score - baseline, 2),
            }, round_num)
            self.research_log.record("analysis", "sota_gap", {
                "gap_analysis": gap_analysis,
                "current_best": round(best_score, 2),
            }, round_num)

            # 2d. Decide: continue or stop?
            budget_after_round = remaining_budget - (round_time / 3600)
            should_continue = self.should_continue_research(
                analysis, best_score, baseline, round_num, budget_after_round,
                self.max_trials,  # each round gets full trial budget
            )
            self.research_log.record("decision", "continue_or_stop", {
                "continue": should_continue,
                "round": round_num,
                "best_score": round(best_score, 2),
            }, round_num)

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
            "total_rounds": min(round_num, self.max_rounds),
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
                    rounds_run=min(round_num, self.max_rounds),
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
                "rounds_run": min(round_num, self.max_rounds),
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
            # Round 1: broad exploration.
            # batch_size scales by model size to keep GPU saturated without OOM
            # on an A10G 24 GB. Small/mid models (< 50M) get 128; large
            # 50–100M get 64 (safer margin for activations+optimizer states).
            def _default_batch(params_m: float, freeze: bool) -> int:
                if freeze:
                    return 128  # linear probe / no grad on backbone
                if params_m >= 50:
                    return 16   # large model + SAM 2-step + EMA + 256px = tight on A10G 24GB
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
            # R1 grid informed by v019 winning config (94.0% on CIFAR-100):
            #   SAM(rho=0.05), lr=2e-4, LLRD=0.1, drop_path=0.2, 30ep,
            #   mlp head, Mixup+CutMix+RandAug, cosine schedule.
            # We sweep key axes around this anchor + EMA on/off ablation.
            v019_base = dict(
                batch_size=_default_batch(params_m, False),
                weight_decay=0.02,
                scheduler="cosine",
                augmentation="advanced",
                freeze_backbone=False,
                adapter="linear_head",
                optimizer="sam",
                mixup=True, mixup_alpha=0.3,
                cutmix=True, cutmix_alpha=0.5,
                randaugment=True, label_smoothing=0.1,
                warmup_epochs=3,
            )
            trials = [
                # --- v019 exact replica (EMA on) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                # --- v019 exact replica (EMA off) — ablation ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": False,
                 "extra": {"drop_path_rate": 0.2}},
                # --- rho sweep (LLRD=0.1 fixed, EMA on) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.02, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                {**v019_base, "lr": 2e-4, "sam_rho": 0.1, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                # --- lr sweep (rho=0.05 fixed, EMA on) ---
                {**v019_base, "lr": 1e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                {**v019_base, "lr": 3e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                # --- LLRD ablation (0.1 vs 0.3 vs 0.7, EMA on) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.3,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.7,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.2}},
                # --- drop_path ablation (0.0 vs 0.2 vs 0.3, EMA on) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.0}},
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.9999,
                 "extra": {"drop_path_rate": 0.3}},
                # --- EMA decay ablation (0.999 vs 0.9999 vs off) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 30, "ema": True, "ema_decay": 0.999,
                 "extra": {"drop_path_rate": 0.2}},
                # --- epochs ablation (20ep EMA on) ---
                {**v019_base, "lr": 2e-4, "sam_rho": 0.05, "backbone_lr_scale": 0.1,
                 "epochs": 20, "ema": True, "ema_decay": 0.9999,
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

        # Untried techniques from catalog
        untried = {k: v for k, v in VISION_TECHNIQUE_CATALOG.items()
                   if k not in tried_techniques}
        logger.info(
            "Technique catalog: %d total, %d tried, %d untried",
            len(VISION_TECHNIQUE_CATALOG), len(tried_techniques), len(untried),
        )

        # 2. Compose concrete configs from untried techniques
        # Strategy: take the best-performing base config (lr=1e-4 or 3e-4,
        # unfreeze, batch=32) and layer untried techniques on top.
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

        configs = []

        # Priority untried techniques (ordered by expected impact)
        priority_techs = [
            "stochastic_depth", "random_erasing", "sam_aggressive",
            "cosine_restarts", "longer_training_30ep",
            "stochastic_depth_strong", "se_attention",
        ]
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
            f"Return ONLY a JSON array. Each config: lr, batch_size, epochs, "
            f"freeze_backbone, adapter, optimizer ('sam'/'adamw'), sam_rho, "
            f"mixup, cutmix, techniques (list of technique names from above).\n"
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

        trials_detail = "\n".join(
            f"  Trial {t.trial_id}: lr={t.config.lr}, bs={t.config.batch_size}, "
            f"freeze={t.config.freeze_backbone}, adapter={t.config.adapter}, "
            f"epochs={t.config.epochs} → score={t.score:.2f}%"
            for t in sorted_trials[:10]
        )

        # Group by key dimensions for pattern analysis
        freeze_scores = {}
        adapter_scores = {}
        lr_scores = {}
        for t in all_trials:
            freeze_scores.setdefault(t.config.freeze_backbone, []).append(t.score)
            adapter_scores.setdefault(t.config.adapter, []).append(t.score)
            lr_scores.setdefault(t.config.lr, []).append(t.score)

        pattern_summary = "Score patterns:\n"
        for key, label in [(freeze_scores, "freeze"), (adapter_scores, "adapter"), (lr_scores, "lr")]:
            for k, scores in key.items():
                avg = sum(scores) / len(scores)
                pattern_summary += f"  {label}={k}: avg={avg:.2f}%, n={len(scores)}\n"

        prompt = (
            f"You are a Research Agent analyzing experiment results (round {round_num}).\n\n"
            f"Base model: {base_model}\n"
            f"Task: {task.description} ({task.num_classes} classes)\n"
            f"Baseline: {baseline:.2f}%\n"
            f"Current best: {current_best:.2f}% (+{current_best - baseline:.2f}%)\n"
            f"Total trials so far: {len(all_trials)}\n\n"
            f"All trial results (sorted by score):\n{trials_detail}\n\n"
            f"{pattern_summary}\n"
            f"Analyze:\n"
            f"1. Which hyperparameters contributed most to performance?\n"
            f"2. What patterns do you see? (e.g., lower LR is better, frozen backbone helps)\n"
            f"3. Are there unexplored promising directions?\n"
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
