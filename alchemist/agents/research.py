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
DEFAULT_LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_BATCH_SIZES = [16, 32, 64]
DEFAULT_FREEZE_OPTIONS = [True, False]
DEFAULT_ADAPTERS = ["none", "linear_head", "lora"]


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
    ):
        self.llm = llm or MockLLMClient()
        self.name = AgentRole.RESEARCH
        self.max_trials = max_trials
        self.max_rounds = max_rounds
        self.executor = executor or LocalExecutor()
        self.research_log = ResearchLog(log_dir=log_dir)

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
            trials = self.run_trials(base_model, task, configs, remaining_budget)
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
            # Round 1: broad exploration
            for lr in DEFAULT_LR_CANDIDATES:
                for freeze in freeze_options:
                    for adapter in ["linear_head", "none"]:
                        configs.append(TrialConfig(
                            lr=lr,
                            batch_size=16 if not freeze else 32,
                            epochs=10,
                            weight_decay=0.01,
                            scheduler="cosine",
                            augmentation="basic",
                            freeze_backbone=freeze,
                            adapter=adapter,
                        ))
        else:
            # Round 2+: targeted refinement based on LLM analysis
            # Ask LLM to suggest specific configs based on prior results
            refine_prompt = (
                f"Based on the prior analysis, suggest {remaining_trials} specific "
                f"hyperparameter configurations as JSON array. Each config should have: "
                f"lr (float), batch_size (int), freeze_backbone (bool), adapter (string: "
                f"'none'/'linear_head'/'lora'), epochs (int).\n\n"
                f"Prior analyses:\n" + "\n".join(prior_analysis or []) + "\n\n"
                f"Return ONLY a JSON array of config objects."
            )
            llm_configs = safe_llm_call(
                self.llm, refine_prompt,
                fallback=[],
            )

            if isinstance(llm_configs, list) and llm_configs:
                for cfg in llm_configs[:remaining_trials]:
                    if isinstance(cfg, dict):
                        configs.append(TrialConfig(
                            lr=cfg.get("lr", 3e-4),
                            batch_size=cfg.get("batch_size", 32),
                            epochs=cfg.get("epochs", 10),
                            weight_decay=cfg.get("weight_decay", 0.01),
                            scheduler=cfg.get("scheduler", "cosine"),
                            augmentation=cfg.get("augmentation", "basic"),
                            freeze_backbone=cfg.get("freeze_backbone", True),
                            adapter=cfg.get("adapter", "linear_head"),
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

    def search_sota(self, task: UserTask) -> str:
        """Search for SoTA results and latest techniques for the given task via LLM.

        Returns a summary of SoTA knowledge that informs experiment design.
        """
        prompt = (
            f"You are a computer vision research assistant. "
            f"For the following task, provide:\n\n"
            f"Task: {task.description} (dataset: {task.name}, {task.num_classes} classes, "
            f"metric: {task.eval_metric})\n\n"
            f"1. What are the current state-of-the-art (SoTA) results on this benchmark?\n"
            f"   - Include model names, accuracy scores, and year\n"
            f"   - Separate 'with pretraining' and 'without pretraining' results\n\n"
            f"2. What techniques are most effective for this benchmark?\n"
            f"   - Optimizers (SGD, AdamW, SAM, LAMB, etc.)\n"
            f"   - LR schedules (cosine, OneCycleLR, warm restarts, etc.)\n"
            f"   - Augmentation (Mixup, CutMix, RandAugment, AugMax, etc.)\n"
            f"   - Regularization (label smoothing, dropout, stochastic depth, etc.)\n"
            f"   - Architecture tricks (SE, CBAM, attention, etc.)\n"
            f"   - Pretrained weights (ImageNet-1K vs 21K vs JFT, etc.)\n\n"
            f"3. What are the key gaps between mid-range (90%) and top (96%) performance?\n\n"
            f"Be concise and specific with numbers."
        )
        try:
            result = self.llm.generate(prompt)
            logger.info("SoTA search completed for task '%s'", task.name)
            return result
        except Exception as e:
            logger.warning("SoTA search failed: %s", e)
            return (
                f"SoTA for {task.name}: "
                f"Best with pretraining ~96% (EffNet-L2+SAM, JFT-300M). "
                f"Best without pretraining ~86% (PyramidNet+CutMix). "
                f"Key techniques: SAM optimizer, larger pretrained models, "
                f"longer training, Mixup+CutMix, label smoothing, stochastic depth."
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
        """LLM suggests new techniques to try based on SoTA gap analysis.

        Returns list of technique configs that can be applied to experiment design.
        """
        history_summary = "\n---\n".join(analysis_history[-3:]) if analysis_history else "None"

        prompt = (
            f"Based on the SoTA analysis and experiment history, suggest 3-5 specific "
            f"new configurations to try.\n\n"
            f"Task: {task.name}, current best: {current_best:.2f}%\n\n"
            f"SoTA knowledge:\n{sota_knowledge[:500]}\n\n"
            f"Recent analysis:\n{history_summary[:500]}\n\n"
            f"Return ONLY a JSON array of config objects. Each must have:\n"
            f"lr (float), batch_size (int), epochs (int), freeze_backbone (bool), "
            f"adapter (string), optimizer (string: 'adamw'/'sam'/'sgd'), "
            f"scheduler (string), mixup (bool), cutmix (bool), "
            f"extra_technique (string: description of any new technique)\n\n"
            f"Focus on techniques NOT yet tried. Be creative but practical."
        )
        result = safe_llm_call(
            self.llm, prompt,
            fallback=[
                {"lr": 0.0003, "batch_size": 64, "epochs": 50, "freeze_backbone": False,
                 "adapter": "mlp", "optimizer": "sam", "scheduler": "cosine",
                 "mixup": True, "cutmix": True, "extra_technique": "SAM optimizer"},
                {"lr": 0.0001, "batch_size": 32, "epochs": 80, "freeze_backbone": False,
                 "adapter": "mlp", "optimizer": "adamw", "scheduler": "cosine",
                 "mixup": True, "cutmix": True, "extra_technique": "longer training + lower LR"},
            ],
        )
        if isinstance(result, list):
            logger.info("LLM suggested %d new techniques", len(result))
            return result
        return []

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

    def run_trials(
        self,
        base_model: str,
        task: UserTask,
        configs: list[TrialConfig],
        budget_hours: float,
    ) -> list[TrialResult]:
        """Run all trial configurations and return results."""
        trials = []
        total_time = 0.0

        for i, config in enumerate(configs):
            if total_time / 3600 > budget_hours:
                logger.warning("Budget exceeded after %d trials", i)
                break

            try:
                result = self._run_single_trial(i + 1, base_model, task, config)
                trials.append(result)
                total_time += result.elapsed_s

                logger.info(
                    "Trial %d/%d: lr=%.4f freeze=%s adapter=%s → %.1f%% (%.0fs)",
                    i + 1, len(configs),
                    config.lr, config.freeze_backbone, config.adapter,
                    result.score, result.elapsed_s,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "Trial %d OOM (freeze=%s, batch=%d) — skipping unfreeze configs",
                        i + 1, config.freeze_backbone, config.batch_size,
                    )
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
                continue
            except Exception as e:
                logger.warning("Trial %d failed: %s", i + 1, e)
                continue

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
    ) -> TrialResult:
        """Run a single trial via the configured executor."""
        return self.executor.run_trial(trial_id, base_model, task, config)
