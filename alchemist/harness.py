"""Three-Agent Harness v2 — 통합 오케스트레이터.

Pipeline:
  1. Benchmark Agent: 모델 탐색 + 순위화 → Leaderboard
  2. Research Agent: 최고 모델 기반 사용자 태스크 최적화 (자체 분석/반복) → ResearchResult
  3. Controller: 통제 + Safety + Ship 기준 판정
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from alchemist.agents.benchmark import BenchmarkAgent
from alchemist.agents.controller import ControllerAgent
from alchemist.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageBus,
    MessageType,
)
from alchemist.agents.research import ResearchAgent
from alchemist.core.executor import TrainingExecutor
from alchemist.core.llm import LLMClient, MockLLMClient
from alchemist.core.schemas import (
    ExperimentState,
    Leaderboard,
    LeaderboardEntry,
    ResearchResult,
    TrialResult,
    UserTask,
)
from alchemist.core.utils import safe_asdict

logger = logging.getLogger(__name__)


class ThreeAgentHarness:
    """Orchestrates Benchmark → Research pipeline for user tasks.

    Usage:
        harness = ThreeAgentHarness()
        result = harness.run(UserTask(
            name="pet_classification",
            description="Classify pet actions",
            data_path="/data/pets/",
            num_classes=5,
        ))
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        log_dir: Path | None = None,
        max_trials: int = 12,
        max_rounds: int = 3,
        human_gate: Callable[[str, dict], bool] | None = None,
        executor: TrainingExecutor | None = None,
    ):
        self.llm = llm or MockLLMClient()
        self.log_dir = log_dir
        self.controller = ControllerAgent(llm=self.llm)
        self.benchmark = BenchmarkAgent(llm=self.llm)
        self.research = ResearchAgent(
            llm=self.llm,
            max_trials=max_trials,
            executor=executor,
            max_rounds=max_rounds,
            log_dir=log_dir,
        )
        self.bus = MessageBus(log_dir=log_dir)
        self.human_gate = human_gate or self._default_human_gate
        self.state = ExperimentState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: UserTask,
        benchmarks: list[str] | None = None,
        budget: float = 100.0,
    ) -> ResearchResult:
        """Full pipeline: Benchmark → Research → Result."""
        trace_id = str(uuid.uuid4())[:12]
        self.state = ExperimentState(budget_total=budget, task=task)

        logger.info("=" * 60)
        logger.info("PIPELINE START: %s", task.name)
        logger.info("=" * 60)

        # Safety pre-check
        safety = self.controller.check_safety(self.state)
        if safety.get("budget_exhausted"):
            logger.error("Cannot start: budget exhausted")
            return ResearchResult(task=task, report="Budget exhausted")

        # Phase 1: Benchmark
        self.state.phase = "benchmarking"
        leaderboard = self.run_benchmark(task, benchmarks, trace_id)
        self.state.leaderboard = leaderboard

        if not leaderboard.entries:
            logger.error("No models found in benchmark")
            return ResearchResult(task=task, report="No models found")

        # Determine base model
        base_model = task.base_model or leaderboard.recommendation
        if not base_model:
            base_model = leaderboard.entries[0].model_name

        # Feed benchmark recommendation context to controller
        if leaderboard.recommendation_reason:
            self.controller._add_context(
                "benchmark_recommendation",
                f"Model: {leaderboard.recommendation} — {leaderboard.recommendation_reason[:500]}",
            )

        # Controller validates recommendation
        valid, reason = self.controller.validate_recommendation(leaderboard, task)
        logger.info("Recommendation validation: %s — %s", valid, reason)

        import timm as _timm
        known_models = set(_timm.list_models())

        def _resolve_timm_id(name: str) -> str | None:
            cand = name.replace("timm/", "", 1)
            if cand in known_models:
                return cand
            if cand.split(".")[0] in known_models:
                return cand.split(".")[0]
            return None

        # Build list of top-K timm-resolvable + compliant candidates
        candidate_names: list[str] = list(leaderboard.candidates) or ([leaderboard.recommendation] if leaderboard.recommendation else [])
        resolved_candidates: list[str] = []
        for name in candidate_names:
            tid = _resolve_timm_id(name)
            if tid and tid not in resolved_candidates:
                resolved_candidates.append(tid)
        # If validator rejected or no candidates resolved, widen the search
        if (not valid) or not resolved_candidates:
            for entry in leaderboard.entries:
                if entry.uses_additional_data:
                    continue
                tid = _resolve_timm_id(entry.model_name)
                if tid and tid not in resolved_candidates:
                    resolved_candidates.append(tid)
                if len(resolved_candidates) >= 3:
                    break
        if not resolved_candidates:
            resolved_candidates = ["convnext_base.fb_in1k"]

        # Top-K baseline evaluation: pick the best-performing model for Research Agent
        if len(resolved_candidates) > 1:
            logger.info(
                "Evaluating %d candidates via quick baseline: %s",
                len(resolved_candidates), resolved_candidates,
            )
            best_score, best_model = -1.0, resolved_candidates[0]
            for cand in resolved_candidates:
                try:
                    score = self.research.executor.evaluate_baseline(cand, task)
                    logger.info("  Candidate %s baseline: %.2f%%", cand, score)
                    if score > best_score:
                        best_score, best_model = score, cand
                except Exception as e:
                    logger.warning("  Candidate %s baseline failed: %s", cand, e)
            logger.info("Winner: %s (%.2f%%)", best_model, best_score)
            base_model = best_model
        else:
            base_model = resolved_candidates[0]

        # Phase 2: Research (Research Agent handles its own iteration loop)
        self.state.phase = "researching"
        result = self.run_research(base_model, task, trace_id)
        self.state.research_result = result

        # Controller judges result (criteria-based, not detailed analysis)
        ship_ok, ship_reason = self.controller.judge_result(result)
        logger.info("Ship judgment: %s — %s", ship_ok, ship_reason)

        if ship_ok:
            self.state.phase = "completed"
        else:
            self.state.phase = "completed"
            logger.warning("Result below expectations: %s", ship_reason)

        # Save research log to disk
        if self.log_dir:
            log_path = self.log_dir / f"research_log_{task.name}_{trace_id}.json"
            self.research.research_log.save_to_disk(log_path)

        # Record history
        self.state.history.append({
            "task": task.name,
            "base_model": base_model,
            "best_score": result.best_score,
            "baseline_score": result.baseline_score,
            "improvement": result.improvement,
            "trials": len(result.trials),
            "ship": ship_ok,
        })

        logger.info("=" * 60)
        logger.info(
            "PIPELINE DONE: %s | %s → %.1f%% (+%.1f%%)",
            task.name, base_model,
            result.best_score, result.improvement,
        )
        logger.info("=" * 60)

        return result

    def run_benchmark(
        self,
        task: UserTask | None = None,
        benchmarks: list[str] | None = None,
        trace_id: str = "",
    ) -> Leaderboard:
        """Run Benchmark Agent only: scout + measure + rank."""
        if not trace_id:
            trace_id = str(uuid.uuid4())[:12]

        logger.info("--- BENCHMARK PHASE ---")
        directive = self.controller.build_benchmark_directive(
            self.state, trace_id, benchmarks,
        )
        self.bus.send(directive)

        response = self.benchmark.handle_directive(directive, task)
        self.bus.send(response)

        if response.msg_type == MessageType.ESCALATION:
            esc_resp = self.controller.handle_escalation(response)
            self.bus.send(esc_resp)
            return Leaderboard()

        leaderboard = self._extract_leaderboard(response)

        logger.info(
            "Leaderboard: %d models, recommend=%s",
            len(leaderboard.entries), leaderboard.recommendation,
        )
        for entry in leaderboard.entries:
            scores_str = ", ".join(f"{k}={v:.1f}" for k, v in entry.scores.items())
            logger.info(
                "  #%d %s (%s, %.0fM): %s",
                entry.overall_rank, entry.model_name,
                entry.backend, entry.params_m, scores_str,
            )

        return leaderboard

    def run_research(
        self,
        base_model: str,
        task: UserTask,
        trace_id: str = "",
    ) -> ResearchResult:
        """Run Research Agent only: optimize base model for user task."""
        if not trace_id:
            trace_id = str(uuid.uuid4())[:12]

        logger.info("--- RESEARCH PHASE ---")
        logger.info("Base model: %s | Task: %s", base_model, task.name)

        directive = self.controller.build_research_directive(
            self.state, base_model, task, trace_id,
        )
        self.bus.send(directive)

        response = self.research.handle_directive(directive)
        self.bus.send(response)

        if response.msg_type == MessageType.ESCALATION:
            esc_resp = self.controller.handle_escalation(response)
            self.bus.send(esc_resp)
            return ResearchResult(base_model=base_model, task=task, report="Failed")

        result = self._extract_research_result(response)

        logger.info(
            "Research done: baseline=%.1f%% → best=%.1f%% (+%.1f%%) in %d trials",
            result.baseline_score, result.best_score,
            result.improvement, len(result.trials),
        )

        return result

    def get_leaderboard(self) -> Leaderboard:
        """Return current leaderboard."""
        return self.state.leaderboard or Leaderboard()

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return full audit log."""
        return [m.to_dict() for m in self.bus.get_log()]

    def get_research_log(self) -> list[dict[str, Any]]:
        """Return the Research Agent's detailed research log."""
        return self.research.research_log.get_entries()

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------

    def _extract_leaderboard(self, msg: AgentMessage) -> Leaderboard:
        lb_data = msg.payload.get("leaderboard", {})
        entries = []
        for e in lb_data.get("entries", []):
            entries.append(LeaderboardEntry(
                model_name=e.get("model_name", ""),
                backend=e.get("backend", ""),
                params_m=e.get("params_m", 0),
                scores=e.get("scores", {}),
                ranks=e.get("ranks", {}),
                overall_rank=e.get("overall_rank", 0),
                source=e.get("source", ""),
                uses_additional_data=bool(e.get("uses_additional_data", False)),
                paper_title=e.get("paper_title", "") or "",
                paper_date=e.get("paper_date"),
            ))
        return Leaderboard(
            entries=entries,
            benchmarks=lb_data.get("benchmarks", []),
            updated_at=lb_data.get("updated_at", ""),
            recommendation=lb_data.get("recommendation", ""),
            recommendation_reason=lb_data.get("recommendation_reason", ""),
            candidates=list(lb_data.get("candidates", [])),
        )

    def _extract_research_result(self, msg: AgentMessage) -> ResearchResult:
        rr = msg.payload.get("research_result", {})
        task_d = rr.get("task", {})
        trials_d = rr.get("trials", [])
        trials = []
        for t in trials_d:
            cfg = t.get("config", {})
            from alchemist.core.schemas import TrialConfig
            trials.append(TrialResult(
                trial_id=t.get("trial_id", 0),
                config=TrialConfig(
                    lr=cfg.get("lr", 1e-3),
                    batch_size=cfg.get("batch_size", 32),
                    epochs=cfg.get("epochs", 10),
                    freeze_backbone=cfg.get("freeze_backbone", True),
                    adapter=cfg.get("adapter", "none"),
                ),
                score=t.get("score", 0),
                elapsed_s=t.get("elapsed_s", 0),
            ))
        return ResearchResult(
            base_model=rr.get("base_model", ""),
            task=UserTask(
                name=task_d.get("name", ""),
                description=task_d.get("description", ""),
                data_path=task_d.get("data_path", ""),
                num_classes=task_d.get("num_classes", 10),
                eval_metric=task_d.get("eval_metric", ""),
            ),
            best_config=rr.get("best_config", {}),
            best_score=rr.get("best_score", 0),
            baseline_score=rr.get("baseline_score", 0),
            improvement=rr.get("improvement", 0),
            trials=trials,
            checkpoint_path=rr.get("checkpoint_path", ""),
            report=rr.get("report", ""),
        )

    @staticmethod
    def _default_human_gate(reason: str, context: dict) -> bool:
        logger.info("HUMAN GATE: %s — auto-approved", reason)
        return True
