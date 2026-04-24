"""Controller Agent — 오케스트레이션 + 안전 통제 에이전트.

Responsibilities (AD-3):
- Pipeline Orchestration: Benchmark → Research 순차 실행
- Safety Guard: budget, 성능 하락, 실험 횟수 제한
- Registry: Leaderboard + Research 이력 관리
- DECIDE: 모델 추천 승인/거부, Ship 기준 판정 (결과 분석은 Research가 담당)
"""

from __future__ import annotations

import logging
from typing import Any

from alchemist.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageType,
    make_directive,
)
from alchemist.core.llm import LLMClient, MockLLMClient, safe_llm_call
from alchemist.core.schemas import (
    Action,
    ActionType,
    ExperimentState,
    Leaderboard,
    ResearchResult,
    UserTask,
)
from alchemist.core.utils import safe_asdict

logger = logging.getLogger(__name__)


class ControllerAgent:
    """Agent ③ — Pipeline Orchestration + Safety + Ship 판정.

    Controller는 실험 결과의 세부 분석은 하지 않습니다.
    Research Agent가 자체 분석/개선을 담당하고,
    Controller는 예산/안전/품질 기준에 따른 최종 ship 판정만 수행합니다.
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or MockLLMClient()
        self.name = AgentRole.CONTROLLER
        self._context: list[dict[str, str]] = []

    def _add_context(self, role: str, content: str) -> None:
        """Add a message to the controller's conversation context."""
        self._context.append({"role": role, "content": content})
        if len(self._context) > 20:
            self._context = self._context[-20:]

    # ------------------------------------------------------------------
    # Directive builders
    # ------------------------------------------------------------------

    def build_benchmark_directive(
        self,
        state: ExperimentState,
        trace_id: str,
        benchmarks: list[str] | None = None,
    ) -> AgentMessage:
        """Build directive for Benchmark Agent."""
        return make_directive(
            to=AgentRole.BENCHMARK,
            payload={
                "benchmarks": benchmarks or ["linear_probe", "knn"],
                "search_query": "latest vision encoder",
            },
            episode=0,
            budget=state.budget_remaining,
            trace_id=trace_id,
        )

    def build_research_directive(
        self,
        state: ExperimentState,
        base_model: str,
        task: UserTask,
        trace_id: str,
        budget: float | None = None,
    ) -> AgentMessage:
        """Build directive for Research Agent, including upstream context."""
        return make_directive(
            to=AgentRole.RESEARCH,
            payload={
                "base_model": base_model,
                "task": safe_asdict(task),
                "budget": budget or state.budget_remaining * 0.8,
                "upstream_context": self.get_research_context(),
            },
            episode=0,
            budget=state.budget_remaining,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def check_safety(self, state: ExperimentState) -> dict[str, Any]:
        """Run all safety checks."""
        issues: dict[str, Any] = {}

        if state.budget_remaining <= 0:
            issues["budget_exhausted"] = True
        elif state.budget_remaining < state.budget_total * 0.2:
            issues["budget_low"] = state.budget_remaining

        return issues

    def validate_recommendation(
        self,
        leaderboard: Leaderboard,
        task: UserTask,
    ) -> tuple[bool, str]:
        """Validate Benchmark Agent's model recommendation.

        Checks hard constraints + LLM-based suitability analysis.
        """
        if not leaderboard.recommendation:
            return False, "No model recommended"

        rec_entry = None
        for entry in leaderboard.entries:
            if entry.model_name == leaderboard.recommendation:
                rec_entry = entry
                break

        if not rec_entry:
            return False, f"Recommended model {leaderboard.recommendation} not in leaderboard"

        # Hard constraint check
        max_params = task.constraints.get("max_params_m")
        if max_params and rec_entry.params_m > max_params:
            return False, (
                f"{rec_entry.model_name} has {rec_entry.params_m}M params, "
                f"exceeds constraint {max_params}M"
            )

        # LLM-based suitability check
        lb_summary = "\n".join(
            f"  #{e.overall_rank} {e.model_name} ({e.params_m:.0f}M): "
            + ", ".join(f"{k}={v:.1f}" for k, v in e.scores.items())
            for e in leaderboard.entries[:5]
        )
        prompt = (
            f"You are the Controller Agent reviewing a model recommendation.\n\n"
            f"Task: {task.description} ({task.num_classes} classes, metric={task.eval_metric})\n"
            f"Constraints: {task.constraints}\n\n"
            f"Leaderboard (top 5):\n{lb_summary}\n\n"
            f"Recommendation: {leaderboard.recommendation}\n"
            f"Reason: {leaderboard.recommendation_reason[:200]}\n\n"
            f"Is this model appropriate for the task and constraints?\n"
            f"Respond with JSON: {{\"approved\": true/false, \"reason\": \"...\"}}"
        )
        llm_result = safe_llm_call(
            self.llm, prompt,
            fallback={"approved": True, "reason": f"Approved: {rec_entry.model_name}"},
        )

        approved = llm_result.get("approved", True)
        reason = llm_result.get("reason",
                                f"Approved: {rec_entry.model_name} (rank={rec_entry.overall_rank})")

        self._add_context("benchmark_result", lb_summary)
        self._add_context("controller_validation", f"approved={approved}, reason={reason}")

        return approved, reason

    def get_research_context(self) -> str:
        """Return accumulated context for the Research Agent."""
        return "\n".join(f"[{c['role']}] {c['content']}" for c in self._context)

    # ------------------------------------------------------------------
    # Mid-trial guardrail — Vision-aware early-stop decision
    # ------------------------------------------------------------------

    def evaluate_trial_progress(
        self,
        progress: dict[str, Any],
        baseline_score: float,
        best_so_far: float,
    ) -> tuple[bool, str]:
        """Decide whether a mid-training trial should continue.

        Applies four vision-aware heuristics on the latest epoch snapshot:
          (1) **Catastrophic forgetting**: val_acc < baseline - 10 %p at epoch ≥ 3
              (typical sign of too-high lr on a pretrained backbone).
          (2) **Hopeless vs best-so-far**: at epoch ≥ 5, val_acc + 5 %p < best_so_far
              with val_acc already > 0 — remaining epochs can't close the gap.
          (3) **Training divergence**: train_loss > 3.0 at epoch ≥ 3 (for a
              cross-entropy classifier) — the optimizer is unstable.
          (4) **Plateau below threshold**: val_acc hasn't improved for 3 epochs
              AND val_acc < 0.9 × best_so_far (not implemented here;
              requires caller to track history).

        Returns ``(keep, reason)``.
        """
        epoch = int(progress.get("epoch", 0))
        # Use task-appropriate metric: val_acc for classification, mAP for detection, etc.
        val = float(
            progress.get("val_acc", 0.0)
            or progress.get("map50_95", 0.0)
            or progress.get("mAP", 0.0)
            or progress.get("mIoU", 0.0)
            or progress.get("score", 0.0)
        )
        train_loss = float(progress.get("train_loss", 0.0))
        total = int(progress.get("total_epochs", 10))

        # Rule (1): catastrophic forgetting — only after warmup has ended.
        if epoch >= 3 and baseline_score > 0 and val < baseline_score - 10.0:
            return False, (
                f"epoch {epoch}/{total}: metric={val:.1f}% is > 10 %p below "
                f"baseline {baseline_score:.1f}% — catastrophic forgetting"
            )

        # Rule (2): hopeless — can't close gap to best_so_far in remaining epochs.
        if best_so_far > 0 and epoch >= 5 and val > 0 and val + 5.0 < best_so_far:
            return False, (
                f"epoch {epoch}/{total}: metric={val:.1f}% still > 5 %p below "
                f"best_so_far {best_so_far:.1f}% — hopeless"
            )

        # Rule (3): optimizer divergence.
        if epoch >= 3 and train_loss > 3.0:
            return False, (
                f"epoch {epoch}/{total}: train_loss={train_loss:.2f} "
                f"> 3.0 — optimizer divergence"
            )

        logger.info(
            "[REASONING] Trial progress evaluation:\n"
            "  Epoch %d/%d, metric=%.1f%%, train_loss=%.2f\n"
            "  Baseline: %.1f%%, Best so far: %.1f%%\n"
            "  Decision: CONTINUE (no early-stop triggers met)",
            epoch, total, val, train_loss, baseline_score, best_so_far,
        )
        return True, f"continue (epoch {epoch}/{total}, metric={val:.1f}%)"

    # ------------------------------------------------------------------
    # Ship judgment (기준 기반, 세부 분석은 Research가 담당)
    # ------------------------------------------------------------------

    def judge_result(
        self,
        result: ResearchResult,
        min_improvement: float = 0.0,
    ) -> tuple[bool, str]:
        """Judge whether research result meets ship criteria.

        Controller는 품질 기준만 확인합니다:
        - 성능 개선이 있는가?
        - 최소 개선 기준을 충족하는가?
        - 예산 내에서 완료되었는가?

        실험 세부 분석(어떤 HP가 좋았는지 등)은 Research Agent가 자체적으로 수행합니다.
        """
        if result.best_score <= result.baseline_score:
            return False, (
                f"No improvement: best={result.best_score:.1f}% "
                f"<= baseline={result.baseline_score:.1f}%"
            )

        if result.improvement < min_improvement:
            return False, (
                f"Improvement {result.improvement:.1f}% "
                f"below minimum {min_improvement:.1f}%"
            )

        return True, (
            f"Approved: {result.baseline_score:.1f}% → {result.best_score:.1f}% "
            f"(+{result.improvement:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Escalation handling
    # ------------------------------------------------------------------

    def handle_escalation(self, msg: AgentMessage) -> AgentMessage:
        """Handle escalation from sub-agents."""
        reason = msg.payload.get("reason", "unknown")
        logger.warning("ESCALATION from %s: %s", msg.from_agent.value, reason)

        response_payload: dict[str, Any] = {"escalation_handled": True}

        if reason == "no_successful_trials":
            response_payload["action"] = "retry_with_simpler_config"
        else:
            response_payload["action"] = "log_and_halt"

        return make_directive(
            to=msg.from_agent,
            payload=response_payload,
            episode=msg.episode,
            budget=msg.budget_remaining,
            trace_id=msg.trace_id,
        )
