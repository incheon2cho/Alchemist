"""Tests for individual agents (v2 — new roles)."""

from alchemist.agents.benchmark import BenchmarkAgent
from alchemist.agents.controller import ControllerAgent
from alchemist.agents.protocol import AgentRole, MessageType, make_directive
from alchemist.agents.research import ResearchAgent
from alchemist.core.llm import MockLLMClient
from alchemist.core.schemas import (
    ExperimentState,
    Leaderboard,
    LeaderboardEntry,
    ResearchResult,
    UserTask,
)


class TestBenchmarkAgent:
    def test_handle_directive_returns_leaderboard(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        directive = make_directive(
            to=AgentRole.BENCHMARK,
            payload={"benchmarks": ["linear_probe", "knn"]},
            episode=0, budget=100.0, trace_id="t1",
        )
        response = agent.handle_directive(directive)
        assert response.msg_type == MessageType.RESPONSE
        assert "leaderboard" in response.payload
        assert response.payload["model_count"] >= 5

    def test_scout_models_returns_known(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        models = agent.scout_models("vision encoder")
        names = [m["name"] for m in models]
        assert "dinov2_vitb14" in names
        assert "vit_s16_dino" in names

    def test_build_leaderboard_ranks_correctly(self):
        agent = BenchmarkAgent()
        models = [
            {"name": "A", "backend": "timm", "params_m": 10, "scores": {"lp": 90, "knn": 85}},
            {"name": "B", "backend": "timm", "params_m": 5, "scores": {"lp": 80, "knn": 88}},
            {"name": "C", "backend": "timm", "params_m": 8, "scores": {"lp": 70, "knn": 70}},
        ]
        lb = agent.build_leaderboard(models, ["lp", "knn"])
        assert lb.entries[0].model_name in ("A", "B")  # top ranked
        assert lb.entries[-1].model_name == "C"

    def test_recommend_respects_constraints(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_name="big", params_m=86, overall_rank=1, scores={"lp": 90}),
            LeaderboardEntry(model_name="small", params_m=22, overall_rank=2, scores={"lp": 80}),
        ], benchmarks=["lp"])
        task = UserTask(name="test", constraints={"max_params_m": 30})
        result = agent.recommend(lb, task)
        assert result.recommendation == "small"

    def test_recommend_no_constraint_picks_best(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_name="best", params_m=86, overall_rank=1),
            LeaderboardEntry(model_name="okay", params_m=22, overall_rank=2),
        ], benchmarks=["lp"])
        task = UserTask(name="test")
        result = agent.recommend(lb, task)
        assert result.recommendation == "best"


class TestResearchAgent:
    def test_handle_directive_returns_result(self):
        agent = ResearchAgent(llm=MockLLMClient(), max_trials=4)
        directive = make_directive(
            to=AgentRole.RESEARCH,
            payload={
                "base_model": "dinov2_vitb14",
                "task": {"name": "test", "description": "test task", "data_path": "/data",
                         "num_classes": 5, "eval_metric": "top1_accuracy"},
                "budget": 20.0,
            },
            episode=0, budget=100.0, trace_id="t1",
        )
        response = agent.handle_directive(directive)
        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["best_score"] > 0
        assert response.payload["trials_run"] > 0

    def test_baseline_score_exists(self):
        agent = ResearchAgent()
        score = agent.evaluate_baseline("dinov2_vitb14", UserTask(name="test"))
        assert 60 <= score <= 85

    def test_design_experiment_returns_configs(self):
        agent = ResearchAgent(llm=MockLLMClient(), max_trials=6)
        configs = agent.design_experiment("dinov2_vitb14", UserTask(name="t"))
        assert len(configs) <= 6
        assert all(hasattr(c, "lr") for c in configs)

    def test_trials_respect_budget(self):
        agent = ResearchAgent(llm=MockLLMClient(), max_trials=100)
        from alchemist.core.schemas import TrialConfig
        configs = [TrialConfig(epochs=1) for _ in range(100)]
        task = UserTask(name="t", data_path="/data")
        trials = agent.run_trials("dinov2_vitb14", task, configs, budget_hours=0.01)
        assert len(trials) < 100


class TestControllerAgent:
    def test_validate_recommendation_ok(self):
        ctrl = ControllerAgent()
        lb = Leaderboard(
            entries=[LeaderboardEntry(model_name="A", params_m=22, overall_rank=1)],
            recommendation="A",
        )
        task = UserTask(constraints={"max_params_m": 30})
        ok, reason = ctrl.validate_recommendation(lb, task)
        assert ok is True

    def test_validate_recommendation_exceeds_params(self):
        ctrl = ControllerAgent()
        lb = Leaderboard(
            entries=[LeaderboardEntry(model_name="A", params_m=86, overall_rank=1)],
            recommendation="A",
        )
        task = UserTask(constraints={"max_params_m": 30})
        ok, reason = ctrl.validate_recommendation(lb, task)
        assert ok is False

    def test_judge_result_improved(self):
        ctrl = ControllerAgent()
        result = ResearchResult(best_score=85.0, baseline_score=70.0, improvement=15.0)
        ok, reason = ctrl.judge_result(result)
        assert ok is True

    def test_judge_result_no_improvement(self):
        ctrl = ControllerAgent()
        result = ResearchResult(best_score=68.0, baseline_score=70.0, improvement=-2.0)
        ok, reason = ctrl.judge_result(result)
        assert ok is False

    def test_safety_budget_exhausted(self):
        ctrl = ControllerAgent()
        state = ExperimentState(budget_total=100, budget_used=100)
        issues = ctrl.check_safety(state)
        assert issues.get("budget_exhausted") is True

    def test_safety_budget_ok(self):
        ctrl = ControllerAgent()
        state = ExperimentState(budget_total=100, budget_used=10)
        issues = ctrl.check_safety(state)
        assert len(issues) == 0
