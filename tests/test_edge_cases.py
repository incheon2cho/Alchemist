"""Edge case tests for v2 agents."""

import json
import random

from alchemist.agents.benchmark import BenchmarkAgent
from alchemist.agents.controller import ControllerAgent
from alchemist.agents.protocol import AgentRole, MessageBus, MessageType, make_directive
from alchemist.agents.research import ResearchAgent
from alchemist.core.llm import LLMClient, MockLLMClient, safe_llm_call
from alchemist.core.schemas import (
    ExperimentState,
    Leaderboard,
    LeaderboardEntry,
    ResearchResult,
    UserTask,
)
from alchemist.harness import ThreeAgentHarness


class TestLLMEdgeCases:
    def test_llm_returns_invalid_json(self):
        class BadLLM(LLMClient):
            def generate(self, prompt: str, system: str = "") -> str:
                return "not json {{{"
        result = safe_llm_call(BadLLM(), "test", fallback={"safe": True})
        assert result == {"safe": True}

    def test_llm_raises_exception(self):
        class ExplodingLLM(LLMClient):
            def generate(self, prompt: str, system: str = "") -> str:
                raise ConnectionError("API down")
        result = safe_llm_call(ExplodingLLM(), "test", fallback={"safe": True})
        assert result == {"safe": True}

    def test_mock_llm_call_count(self):
        mock = MockLLMClient()
        mock.generate("hello")
        mock.generate("world")
        assert mock.call_count == 2


class TestBenchmarkEdgeCases:
    def test_empty_benchmarks_list(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        directive = make_directive(
            to=AgentRole.BENCHMARK, payload={"benchmarks": []},
            episode=0, budget=100.0, trace_id="t",
        )
        response = agent.handle_directive(directive)
        assert response.msg_type == MessageType.RESPONSE

    def test_leaderboard_single_model(self):
        agent = BenchmarkAgent()
        models = [{"name": "solo", "backend": "timm", "params_m": 10, "scores": {"lp": 80}}]
        lb = agent.build_leaderboard(models, ["lp"])
        assert len(lb.entries) == 1
        assert lb.entries[0].overall_rank == 1

    def test_recommend_empty_leaderboard(self):
        agent = BenchmarkAgent(llm=MockLLMClient())
        lb = Leaderboard(entries=[], benchmarks=[])
        task = UserTask(name="test")
        result = agent.recommend(lb, task)
        assert result.recommendation == ""


class TestResearchEdgeCases:
    def test_zero_budget_stops_early(self):
        agent = ResearchAgent(llm=MockLLMClient(), max_trials=10)
        from alchemist.core.schemas import TrialConfig
        configs = [TrialConfig(epochs=1) for _ in range(10)]
        task = UserTask(name="t")
        trials = agent.run_trials("model", task, configs, budget_hours=0.001)
        assert len(trials) < 10

    def test_baseline_unknown_model(self):
        agent = ResearchAgent()
        score = agent.evaluate_baseline("unknown_model_xyz", UserTask(name="t"))
        assert 50 <= score <= 70  # fallback range

    def test_report_generation(self):
        agent = ResearchAgent(llm=MockLLMClient())
        from alchemist.core.schemas import TrialConfig, TrialResult
        best = TrialResult(trial_id=1, config=TrialConfig(lr=3e-4), score=85.0)
        report = agent.generate_report("model", UserTask(name="t"), best, 70.0, [best])
        assert len(report) > 0


class TestControllerEdgeCases:
    def test_validate_missing_recommendation(self):
        ctrl = ControllerAgent()
        lb = Leaderboard(entries=[LeaderboardEntry(model_name="A")], recommendation="")
        ok, reason = ctrl.validate_recommendation(lb, UserTask())
        assert ok is False

    def test_validate_recommendation_not_in_lb(self):
        ctrl = ControllerAgent()
        lb = Leaderboard(
            entries=[LeaderboardEntry(model_name="A")],
            recommendation="B",
        )
        ok, reason = ctrl.validate_recommendation(lb, UserTask())
        assert ok is False

    def test_judge_exact_equal(self):
        ctrl = ControllerAgent()
        result = ResearchResult(best_score=70.0, baseline_score=70.0, improvement=0.0)
        ok, reason = ctrl.judge_result(result)
        assert ok is False

    def test_judge_with_min_improvement(self):
        ctrl = ControllerAgent()
        result = ResearchResult(best_score=72.0, baseline_score=70.0, improvement=2.0)
        ok, reason = ctrl.judge_result(result, min_improvement=5.0)
        assert ok is False


class TestHarnessEdgeCases:
    def test_zero_budget_halts(self):
        harness = ThreeAgentHarness()
        result = harness.run(
            task=UserTask(name="test", data_path="/data"),
            budget=0.0,
        )
        assert result.report == "Budget exhausted"

    def test_deterministic_with_seed(self):
        results = []
        for _ in range(2):
            random.seed(12345)
            harness = ThreeAgentHarness(max_trials=4)
            r = harness.run(task=UserTask(name="t", data_path="/d"))
            results.append(r.best_score)
        assert results[0] == results[1]

    def test_state_updated_after_run(self):
        harness = ThreeAgentHarness(max_trials=4)
        harness.run(task=UserTask(name="test", data_path="/data"))
        assert harness.state.phase == "completed"
        assert harness.state.leaderboard is not None
        assert harness.state.research_result is not None
        assert len(harness.state.history) == 1

    def test_audit_log_json_serializable(self):
        harness = ThreeAgentHarness(max_trials=4)
        harness.run(task=UserTask(name="test", data_path="/data"))
        for entry in harness.get_audit_log():
            json.dumps(entry, ensure_ascii=False)  # Should not raise


class TestProtocol:
    def test_message_bus_basic(self):
        bus = MessageBus()
        bus.send(make_directive(
            to=AgentRole.BENCHMARK, payload={"x": 1},
            episode=0, budget=50.0, trace_id="t",
        ))
        assert len(bus.get_log()) == 1

    def test_message_roundtrip(self):
        from alchemist.agents.protocol import AgentMessage
        msg = AgentMessage(
            from_agent=AgentRole.CONTROLLER, to_agent=AgentRole.RESEARCH,
            msg_type=MessageType.DIRECTIVE, payload={"key": "val"},
        )
        d = msg.to_dict()
        restored = AgentMessage.from_dict(d)
        assert restored.payload == {"key": "val"}


class TestSchemas:
    def test_experiment_state_budget(self):
        s = ExperimentState(budget_total=100, budget_used=75)
        assert s.budget_remaining == 25.0

    def test_experiment_state_overspent(self):
        s = ExperimentState(budget_total=100, budget_used=150)
        assert s.budget_remaining == 0.0

    def test_user_task_defaults(self):
        t = UserTask()
        assert t.eval_metric == "top1_accuracy"
        assert t.num_classes == 10
