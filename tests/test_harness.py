"""Integration tests for ThreeAgentHarness v2."""

import random

from alchemist.core.schemas import UserTask
from alchemist.harness import ThreeAgentHarness


class TestHarnessFullPipeline:
    def test_run_produces_result(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=4)
        result = harness.run(
            task=UserTask(name="test_task", description="test", data_path="/data", num_classes=5),
            budget=100.0,
        )
        assert result.best_score > 0
        assert result.base_model != ""
        assert result.improvement != 0
        assert len(result.trials) > 0
        assert result.checkpoint_path != ""

    def test_run_uses_recommended_model(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=4)
        result = harness.run(
            task=UserTask(name="test", description="test", data_path="/data"),
        )
        # Should use a model from the leaderboard
        assert result.base_model in [
            "dinov2_vitb14", "dinov2_vits14", "vit_b16_dino",
            "vit_s16_dino", "vit_s16_supervised", "vitamin_s_clip", "mvitv2_tiny",
        ]

    def test_run_with_explicit_base_model(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=4)
        result = harness.run(
            task=UserTask(name="test", description="test", data_path="/data",
                          base_model="vit_s16_dino"),
        )
        assert result.base_model == "vit_s16_dino"

    def test_run_with_constraints(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=4)
        result = harness.run(
            task=UserTask(
                name="edge_task", description="edge deployment",
                data_path="/data",
                constraints={"max_params_m": 25},
            ),
        )
        # Should not pick 86M model
        assert result.base_model != ""


class TestHarnessBenchmarkOnly:
    def test_benchmark_returns_leaderboard(self):
        harness = ThreeAgentHarness()
        lb = harness.run_benchmark()
        assert len(lb.entries) >= 5
        assert lb.entries[0].overall_rank <= lb.entries[-1].overall_rank

    def test_benchmark_with_task_gives_recommendation(self):
        harness = ThreeAgentHarness()
        lb = harness.run_benchmark(task=UserTask(name="test", description="test"))
        assert lb.recommendation != ""

    def test_leaderboard_has_scores(self):
        harness = ThreeAgentHarness()
        lb = harness.run_benchmark(benchmarks=["linear_probe", "knn"])
        for entry in lb.entries:
            assert len(entry.scores) >= 2
            assert len(entry.ranks) >= 2


class TestHarnessResearchOnly:
    def test_research_returns_result(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=4)
        result = harness.run_research(
            base_model="dinov2_vitb14",
            task=UserTask(name="test", description="test", data_path="/data", num_classes=5),
        )
        assert result.best_score > result.baseline_score
        assert len(result.trials) > 0

    def test_research_improves_over_baseline(self):
        random.seed(42)
        harness = ThreeAgentHarness(max_trials=8)
        result = harness.run_research(
            base_model="dinov2_vitb14",
            task=UserTask(name="test", description="test", data_path="/data"),
        )
        assert result.improvement > 0


class TestHarnessAuditLog:
    def test_audit_log_populated(self):
        harness = ThreeAgentHarness(max_trials=4)
        harness.run(task=UserTask(name="test", data_path="/data"))
        log = harness.get_audit_log()
        assert len(log) >= 4  # At least: bench directive, bench response, research directive, research response

    def test_audit_log_has_correct_flow(self):
        harness = ThreeAgentHarness(max_trials=4)
        harness.run(task=UserTask(name="test", data_path="/data"))
        log = harness.get_audit_log()
        # First message: controller → benchmark
        assert log[0]["from_agent"] == "controller"
        assert log[0]["to_agent"] == "benchmark"

    def test_disk_logging(self, tmp_path):
        harness = ThreeAgentHarness(max_trials=4, log_dir=tmp_path / "logs")
        harness.run(task=UserTask(name="test", data_path="/data"))
        files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(files) > 0
