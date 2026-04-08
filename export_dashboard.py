#!/usr/bin/env python3
"""Export alchemist agent results to Alchemist_Dashboard JSON format.

Runs the 3-agent pipeline and converts the output to the dashboard's
expected schema, then writes it to the dashboard's data/ directory.

Usage:
    python export_dashboard.py
    python export_dashboard.py --llm claude --llm-model sonnet
    python export_dashboard.py --output /path/to/Alchemist_Dashboard/data/alchemist_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from alchemist.agents.benchmark import KNOWN_MODELS, PUBLISHED_SCORES
from alchemist.core.llm import ClaudeCLIClient, CodexCLIClient, MockLLMClient
from alchemist.core.schemas import (
    Leaderboard,
    ResearchResult,
    TrialResult,
    UserTask,
)
from alchemist.harness import ThreeAgentHarness


def get_model_params(model_name: str) -> float:
    """Look up model parameter count from KNOWN_MODELS."""
    for m in KNOWN_MODELS:
        if m["name"] == model_name:
            return m.get("params_m", 0)
    return 0


def convert_to_dashboard_json(
    result: ResearchResult,
    leaderboard: Leaderboard,
    research_log: list[dict],
    framework_name: str = "Alchemist_v2",
    framework_desc: str = "3-agent pipeline, semi-automated assay",
    framework_version: str = "1.0",
) -> dict:
    """Convert alchemist pipeline output to dashboard JSON schema."""
    now = datetime.now(timezone.utc)
    models = []
    params_m = get_model_params(result.base_model)

    # --- v001: Baseline model ---
    baseline_scores = []
    # Add benchmark scores from leaderboard for the base model
    for entry in leaderboard.entries:
        if entry.model_name == result.base_model:
            for bench, score in entry.scores.items():
                probe_type = "knn" if bench == "knn" else "linear_probe"
                baseline_scores.append({
                    "benchmark": "cifar100",
                    "metric": "top1_accuracy",
                    "probe_type": probe_type,
                    "value": round(score / 100.0, 4),
                })
            break

    # If no leaderboard scores, use baseline_score from result
    if not baseline_scores:
        baseline_scores.append({
            "benchmark": result.task.name or "cifar100",
            "metric": result.task.eval_metric or "top1_accuracy",
            "probe_type": "linear_probe",
            "value": round(result.baseline_score / 100.0, 4),
        })

    models.append({
        "name": "v001",
        "parent": None,
        "description": f"Baseline {result.base_model} (frozen encoder, no fine-tuning)",
        "timestamp": now.isoformat(),
        "proposal_reason": f"Benchmark Agent selected {result.base_model} as top-ranked model",
        "notes": (
            f"Leaderboard rank #1. "
            f"Recommendation: {leaderboard.recommendation_reason[:200]}"
            if leaderboard.recommendation_reason else
            f"Selected as baseline: {result.base_model}"
        ),
        "build": {
            "duration_minutes": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        },
        "training": {
            "epochs": 0,
            "batch_size": 0,
            "learning_rate": 0.0,
            "optimizer": "none",
            "scheduler": "none",
            "dataset": result.task.name or "cifar100",
            "extra": {},
        },
        "scores": baseline_scores,
        "collapse": None,
        "efficiency": {"params_m": params_m, "flops_g": 0},
    })

    # --- v002+: Each trial as a model version ---
    # Extract per-round analysis from research log
    round_analyses = {}
    for entry in research_log:
        if entry.get("phase") == "analysis":
            round_analyses[entry.get("round", 0)] = entry.get("detail", {}).get("analysis", "")

    # Track which round each trial belongs to
    round_starts = {}
    trial_count = 0
    for entry in research_log:
        if entry.get("phase") == "execution":
            r = entry.get("round", 1)
            n = entry.get("detail", {}).get("num_trials", 0)
            round_starts[r] = trial_count
            trial_count += n

    for i, trial in enumerate(result.trials):
        version = f"v{i + 2:03d}"
        parent = f"v{i + 1:03d}" if i > 0 else "v001"

        # Determine which round this trial belongs to
        trial_round = 1
        running = 0
        for entry in research_log:
            if entry.get("phase") == "execution":
                r = entry.get("round", 1)
                n = entry.get("detail", {}).get("num_trials", 0)
                if running + n > i:
                    trial_round = r
                    break
                running += n

        analysis = round_analyses.get(trial_round, "")

        # Build proposal reason from config
        cfg = trial.config
        adapter_desc = cfg.adapter if cfg.adapter != "none" else "no adapter"
        freeze_desc = "frozen backbone" if cfg.freeze_backbone else "unfrozen backbone"
        proposal = f"Round {trial_round}: {adapter_desc}, {freeze_desc}, lr={cfg.lr}"

        # Score in 0-1 range
        trial_scores = [{
            "benchmark": result.task.name or "cifar100",
            "metric": result.task.eval_metric or "top1_accuracy",
            "probe_type": "linear_probe",
            "value": round(trial.score / 100.0, 4),
        }]

        extra = {
            "freeze_backbone": cfg.freeze_backbone,
            "adapter": cfg.adapter,
            "weight_decay": cfg.weight_decay,
            "augmentation": cfg.augmentation,
        }
        extra.update(cfg.extra)

        models.append({
            "name": version,
            "parent": parent,
            "description": f"Trial {trial.trial_id}: {adapter_desc}, {freeze_desc}",
            "timestamp": now.isoformat(),
            "proposal_reason": proposal,
            "notes": (
                f"Score: {trial.score:.2f}%, "
                f"train_loss={trial.train_loss:.3f}, val_loss={trial.val_loss:.3f}. "
                f"Analysis: {analysis[:200]}" if analysis else
                f"Score: {trial.score:.2f}%"
            ),
            "build": {
                "duration_minutes": round(trial.elapsed_s / 60, 2),
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            },
            "training": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.lr,
                "optimizer": "AdamW",
                "scheduler": cfg.scheduler,
                "dataset": result.task.name or "cifar100",
                "extra": extra,
            },
            "scores": trial_scores,
            "collapse": None,
            "efficiency": {"params_m": params_m, "flops_g": 0},
        })

    return {
        "framework": {
            "name": framework_name,
            "description": framework_desc,
            "agent_count": 3,
            "version": framework_version,
        },
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export alchemist results to dashboard JSON"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent.parent / "Alchemist_Dashboard" / "data" / "alchemist_v2.json"),
        help="Output JSON file path",
    )
    parser.add_argument("--llm", choices=["mock", "claude", "codex"], default="mock")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--task-name", type=str, default="cifar100")
    parser.add_argument("--task-desc", type=str, default="CIFAR-100 image classification")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--budget", type=float, default=100.0)
    args = parser.parse_args()

    # Setup LLM
    if args.llm == "claude":
        llm = ClaudeCLIClient(model=args.llm_model or "sonnet")
    elif args.llm == "codex":
        llm = CodexCLIClient(model=args.llm_model)
    else:
        llm = MockLLMClient()

    # Setup task
    task = UserTask(
        name=args.task_name,
        description=args.task_desc,
        num_classes=args.num_classes,
        eval_metric="top1_accuracy",
    )

    # Run pipeline
    log_dir = Path("./logs")
    harness = ThreeAgentHarness(
        llm=llm,
        log_dir=log_dir,
        max_trials=args.max_trials,
        max_rounds=args.max_rounds,
    )

    print("Running alchemist pipeline...")
    result = harness.run(task=task, budget=args.budget)
    leaderboard = harness.get_leaderboard()
    research_log = harness.get_research_log()

    # Convert to dashboard format
    dashboard_json = convert_to_dashboard_json(
        result=result,
        leaderboard=leaderboard,
        research_log=research_log,
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_json, f, ensure_ascii=False, indent=2)

    model_count = len(dashboard_json["models"])
    best = max(m["scores"][0]["value"] for m in dashboard_json["models"])
    print(f"\nExported to: {output_path}")
    print(f"  Models: {model_count} (1 baseline + {model_count - 1} trials)")
    print(f"  Best score: {best:.4f} ({best*100:.2f}%)")
    print(f"  Framework: {dashboard_json['framework']['name']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
