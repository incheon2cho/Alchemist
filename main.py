#!/usr/bin/env python3
"""Alchemist 3-Agent Harness v2 — CLI entry point.

Usage:
    python main.py run --task-name "분류" --task-desc "설명" --data-path /data/
    python main.py benchmark
    python main.py research --base-model dinov2_vitb14 --data-path /data/
    python main.py leaderboard
    python main.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alchemist.core.executor import AWSExecutor, LocalExecutor
from alchemist.core.llm import ClaudeCLIClient, CodexCLIClient, MockLLMClient
from alchemist.core.schemas import UserTask
from alchemist.harness import ThreeAgentHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alchemist 3-Agent Closed-Loop Research Harness"
    )
    parser.add_argument(
        "command",
        choices=["run", "benchmark", "research", "leaderboard", "status"],
        default="run",
        nargs="?",
    )
    parser.add_argument("--task-name", type=str, default="classification")
    parser.add_argument("--task-desc", type=str, default="Image classification task")
    parser.add_argument("--data-path", type=str, default="/data/task/")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--eval-metric", type=str, default="top1_accuracy")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max research iteration rounds (default: 3)")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument(
        "--llm",
        type=str,
        choices=["mock", "claude", "codex"],
        default="mock",
        help="LLM backend: mock (default), claude (Claude CLI), codex (Codex CLI)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name for CLI backend (e.g. sonnet, opus for claude / gpt-5.4 for codex)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for CLI LLM calls (default: 120)",
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=["local", "aws"],
        default="local",
        help="Training executor: local (simulated) or aws (remote GPU)",
    )
    parser.add_argument(
        "--aws-host",
        type=str,
        default=None,
        help="AWS GPU instance SSH host (e.g. ubuntu@ec2-xx.compute.amazonaws.com)",
    )
    parser.add_argument(
        "--aws-key",
        type=str,
        default=None,
        help="Path to SSH private key for AWS instance",
    )
    parser.add_argument(
        "--aws-work-dir",
        type=str,
        default="/home/ubuntu/alchemist",
        help="Remote working directory on AWS instance",
    )
    parser.add_argument(
        "--aws-python",
        type=str,
        default="python",
        help="Python command on AWS instance (e.g. 'conda run -n torch python')",
    )

    args = parser.parse_args()

    if args.llm == "claude":
        llm = ClaudeCLIClient(
            model=args.llm_model or "sonnet",
            timeout=args.llm_timeout,
        )
    elif args.llm == "codex":
        llm = CodexCLIClient(
            model=args.llm_model,
            timeout=args.llm_timeout,
        )
    else:
        llm = MockLLMClient()

    if args.executor == "aws":
        if not args.aws_host:
            print("ERROR: --aws-host is required when using --executor aws", file=sys.stderr)
            return 1
        executor = AWSExecutor(
            host=args.aws_host,
            key_path=args.aws_key,
            remote_work_dir=args.aws_work_dir,
            remote_python=args.aws_python,
        )
    else:
        executor = LocalExecutor()

    harness = ThreeAgentHarness(
        llm=llm,
        log_dir=Path(args.log_dir),
        max_trials=args.max_trials,
        max_rounds=args.max_rounds,
        executor=executor,
    )

    task = UserTask(
        name=args.task_name,
        description=args.task_desc,
        data_path=args.data_path,
        num_classes=args.num_classes,
        eval_metric=args.eval_metric,
        base_model=args.base_model,
    )

    if args.command == "run":
        result = harness.run(task=task, budget=args.budget)
        _print_result(result)
        return 0

    elif args.command == "benchmark":
        lb = harness.run_benchmark(task=task)
        _print_leaderboard(lb)
        return 0

    elif args.command == "research":
        base = args.base_model or "dinov2_vitb14"
        result = harness.run_research(base_model=base, task=task)
        _print_result(result)
        return 0

    elif args.command == "leaderboard":
        harness.run_benchmark(task=task)
        lb = harness.get_leaderboard()
        _print_leaderboard(lb)
        return 0

    elif args.command == "status":
        s = harness.state
        print(f"Phase: {s.phase}")
        print(f"Budget: {s.budget_remaining:.1f}/{s.budget_total:.1f} GPU-hr")
        return 0

    return 1


def _print_leaderboard(lb) -> None:
    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<6}{'Model':<25}{'Backend':<12}{'Params':<10}", end="")
    for b in lb.benchmarks:
        print(f"{b:<15}", end="")
    print()
    print("-" * 70)
    for e in lb.entries:
        print(f"#{e.overall_rank:<5}{e.model_name:<25}{e.backend:<12}{e.params_m:<10.0f}", end="")
        for b in lb.benchmarks:
            score = e.scores.get(b, 0)
            print(f"{score:<15.1f}", end="")
        print()
    if lb.recommendation:
        print(f"\nRecommended: {lb.recommendation}")
        print(f"Reason: {lb.recommendation_reason[:100]}")


def _print_result(result) -> None:
    print("\n" + "=" * 60)
    print("RESEARCH RESULT")
    print("=" * 60)
    print(f"  Task:        {result.task.name}")
    print(f"  Base model:  {result.base_model}")
    print(f"  Baseline:    {result.baseline_score:.1f}%")
    print(f"  Best score:  {result.best_score:.1f}%")
    print(f"  Improvement: +{result.improvement:.1f}%")
    print(f"  Trials:      {len(result.trials)}")
    print(f"  Checkpoint:  {result.checkpoint_path}")
    if result.best_config:
        print(f"  Best config: lr={result.best_config.get('lr')}, "
              f"freeze={result.best_config.get('freeze_backbone')}, "
              f"adapter={result.best_config.get('adapter')}")


if __name__ == "__main__":
    sys.exit(main())
