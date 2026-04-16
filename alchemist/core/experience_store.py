"""VisionExperienceStore — persistent cross-task memory for the Research Agent.

Each completed Research run appends a record capturing the task, winning
config, baseline/best scores, and a short summary of what worked. Future
runs retrieve similar past tasks and feed them into the LLM-based
``suggest_techniques`` prompt so the agent "becomes more expert" with use.

Similarity is lightweight: num_classes bucket + keyword overlap on task
name / description. This is sufficient to pull CIFAR-100 experience into a
Butterfly run (both fine-grained, moderate class count) without requiring
task embeddings.

Storage: JSONL at ``~/.cache/alchemist/experience.jsonl`` (configurable via
``ALCHEMIST_EXPERIENCE_PATH`` env var).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(
    os.environ.get(
        "ALCHEMIST_EXPERIENCE_PATH",
        Path.home() / ".cache" / "alchemist" / "experience.jsonl",
    )
)


def _normalize_text(s: str) -> set[str]:
    """Extract content-word tokens from a task description."""
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
        "and", "or", "is", "are", "be", "from", "as", "by", "this", "that",
        "build", "model", "classify", "image", "images", "dataset",
        "training", "set", "test", "val", "validation", "hardware", "time",
        "budget", "single", "allowed", "evaluation", "metric", "top",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", s.lower())
    return {w for w in words if w not in stop}


class VisionExperienceStore:
    """Persistent experience memory for the Research Agent."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- write ---------------------------------------------------------
    def record(
        self,
        task_name: str,
        task_description: str,
        num_classes: int,
        base_model: str,
        baseline_score: float,
        best_score: float,
        best_config: dict[str, Any],
        techniques_tried: list[str],
        summary: str,
        rounds_run: int = 1,
        total_trials: int = 0,
    ) -> None:
        """Append one completed-run record."""
        record = {
            "timestamp": time.time(),
            "task_name": task_name,
            "task_description": task_description[:500],
            "num_classes": int(num_classes),
            "base_model": base_model,
            "baseline_score": float(baseline_score),
            "best_score": float(best_score),
            "improvement": float(best_score - baseline_score),
            "best_config": best_config,
            "techniques_tried": list(techniques_tried),
            "summary": summary[:500],
            "rounds_run": int(rounds_run),
            "total_trials": int(total_trials),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Experience recorded: %s (%s → %.2f%%, %+.2f%%p)",
            task_name, base_model, best_score, best_score - baseline_score,
        )

    # --- read ----------------------------------------------------------
    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def retrieve_similar(
        self,
        task_name: str,
        task_description: str,
        num_classes: int,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` past experiences most similar to the new task.

        Similarity = (num_classes-bucket match) + (keyword overlap Jaccard).
        """
        entries = self.load_all()
        if not entries:
            return []

        new_tokens = _normalize_text(f"{task_name} {task_description}")
        new_bucket = self._class_bucket(num_classes)

        scored: list[tuple[float, dict[str, Any]]] = []
        for e in entries:
            # Skip trivial self-match (same task name)
            if e.get("task_name") == task_name:
                continue
            e_tokens = _normalize_text(
                f"{e.get('task_name', '')} {e.get('task_description', '')}"
            )
            jaccard = (
                len(new_tokens & e_tokens) / max(len(new_tokens | e_tokens), 1)
                if new_tokens or e_tokens else 0.0
            )
            bucket_match = 1.0 if self._class_bucket(e.get("num_classes", 0)) == new_bucket else 0.3
            score = 0.7 * jaccard + 0.3 * bucket_match
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    @staticmethod
    def _class_bucket(n: int) -> str:
        if n <= 5:
            return "tiny"
        if n <= 20:
            return "small"
        if n <= 200:
            return "medium"
        return "large"

    # --- LLM prompt helper --------------------------------------------
    def summarize_for_prompt(self, entries: list[dict[str, Any]], max_chars: int = 2000) -> str:
        if not entries:
            return ""
        lines = ["Prior experience on similar vision tasks:"]
        for i, e in enumerate(entries, 1):
            cfg = e.get("best_config") or {}
            tech = ", ".join(e.get("techniques_tried") or []) or "(basic HP only)"
            lines.append(
                f"  [{i}] {e['task_name']} ({e['num_classes']}-class, "
                f"{e.get('base_model', '?')}): "
                f"baseline {e['baseline_score']:.1f}% → best {e['best_score']:.1f}% "
                f"({e['improvement']:+.2f}%p). "
                f"Winning config: lr={cfg.get('lr', '?')} "
                f"batch={cfg.get('batch_size', '?')} "
                f"epochs={cfg.get('epochs', '?')} "
                f"freeze={cfg.get('freeze_backbone', '?')} "
                f"mixup={cfg.get('mixup', False)} "
                f"cutmix={cfg.get('cutmix', False)} "
                f"ema={cfg.get('ema', False)}. "
                f"Techniques tried: {tech}. "
                f"Summary: {e.get('summary', '')[:150]}"
            )
        text = "\n".join(lines)
        return text[:max_chars]
