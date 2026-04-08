"""LLM client abstraction with MockLLM for testing.

Supports graceful degradation: if LLM fails, deterministic fallback is used.
Also provides CLI-based clients for Claude Code and OpenAI Codex.
"""

from __future__ import annotations

import json
import logging
import subprocess
import shutil
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text from a prompt."""

    def generate_json(self, prompt: str, system: str = "") -> dict[str, Any]:
        """Generate and parse JSON output."""
        raw = self.generate(prompt, system)
        # Extract JSON from markdown code blocks if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())


class MockLLMClient(LLMClient):
    """Mock LLM for testing — returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._default_responses = {
            "analyze": json.dumps({
                "strengths": ["spatial performance above target"],
                "weaknesses": ["temporal K400 below target by 3%"],
                "recommendations": ["focus on temporal adapter"],
                "focus_areas": ["k400_top1", "ssv2_top1"],
            }),
            "propose": json.dumps({
                "plans": [{
                    "description": "Add TSM adapter for temporal modeling",
                    "architecture_ir": {"temporal_type": "tsm"},
                    "hyperparams": {"lr": 1e-3, "epochs": 50},
                    "compression_strategy": "none",
                }],
            }),
            "refine": "Based on the regression in temporal metrics, "
                      "the TSM adapter should improve K400 by ~2-3%.",
            "reflect": json.dumps({
                "diagnosis": "temporal weakness due to lack of motion modeling",
                "suggestion": "add temporal shift module",
            }),
            "failure": json.dumps({
                "root_cause": "learning rate too high for temporal head",
                "alternatives": [{
                    "description": "Reduce LR to 1e-4 for temporal head",
                    "architecture_ir": {"temporal_type": "tsm"},
                    "hyperparams": {"lr": 1e-4, "epochs": 50},
                    "compression_strategy": "none",
                }],
                "should_escalate": False,
            }),
        }
        self.call_count = 0

    def generate(self, prompt: str, system: str = "") -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()
        # Match against registered responses first
        for key, response in self._responses.items():
            if key.lower() in prompt_lower:
                return response
        # Fall back to defaults
        for key, response in self._default_responses.items():
            if key in prompt_lower:
                return response
        return json.dumps({"result": "ok"})


class ClaudeCLIClient(LLMClient):
    """LLM client that uses the Claude Code CLI (claude -p)."""

    def __init__(self, model: str = "sonnet", timeout: int = 120):
        self._model = model
        self._timeout = timeout
        if not shutil.which("claude"):
            raise RuntimeError("'claude' CLI not found in PATH")

    def generate(self, prompt: str, system: str = "") -> str:
        cmd = ["claude", "-p", "--model", self._model]
        if system:
            cmd += ["--system-prompt", system]
        cmd.append(prompt)
        logger.debug("ClaudeCLI: %s", " ".join(cmd[:6]))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode != 0:
                err = result.stderr.strip()
                logger.warning("ClaudeCLI returned %d: %s", result.returncode, err)
                raise RuntimeError(f"claude CLI error: {err}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {self._timeout}s"
            )


class CodexCLIClient(LLMClient):
    """LLM client that uses the OpenAI Codex CLI (codex exec)."""

    def __init__(self, model: str | None = None, timeout: int = 120):
        self._model = model
        self._timeout = timeout
        if not shutil.which("codex"):
            raise RuntimeError("'codex' CLI not found in PATH")

    def generate(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if self._model:
            cmd += ["-m", self._model]
        cmd.append(full_prompt)
        logger.debug("CodexCLI: %s", " ".join(cmd[:6]))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                err = result.stderr.strip()
                logger.warning("CodexCLI returned %d: %s", result.returncode, err)
                raise RuntimeError(f"codex CLI error: {err or output}")
            return self._parse_codex_output(output)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"codex CLI timed out after {self._timeout}s"
            )

    @staticmethod
    def _parse_codex_output(raw: str) -> str:
        """Extract the actual response from codex exec output.

        codex exec prints a header block, then 'user' message, then 'codex'
        response, then 'tokens used' footer. We extract the codex response.
        """
        lines = raw.split("\n")
        # Find the 'codex' marker line — response follows it
        response_lines: list[str] = []
        capturing = False
        for line in lines:
            if line.strip() == "codex":
                capturing = True
                continue
            if capturing:
                if line.strip() == "tokens used":
                    break
                response_lines.append(line)
        if response_lines:
            return "\n".join(response_lines).strip()
        # Fallback: return everything (in case format changes)
        return raw


def safe_llm_call(
    client: LLMClient,
    prompt: str,
    system: str = "",
    fallback: Any = None,
) -> Any:
    """Call LLM with graceful degradation on failure."""
    try:
        return client.generate_json(prompt, system)
    except Exception as e:
        logger.warning("LLM call failed (%s), using fallback", e)
        return fallback if fallback is not None else {}
