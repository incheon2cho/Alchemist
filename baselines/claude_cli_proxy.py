#!/usr/bin/env python3
"""OpenAI-compatible HTTP proxy routing /v1/chat/completions to Claude CLI.

Purpose: allow tools (e.g., AutoML-Agent) that speak OpenAI's Chat API to
use the local `claude` CLI binary instead — zero API cost.

Endpoint implemented:
    POST /v1/chat/completions
    GET  /v1/models                    (returns a stub list)
    GET  /health                       (liveness)

Model dispatch:
    Any request (regardless of `model` field) is routed to `claude -p`.
    The Claude CLI subscription is the only backend.

JSON mode:
    If the caller sets `response_format={"type":"json_object"}`, a
    suffix is appended to the user content instructing Claude to emit
    ONLY a JSON object (no prose, no ```json fences). The proxy also
    post-processes the response to strip accidental markdown fences.

Usage:
    pip install fastapi uvicorn pydantic
    python3 claude_cli_proxy.py --port 8001 --cmd "claude -p"

Then point the OpenAI client at http://localhost:8001/v1.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("claude-cli-proxy")


# ----------------------------------------------------------------------
# Request/response models (minimal OpenAI chat API surface)
# ----------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]  # accept multipart but flatten to text


class ChatRequest(BaseModel):
    model: str = "claude"
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    # ignored: tools, tool_choice, stop, frequency_penalty, presence_penalty, etc.


# ----------------------------------------------------------------------
# Prompt flattening
# ----------------------------------------------------------------------
def _flatten_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        # ignore image / tool blocks
    return "\n".join(parts)


def build_prompt(req: ChatRequest) -> str:
    """Collapse the OpenAI-style messages list into a single prompt string.

    Uses XML-ish tags that are robust for Claude. System messages are hoisted
    to the top; user/assistant turns are preserved in order.
    """
    sys_parts: list[str] = []
    turn_parts: list[str] = []
    for m in req.messages:
        text = _flatten_content(m.content).strip()
        if not text:
            continue
        if m.role == "system":
            sys_parts.append(text)
        elif m.role == "assistant":
            turn_parts.append(f"<assistant>\n{text}\n</assistant>")
        else:  # user (default)
            turn_parts.append(f"<user>\n{text}\n</user>")

    pieces: list[str] = []
    if sys_parts:
        pieces.append("<system>\n" + "\n\n".join(sys_parts) + "\n</system>")
    pieces.extend(turn_parts)

    # JSON-mode instruction appended *after* the conversation for strong salience
    if req.response_format and req.response_format.get("type") == "json_object":
        pieces.append(
            "<output_format>\n"
            "Respond with EXACTLY ONE JSON object. No prose, no ```json fences, "
            "no comments, nothing else. Your entire response must parse with json.loads().\n"
            "</output_format>"
        )

    return "\n\n".join(pieces)


# ----------------------------------------------------------------------
# Claude CLI invocation
# ----------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_json_fences(s: str) -> str:
    """Remove leading/trailing ```json fences if present."""
    s = s.strip()
    # Fast path: triple-backtick wrapped block
    if s.startswith("```") and s.endswith("```"):
        # Drop first line fence and last fence
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
            while lines and not lines[-1].strip().startswith("```"):
                break
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def call_claude_cli(prompt: str, cmd: list[str], timeout: int = 600) -> str:
    """Invoke Claude CLI with the given prompt and return stdout."""
    log.info("claude invocation: prompt_chars=%d, timeout=%ds", len(prompt), timeout)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"Claude CLI timed out after {timeout}s")

    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error("claude exit=%d stderr=%s", proc.returncode, proc.stderr[:500])
        raise HTTPException(
            502,
            f"Claude CLI failed (exit {proc.returncode}): {proc.stderr[:500]}",
        )
    out = proc.stdout.strip()
    log.info("claude returned: out_chars=%d elapsed=%.1fs", len(out), elapsed)
    return out


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Claude CLI Proxy (OpenAI-compatible)")

# Stashed at startup via argparse
_CLI_CMD: list[str] = ["claude", "-p"]
_DEFAULT_TIMEOUT: int = 600


@app.get("/health")
def health():
    return {"ok": True, "cmd": _CLI_CMD}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "claude", "object": "model", "created": 0, "owned_by": "claude-cli"},
            # Aliases so AutoML-Agent's "gpt-4" / "gpt-4o" route here unchanged.
            {"id": "gpt-4", "object": "model", "created": 0, "owned_by": "claude-cli"},
            {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "claude-cli"},
            {"id": "gpt-4.1", "object": "model", "created": 0, "owned_by": "claude-cli"},
            {"id": "gpt-3.5-turbo", "object": "model", "created": 0, "owned_by": "claude-cli"},
            {"id": "prompt-llama", "object": "model", "created": 0, "owned_by": "claude-cli"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = build_prompt(req)
    raw = call_claude_cli(prompt, _CLI_CMD, timeout=_DEFAULT_TIMEOUT)

    # If caller asked for JSON, strip accidental fences.
    content = raw
    if req.response_format and req.response_format.get("type") == "json_object":
        content = _strip_json_fences(raw)
        # Validate; if invalid, try to locate the first {...} block.
        try:
            json.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                content = match.group(0)

    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(content) // 4)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# Legacy completions (some callers use /v1/completions)
class LegacyRequest(BaseModel):
    model: str = "claude"
    prompt: str = ""
    max_tokens: int | None = None
    temperature: float | None = None


@app.post("/v1/completions")
def legacy_completions(req: LegacyRequest):
    raw = call_claude_cli(req.prompt, _CLI_CMD, timeout=_DEFAULT_TIMEOUT)
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": raw, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": max(1, len(req.prompt) // 4),
            "completion_tokens": max(1, len(raw) // 4),
            "total_tokens": max(1, (len(req.prompt) + len(raw)) // 4),
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--cmd",
        default="claude -p",
        help="Claude CLI invocation (default: 'claude -p'). Pass via prompt stdin.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    global _CLI_CMD, _DEFAULT_TIMEOUT
    _CLI_CMD = args.cmd.split()
    _DEFAULT_TIMEOUT = args.timeout

    # Sanity probe: run claude --version quickly
    probe = subprocess.run(
        [_CLI_CMD[0], "--version"], capture_output=True, text=True, timeout=10
    )
    if probe.returncode == 0:
        log.info("claude CLI detected: %s", probe.stdout.strip())
    else:
        log.warning("claude --version failed: %s", probe.stderr[:200])

    log.info("Starting proxy on http://%s:%d (cmd=%s)", args.host, args.port, _CLI_CMD)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
