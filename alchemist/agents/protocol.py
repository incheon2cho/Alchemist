"""Agent Message Protocol — AD-3A2.

Structured message envelope for 3-Agent communication with audit logging.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    CONTROLLER = "controller"
    BENCHMARK = "benchmark"
    RESEARCH = "research"


class MessageType(str, Enum):
    DIRECTIVE = "directive"      # Controller → Agent 지시
    RESPONSE = "response"        # Agent → Controller 응답
    STATUS = "status"            # Agent → Controller 진행 보고
    ESCALATION = "escalation"    # Agent → Controller 문제 보고


@dataclass
class AgentMessage:
    """Structured message envelope for inter-agent communication."""

    from_agent: AgentRole
    to_agent: AgentRole
    msg_type: MessageType
    payload: dict[str, Any]
    episode: int = 0
    budget_remaining: float = 0.0
    trace_id: str = ""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "msg_id": self.msg_id,
            "from_agent": self.from_agent.value,
            "to_agent": self.to_agent.value,
            "msg_type": self.msg_type.value,
            "payload": self.payload,
            "episode": self.episode,
            "budget_remaining": self.budget_remaining,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentMessage:
        return cls(
            from_agent=AgentRole(d["from_agent"]),
            to_agent=AgentRole(d["to_agent"]),
            msg_type=MessageType(d["msg_type"]),
            payload=d.get("payload", {}),
            episode=d.get("episode", 0),
            budget_remaining=d.get("budget_remaining", 0.0),
            trace_id=d.get("trace_id", ""),
            msg_id=d.get("msg_id", str(uuid.uuid4())[:8]),
            timestamp=d.get("timestamp", ""),
        )


class MessageBus:
    """Central message bus with audit logging.

    All inter-agent messages pass through the bus for logging and routing.
    """

    def __init__(self, log_dir: Path | None = None):
        self._log: list[AgentMessage] = []
        self._log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def send(self, msg: AgentMessage) -> None:
        """Record and log a message."""
        self._log.append(msg)
        logger.info(
            "[%s] %s → %s (%s) ep=%d | %s",
            msg.msg_id,
            msg.from_agent.value,
            msg.to_agent.value,
            msg.msg_type.value,
            msg.episode,
            _payload_summary(msg.payload),
        )
        if self._log_dir:
            self._flush_to_disk(msg)

    def get_log(self) -> list[AgentMessage]:
        return list(self._log)

    def get_episode_log(self, episode: int) -> list[AgentMessage]:
        return [m for m in self._log if m.episode == episode]

    def get_trace(self, trace_id: str) -> list[AgentMessage]:
        return [m for m in self._log if m.trace_id == trace_id]

    def _flush_to_disk(self, msg: AgentMessage) -> None:
        fp = self._log_dir / f"episode_{msg.episode:04d}.jsonl"
        with open(fp, "a") as f:
            f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")


def _payload_summary(payload: dict, max_len: int = 120) -> str:
    keys = list(payload.keys())
    s = ", ".join(f"{k}={_short(payload[k])}" for k in keys[:5])
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def make_directive(
    to: AgentRole,
    payload: dict[str, Any],
    episode: int,
    budget: float,
    trace_id: str,
) -> AgentMessage:
    """Helper to create a Controller directive."""
    return AgentMessage(
        from_agent=AgentRole.CONTROLLER,
        to_agent=to,
        msg_type=MessageType.DIRECTIVE,
        payload=payload,
        episode=episode,
        budget_remaining=budget,
        trace_id=trace_id,
    )


def make_response(
    from_agent: AgentRole,
    payload: dict[str, Any],
    episode: int,
    budget: float,
    trace_id: str,
) -> AgentMessage:
    """Helper to create an agent response."""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=AgentRole.CONTROLLER,
        msg_type=MessageType.RESPONSE,
        payload=payload,
        episode=episode,
        budget_remaining=budget,
        trace_id=trace_id,
    )


def make_escalation(
    from_agent: AgentRole,
    payload: dict[str, Any],
    episode: int,
    trace_id: str,
) -> AgentMessage:
    """Helper to create an escalation message."""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=AgentRole.CONTROLLER,
        msg_type=MessageType.ESCALATION,
        payload=payload,
        episode=episode,
        trace_id=trace_id,
    )
