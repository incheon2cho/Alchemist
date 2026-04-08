"""Tests for Agent Message Protocol."""

import json
from pathlib import Path

from alchemist.agents.protocol import (
    AgentMessage,
    AgentRole,
    MessageBus,
    MessageType,
    make_directive,
    make_escalation,
    make_response,
)


def test_agent_message_roundtrip():
    msg = AgentMessage(
        from_agent=AgentRole.CONTROLLER,
        to_agent=AgentRole.BENCHMARK,
        msg_type=MessageType.DIRECTIVE,
        payload={"models": ["v1", "v2"]},
        episode=3,
        budget_remaining=80.0,
        trace_id="abc123",
    )
    d = msg.to_dict()
    restored = AgentMessage.from_dict(d)
    assert restored.from_agent == AgentRole.CONTROLLER
    assert restored.to_agent == AgentRole.BENCHMARK
    assert restored.payload["models"] == ["v1", "v2"]
    assert restored.episode == 3


def test_message_bus_logging():
    bus = MessageBus()
    msg = make_directive(
        to=AgentRole.RESEARCH,
        payload={"action": "test"},
        episode=1,
        budget=50.0,
        trace_id="t1",
    )
    bus.send(msg)
    assert len(bus.get_log()) == 1
    assert bus.get_episode_log(1)[0].msg_id == msg.msg_id


def test_message_bus_trace():
    bus = MessageBus()
    for i in range(3):
        bus.send(make_directive(
            to=AgentRole.BENCHMARK,
            payload={"i": i},
            episode=1,
            budget=50.0,
            trace_id="trace_x",
        ))
    bus.send(make_directive(
        to=AgentRole.BENCHMARK,
        payload={"i": 99},
        episode=2,
        budget=50.0,
        trace_id="other",
    ))
    assert len(bus.get_trace("trace_x")) == 3


def test_message_bus_disk_logging(tmp_path):
    log_dir = tmp_path / "logs"
    bus = MessageBus(log_dir=log_dir)
    bus.send(make_directive(
        to=AgentRole.RESEARCH,
        payload={"test": True},
        episode=1,
        budget=50.0,
        trace_id="t1",
    ))
    fp = log_dir / "episode_0001.jsonl"
    assert fp.exists()
    lines = fp.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["from_agent"] == "controller"


def test_make_helpers():
    d = make_directive(AgentRole.BENCHMARK, {"x": 1}, 1, 50.0, "t")
    assert d.msg_type == MessageType.DIRECTIVE
    assert d.from_agent == AgentRole.CONTROLLER

    r = make_response(AgentRole.BENCHMARK, {"y": 2}, 1, 50.0, "t")
    assert r.msg_type == MessageType.RESPONSE
    assert r.to_agent == AgentRole.CONTROLLER

    e = make_escalation(AgentRole.RESEARCH, {"reason": "fail"}, 1, "t")
    assert e.msg_type == MessageType.ESCALATION
