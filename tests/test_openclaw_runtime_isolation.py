import asyncio
import inspect

import pytest

from pawbench.agents.impl import openclaw_agent
from pawbench.agents.impl.openclaw_agent import OpenClawAgent


def test_openclaw_agent_identity_is_unique_per_task():
    first = OpenClawAgent(model="custom/Qwen3.5-4B")
    second = OpenClawAgent(model="custom/Qwen3.5-4B")

    assert first._agent_id() != second._agent_id()
    assert str(first._gateway_port) in first._agent_id()
    assert str(second._gateway_port) in second._agent_id()


def test_setup_does_not_delete_ephemeral_agent():
    source = inspect.getsource(OpenClawAgent._setup_limited)
    teardown_source = inspect.getsource(OpenClawAgent.teardown)

    assert "openclaw agents delete" not in source
    assert "openclaw agents delete" not in teardown_source


def test_setup_limiter_caps_only_setup(monkeypatch):
    monkeypatch.setenv("PAWBENCH_OPENCLAW_SETUP_CONCURRENCY", "8")
    active = 0
    maximum = 0

    async def fake_setup_limited(self, environment):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1

    monkeypatch.setattr(OpenClawAgent, "_setup_limited", fake_setup_limited)

    async def exercise():
        agents = [OpenClawAgent(model="custom/Qwen3.5-4B") for _ in range(24)]
        await asyncio.gather(*(agent.setup(None) for agent in agents))

    asyncio.run(exercise())
    assert maximum == 8


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_setup_limiter_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("PAWBENCH_OPENCLAW_SETUP_CONCURRENCY", value)

    async def exercise():
        openclaw_agent._get_openclaw_setup_semaphore()

    with pytest.raises(ValueError):
        asyncio.run(exercise())


def test_session_flush_timeout_is_best_effort(monkeypatch):
    agent = OpenClawAgent(model="custom/Qwen3.5-4B")

    class Environment:
        async def write_file(self, path, content):
            return True

        async def execute_command(self, command, timeout=None):
            raise TimeoutError("slow exec")

    asyncio.run(
        agent._wait_for_session_flush(
            Environment(), agent_id_lower=agent._agent_id().lower()
        )
    )
