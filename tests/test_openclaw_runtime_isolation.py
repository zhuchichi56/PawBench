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


def test_gateway_port_bundles_do_not_overlap_auxiliary_listeners():
    first = OpenClawAgent(model="custom/Qwen3.5-4B")
    second = OpenClawAgent(model="custom/Qwen3.5-4B")

    assert second._gateway_port - first._gateway_port >= 10
    assert first._gateway_port + 2 != second._gateway_port


def test_nested_gateway_uses_owned_process_group_without_broad_kill():
    setup_source = inspect.getsource(OpenClawAgent._start_gateway)
    cleanup_source = inspect.getsource(OpenClawAgent._kill_gateway)
    nested_cleanup_source = inspect.getsource(OpenClawAgent._kill_nested_gateway)

    assert "setsid" in setup_source
    assert "_gateway_group_for_port" in setup_source
    assert "os.killpg" in nested_cleanup_source
    assert "pkill" not in setup_source + cleanup_source + nested_cleanup_source


def test_nested_gateway_group_cleanup_is_exact(monkeypatch):
    import os
    import subprocess
    import time

    process = subprocess.Popen(
        ["bash", "-c", "exec -a openclaw-test sleep 60"],
        start_new_session=True,
    )
    try:
        for _ in range(50):
            members = OpenClawAgent._process_group_members(process.pid)
            if members and all("openclaw" in item[2] for item in members):
                break
            time.sleep(0.01)
        agent = OpenClawAgent(model="custom/Qwen3.5-4B")
        agent._gateway_pgid = process.pid
        monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
        monkeypatch.setattr(agent, "_port_is_open", lambda _port: False)

        class Environment:
            async def execute_command(self, command, timeout=None):
                return {"returncode": 0, "stdout": "", "stderr": ""}

        asyncio.run(agent._kill_gateway(Environment()))
        process.wait(timeout=5)
        assert agent._gateway_pgid is None
        assert not agent._process_group_members(process.pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait(timeout=5)
