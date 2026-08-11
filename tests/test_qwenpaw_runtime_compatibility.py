import asyncio

import pytest

from pawbench.agents.impl.qwenpaw_agent import QwenPawAgent


class FakeEnvironment:
    def __init__(self, result):
        self.result = result
        self.commands = []

    async def execute_command(self, command, timeout=None):
        self.commands.append((command, timeout))
        return self.result


def test_runtime_compatibility_checks_exact_versions_and_import():
    env = FakeEnvironment({"returncode": 0, "stdout": "QWENPAW_RUNTIME_COMPAT_OK\n"})
    agent = QwenPawAgent()

    asyncio.run(agent._assert_runtime_compatibility(env))

    command, timeout = env.commands[0]
    assert "qwenpaw" in command
    assert "1.1.3" in command
    assert "agent-client-protocol" in command
    assert "0.10.1" in command
    assert "SetSessionModelResponse" in command
    assert timeout == 15


def test_runtime_compatibility_fails_before_server_startup():
    env = FakeEnvironment({"returncode": 1, "stderr": "cannot import name SetSessionModelResponse"})
    agent = QwenPawAgent()

    with pytest.raises(RuntimeError, match="runtime compatibility check failed"):
        asyncio.run(agent._assert_runtime_compatibility(env))
