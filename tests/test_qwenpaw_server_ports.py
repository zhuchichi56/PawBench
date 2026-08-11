import asyncio
import socket

import pytest

from pawbench.agents.impl.qwenpaw_agent import QwenPawAgent


class FakeEnvironment:
    def __init__(self, result=None):
        self.result = result or {"returncode": 0, "stdout": "[qwenpaw-server] ready\n"}
        self.commands = []
        self.files = {}

    async def execute_command(self, command, timeout=None):
        self.commands.append((command, timeout))
        return self.result

    async def write_file(self, path, content):
        self.files[path] = content


def test_two_agents_reserve_distinct_host_ports_and_release():
    first = QwenPawAgent()
    second = QwenPawAgent()
    first_port = first._allocate_server_port()
    second_port = second._allocate_server_port()
    try:
        assert first_port != second_port
        assert first._server_url == f"http://127.0.0.1:{first_port}"
        assert second._server_url == f"http://127.0.0.1:{second_port}"
    finally:
        first._release_server_port()
        second._release_server_port()
    assert first._server_url is None
    assert second._server_url is None


def test_allocator_skips_a_port_already_bound_by_an_unrelated_process():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    agent = QwenPawAgent()
    try:
        # The range start is stable and intentionally asserted as a regression guard.
        blocker.bind(("127.0.0.1", 18088))
        blocker.listen(1)
        assert agent._allocate_server_port() != 18088
    finally:
        agent._release_server_port()
        blocker.close()


def test_dynamic_port_reaches_startup_readiness_and_agent_api():
    agent = QwenPawAgent()
    env = FakeEnvironment()
    port = agent._allocate_server_port()
    agent._api_key = "test-key"
    agent._base_url = "http://model.example/v1"
    agent._model_config = type("Model", (), {
        "model_name": "test-model",
        "provider": type("Provider", (), {"value": "custom"})(),
    })()
    try:
        asyncio.run(agent._start_server(env))
        startup_script = env.files["/tmp/qwenpaw_start.py"]
        command = env.commands[-1][0]
        assert f'"--port", "{port}"' in startup_script
        assert "__QWENPAW_SERVER_PORT__" not in startup_script
        assert f"http://127.0.0.1:{port}/api/version" in command

        # Avoid provider lookup details: the server URL assertion occurs in the
        # generated API client regardless of provider kind.
        call_script = agent._build_call_agent_script("test-session")
        assert f"URL          = 'http://127.0.0.1:{port}'" in call_script
        assert "http://127.0.0.1:8088" not in call_script
    finally:
        agent._release_server_port()


def test_server_readiness_failure_is_fatal_and_teardown_releases_port():
    agent = QwenPawAgent()
    env = FakeEnvironment({"returncode": 7, "stderr": "connection refused"})
    agent._api_key = ""
    agent._base_url = ""
    agent._allocate_server_port()
    with pytest.raises(RuntimeError, match="failed readiness"):
        asyncio.run(agent._start_server(env))
    asyncio.run(agent.teardown(env))
    assert agent._server_port is None
    assert agent._port_lock_file is None
