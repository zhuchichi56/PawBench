import asyncio
import subprocess

from pawbench.envs.docker import DockerEnvironment


def test_nested_container_cleanup_is_bounded(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "stop":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = DockerEnvironment(name="task-container")
    environment.container_id = "container-id"
    environment._is_running = True

    asyncio.run(environment.stop())

    assert [call[0][1] for call in calls] == ["stop", "rm"]
    assert all(call[1]["timeout"] == 15 for call in calls)
    assert environment.container_id is None
    assert environment.is_running is False
