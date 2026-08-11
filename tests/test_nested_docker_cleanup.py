import asyncio
import subprocess

from pawbench.envs.docker import DockerEnvironment


def test_nested_container_cleanup_is_bounded(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "stop":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1:3] == ["container", "exists"]:
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = DockerEnvironment(name="task-container")
    environment.container_id = "container-id"
    environment._is_running = True

    asyncio.run(environment.stop())

    assert [call[0][1:] for call in calls] == [
        ["stop", "-t", "5", "task-container"],
        ["rm", "-f", "task-container"],
        ["container", "cleanup", "--rm", "task-container"],
        ["container", "exists", "task-container"],
    ]
    assert [call[1]["timeout"] for call in calls] == [15, 15, 15, 10]
    assert environment.container_id is None
    assert environment.is_running is False
