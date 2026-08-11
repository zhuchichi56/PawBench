import asyncio
import subprocess

from pawbench.envs.docker import DockerEnvironment


def test_nested_container_cleanup_is_bounded(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "stop":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1:3] == ["ps", "-aq"]:
            # Authoritative postcondition: no container record remains.
            return subprocess.CompletedProcess(command, 0, stdout="")
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
        ["ps", "-aq", "--filter", "name=^task-container$"],
    ]
    assert [call[1]["timeout"] for call in calls] == [15, 15, 15, 15]
    assert environment.container_id is None
    assert environment.is_running is False
