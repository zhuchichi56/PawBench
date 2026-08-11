import asyncio
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pawbench.envs.docker import DockerEnvironment


def test_nested_podman_start_transactions_are_serialized(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_run(command, **kwargs):
        nonlocal active, maximum
        if command[:2] == ["docker", "run"]:
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return subprocess.CompletedProcess(command, 0, stdout="container-id\n", stderr="")
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def start_one(index):
        env = DockerEnvironment(name=f"nested-{index}", image="test")
        asyncio.run(env.start())

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(start_one, range(8)))
    assert maximum == 1


def test_nested_podman_stop_rejects_removing_record(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    env = DockerEnvironment(name="stale", image="test")
    env.container_id = "stale-id"

    def fake_run(command, **kwargs):
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="stale-id\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        asyncio.run(env.stop())
    except RuntimeError as exc:
        assert "cleanup incomplete" in str(exc)
    else:
        raise AssertionError("stale removing record was accepted as clean")


def test_start_failure_cleans_created_container_by_name(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    env = DockerEnvironment(name="partial", image="test")
    run_failed = False
    cleanup_seen = False

    def fake_run(command, **kwargs):
        nonlocal run_failed, cleanup_seen
        if command[:2] == ["docker", "run"]:
            run_failed = True
            raise subprocess.TimeoutExpired(command, 60)
        if command[1:3] == ["rm", "-f"] and run_failed:
            cleanup_seen = True
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        asyncio.run(env.start())
    except RuntimeError as exc:
        assert "Timed out starting" in str(exc)
    else:
        raise AssertionError("start timeout was accepted")
    assert cleanup_seen
    assert env.container_id is None
    assert not env.is_running


def test_start_cancellation_waits_for_storage_transaction(monkeypatch):
    env = DockerEnvironment(name="cancelled", image="test")
    release = threading.Event()
    finished = threading.Event()

    def fake_start_sync():
        release.wait(timeout=5)
        env.container_id = "created-after-cancel"
        env._is_running = True
        finished.set()

    monkeypatch.setattr(env, "_start_sync", fake_start_sync)

    async def exercise():
        task = asyncio.create_task(env.start())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("start cancellation was lost")

    asyncio.run(exercise())
    assert finished.is_set()
    assert env.container_id == "created-after-cancel"


def test_cleanup_verification_command_failure_is_not_clean(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    env = DockerEnvironment(name="verify-failed", image="test")
    env.container_id = "id"

    def fake_run(command, **kwargs):
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(command, 125, stdout="", stderr="store error")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        asyncio.run(env.stop())
    except RuntimeError as exc:
        assert "verification-error" in str(exc)
    else:
        raise AssertionError("failed cleanup verification was accepted")
