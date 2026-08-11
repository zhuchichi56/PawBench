"""Regression tests for gateway readiness and cleanup under the hard limit.

Two defects surfaced by run A12 at harness concurrency 32:

* ``_wait_gateway_ready`` accepted an open TCP port. Measured on the benchmark
  image, the gateway binds at 15.3s but cannot complete a WebSocket handshake
  until 105.4s, so the agent connected into ``[ws] handshake timeout`` and lost
  ~90s of its own task budget.
* The outer hard limit cancels ``_run_agent_async``. ``CancelledError`` is a
  ``BaseException``, so ``except Exception`` in the cleanup ``finally`` skipped
  ``env.stop()``, and a failing ``env.stop()`` was swallowed silently. The run
  reached 51 live containers at concurrency 32.
"""

import asyncio
import inspect
import shlex
import socket
import subprocess
import sys
import threading
import time

import pytest

from pawbench import backend
from pawbench.agents.impl import openclaw_agent
from pawbench.agents.impl.openclaw_agent import OpenClawAgent
from pawbench.envs import docker as docker_env


# ── gateway readiness ─────────────────────────────────────────────────────────

def _probe(
    port: int,
    timeout: float,
    log: str,
    marker: str = openclaw_agent._GATEWAY_RUNTIME_READY_MARKER,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", openclaw_agent._GATEWAY_READY_PROBE,
         str(port), str(timeout), log, marker],
        capture_output=True, text=True, timeout=timeout + 30,
    )


def _ready_log(tmp_path, marker: str = openclaw_agent._GATEWAY_RUNTIME_READY_MARKER):
    """A gateway log that already reports the runtime backend ready."""
    path = tmp_path / "openclaw_gateway.log"
    path.write_text(f"[plugins] {marker} (cwd: /app)\n")
    return str(path)


def _serve(handler, ready: threading.Event) -> tuple[int, threading.Thread]:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]

    def serve() -> None:
        listener.settimeout(0.5)
        while not ready.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                handler(conn)
            finally:
                conn.close()
        listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, thread


def test_probe_rejects_a_port_that_only_accepts_tcp(tmp_path):
    """The exact A12 failure: listener bound, handshake never answered."""
    stop = threading.Event()

    def silent(conn):
        conn.recv(1024)
        time.sleep(3)  # what "starting channels and sidecars" looks like

    port, thread = _serve(silent, stop)
    try:
        result = _probe(port, 3, _ready_log(tmp_path))
        assert result.returncode == 1, result
        assert "GATEWAY_NOT_READY" in result.stdout
        assert "GATEWAY_READY\n" not in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_probe_accepts_a_completed_websocket_upgrade(tmp_path):
    stop = threading.Event()

    def upgrade(conn):
        conn.recv(1024)
        conn.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
        )

    port, thread = _serve(upgrade, stop)
    try:
        result = _probe(port, 10, _ready_log(tmp_path))
        assert result.returncode == 0, result
        assert "GATEWAY_READY" in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_probe_rejects_a_plain_http_response(tmp_path):
    """A 404/200 means the router is up but the agent channel is not."""
    stop = threading.Event()

    def not_found(conn):
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")

    port, thread = _serve(not_found, stop)
    try:
        result = _probe(port, 3, _ready_log(tmp_path))
        assert result.returncode == 1, result
        assert "404" in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_probe_becomes_ready_once_the_gateway_finishes_starting(tmp_path):
    """Readiness must be observed by polling, not by a single attempt."""
    stop = threading.Event()
    started = time.monotonic()

    def late(conn):
        conn.recv(1024)
        if time.monotonic() - started < 2:
            return  # close without responding
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")

    port, thread = _serve(late, stop)
    try:
        result = _probe(port, 20, _ready_log(tmp_path))
        assert result.returncode == 0, result
        assert "GATEWAY_READY" in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_wait_gateway_ready_uses_the_websocket_probe():
    calls = []

    class FakeEnv:
        async def execute_command(self, command, timeout=None):
            calls.append((command, timeout))
            return {"returncode": 0, "stdout": "GATEWAY_READY\n"}

    agent = OpenClawAgent(model="custom/Qwen3.5-4B")
    asyncio.run(agent._wait_gateway_ready(FakeEnv(), port=28088))
    command, timeout = calls[0]
    assert shlex.quote(openclaw_agent._GATEWAY_READY_PROBE) in command
    assert f" 28088 {openclaw_agent._GATEWAY_READY_TIMEOUT_S} " in command
    # Both readiness signals have to reach the probe.
    assert shlex.quote(openclaw_agent._GATEWAY_LOG) in command
    assert shlex.quote(openclaw_agent._GATEWAY_RUNTIME_READY_MARKER) in command
    # The exec wall-clock must outlast the probe's own deadline, otherwise the
    # middle timeout kills the probe and the error blames the wrong layer.
    assert timeout > openclaw_agent._GATEWAY_READY_TIMEOUT_S


def test_probe_rejects_a_101_before_the_runtime_backend_is_ready(tmp_path):
    """The A/B failure: 101 answered while acpx was only *registered*.

    A run submitted in that window is dropped with ``gateway closed (1000)``
    and the client silently falls back to the embedded runtime.
    """
    stop = threading.Event()

    def upgrade(conn):
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")

    log = tmp_path / "openclaw_gateway.log"
    log.write_text("[plugins] embedded acpx runtime backend registered\n")
    port, thread = _serve(upgrade, stop)
    try:
        result = _probe(port, 3, str(log))
        assert result.returncode == 1, result
        assert "runtime backend not ready" in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_probe_accepts_once_the_runtime_backend_reports_ready(tmp_path):
    stop = threading.Event()

    def upgrade(conn):
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")

    log = tmp_path / "openclaw_gateway.log"
    log.write_text("[plugins] embedded acpx runtime backend registered\n")

    def finish_starting():
        time.sleep(2)
        with log.open("a") as handle:
            handle.write(
                f"[plugins] {openclaw_agent._GATEWAY_RUNTIME_READY_MARKER}\n"
            )

    writer = threading.Thread(target=finish_starting, daemon=True)
    writer.start()
    port, thread = _serve(upgrade, stop)
    try:
        result = _probe(port, 20, str(log))
        assert result.returncode == 0, result
        assert "GATEWAY_READY" in result.stdout
    finally:
        stop.set()
        writer.join(timeout=10)
        thread.join(timeout=10)


def test_probe_tolerates_a_gateway_log_that_does_not_exist_yet(tmp_path):
    """The gateway is launched with nohup; its log can lag the first poll."""
    stop = threading.Event()

    def upgrade(conn):
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")

    port, thread = _serve(upgrade, stop)
    try:
        result = _probe(port, 3, str(tmp_path / "absent.log"))
        assert result.returncode == 1, result
        assert "runtime backend not ready" in result.stdout
    finally:
        stop.set()
        thread.join(timeout=10)


def test_wait_gateway_ready_raises_when_the_handshake_never_lands():
    class FakeEnv:
        async def execute_command(self, command, timeout=None):
            return {"returncode": 1, "stdout": "GATEWAY_NOT_READY last=timeout"}

    agent = OpenClawAgent(model="custom/Qwen3.5-4B")
    with pytest.raises(RuntimeError, match="gateway did not become ready"):
        asyncio.run(agent._wait_gateway_ready(FakeEnv(), port=28088))


def test_gateway_readiness_budget_covers_the_measured_startup():
    # Measured on openclaw-pawbench:2026.4.24 at concurrency 1: TCP at 15.3s,
    # HTTP 101 at 105.4s. The budget must leave room for setup contention.
    assert openclaw_agent._GATEWAY_READY_TIMEOUT_S >= 200


# ── cleanup under cancellation ────────────────────────────────────────────────

def _cleanup_finally_block() -> str:
    """The env-cleanup ``finally`` — not the later workspace-archive one."""
    source = inspect.getsource(backend.PawBenchBackend._run_agent_async)
    start = source.rindex("finally:", 0, source.index("await agent.teardown"))
    return source[start:source.index("transcript = agent.extract_transcript")]


def test_cleanup_survives_a_base_exception_from_teardown():
    finally_block = _cleanup_finally_block()
    # `except Exception` here let CancelledError skip env.stop() entirely.
    assert finally_block.count("except BaseException") == 2
    assert "except Exception:\n                pass" not in finally_block
    assert finally_block.index("agent.teardown") < finally_block.index("env.stop")


def test_cleanup_failures_are_reported_not_swallowed():
    finally_block = _cleanup_finally_block()
    assert "CLEANUP FAILED" in finally_block
    assert "cleanup_errors" in finally_block
    # The failure has to reach the result row, not just stdout.
    assert "run_error =" in finally_block


def test_teardown_gets_a_longer_lifecycle_lock_budget_than_start():
    # Abandoning a start costs one task; abandoning a teardown leaks a
    # container for the remainder of the run.
    assert (
        docker_env._NESTED_PODMAN_CLEANUP_LOCK_TIMEOUT
        > docker_env._NESTED_PODMAN_COMMAND_TIMEOUT
    )
    assert "timeout=_NESTED_PODMAN_CLEANUP_LOCK_TIMEOUT" in inspect.getsource(
        docker_env.DockerEnvironment._stop_sync
    )
    assert "timeout=" not in inspect.getsource(
        docker_env.DockerEnvironment._start_sync
    ).split("self._lifecycle_lock_acquire(")[1].split(")")[0]


# ── phase-scoped hard limit ───────────────────────────────────────────────────

class _Backend(backend.PawBenchBackend):
    def __init__(self):  # bypass the real __init__
        pass


class _Task:
    task_id = "t1"
    name = "t1"


def _run_phase_limits(inner, *, setup_limit, run_limit):
    b = _Backend()
    phase = {"name": "setup"}

    async def main():
        b._run_agent_async = inner  # type: ignore[assignment]
        return await b._run_with_phase_limits(
            _Task(), object(), {},
            setup_limit=setup_limit, run_limit=run_limit, phase=phase,
        )

    return asyncio.run(main()), phase


def test_setup_phase_has_its_own_bound():
    async def never_finishes(task, agent, config, *, arm_run_deadline=None):
        await asyncio.sleep(30)

    with pytest.raises(asyncio.TimeoutError):
        _run_phase_limits(never_finishes, setup_limit=1, run_limit=600)


def test_task_budget_starts_only_after_setup_returns():
    """A task queued behind other setups must not be cancelled for it."""
    started = time.monotonic()

    async def slow_setup_then_quick_run(task, agent, config, *, arm_run_deadline=None):
        await asyncio.sleep(1.5)          # queued + setting up
        arm_run_deadline()                # setup done: task budget starts now
        await asyncio.sleep(1.0)          # the graded run
        return "done"

    # run_limit alone (2s) is shorter than the total elapsed time (2.5s); the
    # single wait_for this replaced would have cancelled it.
    result, phase = _run_phase_limits(
        slow_setup_then_quick_run, setup_limit=10, run_limit=2
    )
    assert result == "done"
    assert phase["name"] == "run"
    assert time.monotonic() - started > 2.0


def test_run_phase_is_still_bounded_after_arming():
    async def runaway(task, agent, config, *, arm_run_deadline=None):
        arm_run_deadline()
        await asyncio.sleep(30)

    with pytest.raises(asyncio.TimeoutError):
        _run_phase_limits(runaway, setup_limit=30, run_limit=1)


def test_expired_phase_is_reported_truthfully():
    source = inspect.getsource(backend.PawBenchBackend.run_and_grade)
    assert 'phase["name"] == "setup"' in source
    assert "setup did not finish within" in source
    assert "Task exceeded hard wall-clock limit of" in source


def test_setup_and_run_deadlines_are_separate_in_the_runner():
    source = inspect.getsource(backend.PawBenchBackend.run_and_grade)
    # The single outer wait_for is what charged setup to the task budget.
    assert "asyncio.wait_for(" not in source
    assert "_run_with_phase_limits" in source


def test_setup_bound_covers_the_measured_worst_case_queue():
    # One OpenClaw setup measured at ~250s end to end. At harness concurrency 32
    # with a setup limit of 4 the last task waits ~(31/4)*250 + 250 ≈ 2190s, so a
    # smaller bound would cancel tasks that are merely queued — the very defect
    # this replaced.
    assert backend._SETUP_HARD_LIMIT_S >= 2400
