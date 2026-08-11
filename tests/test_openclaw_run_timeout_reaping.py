"""Regression tests for OpenClaw agent-run timeout handling and reaping.

Under nested Podman the container runs with ``--pid host``, so coreutils
``timeout`` puts ``openclaw agent`` in its own process group that survives
``podman rm``. A leaked tree keeps driving the model server and starves every
later task, and the resulting exec timeout used to destroy the run evidence.
"""

import asyncio
import inspect
import os
import signal
import subprocess
import time

import pytest

from pawbench.agents.impl import openclaw_agent
from pawbench.agents.impl.openclaw_agent import OpenClawAgent


def _agent() -> OpenClawAgent:
    return OpenClawAgent(model="custom/Qwen3.5-4B")


def _run_command(agent: OpenClawAgent, *, inner_timeout: int = 240) -> str:
    return agent._agent_run_command(
        provider_str="openai",
        api_key="EMPTY",
        agent_id=agent._agent_id(),
        session_id="pawbench-1-abcdef01",
        inner_timeout=inner_timeout,
        thinking_args="",
        escaped_message="'do the task'",
    )


def test_inner_run_timeout_escalates_to_sigkill():
    command = _run_command(_agent())

    # openclaw agent ignores SIGTERM while waiting on the model server, so a
    # plain `timeout Ns` never returns.
    assert f"--kill-after={openclaw_agent._INNER_RUN_KILL_GRACE_S}s" in command
    assert "timeout --kill-after" in command
    assert "openclaw agent " in command


def test_inner_run_keeps_the_task_budget_unchanged():
    # The per-task budget is benchmark semantics; only the kill escalation is
    # added.
    assert " 240s openclaw agent " in _run_command(_agent(), inner_timeout=240)
    assert " 1800s openclaw agent " in _run_command(_agent(), inner_timeout=1800)


def test_inner_run_records_real_pipeline_status():
    command = _run_command(_agent())

    # `| tee ... || true` masked the timeout exit status, so a task killed at
    # its budget was recorded as a clean success with an empty transcript.
    assert "|| true" not in command
    assert "${PIPESTATUS[0]}" in command
    assert openclaw_agent._RUN_STATUS_FILE in command


@pytest.mark.parametrize(
    "status_text,expected",
    [("124", True), ("137", True), ("124\n", True), ("0", False), ("1", False),
     ("", False), (None, False), ("garbage", False)],
)
def test_inner_run_timed_out_classification(status_text, expected):
    assert OpenClawAgent._inner_run_timed_out(status_text) is expected


def _fake_openclaw(tmp_path, *, sigterm_deaf: bool):
    script = tmp_path / "openclaw"
    trap = "trap '' TERM\n" if sigterm_deaf else ""
    script.write_text(f"#!/bin/bash\n{trap}sleep 60\n")
    script.chmod(0o755)
    return script


def _spawn_agent_tree(tmp_path, session_id, *, sigterm_deaf=False):
    script = _fake_openclaw(tmp_path, sigterm_deaf=sigterm_deaf)
    return subprocess.Popen(
        ["timeout", "60s", str(script), "agent", "--agent", "bench",
         "--session-id", session_id, "--message", "hi"],
    )


def _await_group(session_id):
    for _ in range(400):
        resolved = OpenClawAgent._run_group_for_session(session_id)
        if resolved is not None:
            return resolved
        time.sleep(0.01)
    return None


def _hard_kill_group(process):
    """Leave no descendant behind: `timeout` leads its own process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    for _ in range(200):
        if not OpenClawAgent._process_group_members(process.pid):
            return
        time.sleep(0.01)
    raise AssertionError(f"test process group {process.pid} survived cleanup")


def test_run_group_resolution_requires_the_session_marker(tmp_path):
    session_id = "pawbench-unit-marker"
    process = _spawn_agent_tree(tmp_path, session_id)
    # The host `podman exec` client and the outer exec wrapper carry the same
    # marker inside a single shell string, but they live in the runner's own
    # process group and must never be resolved.
    wrapper = subprocess.Popen(
        ["bash", "-c",
         f"sleep 60 # --session-id {session_id} --message hi openclaw agent"],
    )
    try:
        resolved = _await_group(session_id)
        assert resolved is not None
        pgid, owner_token = resolved
        # coreutils timeout leads the new process group.
        assert pgid == process.pid
        assert pgid != wrapper.pid
        assert owner_token == OpenClawAgent._process_owner_token(process.pid)
        # A different attempt's session must never resolve to this group.
        assert OpenClawAgent._run_group_for_session("pawbench-other-marker") is None
    finally:
        wrapper.kill()
        wrapper.wait(timeout=5)
        _hard_kill_group(process)


def test_reap_nested_agent_run_kills_a_real_leaked_tree(tmp_path, monkeypatch):
    """A SIGTERM-deaf agent tree must still be gone when the task ends."""
    session_id = "pawbench-unit-deaf"
    process = _spawn_agent_tree(tmp_path, session_id, sigterm_deaf=True)
    try:
        assert _await_group(session_id) is not None
        monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
        monkeypatch.setattr(openclaw_agent, "_RUN_REAP_GRACE_S", 2)
        asyncio.run(_agent()._reap_nested_agent_run(session_id))
        assert OpenClawAgent._run_group_for_session(session_id) is None
        assert not OpenClawAgent._process_group_members(process.pid)
    finally:
        _hard_kill_group(process)


def test_reap_is_a_no_op_outside_nested_podman(monkeypatch):
    monkeypatch.delenv("PAWBENCH_PODMAN_NESTED", raising=False)
    called = []
    monkeypatch.setattr(
        OpenClawAgent,
        "_run_group_for_session",
        staticmethod(lambda session_id: called.append(session_id)),
    )
    asyncio.run(_agent()._reap_nested_agent_run("pawbench-unused"))
    assert called == []


def test_reap_never_signals_a_foreign_member_of_a_recycled_group(monkeypatch):
    """PIDs are recycled: the run pgid can already carry someone else's process.

    Such a member must never be signalled, and must never hold the reaper open
    either -- it will not exit, so treating it as ours turns every teardown of
    a recycled group number into a hard failure.
    """
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    foreign = (999, "0::/some-other-container.scope")
    mine = (12345, "0::/this-task.scope")
    monkeypatch.setattr(
        OpenClawAgent,
        "_run_group_for_session",
        staticmethod(lambda _session: (4242, mine)),
    )
    monkeypatch.setattr(
        OpenClawAgent,
        "_process_group_members",
        staticmethod(lambda _pgid: [(4242, os.getuid(), "openclaw agent")]),
    )
    monkeypatch.setattr(
        OpenClawAgent, "_process_owner_token", staticmethod(lambda _pid: foreign)
    )
    sent = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd - 1000, sig))
    )

    asyncio.run(_agent()._reap_nested_agent_run("pawbench-unit-foreign"))
    assert sent == []


def test_reap_refuses_to_signal_a_group_with_no_recorded_owner(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    monkeypatch.setattr(
        OpenClawAgent,
        "_run_group_for_session",
        staticmethod(lambda _session: (4242, None)),
    )
    monkeypatch.setattr(
        OpenClawAgent,
        "_process_group_members",
        staticmethod(lambda _pgid: [(4242, os.getuid(), "openclaw agent")]),
    )
    monkeypatch.setattr(
        OpenClawAgent,
        "_process_owner_token",
        staticmethod(lambda _pid: (12345, "0::/task.scope")),
    )
    with pytest.raises(RuntimeError, match="without a recorded owner"):
        asyncio.run(_agent()._reap_nested_agent_run("pawbench-unit-noowner"))


def test_reap_escalates_to_children_without_the_session_marker(monkeypatch):
    """`timeout` and the launcher may exit while openclaw-agent children live."""
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    pgid = 5150
    uid = os.getuid()
    token = (12345, "0::/task.scope")
    snapshots = iter([
        [(pgid, uid, "timeout 240s openclaw agent --session-id pawbench-s ")],
        [(pgid + 1, uid, "openclaw-agent")],
        [],
    ])
    monkeypatch.setattr(
        OpenClawAgent, "_run_group_for_session", staticmethod(lambda _s: (pgid, token))
    )
    monkeypatch.setattr(
        OpenClawAgent, "_process_group_members", staticmethod(lambda _p: next(snapshots))
    )
    monkeypatch.setattr(
        OpenClawAgent, "_process_owner_token", staticmethod(lambda _p: token)
    )
    sent = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd - 1000, sig))
    )

    agent = _agent()
    waits = iter([False, True])

    async def fake_wait(_pgid, _timeout, _owner_token=None):
        return next(waits)

    monkeypatch.setattr(agent, "_wait_run_group_closed", fake_wait)
    asyncio.run(agent._reap_nested_agent_run("pawbench-s"))
    assert sent == [(pgid, signal.SIGTERM), (pgid + 1, signal.SIGKILL)]


def test_reap_raises_when_a_survivor_remains(monkeypatch):
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    pgid = 5151
    token = (12345, "0::/task.scope")
    member = [(pgid, os.getuid(), "openclaw agent --session-id pawbench-s ")]
    monkeypatch.setattr(
        OpenClawAgent, "_run_group_for_session", staticmethod(lambda _s: (pgid, token))
    )
    monkeypatch.setattr(
        OpenClawAgent, "_process_group_members", staticmethod(lambda _p: member)
    )
    monkeypatch.setattr(
        OpenClawAgent, "_process_owner_token", staticmethod(lambda _p: token)
    )
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: None)

    agent = _agent()

    async def never_closed(_pgid, _timeout, _owner_token=None):
        return False

    monkeypatch.setattr(agent, "_wait_run_group_closed", never_closed)
    with pytest.raises(RuntimeError, match="agent run cleanup incomplete"):
        asyncio.run(agent._reap_nested_agent_run("pawbench-s"))


def test_reap_never_uses_broad_kills():
    sources = "".join(
        inspect.getsource(function)
        for function in (
            OpenClawAgent._reap_nested_agent_run,
            OpenClawAgent._signal_owned_run_members,
            OpenClawAgent._validate_owned_run_group,
            OpenClawAgent._run_group_for_session,
        )
    )
    assert "os.killpg(" not in sources
    assert "pkill" not in sources
    assert "pidfd_send_signal" in sources


def test_signalling_tolerates_a_member_exiting_mid_reap(monkeypatch):
    """A member that exits between the snapshot and pidfd_open is not hostile."""
    token = (12345, "0::/task.scope")
    members = [(700, os.getuid(), "openclaw agent --session-id pawbench-s "),
               (701, os.getuid(), "openclaw-agent")]
    tokens = {700: token, 701: None}
    monkeypatch.setattr(
        OpenClawAgent, "_process_owner_token", staticmethod(lambda pid: tokens[pid])
    )
    closed = []
    sent = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd - 1000, sig))
    )

    OpenClawAgent._signal_owned_run_members(
        700,
        members,
        signal.SIGTERM,
        session_id="pawbench-s",
        require_session_marker=True,
        owner_token=token,
    )
    assert sent == [(700, signal.SIGTERM)]
    # 701 is already gone, so it is dropped before pidfd_open: no fd is opened
    # for it and none has to be closed.
    assert closed == [1700]


def test_run_reports_an_unreapable_tree_without_destroying_evidence():
    source = inspect.getsource(OpenClawAgent.run)
    # Raising out of run() skips the backend's workspace collection, which is
    # exactly the evidence loss being fixed.
    assert "reap_error = str(exc)" in source
    assert "not reap_error" in source
    assert 'reap_error or run_error' in source
    assert source.index("reap_error = str(exc)") < source.index(
        "_sync_workspace_to_output"
    )


def test_run_reaps_and_preserves_evidence_on_exec_timeout(monkeypatch):
    """A timed-out run must still collect its transcript instead of raising."""
    source = inspect.getsource(OpenClawAgent.run)
    assert "except TimeoutError" in source
    assert "_reap_nested_agent_run" in source
    # The reap must run before the artifacts are read, and collection must
    # follow the timeout branch rather than propagating out of run().
    assert source.index("except TimeoutError") < source.index(
        "_reap_nested_agent_run"
    ) < source.index("_sync_workspace_to_output")


def test_teardown_reaps_as_a_backstop():
    source = inspect.getsource(OpenClawAgent.teardown)
    assert "_reap_nested_agent_run" in source
    assert "_run_session_id" in source


def test_session_ids_are_unique_across_concurrent_attempts():
    source = inspect.getsource(OpenClawAgent.run)
    assert "uuid.uuid4()" in source

    seen = {
        f"pawbench-{int(time.time() * 1000)}-{__import__('uuid').uuid4().hex[:8]}"
        for _ in range(500)
    }
    assert len(seen) == 500


def test_backend_reports_agent_timeouts_honestly():
    from pawbench import backend

    source = inspect.getsource(backend.PawBenchBackend._run_agent_async)
    assert 'run_result.get("timed_out"' in source
    assert '"timed_out": run_timed_out' in source
    assert "timed_out=run_timed_out" in source
    assert '"timed_out": False' not in source
