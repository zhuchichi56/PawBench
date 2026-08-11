import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

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


def test_setup_limiter_is_shared_across_runner_threads(monkeypatch):
    monkeypatch.setenv("PAWBENCH_OPENCLAW_SETUP_CONCURRENCY", "4")
    monkeypatch.setattr(openclaw_agent, "_OPENCLAW_SETUP_SEMAPHORE", None)
    monkeypatch.setattr(openclaw_agent, "_OPENCLAW_SETUP_LIMIT", None)
    active = 0
    maximum = 0
    counter_lock = threading.Lock()

    async def fake_setup_limited(self, environment):
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
        await asyncio.sleep(0.03)
        with counter_lock:
            active -= 1

    monkeypatch.setattr(OpenClawAgent, "_setup_limited", fake_setup_limited)

    def run_one(_index):
        agent = OpenClawAgent(model="custom/Qwen3.5-4B")
        asyncio.run(agent.setup(None))

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(run_one, range(24)))

    assert maximum == 4


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_setup_limiter_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("PAWBENCH_OPENCLAW_SETUP_CONCURRENCY", value)

    async def exercise():
        openclaw_agent._get_openclaw_setup_semaphore()

    with pytest.raises(ValueError):
        asyncio.run(exercise())



def test_openclaw_control_commands_use_contention_safe_deadline():
    sources = "".join(
        inspect.getsource(method)
        for method in (
            OpenClawAgent.install,
            OpenClawAgent._setup_limited,
            OpenClawAgent._start_gateway,
            OpenClawAgent._configure_openclaw_json,
            OpenClawAgent.run,
            OpenClawAgent.post_run_collect,
            OpenClawAgent.teardown,
        )
    )
    import re

    assert not re.search(r"timeout=(?:5|10|15|30)(?!\d)", sources)
    assert "_OPENCLAW_CONTROL_TIMEOUT" in sources
    assert "openclaw config set agents.defaults.workspace" not in sources
    assert "agents_cfg['workspace']" in inspect.getsource(
        OpenClawAgent._configure_openclaw_json
    )



def test_openclaw_control_timeout_names_resolve():
    import ast
    from pathlib import Path

    source = Path(openclaw_agent.__file__).read_text()
    tree = ast.parse(source)
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id.startswith("_OPENCLAW_CONTROL_TIMEOUT")
    }
    assert referenced == {"_OPENCLAW_CONTROL_TIMEOUT"}
    assert referenced <= assigned

def test_configure_openclaw_failure_is_fatal():
    agent = OpenClawAgent(
        model="custom/Qwen3.5-4B",
        api_key="EMPTY",
        base_url="http://127.0.0.1:30000/v1",
    )

    class Environment:
        async def write_file(self, path, content):
            return True

        async def execute_command(self, command, timeout=None):
            return {"returncode": 1, "stdout": "", "stderr": "broken config"}

    with pytest.raises(RuntimeError, match="patch failed"):
        asyncio.run(
            agent._configure_openclaw_json(
                Environment(),
                api_key="EMPTY",
                base_url="http://127.0.0.1:30000/v1",
                model_identifier="custom/Qwen3.5-4B",
                explicit_base_url=True,
            )
        )

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
    signal_source = inspect.getsource(OpenClawAgent._signal_owned_gateway_members)

    assert "setsid" in setup_source
    assert "_gateway_group_for_port" in setup_source
    assert "pidfd_open" in signal_source
    assert "pidfd_send_signal" in signal_source
    all_cleanup_source = setup_source + cleanup_source + nested_cleanup_source + signal_source
    assert "os.killpg(" not in all_cleanup_source
    assert "pkill" not in all_cleanup_source


def test_nested_gateway_cleanup_escalates_owned_helper_process(monkeypatch):
    import os
    import signal

    pgid = 42420
    uid = os.getuid()
    owner_token = (12345, "0::/libpod-owned.scope")
    initial = [(pgid, uid, "openclaw gateway"), (pgid + 1, uid, "npm install deps")]
    helper = [(pgid + 1, uid, "npm install deps")]
    snapshots = iter([initial, helper, []])

    agent = OpenClawAgent(model="custom/Qwen3.5-4B")
    agent._gateway_pgid = pgid
    agent._gateway_owner_token = owner_token
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    monkeypatch.setattr(agent, "_port_is_open", lambda _port: False)
    monkeypatch.setattr(agent, "_process_group_members", lambda _pgid: next(snapshots))
    monkeypatch.setattr(
        OpenClawAgent,
        "_process_owner_token",
        staticmethod(lambda _pid: owner_token),
    )
    monkeypatch.setattr(os, "getsid", lambda _pid: pgid)

    wait_results = iter([False, True])

    async def fake_wait(_pgid, _timeout):
        return next(wait_results)

    monkeypatch.setattr(agent, "_wait_nested_gateway_closed", fake_wait)

    opened = []
    closed = []
    sent = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: opened.append(pid) or pid + 1000)
    monkeypatch.setattr(os, "close", lambda pidfd: closed.append(pidfd))
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda pidfd, sig: sent.append((pidfd, sig)),
    )

    class Environment:
        async def execute_command(self, command, timeout=None):
            return {"returncode": 0, "stdout": "", "stderr": ""}

    asyncio.run(agent._kill_gateway(Environment()))
    assert opened == [pgid, pgid + 1, pgid + 1]
    assert closed == [pgid + 1000, pgid + 1001, pgid + 1001]
    assert sent == [
        (pgid + 1000, signal.SIGTERM),
        (pgid + 1001, signal.SIGTERM),
        (pgid + 1001, signal.SIGKILL),
    ]
    assert agent._gateway_pgid is None
    assert agent._gateway_owner_token is None


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
        token = agent._process_owner_token(process.pid)
        agent._gateway_owner_token = token
        monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
        monkeypatch.setattr(agent, "_port_is_open", lambda _port: False)

        class Environment:
            async def execute_command(self, command, timeout=None):
                return {"returncode": 0, "stdout": "", "stderr": ""}

        asyncio.run(agent._kill_gateway(Environment()))
        process.wait(timeout=5)
        assert agent._gateway_pgid is None
        # Ownership-scoped: the node recycles PIDs, so an orphaned group may
        # already carry a foreign process under this number. Only this task's
        # own members have to be gone.
        owned, _foreign = OpenClawAgent._partition_gateway_members(
            process.pid, OpenClawAgent._process_group_members(process.pid), token
        )
        assert not owned
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait(timeout=5)


def test_gateway_cleanup_ignores_a_foreign_member_of_a_recycled_group(monkeypatch):
    """Observed on the benchmark node: PID reuse puts a stranger in our group.

    A brand-new ``setsid`` session was created with a number an orphaned group
    still used, so the group contained a reparented ``sed`` owned by nobody we
    track. Signalling it would be a cross-task kill; waiting for it to exit
    would hang teardown forever and leak the container.
    """
    import os
    import signal

    pgid = 31337
    uid = os.getuid()
    mine = (12345, "0::/this-task.scope")
    foreign_token = (999, "0::/")
    tokens = {pgid: mine, pgid + 3: foreign_token}
    members = [(pgid, uid, "openclaw gateway --port 28088"),
               (pgid + 3, uid, "sed s/^$/unknown/ ")]
    survivor = [(pgid + 3, uid, "sed s/^$/unknown/ ")]
    snapshots = iter([members, survivor, survivor, survivor, survivor])

    agent = OpenClawAgent(model="custom/Qwen3.5-4B")
    agent._gateway_pgid = pgid
    agent._gateway_owner_token = mine
    monkeypatch.setenv("PAWBENCH_PODMAN_NESTED", "1")
    monkeypatch.setattr(agent, "_port_is_open", lambda _port: False)
    monkeypatch.setattr(agent, "_process_group_members", lambda _pgid: next(snapshots))
    monkeypatch.setattr(
        OpenClawAgent,
        "_process_owner_token",
        staticmethod(lambda pid: tokens[pid]),
    )
    monkeypatch.setattr(os, "getsid", lambda _pid: pgid)

    sent = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd - 1000, sig))
    )

    class Environment:
        async def execute_command(self, command, timeout=None):
            return {"returncode": 0, "stdout": "", "stderr": ""}

    asyncio.run(agent._kill_gateway(Environment()))
    # Only the gateway is signalled, and the surviving stranger does not turn a
    # successful teardown into "gateway cleanup incomplete".
    assert sent == [(pgid, signal.SIGTERM)]
    assert agent._gateway_pgid is None
