# -*- coding: utf-8 -*-
"""OpenClaw agent implementation for pawbench evaluation."""

import asyncio
import itertools
import json
import os
import shlex
import signal
import socket
import time
from typing import Any, Dict, List

from pawbench.agents.base import ContainerAgent
from pawbench.agents.constants import AGENT_WORKSPACE
from pawbench.envs.base import BaseEnvironment
from pawbench.llm.model_config import get_model_config, ProviderType


_GATEWAY_PORT = 18789
# OpenClaw also opens auxiliary listeners near the configured gateway port
# (for example browser control at gateway+2). Keep task port bundles disjoint.
_OPENCLAW_PORTS = itertools.count(28088, 20)
_OPENCLAW_SETUP_LIMIT_ENV = "PAWBENCH_OPENCLAW_SETUP_CONCURRENCY"
_OPENCLAW_SETUP_LIMIT_DEFAULT = 8
_OPENCLAW_SETUP_SEMAPHORE: asyncio.Semaphore | None = None
_OPENCLAW_SETUP_LOOP: asyncio.AbstractEventLoop | None = None
_OPENCLAW_SETUP_LIMIT: int | None = None


def _get_openclaw_setup_semaphore() -> asyncio.Semaphore:
    """Return the process-local limiter for container and gateway setup only."""
    global _OPENCLAW_SETUP_LIMIT, _OPENCLAW_SETUP_LOOP, _OPENCLAW_SETUP_SEMAPHORE

    raw_limit = os.environ.get(
        _OPENCLAW_SETUP_LIMIT_ENV, str(_OPENCLAW_SETUP_LIMIT_DEFAULT)
    )
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(f"{_OPENCLAW_SETUP_LIMIT_ENV} must be an integer") from exc
    if limit < 1:
        raise ValueError(f"{_OPENCLAW_SETUP_LIMIT_ENV} must be positive")

    loop = asyncio.get_running_loop()
    if (
        _OPENCLAW_SETUP_SEMAPHORE is None
        or _OPENCLAW_SETUP_LOOP is not loop
        or _OPENCLAW_SETUP_LIMIT != limit
    ):
        _OPENCLAW_SETUP_SEMAPHORE = asyncio.Semaphore(limit)
        _OPENCLAW_SETUP_LOOP = loop
        _OPENCLAW_SETUP_LIMIT = limit
    return _OPENCLAW_SETUP_SEMAPHORE


class OpenClawAgent(ContainerAgent):
    """OpenClaw agent for pawbench evaluation.

    Runs tasks via ``openclaw agent --message``.  Pre-built image
    ``copawbench-openclaw:latest`` already has openclaw 2026.4.24 installed;
    the slow-path installs from npm when only a plain base image is used.
    """

    def __init__(self, name: str = "openclaw", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._gateway_port = next(_OPENCLAW_PORTS)
        self._gateway_pgid: int | None = None

    def _agent_id(self) -> str:
        """Return a task-unique, filesystem-safe OpenClaw agent ID."""
        model = self.config.get("model", "dashscope/qwen3.6-plus")
        slug = model.replace("/", "-").replace(".", "-").lower()[:36]
        return f"bench-{slug}-{self._gateway_port}"

    def _openclaw_model_id(self, model_identifier: str) -> str:
        """Translate a pawbench model identifier to an openclaw model identifier.

        Two translations are applied:

        1. **Alias expansion** – short names like ``opus-4.6`` are expanded to
           their canonical ``provider/model`` form via ``model_config`` aliases
           before any provider-specific rewrite.

        2. **Provider mapping**:
           - ``dashscope/<model>`` → ``qwen/<model>`` (openclaw's native Qwen
             plugin uses ``"qwen"`` as its provider key)
           - ``custom/<model>`` → ``custom-<hostname>/<model>`` where
             ``<hostname>`` is derived from the configured base URL (same logic
             as ``_configure_openclaw_json``).  This ensures ``openclaw agents
             add --model <id>`` registers the agent against the provider entry
             we write into openclaw.json, not the default ``openai`` provider.
        """
        from pawbench.llm.model_config import ModelConfigManager
        # Expand aliases (e.g. "opus-4.6" → "custom/openai.claude-opus-4-6")
        aliases = ModelConfigManager.get_available_models()
        expanded = aliases.get(model_identifier.strip(), model_identifier)

        parts = expanded.split("/", 1)
        if len(parts) == 2:
            provider_str = parts[0].lower()
            model_name = parts[1]
            if provider_str == "dashscope":
                return f"qwen/{model_name}"
            if provider_str == "custom":
                base_url = self.config.get("base_url") or os.environ.get("CUSTOM_BASE_URL", "")
                if base_url:
                    from urllib.parse import urlparse
                    hostname = urlparse(base_url).hostname or "custom"
                    openclaw_provider = "custom-" + hostname.replace(".", "-")
                    return f"{openclaw_provider}/{model_name}"
        return expanded

    def _resolve_api_key(self, model_config=None) -> str:
        return (
            self.config.get("api_key")
            or (model_config.api_key if model_config else None)
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        ) or ""

    # ── installation ──────────────────────────────────────────────────────────

    async def install(self, environment: BaseEnvironment) -> None:
        check = await environment.execute_command("command -v openclaw", timeout=10)
        if check.get("returncode", 1) == 0 and check.get("stdout", "").strip():
            return
        await environment.execute_command(
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y git curl && "
            "node --version | grep -qE 'v(2[2-9]|[3-9][0-9])' || "
            "(curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs)",
            timeout=180,
        )
        await environment.execute_command(
            "npm install -g openclaw@2026.4.24", timeout=180
        )

    # ── setup ─────────────────────────────────────────────────────────────────

    async def setup(self, environment: BaseEnvironment) -> None:
        # Limit only environment/CLI/gateway initialization.  The permit is
        # released before run(), so ready tasks and model requests retain the
        # benchmark's full harness queue width.
        async with _get_openclaw_setup_semaphore():
            await self._setup_limited(environment)

    async def _setup_limited(self, environment: BaseEnvironment) -> None:
        await self.install(environment)

        # Create workspace and openclaw state directories.
        # No symlinks needed — the agent will be bound to AGENT_WORKSPACE
        # directly via ``openclaw agents add --workspace``.
        await environment.execute_command(
            f"mkdir -p {AGENT_WORKSPACE}/output {AGENT_WORKSPACE}/sessions && "
            "mkdir -p /root/.openclaw/agents",
            timeout=15,
        )

        # Wipe the auth profile baked into the image by ``openclaw onboard``
        # in Dockerfile.pawbench-openclaw (Step "Pre-configure openclaw for
        # Alibaba Dashscope"), which stores the placeholder key
        # ``sk-build-placeholder``.  If left in place, the built-in qwen
        # plugin loads that profile at gateway startup and serves it as the
        # active credential for any qwen/<model> request — bypassing the
        # apiKey we write into openclaw.json and producing
        # "401 Incorrect API key".  We also disable the qwen plugin in
        # ``_configure_openclaw_json`` below; this cleanup is the
        # belt-and-suspenders companion to that, ensuring no stale profile
        # gets re-discovered by a future plugin scan.
        await environment.execute_command(
            "rm -f /root/.openclaw/auth-profiles.json "
            "      /root/.openclaw/auth/profiles.json "
            "      /root/.openclaw/auth/*.json 2>/dev/null || true",
            timeout=10,
        )

        model_identifier = self.config.get("model", "dashscope/qwen3.6-plus")
        model_config = get_model_config(model_identifier)
        api_key = self._resolve_api_key(model_config)
        base_url = self.config.get("base_url") or model_config.base_url or ""
        provider_str = (
            model_identifier.split("/", 1)[0].lower()
            if "/" in model_identifier else "openai"
        )

        # Seed openclaw.json before agents add (CLI reads it even without gateway).
        await self._configure_openclaw_json(
            environment,
            api_key=api_key,
            base_url=base_url,
            model_identifier=model_identifier,
            explicit_base_url=bool(self.config.get("base_url")),
        )

        # Kill any gateway BEFORE agents add / config patches.
        #
        # Root cause of EADDRINUSE + PluginLoadFailureError:
        #   1. Starting gateway, then running ``agents add``, rewrites openclaw.json.
        #   2. The gateway's config watcher hot-reloads and may spawn a second listener.
        #   3. setup() then tries to start another gateway → EADDRINUSE on 18789.
        #
        # Fix (matches examples/open_source openclaw_agent): no gateway during
        # config writes; start exactly once after all patches are on disk.
        await self._kill_gateway(environment)

        # Each benchmark attempt owns an ephemeral container plus a unique
        # agent/config identity.  There is therefore no stale agent to delete;
        # invoking ``agents delete`` in a fresh container can block behind
        # OpenClaw plugin initialization and caused queue-wide setup timeouts.
        agent_id = self._agent_id()
        openclaw_model = self._openclaw_model_id(model_identifier)
        env_prefix = self._make_key_env(provider_str, api_key)
        add_result = await environment.execute_command(
            f"{env_prefix}"
            f"openclaw agents add {shlex.quote(agent_id)} "
            f"--model {shlex.quote(openclaw_model)} "
            f"--workspace {shlex.quote(AGENT_WORKSPACE)} "
            "--non-interactive",
            timeout=300,
        )
        if add_result.get("returncode", 1) != 0:
            import logging
            logging.getLogger(__name__).error(
                "openclaw agents add failed (exit %s) — agent=%s model=%s\nstdout: %s\nstderr: %s",
                add_result.get("returncode"),
                agent_id,
                model_identifier,
                (add_result.get("stdout") or "")[:500],
                (add_result.get("stderr") or "")[:500],
            )

        # ``openclaw agents add`` copies the image-baked auth-profiles.json
        # (containing ``sk-build-placeholder``) into the per-agent directory.
        # That file takes priority over the apiKey written in openclaw.json and
        # causes HTTP 401 on every API call.  Overwrite it with the real API key
        # so the gateway uses the correct credential at runtime.
        agent_id_lower = agent_id.lower()
        _provider_for_auth = openclaw_model.split("/")[0]
        auth_profiles_content = json.dumps({
            "version": 1,
            "profiles": {
                f"{_provider_for_auth}:default": {
                    "type": "api_key",
                    "provider": _provider_for_auth,
                    "key": api_key,
                }
            },
        }, indent=2)
        await environment.execute_command(
            f"mkdir -p /root/.openclaw/agents/{agent_id_lower}/agent",
            timeout=10,
        )
        await environment.write_file(
            f"/root/.openclaw/agents/{agent_id_lower}/agent/auth-profiles.json",
            auth_profiles_content,
        )

        # Point openclaw's global default workspace at the benchmark path so
        # the ACPX runtime routes all file I/O there.  ``agents add --workspace``
        # only sets per-agent metadata; this config key controls where write/read
        # operations actually land.
        await environment.execute_command(
            f"{env_prefix}"
            f"openclaw config set agents.defaults.workspace {shlex.quote(AGENT_WORKSPACE)} "
            "2>/dev/null || true",
            timeout=15,
        )

        # ``agents add`` rewrites openclaw.json and may drop fields like
        # ``gateway.mode``.  Patch them back so the gateway reads a fully
        # correct config on its fresh start below.
        await environment.write_file(
            "/tmp/patch_gateway_mode.py",
            "import json, os\n"
            "p = '/root/.openclaw/openclaw.json'\n"
            "d = json.load(open(p)) if os.path.exists(p) else {}\n"
            "d.setdefault('gateway', {})['mode'] = 'local'\n"
            f"d['gateway']['port'] = {self._gateway_port}\n"
            "json.dump(d, open(p, 'w'), indent=2)\n"
            "print('gateway.mode ensured')\n",
        )
        await environment.execute_command(
            "python3 /tmp/patch_gateway_mode.py",
            timeout=10,
        )

        # Re-apply provider/model config after ``agents add`` may have
        # overwritten provider or model-entry fields.
        await self._configure_openclaw_json(
            environment,
            api_key=api_key,
            base_url=base_url,
            model_identifier=model_identifier,
            explicit_base_url=bool(self.config.get("base_url")),
        )

        # Start exactly one gateway after all config is final.
        await self._start_gateway(
            environment, api_key=api_key, provider_str=provider_str
        )

        # ``openclaw agents add`` writes BOOTSTRAP.md (the onboarding wizard
        # trigger) plus SOUL.md, IDENTITY.md, HEARTBEAT.md, TOOLS.md, USER.md
        # (the agent's shipped personality/capability files) into the workspace.
        #
        # We ONLY remove BOOTSTRAP.md to prevent the onboarding wizard from
        # intercepting the first benchmark message.  SOUL.md, IDENTITY.md, and
        # the other files are openclaw's own native capability hints; removing
        # them artificially disadvantages openclaw relative to other agents and
        # is not a fair evaluation baseline.  They also differ from the harness
        # "prompt helps" that copaw/hermes receive — they are part of the agent
        # itself, not injected by the benchmark harness.
        await environment.execute_command(
            f"rm -f {shlex.quote(AGENT_WORKSPACE)}/BOOTSTRAP.md",
            timeout=10,
        )

    # ── openclaw.json configuration ───────────────────────────────────────────

    # Map from pawbench/openclaw provider id → openclaw ``api`` protocol type.
    # Everything not listed here uses ``openai-completions`` (the safe default
    # for any OpenAI-compatible endpoint, including DashScope).
    _PROVIDER_API_TYPE: Dict[str, str] = {
        "anthropic": "anthropic-messages",
        "google":    "google-generative-ai",
        "ollama":    "ollama",
    }

    # Per-provider env var that carries the API key.
    _PROVIDER_KEY_ENV: Dict[str, str] = {
        "dashscope": "DASHSCOPE_API_KEY",
        "qwen":      "DASHSCOPE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }

    # Per-provider set of env vars to export when calling the openclaw CLI.
    # DashScope needs all three legacy names; Anthropic/Google need their own
    # native var plus OPENAI_API_KEY as a safety net for openclaw internals.
    _PROVIDER_CLI_ENVS: Dict[str, List[str]] = {
        "dashscope": ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"],
        "qwen":      ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
        "google":    ["GOOGLE_API_KEY", "OPENAI_API_KEY"],
        "gemini":    ["GOOGLE_API_KEY", "OPENAI_API_KEY"],
    }

    def _make_key_env(self, provider_str: str, api_key: str) -> str:
        """Return 'export VAR=key && export VAR2=key && ' for all env vars this provider needs."""
        env_vars = self._PROVIDER_CLI_ENVS.get(provider_str, ["OPENAI_API_KEY"])
        return " && ".join(f"export {v}={shlex.quote(api_key)}" for v in env_vars) + " && "

    @staticmethod
    def _process_group_members(pgid: int) -> list[tuple[int, int, str]]:
        """Return (pid, uid, cmdline) for one host-visible process group."""
        members: list[tuple[int, int, str]] = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                raw = open(f"/proc/{entry.name}/stat", encoding="utf-8").read()
                fields = raw[raw.rfind(")") + 2 :].split()
                if fields[0] == "Z" or int(fields[2]) != pgid:  # state, ppid, pgrp
                    continue
                uid = os.stat(f"/proc/{entry.name}").st_uid
                cmd = (
                    open(f"/proc/{entry.name}/cmdline", "rb")
                    .read()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                )
                members.append((int(entry.name), uid, cmd))
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        return members

    @staticmethod
    def _port_is_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @classmethod
    def _gateway_group_for_port(cls, port: int) -> int | None:
        """Resolve the host PID group that owns one OpenClaw listener."""
        inodes: set[str] = set()
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                lines = open(table, encoding="utf-8").read().splitlines()[1:]
            except (FileNotFoundError, PermissionError):
                continue
            for line in lines:
                fields = line.split()
                try:
                    local_port = int(fields[1].rsplit(":", 1)[1], 16)
                except (IndexError, ValueError):
                    continue
                if local_port == port and fields[3] == "0A":
                    inodes.add(fields[9])
        if not inodes:
            return None

        groups: set[int] = set()
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                if os.stat(f"/proc/{entry.name}").st_uid != os.getuid():
                    continue
                fd_dir = f"/proc/{entry.name}/fd"
                owns_listener = any(
                    os.readlink(fd.path) in {f"socket:[{inode}]" for inode in inodes}
                    for fd in os.scandir(fd_dir)
                )
                if not owns_listener:
                    continue
                cmd = (
                    open(f"/proc/{entry.name}/cmdline", "rb")
                    .read()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                )
                if "openclaw" not in cmd.lower():
                    continue
                raw = open(f"/proc/{entry.name}/stat", encoding="utf-8").read()
                groups.add(int(raw[raw.rfind(")") + 2 :].split()[2]))
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
                continue
        if len(groups) > 1:
            raise RuntimeError(
                f"multiple OpenClaw process groups own gateway port {port}: {sorted(groups)}"
            )
        return next(iter(groups)) if groups else None

    async def _wait_nested_gateway_closed(self, pgid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._process_group_members(pgid) and not self._port_is_open(
                self._gateway_port
            ):
                return True
            await asyncio.sleep(0.1)
        return not self._process_group_members(pgid) and not self._port_is_open(
            self._gateway_port
        )

    async def _kill_nested_gateway(self, pgid: int) -> None:
        members = self._process_group_members(pgid)
        bad = [
            item
            for item in members
            if item[1] != os.getuid() or "openclaw" not in item[2].lower()
        ]
        if bad:
            raise RuntimeError(f"refusing to signal unowned gateway group {pgid}: {bad}")
        if members:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if not await self._wait_nested_gateway_closed(pgid, 8):
                members = self._process_group_members(pgid)
                bad = [
                    item
                    for item in members
                    if item[1] != os.getuid() or "openclaw" not in item[2].lower()
                ]
                if bad:
                    raise RuntimeError(
                        f"gateway group {pgid} changed before escalation: {bad}"
                    )
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if not await self._wait_nested_gateway_closed(pgid, 5):
            raise RuntimeError(
                f"gateway cleanup incomplete: pgid={pgid} port={self._gateway_port} "
                f"members={self._process_group_members(pgid)}"
            )

    async def _kill_gateway(self, environment: BaseEnvironment) -> None:
        """Stop exactly this task's gateway and prove its listener is closed."""
        if os.environ.get("PAWBENCH_PODMAN_NESTED") == "1":
            if self._gateway_pgid is not None:
                await self._kill_nested_gateway(self._gateway_pgid)
                self._gateway_pgid = None
            elif self._port_is_open(self._gateway_port):
                raise RuntimeError(
                    f"gateway port {self._gateway_port} is occupied without an owned process group"
                )
            try:
                await environment.execute_command(
                    "rm -f /tmp/openclaw_gateway.pgid", timeout=5
                )
            except Exception:
                pass
            return

        result = await environment.execute_command(
            "pgid=$(cat /tmp/openclaw_gateway.pgid 2>/dev/null || true); "
            "if [ -n \"$pgid\" ]; then "
            "  kill -TERM -- -\"$pgid\" 2>/dev/null || true; "
            "  for i in $(seq 1 80); do kill -0 -- -\"$pgid\" 2>/dev/null || break; sleep 0.1; done; "
            "  kill -KILL -- -\"$pgid\" 2>/dev/null || true; "
            "fi; rm -f /tmp/openclaw_gateway.pgid",
            timeout=15,
        )
        if result.get("returncode", 1) != 0:
            raise RuntimeError(f"gateway cleanup failed: {result}")
        self._gateway_pgid = None

    async def _wait_gateway_ready(
        self, environment: BaseEnvironment, *, port: int | None = None
    ) -> None:
        port = self._gateway_port if port is None else port
        wait_cmd = (
            f'python3 -c "'
            "import socket, time; "
            f"deadline=time.time()+45; ok=False\n"
            "while time.time()<deadline:\n"
            f"  s=socket.socket(); s.settimeout(0.3)\n"
            f"  try: s.connect(('127.0.0.1',{port})); ok=True; break\n"
            f"  except Exception: time.sleep(0.3)\n"
            f"  finally:\n"
            f"    try: s.close()\n"
            f"    except Exception: pass\n"
            "print('GATEWAY_READY' if ok else 'GATEWAY_NOT_READY')\""
        )
        result = await environment.execute_command(wait_cmd, timeout=55)
        if result.get("returncode", 1) != 0 or "GATEWAY_READY" not in (
            result.get("stdout") or ""
        ):
            raise RuntimeError(f"gateway did not become ready on port {port}: {result}")

    async def _start_gateway(
        self,
        environment: BaseEnvironment,
        *,
        api_key: str,
        provider_str: str,
    ) -> None:
        if self._gateway_pgid is not None or self._port_is_open(self._gateway_port):
            raise RuntimeError(
                f"gateway start requires a closed owned port: {self._gateway_port}"
            )
        result = await environment.execute_command(
            self._make_key_env(provider_str, api_key)
            + "export OPENCLAW_DISABLE_BONJOUR=1 && "
            "rm -f /tmp/openclaw_gateway.pgid && "
            f"nohup setsid sh -c 'exec openclaw gateway --port {self._gateway_port}' "
            "</dev/null >/tmp/openclaw_gateway.log 2>&1 & "
            "child=$!; echo $child >/tmp/openclaw_gateway.pgid; "
            "echo GATEWAY_LAUNCHED",
            timeout=10,
        )
        if result.get("returncode", 1) != 0 or "GATEWAY_LAUNCHED" not in (
            result.get("stdout") or ""
        ):
            raise RuntimeError(f"gateway launch failed: {result}")

        if os.environ.get("PAWBENCH_PODMAN_NESTED") == "1":
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                self._gateway_pgid = self._gateway_group_for_port(self._gateway_port)
                if self._gateway_pgid is not None:
                    break
                await asyncio.sleep(0.1)
            if self._gateway_pgid is None:
                raise RuntimeError(
                    f"could not resolve gateway process group for port {self._gateway_port}"
                )

        await self._wait_gateway_ready(environment, port=self._gateway_port)

    async def _configure_openclaw_json(
        self,
        environment: BaseEnvironment,
        *,
        api_key: str,
        base_url: str,
        model_identifier: str,
        explicit_base_url: bool = False,
    ) -> None:
        """Patch ``~/.openclaw/openclaw.json`` with provider and model config.

        This replaces the previous dual-path approach (auth-profiles.json +
        conditional openclaw.json patch) with a single, always-executed write:

        * The API key is stored as an env-variable reference so the plaintext
          secret does not appear in the config file. The env vars are set
          before every openclaw invocation anyway.
        * ``base_url`` defaults to the official DashScope endpoint when the
          provider is ``dashscope``/``qwen`` and no override is supplied.
        * Model metadata (``reasoning``, ``contextWindow``, ``maxTokens``) is
          filled in for well-known providers so openclaw can handle thinking
          mode and context-limit decisions correctly.
        * ``thinkingFormat`` is intentionally omitted — valid values in the
          current openclaw schema are ``openai``, ``openrouter``, ``deepseek``,
          and ``zai``; ``qwen``/``qwen-chat-template`` are explicitly excluded.
        * ``compaction`` is set to ``safeguard`` and ``tools.allow`` to ``["*"]``
          to match QwenClawBench defaults and avoid silent task failures.
        """
        try:
            # Expand model aliases before parsing provider/model.  Short names
            # like "opus-4.6" must resolve to "custom/openai.claude-opus-4-6"
            # before we split on "/" to get provider_str and model_name.
            from pawbench.llm.model_config import ModelConfigManager
            _aliases = ModelConfigManager.get_available_models()
            _expanded = _aliases.get(model_identifier.strip(), model_identifier)
            provider_str = (
                _expanded.split("/", 1)[0].lower()
                if "/" in _expanded else "openai"
            )
            model_name = _expanded.split("/")[-1]

            # ── provider id & base URL ─────────────────────────────────────
            #
            # Routing priority:
            #   1. dashscope/qwen  → always "dashscope" custom provider
            #      (built-in "qwen" plugin forces openai-responses, incompatible
            #      with DashScope's openai-completions-only endpoint).
            #   2. openai / anthropic / google / gemini → always use openclaw's
            #      native provider name so auth headers are correct (Anthropic
            #      requires x-api-key not Authorization: Bearer; Google needs its
            #      own signing).  base_url is forwarded when the caller set an
            #      explicit endpoint override (e.g. self-hosted Anthropic proxy);
            #      omitted otherwise so openclaw uses its built-in default URL.
            #   3. Unknown provider string with explicit base_url → custom-<host>
            #      OpenAI-compatible custom endpoint.
            #   4. Fallback → provider_str as-is, no base_url.
            if provider_str in ("dashscope", "qwen"):
                # openclaw's first-party DashScope integration uses "qwen" as
                # its canonical provider ID (see extensions/qwen/).  Using this
                # name makes the agent model reference (qwen/<name> written by
                # _openclaw_model_id) resolve correctly against the provider
                # entry we write here, preventing the "unknown agent id" error.
                openclaw_provider = "qwen"
                if base_url:
                    effective_base_url = base_url
                else:
                    # CN endpoint is more reliable inside mainland; fall back to
                    # international endpoint when DASHSCOPE_INTL is set.
                    import os as _os
                    if _os.environ.get("DASHSCOPE_INTL"):
                        effective_base_url = (
                            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
                        )
                    else:
                        effective_base_url = (
                            "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )
            elif provider_str in ("openai", "anthropic", "google", "gemini"):
                # Always use openclaw's native provider name so auth headers are
                # handled correctly (Anthropic needs x-api-key not Bearer; Google
                # needs its own signing logic).  base_url is set only when the
                # caller explicitly configured a non-default endpoint (e.g. a
                # self-hosted Anthropic-compatible proxy); omitting it for the
                # default case lets openclaw fall back to its own hardcoded URL.
                openclaw_provider = provider_str
                effective_base_url = base_url if explicit_base_url else ""
            elif base_url:
                from urllib.parse import urlparse
                hostname = urlparse(base_url).hostname or "custom"
                openclaw_provider = "custom-" + hostname.replace(".", "-")
                effective_base_url = base_url
            else:
                openclaw_provider = provider_str
                effective_base_url = ""

            # ── api protocol type ──────────────────────────────────────────
            provider_api = self._PROVIDER_API_TYPE.get(provider_str, "openai-completions")

            # ── api key reference ──────────────────────────────────────────
            # Use a literal string key (not an env-ref object).  The env-ref
            # format is rejected by openclaw's config validator when
            # ``agents add`` runs, and the agents add command would abort.
            # The API key is already available in the container environment
            # for every openclaw invocation, so embedding it directly is
            # equivalent but avoids the schema validation failure.
            api_key_entry: Any = api_key

            # openclaw schema requires BOTH "id" and "name" for model list entries
            # under models.providers.<p>.models[]; omitting either causes config
            # validation failure on ``openclaw agents add``.
            model_entry: Dict[str, Any] = {
                "id": model_name,
                "name": model_name,
                "compat": {"supportsTools": True},
            }
            # Pawbench: always route image-tool calls to the model under evaluation.
            # Avoids openclaw built-in fallbacks (gpt-5.4-mini, qwen3.5-plus, …).
            # Override only via agent_config["vision_model"] for a separate VL id.
            vision_model_name: str = self.config.get("vision_model") or model_name
            vision_model_ref = f"{openclaw_provider}/{vision_model_name}"
            _input_modalities = ["text", "image"]

            # Populate context/token limits, reasoning flag, and input modalities
            # for well-known providers.  openclaw uses these to drive thinking
            # mode activation, context-window guards, and tool routing.
            if provider_str in ("dashscope", "qwen"):
                # Set reasoning=True so openclaw activates extended thinking for
                # Qwen3 models.  We no longer forcibly pin api="openai-completions"
                # here; the native "qwen" provider extension handles DashScope
                # routing correctly and openclaw will pick the right API type for
                # the endpoint we supply in prov['baseUrl'].
                model_entry["reasoning"] = True
                model_entry["contextWindow"] = 200000
                model_entry["maxTokens"] = 32768
                model_entry["input"] = _input_modalities
            elif provider_str == "anthropic":
                model_entry["reasoning"] = True
                model_entry["contextWindow"] = 200000
                model_entry["maxTokens"] = 32000
                model_entry["input"] = _input_modalities
            elif provider_str == "openai":
                model_entry["contextWindow"] = 128000
                model_entry["maxTokens"] = 16384
                model_entry["input"] = _input_modalities
            else:
                model_entry["input"] = _input_modalities

            primary = f"{openclaw_provider}/{model_name}"

            # agents.defaults.imageModel → vision_model_ref (see vision_model_name above)
            # Separate VL entry only when imageModel points at a different model id.
            vision_model_entry: Dict[str, Any] = {}
            if vision_model_name and vision_model_name != model_name:
                vision_model_entry = {
                    "id": vision_model_name,
                    "name": vision_model_name,
                    "input": ["text", "image"],
                    "compat": {"supportsTools": False},
                }

            patch_script = (
                "import json, os\n"
                "p = '/root/.openclaw/openclaw.json'\n"
                "os.makedirs(os.path.dirname(p), exist_ok=True)\n"
                "d = json.load(open(p)) if os.path.exists(p) else {}\n"
                # ── provider ──────────────────────────────────────────────
                "providers = d.setdefault('models', {}).setdefault('providers', {})\n"
                f"prov = providers.setdefault({json.dumps(openclaw_provider)}, "
                f"  {{'api': {json.dumps(provider_api)}, 'models': []}})\n"
                f"prov['api'] = {json.dumps(provider_api)}\n"
                f"prov['baseUrl'] = {json.dumps(effective_base_url)}\n"
                + f"prov['apiKey'] = {json.dumps(api_key_entry)}\n"
                # ``auth: "api-key"`` triggers openclaw's early-exit key
                # resolution path (shouldPreferExplicitConfigApiKeyAuth), which
                # reads models.providers.*.apiKey BEFORE checking env vars
                # (QWEN_API_KEY / MODELSTUDIO_API_KEY / DASHSCOPE_API_KEY).
                # Without this, the container's DASHSCOPE_API_KEY env var can
                # shadow the correct key we wrote above, causing HTTP 401.
                # See: src/agents/model-auth.ts resolveApiKeyForProvider()
                + f"prov['auth'] = 'api-key'\n"
                # ── primary model entry ────────────────────────────────────
                "models = prov.setdefault('models', [])\n"
                f"m = next((x for x in models if x.get('id') == {json.dumps(model_name)} or x.get('name') == {json.dumps(model_name)}), None)\n"
                # Use json.loads to embed model_entry: json.dumps outputs JSON
                # booleans (true/false) which are NOT valid Python literals and
                # would raise NameError in the generated patch script.
                # Double-encoding (json.dumps of a json.dumps string) turns the
                # object into a quoted JSON string literal that json.loads can
                # safely parse at runtime.
                f"new_entry = json.loads({json.dumps(json.dumps(model_entry))})\n"
                "if m is None:\n"
                "    models.append(new_entry)\n"
                "else:\n"
                "    m.update(new_entry)\n"
                # ── vision model entry (for image tool) ────────────────────
                # Register the VL model in the same provider so openclaw's
                # config validator accepts the agents.defaults.imageModel ref.
                + (
                    f"vision_entry = json.loads({json.dumps(json.dumps(vision_model_entry))})\n"
                    f"vm = next((x for x in models if x.get('id') == {json.dumps(vision_model_name)}), None)\n"
                    "if vm is None:\n"
                    "    models.append(vision_entry)\n"
                    "else:\n"
                    "    vm.update(vision_entry)\n"
                    if vision_model_entry else ""
                )
                # ── gateway.mode (required since openclaw 2026.4.x) ───────
                # openclaw gateway refuses to start if gateway.mode is absent.
                # agents add may drop this field when rewriting openclaw.json;
                # ensure it is always present after our patch.
                + "gateway_cfg = d.setdefault('gateway', {})\n"
                + "gateway_cfg.setdefault('mode', 'local')\n"
                + f"gateway_cfg['port'] = {self._gateway_port}\n"
                # ── agents.defaults ───────────────────────────────────────
                + "agents_cfg = d.setdefault('agents', {}).setdefault('defaults', {})\n"
                f"agents_cfg['model'] = {{'primary': {json.dumps(primary)}}}\n"
                f"agents_cfg.setdefault('models', {{}})[{json.dumps(primary)}] = "
                f"  {json.dumps({'alias': model_name})}\n"
                # compaction safeguard prevents long-task context overflow
                "agents_cfg.setdefault('compaction', {})['mode'] = 'safeguard'\n"
                # Set timeoutSeconds to a large value so the gateway does not cut the
                # LLM proxy connection mid-call.  The default of 600 causes a 630 s
                # hard cut ((600+30)×1000 ms) that breaks long thinking-mode calls.
                # openclaw's schema rejects 0 (must be >0), so use 86400 (24 h) as an
                # effectively-infinite sentinel.  The real deadline is enforced by the
                # outer asyncio.wait_for in backend.py.
                "agents_cfg['timeoutSeconds'] = 86400\n"
                # ── imageModel: route image-tool calls to VL model ─────────
                # openclaw.image tool resolves agents.defaults.imageModel to
                # pick which model to send vision requests to.  Without this,
                # it falls back to the global default (currently openai/gpt-5.5)
                # which is unreachable with a DashScope API key → 403/Unknown.
                + (
                    f"agents_cfg['imageModel'] = {json.dumps(vision_model_ref)}\n"
                    f"print('vision model set to: {vision_model_ref}')\n"
                )
                # ── disable built-in qwen plugin (dashscope/qwen only) ────
                # The built-in "qwen" plugin (extensions/qwen/) registers its
                # OWN provider entry at gateway startup using credentials from
                # the auth-profile baked into the image by
                # ``openclaw onboard --auth-choice qwen-standard-api-key-cn``
                # in Dockerfile.pawbench-openclaw (placeholder
                # sk-build-placeholder).  When a request is routed for any
                # known Qwen model name, the plugin's runtime resolver
                # OVERRIDES the apiKey / baseUrl we wrote into openclaw.json,
                # silently sending the placeholder key to DashScope → 401
                # "Incorrect API key".
                #
                # Disabling the plugin keeps our provider entry as the sole
                # source of truth.  This is the difference between Claude-only
                # models working (route through "anthropic"/"openai" providers
                # untouched by the qwen plugin) and all 7 non-Claude models
                # 100% failing with HTTP 401 in pawbench-7models-opusjudge-20260525.
                #
                # Applied to BOTH "dashscope" (raw provider str from the model
                # identifier) AND "qwen" (post-translation), so callers using
                # either spelling are protected.
                #
                # Also disable the built-in "openai" plugin and drop stale
                # providers['openai'] entries.  openclaw 2026.5.x auto-enables
                # ``openai/<model>`` at gateway startup (see gateway log:
                # "auto-enabled plugins for openai/qwen3.6-plus"), which
                # overrides our qwen/ provider and routes DashScope calls with
                # wrong auth / quota paths.
                + (
                    "pe = d.setdefault('plugins', {}).setdefault('entries', {})\n"
                    "pe['qwen'] = {'enabled': False}\n"
                    "pe['openai'] = {'enabled': False}\n"
                    "for stale in ('openai', 'dashscope'):\n"
                    "    providers.pop(stale, None)\n"
                    if provider_str in ("dashscope", "qwen") else ""
                )
                # ── commands: enable native skills ────────────────────────
                # QwenClawBench sets commands.native="auto" and
                # commands.nativeSkills="auto" so openclaw can discover and
                # invoke skills defined in the workspace (e.g. SKILL.md files).
                # Without this, skill-related tasks are evaluated at a
                # disadvantage compared to the reference benchmark.
                + "cmd = d.setdefault('commands', {})\n"
                + "cmd.setdefault('native', 'auto')\n"
                + "cmd['nativeSkills'] = 'auto'\n"
                + "cmd.setdefault('restart', False)\n"
                # ── tools ─────────────────────────────────────────────────
                # Allow all tools by default; restrictive defaults may silently
                # block shell/browser tools the agent needs for tasks.
                + "d.setdefault('tools', {})['allow'] = ['*']\n"
                # ── write ─────────────────────────────────────────────────
                + "json.dump(d, open(p, 'w'), indent=2)\n"
                + f"print('configured openclaw.json: provider=' + {json.dumps(openclaw_provider)} "
                + f"+ ' model=' + {json.dumps(model_name)})\n"
            )
            await environment.write_file("/tmp/patch_openclaw.py", patch_script)
            patch_result = await environment.execute_command(
                self._make_key_env(provider_str, api_key)
                + "python3 /tmp/patch_openclaw.py",
                timeout=15,
            )
            if patch_result.get("returncode", 1) != 0:
                import logging
                logging.getLogger(__name__).warning(
                    "_configure_openclaw_json patch failed (exit %s):\n%s\n%s",
                    patch_result.get("returncode"),
                    (patch_result.get("stdout") or "")[:500],
                    (patch_result.get("stderr") or "")[:500],
                )
        except Exception:
            import logging
            logging.getLogger(__name__).exception("_configure_openclaw_json failed")

    # ── gateway helpers ────────────────────────────────────────────────────────

    async def _stabilise_gateway_plugins(self, environment: BaseEnvironment) -> None:
        """Patch the openclaw image so the gateway survives in a Docker container.

        Two issues block the browser-plugin path on a bare ``openclaw-pawbench``
        image and would otherwise be silent failures:

        1. ``bonjour`` (mDNS service advertisement, via ``@homebridge/ciao``)
           tries to broadcast on multicast, which Docker's default bridge does
           not support.  After ~20 s the ``CIAO ANNOUNCEMENT CANCELLED``
           rejection is unhandled and crashes the entire gateway process,
           leaving every subsequent WS request hanging on a 1006 close.
        2. The bundled ``browser`` plugin requires runtime deps
           (``playwright-core``, ``express``, ``undici``,
           ``@modelcontextprotocol/sdk``) that the image may not ship.
           ``openclaw doctor --fix`` is the official way to install just the
           missing ones; it is idempotent on a fully populated image.

        Without these two patches, ``ws://127.0.0.1:18789`` drops every
        browser-related call after the first request — the symptom diagnosed
        on ``gui_002_zh`` (62.5% score, "browser gateway not running").
        """
        await environment.execute_command(
            "openclaw config set plugins.entries.bonjour.enabled false 2>&1 | tail -2 || true",
            timeout=90,
        )
        await environment.execute_command(
            "openclaw doctor --fix 2>&1 | tail -5 || true",
            timeout=180,
        )

    async def _ensure_gateway(self, environment: BaseEnvironment, *, api_key: str = "", provider_str: str = "openai") -> None:
        port = self._gateway_port

        # TCP-based liveness check: verify the gateway port is actually accepting
        # connections, not just that a process with the saved PID exists.
        # A PID-only check misses "half-dead" gateways where the process is alive
        # but the port has stopped responding (e.g. after an internal crash or
        # deadlock).  A bare TCP connect is sufficient — we don't need a full WS
        # handshake, and a 1006 WS rejection would still mean the port is open.
        tcp_check = (
            'python3 -c "'
            "import socket\n"
            f"s = socket.socket(); s.settimeout(2)\n"
            "try:\n"
            f"  s.connect(('127.0.0.1', {port})); print('ALIVE')\n"
            "except Exception: print('DEAD')\n"
            "finally:\n"
            "  try: s.close()\n"
            "  except Exception: pass\n"
            '"'
        )
        r = await environment.execute_command(tcp_check, timeout=5)
        if "ALIVE" in (r.get("stdout") or ""):
            return

        await self._kill_gateway(environment)
        await self._start_gateway(
            environment, api_key=api_key, provider_str=provider_str
        )

    async def _ensure_gateway_strict(self, environment: BaseEnvironment, *, api_key: str = "") -> None:
        """Stronger gateway start used for the browser-plugin investigation.

        Currently NOT called from ``setup()``.  Kept here so we can swap
        ``_ensure_gateway → _ensure_gateway_strict`` once the gateway-stability
        work is completed.

        Differences from ``_ensure_gateway``:

        * Always force-restarts the gateway instead of trusting a port probe
          (a half-dead gateway can keep 18789 listening but reject WS
          sessions with 1006).
        * Polls ``openclaw browser doctor`` instead of a raw socket connect,
          so it returns only when the browser plugin is actually serving
          requests (~15 s after the log line ``ready``).
        """
        provider_str_strict = (
            (self.config.get("model", "dashscope/qwen3.6-plus") or "")
            .split("/", 1)[0]
            .lower()
            or "openai"
        )
        await self._kill_gateway(environment)
        await self._start_gateway(
            environment, api_key=api_key, provider_str=provider_str_strict
        )

        wait_cmd = (
            "for i in $(seq 1 60); do "
            "  if openclaw browser doctor 2>&1 | "
            "     grep -q 'OK gateway: browser control endpoint reachable'; then "
            "    echo GATEWAY_READY; exit 0; "
            "  fi; "
            "  sleep 1; "
            "done; "
            "echo GATEWAY_NOT_READY; "
            "tail -25 /tmp/openclaw_gateway.log 2>/dev/null || true"
        )
        result = await environment.execute_command(wait_cmd, timeout=75)
        if result.get("returncode", 1) != 0 or "GATEWAY_READY" not in (
            result.get("stdout") or ""
        ):
            raise RuntimeError(f"browser gateway did not become ready: {result}")

    async def _wait_for_session_flush(
        self,
        environment: BaseEnvironment,
        *,
        agent_id_lower: str,
    ) -> None:
        """Best-effort wait for OpenClaw to finish writing session artifacts.

        OpenClaw may return from CLI before trajectory/session files are fully
        flushed. Poll briefly for a recent ``model.completed`` event with a
        non-empty ``messagesSnapshot`` to reduce short-transcript races.
        """
        wait_script = (
            "import glob, json, os, time\n"
            f"sessions_dir = '/root/.openclaw/agents/{agent_id_lower}/sessions'\n"
            "deadline = time.time() + 12\n"
            "def has_completed_snapshot(path):\n"
            "    try:\n"
            "        lines = open(path, 'r', encoding='utf-8').read().splitlines()\n"
            "    except Exception:\n"
            "        return False\n"
            "    for line in reversed(lines):\n"
            "        line = line.strip()\n"
            "        if not line:\n"
            "            continue\n"
            "        try:\n"
            "            obj = json.loads(line)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if not isinstance(obj, dict) or obj.get('type') != 'model.completed':\n"
            "            continue\n"
            "        snap = (obj.get('data') or {}).get('messagesSnapshot')\n"
            "        return isinstance(snap, list) and len(snap) > 0\n"
            "    return False\n"
            "while time.time() < deadline:\n"
            "    traj = sorted(glob.glob(os.path.join(sessions_dir, '*.trajectory.jsonl')), key=os.path.getmtime, reverse=True)\n"
            "    if traj and has_completed_snapshot(traj[0]):\n"
            "        print('SESSION_READY')\n"
            "        raise SystemExit(0)\n"
            "    plain = sorted(glob.glob(os.path.join(sessions_dir, '*.jsonl')), key=os.path.getmtime, reverse=True)\n"
            "    plain = [p for p in plain if not p.endswith('.trajectory.jsonl')]\n"
            "    if plain and os.path.getsize(plain[0]) > 0:\n"
            "        print('SESSION_READY_PLAIN')\n"
            "        raise SystemExit(0)\n"
            "    time.sleep(1)\n"
            "print('SESSION_NOT_READY')\n"
        )
        await environment.write_file("/tmp/wait_openclaw_session.py", wait_script)
        try:
            await environment.execute_command(
                "python3 /tmp/wait_openclaw_session.py",
                timeout=15,
            )
        except TimeoutError:
            # This poll is explicitly best-effort.  A slow container exec must
            # not erase an otherwise completed task; post_run_collect() still
            # copies any session artifacts that are available, and the canary
            # rejects an empty transcript before a full run can start.
            return

    # ── run ───────────────────────────────────────────────────────────────────

    async def run(self, instruction: str, environment: BaseEnvironment) -> Dict[str, Any]:
        model_identifier = self.config.get("model", "dashscope/qwen3.6-plus")
        model_config = get_model_config(model_identifier)
        api_key = self._resolve_api_key(model_config)
        provider_str = (
            model_identifier.split("/", 1)[0].lower()
            if "/" in model_identifier else "openai"
        )

        agent_id = self._agent_id()
        agent_id_lower = agent_id.lower()

        # Wipe sessions from any previous run so openclaw starts a fresh
        # conversation for this task.
        await environment.execute_command(
            f"rm -f /root/.openclaw/agents/{agent_id_lower}/sessions/*.jsonl "
            f"/root/.openclaw/agents/{agent_id_lower}/sessions/*.jsonl.lock "
            f"/root/.openclaw/agents/{agent_id_lower}/sessions/sessions.json "
            "2>/dev/null || true",
            timeout=15,
        )

        # Re-check gateway liveness before each task (it may have crashed).
        await self._ensure_gateway(environment, api_key=api_key, provider_str=provider_str)

        # Verify the agent is actually known to the gateway.  In rare cases
        # (inotify exhaustion, timing) the gateway starts without the agent
        # config loaded.  Re-add + restart the gateway when that happens.
        openclaw_model = self._openclaw_model_id(model_identifier)
        env_prefix = self._make_key_env(provider_str, api_key)
        check_result = await environment.execute_command(
            f"test -f /root/.openclaw/agents/{agent_id_lower}/agent/auth-profiles.json "
            f"&& echo {shlex.quote(agent_id)} || true",
            timeout=10,
        )
        check_output = (check_result.get("stdout") or "") + (check_result.get("stderr") or "")
        if agent_id.lower() not in check_output.lower():
            import logging
            logging.getLogger(__name__).warning(
                "Agent %s not found in 'openclaw agents list' at run() start — re-adding and restarting gateway",
                agent_id,
            )
            await self._kill_gateway(environment)
            await environment.execute_command(
                f"{env_prefix}"
                f"openclaw agents add {shlex.quote(agent_id)} "
                f"--model {shlex.quote(openclaw_model)} "
                f"--workspace {shlex.quote(AGENT_WORKSPACE)} "
                "--non-interactive",
                timeout=120,
            )
            # Overwrite placeholder auth-profiles baked in by agents add.
            _provider_for_auth = self._openclaw_model_id(model_identifier).split("/")[0]
            await environment.write_file(
                f"/root/.openclaw/agents/{agent_id.lower()}/agent/auth-profiles.json",
                json.dumps({
                    "version": 1,
                    "profiles": {
                        f"{_provider_for_auth}:default": {
                            "type": "api_key",
                            "provider": _provider_for_auth,
                            "key": api_key,
                        }
                    },
                }, indent=2),
            )
            await environment.write_file(
                "/tmp/patch_gateway_mode.py",
                "import json, os\n"
                "p = '/root/.openclaw/openclaw.json'\n"
                "d = json.load(open(p)) if os.path.exists(p) else {}\n"
                "d.setdefault('gateway', {})['mode'] = 'local'\n"
            f"d['gateway']['port'] = {self._gateway_port}\n"
                "json.dump(d, open(p, 'w'), indent=2)\n"
                "print('gateway.mode ensured')\n",
            )
            await environment.execute_command(
                "python3 /tmp/patch_gateway_mode.py",
                timeout=10,
            )
            base_url = self.config.get("base_url") or model_config.base_url or ""
            await self._configure_openclaw_json(
                environment,
                api_key=api_key,
                base_url=base_url,
                model_identifier=model_identifier,
                explicit_base_url=bool(self.config.get("base_url")),
            )
            await self._start_gateway(
                environment, api_key=api_key, provider_str=provider_str
            )

        escaped = shlex.quote(instruction)
        session_id = f"pawbench-{int(time.time() * 1000)}"

        inner_timeout = int(self.config.get("task_timeout_s") or 1800)

        thinking_level = (
            self.config.get("thinking_level") or self.config.get("thinking") or ""
        )
        thinking_args = (
            f"--thinking {shlex.quote(thinking_level)} " if thinking_level else ""
        )

        run_cmd = (
            self._make_key_env(provider_str, api_key)
            + f"cd {shlex.quote(AGENT_WORKSPACE)} && "
            f"timeout {inner_timeout}s openclaw agent "
            f"--agent {shlex.quote(agent_id)} --session-id {session_id} "
            f"{thinking_args}--message {escaped} "
            f"2>&1 | tee /tmp/openclaw_output.txt || true"
        )
        # Snapshot existing text files before the run starts so that
        # extract_transcript() can exclude pre-staged fixture files.
        # post_run_collect() copies openclaw's internal workspace with cp
        # (no -p), giving all synced files mtime = now, which would cause
        # mtime-based filtering to treat fixtures as agent output.  Storing
        # the pre-run set of basenames gives us a reliable exclusion list.
        _prerun_ls = await environment.execute_command(
            f"find {shlex.quote(AGENT_WORKSPACE)} -type f "
            r"\( -name '*.md' -o -name '*.txt' -o -name '*.csv' \) "
            r"-printf '%f\n' 2>/dev/null || true"
        )
        self._pre_run_file_basenames: set[str] = set(
            _prerun_ls.get("stdout", "").splitlines()
        )
        # Record run start time so extract_transcript() can identify
        # files created during this task (vs pre-mounted fixture files).
        self._run_start_time: float = time.time()
        result = await environment.execute_command(run_cmd, timeout=inner_timeout + 60)
        output_content = (
            await environment.read_file("/tmp/openclaw_output.txt") or result["stdout"]
        )

        # Reduce race where session files are copied before OpenClaw flushes the
        # final model.completed event.
        await self._wait_for_session_flush(
            environment,
            agent_id_lower=agent_id_lower,
        )

        # Copy openclaw session files into workspace/sessions/ so that
        # extract_transcript() can find them.  Trajectory files are preferred
        # (they carry the full messagesSnapshot); plain session JSONL files are
        # copied as a fallback for older openclaw builds.
        await environment.execute_command(
            f"mkdir -p {AGENT_WORKSPACE}/sessions && "
            f"sessions_dir=/root/.openclaw/agents/{agent_id_lower}/sessions && "
            "for f in \"$sessions_dir\"/*.trajectory.jsonl; do "
            '  [ -f "$f" ] || continue; '
            f'  cp "$f" "{AGENT_WORKSPACE}/sessions/"; '
            "done && "
            "for f in \"$sessions_dir\"/*.jsonl; do "
            '  [ -f "$f" ] || continue; '
            '  case "$f" in *.trajectory.jsonl) continue;; esac; '
            f'  cp "$f" "{AGENT_WORKSPACE}/sessions/"; '
            "done",
            timeout=15,
        )

        await self._sync_workspace_to_output(environment, AGENT_WORKSPACE)

        return {
            "success": result["success"],
            "output": output_content,
            "error": result.get("stderr", ""),
            "returncode": result["returncode"],
            "session_data": "",
            "metrics": {
                "execution_time": 0,
                "model_used": model_config.get_full_model_identifier(),
                "provider": model_config.provider.value,
                "model_name": model_config.model_name,
                "session_id": session_id,
            },
        }

    def extract_transcript(
        self,
        local_workspace: "Any",
        stdout: str,
    ) -> "List[Dict[str, Any]]":
        """Extend the base transcript with workspace output files.

        openclaw writes structured results (tables, analysis, reports) to files
        in the workspace but only provides a brief prose summary in its final
        assistant message.  The automated grader searches the transcript for
        specific patterns (Markdown tables, numbers, keywords), so the grader
        misses valid output that lives only in files.

        This override appends a synthetic final assistant message containing the
        content of every text/Markdown file that was created *during* the run
        (identified by mtime >= self._run_start_time).  qwenpaw and hermes are
        unaffected because they do not use this class.
        """
        transcript = super().extract_transcript(local_workspace, stdout)

        run_start: float | None = getattr(self, "_run_start_time", None)
        if run_start is None or local_workspace is None:
            return transcript

        from pathlib import Path as _Path

        ws = _Path(local_workspace)
        if not ws.is_dir():
            return transcript

        # sessions/ contains agent metadata; output/ is a mirror copy created by
        # _sync_workspace_to_output — skip both to avoid double-counting.
        # fixtures/ holds read-only task inputs whose mtime equals the workspace
        # copy time (always >= _run_start_time), so they would be misidentified
        # as agent-produced output — exclude them explicitly.
        _SKIP_TOP = {"sessions", "output", "fixtures"}
        _TEXT_SUFFIXES = {".md", ".txt", ".csv"}
        # openclaw writes these identity/config files on every startup — they
        # are never task output, so exclude them regardless of mtime.
        _SKIP_NAMES = {
            "SOUL.md", "TOOLS.md", "IDENTITY.md", "AGENTS.md",
            "USER.md", "HEARTBEAT.md", "BOOTSTRAP.md", "HEAD",
        }

        # Basenames of files that already existed before the openclaw command
        # ran (snapshot taken in run()).  post_run_collect() copies openclaw's
        # internal workspace without preserving mtime, so every synced file
        # appears "new" to the mtime filter — this set lets us exclude them.
        pre_run_names: set[str] = getattr(self, "_pre_run_file_basenames", set())

        new_files: list[tuple[float, _Path]] = []
        for f in ws.rglob("*"):
            if not f.is_file():
                continue
            try:
                rel_parts = f.relative_to(ws).parts
            except ValueError:
                continue
            if rel_parts and rel_parts[0] in _SKIP_TOP:
                continue
            if f.name in _SKIP_NAMES:
                continue
            if f.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            # Skip files that existed before the run (pre-staged fixtures).
            if f.name in pre_run_names:
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            # Only files written after the task run started.
            if mtime >= run_start:
                new_files.append((mtime, f))

        if not new_files:
            return transcript

        new_files.sort()  # ascending by mtime — chronological order

        parts: list[str] = []
        total_chars = 0
        for _, f in new_files[:10]:
            try:
                content = f.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if len(content) < 30:
                continue
            rel = str(f.relative_to(ws))
            chunk = f"[File: {rel}]\n{content}"
            remaining = 80_000 - total_chars
            if len(chunk) > remaining:
                if remaining < 200:
                    break
                chunk = chunk[:remaining] + "\n[...truncated...]"
            parts.append(chunk)
            total_chars += len(chunk)
            if total_chars >= 80_000:
                break

        if not parts:
            return transcript

        synth_text = "\n\n---\n\n".join(parts)
        transcript.append({
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": synth_text}],
            },
        })
        return transcript

    async def post_run_collect(self, environment: BaseEnvironment) -> None:
        """Sync openclaw internal workspace to the standard benchmark path.

        openclaw's ACPX runtime may write files to the global-default workspace
        (/root/.openclaw/workspace) rather than the per-agent workspace even when
        ``agents add --workspace`` is used.  Mirror all non-bytecode files from
        every known openclaw workspace location into AGENT_WORKSPACE before the
        backend snapshots it for grading.
        """
        _SYNC_CMD = rf"""
DEST={AGENT_WORKSPACE}
mkdir -p "$DEST/output"
for src_dir in /root/.openclaw/workspace ~/.openclaw/workspace; do
  [ -d "$src_dir" ] || continue
  # skip if already pointing at AGENT_WORKSPACE (symlink or same inode)
  real_src=$(realpath "$src_dir" 2>/dev/null || echo "$src_dir")
  real_dst=$(realpath "$DEST" 2>/dev/null || echo "$DEST")
  [ "$real_src" = "$real_dst" ] && continue
  find "$src_dir" -maxdepth 5 -type f \
      ! -path '*/site-packages/*' ! -name '*.pyc' \
      2>/dev/null | while read -r f; do
    rel="${{f#${{src_dir}}/}}"
    dest="$DEST/$rel"
    mkdir -p "$(dirname "$dest")"
    [ ! -s "$dest" ] && [ -s "$f" ] && cp "$f" "$dest" 2>/dev/null || true
  done
done
"""
        await environment.execute_command(_SYNC_CMD, timeout=30)

    async def teardown(self, environment: BaseEnvironment) -> None:
        agent_id = self._agent_id()
        # Each task owns an ephemeral container; deleting the agent through the
        # CLI can block behind the live gateway.  Stop only this task's gateway
        # and let container removal discard its private agent configuration.
        await self._kill_gateway(environment)
        await environment.execute_command(
            "rm -f /tmp/openclaw_output.txt /tmp/patch_openclaw.py",
            timeout=10,
        )

    @property
    def version(self) -> str:
        return "2026.4.24"
