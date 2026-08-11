# -*- coding: utf-8 -*-
"""Docker environment implementation for OpenJudge agent evaluation framework."""

import asyncio
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from pawbench.envs.base import BaseEnvironment


_NESTED_PODMAN_LIFECYCLE_LOCK = threading.Lock()
_NESTED_PODMAN_COMMAND_TIMEOUT = 60


class DockerEnvironment(BaseEnvironment):
    """Docker-based execution environment implementation.

    This implementation uses Docker containers to provide isolated execution
    environments for agents.
    """

    def __init__(
        self,
        name: str,
        image: str = "python:3.11-slim",
        volumes: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[str, str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ):
        """Initialize the Docker environment.

        Args:
            name: Unique name for the environment
            image: Docker image to use
            volumes: Volume mappings for the container
            ports: Port mappings for the container
            environment_vars: Environment variables for the container
            **kwargs: Additional configuration parameters
        """
        super().__init__(name, **kwargs)
        self.image = image
        self.volumes = volumes or {}
        self.ports = ports or {}
        self.environment_vars = environment_vars or {}
        self.container_id: Optional[str] = None
        self._is_running = False

    @staticmethod
    def _lifecycle_lock_acquire(nested: bool) -> None:
        if nested and not _NESTED_PODMAN_LIFECYCLE_LOCK.acquire(
            timeout=_NESTED_PODMAN_COMMAND_TIMEOUT
        ):
            raise RuntimeError("Timed out waiting for nested Podman lifecycle lock")

    @staticmethod
    def _lifecycle_lock_release(nested: bool) -> None:
        if nested:
            _NESTED_PODMAN_LIFECYCLE_LOCK.release()

    @staticmethod
    def _cleanup_container_record_sync(
        name: str, *, nested: bool, graceful: bool
    ) -> tuple[list[str], str]:
        """Run every exact cleanup step and return errors plus remaining IDs."""
        timeout = _NESTED_PODMAN_COMMAND_TIMEOUT if nested else 15
        commands = []
        if graceful:
            commands.append(["docker", "stop", "-t", "5", name])
        commands.extend(
            [
                ["docker", "rm", "-f", name],
                ["docker", "container", "cleanup", "--rm", name],
            ]
        )
        errors: list[str] = []
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"timeout command={command[1:]}")
                continue
            # Absent-container errors are expected during idempotent cleanup;
            # final `docker ps` is the authoritative postcondition.
            if result.returncode not in (0, 1):
                errors.append(
                    f"exit={result.returncode} command={command[1:]} "
                    f"stderr={result.stderr.strip()}"
                )
        try:
            listed = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            errors.append("timeout command=['ps', '-aq']")
            return errors, "verification-timeout"
        if listed.returncode != 0:
            errors.append(
                f"exit={listed.returncode} command=['ps', '-aq'] "
                f"stderr={listed.stderr.strip()}"
            )
            return errors, "verification-error"
        return errors, listed.stdout.strip()

    def _start_sync(self) -> None:
        nested = os.environ.get("PAWBENCH_PODMAN_NESTED") == "1"
        self._lifecycle_lock_acquire(nested)
        try:
            cleanup_errors, remaining = self._cleanup_container_record_sync(
                self.name, nested=nested, graceful=False
            )
            if remaining:
                raise RuntimeError(
                    f"Existing container cleanup incomplete: {self.name} "
                    f"listed={remaining} errors={cleanup_errors}"
                )

            cmd = ["docker", "run", "-d", "--name", self.name]
            if nested:
                cmd.extend(["--network", "host", "--pid", "host"])
            for host_path, container_path in self.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
            for host_port, container_port in self.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
            for key, value in self.environment_vars.items():
                cmd.extend(["-e", f"{key}={value}"])
            cmd.append(self.image)
            cmd.extend(["sleep", "infinity"])

            timeout = _NESTED_PODMAN_COMMAND_TIMEOUT if nested else None
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                cleanup_errors, remaining = self._cleanup_container_record_sync(
                    self.name, nested=nested, graceful=False
                )
                self.container_id = None
                self._is_running = False
                detail = (
                    f"cleanup_remaining={remaining} cleanup_errors={cleanup_errors}"
                )
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"Timed out starting Docker container: {self.name}; {detail}"
                    ) from exc
                raise RuntimeError(
                    f"Failed to start Docker container: {exc.stderr}; {detail}"
                ) from exc
            self.container_id = result.stdout.strip()
            self._is_running = True
        finally:
            self._lifecycle_lock_release(nested)

    @staticmethod
    async def _await_lifecycle_thread(function) -> None:
        """Delay cancellation until the storage transaction has reached closure."""
        task = asyncio.create_task(asyncio.to_thread(function))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                # The synchronous transaction owns its own rollback. Preserve
                # the caller's cancellation after it reaches a terminal state.
                pass
            raise

    async def start(self) -> None:
        """Start the container without blocking another task's event loop."""
        await self._await_lifecycle_thread(self._start_sync)

    def _stop_sync(self) -> None:
        nested = os.environ.get("PAWBENCH_PODMAN_NESTED") == "1"
        self._lifecycle_lock_acquire(nested)
        try:
            errors, remaining = self._cleanup_container_record_sync(
                self.name, nested=nested, graceful=True
            )
            if remaining:
                raise RuntimeError(
                    f"Container cleanup incomplete: {self.name} "
                    f"listed={remaining} errors={errors}"
                )
            self._is_running = False
            self.container_id = None
        finally:
            self._lifecycle_lock_release(nested)

    async def stop(self) -> None:
        """Stop and remove exactly this container without blocking its loop."""
        if not self.container_id:
            return
        await self._await_lifecycle_thread(self._stop_sync)

    def _docker_exec_command(
        self, command: str, *, wait_timeout: int, nested: bool
    ) -> list[str]:
        if nested:
            # Coreutils timeout creates a separate foreground process group and
            # terminates it inside the shared PID namespace. This prevents a
            # timed-out `podman exec` client from leaving `openclaw agents add`
            # or another helper alive to block container removal.
            return [
                "docker",
                "exec",
                self.name,
                "timeout",
                "--kill-after=5s",
                f"{wait_timeout}s",
                "bash",
                "-c",
                command,
            ]
        return ["docker", "exec", self.name, "bash", "-c", command]

    async def execute_command(
        self, command: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a command in the container with nested-process closure."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        wait_timeout = timeout if timeout else 600
        nested = os.environ.get("PAWBENCH_PODMAN_NESTED") == "1"
        cmd = self._docker_exec_command(
            command, wait_timeout=wait_timeout, nested=nested
        )
        # The in-container timeout owns command/process-group termination. The
        # client gets a short grace period to flush stdout and reap the exec.
        client_timeout = wait_timeout + 15 if nested else wait_timeout

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=client_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                snippet = command if len(command) <= 500 else command[:500] + "..."
                raise TimeoutError(
                    f"Command timed out after {wait_timeout} seconds: {snippet}"
                )

            stdout_text = stdout.decode(errors="replace") if stdout else ""
            stderr_text = stderr.decode(errors="replace") if stderr else ""
            if nested and process.returncode in (124, 137):
                snippet = command if len(command) <= 500 else command[:500] + "..."
                raise TimeoutError(
                    f"Command timed out after {wait_timeout} seconds: {snippet}"
                )
            return {
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": process.returncode,
                "success": process.returncode == 0,
            }
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
                "success": False,
            }

    async def copy_to(self, source: Path, destination: str) -> bool:
        """Copy a file from host to container."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        cmd = ["docker", "cp", str(source), f"{self.name}:{destination}"]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    async def copy_from(self, source: str, destination: Path) -> bool:
        """Copy a file from container to host."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        cmd = ["docker", "cp", f"{self.name}:{source}", str(destination)]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    async def write_file(self, path: str, content: str) -> bool:
        """Write content to a file in the container."""
        # Create a temporary file on host and copy to container
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            success = await self.copy_to(Path(temp_path), path)
            Path(temp_path).unlink()
            return success
        except:
            Path(temp_path).unlink()
            return False

    async def read_file(self, path: str) -> Optional[str]:
        """Read content from a file in the container."""
        # Copy file to temporary location and read
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            success = await self.copy_from(path, Path(temp_path))
            if success:
                with open(temp_path, 'r') as f:
                    content = f.read()
                Path(temp_path).unlink()
                return content
            else:
                Path(temp_path).unlink()
                return None
        except:
            Path(temp_path).unlink()
            return None

    @property
    def is_running(self) -> bool:
        """Check if the container is running."""
        return self._is_running