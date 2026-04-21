"""LocalSandbox: runs commands as real subprocesses."""

from __future__ import annotations

import subprocess

from agent_lemon_lime.harness.base import ExecResult


class LocalSandbox:
    """Sandbox backend that executes commands via subprocess."""

    def __init__(self, *, workdir: str | None = None) -> None:
        self._workdir = workdir
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def __enter__(self) -> LocalSandbox:
        self._active = True
        return self

    def __exit__(self, *args: object) -> None:
        self._active = False

    def exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecResult:
        if not self._active:
            raise RuntimeError("LocalSandbox must be used as a context manager")
        cwd = workdir or self._workdir
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=timeout_seconds,
            )
            return ExecResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(exit_code=124, stdout="", stderr="Command timed out")
        except FileNotFoundError as exc:
            return ExecResult(exit_code=127, stdout="", stderr=f"Command not found: {exc}")
