"""MockSandbox: in-process sandbox for local/CI use."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from agent_lemon_lime.harness.base import ExecResult


@dataclass
class _Registration:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    """Returns pre-registered responses without any external process."""

    def __init__(self) -> None:
        self._registry: dict[tuple[str, ...], _Registration] = {}
        self._calls: dict[tuple[str, ...], int] = defaultdict(int)
        self._active = False

    def register_command(
        self,
        command: list[str],
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ) -> None:
        self._registry[tuple(command)] = _Registration(
            stdout=stdout, stderr=stderr, exit_code=exit_code
        )

    def call_count(self, command: list[str]) -> int:
        return self._calls[tuple(command)]

    @property
    def is_active(self) -> bool:
        return self._active

    def __enter__(self) -> MockSandbox:
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
            raise RuntimeError(
                "MockSandbox must be used as a context manager before calling exec()"
            )
        key = tuple(command)
        self._calls[key] += 1
        if key in self._registry:
            reg = self._registry[key]
            return ExecResult(exit_code=reg.exit_code, stdout=reg.stdout, stderr=reg.stderr)
        return ExecResult(
            exit_code=1,
            stdout="",
            stderr=f"MockSandbox: command {list(command)!r} not registered",
        )
