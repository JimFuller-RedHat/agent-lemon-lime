"""AbstractSandbox protocol and shared data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@runtime_checkable
class AbstractSandbox(Protocol):
    """Protocol for sandbox backends (mock, openshell, local-process)."""

    def __enter__(self) -> AbstractSandbox: ...
    def __exit__(self, *args: object) -> None: ...

    def exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecResult: ...

    @property
    def is_active(self) -> bool: ...
