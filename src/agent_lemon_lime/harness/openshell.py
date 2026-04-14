"""OpenshellSandbox: wraps openshell.Sandbox for AbstractSandbox compliance."""

from __future__ import annotations

import contextlib
from typing import Any

from agent_lemon_lime.harness.base import ExecResult


class OpenshellSandbox:
    """Sandbox backed by NVIDIA OpenShell cluster (requires live cluster in production)."""

    def __init__(
        self,
        *,
        cluster: str | None = None,
        timeout: float = 30.0,
        ready_timeout_seconds: float = 120.0,
        _client: Any | None = None,  # injected in tests; openshell type not available statically
    ) -> None:
        self._cluster = cluster
        self._timeout = timeout
        self._ready_timeout = ready_timeout_seconds
        self._test_client: Any | None = _client
        # openshell types are not available statically; use Any for dynamic dispatch
        self._session: Any | None = None
        self._client_instance: Any | None = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def __enter__(self) -> OpenshellSandbox:
        if self._test_client is not None:
            self._client_instance = self._test_client
        else:
            import openshell

            self._client_instance = openshell.SandboxClient.from_active_cluster(
                cluster=self._cluster,
                timeout=self._timeout,
            )
        self._session = self._client_instance.create_session()
        self._active = True
        return self

    def __exit__(self, *args: object) -> None:
        try:
            if self._session is not None:
                with contextlib.suppress(Exception):
                    self._session.delete()
        finally:
            if self._client_instance is not None and self._test_client is None:
                with contextlib.suppress(Exception):
                    self._client_instance.close()
            self._session = None
            self._client_instance = None
            self._active = False

    def exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecResult:
        if not self._active or self._session is None:
            raise RuntimeError("OpenshellSandbox must be used as a context manager")
        raw = self._session.exec(
            command,
            workdir=workdir,
            env=env,
            timeout_seconds=timeout_seconds,
        )
        return ExecResult(
            exit_code=raw.exit_code,
            stdout=raw.stdout,
            stderr=raw.stderr,
        )
