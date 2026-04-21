"""OpenshellSandbox: wraps openshell.Sandbox for AbstractSandbox compliance."""

from __future__ import annotations

import contextlib
import logging
import subprocess
from typing import Any

from agent_lemon_lime.harness.base import ExecResult

logger = logging.getLogger(__name__)


class OpenshellSandbox:
    """Sandbox backed by NVIDIA OpenShell cluster (requires live cluster in production)."""

    def __init__(
        self,
        *,
        cluster: str | None = None,
        timeout: float = 30.0,
        ready_timeout_seconds: float = 120.0,
        workdir: str | None = None,
        setup_command: str | None = None,
        providers: list[str] | None = None,
        policy: Any | None = None,
        image: str | None = None,
        _client: Any | None = None,  # injected in tests; openshell type not available statically
    ) -> None:
        self._cluster = cluster
        self._timeout = timeout
        self._ready_timeout = ready_timeout_seconds
        self._workdir = workdir
        self._setup_command = setup_command
        self._providers = providers or []
        self._policy = policy
        self._image = image
        self._test_client: Any | None = _client
        # openshell types are not available statically; use Any for dynamic dispatch
        self._session: Any | None = None
        self._client_instance: Any | None = None
        self._active = False
        self._enter_depth = 0
        self._cached_draft_policy: list[Any] = []

    @property
    def is_active(self) -> bool:
        return self._active

    def __enter__(self) -> OpenshellSandbox:
        self._enter_depth += 1
        if self._enter_depth > 1:
            return self
        if self._test_client is not None:
            self._client_instance = self._test_client
        else:
            import openshell

            self._client_instance = openshell.SandboxClient.from_active_cluster(
                cluster=self._cluster,
                timeout=self._timeout,
            )
        spec = self._build_spec()
        self._session = self._client_instance.create_session(spec=spec)
        sandbox_name = self._session.sandbox.name
        try:
            self._client_instance.wait_ready(
                sandbox_name,
                timeout_seconds=self._ready_timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Sandbox '{sandbox_name}' was not ready within "
                f"{self._ready_timeout:.0f}s. Increase the timeout with "
                f"sandbox.ready_timeout_seconds in agent-lemon.yaml "
                f"or check cluster health with: openshell status"
            ) from exc
        if self._workdir is not None:
            self._upload_workdir()
        if self._setup_command is not None:
            self._run_setup()
        self._active = True
        return self

    def __exit__(self, *args: object) -> None:
        self._enter_depth -= 1
        if self._enter_depth > 0:
            return
        try:
            self._cached_draft_policy = self._fetch_draft_policy()
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

    def _build_spec(self) -> Any | None:
        """Build a SandboxSpec with providers, policy, and image."""
        if not self._providers and self._policy is None and self._image is None:
            return None
        from openshell._proto import datamodel_pb2

        spec = datamodel_pb2.SandboxSpec()
        if self._providers:
            spec.providers.extend(self._providers)
        if self._policy is not None:
            spec.policy.CopyFrom(self._policy)
        if self._image is not None:
            spec.template.image = self._image
        return spec

    def _run_setup(self) -> None:
        """Run the setup command inside the sandbox (e.g. install deps)."""
        assert self._session is not None
        raw = self._session.exec(["sh", "-c", self._setup_command])
        if raw.exit_code != 0:
            raise RuntimeError(
                f"Sandbox setup command failed (exit {raw.exit_code}): "
                f"{raw.stderr.strip()}"
            )

    def _upload_workdir(self) -> None:
        """Upload local project files into the remote sandbox."""
        assert self._session is not None
        sandbox_name = self._session.sandbox.name
        result = subprocess.run(
            ["openshell", "sandbox", "upload", sandbox_name, self._workdir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload workdir to sandbox {sandbox_name}: "
                f"{result.stderr.strip()}"
            )

    @property
    def sandbox_name(self) -> str | None:
        if self._session is None:
            return None
        return self._session.sandbox.name

    def _fetch_draft_policy(self) -> list[Any]:
        """Fetch draft policy chunks from the OpenShell gateway (live call)."""
        if self._client_instance is None or self._session is None:
            return []
        try:
            from openshell._proto import openshell_pb2

            request = openshell_pb2.GetDraftPolicyRequest(
                name=self._session.sandbox.name,
            )
            response = self._client_instance._stub.GetDraftPolicy(
                request, timeout=self._timeout,
            )
            return list(response.chunks)
        except Exception:
            logger.warning("Failed to fetch draft policy", exc_info=True)
            return []

    def get_draft_policy(self) -> list[Any]:
        """Return draft policy chunks, using cached data if sandbox is closed."""
        if self._active:
            return self._fetch_draft_policy()
        return self._cached_draft_policy

    _INFERENCE_ENV: dict[str, str] = {
        "ANTHROPIC_BASE_URL": "https://inference.local",
        "OPENAI_BASE_URL": "https://inference.local/v1",
    }

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
        merged_env = dict(self._INFERENCE_ENV) if self._providers else {}
        if env:
            merged_env.update(env)
        raw = self._session.exec(
            command,
            workdir=workdir,
            env=merged_env or None,
            timeout_seconds=timeout_seconds,
        )
        return ExecResult(
            exit_code=raw.exit_code,
            stdout=raw.stdout,
            stderr=raw.stderr,
        )
