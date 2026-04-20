"""Pluggable eval backends: protocol, models, and inspect_ai implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_lemon_lime.evals.standard import EvalDomain


class BackendResult(BaseModel):
    """Result from a single eval backend task."""

    name: str
    passed: bool
    score: float | None = None
    summary: str = ""
    details: str = ""
    domain: EvalDomain = EvalDomain.BEHAVIORAL


@runtime_checkable
class EvalBackend(Protocol):
    """Protocol for external eval frameworks."""

    name: str

    def run(
        self, tasks: list[str], model: str, **kwargs: object
    ) -> list[BackendResult]: ...

    def available(self) -> bool: ...
