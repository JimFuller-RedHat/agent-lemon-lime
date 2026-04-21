"""EvalReport model."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.scp.models import SystemCapabilityProfile


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class InferenceConfig:
    provider: str | None = None
    model: str | None = None
    sandbox_type: str = "local"


@dataclass
class EvalReport:
    agent_name: str
    generated_at: str
    summary: EvalSummary
    results: list[EvalResult]
    scp: SystemCapabilityProfile
    violations: list[str] = field(default_factory=list)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
