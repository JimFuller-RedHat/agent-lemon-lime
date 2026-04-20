"""Agent Lemon: eval orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_lemon_lime.config import LemonConfig, RunMode, SandboxConfig
from agent_lemon_lime.evals.runner import EvalCase, EvalResult, EvalRunner
from agent_lemon_lime.harness.base import AbstractSandbox
from agent_lemon_lime.report.models import EvalReport, InferenceConfig
from agent_lemon_lime.report.synthesizer import ReportSynthesizer
from agent_lemon_lime.scp.converter import from_policy_chunks
from agent_lemon_lime.scp.models import SystemCapabilityProfile


@dataclass
class LemonRunResult:
    mode: str
    scp: SystemCapabilityProfile
    report: EvalReport
    violations: list[str]


class LemonAgent:
    """Orchestrates Agent Lemon evaluation runs."""

    def __init__(
        self,
        *,
        config: LemonConfig,
        sandbox: AbstractSandbox,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self.config = config
        self.sandbox = sandbox
        self._sandbox_config = sandbox_config
        self._runner = EvalRunner()
        self._synthesizer = ReportSynthesizer()

    def _inference_config(self) -> InferenceConfig:
        sc = self._sandbox_config
        if sc is None:
            return InferenceConfig()
        return InferenceConfig(
            provider=sc.provider,
            model=sc.model,
            sandbox_type=sc.type,
        )

    def _build_observed_scp(self) -> SystemCapabilityProfile:
        """Build an SCP from sandbox telemetry.

        For OpenshellSandbox, queries the draft policy produced by
        OpenShell's denial analysis. For other sandbox types, returns
        a permissive profile as a starting template.
        """
        from agent_lemon_lime.harness.openshell import OpenshellSandbox

        if isinstance(self.sandbox, OpenshellSandbox):
            chunks = self.sandbox.get_draft_policy()
            return from_policy_chunks(chunks)
        return SystemCapabilityProfile.permissive()

    def run_discovery(
        self,
        *,
        eval_cases: list[EvalCase],
        on_result: Callable[[EvalResult], None] | None = None,
        backend_results: list[EvalResult] | None = None,
    ) -> LemonRunResult:
        """Run eval cases in discovery mode, building an observed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.
            on_result: Optional callback invoked after each case completes.
            backend_results: Optional pre-computed backend results to include.

        Returns:
            LemonRunResult with mode=DISCOVERY, observed SCP, report, and no violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox, on_result=on_result)
        if backend_results:
            results.extend(backend_results)
        observed_scp = self._build_observed_scp()
        report = self._synthesizer.build(
            results,
            scp=observed_scp,
            agent_name=self.config.name,
            inference=self._inference_config(),
        )
        return LemonRunResult(
            mode=RunMode.DISCOVER,
            scp=observed_scp,
            report=report,
            violations=[],
        )

    def run_assert(
        self,
        *,
        eval_cases: list[EvalCase],
        assert_scp: SystemCapabilityProfile,
        _observed_scp: SystemCapabilityProfile | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
        backend_results: list[EvalResult] | None = None,
    ) -> LemonRunResult:
        """Run eval cases in assert mode, checking compliance against allowed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.
            assert_scp: The reference profile defining permitted capabilities.
            _observed_scp: Override the observed SCP (for testing injection only).
            on_result: Optional callback invoked after each case completes.
            backend_results: Optional pre-computed backend results to include.

        Returns:
            LemonRunResult with mode=ASSERT, observed SCP, report, and any violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox, on_result=on_result)
        if backend_results:
            results.extend(backend_results)
        observed = (
            _observed_scp if _observed_scp is not None else self._build_observed_scp()
        )
        violations = observed.assert_subset_of(assert_scp)
        report = self._synthesizer.build(
            results,
            scp=observed,
            agent_name=self.config.name,
            violations=violations,
            inference=self._inference_config(),
        )
        return LemonRunResult(
            mode=RunMode.ASSERT,
            scp=observed,
            report=report,
            violations=violations,
        )
