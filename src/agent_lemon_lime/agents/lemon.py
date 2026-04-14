"""Agent Lemon: eval orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from agent_lemon_lime.config import LemonConfig, RunMode
from agent_lemon_lime.evals.runner import EvalCase, EvalRunner
from agent_lemon_lime.harness.base import AbstractSandbox
from agent_lemon_lime.report.models import EvalReport
from agent_lemon_lime.report.synthesizer import ReportSynthesizer
from agent_lemon_lime.scp.models import SystemCapabilityProfile


@dataclass
class LemonRunResult:
    mode: str
    scp: SystemCapabilityProfile
    report: EvalReport
    violations: list[str]


class LemonAgent:
    """Orchestrates Agent Lemon evaluation runs."""

    def __init__(self, *, config: LemonConfig, sandbox: AbstractSandbox) -> None:
        self.config = config
        self.sandbox = sandbox
        self._runner = EvalRunner()
        self._synthesizer = ReportSynthesizer()

    def _build_observed_scp(self, eval_cases: list[EvalCase]) -> SystemCapabilityProfile:
        """Build an SCP from observed command usage (discovery-mode approximation).

        In a real sandbox integration, we'd parse telemetry. For now, records
        each unique command name as a process observation without network policies.
        """
        seen_commands: set[str] = set()
        for case in eval_cases:
            cmd = case.input.command[0] if case.input.command else ""
            if cmd:
                seen_commands.add(cmd)
        return SystemCapabilityProfile()

    def run_discovery(self, *, eval_cases: list[EvalCase]) -> LemonRunResult:
        """Run eval cases in discovery mode, building an observed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.

        Returns:
            LemonRunResult with mode=DISCOVERY, observed SCP, report, and no violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox)
        observed_scp = self._build_observed_scp(eval_cases)
        report = self._synthesizer.build(results, scp=observed_scp)
        return LemonRunResult(
            mode=RunMode.DISCOVERY,
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
    ) -> LemonRunResult:
        """Run eval cases in assert mode, checking compliance against allowed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.
            assert_scp: The reference profile defining permitted capabilities.
            _observed_scp: Override the observed SCP (for testing injection only).

        Returns:
            LemonRunResult with mode=ASSERT, observed SCP, report, and any violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox)
        observed = (
            _observed_scp if _observed_scp is not None else self._build_observed_scp(eval_cases)
        )
        violations = observed.assert_subset_of(assert_scp)
        report = self._synthesizer.build(results, scp=observed, violations=violations)
        return LemonRunResult(
            mode=RunMode.ASSERT,
            scp=observed,
            report=report,
            violations=violations,
        )
