"""ReportSynthesizer: builds EvalReport and renders Markdown."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalDomain
from agent_lemon_lime.report.models import EvalReport, EvalSummary, InferenceConfig
from agent_lemon_lime.scp.models import SystemCapabilityProfile

_DOMAIN_LABEL = {
    EvalDomain.SAFETY: "[safety]",
    EvalDomain.STABILITY: "[stability]",
    EvalDomain.CORRECTNESS: "[correctness]",
    EvalDomain.SECURITY: "[security]",
}


class ReportSynthesizer:
    def build(
        self,
        results: list[EvalResult],
        *,
        scp: SystemCapabilityProfile,
        agent_name: str = "",
        violations: list[str] | None = None,
        inference: InferenceConfig | None = None,
    ) -> EvalReport:
        passed = sum(1 for r in results if r.passed)
        return EvalReport(
            agent_name=agent_name,
            generated_at=datetime.now(UTC).isoformat(),
            summary=EvalSummary(
                total=len(results),
                passed=passed,
                failed=len(results) - passed,
            ),
            results=results,
            scp=scp,
            violations=violations or [],
            inference=inference or InferenceConfig(),
        )

    def to_markdown(self, report: EvalReport) -> str:
        s = report.summary
        inf = report.inference
        lines: list[str] = [
            "# Agent Lemon Report",
            "",
            f"**Generated:** {report.generated_at}",
            f"**Sandbox:** {inf.sandbox_type}",
        ]
        if inf.provider:
            lines.append(f"**Provider:** {inf.provider}")
        if inf.model:
            lines.append(f"**Model:** {inf.model}")
        lines += [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total | {s.total} |",
            f"| Passed | {s.passed} |",
            f"| Failed | {s.failed} |",
            f"| Pass Rate | {s.pass_rate:.1%} |",
            "",
        ]
        if report.violations:
            lines += ["## SCP Violations", ""]
            for v in report.violations:
                lines.append(f"- {v}")
            lines.append("")

        lines += ["## Evaluation Results", ""]
        for r in report.results:
            icon = "PASS" if r.passed else "FAIL"
            label = _DOMAIN_LABEL.get(r.domain, "")
            lines.append(f"- [{icon}] {label} {r.name}")
            for failure in r.failures:
                lines.append(f"  - Failed: {failure}")
        lines.append("")
        return "\n".join(lines)

    def write(self, report: EvalReport, *, path: pathlib.Path | str) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(report))

    def to_log(self, report: EvalReport, *, mode: str = "discovery") -> str:
        s = report.summary
        inf = report.inference
        lines: list[str] = [
            "=== Agent Lemon Run Log ===",
            f"Agent:    {report.agent_name}",
            f"Mode:     {mode}",
            f"Sandbox:  {inf.sandbox_type}",
        ]
        if inf.provider:
            lines.append(f"Provider: {inf.provider}")
        if inf.model:
            lines.append(f"Model:    {inf.model}")
        lines += [
            f"Started:  {report.generated_at}",
            "",
        ]
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            cmd = " ".join(r.command) if r.command else r.name
            lines += [
                f"--- {r.name} ---",
                f"Command:  {cmd}",
                f"Exit:     {r.output.exit_code}",
                f"Status:   {status}",
            ]
            if r.failures:
                lines.append(f"Failed:   {', '.join(r.failures)}")
            if r.output.stdout.strip():
                lines += ["[stdout]", r.output.stdout.rstrip()]
            if r.output.stderr.strip():
                lines += ["[stderr]", r.output.stderr.rstrip()]
            lines.append("")

        lines += [
            "--- Summary ---",
            f"Total:    {s.total}",
            f"Passed:   {s.passed}",
            f"Failed:   {s.failed}",
            f"Pass Rate: {s.pass_rate:.1%}",
        ]
        if report.violations:
            lines += ["", "--- SCP Violations ---"]
            lines += [f"  {v}" for v in report.violations]
        return "\n".join(lines) + "\n"

    def write_log(
        self, report: EvalReport, *, path: pathlib.Path | str, mode: str = "discovery"
    ) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_log(report, mode=mode))
