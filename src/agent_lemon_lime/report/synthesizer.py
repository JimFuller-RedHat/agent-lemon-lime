"""ReportSynthesizer: builds EvalReport and renders Markdown."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalDomain
from agent_lemon_lime.report.models import EvalReport, EvalSummary
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
        violations: list[str] | None = None,
    ) -> EvalReport:
        passed = sum(1 for r in results if r.passed)
        return EvalReport(
            generated_at=datetime.now(UTC).isoformat(),
            summary=EvalSummary(
                total=len(results),
                passed=passed,
                failed=len(results) - passed,
            ),
            results=results,
            scp=scp,
            violations=violations or [],
        )

    def to_markdown(self, report: EvalReport) -> str:
        s = report.summary
        lines: list[str] = [
            "# Agent Lemon Report",
            "",
            f"**Generated:** {report.generated_at}",
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
