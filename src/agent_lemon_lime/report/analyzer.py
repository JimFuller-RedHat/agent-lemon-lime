"""LLM-powered report analysis."""

from __future__ import annotations

from agent_lemon_lime.report.llm import call_llm
from agent_lemon_lime.report.models import EvalReport

_MAX_OUTPUT_CHARS = 2000


def _build_context(report: EvalReport, *, config_yaml: str = "", log_text: str = "") -> str:
    s = report.summary
    lines = [
        "=== Agent Eval Results ===",
        f"Agent: {report.agent_name}",
        f"Generated: {report.generated_at}",
        "",
        "--- Summary ---",
        f"Total: {s.total}",
        f"Passed: {s.passed}",
        f"Failed: {s.failed}",
        f"Pass Rate: {s.pass_rate:.1%}",
        "",
        "--- Results ---",
    ]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.domain}::{r.name}")
        if r.failures:
            lines.append(f"  Failures: {', '.join(r.failures)}")
        stdout = r.output.stdout.strip()
        if stdout:
            if len(stdout) > _MAX_OUTPUT_CHARS:
                stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n[truncated]"
            lines.append(f"  stdout: {stdout}")
        stderr = r.output.stderr.strip()
        if stderr:
            if len(stderr) > _MAX_OUTPUT_CHARS:
                stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n[truncated]"
            lines.append(f"  stderr: {stderr}")
    lines.append("")

    if report.violations:
        lines.append("--- SCP Violations ---")
        for v in report.violations:
            lines.append(f"  - {v}")
        lines.append("")

    if config_yaml.strip():
        lines.append("--- Agent Config ---")
        lines.append(config_yaml.strip())
        lines.append("")

    if log_text.strip():
        lines.append("--- Full Run Log ---")
        lines.append(log_text.strip())
        lines.append("")

    return "\n".join(lines)


def insert_analysis(markdown: str, analysis: str | None) -> str:
    if not analysis:
        return markdown
    marker = "## Summary"
    idx = markdown.find(marker)
    if idx == -1:
        return markdown
    return markdown[:idx] + analysis.rstrip() + "\n\n" + markdown[idx:]


ANALYSIS_SYSTEM_PROMPT = """\
You are an AI agent evaluation analyst. Analyze the following eval results \
and produce a report with these sections:

## Analysis

### Executive Summary
2-3 sentences on overall agent health and key findings.

### Anomalies
Unexpected patterns, correlated failures, or regressions. If none found, say so.

### Remediation
Concrete, prioritized suggestions. Reference specific eval names and failure messages.

### Risk Assessment
Overall risk: **HIGH**, **MEDIUM**, or **LOW**
Justification in 1-2 sentences."""


def analyze_report(
    report: EvalReport,
    *,
    model: str,
    config_yaml: str = "",
    log_text: str = "",
) -> str | None:
    context = _build_context(report, config_yaml=config_yaml, log_text=log_text)
    return call_llm(
        model=model,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        user_content=context,
        max_tokens=4096,
    )
