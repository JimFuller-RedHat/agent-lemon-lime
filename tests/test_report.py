"""Tests for EvalReport and ReportSynthesizer."""

import pathlib

import pytest

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.report.synthesizer import ReportSynthesizer
from agent_lemon_lime.scp.models import SystemCapabilityProfile


def _make_result(name: str, passed: bool) -> EvalResult:
    return EvalResult(
        name=name,
        passed=passed,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0 if passed else 1,
            stdout="",
            stderr="",
            domain=EvalDomain.CORRECTNESS,
        ),
        failures=[] if passed else ["ExitCodeEvaluator"],
    )


def test_report_summary_counts():
    results = [_make_result("a", True), _make_result("b", True), _make_result("c", False)]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile.permissive())
    assert report.summary.total == 3
    assert report.summary.passed == 2
    assert report.summary.failed == 1
    assert report.summary.pass_rate == pytest.approx(2 / 3)


def test_report_summary_zero_total():
    synth = ReportSynthesizer()
    report = synth.build([], scp=SystemCapabilityProfile())
    assert report.summary.total == 0
    assert report.summary.pass_rate == 0.0


def test_report_markdown_has_key_sections():
    results = [_make_result("test-one", True), _make_result("test-two", False)]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile(), violations=["bad tool"])
    md = synth.to_markdown(report)
    assert "# Agent Lemon Report" in md
    assert "Pass Rate" in md
    assert "## Evaluation Results" in md
    assert "## SCP Violations" in md
    assert "bad tool" in md
    assert "test-one" in md
    assert "test-two" in md


def test_report_markdown_no_violations_section_when_clean():
    results = [_make_result("clean", True)]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile())
    md = synth.to_markdown(report)
    assert "SCP Violations" not in md


def test_report_write_to_file(tmp_path: pathlib.Path):
    results = [_make_result("t", True)]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile())
    out = tmp_path / "report.md"
    synth.write(report, path=out)
    content = out.read_text()
    assert "# Agent Lemon Report" in content
