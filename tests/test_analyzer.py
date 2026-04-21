"""Tests for LLM-powered report analyzer."""

from unittest.mock import patch

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.report.analyzer import (
    _build_context,
    analyze_report,
    insert_analysis,
)
from agent_lemon_lime.report.models import EvalReport, EvalSummary, InferenceConfig
from agent_lemon_lime.scp.models import SystemCapabilityProfile


def _make_result(
    name: str, passed: bool, domain: EvalDomain = EvalDomain.CORRECTNESS
) -> EvalResult:
    return EvalResult(
        name=name,
        passed=passed,
        domain=domain,
        output=EvalOutput(
            exit_code=0 if passed else 1,
            stdout="some output" if passed else "error output",
            stderr="" if passed else "something failed",
            domain=domain,
        ),
        failures=[] if passed else ["ExitCodeEvaluator"],
    )


def _make_report(
    results: list[EvalResult] | None = None,
    violations: list[str] | None = None,
) -> EvalReport:
    if results is None:
        results = [
            _make_result("test-pass", True),
            _make_result("test-fail", False),
        ]
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        agent_name="test-agent",
        generated_at="2026-04-20T14:30:00+00:00",
        summary=EvalSummary(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
        ),
        results=results,
        scp=SystemCapabilityProfile(),
        violations=violations or [],
        inference=InferenceConfig(),
    )


def test_build_context_includes_agent_name():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "test-agent" in ctx


def test_build_context_includes_summary():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "Total: 2" in ctx
    assert "Passed: 1" in ctx
    assert "Failed: 1" in ctx


def test_build_context_includes_result_details():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "test-pass" in ctx
    assert "test-fail" in ctx
    assert "PASS" in ctx
    assert "FAIL" in ctx


def test_build_context_includes_failures():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "ExitCodeEvaluator" in ctx


def test_build_context_includes_violations():
    report = _make_report(violations=["Unauthorized host 'evil.com'"])
    ctx = _build_context(report, config_yaml="")
    assert "Unauthorized host 'evil.com'" in ctx


def test_build_context_includes_config_yaml():
    report = _make_report()
    config_yaml = "name: my-agent\nrun:\n  command: python agent.py\n"
    ctx = _build_context(report, config_yaml=config_yaml)
    assert "name: my-agent" in ctx


def test_build_context_includes_log_text():
    report = _make_report()
    log_text = "=== Agent Lemon Run Log ===\nAgent: test-agent\nDEBUG: tool call started"
    ctx = _build_context(report, config_yaml="", log_text=log_text)
    assert "Full Run Log" in ctx
    assert "DEBUG: tool call started" in ctx


def test_build_context_no_log_text():
    report = _make_report()
    ctx = _build_context(report, config_yaml="", log_text="")
    assert "Full Run Log" not in ctx


def test_build_context_empty_results():
    report = _make_report(results=[])
    ctx = _build_context(report, config_yaml="")
    assert "Total: 0" in ctx


def test_build_context_no_config_yaml():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "Agent Config" not in ctx


def test_build_context_truncates_long_output():
    long_stdout = "x" * 5000
    result = EvalResult(
        name="verbose-test",
        passed=True,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0,
            stdout=long_stdout,
            stderr="",
            domain=EvalDomain.CORRECTNESS,
        ),
    )
    report = _make_report(results=[result])
    ctx = _build_context(report, config_yaml="")
    assert len(ctx) < len(long_stdout)
    assert "[truncated]" in ctx


def test_insert_analysis_before_summary():
    markdown = (
        "# Agent Lemon Report\n"
        "\n"
        "**Generated:** 2026-04-20T14:30:00+00:00\n"
        "**Sandbox:** local\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Value |\n"
    )
    analysis = "## Analysis\n\n### Executive Summary\nAll good.\n"
    result = insert_analysis(markdown, analysis)
    lines = result.split("\n")
    summary_idx = next(i for i, line in enumerate(lines) if line == "## Summary")
    analysis_idx = next(i for i, line in enumerate(lines) if line == "## Analysis")
    assert analysis_idx < summary_idx


def test_insert_analysis_none_returns_unchanged():
    markdown = "# Agent Lemon Report\n\n## Summary\n"
    result = insert_analysis(markdown, None)
    assert result == markdown


def test_insert_analysis_empty_string_returns_unchanged():
    markdown = "# Agent Lemon Report\n\n## Summary\n"
    result = insert_analysis(markdown, "")
    assert result == markdown


def test_analyze_report_anthropic_provider():
    report = _make_report()
    with patch("agent_lemon_lime.report.analyzer.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "## Analysis\n\nGood."
        result = analyze_report(report, model="anthropic/claude-sonnet-4-20250514")
    assert result == "## Analysis\n\nGood."
    mock_call_llm.assert_called_once()


def test_analyze_report_openai_provider():
    report = _make_report()
    with patch("agent_lemon_lime.report.analyzer.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "## Analysis\n\nFine."
        result = analyze_report(report, model="openai/gpt-4o")
    assert result == "## Analysis\n\nFine."
    mock_call_llm.assert_called_once()


def test_analyze_report_unknown_provider():
    report = _make_report()
    result = analyze_report(report, model="ollama/llama3")
    assert result is None


def test_analyze_report_no_slash_in_model():
    report = _make_report()
    result = analyze_report(report, model="claude-sonnet")
    assert result is None


def test_analyze_report_passes_config_yaml():
    report = _make_report()
    with patch("agent_lemon_lime.report.analyzer.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "## Analysis\n\nOK."
        analyze_report(
            report,
            model="anthropic/claude-sonnet-4-20250514",
            config_yaml="name: test\n",
        )
    mock_call_llm.assert_called_once()
    call_kwargs = mock_call_llm.call_args.kwargs
    assert "name: test" in call_kwargs["user_content"]
