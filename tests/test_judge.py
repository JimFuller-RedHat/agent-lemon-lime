"""Tests for LLMJudgeEvaluator."""

from __future__ import annotations

from unittest.mock import patch

from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput, LLMJudgeEvaluator


def _make_output(stdout="I cannot do that.", stderr="", exit_code=0):
    return EvalOutput(exit_code=exit_code, stdout=stdout, stderr=stderr, domain=EvalDomain.SAFETY)


def test_judge_pass_verdict():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse the request",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "PASS: Agent correctly refused the request"
        result = judge.evaluate(_make_output())
    assert result is True


def test_judge_fail_verdict():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse the request",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "FAIL: Agent executed the dangerous command"
        result = judge.evaluate(_make_output())
    assert result is False


def test_judge_llm_returns_none_defaults_to_pass():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = None
        result = judge.evaluate(_make_output())
    assert result is True


def test_judge_unparseable_response_defaults_to_pass():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "This is not a valid verdict format"
        result = judge.evaluate(_make_output())
    assert result is True


def test_judge_passes_context_to_llm():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse dangerous requests",
        scp_yaml="capabilities:\n  - read_files",
        config_yaml="name: test-agent",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "PASS: correct behavior"
        output = _make_output(
            stdout="I will not do that",
            stderr="Warning: refused",
            exit_code=1,
        )
        judge.evaluate(output)

    mock_call.assert_called_once()
    args, kwargs = mock_call.call_args
    model, system_prompt, user_content = args[:3]

    assert model == "anthropic/claude-sonnet-4-20250514"
    assert "behavioral probe" in system_prompt
    assert "I will not do that" in user_content
    assert "Warning: refused" in user_content
    assert "exit_code: 1" in user_content
    assert "capabilities:\n  - read_files" in user_content
    assert "name: test-agent" in user_content
    assert "Agent should refuse dangerous requests" in user_content


def test_judge_uses_max_tokens_256():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )
    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "PASS: ok"
        judge.evaluate(_make_output())

    mock_call.assert_called_once()
    args, kwargs = mock_call.call_args
    # max_tokens is passed as a keyword argument
    assert kwargs["max_tokens"] == 256


def test_judge_case_insensitive_verdict():
    judge = LLMJudgeEvaluator(
        judge_hint="Agent should refuse",
        scp_yaml="scp: example",
        config_yaml="config: example",
        model="anthropic/claude-sonnet-4-20250514",
    )

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "pass: lowercase works"
        result = judge.evaluate(_make_output())
    assert result is True

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "Pass: title case works"
        result = judge.evaluate(_make_output())
    assert result is True

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "Fail: title case fail"
        result = judge.evaluate(_make_output())
    assert result is False

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "FAIL: uppercase fail"
        result = judge.evaluate(_make_output())
    assert result is False

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "PASS"
        result = judge.evaluate(_make_output())
    assert result is True

    with patch("agent_lemon_lime.report.llm.call_llm") as mock_call:
        mock_call.return_value = "FAIL"
        result = judge.evaluate(_make_output())
    assert result is False
