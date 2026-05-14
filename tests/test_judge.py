"""Tests for harness judge integration via scoring bridge."""

from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.scoring import score_eval_output
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput


def _make_output(stdout="I cannot do that.", stderr="", exit_code=0):
    return EvalOutput(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        domain=EvalDomain.SAFETY,
    )


def test_inline_check_pass():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        )
    ]
    scores = score_eval_output(_make_output(exit_code=0), judges)
    assert scores["exit-ok"].value is True


def test_inline_check_fail():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        )
    ]
    scores = score_eval_output(_make_output(exit_code=1), judges)
    assert scores["exit-ok"].value is False
    assert "non-zero" in scores["exit-ok"].rationale


def test_inline_check_stdout_contains():
    judges = [
        JudgeConfig(
            name="has-hello",
            check=('return "hello" in outputs.get("stdout", ""), "missing hello"'),
        )
    ]
    scores = score_eval_output(_make_output(stdout="hello world"), judges)
    assert scores["has-hello"].value is True


def test_inline_check_stderr_empty():
    judges = [
        JudgeConfig(
            name="no-stderr",
            check=(
                'stderr = outputs.get("stderr", "")\n'
                'return stderr.strip() == "", f"stderr: {stderr[:100]}"'
            ),
        )
    ]
    scores = score_eval_output(_make_output(stderr=""), judges)
    assert scores["no-stderr"].value is True

    scores_fail = score_eval_output(_make_output(stderr="WARN: oops"), judges)
    assert scores_fail["no-stderr"].value is False


def test_multiple_judges():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        ),
        JudgeConfig(
            name="has-output",
            check=('return len(outputs.get("stdout", "")) > 0, "empty stdout"'),
        ),
    ]
    scores = score_eval_output(_make_output(stdout="ok", exit_code=0), judges)
    assert len(scores) == 2
    assert scores["exit-ok"].value is True
    assert scores["has-output"].value is True


def test_judge_with_error_returns_error_score():
    judges = [
        JudgeConfig(
            name="broken",
            check="raise ValueError('intentional test error')",
        )
    ]
    scores = score_eval_output(_make_output(), judges)
    assert scores["broken"].error is not None
    assert "intentional" in scores["broken"].error
