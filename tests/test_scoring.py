"""Tests for the scoring bridge module."""

from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.scoring import (
    JudgeScore,
    _eval_output_to_record,
    score_eval_output,
)
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput


def test_judge_score_defaults():
    s = JudgeScore(value=True)
    assert s.value is True
    assert s.rationale == ""
    assert s.error is None


def test_judge_score_with_rationale():
    s = JudgeScore(value=4, rationale="Good output quality")
    assert s.value == 4
    assert s.rationale == "Good output quality"


def test_judge_score_with_error():
    s = JudgeScore(value=None, error="Judge failed to parse")
    assert s.value is None
    assert s.error == "Judge failed to parse"


def test_eval_output_to_record_converts_fields():
    output = EvalOutput(
        exit_code=1,
        stdout="hello",
        stderr="warn",
        domain=EvalDomain.SAFETY,
    )
    record = _eval_output_to_record(output)
    assert record["exit_code"] == 1
    assert record["stdout"] == "hello"
    assert record["stderr"] == "warn"
    assert record["files"] == {}
    assert record["tool_calls"] == []
    assert record["annotations"] == {}


def test_score_eval_output_inline_check_pass():
    output = EvalOutput(
        exit_code=0,
        stdout="ok",
        stderr="",
        domain=EvalDomain.CORRECTNESS,
    )
    judges = [
        JudgeConfig(
            name="exit-code",
            check=('return outputs.get("exit_code", 1) == 0, "non-zero exit"'),
        ),
    ]
    scores = score_eval_output(output, judges)
    assert "exit-code" in scores
    assert scores["exit-code"].value is True


def test_score_eval_output_inline_check_fail():
    output = EvalOutput(
        exit_code=1,
        stdout="",
        stderr="error",
        domain=EvalDomain.CORRECTNESS,
    )
    judges = [
        JudgeConfig(
            name="exit-code",
            check=('return outputs.get("exit_code", 1) == 0, "non-zero exit"'),
        ),
    ]
    scores = score_eval_output(output, judges)
    assert scores["exit-code"].value is False
    assert "non-zero exit" in scores["exit-code"].rationale


def test_score_eval_output_empty_judges():
    output = EvalOutput(
        exit_code=0,
        stdout="ok",
        stderr="",
        domain=EvalDomain.CORRECTNESS,
    )
    scores = score_eval_output(output, [])
    assert scores == {}
