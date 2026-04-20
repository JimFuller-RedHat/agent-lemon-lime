"""Tests for pluggable eval backends."""

from agent_lemon_lime.evals.backends import BackendResult
from agent_lemon_lime.evals.standard import EvalDomain


def test_backend_result_defaults():
    r = BackendResult(name="inspect::arc", passed=True)
    assert r.name == "inspect::arc"
    assert r.passed is True
    assert r.score is None
    assert r.summary == ""
    assert r.details == ""
    assert r.domain == EvalDomain.BEHAVIORAL


def test_backend_result_with_score():
    r = BackendResult(
        name="inspect::hellaswag",
        passed=False,
        score=0.65,
        summary="Score 0.65 below threshold 0.8",
        details="3 of 10 samples failed",
    )
    assert r.score == 0.65
    assert r.passed is False
    assert "0.65" in r.summary
