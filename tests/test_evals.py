"""Tests for standard evaluators and EvalRunner."""

import pytest

from agent_lemon_lime.evals.runner import EvalCase, EvalInput, EvalRunner
from agent_lemon_lime.evals.skills import SkillLoader
from agent_lemon_lime.evals.standard import (
    EvalDomain,
    EvalOutput,
    ExitCodeEvaluator,
    NoErrorOutputEvaluator,
    OutputContainsEvaluator,
)
from agent_lemon_lime.harness.mock import MockSandbox


def test_exit_code_evaluator_pass():
    ev = ExitCodeEvaluator()
    out = EvalOutput(exit_code=0, stdout="ok", stderr="", domain=EvalDomain.CORRECTNESS)
    assert ev.evaluate(out) is True


def test_exit_code_evaluator_fail():
    ev = ExitCodeEvaluator()
    out = EvalOutput(exit_code=1, stdout="", stderr="error", domain=EvalDomain.STABILITY)
    assert ev.evaluate(out) is False


def test_no_error_output_evaluator_clean():
    ev = NoErrorOutputEvaluator()
    out = EvalOutput(exit_code=0, stdout="ok", stderr="", domain=EvalDomain.SAFETY)
    assert ev.evaluate(out) is True


def test_no_error_output_evaluator_noisy():
    ev = NoErrorOutputEvaluator()
    out = EvalOutput(exit_code=0, stdout="ok", stderr="WARN: something", domain=EvalDomain.SAFETY)
    assert ev.evaluate(out) is False


def test_output_contains_evaluator_match():
    ev = OutputContainsEvaluator(expected="hello world")
    out = EvalOutput(exit_code=0, stdout="hello world!", stderr="", domain=EvalDomain.CORRECTNESS)
    assert ev.evaluate(out) is True


def test_output_contains_evaluator_no_match():
    ev = OutputContainsEvaluator(expected="hello world")
    out = EvalOutput(exit_code=0, stdout="goodbye", stderr="", domain=EvalDomain.CORRECTNESS)
    assert ev.evaluate(out) is False


def test_eval_runner_passes():
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    runner = EvalRunner()
    cases = [
        EvalCase(
            name="echo-passes",
            input=EvalInput(command=["echo", "hello"]),
            evaluators=[ExitCodeEvaluator(), OutputContainsEvaluator(expected="hello")],
        )
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].name == "echo-passes"
    assert results[0].failures == []


def test_eval_runner_fails():
    sandbox = MockSandbox()
    sandbox.register_command(["bad", "cmd"], stdout="", exit_code=1)
    runner = EvalRunner()
    cases = [
        EvalCase(
            name="fails",
            input=EvalInput(command=["bad", "cmd"]),
            evaluators=[ExitCodeEvaluator()],
        )
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert results[0].passed is False
    assert "ExitCodeEvaluator" in results[0].failures


def test_eval_runner_multiple_cases():
    sandbox = MockSandbox()
    sandbox.register_command(["ok"], stdout="ok\n", exit_code=0)
    sandbox.register_command(["fail"], stdout="", exit_code=1)
    runner = EvalRunner()
    cases = [
        EvalCase(name="pass", input=EvalInput(command=["ok"]), evaluators=[ExitCodeEvaluator()]),
        EvalCase(name="fail", input=EvalInput(command=["fail"]), evaluators=[ExitCodeEvaluator()]),
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert results[0].passed is True
    assert results[1].passed is False


def test_skill_loader_loads_local_dir(tmp_path):
    (tmp_path / "my_skill.md").write_text("# My Skill\n\nDo something useful.")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "my_skill"
    assert "Do something useful" in skills[0].content


def test_skill_loader_ignores_non_markdown(tmp_path):
    (tmp_path / "ignored.txt").write_text("not a skill")
    (tmp_path / "skill.md").write_text("# Skill")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "skill"


def test_skill_loader_missing_dir_raises(tmp_path):
    loader = SkillLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_dir(tmp_path / "nonexistent")


def test_skill_loader_loads_nested(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("# Nested")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert any(s.name == "nested" for s in skills)
