"""Tests for standard evaluators and EvalRunner."""

import warnings

import pytest

from agent_lemon_lime.evals.loader import load_cases_from_dir
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


def test_loader_converts_prompt_case_when_run_command_provided(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: greeting-responds\n"
        "    description: Says hello\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
        "    expected_output: Hello\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].name == "greeting-responds"
    assert cases[0].input.command == ["python", "agent.py", "--prompt", "Say hello."]


def test_loader_warns_and_skips_prompt_case_without_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: prompt-only\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = load_cases_from_dir(tmp_path)
    assert cases == []
    assert any(
        issubclass(w.category, UserWarning) and "prompt-only" in w.message.args[0]
        for w in caught
    )


def test_loader_command_case_unaffected_by_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: exits-cleanly\n"
        "    input:\n"
        "      command: ['python', 'agent.py']\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].input.command == ["python", "agent.py"]


def test_load_cases_from_sandbox():
    """load_cases_from_sandbox reads eval YAML from inside the sandbox."""
    import textwrap

    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.evals.loader import load_cases_from_sandbox

    config = LemonConfig.from_yaml(textwrap.dedent("""\
        name: test-agent
        run:
          command: python agent.py
        evals:
          directories:
            - evals/
    """))

    yaml_content = textwrap.dedent("""\
        cases:
          - name: sandbox-case
            input:
              command: ['echo', 'hello']
            expected_output: hello
    """)

    sandbox = MockSandbox()
    sandbox.register_command(
        ["find", "evals/", "-name", "*.yaml", "-type", "f"],
        stdout="evals/test.yaml\n",
    )
    sandbox.register_command(
        ["cat", "evals/test.yaml"],
        stdout=yaml_content,
    )

    with sandbox:
        cases = load_cases_from_sandbox(config, sandbox=sandbox)

    assert len(cases) == 1
    assert cases[0].name == "sandbox-case"
    assert cases[0].input.command == ["echo", "hello"]


def test_load_cases_from_sandbox_missing_dir():
    """load_cases_from_sandbox returns empty list when find fails."""
    import textwrap

    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.evals.loader import load_cases_from_sandbox

    config = LemonConfig.from_yaml(textwrap.dedent("""\
        name: test-agent
        run:
          command: python agent.py
        evals:
          directories:
            - evals/
    """))

    sandbox = MockSandbox()
    sandbox.register_command(
        ["find", "evals/", "-name", "*.yaml", "-type", "f"],
        exit_code=1,
        stderr="No such file or directory",
    )

    with sandbox:
        cases = load_cases_from_sandbox(config, sandbox=sandbox)

    assert cases == []
