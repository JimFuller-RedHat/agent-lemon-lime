"""End-to-end integration test: agent-lemon discovers hello world agent."""

import pathlib

import pytest
from agent_eval.config import JudgeConfig

from agent_lemon_lime.agents.lemon import LemonAgent
from agent_lemon_lime.config import LemonConfig
from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import EvalDomain
from agent_lemon_lime.harness.mock import MockSandbox

HELLO_WORLD_CONFIG = pathlib.Path(__file__).parent.parent / "examples/hello_world/agent-lemon.yaml"


@pytest.mark.skipif(not HELLO_WORLD_CONFIG.exists(), reason="hello world example not yet created")
def test_hello_world_discovery():
    config = LemonConfig.from_file(HELLO_WORLD_CONFIG)
    sandbox = MockSandbox()
    sandbox.register_command(
        ["python", "examples/hello_world/agent.py"],
        stdout="Hello, World!\n",
        exit_code=0,
    )
    cases = [
        EvalCase(
            name="hello-world-runs",
            input=EvalInput(
                command=["python", "examples/hello_world/agent.py"],
            ),
            judges=[
                JudgeConfig(
                    name="exit-code",
                    check=('return outputs.get("exit_code", 1) == 0, "non-zero exit"'),
                ),
                JudgeConfig(
                    name="output-contains",
                    check=('return "Hello" in outputs.get("stdout", ""), "missing Hello"'),
                ),
            ],
            domain=EvalDomain.CORRECTNESS,
        )
    ]
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=cases)
    assert result.report.summary.total == 1
    assert result.report.summary.passed == 1
    assert result.scp.filesystem_policy is not None  # SCP was built
