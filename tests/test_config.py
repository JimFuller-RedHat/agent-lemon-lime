"""Tests for LemonConfig — agent-lemon.yaml reader."""

import textwrap
from pathlib import Path

import pytest

from agent_lemon_lime.config import LemonConfig, RunMode

FULL_CONFIG = textwrap.dedent("""\
    name: hello-world-agent
    version: "0.1.0"
    description: "Hello world test agent"

    run:
      command: "python examples/hello_world/agent.py"
      timeout_seconds: 120
      env:
        ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"

    evals:
      directories:
        - examples/hello_world/evals
      skills:
        - path: ./skills
        - git: https://gitlab.cee.redhat.com/product-security/prodsec-skills
          branch: main

    scp:
      output: ".agent-lemon/scp.yaml"
      assert_file: null

    report:
      output: ".agent-lemon/report.md"
      format: markdown
""")


def test_config_loads_full():
    config = LemonConfig.from_yaml(FULL_CONFIG)
    assert config.name == "hello-world-agent"
    assert config.run.command == "python examples/hello_world/agent.py"
    assert config.run.timeout_seconds == 120
    assert config.evals.directories == ["examples/hello_world/evals"]
    assert len(config.evals.skills) == 2
    assert config.evals.skills[0].path == "./skills"
    assert (
        config.evals.skills[1].git
        == "https://gitlab.cee.redhat.com/product-security/prodsec-skills"
    )
    assert config.evals.skills[1].branch == "main"
    assert config.scp.output == ".agent-lemon/scp.yaml"
    assert config.scp.assert_file is None
    assert config.report.format == "markdown"


def test_config_minimal_defaults():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: python agent.py\n")
    assert minimal.run.timeout_seconds == 300
    assert minimal.evals.directories == []
    assert minimal.evals.skills == []
    assert minimal.scp.output == ".agent-lemon/scp.yaml"
    assert minimal.report.format == "markdown"


def test_config_from_dir(tmp_path: Path):
    config_file = tmp_path / "agent-lemon.yaml"
    config_file.write_text("name: test-agent\nrun:\n  command: echo hello\n")
    config = LemonConfig.from_dir(tmp_path)
    assert config.name == "test-agent"


def test_config_from_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="agent-lemon.yaml"):
        LemonConfig.from_dir(tmp_path / "nonexistent")


def test_run_modes_are_strings():
    assert RunMode.DISCOVERY == "discovery"
    assert RunMode.ASSERT == "assert"
