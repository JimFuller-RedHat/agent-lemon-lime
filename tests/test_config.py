"""Tests for LemonConfig — agent-lemon.yaml reader."""

import textwrap
from pathlib import Path

import pytest

from agent_lemon_lime.config import LemonConfig, RunMode, resolve_env

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
    assert RunMode.DISCOVER == "discover"
    assert RunMode.ASSERT == "assert"


def test_config_setup_default_is_none():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.run.setup is None


def test_config_setup_from_yaml():
    yaml_text = "name: my-agent\nrun:\n  command: echo hi\n  setup: uv pip install .\n"
    config = LemonConfig.from_yaml(yaml_text)
    assert config.run.setup == "uv pip install ."


def test_resolve_env_expands_from_os(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    result = resolve_env({"API_KEY": "${MY_KEY}", "PLAIN": "hello"})
    assert result == {"API_KEY": "secret123", "PLAIN": "hello"}


def test_resolve_env_missing_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("NONEXISTENT", raising=False)
    result = resolve_env({"KEY": "${NONEXISTENT}"})
    assert result == {"KEY": ""}


def test_config_sandbox_defaults():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.sandbox.type == "local"
    assert minimal.sandbox.cluster is None
    assert minimal.sandbox.timeout == 30.0
    assert minimal.sandbox.ready_timeout_seconds == 120.0
    assert minimal.sandbox.auto_start_gateway is True


def test_config_sandbox_openshell():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        sandbox:
          type: openshell
          cluster: my-cluster
          timeout: 60.0
          ready_timeout_seconds: 180.0
          auto_start_gateway: false
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert config.sandbox.type == "openshell"
    assert config.sandbox.cluster == "my-cluster"
    assert config.sandbox.timeout == 60.0
    assert config.sandbox.ready_timeout_seconds == 180.0
    assert config.sandbox.auto_start_gateway is False


def test_config_backends_default_empty():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.evals.backends == []


def test_config_backends_from_yaml():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        evals:
          backends:
            - type: inspect
              model: anthropic/claude-opus-4-6
              tasks:
                - arc
                - hellaswag
              score_threshold: 0.8
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert len(config.evals.backends) == 1
    b = config.evals.backends[0]
    assert b.type == "inspect"
    assert b.model == "anthropic/claude-opus-4-6"
    assert b.tasks == ["arc", "hellaswag"]
    assert b.score_threshold == 0.8


def test_config_backend_score_threshold_default():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        evals:
          backends:
            - type: inspect
              model: openai/gpt-4o
              tasks:
                - arc
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert config.evals.backends[0].score_threshold == 1.0


def test_report_config_model_field():
    yaml_text = """\
name: test-agent
version: "0.1.0"
run:
  command: "python agent.py"
report:
  output: ".agent-lemon/report.md"
  model: anthropic/claude-sonnet-4-20250514
"""
    config = LemonConfig.from_yaml(yaml_text)
    assert config.report.model == "anthropic/claude-sonnet-4-20250514"


def test_report_config_model_defaults_to_none():
    yaml_text = """\
name: test-agent
version: "0.1.0"
run:
  command: "python agent.py"
"""
    config = LemonConfig.from_yaml(yaml_text)
    assert config.report.model is None
