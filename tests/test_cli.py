"""Tests for the agent-lemon and agent-lime CLIs."""

from typer.testing import CliRunner

from agent_lemon_lime.cli.lemon import app
from agent_lemon_lime.cli.lime import app as lime_app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_discover_missing_config(tmp_path):
    result = runner.invoke(app, ["discover", "--project-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "agent-lemon.yaml" in result.output


def test_cli_init_creates_config(tmp_path):
    result = runner.invoke(app, ["init", "--project-dir", str(tmp_path), "--name", "my-agent"])
    assert result.exit_code == 0
    config_file = tmp_path / "agent-lemon.yaml"
    assert config_file.exists()
    assert "my-agent" in config_file.read_text()


def test_cli_action_generates_workflow(tmp_path):
    result = runner.invoke(app, ["action", "--output", str(tmp_path / "agent-lemon.yml")])
    assert result.exit_code == 0
    wf = tmp_path / "agent-lemon.yml"
    assert wf.exists()
    assert "agent-lemon" in wf.read_text()


def test_lime_cli_help():
    result = runner.invoke(lime_app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_lime_cli_monitor_requires_scp(tmp_path):
    result = runner.invoke(lime_app, ["monitor", "--otel", "http://localhost:4317"])
    assert result.exit_code != 0
    assert "scp" in result.output.lower()
