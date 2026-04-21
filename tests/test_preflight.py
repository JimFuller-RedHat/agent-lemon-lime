"""Tests for OpenShell pre-flight checks and sandbox factory."""

import subprocess
import textwrap
from unittest.mock import patch

import pytest
import typer

from agent_lemon_lime.config import LemonConfig, SandboxConfig


def _make_config(sandbox_type: str = "local") -> LemonConfig:
    yaml_text = textwrap.dedent(f"""\
        name: test-agent
        run:
          command: echo hi
        sandbox:
          type: {sandbox_type}
          auto_start_gateway: true
    """)
    return LemonConfig.from_yaml(yaml_text)


def test_resolve_sandbox_config_no_overrides():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(config, sandbox_flag=None, no_auto_start=False)
    assert resolved.type == "openshell"
    assert resolved.auto_start_gateway is True


def test_resolve_sandbox_config_flag_overrides_type():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("local")
    resolved = _resolve_sandbox_config(config, sandbox_flag="openshell", no_auto_start=False)
    assert resolved.type == "openshell"


def test_resolve_sandbox_config_no_auto_start_overrides():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(config, sandbox_flag=None, no_auto_start=True)
    assert resolved.auto_start_gateway is False


def test_resolve_sandbox_config_both_overrides():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("local")
    resolved = _resolve_sandbox_config(config, sandbox_flag="openshell", no_auto_start=True)
    assert resolved.type == "openshell"
    assert resolved.auto_start_gateway is False


def _mock_run(responses: dict[tuple[str, ...], subprocess.CompletedProcess]):
    """Return a side_effect function for subprocess.run based on command tuples."""

    def side_effect(cmd, **kwargs):
        key = tuple(cmd)
        if key in responses:
            return responses[key]
        raise ValueError(f"Unexpected command: {cmd}")

    return side_effect


def test_preflight_gateway_running_providers_exist():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Gateway running",
            stderr="",
        ),
        ("openshell", "provider", "list"): subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="anthropic\n",
            stderr="",
        ),
    }
    config = SandboxConfig(type="openshell")
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        _openshell_preflight(config)  # should not raise


def test_preflight_gateway_down_auto_start():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    call_log = []

    def side_effect(cmd, **kwargs):
        call_log.append(tuple(cmd))
        if tuple(cmd) == ("openshell", "status"):
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        if tuple(cmd) == ("openshell", "gateway", "start"):
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="Started", stderr="")
        if tuple(cmd) == ("openshell", "provider", "list"):
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="anthropic\n", stderr=""
            )
        raise ValueError(f"Unexpected command: {cmd}")

    config = SandboxConfig(type="openshell", auto_start_gateway=True)
    with patch("subprocess.run", side_effect=side_effect):
        _openshell_preflight(config)

    assert ("openshell", "gateway", "start") in call_log


def test_preflight_gateway_down_no_auto_start():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="",
        ),
    }
    config = SandboxConfig(type="openshell", auto_start_gateway=False)
    with patch("subprocess.run", side_effect=_mock_run(responses)), pytest.raises(typer.Exit):
        _openshell_preflight(config)


def test_preflight_no_providers():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Gateway running",
            stderr="",
        ),
        ("openshell", "provider", "list"): subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        ),
    }
    config = SandboxConfig(type="openshell")
    with patch("subprocess.run", side_effect=_mock_run(responses)), pytest.raises(typer.Exit):
        _openshell_preflight(config)


def test_create_sandbox_local():
    from agent_lemon_lime.cli.lemon import _create_sandbox
    from agent_lemon_lime.harness.local import LocalSandbox

    config = SandboxConfig(type="local")
    sandbox = _create_sandbox(config, workdir="/tmp/test")
    assert isinstance(sandbox, LocalSandbox)


def test_create_sandbox_openshell():
    from agent_lemon_lime.cli.lemon import _create_sandbox
    from agent_lemon_lime.harness.openshell import OpenshellSandbox

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Running", stderr="",
        ),
        ("openshell", "provider", "list"): subprocess.CompletedProcess(
            args=[], returncode=0, stdout="anthropic\n", stderr="",
        ),
    }
    config = SandboxConfig(type="openshell")
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        sandbox = _create_sandbox(config, workdir="/tmp/test")
    assert isinstance(sandbox, OpenshellSandbox)
    assert sandbox._workdir == "/tmp/test"


# --- _configure_inference tests ---


def test_configure_inference_skipped_when_no_provider_or_model():
    from agent_lemon_lime.cli.lemon import _configure_inference

    config = SandboxConfig(type="openshell")
    with patch("subprocess.run") as mock_run:
        _configure_inference(config)
    mock_run.assert_not_called()


def test_configure_inference_set_both():
    from agent_lemon_lime.cli.lemon import _configure_inference

    model = "claude-sonnet-4-20250514"
    config = SandboxConfig(type="openshell", provider="anthropic", model=model)
    cmd = ("openshell", "inference", "set", "--provider", "anthropic", "--model", model)
    responses = {
        cmd: subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr=""),
    }
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        _configure_inference(config)


def test_configure_inference_provider_only_exits():
    from agent_lemon_lime.cli.lemon import _configure_inference

    config = SandboxConfig(type="openshell", provider="anthropic")
    with pytest.raises(typer.Exit):
        _configure_inference(config)


def test_configure_inference_model_only_exits():
    from agent_lemon_lime.cli.lemon import _configure_inference

    config = SandboxConfig(type="openshell", model="claude-sonnet-4-20250514")
    with pytest.raises(typer.Exit):
        _configure_inference(config)


def test_configure_inference_failure_exits():
    from agent_lemon_lime.cli.lemon import _configure_inference

    model = "claude-sonnet-4-20250514"
    config = SandboxConfig(type="openshell", provider="bad", model=model)
    cmd = ("openshell", "inference", "set", "--provider", "bad", "--model", model)
    responses = {
        cmd: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="unknown provider",
        ),
    }
    with patch("subprocess.run", side_effect=_mock_run(responses)), pytest.raises(typer.Exit):
        _configure_inference(config)


# --- _resolve_sandbox_config with provider/model ---


def test_resolve_sandbox_config_provider_flag():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
        provider_flag="anthropic", model_flag=None,
    )
    assert resolved.provider == "anthropic"
    assert resolved.model is None


def test_resolve_sandbox_config_model_flag():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
        provider_flag=None, model_flag="claude-sonnet-4-20250514",
    )
    assert resolved.model == "claude-sonnet-4-20250514"
    assert resolved.provider is None


def test_resolve_sandbox_config_flags_override_yaml():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    yaml_text = textwrap.dedent("""\
        name: test-agent
        run:
          command: echo hi
        sandbox:
          type: openshell
          provider: old-provider
          model: old-model
    """)
    config = LemonConfig.from_yaml(yaml_text)
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
        provider_flag="new-provider", model_flag="new-model",
    )
    assert resolved.provider == "new-provider"
    assert resolved.model == "new-model"


# --- config parsing ---


def test_config_sandbox_provider_model():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        sandbox:
          type: openshell
          provider: anthropic
          model: claude-sonnet-4-20250514
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert config.sandbox.provider == "anthropic"
    assert config.sandbox.model == "claude-sonnet-4-20250514"


def test_config_sandbox_provider_model_defaults():
    config = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert config.sandbox.provider is None
    assert config.sandbox.model is None


# --- discovery policy ---


def test_resolve_sandbox_config_discovery_policy_flag():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
        discovery_policy_flag="/tmp/custom-policy.yaml",
    )
    assert resolved.discovery_policy == "/tmp/custom-policy.yaml"


def test_config_sandbox_discovery_policy():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        sandbox:
          type: openshell
          discovery_policy: policies/discovery.yaml
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert config.sandbox.discovery_policy == "policies/discovery.yaml"


def test_config_sandbox_discovery_policy_default_is_none():
    config = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert config.sandbox.discovery_policy is None


# --- ready timeout ---


def test_resolve_sandbox_config_ready_timeout_flag():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
        ready_timeout_flag=300.0,
    )
    assert resolved.ready_timeout_seconds == 300.0


def test_resolve_sandbox_config_ready_timeout_default():
    from agent_lemon_lime.cli.lemon import _resolve_sandbox_config

    config = _make_config("openshell")
    resolved = _resolve_sandbox_config(
        config, sandbox_flag=None, no_auto_start=False,
    )
    assert resolved.ready_timeout_seconds == 120.0
