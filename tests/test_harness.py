"""Tests for sandbox abstraction and MockSandbox."""

import importlib.util
from unittest.mock import MagicMock

import pytest

from agent_lemon_lime.harness.base import AbstractSandbox, ExecResult
from agent_lemon_lime.harness.mock import MockSandbox
from agent_lemon_lime.harness.openshell import OpenshellSandbox

_openshell_importable = importlib.util.find_spec("openshell") is not None
try:
    import openshell as _openshell  # noqa: F401
except Exception:
    _openshell_importable = False


def test_exec_result_success():
    r = ExecResult(exit_code=0, stdout="ok", stderr="")
    assert r.success is True


def test_exec_result_failure():
    r = ExecResult(exit_code=1, stdout="", stderr="error")
    assert r.success is False


def test_mock_sandbox_exec_registered():
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    with sandbox:
        result = sandbox.exec(["echo", "hello"])
    assert result.exit_code == 0
    assert result.stdout == "hello\n"


def test_mock_sandbox_exec_unregistered():
    sandbox = MockSandbox()
    with sandbox:
        result = sandbox.exec(["unknown", "cmd"])
    assert result.exit_code == 1
    assert "not registered" in result.stderr


def test_mock_sandbox_records_call_count():
    sandbox = MockSandbox()
    sandbox.register_command(["python", "--version"], stdout="Python 3.13\n", exit_code=0)
    with sandbox:
        sandbox.exec(["python", "--version"])
        sandbox.exec(["python", "--version"])
    assert sandbox.call_count(["python", "--version"]) == 2


def test_mock_sandbox_is_active_only_in_context():
    sandbox = MockSandbox()
    assert not sandbox.is_active
    with sandbox:
        assert sandbox.is_active
    assert not sandbox.is_active


def test_mock_sandbox_requires_context_manager():
    sandbox = MockSandbox()
    with pytest.raises(RuntimeError, match="context manager"):
        sandbox.exec(["echo", "hi"])


def test_abstract_sandbox_protocol():
    """MockSandbox satisfies the AbstractSandbox protocol."""
    sandbox = MockSandbox()
    assert isinstance(sandbox, AbstractSandbox)


def test_openshell_sandbox_exec_delegates():
    """OpenshellSandbox.exec delegates to the openshell session."""
    mock_session = MagicMock()
    mock_session.exec.return_value = MagicMock(exit_code=0, stdout="hello\n", stderr="")
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client)
    with sandbox:
        result = sandbox.exec(["echo", "hello"])

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    mock_session.exec.assert_called_once()


def test_openshell_sandbox_is_active_lifecycle():
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client)
    assert not sandbox.is_active
    with sandbox:
        assert sandbox.is_active
    assert not sandbox.is_active


def test_openshell_sandbox_requires_context():
    sandbox = OpenshellSandbox()
    with pytest.raises(RuntimeError, match="context manager"):
        sandbox.exec(["echo", "hi"])


@pytest.mark.integration
@pytest.mark.skipif(
    not _openshell_importable,
    reason="openshell library not functional in this environment",
)
def test_openshell_real_cluster():
    """Integration: requires a live OpenShell cluster."""
    sandbox = OpenshellSandbox()
    with sandbox:
        result = sandbox.exec(["echo", "integration-ok"])
    assert result.exit_code == 0
    assert "integration-ok" in result.stdout
