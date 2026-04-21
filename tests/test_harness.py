"""Tests for sandbox abstraction and MockSandbox."""

import importlib.util
from unittest.mock import MagicMock, patch

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


def test_openshell_sandbox_uploads_workdir_on_enter():
    """OpenshellSandbox uploads workdir to the sandbox during __enter__."""
    mock_session = MagicMock()
    mock_session.sandbox.name = "sbx-123"
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client, workdir="/tmp/project")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with sandbox:
            pass

    mock_run.assert_called_once_with(
        ["openshell", "sandbox", "upload", "sbx-123", "/tmp/project"],
        capture_output=True,
        text=True,
    )


def test_openshell_sandbox_upload_failure_raises():
    """OpenshellSandbox raises RuntimeError when upload fails."""
    mock_session = MagicMock()
    mock_session.sandbox.name = "sbx-456"
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client, workdir="/tmp/project")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="upload failed"
        )
        with pytest.raises(RuntimeError, match="Failed to upload workdir"):
            sandbox.__enter__()


def test_openshell_sandbox_skips_upload_without_workdir():
    """OpenshellSandbox skips upload when no workdir is set."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client)
    with patch("subprocess.run") as mock_run:
        with sandbox:
            pass

    mock_run.assert_not_called()


def test_openshell_sandbox_runs_setup_command():
    """OpenshellSandbox runs setup command after upload."""
    mock_session = MagicMock()
    mock_session.sandbox.name = "sbx-789"
    mock_session.exec.return_value = MagicMock(exit_code=0, stdout="", stderr="")
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(
        _client=mock_client,
        workdir="/tmp/project",
        setup_command="uv pip install .",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with sandbox:
            pass

    mock_session.exec.assert_called_once_with(
        ["sh", "-c", "uv pip install ."],
    )


def test_openshell_sandbox_setup_failure_raises():
    """OpenshellSandbox raises RuntimeError when setup command fails."""
    mock_session = MagicMock()
    mock_session.sandbox.name = "sbx-fail"
    mock_session.exec.return_value = MagicMock(
        exit_code=1, stdout="", stderr="pip error"
    )
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(
        _client=mock_client,
        workdir="/tmp/project",
        setup_command="uv pip install .",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with pytest.raises(RuntimeError, match="setup command failed"):
            sandbox.__enter__()


def test_openshell_sandbox_skips_setup_without_command():
    """OpenshellSandbox skips setup when no setup_command is set."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client)
    with sandbox:
        pass

    mock_session.exec.assert_not_called()


def test_openshell_sandbox_caches_draft_policy_on_exit():
    """get_draft_policy returns cached data after sandbox exits."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    fake_chunk = MagicMock()
    fake_chunk.rule_name = "httpbin"

    sandbox = OpenshellSandbox(_client=mock_client)
    with patch.object(
        sandbox, "_fetch_draft_policy", return_value=[fake_chunk],
    ):
        with sandbox:
            pass

    assert not sandbox.is_active
    policy = sandbox.get_draft_policy()
    assert len(policy) == 1
    assert policy[0].rule_name == "httpbin"


def test_openshell_sandbox_reentrant():
    """OpenshellSandbox supports nested context manager calls."""
    mock_session = MagicMock()
    mock_session.exec.return_value = MagicMock(exit_code=0, stdout="ok", stderr="")
    mock_client = MagicMock()
    mock_client.create_session.return_value = mock_session

    sandbox = OpenshellSandbox(_client=mock_client)
    with sandbox:
        assert sandbox.is_active
        with sandbox:
            assert sandbox.is_active
            result = sandbox.exec(["echo", "nested"])
            assert result.exit_code == 0
        assert sandbox.is_active
    assert not sandbox.is_active
    mock_client.create_session.assert_called_once()


def test_mock_sandbox_reentrant():
    """MockSandbox supports nested context manager calls."""
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hi"], stdout="hi\n")
    with sandbox:
        assert sandbox.is_active
        with sandbox:
            assert sandbox.is_active
            result = sandbox.exec(["echo", "hi"])
            assert result.stdout == "hi\n"
        assert sandbox.is_active
    assert not sandbox.is_active


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
