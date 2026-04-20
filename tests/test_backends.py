"""Tests for pluggable eval backends."""

import json
import subprocess
from unittest.mock import patch

from agent_lemon_lime.config import BackendConfig
from agent_lemon_lime.evals.backends import BackendResult, InspectBackend, run_backends
from agent_lemon_lime.evals.runner import EvalResult
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


def test_inspect_backend_not_available():
    backend = InspectBackend()
    with patch("shutil.which", return_value=None):
        assert backend.available() is False


def test_inspect_backend_available():
    backend = InspectBackend()
    with patch("shutil.which", return_value="/usr/bin/inspect"):
        assert backend.available() is True


PASSING_LOG = {
    "version": 2,
    "status": "success",
    "eval": {"task": "arc"},
    "results": {
        "scores": [
            {
                "name": "accuracy",
                "metrics": {
                    "accuracy": {"value": 0.92, "name": "accuracy"},
                },
            }
        ],
    },
}

FAILING_LOG = {
    "version": 2,
    "status": "success",
    "eval": {"task": "arc"},
    "results": {
        "scores": [
            {
                "name": "accuracy",
                "metrics": {
                    "accuracy": {"value": 0.65, "name": "accuracy"},
                },
            }
        ],
    },
}

ERROR_LOG = {
    "version": 2,
    "status": "error",
    "eval": {"task": "arc"},
    "error": {"message": "Model returned 429 Too Many Requests"},
    "results": None,
}


def test_inspect_backend_task_passes(tmp_path):
    log_file = tmp_path / "log.json"
    log_file.write_text(json.dumps(PASSING_LOG))

    backend = InspectBackend()

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch("shutil.which", return_value="/usr/bin/inspect"),
        patch("subprocess.run", side_effect=mock_run),
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch(
            "agent_lemon_lime.evals.backends._find_log_file",
            return_value=log_file,
        ),
        patch("shutil.rmtree"),
    ):
        results = backend.run(
            tasks=["arc"],
            model="anthropic/claude-opus-4-6",
            score_threshold=0.8,
        )

    assert len(results) == 1
    assert results[0].name == "inspect::arc"
    assert results[0].passed is True
    assert results[0].score == 0.92


def test_inspect_backend_task_below_threshold(tmp_path):
    log_file = tmp_path / "log.json"
    log_file.write_text(json.dumps(FAILING_LOG))

    backend = InspectBackend()

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch("shutil.which", return_value="/usr/bin/inspect"),
        patch("subprocess.run", side_effect=mock_run),
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch(
            "agent_lemon_lime.evals.backends._find_log_file",
            return_value=log_file,
        ),
        patch("shutil.rmtree"),
    ):
        results = backend.run(tasks=["arc"], model="openai/gpt-4o", score_threshold=0.8)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].score == 0.65
    assert "0.65" in results[0].summary


def test_inspect_backend_task_errors(tmp_path):
    log_file = tmp_path / "log.json"
    log_file.write_text(json.dumps(ERROR_LOG))

    backend = InspectBackend()

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="Error")

    with (
        patch("shutil.which", return_value="/usr/bin/inspect"),
        patch("subprocess.run", side_effect=mock_run),
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch(
            "agent_lemon_lime.evals.backends._find_log_file",
            return_value=log_file,
        ),
        patch("shutil.rmtree"),
    ):
        results = backend.run(tasks=["arc"], model="openai/gpt-4o", score_threshold=1.0)

    assert len(results) == 1
    assert results[0].passed is False
    assert "429" in results[0].details


def test_run_backends_converts_to_eval_results(tmp_path):
    log_file = tmp_path / "log.json"
    log_file.write_text(json.dumps(PASSING_LOG))

    configs = [
        BackendConfig(
            type="inspect",
            model="anthropic/claude-opus-4-6",
            tasks=["arc"],
            score_threshold=0.8,
        ),
    ]

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch("shutil.which", return_value="/usr/bin/inspect"),
        patch("subprocess.run", side_effect=mock_run),
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch("agent_lemon_lime.evals.backends._find_log_file", return_value=log_file),
        patch("shutil.rmtree"),
    ):
        results = run_backends(configs)

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EvalResult)
    assert r.name == "inspect::arc"
    assert r.passed is True
    assert r.domain == EvalDomain.BEHAVIORAL


def test_run_backends_unavailable_backend():
    configs = [
        BackendConfig(
            type="inspect",
            model="openai/gpt-4o",
            tasks=["arc"],
        ),
    ]
    with patch("shutil.which", return_value=None):
        results = run_backends(configs)

    assert len(results) == 1
    assert results[0].passed is False
    assert "not installed" in results[0].output.stdout.lower()


def test_run_backends_empty_config():
    results = run_backends([])
    assert results == []
