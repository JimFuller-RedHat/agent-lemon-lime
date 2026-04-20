"""Pluggable eval backends: protocol, models, and inspect_ai implementation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_lemon_lime.evals.standard import EvalDomain


class BackendResult(BaseModel):
    """Result from a single eval backend task."""

    name: str
    passed: bool
    score: float | None = None
    summary: str = ""
    details: str = ""
    domain: EvalDomain = EvalDomain.BEHAVIORAL


@runtime_checkable
class EvalBackend(Protocol):
    """Protocol for external eval frameworks."""

    name: str

    def run(self, tasks: list[str], model: str, **kwargs: object) -> list[BackendResult]: ...

    def available(self) -> bool: ...


logger = logging.getLogger(__name__)


def _find_log_file(log_dir: Path) -> Path | None:
    """Find the first .json log file in the inspect log directory."""
    json_files = sorted(log_dir.rglob("*.json"))
    return json_files[0] if json_files else None


def _extract_score(results: dict | None) -> float | None:
    """Extract the first metric value from inspect_ai results."""
    if results is None:
        return None
    scores = results.get("scores", [])
    if not scores:
        return None
    metrics = scores[0].get("metrics", {})
    if not metrics:
        return None
    first_metric = next(iter(metrics.values()))
    return first_metric.get("value")


class InspectBackend:
    """Eval backend that shells out to the inspect_ai CLI."""

    name: str = "inspect"

    def available(self) -> bool:
        return shutil.which("inspect") is not None

    def run(
        self,
        tasks: list[str],
        model: str,
        **kwargs: object,
    ) -> list[BackendResult]:
        score_threshold = float(kwargs.get("score_threshold", 1.0))
        results: list[BackendResult] = []
        for task in tasks:
            result = self._run_task(task, model, score_threshold)
            results.append(result)
        return results

    def _run_task(self, task: str, model: str, score_threshold: float) -> BackendResult:
        log_dir = tempfile.mkdtemp(prefix="agent-lemon-inspect-")
        try:
            cmd = [
                "inspect",
                "eval",
                task,
                "--model",
                model,
                "--log-dir",
                log_dir,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            log_file = _find_log_file(Path(log_dir))
            if log_file is None:
                return BackendResult(
                    name=f"inspect::{task}",
                    passed=False,
                    summary=f"No log file produced for task '{task}'",
                    details=proc.stderr,
                )
            log_data = json.loads(log_file.read_text())
            return self._parse_log(task, log_data, score_threshold)
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    def _parse_log(self, task: str, log_data: dict, score_threshold: float) -> BackendResult:
        status = log_data.get("status", "error")
        if status == "error":
            error = log_data.get("error", {})
            msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
            return BackendResult(
                name=f"inspect::{task}",
                passed=False,
                summary=f"Task '{task}' errored: {msg}",
                details=msg,
            )
        score = _extract_score(log_data.get("results"))
        passed = score is not None and score >= score_threshold
        summary = (
            f"score={score:.2f} (threshold={score_threshold})" if score is not None else "no score"
        )
        if not passed and score is not None:
            summary = f"Score {score:.2f} below threshold {score_threshold}"
        return BackendResult(
            name=f"inspect::{task}",
            passed=passed,
            score=score,
            summary=summary,
        )
