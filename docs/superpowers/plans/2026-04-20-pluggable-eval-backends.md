# Pluggable Eval Backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Agent Lemon run standardized behavioral tests from external frameworks (inspect_ai first, promptfoo later) via a pluggable backend protocol, merging results into the unified report.

**Architecture:** Add an `EvalBackend` protocol and `BackendResult` model in a new `backends.py` module. The `InspectBackend` shells out to `inspect eval`, parses JSON logs, and returns `BackendResult` objects. A `run_backends()` function in the runner converts these to `EvalResult` and merges them with YAML eval results. Config-driven via `evals.backends` in `agent-lemon.yaml`.

**Tech Stack:** Python 3.13, Pydantic, subprocess (for `inspect` CLI calls), pytest + unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agent_lemon_lime/config.py` | Modify | Add `BackendConfig` model, add `backends` field to `EvalsConfig` |
| `src/agent_lemon_lime/evals/standard.py` | Modify | Add `BEHAVIORAL` to `EvalDomain` |
| `src/agent_lemon_lime/evals/backends.py` | Create | `EvalBackend` protocol, `BackendResult` model, `InspectBackend` class, `run_backends()` |
| `src/agent_lemon_lime/agents/lemon.py` | Modify | Call `run_backends()` and merge results |
| `tests/test_config.py` | Modify | Tests for `BackendConfig` parsing |
| `tests/test_backends.py` | Create | Tests for `InspectBackend` and `run_backends()` |

---

### Task 1: Add `BackendConfig` to the config model and `BEHAVIORAL` domain

**Files:**
- Modify: `src/agent_lemon_lime/config.py:35-42`
- Modify: `src/agent_lemon_lime/evals/standard.py:11-15`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test for BackendConfig defaults**

In `tests/test_config.py`, add:

```python
def test_config_backends_default_empty():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.evals.backends == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_config.py::test_config_backends_default_empty -v`
Expected: FAIL — `AttributeError: 'EvalsConfig' object has no attribute 'backends'`

- [ ] **Step 3: Write failing test for BackendConfig from YAML**

In `tests/test_config.py`, add:

```python
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
```

- [ ] **Step 4: Write failing test for BackendConfig defaults**

In `tests/test_config.py`, add:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_config.py -k "backend" -v`
Expected: FAIL

- [ ] **Step 6: Implement BackendConfig and add BEHAVIORAL domain**

In `src/agent_lemon_lime/config.py`, add after `SkillSource`:

```python
class BackendConfig(BaseModel):
    type: Literal["inspect"]
    model: str
    tasks: list[str]
    score_threshold: float = 1.0
```

Modify `EvalsConfig` to add:

```python
class EvalsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    skills: list[SkillSource] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)
```

In `src/agent_lemon_lime/evals/standard.py`, add to `EvalDomain`:

```python
class EvalDomain(StrEnum):
    SAFETY = "safety"
    STABILITY = "stability"
    CORRECTNESS = "correctness"
    SECURITY = "security"
    BEHAVIORAL = "behavioral"
```

- [ ] **Step 7: Run all config tests to verify they pass**

Run: `uv run python -m pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 8: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/config.py src/agent_lemon_lime/evals/standard.py tests/test_config.py`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add src/agent_lemon_lime/config.py src/agent_lemon_lime/evals/standard.py tests/test_config.py
git commit -m "feat: add BackendConfig and BEHAVIORAL eval domain"
```

---

### Task 2: Create `BackendResult` model and `EvalBackend` protocol

**Files:**
- Create: `src/agent_lemon_lime/evals/backends.py`
- Create: `tests/test_backends.py`

- [ ] **Step 1: Write failing test for BackendResult model**

Create `tests/test_backends.py`:

```python
"""Tests for pluggable eval backends."""

from agent_lemon_lime.evals.backends import BackendResult
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_backends.py::test_backend_result_defaults tests/test_backends.py::test_backend_result_with_score -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_lemon_lime.evals.backends'`

- [ ] **Step 3: Implement BackendResult and EvalBackend protocol**

Create `src/agent_lemon_lime/evals/backends.py`:

```python
"""Pluggable eval backends: protocol, models, and inspect_ai implementation."""

from __future__ import annotations

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

    def run(
        self, tasks: list[str], model: str, **kwargs: object
    ) -> list[BackendResult]: ...

    def available(self) -> bool: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_backends.py::test_backend_result_defaults tests/test_backends.py::test_backend_result_with_score -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/evals/backends.py tests/test_backends.py`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/agent_lemon_lime/evals/backends.py tests/test_backends.py
git commit -m "feat: add BackendResult model and EvalBackend protocol"
```

---

### Task 3: Implement `InspectBackend`

**Files:**
- Modify: `src/agent_lemon_lime/evals/backends.py`
- Modify: `tests/test_backends.py`

- [ ] **Step 1: Write failing test — inspect not installed**

Append to `tests/test_backends.py`:

```python
from unittest.mock import patch

from agent_lemon_lime.evals.backends import InspectBackend


def test_inspect_backend_not_available():
    backend = InspectBackend()
    with patch("shutil.which", return_value=None):
        assert backend.available() is False


def test_inspect_backend_available():
    backend = InspectBackend()
    with patch("shutil.which", return_value="/usr/bin/inspect"):
        assert backend.available() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_backends.py::test_inspect_backend_not_available -v`
Expected: FAIL — `ImportError: cannot import name 'InspectBackend'`

- [ ] **Step 3: Write failing test — task passes**

Append to `tests/test_backends.py`:

```python
import json
import subprocess
from unittest.mock import patch, MagicMock


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
        patch("agent_lemon_lime.evals.backends._find_log_file", return_value=log_file),
        patch("shutil.rmtree"),
    ):
        results = backend.run(tasks=["arc"], model="anthropic/claude-opus-4-6", score_threshold=0.8)

    assert len(results) == 1
    assert results[0].name == "inspect::arc"
    assert results[0].passed is True
    assert results[0].score == 0.92
```

- [ ] **Step 4: Write failing test — task below threshold**

Append to `tests/test_backends.py`:

```python
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
        patch("agent_lemon_lime.evals.backends._find_log_file", return_value=log_file),
        patch("shutil.rmtree"),
    ):
        results = backend.run(tasks=["arc"], model="openai/gpt-4o", score_threshold=0.8)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].score == 0.65
    assert "0.65" in results[0].summary
```

- [ ] **Step 5: Write failing test — task errors**

Append to `tests/test_backends.py`:

```python
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
        patch("agent_lemon_lime.evals.backends._find_log_file", return_value=log_file),
        patch("shutil.rmtree"),
    ):
        results = backend.run(tasks=["arc"], model="openai/gpt-4o", score_threshold=1.0)

    assert len(results) == 1
    assert results[0].passed is False
    assert "429" in results[0].details
```

- [ ] **Step 6: Implement InspectBackend**

In `src/agent_lemon_lime/evals/backends.py`, add imports at the top:

```python
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
```

Add `_find_log_file` helper and `InspectBackend` class after the protocol:

```python
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

    def _run_task(
        self, task: str, model: str, score_threshold: float
    ) -> BackendResult:
        log_dir = tempfile.mkdtemp(prefix="agent-lemon-inspect-")
        try:
            cmd = [
                "inspect", "eval", task,
                "--model", model,
                "--log-dir", log_dir,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
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

    def _parse_log(
        self, task: str, log_data: dict, score_threshold: float
    ) -> BackendResult:
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
        summary = f"score={score:.2f} (threshold={score_threshold})" if score is not None else "no score"
        if not passed and score is not None:
            summary = f"Score {score:.2f} below threshold {score_threshold}"
        return BackendResult(
            name=f"inspect::{task}",
            passed=passed,
            score=score,
            summary=summary,
        )
```

- [ ] **Step 7: Run all backend tests**

Run: `uv run python -m pytest tests/test_backends.py -v`
Expected: All PASS

- [ ] **Step 8: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/evals/backends.py tests/test_backends.py`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add src/agent_lemon_lime/evals/backends.py tests/test_backends.py
git commit -m "feat: implement InspectBackend with subprocess execution"
```

---

### Task 4: Add `run_backends()` and wire into LemonAgent

**Files:**
- Modify: `src/agent_lemon_lime/evals/backends.py`
- Modify: `src/agent_lemon_lime/agents/lemon.py`
- Modify: `tests/test_backends.py`

- [ ] **Step 1: Write failing test for run_backends()**

Append to `tests/test_backends.py`:

```python
from agent_lemon_lime.config import BackendConfig
from agent_lemon_lime.evals.backends import run_backends
from agent_lemon_lime.evals.runner import EvalResult


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
```

- [ ] **Step 2: Write failing test — backend not installed**

Append to `tests/test_backends.py`:

```python
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
```

- [ ] **Step 3: Write failing test — empty backends list**

Append to `tests/test_backends.py`:

```python
def test_run_backends_empty_config():
    results = run_backends([])
    assert results == []
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_backends.py -k "run_backends" -v`
Expected: FAIL — `ImportError: cannot import name 'run_backends'`

- [ ] **Step 5: Implement run_backends()**

In `src/agent_lemon_lime/evals/backends.py`, add import at the top:

```python
from agent_lemon_lime.config import BackendConfig
from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalOutput
```

Add `_get_backend` and `run_backends` after `InspectBackend`:

```python
_BACKEND_REGISTRY: dict[str, type] = {
    "inspect": InspectBackend,
}


def _get_backend(backend_type: str) -> EvalBackend | None:
    cls = _BACKEND_REGISTRY.get(backend_type)
    if cls is None:
        return None
    return cls()


def _backend_result_to_eval_result(br: BackendResult) -> EvalResult:
    return EvalResult(
        name=br.name,
        passed=br.passed,
        domain=br.domain,
        output=EvalOutput(
            exit_code=0 if br.passed else 1,
            stdout=br.summary,
            stderr=br.details,
            domain=br.domain,
        ),
        failures=[] if br.passed else [br.summary],
    )


def run_backends(
    configs: list[BackendConfig],
) -> list[EvalResult]:
    """Run all configured eval backends and return merged EvalResults."""
    results: list[EvalResult] = []
    for config in configs:
        backend = _get_backend(config.type)
        if backend is None:
            logger.warning("Unknown backend type: %s", config.type)
            continue
        if not backend.available():
            for task in config.tasks:
                results.append(EvalResult(
                    name=f"{backend.name}::{task}",
                    passed=False,
                    domain=EvalDomain.BEHAVIORAL,
                    output=EvalOutput(
                        exit_code=1,
                        stdout=f"Backend '{backend.name}' is not installed. "
                               f"Install it with: pip install inspect-ai",
                        stderr="",
                        domain=EvalDomain.BEHAVIORAL,
                    ),
                    failures=[f"Backend '{backend.name}' not installed"],
                ))
            continue
        backend_results = backend.run(
            tasks=config.tasks,
            model=config.model,
            score_threshold=config.score_threshold,
        )
        results.extend(
            _backend_result_to_eval_result(br) for br in backend_results
        )
    return results
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_backends.py -v`
Expected: All PASS

- [ ] **Step 7: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/evals/backends.py tests/test_backends.py`
Expected: Clean

- [ ] **Step 8: Commit**

```bash
git add src/agent_lemon_lime/evals/backends.py tests/test_backends.py
git commit -m "feat: add run_backends() with result conversion"
```

---

### Task 5: Wire backends into LemonAgent and CLI

**Files:**
- Modify: `src/agent_lemon_lime/agents/lemon.py`
- Modify: `src/agent_lemon_lime/cli/lemon.py`

- [ ] **Step 1: Modify LemonAgent.run_discovery to accept backend results**

In `src/agent_lemon_lime/agents/lemon.py`, modify `run_discovery`:

```python
    def run_discovery(
        self,
        *,
        eval_cases: list[EvalCase],
        on_result: Callable[[EvalResult], None] | None = None,
        backend_results: list[EvalResult] | None = None,
    ) -> LemonRunResult:
        """Run eval cases in discovery mode, building an observed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.
            on_result: Optional callback invoked after each case completes.
            backend_results: Results from external eval backends to merge.

        Returns:
            LemonRunResult with mode=DISCOVERY, observed SCP, report, and no violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox, on_result=on_result)
        if backend_results:
            results.extend(backend_results)
        observed_scp = self._build_observed_scp()
        report = self._synthesizer.build(
            results,
            scp=observed_scp,
            agent_name=self.config.name,
            inference=self._inference_config(),
        )
        return LemonRunResult(
            mode=RunMode.DISCOVER,
            scp=observed_scp,
            report=report,
            violations=[],
        )
```

- [ ] **Step 2: Modify LemonAgent.run_assert the same way**

In `src/agent_lemon_lime/agents/lemon.py`, modify `run_assert`:

```python
    def run_assert(
        self,
        *,
        eval_cases: list[EvalCase],
        assert_scp: SystemCapabilityProfile,
        _observed_scp: SystemCapabilityProfile | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
        backend_results: list[EvalResult] | None = None,
    ) -> LemonRunResult:
        """Run eval cases in assert mode, checking compliance against allowed SCP.

        Args:
            eval_cases: Evaluation cases to execute against the sandbox.
            assert_scp: The reference profile defining permitted capabilities.
            _observed_scp: Override the observed SCP (for testing injection only).
            on_result: Optional callback invoked after each case completes.
            backend_results: Results from external eval backends to merge.

        Returns:
            LemonRunResult with mode=ASSERT, observed SCP, report, and any violations.
        """
        results = self._runner.run(eval_cases, sandbox=self.sandbox, on_result=on_result)
        if backend_results:
            results.extend(backend_results)
        observed = (
            _observed_scp if _observed_scp is not None else self._build_observed_scp()
        )
        violations = observed.assert_subset_of(assert_scp)
        report = self._synthesizer.build(
            results,
            scp=observed,
            agent_name=self.config.name,
            violations=violations,
            inference=self._inference_config(),
        )
        return LemonRunResult(
            mode=RunMode.ASSERT,
            scp=observed,
            report=report,
            violations=violations,
        )
```

- [ ] **Step 3: Wire into discover CLI command**

In `src/agent_lemon_lime/cli/lemon.py`, in the `discover` function, add after the sandbox eval run and before report synthesis. Find the line:

```python
    result = agent.run_discovery(eval_cases=cases, on_result=on_result)
```

Replace with:

```python
    from agent_lemon_lime.evals.backends import run_backends

    backend_results = run_backends(config.evals.backends)
    for br in backend_results:
        on_result(br) if on_result else None

    result = agent.run_discovery(
        eval_cases=cases, on_result=on_result, backend_results=backend_results,
    )
```

Also update the total count for the progress printer. Find:

```python
    n = len(cases)
```

Replace with:

```python
    backend_task_count = sum(len(b.tasks) for b in config.evals.backends)
    n = len(cases) + backend_task_count
```

- [ ] **Step 4: Wire into assert CLI command**

In `src/agent_lemon_lime/cli/lemon.py`, in the `assert_cmd` function, find:

```python
    result = agent.run_assert(eval_cases=cases, assert_scp=assert_scp, on_result=on_result)
```

Replace with:

```python
    from agent_lemon_lime.evals.backends import run_backends

    backend_results = run_backends(config.evals.backends)
    for br in backend_results:
        on_result(br) if on_result else None

    result = agent.run_assert(
        eval_cases=cases, assert_scp=assert_scp,
        on_result=on_result, backend_results=backend_results,
    )
```

Also update the total count. Find:

```python
    n = len(cases)
```

Replace with:

```python
    backend_task_count = sum(len(b.tasks) for b in config.evals.backends)
    n = len(cases) + backend_task_count
```

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `uv run python -m pytest tests/ -v -k "not integration"`
Expected: All PASS (existing tests don't configure backends, so `run_backends([])` returns `[]`)

- [ ] **Step 6: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/agents/lemon.py src/agent_lemon_lime/cli/lemon.py`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/agent_lemon_lime/agents/lemon.py src/agent_lemon_lime/cli/lemon.py
git commit -m "feat: wire eval backends into discover and assert commands"
```

---

### Task 6: Update README with backend documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add backends section to README**

In `README.md`, after the "Evaluator types" section and before the "Sandboxes" section, add:

```markdown
### Eval backends

Agent Lemon can run standardized behavioral tests from external eval frameworks. Results merge into the unified report alongside YAML eval cases.

Configure backends in `agent-lemon.yaml`:

\```yaml
evals:
  directories:
    - evals/
  backends:
    - type: inspect
      model: anthropic/claude-opus-4-6
      tasks:
        - arc
        - hellaswag
        - safety/refusal
      score_threshold: 0.8
\```

Backends run on the host after sandbox evals complete. Each task produces one line in the report:

\```
── behavioral ───────────────────────────────────
inspect::arc                           PASSED  (score: 0.92)
inspect::safety/refusal                FAILED  (score: 0.65)
\```

#### Supported backends

| Backend | Type | Install |
|---------|------|---------|
| [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | `inspect` | `pip install inspect-ai` |

Backends are optional — if not installed, Agent Lemon reports them as failed with an install instruction. They do not affect the SCP (they test model behavior, not system access).
```

- [ ] **Step 2: Update the configuration reference**

In the configuration reference section of `README.md`, add `backends` to the `evals` block:

```yaml
evals:
  directories:
    - evals/
  skills:
    - path: skills/
  backends:                       # external eval frameworks
    - type: inspect               # backend type
      model: anthropic/claude-opus-4-6  # model in backend's format
      tasks: [arc, hellaswag]     # task identifiers
      score_threshold: 0.8        # pass threshold (default: 1.0)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add eval backends section to README"
```
