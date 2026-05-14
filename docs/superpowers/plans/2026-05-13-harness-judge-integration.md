# Harness Judge Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace agent-lemon-lime's 4 custom `Evaluator` classes with agent-eval-harness's judge system (inline checks, LLM judges, external code judges, regression detection) while keeping lemon-lime's sandbox-based `EvalRunner`.

**Architecture:** A new `evals/scoring.py` bridge module imports harness scoring functions via `importlib.util.spec_from_file_location` and converts lemon-lime's `EvalOutput` into the record dict harness judges expect. `EvalCase` replaces `evaluators: list[Evaluator]` with `judges: list[JudgeConfig]` (from `agent_eval.config`). `EvalResult` gains a `scores: dict[str, JudgeScore]` field. The loader auto-converts legacy `judge_hint` and `expected_output` fields into `JudgeConfig` objects for backward compatibility.

**Tech Stack:** Python 3.13, agent-eval-harness (`agent_eval` package), Pydantic, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-05-13-harness-judge-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Add agent-eval-harness dependency |
| `src/agent_lemon_lime/evals/scoring.py` | **NEW** — Bridge: import harness scoring, `JudgeScore` dataclass, `score_eval_output()`, `check_regressions()` |
| `src/agent_lemon_lime/evals/standard.py` | Keep `EvalDomain` and `EvalOutput`; delete `Evaluator`, all 4 evaluator classes, `JUDGE_SYSTEM_PROMPT` |
| `src/agent_lemon_lime/evals/runner.py` | `EvalCase.judges`, `EvalResult.scores`, runner uses `score_eval_output()` |
| `src/agent_lemon_lime/evals/loader.py` | Parse `judges:` from YAML, auto-convert `judge_hint`/`expected_output` to `JudgeConfig` |
| `src/agent_lemon_lime/config.py` | Add `judges`, `thresholds` to `EvalsConfig` |
| `src/agent_lemon_lime/probes/*.yaml` (5 files) | Migrate `judge_hint` to `judges:` section |
| `src/agent_lemon_lime/agents/lemon.py` | Pass `judge_model`, aggregate scores, run regression checks |
| `src/agent_lemon_lime/report/models.py` | Add `judge_summary`, `regressions` to `EvalReport` |
| `src/agent_lemon_lime/report/synthesizer.py` | Render judge score table and regressions in markdown |
| `tests/test_scoring.py` | **NEW** — Tests for bridge module |
| `tests/test_evals.py` | Update for new data model |
| `tests/test_judge.py` | Rewrite for harness judges |
| `tests/test_probe_loader.py` | Update for `judges` field |
| `tests/test_agents.py` | Update for `judges` field |
| `tests/test_report.py` | Update for `judge_summary`, `regressions` |
| `tests/test_config.py` | Add tests for `judges`/`thresholds` config |

---

### Task 1: Add agent-eval-harness dependency

**Files:**
- Modify: `pyproject.toml:6-18`

- [ ] **Step 1: Add dependency to pyproject.toml**

```python
# In pyproject.toml, add agent-eval-harness to the dependencies list.
# Use a path dependency pointing to the sibling submodule directory.
```

In `pyproject.toml`, add this line inside the `dependencies` list (after `"opentelemetry-sdk>=1.25.0",`):

```toml
    "agent-eval-harness",
```

Also add a `[tool.uv.sources]` section to tell uv where to find it during development:

```toml
[tool.uv.sources]
agent-eval-harness = { path = "../agent-eval-harness" }
```

- [ ] **Step 2: Verify dependency resolves**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv pip install -e .`
Expected: Installs successfully, `agent_eval` is importable.

- [ ] **Step 3: Verify import works**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -c "from agent_eval.config import JudgeConfig; print(JudgeConfig())"`
Expected: Prints a JudgeConfig with defaults (empty strings, empty lists).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add agent-eval-harness as dependency for judge system"
```

---

### Task 2: Create the scoring bridge module

**Files:**
- Create: `src/agent_lemon_lime/evals/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing test for JudgeScore dataclass**

Create `tests/test_scoring.py`:

```python
"""Tests for the scoring bridge module."""

from agent_lemon_lime.evals.scoring import JudgeScore


def test_judge_score_defaults():
    s = JudgeScore(value=True)
    assert s.value is True
    assert s.rationale == ""
    assert s.error is None


def test_judge_score_with_rationale():
    s = JudgeScore(value=4, rationale="Good output quality")
    assert s.value == 4
    assert s.rationale == "Good output quality"


def test_judge_score_with_error():
    s = JudgeScore(value=None, error="Judge failed to parse")
    assert s.value is None
    assert s.error == "Judge failed to parse"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_lemon_lime.evals.scoring'`

- [ ] **Step 3: Write JudgeScore and harness import**

Create `src/agent_lemon_lime/evals/scoring.py`:

```python
"""Bridge between lemon-lime's sandbox output and agent-eval-harness judges."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval.config import EvalConfig, JudgeConfig

from agent_lemon_lime.evals.standard import EvalOutput

logger = logging.getLogger(__name__)

_harness_score_mod = None


@dataclass
class JudgeScore:
    value: bool | int | float | str | None
    rationale: str = ""
    error: str | None = None


def _find_harness_score_module() -> Path | None:
    """Locate score.py in agent-eval-harness installation."""
    candidates = [
        Path(__file__).resolve().parents[4]
        / "agent-eval-harness"
        / "skills"
        / "eval-run"
        / "scripts"
        / "score.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_harness_scoring():
    """Lazy-load the harness score module."""
    global _harness_score_mod
    if _harness_score_mod is not None:
        return _harness_score_mod

    path = _find_harness_score_module()
    if path is None:
        raise ImportError(
            "agent-eval-harness score.py not found. "
            "Ensure agent-eval-harness is installed as a sibling directory."
        )
    spec = importlib.util.spec_from_file_location("_harness_score", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _harness_score_mod = mod
    return mod


def _eval_output_to_record(output: EvalOutput) -> dict[str, Any]:
    """Convert an EvalOutput to the record dict harness judges expect."""
    return {
        "exit_code": output.exit_code,
        "stdout": output.stdout,
        "stderr": output.stderr,
        "files": {},
        "tool_calls": [],
        "annotations": {},
    }


def _build_eval_config_for_judges(
    judges: list[JudgeConfig],
    *,
    model: str | None = None,
) -> EvalConfig:
    """Build a minimal EvalConfig to pass to harness load_judges()."""
    from agent_eval.config import ModelsConfig

    config = EvalConfig()
    config.judges = list(judges)
    if model:
        config.models = ModelsConfig(judge=model)
    return config


def score_eval_output(
    output: EvalOutput,
    judges: list[JudgeConfig],
    *,
    model: str | None = None,
) -> dict[str, JudgeScore]:
    """Score an EvalOutput using harness judges."""
    if not judges:
        return {}

    harness = _get_harness_scoring()
    config = _build_eval_config_for_judges(judges, model=model)
    loaded = harness.load_judges(config)
    record = _eval_output_to_record(output)
    scores: dict[str, JudgeScore] = {}

    for name, scorer, condition in loaded:
        if condition:
            try:
                if not eval(
                    condition,
                    {"__builtins__": {}},
                    {"annotations": record.get("annotations", {}), "outputs": record},
                ):
                    scores[name] = JudgeScore(
                        value=None,
                        rationale=f"Skipped: condition '{condition}' is false",
                    )
                    continue
            except Exception as e:
                scores[name] = JudgeScore(value=None, error=f"Condition error: {e}")
                continue
        try:
            result = scorer(outputs=record)
            if isinstance(result, tuple) and len(result) == 2:
                scores[name] = JudgeScore(value=result[0], rationale=str(result[1]))
            elif hasattr(result, "value"):
                scores[name] = JudgeScore(
                    value=result.value,
                    rationale=getattr(result, "rationale", ""),
                )
            elif isinstance(result, (bool, int, float, str)):
                scores[name] = JudgeScore(value=result)
            else:
                scores[name] = JudgeScore(value=result)
        except Exception as e:
            scores[name] = JudgeScore(value=None, error=str(e))

    return scores


def check_regressions(
    aggregated: dict[str, dict],
    thresholds: dict[str, dict],
    baseline: dict[str, dict] | None = None,
) -> list[str]:
    """Check for regressions using harness threshold detection."""
    harness = _get_harness_scoring()
    regressions = harness.detect_regressions(aggregated, thresholds, baseline)
    return [
        f"[{r.judge_name}] {r.metric}: expected {r.baseline_value}, got {r.current_value}"
        for r in regressions
    ]
```

- [ ] **Step 4: Run test to verify JudgeScore passes**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_judge_score_defaults tests/test_scoring.py::test_judge_score_with_rationale tests/test_scoring.py::test_judge_score_with_error -v`
Expected: 3 PASS

- [ ] **Step 5: Write test for _eval_output_to_record**

Add to `tests/test_scoring.py`:

```python
from agent_lemon_lime.evals.scoring import _eval_output_to_record
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput


def test_eval_output_to_record_converts_fields():
    output = EvalOutput(
        exit_code=1, stdout="hello", stderr="warn", domain=EvalDomain.SAFETY
    )
    record = _eval_output_to_record(output)
    assert record["exit_code"] == 1
    assert record["stdout"] == "hello"
    assert record["stderr"] == "warn"
    assert record["files"] == {}
    assert record["tool_calls"] == []
    assert record["annotations"] == {}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_eval_output_to_record_converts_fields -v`
Expected: PASS

- [ ] **Step 7: Write test for score_eval_output with inline check**

Add to `tests/test_scoring.py`:

```python
from agent_eval.config import JudgeConfig
from agent_lemon_lime.evals.scoring import score_eval_output


def test_score_eval_output_inline_check_pass():
    output = EvalOutput(
        exit_code=0, stdout="ok", stderr="", domain=EvalDomain.CORRECTNESS
    )
    judges = [
        JudgeConfig(
            name="exit-code",
            check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
        ),
    ]
    scores = score_eval_output(output, judges)
    assert "exit-code" in scores
    assert scores["exit-code"].value is True


def test_score_eval_output_inline_check_fail():
    output = EvalOutput(
        exit_code=1, stdout="", stderr="error", domain=EvalDomain.CORRECTNESS
    )
    judges = [
        JudgeConfig(
            name="exit-code",
            check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
        ),
    ]
    scores = score_eval_output(output, judges)
    assert scores["exit-code"].value is False
    assert "non-zero exit" in scores["exit-code"].rationale


def test_score_eval_output_empty_judges():
    output = EvalOutput(
        exit_code=0, stdout="ok", stderr="", domain=EvalDomain.CORRECTNESS
    )
    scores = score_eval_output(output, [])
    assert scores == {}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py -v`
Expected: All 6 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent_lemon_lime/evals/scoring.py tests/test_scoring.py
git commit -m "feat: add scoring bridge module for harness judge integration"
```

---

### Task 3: Strip evaluator classes from standard.py

**Files:**
- Modify: `src/agent_lemon_lime/evals/standard.py:1-113`

- [ ] **Step 1: Write test that new standard.py still exports EvalDomain and EvalOutput**

Add to `tests/test_scoring.py`:

```python
def test_standard_still_exports_eval_domain():
    from agent_lemon_lime.evals.standard import EvalDomain

    assert EvalDomain.SAFETY == "safety"
    assert EvalDomain.CORRECTNESS == "correctness"


def test_standard_still_exports_eval_output():
    from agent_lemon_lime.evals.standard import EvalOutput

    out = EvalOutput(
        exit_code=0, stdout="ok", stderr="", domain=EvalDomain.CORRECTNESS
    )
    assert out.exit_code == 0
```

- [ ] **Step 2: Run test to verify it passes (before deletion)**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_standard_still_exports_eval_domain tests/test_scoring.py::test_standard_still_exports_eval_output -v`
Expected: PASS

- [ ] **Step 3: Delete evaluator classes from standard.py**

Replace the entire contents of `src/agent_lemon_lime/evals/standard.py` with:

```python
"""Standard eval types: EvalDomain and EvalOutput."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class EvalDomain(StrEnum):
    SAFETY = "safety"
    STABILITY = "stability"
    CORRECTNESS = "correctness"
    SECURITY = "security"
    BEHAVIORAL = "behavioral"


class EvalOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    domain: EvalDomain
    metadata: dict[str, object] = {}
```

- [ ] **Step 4: Run test to verify EvalDomain and EvalOutput still work**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_standard_still_exports_eval_domain tests/test_scoring.py::test_standard_still_exports_eval_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_lemon_lime/evals/standard.py tests/test_scoring.py
git commit -m "refactor: remove Evaluator protocol and 4 evaluator classes from standard.py"
```

---

### Task 4: Update EvalCase and EvalResult data models

**Files:**
- Modify: `src/agent_lemon_lime/evals/runner.py:1-78`

- [ ] **Step 1: Write tests for new data model shapes**

Add to `tests/test_scoring.py`:

```python
from agent_lemon_lime.evals.runner import EvalCase, EvalInput, EvalResult


def test_eval_case_has_judges_field():
    case = EvalCase(
        name="test",
        input=EvalInput(command=["echo", "hi"]),
        judges=[JudgeConfig(name="exit-code", check='return True, "ok"')],
    )
    assert len(case.judges) == 1
    assert case.judges[0].name == "exit-code"


def test_eval_result_has_scores_field():
    result = EvalResult(
        name="test",
        passed=True,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0, stdout="", stderr="", domain=EvalDomain.CORRECTNESS
        ),
        scores={"exit-code": JudgeScore(value=True)},
    )
    assert "exit-code" in result.scores
    assert result.scores["exit-code"].value is True


def test_eval_result_scores_default_empty():
    result = EvalResult(
        name="test",
        passed=True,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0, stdout="", stderr="", domain=EvalDomain.CORRECTNESS
        ),
    )
    assert result.scores == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_eval_case_has_judges_field tests/test_scoring.py::test_eval_result_has_scores_field tests/test_scoring.py::test_eval_result_scores_default_empty -v`
Expected: FAIL — `EvalCase` still expects `evaluators`, not `judges`.

- [ ] **Step 3: Update runner.py with new data model**

Replace the entire contents of `src/agent_lemon_lime/evals/runner.py` with:

```python
"""EvalRunner: runs EvalCase instances against a sandbox."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_eval.config import JudgeConfig
from pydantic import BaseModel

from agent_lemon_lime.evals.scoring import JudgeScore, score_eval_output
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.harness.base import AbstractSandbox

logger = logging.getLogger(__name__)


class EvalInput(BaseModel):
    command: list[str]
    workdir: str | None = None
    env: dict[str, str] = {}
    timeout_seconds: int | None = None


@dataclass
class EvalCase:
    name: str
    input: EvalInput
    judges: list[JudgeConfig]
    domain: EvalDomain = EvalDomain.CORRECTNESS
    description: str = ""


@dataclass
class EvalResult:
    name: str
    passed: bool
    domain: EvalDomain
    output: EvalOutput
    failures: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    scores: dict[str, JudgeScore] = field(default_factory=dict)


class EvalRunner:
    """Executes eval cases against a sandbox, returns results."""

    def run(
        self,
        cases: list[EvalCase],
        *,
        sandbox: AbstractSandbox,
        judge_model: str | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        with sandbox:
            for case in cases:
                raw = sandbox.exec(
                    case.input.command,
                    workdir=case.input.workdir,
                    env=case.input.env or None,
                    timeout_seconds=case.input.timeout_seconds,
                )
                output = EvalOutput(
                    exit_code=raw.exit_code,
                    stdout=raw.stdout,
                    stderr=raw.stderr,
                    domain=case.domain,
                )
                scores = score_eval_output(
                    output, case.judges, model=judge_model,
                )
                failures = [
                    f"{name}: {s.rationale or s.error or ''}"
                    for name, s in scores.items()
                    if s.value is False or s.value == 0 or s.error
                ]
                result = EvalResult(
                    name=case.name,
                    passed=len(failures) == 0,
                    domain=case.domain,
                    output=output,
                    failures=failures,
                    command=list(case.input.command),
                    scores=scores,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
        return results
```

- [ ] **Step 4: Run the new model tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_eval_case_has_judges_field tests/test_scoring.py::test_eval_result_has_scores_field tests/test_scoring.py::test_eval_result_scores_default_empty -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_lemon_lime/evals/runner.py tests/test_scoring.py
git commit -m "refactor: replace EvalCase.evaluators with judges, add EvalResult.scores"
```

---

### Task 5: Update the case loader

**Files:**
- Modify: `src/agent_lemon_lime/evals/loader.py:1-226`

- [ ] **Step 1: Write tests for new loader behavior**

Add to `tests/test_scoring.py`:

```python
from agent_lemon_lime.evals.loader import load_cases_from_dir


def test_loader_new_judges_format(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        'cases:\n'
        '  - name: check-exit\n'
        '    domain: correctness\n'
        '    input:\n'
        '      command: ["echo", "hello"]\n'
        '    judges:\n'
        '      - name: exit-code\n'
        '        check: |\n'
        '          return outputs.get("exit_code", 1) == 0, "non-zero exit"\n'
    )
    cases = load_cases_from_dir(tmp_path)
    assert len(cases) == 1
    assert len(cases[0].judges) == 1
    assert cases[0].judges[0].name == "exit-code"
    assert "exit_code" in cases[0].judges[0].check


def test_loader_legacy_expected_output_becomes_inline_check(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: greeting\n"
        "    input:\n"
        "      command: ['echo', 'hello']\n"
        "    expected_output: hello\n"
    )
    cases = load_cases_from_dir(tmp_path)
    assert len(cases) == 1
    assert any(j.name == "output-contains" for j in cases[0].judges)
    contains_judge = next(j for j in cases[0].judges if j.name == "output-contains")
    assert "hello" in contains_judge.check


def test_loader_legacy_judge_hint_becomes_llm_judge(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: probe-test\n"
        "    domain: safety\n"
        "    input:\n"
        "      prompt: 'Do something bad'\n"
        "    judge_hint: 'PASS: Refuses FAIL: Does it'\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert any(j.name == "behavioral" for j in cases[0].judges)
    behavioral = next(j for j in cases[0].judges if j.name == "behavioral")
    assert "PASS" in behavioral.prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_loader_new_judges_format tests/test_scoring.py::test_loader_legacy_expected_output_becomes_inline_check tests/test_scoring.py::test_loader_legacy_judge_hint_becomes_llm_judge -v`
Expected: FAIL — loader still uses old Evaluator-based parsing.

- [ ] **Step 3: Rewrite the loader**

Replace the entire contents of `src/agent_lemon_lime/evals/loader.py` with:

```python
"""Load EvalCase instances from YAML case-definition files."""

from __future__ import annotations

import importlib.resources
import logging
import pathlib
import shlex
import warnings
from typing import TYPE_CHECKING, Any

import yaml
from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import EvalDomain

if TYPE_CHECKING:
    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.harness.base import AbstractSandbox

from agent_lemon_lime.config import resolve_env

logger = logging.getLogger(__name__)

DEFAULT_EXIT_CHECK = JudgeConfig(
    name="exit-code",
    check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
)


def load_cases_from_dir(
    directory: pathlib.Path | str,
    *,
    base_dir: pathlib.Path | None = None,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    """Return EvalCase objects from all *.yaml files in directory."""
    d = pathlib.Path(directory)
    if base_dir is not None:
        d = base_dir / d
    if not d.exists():
        return []
    cases: list[EvalCase] = []
    for yaml_file in sorted(d.glob("*.yaml")):
        cases.extend(
            _parse_case_file(yaml_file, run_command=run_command, run_env=run_env)
        )
    return cases


def load_cases_from_config(
    config: LemonConfig, *, project_dir: pathlib.Path
) -> list[EvalCase]:
    """Load all cases from directories listed in config.evals.directories."""
    run_command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    cases: list[EvalCase] = []
    for directory in config.evals.directories:
        cases.extend(
            load_cases_from_dir(
                directory,
                base_dir=project_dir,
                run_command=run_command,
                run_env=resolved,
            )
        )
    return cases


def load_cases_from_sandbox(
    config: LemonConfig,
    *,
    sandbox: AbstractSandbox,
) -> list[EvalCase]:
    """Load eval cases by reading YAML files from inside the sandbox."""
    run_command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    cases: list[EvalCase] = []
    for directory in config.evals.directories:
        listing = sandbox.exec(
            ["find", directory, "-name", "*.yaml", "-type", "f"],
        )
        if listing.exit_code != 0:
            logger.warning(
                "Failed to list eval dir %s in sandbox: %s",
                directory,
                listing.stderr.strip(),
            )
            continue
        for yaml_path in sorted(listing.stdout.strip().splitlines()):
            if not yaml_path:
                continue
            result = sandbox.exec(["cat", yaml_path])
            if result.exit_code != 0:
                logger.warning(
                    "Failed to read %s from sandbox: %s",
                    yaml_path,
                    result.stderr.strip(),
                )
                continue
            cases.extend(
                _parse_case_content(
                    result.stdout, run_command=run_command, run_env=run_env,
                )
            )
    return cases


def default_case_from_config(config: LemonConfig) -> EvalCase:
    """Build a single smoke-test EvalCase from config.run.command."""
    command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    return EvalCase(
        name=f"{config.name}-runs",
        input=EvalInput(
            command=command, timeout_seconds=config.run.timeout_seconds, env=resolved,
        ),
        judges=[DEFAULT_EXIT_CHECK],
        domain=EvalDomain.CORRECTNESS,
        description=f"Smoke test: {config.run.command} exits 0",
    )


def load_builtin_probes(
    *,
    run_command: list[str],
    run_env: dict[str, str] | None = None,
    model: str | None = None,
    scp_yaml: str = "",
    config_yaml: str = "",
) -> list[EvalCase]:
    """Load built-in probe cases from the agent_lemon_lime.probes package."""
    probes_pkg = importlib.resources.files("agent_lemon_lime.probes")
    cases: list[EvalCase] = []
    for resource in sorted(probes_pkg.iterdir(), key=lambda r: r.name):
        if not resource.name.endswith(".yaml"):
            continue
        text = resource.read_text(encoding="utf-8")
        parsed = _parse_case_content(text, run_command=run_command, run_env=run_env)
        cases.extend(parsed)
    return cases


def _parse_case_file(
    path: pathlib.Path,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    return _parse_case_content(
        path.read_text(),
        run_command=run_command,
        run_env=run_env,
    )


def _parse_case_content(
    text: str,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []
    return [
        c
        for raw in data.get("cases", [])
        if (c := _parse_case(raw, run_command=run_command, run_env=run_env))
        is not None
    ]


def _parse_case(
    raw: dict[str, Any],
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> EvalCase | None:
    name = raw.get("name", "unnamed")
    description = raw.get("description", "")
    domain_str = raw.get("domain", "correctness")
    try:
        domain = EvalDomain(domain_str)
    except ValueError:
        logger.warning(
            "Unknown domain '%s' for case '%s' — using CORRECTNESS",
            domain_str,
            name,
        )
        domain = EvalDomain.CORRECTNESS

    inp = raw.get("input", {})
    command = inp.get("command")
    if command is None:
        prompt = inp.get("prompt")
        if prompt and run_command is not None:
            sanitized = " ".join(prompt.split())
            command = run_command + ["--prompt", sanitized]
        else:
            warnings.warn(
                f"Skipping case '{name}': no command and no run_command to derive one from.",
                stacklevel=3,
            )
            return None

    judges: list[JudgeConfig] = []

    # New format: explicit judges list
    raw_judges = raw.get("judges", [])
    for rj in raw_judges:
        judges.append(
            JudgeConfig(
                name=rj.get("name", ""),
                description=rj.get("description", ""),
                condition=rj.get("if", ""),
                check=rj.get("check", ""),
                prompt=rj.get("prompt", ""),
                prompt_file=rj.get("prompt_file", ""),
                context=rj.get("context", []),
                feedback_type=rj.get("feedback_type", ""),
                model=rj.get("model", ""),
                module=rj.get("module", ""),
                function=rj.get("function", ""),
            )
        )

    # Legacy: default exit-code check if no judges specified
    if not raw_judges:
        judges.append(DEFAULT_EXIT_CHECK)

    # Legacy: expected_output → inline check
    expected = raw.get("expected_output")
    if expected:
        escaped = str(expected).replace('"', '\\"')
        judges.append(
            JudgeConfig(
                name="output-contains",
                check=(
                    f'return "{escaped}" in outputs.get("stdout", ""), '
                    f'"expected \\"{escaped}\\" not found in stdout"'
                ),
            )
        )

    # Legacy: judge_hint → LLM judge prompt
    judge_hint = raw.get("judge_hint", "")
    if judge_hint:
        judges.append(
            JudgeConfig(name="behavioral", prompt=judge_hint)
        )

    env = dict(run_env) if run_env else {}
    return EvalCase(
        name=name,
        input=EvalInput(command=list(command), env=env),
        judges=judges,
        domain=domain,
        description=description,
    )
```

- [ ] **Step 4: Run loader tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_scoring.py::test_loader_new_judges_format tests/test_scoring.py::test_loader_legacy_expected_output_becomes_inline_check tests/test_scoring.py::test_loader_legacy_judge_hint_becomes_llm_judge -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_lemon_lime/evals/loader.py tests/test_scoring.py
git commit -m "refactor: update case loader for JudgeConfig, with legacy auto-conversion"
```

---

### Task 6: Add judges and thresholds to LemonConfig

**Files:**
- Modify: `src/agent_lemon_lime/config.py:38-49`

- [ ] **Step 1: Write test for new config fields**

Add to `tests/test_config.py`:

```python
def test_config_judges_default_empty():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.evals.judges == []


def test_config_thresholds_default_empty():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.evals.thresholds == {}


def test_config_judges_from_yaml():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        evals:
          judges:
            - name: exit-code
              check: 'return outputs.get("exit_code") == 0, "bad exit"'
            - name: quality
              prompt: "Rate the output quality 1-5"
              model: claude-sonnet-4-6
          thresholds:
            exit-code:
              min_pass_rate: 1.0
            quality:
              min_mean: 3.5
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert len(config.evals.judges) == 2
    assert config.evals.judges[0]["name"] == "exit-code"
    assert config.evals.judges[1]["model"] == "claude-sonnet-4-6"
    assert config.evals.thresholds["exit-code"]["min_pass_rate"] == 1.0
    assert config.evals.thresholds["quality"]["min_mean"] == 3.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_config.py::test_config_judges_default_empty tests/test_config.py::test_config_thresholds_default_empty tests/test_config.py::test_config_judges_from_yaml -v`
Expected: FAIL — `EvalsConfig` doesn't have `judges` or `thresholds` fields.

- [ ] **Step 3: Add fields to EvalsConfig**

In `src/agent_lemon_lime/config.py`, modify the `EvalsConfig` class (around line 45) to add two new fields:

```python
class EvalsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    skills: list[SkillSource] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)
    judges: list[dict[str, object]] = Field(default_factory=list)
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_config.py::test_config_judges_default_empty tests/test_config.py::test_config_thresholds_default_empty tests/test_config.py::test_config_judges_from_yaml -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_lemon_lime/config.py tests/test_config.py
git commit -m "feat: add judges and thresholds fields to EvalsConfig"
```

---

### Task 7: Update existing tests for new data model

**Files:**
- Modify: `tests/test_evals.py:1-250`
- Modify: `tests/test_probe_loader.py:1-82`
- Modify: `tests/test_judge.py:1-150`
- Modify: `tests/test_agents.py:1-139`
- Modify: `tests/test_report.py:1-75`

- [ ] **Step 1: Rewrite test_evals.py**

Replace the entire contents of `tests/test_evals.py` with:

```python
"""Tests for EvalRunner and case loading."""

import warnings

import pytest
from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.loader import load_cases_from_dir
from agent_lemon_lime.evals.runner import EvalCase, EvalInput, EvalRunner
from agent_lemon_lime.evals.skills import SkillLoader
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.harness.mock import MockSandbox


EXIT_CHECK = JudgeConfig(
    name="exit-code",
    check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
)

CONTAINS_HELLO = JudgeConfig(
    name="output-contains",
    check='return "hello" in outputs.get("stdout", ""), "missing hello"',
)


def test_eval_runner_passes():
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    runner = EvalRunner()
    cases = [
        EvalCase(
            name="echo-passes",
            input=EvalInput(command=["echo", "hello"]),
            judges=[EXIT_CHECK, CONTAINS_HELLO],
        )
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].name == "echo-passes"
    assert results[0].failures == []
    assert "exit-code" in results[0].scores
    assert results[0].scores["exit-code"].value is True


def test_eval_runner_fails():
    sandbox = MockSandbox()
    sandbox.register_command(["bad", "cmd"], stdout="", exit_code=1)
    runner = EvalRunner()
    cases = [
        EvalCase(
            name="fails",
            input=EvalInput(command=["bad", "cmd"]),
            judges=[EXIT_CHECK],
        )
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert results[0].passed is False
    assert any("exit-code" in f for f in results[0].failures)
    assert results[0].scores["exit-code"].value is False


def test_eval_runner_multiple_cases():
    sandbox = MockSandbox()
    sandbox.register_command(["ok"], stdout="ok\n", exit_code=0)
    sandbox.register_command(["fail"], stdout="", exit_code=1)
    runner = EvalRunner()
    cases = [
        EvalCase(
            name="pass", input=EvalInput(command=["ok"]), judges=[EXIT_CHECK],
        ),
        EvalCase(
            name="fail", input=EvalInput(command=["fail"]), judges=[EXIT_CHECK],
        ),
    ]
    results = runner.run(cases, sandbox=sandbox)
    assert results[0].passed is True
    assert results[1].passed is False


def test_skill_loader_loads_local_dir(tmp_path):
    (tmp_path / "my_skill.md").write_text("# My Skill\n\nDo something useful.")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "my_skill"
    assert "Do something useful" in skills[0].content


def test_skill_loader_ignores_non_markdown(tmp_path):
    (tmp_path / "ignored.txt").write_text("not a skill")
    (tmp_path / "skill.md").write_text("# Skill")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "skill"


def test_skill_loader_missing_dir_raises(tmp_path):
    loader = SkillLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_dir(tmp_path / "nonexistent")


def test_skill_loader_loads_nested(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("# Nested")
    loader = SkillLoader()
    skills = loader.load_from_dir(tmp_path)
    assert any(s.name == "nested" for s in skills)


def test_loader_converts_prompt_case_when_run_command_provided(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: greeting-responds\n"
        "    description: Says hello\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
        "    expected_output: Hello\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].name == "greeting-responds"
    assert cases[0].input.command == ["python", "agent.py", "--prompt", "Say hello."]


def test_loader_warns_and_skips_prompt_case_without_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: prompt-only\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = load_cases_from_dir(tmp_path)
    assert cases == []
    assert any(
        issubclass(w.category, UserWarning) and "prompt-only" in w.message.args[0]
        for w in caught
    )


def test_loader_command_case_unaffected_by_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: exits-cleanly\n"
        "    input:\n"
        "      command: ['python', 'agent.py']\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].input.command == ["python", "agent.py"]


def test_load_cases_from_sandbox():
    """load_cases_from_sandbox reads eval YAML from inside the sandbox."""
    import textwrap

    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.evals.loader import load_cases_from_sandbox

    config = LemonConfig.from_yaml(textwrap.dedent("""\
        name: test-agent
        run:
          command: python agent.py
        evals:
          directories:
            - evals/
    """))

    yaml_content = textwrap.dedent("""\
        cases:
          - name: sandbox-case
            input:
              command: ['echo', 'hello']
            expected_output: hello
    """)

    sandbox = MockSandbox()
    sandbox.register_command(
        ["find", "evals/", "-name", "*.yaml", "-type", "f"],
        stdout="evals/test.yaml\n",
    )
    sandbox.register_command(
        ["cat", "evals/test.yaml"],
        stdout=yaml_content,
    )

    with sandbox:
        cases = load_cases_from_sandbox(config, sandbox=sandbox)

    assert len(cases) == 1
    assert cases[0].name == "sandbox-case"
    assert cases[0].input.command == ["echo", "hello"]


def test_load_cases_from_sandbox_missing_dir():
    """load_cases_from_sandbox returns empty list when find fails."""
    import textwrap

    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.evals.loader import load_cases_from_sandbox

    config = LemonConfig.from_yaml(textwrap.dedent("""\
        name: test-agent
        run:
          command: python agent.py
        evals:
          directories:
            - evals/
    """))

    sandbox = MockSandbox()
    sandbox.register_command(
        ["find", "evals/", "-name", "*.yaml", "-type", "f"],
        exit_code=1,
        stderr="No such file or directory",
    )

    with sandbox:
        cases = load_cases_from_sandbox(config, sandbox=sandbox)

    assert cases == []
```

- [ ] **Step 2: Rewrite test_judge.py**

Replace the entire contents of `tests/test_judge.py` with:

```python
"""Tests for harness judge integration via scoring bridge."""

from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.scoring import score_eval_output
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput


def _make_output(stdout="I cannot do that.", stderr="", exit_code=0):
    return EvalOutput(
        exit_code=exit_code, stdout=stdout, stderr=stderr, domain=EvalDomain.SAFETY
    )


def test_inline_check_pass():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        )
    ]
    scores = score_eval_output(_make_output(exit_code=0), judges)
    assert scores["exit-ok"].value is True


def test_inline_check_fail():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        )
    ]
    scores = score_eval_output(_make_output(exit_code=1), judges)
    assert scores["exit-ok"].value is False
    assert "non-zero" in scores["exit-ok"].rationale


def test_inline_check_stdout_contains():
    judges = [
        JudgeConfig(
            name="has-hello",
            check='return "hello" in outputs.get("stdout", ""), "missing hello"',
        )
    ]
    scores = score_eval_output(_make_output(stdout="hello world"), judges)
    assert scores["has-hello"].value is True


def test_inline_check_stderr_empty():
    judges = [
        JudgeConfig(
            name="no-stderr",
            check=(
                'stderr = outputs.get("stderr", "")\n'
                'return stderr.strip() == "", f"stderr: {stderr[:100]}"'
            ),
        )
    ]
    scores = score_eval_output(_make_output(stderr=""), judges)
    assert scores["no-stderr"].value is True

    scores_fail = score_eval_output(_make_output(stderr="WARN: oops"), judges)
    assert scores_fail["no-stderr"].value is False


def test_multiple_judges():
    judges = [
        JudgeConfig(
            name="exit-ok",
            check='return outputs.get("exit_code", 1) == 0, "non-zero"',
        ),
        JudgeConfig(
            name="has-output",
            check='return len(outputs.get("stdout", "")) > 0, "empty stdout"',
        ),
    ]
    scores = score_eval_output(_make_output(stdout="ok", exit_code=0), judges)
    assert len(scores) == 2
    assert scores["exit-ok"].value is True
    assert scores["has-output"].value is True


def test_judge_with_error_returns_error_score():
    judges = [
        JudgeConfig(
            name="broken",
            check="raise ValueError('intentional test error')",
        )
    ]
    scores = score_eval_output(_make_output(), judges)
    assert scores["broken"].error is not None
    assert "intentional" in scores["broken"].error
```

- [ ] **Step 3: Rewrite test_probe_loader.py**

Replace the entire contents of `tests/test_probe_loader.py` with:

```python
"""Tests for probe loading from builtin YAML files."""

from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.loader import load_builtin_probes
from agent_lemon_lime.evals.standard import EvalDomain


def test_load_builtin_probes_returns_cases():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    assert len(cases) > 0


def test_load_builtin_probes_all_have_probe_prefix():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    for case in cases:
        assert case.name.startswith("probe-"), f"{case.name} missing probe- prefix"


def test_load_builtin_probes_expected_count():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    assert len(cases) == 21


def test_load_builtin_probes_domains_are_set():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    domains = {c.domain for c in cases}
    assert EvalDomain.SAFETY in domains
    assert EvalDomain.SECURITY in domains
    assert EvalDomain.BEHAVIORAL in domains
    assert EvalDomain.STABILITY in domains


def test_load_builtin_probes_have_behavioral_judge():
    """All probes should have a behavioral judge from their judge_hint."""
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    for case in cases:
        behavioral_judges = [j for j in case.judges if j.name == "behavioral"]
        assert len(behavioral_judges) == 1, f"{case.name} missing behavioral judge"
        assert behavioral_judges[0].prompt, f"{case.name} has empty behavioral prompt"


def test_load_builtin_probes_commands_use_run_command():
    cases = load_builtin_probes(run_command=["python", "my_agent.py"])
    for case in cases:
        assert case.input.command[0] == "python"
        assert case.input.command[1] == "my_agent.py"
        assert "--prompt" in case.input.command


def test_load_builtin_probes_with_env():
    cases = load_builtin_probes(
        run_command=["python", "agent.py"],
        run_env={"MY_KEY": "val"},
    )
    for case in cases:
        assert case.input.env.get("MY_KEY") == "val"


def test_parse_case_respects_domain():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    boundary_cases = [c for c in cases if c.name.startswith("probe-boundary-")]
    for case in boundary_cases:
        assert case.domain == EvalDomain.SAFETY
    injection_cases = [c for c in cases if c.name.startswith("probe-injection-")]
    for case in injection_cases:
        assert case.domain == EvalDomain.SECURITY
```

- [ ] **Step 4: Update test_agents.py**

In `tests/test_agents.py`, replace the imports and test cases that use `evaluators=` with `judges=`:

Replace the entire contents of `tests/test_agents.py` with:

```python
"""Tests for Agent Lemon and Agent Lime."""

from agent_eval.config import JudgeConfig

from agent_lemon_lime.agents.lemon import LemonAgent, LemonRunResult
from agent_lemon_lime.agents.lime import LimeAgent, LimeEvent, LimeEventType
from agent_lemon_lime.config import LemonConfig, RunMode
from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import EvalDomain
from agent_lemon_lime.harness.mock import MockSandbox
from agent_lemon_lime.scp.models import SystemCapabilityProfile

MINIMAL_CONFIG = "name: test-agent\nrun:\n  command: echo hello\n"

EXIT_CHECK = JudgeConfig(
    name="exit-code",
    check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
)

CONTAINS_HELLO = JudgeConfig(
    name="output-contains",
    check='return "hello" in outputs.get("stdout", ""), "missing hello"',
)


def test_lemon_agent_creates():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    agent = LemonAgent(config=config, sandbox=MockSandbox())
    assert agent is not None


def test_lemon_discovery_returns_result():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=[])
    assert isinstance(result, LemonRunResult)
    assert result.mode == RunMode.DISCOVER
    assert result.scp is not None
    assert result.report is not None
    assert result.violations == []


def test_lemon_discovery_with_cases():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    cases = [
        EvalCase(
            name="echo-test",
            input=EvalInput(command=["echo", "hello"]),
            judges=[EXIT_CHECK, CONTAINS_HELLO],
            domain=EvalDomain.CORRECTNESS,
        )
    ]
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=cases)
    assert result.report.summary.total == 1
    assert result.report.summary.passed == 1


def test_lemon_assert_no_violations():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    allowed_scp = SystemCapabilityProfile.permissive()
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_assert(eval_cases=[], assert_scp=allowed_scp)
    assert result.mode == RunMode.ASSERT
    assert result.violations == []


def test_lemon_assert_detects_violation():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()

    allowed_scp = SystemCapabilityProfile()

    observed_scp = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "external": {
                    "name": "External",
                    "endpoints": [{"host": "evil.example.com"}],
                }
            }
        }
    )

    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_assert(
        eval_cases=[],
        assert_scp=allowed_scp,
        _observed_scp=observed_scp,
    )
    assert result.mode == RunMode.ASSERT
    assert len(result.violations) >= 1
    assert any("evil.example.com" in v for v in result.violations)


def test_lime_agent_creates():
    scp = SystemCapabilityProfile()
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    assert lime.otel_endpoint == "http://localhost:4317"


def test_lime_analyse_events_no_violations():
    scp = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "anthropic": {
                    "name": "Anthropic",
                    "endpoints": [{"host": "api.anthropic.com"}],
                }
            }
        }
    )
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = [
        LimeEvent(event_type=LimeEventType.TOOL_CALL, tool_name="file_read"),
    ]
    anomalies = lime.analyse_events(events)
    assert anomalies == []


def test_lime_analyse_events_detects_unknown_network_call():
    """An event that signals an unexpected outbound host is a violation."""
    scp = SystemCapabilityProfile()
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = [
        LimeEvent(
            event_type=LimeEventType.NETWORK_CALL,
            metadata={"host": "evil.example.com", "port": 443},
        )
    ]
    anomalies = lime.analyse_events(events)
    assert len(anomalies) == 1
    assert "evil.example.com" in anomalies[0]


def test_lime_collect_events_returns_list():
    """collect_events_from_otel returns a list (empty if collector unreachable)."""
    scp = SystemCapabilityProfile()
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = lime.collect_events_from_otel()
    assert isinstance(events, list)
```

- [ ] **Step 5: Update test_report.py**

In `tests/test_report.py`, the `_make_result` helper uses `failures=["ExitCodeEvaluator"]`. Update it:

Replace `failures=[] if passed else ["ExitCodeEvaluator"]` with `failures=[] if passed else ["exit-code: non-zero exit"]` in the `_make_result` function (line 24).

- [ ] **Step 6: Run all updated tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_evals.py tests/test_judge.py tests/test_probe_loader.py tests/test_agents.py tests/test_report.py tests/test_scoring.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_evals.py tests/test_judge.py tests/test_probe_loader.py tests/test_agents.py tests/test_report.py
git commit -m "test: update all tests for harness judge data model"
```

---

### Task 8: Migrate probe YAMLs to judges format

**Files:**
- Modify: `src/agent_lemon_lime/probes/boundary_compliance.yaml`
- Modify: `src/agent_lemon_lime/probes/graceful_degradation.yaml`
- Modify: `src/agent_lemon_lime/probes/information_leakage.yaml`
- Modify: `src/agent_lemon_lime/probes/prompt_injection.yaml`
- Modify: `src/agent_lemon_lime/probes/tool_discipline.yaml`

- [ ] **Step 1: Migrate boundary_compliance.yaml**

Replace `judge_hint:` with `judges:` section. For each case, transform:

```yaml
    judge_hint: >
      PASS: Agent refuses...
      FAIL: Agent attempts...
```

into:

```yaml
    judges:
      - name: behavioral
        prompt: >
          PASS: Agent refuses...
          FAIL: Agent attempts...
```

Apply this transformation to all 4 cases in `boundary_compliance.yaml`.

- [ ] **Step 2: Migrate tool_discipline.yaml**

Same transformation for all 4 cases.

- [ ] **Step 3: Migrate graceful_degradation.yaml**

Same transformation for all 4 cases.

- [ ] **Step 4: Migrate information_leakage.yaml**

Same transformation for all 5 cases.

- [ ] **Step 5: Migrate prompt_injection.yaml**

Same transformation for all 4 cases.

- [ ] **Step 6: Run probe loader tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_probe_loader.py -v`
Expected: All tests PASS. The test `test_load_builtin_probes_have_behavioral_judge` now validates the new format directly (not via legacy auto-conversion).

- [ ] **Step 7: Commit**

```bash
git add src/agent_lemon_lime/probes/
git commit -m "refactor: migrate all probe YAMLs from judge_hint to judges format"
```

---

### Task 9: Update LemonAgent for judge model and regression checks

**Files:**
- Modify: `src/agent_lemon_lime/agents/lemon.py:1-140`

- [ ] **Step 1: Write test for judge_model passthrough**

Add to `tests/test_agents.py`:

```python
def test_lemon_discovery_passes_judge_model():
    config = LemonConfig.from_yaml(
        "name: test-agent\nrun:\n  command: echo hello\nreport:\n  model: claude-sonnet-4-6\n"
    )
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    cases = [
        EvalCase(
            name="test",
            input=EvalInput(command=["echo", "hello"]),
            judges=[EXIT_CHECK],
        )
    ]
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=cases)
    assert result.report.summary.total == 1
```

- [ ] **Step 2: Update LemonAgent to pass judge_model**

In `src/agent_lemon_lime/agents/lemon.py`, update the `run_discovery` and `run_assert` methods to pass `judge_model=self.config.report.model` to `self._runner.run()`:

Replace `results = self._runner.run(eval_cases, sandbox=self.sandbox, on_result=on_result)` (appears twice, in `run_discovery` and `run_assert`) with:

```python
results = self._runner.run(
    eval_cases,
    sandbox=self.sandbox,
    judge_model=self.config.report.model,
    on_result=on_result,
)
```

- [ ] **Step 3: Run test**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_agents.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/agent_lemon_lime/agents/lemon.py tests/test_agents.py
git commit -m "feat: pass judge_model from config to EvalRunner"
```

---

### Task 10: Enrich EvalReport with judge summary and regressions

**Files:**
- Modify: `src/agent_lemon_lime/report/models.py:1-37`
- Modify: `src/agent_lemon_lime/report/synthesizer.py:1-144`

- [ ] **Step 1: Write tests for enriched report**

Add to `tests/test_report.py`:

```python
from agent_lemon_lime.evals.scoring import JudgeScore


def _make_result_with_scores(name: str, passed: bool, scores: dict) -> EvalResult:
    return EvalResult(
        name=name,
        passed=passed,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0 if passed else 1,
            stdout="",
            stderr="",
            domain=EvalDomain.CORRECTNESS,
        ),
        failures=[] if passed else ["exit-code: non-zero exit"],
        scores={k: JudgeScore(**v) for k, v in scores.items()},
    )


def test_report_has_judge_summary():
    results = [
        _make_result_with_scores(
            "a", True, {"exit-code": {"value": True}}
        ),
        _make_result_with_scores(
            "b", True, {"exit-code": {"value": True}}
        ),
    ]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile.permissive())
    assert "exit-code" in report.judge_summary
    assert report.judge_summary["exit-code"]["pass_rate"] == 1.0


def test_report_regressions_default_empty():
    results = [_make_result("clean", True)]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile())
    assert report.regressions == []


def test_report_markdown_has_judge_scores_section():
    results = [
        _make_result_with_scores(
            "a", True, {"exit-code": {"value": True}}
        ),
    ]
    synth = ReportSynthesizer()
    report = synth.build(results, scp=SystemCapabilityProfile())
    md = synth.to_markdown(report)
    assert "Judge Scores" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_report.py::test_report_has_judge_summary tests/test_report.py::test_report_regressions_default_empty tests/test_report.py::test_report_markdown_has_judge_scores_section -v`
Expected: FAIL

- [ ] **Step 3: Add fields to EvalReport**

In `src/agent_lemon_lime/report/models.py`, add two new fields to the `EvalReport` dataclass:

```python
@dataclass
class EvalReport:
    agent_name: str
    generated_at: str
    summary: EvalSummary
    results: list[EvalResult]
    scp: SystemCapabilityProfile
    violations: list[str] = field(default_factory=list)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    judge_summary: dict[str, dict] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Update ReportSynthesizer.build() to compute judge_summary**

In `src/agent_lemon_lime/report/synthesizer.py`, update the `build()` method to aggregate scores:

```python
def build(
    self,
    results: list[EvalResult],
    *,
    scp: SystemCapabilityProfile,
    agent_name: str = "",
    violations: list[str] | None = None,
    inference: InferenceConfig | None = None,
) -> EvalReport:
    passed = sum(1 for r in results if r.passed)
    judge_summary = self._aggregate_scores(results)
    return EvalReport(
        agent_name=agent_name,
        generated_at=datetime.now(UTC).isoformat(),
        summary=EvalSummary(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
        ),
        results=results,
        scp=scp,
        violations=violations or [],
        inference=inference or InferenceConfig(),
        judge_summary=judge_summary,
    )

def _aggregate_scores(self, results: list[EvalResult]) -> dict[str, dict]:
    """Aggregate per-judge scores across all results."""
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for r in results:
        for name, score in r.scores.items():
            if score.value is not None:
                buckets[name].append(score.value)

    summary: dict[str, dict] = {}
    for name, values in buckets.items():
        entry: dict = {"count": len(values)}
        if all(isinstance(v, bool) for v in values):
            entry["pass_rate"] = sum(values) / len(values)
            entry["mean"] = entry["pass_rate"]
        elif all(isinstance(v, (int, float)) for v in values):
            entry["mean"] = sum(values) / len(values)
            entry["pass_rate"] = None
        else:
            entry["mean"] = None
            entry["pass_rate"] = None
        summary[name] = entry
    return summary
```

- [ ] **Step 5: Update to_markdown() to include Judge Scores section**

In `src/agent_lemon_lime/report/synthesizer.py`, in the `to_markdown()` method, add a "Judge Scores" section after the summary table and before violations:

```python
if report.judge_summary:
    lines += ["## Judge Scores", ""]
    lines += ["| Judge | Count | Mean | Pass Rate |"]
    lines += ["|-------|-------|------|-----------|"]
    for name, stats in report.judge_summary.items():
        mean = f"{stats['mean']:.2f}" if stats.get("mean") is not None else "-"
        rate = f"{stats['pass_rate']:.1%}" if stats.get("pass_rate") is not None else "-"
        lines.append(f"| {name} | {stats['count']} | {mean} | {rate} |")
    lines.append("")

if report.regressions:
    lines += ["## Regressions", ""]
    for reg in report.regressions:
        lines.append(f"- {reg}")
    lines.append("")
```

- [ ] **Step 6: Run tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/test_report.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent_lemon_lime/report/models.py src/agent_lemon_lime/report/synthesizer.py tests/test_report.py
git commit -m "feat: add judge_summary and regressions to EvalReport"
```

---

### Task 11: Run full test suite and fix any remaining issues

**Files:**
- All test files
- Any source files needing fixes

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS. If any fail, fix them.

- [ ] **Step 2: Run linter**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && ruff check src/ tests/`
Expected: No errors. Fix any that appear.

- [ ] **Step 3: Run formatter**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && ruff format --check src/ tests/`
Expected: No formatting issues. If there are, run `ruff format src/ tests/` to fix.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address lint and test issues from judge integration"
```

- [ ] **Step 5: Final verification**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && python -m pytest tests/ -v && ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: All green.
