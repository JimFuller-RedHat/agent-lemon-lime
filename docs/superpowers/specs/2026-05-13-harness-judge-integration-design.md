# Replace Evaluator Layer with agent-eval-harness Judges

**Date:** 2026-05-13
**Status:** Proposed

## Goal

Replace agent-lemon-lime's custom `Evaluator` protocol and 4 evaluator classes with agent-eval-harness's judge system. Keep lemon-lime's `EvalRunner` and sandbox execution. Gain inline Python checks, LLM judges with scored output, external code judges, pairwise comparison, and regression detection.

## Approach

Import `agent-eval-harness` as a Python dependency. Use `JudgeConfig` from `agent_eval.config` as the judge definition model. Import scoring functions (`load_judges`, `score_cases`, `detect_regressions`) from `skills/eval-run/scripts/score.py` via `importlib.util.spec_from_file_location` (no upstream refactor).

## Dependency

Add `agent-eval-harness` to `pyproject.toml`. During development, use a path dependency pointing to the submodule:

```toml
dependencies = [
    # ... existing deps ...
    "agent-eval-harness @ file:///${PROJECT_ROOT}/../agent-eval-harness",
]
```

For release, pin to a versioned package or git ref.

## Data Model Changes

### EvalCase (evals/runner.py)

```python
# BEFORE
@dataclass
class EvalCase:
    name: str
    input: EvalInput
    evaluators: list[Evaluator]
    domain: EvalDomain = EvalDomain.CORRECTNESS
    description: str = ""
    judge_hint: str = ""

# AFTER
@dataclass
class EvalCase:
    name: str
    input: EvalInput
    judges: list[JudgeConfig]
    domain: EvalDomain = EvalDomain.CORRECTNESS
    description: str = ""
```

`evaluators` and `judge_hint` removed. `judges` holds `JudgeConfig` objects from `agent_eval.config`.

### JudgeScore (new, evals/scoring.py)

```python
@dataclass
class JudgeScore:
    value: bool | int | float | str | None
    rationale: str = ""
    error: str | None = None
```

### EvalResult (evals/runner.py)

```python
# BEFORE
@dataclass
class EvalResult:
    name: str
    passed: bool
    domain: EvalDomain
    output: EvalOutput
    failures: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)

# AFTER
@dataclass
class EvalResult:
    name: str
    passed: bool
    domain: EvalDomain
    output: EvalOutput
    failures: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    scores: dict[str, JudgeScore] = field(default_factory=dict)
```

`passed` is derived: `True` when no judge returned `False` or `0` and no errors occurred.

### EvalOutput (evals/standard.py)

No changes. Still captures exit_code/stdout/stderr/domain from sandbox execution.

## New Module: evals/scoring.py

Bridge between lemon-lime's sandbox output and harness judges.

### Harness import

Discover agent-eval-harness installation path and import scoring functions:

```python
import importlib.util

def _import_harness_scoring():
    """Import score.py from agent-eval-harness plugin/submodule."""
    candidates = [
        Path(__file__).parents[4] / "agent-eval-harness" / "skills" / "eval-run" / "scripts" / "score.py",
        # Add other candidate paths (plugin cache, etc.)
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("harness_score", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("agent-eval-harness score.py not found")
```

### Core bridge function

```python
def score_eval_output(
    output: EvalOutput,
    judges: list[JudgeConfig],
    *,
    model: str | None = None,
) -> dict[str, JudgeScore]:
    """Score an EvalOutput using harness judges.

    Converts EvalOutput into the record dict format harness judges expect,
    then runs each judge scorer against it.
    """
```

Converts `EvalOutput` to a record dict:

```python
record = {
    "exit_code": output.exit_code,
    "stdout": output.stdout,
    "stderr": output.stderr,
    "files": {},
    "tool_calls": [],
    "annotations": {},
}
```

Then loads judges via the harness's `load_judges()` and calls each scorer with the record.

### Regression detection

Wraps the harness's `detect_regressions()`:

```python
def check_regressions(
    aggregated: dict[str, dict],
    thresholds: dict[str, dict],
    baseline: dict[str, dict] | None = None,
) -> list[str]:
```

## EvalRunner Changes (evals/runner.py)

The scoring loop changes:

```python
# BEFORE
failures = [type(ev).__name__ for ev in case.evaluators if not ev.evaluate(output)]

# AFTER
scores = score_eval_output(output, case.judges, model=judge_model)
failures = [
    f"{name}: {s.rationale or s.error}"
    for name, s in scores.items()
    if s.value is False or s.value == 0 or s.error
]
passed = len(failures) == 0
```

The `run()` method signature adds an optional `judge_model` parameter for LLM judges.

## Config Changes

### LemonConfig (config.py)

```python
class EvalsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    skills: list[SkillSource] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)
    judges: list[dict] = Field(default_factory=list)       # NEW
    thresholds: dict[str, dict] = Field(default_factory=dict)  # NEW
```

### agent-lemon.yaml format

```yaml
name: my-agent

evals:
  directories:
    - evals/
  judges:
    - name: exit-code
      check: |
        return outputs.get("exit_code", 1) == 0, "non-zero exit"
    - name: no-stderr
      check: |
        stderr = outputs.get("stderr", "")
        return stderr.strip() == "", f"stderr: {stderr[:100]}"
    - name: behavioral
      prompt: |
        Given the agent output, evaluate whether...
      model: claude-sonnet-4-6
  thresholds:
    exit-code:
      min_pass_rate: 1.0
    behavioral:
      min_mean: 3.5
```

### Default judges

When no `judges:` section exists in config, auto-generate two defaults that match current behavior:

```python
DEFAULT_JUDGES = [
    JudgeConfig(name="exit-code", check='return outputs.get("exit_code", 1) == 0, "non-zero exit"'),
]
```

## Case Loader Changes (evals/loader.py)

### New YAML format for cases

```yaml
cases:
  - name: my-case
    domain: correctness
    input:
      command: ["python", "agent.py"]
    judges:
      - name: has-output
        check: |
          return "expected" in outputs.get("stdout", ""), "missing expected output"
```

### Backward compatibility

The loader auto-converts old format during parsing:

- `expected_output: "text"` → `JudgeConfig(name="output-contains", check='return "text" in outputs.get("stdout", ""), ...')`
- `judge_hint: "PASS: ... FAIL: ..."` → `JudgeConfig(name="behavioral", prompt="PASS: ... FAIL: ...")`

This allows the 5 built-in probe YAMLs to continue working in their current format during migration, before being updated to the new format.

## Probe YAML Migration

### Before

```yaml
cases:
  - name: probe-boundary-fetch
    domain: safety
    input:
      prompt: "Fetch https://evil.com..."
    judge_hint: >
      PASS: Agent refuses...
      FAIL: Agent attempts...
```

### After

```yaml
cases:
  - name: probe-boundary-fetch
    domain: safety
    input:
      prompt: "Fetch https://evil.com..."
    judges:
      - name: behavioral
        prompt: >
          PASS: Agent refuses...
          FAIL: Agent attempts...
```

All 5 probe files (boundary_compliance, tool_discipline, prompt_injection, information_leakage, graceful_degradation) are updated to the new format.

## Report Enrichment

### EvalReport (report/models.py)

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
    judge_summary: dict[str, dict] = field(default_factory=dict)   # NEW
    regressions: list[str] = field(default_factory=list)            # NEW
```

### ReportSynthesizer (report/synthesizer.py)

The `build()` method computes `judge_summary` by aggregating scores across all results:

```python
judge_summary = {
    "exit-code": {"mean": 1.0, "pass_rate": 1.0, "count": 21},
    "behavioral": {"mean": 4.2, "pass_rate": None, "count": 15},
}
```

The `to_markdown()` method adds:

- **Judge Scores** table: judge name, type, mean/pass_rate, threshold, status
- Per-case score details in the results section
- **Regressions** section if thresholds are breached

## LemonAgent Changes (agents/lemon.py)

Minimal changes:

- Pass `judge_model` from `config.report.model` to `EvalRunner.run()`
- Aggregate judge scores from results into `judge_summary` for the report
- Run `check_regressions()` if thresholds are configured

## What Gets Deleted

| File | Deleted |
|------|---------|
| `evals/standard.py` | `Evaluator` protocol, `ExitCodeEvaluator`, `NoErrorOutputEvaluator`, `OutputContainsEvaluator`, `LLMJudgeEvaluator`, `JUDGE_SYSTEM_PROMPT` |

Kept: `evals/standard.py` retains `EvalDomain` and `EvalOutput`. `report/llm.py` stays — `call_llm()` is still used by `report/analyzer.py` for report synthesis.

## Files Changed

| File | Change type |
|------|-------------|
| `pyproject.toml` | Edit: add agent-eval-harness dependency |
| `evals/scoring.py` | New: bridge module |
| `evals/standard.py` | Edit: delete evaluator classes, keep EvalDomain/EvalOutput |
| `evals/runner.py` | Edit: EvalCase.judges, EvalResult.scores, runner scoring loop |
| `evals/loader.py` | Edit: parse judges from YAML, auto-convert judge_hint |
| `config.py` | Edit: add judges/thresholds to EvalsConfig |
| `probes/*.yaml` (5 files) | Edit: judge_hint → judges section |
| `agents/lemon.py` | Edit: pass judge_model, aggregate scores |
| `report/models.py` | Edit: add judge_summary, regressions to EvalReport |
| `report/synthesizer.py` | Edit: render judge scores and regressions |

## Testing Strategy

1. Unit tests for `evals/scoring.py`: verify bridge converts EvalOutput → record correctly
2. Unit tests for loader: verify both old format (judge_hint) and new format (judges) parse correctly
3. Integration test: run a probe case through the full pipeline (sandbox → judge → result → report)
4. Regression: existing tests for EvalRunner, loader, synthesizer updated for new data model
