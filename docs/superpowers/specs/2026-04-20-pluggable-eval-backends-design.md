# Pluggable Eval Backends — inspect_ai Integration

**Date:** 2026-04-20
**Status:** Approved

## Problem

Agent Lemon only runs custom YAML eval cases (exit code and output substring checks). There is no way to run standardized behavioral evaluations from established frameworks like [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) or promptfoo. Users who want behavioral testing (safety refusals, reasoning benchmarks, tool use correctness) must run these tools separately and manually correlate results.

## Solution

Add a pluggable eval backend system to Agent Lemon. Backends are external eval frameworks that run on the host (not inside the sandbox), produce scored results, and merge into Agent Lemon's unified report. inspect_ai is the first backend; promptfoo will follow the same protocol.

## Design Decisions

- **Subprocess execution, not Python API** — backends are invoked via their CLI (`inspect eval`). This avoids hard dependencies, version conflicts, and keeps agent-lemon-lime lightweight. inspect_ai is expected to be installed separately.
- **Host execution, not sandbox** — inspect_ai runs on the host with its own model credentials. Only the agent-under-test runs in the OpenShell sandbox. This is simpler and avoids routing inspect_ai's own API calls through the inference proxy.
- **One result per task** — each inspect_ai task maps to a single `EvalResult` in the report. Sample-level detail stays in inspect_ai's own logs.
- **Config-driven, no new CLI flags** — backends are configured in `agent-lemon.yaml` under `evals.backends`. The `discover` and `assert` commands pick them up automatically.
- **No SCP contribution** — backend results test model behavior, not system access. They appear in the report but don't influence the Secure Capability Profile.

## Backend Protocol

```python
class BackendResult(BaseModel):
    name: str                    # e.g. "inspect::safety/refusal"
    passed: bool
    score: float | None = None   # 0.0-1.0 if the backend provides one
    summary: str = ""            # one-line description of result
    details: str = ""            # verbose output (stderr, logs, etc.)
    domain: EvalDomain = EvalDomain.BEHAVIORAL


class EvalBackend(Protocol):
    name: str                    # "inspect", "promptfoo"

    def run(self, tasks: list[str], model: str, **kwargs) -> list[BackendResult]:
        """Run the specified tasks and return results."""
        ...

    def available(self) -> bool:
        """Return True if the backend CLI is installed and runnable."""
        ...
```

New backends (promptfoo, etc.) implement this same protocol. Agent Lemon's runner iterates configured backends, calls `run()`, and converts `BackendResult` objects into `EvalResult` objects.

## InspectBackend Implementation

### Execution flow

1. `available()` checks that `inspect` is on PATH via `shutil.which("inspect")`
2. Create a temporary log directory
3. For each task, run: `inspect eval <task> --model <model> --log-dir <tmpdir>`
4. Read the JSON log file from the log directory
5. Map the top-level `results` and `status` fields to a single `BackendResult`
6. Clean up the temporary directory

### Score mapping

- `status == "success"` and aggregate score >= `score_threshold` (configurable, default 1.0) → `passed = True`
- `status == "success"` and score < threshold → `passed = False`
- `status == "error"` → `passed = False`, error details in `details` field

### Model routing

The `--model` flag uses inspect_ai's own provider format (`anthropic/claude-opus-4-6`, `openai/gpt-4o`). This model is called directly by inspect_ai on the host — it is independent of the OpenShell sandbox's inference provider.

### No new dependencies

inspect_ai is not a Python dependency of agent-lemon-lime. It must be installed separately (`pip install inspect-ai`). If not installed, the backend reports itself as unavailable and Agent Lemon prints an actionable error.

## Config Changes

### `agent-lemon.yaml`

New `backends` field under `evals`:

```yaml
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
      score_threshold: 0.8    # optional, default 1.0
```

### Config model

```python
class BackendConfig(BaseModel):
    type: Literal["inspect"]          # "promptfoo" added later
    model: str                        # provider/model in backend's format
    tasks: list[str]                  # task identifiers
    score_threshold: float = 1.0      # pass threshold for aggregate score
```

Added to `EvalsConfig`:

```python
class EvalsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    skills: list[SkillSource] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)
```

Backward-compatible: existing configs without `backends` default to an empty list.

## Result Merging

### BackendResult → EvalResult mapping

```python
EvalResult(
    name="inspect::arc",
    domain=EvalDomain.BEHAVIORAL,
    passed=backend_result.passed,
    failures=[] if passed else [backend_result.summary],
    output=EvalOutput(
        exit_code=0,
        stdout=backend_result.summary,
        stderr=backend_result.details,
    ),
)
```

### New EvalDomain value

Add `BEHAVIORAL = "behavioral"` to the `EvalDomain` enum alongside `CORRECTNESS`, `SAFETY`, `STABILITY`.

### Report appearance

Backend results appear in the same table as YAML evals, grouped by domain:

```
── correctness ──────────────────────────────────
agent-exits-cleanly                    PASSED [25%]
greeting-responds                      PASSED [50%]

── behavioral ───────────────────────────────────
inspect::arc                           PASSED [75%]  (score: 0.92)
inspect::safety/refusal                FAILED [100%] (score: 0.65)
```

When a `BackendResult` has a non-null `score`, it is displayed in the report line. Pass/fail is determined by comparing the score against `score_threshold` from the config.

### Execution order

YAML eval cases run first (in the sandbox), then each backend runs on the host. Results are concatenated before report synthesis.

## Files Touched

### New files

| File | Purpose |
|------|---------|
| `src/agent_lemon_lime/evals/backends.py` | `EvalBackend` protocol, `BackendResult` model, `InspectBackend` class |
| `tests/test_backends.py` | Unit tests for backend protocol and inspect integration |

### Modified files

| File | Change |
|------|--------|
| `src/agent_lemon_lime/config.py` | Add `BackendConfig`, add `backends` field to `EvalsConfig` |
| `src/agent_lemon_lime/evals/standard.py` | Add `BEHAVIORAL` to `EvalDomain` |
| `src/agent_lemon_lime/evals/runner.py` | Add `run_backends()` to iterate configured backends and convert results |
| `src/agent_lemon_lime/cli/lemon.py` | Call `run_backends()` after YAML evals, merge results before report synthesis |
| `tests/test_config.py` | Test `BackendConfig` parsing and defaults |

No new dependencies. Everything uses `subprocess.run` and `json.loads`.

## Testing

Unit tests with mocked `subprocess.run` to simulate `inspect eval`:

1. **Task passes** — mock JSON log with `status: "success"` and score above threshold
2. **Task fails** — mock JSON log with `status: "success"` but score below threshold
3. **Task errors** — mock JSON log with `status: "error"`
4. **Backend not installed** — `shutil.which` returns `None`, backend reports unavailable
5. **Config parsing** — backends present, backends absent (backward compat), invalid type
6. **Result merging** — backend results appear alongside YAML results, grouped by domain

No integration tests — inspect_ai is an external tool and unit tests cover the protocol boundary.
