# OpenShell Sandbox CLI Integration

**Date:** 2026-04-16
**Status:** Approved

## Problem

Agent Lemon's CLI hardcodes `LocalSandbox` for eval execution. There is no way to run evals in an OpenShell sandbox from the CLI, even though `OpenshellSandbox` exists and implements `AbstractSandbox`. Users who want sandboxed eval runs must manage the OpenShell gateway and providers manually, with no feedback from Agent Lemon about infrastructure readiness.

## Solution

Add OpenShell sandbox support to the CLI via:

1. A `SandboxConfig` in `agent-lemon.yaml` (with CLI flag overrides)
2. Pre-flight checks that ensure the gateway is running and providers are configured
3. Automatic gateway start (on by default, opt-out via flag/config)

## Design

### Config changes (`config.py`)

New `SandboxConfig` model:

```python
class SandboxConfig(BaseModel):
    type: Literal["local", "openshell"] = "local"
    cluster: str | None = None
    timeout: float = 30.0
    ready_timeout_seconds: float = 120.0
    auto_start_gateway: bool = True
```

Added to `LemonConfig`:

```python
sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
```

Example YAML:

```yaml
sandbox:
  type: openshell
  cluster: my-cluster
  auto_start_gateway: true
```

Backward-compatible: existing configs without `sandbox` default to `local`.

### CLI changes (`cli/lemon.py`)

**New flags on `discover` and `assert-mode`:**

- `--sandbox [local|openshell]` — overrides `sandbox.type` from config
- `--no-auto-start-gateway` — overrides `sandbox.auto_start_gateway` to `False`

**New functions:**

`_openshell_preflight(config: SandboxConfig) -> None`:
1. Run `openshell status` to check gateway (exit code 0 = running)
2. If not running and `auto_start_gateway` is `True`: run `openshell gateway start`, fail if that errors
3. If not running and `auto_start_gateway` is `False`: print actionable error, exit 1
4. Run `openshell provider list` to check providers exist
5. If no providers: print actionable error with example `openshell provider create` command, exit 1

`_create_sandbox(config: SandboxConfig, workdir: str) -> AbstractSandbox`:
- If `openshell`: run pre-flight, return `OpenshellSandbox(cluster=..., timeout=..., ready_timeout_seconds=...)`
- If `local`: return `LocalSandbox(workdir=workdir)`

`_resolve_sandbox_config(config: LemonConfig, sandbox_flag: str | None, no_auto_start: bool) -> SandboxConfig`:
- Start from `config.sandbox`
- Override `type` if `--sandbox` flag provided
- Override `auto_start_gateway` to `False` if `--no-auto-start-gateway` set

Both `discover` and `assert_mode` replace direct `LocalSandbox` construction with:

```python
sandbox_config = _resolve_sandbox_config(config, sandbox_flag, no_auto_start)
sandbox = _create_sandbox(sandbox_config, workdir=str(project))
```

### Unchanged components

- **`LemonAgent`** — already accepts `AbstractSandbox`
- **`OpenshellSandbox`** — already implements the protocol
- **`harness/base.py`** — no new protocol methods
- **`init` / `action` commands** — unchanged

### Pre-flight behavior matrix

| Gateway status | auto_start_gateway | Providers exist | Result |
|---|---|---|---|
| Running | (any) | Yes | Proceed |
| Running | (any) | No | Exit 1: "No providers configured" |
| Down | True | (any) | Start gateway, then check providers |
| Down | False | (any) | Exit 1: "Gateway not running" |

## Testing

Unit tests covering:

1. **Pre-flight scenarios** — mock `subprocess.run` for `openshell status`, `openshell gateway start`, `openshell provider list`. Four cases from the behavior matrix above.
2. **`_create_sandbox`** — returns correct sandbox type based on config.
3. **`_resolve_sandbox_config`** — CLI flags override YAML config values.
4. **Backward compatibility** — existing tests pass unchanged (default config uses `local`).

## Files touched

| File | Change |
|---|---|
| `src/agent_lemon_lime/config.py` | Add `SandboxConfig`, add `sandbox` field to `LemonConfig` |
| `src/agent_lemon_lime/cli/lemon.py` | Add flags, `_openshell_preflight`, `_create_sandbox`, `_resolve_sandbox_config` |
| `tests/test_preflight.py` (new) | Pre-flight and factory unit tests |

No new dependencies. Pre-flight uses `subprocess.run` to shell out to the `openshell` CLI.
