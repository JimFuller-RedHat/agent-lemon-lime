# OpenShell Sandbox CLI Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `agent-lemon discover` and `agent-lemon assert-mode` run evals in an OpenShell sandbox with automatic gateway and provider pre-flight checks.

**Architecture:** Add a `SandboxConfig` to the existing Pydantic config model. Add a thin pre-flight layer in the CLI that shells out to the `openshell` CLI for gateway/provider checks. CLI flags override config values. No changes to `LemonAgent` or the sandbox protocol.

**Tech Stack:** Python 3.13, Pydantic, Typer, subprocess (for `openshell` CLI calls), pytest + unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agent_lemon_lime/config.py` | Modify | Add `SandboxConfig` model, add `sandbox` field to `LemonConfig` |
| `src/agent_lemon_lime/cli/lemon.py` | Modify | Add CLI flags, `_openshell_preflight`, `_create_sandbox`, `_resolve_sandbox_config` |
| `tests/test_config.py` | Modify | Add tests for `SandboxConfig` defaults and YAML parsing |
| `tests/test_preflight.py` | Create | Pre-flight, factory, and config resolution tests |

---

### Task 1: Add `SandboxConfig` to the config model

**Files:**
- Modify: `src/agent_lemon_lime/config.py:1-69`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test for SandboxConfig defaults**

In `tests/test_config.py`, add:

```python
def test_config_sandbox_defaults():
    minimal = LemonConfig.from_yaml("name: my-agent\nrun:\n  command: echo hi\n")
    assert minimal.sandbox.type == "local"
    assert minimal.sandbox.cluster is None
    assert minimal.sandbox.timeout == 30.0
    assert minimal.sandbox.ready_timeout_seconds == 120.0
    assert minimal.sandbox.auto_start_gateway is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_config.py::test_config_sandbox_defaults -v`
Expected: FAIL — `AttributeError: 'LemonConfig' object has no attribute 'sandbox'`

- [ ] **Step 3: Write failing test for SandboxConfig from YAML**

In `tests/test_config.py`, add:

```python
def test_config_sandbox_openshell():
    yaml_text = textwrap.dedent("""\
        name: my-agent
        run:
          command: echo hi
        sandbox:
          type: openshell
          cluster: my-cluster
          timeout: 60.0
          ready_timeout_seconds: 180.0
          auto_start_gateway: false
    """)
    config = LemonConfig.from_yaml(yaml_text)
    assert config.sandbox.type == "openshell"
    assert config.sandbox.cluster == "my-cluster"
    assert config.sandbox.timeout == 60.0
    assert config.sandbox.ready_timeout_seconds == 180.0
    assert config.sandbox.auto_start_gateway is False
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_config.py::test_config_sandbox_openshell -v`
Expected: FAIL

- [ ] **Step 5: Implement SandboxConfig**

In `src/agent_lemon_lime/config.py`, add after `ReportConfig`:

```python
class SandboxConfig(BaseModel):
    type: Literal["local", "openshell"] = "local"
    cluster: str | None = None
    timeout: float = 30.0
    ready_timeout_seconds: float = 120.0
    auto_start_gateway: bool = True
```

And add to `LemonConfig`:

```python
class LemonConfig(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    run: RunConfig
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    scp: SCPConfig = Field(default_factory=SCPConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
```

- [ ] **Step 6: Run all config tests to verify they pass**

Run: `uv run python -m pytest tests/test_config.py -v`
Expected: All PASS (new tests + existing tests unbroken)

- [ ] **Step 7: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/config.py tests/test_config.py`
Expected: Clean

- [ ] **Step 8: Commit**

```bash
git add src/agent_lemon_lime/config.py tests/test_config.py
git commit -m "feat: add SandboxConfig to LemonConfig"
```

---

### Task 2: Add `_resolve_sandbox_config` helper

**Files:**
- Create: `tests/test_preflight.py`
- Modify: `src/agent_lemon_lime/cli/lemon.py`

- [ ] **Step 1: Write failing tests for config resolution**

Create `tests/test_preflight.py`:

```python
"""Tests for OpenShell pre-flight checks and sandbox factory."""

import textwrap

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_preflight.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_sandbox_config'`

- [ ] **Step 3: Implement `_resolve_sandbox_config`**

In `src/agent_lemon_lime/cli/lemon.py`, add after the existing helper functions (after `_print_session_footer`):

```python
def _resolve_sandbox_config(
    config: LemonConfig,
    sandbox_flag: str | None,
    no_auto_start: bool,
) -> SandboxConfig:
    """Merge CLI flags over YAML sandbox config."""
    resolved = config.sandbox.model_copy()
    if sandbox_flag is not None:
        resolved.type = sandbox_flag  # type: ignore[assignment]
    if no_auto_start:
        resolved.auto_start_gateway = False
    return resolved
```

Add `SandboxConfig` to the imports at the top of the file:

```python
from agent_lemon_lime.config import LemonConfig, SandboxConfig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_preflight.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py
git commit -m "feat: add _resolve_sandbox_config for CLI flag merging"
```

---

### Task 3: Add `_openshell_preflight` function

**Files:**
- Modify: `src/agent_lemon_lime/cli/lemon.py`
- Modify: `tests/test_preflight.py`

- [ ] **Step 1: Write failing test — gateway running, providers exist**

Append to `tests/test_preflight.py`:

```python
import subprocess
from unittest.mock import patch, MagicMock

import typer


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
            args=[], returncode=0, stdout="Gateway running", stderr="",
        ),
        ("openshell", "provider", "list"): subprocess.CompletedProcess(
            args=[], returncode=0, stdout="anthropic\n", stderr="",
        ),
    }
    config = SandboxConfig(type="openshell")
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        _openshell_preflight(config)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_preflight.py::test_preflight_gateway_running_providers_exist -v`
Expected: FAIL — `ImportError: cannot import name '_openshell_preflight'`

- [ ] **Step 3: Write failing test — gateway down, auto-start on**

Append to `tests/test_preflight.py`:

```python
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
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="anthropic\n", stderr="")
        raise ValueError(f"Unexpected command: {cmd}")

    config = SandboxConfig(type="openshell", auto_start_gateway=True)
    with patch("subprocess.run", side_effect=side_effect):
        _openshell_preflight(config)

    assert ("openshell", "gateway", "start") in call_log
```

- [ ] **Step 4: Write failing test — gateway down, auto-start off**

Append to `tests/test_preflight.py`:

```python
def test_preflight_gateway_down_no_auto_start():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="",
        ),
    }
    config = SandboxConfig(type="openshell", auto_start_gateway=False)
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        with pytest.raises(SystemExit):
            _openshell_preflight(config)
```

Add `import pytest` to the top of the file.

- [ ] **Step 5: Write failing test — gateway running, no providers**

Append to `tests/test_preflight.py`:

```python
def test_preflight_no_providers():
    from agent_lemon_lime.cli.lemon import _openshell_preflight

    responses = {
        ("openshell", "status"): subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Gateway running", stderr="",
        ),
        ("openshell", "provider", "list"): subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        ),
    }
    config = SandboxConfig(type="openshell")
    with patch("subprocess.run", side_effect=_mock_run(responses)):
        with pytest.raises(SystemExit):
            _openshell_preflight(config)
```

- [ ] **Step 6: Implement `_openshell_preflight`**

In `src/agent_lemon_lime/cli/lemon.py`, add `import subprocess` at the top, then add after `_resolve_sandbox_config`:

```python
def _openshell_preflight(config: SandboxConfig) -> None:
    """Ensure OpenShell gateway is running and at least one provider is configured."""
    status = subprocess.run(
        ["openshell", "status"],
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        if config.auto_start_gateway:
            console.print("[yellow]Gateway not running — starting...[/yellow]")
            start = subprocess.run(
                ["openshell", "gateway", "start"],
                capture_output=True,
                text=True,
            )
            if start.returncode != 0:
                console.print(
                    f"[red]Error:[/red] Failed to start gateway: {start.stderr.strip()}"
                )
                raise typer.Exit(code=1)
            console.print("[green]Gateway started.[/green]")
        else:
            console.print(
                "[red]Error:[/red] OpenShell gateway is not running.\n"
                "  Start it with: [bold]openshell gateway start[/bold]"
            )
            raise typer.Exit(code=1)

    providers = subprocess.run(
        ["openshell", "provider", "list"],
        capture_output=True,
        text=True,
    )
    if not providers.stdout.strip():
        console.print(
            "[red]Error:[/red] No providers configured.\n"
            "  Add one with: [bold]openshell provider create "
            "--name anthropic --type anthropic --from-existing[/bold]"
        )
        raise typer.Exit(code=1)
```

- [ ] **Step 7: Run all preflight tests**

Run: `uv run python -m pytest tests/test_preflight.py -v`
Expected: All PASS

- [ ] **Step 8: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py
git commit -m "feat: add _openshell_preflight for gateway and provider checks"
```

---

### Task 4: Add `_create_sandbox` and wire into CLI commands

**Files:**
- Modify: `src/agent_lemon_lime/cli/lemon.py`
- Modify: `tests/test_preflight.py`

- [ ] **Step 1: Write failing test for `_create_sandbox` returning LocalSandbox**

Append to `tests/test_preflight.py`:

```python
from agent_lemon_lime.harness.local import LocalSandbox
from agent_lemon_lime.harness.openshell import OpenshellSandbox


def test_create_sandbox_local():
    from agent_lemon_lime.cli.lemon import _create_sandbox

    config = SandboxConfig(type="local")
    sandbox = _create_sandbox(config, workdir="/tmp/test")
    assert isinstance(sandbox, LocalSandbox)


def test_create_sandbox_openshell():
    from agent_lemon_lime.cli.lemon import _create_sandbox

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_preflight.py::test_create_sandbox_local tests/test_preflight.py::test_create_sandbox_openshell -v`
Expected: FAIL — `ImportError: cannot import name '_create_sandbox'`

- [ ] **Step 3: Implement `_create_sandbox`**

In `src/agent_lemon_lime/cli/lemon.py`, add after `_openshell_preflight`:

```python
def _create_sandbox(
    config: SandboxConfig, workdir: str
) -> LocalSandbox | OpenshellSandbox:
    """Build the appropriate sandbox from config, running pre-flight if needed."""
    if config.type == "openshell":
        from agent_lemon_lime.harness.openshell import OpenshellSandbox

        _openshell_preflight(config)
        return OpenshellSandbox(
            cluster=config.cluster,
            timeout=config.timeout,
            ready_timeout_seconds=config.ready_timeout_seconds,
        )
    from agent_lemon_lime.harness.local import LocalSandbox

    return LocalSandbox(workdir=workdir)
```

- [ ] **Step 4: Run factory tests**

Run: `uv run python -m pytest tests/test_preflight.py::test_create_sandbox_local tests/test_preflight.py::test_create_sandbox_openshell -v`
Expected: All PASS

- [ ] **Step 5: Wire into `discover` command**

Modify the `discover` function signature to add the new flags:

```python
@app.command()
def discover(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    scp: Annotated[str | None, typer.Option("--scp", help="Override SCP output path")] = None,
    report: Annotated[
        str | None, typer.Option("--report", help="Override report output path")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print stdout/stderr from each eval case")
    ] = False,
    sandbox: Annotated[
        str | None, typer.Option("--sandbox", help="Sandbox type: local or openshell")
    ] = None,
    no_auto_start_gateway: Annotated[
        bool,
        typer.Option("--no-auto-start-gateway", help="Don't auto-start the OpenShell gateway"),
    ] = False,
) -> None:
```

Replace the sandbox creation line inside `discover`:

```python
    # OLD:
    # sandbox = LocalSandbox(workdir=str(project))

    # NEW:
    sandbox_config = _resolve_sandbox_config(config, sandbox, no_auto_start_gateway)
    sbx = _create_sandbox(sandbox_config, workdir=str(project))
    agent = LemonAgent(config=config, sandbox=sbx)
```

Remove the `from agent_lemon_lime.harness.local import LocalSandbox` import inside `discover` (it's now handled inside `_create_sandbox`).

- [ ] **Step 6: Wire into `assert_mode` command**

Modify the `assert_mode` function signature to add the same flags:

```python
@app.command()
def assert_mode(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    scp: Annotated[str | None, typer.Option("--scp", help="SCP file to assert against")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print stdout/stderr from each eval case")
    ] = False,
    sandbox: Annotated[
        str | None, typer.Option("--sandbox", help="Sandbox type: local or openshell")
    ] = None,
    no_auto_start_gateway: Annotated[
        bool,
        typer.Option("--no-auto-start-gateway", help="Don't auto-start the OpenShell gateway"),
    ] = False,
) -> None:
```

Replace the sandbox creation line inside `assert_mode`:

```python
    # OLD:
    # sandbox = LocalSandbox(workdir=str(project))

    # NEW:
    sandbox_config = _resolve_sandbox_config(config, sandbox, no_auto_start_gateway)
    sbx = _create_sandbox(sandbox_config, workdir=str(project))
    agent = LemonAgent(config=config, sandbox=sbx)
```

Remove the `from agent_lemon_lime.harness.local import LocalSandbox` import inside `assert_mode`.

- [ ] **Step 7: Run existing CLI tests to verify no regressions**

Run: `uv run python -m pytest tests/test_cli.py -v`
Expected: All PASS (existing commands default to `local` sandbox)

- [ ] **Step 8: Run all tests**

Run: `uv run python -m pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All PASS

- [ ] **Step 9: Run ruff**

Run: `uv run ruff check src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py`
Expected: Clean

- [ ] **Step 10: Commit**

```bash
git add src/agent_lemon_lime/cli/lemon.py tests/test_preflight.py
git commit -m "feat: wire OpenShell sandbox into discover and assert-mode commands"
```
