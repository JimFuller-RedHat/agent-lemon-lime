"""CLI for agent-lemon: capability discovery, assertion, and scaffolding."""

from __future__ import annotations

import pathlib
import subprocess
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

from agent_lemon_lime.config import LemonConfig, SandboxConfig
from agent_lemon_lime.evals.runner import EvalResult

if TYPE_CHECKING:
    from agent_lemon_lime.harness.local import LocalSandbox
    from agent_lemon_lime.harness.openshell import OpenshellSandbox

app = typer.Typer(name="agent-lemon", help="Agent Lemon — AI agent capability evaluation.")

console = Console()

_CONFIG_TEMPLATE = """\
name: {name}
version: "0.1.0"
description: "Capability evaluation for {name}"

run:
  command: "python src/agent.py"
  timeout_seconds: 300

evals:
  directories:
    - evals/

scp:
  output: ".agent-lemon/scp.yaml"
  assert_file: null

report:
  output: ".agent-lemon/report.md"
  format: markdown
"""

_GITHUB_ACTION_TEMPLATE = """\
name: Agent Lemon Eval

on:
  push:
    branches: [main]
  pull_request:

jobs:
  agent-lemon:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"

      - name: Install agent-lemon-lime
        run: uv pip install --system agent-lemon-lime

      - name: Run agent-lemon discover
        run: agent-lemon discover
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
"""


def _log_path(config: LemonConfig) -> str:
    """Derive log path: config.report.log if set, else .agent-lemon/{agent-name}.log."""
    return config.report.log or f".agent-lemon/{config.name}.log"


def _make_result_printer(
    total: int, *, verbose: bool
) -> tuple[Callable[[EvalResult], None], list[EvalResult]]:
    """Return (callback, results_list) for live per-case output."""
    seen: list[EvalResult] = []

    def _on_result(r: EvalResult) -> None:
        seen.append(r)
        n = len(seen)
        pct = int(n / total * 100) if total else 100
        pct_str = f"[{pct:3d}%]"
        status_word = "PASSED" if r.passed else "FAILED"
        status = (
            f"[bold green]{status_word}[/bold green]"
            if r.passed
            else f"[bold red]{status_word}[/bold red]"
        )
        label = f"{r.domain}::{r.name}"
        right_len = 1 + len(status_word) + 1 + len(pct_str)
        pad = max(1, console.width - len(label) - right_len)
        console.print(f"{label}{' ' * pad}{status} {pct_str}")
        if verbose:
            _print_captured(r)

    return _on_result, seen


def _print_captured(r: EvalResult) -> None:
    if r.output.stdout.strip():
        console.print("  [dim]Captured stdout:[/dim]")
        for line in r.output.stdout.rstrip().splitlines():
            console.print(f"    {line}")
    if r.output.stderr.strip():
        console.print("  [dim]Captured stderr:[/dim]")
        for line in r.output.stderr.rstrip().splitlines():
            console.print(f"    {line}", style="yellow")


def _print_failures(results: list[EvalResult], *, verbose: bool) -> None:
    failed = [r for r in results if not r.passed]
    if not failed:
        return
    console.print()
    console.rule("FAILURES", style="bold red")
    for r in failed:
        console.rule(r.name, style="red")
        for f in r.failures:
            console.print(f"  [red]{f}[/red]")
        if not verbose:
            _print_captured(r)


def _print_short_summary(results: list[EvalResult]) -> None:
    failed = [r for r in results if not r.passed]
    if not failed:
        return
    console.print()
    console.rule("short test summary info", style="bold")
    for r in failed:
        console.print(f"[bold red]FAILED[/bold red] {r.domain}::{r.name} - {', '.join(r.failures)}")


def _print_session_footer(passed: int, failed: int, elapsed: float) -> None:
    parts = []
    if passed:
        parts.append(f"[bold green]{passed} passed[/bold green]")
    if failed:
        parts.append(f"[bold red]{failed} failed[/bold red]")
    if not parts:
        parts.append("[dim]no tests ran[/dim]")
    console.rule(", ".join(parts) + f" in {elapsed:.2f}s")


def _print_setup(config: LemonConfig, sandbox_config: SandboxConfig) -> None:
    """Print Agent Lemon setup details (called in verbose mode)."""
    console.print("[dim]Agent Lemon Setup:[/dim]")
    console.print(f"  [dim]Agent   :[/dim] {config.name} v{config.version}")
    console.print(f"  [dim]Command :[/dim] {config.run.command}")
    console.print(f"  [dim]Sandbox :[/dim] {sandbox_config.type}")
    if sandbox_config.type == "openshell":
        cluster = sandbox_config.cluster or "(active cluster)"
        console.print(f"  [dim]Cluster :[/dim] {cluster}")
        console.print(f"  [dim]Gateway auto-start:[/dim] {sandbox_config.auto_start_gateway}")
        if sandbox_config.image is not None:
            console.print(f"  [dim]Image   :[/dim] {sandbox_config.image}")
        if sandbox_config.provider is not None:
            console.print(f"  [dim]Provider:[/dim] {sandbox_config.provider}")
        if sandbox_config.model is not None:
            console.print(f"  [dim]Model   :[/dim] {sandbox_config.model}")
    console.print()


def _resolve_sandbox_config(
    config: LemonConfig,
    sandbox_flag: str | None,
    no_auto_start: bool,
    provider_flag: str | None = None,
    model_flag: str | None = None,
    image_flag: str | None = None,
    discovery_policy_flag: str | None = None,
    ready_timeout_flag: float | None = None,
) -> SandboxConfig:
    """Merge CLI flags over YAML sandbox config."""
    resolved = config.sandbox.model_copy()
    if sandbox_flag is not None:
        resolved.type = sandbox_flag  # type: ignore[assignment]
    if no_auto_start:
        resolved.auto_start_gateway = False
    if provider_flag is not None:
        resolved.provider = provider_flag
    if model_flag is not None:
        resolved.model = model_flag
    if image_flag is not None:
        resolved.image = image_flag
    if discovery_policy_flag is not None:
        resolved.discovery_policy = discovery_policy_flag
    if ready_timeout_flag is not None:
        resolved.ready_timeout_seconds = ready_timeout_flag
    return resolved


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
                console.print(f"[red]Error:[/red] Failed to start gateway: {start.stderr.strip()}")
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

    _configure_inference(config)


def _configure_inference(config: SandboxConfig) -> None:
    """Set the gateway-level inference provider/model if configured."""
    if config.provider is None and config.model is None:
        return

    if config.provider is None or config.model is None:
        missing = "model" if config.model is None else "provider"
        console.print(
            f"[red]Error:[/red] Both sandbox.provider and sandbox.model "
            f"are required — {missing} is missing."
        )
        raise typer.Exit(code=1)

    cmd = [
        "openshell", "inference", "set",
        "--provider", config.provider,
        "--model", config.model,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(
            f"[red]Error:[/red] Failed to configure inference: "
            f"{result.stderr.strip()}\n"
            f"  Create a provider with: [bold]openshell provider create "
            f"--name {config.provider} --type {config.provider} --from-existing[/bold]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]Inference configured:[/green] "
        f"provider={config.provider}, model={config.model}"
    )


def _resolve_discovery_policy(
    config: SandboxConfig,
    *,
    permissive_flag: bool = False,
) -> Any:
    """Build the protobuf SandboxPolicy for discovery mode.

    Uses the custom YAML file from config if set, otherwise falls
    back to the built-in restrictive discovery policy. The
    ``--permissive`` flag selects audit-only mode instead.
    """
    from agent_lemon_lime.scp.converter import to_sandbox_policy
    from agent_lemon_lime.scp.models import SystemCapabilityProfile

    if permissive_flag:
        scp = SystemCapabilityProfile.permissive()
        console.print("[dim]Discovery policy:[/dim] permissive (audit-only)")
    elif config.discovery_policy is not None:
        try:
            scp = SystemCapabilityProfile.from_yaml(config.discovery_policy)
        except FileNotFoundError:
            console.print(
                f"[red]Error:[/red] Discovery policy not found: "
                f"{config.discovery_policy}"
            )
            raise typer.Exit(code=1) from None
        console.print(
            f"[dim]Discovery policy:[/dim] {config.discovery_policy}"
        )
    else:
        scp = SystemCapabilityProfile.discovery()
        console.print("[dim]Discovery policy:[/dim] built-in (deny-by-default)")

    return to_sandbox_policy(scp)


def _create_sandbox(
    config: SandboxConfig,
    workdir: str,
    setup_command: str | None = None,
    *,
    discovery: bool = False,
    permissive_flag: bool = False,
) -> LocalSandbox | OpenshellSandbox:
    """Build the appropriate sandbox from config, running pre-flight if needed."""
    if config.type == "openshell":
        from agent_lemon_lime.harness.openshell import OpenshellSandbox

        _openshell_preflight(config)
        providers = [config.provider] if config.provider else []
        policy = None
        upload_workdir = None if config.image else workdir
        return OpenshellSandbox(
            cluster=config.cluster,
            timeout=config.timeout,
            ready_timeout_seconds=config.ready_timeout_seconds,
            workdir=upload_workdir,
            setup_command=setup_command,
            providers=providers,
            policy=policy,
            image=config.image,
        )
    from agent_lemon_lime.harness.local import LocalSandbox

    return LocalSandbox(workdir=workdir)


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
    provider: Annotated[
        str | None, typer.Option("--provider", help="OpenShell inference provider name")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="OpenShell inference model name")
    ] = None,
    image: Annotated[
        str | None, typer.Option("--image", help="OpenShell sandbox container image")
    ] = None,
    discovery_policy: Annotated[
        str | None,
        typer.Option("--discovery-policy", help="Custom discovery policy YAML file"),
    ] = None,
    permissive: Annotated[
        bool,
        typer.Option(
            "--permissive/--no-permissive",
            help="Use permissive (audit-only) policy; enabled by default in discover mode",
        ),
    ] = True,
    ready_timeout: Annotated[
        float | None,
        typer.Option("--ready-timeout", help="Sandbox ready timeout in seconds"),
    ] = None,
) -> None:
    """Discover capabilities: run evals and generate SCP profile + report."""
    from agent_lemon_lime.agents.lemon import LemonAgent
    from agent_lemon_lime.evals.loader import (
        default_case_from_config,
        load_cases_from_config,
        load_cases_from_sandbox,
    )
    from agent_lemon_lime.report.synthesizer import ReportSynthesizer

    t0 = time.monotonic()
    console.rule("agent-lemon: discover")

    try:
        config = LemonConfig.from_dir(project_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    project = pathlib.Path(project_dir).resolve()

    sandbox_config = _resolve_sandbox_config(
        config, sandbox, no_auto_start_gateway, provider, model,
        image_flag=image, discovery_policy_flag=discovery_policy,
        ready_timeout_flag=ready_timeout,
    )
    if verbose:
        _print_setup(config, sandbox_config)

    sbx = _create_sandbox(
        sandbox_config, workdir=str(project),
        setup_command=config.run.setup,
        discovery=True, permissive_flag=permissive,
    )

    image_only = sandbox_config.image is not None and sandbox_config.type == "openshell"
    if image_only:
        sbx.__enter__()
        cases = load_cases_from_sandbox(config, sandbox=sbx)
    else:
        cases = load_cases_from_config(config, project_dir=project)

    smoke_test = not cases
    if smoke_test:
        cases = [default_case_from_config(config)]

    backend_task_count = sum(len(b.tasks) for b in config.evals.backends)
    n = len(cases) + backend_task_count
    note = " [dim](smoke test from run.command)[/dim]" if smoke_test else ""
    console.print(f"collected {n} item{'s' if n != 1 else ''}{note}\n")

    on_result, _ = _make_result_printer(n, verbose=verbose)
    agent = LemonAgent(config=config, sandbox=sbx, sandbox_config=sandbox_config)

    from agent_lemon_lime.evals.backends import run_backends

    backend_results = run_backends(config.evals.backends)
    for br in backend_results:
        if on_result:
            on_result(br)

    result = agent.run_discovery(
        eval_cases=cases, on_result=on_result, backend_results=backend_results,
    )

    if image_only:
        sbx.__exit__(None, None, None)

    scp_path = scp or config.scp.output
    result.scp.to_yaml(scp_path)

    synthesizer = ReportSynthesizer()
    report_path = report or config.report.output
    synthesizer.write(result.report, path=report_path)

    log_path = _log_path(config)
    synthesizer.write_log(result.report, path=log_path, mode="discover")

    _print_failures(result.report.results, verbose=verbose)
    _print_short_summary(result.report.results)

    console.print()
    console.print(f"  [dim]SCP    →[/dim] [green]{scp_path}[/green]")
    console.print(f"  [dim]Report →[/dim] [green]{report_path}[/green]")
    console.print(f"  [dim]Log    →[/dim] [green]{log_path}[/green]")
    console.print()

    s = result.report.summary
    _print_session_footer(s.passed, s.failed, time.monotonic() - t0)


@app.command(name="assert")
def assert_cmd(
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
    provider: Annotated[
        str | None, typer.Option("--provider", help="OpenShell inference provider name")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="OpenShell inference model name")
    ] = None,
    image: Annotated[
        str | None, typer.Option("--image", help="OpenShell sandbox container image")
    ] = None,
    ready_timeout: Annotated[
        float | None,
        typer.Option("--ready-timeout", help="Sandbox ready timeout in seconds"),
    ] = None,
) -> None:
    """Assert mode: run evals against a defined SCP and report violations."""
    from agent_lemon_lime.agents.lemon import LemonAgent
    from agent_lemon_lime.evals.loader import (
        default_case_from_config,
        load_cases_from_config,
        load_cases_from_sandbox,
    )
    from agent_lemon_lime.report.synthesizer import ReportSynthesizer
    from agent_lemon_lime.scp.models import SystemCapabilityProfile

    t0 = time.monotonic()
    console.rule("agent-lemon: assert")

    try:
        config = LemonConfig.from_dir(project_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    scp_path = scp or config.scp.assert_file
    if scp_path is None:
        console.print(
            "[red]Error:[/red] No SCP file specified. "
            "Use --scp or set scp.assert_file in agent-lemon.yaml."
        )
        raise typer.Exit(code=1)

    try:
        assert_scp = SystemCapabilityProfile.from_yaml(scp_path)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] SCP file not found: {scp_path}")
        raise typer.Exit(code=1) from None

    project = pathlib.Path(project_dir).resolve()

    sandbox_config = _resolve_sandbox_config(
        config, sandbox, no_auto_start_gateway, provider, model,
        image_flag=image, ready_timeout_flag=ready_timeout,
    )
    if verbose:
        _print_setup(config, sandbox_config)

    sbx = _create_sandbox(sandbox_config, workdir=str(project), setup_command=config.run.setup)

    image_only = sandbox_config.image is not None and sandbox_config.type == "openshell"
    if image_only:
        sbx.__enter__()
        cases = load_cases_from_sandbox(config, sandbox=sbx)
    else:
        cases = load_cases_from_config(config, project_dir=project)

    smoke_test = not cases
    if smoke_test:
        cases = [default_case_from_config(config)]

    backend_task_count = sum(len(b.tasks) for b in config.evals.backends)
    n = len(cases) + backend_task_count
    note = " [dim](smoke test from run.command)[/dim]" if smoke_test else ""
    console.print(f"collected {n} item{'s' if n != 1 else ''}{note}\n")

    on_result, _ = _make_result_printer(n, verbose=verbose)
    agent = LemonAgent(config=config, sandbox=sbx, sandbox_config=sandbox_config)

    from agent_lemon_lime.evals.backends import run_backends

    backend_results = run_backends(config.evals.backends)
    for br in backend_results:
        if on_result:
            on_result(br)

    result = agent.run_assert(
        eval_cases=cases, assert_scp=assert_scp,
        on_result=on_result, backend_results=backend_results,
    )

    if image_only:
        sbx.__exit__(None, None, None)

    report_path = config.report.output
    synth = ReportSynthesizer()
    synth.write(result.report, path=report_path)

    log_path = _log_path(config)
    synth.write_log(result.report, path=log_path, mode="assert")

    _print_failures(result.report.results, verbose=verbose)
    _print_short_summary(result.report.results)

    if result.violations:
        console.print()
        console.rule("SCP Violations", style="bold red")
        for v in result.violations:
            console.print(f"  [red]•[/red] {v}")

    console.print()
    console.print(f"  [dim]Report →[/dim] [green]{report_path}[/green]")
    console.print(f"  [dim]Log    →[/dim] [green]{log_path}[/green]")
    console.print()

    s = result.report.summary
    _print_session_footer(s.passed, s.failed, time.monotonic() - t0)

    if result.violations:
        raise typer.Exit(code=1)


@app.command()
def init(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    name: Annotated[str, typer.Option("--name", help="Agent name")] = "my-agent",
) -> None:
    """Generate an agent-lemon.yaml config template."""
    config_file = pathlib.Path(project_dir) / "agent-lemon.yaml"
    if config_file.exists():
        console.print(f"[yellow]Warning:[/yellow] {config_file} already exists. Skipping.")
        return

    config_file.write_text(_CONFIG_TEMPLATE.format(name=name))
    console.print(f"[green]Created[/green] {config_file}")


@app.command()
def action(
    output: Annotated[
        str,
        typer.Option("--output", help="Output path for the workflow file"),
    ] = ".github/workflows/agent-lemon.yml",
) -> None:
    """Generate a GitHub Actions workflow file for agent-lemon."""
    out = pathlib.Path(output)
    if out.exists():
        console.print(f"[yellow]Warning:[/yellow] {out} already exists — skipping.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_GITHUB_ACTION_TEMPLATE)
    console.print(f"[green]Generated GitHub Action:[/green] {out}")
