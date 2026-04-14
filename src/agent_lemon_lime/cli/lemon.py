"""CLI for agent-lemon: capability discovery, assertion, and scaffolding."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def _print_summary_table(total: int, passed: int, failed: int) -> None:
    pass_rate = passed / total if total > 0 else 0.0
    table = Table(title="Eval Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total", str(total))
    table.add_row("Passed", str(passed))
    table.add_row("Failed", str(failed))
    table.add_row("Pass Rate", f"{pass_rate:.1%}")
    console.print(table)


@app.command()
def discover(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    scp: Annotated[
        str | None, typer.Option("--scp", help="Override SCP output path")
    ] = None,
    report: Annotated[
        str | None, typer.Option("--report", help="Override report output path")
    ] = None,
) -> None:
    """Discover capabilities: run evals and generate SCP profile + report."""
    from agent_lemon_lime.agents.lemon import LemonAgent
    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.harness.mock import MockSandbox

    console.print(Panel("[bold cyan]Agent Lemon — Discovery Mode[/bold cyan]"))

    try:
        config = LemonConfig.from_dir(project_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    with MockSandbox() as sandbox:
        agent = LemonAgent(config=config, sandbox=sandbox)
        result = agent.run_discovery(eval_cases=[])

    scp_path = scp or config.scp.output
    result.scp.to_yaml(scp_path)
    console.print(f"SCP written to [green]{scp_path}[/green]")

    from agent_lemon_lime.report.synthesizer import ReportSynthesizer

    synthesizer = ReportSynthesizer()
    report_path = report or config.report.output
    synthesizer.write(result.report, path=report_path)
    console.print(f"Report written to [green]{report_path}[/green]")

    s = result.report.summary
    _print_summary_table(s.total, s.passed, s.failed)


@app.command()
def assert_mode(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    scp: Annotated[
        str | None, typer.Option("--scp", help="SCP file to assert against")
    ] = None,
) -> None:
    """Assert mode: run evals against a defined SCP and report violations."""
    from agent_lemon_lime.agents.lemon import LemonAgent
    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.harness.mock import MockSandbox
    from agent_lemon_lime.scp.models import SystemCapabilityProfile

    console.print(Panel("[bold yellow]Agent Lemon — Assert Mode[/bold yellow]"))

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

    with MockSandbox() as sandbox:
        agent = LemonAgent(config=config, sandbox=sandbox)
        result = agent.run_assert(eval_cases=[], assert_scp=assert_scp)

    report_path = config.report.output
    from agent_lemon_lime.report.synthesizer import ReportSynthesizer
    ReportSynthesizer().write(result.report, path=report_path)

    s = result.report.summary
    _print_summary_table(s.total, s.passed, s.failed)

    if result.violations:
        console.print("[bold red]SCP Violations:[/bold red]")
        for v in result.violations:
            console.print(f"  [red]•[/red] {v}")
        raise typer.Exit(code=1)

    console.print("[green]All capability assertions passed.[/green]")


@app.command()
def init(
    project_dir: Annotated[str, typer.Option("--project-dir", help="Project root")] = ".",
    name: Annotated[str, typer.Option("--name", help="Agent name")] = "my-agent",
) -> None:
    """Generate an agent-lemon.yaml config template."""
    config_file = pathlib.Path(project_dir) / "agent-lemon.yaml"
    if config_file.exists():
        console.print(
            f"[yellow]Warning:[/yellow] {config_file} already exists. Skipping."
        )
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
