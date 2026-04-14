"""agent-lime CLI entry point."""

from __future__ import annotations

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_lemon_lime.agents.lime import LimeEvent

app = typer.Typer(
    name="agent-lime",
    help="Monitor a running AI agent for SCP compliance and anomalies.",
)
console = Console()


@app.callback()
def _callback() -> None:
    """agent-lime: monitor a running AI agent for SCP compliance and anomalies."""


@app.command()
def monitor(
    otel: Annotated[str, typer.Option("--otel", help="OTEL collector endpoint")] = "",
    scp: Annotated[str | None, typer.Option("--scp", help="SCP YAML to assert against")] = None,
    interval: Annotated[
        float, typer.Option("--interval", "-i", help="Poll interval in seconds")
    ] = 30.0,
    once: Annotated[bool, typer.Option("--once", help="Run one analysis pass then exit")] = False,
) -> None:
    """Attach to a running agent via OTEL and monitor SCP compliance."""
    if not scp:
        console.print("[red]Error:[/red] --scp <path-to-scp.yaml> is required.")
        raise typer.Exit(code=1)
    if not otel:
        console.print("[red]Error:[/red] --otel <endpoint> is required.")
        raise typer.Exit(code=1)

    from agent_lemon_lime.agents.lime import LimeAgent
    from agent_lemon_lime.scp.models import SystemCapabilityProfile

    try:
        assert_scp = SystemCapabilityProfile.from_yaml(scp)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] SCP file not found: {scp}")
        raise typer.Exit(code=1) from None

    lime = LimeAgent(otel_endpoint=otel, assert_scp=assert_scp, poll_interval_seconds=interval)

    console.print(Panel(
        f"[bold]Agent Lime — Monitor Mode[/bold]\nOTEL: {otel}\nSCP: {scp}\nInterval: {interval}s"
    ))

    iteration = 0
    while True:
        iteration += 1
        events = lime.collect_events_from_otel()
        anomalies = lime.analyse_events(events)
        _print_status(iteration, events, anomalies)
        if once:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
            raise typer.Exit(code=0) from None


def _print_status(iteration: int, events: list[LimeEvent], anomalies: list[str]) -> None:
    table = Table(title=f"Lime Check #{iteration}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Events", str(len(events)))
    anomaly_val = f"[red]{len(anomalies)}[/red]" if anomalies else "0"
    table.add_row("Anomalies", anomaly_val)
    console.print(table)
    for a in anomalies:
        console.print(f"  ⚠  {a}")
