"""agent-lime CLI — placeholder until full implementation."""
import typer

app = typer.Typer(
    name="agent-lime",
    help="Continuously monitor a live AI agent.",
)


@app.callback(invoke_without_command=True)
def main() -> None:
    """agent-lime — monitor AI agents at runtime. Full implementation coming in Task 12."""
    typer.echo("agent-lime is not yet fully implemented. Run with --help for options.")
