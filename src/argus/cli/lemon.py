"""agent-lemon CLI — placeholder until full implementation."""
import typer

app = typer.Typer(
    name="agent-lemon",
    help="Evaluate AI agents for safety, security, stability, and correctness.",
)


@app.callback(invoke_without_command=True)
def main() -> None:
    """agent-lemon — evaluate AI agents. Full implementation coming in Task 12."""
    typer.echo("agent-lemon is not yet fully implemented. Run with --help for options.")
