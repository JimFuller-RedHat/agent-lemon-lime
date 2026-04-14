# AGENTS.md — Context for AI assistants

This is the canonical repo for the agent-lemon-agent-lime system.

## What this repo is

agent-lemon-lime provides two AI agents:
- **Agent Lemon**: CI/eval orchestrator — runs an agent-under-test in a sandbox,
  generates a Secure Capability Profile (SCP) YAML, and synthesises an eval report.
- **Agent Lime**: Production runtime monitor — attaches via OTEL endpoint, asserts SCP
  compliance, detects anomalies, and reports continuously.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/agent_lemon_lime/` | Main package |
| `src/agent_lemon_lime/scp/` | SCP Pydantic models and YAML serialisation |
| `src/agent_lemon_lime/harness/` | Sandbox abstraction (mock + openshell) |
| `src/agent_lemon_lime/evals/` | Evaluators, runner, skill loader |
| `src/agent_lemon_lime/report/` | Report model and Markdown synthesiser |
| `src/agent_lemon_lime/agents/` | Agent Lemon and Agent Lime |
| `src/agent_lemon_lime/cli/` | typer CLIs for lemon and lime |
| `examples/hello_world/` | Minimal test agent for end-to-end testing |
| `tests/` | Unit and integration tests |

## Conventions

- Python 3.13, uv, ruff, ty
- TDD: write failing test first, then implement
- Mock-first: tests use MockSandbox; real OpenshellSandbox requires `pytest.mark.integration`
- SCP YAML uses OpenShell-native field names (`filesystem_policy`, `landlock`, `process`, `network_policies`)
- `agent-lemon.yaml` placed in project root of agent-under-test

## Things agents often get wrong here

- Python module name uses underscores: `agent_lemon_lime` (not `agent-lemon-lime`)
- Entry points in pyproject.toml must use underscore module paths
- OpenshellSandbox requires a live cluster; always use MockSandbox in unit tests
- `SystemCapabilityProfile` is the SCP class (not `SCPProfile`)
- SCP YAML uses real OpenShell field names, not custom ones
- `SystemCapabilityProfile.to_yaml(path)` creates parent directories automatically

## Key files
- Policy: [REDHAT.md](REDHAT.md)
- Config schema: `src/agent_lemon_lime/config.py`
- SCP models: `src/agent_lemon_lime/scp/models.py`
- Agent Lemon: `src/agent_lemon_lime/agents/lemon.py`
- Agent Lime: `src/agent_lemon_lime/agents/lime.py`
