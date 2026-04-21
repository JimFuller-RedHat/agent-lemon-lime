# AGENTS.md — Context for AI assistants

This is the canonical repo for the agent-lemon-agent-lime system.

## What this repo is

agent-lemon-lime provides two AI agents with complementary but distinct roles:

- **Agent Lemon**: Execution-based CI evaluator. Runs an agent-under-test as a subprocess
  with specific inputs, collects outputs, and answers: *"Does the agent produce the right
  outputs for given inputs?"* Generates a Secure Capability Profile (SCP) YAML and a report.
  Eval cases are custom to each agent under test (defined in `agent-lemon.yaml`).

- **Agent Lime**: Observation-based runtime monitor. Attaches passively via OTEL telemetry
  to any running agent workload (tests, interactions, production traffic) without caring how
  the agent is invoked. Answers: *"What did the agent actually do, and was it within bounds?"*
  Uses a session-bracket model: start Lime → let the agent do real work → stop Lime → report
  and assert SCP compliance. Evals are generic and universal (no custom entry point required).

**Key distinction:** Lemon is active (it runs the agent). Lime is passive (it observes the
agent). Lemon evals are custom per-agent. Lime evals are generic across all agents.

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
- Config schema: `src/agent_lemon_lime/config.py`
- SCP models: `src/agent_lemon_lime/scp/models.py`
- Agent Lemon: `src/agent_lemon_lime/agents/lemon.py`
- Agent Lime: `src/agent_lemon_lime/agents/lime.py`
