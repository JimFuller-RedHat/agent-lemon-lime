# Contributing to agent-lemon-lime

agent-lemon-lime is an AI agent evaluation and runtime monitoring library.
See [README.md](README.md) for an overview and [AGENTS.md](AGENTS.md) for
conventions relevant to AI assistants working in this repo.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [AI-Assisted Contributions](#ai-assisted-contributions)
- [Getting Help](#getting-help)

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — dependency management and virtual environments
- Git

Optional (for integration tests only):

- A live OpenShell cluster

## Development Setup

```bash
git clone <repo>
cd agent-lemon-lime
uv sync
```

This installs the package in editable mode along with all dev dependencies
(`pytest`, `ruff`, `ty`).

Verify the setup:

```bash
make check
```

## Development Workflow

### 1. Create a feature branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`.

### 2. Write the test first

This project uses TDD. Write a failing test before implementing. See
[Testing](#testing) for how tests are structured.

### 3. Implement

Follow the code standards below. Keep commits focused and atomic.

### 4. Verify

```bash
make check   # lint + typecheck + unit tests
```

All three must pass before opening a PR.

### 5. Commit

Use conventional commit messages:

```
feat: add LLM-runner evaluator
fix: handle missing agent-lemon.yaml gracefully
docs: clarify SCP field names
test: cover path traversal in LocalSandbox
refactor: extract evaluator base class
```

If an AI assistant contributed substantially, add a trailer per
[REDHAT.md](REDHAT.md):

```
Assisted-by: Claude Sonnet
```

## Code Standards

- **Python 3.13**, formatted and linted with `ruff`, type-checked with `ty`
- 100-character line length
- Google-style docstrings on non-trivial public APIs
- No relative imports — use `agent_lemon_lime.*` absolute paths
- Module name uses underscores: `agent_lemon_lime` (not `agent-lemon-lime`)

### Key conventions (things agents and humans often get wrong)

- `SystemCapabilityProfile` is the SCP class — not `SCPProfile`
- SCP YAML uses OpenShell field names: `filesystem_policy`, `landlock`,
  `process`, `network_policies` — not custom names
- `SystemCapabilityProfile.to_yaml(path)` creates parent directories automatically
- Entry points in `pyproject.toml` must use underscore module paths

### Make targets

```bash
make install    # uv sync
make lint       # ruff check
make typecheck  # ty check
make fix        # auto-fix ruff + format
make test       # unit tests (excludes integration)
make check      # lint + typecheck + test
```

## Testing

Tests live in `tests/`, mirroring the package structure under `src/`.

```bash
make test                    # unit tests only
make test-integration        # requires a live OpenShell cluster
pytest tests/path/to/test.py # single file
```

### Rules

- **Write the test first.** The failing test defines the contract before any
  implementation exists.
- **Use `MockSandbox` in unit tests.** `OpenshellSandbox` requires a live cluster
  and is only allowed in tests marked `@pytest.mark.integration`.
- **Test behaviour, not implementation.** If a refactor breaks tests but not
  behaviour, the tests were wrong.
- **Test edges and errors.** Every error path the code handles needs a test.

### End-to-end with the hello-world example

```bash
cd examples/hello_world
agent-lemon discover
```

This runs all eval cases and produces `.agent-lemon/hello-world-scp.yaml` and
`.agent-lemon/hello-world-report.md`.

## Pull Request Process

### Before submitting

1. `make check` passes (lint + typecheck + unit tests)
2. New behaviour is covered by tests
3. No sensitive data (API keys, credentials, PII) in any file or commit
4. Rebase on latest `main` to avoid merge conflicts

### PR description

Include:

- What changed and why
- Related issues (`Fixes #123`)
- How you verified the change (command output, test names)
- `Assisted-by:` or `Generated-by:` trailer if AI-assisted (see [REDHAT.md](REDHAT.md))

### Review

- All PRs require at least one approval
- Address feedback before merge; keep discussions focused
- PRs are squash-merged to `main`

## AI-Assisted Contributions

This project is developed with AI assistance. Follow [REDHAT.md](REDHAT.md) for
Red Hat's attribution and responsible-use policy. Key points:

- Do not include confidential, personal, or proprietary data in prompts
- Mark AI-assisted commits with `Assisted-by:` or `Generated-by:` trailers
- Review all AI output before integrating — you are responsible for what you commit
- Check target project policies before submitting AI-assisted patches upstream

## Getting Help

- [AGENTS.md](AGENTS.md) — architecture, conventions, and common pitfalls
- [README.md](README.md) — full CLI reference and configuration schema
- [REDHAT.md](REDHAT.md) — AI-assisted development policy
- Open an issue with reproduction steps and relevant logs
