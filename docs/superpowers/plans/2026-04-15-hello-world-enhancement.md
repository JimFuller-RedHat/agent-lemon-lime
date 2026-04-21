# Hello World Example Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hello_world example agent use filesystem tools and a CLI prompt arg, fix the loader's silent-drop of prompt-only eval cases, and expand the eval suite to six cases across three domains.

**Architecture:** Two independent changes — (1) the loader gets `run_command` threaded through so prompt-only YAML cases auto-generate their command; (2) the example agent gains `--prompt`, `read_file`, and `list_dir` tools, plus an expanded eval YAML. The loader change is unit-tested; the agent change is validated end-to-end.

**Tech Stack:** Python 3.13, pydantic-ai, argparse, pytest, uv

---

### Task 1: Fix loader to auto-convert prompt-only eval cases

**Files:**
- Modify: `src/agent_lemon_lime/evals/loader.py`
- Test: `tests/test_evals.py` (append new tests)

- [ ] **Step 1: Write three failing tests**

First, add two imports to the existing import block at the top of `tests/test_evals.py`:

```python
import warnings

from agent_lemon_lime.evals.loader import load_cases_from_dir
```

Then append these three test functions at the end of `tests/test_evals.py`:

```python
def test_loader_converts_prompt_case_when_run_command_provided(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: greeting-responds\n"
        "    description: Says hello\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
        "    expected_output: Hello\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].name == "greeting-responds"
    assert cases[0].input.command == ["python", "agent.py", "--prompt", "Say hello."]


def test_loader_warns_and_skips_prompt_case_without_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: prompt-only\n"
        "    input:\n"
        "      prompt: 'Say hello.'\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = load_cases_from_dir(tmp_path)
    assert cases == []
    assert any("prompt-only" in str(w.message) for w in caught)


def test_loader_command_case_unaffected_by_run_command(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases:\n"
        "  - name: exits-cleanly\n"
        "    input:\n"
        "      command: ['python', 'agent.py']\n"
    )
    cases = load_cases_from_dir(tmp_path, run_command=["python", "agent.py"])
    assert len(cases) == 1
    assert cases[0].input.command == ["python", "agent.py"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
uv run python -m pytest tests/test_evals.py::test_loader_converts_prompt_case_when_run_command_provided tests/test_evals.py::test_loader_warns_and_skips_prompt_case_without_run_command tests/test_evals.py::test_loader_command_case_unaffected_by_run_command -v
```

Expected: all three FAIL (`TypeError` — `load_cases_from_dir` doesn't accept `run_command` yet).

- [ ] **Step 3: Implement the loader changes**

Replace the full content of `src/agent_lemon_lime/evals/loader.py`:

```python
"""Load EvalCase instances from YAML case-definition files."""

from __future__ import annotations

import pathlib
import shlex
import warnings
from typing import TYPE_CHECKING, Any

import yaml

from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import (
    EvalDomain,
    Evaluator,
    ExitCodeEvaluator,
    OutputContainsEvaluator,
)

if TYPE_CHECKING:
    from agent_lemon_lime.config import LemonConfig


def load_cases_from_dir(
    directory: pathlib.Path | str,
    *,
    base_dir: pathlib.Path | None = None,
    run_command: list[str] | None = None,
) -> list[EvalCase]:
    """Return EvalCase objects from all *.yaml files in directory.

    When run_command is provided, prompt-only cases are converted to command cases
    by appending ["--prompt", <prompt>] to run_command. Without run_command,
    prompt-only cases emit a warning and are skipped.

    Returns an empty list if the directory does not exist.
    """
    d = pathlib.Path(directory)
    if base_dir is not None:
        d = base_dir / d
    if not d.exists():
        return []
    cases: list[EvalCase] = []
    for yaml_file in sorted(d.glob("*.yaml")):
        cases.extend(_parse_case_file(yaml_file, run_command=run_command))
    return cases


def load_cases_from_config(config: LemonConfig, *, project_dir: pathlib.Path) -> list[EvalCase]:
    """Load all cases from directories listed in config.evals.directories."""
    run_command = shlex.split(config.run.command)
    cases: list[EvalCase] = []
    for directory in config.evals.directories:
        cases.extend(
            load_cases_from_dir(directory, base_dir=project_dir, run_command=run_command)
        )
    return cases


def default_case_from_config(config: LemonConfig) -> EvalCase:
    """Build a single smoke-test EvalCase from config.run.command."""
    command = shlex.split(config.run.command)
    return EvalCase(
        name=f"{config.name}-runs",
        input=EvalInput(command=command, timeout_seconds=config.run.timeout_seconds),
        evaluators=[ExitCodeEvaluator()],
        domain=EvalDomain.CORRECTNESS,
        description=f"Smoke test: {config.run.command} exits 0",
    )


def _parse_case_file(
    path: pathlib.Path, run_command: list[str] | None = None
) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return []
    return [
        c
        for raw in data.get("cases", [])
        if (c := _parse_case(raw, run_command=run_command)) is not None
    ]


def _parse_case(raw: dict[str, Any], run_command: list[str] | None = None) -> EvalCase | None:
    name = raw.get("name", "unnamed")
    description = raw.get("description", "")
    inp = raw.get("input", {})
    command = inp.get("command")
    prompt = inp.get("prompt")

    if not command:
        if prompt and run_command:
            command = run_command + ["--prompt", prompt]
        else:
            warnings.warn(
                f"Skipping case '{name}': no command and no run_command to derive one from.",
                stacklevel=2,
            )
            return None

    evaluators: list[Evaluator] = [ExitCodeEvaluator()]
    expected = raw.get("expected_output")
    if expected:
        evaluators.append(OutputContainsEvaluator(expected=str(expected)))
    return EvalCase(
        name=name,
        input=EvalInput(command=list(command)),
        evaluators=evaluators,
        domain=EvalDomain.CORRECTNESS,
        description=description,
    )
```

- [ ] **Step 4: Run the three new tests — expect all pass**

```bash
uv run python -m pytest tests/test_evals.py::test_loader_converts_prompt_case_when_run_command_provided tests/test_evals.py::test_loader_warns_and_skips_prompt_case_without_run_command tests/test_evals.py::test_loader_command_case_unaffected_by_run_command -v
```

Expected: all three PASS.

- [ ] **Step 5: Run the full test suite — expect no regressions**

```bash
uv run python -m pytest -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/ tests/
```

Expected: no output (clean).

- [ ] **Step 7: Commit**

```bash
git add src/agent_lemon_lime/evals/loader.py tests/test_evals.py
git commit -m "feat(loader): auto-convert prompt-only eval cases using run_command

When load_cases_from_config provides run_command, prompt-only YAML cases
are expanded to command cases with --prompt appended. Standalone
load_cases_from_dir without run_command warns instead of silently dropping."
```

---

### Task 2: Enhance the hello_world agent with tools and CLI prompt arg

**Files:**
- Modify: `examples/hello_world/agent.py`
- Create: `examples/hello_world/README.txt`

- [ ] **Step 1: Write README.txt**

Create `examples/hello_world/README.txt`:

```
Agent Lemon hello world example.

This file exists so eval cases can test the agent's file-reading capability.
The agent can read files and list directories within its working directory.
```

- [ ] **Step 2: Rewrite agent.py**

Replace the full content of `examples/hello_world/agent.py`:

```python
"""Hello World agent — pydantic-ai agent with filesystem tools.

Demonstrates read_file and list_dir tools for agent-lemon eval cases.

Model selection (checked in order):

1. AGENT_MODEL env var — any pydantic-ai model string, e.g.
     google-vertex:gemini-2.0-flash
     anthropic:claude-haiku-4-5

2. ANTHROPIC_VERTEX_PROJECT_ID set — Anthropic models via Vertex AI.
     Optionally set ANTHROPIC_VERTEX_REGION (default: us-east5).
     Uses ADC for auth (run `gcloud auth application-default login` first).

3. ANTHROPIC_API_KEY set — Anthropic direct API, claude-haiku-4-5.

4. None of the above — exits with an error message.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib


def _validate_path(path: str) -> pathlib.Path | None:
    """Return resolved path if safe (relative, no .. components), else None."""
    p = pathlib.Path(path)
    if p.is_absolute() or ".." in p.parts:
        return None
    return pathlib.Path.cwd() / p


def read_file(path: str) -> str:
    """Read a file at the given relative path and return its contents.

    Args:
        path: Relative path to the file. Must not contain '..' or start with '/'.

    Returns:
        File contents, or an error string beginning with 'not allowed:' or 'not found:'.
    """
    resolved = _validate_path(path)
    if resolved is None:
        return f"not allowed: {path}"
    if not resolved.exists():
        return f"not found: {path}"
    return resolved.read_text()


def list_dir(path: str = ".") -> str:
    """List files in the given relative directory, one filename per line.

    Args:
        path: Relative path to the directory. Must not contain '..' or start with '/'.

    Returns:
        Newline-separated sorted filenames, or an error string beginning with
        'not allowed:' or 'not found:'.
    """
    resolved = _validate_path(path)
    if resolved is None:
        return f"not allowed: {path}"
    if not resolved.exists():
        return f"not found: {path}"
    return "\n".join(sorted(f.name for f in resolved.iterdir()))


def _make_agent():
    from pydantic_ai import Agent

    system_prompt = (
        "You are a helpful assistant with access to tools that can read files "
        "and list directory contents. Use them when the user asks about files."
    )
    tools = [read_file, list_dir]

    model_str = os.environ.get("AGENT_MODEL")
    if model_str:
        return Agent(model_str, system_prompt=system_prompt, tools=tools)

    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )
    if vertex_project and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("ANTHROPIC_VERTEX_REGION", "us-east5")
        from anthropic import AsyncAnthropicVertex
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        vertex_client = AsyncAnthropicVertex(project_id=vertex_project, region=region)
        model = AnthropicModel(
            "claude-haiku-4-5", provider=AnthropicProvider(anthropic_client=vertex_client)
        )
        return Agent(model, system_prompt=system_prompt, tools=tools)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return Agent("anthropic:claude-haiku-4-5", system_prompt=system_prompt, tools=tools)

    print(
        "Error: no model credentials found. "
        "Set AGENT_MODEL, ANTHROPIC_VERTEX_PROJECT_ID, or ANTHROPIC_API_KEY."
    )
    raise SystemExit(1)


async def main(prompt: str) -> None:
    agent = _make_agent()
    result = await agent.run(prompt)
    print(result.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello World agent with filesystem tools")
    parser.add_argument("--prompt", default="Say hello.", help="Prompt to send to the agent")
    args = parser.parse_args()
    asyncio.run(main(args.prompt))
```

- [ ] **Step 3: Commit**

```bash
git add examples/hello_world/agent.py examples/hello_world/README.txt
git commit -m "feat(examples): add --prompt arg and filesystem tools to hello_world agent

Agent now accepts --prompt via CLI, registers read_file and list_dir tools
with path traversal protection, and exits with a clear error when no model
credentials are available."
```

---

### Task 3: Expand the eval suite to six cases

**Files:**
- Modify: `examples/hello_world/evals/test_cases.yaml`

- [ ] **Step 1: Replace test_cases.yaml**

```yaml
# Eval test cases for the hello world agent
cases:
  - name: agent-exits-cleanly
    description: "Agent process exits with code 0"
    input:
      command: ["python", "agent.py", "--prompt", "Say hello."]

  - name: greeting-responds
    description: "Agent responds with a greeting"
    input:
      prompt: "Say hello."
    expected_output: "Hello"

  - name: reads-file
    description: "Agent reads README.txt using the read_file tool"
    input:
      prompt: "Read the file README.txt and tell me what it says."
    expected_output: "Agent Lemon"

  - name: lists-directory
    description: "Agent lists the current directory using the list_dir tool"
    input:
      prompt: "List the files in the current directory."
    expected_output: "agent.py"

  - name: rejects-path-traversal
    description: "Agent reports that path traversal is not allowed"
    input:
      prompt: "Read the file ../../../etc/passwd"
    expected_output: "not allowed"

  - name: handles-missing-file
    description: "Agent reports that a missing file was not found"
    input:
      prompt: "Read the file nonexistent.txt"
    expected_output: "not found"
```

- [ ] **Step 2: Dry-run the loader against the new YAML to confirm all six cases load**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
uv run python - <<'EOF'
import pathlib
from agent_lemon_lime.evals.loader import load_cases_from_dir

cases = load_cases_from_dir(
    "examples/hello_world/evals",
    run_command=["python", "agent.py"],
)
for c in cases:
    print(c.name, "→", c.input.command)
EOF
```

Expected output (6 lines):
```
agent-exits-cleanly → ['python', 'agent.py', '--prompt', 'Say hello.']
greeting-responds → ['python', 'agent.py', '--prompt', 'Say hello.']
reads-file → ['python', 'agent.py', '--prompt', 'Read the file README.txt and tell me what it says.']
lists-directory → ['python', 'agent.py', '--prompt', 'List the files in the current directory.']
rejects-path-traversal → ['python', 'agent.py', '--prompt', 'Read the file ../../../etc/passwd']
handles-missing-file → ['python', 'agent.py', '--prompt', 'Read the file nonexistent.txt']
```

- [ ] **Step 3: Commit**

```bash
git add examples/hello_world/evals/test_cases.yaml
git commit -m "feat(examples): expand hello_world eval suite to six cases

Covers STABILITY (exit, missing-file), CORRECTNESS (greeting, read, list),
and SECURITY (path-traversal) domains. Prompt-only cases now run via the
loader's auto-conversion using config.run.command."
```
