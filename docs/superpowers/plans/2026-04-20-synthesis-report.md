# Synthesis Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional LLM-powered analysis section to Agent Lemon's markdown report that surfaces anomalies, remediation suggestions, and risk assessment.

**Architecture:** A new `analyzer.py` module in the `report/` package makes direct httpx calls to either the Anthropic Messages API or OpenAI Chat Completions API, building context from all eval data and inserting the analysis into the existing markdown report. Activation is config-driven via `report.model` in `agent-lemon.yaml`. All errors are best-effort — analysis failures never block the eval run.

**Tech Stack:** Python 3.13, httpx (existing dependency), pydantic (existing)

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `src/agent_lemon_lime/report/analyzer.py` | `analyze_report()`, `_build_context()`, `_call_anthropic()`, `_call_openai()`, `insert_analysis()` |
| **Create:** `tests/test_analyzer.py` | Unit tests for all analyzer functions |
| **Modify:** `src/agent_lemon_lime/config.py:56-59` | Add `model: str \| None = None` to `ReportConfig` |
| **Modify:** `src/agent_lemon_lime/cli/lemon.py:490-498` | Wire analysis into `discover` command report output |
| **Modify:** `src/agent_lemon_lime/cli/lemon.py:657-662` | Wire analysis into `assert` command report output |

---

### Task 1: Add `model` field to `ReportConfig`

**Files:**
- Modify: `src/agent_lemon_lime/config.py:56-59`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_report_config_model_field():
    yaml_text = """\
name: test-agent
version: "0.1.0"
run:
  command: "python agent.py"
report:
  output: ".agent-lemon/report.md"
  model: anthropic/claude-sonnet-4-20250514
"""
    config = LemonConfig.from_yaml(yaml_text)
    assert config.report.model == "anthropic/claude-sonnet-4-20250514"


def test_report_config_model_defaults_to_none():
    yaml_text = """\
name: test-agent
version: "0.1.0"
run:
  command: "python agent.py"
"""
    config = LemonConfig.from_yaml(yaml_text)
    assert config.report.model is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_config.py::test_report_config_model_field -v`
Expected: FAIL — `model` field doesn't exist on `ReportConfig`, pydantic validation error

- [ ] **Step 3: Add `model` field to `ReportConfig`**

In `src/agent_lemon_lime/config.py`, change `ReportConfig`:

```python
class ReportConfig(BaseModel):
    output: str = ".agent-lemon/report.md"
    log: str | None = None  # defaults to .agent-lemon/{agent-name}.log
    format: Literal["markdown", "json"] = "markdown"
    model: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
git add src/agent_lemon_lime/config.py tests/test_config.py
git commit -m "feat: add model field to ReportConfig for synthesis analysis"
```

---

### Task 2: Implement `_build_context()` and `insert_analysis()`

**Files:**
- Create: `src/agent_lemon_lime/report/analyzer.py`
- Create: `tests/test_analyzer.py`

These are pure string-manipulation functions with no API calls — easy to test in isolation.

- [ ] **Step 1: Write failing tests for `_build_context()`**

Create `tests/test_analyzer.py`:

```python
"""Tests for LLM-powered report analyzer."""

from agent_lemon_lime.evals.runner import EvalResult
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.report.analyzer import _build_context, insert_analysis
from agent_lemon_lime.report.models import EvalReport, EvalSummary, InferenceConfig
from agent_lemon_lime.scp.models import SystemCapabilityProfile


def _make_result(name: str, passed: bool, domain: EvalDomain = EvalDomain.CORRECTNESS) -> EvalResult:
    return EvalResult(
        name=name,
        passed=passed,
        domain=domain,
        output=EvalOutput(
            exit_code=0 if passed else 1,
            stdout="some output" if passed else "error output",
            stderr="" if passed else "something failed",
            domain=domain,
        ),
        failures=[] if passed else ["ExitCodeEvaluator"],
    )


def _make_report(
    results: list[EvalResult] | None = None,
    violations: list[str] | None = None,
) -> EvalReport:
    if results is None:
        results = [
            _make_result("test-pass", True),
            _make_result("test-fail", False),
        ]
    passed = sum(1 for r in results if r.passed)
    return EvalReport(
        agent_name="test-agent",
        generated_at="2026-04-20T14:30:00+00:00",
        summary=EvalSummary(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
        ),
        results=results,
        scp=SystemCapabilityProfile(),
        violations=violations or [],
        inference=InferenceConfig(),
    )


def test_build_context_includes_agent_name():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "test-agent" in ctx


def test_build_context_includes_summary():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "Total: 2" in ctx
    assert "Passed: 1" in ctx
    assert "Failed: 1" in ctx


def test_build_context_includes_result_details():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "test-pass" in ctx
    assert "test-fail" in ctx
    assert "PASS" in ctx
    assert "FAIL" in ctx


def test_build_context_includes_failures():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "ExitCodeEvaluator" in ctx


def test_build_context_includes_violations():
    report = _make_report(violations=["Unauthorized host 'evil.com'"])
    ctx = _build_context(report, config_yaml="")
    assert "Unauthorized host 'evil.com'" in ctx


def test_build_context_includes_config_yaml():
    report = _make_report()
    config_yaml = "name: my-agent\nrun:\n  command: python agent.py\n"
    ctx = _build_context(report, config_yaml=config_yaml)
    assert "name: my-agent" in ctx


def test_build_context_empty_results():
    report = _make_report(results=[])
    ctx = _build_context(report, config_yaml="")
    assert "Total: 0" in ctx


def test_build_context_no_config_yaml():
    report = _make_report()
    ctx = _build_context(report, config_yaml="")
    assert "Agent Config" not in ctx


def test_build_context_truncates_long_output():
    long_stdout = "x" * 5000
    result = EvalResult(
        name="verbose-test",
        passed=True,
        domain=EvalDomain.CORRECTNESS,
        output=EvalOutput(
            exit_code=0,
            stdout=long_stdout,
            stderr="",
            domain=EvalDomain.CORRECTNESS,
        ),
    )
    report = _make_report(results=[result])
    ctx = _build_context(report, config_yaml="")
    assert len(ctx) < len(long_stdout)
    assert "[truncated]" in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -v`
Expected: FAIL — `analyzer` module does not exist

- [ ] **Step 3: Implement `_build_context()`**

Create `src/agent_lemon_lime/report/analyzer.py`:

```python
"""LLM-powered report analysis."""

from __future__ import annotations

from agent_lemon_lime.report.models import EvalReport

_MAX_OUTPUT_CHARS = 2000


def _build_context(report: EvalReport, *, config_yaml: str = "") -> str:
    s = report.summary
    lines = [
        "=== Agent Eval Results ===",
        f"Agent: {report.agent_name}",
        f"Generated: {report.generated_at}",
        "",
        "--- Summary ---",
        f"Total: {s.total}",
        f"Passed: {s.passed}",
        f"Failed: {s.failed}",
        f"Pass Rate: {s.pass_rate:.1%}",
        "",
        "--- Results ---",
    ]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.domain}::{r.name}")
        if r.failures:
            lines.append(f"  Failures: {', '.join(r.failures)}")
        stdout = r.output.stdout.strip()
        if stdout:
            if len(stdout) > _MAX_OUTPUT_CHARS:
                stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n[truncated]"
            lines.append(f"  stdout: {stdout}")
        stderr = r.output.stderr.strip()
        if stderr:
            if len(stderr) > _MAX_OUTPUT_CHARS:
                stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n[truncated]"
            lines.append(f"  stderr: {stderr}")
    lines.append("")

    if report.violations:
        lines.append("--- SCP Violations ---")
        for v in report.violations:
            lines.append(f"  - {v}")
        lines.append("")

    if config_yaml.strip():
        lines.append("--- Agent Config ---")
        lines.append(config_yaml.strip())
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run `_build_context` tests to verify they pass**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "build_context" -v`
Expected: ALL PASS

- [ ] **Step 5: Write failing tests for `insert_analysis()`**

Add to `tests/test_analyzer.py`:

```python
def test_insert_analysis_before_summary():
    markdown = (
        "# Agent Lemon Report\n"
        "\n"
        "**Generated:** 2026-04-20T14:30:00+00:00\n"
        "**Sandbox:** local\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Value |\n"
    )
    analysis = "## Analysis\n\n### Executive Summary\nAll good.\n"
    result = insert_analysis(markdown, analysis)
    lines = result.split("\n")
    summary_idx = next(i for i, l in enumerate(lines) if l == "## Summary")
    analysis_idx = next(i for i, l in enumerate(lines) if l == "## Analysis")
    assert analysis_idx < summary_idx


def test_insert_analysis_none_returns_unchanged():
    markdown = "# Agent Lemon Report\n\n## Summary\n"
    result = insert_analysis(markdown, None)
    assert result == markdown


def test_insert_analysis_empty_string_returns_unchanged():
    markdown = "# Agent Lemon Report\n\n## Summary\n"
    result = insert_analysis(markdown, "")
    assert result == markdown
```

- [ ] **Step 6: Run insert_analysis tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "insert_analysis" -v`
Expected: FAIL — `insert_analysis` not yet defined

- [ ] **Step 7: Implement `insert_analysis()`**

Add to `src/agent_lemon_lime/report/analyzer.py`:

```python
def insert_analysis(markdown: str, analysis: str | None) -> str:
    if not analysis:
        return markdown
    marker = "## Summary"
    idx = markdown.find(marker)
    if idx == -1:
        return markdown
    return markdown[:idx] + analysis.rstrip() + "\n\n" + markdown[idx:]
```

- [ ] **Step 8: Run all Task 2 tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
git add src/agent_lemon_lime/report/analyzer.py tests/test_analyzer.py
git commit -m "feat: add _build_context() and insert_analysis() for synthesis report"
```

---

### Task 3: Implement `_call_anthropic()` and `_call_openai()`

**Files:**
- Modify: `src/agent_lemon_lime/report/analyzer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests for `_call_anthropic()`**

Add to `tests/test_analyzer.py`:

```python
import httpx
from unittest.mock import patch, MagicMock

from agent_lemon_lime.report.analyzer import _call_anthropic, _call_openai


def test_call_anthropic_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": "## Analysis\n\nAll good."}],
        },
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = _call_anthropic("claude-sonnet-4-20250514", "eval context here")
    assert result == "## Analysis\n\nAll good."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "anthropic-version" in call_kwargs.kwargs["headers"]
    assert call_kwargs.kwargs["headers"]["x-api-key"] == "sk-test-key"


def test_call_anthropic_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _call_anthropic("claude-sonnet-4-20250514", "eval context")
    assert result is None


def test_call_anthropic_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        500,
        json={"error": {"message": "Internal Server Error"}},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = _call_anthropic("claude-sonnet-4-20250514", "eval context")
    assert result is None


def test_call_anthropic_empty_content(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": ""}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = _call_anthropic("claude-sonnet-4-20250514", "eval context")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "call_anthropic" -v`
Expected: FAIL — `_call_anthropic` not defined

- [ ] **Step 3: Implement `_call_anthropic()`**

Add to `src/agent_lemon_lime/report/analyzer.py`:

```python
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
You are an AI agent evaluation analyst. Analyze the following eval results \
and produce a report with these sections:

## Analysis

### Executive Summary
2-3 sentences on overall agent health and key findings.

### Anomalies
Unexpected patterns, correlated failures, or regressions. If none found, say so.

### Remediation
Concrete, prioritized suggestions. Reference specific eval names and failure messages.

### Risk Assessment
Overall risk: **HIGH**, **MEDIUM**, or **LOW**
Justification in 1-2 sentences."""


def _call_anthropic(model: str, context: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping analysis")
        return None
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "system": ANALYSIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": context}],
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.warning("Anthropic API returned %d — skipping analysis", resp.status_code)
            return None
        data = resp.json()
        text = data["content"][0]["text"]
        return text if text.strip() else None
    except Exception:
        logger.warning("Anthropic API call failed — skipping analysis", exc_info=True)
        return None
```

- [ ] **Step 4: Run `_call_anthropic` tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "call_anthropic" -v`
Expected: ALL PASS

- [ ] **Step 5: Write failing tests for `_call_openai()`**

Add to `tests/test_analyzer.py`:

```python
def test_call_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "## Analysis\n\nLooks fine."}}
            ],
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        result = _call_openai("gpt-4o", "eval context here")
    assert result == "## Analysis\n\nLooks fine."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "Bearer sk-test-key" in call_kwargs.kwargs["headers"]["Authorization"]


def test_call_openai_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _call_openai("gpt-4o", "eval context")
    assert result is None


def test_call_openai_api_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        500,
        json={"error": {"message": "Server Error"}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = _call_openai("gpt-4o", "eval context")
    assert result is None


def test_call_openai_empty_content(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = _call_openai("gpt-4o", "eval context")
    assert result is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "call_openai" -v`
Expected: FAIL — `_call_openai` not defined

- [ ] **Step 7: Implement `_call_openai()`**

Add to `src/agent_lemon_lime/report/analyzer.py`:

```python
def _call_openai(model: str, context: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping analysis")
        return None
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            },
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.warning("OpenAI API returned %d — skipping analysis", resp.status_code)
            return None
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text if text.strip() else None
    except Exception:
        logger.warning("OpenAI API call failed — skipping analysis", exc_info=True)
        return None
```

- [ ] **Step 8: Run all Task 3 tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "call_" -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
git add src/agent_lemon_lime/report/analyzer.py tests/test_analyzer.py
git commit -m "feat: add _call_anthropic() and _call_openai() for synthesis report"
```

---

### Task 4: Implement `analyze_report()` orchestrator

**Files:**
- Modify: `src/agent_lemon_lime/report/analyzer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_analyzer.py`:

```python
from agent_lemon_lime.report.analyzer import analyze_report


def test_analyze_report_anthropic_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    report = _make_report()
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "## Analysis\n\nGood."}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = analyze_report(report, model="anthropic/claude-sonnet-4-20250514")
    assert result == "## Analysis\n\nGood."


def test_analyze_report_openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    report = _make_report()
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "## Analysis\n\nFine."}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.post", return_value=mock_response):
        result = analyze_report(report, model="openai/gpt-4o")
    assert result == "## Analysis\n\nFine."


def test_analyze_report_unknown_provider():
    report = _make_report()
    result = analyze_report(report, model="ollama/llama3")
    assert result is None


def test_analyze_report_no_slash_in_model():
    report = _make_report()
    result = analyze_report(report, model="claude-sonnet")
    assert result is None


def test_analyze_report_passes_config_yaml(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    report = _make_report()
    mock_response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "## Analysis\n\nOK."}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    with patch("httpx.post", return_value=mock_response) as mock_post:
        analyze_report(
            report,
            model="anthropic/claude-sonnet-4-20250514",
            config_yaml="name: test\n",
        )
    call_body = mock_post.call_args.kwargs["json"]
    user_msg = call_body["messages"][0]["content"]
    assert "name: test" in user_msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "analyze_report" -v`
Expected: FAIL — `analyze_report` not defined

- [ ] **Step 3: Implement `analyze_report()`**

Add to `src/agent_lemon_lime/report/analyzer.py`:

```python
def analyze_report(
    report: EvalReport,
    *,
    model: str,
    config_yaml: str = "",
) -> str | None:
    parts = model.split("/", 1)
    if len(parts) != 2:
        logger.warning("Invalid model format '%s' — expected 'provider/model'", model)
        return None
    provider, model_name = parts
    context = _build_context(report, config_yaml=config_yaml)
    if provider == "anthropic":
        return _call_anthropic(model_name, context)
    if provider == "openai":
        return _call_openai(model_name, context)
    logger.warning("Unknown provider '%s' — skipping analysis", provider)
    return None
```

- [ ] **Step 4: Run all Task 4 tests**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -k "analyze_report" -v`
Expected: ALL PASS

- [ ] **Step 5: Run the full test suite**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest tests/test_analyzer.py -v`
Expected: ALL PASS (all tests from Tasks 2, 3, and 4)

- [ ] **Step 6: Commit**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
git add src/agent_lemon_lime/report/analyzer.py tests/test_analyzer.py
git commit -m "feat: add analyze_report() orchestrator for synthesis report"
```

---

### Task 5: Wire analysis into CLI commands

**Files:**
- Modify: `src/agent_lemon_lime/cli/lemon.py:490-498` (discover command)
- Modify: `src/agent_lemon_lime/cli/lemon.py:657-662` (assert command)

The integration point in both commands replaces the direct `synth.write()` call with: render markdown → optionally run analysis → insert analysis → write to disk.

- [ ] **Step 1: Modify `discover` command**

In `src/agent_lemon_lime/cli/lemon.py`, find the report-writing block in `discover()` (around lines 493-498). Replace:

```python
    synthesizer = ReportSynthesizer()
    report_path = report or config.report.output
    synthesizer.write(result.report, path=report_path)
```

With:

```python
    synthesizer = ReportSynthesizer()
    report_path = report or config.report.output
    md = synthesizer.to_markdown(result.report)

    if config.report.model:
        from agent_lemon_lime.report.analyzer import analyze_report, insert_analysis

        config_yaml = pathlib.Path(project_dir).resolve().joinpath("agent-lemon.yaml")
        config_text = config_yaml.read_text() if config_yaml.exists() else ""
        console.print("[dim]Running LLM analysis...[/dim]")
        analysis = analyze_report(
            result.report,
            model=config.report.model,
            config_yaml=config_text,
        )
        if analysis:
            md = insert_analysis(md, analysis)
            console.print("[green]Analysis complete.[/green]")
        else:
            console.print("[yellow]Analysis skipped (see warnings above).[/yellow]")

    p = pathlib.Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
```

- [ ] **Step 2: Modify `assert` command**

In `src/agent_lemon_lime/cli/lemon.py`, find the report-writing block in `assert_cmd()` (around lines 657-659). Replace:

```python
    report_path = config.report.output
    synth = ReportSynthesizer()
    synth.write(result.report, path=report_path)
```

With:

```python
    report_path = config.report.output
    synth = ReportSynthesizer()
    md = synth.to_markdown(result.report)

    if config.report.model:
        from agent_lemon_lime.report.analyzer import analyze_report, insert_analysis

        config_yaml = pathlib.Path(project_dir).resolve().joinpath("agent-lemon.yaml")
        config_text = config_yaml.read_text() if config_yaml.exists() else ""
        console.print("[dim]Running LLM analysis...[/dim]")
        analysis = analyze_report(
            result.report,
            model=config.report.model,
            config_yaml=config_text,
        )
        if analysis:
            md = insert_analysis(md, analysis)
            console.print("[green]Analysis complete.[/green]")
        else:
            console.print("[yellow]Analysis skipped (see warnings above).[/yellow]")

    p = pathlib.Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run pytest -v`
Expected: ALL PASS — existing CLI tests should still pass since `report.model` defaults to `None`

- [ ] **Step 4: Run linter and type checker**

Run: `cd /home/jfuller/src/argus/agent-lemon-lime && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
cd /home/jfuller/src/argus/agent-lemon-lime
git add src/agent_lemon_lime/cli/lemon.py
git commit -m "feat: wire synthesis analysis into discover and assert commands"
```
