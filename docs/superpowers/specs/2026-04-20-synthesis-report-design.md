# Synthesis Report — LLM-Powered Analysis

**Date:** 2026-04-20
**Status:** Approved

## Problem

Agent Lemon's report is purely mechanical: a pass/fail table and SCP violation list. Users must manually interpret patterns across results — correlating failures, identifying root causes, and deciding what to fix first. There is no automated analysis of what the results mean or what the user should do about them.

## Solution

Add an LLM-powered analysis section to Agent Lemon's markdown report. After the eval run completes and the report is built, an optional synthesis step sends all available context (eval results, SCP, violations, backend scores, log output, agent config) to an LLM and inserts its analysis into the report. The analysis covers an executive summary, anomaly detection, remediation suggestions, and risk assessment.

## Design Decisions

1. **Model config** — New `report.model` field in `ReportConfig` (e.g. `anthropic/claude-sonnet-4-20250514`). Uses host credentials (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`), not sandbox inference.

2. **Output location** — Analysis section inserted into the existing markdown report, after the header block and before "## Summary". Not a separate file.

3. **Inputs to LLM** — All available data: eval results (pass/fail, domain, failures, stdout/stderr), SCP profile, SCP violations, backend scores, log output, and the agent's `agent-lemon.yaml` config.

4. **Activation** — Always runs if `report.model` is configured. No extra CLI flag needed. Skipped silently if `report.model` is not set.

5. **LLM call approach** — Direct `httpx` API calls (already a dependency). Supports two provider prefixes:
   - `anthropic/` → Anthropic Messages API + `ANTHROPIC_API_KEY`
   - `openai/` → OpenAI Chat Completions API + `OPENAI_API_KEY`

6. **Analysis structure** — The LLM produces four markdown sections:
   - **Executive Summary** — 2-3 sentence overview of overall agent health
   - **Anomalies** — Unexpected patterns, correlated failures, regressions
   - **Remediation** — Concrete, prioritized suggestions for fixing issues
   - **Risk Assessment** — Overall risk level (HIGH / MEDIUM / LOW) with justification

7. **Error handling** — Best-effort, never blocks the eval run. Missing API key, API call failures, unknown provider prefix, malformed response — all print a warning via `rich.console` and continue without analysis. The report is still generated, just without the analysis section.

8. **New file** — `src/agent_lemon_lime/report/analyzer.py` containing:
   - `analyze_report()` — public entry point, orchestrates context building and LLM call
   - `_call_anthropic()` — makes the Anthropic Messages API request
   - `_call_openai()` — makes the OpenAI Chat Completions API request
   - `_build_context()` — assembles all eval data into a structured prompt

9. **Integration** — CLI layer calls `analyze_report()` after `ReportSynthesizer.build()` produces the `EvalReport`, before writing to disk. The analysis text is inserted into the markdown output. `LemonAgent` is unchanged.

10. **Testing** — Mock `httpx` responses. Test context building (all data types present, partial data). Test markdown insertion (analysis appears in correct position). Test graceful degradation (missing key, API error, bad prefix, empty response).

## Config Changes

### `agent-lemon.yaml`

New optional `model` field under `report`:

```yaml
report:
  output: ".agent-lemon/report.md"
  format: markdown
  model: anthropic/claude-sonnet-4-20250514
```

### Config model

```python
class ReportConfig(BaseModel):
    output: str = ".agent-lemon/report.md"
    log: str | None = None
    format: Literal["markdown", "json"] = "markdown"
    model: str | None = None  # e.g. "anthropic/claude-sonnet-4-20250514"
```

Backward-compatible: existing configs without `model` default to `None` (analysis skipped).

## Analyzer Module

### `analyze_report()`

```python
def analyze_report(
    report: EvalReport,
    *,
    model: str,
    config_yaml: str = "",
) -> str | None:
    """Run LLM analysis on eval results. Returns markdown text or None on failure."""
```

1. Parse `model` string to extract provider prefix and model name
2. Call `_build_context()` to assemble the prompt
3. Dispatch to `_call_anthropic()` or `_call_openai()` based on prefix
4. Return the LLM's response text, or `None` on any error

### `_build_context()`

Assembles a structured text block from:

- Agent name, generated timestamp
- Summary table (total, passed, failed, pass rate)
- Per-result detail (name, domain, pass/fail, failures, stdout/stderr snippets)
- SCP violations list
- Agent config YAML (if provided)

The context is capped at a reasonable length to avoid token limits — stdout/stderr are truncated per result if they exceed a threshold.

### `_call_anthropic()`

```python
def _call_anthropic(
    model: str,
    context: str,
) -> str | None:
```

- Reads `ANTHROPIC_API_KEY` from environment
- POST to `https://api.anthropic.com/v1/messages` with `anthropic-version: 2023-06-01`
- System prompt instructs the LLM to produce the four analysis sections
- Returns the text content from the first content block, or `None` on error

### `_call_openai()`

```python
def _call_openai(
    model: str,
    context: str,
) -> str | None:
```

- Reads `OPENAI_API_KEY` from environment
- POST to `https://api.openai.com/v1/chat/completions`
- System message with analysis instructions, user message with context
- Returns `choices[0].message.content`, or `None` on error

### System Prompt

The system prompt instructs the LLM to analyze agent eval results and produce:

```
You are an AI agent evaluation analyst. Analyze the following eval results and produce a report with these sections:

## Analysis

### Executive Summary
2-3 sentences on overall agent health and key findings.

### Anomalies
Unexpected patterns, correlated failures, or regressions. If none found, say so.

### Remediation
Concrete, prioritized suggestions. Reference specific eval names and failure messages.

### Risk Assessment
Overall risk: **HIGH**, **MEDIUM**, or **LOW**
Justification in 1-2 sentences.
```

## Markdown Integration

The analysis section is inserted into the markdown report between the header block and "## Summary". `ReportSynthesizer.to_markdown()` is not modified — instead, the CLI layer inserts the analysis text into the rendered markdown string before writing to disk.

### Insertion logic

```python
def insert_analysis(markdown: str, analysis: str) -> str:
    """Insert analysis section after header, before ## Summary."""
```

Finds the `## Summary` line and inserts the analysis block (with a blank line before and after) immediately above it.

## Report Output Example

```markdown
# Agent Lemon Report

**Generated:** 2026-04-20T14:30:00+00:00
**Sandbox:** openshell
**Provider:** anthropic
**Model:** claude-opus-4-6

## Analysis

### Executive Summary
The agent passes all correctness evals but shows concerning behavioral patterns...

### Anomalies
- inspect::agentharm scored 0.65, below the 0.8 threshold...

### Remediation
1. Review system prompt guardrails for harmful content refusal...

### Risk Assessment
Overall risk: **MEDIUM**
Behavioral scores indicate safety guardrails need strengthening...

## Summary

| Metric | Value |
|--------|-------|
| Total | 5 |
| Passed | 4 |
...
```

## Files Touched

### New files

| File | Purpose |
|------|---------|
| `src/agent_lemon_lime/report/analyzer.py` | `analyze_report()`, `_call_anthropic()`, `_call_openai()`, `_build_context()`, `insert_analysis()` |
| `tests/test_analyzer.py` | Unit tests for analyzer module |

### Modified files

| File | Change |
|------|--------|
| `src/agent_lemon_lime/config.py` | Add `model: str \| None = None` to `ReportConfig` |
| `src/agent_lemon_lime/cli/lemon.py` | Call `analyze_report()` after report build, insert into markdown before write |

No new dependencies. Uses `httpx` (already in `pyproject.toml`).

## Testing

Unit tests with mocked `httpx.Client`:

1. **Anthropic provider** — mock successful Messages API response, verify analysis text returned
2. **OpenAI provider** — mock successful Chat Completions response, verify analysis text returned
3. **Missing API key** — `ANTHROPIC_API_KEY` not set, verify returns `None` with no exception
4. **API error** — mock 500 response, verify returns `None` with no exception
5. **Unknown provider** — model string `"ollama/llama3"`, verify returns `None` with warning
6. **Context building** — verify all data types appear in context string (results, SCP, violations, config)
7. **Context with partial data** — empty results, no violations, no config YAML
8. **Markdown insertion** — analysis text appears between header and Summary
9. **Markdown insertion no-op** — `None` analysis leaves report unchanged
10. **Empty LLM response** — empty string from API, verify returns `None`

No integration tests — LLM calls are external and unit tests cover the boundary.
