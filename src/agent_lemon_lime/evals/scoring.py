"""Bridge between lemon-lime's sandbox output and agent-eval-harness judges."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_eval.config import EvalConfig, JudgeConfig, ModelsConfig

from agent_lemon_lime.evals.standard import EvalOutput

logger = logging.getLogger(__name__)

_harness_score_mod = None


@dataclass
class JudgeScore:
    value: bool | int | float | str | None
    rationale: str = ""
    error: str | None = None


def _find_harness_score_module() -> Path | None:
    """Locate score.py in agent-eval-harness installation."""
    candidates = [
        Path(__file__).resolve().parents[4]
        / "agent-eval-harness"
        / "skills"
        / "eval-run"
        / "scripts"
        / "score.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_harness_scoring():
    """Lazy-load the harness score module."""
    global _harness_score_mod
    if _harness_score_mod is not None:
        return _harness_score_mod

    path = _find_harness_score_module()
    if path is None:
        raise ImportError(
            "agent-eval-harness score.py not found. "
            "Ensure agent-eval-harness is installed as a sibling directory."
        )
    spec = importlib.util.spec_from_file_location("_harness_score", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _harness_score_mod = mod
    return mod


def _eval_output_to_record(output: EvalOutput) -> dict[str, Any]:
    """Convert an EvalOutput to the record dict harness judges expect."""
    return {
        "exit_code": output.exit_code,
        "stdout": output.stdout,
        "stderr": output.stderr,
        "files": {},
        "tool_calls": [],
        "annotations": {},
    }


def _build_eval_config_for_judges(
    judges: list[JudgeConfig],
    *,
    model: str | None = None,
) -> EvalConfig:
    """Build a minimal EvalConfig to pass to harness load_judges()."""
    config = EvalConfig()
    config.judges = list(judges)
    if model:
        config.models = ModelsConfig(judge=model)
    return config


def score_eval_output(
    output: EvalOutput,
    judges: list[JudgeConfig],
    *,
    model: str | None = None,
) -> dict[str, JudgeScore]:
    """Score an EvalOutput using harness judges."""
    if not judges:
        return {}

    harness = _get_harness_scoring()
    config = _build_eval_config_for_judges(judges, model=model)
    loaded = harness.load_judges(config)
    record = _eval_output_to_record(output)
    scores: dict[str, JudgeScore] = {}

    for name, scorer, condition in loaded:
        if condition:
            try:
                if not eval(
                    condition,
                    {"__builtins__": {}},
                    {
                        "annotations": record.get("annotations", {}),
                        "outputs": record,
                    },
                ):
                    scores[name] = JudgeScore(
                        value=None,
                        rationale=f"Skipped: condition '{condition}' is false",
                    )
                    continue
            except Exception as e:
                scores[name] = JudgeScore(value=None, error=f"Condition error: {e}")
                continue
        try:
            result = scorer(outputs=record)
            if isinstance(result, tuple) and len(result) == 2:
                scores[name] = JudgeScore(value=result[0], rationale=str(result[1]))
            elif hasattr(result, "value"):
                scores[name] = JudgeScore(
                    value=result.value,
                    rationale=getattr(result, "rationale", ""),
                )
            elif isinstance(result, (bool, int, float, str)):
                scores[name] = JudgeScore(value=result)
            else:
                scores[name] = JudgeScore(value=result)
        except Exception as e:
            scores[name] = JudgeScore(value=None, error=str(e))

    return scores


def check_regressions(
    aggregated: dict[str, dict],
    thresholds: dict[str, dict],
    baseline: dict[str, dict] | None = None,
) -> list[str]:
    """Check for regressions using harness threshold detection."""
    harness = _get_harness_scoring()
    regressions = harness.detect_regressions(aggregated, thresholds, baseline)
    return [
        f"[{r.judge_name}] {r.metric}: expected {r.baseline_value}, got {r.current_value}"
        for r in regressions
    ]
