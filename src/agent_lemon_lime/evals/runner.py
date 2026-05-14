"""EvalRunner: runs EvalCase instances against a sandbox."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_eval.config import JudgeConfig
from pydantic import BaseModel

from agent_lemon_lime.evals.scoring import JudgeScore, score_eval_output
from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput
from agent_lemon_lime.harness.base import AbstractSandbox

logger = logging.getLogger(__name__)


class EvalInput(BaseModel):
    command: list[str]
    workdir: str | None = None
    env: dict[str, str] = {}
    timeout_seconds: int | None = None


@dataclass
class EvalCase:
    name: str
    input: EvalInput
    judges: list[JudgeConfig]
    domain: EvalDomain = EvalDomain.CORRECTNESS
    description: str = ""


@dataclass
class EvalResult:
    name: str
    passed: bool
    domain: EvalDomain
    output: EvalOutput
    failures: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    scores: dict[str, JudgeScore] = field(default_factory=dict)


class EvalRunner:
    """Executes eval cases against a sandbox, returns results."""

    def run(
        self,
        cases: list[EvalCase],
        *,
        sandbox: AbstractSandbox,
        judge_model: str | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        with sandbox:
            for case in cases:
                raw = sandbox.exec(
                    case.input.command,
                    workdir=case.input.workdir,
                    env=case.input.env or None,
                    timeout_seconds=case.input.timeout_seconds,
                )
                output = EvalOutput(
                    exit_code=raw.exit_code,
                    stdout=raw.stdout,
                    stderr=raw.stderr,
                    domain=case.domain,
                )
                scores = score_eval_output(
                    output,
                    case.judges,
                    model=judge_model,
                )
                failures = [
                    f"{name}: {s.rationale or s.error or ''}"
                    for name, s in scores.items()
                    if s.value is False or s.value == 0 or s.error
                ]
                result = EvalResult(
                    name=case.name,
                    passed=len(failures) == 0,
                    domain=case.domain,
                    output=output,
                    failures=failures,
                    command=list(case.input.command),
                    scores=scores,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
        return results
