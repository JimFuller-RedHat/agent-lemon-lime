"""EvalRunner: runs EvalCase instances against a sandbox."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from agent_lemon_lime.evals.standard import EvalDomain, EvalOutput, Evaluator
from agent_lemon_lime.harness.base import AbstractSandbox


class EvalInput(BaseModel):
    command: list[str]
    workdir: str | None = None
    env: dict[str, str] = {}
    timeout_seconds: int | None = None


@dataclass
class EvalCase:
    name: str
    input: EvalInput
    evaluators: list[Evaluator]
    domain: EvalDomain = EvalDomain.CORRECTNESS
    description: str = ""
    judge_hint: str = ""


@dataclass
class EvalResult:
    name: str
    passed: bool
    domain: EvalDomain
    output: EvalOutput
    failures: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)


class EvalRunner:
    """Executes eval cases against a sandbox, returns results."""

    def run(
        self,
        cases: list[EvalCase],
        *,
        sandbox: AbstractSandbox,
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
                failures = [type(ev).__name__ for ev in case.evaluators if not ev.evaluate(output)]
                result = EvalResult(
                    name=case.name,
                    passed=len(failures) == 0,
                    domain=case.domain,
                    output=output,
                    failures=failures,
                    command=list(case.input.command),
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
        return results
