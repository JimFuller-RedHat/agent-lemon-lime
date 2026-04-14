"""Standard evaluators for safety, stability, correctness, and security."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class EvalDomain(StrEnum):
    SAFETY = "safety"
    STABILITY = "stability"
    CORRECTNESS = "correctness"
    SECURITY = "security"


class EvalOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    domain: EvalDomain
    metadata: dict[str, object] = {}


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, output: EvalOutput) -> bool: ...


class ExitCodeEvaluator:
    """Pass if exit_code == 0."""

    def evaluate(self, output: EvalOutput) -> bool:
        return output.exit_code == 0


class NoErrorOutputEvaluator:
    """Pass if stderr is empty."""

    def evaluate(self, output: EvalOutput) -> bool:
        return output.stderr.strip() == ""


class OutputContainsEvaluator:
    """Pass if stdout contains the expected substring."""

    def __init__(self, expected: str) -> None:
        self.expected = expected

    def evaluate(self, output: EvalOutput) -> bool:
        return self.expected in output.stdout
