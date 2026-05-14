"""Standard eval types: EvalDomain and EvalOutput."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class EvalDomain(StrEnum):
    SAFETY = "safety"
    STABILITY = "stability"
    CORRECTNESS = "correctness"
    SECURITY = "security"
    BEHAVIORAL = "behavioral"


class EvalOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    domain: EvalDomain
    metadata: dict[str, object] = {}
