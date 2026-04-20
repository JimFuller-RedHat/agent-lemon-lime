"""LemonConfig: reads agent-lemon.yaml from the project root."""

from __future__ import annotations

import os
import pathlib
import re
from typing import Literal

import yaml
from pydantic import BaseModel, Field


def resolve_env(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} references in env values from os.environ."""
    return {
        k: re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
        for k, v in env.items()
    }

CONFIG_FILENAME = "agent-lemon.yaml"


class RunConfig(BaseModel):
    command: str
    setup: str | None = None
    timeout_seconds: int = 300
    env: dict[str, str] = Field(default_factory=dict)
    workdir: str | None = None


class SkillSource(BaseModel):
    path: str | None = None
    git: str | None = None
    branch: str = "main"


class BackendConfig(BaseModel):
    type: Literal["inspect"]
    model: str
    tasks: list[str]
    score_threshold: float = 1.0


class EvalsConfig(BaseModel):
    directories: list[str] = Field(default_factory=list)
    skills: list[SkillSource] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)


class SCPConfig(BaseModel):
    output: str = ".agent-lemon/scp.yaml"
    assert_file: str | None = None


class ReportConfig(BaseModel):
    output: str = ".agent-lemon/report.md"
    log: str | None = None  # defaults to .agent-lemon/{agent-name}.log
    format: Literal["markdown", "json"] = "markdown"


class SandboxConfig(BaseModel):
    type: Literal["local", "openshell"] = "local"
    cluster: str | None = None
    timeout: float = 30.0
    ready_timeout_seconds: float = 120.0
    auto_start_gateway: bool = True
    provider: str | None = None
    model: str | None = None
    image: str | None = None
    discovery_policy: str | None = None


class LemonConfig(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    run: RunConfig
    evals: EvalsConfig = Field(default_factory=EvalsConfig)
    scp: SCPConfig = Field(default_factory=SCPConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    @classmethod
    def from_yaml(cls, text: str) -> LemonConfig:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> LemonConfig:
        return cls.from_yaml(pathlib.Path(path).read_text())

    @classmethod
    def from_dir(cls, directory: pathlib.Path | str) -> LemonConfig:
        p = pathlib.Path(directory) / CONFIG_FILENAME
        if not p.exists():
            raise FileNotFoundError(
                f"agent-lemon.yaml not found in {directory}. "
                "Run 'agent-lemon init' to generate a template."
            )
        return cls.from_file(p)


class RunMode:
    DISCOVER = "discover"
    ASSERT = "assert"
