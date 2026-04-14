"""SystemCapabilityProfile — Pydantic model for NVIDIA OpenShell policy YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class FilesystemPolicy(BaseModel):
    include_workdir: bool = True
    read_only: list[str] = Field(default_factory=list)
    read_write: list[str] = Field(default_factory=list)


class LandlockPolicy(BaseModel):
    compatibility: Literal["best_effort", "hard_requirement"] = "best_effort"


class ProcessPolicy(BaseModel):
    run_as_user: str = "sandbox"
    run_as_group: str = "sandbox"


class NetworkEndpoint(BaseModel):
    host: str
    port: int = 443
    protocol: Literal["rest", "grpc"] = "rest"
    tls: Literal["terminate", "passthrough"] = "terminate"
    enforcement: Literal["enforce", "audit"] = "enforce"
    access: Literal["read-only", "read-write", "full"] = "read-only"


class NetworkPolicy(BaseModel):
    name: str
    endpoints: list[NetworkEndpoint]
    binaries: list[dict[str, str]] = Field(default_factory=list)


class SystemCapabilityProfile(BaseModel):
    version: int = 1
    filesystem_policy: FilesystemPolicy = Field(default_factory=FilesystemPolicy)
    landlock: LandlockPolicy = Field(default_factory=LandlockPolicy)
    process: ProcessPolicy = Field(default_factory=ProcessPolicy)
    network_policies: dict[str, NetworkPolicy] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> SystemCapabilityProfile:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)
        )

    @classmethod
    def permissive(cls) -> SystemCapabilityProfile:
        """Discovery-mode profile: audit-only, no enforcement."""
        return cls(
            filesystem_policy=FilesystemPolicy(
                include_workdir=True,
                read_only=["/usr", "/lib", "/etc", "/proc"],
                read_write=["/tmp", "/sandbox"],
            ),
            network_policies={
                "all_audit": NetworkPolicy(
                    name="Audit all outbound",
                    endpoints=[NetworkEndpoint(host="*", enforcement="audit")],
                )
            },
        )
