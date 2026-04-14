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
    host: str = Field(..., min_length=1)
    port: int = Field(443, ge=1, le=65535)
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
        """Load a SystemCapabilityProfile from a YAML file.

        Args:
            path: Path to the YAML file (str or Path).

        Raises:
            ValueError: If the file does not contain a YAML mapping.
        """
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            msg = f"Expected a YAML mapping in {path!r}, got {type(data).__name__}"
            raise ValueError(msg)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Write this profile to a YAML file.

        Args:
            path: Destination file path (str or Path).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)
        )

    @classmethod
    def permissive(cls) -> SystemCapabilityProfile:
        """Return a discovery-mode profile with audit-only enforcement."""
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

    def merge(self, other: SystemCapabilityProfile) -> SystemCapabilityProfile:
        """Return a new profile unioning network_policies; other wins on conflict.

        Args:
            other: Profile whose network_policies take precedence on key collision.
        """
        merged_policies = {**self.network_policies, **other.network_policies}
        return self.model_copy(update={"network_policies": merged_policies})

    def assert_subset_of(self, allowed: SystemCapabilityProfile) -> list[str]:
        """Check that every network policy and host in self is permitted by allowed.

        Args:
            allowed: The reference profile defining permitted policies.

        Returns:
            A list of violation strings; empty if this profile is compliant.
        """
        violations: list[str] = []
        for key, policy in self.network_policies.items():
            if key not in allowed.network_policies:
                for ep in policy.endpoints:
                    violations.append(
                        f"Unauthorized network endpoint '{ep.host}' in policy '{key}'"
                    )
            else:
                allowed_hosts = {ep.host for ep in allowed.network_policies[key].endpoints}
                for ep in policy.endpoints:
                    if ep.host not in allowed_hosts:
                        violations.append(
                            f"Unauthorized host '{ep.host}' in network policy '{key}'"
                        )
        return violations
