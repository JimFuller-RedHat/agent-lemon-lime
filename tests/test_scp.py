"""Tests for SystemCapabilityProfile model and YAML I/O."""
from pathlib import Path

from argus.core.scp import (
    FilesystemPolicy,  # noqa: F401 — exported for type-checking by callers
    LandlockPolicy,  # noqa: F401 — exported for type-checking by callers
    NetworkEndpoint,  # noqa: F401 — exported for type-checking by callers
    NetworkPolicy,  # noqa: F401 — exported for type-checking by callers
    ProcessPolicy,  # noqa: F401 — exported for type-checking by callers
    SystemCapabilityProfile,
)


def _minimal_scp_dict() -> dict:
    return {
        "version": 1,
        "filesystem_policy": {
            "include_workdir": True,
            "read_only": ["/usr"],
            "read_write": ["/tmp"],
        },
        "landlock": {"compatibility": "best_effort"},
        "process": {"run_as_user": "sandbox", "run_as_group": "sandbox"},
        "network_policies": {},
    }


def test_scp_roundtrip_yaml(tmp_path: Path) -> None:
    scp = SystemCapabilityProfile.model_validate(_minimal_scp_dict())
    out = tmp_path / "policy.yaml"
    scp.to_yaml(out)
    loaded = SystemCapabilityProfile.from_yaml(out)
    assert loaded == scp


def test_scp_defaults() -> None:
    scp = SystemCapabilityProfile()
    assert scp.version == 1
    assert scp.filesystem_policy.include_workdir is True
    assert scp.network_policies == {}


def test_scp_with_network_policy() -> None:
    data = _minimal_scp_dict()
    data["network_policies"] = {
        "github_api": {
            "name": "GitHub API",
            "endpoints": [{"host": "api.github.com", "port": 443}],
        }
    }
    scp = SystemCapabilityProfile.model_validate(data)
    assert "github_api" in scp.network_policies
    assert scp.network_policies["github_api"].endpoints[0].host == "api.github.com"


def test_scp_from_yaml(tmp_policy_yaml: Path) -> None:
    scp = SystemCapabilityProfile.from_yaml(tmp_policy_yaml)
    assert scp.version == 1


def test_permissive_scp() -> None:
    scp = SystemCapabilityProfile.permissive()
    assert scp.landlock.compatibility == "best_effort"
    # Permissive means no network enforcement
    for policy in scp.network_policies.values():
        for ep in policy.endpoints:
            assert ep.enforcement == "audit"
