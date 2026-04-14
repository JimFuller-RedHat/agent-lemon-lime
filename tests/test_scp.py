"""Tests for SystemCapabilityProfile model and YAML I/O."""

import textwrap
from pathlib import Path

import pytest

from agent_lemon_lime.scp.models import (
    NetworkEndpoint,
    NetworkPolicy,
    SystemCapabilityProfile,
)

SAMPLE_YAML = textwrap.dedent("""\
    version: 1
    filesystem_policy:
      include_workdir: true
      read_only:
        - /usr
        - /lib
      read_write:
        - /tmp
    landlock:
      compatibility: best_effort
    process:
      run_as_user: sandbox
      run_as_group: sandbox
    network_policies: {}
""")


@pytest.fixture
def tmp_policy_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(SAMPLE_YAML)
    return p


def test_scp_roundtrip_yaml(tmp_path: Path) -> None:
    import yaml

    data = yaml.safe_load(SAMPLE_YAML)
    scp = SystemCapabilityProfile.model_validate(data)
    out = tmp_path / "policy.yaml"
    scp.to_yaml(out)
    loaded = SystemCapabilityProfile.from_yaml(out)
    assert loaded == scp


def test_scp_defaults() -> None:
    scp = SystemCapabilityProfile()
    assert scp.version == 1
    assert scp.filesystem_policy.include_workdir is True
    assert scp.network_policies == {}


def test_scp_with_network_endpoint() -> None:
    scp = SystemCapabilityProfile()
    ep = NetworkEndpoint(host="api.anthropic.com", port=443)
    policy = NetworkPolicy(name="Anthropic API", endpoints=[ep])
    scp.network_policies["anthropic"] = policy
    assert scp.network_policies["anthropic"].endpoints[0].host == "api.anthropic.com"


def test_scp_from_yaml_accepts_string_path(tmp_policy_yaml: Path) -> None:
    scp = SystemCapabilityProfile.from_yaml(str(tmp_policy_yaml))
    assert scp.version == 1


def test_scp_from_yaml_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        SystemCapabilityProfile.from_yaml(empty)


def test_permissive_scp() -> None:
    scp = SystemCapabilityProfile.permissive()
    for policy in scp.network_policies.values():
        for ep in policy.endpoints:
            assert ep.enforcement == "audit"


def test_scp_merge_unions_network_policies() -> None:
    a = SystemCapabilityProfile()
    a.network_policies["svc_a"] = NetworkPolicy(
        name="A", endpoints=[NetworkEndpoint(host="a.example.com")]
    )
    b = SystemCapabilityProfile()
    b.network_policies["svc_b"] = NetworkPolicy(
        name="B", endpoints=[NetworkEndpoint(host="b.example.com")]
    )
    merged = a.merge(b)
    assert "svc_a" in merged.network_policies
    assert "svc_b" in merged.network_policies


def test_scp_assert_subset_of_no_violations() -> None:
    allowed = SystemCapabilityProfile()
    allowed.network_policies["anthropic"] = NetworkPolicy(
        name="Anthropic", endpoints=[NetworkEndpoint(host="api.anthropic.com")]
    )
    observed = SystemCapabilityProfile()
    observed.network_policies["anthropic"] = NetworkPolicy(
        name="Anthropic", endpoints=[NetworkEndpoint(host="api.anthropic.com")]
    )
    violations = observed.assert_subset_of(allowed)
    assert violations == []


def test_scp_assert_subset_of_detects_violation() -> None:
    allowed = SystemCapabilityProfile()
    # No network policies allowed
    observed = SystemCapabilityProfile()
    observed.network_policies["external"] = NetworkPolicy(
        name="Unexpected", endpoints=[NetworkEndpoint(host="evil.example.com")]
    )
    violations = observed.assert_subset_of(allowed)
    assert len(violations) == 1
    assert "evil.example.com" in violations[0]
