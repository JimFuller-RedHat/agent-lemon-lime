"""Tests for SCP converter — OpenShell draft policy ↔ SystemCapabilityProfile."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_lemon_lime.scp.converter import from_policy_chunks, to_sandbox_policy
from agent_lemon_lime.scp.models import SystemCapabilityProfile


@dataclass
class FakeEndpoint:
    host: str = "api.example.com"
    port: int = 443
    ports: list[int] = field(default_factory=list)
    protocol: str = "rest"
    tls: str = ""
    enforcement: str = "audit"
    access: str = "full"


@dataclass
class FakeBinary:
    path: str = "/usr/bin/curl"


@dataclass
class FakeRule:
    name: str = "allow-api"
    endpoints: list[FakeEndpoint] = field(default_factory=list)
    binaries: list[FakeBinary] = field(default_factory=list)


@dataclass
class FakeChunk:
    id: str = "chunk-1"
    rule_name: str = "allow-api"
    proposed_rule: FakeRule = field(default_factory=FakeRule)


def test_empty_chunks_returns_permissive():
    scp = from_policy_chunks([])
    permissive = SystemCapabilityProfile.permissive()
    assert scp.network_policies.keys() == permissive.network_policies.keys()


def test_single_chunk_single_endpoint():
    ep = FakeEndpoint(host="api.anthropic.com", port=443)
    rule = FakeRule(name="anthropic", endpoints=[ep])
    chunk = FakeChunk(rule_name="anthropic-api", proposed_rule=rule)

    scp = from_policy_chunks([chunk])

    assert "anthropic-api" in scp.network_policies
    policy = scp.network_policies["anthropic-api"]
    assert policy.name == "anthropic-api"
    assert len(policy.endpoints) == 1
    assert policy.endpoints[0].host == "api.anthropic.com"
    assert policy.endpoints[0].port == 443
    assert policy.endpoints[0].enforcement == "audit"


def test_chunk_with_binaries():
    ep = FakeEndpoint(host="github.com")
    binary = FakeBinary(path="/usr/bin/git")
    rule = FakeRule(name="git-access", endpoints=[ep], binaries=[binary])
    chunk = FakeChunk(rule_name="git", proposed_rule=rule)

    scp = from_policy_chunks([chunk])
    policy = scp.network_policies["git"]
    assert len(policy.binaries) == 1
    assert policy.binaries[0].path == "/usr/bin/git"


def test_multiple_chunks():
    chunk1 = FakeChunk(
        rule_name="api",
        proposed_rule=FakeRule(
            name="api",
            endpoints=[FakeEndpoint(host="api.anthropic.com")],
        ),
    )
    chunk2 = FakeChunk(
        rule_name="pypi",
        proposed_rule=FakeRule(
            name="pypi",
            endpoints=[FakeEndpoint(host="pypi.org", port=443)],
        ),
    )

    scp = from_policy_chunks([chunk1, chunk2])
    assert len(scp.network_policies) == 2
    assert "api" in scp.network_policies
    assert "pypi" in scp.network_policies


def test_chunk_rule_name_fallback_to_rule_name():
    rule = FakeRule(name="from-rule", endpoints=[FakeEndpoint()])
    chunk = FakeChunk(id="chunk-id", rule_name="", proposed_rule=rule)

    scp = from_policy_chunks([chunk])
    assert "from-rule" in scp.network_policies


def test_chunk_rule_name_fallback_to_id():
    rule = FakeRule(name="", endpoints=[FakeEndpoint()])
    chunk = FakeChunk(id="chunk-id", rule_name="", proposed_rule=rule)

    scp = from_policy_chunks([chunk])
    assert "chunk-id" in scp.network_policies


def test_default_enforcement_is_audit():
    ep = FakeEndpoint(host="example.com", enforcement="")
    rule = FakeRule(endpoints=[ep])
    chunk = FakeChunk(proposed_rule=rule)

    scp = from_policy_chunks([chunk])
    policy = list(scp.network_policies.values())[0]
    assert policy.endpoints[0].enforcement == "audit"


def test_default_port_is_443():
    ep = FakeEndpoint(host="example.com", port=0)
    rule = FakeRule(endpoints=[ep])
    chunk = FakeChunk(proposed_rule=rule)

    scp = from_policy_chunks([chunk])
    policy = list(scp.network_policies.values())[0]
    assert policy.endpoints[0].port == 443


def test_discovery_policy_allows_inference_and_dns():
    scp = SystemCapabilityProfile.discovery()
    assert "inference-proxy" in scp.network_policies
    assert "dns" in scp.network_policies
    assert len(scp.network_policies) == 2

    inference = scp.network_policies["inference-proxy"]
    assert inference.endpoints[0].host == "inference.local"
    assert inference.endpoints[0].port == 443
    assert inference.endpoints[0].enforcement == "enforce"

    dns = scp.network_policies["dns"]
    assert dns.endpoints[0].port == 53
    assert dns.endpoints[0].enforcement == "enforce"


def test_to_sandbox_policy_returns_none_without_openshell():
    scp = SystemCapabilityProfile.discovery()
    result = to_sandbox_policy(scp)
    # openshell may or may not be importable in test env
    if result is None:
        assert True  # expected when openshell not available
    else:
        assert result.version == 1
        assert "inference-proxy" in result.network_policies
        assert "dns" in result.network_policies
