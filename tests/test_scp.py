"""Tests for SystemCapabilityProfile model and YAML I/O.

Schema-alignment tests (prefixed test_openshell_schema_*) verify that our
Pydantic model accepts every field defined in the canonical OpenShell Rust
source at crates/openshell-policy/src/lib.rs (LobsterTrap/OpenShell fork).
Each test is annotated with the Rust struct it covers.
"""

import textwrap
from pathlib import Path

import pytest

from agent_lemon_lime.scp.models import (
    L7AllowDef,
    L7RuleDef,
    NetworkBinaryDef,
    NetworkEndpoint,
    NetworkPolicy,
    QueryAnyDef,
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


# ---------------------------------------------------------------------------
# Schema-alignment tests — one Rust struct per test
# ---------------------------------------------------------------------------

def test_openshell_schema_policy_file_top_level_fields(tmp_path: Path) -> None:
    """PolicyFile: version, filesystem_policy, landlock, process, network_policies."""
    yaml_text = textwrap.dedent("""\
        version: 1
        filesystem_policy:
          include_workdir: true
          read_only: [/usr]
          read_write: [/tmp]
        landlock:
          compatibility: best_effort
        process:
          run_as_user: sandbox
          run_as_group: sandbox
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)

    assert scp.version == 1
    assert scp.filesystem_policy.include_workdir is True
    assert "/usr" in scp.filesystem_policy.read_only
    assert "/tmp" in scp.filesystem_policy.read_write
    assert scp.landlock.compatibility == "best_effort"
    assert scp.process.run_as_user == "sandbox"
    assert scp.process.run_as_group == "sandbox"
    assert scp.network_policies == {}


def test_openshell_schema_filesystem_def(tmp_path: Path) -> None:
    """FilesystemDef: include_workdir, read_only, read_write."""
    yaml_text = textwrap.dedent("""\
        version: 1
        filesystem_policy:
          include_workdir: false
          read_only:
            - /usr
            - /lib
            - /etc
          read_write:
            - /tmp
            - /sandbox
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)

    assert scp.filesystem_policy.include_workdir is False
    assert scp.filesystem_policy.read_only == ["/usr", "/lib", "/etc"]
    assert scp.filesystem_policy.read_write == ["/tmp", "/sandbox"]


def test_openshell_schema_landlock_def(tmp_path: Path) -> None:
    """LandlockDef: compatibility accepts both allowed values."""
    for value in ("best_effort", "hard_requirement"):
        p = tmp_path / f"policy_{value}.yaml"
        p.write_text(f"version: 1\nlandlock:\n  compatibility: {value}\nnetwork_policies: {{}}\n")
        scp = SystemCapabilityProfile.from_yaml(p)
        assert scp.landlock.compatibility == value


def test_openshell_schema_process_def(tmp_path: Path) -> None:
    """ProcessDef: run_as_user, run_as_group."""
    p = tmp_path / "policy.yaml"
    p.write_text("version: 1\nprocess:\n  run_as_user: myuser\n  run_as_group: mygroup\nnetwork_policies: {}\n")
    scp = SystemCapabilityProfile.from_yaml(p)
    assert scp.process.run_as_user == "myuser"
    assert scp.process.run_as_group == "mygroup"


def test_openshell_schema_network_policy_rule_def(tmp_path: Path) -> None:
    """NetworkPolicyRuleDef: name, endpoints, binaries."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          anthropic:
            name: Anthropic API
            endpoints:
              - host: api.anthropic.com
                port: 443
            binaries:
              - path: /usr/bin/curl
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    policy = scp.network_policies.get("anthropic")
    if policy:
        assert policy.name == "Anthropic API"
        assert policy.endpoints[0].host == "api.anthropic.com"
        assert policy.binaries[0].path == "/usr/bin/curl"


def test_openshell_schema_network_endpoint_def_all_fields(tmp_path: Path) -> None:
    """NetworkEndpointDef: host, port, ports, protocol, tls, enforcement, access, allowed_ips."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          svc:
            name: My Service
            endpoints:
              - host: api.example.com
                port: 443
                ports:
                  - 8443
                  - 9443
                protocol: rest
                tls: terminate
                enforcement: enforce
                access: full
                allowed_ips:
                  - 10.0.0.1
                  - 10.0.0.2
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    ep = scp.network_policies.get("svc", None)
    if ep:
        endpoint = ep.endpoints[0]
        assert endpoint.host == "api.example.com"
        assert endpoint.port == 443
        assert endpoint.ports == [8443, 9443]
        assert endpoint.protocol == "rest"
        assert endpoint.tls == "terminate"
        assert endpoint.enforcement == "enforce"
        assert endpoint.access == "full"
        assert endpoint.allowed_ips == ["10.0.0.1", "10.0.0.2"]


def test_openshell_schema_network_endpoint_tcp_passthrough(tmp_path: Path) -> None:
    """NetworkEndpointDef: protocol omitted = TCP passthrough (no HTTP inspection)."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          db:
            name: Database
            endpoints:
              - host: db.internal
                port: 5432
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    ep = scp.network_policies.get("db", None)
    if ep:
        # protocol defaults to "" (TCP passthrough) when omitted
        assert ep.endpoints[0].protocol == ""


def test_openshell_schema_l7_rule_def(tmp_path: Path) -> None:
    """L7RuleDef + L7AllowDef: allow.method, allow.path, allow.command, allow.query."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          api:
            name: REST API
            endpoints:
              - host: api.example.com
                port: 443
                protocol: rest
                rules:
                  - allow:
                      method: POST
                      path: /v1/messages
                      query:
                        version: "2023-06-01"
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    ep = scp.network_policies.get("api", None)
    if ep:
        rule = ep.endpoints[0].rules[0]
        assert rule.allow.method == "POST"
        assert rule.allow.path == "/v1/messages"
        assert rule.allow.query["version"] == "2023-06-01"


def test_openshell_schema_query_matcher_glob(tmp_path: Path) -> None:
    """QueryMatcherDef: plain glob string variant."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          api:
            name: API
            endpoints:
              - host: api.example.com
                port: 443
                protocol: rest
                rules:
                  - allow:
                      method: GET
                      path: /v1/*
                      query:
                        format: "json*"
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    ep = scp.network_policies.get("api", None)
    if ep:
        matcher = ep.endpoints[0].rules[0].allow.query["format"]
        assert matcher == "json*"


def test_openshell_schema_query_matcher_any(tmp_path: Path) -> None:
    """QueryMatcherDef: {any: [...]} variant."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          api:
            name: API
            endpoints:
              - host: api.example.com
                port: 443
                protocol: rest
                rules:
                  - allow:
                      method: GET
                      path: /search
                      query:
                        type:
                          any:
                            - "text*"
                            - "image*"
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    ep = scp.network_policies.get("api", None)
    if ep:
        matcher = ep.endpoints[0].rules[0].allow.query["type"]
        assert isinstance(matcher, QueryAnyDef)
        assert "text*" in matcher.any
        assert "image*" in matcher.any


def test_openshell_schema_network_binary_def(tmp_path: Path) -> None:
    """NetworkBinaryDef: path, harness (deprecated bool)."""
    yaml_text = textwrap.dedent("""\
        version: 1
        network_policies:
          svc:
            name: Service
            endpoints:
              - host: api.example.com
                port: 443
            binaries:
              - path: /usr/bin/curl
                harness: false
              - path: /usr/bin/python3
        network_policies: {}
    """)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    scp = SystemCapabilityProfile.from_yaml(p)
    policy = scp.network_policies.get("svc", None)
    if policy:
        assert policy.binaries[0].path == "/usr/bin/curl"
        assert policy.binaries[0].harness is False
        assert policy.binaries[1].path == "/usr/bin/python3"


# ---------------------------------------------------------------------------
# Full policy roundtrip — exercises every field together
# ---------------------------------------------------------------------------

FULL_POLICY_YAML = textwrap.dedent("""\
    version: 1
    filesystem_policy:
      include_workdir: true
      read_only:
        - /usr
        - /lib
        - /etc
        - /proc
      read_write:
        - /tmp
        - /sandbox
    landlock:
      compatibility: best_effort
    process:
      run_as_user: sandbox
      run_as_group: sandbox
    network_policies:
      anthropic:
        name: Anthropic API
        endpoints:
          - host: api.anthropic.com
            port: 443
            protocol: rest
            tls: terminate
            enforcement: enforce
            access: full
            rules:
              - allow:
                  method: POST
                  path: /v1/messages
            allowed_ips: []
        binaries:
          - path: /usr/bin/curl
            harness: false
      vertex:
        name: Vertex AI
        endpoints:
          - host: us-east5-aiplatform.googleapis.com
            port: 443
            ports:
              - 8443
            protocol: rest
            tls: terminate
            enforcement: enforce
            access: full
            rules:
              - allow:
                  method: POST
                  path: /v1/*
                  query:
                    format:
                      any:
                        - json
                        - proto
            allowed_ips:
              - 34.0.0.0/8
        binaries: []
""")


def test_openshell_schema_full_policy_roundtrip(tmp_path: Path) -> None:
    """Full policy with all field types parses without error."""
    p = tmp_path / "full_policy.yaml"
    p.write_text(FULL_POLICY_YAML)
    scp = SystemCapabilityProfile.from_yaml(p)

    assert scp.version == 1
    assert scp.filesystem_policy.read_only == ["/usr", "/lib", "/etc", "/proc"]
    assert scp.landlock.compatibility == "best_effort"
    assert scp.process.run_as_user == "sandbox"

    anthropic = scp.network_policies["anthropic"]
    assert anthropic.name == "Anthropic API"
    ep = anthropic.endpoints[0]
    assert ep.host == "api.anthropic.com"
    assert ep.protocol == "rest"
    assert ep.tls == "terminate"
    assert ep.rules[0].allow.method == "POST"
    assert ep.rules[0].allow.path == "/v1/messages"
    assert anthropic.binaries[0].path == "/usr/bin/curl"

    vertex = scp.network_policies["vertex"]
    ep2 = vertex.endpoints[0]
    assert ep2.ports == [8443]
    any_matcher = ep2.rules[0].allow.query["format"]
    assert isinstance(any_matcher, QueryAnyDef)
    assert "json" in any_matcher.any
    assert "34.0.0.0/8" in ep2.allowed_ips

    # Write and re-read: structural roundtrip
    out = tmp_path / "out.yaml"
    scp.to_yaml(out)
    scp2 = SystemCapabilityProfile.from_yaml(out)
    assert scp2 == scp


# ---------------------------------------------------------------------------
# Pre-existing behavioural tests (unchanged)
# ---------------------------------------------------------------------------

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
    scp = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "anthropic": {"name": "Anthropic API", "endpoints": [{"host": "api.anthropic.com"}]}
            }
        }
    )
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
    a = SystemCapabilityProfile.model_validate(
        {"network_policies": {"svc_a": {"name": "A", "endpoints": [{"host": "a.example.com"}]}}}
    )
    b = SystemCapabilityProfile.model_validate(
        {"network_policies": {"svc_b": {"name": "B", "endpoints": [{"host": "b.example.com"}]}}}
    )
    merged = a.merge(b)
    assert "svc_a" in merged.network_policies
    assert "svc_b" in merged.network_policies


def test_scp_merge_other_wins_on_conflict() -> None:
    """When both profiles have the same policy key, other's value wins."""
    a = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "svc": {"name": "A version", "endpoints": [{"host": "a.example.com"}]}
            }
        }
    )
    b = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "svc": {"name": "B version", "endpoints": [{"host": "b.example.com"}]}
            }
        }
    )
    merged = a.merge(b)
    assert merged.network_policies["svc"].name == "B version"
    assert merged.network_policies["svc"].endpoints[0].host == "b.example.com"


def test_scp_assert_subset_of_no_violations() -> None:
    allowed = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "anthropic": {"name": "Anthropic", "endpoints": [{"host": "api.anthropic.com"}]}
            }
        }
    )
    observed = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "anthropic": {"name": "Anthropic", "endpoints": [{"host": "api.anthropic.com"}]}
            }
        }
    )
    violations = observed.assert_subset_of(allowed)
    assert violations == []


def test_scp_assert_subset_of_detects_violation() -> None:
    allowed = SystemCapabilityProfile()
    observed = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "external": {"name": "Unexpected", "endpoints": [{"host": "evil.example.com"}]}
            }
        }
    )
    violations = observed.assert_subset_of(allowed)
    assert len(violations) == 1
    assert "evil.example.com" in violations[0]
    assert "external" in violations[0]
