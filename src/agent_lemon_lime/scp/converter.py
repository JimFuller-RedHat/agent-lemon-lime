"""Convert between SCP models and OpenShell protobuf messages."""

from __future__ import annotations

import logging
from typing import Any

from agent_lemon_lime.scp.models import (
    NetworkBinaryDef,
    NetworkEndpoint,
    NetworkPolicy,
    SystemCapabilityProfile,
)

logger = logging.getLogger(__name__)


def _convert_endpoint(ep: Any) -> NetworkEndpoint:
    """Map a protobuf NetworkEndpoint to our Pydantic model."""
    return NetworkEndpoint(
        host=ep.host,
        port=ep.port or 443,
        ports=list(ep.ports) if ep.ports else [],
        protocol=ep.protocol or "",
        tls=ep.tls or "",
        enforcement=ep.enforcement or "audit",
        access=ep.access or "full",
    )


def _convert_binary(b: Any) -> NetworkBinaryDef:
    return NetworkBinaryDef(path=b.path)


def from_policy_chunks(chunks: list[Any]) -> SystemCapabilityProfile:
    """Build an SCP from OpenShell PolicyChunk objects.

    Args:
        chunks: List of openshell PolicyChunk protobuf messages.

    Returns:
        SystemCapabilityProfile with network policies derived from
        the draft policy. Returns permissive() if chunks is empty.
    """
    if not chunks:
        return SystemCapabilityProfile.permissive()

    policies: dict[str, NetworkPolicy] = {}
    for chunk in chunks:
        rule = chunk.proposed_rule
        rule_name = chunk.rule_name or rule.name or chunk.id
        endpoints = [_convert_endpoint(ep) for ep in rule.endpoints]
        binaries = [_convert_binary(b) for b in rule.binaries]
        policies[rule_name] = NetworkPolicy(
            name=rule_name,
            endpoints=endpoints,
            binaries=binaries,
        )

    return SystemCapabilityProfile(network_policies=policies)


def to_sandbox_policy(scp: SystemCapabilityProfile) -> Any:
    """Convert an SCP to an OpenShell SandboxPolicy protobuf message.

    Args:
        scp: The SystemCapabilityProfile to convert.

    Returns:
        An openshell SandboxPolicy protobuf message, or None if the
        openshell library is not available.
    """
    try:
        from openshell._proto import sandbox_pb2
    except ImportError:
        logger.warning("openshell not available; cannot build SandboxPolicy")
        return None

    policy = sandbox_pb2.SandboxPolicy(
        version=scp.version,
        filesystem=sandbox_pb2.FilesystemPolicy(
            include_workdir=scp.filesystem_policy.include_workdir,
            read_only=scp.filesystem_policy.read_only,
            read_write=scp.filesystem_policy.read_write,
        ),
        landlock=sandbox_pb2.LandlockPolicy(
            compatibility=scp.landlock.compatibility,
        ),
        process=sandbox_pb2.ProcessPolicy(
            run_as_user=scp.process.run_as_user,
            run_as_group=scp.process.run_as_group,
        ),
    )
    for key, np in scp.network_policies.items():
        endpoints = [
            sandbox_pb2.NetworkEndpoint(
                host=ep.host,
                port=ep.port,
                protocol=ep.protocol,
                tls=ep.tls,
                enforcement=ep.enforcement,
                access=ep.access,
                ports=ep.ports,
            )
            for ep in np.endpoints
        ]
        binaries = [
            sandbox_pb2.NetworkBinary(path=b.path)
            for b in np.binaries
        ]
        policy.network_policies[key].CopyFrom(
            sandbox_pb2.NetworkPolicyRule(
                name=np.name,
                endpoints=endpoints,
                binaries=binaries,
            )
        )
    return policy
