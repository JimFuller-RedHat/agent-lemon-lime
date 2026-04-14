"""Agent Lime: production runtime monitor via OTEL endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent_lemon_lime.scp.models import SystemCapabilityProfile


class LimeEventType(StrEnum):
    TOOL_CALL = "tool_call"
    NETWORK_CALL = "network_call"
    MODEL_REQUEST = "model_request"
    ERROR = "error"


@dataclass
class LimeEvent:
    event_type: LimeEventType
    tool_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class LimeAgent:
    """Monitors a running agent via OTEL endpoint for SCP compliance."""

    def __init__(
        self,
        *,
        otel_endpoint: str,
        assert_scp: SystemCapabilityProfile,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self.otel_endpoint = otel_endpoint
        self.assert_scp = assert_scp
        self.poll_interval_seconds = poll_interval_seconds

    def analyse_events(self, events: list[LimeEvent]) -> list[str]:
        """Return anomaly/violation messages for the given events.

        Args:
            events: Runtime events captured from the OTEL collector.

        Returns:
            List of violation strings; empty if all events are compliant.
        """
        allowed_hosts = {
            ep.host
            for policy in self.assert_scp.network_policies.values()
            for ep in policy.endpoints
        }
        anomalies: list[str] = []
        for event in events:
            if event.event_type == LimeEventType.NETWORK_CALL:
                host = str(event.metadata.get("host", ""))
                if host and host not in allowed_hosts:
                    anomalies.append(
                        f"SCP violation: outbound network call to '{host}' not in allowed profile"
                    )
        return anomalies

    def collect_events_from_otel(self, *, trace_id: str | None = None) -> list[LimeEvent]:
        """Fetch recent events from the OTEL collector.

        Real implementation queries the OTEL collector's trace/metrics endpoint
        via httpx and parses spans into LimeEvents. Returns [] if unreachable.

        Args:
            trace_id: Optional trace ID to filter events. None fetches recent events.

        Returns:
            List of LimeEvent instances; empty list if collector is unreachable.
        """
        return []
