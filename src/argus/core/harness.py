"""OpenShellHarness — subprocess wrapper around the openshell CLI."""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from argus.core.scp import SystemCapabilityProfile


@dataclass
class HarnessResult:
    exit_code: int
    stdout: str
    stderr: str
    capabilities_observed: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def _parse_capabilities(stdout: str, stderr: str) -> list[str]:
    """Extract capability observations from openshell log output."""
    caps: list[str] = []
    combined = stdout + "\n" + stderr
    # Network capability pattern: host=<host> port=<port>
    for match in re.finditer(r"host=(\S+)\s+port=(\d+)", combined):
        caps.append(f"network:{match.group(1)}:{match.group(2)}")
    # Filesystem capability pattern: path=<path>
    for match in re.finditer(r"(?:read|write):\s*path=(\S+)", combined):
        caps.append(f"fs:{match.group(1)}")
    return list(dict.fromkeys(caps))  # deduplicate, preserve order


class OpenShellHarness:
    """Runs a command inside an OpenShell sandbox with the given SCP policy.

    If policy is None, a permissive (audit-only) policy is used for discovery.
    """

    def __init__(self, policy: SystemCapabilityProfile | None = None) -> None:
        self.policy = policy

    def run(
        self,
        command: list[str],
        work_dir: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> HarnessResult:
        effective_policy = self.policy or SystemCapabilityProfile.permissive()

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            policy_path = Path(f.name)

        try:
            effective_policy.to_yaml(policy_path)
            cmd = ["openshell", "sandbox", "exec", "--policy", str(policy_path), "--"] + command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir) if work_dir else None,
                env=env,
            )
        finally:
            policy_path.unlink(missing_ok=True)

        return HarnessResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            capabilities_observed=_parse_capabilities(result.stdout, result.stderr),
        )
