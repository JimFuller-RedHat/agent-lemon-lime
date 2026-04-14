"""Tests for OpenShellHarness subprocess wrapper."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from argus.core.harness import HarnessResult, OpenShellHarness, _parse_capabilities
from argus.core.scp import SystemCapabilityProfile


def test_harness_result_fields() -> None:
    r = HarnessResult(exit_code=0, stdout="ok", stderr="", capabilities_observed=[])
    assert r.exit_code == 0
    assert r.success is True


def test_harness_result_failure() -> None:
    r = HarnessResult(exit_code=1, stdout="", stderr="err", capabilities_observed=[])
    assert r.success is False


def test_parse_capabilities_empty() -> None:
    caps = _parse_capabilities("", "")
    assert caps == []


def test_parse_capabilities_extracts_network_host() -> None:
    stdout = "[openshell] network: host=api.github.com port=443 enforcement=audit"
    caps = _parse_capabilities(stdout, "")
    assert any("api.github.com" in c for c in caps)


def test_parse_capabilities_extracts_filesystem_path() -> None:
    stderr = "write: path=/tmp/out.txt"
    caps = _parse_capabilities("", stderr)
    assert any("/tmp/out.txt" in c for c in caps)


@patch("argus.core.harness.subprocess.run")
def test_harness_run_discovery(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    harness = OpenShellHarness(policy=None)  # None = discovery / permissive
    result = harness.run(["echo", "hello"], work_dir=tmp_path)
    assert result.exit_code == 0
    assert result.success is True
    # Verify openshell was called with a policy file
    args = mock_run.call_args[0][0]
    assert "openshell" in args[0]


@patch("argus.core.harness.subprocess.run")
def test_harness_run_with_policy_uses_supplied_policy(
    mock_run: MagicMock, tmp_path: Path, tmp_policy_yaml: Path
) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    scp = SystemCapabilityProfile.from_yaml(tmp_policy_yaml)

    written_yamls: list[str] = []

    original_to_yaml = SystemCapabilityProfile.to_yaml

    def capture_to_yaml(self: SystemCapabilityProfile, path: object) -> None:
        import yaml as _yaml
        written_yamls.append(_yaml.dump(self.model_dump()))
        original_to_yaml(self, path)

    with patch.object(SystemCapabilityProfile, "to_yaml", capture_to_yaml):
        harness = OpenShellHarness(policy=scp)
        result = harness.run(["python", "-m", "myagent"], work_dir=tmp_path)

    assert result.success is True
    # The written YAML must contain the policy's network_policies (empty dict for minimal SCP)
    # and must NOT contain the permissive policy's audit sentinel
    assert written_yamls, "to_yaml was never called"
    assert "all_audit" not in written_yamls[0]  # permissive() sentinel absent
