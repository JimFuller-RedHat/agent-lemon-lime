"""Shared pytest fixtures for argus test suite."""
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def tmp_policy_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid SCP YAML and return its path."""
    policy = {
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
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.dump(policy))
    return p
