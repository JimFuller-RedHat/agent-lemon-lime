"""Shared pytest fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests requiring an OpenShell cluster"
    )
