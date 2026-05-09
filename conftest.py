"""
conftest.py

Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import os
import pytest

# Ensure no real API calls in tests
os.environ.setdefault(
    "GROQ_API_KEY",
    "gsk-test-key-not-real"
)

os.environ.setdefault(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)


def pytest_configure(config):
    """Register custom markers."""

    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )

    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow"
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip integration tests unless
    --integration flag is passed.
    """

    if not config.getoption("--integration", default=False):

        skip_integration = pytest.mark.skip(
            reason="Use --integration to run"
        )

        for item in items:

            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):

    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires real GROQ_API_KEY)",
    )