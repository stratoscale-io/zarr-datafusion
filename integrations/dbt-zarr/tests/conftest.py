"""Fixtures for the dbt adapter test suite, run against a live zarr-flightsql
server (ZARR_FLIGHTSQL_HOST / ZARR_FLIGHTSQL_PORT, default 127.0.0.1:50051)."""

import os

import pytest

pytest_plugins = ["dbt.tests.fixtures.project"]


@pytest.fixture(scope="class")
def dbt_profile_target():
    return {
        "type": "zarr",
        "host": os.getenv("ZARR_FLIGHTSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("ZARR_FLIGHTSQL_PORT", "50051")),
        "token": os.getenv("ZARR_FLIGHTSQL_TOKEN"),
        "threads": 1,
    }
