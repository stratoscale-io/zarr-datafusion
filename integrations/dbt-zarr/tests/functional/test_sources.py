"""zarr_source: registering a Zarr store from on-run-start.

Store paths are resolved by the server: run it from the zarr-datafusion
repository root (or set ZARR_TEST_DATA to the data directory as the server
sees it), with the test data generated (scripts/generate_data.sh).
"""

import os

import pytest
from dbt.tests.util import run_dbt

DATA = os.getenv("ZARR_TEST_DATA", "data")

sources_yml = """
version: 2
sources:
  - name: raw
    schema: "{{ target.schema }}"
    tables:
      - name: weather
"""

weather_rows_sql = "select count(*) as n from {{ source('raw', 'weather') }}"


class TestZarrSource:
    @pytest.fixture(scope="class")
    def project_config_update(self):
        return {
            "on-run-start": ["{{ zarr_source('weather', var('location')) }}"],
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {"sources.yml": sources_yml, "weather_rows.sql": weather_rows_sql}

    def rows(self, project):
        relation = f"{project.database}.{project.test_schema}.weather_rows"
        return project.run_sql(f"select n from {relation}", fetch="one")[0]

    def test_location_change_takes_effect(self, project):
        # synthetic: time(7) x lat(10) x lon(10)
        run_dbt(["run", "--vars", f"location: {DATA}/synthetic_v3.zarr"])
        assert self.rows(project) == 700

        # a re-run with another store reads the new store
        run_dbt(["run", "--vars", f"location: {DATA}/era5_v3.zarr"])
        assert self.rows(project) != 700
