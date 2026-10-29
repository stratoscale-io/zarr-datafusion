"""dbt's incremental tests, for the `append` strategy."""

from dbt.tests.adapter.basic.test_incremental import (
    BaseIncremental,
    BaseIncrementalNotSchemaChange,
)


class TestIncrementalZarr(BaseIncremental):
    pass


class TestIncrementalNotSchemaChangeZarr(BaseIncrementalNotSchemaChange):
    pass
