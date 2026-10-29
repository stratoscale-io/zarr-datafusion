"""Offline tests for the adapter's pure-Python pieces."""

import datetime
import decimal

import agate
import pytest

from dbt.adapters.zarr.column import ZarrColumn
from dbt.adapters.zarr.impl import _sql_literal


@pytest.mark.parametrize(
    "value, literal",
    [
        (None, "null"),
        (True, "true"),
        (decimal.Decimal("24.72"), "24.72"),
        (7, "7"),
        ("DJF", "'DJF'"),
        ("it's", "'it''s'"),
        (datetime.date(2025, 12, 15), "date '2025-12-15'"),
        (datetime.datetime(2025, 12, 15, 12, 0), "timestamp '2025-12-15T12:00:00'"),
    ],
)
def test_sql_literal(value, literal):
    assert _sql_literal(value) == literal


@pytest.mark.parametrize(
    "dtype, value_type",
    [
        ("Int64", "Int64"),
        ("Dictionary(Int16, Float32)", "Float32"),
        ('Dictionary(Int16, Timestamp(µs, "UTC"))', 'Timestamp(µs, "UTC")'),
    ],
)
def test_column_unwraps_dictionary_types(dtype, value_type):
    assert ZarrColumn("c", dtype).dtype == value_type


def test_column_type_predicates():
    assert ZarrColumn("c", "Dictionary(Int16, Int64)").is_integer()
    assert ZarrColumn("c", "Float32").is_float()
    assert ZarrColumn("c", "Float32").is_number()
    assert ZarrColumn("c", "Utf8").is_string()
    assert not ZarrColumn("c", "Utf8").is_number()


def test_seed_sql_is_one_statement():
    from dbt.adapters.zarr.impl import ZarrAdapter

    table = agate.Table(
        [(decimal.Decimal("1950"), "DJF", decimal.Decimal("-1.53"))],
        ["yr", "seas", "anom"],
        [agate.Number(), agate.Text(), agate.Number()],
    )
    adapter = ZarrAdapter.__new__(ZarrAdapter)  # no config needed for this method
    sql = ZarrAdapter.seed_create_sql(adapter, '"datafusion"."s"."oni"', table, {}, False)
    assert sql.startswith('create or replace table "datafusion"."s"."oni" as')
    assert "cast(column1 as BIGINT) as yr" in sql
    assert "cast(column3 as DOUBLE) as anom" in sql
    assert "(1950, 'DJF', -1.53)" in sql
