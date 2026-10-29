"""The dbt adapter for zarr-datafusion."""

import datetime
import decimal
from typing import Any, Dict, List, Optional

import agate
from dbt.adapters.base.meta import available
from dbt.adapters.sql import SQLAdapter

from dbt.adapters.zarr.column import ZarrColumn
from dbt.adapters.zarr.connections import ZarrConnectionManager


class ZarrAdapter(SQLAdapter):
    ConnectionManager = ZarrConnectionManager
    Column = ZarrColumn

    @classmethod
    def date_function(cls) -> str:
        return "now()"

    @classmethod
    def is_cancelable(cls) -> bool:
        return True

    @classmethod
    def convert_text_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "VARCHAR"

    @classmethod
    def convert_number_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        decimals = agate_table.aggregate(agate.MaxPrecision(col_idx))
        return "DOUBLE" if decimals else "BIGINT"

    @classmethod
    def convert_integer_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "BIGINT"

    @classmethod
    def convert_boolean_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "BOOLEAN"

    @classmethod
    def convert_datetime_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "TIMESTAMP"

    @classmethod
    def convert_date_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "DATE"

    @classmethod
    def convert_time_type(cls, agate_table: agate.Table, col_idx: int) -> str:
        return "TIME"

    def valid_incremental_strategies(self) -> List[str]:
        return ["append"]

    @available
    def seed_create_sql(
        self,
        relation: str,
        agate_table: agate.Table,
        column_types: Optional[Dict[str, str]],
        quote_config: Optional[bool],
    ) -> str:
        """One `CREATE OR REPLACE TABLE ... AS SELECT ... FROM (VALUES ...)`
        holding the whole seed.

        Seeds are small, and a single statement avoids parameter bindings
        (which would need prepared statements), works when the server can't
        see the dbt project's files, and lets the server's catalog journal
        keep the seed across restarts.
        """
        column_types = column_types or {}
        names = list(agate_table.column_names)
        types = [
            column_types.get(name) or self.convert_type(agate_table, idx)
            for idx, name in enumerate(names)
        ]
        quoted = [self.quote_seed_column(name, quote_config) for name in names]

        if not agate_table.rows:
            columns = ", ".join(f"{name} {type_}" for name, type_ in zip(quoted, types))
            return f"create or replace table {relation} ({columns})"

        # VALUES names its columns column1, column2, ...
        select = ", ".join(
            f"cast(column{idx + 1} as {type_}) as {name}"
            for idx, (name, type_) in enumerate(zip(quoted, types))
        )
        rows = ",\n  ".join(
            "(" + ", ".join(_sql_literal(value) for value in row) + ")"
            for row in agate_table.rows
        )
        return f"create or replace table {relation} as\nselect {select}\nfrom (values\n  {rows})"


def _sql_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, decimal.Decimal)):
        return str(value)
    if isinstance(value, datetime.datetime):
        return f"timestamp '{value.isoformat()}'"
    if isinstance(value, datetime.date):
        return f"date '{value.isoformat()}'"
    text = str(value).replace("'", "''")
    return f"'{text}'"
