"""Connections to a zarr-flightsql server through ADBC's Flight SQL driver."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_flightsql import DatabaseOptions
from dbt.adapters.contracts.connection import (
    AdapterResponse,
    Connection,
    ConnectionState,
    Credentials,
)
from dbt.adapters.events.logging import AdapterLogger
from dbt.adapters.sql import SQLConnectionManager
from dbt_common.exceptions import DbtDatabaseError, DbtRuntimeError

logger = AdapterLogger("Zarr")

# DataFusion's default catalog; zarr-flightsql serves a single one.
DEFAULT_CATALOG = "datafusion"


@dataclass
class ZarrCredentials(Credentials):
    # Defaults for the base class's required fields: DataFusion's default
    # catalog and schema.
    database: str = DEFAULT_CATALOG
    schema: str = "public"
    host: str = "127.0.0.1"
    port: int = 50051
    tls: bool = False
    token: Optional[str] = None
    retries: int = 1

    _ALIASES = {"catalog": "database"}

    @property
    def type(self) -> str:
        return "zarr"

    @property
    def unique_field(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def uri(self) -> str:
        scheme = "grpc+tls" if self.tls else "grpc"
        return f"{scheme}://{self.host}:{self.port}"

    def _connection_keys(self) -> Tuple[str, ...]:
        return ("host", "port", "tls", "database", "schema")


class ZarrConnectionManager(SQLConnectionManager):
    TYPE = "zarr"

    @contextmanager
    def exception_handler(self, sql: str):
        try:
            yield
        except flight_sql.Error as e:
            logger.debug(f"Error running SQL: {sql}")
            raise DbtDatabaseError(str(e).strip()) from e
        except DbtRuntimeError:
            raise
        except Exception as e:
            logger.debug(f"Error running SQL: {sql}")
            raise DbtRuntimeError(str(e)) from e

    @classmethod
    def open(cls, connection: Connection) -> Connection:
        if connection.state == ConnectionState.OPEN:
            return connection

        credentials: ZarrCredentials = connection.credentials
        db_kwargs = {}
        if credentials.token:
            db_kwargs[DatabaseOptions.AUTHORIZATION_HEADER.value] = f"Bearer {credentials.token}"

        def connect():
            # The server has no transactions, so stay in autocommit mode.
            return flight_sql.connect(credentials.uri, db_kwargs=db_kwargs, autocommit=True)

        return cls.retry_connection(
            connection,
            connect=connect,
            logger=logger,
            retryable_exceptions=[flight_sql.OperationalError],
            retry_limit=credentials.retries,
        )

    def cancel(self, connection: Connection) -> None:
        connection.handle.close()

    @classmethod
    def get_response(cls, cursor: Any) -> AdapterResponse:
        # ADBC reports -1 when the row count is unknown (e.g. for DDL).
        rows = cursor.rowcount
        return AdapterResponse(_message="OK", rows_affected=rows if rows >= 0 else None)

    # dbt still tracks transactions (materializations call adapter.commit()),
    # but the server has none, so nothing is sent for BEGIN / COMMIT.
    def add_begin_query(self):
        pass

    def add_commit_query(self):
        pass

    @classmethod
    def data_type_code_to_name(cls, type_code: Any) -> str:
        # ADBC's cursor.description reports pyarrow types.
        return str(type_code)
