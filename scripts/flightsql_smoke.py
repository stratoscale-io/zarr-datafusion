"""Smoke test for zarr-flightsql through ADBC's Python DB-API, the client a
dbt adapter would use.

Start a server from the repo root, then run:

    cargo run --features flightsql --bin zarr-flightsql
    uv run --with adbc-driver-flightsql --with pyarrow scripts/flightsql_smoke.py

Options: --uri (default grpc://127.0.0.1:50051), --token, --store (path as
seen by the server, default data/synthetic_v3.zarr).
"""

import argparse
import uuid

import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_flightsql import DatabaseOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="grpc://127.0.0.1:50051")
    parser.add_argument("--token")
    parser.add_argument("--store", default="data/synthetic_v3.zarr")
    args = parser.parse_args()

    db_kwargs = {}
    if args.token:
        db_kwargs[DatabaseOptions.AUTHORIZATION_HEADER.value] = f"Bearer {args.token}"

    # Unique names so the script can rerun against a long-lived server.
    suffix = uuid.uuid4().hex[:8]
    table, view = f"smoke_{suffix}", f"smoke_view_{suffix}"

    # The server has no transactions, so stay in autocommit mode (as a dbt
    # adapter must too); otherwise ADBC warns that it cannot disable it.
    with (
        flight_sql.connect(args.uri, db_kwargs=db_kwargs, autocommit=True) as conn,
        conn.cursor() as cur,
    ):
        # DDL through cursor.execute(), the way dbt sends it.
        cur.execute(f"CREATE EXTERNAL TABLE {table} STORED AS ZARR LOCATION '{args.store}'")
        cur.fetchall()
        cur.execute(
            f"CREATE VIEW {view} AS SELECT time, lat, AVG(temperature) AS avg_t "
            f"FROM {table} GROUP BY time, lat"
        )
        cur.fetchall()

        # dbt's relation cache query.
        cur.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            f"WHERE table_name IN ('{table}', '{view}') ORDER BY table_name"
        )
        relations = cur.fetchall()
        print("relations:", relations)
        assert relations == [(table, "BASE TABLE"), (view, "VIEW")], relations

        # A filtered query through the view, as a downstream model would run.
        cur.execute(f"SELECT COUNT(*), MIN(time), MAX(time) FROM {view} WHERE time = 3")
        row = cur.fetchone()
        print("view where time = 3:", row)
        assert row == (10, 3, 3), row

        # Coordinates arrive hydrated (no dictionary types) as Arrow.
        cur.execute(f"SELECT time, lat, lon, temperature FROM {table} WHERE time = 3")
        arrow_table = cur.fetch_arrow_table()
        print("arrow schema:", arrow_table.schema)
        assert arrow_table.num_rows == 100

        # Errors carry DataFusion's message.
        try:
            cur.execute("SELECT * FROM no_such_table")
            cur.fetchall()
        except flight_sql.Error as e:
            print("error:", str(e).splitlines()[0])
            assert "no_such_table" in str(e)
        else:
            raise AssertionError("expected an error for a missing table")

        cur.execute(f"DROP VIEW {view}")
        cur.fetchall()
        cur.execute(f"DROP TABLE {table}")
        cur.fetchall()

    print("OK")


if __name__ == "__main__":
    main()
