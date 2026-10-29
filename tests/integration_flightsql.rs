//! Integration tests for the Flight SQL server (`zarr_datafusion::flightsql`),
//! over a real gRPC connection on an ephemeral port.

mod common;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use arrow_flight::error::FlightError;
use arrow_flight::sql::client::FlightSqlServiceClient;
use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt, SqlInfo};
use arrow_flight::{FlightDescriptor, FlightInfo};
use common::*;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use prost::Message;
use tonic::transport::server::TcpIncoming;
use tonic::transport::{Channel, Endpoint, Server};
use zarr_datafusion::flightsql::{flight_sql_service, CatalogManager};
use zarr_datafusion::session::{build_session_context, zarr_session_config};

/// Start a server with `weather` (synthetic v3) registered. Returns its
/// address and the session it serves, for comparing against local results.
async fn start_server(token: Option<&str>) -> (SocketAddr, SessionContext) {
    start_server_with(token, None).await
}

/// Like [`start_server`], optionally with a catalog journal at `journal`.
async fn start_server_with(
    token: Option<&str>,
    journal: Option<&Path>,
) -> (SocketAddr, SessionContext) {
    let ctx = build_session_context(zarr_session_config()).unwrap();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION '{SYNTHETIC_V3}'"
    ))
    .await
    .unwrap();

    let incoming = TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = incoming.local_addr().unwrap();
    let catalog = match journal {
        Some(path) => CatalogManager::open(ctx.clone(), path).await.unwrap(),
        None => CatalogManager::new(ctx.clone()),
    };
    let service = flight_sql_service(Arc::new(catalog), token.map(String::from));
    tokio::spawn(
        Server::builder()
            .add_service(service)
            .serve_with_incoming(incoming),
    );
    (addr, ctx)
}

async fn connect(addr: SocketAddr) -> FlightSqlServiceClient<Channel> {
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    FlightSqlServiceClient::new(channel)
}

/// `GetFlightInfo` + `DoGet` on every endpoint, as ADBC does for `execute()`.
/// Returns the schema advertised in `FlightInfo` and the batches received.
async fn run_sql(
    client: &mut FlightSqlServiceClient<Channel>,
    sql: &str,
) -> Result<(Schema, Vec<RecordBatch>), FlightError> {
    let info = client.execute(sql.to_string(), None).await?;
    fetch(client, info).await
}

async fn fetch(
    client: &mut FlightSqlServiceClient<Channel>,
    info: FlightInfo,
) -> Result<(Schema, Vec<RecordBatch>), FlightError> {
    let schema = info.clone().try_decode_schema()?;
    let mut batches = Vec::new();
    for endpoint in info.endpoint {
        let stream = client.do_get(endpoint.ticket.unwrap()).await?;
        let mut received: Vec<RecordBatch> = stream.try_collect().await?;
        batches.append(&mut received);
    }
    Ok((schema, batches))
}

async fn local_sql(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

fn pretty(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).unwrap().to_string()
}

#[tokio::test]
async fn test_flightsql_select_matches_local() {
    let (addr, ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    let sql = "SELECT lat, SUM(temperature) AS s FROM weather WHERE time = 3 \
               GROUP BY lat ORDER BY lat";
    let (_, remote) = run_sql(&mut client, sql).await.unwrap();
    assert_eq!(pretty(&remote), pretty(&local_sql(&ctx, sql).await));
}

/// Coordinates are dictionary-encoded; the Flight encoder hydrates them, so
/// the advertised schema must match what is actually streamed.
#[tokio::test]
async fn test_flightsql_schema_matches_stream_for_dictionary_coords() {
    let (addr, _ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    let (schema, batches) = run_sql(
        &mut client,
        "SELECT time, lat, lon, temperature FROM weather WHERE time = 3",
    )
    .await
    .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 100);
    assert_eq!(&schema, batches[0].schema().as_ref());
    assert!(schema
        .fields()
        .iter()
        .all(|f| !matches!(f.data_type(), DataType::Dictionary(..))));
}

/// DDL runs during `GetFlightInfo`, exactly once: if `DoGet` ran it again,
/// the plain `CREATE VIEW` would fail with "already exists".
#[tokio::test]
async fn test_flightsql_ddl_runs_once() {
    let (addr, _ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    run_sql(
        &mut client,
        "CREATE VIEW warm AS SELECT time, lat, lon FROM weather WHERE temperature > 20",
    )
    .await
    .unwrap();

    let (_, batches) = run_sql(
        &mut client,
        "SELECT COUNT(*) AS n FROM information_schema.tables \
         WHERE table_name = 'warm' AND table_type = 'VIEW'",
    )
    .await
    .unwrap();
    assert_eq!(get_scalar_i64(&batches[0]), 1);
}

#[tokio::test]
async fn test_flightsql_invalid_sql_is_invalid_argument() {
    let (addr, _ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    let cmd = CommandStatementQuery {
        query: "SELECT * FROM no_such_table".to_string(),
        transaction_id: None,
    };
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let status = client
        .inner_mut()
        .get_flight_info(descriptor)
        .await
        .unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    assert!(status.message().contains("no_such_table"), "{status:?}");
}

#[tokio::test]
async fn test_flightsql_clients_share_catalog() {
    let (addr, _ctx) = start_server(None).await;
    let mut a = connect(addr).await;
    let mut b = connect(addr).await;

    run_sql(
        &mut a,
        "CREATE VIEW t3 AS SELECT * FROM weather WHERE time = 3",
    )
    .await
    .unwrap();
    let (_, batches) = run_sql(&mut b, "SELECT COUNT(*) AS n FROM t3")
        .await
        .unwrap();
    assert_eq!(get_scalar_i64(&batches[0]), 100);
}

/// `CommandStatementUpdate` (`execute_update`) returns affected rows.
#[tokio::test]
async fn test_flightsql_execute_update_returns_row_count() {
    let (addr, _ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    let created = client
        .execute_update("CREATE TABLE t (x INT)".to_string(), None)
        .await
        .unwrap();
    assert_eq!(created, 0);
    let inserted = client
        .execute_update("INSERT INTO t VALUES (1), (2), (3)".to_string(), None)
        .await
        .unwrap();
    assert_eq!(inserted, 3);
}

#[tokio::test]
async fn test_flightsql_bearer_auth() {
    let (addr, _ctx) = start_server(Some("s3cret")).await;

    let mut anonymous = connect(addr).await;
    let err = run_sql(&mut anonymous, "SELECT 1").await.unwrap_err();
    assert!(err.to_string().contains("bearer token"), "{err}");

    let mut wrong = connect(addr).await;
    wrong.set_token("nope".to_string());
    assert!(run_sql(&mut wrong, "SELECT 1").await.is_err());

    let mut authed = connect(addr).await;
    authed.set_token("s3cret".to_string());
    run_sql(&mut authed, "SELECT 1").await.unwrap();
}

#[tokio::test]
async fn test_flightsql_sql_info() {
    let (addr, _ctx) = start_server(None).await;
    let mut client = connect(addr).await;

    let info = client
        .get_sql_info(vec![SqlInfo::FlightSqlServerName])
        .await
        .unwrap();
    let (_, batches) = fetch(&mut client, info).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert!(pretty(&batches).contains("zarr-datafusion"));
}

/// With a journal, views created by clients are back after a restart.
#[tokio::test]
async fn test_flightsql_journal_survives_restart() {
    let dir = std::env::temp_dir().join(format!("zarr-flightsql-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let journal = dir.join("catalog.jsonl");
    let _ = std::fs::remove_file(&journal);

    let (addr, _ctx) = start_server_with(None, Some(&journal)).await;
    let mut client = connect(addr).await;
    run_sql(
        &mut client,
        "CREATE VIEW t3 AS SELECT * FROM weather WHERE time = 3",
    )
    .await
    .unwrap();

    // A second server on the same journal, as after a restart.
    let (addr, _ctx) = start_server_with(None, Some(&journal)).await;
    let mut client = connect(addr).await;
    let (_, batches) = run_sql(&mut client, "SELECT COUNT(*) AS n FROM t3")
        .await
        .unwrap();
    assert_eq!(get_scalar_i64(&batches[0]), 100);

    std::fs::remove_dir_all(&dir).unwrap();
}
