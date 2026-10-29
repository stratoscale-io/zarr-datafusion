//! Arrow Flight SQL server for Zarr stores.
//!
//! Serves one shared session to every client (ADBC, JDBC, dbt). Tables are
//! registered with SQL, either by clients or from `--init-sql` files run at
//! startup.
//!
//! Usage:
//!   zarr-flightsql [--addr 127.0.0.1:50051] [--token SECRET] [--init-sql FILE]...
//!
//! The token can also come from `ZARR_FLIGHTSQL_TOKEN`. Clients then send
//! `authorization: Bearer SECRET`.

use std::error::Error;
use std::net::SocketAddr;

use datafusion::prelude::SessionContext;
use datafusion::sql::parser::DFParser;
use tonic::transport::Server;

use zarr_datafusion::flightsql::flight_sql_service;
use zarr_datafusion::session::{build_session_context, zarr_session_config};

struct Args {
    addr: SocketAddr,
    token: Option<String>,
    init_sql: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut addr = "127.0.0.1:50051".to_string();
    let mut token = std::env::var("ZARR_FLIGHTSQL_TOKEN").ok();
    let mut init_sql = Vec::new();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--addr" => addr = it.next().ok_or("--addr needs a value")?,
            "--token" => token = Some(it.next().ok_or("--token needs a value")?),
            "--init-sql" => init_sql.push(it.next().ok_or("--init-sql needs a value")?),
            "-h" | "--help" => return Err(
                "usage: zarr-flightsql [--addr HOST:PORT] [--token SECRET] [--init-sql FILE]..."
                    .into(),
            ),
            other => return Err(format!("unexpected argument: {other}")),
        }
    }

    Ok(Args {
        addr: addr
            .parse()
            .map_err(|e| format!("invalid --addr {addr}: {e}"))?,
        token,
        init_sql,
    })
}

/// Run every statement in `path`, in order, failing on the first error.
async fn run_init_sql(ctx: &SessionContext, path: &str) -> Result<(), Box<dyn Error>> {
    let sql = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    for statement in DFParser::parse_sql(&sql)? {
        let plan = ctx.state().statement_to_plan(statement).await?;
        ctx.execute_logical_plan(plan).await?.collect().await?;
    }
    tracing::info!(path, "ran init SQL");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let ctx = build_session_context(zarr_session_config())?;
    for path in &args.init_sql {
        run_init_sql(&ctx, path).await?;
    }

    println!(
        "zarr-flightsql listening on grpc://{}{}",
        args.addr,
        if args.token.is_some() {
            " (bearer auth)"
        } else {
            ""
        }
    );
    Server::builder()
        .add_service(flight_sql_service(ctx, args.token))
        .serve(args.addr)
        .await?;
    Ok(())
}
