//! Planning, executing and ticket encoding for SQL statements.
//!
//! Clients such as ADBC's DB-API send every statement, DDL included, as a
//! `CommandStatementQuery`: `GetFlightInfo` first, then `DoGet` on the
//! returned ticket. A DDL/DML statement must take effect exactly once, so it
//! runs during `GetFlightInfo` and its (tiny) result is IPC-encoded into the
//! ticket itself. A query is only planned there; its ticket carries the SQL
//! and `DoGet` plans and runs it. Either way the server keeps no per-ticket
//! state.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use tonic::Status;

const QUERY_TAG: u8 = b'q';
const RESULT_TAG: u8 = b'r';

/// The statement handle carried in a `TicketStatementQuery`.
#[derive(Debug)]
pub(crate) enum StatementTicket {
    /// A query to plan and run on `DoGet`.
    Query(String),
    /// The already-computed result of a DDL/DML statement.
    Result(SchemaRef, Vec<RecordBatch>),
}

impl StatementTicket {
    pub(crate) fn encode(&self) -> Result<Bytes> {
        let mut buf = Vec::new();
        match self {
            Self::Query(sql) => {
                buf.push(QUERY_TAG);
                buf.extend_from_slice(sql.as_bytes());
            }
            Self::Result(schema, batches) => {
                buf.push(RESULT_TAG);
                let mut writer = StreamWriter::try_new(&mut buf, schema)?;
                for batch in batches {
                    writer.write(batch)?;
                }
                writer.finish()?;
            }
        }
        Ok(buf.into())
    }

    pub(crate) fn decode(handle: &[u8]) -> Result<Self, Status> {
        match handle.split_first() {
            Some((&QUERY_TAG, sql)) => std::str::from_utf8(sql)
                .map(|s| Self::Query(s.to_string()))
                .map_err(|e| Status::invalid_argument(format!("invalid ticket: {e}"))),
            Some((&RESULT_TAG, ipc)) => {
                let reader = StreamReader::try_new(ipc, None)
                    .map_err(|e| Status::invalid_argument(format!("invalid ticket: {e}")))?;
                let schema = reader.schema();
                let batches = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| Status::invalid_argument(format!("invalid ticket: {e}")))?;
                Ok(Self::Result(schema, batches))
            }
            _ => Err(Status::invalid_argument("invalid ticket")),
        }
    }
}

/// A statement after `GetFlightInfo`: the schema `DoGet` will stream, and
/// the ticket to fetch it with.
#[derive(Debug)]
pub(crate) struct PlannedStatement {
    pub schema: SchemaRef,
    pub ticket: StatementTicket,
}

/// Plan `sql`; run it right away if it changes state (DDL, DML, `COPY`,
/// `SET`), otherwise defer execution to `DoGet`.
pub(crate) async fn plan_statement(ctx: &SessionContext, sql: &str) -> Result<PlannedStatement> {
    let plan = ctx.state().create_logical_plan(sql).await?;
    if has_side_effects(&plan) {
        let df = ctx.execute_logical_plan(plan).await?;
        let schema: SchemaRef = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;
        return Ok(PlannedStatement {
            schema: schema.clone(),
            ticket: StatementTicket::Result(schema, batches),
        });
    }
    Ok(PlannedStatement {
        schema: hydrate_schema(plan.schema().as_arrow()),
        ticket: StatementTicket::Query(sql.to_string()),
    })
}

/// Run `sql` to completion and return the number of rows it affected
/// (the `count` column DataFusion returns for DML/`COPY`; 0 for DDL).
pub(crate) async fn execute_update(ctx: &SessionContext, sql: &str) -> Result<i64> {
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut count = 0;
    for batch in &batches {
        if let Some(col) = batch.column_by_name("count") {
            let col = arrow::compute::cast(col, &DataType::Int64)?;
            count += arrow::compute::sum(
                col.as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .ok_or_else(|| DataFusionError::Internal("count column is not Int64".into()))?,
            )
            .unwrap_or(0);
        }
    }
    Ok(count)
}

fn has_side_effects(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Ddl(_)
            | LogicalPlan::Dml(_)
            | LogicalPlan::Copy(_)
            | LogicalPlan::Statement(_)
    )
}

/// The schema the Flight encoder actually sends: it hydrates dictionary
/// columns (our coordinates) to their value type by default, so the schema
/// advertised in `FlightInfo` must match.
pub(crate) fn hydrate_schema(schema: &Schema) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Dictionary(_, value) => {
                f.as_ref().clone().with_data_type(value.as_ref().clone())
            }
            _ => f.as_ref().clone(),
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// Map a DataFusion error to a gRPC status, keeping the full message so it
/// reaches the client (e.g. dbt's log) intact.
pub(crate) fn to_status(e: DataFusionError) -> Status {
    let message = e.to_string();
    match e.find_root() {
        DataFusionError::Plan(_)
        | DataFusionError::SQL(..)
        | DataFusionError::SchemaError(..)
        | DataFusionError::Configuration(_) => Status::invalid_argument(message),
        DataFusionError::NotImplemented(_) => Status::unimplemented(message),
        _ => Status::internal(message),
    }
}
