//! `FlightSqlService` over one shared `SessionContext`.

use std::pin::Pin;
use std::sync::LazyLock;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::sql::metadata::{SqlInfoData, SqlInfoDataBuilder};
use arrow_flight::sql::server::{FlightSqlService, PeekableFlightDataStream};
use arrow_flight::sql::{
    CommandGetSqlInfo, CommandStatementQuery, CommandStatementUpdate, ProstMessageExt, SqlInfo,
    TicketStatementQuery,
};
use arrow_flight::{
    FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket,
};
use datafusion::prelude::SessionContext;
use futures::{stream, Stream, TryStreamExt};
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use super::statement::{execute_update, plan_statement, to_status, StatementTicket};

static SQL_INFO: LazyLock<SqlInfoData> = LazyLock::new(|| {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "zarr-datafusion");
    builder.append(SqlInfo::FlightSqlServerVersion, env!("CARGO_PKG_VERSION"));
    // Arrow IPC format version, see arrow/format/Schema.fbs.
    builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
    builder.append(SqlInfo::FlightSqlServerReadOnly, false);
    builder.build().expect("static SqlInfo is valid")
});

/// Flight SQL service. Every connection shares `ctx`, so all clients (e.g.
/// the several connections of a multi-threaded dbt run) see one catalog.
#[derive(Clone)]
pub struct ZarrFlightSqlService {
    ctx: SessionContext,
}

impl ZarrFlightSqlService {
    pub fn new(ctx: SessionContext) -> Self {
        Self { ctx }
    }
}

type DoGetStream = <ZarrFlightSqlService as FlightService>::DoGetStream;

#[tonic::async_trait]
impl FlightSqlService for ZarrFlightSqlService {
    type FlightService = Self;

    /// Authentication is a bearer-token interceptor in front of every call,
    /// so the handshake has nothing to check; accept it for clients that
    /// always send one.
    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<
        Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>,
        Status,
    > {
        let response = HandshakeResponse::default();
        Ok(Response::new(Box::pin(stream::once(async {
            Ok(response)
        }))))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let planned = plan_statement(&self.ctx, &query.query)
            .await
            .map_err(to_status)?;
        let handle = planned.ticket.encode().map_err(to_status)?;
        let ticket = TicketStatementQuery {
            statement_handle: handle,
        };
        let endpoint =
            FlightEndpoint::new().with_ticket(Ticket::new(ticket.as_any().encode_to_vec()));
        let info = FlightInfo::new()
            .try_with_schema(&planned.schema)
            .map_err(|e| Status::internal(format!("unable to encode schema: {e}")))?
            .with_endpoint(endpoint)
            .with_descriptor(request.into_inner());
        Ok(Response::new(info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        match StatementTicket::decode(&ticket.statement_handle)? {
            StatementTicket::Query(sql) => {
                let df = self.ctx.sql(&sql).await.map_err(to_status)?;
                let batches = df.execute_stream().await.map_err(to_status)?;
                let schema = batches.schema();
                let batches = batches.map_err(|e| FlightError::Tonic(Box::new(to_status(e))));
                let flight = FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .build(batches)
                    .map_err(Status::from);
                Ok(Response::new(Box::pin(flight)))
            }
            StatementTicket::Result(schema, batches) => {
                let flight = FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .build(stream::iter(batches.into_iter().map(Ok)))
                    .map_err(Status::from);
                Ok(Response::new(Box::pin(flight)))
            }
        }
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        _request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        execute_update(&self.ctx, &query.query)
            .await
            .map_err(to_status)
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let info = FlightInfo::new()
            .try_with_schema(query.into_builder(&SQL_INFO).schema().as_ref())
            .map_err(|e| Status::internal(format!("unable to encode schema: {e}")))?
            .with_endpoint(FlightEndpoint::new().with_ticket(ticket))
            .with_descriptor(request.into_inner());
        Ok(Response::new(info))
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let builder = query.into_builder(&SQL_INFO);
        let schema = builder.schema();
        let batch = builder.build();
        let flight = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream::once(async { batch }))
            .map_err(Status::from);
        Ok(Response::new(Box::pin(flight)))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}
