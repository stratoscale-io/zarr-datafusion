//! Arrow Flight SQL server over a Zarr `SessionContext`.
//!
//! Lets any Flight SQL client (ADBC, the Flight SQL JDBC driver, dbt through
//! a thin adapter) run SQL against Zarr stores and receive Arrow batches,
//! with no conversion of the store. See docs/dbt-flightsql-implementation-plan.md.
//!
//! ```no_run
//! use tonic::transport::Server;
//! use zarr_datafusion::flightsql::flight_sql_service;
//! use zarr_datafusion::session::{build_session_context, zarr_session_config};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = build_session_context(zarr_session_config())?;
//! Server::builder()
//!     .add_service(flight_sql_service(ctx, None))
//!     .serve("127.0.0.1:50051".parse()?)
//!     .await?;
//! # Ok(())
//! # }
//! ```

mod service;
mod statement;

use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionContext;
use tonic::service::interceptor::InterceptedService;
use tonic::service::Interceptor;
use tonic::{Request, Status};

pub use service::ZarrFlightSqlService;

/// Rejects calls without `authorization: Bearer <token>` when a token is set.
#[derive(Clone)]
pub struct BearerAuth {
    expected: Option<String>,
}

impl BearerAuth {
    pub fn new(token: Option<String>) -> Self {
        Self {
            expected: token.map(|t| format!("Bearer {t}")),
        }
    }
}

impl Interceptor for BearerAuth {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(expected) = &self.expected else {
            return Ok(request);
        };
        match request.metadata().get("authorization") {
            Some(value) if value.as_bytes() == expected.as_bytes() => Ok(request),
            _ => Err(Status::unauthenticated("missing or invalid bearer token")),
        }
    }
}

/// The Flight SQL gRPC service for `ctx`, behind optional bearer-token auth.
/// Add it to a `tonic::transport::Server`.
pub fn flight_sql_service(
    ctx: SessionContext,
    token: Option<String>,
) -> InterceptedService<FlightServiceServer<ZarrFlightSqlService>, BearerAuth> {
    FlightServiceServer::with_interceptor(ZarrFlightSqlService::new(ctx), BearerAuth::new(token))
}
