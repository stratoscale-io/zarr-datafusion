//! Shared `SessionContext` construction.
//!
//! One place that wires a DataFusion session for Zarr: the `ZARR` table
//! factory, the Zarr optimizer rules, remote object stores for non-Zarr
//! tables, `COPY TO ... STORED AS ZARR`, and the Zarr UDFs/UDTFs. Used by
//! `zarr-cli` and the Flight SQL server so both behave the same.
//!
//! ```no_run
//! use zarr_datafusion::session::{build_session_context, zarr_session_config};
//!
//! let ctx = build_session_context(zarr_session_config().with_batch_size(16384))?;
//! # Ok::<(), datafusion::error::DataFusionError>(())
//! ```

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::datasource::factory::ZarrTableFactory;
use crate::datasource::object_store_registry::RemoteObjectStoreRegistry;
use crate::optimizer::{
    CardinalityRule, CountStatisticsRule, MinMaxStatisticsRule, ZarrLimitPushdownRule,
};
use crate::udfs::register_metric_udfs;
use crate::udtf::register_zarr_functions;
use crate::writer::register_zarr_write_format;

/// Default config for Zarr sessions: `information_schema` enabled, so
/// `SHOW TABLES` and catalog introspection (e.g. dbt's) work.
pub fn zarr_session_config() -> SessionConfig {
    SessionConfig::new().with_information_schema(true)
}

/// Build a `SessionState` with the Zarr table factory and optimizer rules.
///
/// Function registrations live on the context, so prefer
/// [`build_session_context`] unless you need to customize the state further.
pub fn build_session_state(config: SessionConfig) -> Result<SessionState> {
    // Lets Parquet/CSV/JSON tables use http(s)://, gs:// and s3:// locations
    // (Zarr tables open their own stores).
    let runtime = RuntimeEnvBuilder::new()
        .with_object_store_registry(Arc::new(RemoteObjectStoreRegistry::new()))
        .build_arc()?;
    Ok(SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_table_factory("ZARR".to_string(), Arc::new(ZarrTableFactory) as _)
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))
        .with_physical_optimizer_rule(Arc::new(CardinalityRule::new()))
        .build())
}

/// Build a ready-to-use `SessionContext`: [`build_session_state`] plus
/// `COPY TO ... STORED AS ZARR`, `zarr_describe()`, and the metric UDFs.
pub fn build_session_context(config: SessionConfig) -> Result<SessionContext> {
    let ctx = SessionContext::new_with_state(build_session_state(config)?);
    register_zarr_write_format(&ctx)?;
    register_zarr_functions(&ctx);
    register_metric_udfs(&ctx);
    Ok(ctx)
}
