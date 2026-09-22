//! Object store registry that builds remote stores on first use.
//!
//! Zarr tables open their own stores (see [`crate::reader::storage`]), but
//! DataFusion's built-in formats (Parquet, CSV, JSON) resolve a `LOCATION`
//! through the session's [`ObjectStoreRegistry`], whose default only knows
//! `file://`. This registry fills that gap so a Parquet lookup table on
//! Hugging Face (`https://`), GCS (`gs://`) or S3 (`s3://`) can be joined
//! against a Zarr store in one query:
//!
//! ```sql
//! CREATE EXTERNAL TABLE postcodes STORED AS PARQUET
//!   LOCATION 'https://huggingface.co/datasets/<org>/<repo>/resolve/main/x.parquet';
//! ```
//!
//! Credentials follow the Zarr remote path: GCS tries anonymous access first,
//! S3 reads the standard `AWS_*` environment, HTTP is anonymous.

use std::sync::Arc;

use datafusion::common::{DataFusionError, Result};
use datafusion::execution::object_store::{DefaultObjectStoreRegistry, ObjectStoreRegistry};
use tracing::debug;
use url::Url;
use zarrs_object_store::object_store::aws::AmazonS3Builder;
use zarrs_object_store::object_store::gcp::GoogleCloudStorageBuilder;
use zarrs_object_store::object_store::http::HttpBuilder;
use zarrs_object_store::object_store::ObjectStore;

/// [`ObjectStoreRegistry`] that lazily creates and caches `http(s)`, `gs` and
/// `s3` stores, keyed like DataFusion's default (scheme + host + port).
#[derive(Debug, Default)]
pub struct RemoteObjectStoreRegistry {
    inner: DefaultObjectStoreRegistry,
}

impl RemoteObjectStoreRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ObjectStoreRegistry for RemoteObjectStoreRegistry {
    fn register_store(
        &self,
        url: &Url,
        store: Arc<dyn ObjectStore>,
    ) -> Option<Arc<dyn ObjectStore>> {
        self.inner.register_store(url, store)
    }

    fn deregister_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        self.inner.deregister_store(url)
    }

    fn get_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
        if let Ok(store) = self.inner.get_store(url) {
            return Ok(store);
        }
        let store = build_remote_store(url)?;
        debug!(url = %url, "Registered remote object store");
        self.inner.register_store(url, Arc::clone(&store));
        Ok(store)
    }
}

/// Build a store for the scheme + host of `url`.
fn build_remote_store(url: &Url) -> Result<Arc<dyn ObjectStore>> {
    let host = url
        .host_str()
        .ok_or_else(|| DataFusionError::Configuration(format!("Missing host in URL: {url}")))?;
    let store: Arc<dyn ObjectStore> = match url.scheme() {
        "http" | "https" => {
            let base = match url.port() {
                Some(port) => format!("{}://{host}:{port}", url.scheme()),
                None => format!("{}://{host}", url.scheme()),
            };
            Arc::new(HttpBuilder::new().with_url(base).build()?)
        }
        "gs" => {
            // Public buckets first (no credential lookup), then the environment.
            let anonymous = GoogleCloudStorageBuilder::new()
                .with_bucket_name(host)
                .with_skip_signature(true)
                .build();
            match anonymous {
                Ok(store) => Arc::new(store),
                Err(_) => Arc::new(
                    GoogleCloudStorageBuilder::from_env()
                        .with_bucket_name(host)
                        .build()?,
                ),
            }
        }
        "s3" => Arc::new(AmazonS3Builder::from_env().with_bucket_name(host).build()?),
        scheme => {
            return Err(DataFusionError::Configuration(format!(
                "No object store for scheme '{scheme}' ({url}); supported: file, http(s), gs, s3"
            )))
        }
    };
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_caches_http_store() {
        let registry = RemoteObjectStoreRegistry::new();
        let url = Url::parse("https://huggingface.co/datasets/a/b/resolve/main/x.parquet").unwrap();
        let first = registry.get_store(&url).unwrap();
        let other = Url::parse("https://huggingface.co/other/path").unwrap();
        let second = registry.get_store(&other).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "same host must reuse the store"
        );
    }

    #[test]
    fn local_files_still_resolve() {
        let registry = RemoteObjectStoreRegistry::new();
        assert!(registry
            .get_store(&Url::parse("file:///tmp").unwrap())
            .is_ok());
    }

    #[test]
    fn unknown_scheme_is_an_error() {
        let registry = RemoteObjectStoreRegistry::new();
        assert!(registry
            .get_store(&Url::parse("ftp://example.com/x").unwrap())
            .is_err());
    }
}
