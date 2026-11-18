//! Storage backend factory for local and cloud object stores.
//!
//! Supports:
//! - Local filesystem: plain paths or `file:///path`
//! - AWS S3: `s3://bucket/path`
//! - Google Cloud Storage: `gs://bucket/path`
//! - HTTP/HTTPS: `http://host/path` or `https://host/path`

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};
use url::Url;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::aws::{resolve_bucket_region, AmazonS3, AmazonS3Builder};
use zarrs_object_store::object_store::gcp::GoogleCloudStorageBuilder;
use zarrs_object_store::object_store::http::HttpBuilder;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{ClientOptions, Error as ObjectStoreError};
use zarrs_object_store::AsyncObjectStore;

/// Error type for storage operations
#[derive(Debug)]
pub enum StorageError {
    InvalidUrl(String),
    ObjectStore(ObjectStoreError),
    Filesystem(std::io::Error),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::InvalidUrl(msg) => write!(f, "Invalid URL: {}", msg),
            StorageError::ObjectStore(e) => write!(f, "Object store error: {}", e),
            StorageError::Filesystem(e) => write!(f, "Filesystem error: {}", e),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<ObjectStoreError> for StorageError {
    fn from(e: ObjectStoreError) -> Self {
        StorageError::ObjectStore(e)
    }
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        StorageError::Filesystem(e)
    }
}

/// Storage location with backend type and path information
#[derive(Debug, Clone)]
pub struct StorageLocation {
    /// The original URL/path string
    pub url: String,
    /// Path within the store (for object stores, this is the prefix)
    pub path: String,
    /// Whether this is a remote (cloud) store
    pub is_remote: bool,
}

impl StorageLocation {
    /// Parse a URL or path into a StorageLocation
    pub fn parse(location: &str) -> Result<Self, StorageError> {
        if location.starts_with("s3://")
            || location.starts_with("gs://")
            || location.starts_with("http://")
            || location.starts_with("https://")
        {
            // Remote URL - extract path from URL
            let url = Url::parse(location).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
            let path = url.path().trim_start_matches('/').to_string();
            Ok(StorageLocation {
                url: location.to_string(),
                path,
                is_remote: true,
            })
        } else if location.starts_with("file://") {
            // Local file URL
            let path = location.trim_start_matches("file://").to_string();
            Ok(StorageLocation {
                url: location.to_string(),
                path,
                is_remote: false,
            })
        } else {
            // Plain local path
            Ok(StorageLocation {
                url: location.to_string(),
                path: location.to_string(),
                is_remote: false,
            })
        }
    }
}

/// Create an async storage backend from a URL or path.
///
/// # Supported URL schemes
///
/// - `s3://bucket/path` - AWS S3 (credentials from environment)
/// - `gs://bucket/path` - Google Cloud Storage (credentials from environment or anonymous)
/// - `http://host/path` or `https://host/path` - HTTP/HTTPS remote stores
/// - `file:///path` or plain path - Local filesystem (not supported for async)
///
/// # Environment Variables
///
/// ## S3
/// - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - Credentials
/// - `AWS_DEFAULT_REGION` - Region
/// - `AWS_ENDPOINT` - Custom endpoint (for S3-compatible services)
///
/// ## GCS
/// - `GOOGLE_SERVICE_ACCOUNT` - Path to service account JSON
/// - Or uses Application Default Credentials
/// - Falls back to anonymous access for public buckets
#[instrument(level = "debug")]
pub async fn create_async_store(
    location: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    debug!("Parsing storage location");
    let _loc = StorageLocation::parse(location)?;
    debug!(is_remote = _loc.is_remote, path = %_loc.path, "Location parsed");

    if location.starts_with("s3://") {
        info!("Creating S3 store");
        create_s3_store(location).await
    } else if location.starts_with("gs://") {
        info!("Creating GCS store");
        create_gcs_store(location).await
    } else if location.starts_with("http://") || location.starts_with("https://") {
        info!("Creating HTTP store");
        create_http_store(location).await
    } else {
        debug!("Local filesystem - not supported for async");
        Err(StorageError::InvalidUrl(
            "Local filesystem should use synchronous FilesystemStore".to_string(),
        ))
    }
}

/// Create an S3 storage backend
async fn create_s3_store(
    url: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    let parsed = Url::parse(url).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
    let bucket = parsed
        .host_str()
        .ok_or_else(|| StorageError::InvalidUrl("Missing bucket in S3 URL".to_string()))?;
    let path = parsed.path().trim_start_matches('/');
    let store = s3_object_store(bucket).await?;

    // Returned untracked: byte counting is attached per-execution in `execute_remote`,
    // because this store is cached across queries while stats are per-query.
    let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let object_path = ObjectPath::from(path);

    Ok((async_store, object_path))
}

/// `bucket`, with credentials from the environment, in the bucket's own region.
pub(crate) async fn s3_object_store(bucket: &str) -> Result<AmazonS3, StorageError> {
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
    if let Some(region) = s3_bucket_region(bucket).await {
        builder = builder.with_region(region);
    }
    Ok(builder.build()?)
}

/// The region S3 says `bucket` is in, which the environment's region may not
/// be. None with a custom endpoint, such as MinIO's, or if S3 can't be asked.
pub async fn s3_bucket_region(bucket: &str) -> Option<String> {
    const ENDPOINTS: [&str; 3] = ["AWS_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"];
    if ENDPOINTS.iter().any(|key| std::env::var_os(key).is_some()) {
        return None;
    }
    static REGIONS: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    let mut regions = REGIONS.get_or_init(Default::default).lock().await;
    if let Some(region) = regions.get(bucket) {
        return Some(region.clone());
    }
    match resolve_bucket_region(bucket, &ClientOptions::default()).await {
        Ok(region) => {
            regions.insert(bucket.to_string(), region.clone());
            Some(region)
        }
        Err(e) => {
            warn!(bucket, error = %e, "Couldn't resolve the bucket's region");
            None
        }
    }
}

/// Create a GCS storage backend
#[instrument(level = "debug")]
async fn create_gcs_store(
    url: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    let parsed = Url::parse(url).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
    let bucket = parsed
        .host_str()
        .ok_or_else(|| StorageError::InvalidUrl("Missing bucket in GCS URL".to_string()))?;
    let path = parsed.path().trim_start_matches('/');
    debug!(bucket = bucket, path = path, "Parsed GCS URL");

    // For public buckets, try anonymous access first (faster)
    // Then fall back to credentials from environment
    debug!("Trying anonymous access for public bucket");
    let store = GoogleCloudStorageBuilder::new()
        .with_bucket_name(bucket)
        .with_skip_signature(true)
        .build()
        .or_else(|_| {
            debug!("Anonymous access failed, trying credentials from environment");
            GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()
        })?;

    // Returned untracked: byte counting is attached per-execution in `execute_remote`,
    // because this store is cached across queries while stats are per-query.
    let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let object_path = ObjectPath::from(path);
    info!(
        bucket = bucket,
        path = path,
        "GCS store created successfully"
    );

    Ok((async_store, object_path))
}

/// Create an HTTP/HTTPS storage backend
#[instrument(level = "debug")]
async fn create_http_store(
    url: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    let parsed = Url::parse(url).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;

    // Extract base URL (scheme + host + port) for the HttpBuilder
    let host = parsed
        .host_str()
        .ok_or_else(|| StorageError::InvalidUrl("Missing host in HTTP URL".to_string()))?;
    let base_url = if let Some(port) = parsed.port() {
        format!("{}://{}:{}", parsed.scheme(), host, port)
    } else {
        format!("{}://{}", parsed.scheme(), host)
    };

    // Path within the store (everything after the host)
    let path = parsed.path().trim_start_matches('/');
    debug!(base_url = %base_url, path = path, "Parsed HTTP URL");

    let store = HttpBuilder::new()
        .with_url(&base_url)
        .build()
        .map_err(StorageError::ObjectStore)?;

    // Returned untracked: byte counting is attached per-execution in `execute_remote`,
    // because this store is cached across queries while stats are per-query.
    let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let object_path = ObjectPath::from(path);

    info!(
        base_url = %base_url,
        path = path,
        "HTTP store created successfully"
    );

    Ok((async_store, object_path))
}

/// Check if a location is a remote (cloud or HTTP) URL
pub fn is_remote_url(location: &str) -> bool {
    location.starts_with("s3://")
        || location.starts_with("gs://")
        || location.starts_with("http://")
        || location.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let loc = StorageLocation::parse("s3://my-bucket/path/to/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/to/data.zarr");
    }

    #[test]
    fn test_parse_gcs_url() {
        let loc = StorageLocation::parse("gs://my-bucket/path/to/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/to/data.zarr");
    }

    #[test]
    fn test_parse_local_path() {
        let loc = StorageLocation::parse("/data/synthetic.zarr").unwrap();
        assert!(!loc.is_remote);
        assert_eq!(loc.path, "/data/synthetic.zarr");
    }

    #[test]
    fn test_parse_file_url() {
        let loc = StorageLocation::parse("file:///data/synthetic.zarr").unwrap();
        assert!(!loc.is_remote);
        assert_eq!(loc.path, "/data/synthetic.zarr");
    }

    #[test]
    fn test_is_remote_url() {
        assert!(is_remote_url("s3://bucket/path"));
        assert!(is_remote_url("gs://bucket/path"));
        assert!(is_remote_url("http://example.com/path"));
        assert!(is_remote_url("https://example.com/path"));
        assert!(!is_remote_url("/local/path"));
        assert!(!is_remote_url("file:///local/path"));
    }

    #[test]
    fn test_parse_http_url() {
        let loc = StorageLocation::parse("http://example.com/path/to/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/to/data.zarr");
    }

    #[test]
    fn test_parse_https_url() {
        let loc =
            StorageLocation::parse("https://data.dynamical.org/noaa/gfs/latest.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "noaa/gfs/latest.zarr");
    }

    #[test]
    fn test_parse_https_url_with_port() {
        let loc = StorageLocation::parse("https://example.com:8080/path/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/data.zarr");
    }
}
