//! The store a write goes to: a local directory, or a prefix of an object store.

use std::sync::Arc;

use tokio::runtime::Handle;
use url::Url;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};
use zarrs::storage::ReadableWritableListableStorage;
use zarrs_object_store::object_store::gcp::GoogleCloudStorageBuilder;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::prefix::PrefixStore;
use zarrs_object_store::object_store::ObjectStore;
use zarrs_object_store::AsyncObjectStore;

use crate::reader::storage::s3_object_store;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct TokioBlockOn(Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

/// Open `store_path` for writing: a directory, or an `s3://`, `gs://` or
/// `file://` URL, with credentials from the environment as for reading.
///
/// A URL needs a tokio runtime, and must be opened, and written, outside its
/// async context: from `spawn_blocking`, say.
pub(crate) fn open_target(store_path: &str) -> Result<ReadableWritableListableStorage, BoxError> {
    if !store_path.contains("://") {
        return Ok(Arc::new(FilesystemStore::new(store_path)?));
    }
    let handle = Handle::try_current()
        .map_err(|_| format!("writing to {store_path} needs a tokio runtime"))?;
    let url = Url::parse(store_path)?;
    let bucket = url.host_str().unwrap_or_default();
    let prefix = ObjectPath::from_url_path(url.path())?;
    match url.scheme() {
        "s3" => {
            let store = handle.block_on(s3_object_store(bucket))?;
            Ok(adapt(store, prefix, handle))
        }
        "gs" => {
            let store = GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()?;
            Ok(adapt(store, prefix, handle))
        }
        "file" => Ok(adapt(LocalFileSystem::new(), prefix, handle)),
        scheme => Err(format!("can't write to a {scheme}:// store").into()),
    }
}

fn adapt<T: ObjectStore>(
    store: T,
    prefix: ObjectPath,
    handle: Handle,
) -> ReadableWritableListableStorage {
    let store = Arc::new(AsyncObjectStore::new(PrefixStore::new(store, prefix)));
    Arc::new(AsyncToSyncStorageAdapter::new(store, TokioBlockOn(handle)))
}
