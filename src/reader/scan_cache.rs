//! What a scan's partitions share: the arrays they open, and the coordinates
//! they read whole, which are the same for every partition.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use datafusion::error::{DataFusionError, Result};
use tokio::sync::OnceCell;
use zarrs::array::Array;
use zarrs::storage::{AsyncReadableListableStorage, AsyncReadableListableStorageTraits};

use super::coord::CoordValues;

pub type AsyncArray = Array<dyn AsyncReadableListableStorageTraits>;

type Cells<T> = Mutex<HashMap<String, Arc<OnceCell<T>>>>;

/// One scan's opened arrays and whole coordinates, keyed by array path. Each
/// is read once, by whichever partition asks first, while the others wait.
#[derive(Default)]
pub struct ScanCache {
    arrays: Cells<Arc<AsyncArray>>,
    coords: Cells<CoordValues>,
}

impl std::fmt::Debug for ScanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanCache")
            .field("arrays", &self.arrays.lock().expect("unpoisoned").len())
            .field("coords", &self.coords.lock().expect("unpoisoned").len())
            .finish()
    }
}

fn cell<T>(cells: &Cells<T>, path: &str) -> Arc<OnceCell<T>> {
    cells
        .lock()
        .expect("unpoisoned")
        .entry(path.to_string())
        .or_default()
        .clone()
}

impl ScanCache {
    pub(crate) async fn array(
        &self,
        store: &AsyncReadableListableStorage,
        path: &str,
    ) -> Result<Arc<AsyncArray>> {
        cell(&self.arrays, path)
            .get_or_try_init(|| async {
                Array::async_open(store.clone(), path)
                    .await
                    .map(Arc::new)
                    .map_err(|e| DataFusionError::External(Box::new(e)))
            })
            .await
            .cloned()
    }

    /// The whole coordinate at `path`, from `read` the first time.
    pub(crate) async fn coord<F, Fut>(&self, path: &str, read: F) -> Result<CoordValues>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CoordValues>>,
    {
        cell(&self.coords, path)
            .get_or_try_init(read)
            .await
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn a_coordinate_is_read_once_however_many_ask() {
        let cache = ScanCache::default();
        let reads = AtomicUsize::new(0);
        let read = || async {
            reads.fetch_add(1, Ordering::Relaxed);
            Ok(CoordValues::Int64(vec![1, 2, 3]))
        };
        let values = futures::future::try_join_all((0..4).map(|_| cache.coord("/lat", read)))
            .await
            .unwrap();

        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert!(values.iter().all(|v| v.len() == 3));
        cache.coord("/lon", read).await.unwrap();
        assert_eq!(reads.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn a_failed_read_is_tried_again() {
        let cache = ScanCache::default();
        let failed = cache
            .coord("/lat", || async {
                Err(DataFusionError::Execution("unreachable".to_string()))
            })
            .await;
        assert!(failed.is_err());
        let values = cache
            .coord("/lat", || async { Ok(CoordValues::Int64(vec![1])) })
            .await
            .unwrap();
        assert_eq!(values.len(), 1);
    }
}
