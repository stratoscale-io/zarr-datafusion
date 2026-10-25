//! Icechunk store support (feature `icechunk`).
//!
//! Icechunk repos are versioned Zarr stores. We open one read-only at its `main`
//! branch tip and expose it as a [`zarrs`] async storage backend
//! ([`AsyncIcechunkStore`], which implements the same
//! `AsyncReadable*`/`AsyncListable*` traits our object-store backends do). That
//! lets the existing async schema-inference and read paths treat an icechunk
//! store exactly like any other remote Zarr store — it drops straight into
//! `ZarrTable`'s `cached_remote` slot.
//!
//! This module is deliberately narrow: detection + open, for both local and
//! remote (`gs://`/`s3://`) repos. Group addressing is handled by the caller via
//! the schema-inference/read prefix (see the factory's `group` option), not here.
//! Virtual chunks (byte-range references into archival HDF5/NetCDF) are resolved
//! by icechunk internally; for remote repos we do rewrite persisted S3 virtual-
//! chunk containers to anonymous access (see `open_icechunk_remote`).
//!
//! Known trade-off: icechunk performs its own object I/O, so our
//! `AsyncTrackedStore` byte statistics do not cover reads that go through here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use icechunk::config::{Credentials, S3Options};
use icechunk::repository::VersionInfo;
use icechunk::storage::{
    new_gcs_storage, new_s3_object_store_storage, GcsCredentials, S3Credentials, Storage,
};
use icechunk::virtual_chunks::VirtualChunkContainer;
use icechunk::{new_local_filesystem_storage, ObjectStoreConfig, Repository, RepositoryConfig};
use tracing::{debug, info, warn};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_icechunk::AsyncIcechunkStore;

/// Error opening an icechunk store.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// For a repo whose bucket's region can't be resolved.
const DEFAULT_S3_REGION: &str = "us-east-1";

/// Detect a local icechunk repository by its on-disk layout.
///
/// A local-filesystem icechunk repo has a `snapshots/` directory and a top-level
/// `repo` marker file. (Unlike a remote/object-store icechunk repo it has **no**
/// `refs/` directory — those refs are inlined — so we key on `snapshots/`, which
/// is present in both layouts, plus the `repo` marker to avoid matching an
/// arbitrary directory that merely happens to contain a `snapshots` subdir.)
pub fn is_icechunk_store(path: &str) -> bool {
    let root = Path::new(path);
    let has_snapshots = root.join("snapshots").is_dir();
    let has_repo_marker = root.join("repo").is_file();
    let detected = has_snapshots && has_repo_marker;
    if detected {
        debug!(
            path,
            "Detected local icechunk store (snapshots/ + repo marker)"
        );
    }
    detected
}

/// Open a local icechunk repository read-only at its `main` branch tip and return
/// it as a [`zarrs`] async storage backend.
///
/// The returned store can be handed to `infer_schema_with_meta_async` and the
/// async read path unchanged.
pub async fn open_icechunk_async(path: &str) -> Result<AsyncReadableListableStorage, BoxError> {
    info!(path, "Opening local icechunk store");
    let storage = new_local_filesystem_storage(Path::new(path)).await?;
    // Default config: local native + inlined chunks need no virtual-chunk
    // containers or credentials. (Virtual/remote containers arrive in a later phase.)
    let repo = Repository::open(storage)
        .config(RepositoryConfig::default())
        .execute()
        .await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    // `Arc<AsyncIcechunkStore>` coerces to `AsyncReadableListableStorage` via the
    // blanket `AsyncReadableListableStorageTraits` impl.
    let store: AsyncReadableListableStorage = Arc::new(AsyncIcechunkStore::new(session));
    debug!(
        path,
        "Opened icechunk readonly session at branch tip 'main'"
    );
    Ok(store)
}

/// A parsed `gs://bucket/prefix` or `s3://bucket/prefix` icechunk repo location.
struct RemoteRepoUrl {
    scheme: String,
    bucket: String,
    /// Object prefix of the repo within the bucket (no leading/trailing '/'), or
    /// empty if the repo is at the bucket root.
    prefix: Option<String>,
}

/// Parse a `gs://`/`s3://` URL into (scheme, bucket, repo-prefix).
fn parse_remote_repo_url(location: &str) -> Result<RemoteRepoUrl, BoxError> {
    let url = url::Url::parse(location)?;
    let scheme = url.scheme().to_string();
    if scheme != "gs" && scheme != "s3" {
        return Err(
            format!("unsupported icechunk scheme '{scheme}://' (expected gs:// or s3://)").into(),
        );
    }
    let bucket = url
        .host_str()
        .ok_or_else(|| format!("missing bucket in icechunk URL '{location}'"))?
        .to_string();
    let prefix = url.path().trim_matches('/');
    Ok(RemoteRepoUrl {
        scheme,
        bucket,
        prefix: if prefix.is_empty() {
            None
        } else {
            Some(prefix.to_string())
        },
    })
}

/// Probe whether the object store rooted at `prefix` looks like an icechunk repo.
///
/// Icechunk lays down a `snapshots/` "directory" in every backend layout (local
/// and object-store alike), so we list one level under the repo prefix and look
/// for a `snapshots/` child. This runs on the plain object store (the same one
/// used for Zarr/VirtualiZarr detection), before we commit to opening the repo
/// through the icechunk API.
pub async fn is_icechunk_store_async(
    store: &AsyncReadableListableStorage,
    prefix: &zarrs_object_store::object_store::path::Path,
) -> bool {
    use zarrs::storage::{AsyncListableStorageTraits, StorePrefix};

    let prefix_str = if prefix.as_ref().is_empty() {
        String::new()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let Ok(store_prefix) = StorePrefix::new(&prefix_str) else {
        return false;
    };
    let Ok(entries) = store.list_dir(&store_prefix).await else {
        return false;
    };
    entries.prefixes().iter().any(|p| {
        let name = p
            .as_str()
            .trim_start_matches(&prefix_str)
            .trim_end_matches('/');
        name == "snapshots" || name == "refs"
    })
}

/// Open a REMOTE icechunk repository (`gs://` or `s3://`) read-only at `main`.
///
/// `anonymous` controls both the primary store credentials and how virtual-chunk
/// containers are authorized. When true (the default for public archives like
/// ExtremeWeatherBench's CIRA forecast) we open anonymously AND rewrite every
/// persisted S3 virtual-chunk container to `anonymous: true`, working around an
/// icechunk 2.1.0 bug where a container's persisted `anonymous: false` clobbers
/// the Anonymous credential's skip-signature back off (so the request gets signed
/// and the anonymous fetch fails). See `examples/icechunk_read_cira.rs`.
pub async fn open_icechunk_remote(
    location: &str,
    anonymous: bool,
) -> Result<AsyncReadableListableStorage, BoxError> {
    let RemoteRepoUrl {
        scheme,
        bucket,
        prefix,
    } = parse_remote_repo_url(location)?;
    info!(
        location,
        scheme,
        bucket,
        ?prefix,
        anonymous,
        "Opening remote icechunk store"
    );

    let storage: Arc<dyn Storage + Send + Sync> = match scheme.as_str() {
        "gs" => {
            let creds = if anonymous {
                Some(GcsCredentials::Anonymous)
            } else {
                None // FromEnv
            };
            new_gcs_storage(bucket.clone(), prefix.clone(), creds, None, vec![], vec![])?
        }
        "s3" => {
            let region = crate::reader::storage::s3_bucket_region(&bucket)
                .await
                .unwrap_or_else(|| DEFAULT_S3_REGION.to_string());
            let opts = S3Options::default()
                .with_region(&region)
                .with_anonymous(anonymous);
            let creds = if anonymous {
                Some(S3Credentials::Anonymous)
            } else {
                None
            };
            new_s3_object_store_storage(opts, bucket.clone(), prefix.clone(), creds, vec![], vec![])
                .await?
        }
        other => return Err(format!("unsupported icechunk scheme '{other}'").into()),
    };

    // Phase 1: open with the persisted config to discover virtual-chunk containers.
    let repo0 = Repository::open(storage.clone()).execute().await?;

    // Phase 2: build an override config + authorization map. Copy every persisted
    // container through; for S3 ones, flip `anonymous: true` (the workaround) and
    // authorize with anonymous S3 credentials.
    let mut config = RepositoryConfig::default();
    let mut authorize: HashMap<String, Option<Credentials>> = HashMap::new();
    if let Some(containers) = repo0.config().virtual_chunk_containers.as_ref() {
        for (url_prefix, container) in containers {
            match &container.store {
                ObjectStoreConfig::S3(opts) | ObjectStoreConfig::S3Compatible(opts) => {
                    let opts = if anonymous {
                        opts.clone().with_anonymous(true)
                    } else {
                        opts.clone()
                    };
                    config.set_virtual_chunk_container(VirtualChunkContainer::new(
                        url_prefix.clone(),
                        ObjectStoreConfig::S3(opts),
                    )?)?;
                    let creds = if anonymous {
                        Some(Credentials::S3(S3Credentials::Anonymous))
                    } else {
                        None
                    };
                    authorize.insert(url_prefix.clone(), creds);
                }
                other => {
                    // Preserve non-S3 containers unchanged; leave them
                    // unauthorized (they error clearly only if actually read).
                    warn!(
                        url_prefix,
                        ?other,
                        "icechunk virtual-chunk container is not S3; passing through unauthorized"
                    );
                    config.set_virtual_chunk_container(container.clone())?;
                }
            }
        }
    }

    let repo = Repository::open(storage)
        .config(config)
        .authorize_virtual_chunk_access(authorize)
        .execute()
        .await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    let store: AsyncReadableListableStorage = Arc::new(AsyncIcechunkStore::new(session));
    debug!(
        location,
        "Opened remote icechunk readonly session at branch tip 'main'"
    );
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs::storage::AsyncListableStorageTraits;

    const SYNTHETIC: &str = "data/synthetic.icechunk";

    #[test]
    fn detects_local_icechunk_layout() {
        assert!(
            is_icechunk_store(SYNTHETIC),
            "data/synthetic.icechunk should be detected as an icechunk store"
        );
        // A plain directory (the repo root) is not an icechunk store.
        assert!(!is_icechunk_store("."));
        assert!(!is_icechunk_store("data"));
    }

    #[tokio::test]
    async fn opens_and_lists_root_arrays() {
        let store = open_icechunk_async(SYNTHETIC)
            .await
            .expect("open synthetic.icechunk");
        // The store must list the four arrays the M1 generator wrote at the root.
        let keys = store.list().await.expect("list store keys");
        let joined = keys
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for name in ["time", "lat", "lon", "temperature"] {
            assert!(
                joined.contains(&format!("{name}/zarr.json")),
                "expected {name}/zarr.json in store listing, got:\n{joined}"
            );
        }
    }
}
