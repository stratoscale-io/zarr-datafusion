// Milestone 0 probe: confirms the icechunk + zarrs_icechunk API surface compiles
// against our existing zarrs 0.22 (zarrs_storage 0.4). Not meant to run yet.
#![allow(unused_imports, dead_code, unreachable_code)]
use icechunk::repository::VersionInfo;
use icechunk::storage::new_gcs_storage;
use icechunk::{new_local_filesystem_storage, Repository, RepositoryConfig};
use std::collections::HashMap;
use std::sync::Arc;
use zarrs_icechunk::AsyncIcechunkStore;

async fn _typecheck(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let storage = new_local_filesystem_storage(std::path::Path::new(path)).await?;
    let repo = Repository::open(storage)
        .config(RepositoryConfig::default())
        .execute()
        .await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    let _store = Arc::new(AsyncIcechunkStore::new(session));
    Ok(())
}
fn main() {}
