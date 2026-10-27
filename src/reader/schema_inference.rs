//! Schema inference for Zarr v2 and v3 stores
//!
//! # Assumptions
//!
//! This module assumes a specific Zarr store structure:
//!
//! 1. **Coordinates are 1D arrays**: Any array with `shape.len() == 1` is treated as a coordinate.
//!    Examples: `time(7)`, `lat(10)`, `lon(10)`
//!
//! 2. **Data variables are nD arrays**: Arrays with `shape.len() > 1` are treated as data variables.
//!    Their dimensionality must equal the number of coordinate arrays.
//!
//! 3. **Cartesian product structure**: Data variables are assumed to be the Cartesian product
//!    of all coordinates. For coordinates `[time(7), lat(10), lon(10)]`, data variables must
//!    have shape `[7, 10, 10]` (i.e., `time × lat × lon`).
//!
//! 4. **Dimension ordering**: Coordinates are inferred to match the Zarr arrays' native
//!    dimension ordering when possible (by matching data variable shapes to coordinate sizes).
//!    If the ordering cannot be inferred unambiguously, we fall back to alphabetical ordering.
//!
//! # Example
//!
//! ```text
//! weather.zarr/
//! ├── time/       shape: [7]           → coordinate
//! ├── lat/        shape: [10]          → coordinate
//! ├── lon/        shape: [10]          → coordinate
//! ├── temperature/ shape: [7, 10, 10]  → data variable (time × lat × lon)
//! └── humidity/    shape: [7, 10, 10]  → data variable (time × lat × lon)
//! ```

use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::fs;
use std::path::Path;
use tracing::{debug, info, instrument};

use super::cf_time::CFTimeAttrs;
use super::dtype::{
    dictionary_key_type_for_cardinality, parse_v2_dtype, zarr_dtype_to_arrow,
    zarr_dtype_to_arrow_dictionary,
};

/// Zarr format version
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZarrVersion {
    V2,
    V3,
}

// =============================================================================
// Schema building helper
// =============================================================================

/// Build an Arrow schema from Zarr store metadata.
///
/// This consolidates the identical schema building logic used in:
/// - `infer_schema_from_zmetadata_json`
/// - `infer_schema_with_meta`
/// - `infer_schema_with_meta_async`
///
/// Coordinates use Dictionary encoding for memory efficiency.
/// CF time coordinates use Dictionary with Timestamp(Microsecond, UTC) values.
pub fn build_schema_from_store_meta(meta: &ZarrStoreMeta) -> Schema {
    let mut fields: Vec<Field> = Vec::new();

    // Coordinates use Dictionary encoding for memory efficiency
    // CF time coordinates use Dictionary with Timestamp values
    for coord in &meta.coords {
        // Number of distinct coordinate values determines the dictionary key width
        // (Int16 -> Int32 -> Int64) so large axes don't overflow the keys.
        let cardinality = coord.shape.first().copied().unwrap_or(0) as usize;
        let data_type = if coord
            .cf_time_attrs
            .as_ref()
            .is_some_and(|a| a.is_time_coordinate())
        {
            // CF time coordinate: Dictionary with Timestamp(Microsecond, UTC) values
            DataType::Dictionary(
                Box::new(dictionary_key_type_for_cardinality(cardinality)),
                Box::new(DataType::Timestamp(
                    TimeUnit::Microsecond,
                    Some("UTC".into()),
                )),
            )
        } else {
            zarr_dtype_to_arrow_dictionary(&coord.data_type, cardinality)
        };
        fields.push(Field::new(&coord.name, data_type, false));
    }

    // Data variables use regular arrays
    for var in &meta.data_vars {
        fields.push(Field::new(
            &var.name,
            zarr_dtype_to_arrow(&var.data_type),
            true,
        ));
    }

    Schema::new(fields)
}

/// Detect Zarr version by checking metadata files
pub fn detect_zarr_version(
    store_path: &str,
) -> Result<ZarrVersion, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);

    // Check for zarr.json (V3)
    if root.join("zarr.json").exists() {
        return Ok(ZarrVersion::V3);
    }

    // Check for .zgroup or .zarray (V2)
    if root.join(".zgroup").exists() || root.join(".zarray").exists() {
        return Ok(ZarrVersion::V2);
    }

    // Try to detect by looking at subdirectories
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path.join("zarr.json").exists() {
                return Ok(ZarrVersion::V3);
            }
            if path.join(".zarray").exists() {
                return Ok(ZarrVersion::V2);
            }
        }
    }

    Err("Could not detect Zarr version: no metadata files found".into())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZarrArrayMeta {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<u64>,
    /// Chunk sizes for this array (e.g., [160, 145, 144])
    pub chunks: Option<Vec<u64>>,
    /// Min/max bounds for coordinate arrays (None for data variables)
    /// Stored as (min, max) in f64 for simplicity
    pub coord_min_max: Option<(f64, f64)>,
    /// CF (Climate and Forecast) time attributes for time coordinates
    /// Contains units like "hours since 1900-01-01" and optional calendar
    pub cf_time_attrs: Option<CFTimeAttrs>,
    /// Dimension names for this variable (e.g., ["time", "latitude", "longitude"])
    /// Parsed from `_ARRAY_DIMENSIONS` (xarray/CF convention) or inferred from shape.
    /// None means dimension names are unknown/not yet inferred.
    pub dimensions: Option<Vec<String>>,
}

impl ZarrArrayMeta {
    pub fn is_coordinate(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns true if this is a scalar array (shape=[])
    /// Scalars don't fit the Cartesian product model and should be filtered out
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}

/// Discovered Zarr store structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZarrStoreMeta {
    pub coords: Vec<ZarrArrayMeta>,    // 1D arrays (sorted by name)
    pub data_vars: Vec<ZarrArrayMeta>, // nD arrays
    pub total_rows: usize,             // Product of all coordinate sizes
}

/// Discover all arrays in a Zarr store (v2 or v3)
pub fn discover_arrays(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // First try consolidated metadata (.zmetadata) - works for VirtualiZarr stores
    if let Some(meta) = discover_arrays_from_zmetadata(store_path)? {
        info!(
            coords = meta.coords.len(),
            data_vars = meta.data_vars.len(),
            "Arrays discovered from consolidated metadata"
        );
        return Ok(meta);
    }

    // Fall back to directory scanning
    let version = detect_zarr_version(store_path)?;

    match version {
        ZarrVersion::V2 => discover_arrays_v2(store_path),
        ZarrVersion::V3 => discover_arrays_v3(store_path),
    }
}

/// Infer schema from pre-loaded .zmetadata JSON (for VirtualiZarr stores)
///
/// This is used when the metadata has already been loaded (e.g., by VirtualStoreAdapter)
/// to avoid re-reading from remote storage.
pub fn infer_schema_from_zmetadata_json(
    metadata: &serde_json::Value,
) -> Result<(Schema, ZarrStoreMeta), Box<dyn std::error::Error + Send + Sync>> {
    let meta = discover_arrays_from_json(metadata)?.ok_or("No arrays found in .zmetadata JSON")?;
    let schema = build_schema_from_store_meta(&meta);
    Ok((schema, meta))
}

/// Discover arrays from a pre-loaded .zmetadata JSON value
fn discover_arrays_from_json(
    meta: &serde_json::Value,
) -> Result<Option<ZarrStoreMeta>, Box<dyn std::error::Error + Send + Sync>> {
    let metadata = meta
        .get("metadata")
        .ok_or("Missing 'metadata' key in .zmetadata")?;

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // Parse each array from consolidated metadata
    // Keys are like "temperature_2m/.zarray" or "time/.zattrs"
    for (key, value) in metadata.as_object().ok_or("'metadata' is not an object")? {
        if key.ends_with("/.zarray") {
            let name = key.trim_end_matches("/.zarray").to_string();

            let shape: Vec<u64> = value
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();

            let chunks: Option<Vec<u64>> = value
                .get("chunks")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

            let dtype_raw = value.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");
            let data_type = parse_v2_dtype(dtype_raw);

            // Look for corresponding .zattrs in consolidated metadata
            let zattrs_key = format!("{}/.zattrs", name);
            let zattrs = metadata.get(&zattrs_key);
            // Try CF attributes first, fallback to heuristic for nanosecond epoch
            let cf_time_attrs = zattrs
                .and_then(parse_cf_time_from_attrs)
                .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
            let dimensions = zattrs.and_then(parse_array_dimensions);

            debug!(name = %name, shape = ?shape, chunks = ?chunks, dtype = %data_type, dims = ?dimensions, "Found array in .zmetadata JSON");

            arrays.push(ZarrArrayMeta {
                name,
                data_type,
                shape,
                chunks,
                coord_min_max: None, // Skip min/max for VirtualiZarr (would require S3 access)
                cf_time_attrs,
                dimensions,
            });
        }
    }

    if arrays.is_empty() {
        return Ok(None);
    }

    info!(
        count = arrays.len(),
        "Discovered arrays from .zmetadata JSON"
    );
    // VirtualiZarr: skip min/max computation (requires S3 access)
    Ok(Some(separate_and_sort_arrays_no_stats(arrays)?))
}

/// Try to discover arrays from consolidated .zmetadata file (Zarr v2)
///
/// This handles both regular Zarr stores with consolidated metadata
/// and VirtualiZarr Parquet reference stores.
fn discover_arrays_from_zmetadata(
    store_path: &str,
) -> Result<Option<ZarrStoreMeta>, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);
    let zmetadata_path = root.join(".zmetadata");

    if !zmetadata_path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&zmetadata_path)?;

    // Handle non-standard JSON values (NaN, Infinity) that VirtualiZarr/Zarr can emit
    let content = content
        .replace(":NaN", ":null")
        .replace(": NaN", ": null")
        .replace(":Infinity", ":null")
        .replace(": Infinity", ": null")
        .replace(":-Infinity", ":null")
        .replace(": -Infinity", ": null");

    let meta: serde_json::Value = serde_json::from_str(&content)?;

    let metadata = meta
        .get("metadata")
        .ok_or("Missing 'metadata' key in .zmetadata")?;

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // Parse each array from consolidated metadata
    // Keys are like "temperature_2m/.zarray" or "time/.zattrs"
    for (key, value) in metadata.as_object().ok_or("'metadata' is not an object")? {
        if key.ends_with("/.zarray") {
            let name = key.trim_end_matches("/.zarray").to_string();

            let shape: Vec<u64> = value
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();

            let chunks: Option<Vec<u64>> = value
                .get("chunks")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

            let dtype_raw = value.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");
            let data_type = parse_v2_dtype(dtype_raw);

            // Look for corresponding .zattrs in consolidated metadata
            let zattrs_key = format!("{}/.zattrs", name);
            let zattrs = metadata.get(&zattrs_key);
            // Try CF attributes first, fallback to heuristic for nanosecond epoch
            let cf_time_attrs = zattrs
                .and_then(parse_cf_time_from_attrs)
                .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
            let dimensions = zattrs.and_then(parse_array_dimensions);

            debug!(name = %name, shape = ?shape, chunks = ?chunks, dtype = %data_type, dims = ?dimensions, "Found array in .zmetadata");

            arrays.push(ZarrArrayMeta {
                name,
                data_type,
                shape,
                chunks,
                coord_min_max: None, // Skip min/max for VirtualiZarr (would require S3 access)
                cf_time_attrs,
                dimensions,
            });
        }
    }

    if arrays.is_empty() {
        return Ok(None);
    }

    info!(count = arrays.len(), "Discovered arrays from .zmetadata");

    // For VirtualiZarr stores, skip min/max computation since coordinates
    // are stored as references to remote files
    let is_virtualizarr = super::virtual_store::is_virtualizarr_store(store_path);

    if is_virtualizarr {
        // VirtualiZarr: skip min/max computation (requires S3 access)
        Ok(Some(separate_and_sort_arrays_no_stats(arrays)?))
    } else {
        // Regular consolidated metadata: compute min/max normally
        Ok(Some(separate_and_sort_arrays(arrays, store_path)?))
    }
}

/// Discover arrays in a Zarr v2 store
fn discover_arrays_v2(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);
    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let zarray = path.join(".zarray");
            if zarray.exists() {
                let content = fs::read_to_string(&zarray)?;
                let meta: serde_json::Value = serde_json::from_str(&content)?;

                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let shape: Vec<u64> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                let chunks: Option<Vec<u64>> = meta
                    .get("chunks")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

                // V2 uses numpy dtype format like "<i8", "<f4"
                let dtype_raw = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");

                let data_type = parse_v2_dtype(dtype_raw);

                // Read .zattrs for CF time attributes and _ARRAY_DIMENSIONS
                let (cf_time_attrs, dimensions) = {
                    let zattrs = path.join(".zattrs");
                    if zattrs.exists() {
                        if let Ok(content) = fs::read_to_string(&zattrs) {
                            if let Ok(attrs) = serde_json::from_str::<serde_json::Value>(&content) {
                                // Try CF attributes first, fallback to heuristic
                                let cf = parse_cf_time_from_attrs(&attrs)
                                    .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
                                let dims = parse_array_dimensions(&attrs);
                                (cf, dims)
                            } else {
                                (infer_nanosecond_epoch_from_raw_dtype(dtype_raw), None)
                            }
                        } else {
                            (infer_nanosecond_epoch_from_raw_dtype(dtype_raw), None)
                        }
                    } else {
                        (infer_nanosecond_epoch_from_raw_dtype(dtype_raw), None)
                    }
                };

                arrays.push(ZarrArrayMeta {
                    name,
                    data_type,
                    shape,
                    chunks,
                    coord_min_max: None, // Will be computed in separate_and_sort_arrays
                    cf_time_attrs,
                    dimensions,
                });
            }
        }
    }

    separate_and_sort_arrays(arrays, store_path)
}

/// Discover arrays in a Zarr v3 store
fn discover_arrays_v3(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);
    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let zarr_json = path.join("zarr.json");
            if zarr_json.exists() {
                let content = fs::read_to_string(&zarr_json)?;
                let meta: serde_json::Value = serde_json::from_str(&content)?;

                if meta.get("node_type").and_then(|v| v.as_str()) == Some("array") {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let shape: Vec<u64> = meta
                        .get("shape")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                        .unwrap_or_default();

                    // V3 chunk_grid.configuration.chunk_shape
                    let chunks: Option<Vec<u64>> = meta
                        .get("chunk_grid")
                        .and_then(|v| v.get("configuration"))
                        .and_then(|v| v.get("chunk_shape"))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

                    // V3 data_type is already in a readable format (e.g., "float64", "int64")
                    let dtype_raw = meta
                        .get("data_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("float64");
                    let data_type = dtype_raw.to_string();

                    // V3 stores attributes in zarr.json under "attributes" key
                    // Try CF attributes first, fallback to heuristic for datetime64
                    let cf_time_attrs = meta
                        .get("attributes")
                        .and_then(parse_cf_time_from_attrs)
                        .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
                    // Prefer the xarray V2 attribute; fall back to the native V3
                    // `dimension_names` field (used by e.g. the CIRA forecast).
                    let dimensions = meta
                        .get("attributes")
                        .and_then(parse_array_dimensions)
                        .or_else(|| parse_v3_dimension_names(&meta));

                    arrays.push(ZarrArrayMeta {
                        name,
                        data_type,
                        shape,
                        chunks,
                        coord_min_max: None, // Will be computed in separate_and_sort_arrays
                        cf_time_attrs,
                        dimensions,
                    });
                }
            }
        }
    }

    separate_and_sort_arrays(arrays, store_path)
}

/// Compute min/max for a coordinate array by reading its data
/// Returns None if the computation fails (e.g., unsupported dtype, read error)
fn compute_coord_min_max(
    store_path: &str,
    coord_name: &str,
    data_type: &str,
) -> Option<(f64, f64)> {
    use zarrs::array::Array;
    use zarrs::array::ArraySubset;
    use zarrs::filesystem::FilesystemStore;

    // Open store and array
    let store = FilesystemStore::new(store_path).ok()?;
    let array_path = format!("/{}", coord_name);
    let array = Array::open(store.into(), &array_path).ok()?;

    // Get the full subset (entire 1D array)
    let shape = array.shape();
    let subset = ArraySubset::new_with_start_shape(vec![0], shape.to_vec()).ok()?;

    // Read based on data type and compute min/max
    match data_type {
        "float64" => {
            let data: Vec<f64> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((min, max))
        }
        "float32" => {
            let data: Vec<f32> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            Some((min, max))
        }
        "int64" => {
            let data: Vec<i64> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "int32" => {
            let data: Vec<i32> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "int16" => {
            let data: Vec<i16> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "uint64" => {
            let data: Vec<u64> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "uint32" => {
            let data: Vec<u32> = array.retrieve_array_subset(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        _ => {
            debug!(data_type = %data_type, "Unsupported data type for min/max computation");
            None
        }
    }
}

/// Attempt to infer coordinate ordering from data variable shapes.
///
/// We prefer to preserve the native dimension order of Zarr arrays by matching
/// each data variable's shape to the sizes of discovered coordinates. If a
/// matching data variable cannot be found or the mapping is ambiguous (e.g.,
/// multiple coordinates share the same size), we fall back to alphabetical
/// ordering for stability.
fn infer_coord_order_from_data_vars(
    mut coords: Vec<ZarrArrayMeta>,
    data_vars: &[ZarrArrayMeta],
) -> Vec<ZarrArrayMeta> {
    // If we have nothing to infer from, keep alphabetical ordering
    if coords.is_empty() || data_vars.is_empty() {
        coords.sort_by(|a, b| a.name.cmp(&b.name));
        return coords;
    }

    // Find the first data variable whose dimensionality equals the number
    // of coordinates and whose shape can be matched to coordinate sizes.
    for var in data_vars {
        if var.shape.len() != coords.len() {
            continue;
        }

        let mut ordered: Vec<ZarrArrayMeta> = Vec::with_capacity(coords.len());
        let mut used = vec![false; coords.len()];
        let mut success = true;

        for &dim_size in &var.shape {
            let mut found: Option<usize> = None;
            for (j, c) in coords.iter().enumerate() {
                if !used[j] && c.shape.first() == Some(&dim_size) {
                    found = Some(j);
                    break;
                }
            }

            if let Some(j) = found {
                ordered.push(coords[j].clone());
                used[j] = true;
            } else {
                success = false;
                break;
            }
        }

        if success && ordered.len() == coords.len() {
            return ordered;
        }
    }

    // Fallback to alphabetical ordering if we couldn't infer a mapping
    coords.sort_by(|a, b| a.name.cmp(&b.name));
    coords
}

/// Separate arrays into coordinates and data variables, then sort
/// Also computes min/max for coordinate arrays by reading their data
fn separate_and_sort_arrays(
    arrays: Vec<ZarrArrayMeta>,
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // Use into_iter + partition for single-pass, zero-clone separation
    let (mut coords, mut data_vars): (Vec<_>, Vec<_>) =
        arrays.into_iter().partition(|a| a.is_coordinate());

    // Keep data variables in stable alphabetical order for determinism
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    // Try to reorder coordinates to match Zarr arrays' native dimension order
    // by examining a representative data variable's shape. Fall back to
    // alphabetical ordering when the mapping is ambiguous.
    coords = infer_coord_order_from_data_vars(coords, &data_vars);

    // Compute min/max for each coordinate by reading the data
    for coord in &mut coords {
        if let Some(min_max) = compute_coord_min_max(store_path, &coord.name, &coord.data_type) {
            debug!(
                coord = %coord.name,
                min = min_max.0,
                max = min_max.1,
                "Computed coordinate min/max"
            );
            coord.coord_min_max = Some(min_max);
        }
    }

    // Compute total_rows = product of all coordinate sizes
    let total_rows: usize = coords.iter().map(|c| c.shape[0] as usize).product();

    Ok(ZarrStoreMeta {
        coords,
        data_vars,
        total_rows,
    })
}

/// Separate arrays into coordinates and data variables without computing statistics
/// Used for VirtualiZarr stores where coordinates are stored remotely
fn separate_and_sort_arrays_no_stats(
    arrays: Vec<ZarrArrayMeta>,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // Filter out scalar arrays (shape=[]) - they don't fit the Cartesian product model
    let arrays: Vec<_> = arrays.into_iter().filter(|a| !a.is_scalar()).collect();

    // Use into_iter + partition for single-pass, zero-clone separation
    let (mut coords, mut data_vars): (Vec<_>, Vec<_>) =
        arrays.into_iter().partition(|a| a.is_coordinate());

    // Keep data variables in stable alphabetical order for determinism
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    // Try to reorder coordinates to match Zarr arrays' native dimension order
    coords = infer_coord_order_from_data_vars(coords, &data_vars);

    // Skip min/max computation for VirtualiZarr stores (requires remote access)
    debug!("Skipping min/max computation for VirtualiZarr store");

    // Compute total_rows = product of all coordinate sizes
    let total_rows: usize = coords.iter().map(|c| c.shape[0] as usize).product();

    Ok(ZarrStoreMeta {
        coords,
        data_vars,
        total_rows,
    })
}

/// Infer Arrow schema from Zarr store metadata (v2 or v3)
/// Coordinates use DictionaryArray for memory efficiency (stores unique values once)
pub fn infer_schema(store_path: &str) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
    let (schema, _meta) = infer_schema_with_meta(store_path)?;
    Ok(schema)
}

/// Infer Arrow schema and return the store metadata for statistics
/// This allows caching the metadata for later use during query execution
pub fn infer_schema_with_meta(
    store_path: &str,
) -> Result<(Schema, ZarrStoreMeta), Box<dyn std::error::Error + Send + Sync>> {
    let meta = discover_arrays(store_path)?;
    let schema = build_schema_from_store_meta(&meta);

    // Note: Schema metadata causes issues with DataFusion's optimizer schema comparisons.
    // Instead of storing metadata in the schema, we return ZarrStoreMeta which contains
    // all dimension info. The CLI can access this via the ZarrTable struct.
    Ok((schema, meta))
}

/// Parse CF time attributes from a JSON attributes object
///
/// Looks for "units" attribute containing " since " pattern (e.g., "hours since 1900-01-01")
/// and optional "calendar" attribute.
fn parse_cf_time_from_attrs(attrs: &serde_json::Value) -> Option<CFTimeAttrs> {
    let units = attrs.get("units")?.as_str()?;
    // Only treat as CF time if it contains " since " pattern
    if !units.contains(" since ") {
        return None;
    }
    Some(CFTimeAttrs::new(
        units.to_string(),
        attrs
            .get("calendar")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    ))
}

/// Heuristically infer if a column contains nanosecond epoch timestamps
///
/// Returns synthetic CFTimeAttrs if the RAW dtype (before parsing) is datetime64 (M8):
/// - `<M8[ns]` = little-endian datetime64 with nanosecond resolution
/// - `>M8[ns]` = big-endian datetime64 with nanosecond resolution
///
/// This handles cases where CF metadata is missing but the dtype clearly
/// indicates a datetime (e.g., VirtualiZarr stores with incomplete metadata).
///
/// NOTE: Call this with the RAW dtype string (e.g., "<M8[ns]"), not the parsed type.
fn infer_nanosecond_epoch_from_raw_dtype(raw_dtype: &str) -> Option<CFTimeAttrs> {
    // Check if dtype is datetime64[ns] (numpy M8 type)
    // Zarr uses <M8[ns] for little-endian, >M8[ns] for big-endian
    // M8 = datetime64, m8 = timedelta64
    let is_datetime64_ns = raw_dtype.contains("M8[ns]") || raw_dtype.contains("datetime64[ns]");
    if !is_datetime64_ns {
        return None;
    }

    debug!(raw_dtype = %raw_dtype, "Inferred nanosecond epoch from datetime64[ns] dtype");
    Some(CFTimeAttrs::nanoseconds_since_epoch())
}

/// Parse `_ARRAY_DIMENSIONS` from attributes (xarray/CF convention)
///
/// This is the standard way xarray encodes dimension names in Zarr stores.
/// Example: `{"_ARRAY_DIMENSIONS": ["time", "latitude", "longitude"]}`
fn parse_array_dimensions(attrs: &serde_json::Value) -> Option<Vec<String>> {
    attrs
        .get("_ARRAY_DIMENSIONS")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
}

/// Parse the native Zarr **V3** `dimension_names` top-level field from a zarr.json
/// value (distinct from the V2 `_ARRAY_DIMENSIONS` xarray attribute).
///
/// Per the V3 spec each entry may be `null`; we only return the names when ALL
/// are present, since a partial mapping can't be trusted to order coordinates.
/// This is what lets us read stores like the CIRA forecast, whose arrays carry
/// native `dimension_names` (e.g. `["init_time","lead_time","latitude","longitude"]`)
/// rather than the xarray attribute.
fn parse_v3_dimension_names(zarr_json: &serde_json::Value) -> Option<Vec<String>> {
    let arr = zarr_json.get("dimension_names")?.as_array()?;
    let names: Option<Vec<String>> = arr.iter().map(|v| v.as_str().map(str::to_string)).collect();
    names.filter(|n| !n.is_empty())
}

// =============================================================================
// Async versions for remote object stores
// =============================================================================

use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Async version of discover_arrays for remote object stores
#[instrument(level = "debug", skip_all)]
pub async fn discover_arrays_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // First, try consolidated metadata (.zmetadata) - works for HTTP stores
    // that don't support directory listing
    if let Some(meta) = discover_arrays_from_zmetadata_async(store, prefix).await? {
        info!(
            coords = meta.coords.len(),
            data_vars = meta.data_vars.len(),
            "Arrays discovered from consolidated metadata"
        );
        for coord in &meta.coords {
            debug!(name = %coord.name, shape = ?coord.shape, dtype = %coord.data_type, "Coordinate");
        }
        for var in &meta.data_vars {
            debug!(name = %var.name, shape = ?var.shape, dtype = %var.data_type, "Data variable");
        }
        return Ok(meta);
    }

    // Fall back to directory listing for v2/v3 detection
    debug!("Detecting Zarr version via directory listing");
    let version = detect_zarr_version_async(store, prefix).await?;
    info!(?version, "Zarr version detected");

    let result = match version {
        ZarrVersion::V2 => discover_arrays_v2_async(store, prefix).await,
        ZarrVersion::V3 => discover_arrays_v3_async(store, prefix).await,
    };

    if let Ok(ref meta) = result {
        info!(
            coords = meta.coords.len(),
            data_vars = meta.data_vars.len(),
            "Arrays discovered"
        );
        for coord in &meta.coords {
            debug!(name = %coord.name, shape = ?coord.shape, dtype = %coord.data_type, "Coordinate");
        }
        for var in &meta.data_vars {
            debug!(name = %var.name, shape = ?var.shape, dtype = %var.data_type, "Data variable");
        }
    }

    result
}

/// Async version of detect_zarr_version for remote object stores
pub async fn detect_zarr_version_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrVersion, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::AsyncListableStorageTraits;
    use zarrs::storage::StorePrefix;

    // Check for root zarr.json (V3). Store keys must be relative, so at the root
    // (empty) prefix the key is `zarr.json`, not `/zarr.json`.
    let root = prefix.as_ref().trim_end_matches('/');
    let join = |name: &str| {
        if root.is_empty() {
            name.to_string()
        } else {
            format!("{root}/{name}")
        }
    };
    if store_key_exists(store, &join("zarr.json")).await {
        return Ok(ZarrVersion::V3);
    }

    // Check for root .zgroup (V2)
    if store_key_exists(store, &join(".zgroup")).await {
        return Ok(ZarrVersion::V2);
    }

    // List directories and check first one for version detection
    // StorePrefix requires a trailing slash, EXCEPT the root prefix which is the
    // empty string (zarrs rejects a leading '/'). The root case is hit by stores
    // whose arrays live at the top level, e.g. an icechunk repo.
    let prefix_str = if prefix.as_ref().is_empty() {
        String::new()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store
        .list_dir(&store_prefix)
        .await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    for subdir in entries.prefixes() {
        let subdir_str = subdir.as_str().trim_end_matches('/');
        // Check for zarr.json in subdirectory (V3)
        let v3_path = format!("{}/zarr.json", subdir_str);
        if store_key_exists(store, &v3_path).await {
            return Ok(ZarrVersion::V3);
        }

        // Check for .zarray in subdirectory (V2)
        let v2_path = format!("{}/.zarray", subdir_str);
        if store_key_exists(store, &v2_path).await {
            return Ok(ZarrVersion::V2);
        }
    }

    Err("Could not detect Zarr version: no metadata files found".into())
}

/// Check if a key exists in the async store (without downloading content)
async fn store_key_exists(store: &AsyncReadableListableStorage, key: &str) -> bool {
    use zarrs::storage::{AsyncReadableStorageTraits, StoreKey};

    let store_key = match StoreKey::new(key) {
        Ok(k) => k,
        Err(_) => return false,
    };

    // Use size_key() instead of get() to check existence without downloading content
    matches!(store.size_key(&store_key).await, Ok(Some(_)))
}

/// Read a key from the async store as string
async fn store_get_string(
    store: &AsyncReadableListableStorage,
    key: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::{AsyncReadableStorageTraits, StoreKey};

    let store_key = StoreKey::new(key).map_err(|e| format!("Invalid key '{}': {}", key, e))?;

    let bytes = store
        .get(&store_key)
        .await
        .map_err(|e| format!("Failed to read '{}': {}", key, e))?
        .ok_or_else(|| format!("Key not found: {}", key))?;

    String::from_utf8(bytes.to_vec())
        .map_err(|e| format!("Invalid UTF-8 in '{}': {}", key, e).into())
}

/// Try to discover arrays from consolidated .zmetadata file (Zarr v2)
///
/// This is the preferred method for HTTP stores since they don't support directory listing.
/// The .zmetadata file contains all array metadata in a single JSON file.
#[instrument(level = "debug", skip_all)]
async fn discover_arrays_from_zmetadata_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<Option<ZarrStoreMeta>, Box<dyn std::error::Error + Send + Sync>> {
    let zmetadata_path = if prefix.as_ref().is_empty() {
        ".zmetadata".to_string()
    } else {
        format!("{}/.zmetadata", prefix.as_ref().trim_end_matches('/'))
    };

    debug!(path = %zmetadata_path, "Checking for consolidated metadata");

    // Try to read .zmetadata - return None if not found
    let content = match store_get_string(store, &zmetadata_path).await {
        Ok(c) => c,
        Err(_) => {
            debug!("No .zmetadata found, will use directory listing");
            return Ok(None);
        }
    };

    info!("Found consolidated metadata in .zmetadata");

    let meta: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse .zmetadata: {}", e))?;

    let metadata = meta
        .get("metadata")
        .ok_or("Missing 'metadata' key in .zmetadata")?;

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // Parse each array from consolidated metadata
    // Keys are like "temperature_2m/.zarray" or "time/.zattrs"
    for (key, value) in metadata.as_object().ok_or("'metadata' is not an object")? {
        if key.ends_with("/.zarray") {
            let name = key.trim_end_matches("/.zarray").to_string();

            let shape: Vec<u64> = value
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();

            let chunks: Option<Vec<u64>> = value
                .get("chunks")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

            let dtype_raw = value.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");
            let data_type = parse_v2_dtype(dtype_raw);

            // Look for corresponding .zattrs in consolidated metadata
            let zattrs_key = format!("{}/.zattrs", name);
            let zattrs = metadata.get(&zattrs_key);
            // Try CF attributes first, fallback to heuristic for nanosecond epoch
            let cf_time_attrs = zattrs
                .and_then(parse_cf_time_from_attrs)
                .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
            let dimensions = zattrs.and_then(parse_array_dimensions);

            debug!(name = %name, shape = ?shape, chunks = ?chunks, dtype = %data_type, dims = ?dimensions, "Found array in .zmetadata");

            arrays.push(ZarrArrayMeta {
                name,
                data_type,
                shape,
                chunks,
                coord_min_max: None,
                cf_time_attrs,
                dimensions,
            });
        }
    }

    if arrays.is_empty() {
        debug!("No arrays found in .zmetadata");
        return Ok(None);
    }

    info!(count = arrays.len(), "Discovered arrays from .zmetadata");
    separate_and_sort_arrays_async(store, prefix, arrays)
        .await
        .map(Some)
}

/// Async version of discover_arrays_v2 for remote stores
async fn discover_arrays_v2_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::{AsyncListableStorageTraits, StorePrefix};

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // StorePrefix requires a trailing slash, EXCEPT the root prefix which is the
    // empty string (zarrs rejects a leading '/'). The root case is hit by stores
    // whose arrays live at the top level, e.g. an icechunk repo.
    let prefix_str = if prefix.as_ref().is_empty() {
        String::new()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store
        .list_dir(&store_prefix)
        .await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    for subdir in entries.prefixes() {
        let subdir_str = subdir.as_str().trim_end_matches('/');
        let zarray_path = format!("{}/.zarray", subdir_str);

        // Try to read .zarray metadata
        if let Ok(content) = store_get_string(store, &zarray_path).await {
            let meta: serde_json::Value = serde_json::from_str(&content)?;

            // Extract array name from path (last component)
            let name = subdir_str
                .trim_end_matches('/')
                .rsplit('/')
                .next()
                .unwrap_or("unknown")
                .to_string();

            let shape: Vec<u64> = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();

            let chunks: Option<Vec<u64>> = meta
                .get("chunks")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

            let dtype_raw = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");
            let data_type = parse_v2_dtype(dtype_raw);

            // Read .zattrs for CF time attributes and _ARRAY_DIMENSIONS
            let zattrs_path = format!("{}/.zattrs", subdir_str);
            let (cf_time_attrs, dimensions) =
                if let Ok(attrs_content) = store_get_string(store, &zattrs_path).await {
                    if let Ok(attrs) = serde_json::from_str::<serde_json::Value>(&attrs_content) {
                        // Try CF attributes first, fallback to heuristic
                        (
                            parse_cf_time_from_attrs(&attrs)
                                .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw)),
                            parse_array_dimensions(&attrs),
                        )
                    } else {
                        (infer_nanosecond_epoch_from_raw_dtype(dtype_raw), None)
                    }
                } else {
                    (infer_nanosecond_epoch_from_raw_dtype(dtype_raw), None)
                };

            arrays.push(ZarrArrayMeta {
                name,
                data_type,
                shape,
                chunks,
                coord_min_max: None, // Not computed for async/remote stores yet
                cf_time_attrs,
                dimensions,
            });
        }
    }

    separate_and_sort_arrays_async(store, prefix, arrays).await
}

/// Async version of discover_arrays_v3 for remote stores
async fn discover_arrays_v3_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    use futures::future::join_all;
    use zarrs::storage::{AsyncListableStorageTraits, StorePrefix};

    // StorePrefix requires a trailing slash, EXCEPT the root prefix which is the
    // empty string (zarrs rejects a leading '/'). The root case is hit by stores
    // whose arrays live at the top level, e.g. an icechunk repo.
    let prefix_str = if prefix.as_ref().is_empty() {
        String::new()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store
        .list_dir(&store_prefix)
        .await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    // Collect all subdirectory paths for parallel fetching
    let subdirs: Vec<String> = entries
        .prefixes()
        .iter()
        .map(|subdir| subdir.as_str().trim_end_matches('/').to_string())
        .collect();

    info!(
        num_subdirs = subdirs.len(),
        "Fetching zarr.json metadata in parallel"
    );

    // Fetch all zarr.json files in parallel
    let fetch_futures: Vec<_> = subdirs
        .iter()
        .map(|subdir_str| {
            let zarr_json_path = format!("{}/zarr.json", subdir_str);
            let subdir_owned = subdir_str.clone();
            async move {
                match store_get_string(store, &zarr_json_path).await {
                    Ok(content) => Some((subdir_owned, content)),
                    Err(_) => None,
                }
            }
        })
        .collect();

    let results = join_all(fetch_futures).await;

    // Process results into arrays
    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();
    for result in results.into_iter().flatten() {
        let (subdir_str, content) = result;
        if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
            // Only process arrays (not groups)
            if meta.get("node_type").and_then(|v| v.as_str()) == Some("array") {
                let name = subdir_str
                    .trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .to_string();

                let shape: Vec<u64> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                // V3 chunk_grid.configuration.chunk_shape
                let chunks: Option<Vec<u64>> = meta
                    .get("chunk_grid")
                    .and_then(|v| v.get("configuration"))
                    .and_then(|v| v.get("chunk_shape"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect());

                // V3 data_type is already in a readable format
                let dtype_raw = meta
                    .get("data_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("float64");
                let data_type = dtype_raw.to_string();

                // V3 stores attributes in zarr.json under "attributes" key
                // Try CF attributes first, fallback to heuristic for datetime64
                let cf_time_attrs = meta
                    .get("attributes")
                    .and_then(parse_cf_time_from_attrs)
                    .or_else(|| infer_nanosecond_epoch_from_raw_dtype(dtype_raw));
                // Prefer the xarray V2 attribute; fall back to the native V3
                // `dimension_names` field (used by e.g. the CIRA forecast).
                let dimensions = meta
                    .get("attributes")
                    .and_then(parse_array_dimensions)
                    .or_else(|| parse_v3_dimension_names(&meta));

                arrays.push(ZarrArrayMeta {
                    name,
                    data_type,
                    shape,
                    chunks,
                    coord_min_max: None, // Not computed for async/remote stores yet
                    cf_time_attrs,
                    dimensions,
                });
            }
        }
    }

    separate_and_sort_arrays_async(store, prefix, arrays).await
}

/// Separate arrays into coordinates and data variables (async version)
async fn separate_and_sort_arrays_async(
    _store: &AsyncReadableListableStorage,
    _prefix: &ObjectPath,
    arrays: Vec<ZarrArrayMeta>,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // Filter out scalar arrays (shape=[]) - they don't fit the Cartesian product model
    // Examples: spatial_ref (CRS metadata), other auxiliary scalars
    let arrays: Vec<_> = arrays.into_iter().filter(|a| !a.is_scalar()).collect();

    // Use into_iter + partition for single-pass, zero-clone separation
    let (mut coords, mut data_vars): (Vec<_>, Vec<_>) =
        arrays.into_iter().partition(|a| a.is_coordinate());

    // Keep data variables in stable alphabetical order for determinism
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    // Try to reorder coordinates to match Zarr arrays' native dimension order
    // by examining a representative data variable's shape. Fall back to
    // alphabetical ordering when the mapping is ambiguous.
    coords = infer_coord_order_from_data_vars(coords, &data_vars);

    // TODO: Coordinate min/max statistics for remote stores
    //
    // Currently we skip min/max computation for remote stores to avoid expensive
    // chunk fetches during table registration. This means MIN()/MAX() queries on
    // coordinates will scan data instead of using statistics.
    //
    // Future optimization options:
    // 1. Skip entirely (current) - fastest registration, queries scan data
    // 2. Lazy computation - compute min/max on first MIN/MAX query and cache
    // 3. First/last chunk only - assume sorted coordinates, read only 2 chunks
    //    instead of all chunks (e.g., time/0 and time/19 for 20-chunk array)
    //
    // For now, coord_min_max remains None for remote stores.
    debug!("Skipping min/max computation for remote store (optimization)");

    // Compute total_rows = product of all coordinate sizes
    let total_rows: usize = coords.iter().map(|c| c.shape[0] as usize).product();

    Ok(ZarrStoreMeta {
        coords,
        data_vars,
        total_rows,
    })
}

/// Async version of infer_schema for remote object stores
#[instrument(level = "debug", skip_all)]
pub async fn infer_schema_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
    let (schema, _meta) = infer_schema_with_meta_async(store, prefix).await?;
    Ok(schema)
}

/// Async version of infer_schema that also returns the store metadata
/// This allows caching the metadata for later use during query execution
#[instrument(level = "debug", skip_all)]
pub async fn infer_schema_with_meta_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<(Schema, ZarrStoreMeta), Box<dyn std::error::Error + Send + Sync>> {
    debug!("Starting async schema inference");
    let meta = discover_arrays_async(store, prefix).await?;
    let schema = build_schema_from_store_meta(&meta);

    // Note: Schema metadata causes issues with DataFusion's optimizer schema comparisons.
    // Instead of storing metadata in the schema, we return ZarrStoreMeta which contains
    // all dimension info. The CLI can access this via the ZarrTable struct.
    info!(num_fields = schema.fields().len(), "Schema inferred");
    Ok((schema, meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    // ==================== dimension-name parsing tests ====================

    #[test]
    fn parses_native_v3_dimension_names() {
        // Zarr V3 stores dimension names in a top-level `dimension_names` array
        // (this is how the CIRA forecast arrays are written), not under attributes.
        let z = serde_json::json!({
            "node_type": "array",
            "shape": [4083, 41, 721, 1440],
            "dimension_names": ["init_time", "lead_time", "latitude", "longitude"],
        });
        assert_eq!(
            parse_v3_dimension_names(&z),
            Some(vec![
                "init_time".to_string(),
                "lead_time".to_string(),
                "latitude".to_string(),
                "longitude".to_string(),
            ])
        );
    }

    #[test]
    fn native_v3_dimension_names_absent_or_partial_is_none() {
        // No field at all.
        assert_eq!(parse_v3_dimension_names(&serde_json::json!({})), None);
        // A null entry (allowed by the V3 spec) means the mapping is incomplete
        // and must not be trusted for coordinate ordering.
        let partial = serde_json::json!({ "dimension_names": ["time", null, "lon"] });
        assert_eq!(parse_v3_dimension_names(&partial), None);
    }

    // ==================== detect_zarr_version tests ====================

    #[test]
    fn test_detect_zarr_version_v2() {
        assert_eq!(
            detect_zarr_version("data/synthetic_v2.zarr").unwrap(),
            ZarrVersion::V2
        );
        assert_eq!(
            detect_zarr_version("data/synthetic_v2_blosc.zarr").unwrap(),
            ZarrVersion::V2
        );
    }

    #[test]
    fn test_detect_zarr_version_v3() {
        assert_eq!(
            detect_zarr_version("data/synthetic_v3.zarr").unwrap(),
            ZarrVersion::V3
        );
        assert_eq!(
            detect_zarr_version("data/synthetic_v3_blosc.zarr").unwrap(),
            ZarrVersion::V3
        );
    }

    #[test]
    fn test_detect_zarr_version_error() {
        assert!(detect_zarr_version("data/nonexistent.zarr").is_err());
    }

    // ==================== ZarrArrayMeta tests ====================

    #[test]
    fn test_array_meta_is_coordinate() {
        // 1D arrays are coordinates
        let coord = ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10],
            chunks: Some(vec![10]),
            coord_min_max: Some((0.0, 90.0)),
            cf_time_attrs: None,
            dimensions: None,
        };
        assert!(coord.is_coordinate());

        // 2D and 3D arrays are NOT coordinates
        let data_2d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10, 10],
            chunks: Some(vec![5, 5]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(vec!["lat".to_string(), "lon".to_string()]),
        };
        assert!(!data_2d.is_coordinate());

        let data_3d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![7, 10, 10],
            chunks: Some(vec![7, 5, 5]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(vec![
                "time".to_string(),
                "lat".to_string(),
                "lon".to_string(),
            ]),
        };
        assert!(!data_3d.is_coordinate());
    }

    // ==================== discover_arrays tests ====================

    #[test]
    fn test_discover_arrays_v2() {
        let meta = discover_arrays("data/synthetic_v2.zarr").unwrap();

        // 3 coordinates (native Zarr ordering): time, lat, lon.
        // lat/lon are both size 10, so their relative order is decided by the
        // object store's (now lexicographic) listing order.
        assert_eq!(meta.coords.len(), 3);
        let coord_names: Vec<_> = meta.coords.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(coord_names, vec!["time", "lat", "lon"]);

        // 2 data variables (sorted): humidity, temperature
        assert_eq!(meta.data_vars.len(), 2);
        let var_names: Vec<_> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(var_names, vec!["humidity", "temperature"]);

        // Shapes
        assert_eq!(meta.coords[0].shape, vec![7]); // time
        assert_eq!(meta.coords[1].shape, vec![10]); // lat
        assert_eq!(meta.coords[2].shape, vec![10]); // lon
        assert_eq!(meta.data_vars[0].shape, vec![7, 10, 10]); // humidity
        assert_eq!(meta.data_vars[1].shape, vec![7, 10, 10]); // temperature

        // All dtypes should be int64 (from <i8)
        for arr in meta.coords.iter().chain(meta.data_vars.iter()) {
            assert_eq!(arr.data_type, "int64");
        }
    }

    #[test]
    fn test_discover_arrays_v3() {
        let meta = discover_arrays("data/synthetic_v3.zarr").unwrap();

        // Same structure as v2 (native ordering)
        assert_eq!(meta.coords.len(), 3);
        assert_eq!(meta.data_vars.len(), 2);

        let coord_names: Vec<_> = meta.coords.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(coord_names, vec!["time", "lat", "lon"]);

        let var_names: Vec<_> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(var_names, vec!["humidity", "temperature"]);
    }

    // ==================== infer_schema tests ====================

    #[test]
    fn test_infer_schema_structure() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // 5 fields: 3 coords + 2 data vars
        assert_eq!(schema.fields().len(), 5);

        let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["time", "lat", "lon", "humidity", "temperature"]);
    }

    #[test]
    fn test_infer_schema_coord_types() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // First 3 fields (coordinates) should be Dictionary type, non-nullable
        for i in 0..3 {
            let field = schema.field(i);
            assert!(
                matches!(field.data_type(), DataType::Dictionary(_, _)),
                "Coordinate {} should be Dictionary type",
                field.name()
            );
            assert!(
                !field.is_nullable(),
                "Coordinate {} should not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn test_infer_schema_data_var_types() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // Last 2 fields (data vars) should be regular Int64, nullable
        for i in 3..5 {
            let field = schema.field(i);
            assert_eq!(
                field.data_type(),
                &DataType::Int64,
                "Data var {} should be Int64",
                field.name()
            );
            assert!(
                field.is_nullable(),
                "Data var {} should be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn test_infer_schema_v2_v3_parity() {
        let schema_v2 = infer_schema("data/synthetic_v2.zarr").unwrap();
        let schema_v3 = infer_schema("data/synthetic_v3.zarr").unwrap();

        // Both should produce identical schemas
        assert_eq!(schema_v2.fields().len(), schema_v3.fields().len());

        for (f2, f3) in schema_v2.fields().iter().zip(schema_v3.fields().iter()) {
            assert_eq!(f2.name(), f3.name(), "Field names should match");
            assert_eq!(
                f2.data_type(),
                f3.data_type(),
                "Data types should match for {}",
                f2.name()
            );
            assert_eq!(
                f2.is_nullable(),
                f3.is_nullable(),
                "Nullability should match for {}",
                f2.name()
            );
        }
    }
}
