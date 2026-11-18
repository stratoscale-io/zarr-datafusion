//! Skeleton creation: array metadata + coordinate arrays, no data chunks.
//!
//! The skeleton is the target store before any data variable is materialised.
//! Coordinate arrays are written in full (they *are* the grid); data variables
//! get metadata only. zarrs never writes a chunk that is entirely fill_value, so
//! an unwritten data variable reads back as fill_value everywhere for free —
//! which is exactly the "allocated but empty" semantics we want, and what the
//! Phase 1 exit criterion checks.
//!
//! Dimension order is the order of `SkeletonSpec::coords`, and each data
//! variable is the cartesian product of every coordinate in that order. That
//! matches the read side's assumption (see the Assumptions section of
//! CLAUDE.md), so a skeleton written here is readable by our own scan.
//!
//! `dimension_names` is always written. The reader can infer axes from shape
//! alone on v3, but only when the axis lengths are distinct; writing the names
//! makes the mapping explicit rather than inferred, and is what lets xarray open
//! the result. This is the writer's half of the axis-swap hazard that the
//! round-trip fixture exists to detect (docs/zarr-write-roundtrip-plan.md §2).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zarrs::array::codec::bytes_to_bytes::blosc::{
    BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
};
use zarrs::array::codec::BloscCodec;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::group::GroupBuilder;
use tracing::{debug, info};

use super::store::open_target;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Element type of a written array.
///
/// Deliberately narrow: these are the types the round-trip fixture and the
/// first real target (NDVI) need. Widen when a consumer needs more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteDataType {
    Int64,
    Float32,
    Float64,
}

impl WriteDataType {
    fn to_zarrs(self) -> DataType {
        match self {
            Self::Int64 => zarrs::array::data_type::int64(),
            Self::Float32 => zarrs::array::data_type::float32(),
            Self::Float64 => zarrs::array::data_type::float64(),
        }
    }

    /// Bytes per element — Blosc needs this as its shuffle `typesize`.
    fn size(self) -> usize {
        match self {
            Self::Int64 | Self::Float64 => 8,
            Self::Float32 => 4,
        }
    }

    /// Default hole marker: NaN for floats, 0 for ints.
    ///
    /// Integers have no NaN, so an unwritten int cell is indistinguishable from
    /// a written 0. Callers who care must pass an explicit sentinel.
    fn default_fill_value(self) -> FillValue {
        match self {
            Self::Int64 => FillValue::from(0i64),
            Self::Float32 => FillValue::from(f32::NAN),
            Self::Float64 => FillValue::from(f64::NAN),
        }
    }

    /// Map a query's output Arrow type to a writable Zarr element type.
    ///
    /// Rule: **widen where necessary, never narrow.** `WriteDataType` cannot
    /// represent every Arrow width, so anything narrower is promoted to the
    /// nearest one that holds it losslessly — signed and unsigned integers up to
    /// 32 bits into `Int64`, `Float16`/`Float32` into `Float32`. Narrowing is
    /// refused rather than performed silently, because a truncated write is the
    /// exact class of plausible-looking corruption this whole plan guards against.
    ///
    /// Two types are refused despite being numeric: `UInt64` has no lossless
    /// signed-integer home (its top bit overflows `i64`), and `Float64` stays
    /// `Float64` — it is *not* narrowed to `Float32`. Non-numeric types
    /// (strings, booleans, temporal, dictionaries) have no Zarr element mapping
    /// here yet.
    ///
    /// This function derives a target type *from a query output*, where narrowing
    /// cannot arise. A separately *declared* target dtype that disagrees with the
    /// query output (the future `COPY TO ... OPTIONS` surface) is a different
    /// decision and is not made here.
    pub fn from_arrow(dt: &arrow::datatypes::DataType) -> Result<Self, BoxError> {
        use arrow::datatypes::DataType as A;
        Ok(match dt {
            A::Int8 | A::Int16 | A::Int32 | A::Int64 | A::UInt8 | A::UInt16 | A::UInt32 => {
                Self::Int64
            }
            A::Float16 | A::Float32 => Self::Float32,
            A::Float64 => Self::Float64,
            A::UInt64 => {
                return Err("cannot write UInt64: it has no lossless signed-integer \
                            target; cast the column to Int64 or Float64 first"
                    .into())
            }
            other => {
                return Err(format!(
                    "cannot write Arrow type {other:?} to Zarr yet (supported: \
                     signed/unsigned ints -> Int64, Float16/Float32 -> Float32, \
                     Float64 -> Float64)"
                )
                .into())
            }
        })
    }
}

/// Values of one coordinate axis, dense and in index order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordValues {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

impl CoordValues {
    /// Number of points on this axis.
    pub fn len(&self) -> usize {
        match self {
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
        }
    }

    /// Whether this axis has no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn data_type(&self) -> WriteDataType {
        match self {
            Self::Int64(_) => WriteDataType::Int64,
            Self::Float64(_) => WriteDataType::Float64,
        }
    }
}

/// A coordinate array: a 1-D axis, stored as a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordSpec {
    pub name: String,
    pub values: CoordValues,
    pub attributes: Map<String, Value>,
}

impl CoordSpec {
    pub fn new(name: impl Into<String>, values: CoordValues) -> Self {
        Self {
            name: name.into(),
            values,
            attributes: Map::new(),
        }
    }
}

/// An n-D data variable: metadata now, chunks later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataVarSpec {
    pub name: String,
    pub data_type: WriteDataType,
    /// Hole marker. `None` takes [`WriteDataType::default_fill_value`].
    #[serde(with = "fill_value_bytes")]
    pub fill_value: Option<FillValue>,
    pub attributes: Map<String, Value>,
}

impl DataVarSpec {
    pub fn new(name: impl Into<String>, data_type: WriteDataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            fill_value: None,
            attributes: Map::new(),
        }
    }

    /// Derive a data-variable spec from a query's output field: its name and its
    /// widened element type ([`WriteDataType::from_arrow`]), with the default
    /// fill for that type. Override the fill afterward via [`Self::fill_value`] —
    /// which matters for integers, whose default fill `0` cannot be told apart
    /// from a written `0` in a filtered write.
    pub fn from_arrow_field(field: &arrow::datatypes::Field) -> Result<Self, BoxError> {
        Ok(Self::new(
            field.name(),
            WriteDataType::from_arrow(field.data_type())?,
        ))
    }
}

/// A fill value as its native-endian bytes.
mod fill_value_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use zarrs::array::FillValue;

    pub fn serialize<S: Serializer>(fill: &Option<FillValue>, s: S) -> Result<S::Ok, S::Error> {
        fill.as_ref().map(|f| f.as_ne_bytes().to_vec()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<FillValue>, D::Error> {
        Ok(Option::<Vec<u8>>::deserialize(d)?.map(FillValue::new))
    }
}

/// The full description of a target store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonSpec {
    /// Dimension order. Every data variable spans these axes in this order.
    pub coords: Vec<CoordSpec>,
    pub data_vars: Vec<DataVarSpec>,
    /// Chunk shape for data variables, one entry per coordinate.
    pub chunks: Vec<u64>,
    pub attributes: Map<String, Value>,
}

impl SkeletonSpec {
    pub fn new(coords: Vec<CoordSpec>, data_vars: Vec<DataVarSpec>, chunks: Vec<u64>) -> Self {
        Self {
            coords,
            data_vars,
            chunks,
            attributes: Map::new(),
        }
    }

    /// Array shape: the length of each coordinate, in dimension order.
    fn shape(&self) -> Vec<u64> {
        self.coords.iter().map(|c| c.values.len() as u64).collect()
    }

    fn dimension_names(&self) -> Vec<String> {
        self.coords.iter().map(|c| c.name.clone()).collect()
    }

    /// Reject specs that would produce a store we could not read back.
    fn validate(&self) -> Result<(), BoxError> {
        if self.coords.is_empty() {
            return Err("skeleton needs at least one coordinate".into());
        }
        if self.chunks.len() != self.coords.len() {
            return Err(format!(
                "chunks has {} entries but there are {} coordinates; \
                 a chunk shape is required per dimension",
                self.chunks.len(),
                self.coords.len()
            )
            .into());
        }
        if let Some(bad) = self.chunks.iter().position(|&c| c == 0) {
            return Err(format!("chunk size must be non-zero (dimension {bad})").into());
        }
        for coord in &self.coords {
            if coord.values.is_empty() {
                return Err(format!("coordinate '{}' is empty", coord.name).into());
            }
        }

        // The reader classifies arrays by name and assumes coordinate names are
        // unique; duplicates would silently collapse an axis.
        let mut names: Vec<&str> = self.coords.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if let Some(dup) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(format!("duplicate coordinate name '{}'", dup[0]).into());
        }

        // A data variable sharing a coordinate's name would be discovered as a
        // 1-D coordinate on read, not as a data variable.
        for var in &self.data_vars {
            if names.binary_search(&var.name.as_str()).is_ok() {
                return Err(format!(
                    "data variable '{}' collides with a coordinate name",
                    var.name
                )
                .into());
            }
        }

        Ok(())
    }
}

/// Blosc/LZ4, matching what `scripts/data_gen.py` writes.
fn blosc(type_size: usize) -> Result<BloscCodec, BoxError> {
    BloscCodec::new(
        BloscCompressor::LZ4,
        BloscCompressionLevel::try_from(5u8).map_err(|e| format!("{e:?}"))?,
        None,
        BloscShuffleMode::Shuffle,
        Some(type_size),
    )
    .map_err(|e| format!("blosc codec: {e}").into())
}

/// Create the target store: root group, coordinate arrays (with data), and data
/// variable metadata (without data).
///
/// Overwrites metadata at `store_path` if arrays of the same name already exist.
pub fn create_skeleton(store_path: &str, spec: &SkeletonSpec) -> Result<(), BoxError> {
    spec.validate()?;

    let store = open_target(store_path)?;
    let shape = spec.shape();
    let dimension_names = spec.dimension_names();

    GroupBuilder::new()
        .attributes(spec.attributes.clone())
        .build(store.clone(), "/")?
        .store_metadata()?;

    for coord in &spec.coords {
        let len = coord.values.len() as u64;
        let dtype = coord.values.data_type();
        // One chunk per coordinate: axes are small and always read whole.
        let array = ArrayBuilder::new(
            vec![len],
            vec![len],
            dtype.to_zarrs(),
            dtype.default_fill_value(),
        )
        .bytes_to_bytes_codecs(vec![Arc::new(blosc(dtype.size())?)])
        .dimension_names(Some([coord.name.clone()]))
        .attributes(coord.attributes.clone())
        .build(store.clone(), &format!("/{}", coord.name))?;
        array.store_metadata()?;

        match &coord.values {
            CoordValues::Int64(v) => array.store_chunk(&[0], v.as_slice())?,
            CoordValues::Float64(v) => array.store_chunk(&[0], v.as_slice())?,
        }
        debug!(coord = %coord.name, len, "Coordinate array written");
    }

    for var in &spec.data_vars {
        let fill_value = var
            .fill_value
            .clone()
            .unwrap_or_else(|| var.data_type.default_fill_value());
        let array = ArrayBuilder::new(
            shape.clone(),
            spec.chunks.clone(),
            var.data_type.to_zarrs(),
            fill_value,
        )
        .bytes_to_bytes_codecs(vec![Arc::new(blosc(var.data_type.size())?)])
        .dimension_names(Some(dimension_names.clone()))
        .attributes(var.attributes.clone())
        .build(store.clone(), &format!("/{}", var.name))?;
        // Metadata only — no chunks. Reads return fill_value.
        array.store_metadata()?;
        debug!(var = %var.name, "Data variable metadata written");
    }

    info!(
        path = store_path,
        coords = spec.coords.len(),
        data_vars = spec.data_vars.len(),
        shape = ?shape,
        chunks = ?spec.chunks,
        "Skeleton created"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType as A, Field};

    #[test]
    fn from_arrow_widens_integers_to_int64() {
        for dt in [A::Int8, A::Int16, A::Int32, A::Int64, A::UInt8, A::UInt16, A::UInt32] {
            assert_eq!(
                WriteDataType::from_arrow(&dt).unwrap(),
                WriteDataType::Int64,
                "{dt:?} should widen to Int64"
            );
        }
    }

    #[test]
    fn from_arrow_maps_floats_without_widening_f64() {
        assert_eq!(
            WriteDataType::from_arrow(&A::Float16).unwrap(),
            WriteDataType::Float32
        );
        assert_eq!(
            WriteDataType::from_arrow(&A::Float32).unwrap(),
            WriteDataType::Float32
        );
        // Float64 stays Float64 — it is never narrowed to Float32.
        assert_eq!(
            WriteDataType::from_arrow(&A::Float64).unwrap(),
            WriteDataType::Float64
        );
    }

    #[test]
    fn from_arrow_refuses_uint64_rather_than_truncating() {
        let err = WriteDataType::from_arrow(&A::UInt64).unwrap_err().to_string();
        assert!(err.contains("UInt64"), "{err}");
        assert!(err.contains("lossless"), "{err}");
    }

    #[test]
    fn from_arrow_refuses_unmapped_types() {
        for dt in [A::Utf8, A::Boolean, A::Date32] {
            assert!(
                WriteDataType::from_arrow(&dt).is_err(),
                "{dt:?} has no Zarr element mapping and must be refused"
            );
        }
    }

    #[test]
    fn ndvi_expression_type_resolves_to_float32() {
        // (b08 - b04) / (b08 + b04) over Float32 bands stays Float32 in DataFusion,
        // so the derived NDVI variable is written Float32 with a NaN fill — no
        // silent widening to Float64, no narrowing.
        let field = Field::new("ndvi", A::Float32, true);
        let spec = DataVarSpec::from_arrow_field(&field).unwrap();
        assert_eq!(spec.name, "ndvi");
        assert_eq!(spec.data_type, WriteDataType::Float32);
        assert!(spec.fill_value.is_none(), "default NaN fill for a float var");
    }
}
