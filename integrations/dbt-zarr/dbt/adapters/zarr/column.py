"""Columns typed with DataFusion's (Arrow) type names."""

from dataclasses import dataclass

from dbt.adapters.base.column import Column

_INTEGER_TYPES = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
_FLOAT_TYPES = {"Float16", "Float32", "Float64"}
_STRING_TYPES = {"Utf8", "LargeUtf8", "Utf8View"}


@dataclass
class ZarrColumn(Column):
    """A column as reported by `information_schema.columns`, whose types are
    Arrow type names (`Int64`, `Utf8`, `Timestamp(µs, "UTC")`, ...).

    Zarr coordinates are dictionary-encoded (`Dictionary(Int16, Float32)`);
    dbt only cares about the value type, so that is what `dtype` holds.
    """

    def __post_init__(self):
        self.dtype = self.value_type(self.dtype)

    @staticmethod
    def value_type(dtype: str) -> str:
        if dtype.startswith("Dictionary(") and dtype.endswith(")"):
            _, _, value = dtype[len("Dictionary(") : -1].partition(",")
            return value.strip()
        return dtype

    def is_string(self) -> bool:
        return self.dtype in _STRING_TYPES

    def is_integer(self) -> bool:
        return self.dtype in _INTEGER_TYPES

    def is_float(self) -> bool:
        return self.dtype in _FLOAT_TYPES

    def is_numeric(self) -> bool:
        return self.dtype.startswith("Decimal")

    def is_number(self) -> bool:
        return self.is_integer() or self.is_float() or self.is_numeric()

    def string_size(self) -> int:
        # Arrow strings are unbounded.
        return 2**31 - 1

    @classmethod
    def string_type(cls, size: int) -> str:
        return "VARCHAR"
