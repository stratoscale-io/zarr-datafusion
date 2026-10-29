from dbt.adapters.base import AdapterPlugin

from dbt.adapters.zarr.column import ZarrColumn  # noqa: F401
from dbt.adapters.zarr.connections import ZarrConnectionManager, ZarrCredentials  # noqa: F401
from dbt.adapters.zarr.impl import ZarrAdapter
from dbt.include import zarr

Plugin = AdapterPlugin(
    adapter=ZarrAdapter,
    credentials=ZarrCredentials,
    include_path=zarr.PACKAGE_PATH,
)
