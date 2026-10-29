"""dbt's standard adapter tests (dbt-tests-adapter), for what the adapter
supports: view/table/ephemeral models, seeds, tests and docs. Incremental
models and snapshots are not supported."""

from dbt.tests.adapter.basic.test_adapter_methods import BaseAdapterMethod
from dbt.tests.adapter.basic.test_base import BaseSimpleMaterializations
from dbt.tests.adapter.basic.test_empty import BaseEmpty
from dbt.tests.adapter.basic.test_ephemeral import BaseEphemeral
from dbt.tests.adapter.basic.test_generic_tests import BaseGenericTests
from dbt.tests.adapter.basic.test_singular_tests import BaseSingularTests
from dbt.tests.adapter.basic.test_singular_tests_ephemeral import BaseSingularTestsEphemeral
from dbt.tests.adapter.basic.test_table_materialization import BaseTableMaterialization
from dbt.tests.adapter.basic.test_validate_connection import BaseValidateConnection


class TestSimpleMaterializationsZarr(BaseSimpleMaterializations):
    pass


class TestEmptyZarr(BaseEmpty):
    pass


class TestEphemeralZarr(BaseEphemeral):
    pass


class TestGenericTestsZarr(BaseGenericTests):
    pass


class TestSingularTestsZarr(BaseSingularTests):
    pass


class TestSingularTestsEphemeralZarr(BaseSingularTestsEphemeral):
    pass


class TestTableMaterializationZarr(BaseTableMaterialization):
    pass


class TestAdapterMethodsZarr(BaseAdapterMethod):
    pass


class TestValidateConnectionZarr(BaseValidateConnection):
    pass
