# `.\pytorch\test\test_package.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 从指定的模块中导入测试类，忽略 F401 错误（未使用的导入）
from package.package_a.test_all_leaf_modules_tracer import (
    TestAllLeafModulesTracer,
)
from package.package_a.test_nn_module import TestNnModule  # noqa: F401
from package.test_analyze import TestAnalyze  # noqa: F401
from package.test_dependency_api import TestDependencyAPI  # noqa: F401
from package.test_dependency_hooks import TestDependencyHooks  # noqa: F401
from package.test_digraph import TestDiGraph  # noqa: F401
from package.test_directory_reader import DirectoryReaderTest  # noqa: F401
from package.test_glob_group import TestGlobGroup  # noqa: F401
from package.test_importer import TestImporter  # noqa: F401
from package.test_load_bc_packages import TestLoadBCPackages  # noqa: F401
from package.test_mangling import TestMangling  # noqa: F401
from package.test_misc import TestMisc  # noqa: F401
from package.test_model import ModelTest  # noqa: F401
from package.test_package_fx import TestPackageFX  # noqa: F401
from package.test_package_script import TestPackageScript  # noqa: F401
from package.test_repackage import TestRepackage  # noqa: F401
from package.test_resources import TestResources  # noqa: F401
from package.test_save_load import TestSaveLoad  # noqa: F401

# 如果脚本被直接执行
if __name__ == "__main__":
    # 从 torch.testing._internal.common_utils 中导入 run_tests 函数
    from torch.testing._internal.common_utils import run_tests

    # 运行所有测试
    run_tests()
```