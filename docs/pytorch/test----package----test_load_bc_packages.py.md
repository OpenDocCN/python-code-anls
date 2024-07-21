# `.\pytorch\test\package\test_load_bc_packages.py`

```
# Owner(s): ["oncall: package/deploy"]

# 从 pathlib 库导入 Path 类，从 unittest 库导入 skipIf 函数
from pathlib import Path
from unittest import skipIf

# 从 torch.package 模块中导入 PackageImporter 类
from torch.package import PackageImporter
# 从 torch.testing._internal.common_utils 中导入 IS_FBCODE 和 IS_SANDCASTLE 变量，以及 run_tests 函数
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests

try:
    # 尝试从当前目录中导入 common 模块中的 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则支持直接运行该文件的情况
    # 从 common 模块中导入 PackageTestCase 类
    from common import PackageTestCase

# 设置打包目录为当前文件所在目录的子目录 package_bc
packaging_directory = f"{Path(__file__).parent}/package_bc"

# 定义测试类 TestLoadBCPackages，继承自 PackageTestCase 类
class TestLoadBCPackages(PackageTestCase):
    """Tests for checking loading has backwards compatibility"""

    # 装饰器，条件为 IS_FBCODE 或 IS_SANDCASTLE 为 True 时跳过测试
    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    # 测试方法，测试加载 backward compatible nn 模块
    def test_load_bc_packages_nn_module(self):
        """Tests for backwards compatible nn module"""
        # 创建 PackageImporter 对象，加载 nn 模块的 pickle 文件
        importer1 = PackageImporter(f"{packaging_directory}/test_nn_module.pt")
        loaded1 = importer1.load_pickle("nn_module", "nn_module.pkl")

    # 装饰器，条件为 IS_FBCODE 或 IS_SANDCASTLE 为 True 时跳过测试
    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    # 测试方法，测试加载 backward compatible torchscript 模块
    def test_load_bc_packages_torchscript_module(self):
        """Tests for backwards compatible torchscript module"""
        # 创建 PackageImporter 对象，加载 torchscript 模块的 pickle 文件
        importer2 = PackageImporter(f"{packaging_directory}/test_torchscript_module.pt")
        loaded2 = importer2.load_pickle("torchscript_module", "torchscript_module.pkl")

    # 装饰器，条件为 IS_FBCODE 或 IS_SANDCASTLE 为 True 时跳过测试
    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    # 测试方法，测试加载 backward compatible fx 模块
    def test_load_bc_packages_fx_module(self):
        """Tests for backwards compatible fx module"""
        # 创建 PackageImporter 对象，加载 fx 模块的 pickle 文件
        importer3 = PackageImporter(f"{packaging_directory}/test_fx_module.pt")
        loaded3 = importer3.load_pickle("fx_module", "fx_module.pkl")


# 如果当前脚本作为主程序运行，则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```