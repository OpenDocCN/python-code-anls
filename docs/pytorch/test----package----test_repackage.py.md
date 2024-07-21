# `.\pytorch\test\package\test_repackage.py`

```
# Owner(s): ["oncall: package/deploy"]

# 导入所需模块
from io import BytesIO
from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前目录导入 PackageTestCase
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则从 common 模块导入 PackageTestCase
    # 支持直接运行此文件的情况
    from common import PackageTestCase

class TestRepackage(PackageTestCase):
    """Tests for repackaging."""

    def test_repackage_import_indirectly_via_parent_module(self):
        # 间接通过父模块导入所需的类
        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage
        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage

        # 创建 ImportsDirectlyFromSubSubPackage 的实例
        model_a = ImportsDirectlyFromSubSubPackage()
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将 model_a 保存为 pickle 格式到 buffer 中
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model.py", model_a)

        # 将 buffer 指针置于起始位置
        buffer.seek(0)
        # 使用 PackageImporter 从 buffer 中加载 pickle 格式的数据
        pi = PackageImporter(buffer)
        loaded_model = pi.load_pickle("default", "model.py")

        # 创建 ImportsIndirectlyFromSubPackage 的实例
        model_b = ImportsIndirectlyFromSubPackage()
        # 创建一个新的字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将 model_b 保存为 pickle 格式到 buffer 中
        with PackageExporter(
            buffer,
            importer=(
                pi,  # 使用先前创建的 PackageImporter 对象作为导入器之一
                sys_importer,  # 系统默认的导入器
            ),
        ) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model_b.py", model_b)

# 如果此文件作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```