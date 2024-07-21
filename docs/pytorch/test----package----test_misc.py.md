# `.\pytorch\test\package\test_misc.py`

```
# Owner(s): ["oncall: package/deploy"]
# 导入必要的模块和函数
import inspect  # 提供了从源代码中检索类和函数声明的功能
import os  # 提供了访问操作系统功能的功能
import platform  # 提供了访问平台特定属性的功能
import sys  # 提供了访问与 Python 解释器交互的变量和函数
from io import BytesIO  # 提供了在内存中操作二进制数据的工具
from pathlib import Path  # 提供了处理文件路径的面向对象 API
from textwrap import dedent  # 提供了缩进文本的功能
from unittest import skipIf  # 提供了一个装饰器，用于条件性地跳过测试用例

# 导入 torch 包的相关模块和函数
from torch.package import is_from_package, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError  # 导入打包错误异常类
from torch.testing._internal.common_utils import (
    IS_FBCODE,  # 是否在 Facebook 的代码环境中
    IS_SANDCASTLE,  # 是否在 Sandcastle 环境中
    run_tests,  # 运行测试的函数
    skipIfTorchDynamo,  # 如果是 Torch Dynamo 环境则跳过的装饰器
)

try:
    # 尝试导入本地的 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果导入失败，支持直接在当前文件中运行的情况
    from common import PackageTestCase

class TestMisc(PackageTestCase):
    """Tests for one-off or random functionality. Try not to add to this!"""
    def test_loaders_that_remap_files_work_ok(self):
        # 导入必要的模块和函数
        from importlib.abc import MetaPathFinder
        from importlib.machinery import SourceFileLoader
        from importlib.util import spec_from_loader

        # 定义一个自定义的加载器，用于重定向模块 "module_a"
        class LoaderThatRemapsModuleA(SourceFileLoader):
            def get_filename(self, name):
                # 调用父类方法获取原始的文件名
                result = super().get_filename(name)
                # 如果模块名是 "module_a"，则返回重定向后的文件路径
                if name == "module_a":
                    return os.path.join(
                        os.path.dirname(result), "module_a_remapped_path.py"
                    )
                else:
                    return result

        # 定义一个自定义的查找器，用于查找模块 "module_a" 的规范
        class FinderThatRemapsModuleA(MetaPathFinder):
            def find_spec(self, fullname, path, target):
                """尝试使用所有剩余的 meta_path 查找器找到模块 "module_a" 的原始规范。"""
                if fullname != "module_a":
                    return None
                spec = None
                # 遍历所有的 meta_path 查找器
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    # 如果 finder 具有 find_spec 方法，则使用它来查找规范
                    if hasattr(finder, "find_spec"):
                        spec = finder.find_spec(fullname, path, target=target)
                    # 如果 finder 具有 load_module 方法，则创建规范
                    elif hasattr(finder, "load_module"):
                        spec = spec_from_loader(fullname, finder)
                    # 如果找到了规范，则退出循环
                    if spec is not None:
                        break
                # 确保找到的规范非空且加载器是 SourceFileLoader 类型
                assert spec is not None and isinstance(spec.loader, SourceFileLoader)
                # 使用自定义加载器替换规范中的加载器
                spec.loader = LoaderThatRemapsModuleA(
                    spec.loader.name, spec.loader.path
                )
                return spec

        # 将自定义查找器插入到 sys.meta_path 的开头
        sys.meta_path.insert(0, FinderThatRemapsModuleA())
        # 从 sys.modules 中移除 "module_a"，以便下次导入时使用自定义查找器
        sys.modules.pop("module_a", None)
        try:
            # 创建一个字节流缓冲区
            buffer = BytesIO()
            # 使用 PackageExporter 将 module_a 导出到缓冲区
            with PackageExporter(buffer) as he:
                import module_a

                # 将所有模块都导出到缓冲区
                he.intern("**")
                he.save_module(module_a.__name__)

            # 将缓冲区指针移到开头
            buffer.seek(0)
            # 使用 PackageImporter 从缓冲区导入包
            hi = PackageImporter(buffer)
            # 断言 remapped_path 是否在导入的 module_a 中
            self.assertTrue("remapped_path" in hi.get_source("module_a"))
        finally:
            # 再次从 sys.modules 中移除 "module_a"，确保不影响其他测试
            sys.modules.pop("module_a", None)
            # 从 sys.meta_path 中移除自定义查找器
            sys.meta_path.pop(0)

    def test_python_version(self):
        """
        Tests that the current python version is stored in the package and is available
        via PackageImporter's python_version() method.
        """
        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 使用 PackageExporter 将 SimpleTest 对象保存到缓冲区
        with PackageExporter(buffer) as he:
            from package_a.test_module import SimpleTest

            he.intern("**")
            obj = SimpleTest()
            he.save_pickle("obj", "obj.pkl", obj)

        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 使用 PackageImporter 从缓冲区导入包
        hi = PackageImporter(buffer)

        # 断言当前 Python 版本是否与保存的版本一致
        self.assertEqual(hi.python_version(), platform.python_version())
    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    # 跳过测试：如果在 FBCODE 或 SANDCASTLE 环境下，不允许使用临时文件进行测试
    def test_load_python_version_from_package(self):
        """Tests loading a package with a python version embdded"""
        # 创建 PackageImporter 对象，加载指定路径下的 Python 包文件
        importer1 = PackageImporter(
            f"{Path(__file__).parent}/package_e/test_nn_module.pt"
        )
        # 断言导入的包的 Python 版本为 "3.9.7"
        self.assertEqual(importer1.python_version(), "3.9.7")

    def test_file_structure_has_file(self):
        """
        Test Directory's has_file() method.
        """
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将 package_a.subpackage 的内容保存到缓冲区中
        with PackageExporter(buffer) as he:
            import package_a.subpackage

            # 内部化所有内容
            he.intern("**")
            # 创建 package_a.subpackage 的对象
            obj = package_a.subpackage.PackageASubpackageObject()
            # 将对象保存为 pickle 文件
            he.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)

        # 使用 PackageImporter 导入缓冲区中的内容
        importer = PackageImporter(buffer)
        # 获取导入的文件结构
        file_structure = importer.file_structure()
        # 断言 package_a/subpackage.py 文件存在
        self.assertTrue(file_structure.has_file("package_a/subpackage.py"))
        # 断言 package_a/subpackage 目录不存在
        self.assertFalse(file_structure.has_file("package_a/subpackage"))

    def test_exporter_content_lists(self):
        """
        Test content list API for PackageExporter's contained modules.
        """
        # 使用 PackageExporter 将 package_b 的指定子包导出到一个字节流缓冲区中
        with PackageExporter(BytesIO()) as he:
            import package_b

            # 外部化 package_b.subpackage_1
            he.extern("package_b.subpackage_1")
            # 模拟 package_b.subpackage_2
            he.mock("package_b.subpackage_2")
            # 内部化所有内容
            he.intern("**")
            # 将 package_b.PackageBObject 对象保存为 pickle 文件
            he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
            # 断言外部化的模块列表
            self.assertEqual(he.externed_modules(), ["package_b.subpackage_1"])
            # 断言模拟的模块列表
            self.assertEqual(he.mocked_modules(), ["package_b.subpackage_2"])
            # 断言内部化的模块列表
            self.assertEqual(
                he.interned_modules(),
                ["package_b", "package_b.subpackage_0.subsubpackage_0"],
            )
            # 断言 package_b.subpackage_2 的反向依赖关系
            self.assertEqual(he.get_rdeps("package_b.subpackage_2"), ["package_b"])

        # 测试在拒绝导出 package_b 的情况下引发 PackagingError
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(BytesIO()) as he:
                import package_b

                # 拒绝导出 package_b
                he.deny("package_b")
                # 将 package_b.PackageBObject 对象保存为 pickle 文件
                he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
                # 断言被拒绝导出的模块列表
                self.assertEqual(he.denied_modules(), ["package_b"])

    def test_is_from_package(self):
        """is_from_package should work for objects and modules"""
        import package_a.subpackage

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 创建 package_a.subpackage 的对象
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            # 内部化所有内容
            pe.intern("**")
            # 将对象保存为 pickle 文件
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        # 使用 PackageImporter 导入缓冲区中的内容
        pi = PackageImporter(buffer)
        # 导入 package_a.subpackage 模块
        mod = pi.import_module("package_a.subpackage")
        # 加载 pickle 文件中的对象
        loaded_obj = pi.load_pickle("obj", "obj.pkl")

        # 断言 package_a.subpackage 不是来自于包
        self.assertFalse(is_from_package(package_a.subpackage))
        # 断言 mod 是来自于包
        self.assertTrue(is_from_package(mod))

        # 断言 obj 不是来自于包
        self.assertFalse(is_from_package(obj))
        # 断言 loaded_obj 是来自于包
        self.assertTrue(is_from_package(loaded_obj))
    def test_inspect_class(self):
        """Should be able to retrieve source for a packaged class."""
        # 导入需要测试的包中的子包
        import package_a.subpackage

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 实例化一个包中的对象
        obj = package_a.subpackage.PackageASubpackageObject()

        # 使用 PackageExporter 将对象保存到字节流中
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 使用 PackageImporter 导入包
        pi = PackageImporter(buffer)
        # 获取导入的包中的类对象
        packaged_class = pi.import_module(
            "package_a.subpackage"
        ).PackageASubpackageObject
        # 直接获取包中类的对象
        regular_class = package_a.subpackage.PackageASubpackageObject

        # 使用 inspect 模块获取包中类的源代码行
        packaged_src = inspect.getsourcelines(packaged_class)
        # 直接获取非包装类的源代码行
        regular_src = inspect.getsourcelines(regular_class)
        # 断言包装类和非包装类的源代码应该相等
        self.assertEqual(packaged_src, regular_src)

    def test_dunder_package_present(self):
        """
        The attribute '__torch_package__' should be populated on imported modules.
        """
        # 导入需要测试的包中的子包
        import package_a.subpackage

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 实例化一个包中的对象
        obj = package_a.subpackage.PackageASubpackageObject()

        # 使用 PackageExporter 将对象保存到字节流中
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 使用 PackageImporter 导入包
        pi = PackageImporter(buffer)
        # 导入的模块中应该存在 '__torch_package__' 属性
        mod = pi.import_module("package_a.subpackage")
        self.assertTrue(hasattr(mod, "__torch_package__"))

    def test_dunder_package_works_from_package(self):
        """
        The attribute '__torch_package__' should be accessible from within
        the module itself, so that packaged code can detect whether it's
        being used in a packaged context or not.
        """
        # 导入使用 __torch_package__ 的模块
        import package_a.use_dunder_package as mod

        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 使用 PackageExporter 将模块保存到字节流中
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_module(mod.__name__)

        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 使用 PackageImporter 导入模块
        pi = PackageImporter(buffer)
        # 加载导入的模块
        imported_mod = pi.import_module(mod.__name__)
        # 断言导入的模块中的 is_from_package 方法返回 True
        self.assertTrue(imported_mod.is_from_package())
        # 断言原始模块中的 is_from_package 方法返回 False
        self.assertFalse(mod.is_from_package())

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_std_lib_sys_hackery_checks(self):
        """
        The standard library performs sys.module assignment hackery which
        causes modules who do this hackery to fail on import. See
        https://github.com/pytorch/pytorch/issues/57490 for more information.
        """
        # 导入需要测试的标准库系统模块
        import package_a.std_sys_module_hacks

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 实例化标准库系统模块对象
        mod = package_a.std_sys_module_hacks.Module()

        # 使用 PackageExporter 将对象保存到字节流中
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", mod)

        # 将缓冲区指针移到开头
        buffer.seek(0)
        # 使用 PackageImporter 加载从字节流中加载的对象
        pi = PackageImporter(buffer)
        mod = pi.load_pickle("obj", "obj.pkl")
        # 执行加载的对象
        mod()
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试或其他脚本中定义的主要功能
    run_tests()
```