# `.\pytorch\test\package\test_importer.py`

```py
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO  # 导入BytesIO类，用于创建二进制数据的缓冲区

import torch  # 导入torch库
from torch.package import (  # 从torch.package模块导入以下类和函数
    Importer,
    OrderedImporter,
    PackageExporter,
    PackageImporter,
    sys_importer,
)
from torch.testing._internal.common_utils import run_tests  # 导入测试相关的函数和类

try:
    from .common import PackageTestCase  # 尝试从当前目录导入common模块的PackageTestCase类
except ImportError:
    # 如果导入失败，则支持直接运行该文件的情况
    from common import PackageTestCase  # 从common模块导入PackageTestCase类


class TestImporter(PackageTestCase):
    """Tests for Importer and derived classes."""

    def test_sys_importer(self):
        import package_a  # 导入package_a模块
        import package_a.subpackage  # 导入package_a的subpackage子模块

        # 断言sys_importer正确导入package_a模块
        self.assertIs(sys_importer.import_module("package_a"), package_a)
        # 断言sys_importer正确导入package_a.subpackage子模块
        self.assertIs(
            sys_importer.import_module("package_a.subpackage"), package_a.subpackage
        )

    def test_sys_importer_roundtrip(self):
        import package_a  # 导入package_a模块
        import package_a.subpackage  # 导入package_a的subpackage子模块

        importer = sys_importer  # 将sys_importer赋值给importer变量
        type_ = package_a.subpackage.PackageASubpackageObject  # 获取subpackage中的类型PackageASubpackageObject
        module_name, type_name = importer.get_name(type_)  # 获取类型对应的模块名称和类型名称

        module = importer.import_module(module_name)  # 使用importer导入对应模块
        self.assertIs(getattr(module, type_name), type_)  # 断言模块中的类型名称与type_相同

    def test_single_ordered_importer(self):
        import module_a  # 导入module_a模块，忽略F401警告

        import package_a  # 导入package_a模块

        buffer = BytesIO()  # 创建一个BytesIO对象作为缓冲区
        with PackageExporter(buffer) as pe:
            pe.save_module(package_a.__name__)  # 将package_a模块保存到缓冲区

        buffer.seek(0)  # 将缓冲区指针移动到开头
        importer = PackageImporter(buffer)  # 使用缓冲区创建PackageImporter对象

        # 构建一个仅包含importer的导入环境
        ordered_importer = OrderedImporter(importer)

        # 这个环境返回的模块应该与importer中的相同
        self.assertIs(
            ordered_importer.import_module("package_a"),
            importer.import_module("package_a"),
        )
        # 它不应该是外部Python环境中的模块
        self.assertIsNot(ordered_importer.import_module("package_a"), package_a)

        # 我们没有打包这个模块，所以它不应该可用
        with self.assertRaises(ModuleNotFoundError):
            ordered_importer.import_module("module_a")

    def test_ordered_importer_basic(self):
        import package_a  # 导入package_a模块

        buffer = BytesIO()  # 创建一个BytesIO对象作为缓冲区
        with PackageExporter(buffer) as pe:
            pe.save_module(package_a.__name__)  # 将package_a模块保存到缓冲区

        buffer.seek(0)  # 将缓冲区指针移动到开头
        importer = PackageImporter(buffer)  # 使用缓冲区创建PackageImporter对象

        ordered_importer_sys_first = OrderedImporter(sys_importer, importer)
        # 断言在优先使用sys_importer的顺序导入器中，package_a模块与package_a相同
        self.assertIs(ordered_importer_sys_first.import_module("package_a"), package_a)

        ordered_importer_package_first = OrderedImporter(importer, sys_importer)
        # 断言在优先使用importer的顺序导入器中，package_a模块与importer中的相同
        self.assertIs(
            ordered_importer_package_first.import_module("package_a"),
            importer.import_module("package_a"),
        )
    def test_ordered_importer_whichmodule(self):
        """OrderedImporter's implementation of whichmodule should try each
        underlying importer's whichmodule in order.
        """

        # 定义一个测试用的导入器类 DummyImporter
        class DummyImporter(Importer):
            def __init__(self, whichmodule_return):
                self._whichmodule_return = whichmodule_return

            def import_module(self, module_name):
                raise NotImplementedError

            # 实现 whichmodule 方法，返回预设的 whichmodule_return
            def whichmodule(self, obj, name):
                return self._whichmodule_return

        # 定义一个简单的类 DummyClass
        class DummyClass:
            pass

        # 创建三个 DummyImporter 实例，分别返回 "foo", "bar", "__main__"
        dummy_importer_foo = DummyImporter("foo")
        dummy_importer_bar = DummyImporter("bar")
        dummy_importer_not_found = DummyImporter(
            "__main__"
        )  # __main__ is used as a proxy for "not found" by CPython

        # 创建一个 OrderedImporter 实例，按顺序尝试 foo 和 bar 的导入器
        foo_then_bar = OrderedImporter(dummy_importer_foo, dummy_importer_bar)
        # 断言调用 whichmodule 返回 "foo"
        self.assertEqual(foo_then_bar.whichmodule(DummyClass(), ""), "foo")

        # 创建一个 OrderedImporter 实例，按顺序尝试 bar 和 foo 的导入器
        bar_then_foo = OrderedImporter(dummy_importer_bar, dummy_importer_foo)
        # 断言调用 whichmodule 返回 "bar"
        self.assertEqual(bar_then_foo.whichmodule(DummyClass(), ""), "bar")

        # 创建一个 OrderedImporter 实例，按顺序尝试 __main__ 和 foo 的导入器
        notfound_then_foo = OrderedImporter(
            dummy_importer_not_found, dummy_importer_foo
        )
        # 断言调用 whichmodule 返回 "foo"
        self.assertEqual(notfound_then_foo.whichmodule(DummyClass(), ""), "foo")

    def test_package_importer_whichmodule_no_dunder_module(self):
        """Exercise corner case where we try to pickle an object whose
        __module__ doesn't exist because it's from a C extension.
        """
        # torch.float16 is an example of such an object: it is a C extension
        # type for which there is no __module__ defined. The default pickler
        # finds it using special logic to traverse sys.modules and look up
        # `float16` on each module (see pickle.py:whichmodule).
        #
        # We must ensure that we emulate the same behavior from PackageImporter.
        
        # 使用 torch.float16 作为例子，它是一个来自 C 扩展的对象，没有定义 __module__
        my_dtype = torch.float16

        # 设置一个 PackageExporter 实例，将 my_dtype 保存为 pickle 格式
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_dtype)
        buffer.seek(0)

        # 创建 PackageImporter 实例，从 buffer 中加载 pickle 数据
        importer = PackageImporter(buffer)
        my_loaded_dtype = importer.load_pickle("foo", "foo.pkl")

        # 再次保存一个只含有 PackageImporter 的包
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_loaded_dtype)

        buffer2.seek(0)

        # 创建另一个 PackageImporter 实例，从 buffer2 中加载 pickle 数据
        importer2 = PackageImporter(buffer2)
        my_loaded_dtype2 = importer2.load_pickle("foo", "foo.pkl")

        # 断言两次加载的对象是同一个对象
        self.assertIs(my_dtype, my_loaded_dtype)
        self.assertIs(my_dtype, my_loaded_dtype2)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试用例
    run_tests()
```