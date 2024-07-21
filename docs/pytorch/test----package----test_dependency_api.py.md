# `.\pytorch\test\package\test_dependency_api.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入需要的模块和函数
import importlib
from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

# 导入 PyTorch 相关模块
import torch.nn

# 导入 Torch 的包管理相关异常和工具类
from torch.package import EmptyMatchError, Importer, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests

try:
    # 尝试从当前目录下的 common 模块导入 PackageTestCase
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则从全局范围导入
    # 支持直接运行本文件的情况
    from common import PackageTestCase

# 定义测试类 TestDependencyAPI，继承自 PackageTestCase
class TestDependencyAPI(PackageTestCase):
    """Dependency management API tests.
    - mock()
    - extern()
    - deny()
    """

    # 测试 extern 方法
    def test_extern(self):
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 包装字节流
        with PackageExporter(buffer) as he:
            # 声明依赖关系：导出 package_a.subpackage 和 module_a
            he.extern(["package_a.subpackage", "module_a"])
            # 保存字符串形式的源代码
            he.save_source_string("foo", "import package_a.subpackage; import module_a")
        buffer.seek(0)
        # 使用 PackageImporter 导入字节流中的内容
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        # 导入 module_a 并进行比较
        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        # 断言 module_a 为相同对象
        self.assertIs(module_a, module_a_im)
        # 断言 package_a 不是相同对象
        self.assertIsNot(package_a, package_a_im)
        # 断言 package_a.subpackage 是相同对象
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    # 测试 extern_glob 方法
    def test_extern_glob(self):
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 包装字节流
        with PackageExporter(buffer) as he:
            # 声明依赖关系：导出所有以 package_a. 开头的模块和所有以 module_ 开头的模块
            he.extern(["package_a.*", "module_*"])
            # 保存 package_a 模块
            he.save_module("package_a")
            # 保存字符串形式的源代码
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        # 使用 PackageImporter 导入字节流中的内容
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        # 导入 module_a 并进行比较
        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        # 断言 module_a 为相同对象
        self.assertIs(module_a, module_a_im)
        # 断言 package_a 不是相同对象
        self.assertIsNot(package_a, package_a_im)
        # 断言 package_a.subpackage 是相同对象
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    # 测试 extern_glob 方法，允许匹配为空的情况
    def test_extern_glob_allow_empty(self):
        """
        Test that an error is thrown when a extern glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        # 导入 package_a.subpackage 模块，预期将引发 EmptyMatchError 异常
        import package_a.subpackage  # noqa: F401

        buffer = BytesIO()
        # 使用 PackageExporter 导出时，指定 allow_empty=False，测试空匹配情况
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer) as exporter:
                exporter.extern(include=["package_b.*"], allow_empty=False)
                exporter.save_module("package_a.subpackage")
    def test_deny(self):
        """
        Test marking packages as "deny" during export.
        """
        # 创建一个字节流对象
        buffer = BytesIO()

        # 确保在导出时标记包为"deny"，并验证是否引发了PackagingError异常并包含"denied"字符串
        with self.assertRaisesRegex(PackagingError, "denied"):
            # 使用PackageExporter对象将包标记为"deny"
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.subpackage", "module_a"])
                # 将源字符串"import package_a.subpackage"保存到导出器中

    def test_deny_glob(self):
        """
        Test marking packages as "deny" using globs instead of package names.
        """
        # 创建一个字节流对象
        buffer = BytesIO()

        # 确保使用通配符而不是包名来标记包为"deny"，并验证是否引发PackagingError异常
        with self.assertRaises(PackagingError):
            # 使用PackageExporter对象将包标记为"deny"，支持通配符"package_a.*"和"module_*"
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.*", "module_*"])
                # 将多行源码保存到导出器中

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_mock(self):
        # 创建一个字节流对象
        buffer = BytesIO()

        # 使用PackageExporter对象模拟"package_a.subpackage"和"module_a"
        with PackageExporter(buffer) as he:
            he.mock(["package_a.subpackage", "module_a"])
            # 导出器中保存源字符串"import package_a.subpackage"

        # 将字节流指针移到起始位置
        buffer.seek(0)

        # 创建PackageImporter对象，导入之前导出的模块
        hi = PackageImporter(buffer)

        # 导入模块package_a.subpackage
        import package_a.subpackage

        # 对package_a.subpackage进行一个别名引用
        _ = package_a.subpackage

        # 导入模块module_a
        import module_a

        # 对module_a进行一个别名引用
        _ = module_a

        # 从导入的模块中获取package_a.subpackage模块
        m = hi.import_module("package_a.subpackage")

        # 从模块中获取结果
        r = m.result

        # 确保调用结果会引发NotImplementedError异常并包含"was mocked out"字符串
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_mock_glob(self):
        # 创建一个字节流对象
        buffer = BytesIO()

        # 使用PackageExporter对象模拟"package_a.*"和"module*"
        with PackageExporter(buffer) as he:
            he.mock(["package_a.*", "module*"])
            # 在导出器中保存模块"package_a"和多行源码

        # 将字节流指针移到起始位置
        buffer.seek(0)

        # 创建PackageImporter对象，导入之前导出的模块
        hi = PackageImporter(buffer)

        # 导入模块package_a.subpackage
        import package_a.subpackage

        # 对package_a.subpackage进行一个别名引用
        _ = package_a.subpackage

        # 导入模块module_a
        import module_a

        # 对module_a进行一个别名引用
        _ = module_a

        # 从导入的模块中获取package_a.subpackage模块
        m = hi.import_module("package_a.subpackage")

        # 从模块中获取结果
        r = m.result

        # 确保调用结果会引发NotImplementedError异常并包含"was mocked out"字符串
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()
    def test_mock_glob_allow_empty(self):
        """
        Test that an error is thrown when a mock glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        # 导入 package_a.subpackage 模块，忽略 F401 警告
        import package_a.subpackage  # noqa: F401

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 断言在 PackageExporter 上下文中抛出 EmptyMatchError 异常，
        # 错误消息应包含 "did not match any modules"
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            # 在 PackageExporter 上下文中创建 exporter 对象
            with PackageExporter(buffer) as exporter:
                # 设置 mock，包含 "package_b.*"，不允许为空
                exporter.mock(include=["package_b.*"], allow_empty=False)
                # 保存 module "package_a.subpackage"
                exporter.save_module("package_a.subpackage")

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_pickle_mocked(self):
        # 导入 package_a.subpackage 模块
        import package_a.subpackage

        # 创建 PackageASubpackageObject 实例
        obj = package_a.subpackage.PackageASubpackageObject()
        # 创建 PackageAObject 实例，传入 obj
        obj2 = package_a.PackageAObject(obj)

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 断言在 PackageExporter 上下文中抛出 PackagingError 异常
        with self.assertRaises(PackagingError):
            # 在 PackageExporter 上下文中创建 he 对象
            with PackageExporter(buffer) as he:
                # 设置 mock，包含 "package_a.subpackage"
                he.mock(include="package_a.subpackage")
                # 内部化所有内容
                he.intern("**")
                # 保存 obj2 为 pickle 格式文件 "obj.pkl"
                he.save_pickle("obj", "obj.pkl", obj2)

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_pickle_mocked_all(self):
        # 导入 package_a.subpackage 模块
        import package_a.subpackage

        # 创建 PackageASubpackageObject 实例
        obj = package_a.subpackage.PackageASubpackageObject()
        # 创建 PackageAObject 实例，传入 obj
        obj2 = package_a.PackageAObject(obj)

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 在 PackageExporter 上下文中创建 he 对象
        with PackageExporter(buffer) as he:
            # 内部化 "package_a.**" 下所有内容
            he.intern(include="package_a.**")
            # 设置 mock，包含所有内容
            he.mock("**")
            # 保存 obj2 为 pickle 格式文件 "obj.pkl"
            he.save_pickle("obj", "obj.pkl", obj2)

    def test_allow_empty_with_error(self):
        """
        If an error occurs during packaging, it should not be shadowed by the allow_empty error.
        """
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 断言在 PackageExporter 上下文中抛出 ModuleNotFoundError 异常
        with self.assertRaises(ModuleNotFoundError):
            # 在 PackageExporter 上下文中创建 pe 对象
            with PackageExporter(buffer) as pe:
                # 外部化 "foo" 模块，不允许为空
                pe.extern("foo", allow_empty=False)
                # 保存 module "aodoifjodisfj"，将会出错
                pe.save_module("aodoifjodisfj")

                # 这里永远不会执行到，因此 allow_empty 检查应该引发一个错误。
                # 然而上面的错误更详细地显示了出错信息。
                pe.save_source_string("bar", "import foo\n")

    def test_implicit_intern(self):
        """
        The save_module APIs should implicitly intern the module being saved.
        """
        # 导入 package_a 模块，忽略 F401 警告
        import package_a  # noqa: F401

        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 在 PackageExporter 上下文中创建 he 对象
        with PackageExporter(buffer) as he:
            # 保存 module "package_a"，此操作应隐式地内部化模块
            he.save_module("package_a")
    def test_intern_error(self):
        """测试无法处理所有依赖关系应导致错误。"""

        import package_a.subpackage  # 导入package_a.subpackage模块

        obj = package_a.subpackage.PackageASubpackageObject()  # 实例化subpackage模块中的PackageASubpackageObject对象
        obj2 = package_a.PackageAObject(obj)  # 使用obj实例化package_a模块中的PackageAObject对象

        buffer = BytesIO()  # 创建一个BytesIO对象，用于缓存数据

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as he:  # 使用buffer创建PackageExporter对象he
                he.save_pickle("obj", "obj.pkl", obj2)  # 将obj2以Pickle格式保存到he中的obj.pkl文件中

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module did not match against any action pattern. Extern, mock, or intern it.
                    package_a
                    package_a.subpackage

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

        # Interning all dependencies should work
        with PackageExporter(buffer) as he:  # 重新使用buffer创建PackageExporter对象he
            he.intern(["package_a", "package_a.subpackage"])  # 内部化所有依赖项
            he.save_pickle("obj", "obj.pkl", obj2)  # 将obj2以Pickle格式保存到he中的obj.pkl文件中

    @skipIf(IS_WINDOWS, "extension modules have a different file extension on windows")
    def test_broken_dependency(self):
        """一个无法打包的依赖项应引发PackagingError异常。"""

        def create_module(name):
            spec = importlib.machinery.ModuleSpec(name, self, is_package=False)  # 创建一个模块规范对象spec
            module = importlib.util.module_from_spec(spec)  # 使用spec创建模块对象module
            ns = module.__dict__
            ns["__spec__"] = spec  # 设置模块的__spec__属性
            ns["__loader__"] = self  # 设置模块的__loader__属性
            ns["__file__"] = f"{name}.so"  # 设置模块的__file__属性为name.so
            ns["__cached__"] = None  # 设置模块的__cached__属性为None
            return module

        class BrokenImporter(Importer):
            def __init__(self):
                self.modules = {
                    "foo": create_module("foo"),  # 创建名为foo的模块对象
                    "bar": create_module("bar"),  # 创建名为bar的模块对象
                }

            def import_module(self, module_name):
                return self.modules[module_name]  # 返回指定名称的模块对象

        buffer = BytesIO()  # 创建一个BytesIO对象，用于缓存数据

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer, importer=BrokenImporter()) as exporter:  # 使用buffer和自定义的BrokenImporter对象创建PackageExporter对象exporter
                exporter.intern(["foo", "bar"])  # 内部化依赖项foo和bar
                exporter.save_source_string("my_module", "import foo; import bar")  # 将字符串"import foo; import bar"保存到exporter中的my_module文件中

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module is a C extension module. torch.package supports Python modules only.
                    foo
                    bar

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )
    def test_invalid_import(self):
        """An incorrectly-formed import should raise a PackagingError."""
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 上下文管理器，并期望抛出 PackagingError 异常
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as exporter:
                # 尝试保存一个错误格式的导入语句
                exporter.save_source_string("foo", "from ........ import lol")

        # 断言捕获到的异常消息与预期的多行文本相匹配
        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Dependency resolution failed.
                    foo
                      Context: attempted relative import beyond top-level package

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

    @skipIf(version_info < (3, 7), "mock uses __getattr__ a 3.7 feature")
    def test_repackage_mocked_module(self):
        """Re-packaging a package that contains a mocked module should work correctly."""
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 上下文管理器
        with PackageExporter(buffer) as exporter:
            # 模拟 package_a
            exporter.mock("package_a")
            # 保存一个导入语句到 "foo" 模块
            exporter.save_source_string("foo", "import package_a")

        # 将缓冲区指针移到起始位置
        buffer.seek(0)
        # 使用 PackageImporter 导入模型
        importer = PackageImporter(buffer)
        foo = importer.import_module("foo")

        # 断言对 "package_a" 的调用会引发 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            foo.package_a.get_something()

        # 再次尝试重新打包模型，内部化之前模拟的模块并模拟其它所有内容
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.intern("package_a")
            exporter.mock("**")
            exporter.save_source_string("foo", "import package_a")

        buffer2.seek(0)
        importer2 = PackageImporter(buffer2)
        foo2 = importer2.import_module("foo")

        # 断言对 "package_a" 的调用仍然会引发 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            foo2.package_a.get_something()

    def test_externing_c_extension(self):
        """Externing c extensions modules should allow us to still access them especially those found in torch._C."""
        # 创建一个字节流对象
        buffer = BytesIO()
        # 创建一个 TransformerEncoderLayer 模型
        model = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=64,
            dropout=1.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # 使用 PackageExporter 上下文管理器
        with PackageExporter(buffer) as e:
            # extern 所有以 "torch." 开头的模块
            e.extern("torch.**")
            # intern 所有模块
            e.intern("**")

            # 将模型保存为 pickle 格式
            e.save_pickle("model", "model.pkl", model)
        # 将缓冲区指针移到起始位置
        buffer.seek(0)
        # 使用 PackageImporter 导入模型
        imp = PackageImporter(buffer)
        imp.load_pickle("model", "model.pkl")
# 如果当前脚本被直接执行而非被导入，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```