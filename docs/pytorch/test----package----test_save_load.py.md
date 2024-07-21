# `.\pytorch\test\package\test_save_load.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle
# 导入 BytesIO 类，用于在内存中操作二进制数据流
from io import BytesIO
# 导入 dedent 函数，用于删除多行文本开头的缩进
from textwrap import dedent

# 导入 torch 的打包和导入相关模块
from torch.package import PackageExporter, PackageImporter, sys_importer
# 导入 torch 内部的测试工具函数
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前目录导入 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果导入失败，支持直接运行该文件的情况，从 common 模块导入 PackageTestCase 类
    from common import PackageTestCase

# 导入 Path 类，用于处理文件路径
from pathlib import Path

# 获取当前文件的父目录路径
packaging_directory = Path(__file__).parent

# 定义测试类 TestSaveLoad，继承自 PackageTestCase
class TestSaveLoad(PackageTestCase):
    """Core save_* and loading API tests."""

    # 测试方法：测试保存源文件的功能
    def test_saving_source(self):
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将数据保存到 buffer 中
        with PackageExporter(buffer) as he:
            # 保存指定名称的源文件到打包对象中
            he.save_source_file("foo", str(packaging_directory / "module_a.py"))
            # 保存指定目录下所有文件到打包对象中
            he.save_source_file("foodir", str(packaging_directory / "package_a"))
        # 将 buffer 的读取位置设置到开头
        buffer.seek(0)
        # 使用 PackageImporter 从 buffer 中导入数据
        hi = PackageImporter(buffer)
        # 导入名为 "foo" 的模块
        foo = hi.import_module("foo")
        # 导入名为 "foodir.subpackage" 的模块
        s = hi.import_module("foodir.subpackage")
        # 断言 foo 模块的结果为 "module_a"
        self.assertEqual(foo.result, "module_a")
        # 断言 s 模块的结果为 "package_a.subpackage"
        self.assertEqual(s.result, "package_a.subpackage")

    # 测试方法：测试保存字符串形式源代码的功能
    def test_saving_string(self):
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将数据保存到 buffer 中
        with PackageExporter(buffer) as he:
            # 定义一个多行文本，并删除开头的缩进
            src = dedent(
                """\
                import math
                the_math = math
                """
            )
            # 将定义的源代码字符串保存为名为 "my_mod" 的模块
            he.save_source_string("my_mod", src)
        # 将 buffer 的读取位置设置到开头
        buffer.seek(0)
        # 使用 PackageImporter 从 buffer 中导入数据
        hi = PackageImporter(buffer)
        # 导入名为 "math" 的模块
        m = hi.import_module("math")
        import math

        # 断言导入的 m 与标准库中的 math 模块相同
        self.assertIs(m, math)
        # 导入名为 "my_mod" 的模块
        my_mod = hi.import_module("my_mod")
        # 断言 my_mod 模块中的 math 对象与标准库中的 math 模块相同
        self.assertIs(my_mod.math, math)

    # 测试方法：测试保存已导入模块的功能
    def test_save_module(self):
        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将数据保存到 buffer 中
        with PackageExporter(buffer) as he:
            # 导入 module_a 模块并保存
            import module_a
            he.save_module(module_a.__name__)
            # 导入 package_a 模块并保存
            import package_a
            he.save_module(package_a.__name__)
        # 将 buffer 的读取位置设置到开头
        buffer.seek(0)
        # 使用 PackageImporter 从 buffer 中导入数据
        hi = PackageImporter(buffer)
        # 导入名为 "module_a" 的模块
        module_a_i = hi.import_module("module_a")
        # 断言 module_a_i 模块的结果为 "module_a"
        self.assertEqual(module_a_i.result, "module_a")
        # 断言导入的 module_a 与 module_a_i 不是同一个对象
        self.assertIsNot(module_a, module_a_i)
        # 导入名为 "package_a" 的模块
        package_a_i = hi.import_module("package_a")
        # 断言 package_a_i 模块的结果为 "package_a"
        self.assertEqual(package_a_i.result, "package_a")
        # 断言导入的 package_a_i 与 package_a 不是同一个对象
        self.assertIsNot(package_a_i, package_a)
    def test_dunder_imports(self):
        """Test importing various modules and objects using PackageExporter and PackageImporter."""
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            # Importing package_b to export its object
            import package_b

            # Getting the object from package_b
            obj = package_b.PackageBObject
            # Interning a symbol
            he.intern("**")
            # Saving the object as a pickle
            he.save_pickle("res", "obj.pkl", obj)

        buffer.seek(0)
        # Creating a PackageImporter instance to import from buffer
        hi = PackageImporter(buffer)
        # Loading the pickled object
        loaded_obj = hi.load_pickle("res", "obj.pkl")

        # Importing module package_b
        package_b = hi.import_module("package_b")
        # Asserting the result attribute of package_b
        self.assertEqual(package_b.result, "package_b")

        # Importing built-in module math
        math = hi.import_module("math")
        # Asserting the __name__ attribute of math
        self.assertEqual(math.__name__, "math")

        # Importing a sub-sub package of xml.sax
        xml_sub_sub_package = hi.import_module("xml.sax.xmlreader")
        # Asserting the __name__ attribute of xml_sub_sub_package
        self.assertEqual(xml_sub_sub_package.__name__, "xml.sax.xmlreader")

        # Importing subpackage_1 from package_b
        subpackage_1 = hi.import_module("package_b.subpackage_1")
        # Asserting the result attribute of subpackage_1
        self.assertEqual(subpackage_1.result, "subpackage_1")

        # Importing subpackage_2 from package_b
        subpackage_2 = hi.import_module("package_b.subpackage_2")
        # Asserting the result attribute of subpackage_2
        self.assertEqual(subpackage_2.result, "subpackage_2")

        # Importing subsubpackage_0 from package_b.subpackage_0
        subsubpackage_0 = hi.import_module("package_b.subpackage_0.subsubpackage_0")
        # Asserting the result attribute of subsubpackage_0
        self.assertEqual(subsubpackage_0.result, "subsubpackage_0")

    def test_bad_dunder_imports(self):
        """Test to ensure bad __imports__ don't cause PackageExporter to fail."""
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            # Saving a source string that includes a non-standard import
            e.save_source_string(
                "m", '__import__(these, unresolvable, "things", wont, crash, me)'
            )

    def test_save_module_binary(self):
        """Test saving and importing modules using PackageExporter and PackageImporter."""
        f = BytesIO()
        with PackageExporter(f) as he:
            # Importing module_a to save it
            import module_a
            # Importing package_a to save it
            import package_a

            # Saving module_a
            he.save_module(module_a.__name__)
            # Saving package_a
            he.save_module(package_a.__name__)
        f.seek(0)
        # Creating a PackageImporter instance to import from f
        hi = PackageImporter(f)
        # Importing module_a from the package
        module_a_i = hi.import_module("module_a")
        # Asserting the result attribute of module_a_i
        self.assertEqual(module_a_i.result, "module_a")
        # Ensuring module_a and module_a_i are not the same object
        self.assertIsNot(module_a, module_a_i)
        # Importing package_a from the package
        package_a_i = hi.import_module("package_a")
        # Asserting the result attribute of package_a_i
        self.assertEqual(package_a_i.result, "package_a")
        # Ensuring package_a and package_a_i are not the same object
        self.assertIsNot(package_a_i, package_a)

    def test_pickle(self):
        """Test saving and loading objects using pickle with PackageExporter and PackageImporter."""
        # Importing package_a.subpackage to use its object
        import package_a.subpackage

        # Creating an object from package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        # Creating another object using obj
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            # Interning all symbols
            he.intern("**")
            # Saving obj2 as a pickle
            he.save_pickle("obj", "obj.pkl", obj2)
        buffer.seek(0)
        # Creating a PackageImporter instance to import from buffer
        hi = PackageImporter(buffer)

        # Checking we imported dependencies like package_a.subpackage
        sp = hi.import_module("package_a.subpackage")
        # Checking we didn't import module_a
        with self.assertRaises(ImportError):
            hi.import_module("module_a")

        # Loading the pickled object
        obj_loaded = hi.load_pickle("obj", "obj.pkl")
        # Ensuring obj2 and obj_loaded are not the same object
        self.assertIsNot(obj2, obj_loaded)
        # Ensuring obj_loaded.obj is an instance of PackageASubpackageObject
        self.assertIsInstance(obj_loaded.obj, sp.PackageASubpackageObject)
        # Ensuring PackageASubpackageObject is not the same as sp.PackageASubpackageObject
        self.assertIsNot(
            package_a.subpackage.PackageASubpackageObject, sp.PackageASubpackageObject
        )
    def test_pickle_long_name_with_protocol_4(self):
        import package_a.long_name  # 导入 package_a.long_name 模块

        container = []  # 创建一个空列表 container

        # 避免将一个长达 256 字符的函数直接粘贴到测试中，间接获取函数并添加到 container 中
        package_a.long_name.add_function(container)

        buffer = BytesIO()  # 创建一个 BytesIO 对象 buffer
        with PackageExporter(buffer) as exporter:  # 使用 PackageExporter 导出器处理 buffer
            exporter.intern("**")  # 在导出器中内部化字符串 "**"
            exporter.save_pickle(
                "container", "container.pkl", container, pickle_protocol=4
            )  # 使用协议 4 保存 container 到名为 container.pkl 的文件中

        buffer.seek(0)  # 将 buffer 指针位置移动到起始位置
        importer = PackageImporter(buffer)  # 创建一个 PackageImporter 对象 importer
        unpickled_container = importer.load_pickle("container", "container.pkl")  # 从文件中加载并反序列化 container
        self.assertIsNot(container, unpickled_container)  # 断言 container 与反序列化后的 unpickled_container 不是同一个对象
        self.assertEqual(len(unpickled_container), 1)  # 断言 unpickled_container 的长度为 1
        self.assertEqual(container[0](), unpickled_container[0]())  # 断言 container 和 unpickled_container 中第一个元素的调用结果相等

    def test_exporting_mismatched_code(self):
        """
        If an object with the same qualified name is loaded from different
        packages, the user should get an error if they try to re-save the
        object with the wrong package's source code.
        """
        import package_a.subpackage  # 导入 package_a.subpackage 模块

        obj = package_a.subpackage.PackageASubpackageObject()  # 创建 package_a.subpackage.PackageASubpackageObject 的实例 obj
        obj2 = package_a.PackageAObject(obj)  # 创建 package_a.PackageAObject 的实例 obj2，使用 obj 作为参数

        b1 = BytesIO()  # 创建一个 BytesIO 对象 b1
        with PackageExporter(b1) as pe:  # 使用 PackageExporter 导出器处理 b1
            pe.intern("**")  # 在导出器中内部化字符串 "**"
            pe.save_pickle("obj", "obj.pkl", obj2)  # 将 obj2 保存为 "obj.pkl" 文件

        b1.seek(0)  # 将 b1 指针位置移动到起始位置
        importer1 = PackageImporter(b1)  # 创建一个 PackageImporter 对象 importer1
        loaded1 = importer1.load_pickle("obj", "obj.pkl")  # 从 "obj.pkl" 文件中加载并反序列化对象

        b1.seek(0)  # 将 b1 指针位置再次移动到起始位置
        importer2 = PackageImporter(b1)  # 创建另一个 PackageImporter 对象 importer2
        loaded2 = importer2.load_pickle("obj", "obj.pkl")  # 从 "obj.pkl" 文件中加载并反序列化对象

        def make_exporter():
            pe = PackageExporter(BytesIO(), importer=[importer1, sys_importer])  # 创建一个新的 PackageExporter 对象 pe，导入 importer1 和 sys_importer
            # 确保导入器首先找到 'PackageAObject' 在 'importer1' 中定义的版本。
            return pe  # 返回创建的 PackageExporter 对象 pe

        # 这应该失败。'importer1' 中定义的 'PackageAObject' 类型不一定与 'obj2' 的 'PackageAObject' 版本相同。
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):  # 断言捕获到 pickle.PicklingError 异常
            pe.save_pickle("obj", "obj.pkl", obj2)  # 尝试使用错误的包的源代码重新保存对象，应该触发异常

        # 这也应该失败。'importer1' 中定义的 'PackageAObject' 类型不一定与 'importer2' 中定义的相同。
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):  # 断言捕获到 pickle.PicklingError 异常
            pe.save_pickle("obj", "obj.pkl", loaded2)  # 尝试使用不匹配的包重新保存对象，应该触发异常

        # 这应该成功。'importer1' 中定义的 'PackageAObject' 类型与 loaded1 中使用的版本匹配。
        pe = make_exporter()
        pe.save_pickle("obj", "obj.pkl", loaded1)  # 使用正确的包重新保存 loaded1 对象，应该成功保存
    def test_save_imported_module(self):
        """Saving a module that came from another PackageImporter should work."""
        # 导入需要测试的模块
        import package_a.subpackage

        # 创建 package_a.subpackage 包中的对象
        obj = package_a.subpackage.PackageASubpackageObject()
        # 使用 package_a 包中的对象创建新的对象
        obj2 = package_a.PackageAObject(obj)

        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将数据导出到 buffer 中
        with PackageExporter(buffer) as exporter:
            # 设置导出的参数
            exporter.intern("**")
            # 将 obj2 对象以 pickle 格式保存到 model.pkl 文件中
            exporter.save_pickle("model", "model.pkl", obj2)

        # 将读取指针移动到流的开头
        buffer.seek(0)

        # 创建一个 PackageImporter 对象用于导入数据
        importer = PackageImporter(buffer)
        # 从 model.pkl 文件中加载 pickle 格式的对象
        imported_obj2 = importer.load_pickle("model", "model.pkl")
        # 获取导入对象的模块名
        imported_obj2_module = imported_obj2.__class__.__module__

        # 应该能够成功导出，没有错误发生
        # 创建另一个字节流对象
        buffer2 = BytesIO()
        # 使用 PackageExporter 将数据导出到 buffer2 中
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            # 设置导出的参数
            exporter.intern("**")
            # 将 imported_obj2_module 模块保存到导出流中
            exporter.save_module(imported_obj2_module)

    def test_save_imported_module_using_package_importer(self):
        """Exercise a corner case: re-packaging a module that uses `torch_package_importer`"""
        # 导入需要测试的模块，禁止出现 F401 错误
        import package_a.use_torch_package_importer  # noqa: F401

        # 创建一个字节流对象
        buffer = BytesIO()
        # 使用 PackageExporter 将数据导出到 buffer 中
        with PackageExporter(buffer) as exporter:
            # 设置导出的参数
            exporter.intern("**")
            # 将 package_a.use_torch_package_importer 模块保存到导出流中
            exporter.save_module("package_a.use_torch_package_importer")

        # 将读取指针移动到流的开头
        buffer.seek(0)

        # 创建一个 PackageImporter 对象用于导入数据
        importer = PackageImporter(buffer)

        # 应该能够成功导出，没有错误发生
        # 创建另一个字节流对象
        buffer2 = BytesIO()
        # 使用 PackageExporter 将数据导出到 buffer2 中
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            # 设置导出的参数
            exporter.intern("**")
            # 将 package_a.use_torch_package_importer 模块保存到导出流中
            exporter.save_module("package_a.use_torch_package_importer")
# 如果当前脚本作为主程序运行，则执行 `run_tests()` 函数
if __name__ == "__main__":
    run_tests()
```