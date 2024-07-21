# `.\pytorch\test\package\test_resources.py`

```
# Owner(s): ["oncall: package/deploy"]

# 导入所需模块和函数
from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

# 导入 torch 的包管理相关模块
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前目录导入 PackageTestCase
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则从顶层目录导入 PackageTestCase
    from common import PackageTestCase


@skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
class TestResources(PackageTestCase):
    """Tests for access APIs for packaged resources."""

    def test_resource_reader(self):
        """Test compliance with the get_resource_reader importlib API."""
        
        # 创建一个内存中的缓冲区
        buffer = BytesIO()
        
        # 使用 PackageExporter 将内容保存到缓冲区中
        with PackageExporter(buffer) as pe:
            # 定义一个虚拟的包结构并保存文本内容到包中
            pe.save_text("one", "a.txt", "hello, a!")
            pe.save_text("one", "b.txt", "hello, b!")
            pe.save_text("one", "c.txt", "hello, c!")

            pe.save_text("one.three", "d.txt", "hello, d!")
            pe.save_text("one.three", "e.txt", "hello, e!")

            pe.save_text("two", "f.txt", "hello, f!")
            pe.save_text("two", "g.txt", "hello, g!")
        
        # 将缓冲区指针移动到起始位置
        buffer.seek(0)
        
        # 使用 PackageImporter 从缓冲区中导入包
        importer = PackageImporter(buffer)

        # 获取名为 "one" 的资源读取器
        reader_one = importer.get_resource_reader("one")
        
        # 验证尝试访问不存在的文件时抛出 FileNotFoundError 异常
        with self.assertRaises(FileNotFoundError):
            reader_one.resource_path("a.txt")

        # 验证资源 "a.txt" 是否存在
        self.assertTrue(reader_one.is_resource("a.txt"))
        
        # 验证打开并读取资源 "a.txt" 的内容是否为 "hello, a!"
        self.assertEqual(reader_one.open_resource("a.txt").getbuffer(), b"hello, a!")
        
        # 验证资源 "three" 是否存在
        self.assertFalse(reader_one.is_resource("three"))
        
        # 获取 "one" 中所有内容的列表
        reader_one_contents = list(reader_one.contents())
        
        # 验证 "one" 中的内容列表是否与预期一致
        self.assertSequenceEqual(
            reader_one_contents, ["a.txt", "b.txt", "c.txt", "three"]
        )

        # 获取名为 "two" 的资源读取器
        reader_two = importer.get_resource_reader("two")
        
        # 验证资源 "f.txt" 是否存在
        self.assertTrue(reader_two.is_resource("f.txt"))
        
        # 验证打开并读取资源 "f.txt" 的内容是否为 "hello, f!"
        self.assertEqual(reader_two.open_resource("f.txt").getbuffer(), b"hello, f!")
        
        # 获取 "two" 中所有内容的列表
        reader_two_contents = list(reader_two.contents())
        
        # 验证 "two" 中的内容列表是否与预期一致
        self.assertSequenceEqual(reader_two_contents, ["f.txt", "g.txt"])

        # 获取名为 "one.three" 的资源读取器
        reader_one_three = importer.get_resource_reader("one.three")
        
        # 验证资源 "d.txt" 是否存在
        self.assertTrue(reader_one_three.is_resource("d.txt"))
        
        # 验证打开并读取资源 "d.txt" 的内容是否为 "hello, d!"
        self.assertEqual(
            reader_one_three.open_resource("d.txt").getbuffer(), b"hello, d!"
        )
        
        # 获取 "one.three" 中所有内容的列表
        reader_one_three_contents = list(reader_one_three.contents())
        
        # 验证 "one.three" 中的内容列表是否与预期一致
        self.assertSequenceEqual(reader_one_three_contents, ["d.txt", "e.txt"])

        # 验证尝试获取不存在的包 "nonexistent_package" 时返回 None
        self.assertIsNone(importer.get_resource_reader("nonexistent_package"))
    def test_package_resource_access(self):
        """测试打包的模块能否使用 importlib.resources API 访问包中保存的资源。"""
        # 构建模块源码字符串
        mod_src = dedent(
            """\
            import importlib.resources
            import my_cool_resources

            def secret_message():
                return importlib.resources.read_text(my_cool_resources, 'sekrit.txt')
            """
        )
        # 创建字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将模块源码保存到缓冲区
        with PackageExporter(buffer) as pe:
            pe.save_source_string("foo.bar", mod_src)
            # 在包中保存文本资源
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")

        buffer.seek(0)
        # 创建 PackageImporter 实例用于导入包
        importer = PackageImporter(buffer)
        # 断言导入模块并调用 secret_message 函数返回正确的文本
        self.assertEqual(
            importer.import_module("foo.bar").secret_message(), "my sekrit plays"
        )

    def test_importer_access(self):
        # 创建字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将文本和二进制数据保存到缓冲区
        with PackageExporter(buffer) as he:
            he.save_text("main", "main", "my string")
            he.save_binary("main", "main_binary", b"my string")
            # 构建源码字符串
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                t = resources.load_text('main', 'main')
                b = resources.load_binary('main', 'main_binary')
                """
            )
            # 在包中保存源码字符串
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        # 创建 PackageImporter 实例用于导入包
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        # 断言导入的模块中的 t 和 b 变量值正确
        self.assertEqual(m.t, "my string")
        self.assertEqual(m.b, b"my string")

    def test_resource_access_by_path(self):
        """
        测试打包的代码能否使用 importlib.resources.path 访问资源。
        """
        # 创建字节流缓冲区
        buffer = BytesIO()
        # 使用 PackageExporter 将二进制数据保存到缓冲区
        with PackageExporter(buffer) as he:
            he.save_binary("string_module", "my_string", b"my string")
            # 构建源码字符串
            src = dedent(
                """\
                import importlib.resources
                import string_module

                with importlib.resources.path(string_module, 'my_string') as path:
                    with open(path, mode='r', encoding='utf-8') as f:
                        s = f.read()
                """
            )
            # 在包中保存源码字符串
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        # 创建 PackageImporter 实例用于导入包
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        # 断言导入的模块中的 s 变量值正确
        self.assertEqual(m.s, "my string")
# 如果当前脚本作为主程序运行，执行以下代码
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```