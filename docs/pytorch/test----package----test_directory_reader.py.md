# `.\pytorch\test\package\test_directory_reader.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入所需的模块和库
import os
import zipfile
from sys import version_info
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest import skipIf

# 导入 torch 相关模块
import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
)

# 尝试导入 torchvision 的 resnet18 模型，标志是否成功
try:
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
# 如果导入失败，标志为 False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")

# 尝试导入本地的 PackageTestCase，处理 ImportError 的情况
try:
    from .common import PackageTestCase
except ImportError:
    # 支持直接运行此文件的情况
    from common import PackageTestCase

# 获取当前文件所在目录的路径
from pathlib import Path
packaging_directory = Path(__file__).parent

# 装饰器：跳过在特定环境下运行的测试用例
@skipIf(
    IS_FBCODE or IS_SANDCASTLE or IS_WINDOWS,
    "Tests that use temporary files are disabled in fbcode",
)
class DirectoryReaderTest(PackageTestCase):
    """Tests use of DirectoryReader as accessor for opened packages."""

    # 装饰器：如果没有 torchvision，则跳过相关测试
    @skipIfNoTorchVision
    @skipIf(
        True,
        "Does not work with latest TorchVision, see https://github.com/pytorch/pytorch/issues/81115",
    )
    def test_loading_pickle(self):
        """
        Test basic saving and loading of modules and pickles from a DirectoryReader.
        """
        # 使用 resnet18 模型进行测试
        resnet = resnet18()

        # 创建临时文件名
        filename = self.temp()
        # 使用 PackageExporter 导出模型和数据到压缩包
        with PackageExporter(filename) as e:
            e.intern("**")
            e.save_pickle("model", "model.pkl", resnet)

        # 读取创建的压缩包
        zip_file = zipfile.ZipFile(filename, "r")

        # 创建临时目录并解压缩压缩包到临时目录
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 使用 PackageImporter 加载解压后的压缩包
            importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            # 加载并获取模型对象
            dir_mod = importer.load_pickle("model", "model.pkl")
            # 创建随机输入
            input = torch.rand(1, 3, 224, 224)
            # 验证加载的模型处理输入与原始模型的处理结果是否一致
            self.assertEqual(dir_mod(input), resnet(input))

    def test_loading_module(self):
        """
        Test basic saving and loading of a packages from a DirectoryReader.
        """
        # 导入要测试的 package_a 模块
        import package_a

        # 创建临时文件名
        filename = self.temp()
        # 使用 PackageExporter 导出整个 package_a 模块到压缩包
        with PackageExporter(filename) as e:
            e.save_module("package_a")

        # 读取创建的压缩包
        zip_file = zipfile.ZipFile(filename, "r")

        # 创建临时目录并解压缩压缩包到临时目录
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 使用 PackageImporter 加载解压后的压缩包
            dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            # 导入并获取 package_a 模块对象
            dir_mod = dir_importer.import_module("package_a")
            # 验证加载的模块中的 result 属性与原始 package_a 模块中的 result 是否相等
            self.assertEqual(dir_mod.result, package_a.result)
    def test_loading_has_record(self):
        """
        Test DirectoryReader's has_record().
        """
        # 导入 package_a 模块，禁止检查未使用的导入（ noqa: F401）
        import package_a  # noqa: F401
        
        # 获取临时文件名
        filename = self.temp()
        
        # 使用 PackageExporter 创建压缩文件
        with PackageExporter(filename) as e:
            e.save_module("package_a")

        # 打开压缩文件为 ZipFile 对象
        zip_file = zipfile.ZipFile(filename, "r")

        # 创建临时目录
        with TemporaryDirectory() as temp_dir:
            # 解压缩到临时目录
            zip_file.extractall(path=temp_dir)
            
            # 创建 PackageImporter 对象，导入解压后的目录
            dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            
            # 断言判断是否存在指定路径的记录
            self.assertTrue(dir_importer.zip_reader.has_record("package_a/__init__.py"))
            self.assertFalse(dir_importer.zip_reader.has_record("package_a"))

    # 如果 Python 版本低于 3.7，则跳过测试（ResourceReader API 在 Python 3.7 引入）
    @skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
    def test_resource_reader(self):
        """Tests DirectoryReader as the base for get_resource_reader."""
        # 创建临时文件名
        filename = self.temp()
        # 使用 PackageExporter 创建压缩包，写入文件内容
        with PackageExporter(filename) as pe:
            # 压缩包结构如下:
            #    package
            #    |-- one/
            #    |   |-- a.txt
            #    |   |-- b.txt
            #    |   |-- c.txt
            #    |   +-- three/
            #    |       |-- d.txt
            #    |       +-- e.txt
            #    +-- two/
            #       |-- f.txt
            #       +-- g.txt
            # 保存文本到指定路径
            pe.save_text("one", "a.txt", "hello, a!")
            pe.save_text("one", "b.txt", "hello, b!")
            pe.save_text("one", "c.txt", "hello, c!")

            pe.save_text("one.three", "d.txt", "hello, d!")
            pe.save_text("one.three", "e.txt", "hello, e!")

            pe.save_text("two", "f.txt", "hello, f!")
            pe.save_text("two", "g.txt", "hello, g!")

        # 打开压缩文件
        zip_file = zipfile.ZipFile(filename, "r")

        # 使用临时目录解压缩文件
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 创建 PackageImporter 实例
            importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            # 获取资源读取器，指定路径为 "one"
            reader_one = importer.get_resource_reader("one")

            # 根据路径构建资源路径
            resource_path = os.path.join(
                Path(temp_dir), Path(filename).name, "one", "a.txt"
            )
            # 断言资源路径是否正确
            self.assertEqual(reader_one.resource_path("a.txt"), resource_path)

            # 检查是否存在资源 "a.txt"
            self.assertTrue(reader_one.is_resource("a.txt"))
            # 断言读取 "a.txt" 的内容是否正确
            self.assertEqual(
                reader_one.open_resource("a.txt").getbuffer(), b"hello, a!"
            )
            # 检查资源 "three" 是否存在
            self.assertFalse(reader_one.is_resource("three"))
            # 获取并排序 "one" 资源的内容列表
            reader_one_contents = list(reader_one.contents())
            reader_one_contents.sort()
            # 断言内容列表是否正确
            self.assertSequenceEqual(
                reader_one_contents, ["a.txt", "b.txt", "c.txt", "three"]
            )

            # 获取资源读取器，指定路径为 "two"
            reader_two = importer.get_resource_reader("two")
            # 检查是否存在资源 "f.txt"
            self.assertTrue(reader_two.is_resource("f.txt"))
            # 断言读取 "f.txt" 的内容是否正确
            self.assertEqual(
                reader_two.open_resource("f.txt").getbuffer(), b"hello, f!"
            )
            # 获取并排序 "two" 资源的内容列表
            reader_two_contents = list(reader_two.contents())
            reader_two_contents.sort()
            # 断言内容列表是否正确
            self.assertSequenceEqual(reader_two_contents, ["f.txt", "g.txt"])

            # 获取资源读取器，指定路径为 "one.three"
            reader_one_three = importer.get_resource_reader("one.three")
            # 检查是否存在资源 "d.txt"
            self.assertTrue(reader_one_three.is_resource("d.txt"))
            # 断言读取 "d.txt" 的内容是否正确
            self.assertEqual(
                reader_one_three.open_resource("d.txt").getbuffer(), b"hello, d!"
            )
            # 获取并排序 "one.three" 资源的内容列表
            reader_one_three_contents = list(reader_one_three.contents())
            reader_one_three_contents.sort()
            # 断言内容列表是否正确
            self.assertSequenceEqual(reader_one_three_contents, ["d.txt", "e.txt"])

            # 检查获取不存在的资源读取器返回 None
            self.assertIsNone(importer.get_resource_reader("nonexistent_package"))
    @skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
    def test_package_resource_access(self):
        """Packaged modules should be able to use the importlib.resources API to access
        resources saved in the package.
        """
        # 准备测试用的模块源代码
        mod_src = dedent(
            """\
            import importlib.resources
            import my_cool_resources

            def secret_message():
                return importlib.resources.read_text(my_cool_resources, 'sekrit.txt')
            """
        )
        # 创建临时文件名
        filename = self.temp()
        # 使用PackageExporter创建一个ZIP文件，并将模块源代码保存到其中
        with PackageExporter(filename) as pe:
            pe.save_source_string("foo.bar", mod_src)
            # 向ZIP文件中保存文本资源
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")

        # 打开创建的ZIP文件
        zip_file = zipfile.ZipFile(filename, "r")

        # 使用临时目录进行文件解压
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 创建PackageImporter对象以导入解压后的模块
            dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            # 断言导入的模块能正确访问并返回期望的文本内容
            self.assertEqual(
                dir_importer.import_module("foo.bar").secret_message(),
                "my sekrit plays",
            )

    @skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
    def test_importer_access(self):
        # 创建临时文件名
        filename = self.temp()
        # 使用PackageExporter创建一个ZIP文件，并向其中保存文本和二进制资源
        with PackageExporter(filename) as he:
            he.save_text("main", "main", "my string")
            he.save_binary("main", "main_binary", b"my string")
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                t = resources.load_text('main', 'main')
                b = resources.load_binary('main', 'main_binary')
                """
            )
            # 向ZIP文件中保存源代码字符串，并指定其为包的一部分
            he.save_source_string("main", src, is_package=True)

        # 打开创建的ZIP文件
        zip_file = zipfile.ZipFile(filename, "r")

        # 使用临时目录进行文件解压
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 创建PackageImporter对象以导入解压后的模块
            dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            m = dir_importer.import_module("main")
            # 断言导入的模块能正确访问并返回期望的文本和二进制内容
            self.assertEqual(m.t, "my string")
            self.assertEqual(m.b, b"my string")
    def test_resource_access_by_path(self):
        """
        Tests that packaged code can used importlib.resources.path.
        """
        # 创建临时文件名
        filename = self.temp()
        # 使用 PackageExporter 创建一个包，并保存一个二进制文件
        with PackageExporter(filename) as e:
            e.save_binary("string_module", "my_string", b"my string")
            # 构建源码字符串
            src = dedent(
                """\
                import importlib.resources
                import string_module

                # 使用 importlib.resources.path 获取资源路径
                with importlib.resources.path(string_module, 'my_string') as path:
                    # 打开路径对应的文件，读取内容到变量 s
                    with open(path, mode='r', encoding='utf-8') as f:
                        s = f.read()
                """
            )
            # 将源码字符串保存为 'main' 模块
            e.save_source_string("main", src, is_package=True)

        # 打开创建的 ZIP 文件
        zip_file = zipfile.ZipFile(filename, "r")

        # 创建临时目录，解压 ZIP 文件到该目录
        with TemporaryDirectory() as temp_dir:
            zip_file.extractall(path=temp_dir)
            # 创建 PackageImporter 对象，导入 'main' 模块
            dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
            m = dir_importer.import_module("main")
            # 断言 'main' 模块中的变量 s 是否等于 "my string"
            self.assertEqual(m.s, "my string")

    def test_scriptobject_failure_message(self):
        """
        Test basic saving and loading of a ScriptModule in a directory.
        Currently not supported.
        """
        # 导入模块
        from package_a.test_module import ModWithTensor

        # 对模块进行脚本化
        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        # 创建临时文件名
        filename = self.temp()
        # 使用 PackageExporter 创建一个包，并保存一个 Pickle 文件
        with PackageExporter(filename) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        # 打开创建的 ZIP 文件
        zip_file = zipfile.ZipFile(filename, "r")

        # 断言在加载脚本模块时会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError,
            "Loading ScriptObjects from a PackageImporter created from a "
            "directory is not supported. Use a package archive file instead.",
        ):
            # 创建临时目录，解压 ZIP 文件到该目录
            with TemporaryDirectory() as temp_dir:
                zip_file.extractall(path=temp_dir)
                # 创建 PackageImporter 对象，加载 Pickle 文件
                dir_importer = PackageImporter(Path(temp_dir) / Path(filename).name)
                dir_mod = dir_importer.load_pickle("res", "mod.pkl")
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```