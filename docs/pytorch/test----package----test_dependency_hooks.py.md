# `.\pytorch\test\package\test_dependency_hooks.py`

```py
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO  # 导入 BytesIO 类，用于创建内存中的二进制数据流

from torch.package import PackageExporter  # 导入 PackageExporter 类，用于导出打包内容
from torch.testing._internal.common_utils import run_tests  # 导入测试工具函数 run_tests

try:
    from .common import PackageTestCase  # 尝试从当前目录导入 PackageTestCase 类
except ImportError:
    # 如果导入失败，支持直接运行该文件的情况
    from common import PackageTestCase  # 从 common 模块导入 PackageTestCase 类


class TestDependencyHooks(PackageTestCase):
    """Dependency management hooks API tests.
    - register_mock_hook()
    - register_extern_hook()
    """

    def test_single_hook(self):
        buffer = BytesIO()  # 创建一个空的 BytesIO 对象，用于存储二进制数据

        my_externs = set()  # 创建一个空的集合，用于存储外部依赖模块名

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)  # 添加模块名到外部依赖集合中

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])  # 导出指定的子包和模块
            exporter.register_extern_hook(my_extern_hook)  # 注册外部依赖钩子函数
            exporter.save_source_string("foo", "import module_a")  # 将源代码字符串保存到打包内容中

        self.assertEqual(my_externs, {"module_a"})  # 断言外部依赖集合是否包含正确的模块名

    def test_multiple_extern_hooks(self):
        buffer = BytesIO()  # 创建一个空的 BytesIO 对象，用于存储二进制数据

        my_externs = set()  # 创建一个空的集合，用于存储外部依赖模块名

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)  # 添加模块名到外部依赖集合中

        # 这里还检查了顺序，因为如果值不在集合中，`remove()` 将失败。
        def my_extern_hook2(package_exporter, module_name):
            my_externs.remove(module_name)  # 从外部依赖集合中移除模块名

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])  # 导出指定的子包和模块
            exporter.register_extern_hook(my_extern_hook)  # 注册第一个外部依赖钩子函数
            exporter.register_extern_hook(my_extern_hook2)  # 注册第二个外部依赖钩子函数
            exporter.save_source_string("foo", "import module_a")  # 将源代码字符串保存到打包内容中

        self.assertEqual(my_externs, set())  # 断言外部依赖集合是否为空集合

    def test_multiple_mock_hooks(self):
        buffer = BytesIO()  # 创建一个空的 BytesIO 对象，用于存储二进制数据

        my_mocks = set()  # 创建一个空的集合，用于存储模拟模块名

        def my_mock_hook(package_exporter, module_name):
            my_mocks.add(module_name)  # 添加模块名到模拟集合中

        # 这里还检查了顺序，因为如果值不在集合中，`remove()` 将失败。
        def my_mock_hook2(package_exporter, module_name):
            my_mocks.remove(module_name)  # 从模拟集合中移除模块名

        with PackageExporter(buffer) as exporter:
            exporter.mock(["package_a.subpackage", "module_a"])  # 模拟指定的子包和模块
            exporter.register_mock_hook(my_mock_hook)  # 注册第一个模拟钩子函数
            exporter.register_mock_hook(my_mock_hook2)  # 注册第二个模拟钩子函数
            exporter.save_source_string("foo", "import module_a")  # 将源代码字符串保存到打包内容中

        self.assertEqual(my_mocks, set())  # 断言模拟集合是否为空集合
    # 定义一个测试方法，用于测试移除钩子的功能
    def test_remove_hooks(self):
        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 定义两个空的集合，用于存储外部模块和第二个外部模块的名称
        my_externs = set()
        my_externs2 = set()

        # 定义一个函数，用于将模块名称添加到 my_externs 集合中
        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        # 定义另一个函数，用于将模块名称添加到 my_externs2 集合中
        def my_extern_hook2(package_exporter, module_name):
            my_externs2.add(module_name)

        # 使用 PackageExporter 类和字节流缓冲区创建一个上下文管理器
        with PackageExporter(buffer) as exporter:
            # 声明要导出的外部模块列表
            exporter.extern(["package_a.subpackage", "module_a"])
            # 注册第一个外部钩子函数，并记录其句柄
            handle = exporter.register_extern_hook(my_extern_hook)
            # 注册第二个外部钩子函数
            exporter.register_extern_hook(my_extern_hook2)
            # 移除第一个外部钩子函数的注册
            handle.remove()
            # 保存字符串源代码到 exporter 中，该源代码包含了导入 module_a 的操作
            exporter.save_source_string("foo", "import module_a")

        # 断言第一个外部模块集合为空集
        self.assertEqual(my_externs, set())
        # 断言第二个外部模块集合包含 "module_a" 这个模块名
        self.assertEqual(my_externs2, {"module_a"})

    # 定义一个测试方法，用于测试 extern 和 mock 钩子的功能
    def test_extern_and_mock_hook(self):
        # 创建一个字节流缓冲区
        buffer = BytesIO()

        # 定义两个空的集合，用于存储外部模块和模拟模块的名称
        my_externs = set()
        my_mocks = set()

        # 定义一个函数，用于将外部模块名称添加到 my_externs 集合中
        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        # 定义一个函数，用于将模拟模块名称添加到 my_mocks 集合中
        def my_mock_hook(package_exporter, module_name):
            my_mocks.add(module_name)

        # 使用 PackageExporter 类和字节流缓冲区创建一个上下文管理器
        with PackageExporter(buffer) as exporter:
            # 声明要导出的外部模块 "module_a"
            exporter.extern("module_a")
            # 声明要模拟的包 "package_a"
            exporter.mock("package_a")
            # 注册外部钩子函数 my_extern_hook
            exporter.register_extern_hook(my_extern_hook)
            # 注册模拟钩子函数 my_mock_hook
            exporter.register_mock_hook(my_mock_hook)
            # 保存字符串源代码到 exporter 中，该源代码包含了导入 module_a 和 package_a 的操作
            exporter.save_source_string("foo", "import module_a; import package_a")

        # 断言外部模块集合包含 "module_a" 这个模块名
        self.assertEqual(my_externs, {"module_a"})
        # 断言模拟模块集合包含 "package_a" 这个包名
        self.assertEqual(my_mocks, {"package_a"})
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```