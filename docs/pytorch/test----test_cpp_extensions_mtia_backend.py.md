# `.\pytorch\test\test_cpp_extensions_mtia_backend.py`

```
# Owner(s): ["module: mtia"]

# 导入标准库和第三方库
import os
import shutil
import sys
import tempfile
import unittest

# 导入 PyTorch 相关模块和工具函数
import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_PRIVATEUSE1,
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

# 在更改 TEST_CUDA 之前定义 TEST_ROCM
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
# 更新 TEST_CUDA，确保 CUDA_HOME 存在
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None

# 删除构建路径的函数
def remove_build_path():
    # 如果运行平台是 Windows，不清除扩展构建文件夹
    if sys.platform == "win32":
        return
    # 获取默认的构建根目录
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    # 如果构建根目录存在，则递归删除
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

# 使用装饰器跳过测试条件
@unittest.skipIf(
    IS_ARM64 or not IS_LINUX or TEST_CUDA or TEST_PRIVATEUSE1 or TEST_ROCM or TEST_XPU,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
# 测试类，测试 MTIA 后端与 C++ 扩展的集成
class TestCppExtensionMTIABackend(common.TestCase):
    """Tests MTIA backend with C++ extensions."""

    module = None  # 模块初始化为 None

    def setUp(self):
        super().setUp()
        # 保存旧的工作目录，并临时切换到当前脚本所在目录
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # 恢复到旧的工作目录
        os.chdir(self.old_working_dir)

    @classmethod
    def tearDownClass(cls):
        # 在类销毁前清理构建路径
        remove_build_path()

    @classmethod
    def setUpClass(cls):
        # 在类初始化时清理构建路径，并创建临时的构建目录
        remove_build_path()
        build_dir = tempfile.mkdtemp()
        # 加载伪装的设备保护实现
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",  # 扩展名字
            sources=["cpp_extensions/mtia_extension.cpp"],  # 源文件路径
            build_directory=build_dir,  # 构建目录
            extra_include_paths=[  # 额外的包含路径
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            is_python_module=False,  # 不是 Python 模块
            verbose=True,  # 输出详细信息
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 测试函数，测试获取设备模块
    def test_get_device_module(self):
        device = torch.device("mtia:0")  # 创建 MTIA 设备
        default_stream = torch.get_device_module(device).current_stream()  # 获取当前流
        self.assertEqual(
            default_stream.device_type, int(torch._C._autograd.DeviceType.MTIA)  # 断言设备类型为 MTIA
        )
        print(torch._C.Stream.__mro__)  # 打印 Stream 类的方法解析顺序
        print(torch.cuda.Stream.__mro__)  # 打印 CUDA Stream 类的方法解析顺序

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_basic(self):
        # 获取默认的流对象
        default_stream = torch.mtia.current_stream()
        # 创建一个用户定义的流对象
        user_stream = torch.mtia.Stream()
        # 断言当前流对象是默认流对象
        self.assertEqual(torch.mtia.current_stream(), default_stream)
        # 断言默认流对象和用户流对象不相等
        self.assertNotEqual(default_stream, user_stream)
        # 检查 mtia_extension.cpp，默认流对象的流 ID 从 0 开始
        self.assertEqual(default_stream.stream_id, 0)
        # 断言用户流对象的流 ID 不是 0
        self.assertNotEqual(user_stream.stream_id, 0)
        # 在用户流对象的上下文中执行以下代码块
        with torch.mtia.stream(user_stream):
            # 断言当前流对象是用户流对象
            self.assertEqual(torch.mtia.current_stream(), user_stream)
        # 检查用户流对象是否处于查询状态
        self.assertTrue(user_stream.query())
        # 同步默认流对象
        default_stream.synchronize()
        # 检查默认流对象是否处于查询状态
        self.assertTrue(default_stream.query())

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context(self):
        # 创建两个在不同设备上的流对象
        mtia_stream_0 = torch.mtia.Stream(device="mtia:0")
        mtia_stream_1 = torch.mtia.Stream(device="mtia:0")
        # 打印第一个流对象
        print(mtia_stream_0)
        # 打印第二个流对象
        print(mtia_stream_1)
        # 在 mtia_stream_0 上下文中执行以下代码块
        with torch.mtia.stream(mtia_stream_0):
            # 获取当前流对象
            current_stream = torch.mtia.current_stream()
            # 断言当前流对象应该是 mtia_stream_0
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)

        # 在 mtia_stream_1 上下文中执行以下代码块
        with torch.mtia.stream(mtia_stream_1):
            # 获取当前流对象
            current_stream = torch.mtia.current_stream()
            # 断言当前流对象应该是 mtia_stream_1
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context_different_device(self):
        # 定义两个不同设备上的设备对象
        device_0 = torch.device("mtia:0")
        device_1 = torch.device("mtia:1")
        # 创建两个在不同设备上的流对象
        mtia_stream_0 = torch.mtia.Stream(device=device_0)
        mtia_stream_1 = torch.mtia.Stream(device=device_1)
        # 打印第一个流对象
        print(mtia_stream_0)
        # 打印第二个流对象
        print(mtia_stream_1)
        # 获取原始的当前设备对象
        orig_current_device = torch.mtia.current_device()
        # 在 mtia_stream_0 上下文中执行以下代码块
        with torch.mtia.stream(mtia_stream_0):
            # 获取当前流对象
            current_stream = torch.mtia.current_stream()
            # 断言当前设备对象是 device_0 的索引
            self.assertTrue(torch.mtia.current_device() == device_0.index)
            # 断言当前流对象应该是 mtia_stream_0
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)
        # 恢复原始的当前设备对象
        self.assertTrue(torch.mtia.current_device() == orig_current_device)
        # 在 mtia_stream_1 上下文中执行以下代码块
        with torch.mtia.stream(mtia_stream_1):
            # 获取当前流对象
            current_stream = torch.mtia.current_stream()
            # 断言当前设备对象是 device_1 的索引
            self.assertTrue(torch.mtia.current_device() == device_1.index)
            # 断言当前流对象应该是 mtia_stream_1
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)
        # 恢复原始的当前设备对象
        self.assertTrue(torch.mtia.current_device() == orig_current_device)
    # 定义一个测试方法，用于测试设备上下文管理器的功能
    def test_device_context(self):
        # 创建一个名为 "mtia:0" 的设备对象
        device_0 = torch.device("mtia:0")
        # 创建一个名为 "mtia:1" 的设备对象
        device_1 = torch.device("mtia:1")
        
        # 使用设备上下文管理器，将当前设备设置为 device_0
        with torch.mtia.device(device_0):
            # 断言当前设备是否为 device_0 的索引
            self.assertTrue(torch.mtia.current_device() == device_0.index)
        
        # 使用设备上下文管理器，将当前设备设置为 device_1
        with torch.mtia.device(device_1):
            # 断言当前设备是否为 device_1 的索引
            self.assertTrue(torch.mtia.current_device() == device_1.index)
# 如果这个脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用 common 模块中的 run_tests 函数，用于运行测试套件
    common.run_tests()
```