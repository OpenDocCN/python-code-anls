# `.\pytorch\test\test_cpp_extensions_stream_and_event.py`

```
# Owner(s): ["module: mtia"]

# 导入必要的库和模块
import os
import shutil
import sys
import tempfile
import unittest

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_PRIVATEUSE1,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

# 在修改 TEST_CUDA 之前定义 TEST_ROCM
# 如果 TEST_CUDA 为 True，且 torch 版本支持 ROCm 平台，并且 ROCM_HOME 存在，则定义 TEST_ROCM 为 True
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None

# 如果 CUDA_HOME 存在，则定义 TEST_CUDA 为 True
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None


# 删除构建路径的函数
def remove_build_path():
    # 如果系统平台是 win32，则不清除扩展构建文件夹
    if sys.platform == "win32":
        return
    # 获取默认的构建根目录
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    # 如果默认构建根目录存在，则递归删除它
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)


# 由于我们使用虚假的 MTIA 设备后端来测试通用的 Stream/Event，设备后端彼此互斥。
# 如果满足以下任一条件，则跳过测试：
# - IS_ARM64 为 True
# - 不是 Linux 平台 (not IS_LINUX)
# - TEST_CUDA 为 True
# - TEST_PRIVATEUSE1 为 True
# - TEST_ROCM 为 True
@unittest.skipIf(
    IS_ARM64 or not IS_LINUX or TEST_CUDA or TEST_PRIVATEUSE1 or TEST_ROCM,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionStreamAndEvent(common.TestCase):
    """Tests Stream and Event with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()
        # cpp 扩展使用相对路径。这些路径是相对于此文件的，因此我们将临时更改工作目录
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # 恢复工作目录至测试之前的状态
        os.chdir(self.old_working_dir)

    @classmethod
    def tearDownClass(cls):
        # 删除构建路径
        remove_build_path()

    @classmethod
    def setUpClass(cls):
        # 删除构建路径
        remove_build_path()
        # 创建临时的构建目录
        build_dir = tempfile.mkdtemp()
        # 加载虚假设备保护实现
        src = f"{os.path.abspath(os.path.dirname(__file__))}/cpp_extensions/mtia_extension.cpp"
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",
            sources=[src],
            build_directory=build_dir,
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            is_python_module=False,
            verbose=True,
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 定义测试方法，用于测试流事件操作
    def test_stream_event(self):
        # 创建一个 torch Stream 对象 s
        s = torch.Stream()
        # 断言 s 的设备类型为 MTIA
        self.assertTrue(s.device_type, int(torch._C._autograd.DeviceType.MTIA))
        # 创建一个 torch Event 对象 e
        e = torch.Event()
        # 断言 e 的设备类型为 "mtia"
        self.assertTrue(e.device.type, "mtia")
        # 默认情况下应为 nullptr
        self.assertTrue(e.event_id == 0)
        # 将事件 e 记录到流 s 中
        s.record_event(event=e)
        # 打印记录的事件 e 的信息
        print(f"recorded event 1: {e}")
        # 确保事件 e 的 event_id 不为 0
        self.assertTrue(e.event_id != 0)
        # 在流 s 中记录另一个事件，返回事件对象 e2
        e2 = s.record_event()
        # 打印记录的事件 e2 的信息
        print(f"recorded event 2: {e2}")
        # 确保事件 e2 的 event_id 不为 0
        self.assertTrue(e2.event_id != 0)
        # 确保事件 e2 的 event_id 不等于事件 e 的 event_id
        self.assertTrue(e2.event_id != e.event_id)
        # 同步事件 e
        e.synchronize()
        # 同步事件 e2
        e2.synchronize()
        # 计算事件 e 和事件 e2 之间的时间间隔
        time_elapsed = e.elapsed_time(e2)
        # 打印时间间隔信息
        print(f"time elapsed between e1 and e2: {time_elapsed}")
        # 保存旧的事件 e 的 event_id
        old_event_id = e.event_id
        # 在流 s 中记录事件 e
        e.record(stream=s)
        # 打印记录的事件 e 的信息
        print(f"recorded event 1: {e}")
        # 确保事件 e 的 event_id 等于之前保存的旧 event_id
        self.assertTrue(e.event_id == old_event_id)
# 如果这个脚本被直接执行（而不是被导入为模块），则执行下面的代码
if __name__ == "__main__":
    # 调用 common 模块中的 run_tests 函数，用于执行测试用例
    common.run_tests()
```