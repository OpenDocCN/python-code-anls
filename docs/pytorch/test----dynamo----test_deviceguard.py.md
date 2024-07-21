# `.\pytorch\test\dynamo\test_deviceguard.py`

```
# 导入单元测试模块
import unittest
# 导入 Mock 类，用于创建模拟对象
from unittest.mock import Mock

# 导入 PyTorch 库
import torch

# 导入需要测试的模块和类
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import CudaInterface, DeviceGuard
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU

# 定义测试类 TestDeviceGuard，继承自 torch._dynamo.test_case.TestCase
class TestDeviceGuard(torch._dynamo.test_case.TestCase):
    """
    使用模拟的 DeviceInterface 进行 DeviceGuard 类的单元测试。
    """

    # 测试前的准备工作
    def setUp(self):
        super().setUp()
        # 创建 Mock 对象作为设备接口
        self.device_interface = Mock()

        # 设置 Mock 对象的方法返回值
        self.device_interface.exchange_device = Mock(return_value=0)
        self.device_interface.maybe_exchange_device = Mock(return_value=1)

    # 测试 DeviceGuard 类的基本功能
    def test_device_guard(self):
        # 创建 DeviceGuard 对象，传入 Mock 设备接口和设备索引值
        device_guard = DeviceGuard(self.device_interface, 1)

        # 使用 with 语句块测试 DeviceGuard 对象的上下文管理功能
        with device_guard as _:
            # 断言 exchange_device 方法被调用且参数正确
            self.device_interface.exchange_device.assert_called_once_with(1)
            # 断言设备索引值属性正确设置
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        # 断言 maybe_exchange_device 方法被调用且参数正确
        self.device_interface.maybe_exchange_device.assert_called_once_with(0)
        # 再次断言设备索引值属性不变
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    # 测试 DeviceGuard 类在未指定索引值时的行为
    def test_device_guard_no_index(self):
        # 创建 DeviceGuard 对象，传入 Mock 设备接口和空索引值
        device_guard = DeviceGuard(self.device_interface, None)

        # 使用 with 语句块测试 DeviceGuard 对象的上下文管理功能
        with device_guard as _:
            # 断言 exchange_device 方法未被调用
            self.device_interface.exchange_device.assert_not_called()
            # 断言设备索引值属性正确设置为预设值
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        # 断言 maybe_exchange_device 方法未被调用
        self.device_interface.maybe_exchange_device.assert_not_called()
        # 再次断言设备索引值属性不变
        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


# 如果没有 CUDA 环境，则跳过 CUDA 相关的测试
@unittest.skipIf(not TEST_CUDA, "No CUDA available.")
class TestCUDADeviceGuard(torch._dynamo.test_case.TestCase):
    """
    使用 CudaInterface 进行 DeviceGuard 类的单元测试。
    """

    # 测试前的准备工作
    def setUp(self):
        super().setUp()
        # 创建 CudaInterface 对象作为设备接口
        self.device_interface = CudaInterface

    # 如果没有多 GPU 环境，则跳过多 GPU 相关的测试
    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_device_guard(self):
        # 获取当前 CUDA 设备索引
        current_device = torch.cuda.current_device()

        # 创建 DeviceGuard 对象，传入 CudaInterface 设备接口和设备索引值
        device_guard = DeviceGuard(self.device_interface, 1)

        # 使用 with 语句块测试 DeviceGuard 对象的上下文管理功能
        with device_guard as _:
            # 断言当前 CUDA 设备索引被正确设置
            self.assertEqual(torch.cuda.current_device(), 1)
            # 断言设备索引值属性正确设置
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        # 断言 CUDA 设备索引恢复到原来的值
        self.assertEqual(torch.cuda.current_device(), current_device)
        # 再次断言设备索引值属性不变
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    # 测试 DeviceGuard 类在未指定索引值时的行为
    def test_device_guard_no_index(self):
        # 获取当前 CUDA 设备索引
        current_device = torch.cuda.current_device()

        # 创建 DeviceGuard 对象，传入 CudaInterface 设备接口和空索引值
        device_guard = DeviceGuard(self.device_interface, None)

        # 使用 with 语句块测试 DeviceGuard 对象的上下文管理功能
        with device_guard as _:
            # 断言当前 CUDA 设备索引未改变
            self.assertEqual(torch.cuda.current_device(), current_device)
            # 断言设备索引值属性正确设置为预设值
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        # 再次断言设备索引值属性不变
        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)
#`
# 检查当前脚本是否为主程序入口
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 调用 run_tests 函数执行测试
    run_tests()
```