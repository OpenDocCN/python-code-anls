# `.\pytorch\test\test_cuda_trace.py`

```
# Owner(s): ["module: cuda"]

import sys  # 导入sys模块，用于访问系统相关功能
import unittest  # 导入unittest模块，用于编写和运行单元测试
import unittest.mock  # 导入unittest.mock模块，用于创建模拟对象

import torch  # 导入PyTorch库
import torch.cuda._gpu_trace as gpu_trace  # 导入GPU跟踪模块
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase  # 导入测试相关的工具函数和类

# NOTE: Each test needs to be run in a brand new process, to reset the registered hooks
# and make sure the CUDA streams are initialized for each test that uses them.
# 注意：每个测试需要在全新的进程中运行，以重置已注册的钩子，并确保每个使用CUDA流的测试都初始化了流。

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)  # 如果CUDA不可用，则打印跳过测试的消息到标准错误流
    TestCase = NoTest  # noqa: F811  # 如果没有CUDA测试标志，则将TestCase类替换为NoTest类，防止flake8警告


@torch.testing._internal.common_utils.markDynamoStrictTest  # 使用内部测试工具标记为Dynamo严格测试
class TestCudaTrace(TestCase):
    def setUp(self):
        torch._C._activate_gpu_trace()  # 激活GPU跟踪
        self.mock = unittest.mock.MagicMock()  # 创建一个模拟对象

    def test_event_creation_callback(self):
        gpu_trace.register_callback_for_event_creation(self.mock)  # 注册事件创建的回调函数到模拟对象

        event = torch.cuda.Event()  # 创建CUDA事件对象
        event.record()  # 记录事件
        self.mock.assert_called_once_with(event._as_parameter_.value)  # 断言模拟对象的方法被调用，并传递了事件的值作为参数

    def test_event_deletion_callback(self):
        gpu_trace.register_callback_for_event_deletion(self.mock)  # 注册事件删除的回调函数到模拟对象

        event = torch.cuda.Event()  # 创建CUDA事件对象
        event.record()  # 记录事件
        event_id = event._as_parameter_.value  # 获取事件的值
        del event  # 删除事件对象
        self.mock.assert_called_once_with(event_id)  # 断言模拟对象的方法被调用，并传递了事件的值作为参数

    def test_event_record_callback(self):
        gpu_trace.register_callback_for_event_record(self.mock)  # 注册事件记录的回调函数到模拟对象

        event = torch.cuda.Event()  # 创建CUDA事件对象
        event.record()  # 记录事件
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )  # 断言模拟对象的方法被调用，并传递了事件的值和默认CUDA流作为参数

    def test_event_wait_callback(self):
        gpu_trace.register_callback_for_event_wait(self.mock)  # 注册事件等待的回调函数到模拟对象

        event = torch.cuda.Event()  # 创建CUDA事件对象
        event.record()  # 记录事件
        event.wait()  # 等待事件完成
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )  # 断言模拟对象的方法被调用，并传递了事件的值和默认CUDA流作为参数

    def test_memory_allocation_callback(self):
        gpu_trace.register_callback_for_memory_allocation(self.mock)  # 注册内存分配的回调函数到模拟对象

        tensor = torch.empty(10, 4, device="cuda")  # 在CUDA设备上创建一个空张量
        self.mock.assert_called_once_with(tensor.data_ptr())  # 断言模拟对象的方法被调用，并传递了张量数据指针作为参数

    def test_memory_deallocation_callback(self):
        gpu_trace.register_callback_for_memory_deallocation(self.mock)  # 注册内存释放的回调函数到模拟对象

        tensor = torch.empty(3, 8, device="cuda")  # 在CUDA设备上创建一个空张量
        data_ptr = tensor.data_ptr()  # 获取张量数据指针
        del tensor  # 删除张量对象
        self.mock.assert_called_once_with(data_ptr)  # 断言模拟对象的方法被调用，并传递了张量数据指针作为参数

    def test_stream_creation_callback(self):
        gpu_trace.register_callback_for_stream_creation(self.mock)  # 注册流创建的回调函数到模拟对象

        # see Note [HIP Lazy Streams]
        if torch.version.hip:  # 如果是HIP环境
            user_stream = torch.cuda.Stream()  # 创建CUDA流对象
            with torch.cuda.stream(user_stream):  # 将用户定义的流设置为当前流
                tensor = torch.ones(5, device="cuda")  # 在CUDA设备上创建全为1的张量
        else:
            torch.cuda.Stream()  # 在CUDA环境下创建默认流

        self.mock.assert_called()  # 断言模拟对象的方法被调用

    def test_device_synchronization_callback(self):
        gpu_trace.register_callback_for_device_synchronization(self.mock)  # 注册设备同步的回调函数到模拟对象

        torch.cuda.synchronize()  # 同步CUDA设备
        self.mock.assert_called()  # 断言模拟对象的方法被调用
    # 注册回调函数以便在流同步时进行调用
    def test_stream_synchronization_callback(self):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        # 创建一个 CUDA 流对象
        stream = torch.cuda.Stream()
        # 同步 CUDA 流
        stream.synchronize()
        # 断言回调函数被调用，参数为 CUDA 流的 ID
        self.mock.assert_called_once_with(stream.cuda_stream)

    # 注册回调函数以便在事件同步时进行调用
    def test_event_synchronization_callback(self):
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        # 创建一个 CUDA 事件对象
        event = torch.cuda.Event()
        # 记录 CUDA 事件
        event.record()
        # 同步 CUDA 事件
        event.synchronize()
        # 断言回调函数被调用，参数为 CUDA 事件的值
        self.mock.assert_called_once_with(event._as_parameter_.value)

    # 注册回调函数以便在流同步时进行调用
    def test_memcpy_synchronization(self):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        # 在 CUDA 设备上创建一个张量
        tensor = torch.rand(5, device="cuda")
        # 执行非零元素检测操作
        tensor.nonzero()
        # 断言回调函数被调用，参数为默认 CUDA 流的 ID
        self.mock.assert_called_once_with(torch.cuda.default_stream().cuda_stream)

    # 确保所有追踪回调函数都被调用
    def test_all_trace_callbacks_called(self):
        other = unittest.mock.MagicMock()
        # 注册内存分配追踪回调函数
        gpu_trace.register_callback_for_memory_allocation(self.mock)
        gpu_trace.register_callback_for_memory_allocation(other)

        # 在 CUDA 设备上创建一个空张量
        tensor = torch.empty(10, 4, device="cuda")
        # 断言第一个回调函数被调用，参数为张量的数据指针
        self.mock.assert_called_once_with(tensor.data_ptr())
        # 断言第二个回调函数被调用，参数同样为张量的数据指针
        other.assert_called_once_with(tensor.data_ptr())
# 如果当前脚本作为主程序运行（而非被导入其他模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码或者其他功能
    run_tests()
```