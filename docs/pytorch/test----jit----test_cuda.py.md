# `.\pytorch\test\jit\test_cuda.py`

```
# Owner(s): ["oncall: jit"]

# 引入标准库和第三方库
import gc
import os
import sys
import unittest
from typing import NamedTuple

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    NoTest,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    TEST_CUDA,
)
from torch.testing._internal.jit_utils import JitTestCase

# 将测试目录下的辅助文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果 CUDA 不可用，则跳过测试并输出消息到标准错误流
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    JitTestCase = NoTest  # noqa: F811

# 设置一个标志来指示是否测试大型张量
TEST_LARGE_TENSOR = TEST_CUDA

# 如果 CUDA 可用，则初始化 CUDA 上下文并检查是否有足够的内存分配大型张量
if TEST_CUDA:
    torch.ones(1).cuda()  # 初始化 CUDA 上下文
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 5e9

# 如果脚本直接运行，则引发 RuntimeError 并给出建议的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类 TestCUDA，继承自 JitTestCase
class TestCUDA(JitTestCase):
    """
    A suite of tests for the CUDA API in TorchScript.
    """

    # 在每个测试方法执行后，进行内存回收和 CUDA 缓存清空
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        super().tearDown()

    # 标记为跳过测试，如果是 ROCm 环境
    @skipIfRocm
    # 标记为跳过测试，如果只检测到一个 GPU
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_cuda_synchronize(self):
        # Test device synchronization.

        @torch.jit.script
        def test_device_synchronize():
            # 获取当前 CUDA 设备的索引
            prev_current_device_index = torch.cuda.current_device()
            # 同步当前 CUDA 设备
            torch.cuda.synchronize()
            # 同步指定名称的 CUDA 设备
            torch.cuda.synchronize("cuda")
            # 同步指定索引的 CUDA 设备
            torch.cuda.synchronize("cuda:0")
            # 同步指定索引的 CUDA 设备
            torch.cuda.synchronize(0)
            # 同步指定设备对象表示的 CUDA 设备
            torch.cuda.synchronize(torch.device("cuda:1"))
            # 获取同步后的当前 CUDA 设备索引
            after_current_device_index = torch.cuda.current_device()

            # 检查同步前后的 CUDA 设备索引是否相同
            return prev_current_device_index == after_current_device_index

        @torch.jit.script
        def test_multi_device_synchronize():
            # 同步指定设备对象表示的 CUDA 设备
            torch.cuda.synchronize(torch.device("cuda:0"))
            # 获取当前 CUDA 设备的索引
            prev_current_device_index = torch.cuda.current_device()
            # 同步指定索引的 CUDA 设备
            torch.cuda.synchronize(1)
            # 获取同步后的当前 CUDA 设备索引
            after_current_device_index = torch.cuda.current_device()

            # 检查同步前后的 CUDA 设备索引是否相同
            return prev_current_device_index == after_current_device_index

        # 断言测试设备同步函数返回真值
        self.assertTrue(test_device_synchronize)
        # 使用 FileCheck 检查 JIT 编译后的图形中是否包含 CUDA 同步操作
        FileCheck().check("cuda::synchronize(").run(test_device_synchronize.graph)
        # 断言测试多设备同步函数返回真值
        self.assertTrue(test_multi_device_synchronize)
        # 使用 FileCheck 检查 JIT 编译后的图形中是否包含 CUDA 同步操作
        FileCheck().check("cuda::synchronize(").run(test_multi_device_synchronize.graph)

    def test_stream_args(self):
        # Test stream creation with default arguments
        @torch.jit.script
        def stream_default_args() -> bool:
            # 创建默认参数的 CUDA 流对象
            s = torch.cuda.Stream()
            # 检查 CUDA 流对象的设备索引是否与当前设备索引相同
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_device() -> bool:
            # 使用指定设备优先级创建 CUDA 流对象
            s = torch.cuda.Stream(priority=0)
            # 检查 CUDA 流对象的设备索引是否与当前设备索引相同
            return s.device_index() == torch.cuda.current_device()

        @torch.jit.script
        def stream_default_args_for_priority() -> bool:
            # 创建指定设备对象表示的 CUDA 流对象
            d = torch.device("cuda:1")
            s = torch.cuda.Stream(d)
            # 检查 CUDA 流对象的设备索引是否为指定设备的索引
            return s.device_index() == 1

        @torch.jit.script
        def stream_args_all() -> bool:
            # 创建指定设备对象表示的 CUDA 流对象，并设置优先级
            d = torch.device("cuda:0")
            s = torch.cuda.Stream(d, 0)
            # 检查 CUDA 流对象的设备索引是否为指定设备的索引
            return s.device_index() == 0

        # 断言各测试函数返回真值
        self.assertTrue(stream_default_args)
        self.assertTrue(stream_default_args_for_device)
        self.assertTrue(stream_default_args_for_priority)
        self.assertTrue(stream_args_all)

    def test_event_args(self):
        # Test Event creation with default arguments
        @torch.jit.script
        def event_default_args() -> bool:
            # 创建默认参数的 CUDA 事件对象
            e = torch.cuda.Event()
            # 检查 CUDA 事件对象是否被成功创建
            return e is not None

        # 断言测试函数返回真值
        self.assertTrue(event_default_args)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_current_stream(self):
        # Test current stream on the device and check if the stream device index
        # matches with the device ID
        @torch.jit.script
        def fn():
            # 获取当前 CUDA 设备的索引
            device_index = torch.cuda.current_device()
            # 根据设备索引创建 CUDA 设备对象
            device = torch.device("cuda:" + str(device_index))
            # 获取当前 CUDA 设备的当前流
            s0 = torch.cuda.current_stream(device)
            # 获取 CUDA 设备 cuda:1 的当前流
            s1 = torch.cuda.current_stream(torch.device("cuda:1"))
            # 获取 CUDA 设备 cuda:0 的当前流
            s2 = torch.cuda.current_stream(torch.device("cuda:0"))

            return s0.device_index(), s1.device_index(), s2.device_index()

        # 调用 fn() 函数获取各设备的流索引
        d0, d1, d2 = fn()
        # 默认情况下，当前设备 ID 是 0
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

        # Test current_stream API by passing device ID as an argument and
        # and check if the stream device index matches with the device ID
        @torch.jit.script
        def fn_with_device_index_args():
            # 获取当前 CUDA 设备的索引
            device_index = torch.cuda.current_device()
            # 获取当前 CUDA 设备的当前流
            s0 = torch.cuda.current_stream(device_index)
            # 获取 CUDA 设备 1 的当前流
            s1 = torch.cuda.current_stream(1)
            # 获取 CUDA 设备 0 的当前流
            s2 = torch.cuda.current_stream(0)

            return s0.device_index(), s1.device_index(), s2.device_index()

        # 调用 fn_with_device_index_args() 函数获取各设备的流索引
        d0, d1, d2 = fn_with_device_index_args()
        # 默认情况下，当前设备 ID 是 0
        self.assertEqual(0, d0)
        self.assertEqual(1, d1)
        self.assertEqual(0, d2)
        self.assertEqual(d0, d2)

    @skipIfRocm
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @skipCUDANonDefaultStreamIf(True)
    # Make sure that cuda._exchange_device doesn't get DCE'ed
    @unittest.skipIf(not TEST_CUDA, "Cuda not available")
    def test__exchange_device_op(self):
        # 定义一个函数 fn，接受设备索引和张量作为参数
        def fn(device: int, tensor):
            # 调用 CUDA API 将张量移动到指定设备
            torch.cuda._exchange_device(device)
            # 对张量执行一系列操作：cosine 和 ReLU
            return tensor.cos().relu()

        # 对 fn 函数进行 TorchScript 编译
        fn_s = torch.jit.script(fn)
        # 只检查图的结构，而不运行它。否则，我们需要在多 GPU 的 CI 运行器上运行这个测试，这是不必要的。
        g = fn_s.graph
        # 使用 FileCheck 检查图中是否存在 cuda::_exchange_device 调用
        FileCheck().check("cuda::_exchange_device(").run(g)
        # 对图执行内联传递优化
        torch._C._jit_pass_inline(g)
        # 再次使用 FileCheck 检查图中是否存在 cuda::_exchange_device 调用
        FileCheck().check("cuda::_exchange_device(").run(g)

    # Make sure that cuda._maybe_exchange_device doesn't get DCE'ed
    @unittest.skipIf(not TEST_CUDA, "Cuda not available")
    def test__maybe_exchange_device_op(self):
        # 定义一个内部函数 fn，接受一个整数 device 和一个张量 tensor 作为参数
        def fn(device: int, tensor):
            # 调用 torch.cuda._maybe_exchange_device(device) 函数，可能交换设备
            torch.cuda._maybe_exchange_device(device)
            # 对张量 tensor 执行操作：先计算余弦，再应用整流函数
            return tensor.cos().relu()

        # 使用 torch.jit.script 对函数 fn 进行脚本化
        fn_s = torch.jit.script(fn)
        # 只检查计算图，不运行它。否则，我们需要在一个多 GPU 的 CI 运行器上运行这个测试，这是过度的。
        g = fn_s.graph
        # 使用 FileCheck().check("cuda::_maybe_exchange_device(").run(g) 检查计算图中是否包含指定字符串
        FileCheck().check("cuda::_maybe_exchange_device(").run(g)
        # 对计算图 g 执行内联优化
        torch._C._jit_pass_inline(g)
        # 再次使用 FileCheck().check("cuda::_maybe_exchange_device(").run(g) 检查优化后的计算图中是否包含指定字符串
        FileCheck().check("cuda::_maybe_exchange_device(").run(g)
```