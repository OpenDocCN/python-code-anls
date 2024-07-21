# `.\pytorch\test\test_cuda_multigpu.py`

```py
# Owner(s): ["module: cuda"]

# 导入标准库和第三方库
import collections  # 导入 collections 库
import contextlib  # 导入 contextlib 库，用于创建和管理上下文管理器
import ctypes  # 导入 ctypes 库，用于处理 C 数据类型
import gc  # 导入 gc (垃圾回收) 库
import io  # 导入 io 库，用于处理流操作
import queue  # 导入 queue 库，用于实现队列数据结构
import sys  # 导入 sys 库，提供对 Python 解释器的访问
import tempfile  # 导入 tempfile 库，用于创建临时文件和目录
import threading  # 导入 threading 库，支持多线程编程
import unittest  # 导入 unittest 库，用于编写和运行单元测试

# 导入指定的模块和函数
from itertools import chain, repeat  # 导入 itertools 库中的 chain 和 repeat 函数
from typing import NamedTuple  # 从 typing 模块中导入 NamedTuple 类型

import torch  # 导入 PyTorch 深度学习库
import torch.cuda.comm as comm  # 导入 torch.cuda.comm 模块
from torch.nn.parallel import scatter_gather  # 导入 torch.nn.parallel 模块中的 scatter_gather 函数
from torch.testing._internal.common_cuda import (  # 导入内部测试所需的 CUDA 相关函数和常量
    _create_scaling_case,
    _create_scaling_models_optimizers,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (  # 导入内部测试所需的常用函数和测试类
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    IS_JETSON,
    IS_REMOTE_GPU,
    IS_SANDCASTLE,
    NoTest,
    run_tests,
    serialTest,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    TEST_CUDA,
    TestCase,
)

# 检查是否支持 CUDA，如果不支持则输出错误信息并将 TestCase 设为 NoTest 类
TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 定义 TestCudaMultiGPU 类，继承自 TestCase 类
class TestCudaMultiGPU(TestCase):
    FIFTY_MIL_CYCLES = 50000000  # 定义常量 FIFTY_MIL_CYCLES，表示 50000000 循环次数
    # 检查内存状态的一致性，获取当前 CUDA 设备的内存快照
    def _check_memory_stat_consistency(self):
        snapshot = torch.cuda.memory_snapshot()

        # 创建一个默认字典，用于存储每个设备的期望值
        expected_each_device = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        # 遍历内存快照中的每个段落
        for segment in snapshot:
            expandable = segment["is_expandable"]
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            # 如果段落不可扩展，增加当前分配和当前池的计数
            if not expandable:
                expected["segment.all.current"] += 1
                expected["segment." + pool_str + ".current"] += 1

            # 增加当前分配字节和当前池的分配字节计数
            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment[
                "allocated_size"
            ]

            # 增加当前保留字节和当前池的总保留字节计数
            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            # 增加当前活跃字节和当前池的活跃字节计数
            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            # 增加当前请求字节和当前池的请求字节计数
            expected["requested_bytes.all.current"] += segment["requested_size"]
            expected["requested_bytes." + pool_str + ".current"] += segment[
                "requested_size"
            ]

            # 计算当前段落的所有请求字节之和
            sum_requested = 0
            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                # 如果块状态为 "active_allocated"，增加所有分配计数和当前池的分配计数
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                # 如果块状态以 "active_" 开头，增加所有活跃计数和当前池的活跃计数，并累加请求字节
                if block["state"].startswith("active_"):
                    sum_requested += block["requested_size"]
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                # 如果块状态为 "inactive"，且存在多个块且不可扩展，增加所有分离的计数和当前池的分离计数
                if block["state"] == "inactive" and is_split and not expandable:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block[
                        "size"
                    ]

            # 断言当前段落的所有请求字节总和等于当前请求字节
            self.assertEqual(sum_requested, segment["requested_size"])

        # 对每个设备及其对应的期望值进行断言
        for device, expected in expected_each_device.items():
            stats = torch.cuda.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])
    # 测试 CUDA 同步操作

    def test_cuda_synchronize(self):
        # 同步所有 CUDA 设备上的操作
        torch.cuda.synchronize()
        # 同步指定名称的 CUDA 设备操作（此处为 "cuda"）
        torch.cuda.synchronize("cuda")
        # 同步指定名称的 CUDA 设备操作（此处为 "cuda:0"）
        torch.cuda.synchronize("cuda:0")
        # 同步指定索引的 CUDA 设备操作（此处为索引 0）
        torch.cuda.synchronize(0)
        # 同步指定 CUDA 设备的操作（通过 torch.device 对象指定 "cuda:0"）
        torch.cuda.synchronize(torch.device("cuda:0"))

        # 如果启用了多 GPU 测试
        if TEST_MULTIGPU:
            # 同步指定名称的 CUDA 设备操作（此处为 "cuda:1"）
            torch.cuda.synchronize("cuda:1")
            # 同步指定索引的 CUDA 设备操作（此处为索引 1）
            torch.cuda.synchronize(1)
            # 同步指定 CUDA 设备的操作（通过 torch.device 对象指定 "cuda:1"）
            torch.cuda.synchronize(torch.device("cuda:1"))

        # 测试预期抛出 ValueError 异常，因为指定的设备是 CPU
        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize(torch.device("cpu"))

        # 测试预期抛出 ValueError 异常，因为指定的设备是 "cpu"
        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but"):
            torch.cuda.synchronize("cpu")

    @staticmethod
    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @serialTest()
    # 测试内存统计信息
    def test_memory_stats(self):
        # 执行垃圾回收
        gc.collect()
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        # 对于内存统计信息生成器中的每个测试，检查内存统计一致性
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 测试多 GPU 下的内存统计信息
    def test_memory_stats_multigpu(self):
        # 辅助函数：推进一个生成器并检查是否到达末尾
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        # 生成器 gen0 指向 "cuda:0" 上的内存统计信息
        gen0 = self._test_memory_stats_generator(self, device="cuda:0", N=35)
        # 生成器 gen1 指向 "cuda:1" 上的内存统计信息
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("cuda:1"), N=35
        )
        end0 = end1 = False
        # 同时推进两个生成器，直到两者都到达末尾
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        # 生成器 gen0 指向索引 0 的 GPU 上的内存统计信息
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        # 生成器 gen1 指向 "cuda:1" 上的内存统计信息
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("cuda:1"), N=35
        )
        end0 = end1 = False

        # 同时推进两个生成器，直到两者都到达末尾
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                # 生成 semi-random order，指定 gen1 最多推进的次数（0到2次）
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                # 如果 gen0 结束，gen1 可以一直推进到结束
                gen1_max_times = torch.inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 测试多 GPU 情况下的内存统计信息
    # 定义一个测试函数，测试自动 GPU 功能
    def test_autogpu(self):
        # 创建一个大小为 5x5 的随机张量，并将其放到 GPU 上
        x = torch.randn(5, 5).cuda()
        # 创建另一个大小为 5x5 的随机张量，并将其放到 GPU 上
        y = torch.randn(5, 5).cuda()
        # 断言张量 x 存在于 GPU 设备 0 上
        self.assertEqual(x.get_device(), 0)
        # 再次断言张量 x 存在于 GPU 设备 0 上
        self.assertEqual(x.get_device(), 0)
        
        # 在 GPU 设备 1 上执行以下代码块
        with torch.cuda.device(1):
            # 创建一个大小为 5x5 的随机张量，并将其放到 GPU 上
            z = torch.randn(5, 5).cuda()
            # 断言张量 z 存在于 GPU 设备 1 上
            self.assertEqual(z.get_device(), 1)
            # 执行张量 x 和 y 的加法操作
            q = x.add(y)
            # 断言结果张量 q 存在于 GPU 设备 0 上
            self.assertEqual(q.get_device(), 0)
            # 创建一个大小为 5x5 的随机张量，并将其放到 GPU 上
            w = torch.randn(5, 5).cuda()
            # 断言张量 w 存在于 GPU 设备 1 上
            self.assertEqual(w.get_device(), 1)
            # 断言张量 y 重新放置在 GPU 设备 1 上后仍存在于 GPU 设备 1 上
            self.assertEqual(y.cuda().get_device(), 1)
        
        # 将张量 z 移回 GPU 设备 0
        z = z.cuda()
        # 断言张量 z 存在于 GPU 设备 0 上
        self.assertEqual(z.get_device(), 0)

    # 跳过测试，如果只检测到一个 GPU 设备
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 定义一个新的测试函数
    def test_new(self):
        # 创建一个大小为 3x3 的随机张量，并将其放到 GPU 上
        x = torch.randn(3, 3).cuda()
        # 断言通过给定的数据创建一个新张量，并确保其存在于 GPU 设备 0 上
        self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
        # 断言通过给定的数据创建一个新张量，并确保其存在于 GPU 设备 1 上
        self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

        # 在 GPU 设备 1 上执行以下代码块
        with torch.cuda.device(1):
            # 断言通过给定的数据创建一个新张量，并确保其存在于 GPU 设备 0 上
            self.assertEqual(x.new([0, 1, 2]).get_device(), 0)
            # 断言通过给定的数据创建一个新张量，并确保其存在于 GPU 设备 1 上
            self.assertEqual(x.new([0, 1, 2], device=1).get_device(), 1)

    # 跳过测试，如果只检测到一个 GPU 设备
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 定义一个复制设备测试函数
    def test_copy_device(self):
        # 创建一个大小为 5x5 的随机张量，并将其放到 GPU 上
        x = torch.randn(5, 5).cuda()
        
        # 在 GPU 设备 1 上执行以下代码块
        with torch.cuda.device(1):
            # 创建一个张量 y，其数据与 x 相同，并将其放到 GPU 设备 1 上
            y = x.cuda()
            # 断言张量 y 存在于 GPU 设备 1 上
            self.assertEqual(y.get_device(), 1)
            # 断言 y.cuda() 返回的对象与 y 是相同的对象
            self.assertIs(y.cuda(), y)
            # 创建一个张量 z，其数据与 y 相同，并将其放到 GPU 设备 0 上
            z = y.cuda(0)
            # 断言张量 z 存在于 GPU 设备 0 上
            self.assertEqual(z.get_device(), 0)
            # 断言 z.cuda(0) 返回的对象与 z 是相同的对象
            self.assertIs(z.cuda(0), z)

        # 创建一个大小为 5x5 的随机张量
        x = torch.randn(5, 5)
        # 在 GPU 设备 1 上执行以下代码块
        with torch.cuda.device(1):
            # 创建一个张量 y，其数据与 x 相同，并将其放到 GPU 设备 1 上
            y = x.cuda()
            # 断言张量 y 存在于 GPU 设备 1 上
            self.assertEqual(y.get_device(), 1)
            # 断言 y.cuda() 返回的对象与 y 是相同的对象
            self.assertIs(y.cuda(), y)
            # 创建一个张量 z，其数据与 y 相同，并将其放到 GPU 设备 0 上
            z = y.cuda(0)
            # 断言张量 z 存在于 GPU 设备 0 上
            self.assertEqual(z.get_device(), 0)
            # 断言 z.cuda(0) 返回的对象与 z 是相同的对象
            self.assertIs(z.cuda(0), z)
    # 定义一个用于测试的私有方法，同步当前流的拷贝操作
    def _test_copy_sync_current_stream(self, x, y):
        # 计算 x+1
        x_plus_one = x + 1
        # 创建两个 CUDA 流对象 s0 和 s1，分别绑定到 x 和 y 的设备上
        s0 = torch.cuda.Stream(device=x.device)
        s1 = torch.cuda.Stream(device=y.device)
        # 创建两个额外的 CUDA 流对象 s2 和 s3，也分别绑定到 x 和 y 的设备上
        s2 = torch.cuda.Stream(device=x.device)
        s3 = torch.cuda.Stream(device=y.device)

        # 在 s0 流中执行 CUDA 操作 _sleep，阻塞当前线程 50 million cycles
        with torch.cuda.stream(s0):
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            # 在 s1 流中执行拷贝操作，将 x_plus_one 拷贝到 y
            with torch.cuda.stream(s1):
                y.copy_(x_plus_one)

        # 在 s2 和 s1 流中执行拷贝操作，将 x 拷贝到 y
        with torch.cuda.stream(s2), torch.cuda.stream(s1):
            y.copy_(x)

        # 同步 s1 流，确保前面的拷贝操作完成
        s1.synchronize()
        # 验证 y 和 x 的内容是否相等
        self.assertEqual(y, x)

        # 在 s1 流中执行 CUDA 操作 _sleep，阻塞当前线程 50 million cycles
        with torch.cuda.stream(s1):
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            # 在 s0 流中执行拷贝操作，将 x_plus_one 拷贝到 y
            with torch.cuda.stream(s0):
                y.copy_(x_plus_one)

        # 在 s3 和 s0 流中执行拷贝操作，将 x 拷贝到 y
        with torch.cuda.stream(s3), torch.cuda.stream(s0):
            y.copy_(x)

        # 同步 s0 流，确保前面的拷贝操作完成
        s0.synchronize()
        # 验证 y 和 x 的内容是否相等
        self.assertEqual(y, x)

    # 如果不支持多 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy_streams(self):
        # 创建一个大小为 5x5 的零张量 x0，在第一个 GPU 上
        d0 = torch.device("cuda:0")
        x0 = torch.zeros(5, 5, device=d0)

        # 创建一个大小为 5x5 的零张量 x1，在第二个 GPU 上
        d1 = torch.device("cuda:1")
        x1 = torch.zeros(5, 5, device=d1)
        # 调用 _test_copy_sync_current_stream 方法测试同步拷贝操作
        self._test_copy_sync_current_stream(x0, x1)

        # 创建一个大小为 5x5 的零张量 x2，在第一个 GPU 上
        x2 = torch.zeros(5, 5, device=d0)
        # 再次调用 _test_copy_sync_current_stream 方法测试同步拷贝操作
        self._test_copy_sync_current_stream(x0, x2)

    # 如果不支持多 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_cat_autogpu(self):
        # 创建一个在第一 GPU 上的 4x4 随机张量 x
        x = torch.randn(4, 4).cuda(1)
        # 创建一个在第一 GPU 上的 4x4 随机张量 y
        y = torch.randn(4, 4).cuda(1)
        # 在第一 GPU 上将 x 和 y 沿着第一个维度连接起来，得到张量 z
        z = torch.cat([x, y], 0)
        # 验证 z 和 x 是否在同一个设备上
        self.assertEqual(z.get_device(), x.get_device())

    # 如果 CUDA 设备数大于等于 10，则跳过测试
    @unittest.skipIf(torch.cuda.device_count() >= 10, "Loading a cuda:9 tensor")
    def test_load_nonexistent_device(self):
        # 设置：创建一个序列化的文件对象，目标设备是 'cuda:9'
        tensor = torch.randn(2, device="cuda")
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # 注意：如果序列化的内容发生变化，这段代码未来可能无法工作
        buf = io.BytesIO(buf.getvalue().replace(b"cuda:0", b"cuda:9"))

        # 预期异常信息，尝试在 CUDA 设备 9 上反序列化对象
        msg = r"Attempting to deserialize object on CUDA device 9"
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    # 如果不支持多 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 如果测试不是在多GPU环境下运行，则跳过这个测试
    def test_multigpu_serialization_remap(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]

        def gpu_remap(storage, location):
            # 定义一个函数用于将存储位置从"cuda:1"映射到"cuda:0"
            if location == "cuda:1":
                return storage.cuda(0)

        with tempfile.NamedTemporaryFile() as f:
            # 将数据x保存到临时文件中
            torch.save(x, f)
            f.seek(0)
            # 从临时文件加载数据x，并使用gpu_remap函数映射位置
            x_copy = torch.load(f, map_location=gpu_remap)

        # 检查每个元素的复制是否与原始元素相等，并且类型一致，并且设备为0
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 如果测试不是在多GPU环境下运行，则跳过这个测试
    def test_multigpu_serialization_remap_dict(self):
        x = [torch.randn(4, 4).cuda(0), torch.randn(4, 4).cuda(1)]
        with tempfile.NamedTemporaryFile() as f:
            # 将数据x保存到临时文件中
            torch.save(x, f)
            f.seek(0)
            # 从临时文件加载数据x，并使用字典映射位置从"cuda:1"到"cuda:0"
            x_copy = torch.load(f, map_location={"cuda:1": "cuda:0"})
        # 检查每个元素的复制是否与原始元素相等，并且类型一致，并且设备为0
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), 0)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 如果测试不是在多GPU环境下运行，则跳过这个测试
    def test_multigpu_storage_clone(self):
        # 在cuda:1设备上创建一个张量，并获取其存储
        x = torch.randn(4, 4, device="cuda:1").storage()
        # 克隆存储并检查设备是否一致
        y = x.clone()
        self.assertEqual(x.get_device(), y.get_device())
        # 检查各种类型的张量是否在相同的设备上
        for t in ["byte", "char", "short", "int", "long", "half", "double"]:
            self.assertEqual(getattr(x, t)().get_device(), x.get_device())

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 如果测试不是在多GPU环境下运行，则跳过这个测试
    def test_cuda_set_device(self):
        x = torch.randn(5, 5)
        with torch.cuda.device(1):
            # 将张量移到cuda:1设备并检查设备编号
            self.assertEqual(x.cuda().get_device(), 1)
            torch.cuda.set_device(0)
            # 将设备切换到cuda:0并检查设备编号
            self.assertEqual(x.cuda().get_device(), 0)
            with torch.cuda.device(1):
                # 在内部上下文中再次检查设备编号，应该仍为cuda:1
                self.assertEqual(x.cuda().get_device(), 1)
            # 退出内部上下文后，设备编号应该回到cuda:0
            self.assertEqual(x.cuda().get_device(), 0)
            # 设置设备为cuda:1并检查设备编号
            torch.cuda.set_device(1)
        # 最后再次检查设备编号，应为cuda:0
        self.assertEqual(x.cuda().get_device(), 0)
    def test_current_stream(self):
        # 创建 CUDA 设备对象
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        # 获取当前 CUDA 流对象，默认使用当前设备的默认流
        s0 = torch.cuda.current_stream()
        # 获取指定设备（1号GPU）的当前流对象
        s1 = torch.cuda.current_stream(device=1)
        # 获取指定设备（0号GPU）的当前流对象
        s2 = torch.cuda.current_stream(device=0)

        # 断言检查：当前流的设备应该与对应的CUDA设备相匹配
        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        # 断言检查：不同设备上的默认流应该是同一个对象
        self.assertEqual(s0, s2)

        # 在指定设备（1号GPU）上执行以下操作
        with torch.cuda.device(d1):
            # 获取当前设备（1号GPU）的当前流对象
            s0 = torch.cuda.current_stream()
            # 获取当前设备（1号GPU）的当前流对象（使用设备索引）
            s1 = torch.cuda.current_stream(1)
            # 获取当前设备（0号GPU）的当前流对象（使用设备对象）
            s2 = torch.cuda.current_stream(d0)

        # 断言检查：当前流的设备应该与对应的CUDA设备相匹配
        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        # 断言检查：不同设备上的流对象应该是同一个对象
        self.assertEqual(s0, s1)

        # 断言检查：尝试在CPU设备上获取流对象会抛出预期的错误
        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but got: cpu"):
            torch.cuda.current_stream(torch.device("cpu"))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self):
        # 创建 CUDA 设备对象
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        # 在设备0上执行以下操作
        with torch.cuda.device(d0):
            # 获取默认流对象
            s0 = torch.cuda.default_stream()

        # 在设备1上执行以下操作
        with torch.cuda.device(d1):
            # 获取默认流对象
            s1 = torch.cuda.default_stream()

        # 获取设备0的默认流对象
        s2 = torch.cuda.default_stream(device=0)
        # 获取设备1的默认流对象
        s3 = torch.cuda.default_stream(d1)

        # 断言检查：默认流的设备应该与对应的CUDA设备相匹配
        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        # 断言检查：不同设备上的默认流对象应该是同一个对象
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        # 在设备0上检查当前流是否与设备0的默认流匹配
        with torch.cuda.device(d0):
            self.assertEqual(torch.cuda.current_stream(), s0)

        # 在设备1上检查当前流是否与设备1的默认流匹配
        with torch.cuda.device(d1):
            self.assertEqual(torch.cuda.current_stream(), s1)

        # 断言检查：尝试在CPU设备上获取默认流对象会抛出预期的错误
        with self.assertRaisesRegex(ValueError, "Expected a cuda device, but got: cpu"):
            torch.cuda.default_stream(torch.device("cpu"))

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_stream_event_device(self):
        # 创建 CUDA 设备对象
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        # 创建 CUDA 事件对象
        e0 = torch.cuda.Event()

        # 断言检查：CUDA事件的设备应该为None
        self.assertEqual(None, e0.device)

        # 在设备0上执行以下操作
        with torch.cuda.device(d0):
            # 获取当前设备（0号GPU）的当前流对象
            s0 = torch.cuda.current_stream()
            # 记录事件到当前流
            s0.record_event(e0)

        # 在设备1上执行以下操作
        with torch.cuda.device(d1):
            # 创建新的CUDA流对象
            s1 = torch.cuda.Stream()
            # 记录事件到新创建的流对象
            e1 = s1.record_event()

        # 断言检查：记录事件后，流对象和事件对象的设备应该与预期的设备匹配
        self.assertEqual(s0.device, torch.device("cuda:0"))
        self.assertEqual(e0.device, torch.device("cuda:0"))
        self.assertEqual(s1.device, torch.device("cuda:1"))
        self.assertEqual(e1.device, torch.device("cuda:1"))
    # 定义一个测试方法，用于验证 CUDA 流的上下文切换行为
    def test_stream_context(self):
        # 获取当前默认的 CUDA 流
        s0 = torch.cuda.current_stream()
        # 创建一个指定设备的 CUDA 流
        s1 = torch.cuda.Stream(device=1)
        s2 = torch.cuda.Stream(device=0)

        # 在指定设备上下文中执行代码块
        with torch.cuda.device(s1.device):
            # 记录进入此设备上下文前的当前流
            prev_stream_on_cuda1 = torch.cuda.current_stream()

        # 断言当前流与 s0 相同
        self.assertEqual(torch.cuda.current_stream(), s0)
        # 断言当前设备为设备 0
        self.assertEqual(0, torch.cuda.current_device())

        # 在指定流上下文中执行代码块
        with torch.cuda.stream(s1):
            # 断言当前流与 s1 相同
            self.assertEqual(torch.cuda.current_stream(), s1)
            # 断言当前设备为设备 1
            self.assertEqual(1, torch.cuda.current_device())

            # 在另一个流上下文中执行代码块
            with torch.cuda.stream(s2):
                # 断言当前流与 s2 相同
                self.assertEqual(torch.cuda.current_stream(), s2)
                # 断言当前设备为设备 0
                self.assertEqual(0, torch.cuda.current_device())

                # 在默认流上下文中执行代码块
                with torch.cuda.stream(s0):
                    # 断言当前流与 s0 相同
                    self.assertEqual(torch.cuda.current_stream(), s0)
                    # 断言当前设备为设备 0
                    self.assertEqual(0, torch.cuda.current_device())

                # 断言当前流与 s2 相同
                self.assertEqual(torch.cuda.current_stream(), s2)
                # 断言当前设备为设备 0
                self.assertEqual(0, torch.cuda.current_device())

            # 断言当前流与 s1 相同
            self.assertEqual(torch.cuda.current_stream(), s1)
            # 断言当前设备为设备 1
            self.assertEqual(1, torch.cuda.current_device())

        # 在指定设备上下文中断言之前记录的流与当前流相同
        with torch.cuda.device(s1.device):
            self.assertEqual(prev_stream_on_cuda1, torch.cuda.current_stream())

        # 断言当前流与 s0 相同
        self.assertEqual(torch.cuda.current_stream(), s0)
        # 断言当前设备为设备 0
        self.assertEqual(0, torch.cuda.current_device())

    # 根据是否测试多 GPU，决定是否跳过此测试方法
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_streams_multi_gpu(self):
        # 获取默认 CUDA 流
        default_stream = torch.cuda.current_stream()
        # 断言默认流所在设备为 CUDA 设备 0
        self.assertEqual(default_stream.device, torch.device("cuda:0"))
        # 创建一个指定设备的 CUDA 流
        stream = torch.cuda.Stream(device=1)
        # 断言新创建的流所在设备为 CUDA 设备 1
        self.assertEqual(stream.device, torch.device("cuda:1"))

        # 在指定设备上下文中执行代码块
        with torch.cuda.device(1):
            # 断言当前流的设备为 CUDA 设备 1
            self.assertEqual(torch.cuda.current_stream().device, torch.device("cuda:1"))
            # 断言当前流不等于默认流
            self.assertNotEqual(torch.cuda.current_stream(), default_stream)
    # 定义一个测试方法，用于测试多 GPU 下的流查询
    def test_streams_multi_gpu_query(self):
        # 设定第一个 GPU 设备
        d0 = torch.device("cuda:0")
        # 设定第二个 GPU 设备
        d1 = torch.device("cuda:1")
        # 同步第一个 GPU 的 CUDA 流
        torch.cuda.synchronize(d0)
        # 同步第二个 GPU 的 CUDA 流
        torch.cuda.synchronize(d1)

        # 切换当前 CUDA 设备为 d0
        with torch.cuda.device(d0):
            # 获取当前 CUDA 设备的流对象
            s0 = torch.cuda.current_stream()

        # 切换当前 CUDA 设备为 d1
        with torch.cuda.device(d1):
            # 获取当前 CUDA 设备的流对象
            s1 = torch.cuda.current_stream()
            # 在 d1 设备上执行一个大量计算的休眠操作
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)

        # 断言第一个流对象 s0 的查询状态为真
        self.assertTrue(s0.query())
        # 断言第二个流对象 s1 的查询状态为假
        self.assertFalse(s1.query())

        # 切换当前 CUDA 设备为 d0
        with torch.cuda.device(d0):
            # 断言第一个流对象 s0 的查询状态为真
            self.assertTrue(s0.query())
            # 断言第二个流对象 s1 的查询状态为假
            self.assertFalse(s1.query())

        # 切换当前 CUDA 设备为 d1
        with torch.cuda.device(d1):
            # 断言第一个流对象 s0 的查询状态为真
            self.assertTrue(s0.query())
            # 断言第二个流对象 s1 的查询状态为假
            self.assertFalse(s1.query())

        # 故意在不同的设备上执行操作
        with torch.cuda.device(d0):
            # 同步第二个流对象 s1 在当前设备上
            s1.synchronize()

        # 断言第一个流对象 s0 的查询状态为真
        self.assertTrue(s0.query())
        # 断言第二个流对象 s1 的查询状态为真
        self.assertTrue(s1.query())

        # 切换当前 CUDA 设备为 d0
        with torch.cuda.device(d0):
            # 断言第一个流对象 s0 的查询状态为真
            self.assertTrue(s0.query())
            # 断言第二个流对象 s1 的查询状态为真
            self.assertTrue(s1.query())

        # 切换当前 CUDA 设备为 d1
        with torch.cuda.device(d1):
            # 断言第一个流对象 s0 的查询状态为真
            self.assertTrue(s0.query())
            # 断言第二个流对象 s1 的查询状态为真
            self.assertTrue(s1.query())

    # 根据是否支持多 GPU 进行跳过测试的装饰器
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 定义测试方法，用于测试多 GPU 下的流对象相等性
    def test_streams_multi_gpu_eq(self):
        # 设定第一个 GPU 设备
        d0 = torch.device("cuda:0")
        # 设定第二个 GPU 设备
        d1 = torch.device("cuda:1")

        # 切换当前 CUDA 设备为 d0
        with torch.cuda.device(d0):
            # 获取当前 CUDA 设备的流对象
            s0 = torch.cuda.current_stream()
            # 获取当前 CUDA 设备的流对象
            s1 = torch.cuda.current_stream()

        # 切换当前 CUDA 设备为 d1
        with torch.cuda.device(d1):
            # 获取当前 CUDA 设备的流对象
            s2 = torch.cuda.current_stream()
            # 获取当前 CUDA 设备的流对象
            s3 = torch.cuda.current_stream()

        # 断言流对象相等性
        self.assertTrue(s0 == s0)
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)

        # 断言流对象的设备和 CUDA 流的相等性
        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.cuda_stream, s1.cuda_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.cuda_stream, s3.cuda_stream)
        self.assertNotEqual(s0.device, s3.device)

        # 断言流对象的哈希值相等性
        self.assertEqual(hash(s0), hash(s1))
        self.assertEqual(hash(s2), hash(s3))
        self.assertNotEqual(hash(s0), hash(s3))

    # 根据是否支持多 GPU 进行跳过测试的装饰器
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # 定义测试方法，用于测试多 GPU 下流的优先级
    def test_streams_priority(self):
        # 获取当前 CUDA 流的优先级范围
        low, high = torch.cuda.Stream.priority_range()
        # 创建一个优先级为 low 的第一个 GPU 设备的 CUDA 流对象
        s0 = torch.cuda.Stream(device=0, priority=low)

        # 断言第一个流对象的优先级为 low
        self.assertEqual(low, s0.priority)
        # 断言第一个流对象的设备为 CUDA 设备 "cuda:0"
        self.assertEqual(torch.device("cuda:0"), s0.device)

        # 创建一个优先级为 high 的第二个 GPU 设备的 CUDA 流对象
        s1 = torch.cuda.Stream(device=1, priority=high)

        # 断言第二个流对象的优先级为 high
        self.assertEqual(high, s1.priority)
        # 断言第二个流对象的设备为 CUDA 设备 "cuda:1"
        self.assertEqual(torch.device("cuda:1"), s1.device)
    # 测试函数，验证张量在不同设备上的行为
    def test_tensor_device(self):
        # 断言在默认设备上创建的张量的设备编号为0
        self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 0)
        # 断言在指定设备1上创建的张量的设备编号为1
        self.assertEqual(torch.cuda.FloatTensor(1, device=1).get_device(), 1)
        # 使用上下文管理器将当前设备切换为1，并验证在该设备上创建的张量的设备编号为1
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.FloatTensor(1).get_device(), 1)
            # 在上下文管理器中切换回设备0，并验证在该设备上创建的张量的设备编号为0
            self.assertEqual(torch.cuda.FloatTensor(1, device=0).get_device(), 0)
            # 在上下文管理器中切换回设备1，并验证在该设备上创建的张量的设备编号为1
            self.assertEqual(torch.cuda.FloatTensor(1, device=None).get_device(), 1)

    # 静态方法，用于同步 CUDA 流的操作
    @staticmethod
    def _stream_synchronize(self, spin_time_cycles):
        # 获取当前 CUDA 流
        s = torch.cuda.current_stream()
        # 创建启用计时的 CUDA 事件
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        # 记录 e_tik 事件在当前流 s 上
        e_tik.record(s)
        # 使用 CUDA 的休眠功能模拟指定周期的自旋等待
        torch.cuda._sleep(spin_time_cycles)
        # 记录 e_tok 事件在当前流 s 上
        e_tok.record(s)
        # 同步当前流 s
        s.synchronize()

        # 断言当前流 s 是活跃的
        self.assertTrue(s.query())

        # 不需要检查 e_tik 和 e_tok，因为如果情况不对，elapsed_time 会抛出异常
        # 返回 e_tik 和 e_tok 之间的经过时间
        return e_tik.elapsed_time(e_tok)

    # 静态方法，用于同步 CUDA 事件的操作
    @staticmethod
    def _event_synchronize(self, spin_time_cycles):
        # 获取当前 CUDA 流
        s = torch.cuda.current_stream()
        # 创建启用计时的 CUDA 事件
        e_tik = torch.cuda.Event(enable_timing=True)
        e_tok = torch.cuda.Event(enable_timing=True)

        # 记录 e_tik 事件在当前流 s 上
        e_tik.record(s)
        # 使用 CUDA 的休眠功能模拟指定周期的自旋等待
        torch.cuda._sleep(spin_time_cycles)
        # 记录 e_tok 事件在当前流 s 上
        s.record_event(e_tok)
        # 同步 e_tok 事件
        e_tok.synchronize()

        # 断言当前流 s 是活跃的
        self.assertTrue(s.query())

        # 不需要检查 e_tik 和 e_tok，因为如果情况不对，elapsed_time 会抛出异常
        # 返回 e_tik 和 e_tok 之间的经过时间
        return e_tik.elapsed_time(e_tok)

    # 静态方法，用于等待 CUDA 事件的操作
    @staticmethod
    def _event_wait(self, spin_time_cycles):
        # 获取当前 CUDA 流
        s0 = torch.cuda.current_stream()
        # 创建新的 CUDA 流 s1
        s1 = torch.cuda.Stream()
        # 创建启用计时和阻塞的 CUDA 事件 e_tik 和 e_tok
        e_tik = torch.cuda.Event(blocking=True, enable_timing=True)
        e_tok = torch.cuda.Event(blocking=True, enable_timing=True)

        # 记录 e_tik 事件在当前流 s0 上
        e_tik.record(s0)
        # 使用 CUDA 的休眠功能模拟少量周期的自旋等待
        torch.cuda._sleep(spin_time_cycles - 10)
        # 创建一个阻塞事件 e_sync 并记录在当前流 s0 上
        e_sync = torch.cuda.Event(blocking=True)
        e_sync.record()
        # 在 s1 流上等待 e_sync 事件
        e_sync.wait(s1)
        # 使用 s1 流完成额外的工作
        with torch.cuda.stream(s1):
            torch.cuda._sleep(10)
        # 同步 s1 流
        s1.synchronize()
        # 记录 e_tok 事件在当前流 s0 上
        e_tok.record()
        # 同步 e_tok 事件
        e_tok.synchronize()

        # 断言当前流 s0 和 s1，以及事件 e_sync 是活跃的
        self.assertTrue(s0.query())
        self.assertTrue(s1.query())
        self.assertTrue(e_sync.query())

        # 不需要检查 e_tik 和 e_tok，因为如果情况不对，elapsed_time 会抛出异常
        # 返回 e_tik 和 e_tok 之间的经过时间
        return e_tik.elapsed_time(e_tok)

    # 静态方法，用于在多 GPU 环境中测试流和事件同步，无 GIL 锁定
    @staticmethod
    def _test_stream_event_nogil(self, sync_func, p2c, c2p):
        # 切换到 CUDA 设备 "cuda:1" 上运行
        with torch.cuda.device("cuda:1"):
            # 向 c2p 队列放入值 0
            c2p.put(0)
            # 从 p2c 队列获取值
            p2c.get()
            # 向 sync_func 函数传递当前对象和指定的周期数，将结果放入 c2p 队列
            c2p.put(sync_func(self, TestCudaMultiGPU.FIFTY_MIL_CYCLES))

    # 如果是 ROCm 平台，跳过测试，详情请参考 https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    # 如果只检测到一个 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 定义测试方法，测试无全局解释器锁（GIL）环境下的流和事件同步
    def test_stream_event_nogil(self):
        # 遍历同步函数列表，包括流同步和事件同步
        for sync_func in [
            TestCudaMultiGPU._stream_synchronize,
            TestCudaMultiGPU._event_synchronize,
            TestCudaMultiGPU._event_wait,
        ]:
            # 创建父子进程通信队列
            p2c = queue.Queue()
            c2p = queue.Queue()
            # 创建两个 CUDA 事件对象，用于计时
            e_tik = torch.cuda.Event(enable_timing=True)
            e_tok = torch.cuda.Event(enable_timing=True)

            # 创建线程对象，执行测试函数 _test_stream_event_nogil
            t = threading.Thread(
                target=TestCudaMultiGPU._test_stream_event_nogil,
                args=(self, sync_func, p2c, c2p),
            )
            t.daemon = True  # 设置为守护线程
            t.start()  # 启动线程

            c2p.get()  # 等待子线程初始化完成
            with torch.cuda.device("cuda:0"):
                e_tik.record()  # 记录开始时间
                p2c.put(0)  # 向子线程发送信号
                parent_time = sync_func(self, TestCudaMultiGPU.FIFTY_MIL_CYCLES)  # 执行同步函数，返回父进程时间
                child_time = c2p.get()  # 获取子进程时间
                e_tok.record()  # 记录结束时间
                e_tok.synchronize()  # 同步 CUDA 事件
                total_time = e_tik.elapsed_time(e_tok)  # 计算总耗时

            # 在无 GIL 环境下，父子线程的同步可能会重叠。总执行时间应略长于 五千万次循环，
            # 但明显短于两倍。然而，绝对执行时间测试不可靠，因为在不同硬件和环境中可能会有所不同。
            # 因此，此测试使用相对比较，检查父子线程执行时间之和是否至少比实际执行时间长 40%。
            self.assertGreater(parent_time + child_time, total_time * 1.4)

    # 对于 ROCm 环境，此测试存在问题，参见问题 #62602
    @skipIfRocm  # 如果在 ROCm 环境下跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")  # 如果检测到只有一个 GPU，则跳过测试
    def test_events_wait(self):
        d0 = torch.device("cuda:0")  # 设备 d0 是 cuda:0
        d1 = torch.device("cuda:1")  # 设备 d1 是 cuda:1
        torch.cuda.synchronize(d0)  # 在设备 d0 上同步 CUDA
        torch.cuda.synchronize(d1)  # 在设备 d1 上同步 CUDA

        with torch.cuda.device(d0):
            s0 = torch.cuda.current_stream()  # 获取当前流 s0
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)  # 在设备 d0 上睡眠
            e0 = torch.cuda.Event()  # 创建 CUDA 事件 e0
            s0.record_event(e0)  # 在流 s0 上记录事件 e0

        with torch.cuda.device(d1):
            s1 = torch.cuda.current_stream()  # 获取当前流 s1

        self.assertFalse(s0.query())  # 断言流 s0 未完成
        self.assertTrue(s1.query())  # 断言流 s1 已完成

        s1.wait_event(e0)  # 等待流 s1 上的事件 e0
        s1.synchronize()  # 同步流 s1

        self.assertTrue(e0.query())  # 断言事件 e0 已完成
        self.assertTrue(s0.query())  # 断言流 s0 已完成
        self.assertTrue(s1.query())  # 断言流 s1 已完成

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")  # 如果检测到只有一个 GPU，则跳过测试
    # 定义一个测试方法，用于测试多 GPU 环境下事件查询功能
    def test_events_multi_gpu_query(self):
        # 设定变量 d0 和 d1 分别表示 cuda:0 和 cuda:1 设备
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        # 在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 获取当前 CUDA 流
            s0 = torch.cuda.current_stream()
            # 创建一个 CUDA 事件并记录
            e0 = s0.record_event()
            # 同步 CUDA 流
            s0.synchronize()

        # 在 cuda:1 设备上执行以下操作
        with torch.cuda.device(d1):
            # 获取当前 CUDA 流
            s1 = torch.cuda.current_stream()
            # 在 CUDA 流上执行一个长时间的操作
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            # 创建一个 CUDA 事件并记录
            e1 = s1.record_event()

        # 断言 e0 和 e1 的查询结果
        self.assertTrue(e0.query())
        self.assertFalse(e1.query())

        # 再次在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 断言 e0 和 e1 的查询结果
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        # 再次在 cuda:1 设备上执行以下操作
        with torch.cuda.device(d1):
            # 断言 e0 和 e1 的查询结果
            self.assertTrue(e0.query())
            self.assertFalse(e1.query())

        # 故意在不同设备上执行以下操作
        with torch.cuda.device(d0):
            # 同步 CUDA 事件 e1
            e1.synchronize()

        # 断言 e0 和 e1 的查询结果
        self.assertTrue(e0.query())
        self.assertTrue(e1.query())

        # 再次在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 断言 e0 和 e1 的查询结果
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

        # 再次在 cuda:1 设备上执行以下操作
        with torch.cuda.device(d1):
            # 断言 e0 和 e1 的查询结果
            self.assertTrue(e0.query())
            self.assertTrue(e1.query())

    # 使用 unittest.skipIf 标记条件不满足时跳过测试，用于多 GPU 环境下计时功能的测试
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    @skipIfRocm
    def test_events_multi_gpu_elapsed_time(self):
        # 设定变量 d0 和 d1 分别表示 cuda:0 和 cuda:1 设备
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")

        # 在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 获取当前 CUDA 流
            s0 = torch.cuda.current_stream()
            # 创建一个启用计时的 CUDA 事件 e0，并记录
            e0 = torch.cuda.Event(enable_timing=True)
            # 在 CUDA 流上睡眠 10 毫秒
            torch.cuda._sleep(10)
            s0.record_event(e0)

        # 在 cuda:1 设备上执行以下操作
        with torch.cuda.device(d1):
            # 获取当前 CUDA 流
            s1 = torch.cuda.current_stream()
            # 创建一个启用计时的 CUDA 事件 e1，并记录
            e1 = torch.cuda.Event(enable_timing=True)
            # 在 CUDA 流上执行一个长时间的操作
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            s1.record_event(e1)

        # 同步 CUDA 事件 e0 和 e1
        e0.synchronize()
        e1.synchronize()

        # 在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 断言 e0 和 e1 的间隔时间大于 0
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        # 在 cuda:1 设备上执行以下操作
        with torch.cuda.device(d1):
            # 断言 e0 和 e1 的间隔时间大于 0
            with self.assertRaises(RuntimeError):
                self.assertGreater(e0.elapsed_time(e1), 0)

        # 再次在 cuda:0 设备上执行以下操作
        with torch.cuda.device(d0):
            # 获取当前 CUDA 流
            s0 = torch.cuda.current_stream()
            # 创建一个启用计时的 CUDA 事件 e2，并记录
            e2 = torch.cuda.Event(enable_timing=True)
            # 在 CUDA 流上执行一个长时间的操作
            torch.cuda._sleep(TestCudaMultiGPU.FIFTY_MIL_CYCLES)
            s0.record_event(e2)
            # 同步 CUDA 流
            s0.synchronize()

        # 断言 e0 和 e2 的间隔时间大于 0
        self.assertGreater(e0.elapsed_time(e2), 0)

        # 故意从不同设备调用以下操作
        with torch.cuda.device(d1):
            # 断言 e0 和 e2 的间隔时间大于 0
            self.assertGreater(e0.elapsed_time(e2), 0)
    # 定义一个方法来获取外部流，需要传入一个设备对象作为参数
    def _get_external_stream(self, device):
        # 获取CUDA运行时库的实例
        cudart = torch.cuda.cudart()
        # 创建一个表示流的无符号长整型变量
        stream = ctypes.c_ulonglong(0)
        # 创建一个指向stream的指针，类型为void指针的指针
        stream_p = ctypes.POINTER(ctypes.c_void_p)(stream)
        # 将void指针转换为整型，并获取其值
        stream_p_int = ctypes.cast(stream_p, ctypes.c_void_p).value
        # 使用给定的设备对象进入上下文管理器
        with device:
            try:
                # 调用CUDA运行时库创建CUDA流
                out = cudart.cudaStreamCreate(stream_p_int)
                # 断言CUDA流创建成功
                self.assertEqual(out, 0)
                # 断言stream的值不为0
                self.assertNotEqual(stream.value, 0)
                # 生成器函数的yield语句，返回stream的值
                yield stream.value
            finally:
                # 销毁CUDA流
                out = cudart.cudaStreamDestroy(stream.value)
                # 断言CUDA流销毁成功
                self.assertEqual(out, 0)

    # 测试外部流功能
    def test_external_streams(self):
        # 获取第一个CUDA设备
        device = torch.cuda.device(0)
        # 进入_get_external_stream方法的上下文管理器，并获取流的值
        with self._get_external_stream(device) as stream_v:
            # 创建外部流对象
            ext_stream = torch.cuda.ExternalStream(stream_v)
            # 断言外部流的CUDA流与stream_v相等
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            # 断言外部流的设备索引与device的索引相等
            self.assertEqual(ext_stream.device.index, device.idx)

    # 如果不支持多GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    # 测试多设备下的外部流功能
    def test_external_streams_multi_device(self):
        # 获取第二个CUDA设备
        device = torch.cuda.device(1)
        # 进入_get_external_stream方法的上下文管理器，并获取流的值
        with self._get_external_stream(device) as stream_v:
            # 创建带有指定设备的外部流对象
            ext_stream = torch.cuda.ExternalStream(stream_v, device=device)
            # 断言外部流的CUDA流与stream_v相等
            self.assertEqual(stream_v, ext_stream.cuda_stream)
            # 断言外部流的设备索引与device的索引相等
            self.assertEqual(ext_stream.device.index, device.idx)

    # 如果不支持多GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 测试多GPU环境下固定内存缓存的缓存与重用
    def test_caching_pinned_memory_multi_gpu(self):
        # 获取每毫秒的CPU周期数
        cycles_per_ms = get_cycles_per_ms()

        # 创建一个浮点数张量，并固定在内存中
        t = torch.FloatTensor([1]).pin_memory()
        # 记录t的数据指针
        ptr = t.data_ptr()
        # 在第一个GPU上创建一个浮点数张量
        gpu_tensor0 = torch.cuda.FloatTensor([0], device=0)
        # 在第二个GPU上创建一个浮点数张量
        gpu_tensor1 = torch.cuda.FloatTensor([0], device=1)

        # 切换到第二个GPU，并延迟复制1秒钟
        with torch.cuda.device(1):
            torch.cuda._sleep(int(1000 * cycles_per_ms))  # delay the copy by 1s
            # 使用非阻塞方式将t复制到gpu_tensor1
            gpu_tensor1.copy_(t, non_blocking=True)

        # 删除张量t
        del t
        # 重新创建一个新的浮点数张量，并固定在内存中
        t = torch.FloatTensor([2]).pin_memory()
        # 断言新创建的张量t的数据指针与之前不相同
        self.assertNotEqual(t.data_ptr(), ptr, msg="allocation re-used too soon")

        # 切换到第一个GPU，并使用非阻塞方式将t复制到gpu_tensor0
        with torch.cuda.device(0):
            gpu_tensor0.copy_(t, non_blocking=True)

        # 断言gpu_tensor1的第一个元素为1
        self.assertEqual(gpu_tensor1[0], 1)
        # 断言gpu_tensor0的第一个元素为2
        self.assertEqual(gpu_tensor0[0], 2)

    # 如果不支持多GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 测试获取和设置所有GPU的随机数生成器状态
    def test_get_set_rng_state_all(self):
        # 获取所有GPU的随机数生成器状态
        states = torch.cuda.get_rng_state_all()
        # 在第一个GPU上创建一个标准正态分布的张量，并记录其状态
        before0 = torch.cuda.FloatTensor(100, device=0).normal_()
        # 在第二个GPU上创建一个标准正态分布的张量，并记录其状态
        before1 = torch.cuda.FloatTensor(100, device=1).normal_()
        # 恢复所有GPU的随机数生成器状态
        torch.cuda.set_rng_state_all(states)
        # 在第一个GPU上创建一个新的标准正态分布的张量，并记录其状态
        after0 = torch.cuda.FloatTensor(100, device=0).normal_()
        # 在第二个GPU上创建一个新的标准正态分布的张量，并记录其状态
        after1 = torch.cuda.FloatTensor(100, device=1).normal_()
        # 断言恢复后的第一个GPU的张量与之前相同
        self.assertEqual(before0, after0, atol=0, rtol=0)
        # 断言恢复后的第二个GPU的张量与之前相同
        self.assertEqual(before1, after1, atol=0, rtol=0)
    # 测试函数：验证 `torch.cuda.get_rng_state()` 函数返回的随机数生成器状态
    def test_rng_state_offset(self):
        # 记录测试前的随机数生成器状态
        before = torch.cuda.get_rng_state()
        # 设置随机数生成器状态的偏移量为100
        torch.cuda._set_rng_state_offset(100)
        # 获取设置后的随机数生成器状态的偏移量
        offset = torch.cuda._get_rng_state_offset()
        # 恢复到测试前的随机数生成器状态
        torch.cuda.set_rng_state(before)
        # 断言设置的偏移量是否为100
        self.assertEqual(offset, 100)

    # 测试函数：验证 `torch.cuda.mem_get_info()` 函数的工作是否正常，包括不同设备的调用
    def test_mem_get_info(self):
        # 内部测试函数，接受设备索引参数 `idx`
        def _test(idx):
            # 获取测试前的显存空闲字节数和可用字节数
            before_free_bytes, before_available_bytes = torch.cuda.mem_get_info(idx)
            # 分配一个大小为8MB的张量在指定设备上，以强制获取新的显存块并克服不同平台的块大小差异
            t = torch.randn(1024 * 1024 * 8, device="cuda:" + str(idx))
            # 如果是 Jetson 平台，需要同步显存操作
            if IS_JETSON:
                # 同步操作，确保显存分配完成
                torch.cuda.synchronize()
            # 获取测试后的显存空闲字节数和可用字节数
            after_free_bytes, after_available_bytes = torch.cuda.mem_get_info(idx)

            # 断言显存空闲字节数是否减少
            self.assertLess(after_free_bytes, before_free_bytes)
            # 断言显存可用字节数没有变化
            self.assertEqual(before_available_bytes, after_available_bytes)

        # 在设备0上执行测试
        _test(0)
        # 如果支持多GPU测试，同时在设备1上执行测试
        if TEST_MULTIGPU:
            _test(1)

    # 测试函数：验证 `wrap_with_cuda_memory_check` 是否能成功检测内存泄漏
    def test_cuda_memory_leak_detection(self):
        # 定义一个空列表
        l = []

        # 包装函数，使用 `wrap_with_cuda_memory_check` 装饰器，无内存泄漏情况
        @self.wrap_with_cuda_memory_check
        def no_leak():
            pass

        # 包装函数，使用 `wrap_with_cuda_memory_check` 装饰器，有内存泄漏情况，分配到设备cuda:0
        @self.wrap_with_cuda_memory_check
        def leak_gpu0():
            # 分配一个大小为8MB的张量到cuda:0设备上
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("cuda:0")))

        # 调用无内存泄漏的函数
        no_leak()
        # 匹配的正则表达式，用于验证是否捕获到CUDA驱动API确认的内存泄漏信息
        regex = r"CUDA driver API confirmed .+ on device 0.+"
        # 如果是 Jetson 平台，使用异常处理来验证内存泄漏信息
        if IS_JETSON:
            try:
                leak_gpu0()
            except RuntimeError as e:
                import re

                # 使用正则表达式匹配异常信息
                assert re.match(regex, str(e)), str(e) + "\n does not match: \n" + regex
        else:
            # 在其他平台上，使用断言验证是否捕获到期望的异常信息
            with self.assertRaisesRegex(RuntimeError, regex):
                leak_gpu0()

        # 如果支持多GPU测试
        if TEST_MULTIGPU:

            # 包装函数，使用 `wrap_with_cuda_memory_check` 装饰器，有内存泄漏情况，分配到设备cuda:1
            @self.wrap_with_cuda_memory_check
            def leak_gpu1():
                # 分配一个大小为8MB的张量到cuda:1设备上
                l.append(torch.randn(1024 * 1024 * 8, device=torch.device("cuda:1")))

            # 使用断言验证是否捕获到期望的异常信息，用于验证cuda:1设备上的内存泄漏情况
            with self.assertRaisesRegex(
                RuntimeError, r"CUDA driver API confirmed .+ on device 1.+"
            ):
                leak_gpu1()

    # 跳过测试：当仅检测到一个GPU时，跳过多GPU测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_streaming_backwards_device_transfer(self):
        # This function tests streaming backward transfers between devices using large tensors.
        # It ensures that to()'s backward (CopyBackward) properly interacts with synchronization logic.

        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")

        # Create large tensors on dev1 with requires_grad=True
        size = 2**26
        a = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)
        b = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)

        # Compute the product a*b on dev1
        to_backward_recipient = a * b

        # Sum the tensor on dev0
        s = to_backward_recipient.to(device="cuda:0").sum()

        # Synchronize CUDA streams for dev0 and dev1
        torch.cuda.synchronize(device=dev0)
        torch.cuda.synchronize(device=dev1)

        # Perform backward pass from s
        s.backward()

        # Assert gradients
        self.assertTrue(a.grad.sum().item() == size)
        self.assertTrue(b.grad.sum().item() == size)

        # Test scenario where to_backward_recipient = a*b is used twice
        a.grad = None
        b.grad = None
        to_backward_recipient = a * b

        # Compute sums multiplied by 2 on dev0
        s0 = to_backward_recipient.to(device="cuda:0").sum() * 2.0
        s1 = to_backward_recipient.to(device="cuda:0").sum() * 2.0

        # Synchronize CUDA streams for dev0 and dev1
        torch.cuda.synchronize(device=dev0)
        torch.cuda.synchronize(device=dev1)

        # Perform backward passes from s0 and s1
        s0.backward(retain_graph=True)
        s1.backward()

        # Assert gradients
        self.assertTrue(a.grad.sum().item() == 4 * size)
        self.assertTrue(b.grad.sum().item() == 4 * size)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @unittest.skipIf(IS_SANDCASTLE or IS_REMOTE_GPU, "Does not work on Sandcastle")
    def test_cuda_init_race(self):
        # See https://github.com/pytorch/pytorch/issues/16559
        import subprocess

        # Run a subprocess to check CUDA initialization race condition
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                """\
import torch  # 导入 PyTorch 库
import threading  # 导入线程库

def worker(rank):
    torch.tensor([1.]).cuda(rank)  # 在指定的 GPU 设备上创建张量

t1 = threading.Thread(target=worker, args=(0,))  # 创建线程 t1，目标函数为 worker，参数为 0
t2 = threading.Thread(target=worker, args=(1,))  # 创建线程 t2，目标函数为 worker，参数为 1
t1.start()  # 启动线程 t1
t2.start()  # 启动线程 t2



    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_grad_scaling_device_as_key(self):
        # 确保指向同一设备的不同 "device" 对象在字典中被视为相同的键。
        # GradScaler 依赖此行为，否则可能会导致难以检测的错误（性能损失）。
        d = {}
        t = torch.empty((1,), device="cuda:0")  # 创建一个张量 t，指定在 cuda:0 设备上
        dev0a = torch.device("cuda:0")  # 创建设备对象 dev0a，指向 cuda:0
        dev0b = torch.device("cuda:0")  # 创建设备对象 dev0b，同样指向 cuda:0
        dev1a = torch.device("cuda:1")  # 创建设备对象 dev1a，指向 cuda:1
        dev1b = torch.device("cuda:1")  # 创建设备对象 dev1b，同样指向 cuda:1

        self.assertTrue(hash(dev0a) == hash(dev0b))  # 断言 dev0a 和 dev0b 的哈希值相同
        self.assertTrue(hash(dev1a) == hash(dev1b))  # 断言 dev1a 和 dev1b 的哈希值相同

        d[dev0a] = "0a"  # 将 dev0a 设备对象作为键 "0a" 存入字典 d
        d[dev0b] = "0b"  # 将 dev0b 设备对象作为键 "0b" 存入字典 d，覆盖了之前的 "0a"
        self.assertTrue(len(d) == 1)  # 断言字典 d 的长度为 1
        self.assertTrue(d[dev0a] == "0b")  # 断言字典 d 中 dev0a 对应的值为 "0b"
        d[t.device] = "t"  # 将张量 t 的设备作为键 "t" 存入字典 d
        self.assertTrue(len(d) == 1)  # 再次断言字典 d 的长度为 1
        self.assertTrue(d[dev0a] == "t")  # 断言字典 d 中 dev0a 对应的值为 "t"

        d[dev1a] = "1a"  # 将 dev1a 设备对象作为键 "1a" 存入字典 d
        d[dev1b] = "1b"  # 将 dev1b 设备对象作为键 "1b" 存入字典 d
        self.assertTrue(len(d) == 2)  # 断言字典 d 的长度为 2
        self.assertTrue(d[dev1a] == "1b")  # 断言字典 d 中 dev1a 对应的值为 "1b"



    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_grad_scaling_scale(self):
        scaler = torch.amp.GradScaler(device="cuda", init_scale=2.0)  # 创建一个梯度缩放器，指定在 cuda 设备上，初始缩放比例为 2.0
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:0")  # 创建一个张量 t0，全为 4.0，指定在 cuda:0 设备上
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="cuda:1")  # 创建一个张量 t1，全为 4.0，指定在 cuda:1 设备上
        # 创建一些嵌套的张量迭代器，分布在不同设备上
        outputs = (
            t1.clone(),  # 将 t1 克隆并放入 outputs
            (t0.clone(), t1.clone()),  # 将 t0 和 t1 克隆并作为元组放入 outputs
            [t0.clone(), (t1.clone(), t0.clone())],  # 将 t0 和一个嵌套的元组放入列表中，并放入 outputs
        )
        outputs = scaler.scale(outputs)  # 使用梯度缩放器对 outputs 进行缩放
        self.assertTrue(
            outputs[0] == 8.0  # 断言 outputs 的第一个元素为 8.0
            and outputs[1][0] == 8.0  # 断言 outputs 的第二个元素的第一个元素为 8.0
            and outputs[1][1] == 8.0  # 断言 outputs 的第二个元素的第二个元素为 8.0
            and outputs[2][0] == 8.0  # 断言 outputs 的第三个元素的第一个元素为 8.0
            and outputs[2][1][0] == 8.0  # 断言 outputs 的第三个元素的第二个元素的第一个元素为 8.0
            and outputs[2][1][1] == 8.0  # 断言 outputs 的第三个元素的第二个元素的第二个元素为 8.0
        )
        self.assertTrue(scaler._scale.device == t1.device)  # 断言梯度缩放器的缩放比例所在的设备与 t1 的设备相同



    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    def test_cuda_device_memory_allocated(self):
        from torch.cuda import memory_allocated  # 导入 memory_allocated 函数

        device_count = torch.cuda.device_count()  # 获取 CUDA 设备的数量
        current_alloc = [memory_allocated(idx) for idx in range(device_count)]  # 获取每个设备当前的内存分配量
        x = torch.ones(10, device="cuda:0")  # 创建一个包含 10 个元素的张量 x，指定在 cuda:0 设备上
        self.assertGreater(memory_allocated(0), current_alloc[0])  # 断言 cuda:0 设备的内存分配量大于初始时的分配量
        self.assertTrue(
            all(
                memory_allocated(torch.cuda.device(idx)) == current_alloc[idx]  # 断言每个 CUDA 设备的内存分配量与初始时的分配量相同
                for idx in range(1, device_count)
            )
        )
    # 定义一个测试方法，用于测试数据广播在多GPU上的行为
    def _test_broadcast(self, input):
        # 如果未启用多GPU测试选项，跳过测试并抛出跳过测试的异常
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        
        # 测试正常广播情况
        results = comm.broadcast(input, (0, 1))
        # 遍历广播结果，检查每个张量的设备编号和数据内容是否正确
        for i, t in enumerate(results):
            self.assertEqual(t.get_device(), i)
            self.assertEqual(t, input)
            # 如果输入张量在GPU上，并且设备编号与当前迭代器中的索引相同，验证广播结果未进行复制
            if input.is_cuda and input.get_device() == i:
                self.assertEqual(t.data_ptr(), input.data_ptr())
        
        # 测试使用out参数的广播
        for inplace in [True, False]:
            if inplace:
                # 如果是原地操作，创建两个空张量，分别在设备0和设备1上
                outputs = [
                    torch.empty_like(input, device=0),
                    torch.empty_like(input, device=1),
                ]
            else:
                # 如果不是原地操作，将输入张量移到设备0，创建另一个空张量在设备1上
                outputs = [input.cuda(0), torch.empty_like(input, device=1)]
            
            # 进行广播，将结果存入预先创建的输出张量中
            results = comm.broadcast(input, out=outputs)
            # 遍历结果和预期输出，验证结果张量和预期输出张量引用相同的对象
            for r, o in zip(results, outputs):
                self.assertIs(r, o)
            # 再次遍历结果，验证每个张量的设备编号和数据内容是否正确
            for i, t in enumerate(results):
                self.assertEqual(t.get_device(), i)
                self.assertEqual(t, input)
        
        # 测试错误消息的情况
        # 验证如果同时指定了'devices'和'out'参数，则抛出运行时异常
        with self.assertRaisesRegex(
            RuntimeError, r"Exactly one of 'devices' and 'out'"
        ):
            comm.broadcast(input, (0, 1), out=outputs)
        
        # 验证如果输出张量列表中包含非CUDA张量，则抛出运行时异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to be CUDA tensors, but output tensor at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cpu()])
        
        # 验证如果输出张量列表中的张量形状与输入张量不匹配，则抛出运行时异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all output tensors to have same shape as the source .+ at index 1",
        ):
            comm.broadcast(input, out=[input.cuda(0), input.cuda(1).unsqueeze(0)])
    
    # 测试方法：测试在CPU上的数据广播行为
    def test_broadcast_cpu(self):
        self._test_broadcast(torch.randn(5, 5))
    
    # 测试方法：测试在GPU上的数据广播行为
    def test_broadcast_gpu(self):
        self._test_broadcast(torch.randn(5, 5).cuda())
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 如果只检测到一个GPU，则跳过测试
    def test_broadcast_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            # 生成稀疏张量，形状为(2, 3)，密度为2，偏移为1，不在CPU上，类型为torch.float64
            self.genSparseTensor((2, 3), 2, 1, False, "cuda", torch.float64)[0],
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成形状为(numel,)的随机张量，在GPU上
            torch.randn(numel).cuda(),
            # 生成稀疏张量，形状为(2, 3)，密度为2，偏移为10，不在CPU上，类型为torch.float64
            self.genSparseTensor((2, 3), 2, 10, False, "cuda", torch.float64)[0],
            # 生成稀疏张量，形状为(2, 3)，密度为2，偏移为5，不在CPU上，类型为torch.float64
            self.genSparseTensor((2, 3), 2, 5, False, "cuda", torch.float64)[0],
            # 生成稀疏张量，形状为(3, 3)，密度为2，偏移为7，不在CPU上，类型为torch.int64
            self.genSparseTensor((3, 3), 2, 7, False, "cuda", torch.int64)[0],
            # 生成稀疏张量，形状为(2, 3)，密度为2，偏移为2，不在CPU上，类型为torch.float32
            self.genSparseTensor((2, 3), 2, 2, False, "cuda", torch.float32)[0],
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成稀疏张量，形状为(2, 7)，密度为2，偏移为3，不在CPU上，类型为torch.int64
            self.genSparseTensor((2, 7), 2, 3, False, "cuda", torch.int64)[0],
            # 生成形状为(numel*2,)的随机整型张量，在GPU上
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            # 生成形状为(numel,)的随机张量，在GPU上
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 如果只检测到一个GPU，则跳过测试
    def test_broadcast_coalesced_dense_only(self):
        numel = 5
        num_bytes = numel * 8
        tensors = [
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成形状为(numel,)的随机张量，在GPU上
            torch.randn(numel).cuda(),
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成形状为(numel,)的随机长整型张量，在GPU上
            torch.randn(numel).long().cuda(),
            # 生成形状为(numel*2,)的随机整型张量，在GPU上
            torch.randn(numel * 2).int().cuda(),  # int is 2x shorter
            # 生成形状为(numel,)的随机张量，在GPU上
            torch.randn(numel).cuda(),
        ]
        self._test_broadcast_coalesced(tensors, num_bytes * 5 // 2)
    def test_broadcast_coalesced_empty_tensors(self):
        # 创建一个空的字节类型张量，并将其移动到 CUDA 设备
        tensors = [
            torch.tensor([]).byte().cuda(),
            # 创建一个包含5个随机数的张量，并将其移动到 CUDA 设备
            torch.randn(5).cuda(),
            # 创建一个包含5个随机数的双精度张量，并将其移动到 CUDA 设备
            torch.randn(5).double().cuda(),
        ]
        # 调用 _test_broadcast_coalesced 函数，传入张量列表和缓冲区大小256
        self._test_broadcast_coalesced(tensors, 256)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add(self):
        # 创建一个大小为5x5的随机张量
        x = torch.randn(5, 5)
        # 创建另一个大小为5x5的随机张量
        y = torch.randn(5, 5)
        # 将 x 移动到 CUDA 设备0
        x_cuda = x.cuda(0)
        # 将 y 移动到 CUDA 设备1
        y_cuda = y.cuda(1)
        # 在通信模块中对 x_cuda 和 y_cuda 执行 reduce_add 操作
        result = comm.reduce_add((x_cuda, y_cuda))
        # 断言结果张量的设备是 CUDA 设备0
        self.assertEqual(result.get_device(), 0)
        # 断言结果张量在 CPU 上的值与 x + y 相等
        self.assertEqual(result.cpu(), x + y)

    def _test_reduce_add_coalesced(self, tensors, buffer_size):
        # 复制张量列表，每个张量都在 CUDA 设备1上
        dup_tensors = [tensors, [t.cuda(1) for t in tensors]]

        # 对每个张量对执行 reduce_add 操作，并将结果存储在 r_tensors 中
        r_tensors = [comm.reduce_add(t) for t in zip(*dup_tensors)]
        # 遍历 r_tensors 和 tensors，逐一断言它们的类型和值
        for r, t in zip(r_tensors, tensors):
            self.assertEqualTypeString(r, t)
            self.assertEqual(r.coalesce() if r.is_sparse else r, t * 2)

        # 在通信模块中执行 reduce_add_coalesced 操作，传入复制的张量列表和缓冲区大小
        rc_tensors = comm.reduce_add_coalesced(dup_tensors, buffer_size=buffer_size)
        # 断言 r_tensors 和 rc_tensors 相等
        self.assertEqual(r_tensors, rc_tensors)
        # 遍历 r_tensors 和 rc_tensors，逐一断言它们的类型相同
        for r, rc in zip(r_tensors, rc_tensors):
            self.assertEqualTypeString(rc, r)

        # 由于输入张量分别在 cuda:0 和 cuda:1 上，输出张量必须是新的。
        # 我们可以检查它们是否具有不同的版本计数器。
        # 注意 [ comm.*_coalesced 中的版本计数器 ]
        versions = [t._version for t in rc_tensors]
        # 遍历版本和张量 rc_tensors，逐一断言每个张量的版本与旧版本相同
        for old_version, t in zip(versions, rc_tensors):
            self.assertEqual(t._version, old_version)
            # 将张量 t 置零后，断言其版本比旧版本大1
            t.zero_()
            self.assertEqual(t._version, old_version + 1)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_reduce_add_coalesced(self):
        numel = 5
        num_bytes = numel * 8
        # 创建包含各种稀疏和密集张量的列表，全部移动到 CUDA 设备
        tensors = [
            self.genSparseTensor((2, 3), 2, 1, False, "cuda", torch.float64)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).cuda(),
            self.genSparseTensor((2, 3), 2, 10, False, "cuda", torch.float64)[0],
            self.genSparseTensor((2, 3), 2, 5, False, "cuda", torch.float64)[0],
            self.genSparseTensor((3, 3), 2, 7, False, "cuda", torch.int64)[0],
            self.genSparseTensor((2, 3), 2, 2, False, "cuda", torch.float32)[0],
            torch.randn(numel).long().cuda(),
            torch.randn(numel).long().cuda(),
            self.genSparseTensor((2, 7), 2, 3, False, "cuda", torch.int64)[0],
            torch.randn(numel * 2).int().cuda(),  # int 类型的张量长度是原来的一半
            torch.randn(numel).cuda(),
        ]
        # 调用 _test_reduce_add_coalesced 函数，传入张量列表和缓冲区大小
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 测试方法：测试 reduce_add_coalesced 函数，用于处理稠密张量的加和操作
    def test_reduce_add_coalesced_dense_only(self):
        # 设置张量中元素的数量
        numel = 5
        # 计算总字节数，每个元素占8个字节
        num_bytes = numel * 8
        # 创建包含不同类型和设备的张量列表
        tensors = [
            torch.randn(numel).long().cuda(),        # 长整型张量在 GPU 上
            torch.randn(numel).cuda(),               # 浮点型张量在 GPU 上
            torch.randn(numel).long().cuda(),        # 长整型张量在 GPU 上
            torch.randn(numel).long().cuda(),        # 长整型张量在 GPU 上
            torch.randn(numel * 2).int().cuda(),     # 整型张量在 GPU 上，长度是原始长度的两倍
            torch.randn(numel).cuda(),               # 浮点型张量在 GPU 上
        ]
        # 调用 _test_reduce_add_coalesced 方法，对张量进行加和操作，期望的总字节数是原来的5/2倍
        self._test_reduce_add_coalesced(tensors, num_bytes * 5 // 2)
    
    # 测试方法：测试 scatter 在 CPU 上的使用
    def test_scatter_cpu(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，沿着 dim=0
        self._test_scatter(torch.randn(4, 4), dim=0)
    
    # 测试方法：测试 scatter 在 CPU 上指定维度的使用
    def test_scatter_cpu_dim(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，沿着 dim=1
        self._test_scatter(torch.randn(4, 4), dim=1)
    
    # 测试方法：测试 scatter 在 CPU 上负维度的使用
    def test_scatter_cpu_neg_dim(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，沿着 dim=-2
        self._test_scatter(torch.randn(4, 4), dim=-2)
    
    # 测试方法：测试 scatter 在 CPU 上指定 chunk_sizes 的使用
    def test_scatter_cpu_sizes(self):
        # 调用 _test_scatter 方法，对一个6x4的随机张量进行 scatter 操作，指定 chunk_sizes=(2, 4)
        self._test_scatter(torch.randn(6, 4), chunk_sizes=(2, 4))
    
    # 测试方法：测试 scatter 在 GPU 上的使用
    def test_scatter_gpu(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，张量在 GPU 上，沿着 dim=0
        self._test_scatter(torch.randn(4, 4).cuda(), dim=0)
    
    # 测试方法：测试 scatter 在 GPU 上指定维度的使用
    def test_scatter_gpu_dim(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，张量在 GPU 上，沿着 dim=1
        self._test_scatter(torch.randn(4, 4).cuda(), dim=1)
    
    # 测试方法：测试 scatter 在 GPU 上负维度的使用
    def test_scatter_gpu_neg_dim(self):
        # 调用 _test_scatter 方法，对一个4x4的随机张量进行 scatter 操作，张量在 GPU 上，沿着 dim=-2
        self._test_scatter(torch.randn(4, 4).cuda(), dim=-2)
    
    # 测试方法：测试 scatter 在 GPU 上指定 chunk_sizes 的使用
    def test_scatter_gpu_sizes(self):
        # 调用 _test_scatter 方法，对一个6x4的随机张量进行 scatter 操作，张量在 GPU 上，指定 chunk_sizes=(2, 4)
        self._test_scatter(torch.randn(6, 4).cuda(), chunk_sizes=(2, 4))
    # 定义一个私有方法 _test_gather，用于测试 gather 函数在不同维度下的行为
    def _test_gather(self, dim):
        # 如果未开启多 GPU 测试，则跳过测试，抛出异常
        if not TEST_MULTIGPU:
            raise unittest.SkipTest("only one GPU detected")
        # 在 GPU 0 上生成一个随机张量 x
        x = torch.randn(2, 5, device=0)
        # 在 GPU 1 上生成一个随机张量 y
        y = torch.randn(2, 5, device=1)
        # 计算预期的输出尺寸，沿指定维度 dim 增加 y 的尺寸
        expected_size = list(x.size())
        expected_size[dim] += y.size(dim)
        expected_size = torch.Size(expected_size)

        # 设置目标设备列表，包括 None、cuda:0 和 cpu，如果有多于两个 CUDA 设备则加入 cuda:2
        destinations = [None, torch.device("cuda:0"), torch.device("cpu")]
        if torch.cuda.device_count() > 2:
            destinations.append(torch.device("cuda:2"))
        # 使用 GPU 1 进入上下文环境，测试 gather 函数的行为
        with torch.cuda.device(1):
            for destination in destinations:
                # 根据 destination 确定预期的设备
                if destination is None:
                    expected_device = torch.device("cuda", torch.cuda.current_device())
                else:
                    expected_device = destination
                # 遍历 use_out 的选项，测试 gather 函数的输出
                for use_out in [True, False]:
                    if use_out:
                        # 如果 use_out 为 True，则使用预先创建的 out 张量作为输出
                        out = torch.empty(expected_size, device=expected_device)
                        result = comm.gather((x, y), dim, out=out)
                        self.assertIs(out, result)  # 断言 out 与 result 是同一个对象
                    else:
                        # 如果 use_out 为 False，则根据 destination 参数进行 gather 操作
                        result = comm.gather((x, y), dim, destination=destination)

                    # 断言 gather 结果的设备与预期一致
                    self.assertEqual(result.device, expected_device)
                    # 断言 gather 结果的尺寸与预期一致
                    self.assertEqual(result.size(), expected_size)

                    # 根据 index 切片获取 gather 结果中的子张量，与 x、y 进行比较
                    index = [slice(None, None), slice(None, None)]
                    index[dim] = slice(0, x.size(dim))
                    self.assertEqual(result[tuple(index)], x)
                    index[dim] = slice(x.size(dim), x.size(dim) + y.size(dim))
                    self.assertEqual(result[tuple(index)], y)

        # 测试错误消息是否正确抛出
        with self.assertRaisesRegex(
            RuntimeError, r"'destination' must not be specified"
        ):
            # 测试在指定 destination 时是否抛出异常
            comm.gather(
                (x, y),
                dim,
                destination="cpu",
                out=torch.empty(expected_size, device="cpu"),
            )
        with self.assertRaisesRegex(
            RuntimeError, r"Expected at least one tensor to gather from"
        ):
            # 测试 gather 参数为空时是否抛出异常
            comm.gather(())
        with self.assertRaisesRegex(
            RuntimeError, r"Expected all input tensors to be CUDA tensors, "
        ):
            # 测试输入张量不全为 CUDA 张量时是否抛出异常
            comm.gather((x.cpu(), y))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected all input tensors to have the same number of dimensions",
        ):
            # 测试输入张量维度不一致时是否抛出异常
            comm.gather((x, y.unsqueeze(0)))
        with self.assertRaisesRegex(
            RuntimeError, r"Input tensor at index 1 has invalid shape"
        ):
            # 测试输入张量形状不合法时是否抛出异常
            if dim in [0, -2]:
                comm.gather((x, y[:, 1:]), dim=dim)
            elif dim in [1, -1]:
                comm.gather((x, y[1:, :]), dim=dim)

    # 定义测试方法 test_gather，测试 gather 在维度 0 下的行为
    def test_gather(self):
        self._test_gather(0)

    # 定义测试方法 test_gather_dim，测试 gather 在维度 1 下的行为
    def test_gather_dim(self):
        self._test_gather(1)

    # 定义测试方法 test_gather_neg_dim，测试 gather 在负维度下的行为
    def test_gather_neg_dim(self):
        self._test_gather(-1)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    装饰器，用于跳过测试，如果不满足 TEST_MULTIGPU 条件，则说明只检测到一个 GPU。

        def test_memory_format_scatter_gather(self):
            定义测试方法：测试内存格式的 scatter 和 gather 操作。

            nhwc = torch.randn((10, 3, 32, 32), device="cpu").contiguous(
                memory_format=torch.channels_last
            )
            创建一个形状为 (10, 3, 32, 32) 的张量 nhwc，确保其内存格式为通道优先。

            results = torch.cuda.comm.scatter(nhwc, (0, 1), None, 0)
            使用 torch.cuda.comm.scatter 函数将 nhwc 张量在 GPU 0 和 1 上分散，并返回结果。

            for result in results:
                遍历分散结果中的每个张量 result。

                self.assertFalse(result.is_contiguous())
                断言 result 不是连续的张量。

                self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))
                断言 result 的内存格式为通道优先。

            gathered = torch.cuda.comm.gather(results)
            使用 torch.cuda.comm.gather 函数将分散的结果 gathered 聚集回来。

            self.assertTrue(gathered.is_contiguous(memory_format=torch.channels_last))
            断言 gathered 张量的内存格式为通道优先。

    @unittest.skipIf(not TEST_MULTIGPU, "Test needs multiple GPUs")
    装饰器，用于跳过测试，如果不满足 TEST_MULTIGPU 条件，则说明测试需要多个 GPU。

        def test_scatter_namedtuple(self):
            定义测试方法：测试能够分散命名元组并获取每个元素为预期命名元组类型的列表。

                fields = ("a", "b")
                定义命名元组的字段名为 "a" 和 "b"。

                TestNamedTupleInput_0 = collections.namedtuple("NamedTuple", fields)
                创建一个名为 TestNamedTupleInput_0 的命名元组类，字段为 "a" 和 "b"。

                num_gpus = torch.cuda.device_count()
                获取当前系统中的 GPU 数量。

                a = torch.rand(num_gpus * 2, device=0)
                创建一个在 GPU 0 上的形状为 num_gpus * 2 的随机张量 a。

                b = torch.rand(num_gpus * 2, device=0)
                创建一个在 GPU 0 上的形状为 num_gpus * 2 的随机张量 b。

                a_tensors_for_gpu = [a[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
                将张量 a 切片成 num_gpus 份，每份长度为 2，并将每份移到对应的 GPU 上。

                b_tensors_for_gpu = [b[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
                将张量 b 切片成 num_gpus 份，每份长度为 2，并将每份移到对应的 GPU 上。

                inp = TestNamedTupleInput_0(a, b)
                创建一个 TestNamedTupleInput_0 的实例 inp，参数为 a 和 b。

                target_gpus = [torch.device(i) for i in range(num_gpus)]
                创建一个包含 num_gpus 个 GPU 设备的列表 target_gpus。

                scatter_out = scatter_gather.scatter(inp, target_gpus)
                使用 scatter_gather.scatter 函数将 inp 分散到 target_gpus 上。

                for i, x in enumerate(scatter_out):
                    遍历 scatter_out 中的每个元素 x 和它们的索引 i。

                    self.assertTrue(isinstance(x, type(inp)))
                    断言 x 是 inp 类型的实例。

                    self.assertEqual(x._fields, fields)
                    断言 x 的字段与 fields 相同。

                    expected_a = a_tensors_for_gpu[i]
                    获取预期在第 i 个 GPU 上的张量 a。

                    expected_b = b_tensors_for_gpu[i]
                    获取预期在第 i 个 GPU 上的张量 b。

                    self.assertEqual(expected_a, x.a)
                    断言 x 的字段 a 与预期的张量 expected_a 相同。

                    self.assertEqual(expected_b, x.b)
                    断言 x 的字段 b 与预期的张量 expected_b 相同。

                class TestNamedTupleInput_1(NamedTuple):
                    定义一个新的命名元组类 TestNamedTupleInput_1，字段为 a 和 b。

                    a = torch.rand(num_gpus * 2, device=0)
                    创建一个在 GPU 0 上的形状为 num_gpus * 2 的随机张量 a。

                    b = torch.rand(num_gpus * 2, device=0)
                    创建一个在 GPU 0 上的形状为 num_gpus * 2 的随机张量 b。

                    a_tensors_for_gpu = [a[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
                    将张量 a 切片成 num_gpus 份，每份长度为 2，并将每份移到对应的 GPU 上。

                    b_tensors_for_gpu = [b[2 * i : 2 * i + 2].to(i) for i in range(num_gpus)]
                    将张量 b 切片成 num_gpus 份，每份长度为 2，并将每份移到对应的 GPU 上。

                    inp = TestNamedTupleInput_1(a, b)
                    创建一个 TestNamedTupleInput_1 的实例 inp，参数为 a 和 b。

                    scatter_out = scatter_gather.scatter(inp, target_gpus)
                    使用 scatter_gather.scatter 函数将 inp 分散到 target_gpus 上。

                    for i, x in enumerate(scatter_out):
                        遍历 scatter_out 中的每个元素 x 和它们的索引 i。

                        self.assertTrue(isinstance(x, type(inp)))
                        断言 x 是 inp 类型的实例。

                        self.assertEqual(x._fields, fields)
                        断言 x 的字段与 fields 相同。

                        expected_a = a_tensors_for_gpu[i]
                        获取预期在第 i 个 GPU 上的张量 a。

                        expected_b = b_tensors_for_gpu[i]
                        获取预期在第 i 个 GPU 上的张量 b。

                        self.assertEqual(expected_a, x.a)
                        断言 x 的字段 a 与预期的张量 expected_a 相同。

                        self.assertEqual(expected_b, x.b)
                        断言 x 的字段 b 与预期的张量 expected_b 相同。
    def test_gather_namedtuple(self):
        # 测试能否收集命名元组列表，并返回每个元素为预期张量类型的命名元组。
        fields = ["a", "b"]
        # 创建名为 TestNamedTupleInput_0 的命名元组类，具有字段 'a' 和 'b'
        TestNamedTupleInput_0 = collections.namedtuple("NamedTuple", fields)

        # 获取当前系统中可用的 GPU 数量
        num_gpus = torch.cuda.device_count()
        # 在设备 0 上生成大小为 num_gpus * 2 的随机张量 a
        a = torch.rand(num_gpus * 2, device=0)
        # 在设备 1 上生成大小为 num_gpus * 2 的随机张量 b
        b = torch.rand(num_gpus * 2, device=1)
        # 创建命名元组 TestNamedTupleInput_0 的实例 out1
        out1 = TestNamedTupleInput_0(a, b)

        # 在设备 1 上生成大小为 num_gpus * 2 的随机张量 a
        a = torch.rand(num_gpus * 2, device=1)
        # 在设备 0 上生成大小为 num_gpus * 2 的随机张量 b
        b = torch.rand(num_gpus * 2, device=0)
        # 创建命名元组 TestNamedTupleInput_0 的另一个实例 out2
        out2 = TestNamedTupleInput_0(a, b)

        # 将 out1 和 out2 放入列表 outputs 中
        outputs = [out1, out2]

        # 调用 scatter_gather.gather 函数，在 CPU 上进行测试
        out = scatter_gather.gather(outputs, "cpu")
        # 遍历输出 out 的索引和元素 x
        for i, x in enumerate(out):
            # 断言 x 是 out2 的最后一个元素的类型的实例，即张量
            self.assertTrue(isinstance(x, type(out2[-1])))
            # 将 outputs[0][i] 和 outputs[1][i] 转移到 CPU，并拼接成 cat
            cat = torch.cat((outputs[0][i].to("cpu"), outputs[1][i].to("cpu")))
            # 断言 x 和 cat 相等
            self.assertTrue(torch.equal(x, cat))

        # 调用 scatter_gather.gather 函数，在 GPU 0 上进行测试
        out = scatter_gather.gather(outputs, 0)
        # 遍历输出 out 的索引和元素 x
        for i, x in enumerate(out):
            # 断言 x 是 out2 的最后一个元素的类型的实例，即张量
            self.assertTrue(isinstance(x, type(out2[-1])))
            # 将 outputs[0][i] 和 outputs[1][i] 转移到 GPU 0，并拼接成 cat
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            # 断言 x 和 cat 相等
            self.assertTrue(torch.equal(x, cat))

        # 定义一个名为 TestNamedTupleInput_1 的命名元组类，字段为 a 和 b，类型为 torch.tensor
        class TestNamedTupleInput_1(NamedTuple):
            a: torch.tensor
            b: torch.tensor

        # 在设备 0 上生成大小为 num_gpus * 2 的随机张量 a
        a = torch.rand(num_gpus * 2, device=0)
        # 在设备 1 上生成大小为 num_gpus * 2 的随机张量 b
        b = torch.rand(num_gpus * 2, device=1)
        # 创建命名元组 TestNamedTupleInput_1 的实例 out1
        out1 = TestNamedTupleInput_1(a, b)

        # 在设备 1 上生成大小为 num_gpus * 2 的随机张量 a
        a = torch.rand(num_gpus * 2, device=1)
        # 在设备 0 上生成大小为 num_gpus * 2 的随机张量 b
        b = torch.rand(num_gpus * 2, device=0)
        # 创建命名元组 TestNamedTupleInput_1 的另一个实例 out2
        out2 = TestNamedTupleInput_1(a, b)

        # 将 out1 和 out2 放入列表 outputs 中
        outputs = [out1, out2]

        # 调用 scatter_gather.gather 函数，在 GPU 0 上进行测试
        out = scatter_gather.gather(outputs, 0)
        # 遍历输出 out 的索引和元素 x
        for i, x in enumerate(out):
            # 断言 x 是 out2 的最后一个元素的类型的实例，即张量
            self.assertTrue(isinstance(x, type(out2[-1])))
            # 将 outputs[0][i] 和 outputs[1][i] 转移到 GPU 0，并拼接成 cat
            cat = torch.cat((outputs[0][i].to(0), outputs[1][i].to(0)))
            # 断言 x 和 cat 相等
            self.assertTrue(torch.equal(x, cat))

        # 调用 scatter_gather.gather 函数，在 CPU 上进行测试
        out = scatter_gather.gather(outputs, "cpu")
        # 遍历输出 out 的索引和元素 x
        for i, x in enumerate(out):
            # 断言 x 是 out2 的最后一个元素的类型的实例，即张量
            self.assertTrue(isinstance(x, type(out2[-1])))
            # 将 outputs[0][i] 和 outputs[1][i] 转移到 CPU，并拼接成 cat
            cat = torch.cat((outputs[0][i].to("cpu"), outputs[1][i].to("cpu")))
            # 断言 x 和 cat 相等
            self.assertTrue(torch.equal(x, cat))
# 实例化参数化测试类 TestCudaMultiGPU，以便后续执行多个带参数的测试
instantiate_parametrized_tests(TestCudaMultiGPU)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    # 运行测试函数，通常用于执行单元测试或集成测试
    run_tests()
```