# `.\pytorch\test\profiler\test_profiler.py`

```py
# Owner(s): ["oncall: profiler"]

# 导入所需的标准库和第三方库
import collections  # 导入 collections 库
import gc  # 导入垃圾回收模块
import json  # 导入 JSON 解析模块
import mmap  # 导入内存映射模块
import os  # 导入操作系统功能模块
import pickle  # 导入 pickle 序列化模块
import random  # 导入随机数生成模块
import re  # 导入正则表达式模块
import struct  # 导入处理结构体数据的模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关功能模块
import threading  # 导入多线程模块
import time  # 导入时间处理模块
import unittest  # 导入单元测试框架
from dataclasses import dataclass, field  # 导入数据类相关功能
from typing import List, Optional  # 导入类型提示功能

import expecttest  # 导入自定义的测试模块

import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入神经网络模块
import torch.optim  # 导入优化器模块
import torch.utils.data  # 导入数据加载与处理模块
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall  # 导入 C++ 扩展模块
from torch.autograd.profiler import KinetoStepTracker, profile as _profile  # 导入性能分析模块
from torch.autograd.profiler_legacy import profile as _profile_legacy  # 导入旧版性能分析模块
from torch.profiler import (  # 导入新版性能分析模块相关功能
    _utils,
    DeviceType,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.profiler._pattern_matcher import (  # 导入性能分析模式匹配相关功能
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入多 GPU 测试相关功能
from torch.testing._internal.common_device_type import skipCUDAVersionIn  # 导入 CUDA 版本检查装饰器
from torch.testing._internal.common_utils import (  # 导入常用测试工具函数
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_JETSON,
    IS_LINUX,
    IS_WINDOWS,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_WITH_ASAN,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
)

# 如果 tqdm 没有正确关闭，会使监视线程保持活动状态。
# 这会导致多线程测试中的问题，因为我们检查所有事件及其线程 ID。
# 那些对应于这些残留线程的事件都有一个无效的 TID (uint64_t)(-1)。
# 解决方法是在加载 tqdm 时关闭监视线程。
# 由于这些是单元测试，关闭监视线程是安全的。
try:
    import tqdm
    # 设置 tqdm 的监视间隔为 0，关闭监视线程
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

# 尝试导入 psutil 库，标记是否成功导入
try:
    import psutil
    HAS_PSUTIL = True
except ModuleNotFoundError:
    # 如果未找到 psutil 模块，则标记为未成功导入
    HAS_PSUTIL = False
    psutil = None

# 使用 psutil 的条件下，跳过测试（需要 psutil）
# 不能在 ASAN 下测试
# 在 Windows 上测试不稳定
# CUDA 不可用时跳过测试
@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):
    @skipCUDAVersionIn([(11, 5)])  # 根据 CUDA 版本跳过特定版本的测试
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with CUDA"""
        # 创建一个大小为 1x1 的随机张量，并将其移到 CUDA 设备上
        t = torch.rand(1, 1).cuda()
        # 获取当前进程的信息
        p = psutil.Process()
        # 创建一个大小为 5 的固定长度双端队列，用于存储最近的 5 个内存使用量数据
        last_rss = collections.deque(maxlen=5)
        # 外层循环，迭代 10 次
        for outer_idx in range(10):
            # 使用 _profile 上下文管理器，开启 CUDA 支持的性能分析
            with _profile(use_cuda=True):
                # 内层循环，迭代 1024 次，进行张量的矩阵乘法操作
                for _ in range(1024):
                    t = torch.mm(t, t)

            # 手动触发垃圾回收
            gc.collect()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 获取当前进程的 RSS（Resident Set Size，常驻内存集）并加入队列末尾
            last_rss.append(p.memory_info().rss)

        # 检查队列中的内存使用量是否递增
        is_increasing = all(
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))
        )
        # 计算队列中相邻两元素的最大差值
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        # 使用断言验证内存使用量未递增，并且最大差值未超过 100MB
        self.assertTrue(
            not (is_increasing and max_diff > 100 * 1024),
            msg=f"memory usage is increasing, {str(last_rss)}",
        )

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # 仅测试当 record_shapes 选项启用时，emit_nvtx 是否运行
        with torch.autograd.profiler.emit_nvtx(record_shapes=True) as prof:
            # 创建两个形状为 10x10 的张量，并开启梯度跟踪
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            # 执行张量的加法运算
            z = x + y
            # 调用自定义层 custom_layer 进行操作
            s = custom_layer(z)
            # 计算张量 s 的和
            q = s.sum()
            # 反向传播计算梯度
            q.backward()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_cudagraph_profiling_workaround(self):
        import subprocess

        # 从 #75504 复制的复现场景
        # 在单独的进程中执行，以捕获挂起/非法内存错误，并确保 CUPTI 未初始化
        p = subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
import os  # 导入 os 模块
import torch  # 导入 PyTorch 库
from torch.profiler import ProfilerActivity, profile  # 从 torch.profiler 模块导入 ProfilerActivity 和 profile 函数

def add_one(in_: torch.Tensor):
    return in_ + 1  # 返回输入张量每个元素加一的结果

sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)  # 创建一个在 GPU 上的全零张量，并要求梯度

# 在创建 CUDA 图之前初始化
torch.profiler._utils._init_for_cuda_graphs()

# 使用 torch.cuda.graphs.make_graphed_callables 创建被图形化调用的 add_one 函数
add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))
zeros = torch.zeros(10, device="cuda")  # 创建一个在 GPU 上的全零张量
out = add_one_graphed(zeros)  # 对全零张量执行图形化调用的 add_one 操作
assert out[0] == 1  # 断言输出张量的第一个元素是否为 1

# 使用 CPU 作为活动进行性能分析
with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

# 使用 CUDA 作为活动进行性能分析
with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode"""
        # 避免自动内联优化
        prev_opt = torch._C._get_graph_executor_optimize()
        # 设置图执行器的优化为关闭状态
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            # 使用 TorchScript 注解定义函数，执行矩阵乘法操作
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            # 使用 TorchScript 注解定义函数，计算 a，并调用 ts_method_2 计算 w
            a = x + z
            w = ts_method_2(x, y) + a
            # 返回 w 的元素之和
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个卷积层对象
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )

            def forward(self, x):
                # 在卷积层上执行前向传播
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            # 调用模块的 forward 方法
            return mod(x)

        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            # 创建需要梯度的随机张量 x 和 y
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            # 计算 z 作为 x 和 y 的和
            z = x + y
            # 调用 ts_method_1 函数计算 w
            w = ts_method_1(x, y, z)
            # 计算 v 作为 w 的两倍
            v = 2 * w
            # 对 v 进行反向传播
            v.backward()
            # 创建需要梯度的随机张量 a
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            # 调用 call_module 方法传入 a
            b = call_module(a)
            # 计算 b 的元素之和
            c = b.sum()
            # 对 c 进行反向传播
            c.backward()

        # 遍历性能分析对象的函数事件列表
        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                # 断言栈中至少有一个包含 "test_profiler" 的条目
                self.assertTrue(any("test_profiler" in entry for entry in e.stack))
                # 断言栈中至少有一个包含 "test_source"、"ts_method_1" 或 "ts_method_2" 的条目
                self.assertTrue(
                    any(
                        (
                            "test_source" in entry
                            or "ts_method_1" in entry
                            or "ts_method_2" in entry
                        )
                        for entry in e.stack
                    )
                )

        # TODO: https://github.com/pytorch/kineto/issues/617
        # 如果支持 kineto 且不是在 Windows 环境下
        if kineto_available() and not IS_WINDOWS:
            # 创建临时文件名
            with TemporaryFileName(mode="w+") as fname:
                # 导出 Chrome 跟踪数据到临时文件
                p.export_chrome_trace(fname)
                # 打开临时文件并加载事件列表
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    # 提取满足指定模式的事件
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    # 断言仅有一个匹配项
                    self.assertEqual(
                        len(matches), 1, repr([e["name"] for e in matches])
                    )
                    return matches[0]

                # 提取模块事件和包装器事件
                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                # 断言模块事件的 Python 父级 ID 与包装器事件的 Python ID 相等
                self.assertEqual(
                    module_event["args"]["Python parent id"],
                    wrapper_event["args"]["Python id"],
                )

        # 恢复原始的图执行器优化设置
        torch._C._set_graph_executor_optimize(prev_opt)
    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),  # 参数化测试名称为'basic'，线程规格为((False, False),)
            "multiple_preexisting": ((False, False),) * 2,  # 参数化测试名称为'multiple_preexisting'，线程规格为((False, False), (False, False))
            "open_in_scope": ((True, False),),  # 参数化测试名称为'open_in_scope'，线程规格为((True, False),)
            "close_in_scope": ((False, True),),  # 参数化测试名称为'close_in_scope'，线程规格为((False, True),)
            "complex": (  # 参数化测试名称为'complex'，包含多个线程规格
                # 大量后台线程
                (False, False),
                (False, False),
                (False, False),
                (False, False),
                # 在分析过程中完成的一些线程
                (False, True),
                (False, True),
                # 被分析部分也是多线程的
                (True, False),
                (True, True),
            ),
        }.items(),
        name_fn=lambda name, thread_spec: name,  # 名称函数定义，返回参数化测试的名称
    )
    @serialTest()  # 序列化测试装饰器
    @parametrize("work_in_main_thread", [True, False])  # 参数化测试，测试在主线程中工作的情况
    def payload(self, use_cuda=False):
        x = torch.randn(10, 10)  # 生成一个10x10的随机张量x
        if use_cuda:
            x = x.cuda()  # 如果使用CUDA，则将张量x移动到GPU上
        y = torch.randn(10, 10)  # 生成一个10x10的随机张量y
        if use_cuda:
            y = y.cuda()  # 如果使用CUDA，则将张量y移动到GPU上
        z = torch.mm(x, y)  # 计算张量x和y的矩阵乘积，得到张量z
        z = z + y  # 将张量z和y相加，结果赋给张量z
        if use_cuda:
            z = z.cpu()  # 如果使用CUDA，则将张量z移动到CPU上

    def _check_stats(self, profiler_stats):
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)  # 断言：分析窗口持续时间大于0秒
        self.assertGreater(profiler_stats.number_of_events, 0)  # 断言：事件数量大于0
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)  # 断言：分析准备调用持续时间大于0微秒
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)  # 断言：分析启用调用持续时间大于0微秒
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)  # 断言：分析禁用调用持续时间大于0微秒
        self.assertGreater(
            profiler_stats.parse_kineto_call_duration_us, 0
        )  # 断言：解析Kineto调用持续时间大于0微秒
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )  # 断言：构建函数事件树调用持续时间大于0微秒

    @unittest.skipIf(not kineto_available(), "Kineto is required")  # 如果Kineto不可用，则跳过测试
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()  # 检查CUDA是否在支持的活动中
        with _profile(use_cuda=use_cuda, use_kineto=True):  # 使用CUDA和Kineto进行性能分析
            self.payload(use_cuda=use_cuda)  # 执行payload函数

        # 重新运行以避免初始启动开销
        with _profile(use_cuda=use_cuda, use_kineto=True) as p:  # 使用CUDA和Kineto进行性能分析，并将结果存储在p中
            self.payload(use_cuda=use_cuda)  # 执行payload函数
        output = p.key_averages().table(  # 生成性能分析结果的表格
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",  # 根据CUDA时间或CPU时间进行排序
            row_limit=-1,  # 行数限制为无限制
        )
        # print(output)  # 打印输出表格
        found_gemm = False  # 初始化标志变量found_gemm为False
        found_memcpy = False  # 初始化标志变量found_memcpy为False
        found_mm = False  # 初始化标志变量found_mm为False
        for e in p.function_events:  # 遍历性能分析的函数事件
            if "aten::mm" in e.name:  # 如果事件名称包含'aten::mm'
                found_mm = True  # 设置found_mm为True
            if "gemm" in e.name or "Cijk" in e.name:  # 如果事件名称包含'gemm'或'Cijk'
                found_gemm = True  # 设置found_gemm为True
            if "Memcpy" in e.name or "memcpy" in e.name:  # 如果事件名称包含'Memcpy'或'memcpy'
                found_memcpy = True  # 设置found_memcpy为True
        if use_cuda:
            self.assertTrue(found_gemm)  # 断言：found_gemm为True
            self.assertTrue(found_memcpy)  # 断言：found_memcpy为True
        else:
            self.assertTrue(found_mm)  # 断言：found_mm为True
        self._check_stats(p._stats)  # 检查性能统计信息
        # p.export_chrome_trace("/tmp/test_trace.json")  # 导出Chrome跟踪数据，注释掉以避免实际导出
    # 如果 Kineto 不可用，则跳过测试
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    # 如果不需要多个 GPU，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    # 如果在 ROCm 平台上则跳过测试，因为不支持
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported on ROCm")
    # 测试 Kineto 多 GPU 功能
    def test_kineto_multigpu(self):
        # 使用 profile 上下文管理器，指定记录 CPU 和 CUDA 的性能数据
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # 针对 GPU 0 和 GPU 1 进行迭代
            for gpu_id in [0, 1]:
                # 创建在指定 GPU 上的随机张量 x
                x = torch.randn(10, 10).cuda(gpu_id)
                # 创建在指定 GPU 上的随机张量 y
                y = torch.randn(10, 10).cuda(gpu_id)
                # 执行矩阵乘法操作，并将结果保存在 z 中
                z = x.matmul(y)

        # 初始化变量用于标记是否找到了特定的事件
        found_gemm_0 = False
        found_gemm_1 = False
        found_cuda = False
        # 遍历记录的所有事件
        for evt in prof.events():
            # 如果事件名称包含 "gemm" 并且是 CUDA 设备类型
            if "gemm" in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                # 根据设备索引标记找到了 gemm 事件
                if evt.device_index == 0:
                    found_gemm_0 = True
                elif evt.device_index == 1:
                    found_gemm_1 = True
            # 如果事件名称包含 "cuda" 并且是 CPU 设备类型
            if "cuda" in evt.name.lower() and evt.device_type == DeviceType.CPU:
                # 标记找到了 CUDA 事件
                found_cuda = True

        # 断言找到了两个 gemm 事件和一个 CUDA 事件
        self.assertTrue(found_gemm_0)
        self.assertTrue(found_gemm_1)
        self.assertTrue(found_cuda)
        # 检查性能统计数据
        self._check_stats(prof._stats())

    # 如果在 Jetson 平台上，则跳过测试，因为内存共享问题可能导致 OOM
    @unittest.skipIf(
        IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared"
    )
    # 测试 OOM 跟踪功能
    def test_oom_tracing(self):
        # 定义运行性能分析器的函数，捕获内存和形状记录
        def run_profiler(tensor_creation_fn):
            with _profile(profile_memory=True, record_shapes=True) as prof:
                # 断言创建张量时会抛出 RuntimeError，提示尝试分配内存失败
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        # 创建在 CUDA 设备上会导致 OOM 的张量
        def create_cuda_tensor_oom():
            device = torch.device("cuda:0")
            return torch.empty(1024, 1024, 1024, 20, dtype=torch.float32, device=device)

        # 检查导出的跟踪文件中是否包含 OOM 事件
        def check_trace(fname):
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                # 断言跟踪文件包含 traceEvents
                self.assertTrue("traceEvents" in trace)
                events = trace["traceEvents"]
                found_out_of_memory_events = False
                for evt in events:
                    self.assertTrue("name" in evt)
                    # 如果事件名称为 "[OutOfMemory]"，则标记找到了 OOM 事件
                    if evt["name"] == "[OutOfMemory]":
                        found_out_of_memory_events = True
                        # 断言事件参数中包含 Device Type、Device Id 和 Bytes
                        self.assertTrue("args" in evt)
                        self.assertTrue("Device Type" in evt["args"])
                        self.assertTrue("Device Id" in evt["args"])
                        self.assertTrue("Bytes" in evt["args"])

                        # 断言这是一个瞬时事件，不应包含 dur 和 cat 参数
                        self.assertTrue("dur" not in evt["args"])
                        self.assertTrue("cat" not in evt["args"])
                # 断言至少找到了一个 OOM 事件
                self.assertTrue(found_out_of_memory_events)

        # 如果 CUDA 可用，则执行以下操作
        if torch.cuda.is_available():
            # 使用临时文件名运行性能分析器和跟踪检查
            with TemporaryFileName(mode="w+") as fname:
                prof = run_profiler(create_cuda_tensor_oom)
                check_trace(fname)
    # 使用 unittest 框架的装饰器，跳过测试如果 kineto 不可用的话
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    # 定义测试方法 test_module_hierarchy
    def test_module_hierarchy(self):
        # 定义模块 A，继承自 nn.Module
        class A(nn.Module):
            # 自定义方法 my_new_method，对输入 x 进行乘法运算
            def my_new_method(self, x):
                return x * 3

            # 实现前向传播方法 forward_impl_
            def forward_impl_(self, x, y):
                # 调用 my_new_method 进行乘法运算，再加上 y 返回结果
                return self.my_new_method(x) + y

            # 标准前向传播方法 forward
            def forward(self, x, y):
                # 对 y 减去 2
                y = y - 2
                # 调用 forward_impl_ 进行前向传播
                return self.forward_impl_(x, y)

        # 定义模块 B，继承自 nn.Module
        class B(nn.Module):
            # 实现前向传播方法 forward
            def forward(self, x):
                # 返回输入 x 加 2 的结果
                return x + 2

        # 定义模块 C，继承自 nn.Module
        class C(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建模块 A 的实例 A0 和模块 B 的实例 B0
                self.A0 = A()
                self.B0 = B()

            # 调用模块 B 的 forward 方法
            def call_b(self, x):
                return self.B0.forward(x)

            # 标准前向传播方法 forward
            def forward(self, x, y):
                # 调用模块 A 的 forward 方法和 call_b 方法，返回它们的和
                return self.A0.forward(x, y) + self.call_b(x)

        # 创建模型实例 model，并将其转换为 TorchScript 形式
        model = C()
        model = torch.jit.script(model)

        # 创建输入数据 input_a 和 input_b
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)

        # 创建空字典 op_to_module_hierarchy，用于存储操作名到模块层次的映射关系
        op_to_module_hierarchy = {}

        # 填充 op_to_module_hierarchy 字典
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."
        ]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.",
            "TOP(C)::forward.",
        ]

        # 使用临时文件名进行 CPU 分析，并导出 Chrome 跟踪文件
        with TemporaryFileName(mode="w+") as fname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_modules=True,
            ) as prof:
                # 执行模型，传入 input_a 和 input_b
                model(input_a, input_b)
            # 将分析结果导出到 Chrome 跟踪文件中
            prof.export_chrome_trace(fname)
            # 打开并加载 Chrome 跟踪文件的 JSON 内容
            with open(fname) as f:
                trace = json.load(f)
                # 断言确保 traceEvents 在 JSON 结构中
                assert "traceEvents" in trace
                events = trace["traceEvents"]
                found_memory_events = False
                # 遍历每个事件对象
                for evt in events:
                    # 断言每个事件对象包含 "name" 属性
                    assert "name" in evt
                    if "args" in evt:
                        # 获取操作名
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            # 获取事件的模块层次信息
                            hierarchy = evt["args"]["Module Hierarchy"]
                            # 断言模块层次信息在预期的 op_to_module_hierarchy 中
                            if op_name in op_to_module_hierarchy:
                                assert hierarchy in op_to_module_hierarchy[op_name]
    # 定义一个测试方法，用于测试模型的浮点操作数（FLOPs）计算
    def test_flops(self):
        # 创建一个包含多个层的神经网络模型
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),  # 二维卷积层，输入通道数为16，输出通道数为33，卷积核大小为18x18
            nn.ReLU(),              # ReLU激活函数层
            nn.Linear(243, 243),    # 全连接层，输入特征数243，输出特征数243
            nn.ReLU(),              # 另一个ReLU激活函数层
        )
        # 创建一个输入张量，形状为[40, 16, 18, 260]
        inputs = torch.randn(40, 16, 18, 260)
        # 创建一个嵌套张量，包含两个不同形状的子张量，使用torch.jagged布局
        nested_tensor = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 5))], layout=torch.jagged
        )
        # 使用_profiler_上下文管理器，记录代码块的性能数据，包括形状记录、FLOPs计算和Kineto使用情况
        with _profile(
            record_shapes=True, with_flops=True, use_kineto=kineto_available()
        ) as prof:
            # 将输入数据传递给模型，进行前向传播
            model(inputs)
            # 测试嵌套张量在计算FLOPs时不会导致异常
            nested_tensor = nested_tensor + nested_tensor
        # 从prof对象获取关键统计数据，并生成按输入形状分组的表格，按CPU时间总量排序，限制显示行数为10
        profiler_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        # 断言输出结果中包含"Total MFLOPs"，用于验证FLOPs计算结果是否正确
        self.assertIn("Total MFLOPs", profiler_output)
        
        # 如果Kineto可用且CUDA可用，则继续执行以下代码块
        if not (kineto_available() and torch.cuda.is_available()):
            return
        
        # 使用profile上下文管理器，记录CPU和CUDA的性能数据，包括形状记录和FLOPs计算
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as kineto_profiler:
            # 将输入数据再次传递给模型，进行前向传播
            model(inputs)
        # 从kineto_profiler对象获取关键统计数据，并生成表格，按CUDA自身总时间排序，显示所有行
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1
        )
        # 断言输出结果中包含"Total MFLOPs"，用于验证FLOPs计算结果是否正确
        self.assertIn("Total MFLOPs", profiler_output)
    # 定义测试函数，测试 Kineto Profiler API
    def test_kineto_profiler_api(self):
        # 记录调用次数的列表，初始为0
        called_num = [0]

        # 检查是否支持 CUDA，并设置使用 CUDA 标志
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()

        # 进行性能分析，记录支持的活动
        with profile(activities=supported_activities()):
            # 调用 payload 函数，传入使用 CUDA 的标志
            self.payload(use_cuda=use_cuda)

        # 定义追踪处理函数，处理性能分析对象 p
        def trace_handler(p):
            # 生成性能分析报告表格，根据 CUDA 或 CPU 时间排序
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
                row_limit=-1,
            )
            # 增加调用次数计数
            called_num[0] += 1
            # 导出 Chrome 追踪文件（注释掉的代码）
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")

        # 获取初始步骤计数
        initial_step = KinetoStepTracker.current_step()

        # 开始性能分析，设置调度策略和追踪处理函数
        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            # 执行8次循环
            for idx in range(8):
                # 调用 payload 函数，传入使用 CUDA 的标志
                self.payload(use_cuda=use_cuda)
                # 执行性能分析的一步
                p.step()

        # 断言调用次数为2
        self.assertEqual(called_num[0], 2)
        # 断言当前步骤数为初始步骤数加上8
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        # 在没有调度策略的情况下进行性能分析
        with profile(activities=supported_activities()) as p:
            # 调用 payload 函数两次，传入使用 CUDA 的标志
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)

        # 生成性能分析报告表格，根据 CUDA 或 CPU 时间排序（注释掉的代码）
        # output = p.key_averages().table(
        #     sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
        #     row_limit=-1,
        # )

        # 定义测试用的调度策略
        test_schedule = torch.profiler.schedule(
            skip_first=2, wait=1, warmup=1, active=2, repeat=2
        )
        # 期望的测试调度策略输出列表
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        # 迭代测试调度策略的每一步
        for step in range(len(test_schedule_expected_outputs)):
            # 断言当前步骤的调度策略与期望的一致
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])
    def test_kineto_profiler_multiple_steppers(self):
        # 定义迭代次数
        niters = 8
        # 检查是否支持 CUDA，用于配置 Profiler 活动
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        # 创建简单神经网络模型
        net = SimpleNet()
        # 使用 SGD 优化器，学习率为 0.01，动量为 0.9
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        # 将梯度清零
        opt.zero_grad()
        # 创建输入数据
        inputs = torch.rand(10)

        # 使用 profile 上下文管理器，配置支持的活动进行性能分析
        with profile(activities=supported_activities()):
            # 调用 self.payload 方法，传入 CUDA 使用情况
            self.payload(use_cuda=use_cuda)

        # 定义 optimizer_step 函数，模拟优化器中的 step() 钩子
        def optimizer_step():
            """This simulates a step() hook in the optimizer"""
            # 调用 KinetoStepTracker 的 increment_step 方法，跟踪步骤
            KinetoStepTracker.increment_step("yet_another_step")

        # 记录初始步骤数
        initial_step = KinetoStepTracker.current_step()

        # 定义 run_batch 函数，执行网络模型的一次前向传播、反向传播和优化器步骤
        def run_batch():
            # 网络模型前向传播
            out = net(inputs)
            # 计算损失
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            # 反向传播
            loss.backward()
            # 执行优化器步骤
            opt.step()
            # 手动调用钩子函数 optimizer_step，用于调试，待优化器类添加相关钩子后移除此处代码
            # 参考 https://github.com/pytorch/pytorch/issues/88446
            optimizer_step()

        # 执行多次迭代运行 run_batch 函数
        for idx in range(niters):
            run_batch()

        # 使用 profile 上下文管理器，配置支持的活动和时间调度，进行性能分析
        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        ) as p:
            # 执行多次迭代运行 run_batch 函数，并在每次迭代后调用性能分析器的 step 方法
            for idx in range(niters):
                run_batch()
                p.step()

        # 断言当前步骤数是否为初始步骤数加上两倍迭代次数
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 2 * niters)

    def test_export_stacks(self):
        # 使用 _profile 上下文管理器，配置带堆栈信息、是否支持 Kineto 和实验配置参数
        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            # 创建随机数据张量 x 和 y，并进行矩阵乘法运算
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            # 张量 z 加上 y
            z = z + y

        # 使用临时文件名上下文管理器，以写入和读取模式打开文件
        with TemporaryFileName(mode="w+") as fname:
            # 将性能分析器 p 的堆栈信息导出到文件 fname
            p.export_stacks(fname)
            # 打开文件 fname，并读取所有行到变量 lines
            with open(fname) as f:
                lines = f.readlines()
            # 断言文件行数大于 0，确保堆栈文件不为空
            assert len(lines) > 0, "Empty stacks file"
            # 遍历文件的每一行
            for line in lines:
                is_int = False
                try:
                    # 尝试将每行最后一个空格分隔的部分转换为整数，检查是否大于 0
                    assert int(line.split(" ")[-1]) > 0, "Invalid stacks record"
                    is_int = True
                except ValueError:
                    pass
                # 断言转换成功为整数，确保记录有效
                assert is_int, "Invalid stacks record"

    # 当 Kineto 不可用时，跳过该测试用例
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    # 定义一个测试方法，用于测试 tensorboard_trace_handler 的功能
    def test_tensorboard_trace_handler(self):
        # 检查是否支持 CUDA，以决定是否使用 CUDA 进行性能分析
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        
        # 使用 _profile 上下文管理器，开启性能分析，使用 CUDA 和 Kineto 进行分析
        with _profile(use_cuda=use_cuda, use_kineto=True):
            # 执行 payload 方法，进行一些操作
            self.payload(use_cuda=use_cuda)

        # 在临时目录中创建 tensorboard_trace_handler 的 profile 上下文
        with TemporaryDirectoryName() as dname:
            # 使用 profile 上下文管理器，配置活动和调度参数，并指定 trace 处理程序为 tensorboard_trace_handler
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname),
            ) as p:
                # 多次执行 payload 方法并调用 p.step() 进行性能数据采集
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            # 断言临时目录 dname 存在，即 tensorboard trace 文件生成成功
            self.assertTrue(os.path.exists(dname))
            
            # 统计文件数量，并逐一验证文件名的格式和后缀
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-3:], ["pt", "trace", "json"])
                file_num += 1
            # 断言文件数量符合预期
            self.assertEqual(file_num, 3)

        # 以 gzip 文件格式进行测试
        with TemporaryDirectoryName() as dname:
            # 创建 profile 上下文管理器，配置活动和调度参数，并指定 trace 处理程序为 tensorboard_trace_handler，并启用 gzip 压缩
            p = profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dname, use_gzip=True
                ),
            )
            # 启动性能分析
            p.start()
            # 多次执行 payload 方法并调用 p.step() 进行性能数据采集
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            # 停止性能分析
            p.stop()

            # 断言临时目录 dname 存在，即 tensorboard trace 文件生成成功
            self.assertTrue(os.path.exists(dname))
            
            # 统计文件数量，并逐一验证文件名的格式和后缀
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-5].isdigit() and int(parts[-5]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-4:], ["pt", "trace", "json", "gz"])
                file_num += 1
            # 断言文件数量符合预期
            self.assertEqual(file_num, 3)

    # 如果 Kineto 不可用，则跳过测试
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    # 定义测试函数，用于测试性能分析器的元数据导出功能
    def test_profiler_metadata(self):
        # 创建两个值为1的张量
        t1, t2 = torch.ones(1), torch.ones(1)
        # 使用性能分析器，记录执行过程中的性能数据
        with profile() as prof:
            # 执行张量加法操作
            torch.add(t1, t2)
            # 添加自定义的元数据 "test_key1"
            prof.add_metadata("test_key1", "test_value1")
            # 添加 JSON 格式的元数据 "test_key2"
            prof.add_metadata_json("test_key2", "[1,2,3]")

        # 使用临时文件名，将性能数据导出为 Chrome 追踪文件
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # 打开导出的文件，加载为 JSON 格式的数据
            with open(fname) as f:
                trace = json.load(f)
                # 断言导出的追踪数据包含 "test_key1" 元数据
                assert "test_key1" in trace
                # 断言 "test_key1" 的值等于 "test_value1"
                assert trace["test_key1"] == "test_value1"
                # 断言导出的追踪数据包含 "test_key2" 元数据
                assert "test_key2" in trace
                # 断言 "test_key2" 的值为 [1, 2, 3]
                assert trace["test_key2"] == [1, 2, 3]

    # 定义私有函数，用于测试性能分析器的追踪功能
    def _test_profiler_tracing(self, use_kineto):
        # 使用性能分析器，记录执行过程中的性能数据
        with _profile(use_kineto=use_kineto) as prof:
            # 创建两个值为1的张量
            t1, t2 = torch.ones(1), torch.ones(1)
            # 执行张量加法操作
            torch.add(t1, t2)

        # 使用临时文件名，将性能数据导出为 Chrome 追踪文件
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # 打开导出的文件，验证其是否为有效的 JSON 格式
            with open(fname) as f:
                json.load(f)

        # 使用性能分析器，记录空的性能数据
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # 使用临时文件名，保存空的性能数据导出为 Chrome 追踪文件

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)

        # 如果支持 CUDA，则进行相同的测试
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        # 在 CUDA 设备上执行性能分析
        device = torch.device("cuda:0")
        with _profile(use_cuda=True, use_kineto=use_kineto) as prof:
            # 创建在 CUDA 设备上的两个值为1的张量
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            # 执行张量加法操作
            torch.add(t1, t2)

        # 使用临时文件名，将 CUDA 上的性能数据导出为 Chrome 追踪文件
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # 打开导出的文件，验证其是否为有效的 JSON 格式
            with open(fname) as f:
                json.load(f)

    # 定义测试函数，用于测试性能分析器的追踪功能
    def test_profiler_tracing(self):
        # 调用 _test_profiler_tracing 函数，传入 False 参数
        self._test_profiler_tracing(False)
        # 如果 Kineto 可用，则调用 _test_profiler_tracing 函数，传入 True 参数
        if kineto_available():
            self._test_profiler_tracing(True)
    # 定义测试函数，用于测试性能分析器的操作事件参数
    def test_profiler_op_event_args(self):
        # 启用具体输入记录功能
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        
        # 使用性能分析器，记录张量形状
        with _profile(record_shapes=True) as prof:
            # 创建一个大小为 (64, 32) 的全1张量
            a = torch.ones((64, 32), dtype=torch.float32)
            # 将两个张量a连接并计算sin函数
            c = torch.cat([a, a]).sin()
        
        # 创建临时文件以导出Chrome跟踪数据
        with TemporaryFileName(mode="w+") as fname:
            # 将性能分析结果导出为Chrome跟踪格式
            prof.export_chrome_trace(fname)
            # 打开临时文件并加载JSON数据
            with open(fname) as f:
                j = json.load(f)
                # 从跟踪事件中筛选出所有CPU操作事件
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                # 遍历每个操作事件
                for e in op_events:
                    args = e["args"]
                    # 如果操作事件名称是"aten::ones"
                    if e["name"] == "aten::ones":
                        # 断言输入类型
                        self.assertEqual(
                            args["Input type"],
                            ["ScalarList", "Scalar", "", "", "Scalar"],
                        )
                        # 断言具体输入
                        self.assertEqual(
                            args["Concrete Inputs"], ["[64, 32]", "6", "", "", "False"]
                        )
                    
                    # 如果操作事件名称是"aten::cat"
                    if e["name"] == "aten::cat":
                        # 断言输入维度
                        self.assertEqual(args["Input Dims"], [[[64, 32], [64, 32]], []])
                        # 断言输入类型
                        self.assertEqual(args["Input type"], ["TensorList", "Scalar"])
                    
                    # 检查每个操作是否有记录函数ID
                    self.assertGreaterEqual(
                        args.get("Record function id", -1),
                        0,
                        f"Failed finding record function for op = {e}",
                    )

    # 定义测试函数，用于测试性能分析器的步长信息
    def test_profiler_strides(self):
        # 启用具体输入记录功能
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        
        # 创建一个基础的随机张量
        base_tensor = torch.randn(1024, dtype=torch.float32)
        # 使用as_strided方法创建张量a，指定形状和步长
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        # 使用as_strided方法创建张量b，指定形状和步长
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        
        # 使用性能分析器，记录张量形状
        with _profile(record_shapes=True) as prof:
            # 执行张量a和b的加法操作
            c = torch.add(a, b)

        # 创建临时文件以导出Chrome跟踪数据
        with TemporaryFileName(mode="w+") as fname:
            # 将性能分析结果导出为Chrome跟踪格式
            prof.export_chrome_trace(fname)
            # 打开临时文件并加载JSON数据
            with open(fname) as f:
                j = json.load(f)
                # 从跟踪事件中筛选出所有CPU操作事件
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                # 遍历每个操作事件
                for e in op_events:
                    args = e["args"]
                    # 如果操作事件名称是"aten::add"
                    if e["name"] == "aten::add":
                        # 断言输入步长
                        self.assertEqual(args["Input Strides"], [[17, 1], [25, 2], []])
    # 定义一个测试方法，用于测试前向和反向传播的连接情况
    def test_profiler_fwd_bwd_link(self):
        # 启用性能分析器，并使用 Kineto 进行分析
        with _profile(use_kineto=True) as prof:
            # 创建两个张量 t1 和 t2，均设置为需要梯度计算
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
            # 计算张量 t1 和 t2 的和，并存储在变量 z 中
            z = torch.add(t1, t2)
            # 创建一个张量 y，值为1，用于作为二进制交叉熵函数的目标值
            y = torch.ones(1)
            # 计算 z 和 y 之间的二进制交叉熵损失
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            # 根据损失进行反向传播计算梯度
            loss.backward()
        
        # 使用临时文件名进行操作，写入性能分析数据
        with TemporaryFileName(mode="w+") as fname:
            # 导出性能分析数据到 Chrome 跟踪文件中
            prof.export_chrome_trace(fname)
            # 打开临时文件，加载其中的 JSON 数据
            with open(fname) as f:
                j = json.load(f)
                # 获取事件列表
                events = j["traceEvents"]
                # 初始化字典来存储时间戳到事件名称的映射关系
                ts_to_name = {}
                # 初始化字典来存储流程开始时间戳到流程 ID 的映射关系
                flow_s_to_ts = {}
                # 初始化字典来存储流程结束时间戳到流程 ID 的映射关系
                flow_f_to_ts = {}
                
                # 遍历事件列表
                for e in events:
                    # 如果事件阶段是 "X"，表示是一个任务名称事件
                    if e["ph"] == "X":
                        # 将时间戳和任务名称存储到映射字典中
                        ts_to_name[e["ts"]] = e["name"]
                    # 如果事件中包含 "cat" 和 "name" 字段，并且 cat 是 "fwdbwd"，name 是 "fwdbwd"
                    if (
                        "cat" in e
                        and "name" in e
                        and e["cat"] == "fwdbwd"
                        and e["name"] == "fwdbwd"
                    ):
                        # 如果事件阶段是 "s"，表示流程开始
                        if e["ph"] == "s":
                            # 将流程 ID 和开始时间戳存储到映射字典中
                            flow_s_to_ts[e["id"]] = e["ts"]
                        # 如果事件阶段是 "f"，表示流程结束
                        elif e["ph"] == "f":
                            # 将流程 ID 和结束时间戳存储到映射字典中
                            flow_f_to_ts[e["id"]] = e["ts"]
                
                # 断言流程开始时间戳到流程 ID 的映射字典长度为2
                self.assertEqual(len(flow_s_to_ts), 2)
                # 断言流程结束时间戳到流程 ID 的映射字典长度为2
                self.assertEqual(len(flow_f_to_ts), 2)
                # 断言流程开始时间戳到流程 ID 的映射字典中包含流程 ID 1
                self.assertIn(1, flow_s_to_ts)
                # 断言流程结束时间戳到流程 ID 的映射字典中包含流程 ID 1
                self.assertIn(1, flow_f_to_ts)
                # 断言流程开始时间戳到流程 ID 的映射字典中包含流程 ID 2
                self.assertIn(2, flow_s_to_ts)
                # 断言流程结束时间戳到流程 ID 的映射字典中包含流程 ID 2
                self.assertIn(2, flow_f_to_ts)
                
                # 获取流程 ID 1 的开始时间戳和结束时间戳
                s_ts_1 = flow_s_to_ts[1]
                f_ts_1 = flow_f_to_ts[1]
                # 获取流程 ID 2 的开始时间戳和结束时间戳
                s_ts_2 = flow_s_to_ts[2]
                f_ts_2 = flow_f_to_ts[2]
                
                # 断言这些时间戳都存在于时间戳到事件名称的映射字典中
                self.assertTrue(
                    all(
                        ts in ts_to_name.keys()
                        for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2]
                    )
                )
                # 断言流程 ID 1 的开始事件名称为 "aten::binary_cross_entropy_with_logits"
                self.assertTrue(
                    ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits"
                )
                # 断言流程 ID 2 的开始事件名称为 "aten::add"
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")
    # 定义测试函数，用于验证禁用前向和反向链接的性能分析
    def test_profiler_disable_fwd_bwd_link(self):
        try:
            # 禁用 Torch 分析器中的前向和反向链接
            torch._C._profiler._set_fwd_bwd_enabled_val(False)

            # 使用 Kineto 进行性能分析
            with _profile(use_kineto=True) as prof:
                # 创建两个需要梯度的张量
                t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                # 执行张量加法
                z = torch.add(t1, t2)
                # 创建标签为1的张量 y
                y = torch.ones(1)
                # 计算二元交叉熵损失
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                # 执行反向传播
                loss.backward()

            # 使用临时文件名对象，保存性能分析结果到 Chrome 跟踪文件中
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                # 打开保存的文件，并加载其中的 JSON 数据
                with open(fname) as f:
                    j = json.load(f)
                    # 获取事件列表
                    events = j["traceEvents"]

                    # 遍历事件列表中的每个事件
                    for e in events:
                        # 检查事件的类别不等于 "fwdbwd"
                        self.assertNotEqual(e.get("cat", None), "fwdbwd")
        finally:
            # 最终恢复 Torch 分析器中的前向和反向链接
            torch._C._profiler._set_fwd_bwd_enabled_val(True)

    # 该测试在 Windows 上无法正常工作，可能的原因是 kineto/CUPTI 在该环境下不受支持。
    # 一旦 CI 稳定下来，我们可以缩小条件，以便在 Windows 上进行检查（TODO）。
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(IS_WINDOWS, "Test does not work on Windows")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    # 定义测试函数，用于验证 CUDA 同步事件的性能分析
    def test_profiler_cuda_sync_events(self):
        device = torch.device("cuda:0")
        # 创建在 CUDA 设备上的两个张量
        t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)

        # 定义工作负载函数
        def workload() -> None:
            # 执行张量加法
            torch.add(t1, t2)
            # CUDA 同步
            torch.cuda.synchronize()
            # 再次执行张量加法

        # 定义跟踪和检查函数
        def trace_and_check(exp_config: Optional[_ExperimentalConfig]) -> None:
            with _profile(
                use_kineto=True,
                use_cuda=True,
                experimental_config=exp_config,
            ) as prof:
                # 执行工作负载
                workload()

            with TemporaryFileName(mode="w+") as fname:
                # 将性能分析结果导出到 Chrome 跟踪文件
                prof.export_chrome_trace(fname)
                # 打开保存的文件，并加载其中的 JSON 数据
                with open(fname) as f:
                    j = json.load(f)
                    # 获取事件列表中的类别集合
                    cats = {e.get("cat", None) for e in j["traceEvents"]}
            # 断言预期在类别集合中找到 "cuda_sync" 事件
            self.assertTrue(
                "cuda_sync" in cats,
                "Expected to find cuda_sync event" f" found = {cats}",
            )

        # 打印信息，测试 _ExperimentalConfig 中的 enable_cuda_sync_events
        print("Testing enable_cuda_sync_events in _ExperimentalConfig")
        trace_and_check(exp_config=_ExperimentalConfig(enable_cuda_sync_events=True))

        # 打印信息，测试 _profiler._set_cuda_sync_enabled_val()
        print("Testing _profiler._set_cuda_sync_enabled_val()")
        try:
            # 启用 Torch 分析器中的 CUDA 同步事件
            torch._C._profiler._set_cuda_sync_enabled_val(True)
            trace_and_check(exp_config=None)
        finally:
            # 最终禁用 Torch 分析器中的 CUDA 同步事件
            torch._C._profiler._set_cuda_sync_enabled_val(False)
    def test_profiler_type(self):
        # 获取当前 Autograd 的 profiler 类型
        profiler_type = torch._C._autograd._profiler_type
        # 获取当前活跃的 profiler 类型
        ActiveProfilerType = torch._C._profiler.ActiveProfilerType
        # 断言当前 profiler 类型为 NONE
        self.assertEqual(profiler_type(), ActiveProfilerType.NONE)

        # 进入旧版 Autograd profiler 上下文
        with _profile_legacy():
            # 断言当前 profiler 类型为 LEGACY
            self.assertEqual(profiler_type(), ActiveProfilerType.LEGACY)

        # 进入 Kineto profiler 上下文
        with profile():
            # 断言当前 profiler 类型为 KINETO
            self.assertEqual(profiler_type(), ActiveProfilerType.KINETO)

    def test_profiler_correlation_id(self):
        """
        We expect the correlation_id to be unique across multiple invokation of the profiler,
        So we will reuse id_uniqueness_set.
        """
        # 初始化一个空集合用于存储 correlation_id，保证其唯一性
        id_uniqueness_set = set()
        # 创建一个简单的神经网络模型
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        # 生成输入数据
        inputs = torch.randn(40, 16, 18, 260)
        # 定义 uint32 最大值
        uint32_max = 2**32 - 1
        # 进行 5 次 profiler 上下文
        for i in range(5):
            with profile() as prof:
                model(inputs)
            # 遍历 profiler 的事件
            for event in prof.profiler.kineto_results.events():
                corr_id = event.correlation_id()
                # 检查 correlation_id 在 CPU 设备上的唯一性
                if corr_id and event.device_type() == DeviceType.CPU:
                    self.assertTrue(corr_id not in id_uniqueness_set)
                    id_uniqueness_set.add(corr_id)
                    # 断言 correlation_id 小于 uint32 最大值
                    self.assertTrue(corr_id < uint32_max)

    def test_nested_tensor_with_shapes(self):
        # 创建随机张量
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        # 创建嵌套张量
        inp = torch.nested.nested_tensor([a, b])
        # 使用 profiler 记录张量操作及其形状
        with torch.profiler.profile(record_shapes=True) as prof:
            torch.nn.functional.linear(inp, c, None)
        # 遍历 profiler 的事件
        for e in prof.events():
            if e.name in ("aten::mm", "aten::addmm"):
                # 检查输入张量的形状
                # 故意模糊的测试，以应对未来可能的实现更改
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.input_shapes[0]) > 0)

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_with_environment_variable(self):
        script = """
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
from torch.profiler import supported_activities, profile  # 导入性能分析相关函数
from torch.autograd.profiler import KinetoStepTracker  # 导入 Kineto 步骤追踪器

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)  # 定义第一个全连接层，输入大小为10，输出大小为5
        self.fc2 = nn.Linear(5, 2)   # 定义第二个全连接层，输入大小为5，输出大小为2

    def forward(self, x):
        return self.fc2(self.fc1(x))  # 前向传播函数，先经过fc1，再经过fc2

def payload(use_cuda=False):
    x = torch.randn(10, 10)  # 生成大小为10x10的随机张量x
    if use_cuda:
        x = x.cuda()  # 如果使用CUDA，则将张量x移动到GPU上
    y = torch.randn(10, 10)  # 生成大小为10x10的随机张量y
    if use_cuda:
        y = y.cuda()  # 如果使用CUDA，则将张量y移动到GPU上
    z = torch.mm(x, y)  # 计算x和y的矩阵乘积，得到张量z
    z = z + y  # 将张量z和张量y相加
    if use_cuda:
        z = z.cpu()  # 如果使用CUDA，则将张量z移动到CPU上

niters = 8  # 迭代次数设为8
use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()  # 检查CUDA是否支持

net = SimpleNet()  # 创建SimpleNet的实例net
opt = torch.optim.SGD(net.parameters(), lr=0.01)  # 使用SGD优化器，学习率为0.01
opt.zero_grad()  # 梯度清零
inputs = torch.rand(10)  # 生成大小为10的随机输入张量

with profile(activities=supported_activities()):  # 使用性能分析器profile，监视支持的所有活动
    payload(use_cuda=use_cuda)  # 执行payload函数，传入CUDA使用标志

initial_step = KinetoStepTracker.current_step()  # 获取当前Kineto步骤追踪器的当前步骤数

def run_batch():
    out = net(inputs)  # 将输入inputs输入到网络net中，得到输出out
    loss = torch.nn.functional.cross_entropy(out, torch.rand(2))  # 计算交叉熵损失
    loss.backward()  # 反向传播，计算梯度
    opt.step()  # 执行优化步骤

for _ in range(niters):
    run_batch()  # 执行多次运行批处理函数

with profile(
    activities=supported_activities(),
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
) as p:
    for _ in range(niters):
        run_batch()  # 执行多次运行批处理函数
        p.step()  # 执行性能分析器的步骤

assert KinetoStepTracker.current_step() == initial_step + 2 * niters  # 断言当前Kineto步骤追踪器的步骤数等于初始步骤数加上两倍的迭代次数
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "all", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )

    def test_concrete_inputs_profiling(self):
        x = torch.rand(2, 6)  # 生成大小为2x6的随机张量x
        with profile(record_shapes=True) as p:  # 使用性能分析器，记录张量形状
            y = x.as_strided([4, 3], [1, 4])  # 使用as_strided方法对张量x进行重复

        found = False
        for e in p.events():
            if e.name in ("aten::as_strided"):  # 查找事件名称为aten::as_strided的事件
                found = True
                self.assertTrue(len(e.input_shapes) > 0)  # 断言事件的输入形状列表不为空
                self.assertTrue(len(e.concrete_inputs) > 0)  # 断言事件的具体输入列表不为空
                self.assertEqual([2, 6], e.input_shapes[0])  # 断言第一个输入形状为[2, 6]
                self.assertEqual([4, 3], e.concrete_inputs[1])  # 断言第二个具体输入为[4, 3]
                self.assertEqual([1, 4], e.concrete_inputs[2])  # 断言第三个具体输入为[1, 4]

        self.assertTrue(found, "Expected to find aten::as_strided but did not")  # 断言找到了aten::as_strided事件
    # 定义一个测试方法，用于测试具体输入的性能调试开关
    def test_concrete_inputs_profiling_toggling(self):
        try:
            # 遍历两种情况：before=True, after=False 和 before=False, after=True
            for before, after in [(True, False), (False, True)]:
                # 创建一个 2x6 的随机张量 x
                x = torch.rand(2, 6)
                # 设置记录具体输入的开关值为 before
                torch._C._profiler._set_record_concrete_inputs_enabled_val(before)
                # 使用 profile() 上下文记录性能，包括形状
                with profile(record_shapes=True) as p:
                    # 使用 as_strided 方法操作张量 x
                    y = x.as_strided([4, 3], [1, 4])
                    # 设置记录具体输入的开关值为 after
                    torch._C._profiler._set_record_concrete_inputs_enabled_val(after)

                # 初始化 found 标志为 False
                found = False
                # 遍历 profile 中的事件
                for e in p.events():
                    # 如果事件名称为 "aten::as_strided"
                    if e.name in ("aten::as_strided"):
                        # 标记为找到了目标事件
                        found = True
                        # 断言事件的输入形状列表不为空
                        self.assertTrue(len(e.input_shapes))

                # 断言至少找到一个目标事件，否则输出提示信息
                self.assertTrue(found, "Expected to find aten::as_strided but did not")
        finally:
            # 最终恢复记录具体输入的开关值为 True
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)

    # 测试性能分析器是否启用
    def test_is_profiler_enabled(self):
        # 断言自动求导性能分析器当前为未启用状态
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        # 使用 profile() 上下文启用性能分析器，并进行断言
        with profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        # 断言自动求导性能分析器当前为未启用状态
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        # 使用 torch.autograd.profiler.profile() 上下文启用性能分析器，并进行断言
        with torch.autograd.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        # 断言自动求导性能分析器当前为未启用状态
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

    # 测试 _RecordFunctionFast 类的性能记录功能
    def test_guarded_record_function_fast(self):
        # 创建两个 4x4 的随机张量 x 和 y
        x, y = (torch.rand((4, 4)) for _ in range(2))

        # 使用 profile() 上下文记录性能
        with profile() as p:
            # 创建一个 _RecordFunctionFast 对象 cm
            cm = torch._C._profiler._RecordFunctionFast("guarded_rff")
            # 循环 4 次
            for _ in range(4):
                # 如果自动求导性能分析器已启用，则使用 cm 记录 x.add(y) 操作
                if torch.autograd.profiler._is_profiler_enabled:
                    with cm:
                        x.add(y)
                else:
                    # 否则直接执行 x.add(y) 操作
                    x.add(y)

        # 断言事件列表中 "guarded_rff" 出现的次数不少于 4 次
        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "guarded_rff"]), 4
        )

    # 如果 CUDA 可用，测试事件列表的导出和表格显示功能
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_event_list(self):
        # 创建两个在 CUDA 设备上的随机张量 x 和 y，要求梯度计算
        x, y = (torch.rand((4, 4), requires_grad=True, device="cuda") for _ in range(2))
        # 使用 profile(with_stack=True) 上下文记录性能，包括堆栈信息
        with profile(with_stack=True) as p:
            # 执行 x @ y 的矩阵乘法，然后 relu 并求和
            z = (x @ y).relu().sum()
            # 反向传播
            z.backward()

        # 创建 EventList 对象 event_list，包含 p 的事件列表
        event_list = torch.autograd.profiler_util.EventList(p.events())
        # 使用临时文件名 fname 保存事件列表为 Chrome Trace 格式
        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            # 打开并加载 fname 中的 JSON 数据
            with open(fname) as f:
                json.load(f)

        # 在控制台输出事件列表的表格形式
        event_list.table()

    # 检查所有 GPU 是否存在，并与最大 GPU 数量进行比较
    def _check_all_gpu_present(self, gpu_dict, max_gpu_count):
        # 遍历 0 到 max_gpu_count 的 GPU
        for i in range(0, max_gpu_count):
            # 断言 GPU 字典中 "GPU i" 存在且值为 1
            self.assertEqual(gpu_dict["GPU " + str(i)], 1)

    # 执行 JSON 的合理性测试。检查所有事件是否在分析器开始和结束之间，
    # 同时检查在使用 CUDA 时跟踪中是否存在 GPU 值
    # 验证基本的 JSON 格式的跟踪事件，确保符合特定条件
    def _validate_basic_json(self, traceEvents, cuda_available=False):
        # 定义最大 GPU 数量为 8
        MAX_GPU_COUNT = 8
        # PROFILER_IDX 为倒数第四个元素的索引
        PROFILER_IDX = -4
        # RECORD_END 为列表最后一个元素的索引
        RECORD_END = -1
        # RECORD_START 为倒数第二个元素的索引
        RECORD_START = -2
        # 获取 PROFILER_IDX 处的跟踪事件
        traceEventProfiler = traceEvents[PROFILER_IDX]

        # 断言确保 Profiler 的名称为 "PyTorch Profiler (0)"
        self.assertTrue(traceEventProfiler["name"] == "PyTorch Profiler (0)")
        # 断言确保最后一个跟踪事件的名称为 "Record Window End"
        self.assertTrue(traceEvents[RECORD_END]["name"] == "Record Window End")
        # 断言确保倒数第二个跟踪事件的名称为 "Iteration Start: PyTorch Profiler"
        self.assertTrue(
            traceEvents[RECORD_START]["name"] == "Iteration Start: PyTorch Profiler"
        )
        # 检查分析器开始和结束时间是否在记录区间内
        self.assertGreaterEqual(
            traceEventProfiler["ts"],
            traceEvents[RECORD_START]["ts"],
            "Profiler starts before record!",
        )
        self.assertLessEqual(
            traceEventProfiler["ts"] + traceEventProfiler["dur"],
            traceEvents[RECORD_END]["ts"],
            "Profiler ends after record end!",
        )

        # 创建一个 GPU 字典，用于统计不同 GPU 出现的次数
        gpu_dict = collections.defaultdict(int)
        # 遍历所有跟踪事件
        for i, traceEvent in enumerate(traceEvents):
            # 跳过记录结束和记录开始的索引位置
            if (
                i == len(traceEvents) + RECORD_END
                or i == len(traceEvents) + RECORD_START
            ):
                continue
            # 确保所有有效的跟踪事件都在分析器的时间范围内
            if "ts" in traceEvent:
                self.assertGreaterEqual(
                    traceEvent["ts"],
                    traceEventProfiler["ts"],
                    "Trace event is out of bounds",
                )
            # 某些 Python 事件可能会略微超过记录结束时间，这里只需将事件结束时间与 RECORD_END 进行比较
            if "dur" in traceEvent:
                self.assertLessEqual(
                    traceEvent["ts"] + traceEvent["dur"],
                    traceEvents[RECORD_END]["ts"],
                    "Trace event ends too late!",
                )
            # 获取 GPU 标签，统计每个 GPU 的出现次数
            gpu_value = traceEvent.get("args", {}).get("labels", None)
            if gpu_value and "GPU" in gpu_value:
                gpu_dict[gpu_value] += 1
                # Max PID offset is 5M, based from pytorch/kineto include header:
                # https://github.com/pytorch/kineto/blob/8681ff11e1fa54da39023076c5c43eddd87b7a8a/libkineto/include/output_base.h#L35
                kExceedMaxPid = 5000000
                # 断言确保下一个跟踪事件的排序索引等于 kExceedMaxPid 加上 GPU 序号的整数值
                self.assertTrue(
                    traceEvents[i + 1]["args"]["sort_index"]
                    == kExceedMaxPid + int(gpu_value.split()[1])
                )

        # TODO add checking gpu count if cpuOnly_ is true or not
    # 定义测试辅助函数，用于生成基本的 Chrome 跟踪测试结果
    def _test_chrome_trace_basic_helper(self, with_cuda=False):
        # 根据是否使用 CUDA 设置设备类型
        if with_cuda:
            device = "cuda"
        else:
            device = "cpu"
        # 生成两个随机张量，并将它们移动到指定设备上
        x, y = (torch.rand(4, 4).to(device) for _ in range(2))

        # 使用 torch profiler 捕获代码块执行的性能信息，包括堆栈信息
        with profile(with_stack=True) as p:
            torch.add(x, y)
        # 创建临时文件用于保存 Chrome 跟踪数据
        with TemporaryFileName(mode="w+") as fname:
            # 将性能分析结果导出为 Chrome 跟踪格式
            p.export_chrome_trace(fname)
            # 从文件中加载 JSON 数据
            with open(fname) as f:
                report = json.load(f)
                # 验证基本的 JSON 结构和内容
                self._validate_basic_json(report["traceEvents"], with_cuda)

    # 使用 unittest 装饰器，跳过条件不满足的测试用例
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    # 测试基本的 Chrome 跟踪功能
    def test_basic_chrome_trace(self):
        # 调用测试辅助函数，不使用 CUDA
        self._test_chrome_trace_basic_helper()
        # 如果 CUDA 可用，再次调用测试辅助函数，使用 CUDA
        if torch.cuda.is_available():
            self._test_chrome_trace_basic_helper(with_cuda=True)

    # 使用 unittest 装饰器，跳过条件不满足的测试用例
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    # 测试分析器的时间比例
    def test_profiler_time_scale(self):
        # 定义时间误差边界
        MARGIN_ERROR = 0.5
        SEC_TO_US = 1000 * 1000
        WAIT_TIME = 10
        # 使用 torch profiler 捕获代码块执行的性能信息
        with profile() as p:
            # 记录一个函数调用，测试时间跨度
            with torch.profiler.record_function("test_span"):
                for i in range(WAIT_TIME):
                    torch.rand(4, 4)
                    time.sleep(1)
        # 获取捕获的事件信息
        events = p.events()

        # 确保函数事件的时间比例符合预期
        self.assertTrue(events[0].name == "test_span")
        test_span = events[0]
        self.assertGreaterEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME - MARGIN_ERROR,
            "event out of range",
        )
        self.assertLessEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME + MARGIN_ERROR,
            "event out of range",
        )

        # 确保跟踪时间比例符合预期
        with TemporaryFileName(mode="w+") as fname:
            # 将性能分析结果导出为 Chrome 跟踪格式
            p.export_chrome_trace(fname)
            # 从文件中加载 JSON 数据
            with open(fname) as f:
                report = json.load(f)
            events = report["traceEvents"]
            for event in events:
                if event["name"] == "test_span":
                    self.assertGreaterEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME - MARGIN_ERROR,
                        "profiling out of range",
                    )
                    self.assertLessEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME + MARGIN_ERROR,
                        "profiling out of range",
                    )

    # 辅助函数，用于执行调度器相关测试
    def _schedule_helper(self, warmup, active, repeat):
        # 使用 torch profiler 捕获代码块执行的性能信息，并设置调度器参数
        with profile(
            schedule=torch.profiler.schedule(
                skip_first=0, wait=0, warmup=warmup, active=active, repeat=repeat
            )
        ) as prof:
            for i in range(100):
                torch.add(1, 2)
                # 在每个步骤结束时记录性能数据
                prof.step()
        # 遍历捕获的关键平均值事件
        for ev in prof.key_averages():
            if ev.key == "aten::add":
                return ev.count
        return 0
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    # 使用装饰器 @skipIfTorchDynamo，在启用 Dynamo 时忽略测试
    def test_schedule_function_count(self):
        # 测试调度辅助函数，检查不同参数组合下的返回值是否正确
        self.assertEqual(self._schedule_helper(warmup=0, active=1, repeat=1), 1)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=0), 100)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=10), 50)
        self.assertEqual(self._schedule_helper(warmup=1, active=5, repeat=0), 83)
        self.assertEqual(self._schedule_helper(warmup=10, active=10, repeat=4), 40)
        self.assertEqual(self._schedule_helper(warmup=50, active=1, repeat=0), 1)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义第一个全连接层，输入维度为10，输出维度为5
        self.fc1 = nn.Linear(10, 5)
        # 定义第二个全连接层，输入维度为5，输出维度为2
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # 神经网络的前向传播过程，先通过第一层全连接层，再通过第二层全连接层
        return self.fc2(self.fc1(x))


@dataclass(frozen=True)
class MockKinetoEvent:
    _name: str
    _start_us: int
    _duration_us: int
    _linked_correlation_id: int
    _device_type: int

    @property
    def name(self) -> str:
        # 返回事件名称
        return self._name

    def start_ns(self) -> int:
        # 将微秒转换为纳秒，返回事件开始时间
        return self._start_us * 1000

    def duration_ns(self) -> int:
        # 将微秒转换为纳秒，返回事件持续时间
        return self._duration_us * 1000

    def linked_correlation_id(self) -> int:
        # 返回关联的相关性 ID
        return self._linked_correlation_id

    def device_type(self) -> DeviceType:
        # 如果设备类型为1，则返回CUDA；否则返回CPU
        return DeviceType.CUDA if self._device_type == 1 else DeviceType.CPU


@dataclass(frozen=True)
class MockProfilerEvent:
    _name: str
    id: int
    start_time_ns: int
    duration_time_ns: int
    correlation_id: int = 0
    children: List["MockProfilerEvent"] = field(default_factory=list)
    parent: Optional["MockProfilerEvent"] = None

    @property
    def end_time_ns(self):
        # 计算事件结束时间，即开始时间加上持续时间
        return self.start_time_ns + self.duration_time_ns

    @property
    def name(self) -> str:
        # 返回事件名称
        return self._name

    def __post__init__(self, parent, children):
        # 设置事件的父事件和子事件列表
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "children", children)


class MockNode:
    def __init__(self, name, children) -> None:
        # 初始化 MockNode 类，接受节点名称和子节点列表
        self.name = name
        # 递归创建子节点列表
        self.children = [MockNode(name, i) for name, i in children.items()]


class TestExperimentalUtils(TestCase):
    def make_tree(self) -> List[MockNode]:
        # 创建一个测试用的树结构，返回树中的节点列表
        tree = {
            "root_0": {
                "1": {"2": {}},
                "3": {
                    "4": {},
                    "5": {},
                },
            },
            "root_1": {
                "6": {},
                "7": {},
                "8": {
                    "9": {"10": {}},
                },
            },
        }
        return [MockNode(name, i) for name, i in tree.items()]

    def test_dfs(self) -> None:
        # 测试深度优先搜索（DFS）算法
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_dfs(self.make_tree())),
            "root_0 1 2 3 4 5 root_1 6 7 8 9 10",
        )

    def test_bfs(self) -> None:
        # 测试广度优先搜索（BFS）算法
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_bfs(self.make_tree())),
            "root_0 root_1 1 3 6 7 8 2 4 5 9 10",
        )
    # 生成一个模拟的性能分析器对象
    def generate_mock_profile():
        # CUDA 相关事件模拟数据
        cuda_events = [
            MockKinetoEvent("cudaLaunchKernel", 400, 100, 1, 0),
            MockKinetoEvent("cudaLaunchKernel", 500, 100, 2, 0),
            MockKinetoEvent("cudaLaunchKernel", 600, 100, 3, 0),
            MockKinetoEvent("cudaLaunchKernel", 700, 100, 4, 0),
            MockKinetoEvent("cudaLaunchKernel", 800, 100, 5, 0),
            MockKinetoEvent("cudaLaunchKernel", 1500, 100, 6, 0),
            MockKinetoEvent("GPU", 900, 100, 1, 1),
            MockKinetoEvent("GPU", 1000, 100, 2, 1),
            MockKinetoEvent("GPU", 1100, 100, 3, 1),
            MockKinetoEvent("GPU", 1200, 100, 4, 1),
            MockKinetoEvent("GPU", 1300, 100, 5, 1),
            MockKinetoEvent("GPU", 1700, 100, 6, 1),
        ]
        # CPU 相关事件模拟数据
        cpu_events = [
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 1, 0, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 2, 100000, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 3, 200000, 100000),
            MockProfilerEvent("CPU (Before cudaLaunchKernel)", 4, 300000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 5, 400000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 6, 500000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 7, 600000, 100000),
            MockProfilerEvent("CPU (After cudaLaunchKernel)", 8, 700000, 100000),
            MockProfilerEvent("CPU (After GPU)", 9, 800000, 100000),
            MockProfilerEvent("CPU (After GPU)", 10, 900000, 100000),
            MockProfilerEvent("CPU (After GPU)", 11, 1100000, 100000),
            MockProfilerEvent("CPU (After GPU)", 12, 1200000, 500000),
        ]

        # 创建一个模拟的性能分析器对象
        profiler = unittest.mock.Mock()
        # 设置模拟对象的属性，模拟 Kineto 结果
        profiler.kineto_results = unittest.mock.Mock()
        profiler.kineto_results.events = unittest.mock.Mock(return_value=cuda_events)
        # 设置模拟对象的属性，模拟实验性事件树
        profiler.kineto_results.experimental_event_tree = unittest.mock.Mock(
            return_value=cpu_events
        )
        # 返回模拟的性能分析器对象
        return profiler

    @staticmethod
    # 测试工具方法，计算自身时间
    def test_utils_compute_self_time(self):
        # 使用 profile() 上下文管理器开始性能分析
        with profile() as prof:
            # 创建两个张量，需要计算梯度
            t1, t2 = torch.ones(1, requires_grad=True), torch.ones(
                1, requires_grad=True
            )
            # 执行张量加法
            z = torch.add(t1, t2)
            # 创建一个全为1的张量
            y = torch.ones(1)
            # 计算二元交叉熵损失
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            # 反向传播计算梯度
            loss.backward()
        # 创建 BasicEvaluation 对象，基于性能分析器对象 prof.profiler
        basic_eval = _utils.BasicEvaluation(prof.profiler)
        # 获取基本评估的指标
        metrics = basic_eval.metrics
        # 断言至少有一个指标被计算
        self.assertTrue(len(metrics) > 0)
        # 遍历所有事件的指标
        for event_key, event_metrics in metrics.items():
            # 断言事件的自身时间与其持续时间减去所有子事件的持续时间之和相等
            self.assertEqual(
                event_metrics.self_time_ns,
                event_key.event.duration_time_ns
                - sum(child.duration_time_ns for child in event_key.event.children),
            )
    def test_utils_intervals_overlap(self):
        # 创建一个名为 event 的事件键对象，基于 MockProfilerEvent("Event 1", 1, 5, 5) 的数据
        event = _utils.EventKey(MockProfilerEvent("Event 1", 1, 5, 5))
        
        # 定义一组时间区间列表 intervals
        intervals = [
            _utils.Interval(0, 9),
            _utils.Interval(1, 2),
            _utils.Interval(2, 3),
            _utils.Interval(3, 4),
            _utils.Interval(4, 5),
            _utils.Interval(8, 12),
        ]
        
        # 打印事件键对象与时间区间列表 intervals 的重叠情况
        print(event.intervals_overlap(intervals))
        
        # 断言事件键对象与时间区间列表 intervals 的重叠数量为 5
        self.assertEqual(event.intervals_overlap(intervals), 5)

    def test_utils_compute_queue_depth(self):
        # 定义一个内部函数 format_queue_depth，用于格式化队列深度列表和事件列表的输出
        def format_queue_depth(queue_depth_list, events):
            res = ""
            # 遍历队列深度列表和事件列表，将每个事件的队列深度和名称格式化后添加到结果字符串 res 中
            for data, event in zip(queue_depth_list, events):
                res += f"{data.queue_depth} [{event.name}]\n"
            return res

        # 使用 Mock 数据生成模拟的 profiler 对象，因为时间序列数据不稳定
        profiler = self.generate_mock_profile()
        
        # 创建基础评估对象 basic_evaluation，基于生成的 profiler
        basic_evaluation = _utils.BasicEvaluation(profiler)
        
        # 断言格式化后的队列深度输出符合预期结果
        self.assertExpectedInline(
            format_queue_depth(
                basic_evaluation.queue_depth_list, basic_evaluation.cuda_events
            ),
            """\
    @unittest.skipIf(IS_JETSON, "JSON not behaving as expected on Jetson")
    # 在 Jetson 上运行时，如果 JSON 的行为不符合预期，则跳过测试
    def test_utils_get_optimizable_events(self):
        # 使用加载的模拟性能数据创建 BasicEvaluation 对象
        basic_evaluation = _utils.BasicEvaluation(self.load_mock_profile())
        # 获取可优化事件列表，其中最多包含 2 个事件，并禁用打印
        optimizable_events = basic_evaluation.get_optimizable_events(
            2, print_enable=False
        )
        # 期望输出是可优化事件列表中每个事件的名称
        expected_output = "\n".join(
            [f"{event_key.event.name}" for event_key in optimizable_events]
        )
        # 断言实际输出与期望输出相符
        self.assertExpectedInline(
            expected_output,
            """\
<built-in function _cuda_synchronize>
aten::copy_""",
        )
    # 定义一个测试方法，用于测试性能分析器的名称模式匹配功能
    def test_profiler_name_pattern(self):
        # 创建一个大小为4096x4096的张量，并将所有元素初始化为1
        x = torch.ones((4096, 4096))
        # 使用性能分析器记录以下代码块的性能数据
        with profile() as prof:
            # 执行5次迭代，每次迭代执行矩阵乘法和加法操作
            for _ in range(5):
                x = x @ x  # 矩阵乘法运算
                x = x + x  # 矩阵加法运算
        # 使用给定的性能分析结果和操作名称模式"aten::mm"创建名称模式匹配对象
        matched_events = NamePattern(prof, "aten::mm").matched_events()
        # 从匹配的事件列表中提取事件名称，以换行符连接成字符串
        output = "\n".join([f"{event.name}" for event in matched_events])
        # 使用断言方法验证输出结果符合预期
        self.assertExpectedInline(
            output,
            """\
    # aten::mm 语句，表示PyTorch中矩阵乘法的基本操作
    aten::mm
    # aten::mm 语句，表示PyTorch中矩阵乘法的基本操作
    aten::mm
    # aten::mm 语句，表示PyTorch中矩阵乘法的基本操作
    aten::mm
    # aten::mm 语句，表示PyTorch中矩阵乘法的基本操作
    aten::mm"""

# TODO: Add logic for CUDA version of test
@unittest.skipIf(torch.cuda.is_available(), "Test not working for CUDA")
def test_profiler_pattern_match_helper(self):
    # 创建一个全为1的100x100张量
    x = torch.ones((100, 100))
    # 开始性能分析
    with profile() as prof:
        for _ in range(5):
            # 执行张量乘法操作
            x = x @ x
            # 执行张量加法操作
            x = x + x
    # 获取事件树
    event_tree = prof.profiler.kineto_results.experimental_event_tree()
    # 创建性能分析模式对象
    pattern = Pattern(prof)
    # 断言：事件树的第一个节点的兄弟节点为空
    self.assertEqual([], pattern.siblings_of(event_tree[0])[0])
    # 断言：事件树的第一个节点的兄弟节点为其余的所有节点
    self.assertEqual(event_tree[1:], pattern.siblings_of(event_tree[0])[1])
    # 获取事件树的第一个节点的子节点
    child_nodes = event_tree[0].children
    # 断言：子节点的第一个节点的兄弟节点为空
    self.assertEqual([], pattern.siblings_of(child_nodes[0])[0])
    # 断言：子节点的第一个节点的兄弟节点为其余的所有节点
    self.assertEqual(child_nodes[1:], pattern.siblings_of(child_nodes[0])[1])
    # 断言：事件树的根节点是子节点的第一个节点的第一个子节点的根节点
    self.assertEqual(
        event_tree[0], pattern.root_of(event_tree[0].children[0].children[0])
    )
    # 断言：事件树的最后一个节点的下一个节点为空
    self.assertEqual(None, pattern.next_of(event_tree[-1]))
    # 断言：事件树的第一个节点的下一个节点为事件树的第二个节点
    self.assertEqual(event_tree[1], pattern.next_of(event_tree[0]))
    # 断言：事件树的第一个节点的上一个节点为空
    self.assertEqual(event_tree[0], pattern.prev_of(event_tree[1]))

@unittest.skipIf(
    TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
)
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
def test_profiler_extra_cuda_copy_pattern(self):
    # 不同的CUDA复制模式及其预期匹配结果
    cases = (
        (0, lambda: torch.ones((100, 100), device="cuda")),
        (1, lambda: torch.ones((100, 100)).to("cuda")),
        (1, lambda: torch.zeros((100, 100)).to("cuda")),
        (1, lambda: torch.empty((100, 100)).fill_(5).to("cuda")),
        (1, lambda: torch.ones((100, 100)).cuda()),
        (1, lambda: torch.zeros((100, 100)).cuda()),
        (1, lambda: torch.empty((100, 100)).fill_(5).cuda()),
        (1, lambda: torch.rand((100, 100)).cuda()),
        (1, lambda: torch.randn((100, 100)).cuda()),
        (1, lambda: torch.full((100, 100), 10).cuda()),
        (0, lambda: torch.rand((100, 100)).to(dtype=torch.float16)),
        (0, lambda: torch.rand((100, 100)).half()),
        (0, lambda: torch.rand((100, 100), device="cuda").half()),
    )
    num_matched = []
    for _, fn in cases:
        # 开始性能分析，记录堆栈和形状
        with profile(with_stack=True, record_shapes=True) as prof:
            fn()
        # 创建额外的CUDA复制模式对象
        pattern = ExtraCUDACopyPattern(prof)
        # 获取匹配事件的数量
        num_matched.append(len(pattern.matched_events()))
    # 断言：每个测试用例中匹配的事件数量与预期相符
    self.assertEqual(num_matched, [i for i, _ in cases])

@unittest.skipIf(
    TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
)
    # 定义一个测试方法，用于测试循环索引模式的性能分析器
    def test_profiler_for_loop_indexing_pattern(self):
        # 创建一个大小为100x100的张量，所有元素初始化为1
        x = torch.ones((100, 100))

        # 第一个测试用例：简单的循环，将每个索引i处的元素设置为i
        def case1():
            for i in range(100):
                x[i] = i

        # 第二个测试用例：累加所有索引i处的元素值
        def case2():
            y = 0
            for i in range(100):
                y += x[i]

        # 第三个测试用例：计算所有索引i处的元素的乘积
        def case3():
            y = 1
            for i in range(100):
                y *= x[i]

        # 第四个测试用例：使用矩阵乘法操作100次，结果保存在y中
        def case4():
            y = x
            for _ in range(100):
                y = y @ x

        # 第五个测试用例：在第i行上设置张量x的所有元素为从i开始的100个递增数
        def case5():
            for i in range(100):
                x[i, :] = torch.arange(100) + i

        # 定义测试用例集合，每个元组包含一个标志位和一个测试函数
        cases = ((1, case1), (1, case2), (1, case3), (0, case4), (1, case5))
        
        # 用于存储每个测试用例中匹配的事件数的列表
        num_matched = []

        # 对于每个测试用例，使用性能分析器记录函数运行时的性能信息
        for _, fn in cases:
            with profile(with_stack=True) as prof:
                fn()
            # 创建一个循环索引模式的实例，并获取匹配的事件列表
            pattern = ForLoopIndexingPattern(prof)
            num_matched.append(len(pattern.matched_events()))

        # 断言每个测试用例中匹配的事件数是否与预期相符
        self.assertEqual(num_matched, [i for i, _ in cases])

    # 当CUDA可用时执行的测试方法，用于测试FP32矩阵乘法模式的性能分析器
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_fp32_matmul_pattern(self):
        # 创建一个大小为100x100的CUDA张量，所有元素初始化为1
        x = torch.ones((100, 100), device="cuda")
        
        # 使用性能分析器记录CUDA张量x与自身的矩阵乘法操作
        with profile(with_stack=True) as prof:
            x = x @ x
        
        # 创建一个FP32矩阵乘法模式的实例，并获取匹配的事件列表
        pattern = FP32MatMulPattern(prof)
        
        # 判断是否存在TF32操作，将结果转换为0或1
        has_tf32 = 0 if pattern.skip else 1
        
        # 获取匹配的事件数
        num_matched = len(pattern.matched_events())
        
        # 断言匹配的事件数是否与是否存在TF32操作相符
        self.assertEqual(num_matched, has_tf32)

    # 当CUDA可用时执行的测试方法，用于测试额外的CUDA复制模式的性能分析器
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_extra_cuda_copy_pattern_benchmark(self):
        # 使用性能分析器记录两次CUDA张量创建和复制的操作，并记录形状信息
        with profile(with_stack=True, record_shapes=True) as prof:
            x = torch.ones((100, 100)).to("cuda")
            x = torch.ones((50, 50)).to("cuda")
        
        # 创建一个额外的CUDA复制模式的实例，并获取匹配的事件列表
        pattern = ExtraCUDACopyPattern(prof)
        
        # 对匹配的事件执行基准测试，并获取形状因子映射
        shapes_factor_map = pattern.benchmark(pattern.matched_events())
        
        # 断言形状因子映射的长度是否为2
        self.assertEqual(len(shapes_factor_map), 2)
    # 定义测试函数，用于测试优化器单张量模式的性能
    def test_profiler_optimizer_single_tensor_pattern(self):
        # 创建一个100x100的全1张量作为输入张量
        x = torch.ones((100, 100))
        # 定义测试用例，每个测试用例包含一个标志位和一个生成优化器的lambda函数
        cases = (
            (1, lambda: torch.optim.Adam(model.parameters())),
            (1, lambda: torch.optim.SGD(model.parameters(), lr=0.01)),
            (1, lambda: torch.optim.AdamW(model.parameters())),
            (0, lambda: torch.optim.Adam(model.parameters(), foreach=True)),
            (0, lambda: torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)),
            (0, lambda: torch.optim.AdamW(model.parameters(), foreach=True)),
        )
        # 存储每个测试用例匹配的事件数量
        num_matched = []
        # 遍历每个测试用例
        for _, fn in cases:
            # 使用profile进行性能分析，同时记录堆栈信息
            with profile(with_stack=True) as prof:
                # 构建简单的神经网络模型
                model = nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 10),
                )
                # 调用生成的优化器
                optimizer = fn()
                # 将优化器的梯度置零
                optimizer.zero_grad()
                # 对模型进行前向传播得到预测值
                y_hat = model(x)
                # 计算交叉熵损失
                loss = torch.nn.functional.cross_entropy(
                    y_hat, torch.randint(0, 10, (100,))
                )
                # 反向传播计算梯度
                loss.backward()
                # 优化器执行一步参数更新
                optimizer.step()
            # 创建优化器单张量模式对象
            pattern = OptimizerSingleTensorPattern(prof)
            # 记录匹配到的事件数量
            num_matched.append(len(pattern.matched_events()))
        # 断言每个测试用例匹配的事件数量与预期列表一致
        self.assertEqual(num_matched, [i for i, _ in cases])

    # 定义测试函数，用于测试同步数据加载器模式的性能
    def test_profiler_synchronized_dataloader_pattern(self):
        # 创建一个100x100的随机数据集
        dataset = torch.rand((100, 100))
        # 创建同步数据加载器，批量大小为10
        sync_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        # 创建异步数据加载器，批量大小为10，使用4个工作进程
        async_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=10, num_workers=4
        )
        # 使用profile进行性能分析，同时记录堆栈信息
        with profile(with_stack=True) as prof:
            # 从同步数据加载器获取下一个批次数据
            next(iter(sync_dataloader))
            # 从异步数据加载器获取下一个批次数据
            next(iter(async_dataloader))
        # 创建同步数据加载器模式对象
        pattern = SynchronizedDataLoaderPattern(prof)
        # 获取匹配到的事件数量
        num_matched = len(pattern.matched_events())
        # 断言匹配到的事件数量为1
        self.assertEqual(num_matched, 1)

    # 跳过Torch Dynamo测试装饰器，用于特定条件下的测试跳过
    @skipIfTorchDynamo(
        "pattern checks for aten::_zero op which might not be there with torch.compile'd graph"
    )
    # 定义测试函数，测试不将梯度设置为 None 的模式
    def test_profiler_grad_not_set_to_none_pattern(self):
        # 创建一个大小为 (100, 100) 的张量 x，所有元素为 1
        x = torch.ones((100, 100))
        # 创建一个神经网络模型，包括两个线性层和一个ReLU激活函数
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        # 使用Adam优化器优化模型的参数
        optimizer = torch.optim.Adam(model.parameters())
        # 定义测试案例，每个案例包含一个标志和一个清除梯度的lambda函数
        cases = (
            (0, lambda: optimizer.zero_grad()),  # 标志为0，清除优化器梯度
            (0, lambda: model.zero_grad()),     # 标志为0，清除模型梯度
            (1, lambda: optimizer.zero_grad(set_to_none=False)),  # 标志为1，不将优化器梯度设置为None
            (1, lambda: model.zero_grad(set_to_none=False)),      # 标志为1，不将模型梯度设置为None
        )
        # 存储匹配事件的数量的列表
        num_matched = []
        # 遍历测试案例
        for _, fn in cases:
            # 使用性能分析器进行性能分析，并启用堆栈跟踪
            with profile(with_stack=True) as prof:
                # 将输入 x 输入模型，获取预测结果 y_hat
                y_hat = model(x)
                # 计算交叉熵损失
                loss = torch.nn.functional.cross_entropy(
                    y_hat, torch.randint(0, 10, (100,))
                )
                # 反向传播计算梯度
                loss.backward()
                # 使用优化器更新模型参数
                optimizer.step()
                # 调用清除梯度的函数
                fn()
            # 创建梯度未设置为None模式的实例
            pattern = GradNotSetToNonePattern(prof)
            # 将匹配的事件数量添加到列表中
            num_matched.append(len(pattern.matched_events()))
        # 断言匹配的事件数量与预期的结果列表相等
        self.assertEqual(num_matched, [i for i, _ in cases])

    # 定义测试函数，测试卷积层后面接批量归一化层的模式
    def test_profiler_conv2d_bias_followed_by_batchnorm2d_pattern(self):
        # 创建一个形状为 (1, 3, 32, 32) 的张量 x，其值为随机值
        x = torch.randn((1, 3, 32, 32))
        # 定义测试案例，每个案例包含一个标志和一个神经网络模型
        cases = (
            (1, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1), nn.BatchNorm2d(3))),  # 标志为1，包含卷积层和批量归一化层
            (0, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1, bias=False), nn.BatchNorm2d(3))),  # 标志为0，不包含偏置的卷积层和批量归一化层
            (0, nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1))),  # 标志为0，只包含卷积层
        )
        # 存储匹配事件的数量的列表
        num_matched = []
        # 遍历测试案例
        for _, model in cases:
            # 使用性能分析器进行性能分析，并记录张量形状
            with profile(with_stack=True, record_shapes=True) as prof:
                # 将输入 x 输入模型
                model(x)
            # 创建卷积层后接批量归一化层模式的实例
            pattern = Conv2dBiasFollowedByBatchNorm2dPattern(prof)
            # 将匹配的事件数量添加到列表中
            num_matched.append(len(pattern.matched_events()))
        # 断言匹配的事件数量与预期的结果列表相等
        self.assertEqual(num_matched, [i for i, _ in cases])

    # 跳过Torch Dynamo测试，如果激活了Dynamo
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_profiler_matmul_dim_fp16_pattern(self):
        # 定义测试案例，每个案例包含一个标志和一个形状为 (m, n) 的CUDA张量
        cases = (
            (1, torch.randn((201, 201), device="cuda", dtype=torch.float16)),  # 标志为1，形状为 (201, 201) 的FP16 CUDA张量
            (1, torch.randn((3, 97, 97), device="cuda", dtype=torch.float16)),   # 标志为1，形状为 (3, 97, 97) 的FP16 CUDA张量
            (0, torch.randn((200, 200), device="cuda", dtype=torch.float16)),    # 标志为0，形状为 (200, 200) 的FP16 CUDA张量
            (0, torch.randn((3, 200, 200), device="cuda", dtype=torch.float16)),  # 标志为0，形状为 (3, 200, 200) 的FP16 CUDA张量
        )
        # 存储匹配事件的数量的列表
        num_matched = []
        # 遍历测试案例
        for _, x in cases:
            # 使用性能分析器进行性能分析，并记录张量形状
            with profile(with_stack=True, record_shapes=True) as prof:
                # 执行张量 x 的矩阵乘法操作
                x @ x
            # 创建FP16维度矩阵乘法模式的实例
            pattern = MatMulDimInFP16Pattern(prof)
            # 将匹配的事件数量添加到列表中
            num_matched.append(len(pattern.matched_events()))
        # 断言匹配的事件数量与预期的结果列表相等
        self.assertEqual(num_matched, [i for i, _ in cases])

    # 如果激活了Dynamo，则跳过Torch Dynamo测试
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    # 定义测试函数 test_profiler_pattern_matcher_json_report
    def test_profiler_pattern_matcher_json_report(self):
        # 创建一个大小为 (100, 100) 的全一张量
        x = torch.ones((100, 100))
        # 构建神经网络模型，包括两个线性层和一个ReLU激活层
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        # 使用Adam优化器来优化模型的参数
        optimizer = torch.optim.Adam(model.parameters())
        # 使用性能分析器，并记录调用栈和形状
        with profile(with_stack=True, record_shapes=True) as prof:
            # 对输入数据 x 进行模型推断
            y_hat = model(x)
            # 计算预测结果与真实标签之间的交叉熵损失
            loss = torch.nn.functional.cross_entropy(
                y_hat, torch.randint(0, 10, (100,))
            )
            # 反向传播计算梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
        # 生成所有反模式的报告到当前目录下的JSON文件中，禁止在控制台打印
        report_all_anti_patterns(prof, json_report_dir=".", print_enable=False)
        try:
            # 尝试打开之前生成的JSON报告文件
            with open("./torchtidy_report.json") as f:
                # 加载JSON数据
                report = json.load(f)

            # 从报告中选择以 "test_profiler.py" 结尾的键，用于后续断言
            keys = [k for k in report.keys() if k.endswith("test_profiler.py")]
            # 断言仅有一个报告文件名以 "test_profiler.py" 结尾
            self.assertEqual(len(keys), 1, f"{keys}")
            # 获取第一个符合条件的报告条目
            entry = report[keys[0]]

            # 断言报告条目不为空
            self.assertTrue(len(entry) > 0)
            # 期望的字段顺序为 ["line_number", "name", "url", "message"]
            expected_fields = sorted(["line_number", "name", "url", "message"])
            # 遍历报告条目中的事件
            for event in entry:
                # 实际事件字段的排序
                actual_fields = sorted(event.keys())
                # 断言实际字段与期望字段一致
                self.assertEqual(expected_fields, actual_fields)
        finally:
            # 最终删除生成的JSON报告文件
            os.remove("torchtidy_report.json")

    # 根据不同条件跳过测试，仅在非ARM64架构且为Linux系统时有效
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "x86 linux only cpp unwinding")
# 如果当前脚本作为主程序执行（而不是被导入为模块），则运行测试函数
if __name__ == "__main__":
    # 调用 run_tests() 函数来执行测试用例
    run_tests()
```