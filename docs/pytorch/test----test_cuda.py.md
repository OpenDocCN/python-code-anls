# `.\pytorch\test\test_cuda.py`

```
# Owner(s): ["module: cuda"]

# 导入必要的模块和库
import collections
import contextlib
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from copy import deepcopy
from itertools import product
from random import randint

# 导入 PyTorch 相关模块
import torch
import torch.cuda

# 导入内部测试工具和函数
from torch import inf, nan
from torch.cuda._memory_viz import (
    _profile_to_snapshot,
    profile_plot,
    segment_plot,
    trace_plot,
)
from torch.testing._internal.autocast_test_lists import AutocastTestLists
from torch.testing._internal.common_cuda import (
    _create_scaling_case,
    _get_torch_cuda_version,
    TEST_CUDNN,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyNativeDeviceTypes,
)
from torch.testing._internal.common_optimizers import optim_db, optims, TensorTracker
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    gcIfJetson,
    get_cycles_per_ms,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_FBCODE,
    IS_JETSON,
    IS_LINUX,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    NO_MULTIPROCESSING_SPAWN,
    NoTest,
    parametrize,
    run_tests,
    serialTest,
    skipCUDAMemoryLeakCheckIf,
    skipCUDANonDefaultStreamIf,
    skipIfRocm,
    slowTest,
    subtest,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TEST_NUMPY,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.viz._cycles import observe_tensor_cycles

# load_tests 函数用于在 Sandcastle 平台上自动筛选测试，以进行分片。这行代码用于消除 Flake 警告
load_tests = load_tests

# 如果 CUDA 不可用，则打印错误信息并将 TestCase 设置为 NoTest 类
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 尝试导入 torchvision 模块，如果导入失败，则标记 HAS_TORCHVISION 为 False
try:
    import torchvision.models  # noqa: F401
    from torchvision.models import resnet18  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 如果没有 torchvision，则跳过相关测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 设置一些测试相关的标志和变量
TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_BF16 = False
TEST_PYNVML = not torch.cuda._HAS_PYNVML

# 如果 CUDA 可用，则根据 GPU 总内存设置不同的测试标志
if TEST_CUDA:
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
    TEST_BF16 = torch.cuda.is_bf16_supported()

# 初始化一个变量用于存储每毫秒循环次数
_cycles_per_ms = None


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCuda(TestCase):
    # 设置 CUDA 内存泄漏检查和非默认流检查标志
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    # 设置测试前的初始化操作
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastTestLists(torch.device("cuda:0"))

    # 设置测试完成后的清理操作
    def tearDown(self):
        del self.autocast_lists
        super().tearDown()
    def test_pinned_memory_with_cudaregister(self):
        # 设置 CUDA 分配器的参数，启用 CUDA 主机注册并指定注册线程数为 8
        torch.cuda.memory._set_allocator_settings(
            "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
        )
        # 创建一个包含 20 个元素全为 1 的张量
        t = torch.ones(20)
        # 检查张量是否被固定在内存中
        self.assertFalse(t.is_pinned())
        try:
            # 尝试创建一个非常大的固定内存张量（1 << 21 表示 2 的 21 次方）
            pinned_t = torch.ones(1 << 21).pin_memory()
            # 断言新创建的张量确实被固定在内存中
            self.assertTrue(pinned_t.is_pinned())
            # 再次尝试创建一个更大的固定内存张量（1 << 24 表示 2 的 24 次方）
            pinned_t = torch.ones(1 << 24).pin_memory()
            # 断言新创建的张量确实被固定在内存中
            self.assertTrue(pinned_t.is_pinned())
        except RuntimeError as e:
            # 捕获 RuntimeError 异常，通常是因为某些 GPU 不支持主机和设备端相同的地址空间
            # 此时不做处理，继续执行后续代码
            pass

    def test_pinned_memory_with_cudaregister_multithread(self):
        # 定义线程数量为 4
        num_threads = 4
        # 创建包含多个线程的列表，每个线程都执行 test_pinned_memory_with_cudaregister 函数
        threads = [
            threading.Thread(target=self.test_pinned_memory_with_cudaregister)
            for t in range(num_threads)
        ]
        # 启动所有线程
        for thread in threads:
            thread.start()
        # 等待所有线程执行完毕
        for thread in threads:
            thread.join()

    def test_cudart_register(self):
        # 创建一个包含 20 个元素全为 1 的张量
        t = torch.ones(20)
        # 检查张量是否被固定在内存中
        self.assertFalse(t.is_pinned())
        # 获取 cudart 模块
        cudart = torch.cuda.cudart()
        # 将张量 t 的数据指针注册到 CUDA 主机内存中
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        # 断言注册操作返回值为 0，表示成功
        self.assertEqual(r, 0)
        # 再次检查张量是否被固定在内存中
        self.assertTrue(t.is_pinned())
        # 取消张量 t 在 CUDA 主机内存中的注册
        r = cudart.cudaHostUnregister(t.data_ptr())
        # 断言取消注册操作返回值为 0，表示成功
        self.assertEqual(r, 0)
        # 再次检查张量是否被固定在内存中
        self.assertFalse(t.is_pinned())

    def test_memory_allocation(self):
        # 执行垃圾回收
        gc.collect()
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            # 获取当前 CUDA 已分配的内存量
            prev = torch.cuda.memory_allocated()
            # 使用缓存分配器分配指定大小的内存块
            mem = torch.cuda.caching_allocator_alloc(size)
            # 断言分配后的 CUDA 内存量比之前增加，确保分配成功
            self.assertGreater(torch.cuda.memory_allocated(), prev)
        finally:
            # 最终清理工作：如果 mem 不为 None，则释放已分配的 CUDA 内存块
            if mem is not None:
                torch.cuda.caching_allocator_delete(mem)
                # 断言释放内存后 CUDA 已分配的内存量与之前相同
                self.assertEqual(torch.cuda.memory_allocated(), prev)

    def test_check_error(self):
        # 断言此调用不会引发异常
        torch.cuda.check_error(0)

        # 使用断言上下文，断言这些错误码会引发 torch.cuda.CudaError 异常，其中包含特定错误信息
        with self.assertRaisesRegex(
            torch.cuda.CudaError, "out of memory|hipErrorOutOfMemory"
        ):
            torch.cuda.check_error(2)

    def test_cuda_get_device_name(self):
        # 测试当参数为 None 时的行为
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        device_name_None = torch.cuda.get_device_name(None)
        # 断言使用 None 和当前设备索引获取到的设备名称相同
        self.assertEqual(current_device_name, device_name_None)

        # 测试未提供参数时的行为
        device_name_no_argument = torch.cuda.get_device_name()
        # 断言未提供参数时获取到的设备名称与使用当前设备索引获取到的设备名称相同
        self.assertEqual(current_device_name, device_name_no_argument)
    # 测试 CUDA 设备的能力获取函数
    def test_cuda_get_device_capability(self):
        # 测试当参数为 None 时的行为
        current_device = torch.cuda.current_device()
        current_device_capability = torch.cuda.get_device_capability(current_device)
        device_capability_None = torch.cuda.get_device_capability(None)
        # 断言当前设备的能力与使用 None 作为参数获取的能力相同
        self.assertEqual(current_device_capability, device_capability_None)

        # 测试没有参数时的行为
        device_capability_no_argument = torch.cuda.get_device_capability()
        # 断言当前设备的能力与未提供参数时获取的能力相同
        self.assertEqual(current_device_capability, device_capability_no_argument)

    # 测试 CUDA 内存耗尽的情况
    def test_out_of_memory(self):
        # 在 CUDA 设备上创建一个大小为 1024 的零张量
        tensor = torch.zeros(1024, device="cuda")

        # 根据是否启用 TEST_CUDAMALLOCASYNC，设置不同的 out of memory 错误正则表达式
        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else "Tried to allocate 800000000.00 GiB"
        )
        # 使用 assertRaisesRegex 断言尝试分配超出设备内存限制的异常
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            torch.empty(1024 * 1024 * 1024 * 800000000, dtype=torch.int8, device="cuda")

        # 再次使用 assertRaisesRegex 断言尝试分配超出设备内存限制的异常
        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(
                1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="cuda"
            )

        # 确保内存耗尽错误不会影响后续的内核执行
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    # 如果 TEST_CUDAMALLOCASYNC 或 IS_JETSON 为真，则跳过该测试
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC or IS_JETSON, "Segmentation fault (core dumped)"
    )
    @serialTest()
    def test_out_of_memory_retry(self):
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        # 获取设备 0 的总内存大小
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 根据 TEST_CUDAMALLOCASYNC 设置不同的 out of memory 错误正则表达式
        oom_regex = (
            "would exceed allowed memory"
            if TEST_CUDAMALLOCASYNC
            else "Tried to allocate"
        )
        # 计算尝试分配总内存一半大小的张量
        size = int(total_memory * 0.5)
        a = torch.empty(size, dtype=torch.int8, device="cuda")
        # 使用 assertRaisesRegex 断言尝试分配超出设备内存限制的异常
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            b = torch.empty(size, dtype=torch.int8, device="cuda")
        del a
        b = torch.empty(size, dtype=torch.int8, device="cuda")
        del b
        # 清空 CUDA 缓存和重置内存峰值统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    def test_copy_non_blocking(self):
        def _test_copy_non_blocking(a, b):
            # 创建一个 CUDA 事件对象
            event = torch.cuda.Event()
            # 非阻塞地将张量 b 复制到张量 a
            a.copy_(b, non_blocking=True)
            # 记录事件
            event.record()
            # 同步事件
            event.synchronize()
            # 检查张量 a 和 b 是否相等
            self.assertEqual(a, b)

        # 创建一个大小为 10MB 的全一张量并放在 CUDA 上
        x = torch.ones(10000000, dtype=torch.uint8).cuda()
        # 创建一个大小为 10MB 的全零张量，并将其固定在内存中
        y = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        # 调用内部函数 _test_copy_non_blocking 进行非阻塞拷贝测试
        _test_copy_non_blocking(x, y)

        # 创建一个大小为 10MB 的全零张量并固定在内存中
        x = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        # 创建一个大小为 10MB 的全一张量并放在 CUDA 上
        y = torch.ones(10000000, dtype=torch.uint8).cuda()
        # 调用内部函数 _test_copy_non_blocking 进行非阻塞拷贝测试
        _test_copy_non_blocking(x, y)

        # 测试固定内存数据指针与存储数据指针不相等的情况
        x_base = torch.zeros(10000000, dtype=torch.uint8).pin_memory()
        # 创建一个 x_base 的切片 x，这个切片依然固定在内存中
        x = x_base[1:]
        # 断言切片 x 和原始张量 x_base 都是固定在内存中
        self.assertTrue(x.is_pinned())
        self.assertTrue(x_base.is_pinned())
        # 断言 x_base 的数据指针和 x 的数据指针不相等
        self.assertNotEqual(x_base.data_ptr(), x.data_ptr())
        # 断言 x_base 的存储数据指针和 x 的存储数据指针相等
        self.assertEqual(x_base.storage().data_ptr(), x.storage().data_ptr())
        # 创建一个大小为 10MB-1 的全一张量并放在 CUDA 上
        y = torch.ones(10000000 - 1, dtype=torch.uint8).cuda()
        # 调用内部函数 _test_copy_non_blocking 进行非阻塞拷贝测试
        _test_copy_non_blocking(x, y)
    # 测试在非阻塞模式下进行类型转换的复制操作
    def test_copy_non_blocking_type_conversion(self):
        # 创建一个在 CUDA 设备上全为1的张量
        a = torch.ones(1, device="cuda")
        # 创建一个在 CPU 上、可固定内存的全为0的张量
        b = torch.zeros(1, device="cpu", pin_memory=True)
        # 创建一个在 CUDA 设备上的空张量，数据类型为 long
        c = torch.empty(1, device="cuda", dtype=torch.long)
        # 让 CUDA 设备休眠，时间根据当前设备的每毫秒周期数动态计算
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        # 非阻塞地将张量 a 复制到张量 b
        b.copy_(a, non_blocking=True)
        # 非阻塞地将张量 b 复制到张量 c
        c.copy_(b, non_blocking=True)
        # 断言张量 a 和 c 的值相等，忽略数据类型的精确匹配
        self.assertEqual(a, c, exact_dtype=False)

    # 用于测试非阻塞模式的张量转移操作
    @serialTest()
    def test_to_non_blocking(self):
        # 获取当前 CUDA 流
        stream = torch.cuda.current_stream()

        # 内部函数，测试张量转移到非阻塞模式时的行为
        def _test_to_non_blocking(a, non_blocking, dst):
            # 同步 CUDA 设备的操作
            torch.cuda.synchronize()
            # 将0.1秒的自旋时间推送到流中，如果复制是非阻塞的，
            # 流在查询时几乎肯定是活跃的。
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            # 将张量 a 转移到目标设备 dst，根据 non_blocking 参数决定是否非阻塞
            b = a.to(device=dst, non_blocking=non_blocking)
            # 断言当前流的活跃状态与非阻塞参数的关系
            self.assertEqual(stream.query(), not non_blocking)
            # 同步当前流
            stream.synchronize()
            # 断言张量 a 和 b 的值相等
            self.assertEqual(a, b)
            # 断言张量 b 是否为固定内存（仅在非阻塞模式且目标设备为 CPU 时成立）
            self.assertTrue(b.is_pinned() == (non_blocking and dst == "cpu"))

        # 遍历所有目标设备和非阻塞参数的组合
        for dst, try_non_blocking in product(("cuda", "cpu"), (True, False)):
            # 在与目标设备相反的设备上创建源张量
            src = torch.randn(
                1000000,
                device="cuda" if dst == "cpu" else "cpu",
                pin_memory=True if dst == "cuda" else False,
            )
            # 执行内部函数，测试转移到非阻塞模式时的行为
            _test_to_non_blocking(src, try_non_blocking, dst)

    # 测试默认情况下在 CPU 上阻塞的张量转移操作
    def test_to_cpu_blocking_by_default(self):
        # 在 CUDA 设备上创建随机张量
        src = torch.randn(1000000, device="cuda")
        # 同步 CUDA 设备的操作
        torch.cuda.synchronize()
        # 让 CUDA 设备休眠，时间根据当前设备的每毫秒周期数动态计算
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        # 将张量 src 转移到 CPU 设备（默认阻塞）
        dst = src.to(device="cpu")
        # 断言当前 CUDA 流的查询结果为 True（已完成操作）
        self.assertEqual(torch.cuda.current_stream().query(), True)
        # 断言张量 src 和 dst 的值相等
        self.assertEqual(src, dst)
        # 断言 dst 不是固定内存
        self.assertFalse(dst.is_pinned())

    # 测试带有存储的数组序列化操作
    def test_serialization_array_with_storage(self):
        # 在 CUDA 设备上创建随机张量 x 和 IntTensor y
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        # 创建包含 x、y 和它们的存储的列表 q
        q = [x, y, x, y.storage()]
        # 使用临时文件保存列表 q，并重新加载它
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        # 断言加载后的列表 q_copy 等于原始列表 q，比较时允许误差为0
        self.assertEqual(q_copy, q, atol=0, rtol=0)
        # 修改 q_copy 的第一个元素为全5
        q_copy[0].fill_(5)
        # 断言修改后的第一个元素等于第三个元素（复制前后的 x 张量）
        self.assertEqual(q_copy[0], q_copy[2], atol=0, rtol=0)
        # 断言 q_copy 的各元素类型是否符合预期
        self.assertTrue(isinstance(q_copy[0], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.FloatTensor))
        self.assertTrue(isinstance(q_copy[3], torch.storage.TypedStorage))
        self.assertTrue(isinstance(q_copy[3]._untyped_storage, torch.UntypedStorage))
        # 修改 q_copy 的第二个元素为全10
        q_copy[1].fill_(10)
        # 断言修改后的第四个元素（复制前后的 y 存储）等于 IntStorage(10).fill_(10)
        self.assertEqual(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    # 如果 TEST_CUDAMALLOCASYNC 或 TEST_WITH_ROCM 为真，则跳过此测试
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC or TEST_WITH_ROCM, "temporarily disabled for async"
    )
    @unittest.skipIf(
        _get_torch_cuda_version() >= (12, 2),
        "skipped as explicit workspace allocation is removed",
    )
    # 如果 CUDA 版本大于等于 12.2，则跳过测试，因为显式工作空间分配已移除
    def test_cublas_workspace_explicit_allocation(self):
        a = torch.randn(7, 7, device="cuda", requires_grad=False)
        # 默认工作空间大小设定为 4096 * 2 * 1024 + 16 * 8 * 1024，表示为 4096:2:16:8
        default_workspace_size = 4096 * 2 * 1024 + 16 * 8 * 1024  # :4096:2:16:8
        # 对于 Hopper GPU 期望不同的大小 (32 MiB)
        if torch.cuda.get_device_capability() == (9, 0):
            default_workspace_size = 4096 * 8 * 1024

        def check_workspace_size(inp):
            torch._C._cuda_clearCublasWorkspaces()
            start = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            with torch.no_grad():
                torch.matmul(inp, inp)
            finish = torch.cuda.memory_stats()["active_bytes.all.allocated"]
            return finish - start

        # 检查默认配置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # 检查带有错误用户配置的默认配置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "-1"
        self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)

        # 检查有效的配置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":128:8:64:16:32:32"
        self.assertTrue(abs(check_workspace_size(a) - (3072 * 1024)) < 524288)

        torch._C._cuda_clearCublasWorkspaces()

    # 测试是否允许设置和获取 TF32 加速
    def test_cublas_allow_tf32_get_set(self):
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        if skip_tf32_cublas:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            return

        orig = torch.backends.cuda.matmul.allow_tf32
        # 断言当前 TF32 设置与原始设置一致
        self.assertEqual(torch._C._get_cublas_allow_tf32(), orig)
        # 切换 TF32 设置并验证切换后状态是否正确
        torch.backends.cuda.matmul.allow_tf32 = not orig
        self.assertEqual(torch._C._get_cublas_allow_tf32(), not orig)
        # 恢复原始 TF32 设置
        torch.backends.cuda.matmul.allow_tf32 = orig
    # 测试函数：验证和设置 torch.float32 矩阵乘法精度
    def test_float32_matmul_precision_get_set(self):
        # 保存当前的 float32 矩阵乘法精度设置
        orig = torch.get_float32_matmul_precision()
        # 检查是否应该跳过 TF32 的 cuBLAS 支持设置
        skip_tf32_cublas = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
        )
        # 这段代码主要检查环境变量在测试期间的使用情况，
        # 确保在测试过程中不被其他函数覆盖而修改初始值
        if not skip_tf32_cublas:
            # 如果不跳过，则断言不允许使用 TF32
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            # 断言当前的 float32 矩阵乘法精度设置为 "highest"
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        else:
            # 如果跳过，则断言允许使用 TF32
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        
        # 遍历设置不同的 float32 矩阵乘法精度，并验证
        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        
        # 设置 float32 矩阵乘法精度为 "highest"，并验证
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        
        # 恢复原始的 float32 矩阵乘法精度设置
        torch.set_float32_matmul_precision(orig)

    # 测试函数：验证和设置 CUDA cuBLAS 是否允许 FP16 减少精度运算
    def test_cublas_allow_fp16_reduced_precision_reduction_get_set(self):
        # 保存当前设置
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        # 断言当前设置和底层 C++ 实现一致
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), orig
        )
        # 切换设置并验证
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(), not orig
        )
        # 恢复原始设置
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig

    # 测试函数：验证和设置 CUDA cuBLAS 是否允许 BF16 减少精度运算
    def test_cublas_allow_bf16_reduced_precision_reduction_get_set(self):
        # 保存当前设置
        orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        # 断言当前设置和底层 C++ 实现一致
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), orig
        )
        # 切换设置并验证
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not orig
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(), not orig
        )
        # 恢复原始设置
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig

    # 测试函数：验证和设置 CUDA cuDNN 是否允许 TF32 运算
    def test_cudnn_allow_tf32_get_set(self):
        # 使用 torch.backends.cudnn.flags 上下文管理器设置参数
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            # 断言不允许使用 cuDNN 的 TF32
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            # 断言允许使用 cuDNN 的 TF32
            self.assertTrue(torch.backends.cudnn.allow_tf32)
    # 测试类型转换功能
    def test_type_conversions(self):
        # 创建一个5x5的随机张量
        x = torch.randn(5, 5)
        # 验证类型转换后的张量类型是否为 torch.FloatTensor
        self.assertIsInstance(x.float(), torch.FloatTensor)
        # 验证 CUDA 上双精度张量的类型
        self.assertIsInstance(x.cuda().double(), torch.cuda.DoubleTensor)
        # 验证 CUDA 上单精度张量的类型
        self.assertIsInstance(x.cuda().float(), torch.cuda.FloatTensor)
        # 验证 CPU 上单精度张量的类型
        self.assertIsInstance(x.cuda().float().cpu(), torch.FloatTensor)
        # 验证 CPU 上整型张量的类型
        self.assertIsInstance(x.cuda().float().cpu().int(), torch.IntTensor)

        # 获取张量的存储对象
        y = x.storage()
        # 验证类型转换后的存储类型是否为 torch.FloatStorage
        self.assertIsInstance(y.float(), torch.FloatStorage)
        # 验证 CUDA 上双精度存储类型
        self.assertIsInstance(y.cuda().double(), torch.cuda.DoubleStorage)
        # 验证 CUDA 上单精度存储类型
        self.assertIsInstance(y.cuda().float(), torch.cuda.FloatStorage)
        # 验证 CPU 上单精度存储类型
        self.assertIsInstance(y.cuda().float().cpu(), torch.FloatStorage)
        # 验证 CPU 上整型存储类型
        self.assertIsInstance(y.cuda().float().cpu().int(), torch.IntStorage)

    # 被禁用的测试用例，由于内存不足而失败
    @unittest.skip("was disabled due to not enough memory, but actually it always fail")
    def test_arithmetic_large_tensor(self):
        # 创建一个大小为 2^30 的 CUDA 张量
        x = torch.empty(2**30, device="cuda")

        # 将张量填充为全1，验证求和结果为 2^30
        x.fill_(1)
        self.assertEqual(x.sum(), 2**30)

        # 将张量每个元素加1，验证求和结果为 2^31
        x += 1
        self.assertEqual(x.sum(), 2**31)

        # 将张量每个元素减0.5，验证求和结果为 2^29
        x.fill_(1)
        x -= 0.5
        self.assertEqual(x.sum(), 2**29)

        # 将张量每个元素乘2，验证求和结果为 2^31
        x.fill_(1)
        x *= 2
        self.assertEqual(x.sum(), 2**31)

        # 将张量每个元素除以2，验证求和结果为 2^29
        x.fill_(1)
        x /= 2
        self.assertEqual(x.sum(), 2**29)

    # 测试在 CUDA 上使用 gather 函数处理布尔张量的功能
    def test_gather_bool(self):
        # 创建一个布尔张量 t 在 CUDA 上
        t = torch.tensor([[False, True], [True, True]], device="cuda")
        # 使用 gather 函数根据索引收集数据，验证结果是否符合预期
        self.assertEqual(
            torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]], device="cuda")),
            torch.tensor([[False, False], [True, True]], device="cuda"),
        )

    # 测试手动设置随机数种子对 CUDA 设备的影响
    def test_torch_manual_seed_seeds_cuda_devices(self):
        with freeze_rng_state():
            # 创建一个全零的4x4浮点型张量在 CUDA 上
            x = torch.zeros(4, 4).float().cuda()
            # 设置全局随机种子为2，验证初始化后的随机种子值
            torch.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            # 对张量进行均匀分布初始化
            x.uniform_()
            # 再次设置随机种子为2，验证结果张量与初始化前相同
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            # 验证此时的随机种子值是否为2
            self.assertEqual(torch.cuda.initial_seed(), 2)

    # 测试手动设置 CUDA 随机数种子对比较复杂的场景的影响
    def test_manual_seed(self):
        with freeze_rng_state():
            # 创建一个全零的4x4浮点型张量在 CUDA 上
            x = torch.zeros(4, 4).float().cuda()
            # 设置 CUDA 随机种子为2，验证初始化后的随机种子值
            torch.cuda.manual_seed(2)
            self.assertEqual(torch.cuda.initial_seed(), 2)
            # 对张量进行均匀分布初始化
            x.uniform_()
            # 使用伯努利分布生成随机张量 a，设置种子后再生成张量 b，验证两者相等
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.cuda.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertEqual(x, y)
            self.assertEqual(a, b)
            # 验证此时的 CUDA 随机种子值是否为2
            self.assertEqual(torch.cuda.initial_seed(), 2)
    # 测试当设备名不正确时的异常情况
    def test_specify_improper_device_name(self):
        import os
        
        # 指定临时文件名
        fname = "tempfile.pt"
        try:
            # 断言捕获 RuntimeError 异常并检查其异常信息是否包含 "Invalid device string"
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                # 保存张量到文件中，使用新的 Zip 序列化格式
                torch.save(
                    [torch.nn.Parameter(torch.randn(10, 10))],
                    fname,
                    _use_new_zipfile_serialization=True,
                )
                # 加载文件时指定设备为 "cuda0"
                torch.load(fname, "cuda0")
        finally:
            # 最终清理：如果文件存在则删除
            if os.path.exists(fname):
                os.remove(fname)

    # 测试获取设备索引的异常情况
    def test_get_device_index(self):
        from torch.cuda._utils import _get_device_index
        
        # 断言捕获 RuntimeError 异常并检查其异常信息是否包含 "Invalid device string"
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            # 获取设备 "cuda0" 的索引
            _get_device_index("cuda0", optional=True)
        
        # 断言捕获 ValueError 异常并检查其异常信息是否包含 "Expected a cuda device"
        with self.assertRaisesRegex(ValueError, "Expected a cuda device"):
            # 创建一个 CPU 设备
            cpu_device = torch.device("cpu")
            # 获取 CPU 设备的索引
            _get_device_index(cpu_device, optional=True)

    # 测试序列化包含空元素的张量数组
    def test_serialization_array_with_empty(self):
        # 创建包含 CUDA 张量的数组 x
        x = [torch.randn(4, 4).cuda(), torch.cuda.FloatTensor()]
        # 使用临时文件
        with tempfile.NamedTemporaryFile() as f:
            # 将数组 x 保存到临时文件
            torch.save(x, f)
            f.seek(0)
            # 从文件中加载数据到 x_copy
            x_copy = torch.load(f)
        # 对比原始数组 x 和加载后的数组 x_copy 中的每个元素
        for original, copy in zip(x, x_copy):
            self.assertEqual(copy, original)
            self.assertIs(type(copy), type(original))
            self.assertEqual(copy.get_device(), original.get_device())

    # 如果条件为真，则跳过测试
    @skipCUDANonDefaultStreamIf(True)
    def test_streams(self):
        # 获取默认 CUDA 流
        default_stream = torch.cuda.current_stream()
        # 创建用户定义的 CUDA 流
        user_stream = torch.cuda.Stream()
        # 断言默认 CUDA 流与用户流不相等
        self.assertNotEqual(default_stream, user_stream)
        # 断言默认 CUDA 流的 CUDA 流索引为 0
        self.assertEqual(default_stream.cuda_stream, 0)
        # 断言用户流的 CUDA 流索引不为 0
        self.assertNotEqual(user_stream.cuda_stream, 0)
        # 切换到用户流
        with torch.cuda.stream(user_stream):
            self.assertEqual(torch.cuda.current_stream(), user_stream)
        # 查询用户流是否完成
        self.assertTrue(user_stream.query())
        # 创建固定内存的字节张量
        tensor1 = torch.ByteTensor(5).pin_memory()
        # 将 tensor1 张量异步复制到 GPU 并加 1
        tensor2 = tensor1.cuda(non_blocking=True) + 1
        # 同步默认流
        default_stream.synchronize()
        # 查询默认流是否完成
        self.assertTrue(default_stream.query())

    # 测试 CUDA 流和事件的字符串表示形式
    def test_stream_event_repr(self):
        # 获取当前 CUDA 流
        s = torch.cuda.current_stream()
        # 断言 CUDA 流的字符串表示形式中包含 "torch.cuda.Stream"
        self.assertTrue("torch.cuda.Stream" in s.__repr__())
        # 创建 CUDA 事件
        e = torch.cuda.Event()
        # 断言 CUDA 事件的字符串表示形式中包含 "torch.cuda.Event"
        self.assertTrue("torch.cuda.Event" in e.__repr__())
        # 记录事件 e 到流 s
        s.record_event(e)
        # 断言 CUDA 事件的字符串表示形式中包含 "torch.cuda.Event"
        self.assertTrue("torch.cuda.Event" in e.__repr__())

    # 测试 CUDA 事件的同步和时间测量
    def test_events(self):
        # 获取当前 CUDA 流
        stream = torch.cuda.current_stream()
        # 创建启用时间测量的 CUDA 事件
        event = torch.cuda.Event(enable_timing=True)
        # 查询事件是否完成
        self.assertTrue(event.query())
        # 创建启用时间测量的开始事件
        start_event = torch.cuda.Event(enable_timing=True)
        # 记录开始事件到当前流
        stream.record_event(start_event)
        # 暂停一段时间
        torch.cuda._sleep(int(50 * get_cycles_per_ms()))
        # 记录事件到当前流
        stream.record_event(event)
        # 查询事件是否完成
        self.assertFalse(event.query())
        # 同步事件
        event.synchronize()
        # 查询事件是否完成
        self.assertTrue(event.query())
        # 断言开始事件到结束事件之间的时间大于 0
        self.assertGreater(start_event.elapsed_time(event), 0)
    def test_generic_stream_event(self):
        # 创建一个 CUDA 流对象
        stream = torch.Stream("cuda")
        # 断言流的设备索引与当前 CUDA 设备的索引相同
        self.assertEqual(stream.device_index, torch.cuda.current_device())
        # 创建一个 CUDA 流对象，使用与前面创建的 stream 相同的参数
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        # 断言两个流的 stream_id 相同
        self.assertEqual(stream.stream_id, cuda_stream.stream_id)
        # 断言两个流的 stream_id 不同
        self.assertNotEqual(stream.stream_id, torch.cuda.current_stream().stream_id)

        # 创建两个带有计时功能的 CUDA 事件对象
        event1 = torch.Event("cuda", enable_timing=True)
        event2 = torch.Event("cuda", enable_timing=True)
        # 断言 event1 的 event_id 为 0
        self.assertEqual(event1.event_id, 0)
        # 创建两个大小为 1000 的随机张量 a 和 b
        a = torch.randn(1000)
        b = torch.randn(1000)
        # 在指定 CUDA 流中执行下列代码块
        with torch.cuda.stream(cuda_stream):
            # 将张量 a 和 b 拷贝到 CUDA 设备，非阻塞方式
            a_cuda = a.to("cuda", non_blocking=True)
            b_cuda = b.to("cuda", non_blocking=True)
            # 断言当前流的 stream_id 与 cuda_stream 的 stream_id 相同
            self.assertEqual(stream.stream_id, torch.cuda.current_stream().stream_id)
        # 在 event1 中记录 stream 的状态
        event1.record(stream)
        # 同步等待 event1 的完成
        event1.synchronize()
        # 断言 event1 查询结果为真
        self.assertTrue(event1.query())
        # 在 CUDA 设备上执行张量运算
        c_cuda = a_cuda + b_cuda
        # 在 event2 中记录当前流的状态
        event2.record()
        # 同步等待 event2 的完成
        event2.synchronize()
        # 断言 event2 查询结果为真
        self.assertTrue(event2.query())
        # 断言 event1 和 event2 的 event_id 不同
        self.assertNotEqual(event1.event_id, event2.event_id)
        # 断言 c_cuda 在 CPU 上的结果与 a + b 相等
        self.assertEqual(c_cuda.cpu(), a + b)
        # 断言 event1 和 event2 之间的经过时间大于 0
        self.assertTrue(event1.elapsed_time(event2) > 0)

    def test_record_stream(self):
        # 获取每毫秒的 CPU 周期数
        cycles_per_ms = get_cycles_per_ms()

        # 创建一个 CPU 张量并固定在内存中
        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        # 创建一个与 t 相同大小的 CUDA 张量
        result = torch.cuda.FloatTensor(t.size())
        # 创建一个新的 CUDA 流对象
        stream = torch.cuda.Stream()
        # 创建一个指针列表，用于存储临时数据的地址
        ptr = [None]

        # 执行在后台流中进行 CPU 到 GPU 的数据拷贝
        def perform_copy():
            with torch.cuda.stream(stream):
                # 将 t 拷贝到 CUDA 设备，非阻塞方式
                tmp = t.cuda(non_blocking=True)
                # 记录当前流的状态到 tmp
                ptr[0] = tmp.data_ptr()
            # 等待主流程中的当前流执行完毕
            torch.cuda.current_stream().wait_stream(stream)
            # 记录当前流的状态到 tmp
            tmp.record_stream(torch.cuda.current_stream())
            # 延迟数据拷贝
            torch.cuda._sleep(int(50 * cycles_per_ms))  # delay the copy
            # 将 tmp 拷贝到 result
            result.copy_(tmp)

        # 执行数据拷贝操作
        perform_copy()
        # 在指定 CUDA 流中执行下列代码块
        with torch.cuda.stream(stream):
            # 创建一个与 t 相同大小的新 CUDA 张量，并清零
            tmp2 = torch.cuda.FloatTensor(t.size())
            tmp2.zero_()
            # 断言 tmp2 的数据地址与 ptr[0] 不同，用于验证分配是否太快重用
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation re-used to soon"
            )

        # 断言 result 的值与预期的结果相同
        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        if not TEST_CUDAMALLOCASYNC:
            # 在本地分配器中，预期 tmp 在主流程中的数据拷贝后会在副流中被重用
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(stream):
                # 创建一个与 t 相同大小的新 CUDA 张量
                tmp3 = torch.cuda.FloatTensor(t.size())
                # 断言 tmp3 的数据地址与 ptr[0] 相同，用于验证是否重用分配
                self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not re-used")
    def test_record_stream_on_shifted_view(self):
        # 问题 #27366 的相关测试

        # 此测试检测到意外的块重新分配。为了可靠的测试，
        # 需要对分配张量的流进行隔离。分配器不会重用从另一个流分配的空闲块。
        stream_alloc = torch.cuda.Stream()
        with torch.cuda.stream(stream_alloc):
            base = torch.cuda.FloatTensor([10, 10])

        # 在偏移视图张量上记录另一个流。
        view = base[5:]
        assert view.storage_offset() > 0

        stream_record = torch.cuda.Stream()
        with torch.cuda.stream(stream_record):
            torch.cuda._sleep(int(50 * get_cycles_per_ms()))

        view.record_stream(stream_record)

        # 删除这些张量以尽快释放块。
        data_ptr = base.data_ptr()
        del base, view

        # 新张量不应分配到上述块。
        stream_alloc.synchronize()

        with torch.cuda.stream(stream_alloc):
            try_realloc = torch.cuda.FloatTensor([10, 10])

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)

    def test_noncontiguous_pinned_memory(self):
        # 问题 #3266 的相关测试
        x = torch.arange(0, 10).view((2, 5))
        self.assertEqual(x.t(), x.t().pin_memory())

    def test_caching_pinned_memory(self):
        cycles_per_ms = get_cycles_per_ms()

        # 检查删除后是否重用分配
        t = torch.FloatTensor([1]).pin_memory()
        ptr = t.data_ptr()
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertEqual(t.data_ptr(), ptr, msg="分配未被重用")

        # 检查如果被复制使用则不重用分配
        gpu_tensor = torch.cuda.FloatTensor([0])
        torch.cuda._sleep(int(1000 * cycles_per_ms))  # 延迟1秒复制
        gpu_tensor.copy_(t, non_blocking=True)
        del t
        t = torch.FloatTensor([1]).pin_memory()
        self.assertNotEqual(t.data_ptr(), ptr, msg="分配过早重用")
        self.assertEqual(list(gpu_tensor), [1])

    def test_caching_allocator_record_stream_oom(self):
        """使用 record_stream 调用延迟的分配应在 cuda_malloc_retry 中的内存不足时仍然释放。
        参见问题 #19219"""
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device="cuda")

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device="cuda")
            with torch.cuda.stream(stream):
                y += x
            # 延迟重用 `x`，直到 `stream` 中的所有操作完成
            x.record_stream(stream)
            del x

        # 我们通过分配达到设备容量的上限。为了影响未来的测试，释放任何缓存的块。
        torch.cuda.empty_cache()

    # 历史非法内存访问的测试，参见问题 #17040。
    # 测试函数：test_reduction_gpu_memory_accessing
    def test_reduction_gpu_memory_accessing():
        # 创建一个大小为 [512, 8] 的张量，元素为 1，数据类型为 float32，在 CUDA 设备上
        x = torch.ones(512, 8, dtype=torch.float32, device="cuda")
        # 沿着第一个维度对张量进行求和操作
        torch.sum(x, 0)
    
    # 测试函数：test_sum_fp16
    def test_sum_fp16():
        # 创建一个大小为 10 的张量，元素为 0，在 CUDA 设备上，数据类型为 float16
        x = torch.zeros(10, device="cuda", dtype=torch.float16)
        # 断言张量所有元素求和结果为 0
        self.assertEqual(x.sum(), 0)
    
        # 创建一个大小为 65504 的张量，元素为 1，在 CUDA 设备上，数据类型为 float16
        x = torch.ones(65504, device="cuda", dtype=torch.float16)
        # 断言张量所有元素求和结果为 65504
        self.assertEqual(x.sum(), 65504)
        # 断言张量所有元素以 float32 类型求和结果为 65504
        self.assertEqual(x.sum(dtype=torch.float32), 65504)
    
        # 创建一个大小为 65536 的张量，元素为 1，在 CUDA 设备上，数据类型为 float16
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        # 断言张量所有元素以 float32 类型求和结果为 65536
        self.assertEqual(x.sum(dtype=torch.float32), 65536)
    
        # 以概率 0.0005 在大小为 [1203611] 的张量中生成伯努利分布的值，然后转换到 CUDA 设备上，数据类型为 float16
        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        # 断言张量所有元素求和结果与原始张量在 CPU 上的求和结果一致
        self.assertEqual(x.sum().item(), a.sum().item())
    
        # 以概率 0.0005 在大小为 [100, 121, 80] 的张量中生成伯努利分布的值，然后转换到 CUDA 设备上，数据类型为 float16
        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device="cuda", dtype=torch.float16)
        # 断言张量在第 0 和第 2 维度上的求和结果，转换为 float 类型后在 CPU 上的结果与原始张量的求和结果一致
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))
    
    # 测试函数：test_mean_fp16
    def test_mean_fp16():
        # 创建一个大小为 65536 的张量，元素为 1，在 CUDA 设备上，数据类型为 float16
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        # 断言张量所有元素的均值为 1
        self.assertEqual(x.mean(), 1)
    
        # 创建一个大小为 65536 的张量，元素为 1，在 CUDA 设备上，数据类型为 float16
        x = torch.ones(65536, device="cuda", dtype=torch.float16)
        # 断言张量所有元素的均值以 float32 类型计算结果为 1
        self.assertEqual(x.mean(dtype=torch.float32), 1)
    
    # 测试函数：test_prod_large
    def test_prod_large():
        # 创建一个大小为 240000 的张量，元素为 1，在 CUDA 设备上，数据类型为 float32
        x = torch.ones(240000, device="cuda", dtype=torch.float32)
        # 断言张量所有元素的乘积为 1
        self.assertEqual(x.prod(), 1)
    
        # 使用复数类型进行测试。注意 240000 可以被 4 整除
        for dtype in [torch.cfloat, torch.cdouble]:
            # 创建一个大小为 240000 的张量，元素为 (0+1j)，在 CUDA 设备上，数据类型为给定的复数类型
            x = torch.ones(240000, device="cuda", dtype=dtype) * (0 + 1j)
            # 断言张量所有元素的乘积为 1
            self.assertEqual(x.prod(), 1)
    def test_multinomial_ext(self):
        # 测试来自较早版本 PyTorch 的两个特殊情况（Issue #4858）

        # 创建一个包含频率数据的浮点数张量，使用 CUDA 加速
        freqs = torch.cuda.FloatTensor(
            [
                0.0,
                0.0,
                0.0,
                # ...（省略了中间的频率数据）
                0.0,
                0.0,
            ]
        )

        # 设置 CUDA 随机种子
        torch.cuda.manual_seed(11042)
        # 从频率张量中进行多项式抽样，抽取1000个样本
        sample = torch.multinomial(freqs, 1000, True)
        # 断言抽样结果的最小值不为0
        self.assertNotEqual(freqs[sample].min(), 0)

        # 创建一个形状为 (3421, 2) 的零张量，使用 CUDA 加速
        p = torch.zeros(3421, 2, device="cuda", dtype=torch.float)
        # 将第二列的所有元素设置为1
        p[:, 1] = 1
        # 设置 CUDA 随机种子
        torch.cuda.manual_seed(5214)
        # 从概率张量 p 中进行多项式抽样，每次抽取一个样本
        r = torch.multinomial(p, 1)
        # 断言抽样结果的最小值不为0
        self.assertNotEqual(r.min().item(), 0)

        # 测试来自 Issue #13867 的特殊情况

        # 设置 CUDA 随机种子
        torch.cuda.manual_seed(33)
        # 创建一个包含随机数的张量，使用 CUDA 加速，并确保值大于等于0
        probs = torch.randn(1000000, device="cuda").clamp(min=0) * 3e-5
        # 从 probs 张量中进行多项式抽样，抽取1000000个样本，允许重复抽样
        samples = probs.multinomial(1000000, replacement=True)
        # 断言抽样结果的最小值大于0
        self.assertGreater(probs[samples].min().item(), 0)

    def _spawn_test_multinomial_invalid_probs_cuda(self, probs):
        import subprocess

        try:
            # 启动一个新的子进程来执行 Python 代码
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"""
import sys  # 导入系统模块sys，用于与系统交互
import torch  # 导入PyTorch库
from torch import inf, nan  # 从PyTorch中导入inf（无穷大）和nan（NaN）

try:
    with torch.random.fork_rng(devices=[0]):  # 设置随机数发生器的设备为0
        torch.multinomial(torch.tensor({probs}).to('cuda'), 2, replacement=True)  # 在CUDA设备上执行多项式抽样
        torch.cuda.synchronize()  # 同步CUDA设备上的操作
    sys.exit(-1)  # 如果成功执行上述操作，终止程序并返回-1（不应该到达此处）
except RuntimeError as e:
    sys.exit(-2)  # 如果捕获到RuntimeError，终止程序并返回-2



],
stdout=subprocess.PIPE,  # 将子进程的标准输出重定向到PIPE
stderr=subprocess.PIPE,  # 将子进程的标准错误输出重定向到PIPE
universal_newlines=True,  # 将子进程的输出以文本模式返回
)
out, err = p.communicate(timeout=10)  # 获取子进程的输出和错误输出，超时时间为10秒
p.wait(timeout=10)  # 等待子进程终止，超时时间为10秒
except subprocess.TimeoutExpired as e:
p.kill()  # 如果超时，杀死子进程
out, err = p.communicate()  # 获取被杀死的子进程的输出和错误输出



expected_messages = [
"device-side assert triggered",  # CUDA设备端触发的断言
"Assertion",  # CUDA
"HSA_STATUS_ERROR_EXCEPTION",  # ROCm
"Device-side assertion",  # ROCm设备端触发的断言
]
self.assertTrue(any(msg in out or msg in err for msg in expected_messages))  # 断言测试，检查是否有预期的消息出现在输出或错误输出中



@slowTest  # 标记为慢速测试
@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")  # 如果在ROCm平台上运行，则跳过测试
@unittest.skipIf(
NO_MULTIPROCESSING_SPAWN,
"Disabled for environments that \
don't support multiprocessing with spawn start method",
)
def test_multinomial_invalid_probs_cuda(self):  # 测试CUDA上的多项式抽样（无效概率）
self._spawn_test_multinomial_invalid_probs_cuda([1.0, -1.0, 1.0])  # 调用内部方法进行测试
self._spawn_test_multinomial_invalid_probs_cuda([1.0, inf, 1.0])  # 调用内部方法进行测试
self._spawn_test_multinomial_invalid_probs_cuda([1.0, -inf, 1.0])  # 调用内部方法进行测试
self._spawn_test_multinomial_invalid_probs_cuda([1.0, 1.0, nan])  # 调用内部方法进行测试



@staticmethod
def _mute_init():  # 静态方法：初始化，将标准错误重定向到/dev/null
os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())



def _spawn_method(self, method, arg):  # 方法：使用spawn方法执行并行测试
ctx = torch.multiprocessing.get_context("spawn")  # 获取spawn上下文
with ctx.Pool(1, initializer=self._mute_init) as pool:  # 使用进程池并初始化标准错误重定向
errors = pool.map(method, [arg])  # 映射方法到进程池中的单个进程，执行测试
for e in errors:  # 遍历所有错误
if "device-side assert triggered" not in str(e):  # 如果没有CUDA设备端触发的断言
self.fail(e)  # 测试失败



@staticmethod
def _test_index_bounds_cuda(idx):  # 静态方法：测试CUDA上的索引边界
x = torch.arange(10, device="cuda")  # 在CUDA设备上创建一个包含10个元素的张量
try:
y = x[torch.tensor([idx])]  # 尝试索引张量
return f"x[torch.tensor([{idx})]={y}"  # 返回成功索引的信息
except RuntimeError as err:
return err  # 返回RuntimeError异常



@slowTest  # 标记为慢速测试
@unittest.skipIf(
NO_MULTIPROCESSING_SPAWN,
"Disabled for environments that \
don't support multiprocessing with spawn start method",
)
@skipIfRocm
def test_index_out_of_bounds_exception_cuda(self):  # 测试CUDA上的索引超出边界异常
test_method = TestCuda._test_index_bounds_cuda  # 获取测试方法
# 测试在边界内的访问是否正常工作
self.assertEqual(
test_method(1), "x[torch.tensor([1)]=tensor([1], device='cuda:0')"
)
# 测试索引超出边界是否会导致断言
self._spawn_method(test_method, 11)  # 调用方法进行并行测试



@slowTest  # 标记为慢速测试
@unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")  # 如果内存不足，则跳过测试
@serialTest()  # 标记为串行测试
    def test_huge_index(self):
        # 创建一个大型张量 src，形状为 (15000000, 45)，存储在 CUDA 设备上，数据类型为 long
        src = torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(
            0, 2**22
        )
        # 在 CUDA 设备上生成 src 张量的随机排列索引 idx
        idx = torch.randperm(src.shape[0], device="cuda")
        # 根据随机索引 idx 对 src 进行重新排列，并存储结果在 res 中
        res = src[idx]
        # 将 src 张量先移到 CPU，然后根据 CPU 上的 idx 进行重新排列，结果存储在 res_cpu 中
        res_cpu = src.cpu()[idx.cpu()]
        # 断言两个排列后的张量 res 和 res_cpu 在 CPU 上相等
        self.assertEqual(res.cpu(), res_cpu)

    def test_min_max_inits(self):
        # 测试 THC_reduceAll 是否得到正确的索引初始化
        # 这影响 THC_reduceAll 操作在极端值时的结果
        x = torch.cuda.ByteTensor([0])
        y = torch.cuda.ByteTensor([255])
        expected = torch.cuda.LongTensor([0])[0]

        # 求张量 x 在第 0 维度上的最大值及其索引
        _, v = x.max(dim=0)
        self.assertEqual(v, expected)

        # 求张量 y 在第 0 维度上的最小值及其索引
        _, v = y.min(dim=0)
        self.assertEqual(v, expected)

    def test_nvtx(self):
        # 确保能够看到符号
        # 在 CUDA 中使用 nvtx 接口进行符号标记
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        # 使用 nvtx 接口开始一个命名的范围
        range_handle = torch.cuda.nvtx.range_start("range_start")
        # 结束之前开始的命名范围
        torch.cuda.nvtx.range_end(range_handle)

    def test_bincount_ext(self):
        # 确保 CUDA 代码覆盖率
        input_size = (100000,)
        w = torch.randn(input_size, dtype=torch.double, device="cuda")
        w_cpu = w.cpu()
        # 测试共享内存实现
        t = torch.randint(50, input_size, dtype=torch.int8, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))
        # 测试全局内存实现
        #   参见 `CUDAHistogramMemoryType` 在 SummaryOps.cu 中
        #   50000 * sizeof(int64_t) == 390 KiB，这应该超出任何已知 GPU 的共享内存
        t = torch.randint(50000, input_size, dtype=torch.int64, device="cuda")
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device="cuda")
        # 35488 * 65536 作为 int32 可能导致溢出到负值
        # 产生负的 bin 偏移量
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_tiny_half_norm_(self):
        # 创建一个张量 a，包含从 0 到 24 的数，存储在 CUDA 设备上，数据类型为 float
        a = torch.arange(25).cuda().float()
        # 将张量 a 中的所有元素除以 100000000
        a /= 100000000
        # 将张量 a 转换为半精度浮点数类型，并存储在 b 中
        b = a.half()
        # 断言张量 b 的范数大于 0
        self.assertGreater(b.norm().item(), 0)

    def test_norm_type_conversion(self):
        # 创建一个包含 65536 个元素的张量 a，存储在 CUDA 设备上，数据类型为半精度浮点数
        a = torch.ones(65536).cuda().half()
        # 断言张量 a 的 0 范数，数据类型转换为 float32，结果为 65536
        self.assertEqual(a.norm(p=0, dtype=torch.float32), 65536)

    def test_cuda_memory_leak_detection_propagates_errors(self):
        # 确保 CUDA 内存泄漏检测能够传播错误
        with self.assertRaisesRegex(
            RuntimeError, r"The size of tensor a \(3\) must match"
        ):
            with self.assertLeaksNoCudaTensors():
                # 在 CUDA 设备上创建张量 x 和 y，并相加，期望引发尺寸不匹配的运行时错误
                x = torch.randn(3, 1, device="cuda")
                y = torch.randn(2, 1, device="cuda")
                z = x + y

    @unittest.skipIf(not TEST_MEDIUM_TENSOR, "not enough memory")
    @serialTest()
    def test_cuda_kernel_loop_overflow(self):
        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        # 创建一个在 CUDA 设备上的随机张量，数据类型为半精度浮点数，大小为 2^30 + 1
        x = torch.randn(1, 1, 1, 2**30 + 1, dtype=torch.float16, device="cuda")
        # 期望值为张量 x 的最后一个有效索引处的值
        expected = x[0, 0, 0, 2**30]
        # 对 x 应用平均池化操作
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        # 同步 CUDA 设备上的所有流，确保操作完成
        torch.cuda.synchronize()
        # 断言 y 张量在最后一个有效索引处的值与预期值相等
        self.assertEqual(y[0, 0, 0, 2**30], expected)

    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @gcIfJetson
    @serialTest()
    def test_cuda_kernel_loop_overflow_large(self):
        # Make sure input.numel() > INT_MAX is handled:
        # 创建一个在 CUDA 设备上的随机张量，数据类型为半精度浮点数，大小为 2^31
        x = torch.randn(1, 1, 1, 2**31, dtype=torch.float16, device="cuda")
        # 使用断言检测是否抛出 RuntimeError，并包含 "integer out of range" 的错误信息
        with self.assertRaisesRegex(RuntimeError, "integer out of range"):
            # 对 x 应用平均池化操作
            y = torch.nn.functional.avg_pool2d(x, kernel_size=1)

        # Issue #24309: In extreme cases, the loop variable could overflow and continue
        # the kernel loop with a negative index, causing a RuntimeError (invalid write):
        # 创建一个在 CUDA 设备上的随机张量，数据类型为半精度浮点数，大小为 2^31 - 1
        x = torch.randn(1, 1, 1, 2**31 - 1, dtype=torch.float16, device="cuda")
        # 期望值为张量 x 的倒数第二个有效索引处的值
        expected = x[0, 0, 0, 2**31 - 2]
        # 对 x 应用平均池化操作
        y = torch.nn.functional.avg_pool2d(x, kernel_size=1)
        # 同步 CUDA 设备上的所有流，确保操作完成
        torch.cuda.synchronize()
        # 断言 y 张量在倒数第二个有效索引处的值与预期值相等
        self.assertEqual(y[0, 0, 0, 2**31 - 2], expected)

    # this might create a reference cycle on self...
    def _make_multiply_in_stream(self):
        # 定义一个自定义的 PyTorch 自动求导函数 MultiplyInStream
        class MultiplyInStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, val):
                # 在上下文中保存 val 和当前 CUDA 流
                ctx.val = val
                ctx.stream = torch.cuda.current_stream()
                return x * val

            @staticmethod
            def backward(ctx, grad):
                # 断言当前 CUDA 流与之前保存的流相同
                self.assertEqual(torch.cuda.current_stream(), ctx.stream)
                # 在后台流中延迟执行操作
                torch.cuda._sleep(1000 * 5000)
                return grad * ctx.val, None

        return MultiplyInStream

    @skipCUDANonDefaultStreamIf(True)
    # 定义一个同步测试函数，用于测试流操作的反向传播
    def test_streaming_backwards_sync(self):
        # 获取当前 CUDA 设备的默认流
        default_stream = torch.cuda.current_stream()
        # 创建一个新的 CUDA 流对象
        stream = torch.cuda.Stream()

        # 创建一个 MultiplyInStream 对象，用于测试
        MultiplyInStream = self._make_multiply_in_stream()

        # 在不同于 backward() 流上测试使用梯度的情况
        # 参考 "Stream semantics of backward passes"：https://pytorch.org/docs/stable/notes/cuda.html
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            # 等待当前流与默认流同步
            stream.wait_stream(default_stream)
            # 使用 MultiplyInStream 对象进行操作
            output = MultiplyInStream.apply(x, 2)
            # 计算输出的和，并执行反向传播
            output.sum().backward()
        
        # 等待默认流与当前流同步
        default_stream.wait_stream(stream)
        # 检查梯度是否正确计算
        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        # 检查当前 CUDA 流是否恢复为默认流
        self.assertEqual(torch.cuda.current_stream(), default_stream)

        # 在同一流上测试使用梯度的情况
        # 无论反向传播操作在哪个流上运行，都应该是安全的
        bwd_ambient_stream = torch.cuda.Stream()
        x = torch.randn(5, 5, device="cuda", requires_grad=True)
        with torch.cuda.stream(stream):
            # 等待当前流与默认流同步
            stream.wait_stream(default_stream)
            # 使用 MultiplyInStream 对象进行操作
            output = MultiplyInStream.apply(x, 3)
        with torch.cuda.stream(bwd_ambient_stream):
            # 等待 bwd_ambient_stream 与 stream 同步
            bwd_ambient_stream.wait_stream(stream)
            # 计算输出的和，并执行反向传播
            output.sum().backward()
            # x 首先在 "stream" 上使用，因此其 AccumulateGrad 叶子节点应在 "stream" 上运行
            # 在 backward() 结束时，应该已将 "bwd_ambient_stream" 与 "stream" 同步
            # 因此在这里使用 x.grad 应该是安全的，无需任何同步操作
            self.assertEqual(x.grad, torch.ones_like(x) * 3)
            # 检查当前 CUDA 流是否恢复为 bwd_ambient_stream
            self.assertEqual(torch.cuda.current_stream(), bwd_ambient_stream)
    def test_streaming_backwards_multiple_streams(self):
        # 创建 MultiplyInStream 实例，用于测试
        MultiplyInStream = self._make_multiply_in_stream()

        # 定义一个继承自 torch.nn.Module 的模型类
        class StreamModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 CUDA 事件和两个 CUDA 流
                self.event = torch.cuda.Event()
                self.stream0 = torch.cuda.Stream()
                self.stream1 = torch.cuda.Stream()

            def forward(self, x, x_first_use_on_ambient):
                # 如果 x_first_use_on_ambient 为 True，则克隆输入张量 x
                if x_first_use_on_ambient:
                    x0 = x.clone()
                # 等待当前 CUDA 流执行完毕
                self.stream0.wait_stream(torch.cuda.current_stream())
                self.stream1.wait_stream(torch.cuda.current_stream())
                # 进入 stream0 所代表的 CUDA 流
                with torch.cuda.stream(self.stream0):
                    # 如果 x_first_use_on_ambient 不为 True，则克隆输入张量 x
                    if not x_first_use_on_ambient:
                        x0 = x.clone()
                    # 使用 MultiplyInStream 实例对输入张量进行操作，并记录事件到当前 CUDA 流
                    y0 = MultiplyInStream.apply(x0, 2)
                    self.event.record(stream=torch.cuda.current_stream())

                # 进入 stream1 所代表的 CUDA 流
                with torch.cuda.stream(self.stream1):
                    # 使用 MultiplyInStream 实例对输入张量进行操作，并等待事件在当前 CUDA 流执行完毕
                    y1 = MultiplyInStream.apply(x, 3)
                    self.stream1.wait_event(self.event)
                    # 返回两个操作结果的和
                    return y0 + y1

        # 创建一个新的 CUDA 流
        stream = torch.cuda.Stream()

        # 针对 x_first_use_on_ambient 参数的两种取值进行测试
        for x_first_use_on_ambient in (True, False):
            # 针对不同的参数组合进行测试，包括 out_of_place 和 iters
            for out_of_place, iters in ((True, 1), (False, 1), (False, 5)):
                # 在指定的 CUDA 流中执行以下代码块
                with torch.cuda.stream(stream):
                    # 创建一个在 CUDA 设备上随机初始化的张量 x，并要求计算梯度
                    x = torch.randn(5, 5, device="cuda", requires_grad=True)
                    # 创建 StreamModel 类的实例并移动到 CUDA 设备上
                    model = StreamModel().cuda()
                    # 注册一个钩子函数，用于检查当前 CUDA 流的正确性
                    x.register_hook(
                        lambda grad: self.assertEqual(
                            torch.cuda.current_stream(),
                            stream if x_first_use_on_ambient else model.stream0,
                        )
                    )
                    # 确保模型参数的梯度为 None
                    for p in model.parameters():
                        self.assertTrue(p.grad is None)
                    # 多次执行模型的 forward 方法，并计算损失
                    for i in range(iters):
                        loss = model(x, x_first_use_on_ambient).sum()
                        # 根据 out_of_place 的值选择不同的反向传播方法
                        if out_of_place:
                            x_grad = torch.autograd.grad((loss,), (x,))[0]
                        else:
                            loss.backward()
                # 等待当前 CUDA 流执行完毕
                torch.cuda.current_stream().wait_stream(stream)

                # 根据 out_of_place 的值断言梯度的正确性
                if out_of_place:
                    self.assertEqual(x_grad, torch.ones_like(x) * 5 * iters)
                else:
                    self.assertEqual(x.grad, torch.ones_like(x) * 5 * iters)
    def test_streaming_backwards_sync_graph_root(self):
        # This function tests if bwd ops running on a side stream properly sync with the GraphRoot.
        # The potential bug it targets is a race condition. The test uses multiple trials and
        # torch.cuda._sleep such that if the race condition exists, the test will almost certainly fail,
        # but there's a chance it may spuriously pass. Passing does not guarantee the backend is bug-free,
        # but failure does guarantee there is a bug.
        
        # 创建两个 CUDA 流对象，一个用于前向和反向操作（fwd_bwd_op_stream），另一个用于环境的反向操作（bwd_ambient_stream）
        fwd_bwd_op_stream = torch.cuda.Stream()
        bwd_ambient_stream = torch.cuda.Stream()
        # 验证两个流对象不相同，以确保测试的意义
        self.assertTrue(fwd_bwd_op_stream != bwd_ambient_stream)

        size = int(1e3)

        # 在 CUDA 设备上创建张量 a 和 b，并启用梯度跟踪
        a = torch.full((size,), 2.0, device="cuda", requires_grad=True)
        b = torch.full((size,), 3.0, device="cuda", requires_grad=True)

        # 对于每次试验，共进行 5 次
        for trial in range(5):
            # 同步所有 CUDA 核心，确保前一步操作完成
            torch.cuda.synchronize()
            # 清空张量的梯度
            a.grad = b.grad = None
            
            # 将计算流设置为 fwd_bwd_op_stream，执行张量乘法操作 c = a * b
            with torch.cuda.stream(fwd_bwd_op_stream):
                c = a * b

            # 将计算流设置为 bwd_ambient_stream，执行如下操作
            with torch.cuda.stream(bwd_ambient_stream):
                torch.cuda.synchronize()
                # 在 bwd_ambient_stream 上运行长时间的虚拟内核，延迟填充 grad
                torch.cuda._sleep(int(50 * get_cycles_per_ms()))
                # 使用浮点数值填充 grad 张量
                grad = torch.full((size,), float(trial + 1), device="cuda")

                # 在 fwd_bwd_op_stream 上运行反向传播操作，确保 bwd 操作在消耗 grad 之前与 bwd_ambient_stream 同步
                torch.autograd.backward(tensors=c, grad_tensors=grad)

                # 执行同步操作，以解决 https://github.com/pytorch/pytorch/issues/47028 的问题
                torch.cuda.synchronize()
                # 使用 torch.no_grad() 上下文验证梯度计算结果
                with torch.no_grad():
                    self.assertEqual(a.grad, grad * b)
                    self.assertEqual(b.grad, grad * a)
    def test_streaming_backwards_callback(self):
        # 测试自动求导回调是否正确同步 leaf streams 和包围 backward() 的用户可见流。
        # 如果测试失败，首先怀疑在 torch/csrc/autograd/engine.cpp 中调用 "final_callbacks_" 的同步逻辑。
        
        MultiplyInStream = self._make_multiply_in_stream()

        size = int(1e3)
        # 在 CUDA 设备上创建大小为 size 的全 1 张量 a 和 b，并且要求梯度计算
        a = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)
        b = torch.full((size,), 1, device="cuda", dtype=torch.float, requires_grad=True)

        # 创建三个 CUDA 流对象 s0, s1, s2
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        # 创建一个空列表 stash 用于存储数据
        stash = []

        # 设置 leaf streams 的复杂结构
        # 等待当前流与 s0 同步，并在 s0 流中执行 MultiplyInStream 的操作
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            c = MultiplyInStream.apply(a, 2)

        # 等待当前流与 s1 同步，并在 s1 流中执行 MultiplyInStream 的操作
        s1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s1):
            d = MultiplyInStream.apply(b, 3)
            # 等待 s0 流与当前流同步，并在 s1 流中计算 e = c * d
            s1.wait_stream(s0)
            e = c * d

            # 定义一个函数 clone_leaf_grads，将 a 和 b 的梯度克隆并添加到 stash 中
            def clone_leaf_grads():
                stash.append(a.grad.clone())
                stash.append(b.grad.clone())

            # 在 e 上注册钩子函数，用于安装回调函数
            e.register_hook(
                lambda grad: torch.autograd.Variable._execution_engine.queue_callback(
                    clone_leaf_grads
                )
            )

        # 等待 s1 流与 s2 流同步，并在 s2 流中执行 e.sum().backward()
        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            e.sum().backward()
            # 自动求导引擎应该会同步 s2 与所有 leaf streams，然后在 s2 上运行回调函数 clone_leaf_grads。
            # 如果这些步骤正确执行，安全地检查 stash 中克隆的梯度值应该是安全的。
            self.assertEqual(stash[0], torch.full_like(a, 6))
            self.assertEqual(stash[1], torch.full_like(b, 6))
    # 定义一个测试方法，用于验证 CUDA 下异步断言的行为
    def test_fixed_cuda_assert_async(self):
        # 断言空张量在 CUDA 设备上会引发 RuntimeError，指定异常信息为 "Boolean value of Tensor with no values is ambiguous"
        with self.assertRaisesRegex(
            RuntimeError, "Boolean value of Tensor with no values is ambiguous"
        ):
            # 调用 torch._assert_async 方法，传入空张量作为参数
            torch._assert_async(torch.tensor([], device="cuda"))
        
        # 断言有多个值的张量在 CUDA 设备上会引发 RuntimeError，指定异常信息为 "Boolean value of Tensor with more than one value is ambiguous"
        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            # 调用 torch._assert_async 方法，传入包含两个零值的张量作为参数
            torch._assert_async(torch.tensor([0, 0], device="cuda"))

        # 以下是一些单独的断言测试
        # 检查整数张量在 CUDA 设备上是否通过断言
        torch._assert_async(torch.tensor(1, device="cuda"))
        # 检查浮点数张量在 CUDA 设备上是否通过断言
        torch._assert_async(torch.tensor(0.1, device="cuda"))
        # 检查负浮点数张量在 CUDA 设备上是否通过断言
        torch._assert_async(torch.tensor(-0.1, device="cuda"))
        # 检查布尔值张量在 CUDA 设备上是否通过断言
        torch._assert_async(torch.tensor(True, device="cuda"))
        # 检查复数张量在 CUDA 设备上是否通过断言
        torch._assert_async(torch.tensor(0 + 0.1j, device="cuda"))

        # 准备一组失败的断言语句列表
        fail_stmts = [
            "torch._assert_async(torch.tensor(0, device='cuda'))",
            "torch._assert_async(torch.tensor(0.0, device='cuda'))",
            "torch._assert_async(torch.tensor(False, device='cuda'))",
            "torch._assert_async(torch.tensor(0 + 0j, device='cuda'))",
        ]

        # 导入 subprocess 模块，用于执行外部命令
        import subprocess

        # 遍历失败的断言语句列表
        for stmt in fail_stmts:
            # 使用 self.subTest 方法为每个失败的断言语句创建子测试
            with self.subTest(stmt=stmt):
                # 调用 subprocess.call 执行外部命令，运行给定的断言语句
                r = subprocess.call(
                    [
                        sys.executable,
                        "-c",
                        f"""\
import torch
# 导入 PyTorch 模块

torch.cuda.synchronize()
# 在 CUDA 上同步所有设备

"""
    Test case for CUDA and threading interactions with PyTorch.

    This test case checks for potential race conditions in multi-threaded CUDA operations.
    It creates multiple threads, each performing matrix multiplication and division on CUDA tensors.

    Note: This test involves CUDA streams and synchronization to manage parallel operations
          and ensure correct execution on GPUs.
"""

@unittest.skipIf(TEST_CUDAMALLOCASYNC, "FAIL")
def test_cublas_multiple_threads_same_device():
    # Note, these parameters should be very carefully tuned
    # Too small number makes it hard for the racing condition
    # to happen, while too large number sometimes cause hang
    size = 1024
    num_threads = 2
    trials = 3
    test_iters = 100

    weight = torch.ones((size, size), device="cuda")
    results = {}
    barrier = threading.Barrier(num_threads)

    def _worker(t):
        my_stream = torch.cuda.Stream()
        # Hard sync so we don't need to worry about creating and using tensors
        # across streams or the fact that default streams are thread-local.
        # Those issues are not the target of this test.
        torch.cuda.synchronize()
        # Line up threads to increase likelihood of race conditions.
        barrier.wait()
        with torch.cuda.stream(my_stream):
            for i in range(test_iters):
                # If all threads are sharing the same cublas handle,
                # the following sequence may occur:
                # thread 0 calls cublasSetStream()
                # thread 1 calls cublasSetStream()
                # thread 0 launches its raw gemm, which it thinks is in
                #          its own stream, but is actually in thread 1's stream.
                # thread 0 enqueues its div_, which IS is its own stream,
                #          but actually now races with its gemm.
                results[t] = torch.mm(results[t], weight)
                results[t].div_(float(size))
        torch.cuda.synchronize()

    for _ in range(trials):
        for t in range(num_threads):
            results[t] = torch.ones((size, size), device="cuda")

        threads = [
            threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for t in range(num_threads):
            self.assertEqual(results[t].sum().item(), size * size)

# Test is flaky on Windows (https://github.com/pytorch/pytorch/issues/57401)
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows (see issue 57401)")
@unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
@skipIfRocm
    def test_cudnn_multiple_threads_same_device(self):
        # 测试在同一设备上多线程中懒创建和重用每个线程的 cudnn 句柄逻辑。
        # 如果测试失败，可能表明在 aten/src/ATen/cudnn/Handles.cpp 中的逻辑存在问题。

        # 创建一个形状为 (1, 1, 2, 2) 的全一张量，放置在 CUDA 设备上
        weight = torch.ones((1, 1, 2, 2), device="cuda")

        # 存储测试结果的字典
        results = {}

        # 线程数量和测试次数
        num_threads = 2
        trials = 3
        # 每个线程执行的迭代次数
        test_iters = 1000
        # 创建一个线程同步的屏障，等待所有线程就绪后同时开始
        barrier = threading.Barrier(num_threads)

        # 使用启用了 cudnn 的 CUDA 后端上下文
        with torch.backends.cudnn.flags(enabled=True):

            def _worker(t):
                # 每个线程创建自己的 CUDA 流
                my_stream = torch.cuda.Stream()
                # 强制同步，确保我们不必担心跨流创建和使用张量的问题，
                # 或者默认流是线程本地的问题。这些问题不是此测试的目标。
                torch.cuda.synchronize()
                # 等待所有线程就绪，增加竞争条件的可能性
                barrier.wait()
                # 将操作绑定到当前线程的 CUDA 流
                with torch.cuda.stream(my_stream):
                    for _ in range(test_iters):
                        # 如果所有线程共享同一个 cudnn 句柄，
                        # 下面的顺序可能发生竞争条件：
                        # 线程 0 调用 setCuDNNStreamToCurrent()
                        # 线程 1 调用 setCuDNNStreamToCurrent()
                        # 线程 0 启动其原始卷积，它认为是在自己的流中，
                        #          但实际上是在线程 1 的流中。
                        # 线程 0 将其 div_ 操作排队到自己的流中，
                        #          但现在与其卷积操作竞争。
                        results[t] = torch.nn.functional.conv2d(
                            results[t], weight, padding=0
                        )
                        results[t].div_(4.0)
                torch.cuda.synchronize()

            # 执行多次试验
            for _ in range(trials):
                for t in range(num_threads):
                    # 初始化每个线程的结果张量
                    results[t] = torch.ones((1, 1, 2048, 2048), device="cuda")

                # 创建线程列表
                threads = [
                    threading.Thread(target=_worker, args=(t,))
                    for t in range(num_threads)
                ]

                # 启动所有线程
                for thread in threads:
                    thread.start()
                # 等待所有线程完成
                for thread in threads:
                    thread.join()

                # 断言每个线程结果的总和是否符合预期值
                for t in range(num_threads):
                    self.assertEqual(
                        results[t].sum().item(),
                        (2048 - test_iters) * (2048 - test_iters),
                    )
    # 定义一个测试函数，用于测试在同一设备上多线程使用 cusparse 的情况
    def test_cusparse_multiple_threads_same_device(self):
        # 设置稀疏矩阵的大小
        size = 1024
        # 线程数
        num_threads = 2
        # 每个线程运行的试验次数
        trials = 3
        # 每个试验中的迭代次数
        test_iters = 500

        # 定义一个函数，生成稀疏矩阵，元素为1
        def ones_sparse(size):
            # 在 CUDA 设备上创建从 0 到 size-1 的整数张量
            a = torch.arange(size, device="cuda")
            # 计算 a 和 a 的笛卡尔积，转置并生成稀疏矩阵的坐标
            indices = torch.cartesian_prod(a, a).t()
            # 在 CUDA 设备上创建元素值全为1的稀疏 COO 张量
            values = torch.ones(size * size, device="cuda")
            return torch.sparse_coo_tensor(indices, values)

        # 生成稀疏矩阵 weight
        weight = ones_sparse(size)
        # 用于存储每个线程的计算结果
        results = {}
        # 创建一个线程同步的屏障，使得所有线程可以同时开始执行
        barrier = threading.Barrier(num_threads)

        # 定义每个工作线程的函数
        def _worker(t):
            # 在 CUDA 设备上创建一个新的流
            my_stream = torch.cuda.Stream()
            # 硬同步，确保不需要担心在流之间创建和使用张量
            # 跨流或默认流是线程本地的问题。这些问题不是此测试的目标。
            torch.cuda.synchronize()
            # 等待所有线程就绪，以增加竞争条件的可能性
            barrier.wait()
            # 在新创建的流上执行操作
            with torch.cuda.stream(my_stream):
                for i in range(test_iters):
                    # 如果所有线程共享相同的 cublas 句柄，
                    # 下面的顺序可能发生竞争条件：
                    # 线程0调用 cublasSetStream()
                    # 线程1调用 cublasSetStream()
                    # 线程0启动其原始 gemm，它认为在自己的流中，但实际上在线程1的流中。
                    # 线程0排队其 div_，它确实在自己的流中，但实际上现在与其 gemm 竞争。
                    results[t] = weight.mm(results[t])
                    results[t].div_(float(size))
            torch.cuda.synchronize()

        # 进行多次试验
        for _ in range(trials):
            # 对每个线程初始化结果
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="cuda")

            # 创建线程列表
            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            # 启动所有线程
            for thread in threads:
                thread.start()
            # 等待所有线程结束
            for thread in threads:
                thread.join()

            # 检查每个线程计算的结果是否符合预期
            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    # 定义一个函数，用于在自动类型转换外执行操作
    def _run_autocast_outofplace(
        self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None
    ):
        # 省略部分代码

    # 定义一个函数，根据操作参数可能包含的关键字参数返回相应的参数
    def args_maybe_kwargs(self, op_with_args):
        if len(op_with_args) == 2:
            return op_with_args[0], op_with_args[1], {}
        else:
            return op_with_args[0], op_with_args[1], op_with_args[2]

    # 如果未启用 CUDNN，跳过该测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试使用自动类型转换（autocast）对 Torch 中的 FP16 进行功能验证
    def test_autocast_torch_fp16(self):
        # 启用 CuDNN 并设置为确定性模式
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历包含在 self.autocast_lists.torch_fp16 中的操作和参数组合
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                # 如果 op_with_args 包含三个元素且第三个元素为 True，跳过测试（用于 TEST_WITH_ROCM）
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                # 如果不跳过测试
                if not skip_test:
                    # 运行 out-of-place 的自动类型转换操作，使用 torch.float16
                    self._run_autocast_outofplace(op, args, torch.float16)

    # 如果未安装测试，则跳过测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试使用自动类型转换（autocast）对 Torch 中的 BF16 进行功能验证
    def test_autocast_torch_bf16(self):
        # 启用 CuDNN 并设置为确定性模式
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历包含在 self.autocast_lists.torch_fp16 中的操作和参数组合
            for op_with_args in self.autocast_lists.torch_fp16:
                skip_test = False
                op, args = op_with_args[0], op_with_args[1]
                # 如果 op_with_args 包含三个元素且第三个元素为 True，跳过测试（用于 TEST_WITH_ROCM）
                if len(op_with_args) == 3:
                    skip_test = op_with_args[2]  # TEST_WITH_ROCM
                # 判断是否应该因 CuDNN 错误而报错
                should_error_from_cudnn = "cudnn" in op and (
                    "TORCH_CUDNN_V8_API_DISABLED" in os.environ
                    and int(os.environ["TORCH_CUDNN_V8_API_DISABLED"])
                    or torch.cuda.get_device_capability() < (8, 0)
                )
                should_error_from_not_implemented = should_error_from_cudnn
                # 如果不跳过测试
                if not skip_test:
                    # 如果应该因未实现而报错
                    if should_error_from_not_implemented:
                        # 使用 assertRaises 检查 RuntimeError，确保 op 不支持 bfloat16
                        with self.assertRaises(
                            RuntimeError,
                            msg=str(op) + " should not be supported for bfloat16!",
                        ):
                            self._run_autocast_outofplace(op, args, torch.bfloat16)
                    else:
                        # 如果当前设备支持 BF16
                        if torch.cuda.is_bf16_supported():
                            # 运行 out-of-place 的自动类型转换操作，使用 torch.bfloat16
                            self._run_autocast_outofplace(op, args, torch.bfloat16)
                        else:
                            # 使用 assertRaisesRegex 检查 RuntimeError，确保设备不支持 bfloat16
                            with self.assertRaisesRegex(
                                RuntimeError, "Device does not support bfloat16"
                            ):
                                self._run_autocast_outofplace(op, args, torch.bfloat16)

    # 如果未安装测试，则跳过测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试使用自动类型转换（autocast）对 Torch 中的 FP32 进行功能验证
    def test_autocast_torch_fp32(self):
        # 遍历包含在 self.autocast_lists.torch_fp32 中的操作和参数组合
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            # 运行 out-of-place 的自动类型转换操作，使用 torch.float32，可能包含附加关键字参数
            self._run_autocast_outofplace(
                op, args, torch.float32, add_kwargs=maybe_kwargs
            )

    # 如果未安装测试，则跳过测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试使用自动类型转换（autocast）对 Torch 中需要自动类型转换提升的操作进行验证
    def test_autocast_torch_need_autocast_promote(self):
        # 遍历包含在 self.autocast_lists.torch_need_autocast_promote 中的操作和参数组合
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            # 运行 out-of-place 的自动类型转换操作，使用 torch.float32
            self._run_autocast_outofplace(op, args, torch.float32)
    # 测试自动类型转换（autocast）对于 torch_expect_builtin_promote 的情况
    def test_autocast_torch_expect_builtin_promote(self):
        # 遍历 self.autocast_lists.torch_expect_builtin_promote 中的操作、参数和输出类型
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            # 运行自动类型转换的 out-of-place 操作，期望输出类型为 out_type
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    # 如果没有测试 CUDNN 的支持，则跳过测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 nn_fp16 的情况
    def test_autocast_nn_fp16(self):
        # 设置 CUDNN 后端标志以启用，并确保结果的确定性
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历 self.autocast_lists.nn_fp16 中的操作和参数
            for op, args in self.autocast_lists.nn_fp16:
                # 运行 nn 模块中的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float16
                self._run_autocast_outofplace(
                    op, args, torch.float16, module=torch._C._nn
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 nn_bf16 的情况
    def test_autocast_nn_bf16(self):
        # 设置 CUDNN 后端标志以启用，并确保结果的确定性
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历 self.autocast_lists.nn_fp16 中的操作和参数
            for op, args in self.autocast_lists.nn_fp16:
                # 如果当前设备支持 bfloat16
                if torch.cuda.is_bf16_supported():
                    # 运行 nn 模块中的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.bfloat16
                    self._run_autocast_outofplace(
                        op, args, torch.bfloat16, module=torch._C._nn
                    )
                else:
                    # 否则，抛出 RuntimeError，表明设备不支持 bfloat16
                    with self.assertRaisesRegex(
                        RuntimeError, "Device does not support bfloat16"
                    ):
                        # 尝试运行 nn 模块中的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.bfloat16
                        self._run_autocast_outofplace(
                            op, args, torch.bfloat16, module=torch._C._nn
                        )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 nn_fp32 的情况
    def test_autocast_nn_fp32(self):
        # 遍历 self.autocast_lists.nn_fp32 中的操作和参数
        for op, args in self.autocast_lists.nn_fp32:
            # 运行 nn 模块中的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float32
            self._run_autocast_outofplace(op, args, torch.float32, module=torch._C._nn)

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 linalg_fp16 的情况
    def test_autocast_linalg_fp16(self):
        # 设置 CUDNN 后端标志以启用，并确保结果的确定性
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历 self.autocast_lists.linalg_fp16 中的操作和参数
            for op, args in self.autocast_lists.linalg_fp16:
                # 运行 linalg 模块中的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float16
                self._run_autocast_outofplace(
                    op, args, torch.float16, module=torch._C._linalg
                )

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 methods_fp16 的情况
    def test_autocast_methods_fp16(self):
        # 设置 CUDNN 后端标志以启用，并确保结果的确定性
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历 self.autocast_lists.methods_fp16 中的操作和参数
            for op, args in self.autocast_lists.methods_fp16:
                # 运行无模块指定的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float16
                self._run_autocast_outofplace(op, args, torch.float16, module=None)

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 methods_fp32 的情况
    def test_autocast_methods_fp32(self):
        # 遍历 self.autocast_lists.methods_fp32 中的操作和参数
        for op, args in self.autocast_lists.methods_fp32:
            # 运行无模块指定的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float32
            self._run_autocast_outofplace(op, args, torch.float32, module=None)

    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试自动类型转换对于 methods_expect_builtin_promote 的情况
    def test_autocast_methods_expect_builtin_promote(self):
        # 遍历 self.autocast_lists.methods_expect_builtin_promote 中的操作、参数和输出类型
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            # 运行无模块指定的自动类型转换的 out-of-place 操作，将输入类型转换为 torch.float32，期望输出类型为 out_type
            self._run_autocast_outofplace(
                op, args, torch.float32, module=None, out_type=out_type
            )
    # 测试禁用自动类型转换情况下的操作
    def test_autocast_banned(self):
        # 使用 CUDA 自动类型转换上下文管理器
        with torch.autocast("cuda"):
            # 遍历被禁用的操作列表
            for op, args, module in self.autocast_lists.banned:
                # 断言运行时错误会被抛出
                with self.assertRaises(RuntimeError):
                    # 调用模块对象的指定操作，并传入参数
                    getattr(module, op)(*args)

    # 测试自动类型转换忽略指定类型情况下的操作
    def test_autocast_ignored_types(self):
        # 使用 CUDA 自动类型转换上下文管理器
        with torch.autocast("cuda"):
            # 遍历忽略的数据类型列表
            for ignore_type in (torch.double, torch.int32):
                # 创建在 CUDA 上的数据张量，使用忽略的数据类型
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="cuda:0")
                c_16 = torch.ones((8, 8), dtype=torch.float16, device="cuda:0")

                # 测试如果 CastPolicy::fp16 操作忽略 double 和 int 类型
                # 目前，此策略下的操作不支持整数输入。
                if ignore_type is torch.double:
                    # 断言运行时错误会被抛出
                    with self.assertRaises(RuntimeError):
                        torch.mm(a_ignore, c_16)
                    # 禁用自动类型转换后进行操作，检查数据类型不变
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    # 断言操作后的数据类型与禁用自动类型转换前一致
                    self.assertTrue(
                        torch.mm(a_ignore, b_ignore).dtype is type_no_autocast
                    )

                # 测试如果 CastPolicy::fp32 操作忽略 double 和 int 类型
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.pow(a_ignore, 2.0).dtype
                # 断言操作后的数据类型与禁用自动类型转换前一致
                self.assertTrue(torch.pow(a_ignore, 2.0).dtype is type_no_autocast)

                # 测试如果 CastPolicy::fp32_set_opt_dtype 操作忽略 double 和 int 类型
                with torch.autocast("cuda", enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                # 断言操作后的数据类型与禁用自动类型转换前一致
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # 测试如果 CastPolicy::fp32_append_dtype 操作忽略 double 和 int 类型
                # 目前，此策略下的操作不支持整数输入。
                if ignore_type is torch.double:
                    with torch.autocast("cuda", enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    # 断言操作后的数据类型与禁用自动类型转换前一致
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)
    # 定义一个测试方法，用于测试自定义自动转换功能是否启用
    def test_autocast_custom_enabled(self):
        # 定义一个自定义的 PyTorch 自动求导函数 MyMM
        class MyMM(torch.autograd.Function):
            # 前向传播函数的静态方法装饰器，指定在 CUDA 设备上使用自定义自动转换
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(ctx, a, b):
                # 断言输入张量 a 和 b 的数据类型为 torch.float32
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                # 断言当前是否启用了自动转换
                self.assertTrue(torch.is_autocast_enabled())
                # 保存前向传播所需的张量到上下文中
                ctx.save_for_backward(a, b)
                # 返回矩阵乘积 a.mm(b)
                return a.mm(b)

            # 后向传播函数的静态方法装饰器，指定在 CUDA 设备上使用自定义自动转换
            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                # 断言当前是否启用了自动转换
                self.assertTrue(torch.is_autocast_enabled())
                # 从上下文中恢复保存的张量 a 和 b
                a, b = ctx.saved_tensors
                # 计算 a 和 b 的梯度
                a_grad, b_grad = grad.mm(b.t()), a.t().mm(grad)
                # 断言计算得到的梯度数据类型为 dtype
                self.assertTrue(a_grad.dtype is dtype and b_grad.dtype is dtype)
                # 返回计算得到的梯度
                return a_grad, b_grad

        # 将 MyMM.apply 赋值给变量 mymm
        mymm = MyMM.apply

        # 在 CUDA 设备上创建随机张量 x 和 y，数据类型为 torch.float32，并且需要梯度
        x = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="cuda", dtype=torch.float32, requires_grad=True)

        # 如果 TEST_BF16 为真，则设置 dtypes 为 (torch.float16, torch.bfloat16)，否则为 (torch.float16,)
        dtypes = (torch.float16, torch.bfloat16) if TEST_BF16 else (torch.float16,)
        # 遍历 dtypes 中的每种数据类型 dtype
        for dtype in dtypes:
            # 使用自动转换，在指定的数据类型 dtype 下执行以下代码块
            with torch.cuda.amp.autocast(dtype=dtype):
                # 调用 mymm 函数计算输出
                output = mymm(x, y)
                # 断言输出的数据类型为 dtype
                self.assertTrue(output.dtype is dtype)
                # 计算损失值，这里为输出的所有元素之和
                loss = output.sum()
            # 计算损失值的梯度
            loss.backward()
    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                # 从容器中获取第二层的第一个张量
                b = container[1][0]
                # 断言张量 a 和 b 的数据类型与期望类型相同
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                # 断言自动混合精度未启用
                self.assertFalse(torch.is_autocast_enabled())
                # 保存需要反向传播的张量
                ctx.save_for_backward(a, b)
                # 返回矩阵乘积的结果
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                # 断言自动混合精度未启用
                self.assertFalse(torch.is_autocast_enabled())
                # 获取保存的张量
                a, b = ctx.saved_tensors
                # 返回反向传播的梯度
                return grad.mm(b.t()), None, None

        # 使用 MyMM.apply 创建自定义函数 mymm
        mymm = MyMM.apply

        # 创建随机张量 x，设备为 CUDA，数据类型为 torch.float16，并开启梯度跟踪
        x = torch.randn((8, 8), device="cuda", dtype=torch.float16, requires_grad=True)
        # 将一个输入张量放入嵌套容器中。由于 torch.autograd.Function 无法向非张量的前向参数返回梯度，
        # y 中包含的张量不会接收梯度。显式设置 requires_grad=False 以避免误导性地期望梯度。
        y = (
            0,
            {
                0: torch.randn(
                    (8, 8), device="cuda", dtype=torch.float16, requires_grad=False
                )
            },
        )

        # 在 CUDA 上下文中开启自动混合精度
        with torch.autocast(
            "cuda",
        ):
            # 调用自定义函数 mymm，传入参数 x, y 和期望的数据类型 torch.float32
            output = mymm(x, y, torch.float32)
            # 断言输出的数据类型为 torch.float32
            self.assertTrue(output.dtype is torch.float32)
            # 计算损失函数
            loss = output.sum()
        # 反向传播损失
        loss.backward()

        # 测试当 mymm 在非自动混合精度启用区域运行时，custom_fwd 是否成为空操作
        output = mymm(x, y, torch.float16)
        # 断言输出的数据类型为 torch.float16
        self.assertTrue(output.dtype is torch.float16)
        # 计算损失函数
        loss = output.sum()
        # 反向传播损失
        loss.backward()
    # 定义一个测试方法，用于测试自动类型转换自定义函数的弃用警告
    def test_autocast_custom_deprecated_warning(self):
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as w:

            # 定义一个继承自 torch.autograd.Function 的自定义类 MyMM
            class MyMM(torch.autograd.Function):
                # 前向传播函数的静态方法装饰器，使用了自定义的自动类型转换规则，输入被转换为 torch.float32 类型
                @staticmethod
                @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
                def forward(ctx, x, y):
                    # 保存计算图信息
                    ctx.save_for_backward(x, y)
                    # 断言自动混合精度未启用
                    self.assertFalse(torch.is_autocast_enabled())
                    return x + y

                # 反向传播函数的静态方法装饰器，用于梯度计算
                @staticmethod
                @torch.cuda.amp.custom_bwd
                def backward(ctx, grad):
                    # 恢复保存的张量信息
                    _, _ = ctx.saved_tensors
                    # 断言自动混合精度未启用
                    self.assertFalse(torch.is_autocast_enabled())
                    return grad, grad

            # 断言警告信息中包含特定的提示信息，标明自定义前向和后向传播函数已被弃用
        self.assertRegex(
            str(w[0].message), r"`torch.cuda.amp.custom_fwd\(args...\)` is deprecated."
        )
        self.assertRegex(
            str(w[1].message), r"`torch.cuda.amp.custom_bwd\(args...\)` is deprecated."
        )

        # 使用 MyMM 类的 apply 方法创建对象
        mymm = MyMM.apply
        # 生成随机张量 x 和 y
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        # 使用自动混合精度在 CUDA 上执行自定义函数
        with torch.amp.autocast("cuda"):
            output = mymm(x, y)
            loss = output.sum()
        # 计算损失的梯度
        loss.backward()

    # 定义一个测试方法，用于测试自动类型转换与 JIT 编译的兼容性
    def test_autocast_cat_jit(self):
        # 报告地址 https://github.com/pytorch/pytorch/issues/38958

        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            # 前向传播函数
            def forward(self):
                # 生成随机张量 a 和 b
                a = torch.randn(1)
                b = torch.randn(1)
                # 将 a 和 b 进行连接
                c = torch.cat((a, b), 0)
                # 将连接结果 c 进行堆叠
                d = torch.stack([c, c], 0)
                return d

        # 创建 Model 类的实例
        model = Model()
        # 对模型进行 JIT 编译
        model_jit_script = torch.jit.script(model)

        # 使用自动混合精度在 CUDA 上执行模型的前向传播
        with torch.autocast("cuda", enabled=True):
            model()
            model_jit_script()

    # 跳过测试条件为非真（即 TEST_CUDNN 为 False）的情况下执行测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 定义一个测试方法，用于检测自动类型转换内存泄漏问题
    def test_autocast_cache_leak(self):
        # 报告地址 https://github.com/pytorch/pytorch/issues/48049
        # 用于检测自动类型转换在 `torch.no_grad()` 块中是否会重新缓存相同的参数

        # 创建一个线性层并将其移到 CUDA 上
        linear = torch.nn.Linear(10, 10).to("cuda")
        # 生成随机数据张量，并移到 CUDA 上
        data = torch.randn(1, 10, device="cuda")

        # 使用自动混合精度在 CUDA 上执行以下操作
        with torch.autocast(
            "cuda",
        ):
            with torch.no_grad():
                # 使用线性层进行计算
                out = linear(data)
                # 记录第一次迭代后的内存占用
                first_iter_mem = torch.cuda.memory_allocated()
                # 进行多次迭代，观察内存是否重新分配
                for _ in range(3):
                    out = linear(data)
                # 断言第一次迭代后的内存占用与当前内存占用相同
                self.assertTrue(first_iter_mem == torch.cuda.memory_allocated())
    def test_autocast_checkpointing(self):
        # 创建一个包含三个线性层的神经网络模型，并将其放置在 CUDA 设备上
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).cuda()
        # 创建一个在 CUDA 设备上的随机张量输入，并指定数据类型为 torch.float16，同时需要梯度计算
        input = torch.rand(
            (8, 8), device="cuda", dtype=torch.float16, requires_grad=True
        )
        # 针对 reentrant 参数为 True 和 False 分别测试
        for reentrant in (True, False):
            # 在 CUDA 下启用自动混合精度
            with torch.autocast("cuda"):
                # 使用 checkpoint_sequential 函数执行模型推断，支持重入
                output = checkpoint_sequential(model, 2, input, use_reentrant=reentrant)
            # 断言输出需要梯度计算
            self.assertTrue(output.requires_grad)
            # 断言输出的数据类型为 torch.float16
            self.assertTrue(output.dtype is torch.float16)
            # 对输出进行求和并反向传播梯度
            output.sum().backward()

    def test_cuda_autocast_deprecated_warning(self):
        # 测试在使用过程中捕获 FutureWarning 异常，提醒使用新的 API
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cuda.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cuda', args...\)` instead.",
        ):
            # 在 CUDA 下启用自动混合精度
            with torch.cuda.amp.autocast():
                _ = torch.ones(10)

    @slowTest
    @unittest.skipIf(not TEST_LARGE_TENSOR, "not enough memory")
    @serialTest()
    def test_max_large_axis(self):
        # 创建一个大尺寸的张量，放置在 CUDA 设备上，数据类型为 torch.int8
        x = torch.zeros(2**32, device="cuda", dtype=torch.int8)
        # 设置最后一个元素的值为 1
        x[-1] = 1
        # 求取张量在第 0 维度上的最大值及其索引
        val, idx = x.max(0)
        # 断言最大值为 1
        self.assertEqual(val, 1)
        # 断言最大值索引为 x.shape[0] - 1
        self.assertEqual(idx, x.shape[0] - 1)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_to_numpy(self):
        # 测试将 CUDA 设备上的张量转换为 NumPy 数组是否引发 TypeError
        self.assertRaises(TypeError, lambda: torch.empty(1, device="cuda").numpy())

    def test_graph_is_current_stream_capturing(self):
        # 断言当前 CUDA 流未捕获
        self.assertFalse(torch.cuda.is_current_stream_capturing())

        if TEST_CUDA and (not TEST_WITH_ROCM):
            # 创建一个新的 CUDA 流
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                # 创建一个 CUDA 图对象
                g = torch.cuda.CUDAGraph()
                # 断言当前 CUDA 流未捕获
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                # 开始捕获当前 CUDA 流
                g.capture_begin()
                # 断言当前 CUDA 流已捕获
                self.assertTrue(torch.cuda.is_current_stream_capturing())
                # 结束捕获当前 CUDA 流
                g.capture_end()

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_graph_capture_simple(self):
        # 创建一个新的 CUDA 流
        s = torch.cuda.Stream()

        with torch.cuda.stream(s):
            # 创建一个尺寸为 (1000,) 的全零张量，放置在 CUDA 设备上
            a = torch.full((1000,), 1, device="cuda")
            # 创建一个 CUDA 图对象
            g = torch.cuda.CUDAGraph()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 开始捕获当前 CUDA 流
            g.capture_begin()
            # 执行一个简单的计算操作
            b = a
            for _ in range(10):
                b = b + 1
            # 结束捕获当前 CUDA 流
            g.capture_end()
        # 等待 CUDA 流的执行完成
        torch.cuda.current_stream().wait_stream(s)

        # 回放 CUDA 图
        g.replay()

        # 断言 b 张量所有元素之和为 11000.0
        self.assertTrue(b.sum().item() == 11000.0)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义一个测试函数，用于测试多个生成器和图形的内存统计
    def test_memory_stats_of_multiple_generators_and_graphs(self):
        
        # 清理 CUDA 缓存并收集垃圾的函数
        def clear_cuda_cache():
            gc.collect()  # 执行垃圾回收
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
        
        # 执行一个简单的图形任务，包括在 CUDA 图中捕获和执行随机数生成
        def simple_graph_task(graph):
            s = torch.cuda.Stream()  # 创建 CUDA 流对象
            with torch.cuda.stream(s):  # 使用 CUDA 流
                graph.capture_begin()  # 开始捕获图形操作
                torch.rand(1, device="cuda")  # 在 CUDA 设备上生成随机数
                graph.capture_end()  # 结束捕获
            torch.cuda.current_stream().wait_stream(s)  # 等待流完成
            graph.replay()  # 回放捕获的操作
        
        # 获取 CUDA 内存统计信息的函数
        def get_memory_stats():
            stats = torch.cuda.memory_stats()  # 获取 CUDA 内存统计信息
            num_blocks = stats["active.all.current"]  # 当前活跃内存块数
            total_size = stats["active_bytes.all.current"]  # 当前活跃内存总大小
            return num_blocks, total_size
        
        # 测试函数，用于执行多个图和生成器的测试
        def test(num_graphs, num_generators):
            baseline = get_memory_stats()  # 获取测试前的基准内存统计
            baseline_num_blocks, baseline_total_size = baseline
            
            # 分配 CUDA 图形对象
            graphs = [torch.cuda.CUDAGraph() for _ in range(num_graphs)]
            
            # 分配和管理生成器状态
            default_generator = torch.cuda.default_generators[0]  # 获取默认生成器
            generators = [default_generator.graphsafe_get_state()]  # 获取默认生成器状态
            
            # 从1开始，因为已经添加了一个状态
            for _ in range(1, num_generators):
                generators.append(default_generator.clone_state())  # 克隆生成器状态
            
            # 对每个图形和生成器状态执行简单图形任务
            for graph in graphs:
                for generator_state in generators:
                    graph.register_generator_state(generator_state)  # 注册生成器状态到图形
                simple_graph_task(graph)  # 执行简单图形任务
            
            # 在图形任务后进行断言条件验证
            num_blocks, total_size = get_memory_stats()  # 获取图形任务后的内存统计信息
            # 分配的内存块数应该与生成器数成正比
            expected_blocks_diff = 2 * num_generators
            # 每个块的大小为512，期望的总大小差异
            expected_size_diff = 2 * 512 * num_generators
            
            self.assertTrue(
                (num_blocks - baseline_num_blocks) == expected_blocks_diff,
                "Unexpected number of active blocks.",  # 断言活跃内存块数是否符合预期
            )
            self.assertTrue(
                (total_size - baseline_total_size) == expected_size_diff,
                "Unexpected total memory size.",  # 断言总内存大小是否符合预期
            )
            
            # 清理图形对象并清空 CUDA 缓存
            while graphs:
                graph = graphs.pop()  # 弹出一个图形对象
                del graph  # 删除图形对象
            
            clear_cuda_cache()  # 清空 CUDA 缓存
            
            # 断言清理后内存统计信息是否恢复到基准状态
            self.assertTrue(
                get_memory_stats() == baseline,
                "Memory stats do not match baseline after cleanup.",  # 断言清理后的内存统计是否与基准相匹配
            )
        
        # 使用不同的参数运行测试函数
        test(1, 1)  # 测试单个图形和生成器
        test(3, 2)  # 测试三个图形和两个生成器
        test(10, 20)  # 测试十个图形和二十个生成器
    # 如果不满足条件 `TEST_CUDA_GRAPH`，则跳过该测试，要求 CUDA >= 11.0 或 ROCM >= 5.3 支持图形功能
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义测试函数，验证图形捕获、重置和再捕获功能
    def test_graph_capture_reset_recapture(self):
        # 创建一个 CUDA 流对象
        s = torch.cuda.Stream()

        # 将操作绑定到 CUDA 流上
        with torch.cuda.stream(s):
            # 在 CUDA 设备上创建一个全为1的张量
            a = torch.full((1000,), 1, device="cuda")
            # 创建一个 CUDA 图对象
            g = torch.cuda.CUDAGraph()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 开始捕获 CUDA 图操作
            g.capture_begin()
            # 将变量 b 引用指向变量 a
            b = a
            # 执行一个简单的加法操作，这里仅做示例，实际操作可能更复杂
            for _ in range(10):
                b = b + 1
            # 结束捕获 CUDA 图操作
            g.capture_end()
        # 等待 CUDA 流上的操作完成
        torch.cuda.current_stream().wait_stream(s)

        # 回放 CUDA 图操作
        g.replay()

        # 断言，验证 b 的和是否为 11000.0
        self.assertTrue(b.sum().item() == 11000.0)

        # 重置 CUDA 图对象的状态
        g.reset()

        # 将操作绑定到 CUDA 流上
        with torch.cuda.stream(s):
            # 开始捕获 CUDA 图操作
            g.capture_begin()
            # 将张量 b 中所有元素填充为 2.0
            b.fill_(2.0)
            # 执行一个简单的加法操作，这里仅做示例，实际操作可能更复杂
            for _ in range(10):
                b = b + 2
            # 结束捕获 CUDA 图操作
            g.capture_end()
        # 等待 CUDA 流上的操作完成
        torch.cuda.current_stream().wait_stream(s)

        # 回放 CUDA 图操作
        g.replay()
        # 断言，验证 b 的和是否为 22000.0
        self.assertTrue(b.sum().item() == 22000.0)

        # 重置 CUDA 图对象的状态
        g.reset()
        # 删除 CUDA 图对象
        del g

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义测试函数，验证图形错误处理能力
    def test_graph_error(self):
        # 我们需要在单独的线程中运行此测试，因为触发的错误会使 CUDA 上下文处于不良状态
        script = """
import torch  # 导入PyTorch库

g = torch.cuda.CUDAGraph()  # 创建一个CUDA图对象g

try:
    g.capture_begin()  # 开始捕获CUDA图
except RuntimeError as e:
    if "CUDA graphs must be captured on a non-default stream." in str(e):
        exit(0)  # 如果捕获CUDA图时遇到特定错误，退出程序并返回0
    else:
        exit(1)  # 如果捕获CUDA图时遇到其他错误，退出程序并返回1
exit(2)  # 无论如何都退出程序并返回2（这行代码不会被执行，因为前面已经有了exit(0)或exit(1)）
"""
        try:
            a = subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                self.assertTrue(
                    False,
                    "Error raise by starting capture without a stream is not the expected one",
                )
            elif e.returncode == 2:
                self.assertTrue(
                    False,
                    "Error raised by starting capture without a stream was not caught",
                )

    @unittest.skipIf(
        (not TEST_CUDA) or TEST_WITH_ROCM or int(torch.version.cuda.split(".")[0]) < 11,
        "CUDA >= 11.0 required for graphs",
    )
    def test_graph_warn_if_has_zero_nodes(self):
        with warnings.catch_warnings(record=True) as caught:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                g.capture_begin()
                g.capture_end()
        self.assertTrue(
            any("The CUDA Graph is empty" in str(w.message) for w in caught)
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @unittest.skipIf(
        IS_JETSON, "oom reporting has issues on jetson igx due to partial nvml support"
    )
    def test_graph_capture_oom(self):
        oom_regex = (
            "would exceed allowed memory" if TEST_CUDAMALLOCASYNC else "out of memory"
        )
        with self.assertRaisesRegex(RuntimeError, oom_regex):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                torch.zeros(2**40, device="cuda")

    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @serialTest()
    # 定义一个测试方法，用于测试重复执行 CUDA 图捕获时的 cuBLAS 工作空间内存使用情况
    def test_repeat_graph_capture_cublas_workspace_memory(self):
        # 定义三个维度变量
        (x, y, z) = 1024, 512, 64
        # 在 CUDA 设备上生成随机张量 a 和 b
        a = torch.rand((x, y), device="cuda")
        b = torch.rand((y, z), device="cuda")

        # 预热阶段，执行一次矩阵乘法以加载 cuBLAS 库
        torch.mm(a, b)

        # 获取 CUDA 设备的内存信息：空闲字节数和总字节数
        free_bytes_before, total_bytes = torch.cuda.mem_get_info()
        # 计算使用 cuBLAS 前的已使用内存（GB）
        used_gb_before = (total_bytes - free_bytes_before) / 1e9

        # 开始进行重复执行 CUDA 图捕获的测试，执行 100 次
        for i in range(100):
            # 创建一个 CUDA 图对象
            torch_graph = torch.cuda.CUDAGraph()
            # 在 CUDA 图上下文中执行矩阵乘法操作
            with torch.cuda.graph(torch_graph):
                torch.mm(a, b)
            # 回放 CUDA 图以重现操作序列

        # 获取重复执行后的 CUDA 设备内存信息：空闲字节数和总字节数
        free_bytes_after, _ = torch.cuda.mem_get_info()
        # 计算使用 cuBLAS 后的已使用内存（GB）
        used_gb_after = (total_bytes - free_bytes_after) / 1e9

        # 断言：验证重复执行 cuBLAS 操作后的内存增加是否超过预期
        self.assertFalse(used_gb_before + 0.1 < used_gb_after)

    # 根据测试条件决定是否跳过此测试用例，要求 CUDA >= 11.0 或 ROCM >= 5.3 支持 CUDA 图功能
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 根据测试条件决定是否跳过此测试用例，要求 CUDA >= 11.0 或 ROCM >= 5.3 支持 CUDA 图功能
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 根据测试条件决定是否跳过此测试用例，要求 CUDA >= 11.0 或 ROCM >= 5.3 支持 CUDA 图功能
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义测试函数，测试两个连续 CUDA 图操作的情况
    def test_graph_two_successive(self):
        # 清空 CUDA 缓存，确保测试环境干净
        torch.cuda.empty_cache()

        # 定义大小为 1000 的张量和小缓冲区常量
        size = 1000
        kSmallBuffer = 2097152

        # 定义一个带临时变量的函数，对张量进行操作并返回结果
        def func_with_temps(t, val):
            x = t.clone() + val
            y = t.clone() + val
            return x + y

        # 创建一个 CUDA 流
        s = torch.cuda.Stream()

        # 遍历不同共享内存方式的情况
        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            # 创建两个 CUDA 图对象 g0 和 g1
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            # 在 CUDA 设备上创建全为 1 的张量 a
            a = torch.ones((size,), device="cuda")

            # 等待当前 CUDA 流的完成
            s.wait_stream(torch.cuda.current_stream())
            # 将以下操作放入新的 CUDA 流中执行
            with torch.cuda.stream(s):
                # 如果使用 graph_pool_handle() 共享内存，则 g0_args 中包含图池句柄
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                # 开始捕获 g0 的计算图
                g0.capture_begin(*g0_args)
                b = a.clone()
                for _ in range(5):
                    b = func_with_temps(b, 1)
                # 结束捕获 g0 的计算图
                g0.capture_end()

                # 如果使用 pool() 共享内存，则 g1_args 中包含 g0 的池对象
                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                # 开始捕获 g1 的计算图
                g1.capture_begin(*g1_args)
                for _ in range(5):
                    b = func_with_temps(b, 1)
                # 结束捕获 g1 的计算图
                g1.capture_end()
            # 等待当前 CUDA 流完成所有操作
            torch.cuda.current_stream().wait_stream(s)

            # 混合不相关的急切操作和重放操作
            c = a.clone()
            for _ in range(2):
                c = func_with_temps(c, 3)
            # 重放 g0 的计算图
            g0.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)
            # 重放 g1 的计算图
            g1.replay()
            for _ in range(2):
                c = func_with_temps(c, 3)

            # 断言验证结果
            self.assertEqual(b.sum().item(), size * 3070)
            self.assertEqual(c.sum().item(), size * 442)

            # 如果不支持 CUDAMALLOCASYNC，则进行特定的内存统计检查
            if not TEST_CUDAMALLOCASYNC:
                # 这些统计检查特定于本地分配器
                if share_mem != "Don't share":
                    # 检查未共享内存时的内存保留情况
                    self.assertEqual(
                        reserved_no_sharing  # noqa: F821
                        - torch.cuda.memory_stats()["reserved_bytes.all.current"],
                        kSmallBuffer,
                    )
                else:
                    # 记录未共享内存的保留字节数
                    reserved_no_sharing = torch.cuda.memory_stats()[
                        "reserved_bytes.all.current"
                    ]

            # 释放资源
            del a, b, c, g0, g1
            # 同步所有流操作
            torch.cuda.synchronize()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()

    @unittest.skipIf(
        (not TEST_CUDA_GRAPH)
        or IS_WINDOWS
        or (  # 在 Windows 上似乎仍然存在问题，至少在 CUDA 11.4+ 上是如此
            torch.version.cuda
            and int(torch.version.cuda.split(".")[0]) == 11
            and int(torch.version.cuda.split(".")[1]) < 4
        ),
        "Graph bindings disallow concurrent replay for CUDA < 11.4, see "
        + "https://github.com/pytorch/pytorch/pull/57556",
    )
    # 使用装饰器 @unittest.skipIf 来标记以下测试用例，条件为如果不满足 TEST_CUDA_GRAPH 条件，则跳过执行
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义一个测试函数，用于测试并发回放 CUDA 图形
    def test_graph_concurrent_replay(self):
        # 清空 CUDA 缓存，确保从干净状态开始
        torch.cuda.empty_cache()

        # 设定一个较大的尺寸以帮助暴露竞争条件
        size = 1000000

        # 定义一个带临时变量的函数，接受张量 t 和值 val 作为参数
        def func_with_temps(t, val):
            # 克隆张量 t，并加上值 val，保存为 x
            x = t.clone() + val
            # 再次克隆张量 t，并加上值 val，保存为 y
            y = t.clone() + val
            # 返回 x 和 y 的和
            return x + y

        # 创建一个 CUDA 流对象
        s = torch.cuda.Stream()

        # 针对不同的共享内存方式进行循环测试
        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            # 创建两个 CUDA 图形对象 g0 和 g1
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()

            # 创建两个 CUDA 流对象 s0 和 s1
            s0 = torch.cuda.Stream()
            s1 = torch.cuda.Stream()

            # 在 CUDA 设备上创建一个尺寸为 size 的全一张量 a
            a = torch.ones((size,), device="cuda")

            # 等待当前流上的操作完成
            s.wait_stream(torch.cuda.current_stream())
            
            # 将当前代码块中的操作放入 CUDA 流 s 中执行
            with torch.cuda.stream(s):
                # 根据共享内存方式选择 g0 的参数
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                # 开始捕获 g0 的操作
                g0.capture_begin(*g0_args)
                # 克隆张量 a 并保存为 b
                b = a.clone()
                # 多次调用 func_with_temps 函数来操作 b
                for _ in range(5):
                    b = func_with_temps(b, 1)
                # 结束捕获 g0 的操作
                g0.capture_end()

                # 根据共享内存方式选择 g1 的参数
                g1_args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                # 开始捕获 g1 的操作
                g1.capture_begin(*g1_args)
                # 克隆张量 a 并保存为 c
                c = a.clone()
                # 多次调用 func_with_temps 函数来操作 c
                for _ in range(5):
                    c = func_with_temps(c, 2)
                # 结束捕获 g1 的操作
                g1.capture_end()

            # 等待所有 CUDA 设备上的操作完成
            torch.cuda.synchronize()

            # 使用 s0 流来执行一段睡眠操作，以帮助调整 g0 和 g1 的核心执行
            with torch.cuda.stream(s0):
                torch.cuda._sleep(1000000)
                # 等待 s0 流上的操作完成
                s1.wait_stream(s0)
                # 回放 g0 的捕获操作
                g0.replay()
            # 使用 s1 流来回放 g1 的捕获操作
            with torch.cuda.stream(s1):
                g1.replay()
            
            # 等待所有 CUDA 设备上的操作完成
            torch.cuda.current_stream().wait_stream(s0)
            torch.cuda.current_stream().wait_stream(s1)

            # 如果未启用 TEST_CUDAMALLOCASYNC，并且共享内存方式不为 "Don't share"
            if (not TEST_CUDAMALLOCASYNC) and (share_mem != "Don't share"):
                # 预期并发回放会互相损坏 b 和 c 的数据
                self.assertNotEqual(b.sum().item(), size * 94)
                self.assertNotEqual(c.sum().item(), size * 156)
            else:
                # 如果要么使用本机分配器而不共享内存池，要么使用 cudaMallocAsync，忽略图形池共享提示并应始终安全
                # 预期不会发生内存损坏
                self.assertEqual(b.sum().item(), size * 94)
                self.assertEqual(c.sum().item(), size * 156)

            # 删除变量以释放资源
            del a, b, c, g0, g1
            # 同步所有 CUDA 设备上的操作
            torch.cuda.synchronize()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
    # 如果不满足 TEST_CUDA_GRAPH 条件，则跳过这个测试函数；要求 CUDA >= 11.0 或 ROCM >= 5.3 支持图操作
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义一个测试函数，用于测试三个连续的图操作
    def test_graph_three_successive(self):
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
    
        # 定义向量大小
        size = 1000
    
        # 创建 CUDA 流
        s = torch.cuda.Stream()
    
        # 对于三种共享内存的方式进行迭代
        for share_mem in ("Don't share", "via pool()", "via graph_pool_handle()"):
            # 在 CUDA 设备上创建全1向量 a
            a = torch.ones((size,), device="cuda")
    
            # 创建三个 CUDA 图对象
            g0 = torch.cuda.CUDAGraph()
            g1 = torch.cuda.CUDAGraph()
            g2 = torch.cuda.CUDAGraph()
    
            # 等待当前流完成
            s.wait_stream(torch.cuda.current_stream())
    
            # 在指定 CUDA 流上下文中执行以下代码块
            with torch.cuda.stream(s):
                # 根据共享内存的方式决定 g0 的参数
                g0_args = (
                    (torch.cuda.graph_pool_handle(),)
                    if share_mem == "via graph_pool_handle()"
                    else ()
                )
                # 开始捕获 g0 的操作
                g0.capture_begin(*g0_args)
                # 复制向量 a 为 b
                b = a.clone()
                # 计算 c = b + 1
                c = b + 1
                # 计算 d = b + 2
                d = b + 2
                # 结束捕获 g0 的操作
                g0.capture_end()
    
                # 根据共享内存的方式决定 g1 的参数
                args = (g0.pool(),) if share_mem == "via pool()" else g0_args
                # 开始捕获 g1 的操作
                g1.capture_begin(*args)
                # 计算 e = c + 3
                e = c + 3
                # 删除变量 c
                del c
                # 结束捕获 g1 的操作
                g1.capture_end()
    
                # 开始捕获 g2 的操作
                g2.capture_begin(*args)
                # 计算 f = d + 4
                f = d + 4
                # 结束捕获 g2 的操作
                g2.capture_end()
    
            # 等待流 s 完成
            torch.cuda.current_stream().wait_stream(s)
    
            # 测试按捕获顺序重放操作是否有效
            g0.replay()
            g1.replay()
            g2.replay()
    
            # 断言结果是否正确
            self.assertEqual(e.sum().item(), size * 5)
            self.assertEqual(f.sum().item(), size * 7)
    
            # 测试如果按照 g0、g2、g1 顺序重放操作，只有当它们不共享内存池时才有效
            g0.replay()
            g2.replay()
            g1.replay()
    
            # 根据条件预期的内存损坏情况进行断言
            expect_corruption = (not TEST_CUDAMALLOCASYNC) and (
                share_mem != "Don't share"
            )
            # 如果使用原生分配器且共享内存池，则 g2 的捕获可能会重用 f 中 c 的内存，导致误填充 e
            # 我们重放了 g2 然后是 g1，因此预期 g1 的捕获 "e = c + 3" 错误地填充 e 为 "f 的值 + 3"
            self.assertEqual(
                e.sum().item(), size * (7 + 3) if expect_corruption else size * 5
            )
            self.assertEqual(f.sum().item(), size * 7)
    
            # 释放变量以释放内存
            del a, b, d, e, f, g0, g1, g2
            # 同步 CUDA 设备以确保内存释放
            torch.cuda.synchronize()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
    
    @unittest.skipIf(
        (not TEST_CUDA_GRAPH) or TEST_CUDAMALLOCASYNC,
        "CUDA >= 11.0 or ROCM >= 5.3 required for graphs",
    )
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 测试图形捕获是否推迟尝试在不同流中回收使用的分配空间。参见
    # "Q. Why skip process_events if a capture might be underway?" in c10/cuda/CUDACachingAllocator.cpp

    # 清空 CUDA 缓存，确保开始时 GPU 内存干净
    torch.cuda.empty_cache()

    # 在 CUDA 设备上创建一个全零张量，分配到 GPU 内存
    potential_problem = torch.zeros((3,), device="cuda")
    # 创建另一个全零张量，分配到 GPU 内存
    a = torch.zeros((3,), device="cuda")
    # 创建三个 CUDA 流对象
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    # 创建一个 CUDA 图形对象
    g = torch.cuda.CUDAGraph()

    # 同步所有 CUDA 操作
    torch.cuda.synchronize()
    # 将后续操作放在 s0 流中执行
    with torch.cuda.stream(s0):
        # 记录当前流中的操作，为后续捕获做准备
        potential_problem.record_stream(s0)
        # 在 GPU 上执行一个长时间的计算，模拟资源占用
        torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
        # 修改 potential_problem 张量的值为全一
        potential_problem.fill_(1.0)
    # 删除 potential_problem 张量，释放 GPU 内存
    del potential_problem

    # 将后续操作放在 s1 流中执行
    with torch.cuda.stream(s1):
        # 开始捕获 CUDA 图形
        g.capture_begin()
        # 克隆张量 a 到 b
        b = a.clone()

        # 等待 s1 流上的操作完成
        s2.wait_stream(s1)
        # 将后续操作放在 s2 流中执行
        with torch.cuda.stream(s2):
            # 修改张量 b 的值为全一
            b.fill_(1.0)
            # 记录当前流中的操作，虚拟的记录操作
            b.record_stream(s2)  # dummy record_stream
            # 删除张量 b，释放 GPU 内存
            del b
        # 等待 s2 流上的操作完成
        s1.wait_stream(s2)
        # 结束捕获 CUDA 图形
        g.capture_end()
    # 同步所有 CUDA 操作
    torch.cuda.synchronize()

    # 创建一个全零张量，分配到 GPU 内存
    c = torch.zeros((3,), device="cuda")
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            subtest((False, False, True), decorators=[skipIfRocm]),
            subtest((True, False, True), decorators=[skipIfRocm]),
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[unittest.expectedFailure]),
        ],
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )


    # 根据测试条件决定是否跳过当前测试用例，条件为 CUDA 图形功能是否可用
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 参数化测试用例，对多组参数进行测试
    @parametrize(
        "with_amp,cache_enabled,allow_unused_input",
        [
            # 定义子测试用例，带有特定的修饰符（decorators）
            subtest((False, False, True), decorators=[skipIfRocm]),
            subtest((True, False, True), decorators=[skipIfRocm]),
            subtest((True, True, True), decorators=[unittest.expectedFailure]),
            subtest((False, False, False), decorators=[unittest.expectedFailure]),
        ],
        # 定义测试用例名称的生成函数
        name_fn=lambda x, y, z: "{}{}{}".format(
            {True: "with_amp", False: "without_amp"}[x],
            {True: "_cache_enabled", False: "_cache_disabled"}[y] if x else "",
            {True: "_allow_unused_input", False: "_not_allow_unused_input"}[z],
        ),
    )


    @serialTest()
    def test_graph_make_graphed_callables(
        self, with_amp, cache_enabled, allow_unused_input


    # 定义串行测试，测试图形生成可调用对象的函数
    @serialTest()
    def test_graph_make_graphed_callables(
        self, with_amp, cache_enabled, allow_unused_input


    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )


    # 根据测试条件决定是否跳过当前测试用例，条件为 CUDA 图形功能是否可用
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )


    def test_graph_make_graphed_callables_same_pool(self):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)
        models = []
        num_models = 3
        for _ in range(num_models):
            models.append(
                torch.nn.Sequential(
                    torch.nn.Linear(32, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                ).cuda()
            )
        # we will reuse the same pool for all graph captures
        mempool = torch.cuda.graph_pool_handle()
        graphed_models = []
        for model in models:
            x = torch.randn([64, 32], device="cuda")
            graphed_model = deepcopy(model)
            graphed_model = torch.cuda.make_graphed_callables(
                graphed_model, (x,), pool=mempool
            )
            graphed_models.append(graphed_model)

        for model, graphed_model in zip(models, graphed_models):
            x = torch.randn([64, 32], device="cuda")
            y = model(x)
            yg = graphed_model(x)
            l = y.norm()
            lg = yg.norm()
            l.backward()
            lg.backward()

            self.assertEqual(y, yg)
            self.assertEqual(l, lg)
            for p, pg in zip(model.parameters(), graphed_model.parameters()):
                self.assertEqual(p, pg)
                self.assertEqual(p.grad, pg.grad)
                self.assertNotEqual(p.data_ptr(), pg.data_ptr())
                self.assertNotEqual(p.grad.data_ptr, pg.grad.data_ptr)


    # 测试在相同池中生成图形调用对象的函数
    def test_graph_make_graphed_callables_same_pool(self):
        torch.manual_seed(5)
        torch.cuda.manual_seed(5)
        models = []
        num_models = 3
        for _ in range(num_models):
            models.append(
                torch.nn.Sequential(
                    torch.nn.Linear(32, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                ).cuda()
            )
        # 创建 CUDA 图形池句柄用于重用
        mempool = torch.cuda.graph_pool_handle()
        graphed_models = []
        # 对每个模型进行图形生成对象的处理
        for model in models:
            x = torch.randn([64, 32], device="cuda")
            graphed_model = deepcopy(model)
            graphed_model = torch.cuda.make_graphed_callables(
                graphed_model, (x,), pool=mempool
            )
            graphed_models.append(graphed_model)

        # 对比普通模型和图形生成模型的计算结果和梯度
        for model, graphed_model in zip(models, graphed_models):
            x = torch.randn([64, 32], device="cuda")
            y = model(x)
            yg = graphed_model(x)
            l = y.norm()
            lg = yg.norm()
            l.backward()
            lg.backward()

            self.assertEqual(y, yg)
            self.assertEqual(l, lg)
            # 检查参数及其梯度的一致性，并确保数据指针不同
            for p, pg in zip(model.parameters(), graphed_model.parameters()):
                self.assertEqual(p, pg)
                self.assertEqual(p.grad, pg.grad)
                self.assertNotEqual(p.data_ptr(), pg.data_ptr())
                self.assertNotEqual(p.grad.data_ptr, pg.grad.data_ptr)


    def _test_graphed_optimizer(
        self, steps_warmup, steps_train, optimizer_ctor, kwargs


    # 私有方法，用于测试图形优化器的功能
    def _test_graphed_optimizer(
        self, steps_warmup, steps_train, optimizer_ctor, kwargs
    ):
        # 遍历两种不同的执行情况：实际执行图计算（True）和不执行图计算（False）
        for actually_do_graphs in (True, False):
            # 生成一组随机参数列表，包括张量和标量，使用CUDA设备
            params = [torch.randn((i + 5, i + 5), device="cuda") for i in range(2)] + [
                torch.randn((), device="cuda")
            ]
            # 创建控制组参数的副本，并标记需要梯度
            params_control = [p.clone().requires_grad_() for p in params]
            # 创建图计算组参数的副本，并标记需要梯度
            params_graphed = [p.clone().requires_grad_() for p in params]

            # 生成随机梯度列表，与参数列表对应
            grads = [
                [torch.randn_like(p) for p in params]
                for _ in range(steps_warmup + steps_train)
            ]

            # 控制组测试 (capturable=False)
            # 使用指定的优化器构造函数创建优化器对象，capturable=False表示不捕获图计算
            opt = optimizer_ctor(params_control, capturable=False, **kwargs)

            # 执行优化步骤
            for i in range(steps_warmup + steps_train):
                for j, p in enumerate(params_control):
                    # 将预先生成的梯度应用到控制组参数上
                    p.grad = grads[i][j]
                opt.step()

            # 图计算测试 (capturable=True)
            # 使用指定的优化器构造函数创建优化器对象，capturable=True表示捕获图计算
            opt = optimizer_ctor(params_graphed, capturable=True, **kwargs)

            # 执行预热步骤
            for i in range(steps_warmup):
                for j, p in enumerate(params_graphed):
                    # 将预先生成的梯度应用到图计算组参数上
                    p.grad = grads[i][j]
                opt.step()

            # 如果需要执行实际的图计算
            if actually_do_graphs:
                # 创建CUDA图对象
                g = torch.cuda.CUDAGraph()
                # 在CUDA图上下文中执行优化步骤
                with torch.cuda.graph(g):
                    opt.step()

            # 执行训练步骤
            for i in range(steps_train):
                if actually_do_graphs:
                    for j, p in enumerate(params_graphed):
                        # 将预先生成的梯度复制到图计算组参数上
                        p.grad.copy_(grads[i + steps_warmup][j])
                    # 重播捕获的图计算
                    g.replay()
                else:
                    # 如果不执行图计算，仍然要确保传入capturable=True的构造函数，并执行优化步骤
                    for j, p in enumerate(params_graphed):
                        p.grad = grads[i + steps_warmup][j]
                    opt.step()

            # 断言控制组参数与图计算组参数相等
            for p_control, p_graphed in zip(params_control, params_graphed):
                self.assertEqual(p_control, p_graphed)

    # 如果不支持CUDA图计算，则跳过测试
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 如果不支持CUDA图计算，则跳过测试
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义测试方法，用于验证优化器在显式可捕获参数组的情况下的图优化行为
    def test_graph_optims_with_explicitly_capturable_param_groups(self):
        # 定义初始变量：热身周期和重放次数
        n_warmup, n_replay = 3, 2
        # 遍历不同的优化器类和是否可捕获第二参数组的情况
        for optimizer, second_param_group_capturable in product(
            (
                torch.optim.Adam,
                torch.optim.AdamW,
                torch.optim.ASGD,
                torch.optim.Adamax,
                torch.optim.NAdam,
                torch.optim.RAdam,
                torch.optim.Adadelta,
                torch.optim.RMSprop,
                torch.optim.Rprop,
            ),
            (True, False),
        ):
            # 创建两组参数及其引用
            ref_p1, param1 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            ref_p2, param2 = (
                torch.nn.Parameter(torch.ones(1, device="cuda")) for _ in range(2)
            )
            # 创建两组梯度数据
            grads1, grads2 = (
                [torch.randn_like(param1) for _ in range(n_warmup + n_replay)]
                for _ in range(2)
            )
            # 复制梯度数据以备参考
            ref_grads1, ref_grads2 = (
                [t.clone() for t in tensors] for tensors in (grads1, grads2)
            )
            # 创建参数组列表
            params = [
                {"params": [param1], "capturable": True},
                {"params": [param2], "capturable": second_param_group_capturable},
            ]
            # 初始化优化器对象及其参考对象
            opt = optimizer(params)
            opt_ = optimizer(
                [
                    {"params": [ref_p1], "capturable": False},
                    {"params": [ref_p2], "capturable": False},
                ]
            )

            # 执行热身周期和重放周期的优化步骤
            for i in range(n_warmup + n_replay):
                ref_p1.grad = ref_grads1[i]
                ref_p2.grad = ref_grads2[i]
                opt_.step()

            # 对第一组参数执行热身周期的优化步骤
            for i in range(n_warmup):
                param1.grad = grads1[i]
                param2.grad = grads2[i]
                opt.step()

            # 创建 CUDA 图对象
            g = torch.cuda.CUDAGraph()
            # 如果第二参数组不可捕获，预期会抛出 CUDA 图异常
            if not second_param_group_capturable:
                with self.assertRaisesRegex(RuntimeError, "Attempting CUDA graph"):
                    with torch.cuda.graph(g):
                        opt.step()
            else:
                # 在 CUDA 图上下文中执行优化步骤
                with torch.cuda.graph(g):
                    opt.step()

                # 对重放周期执行参数梯度复制及重放操作
                for i in range(n_replay):
                    param1.grad.copy_(grads1[n_warmup + i])
                    param2.grad.copy_(grads2[n_warmup + i])
                    g.replay()
                # 断言参数与其参考值相等
                self.assertEqual(ref_p1, param1)
                self.assertEqual(ref_p2, param2)

    # 如果不支持 CUDA 图，跳过测试
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 如果不支持 CUDA 图，跳过测试
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义一个测试方法，用于验证 CUDA 图形错误选项
    def test_cuda_graph_error_options(self):
        # 定义一个内部函数 fn，创建一个在 CUDA 设备上的大小为 2000 的全零张量，并进行加法操作
        def fn():
            x = torch.zeros([2000], device="cuda")
            y = x + x + x
            return y

        mem = None  # 初始化一个内存变量

        # 定义一个原始内存分配函数 raw_malloc
        def raw_malloc():
            global mem  # 声明要使用的全局内存变量
            mem = None  # 初始化为 None
            stream = torch.cuda.Stream()  # 创建一个 CUDA 流对象
            try:
                with torch.cuda.stream(stream):
                    # 在 CUDA 流上分配内存
                    mem = torch.cuda.caching_allocator_alloc(1024)
            except BaseException:
                if mem is None:
                    return
            try:
                # 释放 CUDA 分配的内存
                torch.cuda.caching_allocator_delete(mem)
                mem = None
                return None
            except BaseException:
                pass

        # 定义一个在 CUDA 事件上抛出异常的函数 throws_on_cuda_event，接收一个捕获错误模式参数
        def throws_on_cuda_event(capture_error_mode):
            graph = torch.cuda.CUDAGraph()  # 创建一个 CUDA 图形对象
            torch.cuda.synchronize()  # 同步 CUDA 设备
            stream = torch.cuda.Stream()  # 创建一个 CUDA 流对象
            stream.wait_stream(torch.cuda.current_stream())  # 等待当前 CUDA 流结束
            with torch.cuda.stream(stream):
                fn()  # 在 CUDA 流上执行 fn 函数
            stream.synchronize()  # 同步 CUDA 流
            torch.cuda.current_stream().wait_stream(stream)  # 等待当前 CUDA 流结束
            torch.cuda.synchronize()  # 同步 CUDA 设备
            try:
                with torch.cuda.graph(
                    graph, stream=stream, capture_error_mode=capture_error_mode
                ):
                    # 在指定的 CUDA 图形上执行 fn 函数
                    out = fn()
                    # 创建一个线程来调用 raw_malloc 函数
                    thread = threading.Thread(target=raw_malloc)
                    thread.start()
                    thread.join()
            except Exception:
                if mem is not None:
                    torch.cuda.caching_allocator_delete(mem)
                return True

            return False

        # 断言调用 throws_on_cuda_event 函数时，不会抛出 "thread_local" 或 "relaxed" 错误
        self.assertFalse(throws_on_cuda_event("thread_local"))
        self.assertFalse(throws_on_cuda_event("relaxed"))

        # 测试 "global" 模式下是否会抛出异常，这可能会导致进程损坏并使其他测试失败，因此被注释掉
        # self.assertTrue(throws_on_cuda_event("global"))

    # 根据条件是否跳过测试，需要 CUDA 版本 >= 11.0 或 ROCM 版本 >= 5.3
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 定义测试 CUDA 图形分配器是否能传播流的方法
    def test_cuda_graph_allocator_propagates_stream(self):
        # 获取当前 CUDA 内存快照中的段信息
        segments = torch.cuda.memory_snapshot()
        # 记录当前存在的段池 ID 集合
        existing_pools = {s["segment_pool_id"] for s in segments}
        # 在 CUDA 设备上创建一个大小为 10240000 的随机张量
        x = torch.randn(10240000, device="cuda")
        # 创建一个与 x 大小相同的随机张量 y
        y = torch.rand_like(x)
        # 创建一个 CUDA 图形对象 g
        g = torch.cuda.CUDAGraph()
        # 创建两个 CUDA 流对象 s0 和 s1
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())  # 等待当前 CUDA 流结束
        with torch.cuda.stream(s0):
            g.capture_begin()  # 开始捕获 CUDA 图形
            z = x + y  # 执行张量运算
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)  # 等待 s0 流完成
            w = z + y  # 执行张量运算
        s0.wait_stream(s1)  # 等待 s1 流完成
        with torch.cuda.stream(s0):
            g.capture_end()  # 结束捕获 CUDA 图形
        segments = torch.cuda.memory_snapshot()  # 获取 CUDA 内存快照
        # 查找新的段池 ID
        x = [
            s["segment_pool_id"]
            for s in segments
            if s["segment_pool_id"] not in existing_pools
        ]
        # 断言新的段池 ID 数量为 2
        self.assertEqual(len(x), 2)
        # 断言新的段池 ID 中第一个和第二个相等
        self.assertEqual(x[0], x[1])
    # 定义一个测试函数，用于测试批量归一化的统计信息收集功能
    def test_batch_norm_gather_stats(self):
        # 生成一个在 CUDA 设备上的随机张量，形状为 (1, 3, 3, 3)
        input = torch.randn(1, 3, 3, 3, device="cuda")
        # 调用 batch_norm_gather_stats 函数收集统计信息
        mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean=torch.ones(2, 3, device="cuda"),  # 给定均值
            invstd=torch.ones(2, 3, device="cuda"),  # 给定标准差的倒数
            running_mean=None,  # 运行时均值（暂未使用）
            running_var=None,  # 运行时方差（暂未使用）
            momentum=0.1,  # 动量参数
            eps=1e-5,  # epsilon 参数
            count=2,  # 统计计数
        )
        # 断言均值张量的结果为全为 1，且在 CUDA 设备上
        self.assertEqual(mean, torch.ones(3, device="cuda"))
        # 断言标准差的倒数张量的结果为全为 1，且在 CUDA 设备上
        self.assertEqual(invstd, torch.ones(3, device="cuda"))

    # 测试矩阵乘法内存使用情况
    def test_matmul_memory_use(self):
        # 定义获取当前 CUDA 设备上最大内存使用的函数
        def get_max_used():
            torch.cuda.synchronize()  # 同步 CUDA 设备
            val = torch.cuda.max_memory_allocated()  # 获取当前分配的最大内存
            torch.cuda.reset_peak_memory_stats()  # 重置内存统计的峰值
            return val

        # 生成一个在 CUDA 设备上的随机张量 a，形状为 (1, 32, 32)
        a = torch.rand(1, 32, 32, device="cuda")
        # 生成一个在 CUDA 设备上的随机张量 b，形状为 (24, 32, 1)
        b = torch.rand(24, 32, 1, device="cuda")

        get_max_used()  # 获取当前内存使用的峰值

        torch.matmul(a, b)  # 执行矩阵乘法操作

        matmul_mem = get_max_used()  # 获取执行矩阵乘法后的内存使用峰值

        a = a.expand(24, 32, 32)  # 将张量 a 扩展为形状 (24, 32, 32)
        torch.matmul(a, b)  # 再次执行矩阵乘法操作

        matmul_expand_mem = get_max_used()  # 获取扩展后执行乘法的内存使用峰值

        torch.bmm(a, b)  # 执行批量矩阵乘法操作

        bmm_mem = get_max_used()  # 获取批量乘法后的内存使用峰值

        # 断言扩展后执行乘法的内存使用峰值与普通乘法相同
        self.assertEqual(matmul_expand_mem, matmul_mem)
        # 断言批量乘法后的内存使用峰值与普通乘法相同
        self.assertEqual(bmm_mem, matmul_mem)

    @unittest.skipIf(not TEST_WITH_ROCM, "ROCm-only test")
    # 测试 ROCm 环境下的反向传播守卫功能
    def test_rocm_backward_pass_guard(self):
        # 此测试使用了 ROCm 特定的功能

        # 定义一个自定义的 PyTorch 自动求导函数
        class MyFunction(torch.autograd.Function):
            @staticmethod
            # 前向传播函数
            def forward(ctx, tensor, constant):
                self.assertFalse(torch._C._rocm_is_backward_pass())  # 断言当前不处于 ROCm 的反向传播阶段
                ctx.constant = constant  # 保存常数到上下文中
                return tensor * constant  # 返回张量与常数的乘积

            @staticmethod
            # 反向传播函数
            def backward(ctx, grad_output):
                self.assertTrue(torch._C._rocm_is_backward_pass())  # 断言当前处于 ROCm 的反向传播阶段
                return grad_output * ctx.constant, None  # 返回梯度乘以常数，第二个返回值为 None

        # 定义一个简单的 PyTorch 模块
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(()))  # 初始化模块参数 a

            def forward(self, x):
                return MyFunction.apply(x, self.a)  # 使用自定义函数应用于输入张量和参数 a

        model = MyModule()  # 创建模型实例
        criterion = torch.nn.MSELoss(reduction="sum")  # 定义损失函数
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # 定义优化器

        x = torch.randn(5, 5)  # 生成随机输入张量
        result = model(x)  # 模型前向传播
        loss = criterion(result, x)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
    @unittest.skipIf(TEST_MULTIGPU, "Testing on one GPU is sufficient")
    def test_lazy_init(self):
        """Validate that no CUDA calls are made during `import torch` call"""
        
        # 定义一个函数，用于执行给定的 Python 脚本并返回其输出结果
        def check_output(script: str) -> str:
            return (
                subprocess.check_output([sys.executable, "-c", script])
                .decode("ascii")
                .strip()
            )
        
        # 根据测试是否在 ROCm 上执行，选择不同的环境变量名称
        VISIBLE_DEVICES = (
            "HIP_VISIBLE_DEVICES" if TEST_WITH_ROCM else "CUDA_VISIBLE_DEVICES"
        )
        # 准备一个测试脚本，该脚本设置环境变量并检查 CUDA 设备数量
        test_script = f"import os; import torch;os.environ['{VISIBLE_DEVICES}']='32';print(torch.cuda.device_count())"
        # 执行测试脚本并获取其输出结果
        rc = check_output(test_script)
        # 断言 CUDA 设备数量为 0
        self.assertEqual(rc, "0")
        
        if not TEST_WITH_ROCM:
            # 如果不在 ROCm 上测试，进一步验证在导入期间未调用 `cuInit`
            # 通过 ctypes 调用 cuDeviceCountGet()，并期望 CUDA_ERROR_NOT_INITIALIZED == 3
            # 参考 https://github.com/pytorch/pytorch/issues/116276 了解更多细节
            libcuda_name = "libcuda.so.1" if not IS_WINDOWS else "nvcuda.dll"
            # 准备调用 CUDA 驱动 API 的代码片段
            cuda_driver_api_call = (
                f"ctypes.CDLL('{libcuda_name}').cuDeviceGetCount(ctypes.byref(x))"
            )
            # 执行验证调用 cuDeviceGetCount() 并获取其输出结果
            rc = check_output(
                f"import torch; import ctypes;x=ctypes.c_int(-1);print({cuda_driver_api_call})"
            )
            # 断言调用返回 CUDA_ERROR_NOT_INITIALIZED (3)
            self.assertEqual(rc, "3")
import torch  # 导入 PyTorch 库
import os  # 导入操作系统相关功能模块

r1 = torch.cuda.device_count()  # 获取当前可用的 CUDA 设备数量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置环境变量 CUDA_VISIBLE_DEVICES 为 '0'
r2 = torch.cuda.device_count()  # 再次获取当前可用的 CUDA 设备数量

torch.empty(10, device='cuda')  # 在 CUDA 设备上创建一个未初始化的张量

print(f"{r1}, {r2}")  # 打印 r1 和 r2 的值

"""
这部分代码段之后不包含具体的代码，仅作为字符串存在，不需要注释
"""
    def test_direct_traceback(self):
        # 导入需要的函数：gather_traceback 和 symbolize_tracebacks
        from torch._C._profiler import gather_traceback, symbolize_tracebacks

        # 收集当前的堆栈跟踪信息
        c = gather_traceback(True, True, True)
        # 符号化堆栈跟踪信息
        (r,) = symbolize_tracebacks([c])
        # 将结果转换为字符串
        r = str(r)
        # 断言测试中包含特定文件名
        self.assertTrue("test_cuda.py" in r)
        # 断言测试中包含特定关键字
        self.assertTrue("unwind" in r)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_memory_snapshot_with_cpp(self):
        try:
            # 清空 CUDA 内存缓存
            torch.cuda.memory.empty_cache()
            # 记录内存历史记录
            torch.cuda.memory._record_memory_history("state", stacks="all")
            # 在 CUDA 设备上生成随机张量
            x = torch.rand(311, 411, device="cuda")

            # 获取当前 CUDA 内存快照的段信息
            ss = torch.cuda.memory._snapshot()["segments"]
            # 初始化标志变量
            found_it = False
            # 遍历所有段
            for seg in ss:
                # 遍历每个段的块
                for b in seg["blocks"]:
                    # 如果请求大小匹配特定值
                    if b["requested_size"] == 311 * 411 * 4:
                        # 断言块的帧信息中包含特定字符串
                        self.assertTrue("::rand" in str(b["frames"]))
                        found_it = True
            # 断言找到匹配的块
            self.assertTrue(found_it)

        finally:
            # 停止记录内存历史记录
            torch.cuda.memory._record_memory_history(None)

    @skipIfRocm
    def test_memory_profiler_viz(self):
        # 使用 torch.profiler.profile 进行性能分析
        with torch.profiler.profile(
            with_stack=True, profile_memory=True, record_shapes=True
        ) as prof:
            # 在 CUDA 设备上生成随机张量并进行运算
            x = torch.rand(128, 128, device="cuda")
            x * x + x * x
        # 生成性能分析结果的图形表示
        plot = profile_plot(prof)
        # 将性能分析结果转换为 JSON 格式的快照
        plot = json.dumps(_profile_to_snapshot(prof))
        # 断言结果中包含特定文件名
        self.assertTrue("test_cuda.py" in plot)
        # 断言结果中包含特定测试函数名
        self.assertTrue("test_memory_profiler_viz" in plot)
        # 断言结果中包含特定类别信息
        self.assertTrue("category" in plot)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    def test_cycles(self):
        # 初始化触发标志
        fired = False

        # 定义观察器函数
        def observer(html):
            nonlocal fired
            fired = True
            # 断言 HTML 中包含特定字符串
            self.assertTrue("torch.Tensor" in html)
            self.assertTrue("test_cuda" in html)
            self.assertTrue("cell_contents" in html)

        # 监视张量循环引用
        disarm = observe_tensor_cycles(observer)

        # 定义空操作函数
        def noop():
            pass

        try:
            # 定义创建函数
            def create():
                # 在 CUDA 设备上创建空张量
                x = torch.empty(3, 4, device="cuda")

                # 定义递归函数
                def foo(p):
                    if p:
                        return foo(not p)
                    else:
                        return x

                return foo

            # 调用创建函数
            create()
            # 执行垃圾回收
            gc.collect()
            # 确保回调在垃圾回收后运行，以便在下次方法调用后触发
            noop()
            # 断言触发标志已经设置为 True
            self.assertTrue(fired)
        finally:
            # 停止观察张量循环引用
            disarm()

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    # 如果当前运行环境是 ARM64 架构或不是 Linux 系统，则跳过测试，因为 CPP 上下文仅支持 x86 Linux
    def test_memory_plots(self):
        # 对于每一个上下文和堆栈的组合进行测试
        for context, stacks in (
            ("all", "all" if IS_LINUX else "python"),
            ("all", "python"),
            (None, "python"),
        ):
            try:
                # 清空 CUDA 内存缓存
                torch.cuda.memory.empty_cache()
                # 记录 CUDA 内存历史，针对指定的上下文和堆栈
                torch.cuda.memory._record_memory_history(
                    "all", context=context, stacks=stacks
                )

                # 定义一个运行函数，生成一个 CUDA 设备上的随机张量
                def run():
                    x = torch.rand(128, 128, device="cuda")
                    x * x + x * x

                run()
                # 判断是否为 CPP 堆栈
                cpp = stacks == "all"
                # 判断是否记录了特定的上下文
                record_context = context is not None
                # 获取 CUDA 内存快照
                ss = torch.cuda.memory._snapshot()

                # 生成内存追踪图
                tplot = trace_plot(ss)
                # 生成内存段图
                splot = segment_plot(ss)
                # 将快照信息转换为 JSON 格式的文本
                text = json.dumps(ss)

                # 断言：检查是否记录了特定的上下文
                self.assertTrue(record_context == ("test_memory_plots" in text))
                # 断言：检查是否为 CPP 堆栈
                self.assertTrue(cpp == ("::rand" in text))
                # 断言：检查是否包含预期的内存占用大小信息
                self.assertTrue(str(128 * 128 * 4) in text)

            finally:
                # 清空 CUDA 内存历史记录
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    # 如果当前运行环境是 ARM64 架构或不是 Linux 系统，则跳过测试，因为 CPP 上下文仅支持 x86 Linux
    def test_memory_plots_free_stack(self):
        # 对于每一个上下文进行测试
        for context in ["alloc", "all", "state"]:
            try:
                # 清空 CUDA 内存缓存
                torch.cuda.memory.empty_cache()
                # 记录 CUDA 内存历史，针对指定的上下文
                torch.cuda.memory._record_memory_history(context=context)
                x = None

                # 定义分配函数，分配一个 CUDA 设备上的张量
                def thealloc():
                    nonlocal x
                    x = torch.rand(3, 4, device="cuda")

                # 定义释放函数，删除之前分配的张量
                def thefree():
                    nonlocal x
                    del x

                thealloc()
                thefree()
                # 获取 CUDA 内存快照并转换为 JSON 格式的文本
                ss = json.dumps(torch.cuda.memory._snapshot())
                # 断言：检查内存快照中是否包含特定的分配函数
                self.assertTrue(("thefree" in ss) == (context == "all"))
                # 断言：检查内存快照中是否包含特定的释放函数
                self.assertTrue(("thealloc" in ss) == (context != "state"))

            finally:
                # 清空 CUDA 内存历史记录
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    # 如果当前运行环境是 ARM64 架构或不是 Linux 系统，则跳过测试，因为 CPP 上下文仅支持 x86 Linux
    # 定义一个测试方法，用于测试内存图形历史上下文功能
    def test_memory_plots_history_context(self):
        try:
            # 清空CUDA内存缓存
            torch.cuda.memory.empty_cache()
            x = None

            # 定义一个函数，用于在CUDA设备上生成一个随机张量，并捕获到nonlocal变量x
            def should_capture1():
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            # 定义一个函数，用于在CUDA设备上生成一个不同形状的随机张量，并捕获到nonlocal变量x
            def should_not_capture():
                nonlocal x
                x = torch.rand(3, 4, device="cuda")

            # 定义一个函数，再次在CUDA设备上生成一个随机张量，并捕获到nonlocal变量x
            def should_capture2():
                nonlocal x
                x = torch.rand(4, 4, device="cuda")

            # 开始记录内存历史，捕获上下文和Python调用堆栈
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture1()
            # 停止记录内存历史，不捕获上下文
            torch.cuda.memory._record_memory_history(context=None)
            should_not_capture()
            # 再次开始记录内存历史，捕获上下文和Python调用堆栈
            torch.cuda.memory._record_memory_history(context="all", stacks="python")
            should_capture2()

            # 将当前CUDA内存快照转换为JSON字符串
            ss = json.dumps(torch.cuda.memory._snapshot())
            # 断言检查字符串中是否包含特定的函数名，用于验证内存历史是否被正确记录
            self.assertTrue("should_capture1" in ss)
            self.assertTrue("should_not_capture" not in ss)
            self.assertTrue("should_capture2" in ss)
        finally:
            # 最终停止记录内存历史
            torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    @unittest.skipIf(IS_ARM64 or not IS_LINUX, "cpp contexts are x86 linux only")
    # 定义一个测试方法，用于测试内存图形中的空闲段堆栈功能
    def test_memory_plots_free_segment_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                # 清空CUDA内存缓存
                torch.cuda.memory.empty_cache()
                # 开始记录内存历史，捕获特定的上下文
                torch.cuda.memory._record_memory_history(context=context)
                # 在CUDA设备上生成一个随机张量
                x = torch.rand(3, 4, device="cuda")
                # 删除张量x，释放内存
                del x
                # 再次清空CUDA内存缓存
                torch.cuda.memory.empty_cache()

                # 将当前CUDA内存快照转换为JSON字符串
                ss = json.dumps(torch.cuda.memory._snapshot())
                # 断言检查字符串中是否包含特定的函数名，用于验证内存历史是否被正确记录
                self.assertTrue(("empty_cache" in ss) == (context == "all"))
            finally:
                # 最终停止记录内存历史
                torch.cuda.memory._record_memory_history(None)

    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    # 定义一个测试函数，用于测试内存快照脚本的行为
    def test_memory_snapshot_script(self):
        try:
            # 清空 CUDA 内存缓存
            torch.cuda.memory.empty_cache()
            # 记录 CUDA 内存历史记录，使用 Python 堆栈
            torch.cuda.memory._record_memory_history("state", stacks="python")

            # 定义一个 Torch 脚本函数 foo
            @torch.jit.script
            def foo():
                # 在 CUDA 设备上生成一个指定形状的随机张量
                return torch.rand(311, 411, device="cuda")

            # 调用函数 foo，得到张量 x
            x = foo()

            # 获取 CUDA 内存的快照中的段列表
            ss = torch.cuda.memory._snapshot()["segments"]
            # 初始化标志变量 found_it
            found_it = False
            # 遍历快照中的每个段
            for seg in ss:
                # 遍历每个段中的块
                for b in seg["blocks"]:
                    # 如果块的请求大小等于指定值，则执行断言
                    if b["requested_size"] == 311 * 411 * 4:
                        # 断言块的帧名称为 "foo"
                        self.assertTrue(b["frames"][0]["name"] == "foo")
                        # 设置 found_it 标志为 True
                        found_it = True
            # 最终断言 found_it 标志为 True
            self.assertTrue(found_it)

        finally:
            # 最终清除 CUDA 内存历史记录
            torch.cuda.memory._record_memory_history(None)

    @parametrize("max_split_size_mb_setting", [False, True])
    # 定义一个参数化测试函数，用于测试内存耗尽时的行为
    def test_raises_oom(self, max_split_size_mb_setting):
        if max_split_size_mb_setting:
            # 如果设置了 max_split_size_mb，设置 CUDA 分配器的设置
            torch.cuda.memory._set_allocator_settings("max_split_size_mb:1024")
            # 清空 CUDA 内存缓存
            torch.cuda.memory.empty_cache()
        # 断言在 CUDA 设备上创建超大空张量时抛出 OutOfMemoryError
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")

    @unittest.skipIf(
        not (IS_LINUX and os.uname().machine == "x86_64"), "cpp traces only on linux"
    )
    @unittest.skipIf(
        TEST_CUDAMALLOCASYNC, "setContextRecorder not supported by CUDAMallocAsync"
    )
    # 标记测试函数为跳过条件测试，用于特定的环境和条件
    # 定义一个测试方法，用于测试 C++ 内存快照的序列化与记录
    def test_cpp_memory_snapshot_pickle(self):
        # 导入加载内联 C++ 扩展的函数
        from torch.utils.cpp_extension import load_inline

        # 定义 C++ 源码字符串，包含了内存快照和记录内存历史的函数
        source = """
        #include <torch/csrc/cuda/memory_snapshot.h>
        py::object do_snapshot() {
            // 调用 Torch CUDA API 获取内存快照并序列化为字符串
            std::string data = torch::cuda::_memory_snapshot_pickled();
            return py::bytes(data);
        }
        void record(bool e, bool ctx) {
            // 记录 CUDA 内存历史记录
            torch::cuda::_record_memory_history(e, ctx, 10, ctx, ctx);
        }
        """

        # 加载内联 C++ 扩展模块，并指定要导出的函数
        m = load_inline(
            name="snapshot", cpp_sources=[source], functions=["do_snapshot", "record"]
        )

        # 遍历上下文选项（False 和 True）
        for ctx in (False, True):
            try:
                # 记录内存历史记录，记录当前上下文状态
                m.record(True, ctx)

                # 使用 Torch 的 JIT 脚本创建一个函数
                @torch.jit.script
                def the_script_fn():
                    return torch.rand(311, 411, device="cuda")

                # 定义运行函数，执行脚本并反序列化内存快照数据
                def run():
                    t = the_script_fn()
                    return pickle.loads(m.do_snapshot())

                # 运行并获取内存快照数据
                mem = run()
                found = False

                # 遍历内存快照的段落
                for s in mem["segments"]:
                    for b in s["blocks"]:
                        # 检查块的状态是否为活跃分配
                        if b["state"] == "active_allocated":
                            # 检查请求的内存大小是否符合预期
                            if b["requested_size"] == 311 * 411 * 4:
                                if ctx:
                                    frame_text = str(b["frames"])
                                    # 断言包含 C++ 帧信息
                                    self.assertTrue("::rand" in frame_text)
                                    # 断言包含脚本帧信息
                                    self.assertTrue("the_script_fn" in frame_text)
                                    # 断言包含 Python 帧信息
                                    self.assertTrue("case.py" in frame_text)
                                found = True

                # 获取设备跟踪的最后一个操作
                last_action = mem["device_traces"][0][-1]
                # 断言最后一个操作为分配操作
                self.assertTrue(last_action["action"] == "alloc")
                # 断言最后一个操作的大小符合预期
                self.assertTrue(last_action["size"] == 311 * 411 * 4)
                # 断言已找到期望的块信息
                self.assertTrue(found)
            finally:
                # 结束记录内存历史记录，恢复初始上下文状态
                m.record(False, False)

    # 跳过测试，如果 CUDA 异步内存分配临时被禁用
    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    # 定义一个测试方法，用于测试 CUDA 内存分配超出内存时的通知机制
    def test_notifies_oom(self):
        x = False

        # 定义一个回调函数，用于捕获 CUDA 内存耗尽事件
        def cb(device, alloc, device_alloc, device_free):
            nonlocal x
            x = True

        # 注册 CUDA 内存耗尽观察器，将回调函数绑定到其上
        torch._C._cuda_attach_out_of_memory_observer(cb)

        # 使用断言，检测是否抛出 CUDA 内存耗尽错误
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="cuda")
        
        # 断言 x 被设置为 True，即捕获到内存耗尽事件
        self.assertTrue(x)
    # 测试 GPU 内存分配器的模糊测试
    def test_allocator_fuzz(self):
        # 保存当前随机数生成器的状态
        state = random.getstate()
        # 设定固定的随机种子，以便重现测试结果
        random.seed(123)
        # 设定测试迭代次数
        N = 10000
        try:
            # 初始化内存分配记录列表和总内存使用量
            mem = []
            total = 0
            c = 0

            # 定义内部函数：分配内存
            def alloc():
                nonlocal total, c
                # 随机生成分配的张量大小（单位为元素数量）
                b = random.randrange(2 * 1024 * 1024 // 4, 200 * 1024 * 1024 // 4)
                # 创建指定大小的张量，并将其加入内存记录列表
                mem.append((c, torch.full((b,), c, dtype=torch.int32, device="cuda")))
                c += 1
                total += b

            # 定义内部函数：释放内存
            def free():
                nonlocal total
                # 随机选择要释放的内存块索引
                idx = random.randrange(0, len(mem))
                v, x = mem.pop(idx)
                # 断言已释放的张量与其值一致
                assert torch.all(v == x)
                total -= x.numel()

            # 可选操作列表，包括分配内存、释放内存和清空 GPU 缓存
            choices = [alloc, free, torch.cuda.memory.empty_cache]
            # 执行 N 次操作
            for i in range(N):
                # 如果总内存使用量超过阈值，连续释放内存直至低于阈值
                while total >= 1024 * 1024 * 1024 / 4:
                    free()
                # 随机选择并执行一种操作
                (action,) = random.choices(choices, weights=[1, 1 if mem else 0, 0.1])
                action()
        finally:
            # 恢复原先的随机数生成器状态
            random.setstate(state)

    # 跳过测试，若未安装 pynvml 库
    @unittest.skipIf(TEST_PYNVML, "pynvml is not available")
    def test_nvml_get_handler(self):
        # 如果不是 HIP 版本的 PyTorch，则确保获取到 pynvml 处理器
        if not torch.version.hip:
            self.assertTrue(torch.cuda._get_pynvml_handler() is not None)
        else:
            # 否则确保获取到 AMD SMI 处理器
            self.assertTrue(torch.cuda._get_amdsmi_handler() is not None)

    # 跳过测试，若未安装 pynvml 库
    @unittest.skipIf(TEST_PYNVML, "pynvml is not available")
    def test_temperature(self):
        # 确保 GPU 温度在合理范围内（0 到 150 度之间）
        self.assertTrue(0 <= torch.cuda.temperature() <= 150)

    # 跳过测试，若未安装 pynvml 库
    @unittest.skipIf(TEST_PYNVML, "pynvml is not available")
    def test_power_draw(self):
        # 确保 GPU 功耗大于等于 0
        self.assertTrue(torch.cuda.power_draw() >= 0)

    # 跳过测试，若未安装 pynvml 库
    @unittest.skipIf(TEST_PYNVML, "pynvml is not available")
    def test_clock_speed(self):
        # 确保 GPU 时钟速度大于等于 0
        self.assertTrue(torch.cuda.clock_rate() >= 0)
MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1048576
SMALL_BUFFER = 2097152
LARGE_BUFFER = 20971520

# 获取特定内存池中的 CUDA 图段
def get_cudagraph_segments(pool_id):
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]

# 获取所有 CUDA 图段
def get_all_cudagraph_segments():
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]

# 将函数调用和输入 CUDA 图形化
def cudagraphify(fn, inputs, pool=None):
    if not TEST_CUDA_GRAPH:
        raise unittest.SkipTest("cuda graph test is skipped")
    
    # 同步 CUDA 操作
    torch.cuda.synchronize()
    # 创建新的 CUDA 流
    stream = torch.cuda.Stream()
    # 等待当前 CUDA 流
    stream.wait_stream(torch.cuda.current_stream())
    # 在新的 CUDA 流中执行函数调用
    with torch.cuda.stream(stream):
        fn(*inputs)
    # 同步 CUDA 流
    stream.synchronize()
    # 等待当前 CUDA 流与新 CUDA 流同步
    torch.cuda.current_stream().wait_stream(stream)
    # 同步 CUDA 操作
    
    # 创建新的 CUDA 图
    graph = torch.cuda.CUDAGraph()
    # 使用 CUDA 图和 CUDA 流执行函数调用，返回静态输出
    with torch.cuda.graph(graph, stream=stream, pool=pool):
        static_outputs = fn(*inputs)
    
    return graph, static_outputs

# 创建一个指定大小的 uint8 类型的 CUDA 张量
def int8_cuda(size):
    return torch.ones([size], device="cuda", dtype=torch.uint8)

# 计算特定内存池中活跃块的数量
def live_blocks(pool_id):
    blocks = 0
    seg = get_cudagraph_segments(pool_id)
    for segment in get_cudagraph_segments(pool_id):
        for block in segment["blocks"]:
            blocks += block["state"] == "active_allocated"
    return blocks

# 获取张量的元数据信息
def tensor_metadata(x):
    return {
        "nbytes": x.untyped_storage().nbytes(),
        "data_ptr": x.untyped_storage().data_ptr(),
        "size": x.shape,
        "stride": x.stride(),
        "dtype": x.dtype,
        "device": x.device,
        "storage_offset": x.storage_offset(),
    }

# 根据元数据信息重建张量
def reconstruct_from_tensor_metadata(metadata):
    s = torch._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    t = torch.empty([0], device=metadata["device"], dtype=metadata["dtype"])
    t.set_(
        source=s,
        storage_offset=metadata["storage_offset"],
        size=metadata["size"],
        stride=metadata["stride"],
    )
    return t

# 根据条件跳过测试
@unittest.skipIf(TEST_CUDAMALLOCASYNC or TEST_WITH_ROCM, "NYI")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestBlockStateAbsorption(TestCase):
    # 检查检查点块的状态吸收
    def checkCheckpointedBlock(self, before_block, after_block):
        for field in ("size", "state"):
            self.assertEqual(before_block[field], after_block[field])
    # 检查两个状态之间的一致性：before_segments 和 after_segments。after_segments 可能包含额外的段，但 before_segments 中的所有段应完全等同于 after_segments。
    def checkCheckpointedState(self, before_segments, after_segments):
        # 创建一个字典，以地址为键，对应段对象为值，用于快速查找 after_segments 中的段对象
        after_ptr_to_segment = {
            segment["address"]: segment for segment in after_segments
        }

        # 遍历 before_segments 中的每个段
        for before_segment in before_segments:
            # 断言 before_segment 的地址在 after_ptr_to_segment 中
            self.assertTrue(before_segment["address"] in after_ptr_to_segment)
            # 获取对应的 after_segment
            after_segment = after_ptr_to_segment[before_segment["address"]]

            # 检查段对象的各个字段是否相等
            for field in (
                "device",
                "total_size",
                "allocated_size",
                "active_size",
                "segment_type",
                "segment_pool_id",
            ):
                self.assertEqual(before_segment[field], after_segment[field])

            # 检查段中 blocks 的数量是否相等
            self.assertEqual(
                len(before_segment["blocks"]), len(after_segment["blocks"])
            )
            # 遍历并逐个比较 before_segment 和 after_segment 中对应位置的 blocks
            for before_block, after_block in zip(
                before_segment["blocks"], after_segment["blocks"]
            ):
                # 检查每个 block 的一致性
                self.checkCheckpointedBlock(before_block, after_block)

    @staticmethod
    # 设置检查点池的状态，更新 stale_storages_ptr 和 storages_deleters
    def setCheckpointPoolState(
        device, state, stale_storages_ptr, storages_deleters=None
    ):
        # 获取 stale_storages_ptr 中每个元素的 _cdata 属性
        stale_storages_ptr = [t.untyped_storage()._cdata for t in stale_storages_ptr]
        # 如果 storages_deleters 为 None，则设置为空列表；否则获取 storages_deleters 中每个元素的 _cdata 属性
        storages_deleters = (
            []
            if not storages_deleters
            else [t.untyped_storage()._cdata for t in storages_deleters]
        )
        # 调用底层 CUDA 函数设置检查点池的状态
        torch._C._cuda_setCheckpointPoolState(
            device, state, stale_storages_ptr, storages_deleters
        )

    # 检查给定函数的 CUDA 图形化版本在给定输入上的行为
    def checkFunction(self, fn, inputs, pool=None):
        # 将函数 fn 和输入 inputs 转化为 CUDA 图形化对象，并获取输出
        graph, outputs = cudagraphify(fn, inputs, pool=pool)

        # 获取 CUDA 图形的池 ID 和输出的设备索引
        pool_id = graph.pool()
        device = outputs[0].device.index

        # 获取当前 CUDA 图形池中的段信息
        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        # 获取当前设备上的检查点状态
        state = torch._C._cuda_getCheckpointState(device, pool_id)
        # 设置设备上的检查点池状态
        self.setCheckpointPoolState(device, state, [], [])

        # 检查在设置检查点状态后，CUDA 图形的段信息是否一致
        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    # 在每个测试运行前设置测试环境
    def setUp(self):
        super().setUp()
        # 获取当前所有 CUDA 图形的段数量，并记录长度
        self.segment_length = len(get_all_cudagraph_segments())

    # 在每个测试运行后清理测试环境
    def tearDown(self):
        # 同步 CUDA 设备，收集垃圾，清空 CUDA 缓存
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # 断言当前所有 CUDA 图形的段数量与设置前相同
        self.assertEqual(len(get_all_cudagraph_segments()), self.segment_length)

        super().tearDown()

    # 测试一个简单的 CUDA 函数
    def test_simple(self):
        # 定义一个简单的 CUDA 函数 foo
        def foo():
            # 在 CUDA 设备上创建一个全零张量 x，并执行一些操作
            x = torch.zeros([SMALL_SIZE * 8], device="cuda", dtype=torch.uint8)
            x = x + x
            # 执行一些 CUDA 函数并生成结果
            x1 = int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE) + int8_cuda(SMALL_SIZE)
            y = int8_cuda(SMALL_SIZE) + x1
            z = int8_cuda(SMALL_SIZE)
            return x, y, z

        # 测试函数 foo 的 CUDA 图形化版本
        self.checkFunction(foo, [])
    def test_allocated_in_middle_of_segment(self):
        def foo():
            # 创建一个包含11个 int8_cuda(MIN_BLOCK_SIZE) 的列表
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            # 返回列表中第 6 个元素，并对其执行加法操作
            return small_buffers[5].add_(2)

        # 调用检查函数，验证 foo 函数的行为
        self.checkFunction(foo, [])

    def test_multiple_middle_allocations(self):
        def foo():
            # 创建一个包含11个 int8_cuda(MIN_BLOCK_SIZE) 的列表
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            # 返回列表中第 6 和第 9 个元素
            return small_buffers[5], small_buffers[8]

        # 调用检查函数，验证 foo 函数的行为
        self.checkFunction(foo, [])

    def test_middle_allocations_contiguous(self):
        def foo():
            # 创建一个包含11个 int8_cuda(MIN_BLOCK_SIZE) 的列表
            small_buffers = [int8_cuda(MIN_BLOCK_SIZE) for _ in range(11)]
            # 返回列表中第 6 和第 7 个元素
            return small_buffers[5], small_buffers[6]

        # 调用检查函数，验证 foo 函数的行为
        self.checkFunction(foo, [])

    def test_additional_free_following_checkpoint(self):
        def foo():
            # 创建一个大小为 MIN_BLOCK_SIZE 的 int8_cuda 对象，并将其放入元组中返回
            return (int8_cuda(MIN_BLOCK_SIZE),)

        def foo2():
            # 创建一个大小为 MIN_BLOCK_SIZE 的 int8_cuda 对象，并将其放入元组中返回
            return (int8_cuda(MIN_BLOCK_SIZE),)

        # 通过 cudagraphify 函数生成 foo 函数的计算图和输出
        graph, outputs = cudagraphify(foo, [])
        # 获取计算图的池 ID
        pool_id = graph.pool()

        # 获取当前 CUDA 图段
        segments_before_checkpoint = get_cudagraph_segments(pool_id)

        # 获取检查点状态
        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        # 通过 cudagraphify 函数生成 foo2 函数的计算图和输出，使用与 graph 相同的池
        graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())

        # 设置检查点池状态
        self.setCheckpointPoolState(outputs[0].device.index, state, outputs2, [])

        # 删除 outputs2 变量
        del outputs2

        # 检查检查点前后的 CUDA 图段状态
        self.checkCheckpointedState(
            segments_before_checkpoint, get_cudagraph_segments(pool_id)
        )

    # TODO: re-enable
    # def test_additional_free_error(self):
    #     def foo():
    #         return int8_cuda(MIN_BLOCK_SIZE),
    #
    #     def foo2():
    #         return int8_cuda(MIN_BLOCK_SIZE),
    #
    #     graph, outputs = cudagraphify(foo, [])
    #     pool_id = graph.pool()
    #
    #     segments_before_checkpoint = get_cudagraph_segments(pool_id)
    #
    #     state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)
    #
    # graph2, outputs2 = cudagraphify(foo2, [], pool=graph.pool())
    # with self.assertRaisesRegex(Exception, "being manually freed must be passed"):
    #     self.setCheckpointPoolState(outputs[0].device.index, state, [], [])

    def test_tensor_dies_after_checkpoint(self):
        def foo():
            # 创建两个大小为 MIN_BLOCK_SIZE 的 int8_cuda 对象，并将其放入元组中返回
            return int8_cuda(MIN_BLOCK_SIZE), int8_cuda(MIN_BLOCK_SIZE)

        # 通过 cudagraphify 函数生成 foo 函数的计算图和输出
        graph, outputs = cudagraphify(foo, [])
        # 获取计算图的池 ID
        pool_id = graph.pool()
        # 获取第一个输出对象的设备索引
        device = outputs[0].device.index

        # 获取当前 CUDA 图段
        segments_before_checkpoint = get_cudagraph_segments(pool_id)
        # 获取检查点状态
        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        # 获取所有输出对象的数据指针列表
        output_data_ptrs = [output.data_ptr() for output in outputs]

        # 删除 outputs 变量
        del outputs

        # 设置检查点池状态为空
        self.setCheckpointPoolState(device, state, [], [])

        # 断言在检查点后还有 2 个存活的块
        self.assertEqual(live_blocks(pool_id), 2)
        # 手动释放第一个输出对象的 CUDA 缓存分配器
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[0])
        # 断言在释放第一个输出对象后还有 1 个存活的块
        self.assertEqual(live_blocks(pool_id), 1)
        # 手动释放第二个输出对象的 CUDA 缓存分配器
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_data_ptrs[1])
        # 断言在释放第二个输出对象后还有 0 个存活的块
        self.assertEqual(live_blocks(pool_id), 0)
    # 定义一个测试函数，测试将删除函数分配给张量的操作
    def test_assigning_back_deleter_fns_to_tensor(self):
        # 定义一个内部函数 foo，接受参数 x，并返回三个张量的加法结果
        def foo(x):
            return (
                int8_cuda(SMALL_BUFFER) + x,  # 使用 int8_cuda 处理 SMALL_BUFFER 并加上 x
                int8_cuda(SMALL_BUFFER) + x,  # 同上
                int8_cuda(LARGE_BUFFER) + x,  # 使用 int8_cuda 处理 LARGE_BUFFER 并加上 x
            )

        # 创建一个包含单个元素 1 的张量 inp，指定其在 CUDA 设备上
        inp = torch.tensor([1], device="cuda")
        # 使用 cudagraphify 函数将 foo 转换为图形对象 graph，并计算其输出
        graph, outputs = cudagraphify(foo, [inp])
        # 在图形对象上创建一个池，并返回其 ID
        pool_id = graph.pool()
        # 重放图形对象以执行计算
        graph.replay()

        # 获取第一个输出张量的设备索引
        device = outputs[0].device.index

        # 断言每个输出张量的平均值为 2，使用 torch.float 类型计算
        for i in range(len(outputs)):
            self.assertTrue(outputs[i].mean(dtype=torch.float) == 2)

        # 获取输出张量池的状态，用于检查点
        state = torch._C._cuda_getCheckpointState(outputs[0].device.index, pool_id)

        # 创建一个包含每个输出张量非类型化存储数据指针的列表
        output_ptrs = [output.untyped_storage().data_ptr() for output in outputs]
        # 创建一个包含每个输出张量的张量元数据的列表
        ten_metadata = [tensor_metadata(t) for t in outputs]

        # 断言在指定池中活跃的块数为 3
        self.assertEqual(live_blocks(pool_id), 3)

        # 删除 outputs 变量后，再次断言活跃的块数为 0
        del outputs
        self.assertEqual(live_blocks(pool_id), 0)

        # 使用张量元数据重构张量列表
        reconstructed_tensors = [
            reconstruct_from_tensor_metadata(metadata) for metadata in ten_metadata
        ]

        # 断言每个重构的张量的平均值为 2
        for i in range(len(reconstructed_tensors)):
            self.assertTrue(reconstructed_tensors[i].mean(dtype=torch.float) == 2)

        # 修改输入张量 inp 的值并重放图形对象
        inp.add_(1)
        graph.replay()

        # 断言每个重构的张量的平均值为 3
        for i in range(len(reconstructed_tensors)):
            self.assertTrue(reconstructed_tensors[i].mean(dtype=torch.float) == 3)

        # 使用 setCheckpointPoolState 函数设置检查点池的状态
        self.setCheckpointPoolState(
            device, state, [], [reconstructed_tensors[0], reconstructed_tensors[1]]
        )

        # 再次断言在指定池中活跃的块数为 3
        self.assertEqual(live_blocks(pool_id), 3)

        # 将第一个重构的张量设置为 None，并断言在指定池中活跃的块数为 2
        reconstructed_tensors[0] = None
        self.assertEqual(live_blocks(pool_id), 2)

        # 将第二个重构的张量设置为 None，并断言在指定池中活跃的块数为 1
        reconstructed_tensors[1] = None
        self.assertEqual(live_blocks(pool_id), 1)

        # 注释部分不会改变，因为我们没有传递它来交换数据指针
        # 再次断言在指定池中活跃的块数为 1
        reconstructed_tensors[2] = None
        self.assertEqual(live_blocks(pool_id), 1)

        # 使用 torch._C._cuda_cudaCachingAllocator_raw_delete 函数删除第三个输出的数据指针
        torch._C._cuda_cudaCachingAllocator_raw_delete(output_ptrs[2])

        # 最终断言在指定池中活跃的块数为 0
        self.assertEqual(live_blocks(pool_id), 0)

    # 如果没有 TorchVision 应用，跳过此测试
    @skipIfNoTorchVision
    # 测试 ResNet 模型的基本功能
    def test_resnet(self):
        # 导入 torchvision 模块
        import torchvision

        # 创建一个 ResNet-50 模型实例 m
        m = torchvision.models.resnet50()
        # 将模型设置为评估模式
        m.eval()
        # 将模型移动到 CUDA 设备上
        m = m.cuda()

        # 创建一个随机张量 inp，形状为 [1, 3, 255, 255]，在 CUDA 设备上
        inp = torch.rand([1, 3, 255, 255], device="cuda")
        # 使用 self.checkFunction 方法检查模型 m 对输入 inp 的功能
        self.checkFunction(m, [inp])

    # 测试检查池的活动分配
    def test_check_pool_live_allocations(self):
        # 定义一个函数 foo，返回一个包含四个元素的张量，全部为 1，在 CUDA 设备上
        def foo():
            return torch.ones([4], device="cuda")

        # 创建一个 CUDA 图形池句柄 pool 和一个图形对象 graph，计算 foo 的输出
        pool = torch.cuda.graph_pool_handle()
        graph, outputs = cudagraphify(foo, [], pool=pool)

        # 获取第一个输出张量的设备索引
        index = outputs[0].device.index

        # 定义一个检查函数 check，检查指定池中的存活数据指针 live_dps
        def check(live_dps):
            return torch._C._cuda_checkPoolLiveAllocations(index, pool, live_dps)

        # 断言检查函数对包含第一个输出张量数据指针的集合返回 True
        self.assertTrue(check({outputs[0].data_ptr()}))

        # 断言检查函数对包含第一个输出张量数据指针和 0 的集合返回 False
        self.assertFalse(check({outputs[0].data_ptr(), 0}))
        # 断言检查函数对空集合返回 False
        self.assertFalse(check(set()))

        # 删除 outputs 变量后，再次断言检查函数对空集合返回 True
        del outputs
        self.assertTrue(check(set()))
    # 定义一个测试方法，用于测试在线程中向 CUDA 内存池分配资源的情况
    def test_allocate_in_thread_to_pool(self):
        # 定义一个简单的函数 foo，返回一个在 CUDA 设备上生成的随机张量
        def foo():
            return torch.rand([4], device="cuda")

        # 创建一个 CUDA 图形池句柄对象
        pool = torch.cuda.graph_pool_handle()
        # 将函数 foo 转换为 CUDA 图形对象，使用给定的 CUDA 图形池
        graph, outputs = cudagraphify(foo, [], pool=pool)
        # 获取函数 foo 的输出张量所在的 CUDA 设备索引
        device = outputs[0].device.index
        # 删除变量 outputs，释放资源
        del outputs

        # 定义一个上下文管理器，用于管理使用 CUDA 内存池进行新分配的情况
        @contextlib.contextmanager
        def _use_cuda_memory_pool_manager(device, mem_pool):
            """
            Context manager to use cuda graph pool for new allocations. If you use this manager
            all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
            existing_graph should already have been used in a capture, and the mem_pool must already exist.
            """
            # 同步 CUDA 设备，等待操作完成
            torch.cuda.synchronize()
            # 创建一个新的 CUDA 流对象
            stream = torch.cuda.Stream()
            # 等待流的完成
            stream.wait_stream(torch.cuda.current_stream())
            # 创建一个 CUDA 流上下文
            stream_context = torch.cuda.stream(stream)
            stream_context.__enter__()
            # 开始将当前流分配给 CUDA 内存池
            torch._C._cuda_beginAllocateCurrentStreamToPool(device, mem_pool)
            try:
                yield  # 执行代码块，分配资源
            finally:
                # 结束将当前流分配给 CUDA 内存池
                torch._C._cuda_endAllocateCurrentStreamToPool(device, mem_pool)
                # 释放 CUDA 内存池
                torch._C._cuda_releasePool(device, mem_pool)
                # 退出 CUDA 流上下文
                stream_context.__exit__(None, None, None)

        # 获取 CUDA 图形的片段
        segments = get_cudagraph_segments(pool)
        # 断言 CUDA 图形的片段数为 1
        self.assertEqual(len(get_cudagraph_segments(pool)), 1)

        # 定义一个函数 use_pool，使用 CUDA 内存池分配资源
        def use_pool():
            # 定义一个函数 alloc_three，分配三个资源
            def alloc_three():
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                c = a + b

            # 使用 _use_cuda_memory_pool_manager 管理 CUDA 内存池资源分配
            with _use_cuda_memory_pool_manager(device, pool):
                # 执行十次三个资源的分配
                for _ in range(10):
                    alloc_three()

            # 执行三次未使用 CUDA 内存池的资源分配
            alloc_three()

        # 定义一个函数 no_pool，不使用 CUDA 内存池，直接分配资源
        def no_pool():
            # 执行十次两个资源的分配
            for _ in range(10):
                a = int8_cuda(LARGE_BUFFER)
                b = int8_cuda(LARGE_BUFFER)
                del a, b

        # 创建一个线程用于执行使用 CUDA 内存池的资源分配
        graph_thread = threading.Thread(target=use_pool)
        # 创建一个线程用于执行不使用 CUDA 内存池的资源分配
        no_graph_thread = threading.Thread(target=no_pool)
        # 启动两个线程
        graph_thread.start()
        no_graph_thread.start()

        # 等待两个线程执行完毕
        graph_thread.join()
        no_graph_thread.join()

        # 断言 CUDA 图形的片段数为 4
        self.assertEqual(len(get_cudagraph_segments(pool)), 4)

        # 删除变量 graph，释放资源
        del graph

        # 同步 CUDA 设备
        torch.cuda.synchronize()
        # 执行垃圾回收
        gc.collect()
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

        # 断言 CUDA 图形的片段数为 0
        self.assertEqual(len(get_cudagraph_segments(pool)), 0)
    def test_no_triton_on_import(self):
        """定义一个测试函数，用于验证在首次使用 GPU 时不会导入 Triton"""
        # 定义一个包含 Python 脚本的字符串，该脚本导入 sys 和 torch 模块，然后在 GPU 上执行随机张量生成操作，
        # 最后检查 triton 是否在 sys.modules 中
        script = "import sys; import torch; torch.rand(2, device='cuda'); print('triton' in sys.modules)"

        # 使用 subprocess 模块运行指定的 Python 脚本，并捕获输出结果
        rc = (
            subprocess.check_output(
                [sys.executable, "-c", script],
                # 在 Windows 上，默认的当前工作目录可能导致 `import torch` 失败，因此将当前工作目录设置为此脚本所在的目录
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
            # 去除输出结果两端的空白字符，并将字节流解码为 ASCII 字符串
            .strip()
            .decode("ascii")
        )
        # 使用 unittest 框架的断言方法，验证结果是否为 "False"，如果不是则抛出 AssertionError
        self.assertEqual(rc, "False", "Triton was imported when importing torch!")
# 定义一个测试类 TestCudaOptims，继承自 TestCase，用于测试 CUDA 相关的优化器行为
class TestCudaOptims(TestCase):

    # 这些测试将通过 instantiate_device_type_tests 实例化，以应用新的 OptimizerInfo 结构。

    # 使用 onlyNativeDeviceTypes 装饰器，限定只在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    # 使用 optims 装饰器，选择支持融合实现的优化器进行测试
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    # 使用 onlyCUDA 装饰器，限定只在 CUDA 设备上运行测试
    @onlyCUDA
    # 使用 parametrize 装饰器，测试 in_place_unscale 参数为 False 和 True 时的情况
    @parametrize("in_place_unscale", [False, True])
    # 使用 optims 装饰器，选择支持在 CUDA 上融合实现的优化器进行测试
    @optims(
        [optim for optim in optim_db if "cuda" in optim.supports_fused_on],
        dtypes=[torch.float32],
    )
    # 定义测试函数 test_grad_scaler_with_preset_grad_scale，接受 device、dtype、optim_info 和 in_place_unscale 参数
    def test_grad_scaler_with_preset_grad_scale(
        self, device, dtype, optim_info, in_place_unscale
    ):
        # 在 CUDA 设备上创建一个全为 1 的张量 weight，并设置 requires_grad=True
        weight = torch.ones((5, 5), device="cuda", requires_grad=True)
        # 将 weight 的梯度设置为全为 15 的张量
        weight.grad = torch.full_like(weight, fill_value=15)
        # 使用 optim_info 中的优化器类创建一个优化器 opt，设置学习率 lr=0.1，并启用融合优化
        opt = optim_info.optim_cls([weight], lr=0.1, fused=True)
        # 创建一个 GradScaler 对象 scaler，初始化缩放因子为 5
        scaler = torch.amp.GradScaler(init_scale=5)

        # 模拟缩放一个损失
        scaler.scale(torch.ones(5))

        # 如果 in_place_unscale 为 True，则在优化器上执行非原地 unscale 操作
        if in_place_unscale:
            scaler.unscale_(opt)
            # 断言梯度已经被就地除以相应值
            self.assertEqual(weight.grad, torch.full_like(weight, fill_value=3))

        # 用户设置一个 grad_scale 值，该值应该与优化器步骤融合
        opt.grad_scale = torch.Tensor([3]).cuda()
        # 使用 scaler 执行优化器步骤
        scaler.step(opt)

        # 检查用户的 grad_scale 是否被尊重（即梯度是否被除以 5 * 3）
        self.assertEqual(weight.grad, torch.full_like(weight, fill_value=1))

    # 使用 onlyCUDA 装饰器，限定只在 CUDA 设备上运行测试
    @onlyCUDA
    # 使用 unittest.skipIf 装饰器，根据 TEST_CUDA_GRAPH 的值决定是否跳过测试
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    # 使用 parametrize 装饰器，测试 foreach 和 fused 参数的组合情况
    @parametrize("foreach, fused", [(False, False), (True, False), (False, True)])
    # 使用 optims 装饰器，选择支持 foreach 实现且支持在 CUDA 上融合实现的优化器进行测试
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls and "cuda" in optim.supports_fused_on
        ],
        dtypes=[torch.float32],
    )
    def test_graph_grad_scaling(self, device, dtype, optim_info, foreach, fused):
        # 清空 CUDA 缓存，确保测试环境干净
        torch.cuda.empty_cache()

        # 创建一个梯度缩放器，初始化缩放比例为4.0
        scaler = torch.amp.GradScaler(device="cuda", init_scale=4.0)
        
        # 创建一个 CUDA 图形对象
        g = torch.cuda.CUDAGraph()
        
        # 创建一个 CUDA 流对象
        s = torch.cuda.Stream()

        # 在 CUDA 设备上创建一个具有梯度的张量 weight
        weight = torch.ones((100,), device="cuda", requires_grad=True)
        
        # 使用提供的优化器信息创建优化器对象 opt
        opt = optim_info.optim_cls([weight], lr=0.1, foreach=foreach, fused=fused)
        
        # 创建静态输入和静态梯度张量
        static_input = torch.ones_like(weight)
        static_grad = torch.ones_like(weight)

        # 预热阶段
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # 计算损失
            loss = (weight.half() * static_input).sum()
            # 使用梯度缩放器对损失进行缩放和反向传播
            scaler.scale(loss).backward()
        torch.cuda.current_stream().wait_stream(s)

        # 将优化器梯度清零
        opt.zero_grad(set_to_none=True)

        # 捕获阶段
        with torch.cuda.stream(s):
            # 开始捕获 CUDA 图形
            g.capture_begin()
            # 再次计算损失并进行缩放和反向传播
            loss = (weight.half() * static_input).sum()
            scaler.scale(loss).backward()
            # 结束捕获 CUDA 图形
            g.capture_end()

        # 定义输入值和预期结果值列表
        input_vals = [5, 20000, 5, 40000]
        expected_scales = [4, 2, 2, 1]
        expected_growth_trackers = [1, 0, 1, 0]
        expected_grad_vals = [5 * 4, float("inf"), 5 * 2, float("inf")]

        # 遍历输入值和预期结果值列表，执行测试
        for data, scale, growth_tracker, grad_val in zip(
            input_vals, expected_scales, expected_growth_trackers, expected_grad_vals
        ):
            # 更新静态输入值
            static_input.fill_(data)
            # 回放 CUDA 图形
            g.replay()
            # 断言当前 weight 的梯度是否与预期值一致
            self.assertEqual(weight.grad, torch.full_like(weight.grad, grad_val))
            # 使用梯度缩放器进行优化步骤
            scaler.step(opt)
            # 更新梯度缩放器状态
            scaler.update()
            # 断言缩放器的当前缩放比例和增长追踪器是否与预期一致
            self.assertEqual(scaler._scale, scale)
            self.assertEqual(scaler._growth_tracker, growth_tracker)
# 根据给定的测试类 `TestCuda` 实例化参数化测试
instantiate_parametrized_tests(TestCuda)

# 根据给定的测试类 `TestCudaMallocAsync` 实例化参数化测试
instantiate_parametrized_tests(TestCudaMallocAsync)

# 实例化与设备类型相关的测试类 `TestCudaOptims` 的测试，使用全局命名空间
instantiate_device_type_tests(TestCudaOptims, globals())

# 如果当前脚本作为主程序执行，则运行所有测试
if __name__ == "__main__":
    run_tests()
```