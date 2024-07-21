# `.\pytorch\test\inductor\test_max_autotune.py`

```py
# Owner(s): ["module: inductor"]
# 引入所需的模块和库
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统相关功能的模块
import unittest  # 导入单元测试框架模块

from typing import Callable, List, Optional  # 导入类型提示相关功能

import torch  # 导入 PyTorch 深度学习框架
from torch import multiprocessing as mp, nn  # 导入多进程和神经网络模块
from torch._dynamo import reset  # 导入动态分派模块的重置功能
from torch._dynamo.exc import BackendCompilerFailed  # 导入动态分派编译器错误异常
from torch._dynamo.testing import rand_strided, reset_rng_state  # 导入动态分派测试相关功能
from torch._inductor import config  # 导入感应器配置模块
from torch._inductor.autotune_process import (  # 导入自动调整过程相关模块
    BenchmarkRequest,
    CUDA_VISIBLE_DEVICES,
    TuningProcessPool,
)
from torch._inductor.graph import GraphLowering  # 导入图降维模块
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout  # 导入感应器中间表示相关类
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm  # 导入矩阵乘法加操作内核
from torch._inductor.select_algorithm import (  # 导入算法选择器缓存和 Triton 模板调用器
    AlgorithmSelectorCache,
    TritonTemplateCaller,
)
from torch._inductor.test_case import run_tests, TestCase  # 导入测试用例运行和测试案例类

from torch._inductor.utils import fresh_inductor_cache, run_and_get_code  # 导入感应器缓存更新和运行获取代码功能
from torch._inductor.virtualized import V  # 导入虚拟化张量类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入代理张量创建函数
from torch.testing import FileCheck  # 导入文件检查工具
from torch.testing._internal.common_utils import (  # 导入内部测试工具函数
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入感应器内部工具函数和 CUDA 支持检查

# 设置浮点数矩阵乘法的精度为高精度
torch.set_float32_matmul_precision("high")

# 如果有 CUDA 支持，则设置 CUDA 内存分配器设置为不可扩展
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

# 定义 Cutlass 库的路径
_CUTLASS_DIR = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/")


def _get_path_without_sccache() -> str:
    """
    获取不包含 sccache 的 PATH 环境变量。
    """
    # 获取当前的 PATH 环境变量，并移除包含 "/opt/cache/bin" 的路径
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


def benchmark_choice(choice, args, out, expected_out, timings):
    """
    进行选择器的基准测试，并验证输出是否符合预期。
    """
    # 调用选择器的基准测试方法
    result = choice.benchmark(*args, out=out)
    # 如果有预期输出，则验证输出是否与预期输出接近
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out)
    # 将测试结果复制到 timings 张量中
    timings.copy_(torch.tensor(result))


class FailChoiceCaller(ChoiceCaller):
    """
    总是抛出运行时错误的选择调用器。
    """
    def benchmark(self, *args, out):
        raise RuntimeError("This choice caller will always throw")


@instantiate_parametrized_tests
class TestMaxAutotune(TestCase):
    """
    测试最大自动调整功能的测试用例类。
    """
    def _create_buffer(self, name, shape):
        """
        创建一个缓冲区对象。
        """
        return Buffer(name, FixedLayout(torch.device("cuda:0"), torch.float32, shape))
    # 定义一个测试方法，用于测试在子进程中选择操作的性能
    def test_benchmark_choice_in_subproc(self):
        # 创建一个返回零张量的函数，用于构建虚拟图形以构建 GraphLowering 实例
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        # 创建 GraphLowering 对象，将虚拟图形传入
        graph = GraphLowering(gm)

        # 设置图形处理器为当前图形处理对象，以便在下面的上下文中创建基准示例值
        with V.set_graph_handler(graph):
            # 创建四个缓冲区，分别命名为 mat1, mat2, mat3, mat4，具有不同的形状
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            # 创建一个固定布局对象，指定在 CUDA 设备上，数据类型为 torch.float32，形状为 (2, 2)
            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))

            # 通过算法选择器缓存获取每个缓冲区的基准示例值
            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            # 获取布局对象的基准示例值
            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # 预期输出为空
            expected_out = None

            # 绑定 aten_mm_plus_mm 操作，传入四个缓冲区和布局对象
            choice = aten_mm_plus_mm.bind((buf1, buf2, buf3, buf4), layout)

            # 使用 torch 创建一个零张量，用于存储性能计时数据
            # 在子进程中对 Python 列表的修改不会同步回父进程
            timings = torch.zeros(3, dtype=torch.float32)
            # 获取一个多进程上下文，使用 spawn 方式创建子进程
            ctx = mp.get_context("spawn")
            # 创建子进程，目标函数是 benchmark_choice，
            # 传入参数为选择操作、四个基准示例值、输出和预期输出以及性能计时数据
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            # 启动子进程并等待其结束
            child.start()
            child.join()
            # 断言子进程的退出码为 0，表示成功
            self.assertEqual(0, child.exitcode)
            # 打印性能计时数据、输出和预期输出
            print(f"timings is {timings}, out {out}, expected_out {expected_out}")
    def test_benchmark_choice_fail_in_subproc(self):
        # 创建一个返回2x3零张量的虚拟图以构建GraphLowering对象
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        # 使用GraphLowering对象处理虚拟图
        graph = GraphLowering(gm)

        # 使用graph作为图处理器来创建以下基准示例值
        with V.set_graph_handler(graph):
            # 创建四个缓冲区，分别命名为mat1、mat2、mat3、mat4，形状分别为(2, 3)和(3, 2)
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            # 创建一个固定布局，位于cuda:0设备上，数据类型为torch.float32，形状为(2, 2)
            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))

            # 使用AlgorithmSelectorCache获取各个缓冲区的基准示例值
            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            # 使用AlgorithmSelectorCache获取布局的基准示例值
            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # 预期输出为(mat1 @ mat2) + (mat3 @ mat4)
            expected_out = (mat1 @ mat2) + (mat3 @ mat4)

            # 创建一个FailChoiceCaller对象，名为"fail_choice_caller"，参数为空列表，传入None
            choice = FailChoiceCaller("fail_choice_caller", [], None)

            # 使用torch.zeros创建一个dtype为torch.float32的3x1全零张量timings
            timings = torch.zeros(3, dtype=torch.float32)
            # 使用'mp.get_context("spawn")'获取spawn上下文
            ctx = mp.get_context("spawn")
            # 创建一个子进程child，目标函数为benchmark_choice，参数为choice, (mat1, mat2, mat3, mat4), out, expected_out, timings
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            # 启动子进程
            child.start()
            # 等待子进程结束
            child.join()
            # 断言子进程的退出码不为0
            self.assertNotEqual(0, child.exitcode)

    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("autotune_multi_device", (True, False))
    def test_max_autotune_mm_plus_mm(self, autotune_in_subproc, autotune_multi_device):
        """
        This crash previously due to a triton issue: https://github.com/openai/triton/issues/1298 .
        With autotuning in subprocess, we don't crash anymore.
        """
        # 定义矩阵维度 m, n, k
        m, n, k = 2048, 1536, 64

        # 定义矩阵乘法加法函数mm_plus_mm，参数为a, b, c, d
        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        # 生成随机数填充的cuda张量a, b, c, d，分别形状为(m, k), (k, n), (m, k), (k, n)
        a = torch.randn(m, k).cuda()
        b = torch.randn(k, n).cuda()
        c = torch.randn(m, k).cuda()
        d = torch.randn(k, n).cuda()

        # 使用config.patch设置最大自动调整为True，自动调整在子进程中的值为autotune_in_subproc，多设备自动调整为autotune_multi_device
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": autotune_in_subproc,
                "autotune_multi_device": autotune_multi_device,
            }
        ):
            # 调用torch.compile对mm_plus_mm进行编译
            torch.compile(mm_plus_mm)(a, b, c, d)
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic):
        """
        Make sure autotuning mm_plus_mm with zero-size input works without crashes.
        """
        # 定义矩阵尺寸：m为0，n为1536，k为64
        m, n, k = 0, 1536, 64

        # 定义矩阵相加函数 mm_plus_mm
        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        # 创建随机数填充的张量 a, b, c, d，并放在 CUDA 上
        a = torch.randn(m, k).cuda()
        b = torch.randn(k, n).cuda()
        c = torch.randn(m, k).cuda()
        d = torch.randn(k, n).cuda()

        # 使用配置修改 max_autotune 为 True，调用编译后的 mm_plus_mm 函数
        with config.patch({"max_autotune": True}):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm(self, dynamic: bool):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        # 定义矩阵乘法函数 mm，对输入矩阵 a 进行 sin 处理后与 b 相乘
        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        # 创建随机数填充的张量 a, b，并放在 CUDA 上
        a = torch.randn(100, 10).cuda()
        b = torch.randn(10, 100).cuda()

        # 使用配置修改 max_autotune 和 autotune_in_subproc 为 True，调用编译后的 mm 函数
        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_zero_size_input(self, dynamic: bool):
        """
        Make sure autotuning mm with zero-size input works without crashes.
        """

        # 定义矩阵乘法函数 mm，对输入矩阵 a 进行 sin 处理后与 b 相乘
        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        # 创建零尺寸的随机数填充的张量 a, b，并放在 CUDA 上
        a = torch.randn(0, 10).cuda()
        b = torch.randn(10, 100).cuda()

        # 使用配置修改 max_autotune 为 True，调用编译后的 mm 函数
        with config.patch({"max_autotune": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @skipIfRocm
    @parametrize("dynamic", (False, True))
    # 定义一个测试函数，用于测试带自动调优远程缓存的最大值
    def test_max_autotune_remote_caching(self, dynamic: bool):
        # 导入 patch 函数用于模拟
        from unittest.mock import patch

        # 定义一个矩阵乘法函数 mm
        def mm(a, b):
            # 对输入的张量 a 求正弦
            a = torch.sin(a)
            # 返回 a 和 b 的矩阵乘法结果
            return a @ b

        # 创建两个随机张量 a 和 b，存储在 GPU 上
        a = torch.randn(100, 10).cuda()
        b = torch.randn(10, 100).cuda()

        # 定义一个简单的模型类 Model，用于返回输入的和
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # 定义一个函数 f，用于创建 Model 类的实例并进行前向传播
        def f(x, y):
            return Model()(x, y)

        # 创建两个随机张量 x 和 y，存储在 GPU 上
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()

        # 初始化缓存相关变量
        cache = {}
        num_get = 0
        num_put = 0

        # 定义一个自定义缓存类 MyCache
        class MyCache:
            def __init__(self, key, is_autotune=False):
                pass

            # 获取缓存中指定文件名的数据
            def get(self, filename):
                nonlocal cache
                nonlocal num_get
                # 如果文件名不在缓存中，返回 None
                if filename not in cache:
                    return None
                # 将缓存中的 JSON 数据解析为 Python 对象
                ret = json.loads(cache[filename])
                num_get += 1
                return ret

            # 将数据放入缓存中
            def put(self, filename, data):
                nonlocal cache
                nonlocal num_put
                # 将数据转换为 JSON 格式并存入缓存
                cache[filename] = json.dumps(data)
                num_put += 1

        # 根据条件选择缓存模块
        cache_module = (
            "triton.fb.fb_memcache.FbMemcacheRemoteAutotuneCacheBackend"
            if config.is_fbcode()
            else "torch._inductor.remote_cache.RedisRemoteCacheBackend"
        )

        # 使用 patch 函数替换部分配置和缓存模块，并创建 MyCache 实例
        with config.patch(
            {
                "autotune_local_cache": False,
                "autotune_remote_cache": True,
            }
        ), patch.dict(os.environ), patch(cache_module, MyCache, create=True):
            # 移除环境变量中的 TRITON_CACHE_MANAGER
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            # 使用 patch 函数设置 max_autotune 为 True
            with config.patch({"max_autotune": True}):
                # 执行四次循环，每次在新的归纳器缓存环境中编译和执行张量运算 mm
                for _ in range(4):
                    with fresh_inductor_cache():
                        torch.compile(mm, dynamic=dynamic)(a, b)
                    # 重置环境
                    reset()
                # 断言 get 操作次数为 3，put 操作次数为 1
                self.assertEqual(num_get, 3)
                self.assertEqual(num_put, 1)

            # 重置 get 和 put 操作次数为 0
            num_get = 0
            num_put = 0

            # 执行四次循环，每次在新的归纳器缓存环境中编译和执行函数 f
            for _ in range(4):
                with fresh_inductor_cache():
                    torch.compile(f, dynamic=dynamic)(x, y)
                # 重置环境
                reset()
            # 断言 get 操作次数为 3，put 操作次数为 1
            self.assertEqual(num_get, 3)
            self.assertEqual(num_put, 1)
    def test_precompilation_threads(self):
        # 导入所需的模块和类
        import threading
        from typing import Any, Dict
        from unittest.mock import Mock, patch

        # 定义一个模拟的 ChoiceCaller 类
        class FakeChoiceCaller(ChoiceCaller):
            def __init__(self):
                # 调用父类构造函数初始化
                super().__init__("none", [], Mock())
                self.thread_id = None  # 初始化线程 ID 为 None

            # 模拟预编译方法，记录当前线程的标识
            def precompile(self):
                self.thread_id = threading.get_ident()

            # 返回空字符串，模拟调用名字的方法
            def call_name(self) -> str:
                return None

            # 返回空值，模拟转换为可调用对象的方法
            def to_callable(self):
                return None

            # 返回空字符串，模拟哈希键的方法
            def hash_key(self) -> str:
                return None

            # 返回空值，模拟输出节点的方法
            def output_node(self) -> "TensorBox":  # noqa: F821
                return None

        # 创建包含 10 个 FakeChoiceCaller 实例的列表
        fake_choices = [FakeChoiceCaller() for i in range(10)]
        
        # 创建一个假的查找结果字典，所有的 FakeChoiceCaller 映射到固定的值 0.123
        fake_lookup_result = {choice: 0.123 for choice in fake_choices}

        # 定义一个无实际操作的查找函数，返回预设的假查找结果
        def no_lookup(
            choices: List[ChoiceCaller],
            op: str,
            inputs: str,
            benchmark: Callable[[Any], Dict[ChoiceCaller, float]],
        ) -> Dict[ChoiceCaller, float]:
            if benchmark is not None:
                return benchmark(choices)

        # 创建 AlgorithmSelectorCache 的实例
        asc = AlgorithmSelectorCache()

        # 定义一个返回假查找结果的基准函数
        def fake_benchmark_fn(*args, **kwargs):
            return fake_lookup_result

        # 记录主线程的 ID
        main_thread_id = threading.get_ident()

        # 创建一个模拟的调试处理器
        mock_debug_handler = Mock()
        # 保存旧的调试处理器引用
        old_debug_handler = V.debug
        
        try:
            # 设置调试处理器为模拟的调试处理器
            V.set_debug_handler(mock_debug_handler)
            
            # 使用 patch 替换 asc 对象的 lookup 方法为 no_lookup 函数
            with patch.object(asc, "lookup", new=no_lookup):
                # 使用 patch 替换 asc 对象的 make_benchmark_fn 方法为 fake_benchmark_fn 函数
                with patch.object(
                    asc, "make_benchmark_fn", return_value=fake_benchmark_fn
                ):
                    # 使用 config.patch 设置两个配置参数
                    with config.patch(
                        {
                            "autotune_in_subproc": False,
                            "compile_threads": len(fake_choices),
                        }
                    ):
                        # 调用 asc 的方法，模拟算法选择器的调用
                        asc("test_call", fake_choices, [], Mock())
            
            # 验证每个 fake_choice 实例的 precompile 方法是否被调用
            for fake_choice in fake_choices:
                assert (
                    fake_choice.thread_id is not None
                ), "Expected all ChoiceCaller's precompile method to have been called"
                assert (
                    fake_choice.thread_id != main_thread_id
                ), "Expected all ChoiceCaller's precompile method to have been called on separate thread"
        
        finally:
            # 恢复旧的调试处理器
            V.set_debug_handler(old_debug_handler)
    def test_max_autotune_addmm(self, dynamic=False):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        # 禁用 FP16 降低精度优化
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义 addmm 函数，使用 torch.addmm 执行矩阵乘法并添加到输入张量 x 上
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        # 生成随机张量并移至 CUDA 设备
        x = torch.randn(100).cuda()
        a = torch.randn(100, 10).cuda()
        b = torch.randn(10, 100).cuda()

        # 使用配置上下文管理器启用最大自动调整和子进程自动调整
        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            # 使用动态编译执行 addmm 函数
            Y_compiled = torch.compile(addmm, dynamic=dynamic)(x, a, b)
            # 直接调用 addmm 函数
            Y = addmm(x, a, b)
            # 断言两种方式的计算结果接近
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_zero_size_input(self, dynamic):
        """
        Make sure autotuning addmm with zero-size input works without crashes.
        """

        # 定义 addmm 函数，使用 torch.addmm 执行矩阵乘法并添加到输入张量 x 上
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        # 生成随机张量并移至 CUDA 设备
        x = torch.randn(100).cuda()
        a = torch.randn(0, 10).cuda()  # 生成零行的张量
        b = torch.randn(10, 100).cuda()

        # 使用配置上下文管理器启用最大自动调整
        with config.patch({"max_autotune": True}):
            # 使用动态编译执行 addmm 函数
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

    @skipIfRocm
    def test_autotune_conv1x1(self):
        # 假设输入有 3 个通道，输出为 16 个通道
        conv1x1 = (
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
            .to(memory_format=torch.channels_last)  # 转换内存格式为 channels_last
            .cuda()
        )

        # 示例输入张量：批次大小 = 4，通道数 = 3，高度 = 32，宽度 = 32
        # 设置内存格式为 `channels_last`
        input_tensor = (
            torch.randn(4, 3, 32, 32)
            .contiguous(memory_format=torch.channels_last)  # 保证连续性
            .cuda()
        )

        # 使用配置上下文管理器启用最大自动调整和 TRITON 后端的 GEMM 自动调整
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
        ):

            # 使用 torch.compile 装饰器定义动态编译函数 foo
            @torch.compile()
            def foo(mod, x):
                return mod(x)

            # 在无梯度计算环境下运行 foo 函数和 conv1x1 模型以获取输出和代码
            with torch.no_grad():
                out, code = run_and_get_code(foo, conv1x1, input_tensor)

            # 使用 FileCheck 检查生成的代码中是否没有 "extern_kernels.convolution" 的内容
            FileCheck().check_not("extern_kernels.convolution").run(code[0])
            # 断言 conv1x1 模型在输入张量上的输出与预期输出接近
            self.assertEqual(conv1x1(input_tensor), out, atol=1e-2, rtol=0)

    @skipIfRocm
    def test_filled_cache_precompile(self):
        # 定义函数 fn，执行两次矩阵乘法，并将结果转换为 float16 类型
        def fn(a, b, c):
            a = (a @ b) @ c
            a, b, c = (t.to(torch.float16) for t in [a, b, c])
            return (a @ b) @ c

        # 使用 torch.compile 装饰器以最大自动调整和禁用 cudagraphs 模式编译函数 fn
        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)

        # 生成随机输入张量列表并移至 CUDA 设备
        inputs = [torch.rand([256, 256], device="cuda") for _ in range(3)]

        # 重置 Dynamo 状态和清除计数器
        from torch._dynamo.utils import counters
        torch._dynamo.reset()
        counters.clear()

        # 使用编译后的 fn_c 函数执行计算，并断言与未编译的 fn 函数结果接近
        self.assertEqual(fn(*inputs), fn_c(*inputs), atol=1e-2, rtol=1e-2)

        # 再次重置 Dynamo 状态和清除计数器
        torch._dynamo.reset()
        counters.clear()

        # 使用编译后的 fn_c 函数执行计算，并断言未执行选择算法预编译的计数器为零
        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 0)

    @skipIfRocm
    @fresh_inductor_cache()
    # 使用 config.patch 装饰器设置 search_autotune_cache=True，指定测试方法要开启搜索自动调整缓存功能
    @config.patch(search_autotune_cache=True)
    # 定义测试方法 test_search_autotune_cache
    def test_search_autotune_cache(self):
        # 定义内部函数 fn，实现矩阵乘法的计算和类型转换
        def fn(a, b, c):
            a = (a @ b) @ c  # 连续进行矩阵乘法运算
            a, b, c = (t.to(torch.float16) for t in [a, b, c])  # 将输入张量转换为半精度浮点数类型
            return (a @ b) @ c  # 再次进行矩阵乘法运算

        # 使用 torch.compile() 编译函数 fn
        fn_c = torch.compile()(fn)
        # 生成三个随机张量作为输入，存储在列表中，并且在 GPU 上进行计算
        inputs = [torch.rand([256, 256], device="cuda") for _ in range(3)]
        # 导入计数器，用于检查选择算法预编译的次数
        from torch._dynamo.utils import counters

        # 断言调用 fn 和 fn_c 的结果是否接近，指定绝对误差和相对误差容忍度为 1e-2
        self.assertEqual(fn(*inputs), fn_c(*inputs), atol=1e-2, rtol=1e-2)
        # 断言选择算法预编译的次数是否为 0
        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 0)

    # 使用 skipIfRocm 装饰器跳过 ROCm 平台的测试
    @skipIfRocm
    # 使用 fresh_inductor_cache 装饰器刷新导入器缓存
    @fresh_inductor_cache()
    # 使用 config.patch 设置 max_autotune=True 和 max_fusion_size=2，指定测试方法要开启最大自动调整和最大融合大小功能
    @config.patch(max_autotune=True, max_fusion_size=2)
    # 定义测试方法 test_jit_fusion_matches_aot_fusion
    def test_jit_fusion_matches_aot_fusion(self):
        # 定义函数 fn，接受两个参数 x 和 number
        def fn(x, number):
            buf0 = x + x  # 计算 x 的加法
            buf1 = number.item()  # 获取 number 的标量值
            buf2 = x * x  # 计算 x 的乘法
            buf3 = x @ x  # 使用 @ 运算符进行矩阵乘法，创建 MultiTemplateBuffer
            buf4 = x**2  # 计算 x 的平方
            return buf0, buf1, buf2, buf3, buf4  # 返回所有计算结果

        # 定义输入参数 inputs，包括一个随机张量和一个标量张量，存储在 GPU 上
        inputs = (torch.rand([256, 256], device="cuda"), torch.tensor(3, device="cuda"))
        # 使用 torch._export.aot_compile() 将函数 fn 进行 AOT 编译
        torch._export.aot_compile(fn, args=inputs)

    # 使用 config.patch 设置 autotune_local_cache=False 和 autotune_remote_cache=False，关闭本地和远程自动调整缓存
    @config.patch(autotune_local_cache=False, autotune_remote_cache=False)
    # 使用 skipIfRocm 装饰器跳过 ROCm 平台的测试
    @skipIfRocm
    # 定义测试方法 test_precompilations
    def test_precompilations(self):
        # 定义内部函数 fn，实现矩阵乘法的计算和类型转换
        def fn(a, b, c):
            a = (a @ b) @ c  # 连续进行矩阵乘法运算
            a, b, c = (t.to(torch.float16) for t in [a, b, c])  # 将输入张量转换为半精度浮点数类型
            return (a @ b) @ c  # 再次进行矩阵乘法运算

        # 使用 torch.compile(mode="max-autotune-no-cudagraphs") 编译函数 fn
        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        # 生成三个随机张量作为输入，存储在列表中，并且在 GPU 上进行计算
        inputs = [torch.rand([256, 256], device="cuda") for _ in range(3)]

        # 断言调用 fn 和 fn_c 的结果是否接近，指定绝对误差和相对误差容忍度为 1e-2
        self.assertEqual(fn(*inputs), fn_c(*inputs), atol=1e-2, rtol=1e-2)

        # 导入计数器，用于检查选择算法预编译的次数
        from torch._dynamo.utils import counters
        # 断言选择算法预编译的次数是否为 2
        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 2)

    # 定义测试方法 test_cat_addmm
    def test_cat_addmm(self):
        # 定义函数 fn，接受三个输入张量 a, b, c，执行 torch.cat 和 torch.addmm 操作
        def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            return torch.cat(
                [
                    torch.addmm(a, b, c),  # 使用 torch.addmm 进行矩阵乘法运算并拼接结果
                    torch.addmm(b, c, a),  # 使用 torch.addmm 进行矩阵乘法运算并拼接结果
                ],
                1,  # 指定拼接维度为 1
            )

        # 定义输入参数 args，包含三个随机张量，存储在 GPU 上
        args = [
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
            torch.randn(4, 4, device="cuda"),
        ]
        # 使用 config.patch 设置 max_autotune=True 和 max_autotune_gemm_backends="Triton"
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            # 计算预期结果和实际结果，使用 torch.compile 编译函数 fn，并进行断言
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    # 定义一个测试函数，用于测试带有 epilogue 和动态形状的 Triton 模板
    def test_triton_template_with_epilogues_and_dynamic_shape(self):
        # 定义一个函数 fn，接受四个张量参数并返回一个张量
        def fn(
            x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, mul: torch.Tensor
        ) -> torch.Tensor:
            # 计算矩阵乘法并施加 ReLU 激活函数
            return (
                torch.nn.functional.relu(
                    torch.matmul(torch.transpose(x, 0, 1), torch.transpose(w, 0, 1))
                    + bias
                )
                * mul
            )

        # 设置几个常量
        M0 = 5
        M1 = 8
        K = 4
        N = 3
        # 生成随机权重张量并移至 GPU 并转换为半精度
        w = torch.rand(N, K).cuda().half()
        # 生成随机偏置张量并移至 GPU 并转换为半精度
        b = torch.rand(N).cuda().half()

        # 使用配置块修改全局设置
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            # 编译函数 fn，并设置编译参数
            compiled_fn = torch.compile(
                fn, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs"
            )

            # 生成随机输入张量 x0 并移至 GPU 并转换为半精度
            x0 = torch.rand(K, M0).cuda().half()
            # 生成随机乘数张量 mul0 并移至 GPU 并转换为半精度
            mul0 = torch.rand(M0, N).cuda().half()
            # 调用编译后的函数计算结果 y0
            y0 = compiled_fn(x0, w, b, mul0)
            # 计算预期结果 y0_expected
            y0_expected = fn(x0, w, b, mul0)
            # 断言编译后的结果与预期结果接近
            torch.testing.assert_close(y0, y0_expected)

            # 生成随机输入张量 x1 并移至 GPU 并转换为半精度
            x1 = torch.rand(K, M1).cuda().half()
            # 生成随机乘数张量 mul1 并移至 GPU 并转换为半精度
            mul1 = torch.rand(M1, N).cuda().half()
            # 调用编译后的函数计算结果 y1
            y1 = compiled_fn(x1, w, b, mul1)
            # 计算预期结果 y1_expected
            y1_expected = fn(x1, w, b, mul1)
            # 断言编译后的结果与预期结果接近
            torch.testing.assert_close(y1, y1_expected)

    # 使用配置块设置一些测试相关的参数
    @config.patch(
        benchmark_kernel=True,
        fallback_random=True,
        max_autotune_gemm=True,
    )
    # 使用参数化装饰器设置设备为 cpu 和 cuda 进行测试
    @parametrize("device", ("cpu", "cuda"))
    def test_matmul_dropout(self, device):
        # 定义前向传播函数 fwd
        def fwd(a, b):
            # 计算矩阵乘法
            x = a @ b
            # 应用 dropout 操作
            x = torch.nn.functional.dropout(x, 0.1)
            return x

        # 定义函数 fn，计算 fwd 的结果并返回梯度
        def fn(a, b):
            x = fwd(a, b).sum()
            x.backward()
            return a.grad

        # 设置矩阵大小
        N = 128
        # 生成随机输入张量 a，并根据设备选择移至 CPU 或 GPU，并设置为需要梯度计算
        a = torch.randn(N, N, device=device, requires_grad=True)
        # 生成随机输入张量 b，并根据设备选择移至 CPU 或 GPU
        b = torch.randn(N, N, device=device)

        # 编译优化后的函数 fn
        opt_fn = torch.compile(fn)
        # 重置随机数生成器状态
        reset_rng_state()
        # 计算参考结果 ref
        ref = fn(a, b)
        # 重置随机数生成器状态
        reset_rng_state()
        # 计算优化后的结果 act
        act = opt_fn(a, b)

        # 如果矩阵大小小于等于 8，则打印参考结果和优化结果
        if N <= 8:
            print(f"ref\n{ref}\nact\n{act}")
        # 断言优化后的结果与参考结果接近
        torch.testing.assert_close(ref, act, atol=1e-1, rtol=1e-1)

    # 使用配置块设置 max_autotune 参数
    @config.patch(max_autotune=True)
    # 使用 unittest 装饰器跳过条件不满足的测试
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least 2 devices for this test"
    )
    # 测试自动调优设备保护
    def test_autotune_device_guard(self):
        # 生成随机输入张量 x 和 y 并将它们移至第二个 CUDA 设备
        x = torch.randn(1024, 1024, device="cuda:1")
        y = torch.randn(1024, 1024, device="cuda:1")

        # 定义函数 f，计算矩阵乘法
        def f(x, y):
            return x @ y

        # 使用 fresh_inductor_cache 函数重置缓存状态
        with fresh_inductor_cache():
            # 编译优化后的函数 f，并计算结果 act
            act = torch.compile(f)(x, y)
        # 计算参考结果 ref
        ref = f(x, y)
        # 断言优化后的结果与参考结果在一定的误差范围内接近
        self.assertTrue(torch.allclose(act, ref, atol=4 * 1e-3, rtol=4 * 1e-3))
    def test_non_contiguous_input_mm(self):
        """
        Make sure the triton template can work with non-contiguous inputs without crash.
        Check https://github.com/pytorch/pytorch/issues/125437 for more details.
        """
        # 创建非连续布局的输入张量 x 和 y，使用 torch.bfloat16 数据类型，在 CUDA 设备上进行初始化
        x = rand_strided(
            (50257, 32768), (1, 50304), dtype=torch.bfloat16, device="cuda"
        )
        y = rand_strided((32768, 768), (768, 1), dtype=torch.bfloat16, device="cuda")

        @torch.compile(mode="max-autotune")
        def f(x, y):
            # 执行矩阵乘法 x @ y
            return x @ y

        # 计算参考结果
        ref = x @ y
        # 使用优化后的函数计算实际结果
        act = f(x, y)
        # 断言优化后的结果与参考结果在给定的误差容限内接近
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))

    def test_non_contiguous_input_addmm(self):
        # 创建随机的偏置 b，使用 torch.bfloat16 数据类型，在 CUDA 设备上进行初始化
        b = torch.randn((768), dtype=torch.bfloat16, device="cuda")
        # 创建非连续布局的输入张量 x 和 y，使用 torch.bfloat16 数据类型，在 CUDA 设备上进行初始化
        x = rand_strided(
            (50257, 32768), (1, 50304), dtype=torch.bfloat16, device="cuda"
        )
        y = rand_strided((32768, 768), (768, 1), dtype=torch.bfloat16, device="cuda")

        @torch.compile(mode="max-autotune")
        def f(x, y):
            # 执行 b + x @ y 的矩阵乘法
            return torch.addmm(b, x, y)

        # 计算参考结果
        ref = torch.addmm(b, x, y)
        # 使用优化后的函数计算实际结果
        act = f(x, y)
        # 断言优化后的结果与参考结果在给定的误差容限内接近
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
    # 定义一个测试函数，用于测试非连续输入的 torch.bmm 操作
    def test_non_contiguous_input_bmm(self):
        # 生成一个随机的 strided tensor x，形状为 (1, 50257, 32768)，步长为 (0, 1, 50304)，数据类型为 torch.bfloat16，在 CUDA 设备上
        x = rand_strided(
            (1, 50257, 32768), (0, 1, 50304), dtype=torch.bfloat16, device="cuda"
        )
        # 生成一个随机的 strided tensor y，形状为 (1, 32768, 768)，步长为 (0, 768, 1)，数据类型为 torch.bfloat16，在 CUDA 设备上
        y = rand_strided(
            (1, 32768, 768), (0, 768, 1), dtype=torch.bfloat16, device="cuda"
        )

        # 使用 torch.compile 注解，设置编译模式为 "max-autotune"，定义一个函数 f，计算两个输入张量的 bmm 结果
        @torch.compile(mode="max-autotune")
        def f(x, y):
            return torch.bmm(x, y)

        # 计算参考结果，即未经编译优化的 torch.bmm 操作结果
        ref = torch.bmm(x, y)
        # 调用编译后的函数 f，计算编译优化后的 torch.bmm 操作结果
        act = f(x, y)
        # 断言两种结果在一定误差范围内相等
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))

    # 定义一个测试函数，用于测试非连续输入的 torch.mm 加 torch.mm 操作
    def test_non_contiguous_input_mm_plus_mm(self):
        # 生成两对随机的 strided tensor x1, y1 和 x2, y2，形状和步长相似，数据类型为默认浮点型，在 CUDA 设备上
        x1 = rand_strided((50257, 32768), (1, 50304), device="cuda")
        y1 = rand_strided((32768, 768), (768, 1), device="cuda")

        x2 = rand_strided((50257, 32768), (1, 50304), device="cuda")
        y2 = rand_strided((32768, 768), (768, 1), device="cuda")

        # 使用 torch.compile 注解，设置编译模式为 "max-autotune"，定义一个函数 f，计算两组输入张量的 mm 结果后相加
        @torch.compile(mode="max-autotune")
        def f(x1, y1, x2, y2):
            return x1 @ y1 + x2 @ y2

        # 计算参考结果，即未经编译优化的 torch.mm 加 torch.mm 操作结果
        ref = x1 @ y1 + x2 @ y2
        # 调用编译后的函数 f，计算编译优化后的 torch.mm 加 torch.mm 操作结果
        act = f(x1, y1, x2, y2)
        # 断言两种结果在一定误差范围内相等
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))

    # 使用 config.patch 注解，配置最大自动调优、GEMM 后端为 "TRITON"，不回退至原生后端的设置
    def test_no_valid_choices(self):
        # 创建两个零张量 a 和 b，形状为 [2, 2]，在 CUDA 设备上
        a = torch.zeros([2, 2], device="cuda")
        b = torch.zeros([2, 2], device="cuda")
        # 使用 assertRaises 上下文管理器，验证在编译失败时抛出 BackendCompilerFailed 异常，并捕获异常上下文
        with self.assertRaises(BackendCompilerFailed) as context:
            # 使用 torch.compile 注解，尝试编译执行矩阵相乘操作 a.matmul(b)
            torch.compile(lambda a, b: a.matmul(b))(a, b)
        # 断言异常信息中包含 "NoValidChoicesError"
        self.assertIn("NoValidChoicesError", str(context.exception))

    # 使用 parametrize 注解，指定 multi_template 参数为 True 和 False，同时使用 config.patch 注解进行相关配置
    def test_inf_timing(self, multi_template):
        from unittest.mock import patch

        # 保存原始的 AlgorithmSelectorCache.lookup 方法
        lookup = AlgorithmSelectorCache.lookup

        # 定义一个 mock_lookup 方法替换 AlgorithmSelectorCache.lookup 方法，返回所有选择的执行时间为无穷大
        def mock_lookup(self, *args, **kwargs):
            timings = lookup(self, *args, **kwargs)
            return {choice: float("inf") for choice in timings.keys()}

        # 创建两个零张量 a 和 b，形状为 [16, 16]，在 CUDA 设备上
        a = torch.zeros([16, 16], device="cuda")
        b = torch.zeros([16, 16], device="cuda")
        
        # 使用 patch.object 上下文管理器，将 AlgorithmSelectorCache 的 lookup 方法替换为 mock_lookup 方法，
        # 同时使用 config.patch 注解配置 benchmark_epilogue_fusion 参数
        with patch.object(AlgorithmSelectorCache, "lookup", mock_lookup), config.patch(
            benchmark_epilogue_fusion=multi_template
        ):
            # 使用 assertRaises 上下文管理器，验证在编译失败时抛出 BackendCompilerFailed 异常，并捕获异常上下文
            with self.assertRaises(BackendCompilerFailed) as context:
                # 使用 torch.compile 注解，尝试编译执行矩阵相乘操作 a.matmul(b)
                torch.compile(lambda a, b: a.matmul(b))(a, b)
            # 断言异常信息中包含 "NoValidChoicesError"
            self.assertIn("NoValidChoicesError", str(context.exception))
# 定义一个基于BenchmarkRequest的测试类TestBenchmarkRequest
class TestBenchmarkRequest(BenchmarkRequest):
    # 初始化方法，接受一个浮点数value，一个布尔值multi_device和一个可选的字符串parent_visible_devices作为参数
    def __init__(
        self, value: float, multi_device: bool, parent_visible_devices: Optional[str]
    ) -> None:
        # 将参数赋值给对象的属性
        self.value = value
        self.multi_device = multi_device
        self.parent_visible_devices = parent_visible_devices

    # benchmark方法用于执行基准测试，接受多个torch.Tensor作为输入参数和一个可选的torch.Tensor作为输出参数，返回一个浮点数
    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        # 获取环境变量CUDA_VISIBLE_DEVICES的值
        visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES)
        
        # 如果multi_device为False，则验证visible_devices与parent_visible_devices相等
        if not self.multi_device:
            assert visible_devices == self.parent_visible_devices
        else:
            # 如果multi_device为True，则将parent_visible_devices按逗号分隔成多个有效设备，并验证visible_devices存在于其中
            valid_devices = self.parent_visible_devices.split(",")
            assert visible_devices in valid_devices
        
        # 返回对象的value属性作为基准测试结果
        return self.value


# 定义一个基于TritonTemplateCaller的测试类TestTritonTemplateCaller
class TestTritonTemplateCaller(TritonTemplateCaller):
    # 初始化方法，接受一个TestBenchmarkRequest对象bmreq作为参数
    def __init__(self, bmreq: TestBenchmarkRequest):
        self.bmreq = bmreq

    # __str__方法返回字符串"test"
    def __str__(self) -> str:
        return "test"


# 定义一个测试类TestTuningProcess，继承自TestCase
class TestTuningProcess(TestCase):
    # 定义测试方法test_tuning_pool_crash
    def test_tuning_pool_crash(self):
        # 使用config.patch方法设置autotune_multi_device为False，以便禁用多设备自动调整
        with config.patch({"autotune_multi_device": False}):
            # 初始化TuningProcessPool对象tuning_pool
            tuning_pool = TuningProcessPool()
            tuning_pool.initialize()

            # 创建一个TestBenchmarkRequest对象bmreq，parent_visible_devices设置为"invalid"
            bmreq = TestBenchmarkRequest(3.14, False, "invalid")
            # 创建一个TestTritonTemplateCaller对象choice，传入bmreq
            choice = TestTritonTemplateCaller(bmreq)

            # 调用tuning_pool的benchmark方法，传入包含choice的列表，返回基准测试结果timings
            timings = tuning_pool.benchmark([choice])
            # 断言choice在timings中
            self.assertTrue(choice in timings)
            # 断言timings[choice]等于正无穷
            self.assertEqual(timings[choice], float("inf"))

            # 修改choice.bmreq的parent_visible_devices属性为环境变量CUDA_VISIBLE_DEVICES的值
            choice.bmreq.parent_visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES)

            # 再次调用tuning_pool的benchmark方法，传入包含choice的列表，返回基准测试结果timings
            timings = tuning_pool.benchmark([choice])
            # 断言choice在timings中
            self.assertTrue(choice in timings)
            # 断言timings[choice]等于bmreq的value属性
            self.assertEqual(timings[choice], bmreq.value)

            # 终止tuning_pool
            tuning_pool.terminate()
    # 定义一个测试函数，用于测试在多设备环境中的调优池功能
    def test_tuning_pool_multiple_devices(self):
        # 使用配置管理器设置自动调优多设备为 True
        with config.patch({"autotune_multi_device": True}):
            # 根据 CUDA_VISIBLE_DEVICES 环境变量的情况确定可见设备的子集，
            # 如果环境变量中已经设置了 CUDA_VISIBLE_DEVICES，则使用其值，否则使用所有可用设备
            if CUDA_VISIBLE_DEVICES in os.environ:
                visible_devices = os.environ[CUDA_VISIBLE_DEVICES].split(",")
            else:
                visible_devices = [str(d) for d in range(torch.cuda.device_count())]

            # 选择最后两个可见设备，并将其索引作为字符串合并为一个逗号分隔的字符串
            parent_visible_devices = ",".join(visible_devices[-2:])
            # 将选择的设备设置为 CUDA_VISIBLE_DEVICES 环境变量的值
            os.environ[CUDA_VISIBLE_DEVICES] = parent_visible_devices

            # 初始化调优进程池对象
            tuning_pool = TuningProcessPool()
            tuning_pool.initialize()

            # 创建两个测试 Triton 模板调用对象，每个对象使用不同的基准请求对象
            choice1 = TestTritonTemplateCaller(
                TestBenchmarkRequest(3.14, True, parent_visible_devices),
            )
            choice2 = TestTritonTemplateCaller(
                TestBenchmarkRequest(2.718, True, parent_visible_devices),
            )

            # 在调优池中进行基准测试，返回每个选择对象的时间性能数据字典
            timings = tuning_pool.benchmark([choice1, choice2])
            # 断言测试结果，确保每个选择对象的基准请求值等于其对应的时间性能值
            self.assertEqual(timings[choice1], choice1.bmreq.value)
            self.assertEqual(timings[choice2], choice2.bmreq.value)

            # 终止调优进程池
            tuning_pool.terminate()
if __name__ == "__main__":
    # 检查当前模块是否作为主程序执行
    from torch._inductor.utils import is_big_gpu

    # 设置环境变量以在CI环境中执行测试
    if HAS_CUDA and HAS_CPU and is_big_gpu(0):
        # 如果有CUDA、CPU，并且检测到大型GPU设备（索引为0），则运行测试
        run_tests()
```