# `.\pytorch\test\inductor\test_benchmark_fusion.py`

```
# Owner(s): ["module: inductor"]
# 导入标准库模块
import math
import os
import sys

# 导入第三方库模块
import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    slowTest,
    TEST_WITH_ASAN,
)

# 导入项目内部模块
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

# 将测试文件目录加入到系统路径中，使得其中的文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入上下文管理和单元测试模块
import contextlib
import unittest

# 导入项目内部配置和调度器模块
from torch._inductor import config
from torch._inductor.scheduler import Scheduler

# 如果运行环境为 Windows 且是在持续集成环境下
if IS_WINDOWS and IS_CI:
    # 输出警告信息并退出
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    # 抛出跳过测试的异常
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 从项目内部测试模块导入相关函数和类
from inductor.test_torchinductor import check_model, check_model_cuda, copy_tests

# 定义测试用例类，继承自InductorTestCase
class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 设置类级别的上下文管理器，用于配置基准测试相关选项
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "benchmark_kernel": True,
                    "benchmark_fusion": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # 关闭类级别的上下文管理器
        cls._stack.close()
        super().tearDownClass()

# 定义基准融合测试模板类
class BenchmarkFusionTestTemplate:
    # 测试 softmax 函数
    def test_softmax(self):
        # 定义一个简单的函数 f，对输入进行 softmax 操作
        def f(x):
            return torch.nn.functional.softmax(x, dim=-1)

        # 调用通用测试方法，传入随机生成的张量作为参数
        self.common(f, (torch.rand(2, 8192),))

    # 使用 @slowTest 装饰器标记的测试方法，测试 ResNet-18 模型
    @slowTest
    def test_resnet18(self):
        # 导入 torchvision 模块
        import torchvision

        # 创建 ResNet-18 模型实例，并设置为评估模式
        model = torchvision.models.resnet18()
        model.eval()
        batch_size = 16
        # 创建输入张量
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
        # 调用通用测试方法，传入模型和输入张量作为参数，设置允许的误差范围
        self.common(model, inputs, atol=1e-2, rtol=1e-2)
    def test_register_spills(self):
        """
        The test can potentially trigger register spills
        """
        # 保存原来的 benchmark_fused_nodes 函数引用
        old_benchmark_fn = Scheduler.benchmark_fused_nodes

        # 定义新的 benchmark_fused_nodes 函数
        def new_benchmark_fn(scheduler, nodes):
            """
            We override Scheduler.benchmark_fused_nodes to return latency 1.0
            if there are no register spills. Without this, we may not able to
            test the code path handling register spilling because before register
            start spilling, the related fusion may have already been skipped
            due to longer lantency.
            """
            # 调用原 benchmark_fused_nodes 函数并获取返回值
            ms, path = old_benchmark_fn(scheduler, nodes)
            # 如果返回的时间不是无穷大，将其设为 1.0
            if not math.isinf(ms):
                ms = 1.0
            return ms, path

        # 禁用 dynamic_scale_rblock 以便更容易触发寄存器溢出
        with unittest.mock.patch.object(
            Scheduler, "benchmark_fused_nodes", new_benchmark_fn
        ), config.patch("dynamic_scale_rblock", False):
            S = 512

            # 定义函数 f，对输入进行处理并返回输出
            def f(*inputs):
                inputs = list(inputs)
                outputs = []
                out = torch.zeros(S, device=self.device)
                for x in inputs:
                    x = x * 2
                    x = x + 1
                    x = x.sum(dim=-1)
                    outputs.append(x)
                    out = out + x
                return outputs, out

            # 从环境变量中获取输入数量 N，如果不存在则默认为 30
            N = int(os.environ.get("NINP", "30"))
            # 生成 N 个随机输入的列表
            inputs = [torch.randn(S, 2560, device=self.device) for _ in range(N)]
            # 编译函数 f 以进行优化
            opt_f = torch.compile(f)
            opt_f(*inputs)

    def test_foreach_kernel(self):
        """
        Benchmark fusion should skip benchmarking kernels involves foreach kernel
        for now. Without the skipping logic, `codegen_node_schedule` may fail.
        """
        # 创建两个随机张量 a 和 b
        a = torch.randn(1024, 256, device=self.device)
        b = torch.randn(1024, 512, device=self.device)

        # 定义函数 f，对输入进行 _foreach_abs 操作并返回结果
        def f(a, b):
            a, b = torch._foreach_abs([a, b])
            return a + 1, b + 2

        # 调用类中的 common 方法，传入函数 f 和其参数
        self.common(f, (a, b))

    @torch._inductor.config.patch(max_autotune_gemm_backends="TRITON")
    def test_avoid_register_spilling(self):
        # 检查设备是否为 CUDA，如果不是则跳过测试
        if self.device != "cuda":
            raise unittest.SkipTest("CUDA only")

        # 导入 torch.nn.functional 模块中的 gelu 函数
        from torch.nn.functional import gelu

        # 定义一个函数 foo，接受模型 m 和输入 inp，执行一系列操作并返回结果
        def foo(m, inp):
            # 对输入 inp 使用模型 m 进行计算
            curr = m(inp)
            # 初始化临时变量列表
            tmps = []
            # 循环执行 gelu 函数和临时变量相加操作，共执行 4 次
            for _ in range(4):
                curr = gelu(curr)
                for t in tmps:
                    curr = curr + t
                tmps.append(curr)

            # 返回最终结果 curr
            return curr

        # 创建一个包含 2048 个输入和输出的全连接线性模型，使用半精度浮点数，并将其移至 CUDA 设备
        m = torch.nn.Linear(2048, 2048, bias=True).half().cuda()
        # 生成一个 2048x2048 的随机输入张量，使用半精度浮点数，并将其移至 CUDA 设备
        inp = torch.rand([2048, 2048]).half().cuda()

        # 在无梯度计算环境中执行函数 foo 的编译，使用最大自动调优且不使用 CUDA 图
        with torch.no_grad():
            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            # 运行函数 foo_c，并获取其生成的代码
            _, out_code = run_and_get_code(foo_c, m, inp)

            # 偶尔，CI 环境可能将此合并为一个内核，此时跳过测试
            if not out_code[0].count("def triton_") == 2:
                return

            # 应该有多个 Triton 调用
            # 使用 FileCheck 来验证生成的代码中是否包含预期的内容
            FileCheck().check("async_compile.wait").check_count(
                ".run", 2, exactly=True
            ).run(out_code[0])

        # 在配置上下文中，禁用性能融合选项，并在无梯度计算环境中执行
        with config.patch(
            {"benchmark_fusion": False, "epilogue_fusion": False}
        ), torch.no_grad():
            # 重置 Dynamo 计算图优化器
            torch._dynamo.reset()

            # 再次在无梯度计算环境中，执行函数 foo 的编译，使用最大自动调优且不使用 CUDA 图
            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            # 再次运行函数 foo_c，并获取其生成的代码
            _, out_code2 = run_and_get_code(foo_c, m, inp)

        # 遍历生成的代码 out_code[0] 和 out_code2[0]
        for c in out_code[0], out_code2[0]:
            # 使用 FileCheck 来验证生成的代码中是否包含预期的内容
            FileCheck().check("async_compile.wait").check("DeviceGuard").check_count(
                "empty_strided_cuda", 2, exactly=True
            ).check("return").run(c)
# 如果系统支持 CUDA 并且不是用 AddressSanitizer 进行测试
if HAS_CUDA and not TEST_WITH_ASAN:

    # 定义基于 CUDA 的基准融合测试类
    class BenchmarkFusionCudaTest(TestCase):
        # 共同的测试方法来自于检查模型是否支持 CUDA
        common = check_model_cuda
        # 设备类型为 CUDA
        device = "cuda"

    # 复制基准融合测试模板的测试用例到 CUDA 测试类中
    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCudaTest, "cuda")

# 如果系统支持 CPU 并且未启用 Torch MPS（内存池分配服务）
if HAS_CPU and not torch.backends.mps.is_available():

    # 定义基于 CPU 的基准融合测试类
    class BenchmarkFusionCpuTest(TestCase):
        # 共同的测试方法来自于检查模型
        common = check_model
        # 设备类型为 CPU
        device = "cpu"

    # 复制基准融合测试模板的测试用例到 CPU 测试类中
    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCpuTest, "cpu")

# 如果作为主程序运行
if __name__ == "__main__":
    # 导入运行测试的函数
    from torch._inductor.test_case import run_tests

    # 如果系统支持 CPU 或者支持 CUDA
    if HAS_CPU or HAS_CUDA:
        # 运行测试
        run_tests()
```