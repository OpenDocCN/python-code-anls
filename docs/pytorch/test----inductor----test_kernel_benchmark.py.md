# `.\pytorch\test\inductor\test_kernel_benchmark.py`

```py
# Owner(s): ["module: inductor"]
# 导入必要的模块和类
import contextlib  # 提供上下文管理工具的模块
import os  # 提供与操作系统交互的功能
import subprocess  # 提供创建子进程的功能
import sys  # 提供访问与 Python 解释器相关的变量和函数
from unittest.mock import patch  # 提供在测试过程中替换对象的模块

import torch  # PyTorch 深度学习库
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.testing import rand_strided  # 引入随机数据生成的测试工具
from torch._inductor import config  # 加载与模块编译器配置相关的模块
from torch._inductor.codecache import PyCodeCache  # 加载 PyCodeCache 类，用于缓存编译的代码
from torch._inductor.test_case import run_tests, TestCase  # 加载测试用例相关的模块和类
from torch._inductor.utils import fresh_inductor_cache  # 加载用于刷新编译器缓存的工具函数
from torch.testing import FileCheck  # 加载用于测试文件内容的工具
from torch.testing._internal.common_device_type import expectedFailureXPU  # 加载用于定义预期失败的设备类型
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU  # 加载用于模块编译的 GPU 类型和 GPU 存在检查

class TestKernelBenchmark(TestCase):
    device_type = GPU_TYPE  # 设置测试的设备类型为 GPU

    @classmethod
    def setUpClass(cls):
        cls.exit_stack = contextlib.ExitStack()  # 创建上下文管理的堆栈对象
        cls.exit_stack.enter_context(patch.object(config, "benchmark_kernel", True))  # 在配置模块中打开 benchmark_kernel 标志

    @classmethod
    def tearDownClass(cls):
        cls.exit_stack.close()  # 关闭上下文管理的堆栈对象

    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        PyCodeCache.cache.clear()  # 清空编译代码的缓存

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.cache.values():  # 遍历编译代码缓存中的值
            if hasattr(v, "benchmark_compiled_module"):  # 检查对象是否具有 benchmark_compiled_module 属性
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)  # 断言已找到编译模块对象
        return compiled_module  # 返回找到的编译模块对象

    def verify_compiled_kernels(self, GB_count=1):
        compiled_module = self.get_compiled_module()  # 获取编译后的模块对象

        # 在子进程中运行编译后的模块，并检查其输出
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -kc".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # 确保输出中包含带宽信息
        FileCheck().check_count(
            "GB/s",
            GB_count,
            exactly=1,
        ).run(bench_out)
    def verify_remove_inductor_deps(self, compiled_module):
        try:
            # 运行编译后的模块，设置环境变量 TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1
            out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__}".split(),
                env={**os.environ.copy(), "TORCHINDUCTOR_DUMP_LAUNCH_PARAMS": "1"},
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            # 如果运行失败，输出错误信息和异常
            print(
                "Failed when running triton code with TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1",
                e,
            )
            # 打印出错时的输出内容
            print(e.output.decode())
            # 抛出异常
            raise e
        # 导入函数 get_clean_triton 并使用它来清理编译后的模块
        from torch.utils._get_clean_triton import get_clean_triton
        # 获取清理后的 triton 代码
        cleaned_triton = get_clean_triton(
            compiled_module.__file__, f"{compiled_module.__file__}.cleaned"
        )
        # 确保清理后的 triton 代码中不包含 "@triton_heuristics"
        self.assertTrue("@triton_heuristics" not in cleaned_triton)
        try:
            # 再次运行清理后的编译模块
            out = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__}.cleaned".split(),
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            # 如果再次运行失败，输出错误信息和异常
            print("Failed when running cleaned triton", e)
            print(e.output.decode())
            # 打印清理后的 triton 代码内容
            print(cleaned_triton)
            # 抛出异常
            raise e
        # 返回清理后的 triton 代码
        return cleaned_triton

    def check_bandwidth(self, compiled_module, num_gb):
        # 在子进程中运行编译模块，并检查其输出
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -k".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # 确保输出中包含带宽信息
        FileCheck().check_count(
            f"{num_gb} GB ",
            1,
            exactly=1,
        ).run(bench_out)

    def test_pw_kernel_benchmark(self):
        @torch.compile
        def f(x):
            return torch.sin(x) + torch.cos(x)

        inp = torch.rand(2, 3).to(device=GPU_TYPE)
        out = f(inp)
        self.verify_compiled_kernels()

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @fresh_inductor_cache()
    def test_matmul_triton_kernel_benchmark(self):
        # 定义矩阵乘法的尺寸
        M = 12544
        N = 256
        K = 64
        # 创建随机张量 a 和 b，数据类型为 float16，在 GPU 上进行计算
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, K, dtype=torch.float16, device=GPU_TYPE).t()

        @torch.compile
        def f(a, b):
            return torch.relu(a @ b)

        # 调用编译后的函数 f
        f(a, b)
        self.verify_compiled_kernels()

    @expectedFailureXPU
    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @fresh_inductor_cache()
    # 定义一个测试函数，用于测试 Triton 内核的性能基准
    def test_mm_triton_kernel_benchmark(self):
        # 定义矩阵维度参数
        M = 2048
        N = 2432
        K = 1949
        K_2 = 3581
        # 使用 rand_strided 生成随机数据，设备为 CUDA，数据类型为 torch.float16
        a = rand_strided((M, K_2), (K_2, 1), device="cuda", dtype=torch.float16)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float16)

        @torch.compile
        # 定义一个编译函数 f，用于矩阵乘法运算
        def f(a, b):
            # 使用 torch.narrow 对矩阵 a 进行切片操作
            a_1 = torch.narrow(a, 1, 0, K)
            # 计算矩阵乘法 c = a_1 * b
            c = torch.mm(a_1, b)
            return c

        # 执行编译函数 f，计算结果并验证生成的内核
        f(a, b)
        self.verify_compiled_kernels(GB_count=3)

        # 确保正确生成网格信息
        compiled_module = self.get_compiled_module()
        # 打开编译后的模块文件，读取其源代码
        with open(compiled_module.__file__) as f:
            source_code = f.read()
        lines = source_code.split("\n")
        # 查找包含 "meta0 = {" 的行并执行，获取作用域内的变量信息
        meta = [l for l in lines if "meta0 = {" in l]
        scope = {}
        from torch._inductor.kernel.mm_common import mm_grid

        exec(meta[0], scope)
        # 根据元数据 meta0 和矩阵维度 M, N 计算网格信息
        grid = mm_grid(M, N, scope["meta0"])
        # 运行 FileCheck 检查生成的源代码中网格信息的正确性
        FileCheck().check_count(
            f"grid={grid}",
            2,
            exactly=1,
        ).run(source_code)

    # 定义一个测试函数，用于矩阵乘法带宽计算
    def test_matmul_bandwidth_computation(self):
        """
        The test does a matmul and then mul. Without max-autotune, we use
        the matmul in aten. So there is a single triton kernel for mul.
        The kernel we generated is like:

            @triton.jit
            def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):

        Note the in_out_ptr0 argument. It's for a 1000x1000 tensor, but it's
        inplace udpated, so when computing the bandwidth, we should count
        the total memory access as 2 * 1000 * 1000 * 4 = 8MB. This amount is
        what this test asserts.
        """
        # 设置矩阵乘法的浮点精度为高精度模式
        torch.set_float32_matmul_precision("high")  # suggested by a warning

        @torch.compile
        # 定义一个编译函数 f，进行矩阵乘法和乘法运算
        def f(x, y):
            z = x @ y
            w = z * z
            return w

        # 定义矩阵维度参数
        M, N, K = 1000, 1000, 10
        # 生成随机数据矩阵 x 和 y，设备为 GPU_TYPE
        x = torch.rand(M, K).to(device=GPU_TYPE)
        y = torch.rand(K, N).to(device=GPU_TYPE)
        # 执行编译函数 f，计算结果 out
        out = f(x, y)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 检查计算带宽是否符合预期值 0.008
        self.check_bandwidth(compiled_module, 0.008)

    # 定义一个测试函数，用于未使用输入的带宽计算
    def test_unused_input_bandwidth_computation(self):
        M, N = 5, 1000000

        @torch.compile
        # 定义一个编译函数 f，返回参数 a 和 c 的加和
        def f(a, b, c):
            return a + c

        # 生成随机数据矩阵 a, b, c，数据类型为 torch.float16，设备为 GPU_TYPE
        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        # 将输入参数 a, b, c 标记为动态参数
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 0)
        inputs = (a, b, c)
        # 执行编译函数 f，计算结果 out
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 计算带宽，num_gb = (5 * 1000000 + 5 * 1000000 + 5 * 1000000) * 2 / 1e9 = 0.030
        self.check_bandwidth(compiled_module, "0.030")
    def test_reduction_bandwidth_computation(self):
        @torch.compile
        def f(a):
            # 对输入张量 a 按照 dim=1 进行求和
            return torch.sum(a, dim=1)

        # 创建一个形状为 (1000, 20, 1000) 的随机浮点张量 a，存储在 GPU 上
        a = torch.rand(1000, 20, 1000, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a,)
        # 调用编译后的函数 f，并记录输出
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 计算数据传输量的估算值，并进行带宽检查，预期传输量为 "0.042" GB
        self.check_bandwidth(compiled_module, "0.042")

    @config.patch(max_autotune=True)
    def test_fused_layernorm_bandwidth_computation(self):
        M, N = 10, 1000000

        @torch.compile
        def f(a, b, c, d):
            # 计算 x0 = a + b
            x0 = a + b
            # 对 x0 进行 LayerNorm 处理，使用给定的权重 c 和偏置 d
            x1 = torch.nn.functional.layer_norm(
                x0, normalized_shape=(N,), weight=c, bias=d, eps=1e-05
            )
            # 计算 x2 = sigmoid(x1)
            x2 = torch.sigmoid(x1)
            # 返回 x0 * x2 的结果
            return x0 * x2

        # 创建多个随机张量 a, b, c, d，并存储在 GPU 上
        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        d = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c, d)
        # 调用编译后的函数 f，并记录输出
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 计算数据传输量的估算值，并进行带宽检查，预期传输量为 "0.046" GB
        self.check_bandwidth(compiled_module, "0.046")

    def test_slice_add_cat_bandwidth_computation(self):
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            # 从张量 b 中窄化获取特定范围的子张量 x0
            x0 = torch.narrow(b, 1, N, N)
            # 使用广播机制将 x0 和 c 相加
            x1 = x0 + c
            # 在维度 1 上将张量 a 和 x1 进行连接
            return torch.cat([a, x1], dim=1)

        # 创建多个随机张量 a, b, c，并标记为动态张量
        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        inputs = (a, b, c)
        # 调用编译后的函数 f，并记录输出
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 计算数据传输量的估算值，并进行带宽检查，预期传输量为 "0.052" GB
        self.check_bandwidth(compiled_module, "0.052")
    def test_slice_add_bandwidth_computation(self):
        # 设置矩阵维度 M = 5, N = 1000000
        M, N = 5, 1000000

        # 定义编译函数 f，接受三个参数 a, b, c
        @torch.compile
        def f(a, b, c):
            # 从张量 b 中窄切片，沿着第二维度，起始索引 N，长度 N
            x0 = torch.narrow(b, 1, N, N)
            # 返回 a + x0 + c
            return a + x0 + c

        # 生成随机张量 a，形状为 (M, N)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        # 生成随机张量 b，形状为 (M, 5 * N)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        # 生成随机张量 c，形状为 (N,)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        # 标记张量 a 和 b 为动态张量，索引为 0
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        # 将输入张量打包为元组 inputs
        inputs = (a, b, c)
        # 调用函数 f，传入 inputs，计算结果 out
        out = f(*inputs)

        # 获取已编译模块
        compiled_module = self.get_compiled_module()
        # 计算带宽 num_gb，应为 size_a + size_slice_b + size_c + out_size
        # num_gb = (5 * 1000000 + 5 * 1000000 + 1000000 + 5 * 1000000) * 2 / 1e9
        #        = 0.032
        self.check_bandwidth(compiled_module, "0.032")

    def test_mm_slice_add_bandwidth_computation(self):
        # 设置矩阵维度 M = 1000, N = 1000, K = 30
        M, N, K = 1000, 1000, 30

        # 定义编译函数 f，接受三个参数 a, b, c
        @torch.compile
        def f(a, b, c):
            # 计算矩阵乘积 a * b，结果存储在 x0
            x0 = torch.mm(a, b)
            # 从张量 c 中窄切片，沿着第二维度，起始索引 20 * N，长度 N，结果存储在 x1
            x1 = torch.narrow(c, 1, 20 * N, N)
            # 从张量 c 中窄切片，沿着第二维度，起始索引 21 * N，长度 N，结果存储在 x2
            x2 = torch.narrow(c, 1, 21 * N, N)
            # 返回 x0 + x1 + x2
            return x0 + x1 + x2

        # 生成随机张量 a，形状为 (M, K)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        # 生成随机张量 b，形状为 (K, N)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        # 生成随机张量 c，形状为 (N, 100 * N)，数据类型为 torch.float16，存储在 GPU_TYPE 设备上
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        # 将输入张量打包为元组 inputs
        inputs = (a, b, c)
        # 调用函数 f，传入 inputs，计算结果 out
        out = f(*inputs)

        # 获取已编译模块
        compiled_module = self.get_compiled_module()
        
        # 计算带宽 num_gb，应为 x0 + 2 * size_slice_c + size_out
        # num_gb = (1000 * 1000 + 2 * 1000 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.008
        num_gb = "0.008"
        
        # 如果 GPU_TYPE 是 "xpu"
        if GPU_TYPE == "xpu":
            # 在 XPU 后端，mm + add + add 会被融合成 admm + add
            # CUDA 倾向于不融合 add + mm，请检查 torch/_inductor/fx_passes/post_grad.py 中的 should_prefer_unfused_addmm 函数
            num_gb = "0.006"

        # 检查编译模块的带宽是否与 num_gb 匹配
        self.check_bandwidth(compiled_module, num_gb)
    def test_mm_slice_add_bandwidth_computation_2(self):
        # 定义测试函数，计算矩阵乘法和切片加法带宽
        M, N, K = 1000, 1000, 30

        @torch.compile
        def f(a, b, c):
            # 计算矩阵乘法
            x0 = torch.mm(a, b)
            # 对张量 c 进行窄化操作，选择起始位置为 20*N，长度为 N
            x1 = torch.narrow(c, 1, 20 * N, N)
            # 对张量 c 再次进行窄化操作，选择起始位置为 20*N，长度为 N
            x2 = torch.narrow(c, 1, 20 * N, N)
            # 返回矩阵乘法结果和两个窄化操作结果的加和
            return x0 + x1 + x2

        # 随机生成张量 a, b, c，并指定数据类型和设备
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c)
        # 调用函数 f
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 对于点积加法内核，测量字节数
        # 计算所需的全局内存带宽
        # num_gb = x0 + size_slice_c + size_out
        # num_gb = (1000 * 1000 + 1000 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.006
        # 注意，这里只计算一个 size_slice_c，因为两个访问具有相同的索引。
        self.check_bandwidth(compiled_module, "0.006")

    @expectedFailureXPU
    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_slice_mm_bandwidth_computation(self):
        # 定义测试函数，计算切片矩阵乘法带宽
        M, N, K = 1000, 2000, 3000

        @torch.compile
        def f(a, b):
            # 对张量 a 进行窄化操作，选择起始位置为 K，长度为 K
            x = torch.narrow(a, 1, K, K)
            # 返回窄化操作结果与张量 b 的矩阵乘法结果
            return torch.mm(x, b)

        # 随机生成张量 a, b，并指定数据类型和设备
        a = torch.rand(M, 3 * K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        inputs = (a, b)
        # 调用函数 f
        out = f(*inputs)

        # 获取编译后的模块
        compiled_module = self.get_compiled_module()

        # c[1000, 2000] = x[1000, 3000] @ b[3000, 2000]
        # num_gb = (1000 * 2000 + 1000 * 3000 + 3000 * 2000) * 2 / 1e9
        #        = 0.022
        self.check_bandwidth(compiled_module, "0.022")

    def test_star_dep(self):
        """
        Test the bandwidth estimation for StarDep
        """
        # 测试 StarDep 的带宽估算

        @torch.compile
        def f(a, b):
            # 将张量 a 中 b 指定位置的元素设置为 3.0
            a[b] = 3.0

        # 随机生成张量 a, b，并指定数据类型和设备
        a = torch.rand(10000, 5000, device=GPU_TYPE)
        b = torch.randint(
            0, 10000, [20000], device=GPU_TYPE, dtype=torch.int32
        ).unsqueeze(1)
        f(a, b)
        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 20000 * 4 = 80KB for b
        # 20000 * 5000 * 4 = 200MB for a
        self.check_bandwidth(compiled_module, "0.200")

    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    def test_remove_inductor_deps(self):
        # 测试去除电感器依赖

        @torch.compile
        def f(a):
            # 返回张量 a 的余弦值的正弦值
            return a.cos().sin()

        # 随机生成张量 a，并指定数据类型和设备
        a = torch.randn(5, device=GPU_TYPE)
        f(a)
        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 验证去除电感器依赖
        cleaned_triton = self.verify_remove_inductor_deps()

    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    # 定义一个测试方法，用于测试移除电感依赖关系的多个内核
    def test_remove_inductor_deps_multiple_kernels(self):
        # 声明一个 torch 的编译装饰器函数 f，输入参数为 a
        @torch.compile
        def f(a):
            # 计算矩阵 a 与自身的乘积
            a = torch.mm(a, a)
            # 对结果应用余弦函数，再应用正弦函数
            a = a.cos().sin()
            # 再次计算矩阵 a 与自身的乘积
            a = torch.mm(a, a)
            # 对结果应用 softmax 函数，在指定维度上进行归一化
            a = torch.softmax(a, dim=-1)
            # 返回处理后的数据
            return a

        # 生成一个随机张量 a，形状为 5x5，在 GPU 上进行操作
        a = torch.randn(5, 5, device=GPU_TYPE)
        # 调用编译后的函数 f，处理张量 a
        f(a)
        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 验证编译模块是否移除了电感依赖关系
        self.verify_remove_inductor_deps(compiled_module)

    # 以下为一系列使用 config.patch 进行配置的测试方法
    @config.patch("triton.unique_kernel_names", True)
    @config.patch("triton.unique_kernel_names", True)
    @config.patch(benchmark_kernel=False)
    @config.patch(compile_threads=1)
    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    # 定义一个测试方法，用于测试移除电感依赖的模板
    def test_remove_inductor_deps_templates(self):
        # 声明一个 torch 的编译装饰器函数 f，输入参数为 a
        @torch.compile
        def f(a):
            # 计算矩阵 a 与自身的乘积
            a = torch.mm(a, a)
            # 对结果应用余弦函数
            a = a.cos()
            # 再次计算矩阵 a 与自身的乘积
            a = torch.mm(a, a)
            # 对结果应用正弦函数
            a = a.sin()
            # 返回处理后的数据
            return a

        # 生成一个随机张量 a，形状为 128x128，在 GPU 上进行操作
        a = torch.randn(128, 128, device=GPU_TYPE)
        # 调用编译后的函数 f，处理张量 a
        f(a)
        # 获取编译后的模块
        compiled_module = self.get_compiled_module()
        # 验证编译模块是否移除了电感依赖关系
        self.verify_remove_inductor_deps(compiled_module)
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 如果系统中有 GPU 可用（这里的 HAS_GPU 应该是一个预先定义好的变量或常量）
    if HAS_GPU:
        # 运行测试函数（这里假设 run_tests() 是一个定义好的函数用于执行测试）
        run_tests()
```