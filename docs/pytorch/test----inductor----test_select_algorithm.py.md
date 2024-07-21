# `.\pytorch\test\inductor\test_select_algorithm.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的模块和函数
import functools
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
import torch.nn.functional as F
from torch._dynamo.testing import expectedFailureDynamicWrapper
from torch._dynamo.utils import counters
from torch._inductor.autotune_process import TritonBenchmarkRequest
from torch._inductor.test_case import run_tests, TestCase

from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA

# 定义全局变量 aten
aten = torch.ops.aten

# 定义装饰器函数 patches，用于为测试函数应用各种修补和设置
def patches(fn):
    # 定义跳过缓存的函数，用于在自动调优过程中跳过缓存
    def skip_cache(self, choices, name, key, benchmark):
        if benchmark is None:
            return {}
        return benchmark(choices)

    # 对函数 fn 应用一系列修补和设置
    for patcher in [
        dynamo_config.patch(verbose=True),  # 动态配置模块的详细模式修补
        inductor_config.patch(debug=True, max_autotune=True, epilogue_fusion=True),  # 电感器配置模块的调试模式和自动调优设置修补
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),  # 设置算法选择模块的 VERIFY 对象的公差
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),  # 设置算法选择缓存的查找方法，跳过缓存
        torch.backends.cudnn.flags(allow_tf32=False),  # 设置 cuDNN 后端不允许使用 tf32 模式
    ]:
        fn = patcher(fn)

    # 定义装饰后的函数 wrapped，用于包裹原始测试函数 fn
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()  # 清除计数器
        torch.manual_seed(12345)  # 设置随机种子
        assert (
            not torch.backends.cuda.matmul.allow_tf32
        ), "correctness testing is allergic to tf32"  # 断言，确保在正确性测试中不使用 tf32 模式
        return fn(*args, **kwargs)  # 调用原始测试函数并返回其结果

    return wrapped  # 返回装饰后的测试函数

# 定义测试类 TestSelectAlgorithm，继承自 TestCase
class TestSelectAlgorithm(TestCase):
    # 测试函数 test_linear_relu_cuda，预期会失败的测试
    @expectedFailureDynamicWrapper
    @patches  # 应用装饰器 patches
    def test_linear_relu_cuda(self):
        # 定义内部函数 foo，使用 @torch.compile 注释编译成 Torch 脚本
        @torch.compile
        def foo(input, weight, bias):
            return F.relu(F.linear(input, weight, bias))

        # 调用 foo 函数，传入 CUDA 设备上的随机张量作为参数
        foo(
            torch.randn(64, 32, device="cuda"),
            torch.randn(16, 32, device="cuda"),
            torch.randn(1, 16, device="cuda"),
        )
        # 断言，检查算法选择自动调优计数是否为 1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        # 注释：如果选择了 Triton 模板而不是 aten，这些操作将融合成单个内核。

    # 测试函数 test_addmm_cuda，预期会失败的测试
    @expectedFailureDynamicWrapper
    @patches  # 应用装饰器 patches
    def test_addmm_cuda(self):
        # 定义内部函数 foo，使用 @torch.compile 注释编译成 Torch 脚本
        @torch.compile
        def foo(input, weight, bias):
            return torch.addmm(bias, input, weight)

        # 定义输入参数 inps，包含 CUDA 设备上的随机张量
        inps = (
            torch.randn(20, 33, device="cuda"),
            torch.randn(33, 16, device="cuda"),
            torch.randn(20, 16, device="cuda"),
        )

        # 调用 foo 函数，传入 inps 作为参数
        foo(*inps)
        # 断言，检查算法选择自动调优计数是否为 1
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # 进行额外的修补设置，修改算法选择模块的 VERIFY 对象的公差
    @patch.object(select_algorithm, "VERIFY", dict(atol=5e-2, rtol=5e-2))
    @patches  # 应用装饰器 patches
    def test_addmm_fp16(self):
        @torch.compile
        def foo(input, weight, bias):
            return torch.addmm(bias, input, weight)

        inps = (
            torch.randn(2, 320, device="cuda", dtype=torch.half),  # 创建一个大小为 (2, 320) 的随机张量，使用半精度，在 CUDA 设备上
            torch.randn(320, 320, device="cuda", dtype=torch.half).t(),  # 创建一个大小为 (320, 320) 的随机张量并转置，使用半精度，在 CUDA 设备上
            torch.empty(320, device="cuda", dtype=torch.half),  # 创建一个空的大小为 (320,) 的张量，使用半精度，在 CUDA 设备上
        )

        foo(*inps)  # 调用 foo 函数，传入 inps 元组中的参数
        # 自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device="cuda"),  # 创建一个大小为 (8, 32) 的随机张量，在 CUDA 设备上
            torch.randn(32, 8, device="cuda"),  # 创建一个大小为 (32, 8) 的随机张量，在 CUDA 设备上
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # FIXME: Investigate why _int_mm_out_cuda is not compiled on ROCm
    @skipIfRocm  # 如果运行环境是 ROCm，则跳过这个测试
    @patches
    def test__int_mm(self):
        @torch.compile
        def foo(a, b):
            return torch._int_mm(a, b)

        foo(
            torch.randint(-10, 10, (64, 32), device="cuda", dtype=torch.int8),  # 创建一个大小为 (64, 32) 的随机整数张量，使用 int8 类型，在 CUDA 设备上
            torch.randint(-10, 10, (32, 64), device="cuda", dtype=torch.int8),  # 创建一个大小为 (32, 64) 的随机整数张量，使用 int8 类型，在 CUDA 设备上
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_skip(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(8, 32, device="cuda", dtype=torch.float64),  # 创建一个大小为 (8, 32) 的随机张量，使用双精度浮点数类型，在 CUDA 设备上
            torch.randn(32, 8, device="cuda", dtype=torch.float64),  # 创建一个大小为 (32, 8) 的随机张量，使用双精度浮点数类型，在 CUDA 设备上
        )
        # float64 类型不支持 tl.dot()
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @patches
    def test_bmm(self):
        @torch.compile
        def foo(a, b):
            return torch.bmm(a, b)

        foo(
            torch.randn(2, 8, 32, device="cuda"),  # 创建一个大小为 (2, 8, 32) 的随机张量，在 CUDA 设备上
            torch.randn(2, 32, 8, device="cuda"),  # 创建一个大小为 (2, 32, 8) 的随机张量，在 CUDA 设备上
        )
        # 自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_mm_not_even_k(self):
        @torch.compile
        def foo(a, b):
            return torch.mm(a, b)

        foo(
            torch.randn(11, 22, device="cuda"),  # 创建一个大小为 (11, 22) 的随机张量，在 CUDA 设备上
            torch.randn(22, 33, device="cuda"),  # 创建一个大小为 (22, 33) 的随机张量，在 CUDA 设备上
        )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    def test_baddbmm(self):
        @torch.compile
        def foo(a, b, c):
            return torch.baddbmm(c, a, b)

        foo(
            torch.randn(2, 8, 32, device="cuda"),  # 创建一个大小为 (2, 8, 32) 的随机张量，在 CUDA 设备上
            torch.randn(2, 32, 8, device="cuda"),  # 创建一个大小为 (2, 32, 8) 的随机张量，在 CUDA 设备上
            torch.randn(2, 1, 8, device="cuda"),   # 创建一个大小为 (2, 1, 8) 的随机张量，在 CUDA 设备上
        )
        # 自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    @patches
    # 定义一个测试方法，测试矩阵乘法的编译和执行
    def test_mm_plus_mm(self):
        @torch.compile
        # 定义一个编译函数 foo，计算两个矩阵的乘积之和
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        # 调用 foo 函数，传入四个随机初始化的矩阵，使用 CUDA 设备
        foo(
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
            torch.randn(32, 32, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    # 定义第二个测试方法，测试更大尺寸矩阵乘法的编译和执行（使用 CUDA）
    def test_mm_plus_mm2_cuda(self):
        @torch.compile
        # 定义一个编译函数 foo，计算两个矩阵的乘积之和
        def foo(a, b, c, d):
            return (a @ b) + (c @ d)

        # 调用 foo 函数，传入四个更大尺寸的随机初始化矩阵，使用 CUDA 设备
        foo(
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    # 定义第三个测试方法，测试重复参数的矩阵乘法的编译和执行（使用 CUDA）
    def test_mm_dup_args(self):
        @torch.compile
        # 定义一个编译函数 foo，计算矩阵与自身的乘积
        def foo(a):
            return torch.mm(a, a)

        # 调用 foo 函数，传入一个随机初始化的矩阵，使用 CUDA 设备
        foo(torch.randn(32, 32, device="cuda"))
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @patches
    # 定义第四个测试方法，测试视图操作下的矩阵乘法的编译和执行（使用 CUDA）
    def test_mm_dup_args_view(self):
        @torch.compile
        # 定义一个编译函数 foo，计算分片后的矩阵乘积
        def foo(a):
            q = a[:32, :]
            k = a[32:, :]
            return torch.mm(q, k.transpose(0, 1))

        # 调用 foo 函数，传入一个随机初始化的大尺寸矩阵，使用 CUDA 设备
        foo(torch.randn(64, 64, device="cuda"))
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @expectedFailureDynamicWrapper
    @patches
    # 定义第五个测试方法，测试卷积操作的编译和执行
    def test_convolution1(self):
        @torch.compile
        # 定义一个编译函数 foo，执行卷积操作
        def foo(x, w, b):
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(2, 3),
                padding=(4, 5),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        # 调用 foo 函数，传入不同尺寸的随机初始化张量，使用 CUDA 设备
        foo(
            torch.randn(2, 33, 34, 41, device="cuda"),
            torch.randn(34, 33, 3, 3, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @skipIfRocm
    @patches
    # 定义第六个测试方法，测试带有 dropout 的矩阵乘法的编译和执行
    def test_mm_dropout(self):
        @torch.compile
        # 定义一个编译函数 fn，执行矩阵乘法并应用 dropout
        def fn(x1, x2, seed):
            mm_4 = torch.ops.aten.mm.default(x2, x1)
            rnd = torch.ops.prims.inductor_random.default(mm_4.shape, seed, "rand")
            return mm_4 * rnd

        # 调用 fn 函数，传入不同尺寸的随机初始化张量和种子值，使用 CUDA 设备
        fn(
            torch.randn(512, 1024, dtype=torch.float16, device="cuda"),
            torch.randn(384, 512, dtype=torch.float16, device="cuda"),
            torch.tensor(12345, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
    # 使用 torch._inductor.config.patch 方法来修补 conv_1x1_as_mm 参数为 False 的配置
    @torch._inductor.config.patch(conv_1x1_as_mm=False)
    # 定义测试函数 test_convolution2，用于测试卷积操作
    def test_convolution2(self):
        # 使用 torch.compile 装饰器来编译函数 foo
        @torch.compile
        # 定义 foo 函数，实现卷积操作
        def foo(x, w, b):
            # 调用 aten.convolution 函数执行卷积运算，使用默认参数设置
            return aten.convolution(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        # 调用 foo 函数，传入 CUDA 设备上的随机张量进行卷积计算
        foo(
            torch.randn(1, 33, 16, 16, device="cuda"),
            torch.randn(34, 33, 1, 1, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # 使用 patches 和 torch._inductor.config.patch 方法来修补 conv_1x1_as_mm 参数为 True 的配置
    @patches
    @torch._inductor.config.patch(conv_1x1_as_mm=True)
    # 定义测试函数 test_convolution_as_mm，用于测试作为矩阵乘法的卷积操作
    def test_convolution_as_mm(self):
        # 使用 torch.compile 装饰器来编译函数 foo
        @torch.compile
        # 定义 foo 函数，实现卷积操作，输入 x 加 1
        def foo(x, w, b):
            # 调用 aten.convolution 函数执行卷积运算，使用默认参数设置
            return aten.convolution(
                x + 1,
                w,
                b,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
            )

        # 调用 foo 函数，传入 CUDA 设备上的随机张量进行卷积计算
        foo(
            torch.randn(2, 33, 16, 16, device="cuda"),
            torch.randn(34, 33, 1, 1, device="cuda"),
            torch.randn(34, device="cuda"),
        )
        # 断言自动调优检查每个版本的正确性
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    # 定义测试函数 test_TritonTemplateCaller_str，确保 str(TritonTemplateCaller) 不会引发异常
    def test_TritonTemplateCaller_str(self):
        # 设置模块路径为 "abc.py"
        module_path = "abc.py"
        # 创建 TritonBenchmarkRequest 对象 bmreq，传入各种参数
        bmreq = TritonBenchmarkRequest(
            module_path=module_path,
            module_cache_key=None,
            kernel_name=None,
            grid=None,
            extra_args=None,
            num_stages=None,
            num_warps=None,
            input_tensor_meta=None,
            output_tensor_meta=None,
        )
        # 创建 TritonTemplateCaller 对象 caller，传入参数和 bmreq 对象
        caller = select_algorithm.TritonTemplateCaller(
            None, None, None, None, "extra", bmreq
        )
        # 获取 caller 的字符串表示形式
        caller_str = str(caller)
        # 断言 caller_str 是否符合预期的格式
        self.assertEqual(caller_str, f"TritonTemplateCaller({module_path}, extra)")
#`
# 当脚本直接运行时执行以下代码块
if __name__ == "__main__":
    # 从 torch._inductor.utils 模块导入 is_big_gpu 函数
    from torch._inductor.utils import is_big_gpu

    # 检查当前操作系统是否为 Linux，是否支持 CUDA，并且是否是大 GPU
    if IS_LINUX and HAS_CUDA and is_big_gpu(0):
        # 调用 run_tests 函数执行测试
        run_tests()
```