# `.\pytorch\test\inductor\test_multi_kernel.py`

```py
# Owner(s): ["module: inductor"]

# 引入必要的模块和库
import os  # 系统操作相关模块
import re  # 正则表达式模块
import unittest  # 单元测试框架

import torch  # PyTorch 主模块
from torch import nn  # 神经网络模块
from torch._dynamo.testing import reset_rng_state  # 重置随机数生成器状态

# 引入自定义模块和函数
from torch._inductor import config, test_operators  # 引入配置和测试操作
from torch._inductor.codegen.multi_kernel import MultiKernelCall  # 多内核调用
from torch._inductor.test_case import TestCase  # 测试用例基类
from torch._inductor.utils import run_and_get_code  # 运行并获取代码
from torch.nn import functional as F  # 神经网络函数模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 实例化参数化测试
    parametrize,  # 参数化装饰器
)
from torch.testing._internal.inductor_utils import HAS_CUDA  # CUDA 支持判断

# 定义一个示例的神经网络模块
class TransformerSnippet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(64)  # 第一层归一化
        self.ln2 = nn.LayerNorm(64)  # 第二层归一化

    def forward(self, x1, x2):
        x1 = F.dropout(x1, 0.1)  # 对输入 x1 进行 Dropout 操作
        x2 = F.dropout(self.ln1(x2), 0.1)  # 对输入 x2 先进行归一化再 Dropout

        return self.ln2(x1 + x2)  # 返回归一化后的 x1 和 x2 的和再次归一化结果

    def example_inputs(self):
        return (torch.randn(2, 64).cuda(), torch.randn(2, 64).cuda())  # 返回示例输入数据


# 检查给定的 wrapper_code 字符串是否包含多内核代码的函数
def _contains_multi_kernel_code(wrapper_code: str):
    return (
        re.search(r"multi_kernel_[^ ]* = async_compile.multi_kernel[(]", wrapper_code)
        is not None
    )


# 将原始测试 orig_test 包装成启用了 cpp-wrapper 的新测试函数
def make_cpp_wrapper_test(orig_test, **extra_args):
    """
    Wrap an existing test into a new test with cpp-wrapper enabled.

    Make this as a free function rather than staticmethod in MultiKernelTest.
    Otherwise we get 'TypeError: 'staticmethod' object is not callable'
    error in py3.8. (py3.10 works)
    """

    @config.patch("cpp_wrapper", True)
    def fn(self):
        # 可能之前的测试已经用禁用 cpp_wrapper 编译了同一个内核，清除缓存以便重新启用 cpp_wrapper 编译
        from torch._inductor import codecache

        codecache.PyCodeCache.cache_clear()
        return orig_test(self, **extra_args)

    return fn


# 配置多内核测试的参数，并实例化参数化测试
@config.patch(
    {
        "triton.multi_kernel": int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "1")),
        "benchmark_kernel": True,
    }
)
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
    # 定义一个测试函数，用于测试 softmax 函数在多核环境下的行为
    def test_softmax(self, expect_multi_kernel=True):
        # 生成一个随机张量，存储在 GPU 上
        x = torch.rand(2, 1024).cuda()
        # 计算参考的 softmax 结果
        ref = torch.softmax(x, -1)
        # 编译 softmax 函数
        compiled_fn = torch.compile(torch.softmax)
        # 运行并获取编译后的代码以及计算结果
        act, wrapper_code = run_and_get_code(compiled_fn, x, -1)

        # 如果 cpp_wrapper=True，则 wrapper_code 将包含两个条目。
        # 分别是第一次和第二次传递的包装器代码。
        # 在这里，我们主要关心最终传递的包装器代码。
        wrapper_code = wrapper_code[-1]
        # 断言参考结果和实际结果的接近程度
        self.assertTrue(torch.allclose(ref, act))
        # 如果期望多核代码，则断言包装器代码包含多核代码
        if expect_multi_kernel:
            self.assertTrue(_contains_multi_kernel_code(wrapper_code))
        else:
            # 在 fbcode 环境中跳过验证 wrapper_code，因为缺少适当的 CUDA 编译器设置
            # 如果不是在 fbcode 环境下，则断言 wrapper_code 不包含多核代码
            if not config.is_fbcode():
                self.assertFalse(_contains_multi_kernel_code(wrapper_code))

    # 使用参数化装饰器，测试强制使用非持久化缩减的 softmax 函数
    @parametrize("force_kernel", (0, 1))
    @unittest.mock.patch.dict(
        os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
    )
    def test_softmax_force_non_persistent_reduction(self, force_kernel):
        """
        Force a specific sub-kernel being picked by mocking the benchmark result.
        """
        # 生成一个随机张量，存储在 GPU 上
        x = torch.rand(2, 1024).cuda()
        # 模拟的延迟值
        mock_latency = [0.2, 0.2]
        # 确保 force_kernel 被选择的模拟
        mock_latency[force_kernel] = 0.1

        # 定义一个函数 f，对输入张量进行 softmax 操作，并加上 force_kernel 的值
        def f(x):
            return torch.softmax(x, -1) + force_kernel

        # 备份原始的 MultiKernelCall.run_with_argless_kernels 函数
        orig_run = MultiKernelCall.run_with_argless_kernels
        picked_kernel = None

        # 定义一个 mock_run 函数，用于替代原始的 run_with_argless_kernels 函数
        def mock_run(self, kernel_calls):
            # 调用原始函数
            out = orig_run(self, kernel_calls)
            nonlocal picked_kernel
            picked_kernel = self.picked_kernel
            return out

        # 使用 unittest.mock.patch.object 来替换 MultiKernelCall 类的方法
        # mock_run 替换 run_with_argless_kernels 方法
        # lambda 函数替换 benchmark_sub_kernels 方法，返回 mock_latency
        with unittest.mock.patch.object(
            MultiKernelCall, "run_with_argless_kernels", mock_run
        ), unittest.mock.patch.object(
            MultiKernelCall, "benchmark_sub_kernels", lambda *args: mock_latency
        ):
            # 编译函数 f
            torch.compile(f)(x)
        # 断言 picked_kernel 等于 force_kernel
        self.assertEqual(picked_kernel, force_kernel)

    # 使用 config.patch 来设置 "warn_mix_layout" 为 True，测试 warn_mix_layout 功能
    @config.patch("warn_mix_layout", True)
    def test_softmax_warn_mixed_layout(self):
        # 调用 test_softmax 函数
        self.test_softmax()

    # 定义一个测试函数，用于测试 cpp_wrapper 的 softmax 函数
    test_softmax_cpp_wrapper = make_cpp_wrapper_test(
        test_softmax, expect_multi_kernel=False
    )

    # 定义一个测试函数，用于测试 layernorm 函数
    def test_layernorm(self):
        # 创建一个包含 1024 个特征的 LayerNorm 层，存储在 GPU 上
        ln = nn.LayerNorm(1024).cuda()
        # 生成一个随机张量，存储在 GPU 上
        x = torch.rand(2, 1024).cuda()
        # 计算参考的 layernorm 结果
        ref = ln(x)
        # 编译 layernorm 函数
        act = torch.compile(ln)(x)
        # 断言参考结果和实际结果的接近程度
        self.assertTrue(
            torch.allclose(ref, act, atol=1e-4, rtol=1e-4), f"ref:\n{ref}\nact:\n{act}"
        )
    def test_inplace_update(self):
        """
        Inductor generate inplace kernel for mul.
        """
        
        # 定义函数 f，计算 x 沿着最后一个维度求和后的结果乘以 y 的内积
        def f(x, y):
            return x.sum(dim=-1, keepdims=True) * (y @ y)
        
        # 在 GPU 上生成随机张量 x 和 y
        x = torch.rand(1024, 1024).cuda()
        y = torch.rand(1024, 1024).cuda()
        
        # 计算参考结果 ref
        ref = f(x, y)
        
        # 使用 torch.compile 编译函数 f，生成优化的计算图
        act = torch.compile(f)(x, y)
        
        # 断言优化后的计算结果与参考结果 ref 接近
        self.assertTrue(torch.allclose(ref, act))
    
    def test_transformer_snippet(self):
        # 创建一个 TransformerSnippet 模型并移动到 GPU 上
        model = TransformerSnippet().cuda()
        
        # 获取模型的示例输入
        x = model.example_inputs()
        
        # 定义函数 f，接收任意数量的参数，对模型进行前向传播并返回结果 y
        def f(*x):
            y = model(*x)
            return y
        
        # 重置随机数生成器的状态
        reset_rng_state()
        
        # 计算参考结果 ref
        ref = f(*x)
        
        # 使用 torch.compile 编译函数 f，生成优化的计算图
        opt_f = torch.compile(f)
        
        # 重置随机数生成器的状态
        reset_rng_state()
        
        # 计算优化后的结果 act
        act = opt_f(*x)
        
        # 如果配置为使用 fallback_random，不比较张量，因为 inductor 随机数实现与 eager 不同
        if config.fallback_random:
            self.assertTrue(
                torch.allclose(ref, act, atol=1e-4, rtol=1e-4),
                f"ref:\n{ref}\nact:\n{act}",
            )
    
    def test_transformer_snippet_with_fallback_random(self):
        """
        Same as test_transformer_snippet but fallback the random number
        generator to eager so we can check accuracy.
        """
        # 使用 config.patch 将 fallback_random 设置为 True，执行 test_transformer_snippet 测试
        with config.patch("fallback_random", True):
            self.test_transformer_snippet()
    
    def test_batchnorm_training(self):
        """
        For training, batchnorm will tracking running mean/variance during forward pass.
        The kernel generated by inductor currently will pass in those tensors twice as arguments:
        once for input and once for output. They are ruled out as in-out argument because
        they are considered as graph inputs.

        Multi-kernel previously assumes that we never pass the same argument mutli times
        for a kernel. No mater if we change inductor behavior to assure that, it's better
        to make multi-kernel being able to handle those cases.
        """
        # 创建一个在 CUDA 设备上的 nn.BatchNorm2d 模块
        bn = nn.BatchNorm2d(3).to("cuda")
        
        # 使用 torch.compile 编译函数 f，对输入 x 的批量归一化结果进行求和并进行反向传播
        @torch.compile
        def f(x):
            bn(x).sum().backward()
        
        # 运行并获取函数 f 的代码及其包装代码
        _, (wrapper_code, _) = run_and_get_code(
            f, torch.randn(2, 3, 8, 8, device="cuda")
        )
        
        # 断言包装代码中包含多核心代码
        self.assertTrue(_contains_multi_kernel_code(wrapper_code))
    # 定义一个测试函数，用于测试相同参数多次传递时的情况，模拟 BatchNorm 更新运行统计数据的方式
    def test_pass_same_arg_multi_times(self):
        """
        A super simple example that simulate how BatchNorm update the running
        stats.

        Inductor currently pass the same tensor multiple times for the generated
        kernel: once for input and once for output.

        Here is a paster for the generated kernel (without multi-kernel enabled):
        https://gist.github.com/shunting314/f0b446b4b9a28f4940e31dcd3e809cf9
        """

        # 定义内部函数 f，接收两个参数 x 和 y
        def f(x, y):
            # 对 x 进行按维度求和操作，保持维度不变
            x = x.sum(dim=1, keepdim=False)
            # 将 y 更新为 y 的 0.9 倍加上 x 的 0.1 倍
            y.copy_(y * 0.9 + x * 0.1)

        # 在 CUDA 设备上生成随机张量 x 和 y
        x = torch.randn(8, 16, device="cuda")
        y = torch.randn(8, device="cuda")
        # 复制 y 以备后续比较使用
        y_ref = y.clone()

        # 调用函数 f，传入 x 和 y_ref，更新 y_ref
        ref = f(x, y_ref)
        # 编译优化后的函数 f，并传入 x 和 y，更新 y
        act = torch.compile(f)(x, y)
        # 断言 y_ref 和 y 在数值上相近
        self.assertTrue(torch.allclose(y_ref, y))

    # 定义一个测试函数，用于测试使用显式实现缓冲区的情况，作为非持久性规约核的临时缓冲区传递，
    # 但对于持久性规约核可以跳过该缓冲区
    def test_reduction_scratch_buffer(self, force_multi_kernel=1):
        """
        The explicited realized buffer in the test function will be passed in
        as a scratch buffer for the non-persistent reduction kernel but
        can be skipped for the persistent reduction kernel.

        This causes different argument lists for non-persistent reduction kernel and
        persistent reduction kernel.

        Check documentation around torch._inductor.config.triton.multi_kernel about
        how to interpret the force_multi_kernel argument.
        """

        # 定义内部函数 f，接收一个参数 x
        def f(x):
            # 按最后一个维度求和，并保持维度
            x = x.sum(dim=-1, keepdim=True) + x
            # 调用 test_operators.realize 对 x 进行实现
            x = test_operators.realize(x)
            # 再次按最后一个维度求和，并保持维度
            x = x.sum(dim=-1, keepdim=True) + x
            return x

        # 在 CUDA 设备上生成随机张量 x
        x = torch.rand(16, 16, device="cuda")
        # 调用函数 f，计算 ref
        ref = f(x)
        # 使用 config.patch 设置 "triton.multi_kernel" 为 force_multi_kernel，编译优化后的函数 f，并传入 x
        with config.patch("triton.multi_kernel", force_multi_kernel):
            act = torch.compile(f)(x)
        # 断言 ref 和 act 在数值上相近
        self.assertTrue(torch.allclose(ref, act))

    # 使用基准测试选择更快的核心
    test_reduction_scratch_buffer_cpp_wrapper = make_cpp_wrapper_test(
        test_reduction_scratch_buffer, force_multi_kernel=1
    )
    # 强制选择持久性规约。这是一个好的测试，因为持久性规约使用的调用参数比相应的非持久性规约少。
    test_reduction_scratch_buffer_cpp_wrapper_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=2)
    )
    # 强制选择非持久性规约
    test_reduction_scratch_buffer_cpp_wrapper_non_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=3)
    )
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块中导入 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果存在 CUDA 支持，则运行测试
    if HAS_CUDA:
        run_tests()
```