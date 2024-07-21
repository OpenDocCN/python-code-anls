# `.\pytorch\test\inductor\test_torchinductor_dynamic_shapes.py`

```
# Owner(s): ["module: inductor"]

# 引入标准库模块
import contextlib
import importlib

# 引入数学和操作符模块
import math
import operator

# 引入操作系统和系统相关模块
import os
import sys
import unittest

# 引入偏函数工具
from functools import partial

# 引入类型提示工具
from typing import List

# 引入 PyTorch 库
import torch
import torch.library

# 引入测试相关模块和工具函数
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import metrics
from torch._inductor.codegen.common import device_codegens, register_backend_for_device
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.test_case import TestCase
from torch._inductor.virtualized import V

# 引入特定的设备类型测试相关模块和工具函数
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyOn,
)

# 引入通用测试工具函数
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_CI,
    IS_WINDOWS,
    parametrize,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)

# 引入测试工具函数
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU

# 如果在 Windows CI 环境下，打印警告并且跳过测试
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 将 test/ 目录下的文件加入到模块搜索路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 从 inductor.test_torchinductor 模块中导入所需函数和类
from inductor.test_torchinductor import (
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
    TestFailure,
)

# 动态导入 filelock 模块
importlib.import_module("filelock")

# 定义测试失败的情况，设定默认为 xfail，设置 is_skip=True 可以跳过
test_failures = {
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    # 调用 div 函数时仅支持 symint 参数
    "test_AllenaiLongformerBase_repro_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu")
    ),
    "test_conv_inference_heuristics_dynamic_shapes": TestFailure(("cuda", "xpu")),
}

# 如果是在 ROCm 环境下
if TEST_WITH_ROCM:
    # 张量类似物不接近
    test_failures["test_dynamic_stride_nobreak"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_item_to_inputs_kernel_nobreak"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_unbacked_reduction"] = TestFailure(("cpu"), is_skip=True)

# 根据给定的类生成动态形状测试类
def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )

# 创建 DynamicShapesCommonTemplate 类
DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)

# 如果有 CPU 设备
if HAS_CPU:

    # 创建 DynamicShapesCpuTests 类，继承于 TestCase
    class DynamicShapesCpuTests(TestCase):
        common = check_model  # 设置公共方法为 check_model
        device = "cpu"         # 设备为 CPU

    # 复制测试用例到 DynamicShapesCpuTests 类中，对 CPU 进行测试，处理 test_failures
    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, "cpu", test_failures)

# 如果有 GPU 设备且不在 ASAN 测试下
if HAS_GPU and not TEST_WITH_ASAN:

    # 创建 DynamicShapesGPUTests 类，继承于 TestCase
    class DynamicShapesGPUTests(TestCase):
        common = check_model_gpu  # 设置公共方法为 check_model_gpu
        device = GPU_TYPE         # 设备为 GPU_TYPE
    # 调用函数 `copy_tests`，将 `DynamicShapesCommonTemplate` 中的测试用例复制到 `DynamicShapesGPUTests` 中，
    # 使用 GPU 类型 `GPU_TYPE`，并将测试失败情况记录在 `test_failures` 中。
    copy_tests(
        DynamicShapesCommonTemplate, DynamicShapesGPUTests, GPU_TYPE, test_failures
    )
class TestInductorDynamic(TestCase):
    # 定义编译函数为动态编译
    compile_fn = partial(torch.compile, dynamic=True)

    def setUp(self):
        # 如果没有 GPU，则跳过测试（使用 Triton）
        if not HAS_GPU:
            self.skipTest("Triton not available")
        # 重置 Torch 的动态编译环境
        torch._dynamo.reset()
        # 调用父类的 setUp 方法
        TestCase.setUp(self)
        # 使用 contextlib.ExitStack 确保资源的安全释放
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            # 设置 Torch 的动态编译配置
            torch._inductor.config.patch(
                {
                    "debug": False,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # 这个太慢了
                    "implicit_fallbacks": False,
                }
            )
        )

    def tearDown(self):
        # 关闭资源栈
        self._stack.close()
        # 调用父类的 tearDown 方法
        TestCase.tearDown(self)
        # 重置 Torch 的动态编译环境
        torch._dynamo.reset()

    def test_arange_dynamic(self, device):
        def fn(a):
            batch_size = a.numel()
            max_len = a.max()
            return ~(
                # 创建一个从 0 到 max_len 的张量，在设备上运行
                torch.arange(0, max_len, device=a.device)
                .type_as(a)
                .repeat(batch_size, 1)
                .lt(a.unsqueeze(1))
            )

        a = torch.randint(10, 30, (10,), device=device)
        a[0] = 29  # 修正 max_len
        opt = self.compile_fn(fn)
        res = opt(a)
        ref = fn(a)
        self.assertEqual(res, ref)

    def test_shape_as_constant_reciprocal_float_exp(self, device):
        def fn(x, a):
            return x, -1 / a**1.0

        x = torch.rand(10, 20, device=device)
        opt = self.compile_fn(fn)
        res = opt(x, x.size(0))
        ref = fn(x, x.size(0))
        self.assertEqual(res, ref)

    # 目前在 CPU 上不支持，https://github.com/pytorch/pytorch/issues/109897
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_bool_mask_nobreak(self, device):
        def f(x, b):
            return (x[b] * 2).sum()

        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    def test_adaptive_max_pool3d_with_indices(self, device):
        x = 5
        y = torch.rand([9, 10, 9, 8, 6], dtype=torch.float32, device=device)

        def fn(x, y):
            return torch.nn.functional.adaptive_max_pool3d_with_indices(
                output_size=x, input=y, return_indices=True
            )

        opt_f = self.compile_fn(fn)
        r = fn(x, y)
        opt_r = opt_f(x, y)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unwrap_storage_didnt_work_repro(self, device):
        # 定义内部函数f，用于测试
        def f():
            # 创建一个全为11的标量张量
            full = torch.full((), 11)
            # 提取标量张量中的数值（此处应为11）
            i0 = full.item()
            # 检查提取的数值是否为合法的张量大小
            torch._check_is_size(i0)
            # 返回一个形状为i0的全零张量
            return torch.full((i0,), 0)

        # 编译函数f以获取优化版本opt_f
        opt_f = torch.compile(f, fullgraph=True)
        # 直接调用函数f并保存结果r
        r = f()
        # 调用优化后的函数opt_f并保存结果opt_r
        opt_r = opt_f()
        # 断言函数f和优化后的函数opt_f的输出结果一致
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory_nobreak(self, device):
        # 定义函数f，接受输入x和布尔向量b
        def f(x, b):
            # 计算布尔向量b中非零元素的索引
            y = torch.nonzero(b)
            # 返回一个形状与y的大小相同的全零张量
            return x.new_zeros(y.size(0))

        # 编译函数f以获取优化版本opt_f
        opt_f = torch.compile(f, fullgraph=True)
        # 创建一个5维随机张量x
        x = torch.randn(5, device=device)
        # 创建一个布尔向量b，包含True和False
        b = torch.tensor([True, True, False, False, True], device=device)
        # 调用函数f并保存结果r
        r = f(x, b)
        # 调用优化后的函数opt_f并保存结果opt_r
        opt_r = opt_f(x, b)
        # 断言函数f和优化后的函数opt_f的输出结果一致
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_no_realloc(self, device):
        # 定义函数f，接受输入x和y
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x, y):
            # 找到张量x中非零元素的索引
            z = x.nonzero()
            # 将张量z按照给定的大小分割
            return torch.split(z, [y.size(0)])

        # 调用函数f，传入一个张量和一个随机张量
        f(torch.tensor([1, 0, 1, 1, 0, 1, 0]), torch.randn(4))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_nobreak(self, device):
        # 定义函数f，接受输入x
        @torch.compile(fullgraph=True)
        def f(x):
            # 提取张量x中的标量值
            y = x.item()
            # 返回一个形状为y的空张量
            return torch.empty(y)

        # 调用函数f，传入一个标量张量
        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_bool_nobreak(self, device):
        # 定义函数f，接受输入x
        @torch.compile(fullgraph=True)
        def f(x):
            # 直接提取布尔张量x中的标量值
            return x.item()

        # 调用函数f，传入一个布尔张量
        f(torch.tensor([True], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_noops_tensor_repropagate(self, device):
        # 定义函数f，接受输入x
        @torch.compile(fullgraph=True)
        def f(x):
            # 将张量x的元素类型转换为int64
            b = torch.ops.prims.convert_element_type.default(x, torch.int64)
            # 找到张量b中非零元素的索引
            r = b.nonzero()
            # 返回张量r每个元素乘以2的结果
            return r * 2

        # 调用函数f，传入一个int64类型的张量
        f(torch.tensor([0, 4, 2, 0, 1], dtype=torch.int64, device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_zeros_nobreak(self, device):
        # 定义函数f，接受输入x
        @torch.compile(fullgraph=True)
        def f(x):
            # 提取张量x中的标量值
            y = x.item()
            # 创建一个形状为y的空张量
            torch.empty(y)
            # 返回一个形状为y的全零张量
            return x.new_zeros(y)

        # 调用函数f，传入一个标量张量
        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_return(self, device):
        # 定义函数f，接受输入x
        @torch.compile(fullgraph=True)
        def f(x):
            # 提取张量x中的标量值
            y = x.item()
            # 再次提取张量x中的标量值，形成y和z
            z = x.item()
            # 返回y和z的和
            return y + z

        # 调用函数f，传入一个标量张量
        f(torch.tensor([3], device=device))
    # 测试函数，用于检查是否浮点数的 item 方法返回正无穷
    def test_float_item_inf(self, device):
        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        @torch.compile(fullgraph=True)
        def f(x):
            # 检查 x 是否等于正无穷
            return x.item() == math.inf

        # 调用函数 f，并传入一个包含单个浮点数 3.0 的张量

    # 装饰器配置，捕获标量输出和捕获动态输出形状操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    # 测试函数，用于检查是否浮点数的 item 方法返回负无穷
    def test_float_item_neginf(self, device):
        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        @torch.compile(fullgraph=True)
        def f(x):
            # 检查 x 是否等于负无穷
            return x.item() == -math.inf

        # 调用函数 f，并传入一个包含单个浮点数 3.0 的张量

    # 装饰器配置，捕获标量输出
    # 还有一个装饰器配置，启用隐式回退
    def test_item_to_inputs_kernel_nobreak(self, device):
        # 自定义 Torch 操作，命名为 "test::foo"，不修改参数
        @torch.library.custom_op("test::foo", mutates_args=())
        def foo(x: torch.Tensor, y: int) -> torch.Tensor:
            # 克隆张量 x 并返回
            return x.clone()

        # 为 foo 函数注册一个假的实现
        @foo.register_fake
        def _(x: torch.Tensor, y: int) -> torch.Tensor:
            # 克隆张量 x 并返回
            return x.clone()

        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        @torch.compile(fullgraph=True)
        def f(x, r):
            # 将张量 x 转换为标量，赋值给 y
            y = x.item()
            # 调用自定义操作 foo，并传入 r 和 y 作为参数
            return torch.ops.test.foo(r, y)

        # 调用函数 f，并传入两个张量作为参数

    # 标记为预期失败的单元测试
    # 装饰器配置，捕获标量输出和捕获动态输出形状操作
    def test_float_item_return(self, device):
        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        @torch.compile(fullgraph=True)
        def f(x):
            # 返回张量 x 的标量值
            return x.item()

        # 调用函数 f，并传入一个包含单个浮点数 3.0 的张量

    # 装饰器配置，如果测试 CUDA 内存泄漏检查失败，则跳过
    # 装饰器配置，捕获标量输出
    def test_unbacked_index_select(self, device):
        # 测试内部函数是否正确跟踪由 inner_fn 捕获的未支持符号
        def f(x):
            # 将张量 x 转换为标量，赋值给 y
            y = x.item()
            # 使用 torch.index_select 从第一个张量中选择索引
            return torch.index_select(
                torch.ones(y, device=device), 0, torch.tensor([0, 2, 1], device=device)
            )

        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        cf = torch.compile(fullgraph=True)(f)
        # 创建一个包含值为 5 的张量作为参数
        arg = torch.tensor(5, device=device)
        # 断言函数 f 的输出与编译后的函数 cf 的输出相等

    # 装饰器配置，捕获标量输出和捕获动态输出形状操作
    def test_return_unbacked_view_split(self, device):
        def f(values, length_per_key):
            # 将 length_per_key 转换为列表 u0 和 u1
            u0, u1 = length_per_key.tolist()
            # 检查 u0 和 u1 是否是有效的张量大小
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            # 使用 torch.functional.split 将 values 按给定长度 u0 和 u1 分割
            v1, v2 = torch.functional.split(values, [u0, u1])
            # 返回分割后的张量 v1 和 v2
            return v1, v2

        # 使用 Torch 的 JIT 编译器编译函数 f，使其在完整图形模式下运行
        cf = torch.compile(fullgraph=True)(f)
        # 创建一个张量和一个张量作为参数
        args = (
            torch.randn(8, requires_grad=True, device=device),
            torch.tensor([3, 5], device=device),
        )
        # 断言函数 f 的输出与编译后的函数 cf 的输出相等

    # 装饰器配置，捕获标量输出
    def test_case2(self, device):
    # 定义测试函数，测试未支持的矩阵乘法操作
    def test_unbacked_matmul(self, device):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 获取 x 的标量值
            y = x.item()
            # 创建一个全为 1 的张量，进行矩阵乘法操作
            return torch.ones(1, y, device=device) @ torch.ones(y, 1, device=device)

        # 使用 Torch 的编译功能，生成编译后的函数 cf
        cf = torch.compile(fullgraph=True)(f)
        # 创建输入张量 arg
        arg = torch.tensor(5, device=device)
        # 断言未编译和编译后的函数在给定参数下的输出相等
        self.assertEqual(f(arg), cf(arg))

    # 通过 Torch 的配置进行装饰，测试未支持的反向保存操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_unbacked_save_for_backwards(self, device) -> None:
        # 定义自定义操作 _test::_cat，接受参数 t 和 ds
        @torch.library.custom_op("_test::_cat", mutates_args=())
        def _cat(t: torch.Tensor, ds: List[int]) -> torch.Tensor:
            # 返回张量 t 乘以全为 1 的张量，其大小为 ds 中所有元素之和
            return t * t.new_ones([sum(ds)])

        # 注册 _test::_cat 的伪造函数，用于测试目的
        @torch.library.register_fake("_test::_cat")
        def _cat_fake(t: torch.Tensor, ds: List[int]) -> torch.Tensor:
            # 检查 ds 中的每个元素是否为合法大小，返回一个全为空的张量
            [torch._check_is_size(d) for d in ds]
            return t.new_empty([sum(ds)])

        # 定义 _cat_setup_context 函数，用于设置操作的上下文环境
        def _cat_setup_context(ctx, inputs, output):
            pass

        # 定义 _cat_backward 函数，用于计算操作的反向传播梯度
        def _cat_backward(ctx, grad):
            return grad.sum(), None

        # 注册 _test::_cat 的自动求导函数
        torch.library.register_autograd(
            "_test::_cat",
            _cat_backward,
            setup_context=_cat_setup_context,
        )

        # 定义 fn 函数，接受参数 t 和 sizes
        def fn(t, sizes):
            # 调用 _test::_cat 自定义操作，进行计算
            r = torch.ops._test._cat(t, sizes.tolist())
            return r * t

        # 创建随机张量 t，需要梯度计算
        t = torch.randn((), requires_grad=True, device=device)
        # 创建 sizes 张量，指定大小为 [4, 8]
        sizes = torch.tensor([4, 8], dtype=torch.int64, device="cpu")
        # 调用 fn 函数，进行计算
        out = fn(t, sizes)
        # 对计算结果进行求和，并进行反向传播
        out.sum().backward()
        # 期望的梯度值为 t.grad
        expect = t.grad
        # 将 t.grad 设为 None，用于后续测试
        t.grad = None
        # 使用 Torch 编译 fn 函数，后端为 "inductor"，生成完整图并动态执行
        torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)(
            t, sizes
        ).sum().backward()
        # 断言编译后的梯度与预期相等
        self.assertEqual(t.grad, expect)

    # 通过 Torch 的配置进行装饰，测试未支持的归约操作
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_reduction(self, device):
        # 检查是否期望失败
        expect_fail = device == "cpu" and not IS_ARM64
        try:
            # 定义内部函数 f，接受参数 x
            def f(x):
                # 获取 x 的标量值
                y = x.item()
                # 创建一个大小为 y 的全为 1 的张量，并求和
                return torch.ones(y, device=device).sum()

            # 使用 Torch 的编译功能，生成编译后的函数 cf
            cf = torch.compile(fullgraph=True)(f)
            # 创建输入张量 arg
            arg = torch.tensor(5, device=device)
            # 断言未编译和编译后的函数在给定参数下的输出相等
            self.assertEqual(f(arg), cf(arg))
        except Exception:
            # 如果不是期望失败，则抛出异常
            if not expect_fail:
                raise
        else:
            # 如果是期望失败，则测试失败
            if expect_fail:
                self.fail("expected to fail, but actually passed")

    # 通过 Torch 的配置进行装饰，测试未支持的重复大小的操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_cat_unbacked_duplicate_size(self, device):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 获取输入张量 x 的设备信息
            device = x.device
            # 将输入张量 x 列表化为 s 和 s2
            s, s2 = x.tolist()
            # 创建全为 0 的大小为 s 的张量 g，创建全为 1 的大小为 s2 的张量 g2
            g = torch.zeros(s, device=device)
            g2 = torch.ones(s2, device=device)
            # 调用 Torch 的 aten.cat 操作，对 g 和 g2 进行连接
            return torch.ops.aten.cat.default([g, g, g2])

        # 使用 Torch 的编译功能，生成编译后的函数 cf
        cf = torch.compile(fullgraph=True)(f)
        # 创建输入张量 arg，大小为 [4, 6]，设备为 GPU_TYPE
        arg = torch.tensor([4, 6], device=GPU_TYPE)
        # 断言未编译和编译后的函数在给定参数下的输出相等
        self.assertEqual(f(arg), cf(arg))
    # 使用 Torch 的配置装饰器，捕获标量输出和动态输出形状操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    # 定义测试方法 test_unbacked_cat_backwards，接受设备参数
    def test_unbacked_cat_backwards(self, device):
        # 定义内部函数 f，接受输入 x 和权重 w
        def f(x, w):
            # 设备赋值为 w 的设备
            device = w.device
            # 将 x 转换为列表 a 和 b
            a, b = x.tolist()
            # 创建长度为 a 的全 1 张量 ta，并指定设备
            ta = torch.ones(a, device=device)
            # 创建长度为 b 的全 1 张量 tb，并指定设备
            tb = torch.ones(b, device=device)
            # pa 和 pb 分别为 ta 和 tb 与 w 的乘积，使其需要梯度
            pa = ta * w
            pb = tb * w
            # 将 pa 和 pb 连接成一个张量 r
            r = torch.cat([pa, pb])
            # 返回张量 r 的元素和
            return r.sum()
    
        # 创建输入张量 x
        x = torch.tensor([4, 9])
        # 创建随机权重张量 w，并指定需要梯度
        w = torch.randn(1, requires_grad=True)
        # 调用函数 f，并对其结果进行反向传播
        f(x, w).backward()
        # 记录原始权重 w 的梯度
        orig_w = w.grad
        # 将权重 w 的梯度置空
        w.grad = None
    
        # 使用 Torch 编译器编译完整图形的函数 f，并对其结果进行反向传播
        torch.compile(fullgraph=True)(f)(x, w).backward()
        # 断言原始权重 w 的梯度与重新计算的梯度相等
        self.assertEqual(orig_w, w.grad)
    
    # 使用 Torch 的配置装饰器，捕获标量输出和动态输出形状操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    # 定义测试方法 test_unbacked_cat_backwards_save_data_dependent，接受设备参数
    def test_unbacked_cat_backwards_save_data_dependent(self, device):
        # 定义内部函数 f，接受输入 x 和权重 w
        def f(x, w):
            # 设备赋值为 w 的设备
            device = w.device
            # 将 x 转换为列表 a 和 b
            a, b = x.tolist()
            # 创建长度为 a 的全 1 张量 ta，并指定设备
            ta = torch.ones(a, device=device)
            # 创建长度为 b 的全 1 张量 tb，并指定设备
            tb = torch.ones(b, device=device)
            # pa 和 pb 分别为 ta 和 tb 与 w 的乘积，使其需要梯度
            pa = ta * w
            pb = tb * w
            # 将 pa 和 pb 连接成一个张量 r
            r = torch.cat([pa, pb])
            # 返回张量 r 的元素和
            return r
    
        # 创建输入张量 x
        x = torch.tensor([4, 9])
        # 创建随机权重张量 w，并指定需要梯度
        w = torch.randn(1, requires_grad=True)
        # 调用函数 f，并对其结果求和后进行反向传播
        f(x, w).sum().backward()
        # 记录原始权重 w 的梯度
        orig_w = w.grad
        # 将权重 w 的梯度置空
        w.grad = None
    
        # 使用 Torch 编译器编译完整图形的函数 f，对其结果求和后进行反向传播
        torch.compile(fullgraph=True)(f)(x, w).sum().backward()
        # 断言原始权重 w 的梯度与重新计算的梯度相等
        self.assertEqual(orig_w, w.grad)
    
    # 使用 Torch 的配置装饰器，捕获标量输出和动态输出形状操作
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    # 使用 Torch Inductor 的配置装饰器，启用隐式回退
    @torch._inductor.config.patch(implicit_fallbacks=True)
    # 定义测试方法 test_dynamic_stride_nobreak，接受设备参数
    def test_dynamic_stride_nobreak(self, device):
        # 使用 Torch 自定义操作库定义函数 foo，不修改参数
        @torch.library.custom_op("test::foo", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 从输入张量 x 中获取步长值
            stride = x.item()
            # 返回一个空的步长为 stride 的张量，指定设备为 x 的设备
            return torch.empty_strided((1,), (stride,), device=x.device)
    
        # 为函数 foo 注册一个假的版本，接受输入张量 x 并返回一个动态大小的步长
        @foo.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            # 获取 Torch 库的上下文
            ctx = torch.library.get_ctx()
            # 创建一个新的动态大小步长
            stride = ctx.new_dynamic_size()
            # 返回一个空的步长为 stride 的张量，指定设备为 x 的设备
            return torch.empty_strided((1,), (stride,), device=x.device)
    
        # 使用 Torch 编译器编译完整图形的函数 f
        @torch.compile(fullgraph=True)
        def f(x):
            # 调用 Torch 操作库中的 foo 函数，传入输入张量 x
            r = torch.ops.test.foo(x)
            # 从返回张量 r 中获取步长信息
            y = r.stride(0)
            # 返回一个空的张量，其大小由 y 决定，设备为输入张量 x 的设备
            return torch.empty(y, device=x.device)
    
        # 调用函数 f，传入输入张量 [3]，并指定设备
        f(torch.tensor([3], device=device))
    
    # 使用 Torch Inductor 的配置装饰器，禁用 C++ 代码生成
    @torch._inductor.config.patch(disable_cpp_codegen=True)
    def test_floor(self):
        # 定义一个函数 `fn`，参数为 `x`
        def fn(x):
            # 获取张量 `x` 最后一个维度的大小
            n = x.size(-1)
            # 计算 `n * 0.2` 的整数部分，将其加到 `x` 上，并加 1
            y = x + int(n * 0.2) + 1
            return y

        # 编译函数 `fn` 以优化执行
        opt = self.compile_fn(fn)
        # 创建一个随机张量 `x0`，并计算 `fn(x0)` 的预期结果
        x0 = torch.rand(5)
        ref0 = fn(x0)
        # 使用优化后的函数 `opt` 对 `x0` 执行，得到结果 `res0`
        res0 = opt(x0)
        # 断言优化结果 `res0` 与预期结果 `ref0` 相等
        self.assertEqual(ref0, res0)
        # 创建另一个随机张量 `x1`，并计算 `fn(x1)` 的预期结果
        x1 = torch.rand(8)
        ref1 = fn(x1)
        # 使用优化后的函数 `opt` 对 `x1` 执行，得到结果 `res1`
        res1 = opt(x1)
        # 断言优化结果 `res1` 与预期结果 `ref1` 相等
        self.assertEqual(ref1, res1)

    @onlyOn(GPU_TYPE)
    def test_pad_dynamic(self, device):
        # 定义一个函数 `get_same_padding`，计算卷积操作的填充值
        def get_same_padding(x: int, k: int, s: int, d: int):
            return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

        # 定义一个函数 `pad_same`，对输入张量进行相同填充操作
        def pad_same(x, k, s, d=(1, 1), value=0):
            ih, iw = x.size()[-2:]
            # 计算高度和宽度方向的填充值
            pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(
                iw, k[1], s[1], d[1]
            )
            # 如果需要填充，则对输入张量 `x` 进行填充操作
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x,
                    [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                    value=value,
                )
            return x

        # 创建一个随机张量 `x`，形状为 (2, 24, 110, 110)，并指定设备
        x = torch.randn(2, 24, 110, 110, device=device)
        # 编译函数 `pad_same` 以优化执行
        opt = self.compile_fn(pad_same)
        # 使用优化后的函数 `opt` 对 `x` 进行填充操作，得到结果 `res`
        res = opt(x, (5, 5), (2, 2))
        # 对比使用原始函数 `pad_same` 对 `x` 进行填充操作的预期结果 `ref`
        ref = pad_same(x, (5, 5), (2, 2))
        # 断言优化结果 `res` 与预期结果 `ref` 相等
        self.assertEqual(res, ref, atol=0, rtol=0)

    def test_slice_scatter(self, device):
        # 定义一个函数 `fn`，参数为 `i`
        def fn(i):
            # 获取张量 `i` 第一个维度的大小
            s3 = i.size(0)
            # 创建两个张量 `x` 和 `y`，形状分别为 (64, s3) 和 (64, s3 // 2)
            x = torch.ones(64, s3, device=device)
            y = torch.ones(64, s3 // 2, device=device)
            # 使用 `torch.slice_scatter` 函数进行张量操作
            return torch.slice_scatter(x, y, 1, s3 // 2, 2 * (s3 // 2))

        # 创建一个随机张量 `a`，并编译函数 `fn` 以优化执行
        a = torch.randn(16, device=device)
        cfn = self.compile_fn(fn)
        # 计算使用优化后的函数 `cfn` 对 `a` 执行的结果 `actual`
        expect = fn(a)
        actual = cfn(a)
        # 断言优化结果 `actual` 与预期结果 `expect` 相等
        self.assertEqual(expect, actual)

    def test_slice_index_changing_sign(self, device):
        # 定义一个函数 `fn`，参数为 `x` 和 `y`
        def fn(x, y):
            # 获取张量 `y` 的形状
            y0, y1 = y.shape
            # 返回张量 `x` 的切片操作结果
            return x[: (y0 - y1)].clone()

        # 创建一个随机张量 `a` 和编译函数 `fn` 以优化执行
        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # 创建一个形状为 (16, 2) 的随机张量 `b`，并计算 `fn(a, b)` 的预期结果
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        # 使用优化后的函数 `cfn` 对 `a` 和 `b` 执行，得到结果 `actual`
        actual = cfn(a, b)
        # 断言优化结果 `actual` 与预期结果 `expect` 相等
        self.assertEqual(expect, actual)

        # 创建一个形状为 (2, 16) 的随机张量 `b`，并计算 `fn(a, b)` 的预期结果
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        # 使用优化后的函数 `cfn` 对 `a` 和 `b` 执行，得到结果 `actual`
        actual = cfn(a, b)
        # 断言优化结果 `actual` 与预期结果 `expect` 相等
        self.assertEqual(expect, actual)

    def test_sym_stride_lowering(self, device):
        # 定义一个函数 `fn`，参数为 `x`
        def fn(x):
            # 获取张量 `(x + 1)` 在第 0 维上的步长
            s0 = (x + 1).stride(0)
            # 返回张量 `x` 乘以步长 `s0` 的结果
            return x * s0

        # 创建一个随机张量 `a` 和编译函数 `fn` 以优化执行
        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        # 断言优化后的函数 `cfn` 对 `a` 执行的结果与未优化函数 `fn` 的结果相等
        self.assertEqual(fn(a), cfn(a))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # 定义测试函数，用于测试编译后的函数在特定设备上的行为
    def test_item_materialize(self, device):
        # 定义一个简单的函数 fn，计算输入张量 x 按 dim=0 求和后再展平成一维张量，转换为列表返回
        def fn(x):
            return x.sum(dim=0).view(4).tolist()

        # 使用 torch.compile 方法编译函数 fn，fullgraph=True 表示生成完整的计算图
        cfn = torch.compile(fullgraph=True)(fn)

        # 创建一个全为 1 的张量 a，数据类型为 torch.int64，放置在指定设备上
        a = torch.ones(3, 4, dtype=torch.int64, device=device)
        # 断言编译后的函数 cfn 对输入张量 a 的结果与原始函数 fn 的结果相等
        self.assertEqual(cfn(a), fn(a))

    # 定义测试函数，用于测试 fn 函数对输入 x 和 y 的行为
    def test_abs(self, device):
        # 定义函数 fn，接受两个输入 x 和 y，返回根据 y 的形状计算的结果
        def fn(x, y):
            y0, y1 = y.shape
            # 使用切片操作检查在包装代码中的绝对值，乘法测试内核代码中的绝对值
            return x[: abs(y0 - y1)] * abs(y0 - y1)

        # 创建一个形状为 (32, 32) 的随机张量 a，放置在指定设备上
        a = torch.randn(32, 32, device=device)
        # 使用 self.compile_fn 方法编译函数 fn
        cfn = self.compile_fn(fn)

        # 创建一个形状为 (16, 2) 的随机张量 b，用于测试 y0 > y1 的情况
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        # 断言编译后的函数 cfn 对输入 a 和 b 的结果与原始函数 fn 的结果相等
        self.assertEqual(expect, actual)

        # 创建一个形状为 (2, 16) 的随机张量 b，用于测试 y0 < y1 的情况
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        # 断言编译后的函数 cfn 对输入 a 和 b 的结果与原始函数 fn 的结果相等
        self.assertEqual(expect, actual)

    # 定义测试函数，测试 fn 函数对输入 x 和 mul 的行为
    def test_float_is_integer(self, device):
        def fn(x, mul, dim=-1):
            # 获取指定维度 dim 上的大小
            size = x.size(dim)
            # 计算 size 除以 mul 的结果
            m = size / mul
            # 如果 m 是整数，则返回 m，否则返回 size
            if m.is_integer():
                return m
            return size

        # 创建一个形状为 (3, 6, 4, 2) 的随机张量 a，放置在指定设备上
        a = torch.randn((3, 6, 4, 2), device=device)
        # 使用 self.compile_fn 方法编译函数 fn
        cfn = self.compile_fn(fn)

        expect = fn(a, 2)
        actual = cfn(a, 2)
        # 断言编译后的函数 cfn 对输入 a 和 2 的结果与原始函数 fn 的结果相等
        self.assertEqual(expect, actual)

    # 标记为仅在 CPU 上运行的测试函数，测试 fn 函数对输入 x 的行为
    @onlyCPU
    def test_arithmetic_constant_folding(self, device):
        # 定义一个测试函数 test，用于测试编译后函数 fn 的行为
        def test(fn):
            cfn = self.compile_fn(fn)
            expect = fn(3)
            actual = cfn(3)
            # 断言编译后的函数 cfn 对输入 3 的结果与原始函数 fn 的结果相等
            self.assertEqual(expect, actual)

        # 定义一个加法函数 add，返回输入张量 x 与全为 0 的张量的和
        def add(x):
            return x + torch.zeros(3)

        test(add)

        # 定义一个乘法函数 mul，返回输入张量 x 与全为 1 的张量的乘积
        def mul(x):
            return x * torch.ones(3)

        test(mul)

        # 定义一个除法函数 div，返回输入张量 x 与全为 1 的张量的商
        def div(x):
            return x / torch.ones(3)

        test(div)

    # 标记为仅在 CPU 上运行的测试函数，测试 fn 函数对输入 x 的行为
    @onlyCPU
    def test_sub_constant_folding(self, device):
        # 定义一个减法函数 sub，返回输入张量 x 与全为 0 的张量的差
        def sub(x):
            return x - torch.zeros(3)

        # 使用 self.compile_fn 方法编译函数 sub
        cfn = self.compile_fn(sub)
        expect = sub(3)
        actual = cfn(3)
        # 断言编译后的函数 cfn 对输入 3 的结果与原始函数 sub 的结果相等
        self.assertEqual(expect, actual)

    # 定义测试函数，测试 fn 函数对输入 a 的行为
    def test_full_symbolic_value(self, device):
        # 定义一个函数 fn，返回一个形状为 (3,) 的全为 a 的张量和一个形状为 (3,) 的全为 torch.sym_float(a) 的张量
        def fn(a):
            return torch.full((3,), a), torch.full((3,), torch.sym_float(a))

        # 使用 self.compile_fn 方法编译函数 fn
        cfn = self.compile_fn(fn)
        expect = fn(5)
        actual = cfn(5)
        # 断言编译后的函数 cfn 对输入 5 的结果与原始函数 fn 的结果相等
        self.assertEqual(expect, actual)
    # 定义一个测试函数，用于测试插值函数的行为，要求参数device为设备
    def test_interpolate_ceil_eq(self, device):
        # 导入math库中的ceil函数，并赋值给变量ceiling
        ceiling = math.ceil
        # 导入operator库中的truediv函数，并赋值给变量IntTrueDiv
        IntTrueDiv = operator.truediv

        # 定义一个内部函数fn，接受参数t
        def fn(t):
            # 获取t的尺寸，并分别赋值给s0, s2, s3
            s0, s2, s3 = t.size()
            # 根据t的尺寸计算出x的尺寸，使用torch.zeros创建全零张量x，数据类型为torch.bfloat16
            x = torch.zeros(
                (
                    s0,
                    2048,
                    ceiling(IntTrueDiv(2 * ((s2 - 1) // 8) + 2, 1)),
                    ceiling(IntTrueDiv(2 * ((s3 - 1) // 8) + 2, 1)),
                ),
                dtype=torch.bfloat16,
            )
            # 调用torch.nn.functional.interpolate函数对x进行插值操作，scale_factor=2，插值模式为"nearest"
            return torch.nn.functional.interpolate(
                x,
                scale_factor=2,
                mode="nearest",
            )

        # 使用self.compile_fn对fn函数进行编译，得到cfn函数
        cfn = self.compile_fn(fn)
        # 生成一个形状为(4, 16, 18)的随机张量arg
        arg = torch.randn(4, 16, 18)
        # 计算预期结果expect，即fn(arg)
        expect = fn(arg)
        # 计算实际结果actual，即cfn(arg)
        actual = cfn(arg)
        # 使用self.assertEqual断言实际输出等于预期输出
        self.assertEqual(expect, actual)

    # 定义一个测试函数，用于测试全局重新编译的行为，要求参数device为设备
    def test_full_recompiles(self, device):
        # 定义内部函数fn，接受参数x
        def fn(x):
            # 获取x的形状信息，将形状的第二个元素赋值给L
            _, L = x.shape
            # 使用torch.full创建一个形状为(L, L)的张量，元素值为torch.float16的最小正规数，设备为device
            return torch.full((L, L), torch.finfo(torch.float16).min, device=device)

        # 使用self.compile_fn对fn函数进行编译，得到cfn函数
        cfn = self.compile_fn(fn)

        # 导入functools库，使用functools.partial创建一个输入函数input_fn，调用torch.randint生成随机整数张量，设备为device
        import functools
        input_fn = functools.partial(torch.randint, 10, 1000, device=device)

        # 分别调用cfn两次，输入参数分别为input_fn((2, 3))和input_fn((2, 4))
        cfn(input_fn((2, 3)))
        cfn(input_fn((2, 4)))  # 预期此处不重新编译

        # 检查帧0的编译次数
        from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER
        # 使用self.assertEqual断言帧0的编译次数为1
        self.assertEqual(FRAME_COMPILE_COUNTER[0], 1)

    # 使用@parametrize装饰器定义参数化测试函数，用于测试数学运算函数的行为，要求参数device为设备，op为数学函数
    @parametrize(
        "op",
        [
            math.sqrt,
            math.sin,
            math.cos,
            math.cosh,
            math.sin,
            math.sinh,
            math.tan,
            math.tanh,
            math.asin,
            math.acos,
            math.atan,
        ],
    )
    def test_math_ops(self, device, op):
        # 定义内部函数func，接受参数x, fn, a
        def func(x, fn, a):
            # 返回x与fn(a)的加法运算结果
            return x + fn(a)

        # 使用self.compile_fn对func函数进行编译，开启全图模式(fullgraph=True)，得到cfunc函数
        cfunc = self.compile_fn(func, fullgraph=True)
        # 生成一个形状为(10,)的随机张量x，设备为device
        x = torch.rand(10, device=device)
        # 根据op的类型选择a的值，如果op是math.asin或math.acos，则a为-1，否则为12
        a = -1 if op in (math.asin, math.acos) else 12
        # 计算预期结果expected，即func(x, op, a)
        expected = func(x, op, a)
        # 计算实际结果output，即cfunc(x, op, a)
        output = cfunc(x, op, a)
        # 使用self.assertEqual断言实际输出等于预期输出
        self.assertEqual(output, expected)

    # 使用@torch._dynamo.config.patch修饰器定义测试函数，用于测试item操作、非支持步幅及无断点终止的行为，要求参数device为设备
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_unbacked_stride_nobreak(self, device):
        # 定义动态编译函数f，接受参数x
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            # 将x转换为标量a
            a = x.item()
            # 检查a是否是有效大小
            torch._check_is_size(a)
            # 检查a是否大于等于1
            torch._check(a >= 1)
            # 检查a是否小于等于10
            torch._check(a <= 10)
            # 返回一个全1张量，形状为(a, a)
            return torch.ones(a, a)

        # 调用f函数，传入torch.tensor([5], device=device)作为参数
        f(torch.tensor([5], device=device))
    # 定义一个测试方法，用于测试动态形状排序并进行检查，需要指定设备参数
    def test_sort_dynamic_shape_with_check(self, device):
        # 如果是 ROCm 环境或者设备类型不是 GPU，则定义一个检查函数，验证生成的内核数量为 0
        if TEST_WITH_ROCM or torch.device(device).type != GPU_TYPE:

            def check_count(n):
                # 断言生成的内核数量为 0
                self.assertEqual(metrics.generated_kernel_count, 0)

        else:
            # 否则定义一个检查函数，验证生成的内核数量为 n
            def check_count(n):
                self.assertEqual(metrics.generated_kernel_count, n)

        # 测试动态形状，当形状静态已知且小于等于 256 时，生成持久排序内核
        def fn(a, descending):
            # 检查输入张量的最后一个维度是否小于等于 256
            torch._check(a.shape[-1] <= 256)
            # 返回在指定维度上排序后的张量及相关信息
            return a.sort(dim=-1, stable=True, descending=descending)

        # 创建一个形状为 (10, 128) 的随机张量，并将部分区域设置为固定值 1.0
        inp = torch.rand(10, 128, dtype=torch.float32, device=device)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        # 重置度量器对象中的内核计数
        metrics.reset()

        # 使用动态编译方式编译排序函数 fn
        opt_fn = torch.compile(fn, dynamic=True)
        
        # 预期排序结果，按升序排序
        expect = fn(inp, False)
        # 实际调用编译后的优化函数进行排序
        actual = opt_fn(inp, False)
        # 断言排序后的结果与预期结果一致
        self.assertEqual(actual, expect)
        # 检查生成的内核数量是否为 1
        check_count(1)

        # 预期排序结果，按降序排序
        expect = fn(inp, True)
        # 实际调用编译后的优化函数进行排序
        actual = opt_fn(inp, True)
        # 断言排序后的结果与预期结果一致
        self.assertEqual(actual, expect)
        # 检查生成的内核数量是否为 2
        check_count(2)

        # 针对非二次幂大小的输入张量
        inp[:, :120]

        # 再次检查排序结果，按升序排序
        expect = fn(inp, False)
        # 再次调用编译后的优化函数进行排序
        actual = opt_fn(inp, False)
        # 断言排序后的结果与预期结果一致
        self.assertEqual(actual, expect)
        # 检查生成的内核数量是否继续为 2（重用现有的内核）
        check_count(2)

        # 再次检查排序结果，按降序排序
        expect = fn(inp, True)
        # 再次调用编译后的优化函数进行排序
        actual = opt_fn(inp, True)
        # 断言排序后的结果与预期结果一致
        self.assertEqual(actual, expect)
        # 检查生成的内核数量是否继续为 2（重用现有的内核）
        check_count(2)
# 使用给定的测试类和全局变量实例化设备类型测试，允许使用任何类型的处理器单元（XPU）。
instantiate_device_type_tests(TestInductorDynamic, globals(), allow_xpu=True)

if __name__ == "__main__":
    # 从torch._inductor.test_case模块导入运行测试的函数
    from torch._inductor.test_case import run_tests

    # 在ASAN环境下，由于https://github.com/pytorch/pytorch/pull/94068，执行速度较慢
    # 如果有CPU或GPU，并且不在ASAN测试中，则运行需要文件锁的测试
    if (HAS_CPU or HAS_GPU) and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
```