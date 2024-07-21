# `.\pytorch\test\functorch\test_vmap.py`

```
# 导入必要的库和模块
import contextlib  # 提供运行时上下文管理的工具
import functools  # 提供了创建偏函数（partial）和高阶函数（higher-order functions）的工具
import itertools  # 提供用于操作迭代器的函数，如排列组合等
import os  # 提供与操作系统交互的功能
import random  # 提供生成随机数的工具
import types  # 提供操作类型和动态创建类型的工具
import unittest  # 提供单元测试框架
import warnings  # 提供警告相关的功能
from collections import namedtuple  # 提供命名元组的工具
from typing import OrderedDict  # 提供类型注解支持

# 从 common_utils 导入多个函数和常量
from common_utils import (
    check_vmap_fallback,
    compute_quantities_for_vmap_test,
    decorate,
    DisableVmapFallback,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    xfail,
)

# 从 functorch_additional_op_db 导入 additional_op_db
from functorch_additional_op_db import additional_op_db

# 导入 functorch 库
import functorch

# 导入 torch 库和 torch.nn.functional 中的 F 函数
import torch
import torch.nn.functional as F

# 从 functorch 导入特定的函数
from functorch import grad, grad_and_value, jacfwd, jvp, vjp, vmap

# 从 functorch.experimental 导入 chunk_vmap
from functorch.experimental import chunk_vmap

# 从 torch 中导入 Tensor 类型
from torch import Tensor

# 从 torch._C._functorch 中导入 reshape_dim_into 和 reshape_dim_outof 函数
from torch._C._functorch import reshape_dim_into, reshape_dim_outof

# 从 torch._functorch.make_functional 中导入 functional_init_with_buffers 函数
from torch._functorch.make_functional import functional_init_with_buffers

# 从 torch._functorch.vmap 中导入 restore_vmap 函数
from torch._functorch.vmap import restore_vmap

# 从 torch.testing._internal.autograd_function_db 中导入 autograd_function_db
from torch.testing._internal.autograd_function_db import autograd_function_db

# 从 torch.testing._internal.common_cuda 中导入 with_tf32_off
from torch.testing._internal.common_cuda import with_tf32_off

# 从 torch.testing._internal.common_device_type 中导入多个函数和常量
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    tol,
    toleranceOverride,
)

# 从 torch.testing._internal.common_methods_invocations 中导入多个函数和常量
from torch.testing._internal.common_methods_invocations import op_db

# 从 torch.testing._internal.common_utils 中导入多个函数和常量
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    unMarkDynamoStrictTest,
    xfailIfTorchDynamo,
)

# 从 torch.utils._pytree 中导入 pytree
from torch.utils import _pytree as pytree

# 定义一个常量，用于匹配回退警告信息的正则表达式
FALLBACK_REGEX = "There is a performance drop"

# 定义一个上下文管理器类，用于启用 vmap 回退警告
class EnableVmapFallbackWarnings:
    def __enter__(self):
        # 获取当前 vmap 回退警告状态，并设置为开启状态
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        # 恢复之前的 vmap 回退警告状态
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)

# 标记为 Dynamo 严格测试的测试类
@markDynamoStrictTest
class TestVmapAPI(TestCase):
    # 测试非张量输出时是否引发异常
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(ValueError, "got type <class 'float'>"):
            vmap(lambda x: 3.14)(torch.ones(3))

        # 测试多个输出时是否引发异常
        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'>"):
            vmap(multiple_outputs)(torch.ones(3))
    # 测试函数，用于验证如果输入张量的映射维度不同则抛出异常
    def test_different_map_dim_size_raises(self):
        # 创建两个随机张量 x 和 y
        x = torch.randn(2)
        y = torch.randn(3)
        # 预期的异常消息
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        # 测试 vmap 函数对 torch.mul 的应用，预期抛出 ValueError 异常，并匹配预期的异常消息
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        # 测试 vmap 函数对 lambda 函数的应用，使用 in_dims=((0, 0),)，预期抛出 ValueError 异常，并匹配预期的异常消息
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        # 测试 vmap 函数对 lambda 函数的应用，使用 in_dims=({"x": 0, "y": 0},)，预期抛出 ValueError 异常，并匹配预期的异常消息
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    # 测试函数，用于验证如果函数没有输入则抛出异常
    def test_func_with_no_inputs(self):
        # 预期的异常消息
        expected_msg = "got no inputs"

        # 定义一个没有输入的函数 foo
        def foo():
            return torch.randn(3)

        # 定义一个有输入的函数 bar
        def bar(x):
            return torch.randn(3)

        # 测试 vmap 函数对 foo 的应用，预期抛出 ValueError 异常，并匹配预期的异常消息
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        # 测试 vmap 函数对 bar 的应用，预期抛出 ValueError 异常，并匹配预期的异常消息
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    # 测试函数，用于验证如果函数没有张量输入则抛出异常
    def test_func_with_no_tensors(self):
        # 定义一个带有输入 x 的函数 foo
        def foo(x):
            return torch.randn(3)

        # 测试 vmap 函数对 foo 的应用，传入一个标量作为输入，预期抛出 ValueError 异常，并匹配指定的异常消息
        with self.assertRaisesRegex(ValueError, "at least one Tensor"):
            vmap(foo, (None,))(1)

    # 测试函数，用于验证对常数函数的映射
    def test_constant_function(self):
        # 对所有元素都返回常数 3.14 的 lambda 函数应用 vmap 函数
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        # 验证输出是否等于期望的张量 [3.14, 3.14, 3.14]
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    # 测试函数，用于验证对单个输入的函数映射
    def test_single_input(self):
        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)

        # 定义一个对输入 x 求平方的函数 square
        def square(x):
            return x * x

        # 对函数 square 应用 vmap 函数
        output = vmap(square)(x)
        # 验证输出是否等于 x 的每个元素的平方
        self.assertEqual(output, x * x)

    # 测试函数，用于验证对多个输入的函数映射
    def test_multiple_inputs(self):
        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 对 torch.mul 函数应用 vmap 函数
        output = vmap(torch.mul)(x, y)
        # 验证输出是否等于 x 和 y 的逐元素乘积
        self.assertEqual(output, x * y)

    # 测试函数，用于验证对多个输出的函数映射
    def test_multiple_outputs(self):
        # 定义一个返回输入 x 的平方和立方的函数 foo
        def foo(x):
            return x * x, x * x * x

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 对函数 foo 应用 vmap 函数
        outputs = vmap(foo)(x)
        # 验证输出的第一个元素是否等于 x 的平方，第二个元素是否等于 x 的立方
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    # 测试函数，用于验证对返回张量元组的函数映射
    def test_multiple_outputs2(self):
        # 定义一个返回元组 (x, x) 的函数 returns_tuple_of_tensors
        def returns_tuple_of_tensors(x):
            return (x, x)

        # 定义一个返回列表 [x, x] 的函数 returns_list_of_two_tensors
        def returns_list_of_two_tensors(x):
            return [x, x]

        # 定义一个返回列表 [x] 的函数 returns_list_of_one_tensor
        def returns_list_of_one_tensor(x):
            return [x]

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)

        # 应用 vmap 函数对不同的返回值形式进行测试
        # 不应该抛出异常
        vmap(returns_tuple_of_tensors)(x)
        vmap(returns_list_of_two_tensors)(x)
        vmap(returns_list_of_one_tensor)(x)

    # 测试函数，用于验证嵌套映射的行为
    def test_nested_with_same_map_dim(self):
        # 创建两个形状为 (2, 3, 5) 的随机张量 x 和 y
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        # 对 vmap(vmap(torch.mul)) 的应用，验证逐元素乘积的结果
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        # 对 vmap(vmap(vmap(torch.mul))) 的应用，验证逐元素乘积的结果
        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)
    def test_nested_with_diag_embed(self):
        # 对 diag_embed 进行特殊测试，因为它是通过条件功能化进行注册的。
        x = torch.randn(3, 3, 5)
        # 使用 vmap 对 diag_embed 进行双重映射
        output = vmap(vmap(torch.diag_embed))(x)
        self.assertEqual(output, torch.diag_embed(x))

    def test_nested_with_different_map_dim(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        # 使用 vmap 和 lambda 函数实现不同维度映射
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        # 嵌套使用 vmap、lambda 函数实现多维度映射
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        x = torch.randn(3)
        y = torch.randn(5)
        # 在内部 vmap 中执行空操作
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_unsupported_op_err_msg(self):
        # 不支持的 view 操作
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        # TODO: 找到一个 view 操作
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     vmap(torch.ravel)(tensor)

        def out_op(x, y):
            return torch.abs(x, out=y)

        # 使用 vmap 测试自定义函数 out_op
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        # 不支持非张量返回。这是 vmap 的一个限制；不返回张量的函数必须特别处理
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.equal)(tensor, tensor)
    def test_nonzero_out_dims(self):
        # Basic test
        # 创建一个形状为 (2, 3) 的随机张量
        tensor = torch.randn(2, 3)
        # 对张量进行 vmap 操作，使用 lambda 函数保持不变，输出维度设为 1
        result = vmap(lambda x: x, out_dims=1)(tensor)
        # 断言结果与原张量按维度 1 排列后相等
        self.assertEqual(result, tensor.permute(1, 0))
        # 断言结果张量与原张量共享相同的数据指针
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # Test that the batch dimension gets permuted to dim 2
        # 创建一个形状为 (2, 3, 5, 7) 的随机张量
        tensor = torch.randn(2, 3, 5, 7)
        # 对张量进行 vmap 操作，输出维度设为 2
        result = vmap(lambda x: x, out_dims=2)(tensor)
        # 断言结果与原张量按维度 1 和 2 交换后相等
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        # 断言结果张量与原张量共享相同的数据指针
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # negative out_dim
        # 创建一个形状为 (2, 3, 5, 7) 的随机张量
        tensor = torch.randn(2, 3, 5, 7)
        # 对张量进行 vmap 操作，输出维度设为 -1
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        # 断言结果与原张量按维度 1、2 和 3 交换后相等
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        # 断言结果张量与原张量共享相同的数据指针
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # check that out_dims works on ALL outputs
        # 创建两个形状为 (2, 3, 5, 7) 的随机张量
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        # 对两个张量进行 vmap 操作，输出维度设为 2
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        # 断言结果分别与原张量按维度 1 和 2 交换后的结果相等
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

        # use out_dims with the maximum vmap-able tensor dims (64 dims)
        # 定义一个有 64 维的张量形状
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        # 对张量进行 vmap 操作，输出维度设为 2
        result = vmap(lambda x: x, out_dims=2)(tensor)
        # 断言结果的形状与预期的形状相等
        self.assertEqual(result.shape, expected_shape)

        # test something that is not the identity function
        # 定义一个函数 foo，对两个输入张量进行操作
        def foo(x, y):
            return x, x * y, x * y * y

        # 创建两个形状为 (2, 3, 5) 的随机张量
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        # 对张量进行 vmap 操作，输出维度设为 1
        result = vmap(foo, out_dims=1)(x, y)
        # 断言结果分别与原张量按维度 1 排列后相等
        self.assertEqual(
            result,
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

    def test_multiple_out_dims(self):
        # 定义一个返回输入张量两次的函数 foo
        def foo(x):
            return x, x

        # 定义一个返回输入张量及其乘积的函数 bar
        def bar(x, y):
            return x, x, x, x * y

        # 创建两个形状为 (2, 3, 5) 的随机张量
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        # 对张量进行 vmap 操作，输出维度分别为 (0, 1)
        result = vmap(foo, out_dims=(0, 1))(x)
        # 断言结果分别与原张量按维度 0 和 1 排列后相等
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        # 对张量进行 vmap 操作，输出维度分别为 (-1, 0, 1, 2)
        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        # 定义预期的结果元组
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        # 断言结果与预期结果相等
        self.assertEqual(result, expected)
    def test_nested_out_dims(self):
        y = torch.randn(2, 3, 5, 7)

        # Inner vmap has non-zero out_dim
        # 在内部 vmap 中使用非零的 out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # all vmaps have non-zero out_dim
        # 所有的 vmap 都使用非零的 out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # throwing in some negative out_dims
        # 使用一些负的 out_dims
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # testing fn that isn't the identity
        # 测试不是恒等函数的情况
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        def foo(x):
            return x

        # Test that we accept out_dims=(1,) for a function with one output.
        # 测试对于只有一个输出的函数，接受 out_dims=(1,)
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_none_tuple(self):
        def foo(x):
            return x, "hello world"

        tensor = torch.randn(2, 3)
        result = vmap(foo, out_dims=(0, None))(tensor)
        self.assertEqual(result[1], "hello world")
        self.assertEqual(result[0], tensor)

        def foo(x):
            x.add_(1)
            return None, "hello world"

        result = vmap(foo, out_dims=(None, None))(tensor)
        self.assertEqual(result, (None, "hello world"))

    def test_out_dims_none(self):
        def foo(x):
            return x

        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(
            ValueError, "can not return a BatchedTensor when out_dim is None"
        ):
            vmap(foo, out_dims=None)(tensor)

        def foo(x):
            x.add_(1)
            return "hello world"

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, "hello world")

    def test_out_dims_normal_tensor(self):
        def foo(x):
            return torch.arange(3)

        tensor = torch.randn(2, 3)
        result = vmap(foo)(tensor)
        self.assertEqual(result.shape, [2, 3])

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, torch.arange(3))
    # 测试函数，验证 vmap 对象的返回值
    def test_pytree_returns(self):
        # 创建一个 2x3 的随机张量 x
        x = torch.randn(2, 3)

        # 定义函数 f，接受一个参数 x，返回一个元组和列表的嵌套结构
        def f(x):
            # 对输入张量 x 求正弦值，并赋给变量 y
            y = x.sin()
            # 返回 y 本身，以及包含 y 的元组 (y, y)，和包含 y 的列表 [y, (y, y)]
            return y, (y, y), [y, (y, y)]

        # 使用 vmap 函数对 f 进行向量化映射，并解构结果
        y0, (y1, y2), (y3, (y4, y5)) = vmap(f)(x)

        # 断言，验证返回值的正确性
        self.assertEqual(y0, x.sin())
        self.assertEqual(y0, y1)
        self.assertEqual(y2, y1)
        self.assertEqual(y2, y3)
        self.assertEqual(y4, y3)
        self.assertEqual(y5, y4)

    # 测试函数，验证 vmap 对象处理 OrderedDict 返回值的情况
    def test_pytree_odict_returns(self):
        # 创建一个 2x3 的随机张量 x
        x = torch.randn(2, 3)

        # 定义函数 f，接受一个参数 t，返回一个 OrderedDict 对象
        def f(t):
            # 对输入张量 t 求正弦值，并赋给变量 y
            y = t.sin()
            # 返回一个包含 "sin" 键和对应值 y，以及 "cos" 键和对应值 t.cos() 的 OrderedDict 对象
            return OrderedDict([("sin", y), ("cos", t.cos())])

        # 使用 vmap 函数对 f 进行向量化映射，并赋值给 out
        out = vmap(f)(x)

        # 断言，验证 out 是 OrderedDict 类型，并与直接调用 f(x) 的结果 expected 相等
        assert isinstance(out, OrderedDict)
        expected = f(x)
        self.assertEqual(out["sin"], expected["sin"])
        self.assertEqual(out["cos"], expected["cos"])

    # 测试函数，验证 vmap 对象的返回值和输出维度设置
    def test_pytree_returns_outdims(self):
        # 创建一个 2x3 的随机张量 x
        x = torch.randn(2, 3)

        # 定义函数 f，接受一个参数 x，返回一个元组的嵌套结构
        def f(x):
            # 对输入张量 x 求正弦值，并赋给变量 y
            y = x.sin()
            # 返回 y 本身，以及包含 y 的元组 (y, y)
            return y, (y, y)

        # 使用 vmap 函数对 f 进行向量化映射，并设置输出维度为 (0, (0, 1))
        y0, (y1, y2) = vmap(f, out_dims=(0, (0, 1)))(x)

        # 断言，验证返回值的正确性
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, x.sin())
        self.assertEqual(y2, x.sin().t())

    # 测试函数，验证 vmap 对象的返回值和广播设置（简单情况）
    def test_pytree_returns_broadcast_simple(self):
        # 创建一个 2x3 的随机张量 x
        x = torch.randn(2, 3)

        # 定义函数 f，接受一个参数 x，返回一个元组的嵌套结构
        def f(x):
            # 对输入张量 x 求正弦值，并赋给变量 y
            y = x.sin()
            # 返回 y 本身，以及包含 y 的元组 (y, y)
            return y, (y, y)

        # 使用 vmap 函数对 f 进行向量化映射，并设置输出维度为 1（广播设置）
        y0, (y1, y2) = vmap(f, out_dims=1)(x)

        # 断言，验证返回值的正确性
        self.assertEqual(y0, x.sin().t())
        self.assertEqual(y1, y0)
        self.assertEqual(y2, y0)

    # 测试函数，验证 vmap 对象的返回值和广播设置（嵌套情况）
    def test_pytree_returns_broadcast_nested(self):
        # 创建一个 2x3 的随机张量 x
        x = torch.randn(2, 3)

        # 定义函数 f，接受一个参数 x，返回一个元组的嵌套结构
        def f(x):
            # 对输入张量 x 求正弦值，并赋给变量 y
            y = x.sin()
            # 返回 y 本身，以及包含 y 的元组 (y, y)
            return y, (y, y)

        # 使用 vmap 函数对 f 进行向量化映射，并设置输出维度为 (0, 1)（嵌套广播设置）
        y0, (y1, y2) = vmap(f, out_dims=(0, 1))(x)

        # 断言，验证返回值的正确性
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, y0.t())
        self.assertEqual(y2, y0.t())

    # 测试函数，验证当 out_dims 参数不合法时是否抛出 ValueError 异常
    def test_out_dims_must_be_int_or_collection_of_int_err_msg(self):
        # 定义错误消息
        msg = "must be an int, None or a python collection of ints"
        # 创建一个 2x3 的随机张量 tensor
        tensor = torch.randn(2, 3)

        # 断言，使用 vmap 函数时，当 out_dims 参数为非法字符串时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        # 断言，使用 vmap 函数时，当 out_dims 参数为非法元组时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)

    # 测试函数，验证当 out_dims 参数与返回值数量不匹配时是否抛出 ValueError 异常
    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        # 定义错误消息
        msg = "not compatible"
        # 创建一个 2x3x5 的随机张量 x
        x = torch.randn(2, 3, 5)

        # 断言，使用 vmap 函数时，当 out_dims 参数数量过多时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # 断言，使用 vmap 函数时，当 out_dims 参数数量不足时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)
    # 测试函数：test_out_dim_out_of_bounds_err_msg
    def test_out_dim_out_of_bounds_err_msg(self):
        # 定义错误消息
        msg = "Dimension out of range"
        # 创建一个形状为 (2, 3, 5) 的随机张量 x
        x = torch.randn(2, 3, 5)
        # 测试使用 vmap 函数，期望捕获 IndexError 并匹配指定的错误消息
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        # 再次测试，这次期望捕获负数的 out_dims
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    # 测试函数：test_non_zero_in_dims
    def test_non_zero_in_dims(self):
        # 创建一个形状为 (2, 3, 5) 的随机张量 tensor
        tensor = torch.randn(2, 3, 5)

        # 使用 vmap 函数，将批处理维度 (1) 移到最前面
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)

        # 使用 vmap 函数，将 in_dims 指定为 (0, 1)，逐元素相乘
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())

        # 使用 vmap 函数，将 in_dims 指定为 (1, 0)，逐元素相乘
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    # 测试函数：test_none_in_dims
    def test_none_in_dims(self):
        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # 使用 vmap 函数，将 in_dims 指定为 (0, None)，对 x 和 y 逐元素相乘
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # 使用 vmap 函数，将 in_dims 指定为 (0, None)，对 x 和标量 2 逐元素相乘
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    # 测试函数：test_nested_non_default_in_dims
    def test_nested_non_default_in_dims(self):
        # 创建两个形状为 (5, 2, 3) 和 (3, 5, 2) 的随机张量 x 和 y
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)

        # 使用 vmap 函数，嵌套调用三次，分别将 in_dims 指定为 (1, 0)，(1, 2)，(1, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    # 测试函数：test_nested_negative_in_dims
    def test_nested_negative_in_dims(self):
        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # 使用 vmap 函数，将 in_dims 指定为 (-1, -1)，逐元素相乘
        output = vmap(torch.mul, (-1, -1))(x, y)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output, (x * y).permute(1, 0))
    def test_non_default_in_dims_out_dims(self):
        x = torch.randn(2, 3, 5)

        # 使用 vmap 函数对输入张量 x 应用恒等映射，指定输入维度为 1，输出维度为 1
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # 使用 vmap 函数对输入张量 x 应用恒等映射，指定输入维度为 2，输出维度为 1
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            return x * 2

        # 使用 vmap 函数对输入张量 x 应用函数 foo，指定输入维度为 1，输出维度为 1
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # 使用 vmap 函数对输入张量 x 应用函数 foo，指定输入维度为 2，输出维度为 1
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # 嵌套测试
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_item_throws(self):
        def f(x):
            return x.item()

        # 测试调用 vmap 函数对包含 torch.Tensor 的函数 f 是否会抛出异常
        with self.assertRaisesRegex(RuntimeError, r"item\(\) on a Tensor"):
            vmap(f)(torch.randn(3))

    def test_data_dependent_control_flow_throws(self):
        def f(x):
            if x:
                return x
            return 0

        # 测试调用 vmap 函数对包含数据相关控制流的函数 f 是否会抛出异常
        with self.assertRaisesRegex(RuntimeError, r"data-dependent control flow"):
            vmap(f)(torch.randn(3))

    def test_accepts_nested_inputs(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # 单层嵌套测试
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        self.assertEqual(out, x + y)

        # 多层嵌套测试
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        self.assertEqual(out, x + x + y + y)
    # 测试当输入维度类型错误时是否抛出异常信息
    def test_in_dims_wrong_type_err_msg(self):
        # 创建两个随机张量 x 和 y
        x = torch.randn(3)
        y = torch.randn(3)
        # 定义错误信息的正则表达式消息
        msg = r"expected `in_dims` to be int or a \(potentially nested\) tuple"
        # 断言调用 vmap 函数时传入非预期类型的 in_dims 参数会抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        # 同上，使用 set 类型的 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        # 同上，使用字符串类型的 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, "lol")(x, y)
        # 同上，使用嵌套列表作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # 下面这个应该不会抛出异常
        vmap(torch.mul, (0, 0))(x, y)

    # 测试当输入维度不足时是否抛出异常信息
    def test_not_enough_in_dims_err_msg(self):
        # 创建两个随机张量 x 和 y
        x = torch.randn(3)
        y = torch.randn(3)
        # 定义错误信息的正则表达式消息
        msg = r"in_dims is not compatible with the structure of `inputs`"
        # 断言调用 vmap 函数时传入维度不足的 in_dims 参数会抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        # 同上，传入多于所需维度的 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        # 同上，使用包含列表的元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        # 同上，使用嵌套元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        # 下面这个应该不会抛出异常
        vmap(torch.mul, (0, 0))(x, y)

    # 测试当输入维度为整数但输入类型不是张量时是否抛出异常信息
    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        # 定义一个简单的函数 foo 和 bar
        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        # 创建一个随机张量 x
        x = torch.randn(2, 3)

        # 定义错误信息的消息字符串
        msg = "Got in_dim=0 for an input but the input is of type"
        # 断言调用 vmap 函数时传入整数 in_dim 参数但输入不是张量会抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        # 同上，使用元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        # 同上，使用包含列表的元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # 下面这个应该不会抛出异常
        vmap(torch.sum, (0, None))(x, 0)

    # 测试当输入维度超出张量维度范围时是否抛出异常信息
    def test_in_dim_not_in_tensor_err_msg(self):
        # 定义一个简单的函数 foo
        def foo(x):
            return x * x

        # 创建两个随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # 定义错误信息的正则表达式消息
        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        # 断言调用 vmap 函数时传入超出张量维度范围的 in_dim 参数会抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        # 同上，使用元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        # 同上，传入负数作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-3,))(x)
        # 同上，传入超过张量维度的索引作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        # 同上，使用包含列表的元组作为 in_dims 参数
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # 下面这两个应该不会抛出异常
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))
    # 测试函数：验证默认情况下，torch.vmap 在未启用警告时不会触发警告信息
    def test_fallback_does_not_warn_by_default(self):
        # 获取测试用的函数对象
        op = torch._test_functorch_fallback
        # 创建随机张量 x 和 y
        x = torch.randn(11)
        y = torch.randn(11)
        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as wa:
            # 对 op 进行 vmap 操作
            torch.vmap(op)(x, y)
            # 这里唯一的警告是关于实验性质的 "vmap is experimental" 警告，
            # 而非来自 vmap 回退路径的警告。
            # 验证捕获的警告数量是否为 1
            self.assertEqual(len(wa), 1)

    @unittest.expectedFailure
    # 测试函数：验证在启用警告时，torch.vmap 在回退路径发生时会触发警告
    def test_fallback_warns_when_warnings_are_enabled(self):
        # 注意：有一天我们将为 torch.atan2 实现一个批处理规则。
        # 如果/当我们这样做时，应该替换此测试，以测试另一个运算符的回退路径，以避免代码腐化。
        # 获取测试用的函数对象
        op = torch._test_functorch_fallback
        # 创建随机张量 x 和 y
        x = torch.randn(11)
        y = torch.randn(11)
        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as wa:
            # 启用 Vmap 回退警告上下文管理器
            with EnableVmapFallbackWarnings():
                torch.vmap(op)(x, y)
            # 验证捕获的警告数量是否为 2
            self.assertEqual(len(wa), 2)
            # 断言最后一条警告信息是否符合预期的回退正则表达式
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    # 辅助函数：断言是否使用了 vmap 回退路径
    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        return
        # 使用 warnings 模块捕获警告信息
        # with warnings.catch_warnings(record=True) as wa:
        #     # 启用 Vmap 回退警告上下文管理器
        #     with EnableVmapFallbackWarnings():
        #         result = vmap(*vmap_args)(*inputs)
        #     # 验证捕获的警告数量是否为 2
        #     self.assertEqual(len(wa), 2)
        #     # 断言最后一条警告信息是否符合预期的回退正则表达式
        #     self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    # 测试函数：验证在维度大小为 0 时是否会触发 RuntimeError
    def test_fallback_zero_dim(self):
        # 获取测试用的函数对象
        op = torch._test_functorch_fallback
        # 创建随机张量 x 和 y
        x = torch.randn(11)
        y = torch.randn(11)
        # 调用辅助函数来验证是否使用了 vmap 回退路径
        self._assert_uses_vmap_fallback((op,), (x, y))

        # 定义维度 B0 和 B1 的值
        B0, B1 = 0, 3
        # 创建随机张量 x 和 y，其中 x 的第一个维度为 B0
        x = torch.randn(B0, 11)
        y = torch.randn(11)
        # 定义错误消息
        msg = "The fallback path does not support vmap over dims of size 0"

        # 验证是否会触发预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        # 创建随机张量 x 和 y，其中 x 的第一个维度为 B0，第二个维度为 B1
        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        # 验证是否会触发预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)
    # 定义测试函数：测试使用 _test_functorch_fallback 在不同情况下的行为
    def test_fallback_warning(self):
        # 获取 _test_functorch_fallback 函数对象
        op = torch._test_functorch_fallback

        # 创建随机张量 x 和 y
        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        # 断言 op 函数是否会使用 vmap 的回退功能
        self._assert_uses_vmap_fallback((op,), (x, y))

        # 更换张量 x 和 y 的维度顺序后，使用 vmap 执行 op 函数
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # 嵌套使用 vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # 大批量数据测试 (总共 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    # TODO: No clue what is wrong here.
    # 跳过测试函数，暂未实现
    @unittest.skip
    def test_fallback_masked_fill(self):
        # 注意事项：有一天我们会为 masked_fill 实现批处理规则
        # 如果/当我们这样做时，应该使用另一个操作符来测试回退路径，以避免代码老化。
        def run_test(batch_size):
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 13)

            # 断言操作函数 torch.index_add 是否会使用 vmap 的回退功能
            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            # 使用 vmap 执行 torch.index_add 函数
            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 1, 13))
            self.assertEqual(result, expected)

        run_test(batch_size=5)
        run_test(batch_size=1237)

    # 测试多返回值情况下的回退功能
    def test_fallback_multiple_returns(self):
        # 注意事项：有一天我们会为 torch.var_mean 实现批处理规则
        # 如果/当我们这样做时，应该使用另一个操作符来测试回退路径，以避免代码老化。
        B0, B1, B2 = 2, 3, 1237
        tensor = torch.randn(B0, 10)

        # 断言操作函数 torch.var_mean 是否会使用 vmap 的回退功能
        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # 使用 vmap 执行 torch.var_mean 函数
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)

        # 嵌套使用 vmap
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)

        # 大批量数据测试，嵌套使用 vmap
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)
    def test_inplace_fallback_unary(self):
        # 测试在没有额外张量参数的情况下，对不可变方法的原地回退。
        # 这是回退的最简单情况。
        # 注意：将来我们可能会为 acos_ 实现批处理规则。
        # 如果/当我们这样做时，应该将此测试替换为测试另一个操作符上的回退路径，以避免代码腐化。
        
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        x = torch.randn(B0, 5)
        # 断言是否使用了 vmap 的回退
        self._assert_uses_vmap_fallback((op,), (x,))

        # 单个 vmap
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        # 断言结果是 x 本身
        self.assertTrue(result is x)
        # 断言结果等于 x_orig 的 acos 函数结果
        self.assertEqual(result, x_orig.acos())

        # 单个 vmap + 不同的 out_dim 会产生一个视图
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        # 断言结果的基础是 x
        self.assertTrue(result._base is x)
        # 断言结果等于 x_orig 的转置后的 acos 结果
        self.assertEqual(result, x_orig.t().acos())

        # 嵌套 vmap
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        # 断言结果是 x 本身
        self.assertTrue(result is x)
        # 断言结果等于 x_orig 的 acos 结果
        self.assertEqual(result, x_orig.acos())

        # 嵌套 vmap，大批量大小
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        # 断言结果是 x 本身
        self.assertTrue(result is x)
        # 断言结果等于 x_orig 的 acos 结果
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # 注意：将来我们可能会为 atan2_ 实现批处理规则。
        # 如果/当我们这样做时，应该将此测试替换为测试另一个操作符上的回退路径，以避免代码腐化。
        
        op = Tensor.atan2_
        outplace_op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        # 断言是否使用了 vmap 的回退
        self._assert_uses_vmap_fallback((op,), (x, y))

        # 单个 vmap
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        # 断言 x 等于 outplace_op(x_orig, y.movedim(0, 2)) 的结果
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # 嵌套 vmap
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        # 断言 x 等于 outplace_op(x_orig, y.movedim([0, 1], [2, 0])) 的结果
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # 大批量大小 (总共 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        vmap(vmap(vmap(op)))(x, y)
        # 断言 x 等于 outplace_op(x_orig, y.view(B0, B1, B2, 1)) 的结果
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    # ("Fallback isInplaceVmapCompatible check is broken")
    @unittest.expectedFailure
    # 测试函数：test_inplace_fallback_nary_different_levels
    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # 如果有一天我们实现了 atan2_ 的批处理规则
        # If/when we do, this test should be replaced to test the fallback
        # 那么应该替换这个测试来测试另一个操作符的回退路径，以避免过时
        # path on another operator to avoid bitrot.
        # 以避免其它运算符路径的退化
        op = Tensor.atan2_
        # 设置操作符 op 为 Tensor 类的 atan2_ 方法
        outplace_op = torch.atan2
        # 设置 outplace_op 为 torch 的 atan2 方法
        B0, B1 = 2, 3
        # 定义 B0 和 B1 为 2 和 3

        x = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 x
        y = torch.rand(7)
        # 创建 7 个随机数的张量 y
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))
        # 使用 _assert_uses_vmap_fallback 方法验证 vmap 回退

        # op(left, right): All of the levels in right are found in left
        # op(left, right): 在 right 中的所有级别都在 left 中找到
        x_orig = torch.rand(B0, 7)
        # 创建原始张量 x_orig，形状为 B0 行 7 列的随机张量
        x = x_orig.clone()
        # 克隆 x_orig 得到 x
        y = torch.rand(7)
        # 创建 7 个随机数的张量 y
        vmap(op, in_dims=(0, None))(x, y)
        # 使用 vmap 对 op 进行映射，指定输入维度为 (0, None)，应用于 (x, y)
        self.assertEqual(x, outplace_op(x_orig, y))
        # 断言 x 等于 outplace_op(x_orig, y)

        x_orig = torch.rand(B0, B1, 7)
        # 创建原始张量 x_orig，形状为 B0 行 B1 列 7 深度的随机张量
        x = x_orig.clone()
        # 克隆 x_orig 得到 x
        y = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 y
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        # 嵌套使用 vmap 对 op 进行映射，指定输入维度为 (0, None)，应用于 (x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))
        # 断言 x 等于 outplace_op(x_orig, y.view(B0, 1, 7))

        # op(left, right): Some of the levels in right are not found in left
        # op(left, right): right 中的某些级别在 left 中找不到
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
        # 设置错误信息字符串

        x = torch.rand(7)
        # 创建 7 个随机数的张量 x
        y = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            # 断言引发 RuntimeError 异常，异常信息为 msg
            vmap(op, in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        # 创建 B1 行 7 列的随机张量 x
        y = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            # 断言引发 RuntimeError 异常，异常信息为 msg
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        # 创建 B1 行 7 列的随机张量 x
        y = torch.rand(7, B0)
        # 创建 7 行 B0 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            # 断言引发 RuntimeError 异常，异常信息为 msg
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)

        x = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 x
        y = torch.rand(B0, B1, 7)
        # 创建 B0 行 B1 列 7 深度的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            # 断言引发 RuntimeError 异常，异常信息为 msg
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    # 测试函数：test_backward_unsupported_interaction
    def test_backward_unsupported_interaction(self):
        x = torch.randn(3, requires_grad=True)
        # 创建形状为 3 的随机张量 x，并设置 requires_grad 为 True
        y = torch.randn(5)
        # 创建 5 个随机数的张量 y
        grad = torch.randn_like(x)
        # 创建与 x 相同形状的随机张量 grad
        err_msg = r"backward\(\) called inside a functorch transform"
        # 设置错误信息字符串

        def backward_on_vmapped_tensor(x):
            # 定义在 vmapped 张量上进行反向传播的函数 backward_on_vmapped_tensor
            x.sum().backward()

        # FIXME
        # FIXME 标记，表示需要修复的问题
        return self.skipTest(
            "error: element 0 of tensors does not require grad and does not have a grad_fn"
        )
        # 跳过测试，并返回特定的错误信息字符串

        with self.assertRaisesRegex(RuntimeError, err_msg):
            # 断言引发 RuntimeError 异常，异常信息为 err_msg
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            # 定义带有 vmapped 梯度的反向传播函数 backward_with_vmapped_grad
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            # 断言引发 RuntimeError 异常，异常信息为 err_msg
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            # 定义完全无关的反向传播函数 completely_unrelated_backward
            x.sum().backward()
            return y

        with self.assertRaisesRegex(RuntimeError, err_msg):
            # 断言引发 RuntimeError 异常，异常信息为 err_msg
            vmap(completely_unrelated_backward)(y)

    @unittest.expectedFailure
    def test_grad_unsupported_interaction(self):
        # 创建一个需要梯度的随机张量
        input_tensor = torch.randn(3, requires_grad=True)
        # 定义错误消息，用于断言异常时的比较
        err_msg = "autograd.grad.* called inside torch.vmap"

        # 创建另一个需要梯度的随机张量
        captured = torch.randn(3, requires_grad=True)

        # 定义一个函数，其输出用于梯度计算
        def output_to_grad_is_vmapped(input_tensor):
            # 计算输出作为梯度计算的一部分
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        # 使用断言检查在 vmap 内部调用 autograd.grad.* 是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        # 计算另一个输出作为梯度计算的一部分
        output = (input_tensor**2).sum()

        # 定义另一个函数，其输入用于梯度计算
        def input_to_grad_is_vmapped(input_tensor):
            return torch.autograd.grad([output], [input_tensor])[0]

        # 使用断言检查在 vmap 内部调用 autograd.grad.* 是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        # 定义张量的数量 N
        N = 3
        # 创建一个需要梯度的随机张量
        x = torch.randn(N, requires_grad=True)
        # 创建一个不需要梯度的随机张量
        y = torch.randn(N)

        # 定义一个函数，用于计算向量-雅可比积
        def vjp_mul(v):
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        # 创建单位矩阵作为批处理的 v
        batched_v = torch.eye(N)
        # 使用 vmap 对 vjp_mul 函数进行批处理映射，计算雅可比矩阵
        jacobian = vmap(vjp_mul)(batched_v)
        # 使用断言检查计算得到的雅可比矩阵是否与对角矩阵 y 相等
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        # 创建一个随机张量 x
        x = torch.randn(3)
        # 创建一个随机张量 y
        y = torch.randn(2, 3)
        # 使用 functools.partial 对 torch.mul 函数进行部分应用，生成一个结果张量
        result = vmap(functools.partial(torch.mul, x))(y)
        # 使用断言检查结果张量是否等于 x 与 y 的逐元素乘积
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        # 创建一个随机张量 tensor
        tensor = torch.randn(2, 3)
        # 创建一个线性模型，输入输出维度为 3
        model = torch.nn.Linear(3, 3, bias=False)
        # 使用 vmap 对模型进行批处理映射，计算输入 tensor 对应的模型输出
        result = vmap(model)(tensor)
        # 使用断言检查模型输出是否等于输入 tensor 经过模型后的结果
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        # 定义 B0 的值为 7
        B0 = 7
        # 创建一个需要梯度的随机张量 x，其形状为 (2, 3, 4, 5)
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        # 创建一个随机权重张量 weight，其形状为 (3, 3, 1, 1)
        weight = torch.randn(3, 3, 1, 1)
        # 创建一个随机张量 v，其形状为 (B0, 2, 3, 4, 5)
        v = torch.randn(B0, 2, 3, 4, 5)

        # 定义一个函数，用于获取向量-雅可比积
        def get_vjp(v):
            # 使用函数式 API 计算卷积结果
            result = torch.nn.functional.conv2d(x, weight)
            # 计算 result 对 x 的梯度，使用输入 v 作为 grad_outputs
            (grad_x,) = torch.autograd.grad(result, x, v)
            return grad_x

        # 运行 vmap(get_vjp)(v)，这里不应出现错误
        # 卷积的反向传播公式返回一个未定义的 Tensor，因为原始偏置项不存在。
        #
        # 在将来，我们可能会为卷积的反向传播添加一个批处理规则。
        # 当这发生时，我们应修改这个测试以使用不同的操作（和/或创建并使用一个虚拟运算符）来避免代码腐化。
        self._assert_uses_vmap_fallback([get_vjp], [v])
    # 定义测试函数，用于测试 reshape_dim_into 函数的不同情况
    def test_reshape_dim_into(self):
        # 创建一个形状为 (2, 3, 5, 7) 的随机张量 x
        x = torch.randn(2, 3, 5, 7)

        # 调用 reshape_dim_into 函数，将第 0 维展开为 0 维，期望结果与 x.reshape(6, 5, 7) 相等
        y = reshape_dim_into(0, 0, x)
        self.assertEqual(y, x.reshape(6, 5, 7))

        # 将第 0 维展开为 1 维，期望结果与 x.movedim(0, 1).reshape(3, 2 * 5, 7) 相等
        y = reshape_dim_into(0, 1, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        # 将第 0 维展开为 2 维，期望结果与 x.movedim(0, 2).reshape(3, 5, 2 * 7) 相等
        y = reshape_dim_into(0, 2, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        # 将第 1 维展开为 2 维，期望结果与 x.movedim(1, 2).reshape(2, 5, 3 * 7) 相等
        y = reshape_dim_into(1, 2, x)
        self.assertEqual(y, x.movedim(1, 2).reshape(2, 5, 3 * 7))

        # 将第 0 维展开为倒数第 2 维，期望结果与 x.movedim(0, 1).reshape(3, 2 * 5, 7) 相等
        y = reshape_dim_into(0, -2, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        # 将第 0 维展开为倒数第 1 维，期望结果与 x.movedim(0, 2).reshape(3, 5, 2 * 7) 相等
        y = reshape_dim_into(0, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        # 将倒数第 4 维展开为倒数第 1 维，期望结果与 x.movedim(0, 2).reshape(3, 5, 2 * 7) 相等
        y = reshape_dim_into(-4, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

    # 定义测试函数，用于测试 reshape_dim_outof 函数的不同情况
    def test_reshape_dim_outof(self):
        # 创建一个形状为 (12, 12, 12) 的随机张量 x，并将其按维度顺序重排为 (2, 12, 12, 12)
        x = torch.randn(12, 12, 12).permute(2, 1, 0)

        # 将第 0 维展开为 2 维，期望结果与 x.reshape(2, 6, 12, 12) 相等
        y = reshape_dim_outof(0, 2, x)
        self.assertEqual(y, x.reshape(2, 6, 12, 12))

        # 将第 1 维展开为 4 维，期望结果与 x.reshape(12, 4, 3, 12) 相等
        y = reshape_dim_outof(1, 4, x)
        self.assertEqual(y, x.reshape(12, 4, 3, 12))

        # 将第 2 维展开为 6 维，期望结果与 x.reshape(12, 12, 6, 2) 相等
        y = reshape_dim_outof(2, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        # 将倒数第 1 维展开为 6 维，期望结果与 x.reshape(12, 12, 6, 2) 相等
        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        # 测试特殊情况：当第 0 维的大小为 0 时，预期输出的形状为 (12, 12, 6, 0)
        x = torch.randn(12, 12, 0)
        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y.shape, torch.Size((12, 12, 6, 0)))

    # 定义测试函数，用于测试不需要处理未批处理输入的情况
    def test_batch_rule_does_not_need_to_handle_no_batched_input(self):
        # 定义一个简单函数 f，计算 y 与 torch.ones(2) 的点乘后加上 x 的结果
        def f(x, y):
            res = torch.dot(y, torch.ones(2))
            return x + res

        # 创建形状为 (7, 5) 和 (3, 2) 的随机张量 x 和 y
        x = torch.randn(7, 5)
        y = torch.randn(3, 2)

        # 使用 vmap 函数对函数 f 进行批处理映射，in_dims=(0, None) 表示对 x 的第 0 维进行批处理映射，对 y 不做映射
        out = vmap(vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        # 预期输出为 torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        expected = torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        self.assertEqual(out, expected)

    # 定义测试函数，用于测试在 Python 调度程序下的分解
    def test_decomposition_under_python_dispatcher(self):
        # 如果调用 vmap 后仍然使用 Python 调度程序，则会触发错误
        # 此处测试确保 FuncTorchBatchedDecomposition 注册的分解在 Python 调度程序下得到尊重
        t = torch.ones(3, 3) * 5
        with DisableVmapFallback():
            with torch._dispatch.python.enable_python_dispatcher():
                o = torch.vmap(torch.square)(t)
        self.assertEqual(o, torch.square(t))
    # 定义一个测试方法，用于验证在指定设备上的 vmap 和 autocast 的使用
    def _test_vmap_autocast(self, device):
        # 根据设备类型选择自动混合精度的数据类型
        if torch.device(device).type == "cpu":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        # 创建四个在指定设备上的随机张量
        a_float32 = torch.rand(4, 2, 3, device=device)
        b_float32 = torch.rand(4, 3, 2, device=device)
        c_float32 = torch.rand(4, 2, 2, device=device)
        d_float32 = torch.rand(4, 3, 2, device=device)

        # Case 1, 在 vmapped 函数内部使用 autocast
        def func1(x, y, z, w):
            # 在上下文中开启自动混合精度，使用指定的数据类型和设备类型
            with torch.autocast(dtype=amp_dtype, device_type=device):
                # 执行矩阵乘法运算，并验证结果的数据类型是否为指定的混合精度类型
                e_float16 = torch.matmul(x, y)
                assert e_float16.dtype == amp_dtype, e_float16.dtype
                f_float16 = torch.matmul(z, e_float16)
                assert f_float16.dtype == amp_dtype, f_float16.dtype
            # 将最终结果与输入的 w 进行乘法运算并返回
            return torch.matmul(w, f_float16.float())

        # 计算预期的输出结果并进行验证
        expected = func1(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func1)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 2, 在 vmapped 函数外使用 autocast 装饰器
        @torch.autocast(dtype=amp_dtype, device_type=device)
        def func2(x, y, z, w):
            # 执行矩阵乘法运算，并验证结果的数据类型是否为指定的混合精度类型
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            # 将最终结果与输入的 w 进行乘法运算并返回
            return torch.matmul(w, f_float16)

        # 计算预期的输出结果并进行验证
        expected = func2(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func2)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 3, 在 vmapped 函数外部使用 autocast
        def func3(x, y, z, w):
            # 执行矩阵乘法运算，并验证结果的数据类型是否为指定的混合精度类型
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            # 将最终结果与输入的 w 进行乘法运算并返回
            return torch.matmul(w, f_float16)

        # 在上下文中开启自动混合精度，使用指定的数据类型和设备类型
        with torch.autocast(dtype=amp_dtype, device_type=device):
            # 计算预期的输出结果并进行验证
            expected = func3(a_float32, b_float32, c_float32, d_float32)
            out = vmap(func3)(a_float32, b_float32, c_float32, d_float32)

        assert expected.allclose(out)

    # 标记为跳过测试，因为 vmap 和 autocast 在 CPU 上不能正常工作
    @unittest.skip("Somehow, vmap and autocast do not work on CPU")
    def test_vmap_autocast_cpu(self):
        self._test_vmap_autocast("cpu")

    # 标记为跳过测试，如果 CUDA 不可用
    @skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_vmap_autocast_cuda(self):
        self._test_vmap_autocast("cuda")
    # 定义一个测试函数，用于测试 restore_vmap 函数处理 pytree 输入和输出的情况
    def test_restore_vmap_pytree_input_output(self):
        
        # 定义一个简单的函数 f，接受两个参数 x 和 y，计算并返回一个包含两个键值对的字典
        def f(x, y):
            output0 = x[0] + x[1]  # 计算 x 的前两个元素的和
            output1 = y  # 直接将 y 赋给 output1
            return {"a": output0, "b": output1}

        B = 2  # 定义 B 的值为 2
        x0 = torch.randn(B, 3)  # 生成一个形状为 (B, 3) 的随机张量 x0
        x1 = torch.randn(B)  # 生成一个形状为 (B,) 的随机张量 x1
        y = torch.randn(4, B)  # 生成一个形状为 (4, B) 的随机张量 y

        # 调用 restore_vmap 函数处理函数 f，传入的参数为 ((0, 0), 1)，B，"error"，并使用输入 ((x0, x1), y)
        out, out_dims = restore_vmap(f, ((0, 0), 1), B, "error")((x0, x1), y)
        
        # 使用 vmap 函数处理函数 f，指定输入维度 in_dims=((0, 0), 1)，输出维度 out_dims={"a": 0, "b": 1}，输入为 ((x0, x1), y)
        expected = vmap(f, in_dims=((0, 0), 1), out_dims={"a": 0, "b": 1})((x0, x1), y)
        
        # 断言 out 和 expected 相等
        self.assertEqual(out, expected)
        
        # 断言 out_dims 等于 {"a": 0, "b": 1}
        self.assertEqual(out_dims, {"a": 0, "b": 1})

    # 定义一个测试函数，用于测试 restore_vmap 处理没有 vmapped 输入的情况
    def test_restore_vmap_no_vmapped_inputs(self):
        
        # 定义一个简单的函数 f，接受三个参数 x, y, z，直接返回这三个参数组成的元组
        def f(x, y, z):
            return x, y * z, z

        B = 2  # 定义 B 的值为 2
        
        # 定义 x 为一个形状为 (3,) 的随机张量
        x = torch.randn(3)
        
        # 定义 y 为一个形状为 (4,) 的随机张量
        y = torch.randn(4)
        
        # 定义 z 为标量值 5
        z = 5
        
        # 调用 restore_vmap 处理函数 f，传入的参数为 (None, None, None)，B，"error"，并使用输入 (x, y, z)
        out, out_dims = restore_vmap(f, (None, None, None), B, "error")(x, y, z)
        
        # 断言 out 和 f(x, y, z) 相等
        self.assertEqual(out, f(x, y, z))
        
        # 断言 out_dims 等于 (None, None, None)
        self.assertEqual(out_dims, (None, None, None))

    # 定义一个测试函数，用于测试 restore_vmap 处理未展开输出的情况
    def test_restore_vmap_unexpanded_outputs(self):
        
        # 定义一个简单的函数 f，接受两个参数 x 和 y，返回一个元组，包含 y 的三倍、y 的总和以及 None
        def f(x, y):
            # Mix of tensor and non-tensor outputs
            return 3 * y, y.sum(), None

        B = 2  # 定义 B 的值为 2
        
        # 定义 x 为一个形状为 (B, 3) 的随机张量
        x = torch.randn(B, 3)
        
        # 定义 y 为一个形状为 (4,) 的随机张量
        y = torch.randn(4)
        
        # 调用 restore_vmap 处理函数 f，传入的参数为 (0, None)，B，"error"，并使用输入 (x, y)
        out, out_dims = restore_vmap(f, (0, None), B, "error")(x, y)
        
        # 断言 out 和 f(None, y) 相等
        self.assertEqual(out, f(None, y))
        
        # 断言 out_dims 等于 (None, None, None)
        self.assertEqual(out_dims, (None, None, None))

    # 定义一个测试函数，用于测试数据属性的行为
    def test_data_attribute(self):
        
        # 定义一个函数 foo，接受一个参数 x，尝试访问 x 的 data 属性并返回 x
        def foo(x):
            y = x.data  # 尝试访问 x 的 data 属性
            return x

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证异常消息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "accessing `data` under vmap transform"
        ):
            # 使用 vmap 处理函数 foo，传入一个形状为 (3, 3) 的随机张量
            torch.func.vmap(foo)(torch.randn(3, 3))

        # 重新定义函数 foo，接受一个参数 x，尝试修改 x 的 data 属性并返回 x
        def foo(x):
            x.data = torch.ones(3, 3)  # 尝试直接用 torch.ones(3, 3) 修改 x 的 data 属性
            return x

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证异常消息是否包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "mutating directly with `.data` under vmap"
        ):
            # 使用 vmap 处理函数 foo，传入一个形状为 (3, 3) 的随机张量
            torch.func.vmap(foo)(torch.randn(3, 3))
# 将输入按照指定维度切片，返回切片后的结果元组
def slice_inputs(inputs, bdims, i):
    result = []
    # 遍历输入和对应的切片维度
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            # 如果切片维度为None，则直接添加原始输入
            result.append(inp)
        else:
            # 否则，使用select方法在指定维度上进行切片，添加切片后的结果
            result.append(inp.select(bdim, i))
    # 返回切片后的结果元组
    return tuple(result)


# 对操作op进行vmap映射，处理多维度输入和输出
def reference_vmap(op, inputs, in_dims=0, out_dims=0, return_nt=False):
    # 如果in_dims是整数，则转换为相同长度的元组，每个元素为该整数
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    # 计算每个输入张量在指定维度上的大小，并检查它们是否相等
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    assert all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes)
    # 取第一个输入张量在指定维度上的大小作为bdim_size
    bdim_size = bdim_sizes[0]
    # 使用op对每个切片后的输入进行操作，得到结果元组
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    # 断言结果元组长度大于0
    assert len(results) > 0
    # 检查操作是否只返回单个张量
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        # 如果操作只返回单个张量，断言所有结果都是torch.Tensor张量
        assert all(isinstance(result, torch.Tensor) for result in results)
        # 如果out_dims是整数，则转换为长度为1的元组
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        # 如果return_nt为True，将结果转换为嵌套张量
        if return_nt:
            return torch.nested.nested_tensor(list(results))
        else:
            # 否则，沿着指定维度将结果堆叠成张量
            return torch.stack(results, dim=out_dims[0])

    # 断言所有结果都是元组
    assert all(isinstance(result, tuple) for result in results)
    # 计算每个结果元组中元素的数量，并断言它们相等
    num_returns = len(results[0])
    assert all(len(result) == num_returns for result in results)
    # 如果out_dims是整数，则转换为长度为num_returns的元组
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    # 如果return_nt为True，将每个结果分片转换为嵌套张量
    if return_nt:
        return tuple(
            torch.nested.nested_tensor(list(result_shards))
            for result_shards in zip(*results)
        )
    else:
        # 否则，将每个结果分片沿着对应的out_dims维度进行堆叠
        return tuple(
            torch.stack(result_shards, out_dim)
            for result_shards, out_dim in zip(zip(*results), out_dims)
        )


# 定义一个静态方法类，用于生成不同类型的张量
class TensorFactory:
    @staticmethod
    def rand(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device="cpu", dtype=torch.float):
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype) + 1


# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.


# 对vmap函数进行测试，将其结果与参考的vmap实现进行比较
def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    # 使用vmap函数对操作op进行映射，得到结果result
    result = vmap(op, in_dims, out_dims)(*inputs)
    # 检查结果是否包含嵌套张量
    are_nested = [t.is_nested for t in pytree.tree_leaves(result)]
    # 使用reference_vmap函数生成参考结果reference_result
    reference_result = reference_vmap(
        op, inputs, in_dims, out_dims, return_nt=any(are_nested)
    )
    # 使用self.assertEqual断言result与reference_result相等
    self.assertEqual(result, reference_result)
    # 检查操作是否只返回单个张量
    op_has_single_return = not isinstance(result, tuple)
    # 如果需要检查视图性质
    if check_view:
        # 根据操作是否只返回单个结果，将结果封装成元组或单个值
        result_as_tuple = (result,) if op_has_single_return else result
        # 遍历结果元组中的每个输出
        for output in result_as_tuple:
            # 获取输入列表中第一个输入的基本张量（如果存在）
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            # 断言输出的基本张量与第一个输入的基本张量相同
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

    # 如果不需要检查梯度传播，直接返回
    if not check_propagates_grad:
        return

    # 假设输入的第一个元素是一个浮点数张量。检查 vmap 操作是否正确传播了 requires_grad 标志到第零输出。
    # 一些 vmap 操作符实现时假定它们在自动求导下是复合的。
    # 如果操作符改变并不再在自动求导下是复合的，下面的检查应该失败。
    
    # 复制输入列表以避免就地修改
    inputs_clone = list(inputs)
    # 对第一个输入张量进行克隆，并设置 requires_grad 标志
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    # 对操作进行 vmap 包装，并使用克隆后的输入列表进行调用
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    # 根据操作是否只返回单个结果，将结果封装成元组或单个值
    result_as_tuple = (result,) if op_has_single_return else result
    # 断言第一个输出张量需要求导
    self.assertTrue(result[0].requires_grad)
# 检查给定函数对象是否允许使用 vmap 回退功能
def should_allow_vmap_fallback_usage(fn):
    return getattr(fn, "_allow_vmap_fallback_usage", False)


# 设置函数对象允许使用 vmap 回退功能的标志为 True，并返回该函数对象
def allowVmapFallbackUsage(fn):
    fn._allow_vmap_fallback_usage = True
    return fn


# TestVmapBase 的所有测试确保不会调用慢速的 vmap 回退路径。
# 这样做是为了能够逐步为运算符添加批处理规则，以替换相应运算符的慢速 vmap 回退路径。
# 要跳过此检查，请使用 allowVmapFallbackUsage 装饰器。
#
# 注意：请不要直接向 TestVmapBase 添加测试，除非希望它们在 TestVmapBase 的每个子类上运行。
# 可以将它们添加到例如 TestVmapOperators 中。
#
# 注意：TestVmapBase 是一个嵌套类。这样做可以防止测试运行器捕捉并运行它。
class Namespace:
    # 定义一个测试类 TestVmapBase，继承自 TestCase
    class TestVmapBase(TestCase):
        # 初始化方法，接受一个 method_name 参数，默认为 "runTest"
        def __init__(self, method_name="runTest"):
            # 调用父类的初始化方法
            super().__init__(method_name)
    
            # 获取当前实例的测试方法，如果不存在则返回 None
            test_method = getattr(self, method_name, None)
            # 如果测试方法不存在，直接返回
            if test_method is None:
                return
    
            # 检查是否允许使用 vmap 回退路径
            if not should_allow_vmap_fallback_usage(test_method):
                # 如果不允许，则替换当前测试方法为经过 vmap 回退检查包装后的方法
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )
    
        # 定义一个内部方法，用于将给定方法包装成带有 vmap 回退检查的方法
        def _wrap_method_with_vmap_fallback_check(self, method):
            # 定义一个装饰器函数，使用 functools.wraps 方法包装原始方法
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                # 捕获所有警告
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    # 使用 EnableVmapFallbackWarnings 上下文管理器，执行原始方法
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    # 注释部分，用于检查捕获的警告消息是否符合预期
                    # for captured_warning in wa:
                    #     self.assertNotRegex(str(captured_warning.message), FALLBACK_REGEX, msg)
    
            return types.MethodType(wrapper, self)
    
        # 标记该方法允许使用 vmap 回退路径
        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # 未来可以为 torch.var_mean 实现一个批处理规则
            # 当这一规则实现后，请修改示例以使用一个未实现批处理规则的操作符
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))
    
        # 标记该方法预期测试失败
        @unittest.expectedFailure
        def test_vmap_fallback_check(self):
            # 定义一个内部方法，使用 vmap 回退检查包装
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                pass
    
            # 未来可以为 torch.var_mean 实现一个批处理规则
            # 当这一规则实现后，请修改示例以使用一个未实现批处理规则的操作符
            op_using_fallback = torch.var_mean
    
            # 定义一个内部方法，使用 vmap 回退检查包装
            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                vmap(op_using_fallback)(torch.rand(3))
    
            # 调用没有 vmap 回退的方法
            no_fallback(self)
    
            # 断言捕获到 AssertionError 异常
            with self.assertRaises(AssertionError):
                uses_fallback(self)
# 定义一个函数，用于生成测试用例，接受操作符和输入获取器作为参数，默认使用TensorFactory.randn获取输入
def _make_case(op, input_getter=TensorFactory.randn):
    return (op, input_getter)


# 使用markDynamoStrictTest标记装饰的测试类，继承自Namespace.TestVmapBase
@markDynamoStrictTest
class TestVmapOperators(Namespace.TestVmapBase):
    # 定义_vmap_test方法，用于执行_vmap_test函数
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # 定义_vmap_view_test方法，调用_vmap_test方法，同时设置check_view=True
    def _vmap_view_test(self, *args, **kwargs):
        self._vmap_test(*args, **kwargs, check_view=True)

    # 定义_test_unary方法，用于测试一元操作符，接受操作符、获取器、设备和其他参数
    def _test_unary(self, op, getter, device, *args, **kwargs):
        # 使用functools.partial创建一个部分应用的函数test，用于执行_vmap_test方法
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # 单层vmap测试，不同的in_dims / out_dims设置
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # 双层嵌套vmap测试
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    # 使用@parametrize装饰器指定测试用例，包括一元操作符和获取器的组合
    @parametrize(
        "case",
        [
            (torch.abs, TensorFactory.randn),
            (torch.acos, TensorFactory.rand),
            (torch.asin, TensorFactory.rand),
            (torch.atan, TensorFactory.rand),
            (torch.ceil, TensorFactory.randn),
            (torch.cos, TensorFactory.rand),
            (torch.cosh, TensorFactory.rand),
            (torch.digamma, TensorFactory.rand),
            (torch.exp, TensorFactory.randn),
            (torch.expm1, TensorFactory.randn),
            (torch.floor, TensorFactory.randn),
            (torch.frac, TensorFactory.randn),
            (torch.lgamma, TensorFactory.rand),
            (torch.log, TensorFactory.randp1),
            (torch.log10, TensorFactory.randp1),
            (torch.log1p, TensorFactory.randp1),
            (torch.log2, TensorFactory.randp1),
            (torch.neg, TensorFactory.randn),
            (torch.reciprocal, TensorFactory.randp1),
            (torch.relu, TensorFactory.randn),
            (torch.round, TensorFactory.randn),
            (torch.rsqrt, TensorFactory.randp1),
            (torch.sigmoid, TensorFactory.randn),
            (torch.sign, TensorFactory.randn),
            (torch.sin, TensorFactory.rand),
            (torch.sinh, TensorFactory.rand),
            (torch.sqrt, TensorFactory.rand),
            (torch.tan, TensorFactory.rand),
            (torch.tanh, TensorFactory.rand),
            (torch.trunc, TensorFactory.randn),
        ],
        name_fn=lambda x: x[0].__name__,  # 获取操作符的名称作为测试用例的名称
    )
    # 定义测试方法test_unary_pointwise，接受测试用例作为参数
    def test_unary_pointwise(self, case):
        op, getter = case
        # 调用_test_unary方法，传入操作符、获取器和设备参数
        self._test_unary(op, getter, "cpu")

        # 测试原地操作
        method = getattr(Tensor, f'{op.__name__ + "_"}')  # 获取Tensor类中对应操作的方法
        self._test_unary(method, getter, "cpu", check_propagates_grad=False)
    def test_clone(self):
        # Some basic tests

        # 测试 Tensor 对象的 clone 方法，使用 TensorFactory.randn 创建随机张量，使用 "cpu" 进行操作
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")

        # 测试 Tensor 对象的 clone 方法，设置 memory_format 为 torch.preserve_format，使用 TensorFactory.randn 创建随机张量，使用 "cpu" 进行操作
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )

        # 测试 Tensor 对象的 clone 方法，设置 memory_format 为 torch.contiguous_format，使用 TensorFactory.randn 创建随机张量，使用 "cpu" 进行操作
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

        # Test that the per-examples are contiguous when using torch.contiguous_format

        # 定义一个函数 clone_contiguous，使用 memory_format=torch.contiguous_format 进行克隆操作
        def clone_contiguous(x):
            return x.clone(memory_format=torch.contiguous_format)

        # 设置维度 B0 和 B1 的值分别为 3 和 5
        B0, B1 = 3, 5

        # 创建一个大小为 (2, B0, 7) 的张量 x，并使用 vmap 函数对 clone_contiguous 函数进行映射
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)

        # 断言 y 在移动维度 1 到 0 后是连续的张量
        self.assertTrue(y.movedim(1, 0).is_contiguous())

        # 断言 y 的每个子张量 y[:, 0, :] 都是连续的张量
        self.assertTrue(y[:, 0, :].is_contiguous())

        # 创建一个大小为 (2, B0, 7, B1) 的张量 x，并使用 vmap 函数对 vmap(clone_contiguous, in_dims=2) 函数进行映射
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)

        # 断言 y 是连续的张量
        self.assertTrue(y.is_contiguous())

        # 断言 y 的子张量 y[0][0] 是连续的张量
        self.assertTrue(y[0][0].is_contiguous())

        # 创建一个消息，指出 memory_format 仅支持 torch.preserve_format 或 torch.contiguous_format
        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"

        # 使用 self.assertRaisesRegex 断言运行时错误，并匹配预期的消息
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))

        # 使用 self.assertRaisesRegex 断言运行时错误，并匹配预期的消息
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    def test_weird_matmul_case(self):
        # Check that this doesn't crash.
        # https://github.com/pytorch/functorch/issues/417

        # 创建一个大小为 (5, 2, 2, 2) 的张量 x 和一个大小为 (5, 7, 2) 的张量 y
        x = torch.randn(5, 2, 2, 2)
        y = torch.randn(5, 7, 2)

        # 使用 vmap 函数对 torch.matmul 函数进行双重映射，其中 in_dims=(None, 0)
        vmap(vmap(torch.matmul, in_dims=(None, 0)))(x, y)

    @parametrize(
        "case",
        (
            (torch.clamp_min_, TensorFactory.randn),
            (torch.clamp_max_, TensorFactory.randn),
        ),
        name_fn=lambda x: x[0].__name__,
    )
    # 定义一个测试方法，用于测试 clamp 的不同变体
    def test_clamp_inplace_variant(self, case):
        # 获取测试函数的引用
        test = self._vmap_test

        # 定义一个内部函数，用于从 getter 中获取数字
        def get_number(getter):
            return getter([]).item()

        # 解包传入的 case 参数
        op, getter = case

        # 设定设备类型为 CPU
        device = "cpu"
        B0, B1 = 7, 11

        # 单层 vmap 测试: op(Tensor, Tensor)
        # 测试1
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            check_propagates_grad=False,
        )
        # 测试2
        test(
            op,
            (getter([B0], device), getter([B0], device)),
            check_propagates_grad=False,
        )
        # 测试3
        test(
            op,
            (getter([2, B0, 3], device), getter([2, B0, 3], device)),
            in_dims=(1, 1),
            check_propagates_grad=False,
        )
        # 测试4
        test(
            op,
            (getter([B0, 2, 3], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
            check_propagates_grad=False,
        )
        # 测试5
        test(
            op,
            (getter([B0, 2, 3], device), getter([1, 1], device)),
            in_dims=(0, None),
            check_propagates_grad=False,
        )
        # 测试6
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            in_dims=(0, 0),
            check_propagates_grad=False,
        )

        # 嵌套 vmap 测试: op(Tensor, Tensor)
        test(
            vmap(op),
            (getter([B0, B1, 2, 3], device), getter([B0, B1, 1, 3], device)),
            check_propagates_grad=False,
        )

        # Python 数字重载测试: op(Tensor, Number)
        number = get_number(getter)
        # 调用内部方法 _test_unary 进行测试
        self._test_unary(
            lambda t: op(t, number), getter, device, check_propagates_grad=False
        )

    # 参数化测试用例，测试 clamp_min 和 clamp_max 函数
    @parametrize(
        "case",
        [
            subtest(_make_case(torch.clamp_min), name="clamp_min"),
            subtest(_make_case(torch.clamp_max), name="clamp_max"),
        ],
    )
    # 定义测试函数 test_clamp_variant，接受一个参数 case
    def test_clamp_variant(self, case):
        # 将 self._vmap_test 赋值给 test
        test = self._vmap_test

        # 定义内部函数 get_number，接受一个 getter 函数作为参数，返回 getter 函数返回值的第一个元素
        def get_number(getter):
            return getter([]).item()

        # 解包 case 元组，分别赋值给 op 和 getter
        op, getter = case
        # 设备设为 "cpu"
        device = "cpu"
        # 设定 B0 和 B1 的值为 7 和 11
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor) 测试
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(None, 0))

        # Nested vmap: op(Tensor, Tensor) 测试
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number) 测试
        # 获取 getter 函数返回的第一个数值
        number = get_number(getter)
        # 使用 self._test_unary 测试 op(t, number) 的函数，getter 和 device 参数传入
        self._test_unary(lambda t: op(t, number), getter, device)

    # 测试函数 test_copy_
    def test_copy_(self):
        # 创建随机张量 x 和 y，形状为 (3,)
        x = torch.randn(3)
        y = torch.randn(3)
        # 对 x 和 y 进行 vmap(Tensor.copy_) 操作
        vmap(Tensor.copy_)(x, y)
        # 断言 x 和 y 相等
        self.assertEqual(x, y)

        # 创建随机张量 x 和 y，形状分别为 (3,) 和 (3, 2)
        x = torch.randn(3)
        y = torch.randn(3, 2)
        # 对 y 和 x 进行 vmap(Tensor.copy_, in_dims=(1, None)) 操作
        vmap(Tensor.copy_, in_dims=(1, None))(y, x)
        # 断言 y 和 x 扩展后转置后相等
        self.assertEqual(y, x.expand(2, 3).t())

        # 创建随机张量 x 和 y，形状分别为 (3,) 和 (2, 3)
        x = torch.randn(3)
        y = torch.randn(2, 3)
        # 使用 self.assertRaisesRegex 断言运行时错误中包含 "inplace"
        with self.assertRaisesRegex(RuntimeError, "inplace"):
            # 对 x 和 y 进行 vmap(Tensor.copy_, in_dims=(None, 0)) 操作
            vmap(Tensor.copy_, in_dims=(None, 0))(x, y)

    # 测试函数 test_silu_backward
    def test_silu_backward(self):
        # 将 self._vmap_test 赋值给 test
        test = self._vmap_test
        # 设备设为 "cpu"
        device = "cpu"
        # getter 函数设为 TensorFactory.randp1
        getter = TensorFactory.randp1
        # B0 设为 7
        B0 = 7
        # op 设为 torch.ops.aten.silu_backward

        # Single vmap: op(Tensor, Tensor) 测试
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([], device), getter([B0], device)), in_dims=(None, 0))
        test(op, (getter([2, B0], device), getter([2], device)), in_dims=(1, None))

    # 如果满足指定条件，则跳过测试
    @skipIf(
        TEST_WITH_TORCHDYNAMO
        and os.getenv("BUILD_ENVIRONMENT", "") == "linux-focal-py3.8-clang10",
        "Segfaults with dynamo on focal, see https://github.com/pytorch/pytorch/issues/107173",
    )
    @parametrize(
        "case",
        [
            subtest(_make_case(torch.add), name="add"),  # 参数化测试，使用 torch.add 函数生成测试用例，命名为 "add"
            subtest(_make_case(lambda x, y: x + y), name="add_dunder"),  # 参数化测试，使用 lambda 函数生成测试用例，实现加法，命名为 "add_dunder"
            subtest(_make_case(torch.sub), name="sub"),  # 参数化测试，使用 torch.sub 函数生成测试用例，命名为 "sub"
            subtest(_make_case(lambda x, y: x - y), name="sub_dunder"),  # 参数化测试，使用 lambda 函数生成测试用例，实现减法，命名为 "sub_dunder"
            subtest(_make_case(torch.mul), name="mul"),  # 参数化测试，使用 torch.mul 函数生成测试用例，命名为 "mul"
            subtest(_make_case(lambda x, y: x * y), name="mul_dunder"),  # 参数化测试，使用 lambda 函数生成测试用例，实现乘法，命名为 "mul_dunder"
            subtest(
                _make_case(torch.div, input_getter=TensorFactory.randp1), name="div"
            ),  # 参数化测试，使用 torch.div 函数生成测试用例，使用自定义输入生成器 TensorFactory.randp1，命名为 "div"
            subtest(
                _make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
                name="div_dunder",
            ),  # 参数化测试，使用 lambda 函数生成测试用例，实现除法，使用自定义输入生成器 TensorFactory.randp1，命名为 "div_dunder"
            subtest(
                _make_case(torch.pow, input_getter=TensorFactory.randp1), name="pow"
            ),  # 参数化测试，使用 torch.pow 函数生成测试用例，使用自定义输入生成器 TensorFactory.randp1，命名为 "pow"
            subtest(
                _make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
                name="pow_dunder",
            ),  # 参数化测试，使用 lambda 函数生成测试用例，实现幂运算，使用自定义输入生成器 TensorFactory.randp1，命名为 "pow_dunder"
        ],
    )
    # 定义测试函数 test_arithmetic，用于测试运算操作的各种情况
    def test_arithmetic(self, case):
        # 将 self._vmap_test 函数赋值给 test 变量，简化测试调用
        test = self._vmap_test

        # 定义内部函数 get_number，用于从 getter 函数获取一个数值
        def get_number(getter):
            return getter([]).item()

        # 解构 case 元组，获取操作符 op 和 getter 函数
        op, getter = case
        # 设定计算设备为 "cpu"
        device = "cpu"
        # 设定两个测试用例的数值 B0 和 B1
        B0, B1 = 7, 11

        # 单一的 vmap 测试: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))

        # 嵌套的 vmap 测试: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python 数字重载测试: op(Tensor, Number) 和 op(Number, Tensor)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)
        number = get_number(getter)
        self._test_unary(lambda t: op(number, t), getter, device)

        # 类型提升测试: op(Logical Scalar Tensor, Logical Scalar Tensor)
        test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
        test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
        test(op, (getter([B0], device), getter([B0], device)))

        # 类型提升测试: op(Tensor, Logical Scalar Tensor) 和 op(Logical Scalar Tensor, Tensor)
        test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
        test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))

        # 如果 CUDA 可用，则执行下面的测试
        if not torch.cuda.is_available():
            return

        # TODO(rzou): 修复以下部分
        # # 测试跨设备标量
        # number = get_number(getter)
        # self._test_unary(lambda t: op(t, number), getter, device='cuda')
        # self._test_unary(lambda t: op(number, t), getter, device='cuda')
        # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    # 定义测试函数 test_nll_loss，用于测试负对数似然损失函数的各种情况
    def test_nll_loss(self):
        # 将 self._vmap_test 函数赋值给 test 变量，简化测试调用
        test = self._vmap_test
        # 将 PyTorch 的负对数似然损失函数 F.nll_loss 赋值给 op 变量
        op = F.nll_loss
        # 定义批次大小 B
        B = 3

        # 生成随机张量 y 和目标张量 t
        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (B, 2))
        # 测试不同 reduction 模式下的负对数似然损失函数
        test(op, (y, t))
        test(functools.partial(op, reduction="sum"), (y, t))
        test(functools.partial(op, reduction="none"), (y, t))

        # 生成随机张量 y 和部分随机张量 t
        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (2,))
        # 测试不同 in_dims 模式下的负对数似然损失函数
        test(op, (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="sum"), (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="none"), (y, t), in_dims=(0, None))
    # 定义一个测试函数，用于测试自适应平均池化操作
    def test_adaptive_avg_pool2d(self):
        # 将 self._vmap_test 函数赋给 test 变量
        test = self._vmap_test
        # 使用 functools.partial 创建一个自适应平均池化操作的偏函数，输出大小为 (3, 3)
        op = functools.partial(F.adaptive_avg_pool2d, output_size=(3, 3))

        # 创建一个形状为 (3, 5, 7, 9, 11) 的随机张量 x
        x = torch.randn(3, 5, 7, 9, 11)
        # 测试自适应平均池化操作 op 在 x 上的输出结果
        test(op, (x,))
        # 使用 in_dims=(1,) 测试 op 在 x 上的输出结果
        test(op, (x,), in_dims=(1,))
        # 使用 in_dims=(4,) 测试 op 在 x 上的输出结果
        test(op, (x,), in_dims=(4,))

    # 定义一个测试函数，用于测试批次矩阵乘法操作
    def test_bmm(self):
        # 将 torch.bmm 函数赋给 op 变量
        op = torch.bmm
        # 将 self._vmap_test 函数赋给 test 变量
        test = self._vmap_test
        # 设置两个维度 B0 和 B1 分别为 7 和 11

        # 测试形状不匹配时是否会引发 RuntimeError 异常
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # 测试左侧参数进行批量映射后的结果
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # 测试右侧参数进行批量映射后的结果
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # 测试左右两个参数同时进行批量映射后的结果
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )
    def test_conj(self):
        # 获取 torch.conj 函数的引用
        op = torch.conj

        # 定义函数 run_test，接受一个 dtype 参数，用于测试不同数据类型的情况
        def run_test(dtype):
            # 定义函数 get，根据给定形状生成指定数据类型的随机张量
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            # 定义两个测试的 batch 大小 B0 和 B1
            B0, B1 = 7, 11
            # 将 self._vmap_test 函数赋值给 test 变量，用于执行测试
            test = self._vmap_test

            # 单一的 vmap 测试，使用不同的 in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # 双重嵌套的 vmap 测试
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        # 运行正确性测试，分别使用 torch.float 和 torch.cfloat 作为数据类型
        run_test(torch.float)
        run_test(torch.cfloat)

        # 检查 torch.conj 在非复数张量上的行为，确保返回相同的张量
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        # 使用 self.assertEqual 进行断言，验证返回的张量与原张量具有相同的数据指针
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())
    def test_contiguous(self):
        # 定义操作符 op，指向 Tensor.contiguous 方法
        op = Tensor.contiguous

        # 使用 _test_unary 方法测试 op 方法，传入参数 TensorFactory.randn 和 "cpu"
        self._test_unary(op, TensorFactory.randn, "cpu")

        # 检查如果每个例子已经是连续的，contiguous 方法是否返回原始张量
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        # 在 in_dims=2, out_dims=2 的上下文中使用 vmap 对 Tensor.contiguous 进行映射操作
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        # 在 vmap 中查询 memory_format 的内存布局是否可行，预期会抛出 RuntimeError
        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        B0 = 3

        # 创建形状为 (B0, 2, 5, 7) 的随机张量 x
        x = torch.randn(B0, 2, 5, 7)

        # 定义函数 foo，用于验证张量 x 的步幅是否为 (7*5, 7, 1)
        def foo(x):
            assert x.stride() == (7 * 5, 7, 1)
            return x

        # 在 in_dims=None 的上下文中使用 vmap 对 foo 函数进行映射操作
        vmap(foo)(x)

        # 创建形状为 (2, B0, 5, 7) 的随机张量 x，并将第一个和第二个维度交换
        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        # 定义函数 bar，用于验证张量 x 的步幅是否为 (7*5*B0, 7, 1)
        def bar(x):
            assert x.stride() == (7 * 5 * B0, 7, 1)
            return x

        # 在 in_dims=None 的上下文中使用 vmap 对 bar 函数进行映射操作
        vmap(bar)(x)

    def test_chunk(self):
        test = self._vmap_view_test
        op = torch.chunk
        B0, B1, B2 = 7, 11, 13

        # 使用 test 方法测试 torch.chunk 操作，传入参数 (torch.rand(B0, 2, 1024), 15, -1)，in_dims=(0, None, None)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        # 使用 test 方法测试 torch.chunk 操作，传入参数 (torch.rand(2, B0, 1024), 9, 1)，in_dims=(1, None, None)
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        # 使用 vmap 对 torch.chunk 进行映射操作，传入参数 (torch.rand(B1, 1023, B0, 5), 4, 0)，in_dims=(2, None, None)
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        # 使用 vmap 对 vmap(lambda t: op(t, 4, 1), in_dims=2) 进行映射操作，传入参数 (torch.rand(B1, 2, B0, 64, B2),)，in_dims=2
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        # 定义 clamp_cases 元组，包含多个 clamp 操作的 lambda 函数和获取随机张量的 getter 方法
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )

        # 对每个 clamp 操作进行测试
        for op, getter in clamp_cases:
            # 使用 _test_unary 方法测试 op 方法，传入参数 getter 和 "cpu"
            self._test_unary(op, getter, "cpu")
    def test_comparison_ops(self):
        # 定义一个测试函数的偏函数，指定 check_propagates_grad=False
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        # 定义一个获取随机张量的函数
        getter = TensorFactory.randn
        B0, B1 = 7, 11  # 设定两个批次大小

        # 定义一组比较运算符和对应的匿名函数
        ops = (
            torch.eq,                    # 等于运算符
            lambda x, y: x == y,         # 等于的匿名函数
            torch.gt,                    # 大于运算符
            lambda x, y: x > y,          # 大于的匿名函数
            torch.ge,                    # 大于等于运算符
            lambda x, y: x >= y,         # 大于等于的匿名函数
            torch.le,                    # 小于等于运算符
            lambda x, y: x <= y,         # 小于等于的匿名函数
            torch.lt,                    # 小于运算符
            lambda x, y: x < y,          # 小于的匿名函数
            torch.ne,                    # 不等于运算符
            lambda x, y: x != y,         # 不等于的匿名函数
        )

        # 遍历每个比较运算符
        for op in ops:
            # 单层 vmap 测试：op(Tensor, Tensor)
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))

            # 嵌套 vmap 测试：op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3]), getter([B1, 3])),
                in_dims=(0, None),
            )

            # 测试使用数字作为输入
            number = getter([]).item()
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False
            )

    def test_cross_batch_size_three(self):
        # 测试当批次大小为 3 且未指定跨维度参数时的边界情况
        # 根据 cross API，维度将被分配给第一个具有值 3 的维度
        # 确保找到的维度不是批次维度
        op = torch.cross
        test = self._vmap_test
        B0 = B1 = 3
        test(op, (torch.rand(B0, 2, 3), torch.rand(B0, 2, 3)))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B0, B1, 2, 3), torch.rand(B0, B1, 2, 3)),
            in_dims=(None, 1),
        )

    def test_diagonal(self):
        tensor = torch.randn(3, 5, 7, 11, 13)  # 创建一个张量
        test = self._vmap_view_test  # 使用视图测试函数
        op = torch.diagonal  # 对角线操作
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )
    def test_dot(self):
        # 使用 torch.dot 函数的别名 op 进行测试
        op = torch.dot
        # 使用 self._vmap_test 函数的别名 test 进行测试
        test = self._vmap_test
        # 定义两个维度 B0 和 B1
        B0, B1 = 7, 11

        # shape mismatch
        # 初始化错误消息为空字符串
        msg = ""
        # 测试在给定错误消息的情况下是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # left arg is vmapped
        # 测试左侧参数为 vmap 包装的情况
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        # 测试右侧参数为 vmap 包装的情况
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        # 测试两个参数都为 vmap 包装的情况
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        # 使用 torch.Tensor.expand_as 函数的别名 op 进行测试
        op = torch.Tensor.expand_as
        # 使用 self._vmap_view_test 函数的别名 test 进行测试
        test = self._vmap_view_test
        # 定义三个维度 B0、B1 和 B2
        B0, B1, B2 = 7, 11, 13
        # 测试不同形状的张量之间的 expand_as 操作
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        # 测试 vmap 包装下不同形状的张量之间的 expand_as 操作
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))
    # 定义一个测试方法，用于原地填充和置零操作
    def test_fill_and_zero_inplace(self):
        # 使用 functools.partial 创建一个部分函数，延迟执行 _vmap_test 方法，禁止梯度传播检查
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        # 定义两个维度 B0 和 B1
        B0, B1 = 7, 11
        # 定义操作列表，每个操作是一个 lambda 函数
        ops = (
            lambda t: t.fill_(0.1),  # 使用标量值填充张量 t
            lambda t: t.fill_(torch.tensor(0.2)),  # 使用张量值填充张量 t
            lambda t: t.zero_(),  # 将张量 t 置零
        )

        # 遍历操作列表
        for op in ops:
            # 单层 vmap 测试，不同的输入维度 / 输出维度
            test(op, [TensorFactory.randn([B0, 3])])  # 输入维度为默认值
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)  # 指定输入维度为 2
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)  # 指定输入输出维度均为 2

            # 双层嵌套 vmap 测试
            test(vmap(op), [TensorFactory.randn([B0, B1])])  # 默认输入输出维度
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)  # 指定输入维度为 2
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )  # 指定输入输出维度均为 2

        # 当填充操作的值是一个批次张量时进行测试
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        # 检查在写入张量时未进行 vmap 操作时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, ""):
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

    # 测试复杂视图操作
    def _test_complex_views(self, op, dtypes):
        test = self._vmap_view_test  # 获取视图测试函数的引用

        def run_test(op, dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # 单层 vmap 测试，不同的输入维度 / 输出维度
            test(op, [get([B0, 3])])  # 输入维度为默认值
            test(op, [get([3, B0])], in_dims=1)  # 指定输入维度为 1
            test(op, [get([2, 5, B0, 3])], in_dims=2)  # 指定输入维度为 2
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)  # 指定输入输出维度均为 2

            # 双层嵌套 vmap 测试
            test(vmap(op), [get([B0, B1])])  # 默认输入输出维度
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)  # 指定输入维度为 4
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)  # 指定输入输出维度均为 2

        # 遍历数据类型列表，并执行测试
        for dtype in dtypes:
            run_test(op, dtype)

    # 测试 torch.real 函数
    def test_real(self):
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    # 测试 torch.imag 函数
    def test_imag(self):
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    # 测试 torch.view_as_real 函数
    def test_view_as_real(self):
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )
    def test_view_as_complex(self):
        # 定义一个内部函数 run_test，用于测试给定数据类型的不同情况
        def run_test(dtype):
            # 定义一个内部函数 get，返回指定形状的随机张量
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            # 设置操作符 op 为 torch.view_as_complex
            op = torch.view_as_complex
            # 设置测试函数为 self._vmap_view_test
            test = self._vmap_view_test
            B0, B1 = 7, 11

            # 单个 vmap 测试，不同的输入和输出维度
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # 双重嵌套 vmap 测试
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # 有趣的案例 #1: 批次维度直接在大小为2的维度之前
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # 有趣的案例 #2: 张量末尾的批次维度，成功的情况
            # view_as_complex 要求大小为2的维度具有步长为1，以便视图正确运行
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # 有趣的案例 #3: 张量末尾的批次维度，失败的情况
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # 无效的输入：没有大小为2的维度
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # 无效的输入：批次维度大小为2，但逻辑上的最后一个维度不是大小为2
            msg = "Tensor must have a last dimension of size 2"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        # 对于每种数据类型 [torch.float, torch.double]，运行测试函数
        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        # 创建一个复杂张量和一个普通张量
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        # 定义一个函数 foo，根据输入张量是否复杂，返回相应的张量
        def foo(x):
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        # 使用 vmap 对 foo 函数进行批处理，并与预期结果进行断言比较
        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))
    def test_is_floating_point(self):
        # 创建一个包含浮点数的张量
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        # 创建一个包含整数的张量
        long_tensor = torch.tensor([1, 2, 3])

        # 定义一个函数 foo，根据输入张量的浮点性质返回张量 1 或 0
        def foo(x):
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        # 使用 vmap 对 foo 函数进行矢量化映射，分别对浮点数张量和整数张量进行处理，并进行断言
        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def test_is_contiguous(self):
        # 定义一个函数 foo，根据输入张量是否连续返回浮点数张量 1.0 或 0.0
        def foo(x):
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        B0, B1 = 3, 5

        # 单一批次维度
        # 创建一个连续的张量 contig，并进行断言
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        # 创建一个非连续的张量 noncontig，并进行断言
        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # 多个批次维度
        # 创建一个连续的张量 contig，并进行断言
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        # 创建一个非连续的张量 noncontig，并进行断言
        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # 对于空张量，is_contiguous 返回 True
        # 定义一个函数 bar，断言输入张量是连续的
        def bar(x):
            assert x.is_contiguous()
            return x

        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).transpose(-1, -2))

        # 对于其他内存格式的张量，is_contiguous 抛出 RuntimeError
        # 定义一个函数 baz，用于检查指定内存格式下的连续性
        def baz(x, memory_format):
            x.is_contiguous(memory_format=memory_format)
            return x

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

        for mf in (torch.channels_last, torch.channels_last_3d):
            # 对于特定内存格式，使用 torch.compile 创建一个函数 f，检查张量是否连续，抛出 RuntimeError
            @torch.compile(backend="eager", fullgraph=True)
            def f(x):
                if x.is_contiguous(memory_format=mf):
                    return x.sin()
                return x.cos()

            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(f)(torch.randn(3, 3))
    # 定义测试函数 test_unsqueeze，用于测试 torch.unsqueeze 操作
    def test_unsqueeze(self):
        # 将 torch.unsqueeze 赋值给 op 变量
        op = torch.unsqueeze
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        B0, B1 = 7, 11

        # unsqueeze dim 0
        # 测试在维度 0 上执行 unsqueeze 操作
        test(op, (torch.rand(B0, 2, 5), 0), in_dims=(0, None))
        # 测试在维度 0 上执行 unsqueeze 操作（交换 B0 和 2 的位置）
        test(op, (torch.rand(2, B0, 5), 0), in_dims=(1, None))

        # unsqueeze last dim (positive)
        # 测试在最后一个维度上执行 unsqueeze 操作（正索引）
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        # 测试在最后一个维度上执行 unsqueeze 操作（正索引，交换 B0 和 2 的位置）
        test(op, (torch.rand(2, B0, 5), 2), in_dims=(1, None))

        # unsqueeze last dim (negative)
        # 测试在最后一个维度上执行 unsqueeze 操作（负索引）
        test(op, (torch.rand(B0, 2, 5), -1), in_dims=(0, None))
        # 测试在最后一个维度上执行 unsqueeze 操作（负索引，交换 B0 和 2 的位置）
        test(op, (torch.rand(2, B0, 5), -1), in_dims=(1, None))

        # nested vmaps
        # 嵌套的 vmap 测试
        def unsqueeze_0(x):
            return torch.unsqueeze(x, 0)

        def unsqueeze_last(x):
            return torch.unsqueeze(x, -1)

        # bdims in canonical order
        # 在规范顺序下测试 unsqueeze_0 函数
        test(vmap(unsqueeze_0), (torch.rand(B0, B1, 2),))
        # 在规范顺序下测试 unsqueeze_last 函数
        test(vmap(unsqueeze_last), (torch.rand(B0, B1, 2),))

        # wild bdims
        # 使用非规范顺序测试 unsqueeze_0 函数
        test(vmap(unsqueeze_0), (torch.rand(B1, 2, B0),), in_dims=2)
        # 使用非规范顺序测试 unsqueeze_0 函数，指定 in_dims=1
        test(vmap(unsqueeze_0, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)
        # 使用非规范顺序测试 unsqueeze_last 函数
        test(vmap(unsqueeze_last), (torch.rand(B1, 2, B0),), in_dims=2)
        # 使用非规范顺序测试 unsqueeze_last 函数，指定 in_dims=1
        test(vmap(unsqueeze_last, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)

    # 定义测试函数 test_movedim，用于测试 torch.movedim 操作
    def test_movedim(self):
        # 将 torch.movedim 赋值给 op 变量
        op = torch.movedim
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # movedim(tensor, int, int) variant
        # 测试 movedim 操作，接受 tensor, int, int 形式的参数
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) variant
        # 测试 movedim 操作，接受 tensor, intlist, intlist 形式的参数
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
    def test_mm(self):
        # 设置操作为矩阵乘法
        op = torch.mm
        # 设置测试函数为_vmap_test
        test = self._vmap_test
        # 定义两个维度 B0 和 B1
        B0, B1 = 7, 11

        # shape mismatch
        # 定义形状不匹配的错误消息
        msg = "Shape mismatch"
        # 测试：第一个张量的最内层维度与第二个张量的最外层维度不匹配
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        # 测试：第一个张量的最内层维度与第二个张量的最内层维度不匹配
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        # 测试：第一个张量的最外层维度与第二个张量的最内层维度不匹配
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        # 测试：左边的参数进行了 vmap
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        # 测试：使用 vmap 对左边参数进行操作
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        # 测试：右边的参数进行了 vmap
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        # 测试：使用 vmap 对右边参数进行操作
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        # 测试：两个参数都进行了 vmap
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        # 测试：使用 vmap 对两个参数进行操作
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        # 测试：使用 vmap 对第一个参数进行操作，但不对第二个参数进行操作
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)),
            in_dims=(None, 0),
        )

    def test_mv(self):
        # 设置操作为矩阵向量乘法
        op = torch.mv
        # 设置测试函数为_vmap_test
        test = self._vmap_test
        # 定义两个维度 B0 和 B1
        B0, B1 = 7, 11

        # shape mismatch
        # 定义形状不匹配的错误消息为空字符串
        msg = ""
        # 测试：第一个张量的形状不适合进行矩阵向量乘法操作
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        # 测试：第一个张量的形状不适合进行矩阵向量乘法操作
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        # 测试：第一个张量的形状不适合进行矩阵向量乘法操作
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # left arg is vmapped
        # 测试：左边的参数进行了 vmap
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        # 测试：使用 vmap 对左边参数进行操作
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        # 测试：右边的参数进行了 vmap
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        # 测试：使用 vmap 对右边参数进行操作
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        # 测试：两个参数都进行了 vmap
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        # 测试：使用 vmap 对两个参数进行操作
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        # 测试：使用 vmap 对第一个参数进行操作，但不对第二个参数进行操作
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )
    def test_narrow(self):
        # 定义操作符为 torch.narrow
        op = torch.narrow
        # 定义测试函数为 self._vmap_view_test
        test = self._vmap_view_test
        # 定义三个维度 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13

        # 测试 torch.narrow 操作在不同维度上的应用
        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        
        # 使用 vmap 对 torch.narrow 进行向量化操作，并指定输入维度
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        
        # 嵌套使用 vmap 对 torch.narrow 进行双重向量化操作，并指定输入维度
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

    def test_new_empty(self):
        # Tensor.new_empty 是非确定性的，因此只检查输出张量的形状是否符合预期，
        # 并且不会使用 vmap 的后备方式。
        op = Tensor.new_empty

        # 定义两个维度 B0, B1 分别为 7, 11
        B0, B1 = 7, 11

        # 使用 vmap 对 lambda 函数进行向量化，调用 op 函数创建新的空张量
        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        # 使用 vmap 对 lambda 函数进行向量化，调用 op 函数创建新的空张量
        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        # 嵌套使用 vmap 对 lambda 函数进行双重向量化，调用 op 函数创建新的空张量
        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])
    def test_new_empty_strided(self):
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11

        def _test_single_vmap(size, stride, B0):
            # Generate a random tensor of shape (B0,)
            x = torch.randn(B0)
            # Apply vmap to create a tensor with new empty strided storage
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            # Calculate the size of the storage of an empty tensor with specified stride
            S = torch.empty_strided(size, stride).storage().size()
            # Assert that the shape of the result matches [B0] + size
            self.assertEqual(result.shape, [B0] + size)
            # Assert that the stride of the result matches [S] + stride
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            # Generate a random tensor of shape (B0, B1)
            x = torch.randn(B0, B1)
            # Apply vmap twice to create a tensor with new empty strided storage
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            # Calculate the size of the storage of an empty tensor with specified stride
            S = torch.empty_strided(size, stride).storage().size()
            # Assert that the shape of the result matches [B0, B1] + size
            self.assertEqual(result.shape, [B0, B1] + size)
            # Assert that the stride of the result matches [B1 * S, S] + stride
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            # Transpose the input tensor x and apply vmap twice
            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
            # Calculate the size of the storage of an empty tensor with specified stride
            S = x.new_empty_strided(size, stride).storage().size()
            # Assert that the shape of the result matches [B0, B1] + size
            self.assertEqual(result.shape, [B0, B1] + size)
            # Assert that the stride of the result matches [B1 * S, S] + stride
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # Test cases for contiguous tensors
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # Test cases for expanded tensors
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # Additional edge cases
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        # Function under test is Tensor.new_zeros
        op = Tensor.new_zeros
        # Use functools.partial to create a partially applied function for testing
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11

        # Test new_zeros applied directly
        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))

        # Test new_zeros applied with vmap
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        # Function under test is torch.select
        op = torch.select
        # Use self._vmap_view_test for testing
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # Test select with different in_dims configurations
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))

        # Test select with vmap applied
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )
    def test_roll_no_dims(self):
        op = torch.roll  # 将 torch.roll 函数赋值给变量 op
        test = self._vmap_test  # 将 self._vmap_test 方法赋值给变量 test
        B0, B1, B2 = 7, 11, 13  # 定义变量 B0, B1, B2 分别赋值为 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))  # 调用 test 方法，传入参数和指定的输入维度
        test(op, (torch.rand(2, B0, 5), 3), in_dims=(1, None))  # 调用 test 方法，传入参数和指定的输入维度
        test(vmap(lambda t: op(t, 3)), (torch.rand(B1, 2, B0, 5),), in_dims=2)  # 调用 test 方法，传入参数和指定的输入维度
        test(
            vmap(vmap(lambda t: op(t, 3), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )  # 嵌套调用 vmap 和 test 方法，传入参数和指定的输入维度

    def test_stack(self):
        test = self._vmap_test  # 将 self._vmap_test 方法赋值给变量 test
        B0, B1 = 5, 7  # 定义变量 B0, B1 分别赋值为 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.stack(tensors, dim=dim)  # 定义一个函数 op，用于在指定维度 dim 上堆叠输入的张量

            return op

        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))  # 调用 test 方法，传入参数和自定义操作
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))  # 调用 test 方法，传入参数和指定的输入维度
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))  # 调用 test 方法，传入参数和指定的输入维度
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))  # 调用 test 方法，传入参数和指定的输入维度
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 2)),
            in_dims=(None, 0),
        )  # 嵌套调用 vmap 和 test 方法，传入参数和指定的输入维度
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 2)),
            in_dims=(None, 0),
        )  # 嵌套调用 vmap 和 test 方法，传入参数和指定的输入维度

    def test_slice(self):
        test = self._vmap_view_test  # 将 self._vmap_view_test 方法赋值给变量 test
        B0, B1, B2 = 7, 11, 13  # 定义变量 B0, B1, B2 分别赋值为 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))  # 调用 test 方法，传入参数和自定义的切片操作
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)  # 调用 test 方法，传入参数、指定的输入维度和自定义的切片操作
        test(
            vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2
        )  # 嵌套调用 vmap 和 test 方法，传入参数、指定的输入维度和自定义的切片操作
        test(
            vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
            (torch.rand(3, 5, B0, B1, B2),),
            in_dims=2,
        )  # 嵌套调用 vmap 和 test 方法，传入参数、指定的输入维度和自定义的切片操作

    @xfailIfTorchDynamo
    def test_squeeze(self):
        # 定义内部函数 verify_behavior，用于测试操作 op 的批处理行为
        def verify_behavior(op, min_ndim=1):
            test = self._vmap_view_test
            B0, B1 = 1, 11
            # 这些测试不能与需要批处理后维度大于1的操作一起使用
            if min_ndim <= 1:
                # 测试 op 在不同输入维度下的表现
                test(op, (torch.rand(B0),))
                test(op, (torch.rand(B1),))
                test(vmap(op), (torch.rand(B0, B1, 1),))
                test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)
            # 进行其他维度下的测试
            test(op, (torch.rand(B0, 3, 5),))
            test(op, (torch.rand(1, B0, 5),), in_dims=1)
            test(op, (torch.rand(B0, 0, 1, 5, 1),))
            test(op, (torch.rand(B0, 1, 1, 1, 1),))
            test(vmap(op), (torch.rand(B0, B1, 1, 3, 4),))
            test(vmap(op), (torch.rand(B1, 1, B0, 4, 5),), in_dims=2)

        # 调用 verify_behavior 来测试 torch.squeeze 在不同情况下的行为
        verify_behavior(torch.squeeze)
        verify_behavior(lambda x: torch.squeeze(x, dim=0), min_ndim=1)
        verify_behavior(lambda x: torch.squeeze(x, dim=1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-1), min_ndim=2)
        verify_behavior(lambda x: torch.squeeze(x, dim=-2), min_ndim=3)

        # 进行异常处理测试
        msg = ""
        try:
            torch.squeeze(torch.rand(10), dim=1)
        except IndexError as err:
            msg = str(err)
        with self.assertRaises(RuntimeError, msg=msg):
            vmap(lambda x: torch.squeeze(x, dim=1))(torch.rand(10))

    def _test_mean_sum_dim(self, op):
        test = self._vmap_test
        B0, B1 = 5, 7

        # 单个 vmap 测试，不同的 in_dims / out_dims
        test(lambda x: op(x, 0), [torch.randn([B0])])
        test(lambda x: op(x, -1), [torch.randn([B0])])
        test(lambda x: op(x, 0), [torch.randn([B0, 3])])
        test(lambda x: op(x, -1), [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(lambda x: op(x, 2), [torch.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

        # 双重嵌套 vmap 测试
        test(vmap(lambda x: op(x, 0)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -1)), [torch.randn([B0, B1])])
        test(vmap(lambda x: op(x, -2)), [torch.randn([B1, 2, 5, B0, 3])], in_dims=2)
        test(
            vmap(lambda x: op(x, 2), in_dims=2),
            [torch.randn([2, 5, B0, B1, 3])],
            in_dims=2,
            out_dims=2,
        )

    def test_sum_dim(self):
        # 调用 _test_mean_sum_dim 来测试 torch.sum 的行为
        self._test_mean_sum_dim(torch.sum)

    def test_mean_dim(self):
        # 调用 _test_mean_sum_dim 来测试 torch.mean 的行为
        self._test_mean_sum_dim(torch.mean)
    # 定义测试函数 test_argmax_dim，测试 torch.argmax 在不同维度上的行为
    def test_argmax_dim(self):
        # 定义内部函数 test，用于比较单次运行和批量运行的结果是否相等
        def test(f, args):
            # 遍历 get_fallback_and_vmap_exhaustive 返回的每一对 (loop_out, batched_out)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                # 断言单次运行结果等于批量运行结果
                self.assertEqual(loop_out, batched_out)

        B0 = 5  # 设置 B0 的值为 5
        # 测试 lambda 函数，对单个维度的 tensor 进行 torch.argmax
        test(lambda x: torch.argmax(x), [torch.randn(B0)])
        # 测试 lambda 函数，对二维 tensor 进行 torch.argmax
        test(lambda x: torch.argmax(x), [torch.randn(B0, 2, 3)])
        # 测试 lambda 函数，对三维 tensor 在第一个维度上进行 torch.argmax
        test(lambda x: torch.argmax(x, 0), [torch.randn(B0, 2, 3)])
        # 测试 lambda 函数，对三维 tensor 在最后一个维度上进行 torch.argmax
        test(lambda x: torch.argmax(x, -1), [torch.randn(B0, 2, 3)])
        # 测试 lambda 函数，对三维 tensor 在第二个维度上进行 torch.argmax
        test(lambda x: torch.argmax(x, 2), [torch.randn(B0, 2, 3)])

    # 定义 _test_sum_mean 函数，测试 sum 和 mean 函数的批量操作
    def _test_sum_mean(self, op):
        test = self._vmap_test  # 将 self._vmap_test 赋值给 test
        B0, B1 = 5, 7  # 设置 B0 和 B1 的值分别为 5 和 7

        # 单个 vmap，测试不同的 in_dims / out_dims
        test(op, [torch.randn([B0])])
        test(op, [torch.randn([B0, 3])])
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(op, [torch.randn([2, 5, B0, 3])], in_dims=2)

        # 双层嵌套 vmap
        test(vmap(op), [torch.randn([B0, B1])])
        test(vmap(op), [torch.randn([B1, 2, 5, B0, 3])])
        test(vmap(op), [torch.randn([2, 5, B0, B1, 3])], in_dims=2)

    # 定义测试函数 test_sum，测试 torch.sum 的批量操作
    def test_sum(self):
        self._test_sum_mean(torch.sum)

    # 定义测试函数 test_mean，测试 torch.mean 的批量操作
    def test_mean(self):
        self._test_sum_mean(torch.mean)

    # 定义测试函数 test_repeat，测试 tensor.repeat 的批量操作
    def test_repeat(self):
        test = self._vmap_test  # 将 self._vmap_test 赋值给 test
        B0 = 7  # 设置 B0 的值为 7
        op = Tensor.repeat  # 将 Tensor.repeat 赋值给 op
        # 测试 lambda 函数，对 tensor 进行 repeat 操作，指定重复次数为 (2, 3)
        test(lambda x: op(x, (2, 3)), (torch.rand(B0, 1, 1),))
        # 测试 lambda 函数，对 tensor 进行 repeat 操作，指定重复次数为 (2, 3)，并设置 in_dims=1
        test(lambda x: op(x, (2, 3)), (torch.rand(1, B0, 1),), in_dims=1)

    # 标记为跳过 Torch Dynamo 的测试函数 test_slogdet，测试 torch.linalg.slogdet 的批量操作
    @skipIfTorchDynamo()
    def test_slogdet(self):
        # 使用 functools.partial 部分应用 self._vmap_test 函数，设置 check_propagates_grad=False
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0 = 7  # 设置 B0 的值为 7
        op = torch.linalg.slogdet  # 将 torch.linalg.slogdet 赋值给 op
        # 测试 op 函数，对 tensor 进行 slogdet 操作
        test(op, (torch.rand(B0, 1, 1),))
        test(op, (torch.rand(B0, 2, 2),))
        test(op, (torch.rand(B0, 3, 2, 2),))
        test(op, (torch.rand(3, 2, 2, B0),), in_dims=3)

    # 定义测试函数 test_reshape，测试 torch.reshape 的批量操作
    def test_reshape(self):
        test = self._vmap_test  # 将 self._vmap_test 赋值给 test
        B0, B1, B2 = 7, 11, 13  # 设置 B0、B1、B2 的值分别为 7、11、13
        op = torch.reshape  # 将 torch.reshape 赋值给 op
        # 测试 op 函数，对 tensor 进行 reshape 操作，设置 in_dims=(0, None)，并检查是否为 view 操作
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        # 测试 op 函数，对 tensor 进行 reshape 操作，设置 in_dims=(1, None)，并不检查是否为 view 操作
        test(
            op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False
        )
        # 测试 vmap(lambda t: t.reshape([-1])) 函数，对 tensor 进行 reshape 操作，并检查是否为 view 操作
        test(
            vmap(lambda t: t.reshape([-1])),
            (torch.rand(B0, B1, 2, 5),),
            check_view=True,
        )
        # 测试 vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1) 函数，对 tensor 进行 reshape 操作，
        # 设置 in_dims=5，并不检查是否为 view 操作
        test(
            vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
            (torch.rand(3, B1, 2, B2, 5, B0),),
            in_dims=5,
            check_view=False,
        )
    # 定义测试函数 test_reshape_as
    def test_reshape_as(self):
        # 将 self._vmap_test 赋值给 test 变量
        test = self._vmap_test
        # 定义三个变量 B0, B1, B2 分别赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 将 torch.Tensor.reshape_as 赋值给 op 变量
        op = torch.Tensor.reshape_as
        
        # 调用 test 函数，测试 op 在给定参数下的行为
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        
        # 调用 test 函数，测试 op 在给定参数下的行为，并指定输入维度
        test(
            op,
            (torch.rand(2 * 5), torch.rand(B0, 2, 5)),
            in_dims=(None, 0),
            check_view=True,
        )
        
        # 调用 test 函数，测试 op 在给定参数下的行为，并指定输入维度
        test(
            op,
            (torch.rand(B0, 2 * 5), torch.rand(2, 5)),
            in_dims=(0, None),
            check_view=True,
        )
        
        # 调用 test 函数，测试 op 在给定参数下的行为，并指定输入维度
        test(
            op,
            (torch.rand(2, B0, 5), torch.rand(1, 1, 10)),
            in_dims=(1, None),
            check_view=False,
        )
        
        # 调用 vmap(op) 函数，测试 op 在给定参数下的行为
        test(
            vmap(op),
            (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)),
            check_view=True,
        )
        
        # 调用嵌套 vmap(op, in_dims=(2, None)) 的结果进行测试，并指定输入维度
        test(
            vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
            (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(5, 0),
            check_view=False,
        )
    # 定义一个内部函数，用于包装操作函数，返回一个新函数，该函数总是返回指定数据类型的张量
    def scalar_tensor_with_dtype(op):
        def wrapped(*args, **kwargs):
            # 调用操作函数，获取返回的数据类型
            dtype = op(*args, **kwargs)
            # 返回一个数据类型为 dtype，形状为空的张量（标量）
            return torch.ones([], dtype=dtype)

        return wrapped

    # 将测试函数缩写为 test
    test = self._vmap_test
    # 使用 scalar_tensor_with_dtype 函数对 torch.result_type 进行包装
    op = scalar_tensor_with_dtype(torch.result_type)

    # 定义批次大小 B0
    B0 = 2

    # 运行测试函数 test，验证 op 对 (torch.randn(B0), torch.randn(B0, dtype=torch.float64)) 的影响
    test(
        op,
        (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
        check_propagates_grad=False,
    )
    # 运行测试函数 test，验证 op 对 (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)) 的影响
    test(
        op,
        (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
        check_propagates_grad=False,
    )

    # 运行测试函数 test，验证 lambda 函数 op(x, 1) 对 torch.randn(B0) 的影响
    test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
    # 运行测试函数 test，验证 lambda 函数 op(x, 1.6) 对 torch.randn(B0) 的影响
    test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

    # 运行测试函数 test，验证 lambda 函数 op(x, torch.tensor(1)) 对 torch.randn(B0) 的影响
    test(
        lambda x: op(x, torch.tensor(1)),
        (torch.randn(B0),),
        check_propagates_grad=False,
    )
    # 运行测试函数 test，验证 lambda 函数 op(x, torch.tensor(1.6, dtype=torch.double)) 对 torch.randn(B0) 的影响
    test(
        lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
        (torch.randn(B0),),
        check_propagates_grad=False,
    )

    # 运行测试函数 test，验证 op 对 (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)) 的影响
    test(
        op,
        (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
        check_propagates_grad=False,
    )
    # 运行测试函数 test，验证 op 对 (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)) 的影响
    test(
        op,
        (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
        check_propagates_grad=False,
    )

    # 运行测试函数 test，验证 lambda 函数 op(x, 1) 对 torch.randn(B0, 2) 的影响
    test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
    # 运行测试函数 test，验证 lambda 函数 op(x, 1.6) 对 torch.randn(B0, 2) 的影响
    test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

    # 运行测试函数 test，验证 lambda 函数 op(x, torch.tensor(1)) 对 torch.randn(B0, 2) 的影响
    test(
        lambda x: op(x, torch.tensor(1)),
        (torch.randn(B0, 2),),
        check_propagates_grad=False,
    )
    # 运行测试函数 test，验证 lambda 函数 op(x, torch.tensor(1.6, dtype=torch.double)) 对 torch.randn(B0, 2) 的影响
    test(
        lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
        (torch.randn(B0, 2),),
        check_propagates_grad=False,
    )

    # 运行测试函数 test，验证 op 对 (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)) 的影响
    test(
        op,
        (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
        check_propagates_grad=False,
    )
    # 运行测试函数 test，验证 op 对 (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)) 的影响
    test(
        op,
        (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
        check_propagates_grad=False,
    )
    # 定义一个测试方法，用于测试 torch.tensor_split 函数的各种用例
    def test_tensor_split(self):
        # 将 self._vmap_view_test 赋值给 test 变量，以简化后续的调用
        test = self._vmap_view_test
        # 将 torch.tensor_split 函数赋值给 op 变量，以简化后续的调用
        op = torch.tensor_split
        # 定义三个整数变量 B0, B1, B2 分别赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        # 测试 torch.tensor_split 函数的第一种用法，输入为一个整数 indices_or_sections 和一个维度 dim
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        # 同上，但是在不同的维度上进行测试
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        # 使用 vmap 对 torch.tensor_split 进行映射，测试在不同维度上的应用
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap，对包含 lambda 函数的 torch.tensor_split 进行测试
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        # 测试 torch.tensor_split 函数的第二种用法，输入为一个整数列表 indices_or_sections 和一个维度 dim
        test(
            op,
            (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1),
            in_dims=(0, None, None),
        )
        # 同上，但是在不同的维度上进行测试
        test(
            op,
            (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1),
            in_dims=(1, None, None),
        )
        # 使用 vmap 对 torch.tensor_split 进行映射，测试在不同维度上的应用
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap，对包含 lambda 函数的 torch.tensor_split 进行测试
        test(
            vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    # 装饰器函数，条件为不是 Torch Dynamo 的环境下才执行该测试
    @skipIfTorchDynamo("really slow")
    def test_split(self):
        # 将 self._vmap_view_test 赋值给 test 变量，以简化后续的调用
        test = self._vmap_view_test
        # 将 torch.split 函数赋值给 op 变量，以简化后续的调用
        op = torch.split
        # 定义三个整数变量 B0, B1, B2 分别赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        # 测试 torch.split 函数的第一种用法，输入为一个整数 split_size 和一个维度 dim
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        # 同上，但是在不同的维度上进行测试
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        # 使用 vmap 对 torch.split 进行映射，测试在不同维度上的应用
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap，对包含 lambda 函数的 torch.split 进行测试
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.split(self, split_size: List[int], dim)
        # 测试 torch.split 函数的第二种用法，输入为一个整数列表 split_size 和一个维度 dim
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        # 同上，但是在不同的维度上进行测试
        test(
            op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None)
        )
        # 使用 vmap 对 torch.split 进行映射，测试在不同维度上的应用
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap，对包含 lambda 函数的 torch.split 进行测试
        test(
            vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )
    # 定义一个测试函数，用于测试 torch.trace 操作的不同情况
    def test_trace(self):
        # 设置操作函数为 torch.trace
        op = torch.trace
        # 设置测试函数为 self._vmap_test
        test = self._vmap_test
        # 定义三个维度参数 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试 torch.trace 在给定不同形状的张量上的表现
        test(op, (torch.rand(B0, 2, 5),))
        # 在指定输入维度的情况下测试 torch.trace
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        # 对使用 vmap 包装的 torch.trace 进行测试，in_dims=2 表示输入的维度
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 嵌套使用 vmap 包装的 torch.trace 进行测试，in_dims=2 表示输入的维度
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    # 定义一个测试函数，用于测试 torch.transpose 操作的不同情况
    def test_transpose(self):
        # 设置操作函数为 torch.transpose
        op = torch.transpose
        # 设置测试函数为 self._vmap_view_test
        test = self._vmap_view_test

        # 定义三个维度参数 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试在不同维度上进行 torch.transpose 操作
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        # 在指定输入维度的情况下测试 torch.transpose
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        # 对使用 vmap 包装的 torch.transpose 进行测试，in_dims=2 表示输入的维度
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 嵌套使用 vmap 包装的 torch.transpose 进行测试，in_dims=2 表示输入的维度
        test(
            vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 5, B2),),
            in_dims=2,
        )

        # 特殊情况：标量张量的测试
        for dim1, dim2 in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            # 测试对标量张量进行 vmap(lambda x: op(x, dim1, dim2)) 操作
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            self.assertTrue(result is x)

    # 定义一个测试函数，用于测试 torch.t 操作的不同情况
    def test_t(self):
        # 设置操作函数为 torch.t
        op = torch.t
        # 设置测试函数为 self._vmap_view_test
        test = self._vmap_view_test
        # 定义三个维度参数 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试 torch.t 在给定不同形状的张量上的表现
        test(op, (torch.rand(B0, 2, 5),))
        # 在指定输入维度的情况下测试 torch.t
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        # 对使用 vmap 包装的 torch.t 进行测试，in_dims=2 表示输入的维度
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 嵌套使用 vmap 包装的 torch.t 进行测试，in_dims=2 表示输入的维度
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    # 定义一个测试函数，用于测试 torch.Tensor.T 操作的不同情况
    def test_T_numpy(self):
        # 定义操作函数 op，其功能是返回输入张量的转置
        def op(t):
            return t.T

        # 设置测试函数为 self._vmap_view_test
        test = self._vmap_view_test
        # 定义三个维度参数 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试在不同形状的张量上进行 op 操作
        test(op, (torch.rand(B0, 2, 3, 5),))
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        # 对使用 vmap 包装的 op 进行测试，in_dims=2 表示输入的维度
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        # 嵌套使用 vmap 包装的 op 进行测试，in_dims=2 表示输入的维度
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    # 定义一个测试函数，用于测试 torch.Tensor.to 操作的不同情况
    def test_to(self):
        # 设置测试函数为 self._vmap_test
        test = self._vmap_test
        # 定义两个维度参数 B0, B1
        B0, B1 = 7, 11

        # 测试将张量转移到 "cpu" 设备
        test(lambda t: t.to("cpu"), (torch.rand(B0),))
        # 测试将张量转换为指定数据类型 torch.double
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        # 测试将张量转移到另一个张量所在的设备
        test(
            lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64))
        )
        # 测试将张量转移到另一个张量所在的设备，in_dims=(0, None) 表示指定输入维度
        test(
            lambda t, o: t.to(o),
            (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
            in_dims=(0, None),
        )
        # 对使用 vmap 包装的将张量转换为 torch.double 进行测试
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))

        # 还测试一些类型转换方法
        # 将张量转换为 double 类型
        test(lambda t: t.double(), (torch.rand(B0),))
        # 将张量转换为 float 类型
        test(lambda t: t.float(), (torch.rand(B0),))
        # 将张量转换为 int 类型，check_propagates_grad=False 表示不检查梯度传播
        test(lambda t: t.int(), (torch.rand(B0),), check_propagates_grad=False)
        # 将张量转换为 long 类型，check_propagates_grad=False 表示不检查梯度传播
        test(lambda t: t.long(), (torch.rand(B0),), check_propagates_grad=False)
    # 定义测试函数 test_unfold，用于测试 torch.Tensor.unfold 方法
    def test_unfold(self):
        # 将 torch.Tensor.unfold 方法赋值给变量 op
        op = torch.Tensor.unfold
        # 将 self._vmap_view_test 方法赋值给变量 test
        test = self._vmap_view_test
        # 定义三个变量 B0, B1, B2 并赋值为 3, 2, 5
        B0, B1, B2 = 3, 2, 5

        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 7, 11) 的 unfold 操作
        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(7, B0, 11) 的 unfold 操作
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        # 使用 vmap 对 torch.Tensor.unfold 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 7, B0, 11), 1, 5, 1),
            in_dims=(2, None, None, None),
        )
        # 嵌套使用 vmap 对 torch.Tensor.unfold 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4),
            in_dims=(2, None, None, None),
        )

    # 定义测试函数 test_unbind，用于测试 torch.unbind 方法
    def test_unbind(self):
        # 将 self._vmap_view_test 方法赋值给变量 test
        test = self._vmap_view_test
        # 将 torch.unbind 方法赋值给变量 op
        op = torch.unbind
        # 定义三个变量 B0, B1, B2 并赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13

        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 2, 1024) 的 unbind 操作
        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 2, 0) 的 unbind 操作
        test(op, (torch.rand(B0, 2, 0),))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(2, B0, 7) 的 unbind 操作
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        # 使用 vmap 对 torch.unbind 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 1023, B0, 5), 1),
            in_dims=(2, None),
        )
        # 嵌套使用 vmap 对 lambda 函数应用的 unbind 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
            (torch.rand(B1, 2, B0, 32, B2),),
            in_dims=2,
        )

    # 定义测试函数 test_view，用于测试 torch.Tensor.view 方法
    def test_view(self):
        # 将 self._vmap_view_test 方法赋值给变量 test
        test = self._vmap_view_test
        # 定义三个变量 B0, B1, B2 并赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 将 torch.Tensor.view 方法赋值给变量 op
        op = torch.Tensor.view

        # 在发生 RuntimeError 时抛出异常
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])

        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 2 * 5) 的 view 操作
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 4, 5) 的 view 操作
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        # 使用 vmap 对 lambda 函数应用的 view 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        # 嵌套使用 vmap 对 lambda 函数应用的 reshape 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
            (torch.rand(B2, B0, B1, 3, 2, 5),),
            in_dims=1,
        )

    # 定义测试函数 test_view_as，用于测试 torch.Tensor.view_as 方法
    def test_view_as(self):
        # 将 self._vmap_view_test 方法赋值给变量 test
        test = self._vmap_view_test
        # 定义三个变量 B0, B1, B2 并赋值为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 将 torch.Tensor.view_as 方法赋值给变量 op
        op = torch.Tensor.view_as

        # 在发生 RuntimeError 时抛出异常
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))

        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 2 * 5) 的 view_as 操作
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(2 * 5) 的 view_as 操作
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 2 * 5) 的 view_as 操作
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))

        # 调用 self._vmap_view_test 测试方法，测试 torch.rand(B0, 4, 5) 的 view_as 操作
        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))

        # 使用 vmap 对 op 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        # 嵌套使用 vmap 对 op 方法进行批处理，并调用 self._vmap_view_test 测试方法
        test(
            vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)),
            (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(2, 0),
        )
    # 定义一个测试函数，用于测试 Conv 模块的不同设置
    def test_conv2d(self):
        # 定义多个 Conv 模块的设置，每个设置包括模块类型、函数、输入形状
        conv_setups = [
            (torch.nn.Conv1d, torch.conv1d, [2, 4, 15]),
            (torch.nn.Conv2d, torch.conv2d, [2, 4, 15, 20]),
            (torch.nn.Conv3d, torch.conv3d, [2, 4, 15, 20, 25]),
            # (torch.nn.ConvTranspose2d, torch.conv_transpose2d, [2, 4, 15, 20])
        ]
        # 遍历每个设置
        for conv_mod, conv_fn, inp_shape in conv_setups:
            # 创建一个 Conv 模块实例
            mod = conv_mod(4, 8, kernel_size=3)
            # 准备函数参数的数值
            arg_values = [torch.randn(inp_shape), mod.weight, mod.bias]
            kwarg_values = {}
            # 对每种设置，通过测试工具函数进行 fallback 和 vmap 的全面测试
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                # 断言循环输出与批处理输出相等
                self.assertEqual(loop_out, batched_out)

            # 用 None 作为 bias 进行测试
            arg_values = [torch.randn(inp_shape), mod.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                # 断言循环输出与批处理输出相等
                self.assertEqual(loop_out, batched_out)

            # 创建带有更多参数设置的 Conv 模块实例
            mod2 = conv_mod(
                4, 8, kernel_size=3, groups=2, stride=3, padding=1, dilation=2
            )
            kwarg_values = dict(groups=2, stride=3, padding=1, dilation=2)
            # 准备参数数值
            arg_values = [torch.randn(inp_shape), mod2.weight, mod2.bias]
            # 对每种设置，通过测试工具函数进行 fallback 和 vmap 的全面测试
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                # 断言循环输出与批处理输出相等
                self.assertEqual(loop_out, batched_out)

            # 用 None 作为 bias 进行测试
            arg_values = [torch.randn(inp_shape), mod2.weight, None]
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                conv_fn, arg_values, kwarg_values
            ):
                # 断言循环输出与批处理输出相等
                self.assertEqual(loop_out, batched_out)

    # 定义一个测试函数，用于测试 F.one_hot 函数
    def test_one_hot(self):
        # 定义多个样本输入
        sample_inputs = [
            (torch.randint(0, 3, []), 3),
            (torch.randint(0, 3, [2, 3, 4]), 4),
        ]
        # 遍历每个样本输入
        for args in sample_inputs:
            # 对每个输入，通过测试工具函数进行 fallback 和 vmap 的全面测试
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                F.one_hot, args, {}
            ):
                # 断言循环输出与批处理输出相等
                self.assertEqual(loop_out, batched_out)

    # 定义一个测试函数，用于测试 torch.tensor 的复数共轭操作
    def test_conj_bit(self):
        # 创建一个包含复数的张量
        x = torch.tensor([1 + 1j, 2 + 1j])

        # 定义一个内部函数，对输入张量进行复数共轭操作并断言
        def foo(x):
            # 断言输入张量不是共轭的
            assert not x.is_conj()
            # 进行复数共轭操作
            y = x.conj()
            # 断言共轭后的张量是共轭的
            assert y.is_conj()
            return y

        # 对输入张量应用 vmap 函数，对每个元素进行 foo 函数的操作
        res = vmap(foo)(x)
        # 断言结果与原始张量的共轭相等
        self.assertEqual(res, x.conj())
    # 定义测试函数 test_mode_key，用于测试 vmap 函数的多层嵌套效果
    def test_mode_key(self):
        # 定义映射函数 vmap_f，将输入 x 加上从标准正态分布中随机抽取的数值
        def vmap_f(x):
            return x + torch.randn(())

        # 定义简单的映射函数 naive_f，接受输入 x 和形状 shape，并在每个位置加上随机数
        def naive_f(x, shape):
            return x + torch.randn(shape)

        # 设定随机种子为 0
        torch.manual_seed(0)
        # 使用 vmap 函数对 vmap_f 进行多层嵌套映射，其中 randomness 参数设置为 "different"
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3)
        )

        # 重新设定随机种子为 0
        torch.manual_seed(0)
        # 调用 naive_f 函数，传入相同的输入和形状参数
        out2 = naive_f(torch.ones(2, 3), (2, 3))
        # 断言两个输出结果相等
        self.assertEqual(out1, out2)

        # 再次设定随机种子为 0
        torch.manual_seed(0)
        # 使用 vmap 函数对 vmap_f 进行多层嵌套映射，这次输入是一个更大的张量
        out1 = vmap(vmap(vmap_f, randomness="different"), randomness="different")(
            torch.ones(2, 3, 4)
        )

        # 再次设定随机种子为 0
        torch.manual_seed(0)
        # 调用 naive_f 函数，传入相同的输入和形状参数（形状稍有不同）
        out2 = naive_f(torch.ones(2, 3, 4), (2, 3, 1))
        # 断言两个输出结果相等
        self.assertEqual(out1, out2)

        # 断言 torch.randn(()) 返回的张量维度为 0
        self.assertTrue(torch.randn(()).dim() == 0)

    # 参数化测试函数 test_chunk_vmap，测试 chunk_vmap 函数在不同参数设置下的输出
    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    @parametrize("randomness", ["error", "same"])
    def test_chunk_vmap(self, in_dim, out_dim, randomness):
        # 创建一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)

        # 定义函数 f，对输入张量 x 进行正弦函数处理，并根据 randomness 参数添加随机数
        def f(x):
            y = x.sin()
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        # 获取当前的随机数生成器状态
        rs = torch.get_rng_state()
        # 使用 vmap 函数对函数 f 进行向量化映射，期望输出为 expected
        expected = vmap(f, in_dims=in_dim, out_dims=out_dim, randomness=randomness)(x)

        # 针对不同的 chunks 值进行迭代
        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            # 恢复随机数生成器状态到之前保存的 rs
            torch.set_rng_state(rs)
            # 使用 chunk_vmap 函数对函数 f 进行分块映射，期望输出为 expected
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            # 断言输出与期望值相等
            self.assertEqual(output, expected)
    def test_vmap_chunksize(self, in_dim, out_dim, randomness):
        # 创建一个形状为 (4, 5, 6) 的张量 x，元素为标准正态分布随机数
        x = torch.randn(4, 5, 6)
        # 创建一个与 x 具有相同形状的张量 y，元素为标准正态分布随机数
        y = torch.randn_like(x)

        # 定义一个单输入单输出的函数 f(x)
        def f(x):
            # 计算 x 的正弦值
            y = x.sin()
            # 如果 randomness 不是 "error"，则在 y 上加上与 x 同形状的随机数
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        # 准备函数 f 的参数，这里只有一个 x
        f_args = (x,)
        # 准备函数 f 的关键字参数
        f_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # 定义一个嵌套输入单输出的函数 f1(pair)
        def f1(pair):
            # 将 pair 解包为 x 和 y
            x, y = pair
            # 计算 x 的正弦值与 y 的余弦值之和
            z = x.sin() + y.cos()
            # 如果 randomness 不是 "error"，则在 z 上加上与 z 同形状的随机数
            if randomness != "error":
                z = z + torch.rand_like(z)
            return z

        # 准备函数 f1 的参数，这里是一个元组 (x, y)
        f1_args = ((x, y),)
        # 准备函数 f1 的关键字参数
        f1_kwargs = {
            "in_dims": ((in_dim,) * 2,),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # 定义一个单输入嵌套输出的函数 f2(x)
        def f2(x):
            # 计算 x 的正弦值
            y = x.sin()
            # 如果 randomness 不是 "error"，则在 y 上加上与 x 同形状的随机数
            if randomness != "error":
                y = y + torch.rand_like(x)
            # 返回一个字典，包含两个键 "out" 和 "out1"，对应的值分别是 y 和 y + 2
            return {"out": y, "out1": y + 2}

        # 准备函数 f2 的参数，这里只有一个 x
        f2_args = (x,)
        # 准备函数 f2 的关键字参数
        f2_kwargs = {"in_dims": in_dim, "out_dims": out_dim, "randomness": randomness}

        # 定义一个嵌套输入嵌套输出的函数 f3(inp_dict)
        def f3(inp_dict):
            # 从 inp_dict 中取出键为 "inp" 的张量 x 和键为 "inp1" 的张量 y
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            # 计算 x 的正弦值与 y 的余弦值之和
            z = x.sin() + y.cos()
            # 如果 randomness 不是 "error"，则在 z 上加上与 z 同形状的随机数
            if randomness != "error":
                z = z + torch.rand_like(z)
            # 返回一个字典，包含两个键 "z" 和 "tuple"，对应的值分别是 z 和元组 (z, z + 1)
            return {"z": z, "tuple": (z, z + 1)}

        # 准备函数 f3 的参数，这里是一个包含一个字典的元组
        f3_args = (
            {
                "inp": x.index_select(in_dim, torch.tensor([0])).squeeze(in_dim),
                "inp1": y,
            },
        )
        # 准备函数 f3 的关键字参数
        f3_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # 定义一个嵌套输入嵌套输出的函数 f4(inp_dict)
        def f4(inp_dict):
            # 从 inp_dict 中取出键为 "inp" 的标量 x 和键为 "inp1" 的张量 y
            x = inp_dict["inp"]
            y = inp_dict["inp1"]
            # 计算 x 与 y 的余弦值之和
            z = x + y.cos()
            # 如果 randomness 不是 "error"，则在 z 上加上与 z 同形状的随机数
            if randomness != "error":
                z = z + torch.rand_like(z)
            # 返回一个字典，包含两个键 "z" 和 "tuple"，对应的值分别是 z 和元组 (z, z + 1)
            return {"z": z, "tuple": (z, z + 1)}

        # 准备函数 f4 的参数，这里是一个包含两个标量的字典的元组
        f4_args = ({"inp": 2.0, "inp1": y},)
        # 准备函数 f4 的关键字参数
        f4_kwargs = {
            "in_dims": ({"inp": None, "inp1": in_dim},),
            "out_dims": out_dim,
            "randomness": randomness,
        }

        # 准备要测试的所有函数及其参数和关键字参数的元组
        fns_and_args = (
            (f, f_args, f_kwargs),
            (f1, f1_args, f1_kwargs),
            (f2, f2_args, f2_kwargs),
            (f3, f3_args, f3_kwargs),
            (f4, f4_args, f4_kwargs),
        )
        # 遍历每个函数及其参数，执行 vmap 函数的测试
        for fn, args, kwargs in fns_and_args:
            # 保存当前的随机数状态
            rs = torch.get_rng_state()
            # 使用 vmap 函数生成预期的批处理结果
            expected_vmap = vmap(fn, **kwargs)(*args)
            # 对不同的 chunk_size 进行测试
            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                # 恢复之前保存的随机数状态
                torch.set_rng_state(rs)
                # 使用 vmap 函数测试当前 chunk_size 的结果
                output = vmap(fn, chunk_size=chunk_size, **kwargs)(*args)
                # 使用单元测试断言确认输出与预期相等
                self.assertEqual(output, expected_vmap)
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_error(self, in_dim, out_dim, randomness):
        # 创建一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)

        def f(x):
            # 对输入张量 x 求正弦值，并赋给 y
            y = x.sin()
            # 如果 randomness 不等于 "error"，则在 y 上加上与 x 相同形状的随机张量
            if randomness != "error":
                y = y + torch.rand_like(x)
            return y

        # 对于不正确的 `chunk_size` 参数值 (-1, 0)，抛出 ValueError 异常
        for chunk_size in (-1, 0):
            with self.assertRaisesRegex(
                ValueError, "vmap: chunk_size should be None or greater than 0."
            ):
                # 使用 vmap 对函数 f 进行矢量化映射，检查是否抛出异常
                vmap(
                    f,
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(x)

        # 当 `out_dims` 参数与 `outputs` 结构不兼容时，抛出 ValueError 异常
        msg = "out_dims is not compatible with the structure of `outputs`"
        with self.assertRaisesRegex(ValueError, msg):
            # 使用 vmap 对函数 f 进行矢量化映射，检查是否抛出异常
            vmap(
                f,
                in_dims=in_dim,
                out_dims=(out_dim, out_dim),
                randomness=randomness,
                chunk_size=2,
            )(x)

    @parametrize("in_dim", [0, 1])
    @parametrize("out_dim", [0, 1])
    @parametrize("randomness", ["error", "same"])
    def test_vmap_chunksize_composition(self, in_dim, out_dim, randomness):
        x = torch.randn(4, 5, 6)  # 创建一个形状为 (4, 5, 6) 的张量 x，其中元素服从标准正态分布
        y = torch.randn_like(x)  # 创建一个与 x 具有相同形状的张量 y，元素同样服从标准正态分布

        # fn: Single Input/Single Output
        def f(x):
            y = x.sin()  # 计算张量 x 中每个元素的正弦值
            if randomness != "error":
                y = y + torch.rand_like(x)  # 若随机性不为 "error"，则给 y 的每个元素加上一个与 x 形状相同的随机张量
            return y

        f_args = (x,)  # 将 x 打包成元组，作为参数传递给函数 f

        # fn: Nested Input/Single Output
        def f1(pair):
            x, y = pair
            z = x.sin() + y.cos()  # 计算张量 x 和 y 中对应元素的正弦值与余弦值之和
            if randomness != "error":
                z = z + torch.rand_like(z)  # 若随机性不为 "error"，则给 z 的每个元素加上一个与 z 形状相同的随机张量
            return z

        f1_args = ((x, y),)  # 将 (x, y) 打包成元组，作为参数传递给函数 f1

        # fn: Single Input/Nested Output
        def f2(x):
            y = x.sin()  # 计算张量 x 中每个元素的正弦值
            if randomness != "error":
                y = y + torch.rand_like(x)  # 若随机性不为 "error"，则给 y 的每个元素加上一个与 x 形状相同的随机张量
            return {"out": y, "out1": y + 2}  # 返回一个字典，包含键为 "out" 的 y 和键为 "out1" 的 y+2

        f2_args = (x,)  # 将 x 打包成元组，作为参数传递给函数 f2

        # fn: Nested Input/Nested Output
        def f3(inp_dict):
            x = inp_dict["inp"]  # 从输入的字典中获取键为 "inp" 的张量 x
            y = inp_dict["inp1"]  # 从输入的字典中获取键为 "inp1" 的张量 y
            z = x.sin() + y.cos()  # 计算张量 x 和 y 中对应元素的正弦值与余弦值之和
            if randomness != "error":
                z = z + torch.rand_like(z)  # 若随机性不为 "error"，则给 z 的每个元素加上一个与 z 形状相同的随机张量
            return {"z": z, "tuple": (z, z + 1)}  # 返回一个字典，包含键为 "z" 的 z 和键为 "tuple" 的 (z, z+1)

        f3_args = ({"inp": x, "inp1": y},)  # 将 {"inp": x, "inp1": y} 打包成元组，作为参数传递给函数 f3

        for fn, args in ((f, f_args), (f1, f1_args), (f2, f2_args), (f3, f3_args)):
            rs = torch.get_rng_state()  # 获取当前随机数生成器的状态
            expected = vmap(
                vmap(fn, in_dims=in_dim, out_dims=out_dim, randomness=randomness),
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
            )(*args)  # 对函数 fn 应用 vmap，使用给定的参数 args，得到期望的结果

            for chunk_size in (1, 2, 3, 4, 7, 10, 16, 100):
                torch.set_rng_state(rs)  # 恢复随机数生成器的状态
                actual = vmap(
                    vmap(
                        fn,
                        in_dims=in_dim,
                        out_dims=out_dim,
                        randomness=randomness,
                        chunk_size=chunk_size,
                    ),
                    in_dims=in_dim,
                    out_dims=out_dim,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(*args)  # 对函数 fn 应用双重 vmap，使用给定的参数 args 和 chunk_size，得到实际的结果

                self.assertEqual(actual, expected)  # 断言实际结果与期望结果相等
# 实例化一个带参数化测试的对象，基于 TestVmapOperators 类
instantiate_parametrized_tests(TestVmapOperators)


def construct_v(output, batch_size, contig=False):
    # 如果 contig 为 True，则构造一个形状为 (batch_size, *output.shape) 的张量，数据类型和设备与 output 相同
    if contig:
        return torch.randn(
            batch_size, *output.shape, dtype=output.dtype, device=output.device
        )
    # 否则，构造一个形状为 (*output.shape, batch_size) 的张量，数据类型和设备与 output 相同，并将维度 -1 移动到第 0 维
    result = torch.randn(
        *output.shape, batch_size, dtype=output.dtype, device=output.device
    )
    return result.movedim(-1, 0)


def as_tuple(x):
    # 如果 x 是 tuple 类型，则直接返回
    if isinstance(x, tuple):
        return x
    # 如果 x 是 list 类型，则转换为 tuple 后返回
    elif isinstance(x, list):
        return tuple(x)
    # 否则，将 x 包装成单元素的 tuple 后返回
    else:
        return (x,)


def differentiable(args):
    # 返回一个由 args 中是 torch.Tensor 且 requires_grad=True 的元素组成的 tuple
    return tuple(
        arg
        for arg in as_tuple(args)
        if isinstance(arg, torch.Tensor) and arg.requires_grad
    )


def _get_rand_no_zeros(*args, **kwargs):
    # 获取 kwargs 中的 requires_grad 参数，默认为 False
    requires_grad = kwargs.get("requires_grad", False)
    # 复制 kwargs，并将 requires_grad 设置为 False，然后使用这些参数生成一个 torch.rand 张量
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad["requires_grad"] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    # 将生成的张量所有小于 0.1 的元素设置为 0.1，并根据 requires_grad 参数设置 requires_grad 属性
    return result.clamp_min_(0.1).requires_grad_(requires_grad)


@markDynamoStrictTest
class TestVmapBatchedGradient(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        # 调用 _vmap_test 函数并返回其结果
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # 如果 kwargs 为 None，则将其设为一个空字典
        if kwargs is None:
            kwargs = {}
        # 计算 op(*args, **kwargs) 的输出
        outputs = op(*args, **kwargs)
        # 将输出应用 output_process_fn，然后筛选出其中可微分的部分，并转换成 tuple
        outputs = differentiable(output_process_fn(outputs))
        # 对于 contig 取值为 True 和 False 的情况分别生成批处理向量
        for contig in [True, False]:
            batched_vectors = tuple(
                construct_v(out, batch_size, contig) for out in outputs
            )

            def vector_jacobian_product(*vectors):
                # 计算 outputs 对于 args 中可微分部分的向量-雅可比积
                return torch.autograd.grad(
                    outputs, differentiable(args), vectors, retain_graph=True
                )

            self._vmap_test(
                vector_jacobian_product, batched_vectors, check_propagates_grad=False
            )

    # Tests batched second grad computation of outputs = op(*args, **kwargs).
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    #
    # NB: we only test computing batched gradients in the second gradient
    # computation. One specific use case that does this is computing the hessian
    # matrix of a scalar-valued function; this is useful in Bayesian Logistic
    # Regression.
    # It might be useful to have a test that computes batched first gradients and
    # then uses those to compute batched second gradients in the future.
    # 测试函数，用于测试批量计算梯度的二阶导数
    def _batched_grad_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # 如果 kwargs 为 None，则设为空字典
        if kwargs is None:
            kwargs = {}
        # 调用操作 op，并传入参数 args 和 kwargs，得到输出
        outputs = op(*args, **kwargs)
        # 对输出进行可微化处理
        outputs = differentiable(output_process_fn(outputs))
        # 创建与 outputs 同样形状的全 1 张量列表
        ones = tuple(torch.ones_like(out) for out in outputs)
        # 计算 outputs 关于 args 的梯度，创建一个计算图以便计算二阶导数
        first_grads = torch.autograd.grad(
            outputs, differentiable(args), ones, create_graph=True
        )
        # 对 first_grads 进行可微化处理
        first_grads = differentiable(first_grads)
        # 断言：first_grads 的长度不为 0，即第一阶导数至少对某个输入有依赖
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!"
        )

        # 对于 contig 为 True 和 False 分别进行迭代
        for contig in [True, False]:
            # 构造批量向量，每个向量是 first_grads 中对应元素的批量版本
            batched_vectors = tuple(
                construct_v(grad, batch_size, contig) for grad in first_grads
            )

            # 定义向量-海森矩阵积的函数
            def vector_hessian_product(*vectors):
                # 计算 first_grads 关于 args 的向量-海森矩阵积，保留计算图，允许未使用的输入
                outputs = torch.autograd.grad(
                    first_grads,
                    differentiable(args),
                    vectors,
                    retain_graph=True,
                    allow_unused=True,
                )
                # 过滤掉为 None 的输出，并转换为元组
                outputs = tuple(out for out in outputs if out is not None)
                # 断言：输出的长度大于 0
                assert len(outputs) > 0
                return outputs

            # 进行 _vmap_test 测试，检查是否正确传播梯度
            self._vmap_test(
                vector_hessian_product, batched_vectors, check_propagates_grad=False
            )

    # 测试算术操作函数
    def _test_arithmetic(self, op, device, test_grad_grad=True):
        # 创建具有随机数据的张量 x 和 y，要求计算梯度，位于指定设备上
        x = torch.randn(2, 3, requires_grad=True, device=device)
        y = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        scalar = 3.14
        # 分别对 (x, y), (scalar, y), (x, scalar) 进行批量梯度测试
        self._batched_grad_test(op, (x, y))
        self._batched_grad_test(op, (scalar, y))
        self._batched_grad_test(op, (x, scalar))

        # 如果 test_grad_grad 为 True，进行二阶导数测试
        if test_grad_grad:
            self._batched_grad_grad_test(op, (x, y))

    # 测试加法函数
    def test_add(self, device):
        # 使用 torch.add 进行算术操作测试，不测试二阶导数
        self._test_arithmetic(torch.add, device, test_grad_grad=False)
        # 使用 lambda 函数进行加法操作测试，不测试二阶导数
        self._test_arithmetic(lambda x, y: x + y, device, test_grad_grad=False)

    # 测试减法函数
    def test_sub(self, device):
        # 使用 torch.sub 进行算术操作测试，不测试二阶导数
        self._test_arithmetic(torch.sub, device, test_grad_grad=False)
        # 使用 lambda 函数进行减法操作测试，不测试二阶导数
        self._test_arithmetic(lambda x, y: x - y, device, test_grad_grad=False)

    # 测试乘法函数
    def test_mul(self, device):
        # 使用 torch.mul 进行算术操作测试，并测试二阶导数
        self._test_arithmetic(torch.mul, device)
        # 使用 lambda 函数进行乘法操作测试，并测试二阶导数
        self._test_arithmetic(lambda x, y: x * y, device)

    # 测试除法函数
    def test_div(self, device):
        # 使用 torch.div 进行算术操作测试，并测试二阶导数
        self._test_arithmetic(torch.div, device)
        # 使用 lambda 函数进行除法操作测试，并测试二阶导数
        self._test_arithmetic(lambda x, y: x / y, device)

    # 测试二进制交叉熵函数
    def test_binary_cross_entropy(self, device):
        # 创建具有随机数据的张量 x，使用 sigmoid 函数进行处理，要求计算梯度，位于指定设备上
        x = F.sigmoid(torch.randn(3, 2, device=device, requires_grad=True))
        # 创建具有随机数据的目标张量，位于指定设备上
        target = torch.rand(3, 2, device=device)

        # 部分应用二元交叉熵函数，固定目标张量
        op = functools.partial(F.binary_cross_entropy, target=target)

        # 对 x 进行批量梯度测试，不测试二阶导数
        self._batched_grad_test(op, (x,), {})
        # 对 x 进行批量梯度二阶导数测试
        self._batched_grad_grad_test(op, (x,), {})
    # 测试 log_softmax 函数的梯度计算
    def test_log_softmax(self, device):
        # 定义一个偏函数，用于计算 log_softmax 在最后一个维度上的结果
        op = functools.partial(torch.log_softmax, dim=-1)
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(3, 2, device=device, requires_grad=True)

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,), {})
        # 测试 _batched_grad_grad_test 函数，验证 op 在 x 上的二阶梯度
        self._batched_grad_grad_test(op, (x,), {})

    # 测试 expand 函数的梯度计算
    def test_expand(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, device=device, requires_grad=True)

        # 定义一个函数 op，用于将 x 沿着指定维度扩展成新的形状
        def op(x):
            return x.expand(5, 5, 2, 3)

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,))

    # 使用允许 vmap 回退使用的装饰器，测试 index 函数的梯度计算
    @allowVmapFallbackUsage
    def test_index(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 生成一个索引张量 index，用于在 x 上进行索引操作
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        # 定义一个函数 op，对 x 进行平方操作，并根据索引返回结果张量 y
        def op(x):
            y = x * x
            return y[index]

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,))
        # 测试 _batched_grad_grad_test 函数，验证 op 在 x 上的二阶梯度
        self._batched_grad_grad_test(op, (x,))

    # 测试 lgamma 函数的梯度计算
    def test_lgamma(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 测试 _batched_grad_test 函数，验证 Tensor.lgamma 在 x 上的梯度
        self._batched_grad_test(Tensor.lgamma, (x,))
        # 测试 _batched_grad_grad_test 函数，验证 Tensor.lgamma 在 x 上的二阶梯度
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    # 测试 log 函数的梯度计算
    def test_log(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        # 测试 _batched_grad_test 函数，验证 torch.log 在 x 上的梯度
        self._batched_grad_test(torch.log, (x,))
        # 测试 _batched_grad_grad_test 函数，验证 torch.log 在 x 上的二阶梯度
        self._batched_grad_grad_test(torch.log, (x,))

    # 测试 logsumexp 函数的梯度计算
    def test_logsumexp(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        # 定义一个函数 op，用于计算 x 在指定维度上的 logsumexp 结果
        def op(x):
            return torch.logsumexp(x, -1)

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,))
        # 测试 _batched_grad_grad_test 函数，验证 op 在 x 上的二阶梯度
        self._batched_grad_grad_test(op, (x,))

    # 测试 log1p 函数的梯度计算
    def test_log1p(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        # 测试 _batched_grad_test 函数，验证 torch.log1p 在 x 上的梯度
        self._batched_grad_test(torch.log1p, (x,))
        # 测试 _batched_grad_grad_test 函数，验证 torch.log1p 在 x 上的二阶梯度
        self._batched_grad_grad_test(torch.log1p, (x,))

    # 使用允许 vmap 回退使用的装饰器，测试 max 函数的梯度计算
    @allowVmapFallbackUsage
    def test_max(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 测试 _batched_grad_test 函数，验证 torch.max 在 x 上的梯度
        self._batched_grad_test(torch.max, (x,))

    # 使用允许 vmap 回退使用的装饰器，测试 median 函数的梯度计算
    @allowVmapFallbackUsage
    def test_median(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 测试 _batched_grad_test 函数，验证 torch.median 在 x 上的梯度
        self._batched_grad_test(torch.median, (x,))

    # 使用允许 vmap 回退使用的装饰器，测试 min 函数的梯度计算
    @allowVmapFallbackUsage
    def test_min(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 测试 _batched_grad_test 函数，验证 torch.min 在 x 上的梯度
        self._batched_grad_test(torch.min, (x,))

    # 测试 permute 函数的梯度计算
    def test_permute(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        # 定义一个函数 op，用于对 x 进行维度重排
        def op(x):
            return x.permute(2, 0, 1)

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,))

    # 测试 reshape 函数的梯度计算
    def test_reshape(self, device):
        # 生成一个随机张量 x，设备为指定的设备，并且需要计算梯度
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        # 定义一个函数 op，用于对 x 进行形状重塑
        def op(x):
            return x.reshape([2 * 3, 5])

        # 测试 _batched_grad_test 函数，验证 op 在 x 上的梯度
        self._batched_grad_test(op, (x,))

    # 测试 sigmoid 函数的梯度计算
    def test_sigmoid(self, device):
        # 生成一个随机张量 x，设备
    # 测试栈操作，使用给定的设备创建随机张量 x 和 y
    def test_stack(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        # 定义操作函数 op，将 x 和 y 堆叠成一个张量
        def op(x, y):
            return torch.stack([x, y])

        # 调用通用的批处理梯度测试函数 _batched_grad_test，测试 op 函数的梯度计算
        self._batched_grad_test(op, (x, y))

    # 测试选择操作，使用给定的设备创建随机张量 x
    def test_select(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        
        # 测试 lambda 函数，选择张量 x 的第二个元素
        self._batched_grad_test(lambda x: x[1], (x,))
        
        # 测试 lambda 函数，选择张量 x 在第一个维度上的索引为 2 的元素
        self._batched_grad_test(lambda x: x.select(1, 2), (x,))
        
        # 测试 lambda 函数，选择张量 x 在最后一个维度上的索引为 0 的元素
        self._batched_grad_test(lambda x: x.select(-1, 0), (x,))

    # 测试切片操作，使用给定的设备创建随机张量 x
    def test_slice(self, device):
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        
        # 测试 lambda 函数，切片张量 x 的第一个维度，选择索引为 0 到 1 的元素
        self._batched_grad_test(lambda x: x[0:1], (x,))
        
        # 测试 lambda 函数，切片张量 x 的第二个维度，选择索引为 1 到 3 的元素
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        
        # 测试 lambda 函数，使用省略号切片张量 x 的最后两个维度，选择索引为 1 到 3 的元素
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    # 测试迹(trace)操作，使用给定的设备创建随机张量 x
    def test_trace(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        
        # 使用 Tensor 类的 trace 方法计算张量 x 的迹
        self._batched_grad_test(Tensor.trace, (x,))
        
        # 使用函数 sum_grad_trace 计算张量 x 的迹的梯度的和
        x = torch.randn(3, 2, 2, device=device)

        def sum_grad_trace(x):
            return grad(torch.trace)(x).sum()

        # 对函数 sum_grad_trace 使用 vmap 进行批处理计算
        output = vmap(grad(sum_grad_trace))(x)
        self.assertEqual(output, torch.zeros_like(output))

    # 测试 where 操作，使用给定的设备创建随机张量 x 和全为 1 的张量 y
    def test_where(self, device):
        x = torch.randn(3, 2, device=device)
        y = torch.ones(3, 2, device=device)
        
        # 定义函数 f，根据条件选择 x 或 y 的元素
        def f(x, y):
            return torch.where(x > 0, x, y)

        # 使用 vmap 对函数 f 进行批处理计算，检查是否有运行时错误
        vmap(f)(x, y)

        # 使用随机整数张量 x 进行测试，期望抛出 RuntimeError
        x = torch.randint(0, 2, size=(4, 3), dtype=torch.float)

        def f(t):
            return torch.where(t)

        # 使用 assertRaisesRegex 检查是否抛出指定的 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to vmap over aten::where"
        ):
            vmap(f)(x)

    # 测试阈值操作，使用给定的设备创建随机张量 x
    def test_threshold(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        
        # 使用 F 模块中的阈值函数，对张量 x 进行阈值处理
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    # 允许使用 vmap 回退功能的测试，使用给定的设备创建随机张量 leaf
    @allowVmapFallbackUsage
    def test_inplace_view(self, device):
        leaf = torch.randn(4, 5, requires_grad=True)

        # 定义函数 func，进行非平凡的两次可微分操作
        def func(leaf):
            # 计算张量 leaf 的平方，并创建其视图 base
            base = leaf * leaf
            # 选择 base 的第一个元素，并进行就地的余弦函数操作
            view = base[0]
            view.cos_()
            return view

        # 使用通用的批处理梯度测试函数 _batched_grad_test，测试 func 函数的梯度计算
        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    # 允许使用 vmap 回退功能的测试，使用给定的设备创建随机张量 leaf
    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        # 定义函数 func，进行非平凡的两次可微分操作
        def func(leaf):
            # 计算张量 leaf 的平方，并创建其转置视图 base
            base = leaf * leaf
            view = base.transpose(0, 2)
            # 选择 base 的转置视图的第一个元素，并获取其对角线
            view = view[1]
            view = view.diagonal()
            # 选择对角线的每隔两个元素，并进行就地的余弦函数操作
            view = view[::2]
            view.cos_()
            return view

        # 使用通用的批处理梯度测试函数 _batched_grad_test，测试 func 函数的梯度计算
        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})
    # 定义一个测试函数，用于测试对角线操作的梯度计算
    def test_diagonal(self, device):
        # 创建一个形状为 (4, 5) 的随机张量 x，设备为 device，需要计算梯度
        x = torch.randn(4, 5, device=device, requires_grad=True)
        # 调用 _batched_grad_test 方法测试对 x 的 diagonal 操作的梯度计算
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))

        # 创建一个形状为 (3, 4, 5) 的随机张量 x，设备为 device，需要计算梯度
        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        # 调用 _batched_grad_test 方法测试对 x 的 diagonal 操作的梯度计算
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    # 使用装饰器 allowVmapFallbackUsage 定义一个测试函数，用于测试不相关输出的梯度计算
    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        # 定义常量 B0，表示批处理大小为 3
        B0 = 3
        # 创建一个标量随机张量 x，需要计算梯度
        x = torch.randn([], requires_grad=True)
        # 创建一个标量随机张量 y，需要计算梯度
        y = torch.randn([], requires_grad=True)
        # 创建一个形状为 (B0,) 的随机张量 gy，需要计算梯度
        gy = torch.randn(B0, requires_grad=True)

        # 定义一个函数 vjp，接受参数 v，计算 y 对 x 的梯度乘以 v，返回梯度结果
        def vjp(v):
            # 使用 torch.autograd.grad 计算 y 对 x 的梯度乘以 v，允许返回 None
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            # 如果梯度结果为 None，则返回与 x 相同形状的零张量，否则返回计算的梯度结果
            return torch.zeros_like(x) if res is None else res

        # 对 gy 中的每个元素应用 vjp 函数，返回结果为形状为 (B0, *x.shape) 的张量，设备为 device
        result = vmap(vjp)(gy)
        # 断言结果与形状为 (B0, *x.shape) 的零张量相等
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    # 使用装饰器 allowVmapFallbackUsage 定义一个测试函数，用于测试多个不相关输出的梯度计算
    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        # 定义常量 B0，表示批处理大小为 3
        B0 = 3
        # 创建一个标量随机张量 x，需要计算梯度
        x = torch.randn([], requires_grad=True)
        # 创建一个标量随机张量 y，需要计算梯度
        y = torch.randn([], requires_grad=True)
        # 创建一个形状为 (B0,) 的随机张量 gy，需要计算梯度
        gy = torch.randn(B0, requires_grad=True)

        # 定义一个函数 vjp，接受参数 v，计算 y 对 x 的梯度乘以 v，返回梯度结果
        def vjp(v):
            # 使用 torch.autograd.grad 计算 y 对 x 的梯度乘以 v，允许返回 None
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            # 如果梯度结果为 None，则返回与 x 相同形状的零张量，否则返回计算的梯度结果
            return torch.zeros_like(x) if res is None else res

        # 对 gy[0] 应用 vjp 函数，不保存结果
        _ = vjp(gy[0])
        # 对 gy 中的每个元素应用 vjp 函数，返回结果为形状为 (B0, *x.shape) 的张量，设备为 device
        result = vmap(vjp)(gy)
        # 断言结果与形状为 (B0, *x.shape) 的零张量相等
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))
# 定义函数，根据给定的操作信息对象生成操作的别名列表和原地变体列表
def discover_variants(opinfo):
    # 初始化空列表，用于存储操作的别名
    aliases = []
    # 初始化空列表，用于存储原地操作的变体
    inplace_variants = []

    # 如果存在原地操作的变体，则将其添加到原地变体列表中
    if opinfo.inplace_variant:
        inplace_variants.append(opinfo.inplace_variant)

    # 将主要操作的名称添加到别名列表中
    aliases.append(opinfo.op)
    
    # 遍历操作的所有别名
    for alias in opinfo.aliases:
        # 将每个别名的操作名称添加到别名列表中
        aliases.append(alias.op)
        # 如果当前别名有原地变体，则将其添加到原地变体列表中
        if alias.inplace_variant:
            inplace_variants.append(alias.inplace_variant)

    # 返回操作的别名列表和原地变体列表作为结果
    return aliases, inplace_variants


# TODO: 在接近 torch.vmap x torch.compile 可用时启用此测试。
# 标记为非 DynamoStrict 测试
# @markDynamoStrictTest
@unMarkDynamoStrictTest
# 定义测试类 TestVmapOperatorsOpInfo，继承自 TestCase
class TestVmapOperatorsOpInfo(TestCase):
    # 定义测试函数 vmap_outplace_test，用于测试非原地操作的 vmap 效果
    def vmap_outplace_test(
        self, func, args, kwargs, in_dims, check_shape_only=False, postprocess_fn=None
    ):
        # 遍历计算 vmap 测试所需的数量
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func, args, kwargs, in_dims
        ):
            # 如果指定了后处理函数，则对输出进行后处理
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            # 如果仅检查形状，则比较输出的形状是否相同
            if check_shape_only:
                self.assertEqual(vmap_out.shape, loop_out.shape)
                continue
            # 比较 vmap 输出和循环输出是否相等
            self.assertEqual(vmap_out, loop_out)

    # 定义测试函数 vmap_inplace_test，用于测试原地操作的 vmap 效果
    def vmap_inplace_test(self, func, args, kwargs, in_dims, postprocess_fn=None):
        # 注意：此测试假定第一个参数正在被修改。
        # 这是因为所有基于 OpInfo 的测试都假设如此，但最终需要更健壮的解决方案。
        
        # 如果第一个输入维度为 None，则检查是否正确地引发错误，因为无法进行 vmap 原地操作
        if in_dims[0] is None:
            with self.assertRaises(RuntimeError):
                # 遍历计算 vmap 测试所需的数量，验证是否会正确地引发 RuntimeError
                for _ in compute_quantities_for_vmap_test(
                    func,
                    args,
                    kwargs,
                    in_dims,
                    compute_loop_out=False,
                    clone_inputs=True,
                ):
                    pass
            return
        
        # 遍历计算 vmap 测试所需的数量，验证 vmap 原地操作的输出是否正确
        for vmap_out, loop_out in compute_quantities_for_vmap_test(
            func, args, kwargs, in_dims, clone_inputs=True
        ):
            # 如果指定了后处理函数，则对输出进行后处理
            if postprocess_fn is not None:
                loop_out = postprocess_fn(loop_out)
                vmap_out = postprocess_fn(vmap_out)
            # 比较 vmap 输出和循环输出是否相等
            self.assertEqual(vmap_out, loop_out)

    # 定义 opinfo_vmap_test 函数，用于测试给定操作的 vmap 效果
    def opinfo_vmap_test(
        self,
        device,
        dtype,
        op,
        check_has_batch_rule,
        skip_inplace=(),
        postprocess_fn=None,
    ):
        # 这里可以添加对给定操作的 vmap 测试的实现，暂未提供具体代码

    # 使用 @with_tf32_off 装饰器，用于禁用 TF32 模式
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 使用 @ops 装饰器，用于标记测试所需的操作数据库
    @ops(op_db + additional_op_db + autograd_function_db, dtypes=OpDTypes.any_one)
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_vmap_exhaustive",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            # The following is often flaky, but just on windows.
            # We should investigate if it's actually a problem or not.
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1e-02)},
                device_type="cuda",
            ),
        ),
    )


    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )


    @skipOps(
        "TestVmapOperatorsOpInfo",
        "test_vmap_exhaustive",
        vmap_fail.union(
            {
                # RuntimeError: Batch norm got a batched tensor as input while the running_mean or running_var,
                # which will be updated in place, were not batched.
                xfail("native_batch_norm"),
                xfail("_native_batch_norm_legit"),
                # TODO: implement batching rule
                xfail("_batch_norm_with_update"),
                xfail("tril"),  # Exception not raised on error input
                xfail("triu"),  # Exception not raised on error input
                xfail("as_strided", "partial_views"),
                # RuntimeError: output with shape [4, 4] doesn't match the broadcast shape [1, 4, 4]
                xfail("addcdiv"),
                xfail("addcmul"),
                xfail("clamp"),
                xfail("torch.ops.aten._efficient_attention_forward"),  # outputs ints
                # TypeError: expected Tensor as element 0 in argument 0, but got float
                xfail("item"),
            }
        ),
    )


    def test_vmap_exhaustive(self, device, dtype, op):
        # needs to be fixed
        inplace_failure_list = ()
        self.opinfo_vmap_test(
            device,
            dtype,
            op,
            check_has_batch_rule=False,
            skip_inplace=inplace_failure_list,
        )


    @with_tf32_off
    @ops(op_db + additional_op_db + autograd_function_db, dtypes=OpDTypes.any_one)
    @opsToleranceOverride(
        "TestVmapOperatorsOpInfo",
        "test_op_has_batch_rule",
        (
            tol1(
                "linalg.det",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cuda",
            ),
        ),
    )


    @toleranceOverride(
        {
            torch.float32: tol(atol=1e-04, rtol=1e-04),
            torch.complex64: tol(atol=1e-04, rtol=1e-04),
        }
    )
    def test_op_has_batch_rule(self, device, dtype, op):
        # 定义测试函数 test_op_has_batch_rule，测试是否有批处理规则
        # 定义会导致原位操作失败的运算列表
        inplace_failures = (
            "addbmm",
            "addcdiv",
            "addcmul",
            "addmm",
            "addmv",
            "addr",
            "baddbmm",
            "clamp",
            "conj_physical",
            "cumprod",
            "cumsum",
            "floor_divide",
            "fmod",
            "heaviside",
            "hypot",
            "igamma",
            "igammac",
            "index_copy",
            "ldexp",
            "lerp",
            "neg",
            "nextafter",
            "polygamma",
            "pow",
            "remainder",
            "scatter_add",
            "scatter",
            "square",
            "sub",
            "trunc",
            "xlogy",
        )
        # 调用 self.opinfo_vmap_test 方法，测试操作的 vmap 是否正常工作，
        # 设置 check_has_batch_rule=True，跳过 inplace_failures 中的操作
        self.opinfo_vmap_test(
            device, dtype, op, check_has_batch_rule=True, skip_inplace=inplace_failures
        )

    def test_linalg_svd(self, device):
        # 测试线性代数函数 linalg.svd
        # linalg_svd 返回三个张量元组 (U, S, Vh)
        # 给定相同的输入，可能会返回不同的张量，因为 svd 不是唯一的。
        # 为了验证 svd 的正确性，我们计算 U @ diag(S) @ Vh，并检查 vmap 的输出是否与 for 循环的输出匹配。
        def compute_A(out):
            U, S, Vh = out
            m = U.shape[-1]
            n = Vh.shape[-2]
            diag_S = S.new_zeros(*S.shape[:-1], m, n)
            diag_S.diagonal(offset=0, dim1=-2, dim2=-1).copy_(S)
            return U @ diag_S @ Vh

        # 从 op_db 中获取所有名为 "linalg.svd" 的操作信息
        opinfos = [op for op in op_db if op.name == "linalg.svd"]
        assert len(opinfos) > 0

        # 遍历每个 linalg.svd 操作信息，调用 self.opinfo_vmap_test 方法测试 vmap 是否正常工作
        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    def test_linalg_eigh(self, device):
        # 测试线性代数函数 linalg.eigh
        # linalg_eigh 返回两个张量 (Q, L)
        # 给定相同的输入，可能会返回不同的张量，因为特征分解不是唯一的。
        # 为了验证 eigh 的正确性，我们计算 Q @ diag(L) @ Qh，并检查 vmap 的输出是否与 for 循环的输出匹配。
        def compute_A(out):
            L, Q = out
            n = Q.shape[-1]
            diag_L = L.new_zeros(*L.shape[:-1], n, n)
            diag_L.diagonal(offset=0, dim1=-2, dim2=-1).copy_(L)
            Qh = Q.transpose(-2, -1).conj()
            return Q @ diag_L @ Qh

        # 从 op_db 中获取所有名为 "linalg.eigh" 的操作信息
        opinfos = [op for op in op_db if op.name == "linalg.eigh"]
        assert len(opinfos) > 0

        # 遍历每个 linalg.eigh 操作信息，调用 self.opinfo_vmap_test 方法测试 vmap 是否正常工作
        for op in opinfos:
            self.opinfo_vmap_test(
                device,
                torch.float,
                op,
                check_has_batch_rule=True,
                postprocess_fn=compute_A,
            )

    @skipIfTorchDynamo()
    # 测试 torch.slogdet 函数的向量化映射（vmap）的回退情况
    def test_slogdet(self, device):
        # 没有关于此函数的 OpInfo
        def test():
            # 设定批次大小 B
            B = 2
            # 生成形状为 (B, 5, 5) 的随机张量 x，使用给定设备
            x = torch.randn(B, 5, 5, device=device)
            # 使用 vmap 测试 torch.slogdet 函数的 out-of-place 版本
            self.vmap_outplace_test(torch.slogdet, (x,), {}, (0,))

        # 检查 vmap 的回退情况
        check_vmap_fallback(self, test, torch.slogdet)

    # 测试 Tensor.fill_ 方法的向量化映射（vmap）的回退情况
    def test_fill__Tensor(self, device):
        # 没有关于 fill_.Tensor 的 OpInfo，因此添加额外的测试
        def test():
            # 设定批次大小 B
            B = 2
            # 创建不同参数的元组 args
            args = (torch.randn(B, 3, device=device), torch.randn(B))
            # 使用 vmap 测试 Tensor.fill_ 方法的 inplace 版本
            self.vmap_inplace_test(Tensor.fill_, args, {}, (0, 0))

            args = (torch.randn(3, B, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (-1, 0))

            args = (torch.randn(3, device=device), torch.randn(B))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (None, 0))

            args = (torch.randn(3, B, device=device), torch.randn([]))
            self.vmap_inplace_test(Tensor.fill_, args, {}, (1, None))

        # 检查 vmap 的回退情况
        check_vmap_fallback(self, test, Tensor.fill_)

    # 测试卷积的双向传播（double backward）过程的向量化映射（vmap）的回退情况
    def test_conv_double_backward(self, device):
        # 创建与设备相匹配的随机张量
        images = torch.randn(2, 1, 5, 5, device=device)
        weight = torch.randn(2, 1, 2, 2, device=device)
        bias = torch.randn(2, device=device)
        ggI = torch.randn_like(images)
        ggW = torch.randn_like(weight)
        ggb = torch.randn_like(bias)
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        transposed = False
        output_padding = (0, 0)
        groups = 1
        output_mask = (True, True, True)
        # 计算卷积操作的输出梯度 gO
        gO = torch.randn_like(
            F.conv2d(images, weight, bias, stride, padding, dilation, groups)
        )

        # 准备 op 的参数 args
        args = (
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            images,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        # 获取 op 对象
        op = torch.ops.aten._convolution_double_backward

        # 生成 vmap 回退和详尽测试的生成器
        generator = get_fallback_and_vmap_exhaustive(op, args, {})
        # 检查设备是否为 CUDA 8.6，并设置相应的容差
        is_cuda_sm86 = device.startswith("cuda") and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        atol, rtol = (1e-3, 1e-3) if is_cuda_sm86 else (1e-4, 1e-4)

        def test():
            # 遍历生成器中的每个回退和批处理输出
            for loop_out, batched_out in generator:
                # 使用设定的容差检查每个回退和批处理输出是否相等
                self.assertEqual(loop_out, batched_out, atol=atol, rtol=rtol)

        # 检查 vmap 的回退情况
        check_vmap_fallback(self, test, op)

    # 测试 torch.isnan 函数的向量化映射（vmap）的性能，部分检查梯度传播是否禁用
    def test_isnan(self, device):
        # 使用 functools.partial 部分应用 _vmap_test 函数，检查梯度传播是否禁用
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        # 设定 B, N, C, H, W 的大小
        B, N, C, H, W = 2, 3, 24, 5, 7
        # 操作 op 为 torch.isnan
        op = torch.isnan

        # 生成形状为 (B, N, C, H, W) 的随机张量 x
        x = torch.randn(B, N, C, H, W)
        # 将 x 中大于 0 的元素设置为 NaN
        x[x > 0] = float("nan")
        # 使用 _vmap_test 检查 op 函数的性能
        test(self, op, (x,), in_dims=(0))
    # 定义一个测试函数，用于测试在给定设备上的标量求和操作
    def test_sum_scalar(self, device):
        # 创建一个包含单个浮点数 10.0 的张量，并指定设备
        x = torch.tensor([10.0], device=device)
        # 使用 vmap 对 torch.sum 函数进行矢量化映射，对 x 进行求和
        y = vmap(torch.sum)(x)
        # 断言矢量化映射后的结果与原始张量相等
        self.assertEqual(y, x)

        # 使用 vmap 对 lambda 函数进行矢量化映射，对 x 进行求和
        y = vmap(lambda x: x.sum(0))(x)
        # 断言矢量化映射后的结果与原始张量相等
        self.assertEqual(y, x)

        # 使用 vmap 对 lambda 函数进行矢量化映射，在最后一个维度上对 x 进行求和
        y = vmap(lambda x: x.sum(-1))(x)
        # 断言矢量化映射后的结果与原始张量相等
        self.assertEqual(y, x)

    # 定义一个测试函数，用于测试 torch.isinf 函数在给定设备上的运行情况
    def test_isinf(self, device):
        # 使用 functools.partial 创建一个偏函数 _vmap_test，并关闭梯度传播检查
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        # 定义多维张量的维度大小
        B, N, C, H, W = 2, 3, 24, 5, 7
        # 操作函数选择为 torch.isinf
        op = torch.isinf

        # 创建一个形状为 (B, N, C, H, W) 的随机张量 x
        x = torch.randn(B, N, C, H, W)
        # 将 x 中大于 0 的元素设为正无穷
        x[x > 0] = float("inf")
        # 调用 _vmap_test 函数测试 torch.isinf 的运行情况
        test(self, op, (x,), in_dims=(0))

    # 定义一个测试函数，用于测试 torch.ones_like 和 torch.zeros_like 函数的运行情况
    def test_foo_like(self, device):
        # vfdev-5: 可能可以移除这行。Flake8 报告未使用
        # test = functools.partial(_vmap_test, check_propagates_grad=False)

        # 定义多维张量的维度大小
        B, N, C, H, W = 2, 3, 24, 5, 7
        # 遍历操作函数列表，包括 torch.ones_like 和 torch.zeros_like
        for op in [torch.ones_like, torch.zeros_like]:
            # 创建一个形状为 (B, N, C, H, W) 的随机张量 x
            x = torch.randn(B, N, C, H, W)
            # 使用 vmap 对 op 函数进行矢量化映射，以第一个维度作为输入维度
            vmap(op, in_dims=(0,))(
                x,
            )

    # 定义一个测试函数，用于测试 torch.flatten 函数的运行情况
    def test_flatten(self, device):
        # 使用 functools.partial 创建一个偏函数 _vmap_test，并关闭梯度传播检查
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        # 操作函数选择为 torch.flatten
        op = torch.flatten

        # 创建一个形状为 (2, 3, 4, 5) 的随机张量 x
        x = torch.randn(2, 3, 4, 5)
        # 调用 _vmap_test 函数测试 torch.flatten 的运行情况，设置参数 start_dim=1, end_dim=2
        test(self, op, (x, 1, 2), in_dims=(0, None, None))

    # 定义一个测试函数，用于测试 F.group_norm 函数的运行情况
    def test_group_norm(self, device):
        # 使用 functools.partial 创建一个偏函数 _vmap_test，并关闭梯度传播检查
        test = functools.partial(_vmap_test, check_propagates_grad=False)

        # 定义多维张量的维度大小
        B, N, C, H, W = 2, 3, 24, 5, 7
        # 操作函数选择为 F.group_norm
        op = F.group_norm

        # 创建一个形状为 (B, N, C, H, W) 的随机张量 x，以及形状为 (C,) 的随机权重和偏置张量
        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(C)
        bias = torch.randn(C)
        # 调用 _vmap_test 函数测试 F.group_norm 的运行情况，设置参数 dim=3, weight, bias 的输入维度
        test(self, op, (x, 3, weight, bias), in_dims=(0, None, None, None))

        # 创建一个形状为 (B, N, C, H, W) 的随机张量 x，以及形状为 (B, C) 的随机权重和偏置张量
        x = torch.randn(B, N, C, H, W)
        weight = torch.randn(B, C)
        bias = torch.randn(B, C)
        # 调用 _vmap_test 函数测试 F.group_norm 的运行情况，设置参数 dim=4, weight, bias 的输入维度
        test(self, op, (x, 4, weight, bias), in_dims=(0, None, 0, 0))
    # 定义测试函数 test_index_put，使用给定的设备参数
    def test_index_put(self, device):
        # 定义内部测试函数 test，接受函数 f、输入张量 t、索引 idx 和数值 values 作为参数
        def test(f, t, idx, values):
            # 计算基准结果 base，调用 vmap 对函数 f 进行向量化映射，并验证结果是否与 base 相等
            base = f(t[0], idx[0], values[0])
            self.assertEqual(vmap(f, in_dims=(0, 0, 0))(t, idx, values)[0], base)
            # 调用 vmap 对函数 f 进行向量化映射，并验证部分维度的索引处理
            self.assertEqual(
                vmap(f, in_dims=(0, None, None))(t, idx[0], values[0])[0], base
            )
            self.assertEqual(vmap(f, in_dims=(0, None, 0))(t, idx[0], values)[0], base)
            self.assertEqual(vmap(f, in_dims=(0, 0, None))(t, idx, values[0])[0], base)

        # 定义操作函数 f，用于在张量 x 的指定位置 idx[0] 处设置值为 values[0]
        def f(x, y, z):
            x[y] = z
            return x

        # 创建设备上的随机张量 x、全零长整型张量 y 和随机张量 z
        x = torch.randn(3, 4, 5, device=device)
        y = torch.zeros((3, 2), device=device).long()
        z = torch.randn(3, 2, 5, device=device)
        # 调用测试函数 test，对操作函数 f 进行测试
        test(f, x, y, z)

        # 索引最内层维度
        def f(t, idx, values):
            t[:, idx] = values
            return t

        # 创建全零张量 t、全一张量 values 和索引 idx
        t = torch.zeros((3, 2, 3))
        values = torch.ones((3, 1, 2))
        idx = torch.tensor([[1, 2]]).expand((3, 2))
        # 调用测试函数 test，对操作函数 f 进行测试
        test(f, t, idx, values)

        # 索引中间维度
        def f(t, idx, values):
            t[:, idx, :] = values
            return t

        # 创建全零张量 t、全一张量 values 和索引 idx
        t = torch.zeros((3, 2, 3, 3))
        values = torch.ones((3, 1, 2, 3))
        idx = torch.tensor([[0, 2]]).expand((3, 2))
        # 调用测试函数 test，对操作函数 f 进行测试
        test(f, t, idx, values)

        # 使用切片进行索引
        def f(t, values):
            t[:, :2, :] = values
            return t

        # 计算基准结果 base，调用 vmap 对函数 f 进行向量化映射，并验证结果是否与 base 相等
        base = f(t[0], values[0])
        self.assertEqual(vmap(f, in_dims=(0, 0))(t, values)[0], base)
        self.assertEqual(vmap(f, in_dims=(0, None))(t, values[0])[0], base)

        # 使用 index_put_
        tensor = torch.zeros(3, 3, 4)
        value = torch.ones(3, 2)
        idxs = (
            torch.tensor([[0], [1], [2]]),
            torch.tensor([[0]]),
            torch.tensor([1, 2]),
        )
        expected = torch.index_put_(tensor.clone(), idxs, value)

        # 定义操作函数 f，使用 index_put_ 方法在张量 t 的指定索引 idx 处插入值 v
        def f(t, idx, v):
            torch.index_put_(t, idx, v)
            return t

        # 调用 vmap 对函数 f 进行向量化映射，并验证结果是否与预期值 expected 相等
        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), 0))(tensor, idxs[1:], value), expected
        )
        self.assertEqual(
            vmap(f, in_dims=(0, (None, None), None))(tensor, idxs[1:], value[0]),
            expected,
        )

        # 使用布尔掩码进行索引
        B = 2
        x = torch.randn(1, 3, 3)
        gy = torch.randn(B, 1, 3, 3)

        # 定义操作函数 f，根据布尔掩码将小于 1e-09 的元素置零
        def f(x, gy):
            mask = x < 1e-09
            zeros = torch.zeros([])
            index_put = torch.ops.aten.index_put.default(gy, [mask], zeros)
            return index_put

        # 调用 self.vmap_outplace_test 测试方法，对操作函数 f 进行测试
        self.vmap_outplace_test(f, (x, gy), {}, in_dims=(None, 0))
    # 对批量归一化函数进行测试，根据参数配置进行不同情况下的测试

    # 如果不追踪运行统计信息且不处于训练模式，则直接返回，无需测试
    if not track_running_stats and not training:
        return

    # 使用functools.partial创建一个测试函数test，其中_vmap_test是一个辅助函数，不检查梯度传播
    test = functools.partial(_vmap_test, check_propagates_grad=False)
    
    # BN代表Batch Normalization的缩写，使用torch.nn.BatchNorm2d作为BN的实现
    BN = torch.nn.BatchNorm2d
    
    # 定义集成的模型数量和隐藏维度
    ensemble_size = 10
    hidden_dim = 3
    
    # 使用functional_init_with_buffers函数初始化权重、缓冲区等
    # 返回的元组中包含权重、缓冲区以及其他不需要的值
    weights, buffers, _, _, _ = functional_init_with_buffers(BN, [ensemble_size])(
        hidden_dim, affine=affine, track_running_stats=track_running_stats
    )

    # 构建输入数据列表，包含一个随机生成的张量作为输入
    inputs = [torch.randn(ensemble_size, 32, hidden_dim, 16, 16, device=device)]
    # 输入数据的维度列表
    in_dims = [0]

    # 定义一个函数append，用于将输入数据和其维度添加到inputs和in_dims列表中
    def append(inp, in_dim):
        inputs.append(inp)
        in_dims.append(in_dim)

    # 如果追踪运行统计信息，则从缓冲区中获取running_mean和running_var，并添加到inputs和in_dims中
    if track_running_stats:
        running_mean, running_var, _ = buffers
        append(running_mean.to(device), 0)
        append(running_var.to(device), 0)
    else:
        # 否则将None添加到inputs和in_dims中
        append(None, None)
        append(None, None)

    # 如果使用affine参数，则从权重中获取weight和bias，并添加到inputs和in_dims中
    if affine:
        weight, bias = weights
        append(weight.to(device), 0)
        append(bias.to(device), 0)
    else:
        # 否则将None添加到inputs和in_dims中
        append(None, None)
        append(None, None)

    # 添加training参数到inputs中
    append(training, None)

    # 定义一个操作函数op，用于对输入数据进行批量归一化操作
    def op(inp, running_mean, running_var, weight, bias, training):
        # 调用F.batch_norm进行批量归一化操作，返回归一化结果res
        res = F.batch_norm(inp, running_mean, running_var, weight, bias, training)
        # 如果追踪运行统计信息，则返回归一化结果、running_mean和running_var
        if track_running_stats:
            return res, running_mean, running_var
        # 否则只返回归一化结果res
        return res

    # 调用测试函数test，传入操作函数op、输入数据元组inputs以及输入数据的维度元组in_dims
    test(self, op, tuple(inputs), in_dims=tuple(in_dims))
    def test_inplace_on_view(self, device):
        # 定义一个函数 func，对输入的 leaf 执行一系列操作并返回 view
        def func(leaf):
            # 计算 leaf 的平方作为 base
            base = leaf * leaf
            # 将 base 进行转置操作，并赋值给 view
            view = base.transpose(0, 1)
            # 在 view 的子区域 [2:4, 2:4] 上乘以 2
            view[2:4, 2:4] *= 2
            # 对 view 的子区域 [0:2, 0:2] 对角线上的元素取 sin 函数（原地操作）
            view[0:2, 0:2].diagonal().sin_()
            # 更新 view 为其子区域 [1:3, 1:3]
            view = view[1:3, 1:3]
            # 对 view 的所有元素取 cos 函数（原地操作）
            view.cos_()
            # 返回经过操作后的 view
            return view

        # 定义一个函数 push_vjp，用于执行反向传播
        def push_vjp(leaf, gout):
            # 使用自动微分库计算 func 在 leaf 上的梯度函数
            _, vjp_fn = vjp(func, leaf)
            # 计算梯度函数在 gout 上的梯度
            (result,) = vjp_fn(gout)
            # 返回结果梯度
            return result

        # 生成随机的 leaf 和 gout 张量
        leaf = torch.randn(4, 4, device=device)
        gout = torch.randn(2, 2, device=device)
        args = (leaf, gout)

        # 使用 generate_vmap_inputs 生成批处理参数和输入维度信息
        for (
            batched_args,
            in_dims,
            _,
        ) in generate_vmap_inputs(args, {}):
            if in_dims[1] is None:
                # 如果输入维度的第二个元素为 None，则触发一些复合兼容性问题，跳过本次循环
                continue
            # 对 push_vjp 函数进行 vmap 处理，并进行外部测试
            self.vmap_outplace_test(push_vjp, batched_args, {}, in_dims)

    def test_advanced_indexing(self, device):
        # 定义一个测试函数 test，用于比较循环计算和批处理计算的结果是否一致
        def test(f, args):
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}):
                # 断言循环计算结果与批处理计算结果一致
                self.assertEqual(loop_out, batched_out)

        # 定义三个不同的索引函数 f、f2、f3，分别处理不同的维度索引
        def f(x, idx):
            return x[:, idx]

        def f2(x, idx):
            return x[idx, :]

        def f3(x, idx):
            return x[:, :, idx]

        # 定义输入张量 inps 和索引 idxes
        inps = (
            torch.randn(5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, device=device),
            torch.randn(5, 5, 5, 5, 5, device=device),
        )
        idxes = (
            torch.tensor([0, 1, 2], device=device),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1),
            torch.tensor([0, 1, 2], device=device).reshape(3, 1, 1),
        )
        # 对每一组输入和索引进行组合，测试 f、f2、f3 函数的表现
        for inp, idx in itertools.product(inps, idxes):
            test(f, (inp, idx))
            test(f2, (inp, idx))
            test(f3, (inp, idx))

    def test_nested_advanced_indexing(self, device):
        # 创建一个随机张量 e 和索引 idx
        e = torch.rand(7, 4, device=device)
        idx = torch.tensor([0, 1], device=device).view(2, 1)

        # 定义一个简单的参考实现函数 _fake_vmap，用于与实际函数进行比较
        def _fake_vmap(f, in_dims=0, out_dims=0):
            def w(input):
                r = [f(input.select(in_dims, i)) for i in range(input.size(in_dims))]
                return torch.stack(r, out_dims)

            return w

        # 定义一个使用 vmap 的函数 with_vmap，调用 vmap 处理函数 g
        def with_vmap(_vmap):
            def g(idx_):
                def f(e_):
                    return e_[idx_]

                return _vmap(f, in_dims=1)(e)

            # 使用 vmap 处理 idx
            r = _vmap(g)(idx)
            return r

        # 使用 vmap 处理 e 和 _fake_vmap 处理 e，断言两者结果一致
        a = with_vmap(vmap)
        b = with_vmap(_fake_vmap)
        self.assertEqual(a, b)

    @ops(
        # 筛选包含 "linalg" 的操作并添加到 op_db 和 additional_op_db 中
        filter(lambda op: "linalg" in op.name, op_db + additional_op_db),
        # 指定允许的数据类型为 torch.float
        allowed_dtypes=(torch.float,),
    )
    @skipOps(
        "TestVmapOperatorsOpInfo",  # 跳过指定的测试类
        "test_vmap_linalg_failure_1D_input",  # 跳过指定的测试方法
        {  # 定义用于跳过的测试条件
            xfail("linalg.vector_norm"),  # 标记测试为预期失败，因为可以接受向量输入
            xfail("linalg.norm"),  # 标记测试为预期失败，因为可以接受向量输入
            xfail("linalg.norm", "subgradients_at_zero"),  # 标记测试为预期失败，因为可以接受向量输入
            xfail("linalg.vander"),  # 标记测试为预期失败，因为可以接受向量输入
            skip(  # 跳过测试，因为接受张量输入列表，有特殊的测试方式
                "linalg.multi_dot"
            ),  # 跳过测试方法
            xfail("linalg.vecdot"),  # 标记测试为预期失败，在CUDA上会抛出异常
            # 以下是关于 linalg.diagonal 的说明
            # 在 CUDA 上，使用 vmap 会导致 IndexError: Dimension out of range
            # 详细信息参见链接 https://github.com/pytorch/pytorch/runs/8110653462?check_suite_focus=true
            # 但是在本地运行时通过了测试
            xfail("linalg.diagonal"),  # 标记测试为预期失败
            skip("linalg.matrix_norm", ""),  # 跳过测试方法
            skip("linalg.ldl_solve", ""),  # 跳过测试方法
        },
    )
    def test_vmap_linalg_failure_1D_input(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            if sample.input.dim() != 2 or sample.input.shape[0] == 0:
                continue
            test_input = sample.input[0]  # 使用样本输入避免数值不一致问题
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                op(test_input, *sample.args, **sample.kwargs)

            def op_wrapper(inp):
                return op(inp, *sample.args, **sample.kwargs)

            test_input = test_input.expand(test_input.shape[0], test_input.shape[0])  # 扩展输入以通过 linalg 检查
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                return vmap(op_wrapper)(test_input)

    def test_vmap_multi_dot_failure_1D_input(self):
        inputs = (torch.randn(3, 3), torch.randn(3), torch.randn(3, 3))
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            torch.linalg.multi_dot(inputs)

        inputs = tuple(i.expand(i.shape[0], i.shape[0]) for i in inputs)  # 扩展输入以通过 linalg 检查
        with self.assertRaisesRegex(RuntimeError, "tensor 1 must be 2D but got 1D"):
            return vmap(torch.linalg.multi_dot)(inputs)
    # 定义测试函数以检查在 vmap 调用中出现逃逸的情况
    def test_vmap_escaped_error(self):
        # 初始化逃逸变量为 None
        escaped = None

        # 定义内部函数 f(x)，用于设置逃逸变量并返回 x 的平方
        def f(x):
            nonlocal escaped
            escaped = x
            return x**2

        # 创建一个形状为 [3, 3, 3, 3, 3] 的随机张量 x
        x = torch.randn([3, 3, 3, 3, 3])
        # 对 f 函数进行 vmap 处理，以处理张量 x
        vmap(f)(x)

        # 定义常见的错误信息模板，用于捕获运行时异常
        common_message = (
            r"your tensor may have escaped from inside a function being vmapped.*{0}.*"
        )

        # 检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing")
        ):
            escaped.sin()

        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_tensor_inputs_batch_rule")
        ):
            escaped.sin_()

        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_inplace_plumbing")
        ):
            escaped.mul_(1)

        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("binary_cross_entropy_plumbing")
        ):
            torch.nn.functional.binary_cross_entropy(escaped, torch.zeros([3, 3, 3, 3]))

        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_existing_bdim_all_batch_rule")
        ):
            torch.nn.functional.adaptive_max_pool2d(escaped, output_size=(1, 1))

        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_reduction_batch_rule")
        ):
            escaped.argmin()

        # 创建形状为 [4, 4, 4, 4] 的零张量 a 和形状相同的长整型零张量 b
        a = torch.zeros([4, 4, 4, 4])
        b = torch.zeros([4, 4, 4, 4], dtype=torch.long)
        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("boxed_all_tensors_have_optional_bdim")
        ):
            torch.ops.aten.adaptive_max_pool2d_backward(escaped, a, b)

        # 对 f 函数进行 vmap 处理，传入一个二维整型张量作为参数
        vmap(f)(torch.tensor([[0, 0], [0, 0]], dtype=torch.int))
        # 同上，检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 common_message
        with self.assertRaisesRegex(
            RuntimeError, common_message.format("gen_vmap_plumbing_no_returns")
        ):
            torch.ops.aten._linalg_check_errors(escaped, "linalg.inv", is_matrix=False)

    # 测试带异常检测的 vmap
    def test_vmap_with_anomaly_detection(self):
        # 使用 Torch 的自动梯度异常检测
        with torch.autograd.set_detect_anomaly(True):
            # 创建一个所有元素为 -1 的形状为 [3] 的张量 x
            x = torch.zeros(3) - 1

            # 定义函数 fn(x)，返回 x 的和
            def fn(x):
                return x.sum()

            # 对 fn 函数的梯度进行 vmap 处理，应该得到与 x 形状相同的张量
            per_sample_grad = vmap(grad(fn))(x)
            # 断言每个样本的梯度应该是与 x 形状相同的张量
            self.assertEqual(per_sample_grad, torch.ones_like(x))

            # 定义错误函数 bad_fn(x)，返回对 x 开方后的和
            def bad_fn(x):
                return x.sqrt().sum()

            # 定义错误信息模板，用于捕获运行时异常
            err_msg = "Function 'SqrtBackward0' returned nan values in its 0th output."
            # 检查是否捕获到 RuntimeError 异常，错误信息中包含指定的 err_msg
            with self.assertRaisesRegex(RuntimeError, err_msg):
                vmap(grad(bad_fn))(x)
    # 定义测试方法 `test_searchsorted_bucketize`，接受 `self` 和 `device` 参数
    def test_searchsorted_bucketize(self, device):
        # OpInfo 生成带有批次维度中重复样本的测试。
        # 因此我们显式地测试批次中不同的样本。

        # 定义内部测试方法 `test`
        def test():
            # 创建一个张量 `boundaries`，包含两个子张量，每个子张量有五个元素，存储在指定设备上
            boundaries = torch.tensor(
                [[1, 4, 5, 7, 9], [1, 2, 6, 8, 10]], device=device
            )
            # 创建一个标量张量 `v`，存储在指定设备上
            v = torch.tensor(3, device=device)
            # 调用 `self.vmap_outplace_test` 方法，测试 `torch.searchsorted` 函数
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, None))
            # 调用 `self.vmap_outplace_test` 方法，测试 `torch.bucketize` 函数
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (None, 0))
            
            # 重新定义 `boundaries` 张量，其中一个子张量的第三个元素值变为 4
            boundaries = torch.tensor([[1, 4, 5, 7, 9], [1, 2, 4, 8, 9]], device=device)
            # 创建一个向量张量 `v`，包含两个元素，存储在指定设备上
            v = torch.tensor([3, 4], device=device)
            # 调用 `self.vmap_outplace_test` 方法，测试 `torch.searchsorted` 函数
            self.vmap_outplace_test(torch.searchsorted, (boundaries, v), {}, (0, 0))
            # 调用 `self.vmap_outplace_test` 方法，测试 `torch.bucketize` 函数
            self.vmap_outplace_test(torch.bucketize, (v, boundaries), {}, (0, 0))

        # 调用内部测试方法 `test`
        test()
# 使用装饰器标记为 Dynamo 严格测试的类 TestRandomness
@markDynamoStrictTest
class TestRandomness(TestCase):

    # 重置随机数生成器状态的私有方法
    def _reset_random(self, generator, orig_state, use_generator, seed):
        # 如果 use_generator 为真，则使用传入的原始状态设置生成器状态
        return (
            generator.set_state(orig_state)
            if use_generator
            else torch.manual_seed(seed)
        )

    # 获取图像数据的私有方法
    def _get_image(self, batched_input, batch_size, device):
        # 如果 batched_input 是 "first"，返回指定大小全为 1 的张量，设备为指定设备
        if batched_input == "first":
            return torch.ones([batch_size, 3, 3, 14, 14], device=device)
        # 如果 batched_input 是 "last"，返回指定大小全为 1 的张量，设备为指定设备
        if batched_input == "last":
            return torch.ones([3, 3, 14, 14, batch_size], device=device)
        # 否则，断言 batched_input 为 "none"
        assert batched_input == "none"
        # 返回指定大小全为 1 的张量，设备为指定设备
        return torch.ones([3, 3, 14, 14], device=device)

    # 断言张量的所有切片都相等的私有方法
    def _assert_all_slices_equal(self, tensor):
        # 获取预期的张量切片（第一个切片）
        expected = tensor[0]
        # 断言张量的所有切片都与预期切片相等
        self.assertTrue((tensor == expected).all())

    # 断言张量的所有切片都唯一的私有方法
    def _assert_all_slices_unique(self, tensor):
        # 获取张量的第一维度大小
        B0 = tensor.shape[0]
        # 使用 vmap 对张量进行两两比较，判断每对切片是否全等
        slices_equal = vmap(vmap(lambda x, y: (x == y).all(), (0, None)), (None, 0))(
            tensor, tensor
        )
        # 断言 slices_equal 的形状为 (B0, B0)
        assert slices_equal.shape == (B0, B0)
        # 将对角线上的元素置为 0
        slices_equal.diagonal().zero_()
        # 断言 slices_equal 与全零张量形状相同
        self.assertEqual(slices_equal, torch.zeros_like(slices_equal))

    # 在错误模式下断言在执行函数时抛出异常的私有方法
    def _assert_throws_in_error_mode(self, fn, args, in_dims):
        # 使用上下文管理器断言调用 vmap 执行 fn 时抛出带有特定错误信息的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            vmap(fn, in_dims=in_dims, randomness="error")(*args)

    # 在不同模式下的不同位置随机性中断断言的私有方法
    def _assert_throws_in_different_mode_inplace(self, fn, args, in_dims):
        # 使用上下文管理器断言调用 vmap 执行 fn 时抛出带有特定错误信息的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, r"different inplace randomness on an unbatched tensor"
        ):
            vmap(fn, in_dims=in_dims, randomness="different")(*args)

    # 在批处理张量输入时断言相同模式下的异常抛出的私有方法
    def _assert_throws_in_same_mode_batched(self, fn, args, in_dims):
        # 使用上下文管理器断言调用 vmap 执行 fn 时抛出带有特定错误信息的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"Vmap does not currently support same randomness with a batched tensor input",
        ):
            vmap(fn, in_dims=in_dims, randomness="same")(*args)

    # 返回每个批次字符串的输入维度的私有方法
    def _in_dims(self, *batched_strings):
        # 定义获取每个批次字符串输入维度的内部函数
        def get_in_dim(batched_string):
            # 如果 batched_string 是 "first"，返回 0
            if batched_string == "first":
                return 0
            # 如果 batched_string 是 "last"，返回 -1
            if batched_string == "last":
                return -1
            # 否则，断言 batched_string 为 "none"
            assert batched_string == "none"
            # 返回 None
            return None

        # 将 "first" 添加到 batched_strings 中，用作始终批处理的第一维度虚拟参数
        batched_strings = batched_strings + (
            "first",
        )
        # 返回每个批次字符串输入维度的元组
        return tuple(get_in_dim(batched_string) for batched_string in batched_strings)

    # 使用参数化装饰器指定参数 "randomness" 和 "use_generator" 的测试参数
    @parametrize("randomness", ["same", "different", "error"])
    @parametrize("use_generator", [True, False])
    # 定义一个测试函数，用于测试工厂操作
    def test_factory_ops(self, device, randomness, use_generator):
        # 创建一个 Torch 随机数生成器对象，指定设备
        generator = torch.Generator(device=device)
        # 获取生成器的当前状态
        orig_state = generator.get_state()
        # 根据 use_generator 的值选择不同的参数字典
        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )
        # 定义一组操作列表，每个操作是一个 lambda 函数
        ops = [
            lambda _, shape: torch.randn(shape, **kwargs),  # 生成服从标准正态分布的随机数
            lambda _, shape: torch.rand(shape, **kwargs),   # 生成均匀分布的随机数
            lambda _, shape: torch.randint(100, shape, **kwargs),  # 生成范围在 [0, 100) 的整数随机数
            lambda _, shape: torch.randint(5, 100, shape, **kwargs),  # 生成范围在 [5, 100) 的整数随机数
            lambda _, shape: torch.normal(0.0, 1.0, shape, **kwargs),  # 生成服从正态分布的随机数
        ]
        # 定义常量 B0 和 shape
        B0 = 4
        shape = (3, 3)
        seed = 1234567

        # 遍历每个操作函数
        for op in ops:
            # 生成一个传递给操作函数的随机数张量 passed
            passed = torch.randn(B0, device=device)
            # 如果 randomness 参数为 "error"，则调用特定错误模式下的断言函数并返回
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, shape), in_dims=(0, None)
                )
                return

            # 重置随机数生成器的状态，恢复到初始状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 对操作函数使用 vmap 进行映射操作，生成 vmap_result
            vmap_result = vmap(op, in_dims=(0, None), randomness=randomness)(
                passed, shape
            )

            # 再次重置随机数生成器的状态，为了进行后续比较操作
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 如果 randomness 参数为 "different"，则进行不同的断言和预期结果计算
            if randomness == "different":
                expected = op(passed, [B0, *shape])
                self._assert_all_slices_unique(vmap_result)  # 断言 vmap_result 的所有切片是唯一的
                self.assertEqual(vmap_result, expected)  # 比较 vmap_result 和预期结果是否相等
            else:
                # 否则，进行相同的断言和预期结果计算
                expected = op(passed, shape)
                self._assert_all_slices_equal(vmap_result)  # 断言 vmap_result 的所有切片都相等
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)  # 逐个比较 vmap_result 的每个元素和预期结果是否相等
    # 定义一个测试方法，用于测试 torch.randperm 函数在不同设备和随机模式下的行为
    def test_randperm(self, device, randomness, use_generator):
        # 设置一个特殊的批次大小 B0
        B0 = 4
        # 设置随机数种子
        seed = 1234567
        # 生成一个在指定设备上的随机张量
        passed = torch.randn(B0, device=device)

        # 设置全局随机数种子
        torch.manual_seed(seed)
        # 根据设备创建一个随机数生成器
        generator = torch.Generator(device=device)
        # 获取当前随机数生成器的状态
        orig_state = generator.get_state()

        # 根据是否使用生成器创建不同的参数字典
        kwargs = (
            {"device": device, "generator": generator}
            if use_generator
            else {"device": device}
        )

        # 如果随机模式为 "error"，则测试是否能捕获运行时错误异常
        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                # 对传入的 passed 张量进行 vmap 映射，应该触发 RuntimeError 异常
                vmap(lambda _: torch.randperm(10, **kwargs), randomness=randomness)(
                    passed
                )
            return

        # 使用 vmap 函数对 torch.randperm 函数进行映射，根据不同的随机模式
        vmap_result = vmap(
            lambda _: torch.randperm(10, **kwargs), randomness=randomness
        )(passed)
        # 恢复生成器的原始状态
        generator = generator.set_state(orig_state)
        # 恢复全局随机数种子状态
        torch.manual_seed(seed)

        # 如果随机模式为 "different"，则逐一比较每个元素与预期结果的一致性
        if randomness == "different":
            for i in range(B0):
                expected = torch.randperm(10, **kwargs)
                # 在 CUDA 上，检查是否需要验证所有切片的唯一性
                if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                    self._assert_all_slices_unique(vmap_result)
                else:
                    # 比较 vmap 结果中的第 i 个张量与预期的张量是否相等
                    self.assertEqual(vmap_result[i], expected)
        else:
            # 对于其他随机模式，比较整体的 vmap 结果与预期的张量是否相等
            expected = torch.randperm(10, **kwargs)
            # 在 CUDA 上，检查是否需要验证所有切片的相等性
            if TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda":
                self._assert_all_slices_equal(vmap_result)
            else:
                for i in range(B0):
                    # 比较 vmap 结果中的第 i 个张量与预期的张量是否相等
                    self.assertEqual(vmap_result[i], expected)
    # 定义一个测试函数，用于测试dropout操作的效果
    def test_dropout(self, device, randomness, batched_input):
        # 定义dropout操作函数op，返回一个与输入张量相同形状的张量，应用dropout
        def op(t, ignored):
            return torch.nn.functional.dropout(torch.ones_like(t), training=True)

        # 设置批量大小B0为4
        B0 = 4
        # 生成一个形状为(B0,)的随机张量always_batched
        always_batched = torch.randn((B0,))
        # 调用_get_image方法，获取输入batched_input对应的图像数据passed
        passed = self._get_image(batched_input, B0, device)
        # 调用_in_dims方法，获取batched_input的输入维度in_dims
        in_dims = self._in_dims(batched_input)

        # 如果randomness为"error"
        if randomness == "error":
            # 使用assertRaisesRegex检测是否捕获到RuntimeError异常，异常消息为"called random operation while in randomness error mode"
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                # 使用vmap对op进行矢量化映射，对passed和always_batched进行操作
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        # 对op进行vmap矢量化映射，处理passed和always_batched，返回结果存入vmap_result
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # 检查随机性是否在预期范围内，p_estimate应接近0.5
        p_estimate = vmap_result.mean() / 2
        self.assertTrue(p_estimate < 0.75)
        self.assertTrue(p_estimate > 0.25)

        # 如果randomness为"different"，调用_assert_all_slices_unique方法检查所有切片是否唯一
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        # 否则，确认randomness为"same"，调用_assert_all_slices_equal方法检查所有切片是否相等
        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    # 使用参数化装饰器，测试alpha_dropout操作
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    def test_alpha_dropout(self, device, randomness, batched_input):
        # 定义alpha_dropout操作函数op，返回一个与输入张量相同形状的张量，应用alpha_dropout
        def op(t, ignored):
            return torch.nn.functional.alpha_dropout(torch.ones_like(t), training=True)

        # 设置批量大小B0为4
        B0 = 4
        # 生成一个形状为(B0,)的随机张量always_batched
        always_batched = torch.randn((B0,))
        # 调用_get_image方法，获取输入batched_input对应的图像数据passed
        passed = self._get_image(batched_input, B0, device)
        # 调用_in_dims方法，获取batched_input的输入维度in_dims
        in_dims = self._in_dims(batched_input)

        # 如果randomness为"error"
        if randomness == "error":
            # 使用assertRaisesRegex检测是否捕获到RuntimeError异常，异常消息为"called random operation while in randomness error mode"
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                # 使用vmap对op进行矢量化映射，对passed和always_batched进行操作
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return

        # 对op进行vmap矢量化映射，处理passed和always_batched，返回结果存入vmap_result
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # 如果randomness为"different"，调用_assert_all_slices_unique方法检查所有切片是否唯一
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        # 否则，确认randomness为"same"，调用_assert_all_slices_equal方法检查所有切片是否相等
        assert randomness == "same"
        self._assert_all_slices_equal(vmap_result)

    # 参数化装饰器，用于测试alpha_dropout操作，增加了维度参数dim
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("dim", [2, 3])
    # 定义测试方法，用于测试特征丢弃功能
    def test_feature_dropout(self, device, randomness, batched_input, dim):
        # 定义操作函数 `op`，根据维度选择相应的 dropout 函数
        def op(t, ignored):
            f = (
                torch.nn.functional.dropout2d  # 如果维度是 2，选择二维 dropout 函数
                if dim == 2
                else torch.nn.functional.dropout3d  # 如果维度是 3，选择三维 dropout 函数
            )
            # 对输入的张量 t 应用 dropout 操作，并确保处于训练模式
            return f(torch.ones_like(t), training=True)

        # 定义批次大小 B0
        B0 = 4
        # 生成一个随机张量 `always_batched`，用于测试
        always_batched = torch.randn((B0,))
        # 使用自定义方法 `_get_image` 获取处理后的图像张量 `passed`
        passed = self._get_image(batched_input, B0, device)
        
        # 如果维度是 3，根据 `batched_input` 的不同，对 `passed` 进行维度扩展
        if dim == 3:
            unsqueeze_dim = -2 if batched_input == "last" else -1
            passed = passed.unsqueeze(unsqueeze_dim)
        
        # 获取输入维度信息 `in_dims`
        in_dims = self._in_dims(batched_input)

        # 如果 `randomness` 为 "error"，验证是否捕获到 RuntimeError 异常
        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                # 使用 `vmap` 对操作函数 `op` 进行映射，并传入参数 `passed` 和 `always_batched`
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return
        
        # 使用 `vmap` 对操作函数 `op` 进行映射，并传入参数 `passed` 和 `always_batched`
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # 检查结果中的 "feature" 模式
        # 根据维度 `dim` 设置相应的维度索引 `dims`
        dims = [-1, -2] if dim == 2 else [-1, -2, -3]
        # 计算平面元素的数量
        planes_numel = (
            2
            * vmap_result.numel()
            / (vmap_result.shape[0] * vmap_result.shape[1] * vmap_result.shape[2])
        )
        # 对 `vmap_result` 进行求和操作，以获取各个平面的总和
        planes = vmap_result.sum(dims)
        # 检查每个平面是否符合条件，将结果存储在 `result` 中
        result = (planes == 0) ^ (planes == planes_numel)
        # 使用断言检查 `result` 是否全为 True
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        # 如果 `randomness` 为 "different"，则调用 `_assert_all_slices_unique` 方法验证所有切片是否唯一
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return
        
        # 否则，确保 `randomness` 为 "same"
        assert randomness == "same"
        # 调用 `_assert_all_slices_equal` 方法验证所有切片是否相等
        self._assert_all_slices_equal(vmap_result)

    # 参数化测试方法，分别测试不同的 `randomness` 和 `batched_input` 参数组合
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    # 定义测试函数，用于测试 feature_alpha_dropout 函数
    def test_feature_alpha_dropout(self, device, randomness, batched_input):
        
        # 定义操作函数 op，使用 feature_alpha_dropout 对输入进行操作
        def op(t, ignored):
            return torch.nn.functional.feature_alpha_dropout(
                torch.ones_like(t), training=True
            )

        # 定义批量大小 B0
        B0 = 4
        
        # 生成一个随机张量 always_batched，形状为 (B0,)
        always_batched = torch.randn((B0,))
        
        # 调用 _get_image 方法获取输入张量 passed，形状由 batched_input 和 device 决定
        passed = self._get_image(batched_input, B0, device)
        
        # 根据 batched_input 决定是否在 passed 上增加一个维度
        unsqueeze_dim = -2 if batched_input == "last" else -1
        passed = passed.unsqueeze(unsqueeze_dim)
        
        # 获取输入维度信息 in_dims
        in_dims = self._in_dims(batched_input)

        # 如果 randomness 为 "error"，测试是否能捕获 RuntimeError 异常
        if randomness == "error":
            with self.assertRaisesRegex(
                RuntimeError, r"called random operation while in randomness error mode"
            ):
                # 使用 vmap 对 op 进行批处理映射，并传入 passed 和 always_batched
                vmap(op, randomness=randomness, in_dims=in_dims)(passed, always_batched)
            return
        
        # 使用 vmap 对 op 进行批处理映射，并传入 passed 和 always_batched，得到 vmap_result
        vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
            passed, always_batched
        )

        # 检查 alpha dropout 的正确性，这里给出了一个参考链接
        # https://github.com/pytorch/pytorch/issues/74004
        # 检查 "feature" 模式的结果
        dims = [-1, -2, -3]
        planes = vmap_result.sum(dims)
        max_elt = planes.max()
        min_elt = planes.min()
        result = (planes == min_elt) ^ (planes == max_elt)
        
        # 使用断言验证结果与预期一致
        self.assertEqual(result, torch.ones_like(result, dtype=torch.bool))

        # 如果 randomness 为 "different"，调用 _assert_all_slices_unique 验证所有切片都是唯一的
        if randomness == "different":
            self._assert_all_slices_unique(vmap_result)
            return

        # 否则，确认 randomness 为 "same"
        assert randomness == "same"
        # 调用 _assert_all_slices_equal 验证所有切片都相等
        self._assert_all_slices_equal(vmap_result)
    # 定义一个测试函数，用于测试类似函数的行为，接受设备、随机性和批量输入作为参数
    def test_like_functions(self, device, randomness, batched_input):
        # 设定随机种子
        seed = 1234567
        # 支持的操作列表，每个操作是一个 lambda 函数
        supported_ops = [
            lambda t, _: torch.randint_like(t, 20),  # 生成类似于 t 的随机整数张量
            lambda t, _: torch.randint_like(t, 0, 20),  # 生成类似于 t 的指定范围内的随机整数张量
            lambda t, _: torch.rand_like(t),  # 生成类似于 t 的随机浮点数张量
            lambda t, _: torch.randn_like(t),  # 生成类似于 t 的标准正态分布随机张量
        ]
        # 初始批量大小设定为 4
        B0 = 4

        # 对于每个支持的操作，进行迭代
        for op in supported_ops:
            # 生成大小为 B0 的随机标准正态分布张量
            always_batched = torch.randn(B0)
            # 使用 self._get_image 方法获取处理后的图像张量
            passed = self._get_image(batched_input, B0, device)
            # 获取输入张量的维度信息
            in_dims = self._in_dims(batched_input)

            # 如果随机性设置为 "error"，则断言引发 RuntimeError 异常
            if randomness == "error":
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"called random operation while in randomness error mode",
                ):
                    # 使用 vmap 对操作 op 进行批量映射，并传入相关参数
                    vmap(op, in_dims=in_dims, randomness=randomness)(
                        passed, always_batched
                    )
                return

            # 设置随机种子
            torch.manual_seed(seed)
            # 使用 vmap 对操作 op 进行批量映射，并传入相关参数，获取结果
            vmap_result = vmap(op, randomness=randomness, in_dims=in_dims)(
                passed, always_batched
            )

            # 重新设置随机种子
            torch.manual_seed(seed)

            # 如果 batched_input 设置为 "last"，则调整 passed 的维度
            if batched_input == "last":
                passed = passed.movedim(-1, 0)
            
            # 如果随机性设置为 "different"
            if randomness == "different":
                # 如果 batched_input 设置为 "none"，则扩展 passed 的维度
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                # 计算预期结果
                expected = op(passed, 0)

                # 断言所有切片的唯一性
                self._assert_all_slices_unique(vmap_result)
                # 如果不是在 CUDA 上使用 torchdynamo，则比较预期结果与 vmap 结果
                if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                    self.assertEqual(expected, vmap_result)
                return

            # 断言随机性为 "same"
            assert randomness == "same"
            # 如果 batched_input 不是 "none"，则只取 passed 的第一个元素
            if batched_input != "none":
                passed = passed[0]
            # 计算预期结果
            expected = op(passed, 0)
            # 断言所有切片的相等性
            self._assert_all_slices_equal(vmap_result)
            # 如果不是在 CUDA 上使用 torchdynamo，则逐一比较预期结果与 vmap 结果
            if not (TEST_WITH_TORCHDYNAMO and torch.device(device).type == "cuda"):
                for i in range(B0):
                    self.assertEqual(expected, vmap_result[i])
    ):
        # 在指定设备上创建一个随机数生成器对象
        generator = torch.Generator(device=device)
        # 保存生成器的初始状态
        orig_state = generator.get_state()
        # 如果 use_generator 为 True，则将生成器作为参数传递
        kwargs = {"generator": generator} if use_generator else {}
        # 定义一系列操作函数
        ops = [
            lambda t, _: t.random_(**kwargs),            # 在张量上生成随机数
            lambda t, _: t.random_(100, **kwargs),       # 在张量上生成指定范围的随机数
            lambda t, _: t.random_(-5, 100, **kwargs),   # 在张量上生成指定范围的随机数
            lambda t, _: t.normal_(**kwargs),            # 在张量上生成服从正态分布的随机数
            lambda t, _: t.bernoulli_(**kwargs),         # 在张量上生成伯努利分布的随机数
            lambda t, _: t.cauchy_(**kwargs),            # 在张量上生成柯西分布的随机数
            lambda t, _: t.exponential_(**kwargs),       # 在张量上生成指数分布的随机数
            lambda t, _: t.geometric_(0.5, **kwargs),    # 在张量上生成几何分布的随机数
            lambda t, _: t.log_normal_(**kwargs),        # 在张量上生成对数正态分布的随机数
            lambda t, _: t.uniform_(**kwargs),           # 在张量上生成均匀分布的随机数
        ]
        # 定义常量 B0
        B0 = 4
        # 设置随机种子
        seed = 1234567
        # 获取输入张量的维度信息
        in_dims = self._in_dims(batched_input)

        # 对每个操作函数进行迭代
        for op in ops:
            # 因为会进行原地更新，所以先克隆输入
            always_batched = torch.randn(B0, device=device)
            passed = self._get_image(batched_input, B0, device)
            passed_expected = passed.clone()

            # 如果 randomness 为 "error"，则在错误模式下断言会抛出异常
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            # 如果 randomness 为 "different" 并且 batched_input 为 "none"，则在不同模式下断言会抛出异常
            if randomness == "different" and batched_input == "none":
                self._assert_throws_in_different_mode_inplace(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            # 重置随机数生成器到初始状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 对操作进行矢量化映射，处理张量的每个维度上的随机数
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            # 如果 batched_input 为 "last"，则移动期望的张量维度
            if batched_input == "last":
                passed_expected = passed_expected.movedim(-1, 0)
            # 再次重置随机数生成器到初始状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 如果 randomness 为 "different"，则对比预期结果和实际结果是否相等
            if randomness == "different":
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_unique(vmap_result)
                self.assertEqual(vmap_result, expected)
            else:
                # 如果 batched_input 不是 "none"，则克隆预期的张量
                if batched_input != "none":
                    passed_expected = passed_expected[
                        0
                    ].clone()  # bug in pytorch, normal_ on views doesn't work
                expected = op(passed_expected, always_batched)
                self._assert_all_slices_equal(vmap_result)
                # 对比每个批次的结果是否与预期一致
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    # 参数化测试用例，测试在原地生成伯努利分布随机数时的行为
    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    @parametrize("batched_probability", ["first", "last", "none"])
    def test_bernoulli_in_place(
        self, device, use_generator, randomness, batched_input, batched_probability
    ):
        # 设置初始批次大小为 4
        B0 = 4
        # 设定随机种子
        seed = 1234567
        # 创建 Torch 随机数生成器对象，并指定设备
        generator = torch.Generator(device=device)
        # 保存生成器的当前状态
        orig_state = generator.get_state()
        # 如果使用生成器，则传递给参数字典
        kwargs = {"generator": generator} if use_generator else {}
        # 计算输入的维度信息
        in_dims = self._in_dims(batched_input, batched_probability)

        # 定义操作函数，用于执行伯努利采样
        def op(t, p, ignored):
            return t.bernoulli_(p, **kwargs)

        # 由于涉及原地更新，克隆输入数据
        always_batched = torch.randn(B0, device=device)
        # 获取图像输入并确保批次化
        input = self._get_image(batched_input, B0, device)
        # 克隆输入作为预期输入
        input_expected = input.clone()
        # 获取概率并确保批次化，并减去0.5以调整到[-0.5, 0.5]范围内
        probability = self._get_image(batched_probability, B0, device) - 0.5

        # 处理随机性为"error"的情况
        if randomness == "error":
            self._assert_throws_in_error_mode(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        # 处理随机性为"same"且输入批次化不为"none"的情况
        if randomness == "same" and batched_probability != "none":
            self._assert_throws_in_same_mode_batched(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return
        # 处理批次化输入为"none"且概率批次化不为"none"的情况
        if batched_input == "none" and batched_probability != "none":
            # 设置异常信息的正则表达式
            regex = r"there exists a Tensor `other` in extra_args that has more elements than `self`"
            # 断言抛出运行时错误，并检查异常消息
            with self.assertRaisesRegex(RuntimeError, regex):
                vmap(op, in_dims=in_dims, randomness=randomness)(
                    input, probability, always_batched
                )
            return
        # 处理随机性为"different"且输入非批次化的情况
        if randomness == "different" and batched_input == "none":
            self._assert_throws_in_different_mode_inplace(
                op, (input, probability, always_batched), in_dims=in_dims
            )
            return

        # 重置随机状态为初始状态
        self._reset_random(generator, orig_state, use_generator, seed)
        # 使用vmap对操作进行批次映射
        vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
            input, probability, always_batched
        )

        # 再次重置随机状态为初始状态
        self._reset_random(generator, orig_state, use_generator, seed)
        # 如果批次输入为"last"，将预期输入的维度移动到第一维
        if batched_input == "last":
            input_expected = input_expected.movedim(-1, 0)
        # 如果概率批次化为"last"，将概率的维度移动到第一维
        if batched_probability == "last":
            probability = probability.movedim(-1, 0)
        # 处理随机性为"different"的情况
        if randomness == "different":
            # 计算预期输出
            expected = op(input_expected, probability, always_batched)
            # 断言vmap结果的所有切片唯一
            self._assert_all_slices_unique(vmap_result)
            # 断言vmap结果等于预期输出
            self.assertEqual(vmap_result, expected)
        else:
            # 如果输入批次化不为"none"，只使用第一个批次的预期输入
            if batched_input != "none":
                input_expected = input_expected[0]
            # 计算预期输出
            expected = op(input_expected, probability, always_batched)
            # 断言vmap结果的所有切片相等
            self._assert_all_slices_equal(vmap_result)
            # 逐个检查每个批次的vmap结果与预期输出的相等性
            for i in range(B0):
                self.assertEqual(vmap_result[i], expected)
    # 定义一个测试函数，用于测试生成随机数的函数在不同设置下的行为
    def test_random_binary_out_of_place(
        self, device, use_generator, randomness, batched_input, batched_other
    ):
        # 在指定设备上创建一个随机数生成器对象
        generator = torch.Generator(device=device)
        # 获取随机数生成器的当前状态
        orig_state = generator.get_state()
        # 根据是否使用生成器，准备参数字典
        kwargs = {"generator": generator} if use_generator else {}
        # 定义一组操作函数列表，每个函数接受输入张量、其他张量和参数字典作为输入
        ops = [
            lambda t, o, _: torch.normal(t, o, **kwargs),  # 正态分布操作
            lambda t, o, _: torch.binomial(t, (o - 0.5), **kwargs),  # 二项分布操作
        ]

        # 设置常量 B0 为 4
        B0 = 4
        # 设定随机种子
        seed = 1234567
        # 获取输入张量的维度信息
        in_dims = self._in_dims(batched_input, batched_other)

        # 对于每个操作函数进行循环测试
        for op in ops:
            # 创建一个始终批处理的张量，形状为 B0，并且位于指定设备上
            always_batched = torch.randn(B0, device=device)
            # 获取输入图像张量，可能经过批处理处理
            input = self._get_image(batched_input, B0, device)
            # 获取其他图像张量，可能经过批处理处理
            other = self._get_image(batched_other, B0, device)

            # 如果随机性设置为 "error"，则验证在错误模式下是否抛出异常
            if randomness == "error":
                self._assert_throws_in_error_mode(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return  # 结束当前测试函数

            # 如果随机性设置为 "same"，并且输入或其他张量经过了批处理
            if randomness == "same" and (
                batched_input != "none" or batched_other != "none"
            ):
                self._assert_throws_in_same_mode_batched(
                    op, (input, other, always_batched), in_dims=in_dims
                )
                return  # 结束当前测试函数

            # 重置随机数生成器到初始状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 对操作进行向量化映射，返回结果张量
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                input, other, always_batched
            )

            # 如果批处理方式为 "last"，则移动输入或其他张量的维度
            if batched_input == "last":
                input = input.movedim(-1, 0)
            if batched_other == "last":
                other = other.movedim(-1, 0)

            # 再次重置随机数生成器到初始状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)

            # 如果随机性设置为 "different"
            if randomness == "different":
                # 如果输入未经批处理，则扩展输入张量的维度
                if batched_input == "none":
                    input = input.expand(B0, *input.shape)
                # 计算预期结果
                expected = op(input, other, always_batched)
                # 断言所有切片在向量化映射结果中都是唯一的
                self._assert_all_slices_unique(vmap_result)
                # 断言向量化映射结果等于预期结果
                self.assertEqual(vmap_result, expected)
            else:
                # 否则，断言批处理方式为 "none"，即没有批处理
                assert batched_input == "none" and batched_other == "none"
                # 计算预期结果
                expected = op(input, other, always_batched)
                # 断言所有切片在向量化映射结果中都相等
                self._assert_all_slices_equal(vmap_result)
                # 遍历每个元素，断言向量化映射结果等于预期结果
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)

    # 参数化装饰器，用于自动化测试的参数化设置
    @parametrize("use_generator", [True, False])
    @parametrize("randomness", ["error", "same", "different"])
    @parametrize("batched_input", ["first", "last", "none"])
    # 定义测试函数，测试生成随机数的一元操作函数在不同设置下的行为
    def test_random_unary_out_of_place(
        self, device, use_generator, randomness, batched_input
    ):
        # 创建一个 Torch 随机数生成器对象，使用给定的设备
        generator = torch.Generator(device=device)
        # 保存生成器的当前状态，以便稍后重置
        orig_state = generator.get_state()
        # 根据是否使用生成器，准备传递给操作函数的参数
        kwargs = {"generator": generator} if use_generator else {}
        # 定义一组操作函数列表，每个函数接受两个参数 (t, _)，并返回一个 Torch 张量
        ops = [
            lambda t, _: torch.normal(0.0, torch.abs(t), **kwargs),  # 正态分布
            lambda t, _: torch.normal(t, 1.0, **kwargs),             # 正态分布
            lambda t, _: torch.bernoulli(t - 0.5, **kwargs),         # 伯努利分布
            lambda t, _: torch.bernoulli(t, 0.5, **kwargs),          # 伯努利分布
            lambda t, _: torch._standard_gamma(t, **kwargs),         # Gamma 分布
            lambda t, _: torch._sample_dirichlet(t, **kwargs),       # Dirichlet 分布
            lambda t, _: torch.poisson(t, **kwargs),                 # 泊松分布
        ]

        B0 = 4
        seed = 1234567
        # 获取输入数据的维度信息
        in_dims = self._in_dims(batched_input)

        # 遍历操作函数列表
        for op in ops:
            # 生成一个随机张量，用于模拟输入
            always_batched = torch.randn(B0, device=device)
            # 获取处理后的输入数据
            passed = self._get_image(batched_input, B0, device)

            # 根据随机性设置，执行不同的测试模式
            if randomness == "error":
                # 在错误模式下，验证操作 op 是否抛出异常
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return  # 测试完成，退出函数
            if randomness == "same" and batched_input != "none":
                # 在相同模式下，验证操作 op 的输出是否保持一致
                self._assert_throws_in_same_mode_batched(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return  # 测试完成，退出函数

            # 重置随机数生成器状态，确保每次测试开始时的随机数一致
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 使用 vmap 函数对操作 op 进行矢量化映射，生成结果
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            # 再次重置随机数生成器状态
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            if randomness == "different":
                # 在不同模式下，根据输入数据的情况调整 passed 张量的形状
                if batched_input == "none":
                    passed = passed.expand(B0, *passed.shape)
                if batched_input == "last":
                    passed = passed.movedim(-1, 0)
                # 计算期望的操作结果
                expected = op(passed, always_batched)
                # 验证 vmap 结果中的所有切片都是唯一的
                self._assert_all_slices_unique(vmap_result)
                # 使用断言验证 vmap_result 是否等于期望结果
                self.assertEqual(vmap_result, expected)
            else:
                # 在相同模式下，计算期望的操作结果
                expected = op(passed, always_batched)
                # 验证 vmap 结果中的所有切片都是相等的
                self._assert_all_slices_equal(vmap_result)
                # 使用循环逐一验证每个元素是否等于期望的结果
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)
        ):
            # 定义一个函数：将输入扁平化处理，根据批处理调用和批处理位置确定最终维度
            def flatten_input(input, batch_call, batch_location):
                if batch_call and batch_location != "none":
                    final_size = 3  # 最终维度为 [B0, B, N]
                elif not batch_call and batch_location == "none":
                    final_size = 1  # 最终维度为 [N]
                else:
                    final_size = 2  # 最终维度为 [B0, N] 或 [B, N]

                start_idx = final_size - 1
                end_idx = -1
                if batch_location == "last":
                    start_idx -= 1
                    end_idx -= (
                        1  # 使用负索引以获得正确的最终大小
                    )

                # 调用 PyTorch 的 flatten 函数来扁平化输入
                ret = input.flatten(start_idx, end_idx)
                assert ret.dim() == final_size
                return ret

            # 定义一个操作函数，使用输入 input 和 kwargs 来进行多项式抽样
            def op(input, _):
                return torch.multinomial(input, 10, **kwargs)

            # 在指定的设备上创建一个 torch 生成器对象
            generator = torch.Generator(device=device)
            # 获取生成器的当前状态
            orig_state = generator.get_state()
            # 根据 use_generator 变量决定是否在 kwargs 中添加生成器参数
            kwargs = {"generator": generator} if use_generator else {}

            # 定义一个常量 B0 并设置随机种子
            B0 = 4
            seed = 1234567
            # 获取输入的维度信息
            in_dims = self._in_dims(batched_input)

            # 在指定设备上生成一个随机张量 always_batched
            always_batched = torch.randn(B0, device=device)
            # 调用 self._get_image 方法获取处理后的 batched_input
            passed = self._get_image(batched_input, B0, device)
            # 将获取的输入 passed 进行扁平化处理
            passed = flatten_input(passed, batched_call, batched_input)

            # 根据 randomness 的不同值进行不同的处理
            if randomness == "error":
                # 在错误模式下调用 op 函数，并断言会抛出异常
                self._assert_throws_in_error_mode(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return
            if randomness == "same" and batched_input != "none":
                # 在相同模式下调用 op 函数，并断言所有结果相等
                self._assert_throws_in_same_mode_batched(
                    op, (passed, always_batched), in_dims=in_dims
                )
                return

            # 重设生成器的随机状态，根据不同的参数重新初始化生成器
            generator = self._reset_random(generator, orig_state, use_generator, seed)
            # 使用 vmap 函数对 op 函数进行向量化映射处理，得到结果 vmap_result
            vmap_result = vmap(op, in_dims=in_dims, randomness=randomness)(
                passed, always_batched
            )

            # 再次重设生成器的随机状态，以确保与前面一致
            generator = self._reset_random(generator, orig_state, use_generator, seed)

            # 根据 randomness 的不同值进行不同的处理
            if randomness == "different":
                if batched_input == "none":
                    # 如果 batched_input 为 "none"，则扩展 passed 的维度
                    passed = passed.expand(B0, *passed.shape)
                if batched_input == "last":
                    # 如果 batched_input 为 "last"，则调整 passed 的维度
                    passed = passed.movedim(-1, 0)
                # 根据 batched_call 的值，选择是否进行进一步的扁平化处理
                orig_passed_size = passed.shape[:2] if batched_call else passed.shape[:1]
                passed = passed.flatten(0, 1) if batched_call else passed
                # 调用 op 函数得到期望的结果 expected
                expected = op(passed, always_batched)
                expected = expected.reshape(*orig_passed_size, 10)
                # 断言 vmap_result 的所有切片都是唯一的
                self._assert_all_slices_unique(vmap_result)
                # 断言 vmap_result 与 expected 相等
                self.assertEqual(vmap_result, expected)
            else:
                # 如果 randomness 不为 "different"，则直接调用 op 函数得到期望的结果 expected
                expected = op(passed, always_batched)
                # 断言 vmap_result 的所有切片都相等
                self._assert_all_slices_equal(vmap_result)
                # 对于每个 B0，断言 vmap_result[i] 与 expected[i] 相等
                for i in range(B0):
                    self.assertEqual(vmap_result[i], expected)
    def test_unsupported_random(self, device):
        # 生成一个形状为 (3,) 的张量 x，其元素服从标准正态分布，存储在指定设备上
        x = torch.randn(3, device=device)
        # 计算张量 x 的绝对值，结果保存在 y 中
        y = x.abs()
        # 计算张量 x 的绝对值，结果保存在 z 中（与 y 相同）
        z = x.abs()
        # 断言以下代码块抛出 RuntimeError 异常，并且异常消息包含 "calling out variants"
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):
            # 定义一个函数 f(x)，调用 torch.randn 生成形状为 (3,) 的张量，设备与 y 相同
            def f(x):
                return torch.randn(3, device=device, out=y)
            # 对函数 f 进行 vmap 映射，其中 randomness 参数设置为 "same"
            vmap(f, randomness="same")(x)
        # 断言以下代码块抛出 RuntimeError 异常，并且异常消息包含 "calling out variants"
        with self.assertRaisesRegex(RuntimeError, "calling out variants"):
            # 定义一个函数 f(x0, x1)，调用 torch.normal 生成张量，输出结果保存在 x 中
            def f(x0, x1):
                return torch.normal(x, y, out=x)
            # 对函数 f 进行 vmap 映射，其中 randomness 参数设置为 "same"
            vmap(f, randomness="same")(z, z)
        # 断言以下代码块抛出 RuntimeError 异常，并且异常消息包含 "do not yet support"
        with self.assertRaisesRegex(RuntimeError, "do not yet support"):
            # 定义一个函数 f(z)，调用 torch.rrelu 激活函数
            def f(z):
                return torch.rrelu(x)
            # 对函数 f 进行 vmap 映射，其中 randomness 参数设置为 "same"
            vmap(f, randomness="same")(z)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_chunk_vmap(self, in_dim, out_dim):
        # 设置 randomness 变量为 "different"
        randomness = "different"
        # 生成一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)

        # 定义函数 f(x)，计算 x 的正弦值并加上与 x 相同形状的随机张量，返回结果 y
        def f(x):
            y = x.sin() + torch.rand_like(x)
            return y

        # 对于 chunks 中的每个值进行迭代
        for chunks in [1, 2, 3, 4, 7, 10, 16]:
            # 使用 chunk_vmap 函数对函数 f 进行分块映射，设置输入维度和输出维度，并指定 randomness 和 chunks 参数
            output = chunk_vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunks=chunks,
            )(x)
            # 断言所有切片的输出是唯一的
            self._assert_all_slices_unique(output)

    @parametrize("in_dim", [0, 1, 2])
    @parametrize("out_dim", [0, 1, 2])
    def test_vmap_chunksize(self, in_dim, out_dim):
        # 设置 randomness 变量为 "different"
        randomness = "different"
        # 生成一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)

        # 定义函数 f(x)，计算 x 的正弦值并加上与 x 相同形状的随机张量，返回结果 y
        def f(x):
            y = x.sin() + torch.rand_like(x)
            return y

        # 对于 chunk_size 中的每个值进行迭代
        for chunk_size in [1, 2, 3, 4, 7, 10, 16, 100]:
            # 使用 vmap 函数对函数 f 进行映射，设置输入维度和输出维度，并指定 randomness 和 chunk_size 参数
            output = vmap(
                f,
                in_dims=in_dim,
                out_dims=out_dim,
                randomness=randomness,
                chunk_size=chunk_size,
            )(x)
            # 断言所有切片的输出是唯一的
            self._assert_all_slices_unique(output)

    def test_jacfwd_with_random(self):
        # 上面已经检查了行为，这里检查 jacfwd 是否尊重 randomness 参数

        # 生成一个形状为 (3, 4) 的随机张量 x
        x = torch.rand(3, 4)
        # 断言以下代码块抛出 RuntimeError 异常，并且异常消息包含 "called random operation while in randomness error mode"
        with self.assertRaisesRegex(
            RuntimeError, r"called random operation while in randomness error mode"
        ):
            # 对 torch.bernoulli 函数进行 jacfwd 自动求导，期望抛出异常
            jacfwd(torch.bernoulli)(x)

        # 因为 x 没有批处理，所以使用 bernoulli，因为它不会在原地进行随机操作
        # 使用 jacfwd 对 torch.bernoulli 函数进行自动求导，设置 randomness 参数为 "same"
        jacfwd(torch.bernoulli, randomness="same")(x)
        # 使用 jacfwd 对 torch.bernoulli 函数进行自动求导，设置 randomness 参数为 "different"
        jacfwd(torch.bernoulli, randomness="different")(x)

    @parametrize("randomness", ["error", "same", "different"])
    # 定义一个测试函数，用于测试未批量化的情况下的 dropout 行为
    def test_dropout_unbatched(self, device, randomness):
        # 创建一个形状为 (3,) 的张量 x，使用指定的设备
        x = torch.randn(3, device=device)
        # 创建一个形状为 (1, 3) 的张量 y，使用指定的设备
        y = torch.randn(1, 3, device=device)

        # 定义一个内部函数 fn，接受 x 和 y 作为参数
        def fn(x, y):
            # dropout 后的输出应为一个形状为 [B, 1, 3] 的张量 (这里 B=3)
            # 并且将 dropout 后的结果在第一维上求平均
            return x + torch.nn.functional.dropout(y, p=0.5).mean(1)

        # 验证在 `same` 和 `different` 随机性条件下不会引发错误
        # 参考：https://github.com/pytorch/pytorch/issues/92283
        # 根据随机性参数设定不同的上下文管理器 context
        context = (
            self.assertRaises(RuntimeError)  # 如果 randomness 为 "error"，则期望抛出 RuntimeError
            if randomness == "error"
            else contextlib.nullcontext()  # 否则使用空的上下文管理器
        )
        # 使用 context 上下文管理器来执行下面的代码块
        with context:
            # 使用 vmap 函数对 fn 进行批量映射，指定输入维度为 (0, None)，并使用给定的 randomness 参数
            vmap(fn, in_dims=(0, None), randomness=randomness)(x, y)
# 使用装饰器标记为 DynamoStrict 测试的类
@markDynamoStrictTest
class TestTransformFailure(TestCase):
    # 装饰器，如果 TorchDynamo 不可用则跳过测试
    @skipIfTorchDynamo()
    # 参数化测试，测试不同的变换函数
    @parametrize(
        "transform",
        ["vmap", "grad", "grad_and_value", "vjp", "jvp", "jacrev", "jacfwd"],
    )
    # 测试失败情况，使用自动求导函数时应失败
    def test_fails_with_autograd_function(self, device, transform):
        # 失败的构建环境列表
        failed_build_envs = ("linux-focal-py3.8-clang10", "linux-focal-py3.11-clang10")
        # 如果设备是 CPU，并且变换在指定条件下应失败，并且测试标志开启，并且在失败的构建环境中
        if (
            device == "cpu"
            and transform in ["grad", "vmap"]
            and TEST_WITH_TORCHDYNAMO
            and os.getenv("BUILD_ENVIRONMENT", "") in failed_build_envs
        ):
            # 抛出跳过测试的异常，说明预期在指定环境下应该失败
            raise unittest.SkipTest(
                "Unexpected successes on focal with dynamo,"
                + " see https://github.com/pytorch/pytorch/issues/107173"
            )

        # 定义一个测试用的 autograd.Function 类
        class Test(torch.autograd.Function):
            @staticmethod
            def forward(_, input):
                return input

            @staticmethod
            def backward(_, grad_input):
                return grad_input

        # 获取指定的变换函数
        transform = getattr(functorch, transform)

        # 定义一个函数 f(x)，应用测试类的方法
        def f(x):
            return Test.apply(x)

        # 根据变换类型选择合适的输入
        if transform in (grad, grad_and_value):
            input = torch.tensor(4.0)
        else:
            input = torch.randn(5)

        # 处理不同的变换类型
        if transform == vjp:
            transform = functools.partial(transform, f)
        elif transform == jvp:
            input = (input,)
            transform = functools.partial(transform, f, input)
        else:
            transform = transform(f)

        # 使用断言检查运行时错误，确保应该失败
        with self.assertRaisesRegex(RuntimeError, "autograd.Function"):
            transform(input)


# 使用装饰器标记为 DynamoStrict 测试的类
@markDynamoStrictTest
class TestVmapDeviceType(Namespace.TestVmapBase):
    # Vmap 测试方法的实现
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # 测试 _is_all_true 方法，检查其预期行为
    def test__is_all_true(self, device):
        # 定义内部测试函数
        def test():
            # 定义测试函数 f(x)，期望结果为预期的结果
            def f(x, *, expected_result):
                # 调用 _is_all_true 方法
                result = torch.ops.aten._is_all_true(x)
                # 断言结果不是批量张量
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                # 断言结果的形状是 torch.Size([])
                self.assertEqual(result.shape, torch.Size([]))
                # 断言结果的值等于预期的结果
                self.assertEqual(result.item(), expected_result)
                return result

            # 生成一个随机张量 x
            x = torch.rand(10, device=device)
            # 对 x >= 0 进行 vmap 操作，预期结果为 True
            vmap(f)(x >= 0, expected_result=True)
            # 对 x < 0 进行 vmap 操作，预期结果为 False
            vmap(f)(x < 0, expected_result=False)

            # 随机选择一个索引并将其对应的元素变为负数
            x[random.choice(range(10))] *= -1
            # 对 x >= 0 进行 vmap 操作，预期结果为 False
            vmap(f)(x >= 0, expected_result=False)
            # 对 x < 0 进行 vmap 操作，预期结果为 False
            vmap(f)(x < 0, expected_result=False)

            # 生成一个负数的张量 x
            x = -torch.rand(10, device=device)
            # 对 x > 0 进行 vmap 操作，预期结果为 False
            vmap(f)(x > 0, expected_result=False)
            # 对 x <= 0 进行 vmap 操作，预期结果为 True
            vmap(f)(x <= 0, expected_result=True)

        # 检查 vmap 的回退情况
        check_vmap_fallback(self, test, torch._is_all_true)
    # 定义测试方法 test__is_any_true，接受参数 self 和 device
    def test__is_any_true(self, device):
        # 定义内部测试函数 test
        def test():
            # 定义内部函数 f，接受参数 x 和 expected_result（关键字参数）
            def f(x, *, expected_result):
                # 调用 torch.ops.aten._is_any_true 方法计算结果
                result = torch.ops.aten._is_any_true(x)
                # 断言 result 不是批量张量
                self.assertFalse(torch._C._functorch.is_batchedtensor(result))
                # 断言 result 的形状是 torch.Size([])
                self.assertEqual(result.shape, torch.Size([]))
                # 断言 result 的值等于 expected_result
                self.assertEqual(result.item(), expected_result)
                # 返回 result 结果
                return result

            # 创建一个全零的张量 x，形状为 (10,)，在指定设备上
            x = torch.zeros(10, device=device, dtype=torch.bool)
            # 使用 vmap 函数对 f 进行映射，传入 x > 0 和 expected_result=False
            vmap(f)(x > 0, expected_result=False)

            # 修改 x 的第 5 个元素为 True
            x[5] = True
            # 使用 vmap 函数对 f 进行映射，传入 x > 0 和 expected_result=True
            vmap(f)(x > 0, expected_result=True)
            # 使用 vmap 函数对 f 进行映射，传入 x[1::2] 和 expected_result=True
            vmap(f)(x[1::2], expected_result=True)
            # 使用 vmap 函数对 f 进行映射，传入 x[0::2] 和 expected_result=False
            vmap(f)(x[0::2], expected_result=False)

        # 调用 check_vmap_fallback 方法，传入 self、test 函数和 torch._is_any_true 函数
        check_vmap_fallback(self, test, torch._is_any_true)

    # 定义测试方法 test_check_tensor，接受参数 self 和 device
    def test_check_tensor(self, device):
        # 定义内部测试函数 test
        def test():
            # 定义测试大小列表 test_sizes
            test_sizes = [
                (1,),
                (10,),
                (1, 1),
                (1, 10),
                (10, 1),
                (10, 10),
                (1, 1, 1),
                (10, 1, 1),
                (1, 10, 1),
                (10, 10, 10),
            ]

            # 定义内部函数 check_gte_0，接受参数 t，调用 torch._test_check_tensor 检查 t 中的元素是否都 >= 0
            def check_gte_0(t):
                return torch._test_check_tensor(t >= 0)

            # 错误消息字符串
            error_message = "Test message for TORCH_CHECK_TENSOR_ALL"

            # 遍历测试大小列表 test_sizes
            for size in test_sizes:
                # 创建一个随机张量 t_all_gte_0，形状为 size，存储在指定设备上
                t_all_gte_0 = torch.rand(size, device=device)
                # 创建一个所有元素减一的张量 t_all_lt_0
                t_all_lt_0 = t_all_gte_0 - 1

                # 使用 vmap 函数对 check_gte_0 进行映射，传入 t_all_gte_0
                vmap(check_gte_0)(t_all_gte_0)

                # 如果 size 的长度大于等于 2
                if len(size) >= 2:
                    # 使用 vmap 函数对 vmap(check_gte_0) 进行映射，传入 t_all_gte_0
                    vmap(vmap(check_gte_0))(t_all_gte_0)

                # 使用 self.assertRaisesRegex 断言引发 RuntimeError 异常，错误消息为 error_message
                with self.assertRaisesRegex(RuntimeError, error_message):
                    # 使用 vmap 函数对 check_gte_0 进行映射，传入 t_all_lt_0
                    vmap(check_gte_0)(t_all_lt_0)

                # 如果 size 的长度大于等于 2
                if len(size) >= 2:
                    # 使用 self.assertRaisesRegex 断言引发 RuntimeError 异常，错误消息为 error_message
                    with self.assertRaisesRegex(RuntimeError, error_message):
                        # 使用 vmap 函数对 vmap(check_gte_0) 进行映射，传入 t_all_lt_0
                        vmap(vmap(check_gte_0))(t_all_lt_0)

                # 如果 t_all_gte_0 的元素数量大于 1
                if t_all_gte_0.numel() > 1:
                    # 克隆 t_all_gte_0 得到 t_all_gte_0_but_one
                    t_all_gte_0_but_one = t_all_gte_0.clone()
                    # 选择每个维度中的一个随机索引，并将对应元素设置为 -1
                    idx = (random.choice(range(dim_size)) for dim_size in size)
                    t_all_gte_0_but_one[(..., *idx)] = -1

                    # 使用 self.assertRaisesRegex 断言引发 RuntimeError 异常，错误消息为 error_message
                    with self.assertRaisesRegex(RuntimeError, error_message):
                        # 使用 vmap 函数对 check_gte_0 进行映射，传入 t_all_gte_0_but_one
                        vmap(check_gte_0)(t_all_gte_0_but_one)

                    # 如果 size 的长度大于等于 2
                    if len(size) >= 2:
                        # 使用 self.assertRaisesRegex 断言引发 RuntimeError 异常，错误消息为 error_message
                        with self.assertRaisesRegex(RuntimeError, error_message):
                            # 使用 vmap 函数对 vmap(check_gte_0) 进行映射，传入 t_all_gte_0_but_one
                            vmap(vmap(check_gte_0))(t_all_gte_0_but_one)

        # 调用 check_vmap_fallback 方法，传入 self、test 函数和 torch._test_check_tensor 函数
        check_vmap_fallback(self, test, torch._test_check_tensor)
# 将当前测试类标记为 DynamoStrictTest 的测试类
@markDynamoStrictTest
class TestVmapNestedTensor(Namespace.TestVmapBase):
    # 定义一个内部方法 _vmap_test，用于调用 _vmap_test 函数
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    # 创建一个嵌套张量，根据给定的维度 dims 和设备 device
    # dims 应该类似于 [5, None, 10]，其中 None 表示应使用随机的不规则结构
    def _create_nt(self, dims, device):
        # 生成 sizes 列表，用于确定每个维度的大小
        sizes = [
            [
                d if d is not None else torch.randint(2, 10, size=(1,)).item()
                for d in dims[1:]
            ]
            for d in range(dims[0])
        ]
        # 创建嵌套张量，使用 torch.nested.nested_tensor 方法
        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in sizes], device=device
        )

    # 根据给定的 dims 创建一个与另一个嵌套张量 other 的组件数量、形状和不规则结构相匹配的新嵌套张量
    def _nt_from_similar(self, other, dims):
        # 断言 dims 的长度与 other 的维度数量相等
        assert len(dims) == other.dim()
        # 断言 dims 的第一个元素为 -1 或与 other 的第一维度大小相等
        assert dims[0] == -1 or dims[0] == other.size(0)

        # 计算返回的各个组件的大小
        ret_sizes = []
        for t in other.unbind():
            other_size = t.shape
            ret_size = []
            for i, d in enumerate(dims[1:]):
                if d == -1:
                    ret_size.append(other_size[i])
                else:
                    ret_size.append(d)
            ret_sizes.append(ret_size)

        # 创建新的嵌套张量，使用 torch.nested.nested_tensor 方法
        return torch.nested.nested_tensor(
            [torch.randn(*size) for size in ret_sizes], device=other.device
        )

    # 标记该测试方法允许使用 VmapFallbackUsage
    def test_fallback_unary(self, device):
        # 定义一个函数 f，对输入 x 进行一元运算
        def f(x):
            return x.sin() * 5.0 + 4.0

        # 创建一个嵌套张量 nt
        nt = self._create_nt([4, None, 3], device=device)
        # 调用 _vmap_test 方法进行测试
        self._vmap_test(f, (nt,))

    # 标记该测试方法允许使用 VmapFallbackUsage
    def test_fallback_binary(self, device):
        # 定义一个函数 f，对输入 x, y 进行二元运算
        def f(x, y):
            return x @ y

        # 创建两个嵌套张量 x 和 y
        x = self._create_nt([5, None, 3], device=device)
        y = self._create_nt([5, 3, None], device=device)
        # 调用 _vmap_test 方法进行测试
        self._vmap_test(f, (x, y))

    # 标记该测试方法允许使用 VmapFallbackUsage
    def test_fallback_binary_nt_and_unbatched_dense(self, device):
        # 定义一个函数 f，对输入 x, y 进行二元运算
        def f(x, y):
            return x @ y

        # 创建一个嵌套张量 x 和一个密集张量 y
        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 4, device=device)
        # 调用 _vmap_test 方法进行测试，设置输入维度 in_dims=(0, None)
        self._vmap_test(f, (x, y), in_dims=(0, None))

    # 标记该测试方法允许使用 VmapFallbackUsage
    def test_fallback_binary_nt_and_batched_dense(self, device):
        # 定义一个函数 f，对输入 x, y 进行二元运算
        def f(x, y):
            return x @ y

        # 创建一个嵌套张量 x 和一个批处理的密集张量 y
        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(5, 3, 4, device=device)
        # 调用 _vmap_test 方法进行测试
        self._vmap_test(f, (x, y))

    # 测试方法：验证在 Vmap 中嵌套张量的行为类似于密集张量
    def test_nt_acts_as_dense_in_vmap(self, device):
        # 定义一个函数 f，验证输入 x 不是嵌套张量
        def f(x):
            assert not x.is_nested
            return x

        # 创建一个嵌套张量 x
        x = self._create_nt([5, None, 3], device=device)
        # 调用 _vmap_test 方法进行测试
        self._vmap_test(f, (x,))
    # 定义测试方法，验证 torch.cat 在指定维度上拼接两个张量
    def test_cat_batching_rule(self, device):
        # 定义一个函数 f，用于在指定维度上拼接两个张量
        def f(x, y, dim):
            return torch.cat([x, y], dim=dim)

        # 创建两个嵌套张量 x 和 y，维度结构不同，但除了指定维度外其它维度相同，进行 vmap 测试
        x = self._create_nt([3, None, 2], device=device)
        y = self._create_nt([3, None, 2], device=device)
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        # 创建两个嵌套张量 x 和 y，维度结构相同，但在不同维度上进行拼接，进行 vmap 测试
        x = self._create_nt([3, 2, None], device=device)
        y = self._create_nt([3, 2, None], device=device)
        self._vmap_test(functools.partial(f, dim=1), (x, y))

        # 创建嵌套张量 x，再根据 x 创建结构相似的嵌套张量 y，并在指定维度上进行拼接，进行 vmap 测试
        x = self._create_nt([3, 2, None], device=device)
        y = self._nt_from_similar(x, [-1, 4, -1])
        self._vmap_test(functools.partial(f, dim=0), (x, y))

        # 创建嵌套张量 x，再根据 x 创建结构相似的嵌套张量 y，并在指定维度上进行拼接，进行 vmap 测试
        x = self._create_nt([3, None, 2], device=device)
        y = self._nt_from_similar(x, [-1, -1, 4])
        self._vmap_test(functools.partial(f, dim=1), (x, y))

    # 对于嵌套张量，不支持 .shape 调用
    # TODO: Fix this somehow?
    @unittest.expectedFailure
    def test_shape_call(self, device):
        # 定义函数 f，尝试在嵌套张量上调用 .shape[0]
        def f(x):
            x.shape[0]  # 不起作用
            return x

        # 创建一个嵌套张量 x，尝试调用 f 函数，并进行 vmap 测试
        x = self._create_nt([3, None, 2])
        self._vmap_test(f, (x,))

    # 在非零的维度上进行 vmap 操作会引发异常
    def test_nt_with_nonzero_in_dim_raises(self, device):
        # 定义函数 f，尝试对嵌套张量进行 vmap 操作
        def f(x):
            return x

        # 创建一个嵌套张量 x，尝试在非零维度上进行 vmap 测试，期望引发 RuntimeError 异常
        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, in_dims=2)(x)

    # 在非零的输出维度上进行 vmap 操作会引发异常
    def test_nt_with_nonzero_out_dim_raises(self, device):
        # 定义函数 f，尝试对嵌套张量进行 vmap 操作
        def f(x):
            return x

        # 创建一个嵌套张量 x，尝试在非零输出维度上进行 vmap 测试，期望引发 RuntimeError 异常
        x = self._create_nt([3, None, 2], device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Nested tensors can only be vmapped over dim=0"
        ):
            vmap(f, out_dims=2)(x)

    # 对于同时包含嵌套张量和批量密集张量，并在非零的 bdim 上进行 vmap 操作会引发异常
    def test_fallback_with_nt_and_batched_dense_with_nonzero_bdim_raises(self, device):
        # 定义函数 f，尝试对包含嵌套张量和批量密集张量进行 vmap 操作
        def f(x, y):
            return x @ y

        # 创建一个嵌套张量 x 和一个批量密集张量 y，并在非零的 bdim 上进行 vmap 测试，期望引发 RuntimeError 异常
        x = self._create_nt([5, None, 3], device=device)
        y = torch.randn(3, 5, 4, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            "Fallback not supported for mixed nested / non-nested arguments without bdim=0",
        ):
            vmap(f, in_dims=(0, 1))(x, y)

    # 多级 vmap 操作会引发异常，目前仅支持单级 vmap 操作
    def test_multilevel_vmap_raises(self, device):
        # 定义函数 f，尝试进行多级 vmap 操作
        def f(x):
            return x.sin() * 4.0 + 3.0

        # 创建一个多级嵌套张量 x，尝试进行多级 vmap 操作，期望引发 RuntimeError 异常
        x = self._create_nt([2, 2, 2, None], device=device)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(f))(x)

        with self.assertRaisesRegex(
            RuntimeError, "Only one level of vmap is supported"
        ):
            vmap(vmap(vmap(f)))(x)
# 指定需要运行测试的设备类型列表，这些设备类型包括 "cpu" 和 "cuda"
only_for = ("cpu", "cuda")

# 实例化 TestVmapOperatorsOpInfo 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestVmapOperatorsOpInfo, globals(), only_for=only_for)

# 实例化 TestVmapBatchedGradient 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestVmapBatchedGradient, globals(), only_for=only_for)

# 实例化 TestTransformFailure 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestTransformFailure, globals(), only_for=only_for)

# 实例化 TestRandomness 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestRandomness, globals(), only_for=only_for)

# 实例化 TestVmapDeviceType 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestVmapDeviceType, globals(), only_for=only_for)

# 实例化 TestVmapNestedTensor 类的设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestVmapNestedTensor, globals(), only_for=only_for)

# 如果当前脚本被作为主程序执行，则运行所有测试
if __name__ == "__main__":
    run_tests()
```