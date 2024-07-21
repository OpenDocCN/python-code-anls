# `.\pytorch\test\test_legacy_vmap.py`

```
# Owner(s): ["module: vmap"]

# 导入必要的库和模块
import functools
import itertools
import types
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._vmap_internals import vmap
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase

# 定义全局变量，用于保存回退警告信息的正则表达式
FALLBACK_REGEX = r"There is a performance drop"

# 进入上下文管理器，启用回退警告信息的显示
class EnableVmapFallbackWarnings:
    def __enter__(self):
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)

# 测试类，用于测试 vmap API 的遗留功能
class TestVmapAPILegacy(TestCase):
    # 测试非张量输出时是否引发异常
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(
            ValueError, "got type <class 'float'> as the return"
        ):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'> for return 1"):
            vmap(multiple_outputs)(torch.ones(3))

    # 测试不同映射维度大小是否引发异常
    def test_different_map_dim_size_raises(self):
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    # 测试没有输入的函数是否引发异常
    def test_func_with_no_inputs(self):
        expected_msg = "got no inputs"

        def foo():
            return torch.randn(3)

        def bar(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    # 测试常量函数的映射
    def test_constant_function(self):
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    # 测试单个输入的函数映射
    def test_single_input(self):
        x = torch.randn(2, 3)

        def square(x):
            return x * x

        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    # 测试多个输入的函数映射
    def test_multiple_inputs(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    # 测试多个输出的函数映射
    def test_multiple_outputs(self):
        def foo(x):
            return x * x, x * x * x

        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)
    # 定义测试函数，用于测试多输出函数的错误情况
    def test_multiple_outputs_error_cases(self):
        # 定义返回一个张量元组的函数
        def returns_tuple_of_tensors(x):
            return (x, x)

        # 定义返回两个张量列表的函数
        def returns_list_of_two_tensors(x):
            return [x, x]

        # 定义返回一个张量列表的函数
        def returns_list_of_one_tensor(x):
            return [x]

        # 生成一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)

        # 对返回张量元组的函数进行 vmap 操作，不应抛出异常
        vmap(returns_tuple_of_tensors)(x)

        # jax 支持这种情况，但我们暂时不支持，预期抛出异常
        msg = "must only return Tensors, got type <class 'list'>"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_two_tensors)(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_one_tensor)(x)

    # 定义测试函数，用于测试嵌套 vmap 在相同维度映射下的情况
    def test_nested_with_same_map_dim(self):
        # 生成形状为 (2, 3, 5) 的随机张量 x 和 y
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        # 对 torch.mul 函数进行嵌套 vmap 操作
        output = vmap(vmap(torch.mul))(x, y)
        # 断言输出张量与 x * y 相等
        self.assertEqual(output, x * y)

        # 进一步嵌套 vmap 操作
        output = vmap(vmap(vmap(torch.mul)))(x, y)
        # 断言输出张量与 x * y 相等
        self.assertEqual(output, x * y)

    # 定义测试函数，用于测试嵌套 vmap 在不同维度映射下的情况
    def test_nested_with_different_map_dim(self):
        # 生成形状为 (2, 3) 的随机张量 x 和形状为 (5, 3) 的随机张量 y
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        # 对 lambda 函数进行嵌套 vmap 操作
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        # 断言输出张量的形状为 (2, 5, 3) 并且与 x.view(2, 1, 3) * y 相等
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # 生成形状为 (7, 3) 的随机张量 z
        z = torch.randn(7, 3)
        # 进一步嵌套 vmap 操作
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        # 断言输出张量的形状为 (2, 5, 7, 3) 并且与 x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z 相等
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    # 定义测试函数，用于测试在内部 vmap 中的空操作
    def test_noop_in_inner_vmap(self):
        # 生成形状为 (3,) 和 (5,) 的随机张量 x 和 y
        x = torch.randn(3)
        y = torch.randn(5)
        # 对 lambda 函数进行嵌套 vmap 操作
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        # 断言输出张量与 x.view(3, 1).expand(3, 5) 相等
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    # 定义测试函数，用于测试不支持的操作时的错误消息
    def test_unsupported_op_err_msg(self):
        # 测试不支持的 view 操作
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(torch.ravel)(tensor)

        # 测试带有 out 参数的操作函数
        def out_op(x, y):
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        # 测试不支持 TensorList 的情况
        tensor = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(lambda t: torch.atleast_1d([t]))(tensor)

        # 测试不支持非张量返回的情况
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.Tensor.item)(tensor)
    # 测试函数，用于验证在不同的输出维度下的 vmap 函数行为
    def test_nonzero_out_dims(self):
        # 创建一个大小为 (2, 3) 的随机张量
        tensor = torch.randn(2, 3)
        # 使用 vmap 函数对 lambda 函数进行映射，out_dims=1 表示将批处理维度转换到维度 1
        result = vmap(lambda x: x, out_dims=1)(tensor)
        # 验证结果是否与原张量按照维度 1 进行置换后相等
        self.assertEqual(result, tensor.permute(1, 0))
        # 验证结果张量与原张量在内存中的指针是否相同
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # 测试批处理维度被置换到维度 2 的情况
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # 测试使用负数 out_dim 的情况，将批处理维度置换到维度 -1
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # 验证 out_dims 对所有输出都起作用的情况
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

        # 使用最大可 vmap 的张量维度（64 维）进行测试
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        # 验证结果张量的形状是否符合预期
        self.assertEqual(result.shape, expected_shape)

        # 测试非恒等函数的情况
        def foo(x, y):
            return x, x * y, x * y * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        # 验证结果是否与预期的张量按维度 1 进行置换后相等
        self.assertEqual(
            result,
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

    # 测试多个输出维度的情况
    def test_multiple_out_dims(self):
        # 定义一个简单的函数 foo，返回输入的两个相同的张量
        def foo(x):
            return x, x

        # 定义一个复杂一些的函数 bar，返回四个张量的组合
        def bar(x, y):
            return x, x, x, x * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        # 测试使用 out_dims=(0, 1) 的情况，将批处理维度置换到维度 0 和 1
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        # 测试使用 out_dims=(-1, 0, 1, 2) 的情况，将批处理维度置换到多个维度
        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        self.assertEqual(result, expected)
    def test_nested_out_dims(self):
        # 创建一个形状为 (2, 3, 5, 7) 的随机张量 y
        y = torch.randn(2, 3, 5, 7)

        # 内部 vmap 具有非零的 out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        # 断言结果张量的形状为 (2, 5, 3, 7)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        # 断言结果张量等于 y 的按维度重新排列结果
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # 所有 vmap 都具有非零的 out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        # 断言结果张量的形状为 (5, 2, 3, 7)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        # 断言结果张量等于 y 的按维度重新排列结果
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # 添加一些负的 out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        # 断言结果张量的形状为 (5, 7, 3, 2)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        # 断言结果张量等于 y 的按维度重新排列结果
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # 测试不是恒等函数的情况
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        # 断言结果张量的形状为 (3, 2, 5)
        self.assertEqual(result.shape, (3, 2, 5))
        # 断言结果张量等于 (y.view(5, 1, 3) * x) 按维度重新排列结果
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        # 定义一个简单的函数 foo(x)，返回输入 x
        def foo(x):
            return x

        # 测试当函数有一个输出时，接受 out_dims=(1,)
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_must_be_int_or_tuple_of_int_err_msg(self):
        # 错误消息
        msg = "`out_dims` must be an int or a tuple of int"
        tensor = torch.randn(2, 3)
        # 断言传递错误参数时会引发 ValueError 异常并包含正确的错误消息
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=None)(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(None,))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        # 错误消息
        msg = "`out_dims` must have one dim per output"
        x = torch.randn(2, 3, 5)

        # out_dims 数量过多
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # out_dims 数量不足
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)
    # 测试当输出维度超出范围时是否会引发错误消息
    def test_out_dim_out_of_bounds_err_msg(self):
        # 错误消息内容
        msg = "Dimension out of range"
        # 创建一个形状为 (2, 3, 5) 的随机张量
        x = torch.randn(2, 3, 5)
        # 断言使用 vmap 函数对 x 进行映射时是否会抛出 IndexError 异常，异常信息应包含 msg
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        # 同上，但这里 out_dims 为负数
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    # 测试在不同输入维度的情况下 vmap 函数的行为
    def test_non_zero_in_dims(self):
        # 创建一个形状为 (2, 3, 5) 的随机张量
        tensor = torch.randn(2, 3, 5)

        # 使用 vmap 对 tensor 进行映射，将批处理维度移动到最前面，out_dims=1
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)

        # 使用 vmap 对 torch.mul 函数进行映射，in_dims=(0, 1)，相当于对 x 和 y 进行逐元素相乘
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())

        # 使用 vmap 对 torch.mul 函数进行映射，in_dims=(1, 0)，相当于对 x 和 y 的转置进行逐元素相乘
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    # 测试在某些输入维度设置为 None 时 vmap 函数的行为
    def test_none_in_dims(self):
        # 创建两个形状为 (2, 3) 的随机张量 x 和 y
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # 使用 vmap 对 torch.mul 函数进行映射，in_dims=(0, None)，将 y 广播到第一维，然后逐元素相乘
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # 使用 vmap 对 torch.mul 函数进行映射，in_dims=(0, None)，将 scalar 值 2 广播到 x 的所有元素上进行逐元素相乘
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    # 测试嵌套非默认输入维度设置时 vmap 函数的行为
    def test_nested_non_default_in_dims(self):
        # 创建两个形状分别为 (5, 2, 3) 和 (3, 5, 2) 的随机张量 x 和 y
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)

        # 使用 vmap 对 torch.mul 函数进行嵌套映射，in_dims=(1, 0)，相当于对 x 和 y 进行逐元素相乘，并重新排列维度
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    # 测试非默认输入维度和输出维度设置时 vmap 函数的行为
    def test_non_default_in_dims_out_dims(self):
        # 创建一个形状为 (2, 3, 5) 的随机张量 x
        x = torch.randn(2, 3, 5)

        # 使用 vmap 对 lambda 函数进行映射，in_dims=1, out_dims=1，相当于对 x 进行逐元素操作并保持维度不变
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # 使用 vmap 对 lambda 函数进行映射，in_dims=2, out_dims=1，相当于对 x 进行逐元素操作并将维度 1 和 2 交换
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # 定义一个简单的操作函数 foo
        def foo(x):
            return x * 2

        # 使用 vmap 对 foo 函数进行映射，in_dims=1, out_dims=1，相当于对 x 进行逐元素乘以 2 的操作
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # 使用 vmap 对 foo 函数进行映射，in_dims=2, out_dims=1，相当于对 x 进行逐元素乘以 2 并将维度 1 和 2 交换
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # 基本的嵌套测试，对 foo 函数进行多层映射
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)
    # 定义测试方法，用于测试接受嵌套输入的情况
    def test_accepts_nested_inputs(self):
        # 设置常量 B0 为 2
        B0 = 2
        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)

        # 单层嵌套
        # 使用 vmap 对 lambda 函数进行向量化映射，计算元组 (x, y) 中每个元素 z 的 z[0] + z[1]
        out = vmap(lambda z: z[0] + z[1])((x, y))
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 (0,) 对 lambda 函数进行向量化映射
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 ((0, 0),) 对 lambda 函数进行向量化映射
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)

        # 使用列表 [x, y] 进行向量化映射
        out = vmap(lambda z: z[0] + z[1])([x, y])
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 (0,) 对列表 [x, y] 进行向量化映射
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 ([0, 0],) 对列表 [x, y] 进行向量化映射
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)

        # 使用字典 {"x": x, "y": y} 进行向量化映射
        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 (0,) 对字典 {"x": x, "y": y} 进行向量化映射
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)
        # 使用指定的输入维度 ({"x": 0, "y": 0},) 对字典 {"x": x, "y": y} 进行向量化映射
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        # 断言输出 out 等于 x + y
        self.assertEqual(out, x + y)

        # 多层嵌套
        # 定义复杂的 lambda 函数，对多层嵌套的输入进行操作
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        # 对复杂的嵌套结构 {"x": [x, (x,)], "y": [y, y]} 进行向量化映射
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        # 断言输出 out 等于 x + x + y + y
        self.assertEqual(out, x + x + y + y)
    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        # 定义函数 foo，计算输入列表的第一个元素与第二个元素的乘积
        def foo(xy):
            return xy[0] * xy[1]

        # 定义函数 bar，计算 x 与 yz[0]、yz[1] 的乘积
        def bar(x, yz):
            return x * yz[0] * yz[1]

        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)

        # 设置错误信息的消息字符串，针对 in_dim=0 的情况
        msg = "Got in_dim=0 for an input but the input is of type"
        # 断言使用 vmap 对 torch.sum 进行映射时会抛出 ValueError 异常，并且异常信息包含预设的消息字符串
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # 下面这行代码不应该抛出异常
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        # 定义函数 foo，计算输入张量的元素平方
        def foo(x):
            return x * x

        # 生成一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 生成一个形状为 (2, 3) 的随机张量 y
        y = torch.randn(2, 3)

        # 设置错误信息的正则表达式消息字符串，针对 in_dim 不合法的情况
        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        # 断言使用 vmap 对 foo 函数进行映射时会抛出 ValueError 异常，并且异常信息符合预设的正则表达式消息字符串
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-1,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # 下面这两行代码不应该抛出异常
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        # 注意：将来实现 torch.atan2 的批处理规则后，应更改此测试以测试其他操作符的后备路径，避免过时
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            # 对 torch.atan2 使用 vmap 进行映射，记录警告信息
            result = vmap(op)(x, y)
            # 此处唯一的警告是关于 "vmap is experimental" 的警告，而非 vmap 后备路径的警告
            self.assertEqual(len(wa), 1)

    def test_fallback_warns_when_warnings_are_enabled(self):
        # 注意：将来实现 torch.atan2 的批处理规则后，应更改此测试以测试其他操作符的后备路径，避免过时
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                # 启用 vmap 后备路径警告，对 torch.atan2 使用 vmap 进行映射，记录警告信息
                result = vmap(op)(x, y)
            # 预期捕获两条警告信息
            self.assertEqual(len(wa), 2)
            # 断言最后一条警告的消息字符串符合预设的 FALLBACK_REGEX 正则表达式
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)
    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        # 使用 `warnings` 模块捕获警告信息
        with warnings.catch_warnings(record=True) as wa:
            # 启用 `EnableVmapFallbackWarnings` 上下文管理器
            with EnableVmapFallbackWarnings():
                # 使用 `vmap_args` 和 `inputs` 参数调用 `vmap` 函数两次
                result = vmap(*vmap_args)(*inputs)
            # 断言捕获到两个警告信息
            self.assertEqual(len(wa), 2)
            # 断言最后一个警告信息的消息符合预期的正则表达式 `FALLBACK_REGEX`
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        # 注意事项：未来可能为 `torch.atan2` 实现批处理规则
        # 如果实现了，此测试应修改为测试另一个操作符的回退路径，避免过时
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = "The fallback path does not support vmap over dims of size 0"

        # 断言在运行时异常中捕获到指定的消息
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_atan2(self):
        # 注意事项：未来可能为 `torch.atan2` 实现批处理规则
        # 如果实现了，此测试应修改为测试另一个操作符的回退路径，避免过时
        op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        # 回退至 `torch.atan2`
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # 回退至 `torch.atan2`，嵌套 `vmap`
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # 大批量大小（总共 10000）
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))
    # 定义一个测试函数，用于测试 fallback_masked_fill 方法
    def test_fallback_masked_fill(self):
        # 注意事项：未来可能会为 masked_fill 实现批处理规则
        # 如果实现了批处理规则，应该将此测试用例替换为测试另一个运算符的回退路径，以避免代码陈旧
        def run_test(batch_size):
            # 设置批大小为 B0
            B0 = batch_size
            # 创建一个形状为 (B0, 7, 11, 13) 的随机张量 x
            x = torch.randn(B0, 7, 11, 13)
            # 指定操作维度为 0
            dim = 0
            # 创建一个索引张量，包含要操作的索引位置
            index = torch.tensor([0, 4, 2])
            # 创建一个形状为 (B0, 3, 11, 13) 的随机值张量 values
            values = torch.randn(B0, 3, 11, 13)

            # 使用 self._assert_uses_vmap_fallback 方法验证是否使用了 vmap 的回退机制
            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            # 使用 vmap 对 torch.index_add 方法进行批处理
            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            # 计算预期结果，将 values 重塑为 (B0, 3, 11, 13)，并执行 torch.index_add 操作
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 11, 13))
            # 断言批处理结果与预期结果相等
            self.assertEqual(result, expected)

        # 分别运行批大小为 5 和 1237 的测试
        run_test(batch_size=5)
        run_test(batch_size=1237)

    # 定义一个测试函数，用于测试 fallback_multiple_returns 方法
    def test_fallback_multiple_returns(self):
        # 注意事项：未来可能会为 torch.var_mean 实现批处理规则
        # 如果实现了批处理规则，应该将此测试用例替换为测试另一个运算符的回退路径，以避免代码陈旧
        B0, B1, B2 = 2, 3, 1237
        # 创建一个形状为 (B0, 10) 的随机张量 tensor
        tensor = torch.randn(B0, 10)

        # 使用 self._assert_uses_vmap_fallback 方法验证是否使用了 vmap 的回退机制
        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # 验证 torch.var_mean 方法在回退机制下的正确性
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        # 断言批处理结果与预期结果相等
        self.assertEqual(result, expected)

        # 嵌套使用 vmap 进行测试
        # 创建一个形状为 (B0, B1, 10) 的随机张量 tensor
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        # 断言批处理结果与预期结果相等
        self.assertEqual(result, expected)

        # 针对大批量大小和嵌套 vmap 进行测试
        # 创建一个形状为 (B0, B1, B2, 10) 的随机张量 tensor
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        # 断言批处理结果与预期结果相等
        self.assertEqual(result, expected)
    def test_inplace_fallback_unary(self):
        # 测试对不带额外张量参数的原地方法进行原地回退的情况。这是回退的最简单情况。
        # 注意: 将来我们可能会为 acos_ 实现一个批处理规则。
        # 如果/当我们这样做时，应该替换此测试以测试另一个运算符的回退路径，以避免代码老化。
        
        # 定义原地操作符为 acos_
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        # 生成一个形状为 B0 x 5 的随机张量 x
        x = torch.randn(B0, 5)
        # 断言使用 vmap 回退机制
        self._assert_uses_vmap_fallback((op,), (x,))

        # 单个 vmap 操作
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        # 对 x 应用 vmap(op)，并将结果与 x_orig 的 acos() 进行比较
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # 单个 vmap 操作 + 不同的 out_dim 会产生一个视图(!)
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        # 对 x 应用 vmap(op, out_dims=(1,))，并将结果与 x_orig 的转置后的 acos() 进行比较
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())

        # 嵌套 vmap 操作
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        # 对 x 应用 vmap(vmap(op))，并将结果与 x_orig 的 acos() 进行比较
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # 嵌套 vmap 操作，大批量大小
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        # 对 x 应用 vmap(vmap(vmap(op)))，并将结果与 x_orig 的 acos() 进行比较
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # 注意: 将来我们可能会为 atan2_ 实现一个批处理规则。
        # 如果/当我们这样做时，应该替换此测试以测试另一个运算符的回退路径，以避免代码老化。
        
        # 定义操作符为 atan2_
        op = Tensor.atan2_
        outplace_op = torch.atan2

        # 生成形状为 5 x 7 x 11 的随机张量 x 和 y
        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        # 断言使用 vmap 回退机制
        self._assert_uses_vmap_fallback((op,), (x, y))

        # 单个 vmap 操作
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        # 对 x, y 应用 vmap(op, (2, 0))，并将结果与 outplace_op(x_orig, y.movedim(0, 2)) 进行比较
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # 嵌套 vmap 操作
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        # 对 x, y 应用 vmap(vmap(op), (2, 0))，并将结果与 outplace_op(x_orig, y.movedim([0, 1], [2, 0])) 进行比较
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # 大批量大小 (总共 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        # 对 x, y 应用 vmap(vmap(vmap(op)))，并将结果与 outplace_op(x_orig, y.view(B0, B1, B2, 1)) 进行比较
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))
    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        # 定义测试函数，用于测试不同层次下的 inplace 和 fallback 行为
        op = Tensor.atan2_
        # 定义 op 作为 Tensor 类的 atan2_ 方法
        outplace_op = torch.atan2
        # 定义 outplace_op 作为 torch 的 atan2 函数
        B0, B1, B2 = 2, 3, 5
        # 定义测试中使用的批次大小

        x = torch.rand(B0, 7)
        # 创建 B0 行 7 列的随机张量 x
        y = torch.rand(7)
        # 创建长度为 7 的随机张量 y
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))
        # 调用 _assert_uses_vmap_fallback 方法，测试是否使用了 vmap 的 fallback

        # op(left, right): All of the levels in right are found in left
        # op(left, right)：right 中的所有层次都在 left 中找到
        x_orig = torch.rand(B0, 7)
        # 创建原始张量 x_orig，大小为 B0 行 7 列
        x = x_orig.clone()
        # 克隆张量 x_orig 得到 x
        y = torch.rand(7)
        # 创建长度为 7 的随机张量 y
        vmap(op, in_dims=(0, None))(x, y)
        # 对 op 应用 vmap，指定维度 in_dims=(0, None)，并传入 x 和 y
        self.assertEqual(x, outplace_op(x_orig, y))
        # 断言 x 等于 outplace_op(x_orig, y)

        x_orig = torch.rand(B0, B1, 7)
        # 创建原始张量 x_orig，大小为 B0 行 B1 列 7 列
        x = x_orig.clone()
        # 克隆张量 x_orig 得到 x
        y = torch.rand(B0, 7)
        # 创建大小为 B0 行 7 列的随机张量 y
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        # 对 op 应用双层 vmap，指定维度 in_dims=(0, None)，并传入 x 和 y
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))
        # 断言 x 等于 outplace_op(x_orig, y.view(B0, 1, 7))

        # op(left, right): Some of the levels in right are not found in left
        # op(left, right)：right 中的一些层次在 left 中找不到
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
        # 定义错误消息字符串

        x = torch.rand(7)
        # 创建长度为 7 的随机张量 x
        y = torch.rand(B0, 7)
        # 创建大小为 B0 行 7 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)
        # 使用 vmap 对 op 应用，指定维度 in_dims=(None, 0)，并传入 x 和 y，期待抛出 RuntimeError 异常

        x = torch.rand(B1, 7)
        # 创建大小为 B1 行 7 列的随机张量 x
        y = torch.rand(B0, 7)
        # 创建大小为 B0 行 7 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        # 使用双层 vmap 对 op 应用，指定维度 in_dims=(0, None) 和 (None, 0)，并传入 x 和 y，期待抛出 RuntimeError 异常

        x = torch.rand(B1, 7)
        # 创建大小为 B1 行 7 列的随机张量 x
        y = torch.rand(7, B0)
        # 创建大小为 7 行 B0 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)
        # 使用双层 vmap 对 op 应用，指定维度 in_dims=(0, None) 和 (None, 1)，并传入 x 和 y，期待抛出 RuntimeError 异常

        x = torch.rand(B0, 7)
        # 创建大小为 B0 行 7 列的随机张量 x
        y = torch.rand(B0, B1, 7)
        # 创建大小为 B0 行 B1 列 7 列的随机张量 y
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)
        # 使用双层 vmap 对 op 应用，指定维度 in_dims=(None, 0)，并传入 x 和 y，期待抛出 RuntimeError 异常

    def test_backward_unsupported_interaction(self):
        # 测试反向传播与 vmap 的不兼容交互行为
        x = torch.randn(3, requires_grad=True)
        # 创建大小为 3 的随机张量 x，并设置 requires_grad=True，以支持梯度计算
        y = torch.randn(5)
        # 创建大小为 5 的随机张量 y
        grad = torch.randn_like(x)
        # 创建与 x 大小相同的随机梯度张量 grad
        err_msg = r"backward\(\) called inside torch.vmap"
        # 定义错误消息字符串，表示在 torch.vmap 内调用了 backward()

        def backward_on_vmapped_tensor(x):
            # 定义在 vmapped 张量上执行反向传播的函数
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)
        # 使用 vmap 对 backward_on_vmapped_tensor 函数应用，并传入 x，期待抛出 RuntimeError 异常

        def backward_with_vmapped_grad(x, grad):
            # 定义使用 vmapped 梯度进行反向传播的函数
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)
        # 使用 vmap 对 backward_with_vmapped_grad 函数应用，并传入 x 和 grad，期待抛出 RuntimeError 异常

        def completely_unrelated_backward(y):
            # 定义完全不相关的函数进行反向传播
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)
        # 使用 vmap 对 completely_unrelated_backward 函数应用，并传入 y，期待抛出 RuntimeError 异常
    def test_grad_unsupported_interaction(self):
        # 创建一个形状为 (3,) 的张量，开启梯度追踪
        input_tensor = torch.randn(3, requires_grad=True)
        # 定义错误消息字符串，用于异常断言
        err_msg = "autograd.grad.* called inside torch.vmap"

        # 创建一个形状为 (3,) 的张量，开启梯度追踪
        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            # 计算张量 captured 和输入张量 input_tensor 的点乘和，结果需求梯度
            output = (captured * input_tensor).sum()
            # 返回 output 对 captured 的梯度
            return torch.autograd.grad([output], [captured])[0]

        # 断言在 vmap 上下文中调用 output_to_grad_is_vmapped 会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        # 计算输入张量 input_tensor 的平方和
        output = (input_tensor**2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            # 返回 output 对 input_tensor 的梯度
            return torch.autograd.grad([output], [input_tensor])[0]

        # 断言在 vmap 上下文中调用 input_to_grad_is_vmapped 会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        # 定义批次大小 N
        N = 3
        # 创建形状为 (N,) 的张量 x，并开启梯度追踪
        x = torch.randn(N, requires_grad=True)
        # 创建形状为 (N,) 的张量 y
        y = torch.randn(N)

        def vjp_mul(v):
            # 计算 x 乘以 y 的结果，并返回其对 x 的雅可比向量积
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        # 创建单位矩阵 batched_v，形状为 (N, N)
        batched_v = torch.eye(N)
        # 使用 vmap 对 vjp_mul 进行批处理映射，计算雅可比矩阵
        jacobian = vmap(vjp_mul)(batched_v)
        # 断言计算得到的雅可比矩阵等于 torch.diagflat(y)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        # 创建形状为 (3,) 的张量 x
        x = torch.randn(3)
        # 创建形状为 (2, 3) 的张量 y
        y = torch.randn(2, 3)
        # 使用 vmap 对 functools.partial(torch.mul, x) 进行批处理映射，计算结果
        result = vmap(functools.partial(torch.mul, x))(y)
        # 断言计算结果等于 x 乘以 y 的结果
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        # 创建形状为 (2, 3) 的张量 tensor
        tensor = torch.randn(2, 3)
        # 创建一个输入维度为 3、输出维度为 3 的线性模型，无偏置
        model = torch.nn.Linear(3, 3, bias=False)
        # 使用 vmap 对模型 model 进行批处理映射，计算结果
        result = vmap(model)(tensor)
        # 断言计算结果等于模型应用于 tensor 后的结果
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        # 定义 B0 = 7
        B0 = 7
        # 创建形状为 (2, 3, 4, 5) 的张量 x，并开启梯度追踪
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        # 创建形状为 (3, 3, 1, 1) 的张量 weight
        weight = torch.randn(3, 3, 1, 1)
        # 创建形状为 (B0, 2, 3, 4, 5) 的张量 v
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            # 使用函数 torch.nn.functional.conv2d 计算结果
            result = torch.nn.functional.conv2d(x, weight)
            # 计算 result 对 x 的梯度，使用 v 作为梯度输出
            (grad_x,) = torch.autograd.grad(result, x, v)
            # 返回计算得到的梯度 grad_x
            return grad_x

        # 运行 vmap(get_vjp)(v)，断言不会出错
        # 对于卷积的反向传播，由于原始偏置不存在，grad_bias 返回未定义的 Tensor
        #
        # 在将来，可能会为卷积反向传播添加批处理规则。当这种情况发生时，我们应修改此测试使用不同的操作（和/或创建和使用虚拟运算符）以避免陈旧。
        self._assert_uses_vmap_fallback([get_vjp], [v])
# 将输入的列表按照给定的维度进行切片操作，返回切片后的结果元组
def slice_inputs(inputs, bdims, i):
    result = []
    # 遍历输入列表和对应的维度列表
    for inp, bdim in zip(inputs, bdims):
        # 如果当前维度为None，则直接添加原始输入
        if bdim is None:
            result.append(inp)
        else:
            # 否则，按照给定的维度和索引i进行选择操作，并添加到结果列表中
            result.append(inp.select(bdim, i))
    return tuple(result)


# 使用vmap函数对给定操作op进行批量映射处理，处理结果作为元组返回
def reference_vmap(op, inputs, in_dims=0, out_dims=0):
    # 如果in_dims是整数，则转换为元组，每个输入都使用相同的维度
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    # 计算每个输入在指定维度上的尺寸，忽略维度为None的输入
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    # 断言所有非空维度的尺寸相同
    assert all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes)
    # 获取第一个非空维度的尺寸
    bdim_size = bdim_sizes[0]
    # 对每个索引i在输入上执行操作op，并将结果作为元组收集起来
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    # 断言结果的长度大于0
    assert len(results) > 0
    # 判断操作是否只返回单个值
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        # 如果操作只返回单个值，断言所有结果都是torch.Tensor类型
        assert all(isinstance(result, torch.Tensor) for result in results)
        # 如果out_dims是整数，则转换为元组，只有一个输出维度
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        # 在指定维度上堆叠所有结果，并返回
        return torch.stack(results, dim=out_dims[0])

    # 断言所有结果都是元组
    assert all(isinstance(result, tuple) for result in results)
    # 获取每个元组中的返回值数量
    num_returns = len(results[0])
    # 断言所有元组中返回值的数量相同
    assert all(len(result) == num_returns for result in results)
    # 如果out_dims是整数，则转换为元组，每个返回值使用相同的输出维度
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    # 将每个结果分片在指定的输出维度上进行堆叠，并返回结果元组
    return tuple(
        torch.stack(result_shards, out_dim)
        for result_shards, out_dim in zip(zip(*results), out_dims)
    )


# 定义一个静态方法集合TensorFactory，用于生成各种类型的随机张量
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


# _vmap_test方法用于测试vmap函数的正确性，比较其输出与参考结果是否相同
# 还包含一些可选的测试，如检查结果是否是输入的视图和是否正确传播梯度
def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    # 使用vmap函数对给定操作op、输入inputs、输入维度in_dims和输出维度out_dims进行批量映射
    result = vmap(op, in_dims, out_dims)(*inputs)
    # 使用reference_vmap函数对同样的操作和输入进行参考映射，用于比较
    reference_result = reference_vmap(op, inputs, in_dims, out_dims)
    # 断言vmap函数的输出与参考结果相同
    self.assertEqual(result, reference_result)
    # 判断操作是否只返回单个值
    op_has_single_return = not isinstance(result, tuple)

    # 如果check_view为True，则检查结果是否是第一个输入的视图
    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

    # 如果check_propagates_grad为False，则直接返回，不再进行梯度传播测试
    if not check_propagates_grad:
        return
    # 假设第一个输入是浮点张量，检查vmap操作是否正确传播requires_grad标志到第一个输出
    # 创建输入的克隆列表，以确保不改变原始输入列表
    inputs_clone = list(inputs)
    # 将第一个输入张量克隆，并标记为需要梯度计算
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    # 使用 vmap 对操作 op 进行向量化映射，输入为 inputs_clone
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    # 如果操作 op 只返回单个结果，则将其包装为元组
    result_as_tuple = (result,) if op_has_single_return else result
    # 断言第一个输出结果需要梯度计算
    self.assertTrue(result[0].requires_grad)
# 检查函数是否允许使用 vmap 回退方法
def should_allow_vmap_fallback_usage(fn):
    # 返回函数属性 "_allow_vmap_fallback_usage" 的值，如果不存在则返回 False
    return getattr(fn, "_allow_vmap_fallback_usage", False)


# 设置函数允许使用 vmap 回退方法的标志
def allowVmapFallbackUsage(fn):
    # 设置函数属性 "_allow_vmap_fallback_usage" 为 True
    fn._allow_vmap_fallback_usage = True
    # 返回函数本身
    return fn


# TestVmapBaseLegacy 的所有测试确保不会调用慢速的 vmap 回退方法。
# 这样做是为了能够逐步为运算符添加批处理规则，以替换这些运算符的慢速 vmap 回退路径。
# 要跳过这个检查，请使用 allowVmapFallbackUsage 装饰器。
#
# 注意：请勿直接向 TestVmapBaseLegacy 添加测试，除非希望它们在 TestVmapBaseLegacy 的每个子类上运行。
# 请将它们添加到例如 TestVmapOperators 中。
#
# 注意：TestVmapBaseLegacy 是一个嵌套类。这可以防止测试运行器捕获并运行它。
class Namespace:
    # 定义一个测试类 TestVmapBaseLegacy，继承自 TestCase
    class TestVmapBaseLegacy(TestCase):
        
        # 初始化方法，接受一个 method_name 参数，默认为 "runTest"
        def __init__(self, method_name="runTest"):
            # 调用父类 TestCase 的初始化方法
            super().__init__(method_name)
            
            # 获取当前实例中名为 method_name 的方法的引用
            test_method = getattr(self, method_name, None)
            # 如果找不到对应的方法，则直接返回
            if test_method is None:
                return
            
            # 检查是否允许使用 vmap 回退路径
            if not should_allow_vmap_fallback_usage(test_method):
                # 动态设置当前实例中的 method_name 方法，使用 _wrap_method_with_vmap_fallback_check 进行包装
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )
        
        # 定义一个内部方法，用于包装传入的 method 方法，以进行 vmap 回退路径的检查
        def _wrap_method_with_vmap_fallback_check(self, method):
            # 定义警告消息内容
            msg = (
                "Expected the test to not invoke the vmap fallback path, i.e., "
                "all of the operators being tested in this test should have batching "
                "rules implemented. If you are intentionally testing something to "
                "do with the fallback path, use allowVmapFallbackUsage. Otherwise, "
                "please make sure that batching rules are implemented for the "
                "operator(s) being tested."
            )
            
            # 定义一个内部包装方法 wrapper，使用 functools.wraps 对 method 进行包装
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                # 使用 warnings.catch_warnings 捕获警告
                with warnings.catch_warnings(record=True) as wa:
                    # 设置警告过滤器为 "always"
                    warnings.simplefilter("always")
                    # 启用 EnableVmapFallbackWarnings 上下文管理器
                    with EnableVmapFallbackWarnings():
                        # 调用原始的 method 方法
                        method(*args, **kwargs)
                    # 遍历捕获到的警告
                    for captured_warning in wa:
                        # 断言捕获到的警告消息不匹配 FALLBACK_REGEX，否则输出 msg
                        self.assertNotRegex(
                            str(captured_warning.message), FALLBACK_REGEX, msg
                        )
            
            # 返回经过包装后的 wrapper 方法，绑定到当前实例
            return types.MethodType(wrapper, self)
        
        # 使用装饰器 allowVmapFallbackUsage 标记的测试方法，用于测试 vmap 回退路径的正常情况
        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # 指定一个使用 vmap 回退路径的操作符
            op_using_fallback = torch.var_mean
            # 对 op_using_fallback 应用 vmap，并传入一个随机生成的 tensor
            vmap(op_using_fallback)(torch.rand(3))
        
        # 测试 vmap 回退路径的方法
        def test_vmap_fallback_check(self):
            # 内部定义一个不使用回退路径的方法
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                pass
            
            # 指定一个使用 vmap 回退路径的操作符
            op_using_fallback = torch.var_mean
            
            # 内部定义一个使用回退路径的方法
            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                # 对 op_using_fallback 应用 vmap，并传入一个随机生成的 tensor
                vmap(op_using_fallback)(torch.rand(3))
            
            # 调用不使用回退路径的方法
            no_fallback(self)
            
            # 断言抛出 AssertionError 异常，因为使用了回退路径
            with self.assertRaises(AssertionError):
                uses_fallback(self)
class TestVmapOperatorsLegacy(Namespace.TestVmapBaseLegacy):
    # 继承自命名空间的TestVmapBaseLegacy类，用于测试Vmap操作的遗留版本

    def _vmap_test(self, *args, **kwargs):
        # 封装_vmap_test方法，接受任意位置和关键字参数，传递给_vmap_test函数
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        # 封装_vmap_view_test方法，调用_vmap_test方法并设置check_view为True
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        # 封装_test_unary方法，使用functools.partial创建test函数，固定op和getter参数
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # 单层vmap，不同的输入维度和输出维度
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # 双层嵌套vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    def test_unary_pointwise_ops(self):
        # 定义一系列操作和获取器的元组
        cases = [
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
        ]
        for op, getter in cases:
            # 对每个操作和获取器对调用_test_unary方法
            self._test_unary(op, getter, "cpu")
    # 定义测试函数 test_clone，用于测试 clone 方法的不同用法
    def test_clone(self):
        # 执行基本测试，使用 _test_unary 方法测试 x.clone()，并生成随机张量在 CPU 上
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")
        # 执行基本测试，使用 _test_unary 方法测试带有内存格式参数的 x.clone()
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )
        # 执行基本测试，使用 _test_unary 方法测试带有连续内存格式参数的 x.clone()
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

        # 测试当使用 torch.contiguous_format 时，每个示例是否是连续的
        def clone_contiguous(x):
            return x.clone(memory_format=torch.contiguous_format)

        B0, B1 = 3, 5
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        # 断言：当使用 torch.contiguous_format 时，每个示例是否是连续的
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())

        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        # 断言：当使用 torch.contiguous_format 时，每个示例是否是连续的
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())

        # 测试使用不支持的内存格式时是否抛出 RuntimeError
        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    # 定义测试函数 test_bmm，用于测试 torch.bmm 方法的不同用法
    def test_bmm(self):
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # 测试：形状不匹配时是否抛出 RuntimeError
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # 测试：左参数被 vmap 包装时的情况
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # 测试：右参数被 vmap 包装时的情况
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # 测试：左右参数均被 vmap 包装时的情况
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
    def test_contiguous(self):
        # 定义操作函数为 Tensor.contiguous
        op = Tensor.contiguous

        # 使用 _test_unary 方法测试 op 函数，生成 CPU 上的随机张量
        self._test_unary(op, TensorFactory.randn, "cpu")

        # 检查如果每个示例已经是连续的，contiguous 函数会返回原始张量
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        # 将张量维度从0移动到2
        x = x.movedim(0, 2)
        # 在维度2上应用 vmap(Tensor.contiguous) 函数
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        # 断言结果和原始张量是同一个对象
        self.assertTrue(result is x)

        # 异常消息：在 vmap 内部查询内存格式的 is_contiguous 功能尚未实现
        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 3)
        # 断言在运行时出现指定异常消息
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)
    def test_stride(self):
        B0 = 3  # 定义变量 B0，并赋值为 3

        x = torch.randn(B0, 2, 5, 7)  # 创建一个形状为 (B0, 2, 5, 7) 的随机张量 x

        def foo(x):
            assert x.stride() == (7 * 5, 7, 1)  # 断言张量 x 的步长为 (35, 7, 1)
            return x

        vmap(foo)(x)  # 对函数 foo 使用 vmap 进行批处理映射

        x = torch.randn(2, B0, 5, 7).movedim(1, 0)  # 创建一个形状为 (2, B0, 5, 7) 的随机张量 x，并在维度 1 和 0 之间移动维度

        def bar(x):
            assert x.stride() == (7 * 5 * B0, 7, 1)  # 断言张量 x 的步长为 (35 * B0, 7, 1)
            return x

        vmap(bar)(x)  # 对函数 bar 使用 vmap 进行批处理映射

    def test_chunk(self):
        test = self._vmap_view_test  # 将 self._vmap_view_test 方法赋值给变量 test
        op = torch.chunk  # 将 torch.chunk 方法赋值给变量 op
        B0, B1, B2 = 7, 11, 13  # 定义变量 B0, B1, B2 并分别赋值为 7, 11, 13

        # 对 torch.split 方法进行测试，分别测试不同参数下的输入维度
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),  # 使用 TensorFactory.randn 方法生成随机张量，并对其进行 clamp 操作，设置最小值为 -0.5
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),  # 使用 TensorFactory.randn 方法生成随机张量，并对其进行 clamp 操作，设置最大值为 0.5
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),  # 使用 TensorFactory.randn 方法生成随机张量，并对其进行 clamp 操作，设置最小值为 -0.5，最大值为 0.5
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),  # 使用 TensorFactory.randn 方法生成随机张量，并对其进行 clamp_min 操作，设置最小值为 -0.5
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),  # 使用 TensorFactory.randn 方法生成随机张量，并对其进行 clamp_max 操作，设置最大值为 0.5
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, "cpu")  # 对每个 clamp 操作 case 进行一元测试

    def test_comparison_ops(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)  # 使用 functools.partial 创建部分函数应用，将 self._vmap_test 方法的 check_propagates_grad 参数设为 False

        getter = TensorFactory.randn  # 将 TensorFactory.randn 方法赋值给 getter
        B0, B1 = 7, 11  # 定义变量 B0, B1 并分别赋值为 7, 11

        ops = (
            torch.eq, lambda x, y: x == y,  # 等于操作和等于运算的匿名函数
            torch.gt, lambda x, y: x > y,  # 大于操作和大于运算的匿名函数
            torch.ge, lambda x, y: x >= y,  # 大于等于操作和大于等于运算的匿名函数
            torch.le, lambda x, y: x <= y,  # 小于等于操作和小于等于运算的匿名函数
            torch.lt, lambda x, y: x < y,  # 小于操作和小于运算的匿名函数
            torch.ne, lambda x, y: x != y,  # 不等于操作和不等于运算的匿名函数
        )

        for op in ops:
            # 单个 vmap 测试：op(Tensor, Tensor)
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

            # 数字作为输入的测试
            number = getter([]).item()  # 生成一个随机张量并将其作为数字存储在变量 number 中
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False  # 对每个操作进行一元测试
            )
    def test_diagonal(self):
        # 创建一个形状为 (3, 5, 7, 11, 13) 的随机张量
        tensor = torch.randn(3, 5, 7, 11, 13)
        # 将测试函数指定为 self._vmap_view_test
        test = self._vmap_view_test
        # 将操作指定为 torch.diagonal
        op = torch.diagonal
        # 执行测试，使用不同的参数组合
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        # 对张量的每个元素应用 op 函数，其中 op 的参数为 (t, 0, 0, -1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        # 对张量的每个元素应用 vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )

    def test_dot(self):
        # 将操作指定为 torch.dot
        op = torch.dot
        # 将测试函数指定为 self._vmap_test
        test = self._vmap_test
        B0, B1 = 7, 11

        # 检查形状不匹配的异常
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # 左边参数被 vmap 包装
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # 右边参数被 vmap 包装
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # 两个参数都被 vmap 包装
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        # 将操作指定为 torch.Tensor.expand_as
        op = torch.Tensor.expand_as
        # 将测试函数指定为 self._vmap_view_test
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        # 执行测试，使用不同的参数组合
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))
    def test_fill_and_zero_inplace(self):
        # 使用 functools.partial 创建一个部分应用函数 test，传入参数 check_propagates_grad=False
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        # 定义两个变量 B0 和 B1，分别赋值为 7 和 11
        B0, B1 = 7, 11
        # ops 列表包含三个 lambda 表达式，每个 lambda 表达式代表一个操作函数
        ops = (
            lambda t: t.fill_(0.1),  # 使用标量值填充张量 t
            lambda t: t.fill_(torch.tensor(0.2)),  # 使用张量值填充张量 t
            lambda t: t.zero_(),  # 将张量 t 的所有元素置零
        )

        # 对 ops 中的每个操作函数进行迭代
        for op in ops:
            # 对单个 vmap 进行测试，不同的 in_dims 和 out_dims
            test(op, [TensorFactory.randn([B0, 3])])  # 使用 op 函数对形状为 [B0, 3] 的随机张量进行测试
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)  # 使用 op 函数对形状为 [2, 5, B0, 3] 的随机张量进行测试，指定 in_dims=2
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)  # 使用 op 函数对形状为 [2, 5, B0, 3] 的随机张量进行测试，指定 in_dims=2 和 out_dims=2

            # 对双重嵌套的 vmap 进行测试
            test(vmap(op), [TensorFactory.randn([B0, B1])])  # 使用 op 函数对形状为 [B0, B1] 的随机张量进行测试
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)  # 使用 op 函数对形状为 [B1, 2, 5, B0, 3] 的随机张量进行测试，指定 in_dims=2
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )  # 使用 op 函数对形状为 [2, 5, B0, B1, 3] 的随机张量进行测试，指定 in_dims=2 和 out_dims=2

        # 对于 fill_ 操作，当值是一个批次张量时进行测试
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        # 使用 assertRaisesRegex 检测是否抛出 RuntimeError，检查输出形状与广播形状是否匹配
        with self.assertRaisesRegex(
            RuntimeError, r"output with shape .+ doesn't match the broadcast shape"
        ):
            # 当被写入的张量未被 vmap 包装时抛出 RuntimeError
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

    def _test_complex_views(self, op, dtypes):
        # 使用 self._vmap_view_test 函数测试复杂视图操作
        test = self._vmap_view_test

        def run_test(op, dtype):
            # 定义 get 函数，根据给定的形状创建指定数据类型的随机张量
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # 对单个 vmap 进行测试，不同的 in_dims / out_dims
            test(op, [get([B0, 3])])  # 使用 op 函数对形状为 [B0, 3] 的随机张量进行测试
            test(op, [get([3, B0])], in_dims=1)  # 使用 op 函数对形状为 [3, B0] 的随机张量进行测试，指定 in_dims=1
            test(op, [get([2, 5, B0, 3])], in_dims=2)  # 使用 op 函数对形状为 [2, 5, B0, 3] 的随机张量进行测试，指定 in_dims=2
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)  # 使用 op 函数对形状为 [2, 5, B0, 3] 的随机张量进行测试，指定 in_dims=2 和 out_dims=2

            # 对双重嵌套的 vmap 进行测试
            test(vmap(op), [get([B0, B1])])  # 使用 op 函数对形状为 [B0, B1] 的随机张量进行测试
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)  # 使用 op 函数对形状为 [B1, 2, 5, 3, B0] 的随机张量进行测试，指定 in_dims=4
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)  # 使用 op 函数对形状为 [2, 5, B0, B1, 3] 的随机张量进行测试，指定 in_dims=2 和 out_dims=2

        # 对 dtypes 列表中的每种数据类型进行测试
        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        # 使用 torch.real 函数测试复杂视图操作，数据类型为 torch.cfloat 和 torch.cdouble
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        # 使用 torch.imag 函数测试复杂视图操作，数据类型为 torch.cfloat 和 torch.cdouble
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        # 使用 torch.view_as_real 函数测试复杂视图操作，数据类型为 torch.cfloat 和 torch.cdouble
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )
    # 定义一个测试方法，用于测试 torch.view_as_complex 函数的不同情况
    def test_view_as_complex(self):
        # 定义一个内部方法，用于测试给定数据类型的不同形状的张量生成
        def run_test(dtype):
            # 定义一个生成指定形状张量的方法
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            # 获取 torch.view_as_complex 函数的引用
            op = torch.view_as_complex
            # 获取测试方法的引用
            test = self._vmap_view_test
            # 定义两个批次大小
            B0, B1 = 7, 11

            # 单层 vmap，不同的输入维度和输出维度
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # 双层嵌套 vmap
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # 感兴趣的案例 #1：批次维度直接在尺寸为2的维度之前
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # 感兴趣的案例 #2：张量末尾的批次维度，成功的情况
            # view_as_complex 要求尺寸为2的维度具有步幅1，以便视图正常工作
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # 感兴趣的案例 #3：张量末尾的批次维度，失败的情况
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # 无效的输入：没有尺寸为2的维度
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # 无效的输入：批次维度大小为2，但逻辑上的最后一个维度不是尺寸为2
            msg = "Tensor must have a last dimension of size 2"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        # 对于每种数据类型执行测试方法
        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    # 测试是否为复数
    def test_is_complex(self):
        # 创建一个复数类型的张量和一个普通张量
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        # 定义一个函数，用于检查张量是否为复数，并返回相应的标志
        def foo(x):
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        # 使用 vmap 应用 foo 函数，预期复数张量返回1，普通张量返回0
        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))
    # 测试函数，用于检查输入张量是否为浮点数张量，返回结果为每个元素的浮点数判断结果组成的张量
    def test_is_floating_point(self):
        # 创建一个浮点数张量
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        # 创建一个整数张量
        long_tensor = torch.tensor([1, 2, 3])

        # 内部函数，用于判断输入张量元素是否为浮点数，返回结果为每个元素的判断结果组成的张量
        def foo(x):
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        # 使用 vmap 函数将 foo 函数映射到 float_tensor 上，期望输出每个元素都为 1 的张量
        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        # 使用 vmap 函数将 foo 函数映射到 long_tensor 上，期望输出每个元素都为 0 的张量
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    # 测试函数，用于检查输入张量是否为连续张量，返回结果为每个张量是否连续的判断结果组成的张量
    def test_is_contiguous(self):
        # 内部函数，用于判断输入张量是否连续，返回结果为每个张量是否连续的判断结果组成的张量
        def foo(x):
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        B0, B1 = 3, 5

        # 创建一个连续的张量，单一批次维度
        contig = torch.randn(B0, 2, 7)
        # 使用 vmap 函数将 foo 函数映射到 contig 上，期望输出每个批次是否连续的判断结果组成的张量
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        # 创建一个非连续的张量，单一批次维度
        noncontig = torch.randn(2, B0, 7)
        # 使用 vmap 函数将 foo 函数映射到 noncontig 上，指定输入维度为 1，期望输出每个批次是否连续的判断结果组成的张量
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        # 对张量进行维度转换，使其变为非连续
        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        # 使用 vmap 函数将 foo 函数映射到 noncontig 上，期望输出每个批次是否连续的判断结果组成的张量
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        # 创建一个非连续的张量，单一批次维度
        noncontig = torch.randn(2, 7, B0)
        # 使用 vmap 函数将 foo 函数映射到 noncontig 上，指定输入维度为 2，期望输出每个批次是否连续的判断结果组成的张量
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # 创建一个连续的张量，多批次维度
        contig = torch.randn(B0, B1, 3)
        # 使用 vmap 函数将 foo 函数嵌套映射到 contig 上，期望输出每个元素是否连续的判断结果组成的张量
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        # 创建一个连续的张量，多批次维度，批次维度交换顺序
        contig = torch.randn(B1, B0, 3)
        # 使用 vmap 函数将 foo 函数嵌套映射到 contig 上，指定输入维度为 1，期望输出每个元素是否连续的判断结果组成的张量
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        # 对张量进行维度转换，使其变为非连续，批次维度交换顺序
        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        # 使用 vmap 函数将 foo 函数嵌套映射到 contig 上，期望输出每个元素是否连续的判断结果组成的张量
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        # 创建一个非连续的张量，多批次维度
        noncontig = torch.randn(B0, 3, B1)
        # 使用 vmap 函数将 foo 函数嵌套映射到 noncontig 上，指定输入维度为 1，期望输出每个元素是否连续的判断结果组成的张量
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # 对空张量进行 is_contiguous 判断应为 True
        # 内部函数，用于检查输入张量是否连续，断言应为连续张量
        def bar(x):
            assert x.is_contiguous()
            return x

        # 使用 vmap 函数将 bar 函数映射到空张量上，期望不引发异常
        vmap(bar)(torch.randn(B0, 0, 3))
        # 使用 vmap 函数将 bar 函数映射到空张量上，指定输入维度为 1，期望不引发异常
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        # 对空张量进行维度转换，使其变为非连续，应为 False
        vmap(bar)(torch.randn(B0, 0, 3).mT)

        # 内部函数，用于检查输入张量是否连续，但针对指定的内存格式
        def baz(x, memory_format):
            x.is_contiguous(memory_format=memory_format)
            return x

        # 使用 vmap 函数将 baz 函数部分应用到张量上，期望引发 RuntimeError 异常，提示内存格式查询不支持
        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)
    # 定义测试方法 test_movedim，self 参数表示该方法是类的成员方法
    def test_movedim(self):
        # 将 torch.movedim 赋值给 op 变量，以简化代码中的调用
        op = torch.movedim
        # 将 self._vmap_view_test 方法赋值给 test 变量，以简化代码中的调用
        test = self._vmap_view_test
        # 定义三个整数变量 B0, B1, B2 分别为 7, 11, 13，用于后续测试参数的维度设置

        # movedim(tensor, int, int) 变体
        # 测试 movedim 方法，传入参数为 (torch.rand(B0, 2, 5), 0, 1)，设置输入维度为 (0, None, None)
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        # 测试 movedim 方法，传入参数为 (torch.rand(2, B0, 5), 0, 1)，设置输入维度为 (1, None, None)
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        # 使用 vmap 对 movedim 方法进行映射，传入参数为 (torch.rand(B1, 2, B0, 5), 0, 1)，设置输入维度为 (2, None, None)
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap 对 movedim 方法进行两次映射，传入参数为 (torch.rand(B1, 2, B0, 5, B2), 0, 1)，设置输入维度为 (2, None, None)
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) 变体
        # 测试 movedim 方法，传入参数为 (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2])，设置输入维度为 (0, None, None)
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        # 测试 movedim 方法，传入参数为 (torch.rand(2, 3, B0, 5), [1, 0], [0, 2])，设置输入维度为 (1, None, None)
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        # 使用 vmap 对 movedim 方法进行映射，传入参数为 (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0])，设置输入维度为 (2, None, None)
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        # 嵌套使用 vmap 对 movedim 方法进行两次映射，传入参数为 (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0])，设置输入维度为 (2, None, None)
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )

    # 定义测试方法 test_mm，self 参数表示该方法是类的成员方法
    def test_mm(self):
        # 将 torch.mm 赋值给 op 变量，以简化代码中的调用
        op = torch.mm
        # 将 self._vmap_test 方法赋值给 test 变量，以简化代码中的调用
        test = self._vmap_test
        # 定义两个整数变量 B0, B1 分别为 7, 11，用于后续测试参数的维度设置

        # 形状不匹配
        msg = "Shape mismatch"
        # 使用 assertRaisesRegex 检查运行时异常 RuntimeError，并验证异常消息为 "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            # 对 vmap(op) 进行测试，传入参数为 (torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            # 对 vmap(op, in_dims=(0, None)) 进行测试，传入参数为 (torch.randn(B0, 2), torch.randn(2, 2))
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            # 对 vmap(op, in_dims=(None, 0)) 进行测试，传入参数为 (torch.randn(2, 2), torch.randn(B0, 2, 2, 2))
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # 左参数进行 vmap 映射
        # 测试 op 方法，传入参数为 (torch.rand(B0, 2, 5), torch.rand(5, 2))，设置输入维度为 (0, None)
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        # 使用 vmap 对 op 方法进行映射，传入参数为 (torch.rand(B1, B0, 2, 5), torch.rand(5, 2))，设置输入维度为 (1, None)
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # 右参数进行 vmap 映射
        # 测试 op 方法，传入参数为 (torch.rand(2, 5), torch.rand(B0, 5, 2))，设置输入维度为 (None, 0)
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        # 使用 vmap 对 op 方法进行映射，传入参数为 (torch.rand(2, 5), torch.rand(B1, B0, 5, 2))，设置输入维度为 (None, 1)
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # 左右参数均进行 vmap 映射
        # 测试 op 方法，传入参数为 (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2))
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        # 使用 vmap 对 op 方法进行映射，传入参数为 (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2))，设置输入维度为 (1, 0)
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        # 使用 vmap 对 op 方法进行映射，传入参数为 (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2))，设置输入维度为 (
    def test_mv(self):
        # 使用 torch.mv 作为操作符
        op = torch.mv
        # 使用 self._vmap_test 作为测试函数
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch 的错误消息
        msg = "Shape mismatch"
        # 验证在给定的维度下是否抛出 RuntimeError，并匹配错误消息
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # 左边参数进行 vmap 映射
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # 右边参数进行 vmap 映射
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # 左右参数均进行 vmap 映射
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_narrow(self):
        # 使用 torch.narrow 作为操作符
        op = torch.narrow
        # 使用 self._vmap_view_test 作为测试函数
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # 针对不同维度进行 vmap 映射
        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

    def test_new_empty(self):
        # 使用 Tensor.new_empty 作为操作符
        op = Tensor.new_empty
        # Empty 操作的输出在形状上是不确定的，所以仅检查输出张量的形状是否符合预期，并且不会使用 vmap 的后备功能
        B0, B1 = 7, 11

        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])
    def test_new_empty_strided(self):
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11
        
        # 定义内部函数，用于测试单层 vmap 的情况
        def _test_single_vmap(size, stride, B0):
            # 创建 B0 个随机张量
            x = torch.randn(B0)
            # 对张量 x 应用 vmap 和 new_empty_strided 方法
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            # 获取使用 empty_strided 创建的存储大小 S
            S = torch.empty_strided(size, stride).storage().size()
            # 断言结果的形状应为 [B0] + size
            self.assertEqual(result.shape, [B0] + size)
            # 断言结果的步长应为 [S] + stride
            self.assertEqual(result.stride(), [S] + stride)
        
        # 定义内部函数，用于测试双层 vmap 的情况
        def _test_double_vmap(size, stride, B0, B1):
            # 创建形状为 (B0, B1) 的随机张量
            x = torch.randn(B0, B1)
            # 对张量 x 应用两次 vmap 和 new_empty_strided 方法
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            # 获取使用 empty_strided 创建的存储大小 S
            S = torch.empty_strided(size, stride).storage().size()
            # 断言结果的形状应为 [B0, B1] + size
            self.assertEqual(result.shape, [B0, B1] + size)
            # 断言结果的步长应为 [B1 * S, S] + stride
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            # 创建形状为 (B1, B0) 的随机张量
            x = torch.randn(B1, B0)
            # 对张量 x 应用两次 vmap 和 new_empty_strided 方法（在第二维度）
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
            # 获取使用 empty_strided 创建的存储大小 S
            S = x.new_empty_strided(size, stride).storage().size()
            # 断言结果的形状应为 [B0, B1] + size
            self.assertEqual(result.shape, [B0, B1] + size)
            # 断言结果的步长应为 [B1 * S, S] + stride
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # contiguous case
        # 测试连续情况
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # expanded
        # 测试扩展情况
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # some of these cases are pretty strange, just verifying that if
        # empty_strided allows them then BatchedTensor.new_empty_strided
        # can as well
        # 一些情况可能比较奇怪，只是验证如果 empty_strided 允许它们，那么 BatchedTensor.new_empty_strided 也能允许
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)
    def test_stack(self):
        # 将 self._vmap_test 赋值给 test 变量
        test = self._vmap_test
        # 定义两个常量 B0 和 B1 分别赋值为 5 和 7

        # 定义一个函数 get_op(dim)，返回一个操作函数 op
        def get_op(dim):
            # op 函数接受任意数量的张量参数，使用 torch.stack 在指定维度 dim 上堆叠这些张量
            def op(*tensors):
                return torch.stack(tensors, dim=dim)

            return op

        # 使用 test 函数执行 get_op(0) 返回的操作函数，传入两个随机张量作为参数
        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        # 使用 test 函数执行 get_op(0) 返回的操作函数，传入一个形状为 (3,) 和 (B0, 3) 的随机张量作为参数，指定 in_dims=(None, 0)
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        # 使用 test 函数执行 get_op(0) 返回的操作函数，传入一个形状为 (2, 17) 和 (2, 17, B0) 的随机张量作为参数，指定 in_dims=(None, 2)
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        # 使用 test 函数执行 get_op(-1) 返回的操作函数，传入一个形状为 (2, 17) 和 (2, 17, B0) 的随机张量作为参数，指定 in_dims=(None, 2)
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        # 使用 vmap(get_op(0), in_dims=(0, None)) 返回的操作函数，传入一个形状为 (B1, 2) 和 (B0, 2) 的随机张量作为参数，指定 in_dims=(None, 0)
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 2)),
            in_dims=(None, 0),
        )
        # 使用 vmap(get_op(0), in_dims=(0, 0)) 返回的操作函数，传入一个形状为 (B1, 2) 和 (B0, B1, 2) 的随机张量作为参数，指定 in_dims=(None, 0)
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 2)),
            in_dims=(None, 0),
        )

    def test_slice(self):
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        # 定义三个常量 B0, B1, B2 分别赋值为 7, 11, 13

        # 使用 test 函数执行 lambda 函数 t: t[0:1]，传入一个形状为 (B0, 3, 5) 的随机张量作为参数
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        # 使用 test 函数执行 lambda 函数 t: t[:, 1:3]，传入一个形状为 (3, 5, B0) 的随机张量作为参数，指定 in_dims=2
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        # 使用 vmap(lambda t: t[:, 0:1], in_dims=2) 返回的操作函数，传入一个形状为 (3, 5, B0, B1) 的随机张量作为参数，指定 in_dims=2
        test(
            vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2
        )
        # 使用 vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2) 返回的操作函数，传入一个形状为 (3, 5, B0, B1, B2) 的随机张量作为参数，指定 in_dims=2
        test(
            vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
            (torch.rand(3, 5, B0, B1, B2),),
            in_dims=2,
        )

    def test_squeeze(self):
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        # 将 torch.squeeze 函数赋值给 op 变量
        op = torch.squeeze
        # 定义两个常量 B0 和 B1 分别赋值为 1 和 11

        # 使用 test 函数执行 torch.squeeze 函数，传入一个形状为 (B0,) 的随机张量作为参数
        test(op, (torch.rand(B0),))
        # 使用 test 函数执行 torch.squeeze 函数，传入一个形状为 (B0, 3, 5) 的随机张量作为参数
        test(op, (torch.rand(B0, 3, 5),))
        # 使用 test 函数执行 torch.squeeze 函数，传入一个形状为 (1, B0, 5) 的随机张量作为参数，指定 in_dims=1
        test(op, (torch.rand(1, B0, 5),), in_dims=1)
        # 使用 test 函数执行 torch.squeeze 函数，传入一个形状为 (B0, 0, 1, 5, 1) 的随机张量作为参数
        test(op, (torch.rand(B0, 0, 1, 5, 1),))
        # 使用 test 函数执行 torch.squeeze 函数，传入一个形状为 (B0, 1, 1, 1, 1) 的随机张量作为参数
        test(op, (torch.rand(B0, 1, 1, 1, 1),))
        # 使用 test 函数执行 vmap(torch.squeeze) 返回的操作函数，传入一个形状为 (B0, B1, 1) 的随机张量作为参数
        test(vmap(op), (torch.rand(B0, B1, 1),))
        # 使用 test 函数执行 vmap(torch.squeeze) 返回的操作函数，传入一个形状为 (B1, 1, B0) 的随机张量作为参数，指定 in_dims=2
        test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)

    def test_sum_dim(self):
        # 将 self._vmap_test 赋值给 test 变量
        test = self._vmap_test
        # 定义两个常量 B0 和 B1 分别赋值为 5 和 7

        # 使用 test 函数执行 lambda 函数 x: x.sum(())，传入一个形状为 [B0] 的随机张量作为参数
        test(lambda x: x.sum(()), [torch.randn([B0])])
        # 使用 test 函数执行 lambda 函数 x: x.sum(())，传入一个形状为 [B0, 2] 的随机张量作为参数
        test(lambda x: x.sum(()), [torch.randn([B0, 2])])
        # 使用 test 函数执行 lambda 函数 x: x.sum(0)，传入一个形状为 [B0] 的随机张量作为参数
        test(lambda x: x.sum(0), [torch.randn([B0])])
        # 使用 test 函数执行 lambda 函数 x: x.sum(-1)，传入一个形状为 [B0] 的随机张量作为参数
        test(lambda x: x.sum(-1), [torch.randn([B0])])
        # 使用 test 函数执行 lambda 函数 x: x.sum(0)，传入一个形状为 [B0, 3] 的随机张量作为参数
        test(lambda x: x.sum(0), [torch.randn([B0, 3])])
        # 使用 test 函数执行 lambda 函数 x: x.sum(-1)，
    # 定义测试函数 test_reshape，用于测试 reshape 操作
    def test_reshape(self):
        # 将 self._vmap_test 赋值给 test，简化代码
        test = self._vmap_test
        # 定义三个维度 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 定义操作 op 为 torch.reshape
        op = torch.reshape
        # 测试 torch.reshape 操作，传入参数为 (torch.rand(B0, 2 * 5), [2, 5])，in_dims=(0, None)，check_view=True
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        # 测试 torch.reshape 操作，传入参数为 (torch.rand(2, B0, 5), [1, 1, 10])，in_dims=(1, None)，check_view=False
        test(
            op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False
        )
        # 使用 vmap 对 lambda 函数进行批处理，对每个张量 t 执行 t.reshape([-1]) 操作
        test(
            vmap(lambda t: t.reshape([-1])),
            (torch.rand(B0, B1, 2, 5),),
            check_view=True,
        )
        # 嵌套使用 vmap 对 op 进行批处理，其中 op 是 lambda 函数 t.reshape([-1]) 的 vmap 包装
        test(
            vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
            (torch.rand(3, B1, 2, B2, 5, B0),),
            in_dims=5,
            check_view=False,
        )

    # 定义测试函数 test_reshape_as，用于测试 reshape_as 操作
    def test_reshape_as(self):
        # 将 self._vmap_test 赋值给 test，简化代码
        test = self._vmap_test
        # 定义三个维度 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 定义操作 op 为 torch.Tensor.reshape_as
        op = torch.Tensor.reshape_as
        # 测试 torch.Tensor.reshape_as 操作，传入参数为 (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5))，check_view=True
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        # 测试 torch.Tensor.reshape_as 操作，传入参数为 (torch.rand(2 * 5), torch.rand(B0, 2, 5))，in_dims=(None, 0)，check_view=True
        test(
            op,
            (torch.rand(2 * 5), torch.rand(B0, 2, 5)),
            in_dims=(None, 0),
            check_view=True,
        )
        # 测试 torch.Tensor.reshape_as 操作，传入参数为 (torch.rand(B0, 2 * 5), torch.rand(2, 5))，in_dims=(0, None)，check_view=True
        test(
            op,
            (torch.rand(B0, 2 * 5), torch.rand(2, 5)),
            in_dims=(0, None),
            check_view=True,
        )
        # 测试 torch.Tensor.reshape_as 操作，传入参数为 (torch.rand(2, B0, 5), torch.rand(1, 1, 10))，in_dims=(1, None)，check_view=False
        test(
            op,
            (torch.rand(2, B0, 5), torch.rand(1, 1, 10)),
            in_dims=(1, None),
            check_view=False,
        )
        # 使用 vmap 对 op 进行批处理
        test(
            vmap(op),
            (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)),
            check_view=True,
        )
        # 嵌套使用 vmap 对 op 进行批处理，其中 op 是 lambda 函数 vmap(op, in_dims=(2, None)) 的 vmap 包装
        test(
            vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
            (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(5, 0),
            check_view=False,
        )
    # 定义一个测试函数，用于测试特定操作的结果类型
    def test_result_type(self):
        # 定义一个内部函数，返回一个操作的结果类型为标量张量的函数
        def scalar_tensor_with_dtype(op):
            # 定义一个包装函数，调用给定操作并返回指定 dtype 的全 1 标量张量
            def wrapped(*args, **kwargs):
                # 调用给定操作获取其返回的 dtype
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)

            return wrapped

        # 将测试函数 _vmap_test 赋给 test
        test = self._vmap_test
        # 使用 scalar_tensor_with_dtype 函数创建 op 函数，操作为 torch.result_type
        op = scalar_tensor_with_dtype(torch.result_type)

        # 定义常量 B0，值为 2
        B0 = 2

        # 调用 test 函数测试 op 函数对随机张量和指定 dtype 的随机张量的处理
        test(
            op,
            (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        # 使用 lambda 函数测试 op 函数对随机张量和标量的处理
        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

        # 使用 lambda 函数测试 op 函数对随机张量和标量张量的处理
        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )

        # 调用 test 函数测试 op 函数对二维随机张量和指定 dtype 的二维随机张量的处理
        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        # 使用 lambda 函数测试 op 函数对二维随机张量和标量的处理
        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

        # 使用 lambda 函数测试 op 函数对二维随机张量和标量张量的处理
        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )

        # 调用 test 函数测试 op 函数对二维随机张量和指定 dtype 的随机张量的处理
        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

    # 根据条件跳过 Torch Dynamo 测试，因为速度太慢
    @skipIfTorchDynamo("too slow")
    # 定义测试函数 test_tensor_split
    def test_tensor_split(self):
        # 将 self._vmap_view_test 赋值给 test，简化测试调用
        test = self._vmap_view_test
        # 将 torch.tensor_split 赋值给 op，简化调用
        op = torch.tensor_split
        # 定义三个变量 B0, B1, B2 分别赋值为 7, 11, 13，用于后续测试数据的维度设定

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        # 测试1：使用 torch.rand 生成 B0 行 2 列 1024 列的随机张量，分割成 5 部分，按最后一个维度（-1）进行切割
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        # 测试2：使用 torch.rand 生成 2 行 B0 列 1024 列的随机张量，分割成 150 部分，按第二个维度（1）进行切割
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        # 测试3：对 vmap(op) 进行测试，使用 torch.rand 生成 B1 行 1023 列 B0 列 5 列的随机张量，分割成 256 部分，按第一个维度（0）进行切割
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        # 测试4：对 vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)) 进行测试，使用 torch.rand 生成 B1 行 2 列 B0 列 64 列 B2 列的随机张量，按第二个维度（2）进行切割，每个切分大小为 4
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        # 测试5：使用 torch.rand 生成 B0 行 2 列 1024 列的随机张量，按最后一个维度（-1）进行切割，切分大小分别为 [50, 100, 378, 890]
        test(
            op,
            (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1),
            in_dims=(0, None, None),
        )
        # 测试6：使用 torch.rand 生成 2 行 B0 列 1024 列的随机张量，按第二个维度（1）进行切割，切分大小分别为 [50, 100, 212, 345, 0, 378, 890]
        test(
            op,
            (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1),
            in_dims=(1, None, None),
        )
        # 测试7：对 vmap(op) 进行测试，使用 torch.rand 生成 B1 行 1023 列 B0 列 5 列的随机张量，按第一个维度（2）进行切割，切分大小分别为 [50, 100, 212, 345, 0, 378, 890]
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
            in_dims=(2, None, None),
        )
        # 测试8：对 vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)) 进行测试，使用 torch.rand 生成 B1 行 2 列 B0 列 64 列 B2 列的随机张量，按第二个维度（2）进行切割，切分大小分别为 [4, 8, 9, 34, 29]
        test(
            vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    # 定义测试函数 test_split
    def test_split(self):
        # 将 self._vmap_view_test 赋值给 test，简化测试调用
        test = self._vmap_view_test
        # 将 torch.split 赋值给 op，简化调用
        op = torch.split
        # 定义三个变量 B0, B1, B2 分别赋值为 7, 11, 13，用于后续测试数据的维度设定

        # tests for torch.split(self, split_size: int, dim)
        # 测试1：使用 torch.rand 生成 B0 行 2 列 1024 列的随机张量，按最后一个维度（-1）进行切割，每个切分大小为 101
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        # 测试2：使用 torch.rand 生成 2 行 B0 列 1024 列的随机张量，按第二个维度（1）进行切割，每个切分大小为 130
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        # 测试3：对 vmap(op) 进行测试，使用 torch.rand 生成 B1 行 1023 列 B0 列 5 列的随机张量，按第一个维度（0）进行切割，每个切分大小为 256
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        # 测试4：对 vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)) 进行测试，使用 torch.rand 生成 B1 行 2 列 B0 列 64 列 B2 列的随机张量，按第二个维度（2）进行切割，每个切分大小为 4
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.split(self, split_size: List[int], dim)
        # 测试5：使用 torch.rand 生成 B0 行 2 列 1024 列的随机张量，按最后一个维度（-1）进行切割，每个切分大小分别为 [1, 1020, 3]
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        # 测试6：使用 torch.rand 生成 2 行 B0 列 1024 列的随机张量，按第二个维度（1）进行切割，每个切分大小分别为 [100] * 10 + [24]
        test(
            op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None)
        )
        # 测试7：对 vmap(op) 进行测试，使用 torch.rand 生成 B1 行 1023 列 B0 列 5 列的随机张量，
    def test_trace(self):
        # 将 torch.trace 赋值给 op 变量
        op = torch.trace
        # 将 self._vmap_test 赋值给 test 变量
        test = self._vmap_test
        # 定义测试用的维度 B0, B1, B2
        B0, B1, B2 = 7, 11, 13

        # 测试 torch.trace 方法
        test(op, (torch.rand(B0, 2, 5),))
        # 测试 torch.trace 方法，指定输入维度为 1
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        # 测试 vmap(op) 方法，输入为 torch.rand(B1, 2, B0, 5)，指定输入维度为 2
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 测试 vmap(vmap(op, in_dims=2)) 方法，输入为 torch.rand(B1, 2, B0, 5, B2)，指定输入维度为 2
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_transpose(self):
        # 将 torch.transpose 赋值给 op 变量
        op = torch.transpose
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test

        # 定义测试用的维度 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试 lambda 函数，对输入张量使用 op(x, 0, 1)
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        # 测试 lambda 函数，对输入张量使用 op(x, -1, -2)
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        # 测试 lambda 函数，对输入张量使用 op(x, 3, 1)
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        # 测试 lambda 函数，对输入张量使用 op(x, 1, 0)，指定输入维度为 1
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        # 测试 vmap(lambda x: op(x, 0, 1)) 方法，输入为 torch.rand(B1, 2, B0, 5)，指定输入维度为 2
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 测试 vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)) 方法，输入为 torch.rand(B1, 2, B0, 5, B2)，指定输入维度为 2
        test(
            vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 5, B2),),
            in_dims=2,
        )

        # 特殊情况：标量张量
        for dim1, dim2 in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            # 使用 vmap(lambda x: op(x, dim1, dim2)) 测试标量张量 x
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            # 断言结果与原始张量 x 相同
            self.assertTrue(result is x)

    def test_t(self):
        # 将 torch.t 赋值给 op 变量
        op = torch.t
        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        # 定义测试用的维度 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试 torch.t 方法
        test(op, (torch.rand(B0, 2, 5),))
        # 测试 torch.t 方法，指定输入维度为 1
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        # 测试 vmap(op) 方法，输入为 torch.rand(B1, 2, B0, 5)，指定输入维度为 2
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 测试 vmap(vmap(op, in_dims=2)) 方法，输入为 torch.rand(B1, 2, B0, 5, B2)，指定输入维度为 2
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_T_numpy(self):
        # 定义 op 函数，返回输入张量的转置
        def op(t):
            return t.T

        # 将 self._vmap_view_test 赋值给 test 变量
        test = self._vmap_view_test
        # 定义测试用的维度 B0, B1, B2
        B0, B1, B2 = 7, 11, 13
        # 测试 op 函数，输入为 torch.rand(B0, 2, 3, 5)
        test(op, (torch.rand(B0, 2, 3, 5),))
        # 测试 op 函数，输入为 torch.rand(2, B0, 3, 5)，指定输入维度为 1
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        # 测试 vmap(op) 方法，输入为 torch.rand(B1, 2, B0, 5)，指定输入维度为 2
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        # 测试 vmap(op) 方法，输入为 torch.rand(B1, 2, B0, 3, 5)，指定输入维度为 2
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        # 测试 vmap(vmap(op, in_dims=2)) 方法，输入为 torch.rand(B1, 2, B0, 3, B2, 5)，指定输入维度为 2
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        # 将 self._vmap_test 赋值给 test 变量
        test = self._vmap_test
        # 定义测试用的维度 B0, B1
        B0, B1 = 7, 11

        # 测试 lambda 函数，将张量转移到 CPU
        test(lambda t: t.to("cpu"), (torch.rand(B0),))
        # 测试 lambda 函数，将张量转换为 double 类型
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        # 测试 lambda 函数，将张量 t 转换为与张量 o 相同的类型
        test(
            lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64))
        )
        # 测试 lambda 函数，将张量 t 转换为与张量 o 相同的类型，指定输入维度为 (0, None)
        test(
            lambda t, o: t.to(o),
            (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
            in_dims=(0, None),
        )
        # 测试 vmap(lambda t: t.to(torch.double)) 方法，输入为 torch.rand(B0, B1, 3)
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))

        # 还测试一些类型转换的方法
        # 测试 lambda 函数，将张量转换为 double 类型
        test(lambda t: t.double(), (torch.rand(B0),))
        # 测试 lambda 函数，将张量转换为 float 类型
    def test_unfold(self):
        # 将 torch.Tensor.unfold 方法赋给 op 变量
        op = torch.Tensor.unfold
        # 将 self._vmap_view_test 方法赋给 test 变量
        test = self._vmap_view_test
        # 定义三个整数变量 B0, B1, B2 分别为 3, 2, 5
        B0, B1, B2 = 3, 2, 5

        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 7, 11), 0, 2, 1)，设置 in_dims=(0, None, None, None)
        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(7, B0, 11), 1, 4, 2)，设置 in_dims=(1, None, None, None)
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        # 调用 vmap 方法，传入 op 方法及参数 (torch.rand(B1, 7, B0, 11), 1, 5, 1)，设置 in_dims=(2, None, None, None)
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 7, B0, 11), 1, 5, 1),
            in_dims=(2, None, None, None),
        )
        # 调用 vmap 方法，传入嵌套的 vmap 方法及参数 (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4)，设置 in_dims=(2, None, None, None)
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4),
            in_dims=(2, None, None, None),
        )

    def test_unbind(self):
        # 将 self._vmap_view_test 方法赋给 test 变量
        test = self._vmap_view_test
        # 将 torch.unbind 方法赋给 op 变量
        op = torch.unbind
        # 定义三个整数变量 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13

        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 2, 1024), -1)，设置 in_dims=(0, None)
        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 2, 0),)
        test(op, (torch.rand(B0, 2, 0),))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(2, B0, 7), 0)，设置 in_dims=(1, None)
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        # 调用 vmap 方法，传入 op 方法及参数 (torch.rand(B1, 1023, B0, 5), 1)，设置 in_dims=(2, None)
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 1023, B0, 5), 1),
            in_dims=(2, None),
        )
        # 调用 vmap 方法，传入嵌套的 vmap 方法及参数 (lambda t: op(t, dim=1), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
            (torch.rand(B1, 2, B0, 32, B2),),
            in_dims=2,
        )

    def test_view(self):
        # 将 self._vmap_view_test 方法赋给 test 变量
        test = self._vmap_view_test
        # 定义三个整数变量 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 将 torch.Tensor.view 方法赋给 op 变量
        op = torch.Tensor.view

        # 抛出 RuntimeError 异常，如果 view 操作会产生不正确的结果
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])

        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 2 * 5), [2, 5])，设置 in_dims=(0, None)
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 4, 5), [1, 2, 1, 10])，设置 in_dims=(0, None)
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        # 调用 vmap 方法，传入 lambda 函数进行 view 操作，参数为 (torch.rand(B0, B1, 2, 5, 3),)
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        # 调用 vmap 方法，传入嵌套的 vmap 方法及参数 (lambda t: t.reshape([-1]), in_dims=1)
        test(
            vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
            (torch.rand(B2, B0, B1, 3, 2, 5),),
            in_dims=1,
        )

    def test_view_as(self):
        # 将 self._vmap_view_test 方法赋给 test 变量
        test = self._vmap_view_test
        # 定义三个整数变量 B0, B1, B2 分别为 7, 11, 13
        B0, B1, B2 = 7, 11, 13
        # 将 torch.Tensor.view_as 方法赋给 op 变量
        op = torch.Tensor.view_as

        # 抛出 RuntimeError 异常，如果 view 操作会产生不正确的结果
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))

        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(2 * 5), torch.rand(B0, 2, 5))，设置 in_dims=(None, 0)
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 2 * 5), torch.rand(2, 5))，设置 in_dims=(0, None)
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))

        # 调用 test 方法，传入 op 方法及参数 (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10))，设置 in_dims=(0, None)
        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))

        # 调用 vmap 方法，传入 op 方法进行批量操作，参数为 (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10))
        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        # 调用 vmap 方法，传入嵌套的 vmap 方法及
    # 为给定输出和批处理大小构造一个具有随机正态分布的张量
    return torch.randn(
        batch_size, *output.shape, dtype=output.dtype, device=output.device
    )


def as_tuple(x):
    # 如果输入是元组，则直接返回
    if isinstance(x, tuple):
        return x
    # 如果输入是列表，则将其转换为元组后返回
    elif isinstance(x, list):
        return tuple(x)
    # 否则，将输入包装成单元素元组后返回
    else:
        return (x,)


def differentiable(args):
    # 从给定参数中选择出所有是 torch.Tensor 类型且需要梯度计算的元素，返回为元组
    return tuple(
        arg
        for arg in as_tuple(args)
        if isinstance(arg, torch.Tensor) and arg.requires_grad
    )


def _get_rand_no_zeros(*args, **kwargs):
    # 检查是否需要计算梯度
    requires_grad = kwargs.get("requires_grad", False)
    # 复制 kwargs 并将 requires_grad 设为 False，以便在生成随机张量时排除计算梯度
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad["requires_grad"] = False
    # 生成指定参数和关键字参数的随机张量，并将所有小于 0.1 的值设为 0.1
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)


class TestVmapBatchedGradientLegacy(Namespace.TestVmapBaseLegacy):
    def _vmap_test(self, *args, **kwargs):
        # 调用基类方法 _vmap_test
        return _vmap_test(self, *args, **kwargs)

    # 测试 op(*args, **kwargs) 的批处理梯度计算，通过与顺序映射+堆叠的回退比较
    #
    # output_process_fn: 将输出映射到应该被求导部分的函数
    # batch_size: 批处理梯度的批量维度大小
    def _batched_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        if kwargs is None:
            kwargs = {}
        # 调用 op(*args, **kwargs) 获取输出
        outputs = op(*args, **kwargs)
        # 选择输出中需要进行梯度计算的部分并映射处理
        outputs = differentiable(output_process_fn(outputs))
        # 对输出中的每个元素，构造一个具有指定批处理大小的张量
        batched_vectors = tuple(construct_v(out, batch_size) for out in outputs)

        # 定义向量雅可比积
        def vector_jacobian_product(*vectors):
            return torch.autograd.grad(
                outputs, differentiable(args), vectors, retain_graph=True
            )

        # 执行批处理梯度计算测试
        self._vmap_test(
            vector_jacobian_product, batched_vectors, check_propagates_grad=False
        )

    # 测试 op(*args, **kwargs) 的批处理二阶梯度计算，通过与顺序映射+堆叠的回退比较
    #
    # output_process_fn: 将输出映射到应该被求导部分的函数
    # batch_size: 批处理梯度的批量维度大小
    #
    # 注意: 我们只测试在第二阶梯度计算中计算批处理梯度。
    # 第一个使用这种方法的具体用例是计算标量值函数的黑塞矩阵；这在贝叶斯逻辑回归中很有用。
    # 未来可能会有一个测试，先计算批处理一阶梯度，然后再用其计算批处理二阶梯度。
    def _batched_grad_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
    ):
        # 如果 kwargs 为 None，则设置为空字典
        if kwargs is None:
            kwargs = {}
        # 调用 op 函数，传入参数 args 和 kwargs，并接收返回的输出
        outputs = op(*args, **kwargs)
        # 对输出进行不同iable化处理，然后调用 output_process_fn 处理
        outputs = differentiable(output_process_fn(outputs))
        # 创建与 outputs 相同结构的全为 1 的张量
        ones = tuple(torch.ones_like(out) for out in outputs)
        # 使用 torch.autograd.grad 计算 outputs 对 args 的梯度
        # create_graph=True 表示创建用于二阶导数的计算图
        first_grads = torch.autograd.grad(
            outputs, differentiable(args), ones, create_graph=True
        )
        # 对 first_grads 进行不同iable化处理
        first_grads = differentiable(first_grads)
        # 断言 first_grads 的长度不为 0，即至少有一个梯度与输入相关
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!"
        )

        # 生成批量向量，每个向量通过 construct_v 函数从 first_grads 中构造而来
        batched_vectors = tuple(construct_v(grad, batch_size) for grad in first_grads)

        # 定义 vector_hessian_product 函数，用于计算向量的 Hessian 乘积
        def vector_hessian_product(*vectors):
            # 使用 torch.autograd.grad 计算 first_grads 对 args 的向量 v 的 Hessian 乘积
            # retain_graph=True 保留计算图以便多次反向传播
            # allow_unused=True 允许部分梯度为 None
            outputs = torch.autograd.grad(
                first_grads,
                differentiable(args),
                vectors,
                retain_graph=True,
                allow_unused=True,
            )
            # 过滤掉为 None 的输出
            outputs = tuple(out for out in outputs if out is not None)
            # 断言 outputs 的长度大于 0
            assert len(outputs) > 0
            return outputs

        # 调用 self._vmap_test 函数，测试 vector_hessian_product 的批量向量 batched_vectors
        self._vmap_test(
            vector_hessian_product, batched_vectors, check_propagates_grad=False
        )
    # 测试索引操作函数，使用指定设备生成随机张量 x
    def test_index(self, device):
        # 生成一个形状为 (2, 3) 的随机张量 x，并标记为需要梯度计算，使用指定设备
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 创建一个索引张量，用于索引 x 的元素
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        # 定义一个操作函数 op，对输入张量 x 进行平方操作，并根据索引返回指定元素
        def op(x):
            # 对输入张量 x 每个元素进行平方操作
            y = x * x
            # 使用索引 index 从 y 中获取指定位置的元素
            return y[index]

        # 调用自定义的梯度测试函数，测试 op 函数的梯度计算，传入参数 x
        self._batched_grad_test(op, (x,))
        # 调用自定义的二阶梯度测试函数，测试 op 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(op, (x,))

    # 测试 lgamma 函数的梯度计算，使用指定设备生成随机张量 x
    def test_lgamma(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 调用自定义的梯度测试函数，测试 Tensor.lgamma 函数的梯度计算，传入参数 x
        self._batched_grad_test(Tensor.lgamma, (x,))
        # 调用自定义的二阶梯度测试函数，测试 Tensor.lgamma 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    # 测试 log 函数的梯度计算，使用指定设备生成随机张量 x
    def test_log(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        # 调用自定义的梯度测试函数，测试 torch.log 函数的梯度计算，传入参数 x
        self._batched_grad_test(torch.log, (x,))
        # 调用自定义的二阶梯度测试函数，测试 torch.log 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(torch.log, (x,))

    # 测试 logsumexp 函数的梯度计算，使用指定设备生成随机张量 x
    def test_logsumexp(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        # 定义一个操作函数 op，对输入张量 x 沿指定维度计算 logsumexp
        def op(x):
            return torch.logsumexp(x, -1)

        # 调用自定义的梯度测试函数，测试 op 函数的梯度计算，传入参数 x
        self._batched_grad_test(op, (x,))
        # 调用自定义的二阶梯度测试函数，测试 op 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(op, (x,))

    # 测试 log1p 函数的梯度计算，使用指定设备生成随机张量 x
    def test_log1p(self, device):
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        # 调用自定义的梯度测试函数，测试 torch.log1p 函数的梯度计算，传入参数 x
        self._batched_grad_test(torch.log1p, (x,))
        # 调用自定义的二阶梯度测试函数，测试 torch.log1p 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(torch.log1p, (x,))

    # 使用 torch.max 函数测试最大值计算，生成随机张量 x，需要梯度计算，使用指定设备
    @allowVmapFallbackUsage
    def test_max(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 调用自定义的梯度测试函数，测试 torch.max 函数的梯度计算，传入参数 x
        self._batched_grad_test(torch.max, (x,))

    # 使用 torch.median 函数测试中位数计算，生成随机张量 x，需要梯度计算，使用指定设备
    @allowVmapFallbackUsage
    def test_median(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 调用自定义的梯度测试函数，测试 torch.median 函数的梯度计算，传入参数 x
        self._batched_grad_test(torch.median, (x,))

    # 使用 torch.min 函数测试最小值计算，生成随机张量 x，需要梯度计算，使用指定设备
    @allowVmapFallbackUsage
    def test_min(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 调用自定义的梯度测试函数，测试 torch.min 函数的梯度计算，传入参数 x
        self._batched_grad_test(torch.min, (x,))

    # 测试 permute 函数对张量进行维度置换操作，生成随机张量 x，需要梯度计算，使用指定设备
    def test_permute(self, device):
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        # 定义一个操作函数 op，对输入张量 x 进行维度置换，将原始维度 2, 0, 1 重新排列
        def op(x):
            return x.permute(2, 0, 1)

        # 调用自定义的梯度测试函数，测试 op 函数的梯度计算，传入参数 x
        self._batched_grad_test(op, (x,))

    # 测试 reshape 函数对张量进行形状重塑操作，生成随机张量 x，需要梯度计算，使用指定设备
    def test_reshape(self, device):
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        # 定义一个操作函数 op，对输入张量 x 进行形状重塑，将其变形为 (6, 5)
        def op(x):
            return x.reshape([2 * 3, 5])

        # 调用自定义的梯度测试函数，测试 op 函数的梯度计算，传入参数 x
        self._batched_grad_test(op, (x,))

    # 测试 sigmoid 函数的梯度计算，生成随机张量 x，需要梯度计算，使用指定设备
    def test_sigmoid(self, device):
        x = torch.randn(2, 3, requires_grad=True, device=device)
        # 调用自定义的梯度测试函数，测试 Tensor.sigmoid 函数的梯度计算，传入参数 x
        self._batched_grad_test(Tensor.sigmoid, (x,))
        # 调用自定义的二阶梯度测试函数，测试 Tensor.sigmoid 函数的二阶梯度计算，传入参数 x
        self._batched_grad_grad_test(Tensor.sigmoid, (x,))

    # 使用 torch.stack 函数测试张量堆叠操作，生成随机张量 x 和 y，需要梯度计算，使用指定设备
    def test_stack(self, device):
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        # 定义一个操作函数 op，对输入张量 x 和 y 进行堆叠
        def op(x, y):
            return torch.stack([x, y])

        # 调用自定义的梯度测试函数，测试 op 函数的梯度计算，传入参数 x 和 y
        self._batched
    # 定义一个测试函数，用于测试在给定设备上的张量切片操作的梯度计算
    def test_slice(self, device):
        # 创建一个形状为 (2, 3, 5) 的随机张量 x，并将其放置在指定设备上，同时需要计算梯度
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        # 调用自定义函数 _batched_grad_test，测试对 x 进行 x[0:1] 切片操作的梯度计算
        self._batched_grad_test(lambda x: x[0:1], (x,))
        # 调用自定义函数 _batched_grad_test，测试对 x 进行 x[:, 1:3] 切片操作的梯度计算
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        # 调用自定义函数 _batched_grad_test，测试对 x 进行 x[..., 1:3] 切片操作的梯度计算
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    # 定义一个测试函数，用于测试在给定设备上的张量迹(trace)操作的梯度计算
    def test_trace(self, device):
        # 创建一个形状为 (2, 3) 的随机张量 x，并将其放置在指定设备上，同时需要计算梯度
        x = torch.randn(2, 3, device=device, requires_grad=True)
        # 调用自定义函数 _batched_grad_test，测试对 x 进行迹操作的梯度计算
        self._batched_grad_test(Tensor.trace, (x,))

    # 定义一个测试函数，用于测试在给定设备上的阈值化操作的梯度计算
    def test_threshold(self, device):
        # 创建一个形状为 (2, 3) 的随机张量 x，并将其放置在指定设备上，同时需要计算梯度
        x = torch.randn(2, 3, device=device, requires_grad=True)
        # 调用自定义函数 _batched_grad_test，测试对 x 进行 F.threshold 操作的梯度计算
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    # 标记允许在 vmap 回退时使用的装饰器，定义一个测试函数，测试在视图上进行就地操作的梯度计算
    @allowVmapFallbackUsage
    def test_inplace_on_view(self, device):
        # 创建一个形状为 (4, 5) 的随机张量 leaf，并标记需要计算梯度
        leaf = torch.randn(4, 5, requires_grad=True)

        def func(leaf):
            # 确保函数非平凡地两次可微
            base = leaf * leaf
            # 获取 base 的视图，选择第一个元素
            view = base[0]
            # 对视图应用就地操作 view.cos_()
            view.cos_()
            return view

        # 调用自定义函数 _batched_grad_test，测试在 leaf 视图上执行 func 函数的梯度计算
        self._batched_grad_test(func, (leaf,), {})
        # 调用自定义函数 _batched_grad_grad_test，测试在 leaf 视图上执行 func 函数的梯度计算的梯度计算
        self._batched_grad_grad_test(func, (leaf,), {})

    # 标记允许在 vmap 回退时使用的装饰器，定义一个测试函数，测试在多个视图上进行就地操作的梯度计算
    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        # 创建一个形状为 (4, 4, 5) 的随机张量 leaf，并标记需要计算梯度
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        def func(leaf):
            # 确保函数非平凡地两次可微
            base = leaf * leaf
            # 将 base 转置，获取转置后的第一个视图
            view = base.transpose(0, 2)[1]
            # 获取视图的对角线元素
            view = view.diagonal()
            # 获取对角线元素的间隔为 2 的切片
            view = view[::2]
            # 对切片应用就地操作 view.cos_()
            view.cos_()
            return view

        # 调用自定义函数 _batched_grad_test，测试在 leaf 多个视图上执行 func 函数的梯度计算
        self._batched_grad_test(func, (leaf,), {})
        # 调用自定义函数 _batched_grad_grad_test，测试在 leaf 多个视图上执行 func 函数的梯度计算的梯度计算
        self._batched_grad_grad_test(func, (leaf,), {})

    # 定义一个测试函数，用于测试在给定设备上的张量对角线操作的梯度计算
    def test_diagonal(self, device):
        # 创建一个形状为 (4, 5) 的随机张量 x，并将其放置在指定设备上，同时需要计算梯度
        x = torch.randn(4, 5, device=device, requires_grad=True)
        # 调用自定义函数 _batched_grad_test，测试对 x 进行对角线操作的梯度计算
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))

        # 创建一个形状为 (3, 4, 5) 的随机张量 x，并将其放置在指定设备上，同时需要计算梯度
        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        # 调用自定义函数 _batched_grad_test，测试对 x 进行对角线操作的梯度计算
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    # 标记允许在 vmap 回退时使用的装饰器，定义一个测试函数，测试无关输出的梯度计算
    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        B0 = 3
        # 创建一个形状为空的随机张量 x，并标记需要计算梯度
        x = torch.randn([], requires_grad=True)
        # 创建一个形状为空的随机张量 y，并标记需要计算梯度
        y = torch.randn([], requires_grad=True)
        # 创建一个形状为 B0 的随机张量 gy，并标记需要计算梯度
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # 计算 y 对 x 的梯度，并返回结果
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        # 使用 vmap 对 vjp 函数进行矢量化，计算 gy 的梯度
        result = vmap(vjp)(gy)
        # 断言结果与预期的形状相同且全为零张量
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    # 标记允许在 vmap 回退时使用的装饰器，定义一个测试函数，测试多次无关输出的梯度计算
    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        B0 = 3
        # 创建一个形状为空的随机张量 x，并标记需要计算梯度
        x = torch.randn([], requires_grad=True)
        # 创建一个形状为空的随机张量 y，并标记需要计算梯度
        y = torch.randn([], requires_grad=True)
        # 创建一个形状为 B0 的随机张量 gy，并标记需要计算梯度
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # 计算 y 对 x 的梯度，并返回结果
# 调用函数 instantiate_device_type_tests，初始化设备类型测试
instantiate_device_type_tests(
    TestVmapBatchedGradientLegacy,  # 使用 TestVmapBatchedGradientLegacy 类来进行测试
    globals(),  # 使用当前全局命名空间
    None,  # 不传递额外的参数
)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```