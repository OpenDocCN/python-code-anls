# `.\pytorch\test\test_reductions.py`

```
# Owner(s): ["module: tests"]

import contextlib  # 引入上下文管理模块
import torch  # 引入PyTorch库
import numpy as np  # 引入NumPy库

import math  # 引入数学函数库
from typing import Dict, List, Sequence  # 引入类型注解相关的模块
import random  # 引入随机数生成模块
from functools import partial  # 引入函数工具模块中的partial函数
from itertools import product, combinations, permutations  # 引入迭代器工具模块中的函数
import warnings  # 引入警告模块

from torch import inf, nan  # 从torch模块中引入inf和nan常量
from torch.testing import make_tensor  # 从torch.testing模块中引入make_tensor函数
from torch.testing._internal.common_dtype import (  # 从内部common_dtype模块引入多个数据类型相关函数
    all_types_and_complex_and, get_all_math_dtypes, integral_types, complex_types, floating_types_and,
    integral_types_and, floating_and_complex_types_and, all_types_and, all_types,
)
from torch.testing._internal.common_utils import (  # 从内部common_utils模块引入多个测试相关的工具函数
    TestCase, run_tests, skipIfNoSciPy, slowTest, torch_to_numpy_dtype_dict,
    parametrize,
    IS_WINDOWS
)
from torch.testing._internal.common_device_type import (  # 从内部common_device_type模块引入多个设备类型相关函数
    OpDTypes, expectedFailureMeta, instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA, dtypesIfCPU,
    onlyNativeDeviceTypes, onlyCUDA, largeTensorTest, ops, precisionOverride
)
from torch.testing._internal.common_methods_invocations import (  # 从内部common_methods_invocations模块引入多个方法调用相关函数
    ReductionOpInfo, ReductionPythonRefInfo, reduction_ops, reference_masked_ops
)


# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    """生成指定形状、数据类型、设备的输入张量

    Args:
    - shape: 张量的形状
    - dtype: 张量的数据类型
    - device: 张量的设备 ('cpu' 或 'cuda')
    - with_extremal: 是否包含极端值

    Returns:
    - x: 生成的输入张量
    """
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)  # 创建空张量
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # 为避免torch.randn对bfloat16未实现问题，特别处理
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0  # 部分元素置为0
            if with_extremal and dtype.is_floating_point:
                # 使用极端值
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True  # 随机设置部分元素为True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)  # 生成随机整数张量

    return x


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    """生成随机形状的元组

    Args:
    - dim: 元组维度
    - min_size: 每维最小尺寸
    - max_size: 每维最大尺寸

    Returns:
    - shape: 生成的随机形状元组
    """
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)


def _reduced_shape(shape, dim=None, keepdim=False):
    """根据给定的dim和keepdim计算预期的减少后的形状

    Args:
    - shape: 原始形状
    - dim: 减少的维度
    - keepdim: 是否保持维度

    Returns:
    - tuple: 计算得到的减少后的形状元组
    """
    """Computes the expected reduced shape given dim and keepdim
    """
    # 如果未指定要减少的维度（dim），根据 keepdim 返回相应的形状
    if dim is None:
        return [1] * len(shape) if keepdim else []

    # 将负数维度转换为正数维度的集合
    dim = dim if isinstance(dim, Sequence) else [dim]
    dim = {i if i >= 0 else len(shape) + i for i in dim}

    result = []
    # 遍历输入形状的每个维度
    for i, size in enumerate(shape):
        # 如果当前维度不在要减少的维度集合中，则保留其大小
        if i not in dim:
            result.append(size)
        # 否则，如果 keepdim 为真，则将当前维度大小设置为1
        elif keepdim:
            result.append(1)

    # 返回减少维度后的形状
    return result
class TestReductions(TestCase):

    ###########################################################################
    # ReductionOpInfo unit tests
    ###########################################################################

    def _test_dim_keepdim(self, op: ReductionOpInfo, device, *, ndim, **dim_keepdim):
        """Tests output shape for input with ndim and dim and keepdim kwargs"""
        # 生成随机形状，并创建指定设备上的张量
        shape = torch.randint(2, 5, (ndim,)).tolist()
        t = make_tensor(shape, dtype=torch.float, device=device)
        # 生成参数和关键字参数
        args, kwargs = next(op.generate_args_kwargs(t, **dim_keepdim))
        # 调用操作对象，进行降维操作
        result = op(t, *args, **dim_keepdim, **kwargs)
        # 预期的输出形状
        expected_shape = _reduced_shape(shape, **dim_keepdim)
        # 断言输出形状是否符合预期
        self.assertEqual(result.shape, expected_shape, f"""
        expected output shape to be {expected_shape} but got {list(result.shape)}
        for input shape {shape} and {dim_keepdim}
        """)

    # TODO(@heitorschueroff) combine cases with and without keepdim once
    # there's support for a @parametrize decorator.

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default(self, device, op: ReductionOpInfo):
        """Tests that the default dim reduces all dimensions."""
        # 测试默认情况下对所有维度进行降维操作
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default_keepdim(self, device, op: ReductionOpInfo):
        """Tests that the default dim, when keepdim=True, reduces all dimensions to size 1."""
        # 测试当 keepdim=True 时，默认情况下对所有维度进行降维操作至大小为 1
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, keepdim=True)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none(self, device, op: ReductionOpInfo):
        """Tests that dim=None reduces all dimensions."""
        # 测试当 dim=None 时，对所有维度进行降维操作
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=None, when keepdim=True, reduces all dimensions to size 1."""
        # 测试当 dim=None 且 keepdim=True 时，对所有维度进行降维操作至大小为 1
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None, keepdim=True)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_single(self, device, op: ReductionOpInfo):
        """Tests that dim=i reduces dimension i."""
        # 测试指定维度 i 进行降维操作
        self._test_dim_keepdim(op, device, ndim=0, dim=0)
        self._test_dim_keepdim(op, device, ndim=1, dim=0)
        self._test_dim_keepdim(op, device, ndim=2, dim=-1)
        self._test_dim_keepdim(op, device, ndim=3, dim=1)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    # 此处代码未完整，应继续补充注释
    # 定义测试方法，测试当指定的维度参数为单个值且 keepdim=True 时的情况
    def test_dim_single_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=i, when keepdim=True, reduces dimension i to size 1."""
        # 调用测试辅助方法，验证维度为0时的情况
        self._test_dim_keepdim(op, device, ndim=0, dim=0, keepdim=True)
        # 调用测试辅助方法，验证维度为1时的情况
        self._test_dim_keepdim(op, device, ndim=1, dim=0, keepdim=True)
        # 调用测试辅助方法，验证维度为2时的情况
        self._test_dim_keepdim(op, device, ndim=2, dim=-1, keepdim=True)
        # 调用测试辅助方法，验证维度为3时的情况
        self._test_dim_keepdim(op, device, ndim=3, dim=1, keepdim=True)

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为空列表时的情况
    def test_dim_empty(self, device, op: ReductionOpInfo):
        """Tests that dim=[] is a no-op"""
        # 调用测试辅助方法，验证维度为0时的情况
        self._test_dim_keepdim(op, device, ndim=0, dim=[])
        # 调用测试辅助方法，验证维度为2时的情况
        self._test_dim_keepdim(op, device, ndim=2, dim=[])

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为空列表且 keepdim=True 时的情况
    def test_dim_empty_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=[], when keepdim=True, is a no-op"""
        # 调用测试辅助方法，验证维度为0时的情况
        self._test_dim_keepdim(op, device, ndim=0, dim=[], keepdim=True)
        # 调用测试辅助方法，验证维度为2时的情况
        self._test_dim_keepdim(op, device, ndim=2, dim=[], keepdim=True)

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为多个值时的情况
    def test_dim_multi(self, device, op: ReductionOpInfo):
        """Tests that dim=[i, j, ...] reduces dimensions i, j, ...."""
        # 调用测试辅助方法，验证维度为1时的情况
        self._test_dim_keepdim(op, device, ndim=1, dim=[0])
        # 调用测试辅助方法，验证维度为3时的情况
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为多个值且 keepdim=True 时的情况
    def test_dim_multi_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=[i, j, ...], when keepdim=True, reduces dimensions i, j, .... to size 1."""
        # 调用测试辅助方法，验证维度为1时的情况
        self._test_dim_keepdim(op, device, ndim=1, dim=[0], keepdim=True)
        # 调用测试辅助方法，验证维度为3时的情况
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2], keepdim=True)

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为未排序的值时的情况
    def test_dim_multi_unsorted(self, device, op: ReductionOpInfo):
        """Tests that operator correctly handles unsorted dim list."""
        # 调用测试辅助方法，验证维度为4时的情况
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2])

    # 使用 ops 装饰器注册测试方法，筛选支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数为未排序的值且 keepdim=True 时的情况
    def test_dim_multi_unsorted_keepdim(self, device, op: ReductionOpInfo):
        """Tests that operator correctly handles unsorted dim list when keepdim=True."""
        # 调用测试辅助方法，验证维度为4时的情况
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2], keepdim=True)

    # 使用 ops 装饰器注册测试方法，筛选不支持多个维度操作的 reduction_ops，不指定数据类型
    @ops(filter(lambda op: not op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    # 定义测试方法，测试当指定的维度参数包含重复条目时抛出 RuntimeError 的情况
    def test_dim_multi_duplicate(self, device, op: ReductionOpInfo):
        """Tests that an error is raised if dim has duplicate entries."""
        # 使用断言验证调用测试辅助方法时抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 1, 1, 2])
    # 测试不支持多维度的操作是否会引发 TypeError 异常
    def test_dim_multi_unsupported(self, device, op: ReductionOpInfo):
        """Tests that ops claiming to not support multi dim actually don't."""
        # 使用 assertRaises 上下文管理器检查是否引发 TypeError 异常
        with self.assertRaises(TypeError):
            # 调用内部方法 _test_dim_keepdim 进行测试
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    # 对超出边界的维度进行测试，预期会引发 IndexError 异常
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_offbounds(self, device, op: ReductionOpInfo):
        """Tests that passing an off-bounds dim throws"""
        # 使用 assertRaises 上下文管理器检查是否引发 IndexError 异常
        with self.assertRaises(IndexError):
            # 调用内部方法 _test_dim_keepdim 进行测试
            self._test_dim_keepdim(op, device, ndim=2, dim=2)

    # 测试在某些特定维度上，当张量具有超过 64 维时是否会引发异常
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_ndim_limit(self, device, op: ReductionOpInfo):
        """Tests that an exception is raised when reducing a tensor with more
        than 64 dims along some specific dimensions. dim=None is ok"""
        # 创建一个具有 65 维的张量
        t = make_tensor([1] * 65, dtype=torch.float, device=device)
        # 使用 assertRaisesRegex 检查是否引发 RuntimeError 异常，并验证异常信息
        with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
            # 调用 op 函数进行测试，指定维度为 0
            op(t, dim=0)

    # 测试标识值是否对运算符具有恒等性质
    @ops(filter(lambda op: op.identity is not None, reduction_ops), dtypes=OpDTypes.supported)
    def test_identity(self, device, dtype, op: ReductionOpInfo):
        """Tests that the identity value is an identity for the operator"""
        # 创建长度为 10 的张量
        t = make_tensor((10,), dtype=dtype, device=device)
        # 将张量 t 的奇数索引位置设置为运算符 op 的标识值
        t[1::2] = op.identity
        # 生成运算符 op 的参数和关键字参数
        args, kwargs = next(op.generate_args_kwargs(t))
        # 对 t 的偶数索引位置应用运算符 op，得到结果 result
        result = op(t[::2], *args, **kwargs)
        # 对整个张量 t 应用运算符 op，得到结果 result_with_identity
        result_with_identity = op(t, *args, **kwargs)
        # 使用 self.assertEqual 验证结果和带有标识值的结果是否相等
        self.assertEqual(result, result_with_identity, """
        Adding identity value to the input tensor should not change the result.
        """)

    # TODO(@heitorschueroff) Update these to use the nan_policy kwarg once
    # it is added to reduction operators.

    # 测试 nan_policy 参数为 'propagate' 时，NaN 值是否被正确传播到输出
    @ops(filter(lambda op: op.nan_policy == 'propagate', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_propagate(self, device, dtype, op: ReductionOpInfo):
        """Tests that nan is propagated to the output by default"""
        # 创建长度为 5 的张量，其中第 2 个元素设为 NaN
        t = make_tensor((5,), dtype=dtype, device=device)
        t[2] = torch.nan
        # 生成运算符 op 的参数和关键字参数
        args, kwargs = next(op.generate_args_kwargs(t))
        # 对张量 t 应用运算符 op，得到结果 result
        result = op(t, *args, **kwargs)
        # 使用 self.assertTrue 验证结果是否包含 NaN
        self.assertTrue(result.isnan())

    # 测试 nan_policy 参数为 'omit' 时，NaN 值是否不影响结果
    @ops(filter(lambda op: op.nan_policy == 'omit', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_omit(self, device, dtype, op: ReductionOpInfo):
        """Tests that NaN values do not affect the result."""
        # 创建长度为 10 的张量，其中奇数索引位置设为 NaN
        t = make_tensor((10,), dtype=dtype, device=device)
        t[1::2] = torch.nan
        # 生成运算符 op 的参数和关键字参数
        args, kwargs = next(op.generate_args_kwargs(t))
        # 对 t 的偶数索引位置应用运算符 op，得到结果 result
        result = op(t[::2], *args, **kwargs)
        # 对整个张量 t 应用运算符 op，得到结果 result_with_nan
        result_with_nan = op(t, *args, **kwargs)
        # 使用 self.assertEqual 验证结果和带有 NaN 的结果是否相等
        self.assertEqual(result, result_with_nan)

    # 使用所有支持的数据类型进行测试的装饰器
    @ops(reduction_ops, dtypes=OpDTypes.supported)
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_result_dtype(self, device, dtype, op: ReductionOpInfo):
        """Tests that the result has the correct dtype"""
        # 创建一个形状为 (5,) 的张量 t，指定设备和数据类型
        t = make_tensor((5,), dtype=dtype, device=device)
        # 生成操作所需的参数和关键字参数
        args, kwargs = next(op.generate_args_kwargs(t))
        # 使用操作 op 对张量 t 进行操作，得到结果 result
        result: torch.Tensor = op(t, *args, **kwargs)
        # 判断数据类型 dtype 是否为整数或布尔类型
        is_integral = dtype in integral_types_and(torch.bool)
        # 如果操作将整数类型提升为浮点数类型，并且 dtype 是整数类型
        if op.promotes_int_to_float and is_integral:
            # 断言 result 的数据类型为浮点数
            self.assertTrue(torch.is_floating_point(result))
        # 如果操作将整数类型提升为 int64 类型，并且 dtype 是整数类型
        elif op.promotes_int_to_int64 and is_integral:
            # 断言 result 的数据类型为 int64
            self.assertEqual(result.dtype, torch.int64)
        # 如果操作有指定的结果数据类型
        elif op.result_dtype is not None:
            # 断言 result 的数据类型与操作指定的结果数据类型一致
            self.assertEqual(result.dtype, op.result_dtype)
        # 如果操作将复数类型转换为实数类型
        elif op.complex_to_real:
            # 定义一个复数到实数数据类型的映射表
            _complex_to_real_dtype_map = {
                torch.complex128: torch.float64,
                torch.complex64: torch.float32,
                torch.complex32: torch.float16,
            }
            # 断言 result 的数据类型与映射表中对应 dtype 的实数类型一致
            self.assertEqual(result.dtype, _complex_to_real_dtype_map.get(dtype, dtype))
        else:
            # 断言 result 的数据类型与输入的数据类型 dtype 一致
            self.assertEqual(result.dtype, dtype)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_empty_tensor_empty_slice(self, device, op: ReductionOpInfo):
        """Tests for consistent behavior when reducing over an empty slice.

        The rules for reducing over an empty slice are as follows:
            - Return the identity value if the operator has one
            - Otherwise, return NaN if the operator promotes integral dtype to
              floating point dtypes.
            - Otherwise, raise an error

        See discussion here https://github.com/pytorch/pytorch/issues/61901
        """
        # 创建一个形状为 (0, 2, 3) 的张量 t，数据类型为 torch.float，指定设备
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        # 遍历可能的维度 [0] 或者 [0, 2]（如果操作支持多维度）
        for dim in [0] + [[0, 2]] if op.supports_multiple_dims else []:
            # 生成操作所需的参数和关键字参数，指定维度 dim
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            # 如果操作有单位元素值
            if op.identity is not None:
                # 对空切片进行缩减操作应返回单位元素值
                result = op(t, *args, dim=dim, **kwargs)
                # 断言 result 与填充为单位元素值的张量相等
                self.assertEqual(result, torch.full_like(result, op.identity))
            # 如果操作将整数类型提升为浮点数类型
            elif op.promotes_int_to_float:
                # 对空切片进行缩减操作应返回 NaN
                result = op(t, *args, dim=dim, **kwargs)
                # 断言 result 与填充为 NaN 的张量相等
                self.assertEqual(result, torch.full_like(result, torch.nan))
            else:
                # 对空切片进行缩减操作应引发错误
                if isinstance(op, ReductionPythonRefInfo):
                    # 对于参考实现的缩减操作，应抛出 RuntimeError
                    with self.assertRaises(RuntimeError):
                        op(t, *args, dim=dim, **kwargs)
                else:
                    # 否则应抛出 IndexError
                    with self.assertRaises(IndexError):
                        op(t, *args, dim=dim, **kwargs)
    def test_empty_tensor_nonempty_slice(self, device, op: ReductionOpInfo):
        """Tests that reducing a nonempty slice of an empty tensor returns an
        empty tensor with the dimensions reduced."""
        # 创建一个形状为 (0, 2, 3) 的空张量，数据类型为 float，在指定设备上创建
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        # 对于每个支持多维度操作的操作，测试在指定维度上生成参数和关键字参数
        for dim in [1] + [[1, 2]] if op.supports_multiple_dims else []:
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            # 使用指定的操作对张量 t 进行维度为 dim 的减少操作
            result = op(t, *args, dim=dim, **kwargs)
            # 断言结果张量的形状与减少维度后的预期形状相等
            self.assertEqual(result.shape, _reduced_shape(t.shape, dim))

    def _test_noncontiguous(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        """Helper method to test noncontiguous input tensors."""
        # 断言输入张量 t 不是连续的
        assert not t.is_contiguous()

        # 将输入张量 t 转为连续的张量
        t_contig = t.contiguous()
        # 对于每个生成的参数和关键字参数对，使用指定的操作对 t 和 t_contig 进行减少操作
        for args, kwargs in op.generate_args_kwargs(t_contig, **reduction_kwargs):
            kwargs.update(reduction_kwargs)
            # 使用指定的操作对非连续张量 t 进行减少操作
            result = op(t, *args, **kwargs)
            # 使用指定的操作对连续张量 t_contig 进行减少操作，得到预期结果
            expected = op(t_contig, *args, **kwargs)
            # 断言非连续张量 t 和连续张量 t_contig 的减少结果相等
            self.assertEqual(result, expected)

    @ops(reduction_ops)
    def test_noncontiguous_innermost(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing along noncontiguous innermost dimension."""
        # 创建形状为 (10, 10) 的张量 t，数据类型和设备由参数指定，在指定维度上生成减少操作
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        # 使用 _test_noncontiguous 方法测试在内部非连续维度上的减少操作
        self._test_noncontiguous(op, t[:, ::2], dim=1)

    @ops(reduction_ops)
    def test_noncontiguous_outermost(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing along noncontiguous outermost dimension."""
        # 创建形状为 (10, 10) 的张量 t，数据类型和设备由参数指定，在指定维度上生成减少操作
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        # 使用 _test_noncontiguous 方法测试在外部非连续维度上的减少操作
        self._test_noncontiguous(op, t[::2, :], dim=0)

    @ops(reduction_ops)
    def test_noncontiguous_all(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing all dimensions of a noncontiguous tensor."""
        # 创建形状为 (5, 5, 5) 的张量 t，数据类型和设备由参数指定，在指定维度上生成减少操作
        t = make_tensor((5, 5, 5), dtype=dtype, device=device, low=-1, high=1)
        # 使用 _test_noncontiguous 方法测试对所有维度非连续张量的减少操作
        self._test_noncontiguous(op, t[::2, ::3, 1:-1:2])

    @ops(reduction_ops)
    def test_noncontiguous_transposed(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing a transposed tensor."""
        # 创建形状为 (5, 5) 的张量 t，数据类型和设备由参数指定，在转置后的张量上生成减少操作
        t = make_tensor((5, 5), dtype=dtype, device=device, low=-1, high=1)
        # 使用 _test_noncontiguous 方法测试对转置张量的减少操作
        self._test_noncontiguous(op, t.T)

    @ops(reduction_ops)
    def test_noncontiguous_expanded(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing a tensor with expanded singleton dimensions."""
        # 创建形状为 (2, 3) 的张量 t，数据类型和设备由参数指定，将其在指定维度上扩展后生成减少操作
        t = make_tensor((2, 3), dtype=dtype, device=device, low=-1, high=1)
        # 使用 _test_noncontiguous 方法测试对扩展单例维度张量的减少操作
        self._test_noncontiguous(op, t.unsqueeze(1).expand(-1, 5, -1))
    # 定义测试函数，用于比较操作符 op 对于给定输入和缩减参数的效果
    def _test_ref(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        """Compares op against op.ref for the given input and reduction kwargs"""
        # 生成 op 的参数和关键字参数的组合
        for args, kwargs in op.generate_args_kwargs(t, **reduction_kwargs):
            # 更新 kwargs 以包含所有的缩减参数
            kwargs.update(reduction_kwargs)
            # 执行操作 op，并获取结果
            result = op(t, *args, **kwargs)
            # 计算使用 op.ref 得到的期望结果
            expected = op.ref(t.detach().cpu().numpy(), *args, **kwargs)
            # 使用 self.assertEqual 检查结果和期望结果是否相等（dtype 可不完全相同）
            self.assertEqual(result, expected, exact_dtype=False)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    # 测试标量输入情况下，比较操作 op 与参考实现的差异
    def test_ref_scalar_input(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for scalar input tensors"""
        # 使用 make_tensor 创建一个标量输入 tensor，并执行 _test_ref 进行比较
        self._test_ref(op, make_tensor([], dtype=dtype, device=device))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    # 测试小型输入 tensor 的情况下，比较操作 op 与参考实现的差异
    def test_ref_small_input(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for small input tensors"""
        # 创建一个小型输入 tensor t，并执行 _test_ref 进行比较
        t = make_tensor((5, 3, 4, 2), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        self._test_ref(op, t)
        # 对于支持多维度操作的 op，针对指定的维度再次执行 _test_ref 进行比较
        for dim in [0, 1, 3] + ([[0, 2], [1, 3]] if op.supports_multiple_dims else []):
            self._test_ref(op, t, dim=dim)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    # 测试大型 1D 输入 tensor 的情况下，比较操作 op 与参考实现的差异，用于稳定性检查
    def test_ref_large_input_1D(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a large 1D input tensor to check stability"""
        # 使用 make_tensor 创建一个大型 1D 输入 tensor，并执行 _test_ref 进行比较
        self._test_ref(op, make_tensor((2 ** 20,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    # 测试大型 2D 输入 tensor 的情况下，比较操作 op 与参考实现的差异，用于并行性检查
    def test_ref_large_input_2D(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a large 2D input tensor to test parallelism"""
        # 创建一个大型 2D 输入 tensor t，并执行 _test_ref 进行比较，指定维度为 1
        t = make_tensor((32, 2 ** 16), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True)
        self._test_ref(op, t, dim=1)

    @largeTensorTest("8gb")
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    # 测试需要使用 64 位索引的非常大输入 tensor 的情况下，比较操作 op 与参考实现的差异
    def test_ref_large_input_64bit_indexing(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a very large input tensor that requires 64 bit indexing"""
        # 使用 make_tensor 创建一个需要 64 位索引的非常大输入 tensor，并执行 _test_ref 进行比较
        self._test_ref(op, make_tensor((275000000,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    # 对于所有类型和复杂类型以及 torch.half 和 torch.bool 类型，比较操作 op 与参考实现的差异
    # 测试函数，用于测试具有重复值的输入张量上的操作
    def test_ref_duplicate_values(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for input tensors with duplicate values"""
        # 创建一个指定设备和数据类型的张量，形状为 (4, 4)，数值范围在 [-2, 2) 之间，且不包括零
        t = make_tensor((4, 4), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        # 将张量 t 的奇数行和奇数列的值设置为相同，制造重复值情况
        t[::2, ::2] = t[1::2, 1::2]
        # 使用 _test_ref 方法比较 op 和 t
        self._test_ref(op, t)
        # 在维度 0 上测试 op 和 t
        self._test_ref(op, t, dim=0)
        # 在维度 1 上测试 op 和 t
        self._test_ref(op, t, dim=1)

    # 使用 op 与参考值比较，对具有极值的输入张量进行测试
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float32, torch.complex64])
    def test_ref_extremal_values(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for input tensors with extremal values"""
        # 创建一个指定设备和数据类型的张量，形状为 (5,)，数值范围在设备支持的范围内，不包括零
        t = make_tensor((5,), dtype=dtype, device=device, exclude_zero=True)
        # 极值数组，包含 0, 1, nan, inf, -inf
        extremals = [0, 1, nan, inf, -inf]
        # 对每个极值进行测试
        for extremal in extremals:
            # 将张量 t 的第二个元素设置为当前的极值
            t[2] = extremal
            # 使用 _test_ref 方法比较 op 和 t
            self._test_ref(op, t)

    ###########################################################################
    # TODO: Legacy tests - port to ReductionOpInfo
    ###########################################################################

    # 测试函数，验证方差计算的无偏性
    def test_var_unbiased(self, device):
        tensor = torch.randn(100, device=device)
        # 验证在维度 0 上的方差计算是否无偏
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        # 验证在全局范围内的方差计算是否无偏
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        # 验证在指定 unbiased=False 时，方差计算是否正确
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False))

        tensor = torch.tensor([1.0, 2.0], device=device)
        # 验证一维张量的方差计算（无偏）
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        # 验证一维张量的方差计算（有偏）
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        # 验证一维张量的方差计算（无偏）
        self.assertEqual(tensor.var(unbiased=True), 1.0)
        # 验证一维张量的方差计算（有偏）
        self.assertEqual(tensor.var(unbiased=False), 2.0 / 3.0)

        tensor = torch.randn(100, device=device)
        # 验证在维度 0 上的标准差计算是否无偏
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        # 验证在全局范围内的标准差计算是否无偏
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        # 验证在指定 unbiased=False 时，标准差计算是否正确
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False))

    # 测试函数，验证方差计算的稳定性
    def test_var_stability(self, device):
        tensor = torch.tensor([2281.5, 2281.25], device=device)
        # 验证一维张量的方差计算结果
        self.assertEqual(tensor.var(dim=0), 0.03125)
        # 验证全局范围内的方差计算结果
        self.assertEqual(tensor.var(), 0.03125)

    # 测试函数，验证在 uint8 类型下的求和操作是否会溢出
    def test_sum_dim_reduction_uint8_overflow(self, device):
        example = [[-1, 2, 1], [5, 3, 6]]
        x = torch.tensor(example, dtype=torch.uint8, device=device)
        # 验证在不同维度上的 uint8 类型张量的求和结果是否正确
        self.assertEqual(x.sum(dtype=torch.uint8).item(), 16)
        self.assertEqual(x.sum(0, dtype=torch.uint8), torch.tensor([4, 5, 7], dtype=torch.uint8, device=device))
        self.assertEqual(x.sum(1, dtype=torch.uint8), torch.tensor([2, 14], dtype=torch.uint8, device=device))
        y = torch.tensor(example, dtype=torch.uint8, device=device)
        torch.sum(x, 0, out=y)
        # 验证在不同维度上的 uint8 类型张量的求和结果是否正确
        self.assertEqual(x.sum(0, dtype=torch.uint8), y)
    # 测试函数，用于测试在维度小于64时的异常情况
    def test_dim_reduction_less_than_64(self, device):
        # 创建一个长度为65的维度列表，表示一个形状为[1, 1, ..., 1]的张量，共65个维度
        sizes = [1] * 65
        # 生成一个随机张量 x，形状为 sizes，设备为指定的 device
        x = torch.randn(sizes, device=device)
        # 定义一组操作函数，包括 torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var, torch.norm
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.norm]
        # 遍历每个操作函数
        for op in ops:
            # 断言在执行 op(x, dim=64) 时会抛出 RuntimeError，并且错误信息包含 "only tensors with up to 64 dims are supported"
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=64)
            # 断言在执行 op(x, dim=-1) 时会抛出 RuntimeError，并且错误信息包含 "only tensors with up to 64 dims are supported"
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=-1)

    # 标记为只在 CPU 上运行的测试函数，测试在最后一个维度上的降维操作
    @onlyCPU
    # 标记函数支持的数据类型为 torch.float 和 torch.bfloat16
    @dtypes(torch.float, torch.bfloat16)
    def test_dim_reduction_lastdim(self, device, dtype):
        # 生成一个随机张量 x，形状为 [3, 5, 40]，设备为指定的 device，数据类型为指定的 dtype
        x = torch.randn(3, 5, 40, device=device, dtype=dtype)
        # 对 x 进行切片操作，保留前两个维度，将第三个维度从 0 到 40 中的偶数索引切片
        x = x[:, :, 0:40:2]
        # 对 x 进行连续化操作，确保内存布局连续
        x2 = x.contiguous()
        # 定义一组操作函数，包括 torch.norm, torch.argmax, torch.argmin
        ops = [torch.norm, torch.argmax, torch.argmin]
        # 遍历每个操作函数
        for op in ops:
            # 对 x 在最后一个维度上执行操作 op，得到结果 y
            y = op(x, dim=-1)
            # 对 x2 在最后一个维度上执行操作 op，得到结果 y2
            y2 = op(x2, dim=-1)
            # 断言 y 和 y2 的结果相等
            self.assertEqual(y, y2)

    # 标记为只在安装了 SciPy 的环境下运行的测试函数，测试 torch.Tensor 的 logsumexp 方法
    @skipIfNoSciPy
    def test_logsumexp(self, device):
        # 导入 scipy.special 中的 logsumexp 函数
        from scipy.special import logsumexp
        # 生成一个随机张量 a，形状为 [5, 4]，设备为指定的 device
        a = torch.randn(5, 4, device=device)
        # 设置 a 中的特定值为正无穷和负无穷，用于测试极端情况
        a[0, 0] = inf
        a[1, :] = -inf
        # 使用 torch.Tensor 的 logsumexp 方法计算张量 a 在第一维上的结果
        actual = a.logsumexp(1)
        # 使用 numpy 的 logsumexp 函数计算张量 a 在第一维上的期望结果
        expected = logsumexp(a.cpu().numpy(), 1)
        # 断言期望结果的形状与实际结果的形状相等
        self.assertEqual(expected.shape, actual.shape)
        # 断言期望结果与实际结果的值相等
        self.assertEqual(expected, actual)

        # 检查 out 参数是否真的是原地操作
        # 创建一个全零张量 b，形状为 [5, 2]，设备为指定的 device
        b = torch.zeros(5, 2, device=device)
        # 获取张量 b 的第一列 c
        c = b[:, 0]
        # 使用 torch.Tensor 的 logsumexp 方法将张量 a 在第一维上的结果写入 c
        torch.logsumexp(a, 1, out=c)
        # 断言期望结果与张量 b 的第一列的值相等
        self.assertEqual(expected, b[:, 0])

        # 检查整数输入是否提升为浮点数
        # 生成一个随机整数张量 e，形状为 [5, 4]，值在 [-100, 100) 范围内，设备为指定的 device
        e = torch.randint(-100, 100, [5, 4], device=device)
        # 使用 torch.Tensor 的 logsumexp 方法计算张量 e 在第一维上的结果，并将结果转换为 torch.float64 类型
        actual = e.logsumexp(1).to(torch.float64)
        # 使用 numpy 的 logsumexp 函数计算张量 e 在第一维上的期望结果
        expected = logsumexp(e.cpu().numpy(), 1)
        # 断言期望结果的形状与实际结果的形状相等
        self.assertEqual(expected.shape, actual.shape)
        # 断言期望结果与实际结果的值相等
        self.assertEqual(expected, actual)

    # 标记为只在安装了 SciPy 的环境下运行的测试函数，测试复数类型下的操作
    @skipIfNoSciPy
    @dtypes(torch.complex64, torch.complex128)
    @onlyCPU
    # 定义测试函数 test_sum_parallel，接受参数 self 和 device
    def test_sum_parallel(self, device):
        # 为了使用并行分支，我们需要比较相对较大的张量。
        # 即使在单核机器上运行，这些测试仍然会为您提供正确性的信号。

        # 定义内部函数 _run_test，接受参数 size
        def _run_test(size):
            # 遍历 size 的长度加一的范围
            for dim in range(len(size) + 1):
                # nv 是一个随机生成的 0 和 1 组成的数组，形状为 size
                nv = np.round(np.random.rand(*size))  # 0s and 1s
                # 将 nv 转换为 torch 张量 tv
                tv = torch.from_numpy(nv)
                # 如果 numel 大于 Parallel.h 中定义的 grainsize，才会使用并行计算
                self.assertTrue(tv.numel() > 32768)
                # 根据 dim 的值选择不同的维度对 nv 和 tv 进行求和
                if dim == len(size):
                    nvs = nv.sum()
                    tvs = tv.sum()
                else:
                    nvs = nv.sum(dim)
                    tvs = tv.sum(dim)
                # 计算 nv 和 tvs 之间的差的绝对值的总和
                diff = np.abs(nvs - tvs.numpy()).sum()
                # 断言差的绝对值总和为 0
                self.assertEqual(diff, 0)

        # 调用 _run_test 函数进行多个测试
        _run_test([2, 3, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3])
        _run_test([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        _run_test([1, 32 * 8 * 32 * 8])
        _run_test([1, 32770])

    # TODO: kill map2_ (and similar) uses and update to compare with NumPy
    # only works on CPU since this uses map2_, which is only supported on CPU
    # 定义 _testCSelection 函数，接受参数 self, torchfn, mathfn
    def _testCSelection(self, torchfn, mathfn):
        # 创建两个大小为 (100, 100) 的随机张量 a 和 b
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        # 使用 torchfn 函数对 a 和 b 进行操作，得到张量 c
        c = torchfn(a, b)
        # 创建一个与 a 大小相同的全零张量 expected_c
        expected_c = torch.zeros(*size)
        # 使用 map2_ 方法在 expected_c 上应用 mathfn 函数
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        # 断言张量 c 与 expected_c 相等，允许误差为 0
        self.assertEqual(expected_c, c, atol=0, rtol=0)

    # 标记为只在 CPU 上运行的测试用例
    @onlyCPU
    # 定义 test_max_elementwise 函数，接受参数 self 和 device
    def test_max_elementwise(self, device):
        # 调用 _testCSelection 函数，使用 torch.max 函数和 Python 内置的 max 函数
        self._testCSelection(torch.max, max)

    # 标记为只在 CPU 上运行的测试用例
    @onlyCPU
    # 定义 test_min_elementwise 函数，接受参数 self 和 device
    def test_min_elementwise(self, device):
        # 调用 _testCSelection 函数，使用 torch.min 函数和 Python 内置的 min 函数
        self._testCSelection(torch.min, min)

    # 定义 test_all_any 函数，接受参数 self 和 device
    def test_all_any(self, device):
        # 定义内部函数 test，接受参数 size
        def test(size):
            # 创建一个全为 1 的大小为 size 的 torch 张量 x
            x = torch.ones(*size, device=device).byte()
            # 断言 x 所有元素为 True
            self.assertTrue(x.all())
            # 断言 x 至少有一个元素为 True
            self.assertTrue(x.any())

            # 将第 3 个元素设为 0
            x[3] = 0
            # 断言 x 所有元素不全为 True
            self.assertFalse(x.all())
            # 断言 x 至少有一个元素为 True
            self.assertTrue(x.any())

            # 将 x 所有元素设为 0
            x.zero_()
            # 断言 x 所有元素不全为 True
            self.assertFalse(x.all())
            # 断言 x 所有元素至少有一个为 True
            self.assertFalse(x.any())

            # 将 x 所有元素设为 2
            x.fill_(2)
            # 断言 x 所有元素为 True
            self.assertTrue(x.all())
            # 断言 x 至少有一个元素为 True
            self.assertTrue(x.any())

            # 创建一个全为 1 的大小为 size 的 torch 张量 x
            x = torch.ones(*size, device=device).bool()
            # 断言 x 所有元素为 True
            self.assertTrue(x.all())
            # 断言 x 至少有一个元素为 True
            self.assertTrue(x.any())

            # 将第 3 个元素设为 False
            x[3] = False
            # 断言 x 所有元素不全为 True
            self.assertFalse(x.all())
            # 断言 x 至少有一个元素为 True
            self.assertTrue(x.any())

        # 调用 test 函数进行多个测试
        test((10,))
        test((5, 5))
    # 在给定设备上测试所有维度的 torch.Tensor 对象的 all 和 any 方法
    def test_all_any_with_dim(self, device):
        # 内部测试函数，接受一个张量 x
        def test(x):
            # 计算沿着第 0 维度的乘积，并转换为字节类型
            r1 = x.prod(dim=0, keepdim=False).byte()
            # 计算沿着第 0 维度的 all 方法结果
            r2 = x.all(dim=0, keepdim=False)
            # 断言两者形状相同
            self.assertEqual(r1.shape, r2.shape)
            # 断言两者所有元素相等
            self.assertTrue((r1 == r2).all())

            # 计算沿着第 1 维度的和，并在保持维度的情况下进行范围限制和转换为字节类型
            r3 = x.sum(dim=1, keepdim=True).clamp(0, 1).byte()
            # 计算沿着第 1 维度的 any 方法结果
            r4 = x.any(dim=1, keepdim=True)
            # 断言两者形状相同
            self.assertEqual(r3.shape, r4.shape)
            # 断言两者所有元素相等
            self.assertTrue((r3 == r4).all())

        # 调用 test 函数，传入一个特定的张量作为输入
        test(torch.tensor([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], device=device, dtype=torch.uint8))

    # 在给定设备上测试 torch.add 函数的使用，使用命名参数传递
    def test_numpy_named_args(self, device):
        # 创建随机张量 x1 和 x2
        x1 = torch.randn(10, device=device)
        x2 = torch.randn(10, device=device)
        # 使用命名参数 input 和 other 调用 torch.add 函数
        res1 = torch.add(input=x1, other=x2)
        # 使用命名参数 x1 和 x2 调用 torch.add 函数
        res2 = torch.add(x1=x1, x2=x2)
        # 断言两种调用方式得到的结果张量相等
        self.assertEqual(res1, res2)

        # 创建一个大张量 x1
        x1 = torch.randn(10, 10, 10, device=device)
        # 对 x1 沿着第 0 和第 2 维度进行求和，并保持维度
        res1 = x1.sum(dim=(0, 2), keepdim=True)
        # 对 x1 沿着轴 (0, 2) 进行求和，并保持维度
        res2 = x1.sum(axis=(0, 2), keepdims=True)
        # 断言两种调用方式得到的结果张量相等
        self.assertEqual(res1, res2)

    # TODO: kill this and replace with common creation ops
    # 根据给定的形状创建张量字典，包含不同类型的张量
    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True,
                      use_complex=False) -> Dict[str, List[torch.Tensor]]:
        # 支持的浮点数类型张量
        float_types = [torch.double,
                       torch.float]
        # 支持的整数类型张量
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        # 支持的复数类型张量
        complex_types = [torch.complex64,
                         torch.complex128]

        # 创建连续内存的张量
        def make_contiguous(shape, dtype) -> torch.Tensor:
            if dtype in float_types:
                # 创建服从正态分布的张量，并根据给定范围进行缩放和平移
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            # 创建全零张量，元素为在给定范围内随机整数
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        # 创建非连续内存的张量
        def make_non_contiguous(shape, dtype) -> torch.Tensor:
            contig = make_contiguous(shape, dtype)
            # 创建指定形状的空张量，并选择其中一部分作为非连续内存
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            # 断言非连续内存张量确实不是连续的
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        # 创建连续内存的切片张量
        def make_contiguous_slice(size, dtype) -> torch.Tensor:
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            # 断言切片张量确实是连续的
            self.assertTrue(non_contig.is_contiguous())
            return contig

        # 根据使用标志收集需要创建的张量类型
        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        if use_complex:
            types += complex_types
        # 初始化张量字典，包括连续、非连续和切片张量
        tensors: Dict[str, List[torch.Tensor]] = {"cont": [], "noncont": [], "slice": []}
        # 根据每种类型创建张量，并添加到对应列表中
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        # 返回创建的张量字典
        return tensors

    # TODO: refactor this to use comparators from common_utils
    # 使用 common_utils 中的比较器重构此方法
    def _assert_matches_numpy(self, t, n):
        self.assertEqual(n.shape, t.shape)
        if t.dtype == torch.float:
            self.assertEqual(n, t, rtol=1e-03, atol=1e-05, equal_nan=True)
        else:
            self.assertEqual(n, t, equal_nan=True)

    # TODO: update this and tests that use it to use the device argument properly
    # 更新此处及其使用它的测试用例，正确使用 device 参数
    # 定义一个测试函数 _test_dim_ops，用于测试维度操作的函数
    def _test_dim_ops(self, pytorch_op, numpy_op,
                      use_floating=True, use_integral=True, use_complex=False):
        # 定义内部函数 do_one，用于对给定的张量字典进行操作
        def do_one(tensors_dict, dim):
            # 遍历张量字典中的每个类别和张量
            for category, tensors in tensors_dict.items():
                # 如果类别是 "slice"，将维度设为0
                if category == "slice":
                    dim = 0
                # 遍历当前类别下的每个张量
                for tensor in tensors:
                    # 忽略 NumPy 的警告信息
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # 使用 NumPy 操作函数在 CPU 数组上进行操作，得到期望的结果
                        expected = numpy_op(tensor.cpu().numpy(), dim)
                    # 使用 PyTorch 操作函数在张量上进行操作，得到实际结果
                    actual = pytorch_op(tensor, dim)
                    # 断言实际结果与期望结果匹配
                    self._assert_matches_numpy(actual, expected)
                    # 如果 CUDA 可用，再次进行相同操作并断言结果匹配
                    if torch.cuda.is_available():
                        self._assert_matches_numpy(pytorch_op(tensor.cuda(), dim).cpu(), expected)
        
        # 对不同形状的张量进行测试，并指定不同的维度参数
        do_one(self._make_tensors((5, 400000), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((100000, ), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), -1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, -1))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2, 1))

    # 将当前测试标记为一个较慢的测试
    @slowTest
    # 指定只在 CPU 上运行当前测试
    @onlyCPU
    # 定义一个测试函数，用于测试对指定维度进行求和操作
    def test_sum_dim(self, device):
        # 调用 _test_dim_ops 函数，传入 lambda 表达式对张量进行求和操作，以及对应的 numpy 操作
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d),
            use_floating=True, use_integral=True, use_complex=True)

    # 仅在 CPU 上运行的测试函数，用于测试对指定维度进行求平均值操作
    @onlyCPU
    def test_mean_dim(self, device):
        # 调用 _test_dim_ops 函数，传入 lambda 表达式对张量进行求平均值操作，以及对应的 numpy 操作
        self._test_dim_ops(
            lambda t, d: t.mean(d),
            lambda n, d: n.mean(d),
            use_integral=False,
            use_complex=True)

    # 仅在 CPU 上运行的测试函数，用于测试对指定维度进行求标准差操作
    @onlyCPU
    def test_std_dim(self, device):
        # 遍历 unbiased 参数的取值 [False, True]
        for unbiased in [False, True]:
            # 调用 _test_dim_ops 函数，传入 lambda 表达式对张量进行求标准差操作，以及对应的 numpy 操作
            self._test_dim_ops(
                lambda t, d: t.std(d, unbiased=unbiased),
                lambda n, d: n.std(d, ddof=1 if unbiased else 0),
                use_integral=False)

    # 仅在 CPU 上运行的测试函数，用于测试对指定维度进行求方差操作
    @onlyCPU
    def test_var_dim(self, device):
        # 遍历 unbiased 参数的取值 [False, True]
        for unbiased in [False, True]:
            # 调用 _test_dim_ops 函数，传入 lambda 表达式对张量进行求方差操作，以及对应的 numpy 操作
            self._test_dim_ops(
                lambda t, d: t.var(d, unbiased=unbiased),
                lambda n, d: n.var(d, ddof=1 if unbiased else 0),
                use_integral=False)

    # 仅在 CPU 上运行的测试函数，用于测试对指定维度进行求 logsumexp 操作
    @onlyCPU
    @skipIfNoSciPy
    def test_logsumexp_dim(self, device):
        # 导入 scipy.special 中的 logsumexp 函数
        from scipy.special import logsumexp
        # 调用 _test_dim_ops 函数，传入 lambda 表达式对张量进行求 logsumexp 操作，以及对应的 scipy 操作
        self._test_dim_ops(
            lambda t, d: t.logsumexp(d),
            lambda n, d: logsumexp(n, d),
            use_integral=False)

    # 仅在 CPU 上运行的测试函数，用于测试对整数类型张量进行求均值操作，并指定输出类型
    @onlyCPU
    def test_mean_int_with_optdtype(self, device):
        # 创建一个整数类型的张量 a
        a = make_tensor((3, 4, 5), dtype=torch.int64, device=device)

        # 将张量 a 转换为 float32 类型
        a_float = a.to(torch.float32)
        # 断言转换后的张量的均值等于指定输出类型为 float32 的张量 a 的均值
        self.assertEqual(a_float.mean(), a.mean(dtype=torch.float32))

    # TODO: update this and tests that use it to handle device properly
    # 定义测试函数，用于测试整数向上转型的函数
    def _test_reduce_integer_upcast(self, fn, has_out=True, test_complex=True):
        # 定义一个形状为 (3, 4, 5) 的张量
        shape = (3, 4, 5)
        # 调用传入函数 fn 处理全为 1 的张量并获取结果形状
        reduced_shape = fn(torch.ones(shape)).shape

        # 定义内部测试函数 _test_out，接受两种数据类型并验证转型结果
        def _test_out(dtype, other_dtype):
            # 创建一个与 reduced_shape 相同形状和指定数据类型的全 1 张量
            out = torch.ones(reduced_shape, dtype=dtype)
            # 调用 fn 处理输入张量 x，结果存入 out
            result = fn(x, out=out)
            # 验证输出张量 out 的数据类型与 fn 处理结果的数据类型一致
            self.assertIs(out.dtype, result.dtype)
            # 比较 fn 处理输入张量 x 转换为 dtype 后的结果与 result 是否相等
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            # 使用指定数据类型 dtype 调用 fn 处理输入张量 x，结果存入 result
            result = fn(x, out=out, dtype=dtype)
            # 验证输出张量 out 的数据类型与 fn 处理结果的数据类型一致
            self.assertIs(out.dtype, result.dtype)
            # 比较 fn 处理输入张量 x 转换为 dtype 后的结果与 result 是否相等
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            # 'out' 优先于 dtype，验证当同时指定 out 和其他数据类型时是否会引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: fn(x, out=out, dtype=other_dtype))

        # 遍历所有数学数据类型，排除 torch.float16，对张量 x 进行测试
        for dtype in [dtype for dtype in get_all_math_dtypes('cpu') if dtype != torch.float16]:
            # 创建形状为 shape，数据类型为 dtype 的全 1 张量 x
            x = torch.ones(shape, dtype=dtype)
            # 计算预期的 dtype 类型，如果是浮点型或复数型，预期类型为 dtype，否则为 torch.int64
            expected_dtype = dtype if dtype.is_floating_point or dtype.is_complex else torch.int64
            # 验证 fn 处理 x 后的结果数据类型为预期的数据类型 expected_dtype
            self.assertIs(expected_dtype, fn(x).dtype)
            # 比较 fn 处理 x 转换为 expected_dtype 后的结果与原始 fn(x) 的结果是否相等
            self.assertEqual(fn(x.to(expected_dtype)), fn(x))

            # 根据 dtype 类型设置其他数据类型 other_dtype
            if dtype.is_floating_point:
                other_dtype = torch.float32 if dtype == torch.float64 else torch.float64
            elif dtype.is_complex:
                other_dtype = torch.complex64 if dtype == torch.complex128 else torch.complex128
            else:
                other_dtype = torch.int32 if dtype != torch.int32 else torch.int16
            # 验证 fn 处理 x 时，使用其他数据类型 other_dtype 的结果数据类型是否为 other_dtype
            self.assertIs(other_dtype, fn(x, dtype=other_dtype).dtype)
            # 比较 fn 处理 x 转换为 other_dtype 后的结果与原始 fn(x) 的结果是否相等（允许类型不严格匹配）
            self.assertEqual(fn(x.to(other_dtype)), fn(x, dtype=other_dtype), exact_dtype=False)

            # 测试混合整数/浮点/复数类型
            if dtype.is_floating_point:
                mixed_dtypes = [torch.int32, torch.complex64]
            elif dtype.is_complex:
                mixed_dtypes = [torch.int32, torch.float32]
            else:
                mixed_dtypes = [torch.float32, torch.complex64]

            # 遍历混合数据类型 mixed_dtypes，验证 fn 处理 x 使用 mixed_dtype 的结果数据类型是否为 mixed_dtype
            for mixed_dtype in mixed_dtypes:
                self.assertIs(mixed_dtype, fn(x, dtype=mixed_dtype).dtype)
                # 比较 fn 处理 x 转换为 mixed_dtype 后的结果与原始 fn(x) 的结果是否相等（允许类型不严格匹配）
                self.assertEqual(fn(x.to(mixed_dtype)), fn(x, dtype=mixed_dtype), exact_dtype=False)

                # 如果存在 out 参数，则测试 _test_out 函数处理不同数据类型情况
                if has_out:
                    _test_out(dtype, other_dtype)
                    _test_out(dtype, mixed_dtype)

    # 标记仅在 CPU 上运行的测试函数，测试 torch.sum 函数对整数向上转型的处理
    @onlyCPU
    def test_sum_integer_upcast(self, device):
        # 调用 _test_reduce_integer_upcast 测试函数，使用 torch.sum 处理张量 x，不使用 out 参数
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, **kwargs), False)
        # 调用 _test_reduce_integer_upcast 测试函数，使用 torch.sum 处理张量 x，沿着第一个维度求和
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, 0, **kwargs))

    # 标记仅在 CPU 上运行的测试函数，测试 torch.prod 函数对整数向上转型的处理
    @onlyCPU
    def test_prod_integer_upcast(self, device):
        # 调用 _test_reduce_integer_upcast 测试函数，使用 torch.prod 处理张量 x，不使用 out 参数
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, **kwargs), False)
        # 调用 _test_reduce_integer_upcast 测试函数，使用 torch.prod 处理张量 x，沿着第一个维度求积
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, 0, **kwargs))

    # 标记仅在 CPU 上运行的测试函数，测试 torch.cumsum 函数对整数向上转型的处理
    @onlyCPU
    def test_cumsum_integer_upcast(self, device):
        # 调用 _test_reduce_integer_upcast 测试函数，使用 torch.cumsum 处理张量 x，沿着第一个维度累积求和
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumsum(x, 0, **kwargs))

    # 标记仅在 CPU 上运行的测试函数
    # 定义测试方法，用于测试 torch.cumprod 函数对整数类型向上转型的情况
    def test_cumprod_integer_upcast(self, device):
        # 调用通用的测试方法 _test_reduce_integer_upcast，测试 torch.cumprod 函数
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumprod(x, 0, **kwargs))

    # 使用所有数据类型定义测试方法 test_mode
    @dtypes(*all_types())
    def test_mode(self, device, dtype):
        SIZE = 10
        # 创建一个在设备上的张量 x，包含从 1 到 SIZE*SIZE 的连续浮点数，然后重新调整大小为 SIZE x SIZE
        x = torch.arange(1., SIZE * SIZE + 1, device=device, dtype=dtype).clone().resize_(SIZE, SIZE)
        # 将 x 的前两行设置为1
        x[:2] = 1
        # 将 x 的前两列设置为1
        x[:, :2] = 1
        # 保存 x 的初始副本
        x0 = x.clone()

        # 预先计算的结果
        res1val = torch.ones(SIZE, device=device, dtype=dtype)
        # 结果的索引是模式元素最后出现的位置
        res1ind = torch.ones(SIZE, device=device, dtype=torch.long)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        # 计算 x 的模式值和索引
        res2val, res2ind = torch.mode(x, keepdim=False)
        # 断言计算得到的模式值等于预期值，绝对容差和相对容差都为0
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        # 断言计算得到的模式索引等于预期索引，绝对容差和相对容差都为0
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # 测试结果张量的使用
        res2val = torch.tensor((), device=device, dtype=dtype)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        # 将计算的模式值和索引存储到指定的结果张量中
        torch.mode(x, keepdim=False, out=(res2val, res2ind))
        # 断言计算得到的模式值等于预期值，绝对容差和相对容差都为0
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        # 断言计算得到的模式索引等于预期索引，绝对容差和相对容差都为0
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # 测试非默认维度的模式计算
        res2val, res2ind = torch.mode(x, 0, False)
        # 断言计算得到的模式值等于预期值，绝对容差和相对容差都为0
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        # 断言计算得到的模式索引等于预期索引，绝对容差和相对容差都为0
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # 输入张量不变
        self.assertEqual(x, x0, atol=0, rtol=0)

    # 测试模式间隔的私有方法，用于测试不同形状和间隔的输入张量
    def _test_mode_intervals(self, shape, intervals, device, dtype, v=1):
        # 创建一个在设备上的张量 x，包含从0到 shape[1] 的整数，然后扩展为指定形状
        x = torch.arange(0, shape[1], device=device, dtype=dtype).expand(shape)
        # 使 x 连续
        x = x.contiguous()
        # 将 x 的第 v 列设置为 intervals[0][0] 的值
        x[:, v] = intervals[0][0]

        # 将每个间隔的值设置为模式 v
        for (beg, end) in intervals:
            x[:, beg:end] = v

        # 计算 x 每行的模式值和索引
        values, indices = torch.mode(x, -1, False)

        # 检查返回的索引是否对应于返回的值
        self.assertTrue((x.gather(1, indices.unsqueeze(1)).t() == values).all())
        # 检查返回的值是否为模式 v
        self.assertTrue((values == v).all().item())

    # 仅在 CUDA 下进行测试，包含所有数据类型以及 torch.half 和 torch.bfloat16 类型
    @onlyCUDA
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_mode_large(self, device, dtype):
        # 定义用于指定形状和索引 i 的测试集合
        def testset_for_shape(shape, i):
            d = shape[-1]
            # 在中间位置测试模式
            self._test_mode_intervals(shape, [(i, d - i)], device, dtype)
            # 在输入的不连续部分测试模式
            self._test_mode_intervals(shape, [(0, i), (i + 1, d - i - 1), (d - i, d)], device, dtype)

        # 多于一行的（65535）线程块
        testset_for_shape((65536, 10), 3)

        # 最大切片大小（2048）
        testset_for_shape((10, 2048), 10)

        # 大于 2048 的切片大小的简单核心
        testset_for_shape((10, 4096), 10)
    # 定义一个测试函数，测试布尔类型张量的模式计算在给定设备上的行为
    def test_mode_boolean(self, device):
        # 定义不同形状的张量作为测试数据
        shapes = [
            (10, 10),
            (4, 2048),
            (1, 4096),
        ]

        # 遍历每个形状
        for shape in shapes:
            # 创建一个全零布尔类型张量，并指定设备和数据类型
            a = torch.zeros(shape, device=device, dtype=torch.bool)

            # 将张量的右半部分设为 True
            a[:, (shape[1] - 1) // 2:] = True
            # 计算张量在最后一个维度上的模式值和索引
            values, indices = a.mode(-1)
            # 断言模式计算结果的值与预期的全一张量相等
            self.assertEqual(values, torch.ones(shape[0], dtype=torch.bool))
            # 打印索引值
            print(indices)
            # 使用索引从张量中聚集数据，并断言与模式值相等
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)

            # 将张量全部设为 False
            a.fill_(False)
            # 将张量的右半部分设为 True
            a[:, shape[1] // 2 + 1:] = True
            # 计算张量在最后一个维度上的模式值和索引
            values, indices = a.mode(-1)
            # 打印索引值
            print(indices)
            # 断言模式计算结果的值与预期的全零张量相等
            self.assertEqual(values, torch.zeros(shape[0], dtype=torch.bool))
            # 使用索引从张量中聚集数据，并断言与模式值相等
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)


    @expectedFailureMeta  # mode only supports CPU and CUDA device type
    @onlyNativeDeviceTypes
    # 定义一个测试函数，测试在设备类型错误的情况下，torch.mode 函数的行为
    def test_mode_wrong_dtype(self, device):
        # 定义一个内部函数，测试不同数据类型的情况
        def test_for_dtypes(x_ty, v_ty, i_ty, message):
            # 创建一个全一张量 x，并指定设备和数据类型
            x = torch.ones(10, device=device, dtype=x_ty)
            # 创建一个全一张量 v，并指定设备和数据类型
            v = torch.ones(10, device=device, dtype=v_ty)
            # 创建一个全一张量 i，并指定设备和数据类型
            i = torch.ones(10, device=device, dtype=i_ty)

            # 使用断言检查调用 torch.mode 时的错误信息
            with self.assertRaisesRegex(RuntimeError, message):
                torch.mode(x, -1, True, out=(v, i))

        # 定义错误消息模板
        err_msg = "expected scalar type .* but got .* for "
        values_err = err_msg + "values"
        indices_err = err_msg + "indices"

        # 调用内部函数测试不同数据类型的错误情况
        test_for_dtypes(torch.uint8, torch.int8, torch.long, values_err)
        test_for_dtypes(torch.int8, torch.int16, torch.long, values_err)
        test_for_dtypes(torch.int32, torch.float32, torch.long, values_err)
        test_for_dtypes(torch.float32, torch.float64, torch.long, values_err)

        test_for_dtypes(torch.uint8, torch.uint8, torch.int8, indices_err)
        test_for_dtypes(torch.int8, torch.int8, torch.int16, indices_err)
        test_for_dtypes(torch.int32, torch.int32, torch.float32, indices_err)
        test_for_dtypes(torch.float32, torch.float32, torch.float64, indices_err)


    @onlyCUDA
    # 定义一个测试函数，测试在 CUDA 设备上调用时的错误情况
    def test_mode_wrong_device(self, device):
        # 创建一个 CPU 上的张量 x
        x = torch.ones(2)

        # 使用断言检查调用 torch.mode 时的错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for values"):
            # 创建一个空的 values 张量，并指定设备
            values = torch.tensor([], device=device)
            # 调用 torch.mode 操作，期望抛出设备类型错误
            torch.mode(x, -1, True, out=(values, torch.tensor([], dtype=torch.long)))

        # 使用断言检查调用 torch.mode 时的错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for indices"):
            # 创建一个空的 indices 张量，并指定设备
            indices = torch.tensor([], device=device)
            # 调用 torch.mode 操作，期望抛出设备类型错误
            torch.mode(x, -1, True, out=(torch.tensor([]), indices))

    # TODO: make work on CUDA, too
    # 标记一个功能尚未实现，在 CUDA 上也能正常工作
    @onlyCPU
    # 测试函数，用于验证 torch.Tensor 的 accreal 类型转换
    def test_accreal_type(self, device) -> None:
        # 创建一个形状为 (2, 3, 4) 的全为1的张量 x
        x = torch.ones(2, 3, 4)
        # 验证对 x 进行 double() 后的和的计算结果是否为 float 类型
        self.assertIsInstance(x.double().sum().item(), float)
        # 验证对 x 进行 float() 后的和的计算结果是否为 float 类型
        self.assertIsInstance(x.float().sum().item(), float)
        # 验证对 x 进行 long() 后的和的计算结果是否为 int 类型
        self.assertIsInstance(x.long().sum().item(), int)
        # 验证对 x 进行 int() 后的和的计算结果是否为 int 类型
        self.assertIsInstance(x.int().sum().item(), int)
        # 验证对 x 进行 short() 后的和的计算结果是否为 int 类型
        self.assertIsInstance(x.short().sum().item(), int)
        # 验证对 x 进行 char() 后的和的计算结果是否为 int 类型
        self.assertIsInstance(x.char().sum().item(), int)
        # 验证对 x 进行 byte() 后的和的计算结果是否为 int 类型
        self.assertIsInstance(x.byte().sum().item(), int)

    # 测试函数，用于验证 torch.Tensor 的 var_mean 操作在不同维度上的计算
    def test_var_mean_some_dims(self, device):
        # 定义张量的维度
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)

        # 在指定设备上创建一个随机张量 x
        x = torch.rand(sizes, device=device)
        # 遍历 2 到 dims 之间的所有可能的维度组合
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for unbiased in [False, True]:
                    for keepdim in [False, True]:
                        # 使用 torch.var_mean 函数计算指定维度上的方差和均值
                        var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        # 使用张量对象方法计算指定维度上的方差
                        var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        # 使用张量对象方法计算指定维度上的均值
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        # 验证 torch.var_mean 的结果与张量对象方法计算的结果是否一致
                        self.assertEqual(var1, var2)
                        self.assertEqual(mean1, mean2)

    # TODO: this should be a generic opinfo test
    # 测试函数，用于验证 torch.Tensor 的 all 和 any 方法在空张量上的行为
    def test_all_any_empty(self, device):
        # 创建一个空的 ByteTensor 并放置在指定设备上
        x = torch.ByteTensor().to(device)
        # 验证空张量上的 all 方法返回 True
        self.assertTrue(x.all())
        # 验证空张量上的 any 方法返回 False
        self.assertFalse(x.any())

        # 创建一个空的 BoolTensor 并放置在指定设备上
        x = torch.BoolTensor().to(device)
        # 验证空张量上的 all 方法返回 True
        self.assertTrue(x.all())
        # 验证空张量上的 any 方法返回 False
        self.assertFalse(x.any())

    # 测试函数，用于验证 torch.Tensor 的 all 方法在指定情况下的行为
    def test_all_issue117215(self, device):
        # 获取 torch.uint8 类型的数据信息
        info = torch.iinfo(torch.uint8)
        # 创建一个指定范围内随机整数张量 a，并放置在指定设备上
        a = torch.randint(info.min, info.max, (73, 11, 3, 17), dtype=torch.uint8)
        # 使用 torch.all 函数对张量 a 进行计算
        b = torch.all(a, dim=0)
        # 使用 to(torch.bool) 方法将张量 a 转换为布尔型张量后再使用 all 方法计算
        c = a.to(torch.bool).all(dim=0)
        # 验证两种计算方式的结果是否一致
        self.assertEqual(torch.ne(b, c).sum(), 0)

    # 测试函数，用于验证 torch.Tensor 的 max 方法在包含无穷大元素时的行为
    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_max_with_inf(self, device, dtype):
        # 创建一个包含无穷大元素的张量 a，并放置在指定设备上
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        # 验证对张量 a 进行 max(dim=1) 后得到的值是否全为 inf
        self.assertTrue(torch.all(torch.max(a, dim=1).values == inf).item())
        # 验证对张量 a 进行 amax(dim=1) 后得到的值是否全为 inf
        self.assertTrue(torch.all(torch.amax(a, dim=1) == inf).item())
        # 验证对张量 a 进行 max() 后得到的值是否为 inf
        self.assertTrue(torch.max(a).item() == inf)
        # 验证对张量 a 进行 amax() 后得到的值是否为 inf
        self.assertTrue(torch.amax(a).item() == inf)

    # 测试函数，用于验证 torch.Tensor 的 min 方法在包含无穷大元素时的行为
    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double)
    def test_min_with_inf(self, device, dtype):
        # 创建一个包含无穷大元素的张量 a，并放置在指定设备上
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        # 验证对张量 a 进行 min(dim=1) 后得到的值是否全为 -inf
        self.assertTrue(torch.all(torch.min(a, dim=1).values == (-inf)).item())
        # 验证对张量 a 进行 amin(dim=1) 后得到的值是否全为 -inf
        self.assertTrue(torch.all(torch.amin(a, dim=1) == (-inf)).item())
        # 验证对张量 a 进行 min() 后得到的值是否为 -inf
        self.assertTrue(torch.min(a).item() == -inf)
        # 验证对张量 a 进行 amin() 后得到的值是否为 -inf
        self.assertTrue(torch.amin(a).item() == -inf)
    # 定义一个帮助函数，用于测试 torchfn 和 reffn 的最大和最小函数
    def _test_minmax_helper(self, torchfn, reffn, device, dtype, skip_indices=False):
        
        # 创建输入数据的辅助函数，根据给定的形状、设备和数据类型生成随机或随机整数数据
        def create_input(shape, device, dtype):
            if dtype.is_floating_point:
                return torch.randn(*shape, device=device, dtype=dtype)
            else:
                low = 0 if dtype == torch.bool else -1000
                high = 2 if dtype == torch.bool else 1000
                return torch.randint(low, high, shape, device=device, dtype=dtype)
        
        # 创建一个大小为 (100, 100) 的输入数据 x
        x = create_input((100, 100), device, dtype)
        # 使用 torchfn 和 reffn 进行比较
        self.compare_with_numpy(torchfn, reffn, x)
        
        # 创建一个非连续的输入数据 x
        x = create_input((10, 10, 10), device, dtype)
        x = x[:, 4]  # 选择 x 的第二个维度的第五个索引
        # 使用 torchfn 和 reffn 进行比较
        self.compare_with_numpy(torchfn, reffn, x)

        # 定义一个获取值的函数，用于处理可能是元组的返回值 x
        def get_values(x):
            if isinstance(x, tuple):
                return x[0]
            return x

        # 如果不跳过索引操作
        if not skip_indices:
            size = 5
            # 创建一个大小为 (size, size) 的输入数据 x
            x = create_input((size, size), device, dtype)
            inputs = (x, x.t())  # 输入数据为 x 和 x 的转置
            dims = (0, 1)  # 维度选择为 0 和 1
            # 遍历 inputs 和 dims 的笛卡尔积
            for xinp, d in product(inputs, dims):
                # 使用 torchfn 对 xinp 进行计算，获取结果，并与 reffn 的结果进行比较
                self.compare_with_numpy(lambda x: get_values(torchfn(x, d, False)), lambda x: reffn(x, d, keepdims=False), xinp)
                result = torchfn(xinp, d, False)
                # 如果结果是元组
                if isinstance(result, tuple):
                    v, i = result
                    # 如果维度 d 是 1
                    if d == 1:
                        # 检查是否满足条件：xinp 在维度 0 上索引为 i 的值等于 v
                        self.assertEqual(xinp[torch.arange(size), i], v, atol=0, rtol=0)
                    else:
                        # 检查是否满足条件：xinp 在维度 1 上索引为 i 的值等于 v
                        self.assertEqual(xinp[i, torch.arange(size)], v, atol=0, rtol=0)
        
        # 如果数据类型是浮点型
        if dtype.is_floating_point:
            # 遍历一些索引值来设置 x 中的 NaN 值
            for index in (0, 4, 99):
                x = create_input((100,), device, dtype)
                x[index] = nan  # 将指定索引处的值设置为 NaN
                # 如果不跳过索引操作
                if not skip_indices:
                    result = torchfn(x, 0)
                    v = get_values(result)
                    # 断言 v 的值等于 NaN
                    self.assertEqual(v, nan)
                    # 如果结果是元组
                    if isinstance(result, tuple):
                        i = result[1]
                        # 断言 i 的值等于 index
                        self.assertEqual(i, index)
                # 断言 torchfn(x) 的结果为 NaN
                self.assertEqual(torchfn(x), nan)

    # 使用 @dtypesIfCPU 和 @dtypesIfCUDA 装饰器来测试 torch.max 函数
    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    def test_max(self, device, dtype):
        # 调用 _test_minmax_helper 函数，测试 torch.max 函数
        self._test_minmax_helper(torch.max, np.amax, device, dtype)

    # 使用 @dtypesIfCPU 和 @dtypesIfCUDA 装饰器来测试 torch.min 函数
    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    def test_min(self, device, dtype):
        # 调用 _test_minmax_helper 函数，测试 torch.min 函数
        self._test_minmax_helper(torch.min, np.amin, device, dtype)
    # 测试函数，用于测试 torch.amin 方法与 np.amin 方法的输出是否一致
    def test_amin(self, device, dtype):
        self._test_minmax_helper(torch.amin, np.amin, device, dtype)

    # 根据设备与数据类型选择性地测试 torch.amax 方法
    @dtypesIfCPU(torch.half, torch.float, torch.double, torch.int, torch.long, torch.bool)
    @dtypesIfCUDA(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypes(torch.float, torch.double)
    def test_amax(self, device, dtype):
        self._test_minmax_helper(torch.amax, np.amax, device, dtype)

    # 仅对原生设备类型进行测试，测试 torch.aminmax 方法的封装函数
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.bfloat16)
    def test_aminmax(self, device, dtype):

        # 封装函数，调用 torch.aminmax 方法的第一个返回值
        def _amin_wrapper(x, dim=None, keepdims=False):
            return torch.aminmax(x, dim=dim, keepdim=keepdims)[0]

        # 封装函数，调用 torch.aminmax 方法的第二个返回值
        def _amax_wrapper(x, dim=None, keepdims=False):
            return torch.aminmax(x, dim=dim, keepdim=keepdims)[1]

        # 分别测试 _amin_wrapper 与 _amax_wrapper 函数
        self._test_minmax_helper(_amin_wrapper, np.amin, device, dtype)
        self._test_minmax_helper(_amax_wrapper, np.amax, device, dtype)

    # 仅对复数类型进行测试，测试不支持 0 维度的 torch.aminmax 方法
    def test_invalid_0dim_aminmax(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            torch.aminmax(torch.tensor(1., dtype=dtype, device=device), dim=0)

    # TODO: bincount 不是经典的归约操作，也许这个测试套件是关于归约和汇总操作？
    # TODO: 有多少个 var 稳定性测试？
    # 测试 tensor 的方差稳定性，包括内部维度、整体以及外部维度的情况
    def test_var_stability2(self, device):
        tensor = torch.FloatTensor([2281.5, 2281.25]).to(device)

        # 测试内部维度的方差稳定性
        self.assertEqual(tensor.var(0), 0.03125)

        # 测试整体方差稳定性
        self.assertEqual(tensor.var(), 0.03125)

        # 测试外部维度的方差稳定性
        tensor = tensor.unsqueeze(1)
        self.assertEqual(tensor.var(0), 0.03125)

    # 仅对 CPU 进行测试，使用 torch.bfloat16 与 torch.float16 数据类型
    @onlyCPU
    @dtypes(torch.bfloat16, torch.float16)
    # 定义一个测试方法，用于测试非连续维度的求和操作，包括设备和数据类型参数
    def test_sum_noncontig_lowp(self, device, dtype) -> None:
        # 定义不同维度的序列字典，每个维度对应的是下标列表
        dim_sequences = {
            2: [0, 1],
            3: [0, 1, 2],
            4: [0, 1, 2, 3],
            5: [0, 1, 2, 3, 4],
        }

        # 定义创建非连续输入的内部函数，根据不同维度对输入张量进行切片
        def create_noncontig_inputs(x, ndim):
            if ndim == 2:
                return x[::2, ::2]
            elif ndim == 3:
                return x[::2, ::2, ::2]
            elif ndim == 4:
                return x[::2, ::2, ::2, ::2]
            elif ndim == 5:
                return x[::2, ::2, ::2, ::2, ::2]

        # 定义辅助函数，用于执行测试操作，包括张量的重排列、求和和断言比较
        def helper(self, shape, reduce_dims, device, dtype):
            # 对每个维度序列的全排列进行迭代
            for permute_list in list(permutations(dim_sequences[len(shape)], len(shape))):
                # 创建指定形状的全1张量，并根据当前形状进行非连续输入创建
                x = torch.ones(shape, device=device, dtype=dtype)
                x = create_noncontig_inputs(x, len(shape))
                # 使用当前排列重排张量
                x_trans = x.permute(permute_list)
                # 对重排后的张量进行指定维度的求和操作
                x_sum = torch.sum(x_trans, reduce_dims)
                # 将重排后的张量转换为浮点型，进行参考求和操作
                x_trans_ref = x_trans.float()
                x_sum_ref = torch.sum(x_trans_ref, reduce_dims)
                # 断言两个求和结果是否相等
                self.assertEqual(x_sum, x_sum_ref.to(dtype=dtype))

        # 定义不同形状的测试用例列表
        shapes = [
            (50, 50),
            (50, 50, 50),
            (10, 50, 30, 30),
            (10, 5, 10, 50, 7),
        ]

        # 遍历每个形状的测试用例
        for shape in shapes:
            # 对每个形状的测试用例，遍历可能的维度组合
            for i in range(1, len(shape) + 1):
                reduce_dims = list(combinations(dim_sequences[len(shape)], i))
                # 对每个维度组合，调用辅助函数执行测试
                for reduce_dim in reduce_dims:
                    helper(self, shape, reduce_dim, device, dtype)


    # 标记为只在 CPU 上运行的测试，并指定数据类型
    @onlyCPU
    @dtypes(torch.bool, torch.double)
    def test_sum_all(self, device, dtype) -> None:
        # 定义检查张量全局求和的函数
        def check_sum_all(tensor: torch.Tensor) -> None:
            # 将张量重塑为一维数组并转换为 Python 列表
            pylist = tensor.reshape(-1).tolist()
            # 断言张量的总和与 Python 列表的总和相等
            self.assertEqual(tensor.sum(), sum(pylist))

        # 如果数据类型不是布尔型
        if dtype != torch.bool:
            # 对不同张量进行全局求和的检查
            check_sum_all(torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device=device))
            check_sum_all(torch.randn(200000, dtype=dtype, device=device))
            check_sum_all(torch.randn(2000, 2, dtype=dtype, device=device)[:, 0])
        else:
            # 对布尔型张量进行全局求和的检查
            check_sum_all(torch.tensor([True, False, True], dtype=torch.bool, device=device))
    # 定义一个用于测试内存格式转换的函数，接受设备、输入生成器函数、转换函数、内存格式等参数
    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):

        # 断言内存格式为通道最后或三维通道最后
        assert memory_format == torch.channels_last or memory_format == torch.channels_last_3d

        # 生成一个通道最后的张量 xc
        xc = input_generator_fn(device)
        # 如果内存格式为通道最后，则对 xc 进行降采样操作
        if memory_format == torch.channels_last:
            xc = xc[..., ::2, ::2]
        else:
            xc = xc[..., ::2, ::2, ::2]

        # 使用 transformation_fn 对 xc 进行转换，并保持当前的内存格式
        clone = transformation_fn(xc, memory_format=torch.preserve_format)
        # 断言 clone 不是连续的
        self.assertFalse(clone.is_contiguous())
        # 断言 clone 在指定的内存格式下是连续的
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        # 断言 xc 不是连续的
        self.assertFalse(xc.is_contiguous())
        # 断言 xc 在指定的内存格式下不是连续的
        self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        # 如果需要比较数据，则断言 xc 与 clone 相等（通过转换到 xc 的设备上）
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        # 重新生成一个通道最后的张量 xc
        xc = input_generator_fn(device)
        # 使用 transformation_fn 对 xc 进行转换，并使用连续的内存格式
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        # 断言 clone 是连续的
        self.assertTrue(clone.is_contiguous())
        # 断言 clone 在指定的内存格式下不是连续的
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        # 如果需要比较数据，则断言 xc 与 clone 相等（通过转换到 xc 的设备上）
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        # 重新生成一个通道最后的张量 xc
        xc = input_generator_fn(device)
        # 使用 transformation_fn 对 xc 进行转换，使用默认的内存格式
        clone = transformation_fn(xc)
        # 如果默认为保持内存格式，则断言 clone 不是连续的，并在指定的内存格式下是连续的
        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            # 否则断言 clone 是连续的，并在指定的内存格式下不是连续的
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        # 如果需要比较数据，则断言 xc 与 clone 相等（通过转换到 xc 的设备上）
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        # 生成一个形状为 (3, 4, 5, 6, 7, 8, 9) 的随机张量 x
        x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
        # 随机对 x 进行 10 次维度置换，并断言置换后的步长与使用保持内存格式的转换后的步长相等
        for _ in range(10):
            permutation = list(range(len(x.shape)))
            random.shuffle(permutation)
            x = x.permute(permutation)
            self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    # 仅在 CPU 上执行的测试装饰器
    @onlyCPU
    # 仅对双精度浮点数类型执行的数据类型装饰器
    @dtypes(torch.double)
    # 测试求和操作的函数，接受设备和数据类型参数
    def test_sum_out(self, device, dtype: torch.dtype) -> None:
        # 生成一个形状为 (100, 100) 的随机张量 x，指定设备和数据类型
        x = torch.rand(100, 100, dtype=dtype, device=device)
        # 对 x 沿着第一维度求和，返回结果为 res1
        res1 = torch.sum(x, 1)
        # 创建一个空的张量 res2，与指定的数据类型和设备
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 对 x 沿着第一维度求和，将结果存入 res2
        torch.sum(x, 1, out=res2)
        # 断言 res1 与 res2 相等
        self.assertEqual(res1, res2)

        # 生成一个形状为 (100, 100, 100) 的随机张量 x，指定设备和数据类型
        x = torch.rand(100, 100, 100, dtype=dtype, device=device)
        # 先沿着第三维度求和，然后沿着第二维度求和，返回结果为 res1
        res1 = x.sum(2).sum(1)
        # 创建一个空的张量 res2，与指定的数据类型和设备
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 对 x 沿着第二和第三维度求和，将结果存入 res2
        torch.sum(x, (2, 1), out=res2)
        # 断言 res1 与 res2 相等
        self.assertEqual(res1, res2)

    # 仅在 CUDA 上执行的测试装饰器
    @onlyCUDA
    # 仅对半精度浮点数和单精度浮点数类型执行的数据类型装饰器
    @dtypes(torch.float16, torch.float32)
    # 定义一个测试函数，用于测试在指定设备和数据类型上的 torch.prod 函数
    def test_prod_gpu(self, device, dtype):
        # 创建一个张量 x，包含数据 [2, 3, 6, 9, 8]，指定设备和数据类型
        x = torch.tensor([2, 3, 6, 9, 8], dtype=dtype, device=device)

        # 检查所有组合情况：fp16 输入 - fp16 输出，fp16 输入 - fp32 输出，
        # fp32 输入 - fp16 输出，fp32 输入 - fp32 输出
        for dtype_output in [torch.float16, torch.float32]:
            # 创建预期的结果张量，数据类型为 dtype_output，包含数据 2592，指定设备
            result_expected = torch.tensor(2592, dtype=dtype_output, device=device)
            
            # 使用 torch.prod 计算张量 x 的乘积，结果张量数据类型为 dtype_output
            output = torch.prod(x, dtype=dtype_output)
            # 断言计算结果与预期结果相等
            self.assertEqual(output, result_expected)

            # 使用张量 x 的实例方法 prod 计算乘积，结果张量数据类型为 dtype_output
            output = x.prod(dtype=dtype_output)
            # 断言计算结果与预期结果相等
            self.assertEqual(output, result_expected)

    # 标记为仅在 CPU 上运行的测试函数，使用指定的设备和数据类型
    @onlyCPU
    @dtypes(torch.float)
    def test_prod(self, device, dtype):
        # 创建一个随机张量 x，形状为 (100, 100)，数据类型为 dtype，指定设备
        x = torch.rand(100, 100, dtype=dtype, device=device)
        
        # 使用 torch.prod 计算张量 x 沿着第一维的乘积
        res1 = torch.prod(x, 1)
        # 创建一个空张量 res2，数据类型为 dtype，指定设备
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 使用 torch.prod 计算张量 x 沿着第一维的乘积，并将结果写入 res2
        torch.prod(x, 1, out=res2)
        # 断言 res1 与 res2 相等
        self.assertEqual(res1, res2)

    # 标记为仅在 CPU 上运行的测试函数，使用指定的设备和数据类型
    @onlyCPU
    @dtypes(torch.float16, torch.bfloat16)
    def test_prod_lowp(self, device, dtype):
        # 创建一个随机张量 x，形状为 (100, 100)，数据类型为 dtype，指定设备
        x = torch.rand(100, 100, dtype=dtype, device=device)
        # 将 x 转换为 float 类型的参考张量 x_ref
        x_ref = x.float()
        
        # 使用 torch.prod 计算张量 x 沿着第一维的乘积
        res1 = torch.prod(x, 1)
        # 使用 torch.prod 计算张量 x_ref 沿着第一维的乘积
        res2 = torch.prod(x_ref, 1)
        # 断言 res1 与 res2（转换为数据类型 dtype）相等
        self.assertEqual(res1, res2.to(dtype=dtype))

        # 使用 torch.prod 计算张量 x 沿着第零维的乘积
        res1 = torch.prod(x, 0)
        # 使用 torch.prod 计算张量 x_ref 沿着第零维的乘积
        res2 = torch.prod(x_ref, 0)
        # 断言 res1 与 res2（转换为数据类型 dtype）相等
        self.assertEqual(res1, res2.to(dtype=dtype))

    # 定义一个测试函数，测试在指定设备上的 torch.prod 函数，使用布尔类型的输入
    def test_prod_bool(self, device):
        # 定义不同的布尔值组合作为输入
        vals = [[True, True], [True, False], [False, False], []]
        for val in vals:
            # 使用 torch.tensor 创建张量，数据类型为 torch.bool，指定设备
            result = torch.prod(torch.tensor(val, device=device), dtype=torch.bool).item()
            # 使用 numpy 计算相同输入的乘积结果，数据类型为 bool
            expect = np.prod(np.array(val), dtype=bool)
            # 断言 torch.prod 的计算结果与 numpy 计算结果相等
            self.assertEqual(result, expect)

            # 使用 torch.prod 计算张量的乘积，不指定数据类型
            result = torch.prod(torch.tensor(val, device=device)).item()
            # 使用 numpy 计算相同输入的乘积结果
            expect = np.prod(np.array(val))
            # 断言 torch.prod 的计算结果与 numpy 计算结果相等
            self.assertEqual(result, expect)

    # 标记为仅在 CPU 上运行的测试函数，测试在混合设备环境中的 torch.max 函数
    @onlyCPU
    def test_max_mixed_devices(self, device):
        # 创建一个在指定设备上的随机张量 a
        a = torch.randn(10, device=device)
        # 如果 CUDA 可用，则创建一个在 GPU 上的随机张量 values 和一个 LongTensor 类型的 indices
        if torch.cuda.is_available():
            values = torch.randn(10).cuda()
            indices = torch.cuda.LongTensor()
            # 断言调用 torch.max 时会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: torch.max(a, 0, out=(values, indices)))
            # 断言调用 torch.amax 时会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: torch.amax(a, 0, out=values))

    # 标记为仅在 CPU 上运行的测试函数，测试在混合设备环境中的 torch.min 函数
    @onlyCPU
    def test_min_mixed_devices(self, device):
        # 创建一个在指定设备上的随机张量 a
        a = torch.randn(10, device=device)
        # 如果 CUDA 可用，则创建一个在 GPU 上的随机张量 values 和一个 LongTensor 类型的 indices
        if torch.cuda.is_available():
            values = torch.randn(10).cuda()
            indices = torch.cuda.LongTensor()
            # 断言调用 torch.min 时会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: torch.min(a, 0, out=(values, indices)))
            # 断言调用 torch.amin 时会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: torch.amin(a, 0, out=values))

    # TODO: 考虑与 bincount 测试重构
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    # 定义一个测试函数，用于测试 torch.nansum 方法
    def test_nansum(self, device, dtype):
        # 构建参数列表，包括是否连续和维度选择
        args = product(
            (True, False),  # noncontiguous，是否非连续
            (0, 1, None),   # dim，选择的维度或者None
        )
        # 在指定设备和数据类型上创建一个空的张量
        zero = torch.zeros((), device=device, dtype=dtype)

        # 遍历参数列表
        for noncontiguous, dim in args:
            # 随机生成一个缩放值
            scale = random.randint(10, 100)
            # 创建一个指定形状的张量，填充随机值，并可能设置为非连续
            x = make_tensor((17, 17), device=device, dtype=dtype,
                            low=-scale, high=scale, noncontiguous=noncontiguous)

            # 如果数据类型是浮点型
            if dtype.is_floating_point:
                # 创建一个NaN掩码
                nan_mask = x < 0.2 * scale
                # 用零值替换NaN值，创建一个没有NaN值的张量
                x_nonan = torch.where(nan_mask, zero, x)
                # 将张量中满足NaN掩码条件的元素设置为NaN
                x[nan_mask] = np.nan
            else:
                # 如果数据类型不是浮点型，则直接使用x作为没有NaN的张量
                x_nonan = x

            # 根据维度参数进行设置
            dim_kwargs = {} if dim is None else {"dim": dim}
            # 计算预期的和，排除NaN值
            expect = torch.sum(x_nonan, **dim_kwargs)
            # 使用torch.nansum计算实际的和
            actual = torch.nansum(x, **dim_kwargs)
            # 断言预期和实际相等
            self.assertEqual(expect, actual)

    # 测试使用numpy函数与torch函数进行比较的函数
    def _test_reduction_function_with_numpy(self, torch_func, np_func, device, dtype,
                                            with_extremal=False, atol=None, rtol=None,
                                            exact_dtype=True, with_keepdim=False):
        # 测试0到3维张量
        for ndims in range(0, 4):
            # 随机生成张量的形状
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for n in range(ndims + 1):
                # 生成所有维度组合的排列
                for c in combinations(list(range(ndims)), n):
                    # 生成所有排列的组合维度
                    for count_dim in permutations(c):
                        # 生成输入张量
                        x = _generate_input(shape, dtype, device, with_extremal)

                        # 如果没有指定count_dim，则默认dims=None
                        if count_dim == ():
                            self.compare_with_numpy(torch_func, np_func, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)
                        else:
                            # 指定了dims参数的情况
                            if with_keepdim:
                                # 创建保持维度的torch和numpy函数
                                torch_func_partial = partial(torch_func, keepdim=True, dim=count_dim)
                                np_func_partial = partial(np_func, keepdims=True, axis=count_dim)
                            else:
                                torch_func_partial = partial(torch_func, dim=count_dim)
                                np_func_partial = partial(np_func, axis=count_dim)
                            self.compare_with_numpy(torch_func_partial, np_func_partial, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)

    # 测试count_nonzero函数的具体实现
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_count_nonzero(self, device, dtype):
        # 使用numpy函数与torch函数进行比较，未指定极端情况下
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype)
        # 使用numpy函数与torch函数进行比较，指定了极端情况下
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype, True)
    # TODO: Investigate why the output is not close to numpy.
    def _get_relaxed_tolerances_for(self, dtype):
        # 根据数据类型返回适当的容差值，用于数值比较
        if dtype == torch.float16:
            atol = 0.4
            rtol = 1e-2
        elif dtype == torch.float32:
            atol = 7e-05
            rtol = 3e-06
        else:
            # 默认情况下不设置容差值
            atol = None
            rtol = None
        return atol, rtol

    def _test_sum_reduction_vs_numpy(self, torch_fn, np_fn, device, dtype, with_keepdim=False, with_extremal=False):
        def is_integral(dtype):
            # 判断给定的数据类型是否为整数类型
            return dtype in integral_types()

        exact_dtype = True
        # 在 Windows 环境下，当前版本的 `numpy` 将所有较低的整数类型提升为 int32，而 `torch` 则提升为 int64。
        # 因此我们在这种情况下跳过对确切数据类型的检查。
        # 参考链接：https://dr.pytorch.org/api/view-log-full?build_id=122051580
        # PR：https://github.com/pytorch/pytorch/pull/38628#issuecomment-655905370
        if IS_WINDOWS and is_integral(dtype):
            exact_dtype = False
        # 对于 uint8 类型，numpy 会提升为 uint64，而 torch 提升为 int64。
        # 因此我们也必须跳过这种情况的检查。
        if dtype == torch.uint8:
            exact_dtype = False

        # TODO: Investigate why the output is not close to numpy.
        # 获取松散容差值，用于数值比较
        atol, rtol = self._get_relaxed_tolerances_for(dtype)
        # 使用 numpy 对比测试 torch 的求和函数
        self._test_reduction_function_with_numpy(torch_fn, np_fn, device, dtype,
                                                 atol=atol, rtol=rtol, exact_dtype=exact_dtype,
                                                 with_keepdim=with_keepdim, with_extremal=with_extremal)

    @onlyNativeDeviceTypes
    @dtypes(*set(all_types_and(torch.half)) - {torch.uint8})
    def test_sum_vs_numpy(self, device, dtype):
        # 测试 torch 的求和函数与 numpy 的对比
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_keepdim=True)

    @onlyNativeDeviceTypes
    @dtypes(*set(all_types_and(torch.half)) - {torch.uint8})
    def test_nansum_vs_numpy(self, device, dtype):
        # 测试 torch 的 nansum 函数与 numpy 的对比
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_keepdim=True)

    @onlyCPU
    @dtypes(*complex_types())
    def test_nansum_complex(self, device, dtype):
        # 对于复数类型，测试 torch 的 nansum 函数是否会引发异常
        x = torch.randn((3, 3, 3), device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "nansum does not support complex inputs"):
            torch.nansum(x)

    @dtypes(*all_types_and(torch.half))
    # 测试函数，用于测试 torch.nansum 的输出数据类型
    def test_nansum_out_dtype(self, device, dtype):
        # 将输出数据类型设为与输入数据类型一致
        out_dtype = dtype
        # 根据输出数据类型是否为浮点数，选择输入数据类型列表
        inp_dtypes = all_types_and(torch.half) if out_dtype.is_floating_point else integral_types()
        # 遍历输入数据类型列表
        for inp_dtype in inp_dtypes:
            # TODO: 调查为什么输出结果与 NumPy 不接近的问题
            # 获取相对宽松的容差值
            atol, rtol = self._get_relaxed_tolerances_for(dtype)
            # 生成随机形状的输入张量
            shape = _rand_shape(random.randint(2, 5), min_size=5, max_size=10)
            x = _generate_input(shape, inp_dtype, device, with_extremal=False)
            # 部分应用 torch.nansum 函数，指定输出数据类型
            torch_fn = partial(torch.nansum, dtype=out_dtype)
            # 将输出数据类型映射为 NumPy 类型
            np_out_dtype = torch_to_numpy_dtype_dict[out_dtype]
            np_fn = partial(np.nansum, dtype=np_out_dtype)
            # 使用自定义函数比较 torch 和 NumPy 的结果
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None, atol=atol, rtol=rtol)

    @dtypes(*all_types_and(torch.half))
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    # TODO: 此测试的一部分涵盖了 torch.norm，应由 test_linalg 来完成
    @onlyNativeDeviceTypes
    def test_repeated_dim(self, device):
        # 定义操作列表
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.norm]
        # 创建指定设备上的随机张量
        x = torch.randn(3, 3, 3, 3, device=device)

        # 预期的运行时错误信息
        error_msg = r'appears multiple times in the list of dims'
        # 遍历操作列表
        for op in ops:
            # 遍历维度元组列表
            for dim in [(0, 0), (0, -4)]:
                # 断言操作在指定维度上会抛出预期的 RuntimeError
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    op(x, dim=dim)

    # TODO: 更新此测试以便与 NumPy 进行比较
    @onlyCUDA
    def test_var(self, device):
        # 创建 CPU 上的随机张量，并转移到指定设备上
        cpu_tensor = torch.randn(2, 3, 3)
        device_tensor = cpu_tensor.to(device)
        # 比较设备上张量与 CPU 上张量的方差和标准差
        self.assertEqual(device_tensor.var(), cpu_tensor.var())
        self.assertEqual(device_tensor.var(1), cpu_tensor.var(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))
        self.assertEqual(device_tensor.std(), cpu_tensor.std())
        self.assertEqual(device_tensor.std(1), cpu_tensor.std(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))

        # 创建大尺寸、不规则输入的 CPU 上的随机张量，并转移到指定设备上
        cpu_tensor = torch.randn(100)
        device_tensor = cpu_tensor.to(device)
        # 比较设备上张量与 CPU 上张量的方差
        self.assertEqual(device_tensor.var(), cpu_tensor.var())

    # TODO: 更新此测试以便与 NumPy 进行比较
    @onlyCUDA
    def test_var_large_input(self, device):
        # 创建大尺寸、不规则输入的 CPU 上的随机张量，并转移到指定设备上
        cpu_tensor = torch.randn(2 * 32 * 1024 + 1, 2, 67)
        device_tensor = cpu_tensor.to(device)

        # 比较设备上张量与 CPU 上张量的指定维度的方差
        self.assertEqual(cpu_tensor.var(2), device_tensor.var(2))

    # TODO: 更新此测试以便与 NumPy 进行比较，而不是与 CPU 进行比较
    @onlyCUDA
    @dtypes(torch.double)
    def test_sum_noncontig(self, device, dtype):
        # 创建指定设备上的随机张量，并按指定维度排列
        x = torch.randn(1, 75, 57, 20, dtype=dtype, device=device).permute(0, 3, 1, 2)
        y = x.cpu()
        # 比较设备上张量与 CPU 上张量的总和和指定维度上的总和
        self.assertEqual(x.sum().cpu(), y.sum())
        self.assertEqual(x.sum(dim=(-1, -2)).cpu(), y.sum(dim=(-1, -2)))
        self.assertEqual(x.sum(dim=(1, 3)).cpu(), y.sum(dim=(1, 3)))
    # TODO: update this to compare against NumPy instead of CPU
    @onlyCUDA
    def test_min_max_nan(self, device):
        # 定义一系列测试函数和对应的名称
        tests = [(lambda x: x.min(), 'min'),
                 (lambda x: x.max(), 'max'),
                 (lambda x: x.amin(), 'amin'),
                 (lambda x: x.amax(), 'amax'),
                 (lambda x: x.min(0).values, 'min_dim'),
                 (lambda x: x.max(0).values, 'max_dim'),
                 (lambda x: x.amin(0), 'amin_dim'),
                 (lambda x: x.amax(0), 'amax_dim')]
        # 对每一个测试进行迭代
        for f, name in tests:
            # 创建一个 5x5 的张量 a，其中一个元素设为 NaN
            a = torch.arange(25.0).view(5, 5)
            a[2, 2] = nan  # 将索引为 (2, 2) 的元素设为 NaN
            # 将张量 a 移动到指定设备，然后应用当前的测试函数 f，并在 CPU 上运行结果
            actual = f(a.to(device)).cpu()
            # 在未移动到指定设备的原始张量 a 上应用测试函数 f，并在 CPU 上运行结果
            expected = f(a).cpu()
            # 断言实际结果中的 NaN 与预期结果中的 NaN 是否一致，用于名称为 name 的测试
            self.assertEqual(torch.isnan(actual), torch.isnan(expected), msg=f'nans for {name}')
            # 断言实际结果中非 NaN 值与预期结果中非 NaN 值是否一致，用于名称为 name 的测试
            self.assertEqual(actual[~torch.isnan(actual)],
                             expected[~torch.isnan(expected)], msg=f'nans for {name}')

    # TODO: make this test generic using OpInfos
    @onlyCUDA
    def test_sum_cpu_device_mismatch(self, device):
        # 创建一个在指定设备上的随机张量 x，以及在 CPU 上的随机张量 y
        x = torch.randn(20, dtype=torch.float32, device=device)
        y = torch.randn(1, dtype=torch.float32)

        # 期望的错误信息字符串，表明输出张量应该在指定设备上，但实际却在 CPU 上
        err_string = f"Expected out tensor to have device {device}, but got cpu instead"

        # 使用断言，预期会抛出 RuntimeError，并且错误信息包含 err_string
        with self.assertRaisesRegex(RuntimeError, err_string):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)

        # 如果设备类型为 'cuda'，则测试从半精度到单精度的类型提升
        if self.device_type == 'cuda':
            x = x.half()  # 将张量 x 转换为半精度
            # 再次使用断言，预期会抛出 RuntimeError，并且错误信息包含 err_string
            with self.assertRaisesRegex(RuntimeError, err_string):
                torch.sum(x, dim=[0], dtype=torch.float32, out=y)

    # Assert for illegal dtype would not be raised on XLA
    @onlyNativeDeviceTypes
    # 测试在非法数据类型情况下的最大值和最小值计算
    def test_minmax_illegal_dtype(self, device):
        # 创建一个大小为 5x5 的张量 x，数据类型为 torch.float32，在指定设备上生成随机数
        x = torch.randn(5, 5, dtype=torch.float32, device=device)
        # 创建一个大小为 5 的空张量 valid_values，数据类型为 torch.float32，在指定设备上
        valid_values = torch.empty(5, dtype=torch.float32, device=device)
        # 创建一个大小为 5 的空张量 valid_indices，数据类型为 torch.long，在指定设备上
        valid_indices = torch.empty(5, dtype=torch.long, device=device)
        # 创建一个大小为 5 的空张量 illegal_values，数据类型为 torch.int，在指定设备上
        illegal_values = torch.empty(5, dtype=torch.int, device=device)
        # 创建一个大小为 5 的空张量 illegal_indices，数据类型为 torch.double，在指定设备上
        illegal_indices = torch.empty(5, dtype=torch.double, device=device)
        # 计算张量 x 沿着第 0 维的最大值，结果分别存储在 valid_values 和 valid_indices 中
        torch.max(x, dim=0, out=(valid_values, valid_indices))
        # 计算张量 x 沿着第 0 维的最小值，结果分别存储在 valid_values 和 valid_indices 中
        torch.min(x, dim=0, out=(valid_values, valid_indices))
        # 计算张量 x 沿着第 0 维的最大值（等效于 torch.max），结果存储在 valid_values 中
        torch.amax(x, dim=0, out=valid_values)
        # 计算张量 x 沿着第 0 维的最小值（等效于 torch.min），结果存储在 valid_values 中
        torch.amin(x, dim=0, out=valid_values)
        # 设置预期的异常消息，用于捕获 RuntimeError 异常
        rmsg = r'scalar type|dtype'
        # 预期 RuntimeError 异常，因为 illegal_values 的数据类型不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, valid_indices))
        # 预期 RuntimeError 异常，因为 illegal_values 的数据类型不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, valid_indices))
        # 预期 RuntimeError 异常，因为 illegal_indices 的数据类型不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(valid_values, illegal_indices))
        # 预期 RuntimeError 异常，因为 illegal_indices 的数据类型不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(valid_values, illegal_indices))
        # 预期 RuntimeError 异常，因为 illegal_values 和 illegal_indices 的数据类型都不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, illegal_indices))
        # 预期 RuntimeError 异常，因为 illegal_values 和 illegal_indices 的数据类型都不符合要求
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, illegal_indices))
    # 定义一个测试函数，用于测试维度缩减函数的行为
    def test_dim_reduction_fns(self, device, dtype, fn_name):
        # 定义一个内部函数，根据函数名获取对应的 torch 函数属性
        def normfn_attr(t, dim, keepdim=False, out=None):
            attr = torch.norm
            return attr(t, 2, dim, keepdim, out=out)

        # 根据给定的函数名获取对应的 torch 函数属性，如果是 'norm' 则使用自定义的 normfn_attr 函数
        fn_attr = getattr(torch, fn_name) if fn_name != "norm" else normfn_attr

        # 定义一个函数 fn，调用前面获取的 torch 函数属性进行计算
        def fn(x, dim, keepdim=False, out=None):
            ans = fn_attr(x, dim, keepdim=keepdim, out=out)
            return ans if not isinstance(ans, tuple) else ans[0]

        # 定义另一个函数 fn_tuple，直接调用前面获取的 torch 函数属性进行计算
        def fn_tuple(x, dim, keepdim=False, out=None):
            return fn_attr(x, dim, keepdim=keepdim, out=out)

        # 定义一个测试多维情况的函数
        def test_multidim(x, dim):
            # 断言不保留维度的计算结果与保留维度的计算结果在指定维度上的形状一致
            self.assertEqual(fn(x, dim).unsqueeze(dim), fn(x, dim, keepdim=True))
            # 断言不保留维度的计算结果维度比原张量维度少 1
            self.assertEqual(x.ndimension() - 1, fn(x, dim).ndimension())
            # 断言保留维度的计算结果维度与原张量维度一致
            self.assertEqual(x.ndimension(), fn(x, dim, keepdim=True).ndimension())

        # 一般情况下的测试
        x = torch.randn(3, 4, 5, device=device)
        dim = random.randint(0, 2)
        test_multidim(x, dim)

        # 检查一维张量的行为
        x = torch.randn(1, device=device)
        dim = 0
        self.assertEqual(fn(x, dim).shape, ())  # 断言计算结果为标量
        self.assertEqual(fn(x, dim, keepdim=True).shape, (1,))  # 断言保留维度后结果为长度为 1 的张量

        # 检查减少单个维度的情况
        dims = [3, 4, 5]
        singleton_dim = random.randint(0, 2)
        dims[singleton_dim] = 1
        x = torch.randn(dims, device=device)
        test_multidim(x, singleton_dim)

        # 检查使用输出关键字参数的减少操作
        if fn_name in ['median', 'nanmedian', 'mode', 'max', 'min']:
            y = torch.randn(5, 3, device=device)
            values = torch.randn(5, 3, device=device)
            indices = torch.zeros(5, 3, device=device).long() - 1
            fn_tuple(y, 1, keepdim=False, out=(values[:, 1], indices[:, 1]))
            values_expected, indices_expected = fn_tuple(y, 1, keepdim=False)
            self.assertEqual(values[:, 1], values_expected,
                             msg=f'{fn_name} values with out= kwarg')  # 断言值与预期输出一致
            self.assertEqual(indices[:, 1], indices_expected,
                             msg=f'{fn_name} indices with out= kwarg')  # 断言索引与预期输出一致
            return

        # 普通情况下的测试
        x = torch.randn(5, 3, device=device)
        y = torch.randn(5, 3, device=device)
        fn(y, 1, keepdim=False, out=x[:, 1])
        expected = fn(y, 1, keepdim=False)
        self.assertEqual(x[:, 1], expected, msg=f'{fn_name} with out= kwarg')  # 断言输出与预期一致

    # 仅在CUDA环境下执行该测试函数
    @onlyCUDA
    # 标记为大张量测试，限制在10GB内存环境下执行
    @largeTensorTest('10GB')
    def test_reduction_split(self, device):
        # 测试当存在32位索引分割时的减少操作
        # https://github.com/pytorch/pytorch/issues/37583
        input_ = torch.randn(5, 14400, 14400, device=device)
        result = input_.sum(dim=0)
        expect = input_[0] + input_[1] + input_[2] + input_[3] + input_[4]
        self.assertEqual(result, expect)  # 断言结果与预期一致

    # 仅在CUDA环境下执行该测试函数
    @onlyCUDA
    # 标记支持的数据类型为半精度浮点、单精度浮点、双精度浮点和BF16
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    @onlyCUDA
    # 标记支持的数据类型为半精度浮点、单精度浮点、双精度浮点和BF16
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    # 定义一个测试函数，用于验证向量化操作的输出
    def test_reduction_vectorize_along_output(self, device, dtype):
        # 定义内部函数，执行测试
        def run_test(input_):
            # 获取输入张量的形状信息 M, N
            M, N = input_.shape
            # 将输入张量置零
            input_.zero_()
            # 针对最小的 M 和 N 中的较小值进行循环
            for i in range(min(M, N)):
                # 设置对角线上的元素为 1
                input_[i][i] = 1
            # 沿着 dim=0 计算最大值索引
            output1 = input_.argmax(dim=0)
            # 沿着 dim=0 计算元素的和
            output2 = input_.sum(dim=0)
            # 再次循环验证输出结果
            for i in range(min(M, N)):
                # 断言：输出1中的值应等于索引 i
                self.assertEqual(output1[i], i)
                # 断言：输出2中的值应等于 1
                self.assertEqual(output2[i], 1)
        # 执行测试：使用零张量作为输入，指定设备和数据类型
        run_test(torch.zeros(64, 64, dtype=dtype, device=device))
        # 执行测试：使用零张量作为输入，但从索引2开始，视图形状为 64x64
        run_test(torch.zeros(64 * 64 + 2, dtype=dtype, device=device)[2:].view(64, 64))
        # 执行测试：使用零张量作为输入，形状为 64x62
        run_test(torch.zeros(64, 62, dtype=dtype, device=device))
        # 执行测试：使用零张量作为输入，形状为 64x2
        run_test(torch.zeros(64, 2, dtype=dtype, device=device))
        # 执行测试：使用零张量作为输入，但从索引1开始，视图形状为 64x64
        run_test(torch.zeros(64 * 64 + 1, dtype=dtype, device=device)[1:].view(64, 64))
        # 执行测试：使用零张量作为输入，形状为 64x61
        run_test(torch.zeros(64, 61, dtype=dtype, device=device))
        # 执行测试：使用零张量作为输入，形状为 64x1
        run_test(torch.zeros(64, 1, dtype=dtype, device=device))

    @onlyCUDA
    # 使用装饰器，标记该测试仅在 CUDA 设备上运行
    def test_argminmax_large_axis(self, device):
        # 回归测试，验证在大轴上的最大最小值索引计算
        x = torch.zeros(2**31, device=device, dtype=torch.int8)
        x[-1] = 1
        # 断言：计算张量 x 在轴 0 上的最大值索引应为 x.shape[0] - 1
        self.assertEqual(x.argmax(0), x.shape[0] - 1)
        # 断言：计算张量 x 在轴 0 上的最大值索引应为 x.shape[0] - 1
        self.assertEqual(x.max(0).indices, x.shape[0] - 1)
        x[-1] = -1
        # 断言：计算张量 x 在轴 0 上的最小值索引应为 x.shape[0] - 1
        self.assertEqual(x.argmin(0), x.shape[0] - 1)
        # 断言：计算张量 x 在轴 0 上的最小值索引应为 x.shape[0] - 1
        self.assertEqual(x.min(0).indices, x.shape[0] - 1)

    # 测试函数，验证在维度为1的轴上计算最大最小值索引
    def test_argminmax_axis_with_dim_one(self, device):
        # 参考链接：https://github.com/pytorch/pytorch/issues/38922
        n = 32768
        # 创建一个形状为 (1, n) 的零张量 x
        x = torch.zeros(1, n)
        # 断言：计算张量 x 在轴 0 上的最大值索引应为形状为 (n,) 的零张量
        self.assertEqual(x.argmax(dim=0), torch.zeros(n, dtype=torch.int64))
        # 断言：计算张量 x 在轴 0 上的最小值索引应为形状为 (n,) 的零张量

        self.assertEqual(x.argmin(dim=0), torch.zeros(n, dtype=torch.int64))

        # 断言：计算张量 x 在轴 -2 上的最大值索引应为形状为 (n,) 的零张量
        self.assertEqual(x.argmax(dim=-2), torch.zeros(n, dtype=torch.int64))
        # 断言：计算张量 x 在轴 -2 上的最小值索引应为形状为 (n,) 的零张量
        self.assertEqual(x.argmin(dim=-2), torch.zeros(n, dtype=torch.int64))

        # 断言：计算张量 x 在轴 0 上的最大值索引应为形状为 (1, n) 的零张量
        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        # 断言：计算张量 x 在轴 0 上的最小值索引应为形状为 (1, n) 的零张量
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

        # 断言：计算张量 x 在轴 -2 上的最大值索引应为形状为 (1, n) 的零张量
        self.assertEqual(x.argmax(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        # 断言：计算张量 x 在轴 -2 上的最小值索引应为形状为 (1, n) 的零张量
        self.assertEqual(x.argmin(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

    # 使用装饰器，标记此测试函数对多种数据类型有效
    @dtypes(torch.int, torch.long, torch.float, torch.double)
    # 使用装饰器，标记此测试函数在 CUDA 设备上支持的数据类型
    @dtypesIfCUDA(torch.int, torch.long, torch.half, torch.float, torch.double)
    # 定义一个测试函数，用于测试在实际数值上计算中位数的功能
    def test_median_real_values(self, device, dtype):
        # 生成随机的0-3维大小的列表
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        # 遍历各个大小
        for size in sizes:
            # 创建随机输入张量
            t = torch.randn(size, device=device).type(dtype)
            # 将张量转换为 numpy 数组
            t_numpy = t.cpu().numpy()
            # 计算张量的中位数
            res = t.median()
            # 断言张量的中位数与 NaN 值的中位数相等
            self.assertEqual(res, t.nanmedian())
            # 计算中位数的索引
            k = int((t.numel() - 1) / 2)
            # 断言张量的中位数与排序后的第 k 个元素相等
            self.assertEqual(res, t.view(-1).sort()[0][k])
            if t.numel() % 2 == 1:
                # 对于奇数个元素的张量，可以使用 numpy 进行验证，
                # 因为 numpy 返回两个中位数的均值，而 torch 返回较小的那个
                self.assertEqual(res.cpu().numpy(), np.median(t_numpy))
            # 遍历张量的各个维度
            for dim in range(t.ndim):
                # 计算指定维度上的中位数
                res = t.median(dim, True)
                # 断言张量在指定维度上的中位数与 NaN 值的中位数相等
                self.assertEqual(res, t.nanmedian(dim, True))
                # 获取指定维度的大小
                size = t.size(dim) if t.ndim > 0 else 1
                # 计算中位数的索引
                k = int((size - 1) / 2)
                # 断言张量在指定维度上排序后的第 k 个元素与计算得到的中位数相等
                self.assertEqual(res[0], (t.sort(dim)[0]).select(dim, k).unsqueeze_(dim))
                # 断言张量在指定维度上的 gather 操作结果与中位数索引的值相等
                self.assertEqual(res[0], t.gather(dim, res[1]))
                if size % 2 == 1:
                    # 对于奇数个元素的张量，可以使用 numpy 进行验证，
                    # 因为 numpy 返回两个中位数的均值，而 torch 返回较小的那个
                    self.assertEqual(res[0].cpu().numpy(), np.median(t_numpy, dim, keepdims=True), exact_dtype=False)
    def test_median_nan_values(self, device, dtype):
        # 生成随机的0到3维大小的列表
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        for size in sizes:
            # 创建具有NaN值的随机输入张量
            t = torch.rand(size, device=device, dtype=dtype)
            t.masked_fill_(t < 0.1, float('nan'))
            t_numpy = t.cpu().numpy()
            for op in [torch.median, torch.nanmedian]:
                numpy_op = np.median if op == torch.median else np.nanmedian
                res = op(t)
                num_nan = t.isnan().sum()
                if op == torch.median and num_nan > 0:
                    k = t.numel() - 1
                else:
                    k = int((t.numel() - num_nan - 1) / 2)
                # 断言torch中的计算结果与排序后的张量中第k个值相等
                self.assertEqual(res, t.view(-1).sort()[0][k])
                if (t.numel() - num_nan) % 2 == 1:
                    # 仅当张量大小为奇数时，才能与numpy进行比较，因为numpy返回两个中位数的均值，而torch返回较小的值
                    self.assertEqual(res.item(), numpy_op(t.cpu().numpy()))
                for dim in range(t.ndim):
                    res = op(t, dim, True)
                    size = t.size(dim) if t.ndim > 0 else 1
                    num_nan = t.isnan().sum(dim, True)
                    if op == torch.median:
                        k = torch.where(num_nan > 0, size - 1, int((size - 1) / 2))
                    else:
                        k = ((size - num_nan - 1) / 2).type(torch.long)
                    # 断言torch中的计算结果与排序后的张量中第k个值相等
                    self.assertEqual(res[0], (t.sort(dim)[0]).gather(dim, k))
                    self.assertEqual(res[0], t.gather(dim, res[1]))
                    # 仅当张量大小为奇数时，才能与numpy进行比较，因为numpy返回两个中位数的均值，而torch返回较小的值
                    mask = (size - num_nan) % 2 == 1
                    res = res[0].masked_select(mask).cpu()
                    ref = numpy_op(t_numpy, dim, keepdims=True)[mask.cpu().numpy()]
                    self.assertEqual(res, torch.from_numpy(ref))
    # 定义测试函数 test_median_corner_cases，接受参数 self 和 device
    def test_median_corner_cases(self, device):
        # 定义内部函数 check，接受 op（操作函数）、a（输入数据）、args（额外参数）、key（期望结果）
        def check(op, a, args, key):
            # 使用 torch.tensor 将 a 转换为张量 t，并指定设备为 device
            t = torch.tensor(a, device=device)
            # 使用操作函数 op 对张量 t 和 args 执行操作，将结果存储在 res 中
            res = op(t, *args)
            # 根据条件判断是否需要将 key 转换为张量
            if not args:
                key = torch.tensor(key, device=device)
            else:
                if len(key) == 1:
                    key = torch.tensor(key[0], device=device)
                    res = res[0]
                else:
                    key = (torch.tensor(key[0], device=device), torch.tensor(key[1], device=device))
            # 使用 self.assertEqual 断言 res 和 key 的值相等
            self.assertEqual(res, key)

        # 定义 NaN 值
        nan = float('nan')
        
        # 执行一系列测试用例，验证对 nan 值的处理
        check(torch.median, nan, [], nan)
        check(torch.median, [], [], nan)
        check(torch.nanmedian, nan, [], nan)
        check(torch.median, nan, [0], [nan, 0])
        check(torch.nanmedian, nan, [0], [nan, 0])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])

        # 验证二维张量的中位数计算
        # 注意：由于索引结果不确定，此处仅检查值是否正确
        check(torch.median, [[nan, nan], [1, 2]], [0], [[nan, nan]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [0], [[1, 2.]])
        check(torch.median, [[nan, nan], [1, 2]], [1], [[nan, 1]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [1], [[nan, 1.]])

        # 验证不连续和步幅张量的中位数计算
        a = torch.arange(12, device=device)
        self.assertEqual(a[::2].median(), torch.tensor(4, device=device))
        self.assertEqual(a[::2].nanmedian(), torch.tensor(4, device=device))

        # 重新调整张量形状为 3x4
        a.resize_(3, 4)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[::2, ::2].median(-1)[0], torch.tensor([0, 8], device=device))
        self.assertEqual(a[::2, ::2].nanmedian(-1)[0], torch.tensor([0, 8], device=device))

        # 重新调整张量形状为 2x3x2
        a.resize_(2, 3, 2)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[:, ::2, :].median(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))
        self.assertEqual(a[:, ::2, :].nanmedian(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))
    def test_quantile(self, device, dtype):
        # Generate some random test cases
        ops = ['quantile', 'nanquantile']  # 定义操作列表，包括 quantile 和 nanquantile
        inputs = [tuple(np.random.randint(2, 10, size=i)) for i in range(1, 4)]  # 生成不同长度的随机元组作为输入
        quantiles = [tuple(np.random.rand(i)) for i in range(0, 5)]  # 生成不同长度的随机元组作为分位数
        keepdims = [True, False]  # 保留维度或不保留维度的选项

        # Add corner cases
        inputs.extend([0.75, (1,), (1, 1), (1, 2, 1)])  # 添加特殊情况到输入列表
        inputs.extend([[float('nan')], [[float('nan'), float('nan')], [1, 2]]])  # 添加包含 NaN 的输入列表
        inputs.extend([[[float('nan'), float('nan')], [float('nan'), 2]]])  # 添加多维包含 NaN 的输入列表
        quantiles.extend([0.5, [0., 1.], np.random.rand(10)])  # 添加额外的分位数作为特殊情况

        # Enumerate all input combinations
        for op, x, q, keepdim in product(ops, inputs, quantiles, keepdims):
            if type(x) is tuple:
                a = torch.randn(x, dtype=dtype, device=device)
                # Make some random elements NaN
                a.masked_fill_(torch.randint_like(a, 20) == 0, float('nan'))  # 将张量中的一些随机元素设置为 NaN
            else:
                a = torch.tensor(x, dtype=dtype, device=device)

            q = torch.tensor(q, dtype=dtype, device=device)  # 创建分位数的张量

            torch_op = getattr(torch, op)  # 获取 torch 模块中对应的操作函数
            numpy_op = getattr(np, op)  # 获取 numpy 模块中对应的操作函数

            # Compute quantile along every dimension and flattened tensor
            interpolations = ('linear', 'lower', 'higher', 'midpoint', 'nearest')
            for interpolation, dim in product(interpolations,
                                              [None] + list(range(a.ndim))):
                result = torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation)
                expected = numpy_op(a.cpu().numpy(), q.cpu().numpy(), dim,
                                    interpolation=interpolation, keepdims=keepdim)
                self.assertEqual(result.cpu(), torch.from_numpy(np.array(expected)).type(result.type()))
                # 断言计算结果与预期结果相等

                # Test out variation
                out = torch.empty_like(result)
                torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
                self.assertEqual(out.cpu(), result.cpu())
                # 测试使用指定输出张量的情况

    def test_quantile_backward(self, device):
        def check(a, q, dim, expected_grad, ops=(torch.quantile, torch.nanquantile)):
            for op in ops:
                t = torch.tensor(a, device=device, requires_grad=True)
                op(t, torch.tensor(q, device=device), dim).sum().backward()
                self.assertEqual(t.grad, expected_grad)
                # 断言计算的梯度与预期梯度相等

        check([1., 2, 3], 0.5, 0, [0, 1, 0])  # 检查简单的张量输入的梯度计算
        check([1., 2, 3, 4], 0.5, 0, [0, 0.5, 0.5, 0])  # 检查更长张量的梯度计算
        check([3., 1, 4, 2], 0.5, 0, [0.5, 0, 0, 0.5])  # 检查不同顺序的张量的梯度计算
        check([1., 2, 3, 4], [0.25, 0.5, 0.75], 0, [0.25, 1.25, 1.25, 0.25])  # 检查多个分位数的梯度计算
        check([[1., 2], [2, 1]], 0., 0, [[1, 0], [0, 1]])  # 检查二维张量沿第一维度的梯度计算
        check([[1., 2], [4, 3]], 1., 1, [[0, 1], [1, 0]])  # 检查二维张量沿第二维度的梯度计算
        check([1, float('nan'), 2], 0.5, 0, [0, 1, 0], [torch.quantile])  # 检查包含 NaN 的张量的梯度计算
        check([1, float('nan'), 2], 0.5, 0, [0.5, 0, 0.5], [torch.nanquantile])  # 检查使用 nanquantile 的梯度计算
    # 定义一个测试函数，用于测试 torch.quantile 函数的异常情况
    def test_quantile_error(self, device):
        # 定义内部函数 check，用于检查函数调用时是否会引发特定的 RuntimeError 异常，并验证异常消息
        def check(a, q, args, kwargs, message):
            # 使用 assertRaisesRegex 断言语句来验证是否抛出指定异常，并检查异常消息是否包含特定信息
            with self.assertRaisesRegex(RuntimeError, r'quantile\(\) ' + message):
                # 将输入列表 a 转换为 tensor 对象 at，并指定设备 device
                at = torch.tensor(a, device=device)
                # 如果 q 是列表，则将其转换为 tensor 对象 qt，否则保持原样
                qt = torch.tensor(q, device=device) if isinstance(q, list) else q
                # 调用 torch.quantile 函数，检查是否会抛出期望的异常
                torch.quantile(at, qt, *args, **kwargs)

        # 检查空列表 a 的情况是否引发 'input tensor must be non-empty' 异常
        check([], 0.5, [], {}, r'input tensor must be non-empty')
        # 检查 q 是二维列表的情况是否引发 'q must be a scalar or 1D tensor' 异常
        check([1.], [[1.]], [], {}, r'q must be a scalar or 1D tensor')
        # 检查输入列表 a 是整数的情况是否引发 'input tensor must be either float or double dtype' 异常
        check([1], 0.5, [], {}, r'input tensor must be either float or double dtype')
        # 检查 q 和输入列表 a 的数据类型不匹配是否引发 'q tensor must be same dtype as the input tensor' 异常
        check([1.], [1], [], {}, r'q tensor must be same dtype as the input tensor')
        # 检查 q 超出范围 (小于 0) 是否引发 'q must be in the range [0, 1] but got -1' 异常
        check([1.], -1., [], {}, r'q must be in the range \[0, 1\] but got -1')
        # 检查 q 超出范围 (大于 1) 是否引发 'q must be in the range [0, 1] but got 1.1' 异常
        check([1.], 1.1, [], {}, r'q must be in the range \[0, 1\] but got 1.1')
        # 检查输出张量的数据类型与输入张量不匹配是否引发 'out tensor must be same dtype as the input tensor' 异常
        check([1.], 0.5, [], {'out': torch.empty([], dtype=torch.int32, device=device)},
              r'out tensor must be same dtype as the input tensor')
        # 检查插值参数不合法的情况是否引发 'interpolation must be one of linear, lower, higher, midpoint or nearest, but got random_mode' 异常
        check([1.], [1.], [None, False], {'interpolation': 'random_mode'},
              r"interpolation must be one of linear, lower, higher, midpoint or nearest, but got random_mode")

        # 如果设备类型为 CPU，检查 q 值超出范围是否引发 'q values must be in the range [0, 1]' 异常
        if self.device_type == "cpu":
            check([1.], [0.5, 1.1, -1], [], {}, r'q values must be in the range \[0, 1\]')

        # 如果设备类型为 CUDA，检查 q 和输出张量是否与输入张量在同一设备上，验证是否引发相应异常
        if self.device_type == "cuda":
            # 检查 q 张量不在同一设备上是否引发 'quantile() q tensor must be on the same device as the input tensor' 异常
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) q tensor must be on the same device as the input tensor'):
                torch.randn(1, device=device).quantile(torch.tensor(0.5))
            # 检查输出张量不在同一设备上是否引发 'quantile() out tensor must be on the same device as the input tensor' 异常
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) out tensor must be on the same device as the input tensor'):
                torch.quantile(torch.randn(1, device=device), 0.5, out=torch.scalar_tensor(1))

    # 定义一个测试函数，用于验证 torch.std_mean 函数在不同维度和参数配置下的结果是否与直接调用 std 和 mean 函数的结果一致
    def test_std_mean(self, device):
        # 创建一个随机张量 x，维度为 (100, 50, 20)，指定设备 device
        x = torch.rand(100, 50, 20, device=device)
        # 遍历张量的每个维度 dim
        for dim in range(x.dim()):
            # 遍历 unbiased 和 keepdim 参数的所有组合
            for unbiased in [False, True]:
                for keepdim in [False, True]:
                    # 调用 torch.std_mean 函数计算标准差和均值 std1, mean1
                    std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    # 调用 x.std 和 x.mean 函数计算标准差和均值 std2, mean2
                    std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    # 使用断言语句验证 torch.std_mean 的结果与直接调用 std 和 mean 函数的结果一致
                    self.assertEqual(std1, std2)
                    self.assertEqual(mean1, mean2)

    # 定义一个测试函数，用于验证 torch.std_mean 函数在所有维度和参数配置下的结果是否与直接调用 std 和 mean 函数的结果一致
    def test_std_mean_all_dims(self, device):
        # 创建一个随机张量 x，维度为 (100, 50, 20)，指定设备 device
        x = torch.rand(100, 50, 20, device=device)
        # 遍历 unbiased 参数的所有组合
        for unbiased in [False, True]:
            # 调用 torch.std_mean 函数计算标准差和均值 std1, mean1
            std1, mean1 = torch.std_mean(x, unbiased=unbiased)
            # 调用 x.std 和 x.mean 函数计算标准差和均值 std2, mean2
            std2 = x.std(unbiased=unbiased)
            mean2 = x.mean()
            # 使用断言语句验证 torch.std_mean 的结果与直接调用 std 和 mean 函数的结果一致
            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)
    # 定义测试函数，用于测试 torch.var_mean 方法的正确性
    def test_var_mean(self, device):
        # 创建一个随机张量 x，形状为 (100, 300, 50)，在指定设备上
        x = torch.rand(100, 300, 50, device=device)
        # 遍历张量 x 的维度数量
        for dim in range(x.dim()):
            # 遍历 unbiased 参数的两种取值：False 和 True
            for unbiased in [False, True]:
                # 遍历 keepdim 参数的两种取值：False 和 True
                for keepdim in [False, True]:
                    # 调用 torch.var_mean 方法计算指定维度上的方差和均值
                    var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    # 使用张量 x 的方法分别计算指定维度上的方差和均值
                    var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    # 断言两种方法计算得到的方差和均值是否相等
                    self.assertEqual(var1, var2)
                    self.assertEqual(mean1, mean2)

    # 定义测试函数，用于测试 torch.var_mean 方法在全维度计算时的正确性
    def test_var_mean_all_dims(self, device):
        # 创建一个随机张量 x，形状为 (100, 50, 20)，在指定设备上
        x = torch.rand(100, 50, 20, device=device)
        # 遍历 unbiased 参数的两种取值：False 和 True
        for unbiased in [False, True]:
            # 调用 torch.var_mean 方法计算张量 x 在全维度上的方差和均值
            var1, mean1 = torch.var_mean(x, unbiased=unbiased)
            # 使用张量 x 的方法分别计算全维度上的方差和均值
            var2 = x.var(unbiased=unbiased)
            mean2 = x.mean()
            # 断言两种方法计算得到的方差和均值是否相等
            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    # 定义测试函数，用于测试 torch.std_mean 方法在部分维度计算时的正确性
    def test_std_mean_some_dims(self, device):
        # 定义张量 x 的形状
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)
        # 创建一个随机张量 x，形状由 sizes 定义，放置在指定设备上
        x = torch.rand(sizes, device=device)
        # 遍历 num_of_dims，从 2 开始到 dims-1
        for num_of_dims in range(2, dims):
            # 生成 num_of_dims 维度的组合列表
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            # 遍历 dim_list 中的每个维度组合
            for dim in dim_list:
                # 遍历 unbiased 参数的两种取值：False 和 True
                for unbiased in [False, True]:
                    # 遍历 keepdim 参数的两种取值：False 和 True
                    for keepdim in [False, True]:
                        # 调用 torch.std_mean 方法计算指定维度上的标准差和均值
                        std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        # 使用张量 x 的方法分别计算指定维度上的标准差和均值
                        std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        # 断言两种方法计算得到的标准差和均值是否相等
                        self.assertEqual(std1, std2)
                        self.assertEqual(mean1, mean2)
    # 定义一个方法，用于比较标准差和方差计算的结果与对应的 NumPy 计算结果是否一致
    def _compare_std_var_with_numpy(self, op, device, dtype, input, dim,
                                    keepdim, unbiased, use_out):
        # 将输入张量转换为 NumPy 数组，根据数据类型可能需要先将其转换为 float
        a = input.cpu().numpy() if input.dtype is not torch.bfloat16 else input.float().cpu().numpy()
        
        # 定义传递给 NumPy 函数的关键字参数
        numpy_kwargs = {
            'axis': dim,
            'keepdims': keepdim,
            'ddof': 1 if unbiased else 0,
        }

        # 如果维度 dim 为 None，则从参数字典中删除 'axis' 和 'keepdims' 键
        if dim is None:
            del numpy_kwargs['axis']
            del numpy_kwargs['keepdims']

        # 根据操作类型选择对应的 Torch 和 NumPy 函数
        if op == 'var':
            torch_op = torch.var
            numpy_op = np.var
        elif op == 'std':
            torch_op = torch.std
            numpy_op = np.std
        else:
            self.fail("Unknown op!")

        # 使用 NumPy 函数计算结果
        numpy_result = numpy_op(a, **numpy_kwargs)

        # 根据输入的不同条件调用 Torch 函数计算结果
        if dim is None and use_out is False:
            torch_result = torch_op(input, unbiased)
        elif dim is not None and use_out is False:
            torch_result = torch_op(input, dim, unbiased, keepdim)
        elif dim is not None and use_out is True:
            out = torch.empty(0, device=device, dtype=dtype)
            torch_result = torch_op(input, dim, unbiased, keepdim, out=out)
        else:
            out = torch.empty(0, device=device, dtype=dtype)
            torch_result = torch_op(input, dim, unbiased, keepdim, out=out)

        # 检查 Torch 计算结果与 NumPy 计算结果是否相等，根据输入数据类型进行精确匹配
        exact_dtype = input.dtype not in (torch.bfloat16, torch.complex32, torch.complex64, torch.complex128)
        self.assertEqual(torch_result, numpy_result, exact_dtype=exact_dtype)

    # 使用不同的数据类型进行测试，比较 Torch 计算的方差和 NumPy 计算的方差是否一致
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_var_vs_numpy(self, device, dtype):
        _size = (20, 20)

        # 对于每个测试用例，调用 _compare_std_var_with_numpy 方法进行方差比较
        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('var', device, dtype, *test_case)

    # 使用不同的数据类型进行测试，比较 Torch 计算的标准差和 NumPy 计算的标准差是否一致
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_std_vs_numpy(self, device, dtype):
        _size = (20, 20)

        # 对于每个测试用例，调用 _compare_std_var_with_numpy 方法进行标准差比较
        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('std', device, dtype, *test_case)

    # 使用不同的数据类型进行测试，比较 Torch 计算的标准差和方差与 NumPy 计算的结果是否一致
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 测试变量修正与 NumPy 的对比
    def test_var_correction_vs_numpy(self, device, dtype):
        _size = (20, 20)  # 定义 tensor 的大小为 (20, 20)
        test_args = [
            *product(
                # dim 维度参数的选择：None、0、1
                (None, 0, 1),
                # correction 修正参数的选择：None、0、10、30
                (None, 0, 10, 30),
                # keepdim 是否保持维度参数的选择：False、True
                (False, True),
            ),
            [None, -100, True],  # 负修正参数的特殊情况
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)  # 创建指定设备和数据类型的 tensor
        array = tensor.cpu().numpy()  # 将 tensor 转换为 NumPy 数组

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # 当修正参数为 None 时，使用 NumPy 默认值 1，以兼容 torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.var(array, **numpy_kwargs))  # 计算 NumPy 数组的方差
            torch_res = torch.var(tensor, dim=dim, correction=correction, keepdim=keepdim)  # 计算 torch tensor 的方差

            # 处理 inf 和 nan 的结果，这些结果对机器精度很敏感，视为等价处理
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)  # 断言 torch 计算结果与 NumPy 计算结果相等

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 测试标准差修正与 NumPy 的对比
    def test_std_correction_vs_numpy(self, device, dtype):
        _size = (20, 20)  # 定义 tensor 的大小为 (20, 20)
        test_args = [
            *product(
                # dim 维度参数的选择：None、0、1
                (None, 0, 1),
                # correction 修正参数的选择：None、0、10、30
                (None, 0, 10, 30),
                # keepdim 是否保持维度参数的选择：False、True
                (False, True),
            ),
            [None, -100, True],  # 负修正参数的特殊情况
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)  # 创建指定设备和数据类型的 tensor
        array = tensor.cpu().numpy()  # 将 tensor 转换为 NumPy 数组

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # 当修正参数为 None 时，使用 NumPy 默认值 1，以兼容 torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.std(array, **numpy_kwargs))  # 计算 NumPy 数组的标准差
            torch_res = torch.std(tensor, dim=dim, correction=correction, keepdim=keepdim)  # 计算 torch tensor 的标准差

            # 处理 inf 和 nan 的结果，这些结果对机器精度很敏感，视为等价处理
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)  # 断言 torch 计算结果与 NumPy 计算结果相等

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义一个测试函数，用于测试标准差和均值校正功能
    def test_std_mean_correction(self, device, dtype):
        # 定义一个尺寸为 (20, 20) 的张量
        _size = (20, 20)
        # 定义测试参数列表，包括三个维度：dim、correction、keepdim，以及一个负校正的示例
        test_args = [
            *product(
                # dim 参数的取值：None、0、1
                (None, 0, 1),
                # correction 参数的取值：None、0、10、30
                (None, 0, 10, 30),
                # keepdim 参数的取值：False、True
                (False, True),
            ),
            [None, -100, True],  # 负校正的情况
        ]

        # 根据设备和数据类型生成一个指定尺寸的张量
        tensor = make_tensor(_size, device=device, dtype=dtype)

        # 遍历测试参数
        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            # 使用 torch.std 计算张量的标准差
            std1 = torch.std(tensor, **kwargs)
            # 如果 dim 不为 None，则使用 torch.mean 计算指定维度上的均值
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                # 否则，计算整体的均值，并根据 keepdim 决定是否保持维度
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            # 使用自定义函数 torch.std_mean 同时计算标准差和均值
            std2, mean2 = torch.std_mean(tensor, **kwargs)

            # 断言两种方法计算的标准差和均值是否一致
            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)

    # 使用装饰器定义测试函数，测试方差和均值校正功能
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_var_mean_correction(self, device, dtype):
        # 定义一个尺寸为 (20, 20) 的张量
        _size = (20, 20)
        # 定义测试参数列表，包括三个维度：dim、correction、keepdim，以及一个负校正的示例
        test_args = [
            *product(
                # dim 参数的取值：None、0、1
                (None, 0, 1),
                # correction 参数的取值：None、0、10、30
                (None, 0, 10, 30),
                # keepdim 参数的取值：False、True
                (False, True),
            ),
            [None, -100, True],  # 负校正的情况
        ]

        # 根据设备和数据类型生成一个指定尺寸的张量
        tensor = make_tensor(_size, device=device, dtype=dtype)

        # 遍历测试参数
        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            # 使用 torch.var 计算张量的方差
            var1 = torch.var(tensor, **kwargs)
            # 如果 dim 不为 None，则使用 torch.mean 计算指定维度上的均值
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                # 否则，计算整体的均值，并根据 keepdim 决定是否保持维度
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            # 使用自定义函数 torch.var_mean 同时计算方差和均值
            var2, mean2 = torch.var_mean(tensor, **kwargs)

            # 断言两种方法计算的方差和均值是否一致
            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    # 定义一个测试函数，用于测试当自由度参数无效时是否能够正确发出警告
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_warn_invalid_degrees_of_freedom(self, device, dtype):
        # 定义一个内部函数，用于检查是否正确发出警告
        def _assert_warning(_func, _tensor, _correction):
            # 捕获警告消息
            with warnings.catch_warnings(record=True) as w:
                _func(_tensor, dim=-1, correction=_correction)
            # 断言警告消息中包含特定文本
            self.assertIn('degrees of freedom is <= 0', str(w[0].message))

        # 定义校正值为 20 的尺寸
        correction = 20
        size = (10, correction)
        # 根据设备和数据类型生成一个指定尺寸的张量
        tensor = make_tensor(size, dtype=dtype, device=device)
        # 针对 torch.std、torch.var、torch.var_mean、torch.std_mean 四个函数分别测试警告
        for f in [torch.std, torch.var, torch.var_mean, torch.std_mean]:
            _assert_warning(f, tensor, correction)
    # 定义一个测试函数，测试 torch.amin 和 torch.amax 函数在指定维度上的计算结果
    def test_amin_amax_some_dims(self, device):
        # 定义一个张量的大小
        sizes = (4, 6, 7, 5, 3)
        # 获取张量的维度
        dims = len(sizes)
        # 在指定设备上生成一个随机张量
        x = torch.rand(sizes, device=device)
        # 遍历从2到dims之间的维度数
        for num_of_dims in range(2, dims):
            # 生成所有可能的维度组合
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            # 遍历每个维度组合
            for dim in dim_list:
                # 遍历是否保持维度的选项
                for keepdim in [False, True]:
                    # 使用指定维度和保持维度的选项计算最小值和最大值
                    amin1 = torch.amin(x, dim=dim, keepdim=keepdim)
                    amax1 = torch.amax(x, dim=dim, keepdim=keepdim)
                    # 初始化另一种计算方式的最小值和最大值
                    amin2 = x
                    amax2 = x
                    # 遍历每个维度
                    for i, d in enumerate(dim):
                        # 如果不保持维度，则调整维度值
                        if not keepdim:
                            d -= i
                        # 使用指定维度和保持维度的选项计算最小值和最大值
                        amin2 = torch.amin(amin2, dim=d, keepdim=keepdim)
                        amax2 = torch.amax(amax2, dim=d, keepdim=keepdim)
                    # 断言两种计算方式得到的结果是否相等
                    self.assertEqual(amin1, amin2)
                    self.assertEqual(amax1, amax2)

    # 仅在 CPU 上运行的测试函数，测试 torch.histc 函数在低精度下的计算结果
    @onlyCPU
    @dtypes(torch.bfloat16, torch.half)
    def test_histc_lowp(self, device, dtype):
        # 使用指定参数运行 torch.histc 函数
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=dtype, device=device), bins=4, min=0, max=3)
        # 断言计算结果是否与预期结果相等
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=dtype, device=device),
            actual)
        # 断言计算结果的数据类型是否正确
        self.assertEqual(actual.dtype, dtype)

    """
    运行 torch.histogram 和 numpy.histogram 函数，并断言它们的输出结果是否相等。
    """

    # 仅在 CPU 上运行的测试函数，测试 torch.histogramdd 函数在指定精度下的计算结果
    @onlyCPU
    @dtypes(torch.float32)
    """
    运行 torch.histogramdd 和 numpy.histogramdd 函数，并断言它们的输出结果是否相等。
    """
    # 定义一个测试函数，用于测试 torch.histogramdd 的功能
    def _test_histogramdd_numpy(self, t, bins, bin_range, weights, density):
        # 定义一个函数，将 torch 张量转换为 numpy 数组
        def to_np(t):
            # 如果 t 是列表，则递归地将列表中的每个元素转换为 numpy 数组
            if type(t) == list:
                return list(map(to_np, t))
            # 如果 t 不是 torch 张量，则直接返回 t
            if not torch.is_tensor(t):
                return t
            # 如果 t 是 torch 张量，则先将其移到 CPU 上，再转换为 numpy 数组
            return t.cpu().numpy()

        # 封装了 numpy.histogramdd 的函数，实现了 torch 张量与 numpy 数组之间的转换
        def reference_histogramdd(t, bins, bin_range, weights, density, dtype):
            # 将输入的 torch 张量 t、bins、weights 转换为 numpy 数组
            (np_t, np_bins, np_weights) = map(to_np, [t, bins, weights])

            # numpy.histogramdd 只接受 (N, D) 形状的输入
            D = np_t.shape[-1]
            N = np.prod(np_t.shape[:-1])
            reshaped_t = np.reshape(np_t, (N, D))
            reshaped_wt = np.reshape(np_weights, (N,)) if np_weights is not None else None

            # 当 D=0 时，numpy.histogramdd 抛出错误
            if D == 0:
                return (torch.tensor(float('nan') if density else 0.), [])

            # numpy.histogramdd 需要将范围以 (lower, upper) 元组的形式传递给 range 参数
            reshaped_range = None if not bin_range else [(bin_range[2 * i], bin_range[2 * i + 1]) for i in range(D)]

            # 调用 numpy.histogramdd 函数进行直方图计算
            (np_hist, np_bin_edges) = np.histogramdd(reshaped_t, np_bins,
                                                     range=reshaped_range, weights=reshaped_wt, density=density)

            # 将 numpy 计算得到的直方图 np_hist 转换为 torch 张量，并设置数据类型为 dtype
            return (torch.from_numpy(np_hist).to(dtype), [torch.from_numpy(t).to(dtype) for t in np_bin_edges])

        # 调用 torch.histogramdd 计算实际的直方图和边界
        (actual_hist, actual_bin_edges) = torch.histogramdd(t, bins, range=bin_range, weight=weights, density=density)
        # 调用封装的 reference_histogramdd 函数计算预期的直方图和边界
        (expected_hist, expected_bin_edges) = reference_histogramdd(t, bins, bin_range, weights, density, actual_hist.dtype)

        # 获取维度 D
        D = len(actual_bin_edges)
        # 断言实际的直方图和预期的直方图维度相同
        self.assertEqual(D, len(expected_bin_edges))

        """
        Works around linspace discrepancies by passing torch's constructed bin_edges to numpy.
        When bin edges are not explicitly defined, histogram uses the linspace operator internally
        to construct the sequence of bin edges. In some cases, torch.linspace output differs slightly
        from numpy.linspace output.
        Issue: https://github.com/pytorch/pytorch/issues/58758
        """
        # 如果 bins 不是 torch 张量，则进行一些处理以确保一致性
        if not torch.is_tensor(bins):
            # 对每个维度进行断言，以检查实际的边界和预期的边界是否接近
            for dim in range(D):
                self.assertEqual(actual_bin_edges[dim], expected_bin_edges[dim], atol=1e-5, rtol=1e-5)
            # 再次调用 reference_histogramdd 函数，这次将实际的边界作为 bins 参数传递
            (expected_hist, expected_bin_edges) = reference_histogramdd(
                t, actual_bin_edges, bin_range, weights, density, actual_hist.dtype)
            # 断言维度仍然相同
            self.assertEqual(D, len(expected_bin_edges))

        # 断言实际的直方图和预期的直方图相等
        self.assertEqual(actual_hist, expected_hist)
        # 对每个维度再次断言实际的边界和预期的边界相等
        for dim in range(D):
            self.assertEqual(actual_bin_edges[dim], expected_bin_edges[dim])

    # 添加装饰器，只在 CPU 上运行测试
    @onlyCPU
    # 添加装饰器，指定输入张量的数据类型为 torch.float32
    @dtypes(torch.float32)
    # 定义一个测试方法，用于测试 histogramdd 函数在不同条件下的行为
    def test_histogramdd(self, device, dtype):
        # 定义不同的形状列表，用于生成测试数据
        shapes = (
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5),
            (7, 7, 7, 7),
            (16, 8, 4, 2),
            (10, 10, 10),
            (7, 0, 3),
            (5, 0),)

        # 遍历所有可能的组合，测试各种参数条件下的直方图生成
        for contig, bins_contig, weighted, density, shape in \
                product([True, False], [True, False], [True, False], [True, False], shapes):
            # 获取最后一个维度的大小
            D = shape[-1]

            # 生成指定形状和数据类型的张量，用于测试
            values = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9, noncontiguous=not contig)
            # 如果 weighted 为 True，则生成权重张量，否则设为 None
            weights = (
                make_tensor(shape[:-1], dtype=dtype, device=device, low=0, high=9, noncontiguous=not contig)
                if weighted
                else None
            )

            # 测试传递单个 bin 数量的情况
            bin_ct = random.randint(1, 5)
            self._test_histogramdd_numpy(values, bin_ct, None, weights, density)

            # 测试每个维度传递不同 bin 数量的情况
            bin_ct = [random.randint(1, 5) for dim in range(D)]
            self._test_histogramdd_numpy(values, bin_ct, None, weights, density)

            # 测试传递自定义直方图范围的情况
            bin_range_tuples = [sorted((random.uniform(-9, 9), random.uniform(-9, 9))) for dim in range(D)]
            bin_range = [elt for t in bin_range_tuples for elt in t]
            self._test_histogramdd_numpy(values, bin_ct, bin_range, weights, density)

            # 测试范围最小值等于最大值的情况
            for dim in range(D):
                bin_range[2 * dim + 1] = bin_range[2 * dim]
            self._test_histogramdd_numpy(values, bin_ct, bin_range, weights, density)

            # 测试传递自定义 bin 边界的情况
            bin_edges = [make_tensor(ct + 1, dtype=dtype, device=device, low=-9, high=9).msort() for ct in bin_ct]
            if not bins_contig:
                # 因为 msort 总是产生连续的输出，所以在非连续情况下需要处理
                bin_edges_noncontig = [
                    make_tensor(ct + 1, dtype=dtype, device=device, noncontiguous=not bins_contig)
                    for ct in bin_ct
                ]
                for dim in range(D):
                    bin_edges_noncontig[dim].copy_(bin_edges[dim])
                bin_edges = bin_edges_noncontig
            # 检查每个维度的张量是否连续
            for dim in range(D):
                self.assertEqual(bin_edges[dim].is_contiguous(), bins_contig)
            self._test_histogramdd_numpy(values, bin_edges, None, weights, density)
    # 测试直方图函数在特定条件下的错误处理机制

    # 检查当输入张量的数据类型为 torch.int32 时，调用 torch.histogram 函数是否引发 RuntimeError 异常，并确保异常消息包含 'not implemented for'
    with self.assertRaisesRegex(RuntimeError, 'not implemented for'):
        # 创建一个空的标量整数张量，并指定设备和数据类型
        values = make_tensor((), dtype=torch.int32, device=device)
        # 调用 torch.histogram 函数尝试生成直方图
        torch.histogram(values, 1)

    # 根据输入的数据类型不同，测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'input tensor and bins tensors should have the same dtype'
    inconsistent_dtype = torch.float32 if dtype != torch.float32 else torch.float64
    with self.assertRaisesRegex(RuntimeError, 'input tensor and bins tensors should have the same dtype'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 创建数据类型不同的空标量张量作为 bins
        bins = make_tensor((), dtype=inconsistent_dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图
        torch.histogram(values, bins)

    # 类似上面的测试，这次是测试是否引发 RuntimeError 异常，异常消息包含 'input tensor and weight tensor should have the same dtype'
    with self.assertRaisesRegex(RuntimeError, 'input tensor and weight tensor should have the same dtype'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 创建数据类型不同的空标量张量作为 weight
        weight = make_tensor((), dtype=inconsistent_dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 weight 参数
        torch.histogram(values, 1, weight=weight)

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'input tensor and hist tensor should have the same dtype'
    with self.assertRaisesRegex(RuntimeError, 'input tensor and hist tensor should have the same dtype'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 创建数据类型不同的空标量张量作为 hist
        hist = make_tensor((), dtype=inconsistent_dtype, device=device)
        # 创建指定数据类型的空标量张量作为 bin_edges
        bin_edges = make_tensor((), dtype=dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 out 参数
        torch.histogram(values, 1, out=(hist, bin_edges))

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'input tensor and bin_edges tensor should have the same dtype'
    with self.assertRaisesRegex(RuntimeError, 'input tensor and bin_edges tensor should have the same dtype'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 创建指定数据类型的空标量张量作为 hist
        hist = make_tensor((), dtype=dtype, device=device)
        # 创建数据类型不同的空标量张量作为 bin_edges
        bin_edges = make_tensor((), dtype=inconsistent_dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 out 参数
        torch.histogram(values, 1, out=(hist, bin_edges))

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'bins tensor should have one dimension'
    with self.assertRaisesRegex(RuntimeError, 'bins tensor should have one dimension'):
        # 创建指定形状的张量
        t = make_tensor((2, 2), dtype=dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 t 作为 bins 参数
        torch.histogram(t, t)

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'bins tensor should have at least 1 element'
    with self.assertRaisesRegex(RuntimeError, 'bins tensor should have at least 1 element'):
        # 创建形状为 (0,) 的张量
        t = make_tensor((0), dtype=dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 t 作为 bins 参数
        torch.histogram(t, t)

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息包含 'bins must be > 0'
    with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入负数作为 bins 参数
        torch.histogram(values, -1)

    # 测试调用 torch.histogram 函数时是否引发 RuntimeError 异常，异常消息未完全定义完，直接截断了
    with self.assertRaisesRegex(RuntimeError, 'if weight tensor is provided it should have the same shape \ ...'):
        # 创建指定数据类型的空标量张量
        values = make_tensor((), dtype=dtype, device=device)
        # 调用 torch.histogram 函数尝试生成直方图，并传入 weight 参数
        torch.histogram(values, 1, weight=weight)
# 测试用例：验证 torch.histogram 函数在不同情况下是否能正确抛出异常

    # 测试：当参数组合无效时，应抛出 TypeError 异常
    with self.assertRaisesRegex(TypeError, 'received an invalid combination of arguments'):
        # 创建一个形状为空的张量 values，并指定数据类型和设备
        values = make_tensor((), dtype=dtype, device=device)
        # 创建一个形状为空的张量 bin_edges，并指定数据类型和设备
        bin_edges = make_tensor((), dtype=dtype, device=device)
        # 调用 torch.histogram 函数，期望抛出异常
        torch.histogram(values, bin_edges, range=(0, 1))

    # 测试：当最小值大于最大值时，应抛出 RuntimeError 异常
    with self.assertRaisesRegex(RuntimeError, 'min should not exceed max'):
        # 创建一个形状为空的张量 values，并指定数据类型和设备
        values = make_tensor((), dtype=dtype, device=device)
        # 调用 torch.histogram 函数，期望抛出异常
        torch.histogram(values, 2, range=(1, 0))

    # 测试：当范围包含 NaN 时，应抛出 RuntimeError 异常
    with self.assertRaisesRegex(RuntimeError, r'range \[nan, nan\] is not finite'):
        # 创建一个包含 NaN 的张量 values，并指定数据类型和设备
        values = torch.tensor([float("nan")], device=device, dtype=dtype)
        # 调用 torch.histogram 函数，期望抛出异常
        torch.histogram(values, 2)
    # 定义一个测试方法，用于测试空张量的比较运算符
    def test_tensor_compare_ops_empty(self, device):
        # 定义一个空张量的形状
        shape = (2, 0, 4)
        # 在指定设备上生成一个随机张量作为主输入
        master_input = torch.randn(shape, device=device)
        # 使用 numpy 创建一个空的张量作为对照输入
        np_input = np.empty(shape)
        # 定义测试函数列表，每个元素包括函数名、torch 中的函数、以及对应的 numpy 函数
        test_functions = [
            ('amax', torch.amax, np.amax),
            ('amin', torch.amin, np.amin),
            ('max', lambda *args, **kwargs: torch.max(*args, **kwargs).values, np.max),
            ('min', lambda *args, **kwargs: torch.min(*args, **kwargs).values, np.min),
            ('median', lambda *args, **kwargs: torch.median(*args, **kwargs).values, np.median),
        ]

        # 遍历测试函数列表
        for name, fn, np_function in test_functions:
            # 设置错误消息，用于测试失败时的输出
            error_msg = f"test function: {name}"

            # 检查沿着指定维度（dim=2）是否发生了缩减操作，同时检查是否保留了数据类型，与 numpy 函数保持兼容性
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2),
                             fn(master_input, dim=2).cpu().numpy(), msg=error_msg, exact_dtype=False)

            # 检查沿着负方向的指定维度（dim=-1）是否发生了缩减操作，同时检查是否保留了数据类型，与 numpy 函数保持兼容性
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1),
                             fn(master_input, dim=-1).cpu().numpy(), msg=error_msg, exact_dtype=False)

            # 检查沿着指定维度（dim=2）是否发生了缩减操作，并保持维度为1，同时检查是否保留了数据类型，与 numpy 函数保持兼容性
            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2, keepdims=True),
                             fn(master_input, dim=2, keepdim=True).cpu().numpy(), msg=error_msg, exact_dtype=False)

            # 检查沿着负方向的指定维度（dim=-1）是否发生了缩减操作，并保持维度为1，同时检查是否保留了数据类型，与 numpy 函数保持兼容性
            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1, keepdims=True),
                             fn(master_input, dim=-1, keepdim=True).cpu().numpy(), msg=error_msg, exact_dtype=False)

            # 检查在指定为零维度（dim=1）上进行缩减操作时，函数是否会引发错误
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))

    # 用于确保使用比较运算符对零维张量（即空张量）进行缩减操作时，若没有指定 `dim` 参数会引发错误的测试。
    # 这部分测试与 `test_tensor_compare_ops_empty` 中的不同之处在于，前者未指定 `dim` 参数时不会抛出错误。
    # 同时，检查 argmax 的返回类型需要提供一个与输入张量不同的 dtype 参数。此外，numpy 测试中也存在差异。
    # 定义一个测试方法，用于测试张量比较运算中的 argmax、argmin 和 kthvalue 函数在空维度上的行为
    def test_tensor_compare_ops_argmax_argmix_kthvalue_dim_empty(self, device):
        # 定义一个形状为 (2, 0, 4) 的张量
        shape = (2, 0, 4)
        # 生成一个随机张量作为主输入，指定设备
        master_input = torch.randn(shape, device=device)
        # 创建一个形状相同但未初始化的 NumPy 数组
        np_input = np.empty(shape)
        # 定义测试函数的列表，每个元素包含名称、Torch 函数、Torch 函数的参数、NumPy 函数
        test_functions = [
            ('argmax', torch.argmax, {'dtype': torch.int64}, np.argmax),
            ('argmin', torch.argmin, {'dtype': torch.int64}, np.argmin),
            ('kthvalue', lambda *args, k=1, **kwargs: torch.kthvalue(*args, k=1, **kwargs).values,
             {}, lambda *args, k=1, axis=None, **kwargs: np.partition(*args, k, **kwargs).take(k - 1, axis=axis))
        ]

        # 对于每个测试函数，执行以下操作
        for name, fn, dtype, np_function in test_functions:
            # 设置错误消息的基础部分
            error_msg = f"test function: {name}"
            # 测试在维度 dim=2 上的 argmax 或 argmin 结果是否与预期的空张量相等
            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=2), msg=error_msg)
            # 测试相应的 NumPy 函数是否产生与预期相同的结果
            self.assertEqual(
                np_function(np_input, axis=2), fn(master_input, dim=2).cpu().numpy(), msg=error_msg, exact_dtype=False
            )

            # 测试在维度 dim=-1 上的 argmax 或 argmin 结果是否与预期的空张量相等
            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=-1), msg=error_msg)
            # 测试相应的 NumPy 函数是否产生与预期相同的结果
            self.assertEqual(
                np_function(np_input, axis=-1), fn(master_input, dim=-1).cpu().numpy(), msg=error_msg, exact_dtype=False
            )

            # 在 NumPy 中不存在保持维度的变体，因此跳过此部分的测试
            # 测试在维度 dim=2 且保持维度的情况下，argmax 或 argmin 结果是否与预期的空张量相等
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            # 测试在维度 dim=-1 且保持维度的情况下，argmax 或 argmin 结果是否与预期的空张量相等
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)

            # 检查函数是否在指定的零维度上引发错误
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))
            # 对于非 kthvalue 函数，检查函数是否在不指定维度时引发错误
            if name != 'kthvalue':
                self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input))

    # 测试以确保使用数学运算符对零维张量（即空张量）进行降维时的工作情况，并且在指定的维度为 0 时引发错误
    # 尽管与 test_tensor_compare_ops_optional_dim_empty 和 test_tensor_compare_ops_empty 存在一些重复，
    # 但这些测试被单独保留，因为数学运算符的测试还需要使用 allclose() 或 isinf() 来检查返回的数据的正确性，
    # 而这在前面的测试中并不存在。
    @skipIfNoSciPy
    # 测试以确保 any() 和 all() 函数在零维张量上的工作情况
    # 与检查零维张量的降维的其他测试分开进行，因为这些测试的行为与前面的测试有显著不同
    # 定义一个测试方法，用于测试在空维度上进行任意和全部操作
    def test_reduction_empty_any_all(self, device):
        # 定义一个形状为 (2, 0, 4) 的张量
        shape = (2, 0, 4)
        # 生成一个指定设备上的随机张量 x
        x = torch.randn(shape, device=device)

        # 针对所有类型和复杂类型及 torch.half 和 torch.bool 进行迭代测试
        for dtype in all_types_and_complex_and(torch.half, torch.bool):
            # 如果 dtype 是 torch.uint8，则输出数据类型也为 torch.uint8
            if dtype == torch.uint8:
                out_dtype = torch.uint8
            else:
                # 否则输出数据类型为 torch.bool，因为 all/any 的输出类型与输入 dtype 无关
                out_dtype = torch.bool  # output of all/any is bool irrespective of input dtype

            # 将 x 转换为当前的 dtype
            xb = x.to(dtype)
            yb = x.to(dtype)

            # 进行 any 操作的断言
            self.assertEqual((2, 0), xb.any(2).shape)
            self.assertEqual((2, 0, 1), xb.any(2, keepdim=True).shape)
            self.assertEqual(torch.zeros((2, 4), device=device, dtype=out_dtype), xb.any(1))
            self.assertEqual(torch.zeros((2, 1, 4), device=device, dtype=out_dtype), xb.any(1, keepdim=True))
            self.assertEqual(torch.zeros((), device=device, dtype=out_dtype), xb.any())

            # 进行 all 操作的断言
            self.assertEqual((2, 0), xb.all(2).shape)
            self.assertEqual((2, 0, 1), xb.all(2, keepdim=True).shape)
            self.assertEqual(torch.ones((2, 4), device=device, dtype=out_dtype), xb.all(1))
            self.assertEqual(torch.ones((2, 1, 4), device=device, dtype=out_dtype), xb.all(1, keepdim=True))
            self.assertEqual(torch.ones((), device=device, dtype=out_dtype), xb.all())

    # TODO: can these be merged with their respective OpInfos?
    # 定义测试 reduce 操作的数据类型的方法
    def test_reduce_dtype(self, device):
        # 定义测试 reduction 方法的内部函数
        def test_reduction(op, has_no_dim, takes_dtype=True):
            # 创建一个形状为 (3, 3) 的随机张量 x，数据类型为 torch.float
            x = torch.randn(3, 3, dtype=torch.float, requires_grad=True, device=device)

            # 如果 has_no_dim 为真，则执行以下操作
            if has_no_dim:
                # 对 op(x) 和 op(x, dtype=torch.double) 分别计算梯度
                grad1, = torch.autograd.grad([op(x)], [x])
                grad2, = torch.autograd.grad([op(x, dtype=torch.double)], [x])
                # 断言两个梯度值相等
                self.assertEqual(grad1, grad2)
                # 断言 grad2 的数据类型为 torch.float
                self.assertEqual(grad2.dtype, torch.float)

            # 创建一个与 op(x, dim=0) 形状相同的随机梯度 gi，数据类型为 torch.float
            gi = torch.randn(op(x, dim=0).shape, dtype=torch.float, device=device)
            # 分别对 op(x, dim=0) 计算梯度
            grad1, = torch.autograd.grad([op(x, dim=0)], [x], gi)
            # 如果 takes_dtype 为真，则对 op(x, dim=0, dtype=torch.double) 计算梯度
            if takes_dtype:
                grad2, = torch.autograd.grad([op(x, dim=0, dtype=torch.double)], [x], gi.double())
            else:
                grad2, = torch.autograd.grad([op(x.double(), dim=0)], [x], gi.double())
            # 断言两个梯度值相等
            self.assertEqual(grad1, grad2)
            # 断言 grad2 的数据类型为 torch.float
            self.assertEqual(grad2.dtype, torch.float)

        # 测试 torch.sum 方法
        test_reduction(torch.sum, True)
        # 测试 torch.prod 方法
        test_reduction(torch.prod, True)
        # 测试 torch.cumsum 方法
        test_reduction(torch.cumsum, False)
        # 测试 torch.cumprod 方法
        test_reduction(torch.cumprod, False)
        # 测试 torch.logcumsumexp 方法，该方法不需要指定 dtype 参数
        test_reduction(torch.logcumsumexp, False, takes_dtype=False)

    @ops(reference_masked_ops)
    # 定义一个测试函数，用于测试带有遮罩的张量在只有步长的情况下进行的减少操作，使用numpy的减少操作作为参考。
    def test_reference_masked(self, device, dtype, op):
        """Test masked reduction operations on strided-only tensors using
        numpy reductions as reference.
        """

        # 定义一个辅助函数，将torch张量转换为numpy数组
        def to_numpy(input):
            if input.dtype is torch.bfloat16:
                return input.cpu().to(torch.float32).numpy()
            else:
                return input.cpu().numpy()

        # 调用op对象的sample_inputs_func方法生成测试样本
        samples = op.sample_inputs_func(op, device, dtype, requires_grad=False)
        # 遍历每个样本输入
        for sample_input in samples:
            t = sample_input.input
            # 使用op对象执行操作，得到实际结果
            actual = op(t, *sample_input.args, **sample_input.kwargs)
            # 判断是否是精确的数据类型
            exact_dtype = not (t.dtype is torch.bfloat16
                               or (op.promotes_int_to_float and not torch.is_floating_point(t)))
            # 调用op对象的ref方法，生成期望结果
            expected = op.ref(to_numpy(t), *sample_input.args,
                              **dict(
                                  # `identity`被映射到numpy减少函数的`initial`参数
                                  identity=torch.masked._reduction_identity(op.name, t),
                                  **sample_input.kwargs))

            # 解决问题 https://github.com/pytorch/pytorch/issues/66556
            expected = np.asarray(expected)  # 将numpy标量转换为numpy.ndarray实例

            # numpy在Windows上会产生uint32类型的数据，因此需要额外判断精确的数据类型
            if expected.dtype in [np.uint64, np.uint32]:
                exact_dtype = False

            # 如果输入张量元素数量小于10，则生成消息，否则消息为None
            msg = ("Failed to produce expected results! Input tensor was"
                   f" {t}, torch result is {actual}, and reference result is"
                   f" {expected}.") if t.numel() < 10 else None

            # 断言torch的实际结果与期望结果相等，如果不相等则输出消息，同时检查精确的数据类型
            self.assertEqual(actual, expected, msg, exact_dtype=exact_dtype)

    # 仅在CUDA环境下执行该测试
    @onlyCUDA
    # 对大型张量执行测试，大小为"8GB"
    @largeTensorTest("8GB")
    # 用torch.half、torch.chalf和torch.bfloat16类型的数据类型执行测试
    @dtypes(torch.half, torch.chalf, torch.bfloat16)
    def test_reductions_large_half_tensors(self, device, dtype):
        # 创建一个大小为2^31的全1张量，并指定设备和数据类型
        t = torch.ones(2**31, device=device, dtype=dtype)
        # 将张量的后半部分设置为-1
        t[2**30:] = -1
        # 创建一个预期结果为0的张量，指定设备和数据类型
        expected = torch.tensor(0, device=device, dtype=dtype)
        # 断言torch.sum(t)的结果与预期结果相等
        self.assertEqual(torch.sum(t), expected)

        # 对于torch.chalf数据类型，期望产生RuntimeError，错误消息为"not implemented for 'ComplexHalf'"
        err_msg = "not implemented for 'ComplexHalf'"
        ctx = self.assertRaisesRegex(
            RuntimeError, err_msg) if dtype is torch.chalf else contextlib.nullcontext()
        # 使用上下文管理器ctx，验证torch.mean(t)的结果与预期结果相等
        with ctx:
            self.assertEqual(torch.mean(t), expected)
# 调用函数 instantiate_device_type_tests，并将 TestReductions 作为参数传递，全局作用域中查找所需的函数和变量
instantiate_device_type_tests(TestReductions, globals())

# 检查当前脚本是否作为主程序运行
if __name__ == '__main__':
    # 如果是主程序，则执行函数 run_tests()
    run_tests()
```