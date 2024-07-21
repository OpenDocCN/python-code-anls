# `.\pytorch\test\torch_np\test_basic.py`

```
# Owner(s): ["module: dynamo"]

import functools  # 导入 functools 模块，用于创建偏函数
import inspect  # 导入 inspect 模块，用于获取对象信息
from unittest import expectedFailure as xfail, skipIf as skip  # 导入 unittest 模块中的 expectedFailure 和 skipIf 装饰器

import numpy as _np  # 导入 numpy 库，命名为 _np
from pytest import raises as assert_raises  # 从 pytest 中导入 raises 函数，命名为 assert_raises

import torch  # 导入 torch 库

import torch._numpy as w  # 导入 torch._numpy 模块，命名为 w
import torch._numpy._ufuncs as _ufuncs  # 导入 torch._numpy._ufuncs 模块，命名为 _ufuncs
import torch._numpy._util as _util  # 导入 torch._numpy._util 模块，命名为 _util
from torch._numpy.testing import assert_allclose, assert_equal  # 从 torch._numpy.testing 中导入 assert_allclose 和 assert_equal 函数

from torch.testing._internal.common_cuda import TEST_CUDA  # 从 torch.testing._internal.common_cuda 中导入 TEST_CUDA 变量
from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 中导入以下函数和类
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

# These function receive one array_like arg and return one array_like result
one_arg_funcs = [  # 定义接收一个 array_like 参数并返回一个 array_like 结果的函数列表
    w.asarray,  # 将 w.asarray 函数添加到列表中
    w.empty_like,  # 将 w.empty_like 函数添加到列表中
    w.ones_like,  # 将 w.ones_like 函数添加到列表中
    w.zeros_like,  # 将 w.zeros_like 函数添加到列表中
    functools.partial(w.full_like, fill_value=42),  # 使用 functools.partial 创建 fill_value 参数为 42 的 w.full_like 偏函数并添加到列表中
    w.corrcoef,  # 将 w.corrcoef 函数添加到列表中
    w.squeeze,  # 将 w.squeeze 函数添加到列表中
    w.argmax,  # 将 w.argmax 函数添加到列表中
    # w.bincount,     # XXX: input dtypes
    w.prod,  # 将 w.prod 函数添加到列表中
    w.sum,  # 将 w.sum 函数添加到列表中
    w.real,  # 将 w.real 函数添加到列表中
    w.imag,  # 将 w.imag 函数添加到列表中
    w.angle,  # 将 w.angle 函数添加到列表中
    w.real_if_close,  # 将 w.real_if_close 函数添加到列表中
    w.isreal,  # 将 w.isreal 函数添加到列表中
    w.iscomplex,  # 将 w.iscomplex 函数添加到列表中
    w.isneginf,  # 将 w.isneginf 函数添加到列表中
    w.isposinf,  # 将 w.isposinf 函数添加到列表中
    w.i0,  # 将 w.i0 函数添加到列表中
    w.copy,  # 将 w.copy 函数添加到列表中
    w.array,  # 将 w.array 函数添加到列表中
    w.round,  # 将 w.round 函数添加到列表中
    w.around,  # 将 w.around 函数添加到列表中
    w.flip,  # 将 w.flip 函数添加到列表中
    w.vstack,  # 将 w.vstack 函数添加到列表中
    w.hstack,  # 将 w.hstack 函数添加到列表中
    w.dstack,  # 将 w.dstack 函数添加到列表中
    w.column_stack,  # 将 w.column_stack 函数添加到列表中
    w.row_stack,  # 将 w.row_stack 函数添加到列表中
    w.flatnonzero,  # 将 w.flatnonzero 函数添加到列表中
]

ufunc_names = _ufuncs._unary  # 获取 _ufuncs 模块中的一元函数名称列表
ufunc_names.remove("invert")  # 移除 "invert" 函数名，因为 torch 中没有对应的实现 'Float' 类型的 bitwise_not_cpu 函数
ufunc_names.remove("bitwise_not")  # 移除 "bitwise_not" 函数名

one_arg_funcs += [getattr(_ufuncs, name) for name in ufunc_names]  # 将 _ufuncs 模块中除去特定函数名后的其他一元函数添加到 one_arg_funcs 列表中


@instantiate_parametrized_tests
class TestOneArr(TestCase):
    """Base for smoke tests of one-arg functions: (array_like) -> (array_like)

    Accepts array_likes, torch.Tensors, w.ndarays; returns an ndarray
    """

    @parametrize("func", one_arg_funcs)
    def test_asarray_tensor(self, func):
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t)

        assert isinstance(ta, w.ndarray)  # 断言 ta 是 w.ndarray 类型的对象

    @parametrize("func", one_arg_funcs)
    def test_asarray_list(self, func):
        lst = [[1.0, 2, 3], [4, 5, 6]]
        la = func(lst)

        assert isinstance(la, w.ndarray)  # 断言 la 是 w.ndarray 类型的对象

    @parametrize("func", one_arg_funcs)
    def test_asarray_array(self, func):
        a = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        la = func(a)

        assert isinstance(la, w.ndarray)  # 断言 la 是 w.ndarray 类型的对象


one_arg_axis_funcs = [  # 定义接收一个 array_like 和一个 axis 参数并返回一个 array_like 结果的函数列表
    w.argmax,  # 将 w.argmax 函数添加到列表中
    w.argmin,  # 将 w.argmin 函数添加到列表中
    w.prod,  # 将 w.prod 函数添加到列表中
    w.sum,  # 将 w.sum 函数添加到列表中
    w.all,  # 将 w.all 函数添加到列表中
    w.any,  # 将 w.any 函数添加到列表中
    w.mean,  # 将 w.mean 函数添加到列表中
    w.argsort,  # 将 w.argsort 函数添加到列表中
    w.std,  # 将 w.std 函数添加到列表中
    w.var,  # 将 w.var 函数添加到列表中
    w.flip,  # 将 w.flip 函数添加到列表中
]


@instantiate_parametrized_tests
class TestOneArrAndAxis(TestCase):
    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_tensor(self, func, axis):
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)  # 断言 ta 是 w.ndarray 类型的对象

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_list(self, func, axis):
        t = [[1.0, 2, 3], [4, 5, 6]]
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)  # 断言 ta 是 w.ndarray 类型的对象
    # 使用参数化装饰器@parametrize为测试方法创建多个参数组合
    @parametrize("axis", [0, 1, -1, None])
    # 定义测试方法，测试对数组进行某个操作func，并指定轴向axis
    def test_andaxis_array(self, func, axis):
        # 创建一个包含浮点数的二维数组
        t = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        # 调用被测试的函数func，传入数组t和指定的轴向axis，返回结果
        ta = func(t, axis=axis)
        # 断言返回的结果ta是w.ndarray类型的实例
        assert isinstance(ta, w.ndarray)
@instantiate_parametrized_tests
class TestOneArrAndAxesTuple(TestCase):
    """Defines a test case class for parametrized tests related to array manipulation."""

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_tensor(self, func, axes):
        """Test method for tensor input with tuple of axes."""

        t = torch.ones((1, 2, 3))  # Create a tensor of shape (1, 2, 3) filled with ones
        ta = func(t, axes=axes)  # Apply transpose function `func` with specified `axes`
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`

        # Specific test for np.transpose behavior when axes is None
        if axes is None:
            newshape = (3, 2, 1)  # Define expected new shape
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(w.ndim(t)))  # Calculate new shape based on axes
        assert ta.shape == newshape  # Assert that the shape of the result matches the expected shape

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_list(self, func, axes):
        """Test method for list input with tuple of axes."""

        t = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]  # Create a list representing a tensor
        ta = func(t, axes=axes)  # Apply transpose function `func` with specified `axes`
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_array(self, func, axes):
        """Test method for array input with tuple of axes."""

        t = w.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])  # Create an array using `w.asarray`
        ta = func(t, axes=axes)  # Apply transpose function `func` with specified `axes`
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`

        if axes is None:
            newshape = (3, 2, 1)  # Define expected new shape when axes is None
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(t.ndim))  # Calculate new shape based on axes
        assert ta.shape == newshape  # Assert that the shape of the result matches the expected shape


arr_shape_funcs = [
    w.reshape,
    w.empty_like,
    w.ones_like,
    functools.partial(w.full_like, fill_value=42),
    w.broadcast_to,
]


@instantiate_parametrized_tests
class TestOneArrAndShape(TestCase):
    """Defines a test case class for parametrized tests related to array reshaping and manipulation."""

    def setUp(self):
        """Set up method executed before each test method."""
        self.shape = (2, 3)  # Define a common shape used in tests
        self.shape_arg_name = {
            w.reshape: "newshape",
        }  # Dictionary mapping functions to their expected argument names

    @parametrize("func", arr_shape_funcs)
    def test_andshape_tensor(self, func):
        """Test method for tensor input with shape parameter."""

        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])  # Create a tensor
        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}  # Prepare shape argument
        ta = func(t, **shape_dict)  # Apply function `func` with specified shape argument
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`
        assert ta.shape == self.shape  # Assert that the shape of the result matches the expected shape

    @parametrize("func", arr_shape_funcs)
    def test_andshape_list(self, func):
        """Test method for list input with shape parameter."""

        t = [[1, 2, 3], [4, 5, 6]]  # Create a list representing an array
        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}  # Prepare shape argument
        ta = func(t, **shape_dict)  # Apply function `func` with specified shape argument
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`
        assert ta.shape == self.shape  # Assert that the shape of the result matches the expected shape

    @parametrize("func", arr_shape_funcs)
    def test_andshape_array(self, func):
        """Test method for array input with shape parameter."""

        t = w.asarray([[1, 2, 3], [4, 5, 6]])  # Create an array using `w.asarray`
        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}  # Prepare shape argument
        ta = func(t, **shape_dict)  # Apply function `func` with specified shape argument
        assert isinstance(ta, w.ndarray)  # Assert that the result is of type `w.ndarray`
        assert ta.shape == self.shape  # Assert that the shape of the result matches the expected shape


one_arg_scalar_funcs = [(w.size, _np.size), (w.shape, _np.shape), (w.ndim, _np.ndim)]


@instantiate_parametrized_tests
class TestOneArrToScalar(TestCase):
    """Defines a test case class for parametrized tests related to array to scalar conversion."""

    @parametrize("func, np_func", one_arg_scalar_funcs)
    # 定义一个测试方法，用于测试将张量转换为标量的函数
    def test_toscalar_tensor(self, func, np_func):
        # 创建一个包含两个行三列的张量
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        # 调用给定的函数将张量 t 转换为标量 ta
        ta = func(t)
        # 调用对应的 NumPy 函数将 NumPy 数组 _np.asarray(t) 转换为标量 tn
        tn = np_func(_np.asarray(t))

        # 断言 ta 不是 w.ndarray 类型
        assert not isinstance(ta, w.ndarray)
        # 断言 ta 等于 tn
        assert ta == tn

    # 使用参数化装饰器，针对每个指定的函数进行测试，将列表转换为标量
    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_list(self, func, np_func):
        # 创建一个包含两个子列表的列表
        t = [[1, 2, 3], [4, 5, 6]]
        # 调用给定的函数将列表 t 转换为标量 ta
        ta = func(t)
        # 调用对应的 NumPy 函数将列表 t 转换为标量 tn
        tn = np_func(t)

        # 断言 ta 不是 w.ndarray 类型
        assert not isinstance(ta, w.ndarray)
        # 断言 ta 等于 tn
        assert ta == tn

    # 使用参数化装饰器，针对每个指定的函数进行测试，将数组转换为标量
    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_array(self, func, np_func):
        # 使用 w.asarray 将列表转换为数组
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        # 调用给定的函数将数组 t 转换为标量 ta
        ta = func(t)
        # 调用对应的 NumPy 函数将数组 t 转换为标量 tn
        tn = np_func(t)

        # 断言 ta 不是 w.ndarray 类型
        assert not isinstance(ta, w.ndarray)
        # 断言 ta 等于 tn
        assert ta == tn
shape_funcs = [w.zeros, w.empty, w.ones, functools.partial(w.full, fill_value=42)]
# 定义了一个包含四个函数的列表，用于创建不同形状的数组

@instantiate_parametrized_tests
class TestShapeLikeToArray(TestCase):
    """Smoke test (shape_like) -> array."""

    shape = (3, 4)

    @parametrize("func", shape_funcs)
    def test_shape(self, func):
        # 使用给定的函数创建数组
        a = func(self.shape)

        # 断言数组类型为 w.ndarray
        assert isinstance(a, w.ndarray)
        # 断言数组形状与预期形状相同
        assert a.shape == self.shape


seq_funcs = [w.atleast_1d, w.atleast_2d, w.atleast_3d, w.broadcast_arrays]
# 定义了一个包含四个函数的列表，用于处理数组序列

@instantiate_parametrized_tests
class TestSequenceOfArrays(TestCase):
    """Smoke test (sequence of arrays) -> (sequence of arrays)."""

    @parametrize("func", seq_funcs)
    def test_single_tensor(self, func):
        # 创建一个 torch 张量
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        # 使用指定函数处理张量
        ta = func(t)

        # 如果函数是 broadcast_arrays，则返回一个元组，否则返回一个数组
        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = ta[0] if unpack else ta

        # 断言结果类型为 w.ndarray
        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_single_list(self, func):
        # 创建一个列表
        lst = [[1, 2, 3], [4, 5, 6]]
        # 使用指定函数处理列表
        la = func(lst)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        # 断言结果类型为 w.ndarray
        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_single_array(self, func):
        # 创建一个 w.asarray 数组
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        # 使用指定函数处理数组
        la = func(a)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        # 断言结果类型为 w.ndarray
        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_several(self, func):
        # 准备多个输入数组
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        # 使用指定函数处理多个数组
        result = func(*arys)
        # 断言结果为元组或列表
        assert isinstance(result, (tuple, list))
        # 断言结果长度与输入数组长度相同
        assert len(result) == len(arys)
        # 断言结果中所有元素类型为 w.ndarray
        assert all(isinstance(_, w.ndarray) for _ in result)


seq_to_single_funcs = [
    w.concatenate,
    w.stack,
    w.vstack,
    w.hstack,
    w.dstack,
    w.column_stack,
    w.row_stack,
]
# 定义了一个包含多个函数的列表，用于将多个数组转换为单个数组

@instantiate_parametrized_tests
class TestSequenceOfArraysToSingle(TestCase):
    """Smoke test (sequence of arrays) -> (array)."""

    @parametrize("func", seq_to_single_funcs)
    def test_several(self, func):
        # 准备多个输入数组
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        # 使用指定函数处理多个数组
        result = func(arys)
        # 断言结果类型为 w.ndarray
        assert isinstance(result, w.ndarray)


single_to_seq_funcs = (
    w.nonzero,
    # https://github.com/Quansight-Labs/numpy_pytorch_interop/pull/121#discussion_r1172824545
    # w.tril_indices_from,
    # w.triu_indices_from,
    w.where,
)
# 定义了一个包含多个函数的元组，用于将单个数组转换为序列

@instantiate_parametrized_tests
class TestArrayToSequence(TestCase):
    """Smoke test array -> (tuple of arrays)."""

    @parametrize("func", single_to_seq_funcs)
    # 定义一个测试方法，用于测试将张量转换为数组的功能
    def test_asarray_tensor(self, func):
        # 创建一个张量对象
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        # 使用给定的函数将张量转换为数组
        ta = func(t)
    
        # 断言返回值是一个元组
        assert isinstance(ta, tuple)
        # 断言元组中的每个元素都是 w.ndarray 类型的对象
        assert all(isinstance(x, w.ndarray) for x in ta)
    
    # 使用 parametrize 装饰器，将不同的函数 func 应用于同一个测试方法
    @parametrize("func", single_to_seq_funcs)
    def test_asarray_list(self, func):
        # 创建一个二维列表
        lst = [[1, 2, 3], [4, 5, 6]]
        # 使用给定的函数将列表转换为数组
        la = func(lst)
    
        # 断言返回值是一个元组
        assert isinstance(la, tuple)
        # 断言元组中的每个元素都是 w.ndarray 类型的对象
        assert all(isinstance(x, w.ndarray) for x in la)
    
    # 使用 parametrize 装饰器，将不同的函数 func 应用于同一个测试方法
    @parametrize("func", single_to_seq_funcs)
    def test_asarray_array(self, func):
        # 创建一个 w.asarray 返回的数组对象
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        # 使用给定的函数将数组转换为数组（应为不改变类型）
        la = func(a)
    
        # 断言返回值是一个元组
        assert isinstance(la, tuple)
        # 断言元组中的每个元素都是 w.ndarray 类型的对象
        assert all(isinstance(x, w.ndarray) for x in la)
funcs_and_args = [
    (w.linspace, (0, 10, 11)),  # 使用 linspace 函数生成从 0 到 10 的 11 个均匀分布的数列
    (w.logspace, (1, 2, 5)),    # 使用 logspace 函数生成从 10^1 到 10^2 的 5 个对数分布的数列
    (w.logspace, (1, 2, 5, 11)),  # 使用 logspace 函数生成从 10^1 到 10^2 的 11 个对数分布的数列
    (w.geomspace, (1, 1000, 5, 11)),  # 使用 geomspace 函数生成从 1 到 1000 的 11 个几何分布的数列
    (w.eye, (5, 6)),           # 使用 eye 函数生成一个 5x6 的单位矩阵
    (w.identity, (3,)),        # 使用 identity 函数生成一个 3x3 的单位矩阵
    (w.arange, (5,)),          # 使用 arange 函数生成从 0 到 5 的整数序列
    (w.arange, (5, 8)),        # 使用 arange 函数生成从 5 到 8 的整数序列
    (w.arange, (5, 8, 0.5)),   # 使用 arange 函数生成从 5 到 8，步长为 0.5 的浮点数序列
    (w.tri, (3, 3, -1)),       # 使用 tri 函数生成一个 3x3 的下三角矩阵
]

@instantiate_parametrized_tests
class TestPythonArgsToArray(TestCase):
    """Smoke_test (sequence of scalars) -> (array)"""

    @parametrize("func, args", funcs_and_args)
    def test_argstoarray_simple(self, func, args):
        # 调用指定的函数和参数生成数组
        a = func(*args)
        # 断言生成的对象是一个数组
        assert isinstance(a, w.ndarray)


class TestNormalizations(TestCase):
    """Smoke test generic problems with normalizations."""

    def test_unknown_args(self):
        # 检查对带有未知参数的函数调用是否会引发 TypeError
        a = w.arange(7) % 2 == 0

        # 测试未知的位置参数是否会引发 TypeError
        with assert_raises(TypeError):
            w.nonzero(a, "kaboom")

        # 测试未知的关键字参数是否会引发 TypeError
        with assert_raises(TypeError):
            w.nonzero(a, oops="ouch")

    def test_too_few_args_positional(self):
        # 测试当位置参数数量不足时是否会引发 TypeError
        with assert_raises(TypeError):
            w.nonzero()

    def test_unknown_args_with_defaults(self):
        # 测试带有默认值的函数调用，其参数数量是否符合预期
        w.eye(3)

        # 测试带有过多位置参数的函数调用是否会引发 TypeError
        with assert_raises(TypeError):
            w.eye()


class TestCopyTo(TestCase):
    def test_copyto_basic(self):
        # 测试基本的数组复制操作
        dst = w.empty(4)
        src = w.arange(4)
        w.copyto(dst, src)
        # 断言复制后目标数组与源数组相等
        assert (dst == src).all()

    def test_copytobcast(self):
        # 测试不同形状的数组复制是否会报错或者是否能够进行广播
        dst = w.empty((4, 2))
        src = w.arange(4)

        # 测试无法进行广播的情况是否会引发 RuntimeError
        with assert_raises(RuntimeError):
            w.copyto(dst, src)

        # 测试能够进行广播的情况
        dst = w.empty((2, 4))
        w.copyto(dst, src)
        assert (dst == src).all()

    def test_copyto_typecast(self):
        # 测试数组复制时的类型转换
        dst = w.empty(4, dtype=int)
        src = w.arange(4, dtype=float)

        # 测试不允许类型转换时是否会引发 TypeError
        with assert_raises(TypeError):
            w.copyto(dst, src, casting="no")

        # 强制进行类型转换
        w.copyto(dst, src, casting="unsafe")
        assert (dst == src).all()


class TestDivmod(TestCase):
    def test_divmod_out(self):
        # 测试 divmod 函数的输出参数设置
        x1 = w.arange(8, 15)
        x2 = w.arange(4, 11)

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        # 断言计算结果与预期一致
        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

        out1, out2 = out
        # 断言输出参数与预期一致
        assert quot is out[0]
        assert rem is out[1]

    def test_divmod_out_list(self):
        # 测试 divmod 函数对列表的输出参数设置
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        # 断言输出参数与预期一致
        assert quot is out[0]
        assert rem is out[1]

    @xfail  # ("out1, out2 not implemented")
    # 定义测试函数，测试 divmod 函数处理正整数参数的情况
    def test_divmod_pos_only(self):
        # 创建两个包含整数的列表
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        # 使用 w.empty_like 函数创建两个与 x1 相同形状的空数组 out1 和 out2
        out1, out2 = w.empty_like(x1), w.empty_like(x1)

        # 调用 w.divmod 函数，计算 x1 与 x2 的商和余数，并将结果存入 out1 和 out2
        quot, rem = w.divmod(x1, x2, out1, out2)

        # 断言返回的商 quot 和余数 rem 分别与 out1 和 out2 相等
        assert quot is out1
        assert rem is out2

    # 定义测试函数，测试 divmod 函数处理没有输出参数的情况
    def test_divmod_no_out(self):
        # 创建包含整数的数组 x1 和 x2
        x1 = w.array([4, 5, 6])
        x2 = w.array([2, 1, 2])

        # 调用 w.divmod 函数，计算 x1 与 x2 的商和余数，不指定输出参数
        quot, rem = w.divmod(x1, x2)

        # 断言计算结果的商 quot 等于 x1 与 x2 的整数除法结果，余数 rem 等于 x1 与 x2 的取模结果
        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

    # 定义测试函数，测试 divmod 函数同时使用位置参数和关键字参数进行输出
    def test_divmod_out_both_pos_and_kw(self):
        # 创建一个空的数组 o
        o = w.empty(1)
        
        # 使用 assert_raises 检查 TypeError 是否被正确抛出，因为使用了重复的输出参数
        with assert_raises(TypeError):
            w.divmod(1, 2, o, o, out=(o, o))
class TestSmokeNotImpl(TestCase):
    def test_nimpl_basic(self):
        # smoke test that the "NotImplemented" annotation is picked up
        # 烟雾测试，检查是否捕获到"未实现"注释异常
        with assert_raises(NotImplementedError):
            # 检查调用w.empty(3)时是否抛出NotImplementedError异常
            w.empty(3, like="ooops")


@instantiate_parametrized_tests
class TestDefaultDtype(TestCase):
    def test_defaultdtype_defaults(self):
        # by default, both floats and ints 64 bit
        # 默认情况下，浮点数和整数均为64位
        x = w.empty(3)
        z = x + 1j * x

        assert x.dtype.torch_dtype == torch.float64
        assert z.dtype.torch_dtype == torch.complex128

        assert w.arange(3).dtype.torch_dtype == torch.int64

    @parametrize("dt", ["pytorch", "float32", torch.float32])
    def test_set_default_float(self, dt):
        try:
            w.set_default_dtype(fp_dtype=dt)

            x = w.empty(3)
            z = x + 1j * x

            assert x.dtype.torch_dtype == torch.float32
            assert z.dtype.torch_dtype == torch.complex64

        finally:
            # restore the
            # 恢复默认的数据类型设置为"numpy"
            w.set_default_dtype(fp_dtype="numpy")


@skip(_np.__version__ <= "1.23", reason="from_dlpack is new in NumPy 1.23")
class TestExport(TestCase):
    def test_exported_objects(self):
        # test exported functions not in NumPy
        # 测试导出的函数是否不在NumPy中
        exported_fns = (
            x
            for x in dir(w)
            if inspect.isfunction(getattr(w, x))
            and not x.startswith("_")
            and x != "set_default_dtype"
        )
        diff = set(exported_fns).difference(set(dir(_np)))
        assert len(diff) == 0, str(diff)


class TestCtorNested(TestCase):
    def test_arrays_in_lists(self):
        # test array conversion in nested lists
        # 测试嵌套列表中的数组转换
        lst = [[1, 2], [3, w.array(4)]]
        assert_equal(w.asarray(lst), [[1, 2], [3, 4]])


class TestMisc(TestCase):
    def test_ndarrays_to_tensors(self):
        # test conversion of ndarrays to tensors
        # 测试将ndarrays转换为张量
        out = _util.ndarrays_to_tensors(((w.asarray(42), 7), 3))
        assert len(out) == 2
        assert isinstance(out[0], tuple) and len(out[0]) == 2
        assert isinstance(out[0][0], torch.Tensor)

    @skip(not TEST_CUDA, reason="requires cuda")
    def test_f16_on_cuda(self):
        # make sure operations with float16 tensors give same results on CUDA and on CPU
        # 确保在CUDA和CPU上对float16张量进行的操作结果相同
        t = torch.arange(5, dtype=torch.float16)
        assert_allclose(w.vdot(t.cuda(), t.cuda()), w.vdot(t, t))
        assert_allclose(w.inner(t.cuda(), t.cuda()), w.inner(t, t))
        assert_allclose(w.matmul(t.cuda(), t.cuda()), w.matmul(t, t))
        assert_allclose(w.einsum("i,i", t.cuda(), t.cuda()), w.einsum("i,i", t, t))

        assert_allclose(w.mean(t.cuda()), w.mean(t))

        assert_allclose(w.cov(t.cuda(), t.cuda()), w.cov(t, t).tensor.cuda())
        assert_allclose(w.corrcoef(t.cuda()), w.corrcoef(t).tensor.cuda())


if __name__ == "__main__":
    run_tests()
```