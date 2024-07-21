# `.\pytorch\test\torch_np\test_ndarray_methods.py`

```py
# Owner(s): ["module: dynamo"]

# 导入必要的模块和函数
import itertools
from unittest import expectedFailure as xfail, skipIf as skipif

import numpy

import pytest
from pytest import raises as assert_raises

# 导入测试相关的 Torch 内部工具和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo,
)

# 根据 TEST_WITH_TORCHDYNAMO 的值选择性地导入 numpy 和 assert_equal
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


# 定义测试类 TestIndexing，继承自 TestCase
class TestIndexing(TestCase):
    
    # 标记为条件跳过的测试用例，当 TEST_WITH_TORCHDYNAMO 为 True 时跳过
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attr, type of a[0, 0]")
    def test_indexing_simple(self):
        # 创建一个 numpy 数组
        a = np.array([[1, 2, 3], [4, 5, 6]])

        # 断言 a[0, 0] 的类型为 np.ndarray
        assert isinstance(a[0, 0], np.ndarray)
        # 断言 a[0, :] 的类型为 np.ndarray
        assert isinstance(a[0, :], np.ndarray)
        # 断言 a[0, :].tensor._base 是 a.tensor
        assert a[0, :].tensor._base is a.tensor

    # 定义测试方法 test_setitem
    def test_setitem(self):
        # 创建一个 numpy 数组
        a = np.array([[1, 2, 3], [4, 5, 6]])
        # 修改数组元素值
        a[0, 0] = 8
        # 断言 a 的类型为 np.ndarray
        assert isinstance(a, np.ndarray)
        # 断言 a 数组等于指定的值
        assert_equal(a, [[8, 2, 3], [4, 5, 6]])


# 定义测试类 TestReshape，继承自 TestCase
class TestReshape(TestCase):
    
    # 标记为条件跳过的测试用例，当 TEST_WITH_TORCHDYNAMO 为 True 时跳过
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_reshape_function(self):
        # 创建一个二维列表
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # 期望的重塑后结果
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        # 使用 np.reshape 函数进行重塑，并进行断言
        assert np.all(np.reshape(arr, (2, 6)) == tgt)

        # 将 arr 转换为 numpy 数组
        arr = np.asarray(arr)
        # 断言 np.transpose(arr, (1, 0)).tensor._base 是 arr.tensor
        assert np.transpose(arr, (1, 0)).tensor._base is arr.tensor

    # 标记为条件跳过的测试用例，当 TEST_WITH_TORCHDYNAMO 为 True 时跳过
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_reshape_method(self):
        # 创建一个 numpy 数组
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        # 记录 arr 的原始形状
        arr_shape = arr.shape

        # 期望的重塑后结果
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

        # 使用 arr.reshape(2, 6) 进行重塑，并进行断言
        assert np.all(arr.reshape(2, 6) == tgt)
        # 断言 arr.reshape(2, 6).tensor._base 是 arr.tensor，重塑保持基本张量
        assert arr.reshape(2, 6).tensor._base is arr.tensor
        # 断言 arr 的形状与原始形状相同
        assert arr.shape == arr_shape

        # XXX: move out to dedicated test(s)
        # 断言 arr.reshape(2, 6).tensor._base 是 arr.tensor
        assert arr.reshape(2, 6).tensor._base is arr.tensor

        # 使用 arr.reshape((2, 6)) 进行重塑，并进行断言
        assert np.all(arr.reshape((2, 6)) == tgt)
        # 断言 arr.reshape((2, 6)).tensor._base 是 arr.tensor
        assert arr.reshape((2, 6)).tensor._base is arr.tensor
        # 断言 arr 的形状与原始形状相同
        assert arr.shape == arr_shape

        # 期望的重塑后结果
        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        # 使用 arr.reshape(3, 4) 进行重塑，并进行断言
        assert np.all(arr.reshape(3, 4) == tgt)
        # 断言 arr.reshape(3, 4).tensor._base 是 arr.tensor
        assert arr.reshape(3, 4).tensor._base is arr.tensor
        # 断言 arr 的形状与原始形状相同
        assert arr.shape == arr_shape

        # 使用 arr.reshape((3, 4)) 进行重塑，并进行断言
        assert np.all(arr.reshape((3, 4)) == tgt)
        # 断言 arr.reshape((3, 4)).tensor._base 是 arr.tensor
        assert arr.reshape((3, 4)).tensor._base is arr.tensor
        # 断言 arr 的形状与原始形状相同


# XXX : order='C' / 'F'
#        tgt = [[1, 4, 7, 10],
#               [2, 5, 8, 11],
#               [3, 6, 9, 12]]
#        assert np.all(arr.T.reshape((3, 4), order='C') == tgt)
#
#        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
#        assert_equal(arr.reshape((3, 4), order='F'), tgt)
#


# 定义测试类 TestTranspose，继承自 TestCase
class TestTranspose(TestCase):
    pass  # 该类暂无测试方法，保留为后续扩展
    # 如果条件 TEST_WITH_TORCHDYNAMO 为真，则跳过测试，原因是 ".tensor attribute"
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    # 定义测试函数 test_transpose_function
    def test_transpose_function(self):
        # 创建一个二维数组 arr
        arr = [[1, 2], [3, 4], [5, 6]]
        # 创建一个目标数组 tgt，表示 arr 的转置结果
        tgt = [[1, 3, 5], [2, 4, 6]]
        # 断言 np.transpose(arr, (1, 0)) 的结果与 tgt 相等
        assert_equal(np.transpose(arr, (1, 0)), tgt)

        # 将 arr 转换为 NumPy 数组
        arr = np.asarray(arr)
        # 断言 np.transpose(arr, (1, 0)).tensor._base 与 arr.tensor._base 相同
        assert np.transpose(arr, (1, 0)).tensor._base is arr.tensor

    # 如果条件 TEST_WITH_TORCHDYNAMO 为真，则跳过测试，原因是 ".tensor attribute"
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    # 定义测试函数 test_transpose_method
    def test_transpose_method(self):
        # 创建一个二维数组 a
        a = np.array([[1, 2], [3, 4]])
        # 断言 a.transpose() 的结果与 [[1, 3], [2, 4]] 相等
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        # 断言 a.transpose(None) 的结果与 [[1, 3], [2, 4]] 相等
        assert_equal(a.transpose(None), [[1, 3], [2, 4]])
        # 断言 lambda 表达式调用时抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0))
        # 断言 lambda 表达式调用时抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 0))
        # 断言 lambda 表达式调用时抛出 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 1, 2))

        # 断言 a.transpose().tensor._base 与 a.tensor._base 相同
        assert a.transpose().tensor._base is a.tensor
class TestRavel(TestCase):
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_ravel_function(self):
        # 定义二维列表
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # 目标展平结果
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # 断言 np.ravel 函数是否按预期工作
        assert_equal(np.ravel(a), tgt)

        # 将二维列表转换为 numpy 数组
        arr = np.asarray(a)
        # 断言 numpy 数组的 ravel 方法是否按预期工作
        assert np.ravel(arr).tensor._base is arr.tensor

    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_ravel_method(self):
        # 创建二维 numpy 数组
        a = np.array([[0, 1], [2, 3]])
        # 断言 numpy 数组的 ravel 方法是否按预期工作
        assert_equal(a.ravel(), [0, 1, 2, 3])

        # 断言 numpy 数组的 ravel 方法生成的结果的 tensor 属性是否正确
        assert a.ravel().tensor._base is a.tensor


class TestNonzero(TestCase):
    def test_nonzero_trivial(self):
        # 空数组的 nonzero 测试
        assert_equal(np.nonzero(np.array([])), ([],))
        assert_equal(np.array([]).nonzero(), ([],))

        # 只有一个零元素的数组的 nonzero 测试
        assert_equal(np.nonzero(np.array([0])), ([],))
        assert_equal(np.array([0]).nonzero(), ([],))

        # 只有一个非零元素的数组的 nonzero 测试
        assert_equal(np.nonzero(np.array([1])), ([0],))
        assert_equal(np.array([1]).nonzero(), ([0],))

    def test_nonzero_onedim(self):
        # 一维数组的 nonzero 测试
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))
        assert_equal(x.nonzero(), ([0, 2, 3, 6],))

    def test_nonzero_twodim(self):
        # 二维数组的 nonzero 测试
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))
        assert_equal(x.nonzero(), ([0, 1, 1], [1, 0, 2]))

        # 对角线为 1 的单位矩阵的 nonzero 测试
        x = np.eye(3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))
        assert_equal(x.nonzero(), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # 稀疏矩阵的 nonzero 测试
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))
            assert_equal(c.nonzero()[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_array_method(self):
        # 测试数组对象方法中的 nonzero 方法
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)


@instantiate_parametrized_tests
class TestArgmaxArgminCommon(TestCase):
    sizes = [
        (),
        (3,),
        (3, 2),
        (2, 3),
        (3, 3),
        (2, 3, 4),
        (4, 3, 2),
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
        (64,),
        (128,),
        (256,),
    ]

    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails on NumPy 1.21.x")
    @parametrize(
        "size, axis",
        list(
            itertools.chain(
                *[
                    [
                        (size, axis)
                        for axis in list(range(-len(size), len(size))) + [None]
                    ]
                    for size in sizes
                ]
            )
        ),
    )
    @parametrize("method", [np.argmax, np.argmin])
    def test_np_argmin_argmax_keepdims(self, size, axis, method):
        # 创建一个空的 NumPy 数组，形状由参数 size 决定
        arr = np.empty(shape=size)

        # 处理连续的数组
        if axis is None:
            # 如果 axis 为 None，则将数组形状设为全为 1 的形式
            new_shape = [1 for _ in range(len(size))]
        else:
            # 否则，根据指定的 axis 将其对应位置设置为 1，其它位置与原数组 size 保持一致
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        # 执行 method 函数对数组 arr 进行操作，将结果 reshape 成 new_shape 形状
        _res_orig = method(arr, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        # 再次执行 method 函数，保持维度，结果存储在 res 中
        res = method(arr, axis=axis, keepdims=True)
        # 断言保持维度后的结果与 reshape 后的结果相等
        assert_equal(res, res_orig)
        # 断言保持维度后的结果的形状与 new_shape 相等
        assert res.shape == new_shape

        # 创建一个与 res 相同形状的空数组 outarray
        outarray = np.empty(res.shape, dtype=res.dtype)
        # 将 method 函数应用于 arr，结果存储在 outarray 中，保持维度
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        # 断言 res1 和 outarray 是同一个对象
        assert res1 is outarray
        # 断言 res1 与 res 相等
        assert_equal(res, outarray)

        if len(size) > 0:
            # 如果数组 size 的长度大于 0
            wrong_shape = list(new_shape)
            if axis is not None:
                # 如果 axis 不为 None，则将对应位置设置为 2
                wrong_shape[axis] = 2
            else:
                # 否则将第一个位置设置为 2
                wrong_shape[0] = 2
            # 创建一个与 wrong_shape 相同形状的空数组 wrong_outarray
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 断言执行 method 函数时，传递错误形状的数组会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

        # 处理非连续的数组
        if axis is None:
            # 如果 axis 为 None，则将数组形状设为全为 1 的形式
            new_shape = [1 for _ in range(len(size))]
        else:
            # 否则，将 size 的反向列表赋值给 new_shape，并在指定的 axis 位置设置为 1
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        # 执行 method 函数对数组 arr 的转置进行操作，结果存储在 _res_orig 中，并 reshape 成 new_shape 形状
        _res_orig = method(arr.T, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        # 再次执行 method 函数，保持维度，结果存储在 res 中
        res = method(arr.T, axis=axis, keepdims=True)
        # 断言保持维度后的结果与 reshape 后的结果相等
        assert_equal(res, res_orig)
        # 断言保持维度后的结果的形状与 new_shape 相等
        assert res.shape == new_shape
        # 创建一个与 new_shape 反向的形状的空数组 outarray，并对其转置
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T
        # 将 method 函数应用于 arr 的转置，结果存储在 outarray 中，保持维度
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        # 断言 res1 和 outarray 是同一个对象
        assert res1 is outarray
        # 断言 res1 与 res 相等
        assert_equal(res, outarray)

        if len(size) > 0:
            # 如果数组 size 的长度大于 0
            # 如果 axis 不为 None，则将对应位置设置为 2
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                # 否则将第一个位置设置为 2
                wrong_shape[0] = 2
            # 创建一个与 wrong_shape 相同形状的空数组 wrong_outarray
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 断言执行 method 函数时，传递错误形状的数组会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)
    # 定义一个测试方法，用于测试数组的所有元素
    def test_all(self, method):
        # 创建一个 5 维的 numpy 数组，包含从 0 到 (4*5*6*7*8-1) 的所有整数
        a = np.arange(4 * 5 * 6 * 7 * 8).reshape((4, 5, 6, 7, 8))
        # 获取指定方法（如 'argmax' 或 'argmin'）的函数引用
        arg_method = getattr(a, "arg" + method)
        # 获取指定方法（如 'max' 或 'min'）的函数引用
        val_method = getattr(a, method)
        # 遍历数组的维度
        for i in range(a.ndim):
            # 使用指定方法获取每个维度的最大值或最小值
            a_maxmin = val_method(i)
            # 使用指定方法获取每个维度最大值或最小值的索引
            aarg_maxmin = arg_method(i)
            # 创建一个包含除当前维度以外的所有维度索引的列表
            axes = list(range(a.ndim))
            axes.remove(i)
            # 使用 choose 函数验证最大值或最小值索引的正确性
            assert np.all(a_maxmin == aarg_maxmin.choose(*a.transpose(i, *axes)))

    @parametrize("method", ["argmax", "argmin"])
    # 定义一个参数化测试方法，测试输出的形状
    def test_output_shape(self, method):
        # 创建一个形状为 (10, 5) 的全为 1 的数组
        a = np.ones((10, 5))
        # 获取指定方法（如 'argmax' 或 'argmin'）的函数引用
        arg_method = getattr(a, method)

        # 检查一些简单的形状不匹配情况
        out = np.ones(11, dtype=np.int_)

        # 使用 assert_raises 检测是否抛出 ValueError 异常
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        out = np.ones((2, 5), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        # 这些情况可能可以放宽（以前版本甚至允许）
        out = np.ones((1, 10), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        # 创建一个形状为 (10,) 的全为 1 的数组
        out = np.ones(10, dtype=np.int_)
        # 使用指定方法获取全局最大值或最小值的索引
        arg_method(-1, out=out)
        # 检查输出数组是否等于使用指定方法获取的索引
        assert_equal(out, arg_method(-1))

    @parametrize("ndim", [0, 1])
    @parametrize("method", ["argmax", "argmin"])
    # 定义一个参数化测试方法，测试返回值是否是输出值
    def test_ret_is_out(self, ndim, method):
        # 创建一个形状为 (4, 256^ndim) 的全为 1 的数组
        a = np.ones((4,) + (256,) * ndim)
        # 获取指定方法（如 'argmax' 或 'argmin'）的函数引用
        arg_method = getattr(a, method)
        # 创建一个形状为 (256^ndim,) 的空数组
        out = np.empty((256,) * ndim, dtype=np.intp)
        # 使用指定方法获取全局最大值或最小值的索引，并将结果保存在 out 中
        ret = arg_method(axis=0, out=out)
        # 断言返回的索引数组和 out 相同
        assert ret is out

    @parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    # 定义一个参数化测试方法，测试 ndarray 方法与 numpy 方法的一致性
    def test_np_vs_ndarray(self, arr_method, np_method):
        # 创建一个形状为 (2, 3) 的数组，包含从 0 到 5 的所有整数
        a = np.arange(6).reshape((2, 3))
        # 获取指定方法（如 'argmax' 或 'argmin'）的 ndarray 方法引用
        arg_method = getattr(a, arr_method)

        # 检查关键字参数
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        # 使用 ndarray 方法和 numpy 方法分别计算最大值或最小值的索引，并比较结果
        assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
        # 断言 out1 和 out2 相等
        assert_equal(out1, out2)

    @parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    # 定义一个参数化测试方法，测试 ndarray 方法与 numpy 方法的一致性（使用位置参数）
    def test_np_vs_ndarray_positional(self, arr_method, np_method):
        # 创建一个形状为 (2, 3) 的数组，包含从 0 到 5 的所有整数
        a = np.arange(6).reshape((2, 3))
        # 获取指定方法（如 'argmax' 或 'argmin'）的 ndarray 方法引用
        arg_method = getattr(a, arr_method)

        # 检查位置参数
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        # 使用 ndarray 方法和 numpy 方法分别计算最大值或最小值的索引，并比较结果
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        # 断言 out1 和 out2 相等
        assert_equal(out1, out2)
@instantiate_parametrized_tests
class TestArgmax(TestCase):
    # 定义使用例数据（usg_data），包含多个测试案例，每个案例是一个列表和一个预期结果
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),
        ([3, 3, 3, 3, 2, 2, 2, 2], 0),
        ([0, 1, 2, 3, 4, 5, 6, 7], 7),
        ([7, 6, 5, 4, 3, 2, 1, 0], 0),
    ]
    
    # 定义特殊组合例数据（sg_data），包含usg_data和额外的测试案例
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),
        ([1, 2, 3, 4, -1, -2, -3, -4], 3),
    ]
    
    # 创建数据数组（darr），包含多个元组，每个元组由一个numpy数组和一个预期结果组成
    darr = [
        # 从usg_data中选择每个列表，转换为给定的数据类型t的numpy数组，并与预期结果d[1]一起组成元组
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    
    # 将sg_data中的每个列表，转换为不同的数据类型t的numpy数组，并与预期结果d[1]一起组成元组，添加到darr中
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    
    # 将特殊情况的数据组合，每个情况包含一个元组，由列表和预期结果组成，转换为浮点数类型t的numpy数组，并添加到darr中
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # 用于触发SIMD多级（x4, x1）内部循环的尾部
                    # 在不同SIMD宽度的变体上
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    # 定义包含各种测试数据的列表，包括 NaN 和复杂数值
    nan_arr = darr + [
        # 使用 subtest 函数生成测试数据，带有特定装饰器 xpassIfTorchDynamo
        subtest(([0, 1, 2, 3, complex(0, np.nan)], 4), decorators=[xpassIfTorchDynamo]),
        subtest(([0, 1, 2, 3, complex(np.nan, 0)], 4), decorators=[xpassIfTorchDynamo]),
        subtest(([0, 1, 2, complex(np.nan, 0), 3], 3), decorators=[xpassIfTorchDynamo]),
        subtest(([0, 1, 2, complex(0, np.nan), 3], 3), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(0, np.nan), 0, 1, 2, 3], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(np.nan, np.nan), 0, 1, 2, 3], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(0, 0), complex(0, 2), complex(0, 1)], 1), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(1, 0), complex(0, 2), complex(0, 1)], 0), decorators=[xpassIfTorchDynamo]),
        subtest(([complex(1, 0), complex(0, 2), complex(1, 1)], 2), decorators=[xpassIfTorchDynamo]),
        # 测试数据包含布尔数组和期望的最大值位置
        ([False, False, False, False, True], 4),
        ([False, False, False, True, False], 3),
        ([True, False, False, False, False], 0),
        ([True, False, True, False, False], 0),
    ]

    # 使用 parametrize 装饰器将 nan_arr 列表中的每个数据元组作为参数传入 test_combinations 函数
    @parametrize("data", nan_arr)
    def test_combinations(self, data):
        # 解包测试数据元组
        arr, pos = data

        # 求 arr 中的最大值
        val = np.max(arr)

        # 断言最大值的索引位置等于预期位置 pos
        assert_equal(np.argmax(arr), pos)  # , err_msg="%r" % arr)
        # 断言 arr 中最大值等于预期值 val
        assert_equal(arr[np.argmax(arr)], val)  # , err_msg="%r" % arr)

        # 将 arr 扩展 129 倍，以测试 SIMD 循环
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        # 断言扩展后的 arr 的最大值索引位置等于 rpos
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言扩展后的 arr 的最大值等于 val
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")

        # 创建与 arr 最小值重复 513 次的 padd 数组，并将其与 arr 连接起来
        padd = np.repeat(np.min(arr), 513)
        rarr = np.concatenate((arr, padd))
        rpos = pos
        # 断言连接后的 arr 的最大值索引位置等于 rpos
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言连接后的 arr 的最大值等于 val
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")
    # 定义一个测试方法，用于测试最大有符号整数的情况
    def test_maximum_signed_integers(self):
        # 创建一个包含最大值、最小值和一个常规值的 int8 类型的 NumPy 数组
        a = np.array([1, 2**7 - 1, -(2**7)], dtype=np.int8)
        # 断言最大值的索引为 1
        assert_equal(np.argmax(a), 1)

        # 创建一个包含最大值、最小值和一个常规值的 int16 类型的 NumPy 数组
        a = np.array([1, 2**15 - 1, -(2**15)], dtype=np.int16)
        # 断言最大值的索引为 1
        assert_equal(np.argmax(a), 1)

        # 创建一个包含最大值、最小值和一个常规值的 int32 类型的 NumPy 数组
        a = np.array([1, 2**31 - 1, -(2**31)], dtype=np.int32)
        # 断言最大值的索引为 1
        assert_equal(np.argmax(a), 1)

        # 创建一个包含最大值、最小值和一个常规值的 int64 类型的 NumPy 数组
        a = np.array([1, 2**63 - 1, -(2**63)], dtype=np.int64)
        # 断言最大值的索引为 1
        assert_equal(np.argmax(a), 1)
@instantiate_parametrized_tests
class TestArgmin(TestCase):
    # 定义包含多个参数化测试数据的类，用于测试 argmin 函数
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),  # 示例测试数据 1
        ([3, 3, 3, 3, 2, 2, 2, 2], 4),  # 示例测试数据 2
        ([0, 1, 2, 3, 4, 5, 6, 7], 0),  # 示例测试数据 3
        ([7, 6, 5, 4, 3, 2, 1, 0], 7),  # 示例测试数据 4
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),  # 扩展的示例测试数据 5
        ([1, 2, 3, 4, -1, -2, -3, -4], 7),  # 扩展的示例测试数据 6
    ]
    darr = [
        (np.array(d[0], dtype=t), d[1])  # 创建包含 usg_data 的 NumPy 数组
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])  # 创建包含 sg_data 的 NumPy 数组
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])  # 创建其他复杂测试数据的 NumPy 数组
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widthes
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    nan_arr = darr + [
        subtest(([0, 1, 2, 3, complex(0, np.nan)], 4), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([0, 1, 2, 3, complex(np.nan, 0)], 4), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([0, 1, 2, complex(np.nan, 0), 3], 3), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([0, 1, 2, complex(0, np.nan), 3], 3), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([complex(0, np.nan), 0, 1, 2, 3], 0), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([complex(np.nan, np.nan), 0, 1, 2, 3], 0), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xfail],  # 包含 NaN 的子测试
        ),
        subtest(
            ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xfail],  # 包含 NaN 的子测试
        ),
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
            decorators=[xfail],  # 包含 NaN 的子测试
        ),
        subtest(([complex(0, 0), complex(0, 2), complex(0, 1)], 0), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([complex(1, 0), complex(0, 2), complex(0, 1)], 2), decorators=[xfail]),  # 包含 NaN 的子测试
        subtest(([complex(1, 0), complex(0, 2), complex(1, 1)], 1), decorators=[xfail]),  # 包含 NaN 的子测试
        ([True, True, True, True, False], 4),  # 示例测试数据 7
        ([True, True, True, False, True], 3),  # 示例测试数据 8
        ([False, True, True, True, True], 0),  # 示例测试数据 9
        ([False, True, False, True, True], 0),  # 示例测试数据 10
    ]

    @parametrize("data", nan_arr)
    # 参数化装饰器，用于将 nan_arr 中的测试数据传递给被装饰的测试函数
    # 测试组合函数，接收一个数据元组 (arr, pos)
    def test_combinations(self, data):
        # 解包数据元组
        arr, pos = data

        # 如果数组 arr 中的元素类型为复数，则标记测试为失败，给出原因
        if np.asarray(arr).dtype.kind in "c":
            pytest.xfail(reason="'min_values_cpu' not implemented for 'ComplexDouble'")

        # 计算数组 arr 中的最小值
        min_val = np.min(arr)

        # 断言：数组 arr 中最小值的索引应与 pos 相等，如果不相等则输出错误信息
        assert_equal(np.argmin(arr), pos, err_msg=f"{arr!r}")
        
        # 断言：数组 arr 中最小值的值应与计算出的最小值 min_val 相等，如果不相等则输出错误信息
        assert_equal(arr[np.argmin(arr)], min_val, err_msg=f"{arr!r}")

        # 添加填充以测试 SIMD 循环效果
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        
        # 断言：扩展后数组 rarr 中最小值的索引应与 rpos 相等，如果不相等则输出错误信息
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        
        # 断言：扩展后数组 rarr 中最小值的值应与计算出的最小值 min_val 相等，如果不相等则输出错误信息
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

        # 创建填充数组 padd，其中的元素是 arr 的最大值，重复 513 次
        padd = np.repeat(np.max(arr), 513)
        # 将填充数组 padd 与数组 arr 连接起来，形成新的数组 rarr
        rarr = np.concatenate((arr, padd))
        rpos = pos
        
        # 断言：连接后数组 rarr 中最小值的索引应与 rpos 相等，如果不相等则输出错误信息
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        
        # 断言：连接后数组 rarr 中最小值的值应与计算出的最小值 min_val 相等，如果不相等则输出错误信息
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

    # 测试各种有符号整数的最小值索引
    def test_minimum_signed_integers(self):
        # 测试 int8 类型数组的最小值索引
        a = np.array([1, -(2**7), -(2**7) + 1, 2**7 - 1], dtype=np.int8)
        assert_equal(np.argmin(a), 1)

        # 测试 int16 类型数组的最小值索引
        a = np.array([1, -(2**15), -(2**15) + 1, 2**15 - 1], dtype=np.int16)
        assert_equal(np.argmin(a), 1)

        # 测试 int32 类型数组的最小值索引
        a = np.array([1, -(2**31), -(2**31) + 1, 2**31 - 1], dtype=np.int32)
        assert_equal(np.argmin(a), 1)

        # 测试 int64 类型数组的最小值索引
        a = np.array([1, -(2**63), -(2**63) + 1, 2**63 - 1], dtype=np.int64)
        assert_equal(np.argmin(a), 1)
class TestAmax(TestCase):
    # 定义测试类 TestAmax，继承自 TestCase
    def test_basic(self):
        # 定义测试方法 test_basic
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 定义列表 a 包含整数和浮点数
        assert_equal(np.amax(a), 10.0)
        # 断言 a 的最大值等于 10.0
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        # 定义二维列表 b
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        # 断言 b 按列的最大值分别为 [8.0, 10.0, 9.0]
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])
        # 断言 b 按行的最大值分别为 [9.0, 10.0, 8.0]

        arr = np.asarray(a)
        # 将列表 a 转换为 numpy 数组 arr
        assert_equal(np.amax(arr), arr.max())
        # 断言 arr 的最大值等于 arr 的最大值


class TestAmin(TestCase):
    # 定义测试类 TestAmin，继承自 TestCase
    def test_basic(self):
        # 定义测试方法 test_basic
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 定义列表 a 包含整数和浮点数
        assert_equal(np.amin(a), -5.0)
        # 断言 a 的最小值等于 -5.0
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        # 定义二维列表 b
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        # 断言 b 按列的最小值分别为 [3.0, 3.0, 2.0]
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])
        # 断言 b 按行的最小值分别为 [3.0, 4.0, 2.0]

        arr = np.asarray(a)
        # 将列表 a 转换为 numpy 数组 arr
        assert_equal(np.amin(arr), arr.min())
        # 断言 arr 的最小值等于 arr 的最小值


class TestContains(TestCase):
    # 定义测试类 TestContains，继承自 TestCase
    def test_contains(self):
        # 定义测试方法 test_contains
        a = np.arange(12).reshape(3, 4)
        # 创建一个 3x4 的 numpy 数组 a
        assert 2 in a
        # 断言 2 是否在数组 a 中
        assert 42 not in a
        # 断言 42 是否不在数组 a 中


@instantiate_parametrized_tests
class TestNoExtraMethods(TestCase):
    # 定义测试类 TestNoExtraMethods，继承自 TestCase，并使用装饰器 instantiate_parametrized_tests
    # 确保 ndarray 不包含额外的方法/属性
    # >>> set(dir(a)) - set(dir(a.tensor.numpy()))
    @parametrize("name", ["fn", "ivar", "method", "name", "plain", "rvar"])
    # 使用参数化测试装饰器 parametrize，测试不合法的属性访问
    def test_extra_methods(self, name):
        # 定义测试方法 test_extra_methods，接受参数 name
        a = np.ones(3)
        # 创建一个元素全为 1 的长度为 3 的 numpy 数组 a
        with pytest.raises(AttributeError):
            getattr(a, name)
            # 使用 getattr 尝试获取属性 name，期望抛出 AttributeError 异常


class TestIter(TestCase):
    # 定义测试类 TestIter，继承自 TestCase
    @skipIfTorchDynamo()
    # 使用装饰器 skipIfTorchDynamo，条件为真时跳过测试
    def test_iter_1d(self):
        # 定义测试方法 test_iter_1d
        # numpy 生成数组标量，我们要测试0维数组
        a = np.arange(5)
        # 创建一个包含 0 到 4 的 numpy 数组 a
        lst = list(a)
        # 将数组 a 转换为列表 lst
        assert all(type(x) == np.ndarray for x in lst), f"{[type(x) for x in lst]}"
        # 断言 lst 中所有元素的类型均为 numpy 数组
        assert all(x.ndim == 0 for x in lst)
        # 断言 lst 中所有数组的维度均为 0

    def test_iter_2d(self):
        # 定义测试方法 test_iter_2d
        # numpy 沿着第0轴迭代
        a = np.arange(5)[None, :]
        # 创建一个形状为 (1, 5) 的 numpy 数组 a
        lst = list(a)
        # 将数组 a 转换为列表 lst
        assert len(lst) == 1
        # 断言 lst 的长度为 1
        assert type(lst[0]) == np.ndarray
        # 断言 lst 的第一个元素的类型为 numpy 数组
        assert_equal(lst[0], np.arange(5))
        # 断言 lst 的第一个元素与 np.arange(5) 相等


if __name__ == "__main__":
    run_tests()
    # 如果作为主程序运行，则执行测试运行函数 run_tests()
```