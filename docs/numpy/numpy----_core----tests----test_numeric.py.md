# `.\numpy\numpy\_core\tests\test_numeric.py`

```py
# 导入系统模块
import sys
# 导入警告模块
import warnings
# 导入迭代工具模块
import itertools
# 导入平台信息模块
import platform
# 导入 pytest 测试框架
import pytest
# 导入数学模块
import math
# 导入 Decimal 类
from decimal import Decimal

# 导入 NumPy 库并指定别名 np
import numpy as np
# 导入 NumPy 的核心部分
from numpy._core import umath, sctypes
# 导入对象到数据类型转换函数
from numpy._core.numerictypes import obj2sctype
# 导入轴异常类
from numpy.exceptions import AxisError
# 导入随机数生成函数
from numpy.random import rand, randint, randn
# 导入 NumPy 的测试工具集
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_warns, assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )
# 导入 NumPy 的有理数测试模块
from numpy._core._rational_tests import rational
# 导入 NumPy 的掩码数组模块
from numpy import ma

# 导入假设测试框架的给定和策略模块
from hypothesis import given, strategies as st
# 导入假设测试框架的额外 NumPy 扩展模块
from hypothesis.extra import numpy as hynp


# 定义一个测试类 TestResize
class TestResize:
    # 定义测试方法 test_copies
    def test_copies(self):
        # 创建一个二维数组 A
        A = np.array([[1, 2], [3, 4]])
        # 预期的数组 Ar1，通过 resize 将 A 调整为 (2, 4) 大小
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar1 相等
        assert_equal(np.resize(A, (2, 4)), Ar1)

        # 预期的数组 Ar2，通过 resize 将 A 调整为 (4, 2) 大小
        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar2 相等
        assert_equal(np.resize(A, (4, 2)), Ar2)

        # 预期的数组 Ar3，通过 resize 将 A 调整为 (4, 3) 大小
        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar3 相等
        assert_equal(np.resize(A, (4, 3)), Ar3)

    # 定义测试方法 test_repeats
    def test_repeats(self):
        # 创建一个一维数组 A
        A = np.array([1, 2, 3])
        # 预期的数组 Ar1，通过 resize 将 A 调整为 (2, 4) 大小
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar1 相等
        assert_equal(np.resize(A, (2, 4)), Ar1)

        # 预期的数组 Ar2，通过 resize 将 A 调整为 (4, 2) 大小
        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar2 相等
        assert_equal(np.resize(A, (4, 2)), Ar2)

        # 预期的数组 Ar3，通过 resize 将 A 调整为 (4, 3) 大小
        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        # 使用 assert_equal 断言 resize 的结果与预期数组 Ar3 相等
        assert_equal(np.resize(A, (4, 3)), Ar3)

    # 定义测试方法 test_zeroresize
    def test_zeroresize(self):
        # 创建一个二维数组 A
        A = np.array([[1, 2], [3, 4]])
        # 调整 A 的大小为 (0,)
        Ar = np.resize(A, (0,))
        # 使用 assert_array_equal 断言 Ar 与空数组相等
        assert_array_equal(Ar, np.array([]))
        # 使用 assert_equal 断言 Ar 的数据类型与 A 相同
        assert_equal(A.dtype, Ar.dtype)

        # 调整 A 的大小为 (0, 2)
        Ar = np.resize(A, (0, 2))
        # 使用 assert_equal 断言 Ar 的形状为 (0, 2)
        assert_equal(Ar.shape, (0, 2))

        # 调整 A 的大小为 (2, 0)
        Ar = np.resize(A, (2, 0))
        # 使用 assert_equal 断言 Ar 的形状为 (2, 0)
        assert_equal(Ar.shape, (2, 0))

    # 定义测试方法 test_reshape_from_zero
    def test_reshape_from_zero(self):
        # 创建一个零长度的结构化数组 A，数据类型为单个浮点数字段
        A = np.zeros(0, dtype=[('a', np.float32)])
        # 调整 A 的大小为 (2, 1)
        Ar = np.resize(A, (2, 1))
        # 使用 assert_array_equal 断言 Ar 与形状为 (2, 1) 的零数组相等，数据类型与 A 相同
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
        # 使用 assert_equal 断言 Ar 的数据类型与 A 相同
        assert_equal(A.dtype, Ar.dtype)

    # 定义测试方法 test_negative_resize
    def test_negative_resize(self):
        # 创建一个从 0 到 9 的浮点数数组 A
        A = np.arange(0, 10, dtype=np.float32)
        # 设置新的形状为负数
        new_shape = (-10, -1)
        # 使用 pytest.raises 检查 resize 函数是否引发 ValueError 异常，异常信息匹配 "negative"
        with pytest.raises(ValueError, match=r"negative"):
            np.resize(A, new_shape=new_shape)

    # 定义测试方法 test_subclass
    def test_subclass(self):
        # 定义一个继承自 np.ndarray 的子类 MyArray
        class MyArray(np.ndarray):
            __array_priority__ = 1.

        # 创建一个 MyArray 类的视图 my_arr，包含一个整数 1
        my_arr = np.array([1]).view(MyArray)
        # 使用 assert 断言 resize 函数返回的类型与 MyArray 相同
        assert type(np.resize(my_arr, 5)) is MyArray
        # 使用 assert 断言 resize 函数返回的类型与 MyArray 相同
        assert type(np.resize(my_arr, 0)) is MyArray

        # 创建一个空数组的 MyArray 类的视图 my_arr
        my_arr = np.array([]).view(MyArray)
        # 使用 assert 断言 resize 函数返回的类型与 MyArray 相同
        assert type(np.resize(my_arr, 5)) is MyArray

# 定义一个测试类 TestNonarrayArgs
class TestNonarrayArgs:
    # 检查函数非数组参数是否被包装在数组中的测试方法 test_choose
    def test_choose(self):
        # 定义一个选择列表 choices
        choices = [[0, 1, 2],
                   [3, 4, 5],
                   [5, 6, 7]]
        # 目标列表 tgt
        tgt = [5, 1, 5]
        # 一个整数列表 a
        a = [2, 0, 1]

        # 使用 np.choose 函数，从 choices 中选择元素填充到 a 中
        out = np.choose(a, choices)
        # 使用 assert_equal 断言 out 与目标列表 tgt 相等
    # 定义一个测试方法，测试 np.clip() 函数的功能
    def test_clip(self):
        # 创建一个包含整数的列表
        arr = [-1, 5, 2, 3, 10, -4, -9]
        # 使用 np.clip() 将数组中小于2的元素设为2，大于7的元素设为7
        out = np.clip(arr, 2, 7)
        # 预期的输出结果
        tgt = [2, 5, 2, 3, 7, 2, 2]
        # 使用 assert_equal() 断言 out 和 tgt 相等
        assert_equal(out, tgt)

    # 定义一个测试方法，测试 np.compress() 函数的功能
    def test_compress(self):
        # 创建一个二维列表
        arr = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
        # 压缩数组，只保留索引为 0 和 1 的行
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        # 使用 assert_equal() 断言 out 和 tgt 相等
        assert_equal(out, tgt)

    # 定义一个测试方法，测试 np.count_nonzero() 函数的功能
    def test_count_nonzero(self):
        # 创建一个二维列表
        arr = [[0, 1, 7, 0, 0],
               [3, 0, 0, 2, 19]]
        # 计算每行非零元素的个数
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        # 使用 assert_equal() 断言 out 和 tgt 相等
        assert_equal(out, tgt)

    # 定义一个测试方法，测试 np.diagonal() 函数的功能
    def test_diagonal(self):
        # 创建一个二维列表
        a = [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11]]
        # 获取数组的对角线元素
        out = np.diagonal(a)
        # 预期的对角线元素列表
        tgt = [0, 5, 10]
        # 使用 assert_equal() 断言 out 和 tgt 相等
        assert_equal(out, tgt)

    # 定义一个测试方法，测试 np.mean() 函数的功能
    def test_mean(self):
        # 创建一个二维列表
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言整体平均值是否等于 3.5
        assert_(np.mean(A) == 3.5)
        # 断言按列计算平均值是否与给定数组相等
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        # 断言按行计算平均值是否与给定数组相等
        assert_(np.all(np.mean(A, 1) == np.array([2., 5.])))
        
        # 捕获 RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 断言空数组的平均值是否为 NaN
            assert_(np.isnan(np.mean([])))
            # 断言警告类型是否为 RuntimeWarning
            assert_(w[0].category is RuntimeWarning)

    # 定义一个测试方法，测试 np.ptp() 函数的功能
    def test_ptp(self):
        # 创建一个列表
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 计算数组元素的峰值到峰值（最大值与最小值的差）
        assert_equal(np.ptp(a, axis=0), 15.0)

    # 定义一个测试方法，测试 np.prod() 函数的功能
    def test_prod(self):
        # 创建一个二维列表
        arr = [[1, 2, 3, 4],
               [5, 6, 7, 9],
               [10, 3, 4, 5]]
        # 沿着最后一个轴计算数组元素的乘积
        tgt = [24, 1890, 600]
        assert_equal(np.prod(arr, axis=-1), tgt)

    # 定义一个测试方法，测试 np.ravel() 函数的功能
    def test_ravel(self):
        # 创建一个二维列表
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # 将多维数组展平为一维数组
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    # 定义一个测试方法，测试 np.repeat() 函数的功能
    def test_repeat(self):
        # 创建一个列表
        a = [1, 2, 3]
        # 将数组中的每个元素重复两次
        tgt = [1, 1, 2, 2, 3, 3]
        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    # 定义一个测试方法，测试 np.reshape() 函数的功能
    def test_reshape(self):
        # 创建一个二维列表
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # 将数组重新调整为指定的形状
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    # 定义一个测试方法，测试 np.reshape() 函数的 shape 参数
    def test_reshape_shape_arg(self):
        # 创建一个包含 0 到 11 的数组
        arr = np.arange(12)
        shape = (3, 4)
        expected = arr.reshape(shape)
        
        # 测试错误处理，验证不能同时指定 newshape 和 shape 参数
        with pytest.raises(
            TypeError,
            match="You cannot specify 'newshape' and 'shape' "
                  "arguments at the same time."
        ):
            np.reshape(arr, shape=shape, newshape=shape)
        
        # 测试错误处理，验证缺少 shape 参数时的处理
        with pytest.raises(
            TypeError,
            match=r"reshape\(\) missing 1 required positional "
                  "argument: 'shape'"
        ):
            np.reshape(arr)
        
        # 断言使用 shape 参数的各种方式
        assert_equal(np.reshape(arr, shape), expected)
        assert_equal(np.reshape(arr, shape, order="C"), expected)
        assert_equal(np.reshape(arr, shape=shape), expected)
        assert_equal(np.reshape(arr, shape=shape, order="C"), expected)
        
        # 测试警告处理，验证使用 newshape 参数的警告
        with pytest.warns(DeprecationWarning):
            actual = np.reshape(arr, newshape=shape)
            assert_equal(actual, expected)
    # 定义一个测试方法，用于测试 numpy 数组的 reshape 方法的不同参数组合
    def test_reshape_copy_arg(self):
        # 创建一个形状为 (2, 3, 4) 的 numpy 数组，包含 24 个元素
        arr = np.arange(24).reshape(2, 3, 4)
        # 创建一个按列主序存储的 arr 的拷贝
        arr_f_ord = np.array(arr, order="F")
        # 定义一个新的形状 (12, 2)
        shape = (12, 2)

        # 检查 np.reshape 是否与原数组共享内存
        assert np.shares_memory(np.reshape(arr, shape), arr)
        # 检查 np.reshape 使用 order="C" 是否与原数组共享内存
        assert np.shares_memory(np.reshape(arr, shape, order="C"), arr)
        # 检查 np.reshape 使用 order="F" 是否与按列主序的 arr 共享内存
        assert np.shares_memory(
            np.reshape(arr_f_ord, shape, order="F"), arr_f_ord)
        # 检查 np.reshape 使用 copy=None 是否与原数组共享内存
        assert np.shares_memory(np.reshape(arr, shape, copy=None), arr)
        # 检查 np.reshape 使用 copy=False 是否与原数组共享内存
        assert np.shares_memory(np.reshape(arr, shape, copy=False), arr)
        # 检查 ndarray 对象的实例方法 reshape 使用 copy=False 是否与原数组共享内存
        assert np.shares_memory(arr.reshape(shape, copy=False), arr)
        # 检查 np.reshape 使用 copy=True 是否未与原数组共享内存
        assert not np.shares_memory(np.reshape(arr, shape, copy=True), arr)
        # 检查 np.reshape 使用 order="C", copy=True 是否未与原数组共享内存
        assert not np.shares_memory(
            np.reshape(arr, shape, order="C", copy=True), arr)
        # 检查 np.reshape 使用 order="F", copy=True 是否未与原数组共享内存
        assert not np.shares_memory(
            np.reshape(arr, shape, order="F", copy=True), arr)
        # 检查 np.reshape 使用 order="F", copy=None 是否未与原数组共享内存
        assert not np.shares_memory(
            np.reshape(arr, shape, order="F", copy=None), arr)

        # 定义一个错误消息，用于检查在使用 order="F", copy=False 时是否会引发 ValueError
        err_msg = "Unable to avoid creating a copy while reshaping."
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=err_msg):
            np.reshape(arr, shape, order="F", copy=False)
        with pytest.raises(ValueError, match=err_msg):
            np.reshape(arr_f_ord, shape, order="C", copy=False)

    # 定义一个测试方法，用于测试 numpy 的 around 方法
    def test_round(self):
        # 创建一个浮点数数组 arr
        arr = [1.56, 72.54, 6.35, 3.25]
        # 创建一个预期的结果数组 tgt
        tgt = [1.6, 72.5, 6.4, 3.2]
        # 检查 np.around 是否正确地将 arr 中的元素四舍五入到指定小数位数
        assert_equal(np.around(arr, decimals=1), tgt)
        # 创建一个 np.float64 类型的标量 s
        s = np.float64(1.)
        # 检查 s.round() 返回的类型是否为 np.float64
        assert_(isinstance(s.round(), np.float64))
        # 检查 s.round() 是否正确返回四舍五入后的整数值 1
        assert_equal(s.round(), 1.)

    # 使用 pytest 的参数化装饰器定义一个测试方法，用于测试不同数据类型的 round 函数
    @pytest.mark.parametrize('dtype', [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ])
    def test_dunder_round(self, dtype):
        # 创建一个指定数据类型的标量 s
        s = dtype(1)
        # 检查 round(s) 返回的类型是否为 int
        assert_(isinstance(round(s), int))
        # 检查 round(s, None) 返回的类型是否为 int
        assert_(isinstance(round(s, None), int))
        # 检查 round(s, ndigits=None) 返回的类型是否为 int
        assert_(isinstance(round(s, ndigits=None), int))
        # 检查 round(s) 是否正确返回整数值 1
        assert_equal(round(s), 1)
        # 检查 round(s, None) 是否正确返回整数值 1
        assert_equal(round(s, None), 1)
        # 检查 round(s, ndigits=None) 是否正确返回整数值 1
        assert_equal(round(s, ndigits=None), 1)

    # 使用 pytest 的参数化装饰器定义一个测试方法，用于测试 round 函数的边界情况
    @pytest.mark.parametrize('val, ndigits', [
        # 定义一个参数化测试用例，跳过由于超出 int32 范围引发的异常
        pytest.param(2**31 - 1, -1,
            marks=pytest.mark.skip(reason="Out of range of int32")
        ),
        # 定义一个正常的参数化测试用例，用于测试大整数的四舍五入
        (2**31 - 1, 1-math.ceil(math.log10(2**31 - 1))),
        # 定义一个正常的参数化测试用例，用于测试大整数的四舍五入
        (2**31 - 1, -math.ceil(math.log10(2**31 - 1)))
    ])
    def test_dunder_round_edgecases(self, val, ndigits):
        # 检查 round(val, ndigits) 是否正确返回预期的结果
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))
    # 测试 np.float64 的精度和 round 函数的使用
    def test_dunder_round_accuracy(self):
        # 创建一个 np.float64 类型的浮点数 f
        f = np.float64(5.1 * 10**73)
        # 断言 round(f, -73) 的返回类型是 np.float64
        assert_(isinstance(round(f, -73), np.float64))
        # 断言 round(f, -73) 的值与 5.0 * 10**73 在数值上相等
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        # 断言 round(f, ndigits=-73) 的返回类型是 np.float64
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        # 断言 round(f, ndigits=-73) 的值与 5.0 * 10**73 在数值上相等
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        # 创建一个 np.int64 类型的整数 i
        i = np.int64(501)
        # 断言 round(i, -2) 的返回类型是 np.int64
        assert_(isinstance(round(i, -2), np.int64))
        # 断言 round(i, -2) 的值与 500 在数值上相等
        assert_array_max_ulp(round(i, -2), 500)
        # 断言 round(i, ndigits=-2) 的返回类型是 np.int64
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        # 断言 round(i, ndigits=-2) 的值与 500 在数值上相等
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    # 使用 pytest 的标记 @pytest.mark.xfail 来标记测试为预期失败，并添加失败的原因
    @pytest.mark.xfail(raises=AssertionError, reason="gh-15896")
    def test_round_py_consistency(self):
        # 创建一个普通的浮点数 f
        f = 5.1 * 10**73
        # 断言 round(np.float64(f), -73) 的结果与 round(f, -73) 的结果在数值上相等
        assert_equal(round(np.float64(f), -73), round(f, -73))

    # 测试 np.searchsorted 函数的使用
    def test_searchsorted(self):
        # 创建一个有序数组 arr
        arr = [-8, -5, -1, 3, 6, 10]
        # 使用 np.searchsorted 在 arr 中查找数字 0 的插入点并返回
        out = np.searchsorted(arr, 0)
        # 断言插入点的位置为 3
        assert_equal(out, 3)

    # 测试 np.size 函数的使用
    def test_size(self):
        # 创建一个二维数组 A
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言 np.size(A) 的结果为 6
        assert_(np.size(A) == 6)
        # 断言 np.size(A, 0) 的结果为 2
        assert_(np.size(A, 0) == 2)
        # 断言 np.size(A, 1) 的结果为 3
        assert_(np.size(A, 1) == 3)

    # 测试 np.squeeze 函数的使用
    def test_squeeze(self):
        # 创建一个三维数组 A
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        # 断言 np.squeeze(A) 的形状为 (3, 3)
        assert_equal(np.squeeze(A).shape, (3, 3))
        # 断言 np.squeeze(np.zeros((1, 3, 1))) 的形状为 (3,)
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        # 断言 np.squeeze(np.zeros((1, 3, 1)), axis=0) 的形状为 (3, 1)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        # 断言 np.squeeze(np.zeros((1, 3, 1)), axis=-1) 的形状为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        # 断言 np.squeeze(np.zeros((1, 3, 1)), axis=2) 的形状为 (1, 3)
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        # 断言 np.squeeze([np.zeros((3, 1))]) 的形状为 (3,)
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        # 断言 np.squeeze([np.zeros((3, 1))], axis=0) 的形状为 (3, 1)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        # 断言 np.squeeze([np.zeros((3, 1))], axis=2) 的形状为 (1, 3)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        # 断言 np.squeeze([np.zeros((3, 1))], axis=-1) 的形状为 (1, 3)
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    # 测试 np.std 函数的使用
    def test_std(self):
        # 创建一个二维数组 A
        A = [[1, 2, 3], [4, 5, 6]]
        # 断言 np.std(A) 的结果约等于 1.707825127659933
        assert_almost_equal(np.std(A), 1.707825127659933)
        # 断言 np.std(A, 0) 的结果为 [1.5, 1.5, 1.5]
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        # 断言 np.std(A, 1) 的结果为 [0.81649658, 0.81649658]
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))

        # 使用 warnings 捕获 RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 断言 np.std([]) 返回 NaN
            assert_(np.isnan(np.std([])))
            # 断言捕获到的第一个 warning 是 RuntimeWarning 类型
            assert_(w[0].category is RuntimeWarning)

    # 测试 np.swapaxes 函数的使用
    def test_swapaxes(self):
        # 创建两个三维数组 tgt 和 a
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        # 使用 np.swapaxes 将 a 的第 0 轴和第 2 轴交换
        out = np.swapaxes(a, 0, 2)
        # 断言交换后的数组 out 与目标数组 tgt 在数值上相等
        assert_equal(out, tgt)

    # 测试 np.sum 函数的使用
    def test_sum(self):
        # 创建一个二维数组 m
        m = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        # 使用 np.sum 对 m 按行求和并保持维度
        out = np.sum(m, axis=1, keepdims=True)
        # 断言求和后的数组 out 与目标数组 tgt 在数值上相等
        assert_equal(tgt, out)
    def test_take(self):
        tgt = [2, 3, 5]  # 目标输出列表，表示预期的取值结果
        indices = [1, 2, 4]  # 索引列表，指定从数组 `a` 中取值的索引
        a = [1, 2, 3, 4, 5]  # 原始数组

        out = np.take(a, indices)  # 使用索引从数组 `a` 中取值
        assert_equal(out, tgt)  # 断言取出的值与预期目标值列表相等

        pairs = [
            (np.int32, np.int32), (np.int32, np.int64),
            (np.int64, np.int32), (np.int64, np.int64)
        ]
        for array_type, indices_type in pairs:
            x = np.array([1, 2, 3, 4, 5], dtype=array_type)  # 创建指定类型的数组 `x`
            ind = np.array([0, 2, 2, 3], dtype=indices_type)  # 创建指定类型的索引数组 `ind`
            tgt = np.array([1, 3, 3, 4], dtype=array_type)  # 预期的取值结果数组
            out = np.take(x, ind)  # 使用索引从数组 `x` 中取值
            assert_equal(out, tgt)  # 断言取出的值与预期目标值数组相等
            assert_equal(out.dtype, tgt.dtype)  # 断言取出的值的数据类型与预期目标值数组的数据类型相等  

    def test_trace(self):
        c = [[1, 2], [3, 4], [5, 6]]  # 输入的二维数组 `c`
        assert_equal(np.trace(c), 5)  # 断言计算 `c` 的迹（主对角线元素之和）为 5

    def test_transpose(self):
        arr = [[1, 2], [3, 4], [5, 6]]  # 输入的二维数组 `arr`
        tgt = [[1, 3, 5], [2, 4, 6]]  # 预期的转置后的二维数组
        assert_equal(np.transpose(arr, (1, 0)), tgt)  # 断言通过指定轴的顺序进行转置后与预期结果相等
        assert_equal(np.matrix_transpose(arr), tgt)  # 断言使用 `matrix_transpose` 函数转置后与预期结果相等

    def test_var(self):
        A = [[1, 2, 3], [4, 5, 6]]  # 输入的二维数组 `A`
        assert_almost_equal(np.var(A), 2.9166666666666665)  # 断言计算 `A` 的总体方差与预期值近似相等
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))  # 断言计算 `A` 按列的方差与预期结果数组近似相等
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))  # 断言计算 `A` 按行的方差与预期结果数组近似相等

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.var([])))  # 断言计算空数组的方差会产生 RuntimeWarning
            assert_(w[0].category is RuntimeWarning)  # 断言捕获的警告类型为 RuntimeWarning

        B = np.array([None, 0])  # 输入的数组 `B` 包含 None 和 0
        B[0] = 1j  # 将 `B` 的第一个元素设置为虚数
        assert_almost_equal(np.var(B), 0.25)  # 断言计算 `B` 的方差与预期值近似相等

    def test_std_with_mean_keyword(self):
        # 设置随机数种子以使测试结果可重现
        rng = np.random.RandomState(1234)
        A = rng.randn(10, 20, 5) + 0.5  # 生成具有指定均值的随机数组 `A`

        mean_out = np.zeros((10, 1, 5))  # 形状为 (10, 1, 5) 的零数组，用于接收均值
        std_out = np.zeros((10, 1, 5))  # 形状为 (10, 1, 5) 的零数组，用于接收标准差

        mean = np.mean(A,
                       out=mean_out,
                       axis=1,
                       keepdims=True)  # 计算数组 `A` 指定轴上的均值，并将结果存入 `mean_out`

        assert mean_out is mean  # 断言返回的均值对象与指定的对象相同

        std = np.std(A,
                     out=std_out,
                     axis=1,
                     keepdims=True,
                     mean=mean)  # 计算数组 `A` 指定轴上的标准差，并将结果存入 `std_out`，使用给定的均值

        assert std_out is std  # 断言返回的标准差对象与指定的对象相同

        assert std.shape == mean.shape  # 断言返回的均值和标准差的形状相同
        assert std.shape == (10, 1, 5)  # 断言返回的均值和标准差的形状为 (10, 1, 5)

        std_old = np.std(A, axis=1, keepdims=True)  # 计算数组 `A` 指定轴上的标准差

        assert std_old.shape == mean.shape  # 断言返回的旧标准差的形状与均值的形状相同
        assert_almost_equal(std, std_old)  # 断言返回的新旧标准差的值近似相等
    # 定义一个测试函数，用于测试带有 `mean` 关键字参数的变量
    def test_var_with_mean_keyword(self):
        # 设置随机种子以保证测试结果的可复现性
        rng = np.random.RandomState(1234)
        # 生成一个形状为 (10, 20, 5) 的随机数组 A，并将其每个元素增加 0.5
        A = rng.randn(10, 20, 5) + 0.5

        # 创建一个与 mean 相同形状的全零数组 mean_out
        mean_out = np.zeros((10, 1, 5))
        # 创建一个与 var 相同形状的全零数组 var_out
        var_out = np.zeros((10, 1, 5))

        # 计算数组 A 在 axis=1 上的均值，将结果存储在 mean_out 中，并保持维度为 1
        mean = np.mean(A,
                       out=mean_out,
                       axis=1,
                       keepdims=True)

        # 断言 mean_out 与 mean 是同一个对象
        assert mean_out is mean

        # 计算数组 A 在 axis=1 上的方差，将结果存储在 var_out 中，并保持维度为 1
        # 使用先前计算的均值 mean 作为参数传入
        var = np.var(A,
                     out=var_out,
                     axis=1,
                     keepdims=True,
                     mean=mean)

        # 断言 var_out 与 var 是同一个对象
        assert var_out is var

        # 断言 mean 和 var 的形状相同且为 (10, 1, 5)
        assert var.shape == mean.shape
        assert var.shape == (10, 1, 5)

        # 使用单独算法计算数组 A 在 axis=1 上的方差，与先前计算的均值 mean 无关
        var_old = np.var(A, axis=1, keepdims=True)

        # 断言 var_old 和 mean 的形状相同
        assert var_old.shape == mean.shape
        # 断言 var 与 var_old 的近似相等
        assert_almost_equal(var, var_old)

    # 定义一个测试函数，测试带有 `mean` 关键字参数且 `keepdims=False` 的标准差计算
    def test_std_with_mean_keyword_keepdims_false(self):
        rng = np.random.RandomState(1234)
        A = rng.randn(10, 20, 5) + 0.5

        # 计算数组 A 在 axis=1 上的均值，保持维度不变
        mean = np.mean(A,
                       axis=1,
                       keepdims=True)

        # 计算数组 A 在 axis=1 上的标准差，不保持维度，使用先前计算的均值 mean 作为参数传入
        std = np.std(A,
                     axis=1,
                     keepdims=False,
                     mean=mean)

        # 断言 std 的形状应为 (10, 5)
        assert std.shape == (10, 5)

        # 使用单独算法计算数组 A 在 axis=1 上的标准差和均值，不保持维度
        std_old = np.std(A, axis=1, keepdims=False)
        mean_old = np.mean(A, axis=1, keepdims=False)

        # 断言 std_old 和 mean_old 的形状相同
        assert std_old.shape == mean_old.shape
        # 断言 std 和 std_old 的值相等
        assert_equal(std, std_old)

    # 定义一个测试函数，测试带有 `mean` 关键字参数且 `keepdims=False` 的方差计算
    def test_var_with_mean_keyword_keepdims_false(self):
        rng = np.random.RandomState(1234)
        A = rng.randn(10, 20, 5) + 0.5

        # 计算数组 A 在 axis=1 上的均值，保持维度不变
        mean = np.mean(A,
                       axis=1,
                       keepdims=True)

        # 计算数组 A 在 axis=1 上的方差，不保持维度，使用先前计算的均值 mean 作为参数传入
        var = np.var(A,
                     axis=1,
                     keepdims=False,
                     mean=mean)

        # 断言 var 的形状应为 (10, 5)
        assert var.shape == (10, 5)

        # 使用单独算法计算数组 A 在 axis=1 上的方差和均值，不保持维度
        var_old = np.var(A, axis=1, keepdims=False)
        mean_old = np.mean(A, axis=1, keepdims=False)

        # 断言 var_old 和 mean_old 的形状相同
        assert var_old.shape == mean_old.shape
        # 断言 var 和 var_old 的值相等
        assert_equal(var, var_old)
    def test_std_with_mean_keyword_where_nontrivial(self):
        # 使用种子1234创建一个随机数生成器
        rng = np.random.RandomState(1234)
        # 生成一个形状为(10, 20, 5)的随机数组A，均值为0.5
        A = rng.randn(10, 20, 5) + 0.5

        # 创建一个布尔数组where，标记数组A中大于0.5的元素位置
        where = A > 0.5

        # 计算数组A在第1轴上的均值，保持维度，并且只考虑where为True的元素
        mean = np.mean(A,
                       axis=1,
                       keepdims=True,
                       where=where)

        # 计算数组A在第1轴上的标准差，不保持维度，同时指定均值和where条件
        std = np.std(A,
                     axis=1,
                     keepdims=False,
                     mean=mean,
                     where=where)

        # 断言标准差数组std的形状应为(10, 5)
        assert std.shape == (10, 5)

        # 断言std与使用单独算法计算的标准差std_old相等
        std_old = np.std(A, axis=1, where=where)
        mean_old = np.mean(A, axis=1, where=where)
        assert_equal(std, std_old)

    def test_var_with_mean_keyword_where_nontrivial(self):
        # 使用种子1234创建一个随机数生成器
        rng = np.random.RandomState(1234)
        # 生成一个形状为(10, 20, 5)的随机数组A，均值为0.5
        A = rng.randn(10, 20, 5) + 0.5

        # 创建一个布尔数组where，标记数组A中大于0.5的元素位置
        where = A > 0.5

        # 计算数组A在第1轴上的均值，保持维度，并且只考虑where为True的元素
        mean = np.mean(A,
                       axis=1,
                       keepdims=True,
                       where=where)

        # 计算数组A在第1轴上的方差，不保持维度，同时指定均值和where条件
        var = np.var(A,
                     axis=1,
                     keepdims=False,
                     mean=mean,
                     where=where)

        # 断言方差数组var的形状应为(10, 5)
        assert var.shape == (10, 5)

        # 断言var与使用单独算法计算的方差var_old相等
        var_old = np.var(A, axis=1, where=where)
        mean_old = np.mean(A, axis=1, where=where)
        assert_equal(var, var_old)

    def test_std_with_mean_keyword_multiple_axis(self):
        # 设置种子以便测试可重现性
        rng = np.random.RandomState(1234)
        # 生成一个形状为(10, 20, 5)的随机数组A，均值为0.5
        A = rng.randn(10, 20, 5) + 0.5

        # 指定多轴axis为(0, 2)
        axis = (0, 2)

        # 计算数组A在指定轴axis上的均值，保持维度
        mean = np.mean(A,
                       out=None,
                       axis=axis,
                       keepdims=True)

        # 计算数组A在指定轴axis上的标准差，不保持维度，并指定均值
        std = np.std(A,
                     out=None,
                     axis=axis,
                     keepdims=False,
                     mean=mean)

        # 断言标准差数组std的形状应为(20,)
        assert std.shape == (20,)

        # 断言std与使用单独算法计算的标准差std_old相等
        std_old = np.std(A, axis=axis, keepdims=False)
        assert_almost_equal(std, std_old)
    def test_std_with_mean_keyword_axis_None(self):
        # 设置随机种子以确保测试可重现性
        rng = np.random.RandomState(1234)
        # 生成一个形状为 (10, 20, 5) 的随机数组，并加上偏移量 0.5
        A = rng.randn(10, 20, 5) + 0.5

        axis = None

        # 计算 A 的均值，axis=None 表示在所有维度上求均值，keepdims=True 保持维度不变
        mean = np.mean(A,
                       out=None,
                       axis=axis,
                       keepdims=True)

        # 计算 A 的标准差，axis=None 表示在所有维度上求标准差，mean 参数使用之前计算的均值
        std = np.std(A,
                     out=None,
                     axis=axis,
                     keepdims=False,
                     mean=mean)

        # 返回的标准差应该是一个标量
        assert std.shape == ()

        # 检查计算的标准差是否与单独算法计算的结果几乎相等
        std_old = np.std(A, axis=axis, keepdims=False)
        assert_almost_equal(std, std_old)

    def test_std_with_mean_keyword_keepdims_true_masked(self):

        # 创建一个带掩码的数组 A
        A = ma.array([[2., 3., 4., 5.],
                      [1., 2., 3., 4.]],
                     mask=[[True, False, True, False],
                           [True, False, True, False]])

        # 创建一个带掩码的数组 B
        B = ma.array([[100., 3., 104., 5.],
                      [101., 2., 103., 4.]],
                     mask=[[True, False, True, False],
                           [True, False, True, False]])

        # 创建一个输出数组 mean_out 和 std_out，用于存储均值和标准差的计算结果
        mean_out = ma.array([[0., 0., 0., 0.]],
                            mask=[[False, False, False, False]])
        std_out = ma.array([[0., 0., 0., 0.]],
                           mask=[[False, False, False, False]])

        axis = 0

        # 计算 A 的均值，axis=0 表示沿着第一个维度计算均值，keepdims=True 保持维度不变
        mean = np.mean(A, out=mean_out,
                       axis=axis, keepdims=True)

        # 计算 A 的标准差，axis=0 表示沿着第一个维度计算标准差，mean 参数使用之前计算的均值，keepdims=True 保持维度不变
        std = np.std(A, out=std_out,
                     axis=axis, keepdims=True,
                     mean=mean)

        # 返回的均值和标准差的形状应该相同
        assert std.shape == mean.shape
        assert std.shape == (1, 4)

        # 检查计算的均值和标准差是否与单独算法计算的结果几乎相等
        std_old = np.std(A, axis=axis, keepdims=True)
        mean_old = np.mean(A, axis=axis, keepdims=True)
        assert std_old.shape == mean_old.shape
        assert_almost_equal(std, std_old)
        assert_almost_equal(mean, mean_old)

        # 检查输出的均值和标准差对象是否与原始对象相同
        assert mean_out is mean
        assert std_out is std

        # 检查带掩码的元素是否被忽略
        mean_b = np.mean(B, axis=axis, keepdims=True)
        std_b = np.std(B, axis=axis, keepdims=True, mean=mean_b)
        assert_almost_equal(std, std_b)
        assert_almost_equal(mean, mean_b)
    # 定义一个测试函数，用于测试带有特定参数的方差和均值计算的行为
    def test_var_with_mean_keyword_keepdims_true_masked(self):

        # 创建一个带有掩码的掩码数组 A
        A = ma.array([[2., 3., 4., 5.],
                      [1., 2., 3., 4.]],
                     mask=[[True, False, True, False],
                           [True, False, True, False]])

        # 创建另一个带有掩码的掩码数组 B，与 A 形状相同
        B = ma.array([[100., 3., 104., 5.],
                      [101., 2., 103., 4.]],
                      mask=[[True, False, True, False],
                            [True, False, True, False]])

        # 创建一个输出数组 mean_out，用于存储均值的结果
        mean_out = ma.array([[0., 0., 0., 0.]],
                            mask=[[False, False, False, False]])

        # 创建一个输出数组 var_out，用于存储方差的结果
        var_out = ma.array([[0., 0., 0., 0.]],
                           mask=[[False, False, False, False]])

        # 设置计算的轴向为 0
        axis = 0

        # 使用 ma 数组 A 计算均值，结果存储在 mean_out 中，保持维度为 True
        mean = np.mean(A, out=mean_out,
                       axis=axis, keepdims=True)

        # 使用 ma 数组 A 计算方差，结果存储在 var_out 中，保持维度为 True，并且使用先前计算的均值 mean
        var = np.var(A, out=var_out,
                     axis=axis, keepdims=True,
                     mean=mean)

        # 断言均值和方差的形状应该相同，为 (1, 4)
        assert var.shape == mean.shape
        assert var.shape == (1, 4)

        # 断言使用 np.var 和 np.mean 单独计算的旧方差和均值与新计算的结果几乎相等
        var_old = np.var(A, axis=axis, keepdims=True)
        mean_old = np.mean(A, axis=axis, keepdims=True)

        assert var_old.shape == mean_old.shape
        assert_almost_equal(var, var_old)
        assert_almost_equal(mean, mean_old)

        # 断言 mean_out 和 mean 指向相同的对象
        assert mean_out is mean
        # 断言 var_out 和 var 指向相同的对象
        assert var_out is var

        # 断言掩码元素应该被忽略
        mean_b = np.mean(B, axis=axis, keepdims=True)
        var_b = np.var(B, axis=axis, keepdims=True, mean=mean_b)
        assert_almost_equal(var, var_b)
        assert_almost_equal(mean, mean_b)
class TestIsscalar:
    def test_isscalar(self):
        # 断言：3.1 是标量
        assert_(np.isscalar(3.1))
        # 断言：np.int16(12345) 是标量
        assert_(np.isscalar(np.int16(12345)))
        # 断言：False 是标量
        assert_(np.isscalar(False))
        # 断言：'numpy' 是标量
        assert_(np.isscalar('numpy'))
        # 断言：[3.1] 不是标量
        assert_(not np.isscalar([3.1]))
        # 断言：None 不是标量
        assert_(not np.isscalar(None))

        # PEP 3141 标准
        from fractions import Fraction
        # 断言：Fraction(5, 17) 是标量
        assert_(np.isscalar(Fraction(5, 17)))
        from numbers import Number
        # 断言：Number() 是标量
        assert_(np.isscalar(Number()))


class TestBoolScalar:
    def test_logical(self):
        f = np.False_
        t = np.True_
        s = "xyz"
        # 断言：True and "xyz" 返回 "xyz"
        assert_((t and s) is s)
        # 断言：False and "xyz" 返回 False
        assert_((f and s) is f)

    def test_bitwise_or(self):
        f = np.False_
        t = np.True_
        # 断言：True | True 返回 True
        assert_((t | t) is t)
        # 断言：False | True 返回 True
        assert_((f | t) is t)
        # 断言：True | False 返回 True
        assert_((t | f) is t)
        # 断言：False | False 返回 False
        assert_((f | f) is f)

    def test_bitwise_and(self):
        f = np.False_
        t = np.True_
        # 断言：True & True 返回 True
        assert_((t & t) is t)
        # 断言：False & True 返回 False
        assert_((f & t) is f)
        # 断言：True & False 返回 False
        assert_((t & f) is f)
        # 断言：False & False 返回 False
        assert_((f & f) is f)

    def test_bitwise_xor(self):
        f = np.False_
        t = np.True_
        # 断言：True ^ True 返回 False
        assert_((t ^ t) is f)
        # 断言：False ^ True 返回 True
        assert_((f ^ t) is t)
        # 断言：True ^ False 返回 True
        assert_((t ^ f) is t)
        # 断言：False ^ False 返回 False
        assert_((f ^ f) is f)


class TestBoolArray:
    def setup_method(self):
        # 设置用于 SIMD 测试的偏移量
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False

    def test_all_any(self):
        # 断言：self.t 中所有元素为 True
        assert_(self.t.all())
        # 断言：self.t 中至少一个元素为 True
        assert_(self.t.any())
        # 断言：self.f 中所有元素为 False
        assert_(not self.f.all())
        # 断言：self.f 中至少一个元素为 False
        assert_(not self.f.any())
        # 断言：self.nm 中至少一个元素为 True
        assert_(self.nm.any())
        # 断言：self.im 中至少一个元素为 True
        assert_(self.im.any())
        # 断言：self.nm 中所有元素不全为 True
        assert_(not self.nm.all())
        # 断言：self.im 中所有元素不全为 True
        assert_(not self.im.all())

        # 检查所有位置的错误元素
        for i in range(256 - 7):
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            # 断言：d 中至少一个元素为 True
            assert_(np.any(d))
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            # 断言：e 中所有元素不全为 True
            assert_(not np.all(e))
            # 断言：e 等于 d 的按位取反
            assert_array_equal(e, ~d)

        # 大数组测试，用于测试被阻塞的 libc 循环
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            # 断言：d 中至少一个元素为 True，附加消息为当前位置 i
            assert_(np.any(d), msg="%r" % i)
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            # 断言：e 中所有元素不全为 True，附加消息为当前位置 i
            assert_(not np.all(e), msg="%r" % i)
    # 测试逻辑非操作符 ~ 的功能
    def test_logical_not_abs(self):
        # 断言按位取反后的数组与预期的假数组相等
        assert_array_equal(~self.t, self.f)
        # 断言对按位取反后的数组执行绝对值操作与预期的假数组相等
        assert_array_equal(np.abs(~self.t), self.f)
        # 断言对按位取反后的假数组执行绝对值操作与预期的真数组相等
        assert_array_equal(np.abs(~self.f), self.t)
        # 断言对真数组执行绝对值操作与预期的真数组相等
        assert_array_equal(np.abs(self.f), self.f)
        # 断言对绝对值后的假数组执行按位取反操作与预期的真数组相等
        assert_array_equal(~np.abs(self.f), self.t)
        # 断言对绝对值后的真数组执行按位取反操作与预期的假数组相等
        assert_array_equal(~np.abs(self.t), self.f)
        # 断言对按位取反后的非零数组执行绝对值操作与预期的非零数组相等
        assert_array_equal(np.abs(~self.nm), self.im)
        # 使用 np.logical_not 函数对真数组进行按位取反，将结果输出到预定义的数组 self.o 中
        np.logical_not(self.t, out=self.o)
        # 断言按位取反后的结果数组与预期的假数组相等
        assert_array_equal(self.o, self.f)
        # 使用 np.abs 函数对真数组进行绝对值操作，将结果输出到预定义的数组 self.o 中
        np.abs(self.t, out=self.o)
        # 断言绝对值后的结果数组与预期的真数组相等
        assert_array_equal(self.o, self.t)

    # 测试逻辑与、逻辑或和逻辑异或操作符 & | ^ 的功能
    def test_logical_and_or_xor(self):
        # 断言真与真的逻辑或操作与预期的真数组相等
        assert_array_equal(self.t | self.t, self.t)
        # 断言假与假的逻辑或操作与预期的假数组相等
        assert_array_equal(self.f | self.f, self.f)
        # 断言真与假的逻辑或操作与预期的真数组相等
        assert_array_equal(self.t | self.f, self.t)
        # 断言假与真的逻辑或操作与预期的真数组相等
        assert_array_equal(self.f | self.t, self.t)
        # 使用 np.logical_or 函数对两个真数组进行逻辑或操作，将结果输出到预定义的数组 self.o 中
        np.logical_or(self.t, self.t, out=self.o)
        # 断言逻辑或操作后的结果数组与预期的真数组相等
        assert_array_equal(self.o, self.t)
        # 断言真与真的逻辑与操作与预期的真数组相等
        assert_array_equal(self.t & self.t, self.t)
        # 断言假与假的逻辑与操作与预期的假数组相等
        assert_array_equal(self.f & self.f, self.f)
        # 断言真与假的逻辑与操作与预期的假数组相等
        assert_array_equal(self.t & self.f, self.f)
        # 断言假与真的逻辑与操作与预期的假数组相等
        assert_array_equal(self.f & self.t, self.f)
        # 使用 np.logical_and 函数对两个真数组进行逻辑与操作，将结果输出到预定义的数组 self.o 中
        np.logical_and(self.t, self.t, out=self.o)
        # 断言逻辑与操作后的结果数组与预期的真数组相等
        assert_array_equal(self.o, self.t)
        # 断言真与真的逻辑异或操作与预期的假数组相等
        assert_array_equal(self.t ^ self.t, self.f)
        # 断言假与假的逻辑异或操作与预期的假数组相等
        assert_array_equal(self.f ^ self.f, self.f)
        # 断言真与假的逻辑异或操作与预期的真数组相等
        assert_array_equal(self.t ^ self.f, self.t)
        # 断言假与真的逻辑异或操作与预期的真数组相等
        assert_array_equal(self.f ^ self.t, self.t)
        # 使用 np.logical_xor 函数对两个真数组进行逻辑异或操作，将结果输出到预定义的数组 self.o 中
        np.logical_xor(self.t, self.t, out=self.o)
        # 断言逻辑异或操作后的结果数组与预期的假数组相等
        assert_array_equal(self.o, self.f)

        # 断言非零数组与真数组的逻辑与操作结果与预期的非零数组相等
        assert_array_equal(self.nm & self.t, self.nm)
        # 断言非零整数与假数组的逻辑与操作结果为 False
        assert_array_equal(self.im & self.f, False)
        # 断言非零数组与 True 的逻辑与操作结果与预期的非零数组相等
        assert_array_equal(self.nm & True, self.nm)
        # 断言非零整数与 False 的逻辑与操作结果与预期的假数组相等
        assert_array_equal(self.im & False, self.f)
        # 断言非零数组与真数组的逻辑或操作结果与预期的真数组相等
        assert_array_equal(self.nm | self.t, self.t)
        # 断言非零整数与假数组的逻辑或操作结果与预期的非零整数相等
        assert_array_equal(self.im | self.f, self.im)
        # 断言非零数组与 True 的逻辑或操作结果为 True
        assert_array_equal(self.nm | True, self.t)
        # 断言非零整数与 False 的逻辑或操作结果与预期的非零整数相等
        assert_array_equal(self.im | False, self.im)
        # 断言非零数组与真数组的逻辑异或操作结果与预期的非零整数相等
        assert_array_equal(self.nm ^ self.t, self.im)
        # 断言非零整数与假数组的逻辑异或操作结果与预期的非零整数相等
        assert_array_equal(self.im ^ self.f, self.im)
        # 断言非零数组与 True 的逻辑异或操作结果与预期的非零整数相等
        assert_array_equal(self.nm ^ True, self.im)
        # 断言非零整数与 False 的逻辑异或操作结果与预期的非零整数相等
        assert_array_equal(self.im ^ False, self.im)
# 定义一个测试类 TestBoolCmp，用于布尔比较的测试
class TestBoolCmp:
    # 设置每个测试方法的初始化操作
    def setup_method(self):
        # 创建一个长度为256的全1浮点数组 self.f
        self.f = np.ones(256, dtype=np.float32)
        # 创建一个与 self.f 同样长度的全1布尔数组 self.ef
        self.ef = np.ones(self.f.size, dtype=bool)
        # 创建一个长度为128的全1双精度浮点数组 self.d
        self.d = np.ones(128, dtype=np.float64)
        # 创建一个与 self.d 同样长度的全1布尔数组 self.ed
        self.ed = np.ones(self.d.size, dtype=bool)
        
        # 为 self.f 和 self.ef 分别生成所有256位SIMD向量的排列值
        s = 0
        for i in range(32):
            self.f[s:s+8] = [i & 2**x for x in range(8)]
            self.ef[s:s+8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        
        # 为 self.d 和 self.ed 分别生成所有128位SIMD向量的排列值
        s = 0
        for i in range(16):
            self.d[s:s+4] = [i & 2**x for x in range(4)]
            self.ed[s:s+4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        # 复制 self.f 和 self.d 到新的数组 self.nf 和 self.nd
        self.nf = self.f.copy()
        self.nd = self.d.copy()
        
        # 将 self.nf 中在 self.ef 中对应位置为 True 的元素设置为 NaN
        self.nf[self.ef] = np.nan
        # 将 self.nd 中在 self.ed 中对应位置为 True 的元素设置为 NaN
        self.nd[self.ed] = np.nan

        # 复制 self.f 和 self.d 到新的数组 self.inff 和 self.infd
        self.inff = self.f.copy()
        self.infd = self.d.copy()
        
        # 设置 self.inff 中每隔3个位置的元素，对应在 self.ef 中为 True 的元素为正无穷
        self.inff[::3][self.ef[::3]] = np.inf
        # 设置 self.infd 中每隔3个位置的元素，对应在 self.ed 中为 True 的元素为正无穷
        self.infd[::3][self.ed[::3]] = np.inf
        # 设置 self.inff 中每隔3个位置的元素，对应在 self.ef 中为 True 的元素为负无穷
        self.inff[1::3][self.ef[1::3]] = -np.inf
        # 设置 self.infd 中每隔3个位置的元素，对应在 self.ed 中为 True 的元素为负无穷
        self.infd[1::3][self.ed[1::3]] = -np.inf
        # 设置 self.inff 中每隔3个位置的元素，对应在 self.ef 中为 True 的元素为 NaN
        self.inff[2::3][self.ef[2::3]] = np.nan
        # 设置 self.infd 中每隔3个位置的元素，对应在 self.ed 中为 True 的元素为 NaN
        self.infd[2::3][self.ed[2::3]] = np.nan
        
        # 复制 self.ef 到新的数组 self.efnonan
        self.efnonan = self.ef.copy()
        # 将 self.efnonan 中每隔3个位置的元素置为 False
        self.efnonan[2::3] = False
        # 复制 self.ed 到新的数组 self.ednonan
        self.ednonan = self.ed.copy()
        # 将 self.ednonan 中每隔3个位置的元素置为 False
        self.ednonan[2::3] = False

        # 复制 self.f 和 self.d 到新的数组 self.signf 和 self.signd
        self.signf = self.f.copy()
        self.signd = self.d.copy()
        
        # 将 self.signf 中在 self.ef 中对应位置为 True 的元素乘以 -1
        self.signf[self.ef] *= -1.
        # 将 self.signd 中在 self.ed 中对应位置为 True 的元素乘以 -1
        self.signd[self.ed] *= -1.
        
        # 设置 self.signf 中每隔6个位置的元素，对应在 self.ef 中为 True 的元素为负无穷
        self.signf[1::6][self.ef[1::6]] = -np.inf
        # 设置 self.signd 中每隔6个位置的元素，对应在 self.ed 中为 True 的元素为负无穷
        self.signd[1::6][self.ed[1::6]] = -np.inf
        
        # 对于非 RISC-V 架构，进行特定的 NaN 测试
        if platform.machine() != 'riscv64':
            # 设置 self.signf 中每隔6个位置的元素，对应在 self.ef 中为 True 的元素为负 NaN
            self.signf[3::6][self.ef[3::6]] = -np.nan
        # 设置 self.signd 中每隔6个位置的元素，对应在 self.ed 中为 True 的元素为负 NaN
        self.signd[3::6][self.ed[3::6]] = -np.nan
        
        # 设置 self.signf 中每隔6个位置的元素，对应在 self.ef 中为 True 的元素为负零
        self.signf[4::6][self.ef[4::6]] = -0.
        # 设置 self.signd 中每隔6个位置的元素，对应在 self.ed 中为 True 的元素为负零
        self.signd[4::6][self.ed[4::6]] = -0.
    def test_float(self):
        # 浮点数测试函数

        # 循环4次，每次偏移量不同，执行以下断言
        for i in range(4):
            # 断言切片后的self.f大于0的元素与self.ef的元素相等
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            # 断言切片后的self.f减去1大于等于0的元素与self.ef的元素相等
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            # 断言切片后的self.f等于0的元素与self.ef的元素按位取反后相等
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            # 断言切片后的-self.f小于0的元素与self.ef的元素相等
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            # 断言切片后的-self.f加1小于等于0的元素与self.ef的元素相等
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            # 计算切片后的self.f不等于0的结果
            r = self.f[i:] != 0
            # 断言r与self.ef的元素相等
            assert_array_equal(r, self.ef[i:])
            # 计算切片后的self.f不等于np.zeros_like(self.f)的结果
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            # 计算0不等于切片后的self.f的结果
            r3 = 0 != self.f[i:]
            # 断言r与r2、r3的结果相等
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # 检查布尔值转换为int8是否与0x1相等
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # 对amd64架构上的isnan函数采用相同的代码路径
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    def test_double(self):
        # 双精度测试函数

        # 循环2次，每次偏移量不同，执行以下断言
        for i in range(2):
            # 断言切片后的self.d大于0的元素与self.ed的元素相等
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            # 断言切片后的self.d减去1大于等于0的元素与self.ed的元素相等
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            # 断言切片后的self.d等于0的元素与self.ed的元素按位取反后相等
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            # 断言切片后的-self.d小于0的元素与self.ed的元素相等
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            # 断言切片后的-self.d加1小于等于0的元素与self.ed的元素相等
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            # 计算切片后的self.d不等于0的结果
            r = self.d[i:] != 0
            # 断言r与self.ed的元素相等
            assert_array_equal(r, self.ed[i:])
            # 计算切片后的self.d不等于np.zeros_like(self.d)的结果
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            # 计算0不等于切片后的self.d的结果
            r3 = 0 != self.d[i:]
            # 断言r与r2、r3的结果相等
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # 检查布尔值转换为int8是否与0x1相等
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # 对amd64架构上的isnan函数采用相同的代码路径
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])
# 定义一个测试类 TestSeterr
class TestSeterr:
    
    # 测试默认错误状态
    def test_default(self):
        # 获取当前的错误状态
        err = np.geterr()
        # 断言当前错误状态与预期的默认值相符
        assert_equal(err,
                     dict(divide='warn',
                          invalid='warn',
                          over='warn',
                          under='ignore')
                     )
    
    # 测试设置错误状态
    def test_set(self):
        # 在新的错误状态下执行以下代码块
        with np.errstate():
            # 获取当前的错误状态
            err = np.seterr()
            # 设置新的错误状态，记录旧的错误状态
            old = np.seterr(divide='print')
            # 断言获取到的错误状态与设置的旧状态相同
            assert_(err == old)
            # 获取新的错误状态
            new = np.seterr()
            # 断言设置后的新的除法错误处理方式为打印
            assert_(new['divide'] == 'print')
            # 设置新的错误状态，使得溢出时触发异常
            np.seterr(over='raise')
            # 断言当前溢出错误处理方式已设置为触发异常
            assert_(np.geterr()['over'] == 'raise')
            # 断言新的除法错误处理方式仍为打印
            assert_(new['divide'] == 'print')
            # 恢复旧的错误状态
            np.seterr(**old)
            # 断言当前错误状态已恢复为旧的状态
            assert_(np.geterr() == old)
    
    # 在指定条件下跳过此测试，条件包括 WebAssembly 不支持浮点异常处理和特定 ARM 架构的问题
    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    @pytest.mark.skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    # 测试除法错误处理
    def test_divide_err(self):
        # 在除法错误触发异常的上下文中执行以下代码块
        with np.errstate(divide='raise'):
            # 使用 assert_raises 断言浮点异常已触发
            with assert_raises(FloatingPointError):
                np.array([1.]) / np.array([0.])
            
            # 设置除法错误处理方式为忽略
            np.seterr(divide='ignore')
            # 再次执行除法运算
            np.array([1.]) / np.array([0.])


# 定义一个测试类 TestFloatExceptions
class TestFloatExceptions:
    
    # 断言特定的浮点异常已触发
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        # 获取 x 的数据类型
        ftype = type(x)
        try:
            # 执行浮点操作 flop(x, y)，预期触发浮点异常
            flop(x, y)
            # 如果没有触发异常，断言失败
            assert_(False,
                    "Type %s did not raise fpe error '%s'." % (ftype, fpeerr))
        except FloatingPointError as exc:
            # 断言触发的异常信息包含预期的浮点异常信息
            assert_(str(exc).find(fpeerr) >= 0,
                    "Type %s raised wrong fpe error '%s'." % (ftype, exc))
    
    # 检查特定的操作是否触发浮点异常
    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # 检查是否触发浮点异常
        #
        # 给定一个浮点操作 `flop` 和两个标量值，检查操作是否引发由 `fpeerr` 指定的浮点异常。
        # 测试所有情况，包括使用 0-d 数组标量。
        
        # 检查普通标量情况
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        # 检查第一个标量为 0-d 数组标量的情况
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        # 检查第二个标量为 0-d 数组标量的情况
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        # 检查两个标量都为 0-d 数组标量的情况
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    # 对所有实数和复数浮点类型进行测试
    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    @pytest.mark.parametrize("typecode", np.typecodes["AllFloat"])
    # 测试浮点异常处理函数
    def test_floating_exceptions(self, typecode):
        # 如果运行在 BSD 平台并且类型码是 'gG' 中的一种，则跳过测试
        if 'bsd' in sys.platform and typecode in 'gG':
            pytest.skip(reason="Fallback impl for (c)longdouble may not raise "
                               "FPE errors as expected on BSD OSes, "
                               "see gh-24876, gh-23379")

        # 在上下文管理器中设置所有浮点错误状态为引发异常
        with np.errstate(all='raise'):
            # 将类型码转换为对应的数据类型
            ftype = obj2sctype(typecode)
            
            # 如果数据类型的种类是浮点数
            if np.dtype(ftype).kind == 'f':
                # 获取该类型的极限值信息
                fi = np.finfo(ftype)
                ft_tiny = fi._machar.tiny  # 最小正数
                ft_max = fi.max  # 最大正数
                ft_eps = fi.eps  # 机器精度
                underflow = 'underflow'  # 下溢出
                divbyzero = 'divide by zero'  # 除以零
            else:
                # 如果是 'c' 复数类型，获取其对应的实数数据类型
                rtype = type(ftype(0).real)
                fi = np.finfo(rtype)
                ft_tiny = ftype(fi._machar.tiny)  # 最小正数
                ft_max = ftype(fi.max)  # 最大正数
                ft_eps = ftype(fi.eps)  # 机器精度
                underflow = ''  # 下溢出
                divbyzero = ''  # 除以零
            
            overflow = 'overflow'  # 上溢出
            invalid = 'invalid'  # 无效操作

            # 对各种情况进行异常检测
            if not np.isnan(ft_tiny):
                # 测试下溢出情况
                self.assert_raises_fpe(underflow,
                                    lambda a, b: a/b, ft_tiny, ft_max)
                self.assert_raises_fpe(underflow,
                                    lambda a, b: a*b, ft_tiny, ft_tiny)
            # 测试上溢出情况
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a*b, ft_max, ftype(2))
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a/b, ft_max, ftype(0.5))
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a+b, ft_max, ft_max*ft_eps)
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a-b, -ft_max, ft_max*ft_eps)
            self.assert_raises_fpe(overflow,
                                   np.power, ftype(2), ftype(2**fi.nexp))
            # 测试除以零情况
            self.assert_raises_fpe(divbyzero,
                                   lambda a, b: a/b, ftype(1), ftype(0))
            # 测试无效操作情况
            self.assert_raises_fpe(
                invalid, lambda a, b: a/b, ftype(np.inf), ftype(np.inf)
            )
            self.assert_raises_fpe(invalid,
                                   lambda a, b: a/b, ftype(0), ftype(0))
            self.assert_raises_fpe(
                invalid, lambda a, b: a-b, ftype(np.inf), ftype(np.inf)
            )
            self.assert_raises_fpe(
                invalid, lambda a, b: a+b, ftype(np.inf), ftype(-np.inf)
            )
            self.assert_raises_fpe(invalid,
                                   lambda a, b: a*b, ftype(0), ftype(np.inf))
    # 使用 pytest 标记，如果在 WebAssembly 环境中，则跳过此测试（原因是 WebAssembly 不支持浮点数异常）
    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    # 定义一个测试函数 test_warnings，用于测试警告路径代码
    def test_warnings(self):
        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器，使得所有警告都被记录
            warnings.simplefilter("always")
            # 设置 NumPy 的错误状态为警告模式
            with np.errstate(all="warn"):
                # 测试浮点数除法中的除以零情况
                np.divide(1, 0.)
                # 断言捕获的警告数量为 1
                assert_equal(len(w), 1)
                # 断言最后一条警告消息中包含 "divide by zero" 字样
                assert_("divide by zero" in str(w[0].message))
                # 测试浮点数溢出情况
                np.array(1e300) * np.array(1e300)
                # 断言捕获的警告数量为 2
                assert_equal(len(w), 2)
                # 断言最后一条警告消息中包含 "overflow" 字样
                assert_("overflow" in str(w[-1].message))
                # 测试无穷大减去无穷大情况
                np.array(np.inf) - np.array(np.inf)
                # 断言捕获的警告数量为 3
                assert_equal(len(w), 3)
                # 断言最后一条警告消息中包含 "invalid value" 字样
                assert_("invalid value" in str(w[-1].message))
                # 测试浮点数下溢情况
                np.array(1e-300) * np.array(1e-300)
                # 断言捕获的警告数量为 4
                assert_equal(len(w), 4)
                # 断言最后一条警告消息中包含 "underflow" 字样
                assert_("underflow" in str(w[-1].message))
# 定义一个测试类 TestTypes，用于测试数据类型转换的相关功能
class TestTypes:
    
    # 定义测试类型强制转换的方法
    def test_coercion(self):
        
        # 定义一个内部函数 res_type，计算两个数组相加后的数据类型
        def res_type(a, b):
            return np.add(a, b).dtype
        
        # 调用测试类的 check_promotion_cases 方法进行测试类型提升的情况
        self.check_promotion_cases(res_type)
        
        # 对特定情况进行测试：浮点数/复数标量 * 布尔值/int8 数组
        # 保证不会使浮点数/复数类型变窄
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.longdouble(1.234) * a
            assert_equal(b.dtype, np.dtype(np.longdouble),
                         "array type %s" % a.dtype)
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype('f4'), "array type %s" % a.dtype)
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype('f2'), "array type %s" % a.dtype)
            
            b = 1.234j * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.clongdouble(1.234j) * a
            assert_equal(b.dtype, np.dtype(np.clongdouble),
                         "array type %s" % a.dtype)
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype('c8'), "array type %s" % a.dtype)
        
        # 下面的用例有问题，需要更多的修改来解决其复杂的副作用
        #
        # 用例：(1-t)*a，其中 't' 是布尔数组，'a' 是 float32，不应该提升为 float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # 或许应该使用 ~t（按位取反）更为合适，但这样可能不够直观，并且在 't' 是整数数组而不是布尔数组时会失败：
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    # 定义测试结果类型的方法
    def test_result_type(self):
        # 调用测试类的 check_promotion_cases 方法测试结果类型的情况
        self.check_promotion_cases(np.result_type)
        # 断言 np.result_type(None) 的返回结果应为 np.dtype(None)
        assert_(np.result_type(None) == np.dtype(None))
    def test_promote_types_endian(self):
        # 检验 promote_types 函数是否总是返回本地字节顺序的类型

        # 检验小端序 '<i8' 和 '<i8' 是否被提升为 'i8' 类型
        assert_equal(np.promote_types('<i8', '<i8'), np.dtype('i8'))
        # 检验大端序 '>i8' 和 '>i8' 是否被提升为 'i8' 类型
        assert_equal(np.promote_types('>i8', '>i8'), np.dtype('i8'))

        # 检验大端序 '>i8' 和 '>U16' 是否被提升为 'U21' 类型
        assert_equal(np.promote_types('>i8', '>U16'), np.dtype('U21'))
        # 检验小端序 '<i8' 和 '<U16' 是否被提升为 'U21' 类型
        assert_equal(np.promote_types('<i8', '<U16'), np.dtype('U21'))
        # 检验大端序 '>U16' 和 '>i8' 是否被提升为 'U21' 类型
        assert_equal(np.promote_types('>U16', '>i8'), np.dtype('U21'))
        # 检验小端序 '<U16' 和 '<i8' 是否被提升为 'U21' 类型
        assert_equal(np.promote_types('<U16', '<i8'), np.dtype('U21'))

        # 检验小端序 '<S5' 和 '<U8' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('<S5', '<U8'), np.dtype('U8'))
        # 检验大端序 '>S5' 和 '>U8' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('>S5', '>U8'), np.dtype('U8'))
        # 检验小端序 '<U8' 和 '<S5' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('<U8', '<S5'), np.dtype('U8'))
        # 检验大端序 '>U8' 和 '>S5' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('>U8', '>S5'), np.dtype('U8'))
        # 检验小端序 '<U5' 和 '<U8' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('<U5', '<U8'), np.dtype('U8'))
        # 检验大端序 '>U8' 和 '>U5' 是否被提升为 'U8' 类型
        assert_equal(np.promote_types('>U8', '>U5'), np.dtype('U8'))

        # 检验小端序 '<M8' 和 '<M8' 是否被提升为 'M8' 类型
        assert_equal(np.promote_types('<M8', '<M8'), np.dtype('M8'))
        # 检验大端序 '>M8' 和 '>M8' 是否被提升为 'M8' 类型
        assert_equal(np.promote_types('>M8', '>M8'), np.dtype('M8'))
        # 检验小端序 '<m8' 和 '<m8' 是否被提升为 'm8' 类型
        assert_equal(np.promote_types('<m8', '<m8'), np.dtype('m8'))
        # 检验大端序 '>m8' 和 '>m8' 是否被提升为 'm8' 类型
        assert_equal(np.promote_types('>m8', '>m8'), np.dtype('m8'))

    def test_can_cast_and_promote_usertypes(self):
        # 测试 can_cast 和 promote_types 函数对用户定义类型的支持

        # rational 类型定义了对有符号整数和布尔类型的安全转换，
        # rational 类型本身可以安全地转换为双精度浮点数。

        # valid_types 列表中的类型应当能够安全转换为 rational 类型
        valid_types = ["int8", "int16", "int32", "int64", "bool"]
        # invalid_types 中的类型不应当能够安全转换为 rational 类型
        invalid_types = "BHILQP" + "FDG" + "mM" + "f" + "V"

        # 获取 rational 类型的 NumPy dtype 对象
        rational_dt = np.dtype(rational)
        # 对于 valid_types 中的每一种 NumPy dtype 类型
        for numpy_dtype in valid_types:
            numpy_dtype = np.dtype(numpy_dtype)
            # 断言 numpy_dtype 能够转换为 rational_dt
            assert np.can_cast(numpy_dtype, rational_dt)
            # 断言 promote_types(numpy_dtype, rational_dt) 返回 rational_dt
            assert np.promote_types(numpy_dtype, rational_dt) is rational_dt

        # 对于 invalid_types 中的每一种 NumPy dtype 类型
        for numpy_dtype in invalid_types:
            numpy_dtype = np.dtype(numpy_dtype)
            # 断言 numpy_dtype 不能转换为 rational_dt
            assert not np.can_cast(numpy_dtype, rational_dt)
            # 使用 pytest 断言 promote_types(numpy_dtype, rational_dt) 抛出 TypeError 异常
            with pytest.raises(TypeError):
                np.promote_types(numpy_dtype, rational_dt)

        # 获取 double 类型的 NumPy dtype 对象
        double_dt = np.dtype("double")
        # 断言 rational_dt 能够转换为 double_dt
        assert np.can_cast(rational_dt, double_dt)
        # 断言 promote_types(double_dt, rational_dt) 返回 double_dt
        assert np.promote_types(double_dt, rational_dt) is double_dt

    @pytest.mark.parametrize("swap", ["", "swap"])
    @pytest.mark.parametrize("string_dtype", ["U", "S"])
    # 定义测试函数，用于测试类型提升函数对字符串类型的操作
    def test_promote_types_strings(self, swap, string_dtype):
        # 根据参数 swap 的值来选择使用不同的类型提升函数
        if swap == "swap":
            promote_types = lambda a, b: np.promote_types(b, a)
        else:
            promote_types = np.promote_types

        # 将参数 string_dtype 赋值给变量 S，表示字符串类型
        S = string_dtype

        # 测试用例：将数字类型与未定大小的字符串类型进行提升
        assert_equal(promote_types('bool', S), np.dtype(S+'5'))
        assert_equal(promote_types('b', S), np.dtype(S+'4'))
        assert_equal(promote_types('u1', S), np.dtype(S+'3'))
        assert_equal(promote_types('u2', S), np.dtype(S+'5'))
        assert_equal(promote_types('u4', S), np.dtype(S+'10'))
        assert_equal(promote_types('u8', S), np.dtype(S+'20'))
        assert_equal(promote_types('i1', S), np.dtype(S+'4'))
        assert_equal(promote_types('i2', S), np.dtype(S+'6'))
        assert_equal(promote_types('i4', S), np.dtype(S+'11'))
        assert_equal(promote_types('i8', S), np.dtype(S+'21'))
        
        # 测试用例：将数字类型与已定大小的字符串类型进行提升
        assert_equal(promote_types('bool', S+'1'), np.dtype(S+'5'))
        assert_equal(promote_types('bool', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('b', S+'1'), np.dtype(S+'4'))
        assert_equal(promote_types('b', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u1', S+'1'), np.dtype(S+'3'))
        assert_equal(promote_types('u1', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u2', S+'1'), np.dtype(S+'5'))
        assert_equal(promote_types('u2', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u4', S+'1'), np.dtype(S+'10'))
        assert_equal(promote_types('u4', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u8', S+'1'), np.dtype(S+'20'))
        assert_equal(promote_types('u8', S+'30'), np.dtype(S+'30'))
        
        # 测试用例：将对象类型与字符串类型进行提升
        assert_equal(promote_types('O', S+'30'), np.dtype('O'))

    # 标记为参数化测试，测试无效的空类型提升情况
    @pytest.mark.parametrize(["dtype1", "dtype2"],
            [[np.dtype("V6"), np.dtype("V10")],  # 形状不匹配
             [np.dtype([("name1", "i8")]), np.dtype([("name2", "i8")])],  # 名称不匹配
            ])
    def test_invalid_void_promotion(self, dtype1, dtype2):
        # 断言在类型提升时抛出 TypeError 异常
        with pytest.raises(TypeError):
            np.promote_types(dtype1, dtype2)

    # 标记为参数化测试，测试有效的空类型提升情况
    @pytest.mark.parametrize(["dtype1", "dtype2"],
            [[np.dtype("V10"), np.dtype("V10")],  # 相同形状
             [np.dtype([("name1", "i8")]),
              np.dtype([("name1", np.dtype("i8").newbyteorder())])],  # 相同名称但字节顺序不同
             [np.dtype("i8,i8"), np.dtype("i8,>i8")],  # 元素类型匹配但字节顺序不同
             [np.dtype("i8,i8"), np.dtype("i4,i4")],  # 元素类型不匹配
            ])
    def test_valid_void_promotion(self, dtype1, dtype2):
        # 断言对于有效的类型提升情况，返回的类型与第一个参数类型相同
        assert np.promote_types(dtype1, dtype2) == dtype1

    # 标记为参数化测试，测试不同数据类型的字符串类型提升
    @pytest.mark.parametrize("dtype",
            list(np.typecodes["All"]) +
            ["i,i", "10i", "S3", "S100", "U3", "U100", rational])
    # 定义一个测试函数，用于测试相同类型的元数据保留情况
    def test_promote_identical_types_metadata(self, dtype):
        # 创建一个包含元数据的字典
        metadata = {1: 1}
        # 使用给定的 dtype 和 metadata 创建一个新的数据类型对象
        dtype = np.dtype(dtype, metadata=metadata)

        # 调用 numpy 的 promote_types 函数，传入相同的 dtype 两次以确保类型提升时元数据得到保留
        res = np.promote_types(dtype, dtype)
        # 断言结果类型的 metadata 与原始 dtype 的 metadata 相同
        assert res.metadata == dtype.metadata

        # 对 dtype 进行字节交换以确保字节顺序为本地顺序
        dtype = dtype.newbyteorder()
        if dtype.isnative:
            # 如果 dtype 已经是本地顺序，则无需进行后续操作，直接返回
            return

        # 当 dtype 不是本地顺序时，再次调用 promote_types 函数
        res = np.promote_types(dtype, dtype)

        # 在字节交换时，通常（目前）会丢失元数据，除非 dtype 是 unicode 类型
        if dtype.char != "U":
            # 断言结果类型的 metadata 应为 None
            assert res.metadata is None
        else:
            # 断言结果类型的 metadata 应与原始 dtype 的 metadata 相同
            assert res.metadata == metadata
        # 断言结果类型为本地顺序
        assert res.isnative

    # 标记此测试为慢速测试
    @pytest.mark.slow
    # 忽略关于未来警告的数字提升的警告
    @pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
    # 使用参数化测试，测试不同的 dtype1 和 dtype2 组合
    @pytest.mark.parametrize(["dtype1", "dtype2"],
            itertools.product(
                # 列出所有 numpy 支持的类型码，并添加一些特定的类型组合
                list(np.typecodes["All"]) +
                ["i,i", "S3", "S100", "U3", "U100", rational],
                repeat=2))
    def test_promote_types_metadata(self, dtype1, dtype2):
        """
        Metadata handling in promotion does not appear formalized
        right now in NumPy. This test should thus be considered to
        document behaviour, rather than test the correct definition of it.

        This test is very ugly, it was useful for rewriting part of the
        promotion, but probably should eventually be replaced/deleted
        (i.e. when metadata handling in promotion is better defined).
        """
        # 定义两个元数据字典
        metadata1 = {1: 1}
        metadata2 = {2: 2}
        # 使用给定的 dtype 和元数据创建 dtype1 和 dtype2
        dtype1 = np.dtype(dtype1, metadata=metadata1)
        dtype2 = np.dtype(dtype2, metadata=metadata2)

        try:
            # 尝试对 dtype1 和 dtype2 进行类型提升
            res = np.promote_types(dtype1, dtype2)
        except TypeError:
            # 如果类型提升失败，此测试仅检查元数据
            return

        if res.char not in "USV" or res.names is not None or res.shape != ():
            # 除了字符串 dtype（和未结构化的 void 类型），其他类型在提升时通常会丢失元数据
            # （除非两个 dtype 完全相同）。
            # 在某些情况下，结构化的 dtype 也不会丢失，但会受到限制。
            assert res.metadata is None
        elif res == dtype1:
            # 如果结果是 dtype1，则通常不会改变返回
            assert res is dtype1
        elif res == dtype2:
            # dtype1 可能已被转换为与 dtype2 相同的类型/种类。
            # 如果结果 dtype 与 dtype2 相同，则当前选择 dtype1 的转换版本，已经丢失了元数据：
            if np.promote_types(dtype1, dtype2.kind) == dtype2:
                res.metadata is None
            else:
                res.metadata == metadata2
        else:
            assert res.metadata is None

        # 尝试字节交换后的版本再次进行测试
        dtype1 = dtype1.newbyteorder()
        # 检查字节交换后的结果与之前的结果是否相同
        assert dtype1.metadata == metadata1
        res_bs = np.promote_types(dtype1, dtype2)
        assert res_bs == res
        assert res_bs.metadata == res.metadata
    # 定义一个测试方法，用于验证数据类型转换是否可行
    def test_can_cast(self):
        # 断言可以从 np.int32 转换到 np.int64
        assert_(np.can_cast(np.int32, np.int64))
        # 断言可以从 np.float64 转换到 complex
        assert_(np.can_cast(np.float64, complex))
        # 断言无法从 complex 转换到 float
        assert_(not np.can_cast(complex, float))

        # 断言可以从 'i8'（int64）转换到 'f8'（float64）
        assert_(np.can_cast('i8', 'f8'))
        # 断言无法从 'i8'（int64）转换到 'f4'（float32）
        assert_(not np.can_cast('i8', 'f4'))
        # 断言可以从 'i4'（int32）转换到 'S11'（长度为11的字符串）
        assert_(np.can_cast('i4', 'S11'))

        # 断言可以从 'i8'（int64）转换到 'i8'（int64），不允许安全转换
        assert_(np.can_cast('i8', 'i8', 'no'))
        # 断言无法从 '<i8' 转换到 '>i8'，不允许安全转换
        assert_(not np.can_cast('<i8', '>i8', 'no'))

        # 断言可以从 '<i8' 安全转换到 '>i8'，等价转换
        assert_(np.can_cast('<i8', '>i8', 'equiv'))
        # 断言无法从 '<i4' 转换到 '>i8'，等价转换
        assert_(not np.can_cast('<i4', '>i8', 'equiv'))

        # 断言可以从 '<i4' 安全转换到 '>i8'，安全转换
        assert_(np.can_cast('<i4', '>i8', 'safe'))
        # 断言无法从 '<i8' 安全转换到 '>i4'，安全转换
        assert_(not np.can_cast('<i8', '>i4', 'safe'))

        # 断言可以从 '<i8' 同类别转换到 '>i4'，同类别转换
        assert_(np.can_cast('<i8', '>i4', 'same_kind'))
        # 断言无法从 '<i8' 同类别转换到 '>u4'，同类别转换
        assert_(not np.can_cast('<i8', '>u4', 'same_kind'))

        # 断言可以从 '<i8' 不安全地转换到 '>u4'，不安全转换
        assert_(np.can_cast('<i8', '>u4', 'unsafe'))

        # 断言可以从 'bool' 转换到 'S5'（长度为5的字符串）
        assert_(np.can_cast('bool', 'S5'))
        # 断言无法从 'bool' 转换到 'S4'（长度为4的字符串）
        assert_(not np.can_cast('bool', 'S4'))

        # 断言可以从 'b'（布尔值）转换到 'S4'（长度为4的字符串）
        assert_(np.can_cast('b', 'S4'))
        # 断言无法从 'b'（布尔值）转换到 'S3'（长度为3的字符串）
        assert_(not np.can_cast('b', 'S3'))

        # 断言可以从 'u1'（无符号整数）转换到 'S3'（长度为3的字符串）
        assert_(np.can_cast('u1', 'S3'))
        # 断言无法从 'u1'（无符号整数）转换到 'S2'（长度为2的字符串）
        assert_(not np.can_cast('u1', 'S2'))
        # 断言可以从 'u2'（无符号整数）转换到 'S5'（长度为5的字符串）
        assert_(np.can_cast('u2', 'S5'))
        # 断言无法从 'u2'（无符号整数）转换到 'S4'（长度为4的字符串）
        assert_(not np.can_cast('u2', 'S4'))
        # 断言可以从 'u4'（无符号整数）转换到 'S10'（长度为10的字符串）
        assert_(np.can_cast('u4', 'S10'))
        # 断言无法从 'u4'（无符号整数）转换到 'S9'（长度为9的字符串）
        assert_(not np.can_cast('u4', 'S9'))
        # 断言可以从 'u8'（无符号整数）转换到 'S20'（长度为20的字符串）
        assert_(np.can_cast('u8', 'S20'))
        # 断言无法从 'u8'（无符号整数）转换到 'S19'（长度为19的字符串）
        assert_(not np.can_cast('u8', 'S19'))

        # 断言可以从 'i1'（有符号整数）转换到 'S4'（长度为4的字符串）
        assert_(np.can_cast('i1', 'S4'))
        # 断言无法从 'i1'（有符号整数）转换到 'S3'（长度为3的字符串）
        assert_(not np.can_cast('i1', 'S3'))
        # 断言可以从 'i2'（有符号整数）转换到 'S6'（长度为6的字符串）
        assert_(np.can_cast('i2', 'S6'))
        # 断言无法从 'i2'（有符号整数）转换到 'S5'（长度为5的字符串）
        assert_(not np.can_cast('i2', 'S5'))
        # 断言可以从 'i4'（有符号整数）转换到 'S11'（长度为11的字符串）
        assert_(np.can_cast('i4', 'S11'))
        # 断言无法从 'i4'（有符号整数）转换到 'S10'（长度为10的字符串）
        assert_(not np.can_cast('i4', 'S10'))
        # 断言可以从 'i8'（有符号整数）转换到 'S21'（长度为21的字符串）
        assert_(np.can_cast('i8', 'S21'))
        # 断言无法从 'i8'（有符号整数）转换到 'S20'（长度为20的字符串）
        assert_(not np.can_cast('i8', 'S20'))

        # 断言可以从 'bool' 转换到 'S5'（长度为5的字符串）
        assert_(np.can_cast('bool', 'S5'))
        # 断言无法从 'bool' 转换到 'S4'（长度为4的字符串）
        assert_(not np.can_cast('bool', 'S4'))

        # 断言可以从 'b'（布尔值）转换到 'U4'（unicode编码，长度为4的字符串）
        assert_(np.can_cast('b', 'U4'))
        # 断言无法从 'b'（布尔值）转换到 'U3'（unicode编码，长度为3的字符串）
        assert_(not np.can_cast('b', 'U3'))

        # 断言可以从 'u1'（无符号整数）转换到 'U3'（unicode编码，长度为3的字符串）
        assert_(np.can_cast('u1', 'U3'))
        # 断言无法从 'u1'（无符号整数）转换到 'U2'（unicode编码，长度为2的字符串）
        assert_(not np.can_cast('u1', 'U2'))
        # 断言可以从 'u2'（无符号整数）转换到 'U5'（unicode编码，长度为5的字符串）
        assert_(np.can_cast('u2', 'U5'))
        # 断言无法从 'u2'（无符号整数）转换到 'U4'（unicode编码，长度为4的字符串）
        assert_(not np.can_cast('u2', 'U4'))
        # 断言可以从 'u4'（无符号整数）转换到 'U10'（unicode编码，长度为10的字符串）
        assert_(np.can_cast('u4', 'U10'))
        # 断言无法从 'u4'（无符号整数）转换到 'U9'（unicode编码，长度
    def test_can_cast_simple_to_structured(self):
        # Non-structured can only be cast to structured in 'unsafe' mode.
        assert_(not np.can_cast('i4', 'i4,i4'))  # 检查非结构化类型是否可以转换为结构化类型
        assert_(not np.can_cast('i4', 'i4,i2'))  # 检查非结构化类型是否可以转换为较小结构化类型
        assert_(np.can_cast('i4', 'i4,i4', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将非结构化类型转换为结构化类型
        assert_(np.can_cast('i4', 'i4,i2', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将非结构化类型转换为较小结构化类型
        # Even if there is just a single field which is OK.
        assert_(not np.can_cast('i2', [('f1', 'i4')]))  # 检查是否可以将非结构化类型转换为仅包含一个字段的结构化类型
        assert_(not np.can_cast('i2', [('f1', 'i4')], casting='same_kind'))  # 在'same_kind'模式下检查是否可以将非结构化类型转换为相同类型的结构化类型
        assert_(np.can_cast('i2', [('f1', 'i4')], casting='unsafe'))  # 在'unsafe'模式下检查是否可以将非结构化类型转换为结构化类型
        # It should be the same for recursive structured or subarrays.
        assert_(not np.can_cast('i2', [('f1', 'i4,i4')]))  # 检查是否可以将非结构化类型转换为具有递归结构或子数组的结构化类型
        assert_(np.can_cast('i2', [('f1', 'i4,i4')], casting='unsafe'))  # 在'unsafe'模式下检查是否可以将非结构化类型转换为具有递归结构或子数组的结构化类型
        assert_(not np.can_cast('i2', [('f1', '(2,3)i4')]))  # 检查是否可以将非结构化类型转换为具有多维子数组的结构化类型
        assert_(np.can_cast('i2', [('f1', '(2,3)i4')], casting='unsafe'))  # 在'unsafe'模式下检查是否可以将非结构化类型转换为具有多维子数组的结构化类型

    def test_can_cast_structured_to_simple(self):
        # Need unsafe casting for structured to simple.
        assert_(not np.can_cast([('f1', 'i4')], 'i4'))  # 检查是否可以将结构化类型转换为非结构化类型
        assert_(np.can_cast([('f1', 'i4')], 'i4', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将结构化类型转换为非结构化类型
        assert_(np.can_cast([('f1', 'i4')], 'i2', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将结构化类型转换为较小非结构化类型
        # Since it is unclear what is being cast, multiple fields to
        # single should not work even for unsafe casting.
        assert_(not np.can_cast('i4,i4', 'i4', casting='unsafe'))  # 检查是否可以将多个字段的结构化类型转换为单一的非结构化类型，即使在'unsafe'模式下也不行
        # But a single field inside a single field is OK.
        assert_(not np.can_cast([('f1', [('x', 'i4')])], 'i4'))  # 检查是否可以将包含单一字段的结构化类型转换为非结构化类型
        assert_(np.can_cast([('f1', [('x', 'i4')])], 'i4', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将包含单一字段的结构化类型转换为非结构化类型
        # And a subarray is fine too - it will just take the first element
        # (arguably not very consistently; might also take the first field).
        assert_(not np.can_cast([('f0', '(3,)i4')], 'i4'))  # 检查是否可以将包含子数组的结构化类型转换为非结构化类型
        assert_(np.can_cast([('f0', '(3,)i4')], 'i4', casting='unsafe'))  # 在'unsafe'模式下检查是否可以将包含子数组的结构化类型转换为非结构化类型
        # But a structured subarray with multiple fields should fail.
        assert_(not np.can_cast([('f0', ('i4,i4'), (2,))], 'i4',
                                casting='unsafe'))  # 检查是否可以将包含多个字段的结构化子数组类型转换为非结构化类型，即使在'unsafe'模式下也不行

    @pytest.mark.xfail(np._get_promotion_state() != "legacy",
            reason="NEP 50: no python int/float/complex support (yet)")
    def test_can_cast_values(self):
        # gh-5917
        for dt in sctypes['int'] + sctypes['uint']:
            ii = np.iinfo(dt)
            assert_(np.can_cast(ii.min, dt))  # 检查数据类型是否能够容纳其最小值
            assert_(np.can_cast(ii.max, dt))  # 检查数据类型是否能够容纳其最大值
            assert_(not np.can_cast(ii.min - 1, dt))  # 检查数据类型是否能够容纳其最小值减一的值
            assert_(not np.can_cast(ii.max + 1, dt))  # 检查数据类型是否能够容纳其最大值加一的值

        for dt in sctypes['float']:
            fi = np.finfo(dt)
            assert_(np.can_cast(fi.min, dt))  # 检查数据类型是否能够容纳其最小值
            assert_(np.can_cast(fi.max, dt))  # 检查数据类型是否能够容纳其最大值

    @pytest.mark.parametrize("dtype",
            list("?bhilqBHILQefdgFDG") + [rational])
    # 定义测试函数，用于验证可以将标量类型转换为指定类型
    def test_can_cast_scalars(self, dtype):
        # 基本测试，确保在can_cast中支持标量类型的转换
        # （并未详尽检查其行为）。

        # 将输入的dtype转换为NumPy的数据类型对象
        dtype = np.dtype(dtype)
        # 使用dtype创建一个标量（scalar）值为0的实例
        scalar = dtype.type(0)

        # 断言：检查是否可以将scalar转换为'int64'类型
        assert np.can_cast(scalar, "int64") == np.can_cast(dtype, "int64")
        # 断言：检查是否可以将scalar转换为'float32'类型，允许不安全的转换
        assert np.can_cast(scalar, "float32", casting="unsafe")
# 自定义异常类，用于测试异常在 fromiter 中的传播
class NIterError(Exception):
    pass


# 测试类 TestFromiter
class TestFromiter:
    
    # 生成器方法，返回一个生成平方数的生成器对象
    def makegen(self):
        return (x**2 for x in range(24))
    
    # 测试 fromiter 方法的返回类型
    def test_types(self):
        # 使用生成器创建 np.int32 类型的数组
        ai32 = np.fromiter(self.makegen(), np.int32)
        # 使用生成器创建 np.int64 类型的数组
        ai64 = np.fromiter(self.makegen(), np.int64)
        # 使用生成器创建 float 类型的数组
        af = np.fromiter(self.makegen(), float)
        # 断言数组的数据类型符合预期
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))
    
    # 测试 fromiter 方法返回数组的长度
    def test_lengths(self):
        # 期望的数组，用于比较长度
        expected = np.array(list(self.makegen()))
        # 使用生成器创建 int 类型的数组 a
        a = np.fromiter(self.makegen(), int)
        # 使用生成器创建长度为 20 的 int 类型数组 a20
        a20 = np.fromiter(self.makegen(), int, 20)
        # 断言数组 a 的长度与期望数组 expected 的长度相同
        assert_(len(a) == len(expected))
        # 断言数组 a20 的长度为 20
        assert_(len(a20) == 20)
        # 断言从生成器中创建数组时，如果指定的长度比生成器中的元素个数还要大，则抛出 ValueError 异常
        assert_raises(ValueError, np.fromiter,
                          self.makegen(), int, len(expected) + 10)
    
    # 测试 fromiter 方法返回数组的值
    def test_values(self):
        # 期望的数组，用于比较值
        expected = np.array(list(self.makegen()))
        # 使用生成器创建 int 类型的数组 a
        a = np.fromiter(self.makegen(), int)
        # 使用生成器创建长度为 20 的 int 类型数组 a20
        a20 = np.fromiter(self.makegen(), int, 20)
        # 断言数组 a 的所有元素与期望的数组 expected 的所有元素相等
        assert_(np.all(a == expected, axis=0))
        # 断言数组 a20 的所有元素与期望的数组 expected 的前 20 个元素相等
        assert_(np.all(a20 == expected[:20], axis=0))
    
    # 加载数据的方法，用于测试 issue 2592
    def load_data(self, n, eindex):
        # 在迭代器中的特定索引处引发异常
        for e in range(n):
            if e == eindex:
                raise NIterError('error at index %s' % eindex)
            yield e
    
    # 参数化测试方法，测试异常是否正确引发
    @pytest.mark.parametrize("dtype", [int, object])
    @pytest.mark.parametrize(["count", "error_index"], [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # 测试迭代异常是否被正确引发。数据/生成器有 `count` 个元素，但在 `error_index` 处发生错误
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)
    
    # 参数化测试方法，测试 dtype 为字符串类型时的异常情况
    @pytest.mark.parametrize("dtype", ["S", "S0", "V0", "U0"])
    def test_empty_not_structured(self, dtype):
        # 注意："S0" 可能在某些情况下被允许，只要 "S"（没有任何长度）被拒绝即可。
        # 断言从空列表创建指定 dtype 的数组时，抛出 ValueError 异常并提示"Must specify length"
        with pytest.raises(ValueError, match="Must specify length"):
            np.fromiter([], dtype=dtype)
    
    # 参数化测试方法，测试不同 dtype 和数据的情况
    @pytest.mark.parametrize(["dtype", "data"],
            [("d", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
             ("O", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
             ("i,O", [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]),
             # 子数组 dtype（重要因为它们的维度最终出现在结果数组的维度中）
             ("2i", [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]),
             (np.dtype(("O", (2, 3))),
              [((1, 2, 3), (3, 4, 5)), ((3, 2, 1), (5, 4, 3))])])
    @pytest.mark.parametrize("length_hint", [0, 1])
    # 定义测试方法，用于验证在不同数据类型、数据和长度提示下的 np.fromiter 函数行为
    def test_growth_and_complicated_dtypes(self, dtype, data, length_hint):
        # 将输入的 dtype 转换为 NumPy 的数据类型对象
        dtype = np.dtype(dtype)

        # 对数据进行乘法操作，以确保稍微重新分配一些内存空间
        data = data * 100  # 确保重新分配一些内存空间

        # 定义一个内部迭代器类 MyIter
        class MyIter:
            # 用于返回估计长度提示的特殊方法
            def __length_hint__(self):
                # 只需要返回一个估计值，这样的返回值是合法的
                return length_hint  # 返回预估的长度提示，通常为 0 或 1

            # 实现迭代器的迭代方法
            def __iter__(self):
                return iter(data)

        # 使用 np.fromiter 函数根据 MyIter 类生成一个 NumPy 数组
        res = np.fromiter(MyIter(), dtype=dtype)
        # 创建一个期望的 NumPy 数组，用于与结果进行比较
        expected = np.array(data, dtype=dtype)

        # 使用断言验证 res 和 expected 数组是否相等
        assert_array_equal(res, expected)

    # 定义测试空结果的方法
    def test_empty_result(self):
        # 定义一个内部迭代器类 MyIter
        class MyIter:
            # 实现返回长度提示的方法
            def __length_hint__(self):
                return 10  # 返回长度提示为 10

            # 实现迭代方法
            def __iter__(self):
                return iter([])  # 返回一个空的迭代器

        # 使用 np.fromiter 函数从 MyIter 类生成一个 NumPy 数组，数据类型为双精度浮点型
        res = np.fromiter(MyIter(), dtype="d")
        # 使用断言验证结果数组的形状是否为 (0,)
        assert res.shape == (0,)
        # 使用断言验证结果数组的数据类型是否为双精度浮点型
        assert res.dtype == "d"

    # 定义测试迭代器包含的项目数量过少的情况
    def test_too_few_items(self):
        # 定义期望的错误消息，用于验证抛出异常时的匹配条件
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        # 使用 pytest 的断言验证从包含三个项目的列表生成的数组是否抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    # 定义测试插入项目时失败的情况
    def test_failed_itemsetting(self):
        # 使用 pytest 的断言验证从包含 None 的列表生成的数组是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # 创建一个生成器表达式，每次迭代生成一个元组 (2, 3, 4)，共生成 5 次
        iterable = ((2, 3, 4) for i in range(5))
        # 使用 pytest 的断言验证从该生成器生成的数组是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))
class TestNonzero:
    # 定义测试用例类 TestNonzero

    def test_nonzero_trivial(self):
        # 测试空数组的情况
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        # 测试包含单个元素的数组，元素值为 0 或 1 的情况
        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype='?')), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_zerodim(self):
        # 测试在零维数组上调用 nonzero 函数的情况，预期会引发 ValueError 异常
        err_msg = "Calling nonzero on 0d arrays is not allowed"
        with assert_raises_regex(ValueError, err_msg):
            np.nonzero(np.array(0))
        with assert_raises_regex(ValueError, err_msg):
            np.array(1).nonzero()

    def test_nonzero_onedim(self):
        # 测试一维数组的情况
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

        # 测试结构化数组 x，包含多个字段 ('a', 'b', 'c', 'd')，分别统计非零元素个数和非零元素的位置
        x = np.array([(1, 2, -5, -3), (0, 0, 2, 7), (1, 1, 0, 1), (-1, 3, 1, 0), (0, 7, 0, 4)],
                     dtype=[('a', 'i4'), ('b', 'i2'), ('c', 'i1'), ('d', 'i8')])
        assert_equal(np.count_nonzero(x['a']), 3)
        assert_equal(np.count_nonzero(x['b']), 4)
        assert_equal(np.count_nonzero(x['c']), 3)
        assert_equal(np.count_nonzero(x['d']), 4)
        assert_equal(np.nonzero(x['a']), ([0, 2, 3],))
        assert_equal(np.nonzero(x['b']), ([0, 2, 3, 4],))
    # 定义测试方法以检验二维数组的非零元素计数
    def test_nonzero_twodim(self):
        # 创建一个二维数组 x
        x = np.array([[0, 1, 0], [2, 0, 3]])
        # 断言：将 x 转换为指定类型后非零元素的数量应为 3
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        # 断言：返回 x 中非零元素的索引
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        # 创建一个单位矩阵 x
        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        # 断言：返回 x 中非零元素的索引
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

        # 创建一个结构化数组 x，包含两个字段 'a' 和 'b'
        x = np.array([[(0, 1), (0, 0), (1, 11)],
                   [(1, 1), (1, 0), (0, 0)],
                   [(0, 0), (1, 5), (0, 1)]], dtype=[('a', 'f4'), ('b', 'u1')])
        # 断言：返回结构化数组 x 中字段 'a' 的非零元素的数量
        assert_equal(np.count_nonzero(x['a']), 4)
        # 断言：返回结构化数组 x 中字段 'b' 的非零元素的数量
        assert_equal(np.count_nonzero(x['b']), 5)
        # 断言：返回结构化数组 x 中字段 'a' 非零元素的索引
        assert_equal(np.nonzero(x['a']), ([0, 1, 1, 2], [2, 0, 1, 1]))
        # 断言：返回结构化数组 x 中字段 'b' 非零元素的索引
        assert_equal(np.nonzero(x['b']), ([0, 0, 1, 2, 2], [0, 2, 0, 1, 2]))

        # 断言：结构化数组字段 'a' 的转置不是内存对齐的
        assert_(not x['a'].T.flags.aligned)
        # 断言：返回结构化数组字段 'a' 转置后的非零元素的数量
        assert_equal(np.count_nonzero(x['a'].T), 4)
        # 断言：返回结构化数组字段 'b' 转置后的非零元素的数量
        assert_equal(np.count_nonzero(x['b'].T), 5)
        # 断言：返回结构化数组字段 'a' 转置后非零元素的索引
        assert_equal(np.nonzero(x['a'].T), ([0, 1, 1, 2], [1, 1, 2, 0]))
        # 断言：返回结构化数组字段 'b' 转置后非零元素的索引
        assert_equal(np.nonzero(x['b'].T), ([0, 0, 1, 2, 2], [0, 1, 2, 0, 2]))

    # 定义测试方法以检验稀疏数组的特殊情况
    def test_sparse(self):
        # 测试特殊的稀疏条件布尔值路径
        for i in range(20):
            # 创建长度为 200 的布尔类型全零数组 c
            c = np.zeros(200, dtype=bool)
            # 每隔 20 个元素设置一个为 True
            c[i::20] = True
            # 断言：返回数组 c 中非零元素的索引，应为等差数列
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            # 创建长度为 400 的布尔类型全零数组 c
            c = np.zeros(400, dtype=bool)
            # 设置一段区间为 True
            c[10 + i:20 + i] = True
            # 设置指定位置为 True
            c[20 + i*2] = True
            # 断言：返回数组 c 中非零元素的索引，应包括一个点和一个区间
            assert_equal(np.nonzero(c)[0],
                         np.concatenate((np.arange(10 + i, 20 + i), [20 + i*2])))

    # 定义测试方法以检验返回类型
    def test_return_type(self):
        # 定义类 C，继承自 np.ndarray
        class C(np.ndarray):
            pass

        # 遍历类的视图类型：C 和 np.ndarray
        for view in (C, np.ndarray):
            # 遍历不同维度的数组
            for nd in range(1, 4):
                # 创建形状为 (2, ..., nd+1) 的数组 x
                shape = tuple(range(2, 2+nd))
                x = np.arange(np.prod(shape)).reshape(shape).view(view)
                # 遍历数组 x 的非零元素索引
                for nzx in (np.nonzero(x), x.nonzero()):
                    # 断言：非零元素索引的类型应为 np.ndarray
                    for nzx_i in nzx:
                        assert_(type(nzx_i) is np.ndarray)
                        # 断言：非零元素索引数组可写
                        assert_(nzx_i.flags.writeable)
    # 定义测试函数，用于测试 np.count_nonzero 函数在不同轴上的功能
    def test_count_nonzero_axis(self):
        # 创建一个二维 NumPy 数组 m，用于测试
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        # 期望的结果数组，表示每列非零元素的数量
        expected = np.array([1, 1, 1, 1, 1])
        # 检查沿着 axis=0（列）的非零元素数量是否与期望结果一致
        assert_equal(np.count_nonzero(m, axis=0), expected)

        # 期望的结果数组，表示每行非零元素的数量
        expected = np.array([2, 3])
        # 检查沿着 axis=1（行）的非零元素数量是否与期望结果一致
        assert_equal(np.count_nonzero(m, axis=1), expected)

        # 检查传入多个轴时是否引发 ValueError
        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        
        # 检查传入非整数轴时是否引发 TypeError
        assert_raises(TypeError, np.count_nonzero, m, axis='foo')
        
        # 检查超出数组维度范围的轴是否引发 AxisError
        assert_raises(AxisError, np.count_nonzero, m, axis=3)
        
        # 检查传入非法数组对象作为轴时是否引发 TypeError
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))
    def test_count_nonzero_axis_all_dtypes(self):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis
        # 定义错误信息格式字符串
        msg = "Mismatch for dtype: %s"

        def assert_equal_w_dt(a, b, err_msg):
            # 断言两个数组的数据类型相同
            assert_equal(a.dtype, b.dtype, err_msg=err_msg)
            # 断言两个数组相等
            assert_equal(a, b, err_msg=err_msg)

        # 遍历所有数据类型
        for dt in np.typecodes['All']:
            # 根据数据类型构建错误信息
            err_msg = msg % (np.dtype(dt).name,)

            if dt != 'V':  # 排除 'V' 类型
                if dt != 'M':  # 排除 'M' 类型
                    # 创建一个全零矩阵
                    m = np.zeros((3, 3), dtype=dt)
                    # 创建一个全一数组
                    n = np.ones(1, dtype=dt)

                    # 修改矩阵的部分值
                    m[0, 0] = n[0]
                    m[1, 0] = n[0]

                else:  # 处理 'M' 类型，即 np.datetime64
                    # 创建一个日期字符串数组
                    m = np.array(['1970-01-01'] * 9)
                    # 调整形状为 (3, 3)
                    m = m.reshape((3, 3))

                    # 修改部分日期值
                    m[0, 0] = '1970-01-12'
                    m[1, 0] = '1970-01-12'
                    # 转换为指定数据类型
                    m = m.astype(dt)

                # 预期的结果数组
                expected = np.array([2, 0, 0], dtype=np.intp)
                # 断言函数计算的非零元素数量与预期结果相等
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([1, 1, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(2)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)

            if dt == 'V':  # 处理 'V' 类型，即 np.void
                # 对于 np.void 类型，特殊处理
                m = np.array([np.void(1)] * 6).reshape((2, 3))

                expected = np.array([0, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(0)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)
    def test_count_nonzero_axis_consistent(self):
        # 检查在非特殊情况下，有效轴的行为是否一致（因此正确），通过将其与一个整数数组进行比较，
        # 然后将其转换为通用对象 dtype
        from itertools import combinations, permutations

        axis = (0, 1, 2, 3)  # 定义轴的组合
        size = (5, 5, 5, 5)  # 定义数组的大小
        msg = "Mismatch for axis: %s"  # 错误消息模板

        rng = np.random.RandomState(1234)  # 创建随机数生成器实例
        m = rng.randint(-100, 100, size=size)  # 生成随机整数数组 m
        n = m.astype(object)  # 将 m 转换为对象类型数组 n

        for length in range(len(axis)):  # 对于每个轴长度进行迭代
            for combo in combinations(axis, length):  # 对于每个轴组合进行迭代
                for perm in permutations(combo):  # 对于每个轴组合的排列进行迭代
                    assert_equal(
                        np.count_nonzero(m, axis=perm),  # 计算 m 在指定轴上非零元素的数量
                        np.count_nonzero(n, axis=perm),  # 计算 n 在指定轴上非零元素的数量
                        err_msg=msg % (perm,))  # 断言两者相等，否则输出错误消息

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])  # 创建数组 a
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))  # 检查空轴上非零元素的数量与布尔转换后的 a 是否相等

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0],  # 创建数组 a
                      [0, 3, 5, 0],
                      [7, 9, 2, 0]])
        assert_equal(np.count_nonzero(a, axis=0, keepdims=True),  # 检查沿轴 0 保持维度的非零元素数量
                     [[1, 2, 3, 0]])
        assert_equal(np.count_nonzero(a, axis=1, keepdims=True),  # 检查沿轴 1 保持维度的非零元素数量
                     [[1], [2], [3]])
        assert_equal(np.count_nonzero(a, keepdims=True),  # 检查整个数组保持维度的非零元素数量
                     [[6]])

    def test_array_method(self):
        # 测试数组方法调用非零元素的工作情况
        m = np.array([[1, 0, 0], [4, 0, 6]])  # 创建数组 m
        tgt = [[0, 1, 1], [0, 0, 2]]  # 预期结果数组

        assert_equal(m.nonzero(), tgt)  # 断言 m 的非零元素索引与预期结果相等

    def test_nonzero_invalid_object(self):
        # gh-9295
        a = np.array([np.array([1, 2]), 3], dtype=object)  # 创建对象类型数组 a，包含一个非数组对象
        assert_raises(ValueError, np.nonzero, a)  # 断言调用 np.nonzero(a) 会引发 ValueError 异常

        class BoolErrors:
            def __bool__(self):
                raise ValueError("Not allowed")

        assert_raises(ValueError, np.nonzero, np.array([BoolErrors()]))  # 断言调用 np.nonzero() 对包含自定义对象的数组会引发 ValueError 异常
    def test_nonzero_sideeffect_safety(self):
        # 定义一个名为 `FalseThenTrue` 的类
        class FalseThenTrue:
            # 类变量 `_val` 初始化为 False
            _val = False
            # 定义 `__bool__` 方法，返回 `_val` 的值，并在返回前将 `_val` 置为 True
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = True

        # 定义一个名为 `TrueThenFalse` 的类
        class TrueThenFalse:
            # 类变量 `_val` 初始化为 True
            _val = True
            # 定义 `__bool__` 方法，返回 `_val` 的值，并在返回前将 `_val` 置为 False
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = False

        # 在第二次循环中结果增长
        a = np.array([True, FalseThenTrue()])
        # 断言调用 `np.nonzero` 时会引发 `RuntimeError` 异常
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[True], [FalseThenTrue()]])
        # 断言调用 `np.nonzero` 时会引发 `RuntimeError` 异常
        assert_raises(RuntimeError, np.nonzero, a)

        # 在第二次循环中结果缩小
        a = np.array([False, TrueThenFalse()])
        # 断言调用 `np.nonzero` 时会引发 `RuntimeError` 异常
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[False], [TrueThenFalse()]])
        # 断言调用 `np.nonzero` 时会引发 `RuntimeError` 异常
        assert_raises(RuntimeError, np.nonzero, a)

    def test_nonzero_sideffects_structured_void(self):
        # 检查结构化的 void 类型不会改变原始数组的对齐标志
        arr = np.zeros(5, dtype="i1,i8,i8")  # `ones` 可能会短路
        assert arr.flags.aligned  # 结构体被认为是“对齐的”
        assert not arr["f2"].flags.aligned
        # 确保 `nonzero/count_nonzero` 不会翻转标志：
        np.nonzero(arr)
        assert arr.flags.aligned
        np.count_nonzero(arr)
        assert arr.flags.aligned

    def test_nonzero_exception_safe(self):
        # gh-13930

        # 定义一个名为 `ThrowsAfter` 的类
        class ThrowsAfter:
            def __init__(self, iters):
                self.iters_left = iters

            # 定义 `__bool__` 方法，如果 `iters_left` 为 0 则抛出 `ValueError` 异常，否则返回 True
            def __bool__(self):
                if self.iters_left == 0:
                    raise ValueError("called `iters` times")

                self.iters_left -= 1
                return True

        """
        测试确保在出现错误状态时会引发 `ValueError` 而不是 `SystemError`

        如果在设置错误状态后调用 `__bool__` 函数，Python (cpython) 将引发 `SystemError`。
        """

        # 断言第一次循环中的异常被正确处理
        a = np.array([ThrowsAfter(5)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # 对于一维循环，在第二次循环中引发异常
        a = np.array([ThrowsAfter(15)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # 对于 n 维循环，在第二次循环中引发异常
        a = np.array([[ThrowsAfter(15)]]*10)
        assert_raises(ValueError, np.nonzero, a)
    # 定义一个测试方法，用于验证结构化数据类型的线程安全性
    def test_structured_threadsafety(self):
        # 对于结构化数据类型，非零和一些其他函数应该是线程安全的，参见 gh-15387。
        # 此测试可能表现出随机性。

        # 导入线程池执行器
        from concurrent.futures import ThreadPoolExecutor

        # 创建一个深度嵌套的 dtype，增加出错概率：
        dt = np.dtype([("", "f8")])
        dt = np.dtype([("", dt)])
        dt = np.dtype([("", dt)] * 2)

        # 数组应该足够大，以确保可能出现线程问题
        arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]

        # 定义一个处理函数，调用数组的非零元素查找函数
        def func(arr):
            arr.nonzero()

        # 创建线程池执行器，最大工作线程数为 8
        tpe = ThreadPoolExecutor(max_workers=8)

        # 提交任务到线程池
        futures = [tpe.submit(func, arr) for _ in range(10)]
        
        # 等待所有任务完成
        for f in futures:
            f.result()

        # 断言数组的 dtype 应该与预期的 dtype 相同
        assert arr.dtype is dt
class TestIndex:
python
class TestIndex:
    def test_boolean(self):
        # 生成一个3x5x8的随机整数数组
        a = rand(3, 5, 8)
        # 生成一个5x8的随机浮点数数组
        V = rand(5, 8)
        # 生成一个长度为15的随机整数数组，元素取值范围为0到5
        g1 = randint(0, 5, size=15)
        # 生成一个长度为15的随机整数数组，元素取值范围为0到8
        g2 = randint(0, 8, size=15)
        # 根据索引g1和g2，将V中对应位置的值取反
        V[g1, g2] = -V[g1, g2]
        # 断言：通过布尔索引，验证a数组中V大于0的元素是否等于a[:, V > 0]中的元素
        assert_((np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all())

    def test_boolean_edgecase(self):
        # 创建一个空的int32类型的数组a
        a = np.array([], dtype='int32')
        # 创建一个空的布尔类型数组b
        b = np.array([], dtype='bool')
        # 使用布尔数组b作为索引，从数组a中选择元素构成新数组c
        c = a[b]
        # 断言：验证c数组是否为空数组
        assert_equal(c, [])
        # 断言：验证c数组的数据类型是否为int32
        assert_equal(c.dtype, np.dtype('int32'))


class TestBinaryRepr:
    def test_zero(self):
        # 断言：验证0的二进制表示是否为字符串'0'
        assert_equal(np.binary_repr(0), '0')

    def test_positive(self):
        # 断言：验证10的二进制表示是否为字符串'1010'
        assert_equal(np.binary_repr(10), '1010')
        # 断言：验证12522的二进制表示是否为字符串'11000011101010'
        assert_equal(np.binary_repr(12522),
                     '11000011101010')
        # 断言：验证10736848的二进制表示是否为字符串'101000111101010011010000'
        assert_equal(np.binary_repr(10736848),
                     '101000111101010011010000')

    def test_negative(self):
        # 断言：验证-1的二进制表示是否为字符串'-1'
        assert_equal(np.binary_repr(-1), '-1')
        # 断言：验证-10的二进制表示是否为字符串'-1010'
        assert_equal(np.binary_repr(-10), '-1010')
        # 断言：验证-12522的二进制表示是否为字符串'-11000011101010'
        assert_equal(np.binary_repr(-12522),
                     '-11000011101010')
        # 断言：验证-10736848的二进制表示是否为字符串'-101000111101010011010000'
        assert_equal(np.binary_repr(-10736848),
                     '-101000111101010011010000')

    def test_sufficient_width(self):
        # 断言：验证0的二进制表示，指定宽度为5，是否为字符串'00000'
        assert_equal(np.binary_repr(0, width=5), '00000')
        # 断言：验证10的二进制表示，指定宽度为7，是否为字符串'0001010'
        assert_equal(np.binary_repr(10, width=7), '0001010')
        # 断言：验证-5的二进制表示，指定宽度为7，是否为字符串'1111011'
        assert_equal(np.binary_repr(-5, width=7), '1111011')

    def test_neg_width_boundaries(self):
        # 见gh-8670

        # 确保在进行更彻底的测试之前，不会破坏问题中的示例。
        # 断言：验证-128的二进制表示，指定宽度为8，是否为字符串'10000000'
        assert_equal(np.binary_repr(-128, width=8), '10000000')

        # 循环测试负数的边界情况
        for width in range(1, 11):
            # 计算-2^(width-1)
            num = -2**(width - 1)
            # 期望的二进制表示结果
            exp = '1' + (width - 1) * '0'
            # 断言：验证num的二进制表示，指定宽度为width，是否等于exp
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        # 见gh-14289.
        # 断言：验证-2**62的int64类型的二进制表示，指定宽度为64，是否为字符串'11'后跟62个'0'
        assert_equal(np.binary_repr(np.int64(-2**62), width=64),
                     '11' + '0'*62)


class TestBaseRepr:
    def test_base3(self):
        # 断言：验证3^5的3进制表示是否为字符串'100000'
        assert_equal(np.base_repr(3**5, 3), '100000')

    def test_positive(self):
        # 断言：验证12的10进制表示是否为字符串'12'
        assert_equal(np.base_repr(12, 10), '12')
        # 断言：验证12的10进制表示，指定宽度为4，是否为字符串'000012'
        assert_equal(np.base_repr(12, 10, 4), '000012')
        # 断言：验证12的4进制表示是否为字符串'30'
        assert_equal(np.base_repr(12, 4), '30')
        # 断言：验证3731624803700888的36进制表示是否为字符串'10QR0ROFCEW'
        assert_equal(np.base_repr(3731624803700888, 36), '10QR0ROFCEW')

    def test_negative(self):
        # 断言：验证-12的10进制表示是否为字符串'-12'
        assert_equal(np.base_repr(-12, 10), '-12')
        # 断言：验证-12的10进制表示，指定宽度为4，是否为字符串'-000012'
        assert_equal(np.base_repr(-12, 10, 4), '-000012')
        # 断言：验证-12的4进制表示是否为字符串'-30'
        assert_equal(np.base_repr(-12, 4), '-30')

    def test_base_range(self):
        # 断言：验证基数为1时，抛出ValueError异常
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        # 断言：验证基数为37时，抛出ValueError异常
        with assert_raises(ValueError):
            np.base_repr(1, 37)

    def test_minimal_signed_int(self):
        # 断言：验证np.int8(-128)的最小二进制表示是否为字符串'-10000000'
        assert_equal(np.base_repr(np.int8(-128)), '-10000000')
    # 这些是0维数组，曾经是一个特殊情况，
    # 在(e0 == e0).all()中会触发异常
    e0 = np.array(0, dtype="int")
    e1 = np.array(1, dtype="float")
    
    # 生成器函数的yield语句，产生一组测试数据，
    # 每组数据由四个元素组成：x, y, nan_equal, expected_result
    yield (e0, e0.copy(), None, True)
    yield (e0, e0.copy(), False, True)
    yield (e0, e0.copy(), True, True)
    
    # 
    yield (e1, e1.copy(), None, True)
    yield (e1, e1.copy(), False, True)
    yield (e1, e1.copy(), True, True)
    
    # 不包含NaN值的数组
    a12 = np.array([1, 2])
    a12b = a12.copy()
    a123 = np.array([1, 2, 3])
    a13 = np.array([1, 3])
    a34 = np.array([3, 4])
    
    aS1 = np.array(["a"], dtype="S1")
    aS1b = aS1.copy()
    aS1u4 = np.array([("a", 1)], dtype="S1,u4")
    aS1u4b = aS1u4.copy()
    
    yield (a12, a12b, None, True)
    yield (a12, a12, None, True)
    yield (a12, a123, None, False)
    yield (a12, a34, None, False)
    yield (a12, a13, None, False)
    yield (aS1, aS1b, None, True)
    yield (aS1, aS1, None, True)
    
    # 非浮点数数据类型，equal_nan应无影响，
    yield (a123, a123, None, True)
    yield (a123, a123, False, True)
    yield (a123, a123, True, True)
    yield (a123, a123.copy(), None, True)
    yield (a123, a123.copy(), False, True)
    yield (a123, a123.copy(), True, True)
    yield (a123.astype("float"), a123.astype("float"), None, True)
    yield (a123.astype("float"), a123.astype("float"), False, True)
    yield (a123.astype("float"), a123.astype("float"), True, True)
    
    # 可以包含None值的数组
    b1 = np.array([1, 2, np.nan])
    b2 = np.array([1, np.nan, 2])
    b3 = np.array([1, 2, np.inf])
    b4 = np.array(np.nan)
    
    # 实例相同
    yield (b1, b1, None, False)
    yield (b1, b1, False, False)
    yield (b1, b1, True, True)
    
    # 相等但不是同一实例
    yield (b1, b1.copy(), None, False)
    yield (b1, b1.copy(), False, False)
    yield (b1, b1.copy(), True, True)
    
    # 一旦去除NaN，相同
    yield (b1, b2, None, False)
    yield (b1, b2, False, False)
    yield (b1, b2, True, False)
    
    # NaN与Inf不混淆
    yield (b1, b3, None, False)
    yield (b1, b3, False, False)
    yield (b1, b3, True, False)
    
    # 全是NaN
    yield (b4, b4, None, False)
    yield (b4, b4, False, False)
    yield (b4, b4, True, True)
    yield (b4, b4.copy(), None, False)
    yield (b4, b4.copy(), False, False)
    yield (b4, b4.copy(), True, True)
    
    t1 = b1.astype("timedelta64")
    t2 = b2.astype("timedelta64")
    
    # 时间增量具有特定性质
    yield (t1, t1, None, False)
    yield (t1, t1, False, False)
    yield (t1, t1, True, True)
    
    yield (t1, t1.copy(), None, False)
    yield (t1, t1.copy(), False, False)
    yield (t1, t1.copy(), True, True)
    
    yield (t1, t2, None, False)
    yield (t1, t2, False, False)
    yield (t1, t2, True, False)
    
    # 多维数组
    md1 = np.array([[0, 1], [np.nan, 1]])
    
    yield (md1, md1, None, False)
    # 生成器函数中的第一个 yield 语句，返回元组 (md1, md1, False, False)
    yield (md1, md1, False, False)
    # 生成器函数中的第二个 yield 语句，返回元组 (md1, md1, True, True)
    yield (md1, md1, True, True)
    # 生成器函数中的第三个 yield 语句，返回元组 (md1, md1.copy(), None, False)
    yield (md1, md1.copy(), None, False)
    # 生成器函数中的第四个 yield 语句，返回元组 (md1, md1.copy(), False, False)
    yield (md1, md1.copy(), False, False)
    # 生成器函数中的第五个 yield 语句，返回元组 (md1, md1.copy(), True, True)
    yield (md1, md1.copy(), True, True)

    # 创建两个复数值相同的数组，都为 nan+nan.j，但是是同一个实例
    cplx1, cplx2 = [np.array([np.nan + np.nan * 1j])] * 2

    # 创建两个复数，其中一个实部为 1，虚部为 nan；另一个实部为 nan，虚部为 1
    cplx3, cplx4 = np.complex64(1, np.nan), np.complex64(np.nan, 1)

    # 生成器函数中的下一组 yield 语句，返回两个复数 cplx1 和 cplx2 的元组，以及其他参数
    yield (cplx1, cplx2, None, False)
    # 生成器函数中的下一组 yield 语句，返回两个复数 cplx1 和 cplx2 的元组，以及其他参数
    yield (cplx1, cplx2, False, False)
    # 生成器函数中的下一组 yield 语句，返回两个复数 cplx1 和 cplx2 的元组，以及其他参数
    yield (cplx1, cplx2, True, True)

    # 生成器函数中的下一组 yield 语句，返回两个复数 cplx3 和 cplx4 的元组，以及其他参数
    yield (cplx3, cplx4, None, False)
    # 生成器函数中的下一组 yield 语句，返回两个复数 cplx3 和 cplx4 的元组，以及其他参数
    yield (cplx3, cplx4, False, False)
    # 生成器函数中的最后一个 yield 语句，返回两个复数 cplx3 和 cplx4 的元组，以及其他参数
    yield (cplx3, cplx4, True, True)
# 定义一个测试类 TestArrayComparisons，用于测试数组比较的各种情况
class TestArrayComparisons:
    
    # 使用 pytest 的参数化装饰器，为 test_array_equal_equal_nan 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "bx,by,equal_nan,expected", _test_array_equal_parametrizations()
    )
    def test_array_equal_equal_nan(self, bx, by, equal_nan, expected):
        """
        This test array_equal for a few combinaison:

        - are the two inputs the same object or not (same object many not
          be equal if contains NaNs)
        - Whether we should consider or not, NaNs, being equal.

        """
        # 根据 equal_nan 参数决定是否考虑 NaN，在 numpy 中比较 bx 和 by 数组是否相等
        if equal_nan is None:
            res = np.array_equal(bx, by)
        else:
            res = np.array_equal(bx, by, equal_nan=equal_nan)
        # 断言比较结果是否符合预期
        assert_(res is expected)
        # 断言返回结果的类型是否为布尔值
        assert_(type(res) is bool)

    # 测试 None 在数组元素比较中的行为
    def test_none_compares_elementwise(self):
        # 创建一个包含 None 的对象数组 a
        a = np.array([None, 1, None], dtype=object)
        # 断言数组元素与 None 的比较结果是否符合预期
        assert_equal(a == None, [True, False, True])
        assert_equal(a != None, [False, True, False])

        # 创建一个全为 1 的数组 a
        a = np.ones(3)
        # 断言数组元素与 None 的比较结果是否符合预期
        assert_equal(a == None, [False, False, False])
        assert_equal(a != None, [True, True, True])

    # 测试数组是否等价的函数 np.array_equiv 的行为
    def test_array_equiv(self):
        # 比较两个相等的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        # 比较两个不等长的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        # 比较两个不相等的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        # 比较两个部分相等的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

        # 比较两个相等的数组是否等价
        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        assert_(res)
        assert_(type(res) is bool)
        # 比较两个二维数组是否等价
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        assert_(res)
        assert_(type(res) is bool)
        # 比较两个不等长的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        # 比较两个不同维度的数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        assert_(not res)
        assert_(type(res) is bool)
        # 比较两个不等长的二维数组是否等价
        res = np.array_equiv(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert_(not res)
        assert_(type(res) is bool)

    # 使用 pytest 的参数化装饰器，为 test_compare_unstructured_voids 方法提供数据类型参数化测试
    @pytest.mark.parametrize("dtype", ["V0", "V3", "V10"])
    def test_compare_unstructured_voids(self, dtype):
        # 创建一个全为零的指定数据类型的数组
        zeros = np.zeros(3, dtype=dtype)

        # 断言数组与自身的比较结果是否全为 True
        assert_array_equal(zeros, zeros)
        # 断言数组与自身的不等比较结果是否全为 False
        assert not (zeros != zeros).any()

        # 对于数据类型为 "V0" 的情况，无法测试实际不同数据的不等比较
        if dtype == "V0":
            return

        # 创建一个包含非零数据的指定数据类型的数组
        nonzeros = np.array([b"1", b"2", b"3"], dtype=dtype)

        # 断言数组 zeros 与 nonzeros 的等比较结果是否全为 False
        assert not (zeros == nonzeros).any()
        # 断言数组 zeros 与 nonzeros 的不等比较结果是否全为 True
        assert (zeros != nonzeros).all()


# 定义一个辅助函数 assert_array_strict_equal，用于严格比较两个数组是否相等
def assert_array_strict_equal(x, y):
    assert_array_equal(x, y)
    # 检查标志位，32 位架构通常不提供 16 字节对齐
    if ((x.dtype.alignment <= 8 or
            np.intp().dtype.itemsize != 4) and
            sys.platform != 'win32'):
        assert_(x.flags == y.flags)
    else:
        # 断言两个数组的owndata属性相同
        assert_(x.flags.owndata == y.flags.owndata)
        # 断言两个数组的writeable属性相同
        assert_(x.flags.writeable == y.flags.writeable)
        # 断言两个数组的C顺序连续性属性相同
        assert_(x.flags.c_contiguous == y.flags.c_contiguous)
        # 断言两个数组的Fortran顺序连续性属性相同
        assert_(x.flags.f_contiguous == y.flags.f_contiguous)
        # 断言两个数组的writebackifcopy属性相同
        assert_(x.flags.writebackifcopy == y.flags.writebackifcopy)
    # 检查数组的字节顺序是否相同
    assert_(x.dtype.isnative == y.dtype.isnative)
class TestClip:
    # 设置测试前的准备方法，初始化 self.nr 和 self.nc
    def setup_method(self):
        self.nr = 5
        self.nc = 3

    # 使用 a.clip 方法实现快速剪裁操作
    def fastclip(self, a, m, M, out=None, **kwargs):
        return a.clip(m, M, out=out, **kwargs)

    # 使用选择器方法验证 fastclip 结果
    def clip(self, a, m, M, out=None):
        # 使用 np.less 和 np.greater 生成选择器
        selector = np.less(a, m) + 2*np.greater(a, M)
        # 使用选择器进行元素选择，输出到 out 参数
        return selector.choose((a, m, M), out=out)

    # 便利函数：生成 n 行 m 列的随机数数组
    def _generate_data(self, n, m):
        return randn(n, m)

    # 便利函数：生成 n 行 m 列的复数随机数数组
    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.j * rand(n, m)

    # 便利函数：生成 n 行 m 列的单精度浮点数数组
    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(np.float32)

    # 便利函数：根据系统字节顺序对数组进行转换
    def _neg_byteorder(self, a):
        a = np.asarray(a)
        if sys.byteorder == 'little':
            a = a.astype(a.dtype.newbyteorder('>'))
        else:
            a = a.astype(a.dtype.newbyteorder('<'))
        return a

    # 便利函数：生成非本机字节顺序的数据数组
    def _generate_non_native_data(self, n, m):
        data = randn(n, m)
        data = self._neg_byteorder(data)
        # 断言数据类型不是本机字节顺序
        assert_(not data.dtype.isnative)
        return data

    # 便利函数：生成 n 行 m 列的 64 位整数数组
    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int64)

    # 便利函数：生成 n 行 m 列的 32 位整数数组
    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int32)

    # 实际测试用例

    # 使用 pytest 参数化标记，测试特定数据类型的 clip 方法
    @pytest.mark.parametrize("dtype", '?bhilqpBHILQPefdgFDGO')
    def test_ones_pathological(self, dtype):
        # 用于保留 gh-12519 中描述的行为；amin > amax 的行为可能在未来发生变化
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        # 使用 np.clip 方法，将数组 arr 中小于 0 和大于 1 的值裁剪到 0 和 1 之间
        actual = np.clip(arr, 1, 0)
        if dtype == 'O':
            # 断言数组 actual 和 expected 的列表表示相同
            assert actual.tolist() == expected.tolist()
        else:
            # 断言数组 actual 和 expected 严格相等
            assert_equal(actual, expected)

    # 测试简单的双精度输入
    def test_simple_double(self):
        # 使用 _generate_data 生成形状为 (self.nr, self.nc) 的数组 a
        a = self._generate_data(self.nr, self.nc)
        m = 0.1
        M = 0.6
        # 使用 fastclip 方法进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用 clip 方法进行剪裁操作
        act = self.clip(a, m, M)
        # 断言数组 ac 和 act 严格相等
        assert_array_strict_equal(ac, act)

    # 测试简单的整数输入
    def test_simple_int(self):
        # 使用 _generate_int_data 生成形状为 (self.nr, self.nc) 的整数数组 a
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(int)
        m = -2
        M = 4
        # 使用 fastclip 方法进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用 clip 方法进行剪裁操作
        act = self.clip(a, m, M)
        # 断言数组 ac 和 act 严格相等
        assert_array_strict_equal(ac, act)

    # 测试双精度输入和数组形式的 min/max
    def test_array_double(self):
        # 使用 _generate_data 生成形状为 (self.nr, self.nc) 的数组 a
        a = self._generate_data(self.nr, self.nc)
        m = np.zeros(a.shape)
        M = m + 0.5
        # 使用 fastclip 方法进行剪裁操作
        ac = self.fastclip(a, m, M)
        # 使用 clip 方法进行剪裁操作
        act = self.clip(a, m, M)
        # 断言数组 ac 和 act 严格相等
        assert_array_strict_equal(ac, act)
    def test_simple_nonnative(self):
        # Test non native double input with scalar min/max.
        # 测试非本地双精度输入与标量最小/最大值。
        a = self._generate_non_native_data(self.nr, self.nc)
        # 设定最小值和最大值
        m = -0.5
        M = 0.6
        # 使用快速剪裁函数对数据进行处理
        ac = self.fastclip(a, m, M)
        # 使用普通剪裁函数对数据进行处理
        act = self.clip(a, m, M)
        # 断言处理后的数组相等
        assert_array_equal(ac, act)

        # Test native double input with non native double scalar min/max.
        # 测试本地双精度输入与非本地双精度标量最小/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = self._neg_byteorder(0.6)
        # 断言最大值不是本地类型
        assert_(not M.dtype.isnative)
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        # 断言处理后的数组相等
        assert_array_equal(ac, act)

    def test_simple_complex(self):
        # Test native complex input with native double scalar min/max.
        # 测试本地复杂输入与本地双精度标量最小/最大值。
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        m = -0.5
        M = 1.
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        # 严格断言处理后的数组相等
        assert_array_strict_equal(ac, act)

        # Test native input with complex double scalar min/max.
        # 测试本地输入与复杂双精度标量最小/最大值。
        a = 3 * self._generate_data(self.nr, self.nc)
        m = -0.5 + 1.j
        M = 1. + 2.j
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        # 严格断言处理后的数组相等
        assert_array_strict_equal(ac, act)

    def test_clip_complex(self):
        # Address Issue gh-5354 for clipping complex arrays
        # Test native complex input without explicit min/max
        # ie, either min=None or max=None
        # 解决复杂数组剪裁的问题 gh-5354
        # 测试本地复杂输入，没有显式的最小/最大值，即 min=None 或 max=None
        a = np.ones(10, dtype=complex)
        m = a.min()
        M = a.max()
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        # 严格断言处理后的数组相等
        assert_array_strict_equal(am, a)
        assert_array_strict_equal(aM, a)

    def test_clip_non_contig(self):
        # Test clip for non contiguous native input and native scalar min/max.
        # 测试非连续本地输入和本地标量最小/最大值的剪裁。
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        # 断言数组不是 F-contiguous 或 C-contiguous
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        # 严格断言处理后的数组相等
        assert_array_strict_equal(ac, act)

    def test_simple_out(self):
        # Test native double input with scalar min/max.
        # 测试本地双精度输入与标量最小/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        # 使用剪裁函数，并将结果存储在给定的数组中
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        # 严格断言处理后的数组相等
        assert_array_strict_equal(ac, act)

    @pytest.mark.parametrize("casting", [None, "unsafe"])
    # 测试函数：使用 fastclip 方法对输入数据进行截断处理，验证输出是否符合预期
    def test_simple_int32_inout(self, casting):
        # 生成一个 int32 类型的数据数组
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M，类型为 np.float64
        m = np.float64(0)
        M = np.float64(2)
        # 创建一个与 a 相同形状的 int32 类型的零数组 ac 和其副本 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 如果 casting 为 None，则期望抛出 TypeError 异常
        if casting is None:
            with pytest.raises(TypeError):
                self.fastclip(a, m, M, ac, casting=casting)
        else:
            # 否则，使用 fastclip 方法进行截断操作，并将结果存储在 ac 中
            self.fastclip(a, m, M, ac, casting=casting)
            # 同时使用 clip 方法进行截断操作，并将结果存储在 act 中
            self.clip(a, m, M, act)
            # 断言 ac 和 act 数组在严格意义上相等
            assert_array_strict_equal(ac, act)

    # 测试函数：验证 fastclip 方法在 int32 输入、int64 输出条件下的截断效果
    def test_simple_int64_out(self):
        # 生成一个 int32 类型的数据数组
        a = self._generate_int32_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M，类型为 np.int32
        m = np.int32(-1)
        M = np.int32(1)
        # 创建一个与 a 相同形状的 int64 类型的零数组 ac 和其副本 act
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        # 使用 fastclip 方法进行截断操作，并将结果存储在 ac 中
        self.fastclip(a, m, M, ac)
        # 同时使用 clip 方法进行截断操作，并将结果存储在 act 中
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组在严格意义上相等
        assert_array_strict_equal(ac, act)

    # 测试函数：验证 fastclip 方法在 int32 输入、int32 输出条件下的截断效果
    def test_simple_int64_inout(self):
        # 生成一个 int32 类型的数据数组
        a = self._generate_int32_data(self.nr, self.nc)
        # 创建一个与 a 相同形状的 float64 类型的零数组 m，并设置最大值 M 为 float64 类型的 1.0
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        # 创建一个与 a 相同形状的 int32 类型的零数组 ac 和其副本 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用 fastclip 方法进行截断操作，并将结果存储在 ac 中，传递 "unsafe" 作为 casting 参数以消除警告
        self.fastclip(a, m, M, out=ac, casting="unsafe")
        # 同时使用 clip 方法进行截断操作，并将结果存储在 act 中
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组在严格意义上相等
        assert_array_strict_equal(ac, act)

    # 测试函数：验证 fastclip 方法在 double 输入、int32 输出条件下的截断效果
    def test_simple_int32_out(self):
        # 生成一个 double 类型的数据数组
        a = self._generate_data(self.nr, self.nc)
        # 设置最小值 m 和最大值 M，类型为 double
        m = -1.0
        M = 2.0
        # 创建一个与 a 相同形状的 int32 类型的零数组 ac 和其副本 act
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用 fastclip 方法进行截断操作，并将结果存储在 ac 中，传递 "unsafe" 作为 casting 参数以消除警告
        self.fastclip(a, m, M, out=ac, casting="unsafe")
        # 同时使用 clip 方法进行截断操作，并将结果存储在 act 中
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组在严格意义上相等
        assert_array_strict_equal(ac, act)

    # 测试函数：验证 fastclip 方法在 double 输入、double 数组 min/max 条件下的 in-place 截断效果
    def test_simple_inplace_01(self):
        # 生成一个 double 类型的数据数组，并将其副本保存在 ac 中
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        # 创建一个与 a 相同形状的零数组 m，并设置最大值 M 为 double 类型的 1.0
        m = np.zeros(a.shape)
        M = 1.0
        # 使用 fastclip 方法进行 in-place 截断操作
        self.fastclip(a, m, M, a)
        # 同时使用 clip 方法进行截断操作，并将结果存储在 ac 中
        self.clip(a, m, M, ac)
        # 断言 a 和 ac 数组在严格意义上相等
        assert_array_strict_equal(a, ac)

    # 测试函数：验证 fastclip 方法在 double 输入、scalar min/max 条件下的 in-place 截断效果
    def test_simple_inplace_02(self):
        # 生成一个 double 类型的数据数组，并将其副本保存在 ac 中
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        # 设置最小值 m 和最大值 M，类型为 double
        m = -0.5
        M = 0.6
        # 使用 fastclip 方法进行 in-place 截断操作
        self.fastclip(a, m, M, a)
        # 同时使用 clip 方法进行截断操作，并将结果存储在 ac 中
        self.clip(ac, m, M, ac)
        # 断言 a 和 ac 数组在严格意义上相等
        assert_array_strict_equal(a, ac)

    # 测试函数：验证 fastclip 方法在非连续的 double 输入数据和 double scalar min/max 条件下的 in-place 截断效果
    def test_noncontig_inplace(self):
        # 生成一个双倍大小的 double 类型数据数组，并按步长取样以创建非连续数组 a
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        # 断言 a 不是 Fortran 连续的
        assert_(not a.flags['F_CONTIGUOUS'])
        # 断言 a 不是 C 连续的
        assert_(not a.flags['C_CONTIGUOUS'])
        # 将 a 的副本保存在 ac 中
        ac = a.copy()
        # 设置最小值 m 和最大值 M，类型为 double
        m = -0.5
        M = 0.6
        # 使用 fastclip 方法进行 in-place 截断操作
        self.fastclip(a, m, M, a)
        # 同时使用 clip 方法进行截断操作，并将结果存储在 ac 中
        self.clip(ac, m, M, ac)
        # 断言 a 和 ac 数组在内容上相等
        assert_array_equal(a, ac)
    def test_type_cast_01(self):
        # 测试原生双精度输入和标量最小/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        # 使用快速剪切函数处理数据
        ac = self.fastclip(a, m, M)
        # 使用普通剪切函数处理数据
        act = self.clip(a, m, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_02(self):
        # 测试原生 int32 输入和 int32 标量最小/最大值。
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(np.int32)
        m = -2
        M = 4
        # 使用快速剪切函数处理数据
        ac = self.fastclip(a, m, M)
        # 使用普通剪切函数处理数据
        act = self.clip(a, m, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_03(self):
        # 测试原生 int32 输入和 float64 标量最小/最大值。
        a = self._generate_int32_data(self.nr, self.nc)
        m = -2
        M = 4
        # 使用快速剪切函数处理数据，将标量最小/最大值转换为 float64
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        # 使用普通剪切函数处理数据，将标量最小/最大值转换为 float64
        act = self.clip(a, np.float64(m), np.float64(M))
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_04(self):
        # 测试原生 int32 输入和 float32 标量最小/最大值。
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float32(-2)
        M = np.float32(4)
        # 使用快速剪切函数处理数据
        act = self.fastclip(a, m, M)
        # 使用普通剪切函数处理数据
        ac = self.clip(a, m, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_05(self):
        # 测试原生 int32 输入和双精度数组最小/最大值。
        a = self._generate_int_data(self.nr, self.nc)
        m = -0.5
        M = 1.
        # 使用快速剪切函数处理数据，将数组最小值转换为双精度
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        # 使用普通剪切函数处理数据，将数组最小值转换为双精度
        act = self.clip(a, m * np.zeros(a.shape), M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_06(self):
        # 测试原生输入和非原生标量最小值/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = 0.5
        m_s = self._neg_byteorder(m)
        M = 1.
        # 使用普通剪切函数处理数据
        act = self.clip(a, m_s, M)
        # 使用快速剪切函数处理数据
        ac = self.fastclip(a, m_s, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_07(self):
        # 测试非原生输入和原生数组最小值/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.
        # 将输入数据转换为非原生字节顺序
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        # 使用普通剪切函数处理数据
        act = a_s.clip(m, M)
        # 使用快速剪切函数处理数据
        ac = self.fastclip(a_s, m, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)

    def test_type_cast_08(self):
        # 测试非原生输入和原生标量最小值/最大值。
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.
        # 将输入数据转换为非原生字节顺序
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        # 使用快速剪切函数处理数据
        ac = self.fastclip(a_s, m, M)
        # 使用普通剪切函数处理数据
        act = a_s.clip(m, M)
        # 断言两个处理结果的严格相等性
        assert_array_strict_equal(ac, act)
    def test_type_cast_09(self):
        # 测试使用非本地数组的最小/最大值进行本地转换。
        # 生成测试数据
        a = self._generate_data(self.nr, self.nc)
        # 创建一个与 a 形状相同的全为 -0.5 的数组 m
        m = -0.5 * np.ones(a.shape)
        # 最大值 M 为 1
        M = 1.
        # 将 m 转换为非本地字节顺序
        m_s = self._neg_byteorder(m)
        # 断言 m_s 的数据类型不是本地的
        assert_(not m_s.dtype.isnative)
        # 使用 fastclip 方法对 a 进行剪裁操作，使用 m_s 和 M 作为参数
        ac = self.fastclip(a, m_s, M)
        # 使用 clip 方法对 a 进行剪裁操作，使用 m_s 和 M 作为参数
        act = self.clip(a, m_s, M)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_type_cast_10(self):
        # 测试本地 int32 输入与浮点 min/max 以及浮点输出参数的情况。
        # 生成测试数据
        a = self._generate_int_data(self.nr, self.nc)
        # 创建一个与 a 形状相同的全为零的数组 b，数据类型为 np.float32
        b = np.zeros(a.shape, dtype=np.float32)
        # 最小值 m 和最大值 M 使用 np.float32 类型
        m = np.float32(-0.5)
        M = np.float32(1)
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 b
        act = self.clip(a, m, M, out=b)
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 b
        ac = self.fastclip(a, m, M, out=b)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_type_cast_11(self):
        # 测试非本地数据与本地标量、最小/最大值以及非本地输出参数的情况。
        # 生成测试数据
        a = self._generate_non_native_data(self.nr, self.nc)
        # 创建 b 作为 a 的副本，并将其转换为与 a 不同的字节顺序
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder('>'))
        # 创建 bt 作为 b 的副本
        bt = b.copy()
        # 最小值 m 为 -0.5，最大值 M 为 1
        m = -0.5
        M = 1.
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 b
        self.fastclip(a, m, M, out=b)
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 bt
        self.clip(a, m, M, out=bt)
        # 断言 b 和 bt 数组严格相等
        assert_array_strict_equal(b, bt)

    def test_type_cast_12(self):
        # 测试本地 int32 输入与 int32 min/max 以及浮点输出参数的情况。
        # 生成测试数据
        a = self._generate_int_data(self.nr, self.nc)
        # 创建一个与 a 形状相同的全为零的数组 b，数据类型为 np.float32
        b = np.zeros(a.shape, dtype=np.float32)
        # 最小值 m 和最大值 M 使用 np.int32 类型
        m = np.int32(0)
        M = np.int32(1)
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 b
        act = self.clip(a, m, M, out=b)
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 b
        ac = self.fastclip(a, m, M, out=b)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple(self):
        # 测试本地 double 输入与标量 min/max 的情况。
        # 生成测试数据
        a = self._generate_data(self.nr, self.nc)
        # 最小值 m 为 -0.5，最大值 M 为 0.6
        m = -0.5
        M = 0.6
        # 创建与 a 形状相同的全零数组 ac 和 act
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 ac
        self.fastclip(a, m, M, ac)
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple2(self):
        # 测试本地 int32 输入与 double min/max 以及 int32 输出参数的情况。
        # 生成测试数据
        a = self._generate_int32_data(self.nr, self.nc)
        # 最小值 m 和最大值 M 使用 np.float64 类型
        m = np.float64(0)
        M = np.float64(2)
        # 创建一个与 a 形状相同的全为零的数组 ac，数据类型为 np.int32
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 ac，并使用 "unsafe" 转换
        self.fastclip(a, m, M, out=ac, casting="unsafe")
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        # 测试本地 int32 输入与 int32 scalar min/max 以及 int64 输出参数的情况。
        # 生成测试数据
        a = self._generate_int32_data(self.nr, self.nc)
        # 最小值 m 和最大值 M 使用 np.int32 类型
        m = np.int32(-1)
        M = np.int32(1)
        # 创建一个与 a 形状相同的全为零的数组 ac 和 act，数据类型为 np.int64
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        # 使用 fastclip 方法对 a 进行剪裁操作，指定输出参数为 ac
        self.fastclip(a, m, M, ac)
        # 使用 clip 方法对 a 进行剪裁操作，指定输出参数为 act
        self.clip(a, m, M, act)
        # 断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)
    def test_clip_with_out_array_int32(self):
        # 测试当输出为 int32 数组时，使用 native int32 输入，设置双数组的最小/最大值和 int32 输出
        a = self._generate_int32_data(self.nr, self.nc)
        # 创建与 a 相同形状的全零数组 m，数据类型为 np.float64
        m = np.zeros(a.shape, np.float64)
        # 创建一个标量 M，数据类型为 np.float64
        M = np.float64(1)
        # 创建一个与 a 相同形状的全零数组 ac，数据类型为 np.int32
        ac = np.zeros(a.shape, dtype=np.int32)
        # 对 ac 进行深拷贝，生成 act 数组
        act = ac.copy()
        # 调用 self.fastclip 方法，使用不安全的类型转换 ("unsafe")，将结果输出到 ac 中
        self.fastclip(a, m, M, out=ac, casting="unsafe")
        # 调用 self.clip 方法，将结果输出到 act 中
        self.clip(a, m, M, act)
        # 使用 assert_array_strict_equal 函数断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_outint32(self):
        # 测试当输出为 int 数组时，使用 native double 输入，设置标量的最小/最大值和 int 输出
        a = self._generate_data(self.nr, self.nc)
        # 设置 m 和 M 的值分别为 -1.0 和 2.0
        m = -1.0
        M = 2.0
        # 创建一个与 a 相同形状的全零数组 ac，数据类型为 np.int32
        ac = np.zeros(a.shape, dtype=np.int32)
        # 对 ac 进行深拷贝，生成 act 数组
        act = ac.copy()
        # 调用 self.fastclip 方法，使用不安全的类型转换 ("unsafe")，将结果输出到 ac 中
        self.fastclip(a, m, M, out=ac, casting="unsafe")
        # 调用 self.clip 方法，将结果输出到 act 中
        self.clip(a, m, M, act)
        # 使用 assert_array_strict_equal 函数断言 ac 和 act 数组严格相等
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_transposed(self):
        # 测试当输出为转置数组时，验证 out 参数在转置时的工作情况
        a = np.arange(16).reshape(4, 4)
        # 创建一个与 a 形状相同的空数组 out，并进行转置操作
        out = np.empty_like(a).T
        # 使用 a.clip 方法，将介于 4 和 10 之间的值裁剪到 out 中
        a.clip(4, 10, out=out)
        # 调用 self.clip 方法，裁剪介于 4 和 10 之间的 a，并生成预期结果 expected
        expected = self.clip(a, 4, 10)
        # 使用 assert_array_equal 函数断言 out 和 expected 数组相等
        assert_array_equal(out, expected)

    def test_clip_with_out_memory_overlap(self):
        # 测试当 out 参数具有内存重叠时的工作情况
        a = np.arange(16).reshape(4, 4)
        # 对 a 进行深拷贝，生成 ac 数组
        ac = a.copy()
        # 使用 a[:-1].clip 方法，将介于 4 和 10 之间的值裁剪到 a[1:] 中
        a[:-1].clip(4, 10, out=a[1:])
        # 调用 self.clip 方法，裁剪介于 4 和 10 之间的 ac[:-1]，生成预期结果 expected
        expected = self.clip(ac[:-1], 4, 10)
        # 使用 assert_array_equal 函数断言 a[1:] 和 expected 数组相等
        assert_array_equal(a[1:], expected)

    def test_clip_inplace_array(self):
        # 测试当 in-place 裁剪数组时，使用 native double 输入和数组最小/最大值
        a = self._generate_data(self.nr, self.nc)
        # 对 a 进行深拷贝，生成 ac 数组
        ac = a.copy()
        # 创建一个与 a 相同形状的全零数组 m
        m = np.zeros(a.shape)
        # 设置 M 的值为 1.0
        M = 1.0
        # 调用 self.fastclip 方法，将结果输出到 a 中
        self.fastclip(a, m, M, a)
        # 调用 self.clip 方法，将结果输出到 ac 中
        self.clip(a, m, M, ac)
        # 使用 assert_array_strict_equal 函数断言 a 和 ac 数组严格相等
        assert_array_strict_equal(a, ac)

    def test_clip_inplace_simple(self):
        # 测试当 in-place 裁剪简单数组时，使用 native double 输入和标量最小/最大值
        a = self._generate_data(self.nr, self.nc)
        # 对 a 进行深拷贝，生成 ac 数组
        ac = a.copy()
        # 设置 m 和 M 的值分别为 -0.5 和 0.6
        m = -0.5
        M = 0.6
        # 调用 self.fastclip 方法，将结果输出到 a 中
        self.fastclip(a, m, M, a)
        # 调用 self.clip 方法，将结果输出到 ac 中
        self.clip(a, m, M, ac)
        # 使用 assert_array_strict_equal 函数断言 a 和 ac 数组严格相等
        assert_array_strict_equal(a, ac)

    def test_clip_func_takes_out(self):
        # 确保 clip() 函数接受 out 参数
        a = self._generate_data(self.nr, self.nc)
        # 对 a 进行深拷贝，生成 ac 数组
        ac = a.copy()
        # 设置 m 和 M 的值分别为 -0.5 和 0.6
        m = -0.5
        M = 0.6
        # 调用 np.clip 方法，将结果输出到 a，并生成 a2 数组
        a2 = np.clip(a, m, M, out=a)
        # 调用 self.clip 方法，将结果输出到 ac 中
        self.clip(a, m, M, ac)
        # 使用 assert_array_strict_equal 函数断言 a2 和 ac 数组严格相等
        assert_array_strict_equal(a2, ac)
        # 使用 assert_ 函数断言 a2 和 a 引用相同对象
        assert_(a2 is a)

    def test_clip_nan(self):
        # 测试当输入包含 NaN 时的裁剪行为
        d = np.arange(7.)
        # 使用 np.nan 作为最小值裁剪 d，结果应为 np.nan
        assert_equal(d.clip(min=np.nan), np.nan)
        # 使用 np.nan 作为最大值裁剪 d，结果应为 np.nan
        assert_equal(d.clip(max=np.nan), np.nan)
        # 使用 np.nan 作为最小和最大值裁剪 d，结果应为 np.nan
        assert_equal(d.clip(min=np.nan, max=np.nan), np.nan)
        # 使用 -2 作为最小值，np.nan 作为最大值裁剪 d，结果应为 np.nan
        assert_equal(d.clip(min=-2, max=np.nan), np.nan)
        # 使用 np.nan 作为最小值，10 作为最大值裁剪 d，结果应为 np.nan
        assert_equal(d.clip(min=np.nan, max=10), np.nan)

    def test_object_clip(self):
        # 测试对象数组的裁剪行为
        a = np.arange(10, dtype=object)
        # 对对象数组 a 进行裁剪，使其每个元素介于 1 和 5 之间
        actual = np.clip(a, 1, 5)
        # 生成预期结果数组 expected
        expected = np.array([1, 1, 2, 3, 4, 5, 5, 5, 5, 5])
        # 使用 assert 函数断言 actual 和 expected 数组相等
        assert actual.tolist() == expected.tolist()
    # 定义一个测试函数，测试当使用 None 作为最小值和最大值时是否会引发 ValueError 异常
    def test_clip_all_none(self):
        # 创建一个包含 0 到 9 的对象数组
        a = np.arange(10, dtype=object)
        # 使用断言检查是否会引发 ValueError 异常，异常消息中应包含 'max or min'
        with assert_raises_regex(ValueError, 'max or min'):
            np.clip(a, None, None)

    # 定义一个测试函数，测试当使用无效的转换方式时是否会引发 ValueError 异常
    def test_clip_invalid_casting(self):
        # 创建一个包含 0 到 9 的对象数组
        a = np.arange(10, dtype=object)
        # 使用断言检查是否会引发 ValueError 异常，异常消息中应包含 'casting must be one of'
        with assert_raises_regex(ValueError,
                                 'casting must be one of'):
            # 调用自定义的 fastclip 函数，尝试使用无效的 casting 参数 'garbage'
            self.fastclip(a, 1, 8, casting="garbage")

    # 使用 pytest 的参数化装饰器，对不同的 amin 和 amax 进行多组测试
    @pytest.mark.parametrize("amin, amax", [
        # 两个标量
        (1, 0),
        # 混合标量和数组
        (1, np.zeros(10)),
        # 两个数组
        (np.ones(10), np.zeros(10)),
        ])
    # 定义一个测试函数，测试 np.clip 对最小值和最大值的翻转情况
    def test_clip_value_min_max_flip(self, amin, amax):
        # 创建一个包含 0 到 9 的 int64 类型的数组
        a = np.arange(10, dtype=np.int64)
        # 根据 ufunc_docstrings.py 的要求，计算预期结果
        expected = np.minimum(np.maximum(a, amin), amax)
        # 调用 np.clip 函数，计算实际结果
        actual = np.clip(a, amin, amax)
        # 使用断言检查实际结果与预期结果是否相等
        assert_equal(actual, expected)

    # 使用 pytest 的参数化装饰器，对不同的 arr, amin, amax 和 exp 进行多组测试
    @pytest.mark.parametrize("arr, amin, amax, exp", [
        # 针对 npy_ObjectClip 中的一个 bug，基于 hypothesis 生成的一个案例
        (np.zeros(10, dtype=object),
         0,
         -2**64+1,
         np.full(10, -2**64+1, dtype=object)),
        # 针对 NPY_TIMEDELTA_MAX 中的一个 bug，基于 hypothesis 生成的一个案例
        (np.zeros(10, dtype='m8') - 1,
         0,
         0,
         np.zeros(10, dtype='m8')),
    ])
    # 定义一个测试函数，测试 np.clip 在特定问题案例下的行为
    def test_clip_problem_cases(self, arr, amin, amax, exp):
        # 调用 np.clip 函数，计算实际结果
        actual = np.clip(arr, amin, amax)
        # 使用断言检查实际结果与预期结果是否相等
        assert_equal(actual, exp)

    # 使用 pytest 的参数化装饰器，对不同的 arr, amin 和 amax 进行多组测试
    @pytest.mark.parametrize("arr, amin, amax", [
        # 针对 hypothesis 中的一个有问题的标量 nan 案例
        (np.zeros(10, dtype=np.int64),
         np.array(np.nan),
         np.zeros(10, dtype=np.int32)),
    ])
    # 定义一个测试函数，测试 np.clip 在处理标量 nan 时的传播行为
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        # 根据比较要求，计算预期结果
        expected = np.minimum(np.maximum(arr, amin), amax)
        # 调用 np.clip 函数，计算实际结果
        actual = np.clip(arr, amin, amax)
        # 使用断言检查实际结果与预期结果是否相等
        assert_equal(actual, expected)

    # 使用 pytest 的标记 xfail，注明这个测试用例预期会失败，原因是传播行为不符合规范
    @pytest.mark.xfail(reason="propagation doesn't match spec")
    # 使用 pytest 的参数化装饰器，对不同的 arr, amin 和 amax 进行多组测试
    @pytest.mark.parametrize("arr, amin, amax", [
        # 针对 np.timedelta64('NaT') 的传播问题
        (np.array([1] * 10, dtype='m8'),
         np.timedelta64('NaT'),
         np.zeros(10, dtype=np.int32)),
    ])
    # 定义一个测试函数，测试 np.clip 在处理 NaT 传播时的行为
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_NaT_propagation(self, arr, amin, amax):
        # 根据比较要求，计算预期结果
        expected = np.minimum(np.maximum(arr, amin), amax)
        # 调用 np.clip 函数，计算实际结果
        actual = np.clip(arr, amin, amax)
        # 使用断言检查实际结果与预期结果是否相等
        assert_equal(actual, expected)

    # 使用 hypothesis 的 given 装饰器，定义一个数据生成函数
    @given(
        data=st.data(),
        arr=hynp.arrays(
            dtype=hynp.integer_dtypes() | hynp.floating_dtypes(),
            shape=hynp.array_shapes()
        )
    )
    def test_clip_property(self, data, arr):
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        
        # 定义包含整数和浮点数的数据类型集合
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        
        # 使用 `hynp.mutually_broadcastable_shapes` 生成与给定数组形状兼容的两个输入形状及结果形状
        in_shapes, result_shape = data.draw(
            hynp.mutually_broadcastable_shapes(
                num_shapes=2, base_shape=arr.shape
            )
        )
        
        # 从定义的数据类型集合中选择一个不包含 NaN 的标量作为值
        s = numeric_dtypes.flatmap(
            lambda x: hynp.from_dtype(x, allow_nan=False))
        
        # 从数据生成器中绘制一个在给定形状内的数组，要求不包含 NaN 值
        amin = data.draw(s | hynp.arrays(dtype=numeric_dtypes,
            shape=in_shapes[0], elements={"allow_nan": False}))
        
        # 从数据生成器中绘制另一个在给定形状内的数组，要求不包含 NaN 值
        amax = data.draw(s | hynp.arrays(dtype=numeric_dtypes,
            shape=in_shapes[1], elements={"allow_nan": False}))

        # 计算使用 np.clip 函数得到的结果和期望的结果，并进行比较
        result = np.clip(arr, amin, amax)
        # 确定结果的数据类型
        t = np.result_type(arr, amin, amax)
        # 计算期望的结果，即对 arr 在 amin 和 amax 之间进行裁剪的结果
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        # 断言结果的数据类型和期望一致
        assert result.dtype == t
        # 断言结果和期望结果数组内容一致
        assert_array_equal(result, expected)
class TestAllclose:
    # 定义公共相对误差和绝对误差阈值
    rtol = 1e-5
    atol = 1e-8

    # 初始化方法，设置无效数错误忽略
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    # 清理方法，恢复之前的错误设置
    def teardown_method(self):
        np.seterr(**self.olderr)

    # 检查两个数组是否全部接近的测试方法
    def tst_allclose(self, x, y):
        assert_(np.allclose(x, y), "%s and %s not close" % (x, y))

    # 检查两个数组是否不全部接近的测试方法
    def tst_not_allclose(self, x, y):
        assert_(not np.allclose(x, y), "%s and %s shouldn't be close" % (x, y))

    # 参数化测试工厂方法，测试所有应接近的情况
    def test_ip_allclose(self):
        # 设置测试数据集
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [([1, 0], [1, 0]),                      # 相等的小数组
                ([atol], [0]),                         # 绝对误差阈值与零比较
                ([1], [1+rtol+atol]),                  # 相对和绝对误差都超过的情况
                (arr, arr + arr*rtol),                 # 相等的大数组
                (arr, arr + arr*rtol + atol*2),        # 相对和绝对误差都超过的大数组
                (aran, aran + aran*rtol),              # 大多维数组
                (np.inf, np.inf),                      # 无穷大与自身比较
                (np.inf, [np.inf])]                    # 无穷大与数组比较

        # 遍历测试数据，执行接近性测试
        for (x, y) in data:
            self.tst_allclose(x, y)

    # 参数化测试工厂方法，测试所有不应接近的情况
    def test_ip_not_allclose(self):
        # 设置测试数据集
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [([np.inf, 0], [1, np.inf]),             # 无穷大和零不应接近
                ([np.inf, 0], [1, 0]),                 # 无穷大和有限数不应接近
                ([np.inf, np.inf], [1, np.inf]),       # 两个无穷大不应接近
                ([np.inf, np.inf], [1, 0]),            # 一个无穷大和一个有限数不应接近
                ([-np.inf, 0], [np.inf, 0]),           # 无穷小和有限数不应接近
                ([np.nan, 0], [np.nan, 0]),            # NaN 不应接近任何数
                ([atol*2], [0]),                       # 绝对误差超过阈值不应接近零
                ([1], [1+rtol+atol*2]),                # 相对和绝对误差都超过的小数组
                (aran, aran + aran*atol + atol*2),     # 多维数组相对和绝对误差都超过的情况
                (np.array([np.inf, 1]), np.array([0, np.inf]))]  # 数组中有无穷大不应接近

        # 遍历测试数据，执行不接近性测试
        for (x, y) in data:
            self.tst_not_allclose(x, y)

    # 测试是否修改了参数的方法
    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.allclose(x, y)
        # 检查数组 x 和 y 是否未被修改
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    # 测试最小整数是否接近自身的方法
    def test_min_int(self):
        # 可能因为 abs(min_int) == min_int 而产生问题
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)
        assert_(np.allclose(a, a))

    # 测试包含 NaN 值的数组是否被认为接近自身的方法
    def test_equalnan(self):
        x = np.array([1.0, np.nan])
        assert_(np.allclose(x, x, equal_nan=True))

    # 测试 allclose 是否不保留子类型的方法
    def test_return_class_is_ndarray(self):
        # Issue gh-6475
        # 检查 allclose 是否不保留子类型
        class Foo(np.ndarray):
            def __new__(cls, *args, **kwargs):
                return np.array(*args, **kwargs).view(cls)

        a = Foo([1])
        assert_(type(np.allclose(a, a)) is bool)


class TestIsclose:
    # 定义公共相对误差和绝对误差阈值
    rtol = 1e-5
    atol = 1e-8
    # 在对象初始化时设置初始值
    def _setup(self):
        # 将对象的公差 atol 存入局部变量
        atol = self.atol
        # 将对象的相对公差 rtol 存入局部变量
        rtol = self.rtol
        # 创建包含100和1000的NumPy数组
        arr = np.array([100, 1000])
        # 创建一个形状为(5, 5, 5)的NumPy数组
        aran = np.arange(125).reshape((5, 5, 5))

        # 设置所有关闭测试用例的列表
        self.all_close_tests = [
                # 第一个测试用例: 比较 [1, 0] 和 [1, 0]
                ([1, 0], [1, 0]),
                # 第二个测试用例: 比较 [atol] 和 [0]
                ([atol], [0]),
                # 第三个测试用例: 比较 [1] 和 [1 + rtol + atol]
                ([1], [1 + rtol + atol]),
                # 第四个测试用例: 比较 arr 和 arr + arr*rtol
                (arr, arr + arr*rtol),
                # 第五个测试用例: 比较 arr 和 arr + arr*rtol + atol
                (arr, arr + arr*rtol + atol),
                # 第六个测试用例: 比较 aran 和 aran + aran*rtol
                (aran, aran + aran*rtol),
                # 第七个测试用例: 比较 np.inf 和 np.inf
                (np.inf, np.inf),
                # 第八个测试用例: 比较 np.inf 和 [np.inf]
                (np.inf, [np.inf]),
                # 第九个测试用例: 比较 [np.inf, -np.inf] 和 [np.inf, -np.inf]
                ([np.inf, -np.inf], [np.inf, -np.inf]),
                ]
        
        # 设置无关闭测试用例的列表
        self.none_close_tests = [
                # 第一个测试用例: 比较 [np.inf, 0] 和 [1, np.inf]
                ([np.inf, 0], [1, np.inf]),
                # 第二个测试用例: 比较 [np.inf, -np.inf] 和 [1, 0]
                ([np.inf, -np.inf], [1, 0]),
                # 第三个测试用例: 比较 [np.inf, np.inf] 和 [1, -np.inf]
                ([np.inf, np.inf], [1, -np.inf]),
                # 第四个测试用例: 比较 [np.inf, np.inf] 和 [1, 0]
                ([np.inf, np.inf], [1, 0]),
                # 第五个测试用例: 比较 [np.nan, 0] 和 [np.nan, -np.inf]
                ([np.nan, 0], [np.nan, -np.inf]),
                # 第六个测试用例: 比较 [atol*2] 和 [0]
                ([atol*2], [0]),
                # 第七个测试用例: 比较 [1] 和 [1 + rtol + atol*2]
                ([1], [1 + rtol + atol*2]),
                # 第八个测试用例: 比较 aran 和 aran + rtol*1.1*aran + atol*1.1
                (aran, aran + rtol*1.1*aran + atol*1.1),
                # 第九个测试用例: 比较 np.array([np.inf, 1]) 和 np.array([0, np.inf])
                (np.array([np.inf, 1]), np.array([0, np.inf])),
                ]
        
        # 设置部分关闭测试用例的列表
        self.some_close_tests = [
                # 第一个测试用例: 比较 [np.inf, 0] 和 [np.inf, atol*2]
                ([np.inf, 0], [np.inf, atol*2]),
                # 第二个测试用例: 比较 [atol, 1, 1e6*(1 + 2*rtol) + atol] 和 [0, np.nan, 1e6]
                ([atol, 1, 1e6*(1 + 2*rtol) + atol], [0, np.nan, 1e6]),
                # 第三个测试用例: 比较 np.arange(3) 和 [0, 1, 2.1]
                (np.arange(3), [0, 1, 2.1]),
                # 第四个测试用例: 比较 np.nan 和 [np.nan, np.nan, np.nan]
                (np.nan, [np.nan, np.nan, np.nan]),
                # 第五个测试用例: 比较 [0] 和 [atol, np.inf, -np.inf, np.nan]
                ([0], [atol, np.inf, -np.inf, np.nan]),
                # 第六个测试用例: 比较 0 和 [atol, np.inf, -np.inf, np.nan]
                (0, [atol, np.inf, -np.inf, np.nan]),
                ]
        
        # 设置部分关闭测试结果的列表
        self.some_close_results = [
                # 第一个测试结果: [True, False]
                [True, False],
                # 第二个测试结果: [True, False, False]
                [True, False, False],
                # 第三个测试结果: [True, True, False]
                [True, True, False],
                # 第四个测试结果: [False, False, False]
                [False, False, False],
                # 第五个测试结果: [True, False, False, False]
                [True, False, False, False],
                # 第六个测试结果: [True, False, False, False]
                [True, False, False, False],
                ]

    # 测试 np.isclose 方法的函数
    def test_ip_isclose(self):
        # 调用对象的 _setup 方法进行初始化设置
        self._setup()
        # 将部分关闭测试用例赋值给变量 tests
        tests = self.some_close_tests
        # 将部分关闭测试结果赋值给变量 results
        results = self.some_close_results
        # 对于 tests 和 results 中的每组测试用例和结果
        for (x, y), result in zip(tests, results):
            # 断言 np.isclose(x, y) 的结果与预期结果 result 相等
            assert_array_equal(np.isclose(x, y), result)

        # 设置 x 和 y 的值为特定的 NumPy 数组
        x = np.array([2.1, 2.1, 2.1, 2.1, 5, np.nan])
        y = np.array([2, 2, 2, 2, np.nan, 5])
        # 设置特定的公差 atol 和相对公差 rtol
        atol = [0.11, 0.09, 1e-8, 1e-8, 1, 1]
        rtol = [1e-8, 1e-8, 0.06, 0.04, 1, 1]
        # 设置期望的结果 expected
        expected = np.array([True, False, True, False, False, False])
        # 断言 np.isclose(x, y, rtol=rtol, atol=atol) 的结果与期望的结果 expected 相等
        assert_array_equal(np.isclose(x, y, rtol=rtol, atol=atol), expected)

        # 设置错误消息
        message = "operands could not be broadcast together..."
        # 设置特定的 atol 数组
        atol = np.array([1e-8, 1e-8])
        # 使用 assert_raises 检查调用 np.isclose(x, y, atol=atol) 是否会引发 ValueError 错误，并验证错误消息
        with assert_raises(ValueError, msg=message):
            np.isclose(x, y, atol=atol)

        # 设置特定的 rtol 数组
        rtol = np.array([1e-5, 1e-5])
        # 使用 assert_raises 检查调用 np.isclose(x, y, rtol=rtol) 是否会引发 ValueError 错误，并验证错误消息
        with assert_raises(ValueError, msg=message):
            np.isclose(x, y, rtol=rtol)
    # 测试 NEP 50 中的 np.isclose 函数在浮点数比较中的行为

    def test_nep50_isclose(self):
        # 使用 np.finfo('f8').eps 计算出小于 1 的最大浮点数
        below_one = float(1. - np.finfo('f8').eps)
        # 创建一个 float32 类型的 numpy 数组，值为接近 1 但在 float32 精度下是精确的
        f32 = np.array(below_one, 'f4')  # This is just 1 at float32 precision
        # 断言 f32 大于比它稍微小一点的 float32 数组
        assert f32 > np.array(below_one)
        # 断言 NEP 50 中的广播特性，比较 f32 和 below_one 是否相等
        assert f32 == below_one
        # 测试 np.isclose 函数对比参数的行为，要求它们非常接近（atol 和 rtol 为 0）
        assert np.isclose(f32, below_one, atol=0, rtol=0)
        # 测试 np.isclose 函数对比参数的行为，要求它们非常接近（atol 为 below_one）
        assert np.isclose(f32, np.float32(0), atol=below_one)
        # 测试 np.isclose 函数对比参数的行为，atol 为 0，rtol 为 below_one 的一半
        assert np.isclose(f32, 2, atol=0, rtol=below_one / 2)
        # 断言 np.isclose 不能成功比较 f32 和 np.float64(below_one)
        assert not np.isclose(f32, np.float64(below_one), atol=0, rtol=0)
        # 断言 np.isclose 不能成功比较 f32 和 np.float32(0)，因为 atol 是 np.float64(below_one)
        assert not np.isclose(f32, np.float32(0), atol=np.float64(below_one))
        # 断言 np.isclose 不能成功比较 f32 和 2，因为 rtol 是 np.float64(below_one/2)
        assert not np.isclose(f32, 2, atol=0, rtol=np.float64(below_one / 2))

    # 断言 np.all(np.isclose(x, y)) 成立，否则输出指定信息
    def tst_all_isclose(self, x, y):
        assert_(np.all(np.isclose(x, y)), "%s and %s not close" % (x, y))

    # 断言 np.any(np.isclose(x, y)) 不成立，否则输出指定信息
    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    # 比较 np.isclose(x, y) 和 np.allclose(x, y) 的结果
    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            # 如果 x 和 y 都是标量，则比较 np.isclose 和 np.allclose 的结果是否相等
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            # 如果 x 和 y 是数组，则比较 np.isclose 和 np.allclose 的结果是否相等
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))

    # 测试所有在 self.all_close_tests 中的测试用例
    def test_ip_all_isclose(self):
        self._setup()
        for (x, y) in self.all_close_tests:
            self.tst_all_isclose(x, y)

        # 创建两个数组 x 和 y，指定它们的绝对误差 (atol) 和相对误差 (rtol)，同时允许 NaN 相等
        x = np.array([2.3, 3.6, 4.4, np.nan])
        y = np.array([2, 3, 4, np.nan])
        atol = [0.31, 0, 0, 1]
        rtol = [0, 0.21, 0.11, 1]
        # 断言 np.allclose 在指定的误差范围内比较 x 和 y，允许 NaN 相等
        assert np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        # 断言 np.allclose 在指定的误差范围内比较 x 和 y，不允许 NaN 相等
        assert not np.allclose(x, y, atol=0.1, rtol=0.1, equal_nan=True)

        # 展示 gh-14330 已经解决的情况，比较包含 NaN 的数组
        assert np.allclose([1, 2, float('nan')], [1, 2, float('nan')],
                           atol=[1, 1, 1], equal_nan=True)

    # 测试所有在 self.none_close_tests 中的测试用例
    def test_ip_none_isclose(self):
        self._setup()
        for (x, y) in self.none_close_tests:
            self.tst_none_isclose(x, y)

    # 测试所有在 self.all_close_tests、self.none_close_tests 和 self.some_close_tests 中的测试用例
    def test_ip_isclose_allclose(self):
        self._setup()
        tests = (self.all_close_tests + self.none_close_tests +
                 self.some_close_tests)
        for (x, y) in tests:
            self.tst_isclose_allclose(x, y)

    # 断言 np.isclose(np.nan, np.nan, equal_nan=True) 等于 [True]
    def test_equal_nan(self):
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        # 创建包含 NaN 的数组 arr
        arr = np.array([1.0, np.nan])
        # 断言 np.isclose 对比数组 arr 自身，结果为 [True, True]
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])
    def test_masked_arrays(self):
        # 确保在参数互换时测试输出类型。

        # 创建一个被掩码处理的数组，掩码的条件是 [True, True, False]，数组元素是 [0, 1, 2]
        x = np.ma.masked_where([True, True, False], np.arange(3))
        # 断言掩码数组的类型与 np.isclose(2, x) 相同
        assert_(type(x) is type(np.isclose(2, x)))
        # 断言掩码数组的类型与 np.isclose(x, 2) 相同
        assert_(type(x) is type(np.isclose(x, 2)))

        # 创建一个被掩码处理的数组，掩码的条件是 [True, True, False]，数组元素是 [nan, inf, nan]
        x = np.ma.masked_where([True, True, False], [np.nan, np.inf, np.nan])
        # 断言掩码数组的类型与 np.isclose(np.inf, x) 相同
        assert_(type(x) is type(np.isclose(np.inf, x)))
        # 断言掩码数组的类型与 np.isclose(x, np.inf) 相同
        assert_(type(x) is type(np.isclose(x, np.inf)))

        # 创建一个被掩码处理的数组，掩码的条件是 [True, True, False]，数组元素是 [nan, nan, nan]
        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        # 使用 equal_nan=True 来比较，创建一个用于掩码的数组 y
        y = np.isclose(np.nan, x, equal_nan=True)
        # 断言 x 和 y 的类型相同
        assert_(type(x) is type(y))
        # 确保掩码没有被修改
        assert_array_equal([True, True, False], y.mask)
        # 使用 equal_nan=True 来比较，创建一个用于掩码的数组 y
        y = np.isclose(x, np.nan, equal_nan=True)
        # 断言 x 和 y 的类型相同
        assert_(type(x) is type(y))
        # 确保掩码没有被修改
        assert_array_equal([True, True, False], y.mask)

        # 创建一个被掩码处理的数组，掩码的条件是 [True, True, False]，数组元素是 [nan, nan, nan]
        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        # 使用 equal_nan=True 来比较 x 和 x，创建一个用于掩码的数组 y
        y = np.isclose(x, x, equal_nan=True)
        # 断言 x 和 y 的类型相同
        assert_(type(x) is type(y))
        # 确保掩码没有被修改
        assert_array_equal([True, True, False], y.mask)

    def test_scalar_return(self):
        # 断言 np.isclose(1, 1) 返回的是一个标量
        assert_(np.isscalar(np.isclose(1, 1)))

    def test_no_parameter_modification(self):
        # 创建两个数组 x 和 y，包含无穷大值
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        # 比较 x 和 y 的近似程度，但不修改参数
        np.isclose(x, y)
        # 断言 x 没有被修改
        assert_array_equal(x, np.array([np.inf, 1]))
        # 断言 y 没有被修改
        assert_array_equal(y, np.array([0, np.inf]))

    def test_non_finite_scalar(self):
        # GH7014，当比较两个标量时，输出也应为标量
        # 断言 np.isclose(np.inf, -np.inf) 的结果是 np.False_
        assert_(np.isclose(np.inf, -np.inf) is np.False_)
        # 断言 np.isclose(0, np.inf) 的结果是 np.False_
        assert_(np.isclose(0, np.inf) is np.False_)
        # 断言 np.isclose(0, np.inf) 返回的类型是 np.bool
        assert_(type(np.isclose(0, np.inf)) is np.bool)

    def test_timedelta(self):
        # 只要 `atol` 是整数或 timedelta64，allclose 目前也适用于 timedelta64
        # 创建一个 timedelta64 类型的数组 a，包含一个 "NaT" 字符串
        a = np.array([[1, 2, 3, "NaT"]], dtype="m8[ns]")
        # 确保使用 atol=0 和 equal_nan=True 的条件下，所有元素都近似相等
        assert np.isclose(a, a, atol=0, equal_nan=True).all()
        # 确保使用 atol=np.timedelta64(1, "ns") 和 equal_nan=True 的条件下，所有元素都近似相等
        assert np.isclose(a, a, atol=np.timedelta64(1, "ns"), equal_nan=True).all()
        # 使用 allclose 确保所有元素都近似相等，同时使用 atol=0 和 equal_nan=True
        assert np.allclose(a, a, atol=0, equal_nan=True)
        # 使用 allclose 确保所有元素都近似相等，同时使用 atol=np.timedelta64(1, "ns") 和 equal_nan=True
        assert np.allclose(a, a, atol=np.timedelta64(1, "ns"), equal_nan=True)
# 定义一个测试类 TestStdVar，用于测试 np.var 和 np.std 函数的行为
class TestStdVar:
    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 创建一个包含整数的 NumPy 数组 A
        self.A = np.array([1, -1, 1, -1])
        # 设置真实方差的期望值为 1
        self.real_var = 1

    # 测试基本的 np.var 和 np.std 函数
    def test_basic(self):
        # 断言 np.var(A) 几乎等于 self.real_var
        assert_almost_equal(np.var(self.A), self.real_var)
        # 断言 np.std(A) 的平方几乎等于 self.real_var
        assert_almost_equal(np.std(self.A)**2, self.real_var)

    # 测试标量输入的 np.var 和 np.std 函数
    def test_scalars(self):
        # 断言 np.var(1) 应该等于 0
        assert_equal(np.var(1), 0)
        # 断言 np.std(1) 应该等于 0
        assert_equal(np.std(1), 0)

    # 测试 ddof 参数为 1 的 np.var 和 np.std 函数
    def test_ddof1(self):
        # 使用 ddof=1 参数计算 np.var(A)，与预期值进行比较
        assert_almost_equal(np.var(self.A, ddof=1),
                            self.real_var * len(self.A) / (len(self.A) - 1))
        # 使用 ddof=1 参数计算 np.std(A)，与预期值进行比较
        assert_almost_equal(np.std(self.A, ddof=1)**2,
                            self.real_var * len(self.A) / (len(self.A) - 1))

    # 测试 ddof 参数为 2 的 np.var 和 np.std 函数
    def test_ddof2(self):
        # 使用 ddof=2 参数计算 np.var(A)，与预期值进行比较
        assert_almost_equal(np.var(self.A, ddof=2),
                            self.real_var * len(self.A) / (len(self.A) - 2))
        # 使用 ddof=2 参数计算 np.std(A)，与预期值进行比较
        assert_almost_equal(np.std(self.A, ddof=2)**2,
                            self.real_var * len(self.A) / (len(self.A) - 2))

    # 测试 correction 参数的 np.var 和 np.std 函数
    def test_correction(self):
        # 断言 np.var(A, correction=1) 等价于 np.var(A, ddof=1)
        assert_almost_equal(
            np.var(self.A, correction=1), np.var(self.A, ddof=1)
        )
        # 断言 np.std(A, correction=1) 等价于 np.std(A, ddof=1)
        assert_almost_equal(
            np.std(self.A, correction=1), np.std(self.A, ddof=1)
        )

        # 当同时提供 ddof 和 correction 参数时，验证抛出 ValueError 异常
        err_msg = "ddof and correction can't be provided simultaneously."
        with assert_raises_regex(ValueError, err_msg):
            np.var(self.A, ddof=1, correction=0)

        with assert_raises_regex(ValueError, err_msg):
            np.std(self.A, ddof=1, correction=1)

    # 测试将输出保存在标量数组中的 np.var 和 np.std 函数
    def test_out_scalar(self):
        # 创建一个包含整数的 NumPy 数组 d
        d = np.arange(10)
        # 创建一个标量的 NumPy 数组 out
        out = np.array(0.)
        # 使用 np.std(d, out=out) 函数，验证返回的 r 是否等于 out
        r = np.std(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        # 使用 np.var(d, out=out) 函数，验证返回的 r 是否等于 out
        r = np.var(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        # 使用 np.mean(d, out=out) 函数，验证返回的 r 是否等于 out
        r = np.mean(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)


# 定义一个测试类 TestStdVarComplex，用于测试复数输入的 np.var 和 np.std 函数
class TestStdVarComplex:
    # 测试基本的 np.var 和 np.std 函数，包含复数输入
    def test_basic(self):
        # 创建一个包含复数的 NumPy 数组 A
        A = np.array([1, 1.j, -1, -1.j])
        # 设置真实方差的期望值为 1
        real_var = 1
        # 断言 np.var(A) 几乎等于 real_var
        assert_almost_equal(np.var(A), real_var)
        # 断言 np.std(A) 的平方几乎等于 real_var
        assert_almost_equal(np.std(A)**2, real_var)

    # 测试标量复数输入的 np.var 和 np.std 函数
    def test_scalars(self):
        # 断言 np.var(1j) 应该等于 0
        assert_equal(np.var(1j), 0)
        # 断言 np.std(1j) 应该等于 0
        assert_equal(np.std(1j), 0)


# 定义一个测试类 TestCreationFuncs，用于测试 np.ones, np.zeros, np.empty 和 np.full 函数的行为
class TestCreationFuncs:
    # 在每个测试方法运行之前设置测试环境
    def setup_method(self):
        # 获取所有 NumPy 数组的数据类型，包括 void, bytes, str 类型
        dtypes = {np.dtype(tp) for tp in itertools.chain(*sctypes.values())}
        # 获取可变大小的数据类型，如 void, bytes, str 类型
        variable_sized = {tp for tp in dtypes if tp.str.endswith('0')}
        # 定义一个用于排序的 key 函数
        keyfunc = lambda dtype: dtype.str
        # 获取不包含可变大小数据类型的所有数据类型，并按照 keyfunc 进行排序
        self.dtypes = sorted(dtypes - variable_sized |
                             {np.dtype(tp.str.replace("0", str(i)))
                              for tp in variable_sized for i in range(1, 10)},
                             key=keyfunc)
        # 将包含 void, bytes, str 类型的数据类型转换为 type 类型，并追加到 dtypes 中
        self.dtypes += [type(dt) for dt in sorted(dtypes, key=keyfunc)]
        # 定义一个包含 'C' 和 'F' 键值对的顺序字典 orders，用于描述内存布局
        self.orders = {'C': 'c_contiguous', 'F': 'f_contiguous'}
        # 定义一个整数 ndims，表示数组的维度为 10
        self.ndims = 10
    # 定义一个用于检查数组创建函数的方法，接受一个函数和可选的填充值作为参数
    def check_function(self, func, fill_value=None):
        # 定义参数的组合，包括大小、维度、顺序和数据类型
        par = ((0, 1, 2),
               range(self.ndims),
               self.orders,
               self.dtypes)
        
        fill_kwarg = {}
        # 如果填充值不为None，则创建填充关键字参数的字典
        if fill_value is not None:
            fill_kwarg = {'fill_value': fill_value}

        # 遍历所有参数的组合
        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]

            # 检查是否是空类型（VoidDType），或者是结构化类型（以'|V'开头的dtype字符串）
            is_void = dtype is np.dtypes.VoidDType or (
                isinstance(dtype, np.dtype) and dtype.str.startswith('|V'))

            # 如果有填充值且当前dtype是空类型，则跳过当前循环
            if fill_kwarg and is_void:
                continue

            # 调用传入的函数创建数组
            arr = func(shape, order=order, dtype=dtype,
                       **fill_kwarg)

            # 断言数组的dtype与期望的dtype相符
            if isinstance(dtype, np.dtype):
                assert_equal(arr.dtype, dtype)
            elif isinstance(dtype, type(np.dtype)):
                if dtype in (np.dtypes.StrDType, np.dtypes.BytesDType):
                    dtype_str = np.dtype(dtype.type).str.replace('0', '1')
                    assert_equal(arr.dtype, np.dtype(dtype_str))
                else:
                    assert_equal(arr.dtype, np.dtype(dtype.type))
            
            # 断言数组的标志符合预期的顺序
            assert_(getattr(arr.flags, self.orders[order]))

            # 如果有填充值，则断言数组中的所有元素都与填充值相等
            if fill_value is not None:
                if arr.dtype.str.startswith('|S'):
                    val = str(fill_value)
                else:
                    val = fill_value
                assert_equal(arr, dtype.type(val))

    # 测试函数：测试np.zeros函数
    def test_zeros(self):
        self.check_function(np.zeros)

    # 测试函数：测试np.ones函数
    def test_ones(self):
        self.check_function(np.ones)

    # 测试函数：测试np.empty函数
    def test_empty(self):
        self.check_function(np.empty)

    # 测试函数：测试np.full函数，使用填充值0和1分别进行测试
    def test_full(self):
        self.check_function(np.full, 0)
        self.check_function(np.full, 1)

    # 标记为跳过的测试函数，条件是不具备引用计数（refcount）功能的Python环境
    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_for_reference_leak(self):
        # 确保我们有一个引用对象
        dim = 1
        beg = sys.getrefcount(dim)
        
        # 创建多维数组并检查引用计数不变
        np.zeros([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.ones([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.empty([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.full([dim]*10, 0)
        assert_(sys.getrefcount(dim) == beg)
    # 定义一个测试类，用于测试 `ones_like`, `zeros_like`, `empty_like` 和 `full_like` 函数
    class TestLikeFuncs:

        # 在每个测试方法运行之前设置测试数据
        def setup_method(self):
            self.data = [
                    # 数组标量
                    (np.array(3.), None),
                    (np.array(3), 'f8'),
                    # 1维数组
                    (np.arange(6, dtype='f4'), None),
                    (np.arange(6), 'c16'),
                    # 2维 C 排列数组
                    (np.arange(6).reshape(2, 3), None),
                    (np.arange(6).reshape(3, 2), 'i1'),
                    # 2维 F 排列数组
                    (np.arange(6).reshape((2, 3), order='F'), None),
                    (np.arange(6).reshape((3, 2), order='F'), 'i1'),
                    # 3维 C 排列数组
                    (np.arange(24).reshape(2, 3, 4), None),
                    (np.arange(24).reshape(4, 3, 2), 'f4'),
                    # 3维 F 排列数组
                    (np.arange(24).reshape((2, 3, 4), order='F'), None),
                    (np.arange(24).reshape((4, 3, 2), order='F'), 'f4'),
                    # 3维非 C/F 排列数组
                    (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None),
                    (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), '?'),
                         ]
            self.shapes = [(), (5,), (5, 6,), (5, 6, 7,)]

        # 比较数组值的方法，dz 是待检查数组，value 是预期值，fill_value 是填充标志
        def compare_array_value(self, dz, value, fill_value):
            if value is not None:
                if fill_value:
                    # 转换接近 np.full_like 的行为，但将来可能直接转换而不会出错（当前方法不会）
                    z = np.array(value).astype(dz.dtype)
                    assert_(np.all(dz == z))
                else:
                    assert_(np.all(dz == value))

        # 测试 np.ones_like 函数
        def test_ones_like(self):
            self.check_like_function(np.ones_like, 1)

        # 测试 np.zeros_like 函数
        def test_zeros_like(self):
            self.check_like_function(np.zeros_like, 0)

        # 测试 np.empty_like 函数
        def test_empty_like(self):
            self.check_like_function(np.empty_like, None)

        # 测试 np.full_like 函数
        def test_filled_like(self):
            self.check_like_function(np.full_like, 0, True)
            self.check_like_function(np.full_like, 1, True)
            self.check_like_function(np.full_like, 1000, True)
            self.check_like_function(np.full_like, 123.456, True)
            # Inf 转换为整数会导致无效值错误，此处忽略这些错误
            with np.errstate(invalid="ignore"):
                self.check_like_function(np.full_like, np.inf, True)

        # 使用参数化测试装饰器定义多个参数化测试，likefunc 参数为函数，dtype 参数为数据类型
        @pytest.mark.parametrize('likefunc', [np.empty_like, np.full_like,
                                              np.zeros_like, np.ones_like])
        @pytest.mark.parametrize('dtype', [str, bytes])
    # 定义一个测试函数，用于测试指定的 likefunc 和 dtype 参数
    def test_dtype_str_bytes(self, likefunc, dtype):
        # Regression test for gh-19860
        # 创建一个二维数组 a，包含 16 个元素，reshape 成 2 行 8 列
        a = np.arange(16).reshape(2, 8)
        # 创建数组 b，从 a 中选择所有行，但列的步长为 2，使得 b 不是连续的内存布局
        b = a[:, ::2]  # Ensure b is not contiguous.
        # 根据 likefunc 的类型选择不同的 kwargs 参数
        kwargs = {'fill_value': ''} if likefunc == np.full_like else {}
        # 使用 likefunc 函数对数组 b 进行操作，指定 dtype 和可能的 kwargs 参数
        result = likefunc(b, dtype=dtype, **kwargs)
        # 根据 dtype 的类型进行断言
        if dtype == str:
            # 如果 dtype 是 str，验证结果数组的步长为 (16, 4)
            assert result.strides == (16, 4)
        else:
            # 如果 dtype 是 bytes
            # 验证结果数组的步长为 (4, 1)
            assert result.strides == (4, 1)
class TestCorrelate:
    # 定义测试类 TestCorrelate
    def _setup(self, dt):
        # 定义私有方法 _setup，用于设置测试数据
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        # 创建数组 self.x，包含整数或浮点数，根据参数 dt 决定数据类型
        self.xs = np.arange(1, 20)[::3]
        # 创建数组 self.xs，包含从 1 到 19 的整数，步长为 3
        self.y = np.array([-1, -2, -3], dtype=dt)
        # 创建数组 self.y，包含整数或浮点数，根据参数 dt 决定数据类型
        self.z1 = np.array([-3., -8., -14., -20., -26., -14., -5.], dtype=dt)
        # 创建数组 self.z1，包含整数或浮点数，根据参数 dt 决定数据类型
        self.z1_4 = np.array([-2., -5., -8., -11., -14., -5.], dtype=dt)
        # 创建数组 self.z1_4，包含整数或浮点数，根据参数 dt 决定数据类型
        self.z1r = np.array([-15., -22., -22., -16., -10., -4., -1.], dtype=dt)
        # 创建数组 self.z1r，包含整数或浮点数，根据参数 dt 决定数据类型
        self.z2 = np.array([-5., -14., -26., -20., -14., -8., -3.], dtype=dt)
        # 创建数组 self.z2，包含整数或浮点数，根据参数 dt 决定数据类型
        self.z2r = np.array([-1., -4., -10., -16., -22., -22., -15.], dtype=dt)
        # 创建数组 self.z2r，包含整数或浮点数，根据参数 dt 决定数据类型
        self.zs = np.array([-3., -14., -30., -48., -66., -84.,
                           -102., -54., -19.], dtype=dt)
        # 创建数组 self.zs，包含整数或浮点数，根据参数 dt 决定数据类型

    def test_float(self):
        # 定义测试方法 test_float，测试浮点数类型的数据
        self._setup(float)
        # 调用 _setup 方法，使用浮点数类型设置测试数据
        z = np.correlate(self.x, self.y, 'full')
        # 计算 self.x 和 self.y 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z1)
        # 断言 z 与预期的 self.z1 几乎相等
        z = np.correlate(self.x, self.y[:-1], 'full')
        # 计算 self.x 和 self.y 的前 n-1 个元素的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z1_4)
        # 断言 z 与预期的 self.z1_4 几乎相等
        z = np.correlate(self.y, self.x, 'full')
        # 计算 self.y 和 self.x 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z2)
        # 断言 z 与预期的 self.z2 几乎相等
        z = np.correlate(self.x[::-1], self.y, 'full')
        # 计算 self.x 的逆序和 self.y 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z1r)
        # 断言 z 与预期的 self.z1r 几乎相等
        z = np.correlate(self.y, self.x[::-1], 'full')
        # 计算 self.y 和 self.x 的逆序的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z2r)
        # 断言 z 与预期的 self.z2r 几乎相等
        z = np.correlate(self.xs, self.y, 'full')
        # 计算 self.xs 和 self.y 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.zs)
        # 断言 z 与预期的 self.zs 几乎相等

    def test_object(self):
        # 定义测试方法 test_object，测试对象数据类型的数据
        self._setup(Decimal)
        # 调用 _setup 方法，使用 Decimal 类型设置测试数据
        z = np.correlate(self.x, self.y, 'full')
        # 计算 self.x 和 self.y 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z1)
        # 断言 z 与预期的 self.z1 几乎相等
        z = np.correlate(self.y, self.x, 'full')
        # 计算 self.y 和 self.x 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, self.z2)
        # 断言 z 与预期的 self.z2 几乎相等

    def test_no_overwrite(self):
        # 定义测试方法 test_no_overwrite，测试不覆盖的情况
        d = np.ones(100)
        # 创建数组 d，包含 100 个值为 1 的元素
        k = np.ones(3)
        # 创建数组 k，包含 3 个值为 1 的元素
        np.correlate(d, k)
        # 计算数组 d 和 k 的互相关
        assert_array_equal(d, np.ones(100))
        # 断言数组 d 与一个值为 1 的数组相等
        assert_array_equal(k, np.ones(3))
        # 断言数组 k 与一个值为 1 的数组相等

    def test_complex(self):
        # 定义测试方法 test_complex，测试复数类型的数据
        x = np.array([1, 2, 3, 4+1j], dtype=complex)
        # 创建复数数组 x，包含复数和实数部分
        y = np.array([-1, -2j, 3+1j], dtype=complex)
        # 创建复数数组 y，包含复数和实数部分
        r_z = np.array([3-1j, 6, 8+1j, 11+5j, -5+8j, -4-1j], dtype=complex)
        # 创建复数数组 r_z，包含复数和实数部分
        r_z = r_z[::-1].conjugate()
        # 将 r_z 数组逆序并共轭
        z = np.correlate(y, x, mode='full')
        # 计算 y 和 x 的互相关，模式为 'full'，结果赋给 z
        assert_array_almost_equal(z, r_z)
        # 断言 z 与预期的 r_z 几乎相等

    def test_zero_size(self):
        # 定义测试方法 test_zero_size，测试零大小的输入
        with pytest.raises(ValueError):
            # 断言抛出 ValueError 异常
            np.correlate(np.array([]), np.ones(1000), mode='full')
            # 计算空数组和包含 1000 个值为 1 的数组的互相关，模式为 'full'
        with pytest.raises(ValueError):
            # 断言抛出 ValueError 异常
            np.correlate(np.ones(1000), np.array([]), mode='full')
            # 计算包含 1000 个值为 1 的数组和空数组的互相关，模式为 'full'

    def test_mode(self):
        # 定义测试方法 test_mode，测试不同的计算模式
        d = np.ones(100)
        # 创建数组 d，包含 100 个值为 1 的元素
        k = np.ones(3)
        # 创建数组 k，包含 3 个值为 1 的元素
        default_mode = np.correlate(d, k, mode='valid')
        # 计算数组 d 和 k 的互相关，模式为 'valid'，结果赋给 default_mode
        with assert_warns(Dep
    # 定义测试函数 `test_object`
    def test_object(self):
        # 创建包含 100 个浮点数 1.0 的列表 `d`
        d = [1.] * 100
        # 创建包含 3 个浮点数 1.0 的列表 `k`
        k = [1.] * 3
        # 断言使用 `np.convolve` 对 `d` 和 `k` 进行卷积后，结果数组中间部分与预期的全 3 数组相等
        assert_array_almost_equal(np.convolve(d, k)[2:-2], np.full(98, 3))

    # 定义测试函数 `test_no_overwrite`
    def test_no_overwrite(self):
        # 创建包含 100 个 1.0 的 numpy 数组 `d`
        d = np.ones(100)
        # 创建包含 3 个 1.0 的 numpy 数组 `k`
        k = np.ones(3)
        # 执行 `np.convolve` 操作，但不保存结果
        np.convolve(d, k)
        # 断言 `d` 仍然是包含 100 个 1.0 的数组
        assert_array_equal(d, np.ones(100))
        # 断言 `k` 仍然是包含 3 个 1.0 的数组
        assert_array_equal(k, np.ones(3))

    # 定义测试函数 `test_mode`
    def test_mode(self):
        # 创建包含 100 个 1.0 的 numpy 数组 `d`
        d = np.ones(100)
        # 创建包含 3 个 1.0 的 numpy 数组 `k`
        k = np.ones(3)
        # 使用默认的 'full' 模式对 `d` 和 `k` 执行 `np.convolve` 操作
        default_mode = np.convolve(d, k, mode='full')
        # 使用 'f' 模式对 `d` 和 `k` 执行 `np.convolve` 操作，并警告其已弃用
        with assert_warns(DeprecationWarning):
            full_mode = np.convolve(d, k, mode='f')
        # 断言 'f' 模式下的卷积结果与 'full' 模式下一致
        assert_array_equal(full_mode, default_mode)
        # 使用非法的整数模式（-1），预期引发 ValueError
        with assert_raises(ValueError):
            np.convolve(d, k, mode=-1)
        # 使用整数模式 2，断言其结果与 'full' 模式下一致
        assert_array_equal(np.convolve(d, k, mode=2), full_mode)
        # 使用非法的模式参数 None，预期引发 TypeError
        with assert_raises(TypeError):
            np.convolve(d, k, mode=None)
# 定义一个测试类 TestArgwhere，用于测试 np.argwhere 函数的行为
class TestArgwhere:

    # 使用 pytest.mark.parametrize 装饰器，对 test_nd 方法进行参数化，参数为 nd 可以是 0, 1, 2
    @pytest.mark.parametrize('nd', [0, 1, 2])
    # 定义测试方法 test_nd，接受参数 nd
    def test_nd(self, nd):
        # 创建一个形状为 (2,)*nd 的空布尔数组 x
        x = np.empty((2,)*nd, bool)

        # 将数组 x 所有元素设为 False
        x[...] = False
        # 断言 np.argwhere(x) 的形状为 (0, nd)
        assert_equal(np.argwhere(x).shape, (0, nd))

        # 将数组 x 所有元素设为 False，然后将第一个元素设为 True
        x[...] = False
        x.flat[0] = True
        # 断言 np.argwhere(x) 的形状为 (1, nd)
        assert_equal(np.argwhere(x).shape, (1, nd))

        # 将数组 x 所有元素设为 True，然后将第一个元素设为 False
        x[...] = True
        x.flat[0] = False
        # 断言 np.argwhere(x) 的形状为 (x.size - 1, nd)，即除去一个 False 元素后的形状
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # 将数组 x 所有元素设为 True
        x[...] = True
        # 断言 np.argwhere(x) 的形状为 (x.size, nd)，即所有元素都为 True 的情况下的形状
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    # 定义测试方法 test_2D
    def test_2D(self):
        # 创建一个形状为 (2, 3) 的二维数组 x，内容为 0 到 5
        x = np.arange(6).reshape((2, 3))
        # 断言 np.argwhere(x > 1) 的结果，找出 x 中大于 1 的元素的索引
        assert_array_equal(np.argwhere(x > 1),
                           [[0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2]])

    # 定义测试方法 test_list
    def test_list(self):
        # 断言 np.argwhere([4, 0, 2, 1, 3]) 的结果，找出列表中非零元素的索引
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


# 定义一个测试类 TestRoll，用于测试 np.roll 函数的行为
class TestRoll:

    # 定义测试方法 test_roll1d，测试一维数组的滚动操作
    def test_roll1d(self):
        # 创建一个包含 0 到 9 的一维数组 x
        x = np.arange(10)
        # 对数组 x 进行向右滚动两个位置
        xr = np.roll(x, 2)
        # 断言结果 xr 是否等于 [8, 9, 0, 1, 2, 3, 4, 5, 6, 7]
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))

    # 定义测试方法 test_roll2d，测试二维数组的滚动操作
    def test_roll2d(self):
        # 创建一个形状为 (2, 5) 的二维数组 x2，内容为 0 到 9
        x2 = np.reshape(np.arange(10), (2, 5))
        # 对二维数组 x2 进行沿第一个轴的向右滚动一个位置
        x2r = np.roll(x2, 1)
        # 断言结果 x2r 是否等于 [[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        # 对二维数组 x2 进行沿第零轴（行）的向下滚动一个位置
        x2r = np.roll(x2, 1, axis=0)
        # 断言结果 x2r 是否等于 [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 对二维数组 x2 进行沿第一轴（列）的向右滚动一个位置
        x2r = np.roll(x2, 1, axis=1)
        # 断言结果 x2r 是否等于 [[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # 同时沿多个轴滚动
        x2r = np.roll(x2, 1, axis=(0, 1))
        # 断言结果 x2r 是否等于 [[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        # 反向沿多个轴滚动
        x2r = np.roll(x2, (1, 0), axis=(0, 1))
        # 断言结果 x2r 是否等于 [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 沿多个轴反向滚动
        x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        # 断言结果 x2r 是否等于 [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        # 在同一轴上多次滚动
        x2r = np.roll(x2, (0, 1), axis=(0, 1))
        # 断言结果 x2r 是否等于 [[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # 在同一轴上反向多次滚动
        x2r = np.roll(x2, (0, -1), axis=(0, 1))
        # 断言结果 x2r 是否等于 [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        # 沿多个轴多次滚动
        x2r = np.roll(x2, (1, 1), axis=(0, 1))
        # 断言结果 x2r 是否等于 [[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        # 沿多
    # 定义一个测试方法，用于测试空 NumPy 数组的滚动操作
    def test_roll_empty(self):
        # 创建一个空的 NumPy 数组
        x = np.array([])
        # 断言滚动空数组 x 1 个位置后的结果与空数组相等
        assert_equal(np.roll(x, 1), np.array([]))
# 定义一个测试类 TestRollaxis，用于测试 np.rollaxis 函数的行为
class TestRollaxis:

    # 预期的形状字典，索引为 (axis, start)，适用于形状为 (1, 2, 3, 4) 的数组
    tgtshape = {(0, 0): (1, 2, 3, 4), (0, 1): (1, 2, 3, 4),
                (0, 2): (2, 1, 3, 4), (0, 3): (2, 3, 1, 4),
                (0, 4): (2, 3, 4, 1),
                (1, 0): (2, 1, 3, 4), (1, 1): (1, 2, 3, 4),
                (1, 2): (1, 2, 3, 4), (1, 3): (1, 3, 2, 4),
                (1, 4): (1, 3, 4, 2),
                (2, 0): (3, 1, 2, 4), (2, 1): (1, 3, 2, 4),
                (2, 2): (1, 2, 3, 4), (2, 3): (1, 2, 3, 4),
                (2, 4): (1, 2, 4, 3),
                (3, 0): (4, 1, 2, 3), (3, 1): (1, 4, 2, 3),
                (3, 2): (1, 2, 4, 3), (3, 3): (1, 2, 3, 4),
                (3, 4): (1, 2, 3, 4)}

    # 测试异常情况
    def test_exceptions(self):
        # 创建形状为 (1, 2, 3, 4) 的数组 a
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4)
        # 断言在超出范围的轴参数情况下抛出 AxisError 异常
        assert_raises(AxisError, np.rollaxis, a, -5, 0)
        assert_raises(AxisError, np.rollaxis, a, 0, -5)
        assert_raises(AxisError, np.rollaxis, a, 4, 0)
        assert_raises(AxisError, np.rollaxis, a, 0, 5)

    # 测试结果情况
    def test_results(self):
        # 创建形状为 (1, 2, 3, 4) 的数组 a 的副本
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4).copy()
        # 获取数组 a 的索引
        aind = np.indices(a.shape)
        # 断言数组 a 拥有自己的数据（非视图）
        assert_(a.flags['OWNDATA'])
        # 遍历预期形状字典中的每对 (i, j)
        for (i, j) in self.tgtshape:
            # 正轴，正起始位置
            res = np.rollaxis(a, axis=i, start=j)
            # 获取索引 i0, i1, i2, i3
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            # 断言结果数组的某个元素等于原始数组 a 的对应元素
            assert_(np.all(res[i0, i1, i2, i3] == a))
            # 断言结果数组的形状与预期的形状相符
            assert_(res.shape == self.tgtshape[(i, j)], str((i,j)))
            # 断言结果数组不拥有自己的数据（是视图）
            assert_(not res.flags['OWNDATA'])

            # 负轴，正起始位置
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, j)])
            assert_(not res.flags['OWNDATA'])

            # 正轴，负起始位置
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, 4 - jp)])
            assert_(not res.flags['OWNDATA'])

            # 负轴，负起始位置
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, 4 - jp)])
            assert_(not res.flags['OWNDATA'])
    # 测试 np.moveaxis 函数，将指定轴移动到新的位置
    def test_move_to_end(self):
        # 创建一个随机数组，形状为 (5, 6, 7)
        x = np.random.randn(5, 6, 7)
        # 遍历不同的源轴和预期结果
        for source, expected in [(0, (6, 7, 5)),
                                 (1, (5, 7, 6)),
                                 (2, (5, 6, 7)),
                                 (-1, (5, 6, 7))]:
            # 调用 np.moveaxis 函数，移动源轴到末尾位置，获取实际结果的形状
            actual = np.moveaxis(x, source, -1).shape
            # 使用 assert_ 函数断言实际结果与预期结果相等
            assert_(actual, expected)

    # 测试 np.moveaxis 函数，将指定轴移动到新的位置
    def test_move_new_position(self):
        # 创建一个随机数组，形状为 (1, 2, 3, 4)
        x = np.random.randn(1, 2, 3, 4)
        # 遍历不同的源轴、目标位置和预期结果
        for source, destination, expected in [
                (0, 1, (2, 1, 3, 4)),
                (1, 2, (1, 3, 2, 4)),
                (1, -1, (1, 3, 4, 2)),
                ]:
            # 调用 np.moveaxis 函数，将源轴移动到目标位置，获取实际结果的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用 assert_ 函数断言实际结果与预期结果相等
            assert_(actual, expected)

    # 测试 np.moveaxis 函数，确保轴的顺序不变
    def test_preserve_order(self):
        # 创建一个全零数组，形状为 (1, 2, 3, 4)
        x = np.zeros((1, 2, 3, 4))
        # 遍历不同的源轴和目标位置
        for source, destination in [
                (0, 0),
                (3, -1),
                (-1, 3),
                ([0, -1], [0, -1]),
                ([2, 0], [2, 0]),
                (range(4), range(4)),
                ]:
            # 调用 np.moveaxis 函数，移动源轴到目标位置，获取实际结果的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用 assert_ 函数断言实际结果与预期形状相同
            assert_(actual, (1, 2, 3, 4))

    # 测试 np.moveaxis 函数，移动多个轴
    def test_move_multiples(self):
        # 创建一个全零数组，形状为 (0, 1, 2, 3)
        x = np.zeros((0, 1, 2, 3))
        # 遍历不同的源轴列表、目标位置列表和预期结果
        for source, destination, expected in [
                ([0, 1], [2, 3], (2, 3, 0, 1)),
                ([2, 3], [0, 1], (2, 3, 0, 1)),
                ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
                ([3, 0], [1, 0], (0, 3, 1, 2)),
                ([0, 3], [0, 1], (0, 3, 1, 2)),
                ]:
            # 调用 np.moveaxis 函数，移动源轴列表到目标位置列表，获取实际结果的形状
            actual = np.moveaxis(x, source, destination).shape
            # 使用 assert_ 函数断言实际结果与预期形状相同
            assert_(actual, expected)

    # 测试 np.moveaxis 函数，处理错误情况
    def test_errors(self):
        # 创建一个随机数组，形状为 (1, 2, 3)
        x = np.random.randn(1, 2, 3)
        # 使用 assert_raises_regex 函数断言调用 np.moveaxis 函数时的异常情况
        assert_raises_regex(AxisError, 'source.*out of bounds',
                            np.moveaxis, x, 3, 0)
        assert_raises_regex(AxisError, 'source.*out of bounds',
                            np.moveaxis, x, -4, 0)
        assert_raises_regex(AxisError, 'destination.*out of bounds',
                            np.moveaxis, x, 0, 5)
        assert_raises_regex(ValueError, 'repeated axis in `source`',
                            np.moveaxis, x, [0, 0], [0, 1])
        assert_raises_regex(ValueError, 'repeated axis in `destination`',
                            np.moveaxis, x, [0, 1], [1, 1])
        assert_raises_regex(ValueError, 'must have the same number',
                            np.moveaxis, x, 0, [0, 1])
        assert_raises_regex(ValueError, 'must have the same number',
                            np.moveaxis, x, [0, 1], [0])

    # 测试 np.moveaxis 函数，处理类数组对象
    def test_array_likes(self):
        # 创建一个 MaskedArray，形状为 (1, 2, 3)，全零
        x = np.ma.zeros((1, 2, 3))
        # 调用 np.moveaxis 函数，移动轴，保持形状不变
        result = np.moveaxis(x, 0, 0)
        # 使用 assert_ 函数断言形状与预期一致，并且结果是 MaskedArray 类型
        assert_(x.shape, result.shape)
        assert_(isinstance(result, np.ma.MaskedArray))

        # 创建一个列表 [1, 2, 3]
        x = [1, 2, 3]
        # 调用 np.moveaxis 函数，移动轴，保持不变
        result = np.moveaxis(x, 0, 0)
        # 使用 assert_ 函数断言结果与原始列表相同，并且结果是 np.ndarray 类型
        assert_(x, list(result))
        assert_(isinstance(result, np.ndarray))
# 定义一个名为 TestCross 的测试类，用于测试 np.cross 函数的不同用例
class TestCross:
    # 在运行该测试方法时忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:.*2-dimensional vectors.*:DeprecationWarning"
    )
    # 测试两个二维向量的叉乘
    def test_2x2(self):
        u = [1, 2]  # 第一个二维向量
        v = [3, 4]  # 第二个二维向量
        z = -2  # 期望的叉乘结果
        cp = np.cross(u, v)  # 计算向量 u 和 v 的叉乘
        assert_equal(cp, z)  # 断言计算结果与期望值相等
        cp = np.cross(v, u)  # 计算向量 v 和 u 的叉乘
        assert_equal(cp, -z)  # 断言计算结果与期望值相等

    # 在运行该测试方法时忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:.*2-dimensional vectors.*:DeprecationWarning"
    )
    # 测试一个二维向量和一个三维向量的叉乘
    def test_2x3(self):
        u = [1, 2]  # 一个二维向量
        v = [3, 4, 5]  # 一个三维向量
        z = np.array([10, -5, -2])  # 期望的叉乘结果
        cp = np.cross(u, v)  # 计算向量 u 和 v 的叉乘
        assert_equal(cp, z)  # 断言计算结果与期望值相等
        cp = np.cross(v, u)  # 计算向量 v 和 u 的叉乘
        assert_equal(cp, -z)  # 断言计算结果与期望值相等

    # 测试两个三维向量的叉乘
    def test_3x3(self):
        u = [1, 2, 3]  # 第一个三维向量
        v = [4, 5, 6]  # 第二个三维向量
        z = np.array([-3, 6, -3])  # 期望的叉乘结果
        cp = np.cross(u, v)  # 计算向量 u 和 v 的叉乘
        assert_equal(cp, z)  # 断言计算结果与期望值相等
        cp = np.cross(v, u)  # 计算向量 v 和 u 的叉乘
        assert_equal(cp, -z)  # 断言计算结果与期望值相等

    # 在运行该测试方法时忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:.*2-dimensional vectors.*:DeprecationWarning"
    )
    # 测试广播形式下的多个向量叉乘
    def test_broadcasting(self):
        # Ticket #2624 (Trac #2032)
        u = np.tile([1, 2], (11, 1))  # 在行方向上复制向量 [1, 2]，共 11 行
        v = np.tile([3, 4], (11, 1))  # 在行方向上复制向量 [3, 4]，共 11 行
        z = -2  # 期望的叉乘结果
        assert_equal(np.cross(u, v), z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(v, u), -z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(u, u), 0)  # 断言向量自身叉乘结果为零

        u = np.tile([1, 2], (11, 1)).T  # 在列方向上复制向量 [1, 2]，共 11 列
        v = np.tile([3, 4, 5], (11, 1))  # 在行方向上复制向量 [3, 4, 5]，共 11 行
        z = np.tile([10, -5, -2], (11, 1))  # 期望的叉乘结果
        assert_equal(np.cross(u, v, axisa=0), z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(v, u.T), -z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(v, v), 0)  # 断言向量自身叉乘结果为零

        u = np.tile([1, 2, 3], (11, 1)).T  # 在列方向上复制向量 [1, 2, 3]，共 11 列
        v = np.tile([3, 4], (11, 1)).T  # 在列方向上复制向量 [3, 4]，共 11 列
        z = np.tile([-12, 9, -2], (11, 1))  # 期望的叉乘结果
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(v.T, u.T), -z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(u.T, u.T), 0)  # 断言向量自身叉乘结果为零

        u = np.tile([1, 2, 3], (5, 1))  # 在行方向上复制向量 [1, 2, 3]，共 5 行
        v = np.tile([4, 5, 6], (5, 1)).T  # 在列方向上复制向量 [4, 5, 6]，共 5 列
        z = np.tile([-3, 6, -3], (5, 1))  # 期望的叉乘结果
        assert_equal(np.cross(u, v, axisb=0), z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(v.T, u), -z)  # 断言计算结果与期望值相等
        assert_equal(np.cross(u, u), 0)  # 断言向量自身叉乘结果为零

    # 在运行该测试方法时忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:.*2-dimensional vectors.*:DeprecationWarning"
    )
    # 测试不同形状的向量广播叉乘后的结果形状
    def test_broadcasting_shapes(self):
        u = np.ones((2, 1, 3))  # 形状为 (2, 1, 3) 的全 1 数组
        v = np.ones((5, 3))  # 形状为 (5, 3) 的全 1 数组
        assert_equal(np.cross(u, v).shape, (2, 5, 3))  # 断言叉乘结果的形状符合预期
        u = np.ones((10, 3, 5))  # 形状为 (10, 3, 5) 的全 1 数组
        v = np.ones((2, 5))  # 形状为 (2, 5) 的全 1 数组
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))  # 断言叉乘结果的形状符合预期
        assert_raises(AxisError, np.cross, u, v, axisa=1, axisb=2)  # 断言会抛出 AxisError 异常
        assert_raises(AxisError, np.cross, u, v, axisa=3, axisb=0)  # 断言会抛出 AxisError 异常
        u = np.ones((10, 3, 5, 7))  # 形状为 (10, 3, 5, 7) 的全 1 数组
        v = np.ones((5, 7
    # 定义一个测试方法，用于检验混合数据类型在 np.cross 函数中的行为，针对 GitHub 问题 #19138 进行回归测试
    def test_uint8_int32_mixed_dtypes(self):
        # 创建一个包含一个元素的二维数组，元素是 np.uint8 类型
        u = np.array([[195, 8, 9]], np.uint8)
        # 创建一个包含三个元素的一维数组，元素是 np.int32 类型
        v = np.array([250, 166, 68], np.int32)
        # 创建一个二维数组，包含三个元素，元素是 np.int32 类型
        z = np.array([[950, 11010, -30370]], dtype=np.int32)
        # 断言 np.cross(v, u) 的结果与 z 相等
        assert_equal(np.cross(v, u), z)
        # 断言 np.cross(u, v) 的结果与 -z 相等
        assert_equal(np.cross(u, v), -z)

    # 使用 pytest 的参数化装饰器标记这个测试方法，提供不同的参数组合进行测试
    @pytest.mark.parametrize("a, b", [(0, [1, 2]), ([1, 2], 3)])
    # 定义测试方法，用于测试零维数组输入时 np.cross 函数的行为
    def test_zero_dimension(self, a, b):
        # 使用 pytest.raises 检测是否会抛出 ValueError 异常，并捕获异常对象到 exc 变量
        with pytest.raises(ValueError) as exc:
            # 调用 np.cross(a, b)，期望会抛出异常
            np.cross(a, b)
        # 断言异常对象的字符串表示中包含 "At least one array has zero dimension"
        assert "At least one array has zero dimension" in str(exc.value)
def test_outer_out_param():
    # 创建一个长度为5的全1数组
    arr1 = np.ones((5,))
    # 创建一个长度为2的全1数组
    arr2 = np.ones((2,))
    # 在[-2, 2]之间生成5个等间距的数列
    arr3 = np.linspace(-2, 2, 5)
    # 创建一个形状为(5,5)的空数组
    out1 = np.ndarray(shape=(5,5))
    # 创建一个形状为(2,5)的空数组
    out2 = np.ndarray(shape=(2, 5))
    # 计算arr1和arr3的外积，结果保存到out1中
    res1 = np.outer(arr1, arr3, out1)
    # 断言计算结果与out1相等
    assert_equal(res1, out1)
    # 计算arr2和arr3的外积，结果保存到out2中，并断言结果与out2相等
    assert_equal(np.outer(arr2, arr3, out2), out2)


class TestIndices:

    def test_simple(self):
        # 创建一个4x3的二维数组，其中x表示行索引，y表示列索引
        [x, y] = np.indices((4, 3))
        # 断言x的值与预期的二维数组相等
        assert_array_equal(x, np.array([[0, 0, 0],
                                        [1, 1, 1],
                                        [2, 2, 2],
                                        [3, 3, 3]]))
        # 断言y的值与预期的二维数组相等
        assert_array_equal(y, np.array([[0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2]]))

    def test_single_input(self):
        # 创建一个长度为4的一维数组，表示索引数组x，断言其值与预期的一维数组相等
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))

        # 创建一个长度为4的一维数组，表示索引数组x（稀疏表示），断言其值与预期的一维数组相等
        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        # 断言对于空元组，返回一个空数组
        assert_array_equal([], np.indices(()))
        # 断言对于空元组（稀疏表示），返回一个空数组
        assert_array_equal([], np.indices((), sparse=True))
        # 断言对于(0,)形状的元组，返回一个包含一个空数组的数组
        assert_array_equal([[]], np.indices((0,)))
        # 断言对于(0,)形状的元组（稀疏表示），返回一个包含一个空数组的数组
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        # 创建一个稀疏表示的索引数组x和y，断言其值与预期的二维数组相等
        [x, y] = np.indices((4,3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dims", [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        # 根据给定的dtype和dims创建索引数组inds，并断言其dtype与预期一致
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)

        # 对于稀疏表示的每个数组，断言其dtype与预期一致
        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)


class TestRequire:
    flag_names = ['C', 'C_CONTIGUOUS', 'CONTIGUOUS',
                  'F', 'F_CONTIGUOUS', 'FORTRAN',
                  'A', 'ALIGNED',
                  'W', 'WRITEABLE',
                  'O', 'OWNDATA']

    def generate_all_false(self, dtype):
        # 创建一个包含一个名为'a'的字段的零数组，设置其为不可写
        arr = np.zeros((2, 2), [('junk', 'i1'), ('a', dtype)])
        arr.setflags(write=False)
        # 获取字段'a'，断言其不是C连续的，不是F连续的，不是OWNDATA的，不可写，不是对齐的
        a = arr['a']
        assert_(not a.flags['C'])
        assert_(not a.flags['F'])
        assert_(not a.flags['O'])
        assert_(not a.flags['W'])
        assert_(not a.flags['A'])
        # 返回字段'a'
        return a

    def set_and_check_flag(self, flag, dtype, arr):
        # 如果dtype为None，则使用arr的dtype
        if dtype is None:
            dtype = arr.dtype
        # 将arr转换为指定dtype的数组，并设置指定的flag，断言设置后相应flag为True，并且dtype与预期一致
        b = np.require(arr, dtype, [flag])
        assert_(b.flags[flag])
        assert_(b.dtype == dtype)

        # 进一步调用np.require应该返回相同的数组，除非指定了OWNDATA
        c = np.require(b, None, [flag])
        if flag[0] != 'O':
            assert_(c is b)
        else:
            assert_(c.flags[flag])
    # 定义测试方法，测试对每个组合的 id、fd、flag 进行生成和设置检查
    def test_require_each(self):
        # 定义 id 和 fd 的可能取值
        id = ['f8', 'i4']
        fd = [None, 'f8', 'c16']
        # 对 id、fd 和 flag 的所有组合进行迭代
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            # 生成一个全为假值的数组
            a = self.generate_all_false(idtype)
            # 设置并检查指定标志位和类型后的数组
            self.set_and_check_flag(flag, fdtype,  a)

    # 测试对未知需求的处理，预期引发 KeyError 异常
    def test_unknown_requirement(self):
        # 生成一个全为假值的 'f8' 类型数组
        a = self.generate_all_false('f8')
        # 断言调用 np.require 函数时会引发 KeyError 异常
        assert_raises(KeyError, np.require, a, None, 'Q')

    # 测试对非数组输入的处理
    def test_non_array_input(self):
        # 要求将列表转换为 'i4' 类型数组，要求包含 'C', 'A', 'O' 标志位
        a = np.require([1, 2, 3, 4], 'i4', ['C', 'A', 'O'])
        # 断言数组的 'O', 'C', 'A' 标志位都为真
        assert_(a.flags['O'])
        assert_(a.flags['C'])
        assert_(a.flags['A'])
        # 断言数组的数据类型为 'i4'
        assert_(a.dtype == 'i4')
        # 断言数组的值等于原始列表
        assert_equal(a, [1, 2, 3, 4])

    # 测试同时设置 'C' 和 'F' 标志位的处理，预期引发 ValueError 异常
    def test_C_and_F_simul(self):
        # 生成一个全为假值的 'f8' 类型数组
        a = self.generate_all_false('f8')
        # 断言调用 np.require 函数时会引发 ValueError 异常
        assert_raises(ValueError, np.require, a, None, ['C', 'F'])

    # 测试保留数组子类的处理
    def test_ensure_array(self):
        # 定义一个继承自 np.ndarray 的数组子类
        class ArraySubclass(np.ndarray):
            pass
        # 创建一个 ArraySubclass 类型的数组对象
        a = ArraySubclass((2, 2))
        # 要求将数组对象转换为标准的 np.ndarray 类型
        b = np.require(a, None, ['E'])
        # 断言 b 是标准的 np.ndarray 类型
        assert_(type(b) is np.ndarray)

    # 测试保留子类类型的标志位设置
    def test_preserve_subtype(self):
        # 定义一个继承自 np.ndarray 的数组子类
        class ArraySubclass(np.ndarray):
            pass
        # 对每个标志位进行迭代
        for flag in self.flag_names:
            # 创建一个 ArraySubclass 类型的数组对象
            a = ArraySubclass((2, 2))
            # 设置并检查指定标志位后的数组
            self.set_and_check_flag(flag, None, a)
class TestBroadcast:
    def test_broadcast_in_args(self):
        # 定义四个不同形状的数组
        arrs = [np.empty((6, 7)), np.empty((5, 6, 1)), np.empty((7,)),
                np.empty((5, 1, 7))]
        # 创建四个不同的广播对象
        mits = [np.broadcast(*arrs),
                np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])),
                np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])),
                np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])),
                np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1])]
        # 遍历每个广播对象
        for mit in mits:
            # 断言广播对象的形状为 (5, 6, 7)
            assert_equal(mit.shape, (5, 6, 7))
            # 断言广播对象的维度为 3
            assert_equal(mit.ndim, 3)
            # 断言广播对象的维度属性为 3（假设此处可能是笔误，正确应为 ndim）
            assert_equal(mit.nd, 3)
            # 断言广播对象的迭代器数量为 4
            assert_equal(mit.numiter, 4)
            # 验证每个数组与其对应迭代器的基础对象相同
            for a, ia in zip(arrs, mit.iters):
                assert_(a is ia.base)

    def test_broadcast_single_arg(self):
        # 测试单个数组的广播
        # 创建包含一个数组的列表
        arrs = [np.empty((5, 6, 7))]
        # 创建广播对象
        mit = np.broadcast(*arrs)
        # 断言广播对象的形状为 (5, 6, 7)
        assert_equal(mit.shape, (5, 6, 7))
        # 断言广播对象的维度为 3
        assert_equal(mit.ndim, 3)
        # 断言广播对象的维度属性为 3（假设此处可能是笔误，正确应为 ndim）
        assert_equal(mit.nd, 3)
        # 断言广播对象的迭代器数量为 1
        assert_equal(mit.numiter, 1)
        # 验证数组与其对应迭代器的基础对象相同
        assert_(arrs[0] is mit.iters[0].base)

    def test_number_of_arguments(self):
        # 测试不同数量的数组作为参数的情况
        arr = np.empty((5,))
        # 循环测试数组数量从 0 到 69
        for j in range(70):
            arrs = [arr] * j
            # 如果数组数量超过 64，应该引发 ValueError
            if j > 64:
                assert_raises(ValueError, np.broadcast, *arrs)
            else:
                # 创建广播对象
                mit = np.broadcast(*arrs)
                # 断言广播对象的迭代器数量为 j
                assert_equal(mit.numiter, j)

    def test_broadcast_error_kwargs(self):
        # 测试使用关键字参数创建广播对象
        arrs = [np.empty((5, 6, 7))]
        # 创建两个相同的广播对象
        mit  = np.broadcast(*arrs)
        mit2 = np.broadcast(*arrs, **{})
        # 断言两个广播对象的形状相同
        assert_equal(mit.shape, mit2.shape)
        # 断言两个广播对象的维度相同
        assert_equal(mit.ndim, mit2.ndim)
        # 断言两个广播对象的维度属性相同（假设此处可能是笔误，正确应为 ndim）
        assert_equal(mit.nd, mit2.nd)
        # 断言两个广播对象的迭代器数量相同
        assert_equal(mit.numiter, mit2.numiter)
        # 验证两个广播对象中第一个数组的基础对象相同
        assert_(mit.iters[0].base is mit2.iters[0].base)

        # 使用关键字参数创建广播对象时应该引发 ValueError
        assert_raises(ValueError, np.broadcast, 1, **{'x': 1})

    def test_shape_mismatch_error_message(self):
        # 测试形状不匹配时的错误消息
        with pytest.raises(ValueError, match=r"arg 0 with shape \(1, 3\) and "
                                             r"arg 2 with shape \(2,\)"):
            np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])


class TestKeepdims:

    class sub_array(np.ndarray):
        def sum(self, axis=None, dtype=None, out=None):
            return np.ndarray.sum(self, axis, dtype, out, keepdims=True)

    def test_raise(self):
        # 测试抛出异常情况
        sub_class = self.sub_array
        x = np.arange(30).view(sub_class)
        # 断言调用 np.sum 时会引发 TypeError
        assert_raises(TypeError, np.sum, x, keepdims=True)


class TestTensordot:

    def test_zero_dimension(self):
        # 测试处理零维数组的情况
        # 创建两个零维数组
        a = np.ndarray((3,0))
        b = np.ndarray((0,4))
        # 执行张量积运算
        td = np.tensordot(a, b, (1, 0))
        # 断言结果与使用 np.dot 函数的结果相同
        assert_array_equal(td, np.dot(a, b))
        # 断言结果与使用 np.einsum 函数的结果相同
        assert_array_equal(td, np.einsum('ij,jk', a, b))
    # 定义一个测试方法，用于测试零维数组的情况
    def test_zero_dimensional(self):
        # 标记：gh-12130，指示这个测试与GitHub上的issue编号12130相关联
        arr_0d = np.array(1)  # 创建一个零维数组，包含单个元素1
        # 使用 np.tensordot 对两个零维数组 arr_0d 进行张量点积，[]表示无需收缩任何轴
        ret = np.tensordot(arr_0d, arr_0d, ([], []))  # contracting no axes is well defined
        # 断言：验证返回的张量点积结果与输入的零维数组 arr_0d 相等
        assert_array_equal(ret, arr_0d)
class TestAsType:

    def test_astype(self):
        data = [[1, 2], [3, 4]]  # 定义一个二维列表作为测试数据
        actual = np.astype(  # 调用 np.astype 方法进行类型转换
            np.array(data, dtype=np.int64), np.uint32  # 将二维列表转换为 np.int64 类型的 NumPy 数组，再转换为 np.uint32 类型
        )
        expected = np.array(data, dtype=np.uint32)  # 创建预期结果的 NumPy 数组，数据类型为 np.uint32

        assert_array_equal(actual, expected)  # 断言 actual 和 expected 数组内容相等
        assert_equal(actual.dtype, expected.dtype)  # 断言 actual 和 expected 数组的数据类型相等

        assert np.shares_memory(  # 使用 np.shares_memory 检查两个数组是否共享内存
            actual, np.astype(actual, actual.dtype, copy=False)  # 调用 np.astype 方法进行类型转换，不复制数据
        )

        with pytest.raises(TypeError, match="Input should be a NumPy array"):  # 使用 pytest 断言捕获 TypeError 异常，检查异常信息
            np.astype(data, np.float64)  # 尝试对非 NumPy 数组进行类型转换，期望引发异常
```