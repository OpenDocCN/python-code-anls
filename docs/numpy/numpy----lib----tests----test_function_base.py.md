# `.\numpy\numpy\lib\tests\test_function_base.py`

```
# 导入运算符模块，提供对Python内置操作符的额外支持
import operator
# 引入警告模块，用于处理警告信息
import warnings
# 引入sys模块，提供对Python运行时环境的访问
import sys
# 引入decimal模块，支持十进制数运算
import decimal
# 从fractions模块中导入Fraction类，支持有理数运算
from fractions import Fraction
# 引入math模块，提供数学函数和常数
import math
# 引入pytest模块，用于编写和运行Python的单元测试
import pytest
# 引入hypothesis模块，支持基于属性的测试生成
import hypothesis
# 从hypothesis.extra.numpy模块中导入arrays函数，支持生成NumPy数组的策略
from hypothesis.extra.numpy import arrays
# 引入hypothesis.strategies模块，提供更多测试策略
import hypothesis.strategies as st
# 从functools模块中导入partial函数，支持创建偏函数
from functools import partial

# 导入NumPy库，并且从中导入多个函数和类
import numpy as np
from numpy import (
    ma, angle, average, bartlett, blackman, corrcoef, cov,
    delete, diff, digitize, extract, flipud, gradient, hamming, hanning,
    i0, insert, interp, kaiser, meshgrid, piecewise, place, rot90,
    select, setxor1d, sinc, trapezoid, trim_zeros, unwrap, unique, vectorize
    )
# 从NumPy异常模块导入AxisError异常类
from numpy.exceptions import AxisError
# 从NumPy测试模块导入多个断言函数和工具函数
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_allclose,
    assert_warns, assert_raises_regex, suppress_warnings, HAS_REFCOUNT,
    IS_WASM, NOGIL_BUILD
    )
# 导入NumPy的函数基础实现模块
import numpy.lib._function_base_impl as nfb
# 从NumPy随机模块中导入rand函数，支持生成随机数数组
from numpy.random import rand
# 从NumPy核心数值模块中导入normalize_axis_tuple函数，用于规范化轴元组

def get_mat(n):
    # 生成一个从0到n-1的整数数组
    data = np.arange(n)
    # 计算data与data的外积并返回结果
    data = np.add.outer(data, data)
    return data

def _make_complex(real, imag):
    """
    Like real + 1j * imag, but behaves as expected when imag contains non-finite
    values
    """
    # 根据real和imag创建一个复数数组，处理imag中包含的非有限值
    ret = np.zeros(np.broadcast(real, imag).shape, np.complex128)
    ret.real = real
    ret.imag = imag
    return ret

class TestRot90:
    def test_basic(self):
        # 测试当输入不符合预期时，是否能够正确引发ValueError异常
        assert_raises(ValueError, rot90, np.ones(4))
        assert_raises(ValueError, rot90, np.ones((2,2,2)), axes=(0,1,2))
        assert_raises(ValueError, rot90, np.ones((2,2)), axes=(0,2))
        assert_raises(ValueError, rot90, np.ones((2,2)), axes=(1,1))
        assert_raises(ValueError, rot90, np.ones((2,2,2)), axes=(-2,1))

        # 创建一个二维数组a
        a = [[0, 1, 2],
             [3, 4, 5]]
        # 创建与a旋转90度后的期望结果b1、b2、b3、b4
        b1 = [[2, 5],
              [1, 4],
              [0, 3]]
        b2 = [[5, 4, 3],
              [2, 1, 0]]
        b3 = [[3, 0],
              [4, 1],
              [5, 2]]
        b4 = [[0, 1, 2],
              [3, 4, 5]]

        # 使用不同的旋转因子k，验证rot90函数的输出是否与预期相符
        for k in range(-3, 13, 4):
            assert_equal(rot90(a, k=k), b1)
        for k in range(-2, 13, 4):
            assert_equal(rot90(a, k=k), b2)
        for k in range(-1, 13, 4):
            assert_equal(rot90(a, k=k), b3)
        for k in range(0, 13, 4):
            assert_equal(rot90(a, k=k), b4)

        # 验证rot90函数在指定轴顺序时，是否能够正确恢复原始数组a
        assert_equal(rot90(rot90(a, axes=(0,1)), axes=(1,0)), a)
        assert_equal(rot90(a, k=1, axes=(1,0)), rot90(a, k=-1, axes=(0,1)))

    def test_axes(self):
        # 创建一个三维数组a
        a = np.ones((50, 40, 3))
        # 验证默认情况下，rot90函数对数组形状的影响
        assert_equal(rot90(a).shape, (40, 50, 3))
        # 验证指定不同轴顺序时，rot90函数的输出是否一致
        assert_equal(rot90(a, axes=(0,2)), rot90(a, axes=(0,-1)))
        assert_equal(rot90(a, axes=(1,2)), rot90(a, axes=(-2,-1)))
    # 定义一个测试函数，用于测试矩阵旋转操作
    def test_rotation_axes(self):
        # 创建一个形状为 (2, 2, 2) 的 NumPy 数组 a，其中包含从 0 到 7 的连续整数
        a = np.arange(8).reshape((2,2,2))

        # 定义旋转90度后的预期结果，关于轴 (0, 1) 的旋转结果
        a_rot90_01 = [[[2, 3],
                       [6, 7]],
                      [[0, 1],
                       [4, 5]]]
        # 定义关于轴 (1, 2) 的旋转结果
        a_rot90_12 = [[[1, 3],
                       [0, 2]],
                      [[5, 7],
                       [4, 6]]]
        # 定义关于轴 (2, 0) 的旋转结果
        a_rot90_20 = [[[4, 0],
                       [6, 2]],
                      [[5, 1],
                       [7, 3]]]
        # 定义关于轴 (1, 0) 的旋转结果
        a_rot90_10 = [[[4, 5],
                       [0, 1]],
                      [[6, 7],
                       [2, 3]]]

        # 断言矩阵 a 经过关于轴 (0, 1) 的旋转结果应为 a_rot90_01
        assert_equal(rot90(a, axes=(0, 1)), a_rot90_01)
        # 断言矩阵 a 经过关于轴 (1, 0) 的旋转结果应为 a_rot90_10
        assert_equal(rot90(a, axes=(1, 0)), a_rot90_10)
        # 断言矩阵 a 经过关于轴 (1, 2) 的旋转结果应为 a_rot90_12
        assert_equal(rot90(a, axes=(1, 2)), a_rot90_12)

        # 循环测试矩阵 a 经过不同次数 k 的关于轴 (2, 0) 的旋转结果是否与预期一致
        for k in range(1, 5):
            assert_equal(rot90(a, k=k, axes=(2, 0)),
                         rot90(a_rot90_20, k=k-1, axes=(2, 0)))
class TestFlip:

    def test_axes(self):
        # 测试在指定轴上翻转数组，预期引发 AxisError 异常
        assert_raises(AxisError, np.flip, np.ones(4), axis=1)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=2)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=-3)
        assert_raises(AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))

    def test_basic_lr(self):
        # 测试基本的左右翻转操作
        a = get_mat(4)  # 调用 get_mat 函数获取一个矩阵 a
        b = a[:, ::-1]  # 通过切片实现左右翻转，赋值给 b
        assert_equal(np.flip(a, 1), b)  # 断言 np.flip 在水平方向翻转后与 b 相等
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[2, 1, 0],
             [5, 4, 3]]
        assert_equal(np.flip(a, 1), b)  # 断言 np.flip 在水平方向翻转后与 b 相等

    def test_basic_ud(self):
        # 测试基本的上下翻转操作
        a = get_mat(4)  # 调用 get_mat 函数获取一个矩阵 a
        b = a[::-1, :]  # 通过切片实现上下翻转，赋值给 b
        assert_equal(np.flip(a, 0), b)  # 断言 np.flip 在垂直方向翻转后与 b 相等
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[3, 4, 5],
             [0, 1, 2]]
        assert_equal(np.flip(a, 0), b)  # 断言 np.flip 在垂直方向翻转后与 b 相等

    def test_3d_swap_axis0(self):
        # 测试在三维数组中沿第 0 轴交换
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])
        b = np.array([[[4, 5],
                       [6, 7]],
                      [[0, 1],
                       [2, 3]]])
        assert_equal(np.flip(a, 0), b)  # 断言 np.flip 在第 0 轴翻转后与 b 相等

    def test_3d_swap_axis1(self):
        # 测试在三维数组中沿第 1 轴交换
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])
        b = np.array([[[2, 3],
                       [0, 1]],
                      [[6, 7],
                       [4, 5]]])
        assert_equal(np.flip(a, 1), b)  # 断言 np.flip 在第 1 轴翻转后与 b 相等

    def test_3d_swap_axis2(self):
        # 测试在三维数组中沿第 2 轴交换
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])
        b = np.array([[[1, 0],
                       [3, 2]],
                      [[5, 4],
                       [7, 6]]])
        assert_equal(np.flip(a, 2), b)  # 断言 np.flip 在第 2 轴翻转后与 b 相等

    def test_4d(self):
        # 测试在四维数组中的各个轴进行翻转操作
        a = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        for i in range(a.ndim):
            assert_equal(np.flip(a, i),
                         np.flipud(a.swapaxes(0, i)).swapaxes(i, 0))
            # 断言 np.flip 在第 i 轴翻转后与垂直翻转后再交换轴得到的结果相等

    def test_default_axis(self):
        # 测试在默认情况下进行数组的翻转操作
        a = np.array([[1, 2, 3],
                      [4, 5, 6]])
        b = np.array([[6, 5, 4],
                      [3, 2, 1]])
        assert_equal(np.flip(a), b)  # 断言 np.flip 在默认情况下翻转后与 b 相等

    def test_multiple_axes(self):
        # 测试在多个轴上进行数组的翻转操作
        a = np.array([[[0, 1],
                       [2, 3]],
                      [[4, 5],
                       [6, 7]]])

        assert_equal(np.flip(a, axis=()), a)  # 断言 np.flip 在空轴上的翻转结果与 a 相等

        b = np.array([[[5, 4],
                       [7, 6]],
                      [[1, 0],
                       [3, 2]]])

        assert_equal(np.flip(a, axis=(0, 2)), b)  # 断言 np.flip 在指定的轴上翻转后与 b 相等

        c = np.array([[[3, 2],
                       [1, 0]],
                      [[7, 6],
                       [5, 4]]])

        assert_equal(np.flip(a, axis=(1, 2)), c)  # 断言 np.flip 在指定的轴上翻转后与 c 相等


class TestAny:
    # 定义一个测试方法，用于测试基本的 np.any() 函数
    def test_basic(self):
        # 创建几个示例数组
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        
        # 断言：检查数组 y1 中是否存在非零元素
        assert_(np.any(y1))
        
        # 断言：检查数组 y3 中是否存在非零元素
        assert_(np.any(y3))
        
        # 断言：检查数组 y2 中是否所有元素都为零
        assert_(not np.any(y2))

    # 定义另一个测试方法，用于测试 np.any() 在多维数组上的应用
    def test_nd(self):
        # 创建一个二维数组 y1
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        
        # 断言：检查数组 y1 中是否存在非零元素
        assert_(np.any(y1))
        
        # 断言：检查数组 y1 在每列（沿 axis=0）上是否存在至少一个非零元素
        assert_array_equal(np.any(y1, axis=0), [1, 1, 0])
        
        # 断言：检查数组 y1 在每行（沿 axis=1）上是否存在至少一个非零元素
        assert_array_equal(np.any(y1, axis=1), [0, 1, 1])
class TestAll:

    def test_basic(self):
        # 定义几个测试用例
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        # 验证条件是否为假，应返回 False
        assert_(not np.all(y1))
        # 验证条件是否全部为真，应返回 True
        assert_(np.all(y3))
        # 验证条件是否为假，应返回 False
        assert_(not np.all(y2))
        # 验证条件是否全部为假，应返回 True
        assert_(np.all(~np.array(y2)))

    def test_nd(self):
        # 定义一个二维数组作为测试用例
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        # 验证条件是否为假，应返回 False
        assert_(not np.all(y1))
        # 按指定轴（axis=0）验证条件是否全部为真，应返回 [False, False, True]
        assert_array_equal(np.all(y1, axis=0), [0, 0, 1])
        # 按指定轴（axis=1）验证条件是否全部为真，应返回 [False, False, True]
        assert_array_equal(np.all(y1, axis=1), [0, 0, 1])


@pytest.mark.parametrize("dtype", ["i8", "U10", "object", "datetime64[ms]"])
def test_any_and_all_result_dtype(dtype):
    # 创建一个全为 1 的数组，指定数据类型为参数 dtype
    arr = np.ones(3, dtype=dtype)
    # 验证数组中是否存在任意非零元素，返回的 dtype 应为布尔类型
    assert np.any(arr).dtype == np.bool
    # 验证数组中所有元素是否都为真，返回的 dtype 应为布尔类型
    assert np.all(arr).dtype == np.bool


class TestCopy:

    def test_basic(self):
        # 创建一个二维数组
        a = np.array([[1, 2], [3, 4]])
        # 复制数组 a，验证副本与原数组相等
        a_copy = np.copy(a)
        assert_array_equal(a, a_copy)
        # 修改副本的值，验证原数组未改变
        a_copy[0, 0] = 10
        assert_equal(a[0, 0], 1)
        assert_equal(a_copy[0, 0], 10)

    def test_order(self):
        # 创建一个二维数组，默认使用 C 风格（行优先）存储
        a = np.array([[1, 2], [3, 4]])
        assert_(a.flags.c_contiguous)
        assert_(not a.flags.f_contiguous)
        # 创建一个使用 Fortran 风格（列优先）存储的数组
        a_fort = np.array([[1, 2], [3, 4]], order="F")
        assert_(not a_fort.flags.c_contiguous)
        assert_(a_fort.flags.f_contiguous)
        # 复制数组 a 和 a_fort，验证复制后的存储方式是否不变
        a_copy = np.copy(a)
        assert_(a_copy.flags.c_contiguous)
        assert_(not a_copy.flags.f_contiguous)
        a_fort_copy = np.copy(a_fort)
        assert_(not a_fort_copy.flags.c_contiguous)
        assert_(a_fort_copy.flags.f_contiguous)

    def test_subok(self):
        # 创建一个掩码数组
        mx = ma.ones(5)
        # 复制数组 mx，不保留子类，验证复制后不是掩码数组
        assert_(not ma.isMaskedArray(np.copy(mx, subok=False)))
        # 复制数组 mx，保留子类，验证复制后仍为掩码数组
        assert_(ma.isMaskedArray(np.copy(mx, subok=True)))
        # 默认行为，复制数组 mx，验证复制后不是掩码数组
        assert_(not ma.isMaskedArray(np.copy(mx)))


class TestAverage:

    def test_basic(self):
        # 创建一维数组作为测试用例
        y1 = np.array([1, 2, 3])
        # 验证数组的平均值，按指定轴（axis=0）应为 2.0
        assert_(average(y1, axis=0) == 2.)
        # 创建一维浮点数组作为测试用例
        y2 = np.array([1., 2., 3.])
        assert_(average(y2, axis=0) == 2.)
        # 创建一维全为 0 的数组作为测试用例
        y3 = [0., 0., 0.]
        assert_(average(y3, axis=0) == 0.)

        # 创建二维全为 1 的数组作为测试用例
        y4 = np.ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        # 验证按指定轴（axis=0）的平均值近似等于 average 函数计算的结果
        assert_almost_equal(y4.mean(0), average(y4, 0))
        # 验证按指定轴（axis=1）的平均值近似等于 average 函数计算的结果
        assert_almost_equal(y4.mean(1), average(y4, 1))

        # 创建随机生成的二维数组作为测试用例
        y5 = rand(5, 5)
        assert_almost_equal(y5.mean(0), average(y5, 0))
        assert_almost_equal(y5.mean(1), average(y5, 1))

    @pytest.mark.parametrize(
        'x, axis, expected_avg, weights, expected_wavg, expected_wsum',
        [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
         ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]],
          [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])],
    )
    # 定义一个测试方法，用于测试带有权重和返回加权和的平均值
    def test_basic_keepdims(self, x, axis, expected_avg,
                            weights, expected_wavg, expected_wsum):
        # 计算数组 x 指定轴向的平均值，并保持维度
        avg = np.average(x, axis=axis, keepdims=True)
        # 断言计算得到的平均值的形状与预期的形状相同
        assert avg.shape == np.shape(expected_avg)
        # 断言计算得到的平均值与预期的值相等
        assert_array_equal(avg, expected_avg)

        # 计算数组 x 指定轴向的加权平均值，并保持维度
        wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
        # 断言计算得到的加权平均值的形状与预期的形状相同
        assert wavg.shape == np.shape(expected_wavg)
        # 断言计算得到的加权平均值与预期的值相等
        assert_array_equal(wavg, expected_wavg)

        # 计算数组 x 指定轴向的加权平均值和加权和，并保持维度
        wavg, wsum = np.average(x, axis=axis, weights=weights, returned=True,
                                keepdims=True)
        # 断言计算得到的加权平均值的形状与预期的形状相同
        assert wavg.shape == np.shape(expected_wavg)
        # 断言计算得到的加权平均值与预期的值相等
        assert_array_equal(wavg, expected_wavg)
        # 断言计算得到的加权和的形状与预期的形状相同
        assert wsum.shape == np.shape(expected_wsum)
        # 断言计算得到的加权和与预期的值相等
        assert_array_equal(wsum, expected_wsum)

    # 测试不同权重的情况
    def test_weights(self):
        # 创建数组 y 和权重 w
        y = np.arange(10)
        w = np.arange(10)
        # 计算 y 的加权平均值
        actual = average(y, weights=w)
        # 计算预期的加权平均值
        desired = (np.arange(10) ** 2).sum() * 1. / np.arange(10).sum()
        # 断言计算得到的加权平均值与预期的值相近
        assert_almost_equal(actual, desired)

        # 创建二维数组 y1 和权重 w0
        y1 = np.array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        # 计算 y1 沿轴 0 的加权平均值
        actual = average(y1, weights=w0, axis=0)
        # 计算预期的加权平均值
        desired = np.array([3., 4., 5.])
        # 断言计算得到的加权平均值与预期的值相近
        assert_almost_equal(actual, desired)

        # 创建权重 w1
        w1 = [0, 0, 1]
        # 计算 y1 沿轴 1 的加权平均值
        actual = average(y1, weights=w1, axis=1)
        # 计算预期的加权平均值
        desired = np.array([3., 6.])
        # 断言计算得到的加权平均值与预期的值相近
        assert_almost_equal(actual, desired)

        # 测试当输入 y1 和权重 w1 的形状不同但未指定轴时的情况
        with pytest.raises(
                TypeError,
                match="Axis must be specified when shapes of a "
                      "and weights differ"):
            average(y1, weights=w1)

        # 创建二维权重 w2
        w2 = [[0, 0, 1], [0, 0, 2]]
        # 计算 y1 沿轴 1 的加权平均值
        desired = np.array([3., 6.])
        assert_array_equal(average(y1, weights=w2, axis=1), desired)
        # 计算 y1 的加权平均值
        assert_equal(average(y1, weights=w2), 5.)

        # 创建浮点数组 y3 和双精度浮点权重 w3
        y3 = rand(5).astype(np.float32)
        w3 = rand(5).astype(np.float64)
        # 断言 np.average 计算得到的结果数据类型与 y3 和 w3 的结果类型相同
        assert_(np.average(y3, weights=w3).dtype == np.result_type(y3, w3))

        # 测试 keepdims=False 和 keepdims=True 的权重情况
        x = np.array([2, 3, 4]).reshape(3, 1)
        w = np.array([4, 5, 6]).reshape(3, 1)

        # 计算 x 沿轴 1 的加权平均值，不保持维度
        actual = np.average(x, weights=w, axis=1, keepdims=False)
        # 计算预期的加权平均值
        desired = np.array([2., 3., 4.])
        # 断言计算得到的加权平均值与预期的值相等
        assert_array_equal(actual, desired)

        # 计算 x 沿轴 1 的加权平均值，保持维度
        actual = np.average(x, weights=w, axis=1, keepdims=True)
        # 计算预期的加权平均值
        desired = np.array([[2.], [3.], [4.]])
        # 断言计算得到的加权平均值与预期的值相等
        assert_array_equal(actual, desired)
    # 测试函数：验证在权重和输入维度不同的情况下的平均计算

    # 创建一个形状为 (2, 2, 3) 的 NumPy 数组 y，包含值 0 到 11
    y = np.arange(12).reshape(2, 2, 3)
    # 创建一个形状为 (2, 2, 3) 的 NumPy 数组 w，包含权重值
    w = np.array([0., 0., 1., .5, .5, 0., 0., .5, .5, 1., 0., 0.])\
        .reshape(2, 2, 3)

    # 从 w 中提取第一个轴的权重子数组
    subw0 = w[:, :, 0]
    # 在轴 (0, 1) 上计算 y 的加权平均值，使用 subw0 作为权重
    actual = average(y, axis=(0, 1), weights=subw0)
    # 期望的结果是 [7., 8., 9.]
    desired = np.array([7., 8., 9.])
    assert_almost_equal(actual, desired)

    # 从 w 中提取第二个轴的权重子数组
    subw1 = w[1, :, :]
    # 在轴 (1, 2) 上计算 y 的加权平均值，使用 subw1 作为权重
    actual = average(y, axis=(1, 2), weights=subw1)
    # 期望的结果是 [2.25, 8.25]
    desired = np.array([2.25, 8.25])
    assert_almost_equal(actual, desired)

    # 从 w 中提取第三个轴的权重子数组
    subw2 = w[:, 0, :]
    # 在轴 (0, 2) 上计算 y 的加权平均值，使用 subw2 作为权重
    actual = average(y, axis=(0, 2), weights=subw2)
    # 期望的结果是 [4.75, 7.75]
    desired = np.array([4.75, 7.75])
    assert_almost_equal(actual, desired)

    # 当权重的形状与指定轴的 a 数组形状不一致时，引发 ValueError 异常
    with pytest.raises(
            ValueError,
            match="Shape of weights must be consistent with "
                  "shape of a along specified axis"):
        average(y, axis=(0, 1, 2), weights=subw0)

    with pytest.raises(
            ValueError,
            match="Shape of weights must be consistent with "
                  "shape of a along specified axis"):
        average(y, axis=(0, 1), weights=subw1)

    # 交换轴应该等同于转置权重数组
    actual = average(y, axis=(1, 0), weights=subw0)
    desired = average(y, axis=(0, 1), weights=subw0.T)
    assert_almost_equal(actual, desired)

    # 如果在所有轴上进行平均，输出应为 float 类型
    actual = average(y, axis=(0, 1, 2), weights=w)
    assert_(actual.ndim == 0)


```    
    # 测试函数：验证返回值是否正确

    # 创建一个形状为 (2, 3) 的 NumPy 数组 y
    y = np.array([[1, 2, 3], [4, 5, 6]])

    # 测试在没有权重的情况下的 average 函数返回值
    avg, scl = average(y, returned=True)
    assert_equal(scl, 6.)

    # 测试在指定轴为 0 时，返回的缩放因子数组是否正确
    avg, scl = average(y, 0, returned=True)
    assert_array_equal(scl, np.array([2., 2., 2.]))

    # 测试在指定轴为 1 时，返回的缩放因子数组是否正确
    avg, scl = average(y, 1, returned=True)
    assert_array_equal(scl, np.array([3., 3.]))

    # 测试在有权重的情况下，指定轴为 0 时，返回的缩放因子数组是否正确
    w0 = [1, 2]
    avg, scl = average(y, weights=w0, axis=0, returned=True)
    assert_array_equal(scl, np.array([3., 3., 3.]))

    # 测试在有权重的情况下，指定轴为 1 时，返回的缩放因子数组是否正确
    w1 = [1, 2, 3]
    avg, scl = average(y, weights=w1, axis=1, returned=True)
    assert_array_equal(scl, np.array([6., 6.]))

    # 测试在有权重的情况下，指定轴为 1 时，返回的缩放因子数组是否正确
    w2 = [[0, 0, 1], [1, 2, 3]]
    avg, scl = average(y, weights=w2, axis=1, returned=True)
    assert_array_equal(scl, np.array([1., 6.]))



    # 测试函数：验证子类的情况下的 average 函数行为

    # 创建一个自定义子类的 NumPy 数组 a 和权重数组 w
    class subclass(np.ndarray):
        pass
    a = np.array([[1,2],[3,4]]).view(subclass)
    w = np.array([[1,2],[3,4]]).view(subclass)

    # 验证在没有权重的情况下，average 函数返回值的类型为子类 subclass
    assert_equal(type(np.average(a)), subclass)

    # 验证在有权重的情况下，average 函数返回值的类型为子类 subclass
    assert_equal(type(np.average(a, weights=w)), subclass)
    # 定义一个测试方法，用于测试数据类型的提升（upcasting）
    def test_upcasting(self):
        # 定义多组数据类型元组
        typs = [('i4', 'i4', 'f8'), ('i4', 'f4', 'f8'), ('f4', 'i4', 'f8'),
                 ('f4', 'f4', 'f4'), ('f4', 'f8', 'f8')]
        # 遍历每组数据类型元组
        for at, wt, rt in typs:
            # 创建具有给定数据类型的数组 a 和 w
            a = np.array([[1,2],[3,4]], dtype=at)
            w = np.array([[1,2],[3,4]], dtype=wt)
            # 断言加权平均值的结果数据类型与期望的数据类型相同
            assert_equal(np.average(a, weights=w).dtype, np.dtype(rt))

    # 定义一个测试方法，用于测试对象数据类型（object dtype）
    def test_object_dtype(self):
        # 创建包含 Decimal 对象的数组 a
        a = np.array([decimal.Decimal(x) for x in range(10)])
        # 创建包含 Decimal 对象的数组 w，并对其进行归一化处理
        w = np.array([decimal.Decimal(1) for _ in range(10)])
        w /= w.sum()
        # 断言数组 a 按权重 w 计算的均值接近于期望的均值
        assert_almost_equal(a.mean(0), average(a, weights=w))

    # 定义一个测试方法，用于测试不带数据类型的 average 函数
    def test_average_class_without_dtype(self):
        # 查看 GitHub 问题号 #21988
        # 创建包含 Fraction 对象的数组 a
        a = np.array([Fraction(1, 5), Fraction(3, 5)])
        # 断言数组 a 的平均值等于期望的分数值
        assert_equal(np.average(a), Fraction(2, 5))
# 定义一个测试类 TestSelect，用于测试 select 函数的不同用例
class TestSelect:
    # 静态变量 choices，包含三个 NumPy 数组作为选择项
    choices = [np.array([1, 2, 3]),
               np.array([4, 5, 6]),
               np.array([7, 8, 9])]
    # 静态变量 conditions，包含三个 NumPy 数组作为条件
    conditions = [np.array([False, False, False]),
                  np.array([False, True, False]),
                  np.array([False, False, True])]

    # 内部方法 _select，根据条件从给定的值列表中选择值
    def _select(self, cond, values, default=0):
        # 初始化输出列表
        output = []
        # 遍历条件的长度
        for m in range(len(cond)):
            # 根据条件选择对应的值，如果条件满足则选择对应的值列表中的值，否则选择默认值
            output += [V[m] for V, C in zip(values, cond) if C[m]] or [default]
        return output

    # 测试基本用例的方法 test_basic
    def test_basic(self):
        # 复制静态变量 choices 和 conditions
        choices = self.choices
        conditions = self.conditions
        # 断言 select 函数对给定的条件和选择项返回的结果与 _select 方法相同
        assert_array_equal(select(conditions, choices, default=15),
                           self._select(conditions, choices, default=15))
        # 断言 choices 的长度为 3
        assert_equal(len(choices), 3)
        # 断言 conditions 的长度为 3
        assert_equal(len(conditions), 3)

    # 测试广播功能的方法 test_broadcasting
    def test_broadcasting(self):
        # 定义包含广播条件的列表
        conditions = [np.array(True), np.array([False, True, False])]
        # 定义包含广播选择项的列表
        choices = [1, np.arange(12).reshape(4, 3)]
        # 断言 select 函数对给定的条件和选择项返回的结果与预期的全为 1 的数组相同
        assert_array_equal(select(conditions, choices), np.ones((4, 3)))
        # 断言 select 函数对包含默认值广播的条件和选择项返回的结果的形状为 (1,)
        assert_equal(select([True], [0], default=[0]).shape, (1,))

    # 测试返回数据类型的方法 test_return_dtype
    def test_return_dtype(self):
        # 断言 select 函数对给定的条件和选择项以及复数默认值返回的结果数据类型为 np.complex128
        assert_equal(select(self.conditions, self.choices, 1j).dtype,
                     np.complex128)
        # 断言如果选择项是 int8 类型的，select 函数对给定的条件返回的结果数据类型为 np.int8
        choices = [choice.astype(np.int8) for choice in self.choices]
        assert_equal(select(self.conditions, choices).dtype, np.int8)

        # 创建包含 NaN 的数组 d
        d = np.array([1, 2, 3, np.nan, 5, 7])
        # 创建标志数组 m，指示 d 中的 NaN 值
        m = np.isnan(d)
        # 断言 select 函数对包含标志数组 m 和数组 d 返回的结果与预期的结果列表相同
        assert_equal(select([m], [d]), [0, 0, 0, np.nan, 0, 0])

    # 测试弃用空列表参数的方法 test_deprecated_empty
    def test_deprecated_empty(self):
        # 断言 select 函数对空的条件和选择项列表抛出 ValueError 异常
        assert_raises(ValueError, select, [], [], 3j)
        assert_raises(ValueError, select, [], [])

    # 测试非布尔类型条件的方法 test_non_bool_deprecation
    def test_non_bool_deprecation(self):
        # 复制静态变量 choices 和 conditions，并将第一个条件转换为 int_ 类型
        choices = self.choices
        conditions = self.conditions[:]
        conditions[0] = conditions[0].astype(np.int_)
        # 断言 select 函数对给定的非布尔类型条件和选择项抛出 TypeError 异常
        assert_raises(TypeError, select, conditions, choices)
        # 将第一个条件再次转换为 uint8 类型
        conditions[0] = conditions[0].astype(np.uint8)
        # 断言 select 函数对给定的非布尔类型条件和选择项抛出 TypeError 异常
        assert_raises(TypeError, select, conditions, choices)
        # 断言 select 函数对给定的非布尔类型条件和选择项抛出 TypeError 异常
        assert_raises(TypeError, select, conditions, choices)

    # 测试多个参数的方法 test_many_arguments
    def test_many_arguments(self):
        # 创建包含 100 个相同值为 False 的条件列表
        conditions = [np.array([False])] * 100
        # 创建包含 100 个相同值为 1 的选择项列表
        choices = [np.array([1])] * 100
        # 调用 select 函数对给定的条件和选择项列表
        select(conditions, choices)


class TestInsert:
    # 定义测试方法，用于测试 insert 函数的各种用法和场景
    def test_basic(self):
        # 创建一个列表 a
        a = [1, 2, 3]
        # 测试在索引 0 处插入值 1
        assert_equal(insert(a, 0, 1), [1, 1, 2, 3])
        # 测试在索引 3 处插入值 1
        assert_equal(insert(a, 3, 1), [1, 2, 3, 1])
        # 测试在索引 [1, 1, 1] 处分别插入列表 [1, 2, 3] 的元素
        assert_equal(insert(a, [1, 1, 1], [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        # 测试在索引 1 处插入列表 [1, 2, 3]
        assert_equal(insert(a, 1, [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        # 测试在索引 [1, -1, 3] 处分别插入值 9
        assert_equal(insert(a, [1, -1, 3], 9), [1, 9, 2, 9, 3, 9])
        # 测试在逆序索引处插入值 9
        assert_equal(insert(a, slice(-1, None, -1), 9), [9, 1, 9, 2, 9, 3])
        # 测试在索引 [-1, 1, 3] 处分别插入列表 [7, 8, 9] 的元素
        assert_equal(insert(a, [-1, 1, 3], [7, 8, 9]), [1, 8, 2, 7, 3, 9])

        # 创建一个 numpy 数组 b，元素为 [0, 1]，数据类型为 np.float64
        b = np.array([0, 1], dtype=np.float64)
        # 测试在索引 0 处插入 b[0] 的值
        assert_equal(insert(b, 0, b[0]), [0., 0., 1.])
        # 测试在空索引处插入空列表
        assert_equal(insert(b, [], []), b)
        
        # 未来处理布尔值的方式可能不同：
        # 使用警告捕获来测试插入布尔值的行为
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            assert_equal(
                insert(a, np.array([True] * 4), 9), [1, 9, 9, 9, 9, 2, 3])
            assert_(w[0].category is FutureWarning)

    # 定义多维测试方法，用于测试 insert 函数在多维数组上的操作
    def test_multidim(self):
        # 创建一个二维列表 a，内容为 [[1, 1, 1]]
        a = [[1, 1, 1]]
        # 预期插入后的结果列表 r
        r = [[2, 2, 2],
             [1, 1, 1]]
        # 测试在索引 0 处插入值 [1]
        assert_equal(insert(a, 0, [1]), [1, 1, 1, 1])
        # 测试在索引 0 处沿轴 0 插入值 [2, 2, 2]
        assert_equal(insert(a, 0, [2, 2, 2], axis=0), r)
        # 测试在索引 0 处沿轴 0 插入值 2
        assert_equal(insert(a, 0, 2, axis=0), r)
        # 测试在索引 2 处沿轴 1 插入值 2
        assert_equal(insert(a, 2, 2, axis=1), [[1, 1, 2, 1]])

        # 创建一个二维 numpy 数组 a
        a = np.array([[1, 1], [2, 2], [3, 3]])
        # 创建预期结果数组 b
        b = np.arange(1, 4).repeat(3).reshape(3, 3)
        # 创建预期结果数组 c
        c = np.concatenate(
            (a[:, 0:1], np.arange(1, 4).repeat(3).reshape(3, 3).T,
             a[:, 1:2]), axis=1)
        # 测试在索引 [1] 处沿轴 1 插入值 [[1], [2], [3]]
        assert_equal(insert(a, [1], [[1], [2], [3]], axis=1), b)
        # 测试在索引 [1] 处沿轴 1 插入值 [1, 2, 3]
        assert_equal(insert(a, [1], [1, 2, 3], axis=1), c)
        # 标量在此处的行为与上述相反：
        # 测试在索引 1 处沿轴 1 插入值 [1, 2, 3]
        assert_equal(insert(a, 1, [1, 2, 3], axis=1), b)
        # 测试在索引 1 处沿轴 1 插入值 [[1], [2], [3]]
        assert_equal(insert(a, 1, [[1], [2], [3]], axis=1), c)

        # 创建一个二维 numpy 数组 a
        a = np.arange(4).reshape(2, 2)
        # 测试在索引 1 处沿轴 1 插入 a[:, 1] 的值
        assert_equal(insert(a[:, :1], 1, a[:, 1], axis=1), a)
        # 测试在索引 1 处沿轴 0 插入 a[1, :] 的值
        assert_equal(insert(a[:1, :], 1, a[1, :], axis=0), a)

        # 负数轴值
        a = np.arange(24).reshape((2, 3, 4))
        # 测试在索引 1 处沿负轴 -1 插入 a[:, :, 3] 的值
        assert_equal(insert(a, 1, a[:, :, 3], axis=-1),
                     insert(a, 1, a[:, :, 3], axis=2))
        # 测试在索引 1 处沿负轴 -2 插入 a[:, 2, :] 的值
        assert_equal(insert(a, 1, a[:, 2, :], axis=-2),
                     insert(a, 1, a[:, 2, :], axis=1))

        # 无效的轴值
        assert_raises(AxisError, insert, a, 1, a[:, 2, :], axis=3)
        assert_raises(AxisError, insert, a, 1, a[:, 2, :], axis=-4)

        # 负数轴值
        a = np.arange(24).reshape((2, 3, 4))
        # 测试在索引 1 处沿负轴 -1 插入 a[:, :, 3] 的值
        assert_equal(insert(a, 1, a[:, :, 3], axis=-1),
                     insert(a, 1, a[:, :, 3], axis=2))
        # 测试在索引 1 处沿负轴 -2 插入 a[:, 2, :] 的值
        assert_equal(insert(a, 1, a[:, 2, :], axis=-2),
                     insert(a, 1, a[:, 2, :], axis=1))
    # 定义测试方法，验证插入操作对于标量数组的异常处理
    def test_0d(self):
        # 创建一个标量数组 a
        a = np.array(1)
        # 验证在指定轴向插入空列表时是否引发 AxisError 异常
        with pytest.raises(AxisError):
            insert(a, [], 2, axis=0)
        # 验证在指定轴向插入非法类型时是否引发 TypeError 异常
        with pytest.raises(TypeError):
            insert(a, [], 2, axis="nonsense")

    # 定义测试方法，验证插入操作对于子类数组的处理
    def test_subclass(self):
        # 定义一个继承自 np.ndarray 的子类 SubClass
        class SubClass(np.ndarray):
            pass
        # 创建一个 SubClass 实例 a，视图包含从 0 到 9 的整数
        a = np.arange(10).view(SubClass)
        # 验证插入操作返回的结果是否为 SubClass 类型
        assert_(isinstance(np.insert(a, 0, [0]), SubClass))
        assert_(isinstance(np.insert(a, [], []), SubClass))
        assert_(isinstance(np.insert(a, [0, 1], [1, 2]), SubClass))
        assert_(isinstance(np.insert(a, slice(1, 2), [1, 2]), SubClass))
        assert_(isinstance(np.insert(a, slice(1, -2, -1), []), SubClass))
        # 在未来的某个版本中，这会导致错误：
        # 创建一个标量数组 a，将其视图转换为 SubClass 类型
        a = np.array(1).view(SubClass)
        # 验证插入操作返回的结果是否为 SubClass 类型
        assert_(isinstance(np.insert(a, 0, [0]), SubClass))

    # 定义测试方法，验证插入操作对于索引数组的处理
    def test_index_array_copied(self):
        # 创建一个数组 x 包含三个值为 1 的元素
        x = np.array([1, 1, 1])
        # 执行插入操作，不对 x 进行修改
        np.insert([0, 1, 2], x, [3, 4, 5])
        # 验证 x 的值未被修改
        assert_equal(x, np.array([1, 1, 1]))

    # 定义测试方法，验证插入操作对于结构化数组的处理
    def test_structured_array(self):
        # 创建一个结构化数组 a，包含两个字段：foo (整数) 和 bar (字符串，长度为 1)
        a = np.array([(1, 'a'), (2, 'b'), (3, 'c')],
                     dtype=[('foo', 'i'), ('bar', 'S1')])
        # 定义要插入的值 val
        val = (4, 'd')
        # 在索引 0 处插入值 val，验证插入后的第一个元素是否符合预期
        b = np.insert(a, 0, val)
        assert_array_equal(b[0], np.array(val, dtype=b.dtype))
        # 定义要插入的值列表 val
        val = [(4, 'd')] * 2
        # 在索引 0 和 2 处插入值 val，验证插入后的第一个和第四个元素是否符合预期
        b = np.insert(a, [0, 2], val)
        assert_array_equal(b[[0, 3]], np.array(val, dtype=b.dtype))

    # 定义测试方法，验证插入操作对于浮点数索引的异常处理
    def test_index_floats(self):
        # 验证在插入操作中使用浮点数索引时是否引发 IndexError 异常
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([1.0, 2.0]), [10, 20])
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([], dtype=float), [])

    # 使用 pytest 的参数化装饰器，定义测试方法，验证插入操作对于超出索引范围的处理
    @pytest.mark.parametrize('idx', [4, -4])
    def test_index_out_of_bounds(self, idx):
        # 验证在插入操作中使用超出索引范围的索引时是否引发 IndexError 异常，且异常信息匹配 'out of bounds'
        with pytest.raises(IndexError, match='out of bounds'):
            np.insert([0, 1, 2], [idx], [3, 4])
class TestAmax:

    def test_basic(self):
        # 定义列表 a 包含一组数字
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 断言 np.amax(a) 的结果等于 10.0
        assert_equal(np.amax(a), 10.0)
        
        # 定义二维列表 b
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        # 断言 np.amax(b, axis=0) 的结果等于 [8.0, 10.0, 9.0]
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        # 断言 np.amax(b, axis=1) 的结果等于 [9.0, 10.0, 8.0]
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])


class TestAmin:

    def test_basic(self):
        # 定义列表 a 包含一组数字
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 断言 np.amin(a) 的结果等于 -5.0
        assert_equal(np.amin(a), -5.0)
        
        # 定义二维列表 b
        b = [[3, 6.0, 9.0],
             [4, 10.0, 5.0],
             [8, 3.0, 2.0]]
        # 断言 np.amin(b, axis=0) 的结果等于 [3.0, 3.0, 2.0]
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        # 断言 np.amin(b, axis=1) 的结果等于 [3.0, 4.0, 2.0]
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])


class TestPtp:

    def test_basic(self):
        # 创建 np.array a
        a = np.array([3, 4, 5, 10, -3, -5, 6.0])
        # 断言 np.ptp(a, axis=0) 的结果等于 15.0
        assert_equal(np.ptp(a, axis=0), 15.0)
        
        # 创建 np.array b
        b = np.array([[3, 6.0, 9.0],
                      [4, 10.0, 5.0],
                      [8, 3.0, 2.0]])
        # 断言 np.ptp(b, axis=0) 的结果等于 [5.0, 7.0, 7.0]
        assert_equal(np.ptp(b, axis=0), [5.0, 7.0, 7.0])
        # 断言 np.ptp(b, axis=-1) 的结果等于 [6.0, 6.0, 6.0]
        assert_equal(np.ptp(b, axis=-1), [6.0, 6.0, 6.0])

        # 断言 np.ptp(b, axis=0, keepdims=True) 的结果等于 [[5.0, 7.0, 7.0]]
        assert_equal(np.ptp(b, axis=0, keepdims=True), [[5.0, 7.0, 7.0]])
        # 断言 np.ptp(b, axis=(0, 1), keepdims=True) 的结果等于 [[8.0]]
        assert_equal(np.ptp(b, axis=(0, 1), keepdims=True), [[8.0]])


class TestCumsum:

    def test_basic(self):
        # 定义列表 ba 和二维列表 ba2
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        
        # 遍历不同的 np 数据类型
        for ctype in [np.int8, np.uint8, np.int16, np.uint16, np.int32,
                      np.uint32, np.float32, np.float64, np.complex64,
                      np.complex128]:
            # 根据 ctype 创建 np.array a 和 a2
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)

            # 定义目标数组 tgt
            tgt = np.array([1, 3, 13, 24, 30, 35, 39], ctype)
            # 断言 np.cumsum(a, axis=0) 的结果等于 tgt
            assert_array_equal(np.cumsum(a, axis=0), tgt)

            tgt = np.array(
                [[1, 2, 3, 4], [6, 8, 10, 13], [16, 11, 14, 18]], ctype)
            # 断言 np.cumsum(a2, axis=0) 的结果等于 tgt
            assert_array_equal(np.cumsum(a2, axis=0), tgt)

            tgt = np.array(
                [[1, 3, 6, 10], [5, 11, 18, 27], [10, 13, 17, 22]], ctype)
            # 断言 np.cumsum(a2, axis=1) 的结果等于 tgt
            assert_array_equal(np.cumsum(a2, axis=1), tgt)


class TestProd:

    def test_basic(self):
        # 定义列表 ba 和二维列表 ba2
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        
        # 遍历不同的 np 数据类型
        for ctype in [np.int16, np.uint16, np.int32, np.uint32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            # 根据 ctype 创建 np.array a 和 a2
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            
            # 对于某些类型抛出异常
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, np.prod, a)
                assert_raises(ArithmeticError, np.prod, a2, 1)
            else:
                # 断言 a.prod(axis=0) 的结果等于 26400
                assert_equal(a.prod(axis=0), 26400)
                # 断言 a2.prod(axis=0) 的结果等于 [50, 36, 84, 180]
                assert_array_equal(a2.prod(axis=0),
                                   np.array([50, 36, 84, 180], ctype))
                # 断言 a2.prod(axis=-1) 的结果等于 [24, 1890, 600]
                assert_array_equal(a2.prod(axis=-1),
                                   np.array([24, 1890, 600], ctype))


class TestCumprod:

    def test_basic(self):
        pass  # 未完成的测试类，暂无代码
    # 定义一个测试方法，用于测试基本功能
    def test_basic(self):
        # 创建一个包含整数和浮点数的列表
        ba = [1, 2, 10, 11, 6, 5, 4]
        # 创建一个包含多个子列表的二维列表
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        # 遍历不同的 NumPy 数据类型
        for ctype in [np.int16, np.uint16, np.int32, np.uint32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            # 使用指定的数据类型创建 NumPy 数组
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            # 如果数据类型在指定的列表中，则触发算术错误异常
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, np.cumprod, a)
                assert_raises(ArithmeticError, np.cumprod, a2, 1)
                assert_raises(ArithmeticError, np.cumprod, a)
            else:
                # 对数组进行累积乘积操作，并比较结果
                assert_array_equal(np.cumprod(a, axis=-1),
                                   np.array([1, 2, 20, 220,
                                             1320, 6600, 26400], ctype))
                assert_array_equal(np.cumprod(a2, axis=0),
                                   np.array([[1, 2, 3, 4],
                                             [5, 12, 21, 36],
                                             [50, 36, 84, 180]], ctype))
                assert_array_equal(np.cumprod(a2, axis=-1),
                                   np.array([[1, 2, 6, 24],
                                             [5, 30, 210, 1890],
                                             [10, 30, 120, 600]], ctype))
class TestDiff:

    # 测试基本的diff函数
    def test_basic(self):
        x = [1, 4, 6, 7, 12]
        out = np.array([3, 2, 1, 5])
        out2 = np.array([-1, -1, 4])
        out3 = np.array([0, 5])
        assert_array_equal(diff(x), out)  # 检查默认参数下的差分结果
        assert_array_equal(diff(x, n=2), out2)  # 检查n=2时的差分结果
        assert_array_equal(diff(x, n=3), out3)  # 检查n=3时的差分结果

        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        out = np.array([1.1, 0.8, -3.2, 0.1])
        assert_almost_equal(diff(x), out)  # 检查浮点数列表的差分结果

        x = [True, True, False, False]
        out = np.array([False, True, False])
        out2 = np.array([True, True])
        assert_array_equal(diff(x), out)  # 检查布尔列表的差分结果
        assert_array_equal(diff(x, n=2), out2)  # 检查布尔列表在n=2时的差分结果

    # 测试指定轴向的diff函数
    def test_axis(self):
        x = np.zeros((10, 20, 30))
        x[:, 1::2, :] = 1
        exp = np.ones((10, 19, 30))
        exp[:, 1::2, :] = -1
        assert_array_equal(diff(x), np.zeros((10, 20, 29)))  # 检查默认轴向下的差分结果
        assert_array_equal(diff(x, axis=-1), np.zeros((10, 20, 29)))  # 检查指定轴向为-1的差分结果
        assert_array_equal(diff(x, axis=0), np.zeros((9, 20, 30)))  # 检查指定轴向为0的差分结果
        assert_array_equal(diff(x, axis=1), exp)  # 检查指定轴向为1的差分结果
        assert_array_equal(diff(x, axis=-2), exp)  # 检查指定轴向为-2的差分结果
        assert_raises(AxisError, diff, x, axis=3)  # 检查超出范围的轴向抛出异常
        assert_raises(AxisError, diff, x, axis=-4)  # 检查超出范围的负轴向抛出异常

        x = np.array(1.11111111111, np.float64)
        assert_raises(ValueError, diff, x)  # 检查输入为单个浮点数时抛出异常

    # 测试多维数组的diff函数
    def test_nd(self):
        x = 20 * rand(10, 20, 30)
        out1 = x[:, :, 1:] - x[:, :, :-1]
        out2 = out1[:, :, 1:] - out1[:, :, :-1]
        out3 = x[1:, :, :] - x[:-1, :, :]
        out4 = out3[1:, :, :] - out3[:-1, :, :]
        assert_array_equal(diff(x), out1)  # 检查默认参数下的多维差分结果
        assert_array_equal(diff(x, n=2), out2)  # 检查n=2时的多维差分结果
        assert_array_equal(diff(x, axis=0), out3)  # 检查指定轴向为0的多维差分结果
        assert_array_equal(diff(x, n=2, axis=0), out4)  # 检查指定轴向为0且n=2的多维差分结果

    # 测试n参数的diff函数
    def test_n(self):
        x = list(range(3))
        assert_raises(ValueError, diff, x, n=-1)  # 检查n为负数时抛出异常
        output = [diff(x, n=n) for n in range(1, 5)]
        expected = [[1, 1], [0], [], []]
        assert_(diff(x, n=0) is x)  # 检查n=0时返回原始列表
        for n, (expected, out) in enumerate(zip(expected, output), start=1):
            assert_(type(out) is np.ndarray)  # 检查输出为NumPy数组
            assert_array_equal(out, expected)  # 检查不同n值下的差分结果
            assert_equal(out.dtype, np.int_)  # 检查输出数组的数据类型为整数
            assert_equal(len(out), max(0, len(x) - n))  # 检查输出数组的长度

    # 测试日期时间对象的diff函数
    def test_times(self):
        x = np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64)
        expected = [
            np.array([1, 1], dtype='timedelta64[D]'),
            np.array([0], dtype='timedelta64[D]'),
        ]
        expected.extend([np.array([], dtype='timedelta64[D]')] * 3)
        for n, exp in enumerate(expected, start=1):
            out = diff(x, n=n)
            assert_array_equal(out, exp)  # 检查日期时间对象的差分结果
            assert_equal(out.dtype, exp.dtype)  # 检查输出数组的数据类型
    # 定义一个测试方法，用于测试差分函数在子类中的行为
    def test_subclass(self):
        # 创建一个结构化数组 x，包含数据和掩码
        x = ma.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                     mask=[[False, False], [True, False],
                           [False, True], [True, True], [False, False]])
        # 调用 diff 函数计算 x 的差分结果
        out = diff(x)
        # 断言差分后的数据部分与预期一致
        assert_array_equal(out.data, [[1], [1], [1], [1], [1]])
        # 断言差分后的掩码部分与预期一致
        assert_array_equal(out.mask, [[False], [True], [True], [True], [False]])
        # 断言差分后的结果类型与原数组类型一致
        assert_(type(out) is type(x))

        # 对于 n=3 的差分操作
        out3 = diff(x, n=3)
        # 断言差分后的数据部分为空
        assert_array_equal(out3.data, [[], [], [], [], []])
        # 断言差分后的掩码部分为空
        assert_array_equal(out3.mask, [[], [], [], [], []])
        # 断言差分后的结果类型与原数组类型一致
        assert_(type(out3) is type(x))

    # 定义一个测试方法，用于测试在数组前添加元素时的差分函数行为
    def test_prepend(self):
        # 创建一个简单的数组 x
        x = np.arange(5) + 1
        # 断言在数组前添加 0 后的差分结果与全为 1 的数组一致
        assert_array_equal(diff(x, prepend=0), np.ones(5))
        # 断言在数组前添加 [0] 后的差分结果与全为 1 的数组一致
        assert_array_equal(diff(x, prepend=[0]), np.ones(5))
        # 断言对差分结果进行累积求和后应该得到原数组 x
        assert_array_equal(np.cumsum(np.diff(x, prepend=0)), x)
        # 断言在数组前添加 [-1, 0] 后的差分结果与全为 1 的数组一致
        assert_array_equal(diff(x, prepend=[-1, 0]), np.ones(6))

        # 创建一个二维数组 x
        x = np.arange(4).reshape(2, 2)
        # 在第二维度上添加 0 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=1, prepend=0)
        expected = [[0, 1], [2, 1]]
        assert_array_equal(result, expected)
        # 在第二维度上添加 [[0], [0]] 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=1, prepend=[[0], [0]])
        assert_array_equal(result, expected)

        # 在第一维度上添加 0 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=0, prepend=0)
        expected = [[0, 1], [2, 2]]
        assert_array_equal(result, expected)
        # 在第一维度上添加 [[0, 0]] 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=0, prepend=[[0, 0]])
        assert_array_equal(result, expected)

        # 当前场景下不支持指定形状的 prepend 值，应抛出 ValueError 异常
        assert_raises(ValueError, np.diff, x, prepend=np.zeros((3,3)))

        # 在超出数组维度的轴上添加 0，应抛出 AxisError 异常
        assert_raises(AxisError, diff, x, prepend=0, axis=3)

    # 定义一个测试方法，用于测试在数组末尾添加元素时的差分函数行为
    def test_append(self):
        # 创建一个简单的数组 x
        x = np.arange(5)
        # 在数组末尾添加 0 后的差分结果与预期的一维数组一致
        result = diff(x, append=0)
        expected = [1, 1, 1, 1, -4]
        assert_array_equal(result, expected)
        # 在数组末尾添加 [0] 后的差分结果与预期的一维数组一致
        result = diff(x, append=[0])
        assert_array_equal(result, expected)
        # 在数组末尾添加 [0, 2] 后的差分结果与预期的一维数组一致
        result = diff(x, append=[0, 2])
        expected = expected + [2]
        assert_array_equal(result, expected)

        # 创建一个二维数组 x
        x = np.arange(4).reshape(2, 2)
        # 在第二维度上添加 0 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=1, append=0)
        expected = [[1, -1], [1, -3]]
        assert_array_equal(result, expected)
        # 在第二维度上添加 [[0], [0]] 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=1, append=[[0], [0]])
        assert_array_equal(result, expected)

        # 在第一维度上添加 0 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=0, append=0)
        expected = [[2, 2], [-2, -3]]
        assert_array_equal(result, expected)
        # 在第一维度上添加 [[0, 0]] 后的差分结果与预期的二维数组一致
        result = np.diff(x, axis=0, append=[[0, 0]])
        assert_array_equal(result, expected)

        # 当前场景下不支持指定形状的 append 值，应抛出 ValueError 异常
        assert_raises(ValueError, np.diff, x, append=np.zeros((3,3)))

        # 在超出数组维度的轴上添加 0，应抛出 AxisError 异常
        assert_raises(AxisError, diff, x, append=0, axis=3)
    # 定义一个测试类 TestDelete，用于测试 numpy 的 delete 函数
class TestDelete:

    # 设置测试方法的初始化操作
    def setup_method(self):
        # 创建一个长度为 5 的 NumPy 数组 a，包含 [0, 1, 2, 3, 4]
        self.a = np.arange(5)
        # 创建一个形状为 (1, 5, 2) 的 NumPy 数组 nd_a，包含 [0, 0], [1, 1], ..., [4, 4]
        self.nd_a = np.arange(5).repeat(2).reshape(1, 5, 2)

    # 定义一个内部方法，用于检查切片操作的逆操作
    def _check_inverse_of_slicing(self, indices):
        # 删除数组 self.a 中指定索引处的元素，生成新的数组 a_del
        a_del = delete(self.a, indices)
        # 按指定轴删除数组 self.nd_a 中的元素，生成新的数组 nd_a_del
        nd_a_del = delete(self.nd_a, indices, axis=1)
        # 构建错误信息消息字符串
        msg = 'Delete failed for obj: %r' % indices
        # 断言删除元素后的 a_del 与 self.a 的对称差集应该等于原始 self.a
        assert_array_equal(setxor1d(a_del, self.a[indices, ]), self.a,
                           err_msg=msg)
        # 计算删除元素后 nd_a_del 的第一个子数组的指定列的对称差集，应该等于原始 self.nd_a 中相应的列
        xor = setxor1d(nd_a_del[0,:, 0], self.nd_a[0, indices, 0])
        assert_array_equal(xor, self.nd_a[0,:, 0], err_msg=msg)

    # 定义测试切片操作的方法
    def test_slices(self):
        # 定义不同的起始、结束和步长组合
        lims = [-6, -2, 0, 1, 2, 4, 5]
        steps = [-3, -1, 1, 3]
        # 遍历所有起始、结束和步长的组合
        for start in lims:
            for stop in lims:
                for step in steps:
                    # 创建切片对象 s
                    s = slice(start, stop, step)
                    # 调用内部方法，检查切片操作的逆操作
                    self._check_inverse_of_slicing(s)

    # 定义测试 fancy 索引操作的方法
    def test_fancy(self):
        # 调用内部方法，检查 fancy 索引操作的逆操作，传入数组索引
        self._check_inverse_of_slicing(np.array([[0, 1], [2, 1]]))
        # 使用 pytest 检查超出范围的索引是否会引发 IndexError
        with pytest.raises(IndexError):
            delete(self.a, [100])
        with pytest.raises(IndexError):
            delete(self.a, [-100])

        # 调用内部方法，检查 fancy 索引操作的逆操作，传入列表索引
        self._check_inverse_of_slicing([0, -1, 2, 2])

        # 调用内部方法，检查 fancy 索引操作的逆操作，传入布尔数组索引
        self._check_inverse_of_slicing([True, False, False, True, False])

        # 使用 pytest 检查非法的索引类型是否会引发 ValueError
        with pytest.raises(ValueError):
            delete(self.a, True)
        with pytest.raises(ValueError):
            delete(self.a, False)

        # 使用 pytest 检查索引项不足时是否会引发 ValueError
        with pytest.raises(ValueError):
            delete(self.a, [False]*4)

    # 定义测试单一索引操作的方法
    def test_single(self):
        # 调用内部方法，检查单一索引操作的逆操作
        self._check_inverse_of_slicing(0)
        self._check_inverse_of_slicing(-4)

    # 定义测试 0 维数组操作的方法
    def test_0d(self):
        # 创建一个 0 维 NumPy 数组 a
        a = np.array(1)
        # 使用 pytest 检查删除 0 维数组的轴时是否会引发 AxisError
        with pytest.raises(AxisError):
            delete(a, [], axis=0)
        # 使用 pytest 检查使用无效轴类型时是否会引发 TypeError
        with pytest.raises(TypeError):
            delete(a, [], axis="nonsense")

    # 定义测试子类操作的方法
    def test_subclass(self):
        # 定义一个名为 SubClass 的子类，继承自 np.ndarray
        class SubClass(np.ndarray):
            pass
        # 将 self.a 转换为 SubClass 类型的对象 a
        a = self.a.view(SubClass)
        # 使用 isinstance 断言删除操作后返回的对象是否仍为 SubClass 类型
        assert_(isinstance(delete(a, 0), SubClass))
        assert_(isinstance(delete(a, []), SubClass))
        assert_(isinstance(delete(a, [0, 1]), SubClass))
        assert_(isinstance(delete(a, slice(1, 2)), SubClass))
        assert_(isinstance(delete(a, slice(1, -2)), SubClass))

    # 定义测试数组顺序保留操作的方法
    def test_array_order_preserve(self):
        # 创建一个 Fortran 排序的 2x5 数组 k
        k = np.arange(10).reshape(2, 5, order='F')
        # 删除 k 中指定的切片范围，生成新的数组 m
        m = delete(k, slice(60, None), axis=1)

        # 'k' 是 Fortran 排序的，'m' 应该与 'k' 具有相同的顺序，并且不应该变成 C 排序
        assert_equal(m.flags.c_contiguous, k.flags.c_contiguous)
        assert_equal(m.flags.f_contiguous, k.flags.f_contiguous)

    # 定义测试索引浮点数的方法
    def test_index_floats(self):
        # 使用 pytest 检查使用浮点数索引是否会引发 IndexError
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([], dtype=float))

    # 使用 pytest 的 parametrize 标记，定义参数化测试
    @pytest.mark.parametrize("indexer", [np.array([1]), [1]])
    # 定义一个测试方法，用于测试从数组中删除单个元素的情况
    def test_single_item_array(self, indexer):
        # 删除数组 `self.a` 中索引为 1 的元素，并赋值给 `a_del_int`
        a_del_int = delete(self.a, 1)
        # 删除数组 `self.a` 中使用 `indexer` 指定的索引或索引数组的元素，并赋值给 `a_del`
        a_del = delete(self.a, indexer)
        # 断言删除后的结果 `a_del_int` 应与 `a_del` 相等
        assert_equal(a_del_int, a_del)

        # 删除多维数组 `self.nd_a` 中第二列（axis=1）的元素，并赋值给 `nd_a_del_int`
        nd_a_del_int = delete(self.nd_a, 1, axis=1)
        # 删除多维数组 `self.nd_a` 中第一个列（axis=1）的元素，并赋值给 `nd_a_del`
        nd_a_del = delete(self.nd_a, np.array([1]), axis=1)
        # 断言删除后的结果 `nd_a_del_int` 应与 `nd_a_del` 相等
        assert_equal(nd_a_del_int, nd_a_del)

    # 定义另一个测试方法，用于测试在整数数组中特殊处理不影响非整数数组的情况
    def test_single_item_array_non_int(self):
        # 如果 `False` 被转换为 `0`，则会删除元素，这里确保不会影响元素
        # 删除整数数组 `np.ones(1)` 中使用整数数组 `np.array([False])` 指定的索引，结果赋值给 `res`
        res = delete(np.ones(1), np.array([False]))
        # 断言删除后的结果 `res` 应与 `np.ones(1)` 相等
        assert_array_equal(res, np.ones(1))

        # 从二维数组 `np.ones((3, 1))` 中按指定的布尔掩码 `false_mask` 删除元素，赋值给 `res`
        res = delete(x, false_mask, axis=-1)
        # 断言删除后的结果 `res` 应与原数组 `x` 相等
        assert_array_equal(res, x)

        # 从二维数组 `np.ones((3, 1))` 中按指定的布尔掩码 `true_mask` 删除元素，赋值给 `res`
        res = delete(x, true_mask, axis=-1)
        # 断言删除后的结果 `res` 应为空（删除了所有元素）
        assert_array_equal(res, x[:, :0])

        # 对象或例如时间增量不应被允许作为索引
        # 使用 `object` 类型的数组索引删除操作，应引发 `IndexError` 异常
        with pytest.raises(IndexError):
            delete(np.ones(2), np.array([0], dtype=object))

        # 时间增量有时是“整数型”，但显然不应该被允许作为索引
        # 使用时间增量类型的数组索引删除操作，应引发 `IndexError` 异常
        with pytest.raises(IndexError):
            delete(np.ones(2), np.array([0], dtype="m8[ns]"))
class TestGradient:

    def test_basic(self):
        v = [[1, 1], [3, 4]]  # 创建一个二维列表 v
        x = np.array(v)  # 将二维列表转换为 NumPy 数组 x
        dx = [np.array([[2., 3.], [2., 3.]]),  # 预期的梯度结果 dx
              np.array([[0., 0.], [1., 1.]])]  
        assert_array_equal(gradient(x), dx)  # 断言调用 gradient 函数后返回值与 dx 相等
        assert_array_equal(gradient(v), dx)  # 断言调用 gradient 函数后返回值与 dx 相等

    def test_args(self):
        dx = np.cumsum(np.ones(5))  # 生成一个累积和数组 dx
        dx_uneven = [1., 2., 5., 9., 11.]  # 不均匀的距离数组 dx_uneven
        f_2d = np.arange(25).reshape(5, 5)  # 创建一个 5x5 的二维数组 f_2d

        # distances must be scalars or have size equal to gradient[axis]
        gradient(np.arange(5), 3.)  # 调用 gradient 函数，第二个参数为标量 3.0
        gradient(np.arange(5), np.array(3.))  # 调用 gradient 函数，第二个参数为数组 [3.]
        gradient(np.arange(5), dx)  # 调用 gradient 函数，第二个参数为数组 dx
        # dy is set equal to dx because scalar
        gradient(f_2d, 1.5)  # 调用 gradient 函数，第二个参数为标量 1.5
        gradient(f_2d, np.array(1.5))  # 调用 gradient 函数，第二个参数为数组 [1.5]

        gradient(f_2d, dx_uneven, dx_uneven)  # 调用 gradient 函数，使用不均匀距离数组 dx_uneven
        # mix between even and uneven spaces and
        # mix between scalar and vector
        gradient(f_2d, dx, 2)  # 调用 gradient 函数，第二个参数为数组 dx，第三个参数为标量 2

        # 2D but axis specified
        gradient(f_2d, dx, axis=1)  # 调用 gradient 函数，第二个参数为数组 dx，指定 axis=1

        # 2d coordinate arguments are not yet allowed
        assert_raises_regex(ValueError, '.*scalars or 1d',
            gradient, f_2d, np.stack([dx]*2, axis=-1), 1)  # 使用 assert_raises_regex 检查是否抛出 ValueError 异常

    def test_badargs(self):
        f_2d = np.arange(25).reshape(5, 5)  # 创建一个 5x5 的二维数组 f_2d
        x = np.cumsum(np.ones(5))  # 生成一个累积和数组 x

        # wrong sizes
        assert_raises(ValueError, gradient, f_2d, x, np.ones(2))  # 使用 assert_raises 检查是否抛出 ValueError 异常
        assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))  # 使用 assert_raises 检查是否抛出 ValueError 异常
        assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))  # 使用 assert_raises 检查是否抛出 ValueError 异常
        # wrong number of arguments
        assert_raises(TypeError, gradient, f_2d, x)  # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, gradient, f_2d, x, axis=(0,1))  # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, gradient, f_2d, x, x, x)  # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)  # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)  # 使用 assert_raises 检查是否抛出 TypeError 异常
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)  # 使用 assert_raises 检查是否抛出 TypeError 异常

    def test_datetime64(self):
        # Make sure gradient() can handle special types like datetime64
        x = np.array(
            ['1910-08-16', '1910-08-11', '1910-08-10', '1910-08-12',
             '1910-10-12', '1910-12-12', '1912-12-12'],
            dtype='datetime64[D]')  # 创建一个包含日期字符串的 datetime64 数组 x
        dx = np.array(
            [-5, -3, 0, 31, 61, 396, 731],
            dtype='timedelta64[D]')  # 创建一个时间间隔数组 dx
        assert_array_equal(gradient(x), dx)  # 断言调用 gradient 函数后返回值与 dx 相等
        assert_(dx.dtype == np.dtype('timedelta64[D]'))  # 断言 dx 的数据类型为 timedelta64[D]

    def test_masked(self):
        # Make sure that gradient supports subclasses like masked arrays
        x = np.ma.array([[1, 1], [3, 4]],
                        mask=[[False, False], [False, False]])  # 创建一个掩码数组 x
        out = gradient(x)[0]  # 调用 gradient 函数，并获取返回的第一个结果
        assert_equal(type(out), type(x))  # 断言输出的类型与输入的类型相同
        # And make sure that the output and input don't have aliased mask
        # arrays
        assert_(x._mask is not out._mask)  # 断言输出和输入的掩码数组不是同一个实例
        # Also check that edge_order=2 doesn't alter the original mask
        x2 = np.ma.arange(5)  # 创建一个掩码数组 x2
        x2[2] = np.ma.masked  # 将 x2 中的第二个元素设置为掩码值
        np.gradient(x2, edge_order=2)  # 调用 np.gradient 函数，指定 edge_order=2
        assert_array_equal(x2.mask, [False, False, True, False, False])  # 断言 x2 的掩码数组没有改变
    def test_second_order_accurate(self):
        # 测试相对数值误差是否小于3%对于这个示例问题。
        # 这对于所有内部和边界点来说，对应于二阶精确的有限差分。
        
        # 创建一个等间距的包含10个点的数组，范围从0到1
        x = np.linspace(0, 1, 10)
        # 计算数组中相邻两点的距离
        dx = x[1] - x[0]
        # 根据示例问题定义的函数，计算对应的 y 值
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        # 根据解析解，计算分析值
        analytical = 6 * x ** 2 + 8 * x + 2
        # 计算数值梯度并计算相对误差
        num_error = np.abs((np.gradient(y, dx, edge_order=2) / analytical) - 1)
        # 断言相对误差是否全部小于3%
        assert_(np.all(num_error < 0.03) == True)

        # 使用非均匀间距测试
        # 设定随机种子以便复现
        np.random.seed(0)
        # 创建一个包含10个随机数的数组，并按升序排列
        x = np.sort(np.random.random(10))
        # 根据示例问题定义的函数，计算对应的 y 值
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        # 根据解析解，计算分析值
        analytical = 6 * x ** 2 + 8 * x + 2
        # 计算数值梯度并计算相对误差
        num_error = np.abs((np.gradient(y, x, edge_order=2) / analytical) - 1)
        # 断言相对误差是否全部小于3%
        assert_(np.all(num_error < 0.03) == True)
    def test_spacing(self):
        # 创建一个 numpy 数组 f，包含浮点数
        f = np.array([0, 2., 3., 4., 5., 5.])
        # 使用 np.tile 函数将 f 复制成一个 6x6 的数组，并与 f 本身进行加法操作
        f = np.tile(f, (6,1)) + f.reshape(-1, 1)
        # 创建一个包含浮点数的 numpy 数组 x_uneven
        x_uneven = np.array([0., 0.5, 1., 3., 5., 7.])
        # 创建一个包含整数的 numpy 数组 x_even
        x_even = np.arange(6.)

        # 创建四个 numpy 数组，每个数组都是包含指定浮点数列表的 6x6 数组的副本
        fdx_even_ord1 = np.tile([2., 1.5, 1., 1., 0.5, 0.], (6,1))
        fdx_even_ord2 = np.tile([2.5, 1.5, 1., 1., 0.5, -0.5], (6,1))
        fdx_uneven_ord1 = np.tile([4., 3., 1.7, 0.5, 0.25, 0.], (6,1))
        fdx_uneven_ord2 = np.tile([5., 3., 1.7, 0.5, 0.25, -0.25], (6,1))

        # 对于等间距的情况
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            # 计算 f 在指定轴上的梯度，并进行断言检查
            res1 = gradient(f, 1., axis=(0,1), edge_order=edge_order)
            res2 = gradient(f, x_even, x_even,
                            axis=(0,1), edge_order=edge_order)
            res3 = gradient(f, x_even, x_even,
                            axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, 1., axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_almost_equal(res2, exp_res.T)

            res1 = gradient(f, 1., axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_array_equal(res2, exp_res)

        # 对于不等间距的情况
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            # 计算 f 在不等间距轴上的梯度，并进行断言检查
            res1 = gradient(f, x_uneven, x_uneven,
                            axis=(0,1), edge_order=edge_order)
            res2 = gradient(f, x_uneven, x_uneven,
                            axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_almost_equal(res1, exp_res.T)

            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_almost_equal(res1, exp_res)

        # 对于混合间距的情况
        res1 = gradient(f, x_even, x_uneven, axis=(0,1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1,0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord1.T)
        assert_almost_equal(res1[1], fdx_uneven_ord1)

        res1 = gradient(f, x_even, x_uneven, axis=(0,1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1,0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord2.T)
        assert_almost_equal(res1[1], fdx_uneven_ord2)
    def test_specific_axes(self):
        # Testing that gradient can work on a given axis only

        # 定义一个二维数组
        v = [[1, 1], [3, 4]]
        # 将二维数组转换为 NumPy 数组
        x = np.array(v)
        # 预期的梯度结果列表
        dx = [np.array([[2., 3.], [2., 3.]]),
              np.array([[0., 0.], [1., 1.]])]

        # 测试在指定轴上计算梯度，比较结果与预期值
        assert_array_equal(gradient(x, axis=0), dx[0])
        assert_array_equal(gradient(x, axis=1), dx[1])
        assert_array_equal(gradient(x, axis=-1), dx[1])
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])

        # 测试 axis=None，即所有轴上的梯度计算，与未给定 axis 关键字相同
        assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])

        # 测试变长参数的顺序
        assert_array_equal(gradient(x, 2, 3, axis=(1, 0)),
                           [dx[1]/2.0, dx[0]/3.0])

        # 测试超过最大变长参数数量的情况
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)

        # 测试超出范围的轴号
        assert_raises(AxisError, gradient, x, axis=3)
        assert_raises(AxisError, gradient, x, axis=-3)

    def test_timedelta64(self):
        # Make sure gradient() can handle special types like timedelta64

        # 创建 timedelta64 类型的 NumPy 数组
        x = np.array(
            [-5, -3, 10, 12, 61, 321, 300],
            dtype='timedelta64[D]')
        # 预期的梯度结果数组
        dx = np.array(
            [2, 7, 7, 25, 154, 119, -21],
            dtype='timedelta64[D]')

        # 检查梯度计算结果是否与预期相等
        assert_array_equal(gradient(x), dx)
        # 确保梯度计算结果的数据类型为 timedelta64[D]
        assert_(dx.dtype == np.dtype('timedelta64[D]'))

    def test_inexact_dtypes(self):
        # Iterate over different floating point dtypes to check dtype consistency

        # 遍历不同的浮点数数据类型，检查梯度计算结果的数据类型是否一致
        for dt in [np.float16, np.float32, np.float64]:
            # 创建指定数据类型的 NumPy 数组
            x = np.array([1, 2, 3], dtype=dt)
            # 检查梯度计算结果的数据类型是否与 diff 函数的结果一致
            assert_equal(gradient(x).dtype, np.diff(x).dtype)

    def test_values(self):
        # needs at least 2 points for edge_order == 1
        # 需要至少两个点以计算 edge_order == 1 的梯度
        gradient(np.arange(2), edge_order=1)
        # needs at least 3 points for edge_order == 2
        # 需要至少三个点以计算 edge_order == 2 的梯度
        gradient(np.arange(3), edge_order=2)

        # 检查在不满足条件时是否引发 ValueError 异常
        assert_raises(ValueError, gradient, np.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(2), edge_order=2)

    @pytest.mark.parametrize('f_dtype', [np.uint8, np.uint16,
                                         np.uint32, np.uint64])
    def test_f_decreasing_unsigned_int(self, f_dtype):
        # Test gradient behavior on decreasing unsigned integer types

        # 创建无符号整数类型的 NumPy 数组
        f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
        # 计算数组的梯度，预期结果每个元素为 -1
        g = gradient(f)
        assert_array_equal(g, [-1]*len(f))

    @pytest.mark.parametrize('f_dtype', [np.int8, np.int16,
                                         np.int32, np.int64])
    # 定义测试函数，测试有符号整数的大跃变
    def test_f_signed_int_big_jump(self, f_dtype):
        # 获取给定数据类型的最大整数值
        maxint = np.iinfo(f_dtype).max
        # 创建一个包含两个元素的 NumPy 数组
        x = np.array([1, 3])
        # 创建一个包含两个元素的 NumPy 数组，其中第二个元素为给定数据类型的最大整数值，第一个元素为 -1
        f = np.array([-1, maxint], dtype=f_dtype)
        # 计算 f 对 x 的梯度
        dfdx = gradient(f, x)
        # 断言梯度数组等于 [(maxint + 1) // 2, (maxint + 1) // 2]
        assert_array_equal(dfdx, [(maxint + 1) // 2]*2)

    # 使用参数化测试装饰器定义测试函数，测试无符号整数的降序情况
    @pytest.mark.parametrize('x_dtype', [np.uint8, np.uint16,
                                         np.uint32, np.uint64])
    def test_x_decreasing_unsigned(self, x_dtype):
        # 创建一个降序排列的 NumPy 数组，数据类型由参数 x_dtype 指定
        x = np.array([3, 2, 1], dtype=x_dtype)
        # 创建一个与 x 相同长度的 NumPy 数组，包含整数 0, 2, 4
        f = np.array([0, 2, 4])
        # 计算 f 对 x 的梯度
        dfdx = gradient(f, x)
        # 断言梯度数组等于 [-2, -2, -2]（数组长度与 x 相同）
        assert_array_equal(dfdx, [-2]*len(x))

    # 使用参数化测试装饰器定义测试函数，测试有符号整数的大跃变
    @pytest.mark.parametrize('x_dtype', [np.int8, np.int16,
                                         np.int32, np.int64])
    def test_x_signed_int_big_jump(self, x_dtype):
        # 获取给定数据类型的最小整数值和最大整数值
        minint = np.iinfo(x_dtype).min
        maxint = np.iinfo(x_dtype).max
        # 创建一个包含两个元素的 NumPy 数组，其中一个元素为 -1，另一个为给定数据类型的最大整数值
        x = np.array([-1, maxint], dtype=x_dtype)
        # 创建一个包含两个元素的 NumPy 数组，第一个元素为给定数据类型最小整数值除以 2，第二个元素为 0
        f = np.array([minint // 2, 0])
        # 计算 f 对 x 的梯度
        dfdx = gradient(f, x)
        # 断言梯度数组等于 [0.5, 0.5]（数组长度与 x 相同）
        assert_array_equal(dfdx, [0.5, 0.5])

    # 定义测试函数，测试 np.gradient 函数的返回类型
    def test_return_type(self):
        # 对 ([1, 2], [2, 3]) 应用 np.gradient 函数，返回结果保存在 res 中
        res = np.gradient(([1, 2], [2, 3]))
        # 断言 res 的类型为 tuple
        assert type(res) is tuple
class TestAngle:
    
    def test_basic(self):
        # 创建包含复数的列表
        x = [1 + 3j, np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2,
             1, 1j, -1, -1j, 1 - 3j, -1 + 3j]
        # 计算列表中每个复数的幅角，返回一个数组
        y = angle(x)
        # 预期的幅角列表
        yo = [
            np.arctan(3.0 / 1.0),
            np.arctan(1.0), 0, np.pi / 2, np.pi, -np.pi / 2.0,
            -np.arctan(3.0 / 1.0), np.pi - np.arctan(3.0 / 1.0)]
        # 计算角度制的幅角
        z = angle(x, deg=True)
        # 将弧度转换为角度制
        zo = np.array(yo) * 180 / np.pi
        # 断言计算得到的幅角与预期的幅角几乎相等
        assert_array_almost_equal(y, yo, 11)
        # 断言计算得到的角度制幅角与预期的角度制幅角几乎相等
        assert_array_almost_equal(z, zo, 11)

    def test_subclass(self):
        # 创建包含掩码数组的列表
        x = np.ma.array([1 + 3j, 1, np.sqrt(2)/2 * (1 + 1j)])
        # 将第二个元素设为掩码
        x[1] = np.ma.masked
        # 预期的幅角掩码数组
        expected = np.ma.array([np.arctan(3.0 / 1.0), 0, np.arctan(1.0)])
        # 将第二个元素设为掩码
        expected[1] = np.ma.masked
        # 计算掩码数组的幅角
        actual = angle(x)
        # 断言实际得到的类型与预期的类型相同
        assert_equal(type(actual), type(expected))
        # 断言实际得到的掩码与预期的掩码相同
        assert_equal(actual.mask, expected.mask)
        # 断言实际得到的幅角数组与预期的幅角数组相同
        assert_equal(actual, expected)


class TestTrimZeros:
    
    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    d = a.astype(object)

    def values(self):
        # 返回包含属性值的生成器
        attr_names = ('a', 'b', 'c', 'd')
        return (getattr(self, name) for name in attr_names)

    def test_basic(self):
        # 创建切片对象
        slc = np.s_[2:-1]
        # 遍历所有属性值数组
        for arr in self.values():
            # 调用函数移除数组开头和结尾的零
            res = trim_zeros(arr)
            # 断言移除零后的结果与预期的切片数组相同
            assert_array_equal(res, arr[slc])

    def test_leading_skip(self):
        # 创建切片对象
        slc = np.s_[:-1]
        # 遍历所有属性值数组
        for arr in self.values():
            # 调用函数移除数组开头的零
            res = trim_zeros(arr, trim='b')
            # 断言移除零后的结果与预期的切片数组相同
            assert_array_equal(res, arr[slc])

    def test_trailing_skip(self):
        # 创建切片对象
        slc = np.s_[2:]
        # 遍历所有属性值数组
        for arr in self.values():
            # 调用函数移除数组结尾的零
            res = trim_zeros(arr, trim='F')
            # 断言移除零后的结果与预期的切片数组相同
            assert_array_equal(res, arr[slc])

    def test_all_zero(self):
        # 遍历所有属性值数组
        for _arr in self.values():
            # 创建与属性值数组类型相同的全零数组
            arr = np.zeros_like(_arr, dtype=_arr.dtype)

            # 调用函数移除数组开头的零
            res1 = trim_zeros(arr, trim='B')
            # 断言移除零后结果数组的长度为零
            assert len(res1) == 0

            # 调用函数移除数组结尾的零
            res2 = trim_zeros(arr, trim='f')
            # 断言移除零后结果数组的长度为零
            assert len(res2) == 0

    def test_size_zero(self):
        # 创建长度为零的全零数组
        arr = np.zeros(0)
        # 调用函数移除数组中的零
        res = trim_zeros(arr)
        # 断言移除零后结果数组与原数组相同
        assert_array_equal(arr, res)

    @pytest.mark.parametrize(
        'arr',
        [np.array([0, 2**62, 0]),
         np.array([0, 2**63, 0]),
         np.array([0, 2**64, 0])]
    )
    def test_overflow(self, arr):
        # 创建切片对象
        slc = np.s_[1:2]
        # 调用函数移除数组中的零
        res = trim_zeros(arr)
        # 断言移除零后结果数组与预期的切片数组相同
        assert_array_equal(res, arr[slc])

    def test_no_trim(self):
        # 创建包含 None 的数组
        arr = np.array([None, 1, None])
        # 调用函数移除数组中的零
        res = trim_zeros(arr)
        # 断言移除零后结果数组与原数组相同
        assert_array_equal(arr, res)

    def test_list_to_list(self):
        # 调用函数移除列表中的零
        res = trim_zeros(self.a.tolist())
        # 断言移除零后结果与预期的列表相同
        assert isinstance(res, list)


class TestExtins:
    
    def test_basic(self):
        # 创建整数数组
        a = np.array([1, 3, 2, 1, 2, 3, 3])
        # 提取满足条件的数组元素
        b = extract(a > 1, a)
        # 断言提取后的数组与预期的数组相同
        assert_array_equal(b, [3, 2, 2, 3, 3])
    def test_place(self):
        # 确保非 np.ndarray 对象会抛出 TypeError 而不是无动作
        assert_raises(TypeError, place, [1, 2, 3], [True, False], [0, 1])

        # 创建一个包含整数的 NumPy 数组
        a = np.array([1, 4, 3, 2, 5, 8, 7])
        # 使用 place 函数根据条件数组修改原数组
        place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        # 验证修改后的数组是否符合预期
        assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])

        # 使用空的条件数组重新调用 place 函数
        place(a, np.zeros(7), [])
        # 验证数组是否恢复为原始状态
        assert_array_equal(a, np.arange(1, 8))

        # 再次调用 place 函数，使用不同的条件数组和替换数组
        place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        # 验证数组再次修改是否符合预期
        assert_array_equal(a, [8, 2, 9, 4, 8, 6, 9])
        # 使用 lambda 函数和 assert_raises_regex 检查特定错误是否被抛出
        assert_raises_regex(ValueError, "Cannot insert from an empty array",
                            lambda: place(a, [0, 0, 0, 0, 0, 1, 0], []))

        # 测试针对问题 #6974 的情况
        a = np.array(['12', '34'])
        # 使用 place 函数根据条件数组修改字符串数组
        place(a, [0, 1], '9')
        # 验证修改后的数组是否符合预期
        assert_array_equal(a, ['12', '9'])

    def test_both(self):
        # 创建一个随机数的 NumPy 数组
        a = rand(10)
        # 创建一个布尔掩码
        mask = a > 0.5
        # 创建原数组的副本
        ac = a.copy()
        # 使用 extract 函数根据掩码提取元素
        c = extract(mask, a)
        # 使用 place 函数根据掩码将数组中的元素替换为标量值
        place(a, mask, 0)
        # 使用 place 函数根据掩码将数组中的元素替换为另一个数组的值
        place(a, mask, c)
        # 验证数组是否按预期修改
        assert_array_equal(a, ac)
# _foo1 and _foo2 are used in some tests in TestVectorize.

# 定义一个函数 _foo1，接受参数 x 和可选参数 y，默认值为 1.0，返回 y 乘以 x 的向下取整结果
def _foo1(x, y=1.0):
    return y*math.floor(x)

# 定义一个函数 _foo2，接受参数 x、y 和可选参数 z，默认值为 0.0，返回 y 乘以 x 的向下取整结果加上 z
def _foo2(x, y=1.0, z=0.0):
    return y*math.floor(x) + z

# 定义一个测试类 TestVectorize
class TestVectorize:

    # 测试简单的向量化函数
    def test_simple(self):
        # 定义一个内部函数 addsubtract，接受两个参数 a 和 b，根据条件返回 a - b 或者 a + b
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        # 使用 vectorize 函数将 addsubtract 函数向量化
        f = vectorize(addsubtract)
        # 调用向量化后的函数 f，传入两个数组作为参数，并将结果赋给 r
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        # 断言结果数组 r 与预期数组 [1, 6, 1, 2] 相等
        assert_array_equal(r, [1, 6, 1, 2])

    # 测试向量化函数中处理标量参数的情况
    def test_scalar(self):
        # 定义一个内部函数 addsubtract，接受两个参数 a 和 b，根据条件返回 a - b 或者 a + b
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        # 使用 vectorize 函数将 addsubtract 函数向量化
        f = vectorize(addsubtract)
        # 调用向量化后的函数 f，传入一个数组和一个标量作为参数，并将结果赋给 r
        r = f([0, 3, 6, 9], 5)
        # 断言结果数组 r 与预期数组 [5, 8, 1, 4] 相等
        assert_array_equal(r, [5, 8, 1, 4])

    # 测试处理大量数据的情况
    def test_large(self):
        # 生成一个包含 10000 个数的等差数列 x
        x = np.linspace(-3, 2, 10000)
        # 使用 vectorize 函数将 lambda 函数向量化
        f = vectorize(lambda x: x)
        # 调用向量化后的函数 f，传入数组 x，并将结果赋给 y
        y = f(x)
        # 断言结果数组 y 与输入数组 x 相等
        assert_array_equal(y, x)

    # 测试向量化数学函数的情况
    def test_ufunc(self):
        # 使用 vectorize 函数将 math.cos 函数向量化
        f = vectorize(math.cos)
        # 生成一个包含几个角度值的数组 args
        args = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        # 调用向量化后的函数 f，传入数组 args，并将结果赋给 r1
        r1 = f(args)
        # 调用 numpy 自带的 cos 函数，传入数组 args，并将结果赋给 r2
        r2 = np.cos(args)
        # 断言结果数组 r1 与 r2 在相等误差范围内相等
        assert_array_almost_equal(r1, r2)

    # 测试带关键字参数的函数向量化情况
    def test_keywords(self):
        # 定义一个带有关键字参数的函数 foo，接受参数 a 和可选参数 b，默认值为 1，返回 a 加上 b 的结果
        def foo(a, b=1):
            return a + b

        # 使用 vectorize 函数将 foo 函数向量化
        f = vectorize(foo)
        # 生成一个整数数组 args
        args = np.array([1, 2, 3])
        # 调用向量化后的函数 f，传入数组 args，并将结果赋给 r1
        r1 = f(args)
        # 生成一个预期结果数组 r2
        r2 = np.array([2, 3, 4])
        # 断言结果数组 r1 与预期数组 r2 相等
        assert_array_equal(r1, r2)
        # 再次调用向量化后的函数 f，传入数组 args 和一个标量 2，并将结果赋给 r1
        r1 = f(args, 2)
        # 生成另一个预期结果数组 r2
        r2 = np.array([3, 4, 5])
        # 断言结果数组 r1 与预期数组 r2 相等
        assert_array_equal(r1, r2)

    # 测试带有输出类型参数的函数向量化情况，测试顺序 1
    def test_keywords_with_otypes_order1(self):
        # 使用 vectorize 函数将 _foo1 函数向量化，并指定输出类型为 float
        f = vectorize(_foo1, otypes=[float])
        # 调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0) 和一个标量 1.0，并将结果赋给 r1
        r1 = f(np.arange(3.0), 1.0)
        # 再次调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0)，并将结果赋给 r2
        r2 = f(np.arange(3.0))
        # 断言结果数组 r1 与 r2 相等
        assert_array_equal(r1, r2)

    # 测试带有输出类型参数的函数向量化情况，测试顺序 2
    def test_keywords_with_otypes_order2(self):
        # 使用 vectorize 函数将 _foo1 函数向量化，并指定输出类型为 float
        f = vectorize(_foo1, otypes=[float])
        # 调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0)，并将结果赋给 r1
        r1 = f(np.arange(3.0))
        # 再次调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0) 和一个标量 1.0，并将结果赋给 r2
        r2 = f(np.arange(3.0), 1.0)
        # 断言结果数组 r1 与 r2 相等
        assert_array_equal(r1, r2)

    # 测试带有输出类型参数的函数向量化情况，测试顺序 3
    def test_keywords_with_otypes_order3(self):
        # 使用 vectorize 函数将 _foo1 函数向量化，并指定输出类型为 float
        f = vectorize(_foo1, otypes=[float])
        # 调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0)，并将结果赋给 r1
        r1 = f(np.arange(3.0))
        # 再次调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0) 和一个关键字参数 y=1.0，并将结果赋给 r2
        r2 = f(np.arange(3.0), y=1.0)
        # 第三次调用向量化后的函数 f，传入一个浮点数数组 np.arange(3.0)，并将结果赋给 r3
        r3 = f(np.arange(3.0))
        # 断言结果数组 r1 与 r2 相等
        assert_array_equal(r1, r2)
        # 断言结果数组 r1 与 r3 相等
        assert_array_equal(r1, r3)
    def test_keywords_with_otypes_several_kwd_args1(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        # 使用 vectorize 函数对 _foo2 函数进行向量化处理，并指定输出类型为 float
        f = vectorize(_foo2, otypes=[float])
        # 测试 ufunc 的缓存功能，函数调用顺序对测试结果有重要影响
        r1 = f(10.4, z=100)
        r2 = f(10.4, y=-1)
        r3 = f(10.4)
        # 断言各函数调用的结果与直接调用 _foo2 函数的结果相等
        assert_equal(r1, _foo2(10.4, z=100))
        assert_equal(r2, _foo2(10.4, y=-1))
        assert_equal(r3, _foo2(10.4))

    def test_keywords_with_otypes_several_kwd_args2(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        # 使用 vectorize 函数对 _foo2 函数进行向量化处理，并指定输出类型为 float
        f = vectorize(_foo2, otypes=[float])
        # 测试 ufunc 的缓存功能，函数调用顺序对测试结果有重要影响
        r1 = f(z=100, x=10.4, y=-1)
        r2 = f(1, 2, 3)
        # 断言各函数调用的结果与直接调用 _foo2 函数的结果相等
        assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
        assert_equal(r2, _foo2(1, 2, 3))

    def test_keywords_no_func_code(self):
        # This needs to test a function that has keywords but
        # no func_code attribute, since otherwise vectorize will
        # inspect the func_code.
        # 导入 random 模块，测试 vectorize 能否处理没有 func_code 属性的函数
        import random
        try:
            vectorize(random.randrange)  # Should succeed
        except Exception:
            raise AssertionError()

    def test_keywords2_ticket_2100(self):
        # Test kwarg support: enhancement ticket 2100
        # 定义一个简单的函数 foo，对其进行向量化处理
        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        # 创建一个 NumPy 数组作为参数
        args = np.array([1, 2, 3])
        r1 = f(a=args)
        r2 = np.array([2, 3, 4])
        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(r1, r2)
        r1 = f(b=1, a=args)
        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(r1, r2)
        r1 = f(args, b=2)
        r2 = np.array([3, 4, 5])
        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(r1, r2)

    def test_keywords3_ticket_2100(self):
        # Test excluded with mixed positional and kwargs: ticket 2100
        # 定义一个多项式求值函数 mypolyval
        def mypolyval(x, p):
            _p = list(p)
            res = _p.pop(0)
            while _p:
                res = res * x + _p.pop(0)
            return res

        # 使用 vectorize 函数对 mypolyval 进行向量化处理，排除 'p' 和第二个参数（默认为 1）
        vpolyval = np.vectorize(mypolyval, excluded=['p', 1])
        ans = [3, 6]
        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))

    def test_keywords4_ticket_2100(self):
        # Test vectorizing function with no positional args.
        # 定义一个带有关键字参数的向量化函数 f
        @vectorize
        def f(**kw):
            res = 1.0
            for _k in kw:
                res *= kw[_k]
            return res

        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])

    def test_keywords5_ticket_2100(self):
        # Test vectorizing function with no kwargs args.
        # 定义一个带有可变位置参数的向量化函数 f
        @vectorize
        def f(*v):
            return np.prod(v)

        # 断言向量化函数的输出与预期结果一致
        assert_array_equal(f([1, 2], [3, 4]), [3, 8])
    def test_coverage1_ticket_2100(self):
        # 测试覆盖率：ticket 2100
        def foo():
            # 定义一个简单的函数 foo，返回整数 1
            return 1

        # 对函数 foo 进行向量化处理
        f = vectorize(foo)
        # 断言向量化后的函数调用结果与预期结果相等
        assert_array_equal(f(), 1)

    def test_assigning_docstring(self):
        # 测试赋值文档字符串功能
        def foo(x):
            """Original documentation"""
            # 原始的文档字符串
            return x

        # 对函数 foo 进行向量化处理
        f = vectorize(foo)
        # 断言向量化后函数的文档字符串与原始函数的文档字符串相等
        assert_equal(f.__doc__, foo.__doc__)

        # 定义新的文档字符串
        doc = "Provided documentation"
        # 使用新的文档字符串对函数 foo 进行向量化处理
        f = vectorize(foo, doc=doc)
        # 断言向量化后函数的文档字符串与新定义的文档字符串相等
        assert_equal(f.__doc__, doc)

    def test_UnboundMethod_ticket_1156(self):
        # 未绑定方法的测试：ticket 1156
        # 定义一个简单的类 Foo
        class Foo:
            b = 2

            def bar(self, a):
                # 返回 a 的 b 次方
                return a ** self.b

        # 实例化 Foo 类并对 bar 方法进行向量化处理，测试其结果
        assert_array_equal(vectorize(Foo().bar)(np.arange(9)),
                           np.arange(9) ** 2)
        # 对类 Foo 的 bar 方法进行向量化处理，测试其结果
        assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)),
                           np.arange(9) ** 2)

    def test_execution_order_ticket_1487(self):
        # 执行顺序的测试：ticket 1487
        # 创建一个简单的向量化函数 f1
        f1 = vectorize(lambda x: x)
        # 使用不同的输入顺序测试 f1 的结果
        res1a = f1(np.arange(3))
        res1b = f1(np.arange(0.1, 3))
        # 创建另一个简单的向量化函数 f2
        f2 = vectorize(lambda x: x)
        # 使用不同的输入顺序测试 f2 的结果
        res2b = f2(np.arange(0.1, 3))
        res2a = f2(np.arange(3))
        # 断言两次执行的结果一致
        assert_equal(res1a, res2a)
        assert_equal(res1b, res2b)

    def test_string_ticket_1892(self):
        # 字符串向量化测试：ticket 1892
        # 创建一个简单的向量化函数 f
        f = np.vectorize(lambda x: x)
        # 创建一个长字符串 s
        s = '0123456789' * 10
        # 断言向量化处理后的字符串与原始字符串相等
        assert_equal(s, f(s))

    def test_cache(self):
        # 确保每个参数只调用一次向量化函数
        _calls = [0]

        @vectorize
        def f(x):
            # 计数调用次数
            _calls[0] += 1
            return x ** 2

        # 启用函数缓存
        f.cache = True
        # 创建输入数组 x
        x = np.arange(5)
        # 断言向量化处理后的结果与预期结果相等
        assert_array_equal(f(x), x * x)
        # 断言函数调用次数与输入数组长度相等
        assert_equal(_calls[0], len(x))

    def test_otypes(self):
        # 指定输出类型的测试
        # 创建一个简单的向量化函数 f
        f = np.vectorize(lambda x: x)
        # 指定输出类型为整数 'i'
        f.otypes = 'i'
        # 创建输入数组 x
        x = np.arange(5)
        # 断言向量化处理后的结果与预期结果相等
        assert_array_equal(f(x), x)
    # 定义测试函数，用于测试 `_parse_gufunc_signature` 方法的不同输入情况
    def test_parse_gufunc_signature(self):
        # 测试单输入，无输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(x)->()'), ([('x',)], [()]))
        # 测试两个输入，无输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(x,y)->()'),
                     ([('x', 'y')], [()]))
        # 测试两个输入分别作为元组，无输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(x),(y)->()'),
                     ([('x',), ('y',)], [()]))
        # 测试单输入，单输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(x)->(y)'),
                     ([('x',)], [('y',)]))
        # 测试单输入，两个输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(x)->(y),()'),
                     ([('x',)], [('y',), ()]))
        # 测试三个输入，两个输出的情况，验证返回的参数和返回值的格式
        assert_equal(nfb._parse_gufunc_signature('(),(a,b,c),(d)->(d,e)'),
                     ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))

        # 测试带有空格的输入，验证空格是否被忽略
        assert_equal(nfb._parse_gufunc_signature('(x )->()'), ([('x',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('( x , y )->(  )'),
                     ([('x', 'y')], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x),( y) ->()'),
                     ([('x',), ('y',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(  x)-> (y )  '),
                     ([('x',)], [('y',)]))
        assert_equal(nfb._parse_gufunc_signature(' (x)->( y),( )'),
                     ([('x',)], [('y',), ()]))
        assert_equal(nfb._parse_gufunc_signature(
                     '(  ), ( a,  b,c )  ,(  d)   ->   (d  ,  e)'),
                     ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))

        # 测试错误的输入，应该抛出 ValueError 异常
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x)(y)->()')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x),(y)->')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('((x))->(x)')
    # 定义测试函数 test_signature_outer，测试向量化函数 vectorize 的使用，指定函数签名为 '(a),(b)->(a,b)'
    def test_signature_outer(self):
        # 使用 vectorize 函数向量化 np.outer 函数，生成新的函数 f
        f = vectorize(np.outer, signature='(a),(b)->(a,b)')
        # 调用向量化后的函数 f，计算 [1, 2] 和 [1, 2, 3] 的外积
        r = f([1, 2], [1, 2, 3])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])

        # 再次调用向量化后的函数 f，计算 [[[1, 2]]] 和 [1, 2, 3] 的外积
        r = f([[[1, 2]]], [1, 2, 3])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])

        # 再次调用向量化后的函数 f，计算 [[1, 0], [2, 0]] 和 [1, 2, 3] 的外积
        r = f([[1, 0], [2, 0]], [1, 2, 3])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]],
                               [[2, 4, 6], [0, 0, 0]]])

        # 再次调用向量化后的函数 f，计算 [1, 2] 和 [[1, 2, 3], [0, 0, 0]] 的外积
        r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]],
                               [[0, 0, 0], [0, 0, 0]]])

    # 定义测试函数 test_signature_computed_size，测试 vectorize 函数的使用，指定 lambda 函数签名为 '(n)->(m)'
    def test_signature_computed_size(self):
        # 使用 vectorize 函数向量化 lambda 函数，生成新的函数 f
        f = vectorize(lambda x: x[:-1], signature='(n)->(m)')
        # 调用向量化后的函数 f，对 [1, 2, 3] 执行操作，截取最后一个元素前的所有元素
        r = f([1, 2, 3])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [1, 2])

        # 再次调用向量化后的函数 f，对 [[1, 2, 3], [2, 3, 4]] 执行操作，截取每个子列表的最后一个元素前的所有元素
        r = f([[1, 2, 3], [2, 3, 4]])
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [[1, 2], [2, 3]])

    # 定义测试函数 test_signature_excluded，测试 vectorize 函数的使用，指定原始函数 foo 和签名为 '()->()'，排除参数 'b'
    def test_signature_excluded(self):

        # 定义原始函数 foo，接受参数 a 和可选参数 b，默认值为 1，返回 a + b
        def foo(a, b=1):
            return a + b

        # 使用 vectorize 函数向量化 foo 函数，生成新的函数 f，排除参数 'b'
        f = vectorize(foo, signature='()->()', excluded={'b'})
        # 断言调用 f([1, 2, 3]) 的结果与预期的数组相等
        assert_array_equal(f([1, 2, 3]), [2, 3, 4])
        # 断言调用 f([1, 2, 3], b=0) 的结果与预期的数组相等
        assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])

    # 定义测试函数 test_signature_otypes，测试 vectorize 函数的使用，指定 lambda 函数签名为 '(n)->(n)'，指定输出类型为 'float64'
    def test_signature_otypes(self):
        # 使用 vectorize 函数向量化 lambda 函数，生成新的函数 f，指定输出类型为 'float64'
        f = vectorize(lambda x: x, signature='(n)->(n)', otypes=['float64'])
        # 调用向量化后的函数 f，对 [1, 2, 3] 执行操作
        r = f([1, 2, 3])
        # 断言计算结果的数据类型与预期的 'float64' 类型相等
        assert_equal(r.dtype, np.dtype('float64'))
        # 断言计算结果与预期的数组相等
        assert_array_equal(r, [1, 2, 3])

    # 定义测试函数 test_signature_invalid_inputs，测试 vectorize 函数的使用，指定原始函数为 operator.add，签名为 '(n),(n)->(n)'
    def test_signature_invalid_inputs(self):
        # 使用 vectorize 函数向量化 operator.add 函数，生成新的函数 f
        f = vectorize(operator.add, signature='(n),(n)->(n)')
        # 使用 assert_raises_regex 检测调用 f([1, 2]) 时抛出的 TypeError 异常信息中是否包含 'wrong number of positional'
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f([1, 2])
        # 使用 assert_raises_regex 检测调用 f(1, 2) 时抛出的 ValueError 异常信息中是否包含 'does not have enough dimensions'
        with assert_raises_regex(
                ValueError, 'does not have enough dimensions'):
            f(1, 2)
        # 使用 assert_raises_regex 检测调用 f([1, 2], [1, 2, 3]) 时抛出的 ValueError 异常信息中是否包含 'inconsistent size for core dimension'
        with assert_raises_regex(
                ValueError, 'inconsistent size for core dimension'):
            f([1, 2], [1, 2, 3])

        # 使用 vectorize 函数向量化 operator.add 函数，生成新的函数 f，签名为 '()->()'
        f = vectorize(operator.add, signature='()->()')
        # 使用 assert_raises_regex 检测调用 f(1, 2) 时抛出的 TypeError 异常信息中是否包含 'wrong number of positional'
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f(1, 2)

    # 定义测试函数 test_signature_invalid_outputs，测试 vectorize 函数的使用，指定 lambda 函数签名为 '(n)->(n)'
    def test_signature_invalid_outputs(self):

        # 使用 vectorize 函数向量化 lambda 函数，生成新的函数 f，签名为 '(n)->(n)'
        f = vectorize(lambda x: x[:-1], signature='(n)->(n)')
        # 使用 assert_raises_regex 检测调用 f([1, 2, 3]) 时抛出的 ValueError 异常信息中是否包含 'inconsistent size for core dimension'
        with assert_raises_regex(
                ValueError, 'inconsistent size for core dimension'):
            f([1, 2, 3])

        # 使用 vectorize 函数向量化 lambda 函数，生成新的函数 f，签名为 '()->(),()'
        f = vectorize(lambda x: x, signature='()->(),()')
        # 使用 assert_raises_regex 检测调用 f(1) 时抛出的 ValueError 异常信息中是否包含 'wrong number of outputs'
        with assert_raises_regex(ValueError, 'wrong number of outputs'):
            f(1)

        # 使用 vectorize 函数向量化 lambda 函数，生成新的函数 f，签名为 '()->()'
        f = vectorize(lambda x: (x, x), signature='()->()')
        # 使用 assert_raises_regex 检测调用 f([1, 2]) 时抛出的 ValueError 异常信息中是否包含 'wrong number of outputs'
        with assert_raises_regex(ValueError, '
    def test_size_zero_output(self):
        # 测试向量化函数处理大小为零的输入时是否会引发 ValueError 异常，关注问题编号 5868
        f = np.vectorize(lambda x: x)
        # 创建一个大小为零、dtype 为 int 的二维零数组
        x = np.zeros([0, 5], dtype=int)
        # 断言调用 f(x) 会引发 ValueError 异常，错误信息中包含 'otypes'
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)

        # 设置 f 的 otypes 属性为 'i'
        f.otypes = 'i'
        # 断言调用 f(x) 返回与输入 x 相同的数组
        assert_array_equal(f(x), x)

        # 使用自定义签名 '()->()' 创建向量化函数 f，预期会引发 ValueError 异常
        f = np.vectorize(lambda x: x, signature='()->()')
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)

        # 使用自定义签名 '()->()' 和 otypes='i' 创建向量化函数 f
        assert_array_equal(f(x), x)

        # 使用自定义签名 '(n)->(n)' 和 otypes='i' 创建向量化函数 f
        assert_array_equal(f(x), x)

        # 使用自定义签名 '(n)->(n)' 创建向量化函数 f，并对输入的转置进行断言
        assert_array_equal(f(x.T), x.T)

        # 使用自定义签名 '()->(n)' 和 otypes='i' 创建向量化函数 f，预期会引发 ValueError 异常
        with assert_raises_regex(ValueError, 'new output dimensions'):
            f(x)

    def test_subclasses(self):
        # 定义一个继承自 np.ndarray 的子类 subclass
        class subclass(np.ndarray):
            pass

        # 创建一个 subclass 的实例 m 和 v
        m = np.array([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.]]).view(subclass)
        v = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).view(subclass)
        
        # 创建一个接受两个矩阵参数并返回矩阵的广义函数 matvec
        matvec = np.vectorize(np.matmul, signature='(m,m),(m)->(m)')
        # 使用 matvec 计算 m 和 v 的矩阵乘法，并断言返回结果的类型为 subclass
        r = matvec(m, v)
        assert_equal(type(r), subclass)
        # 断言矩阵乘法的结果与预期相符
        assert_equal(r, [[1., 3., 2.], [4., 6., 5.], [7., 9., 8.]])

        # 创建一个元素级别的向量化函数 mult
        mult = np.vectorize(lambda x, y: x*y)
        # 使用 mult 计算 m 和 v 的元素级乘法，并断言返回结果的类型为 subclass
        r = mult(m, v)
        assert_equal(type(r), subclass)
        # 断言元素级乘法的结果与预期相符
        assert_equal(r, m * v)

    def test_name(self):
        # 测试通过 np.vectorize 装饰器定义的向量化函数的 __name__ 属性是否正确
        @np.vectorize
        def f2(a, b):
            return a + b

        assert f2.__name__ == 'f2'

    def test_decorator(self):
        # 测试使用 @vectorize 装饰器定义的向量化函数 addsubtract
        @vectorize
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        # 对输入数组 [0, 3, 6, 9] 和 [1, 3, 5, 7] 执行 addsubtract 函数，并断言结果与预期相符
        r = addsubtract([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_docstring(self):
        # 测试向量化函数的文档字符串是否正确设置
        @vectorize
        def f(x):
            """Docstring"""
            return x

        # 在非优化模式下，断言向量化函数 f 的文档字符串为 "Docstring"
        if sys.flags.optimize < 2:
            assert f.__doc__ == "Docstring"

    def test_partial(self):
        # 测试 np.vectorize 的部分应用功能
        def foo(x, y):
            return x + y

        # 创建部分应用的函数 bar，并使用 np.vectorize 创建向量化函数 vbar
        bar = partial(foo, 3)
        vbar = np.vectorize(bar)
        # 断言调用 vbar(1) 返回值为 4
        assert vbar(1) == 4

    def test_signature_otypes_decorator(self):
        # 测试使用 @vectorize 装饰器定义的向量化函数 f，设置了自定义签名和输出类型
        @vectorize(signature='(n)->(n)', otypes=['float64'])
        def f(x):
            return x

        # 调用 f([1, 2, 3])，断言返回结果的 dtype 为 float64，且与输入相符
        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype('float64'))
        assert_array_equal(r, [1, 2, 3])
        # 断言向量化函数 f 的 __name__ 属性为 'f'
        assert f.__name__ == 'f'

    def test_bad_input(self):
        # 测试传入不正确的参数到 np.vectorize 是否会引发 TypeError
        with assert_raises(TypeError):
            A = np.vectorize(pyfunc=3)

    def test_no_keywords(self):
        # 测试在向 np.vectorize 函数中传入非关键字字符串是否会引发 TypeError
        with assert_raises(TypeError):
            @np.vectorize("string")
            def foo():
                return "bar"
    # 定义测试方法 test_positional_regression_9477，用于验证修复 #9477 后仍能正确转发第一个关键字参数作为位置参数的功能
    def test_positional_regression_9477(self):
        # 使用 vectorize 函数将 lambda 函数 x -> x 向量化，指定输入类型为 'float64'
        f = vectorize((lambda x: x), ['float64'])
        # 调用向量化后的函数 f，传入参数 [2]，并接收返回值
        r = f([2])
        # 断言返回值 r 的数据类型为 'float64'
        assert_equal(r.dtype, np.dtype('float64'))

    # 定义测试方法 test_datetime_conversion，用于测试日期时间类型的转换
    def test_datetime_conversion(self):
        # 定义目标输出类型 otype 为 "datetime64[ns]"
        otype = "datetime64[ns]"
        # 创建一个 numpy 数组 arr，包含日期字符串，指定数据类型为 'datetime64[ns]'
        arr = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], 
                       dtype='datetime64[ns]')
        # 使用 np.vectorize 函数将 lambda 函数 x -> x 向量化，指定签名为 "(i)->(j)"，目标输出类型为 otype
        # 对数组 arr 执行向量化操作，期望得到的结果与 arr 相等
        assert_array_equal(np.vectorize(lambda x: x, signature="(i)->(j)",
                                        otypes=[otype])(arr), arr)
# 定义一个测试类 TestLeaks，用于测试内存泄漏相关的情况
class TestLeaks:
    # 内部定义一个类 A
    class A:
        # 类变量 iters 设定为 20
        iters = 20
        
        # 实例方法 bound，接受任意参数并返回 0
        def bound(self, *args):
            return 0
        
        # 静态方法 unbound，接受任意参数并返回 0
        @staticmethod
        def unbound(*args):
            return 0
    
    # 装饰器标记此测试方法，如果没有引用计数的功能则跳过执行
    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    # 装饰器标记此测试方法，如果是无GIL构建则跳过执行，因为线程启动会使测试不稳定
    @pytest.mark.skipif(NOGIL_BUILD,
                        reason=("Functions are immortalized if a thread is "
                                "launched, making this test flaky"))
    # 参数化测试方法，name 为 'bound' 或 'unbound'，incr 分别为 A.iters 或 0
    @pytest.mark.parametrize('name, incr', [
            ('bound', A.iters),
            ('unbound', 0),
            ])
    # 定义测试方法 test_frompyfunc_leaks，接受参数 name 和 incr
    def test_frompyfunc_leaks(self, name, incr):
        # 作为 gh-11867 中的公开函数 np.vectorized 暴露，但问题源于 frompyfunc。
        # class.attribute = np.frompyfunc(<method>) 创建一个循环引用，如果 <method> 是绑定类方法。
        # 在 CPython 3 中需要进行垃圾回收来打破循环引用。
        import gc
        # 从 self.A 中获取名为 name 的函数 A_func
        A_func = getattr(self.A, name)
        # 禁用垃圾回收
        gc.disable()
        try:
            # 获取 A_func 的引用计数
            refcount = sys.getrefcount(A_func)
            # 执行 self.A.iters 次循环
            for i in range(self.A.iters):
                # 创建类 A 的实例 a
                a = self.A()
                # 将 getattr(a, name) 包装成 np.frompyfunc 的函数对象赋值给 a.f
                a.f = np.frompyfunc(getattr(a, name), 1, 1)
                # 使用 a.f 处理 np.arange(10) 的输出
                out = a.f(np.arange(10))
            # 将 a 置为 None，触发垃圾回收
            a = None
            # 如果 incr 非零，则 A_func 是一个循环引用的一部分
            assert_equal(sys.getrefcount(A_func), refcount + incr)
            # 执行 5 次垃圾回收
            for i in range(5):
                gc.collect()
            # 确认回收后 A_func 的引用计数与初始时相同
            assert_equal(sys.getrefcount(A_func), refcount)
        finally:
            # 恢复垃圾回收
            gc.enable()


# 定义另一个测试类 TestDigitize
class TestDigitize:

    # 测试正向情况的方法 test_forward
    def test_forward(self):
        # 创建数组 x，从 -6 到 4
        x = np.arange(-6, 5)
        # 创建数组 bins，从 -5 到 4
        bins = np.arange(-5, 5)
        # 断言 digitize(x, bins) 的结果与 np.arange(11) 相等
        assert_array_equal(digitize(x, bins), np.arange(11))

    # 测试反向情况的方法 test_reverse
    def test_reverse(self):
        # 创建数组 x，从 5 到 -6，步长为 -1
        x = np.arange(5, -6, -1)
        # 创建数组 bins，从 5 到 -5，步长为 -1
        bins = np.arange(5, -5, -1)
        # 断言 digitize(x, bins) 的结果与 np.arange(11) 相等
        assert_array_equal(digitize(x, bins), np.arange(11))

    # 测试随机情况的方法 test_random
    def test_random(self):
        # 创建长度为 10 的随机数组 x
        x = rand(10)
        # 创建 bin 数组，其范围为 x 最小值到最大值，共 10 个间隔
        bin = np.linspace(x.min(), x.max(), 10)
        # 断言 digitize(x, bin) 的所有结果都不为 0
        assert_(np.all(digitize(x, bin) != 0))

    # 测试右侧边界情况的方法 test_right_basic
    def test_right_basic(self):
        # 定义列表 x，包含多个数字
        x = [1, 5, 4, 10, 8, 11, 0]
        # 定义 bins 列表，包含多个分界点
        bins = [1, 5, 10]
        # 默认答案列表，用于比较 digitize(x, bins) 的结果
        default_answer = [1, 2, 1, 3, 2, 3, 0]
        # 断言 digitize(x, bins) 的结果与 default_answer 相等
        assert_array_equal(digitize(x, bins), default_answer)
        # 右侧边界答案列表，用于比较 digitize(x, bins, True) 的结果
        right_answer = [0, 1, 1, 2, 2, 3, 0]
        # 断言 digitize(x, bins, True) 的结果与 right_answer 相等
        assert_array_equal(digitize(x, bins, True), right_answer)

    # 测试右侧边界情况的方法 test_right_open
    def test_right_open(self):
        # 创建数组 x，从 -6 到 4
        x = np.arange(-6, 5)
        # 创建数组 bins，从 -6 到 3
        bins = np.arange(-6, 4)
        # 断言 digitize(x, bins, True) 的结果与 np.arange(11) 相等
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    # 测试反向右侧边界情况的方法 test_right_open_reverse
    def test_right_open_reverse(self):
        # 创建数组 x，从 5 到 -6，步长为 -1
        x = np.arange(5, -6, -1)
        # 创建数组 bins，从 4 到 -6，步长为 -1
        bins = np.arange(4, -6, -1)
        # 断言 digitize(x, bins, True) 的结果与 np.arange(11) 相等
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    # 测试随机反向右侧边界情况的方法 test_right_open_random
    def test_right_open_random(self):
        # 创建长度为 10 的随机数组 x
        x = rand(10)
        # 创建 bin 数组，其范围为 x 最小值到最大值，共 10 个间隔
        bins = np.linspace(x.min(), x.max(), 10)
        # 断言 digitize(x, bins, True) 的所有结果都不等于 10
        assert_(np.all(digitize(x, bins, True) != 10))
    # 定义一个测试方法，用于测试 digitize 函数在单调性检查时的行为
    def test_monotonic(self):
        # 准备输入数据 x 和 bins
        x = [-1, 0, 1, 2]
        bins = [0, 0, 1]
        # 断言 digitize 函数对非单调性的处理结果是否正确
        assert_array_equal(digitize(x, bins, False), [0, 2, 3, 3])
        # 断言 digitize 函数对单调性的处理结果是否正确
        assert_array_equal(digitize(x, bins, True), [0, 0, 2, 3])
        
        # 更换 bins 进行新的测试
        bins = [1, 1, 0]
        assert_array_equal(digitize(x, bins, False), [3, 2, 0, 0])
        assert_array_equal(digitize(x, bins, True), [3, 3, 2, 0])
        
        # 更换 bins 进行新的测试
        bins = [1, 1, 1, 1]
        assert_array_equal(digitize(x, bins, False), [0, 0, 4, 4])
        assert_array_equal(digitize(x, bins, True), [0, 0, 0, 4])
        
        # 测试不合法的 bins 输入是否会触发 ValueError
        bins = [0, 0, 1, 0]
        assert_raises(ValueError, digitize, x, bins)
        
        # 测试不合法的 bins 输入是否会触发 ValueError
        bins = [1, 1, 0, 1]
        assert_raises(ValueError, digitize, x, bins)

    # 定义一个测试方法，用于测试 digitize 函数在类型转换错误时的行为
    def test_casting_error(self):
        # 准备输入数据 x 和 bins，其中 x 包含复数类型
        x = [1, 2, 3 + 1.j]
        bins = [1, 2, 3]
        # 断言 digitize 函数对类型转换错误是否会触发 TypeError
        assert_raises(TypeError, digitize, x, bins)
        
        # 交换 x 和 bins 进行新的测试
        x, bins = bins, x
        assert_raises(TypeError, digitize, x, bins)

    # 定义一个测试方法，用于测试 digitize 函数在返回类型上的行为
    def test_return_type(self):
        # 函数返回索引时应始终返回基础的 ndarray 类型
        class A(np.ndarray):
            pass
        # 创建两个自定义 ndarray 对象 a 和 b
        a = np.arange(5).view(A)
        b = np.arange(1, 3).view(A)
        # 断言 digitize 函数返回结果不是 A 类型的实例
        assert_(not isinstance(digitize(b, a, False), A))
        assert_(not isinstance(digitize(b, a, True), A))

    # 定义一个测试方法，用于测试 digitize 函数在处理大整数时的行为
    def test_large_integers_increasing(self):
        # gh-11022: 测试当 x 是一个接近 2 的 54 次方的整数时的行为
        x = 2**54  # 在 float 中会丢失精度
        # 断言 digitize 函数对此情况的处理结果是否正确
        assert_equal(np.digitize(x, [x - 1, x + 1]), 1)

    # 使用 pytest.mark.xfail 标记的测试方法，用于测试 digitize 函数在处理大整数时可能出现的问题
    @pytest.mark.xfail(
        reason="gh-11022: np._core.multiarray._monoticity loses precision")
    def test_large_integers_decreasing(self):
        # gh-11022: 测试当 x 是一个接近 2 的 54 次方的整数时的行为
        x = 2**54  # 在 float 中会丢失精度
        # 断言 digitize 函数对此情况的处理结果是否正确
        assert_equal(np.digitize(x, [x + 1, x - 1]), 1)
class TestUnwrap:

    def test_simple(self):
        # 检查 unwrap 函数是否能够移除大于 2*pi 的跳跃
        assert_array_equal(unwrap([1, 1 + 2 * np.pi]), [1, 1])
        # 检查 unwrap 函数是否能够保持连续性
        assert_(np.all(diff(unwrap(rand(10) * 100)) < np.pi))

    def test_period(self):
        # 检查 unwrap 函数在指定周期下能够移除大于指定值（255）的跳跃
        assert_array_equal(unwrap([1, 1 + 256], period=255), [1, 2])
        # 检查 unwrap 函数在指定周期下能够保持连续性
        assert_(np.all(diff(unwrap(rand(10) * 1000, period=255)) < 255))
        
        # 检查简单情况
        simple_seq = np.array([0, 75, 150, 225, 300])
        wrap_seq = np.mod(simple_seq, 255)
        assert_array_equal(unwrap(wrap_seq, period=255), simple_seq)
        
        # 检查自定义不连续值
        uneven_seq = np.array([0, 75, 150, 225, 300, 430])
        wrap_uneven = np.mod(uneven_seq, 250)
        
        # 测试没有指定不连续值时的 unwrap 函数返回结果
        no_discont = unwrap(wrap_uneven, period=250)
        assert_array_equal(no_discont, [0, 75, 150, 225, 300, 180])
        
        # 测试指定较小的不连续值时的 unwrap 函数返回结果
        sm_discont = unwrap(wrap_uneven, period=250, discont=140)
        assert_array_equal(sm_discont, [0, 75, 150, 225, 300, 430])
        
        # 检查返回结果的数据类型与原始数据类型是否一致
        assert sm_discont.dtype == wrap_uneven.dtype


@pytest.mark.parametrize(
    "dtype", "O" + np.typecodes["AllInteger"] + np.typecodes["Float"]
)
@pytest.mark.parametrize("M", [0, 1, 10])
class TestFilterwindows:

    def test_hanning(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        # 使用 hanning 函数生成窗口
        w = hanning(scalar)
        
        # 根据数据类型确定参考数据类型
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        
        # 检查生成的窗口数据类型是否与参考数据类型一致
        assert w.dtype == ref_dtype

        # 检查窗口是否对称
        assert_equal(w, flipud(w))

        # 检查已知值
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.500, 4)

    def test_hamming(self, dtype: str, M: int) -> None:
        scalar = np.array(M, dtype=dtype)[()]

        # 使用 hamming 函数生成窗口
        w = hamming(scalar)
        
        # 根据数据类型确定参考数据类型
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        
        # 检查生成的窗口数据类型是否与参考数据类型一致
        assert w.dtype == ref_dtype

        # 检查窗口是否对称
        assert_equal(w, flipud(w))

        # 检查已知值
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.9400, 4)
    # 使用 Bartlett 窗函数测试函数
    def test_bartlett(self, dtype: str, M: int) -> None:
        # 将输入的 M 转换为指定数据类型的标量
        scalar = np.array(M, dtype=dtype)[()]

        # 调用 Bartlett 窗函数生成窗口向量 w
        w = bartlett(scalar)
        
        # 根据数据类型确定参考数据类型 ref_dtype
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        
        # 断言生成的窗口向量 w 的数据类型与参考数据类型 ref_dtype 相同
        assert w.dtype == ref_dtype

        # 检查窗口向量 w 是否对称
        assert_equal(w, flipud(w))

        # 检查已知情况下的窗口向量 w 的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            # 对比窗口向量 w 的总和与预期值 4.4444（保留 4 位小数）
            assert_almost_equal(np.sum(w, axis=0), 4.4444, 4)

    # 使用 Blackman 窗函数测试函数
    def test_blackman(self, dtype: str, M: int) -> None:
        # 将输入的 M 转换为指定数据类型的标量
        scalar = np.array(M, dtype=dtype)[()]

        # 调用 Blackman 窗函数生成窗口向量 w
        w = blackman(scalar)
        
        # 根据数据类型确定参考数据类型 ref_dtype
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        
        # 断言生成的窗口向量 w 的数据类型与参考数据类型 ref_dtype 相同
        assert w.dtype == ref_dtype

        # 检查窗口向量 w 是否对称
        assert_equal(w, flipud(w))

        # 检查已知情况下的窗口向量 w 的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            # 对比窗口向量 w 的总和与预期值 3.7800（保留 4 位小数）
            assert_almost_equal(np.sum(w, axis=0), 3.7800, 4)

    # 使用 Kaiser 窗函数测试函数
    def test_kaiser(self, dtype: str, M: int) -> None:
        # 将输入的 M 转换为指定数据类型的标量
        scalar = np.array(M, dtype=dtype)[()]

        # 调用 Kaiser 窗函数生成窗口向量 w，beta 参数设为 0
        w = kaiser(scalar, 0)
        
        # 根据数据类型确定参考数据类型 ref_dtype
        if dtype == "O":
            ref_dtype = np.float64
        else:
            ref_dtype = np.result_type(scalar.dtype, np.float64)
        
        # 断言生成的窗口向量 w 的数据类型与参考数据类型 ref_dtype 相同
        assert w.dtype == ref_dtype

        # 检查窗口向量 w 是否对称
        assert_equal(w, flipud(w))

        # 检查已知情况下的窗口向量 w 的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            # 对比窗口向量 w 的总和与预期值 10，允许误差为 1e-15
            assert_almost_equal(np.sum(w, axis=0), 10, 15)
class TestTrapezoid:

    def test_simple(self):
        x = np.arange(-10, 10, .1)
        r = trapezoid(np.exp(-.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
        # 检查正态分布的积分是否等于1
        assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None,:, None] + z[None, None,:]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # 对于 n 维的 `x`
        r = trapezoid(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapezoid(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapezoid(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)

        # 对于 1 维的 `x`
        r = trapezoid(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapezoid(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapezoid(q, x=z, axis=2)
        assert_almost_equal(r, qz)

    def test_masked(self):
        # 测试掩码数组在函数处理时表现为掩码部分值为0的效果
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0  # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        assert_almost_equal(trapezoid(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapezoid(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        assert_almost_equal(trapezoid(y, xm), r)


class TestSinc:

    def test_simple(self):
        assert_(sinc(0) == 1)
        w = sinc(np.linspace(-1, 1, 100))
        # 检查对称性
        assert_array_almost_equal(w, flipud(w), 7)

    def test_array_like(self):
        x = [0, 0.5]
        y1 = sinc(np.array(x))
        y2 = sinc(list(x))
        y3 = sinc(tuple(x))
        assert_array_equal(y1, y2)
        assert_array_equal(y1, y3)


class TestUnique:

    def test_simple(self):
        x = np.array([4, 3, 2, 1, 1, 2, 3, 4, 0])
        assert_(np.all(unique(x) == [0, 1, 2, 3, 4]))
        assert_(unique(np.array([1, 1, 1, 1, 1])) == np.array([1]))
        x = ['widget', 'ham', 'foo', 'bar', 'foo', 'ham']
        assert_(np.all(unique(x) == ['bar', 'foo', 'ham', 'widget']))
        x = np.array([5 + 6j, 1 + 1j, 1 + 10j, 10, 5 + 6j])
        assert_(np.all(unique(x) == [1 + 1j, 1 + 10j, 5 + 6j, 10]))


class TestCheckFinite:
    def test_simple(self):
        # 创建一个包含整数的列表
        a = [1, 2, 3]
        # 创建一个包含整数和np.inf的列表
        b = [1, 2, np.inf]
        # 创建一个包含整数和np.nan的列表
        c = [1, 2, np.nan]
        # 检查数组a是否包含有限元素，如果不是，则抛出值错误
        np.asarray_chkfinite(a)
        # 断言调用np.asarray_chkfinite(b)会抛出值错误
        assert_raises(ValueError, np.asarray_chkfinite, b)
        # 断言调用np.asarray_chkfinite(c)会抛出值错误
        assert_raises(ValueError, np.asarray_chkfinite, c)

    def test_dtype_order(self):
        # 回归测试，检查在缺少dtype和order参数时的行为
        # 创建一个包含整数的列表a
        a = [1, 2, 3]
        # 调用np.asarray_chkfinite，将列表a转换为一个包含有限元素的数组，指定了数据类型为np.float64，使用F顺序
        a = np.asarray_chkfinite(a, order='F', dtype=np.float64)
        # 断言数组a的数据类型为np.float64
        assert_(a.dtype == np.float64)
class TestCorrCoef:
    # 定义测试用例中的样本数据 A
    A = np.array(
        [[0.15391142, 0.18045767, 0.14197213],
         [0.70461506, 0.96474128, 0.27906989],
         [0.9297531, 0.32296769, 0.19267156]])
    # 定义测试用例中的样本数据 B
    B = np.array(
        [[0.10377691, 0.5417086, 0.49807457],
         [0.82872117, 0.77801674, 0.39226705],
         [0.9314666, 0.66800209, 0.03538394]])
    # 预期结果 1，用于验证单个输入的相关系数计算结果
    res1 = np.array(
        [[1., 0.9379533, -0.04931983],
         [0.9379533, 1., 0.30007991],
         [-0.04931983, 0.30007991, 1.]])
    # 预期结果 2，用于验证两个输入的相关系数计算结果
    res2 = np.array(
        [[1., 0.9379533, -0.04931983, 0.30151751, 0.66318558, 0.51532523],
         [0.9379533, 1., 0.30007991, -0.04781421, 0.88157256, 0.78052386],
         [-0.04931983, 0.30007991, 1., -0.96717111, 0.71483595, 0.83053601],
         [0.30151751, -0.04781421, -0.96717111, 1., -0.51366032, -0.66173113],
         [0.66318558, 0.88157256, 0.71483595, -0.51366032, 1., 0.98317823],
         [0.51532523, 0.78052386, 0.83053601, -0.66173113, 0.98317823, 1.]])

    # 测试：验证非数组输入的相关系数计算
    def test_non_array(self):
        assert_almost_equal(np.corrcoef([0, 1, 0], [1, 0, 1]),
                            [[1., -1.], [-1.,  1.]])

    # 测试：验证简单的相关系数计算
    def test_simple(self):
        tgt1 = corrcoef(self.A)
        assert_almost_equal(tgt1, self.res1)
        assert_(np.all(np.abs(tgt1) <= 1.0))

        tgt2 = corrcoef(self.A, self.B)
        assert_almost_equal(tgt2, self.res2)
        assert_(np.all(np.abs(tgt2) <= 1.0))

    # 测试：验证 ddof 参数对相关系数计算的影响
    def test_ddof(self):
        # ddof 引发 DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, ddof=-1)
            sup.filter(DeprecationWarning)
            # ddof 对函数几乎没有影响
            assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)
            assert_almost_equal(corrcoef(self.A, ddof=3), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=3), self.res2)

    # 测试：验证 bias 参数对相关系数计算的影响
    def test_bias(self):
        # bias 引发 DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)
            assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)
            sup.filter(DeprecationWarning)
            # bias 对函数几乎没有影响
            assert_almost_equal(corrcoef(self.A, bias=1), self.res1)

    # 测试：验证复杂输入的相关系数计算
    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)
        tgt = np.array([[1., -1.j], [1.j, 1.]])
        assert_allclose(res, tgt)
        assert_(np.all(np.abs(res) <= 1.0))

    # 测试：验证两个输入的相关系数计算
    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(np.corrcoef(x, y), np.array([[1., -1.j], [1.j, 1.]]))
    # 测试空输入情况的函数
    def test_empty(self):
        # 捕获运行时警告，并记录它们
        with warnings.catch_warnings(record=True):
            # 设置警告过滤器，始终触发 RuntimeWarning
            warnings.simplefilter('always', RuntimeWarning)
            # 断言计算空数组的相关系数返回 NaN
            assert_array_equal(corrcoef(np.array([])), np.nan)
            # 断言计算空数组（0行2列）的相关系数返回空数组（0行0列）
            assert_array_equal(corrcoef(np.array([]).reshape(0, 2)),
                               np.array([]).reshape(0, 0))
            # 断言计算空数组（2行0列）的相关系数返回包含 NaN 的2x2数组
            assert_array_equal(corrcoef(np.array([]).reshape(2, 0)),
                               np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    # 测试极端情况的函数
    def test_extreme(self):
        # 定义一个极端的输入数组 x
        x = [[1e-100, 1e100], [1e100, 1e-100]]
        # 在处理过程中，如果有任何异常，将其视为错误
        with np.errstate(all='raise'):
            # 计算数组 x 的相关系数
            c = corrcoef(x)
        # 断言计算结果 c 与预期的相等
        assert_array_almost_equal(c, np.array([[1., -1.], [-1., 1.]]))
        # 断言计算结果 c 的绝对值均不大于 1.0
        assert_(np.all(np.abs(c) <= 1.0))

    # 使用不同数据类型进行相关系数计算的参数化测试
    @pytest.mark.parametrize("test_type", [np.half, np.single, np.double, np.longdouble])
    def test_corrcoef_dtype(self, test_type):
        # 将输入数组 A 转换为指定的数据类型 test_type
        cast_A = self.A.astype(test_type)
        # 计算指定数据类型 test_type 的相关系数
        res = corrcoef(cast_A, dtype=test_type)
        # 断言结果 res 的数据类型与 test_type 相等
        assert test_type == res.dtype


注释：
这些注释说明了每个测试函数的功能和每行代码的作用，以及测试中使用的特定条件和断言。
class TestCov:
    # 定义测试用例中的各种输入数据和预期输出结果
    x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
    res1 = np.array([[1., -1.], [-1., 1.]])
    x2 = np.array([0.0, 1.0, 2.0], ndmin=2)
    frequencies = np.array([1, 4, 1])
    x2_repeats = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [2.0]]).T
    res2 = np.array([[0.4, -0.4], [-0.4, 0.4]])
    unit_frequencies = np.ones(3, dtype=np.int_)
    weights = np.array([1.0, 4.0, 1.0])
    res3 = np.array([[2. / 3., -2. / 3.], [-2. / 3., 2. / 3.]])
    unit_weights = np.ones(3)
    x3 = np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])

    def test_basic(self):
        # 测试基本情况下协方差函数的结果是否符合预期
        assert_allclose(cov(self.x1), self.res1)

    def test_complex(self):
        # 测试复杂输入情况下协方差函数的结果是否符合预期
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = np.array([[1., -1.j], [1.j, 1.]])
        assert_allclose(cov(x), res)
        assert_allclose(cov(x, aweights=np.ones(3)), res)

    def test_xy(self):
        # 测试两个不同变量之间的协方差计算是否正确
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(cov(x, y), np.array([[1., -1.j], [1.j, 1.]]))

    def test_empty(self):
        # 测试空数组输入时协方差函数的行为是否符合预期
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(cov(np.array([])), np.nan)
            assert_array_equal(cov(np.array([]).reshape(0, 2)),
                               np.array([]).reshape(0, 0))
            assert_array_equal(cov(np.array([]).reshape(2, 0)),
                               np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_wrong_ddof(self):
        # 测试不正确的自由度参数设置时协方差函数是否能正确处理
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(cov(self.x1, ddof=5),
                               np.array([[np.inf, -np.inf],
                                         [-np.inf, np.inf]]))

    def test_1D_rowvar(self):
        # 测试在行变量模式下计算一维数据的协方差是否和列变量模式一致
        assert_allclose(cov(self.x3), cov(self.x3, rowvar=False))
        y = np.array([0.0780, 0.3107, 0.2111, 0.0334, 0.8501])
        assert_allclose(cov(self.x3, y), cov(self.x3, y, rowvar=False))

    def test_1D_variance(self):
        # 测试计算一维数据方差时是否和 NumPy 的 var 函数结果一致
        assert_allclose(cov(self.x3, ddof=1), np.var(self.x3, ddof=1))

    def test_fweights(self):
        # 测试使用频率权重进行协方差计算时的各种情况是否能正确处理
        assert_allclose(cov(self.x2, fweights=self.frequencies),
                        cov(self.x2_repeats))
        assert_allclose(cov(self.x1, fweights=self.frequencies),
                        self.res2)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies),
                        self.res1)
        nonint = self.frequencies + 0.5
        assert_raises(TypeError, cov, self.x1, fweights=nonint)
        f = np.ones((2, 3), dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = np.ones(2, dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = -1 * np.ones(3, dtype=np.int_)
        assert_raises(ValueError, cov, self.x1, fweights=f)
    # 测试带加权系数的协方差计算函数
    def test_aweights(self):
        # 断言计算出的协方差与预期结果 self.res3 很接近，使用加权系数 self.weights
        assert_allclose(cov(self.x1, aweights=self.weights), self.res3)
        # 断言带有 3.0 倍加权系数的协方差与未加权系数时的协方差结果一致
        assert_allclose(cov(self.x1, aweights=3.0 * self.weights),
                        cov(self.x1, aweights=self.weights))
        # 断言使用单位权重时的协方差与预期结果 self.res1 很接近
        assert_allclose(cov(self.x1, aweights=self.unit_weights), self.res1)
        # 创建一个形状为 (2, 3) 的全一数组 w，并断言传递该数组会引发 RuntimeError
        w = np.ones((2, 3))
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        # 创建一个长度为 2 的全一数组 w，并断言传递该数组会引发 RuntimeError
        w = np.ones(2)
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        # 创建一个长度为 3 的全负一数组 w，并断言传递该数组会引发 ValueError
        w = -1.0 * np.ones(3)
        assert_raises(ValueError, cov, self.x1, aweights=w)

    # 测试同时使用单位频率权重和加权系数的协方差计算函数
    def test_unit_fweights_and_aweights(self):
        # 断言同时使用频率权重 self.frequencies 和单位加权系数 self.unit_weights 时的协方差与预期结果 self.x2_repeats 很接近
        assert_allclose(cov(self.x2, fweights=self.frequencies,
                            aweights=self.unit_weights),
                        cov(self.x2_repeats))
        # 断言同时使用频率权重 self.frequencies 和单位加权系数 self.unit_weights 时的协方差与预期结果 self.res2 很接近
        assert_allclose(cov(self.x1, fweights=self.frequencies,
                            aweights=self.unit_weights),
                        self.res2)
        # 断言同时使用单位频率权重 self.unit_frequencies 和单位加权系数 self.unit_weights 时的协方差与预期结果 self.res1 很接近
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.unit_weights),
                        self.res1)
        # 断言同时使用单位频率权重 self.unit_frequencies 和加权系数 self.weights 时的协方差与预期结果 self.res3 很接近
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.weights),
                        self.res3)
        # 断言同时使用单位频率权重 self.unit_frequencies 和 3.0 倍加权系数 self.weights 时的协方差与未加权系数时的协方差结果一致
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=3.0 * self.weights),
                        cov(self.x1, aweights=self.weights))
        # 断言同时使用单位频率权重 self.unit_frequencies 和单位加权系数 self.unit_weights 时的协方差与预期结果 self.res1 很接近
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies,
                            aweights=self.unit_weights),
                        self.res1)

    # 使用参数化测试类型 test_type，测试不同数据类型 test_type 下的协方差计算函数
    @pytest.mark.parametrize("test_type", [np.half, np.single, np.double, np.longdouble])
    def test_cov_dtype(self, test_type):
        # 将 self.x1 转换为指定的数据类型 test_type
        cast_x1 = self.x1.astype(test_type)
        # 使用指定数据类型 test_type 计算协方差，预期结果存储在 res 中
        res = cov(cast_x1, dtype=test_type)
        # 断言计算出的协方差的数据类型与输入的 test_type 相符
        assert test_type == res.dtype
class Test_I0:

    def test_simple(self):
        # 断言 i0(0.5) 几乎等于给定的数值
        assert_almost_equal(
            i0(0.5),
            np.array(1.0634833707413234))

        # 因为实现是分段的，需要至少一个大于 8 的测试
        A = np.array([0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0])
        expected = np.array([1.06307822, 1.12518299, 1.01214991, 1.00006049, 2815.71662847])
        # 断言 i0(A) 几乎等于预期的数组
        assert_almost_equal(i0(A), expected)
        # 断言 i0(-A) 几乎等于预期的数组
        assert_almost_equal(i0(-A), expected)

        B = np.array([[0.827002, 0.99959078],
                      [0.89694769, 0.39298162],
                      [0.37954418, 0.05206293],
                      [0.36465447, 0.72446427],
                      [0.48164949, 0.50324519]])
        expected_B = np.array([[1.17843223, 1.26583466],
                               [1.21147086, 1.03898290],
                               [1.03633899, 1.00067775],
                               [1.03352052, 1.13557954],
                               [1.05884290, 1.06432317]])
        # 断言 i0(B) 几乎等于预期的二维数组
        assert_almost_equal(i0(B), expected_B)
        # Regression test for gh-11205
        # 对于单独测试的回归测试
        i0_0 = np.i0([0.])
        # 断言 i0_0 的形状为 (1,)
        assert_equal(i0_0.shape, (1,))
        # 断言 np.i0([0.]) 几乎等于数组 [1.]
        assert_array_equal(np.i0([0.]), np.array([1.]))

    def test_non_array(self):
        a = np.arange(4)

        class array_like:
            __array_interface__ = a.__array_interface__

            def __array_wrap__(self, arr, context, return_scalar):
                return self

        # 例如，pandas Series 通过 array-wrap 在 ufunc 调用中保持有效
        assert isinstance(np.abs(array_like()), array_like)
        exp = np.i0(a)
        res = np.i0(array_like())
        # 断言 np.i0(array_like()) 等于预期的数组 exp
        assert_array_equal(exp, res)

    def test_complex(self):
        a = np.array([0, 1 + 2j])
        # 使用 pytest 检查复数值时应抛出 TypeError
        with pytest.raises(TypeError, match="i0 not supported for complex values"):
            res = i0(a)


class TestKaiser:

    def test_simple(self):
        # 断言 kaiser(1, 1.0) 的结果是有限的
        assert_(np.isfinite(kaiser(1, 1.0)))
        # 断言 kaiser(0, 1.0) 几乎等于空数组
        assert_almost_equal(kaiser(0, 1.0),
                            np.array([]))
        # 断言 kaiser(2, 1.0) 几乎等于预期的数组
        assert_almost_equal(kaiser(2, 1.0),
                            np.array([0.78984831, 0.78984831]))
        # 断言 kaiser(5, 1.0) 几乎等于预期的数组
        assert_almost_equal(kaiser(5, 1.0),
                            np.array([0.78984831, 0.94503323, 1.,
                                      0.94503323, 0.78984831]))
        # 断言 kaiser(5, 1.56789) 几乎等于预期的数组
        assert_almost_equal(kaiser(5, 1.56789),
                            np.array([0.58285404, 0.88409679, 1.,
                                      0.88409679, 0.58285404]))

    def test_int_beta(self):
        # 测试整数形式的 beta 值
        kaiser(3, 4)


class TestMeshgrid:

    def test_simple(self):
        # 创建 meshgrid
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        # 断言 X 等于预期的二维数组
        assert_array_equal(X, np.array([[1, 2, 3],
                                        [1, 2, 3],
                                        [1, 2, 3],
                                        [1, 2, 3]]))
        # 断言 Y 等于预期的二维数组
        assert_array_equal(Y, np.array([[4, 4, 4],
                                        [5, 5, 5],
                                        [6, 6, 6],
                                        [7, 7, 7]]))
    def test_single_input(self):
        # 单个输入的测试用例
        [X] = meshgrid([1, 2, 3, 4])
        # 验证输出数组与预期数组是否相等
        assert_array_equal(X, np.array([1, 2, 3, 4]))

    def test_no_input(self):
        # 没有输入的测试用例
        args = []
        # 验证空输入情况下的输出数组是否为空
        assert_array_equal([], meshgrid(*args))
        # 验证空输入情况下的输出数组是否为空，并且不进行复制
        assert_array_equal([], meshgrid(*args, copy=False))

    def test_indexing(self):
        # 索引测试用例
        x = [1, 2, 3]
        y = [4, 5, 6, 7]
        [X, Y] = meshgrid(x, y, indexing='ij')
        # 验证输出的 X 数组是否符合预期
        assert_array_equal(X, np.array([[1, 1, 1, 1],
                                        [2, 2, 2, 2],
                                        [3, 3, 3, 3]]))
        # 验证输出的 Y 数组是否符合预期
        assert_array_equal(Y, np.array([[4, 5, 6, 7],
                                        [4, 5, 6, 7],
                                        [4, 5, 6, 7]]))

        # 测试预期的形状：
        z = [8, 9]
        # 验证不同参数组合下输出数组的形状
        assert_(meshgrid(x, y)[0].shape == (4, 3))
        assert_(meshgrid(x, y, indexing='ij')[0].shape == (3, 4))
        assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        assert_(meshgrid(x, y, z, indexing='ij')[0].shape == (3, 4, 2))
        # 验证当使用无效索引时是否引发 ValueError
        assert_raises(ValueError, meshgrid, x, y, indexing='notvalid')

    def test_sparse(self):
        # 稀疏矩阵测试用例
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7], sparse=True)
        # 验证输出的 X 数组是否符合预期
        assert_array_equal(X, np.array([[1, 2, 3]]))
        # 验证输出的 Y 数组是否符合预期
        assert_array_equal(Y, np.array([[4], [5], [6], [7]]))

    def test_invalid_arguments(self):
        # 无效参数测试用例
        # 验证 meshgrid 是否会报告无效参数
        # GitHub 上的回归测试，Issue #4755: https://github.com/numpy/numpy/issues/4755
        assert_raises(TypeError, meshgrid,
                      [1, 2, 3], [4, 5, 6, 7], indices='ij')

    def test_return_type(self):
        # 返回类型测试用例
        # 验证返回数组中的数据类型是否正确
        # GitHub 上的回归测试，Issue #5297: https://github.com/numpy/numpy/issues/5297
        x = np.arange(0, 10, dtype=np.float32)
        y = np.arange(10, 20, dtype=np.float64)

        X, Y = np.meshgrid(x,y)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # 复制测试
        X, Y = np.meshgrid(x,y, copy=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # 稀疏矩阵测试
        X, Y = np.meshgrid(x,y, sparse=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

    def test_writeback(self):
        # Issue 8561
        # 写回测试用例
        X = np.array([1.1, 2.2])
        Y = np.array([3.3, 4.4])
        x, y = np.meshgrid(X, Y, sparse=False, copy=True)

        x[0, :] = 0
        # 验证写回后的第一行是否全部为 0
        assert_equal(x[0, :], 0)
        # 验证写回后的第二行是否恢复为原始数组 X
        assert_equal(x[1, :], X)

    def test_nd_shape(self):
        # 多维形状测试用例
        a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        expected_shape = (2, 1, 3, 4, 5)
        # 验证每个输出数组的形状是否符合预期
        assert_equal(a.shape, expected_shape)
        assert_equal(b.shape, expected_shape)
        assert_equal(c.shape, expected_shape)
        assert_equal(d.shape, expected_shape)
        assert_equal(e.shape, expected_shape)
    # 定义一个测试函数，用于测试多维数组的 meshgrid 函数生成的结果
    def test_nd_values(self):
        # 使用 numpy 的 meshgrid 函数生成三维数组 a, b, c
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])
        # 断言：验证生成的数组 a 是否符合预期值
        assert_equal(a, [[[0, 0, 0]], [[0, 0, 0]]])
        # 断言：验证生成的数组 b 是否符合预期值
        assert_equal(b, [[[1, 1, 1]], [[2, 2, 2]]])
        # 断言：验证生成的数组 c 是否符合预期值
        assert_equal(c, [[[3, 4, 5]], [[3, 4, 5]]])

    # 定义另一个测试函数，用于测试带索引方式的多维数组的 meshgrid 函数生成的结果
    def test_nd_indexing(self):
        # 使用 numpy 的 meshgrid 函数生成带索引的三维数组 a, b, c
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing='ij')
        # 断言：验证生成的数组 a 是否符合预期值
        assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
        # 断言：验证生成的数组 b 是否符合预期值
        assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
        # 断言：验证生成的数组 c 是否符合预期值
        assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])
class TestPiecewise:

    def test_simple(self):
        # Condition is single bool list
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        x = piecewise([0, 0], np.array([True, False]), [1])
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        x = piecewise([0, 0], np.array([1, 0]), [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

        # Condition is a list with lambda function
        x = piecewise([0, 0], [[False, True]], [lambda x:-1])
        assert_array_equal(x, [0, -1])

        # Testing assertion for incorrect number of functions
        assert_raises_regex(ValueError, '1 or 2 functions are expected',
            piecewise, [0, 0], [[False, True]], [])
        assert_raises_regex(ValueError, '1 or 2 functions are expected',
            piecewise, [0, 0], [[False, True]], [1, 2, 3])

    def test_two_conditions(self):
        # Testing piecewise with two conditions
        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_scalar_domains_three_conditions(self):
        # Testing piecewise with scalar domains and three conditions
        x = piecewise(3, [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    def test_default(self):
        # Testing default values in piecewise function
        # No value specified for x[1], should default to 0
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        # Testing piecewise function with 0-dimensional arrays
        x = np.array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [True, False], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

        # Testing piecewise function with 3 ranges
        y = piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    def test_0d_comparison(self):
        # Testing piecewise function with 0-dimensional arrays and comparisons
        x = 3
        y = piecewise(x, [x <= 3, x > 3], [4, 0])  # Should succeed.
        assert_equal(y, 4)

        # Testing piecewise function with 3 ranges
        x = 4
        y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)

        # Testing assertion for incorrect number of functions
        assert_raises_regex(ValueError, '2 or 3 functions are expected',
            piecewise, x, [x <= 3, x > 3], [1])
        assert_raises_regex(ValueError, '2 or 3 functions are expected',
            piecewise, x, [x <= 3, x > 3], [1, 1, 1, 1])

    def test_0d_0d_condition(self):
        # Testing piecewise function with 0-dimensional arrays and conditions
        x = np.array(3)
        c = np.array(x > 3)
        y = piecewise(x, [c], [1, 2])
        assert_equal(y, 2)
    # 定义一个测试方法，用于测试 piecewise 函数处理多维数组和额外功能的情况
    def test_multidimensional_extrafunc(self):
        # 创建一个二维 NumPy 数组 x
        x = np.array([[-2.5, -1.5, -0.5],
                      [0.5, 1.5, 2.5]])
        # 调用 piecewise 函数处理数组 x，根据条件数组和函数列表进行分段处理
        y = piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        # 使用 assert_array_equal 断言函数，验证处理结果 y 与预期的 NumPy 数组相等
        assert_array_equal(y, np.array([[-1., -1., -1.],
                                        [3., 3., 1.]]))

    # 定义一个测试方法，用于测试 piecewise 函数处理子类的情况
    def test_subclasses(self):
        # 定义一个名为 subclass 的子类，继承自 np.ndarray
        class subclass(np.ndarray):
            pass
        # 创建一个长度为 5 的浮点数数组 x，并将其视图类型设置为 subclass
        x = np.arange(5.).view(subclass)
        # 调用 piecewise 函数处理数组 x，根据条件数组和函数列表进行分段处理
        r = piecewise(x, [x<2., x>=4], [-1., 1., 0.])
        # 使用 assert_equal 断言函数，验证处理结果 r 的类型与预期的 subclass 相等
        assert_equal(type(r), subclass)
        # 使用 assert_equal 断言函数，验证处理结果 r 与预期的数组相等
        assert_equal(r, [-1., -1., 0., 0., 1.])
class TestBincount:

    # 测试简单情况下的 np.bincount 函数调用
    def test_simple(self):
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    # 测试包含不同元素的 np.bincount 函数调用
    def test_simple2(self):
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    # 测试带权重的 np.bincount 函数调用
    def test_simple_weight(self):
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    # 测试带权重且包含重复元素的 np.bincount 函数调用
    def test_simple_weight2(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    # 测试带 minlength 参数的 np.bincount 函数调用
    def test_with_minlength(self):
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))
        x = []
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([]))

    # 测试带 minlength 参数且比最大元素值小的 np.bincount 函数调用
    def test_with_minlength_smaller_than_maxvalue(self):
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    # 测试带 minlength 和 weights 参数的 np.bincount 函数调用
    def test_with_minlength_and_weights(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

    # 测试空数组作为输入的 np.bincount 函数调用
    def test_empty(self):
        x = np.array([], dtype=int)
        y = np.bincount(x)
        assert_array_equal(x, y)

    # 测试带 minlength 参数的空数组输入的 np.bincount 函数调用
    def test_empty_with_minlength(self):
        x = np.array([], dtype=int)
        y = np.bincount(x, minlength=5)
        assert_array_equal(y, np.zeros(5, dtype=int))

    # 测试使用不正确类型 minlength 参数的 np.bincount 函数调用
    def test_with_incorrect_minlength(self):
        x = np.array([], dtype=int)
        assert_raises_regex(TypeError,
                            "'str' object cannot be interpreted",
                            lambda: np.bincount(x, minlength="foobar"))
        assert_raises_regex(ValueError,
                            "must not be negative",
                            lambda: np.bincount(x, minlength=-1))

        x = np.arange(5)
        assert_raises_regex(TypeError,
                            "'str' object cannot be interpreted",
                            lambda: np.bincount(x, minlength="foobar"))
        assert_raises_regex(ValueError,
                            "must not be negative",
                            lambda: np.bincount(x, minlength=-1))

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    # 定义测试方法，用于检查数据类型的引用泄漏
    def test_dtype_reference_leaks(self):
        # 标记：gh-6805，指明此测试相关联的GitHub问题编号
        # 获取 np.intp 数据类型对象的引用计数
        intp_refcount = sys.getrefcount(np.dtype(np.intp))
        # 获取 np.double 数据类型对象的引用计数
        double_refcount = sys.getrefcount(np.dtype(np.double))

        # 循环执行 10 次，调用 np.bincount([1, 2, 3]) 进行计算
        for j in range(10):
            np.bincount([1, 2, 3])
        
        # 断言：np.intp 数据类型对象的引用计数未变
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        # 断言：np.double 数据类型对象的引用计数未变
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

        # 再次循环执行 10 次，调用 np.bincount([1, 2, 3], [4, 5, 6]) 进行计算
        for j in range(10):
            np.bincount([1, 2, 3], [4, 5, 6])
        
        # 断言：np.intp 数据类型对象的引用计数仍未变
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        # 断言：np.double 数据类型对象的引用计数仍未变
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

    # 使用 pytest 标记参数化测试，参数为 vals，接收不同的输入值
    @pytest.mark.parametrize("vals", [[[2, 2]], 2])
    # 定义测试方法，用于检查输入不是 1-D 的情况
    def test_error_not_1d(self, vals):
        # 测试将 vals 转换为 numpy 数组
        vals_arr = np.asarray(vals)
        # 使用 assert_raises 断言 ValueError 异常会被抛出
        with assert_raises(ValueError):
            np.bincount(vals_arr)
        # 同样使用 assert_raises 断言 ValueError 异常会被抛出
        with assert_raises(ValueError):
            np.bincount(vals)
# 定义一个测试类 TestInterp，用于测试 interp 函数的各种情况和行为
class TestInterp:

    # 测试函数，用于测试 interp 函数在抛出异常时的行为
    def test_exceptions(self):
        # 断言调用 interp 函数时会引发 ValueError 异常，期望输入参数中包含 0 和空列表
        assert_raises(ValueError, interp, 0, [], [])
        # 断言调用 interp 函数时会引发 ValueError 异常，期望输入参数中包含 0 和一个元素的列表以及一个长度为 2 的列表
        assert_raises(ValueError, interp, 0, [0], [1, 2])
        # 断言调用 interp 函数时会引发 ValueError 异常，期望输入参数中包含 0、包含两个元素的列表以及一个长度为 2 的列表，并且设置 period 参数为 0
        assert_raises(ValueError, interp, 0, [0, 1], [1, 2], period=0)
        # 断言调用 interp 函数时会引发 ValueError 异常，期望输入参数中包含 0、空列表和空列表，并且设置 period 参数为 360
        assert_raises(ValueError, interp, 0, [], [], period=360)
        # 断言调用 interp 函数时会引发 ValueError 异常，期望输入参数中包含 0、一个元素的列表和一个长度为 2 的列表，并且设置 period 参数为 360
        assert_raises(ValueError, interp, 0, [0], [1, 2], period=360)

    # 测试函数，用于测试 interp 函数在基本情况下的行为
    def test_basic(self):
        # 创建一个包含五个元素的等差数列 x：[0.0, 0.25, 0.5, 0.75, 1.0]
        x = np.linspace(0, 1, 5)
        # 创建一个包含五个元素的等差数列 y：[0.0, 0.25, 0.5, 0.75, 1.0]
        y = np.linspace(0, 1, 5)
        # 创建一个包含五十个元素的等差数列 x0：[0.0, 0.02040816, 0.04081633, ..., 0.97959184, 1.0]
        x0 = np.linspace(0, 1, 50)
        # 断言调用 np.interp 函数时，返回值与 x0 数列相同
        assert_almost_equal(np.interp(x0, x, y), x0)

    # 测试函数，用于测试 interp 函数在左右边界行为上的不同表现
    def test_right_left_behavior(self):
        # 使用循环测试不同大小的 xp 数组，以检验不同的代码路径
        for size in range(1, 10):
            # 创建一个长度为 size 的双精度浮点数等差数列 xp：[0.0, 1.0, 2.0, ..., size-1]
            xp = np.arange(size, dtype=np.double)
            # 创建一个长度为 size 的双精度浮点数全为 1 的数组 yp：[1.0, 1.0, ..., 1.0]
            yp = np.ones(size, dtype=np.double)
            # 创建一个包含四个特定点的数组 incpts：[-1.0, 0.0, size-1.0, size]
            incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
            # 创建 decpts 数组，其值为 incpts 数组的逆序
            decpts = incpts[::-1]

            # 使用 interp 函数计算 incpts 数组的插值结果 incres
            incres = interp(incpts, xp, yp)
            # 使用 interp 函数计算 decpts 数组的插值结果 decres
            decres = interp(decpts, xp, yp)
            # 创建一个长度为 4 的目标数组 inctgt，其元素都为 1.0
            inctgt = np.array([1, 1, 1, 1], dtype=float)
            # 创建一个长度为 4 的逆序目标数组 dectgt，其元素也都为 1.0
            dectgt = inctgt[::-1]
            # 断言 incres 和 inctgt 数组相等
            assert_equal(incres, inctgt)
            # 断言 decres 和 dectgt 数组相等
            assert_equal(decres, dectgt)

            # 使用 interp 函数计算带有 left=0 参数的 incpts 数组的插值结果 incres
            incres = interp(incpts, xp, yp, left=0)
            # 使用 interp 函数计算带有 left=0 参数的 decpts 数组的插值结果 decres
            decres = interp(decpts, xp, yp, left=0)
            # 创建一个包含四个特定点的目标数组 inctgt，其第一个元素为 0.0，其余为 1.0
            inctgt = np.array([0, 1, 1, 1], dtype=float)
            # 创建一个逆序的目标数组 dectgt，其第一个元素为 0.0，其余为 1.0
            dectgt = inctgt[::-1]
            # 断言 incres 和 inctgt 数组相等
            assert_equal(incres, inctgt)
            # 断言 decres 和 dectgt 数组相等
            assert_equal(decres, dectgt)

            # 使用 interp 函数计算带有 right=2 参数的 incpts 数组的插值结果 incres
            incres = interp(incpts, xp, yp, right=2)
            # 使用 interp 函数计算带有 right=2 参数的 decpts 数组的插值结果 decres
            decres = interp(decpts, xp, yp, right=2)
            # 创建一个长度为 4 的目标数组 inctgt，前三个元素为 1.0，最后一个元素为 2.0
            inctgt = np.array([1, 1, 1, 2], dtype=float)
            # 创建一个逆序的目标数组 dectgt，前三个元素为 1.0，最后一个元素为 2.0
            dectgt = inctgt[::-1]
            # 断言 incres 和 inctgt 数组相等
            assert_equal(incres, inctgt)
            # 断言 decres 和 dectgt 数组相等
            assert_equal(decres, dectgt)

            # 使用 interp 函数计算带有 left=0 和 right=2 参数的 incpts 数组的插值结果 incres
            incres = interp(incpts, xp, yp, left=0, right=2)
            # 使用 interp 函数计算带有 left=0 和 right=2 参数的 decpts 数组的插值结果 decres
            decres = interp(decpts, xp, yp, left=0, right=2)
            # 创建一个包含四个特定点的目标数组 inctgt，其元素依次为 0.0, 1.0, 1.0, 2.0
            inctgt = np.array([0, 1, 1, 2], dtype=float)
            # 创建一个逆序的目标数组 dectgt，其元素依次为 0.0, 1.0, 1.0, 2.0
            dectgt = inctgt[::-1]
            # 断言 incres 和 inctgt 数组相等
            assert_equal(incres, inctgt)
            # 断言 decres 和 dectgt 数组相等
            assert_equal(decres, dectgt)

    # 测试函数，用于测试 interp 函数在标量插值点的行为
    def test_scalar_interpolation_point(self):
        # 创建一个包含五个元素的等差数列 x：[0.0, 0.25, 0.5, 0.75, 1.0]
        x = np.linspace(0, 1, 5)
        # 创建一个包含五个元素的等差数列 y：[0.0, 0.25, 0.5, 0.75, 1.0]
        y = np.linspace(0, 1, 5)
        # 使用 np.interp 函数计算标量
    @pytest.fixture(params=[
        lambda x: np.float64(x),
        lambda x: _make_complex(x, 0),
        lambda x: _make_complex(0, x),
        lambda x: _make_complex(x, np.multiply(x, -2))
    ], ids=[
        'real',
        'complex-real',
        'complex-imag',
        'complex-both'
    ])
    def sc(self, request):
        """ 
        定义一个 Pytest 的 Fixture，用于提供不同的缩放函数参数化
        """
        return request.param

    def test_non_finite_any_nan(self, sc):
        """ 
        测试确保 NaN 被传播
        """
        assert_equal(np.interp(0.5, [np.nan,      1], sc([     0,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0, np.nan], sc([     0,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0,      1], sc([np.nan,     10])), sc(np.nan))
        assert_equal(np.interp(0.5, [     0,      1], sc([     0, np.nan])), sc(np.nan))

    def test_non_finite_inf(self, sc):
        """ 
        测试确保在无穷边界之间插值得到 NaN
        """
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([      0,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0,       1], sc([-np.inf, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0,       1], sc([+np.inf, -np.inf])), sc(np.nan))

        # 除非 y 值相等
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([     10,      10])), sc(10))

    def test_non_finite_half_inf_xf(self, sc):
        """ 
        测试当两个轴都有一个边界在无穷时插值得到 NaN
        """
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([-np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([+np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([      0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf,       1], sc([      0, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([-np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([+np.inf,      10])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([      0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([      0, +np.inf])), sc(np.nan))

    def test_non_finite_half_inf_x(self, sc):
        """ 
        测试当 x 轴有一个边界在无穷时的插值
        """
        assert_equal(np.interp(0.5, [-np.inf, -np.inf], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [-np.inf, 1      ], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [      0, +np.inf], sc([0, 10])), sc(0))
        assert_equal(np.interp(0.5, [+np.inf, +np.inf], sc([0, 10])), sc(0))
    def test_non_finite_half_inf_f(self, sc):
        """ Test interp where the f axis has a bound at inf """
        # 检验在 f 轴有无穷大边界时的插值
        assert_equal(np.interp(0.5, [0, 1], sc([      0, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([      0, +np.inf])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf,      10])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf,      10])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, +np.inf])), sc(+np.inf))

    def test_complex_interp(self):
        # 测试复数插值
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5))*1.0j
        x0 = 0.3
        y0 = x0 + (1+x0)*1.0j
        assert_almost_equal(np.interp(x0, x, y), y0)
        # 测试复数左右边界
        x0 = -1
        left = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, left=left), left)
        x0 = 2.0
        right = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, right=right), right)
        # 测试复数非有限值
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2+1j, np.inf, 4]
        y = [1, 2+1j, np.inf+0.5j, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), y)
        # 测试复数周期性插值
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5+1.0j, 10+2j, 3+3j, 4+4j]
        y = [7.5+1.5j, 5.+1.0j, 8.75+1.75j, 6.25+1.25j, 3.+3j, 3.25+3.25j,
             3.5+3.5j, 3.75+3.75j]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)

    def test_zero_dimensional_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.array(.3)
        assert_almost_equal(np.interp(x0, x, y), x0)

        xp = np.array([0, 2, 4])
        fp = np.array([1, -1, 1])

        actual = np.interp(np.array(1), xp, fp)
        assert_equal(actual, 0)
        assert_(isinstance(actual, np.float64))

        actual = np.interp(np.array(4.5), xp, fp, period=4)
        assert_equal(actual, 0.5)
        assert_(isinstance(actual, np.float64))

    def test_if_len_x_is_small(self):
        xp = np.arange(0, 10, 0.0001)
        fp = np.sin(xp)
        assert_almost_equal(np.interp(np.pi, xp, fp), 0.0)

    def test_period(self):
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5, 10, 3, 4]
        y = [7.5, 5., 8.75, 6.25, 3., 3.25, 3.5, 3.75]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
        x = np.array(x, order='F').reshape(2, -1)
        y = np.array(y, order='C').reshape(2, -1)
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
class TestPercentile:

    def test_basic(self):
        # 创建一个长度为8的一维数组，元素为0到3.5的等差数列
        x = np.arange(8) * 0.5
        # 断言：计算数组x的0%分位数应为0.0
        assert_equal(np.percentile(x, 0), 0.)
        # 断言：计算数组x的100%分位数应为3.5
        assert_equal(np.percentile(x, 100), 3.5)
        # 断言：计算数组x的50%分位数应为1.75
        assert_equal(np.percentile(x, 50), 1.75)
        # 将数组x的第1个元素设为NaN
        x[1] = np.nan
        # 断言：计算数组x的0%分位数应为NaN
        assert_equal(np.percentile(x, 0), np.nan)
        # 断言：计算数组x的0%分位数（使用nearest方法）应为NaN
        assert_equal(np.percentile(x, 0, method='nearest'), np.nan)
        # 断言：计算数组x的0%分位数（使用inverted_cdf方法）应为NaN
        assert_equal(np.percentile(x, 0, method='inverted_cdf'), np.nan)
        # 断言：使用权重计算数组x的0%分位数（使用inverted_cdf方法）应为NaN
        assert_equal(
            np.percentile(x, 0, method='inverted_cdf',
                          weights=np.ones_like(x)),
            np.nan,
        )

    def test_fraction(self):
        # 创建一个分数列表，分子为0到7，分母为2
        x = [Fraction(i, 2) for i in range(8)]

        # 计算分数列表x的0%分位数，应为Fraction(0)
        p = np.percentile(x, Fraction(0))
        assert_equal(p, Fraction(0))
        assert_equal(type(p), Fraction)

        # 计算分数列表x的100%分位数，应为Fraction(7/2)
        p = np.percentile(x, Fraction(100))
        assert_equal(p, Fraction(7, 2))
        assert_equal(type(p), Fraction)

        # 计算分数列表x的50%分位数，应为Fraction(7/4)
        p = np.percentile(x, Fraction(50))
        assert_equal(p, Fraction(7, 4))
        assert_equal(type(p), Fraction)

        # 计算分数列表x的50%分位数（作为数组），应为array([Fraction(7/4)])
        p = np.percentile(x, [Fraction(50)])
        assert_equal(p, np.array([Fraction(7, 4)]))
        assert_equal(type(p), np.ndarray)

    def test_api(self):
        # 创建一个长度为5的全1数组
        d = np.ones(5)
        # 调用np.percentile计算数组d的5%分位数，忽略其他参数
        np.percentile(d, 5, None, None, False)
        # 调用np.percentile计算数组d的5%分位数，使用线性插值
        np.percentile(d, 5, None, None, False, 'linear')
        # 创建一个形状为(1,)的全1数组
        o = np.ones((1,))
        # 调用np.percentile计算数组d的5%分位数，使用线性插值和自定义输出数组o
        np.percentile(d, 5, None, o, False, 'linear')

    def test_complex(self):
        # 创建一个包含复数的数组，数据类型为复数
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='G')
        # 断言：尝试计算复数数组arr_c的分位数会引发TypeError异常
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        # 同上，使用不同的复数数据类型
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='D')
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='F')
        assert_raises(TypeError, np.percentile, arr_c, 0.5)

    def test_2D(self):
        # 创建一个二维数组
        x = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [4, 4, 3],
                      [1, 1, 1],
                      [1, 1, 1]])
        # 断言：计算二维数组x每列的50%分位数，应为[1, 1, 1]
        assert_array_equal(np.percentile(x, 50, axis=0), [1, 1, 1])

    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    def test_linear_nan_1D(self, dtype):
        # 使用np.asarray创建一个包含NaN的一维数组，数据类型为参数dtype指定的类型
        arr = np.asarray([15.0, np.nan, 35.0, 40.0, 50.0], dtype=dtype)
        # 调用np.percentile计算数组arr的40.0%分位数，使用线性插值方法，预期结果为NaN
        res = np.percentile(
            arr,
            40.0,
            method="linear")
        # 断言：计算结果应为NaN
        np.testing.assert_equal(res, np.nan)
        # 断言：计算结果的数据类型应与输入数组arr的数据类型相同
        np.testing.assert_equal(res.dtype, arr.dtype)

    # 类变量，包含整数类型代码对和浮点数类型np.float64的组合
    H_F_TYPE_CODES = [(int_type, np.float64)
                      for int_type in np.typecodes["AllInteger"]
                      ] + [(np.float16, np.float16),
                           (np.float32, np.float32),
                           (np.float64, np.float64),
                           (np.longdouble, np.longdouble),
                           (np.dtype("O"), np.float64)]
    # 使用 pytest 的 parametrize 装饰器为函数 test_linear_interpolation 添加多个参数组合的测试用例
    @pytest.mark.parametrize(["function", "quantile"],
                             [(np.quantile, 0.4),
                              (np.percentile, 40.0)])
    # 使用 pytest 的 parametrize 装饰器为函数 test_linear_interpolation 添加参数化测试用例，参数来自 H_F_TYPE_CODES
    @pytest.mark.parametrize(["input_dtype", "expected_dtype"], H_F_TYPE_CODES)
    # 使用 pytest 的 parametrize 装饰器为函数 test_linear_interpolation 添加多个参数组合的测试用例
    @pytest.mark.parametrize(["method", "weighted", "expected"],
                              [("inverted_cdf", False, 20),
                               ("inverted_cdf", True, 20),
                               ("averaged_inverted_cdf", False, 27.5),
                               ("closest_observation", False, 20),
                               ("interpolated_inverted_cdf", False, 20),
                               ("hazen", False, 27.5),
                               ("weibull", False, 26),
                               ("linear", False, 29),
                               ("median_unbiased", False, 27),
                               ("normal_unbiased", False, 27.125),
                               ])
    # 定义测试函数 test_linear_interpolation，用于测试线性插值的各种方法
    def test_linear_interpolation(self,
                                  function,
                                  quantile,
                                  method,
                                  weighted,
                                  expected,
                                  input_dtype,
                                  expected_dtype):
        # 调整 expected_dtype 为指定的 numpy 数据类型
        expected_dtype = np.dtype(expected_dtype)
        # 如果当前 numpy 推广状态为 "legacy"，则使用 np.promote_types 将 expected_dtype 提升为 np.float64
        if np._get_promotion_state() == "legacy":
            expected_dtype = np.promote_types(expected_dtype, np.float64)

        # 创建输入数组 arr，根据指定的 input_dtype 类型
        arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
        # 根据 weighted 参数决定是否创建权重数组 weights
        weights = np.ones_like(arr) if weighted else None
        # 如果 input_dtype 是 np.longdouble 类型，则根据 function 类型选择测试函数
        if input_dtype is np.longdouble:
            if function is np.quantile:
                # 对于 np.quantile，需要精确匹配到 0.4，因此使用 input_dtype("0.4") 进行设置
                quantile = input_dtype("0.4")
            # 对于其他情况，使用 np.testing.assert_array_almost_equal_nulp 进行测试
            test_function = np.testing.assert_almost_equal
        else:
            # 对于非 np.longdouble 类型，使用 np.testing.assert_array_almost_equal_nulp 进行测试
            test_function = np.testing.assert_array_almost_equal_nulp

        # 调用指定的 function 函数，计算 actual 值
        actual = function(arr, quantile, method=method, weights=weights)

        # 使用 test_function 进行 actual 与期望值 expected 的测试
        test_function(actual, expected_dtype.type(expected))

        # 如果 method 是 ["inverted_cdf", "closest_observation"] 中的一种
        if method in ["inverted_cdf", "closest_observation"]:
            # 如果 input_dtype 是 "O"，则断言 actual 转换为 np.float64 类型
            if input_dtype == "O":
                np.testing.assert_equal(np.asarray(actual).dtype, np.float64)
            else:
                # 否则，断言 actual 与 input_dtype 相同类型
                np.testing.assert_equal(np.asarray(actual).dtype,
                                        np.dtype(input_dtype))
        else:
            # 否则，断言 actual 与 expected_dtype 相同类型
            np.testing.assert_equal(np.asarray(actual).dtype,
                                    np.dtype(expected_dtype))

    # 创建 TYPE_CODES，包含所有整数和浮点数类型，以及 "O" 类型
    TYPE_CODES = np.typecodes["AllInteger"] + np.typecodes["Float"] + "O"

    # 使用 pytest 的 parametrize 装饰器为函数 test_linear_interpolation 添加参数化测试用例，参数为 TYPE_CODES
    @pytest.mark.parametrize("dtype", TYPE_CODES)
    # 定义一个测试方法，用于测试 np.percentile 函数的 'lower' 和 'higher' 方法
    def test_lower_higher(self, dtype):
        # 断言对于一个从 0 到 9 的整数数组，计算第 50% 百分位数时，使用 'lower' 方法应该得到 4
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50,
                                   method='lower'), 4)
        # 断言对于一个从 0 到 9 的整数数组，计算第 50% 百分位数时，使用 'higher' 方法应该得到 5
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50,
                                   method='higher'), 5)

    # 使用 pytest 的参数化标记，测试 np.percentile 函数的 'midpoint' 方法
    @pytest.mark.parametrize("dtype", TYPE_CODES)
    def test_midpoint(self, dtype):
        # 断言对于一个从 0 到 9 的整数数组，计算第 51% 百分位数时，使用 'midpoint' 方法应该得到 4.5
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 51,
                                   method='midpoint'), 4.5)
        # 断言对于一个从 1 到 9 的整数数组，计算第 50% 百分位数时，使用 'midpoint' 方法应该得到 5
        assert_equal(np.percentile(np.arange(9, dtype=dtype) + 1, 50,
                                   method='midpoint'), 5)
        # 断言对于一个从 0 到 10 的整数数组，计算第 51% 百分位数时，使用 'midpoint' 方法应该得到 5.5
        assert_equal(np.percentile(np.arange(11, dtype=dtype), 51,
                                   method='midpoint'), 5.5)
        # 断言对于一个从 0 到 10 的整数数组，计算第 50% 百分位数时，使用 'midpoint' 方法应该得到 5
        assert_equal(np.percentile(np.arange(11, dtype=dtype), 50,
                                   method='midpoint'), 5)

    # 使用 pytest 的参数化标记，测试 np.percentile 函数的 'nearest' 方法
    @pytest.mark.parametrize("dtype", TYPE_CODES)
    def test_nearest(self, dtype):
        # 断言对于一个从 0 到 9 的整数数组，计算第 51% 百分位数时，使用 'nearest' 方法应该得到 5
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 51,
                                   method='nearest'), 5)
        # 断言对于一个从 0 到 9 的整数数组，计算第 49% 百分位数时，使用 'nearest' 方法应该得到 4
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 49,
                                   method='nearest'), 4)

    # 测试线性插值和外推的情况
    def test_linear_interpolation_extrapolation(self):
        # 创建一个包含 5 个随机数的数组
        arr = np.random.rand(5)

        # 计算数组 arr 的第 100% 百分位数，应该等于数组中的最大值
        actual = np.percentile(arr, 100)
        np.testing.assert_equal(actual, arr.max())

        # 计算数组 arr 的第 0% 百分位数，应该等于数组中的最小值
        actual = np.percentile(arr, 0)
        np.testing.assert_equal(actual, arr.min())

    # 测试序列的情况
    def test_sequence(self):
        # 创建一个包含 [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] 的数组 x
        x = np.arange(8) * 0.5
        # 断言计算数组 x 的第 [0%, 100%, 50%] 百分位数时，应该分别得到 [0, 3.5, 1.75]
        assert_equal(np.percentile(x, [0, 100, 50]), [0, 3.5, 1.75])
    # 定义一个测试方法来测试 np.percentile 函数的不同用法
    def test_axis(self):
        # 创建一个 3x4 的数组 x，包含从 0 到 11 的整数
        x = np.arange(12).reshape(3, 4)

        # 断言 np.percentile 对 x 求得的百分位数 (25%, 50%, 100%) 分别为 [2.75, 5.5, 11.0]
        assert_equal(np.percentile(x, (25, 50, 100)), [2.75, 5.5, 11.0])

        # 定义预期的按列计算百分位数结果 r0
        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        # 断言 np.percentile 对 x 按列（axis=0）求得的百分位数与 r0 相等
        assert_equal(np.percentile(x, (25, 50, 100), axis=0), r0)

        # 定义预期的按行计算百分位数结果 r1
        r1 = [[0.75, 1.5, 3], [4.75, 5.5, 7], [8.75, 9.5, 11]]
        # 断言 np.percentile 对 x 按行（axis=1）求得的百分位数与 r1 转置后的结果相等
        assert_equal(np.percentile(x, (25, 50, 100), axis=1), np.array(r1).T)

        # 确保 qth axis 参数始终在第一位，如同 np.array(old_percentile(..))
        x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        # 断言 np.percentile 对 x 求得的 (25%, 50%) 的形状为 (2,)
        assert_equal(np.percentile(x, (25, 50)).shape, (2,))
        # 断言 np.percentile 对 x 求得的 (25%, 50%, 75%) 的形状为 (3,)
        assert_equal(np.percentile(x, (25, 50, 75)).shape, (3,))
        # 断言 np.percentile 对 x 按 axis=0 求得的 (25%, 50%) 的形状为 (2, 4, 5, 6)
        assert_equal(np.percentile(x, (25, 50), axis=0).shape, (2, 4, 5, 6))
        # 断言 np.percentile 对 x 按 axis=1 求得的 (25%, 50%) 的形状为 (2, 3, 5, 6)
        assert_equal(np.percentile(x, (25, 50), axis=1).shape, (2, 3, 5, 6))
        # 断言 np.percentile 对 x 按 axis=2 求得的 (25%, 50%) 的形状为 (2, 3, 4, 6)
        assert_equal(np.percentile(x, (25, 50), axis=2).shape, (2, 3, 4, 6))
        # 断言 np.percentile 对 x 按 axis=3 求得的 (25%, 50%) 的形状为 (2, 3, 4, 5)
        assert_equal(np.percentile(x, (25, 50), axis=3).shape, (2, 3, 4, 5))
        # 断言 np.percentile 对 x 按 axis=1 求得的 (25%, 50%, 75%) 的形状为 (3, 3, 5, 6)
        assert_equal(np.percentile(x, (25, 50, 75), axis=1).shape, (3, 3, 5, 6))
        # 断言 np.percentile 对 x 求得的 (25%, 50%) 使用 method="higher" 的形状为 (2,)
        assert_equal(np.percentile(x, (25, 50), method="higher").shape, (2,))
        # 断言 np.percentile 对 x 求得的 (25%, 50%, 75%) 使用 method="higher" 的形状为 (3,)
        assert_equal(np.percentile(x, (25, 50, 75), method="higher").shape, (3,))
        # 断言 np.percentile 对 x 按 axis=0 求得的 (25%, 50%) 使用 method="higher" 的形状为 (2, 4, 5, 6)
        assert_equal(np.percentile(x, (25, 50), axis=0, method="higher").shape, (2, 4, 5, 6))
        # 断言 np.percentile 对 x 按 axis=1 求得的 (25%, 50%) 使用 method="higher" 的形状为 (2, 3, 5, 6)
        assert_equal(np.percentile(x, (25, 50), axis=1, method="higher").shape, (2, 3, 5, 6))
        # 断言 np.percentile 对 x 按 axis=2 求得的 (25%, 50%) 使用 method="higher" 的形状为 (2, 3, 4, 6)
        assert_equal(np.percentile(x, (25, 50), axis=2, method="higher").shape, (2, 3, 4, 6))
        # 断言 np.percentile 对 x 按 axis=3 求得的 (25%, 50%) 使用 method="higher" 的形状为 (2, 3, 4, 5)
        assert_equal(np.percentile(x, (25, 50), axis=3, method="higher").shape, (2, 3, 4, 5))
        # 断言 np.percentile 对 x 按 axis=1 求得的 (25%, 50%, 75%) 使用 method="higher" 的形状为 (3, 3, 5, 6)
        assert_equal(np.percentile(x, (25, 50, 75), axis=1, method="higher").shape, (3, 3, 5, 6))
    # 定义单元测试方法，用于测试 np.percentile 函数的标量返回值和轴向操作
    def test_scalar_q(self):
        # 创建一个 3x4 的数组 x，元素从 0 到 11
        x = np.arange(12).reshape(3, 4)
        # 断言计算数组 x 的第 50 百分位数，并验证结果为 5.5
        assert_equal(np.percentile(x, 50), 5.5)
        # 断言第 50 百分位数的返回值是标量
        assert_(np.isscalar(np.percentile(x, 50)))
        # 创建预期结果数组 r0，用于验证按列计算第 50 百分位数的结果
        r0 = np.array([4.,  5.,  6.,  7.])
        # 断言按列计算数组 x 的第 50 百分位数，验证结果是否与 r0 相等
        assert_equal(np.percentile(x, 50, axis=0), r0)
        # 断言按列计算数组 x 的第 50 百分位数，验证结果的形状是否与 r0 相等
        assert_equal(np.percentile(x, 50, axis=0).shape, r0.shape)
        # 创建预期结果数组 r1，用于验证按行计算第 50 百分位数的结果
        r1 = np.array([1.5,  5.5,  9.5])
        # 断言按行计算数组 x 的第 50 百分位数，验证结果是否与 r1 相近（精度相等）
        assert_almost_equal(np.percentile(x, 50, axis=1), r1)
        # 断言按行计算数组 x 的第 50 百分位数，验证结果的形状是否与 r1 相等
        assert_equal(np.percentile(x, 50, axis=1).shape, r1.shape)

        # 创建一个空的输出数组 out，用于测试输出参数功能
        out = np.empty(1)
        # 断言计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否为 5.5
        assert_equal(np.percentile(x, 50, out=out), 5.5)
        # 断言 out 的值是否等于计算出的第 50 百分位数值 5.5
        assert_equal(out, 5.5)
        # 重设 out 的大小，用于按列计算第 50 百分位数的输出参数测试
        out = np.empty(4)
        # 断言按列计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否等于 r0
        assert_equal(np.percentile(x, 50, axis=0, out=out), r0)
        # 断言 out 的值是否等于按列计算的第 50 百分位数结果 r0
        assert_equal(out, r0)
        # 重设 out 的大小，用于按行计算第 50 百分位数的输出参数测试
        out = np.empty(3)
        # 断言按行计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否等于 r1
        assert_equal(np.percentile(x, 50, axis=1, out=out), r1)
        # 断言 out 的值是否等于按行计算的第 50 百分位数结果 r1

        # 再次测试不同方法的百分位数计算，以验证对于旧版本的兼容性
        x = np.arange(12).reshape(3, 4)
        # 断言计算数组 x 的第 50 百分位数（采用 'lower' 方法），验证结果为 5.0
        assert_equal(np.percentile(x, 50, method='lower'), 5.)
        # 断言第 50 百分位数的返回值是标量
        assert_(np.isscalar(np.percentile(x, 50)))
        # 创建预期结果数组 r0，用于验证按列计算第 50 百分位数的 'lower' 方法结果
        r0 = np.array([4.,  5.,  6.,  7.])
        # 计算数组 x 按列使用 'lower' 方法计算的第 50 百分位数
        c0 = np.percentile(x, 50, method='lower', axis=0)
        # 断言 'lower' 方法按列计算数组 x 的第 50 百分位数，验证结果是否与 r0 相等
        assert_equal(c0, r0)
        # 断言 'lower' 方法按列计算数组 x 的第 50 百分位数，验证结果的形状是否与 r0 相等
        assert_equal(c0.shape, r0.shape)
        # 创建预期结果数组 r1，用于验证按行计算第 50 百分位数的 'lower' 方法结果
        r1 = np.array([1.,  5.,  9.])
        # 计算数组 x 按行使用 'lower' 方法计算的第 50 百分位数
        c1 = np.percentile(x, 50, method='lower', axis=1)
        # 断言 'lower' 方法按行计算数组 x 的第 50 百分位数，验证结果是否与 r1 相近（精度相等）
        assert_almost_equal(c1, r1)
        # 断言 'lower' 方法按行计算数组 x 的第 50 百分位数，验证结果的形状是否与 r1 相等

        # 创建一个空的输出数组 out，用于测试 'lower' 方法的输出参数功能
        out = np.empty((), dtype=x.dtype)
        # 使用 'lower' 方法计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否为 5
        c = np.percentile(x, 50, method='lower', out=out)
        # 断言 'lower' 方法计算的第 50 百分位数结果是否等于 5
        assert_equal(c, 5)
        # 断言 out 的值是否等于 'lower' 方法计算的第 50 百分位数结果 5
        assert_equal(out, 5)
        # 重设 out 的大小，用于 'lower' 方法按列计算第 50 百分位数的输出参数测试
        out = np.empty(4, dtype=x.dtype)
        # 使用 'lower' 方法按列计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否等于 r0
        c = np.percentile(x, 50, method='lower', axis=0, out=out)
        # 断言 out 的值是否等于 'lower' 方法按列计算的第 50 百分位数结果 r0
        assert_equal(out, r0)
        # 重设 out 的大小，用于 'lower' 方法按行计算第 50 百分位数的输出参数测试
        out = np.empty(3, dtype=x.dtype)
        # 使用 'lower' 方法按行计算数组 x 的第 50 百分位数，并将结果存入 out，验证计算结果是否等于 r1
        c = np.percentile(x, 50, method='lower', axis=1, out=out)
        # 断言 out 的值是否等于 'lower' 方法按行计算的第 50 百分位数结果 r1
        assert_equal(out, r1)

    # 定义异常测试方法，用于验证 np.percentile 函数在异常情况下的行为
    def test_exception(self):
        # 断言当百分位数大于 100 时，np.percentile 函数引发 ValueError 异常
        assert_raises(ValueError, np.percentile, [1], 101)
        # 断言当百分位数小于 0 时，np.percentile 函数引发 ValueError 异常
        assert_raises(ValueError, np.percentile, [1], -1)
        # 断言当使用不存在的方法 'foobar' 时，np.percentile 函数引发 ValueError 异常
        assert_raises(ValueError, np.percentile, [1, 2], 56, method='foobar')
        # 断言当百分位数列表包含超出 0 到 100 范围的值时，np.percentile 函数引发 ValueError 异常
        assert_raises(ValueError, np.percentile, [1], list(range(50)) + [101
    # 定义一个测试函数，用于测试百分位数计算函数的输出
    def test_percentile_out(self, percentile, with_weights):
        # 根据是否包含权重选择输出的数据类型
        out_dtype = int if with_weights else float
        # 创建一个包含三个元素的数组
        x = np.array([1, 2, 3])
        # 创建一个与 x 形状相同且数据类型为 out_dtype 的全零数组
        y = np.zeros((3,), dtype=out_dtype)
        # 定义百分位数的序列
        p = (1, 2, 3)
        # 如果 with_weights 为 True，则创建一个与 x 形状相同的全一数组作为权重，否则设为 None
        weights = np.ones_like(x) if with_weights else None
        # 计算百分位数，并将结果存入 y 中，同时返回结果 r
        r = percentile(x, p, out=y, weights=weights)
        # 断言 r 和 y 是同一个对象
        assert r is y
        # 断言调用 percentile 函数的结果与 y 相等
        assert_equal(percentile(x, p, weights=weights), y)

        # 创建一个包含两行三列的二维数组
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])
        # 创建一个与 x 形状相同且数据类型为 out_dtype 的全零数组
        y = np.zeros((3, 3), dtype=out_dtype)
        # 如果 with_weights 为 True，则创建一个与 x 形状相同的全一数组作为权重，否则设为 None
        weights = np.ones_like(x) if with_weights else None
        # 计算沿着第一个维度的百分位数，并将结果存入 y 中，同时返回结果 r
        r = percentile(x, p, axis=0, out=y, weights=weights)
        # 断言 r 和 y 是同一个对象
        assert r is y
        # 断言调用 percentile 函数的结果与 y 相等
        assert_equal(percentile(x, p, weights=weights, axis=0), y)

        # 创建一个与 x 形状相同且数据类型为 out_dtype 的全零数组
        y = np.zeros((3, 2), dtype=out_dtype)
        # 计算沿着第二个维度的百分位数，并将结果存入 y 中
        percentile(x, p, axis=1, out=y, weights=weights)
        # 断言调用 percentile 函数的结果与 y 相等
        assert_equal(percentile(x, p, weights=weights, axis=1), y)

        # 创建一个 3x4 的数组，其元素值为 0 到 11
        x = np.arange(12).reshape(3, 4)
        # 如果 with_weights 为 True，则创建一个特定数组作为预期结果，否则创建另一个预期结果数组
        if with_weights:
            r0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        else:
            r0 = np.array([[2., 3., 4., 5.], [4., 5., 6., 7.]])
        # 创建一个与 r0 形状相同且数据类型为 out_dtype 的空数组
        out = np.empty((2, 4), dtype=out_dtype)
        # 如果 with_weights 为 True，则创建一个与 x 形状相同的全一数组作为权重，否则设为 None
        weights = np.ones_like(x) if with_weights else None
        # 计算沿着第一个维度的百分位数，并将结果存入 out 中，同时返回结果
        assert_equal(
            percentile(x, (25, 50), axis=0, out=out, weights=weights), r0
        )
        # 断言 out 和 r0 相等
        assert_equal(out, r0)
        
        # 创建一个与 r1 形状相同的空数组
        r1 = np.array([[0.75,  4.75,  8.75], [1.5,  5.5,  9.5]])
        # 创建一个形状为 (2, 3) 的空数组
        out = np.empty((2, 3))
        # 计算沿着第二个维度的百分位数，并将结果存入 out 中
        assert_equal(np.percentile(x, (25, 50), axis=1, out=out), r1)
        # 断言 out 和 r1 相等
        assert_equal(out, r1)

        # 创建一个与 r0 形状相同的数组，数据类型为 x 的数据类型
        r0 = np.array([[0,  1,  2, 3], [4, 5, 6, 7]])
        # 创建一个形状为 (2, 4) 的空数组，数据类型为 x 的数据类型
        out = np.empty((2, 4), dtype=x.dtype)
        # 计算沿着第一个维度的百分位数，并将结果存入 out 中，使用 'lower' 方法
        c = np.percentile(x, (25, 50), method='lower', axis=0, out=out)
        # 断言 c 和 r0 相等
        assert_equal(c, r0)
        # 断言 out 和 r0 相等
        assert_equal(out, r0)

        # 创建一个与 r1 形状相同的数组，数据类型为 x 的数据类型
        r1 = np.array([[0,  4,  8], [1,  5,  9]])
        # 创建一个形状为 (2, 3) 的空数组，数据类型为 x 的数据类型
        out = np.empty((2, 3), dtype=x.dtype)
        # 计算沿着第二个维度的百分位数，并将结果存入 out 中，使用 'lower' 方法
        c = np.percentile(x, (25, 50), method='lower', axis=1, out=out)
        # 断言 c 和 r1 相等
        assert_equal(c, r1)
        # 断言 out 和 r1 相等
        assert_equal(out, r1)
    def test_percentile_empty_dim(self):
        # 创建一个 11 行、1 列、2 深度的 NumPy 数组
        d = np.arange(11 * 2).reshape(11, 1, 2, 1)
        # 对指定轴(axis=0)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=0).shape, (1, 2, 1))
        # 对指定轴(axis=1)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=1).shape, (11, 2, 1))
        # 对指定轴(axis=2)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=2).shape, (11, 1, 1))
        # 对指定轴(axis=3)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=3).shape, (11, 1, 2))
        # 对指定轴(axis=-1，即第三轴)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=-1).shape, (11, 1, 2))
        # 对指定轴(axis=-2，即第二轴)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=-2).shape, (11, 1, 1))
        # 对指定轴(axis=-3，即第一轴)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=-3).shape, (11, 2, 1))
        # 对指定轴(axis=-4)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=-4).shape, (1, 2, 1))

        # 使用中点法(method='midpoint')对指定轴(axis=2)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=2,
                                         method='midpoint').shape,
                           (11, 1, 1))
        # 使用中点法(method='midpoint')对指定轴(axis=-2)计算百分位数并验证形状
        assert_array_equal(np.percentile(d, 50, axis=-2,
                                         method='midpoint').shape,
                           (11, 1, 1))

        # 对多个百分位数进行计算并验证形状
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=0)).shape,
                           (2, 1, 2, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=1)).shape,
                           (2, 11, 2, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=2)).shape,
                           (2, 11, 1, 1))
        assert_array_equal(np.array(np.percentile(d, [10, 50], axis=3)).shape,
                           (2, 11, 1, 2))

    def test_percentile_no_overwrite(self):
        # 创建一个 NumPy 数组
        a = np.array([2, 3, 4, 1])
        # 对数组计算指定百分位数，不覆盖输入数组，验证输入数组是否不变
        np.percentile(a, [50], overwrite_input=False)
        assert_equal(a, np.array([2, 3, 4, 1]))

        # 重新创建数组
        a = np.array([2, 3, 4, 1])
        # 对数组计算指定百分位数，默认覆盖输入数组，验证输入数组是否不变
        np.percentile(a, [50])
        assert_equal(a, np.array([2, 3, 4, 1]))

    def test_no_p_overwrite(self):
        # 创建一个等差数列作为百分位数
        p = np.linspace(0., 100., num=5)
        # 对数组计算指定百分位数，并使用中点法(method="midpoint")
        np.percentile(np.arange(100.), p, method="midpoint")
        # 验证百分位数数组是否未被修改
        assert_array_equal(p, np.linspace(0., 100., num=5))
        # 将百分位数数组转换为列表
        p = np.linspace(0., 100., num=5).tolist()
        # 对数组计算指定百分位数，并使用中点法(method="midpoint")
        np.percentile(np.arange(100.), p, method="midpoint")
        # 验证百分位数数组是否未被修改
        assert_array_equal(p, np.linspace(0., 100., num=5).tolist())

    def test_percentile_overwrite(self):
        # 创建一个 NumPy 数组
        a = np.array([2, 3, 4, 1])
        # 对数组计算指定百分位数，并覆盖输入数组
        b = np.percentile(a, [50], overwrite_input=True)
        # 验证输出数组的值
        assert_equal(b, np.array([2.5]))

        # 对列表直接计算指定百分位数，并覆盖输入列表
        b = np.percentile([2, 3, 4, 1], [50], overwrite_input=True)
        # 验证输出数组的值
        assert_equal(b, np.array([2.5]))
    # 定义测试方法，用于验证在扩展轴上使用 np.percentile 函数的行为
    def test_extended_axis(self):
        # 创建一个形状为 (71, 23) 的正态分布随机数组成的数组 o
        o = np.random.normal(size=(71, 23))
        # 将 o 数组在第三个轴向上堆叠 10 次，形成新的数组 x
        x = np.dstack([o] * 10)
        # 验证在多个轴上计算百分位数的结果是否与在 o 上计算的结果相等
        assert_equal(np.percentile(x, 30, axis=(0, 1)), np.percentile(o, 30))
        # 将 x 数组的轴向进行移动，将最后一个轴移动到第一个位置
        x = np.moveaxis(x, -1, 0)
        # 再次验证在新的轴向上计算百分位数的结果是否与在 o 上计算的结果相等
        assert_equal(np.percentile(x, 30, axis=(-2, -1)), np.percentile(o, 30))
        # 交换 x 数组的第一和第二个轴，并创建其副本
        x = x.swapaxes(0, 1).copy()
        # 再次验证在交换后的轴向上计算百分位数的结果是否与在 o 上计算的结果相等
        assert_equal(np.percentile(x, 30, axis=(0, -1)), np.percentile(o, 30))
        # 再次交换 x 数组的第一和第二个轴，并创建其副本

        assert_equal(np.percentile(x, [25, 60], axis=(0, 1, 2)),
                     np.percentile(x, [25, 60], axis=None))
        # 验证在多个指定轴上计算多个百分位数的结果是否与在无轴指定情况下计算的结果相等
        assert_equal(np.percentile(x, [25, 60], axis=(0,)),
                     np.percentile(x, [25, 60], axis=0))

        # 创建一个形状为 (3, 5, 7, 11) 的数组 d，并随机打乱其元素顺序
        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        np.random.shuffle(d.ravel())
        # 验证在多个轴上计算百分位数的结果是否与在特定切片上计算的结果相等
        assert_equal(np.percentile(d, 25,  axis=(0, 1, 2))[0],
                     np.percentile(d[:,:,:, 0].flatten(), 25))
        assert_equal(np.percentile(d, [10, 90], axis=(0, 1, 3))[:, 1],
                     np.percentile(d[:,:, 1,:].flatten(), [10, 90]))
        assert_equal(np.percentile(d, 25, axis=(3, 1, -4))[2],
                     np.percentile(d[:,:, 2,:].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(3, 1, 2))[2],
                     np.percentile(d[2,:,:,:].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(3, 2))[2, 1],
                     np.percentile(d[2, 1,:,:].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(1, -2))[2, 1],
                     np.percentile(d[2,:,:, 1].flatten(), 25))
        assert_equal(np.percentile(d, 25, axis=(1, 3))[2, 2],
                     np.percentile(d[2,:, 2,:].flatten(), 25))

    # 定义测试方法，用于验证在非法的扩展轴参数情况下 np.percentile 函数是否会引发异常
    def test_extended_axis_invalid(self):
        # 创建一个形状为 (3, 5, 7, 11) 的全一数组 d
        d = np.ones((3, 5, 7, 11))
        # 验证当轴参数为非法负数时，np.percentile 函数是否会引发 AxisError 异常
        assert_raises(AxisError, np.percentile, d, axis=-5, q=25)
        assert_raises(AxisError, np.percentile, d, axis=(0, -5), q=25)
        # 验证当轴参数超出数组维度时，np.percentile 函数是否会引发 AxisError 异常
        assert_raises(AxisError, np.percentile, d, axis=4, q=25)
        assert_raises(AxisError, np.percentile, d, axis=(0, 4), q=25)
        # 验证当轴参数重复指定相同轴时，np.percentile 函数是否会引发 ValueError 异常
        assert_raises(ValueError, np.percentile, d, axis=(1, 1), q=25)
        assert_raises(ValueError, np.percentile, d, axis=(-1, -1), q=25)
        assert_raises(ValueError, np.percentile, d, axis=(3, -1), q=25)
    # 定义一个测试方法，用于测试 np.percentile 函数在保持维度参数 keepdims=True 下的行为
    def test_keepdims(self):
        # 创建一个形状为 (3, 5, 7, 11) 的全为1的数组 d
        d = np.ones((3, 5, 7, 11))
        # 测试在所有轴上应用百分位数函数，并保持维度为 (1, 1, 1, 1)
        assert_equal(np.percentile(d, 7, axis=None, keepdims=True).shape,
                     (1, 1, 1, 1))
        # 测试在轴 (0, 1) 上应用百分位数函数，并保持维度为 (1, 1, 7, 11)
        assert_equal(np.percentile(d, 7, axis=(0, 1), keepdims=True).shape,
                     (1, 1, 7, 11))
        # 测试在轴 (0, 3) 上应用百分位数函数，并保持维度为 (1, 5, 7, 1)
        assert_equal(np.percentile(d, 7, axis=(0, 3), keepdims=True).shape,
                     (1, 5, 7, 1))
        # 测试在轴 (1,) 上应用百分位数函数，并保持维度为 (3, 1, 7, 11)
        assert_equal(np.percentile(d, 7, axis=(1,), keepdims=True).shape,
                     (3, 1, 7, 11))
        # 测试在所有轴上应用百分位数函数，并保持维度为 (1, 1, 1, 1)
        assert_equal(np.percentile(d, 7, (0, 1, 2, 3), keepdims=True).shape,
                     (1, 1, 1, 1))
        # 测试在轴 (0, 1, 3) 上应用百分位数函数，并保持维度为 (1, 1, 7, 1)
        assert_equal(np.percentile(d, 7, axis=(0, 1, 3), keepdims=True).shape,
                     (1, 1, 7, 1))

        # 测试在轴 (0, 1, 3) 上应用百分位数函数，返回两个百分位数的形状 (2, 1, 1, 7, 1)
        assert_equal(np.percentile(d, [1, 7], axis=(0, 1, 3),
                                   keepdims=True).shape, (2, 1, 1, 7, 1))
        # 测试在轴 (0, 3) 上应用百分位数函数，返回两个百分位数的形状 (2, 1, 5, 7, 1)
        assert_equal(np.percentile(d, [1, 7], axis=(0, 3),
                                   keepdims=True).shape, (2, 1, 5, 7, 1))

    # 使用 pytest 的参数化装饰器，测试在不同参数下 np.percentile 函数的行为
    @pytest.mark.parametrize('q', [7, [1, 7]])
    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,     # 测试在所有轴上应用百分位数函数
            1,        # 测试在单个轴上应用百分位数函数
            (1,),     # 测试在单个轴上应用百分位数函数（元组形式）
            (0, 1),   # 测试在两个轴上应用百分位数函数
            (-3, -1), # 测试在负数索引的两个轴上应用百分位数函数
        ]
    )
    # 定义一个测试方法，测试 np.percentile 函数在保持维度参数 keepdims=True 下的输出行为
    def test_keepdims_out(self, q, axis):
        # 创建一个形状为 (3, 5, 7, 11) 的全为1的数组 d
        d = np.ones((3, 5, 7, 11))
        # 如果轴参数 axis 是 None，则将 shape_out 设置为全为1的数组的形状
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            # 否则，使用 normalize_axis_tuple 函数对轴参数进行规范化处理，并生成相应的形状
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        # 将 q 的形状与 shape_out 连接形成最终的输出形状
        shape_out = np.shape(q) + shape_out

        # 创建一个空数组 out，其形状为 shape_out
        out = np.empty(shape_out)
        # 使用 np.percentile 函数计算 d 在给定轴上的百分位数，并将结果存储在 out 中
        result = np.percentile(d, q, axis=axis, keepdims=True, out=out)
        # 断言 result 与 out 是同一个对象
        assert result is out
        # 断言 result 的形状与 shape_out 相等
        assert_equal(result.shape, shape_out)

    # 定义一个测试方法，测试 np.percentile 函数在使用输出参数 out 时的行为
    def test_out(self):
        # 创建一个形状为 (4,) 的全为0的数组 o 和形状为 (3, 4) 的全为1的数组 d
        o = np.zeros((4,))
        d = np.ones((3, 4))
        # 测试将 d 在第0轴上的百分位数存储到数组 o 中，并断言结果与 o 相等
        assert_equal(np.percentile(d, 0, 0, out=o), o)
        # 测试将 d 在第0轴上的百分位数存储到数组 o 中（使用最近法作为方法），并断言结果与 o 相等
        assert_equal(np.percentile(d, 0, 0, method='nearest', out=o), o)
        # 创建一个形状为 (3,) 的全为0的数组 o
        o = np.zeros((3,))
        # 测试将 d 在第1轴上的第1个百分位数存储到数组 o 中，并断言结果与 o 相等
        assert_equal(np.percentile(d, 1, 1, out=o), o)
        # 测试将 d 在第1轴上的第1个百分位数存储到数组 o 中（使用最近法作为方法），并断言结果与 o 相等
        assert_equal(np.percentile(d, 1, 1, method='nearest', out=o), o)

        # 创建一个形状为 () 的全为0的数组 o
        o = np.zeros(())
        # 测试将 d 的第2个百分位数存储到数组 o 中，并断言结果与 o 相等
        assert_equal(np.percentile(d, 2, out=o), o)
        # 测试将 d 的第2个百分位数存储到数组 o 中（使用最近法作为方法），并断言结果与 o 相等
        assert_equal(np.percentile(d, 2, method='nearest', out=o), o)

    # 使用 pytest 的参数化装饰器，测试不同的方法和权重标志对 np.percentile 函数的影响
    @pytest.mark.parametrize("method, weighted", [
        ("linear", False),
        ("nearest", False),
        ("inverted_cdf", False),
        ("inverted_cdf", True),
    ])
    # 测试带有 NaN 的情况下，使用 np.percentile 函数的行为
    def test_out_nan(self, method, weighted):
        # 如果 weighted 为 True，则使用权重为全1的数组
        if weighted:
            kwargs = {"weights": np.ones((3, 4)), "method": method}
        else:
            # 否则只传递 method 参数
            kwargs = {"method": method}
        # 使用 warnings 模块捕获运行时警告
        with warnings.catch_warnings(record=True):
            # 过滤掉所有 RuntimeWarning
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 创建一个全零数组 o 和一个全1数组 d，并将 d 中的一个元素设为 NaN
            o = np.zeros((4,))
            d = np.ones((3, 4))
            d[2, 1] = np.nan
            # 断言调用 np.percentile 函数后返回的结果与预期的 o 数组相等
            assert_equal(np.percentile(d, 0, 0, out=o, **kwargs), o)

            # 创建一个全零数组 o，并在调用 np.percentile 函数时指定轴向
            o = np.zeros((3,))
            assert_equal(np.percentile(d, 1, 1, out=o, **kwargs), o)

            # 创建一个标量全零数组 o，并调用 np.percentile 函数计算结果
            o = np.zeros(())
            assert_equal(np.percentile(d, 1, out=o, **kwargs), o)

    # 测试包含 NaN 的数组在计算百分位数时的行为
    def test_nan_behavior(self):
        # 创建一个浮点类型的数组 a，其中一个元素设为 NaN
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        # 断言计算百分位数时，如果数组中包含 NaN，则结果应为 NaN
        assert_equal(np.percentile(a, 0.3), np.nan)
        # 断言指定轴向计算百分位数时，如果数组中包含 NaN，则结果应为 NaN
        assert_equal(np.percentile(a, 0.3, axis=0), np.nan)
        # 断言指定多个百分位数时，如果数组中包含 NaN，则结果应为包含多个 NaN 的数组
        assert_equal(np.percentile(a, [0.3, 0.6], axis=0),
                     np.array([np.nan] * 2))

        # 创建一个形状为 (2, 3, 4) 的浮点类型数组 a，其中多个元素设为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 没有指定轴向时，断言计算百分位数时，如果数组中包含 NaN，则结果应为 NaN
        assert_equal(np.percentile(a, 0.3), np.nan)
        # 没有指定轴向时，断言结果的维度应为 0
        assert_equal(np.percentile(a, 0.3).ndim, 0)

        # 指定 axis=0 时，断言结果与预期的 b 数组相等，且其中含有 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 0), b)

        # 指定 axis=0 时，断言结果与预期的 b 数组相等，且其中含有多个 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4),
                          [0.3, 0.6], 0)
        b[:, 2, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 0), b)

        # 指定 axis=1 时，断言结果与预期的 b 数组相等，且其中含有 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 1), b)

        # 指定 axis=1 时，断言结果与预期的 b 数组相等，且其中含有多个 NaN
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4), [0.3, 0.6], 1)
        b[:, 1, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 1), b)

        # 指定 axis=(0, 2) 时，断言结果与预期的 b 数组相等，且其中含有 NaN
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.percentile(a, 0.3, (0, 2)), b)

        # 指定 axis=(0, 2) 时，使用 method='nearest'，断言结果与预期的 b 数组相等，且其中含有多个 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4),
                          [0.3, 0.6], (0, 2), method='nearest')
        b[:, 1] = np.nan
        b[:, 2] = np.nan
        assert_equal(np.percentile(
            a, [0.3, 0.6], (0, 2), method='nearest'), b)
    # 定义一个测试函数，用于测试处理 NaN 值的情况
    def test_nan_q(self):
        # 在处理 NaN 值时抛出 ValueError 异常，并检查异常消息是否包含指定字符串
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], np.nan)
        # 同样地，在处理包含 NaN 的列表时抛出 ValueError 异常
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], [np.nan])
        # 使用 np.linspace 创建等间隔的数组，并将第一个元素设置为 NaN
        q = np.linspace(1.0, 99.0, 16)
        q[0] = np.nan
        # 再次测试处理包含 NaN 的数组时抛出 ValueError 异常
        with pytest.raises(ValueError, match="Percentiles must be in"):
            np.percentile([1, 2, 3, 4.0], q)

    # 使用 pytest 的参数化标记指定多个参数化组合来测试 NaT 的基本行为
    @pytest.mark.parametrize("dtype", ["m8[D]", "M8[s]"])
    @pytest.mark.parametrize("pos", [0, 23, 10])
    def test_nat_basic(self, dtype, pos):
        # TODO: 注意，修复 NaT 之后时间可能存在问题的四舍五入！
        # 设置一个 numpy 数组，用于测试 NaT 和 NaN 的基本行为：
        a = np.arange(0, 24, dtype=dtype)
        # 将指定位置的元素设置为 "NaT"
        a[pos] = "NaT"
        # 测试计算数组的百分位数，并检查结果的数据类型是否与输入一致
        res = np.percentile(a, 30)
        assert res.dtype == dtype
        # 检查计算结果是否为 NaT
        assert np.isnat(res)
        # 再次测试计算多个百分位数时的行为，确保结果的数据类型和 NaT 的判断结果均符合预期
        res = np.percentile(a, [30, 60])
        assert res.dtype == dtype
        assert np.isnat(res).all()

        # 创建一个二维数组，并将指定位置的元素设置为 "NaT"
        a = np.arange(0, 24*3, dtype=dtype).reshape(-1, 3)
        a[pos, 1] = "NaT"
        # 按列计算数组的百分位数，检查结果中指定位置的元素是否为 NaT
        res = np.percentile(a, 30, axis=0)
        assert_array_equal(np.isnat(res), [False, True, False])
quantile_methods = [
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear',
    'median_unbiased', 'normal_unbiased', 'nearest', 'lower', 'higher',
    'midpoint']

methods_supporting_weights = ["inverted_cdf"]

class TestQuantile:
    # most of this is already tested by TestPercentile

    def V(self, x, y, alpha):
        # Identification function used in several tests.
        # 返回比较结果的值，用于多个测试中的标识功能
        return (x >= y) - alpha

    def test_max_ulp(self):
        x = [0.0, 0.2, 0.4]
        a = np.quantile(x, 0.45)
        # The default linear method would result in 0 + 0.2 * (0.45/2) = 0.18.
        # 默认的线性方法会导致结果为 0 + 0.2 * (0.45/2) = 0.18。
        # 0.18 不是精确可表示的，该公式导致了一个 1 ULP 的不同结果。确保结果在 1 ULP 内，参见 gh-20331。
        np.testing.assert_array_max_ulp(a, 0.18, maxulp=1)

    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.quantile(x, 0), 0.)
        assert_equal(np.quantile(x, 1), 3.5)
        assert_equal(np.quantile(x, 0.5), 1.75)

    def test_correct_quantile_value(self):
        a = np.array([True])
        tf_quant = np.quantile(True, False)
        # 测试对 True 和 False 的 quantile 结果
        assert_equal(tf_quant, a[0])
        assert_equal(type(tf_quant), a.dtype)
        a = np.array([False, True, True])
        quant_res = np.quantile(a, a)
        # 测试对数组 a 自身的 quantile 结果
        assert_array_equal(quant_res, a)
        assert_equal(quant_res.dtype, a.dtype)

    def test_fraction(self):
        # fractional input, integral quantile
        x = [Fraction(i, 2) for i in range(8)]
        q = np.quantile(x, 0)
        # 测试分数输入和整数 quantile
        assert_equal(q, 0)
        assert_equal(type(q), Fraction)

        q = np.quantile(x, 1)
        assert_equal(q, Fraction(7, 2))
        assert_equal(type(q), Fraction)

        q = np.quantile(x, .5)
        assert_equal(q, 1.75)
        assert_equal(type(q), np.float64)

        q = np.quantile(x, Fraction(1, 2))
        assert_equal(q, Fraction(7, 4))
        assert_equal(type(q), Fraction)

        q = np.quantile(x, [Fraction(1, 2)])
        assert_equal(q, np.array([Fraction(7, 4)]))
        assert_equal(type(q), np.ndarray)

        q = np.quantile(x, [[Fraction(1, 2)]])
        assert_equal(q, np.array([[Fraction(7, 4)]]))
        assert_equal(type(q), np.ndarray)

        # repeat with integral input but fractional quantile
        x = np.arange(8)
        assert_equal(np.quantile(x, Fraction(1, 2)), Fraction(7, 2))

    def test_complex(self):
        #See gh-22652
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='G')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='D')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5+3.0j, 2.1+0.5j, 1.6+2.3j], dtype='F')
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
    def test_no_p_overwrite(self):
        # 定义测试函数，验证 np.quantile() 不会覆盖原始数组 p0
        # 创建初始数组 p0，深复制一份给数组 p
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        # 调用 np.quantile() 计算分位数，不会修改数组 p
        np.quantile(np.arange(100.), p, method="midpoint")
        # 断言保证数组 p 与原始 p0 相等
        assert_array_equal(p, p0)

        # 将 p0 转换为列表
        p0 = p0.tolist()
        # 将 p 转换为列表
        p = p.tolist()
        # 再次调用 np.quantile() 计算分位数，不会修改数组 p
        np.quantile(np.arange(100.), p, method="midpoint")
        # 断言保证数组 p 与原始 p0 相等
        assert_array_equal(p, p0)

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_quantile_preserve_int_type(self, dtype):
        # 测试 np.quantile() 保持整数类型
        res = np.quantile(np.array([1, 2], dtype=dtype), [0.5],
                          method="nearest")
        # 断言检查返回结果的数据类型与输入的 dtype 是否一致
        assert res.dtype == dtype

    @pytest.mark.parametrize("method", quantile_methods)
    def test_q_zero_one(self, method):
        # 测试 np.quantile() 对数组取分位数为 0 和 1 的情况
        # gh-24710
        arr = [10, 11, 12]
        quantile = np.quantile(arr, q = [0, 1], method=method)
        # 断言保证返回的分位数结果与预期的数组一致
        assert_equal(quantile,  np.array([10, 12]))

    @pytest.mark.parametrize("method", quantile_methods)
    def test_quantile_monotonic(self, method):
        # GH 14685
        # 测试当 p0 是有序的时候，np.quantile() 返回值是否单调递增
        # 同时测试边界值的处理是否正确
        p0 = np.linspace(0, 1, 101)
        quantile = np.quantile(np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9,
                                         8, 8, 7]) * 0.1, p0, method=method)
        # 断言保证返回的分位数结果是排序后的
        assert_equal(np.sort(quantile), quantile)

        # 另外测试数据点数量是可以被整除的情况：
        quantile = np.quantile([0., 1., 2., 3.], p0, method=method)
        # 断言保证返回的分位数结果是排序后的
        assert_equal(np.sort(quantile), quantile)

    @hypothesis.given(
            arr=arrays(dtype=np.float64,
                       shape=st.integers(min_value=3, max_value=1000),
                       elements=st.floats(allow_infinity=False, allow_nan=False,
                                          min_value=-1e300, max_value=1e300)))
    def test_quantile_monotonic_hypo(self, arr):
        # 使用 Hypothesis 测试 np.quantile() 返回值是否单调递增
        p0 = np.arange(0, 1, 0.01)
        quantile = np.quantile(arr, p0)
        # 断言保证返回的分位数结果是排序后的
        assert_equal(np.sort(quantile), quantile)

    def test_quantile_scalar_nan(self):
        # 测试 np.quantile() 处理包含 NaN 的情况
        a = np.array([[10., 7., 4.], [3., 2., 1.]])
        a[0][1] = np.nan
        actual = np.quantile(a, 0.5)
        # 断言保证返回的结果是标量且与预期的 NaN 相等
        assert np.isscalar(actual)
        assert_equal(np.quantile(a, 0.5), np.nan)

    @pytest.mark.parametrize("weights", [False, True])
    @pytest.mark.parametrize("method", quantile_methods)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    def test_quantile_identification_equation(self, weights, method, alpha):
        # 测试经验CDF中识别方程是否成立：
        #   E[V(x, Y)] = 0  <=>  x 是分位数
        # 其中 Y 是我们观察到值的随机变量，V(x, y) 是分位数（在水平 alpha 下）的规范识别函数，参见
        # https://doi.org/10.48550/arXiv.0912.0902
        if weights and method not in methods_supporting_weights:
            # 如果使用了权重但方法不支持，跳过测试
            pytest.skip("Weights not supported by method.")
        rng = np.random.default_rng(4321)
        # 我们选择 n 和 alpha 使我们覆盖 3 种情况：
        #  - n * alpha 是整数
        #  - n * alpha 是向下舍入的浮点数
        #  - n * alpha 是向上舍入的浮点数
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        w = rng.integers(low=0, high=10, size=n) if weights else None
        # 计算分位数 x
        x = np.quantile(y, alpha, method=method, weights=w)

        if method in ("higher",):
            # 这些方法不满足识别方程
            assert np.abs(np.mean(self.V(x, y, alpha))) > 0.1 / n
        elif int(n * alpha) == n * alpha and not weights:
            # 我们可以期望精确结果，机器精度上
            assert_allclose(
                np.average(self.V(x, y, alpha), weights=w), 0, atol=1e-14,
            )
        else:
            # V = (x >= y) - alpha 不能完全等于零，但在“样本精度”内
            assert_allclose(np.average(self.V(x, y, alpha), weights=w), 0,
                atol=1 / n / np.amin([alpha, 1 - alpha]))

    @pytest.mark.parametrize("weights", [False, True])
    @pytest.mark.parametrize("method", quantile_methods)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    @pytest.mark.parametrize("method", methods_supporting_weights)
    @pytest.mark.parametrize("alpha", [0.2, 0.5, 0.9])
    def test_quantile_constant_weights(self, method, alpha):
        rng = np.random.default_rng(4321)
        # 我们选择 n 和 alpha 使我们有以下情况：
        #  - n * alpha 是整数
        #  - n * alpha 是向下舍入的浮点数
        #  - n * alpha 是向上舍入的浮点数
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)
        # 计算分位数 q
        q = np.quantile(y, alpha, method=method)

        w = np.ones_like(y)
        # 计算带权重的分位数 qw
        qw = np.quantile(y, alpha, method=method, weights=w)
        assert_allclose(qw, q)

        w = 8.125 * np.ones_like(y)
        # 计算带权重的分位数 qw
        qw = np.quantile(y, alpha, method=method, weights=w)
        assert_allclose(qw, q)

    @pytest.mark.parametrize("method", methods_supporting_weights)
    @pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.9, 1])
    # 定义一个测试方法，用于测试带整数权重的分位数计算
    def test_quantile_with_integer_weights(self, method, alpha):
        # Integer weights can be interpreted as repeated observations.
        # 整数权重可以被解释为重复的观测值。

        # 设置一个随机数生成器对象，种子为4321
        rng = np.random.default_rng(4321)

        # 选择 n 和 alpha 以确保我们覆盖以下情况：
        #  - n * alpha 是整数
        #  - n * alpha 是一个被向下取整的浮点数
        #  - n * alpha 是一个被向上取整的浮点数
        n = 102  # n * alpha = 20.4, 51. , 91.8
        y = rng.random(n)  # 生成长度为 n 的随机数组 y
        w = rng.integers(low=0, high=10, size=n, dtype=np.int32)  # 生成长度为 n 的随机整数权重数组 w

        # 使用带权重的方法计算 y 的 alpha 分位数
        qw = np.quantile(y, alpha, method=method, weights=w)

        # 将 y 重复 w 次后，计算其 alpha 分位数
        q = np.quantile(np.repeat(y, w), alpha, method=method)

        # 断言两种计算方法的结果是否接近
        assert_allclose(qw, q)

    @pytest.mark.parametrize("method", methods_supporting_weights)
    # 定义一个参数化测试方法，用于测试带权重和轴的分位数计算
    def test_quantile_with_weights_and_axis(self, method):
        # 设置一个随机数生成器对象，种子为4321
        rng = np.random.default_rng(4321)

        # 生成一个 2x10x3 的随机数组 y
        y = rng.random((2, 10, 3))

        # 生成一个长度为 10 的随机非负权重数组 w
        w = np.abs(rng.random(10))

        alpha = 0.5  # 设置分位数 alpha 值为 0.5

        # 计算 y 在 axis=1 上的 alpha 分位数，使用权重 w 和指定方法
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)

        # 初始化一个与 q 同形状的全零数组 q_res
        q_res = np.zeros(shape=(2, 3))

        # 循环计算每个 (i, j) 对应的分位数
        for i in range(2):
            for j in range(3):
                q_res[i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w
                )

        # 断言计算结果是否接近
        assert_allclose(q, q_res)

        # 对于 1 维权重和 1 维 alpha 的情况
        alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]  # shape (6,)
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)

        # 初始化一个与 q 同形状的全零数组 q_res
        q_res = np.zeros(shape=(6, 2, 3))

        # 循环计算每个 (i, j) 对应的分位数
        for i in range(2):
            for j in range(3):
                q_res[:, i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w
                )

        # 断言计算结果是否接近
        assert_allclose(q, q_res)

        # 对于 1 维权重和 2 维 alpha 的情况
        alpha = [[0, 0.2], [0.4, 0.6], [0.8, 1]]  # shape (3, 2)
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)

        # 重塑 q_res 以匹配 q 的形状
        q_res = q_res.reshape((3, 2, 2, 3))

        # 断言计算结果是否接近
        assert_allclose(q, q_res)

        # 当权重的形状等于 y 的形状时
        w = np.abs(rng.random((2, 10, 3)))
        alpha = 0.5
        q = np.quantile(y, alpha, weights=w, method=method, axis=1)

        # 初始化一个与 q 同形状的全零数组 q_res
        q_res = np.zeros(shape=(2, 3))

        # 循环计算每个 (i, j) 对应的分位数
        for i in range(2):
            for j in range(3):
                q_res[i, j] = np.quantile(
                    y[i, :, j], alpha, method=method, weights=w[i, :, j]
                )

        # 断言计算结果是否接近
        assert_allclose(q, q_res)

    # 测试当权重包含负值时是否会触发 ValueError
    def test_quantile_weights_raises_negative_weights(self):
        y = [1, 2]
        w = [-0.5, 1]

        # 使用 pytest 断言捕获 ValueError 异常，确保错误信息包含指定文本
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            np.quantile(y, 0.5, weights=w, method="inverted_cdf")

    @pytest.mark.parametrize(
            "method",
            sorted(set(quantile_methods) - set(methods_supporting_weights)),
    )
    # 定义一个测试方法，用于测试在给定方法下使用权重时是否引发异常
    def test_quantile_weights_raises_unsupported_methods(self, method):
        # 创建一个包含数值的列表 y 和权重列表 w
        y = [1, 2]
        w = [0.5, 1]
        # 定义异常消息
        msg = "Only method 'inverted_cdf' supports weights"
        # 使用 pytest 断言检查是否会引发 ValueError 异常，并验证异常消息是否匹配预期消息
        with pytest.raises(ValueError, match=msg):
            np.quantile(y, 0.5, weights=w, method=method)

    # 定义一个测试方法，用于测试在 weibull 方法下的分位数计算
    def test_weibull_fraction(self):
        # 创建一个分数列表 arr，包含两个分数
        arr = [Fraction(0, 1), Fraction(1, 10)]
        # 使用 numpy 的 quantile 函数计算指定分位数的值，method 参数设置为 'weibull'
        quantile = np.quantile(arr, [0, ], method='weibull')
        # 使用 assert_equal 断言检查计算结果是否等于预期值，预期值是一个包含单个分数的 numpy 数组
        assert_equal(quantile, np.array(Fraction(0, 1)))
        # 再次使用 quantile 函数计算不同分位数的值，method 参数仍设置为 'weibull'
        quantile = np.quantile(arr, [Fraction(1, 2)], method='weibull')
        # 使用 assert_equal 断言检查计算结果是否等于预期值，预期值是一个包含单个分数的 numpy 数组
        assert_equal(quantile, np.array(Fraction(1, 20)))
    # 定义测试类 TestLerp，用于测试线性插值函数 _lerp 的行为
    class TestLerp:
        # 使用 Hypothesis 库的给定测试数据生成器，测试线性插值函数在单调性上的表现
        @hypothesis.given(t0=st.floats(allow_nan=False, allow_infinity=False,
                                       min_value=0, max_value=1),
                          t1=st.floats(allow_nan=False, allow_infinity=False,
                                       min_value=0, max_value=1),
                          a=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300),
                          b=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300))
        def test_linear_interpolation_formula_monotonic(self, t0, t1, a, b):
            # 计算 t0 和 t1 下的线性插值结果
            l0 = nfb._lerp(a, b, t0)
            l1 = nfb._lerp(a, b, t1)
            # 根据条件断言插值结果的单调性
            if t0 == t1 or a == b:
                assert l0 == l1  # 不是很有趣的情况
            elif (t0 < t1) == (a < b):
                assert l0 <= l1
            else:
                assert l0 >= l1

        # 使用 Hypothesis 库的给定测试数据生成器，测试线性插值函数在范围内的性质
        @hypothesis.given(t=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=0, max_value=1),
                          a=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300),
                          b=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300))
        def test_linear_interpolation_formula_bounded(self, t, a, b):
            # 根据 a 和 b 的关系断言插值结果在给定范围内
            if a <= b:
                assert a <= nfb._lerp(a, b, t) <= b
            else:
                assert b <= nfb._lerp(a, b, t) <= a

        # 使用 Hypothesis 库的给定测试数据生成器，测试线性插值函数的对称性质
        @hypothesis.given(t=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=0, max_value=1),
                          a=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300),
                          b=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e300, max_value=1e300))
        def test_linear_interpolation_formula_symmetric(self, t, a, b):
            # 使用 _lerp 函数分别计算左右对称的插值结果
            # 双重减法用于消除 t < 0.5 时的额外精度问题
            left = nfb._lerp(a, b, 1 - (1 - t))
            right = nfb._lerp(b, a, 1 - t)
            # 断言左右对称的插值结果接近
            assert_allclose(left, right)

        # 测试线性插值函数处理零维输入的情况
        def test_linear_interpolation_formula_0d_inputs(self):
            # 创建包含单个元素的 NumPy 数组作为输入
            a = np.array(2)
            b = np.array(5)
            t = np.array(0.2)
            # 断言对于零维输入，插值函数的结果正确
            assert nfb._lerp(a, b, t) == 2.6


    class TestMedian:
    # 定义测试方法，用于测试 numpy 中 median 函数的各种用法
    def test_basic(self):
        # 创建包含单个元素的 numpy 数组
        a0 = np.array(1)
        # 创建从0到1的整数数组
        a1 = np.arange(2)
        # 创建从0到5的整数数组，并将其重塑为2行3列的数组
        a2 = np.arange(6).reshape(2, 3)
        # 断言：计算 a0 的中位数为1
        assert_equal(np.median(a0), 1)
        # 断言：计算 a1 的中位数接近0.5
        assert_allclose(np.median(a1), 0.5)
        # 断言：计算 a2 的中位数接近2.5
        assert_allclose(np.median(a2), 2.5)
        # 断言：计算 a2 沿着 axis=0 轴的中位数为 [1.5, 2.5, 3.5]
        assert_allclose(np.median(a2, axis=0), [1.5,  2.5,  3.5])
        # 断言：计算 a2 沿着 axis=1 轴的中位数为 [1, 4]
        assert_equal(np.median(a2, axis=1), [1, 4])
        # 断言：计算 a2 的扁平化版本（即不考虑轴向）的中位数为 2.5
        assert_allclose(np.median(a2, axis=None), 2.5)

        # 创建浮点数数组 a
        a = np.array([0.0444502, 0.0463301, 0.141249, 0.0606775])
        # 断言：计算 a[1] 和 a[3] 的均值等于 a 的中位数
        assert_almost_equal((a[1] + a[3]) / 2., np.median(a))
        # 创建浮点数数组 a
        a = np.array([0.0463301, 0.0444502, 0.141249])
        # 断言：计算 a[0] 等于 a 的中位数
        assert_equal(a[0], np.median(a))
        # 创建浮点数数组 a
        a = np.array([0.0444502, 0.141249, 0.0463301])
        # 断言：计算 a[-1] 等于 a 的中位数
        assert_equal(a[-1], np.median(a))
        # 断言：检查返回的中位数是一个标量
        assert_equal(np.median(a).ndim, 0)
        # 将 a 的第二个元素设为 NaN（Not a Number）
        a[1] = np.nan
        # 断言：检查返回的中位数是一个标量
        assert_equal(np.median(a).ndim, 0)

    # 定义测试轴关键字的方法
    def test_axis_keyword(self):
        # 创建二维整数数组 a3
        a3 = np.array([[2, 3],
                       [0, 1],
                       [6, 7],
                       [4, 5]])
        # 遍历数组 a3 和随机生成的数组，分别进行测试
        for a in [a3, np.random.randint(0, 100, size=(2, 3, 4))]:
            # 备份原始数组 a
            orig = a.copy()
            # 计算不考虑任何轴向的中位数
            np.median(a, axis=None)
            # 对数组 a 沿着每个轴进行中位数计算
            for ax in range(a.ndim):
                np.median(a, axis=ax)
            # 断言：原始数组 a 未被改变
            assert_array_equal(a, orig)

        # 断言：计算 a3 沿着 axis=0 轴的中位数为 [3, 4]
        assert_allclose(np.median(a3, axis=0), [3,  4])
        # 断言：计算 a3 转置后沿着 axis=1 轴的中位数为 [3, 4]
        assert_allclose(np.median(a3.T, axis=1), [3,  4])
        # 断言：计算 a3 的中位数为 3.5
        assert_allclose(np.median(a3), 3.5)
        # 断言：计算 a3 的扁平化版本的中位数为 3.5
        assert_allclose(np.median(a3, axis=None), 3.5)
        # 断言：计算 a3 转置后的扁平化版本的中位数为 3.5
        assert_allclose(np.median(a3.T), 3.5)
    def test_overwrite_keyword(self):
        # 创建多维数组 a3
        a3 = np.array([[2, 3],
                       [0, 1],
                       [6, 7],
                       [4, 5]])
        # 创建标量数组 a0
        a0 = np.array(1)
        # 创建一维数组 a1
        a1 = np.arange(2)
        # 创建二维数组 a2
        a2 = np.arange(6).reshape(2, 3)
        
        # 断言：对 a0 的拷贝求中位数，覆盖原输入
        assert_allclose(np.median(a0.copy(), overwrite_input=True), 1)
        # 断言：对 a1 的拷贝求中位数，覆盖原输入
        assert_allclose(np.median(a1.copy(), overwrite_input=True), 0.5)
        # 断言：对 a2 的拷贝求中位数，覆盖原输入
        assert_allclose(np.median(a2.copy(), overwrite_input=True), 2.5)
        # 断言：对 a2 的拷贝按列求中位数，覆盖原输入
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=0),
                        [1.5,  2.5,  3.5])
        # 断言：对 a2 的拷贝按行求中位数，覆盖原输入
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=1), [1, 4])
        # 断言：对 a2 的拷贝求全局中位数，覆盖原输入
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=None), 2.5)
        # 断言：对 a3 的拷贝按列求中位数，覆盖原输入
        assert_allclose(
            np.median(a3.copy(), overwrite_input=True, axis=0), [3,  4])
        # 断言：对 a3 的拷贝按行求中位数，覆盖原输入
        assert_allclose(np.median(a3.T.copy(), overwrite_input=True, axis=1),
                        [3,  4])

        # 创建三维数组 a4，对其打乱顺序
        a4 = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
        np.random.shuffle(a4.ravel())
        # 断言：对 a4 求全局中位数，使用拷贝并覆盖原输入
        assert_allclose(np.median(a4, axis=None),
                        np.median(a4.copy(), axis=None, overwrite_input=True))
        # 断言：对 a4 按列求中位数，使用拷贝并覆盖原输入
        assert_allclose(np.median(a4, axis=0),
                        np.median(a4.copy(), axis=0, overwrite_input=True))
        # 断言：对 a4 按行求中位数，使用拷贝并覆盖原输入
        assert_allclose(np.median(a4, axis=1),
                        np.median(a4.copy(), axis=1, overwrite_input=True))
        # 断言：对 a4 按深度方向求中位数，使用拷贝并覆盖原输入
        assert_allclose(np.median(a4, axis=2),
                        np.median(a4.copy(), axis=2, overwrite_input=True))

    def test_array_like(self):
        # 创建列表 x
        x = [1, 2, 3]
        # 断言：求列表 x 的中位数
        assert_almost_equal(np.median(x), 2)
        # 创建二维列表 x2
        x2 = [x]
        # 断言：求列表 x2 的中位数
        assert_almost_equal(np.median(x2), 2)
        # 断言：对列表 x2 按列求中位数，应与原列表 x 相同
        assert_allclose(np.median(x2, axis=0), x)

    def test_subclass(self):
        # gh-3846
        # 定义继承自 np.ndarray 的子类 MySubClass
        class MySubClass(np.ndarray):

            def __new__(cls, input_array, info=None):
                # 将输入数组转换为 np.ndarray，并视为 MySubClass 类型返回
                obj = np.asarray(input_array).view(cls)
                obj.info = info
                return obj

            # 自定义 mean 方法，返回固定值 -7
            def mean(self, axis=None, dtype=None, out=None):
                return -7

        # 创建 MySubClass 类的实例 a
        a = MySubClass([1, 2, 3])
        # 断言：对 MySubClass 实例 a 求中位数，应为 -7
        assert_equal(np.median(a), -7)

    @pytest.mark.parametrize('arr',
                             ([1., 2., 3.], [1., np.nan, 3.], np.nan, 0.))
    def test_subclass2(self, arr):
        """Check that we return subclasses, even if a NaN scalar."""
        # 定义继承自 np.ndarray 的子类 MySubclass
        class MySubclass(np.ndarray):
            pass

        # 将输入数组 arr 转换为 MySubclass 类型并求中位数
        m = np.median(np.array(arr).view(MySubclass))
        # 断言：返回值 m 是 MySubclass 类的实例
        assert isinstance(m, MySubclass)

    def test_out(self):
        # 创建全零数组 o
        o = np.zeros((4,))
        # 创建全一数组 d
        d = np.ones((3, 4))
        # 断言：对 d 按列求中位数，结果存入数组 o，且 o 应全为零
        assert_equal(np.median(d, 0, out=o), o)
        # 重新创建全零数组 o
        o = np.zeros((3,))
        # 断言：对 d 按行求中位数，结果存入数组 o，且 o 应全为零
        assert_equal(np.median(d, 1, out=o), o)
        # 重新创建全零数组 o
        o = np.zeros(())
        # 断言：对 d 求全局中位数，结果存入数组 o，且 o 应为零
        assert_equal(np.median(d, out=o), o)
    # 测试函数：测试在存在 NaN 值情况下的 np.median 行为
    def test_out_nan(self):
        # 捕获并记录所有警告
        with warnings.catch_warnings(record=True):
            # 设置运行时警告的过滤器
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 创建一个全零数组 o 和一个包含 NaN 值的数组 d
            o = np.zeros((4,))
            d = np.ones((3, 4))
            d[2, 1] = np.nan
            # 测试 np.median 函数在指定轴上计算中位数，并将结果存储到数组 o 中
            assert_equal(np.median(d, 0, out=o), o)
            # 重设 o 为全零数组，并测试 np.median 函数在另一轴上的中位数计算
            o = np.zeros((3,))
            assert_equal(np.median(d, 1, out=o), o)
            # 重设 o 为标量 0，并测试 np.median 函数在没有指定轴时的中位数计算
            o = np.zeros(())
            assert_equal(np.median(d, out=o), o)

    # 测试函数：测试不同情况下 NaN 值对 np.median 的影响
    def test_nan_behavior(self):
        # 创建一个浮点类型数组 a，并将其中一个元素设为 NaN
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        # 验证 np.median 函数在包含 NaN 值的数组上返回 NaN
        assert_equal(np.median(a), np.nan)
        # 验证 np.median 函数在指定轴上计算中位数时，如果存在 NaN 值则返回 NaN
        assert_equal(np.median(a, axis=0), np.nan)

        # 创建一个三维数组 a，并设置部分元素为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 测试 np.median 函数在没有指定轴时处理数组中的 NaN 值
        assert_equal(np.median(a), np.nan)
        # 验证 np.median 返回的结果维度为 0（标量）
        assert_equal(np.median(a).ndim, 0)

        # 在 axis=0 上计算中位数，并验证结果 b 中的 NaN 值处理
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 0), b)

        # 在 axis=1 上计算中位数，并验证结果 b 中的 NaN 值处理
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.median(a, 1), b)

        # 在 axis=(0, 2) 上计算中位数，并验证结果 b 中的 NaN 值处理
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.median(a, (0, 2)), b)

    # 根据条件跳过测试：在 WASM 环境下，浮点数错误无法正确处理
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work correctly")
    def test_empty(self):
        # 创建一个空的浮点数数组 a
        a = np.array([], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            # 设置运行时警告的过滤器
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 验证 np.median 函数在空数组上返回 NaN，并检查是否生成了 RuntimeWarning
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)
            # 检查警告数量是否为 2
            assert_equal(len(w), 2)

        # 创建一个三维的空数组 a，并在没有指定轴的情况下测试 np.median 函数处理 NaN 值的情况
        a = np.array([], dtype=float, ndmin=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 验证 np.median 函数在空数组上返回 NaN，并检查是否生成了 RuntimeWarning
            assert_equal(np.median(a), np.nan)
            assert_(w[0].category is RuntimeWarning)

        # 创建一个二维的空数组 b，并在指定轴上测试 np.median 函数处理 NaN 值的情况
        b = np.array([], dtype=float, ndmin=2)
        # 验证 np.median 函数在指定轴上返回的结果与预期的空数组 b 相等
        assert_equal(np.median(a, axis=0), b)
        assert_equal(np.median(a, axis=1), b)

        # 创建一个二维的数组 b，其中元素为 NaN，然后在指定轴上测试 np.median 函数处理 NaN 值的情况
        b = np.array(np.nan, dtype=float, ndmin=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            # 验证 np.median 函数在指定轴上处理包含 NaN 值的数组返回的结果与预期的数组 b 相等
            assert_equal(np.median(a, axis=2), b)
            assert_(w[0].category is RuntimeWarning)

    # 测试函数：测试 np.median 函数对对象数组的处理
    def test_object(self):
        # 创建一个浮点数数组 o，并验证在转换为对象类型后，np.median 返回结果类型为 float
        o = np.arange(7.)
        assert_(type(np.median(o.astype(object))), float)
        # 将数组 o 的某个元素设为 NaN，并再次验证 np.median 返回结果类型为 float
        o[2] = np.nan
        assert_(type(np.median(o.astype(object))), float)
    # 测试函数，用于验证 NumPy 中对于扩展轴的操作
    def test_extended_axis(self):
        # 创建一个形状为 (71, 23) 的随机数组成的矩阵 o
        o = np.random.normal(size=(71, 23))
        # 将 o 在第三个维度上重复 10 次，构成一个三维数组 x
        x = np.dstack([o] * 10)
        # 断言：计算沿着 (0, 1) 轴的中位数，与 o 的中位数相等
        assert_equal(np.median(x, axis=(0, 1)), np.median(o))
        # 将 x 的最后一个维度移动到第一个维度
        x = np.moveaxis(x, -1, 0)
        # 断言：计算沿着 (-2, -1) 轴的中位数，与 o 的中位数相等
        assert_equal(np.median(x, axis=(-2, -1)), np.median(o))
        # 交换 x 的第一和第二个维度，并复制数组
        x = x.swapaxes(0, 1).copy()
        # 断言：计算沿着 (0, -1) 轴的中位数，与 o 的中位数相等
        assert_equal(np.median(x, axis=(0, -1)), np.median(o))

        # 断言：计算沿着 (0, 1, 2) 轴的中位数，应与不指定轴的中位数相等
        assert_equal(np.median(x, axis=(0, 1, 2)), np.median(x, axis=None))
        # 断言：计算沿着 (0,) 轴的中位数，与沿着 0 轴的中位数相等
        assert_equal(np.median(x, axis=(0, )), np.median(x, axis=0))
        # 断言：计算沿着 (-1,) 轴的中位数，与沿着 -1 轴的中位数相等
        assert_equal(np.median(x, axis=(-1, )), np.median(x, axis=-1))

        # 创建一个形状为 (3, 5, 7, 11) 的数组 d，并随机打乱其中元素的顺序
        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        np.random.shuffle(d.ravel())
        # 断言：计算沿着 (0, 1, 2) 轴的中位数，与选取第一个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(0, 1, 2))[0],
                     np.median(d[:,:,:, 0].flatten()))
        # 断言：计算沿着 (0, 1, 3) 轴的中位数，与选取第二个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(0, 1, 3))[1],
                     np.median(d[:,:, 1,:].flatten()))
        # 断言：计算沿着 (3, 1, -4) 轴的中位数，与选取第三个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(3, 1, -4))[2],
                     np.median(d[:,:, 2,:].flatten()))
        # 断言：计算沿着 (3, 1, 2) 轴的中位数，与选取第四个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(3, 1, 2))[2],
                     np.median(d[2,:,:,:].flatten()))
        # 断言：计算沿着 (3, 2) 轴的中位数，与选取第四个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(3, 2))[2, 1],
                     np.median(d[2, 1,:,:].flatten()))
        # 断言：计算沿着 (1, -2) 轴的中位数，与选取第四个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(1, -2))[2, 1],
                     np.median(d[2,:,:, 1].flatten()))
        # 断言：计算沿着 (1, 3) 轴的中位数，与选取第四个维度的切片后计算的中位数相等
        assert_equal(np.median(d, axis=(1, 3))[2, 2],
                     np.median(d[2,:, 2,:].flatten()))

    # 测试函数，用于验证对于不合法的扩展轴参数，是否能正确抛出异常
    def test_extended_axis_invalid(self):
        # 创建一个全为 1 的数组 d，形状为 (3, 5, 7, 11)
        d = np.ones((3, 5, 7, 11))
        # 断言：当指定的轴为 -5 时，应该抛出 AxisError 异常
        assert_raises(AxisError, np.median, d, axis=-5)
        # 断言：当指定的轴为 (0, -5) 时，应该抛出 AxisError 异常
        assert_raises(AxisError, np.median, d, axis=(0, -5))
        # 断言：当指定的轴为 4 时，应该抛出 AxisError 异常
        assert_raises(AxisError, np.median, d, axis=4)
        # 断言：当指定的轴为 (0, 4) 时，应该抛出 AxisError 异常
        assert_raises(AxisError, np.median, d, axis=(0, 4))
        # 断言：当指定的轴为 (1, 1) 时，应该抛出 ValueError 异常
        assert_raises(ValueError, np.median, d, axis=(1, 1))

    # 测试函数，用于验证在保留维度的情况下计算中位数的结果
    def test_keepdims(self):
        # 创建一个全为 1 的数组 d，形状为 (3, 5, 7, 11)
        d = np.ones((3, 5, 7, 11))
        # 断言：计算不指定轴的中位数并保留所有维度，结果应该是形状为 (1, 1, 1, 1) 的数组
        assert_equal(np.median(d, axis=None, keepdims=True).shape,
                     (1, 1, 1, 1))
        # 断言：计算沿着 (0, 1) 轴的中位数并保留所有维度，结果应该是形状为 (1, 1, 7, 11) 的数组
        assert_equal(np.median(d, axis=(0, 1), keepdims=True).shape,
                     (1, 1, 7, 11))
        # 断言：计算沿着 (0, 3) 轴的中位数并保留所有维度，结果应该是形状为 (1, 5, 7, 1) 的数组
        assert_equal(np.median(d, axis=(0, 3), keepdims=True).shape,
                     (1, 5, 7, 1))
        # 断言：计算沿着 (1,) 轴的中位数并保留所有维度，结果应该是形状为 (3, 1, 7, 11) 的数组
        assert_equal(np.median(d, axis=(1,), keepdims=True).shape,
                     (3, 1, 7, 11))
        # 断言：计算沿着所有轴的中位数并保留所有维度，结果应该是形状为 (1, 1, 1, 1) 的数组
        assert_equal(np.median(d, axis=(0, 1, 2, 3), keepdims=True).shape,
                     (1, 1, 1, 1))
        # 断言：计算沿着 (0, 1, 3) 轴的中位数并保
    # 定义一个测试函数，用于测试在指定轴上保持维度的情况
    def test_keepdims_out(self, axis):
        # 创建一个形状为 (3, 5, 7, 11) 的全一数组 d
        d = np.ones((3, 5, 7, 11))
        # 如果 axis 参数为 None，则输出形状为 d 的维度全为 1 的元组
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            # 对轴参数进行规范化，确保在有效的轴范围内
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            # 根据规范化后的轴参数，构建输出形状 shape_out
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        # 创建一个空数组 out，形状为 shape_out
        out = np.empty(shape_out)
        # 计算数组 d 在指定轴上的中位数，并保持输出的维度不变，将结果存储在 out 中
        result = np.median(d, axis=axis, keepdims=True, out=out)
        # 断言计算得到的结果对象与 out 是同一个对象
        assert result is out
        # 断言计算得到的结果的形状与预期的 shape_out 相同
        assert_equal(result.shape, shape_out)

    @pytest.mark.parametrize("dtype", ["m8[s]"])
    @pytest.mark.parametrize("pos", [0, 23, 10])
    # 定义一个测试函数，用于测试 NaT 值的行为
    def test_nat_behavior(self, dtype, pos):
        # TODO: Median does not support Datetime, due to `mean`.
        # NaT and NaN should behave the same, do basic tests for NaT.
        # 创建一个 datetime64 类型的数组 a，其中一个位置设置为 NaT
        a = np.arange(0, 24, dtype=dtype)
        a[pos] = "NaT"
        # 计算数组 a 的中位数，期望结果的 dtype 与 a 的 dtype 相同
        res = np.median(a)
        assert res.dtype == dtype
        # 断言计算得到的结果是 NaT 类型
        assert np.isnat(res)
        # 计算数组 a 的分位数（30% 和 60%），期望结果的 dtype 与 a 的 dtype 相同
        res = np.percentile(a, [30, 60])
        assert res.dtype == dtype
        # 断言计算得到的结果都是 NaT 类型
        assert np.isnat(res).all()

        # 创建一个 datetime64 类型的二维数组 a，其中一个位置设置为 NaT
        a = np.arange(0, 24*3, dtype=dtype).reshape(-1, 3)
        a[pos, 1] = "NaT"
        # 计算数组 a 在 axis=0 上的中位数，期望结果的第二个元素是 NaT 类型，其余为非 NaT
        res = np.median(a, axis=0)
        assert_array_equal(np.isnat(res), [False, True, False])
# 定义名为 TestSortComplex 的测试类
class TestSortComplex:

    # 使用 pytest.mark.parametrize 装饰器，为 test_sort_real 方法定义参数化测试
    @pytest.mark.parametrize("type_in, type_out", [
        ('l', 'D'),  # 当 type_in 为 'l' 时，预期 type_out 为 'D'
        ('h', 'F'),  # 当 type_in 为 'h' 时，预期 type_out 为 'F'
        ('H', 'F'),  # 当 type_in 为 'H' 时，预期 type_out 为 'F'
        ('b', 'F'),  # 当 type_in 为 'b' 时，预期 type_out 为 'F'
        ('B', 'F'),  # 当 type_in 为 'B' 时，预期 type_out 为 'F'
        ('g', 'G'),  # 当 type_in 为 'g' 时，预期 type_out 为 'G'
        ])
    # 定义测试 sort_real，参数为 type_in 和 type_out
    def test_sort_real(self, type_in, type_out):
        # 准备测试数据，创建 numpy 数组 a，指定数据类型为 type_in
        a = np.array([5, 3, 6, 2, 1], dtype=type_in)
        # 使用 np.sort_complex() 对数组 a 进行排序
        actual = np.sort_complex(a)
        # 使用 np.sort() 对数组 a 进行排序，然后转换为 type_out 类型，作为预期结果
        expected = np.sort(a).astype(type_out)
        # 断言 actual 和 expected 相等
        assert_equal(actual, expected)
        # 断言 actual 的数据类型与 expected 的数据类型相等
        assert_equal(actual.dtype, expected.dtype)

    # 定义测试 sort_complex，测试处理复数输入的情况
    def test_sort_complex(self):
        # 准备测试数据，创建包含复数的 numpy 数组 a，数据类型为 'D' (双精度复数)
        a = np.array([2 + 3j, 1 - 2j, 1 - 3j, 2 + 1j], dtype='D')
        # 创建预期结果的 numpy 数组 expected
        expected = np.array([1 - 3j, 1 - 2j, 2 + 1j, 2 + 3j], dtype='D')
        # 使用 np.sort_complex() 对数组 a 进行复数排序
        actual = np.sort_complex(a)
        # 断言 actual 和 expected 相等
        assert_equal(actual, expected)
        # 断言 actual 的数据类型与 expected 的数据类型相等
        assert_equal(actual.dtype, expected.dtype)
```