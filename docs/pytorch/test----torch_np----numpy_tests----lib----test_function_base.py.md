# `.\pytorch\test\torch_np\numpy_tests\lib\test_function_base.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和函数
import functools  # 导入 functools 模块
import math  # 导入 math 模块
import operator  # 导入 operator 模块
import sys  # 导入 sys 模块
import warnings  # 导入 warnings 模块
from fractions import Fraction  # 从 fractions 模块导入 Fraction 类

# 导入 unittest 中的 expectedFailure 和 skipIf 别名为 xfail 和 skipif
from unittest import expectedFailure as xfail, skipIf as skipif

# 导入 hypothesis 库，并使用别名 st 作为 strategies
import hypothesis
import hypothesis.strategies as st

import numpy  # 导入 numpy 库

import pytest  # 导入 pytest 库
from hypothesis.extra.numpy import arrays  # 从 hypothesis.extra.numpy 导入 arrays
from pytest import raises as assert_raises  # 从 pytest 导入 raises 别名为 assert_raises

# 导入 torch.testing._internal.common_utils 中的特定函数和类
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

# 定义 skip 函数作为 functools.partial(skipif, True) 的别名
skip = functools.partial(skipif, True)

# 设置几个布尔型变量的值
HAS_REFCOUNT = True
IS_WASM = False
IS_PYPY = False

# FIXME: make from torch._numpy
# 下面的导入语句被注释掉，因为如果导入了这些函数，一些测试可能会因为错误的原因而通过
# from numpy lib import digitize, piecewise, trapz, select, trim_zeros, interp
from numpy.lib import delete, extract, insert, msort, place, setxor1d, unwrap, vectorize

# 根据 TEST_WITH_TORCHDYNAMO 的值选择导入 numpy 还是 torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        angle,
        bartlett,
        blackman,
        corrcoef,
        cov,
        diff,
        digitize,
        flipud,
        gradient,
        hamming,
        hanning,
        i0,
        interp,
        kaiser,
        meshgrid,
        sinc,
        trapz,
        trim_zeros,
        unique,
    )
    from numpy.core.numeric import normalize_axis_tuple  # 从 numpy.core.numeric 导入 normalize_axis_tuple
    from numpy.random import rand  # 从 numpy.random 导入 rand

    from numpy.testing import (
        assert_,  # IS_PYPY,
        assert_allclose,  # IS_PYPY,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,  # HAS_REFCOUNT, IS_WASM
    )
else:
    import torch._numpy as np
    from torch._numpy import (
        angle,
        bartlett,
        blackman,
        corrcoef,
        cov,
        diff,
        flipud,
        gradient,
        hamming,
        hanning,
        i0,
        kaiser,
        meshgrid,
        sinc,
        unique,
    )
    from torch._numpy._util import normalize_axis_tuple  # 从 torch._numpy._util 导入 normalize_axis_tuple
    from torch._numpy.random import rand  # 从 torch._numpy.random 导入 rand

    from torch._numpy.testing import (
        assert_,  # IS_PYPY,
        assert_allclose,  # IS_PYPY,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,  # HAS_REFCOUNT, IS_WASM
    )


# 定义函数 get_mat，生成一个 numpy 数组
def get_mat(n):
    data = np.arange(n)  # 创建一个长度为 n 的 numpy 数组
    # data = np.add.outer(data, data)  # 注释掉的代码，生成一个 n x n 的数组
    data = data[:, None] + data[None, :]  # 生成一个 n x n 的数组
    return data  # 返回生成的数组


def _make_complex(real, imag):
    """
    Like real + 1j * imag, but behaves as expected when imag contains non-finite
    values
    """
    ret = np.zeros(np.broadcast(real, imag).shape, np.complex_)
    ret.real = real
    ret.imag = imag
    return ret


    # 返回变量 ret 的值作为函数的返回结果
    return ret
# 定义一个测试类 TestRot90，用于测试 np.rot90 函数的不同用例
class TestRot90(TestCase):
    # 定义测试方法 test_basic，用于测试 np.rot90 函数在不同情况下是否能正确抛出异常
    def test_basic(self):
        # 测试 np.rot90 对于输入为一维数组时是否能抛出 ValueError 异常
        assert_raises(ValueError, np.rot90, np.ones(4))
        # 测试 np.rot90 对于输入为三维数组，且指定的轴超出数组维度时是否能抛出 ValueError 或 RuntimeError 异常
        assert_raises(
            (ValueError, RuntimeError), np.rot90, np.ones((2, 2, 2)), axes=(0, 1, 2)
        )
        # 测试 np.rot90 对于输入为二维数组，但指定的轴超出数组维度时是否能抛出 ValueError 异常
        assert_raises(ValueError, np.rot90, np.ones((2, 2)), axes=(0, 2))
        # 测试 np.rot90 对于输入为二维数组，但指定的轴重复时是否能抛出 ValueError 异常
        assert_raises(ValueError, np.rot90, np.ones((2, 2)), axes=(1, 1))
        # 测试 np.rot90 对于输入为三维数组，但指定的轴包含负数时是否能抛出 ValueError 异常
        assert_raises(ValueError, np.rot90, np.ones((2, 2, 2)), axes=(-2, 1))

        # 定义二维数组 a 和其预期的旋转结果 b1, b2, b3, b4
        a = [[0, 1, 2], [3, 4, 5]]
        b1 = [[2, 5], [1, 4], [0, 3]]
        b2 = [[5, 4, 3], [2, 1, 0]]
        b3 = [[3, 0], [4, 1], [5, 2]]
        b4 = [[0, 1, 2], [3, 4, 5]]

        # 使用循环测试 np.rot90 对于不同 k 值的旋转结果是否与预期相符
        for k in range(-3, 13, 4):
            assert_equal(np.rot90(a, k=k), b1)
        for k in range(-2, 13, 4):
            assert_equal(np.rot90(a, k=k), b2)
        for k in range(-1, 13, 4):
            assert_equal(np.rot90(a, k=k), b3)
        for k in range(0, 13, 4):
            assert_equal(np.rot90(a, k=k), b4)

        # 测试 np.rot90 的组合旋转是否能还原原始数组 a
        assert_equal(np.rot90(np.rot90(a, axes=(0, 1)), axes=(1, 0)), a)
        # 测试 np.rot90 在指定轴和 k 值时，结果是否等价于在不同轴和相反 k 值的情况
        assert_equal(np.rot90(a, k=1, axes=(1, 0)), np.rot90(a, k=-1, axes=(0, 1)))

    # 定义测试方法 test_axes，用于测试 np.rot90 在不同轴上的旋转结果是否正确
    def test_axes(self):
        # 创建一个形状为 (50, 40, 3) 的全为 1 的数组 a
        a = np.ones((50, 40, 3))
        # 测试默认情况下 np.rot90 对于轴 0 和 1 的旋转结果是否与对轴 0 和 -1 的旋转结果相等
        assert_equal(np.rot90(a).shape, (40, 50, 3))
        # 测试 np.rot90 对于轴 0 和 2 的旋转结果是否与对轴 0 和 -1 的旋转结果相等
        assert_equal(np.rot90(a, axes=(0, 2)), np.rot90(a, axes=(0, -1)))
        # 测试 np.rot90 对于轴 1 和 2 的旋转结果是否与对轴 -2 和 -1 的旋转结果相等
        assert_equal(np.rot90(a, axes=(1, 2)), np.rot90(a, axes=(-2, -1)))

    # 定义测试方法 test_rotation_axes，用于测试 np.rot90 在不同轴和 k 值下的旋转结果是否正确
    def test_rotation_axes(self):
        # 创建一个形状为 (2, 2, 2) 的数组 a，内容为 [0, 1, 2, 3, 4, 5, 6, 7]
        a = np.arange(8).reshape((2, 2, 2))

        # 定义数组 a 在不同轴和 k 值下的预期旋转结果
        a_rot90_01 = [[[2, 3], [6, 7]], [[0, 1], [4, 5]]]
        a_rot90_12 = [[[1, 3], [0, 2]], [[5, 7], [4, 6]]]
        a_rot90_20 = [[[4, 0], [6, 2]], [[5, 1], [7, 3]]]
        a_rot90_10 = [[[4, 5], [0, 1]], [[6, 7], [2, 3]]]

        # 使用 assert_equal 测试 np.rot90 在不同轴上旋转的结果是否与预期相符
        assert_equal(np.rot90(a, axes=(0, 1)), a_rot90_01)
        assert_equal(np.rot90(a, axes=(1, 0)), a_rot90_10)
        assert_equal(np.rot90(a, axes=(1, 2)), a_rot90_12)

        # 使用循环测试 np.rot90 在不同轴和 k 值下的旋转结果是否与预期相符
        for k in range(1, 5):
            assert_equal(
                np.rot90(a, k=k, axes=(2, 0)),
                np.rot90(a_rot90_20, k=k - 1, axes=(2, 0)),
            )


# 定义一个测试类 TestFlip，用于测试 np.flip 函数的不同用例
class TestFlip(TestCase):
    # 定义测试方法 test_axes，用于测试 np.flip 函数在不同情况下是否能正确抛出 AxisError 异常
    def test_axes(self):
        # 测试 np.flip 对于输入为一维数组时是否能抛出 AxisError 异常
        assert_raises(np.AxisError, np.flip, np.ones(4), axis=1)
        # 测试 np.flip 对于输入为二维数组，但指定的轴超出数组维度时是否能抛出 AxisError 异常
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=2)
        # 测试 np.flip 对于输入为二维数组，但指定的轴为负数且超出数组维度时是否能抛出 AxisError 异常
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=-3)
        # 测试 np.flip 对于输入为二维数组，但指定的轴包含超出数组维度的索引时是否能抛出 AxisError 异常
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))

    # 定义被跳过的测试方法 test_basic_lr，原因是该方法使用了不支持的 [::-1] 索引操作
    @skip(reason="no [::-1] indexing")
    def test_basic_lr(self):
        # 使用 get_mat 函数获取形状为 4 的矩阵 a
        a = get_mat(4)
        # 对数组 a 的第二个轴进行左右翻转，并将结果与预期的翻转结果 b 进行比较
        b = a[:, ::-1]
        assert_equal(np.flip(a, 1), b)
        # 对二维数组 a 进行上下翻转，并将结果与预期的翻转结果 b 进行比较
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[2, 1, 0], [5, 4, 3]]
        assert_equal(np.flip(a, 1),
    # 定义一个测试函数，用于测试在三维数组中交换不同轴的效果（轴0交换）
    def test_3d_swap_axis0(self):
        # 创建一个三维数组 a
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # 预期的交换轴0后的数组 b
        b = np.array([[[4, 5], [6, 7]], [[0, 1], [2, 3]]])

        # 断言交换轴0后数组 a 等于数组 b
        assert_equal(np.flip(a, 0), b)

    # 定义一个测试函数，用于测试在三维数组中交换不同轴的效果（轴1交换）
    def test_3d_swap_axis1(self):
        # 创建一个三维数组 a
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # 预期的交换轴1后的数组 b
        b = np.array([[[2, 3], [0, 1]], [[6, 7], [4, 5]]])

        # 断言交换轴1后数组 a 等于数组 b
        assert_equal(np.flip(a, 1), b)

    # 定义一个测试函数，用于测试在三维数组中交换不同轴的效果（轴2交换）
    def test_3d_swap_axis2(self):
        # 创建一个三维数组 a
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # 预期的交换轴2后的数组 b
        b = np.array([[[1, 0], [3, 2]], [[5, 4], [7, 6]]])

        # 断言交换轴2后数组 a 等于数组 b
        assert_equal(np.flip(a, 2), b)

    # 定义一个测试函数，用于测试在四维数组中交换任意轴的效果
    def test_4d(self):
        # 创建一个四维数组 a
        a = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        # 遍历数组的每个维度
        for i in range(a.ndim):
            # 断言使用 flip 方法交换第 i 轴后的数组等于先交换第0轴和第 i 轴，再交换回来后的数组
            assert_equal(np.flip(a, i), np.flipud(a.swapaxes(0, i)).swapaxes(i, 0))

    # 定义一个测试函数，用于测试在二维数组中默认在第0轴上的翻转效果
    def test_default_axis(self):
        # 创建一个二维数组 a
        a = np.array([[1, 2, 3], [4, 5, 6]])
        # 预期的在第0轴上的翻转后的数组 b
        b = np.array([[6, 5, 4], [3, 2, 1]])
        # 断言在第0轴上的翻转后数组 a 等于数组 b
        assert_equal(np.flip(a), b)

    # 定义一个测试函数，用于测试在三维数组中多个轴同时翻转的效果
    def test_multiple_axes(self):
        # 创建一个三维数组 a
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # 断言在空轴列表上翻转后的数组 a 等于数组 a 本身
        assert_equal(np.flip(a, axis=()), a)

        # 预期的在轴(0, 2)上翻转后的数组 b
        b = np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])
        # 断言在轴(0, 2)上翻转后数组 a 等于数组 b
        assert_equal(np.flip(a, axis=(0, 2)), b)

        # 预期的在轴(1, 2)上翻转后的数组 c
        c = np.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]])
        # 断言在轴(1, 2)上翻转后数组 a 等于数组 c
        assert_equal(np.flip(a, axis=(1, 2)), c)
class TestAny(TestCase):
    # 定义测试类 TestAny，继承自 TestCase
    def test_basic(self):
        # 定义基本测试方法 test_basic
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert_(np.any(y1))
        # 断言：y1 中存在至少一个非零元素
        assert_(np.any(y3))
        # 断言：y3 中存在至少一个非零元素
        assert_(not np.any(y2))
        # 断言：y2 中不存在非零元素

    def test_nd(self):
        # 定义多维测试方法 test_nd
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert_(np.any(y1))
        # 断言：y1 中存在至少一个非零元素
        assert_array_equal(np.sometrue(y1, axis=0), [1, 1, 0])
        # 断言：沿着列方向（axis=0），y1 中至少有一个元素为真，期望结果为 [1, 1, 0]
        assert_array_equal(np.sometrue(y1, axis=1), [0, 1, 1])
        # 断言：沿着行方向（axis=1），y1 中至少有一个元素为真，期望结果为 [0, 1, 1]


class TestAll(TestCase):
    # 定义测试类 TestAll，继承自 TestCase
    def test_basic(self):
        # 定义基本测试方法 test_basic
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert_(not np.all(y1))
        # 断言：y1 中并非所有元素都为真
        assert_(np.all(y3))
        # 断言：y3 中所有元素都为真
        assert_(not np.all(y2))
        # 断言：y2 中并非所有元素都为真
        assert_(np.all(~np.array(y2)))
        # 断言：取 y2 的反转结果后，所有元素都为真

    def test_nd(self):
        # 定义多维测试方法 test_nd
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert_(not np.all(y1))
        # 断言：y1 中并非所有元素都为真
        assert_array_equal(np.alltrue(y1, axis=0), [0, 0, 1])
        # 断言：沿着列方向（axis=0），y1 中所有元素都为真，期望结果为 [0, 0, 1]
        assert_array_equal(np.alltrue(y1, axis=1), [0, 0, 1])
        # 断言：沿着行方向（axis=1），y1 中所有元素都为真，期望结果为 [0, 0, 1]


class TestCopy(TestCase):
    # 定义测试类 TestCopy，继承自 TestCase
    def test_basic(self):
        # 定义基本测试方法 test_basic
        a = np.array([[1, 2], [3, 4]])
        a_copy = np.copy(a)
        # 复制数组 a 得到 a_copy
        assert_array_equal(a, a_copy)
        # 断言：a 和 a_copy 的内容相同
        a_copy[0, 0] = 10
        # 修改 a_copy 的第一个元素
        assert_equal(a[0, 0], 1)
        # 断言：a 的第一个元素未被修改
        assert_equal(a_copy[0, 0], 10)
        # 断言：a_copy 的第一个元素已被修改

    @xpassIfTorchDynamo  # (reason="order='F' not implemented")
    def test_order(self):
        # 定义测试方法 test_order，并标记为条件通过
        # It turns out that people rely on np.copy() preserving order by
        # default; changing this broke scikit-learn:
        # github.com/scikit-learn/scikit-learn/commit/7842748cf777412c506
        a = np.array([[1, 2], [3, 4]])
        assert_(a.flags.c_contiguous)
        # 断言：数组 a 是 C 连续存储的
        assert_(not a.flags.f_contiguous)
        # 断言：数组 a 不是 Fortran 连续存储的
        a_fort = np.array([[1, 2], [3, 4]], order="F")
        # 创建 Fortran 顺序的数组 a_fort
        assert_(not a_fort.flags.c_contiguous)
        # 断言：数组 a_fort 不是 C 连续存储的
        assert_(a_fort.flags.f_contiguous)
        # 断言：数组 a_fort 是 Fortran 连续存储的
        a_copy = np.copy(a)
        # 复制数组 a 得到 a_copy
        assert_(a_copy.flags.c_contiguous)
        # 断言：数组 a_copy 是 C 连续存储的
        assert_(not a_copy.flags.f_contiguous)
        # 断言：数组 a_copy 不是 Fortran 连续存储的
        a_fort_copy = np.copy(a_fort)
        # 复制数组 a_fort 得到 a_fort_copy
        assert_(not a_fort_copy.flags.c_contiguous)
        # 断言：数组 a_fort_copy 不是 C 连续存储的
        assert_(a_fort_copy.flags.f_contiguous)
        # 断言：数组 a_fort_copy 是 Fortran 连续存储的


@instantiate_parametrized_tests
# 根据参数化测试实例化测试类
class TestAverage(TestCase):
    # 定义测试类 TestAverage，继承自 TestCase
    def test_basic(self):
        # 定义基本测试方法 test_basic
        y1 = np.array([1, 2, 3])
        assert_(np.average(y1, axis=0) == 2.0)
        # 断言：计算 y1 的平均值（沿轴0），并断言其为2.0
        y2 = np.array([1.0, 2.0, 3.0])
        assert_(np.average(y2, axis=0) == 2.0)
        # 断言：计算 y2 的平均值（沿轴0），并断言其为2.0
        y3 = [0.0, 0.0, 0.0]
        assert_(np.average(y3, axis=0) == 0.0)
        # 断言：计算 y3 的平均值（沿轴0），并断言其为0.0

        y4 = np.ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        # 创建一个全为1的4x4数组 y4，并修改其部分元素
        assert_almost_equal(y4.mean(0), np.average(y4, 0))
        # 断言：y4 沿轴0的平均值近似等于沿轴0的平均值
        assert_almost_equal(y4.mean(1), np.average(y4, 1))
        # 断言：y4 沿轴1的平均值近似等于沿轴1的平均值

        y5 = rand(5, 5)
        # 创建一个随机5x5数组 y5
        assert_almost_equal(y5.mean(0), np.average(y5, 0))
        # 断言：y5 沿轴0的平均值近似等于沿轴0的平均值
        assert_almost_equal(y5.mean(1), np.average(y5, 1))
        # 断言：y5 沿轴1的平均值近似等于沿轴1的平均值

    @skip(reason="NP_VER: fails on CI")
    # 标记为跳过，原因为 "NP_VER: fails on CI"
    @parametrize(
        "x, axis, expected_avg, weights, expected_wavg, expected_wsum",
        [  # 参数化测试用例，包括多组输入和期望输出
            ([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),  # 第一组测试参数和期望结果
            (
                [[1, 2, 5], [1, 6, 11]],
                0,
                [[1.0, 4.0, 8.0]],  # 第二组测试参数的期望平均值
                [1, 3],  # 第二组测试参数的权重
                [[1.0, 5.0, 9.5]],  # 第二组测试参数的加权平均值
                [[4, 4, 4]],  # 第二组测试参数的加权和
            ),
        ],
    )
    def test_basic_keepdims(
        self, x, axis, expected_avg, weights, expected_wavg, expected_wsum
    ):
        # 计算未加权平均值，并验证其形状和期望值是否一致
        avg = np.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        # 计算加权平均值，并验证其形状和期望值是否一致
        wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        # 计算加权平均值和加权和，并验证它们的形状和期望值是否一致
        wavg, wsum = np.average(
            x, axis=axis, weights=weights, returned=True, keepdims=True
        )
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    @skip(reason="NP_VER: fails on CI")  # 标记为跳过测试，并注明原因
    def test_weights(self):
        y = np.arange(10)
        w = np.arange(10)
        actual = np.average(y, weights=w)
        desired = (np.arange(10) ** 2).sum() * 1.0 / np.arange(10).sum()
        assert_almost_equal(actual, desired)

        y1 = np.array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        actual = np.average(y1, weights=w0, axis=0)
        desired = np.array([3.0, 4.0, 5.0])
        assert_almost_equal(actual, desired)

        w1 = [0, 0, 1]
        actual = np.average(y1, weights=w1, axis=1)
        desired = np.array([3.0, 6.0])
        assert_almost_equal(actual, desired)

        # 这里应该引发一个错误。我们可以测试这个吗？
        # assert_equal(average(y1, weights=w1), 9./2.)

        # 二维情况
        w2 = [[0, 0, 1], [0, 0, 2]]
        desired = np.array([3.0, 6.0])
        assert_array_equal(np.average(y1, weights=w2, axis=1), desired)
        assert_equal(np.average(y1, weights=w2), 5.0)

        y3 = rand(5).astype(np.float32)
        w3 = rand(5).astype(np.float64)

        assert_(np.average(y3, weights=w3).dtype == np.result_type(y3, w3))

        # 测试 `keepdims=False` 和 `keepdims=True` 的权重计算
        x = np.array([2, 3, 4]).reshape(3, 1)
        w = np.array([4, 5, 6]).reshape(3, 1)

        actual = np.average(x, weights=w, axis=1, keepdims=False)
        desired = np.array([2.0, 3.0, 4.0])
        assert_array_equal(actual, desired)

        actual = np.average(x, weights=w, axis=1, keepdims=True)
        desired = np.array([[2.0], [3.0], [4.0]])
        assert_array_equal(actual, desired)
    def test_returned(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])

        # No weights
        # 计算 y 的平均值，同时返回总权重
        avg, scl = np.average(y, returned=True)
        assert_equal(scl, 6.0)

        # 计算 y 按列的平均值，并返回每列的总权重
        avg, scl = np.average(y, 0, returned=True)
        assert_array_equal(scl, np.array([2.0, 2.0, 2.0]))

        # 计算 y 按行的平均值，并返回每行的总权重
        avg, scl = np.average(y, 1, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0]))

        # With weights
        w0 = [1, 2]
        # 使用权重 w0 计算 y 按列的加权平均值，并返回每列的总权重
        avg, scl = np.average(y, weights=w0, axis=0, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0, 3.0]))

        w1 = [1, 2, 3]
        # 使用权重 w1 计算 y 按行的加权平均值，并返回每行的总权重
        avg, scl = np.average(y, weights=w1, axis=1, returned=True)
        assert_array_equal(scl, np.array([6.0, 6.0]))

        w2 = [[0, 0, 1], [1, 2, 3]]
        # 使用权重 w2 计算 y 按行的加权平均值，并返回每行的总权重
        avg, scl = np.average(y, weights=w2, axis=1, returned=True)
        assert_array_equal(scl, np.array([1.0, 6.0]))

    def test_upcasting(self):
        typs = [
            ("i4", "i4", "f8"),
            ("i4", "f4", "f8"),
            ("f4", "i4", "f8"),
            ("f4", "f4", "f4"),
            ("f4", "f8", "f8"),
        ]
        for at, wt, rt in typs:
            a = np.array([[1, 2], [3, 4]], dtype=at)
            w = np.array([[1, 2], [3, 4]], dtype=wt)
            # 使用不同的数据类型计算加权平均值，并验证结果的数据类型
            assert_equal(np.average(a, weights=w).dtype, np.dtype(rt))

    @skip(reason="support Fraction objects?")
    def test_average_class_without_dtype(self):
        # see gh-21988
        a = np.array([Fraction(1, 5), Fraction(3, 5)])
        # 使用分数对象计算平均值，目前不支持，因此跳过测试
        assert_equal(np.average(a), Fraction(2, 5))
# 标记测试类为预期失败，理由是待实现
@xfail  # (reason="TODO: implement")
class TestSelect(TestCase):
    # 定义选项列表，包含三个 NumPy 数组
    choices = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    # 定义条件列表，每个条件为一个 NumPy 数组
    conditions = [
        np.array([False, False, False]),
        np.array([False, True, False]),
        np.array([False, False, True]),
    ]

    # 定义私有方法 `_select`，用于根据条件选择值
    def _select(self, cond, values, default=0):
        output = []
        # 遍历条件列表
        for m in range(len(cond)):
            # 如果条件满足，则将对应值添加到输出列表中，否则添加默认值
            output += [V[m] for V, C in zip(values, cond) if C[m]] or [default]
        return output

    # 测试基本功能
    def test_basic(self):
        choices = self.choices
        conditions = self.conditions
        # 断言 select 函数的输出与 _select 方法的输出相等
        assert_array_equal(
            select(conditions, choices, default=15),
            self._select(conditions, choices, default=15),
        )
        # 断言选项列表的长度为 3
        assert_equal(len(choices), 3)
        # 断言条件列表的长度为 3
        assert_equal(len(conditions), 3)

    # 测试广播功能
    def test_broadcasting(self):
        conditions = [np.array(True), np.array([False, True, False])]
        choices = [1, np.arange(12).reshape(4, 3)]
        # 断言 select 函数的输出与预期输出相等
        assert_array_equal(select(conditions, choices), np.ones((4, 3)))
        # 断言默认值可以进行广播
        assert_equal(select([True], [0], default=[0]).shape, (1,))

    # 测试返回值的数据类型
    def test_return_dtype(self):
        # 断言 select 函数的输出数据类型为复数
        assert_equal(select(self.conditions, self.choices, 1j).dtype, np.complex_)
        # 断言如果条件是标量默认值，则选择的数组元素需要更强
        choices = [choice.astype(np.int8) for choice in self.choices]
        assert_equal(select(self.conditions, choices).dtype, np.int8)

        d = np.array([1, 2, 3, np.nan, 5, 7])
        m = np.isnan(d)
        # 断言根据条件选择的结果
        assert_equal(select([m], [d]), [0, 0, 0, np.nan, 0, 0])

    # 测试废弃的空参数
    def test_deprecated_empty(self):
        # 断言抛出 ValueError 异常，因为选择和条件列表为空
        assert_raises(ValueError, select, [], [], 3j)
        assert_raises(ValueError, select, [], [])

    # 测试非布尔类型的废弃
    def test_non_bool_deprecation(self):
        choices = self.choices
        conditions = self.conditions[:]
        conditions[0] = conditions[0].astype(np.int_)
        # 断言抛出 TypeError 异常，因为条件类型不是布尔型
        assert_raises(TypeError, select, conditions, choices)
        conditions[0] = conditions[0].astype(np.uint8)
        assert_raises(TypeError, select, conditions, choices)
        assert_raises(TypeError, select, conditions, choices)

    # 测试多参数
    def test_many_arguments(self):
        # 断言选择函数支持多个参数
        # 这个测试过去受到 NPY_MAXARGS == 32 的限制
        conditions = [np.array([False])] * 100
        choices = [np.array([1])] * 100
        select(conditions, choices)


# 标记测试类为通过，如果 Torch Dynamo 可用
@xpassIfTorchDynamo  # (reason="TODO: implement")
# 实例化参数化测试类
@instantiate_parametrized_tests
class TestInsert(TestCase):
    # 定义一个测试函数，用于测试插入操作的不同情况
    def test_basic(self):
        # 创建一个列表 a
        a = [1, 2, 3]
        # 测试在索引 0 处插入值 1
        assert_equal(insert(a, 0, 1), [1, 1, 2, 3])
        # 测试在索引 3 处插入值 1
        assert_equal(insert(a, 3, 1), [1, 2, 3, 1])
        # 测试在索引 [1, 1, 1] 处插入列表 [1, 2, 3]
        assert_equal(insert(a, [1, 1, 1], [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        # 测试在索引 1 处插入列表 [1, 2, 3]
        assert_equal(insert(a, 1, [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        # 测试在索引 [1, -1, 3] 处插入值 9
        assert_equal(insert(a, [1, -1, 3], 9), [1, 9, 2, 9, 3, 9])
        # 测试在切片索引 -1 到末尾处插入值 9
        assert_equal(insert(a, slice(-1, None, -1), 9), [9, 1, 9, 2, 9, 3])
        # 测试在索引 [-1, 1, 3] 处插入列表 [7, 8, 9]
        assert_equal(insert(a, [-1, 1, 3], [7, 8, 9]), [1, 8, 2, 7, 3, 9])
        
        # 创建一个 numpy 数组 b，指定数据类型为 np.float64
        b = np.array([0, 1], dtype=np.float64)
        # 测试在索引 0 处插入 b[0] 的值
        assert_equal(insert(b, 0, b[0]), [0.0, 0.0, 1.0])
        # 测试插入空列表时的情况
        assert_equal(insert(b, [], []), b)
        
        # 未来可能会对布尔类型处理方式有所不同：
        # 使用警告捕获器，测试在索引处插入布尔值数组的情况，并确保引发 FutureWarning 警告
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", FutureWarning)
            assert_equal(insert(a, np.array([True] * 4), 9), [1, 9, 9, 9, 9, 2, 3])
            assert_(w[0].category is FutureWarning)

    # 测试多维数组的插入操作
    def test_multidim(self):
        # 创建一个二维列表 a
        a = [[1, 1, 1]]
        # 预期插入值 [1] 后的结果为 [1, 1, 1, 1]
        assert_equal(insert(a, 0, [1]), [1, 1, 1, 1])
        # 预期在 axis=0 上插入值 [2, 2, 2] 后的结果为 [[2, 2, 2], [1, 1, 1]]
        assert_equal(insert(a, 0, [2, 2, 2], axis=0), [[2, 2, 2], [1, 1, 1]])
        # 预期在 axis=0 上插入值 2 后的结果为 [[2, 2, 2], [1, 1, 1]]
        assert_equal(insert(a, 0, 2, axis=0), [[2, 2, 2], [1, 1, 1]])
        # 预期在 axis=1 上插入值 2 后的结果为 [[1, 1, 2, 1]]
        assert_equal(insert(a, 2, 2, axis=1), [[1, 1, 2, 1]])

        # 创建一个二维 numpy 数组 a
        a = np.array([[1, 1], [2, 2], [3, 3]])
        # 创建一个预期插入操作后的结果数组 b
        b = np.arange(1, 4).repeat(3).reshape(3, 3)
        # 创建一个预期插入操作后的结果数组 c
        c = np.concatenate(
            (a[:, 0:1], np.arange(1, 4).repeat(3).reshape(3, 3).T, a[:, 1:2]), axis=1
        )
        # 在 axis=1 上插入列表 [[1], [2], [3]] 后，预期结果为 b
        assert_equal(insert(a, [1], [[1], [2], [3]], axis=1), b)
        # 在 axis=1 上插入列表 [1, 2, 3] 后，预期结果为 c
        assert_equal(insert(a, [1], [1, 2, 3], axis=1), c)
        # 在 axis=1 上插入标量 [1, 2, 3] 后，预期结果为 b（与插入列表相反的情况）
        assert_equal(insert(a, 1, [1, 2, 3], axis=1), b)
        # 在 axis=1 上插入列表 [[1], [2], [3]] 后，预期结果为 c
        assert_equal(insert(a, 1, [[1], [2], [3]], axis=1), c)

        # 创建一个二维数组 a
        a = np.arange(4).reshape(2, 2)
        # 在 axis=1 上插入 a[:, 1] 的值后，预期结果为 a
        assert_equal(insert(a[:, :1], 1, a[:, 1], axis=1), a)
        # 在 axis=0 上插入 a[1, :] 的值后，预期结果为 a
        assert_equal(insert(a[:1, :], 1, a[1, :], axis=0), a)

        # 测试负数索引 axis=-1 的情况
        a = np.arange(24).reshape((2, 3, 4))
        # 在 axis=-1 上插入 a[:, :, 3] 的值后，与 axis=2 上相同插入操作结果相同
        assert_equal(
            insert(a, 1, a[:, :, 3], axis=-1), insert(a, 1, a[:, :, 3], axis=2)
        )
        # 在 axis=-2 上插入 a[:, 2, :] 的值后，与 axis=1 上相同插入操作结果相同
        assert_equal(
            insert(a, 1, a[:, 2, :], axis=-2), insert(a, 1, a[:, 2, :], axis=1)
        )

        # 测试无效的 axis 值
        assert_raises(np.AxisError, insert, a, 1, a[:, 2, :], axis=3)
        assert_raises(np.AxisError, insert, a, 1, a[:, 2, :], axis=-4)

        # 负数索引 axis=-1 的情况
        a = np.arange(24).reshape((2, 3, 4))
        # 在 axis=-1 上插入 a[:, :, 3] 的值后，与 axis=2 上相同插入操作结果相同
        assert_equal(
            insert(a, 1, a[:, :, 3], axis=-1), insert(a, 1, a[:, :, 3], axis=2)
        )
        # 在 axis=-2 上插入 a[:, 2, :] 的值后，与 axis=1 上相同插入操作结果相同
        assert_equal(
            insert(a, 1, a[:, 2, :], axis=-2), insert(a, 1, a[:, 2, :], axis=1)
        )
    # 定义一个测试方法，测试在特定情况下插入操作的异常情况

    def test_0d(self):
        # 创建一个标量的 NumPy 数组
        a = np.array(1)
        # 使用 pytest 来验证在指定轴向插入空列表时是否会引发 AxisError 异常
        with pytest.raises(np.AxisError):
            insert(a, [], 2, axis=0)
        # 使用 pytest 来验证在指定轴向插入字符串轴向时是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            insert(a, [], 2, axis="nonsense")

    # 定义一个测试方法，验证插入操作中索引数组是否被复制

    def test_index_array_copied(self):
        # 创建一个 NumPy 数组 x
        x = np.array([1, 1, 1])
        # 调用 np.insert() 在索引数组中插入值，但未将结果赋值给任何变量
        np.insert([0, 1, 2], x, [3, 4, 5])
        # 使用 assert_equal 验证数组 x 是否仍然是原始值，表明索引数组在插入时被复制
        assert_equal(x, np.array([1, 1, 1]))

    # 定义一个测试方法，验证在插入操作中使用浮点数索引时是否会引发 IndexError 异常

    def test_index_floats(self):
        # 使用 pytest 验证在插入操作中使用浮点数索引是否会引发 IndexError 异常
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([1.0, 2.0]), [10, 20])
        # 使用 pytest 验证在插入操作中使用空的浮点数索引是否会引发 IndexError 异常
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([], dtype=float), [])

    # 定义一个带有参数化装饰器的测试方法，测试超出边界的索引插入操作

    @skip(reason="NP_VER: fails on CI")
    @parametrize("idx", [4, -4])
    def test_index_out_of_bounds(self, idx):
        # 使用 pytest 验证在超出边界的索引插入操作中是否会引发 IndexError 异常，并且异常信息包含 "out of bounds"
        with pytest.raises(IndexError, match="out of bounds"):
            np.insert([0, 1, 2], [idx], [3, 4])
class TestAmax(TestCase):
    # 测试 np.amax 函数的基本用法
    def test_basic(self):
        # 定义一个包含数字的列表
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 断言列表中的最大值为 10.0
        assert_equal(np.amax(a), 10.0)
        # 定义一个包含数字的二维列表
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        # 断言二维列表在每列上的最大值
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        # 断言二维列表在每行上的最大值
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])


class TestAmin(TestCase):
    # 测试 np.amin 函数的基本用法
    def test_basic(self):
        # 定义一个包含数字的列表
        a = [3, 4, 5, 10, -3, -5, 6.0]
        # 断言列表中的最小值为 -5.0
        assert_equal(np.amin(a), -5.0)
        # 定义一个包含数字的二维列表
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        # 断言二维列表在每列上的最小值
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        # 断言二维列表在每行上的最小值
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])


class TestPtp(TestCase):
    # 测试 np.ptp 函数的基本用法
    def test_basic(self):
        # 创建一个 NumPy 数组
        a = np.array([3, 4, 5, 10, -3, -5, 6.0])
        # 断言数组在某一轴上的峰值-峰值距离为 15.0
        assert_equal(a.ptp(axis=0), 15.0)
        # 创建一个二维 NumPy 数组
        b = np.array([[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]])
        # 断言数组在每列上的峰值-峰值距离
        assert_equal(b.ptp(axis=0), [5.0, 7.0, 7.0])
        # 断言数组在每行上的峰值-峰值距离
        assert_equal(b.ptp(axis=-1), [6.0, 6.0, 6.0])
        # 断言数组在每列上的峰值-峰值距离并保持维度
        assert_equal(b.ptp(axis=0, keepdims=True), [[5.0, 7.0, 7.0]])
        # 断言数组在多轴上的峰值-峰值距离并保持维度
        assert_equal(b.ptp(axis=(0, 1), keepdims=True), [[8.0]])


class TestCumsum(TestCase):
    # 测试 np.cumsum 函数的基本用法
    def test_basic(self):
        # 定义一个包含数字的列表
        ba = [1, 2, 10, 11, 6, 5, 4]
        # 定义一个包含数字的二维列表
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        # 针对不同的数据类型执行累积和计算
        for ctype in [
            np.int8,
            np.uint8,
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            # 将列表转换为指定数据类型的 NumPy 数组
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)

            # 预期的累积和结果
            tgt = np.array([1, 3, 13, 24, 30, 35, 39], ctype)
            # 断言在指定轴上的累积和
            assert_array_equal(np.cumsum(a, axis=0), tgt)

            # 预期的二维累积和结果
            tgt = np.array([[1, 2, 3, 4], [6, 8, 10, 13], [16, 11, 14, 18]], ctype)
            # 断言在指定轴上的累积和
            assert_array_equal(np.cumsum(a2, axis=0), tgt)

            # 预期的二维累积和结果
            tgt = np.array([[1, 3, 6, 10], [5, 11, 18, 27], [10, 13, 17, 22]], ctype)
            # 断言在指定轴上的累积和
            assert_array_equal(np.cumsum(a2, axis=1), tgt)


class TestProd(TestCase):
    # 测试 np.prod 函数的基本用法
    def test_basic(self):
        # 定义一个包含数字的列表
        ba = [1, 2, 10, 11, 6, 5, 4]
        # 定义一个包含数字的二维列表
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        # 针对不同的数据类型执行累积乘计算
        for ctype in [
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            # 将列表转换为指定数据类型的 NumPy 数组
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)

            # 如果数据类型在指定范围内，则引发算术错误
            if ctype in ["1", "b"]:
                assert_raises(ArithmeticError, np.prod, a)
                assert_raises(ArithmeticError, np.prod, a2, 1)
            else:
                # 断言在指定轴上的累积乘积
                assert_equal(a.prod(axis=0), 26400)
                # 断言在指定轴上的二维累积乘积
                assert_array_equal(a2.prod(axis=0), np.array([50, 36, 84, 180], ctype))
                # 断言在指定轴上的二维累积乘积
                assert_array_equal(a2.prod(axis=-1), np.array([24, 1890, 600], ctype))
    # 定义一个测试方法，用于测试 numpy 的 cumprod 函数在不同数据类型下的行为
    def test_basic(self):
        # 创建一个整数和浮点数混合的列表
        ba = [1, 2, 10, 11, 6, 5, 4]
        # 创建一个包含多个整数和浮点数列表的列表
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        # 遍历不同的 numpy 数据类型
        for ctype in [
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            # 使用当前数据类型创建 numpy 数组 a 和 a2
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            # 如果当前数据类型在指定的异常列表中
            if ctype in ["1", "b"]:
                # 断言在执行 np.cumprod(a) 和 np.cumprod(a2, 1) 时会抛出 ArithmeticError 异常
                assert_raises(ArithmeticError, np.cumprod, a)
                assert_raises(ArithmeticError, np.cumprod, a2, 1)
                assert_raises(ArithmeticError, np.cumprod, a)
            else:
                # 断言 np.cumprod(a, axis=-1) 的结果与预期结果相等
                assert_array_equal(
                    np.cumprod(a, axis=-1),
                    np.array([1, 2, 20, 220, 1320, 6600, 26400], ctype),
                )
                # 断言 np.cumprod(a2, axis=0) 的结果与预期结果相等
                assert_array_equal(
                    np.cumprod(a2, axis=0),
                    np.array([[1, 2, 3, 4], [5, 12, 21, 36], [50, 36, 84, 180]], ctype),
                )
                # 断言 np.cumprod(a2, axis=-1) 的结果与预期结果相等
                assert_array_equal(
                    np.cumprod(a2, axis=-1),
                    np.array(
                        [[1, 2, 6, 24], [5, 30, 210, 1890], [10, 30, 120, 600]], ctype
                    ),
                )
class TestDiff(TestCase):
    def test_basic(self):
        x = [1, 4, 6, 7, 12]
        out = np.array([3, 2, 1, 5])
        out2 = np.array([-1, -1, 4])
        out3 = np.array([0, 5])
        # 检查 diff 函数对 x 的计算结果是否与预期的 out 相等
        assert_array_equal(diff(x), out)
        # 检查 diff 函数对 x 的计算结果（n=2）是否与预期的 out2 相等
        assert_array_equal(diff(x, n=2), out2)
        # 检查 diff 函数对 x 的计算结果（n=3）是否与预期的 out3 相等
        assert_array_equal(diff(x, n=3), out3)

        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        out = np.array([1.1, 0.8, -3.2, 0.1])
        # 检查 diff 函数对浮点数列表 x 的计算结果是否与预期的 out 相近
        assert_almost_equal(diff(x), out)

        x = [True, True, False, False]
        out = np.array([False, True, False])
        out2 = np.array([True, True])
        # 检查 diff 函数对布尔值列表 x 的计算结果是否与预期的 out 相等
        assert_array_equal(diff(x), out)
        # 检查 diff 函数对布尔值列表 x 的计算结果（n=2）是否与预期的 out2 相等
        assert_array_equal(diff(x, n=2), out2)

    def test_axis(self):
        x = np.zeros((10, 20, 30))
        x[:, 1::2, :] = 1
        exp = np.ones((10, 19, 30))
        exp[:, 1::2, :] = -1
        # 检查在不同轴上应用 diff 函数的结果是否与预期相等
        assert_array_equal(diff(x), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=-1), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=0), np.zeros((9, 20, 30)))
        assert_array_equal(diff(x, axis=1), exp)
        assert_array_equal(diff(x, axis=-2), exp)
        # 检查当指定不存在的轴时，diff 函数是否会引发 AxisError 异常
        assert_raises(np.AxisError, diff, x, axis=3)
        assert_raises(np.AxisError, diff, x, axis=-4)

        x = np.array(1.11111111111, np.float64)
        # 检查当输入不是数组而是标量时，diff 函数是否会引发 ValueError 异常
        assert_raises(ValueError, diff, x)

    def test_nd(self):
        x = 20 * rand(10, 20, 30)
        out1 = x[:, :, 1:] - x[:, :, :-1]
        out2 = out1[:, :, 1:] - out1[:, :, :-1]
        out3 = x[1:, :, :] - x[:-1, :, :]
        out4 = out3[1:, :, :] - out3[:-1, :, :]
        # 检查在多维数组上应用 diff 函数的结果是否与预期相等
        assert_array_equal(diff(x), out1)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, axis=0), out3)
        assert_array_equal(diff(x, n=2, axis=0), out4)

    def test_n(self):
        x = list(range(3))
        # 检查当 n 为负数时，diff 函数是否会引发 ValueError 异常
        assert_raises(ValueError, diff, x, n=-1)
        output = [diff(x, n=n) for n in range(1, 5)]
        expected_output = [[1, 1], [0], [], []]
        # 检查对于不同的 n 值，diff 函数的输出是否符合预期
        for n, (expected, out) in enumerate(zip(expected_output, output), start=1):
            assert_(type(out) is np.ndarray)
            assert_array_equal(out, expected)
            assert_equal(out.dtype, np.int_)
            assert_equal(len(out), max(0, len(x) - n))
    # 定义一个测试函数，用于测试 diff 函数的 prepend 参数
    def test_prepend(self):
        # 创建一个包含数字 1 到 5 的 NumPy 数组
        x = np.arange(5) + 1
        # 调用 diff 函数，预期结果为数组中每个元素的差值，并在数组前面添加 0
        assert_array_equal(diff(x, prepend=0), np.ones(5))
        # 调用 diff 函数，预期结果同上，但 prepend 参数传入的是一个包含单个元素 0 的列表
        assert_array_equal(diff(x, prepend=[0]), np.ones(5))
        # 使用 np.cumsum 函数计算 x 数组的累积和，预期结果应与原数组 x 相等
        assert_array_equal(np.cumsum(np.diff(x, prepend=0)), x)
        # 调用 diff 函数，预期结果为数组中每个元素的差值，并在数组前面添加两个元素 -1 和 0
        assert_array_equal(diff(x, prepend=[-1, 0]), np.ones(6))

        # 创建一个 2x2 的二维数组 x
        x = np.arange(4).reshape(2, 2)
        # 在沿着第二维度计算差分时，在每行前面添加 0
        result = np.diff(x, axis=1, prepend=0)
        expected = [[0, 1], [2, 1]]
        assert_array_equal(result, expected)
        # 在沿着第二维度计算差分时，prepend 参数为二维数组，每行分别添加 0
        result = np.diff(x, axis=1, prepend=[[0], [0]])
        assert_array_equal(result, expected)

        # 在沿着第一维度计算差分时，在每列前面添加 0
        result = np.diff(x, axis=0, prepend=0)
        expected = [[0, 1], [2, 2]]
        assert_array_equal(result, expected)
        # 在沿着第一维度计算差分时，prepend 参数为二维数组，每列分别添加 0
        result = np.diff(x, axis=0, prepend=[[0, 0]])
        assert_array_equal(result, expected)

        # 调用 diff 函数时，如果 prepend 参数的形状与数组 x 不匹配，预期会引发 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), np.diff, x, prepend=np.zeros((3, 3)))

        # 调用 diff 函数时，如果指定的轴超出了数组的维度范围，预期会引发 np.AxisError 异常
        assert_raises(np.AxisError, diff, x, prepend=0, axis=3)

    # 定义一个测试函数，用于测试 diff 函数的 append 参数
    def test_append(self):
        # 创建一个包含数字 0 到 4 的 NumPy 数组
        x = np.arange(5)
        # 调用 diff 函数，预期结果为数组中每个元素的差值，并在数组末尾添加 0
        result = diff(x, append=0)
        expected = [1, 1, 1, 1, -4]
        assert_array_equal(result, expected)
        # 调用 diff 函数，预期结果同上，但 append 参数传入的是一个包含单个元素 0 的列表
        result = diff(x, append=[0])
        assert_array_equal(result, expected)
        # 调用 diff 函数，预期结果为数组中每个元素的差值，并在数组末尾添加两个元素 0 和 2
        result = diff(x, append=[0, 2])
        expected = expected + [2]
        assert_array_equal(result, expected)

        # 创建一个 2x2 的二维数组 x
        x = np.arange(4).reshape(2, 2)
        # 在沿着第二维度计算差分时，在每行末尾添加 0
        result = np.diff(x, axis=1, append=0)
        expected = [[1, -1], [1, -3]]
        assert_array_equal(result, expected)
        # 在沿着第二维度计算差分时，append 参数为二维数组，每行分别添加 0
        result = np.diff(x, axis=1, append=[[0], [0]])
        assert_array_equal(result, expected)

        # 在沿着第一维度计算差分时，在每列末尾添加 0
        result = np.diff(x, axis=0, append=0)
        expected = [[2, 2], [-2, -3]]
        assert_array_equal(result, expected)
        # 在沿着第一维度计算差分时，append 参数为二维数组，每列分别添加 0
        result = np.diff(x, axis=0, append=[[0, 0]])
        assert_array_equal(result, expected)

        # 调用 diff 函数时，如果 append 参数的形状与数组 x 不匹配，预期会引发 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), np.diff, x, append=np.zeros((3, 3)))

        # 调用 diff 函数时，如果指定的轴超出了数组的维度范围，预期会引发 np.AxisError 异常
        assert_raises(np.AxisError, diff, x, append=0, axis=3)
# 标记为 torch dynamo 测试用例，待实现
@xpassIfTorchDynamo  # (reason="TODO: implement")
# 实例化参数化测试
@instantiate_parametrized_tests
class TestDelete(TestCase):
    # 设置测试用例的初始化方法
    def setUp(self):
        # 创建一个包含0到4的数组
        self.a = np.arange(5)
        # 创建一个包含0到4的数组，重复每个元素两次，然后将其形状变为1x5x2
        self.nd_a = np.arange(5).repeat(2).reshape(1, 5, 2)

    # 检查切片的逆操作
    def _check_inverse_of_slicing(self, indices):
        # 删除数组中的指定索引
        a_del = delete(self.a, indices)
        # 沿指定轴删除数组中的指定索引
        nd_a_del = delete(self.nd_a, indices, axis=1)
        # 断言删除操作是否正确
        msg = f"Delete failed for obj: {indices!r}"
        assert_array_equal(setxor1d(a_del, self.a[indices,]), self.a, err_msg=msg)
        xor = setxor1d(nd_a_del[0, :, 0], self.nd_a[0, indices, 0])
        assert_array_equal(xor, self.nd_a[0, :, 0], err_msg=msg)

    # 测试切片操作
    def test_slices(self):
        # 定义切片的起始、结束和步长
        lims = [-6, -2, 0, 1, 2, 4, 5]
        steps = [-3, -1, 1, 3]
        for start in lims:
            for stop in lims:
                for step in steps:
                    s = slice(start, stop, step)
                    self._check_inverse_of_slicing(s)

    # 测试高级索引操作
    def test_fancy(self):
        self._check_inverse_of_slicing(np.array([[0, 1], [2, 1]]))
        # 断言索引超出范围会引发 IndexError
        with pytest.raises(IndexError):
            delete(self.a, [100])
        with pytest.raises(IndexError):
            delete(self.a, [-100])

        self._check_inverse_of_slicing([0, -1, 2, 2])

        self._check_inverse_of_slicing([True, False, False, True, False])

        # 非法操作，使用这些索引会改变维度
        with pytest.raises(ValueError):
            delete(self.a, True)
        with pytest.raises(ValueError):
            delete(self.a, False)

        # 索引数量不足
        with pytest.raises(ValueError):
            delete(self.a, [False] * 4)

    # 测试单个索引操作
    def test_single(self):
        self._check_inverse_of_slicing(0)
        self._check_inverse_of_slicing(-4)

    # 测试0维数组操作
    def test_0d(self):
        a = np.array(1)
        # 断言删除操作会引发 AxisError
        with pytest.raises(np.AxisError):
            delete(a, [], axis=0)
        with pytest.raises(TypeError):
            delete(a, [], axis="nonsense")

    # 测试数组顺序保留
    def test_array_order_preserve(self):
        # 查看 gh-7113
        k = np.arange(10).reshape(2, 5, order="F")
        m = delete(k, slice(60, None), axis=1)

        # 'k' 是 Fortran 排序的，'m' 应该具有与 'k' 相同的排序，而不是变为 C 排序
        assert_equal(m.flags.c_contiguous, k.flags.c_contiguous)
        assert_equal(m.flags.f_contiguous, k.flags.f_contiguous)

    # 测试索引为浮点数
    def test_index_floats(self):
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([], dtype=float))

    @parametrize(
        "indexer", [subtest(np.array([1]), name="array([1])"), subtest([1], name="[1]")]
    )
    # 定义一个测试方法，用于测试删除单个元素的数组操作
    def test_single_item_array(self, indexer):
        # 删除数组 `self.a` 中索引为 1 的元素，返回新数组 `a_del_int`
        a_del_int = delete(self.a, 1)
        # 删除数组 `self.a` 中由参数 `indexer` 指定的元素，返回新数组 `a_del`
        a_del = delete(self.a, indexer)
        # 断言 `a_del_int` 和 `a_del` 应该相等
        assert_equal(a_del_int, a_del)

        # 删除多维数组 `self.nd_a` 中第二列（axis=1）的元素，返回新数组 `nd_a_del_int`
        nd_a_del_int = delete(self.nd_a, 1, axis=1)
        # 删除多维数组 `self.nd_a` 中由参数 `np.array([1])` 指定的列（axis=1）的元素，返回新数组 `nd_a_del`
        nd_a_del = delete(self.nd_a, np.array([1]), axis=1)
        # 断言 `nd_a_del_int` 和 `nd_a_del` 应该相等
        assert_equal(nd_a_del_int, nd_a_del)

    # 定义一个测试方法，用于测试非整数数组的特殊处理情况
    def test_single_item_array_non_int(self):
        # 当数组为整数时，特殊处理确保不影响非整数数组
        # 如果 `False` 被转换为 `0`，那么它会删除元素：
        res = delete(np.ones(1), np.array([False]))
        # 断言删除操作后，结果 `res` 应该仍然是全为 1 的数组
        assert_array_equal(res, np.ones(1))

        # 测试更复杂的情况（带有轴参数），来自于 issue gh-21840
        x = np.ones((3, 1))
        false_mask = np.array([False], dtype=bool)
        true_mask = np.array([True], dtype=bool)

        # 删除二维数组 `x` 的最后一列（axis=-1），结果应该等于原数组 `x`
        res = delete(x, false_mask, axis=-1)
        assert_array_equal(res, x)
        # 删除二维数组 `x` 的第一列（axis=-1），预期结果应该是空的二维数组
        res = delete(x, true_mask, axis=-1)
        assert_array_equal(res, x[:, :0])
@instantiate_parametrized_tests
class TestGradient(TestCase):
    # 定义测试类 TestGradient，继承自 TestCase，用于测试 gradient 函数

    def test_basic(self):
        # 测试基本情况
        v = [[1, 1], [3, 4]]  # 定义一个二维列表 v
        x = np.array(v)  # 将 v 转换为 NumPy 数组 x
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]  # 预期的梯度值 dx
        assert_array_equal(gradient(x), dx)  # 断言调用 gradient 函数得到的梯度值与预期的 dx 相等
        assert_array_equal(gradient(v), dx)  # 断言调用 gradient 函数得到的梯度值与预期的 dx 相等

    def test_args(self):
        # 测试不同参数组合
        dx = np.cumsum(np.ones(5))  # 生成一个累积和的 NumPy 数组 dx
        dx_uneven = [1.0, 2.0, 5.0, 9.0, 11.0]  # 不均匀间隔的梯度步长列表
        f_2d = np.arange(25).reshape(5, 5)  # 创建一个 5x5 的二维数组 f_2d

        # 测试不同参数传递给 gradient 函数的情况
        gradient(np.arange(5), 3.0)  # 使用 scalar 作为参数调用 gradient 函数
        gradient(np.arange(5), np.array(3.0))  # 使用 NumPy 数组作为参数调用 gradient 函数
        gradient(np.arange(5), dx)  # 使用累积和数组 dx 作为参数调用 gradient 函数
        gradient(f_2d, 1.5)  # 在二维数组 f_2d 上使用 scalar 参数调用 gradient 函数
        gradient(f_2d, np.array(1.5))  # 在二维数组 f_2d 上使用 NumPy 数组参数调用 gradient 函数

        gradient(f_2d, dx_uneven, dx_uneven)  # 混合使用不均匀步长列表 dx_uneven 和自身作为参数调用 gradient 函数
        gradient(f_2d, dx, 2)  # 混合使用步长数组 dx 和 scalar 参数调用 gradient 函数

        gradient(f_2d, dx, axis=1)  # 在指定轴 axis=1 上调用 gradient 函数

        # 当二维坐标参数不允许时引发 ValueError
        assert_raises_regex(
            ValueError,
            ".*scalars or 1d",
            gradient,
            f_2d,
            np.stack([dx] * 2, axis=-1),
            1,
        )

    def test_badargs(self):
        # 测试不良参数情况
        f_2d = np.arange(25).reshape(5, 5)  # 创建一个 5x5 的二维数组 f_2d
        x = np.cumsum(np.ones(5))  # 生成一个累积和的 NumPy 数组 x

        # 断言调用 gradient 函数时传递错误大小的参数会引发 ValueError
        assert_raises(ValueError, gradient, f_2d, x, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))

        # 断言调用 gradient 函数时传递错误数量的参数会引发 TypeError
        assert_raises(TypeError, gradient, f_2d, x)
        assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
        assert_raises(TypeError, gradient, f_2d, x, x, x)
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)

    def test_second_order_accurate(self):
        # 测试数值二阶精度
        # 测试相对数值误差小于 3% 的情况，对于这个示例问题，这对应于所有内部和边界点的二阶精度有限差分
        x = np.linspace(0, 1, 10)  # 创建一个从 0 到 1 的 10 个均匀间隔的数组 x
        dx = x[1] - x[0]  # 计算步长 dx
        y = 2 * x**3 + 4 * x**2 + 2 * x  # 计算函数 y
        analytical = 6 * x**2 + 8 * x + 2  # 计算解析梯度
        num_error = np.abs((np.gradient(y, dx, edge_order=2) / analytical) - 1)  # 计算数值误差
        assert_(np.all(num_error < 0.03).item() is True)  # 断言所有数值误差小于 3%

        # 在不均匀间隔上进行测试
        np.random.seed(0)
        x = np.sort(np.random.random(10))  # 创建一个随机排序的不均匀间隔数组 x
        y = 2 * x**3 + 4 * x**2 + 2 * x  # 计算函数 y
        analytical = 6 * x**2 + 8 * x + 2  # 计算解析梯度
        num_error = np.abs((np.gradient(y, x, edge_order=2) / analytical) - 1)  # 计算数值误差
        assert_(np.all(num_error < 0.03).item() is True)  # 断言所有数值误差小于 3%
    # 定义测试函数 `test_spacing`，用于测试梯度计算函数在不同间距情况下的表现
    def test_spacing(self):
        # 创建包含浮点数的 NumPy 数组 `f`
        f = np.array([0, 2.0, 3.0, 4.0, 5.0, 5.0])
        # 使用 `np.tile` 对数组 `f` 进行复制，同时调整形状
        f = np.tile(f, (6, 1)) + f.reshape(-1, 1)
        # 创建两种不同的 NumPy 数组 `x_uneven` 和 `x_even`
        x_uneven = np.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])
        x_even = np.arange(6.0)

        # 定义各种情况下的梯度期望值数组 `fdx_even_ord1`, `fdx_even_ord2`, `fdx_uneven_ord1`, `fdx_uneven_ord2`
        fdx_even_ord1 = np.tile([2.0, 1.5, 1.0, 1.0, 0.5, 0.0], (6, 1))
        fdx_even_ord2 = np.tile([2.5, 1.5, 1.0, 1.0, 0.5, -0.5], (6, 1))
        fdx_uneven_ord1 = np.tile([4.0, 3.0, 1.7, 0.5, 0.25, 0.0], (6, 1))
        fdx_uneven_ord2 = np.tile([5.0, 3.0, 1.7, 0.5, 0.25, -0.25], (6, 1))

        # 对等间距情况进行循环测试
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            # 测试在二维数组 `f` 上使用 `gradient` 函数计算梯度
            res1 = gradient(f, 1.0, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_even, x_even, axis=(0, 1), edge_order=edge_order)
            res3 = gradient(f, x_even, x_even, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            # 测试在一维数组 `f` 的各个轴上使用 `gradient` 函数计算梯度
            res1 = gradient(f, 1.0, axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_almost_equal(res2, exp_res.T)

            res1 = gradient(f, 1.0, axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_array_equal(res2, exp_res)

        # 对不等间距情况进行循环测试
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            # 测试在二维数组 `f` 上使用 `gradient` 函数计算梯度
            res1 = gradient(f, x_uneven, x_uneven, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_uneven, x_uneven, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            # 测试在一维数组 `f` 的各个轴上使用 `gradient` 函数计算梯度
            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_almost_equal(res1, exp_res.T)

            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_almost_equal(res1, exp_res)

        # 对混合间距情况进行测试
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord1.T)
        assert_almost_equal(res1[1], fdx_uneven_ord1)

        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord2.T)
        assert_almost_equal(res1[1], fdx_uneven_ord2)
    # 定义测试函数，用于测试梯度在特定轴上的计算功能
    def test_specific_axes(self):
        # Testing that gradient can work on a given axis only
        
        # 定义一个二维数组作为输入数据
        v = [[1, 1], [3, 4]]
        # 将列表转换为NumPy数组
        x = np.array(v)
        # 预期在轴0上的梯度值
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(gradient(x, axis=0), dx[0])
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(gradient(x, axis=1), dx[1])
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(gradient(x, axis=-1), dx[1])
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])

        # 测试axis=None，即所有轴都计算梯度
        assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
        # 测试省略axis关键字，与axis=None效果相同
        assert_almost_equal(gradient(x, axis=None), gradient(x))

        # 测试变长参数的顺序
        assert_array_equal(gradient(x, 2, 3, axis=(1, 0)), [dx[1] / 2.0, dx[0] / 3.0])
        # 测试超过最大变长参数数目的情况
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)

        # 断言在给定错误的轴参数时会抛出异常
        assert_raises(np.AxisError, gradient, x, axis=3)
        assert_raises(np.AxisError, gradient, x, axis=-3)
        # 断言在给定不合法的轴参数时会抛出异常
        # assert_raises(TypeError, gradient, x, axis=[1,])

    # 测试函数，用于检查在不精确数据类型下的梯度计算
    def test_inexact_dtypes(self):
        # 循环遍历不同的浮点数数据类型
        for dt in [np.float16, np.float32, np.float64]:
            # 创建指定数据类型的NumPy数组
            x = np.array([1, 2, 3], dtype=dt)
            # 断言梯度计算的数据类型与np.diff函数计算的数据类型相同
            assert_equal(gradient(x).dtype, np.diff(x).dtype)

    # 测试函数，用于检查梯度计算中的特定值
    def test_values(self):
        # 对于edge_order == 1，至少需要2个点
        gradient(np.arange(2), edge_order=1)
        # 对于edge_order == 2，至少需要3个点
        gradient(np.arange(3), edge_order=2)

        # 断言在边界条件下会抛出ValueError异常
        assert_raises(ValueError, gradient, np.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(2), edge_order=2)

    # 参数化测试函数，针对无符号整数进行梯度计算
    @parametrize(
        "f_dtype",
        [
            np.uint8,
        ],
    )
    def test_f_decreasing_unsigned_int(self, f_dtype):
        # 创建指定数据类型的NumPy数组
        f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
        # 计算数组的梯度
        g = gradient(f)
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(g, [-1] * len(f))

    # 参数化测试函数，针对带大跳变的有符号整数进行梯度计算
    @parametrize("f_dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_f_signed_int_big_jump(self, f_dtype):
        # 获取指定数据类型的最大整数值
        maxint = np.iinfo(f_dtype).max
        x = np.array([1, 3])
        # 创建指定数据类型和数值的NumPy数组
        f = np.array([-1, maxint], dtype=f_dtype)
        # 计算数组f在数组x处的梯度
        dfdx = gradient(f, x)
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(dfdx, [(maxint + 1) // 2] * 2)

    # 参数化测试函数，针对无符号整数进行梯度计算
    @parametrize(
        "x_dtype",
        [
            np.uint8,
        ],
    )
    def test_x_decreasing_unsigned(self, x_dtype):
        # 创建指定数据类型的NumPy数组
        x = np.array([3, 2, 1], dtype=x_dtype)
        f = np.array([0, 2, 4])
        # 计算数组f在数组x处的梯度
        dfdx = gradient(f, x)
        # 断言计算得到的梯度与预期的梯度相等
        assert_array_equal(dfdx, [-2] * len(x))

    # 参数化测试函数，针对带有符号整数进行梯度计算
    @parametrize("x_dtype", [np.int8, np.int16, np.int32, np.int64])
    # 定义一个测试函数，用于测试梯度计算在大幅跳跃时的情况，测试对象为指定的数据类型 x_dtype
    def test_x_signed_int_big_jump(self, x_dtype):
        # 获取 x_dtype 类型数据的最小值
        minint = np.iinfo(x_dtype).min
        # 获取 x_dtype 类型数据的最大值
        maxint = np.iinfo(x_dtype).max
        # 创建一个包含两个元素的 NumPy 数组 x，其中包括 -1 和 maxint 两个值，数据类型为 x_dtype
        x = np.array([-1, maxint], dtype=x_dtype)
        # 创建一个包含两个元素的 NumPy 数组 f，其中包括 minint 除以 2 和 0 两个值
        f = np.array([minint // 2, 0])
        # 使用 gradient 函数计算函数 f 在点集 x 处的梯度 dfdx
        dfdx = gradient(f, x)
        # 断言计算得到的梯度 dfdx 应该与预期的数组 [0.5, 0.5] 相等
        assert_array_equal(dfdx, [0.5, 0.5])
class TestAngle(TestCase):
    # 测试角度函数的单元测试类

    def test_basic(self):
        # 测试基本情况
        x = [
            1 + 3j,
            np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2,
            1,
            1j,
            -1,
            -1j,
            1 - 3j,
            -1 + 3j,
        ]
        # 定义测试用例 x，包括复数和实数
        y = angle(x)
        # 计算角度函数的结果
        yo = [
            np.arctan(3.0 / 1.0),
            np.arctan(1.0),
            0,
            np.pi / 2,
            np.pi,
            -np.pi / 2.0,
            -np.arctan(3.0 / 1.0),
            np.pi - np.arctan(3.0 / 1.0),
        ]
        # 预期的角度值列表
        z = angle(x, deg=True)
        # 使用角度函数计算角度值，返回角度的度数表示
        zo = np.array(yo) * 180 / np.pi
        # 将弧度转换为度数的预期结果
        assert_array_almost_equal(y, yo, 11)
        # 断言角度函数的计算结果与预期值相近
        assert_array_almost_equal(z, zo, 11)
        # 断言带有角度单位转换的计算结果与预期值相近


@xpassIfTorchDynamo
@instantiate_parametrized_tests
class TestTrimZeros(TestCase):
    # 测试去除零元素函数的单元测试类

    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    # d = a.astype(object)

    def values(self):
        # 返回测试用例数据的生成器
        attr_names = (
            "a",
            "b",
            "c",
        )  # "d")
        return (getattr(self, name) for name in attr_names)

    def test_basic(self):
        # 测试基本情况
        slc = np.s_[2:-1]
        # 切片定义，选择从第三个元素到倒数第二个元素
        for arr in self.values():
            # 对于每个属性值生成的测试用例数组
            res = trim_zeros(arr)
            # 调用去除零元素函数处理数组
            assert_array_equal(res, arr[slc])
            # 断言去除零元素后的结果与预期切片结果相同

    def test_leading_skip(self):
        # 测试跳过前导零元素的情况
        slc = np.s_[:-1]
        # 切片定义，选择除了最后一个元素的所有元素
        for arr in self.values():
            # 对于每个属性值生成的测试用例数组
            res = trim_zeros(arr, trim="b")
            # 调用去除零元素函数，跳过前导零元素
            assert_array_equal(res, arr[slc])
            # 断言去除零元素后的结果与预期切片结果相同

    def test_trailing_skip(self):
        # 测试跳过末尾零元素的情况
        slc = np.s_[2:]
        # 切片定义，选择从第三个元素开始的所有元素
        for arr in self.values():
            # 对于每个属性值生成的测试用例数组
            res = trim_zeros(arr, trim="F")
            # 调用去除零元素函数，跳过末尾零元素
            assert_array_equal(res, arr[slc])
            # 断言去除零元素后的结果与预期切片结果相同

    def test_all_zero(self):
        # 测试所有元素为零的情况
        for _arr in self.values():
            # 对于每个属性值生成的测试用例数组
            arr = np.zeros_like(_arr, dtype=_arr.dtype)
            # 创建与原数组类型相同的全零数组

            res1 = trim_zeros(arr, trim="B")
            # 调用去除零元素函数，从前开始处理
            assert len(res1) == 0
            # 断言结果数组长度为零

            res2 = trim_zeros(arr, trim="f")
            # 调用去除零元素函数，从后开始处理
            assert len(res2) == 0
            # 断言结果数组长度为零

    def test_size_zero(self):
        # 测试空数组的情况
        arr = np.zeros(0)
        # 创建一个空数组
        res = trim_zeros(arr)
        # 调用去除零元素函数
        assert_array_equal(arr, res)
        # 断言去除零元素后的结果与预期结果相同

    @parametrize(
        "arr",
        [
            np.array([0, 2**62, 0]),
            # np.array([0, 2**63, 0]),  # FIXME
            # np.array([0, 2**64, 0])
        ],
    )
    def test_overflow(self, arr):
        # 测试溢出的情况
        slc = np.s_[1:2]
        # 切片定义，选择第二个元素
        res = trim_zeros(arr)
        # 调用去除零元素函数
        assert_array_equal(res, arr[slc])
        # 断言去除零元素后的结果与预期切片结果相同

    def test_no_trim(self):
        # 测试不去除零元素的情况
        arr = np.array([None, 1, None])
        # 创建包含 None 的数组
        res = trim_zeros(arr)
        # 调用去除零元素函数
        assert_array_equal(arr, res)
        # 断言去除零元素后的结果与原数组相同

    def test_list_to_list(self):
        # 测试从列表到列表的情况
        res = trim_zeros(self.a.tolist())
        # 调用去除零元素函数处理列表
        assert isinstance(res, list)
        # 断言结果是一个列表
    # 定义测试函数 test_place，用于测试 numpy 的 place 函数的功能
    def test_place(self):
        # 确保非 np.ndarray 对象引发 TypeError 错误，而不是不执行任何操作
        assert_raises(TypeError, place, [1, 2, 3], [True, False], [0, 1])

        # 创建一个 numpy 数组 a
        a = np.array([1, 4, 3, 2, 5, 8, 7])
        # 使用 place 函数将指定位置替换为指定值
        place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        # 确保数组 a 的值符合预期
        assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])

        # 使用 place 函数将数组 a 中满足条件的位置替换为指定值
        place(a, np.zeros(7), [])
        # 确保数组 a 的值符合预期，即为从 1 到 7 的连续整数
        assert_array_equal(a, np.arange(1, 8))

        # 再次使用 place 函数，这次替换的值是一个数组
        place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        # 确保数组 a 的值符合预期
        assert_array_equal(a, [8, 2, 9, 4, 8, 6, 9])

        # 使用 lambda 函数测试当插入数组为空时是否会引发 ValueError
        assert_raises_regex(
            ValueError,
            "Cannot insert from an empty array",
            lambda: place(a, [0, 0, 0, 0, 0, 1, 0], []),
        )

        # 检查问题 #6974
        a = np.array(["12", "34"])
        # 使用 place 函数将指定位置替换为指定值
        place(a, [0, 1], "9")
        # 确保数组 a 的值符合预期
        assert_array_equal(a, ["12", "9"])

    # 定义测试函数 test_both，用于测试同时使用 numpy 的 extract 和 place 函数的功能
    def test_both(self):
        # 创建一个随机数组 a
        a = rand(10)
        # 创建一个布尔数组 mask，用于标记数组 a 中大于 0.5 的元素位置
        mask = a > 0.5
        # 创建数组 a 的副本 ac
        ac = a.copy()
        # 使用 extract 函数根据 mask 从数组 a 中提取元素形成新数组 c
        c = extract(mask, a)
        # 使用 place 函数将 mask 标记的位置替换为 0
        place(a, mask, 0)
        # 使用 place 函数将 mask 标记的位置替换为数组 c 中的值
        place(a, mask, c)
        # 确保数组 a 的值符合预期
        assert_array_equal(a, ac)
# _foo1 and _foo2 are utility functions used in some tests in TestVectorize.

def _foo1(x, y=1.0):
    # Returns y multiplied by the floor of x using math.floor
    return y * math.floor(x)

def _foo2(x, y=1.0, z=0.0):
    # Returns y multiplied by the floor of x plus z using math.floor
    return y * math.floor(x) + z

@skip  # (reason="vectorize not implemented")
class TestVectorize(TestCase):
    def test_simple(self):
        # Defines a nested function addsubtract(a, b) that returns a - b if a > b, otherwise a + b
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        # Vectorizes the addsubtract function
        f = vectorize(addsubtract)
        # Applies the vectorized function f to arrays [0, 3, 6, 9] and [1, 3, 5, 7]
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        # Asserts that r is equal to [1, 6, 1, 2]
        assert_array_equal(r, [1, 6, 1, 2])

    def test_scalar(self):
        # Defines a nested function addsubtract(a, b) that returns a - b if a > b, otherwise a + b
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        # Vectorizes the addsubtract function
        f = vectorize(addsubtract)
        # Applies the vectorized function f to arrays [0, 3, 6, 9] and scalar 5
        r = f([0, 3, 6, 9], 5)
        # Asserts that r is equal to [5, 8, 1, 4]
        assert_array_equal(r, [5, 8, 1, 4])

    def test_large(self):
        # Generates a numpy array x with 10000 elements evenly spaced from -3 to 2
        x = np.linspace(-3, 2, 10000)
        # Vectorizes the lambda function lambda x: x which returns x
        f = vectorize(lambda x: x)
        # Applies the vectorized function f to array x
        y = f(x)
        # Asserts that y is equal to x
        assert_array_equal(y, x)

    def test_ufunc(self):
        # Vectorizes the math.cos function
        f = vectorize(math.cos)
        # Creates a numpy array args with values [0, 0.5 * pi, pi, 1.5 * pi, 2 * pi]
        args = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        # Applies the vectorized function f to array args
        r1 = f(args)
        # Computes the numpy cosine of args
        r2 = np.cos(args)
        # Asserts that r1 is almost equal to r2
        assert_array_almost_equal(r1, r2)

    def test_keywords(self):
        # Defines a nested function foo(a, b=1) that returns a + b
        def foo(a, b=1):
            return a + b

        # Vectorizes the foo function
        f = vectorize(foo)
        # Creates a numpy array args with values [1, 2, 3]
        args = np.array([1, 2, 3])
        # Applies the vectorized function f to array args
        r1 = f(args)
        # Creates a numpy array r2 with values [2, 3, 4]
        r2 = np.array([2, 3, 4])
        # Asserts that r1 is equal to r2
        assert_array_equal(r1, r2)
        # Applies the vectorized function f to array args with additional argument 2
        r1 = f(args, 2)
        # Creates a numpy array r2 with values [3, 4, 5]
        r2 = np.array([3, 4, 5])
        # Asserts that r1 is equal to r2
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order1(self):
        # Vectorizes the _foo1 function with output type float
        f = vectorize(_foo1, otypes=[float])
        # Applies f to a numpy array np.arange(3.0) with y=1.0
        r1 = f(np.arange(3.0), 1.0)
        # Applies f again to a numpy array np.arange(3.0)
        r2 = f(np.arange(3.0))
        # Asserts that r1 is equal to r2
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order2(self):
        # Vectorizes the _foo1 function with output type float
        f = vectorize(_foo1, otypes=[float])
        # Applies f to a numpy array np.arange(3.0)
        r1 = f(np.arange(3.0))
        # Applies f to a numpy array np.arange(3.0) with y=1.0
        r2 = f(np.arange(3.0), 1.0)
        # Asserts that r1 is equal to r2
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order3(self):
        # Vectorizes the _foo1 function with output type float
        f = vectorize(_foo1, otypes=[float])
        # Applies f to a numpy array np.arange(3.0)
        r1 = f(np.arange(3.0))
        # Applies f to a numpy array np.arange(3.0) with y=1.0
        r2 = f(np.arange(3.0), y=1.0)
        # Applies f again to a numpy array np.arange(3.0)
        r3 = f(np.arange(3.0))
        # Asserts that r1 is equal to r2 and r1 is equal to r3
        assert_array_equal(r1, r2)
        assert_array_equal(r1, r3)
    def test_keywords_with_otypes_several_kwd_args1(self):
        # gh-1620 确保关键字参数的不同使用方式不会破坏向量化函数。
        f = vectorize(_foo2, otypes=[float])
        # 我们测试向量化函数的ufunc缓存，因此这些函数调用的顺序是测试的重要部分。
        r1 = f(10.4, z=100)
        r2 = f(10.4, y=-1)
        r3 = f(10.4)
        assert_equal(r1, _foo2(10.4, z=100))
        assert_equal(r2, _foo2(10.4, y=-1))
        assert_equal(r3, _foo2(10.4))

    def test_keywords_with_otypes_several_kwd_args2(self):
        # gh-1620 确保关键字参数的不同使用方式不会破坏向量化函数。
        f = vectorize(_foo2, otypes=[float])
        # 我们测试向量化函数的ufunc缓存，因此这些函数调用的顺序是测试的重要部分。
        r1 = f(z=100, x=10.4, y=-1)
        r2 = f(1, 2, 3)
        assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
        assert_equal(r2, _foo2(1, 2, 3))

    def test_keywords_no_func_code(self):
        # 这个测试需要一个带有关键字但没有func_code属性的函数，
        # 因为否则vectorize将会检查func_code。
        import random

        try:
            vectorize(random.randrange)  # 应该成功
        except Exception:
            raise AssertionError  # noqa: B904

    def test_keywords2_ticket_2100(self):
        # 测试关键字参数的支持：增强票2100

        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(a=args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(b=1, a=args)
        assert_array_equal(r1, r2)
        r1 = f(args, b=2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords3_ticket_2100(self):
        # 测试混合位置参数和关键字参数的排除：票2100

        def mypolyval(x, p):
            _p = list(p)
            res = _p.pop(0)
            while _p:
                res = res * x + _p.pop(0)
            return res

        vpolyval = np.vectorize(mypolyval, excluded=["p", 1])
        ans = [3, 6]
        assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))

    def test_keywords4_ticket_2100(self):
        # 测试没有位置参数的向量化函数。
        @vectorize
        def f(**kw):
            res = 1.0
            for _k in kw:
                res *= kw[_k]
            return res

        assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])

    def test_keywords5_ticket_2100(self):
        # 测试没有关键字参数的向量化函数。
        @vectorize
        def f(*v):
            return np.prod(v)

        assert_array_equal(f([1, 2], [3, 4]), [3, 8])
    # 测试函数：测试 vectorize 函数处理简单函数的情况
    def test_coverage1_ticket_2100(self):
        # 定义一个简单的函数 foo，返回整数 1
        def foo():
            return 1

        # 使用 vectorize 函数对 foo 进行向量化处理
        f = vectorize(foo)
        # 断言调用 f() 返回结果为数组 [1]
        assert_array_equal(f(), 1)

    # 测试函数：测试 vectorize 函数对带有文档字符串的函数的处理
    def test_assigning_docstring(self):
        # 定义一个带有文档字符串的函数 foo，文档内容为 "Original documentation"
        def foo(x):
            """Original documentation"""
            return x

        # 使用 vectorize 函数对 foo 进行向量化处理
        f = vectorize(foo)
        # 断言向量化后的函数 f 的文档字符串与原始函数 foo 的文档字符串相等
        assert_equal(f.__doc__, foo.__doc__)

        # 定义一个新的文档字符串
        doc = "Provided documentation"
        # 使用带有新文档字符串的参数再次对 foo 进行向量化处理
        f = vectorize(foo, doc=doc)
        # 断言向量化后的函数 f 的文档字符串与新文档字符串相等
        assert_equal(f.__doc__, doc)

    # 测试函数：测试 vectorize 函数对类的未绑定方法的处理
    def test_UnboundMethod_ticket_1156(self):
        # 定义一个简单的类 Foo，包含类变量 b 和方法 bar
        class Foo:
            b = 2

            def bar(self, a):
                return a**self.b

        # 断言使用 vectorize 处理 Foo 类的实例方法 bar 后，对 np.arange(9) 的处理结果正确
        assert_array_equal(vectorize(Foo().bar)(np.arange(9)), np.arange(9) ** 2)
        # 断言使用 vectorize 处理 Foo 类的未绑定方法 bar 后，对 np.arange(9) 的处理结果正确
        assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)), np.arange(9) ** 2)

    # 测试函数：测试 vectorize 函数的执行顺序依赖问题
    def test_execution_order_ticket_1487(self):
        # 使用 lambda 函数创建两个独立的 vectorize 函数 f1 和 f2
        f1 = vectorize(lambda x: x)
        res1a = f1(np.arange(3))
        res1b = f1(np.arange(0.1, 3))
        f2 = vectorize(lambda x: x)
        res2b = f2(np.arange(0.1, 3))
        res2a = f2(np.arange(3))
        # 断言两个 vectorize 函数处理相同输入时的结果一致，验证不依赖执行顺序
        assert_equal(res1a, res2a)
        assert_equal(res1b, res2b)

    # 测试函数：测试 vectorize 函数对字符串的向量化处理
    def test_string_ticket_1892(self):
        # 使用 vectorize 函数处理 lambda 函数，验证对字符串的向量化处理
        f = np.vectorize(lambda x: x)
        s = "0123456789" * 10
        # 断言字符串 s 经过向量化处理后结果与原字符串相等
        assert_equal(s, f(s))

    # 测试函数：测试 vectorize 函数的缓存机制
    def test_cache(self):
        # 定义一个带缓存的 vectorize 函数 f，用于计算输入的平方
        _calls = [0]

        @vectorize
        def f(x):
            _calls[0] += 1
            return x**2

        f.cache = True
        x = np.arange(5)
        # 断言缓存启用后，函数 f 对每个参数值只调用一次
        assert_array_equal(f(x), x * x)
        assert_equal(_calls[0], len(x))

    # 测试函数：测试 vectorize 函数的输出类型设置
    def test_otypes(self):
        # 使用 vectorize 函数处理 lambda 函数，指定输出类型为整数
        f = np.vectorize(lambda x: x)
        f.otypes = "i"
        x = np.arange(5)
        # 断言向量化后的函数 f 输出结果类型为整数
        assert_array_equal(f(x), x)

    # 测试函数：测试 vectorize 函数的签名设置，计算数组均值
    def test_signature_mean_last(self):
        # 定义一个计算数组均值的函数 mean
        def mean(a):
            return a.mean()

        # 使用 vectorize 函数处理 mean 函数，指定输入输出的签名形式
        f = vectorize(mean, signature="(n)->()")
        r = f([[1, 3], [2, 4]])
        # 断言使用指定签名后，函数 f 对输入数组的处理结果正确
        assert_array_equal(r, [2, 3])

    # 测试函数：测试 vectorize 函数的签名设置，将数组中心化
    def test_signature_center(self):
        # 定义一个将数组中心化的函数 center
        def center(a):
            return a - a.mean()

        # 使用 vectorize 函数处理 center 函数，指定输入输出的签名形式
        f = vectorize(center, signature="(n)->(n)")
        r = f([[1, 3], [2, 4]])
        # 断言使用指定签名后，函数 f 对输入数组的处理结果正确
        assert_array_equal(r, [[-1, 1], [-1, 1]])

    # 测试函数：测试 vectorize 函数的签名设置，输出两个值
    def test_signature_two_outputs(self):
        # 使用 vectorize 函数处理 lambda 函数，指定输入输出的签名形式
        f = vectorize(lambda x: (x, x), signature="()->(),()")
        r = f([1, 2, 3])
        # 断言使用指定签名后，函数 f 返回的结果为元组，且包含两个相同的数组
        assert_(isinstance(r, tuple) and len(r) == 2)
        assert_array_equal(r[0], [1, 2, 3])
        assert_array_equal(r[1], [1, 2, 3])
    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_outer(self):
        # 使用 vectorize 函数创建一个向量化的 np.outer 函数，签名指定为 "(a),(b)->(a,b)"
        f = vectorize(np.outer, signature="(a),(b)->(a,b)")
        # 调用向量化函数计算结果
        r = f([1, 2], [1, 2, 3])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])

        # 再次调用向量化函数计算结果，用于不同的输入形状
        r = f([[[1, 2]]], [1, 2, 3])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])

        # 再次调用向量化函数计算结果，用于不同的输入形状
        r = f([[1, 0], [2, 0]], [1, 2, 3])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]], [[2, 4, 6], [0, 0, 0]]])

        # 再次调用向量化函数计算结果，用于不同的输入形状
        r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]], [[0, 0, 0], [0, 0, 0]]])

    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_computed_size(self):
        # 使用 lambda 函数创建一个向量化的函数，用于截取输入数组的最后一个元素，签名为 "(n)->(m)"
        f = vectorize(lambda x: x[:-1], signature="(n)->(m)")
        # 调用向量化函数计算结果
        r = f([1, 2, 3])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [1, 2])

        # 再次调用向量化函数计算结果，用于不同的输入形状
        r = f([[1, 2, 3], [2, 3, 4]])
        # 断言计算结果与预期值相等
        assert_array_equal(r, [[1, 2], [2, 3]])

    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_excluded(self):
        # 定义一个普通函数 foo，用于将两个参数相加
        def foo(a, b=1):
            return a + b

        # 使用 vectorize 函数创建一个向量化的 foo 函数，签名为 "()->()"，排除参数 "b"
        f = vectorize(foo, signature="()->()", excluded={"b"})
        # 断言向量化函数计算结果与预期值相等
        assert_array_equal(f([1, 2, 3]), [2, 3, 4])
        assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])

    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_otypes(self):
        # 使用 lambda 函数创建一个向量化的函数，直接返回输入值，签名为 "(n)->(n)"，指定输出类型为 "float64"
        f = vectorize(lambda x: x, signature="(n)->(n)", otypes=["float64"])
        # 调用向量化函数计算结果
        r = f([1, 2, 3])
        # 断言计算结果的数据类型与预期的 float64 类型相等
        assert_equal(r.dtype, np.dtype("float64"))
        # 断言计算结果与预期值相等
        assert_array_equal(r, [1, 2, 3])

    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_invalid_inputs(self):
        # 使用 vectorize 函数创建一个向量化的 operator.add 函数，签名为 "(n),(n)->(n)"
        f = vectorize(operator.add, signature="(n),(n)->(n)")
        # 使用 assert_raises_regex 断言捕获 TypeError 异常，并验证异常消息
        with assert_raises_regex(TypeError, "wrong number of positional"):
            f([1, 2])
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息
        with assert_raises_regex(ValueError, "does not have enough dimensions"):
            f(1, 2)
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息
        with assert_raises_regex(ValueError, "inconsistent size for core dimension"):
            f([1, 2], [1, 2, 3])

        # 使用 vectorize 函数创建一个向量化的 operator.add 函数，签名为 "()->()"
        f = vectorize(operator.add, signature="()->()")
        # 使用 assert_raises_regex 断言捕获 TypeError 异常，并验证异常消息
        with assert_raises_regex(TypeError, "wrong number of positional"):
            f(1, 2)

    # 定义一个测试方法，用于测试带有特定签名的向量化函数
    def test_signature_invalid_outputs(self):
        # 使用 lambda 函数创建一个向量化的函数，用于截取输入数组的最后一个元素，签名为 "(n)->(n)"
        f = vectorize(lambda x: x[:-1], signature="(n)->(n)")
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息
        with assert_raises_regex(ValueError, "inconsistent size for core dimension"):
            f([1, 2, 3])

        # 使用 vectorize 函数创建一个向量化的 lambda 函数，签名为 "()->(),()"
        f = vectorize(lambda x: x, signature="()->(),()")
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息
        with assert_raises_regex(ValueError, "wrong number of outputs"):
            f(1)

        # 使用 vectorize 函数创建一个向量化的 lambda 函数，签名为 "()->()"
        f = vectorize(lambda x: (x, x), signature="()->()")
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息
        with assert_raises_regex(ValueError, "wrong number of outputs"):
            f([1, 2])
    def test_size_zero_output(self):
        # 定义一个测试方法，用于测试大小为零的输出
        # 见问题号 5868

        # 使用 np.vectorize 创建一个函数 f，该函数对输入元素不做任何修改
        f = np.vectorize(lambda x: x)
        
        # 创建一个大小为 [0, 5] 的零矩阵 x，数据类型为整数
        x = np.zeros([0, 5], dtype=int)
        
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息包含 "otypes"
        with assert_raises_regex(ValueError, "otypes"):
            f(x)

        # 设置 f 的 otypes 属性为 "i"
        f.otypes = "i"
        
        # 断言调用 f(x) 的结果与 x 相等
        assert_array_equal(f(x), x)

        # 使用特定签名 "()->()" 创建 np.vectorize 函数 f
        f = np.vectorize(lambda x: x, signature="()->()")
        
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息包含 "otypes"
        with assert_raises_regex(ValueError, "otypes"):
            f(x)

        # 使用特定签名 "()->()" 和 otypes="i" 创建 np.vectorize 函数 f
        f = np.vectorize(lambda x: x, signature="()->()", otypes="i")
        
        # 断言调用 f(x) 的结果与 x 相等
        assert_array_equal(f(x), x)

        # 使用特定签名 "(n)->(n)" 和 otypes="i" 创建 np.vectorize 函数 f
        f = np.vectorize(lambda x: x, signature="(n)->(n)", otypes="i")
        
        # 断言调用 f(x) 的结果与 x 相等
        assert_array_equal(f(x), x)

        # 使用特定签名 "(n)->(n)" 创建 np.vectorize 函数 f
        f = np.vectorize(lambda x: x, signature="(n)->(n)")
        
        # 断言调用 f(x.T) 的结果与 x.T 相等
        assert_array_equal(f(x.T), x.T)

        # 使用特定签名 "()->(n)" 和 otypes="i" 创建 np.vectorize 函数 f
        f = np.vectorize(lambda x: [x], signature="()->(n)", otypes="i")
        
        # 使用 assert_raises_regex 断言捕获 ValueError 异常，并验证异常消息包含 "new output dimensions"
        with assert_raises_regex(ValueError, "new output dimensions"):
            f(x)
# 使用装饰器 @xpassIfTorchDynamo 对 TestDigitize 类进行标记，说明此类是为了测试 digitize 函数而创建的测试集合
@xpassIfTorchDynamo  # (reason="TODO: implement")
class TestDigitize(TestCase):
    # 定义测试方法 test_forward，验证 digitize 函数对正向输入的处理是否正确
    def test_forward(self):
        x = np.arange(-6, 5)
        bins = np.arange(-5, 5)
        assert_array_equal(digitize(x, bins), np.arange(11))

    # 定义测试方法 test_reverse，验证 digitize 函数对反向输入的处理是否正确
    def test_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(5, -5, -1)
        assert_array_equal(digitize(x, bins), np.arange(11))

    # 定义测试方法 test_random，验证 digitize 函数对随机输入的处理是否正确
    def test_random(self):
        x = rand(10)
        bin = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bin) != 0))

    # 定义测试方法 test_right_basic，验证 digitize 函数在指定右侧开放边界时的基本处理是否正确
    def test_right_basic(self):
        x = [1, 5, 4, 10, 8, 11, 0]
        bins = [1, 5, 10]
        default_answer = [1, 2, 1, 3, 2, 3, 0]
        assert_array_equal(digitize(x, bins), default_answer)
        right_answer = [0, 1, 1, 2, 2, 3, 0]
        assert_array_equal(digitize(x, bins, True), right_answer)

    # 定义测试方法 test_right_open，验证 digitize 函数在指定右侧开放边界时的处理是否正确
    def test_right_open(self):
        x = np.arange(-6, 5)
        bins = np.arange(-6, 4)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    # 定义测试方法 test_right_open_reverse，验证 digitize 函数在指定右侧开放边界时对反向输入的处理是否正确
    def test_right_open_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(4, -6, -1)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    # 定义测试方法 test_right_open_random，验证 digitize 函数在指定右侧开放边界时对随机输入的处理是否正确
    def test_right_open_random(self):
        x = rand(10)
        bins = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bins, True) != 10))

    # 定义测试方法 test_monotonic，验证 digitize 函数对单调性输入的处理是否正确
    def test_monotonic(self):
        x = [-1, 0, 1, 2]
        bins = [0, 0, 1]
        assert_array_equal(digitize(x, bins, False), [0, 2, 3, 3])
        assert_array_equal(digitize(x, bins, True), [0, 0, 2, 3])
        bins = [1, 1, 0]
        assert_array_equal(digitize(x, bins, False), [3, 2, 0, 0])
        assert_array_equal(digitize(x, bins, True), [3, 3, 2, 0])
        bins = [1, 1, 1, 1]
        assert_array_equal(digitize(x, bins, False), [0, 0, 4, 4])
        assert_array_equal(digitize(x, bins, True), [0, 0, 0, 4])
        bins = [0, 0, 1, 0]
        assert_raises(ValueError, digitize, x, bins)
        bins = [1, 1, 0, 1]
        assert_raises(ValueError, digitize, x, bins)

    # 定义测试方法 test_casting_error，验证 digitize 函数对类型错误输入的处理是否正确
    def test_casting_error(self):
        x = [1, 2, 3 + 1.0j]
        bins = [1, 2, 3]
        assert_raises(TypeError, digitize, x, bins)
        x, bins = bins, x
        assert_raises(TypeError, digitize, x, bins)

    # 定义测试方法 test_large_integers_increasing，验证 digitize 函数对大整数递增输入的处理是否正确
    def test_large_integers_increasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x - 1, x + 1]), 1)

    # 使用装饰器 @xfail 对 test_large_integers_decreasing 方法进行标记，表明该测试目前预期失败
    @xfail  # "gh-11022: np.core.multiarray._monoticity loses precision"
    def test_large_integers_decreasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x + 1, x - 1]), 1)


# 使用装饰器 @skip 对 TestUnwrap 类进行标记，说明此类测试尚未实现
@skip  # (reason="TODO: implement; here unwrap if from numpy")
class TestUnwrap(TestCase):
    # 定义一个测试方法，用于简单的测试 unwrap 函数
    def test_simple(self):
        # 断言：检查 unwrap 函数能否移除大于 2*pi 的跳跃
        assert_array_equal(unwrap([1, 1 + 2 * np.pi]), [1, 1])
        # 断言：检查 unwrap 函数能否保持连续性
        assert_(np.all(diff(unwrap(rand(10) * 100)) < np.pi))

    # 定义另一个测试方法，用于测试带周期参数的 unwrap 函数
    def test_period(self):
        # 断言：检查带周期参数的 unwrap 函数能否移除大于 255 的跳跃
        assert_array_equal(unwrap([1, 1 + 256], period=255), [1, 2])
        # 断言：检查带周期参数的 unwrap 函数能否保持连续性
        assert_(np.all(diff(unwrap(rand(10) * 1000, period=255)) < 255))
        
        # 检查简单情况
        simple_seq = np.array([0, 75, 150, 225, 300])
        wrap_seq = np.mod(simple_seq, 255)
        assert_array_equal(unwrap(wrap_seq, period=255), simple_seq)
        
        # 检查自定义不连续值
        uneven_seq = np.array([0, 75, 150, 225, 300, 430])
        wrap_uneven = np.mod(uneven_seq, 250)
        # 检查 unwrap 函数对于带自定义不连续值的序列的处理
        no_discont = unwrap(wrap_uneven, period=250)
        assert_array_equal(no_discont, [0, 75, 150, 225, 300, 180])
        # 检查 unwrap 函数对于带小不连续值的序列的处理
        sm_discont = unwrap(wrap_uneven, period=250, discont=140)
        assert_array_equal(sm_discont, [0, 75, 150, 225, 300, 430])
        # 断言：检查返回的结果类型是否与输入序列相同
        assert sm_discont.dtype == wrap_uneven.dtype
# 定义一个测试类 TestFilterwindows，用于测试滤波窗口函数的功能
@instantiate_parametrized_tests
class TestFilterwindows(TestCase):

    # 使用参数化装饰器 parametrize，定义测试函数 test_hanning，测试汉宁窗口函数
    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # 指定参数化测试的数据类型，包括整数和浮点数类型
    @parametrize("M", [0, 1, 10])  # 参数化测试 M 参数，测试值为 0、1、10
    def test_hanning(self, dtype: str, M: int) -> None:
        scalar = M  # 将 M 参数赋给 scalar 变量

        w = hanning(scalar)  # 调用 hanning 函数生成长度为 scalar 的汉宁窗口
        ref_dtype = np.result_type(dtype, np.float64)  # 根据 dtype 确定参考的数据类型为 np.float64
        assert w.dtype == ref_dtype  # 断言生成的窗口 w 的数据类型与 ref_dtype 相符

        # 检查窗口 w 的对称性，使用 assert_allclose 函数，指定容差为 1e-15
        assert_allclose(w, flipud(w), atol=1e-15)

        # 检查已知的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))  # 如果 scalar 小于 1，则 w 应为空数组
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))  # 如果 scalar 等于 1，则 w 应为长度为 1 的全一数组
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.500, 4)  # 否则，对 w 进行近似相等的数值检查

    # 使用参数化装饰器 parametrize，定义测试函数 test_hamming，测试哈明窗口函数
    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # 指定参数化测试的数据类型，包括整数和浮点数类型
    @parametrize("M", [0, 1, 10])  # 参数化测试 M 参数，测试值为 0、1、10
    def test_hamming(self, dtype: str, M: int) -> None:
        scalar = M  # 将 M 参数赋给 scalar 变量

        w = hamming(scalar)  # 调用 hamming 函数生成长度为 scalar 的哈明窗口
        ref_dtype = np.result_type(dtype, np.float64)  # 根据 dtype 确定参考的数据类型为 np.float64
        assert w.dtype == ref_dtype  # 断言生成的窗口 w 的数据类型与 ref_dtype 相符

        # 检查窗口 w 的对称性，使用 assert_allclose 函数，指定容差为 1e-15
        assert_allclose(w, flipud(w), atol=1e-15)

        # 检查已知的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))  # 如果 scalar 小于 1，则 w 应为空数组
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))  # 如果 scalar 等于 1，则 w 应为长度为 1 的全一数组
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.9400, 4)  # 否则，对 w 进行近似相等的数值检查

    # 使用参数化装饰器 parametrize，定义测试函数 test_bartlett，测试巴特利特窗口函数
    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # 指定参数化测试的数据类型，包括整数和浮点数类型
    @parametrize("M", [0, 1, 10])  # 参数化测试 M 参数，测试值为 0、1、10
    def test_bartlett(self, dtype: str, M: int) -> None:
        scalar = M  # 将 M 参数赋给 scalar 变量

        w = bartlett(scalar)  # 调用 bartlett 函数生成长度为 scalar 的巴特利特窗口
        ref_dtype = np.result_type(dtype, np.float64)  # 根据 dtype 确定参考的数据类型为 np.float64
        assert w.dtype == ref_dtype  # 断言生成的窗口 w 的数据类型与 ref_dtype 相符

        # 检查窗口 w 的对称性，使用 assert_allclose 函数，指定容差为 1e-15
        assert_allclose(w, flipud(w), atol=1e-15)

        # 检查已知的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))  # 如果 scalar 小于 1，则 w 应为空数组
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))  # 如果 scalar 等于 1，则 w 应为长度为 1 的全一数组
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.4444, 4)  # 否则，对 w 进行近似相等的数值检查

    # 使用参数化装饰器 parametrize，定义测试函数 test_blackman，测试布莱克曼窗口函数
    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # 指定参数化测试的数据类型，包括整数和浮点数类型
    @parametrize("M", [0, 1, 10])  # 参数化测试 M 参数，测试值为 0、1、10
    def test_blackman(self, dtype: str, M: int) -> None:
        scalar = M  # 将 M 参数赋给 scalar 变量

        w = blackman(scalar)  # 调用 blackman 函数生成长度为 scalar 的布莱克曼窗口
        ref_dtype = np.result_type(dtype, np.float64)  # 根据 dtype 确定参考的数据类型为 np.float64
        assert w.dtype == ref_dtype  # 断言生成的窗口 w 的数据类型与 ref_dtype 相符

        # 检查窗口 w 的对称性，使用 assert_allclose 函数，指定容差为 1e-15
        assert_allclose(w, flipud(w), atol=1e-15)

        # 检查已知的值
        if scalar < 1:
            assert_array_equal(w, np.array([]))  # 如果 scalar 小于 1，则 w 应为空数组
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))  # 如果 scalar 等于 1，则 w 应为长度为 1 的全一数组
        else:
            assert_almost_equal(np.sum(w, axis=0), 3.7800, 4)  # 否则，对 w 进行近似相等的数值检查
    # 定义一个测试方法，用于测试 Kaiser 窗函数的行为
    def test_kaiser(self, dtype: str, M: int) -> None:
        # 将输入的 M 参数赋值给 scalar 变量
        scalar = M

        # 调用 kaiser 函数，生成 Kaiser 窗口 w，窗口长度为 scalar，β参数为 0
        w = kaiser(scalar, 0)

        # 根据输入的 dtype 和 np.float64 计算出参考的数据类型
        ref_dtype = np.result_type(dtype, np.float64)

        # 断言生成的 Kaiser 窗口 w 的数据类型与参考数据类型 ref_dtype 相同
        assert w.dtype == ref_dtype

        # 检查 Kaiser 窗口 w 是否对称
        assert_equal(w, flipud(w))

        # 检查已知条件下 Kaiser 窗口 w 的值
        if scalar < 1:
            # 如果 scalar 小于 1，则期望 w 是一个空数组
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            # 如果 scalar 等于 1，则期望 w 是一个包含一个元素的数组，元素值为 1
            assert_array_equal(w, np.ones(1))
        else:
            # 对于其他情况，验证 Kaiser 窗口 w 的加和结果接近 10，精度到小数点后 15 位
            assert_almost_equal(np.sum(w, axis=0), 10, 15)
# 使用装饰器标记为测试跳过，理由是尚未实现
@xpassIfTorchDynamo  # (reason="TODO: implement")
class TestTrapz(TestCase):
    def test_simple(self):
        # 创建一个从-10到10，步长为0.1的数组
        x = np.arange(-10, 10, 0.1)
        # 计算正态分布函数的积分，dx为步长0.1
        r = trapz(np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), dx=0.1)
        # 检查正态分布的积分是否等于1
        assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        # 创建一维数组x、y、z，分别包含0到1、0到2、0到3的值
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        # 计算每个维度上的权重数组，首尾元素权重除以2
        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        # 创建三维数组q，包含x、y、z的组合
        q = x[:, None, None] + y[None, :, None] + z[None, None, :]

        # 沿各轴求加权和
        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # 对多维数组进行积分，指定积分的轴
        r = trapz(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)

        # 对一维数组进行积分，默认轴为0
        r = trapz(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z, axis=2)
        assert_almost_equal(r, qz)


class TestSinc(TestCase):
    def test_simple(self):
        # 断言sinc(0)等于1
        assert_(sinc(0) == 1)
        # 计算在-1到1之间的100个点上的sinc函数，检查其对称性
        w = sinc(np.linspace(-1, 1, 100))
        assert_array_almost_equal(w, np.flipud(w), 7)

    def test_array_like(self):
        # 使用不同的输入类型x，测试sinc函数的输出是否相同
        x = [0, 0.5]
        y1 = sinc(np.array(x))
        y2 = sinc(list(x))
        y3 = sinc(tuple(x))
        assert_array_equal(y1, y2)
        assert_array_equal(y1, y3)


class TestUnique(TestCase):
    def test_simple(self):
        # 创建包含重复元素的数组x，断言unique函数去重后的结果是否正确
        x = np.array([4, 3, 2, 1, 1, 2, 3, 4, 0])
        assert_(np.all(unique(x) == [0, 1, 2, 3, 4]))

        # 断言对包含相同元素的数组的处理结果是否正确
        assert_(unique(np.array([1, 1, 1, 1, 1])) == np.array([1]))

    @xpassIfTorchDynamo  # (reason="unique not implemented for 'ComplexDouble'")
    def test_simple_complex(self):
        # 创建包含复数的数组x，断言unique函数去重后的结果是否正确
        x = np.array([5 + 6j, 1 + 1j, 1 + 10j, 10, 5 + 6j])
        assert_(np.all(unique(x) == [1 + 1j, 1 + 10j, 5 + 6j, 10]))


@xpassIfTorchDynamo  # (reason="TODO: implement")
class TestCheckFinite(TestCase):
    def test_simple(self):
        # 创建包含无穷大和NaN值的数组，测试np.lib.asarray_chkfinite函数的行为
        a = [1, 2, 3]
        b = [1, 2, np.inf]
        c = [1, 2, np.nan]
        np.lib.asarray_chkfinite(a)
        assert_raises(ValueError, np.lib.asarray_chkfinite, b)
        assert_raises(ValueError, np.lib.asarray_chkfinite, c)

    def test_dtype_order(self):
        # 检查np.lib.asarray_chkfinite函数的dtype和order参数
        # 对包含整数的数组a进行检查，并确保其类型为np.float64
        a = [1, 2, 3]
        a = np.lib.asarray_chkfinite(a, order="F", dtype=np.float64)
        assert_(a.dtype == np.float64)


@instantiate_parametrized_tests
class TestCorrCoef(TestCase):
    A = np.array(
        [
            [0.15391142, 0.18045767, 0.14197213],  # 创建一个大小为3x3的NumPy数组A，包含指定的浮点数值
            [0.70461506, 0.96474128, 0.27906989],
            [0.9297531, 0.32296769, 0.19267156],
        ]
    )
    B = np.array(
        [
            [0.10377691, 0.5417086, 0.49807457],  # 创建一个大小为3x3的NumPy数组B，包含指定的浮点数值
            [0.82872117, 0.77801674, 0.39226705],
            [0.9314666, 0.66800209, 0.03538394],
        ]
    )
    res1 = np.array(
        [
            [1.0, 0.9379533, -0.04931983],  # 创建一个大小为3x3的NumPy数组res1，包含指定的浮点数值
            [0.9379533, 1.0, 0.30007991],
            [-0.04931983, 0.30007991, 1.0],
        ]
    )
    res2 = np.array(
        [
            [1.0, 0.9379533, -0.04931983, 0.30151751, 0.66318558, 0.51532523],  # 创建一个大小为6x6的NumPy数组res2，包含指定的浮点数值
            [0.9379533, 1.0, 0.30007991, -0.04781421, 0.88157256, 0.78052386],
            [-0.04931983, 0.30007991, 1.0, -0.96717111, 0.71483595, 0.83053601],
            [0.30151751, -0.04781421, -0.96717111, 1.0, -0.51366032, -0.66173113],
            [0.66318558, 0.88157256, 0.71483595, -0.51366032, 1.0, 0.98317823],
            [0.51532523, 0.78052386, 0.83053601, -0.66173113, 0.98317823, 1.0],
        ]
    )

    def test_non_array(self):
        assert_almost_equal(
            np.corrcoef([0, 1, 0], [1, 0, 1]), [[1.0, -1.0], [-1.0, 1.0]]  # 测试非数组输入情况下的相关系数计算，断言其结果与预期值相近
        )

    def test_simple(self):
        tgt1 = corrcoef(self.A)  # 计算数组A的相关系数矩阵，并将结果保存在tgt1中
        assert_almost_equal(tgt1, self.res1)  # 断言计算得到的相关系数矩阵与预期结果res1相近
        assert_(np.all(np.abs(tgt1) <= 1.0))  # 断言相关系数矩阵中所有元素的绝对值均不超过1

        tgt2 = corrcoef(self.A, self.B)  # 计算数组A和B的相关系数矩阵，并将结果保存在tgt2中
        assert_almost_equal(tgt2, self.res2)  # 断言计算得到的相关系数矩阵与预期结果res2相近
        assert_(np.all(np.abs(tgt2) <= 1.0))  # 断言相关系数矩阵中所有元素的绝对值均不超过1

    @skip(reason="deprecated in numpy, ignore")
    def test_ddof(self):
        # ddof raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, ddof=-1)  # 断言在使用ddof=-1时，调用corrcoef函数会触发DeprecationWarning
            sup.filter(DeprecationWarning)
            # ddof has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)  # 断言在使用ddof=-1时，计算得到的相关系数矩阵与预期结果res1相近
            assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)  # 断言在使用ddof=-1时，计算得到的相关系数矩阵与预期结果res2相近
            assert_almost_equal(corrcoef(self.A, ddof=3), self.res1)  # 断言在使用ddof=3时，计算得到的相关系数矩阵与预期结果res1相近
            assert_almost_equal(corrcoef(self.A, self.B, ddof=3), self.res2)  # 断言在使用ddof=3时，计算得到的相关系数矩阵与预期结果res2相近

    @skip(reason="deprecated in numpy, ignore")
    def test_bias(self):
        # bias raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)  # 断言在调用时，使用bias参数会触发DeprecationWarning
            assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)  # 断言在调用时，使用bias参数会触发DeprecationWarning
            sup.filter(DeprecationWarning)
            # bias has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, bias=1), self.res1)  # 断言在使用bias=1时，计算得到的相关系数矩阵与预期结果res1相近

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)  # 计算数组x的相关系数矩阵，并将结果保存在res中
        tgt = np.array([[1.0, -1.0j], [1.0j, 1.0]])  # 创建一个大小为2x2的NumPy数组tgt，包含指定的复数值
        assert_allclose(res, tgt)  # 断言计算得到的相关系数矩阵与预期结果tgt非常接近
        assert_(np.all(np.abs(res) <= 1.0))  # 断言相关系数矩阵中所有元素的绝对值均不超过1
    # 定义测试方法 test_xy，用于测试 np.corrcoef 函数的行为
    def test_xy(self):
        # 创建包含复数的 numpy 数组 x
        x = np.array([[1, 2, 3]])
        # 创建包含复数的 numpy 数组 y
        y = np.array([[1j, 2j, 3j]])
        # 断言 np.corrcoef 函数对 x 和 y 的计算结果与预期值相近
        assert_allclose(np.corrcoef(x, y), np.array([[1.0, -1.0j], [1.0j, 1.0]]))

    # 定义测试方法 test_empty，用于测试处理空数组时的 np.corrcoef 函数的行为
    def test_empty(self):
        # 捕获运行时警告
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            # 断言对空数组调用 np.corrcoef 应返回 NaN
            assert_array_equal(corrcoef(np.array([])), np.nan)
            # 断言对形状为 (0, 2) 的空数组调用 np.corrcoef 应返回形状为 (0, 0) 的空数组
            assert_array_equal(
                corrcoef(np.array([]).reshape(0, 2)), np.array([]).reshape(0, 0)
            )
            # 断言对形状为 (2, 0) 的空数组调用 np.corrcoef 应返回包含 NaN 的 2x2 数组
            assert_array_equal(
                corrcoef(np.array([]).reshape(2, 0)),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            )

    # 定义测试方法 test_extreme，用于测试在极端数值情况下 np.corrcoef 函数的行为
    def test_extreme(self):
        # 创建包含极端数值的二维列表 x
        x = [[1e-100, 1e100], [1e100, 1e-100]]
        # 计算 x 的相关系数矩阵 c
        c = corrcoef(x)
        # 断言 c 与预期的相关系数矩阵接近
        assert_array_almost_equal(c, np.array([[1.0, -1.0], [-1.0, 1.0]]))
        # 断言 c 的所有元素的绝对值均小于等于 1.0
        assert_(np.all(np.abs(c) <= 1.0))

    # 使用参数化装饰器 parametrize 定义测试方法 test_corrcoef_dtype，测试 np.corrcoef 函数在不同数据类型下的行为
    @parametrize("test_type", [np.half, np.single, np.double])
    def test_corrcoef_dtype(self, test_type):
        # 将测试数据 self.A 转换为指定的数据类型 test_type
        cast_A = self.A.astype(test_type)
        # 调用 np.corrcoef 函数，指定数据类型为 test_type
        res = corrcoef(cast_A, dtype=test_type)
        # 断言 np.corrcoef 返回的结果数据类型与 test_type 相同
        assert test_type == res.dtype
# 使用装饰器实例化参数化测试类
@instantiate_parametrized_tests
# 创建一个测试类 TestCov，继承自 TestCase
class TestCov(TestCase):
    # 定义测试类的静态成员变量 x1，表示一个2x3的numpy数组
    x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
    # 预期的结果 res1，表示一个2x2的numpy数组
    res1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    # 定义测试类的静态成员变量 x2，表示一个1x3的numpy数组，使用ndmin参数指定至少是二维数组
    x2 = np.array([0.0, 1.0, 2.0], ndmin=2)
    # frequencies，表示一个包含3个整数的numpy数组
    frequencies = np.array([1, 4, 1])
    # x2_repeats，表示一个1x6的numpy数组
    x2_repeats = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [2.0]]).T
    # 预期的结果 res2，表示一个2x2的numpy数组
    res2 = np.array([[0.4, -0.4], [-0.4, 0.4]])
    # unit_frequencies，表示一个包含3个整数的numpy数组
    unit_frequencies = np.ones(3, dtype=np.int_)
    # weights，表示一个包含3个浮点数的numpy数组
    weights = np.array([1.0, 4.0, 1.0])
    # 预期的结果 res3，表示一个2x2的numpy数组
    res3 = np.array([[2.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 2.0 / 3.0]])
    # unit_weights，表示一个包含3个浮点数的numpy数组
    unit_weights = np.ones(3)
    # x3，表示一个包含5个浮点数的numpy数组
    x3 = np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])

    # 定义测试方法 test_basic，用于测试 cov 函数对 x1 的计算结果是否与 res1 相符
    def test_basic(self):
        assert_allclose(cov(self.x1), self.res1)

    # 定义测试方法 test_complex，测试对复杂数据 x 的协方差计算结果是否与 res 相符
    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = np.array([[1.0, -1.0j], [1.0j, 1.0]])
        assert_allclose(cov(x), res)
        assert_allclose(cov(x, aweights=np.ones(3)), res)

    # 定义测试方法 test_xy，测试对两个输入数组 x 和 y 的协方差计算结果是否正确
    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(cov(x, y), np.array([[1.0, -1.0j], [1.0j, 1.0]]))

    # 定义测试方法 test_empty，测试对空数组的协方差计算是否返回预期的空值
    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(cov(np.array([])), np.nan)
            assert_array_equal(
                cov(np.array([]).reshape(0, 2)), np.array([]).reshape(0, 0)
            )
            assert_array_equal(
                cov(np.array([]).reshape(2, 0)),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            )

    # 定义测试方法 test_wrong_ddof，测试当自由度参数 ddof 不合理时是否会触发警告或异常
    def test_wrong_ddof(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(
                cov(self.x1, ddof=5), np.array([[np.inf, -np.inf], [-np.inf, np.inf]])
            )

    # 定义测试方法 test_1D_rowvar，测试对一维数据的协方差计算是否与转置后的结果一致
    def test_1D_rowvar(self):
        assert_allclose(cov(self.x3), cov(self.x3, rowvar=False))
        y = np.array([0.0780, 0.3107, 0.2111, 0.0334, 0.8501])
        assert_allclose(cov(self.x3, y), cov(self.x3, y, rowvar=False))

    # 定义测试方法 test_1D_variance，测试对一维数据的协方差计算与方差计算结果是否一致
    def test_1D_variance(self):
        assert_allclose(cov(self.x3, ddof=1), np.var(self.x3, ddof=1))

    # 定义测试方法 test_fweights，测试对带权重的数据进行协方差计算时是否返回预期结果
    def test_fweights(self):
        assert_allclose(cov(self.x2, fweights=self.frequencies), cov(self.x2_repeats))
        assert_allclose(cov(self.x1, fweights=self.frequencies), self.res2)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies), self.res1)
        nonint = self.frequencies + 0.5
        assert_raises((TypeError, RuntimeError), cov, self.x1, fweights=nonint)
        f = np.ones((2, 3), dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = np.ones(2, dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = -1 * np.ones(3, dtype=np.int_)
        assert_raises((ValueError, RuntimeError), cov, self.x1, fweights=f)
    # 测试加权协方差计算函数 `cov` 的功能
    def test_aweights(self):
        # 断言使用给定加权值计算的协方差与预期结果 `self.res3` 接近
        assert_allclose(cov(self.x1, aweights=self.weights), self.res3)
        # 断言使用3倍加权值计算的协方差与使用原始加权值计算的协方差接近
        assert_allclose(
            cov(self.x1, aweights=3.0 * self.weights),
            cov(self.x1, aweights=self.weights),
        )
        # 断言使用单位加权值计算的协方差与预期结果 `self.res1` 接近
        assert_allclose(cov(self.x1, aweights=self.unit_weights), self.res1)
        # 创建一个形状为 (2, 3) 的全一数组 `w`
        w = np.ones((2, 3))
        # 断言当传入非法加权数组时，抛出 RuntimeError 异常
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        # 创建一个长度为2的全一数组 `w`
        w = np.ones(2)
        # 断言当传入非法加权数组时，抛出 RuntimeError 异常
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        # 创建一个长度为3的全负一数组 `w`
        w = -1.0 * np.ones(3)
        # 断言当传入非法加权数组时，抛出 ValueError 或 RuntimeError 异常
        assert_raises((ValueError, RuntimeError), cov, self.x1, aweights=w)

    # 测试同时使用频率权重和加权值计算函数 `cov` 的功能
    def test_unit_fweights_and_aweights(self):
        # 断言使用频率权重和单位加权值计算的协方差与预期结果 `self.x2_repeats` 接近
        assert_allclose(
            cov(self.x2, fweights=self.frequencies, aweights=self.unit_weights),
            cov(self.x2_repeats),
        )
        # 断言使用频率权重和单位加权值计算的协方差与预期结果 `self.res2` 接近
        assert_allclose(
            cov(self.x1, fweights=self.frequencies, aweights=self.unit_weights),
            self.res2,
        )
        # 断言使用单位频率权重和单位加权值计算的协方差与预期结果 `self.res1` 接近
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights),
            self.res1,
        )
        # 断言使用单位频率权重和原始加权值计算的协方差与预期结果 `self.res3` 接近
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.weights),
            self.res3,
        )
        # 断言使用单位频率权重和3倍原始加权值计算的协方差与使用原始加权值计算的协方差接近
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=3.0 * self.weights),
            cov(self.x1, aweights=self.weights),
        )
        # 断言使用单位频率权重和单位加权值计算的协方差与预期结果 `self.res1` 接近
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights),
            self.res1,
        )

    # 使用不同数据类型测试函数 `cov` 的功能
    @parametrize("test_type", [np.half, np.single, np.double])
    def test_cov_dtype(self, test_type):
        # 将数据 `self.x1` 转换为指定的测试数据类型 `test_type`
        cast_x1 = self.x1.astype(test_type)
        # 使用指定数据类型计算协方差 `cov`，并将结果存储在 `res` 中
        res = cov(cast_x1, dtype=test_type)
        # 断言计算出的协方差结果的数据类型与预期的 `test_type` 一致
        assert test_type == res.dtype
class Test_I0(TestCase):
    # 测试 i0 函数的单一输入值
    def test_simple(self):
        # 断言 i0(0.5) 函数返回接近于 1.0634833707413234 的结果数组
        assert_almost_equal(i0(0.5), np.array(1.0634833707413234))

        # 由于实现方式分段，至少需要一个大于 8 的测试
        A = np.array([0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0])
        # 预期的 i0 函数应用于 A 数组后的结果
        expected = np.array(
            [1.06307822, 1.12518299, 1.01214991, 1.00006049, 2815.71662847]
        )
        # 断言 i0(A) 函数返回的结果与预期的结果数组 expected 接近
        assert_almost_equal(i0(A), expected)
        # 断言 i0(-A) 函数返回的结果与预期的结果数组 expected 接近
        assert_almost_equal(i0(-A), expected)

        # 多维数组 B 的测试
        B = np.array(
            [
                [0.827002, 0.99959078],
                [0.89694769, 0.39298162],
                [0.37954418, 0.05206293],
                [0.36465447, 0.72446427],
                [0.48164949, 0.50324519],
            ]
        )
        # 断言 i0(B) 函数返回的结果与预期的多维数组接近
        assert_almost_equal(
            i0(B),
            np.array(
                [
                    [1.17843223, 1.26583466],
                    [1.21147086, 1.03898290],
                    [1.03633899, 1.00067775],
                    [1.03352052, 1.13557954],
                    [1.05884290, 1.06432317],
                ]
            ),
        )
        # gh-11205 的回归测试
        i0_0 = np.i0([0.0])
        # 断言 i0_0 的形状为 (1,)
        assert_equal(i0_0.shape, (1,))
        # 断言 np.i0([0.0]) 返回的结果数组与预期的数组 [1.0] 相等
        assert_array_equal(np.i0([0.0]), np.array([1.0]))

    # 测试 i0 函数对复数的处理
    def test_complex(self):
        a = np.array([0, 1 + 2j])
        # 使用 pytest 来检查对复数值的异常处理
        with pytest.raises(
            (TypeError, RuntimeError),
            # 匹配错误信息 "i0 not supported for complex values"
            # (实际上此注释应该在 match 参数之前)
            match="i0 not supported for complex values",
        ):
            res = i0(a)


class TestKaiser(TestCase):
    # 测试 kaiser 函数的基本功能
    def test_simple(self):
        # 断言 kaiser(1, 1.0) 返回的值是有限的
        assert_(np.isfinite(kaiser(1, 1.0)))
        # 断言 kaiser(0, 1.0) 返回空数组
        assert_almost_equal(kaiser(0, 1.0), np.array([]))
        # 断言 kaiser(2, 1.0) 返回的数组接近于 [0.78984831, 0.78984831]
        assert_almost_equal(kaiser(2, 1.0), np.array([0.78984831, 0.78984831]))
        # 断言 kaiser(5, 1.0) 返回的数组接近于 [0.78984831, 0.94503323, 1.0, 0.94503323, 0.78984831]
        assert_almost_equal(
            kaiser(5, 1.0),
            np.array([0.78984831, 0.94503323, 1.0, 0.94503323, 0.78984831]),
        )
        # 断言 kaiser(5, 1.56789) 返回的数组接近于 [0.58285404, 0.88409679, 1.0, 0.88409679, 0.58285404]
        assert_almost_equal(
            kaiser(5, 1.56789),
            np.array([0.58285404, 0.88409679, 1.0, 0.88409679, 0.58285404]),
        )

    # 测试 kaiser 函数对整数 beta 参数的处理
    def test_int_beta(self):
        # 调用 kaiser(3, 4) 来测试整数 beta 参数


# 跳过 TestMsort 测试类的原因是 msort 已经被弃用，不再关注
@skip(reason="msort is deprecated, do not bother")
class TestMsort(TestCase):
    # 测试 msort 函数的基本功能
    def test_simple(self):
        A = np.array(
            [
                [0.44567325, 0.79115165, 0.54900530],
                [0.36844147, 0.37325583, 0.96098397],
                [0.64864341, 0.52929049, 0.39172155],
            ]
        )
        # 使用 pytest 来检查对 msort 函数被弃用的警告信息
        with pytest.warns(DeprecationWarning, match="msort is deprecated"):
            # 断言 msort(A) 返回的结果数组接近于排序后的 A 数组
            assert_almost_equal(
                msort(A),
                np.array(
                    [
                        [0.36844147, 0.37325583, 0.39172155],
                        [0.44567325, 0.52929049, 0.54900530],
                        [0.64864341, 0.79115165, 0.96098397],
                    ]
                ),
            )


class TestMeshgrid(TestCase):
    # 测试 meshgrid 函数的功能
    # 测试简单的 meshgrid 情况
    def test_simple(self):
        # 创建 X 和 Y，分别为两个数组的网格
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        # 断言 X 和 Y 的值与期望的数组相等
        assert_array_equal(X, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        assert_array_equal(Y, np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]))

    # 测试单个输入的 meshgrid 情况
    def test_single_input(self):
        # 创建 X，为单个数组的网格
        [X] = meshgrid([1, 2, 3, 4])
        # 断言 X 的值与期望的数组相等
        assert_array_equal(X, np.array([1, 2, 3, 4]))

    # 测试无输入的 meshgrid 情况
    def test_no_input(self):
        # 定义空参数列表
        args = []
        # 断言调用 meshgrid 函数返回空数组
        assert_array_equal([], meshgrid(*args))
        # 断言调用 meshgrid 函数返回空数组（copy=False 参数不影响结果）
        assert_array_equal([], meshgrid(*args, copy=False))

    # 测试索引设置的 meshgrid 情况
    def test_indexing(self):
        # 定义两个数组 x 和 y
        x = [1, 2, 3]
        y = [4, 5, 6, 7]
        # 创建 X 和 Y，指定索引方式为 "ij"
        [X, Y] = meshgrid(x, y, indexing="ij")
        # 断言 X 和 Y 的值与期望的数组相等
        assert_array_equal(X, np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
        assert_array_equal(Y, np.array([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]))

        # 测试预期的形状：
        z = [8, 9]
        # 断言调用 meshgrid 函数返回的第一个数组的形状为 (4, 3)
        assert_(meshgrid(x, y)[0].shape == (4, 3))
        # 断言调用 meshgrid 函数返回的第一个数组的形状为 (3, 4)（指定索引方式为 "ij"）
        assert_(meshgrid(x, y, indexing="ij")[0].shape == (3, 4))
        # 断言调用 meshgrid 函数返回的第一个数组的形状为 (4, 3, 2)
        assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        # 断言调用 meshgrid 函数返回的第一个数组的形状为 (3, 4, 2)（指定索引方式为 "ij"）
        assert_(meshgrid(x, y, z, indexing="ij")[0].shape == (3, 4, 2))

        # 测试非法参数索引方式
        assert_raises(ValueError, meshgrid, x, y, indexing="notvalid")

    # 测试稀疏网格的 meshgrid 情况
    def test_sparse(self):
        # 创建 X 和 Y，分别为两个数组的稀疏网格
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7], sparse=True)
        # 断言 X 和 Y 的值与期望的数组相等
        assert_array_equal(X, np.array([[1, 2, 3]]))
        assert_array_equal(Y, np.array([[4], [5], [6], [7]]))

    # 测试无效参数的 meshgrid 情况
    def test_invalid_arguments(self):
        # 测试 meshgrid 对于无效参数的异常处理
        # 对问题编号 #4755 进行回归测试
        assert_raises(TypeError, meshgrid, [1, 2, 3], [4, 5, 6, 7], indices="ij")

    # 测试返回数组类型的 meshgrid 情况
    def test_return_type(self):
        # 对返回数组中的数据类型进行测试
        # 对问题编号 #5297 进行回归测试
        x = np.arange(0, 10, dtype=np.float32)
        y = np.arange(10, 20, dtype=np.float64)

        # 创建 X 和 Y
        X, Y = np.meshgrid(x, y)

        # 断言 X 和 Y 的数据类型与 x 和 y 相同
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # 使用 copy=True 创建 X 和 Y
        X, Y = np.meshgrid(x, y, copy=True)

        # 断言 X 和 Y 的数据类型与 x 和 y 相同
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # 使用 sparse=True 创建 X 和 Y
        X, Y = np.meshgrid(x, y, sparse=True)

        # 断言 X 和 Y 的数据类型与 x 和 y 相同
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

    # 测试写回的 meshgrid 情况
    def test_writeback(self):
        # 对问题编号 #8561 进行回归测试
        X = np.array([1.1, 2.2])
        Y = np.array([3.3, 4.4])
        # 使用 sparse=False 和 copy=True 创建 x 和 y
        x, y = np.meshgrid(X, Y, sparse=False, copy=True)

        # 修改 x 的第一行为 0，并断言修改成功
        x[0, :] = 0
        assert_equal(x[0, :], 0)
        assert_equal(x[1, :], X)

    # 测试多维形状的 meshgrid 情况
    def test_nd_shape(self):
        # 使用不同维度范围的 meshgrid 创建多个数组
        a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        # 预期的数组形状
        expected_shape = (2, 1, 3, 4, 5)
        # 断言每个数组的形状与预期形状相等
        assert_equal(a.shape, expected_shape)
        assert_equal(b.shape, expected_shape)
        assert_equal(c.shape, expected_shape)
        assert_equal(d.shape, expected_shape)
        assert_equal(e.shape, expected_shape)
    # 定义一个测试函数，用于测试多维网格生成器的数值情况
    def test_nd_values(self):
        # 使用 NumPy 的 meshgrid 函数创建三个网格数组 a, b, c
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])
        # 断言数组 a 的值符合预期 [[[0, 0, 0]], [[0, 0, 0]]]
        assert_equal(a, [[[0, 0, 0]], [[0, 0, 0]]])
        # 断言数组 b 的值符合预期 [[[1, 1, 1]], [[2, 2, 2]]]
        assert_equal(b, [[[1, 1, 1]], [[2, 2, 2]]])
        # 断言数组 c 的值符合预期 [[[3, 4, 5]], [[3, 4, 5]]]
        assert_equal(c, [[[3, 4, 5]], [[3, 4, 5]]])

    # 定义一个测试函数，用于测试多维网格生成器的索引方式
    def test_nd_indexing(self):
        # 使用 NumPy 的 meshgrid 函数创建三个网格数组 a, b, c，指定索引方式为 "ij"
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing="ij")
        # 断言数组 a 的值符合预期 [[[0, 0, 0], [0, 0, 0]]]
        assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
        # 断言数组 b 的值符合预期 [[[1, 1, 1], [2, 2, 2]]]
        assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
        # 断言数组 c 的值符合预期 [[[3, 4, 5], [3, 4, 5]]]
        assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])
@xfail  # (reason="TODO: implement")
class TestPiecewise(TestCase):
    # 定义测试方法 test_simple，用于测试 piecewise 函数的简单情况
    def test_simple(self):
        # Condition is single bool list
        # 当条件为单个布尔列表时，测试 piecewise 函数的行为
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        # 当条件为包含单个布尔列表的列表时，测试 piecewise 函数的行为
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        # 当条件为单个布尔数组时，测试 piecewise 函数的行为
        x = piecewise([0, 0], np.array([True, False]), [1])
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        # 当条件为单个整数数组时，测试 piecewise 函数的行为
        x = piecewise([0, 0], np.array([1, 0]), [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        # 当条件为包含整数数组的列表时，测试 piecewise 函数的行为
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

        # Condition is list with lambda function
        # 当条件为包含 lambda 函数的列表时，测试 piecewise 函数的行为
        x = piecewise([0, 0], [[False, True]], [lambda x: -1])
        assert_array_equal(x, [0, -1])

        # Testing ValueError for incorrect number of functions
        # 测试当函数数量不正确时，是否会抛出 ValueError
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            piecewise,
            [0, 0],
            [[False, True]],
            [],
        )
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            piecewise,
            [0, 0],
            [[False, True]],
            [1, 2, 3],
        )

    # 定义测试方法 test_two_conditions，测试 piecewise 函数在两个条件下的行为
    def test_two_conditions(self):
        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    # 定义测试方法 test_scalar_domains_three_conditions，测试 piecewise 函数在三个条件下的行为
    def test_scalar_domains_three_conditions(self):
        x = piecewise(3, [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    # 定义测试方法 test_default，测试 piecewise 函数的默认值行为
    def test_default(self):
        # No value specified for x[1], should be 0
        # 对于未指定 x[1] 值的情况，应为 0
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        # 应将 x[1] 设置为 3
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    # 定义测试方法 test_0d，测试 piecewise 函数在 0 维数据下的行为
    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [True, False], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

        # With 3 ranges (It was failing, before)
        # 测试包含 3 个区间的情况（此前失败的情况）
        y = piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    # 定义测试方法 test_0d_comparison，测试 piecewise 函数在 0 维数据下的比较行为
    def test_0d_comparison(self):
        x = 3
        y = piecewise(x, [x <= 3, x > 3], [4, 0])  # Should succeed.
        assert_equal(y, 4)

        # With 3 ranges (It was failing, before)
        # 测试包含 3 个区间的情况（此前失败的情况）
        x = 4
        y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)

        # Testing ValueError for incorrect number of functions
        # 测试当函数数量不正确时，是否会抛出 ValueError
        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            piecewise,
            x,
            [x <= 3, x > 3],
            [1],
        )
        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            piecewise,
            x,
            [x <= 3, x > 3],
            [1, 1, 1, 1],
        )
    # 定义一个测试函数，测试当输入为标量时的 piecewise 函数行为
    def test_0d_0d_condition(self):
        # 创建一个包含单个元素的 NumPy 数组，即标量
        x = np.array(3)
        # 创建一个布尔数组，指示 x 是否大于 3
        c = np.array(x > 3)
        # 使用 piecewise 函数处理 x，根据条件 c 选择返回值 [1, 2]
        y = piecewise(x, [c], [1, 2])
        # 断言 y 的值等于 2
        assert_equal(y, 2)
    
    # 定义一个测试函数，测试当输入为多维数组时的 piecewise 函数行为
    def test_multidimensional_extrafunc(self):
        # 创建一个包含两行三列的二维 NumPy 数组
        x = np.array([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        # 使用 piecewise 函数处理 x，根据条件 x < 0 和 x >= 2 选择返回值 [-1, 1, 3]
        y = piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        # 断言 y 的值与指定的 NumPy 数组相等
        assert_array_equal(y, np.array([[-1.0, -1.0, -1.0], [3.0, 3.0, 1.0]]))
@instantiate_parametrized_tests
class TestBincount(TestCase):
    # 测试类，用于测试 np.bincount 函数的各种情况

    def test_simple(self):
        # 简单测试，对包含0到3的数组进行统计
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    def test_simple2(self):
        # 另一个简单测试，对指定数组进行统计
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    def test_simple_weight(self):
        # 测试带权重的情况，对数组进行统计并应用权重
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    def test_simple_weight2(self):
        # 另一个带权重的测试，对指定数组进行统计并应用权重
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    def test_with_minlength(self):
        # 测试带最小长度参数的情况，对空数组和指定数组进行统计
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))
        x = []
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([]))

    def test_with_minlength_smaller_than_maxvalue(self):
        # 测试最小长度小于最大值的情况，对指定数组进行统计
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    def test_with_minlength_and_weights(self):
        # 测试带最小长度和权重的情况，对指定数组进行统计并应用权重
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

    def test_empty(self):
        # 测试空数组的情况
        x = np.array([], dtype=int)
        y = np.bincount(x)
        assert_array_equal(x, y)

    def test_empty_with_minlength(self):
        # 测试空数组并带最小长度参数的情况
        x = np.array([], dtype=int)
        y = np.bincount(x, minlength=5)
        assert_array_equal(y, np.zeros(5, dtype=int))

    def test_with_incorrect_minlength(self):
        # 测试带错误最小长度参数的情况，应触发异常
        x = np.array([], dtype=int)
        assert_raises(
            TypeError,
            # 尝试使用非法字符串作为最小长度参数
            lambda: np.bincount(x, minlength="foobar"),
        )
        assert_raises(
            (ValueError, RuntimeError),
            # 尝试使用负数作为最小长度参数
            lambda: np.bincount(x, minlength=-1),
        )

        x = np.arange(5)
        assert_raises(
            TypeError,
            # 尝试使用非法字符串作为最小长度参数
            lambda: np.bincount(x, minlength="foobar"),
        )
        assert_raises(
            (ValueError, RuntimeError),
            # 尝试使用负数作为最小长度参数
            lambda: np.bincount(x, minlength=-1),
        )

    @skipIfTorchDynamo()  # flaky test
    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    # 定义测试方法，用于验证数据类型引用泄漏
    def test_dtype_reference_leaks(self):
        # 注释：gh-6805，这是一个 GitHub 问题的引用标识符
        # 获取整型指针类型的引用计数
        intp_refcount = sys.getrefcount(np.dtype(np.intp))
        # 获取双精度浮点类型的引用计数
        double_refcount = sys.getrefcount(np.dtype(np.double))

        # 循环执行 10 次
        for j in range(10):
            # 对指定数组执行 np.bincount 操作
            np.bincount([1, 2, 3])
        
        # 断言：整型指针类型的引用计数没有变化
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        # 断言：双精度浮点类型的引用计数没有变化
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

        # 再次循环执行 10 次
        for j in range(10):
            # 对指定数组和权重数组执行 np.bincount 操作
            np.bincount([1, 2, 3], [4, 5, 6])
        
        # 断言：整型指针类型的引用计数没有变化
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        # 断言：双精度浮点类型的引用计数没有变化
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

    # 参数化测试装饰器，用于测试非一维数据情况
    @parametrize("vals", [[[2, 2]], 2])
    def test_error_not_1d(self, vals):
        # 测试值必须是一维数组或嵌套列表形式
        # 将 vals 转换为 NumPy 数组
        vals_arr = np.asarray(vals)
        # 使用 assert_raises 检查是否抛出 ValueError 或 RuntimeError 异常
        with assert_raises((ValueError, RuntimeError)):
            np.bincount(vals_arr)
        with assert_raises((ValueError, RuntimeError)):
            np.bincount(vals)
# 使用 parametrize 函数创建一个包含多个测试参数的参数化测试场景
parametrize_interp_sc = parametrize(
    "sc",
    [
        subtest(lambda x: np.float_(x), name="real"),  # 测试实数的情况
        subtest(lambda x: _make_complex(x, 0), name="complex-real"),  # 测试复数的实部
        subtest(lambda x: _make_complex(0, x), name="complex-imag"),  # 测试复数的虚部
        subtest(lambda x: _make_complex(x, np.multiply(x, -2)), name="complex-both"),  # 测试复数的实部和虚部
    ],
)

# 使用装饰器 @xpassIfTorchDynamo，暂时跳过 Torch Dynamo 相关的测试
# 使用装饰器 @instantiate_parametrized_tests，实例化参数化测试
class TestInterp(TestCase):
    def test_exceptions(self):
        # 测试异常情况下是否能正确抛出 ValueError 异常
        assert_raises(ValueError, interp, 0, [], [])
        assert_raises(ValueError, interp, 0, [0], [1, 2])
        assert_raises(ValueError, interp, 0, [0, 1], [1, 2], period=0)
        assert_raises(ValueError, interp, 0, [], [], period=360)
        assert_raises(ValueError, interp, 0, [0], [1, 2], period=360)

    def test_basic(self):
        # 测试基本的插值功能
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.linspace(0, 1, 50)
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_right_left_behavior(self):
        # 测试不同参数下 interp 函数的行为
        # size == 1 时使用特殊处理，1 < size < 5 时使用线性搜索，size >= 5 时使用局部搜索和可能的二分搜索
        for size in range(1, 10):
            xp = np.arange(size, dtype=np.double)
            yp = np.ones(size, dtype=np.double)
            incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
            decpts = incpts[::-1]

            # 测试默认情况下 interp 函数的结果是否符合预期
            incres = interp(incpts, xp, yp)
            decres = interp(decpts, xp, yp)
            inctgt = np.array([1, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            # 测试指定 left 参数后 interp 函数的结果是否符合预期
            incres = interp(incpts, xp, yp, left=0)
            decres = interp(decpts, xp, yp, left=0)
            inctgt = np.array([0, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            # 测试指定 right 参数后 interp 函数的结果是否符合预期
            incres = interp(incpts, xp, yp, right=2)
            decres = interp(decpts, xp, yp, right=2)
            inctgt = np.array([1, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            # 测试同时指定 left 和 right 参数后 interp 函数的结果是否符合预期
            incres = interp(incpts, xp, yp, left=0, right=2)
            decres = interp(decpts, xp, yp, left=0, right=2)
            inctgt = np.array([0, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)
    def test_scalar_interpolation_point(self):
        # 定义测试函数，测试 np.interp() 函数对标量插值的行为

        # 创建等间距的数组 x 和 y
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)

        # 测试 x0 为整数时的插值是否正确
        x0 = 0
        assert_almost_equal(np.interp(x0, x, y), x0)

        # 测试 x0 为浮点数时的插值是否正确
        x0 = 0.3
        assert_almost_equal(np.interp(x0, x, y), x0)

        # 测试 x0 为 np.float32 类型的插值是否正确
        x0 = np.float32(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0)

        # 测试 x0 为 np.float64 类型的插值是否正确
        x0 = np.float64(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0)

        # 测试 x0 为 NaN 时的插值是否正确
        x0 = np.nan
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_non_finite_behavior_exact_x(self):
        # 定义测试函数，测试在 x 精确匹配时的 np.interp() 函数的行为

        # 定义 x 和 xp，以及 fp 中含有无穷大的情况下的插值
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.inf, np.inf, 4])

        # 定义 fp 中含有 NaN 的情况下的插值
        fp = [1, 2, np.nan, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.nan, np.nan, 4])

    @parametrize_interp_sc
    def test_non_finite_any_nan(self, sc):
        """test that nans are propagated"""
        # 参数化测试函数，测试 NaN 是否被正确传播

        # 测试在 x 为 NaN 时的插值是否返回 sc(np.nan)
        assert_equal(np.interp(0.5, [np.nan, 1], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, np.nan], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([np.nan, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([0, np.nan])), sc(np.nan))

    @parametrize_interp_sc
    def test_non_finite_inf(self, sc):
        """Test that interp between opposite infs gives nan"""
        # 参数化测试函数，测试在无穷大的情况下是否返回 NaN

        # 测试在 x 和 y 分别为相反无穷大的情况下是否返回 sc(np.nan)
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, -np.inf])), sc(np.nan))

        # 除非 y 值相等的情况
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([10, 10])), sc(10))

    @parametrize_interp_sc
    def test_non_finite_half_inf_xf(self, sc):
        """Test that interp where both axes have a bound at inf gives nan"""
        # 参数化测试函数，测试当 x 和 y 轴同时有边界为无穷大时是否返回 NaN

        # 测试不同边界条件下的插值是否返回 sc(np.nan)
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([-np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([+np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([-np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([+np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, +np.inf])), sc(np.nan))

    @parametrize_interp_sc
    def test_non_finite_half_inf_x(self, sc):
        """Test interp where the x axis has a bound at inf"""
        # 参数化测试函数，测试当 x 轴有边界为无穷大时的插值行为

        # 测试不同边界条件下的插值是否返回预期值
        assert_equal(np.interp(0.5, [-np.inf, -np.inf], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, 10])), sc(0))
        assert_equal(np.interp(0.5, [+np.inf, +np.inf], sc([0, 10])), sc(0))
    def test_non_finite_half_inf_f(self, sc):
        """Test interp where the f axis has a bound at inf"""
        # 测试当 f 轴在正无穷处有边界时的插值
        assert_equal(np.interp(0.5, [0, 1], sc([0, -np.inf])), sc(-np.inf))
        # 确保在 [0, 1] 区间内插值为 -np.inf
        assert_equal(np.interp(0.5, [0, 1], sc([0, +np.inf])), sc(+np.inf))
        # 确保在 [0, 1] 区间内插值为 +np.inf
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, 10])), sc(-np.inf))
        # 确保在 [0, 1] 区间内插值为 -np.inf
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, 10])), sc(+np.inf))
        # 确保在 [0, 1] 区间内插值为 -np.inf
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, -np.inf])), sc(-np.inf))
        # 确保在 [0, 1] 区间内插值为 -np.inf
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, +np.inf])), sc(+np.inf))

    def test_complex_interp(self):
        # test complex interpolation
        # 测试复数插值
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5)) * 1.0j
        x0 = 0.3
        y0 = x0 + (1 + x0) * 1.0j
        assert_almost_equal(np.interp(x0, x, y), y0)
        # test complex left and right
        # 测试复数左右边界
        x0 = -1
        left = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, left=left), left)
        x0 = 2.0
        right = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, right=right), right)
        # test complex non finite
        # 测试复数非有限值
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2 + 1j, np.inf, 4]
        y = [1, 2 + 1j, np.inf + 0.5j, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), y)
        # test complex periodic
        # 测试复数周期插值
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5 + 1.0j, 10 + 2j, 3 + 3j, 4 + 4j]
        y = [
            7.5 + 1.5j,
            5.0 + 1.0j,
            8.75 + 1.75j,
            6.25 + 1.25j,
            3.0 + 3j,
            3.25 + 3.25j,
            3.5 + 3.5j,
            3.75 + 3.75j,
        ]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)

    def test_zero_dimensional_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.array(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0.item())
        # Ensure interpolation at a zero-dimensional point returns the point itself
        # 确保在零维插值点返回该点本身

        xp = np.array([0, 2, 4])
        fp = np.array([1, -1, 1])

        actual = np.interp(np.array(1), xp, fp)
        assert_equal(actual, 0)
        assert_(isinstance(actual, (np.float64, np.ndarray)))
        # Ensure interpolation at a point outside the range returns the corresponding boundary value
        # 确保在超出范围的点插值返回对应的边界值

        actual = np.interp(np.array(4.5), xp, fp, period=4)
        assert_equal(actual, 0.5)
        assert_(isinstance(actual, (np.float64, np.ndarray)))
        # Ensure interpolation with a specified period returns correct results
        # 确保使用指定周期进行插值返回正确结果

    def test_if_len_x_is_small(self):
        xp = np.arange(0, 10, 0.0001)
        fp = np.sin(xp)
        assert_almost_equal(np.interp(np.pi, xp, fp), 0.0)
        # Ensure interpolation works correctly when the length of x is small
        # 确保在 x 的长度很小时插值功能正常运行

    def test_period(self):
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5, 10, 3, 4]
        y = [7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
        # Ensure periodic interpolation returns correct results
        # 确保周期插值返回正确的结果

        x = np.array(x, order="F").reshape(2, -1)
        y = np.array(y, order="C").reshape(2, -1)
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
        # Ensure interpolation works correctly with Fortran-ordered x and C-ordered y
        # 确保在使用Fortran顺序的x和C顺序的y时插值功能正常运行
# 使用装饰器实例化带参数化测试的测试类
@instantiate_parametrized_tests
class TestPercentile(TestCase):
    
    # 被跳过的测试方法，因为在持续集成环境下失败了，原因是"NP_VER: fails on CI; no method="
    @skip(reason="NP_VER: fails on CI; no method=")
    def test_basic(self):
        # 创建数组 x，包含元素 [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        x = np.arange(8) * 0.5
        # 断言：计算 x 的百分位数 0%，应为 0.0
        assert_equal(np.percentile(x, 0), 0.0)
        # 断言：计算 x 的百分位数 100%，应为 3.5
        assert_equal(np.percentile(x, 100), 3.5)
        # 断言：计算 x 的百分位数 50%，应为 1.75
        assert_equal(np.percentile(x, 50), 1.75)
        # 修改 x 中的第二个元素为 NaN
        x[1] = np.nan
        # 断言：计算 x 的百分位数 0%，预期结果为 NaN
        assert_equal(np.percentile(x, 0), np.nan)
        # 断言：使用最近方法计算 x 的百分位数 0%，预期结果为 NaN
        assert_equal(np.percentile(x, 0, method="nearest"), np.nan)

    # 被跳过的测试方法，因为不支持 Fraction 对象，原因是"support Fraction objects?"
    @skip(reason="support Fraction objects?")
    def test_fraction(self):
        # 创建包含 Fraction 对象的列表 x
        x = [Fraction(i, 2) for i in range(8)]
        
        # 计算 x 的百分位数，使用 Fraction(0)
        p = np.percentile(x, Fraction(0))
        assert_equal(p, Fraction(0))
        assert_equal(type(p), Fraction)

        # 计算 x 的百分位数，使用 Fraction(100)
        p = np.percentile(x, Fraction(100))
        assert_equal(p, Fraction(7, 2))
        assert_equal(type(p), Fraction)

        # 计算 x 的百分位数，使用 Fraction(50)
        p = np.percentile(x, Fraction(50))
        assert_equal(p, Fraction(7, 4))
        assert_equal(type(p), Fraction)

        # 计算 x 的百分位数，使用 [Fraction(50)]
        p = np.percentile(x, [Fraction(50)])
        assert_equal(p, np.array([Fraction(7, 4)]))
        assert_equal(type(p), np.ndarray)

    # 测试 np.percentile 的 API 调用
    def test_api(self):
        d = np.ones(5)
        np.percentile(d, 5, None, None, False)
        np.percentile(d, 5, None, None, False, "linear")
        o = np.ones((1,))
        np.percentile(d, 5, None, o, False, "linear")

    # 被标记为预期失败的测试方法，原因是"TODO: implement"
    @xfail(reason="TODO: implement")
    def test_complex(self):
        # 创建复数数组 arr_c
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="D")
        # 断言：尝试计算复数数组的百分位数，应引发 TypeError
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="F")
        # 断言：尝试计算复数数组的百分位数，应引发 TypeError
        assert_raises(TypeError, np.percentile, arr_c, 0.5)

    # 测试二维数组 x 的百分位数计算
    def test_2D(self):
        x = np.array([[1, 1, 1], [1, 1, 1], [4, 4, 3], [1, 1, 1], [1, 1, 1]])
        assert_array_equal(np.percentile(x, 50, axis=0), [1, 1, 1])

    # 被跳过的测试方法，因为在持续集成环境下失败了，原因是"NP_VER: fails on CI; no method="
    @skip(reason="NP_VER: fails on CI; no method=")
    @xpassIfTorchDynamo  # (reason="TODO: implement")
    @parametrize("dtype", np.typecodes["Float"])
    def test_linear_nan_1D(self, dtype):
        # 创建包含 NaN 的数组 arr
        arr = np.asarray([15.0, np.NAN, 35.0, 40.0, 50.0], dtype=dtype)
        # 计算 arr 的百分位数 40%，使用方法 "linear"
        res = np.percentile(arr, 40.0, method="linear")
        # 断言：计算结果应为 NaN
        np.testing.assert_equal(res, np.NAN)
        # 断言：计算结果的数据类型应与 arr 的数据类型相同
        np.testing.assert_equal(res.dtype, arr.dtype)

    # 创建整数和 np.float64 对象的元组列表，用于测试用例
    H_F_TYPE_CODES = [
        (int_type, np.float64) for int_type in "Bbhil"  # np.typecodes["AllInteger"]
    ] + [
        (np.float16, np.float16),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ]

    # 被跳过的测试方法，因为 NEP 50 是 1.24 版本中的新特性
    @skip(reason="NEP 50 is new in 1.24")
    @parametrize("input_dtype, expected_dtype", H_F_TYPE_CODES)
    # 参数化测试，针对不同的方法和期望结果进行多次测试
    @parametrize(
        "method, expected",
        [
            ("inverted_cdf", 20),
            ("averaged_inverted_cdf", 27.5),
            ("closest_observation", 20),
            ("interpolated_inverted_cdf", 20),
            ("hazen", 27.5),
            ("weibull", 26),
            ("linear", 29),
            ("median_unbiased", 27),
            ("normal_unbiased", 27.125),
        ],
    )
    # 测试线性插值方法
    def test_linear_interpolation(self, method, expected, input_dtype, expected_dtype):
        # 将期望的数据类型转换为 numpy 的数据类型
        expected_dtype = np.dtype(expected_dtype)

        # 如果 numpy 版本是 legacy，使用 np.promote_types 将期望数据类型提升为 np.float64
        if (
            hasattr(np, "_get_promotion_state")
            and np._get_promotion_state() == "legacy"
        ):
            expected_dtype = np.promote_types(expected_dtype, np.float64)

        # 创建输入数组 arr，使用指定的数据类型
        arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
        # 计算 arr 数组的第 40% 分位数，使用给定的方法
        actual = np.percentile(arr, 40.0, method=method)

        # 使用 np.testing.assert_almost_equal 函数进行近似相等的断言，精度为 14 位小数
        np.testing.assert_almost_equal(actual, expected_dtype.type(expected), 14)

        # 如果方法是 "inverted_cdf" 或 "closest_observation"，则检查 actual 的数据类型
        if method in ["inverted_cdf", "closest_observation"]:
            np.testing.assert_equal(np.asarray(actual).dtype, np.dtype(input_dtype))
        else:
            np.testing.assert_equal(np.asarray(actual).dtype, np.dtype(expected_dtype))

    # 定义类型代码 TYPE_CODES，包括所有整数和浮点数类型
    TYPE_CODES = np.typecodes["AllInteger"] + np.typecodes["Float"]

    # 跳过测试，原因是 NP_VER: fails on CI; no method=
    @skip(reason="NP_VER: fails on CI; no method=")
    # 参数化测试，针对 TYPE_CODES 中的每种数据类型进行测试
    @parametrize("dtype", TYPE_CODES)
    # 测试 lower 和 higher 方法
    def test_lower_higher(self, dtype):
        # 断言 lower 方法的结果
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50, method="lower"), 4)
        # 断言 higher 方法的结果
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 50, method="higher"), 5)

    # 跳过测试，原因是 NP_VER: fails on CI; no method=
    @skip(reason="NP_VER: fails on CI; no method=")
    # 参数化测试，针对 TYPE_CODES 中的每种数据类型进行测试
    @parametrize("dtype", TYPE_CODES)
    # 测试 midpoint 方法
    def test_midpoint(self, dtype):
        # 断言 midpoint 方法的结果
        assert_equal(
            np.percentile(np.arange(10, dtype=dtype), 51, method="midpoint"), 4.5
        )
        assert_equal(
            np.percentile(np.arange(9, dtype=dtype) + 1, 50, method="midpoint"), 5
        )
        assert_equal(
            np.percentile(np.arange(11, dtype=dtype), 51, method="midpoint"), 5.5
        )
        assert_equal(
            np.percentile(np.arange(11, dtype=dtype), 50, method="midpoint"), 5
        )

    # 跳过测试，原因是 NP_VER: fails on CI; no method=
    @skip(reason="NP_VER: fails on CI; no method=")
    # 参数化测试，针对 TYPE_CODES 中的每种数据类型进行测试
    @parametrize("dtype", TYPE_CODES)
    # 测试 nearest 方法
    def test_nearest(self, dtype):
        # 断言 nearest 方法的结果
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 51, method="nearest"), 5)
        assert_equal(np.percentile(np.arange(10, dtype=dtype), 49, method="nearest"), 4)

    # 测试线性插值和外推方法
    def test_linear_interpolation_extrapolation(self):
        # 创建一个包含随机数的数组 arr
        arr = np.random.rand(5)

        # 计算 arr 数组的第 100% 分位数，应等于数组的最大值
        actual = np.percentile(arr, 100)
        np.testing.assert_equal(actual, arr.max())

        # 计算 arr 数组的第 0% 分位数，应等于数组的最小值
        actual = np.percentile(arr, 0)
        np.testing.assert_equal(actual, arr.min())

    # 测试序列中的分位数计算
    def test_sequence(self):
        # 创建一个数组 x，包含序列 [0, 0.5, 1.0, ..., 3.5]
        x = np.arange(8) * 0.5
        # 断言 x 数组的第 [0%, 100%, 50%] 分位数
        assert_equal(np.percentile(x, [0, 100, 50]), [0, 3.5, 1.75])

    # 跳过测试，原因是 NP_VER: fails on CI
    # 定义一个测试方法，用于测试 np.percentile 函数在不同轴上的行为
    def test_axis(self):
        # 创建一个 3x4 的数组 x，包含元素从 0 到 11
        x = np.arange(12).reshape(3, 4)

        # 断言计算 x 数组中第 25%、50% 和 100% 分位数的值，并与给定值 [2.75, 5.5, 11.0] 进行比较
        assert_equal(np.percentile(x, (25, 50, 100)), [2.75, 5.5, 11.0])

        # 定义期望的结果 r0，包含了 x 数组在轴 0 上的第 25%、50% 和 100% 分位数的值
        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        # 断言计算 x 数组在轴 0 上的第 25%、50% 和 100% 分位数的值，并与 r0 进行比较
        assert_equal(np.percentile(x, (25, 50, 100), axis=0), r0)

        # 定义期望的结果 r1，包含了 x 数组在轴 1 上的第 25%、50% 和 100% 分位数的值的转置
        r1 = [[0.75, 1.5, 3], [4.75, 5.5, 7], [8.75, 9.5, 11]]
        # 断言计算 x 数组在轴 1 上的第 25%、50% 和 100% 分位数的值的转置，并与 r1 转置后的 np.array 进行比较
        assert_equal(np.percentile(x, (25, 50, 100), axis=1), np.array(r1).T)

        # 确保 qth 轴始终作为 np.array(old_percentile(..)) 的第一个参数
        x = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)

        # 断言计算 x 数组的第 25% 和 50% 分位数的形状，并与期望的形状 (2,) 进行比较
        assert_equal(np.percentile(x, (25, 50)).shape, (2,))
        assert_equal(np.percentile(x, (25, 50, 75)).shape, (3,))
        assert_equal(np.percentile(x, (25, 50), axis=0).shape, (2, 4, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=1).shape, (2, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=2).shape, (2, 3, 4, 6))
        assert_equal(np.percentile(x, (25, 50), axis=3).shape, (2, 3, 4, 5))
        assert_equal(np.percentile(x, (25, 50, 75), axis=1).shape, (3, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50), method="higher").shape, (2,))
        assert_equal(np.percentile(x, (25, 50, 75), method="higher").shape, (3,))
        assert_equal(np.percentile(x, (25, 50), axis=0, method="higher").shape, (2, 4, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=1, method="higher").shape, (2, 3, 5, 6))
        assert_equal(np.percentile(x, (25, 50), axis=2, method="higher").shape, (2, 3, 4, 6))
        assert_equal(np.percentile(x, (25, 50), axis=3, method="higher").shape, (2, 3, 4, 5))
        assert_equal(np.percentile(x, (25, 50, 75), axis=1, method="higher").shape, (3, 3, 5, 6))

    # 根据 NumPy 版本跳过测试，如果当前版本小于 1.22
    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails with NumPy 1.21.2 on CI")
    # 定义测试函数，验证 np.percentile 的标量结果计算
    def test_scalar_q(self):
        # 创建一个 3x4 的数组
        x = np.arange(12).reshape(3, 4)
        # 断言计算数组 x 的第50百分位数结果为 5.5
        assert_equal(np.percentile(x, 50), 5.5)
        
        # 断言计算数组 x 沿 axis=0 的第50百分位数结果为 r0
        r0 = np.array([4.0, 5.0, 6.0, 7.0])
        assert_equal(np.percentile(x, 50, axis=0), r0)
        # 断言计算数组 x 沿 axis=0 的第50百分位数结果的形状与 r0 相同
        assert_equal(np.percentile(x, 50, axis=0).shape, r0.shape)
        
        # 断言计算数组 x 沿 axis=1 的第50百分位数结果为 r1
        r1 = np.array([1.5, 5.5, 9.5])
        assert_almost_equal(np.percentile(x, 50, axis=1), r1)
        # 断言计算数组 x 沿 axis=1 的第50百分位数结果的形状与 r1 相同
        assert_equal(np.percentile(x, 50, axis=1).shape, r1.shape)

        # 使用指定的输出数组 out 进行百分位数计算
        out = np.empty(1)
        assert_equal(np.percentile(x, 50, out=out), 5.5)
        assert_equal(out, 5.5)
        
        out = np.empty(4)
        assert_equal(np.percentile(x, 50, axis=0, out=out), r0)
        assert_equal(out, r0)
        
        out = np.empty(3)
        assert_equal(np.percentile(x, 50, axis=1, out=out), r1)
        assert_equal(out, r1)

        # 再次测试，使用 method="lower" 参数计算第50百分位数
        x = np.arange(12).reshape(3, 4)
        assert_equal(np.percentile(x, 50, method="lower"), 5.0)
        # 断言 np.percentile 返回的结果是标量
        assert_(np.isscalar(np.percentile(x, 50)))
        
        # 使用 method="lower" 参数计算数组 x 沿 axis=0 的第50百分位数
        r0 = np.array([4.0, 5.0, 6.0, 7.0])
        c0 = np.percentile(x, 50, method="lower", axis=0)
        assert_equal(c0, r0)
        assert_equal(c0.shape, r0.shape)
        
        # 使用 method="lower" 参数计算数组 x 沿 axis=1 的第50百分位数
        r1 = np.array([1.0, 5.0, 9.0])
        c1 = np.percentile(x, 50, method="lower", axis=1)
        assert_almost_equal(c1, r1)
        assert_equal(c1.shape, r1.shape)

    @xfail  # (reason="numpy: x.dtype is int, out is int; torch: result is float")
    # 定义被标记为 xfail 的测试函数，预期会失败
    def test_scalar_q_2(self):
        x = np.arange(12).reshape(3, 4)
        # 创建与 x 相同数据类型的空数组 out
        out = np.empty((), dtype=x.dtype)
        # 使用 method="lower" 参数计算数组 x 的第50百分位数，并将结果存入 out
        c = np.percentile(x, 50, method="lower", out=out)
        assert_equal(c, 5)
        assert_equal(out, 5)
        
        # 创建与 x 相同数据类型的长度为 4 的空数组 out
        out = np.empty(4, dtype=x.dtype)
        # 使用 method="lower" 参数计算数组 x 沿 axis=0 的第50百分位数，并将结果存入 out
        c = np.percentile(x, 50, method="lower", axis=0, out=out)
        assert_equal(c, r0)
        assert_equal(out, r0)
        
        # 创建与 x 相同数据类型的长度为 3 的空数组 out
        out = np.empty(3, dtype=x.dtype)
        # 使用 method="lower" 参数计算数组 x 沿 axis=1 的第50百分位数，并将结果存入 out
        c = np.percentile(x, 50, method="lower", axis=1, out=out)
        assert_equal(c, r1)
        assert_equal(out, r1)

    @skip(reason="NP_VER: fails on CI; no method=")
    # 定义被标记为 skip 的测试函数，跳过执行，原因是 NP_VER 环境下方法不支持
    def test_exception(self):
        # 断言调用 np.percentile 时捕获 RuntimeError 或 ValueError 异常
        assert_raises(
            (RuntimeError, ValueError), np.percentile, [1, 2], 56, method="foobar"
        )
        assert_raises((RuntimeError, ValueError), np.percentile, [1], 101)
        assert_raises((RuntimeError, ValueError), np.percentile, [1], -1)
        assert_raises(
            (RuntimeError, ValueError), np.percentile, [1], list(range(50)) + [101]
        )
        assert_raises(
            (RuntimeError, ValueError), np.percentile, [1], list(range(50)) + [-0.1]
        )

    # 定义测试函数，验证 np.percentile 对于包含单个列表的情况
    def test_percentile_list(self):
        assert_equal(np.percentile([1, 2, 3], 0), 1)

    @skip(reason="NP_VER: fails on CI; no method=")
    # 定义被标记为 skip 的测试函数，跳过执行，原因是 NP_VER 环境下方法不支持
    # 定义测试方法，用于测试 numpy.percentile 函数的各种用法和参数组合
    def test_percentile_out(self):
        # 创建一个包含元素 [1, 2, 3] 的 numpy 数组 x
        x = np.array([1, 2, 3])
        # 创建一个形状为 (3,) 的全零 numpy 数组 y
        y = np.zeros((3,))
        # 定义百分位数数组 p，包含元素 (1, 2, 3)
        p = (1, 2, 3)
        # 将 x 中百分位数 p 的结果存入 y 中
        np.percentile(x, p, out=y)
        # 断言 np.percentile 函数计算出的百分位数与预期的 y 相等
        assert_equal(np.percentile(x, p), y)

        # 创建一个二维 numpy 数组 x
        x = np.array([[1, 2, 3], [4, 5, 6]])

        # 创建一个形状为 (3, 3) 的全零 numpy 数组 y
        y = np.zeros((3, 3))
        # 计算 x 沿第一维度（列方向）的百分位数 p，并将结果存入 y 中
        np.percentile(x, p, axis=0, out=y)
        # 断言 np.percentile 函数计算出的百分位数与预期的 y 相等
        assert_equal(np.percentile(x, p, axis=0), y)

        # 创建一个形状为 (3, 2) 的全零 numpy 数组 y
        y = np.zeros((3, 2))
        # 计算 x 沿第二维度（行方向）的百分位数 p，并将结果存入 y 中
        np.percentile(x, p, axis=1, out=y)
        # 断言 np.percentile 函数计算出的百分位数与预期的 y 相等
        assert_equal(np.percentile(x, p, axis=1), y)

        # 创建一个形状为 (3, 4) 的 numpy 数组 x
        x = np.arange(12).reshape(3, 4)
        # 定义预期的百分位数结果 r0
        r0 = np.array([[2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]])
        # 创建一个形状为 (2, 4) 的空 numpy 数组 out
        out = np.empty((2, 4))
        # 计算 x 沿第一维度（列方向）的百分位数 (25, 50)，并将结果存入 out 中
        assert_equal(np.percentile(x, (25, 50), axis=0, out=out), r0)
        # 断言 out 中的结果与预期的 r0 相等
        assert_equal(out, r0)
        
        # 定义预期的百分位数结果 r1
        r1 = np.array([[0.75, 4.75, 8.75], [1.5, 5.5, 9.5]])
        # 创建一个形状为 (2, 3) 的空 numpy 数组 out
        out = np.empty((2, 3))
        # 计算 x 沿第二维度（行方向）的百分位数 (25, 50)，并将结果存入 out 中
        assert_equal(np.percentile(x, (25, 50), axis=1, out=out), r1)
        # 断言 out 中的结果与预期的 r1 相等
        assert_equal(out, r1)

        # 定义预期的百分位数结果 r0
        r0 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        # 创建一个形状为 (2, 4)、数据类型与 x 相同的空 numpy 数组 out
        out = np.empty((2, 4), dtype=x.dtype)
        # 计算 x 沿第一维度（列方向）的百分位数 (25, 50)，使用 method="lower"，并将结果存入 out 中
        c = np.percentile(x, (25, 50), method="lower", axis=0, out=out)
        # 断言计算出的 c 与预期的 r0 相等
        assert_equal(c, r0)
        # 断言 out 中的结果与预期的 r0 相等
        assert_equal(out, r0)
        
        # 定义预期的百分位数结果 r1
        r1 = np.array([[0, 4, 8], [1, 5, 9]])
        # 创建一个形状为 (2, 3)、数据类型与 x 相同的空 numpy 数组 out
        out = np.empty((2, 3), dtype=x.dtype)
        # 计算 x 沿第二维度（行方向）的百分位数 (25, 50)，使用 method="lower"，并将结果存入 out 中
        c = np.percentile(x, (25, 50), method="lower", axis=1, out=out)
        # 断言计算出的 c 与预期的 r1 相等
        assert_equal(c, r1)
        # 断言 out 中的结果与预期的 r1 相等
        assert_equal(out, r1)

    @skip(reason="NP_VER: fails on CI; no method=")
    # 定义跳过该测试方法的装饰器，原因是在 CI 中出现失败，且不使用 method 参数
    def test_percentile_empty_dim(self):
        # 验证空维度保持不变
        # 创建一个形状为 (11, 1, 2, 1) 的 numpy 数组 d
        d = np.arange(11 * 2).reshape(11, 1, 2, 1)
        # 断言计算 d 沿不同轴的百分位数后形状的正确性
        assert_array_equal(np.percentile(d, 50, axis=0).shape, (1, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=1).shape, (11, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=2).shape, (11, 1, 1))
        assert_array_equal(np.percentile(d, 50, axis=3).shape, (11, 1, 2))
        assert_array_equal(np.percentile(d, 50, axis=-1).shape, (11, 1, 2))
        assert_array_equal(np.percentile(d, 50, axis=-2).shape, (11, 1, 1))
        assert_array_equal(np.percentile(d, 50, axis=-3).shape, (11, 2, 1))
        assert_array_equal(np.percentile(d, 50, axis=-4).shape, (1, 2, 1))

        # 验证使用 method="midpoint" 计算百分位数后形状的正确性
        assert_array_equal(
            np.percentile(d, 50, axis=2, method="midpoint").shape, (11, 1, 1)
        )
        assert_array_equal(
            np.percentile(d, 50, axis=-2, method="midpoint").shape, (11, 1, 1)
        )

        # 验证计算多个百分位数后结果的形状正确性
        assert_array_equal(
            np.array(np.percentile(d, [10, 50], axis=0)).shape, (2, 1, 2, 1)
        )
        assert_array_equal(
            np.array(np.percentile(d, [10, 50], axis=1)).shape, (2, 11, 2, 1)
        )
        assert_array_equal(
            np.array(np.percentile(d, [10, 50], axis=2)).shape, (2, 11, 1, 1)
        )
        assert_array_equal(
            np.array(np.percentile(d, [10, 50], axis=3)).shape, (2, 11, 1, 2)
        )
    # 定义一个测试函数，测试 np.percentile 函数在不覆盖输入数组的情况下的行为
    def test_percentile_no_overwrite(self):
        # 创建一个 NumPy 数组
        a = np.array([2, 3, 4, 1])
        # 调用 np.percentile 计算百分位数，不覆盖输入数组，返回结果并未赋值
        np.percentile(a, [50], overwrite_input=False)
        # 断言输入数组不变
        assert_equal(a, np.array([2, 3, 4, 1]))

        # 重新设置数组
        a = np.array([2, 3, 4, 1])
        # 调用 np.percentile 计算百分位数，默认覆盖输入数组，未赋值返回结果
        np.percentile(a, [50])
        # 断言输入数组不变
        assert_equal(a, np.array([2, 3, 4, 1]))

    # 跳过测试的装饰器，指定跳过的原因为 "NP_VER: fails on CI; no method="
    @skip(reason="NP_VER: fails on CI; no method=")
    # 定义一个测试函数，测试 np.percentile 函数在不同参数下的行为
    def test_no_p_overwrite(self):
        # 创建一个等分数组
        p = np.linspace(0.0, 100.0, num=5)
        # 调用 np.percentile 计算百分位数，使用 "midpoint" 方法，不覆盖输入数组
        np.percentile(np.arange(100.0), p, method="midpoint")
        # 断言数组 p 没有改变
        assert_array_equal(p, np.linspace(0.0, 100.0, num=5))
        
        # 将 p 转换为列表
        p = np.linspace(0.0, 100.0, num=5).tolist()
        # 再次调用 np.percentile 计算百分位数，使用 "midpoint" 方法，不覆盖输入数组
        np.percentile(np.arange(100.0), p, method="midpoint")
        # 断言数组 p 没有改变
        assert_array_equal(p, np.linspace(0.0, 100.0, num=5).tolist())

    # 定义一个测试函数，测试 np.percentile 函数在覆盖输入数组的情况下的行为
    def test_percentile_overwrite(self):
        # 创建一个 NumPy 数组
        a = np.array([2, 3, 4, 1])
        # 调用 np.percentile 计算百分位数，覆盖输入数组，返回结果给变量 b
        b = np.percentile(a, [50], overwrite_input=True)
        # 断言变量 b 的值等于 [2.5]
        assert_equal(b, np.array([2.5]))

        # 直接使用列表调用 np.percentile，计算百分位数，覆盖输入数组，返回结果给变量 b
        b = np.percentile([2, 3, 4, 1], [50], overwrite_input=True)
        # 断言变量 b 的值等于 [2.5]
        assert_equal(b, np.array([2.5]))

    # 跳过测试的装饰器，指定跳过的原因为 "pytorch percentile does not support tuple axes."
    @xpassIfTorchDynamo  # (reason="pytorch percentile does not support tuple axes.")
    # 定义一个测试函数，测试 np.percentile 函数在多维数组及轴参数下的行为
    def test_extended_axis(self):
        # 创建一个形状为 (71, 23) 的正态分布随机数组 o
        o = np.random.normal(size=(71, 23))
        # 创建一个沿着第三维度堆叠 10 次的数组 x
        x = np.dstack([o] * 10)
        # 断言沿着指定轴 (0, 1) 的百分位数等于 o 的百分位数的单个元素
        assert_equal(np.percentile(x, 30, axis=(0, 1)), np.percentile(o, 30).item())
        
        # 将 x 的最后一个轴移动到第一个轴
        x = np.moveaxis(x, -1, 0)
        # 断言沿着指定轴 (-2, -1) 的百分位数等于 o 的百分位数的单个元素
        assert_equal(np.percentile(x, 30, axis=(-2, -1)), np.percentile(o, 30).item())
        
        # 恢复 x 的原始轴顺序
        x = x.swapaxes(0, 1).copy()
        # 断言沿着指定轴 (0, -1) 的百分位数等于 o 的百分位数的单个元素
        assert_equal(np.percentile(x, 30, axis=(0, -1)), np.percentile(o, 30).item())
        
        # 再次恢复 x 的原始轴顺序
        x = x.swapaxes(0, 1).copy()

        # 断言沿着多个轴 (0, 1, 2) 的百分位数等于沿着所有轴的百分位数
        assert_equal(
            np.percentile(x, [25, 60], axis=(0, 1, 2)),
            np.percentile(x, [25, 60], axis=None),
        )
        # 断言沿着指定轴 (0,) 的百分位数等于沿着第一个轴的百分位数
        assert_equal(
            np.percentile(x, [25, 60], axis=(0,)), np.percentile(x, [25, 60], axis=0)
        )

        # 创建一个形状为 (3, 5, 7, 11) 的数组 d，其中元素为 0 到 3*5*7*11-1 的整数
        d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
        # 打乱数组 d 的元素顺序
        np.random.shuffle(d.ravel())
        # 断言沿着指定轴 (0, 1, 2) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, 25, axis=(0, 1, 2))[0],
            np.percentile(d[:, :, :, 0].flatten(), 25),
        )
        # 断言沿着指定轴 (0, 1, 3) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, [10, 90], axis=(0, 1, 3))[:, 1],
            np.percentile(d[:, :, 1, :].flatten(), [10, 90]),
        )
        # 断言沿着指定轴 (3, 1, -4) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, 25, axis=(3, 1, -4))[2],
            np.percentile(d[:, :, 2, :].flatten(), 25),
        )
        # 断言沿着指定轴 (3, 1, 2) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, 25, axis=(3, 1, 2))[2],
            np.percentile(d[2, :, :, :].flatten(), 25),
        )
        # 断言沿着指定轴 (3, 2) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, 25, axis=(3, 2))[2, 1],
            np.percentile(d[2, 1, :, :].flatten(), 25),
        )
        # 断言沿着指定轴 (1, -2) 的百分位数等于沿着特定切片的百分位数
        assert_equal(
            np.percentile(d, 25, axis=(1, -2))[2,
    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        # 测试当指定的轴超出数组维度时是否引发 AxisError 异常
        assert_raises(np.AxisError, np.percentile, d, axis=-5, q=25)
        # 测试当指定的轴包含无效的负索引时是否引发 AxisError 异常
        assert_raises(np.AxisError, np.percentile, d, axis=(0, -5), q=25)
        # 测试当指定的轴超出数组维度时是否引发 AxisError 异常
        assert_raises(np.AxisError, np.percentile, d, axis=4, q=25)
        # 测试当指定的轴超出数组维度时是否引发 AxisError 异常
        assert_raises(np.AxisError, np.percentile, d, axis=(0, 4), q=25)
        # 测试当同一轴在参数中被重复指定时是否引发 ValueError 异常
        assert_raises(ValueError, np.percentile, d, axis=(1, 1), q=25)
        # 测试当同一轴在参数中被重复指定时是否引发 ValueError 异常
        assert_raises(ValueError, np.percentile, d, axis=(-1, -1), q=25)
        # 测试当包含无效轴索引时是否引发 ValueError 异常
        assert_raises(ValueError, np.percentile, d, axis=(3, -1), q=25)

    def test_keepdims(self):
        d = np.ones((3, 5, 7, 11))
        # 测试当 keepdims=True 时，percentile 函数返回的数组形状是否符合预期
        assert_equal(np.percentile(d, 7, axis=None, keepdims=True).shape, (1, 1, 1, 1))

    @xfail  # (reason="pytorch percentile does not support tuple axes.")
    def test_keepdims_2(self):
        # 测试当 keepdims=True 且使用元组指定轴时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, 7, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11)
        )
        # 测试当 keepdims=True 且使用元组指定轴时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, 7, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1)
        )
        # 测试当 keepdims=True 且使用元组指定轴时，percentile 函数返回的数组形状是否符合预期
        assert_equal(np.percentile(d, 7, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
        # 测试当 keepdims=True 且使用元组指定轴时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, 7, (0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1)
        )
        # 测试当 keepdims=True 且使用元组指定轴时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, 7, axis=(0, 1, 3), keepdims=True).shape, (1, 1, 7, 1)
        )

        # 测试当 keepdims=True 且使用元组指定轴以及多个分位数时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, [1, 7], axis=(0, 1, 3), keepdims=True).shape,
            (2, 1, 1, 7, 1),
        )
        # 测试当 keepdims=True 且使用元组指定轴以及多个分位数时，percentile 函数返回的数组形状是否符合预期
        assert_equal(
            np.percentile(d, [1, 7], axis=(0, 3), keepdims=True).shape, (2, 1, 5, 7, 1)
        )

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @parametrize(
        "q",
        [
            7,
            subtest(
                [1, 7],
                decorators=[
                    skip(reason="Keepdims wrapper incorrect for multiple q"),
                ],
            ),
        ],
    )
    @parametrize(
        "axis",
        [
            None,
            1,
            subtest((1,)),
            subtest(
                (0, 1),
                decorators=[
                    skip(reason="Tuple axes"),
                ],
            ),
            subtest(
                (-3, -1),
                decorators=[
                    skip(reason="Tuple axes"),
                ],
            ),
        ],
    )
    # 定义一个测试函数，用于测试 np.percentile 的 keepdims=True 参数的行为
    def test_keepdims_out(self, q, axis):
        # 创建一个全为1的多维数组 d
        d = np.ones((3, 5, 7, 11))
        # 如果 axis 为 None，则 shape_out 是一个元组，包含 d 的维度个数个元素，每个元素为1
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            # 根据传入的 axis 参数，计算规范化后的 axis_norm
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            # 根据 axis_norm 构建 shape_out 元组，保持非 axis 维度不变，axis 维度为1
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim)
            )
        # 将 q 的形状与 shape_out 相加，形成最终的 shape_out
        shape_out = np.shape(q) + shape_out

        # 创建一个空数组 out，形状为 shape_out
        out = np.empty(shape_out)
        # 使用 np.percentile 计算数组 d 的百分位数，将结果存入 out 中，并保持维度
        result = np.percentile(d, q, axis=axis, keepdims=True, out=out)
        # 断言结果 result 是之前创建的 out 数组
        assert result is out
        # 断言 result 的形状与 shape_out 相同
        assert_equal(result.shape, shape_out)

    # 标记为跳过此测试函数，原因是在特定环境中版本兼容性问题导致失败
    @skip(reason="NP_VER: fails on CI; no method=")
    def test_out(self):
        # 创建一个全为0的一维数组 o
        o = np.zeros((4,))
        # 创建一个全为1的二维数组 d
        d = np.ones((3, 4))
        # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 0, 0, out=o), o)
        # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 0, 0, method="nearest", out=o), o)
        
        # 重新初始化 o，创建一个全为0的一维数组
        o = np.zeros((3,))
        # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 1, 1, out=o), o)
        # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 1, 1, method="nearest", out=o), o)
        
        # 重新初始化 o，创建一个全为0的零维数组
        o = np.zeros(())
        # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 2, out=o), o)
        # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
        assert_equal(np.percentile(d, 2, method="nearest", out=o), o)

    # 标记为跳过此测试函数，原因是在特定环境中版本兼容性问题导致失败
    @skip(reason="NP_VER: fails on CI; no method=")
    def test_out_nan(self):
        # 在警告上下文中运行以下代码块
        with warnings.catch_warnings(record=True):
            # 始终警告运行时警告
            warnings.filterwarnings("always", "", RuntimeWarning)
            # 创建一个全为0的一维数组 o
            o = np.zeros((4,))
            # 创建一个全为1的二维数组 d
            d = np.ones((3, 4))
            # 将第三行第二列的元素设为 NaN
            d[2, 1] = np.nan
            # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 0, 0, out=o), o)
            # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 0, 0, method="nearest", out=o), o)
            
            # 重新初始化 o，创建一个全为0的一维数组
            o = np.zeros((3,))
            # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 1, 1, out=o), o)
            # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 1, 1, method="nearest", out=o), o)
            
            # 重新初始化 o，创建一个全为0的零维数组
            o = np.zeros(())
            # 断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 1, out=o), o)
            # 使用 method="nearest" 参数，再次断言 np.percentile 返回的结果与预期的 o 数组相同，并将结果存入 o 中
            assert_equal(np.percentile(d, 1, method="nearest", out=o), o)

    # 标记为跳过此测试函数，原因是在特定环境中版本兼容性问题导致失败
    @skip(reason="NP_VER: fails on CI; no method=")
    @xpassIfTorchDynamo  # (reason="np.percentile undocumented nan weirdness")
    # 测试处理 NaN 的行为
    def test_nan_behavior(self):
        # 创建一个包含 24 个浮点数的 NumPy 数组
        a = np.arange(24, dtype=float)
        # 将索引为 2 的元素设置为 NaN
        a[2] = np.nan
        # 断言计算数组 a 中的百分位数为 NaN
        assert_equal(np.percentile(a, 0.3), np.nan)
        # 断言计算数组 a 中沿 axis=0 的百分位数为 NaN
        assert_equal(np.percentile(a, 0.3, axis=0), np.nan)
        # 断言计算数组 a 中沿 axis=0 的多个百分位数为包含 NaN 的数组
        assert_equal(np.percentile(a, [0.3, 0.6], axis=0), np.array([np.nan] * 2))

        # 调整数组形状为 (2, 3, 4)，并设置部分元素为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 没有指定 axis，断言计算数组 a 中的百分位数为 NaN
        assert_equal(np.percentile(a, 0.3), np.nan)
        # 断言计算数组 a 中的百分位数的维度为 0
        assert_equal(np.percentile(a, 0.3).ndim, 0)

        # 计算沿 axis=0 的百分位数，并将部分值设置为 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 0), b)

        # 计算沿 axis=0 的多个百分位数，并将部分值设置为 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), [0.3, 0.6], 0)
        b[:, 2, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 0), b)

        # 计算沿 axis=1 的百分位数，并将部分值设置为 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.percentile(a, 0.3, 1), b)

        # 计算沿 axis=1 的多个百分位数，并将部分值设置为 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), [0.3, 0.6], 1)
        b[:, 1, 3] = np.nan
        b[:, 1, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], 1), b)

        # 计算沿 axis=(0, 2) 的百分位数，并将部分值设置为 NaN
        b = np.percentile(np.arange(24, dtype=float).reshape(2, 3, 4), 0.3, (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.percentile(a, 0.3, (0, 2)), b)

        # 计算沿 axis=(0, 2) 的多个百分位数，并将部分值设置为 NaN
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4), [0.3, 0.6], (0, 2)
        )
        b[:, 1] = np.nan
        b[:, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], (0, 2)), b)

        # 使用 method='nearest' 计算沿 axis=(0, 2) 的多个百分位数，并将部分值设置为 NaN
        b = np.percentile(
            np.arange(24, dtype=float).reshape(2, 3, 4),
            [0.3, 0.6],
            (0, 2),
            method="nearest",
        )
        b[:, 1] = np.nan
        b[:, 2] = np.nan
        assert_equal(np.percentile(a, [0.3, 0.6], (0, 2), method="nearest"), b)

    # 测试处理包含 NaN 的情况
    def test_nan_q(self):
        # 使用 pytest 断言，验证传入包含 NaN 的列表时会抛出 RuntimeError 或 ValueError 异常
        with pytest.raises((RuntimeError, ValueError)):
            np.percentile([1, 2, 3, 4.0], np.nan)
        with pytest.raises((RuntimeError, ValueError)):
            np.percentile([1, 2, 3, 4.0], [np.nan])
        # 创建一个包含 NaN 的 NumPy 数组，并传入包含 NaN 的百分位数组，验证会抛出异常
        q = np.linspace(1.0, 99.0, 16)
        q[0] = np.nan
        with pytest.raises((RuntimeError, ValueError)):
            np.percentile([1, 2, 3, 4.0], q)
# 使用装饰器实例化参数化测试，该测试类针对 np.quantile 的各种情况进行测试
@instantiate_parametrized_tests
class TestQuantile(TestCase):
    # 大部分情况已经由 TestPercentile 测试过了

    # 跳过这个测试的原因是不需要关注 1 ULP（最小单位舍入误差）
    @skip(reason="do not chase 1ulp")
    def test_max_ulp(self):
        # 定义输入数组 x
        x = [0.0, 0.2, 0.4]
        # 计算 x 的第 0.45 分位数
        a = np.quantile(x, 0.45)
        # 使用默认的线性方法，结果应为 0 + 0.2 * (0.45/2) = 0.18。
        # 0.18 无法精确表示，这个公式会导致与期望结果相差 1 ULP。
        # 确保结果在 1 ULP 之内，参见 gh-20331
        np.testing.assert_array_max_ulp(a, 0.18, maxulp=1)

    # 基本的量化测试
    def test_basic(self):
        # 创建数组 x，包含 [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
        x = np.arange(8) * 0.5
        # 测试分位数为 0 时的结果应为 0.0
        assert_equal(np.quantile(x, 0), 0.0)
        # 测试分位数为 1 时的结果应为 3.5
        assert_equal(np.quantile(x, 1), 3.5)
        # 测试分位数为 0.5 时的结果应为 1.75
        assert_equal(np.quantile(x, 0.5), 1.75)

    # 由于整数或布尔数组量化的问题，此测试预期会失败
    @xfail  # (reason="quantile w/integers or bools")
    def test_correct_quantile_value(self):
        # 创建布尔数组 a = np.array([True])
        a = np.array([True])
        # 尝试量化 True 和 False，预期结果应为 a[0]，即 True
        tf_quant = np.quantile(True, False)
        assert_equal(tf_quant, a[0])
        assert_equal(type(tf_quant), a.dtype)
        # 创建布尔数组 a = np.array([False, True, True])
        a = np.array([False, True, True])
        # 尝试以 a 作为分位数数组来量化 a，预期结果应为 a 本身
        quant_res = np.quantile(a, a)
        assert_array_equal(quant_res, a)
        assert_equal(quant_res.dtype, a.dtype)

    # 支持分数数组的测试，但当前不支持分数数组，因此跳过
    @skip(reason="support arrays of Fractions?")
    def test_fraction(self):
        # 创建包含分数的数组 x
        x = [Fraction(i, 2) for i in range(8)]
        # 对 x 求分位数为 0，预期结果应为 0
        q = np.quantile(x, 0)
        assert_equal(q, 0)
        assert_equal(type(q), Fraction)

        # 对 x 求分位数为 1，预期结果应为 Fraction(7, 2)
        q = np.quantile(x, 1)
        assert_equal(q, Fraction(7, 2))
        assert_equal(type(q), Fraction)

        # 对 x 求分位数为 1/2，预期结果应为 Fraction(7, 4)
        q = np.quantile(x, Fraction(1, 2))
        assert_equal(q, Fraction(7, 4))
        assert_equal(type(q), Fraction)

        # 对 x 求分位数为 [1/2]，预期结果应为 np.array([Fraction(7, 4)])
        q = np.quantile(x, [Fraction(1, 2)])
        assert_equal(q, np.array([Fraction(7, 4)]))
        assert_equal(type(q), np.ndarray)

        # 对 x 求分位数为 [[1/2]]，预期结果应为 np.array([[Fraction(7, 4)]])
        q = np.quantile(x, [[Fraction(1, 2)]])
        assert_equal(q, np.array([[Fraction(7, 4)]]))
        assert_equal(type(q), np.ndarray)

        # 重复整数输入但分位数为分数的测试
        x = np.arange(8)
        assert_equal(np.quantile(x, Fraction(1, 2)), Fraction(7, 2))

    # 复数数组的测试，预期会引发 TypeError
    @skip(reason="does not raise in numpy?")
    def test_complex(self):
        # 见 gh-22652
        # 创建复数数组 arr_c
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="D")
        # 期望在这里引发 TypeError 异常
        assert_raises(TypeError, np.quantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="F")
        # 期望在这里引发 TypeError 异常
        assert_raises(TypeError, np.quantile, arr_c, 0.5)

    # 针对 NumPy 版本 1.21.2 的 CI 失败的测试，需要跳过
    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails with NumPy 1.21.2 on CI")
    def test_no_p_overwrite(self):
        # 这个测试值得重新测试，因为 quantile 不会创建副本
        # 创建 p0 和 p 数组
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        # 对 0 到 99 的浮点数数组进行 quantile 计算，不修改 p
        np.quantile(np.arange(100.0), p, method="midpoint")
        assert_array_equal(p, p0)

        p0 = p0.tolist()
        p = p.tolist()
        # 同样的测试，将 p 和 p0 转为列表再进行测试
        np.quantile(np.arange(100.0), p, method="midpoint")
        assert_array_equal(p, p0)
    # 跳过此测试用例，理由是希望保留整数数据类型的分位数计算结果
    @skip(reason="XXX: make quantile preserve integer dtypes")
    # 参数化测试，使用给定的整数数据类型进行测试
    @parametrize("dtype", "Bbhil")  # np.typecodes["AllInteger"])
    def test_quantile_preserve_int_type(self, dtype):
        # 使用指定的数据类型创建数组，并计算其最近方法（nearest method）的分位数
        res = np.quantile(np.array([1, 2], dtype=dtype), [0.5], method="nearest")
        # 断言计算得到的分位数结果的数据类型与输入的数据类型一致
        assert res.dtype == dtype

    # 如果 NumPy 版本小于 1.22，则跳过此测试，理由是在较早的版本上会失败
    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails with NumPy 1.21.2 on CI")
    # 参数化测试，测试不同的分位数计算方法
    @parametrize(
        "method",
        [
            subtest(
                "inverted_cdf",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "averaged_inverted_cdf",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "closest_observation",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "interpolated_inverted_cdf",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "hazen",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "weibull",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            "linear",
            subtest(
                "median_unbiased",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            subtest(
                "normal_unbiased",
                decorators=[
                    xpassIfTorchDynamo,
                ],
            ),
            "nearest",
            "lower",
            "higher",
            "midpoint",
        ],
    )
    def test_quantile_monotonic(self, method):
        # GH 14685
        # 测试分位数函数的返回值在有序情况下是否单调递增
        # 同时也测试边界值是否正确处理
        p0 = np.linspace(0, 1, 101)
        # 使用给定的数据集计算分位数，使用指定的方法
        quantile = np.quantile(
            np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7]) * 0.1,
            p0,
            method=method,
        )
        # 断言计算得到的分位数结果是否是有序的
        assert_equal(np.sort(quantile), quantile)

        # 进行另一组数据的测试，数据点数量明显可被分割
        quantile = np.quantile([0.0, 1.0, 2.0, 3.0], p0, method=method)
        # 断言计算得到的分位数结果是否是有序的
        assert_equal(np.sort(quantile), quantile)

    # 跳过此测试用例，理由是没有假设条件
    @skip(reason="no hypothesis")
    # 使用假设条件进行参数化测试，测试不同的浮点数组
    @hypothesis.given(
        arr=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=3, max_value=1000),
            elements=st.floats(
                allow_infinity=False, allow_nan=False, min_value=-1e300, max_value=1e300
            ),
        )
    )
    def test_quantile_monotonic_hypo(self, arr):
        # 创建一个从 0 到 1，步长为 0.01 的数组
        p0 = np.arange(0, 1, 0.01)
        # 使用假设的数组计算分位数
        quantile = np.quantile(arr, p0)
        # 断言计算得到的分位数结果是否是有序的
        assert_equal(np.sort(quantile), quantile)
    # 定义一个测试函数，用于测试处理包含 NaN 值的数组的分位数计算
    def test_quantile_scalar_nan(self):
        # 创建一个 NumPy 数组，包含两个子数组，每个子数组包含三个浮点数
        a = np.array([[10.0, 7.0, 4.0], [3.0, 2.0, 1.0]])
        # 将第一个子数组的第二个元素设置为 NaN（Not a Number）
        a[0][1] = np.nan
        # 计算数组 a 的中位数（50% 分位数），返回一个标量值
        actual = np.quantile(a, 0.5)
        # 使用 assert_equal 断言函数检查计算得到的中位数是否等于 NaN
        assert_equal(np.quantile(a, 0.5), np.nan)
@instantiate_parametrized_tests
class TestMedian(TestCase):
    # Median 测试类，用于测试 numpy 中的中位数计算功能

    def test_basic(self):
        # 基本测试函数
        a0 = np.array(1)  # 创建一个包含单个元素的 numpy 数组
        a1 = np.arange(2)  # 创建一个包含 [0, 1] 的 numpy 数组
        a2 = np.arange(6).reshape(2, 3)  # 创建一个 2x3 的 numpy 数组

        # 测试 np.median 对单个元素的计算
        assert_equal(np.median(a0), 1)
        # 测试 np.median 对数组 [0, 1] 的计算
        assert_allclose(np.median(a1), 0.5)
        # 测试 np.median 对数组 [[0, 1, 2], [3, 4, 5]] 的计算
        assert_allclose(np.median(a2), 2.5)
        # 测试 np.median 对数组 [[0, 1, 2], [3, 4, 5]] 沿 axis=0 的计算
        assert_allclose(np.median(a2, axis=0), [1.5, 2.5, 3.5])
        # 测试 np.median 对数组 [[0, 1, 2], [3, 4, 5]] 沿 axis=1 的计算
        assert_equal(np.median(a2, axis=1), [1, 4])
        # 测试 np.median 对数组 [[0, 1, 2], [3, 4, 5]] 的全局中位数计算
        assert_allclose(np.median(a2, axis=None), 2.5)

        # 额外的中位数计算测试
        a = np.array([0.0444502, 0.0463301, 0.141249, 0.0606775])
        assert_almost_equal((a[1] + a[3]) / 2.0, np.median(a))
        a = np.array([0.0463301, 0.0444502, 0.141249])
        assert_equal(a[0], np.median(a))
        a = np.array([0.0444502, 0.141249, 0.0463301])
        assert_equal(a[-1], np.median(a))

    @xfail  # (reason="median: scalar output vs 0-dim")
    def test_basic_2(self):
        # 第二个基本测试，测试包含 nan 值的数组的中位数计算
        a = np.array([0.0444502, 0.141249, 0.0463301])
        assert_equal(np.median(a).ndim, 0)
        a[1] = np.nan
        assert_equal(np.median(a).ndim, 0)

    def test_axis_keyword(self):
        # 测试 axis 关键字参数
        a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])

        # 对不同维度的数组进行中位数计算
        for a in [a3, np.random.randint(0, 100, size=(2, 3, 4))]:
            orig = a.copy()
            np.median(a, axis=None)  # 全局中位数计算
            for ax in range(a.ndim):
                np.median(a, axis=ax)  # 沿指定轴计算中位数
            assert_array_equal(a, orig)

        # 针对特定数组进行的中位数计算和断言
        assert_allclose(np.median(a3, axis=0), [3, 4])
        assert_allclose(np.median(a3.T, axis=1), [3, 4])
        assert_allclose(np.median(a3), 3.5)
        assert_allclose(np.median(a3, axis=None), 3.5)
        assert_allclose(np.median(a3.T), 3.5)
    # 定义测试函数，用于测试 numpy 中的 median 函数的不同用法，检验其覆盖输入的功能
    def test_overwrite_keyword(self):
        # 创建一个二维数组 a3
        a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])
        # 创建一个包含单个元素的数组 a0
        a0 = np.array(1)
        # 创建一个一维数组 a1
        a1 = np.arange(2)
        # 创建一个二维数组 a2
        a2 = np.arange(6).reshape(2, 3)
        # 断言：计算数组 a0 的中位数，覆盖输入为 True，应等于 1
        assert_allclose(np.median(a0.copy(), overwrite_input=True), 1)
        # 断言：计算数组 a1 的中位数，覆盖输入为 True，应等于 0.5
        assert_allclose(np.median(a1.copy(), overwrite_input=True), 0.5)
        # 断言：计算数组 a2 的中位数，覆盖输入为 True，应等于 2.5
        assert_allclose(np.median(a2.copy(), overwrite_input=True), 2.5)
        # 断言：计算数组 a2 沿 axis=0 的中位数，覆盖输入为 True，应等于 [1.5, 2.5, 3.5]
        assert_allclose(
            np.median(a2.copy(), overwrite_input=True, axis=0), [1.5, 2.5, 3.5]
        )
        # 断言：计算数组 a2 沿 axis=1 的中位数，覆盖输入为 True，应等于 [1, 4]
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=1), [1, 4])
        # 断言：计算数组 a2 的全局中位数，覆盖输入为 True，应等于 2.5
        assert_allclose(np.median(a2.copy(), overwrite_input=True, axis=None), 2.5)
        # 断言：计算数组 a3 沿 axis=0 的中位数，覆盖输入为 True，应等于 [3, 4]
        assert_allclose(np.median(a3.copy(), overwrite_input=True, axis=0), [3, 4])
        # 断言：计算数组 a3 的转置矩阵沿 axis=1 的中位数，覆盖输入为 True，应等于 [3, 4]
        assert_allclose(np.median(a3.T.copy(), overwrite_input=True, axis=1), [3, 4])

        # 创建一个三维数组 a4，包含 3*4*5 个元素
        a4 = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
        # 对数组 a4 进行随机打乱
        np.random.shuffle(a4.ravel())
        # 断言：计算数组 a4 的全局中位数，应等于通过复制的方式计算的全局中位数
        assert_allclose(
            np.median(a4, axis=None),
            np.median(a4.copy(), axis=None, overwrite_input=True),
        )
        # 断言：计算数组 a4 沿 axis=0 的中位数，应等于通过复制的方式计算的中位数
        assert_allclose(
            np.median(a4, axis=0), np.median(a4.copy(), axis=0, overwrite_input=True)
        )
        # 断言：计算数组 a4 沿 axis=1 的中位数，应等于通过复制的方式计算的中位数
        assert_allclose(
            np.median(a4, axis=1), np.median(a4.copy(), axis=1, overwrite_input=True)
        )
        # 断言：计算数组 a4 沿 axis=2 的中位数，应等于通过复制的方式计算的中位数
        assert_allclose(
            np.median(a4, axis=2), np.median(a4.copy(), axis=2, overwrite_input=True)
        )

    # 定义测试函数，用于测试 numpy 中 median 函数对于类数组的处理
    def test_array_like(self):
        # 创建列表 x
        x = [1, 2, 3]
        # 断言：计算列表 x 的中位数，应等于 2
        assert_almost_equal(np.median(x), 2)
        # 创建列表的列表 x2，其中包含列表 x
        x2 = [x]
        # 断言：计算列表 x2 的中位数，应等于 2
        assert_almost_equal(np.median(x2), 2)
        # 断言：计算列表 x2 沿 axis=0 的中位数，应等于列表 x
        assert_allclose(np.median(x2, axis=0), x)

    # 定义测试函数，用于测试 numpy 中 median 函数的 out 参数
    def test_out(self):
        # 创建全零数组 o，形状为 (4,)
        o = np.zeros((4,))
        # 创建全一数组 d，形状为 (3, 4)
        d = np.ones((3, 4))
        # 断言：计算数组 d 沿 axis=0 的中位数，并将结果存入数组 o，应与 o 相等
        assert_equal(np.median(d, 0, out=o), o)
        # 重新创建全零数组 o，形状为 (3,)
        o = np.zeros((3,))
        # 断言：计算数组 d 沿 axis=1 的中位数，并将结果存入数组 o，应与 o 相等
        assert_equal(np.median(d, 1, out=o), o)
        # 重新创建全零数组 o，形状为 ()
        o = np.zeros(())
        # 断言：计算数组 d 的全局中位数，并将结果存入数组 o，应与 o 相等
        assert_equal(np.median(d, out=o), o)

    # 定义测试函数，用于测试 numpy 中 median 函数的处理 NaN 值的情况
    def test_out_nan(self):
        # 捕获 RuntimeWarning 警告
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always", "", RuntimeWarning)
            # 创建全零数组 o，形状为 (4,)
            o = np.zeros((4,))
            # 创建全一数组 d，形状为 (3, 4)
            d = np.ones((3, 4))
            # 将数组 d 中的特定位置设为 NaN
            d[2, 1] = np.nan
            # 断言：计算数组 d 沿 axis=0 的中位数，并将结果存入数组 o，应与 o 相等
            assert_equal(np.median(d, 0, out=o), o)
            # 重新创建全零数组 o，形状为 (3,)
            o = np.zeros((3,))
            # 断言：计算数组 d 沿 axis=1 的中位数，并将结果存入数组 o，应与 o 相等
            assert_equal(np.median(d, 1, out=o), o)
            # 重新创建全零数组 o，形状为 ()
            o = np.zeros(())
            # 断言：计算数组 d 的全局中位数，并将结果存入数组 o，应与 o 相等
            assert_equal(np.median(d, out=o), o)
    # 定义测试函数，用于测试 np.median 的处理 NaN 的行为
    def test_nan_behavior(self):
        # 创建一个长度为24的浮点型数组 a，并将第二个元素设为 NaN
        a = np.arange(24, dtype=float)
        a[2] = np.nan
        # 断言对数组 a 计算中位数得到的结果是 NaN
        assert_equal(np.median(a), np.nan)
        # 断言在 axis=0 方向上计算中位数得到的结果是 NaN
        assert_equal(np.median(a, axis=0), np.nan)

        # 重新初始化数组 a，形状为 (2, 3, 4)，并将其中的两个元素设为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 没有指定 axis，断言对数组 a 计算中位数得到的结果是 NaN
        assert_equal(np.median(a), np.nan)
        #      assert_equal(np.median(a).ndim, 0)

        # 在 axis=0 方向上计算中位数，创建一个预期的中位数数组 b，并将其中两个元素设为 NaN
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        # 断言在 axis=0 方向上计算中位数得到的结果与预期的 b 相等
        assert_equal(np.median(a, 0), b)

        # 在 axis=1 方向上计算中位数，创建一个预期的中位数数组 b，并将其中两个元素设为 NaN
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        # 断言在 axis=1 方向上计算中位数得到的结果与预期的 b 相等
        assert_equal(np.median(a, 1), b)

    @xpassIfTorchDynamo  # (reason="median: does not support tuple axes")
    # 定义测试函数，测试在指定轴 (0, 2) 上计算中位数时的处理 NaN 的行为
    def test_nan_behavior_2(self):
        # 创建一个形状为 (2, 3, 4) 的浮点型数组 a，并将其中一个元素设为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 在 axis=(0, 2) 方向上计算中位数，创建一个预期的中位数数组 b，并将其中一个元素设为 NaN
        b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), (0, 2))
        b[1] = np.nan
        b[2] = np.nan
        # 断言在 axis=(0, 2) 方向上计算中位数得到的结果与预期的 b 相等
        assert_equal(np.median(a, (0, 2)), b)

    @xfail  # (reason="median: scalar vs 0-dim")
    # 定义测试函数，测试在没有指定轴时计算中位数的行为
    def test_nan_behavior_3(self):
        # 创建一个形状为 (2, 3, 4) 的浮点型数组 a，并将其中一个元素设为 NaN
        a = np.arange(24, dtype=float).reshape(2, 3, 4)
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # 没有指定轴，断言对数组 a 计算中位数的结果的维度是 0
        assert_equal(np.median(a).ndim, 0)

    @xpassIfTorchDynamo  # (reason="median: torch.quantile does not handle empty tensors")
    @skipif(IS_WASM, reason="fp errors don't work correctly")
    # 定义测试函数，测试对空数组计算中位数的行为
    def test_empty(self):
        # 创建一个空的浮点型数组 a
        a = np.array([], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", RuntimeWarning)
            # 断言对空数组 a 计算中位数得到的结果是 NaN
            assert_equal(np.median(a), np.nan)
            # 断言捕获到 RuntimeWarning 警告
            assert_(w[0].category is RuntimeWarning)
            # 断言捕获到两条警告
            assert_equal(len(w), 2)

        # 创建一个形状为 (1, 1, 0) 的浮点型数组 a
        a = np.array([], dtype=float, ndmin=3)
        # 没有指定轴，断言对空数组 a 计算中位数得到的结果是 NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", RuntimeWarning)
            assert_equal(np.median(a), np.nan)
            # 断言捕获到 RuntimeWarning 警告
            assert_(w[0].category is RuntimeWarning)

        # 在 axis=0 和 axis=1 方向上计算中位数，创建一个预期的中位数数组 b
        b = np.array([], dtype=float, ndmin=2)
        # 断言在 axis=0 方向上计算中位数得到的结果与预期的 b 相等
        assert_equal(np.median(a, axis=0), b)
        # 断言在 axis=1 方向上计算中位数得到的结果与预期的 b 相等
        assert_equal(np.median(a, axis=1), b)

        # 在 axis=2 方向上计算中位数，创建一个预期的中位数数组 b
        b = np.array(np.nan, dtype=float, ndmin=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", RuntimeWarning)
            # 断言在 axis=2 方向上计算中位数得到的结果与预期的 b 相等
            assert_equal(np.median(a, axis=2), b)
            # 断言捕获到 RuntimeWarning 警告
            assert_(w[0].category is RuntimeWarning)

    @xpassIfTorchDynamo  # (reason="median: tuple axes not implemented")
    # 定义一个测试方法，用于测试 NumPy 中 median 函数的扩展轴功能

    # 创建一个形状为 (71, 23) 的随机数组 o
    o = np.random.normal(size=(71, 23))
    # 使用 np.dstack 将 o 数组沿第三维叠加 10 次，形成 x
    x = np.dstack([o] * 10)
    # 断言：计算 x 在第 0 和 1 维度上的中位数，应等于 o 的中位数（转换为标量）
    assert_equal(np.median(x, axis=(0, 1)), np.median(o).item())
    # 使用 np.moveaxis 将 x 的最后一维移动到第一维
    x = np.moveaxis(x, -1, 0)
    # 断言：计算 x 在倒数第二和倒数第一维度上的中位数，应等于 o 的中位数（转换为标量）
    assert_equal(np.median(x, axis=(-2, -1)), np.median(o).item())
    # 使用 x.swapaxes(0, 1) 交换 x 的第一和第二维，并创建其副本
    x = x.swapaxes(0, 1).copy()
    # 断言：计算 x 在第 0 和倒数第一维度上的中位数，应等于 o 的中位数（转换为标量）
    assert_equal(np.median(x, axis=(0, -1)), np.median(o).item())

    # 断言：计算 x 在所有维度上的中位数，应等于计算 x 在没有指定轴的情况下的中位数
    assert_equal(np.median(x, axis=(0, 1, 2)), np.median(x, axis=None))
    # 断言：计算 x 在第 0 维度上的中位数，应等于计算 x 在第 0 维度上的中位数
    assert_equal(np.median(x, axis=(0,)), np.median(x, axis=0))
    # 断言：计算 x 在倒数第一维度上的中位数，应等于计算 x 在倒数第一维度上的中位数
    assert_equal(np.median(x, axis=(-1,)), np.median(x, axis=-1))

    # 创建一个形状为 (3, 5, 7, 11) 的数组 d，其中元素值为 0 到 3*5*7*11-1
    d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
    # 随机打乱数组 d 中的元素
    np.random.shuffle(d.ravel())
    # 断言：计算 d 在前三个维度上的中位数，并比较第一个值与 d 在第一个维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(0, 1, 2))[0], np.median(d[:, :, :, 0].flatten())
    )
    # 断言：计算 d 在前三个维度上的中位数，并比较第二个值与 d 在第二个维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(0, 1, 3))[1], np.median(d[:, :, 1, :].flatten())
    )
    # 断言：计算 d 在第四个和第二个维度上的中位数，并比较第三个值与 d 在第二个维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(3, 1, -4))[2], np.median(d[:, :, 2, :].flatten())
    )
    # 断言：计算 d 在第四个和第二个维度上的中位数，并比较第四个值与 d 在第一维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(3, 1, 2))[2], np.median(d[2, :, :, :].flatten())
    )
    # 断言：计算 d 在第四个和第三个维度上的中位数，并比较第五个值与 d 在第一维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(3, 2))[2, 1], np.median(d[2, 1, :, :].flatten())
    )
    # 断言：计算 d 在第二个和倒数第二个维度上的中位数，并比较第六个值与 d 在第二维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(1, -2))[2, 1], np.median(d[2, :, :, 1].flatten())
    )
    # 断言：计算 d 在第二个和第四个维度上的中位数，并比较第七个值与 d 在第二维度上某个平面的中位数
    assert_equal(
        np.median(d, axis=(1, 3))[2, 2], np.median(d[2, :, 2, :].flatten())
    )

# 定义一个测试方法，用于测试在使用无效轴参数时是否引发异常
def test_extended_axis_invalid():
    # 创建一个全为 1 的形状为 (3, 5, 7, 11) 的数组 d
    d = np.ones((3, 5, 7, 11))
    # 断言：使用一个超出范围的轴参数（-5），期望引发 np.AxisError 异常
    assert_raises(np.AxisError, np.median, d, axis=-5)
    # 断言：使用一个超出范围的轴参数（0 和 -5 的组合），期望引发 np.AxisError 异常
    assert_raises(np.AxisError, np.median, d, axis=(0, -5))
    # 断言：使用一个超出范围的轴参数（4），期望引发 np.AxisError 异常
    assert_raises(np.AxisError, np.median, d, axis=4)
    # 断言：使用一个超出范围的轴参数（0 和 4 的组合），期望引发 np.AxisError 异常
    assert_raises(np.AxisError, np.median, d, axis=(0, 4))
    # 断言：使用一个无效的轴参数（在同一轴上重复），期望引发 ValueError 异常
    assert_raises(ValueError, np.median, d, axis=(1, 1))

# 定义一个测试方法，用于测试在使用 keepdims 参数时的中位数函数行为
@xpassIfTorchDynamo  # (reason="median: tuple axis")
def test_keepdims_2():
    # 创建一个全为 1 的形状为 (3, 5, 7, 11) 的数组 d
    d = np.ones((3, 5, 7, 11))
    # 断言：计算 d 在第 0 和第 1 维度上的中位数，并保持维度，期望结果形状为 (1, 1, 7, 11)
    assert_equal(np.median(d, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11))
    # 断言：计算 d 在第 0 和第 3 维度上的中位数，并保持维度，期望结果形状为 (1, 5, 7, 1)
    assert_equal(np.median(d, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1))
    # 断言：计算 d 在第 1 维度上的中位数，并保持维度，期望结果形状为 (3, 1, 7, 11)
    assert_equal(np.median(d, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
    # 断言：计算 d 在所有维度上的中位数，并保持维度，期望结果形状为 (1, 1, 1, 1)
    assert_equal(np.median(d, axis=(0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1))
    # 断言
    @parametrize(
        "axis",
        [  # 参数化测试的参数列表开始
            None,  # axis 为 None 的情况
            1,  # axis 为整数 1 的情况
            subtest((1,)),  # 使用 subtest 函数创建的参数为 (1,) 的情况
            subtest(  # 使用 subtest 函数创建的参数为 (0, 1) 的情况
                (0, 1),
                decorators=[  # 使用修饰器列表开始
                    skip(reason="Tuple axes"),  # 使用 skip 修饰器，原因是 Tuple axes
                ],
            ),
            subtest(  # 使用 subtest 函数创建的参数为 (-3, -1) 的情况
                (-3, -1),
                decorators=[  # 使用修饰器列表开始
                    skip(reason="Tuple axes"),  # 使用 skip 修饰器，原因是 Tuple axes
                ],
            ),
        ],  # 参数化测试的参数列表结束
    )
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))  # 创建一个形状为 (3, 5, 7, 11) 的全为 1 的 numpy 数组 d
        if axis is None:  # 如果 axis 参数为 None
            shape_out = (1,) * d.ndim  # shape_out 为一个元组，每个维度的大小为 1
        else:  # 如果 axis 参数不为 None
            axis_norm = normalize_axis_tuple(axis, d.ndim)  # 调用 normalize_axis_tuple 函数，得到标准化后的 axis 元组
            shape_out = tuple(  # 根据 axis_norm 和 d 的维度创建 shape_out 元组
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim)
            )
        out = np.empty(shape_out)  # 创建一个形状为 shape_out 的空 numpy 数组 out
        result = np.median(d, axis=axis, keepdims=True, out=out)  # 计算 d 的中位数，将结果存储在 out 中
        assert result is out  # 断言 result 和 out 是同一个对象
        assert_equal(result.shape, shape_out)  # 断言 result 的形状与 shape_out 相等
@xpassIfTorchDynamo  # (reason="TODO: implement")
@instantiate_parametrized_tests
class TestSortComplex(TestCase):
    @parametrize(
        "type_in, type_out",
        [
            ("l", "D"),
            ("h", "F"),
            ("H", "F"),
            ("b", "F"),
            ("B", "F"),
            ("g", "G"),
        ],
    )
    def test_sort_real(self, type_in, type_out):
        # sort_complex() type casting for real input types
        # 创建一个包含整数数组的 NumPy 数组，根据 type_in 指定的类型进行数据类型转换
        a = np.array([5, 3, 6, 2, 1], dtype=type_in)
        # 调用 np.sort_complex() 对复数进行排序，返回复数数组
        actual = np.sort_complex(a)
        # 调用 np.sort() 对实数数组进行排序，并将结果转换为 type_out 指定的数据类型
        expected = np.sort(a).astype(type_out)
        # 断言实际输出和预期输出是否相等
        assert_equal(actual, expected)
        # 断言实际输出的数据类型和预期输出的数据类型是否相等
        assert_equal(actual.dtype, expected.dtype)

    def test_sort_complex(self):
        # sort_complex() handling of complex input
        # 创建一个包含复数的 NumPy 数组，指定数据类型为双精度复数 "D"
        a = np.array([2 + 3j, 1 - 2j, 1 - 3j, 2 + 1j], dtype="D")
        # 预期的复数数组，按照指定顺序排列
        expected = np.array([1 - 3j, 1 - 2j, 2 + 1j, 2 + 3j], dtype="D")
        # 调用 np.sort_complex() 对复数数组进行排序
        actual = np.sort_complex(a)
        # 断言实际输出和预期输出是否相等
        assert_equal(actual, expected)
        # 断言实际输出的数据类型和预期输出的数据类型是否相等
        assert_equal(actual.dtype, expected.dtype)


if __name__ == "__main__":
    # 运行所有测试用例
    run_tests()
```