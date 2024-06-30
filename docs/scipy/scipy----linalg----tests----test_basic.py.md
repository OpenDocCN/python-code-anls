# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_basic.py`

```
# 导入必要的模块和库
import itertools  # 导入 itertools 模块，用于高效迭代工具
import warnings  # 导入 warnings 模块，用于处理警告

import numpy as np  # 导入 NumPy 库并重命名为 np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,  # 导入 NumPy 中的多个函数和类
                   float32)
from numpy.random import random  # 导入 NumPy 中 random 模块中的 random 函数

from numpy.testing import (assert_equal, assert_almost_equal, assert_,  # 导入 NumPy 测试模块中的多个断言函数
                           assert_array_almost_equal, assert_allclose,
                           assert_array_equal, suppress_warnings)
import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数并重命名为 assert_raises

from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,  # 导入 SciPy 线性代数模块中的多个函数和类
                          solve_banded, solveh_banded, solve_triangular,
                          solve_circulant, circulant, LinAlgError, block_diag,
                          matrix_balance, qr, LinAlgWarning)

from scipy.linalg._testutils import assert_no_overwrite  # 导入 SciPy 线性代数测试工具模块中的 assert_no_overwrite 函数
from scipy._lib._testutils import check_free_memory, IS_MUSL  # 导入 SciPy 内部测试工具模块中的函数和变量
from scipy.linalg.blas import HAS_ILP64  # 导入 SciPy 线性代数模块中的 HAS_ILP64 变量

REAL_DTYPES = (np.float32, np.float64, np.longdouble)  # 定义包含实数类型的元组
COMPLEX_DTYPES = (np.complex64, np.complex128, np.clongdouble)  # 定义包含复数类型的元组
DTYPES = REAL_DTYPES + COMPLEX_DTYPES  # 合并实数和复数类型的元组

def _eps_cast(dtyp):
    """获取 dtype 的机器精度，并可能将其降级为 BLAS 类型。"""
    dt = dtyp
    if dt == np.longdouble:  # 如果 dtype 是 np.longdouble，将其替换为 np.float64
        dt = np.float64
    elif dt == np.clongdouble:  # 如果 dtype 是 np.clongdouble，将其替换为 np.complex128
        dt = np.complex128
    return np.finfo(dt).eps  # 返回指定 dtype 的机器精度

class TestSolveBanded:

    def test_real(self):
        """测试解实系数带状线性方程组的方法。"""
        a = array([[1.0, 20, 0, 0],  # 创建实系数系数矩阵 a
                   [-30, 4, 6, 0],
                   [2, 1, 20, 2],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2],  # 创建带状矩阵 ab
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1  # 设置下带宽 l 和上带宽 u
        b4 = array([10.0, 0.0, 2.0, 14.0])  # 创建测试向量 b4
        b4by1 = b4.reshape(-1, 1)  # 将 b4 转换为列向量
        b4by2 = array([[2, 1],  # 创建其他形状的测试向量
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],  # 创建其他形状的测试向量
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:  # 遍历所有的测试向量
            x = solve_banded((l, u), ab, b)  # 使用 solve_banded 解带状线性方程组
            assert_array_almost_equal(dot(a, x), b)  # 断言解 x 满足方程组的精度要求

    def test_complex(self):
        """测试解复系数带状线性方程组的方法。"""
        a = array([[1.0, 20, 0, 0],  # 创建复系数系数矩阵 a
                   [-30, 4, 6, 0],
                   [2j, 1, 20, 2j],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2j],  # 创建带状矩阵 ab
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2j, -1, 0, 0]])
        l, u = 2, 1  # 设置下带宽 l 和上带宽 u
        b4 = array([10.0, 0.0, 2.0, 14.0j])  # 创建测试向量 b4
        b4by1 = b4.reshape(-1, 1)  # 将 b4 转换为列向量
        b4by2 = array([[2, 1],  # 创建其他形状的测试向量
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],  # 创建其他形状的测试向量
                       [0, 0, 0, 1j],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:  # 遍历所有的测试向量
            x = solve_banded((l, u), ab, b)  # 使用 solve_banded 解带状线性方程组
            assert_array_almost_equal(dot(a, x), b)  # 断言解 x 满足方程组的精度要求
    # 定义测试函数，用于测试解三对角线性方程组的实数情况
    def test_tridiag_real(self):
        # 定义三对角矩阵 ab
        ab = array([[0.0, 20, 6, 2],
                   [1, 4, 20, 14],
                   [-30, 1, 7, 0]])
        # 根据 ab 构造三对角矩阵 a
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(
                                                                ab[2, :-1], -1)
        # 定义不同形式的向量 b4
        b4 = array([10.0, 0.0, 2.0, 14.0])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        # 遍历不同的 b，解三对角线性方程组并断言解的正确性
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    # 定义测试函数，用于测试解三对角线性方程组的复数情况
    def test_tridiag_complex(self):
        # 定义复数三对角矩阵 ab
        ab = array([[0.0, 20, 6, 2j],
                   [1, 4, 20, 14],
                   [-30, 1, 7, 0]])
        # 根据 ab 构造三对角矩阵 a
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(
                                                               ab[2, :-1], -1)
        # 定义不同形式的向量 b4
        b4 = array([10.0, 0.0, 2.0, 14.0j])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        # 遍历不同的 b，解三对角线性方程组并断言解的正确性
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    # 定义测试函数，用于测试 solve_banded 函数在 check_finite=False 时的行为
    def test_check_finite(self):
        # 定义测试所需的矩阵 ab 和向量 b4
        a = array([[1.0, 20, 0, 0],
                   [-30, 4, 6, 0],
                   [2, 1, 20, 2],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1
        b4 = array([10.0, 0.0, 2.0, 14.0])
        # 使用 solve_banded 解三对角线性方程组并断言解的正确性，忽略有限性检查
        x = solve_banded((l, u), ab, b4, check_finite=False)
        assert_array_almost_equal(dot(a, x), b4)

    # 定义测试函数，用于测试 solve_banded 函数在输入数据形状不合法时抛出异常的情况
    def test_bad_shape(self):
        # 定义测试所需的矩阵 ab 和不合法的向量 bad
        ab = array([[0.0, 20, 6, 2],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1
        bad = array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 4)
        # 断言 solve_banded 在输入不合法数据时抛出 ValueError 异常
        assert_raises(ValueError, solve_banded, (l, u), ab, bad)
        assert_raises(ValueError, solve_banded, (l, u), ab, [1.0, 2.0])

        # 断言 (l,u) 的值与 ab 不兼容时，solve_banded 抛出 ValueError 异常
        assert_raises(ValueError, solve_banded, (1, 1), ab, [1.0, 2.0])

    # 定义测试函数，用于测试解 1x1 矩阵的情况
    def test_1x1(self):
        # 定义 1x1 矩阵 b 和其对应的三对角矩阵
        b = array([[1., 2., 3.]])
        x = solve_banded((1, 1), [[0], [2], [0]], b)
        # 断言解 x 的结果与预期值相等
        assert_array_equal(x, [[0.5, 1.0, 1.5]])
        # 断言解 x 的数据类型为 float64
        assert_equal(x.dtype, np.dtype('f8'))
        # 断言矩阵 b 的值不变
        assert_array_equal(b, [[1.0, 2.0, 3.0]])
    # 定义一个测试方法，用于测试带有原生列表作为参数的情况
    def test_native_list_arguments(self):
        # 定义一个二维原生列表 a，包含四个子列表，每个子列表有四个元素
        a = [[1.0, 20, 0, 0],
             [-30, 4, 6, 0],
             [2, 1, 20, 2],
             [0, -1, 7, 14]]
        # 定义一个二维原生列表 ab，包含四个子列表，每个子列表有四个元素
        ab = [[0.0, 20, 6, 2],
              [1, 4, 20, 14],
              [-30, 1, 7, 0],
              [2, -1, 0, 0]]
        # 定义变量 l 和 u，分别赋值为 2 和 1
        l, u = 2, 1
        # 定义一个列表 b，包含四个浮点数元素
        b = [10.0, 0.0, 2.0, 14.0]
        # 调用 solve_banded 函数，传入参数 (l, u)，ab 和 b，计算得到 x
        x = solve_banded((l, u), ab, b)
        # 断言 dot(a, x) 函数的返回值与 b 在数值上近似相等
        assert_array_almost_equal(dot(a, x), b)

    # 使用 pytest 的参数化功能定义一个测试方法，测试 ab 和 b 均为空的情况
    @pytest.mark.parametrize('dt_ab', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt_ab, dt_b):
        # 创建一个空的 NumPy 数组 ab，数据类型为 dt_ab，包含一个空的子数组
        ab = np.array([[]], dtype=dt_ab)
        # 创建一个空的 NumPy 数组 b，数据类型为 dt_b
        b = np.array([], dtype=dt_b)
        # 调用 solve_banded 函数，传入参数 (0, 0)，ab 和 b，计算得到 x
        x = solve_banded((0, 0), ab, b)
        
        # 断言 x 的形状为 (0,)，即空数组
        assert x.shape == (0,)
        # 断言 x 的数据类型与 solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)) 函数返回值的数据类型相同
        assert x.dtype == solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)).dtype
        
        # 创建一个空的 NumPy 数组 b，形状为 (0, 0)，数据类型为 dt_b
        b = np.empty((0, 0), dtype=dt_b)
        # 再次调用 solve_banded 函数，传入参数 (0, 0)，ab 和 b，计算得到 x
        x = solve_banded((0, 0), ab, b)
        
        # 断言 x 的形状为 (0, 0)，即空的二维数组
        assert x.shape == (0, 0)
        # 断言 x 的数据类型与 solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)) 函数返回值的数据类型相同
        assert x.dtype == solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)).dtype
class TestSolveHBanded:

    def test_01_upper(self):
        # 解决上三角带状线性方程组
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        # 将右手边作为一维数组传入。
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_upper(self):
        # 解决上三角带状线性方程组
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_03_upper(self):
        # 解决上三角带状线性方程组
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        # 将右手边作为形状为 (3,1) 的二维数组传入。
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0]).reshape(-1, 1)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, array([0., 1., 0., 0.]).reshape(-1, 1))

    def test_01_lower(self):
        # 解决下三角带状线性方程组
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        #
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, -99],
                    [2.0, 2.0, 0.0, 0.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b, lower=True)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_lower(self):
        # 解决下三角带状线性方程组
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, -99],
                    [2.0, 2.0, 0.0, 0.0]])
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]])
        x = solveh_banded(ab, b, lower=True)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)
    def test_01_float32(self):
        # 解决以下带有带宽矩阵的线性方程组：
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0, 2.0], dtype=float32)
        # 调用 solveh_banded 函数求解线性方程组
        x = solveh_banded(ab, b)
        # 断言求解结果与预期结果的近似性
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_float32(self):
        # 解决以下带有带宽矩阵的线性方程组：
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]], dtype=float32)
        # 调用 solveh_banded 函数求解线性方程组
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        # 断言求解结果与预期结果的近似性
        assert_array_almost_equal(x, expected)

    def test_01_complex(self):
        # 解决以下带有带宽矩阵的线性方程组：
        # [ 4 -j  2  0]     [2-j]
        # [ j  4 -j  2] X = [4-j]
        # [ 2  j  4 -j]     [4+j]
        # [ 0  2  j  4]     [2+j]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, -1.0j, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([2-1.0j, 4.0-1j, 4+1j, 2+1j])
        # 调用 solveh_banded 函数求解线性方程组
        x = solveh_banded(ab, b)
        # 断言求解结果与预期结果的近似性
        assert_array_almost_equal(x, [0.0, 1.0, 1.0, 0.0])

    def test_02_complex(self):
        # 解决以下带有带宽矩阵的线性方程组：
        # [ 4 -j  2  0]     [2-j 2+4j]
        # [ j  4 -j  2] X = [4-j -1-j]
        # [ 2  j  4 -j]     [4+j 4+2j]
        # [ 0  2  j  4]     [2+j j]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, -1.0j, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([[2-1j, 2+4j],
                   [4.0-1j, -1-1j],
                   [4.0+1j, 4+2j],
                   [2+1j, 1j]])
        # 调用 solveh_banded 函数求解线性方程组
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0j],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 0.0]])
        # 断言求解结果与预期结果的近似性
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_upper(self):
        # 解决以下带有带宽矩阵的线性方程组：
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # 使用一维数组作为右侧向量。
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        # 调用 solveh_banded 函数求解线性方程组
        x = solveh_banded(ab, b)
        # 断言求解结果与预期结果的近似性
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])
    def test_tridiag_02_upper(self):
        # 解决上三角形的三对角线性方程组：
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        
        # 定义带宽矩阵 ab 和右侧向量 b
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]])
        
        # 调用 solveh_banded 函数求解方程组
        x = solveh_banded(ab, b)
        
        # 期望的解
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        
        # 断言解 x 与期望值 expected 几乎相等
        assert_array_almost_equal(x, expected)

    def test_tridiag_03_upper(self):
        # 解决上三角形的三对角线性方程组：
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # 其中右侧向量作为形状为 (3,1) 的二维数组
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0]).reshape(-1, 1)
        
        # 调用 solveh_banded 函数求解方程组
        x = solveh_banded(ab, b)
        
        # 断言解 x 与期望的数组几乎相等
        assert_array_almost_equal(x, array([0.0, 1.0, 0.0]).reshape(-1, 1))

    def test_tridiag_01_lower(self):
        # 解决下三角形的三对角线性方程组：
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        #
        ab = array([[4.0, 4.0, 4.0],
                    [1.0, 1.0, -99]])
        b = array([1.0, 4.0, 1.0])
        
        # 调用 solveh_banded 函数求解方程组，lower=True 表示解下三角形方程
        x = solveh_banded(ab, b, lower=True)
        
        # 断言解 x 与期望值几乎相等
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_lower(self):
        # 解决下三角形的三对角线性方程组：
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        #
        ab = array([[4.0, 4.0, 4.0],
                    [1.0, 1.0, -99]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]])
        
        # 调用 solveh_banded 函数求解方程组，lower=True 表示解下三角形方程
        x = solveh_banded(ab, b, lower=True)
        
        # 期望的解
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        
        # 断言解 x 与期望值 expected 几乎相等
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_float32(self):
        # 使用 float32 类型解决下三角形的三对角线性方程组：
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        #
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0], dtype=float32)
        
        # 调用 solveh_banded 函数求解方程组
        x = solveh_banded(ab, b)
        
        # 断言解 x 与期望值几乎相等
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_float32(self):
        # 使用 float32 类型解决上三角形的三对角线性方程组：
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        #
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]], dtype=float32)
        
        # 调用 solveh_banded 函数求解方程组
        x = solveh_banded(ab, b)
        
        # 期望的解
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        
        # 断言解 x 与期望值 expected 几乎相等
        assert_array_almost_equal(x, expected)
    def test_tridiag_01_complex(self):
        # 解三对角线性方程组:
        # [ 4 -j 0]     [ -j]
        # [ j 4 -j] X = [4-j]
        # [ 0 j  4]     [4+j]
        #
        ab = array([[-99, -1.0j, -1.0j], [4.0, 4.0, 4.0]])
        b = array([-1.0j, 4.0-1j, 4+1j])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 1.0])

    def test_tridiag_02_complex(self):
        # 解三对角线性方程组:
        # [ 4 -j 0]     [ -j    4j]
        # [ j 4 -j] X = [4-j  -1-j]
        # [ 0 j  4]     [4+j   4  ]
        #
        ab = array([[-99, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0]])
        b = array([[-1j, 4.0j],
                   [4.0-1j, -1.0-1j],
                   [4.0+1j, 4.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0j],
                          [1.0, 0.0],
                          [1.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_check_finite(self):
        # 解三对角线性方程组，禁用有限性检查：
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # 将右侧向量作为一维数组传入。
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b, check_finite=False)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_bad_shapes(self):
        # 检查当输入形状不合法时，是否能正确抛出 ValueError 异常。
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0]])
        assert_raises(ValueError, solveh_banded, ab, b)
        assert_raises(ValueError, solveh_banded, ab, [1.0, 2.0])
        assert_raises(ValueError, solveh_banded, ab, [1.0])

    def test_1x1(self):
        # 解 1x1 的三对角线性方程组：
        x = solveh_banded([[1]], [[1, 2, 3]])
        assert_array_equal(x, [[1.0, 2.0, 3.0]])
        assert_equal(x.dtype, np.dtype('f8'))

    def test_native_list_arguments(self):
        # 使用 Python 原生列表传递参数，与 test_01_upper 功能相同。
        ab = [[0.0, 0.0, 2.0, 2.0],
              [-99, 1.0, 1.0, 1.0],
              [4.0, 4.0, 4.0, 4.0]]
        b = [1.0, 4.0, 1.0, 2.0]
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    @pytest.mark.parametrize('dt_ab', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt_ab, dt_b):
        # ab 包含一个空行，对应于对角线
        ab = np.array([[]], dtype=dt_ab)
        b = np.array([], dtype=dt_b)
        x = solveh_banded(ab, b)

        assert x.shape == (0,)
        assert x.dtype == solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)).dtype

        b = np.empty((0, 0), dtype=dt_b)
        x = solveh_banded(ab, b)

        assert x.shape == (0, 0)
        assert x.dtype == solve(np.eye(1, dtype=dt_ab), np.ones(1, dtype=dt_b)).dtype
class TestSolve:
    # 设置方法：初始化测试环境，使随机数生成可预测
    def setup_method(self):
        np.random.seed(1234)

    # 测试用例：测试特定日期的错误
    def test_20Feb04_bug(self):
        # 定义矩阵 a
        a = [[1, 1], [1.0, 0]]  # ok
        # 求解方程 solve(a, [1, 0j])，并验证结果精度
        x0 = solve(a, [1, 0j])
        assert_array_almost_equal(dot(a, x0), [1, 0])

        # 测试失败案例：使用 clapack.zgesv(..,rowmajor=0) 失败
        a = [[1, 1], [1.2, 0]]
        b = [1, 0j]
        # 求解方程 solve(a, b)，并验证结果精度
        x0 = solve(a, b)
        assert_array_almost_equal(dot(a, x0), [1, 0])

    # 测试用例：简单情况下的测试
    def test_simple(self):
        # 定义矩阵 a
        a = [[1, 20], [-30, 4]]
        # 循环遍历不同的 b 值
        for b in ([[1, 0], [0, 1]],
                  [1, 0],
                  [[2, 1], [-30, 4]]
                  ):
            # 求解方程 solve(a, b)，并验证结果精度
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    # 测试用例：复数情况下的简单测试
    def test_simple_complex(self):
        # 定义复数矩阵 a
        a = array([[5, 2], [2j, 4]], 'D')
        # 循环遍历不同的 b 值
        for b in ([1j, 0],
                  [[1j, 1j], [0, 2]],
                  [1, 0j],
                  array([1, 0], 'D'),
                  ):
            # 求解复数方程 solve(a, b)，并验证结果精度
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    # 测试用例：正定矩阵情况下的简单测试
    def test_simple_pos(self):
        # 定义正定矩阵 a
        a = [[2, 3], [3, 5]]
        # 循环遍历不同的 b 值
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0]
                      ):
                # 求解方程 solve(a, b)，并验证结果精度，假设 a 为正定矩阵
                x = solve(a, b, assume_a='pos', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    # 测试用例：正定对称矩阵情况下的复数测试
    def test_simple_pos_complexb(self):
        # 定义正定矩阵 a
        a = [[5, 2], [2, 4]]
        # 循环遍历不同的 b 值
        for b in ([1j, 0],
                  [[1j, 1j], [0, 2]],
                  ):
            # 求解方程 solve(a, b)，并验证结果精度，假设 a 为正定矩阵
            x = solve(a, b, assume_a='pos')
            assert_array_almost_equal(dot(a, x), b)

    # 测试用例：对称矩阵情况下的简单测试
    def test_simple_sym(self):
        # 定义对称矩阵 a
        a = [[2, 3], [3, -5]]
        # 循环遍历不同的 b 值
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0]
                      ):
                # 求解方程 solve(a, b)，并验证结果精度，假设 a 为对称矩阵
                x = solve(a, b, assume_a='sym', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    # 测试用例：对称矩阵情况下的复数测试
    def test_simple_sym_complexb(self):
        # 定义对称矩阵 a
        a = [[5, 2], [2, -4]]
        # 循环遍历不同的 b 值
        for b in ([1j, 0],
                  [[1j, 1j], [0, 2]]
                  ):
            # 求解方程 solve(a, b)，并验证结果精度，假设 a 为对称矩阵
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    # 测试用例：对称矩阵情况下的复数测试
    def test_simple_sym_complex(self):
        # 定义对称矩阵 a，包含复数
        a = [[5, 2+1j], [2+1j, -4]]
        # 循环遍历不同的 b 值
        for b in ([1j, 0],
                  [1, 0],
                  [[1j, 1j], [0, 2]]
                  ):
            # 求解方程 solve(a, b)，并验证结果精度，假设 a 为对称矩阵
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    # 测试用例：Hermitian 实际上对称的矩阵情况下的简单测试
    def test_simple_her_actuallysym(self):
        # 定义 Hermitian 实际上对称的矩阵 a
        a = [[2, 3], [3, -5]]
        # 循环遍历不同的 b 值
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0],
                      [1j, 0],
                      ):
                # 求解方程 solve(a, b)，并验证结果精度，假设 a 为 Hermitian 矩阵
                x = solve(a, b, assume_a='her', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    # 测试用例：Hermitian 矩阵情况下的简单测试
    def test_simple_her(self):
        # 定义 Hermitian 矩阵 a
        a = [[5, 2+1j], [2-1j, -4]]
        # 循环遍历不同的 b 值
        for b in ([1j, 0],
                  [1, 0],
                  [[1j, 1j], [0, 2]]
                  ):
            # 求解方程 solve(a, b)，并验证结果精度，假设 a 为 Hermitian 矩阵
            x = solve(a, b, assume_a='her')
            assert_array_almost_equal(dot(a, x), b)
    # 定义一个测试函数，用于验证解决具有随机复数矩阵的线性方程组的正确性
    def test_nils_20Feb04(self):
        # 设定矩阵的大小
        n = 2
        # 创建一个随机复数矩阵 A
        A = random([n, n])+random([n, n])*1j
        # 初始化一个大小为 n x n 的双精度零矩阵 X
        X = zeros((n, n), 'D')
        # 计算 A 的逆矩阵 Ainv
        Ainv = inv(A)
        # 初始化一个大小为 n x n 的复数单位矩阵 R
        R = identity(n)+identity(n)*0j
        # 对于 R 的每一列进行迭代
        for i in arange(0, n):
            # 取出 R 的第 i 列
            r = R[:, i]
            # 解决方程 A * X[:, i] = r，并将解存储在 X 的第 i 列中
            X[:, i] = solve(A, r)
        # 断言 X 与 Ainv 的值近似相等
        assert_array_almost_equal(X, Ainv)

    # 定义一个测试函数，用于验证解决具有随机矩阵的线性方程组的正确性
    def test_random(self):
        # 设定矩阵的大小
        n = 20
        # 创建一个随机矩阵 a
        a = random([n, n])
        # 对角线元素加上一个偏移量
        for i in range(n):
            a[i, i] = 20*(.1+a[i, i])
        # 进行 4 次迭代
        for i in range(4):
            # 创建一个随机矩阵 b
            b = random([n, 3])
            # 解决方程 a * x = b，并检查解的精度
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    # 定义一个测试函数，用于验证解决具有随机复数矩阵的线性方程组的正确性
    def test_random_complex(self):
        # 设定矩阵的大小
        n = 20
        # 创建一个随机复数矩阵 a
        a = random([n, n]) + 1j * random([n, n])
        # 对角线元素加上一个偏移量
        for i in range(n):
            a[i, i] = 20*(.1+a[i, i])
        # 进行 2 次迭代
        for i in range(2):
            # 创建一个随机矩阵 b
            b = random([n, 3])
            # 解决方程 a * x = b，并检查解的精度
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    # 定义一个测试函数，用于验证解决具有对称随机矩阵的线性方程组的正确性
    def test_random_sym(self):
        # 设定矩阵的大小
        n = 20
        # 创建一个随机矩阵 a
        a = random([n, n])
        # 对角线元素加上一个偏移量，并确保对称性
        for i in range(n):
            a[i, i] = abs(20*(.1+a[i, i]))
            for j in range(i):
                a[i, j] = a[j, i]
        # 进行 4 次迭代
        for i in range(4):
            # 创建一个随机向量 b
            b = random([n])
            # 解决方程 a * x = b，假设矩阵 a 为正定
            x = solve(a, b, assume_a="pos")
            assert_array_almost_equal(dot(a, x), b)

    # 定义一个测试函数，用于验证解决具有对称随机复数矩阵的线性方程组的正确性
    def test_random_sym_complex(self):
        # 设定矩阵的大小
        n = 20
        # 创建一个随机复数矩阵 a
        a = random([n, n])
        a = a + 1j*random([n, n])
        # 对角线元素加上一个偏移量，并确保对称性
        for i in range(n):
            a[i, i] = abs(20*(.1+a[i, i]))
            for j in range(i):
                a[i, j] = conjugate(a[j, i])
        # 创建一个随机复数向量 b
        b = random([n])+2j*random([n])
        # 进行 2 次迭代
        for i in range(2):
            # 解决方程 a * x = b，假设矩阵 a 为正定
            x = solve(a, b, assume_a="pos")
            assert_array_almost_equal(dot(a, x), b)

    # 定义一个测试函数，用于验证解决具有给定矩阵和向量的线性方程组的正确性
    def test_check_finite(self):
        # 给定一个特定矩阵 a 和一组测试向量 b
        a = [[1, 20], [-30, 4]]
        # 对每个测试向量 b 进行迭代
        for b in ([[1, 0], [0, 1]], [1, 0],
                  [[2, 1], [-30, 4]]):
            # 解决方程 a * x = b，允许非有限值
            x = solve(a, b, check_finite=False)
            assert_array_almost_equal(dot(a, x), b)

    # 定义一个测试函数，用于验证解决具有标量 a 和一维向量 b 的线性方程组的正确性
    def test_scalar_a_and_1D_b(self):
        # 设定标量 a 和一维向量 b
        a = 1
        b = [1, 2, 3]
        # 解决方程 a * x = b
        x = solve(a, b)
        # 断言解 x 的展平值与向量 b 的值近似相等
        assert_array_almost_equal(x.ravel(), b)
        # 断言解 x 的形状为 (3,)，验证标量 a 和一维向量 b 的情况
        assert_(x.shape == (3,), 'Scalar_a_1D_b test returned wrong shape')

    # 定义一个测试函数，用于验证解决具有给定矩阵和向量的线性方程组的正确性
    def test_simple2(self):
        # 给定特定的矩阵 a 和向量 b
        a = np.array([[1.80, 2.88, 2.05, -0.89],
                      [525.00, -295.00, -95.00, -380.00],
                      [1.58, -2.69, -2.90, -1.04],
                      [-1.11, -0.66, -0.59, 0.80]])

        b = np.array([[9.52, 18.47],
                      [2435.00, 225.00],
                      [0.77, -13.28],
                      [-6.22, -6.21]])

        # 解决方程 a * x = b，并检查解的精度
        x = solve(a, b)
        # 断言解 x 与给定值的近似相等
        assert_array_almost_equal(x, np.array([[1., -1, 3, -5],
                                               [3, 2, 4, 1]]).T)
    def test_simple_complex2(self):
        # 创建复数类型的二维数组 a，包含复数元素
        a = np.array([[-1.34+2.55j, 0.28+3.17j, -6.39-2.20j, 0.72-0.92j],
                      [-1.70-14.10j, 33.10-1.50j, -1.50+13.40j, 12.90+13.80j],
                      [-3.29-2.39j, -1.91+4.42j, -0.14-1.35j, 1.72+1.35j],
                      [2.41+0.39j, -0.56+1.47j, -0.83-0.69j, -1.96+0.67j]])

        # 创建复数类型的二维数组 b，包含复数元素
        b = np.array([[26.26+51.78j, 31.32-6.70j],
                      [64.30-86.80j, 158.60-14.20j],
                      [-5.75+25.31j, -2.15+30.19j],
                      [1.16+2.57j, -2.56+7.55j]])

        # 调用 solve 函数求解方程组，返回复数类型的解 x，并进行精确度比较
        x = solve(a, b)
        assert_array_almost_equal(x, np. array([[1+1.j, -1-2.j],
                                                [2-3.j, 5+1.j],
                                                [-4-5.j, -3+4.j],
                                                [6.j, 2-3.j]]))

    def test_hermitian(self):
        # 创建 Hermitian 矩阵 a，使用上三角矩阵来确保 Hermitian 性质
        a = np.array([[-1.84, 0.11-0.11j, -1.78-1.18j, 3.91-1.50j],
                      [0, -4.63, -1.84+0.03j, 2.21+0.21j],
                      [0, 0, -8.87, 1.58-0.90j],
                      [0, 0, 0, -1.36]])
        # 创建复数类型的二维数组 b
        b = np.array([[2.98-10.18j, 28.68-39.89j],
                      [-9.58+3.88j, -24.79-8.40j],
                      [-0.77-16.05j, 4.23-70.02j],
                      [7.79+5.48j, -35.39+18.01j]])
        # 预期的解 x
        res = np.array([[2.+1j, -8+6j],
                        [3.-2j, 7-2j],
                        [-1+2j, -1+5j],
                        [1.-1j, 3-4j]])
        # 调用 solve 函数求解方程组，指定 assume_a='her' 表示输入矩阵 a 是 Hermitian 的，并进行精确度比较
        x = solve(a, b, assume_a='her')
        assert_array_almost_equal(x, res)
        # 对 a 的共轭转置进行求解，测试下三角数据，指定 assume_a='her' 和 lower=True
        x = solve(a.conj().T, b, assume_a='her', lower=True)
        assert_array_almost_equal(x, res)

    def test_pos_and_sym(self):
        # 创建正定矩阵 A
        A = np.arange(1, 10).reshape(3, 3)
        # 调用 solve 函数求解方程组，指定 assume_a='pos' 表示输入矩阵是正定的，并进行精确度比较
        x = solve(np.tril(A)/9, np.ones(3), assume_a='pos')
        assert_array_almost_equal(x, [9., 1.8, 1.])
        # 调用 solve 函数求解方程组，指定 assume_a='sym' 表示输入矩阵是对称的，并进行精确度比较
        x = solve(np.tril(A)/9, np.ones(3), assume_a='sym')
        assert_array_almost_equal(x, [9., 1.8, 1.])

    def test_singularity(self):
        # 创建一个奇异矩阵 a
        a = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 0, 1, 0, 1],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # 创建列向量 b
        b = np.arange(9)[:, None]
        # 断言求解奇异矩阵会引发 LinAlgError 异常
        assert_raises(LinAlgError, solve, a, b)

    def test_ill_condition_warning(self):
        # 创建病态矩阵 a 和向量 b
        a = np.array([[1, 1], [1+1e-16, 1-1e-16]])
        b = np.ones(2)
        # 捕获 LinAlgWarning 异常并断言
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert_raises(LinAlgWarning, solve, a, b)
    # 测试函数：测试多个右手边参数的情况
    def test_multiple_rhs(self):
        # 创建一个 2x2 的单位矩阵 a
        a = np.eye(2)
        # 创建一个形状为 (2, 3, 4) 的随机数组 b
        b = np.random.rand(2, 3, 4)
        # 使用 solve 函数解方程 a * x = b
        x = solve(a, b)
        # 断言解出的 x 与 b 几乎相等
        assert_array_almost_equal(x, b)

    # 测试函数：测试带有转置关键字的情况
    def test_transposed_keyword(self):
        # 创建一个 3x3 的数组 A
        A = np.arange(9).reshape(3, 3) + 1
        # 使用 solve 函数解方程 np.tril(A)/9 * x = np.ones(3)，并指定 transposed=True
        x = solve(np.tril(A)/9, np.ones(3), transposed=True)
        # 断言解出的 x 与 [1.2, 0.2, 1] 几乎相等
        assert_array_almost_equal(x, [1.2, 0.2, 1])
        # 使用 solve 函数解方程 np.tril(A)/9 * x = np.ones(3)，并指定 transposed=False
        x = solve(np.tril(A)/9, np.ones(3), transposed=False)
        # 断言解出的 x 与 [9, -5.4, -1.2] 几乎相等
        assert_array_almost_equal(x, [9, -5.4, -1.2])

    # 测试函数：测试未实现的转置情况
    def test_transposed_notimplemented(self):
        # 创建一个 3x3 的复数单位矩阵 a
        a = np.eye(3).astype(complex)
        # 使用 solve 函数解方程 a * x = a，并指定 transposed=True
        # 预期抛出 NotImplementedError 异常
        with assert_raises(NotImplementedError):
            solve(a, a, transposed=True)

    # 测试函数：测试非方阵 a 的情况
    def test_nonsquare_a(self):
        # 断言调用 solve 函数时，传入一个非方阵作为参数 a 会抛出 ValueError 异常
        assert_raises(ValueError, solve, [1, 2], 1)

    # 测试函数：测试与 1 维 b 大小不匹配的情况
    def test_size_mismatch_with_1D_b(self):
        # 断言解方程 np.eye(3) * x = np.ones(3)，并与 np.ones(3) 几乎相等
        assert_array_almost_equal(solve(np.eye(3), np.ones(3)), np.ones(3))
        # 断言调用 solve 函数时，传入大小不匹配的 b 会抛出 ValueError 异常
        assert_raises(ValueError, solve, np.eye(3), np.ones(4))

    # 测试函数：测试假定参数 assume_a 的情况
    def test_assume_a_keyword(self):
        # 断言调用 solve 函数时，传入未知假定参数 'zxcv' 会抛出 ValueError 异常
        assert_raises(ValueError, solve, 1, 1, assume_a='zxcv')

    # 使用 pytest.mark.skip 标记：在 OS X 上失败 (gh-7500)，在 Windows 上崩溃 (gh-8064)
    @pytest.mark.skip(reason="Failure on OS X (gh-7500), crash on Windows (gh-8064)")
    # 定义一个测试方法，用于测试所有类型和尺寸的例程组合
    def test_all_type_size_routine_combinations(self):
        # 定义不同的尺寸和假设集合
        sizes = [10, 100]
        assume_as = ['gen', 'sym', 'pos', 'her']
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        
        # 使用itertools.product生成所有尺寸、假设和数据类型的组合
        for size, assume_a, dtype in itertools.product(sizes, assume_as,
                                                       dtypes):
            # 检查数据类型是否复数
            is_complex = dtype in (np.complex64, np.complex128)
            # 如果假设是'her'且数据类型不是复数，则跳过当前循环
            if assume_a == 'her' and not is_complex:
                continue
            
            # 构建错误消息，用于测试失败时的输出
            err_msg = (f"Failed for size: {size}, assume_a: {assume_a},"
                       f"dtype: {dtype}")
            
            # 生成随机矩阵a和向量b，类型为dtype
            a = np.random.randn(size, size).astype(dtype)
            b = np.random.randn(size).astype(dtype)
            
            # 如果数据类型是复数，则生成复数随机数并加到a中
            if is_complex:
                a = a + (1j*np.random.randn(size, size)).astype(dtype)
            
            # 根据假设a的不同类型，修改矩阵a的值
            if assume_a == 'sym':
                # 如果假设为'sym'，则a变成对称矩阵
                a = a + a.T
            elif assume_a == 'her':
                # 如果假设为'her'，处理Hermitian矩阵的情况
                a = a + a.T.conj()
            elif assume_a == 'pos':
                # 如果假设为'pos'，则构造正定矩阵
                a = a.conj().T.dot(a) + 0.1*np.eye(size)
            
            # 根据数据类型设置误差容差值tol
            tol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-6
            
            # 如果假设为'gen', 'sym', 'her'，则修改误差容差tol
            if assume_a in ['gen', 'sym', 'her']:
                # 还原之前的容差值
                #   4b4a6e7c34fa4060533db38f9a819b98fa81476c
                if dtype in (np.float32, np.complex64):
                    tol *= 10
            
            # 解线性方程组a*x=b，使用solve函数进行求解
            x = solve(a, b, assume_a=assume_a)
            
            # 检查解x是否满足数值近似条件
            assert_allclose(a.dot(x), b,
                            atol=tol * size,
                            rtol=tol * size,
                            err_msg=err_msg)
            
            # 对于假设为'sym'且数据类型不是复数的情况，再次进行solve求解
            if assume_a == 'sym' and dtype not in (np.complex64,
                                                   np.complex128):
                x = solve(a, b, assume_a=assume_a, transposed=True)
                assert_allclose(a.dot(x), b,
                                atol=tol * size,
                                rtol=tol * size,
                                err_msg=err_msg)
    
    # 使用pytest的参数化标记定义一个测试方法，测试不同的数据类型dt_a和dt_b
    @pytest.mark.parametrize('dt_a', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt_a, dt_b):
        # 创建空的矩阵a和向量b，类型分别为dt_a和dt_b
        a = np.empty((0, 0), dtype=dt_a)
        b = np.empty(0, dtype=dt_b)
        
        # 解线性方程组a*x=b，使用solve函数进行求解
        x = solve(a, b)
        
        # 检查解x是否为空
        assert x.size == 0
        # 检查解x的数据类型与非空解的数据类型相同
        dt_nonempty = solve(np.eye(2, dtype=dt_a), np.ones(2, dtype=dt_b)).dtype
        assert x.dtype == dt_nonempty
    
    # 定义一个测试方法，测试空右侧向量b的情况
    def test_empty_rhs(self):
        # 创建单位矩阵a和空列表作为右侧向量b
        a = np.eye(2)
        b = [[], []]
        
        # 解线性方程组a*x=b，使用solve函数进行求解
        x = solve(a, b)
        
        # 检查解x是否为空数组
        assert_(x.size == 0, 'Returned array is not empty')
        # 检查解x的形状是否为(2, 0)
        assert_(x.shape == (2, 0), 'Returned empty array shape is wrong')
class TestSolveTriangular:

    def test_simple(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        # 创建一个2x2的数组A
        A = array([[1, 0], [1, 2]])
        # 创建一个长度为2的向量b
        b = [1, 1]
        # 解线性方程组A * x = b，其中A是下三角矩阵
        sol = solve_triangular(A, b, lower=True)
        # 断言解sol与预期结果[1, 0]几乎相等
        assert_array_almost_equal(sol, [1, 0])

        # 检查对于非连续矩阵也能正常工作
        sol = solve_triangular(A.T, b, lower=False)
        # 断言解sol与预期结果[0.5, 0.5]几乎相等
        assert_array_almost_equal(sol, [.5, .5])

        # 检查与trans=1的结果一致
        sol = solve_triangular(A, b, lower=True, trans=1)
        # 断言解sol与预期结果[0.5, 0.5]几乎相等
        assert_array_almost_equal(sol, [.5, .5])

        # 将b设置为单位矩阵
        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        # 断言解sol与预期结果[[1., -.5], [0, 0.5]]几乎相等
        assert_array_almost_equal(sol, [[1., -.5], [0, 0.5]])

    def test_simple_complex(self):
        """
        solve_triangular on a simple 2x2 complex matrix
        """
        # 创建一个2x2复数矩阵A
        A = array([[1+1j, 0], [1j, 2]])
        # 将b设置为单位矩阵
        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        # 断言解sol与预期结果[[.5-.5j, -.25-.25j], [0, 0.5]]几乎相等
        assert_array_almost_equal(sol, [[.5-.5j, -.25-.25j], [0, 0.5]])

        # 使用复数单位对角阵检查其他选项组合
        b = np.diag([1+1j, 1+2j])
        sol = solve_triangular(A, b, lower=True, trans=0)
        # 断言解sol与预期结果[[1, 0], [-0.5j, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5+1j]])

        sol = solve_triangular(A, b, lower=True, trans=1)
        # 断言解sol与预期结果[[1, 0.25-0.75j], [0, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1, 0.25-0.75j], [0, 0.5+1j]])

        sol = solve_triangular(A, b, lower=True, trans=2)
        # 断言解sol与预期结果[[1j, -0.75-0.25j], [0, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1j, -0.75-0.25j], [0, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=0)
        # 断言解sol与预期结果[[1, 0.25-0.75j], [0, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1, 0.25-0.75j], [0, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=1)
        # 断言解sol与预期结果[[1, 0], [-0.5j, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=2)
        # 断言解sol与预期结果[[1j, 0], [-0.5, 0.5+1j]]几乎相等
        assert_array_almost_equal(sol, [[1j, 0], [-0.5, 0.5+1j]])

    def test_check_finite(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        # 创建一个2x2的数组A
        A = array([[1, 0], [1, 2]])
        # 创建一个长度为2的向量b
        b = [1, 1]
        # 解线性方程组A * x = b，其中A是下三角矩阵，关闭有限性检查
        sol = solve_triangular(A, b, lower=True, check_finite=False)
        # 断言解sol与预期结果[1, 0]几乎相等
        assert_array_almost_equal(sol, [1, 0])

    @pytest.mark.parametrize('dt_a', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt_a, dt_b):
        # 创建一个空的2x2数组a
        a = np.empty((0, 0), dtype=dt_a)
        # 创建一个空的长度为0的向量b
        b = np.empty(0, dtype=dt_b)
        # 解线性方程组a * x = b
        x = solve_triangular(a, b)

        # 断言解x的大小为0
        assert x.size == 0
        # 检查非空解的数据类型
        dt_nonempty = solve_triangular(
            np.eye(2, dtype=dt_a), np.ones(2, dtype=dt_b)
        ).dtype
        # 断言解x的数据类型与非空解的数据类型相同
        assert x.dtype == dt_nonempty

    def test_empty_rhs(self):
        # 创建一个2x2的单位矩阵a
        a = np.eye(2)
        # 创建一个空的2x0数组b
        b = [[], []]
        # 解线性方程组a * x = b
        x = solve_triangular(a, b)
        # 断言解x的大小为0，说明解是空的
        assert_(x.size == 0, 'Returned array is not empty')
        # 断言解x的形状为(2, 0)，说明解是一个空的2x0矩阵
        assert_(x.shape == (2, 0), 'Returned empty array shape is wrong')
    # 设置测试方法的初始化，使用固定的随机种子以确保结果可重复性
    def setup_method(self):
        np.random.seed(1234)

    # 测试简单的情况
    def test_simple(self):
        # 创建一个 2x2 的矩阵 a
        a = [[1, 2], [3, 4]]
        # 计算矩阵 a 的逆矩阵
        a_inv = inv(a)
        # 断言矩阵 a 乘以其逆矩阵等于单位矩阵
        assert_array_almost_equal(dot(a, a_inv), np.eye(2))
        
        # 创建一个 3x3 的矩阵 a
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        # 计算矩阵 a 的逆矩阵
        a_inv = inv(a)
        # 断言矩阵 a 乘以其逆矩阵等于单位矩阵
        assert_array_almost_equal(dot(a, a_inv), np.eye(3))

    # 测试随机矩阵
    def test_random(self):
        n = 20
        for i in range(4):
            # 创建一个 n x n 的随机矩阵 a
            a = random([n, n])
            # 将对角线元素调整为稍大于原值的 20 倍
            for i in range(n):
                a[i, i] = 20 * (.1 + a[i, i])
            # 计算矩阵 a 的逆矩阵
            a_inv = inv(a)
            # 断言矩阵 a 乘以其逆矩阵等于单位矩阵
            assert_array_almost_equal(dot(a, a_inv), identity(n))

    # 测试包含复数的简单情况
    def test_simple_complex(self):
        # 创建一个包含复数的 2x2 矩阵 a
        a = [[1, 2], [3, 4j]]
        # 计算矩阵 a 的逆矩阵
        a_inv = inv(a)
        # 断言矩阵 a 乘以其逆矩阵等于 [[1, 0], [0, 1]]
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])

    # 测试包含复数的随机矩阵
    def test_random_complex(self):
        n = 20
        for i in range(4):
            # 创建一个包含复数的 n x n 随机矩阵 a
            a = random([n, n]) + 2j * random([n, n])
            # 将对角线元素调整为稍大于原值的 20 倍
            for i in range(n):
                a[i, i] = 20 * (.1 + a[i, i])
            # 计算矩阵 a 的逆矩阵
            a_inv = inv(a)
            # 断言矩阵 a 乘以其逆矩阵等于单位矩阵
            assert_array_almost_equal(dot(a, a_inv), identity(n))

    # 测试带有 check_finite=False 的简单情况
    def test_check_finite(self):
        # 创建一个 2x2 的矩阵 a
        a = [[1, 2], [3, 4]]
        # 计算矩阵 a 的逆矩阵，不检查有限性
        a_inv = inv(a, check_finite=False)
        # 断言矩阵 a 乘以其逆矩阵等于 [[1, 0], [0, 1]]
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])

    # 使用 pytest 的参数化功能，测试空矩阵的情况
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 创建一个空的 dtype 类型为 dt 的矩阵 a
        a = np.empty((0, 0), dtype=dt)
        # 计算矩阵 a 的逆矩阵
        a_inv = inv(a)
        # 断言逆矩阵的大小为 0
        assert a_inv.size == 0
        # 断言逆矩阵的 dtype 与单位矩阵的 dtype 相同
        assert a_inv.dtype == inv(np.eye(2, dtype=dt)).dtype
# 定义一个测试类 TestDet
class TestDet:
    
    # 在每个测试方法运行之前设置随机数生成器
    def setup_method(self):
        self.rng = np.random.default_rng(1680305949878959)
    
    # 测试 1x1 的情况，所有维度都是单元素的情况
    def test_1x1_all_singleton_dims(self):
        # 创建一个 1x1 的 numpy 数组
        a = np.array([[1]])
        # 计算该数组的行列式
        deta = det(a)
        # 断言行列式的数据类型为双精度浮点数
        assert deta.dtype.char == 'd'
        # 断言行列式是一个标量
        assert np.isscalar(deta)
        # 断言行列式的值为 1.0
        assert deta == 1.0
        
        # 创建一个复杂类型的 1x1 numpy 数组
        a = np.array([[[[1]]]], dtype='f')
        # 计算该数组的行列式
        deta = det(a)
        # 断言行列式的数据类型为双精度浮点数
        assert deta.dtype.char == 'd'
        # 断言行列式是一个标量
        assert np.isscalar(deta)
        # 断言行列式的值为 1.0
        assert deta == 1.0
        
        # 创建一个复数类型的 1x1 numpy 数组
        a = np.array([[[1 + 3.j]]], dtype=np.complex64)
        # 计算该数组的行列式
        deta = det(a)
        # 断言行列式的数据类型为双精度复数
        assert deta.dtype.char == 'D'
        # 断言行列式是一个标量
        assert np.isscalar(deta)
        # 断言行列式的值为 1.0 + 3.0j
        assert deta == 1.0 + 3.0j
    
    # 测试 1x1x1x1 堆叠输入输出的情况
    def test_1by1_stacked_input_output(self):
        # 使用随机数生成器创建一个形状为 [4, 5, 1, 1] 的单精度浮点数数组
        a = self.rng.random([4, 5, 1, 1], dtype=np.float32)
        # 计算该数组的行列式
        deta = det(a)
        # 断言行列式的数据类型为双精度浮点数
        assert deta.dtype.char == 'd'
        # 断言行列式的形状为 (4, 5)
        assert deta.shape == (4, 5)
        # 使用 assert_allclose 检查行列式的计算结果与原始数组的挤压版本的接近度
        assert_allclose(deta, np.squeeze(a))
        
        # 使用随机数生成器创建一个形状为 [4, 5, 1, 1] 的单精度复数数组
        a = self.rng.random([4, 5, 1, 1], dtype=np.float32) * np.complex64(1.j)
        # 计算该数组的行列式
        deta = det(a)
        # 断言行列式的数据类型为双精度复数
        assert deta.dtype.char == 'D'
        # 断言行列式的形状为 (4, 5)
        assert deta.shape == (4, 5)
        # 使用 assert_allclose 检查行列式的计算结果与原始数组的挤压版本的接近度
        assert_allclose(deta, np.squeeze(a))
    
    # 使用 pytest.mark.parametrize 装饰器指定多个测试参数，测试不同形状的行列式计算结果（实数和复数）
    @pytest.mark.parametrize('shape', [[2, 2], [20, 20], [3, 2, 20, 20]])
    def test_simple_det_shapes_real_complex(self, shape):
        # 使用随机数生成器创建形状为 shape 的均匀分布的实数数组
        a = self.rng.uniform(-1., 1., size=shape)
        # 分别计算数组的行列式，使用 det 函数和 np.linalg.det 函数
        d1, d2 = det(a), np.linalg.det(a)
        # 使用 assert_allclose 检查两种方法计算得到的行列式结果的接近度
        assert_allclose(d1, d2)
        
        # 使用随机数生成器创建形状为 shape 的复数数组
        b = self.rng.uniform(-1., 1., size=shape) * 1j
        b += self.rng.uniform(-0.5, 0.5, size=shape)
        # 分别计算数组的行列式，使用 det 函数和 np.linalg.det 函数
        d3, d4 = det(b), np.linalg.det(b)
        # 使用 assert_allclose 检查两种方法计算得到的行列式结果的接近度
        assert_allclose(d3, d4)
    # 测试已知的特定行列式值

    # 创建一个 8x8 的 Hadamard 矩阵 a
    a = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, -1, 1, -1, 1, -1, 1, -1],
                  [1, 1, -1, -1, 1, 1, -1, -1],
                  [1, -1, -1, 1, 1, -1, -1, 1],
                  [1, 1, 1, 1, -1, -1, -1, -1],
                  [1, -1, 1, -1, -1, 1, -1, 1],
                  [1, 1, -1, -1, -1, -1, 1, 1],
                  [1, -1, -1, 1, -1, 1, 1, -1]])

    # 断言计算出的矩阵 a 的行列式接近于 4096.0
    assert_allclose(det(a), 4096.)

    # 创建一个 5x5 的连续数值数组，预期其行列式接近于 0
    assert_allclose(det(np.arange(25).reshape(5, 5)), 0.)

    # 创建一个复数类型的 4x4 对角线反向的矩阵 a
    # 其右上角的子块行列式为 (-2+1j)，左下角的子块行列式为 (-2-1j)
    # 计算矩阵 a 的行列式，预期结果为 5.0
    a = np.array([[0.+0.j, 0.+0.j, 0.-1.j, 1.-1.j],
                  [0.+0.j, 0.+0.j, 1.+0.j, 0.-1.j],
                  [0.+1.j, 1.+1.j, 0.+0.j, 0.+0.j],
                  [1.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]], dtype=np.complex64)
    assert_allclose(det(a), 5.+0.j)

    # 创建一个复数类型的 8x8 矩阵 a，为 Fiedler 伴随矩阵
    # 预期矩阵 a 的行列式为 9.0
    a = np.array([[-2., -3., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [0., -4., 0., -5., 1., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., -6., 0., -7., 1., 0.],
                  [0., 0., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., -8., 0., -9.],
                  [0., 0., 0., 0., 0., 1., 0., 0.]])*1.j
    assert_allclose(det(a), 9.)

# 在 Windows 和其他平台中，'g' 和 'G' 类型在处理上有所不同
@pytest.mark.parametrize('typ', [x for x in np.typecodes['All'][:20]
                                 if x not in 'gG'])
def test_sample_compatible_dtype_input(self, typ):
    # 创建一个 n x n 大小的随机数组 a，其类型由 typ 参数决定
    n = 4
    a = self.rng.random([n, n]).astype(typ)  # 具体数值并不重要
    # 断言计算矩阵 a 的行列式为 np.float64 或 np.complex128 类型
    assert isinstance(det(a), (np.float64, np.complex128))

def test_incompatible_dtype_input(self):
    # 双斜杠用于转义 pytest 的正则表达式
    msg = 'cannot be cast to float\\(32, 64\\)'

    # 使用不兼容的数据类型 c 创建数组，预期引发 TypeError，并匹配 msg 中的错误信息
    for c, t in zip('SUO', ['bytes8', 'str32', 'object']):
        with assert_raises(TypeError, match=msg):
            det(np.array([['a', 'b']]*2, dtype=c))
    with assert_raises(TypeError, match=msg):
        det(np.array([[b'a', b'b']]*2, dtype='V'))
    with assert_raises(TypeError, match=msg):
        det(np.array([[100, 200]]*2, dtype='datetime64[s]'))
    with assert_raises(TypeError, match=msg):
        det(np.array([[100, 200]]*2, dtype='timedelta64[s]'))
    # 测试空矩阵的边缘情况
    def test_empty_edge_cases(self):
        # 断言空矩阵的行列式为1.0
        assert_allclose(det(np.empty([0, 0])), 1.)
        # 断言三维空矩阵的行列式为空数组
        assert_allclose(det(np.empty([0, 0, 0])), np.array([]))
        # 断言三维矩阵的行列式为包含三个1.0的数组
        assert_allclose(det(np.empty([3, 0, 0])), np.array([1., 1., 1.]))
        # 使用assert_raises检查空三维矩阵最后两个维度是否符合预期，应引发ValueError异常
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.empty([0, 0, 3]))
        # 使用assert_raises检查一维空数组是否符合预期，应引发ValueError异常
        with assert_raises(ValueError, match='at least two-dimensional'):
            det(np.array([]))
        # 使用assert_raises检查二维空数组是否符合预期，应引发ValueError异常
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.array([[]]))
        # 使用assert_raises检查三维空数组是否符合预期，应引发ValueError异常
        with assert_raises(ValueError, match='Last 2 dimensions'):
            det(np.array([[[]]]))

    # 使用pytest.mark.parametrize装饰器，对不同数据类型进行测试
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    def test_empty_dtype(self, dt):
        # 创建空矩阵a，指定数据类型为dt
        a = np.empty((0, 0), dtype=dt)
        # 计算矩阵a的行列式
        d = det(a)
        # 断言行列式的形状为空元组
        assert d.shape == ()
        # 断言行列式的数据类型与单位矩阵的行列式数据类型相同
        assert d.dtype == det(np.eye(2, dtype=dt)).dtype

        # 创建三维空矩阵a，指定数据类型为dt
        a = np.empty((3, 0, 0), dtype=dt)
        # 计算矩阵a的行列式
        d = det(a)
        # 断言行列式的形状为(3,)
        assert d.shape == (3,)
        # 创建三维包含一个元素的空矩阵，指定数据类型为dt
        assert d.dtype == det(np.empty((3, 1, 1), dtype=dt)).dtype

    # 测试是否覆盖输入矩阵a
    def test_overwrite_a(self):
        # 如果满足以下条件，则应覆盖输入矩阵a:
        #   - 数据类型是'fdFD'之一
        #   - 是C连续的
        #   - 可写
        a = np.arange(9).reshape(3, 3).astype(np.float32)
        # 创建a的副本ac
        ac = a.copy()
        # 计算矩阵a的行列式，覆盖原始矩阵a
        deta = det(ac, overwrite_a=True)
        # 断言行列式为0.0
        assert_allclose(deta, 0.)
        # 断言原始矩阵a和副本ac不完全相等
        assert not (a == ac).all()

    # 测试只读数组的情况
    def test_readonly_array(self):
        # 创建矩阵a
        a = np.array([[2., 0., 1.], [5., 3., -1.], [1., 1., 1.]])
        # 将矩阵a设为只读
        a.setflags(write=False)
        # 断言计算矩阵a的行列式，覆盖只读设置
        assert_allclose(det(a, overwrite_a=True), 10.)

    # 测试包含无穷大的数组情况
    def test_simple_check_finite(self):
        # 创建包含无穷大的数组a
        a = [[1, 2], [3, np.inf]]
        # 使用assert_raises检查数组a是否包含无穷大，应引发ValueError异常
        with assert_raises(ValueError, match='array must not contain'):
            det(a)
# 定义函数 direct_lstsq，用于通过最小二乘法求解线性方程组
def direct_lstsq(a, b, cmplx=0):
    # 对矩阵 a 进行转置操作
    at = transpose(a)
    # 如果 cmplx 参数为真，则对转置后的矩阵 at 进行共轭操作
    if cmplx:
        at = conjugate(at)
    # 计算矩阵 a 转置后的乘积
    a1 = dot(at, a)
    # 计算矩阵 a 转置后与向量 b 的乘积
    b1 = dot(at, b)
    # 使用 solve 函数求解线性方程组，并返回结果
    return solve(a1, b1)


# 定义测试类 TestLstsq
class TestLstsq:
    # 设定 Lapack 驱动器的可能取值
    lapack_drivers = ('gelsd', 'gelss', 'gelsy', None)

    # 定义测试方法 test_simple_exact
    def test_simple_exact(self):
        # 遍历实数数据类型的可能取值
        for dtype in REAL_DTYPES:
            # 创建一个二维数组 a，数据类型为 dtype
            a = np.array([[1, 20], [-30, 4]], dtype=dtype)
            # 遍历 Lapack 驱动器的可能取值
            for lapack_driver in TestLstsq.lapack_drivers:
                # 遍历是否覆盖的选项
                for overwrite in (True, False):
                    # 遍历不同的测试向量 bt
                    for bt in (((1, 0), (0, 1)), (1, 0),
                               ((2, 1), (-30, 4))):
                        # 在可能被后续代码覆盖的情况下，存储数组 a 的副本
                        a1 = a.copy()
                        # 创建测试向量 b，数据类型为 dtype
                        b = np.array(bt, dtype=dtype)
                        # 在可能被后续代码覆盖的情况下，存储数组 b 的副本
                        b1 = b.copy()
                        # 调用 lstsq 函数求解线性方程组
                        out = lstsq(a1, b1,
                                    lapack_driver=lapack_driver,
                                    overwrite_a=overwrite,
                                    overwrite_b=overwrite)
                        # 获取解 x 和残差 r
                        x = out[0]
                        r = out[2]
                        # 断言残差 r 等于 2，用于验证结果是否为高效秩 2
                        assert_(r == 2,
                                'expected efficient rank 2, got %s' % r)
                        # 断言通过 dot 函数计算的误差在允许范围内，用于验证解的准确性
                        assert_allclose(dot(a, x), b,
                                        atol=25 * _eps_cast(a1.dtype),
                                        rtol=25 * _eps_cast(a1.dtype),
                                        err_msg="driver: %s" % lapack_driver)
    def test_simple_overdet(self):
        # 针对 REAL_DTYPES 中的每种数据类型进行测试
        for dtype in REAL_DTYPES:
            # 创建一个二维数组 a，数据类型为 dtype
            a = np.array([[1, 2], [4, 5], [3, 4]], dtype=dtype)
            # 创建一个一维数组 b，数据类型为 dtype
            b = np.array([1, 2, 3], dtype=dtype)
            # 遍历 TestLstsq 类中的 lapack_drivers 列表
            for lapack_driver in TestLstsq.lapack_drivers:
                # 遍历 overwrite_a 和 overwrite_b 的所有可能取值
                for overwrite in (True, False):
                    # 备份数组 a 和 b 的值，以防它们在后续被修改
                    a1 = a.copy()
                    b1 = b.copy()
                    # 调用 lstsq 函数进行最小二乘解
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)
                    # 获取最小二乘解 x
                    x = out[0]
                    # 根据 lapack_driver 的值计算残差
                    if lapack_driver == 'gelsy':
                        residuals = np.sum((b - a.dot(x))**2)
                    else:
                        residuals = out[1]
                    # 获取矩阵的有效秩 r
                    r = out[2]
                    # 断言有效秩为 2
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    # 断言计算得到的残差与 dot(a, x) - b 的平方差近似相等
                    assert_allclose(abs((dot(a, x) - b)**2).sum(axis=0),
                                    residuals,
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)
                    # 断言最小二乘解 x 的值近似等于 (-0.428571428571429, 0.85714285714285)
                    assert_allclose(x, (-0.428571428571429, 0.85714285714285),
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)
    # 定义一个测试方法，用于测试复杂类型的线性方程组求解
    def test_simple_overdet_complex(self):
        # 对于复杂数据类型中的每种类型
        for dtype in COMPLEX_DTYPES:
            # 创建一个二维数组 a，包含复数和实数，指定数据类型为当前类型
            a = np.array([[1+2j, 2], [4, 5], [3, 4]], dtype=dtype)
            # 创建一个一维数组 b，包含复数和实数，指定数据类型为当前类型
            b = np.array([1, 2+4j, 3], dtype=dtype)
            # 遍历所有的 LAPACK 驱动器
            for lapack_driver in TestLstsq.lapack_drivers:
                # 遍历是否覆盖现有数据的选项 True 和 False
                for overwrite in (True, False):
                    # 在可能被后续操作覆盖的情况下存储变量的当前值
                    a1 = a.copy()
                    b1 = b.copy()
                    # 调用 lstsq 函数求解线性方程组，返回解 x 和其他信息
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)

                    # 获取解 x
                    x = out[0]
                    # 如果使用的是 'gelsy' 驱动器，计算残差
                    if lapack_driver == 'gelsy':
                        res = b - a.dot(x)
                        residuals = np.sum(res * res.conj())
                    else:
                        residuals = out[1]
                    # 获取秩 r
                    r = out[2]
                    # 断言秩为 2，用于验证期望的秩是否达到
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    # 断言解 x 的平方差的绝对值之和接近残差，用于验证求解精度
                    assert_allclose(abs((dot(a, x) - b)**2).sum(axis=0),
                                    residuals,
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)
                    # 断言解 x 的每个元素接近预期值，用于验证求解精度
                    assert_allclose(
                                x, (-0.4831460674157303 + 0.258426966292135j,
                                    0.921348314606741 + 0.292134831460674j),
                                rtol=25 * _eps_cast(a1.dtype),
                                atol=25 * _eps_cast(a1.dtype),
                                err_msg="driver: %s" % lapack_driver)

    # 定义一个测试方法，用于测试欠定的线性方程组求解
    def test_simple_underdet(self):
        # 对于实数数据类型中的每种类型
        for dtype in REAL_DTYPES:
            # 创建一个二维数组 a，包含整数，指定数据类型为当前类型
            a = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            # 创建一个一维数组 b，包含整数，指定数据类型为当前类型
            b = np.array([1, 2], dtype=dtype)
            # 遍历所有的 LAPACK 驱动器
            for lapack_driver in TestLstsq.lapack_drivers:
                # 遍历是否覆盖现有数据的选项 True 和 False
                for overwrite in (True, False):
                    # 在可能被后续操作覆盖的情况下存储变量的当前值
                    a1 = a.copy()
                    b1 = b.copy()
                    # 调用 lstsq 函数求解线性方程组，返回解 x 和其他信息
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)

                    # 获取解 x
                    x = out[0]
                    # 获取秩 r
                    r = out[2]
                    # 断言秩为 2，用于验证期望的秩是否达到
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    # 断言解 x 的每个元素接近预期值，用于验证求解精度
                    assert_allclose(x, (-0.055555555555555, 0.111111111111111,
                                        0.277777777777777),
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)

    # 使用参数化测试的方式定义测试用例
    @pytest.mark.parametrize("dtype", REAL_DTYPES)
    @pytest.mark.parametrize("n", (20, 200))
    @pytest.mark.parametrize("lapack_driver", lapack_drivers)
    # 使用 pytest 的参数化装饰器，测试随机精确解算法的不同参数组合
    @pytest.mark.parametrize("overwrite", (True, False))
    def test_random_exact(self, dtype, n, lapack_driver, overwrite):
        # 创建一个指定种子的随机数生成器
        rng = np.random.RandomState(1234)
    
        # 生成一个随机的 n x n 数组 a，使用指定的数据类型 dtype
        a = np.asarray(rng.random([n, n]), dtype=dtype)
        # 对角线上的元素增加一个常数，保证矩阵 a 可逆
        for i in range(n):
            a[i, i] = 20 * (0.1 + a[i, i])
        
        # 执行 4 次循环，生成随机的 n x 3 数组 b
        for i in range(4):
            b = np.asarray(rng.random([n, 3]), dtype=dtype)
            # 在可能被后续代码覆盖的情况下存储值的副本
            a1 = a.copy()
            b1 = b.copy()
            # 调用 lstsq 函数求解线性最小二乘问题
            out = lstsq(a1, b1,
                        lapack_driver=lapack_driver,
                        overwrite_a=overwrite,
                        overwrite_b=overwrite)
            # 获取解 x 和残差 r
            x = out[0]
            r = out[2]
            # 断言残差 r 等于预期值 n，用于验证解的有效秩
            assert_(r == n, f'expected efficient rank {n}, '
                    f'got {r}')
            
            # 根据数据类型为 np.float32 或非 np.float32 分别进行断言
            if dtype is np.float32:
                # 使用 assert_allclose 检查 dot(a, x) 是否接近 b
                assert_allclose(
                          dot(a, x), b,
                          rtol=500 * _eps_cast(a1.dtype),
                          atol=500 * _eps_cast(a1.dtype),
                          err_msg="driver: %s" % lapack_driver)
            else:
                assert_allclose(
                          dot(a, x), b,
                          rtol=1000 * _eps_cast(a1.dtype),
                          atol=1000 * _eps_cast(a1.dtype),
                          err_msg="driver: %s" % lapack_driver)
    
    # 根据操作系统为 Musl libc 的条件跳过测试，避免在 Alpine 上导致段错误
    @pytest.mark.skipif(IS_MUSL, reason="may segfault on Alpine, see gh-17630")
    # 使用 pytest 的参数化装饰器，测试复数数据类型的线性最小二乘解算法
    @pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
    # 使用 pytest 的参数化装饰器，测试不同尺寸的矩阵 n
    @pytest.mark.parametrize("n", (20, 200))
    # 使用 pytest 的参数化装饰器，测试不同 LAPACK 驱动程序
    @pytest.mark.parametrize("lapack_driver", lapack_drivers)
    # 使用 pytest 的参数化装饰器，测试是否覆盖输入数组的标志
    @pytest.mark.parametrize("overwrite", (True, False))
    # 定义一个测试方法，用于测试具有随机复杂性和精确性的情况
    def test_random_complex_exact(self, dtype, n, lapack_driver, overwrite):
        # 创建指定种子的随机数生成器
        rng = np.random.RandomState(1234)

        # 创建一个随机复数矩阵 a，数据类型为 dtype
        a = np.asarray(rng.random([n, n]) + 1j*rng.random([n, n]),
                       dtype=dtype)
        # 将对角线上的元素加权，增加其值
        for i in range(n):
            a[i, i] = 20 * (0.1 + a[i, i])

        # 迭代两次，创建随机矩阵 b，数据类型为 dtype
        for i in range(2):
            b = np.asarray(rng.random([n, 3]), dtype=dtype)
            # 备份 a 和 b 的副本，以防后续被覆盖
            a1 = a.copy()
            b1 = b.copy()
            # 调用最小二乘法求解方程 lstsq
            out = lstsq(a1, b1, lapack_driver=lapack_driver,
                        overwrite_a=overwrite,
                        overwrite_b=overwrite)
            # 获取结果中的解 x 和残差 r
            x = out[0]
            r = out[2]
            # 断言残差 r 等于 n，以保证获得预期的有效秩 n
            assert_(r == n, f'expected efficient rank {n}, '
                    f'got {r}')
            # 根据数据类型 dtype 判断是否为复数类型 np.complex64
            if dtype is np.complex64:
                # 断言 dot(a, x) 接近 b，使用相对和绝对误差进行比较
                assert_allclose(
                          dot(a, x), b,
                          rtol=400 * _eps_cast(a1.dtype),
                          atol=400 * _eps_cast(a1.dtype),
                          err_msg="driver: %s" % lapack_driver)
            else:
                # 断言 dot(a, x) 接近 b，使用相对和绝对误差进行比较
                assert_allclose(
                          dot(a, x), b,
                          rtol=1000 * _eps_cast(a1.dtype),
                          atol=1000 * _eps_cast(a1.dtype),
                          err_msg="driver: %s" % lapack_driver)

    # 定义一个测试方法，用于测试过定的情况
    def test_random_overdet(self):
        # 创建指定种子的随机数生成器
        rng = np.random.RandomState(1234)
        # 遍历实数数据类型的列表 REAL_DTYPES
        for dtype in REAL_DTYPES:
            # 遍历不同的 n 和 m 组合
            for (n, m) in ((20, 15), (200, 2)):
                # 遍历 LAPACK 驱动程序的列表
                for lapack_driver in TestLstsq.lapack_drivers:
                    # 遍历 overwrite 的两种取值 True 和 False
                    for overwrite in (True, False):
                        # 创建一个随机矩阵 a，数据类型为 dtype，大小为 n x m
                        a = np.asarray(rng.random([n, m]), dtype=dtype)
                        # 将对角线上的元素加权，增加其值
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])

                        # 迭代四次，创建随机矩阵 b，数据类型为 dtype
                        for i in range(4):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            # 备份 a 和 b 的副本，以防后续被覆盖
                            a1 = a.copy()
                            b1 = b.copy()
                            # 调用最小二乘法求解方程 lstsq
                            out = lstsq(a1, b1,
                                        lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            # 获取结果中的解 x 和残差 r
                            x = out[0]
                            r = out[2]
                            # 断言残差 r 等于 m，以保证获得预期的有效秩 m
                            assert_(r == m, f'expected efficient rank {m}, '
                                    f'got {r}')
                            # 断言 x 接近 direct_lstsq(a, b, cmplx=0)，使用相对和绝对误差进行比较
                            assert_allclose(
                                          x, direct_lstsq(a, b, cmplx=0),
                                          rtol=25 * _eps_cast(a1.dtype),
                                          atol=25 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)
    # 定义一个测试方法，用于测试复杂随机数的最小二乘解
    def test_random_complex_overdet(self):
        # 设置随机数生成器，并指定种子为1234
        rng = np.random.RandomState(1234)
        # 遍历复杂数类型的数据类型列表
        for dtype in COMPLEX_DTYPES:
            # 遍历不同的矩阵尺寸组合
            for (n, m) in ((20, 15), (200, 2)):
                # 遍历使用的 LAPACK 驱动器类型
                for lapack_driver in TestLstsq.lapack_drivers:
                    # 遍历是否覆盖原始数据的选项
                    for overwrite in (True, False):
                        # 生成一个随机复数矩阵 a，维度为 n x m
                        a = np.asarray(rng.random([n, m]) + 1j*rng.random([n, m]),
                                       dtype=dtype)
                        # 对角线上的每个元素乘以一个常数
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        # 生成一个随机复数矩阵 b，维度为 n x 3
                        for i in range(2):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            # 备份矩阵 a 和 b 的值，以防它们后续被修改
                            a1 = a.copy()
                            b1 = b.copy()
                            # 调用最小二乘求解函数 lstsq，返回解 x 和其他信息
                            out = lstsq(a1, b1,
                                        lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            x = out[0]  # 获取最小二乘解
                            r = out[2]  # 获取有效秩（rank）
                            # 断言有效秩等于 m
                            assert_(r == m, f'expected efficient rank {m}, '
                                    f'got {r}')
                            # 检查解 x 是否接近直接求解结果，设置相对和绝对容差
                            assert_allclose(
                                      x, direct_lstsq(a, b, cmplx=1),
                                      rtol=25 * _eps_cast(a1.dtype),
                                      atol=25 * _eps_cast(a1.dtype),
                                      err_msg="driver: %s" % lapack_driver)
    # 定义测试函数，用于检查矩阵的解是否正确
    def test_check_finite(self):
        # 使用 suppress_warnings 上下文管理器来忽略特定的警告信息
        with suppress_warnings() as sup:
            # 在某些 OSX 系统上，该测试会触发警告 (gh-7538)
            sup.filter(RuntimeWarning,
                       "internal gelsd driver lwork query error,.*"
                       "Falling back to 'gelss' driver.")

        # 创建一个测试用的矩阵 a
        at = np.array(((1, 20), (-30, 4)))
        # 使用 itertools.product 来生成各种参数组合
        for dtype, bt, lapack_driver, overwrite, check_finite in \
            itertools.product(REAL_DTYPES,
                              (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))),
                              TestLstsq.lapack_drivers,
                              (True, False),
                              (True, False)):

            # 将 at 矩阵转换为指定数据类型的数组 a
            a = at.astype(dtype)
            # 创建数组 b，根据 bt 变量
            b = np.array(bt, dtype=dtype)
            # 复制数组 a 和 b 到 a1 和 b1，以防它们在后续被覆盖修改
            a1 = a.copy()
            b1 = b.copy()
            # 调用 lstsq 函数求解线性方程组
            out = lstsq(a1, b1, lapack_driver=lapack_driver,
                        check_finite=check_finite, overwrite_a=overwrite,
                        overwrite_b=overwrite)
            # 获取解 x 和残差矩阵 r
            x = out[0]
            r = out[2]
            # 断言残差矩阵的秩为 2，期望得到高效的秩 2 解
            assert_(r == 2, 'expected efficient rank 2, got %s' % r)
            # 使用 assert_allclose 检查解 x 是否满足精度要求
            assert_allclose(dot(a, x), b,
                            rtol=25 * _eps_cast(a.dtype),
                            atol=25 * _eps_cast(a.dtype),
                            err_msg="driver: %s" % lapack_driver)

    # 定义测试函数，用于检查对空矩阵的处理
    def test_empty(self):
        # 使用 for 循环遍历不同的空矩阵形状组合
        for a_shape, b_shape in (((0, 2), (0,)),
                                 ((0, 4), (0, 2)),
                                 ((4, 0), (4,)),
                                 ((4, 0), (4, 2))):
            # 创建数组 b，形状由 b_shape 决定
            b = np.ones(b_shape)
            # 调用 lstsq 函数求解空矩阵 a 和数组 b 的线性方程组
            x, residues, rank, s = lstsq(np.zeros(a_shape), b)
            # 断言得到的解 x 是与 a_shape 相同形状的零数组
            assert_equal(x, np.zeros((a_shape[1],) + b_shape[1:]))
            # 计算预期的残差值
            residues_should_be = (np.empty((0,)) if a_shape[1]
                                  else np.linalg.norm(b, axis=0)**2)
            # 断言得到的残差与预期值一致
            assert_equal(residues, residues_should_be)
            # 断言秩为 0，即期望得到的秩为 0
            assert_(rank == 0, 'expected rank 0')
            # 断言 s 为空数组
            assert_equal(s, np.empty((0,)))

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，测试不同数据类型的空矩阵处理
    @pytest.mark.parametrize('dt_a', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty_dtype(self, dt_a, dt_b):
        # 创建空矩阵 a 和空数组 b，分别使用指定的数据类型 dt_a 和 dt_b
        a = np.empty((0, 0), dtype=dt_a)
        b = np.empty(0, dtype=dt_b)
        # 调用 lstsq 函数求解空矩阵 a 和空数组 b 的线性方程组
        x, residues, rank, s = lstsq(a, b)

        # 断言解 x 的大小为 0
        assert x.size == 0
        # 调用 lstsq 函数解非空矩阵和数组得到的解的数据类型
        dt_nonempty = lstsq(np.eye(2, dtype=dt_a), np.ones(2, dtype=dt_b))[0].dtype
        # 断言解 x 的数据类型与非空矩阵解的数据类型一致
        assert x.dtype == dt_nonempty
# 定义一个名为 TestPinv 的测试类
class TestPinv:
    
    # 在每个测试方法执行前，设置随机数种子为 1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试 pinv 函数对实数矩阵的行为
    def test_simple_real(self):
        # 创建一个实数类型的 3x3 数组 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 断言矩阵 a 与其伪逆相乘应接近单位矩阵
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    # 测试 pinv 函数对复数矩阵的行为
    def test_simple_complex(self):
        # 创建一个复数类型的 3x3 数组 a
        a = (array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float) +
             1j * array([[10, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float))
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 断言矩阵 a 与其伪逆相乘应接近单位矩阵
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    # 测试 pinv 函数对奇异矩阵的行为
    def test_simple_singular(self):
        # 创建一个奇异矩阵 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 期望的伪逆结果
        expected = array([[-6.38888889e-01, -1.66666667e-01, 3.05555556e-01],
                          [-5.55555556e-02, 1.30136518e-16, 5.55555556e-02],
                          [5.27777778e-01, 1.66666667e-01, -1.94444444e-01]])
        # 断言计算出的伪逆与期望值接近
        assert_array_almost_equal(a_pinv, expected)

    # 测试 pinv 函数对列数小于行数的矩阵的行为
    def test_simple_cols(self):
        # 创建一个 2x3 数组 a
        a = array([[1, 2, 3], [4, 5, 6]], dtype=float)
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 期望的伪逆结果
        expected = array([[-0.94444444, 0.44444444],
                          [-0.11111111, 0.11111111],
                          [0.72222222, -0.22222222]])
        # 断言计算出的伪逆与期望值接近
        assert_array_almost_equal(a_pinv, expected)

    # 测试 pinv 函数对行数小于列数的矩阵的行为
    def test_simple_rows(self):
        # 创建一个 3x2 数组 a
        a = array([[1, 2], [3, 4], [5, 6]], dtype=float)
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 期望的伪逆结果
        expected = array([[-1.33333333, -0.33333333, 0.66666667],
                          [1.08333333, 0.33333333, -0.41666667]])
        # 断言计算出的伪逆与期望值接近
        assert_array_almost_equal(a_pinv, expected)

    # 测试 pinv 函数在 check_finite=False 时的行为
    def test_check_finite(self):
        # 创建一个包含非有限元素的 3x3 数组 a
        a = array([[1, 2, 3], [4, 5, 6.], [7, 8, 10]])
        # 计算矩阵 a 的伪逆，关闭有限性检查
        a_pinv = pinv(a, check_finite=False)
        # 断言矩阵 a 与其伪逆相乘应接近单位矩阵
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    # 测试 pinv 函数接受原生 Python 列表作为参数的行为
    def test_native_list_argument(self):
        # 创建一个原生 Python 列表形式的 3x3 数组 a
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 期望的伪逆结果
        expected = array([[-6.38888889e-01, -1.66666667e-01, 3.05555556e-01],
                          [-5.55555556e-02, 1.30136518e-16, 5.55555556e-02],
                          [5.27777778e-01, 1.66666667e-01, -1.94444444e-01]])
        # 断言计算出的伪逆与期望值接近
        assert_array_almost_equal(a_pinv, expected)
    def test_atol_rtol(self):
        # 定义矩阵维度
        n = 12
        # 获取一个随机的正交矩阵用于混洗
        q, _ = qr(np.random.rand(n, n))
        # 创建一个 7x5 的浮点数数组
        a_m = np.arange(35.0).reshape(7, 5)
        # 创建数组 a 的副本
        a = a_m.copy()
        # 修改数组 a 的第一个元素为 0.001
        a[0, 0] = 0.001
        # 绝对误差容限
        atol = 1e-5
        # 相对误差容限
        rtol = 0.05
        # 计算使用 a_m 的伪逆矩阵 a_p，指定绝对误差容限
        a_p = pinv(a_m, atol=atol, rtol=0.)
        # 计算差异 adiff1 = a @ a_p @ a - a
        adiff1 = a @ a_p @ a - a
        # 计算差异 adiff2 = a_m @ a_p @ a_m - a_m
        adiff2 = a_m @ a_p @ a_m - a_m
        # 确保 adiff1 的 Frobenius 范数约为 5e-4，指定绝对误差容限
        assert_allclose(np.linalg.norm(adiff1), 5e-4, atol=5.e-4)
        # 确保 adiff2 的 Frobenius 范数约为 5e-14，指定绝对误差容限
        assert_allclose(np.linalg.norm(adiff2), 5e-14, atol=5.e-14)

        # 再次计算，但是通过相对误差容限去掉另一个奇异值约为 4.234
        a_p = pinv(a_m, atol=atol, rtol=rtol)
        # 计算差异 adiff1 = a @ a_p @ a - a
        adiff1 = a @ a_p @ a - a
        # 计算差异 adiff2 = a_m @ a_p @ a_m - a_m
        adiff2 = a_m @ a_p @ a_m - a_m
        # 确保 adiff1 的 Frobenius 范数约为 4.233，指定相对误差容限
        assert_allclose(np.linalg.norm(adiff1), 4.233, rtol=0.01)
        # 确保 adiff2 的 Frobenius 范数约为 4.233，指定相对误差容限
        assert_allclose(np.linalg.norm(adiff2), 4.233, rtol=0.01)

    @pytest.mark.parametrize('dt', [float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 创建一个空的数组 a，指定数据类型 dt
        a = np.empty((0, 0), dtype=dt)
        # 计算空数组 a 的伪逆矩阵 a_pinv
        a_pinv = pinv(a)
        # 确保 a_pinv 的大小为 0
        assert a_pinv.size == 0
        # 确保 a_pinv 的数据类型与单位矩阵的数据类型相同
        assert a_pinv.dtype == pinv(np.eye(2, dtype=dt)).dtype
# 定义一个测试类 TestPinvSymmetric，用于测试 pinvh 函数的不同情况
class TestPinvSymmetric:

    # 在每个测试方法运行前调用的设置方法，设置随机种子为 1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试 pinvh 函数处理实数情况
    def test_simple_real(self):
        # 创建一个实数类型的数组 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        # 计算 a 与其转置的乘积，使其成为对称矩阵
        a = np.dot(a, a.T)
        # 计算矩阵 a 的伪逆
        a_pinv = pinvh(a)
        # 断言：a 乘以其伪逆应该接近单位矩阵
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    # 测试 pinvh 处理非正矩阵的情况
    def test_nonpositive(self):
        # 创建一个实数类型的数组 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        # 计算 a 与其转置的乘积，使其成为对称矩阵
        a = np.dot(a, a.T)
        # 对矩阵 a 进行奇异值分解
        u, s, vt = np.linalg.svd(a)
        # 将第一个奇异值取反，得到非正对称奇异矩阵
        s[0] *= -1
        a = np.dot(u * s, vt)  # a 现在是对称的非正奇异矩阵
        # 计算矩阵 a 的伪逆
        a_pinv = pinv(a)
        # 计算矩阵 a 的厄米特伪逆
        a_pinvh = pinvh(a)
        # 断言：a 的伪逆应该接近其厄米特伪逆
        assert_array_almost_equal(a_pinv, a_pinvh)

    # 测试 pinvh 函数处理复数情况
    def test_simple_complex(self):
        # 创建一个复数类型的数组 a
        a = (array([[1, 2, 3], [4, 5, 6], [7, 8, 10]],
             dtype=float) + 1j * array([[10, 8, 7], [6, 5, 4], [3, 2, 1]],
                                       dtype=float))
        # 计算 a 与其共轭转置的乘积，使其成为对称复数矩阵
        a = np.dot(a, a.conj().T)
        # 计算复数矩阵 a 的厄米特伪逆
        a_pinv = pinvh(a)
        # 断言：a 乘以其伪逆应该接近单位矩阵
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    # 测试 pinvh 函数处理传入原生列表作为参数的情况
    def test_native_list_argument(self):
        # 创建一个实数类型的数组 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        # 计算 a 与其转置的乘积，使其成为对称矩阵
        a = np.dot(a, a.T)
        # 将数组 a 转换为原生列表，然后计算其伪逆
        a_pinv = pinvh(a.tolist())
        # 断言：a 乘以其伪逆应该接近单位矩阵
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    # 测试 pinvh 函数处理存在零特征值的情况
    def test_zero_eigenvalue(self):
        # 创建一个实数类型的数组 a，包含一个零特征值
        a = np.array([[1,-1, 0], [-1, 2, -1], [0, -1, 1]])
        # 计算矩阵 a 的伪逆
        p = pinvh(a)
        # 断言：a 乘以其伪逆应该接近于伪逆本身，使用相对公差 1e-15
        assert_allclose(p @ a @ p, p, atol=1e-15)
        # 断言：a 乘以其伪逆应该接近于 a 本身，使用相对公差 1e-15
        assert_allclose(a @ p @ a, a, atol=1e-15)

    # 测试 pinvh 函数处理 atol 和 rtol 参数的情况
    def test_atol_rtol(self):
        n = 12
        # 获取一个随机正交矩阵用于重排
        q, _ = qr(np.random.rand(n, n))
        # 创建一个对角矩阵 a，包含不同大小的对角元素
        a = np.diag([4, 3, 2, 1, 0.99e-4, 0.99e-5] + [0.99e-6]*(n-6))
        a = q.T @ a @ q
        # 创建一个修改后的对角矩阵 a_m，将较小对角元素置为零
        a_m = np.diag([4, 3, 2, 1, 0.99e-4, 0.] + [0.]*(n-6))
        a_m = q.T @ a_m @ q
        atol = 1e-5
        rtol = (4.01e-4 - 4e-5)/4
        # 只使用绝对截断，使得 a_p 逼近 a_modified
        a_p = pinvh(a, atol=atol, rtol=0.)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        # 现在 adiff1 应在 atol 值附近波动，因为截断
        # adiff2 应该非常小
        assert_allclose(norm(adiff1), atol, rtol=0.1)
        assert_allclose(norm(adiff2), 1e-12, atol=1e-11)

        # 现在使用 rtol 取消 atol 值
        a_p = pinvh(a, atol=atol, rtol=rtol)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        # adiff1 和 adiff2 应升高到 ~1e-4，因为不匹配
        assert_allclose(norm(adiff1), 1e-4, rtol=0.1)
        assert_allclose(norm(adiff2), 1e-4, rtol=0.1)

    # 使用参数化测试 dt 测试 pinvh 函数处理空矩阵的情况
    @pytest.mark.parametrize('dt', [float, np.float32, complex, np.complex64])
    def test_empty(self, dt):
        # 创建一个空的数组 a，指定数据类型为 dt
        a = np.empty((0, 0), dtype=dt)
        # 计算空数组 a 的伪逆
        a_pinv = pinvh(a)
        # 断言：a 的伪逆应该是一个空数组
        assert a_pinv.size == 0
        # 断言：a 的伪逆的数据类型应该与单位矩阵的数据类型一致
        assert a_pinv.dtype == pinv(np.eye(2, dtype=dt)).dtype
@pytest.mark.parametrize('scale', (1e-20, 1., 1e20))
@pytest.mark.parametrize('pinv_', (pinv, pinvh))
def test_auto_rcond(scale, pinv_):
    # 创建一个2x2的 numpy 数组 x，元素根据参数 scale 变化
    x = np.array([[1, 0], [0, 1e-10]]) * scale
    # 创建预期的逆矩阵 expected，对角线元素为 x 的对角线元素的倒数
    expected = np.diag(1. / np.diag(x))
    # 计算 x 的伪逆矩阵，根据参数 pinv_ 来选择具体的伪逆函数
    x_inv = pinv_(x)
    # 断言 x_inv 是否接近于 expected
    assert_allclose(x_inv, expected)


class TestVectorNorms:

    def test_types(self):
        # 遍历所有浮点数类型的数据类型
        for dtype in np.typecodes['AllFloat']:
            # 创建一个浮点数类型的数组 x
            x = np.array([1, 2, 3], dtype=dtype)
            # 计算容差 tol，取 1e-15 和 np.finfo(dtype).eps.real * 20 中较大的值
            tol = max(1e-15, np.finfo(dtype).eps.real * 20)
            # 断言 norm(x) 是否接近于 sqrt(14)，相对误差为 rtol=tol
            assert_allclose(norm(x), np.sqrt(14), rtol=tol)
            # 断言 norm(x, 2) 是否接近于 sqrt(14)，相对误差为 rtol=tol
            assert_allclose(norm(x, 2), np.sqrt(14), rtol=tol)

        # 遍历所有复数类型的数据类型
        for dtype in np.typecodes['Complex']:
            # 创建一个复数类型的数组 x
            x = np.array([1j, 2j, 3j], dtype=dtype)
            # 计算容差 tol，取 1e-15 和 np.finfo(dtype).eps.real * 20 中较大的值
            tol = max(1e-15, np.finfo(dtype).eps.real * 20)
            # 断言 norm(x) 是否接近于 sqrt(14)，相对误差为 rtol=tol
            assert_allclose(norm(x), np.sqrt(14), rtol=tol)
            # 断言 norm(x, 2) 是否接近于 sqrt(14)，相对误差为 rtol=tol
            assert_allclose(norm(x, 2), np.sqrt(14), rtol=tol)

    def test_overflow(self):
        # 对于溢出情况，相比 numpy 的 norm，这个实现更安全
        a = array([1e20], dtype=float32)
        assert_almost_equal(norm(a), a)

    def test_stable(self):
        # 比 numpy 的 norm 更稳定
        a = array([1e4] + [1]*10000, dtype=float32)
        try:
            # 使用双精度计算 snrm，误差应在 1e-2 以内
            assert_allclose(norm(a) - 1e4, 0.5, atol=1e-2)
        except AssertionError:
            # 如果双精度计算不可用，回退到单精度结果，应等于 np.linalg.norm 的结果
            msg = ": Result should equal either 0.0 or 0.5 (depending on " \
                  "implementation of snrm2)."
            assert_almost_equal(norm(a) - 1e4, 0.0, err_msg=msg)

    def test_zero_norm(self):
        # 断言 norm([1, 0, 3], 0) 是否等于 2
        assert_equal(norm([1, 0, 3], 0), 2)
        # 断言 norm([1, 2, 3], 0) 是否等于 3
        assert_equal(norm([1, 2, 3], 0), 3)

    def test_axis_kwd(self):
        # 创建一个三维数组 a
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        # 断言 norm(a, axis=1) 是否接近于指定值
        assert_allclose(norm(a, axis=1), [[3.60555128, 4.12310563]] * 2)
        # 断言 norm(a, 1, axis=1) 是否等于指定值
        assert_allclose(norm(a, 1, axis=1), [[5.] * 2] * 2)

    def test_keepdims_kwd(self):
        # 创建一个三维数组 a
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        # 计算 norm(a, axis=1, keepdims=True)，并断言其接近于指定值
        b = norm(a, axis=1, keepdims=True)
        assert_allclose(b, [[[3.60555128, 4.12310563]]] * 2)
        # 断言 b 的形状为 (2, 1, 2)
        assert_(b.shape == (2, 1, 2))
        # 断言 norm(a, 1, axis=2, keepdims=True) 是否接近于指定值
        assert_allclose(norm(a, 1, axis=2, keepdims=True), [[[3.], [7.]]] * 2)

    @pytest.mark.skipif(not HAS_ILP64, reason="64-bit BLAS required")
    def test_large_vector(self):
        # 检查内存是否足够
        check_free_memory(free_mb=17000)
        # 创建一个长度为 2**31 的零数组 x，数据类型为 np.float64
        x = np.zeros([2**31], dtype=np.float64)
        x[-1] = 1
        # 计算数组 x 的 norm，并断言其接近于 1.0
        res = norm(x)
        del x
        assert_allclose(res, 1.0)


class TestMatrixNorms:
    # 测试不同矩阵规范的函数
    def test_matrix_norms(self):
        # 设置随机种子以确保可复现性
        np.random.seed(1234)
        # 针对不同的形状(n, m)和数据类型t进行测试
        for n, m in (1, 1), (1, 3), (3, 1), (4, 4), (4, 5), (5, 4):
            for t in np.float32, np.float64, np.complex64, np.complex128, np.int64:
                # 生成具有特定形状和数据类型的随机矩阵A
                A = 10 * np.random.randn(n, m).astype(t)
                # 如果A的数据类型是复数类型，将A设为复数并标记t_high为np.complex128
                if np.issubdtype(A.dtype, np.complexfloating):
                    A = (A + 10j * np.random.randn(n, m)).astype(t)
                    t_high = np.complex128
                else:
                    t_high = np.float64
                # 对于每种规范'order'，计算实际的矩阵规范和期望的矩阵规范
                for order in (None, 'fro', 1, -1, 2, -2, np.inf, -np.inf):
                    actual = norm(A, ord=order)
                    desired = np.linalg.norm(A, ord=order)
                    # SciPy可能返回更高精度的矩阵规范，这是使用LAPACK的结果
                    if not np.allclose(actual, desired):
                        # 如果实际结果和期望结果不相等，则将期望的矩阵规范基于t_high进行计算
                        desired = np.linalg.norm(A.astype(t_high), ord=order)
                        # 使用assert_allclose验证实际结果和修正后的期望结果
                        assert_allclose(actual, desired)

    # 测试带有axis关键字的函数
    def test_axis_kwd(self):
        # 创建一个三维数组a
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        # 计算沿指定轴(axis=(1, 0))的无穷范数b
        b = norm(a, ord=np.inf, axis=(1, 0))
        # 计算通过交换轴后的数组a的无穷范数c
        c = norm(np.swapaxes(a, 0, 1), ord=np.inf, axis=(0, 1))
        # 计算沿axis=(0, 1)的一范数d
        d = norm(a, ord=1, axis=(0, 1))
        # 使用assert_allclose验证b, c, d之间的近似性
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        # 确保b, c, d具有相同的形状
        assert_(b.shape == c.shape == d.shape)
        # 计算沿axis=(1, 0)的一范数b
        b = norm(a, ord=1, axis=(1, 0))
        # 计算通过交换轴后的数组a的一范数c
        c = norm(np.swapaxes(a, 0, 1), ord=1, axis=(0, 1))
        # 计算沿axis=(0, 1)的无穷范数d
        d = norm(a, ord=np.inf, axis=(0, 1))
        # 使用assert_allclose验证b, c, d之间的近似性
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        # 确保b, c, d具有相同的形状

    # 测试带有keepdims关键字的函数
    def test_keepdims_kwd(self):
        # 创建一个四维数组a
        a = np.arange(120, dtype='d').reshape(2, 3, 4, 5)
        # 计算沿axis=(1, 0)的无穷范数b，并保持维度
        b = norm(a, ord=np.inf, axis=(1, 0), keepdims=True)
        # 计算沿axis=(0, 1)的一范数c，并保持维度
        c = norm(a, ord=1, axis=(0, 1), keepdims=True)
        # 使用assert_allclose验证b和c之间的近似性
        assert_allclose(b, c)
        # 确保b和c具有相同的形状

    # 测试空数组的函数
    def test_empty(self):
        # 创建一个空的2x0数组a
        a = np.empty((0, 0))
        # 验证对空数组的规范为0
        assert_allclose(norm(a), 0.)
        # 验证沿axis=0的规范为全0数组
        assert_allclose(norm(a, axis=0), np.zeros((0,)))
        # 验证保持维度后的规范为1x1的全0数组
        assert_allclose(norm(a, keepdims=True), np.zeros((1, 1)))

        # 创建一个空的2x3数组a
        a = np.empty((0, 3))
        # 验证对空数组的规范为0
        assert_allclose(norm(a), 0.)
        # 验证沿axis=0的规范为3个元素全0数组
        assert_allclose(norm(a, axis=0), np.zeros((3,)))
        # 验证保持维度后的规范为1x1的全0数组
        assert_allclose(norm(a, keepdims=True), np.zeros((1, 1)))
class TestOverwrite:
    # 定义测试类 TestOverwrite
    def test_solve(self):
        # 测试 solve 函数，验证不覆盖现有内容
        assert_no_overwrite(solve, [(3, 3), (3,)])

    def test_solve_triangular(self):
        # 测试 solve_triangular 函数，验证不覆盖现有内容
        assert_no_overwrite(solve_triangular, [(3, 3), (3,)])

    def test_solve_banded(self):
        # 测试 solve_banded 函数，验证不覆盖现有内容
        assert_no_overwrite(lambda ab, b: solve_banded((2, 1), ab, b),
                            [(4, 6), (6,)])

    def test_solveh_banded(self):
        # 测试 solveh_banded 函数，验证不覆盖现有内容
        assert_no_overwrite(solveh_banded, [(2, 6), (6,)])

    def test_inv(self):
        # 测试 inv 函数，验证不覆盖现有内容
        assert_no_overwrite(inv, [(3, 3)])

    def test_det(self):
        # 测试 det 函数，验证不覆盖现有内容
        assert_no_overwrite(det, [(3, 3)])

    def test_lstsq(self):
        # 测试 lstsq 函数，验证不覆盖现有内容
        assert_no_overwrite(lstsq, [(3, 2), (3,)])

    def test_pinv(self):
        # 测试 pinv 函数，验证不覆盖现有内容
        assert_no_overwrite(pinv, [(3, 3)])

    def test_pinvh(self):
        # 测试 pinvh 函数，验证不覆盖现有内容
        assert_no_overwrite(pinvh, [(3, 3)])


class TestSolveCirculant:

    def test_basic1(self):
        # 测试基本情况1
        c = np.array([1, 2, 3, 5])
        b = np.array([1, -1, 1, 0])
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_basic2(self):
        # 测试基本情况2，其中 b 是一个二维矩阵
        c = np.array([1, 2, -3, -5])
        b = np.arange(12).reshape(4, 3)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_basic3(self):
        # 测试基本情况3，其中 b 是一个三维矩阵
        c = np.array([1, 2, -3, -5])
        b = np.arange(24).reshape(4, 3, 2)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_complex(self):
        # 复杂情况，其中 b 和 c 均为复数
        c = np.array([1+2j, -3, 4j, 5])
        b = np.arange(8).reshape(4, 2) + 0.5j
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_random_b_and_c(self):
        # 随机 b 和 c 的情况
        np.random.seed(54321)
        c = np.random.randn(50)
        b = np.random.randn(50)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_singular(self):
        # c 形成奇异循环矩阵的情况
        c = np.array([1, 1, 0, 0])
        b = np.array([1, 2, 3, 4])
        x = solve_circulant(c, b, singular='lstsq')
        y, res, rnk, s = lstsq(circulant(c), b)
        assert_allclose(x, y)
        assert_raises(LinAlgError, solve_circulant, x, y)
    def test_axis_args(self):
        # Test use of caxis, baxis and outaxis.

        # c has shape (2, 1, 4)
        c = np.array([[[-1, 2.5, 3, 3.5]], [[1, 6, 6, 6.5]]])

        # b has shape (3, 4)
        b = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, -1, 0, 0]])

        # Calculate x using solve_circulant with baxis=1
        x = solve_circulant(c, b, baxis=1)
        assert_equal(x.shape, (4, 2, 3))
        
        # Prepare expected output based on circulant matrices of c and b.T
        expected = np.empty_like(x)
        expected[:, 0, :] = solve(circulant(c[0]), b.T)
        expected[:, 1, :] = solve(circulant(c[1]), b.T)
        assert_allclose(x, expected)

        # Calculate x using solve_circulant with baxis=1 and outaxis=-1
        x = solve_circulant(c, b, baxis=1, outaxis=-1)
        assert_equal(x.shape, (2, 3, 4))
        assert_allclose(np.moveaxis(x, -1, 0), expected)

        # Calculate x using np.swapaxes(c, 1, 2) and b.T with caxis=1
        x = solve_circulant(np.swapaxes(c, 1, 2), b.T, caxis=1)
        assert_equal(x.shape, (4, 2, 3))
        assert_allclose(x, expected)

    def test_native_list_arguments(self):
        # Same as test_basic1 using python's native list.
        c = [1, 2, 3, 5]
        b = [1, -1, 1, 0]

        # Calculate x using solve_circulant with lists c and b
        x = solve_circulant(c, b)
        
        # Calculate y using solve with circulant matrices of c and b
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    @pytest.mark.parametrize('dt_c', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt_c, dt_b):
        c = np.array([], dtype=dt_c)
        b = np.array([], dtype=dt_b)

        # Calculate x using solve_circulant with empty arrays c and b
        x = solve_circulant(c, b)
        assert x.shape == (0,)
        assert x.dtype == solve_circulant(np.arange(3, dtype=dt_c),
                                          np.ones(3, dtype=dt_b)).dtype

        b = np.empty((0, 0), dtype=dt_b)

        # Calculate x1 using solve_circulant with empty array c and 2D empty array b
        x1 = solve_circulant(c, b)
        assert x1.shape == (0, 0)
        assert x1.dtype == x.dtype
# 定义名为 TestMatrix_Balance 的测试类
class TestMatrix_Balance:

    # 测试当输入参数为字符串时，是否会抛出 ValueError 异常
    def test_string_arg(self):
        assert_raises(ValueError, matrix_balance, 'Some string for fail')

    # 测试当输入参数包含无穷大或 NaN 时，是否会抛出 ValueError 异常
    def test_infnan_arg(self):
        assert_raises(ValueError, matrix_balance,
                      np.array([[1, 2], [3, np.inf]]))
        assert_raises(ValueError, matrix_balance,
                      np.array([[1, 2], [3, np.nan]]))

    # 测试矩阵平衡函数对于简单矩阵的缩放效果
    def test_scaling(self):
        _, y = matrix_balance(np.array([[1000, 1], [1000, 0]]))
        # 使用 LAPACK 3.5.0 前后的结果进行比较，主要检查对数缩放的正确性
        assert_allclose(np.diff(np.log2(np.diag(y))), [5])

    # 测试矩阵平衡函数在特定顺序下的效果
    def test_scaling_order(self):
        A = np.array([[1, 0, 1e-4], [1, 1, 1e-2], [1e4, 1e2, 1]])
        x, y = matrix_balance(A)
        # 检查平衡后矩阵的解是否正确
        assert_allclose(solve(y, A).dot(y), x)

    # 测试矩阵平衡函数在分离模式下的效果
    def test_separate(self):
        _, (y, z) = matrix_balance(np.array([[1000, 1], [1000, 0]]),
                                   separate=1)
        # 检查对数缩放的正确性
        assert_equal(np.diff(np.log2(y)), [5])
        # 检查 z 是否与预期的数组近似相等
        assert_allclose(z, np.arange(2))

    # 测试矩阵平衡函数在排列和分离模式下的效果
    def test_permutation(self):
        A = block_diag(np.ones((2, 2)), np.tril(np.ones((2, 2))),
                       np.ones((3, 3)))
        x, (y, z) = matrix_balance(A, separate=1)
        # 检查平衡后的 y 是否全部接近于 1
        assert_allclose(y, np.ones_like(y))
        # 检查 z 是否与预期的数组近似相等
        assert_allclose(z, np.array([0, 1, 6, 5, 4, 3, 2]))

    # 测试矩阵平衡函数在排列和缩放模式下的效果
    def test_perm_and_scaling(self):
        # 不同情况下的测试用例
        cases = (
                 # Case 0
                 np.array([[0., 0., 0., 0., 0.000002],
                           [0., 0., 0., 0., 0.],
                           [2., 2., 0., 0., 0.],
                           [2., 2., 0., 0., 0.],
                           [0., 0., 0.000002, 0., 0.]]),
                 # Case 1 user reported GH-7258
                 np.array([[-0.5, 0., 0., 0.],
                           [0., -1., 0., 0.],
                           [1., 0., -0.5, 0.],
                           [0., 1., 0., -1.]]),
                 # Case 2 user reported GH-7258
                 np.array([[-3., 0., 1., 0.],
                           [-1., -1., -0., 1.],
                           [-3., -0., -0., 0.],
                           [-1., -0., 1., -1.]])
                 )

        # 遍历所有测试用例
        for A in cases:
            x, y = matrix_balance(A)
            x, (s, p) = matrix_balance(A, separate=1)
            # 构建排列的逆映射 ip
            ip = np.empty_like(p)
            ip[p] = np.arange(A.shape[0])
            # 检查平衡后的 y 是否与预期的对角线矩阵接近
            assert_allclose(y, np.diag(s)[ip, :])
            # 检查平衡后的矩阵乘积是否与原始矩阵接近
            assert_allclose(solve(y, A).dot(y), x)

    # 使用 pytest 的参数化装饰器，测试不同数据类型的输入
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    # 定义一个测试方法，用于测试空数组的平衡性功能
    def test_empty(self, dt):
        # 创建一个空的 NumPy 数组 `a`，数据类型为 `dt`
        a = np.empty((0, 0), dtype=dt)
        # 调用 `matrix_balance` 函数对数组 `a` 进行平衡处理，返回 `b` 和 `t`
        b, t = matrix_balance(a)

        # 断言 `b` 的大小为 0
        assert b.size == 0
        # 断言 `t` 的大小为 0
        assert t.size == 0

        # 对单位矩阵调用 `matrix_balance`，返回 `b_n` 和 `t_n`
        b_n, t_n = matrix_balance(np.eye(2, dtype=dt))
        # 断言 `b` 的数据类型与 `b_n` 的数据类型相同
        assert b.dtype == b_n.dtype
        # 断言 `t` 的数据类型与 `t_n` 的数据类型相同
        assert t.dtype == t_n.dtype

        # 调用 `matrix_balance` 函数，并指定 `separate=True`，返回 `b` 和 `(scale, perm)`
        b, (scale, perm) = matrix_balance(a, separate=True)
        # 断言 `b` 的大小为 0
        assert b.size == 0
        # 断言 `scale` 的大小为 0
        assert scale.size == 0
        # 断言 `perm` 的大小为 0
        assert perm.size == 0

        # 再次调用 `matrix_balance` 函数，并指定 `separate=True`，返回 `b_n` 和 `(scale_n, perm_n)`
        b_n, (scale_n, perm_n) = matrix_balance(a, separate=True)
        # 断言 `b` 的数据类型与 `b_n` 的数据类型相同
        assert b.dtype == b_n.dtype
        # 断言 `scale` 的数据类型与 `scale_n` 的数据类型相同
        assert scale.dtype == scale_n.dtype
        # 断言 `perm` 的数据类型与 `perm_n` 的数据类型相同
        assert perm.dtype == perm_n.dtype
```