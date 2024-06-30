# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_cholesky.py`

```
import pytest  # 导入 pytest 库，用于编写和运行测试
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.testing import assert_array_almost_equal  # 导入 NumPy 测试模块中的数组近似相等断言
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 断言并重命名为 assert_raises

from numpy import array, transpose, dot, conjugate, zeros_like, empty  # 导入 NumPy 中的数组、转置、点积、共轭等函数
from numpy.random import random  # 导入 NumPy 中的随机数生成函数
from scipy.linalg import (cholesky, cholesky_banded, cho_solve_banded,
     cho_factor, cho_solve)  # 导入 SciPy 线性代数模块中的 Cholesky 分解相关函数

from scipy.linalg._testutils import assert_no_overwrite  # 导入 SciPy 线性代数测试工具模块中的不覆写断言


class TestCholesky:
    
    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]  # 创建一个对称矩阵 a
        c = cholesky(a)  # 对矩阵 a 进行 Cholesky 分解，得到下三角矩阵 c
        assert_array_almost_equal(dot(transpose(c), c), a)  # 断言 c 的转置与 c 的矩阵乘积近似等于原始矩阵 a
        c = transpose(c)  # 将 c 转置
        a = dot(c, transpose(c))  # 重新计算原始矩阵 a，应与之前的 a 近似相等
        assert_array_almost_equal(cholesky(a, lower=1), c)  # 断言再次对 a 进行 Cholesky 分解，应得到下三角矩阵 c

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [3, 3, 6]]  # 创建一个对称矩阵 a
        c = cholesky(a, check_finite=False)  # 使用 check_finite=False 对 a 进行 Cholesky 分解，不检查有限性
        assert_array_almost_equal(dot(transpose(c), c), a)  # 断言 c 的转置与 c 的矩阵乘积近似等于原始矩阵 a
        c = transpose(c)  # 将 c 转置
        a = dot(c, transpose(c))  # 重新计算原始矩阵 a，应与之前的 a 近似相等
        assert_array_almost_equal(cholesky(a, lower=1, check_finite=False), c)  # 断言再次对 a 进行 Cholesky 分解，应得到下三角矩阵 c

    def test_simple_complex(self):
        m = array([[3+1j, 3+4j, 5], [0, 2+2j, 2+7j], [0, 0, 7+4j]])  # 创建一个复数矩阵 m
        a = dot(transpose(conjugate(m)), m)  # 计算 m 的共轭转置与 m 的乘积，得到对称复数矩阵 a
        c = cholesky(a)  # 对复数矩阵 a 进行 Cholesky 分解，得到下三角矩阵 c
        a1 = dot(transpose(conjugate(c)), c)  # 计算 c 的共轭转置与 c 的乘积
        assert_array_almost_equal(a, a1)  # 断言 a 与 a1 近似相等
        c = transpose(c)  # 将 c 转置
        a = dot(c, transpose(conjugate(c)))  # 计算 c 与其共轭转置的乘积
        assert_array_almost_equal(cholesky(a, lower=1), c)  # 断言再次对 a 进行 Cholesky 分解，应得到下三角矩阵 c

    def test_random(self):
        n = 20  # 设置矩阵维度
        for k in range(2):  # 循环两次
            m = random([n, n])  # 生成随机的 n x n 数组 m
            for i in range(n):  # 遍历数组的每一行
                m[i, i] = 20*(.1+m[i, i])  # 对角线元素加上一个较大的值
            a = dot(transpose(m), m)  # 计算 m 的转置与 m 的乘积，得到对称矩阵 a
            c = cholesky(a)  # 对矩阵 a 进行 Cholesky 分解，得到下三角矩阵 c
            a1 = dot(transpose(c), c)  # 计算 c 的转置与 c 的乘积
            assert_array_almost_equal(a, a1)  # 断言 a 与 a1 近似相等
            c = transpose(c)  # 将 c 转置
            a = dot(c, transpose(c))  # 重新计算原始矩阵 a，应与之前的 a 近似相等
            assert_array_almost_equal(cholesky(a, lower=1), c)  # 断言再次对 a 进行 Cholesky 分解，应得到下三角矩阵 c

    def test_random_complex(self):
        n = 20  # 设置矩阵维度
        for k in range(2):  # 循环两次
            m = random([n, n])+1j*random([n, n])  # 生成随机的 n x n 复数数组 m
            for i in range(n):  # 遍历数组的每一行
                m[i, i] = 20*(.1+abs(m[i, i]))  # 对角线元素加上一个较大的值
            a = dot(transpose(conjugate(m)), m)  # 计算 m 的共轭转置与 m 的乘积，得到对称复数矩阵 a
            c = cholesky(a)  # 对复数矩阵 a 进行 Cholesky 分解，得到下三角矩阵 c
            a1 = dot(transpose(conjugate(c)), c)  # 计算 c 的共轭转置与 c 的乘积
            assert_array_almost_equal(a, a1)  # 断言 a 与 a1 近似相等
            c = transpose(c)  # 将 c 转置
            a = dot(c, transpose(conjugate(c)))  # 计算 c 与其共轭转置的乘积
            assert_array_almost_equal(cholesky(a, lower=1), c)  # 断言再次对 a 进行 Cholesky 分解，应得到下三角矩阵 c

    @pytest.mark.xslow  # 声明此测试用例为较慢的测试
    def test_int_overflow(self):
       # 回归测试 https://github.com/scipy/scipy/issues/17436
       # 问题是在清零未使用的三角部分时发生整数溢出
       n = 47_000  # 设置矩阵维度
       x = np.eye(n, dtype=np.float64, order='F')  # 创建 n x n 的单位矩阵 x
       x[:4, :4] = np.array([[4, -2, 3, -1],  # 更新矩阵 x 的前 4 行前 4 列的元素
                             [-2, 4, -3, 1],
                             [3, -3, 5, 0],
                             [-1, 1, 0, 5]])

       cholesky(x, check_finite=False, overwrite_a=True)  # 调用 Cholesky 分解函数，不检查有限性，允许覆写输入矩阵 x

    @pytest.mark.parametrize('dt', [int, float,
    # 使用 pytest 的参数化功能，为 dt_b 参数设定多个测试类型：整数、浮点数、NumPy 中的浮点数、复数、NumPy 中的复数
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    # 定义测试方法 test_empty，参数为 dt 和 dt_b
    def test_empty(self, dt, dt_b):
        # 创建一个空的数组 a，数据类型为 dt
        a = empty((0, 0), dtype=dt)

        # 对数组 a 进行 Cholesky 分解，得到结果 c
        c = cholesky(a)
        # 断言 Cholesky 分解结果 c 的形状为 (0, 0)
        assert c.shape == (0, 0)
        # 断言 Cholesky 分解结果 c 的数据类型与单位矩阵 np.eye(2, dtype=dt) 的数据类型相同
        assert c.dtype == cholesky(np.eye(2, dtype=dt)).dtype

        # 创建一个包含 c 和 True 的元组 c_and_lower
        c_and_lower = (c, True)
        # 创建一个空的 NumPy 数组 b，数据类型为 dt_b
        b = np.asarray([], dtype=dt_b)
        # 使用 Cholesky 分解结果 c_and_lower 对方程 cho_solve 进行求解，得到结果 x
        x = cho_solve(c_and_lower, b)
        # 断言结果 x 的形状为 (0,)
        assert x.shape == (0,)
        # 断言结果 x 的数据类型与单位矩阵 np.eye(2, dtype=dt) 和 np.ones(2, dtype=dt_b) 的数据类型相同
        assert x.dtype == cho_solve((np.eye(2, dtype=dt), True),
                                     np.ones(2, dtype=dt_b)).dtype

        # 创建一个空的数组 b，形状为 (0, 0)，数据类型为 dt_b
        b = empty((0, 0), dtype=dt_b)
        # 使用 Cholesky 分解结果 c_and_lower 对方程 cho_solve 进行求解，得到结果 x
        x = cho_solve(c_and_lower, b)
        # 断言结果 x 的形状为 (0, 0)
        assert x.shape == (0, 0)
        # 断言结果 x 的数据类型与单位矩阵 np.eye(2, dtype=dt) 和 np.ones(2, dtype=dt_b) 的数据类型相同
        assert x.dtype == cho_solve((np.eye(2, dtype=dt), True),
                                     np.ones(2, dtype=dt_b)).dtype

        # 创建不同类型的空数组：a1 为空的一维数组，a2 为包含一个空数组的二维数组，a3 和 a4 分别为空的一维和二维 Python 列表
        a1 = array([])
        a2 = array([[]])
        a3 = []
        a4 = [[]]
        # 针对每种数组类型，使用 cholesky 函数断言会抛出 ValueError 异常
        for x in ([a1, a2, a3, a4]):
            assert_raises(ValueError, cholesky, x)
class TestCholeskyBanded:
    """Tests for cholesky_banded() and cho_solve_banded."""

    def test_check_finite(self):
        # Symmetric positive definite banded matrix `a`
        a = array([[4.0, 1.0, 0.0, 0.0],
                   [1.0, 4.0, 0.5, 0.0],
                   [0.0, 0.5, 4.0, 0.2],
                   [0.0, 0.0, 0.2, 4.0]])
        # Banded storage form of `a`.
        ab = array([[-1.0, 1.0, 0.5, 0.2],
                    [4.0, 4.0, 4.0, 4.0]])
        # Perform Cholesky decomposition on `ab` without checking for finite numbers
        c = cholesky_banded(ab, lower=False, check_finite=False)
        # Initialize an upper triangular matrix `ufac` with zeros
        ufac = zeros_like(a)
        # Populate the diagonal of `ufac` with the last row of `c`
        ufac[list(range(4)), list(range(4))] = c[-1]
        # Populate the upper sub-diagonals of `ufac` with elements from `c`
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        # Assert that `a` is almost equal to the dot product of `ufac` transposed and `ufac`
        assert_array_almost_equal(a, dot(ufac.T, ufac))

        # Solving the linear system `c * x = b` where `b` is the given array
        b = array([0.0, 0.5, 4.2, 4.2])
        # Solve using the Cholesky factor `c`, ignoring finiteness of inputs
        x = cho_solve_banded((c, False), b, check_finite=False)
        # Assert that `x` is almost equal to the expected solution array
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_upper_real(self):
        # Symmetric positive definite banded matrix `a`
        a = array([[4.0, 1.0, 0.0, 0.0],
                   [1.0, 4.0, 0.5, 0.0],
                   [0.0, 0.5, 4.0, 0.2],
                   [0.0, 0.0, 0.2, 4.0]])
        # Banded storage form of `a`.
        ab = array([[-1.0, 1.0, 0.5, 0.2],
                    [4.0, 4.0, 4.0, 4.0]])
        # Perform Cholesky decomposition on `ab` assuming it's upper triangular
        c = cholesky_banded(ab, lower=False)
        # Initialize an upper triangular matrix `ufac` with zeros
        ufac = zeros_like(a)
        # Populate the diagonal of `ufac` with the last row of `c`
        ufac[list(range(4)), list(range(4))] = c[-1]
        # Populate the upper sub-diagonals of `ufac` with elements from `c`
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        # Assert that `a` is almost equal to the dot product of `ufac` transposed and `ufac`
        assert_array_almost_equal(a, dot(ufac.T, ufac))

        # Solving the linear system `c * x = b` where `b` is the given array
        b = array([0.0, 0.5, 4.2, 4.2])
        # Solve using the Cholesky factor `c`
        x = cho_solve_banded((c, False), b)
        # Assert that `x` is almost equal to the expected solution array
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_upper_complex(self):
        # Hermitian positive definite banded matrix `a`
        a = array([[4.0, 1.0, 0.0, 0.0],
                   [1.0, 4.0, 0.5, 0.0],
                   [0.0, 0.5, 4.0, -0.2j],
                   [0.0, 0.0, 0.2j, 4.0]])
        # Banded storage form of `a`.
        ab = array([[-1.0, 1.0, 0.5, -0.2j],
                    [4.0, 4.0, 4.0, 4.0]])
        # Perform Cholesky decomposition on `ab` assuming it's upper triangular
        c = cholesky_banded(ab, lower=False)
        # Initialize an upper triangular matrix `ufac` with zeros
        ufac = zeros_like(a)
        # Populate the diagonal of `ufac` with the last row of `c`
        ufac[list(range(4)), list(range(4))] = c[-1]
        # Populate the upper sub-diagonals of `ufac` with elements from `c`
        ufac[(0, 1, 2), (1, 2, 3)] = c[0, 1:]
        # Assert that `a` is almost equal to the dot product of the conjugate transpose of `ufac` and `ufac`
        assert_array_almost_equal(a, dot(ufac.conj().T, ufac))

        # Solving the linear system `c * x = b` where `b` is the given array
        b = array([0.0, 0.5, 4.0-0.2j, 0.2j + 4.0])
        # Solve using the Cholesky factor `c`
        x = cho_solve_banded((c, False), b)
        # Assert that `x` is almost equal to the expected solution array
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])
    def test_lower_real(self):
        # Symmetric positive definite banded matrix `a`
        a = array([[4.0, 1.0, 0.0, 0.0],
                   [1.0, 4.0, 0.5, 0.0],
                   [0.0, 0.5, 4.0, 0.2],
                   [0.0, 0.0, 0.2, 4.0]])
        # Banded storage form of `a`.
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 0.5, 0.2, -1.0]])
        # Perform Cholesky decomposition on the banded matrix `ab` with lower=True
        c = cholesky_banded(ab, lower=True)
        # Initialize a zero matrix `lfac` of the same shape as `a`
        lfac = zeros_like(a)
        # Fill the diagonal of `lfac` with the lower triangle of `c`
        lfac[list(range(4)), list(range(4))] = c[0]
        # Fill the subdiagonal of `lfac` with part of the second row of `c`
        lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
        # Assert that the matrix `a` is almost equal to `lfac` multiplied by its transpose
        assert_array_almost_equal(a, dot(lfac, lfac.T))

        # Solve the linear system `cb_and_lower` * `x` = `b` where `cb_and_lower` is `c` and lower=True
        b = array([0.0, 0.5, 4.2, 4.2])
        x = cho_solve_banded((c, True), b)
        # Assert that `x` is almost equal to [0.0, 0.0, 1.0, 1.0]
        assert_array_almost_equal(x, [0.0, 0.0, 1.0, 1.0])

    def test_lower_complex(self):
        # Hermitian positive definite banded matrix `a`
        a = array([[4.0, 1.0, 0.0, 0.0],
                   [1.0, 4.0, 0.5, 0.0],
                   [0.0, 0.5, 4.0, -0.2j],
                   [0.0, 0.0, 0.2j, 4.0]])
        # Banded storage form of `a`.
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 0.5, 0.2j, -1.0]])
        # Perform Cholesky decomposition on the banded matrix `ab` with lower=True
        c = cholesky_banded(ab, lower=True)
        # Initialize a zero matrix `lfac` of the same shape as `a`
        lfac = zeros_like(a)
        # Fill the diagonal of `lfac` with the lower triangle of `c`
        lfac[list(range(4)), list(range(4))] = c[0]
        # Fill the subdiagonal of `lfac` with part of the second row of `c`
        lfac[(1, 2, 3), (0, 1, 2)] = c[1, :3]
        # Assert that the matrix `a` is almost equal to `lfac` multiplied by its conjugate transpose
        assert_array_almost_equal(a, dot(lfac, lfac.conj().T))

        # Solve the linear system `cb_and_lower` * `x` = `b` where `cb_and_lower` is `c` and lower=True
        b = array([0.0, 0.5j, 3.8j, 3.8])
        x = cho_solve_banded((c, True), b)
        # Assert that `x` is almost equal to [0.0, 0.0, 1.0j, 1.0]
        assert_array_almost_equal(x, [0.0, 0.0, 1.0j, 1.0])

    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt, dt_b):
        # Create an empty array `ab` of shape (0, 0) with specified data type `dt`
        ab = empty((0, 0), dtype=dt)
        
        # Perform Cholesky decomposition on the empty matrix `ab`
        cb = cholesky_banded(ab)
        # Assert that the shape of `cb` is (0, 0)
        assert cb.shape == (0, 0)
        
        # Perform Cholesky decomposition on a specific 2x2 matrix of dtype `dt` and assign to `m`
        m = cholesky_banded(np.array([[0, 0], [1, 1]], dtype=dt))
        # Assert that the dtype of `cb` is the same as `m`
        assert cb.dtype == m.dtype
        
        # Define `cb_and_lower` as `(cb, True)`
        cb_and_lower = (cb, True)
        # Create an empty array `b` of dtype `dt_b`
        b = np.asarray([], dtype=dt_b)
        # Solve the linear system `cb_and_lower` * `x` = `b` and assign to `x`
        x = cho_solve_banded(cb_and_lower, b)
        # Assert that the shape of `x` is (0,)
        assert x.shape == (0,)
        
        # Determine the dtype of the result of solving the system `m` * `x` = `[1, 1]`
        dtype_nonempty = cho_solve_banded((m, True), np.ones(2, dtype=dt_b)).dtype
        # Assert that the dtype of `x` is equal to `dtype_nonempty`
        assert x.dtype == dtype_nonempty
        
        # Create an empty array `b` of shape (0, 0) with dtype `dt_b`
        b = empty((0, 0), dtype=dt_b)
        # Solve the linear system `cb_and_lower` * `x` = `b` and assign to `x`
        x = cho_solve_banded(cb_and_lower, b)
        # Assert that the shape of `x` is (0, 0)
        assert x.shape == (0, 0)
        # Assert that the dtype of `x` is equal to `dtype_nonempty`
        assert x.dtype == dtype_nonempty
# 定义一个测试类 TestOverwrite，用于测试 cholesky、cho_factor、cho_solve、cholesky_banded 和 cho_solve_banded 函数
class TestOverwrite:
    # 定义测试方法 test_cholesky，测试 cholesky 函数
    def test_cholesky(self):
        # 断言调用 assert_no_overwrite 函数，验证 cholesky 函数不会被覆盖，参数为 [(3, 3)]
        assert_no_overwrite(cholesky, [(3, 3)])

    # 定义测试方法 test_cho_factor，测试 cho_factor 函数
    def test_cho_factor(self):
        # 断言调用 assert_no_overwrite 函数，验证 cho_factor 函数不会被覆盖，参数为 [(3, 3)]
        assert_no_overwrite(cho_factor, [(3, 3)])

    # 定义测试方法 test_cho_solve，测试 cho_solve 函数
    def test_cho_solve(self):
        # 创建一个 3x3 的数组 x
        x = array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        # 对 x 进行 Cholesky 分解，得到 xcho
        xcho = cho_factor(x)
        # 断言调用 assert_no_overwrite 函数，验证 lambda 函数 cho_solve(xcho, b) 不会被覆盖，参数为 [(3,)]
        assert_no_overwrite(lambda b: cho_solve(xcho, b), [(3,)])

    # 定义测试方法 test_cholesky_banded，测试 cholesky_banded 函数
    def test_cholesky_banded(self):
        # 断言调用 assert_no_overwrite 函数，验证 cholesky_banded 函数不会被覆盖，参数为 [(2, 3)]
        assert_no_overwrite(cholesky_banded, [(2, 3)])

    # 定义测试方法 test_cho_solve_banded，测试 cho_solve_banded 函数
    def test_cho_solve_banded(self):
        # 创建一个 2x3 的数组 x
        x = array([[0, -1, -1], [2, 2, 2]])
        # 对 x 进行带状 Cholesky 分解，得到 xcho
        xcho = cholesky_banded(x)
        # 断言调用 assert_no_overwrite 函数，验证 lambda 函数 cho_solve_banded((xcho, False), b) 不会被覆盖，参数为 [(3,)]
        assert_no_overwrite(lambda b: cho_solve_banded((xcho, False), b),
                            [(3,)])

# 定义一个测试类 TestChoFactor，用于测试 cho_factor 函数
class TestChoFactor:
    # 使用 pytest 的参数化装饰器，测试空数组情况，参数为 int、float、np.float32、complex、np.complex64
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    # 定义测试方法 test_empty，测试 cho_factor 函数在空数组情况下的行为
    def test_empty(self, dt):
        # 创建一个空的 dtype 为 dt 的数组 a
        a = np.empty((0, 0), dtype=dt)
        # 对空数组 a 进行 Cholesky 分解，得到 x 和 lower
        x, lower = cho_factor(a)

        # 断言 x 的形状为 (0, 0)
        assert x.shape == (0, 0)

        # 对单位矩阵进行 Cholesky 分解，得到 xx 和 lower
        xx, lower = cho_factor(np.eye(2, dtype=dt))
        # 断言 x 和 xx 的数据类型相同
        assert x.dtype == xx.dtype
```