# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_solve_toeplitz.py`

```
"""Test functions for linalg._solve_toeplitz module
"""
# 导入所需的库和模块
import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose

# 导入 pytest 相关的断言函数
import pytest
from pytest import raises as assert_raises


def test_solve_equivalence():
    # 对于 Toeplitz 矩阵，solve_toeplitz() 应当等价于 solve()。
    random = np.random.RandomState(1234)
    for n in (1, 2, 3, 10):
        c = random.randn(n)
        if random.rand() < 0.5:
            c = c + 1j * random.randn(n)
        r = random.randn(n)
        if random.rand() < 0.5:
            r = r + 1j * random.randn(n)
        y = random.randn(n)
        if random.rand() < 0.5:
            y = y + 1j * random.randn(n)

        # 当提供列和行时，检查等价性。
        actual = solve_toeplitz((c,r), y)
        desired = solve(toeplitz(c, r=r), y)
        assert_allclose(actual, desired)

        # 当只提供列而不提供行时，检查等价性。
        actual = solve_toeplitz(c, b=y)
        desired = solve(toeplitz(c), y)
        assert_allclose(actual, desired)


def test_multiple_rhs():
    random = np.random.RandomState(1234)
    c = random.randn(4)
    r = random.randn(4)
    for offset in [0, 1j]:
        for yshape in ((4,), (4, 3), (4, 3, 2)):
            y = random.randn(*yshape) + offset
            actual = solve_toeplitz((c,r), b=y)
            desired = solve(toeplitz(c, r=r), y)
            assert_equal(actual.shape, yshape)
            assert_equal(desired.shape, yshape)
            assert_allclose(actual, desired)


def test_native_list_arguments():
    # 使用原生列表作为参数进行测试
    c = [1,2,4,7]
    r = [1,3,9,12]
    y = [5,1,4,2]
    actual = solve_toeplitz((c,r), y)
    desired = solve(toeplitz(c, r=r), y)
    assert_allclose(actual, desired)


def test_zero_diag_error():
    # 当对角线元素为零时，Levinson-Durbin 实现会抛出 LinAlgError。
    random = np.random.RandomState(1234)
    n = 4
    c = random.randn(n)
    r = random.randn(n)
    y = random.randn(n)
    c[0] = 0
    assert_raises(np.linalg.LinAlgError,
        solve_toeplitz, (c, r), b=y)


def test_wikipedia_counterexample():
    # Levinson-Durbin 实现在其他情况下也会失败。
    # 此示例来自 Wikipedia 文章的讨论页。
    random = np.random.RandomState(1234)
    c = [2, 2, 1]
    y = random.randn(3)
    assert_raises(np.linalg.LinAlgError, solve_toeplitz, c, b=y)


def test_reflection_coeffs():
    # 检查部分解是否由反射系数给出

    random = np.random.RandomState(1234)
    y_d = random.randn(10)
    y_z = random.randn(10) + 1j
    reflection_coeffs_d = [1]
    reflection_coeffs_z = [1]
    for i in range(2, 10):
        reflection_coeffs_d.append(solve_toeplitz(y_d[:(i-1)], b=y_d[1:i])[-1])
        reflection_coeffs_z.append(solve_toeplitz(y_z[:(i-1)], b=y_z[1:i])[-1])
    # 将数组 y_d 的倒数第2到第1个元素与倒数第1个元素连接起来，形成新的数组
    y_d_concat = np.concatenate((y_d[-2:0:-1], y_d[:-1]))
    # 将数组 y_z 的倒数第2到第1个元素的共轭与倒数第1个元素连接起来，形成新的数组
    y_z_concat = np.concatenate((y_z[-2:0:-1].conj(), y_z[:-1]))
    # 使用 Levinson-Durbin 递归算法计算 y_d_concat 的自相关系数，并返回预测误差，ref_d 为返回的自相关系数
    _, ref_d = levinson(y_d_concat, b=y_d[1:])
    # 使用 Levinson-Durbin 递归算法计算 y_z_concat 的自相关系数，并返回预测误差，ref_z 为返回的自相关系数
    _, ref_z = levinson(y_z_concat, b=y_z[1:])

    # 使用 assert_allclose 函数检查 reflection_coeffs_d 是否与 ref_d 的前 n-1 个元素在允许误差范围内相等
    assert_allclose(reflection_coeffs_d, ref_d[:-1])
    # 使用 assert_allclose 函数检查 reflection_coeffs_z 是否与 ref_z 的前 n-1 个元素在允许误差范围内相等
    assert_allclose(reflection_coeffs_z, ref_z[:-1])
@pytest.mark.xfail(reason='Instability of Levinson iteration')
# 标记此测试为预期失败，原因是Levinson迭代的不稳定性
def test_unstable():
    # 这是一个“高斯Toeplitz矩阵”，如I. Gohbert、T. Kailath和V. Olshevsky在“具有位移结构矩阵的部分主元高斯消元的快速算法”中提到的示例2
    # 《计算数学》，64，212（1995），第1557-1576页，对Levinson递归可能不稳定。
    # 其他快速Toeplitz求解器如GKO或Burg应该更好。

    random = np.random.RandomState(1234)
    n = 100
    c = 0.9 ** (np.arange(n)**2)
    y = random.randn(n)

    solution1 = solve_toeplitz(c, b=y)
    solution2 = solve(toeplitz(c), y)

    # 断言两种解法的结果接近
    assert_allclose(solution1, solution2)


@pytest.mark.parametrize('dt_c', [int, float, np.float32, complex, np.complex64])
@pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
# 参数化测试，对不同数据类型dt_c和dt_b进行多组测试
def test_empty(dt_c, dt_b):
    c = np.array([], dtype=dt_c)
    b = np.array([], dtype=dt_b)
    x = solve_toeplitz(c, b)
    # 断言解的形状为(0,)
    assert x.shape == (0,)
    # 断言解的数据类型与solve_toeplitz(np.array([2, 1], dtype=dt_c), np.ones(2, dtype=dt_b))的数据类型相同
    assert x.dtype == solve_toeplitz(np.array([2, 1], dtype=dt_c),
                                      np.ones(2, dtype=dt_b)).dtype

    b = np.empty((0, 0), dtype=dt_b)
    x1 = solve_toeplitz(c, b)
    # 断言解的形状为(0, 0)
    assert x1.shape == (0, 0)
    # 断言解的数据类型与上一个解x的数据类型相同
    assert x1.dtype == x.dtype
```