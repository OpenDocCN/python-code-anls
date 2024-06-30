# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_minres.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy.linalg import norm  # 导入 norm 函数，用于计算向量或矩阵的范数
from numpy.testing import assert_equal, assert_allclose, assert_  # 导入测试相关函数
from scipy.sparse.linalg._isolve import minres  # 导入最小残差算法 minres

from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数，用于测试异常处理


def get_sample_problem():
    # 创建一个随机的 10x10 对称矩阵
    np.random.seed(1234)
    matrix = np.random.rand(10, 10)
    matrix = matrix + matrix.T  # 将矩阵转换为对称矩阵
    # 创建一个长度为 10 的随机向量
    vector = np.random.rand(10)
    return matrix, vector


def test_singular():
    A, b = get_sample_problem()
    A[0, ] = 0  # 将矩阵 A 的第一行设为零向量
    b[0] = 0  # 将向量 b 的第一个元素设为零
    xp, info = minres(A, b)  # 使用最小残差算法求解线性方程组
    assert_equal(info, 0)  # 断言 info 的值为 0
    assert norm(A @ xp - b) <= 1e-5 * norm(b)  # 断言求解结果的残差小于给定的阈值


def test_x0_is_used_by():
    A, b = get_sample_problem()
    # 生成一个随机的初始向量 x0
    np.random.seed(12345)
    x0 = np.random.rand(10)
    trace = []

    def trace_iterates(xk):
        trace.append(xk)
    # 使用 minres 求解线性方程组，同时传入初始向量 x0 和迭代回调函数 trace_iterates
    minres(A, b, x0=x0, callback=trace_iterates)
    trace_with_x0 = trace

    trace = []
    minres(A, b, callback=trace_iterates)  # 使用 minres 求解线性方程组，仅传入迭代回调函数
    assert_(not np.array_equal(trace_with_x0[0], trace[0]))  # 断言两次求解的迭代结果不相等


def test_shift():
    A, b = get_sample_problem()
    shift = 0.5  # 定义一个偏移值
    shifted_A = A - shift * np.eye(10)  # 对矩阵 A 进行对角线元素的偏移
    x1, info1 = minres(A, b, shift=shift)  # 使用带偏移的最小残差算法求解线性方程组
    x2, info2 = minres(shifted_A, b)  # 使用不带偏移的最小残差算法求解线性方程组
    assert_equal(info1, 0)  # 断言带偏移求解的 info 值为 0
    assert_allclose(x1, x2, rtol=1e-5)  # 断言带偏移和不带偏移求解的结果接近


def test_asymmetric_fail():
    """Asymmetric matrix should raise `ValueError` when check=True"""
    A, b = get_sample_problem()
    A[1, 2] = 1  # 修改矩阵 A 使其变为非对称矩阵
    A[2, 1] = 2
    with assert_raises(ValueError):  # 使用 assert_raises 断言期望抛出 ValueError 异常
        xp, info = minres(A, b, check=True)


def test_minres_non_default_x0():
    np.random.seed(1234)
    rtol = 1e-6
    a = np.random.randn(5, 5)
    a = np.dot(a, a.T)  # 生成一个对称矩阵 a
    b = np.random.randn(5)
    c = np.random.randn(5)
    x = minres(a, b, x0=c, rtol=rtol)[0]  # 使用自定义的初始向量 c 求解线性方程组
    assert norm(a @ x - b) <= rtol * norm(b)  # 断言求解结果的残差小于给定的阈值


def test_minres_precond_non_default_x0():
    np.random.seed(12345)
    rtol = 1e-6
    a = np.random.randn(5, 5)
    a = np.dot(a, a.T)  # 生成一个对称矩阵 a
    b = np.random.randn(5)
    c = np.random.randn(5)
    m = np.random.randn(5, 5)
    m = np.dot(m, m.T)  # 生成一个对称的预处理矩阵 m
    x = minres(a, b, M=m, x0=c, rtol=rtol)[0]  # 使用自定义的初始向量 c 和预处理矩阵 m 求解线性方程组
    assert norm(a @ x - b) <= rtol * norm(b)  # 断言求解结果的残差小于给定的阈值


def test_minres_precond_exact_x0():
    np.random.seed(1234)
    rtol = 1e-6
    a = np.eye(10)  # 生成单位矩阵 a
    b = np.ones(10)
    c = np.ones(10)
    m = np.random.randn(10, 10)
    m = np.dot(m, m.T)  # 生成一个对称的预处理矩阵 m
    x = minres(a, b, M=m, x0=c, rtol=rtol)[0]  # 使用自定义的初始向量 c 和预处理矩阵 m 求解线性方程组
    assert norm(a @ x - b) <= rtol * norm(b)  # 断言求解结果的残差小于给定的阈值
```