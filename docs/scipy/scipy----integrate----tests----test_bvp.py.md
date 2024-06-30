# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_bvp.py`

```
import sys
# 导入系统模块

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
# 尝试导入旧版的StringIO模块，如果失败则导入新版的StringIO模块

import numpy as np
# 导入NumPy库

from numpy.testing import (assert_, assert_array_equal, assert_allclose,
                           assert_equal)
# 导入NumPy测试相关的断言函数

from pytest import raises as assert_raises
# 导入pytest中的raises函数并重命名为assert_raises

from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
                                  estimate_bc_jac, compute_jac_indices,
                                  construct_global_jac, solve_bvp)
# 导入SciPy库中相关的稀疏矩阵、特殊函数和积分求解模块

def exp_fun(x, y):
    return np.vstack((y[1], y[0]))
# 定义一个函数，返回y向量的反序列

def exp_fun_jac(x, y):
    df_dy = np.empty((2, 2, x.shape[0]))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = 1
    df_dy[1, 1] = 0
    return df_dy
# 定义一个函数，返回雅可比矩阵

def exp_bc(ya, yb):
    return np.hstack((ya[0] - 1, yb[0]))
# 定义一个函数，返回y向量的初始和末尾值

def exp_bc_complex(ya, yb):
    return np.hstack((ya[0] - 1 - 1j, yb[0]))
# 定义一个函数，返回包含实数和虚数的y向量的初始值和末尾值

def exp_bc_jac(ya, yb):
    dbc_dya = np.array([
        [1, 0],
        [0, 0]
    ])
    dbc_dyb = np.array([
        [0, 0],
        [1, 0]
    ])
    return dbc_dya, dbc_dyb
# 定义一个函数，返回雅可比矩阵

def exp_sol(x):
    return (np.exp(-x) - np.exp(x - 2)) / (1 - np.exp(-2))
# 定义一个函数，返回数学公式

def sl_fun(x, y, p):
    return np.vstack((y[1], -p[0]**2 * y[0]))
# 定义一个函数，返回矩阵

def sl_fun_jac(x, y, p):
    n, m = y.shape
    df_dy = np.empty((n, 2, m))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = -p[0]**2
    df_dy[1, 1] = 0

    df_dp = np.empty((n, 1, m))
    df_dp[0, 0] = 0
    df_dp[1, 0] = -2 * p[0] * y[0]

    return df_dy, df_dp
# 定义一个函数，返回

def sl_bc(ya, yb, p):
    return np.hstack((ya[0], yb[0], ya[1] - p[0]))
# 定义一个函数，返回

def sl_bc_jac(ya, yb, p):
    dbc_dya = np.zeros((3, 2))
    dbc_dya[0, 0] = 1
    dbc_dya[2, 1] = 1

    dbc_dyb = np.zeros((3, 2))
    dbc_dyb[1, 0] = 1

    dbc_dp = np.zeros((3, 1))
    dbc_dp[2, 0] = -1

    return dbc_dya, dbc_dyb, dbc_dp
# 定义一个函数，返回

def sl_sol(x, p):
    return np.sin(p[0] * x)
# 定义一个函数，返回

def emden_fun(x, y):
    return np.vstack((y[1], -y[0]**5))
# 定义一个函数，返回

def emden_fun_jac(x, y):
    df_dy = np.empty((2, 2, x.shape[0]))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = -5 * y[0]**4
    df_dy[1, 1] = 0
    return df_dy
# 定义一个函数，返回

def emden_bc(ya, yb):
    return np.array([ya[1], yb[0] - (3/4)**0.5])
# 定义一个函数，返回

def emden_bc_jac(ya, yb):
    dbc_dya = np.array([
        [0, 1],
        [0, 0]
    ])
    dbc_dyb = np.array([
        [0, 0],
        [1, 0]
    ])
    return dbc_dya, dbc_dyb
# 定义一个函数，返回

def emden_sol(x):
    return (1 + x**2/3)**-0.5
# 定义一个函数，返回

def undefined_fun(x, y):
    return np.zeros_like(y)
# 定义一个函数，返回

def undefined_bc(ya, yb):
    return np.array([ya[0], yb[0] - 1])
# 定义一个函数，返回

def big_fun(x, y):
    f = np.zeros_like(y)
    f[::2] = y[1::2]
    return f
# 定义一个函数，返回

def big_bc(ya, yb):
    return np.hstack((ya[::2], yb[::2] - 1))
# 定义一个函数，返回

def big_sol(x, n):
    y = np.ones((2 * n, x.size))
    y[::2] = x
    return x
# 定义一个函数，返回

def big_fun_with_parameters(x, y, p):
    """ Big version of sl_fun, with two parameters.

    The two differential equations represented by sl_fun are broadcast to the
    # 创建一个与 y 具有相同形状的零数组，用于存储微分方程的结果
    f = np.zeros_like(y)
    # 对 f 数组进行赋值操作：
    # 将 y 数组中索引为偶数的元素赋值给 f 数组中对应索引的元素
    f[::2] = y[1::2]
    # 将 y 数组中索引为 1、5、9 等的元素应用微分方程 dy[1]/dt = -p[0]**2 * y[::4]，并赋值给 f 数组中对应索引的元素
    f[1::4] = -p[0]**2 * y[::4]
    # 将 y 数组中索引为 3、7、11 等的元素应用微分方程 dy[3]/dt = -p[1]**2 * y[2::4]，并赋值给 f 数组中对应索引的元素
    f[3::4] = -p[1]**2 * y[2::4]
    # 返回计算得到的 f 数组，其中包含了按照微分方程计算的 y 的导数值
    return f
# 定义一个函数，用于计算带有参数的大规模方程的雅可比矩阵
def big_fun_with_parameters_jac(x, y, p):
    # 获取 y 的形状，n 表示行数，m 表示列数
    n, m = y.shape
    # 初始化一个大小为 (n, n, m) 的全零数组，用于存储结果
    df_dy = np.zeros((n, n, m))
    # 设置 df_dy 的部分元素值为 1，选择性地对 df_dy 进行赋值
    df_dy[range(0, n, 2), range(1, n, 2)] = 1
    df_dy[range(1, n, 4), range(0, n, 4)] = -p[0]**2
    df_dy[range(3, n, 4), range(2, n, 4)] = -p[1]**2

    # 初始化一个大小为 (n, 2, m) 的全零数组，用于存储结果
    df_dp = np.zeros((n, 2, m))
    # 对 df_dp 的部分元素进行赋值，涉及到参数 p 和 y 的特定元素
    df_dp[range(1, n, 4), 0] = -2 * p[0] * y[range(0, n, 4)]
    df_dp[range(3, n, 4), 1] = -2 * p[1] * y[range(2, n, 4)]

    # 返回计算得到的 df_dy 和 df_dp
    return df_dy, df_dp


# 定义一个函数，用于计算带有参数的大规模边界条件的向量
def big_bc_with_parameters(ya, yb, p):
    # 返回连接后的 ya 和 yb 的部分元素，以及额外添加的两个边界条件
    return np.hstack((ya[::2], yb[::2], ya[1] - p[0], ya[3] - p[1]))


# 定义一个函数，用于计算带有参数的大规模边界条件的雅可比矩阵
def big_bc_with_parameters_jac(ya, yb, p):
    # 获取 ya 的行数
    n = ya.shape[0]
    # 初始化大小为 (n + 2, n) 的全零数组，用于存储 dbc_dya
    dbc_dya = np.zeros((n + 2, n))
    # 初始化大小为 (n + 2, n) 的全零数组，用于存储 dbc_dyb
    dbc_dyb = np.zeros((n + 2, n))

    # 设置 dbc_dya 和 dbc_dyb 的部分元素值为 1，选择性地对它们进行赋值
    dbc_dya[range(n // 2), range(0, n, 2)] = 1
    dbc_dyb[range(n // 2, n), range(0, n, 2)] = 1

    # 初始化大小为 (n + 2, 2) 的全零数组，用于存储 dbc_dp
    dbc_dp = np.zeros((n + 2, 2))
    # 对 dbc_dp 的部分元素进行赋值，涉及到参数 p 和 ya 的特定元素
    dbc_dp[n, 0] = -1
    dbc_dya[n, 1] = 1
    dbc_dp[n + 1, 1] = -1
    dbc_dya[n + 1, 3] = 1

    # 返回计算得到的 dbc_dya, dbc_dyb 和 dbc_dp
    return dbc_dya, dbc_dyb, dbc_dp


# 定义一个函数，用于计算带有参数的大规模常微分方程的解
def big_sol_with_parameters(x, p):
    # 返回由两个正弦函数组成的列向量
    return np.vstack((np.sin(p[0] * x), np.sin(p[1] * x)))


# 定义一个函数，用于计算冲击方程的右侧函数
def shock_fun(x, y):
    eps = 1e-3
    # 返回冲击方程的右侧向量
    return np.vstack((
        y[1],
        -(x * y[1] + eps * np.pi**2 * np.cos(np.pi * x) +
          np.pi * x * np.sin(np.pi * x)) / eps
    ))


# 定义一个函数，用于计算冲击方程的边界条件
def shock_bc(ya, yb):
    # 返回冲击方程的边界条件向量
    return np.array([ya[0] + 2, yb[0]])


# 定义一个函数，用于计算冲击方程的解析解
def shock_sol(x):
    eps = 1e-3
    k = np.sqrt(2 * eps)
    # 返回冲击方程的解析解
    return np.cos(np.pi * x) + erf(x / k) / erf(1 / k)


# 定义一个函数，用于计算非线性边界条件方程的右侧函数
def nonlin_bc_fun(x, y):
    # 返回非线性边界条件方程的右侧向量
    return np.stack([y[1], np.zeros_like(x)])


# 定义一个函数，用于计算非线性边界条件方程的边界条件
def nonlin_bc_bc(ya, yb):
    # 分别获取 ya 和 yb 的两个分量
    phiA, phipA = ya
    phiC, phipC = yb

    kappa, ioA, ioC, V, f = 1.64, 0.01, 1.0e-4, 0.5, 38.9

    # 计算并返回非线性边界条件方程的边界条件向量
    hA = 0.0 - phiA - 0.0
    iA = ioA * (np.exp(f * hA) - np.exp(-f * hA))
    res0 = iA + kappa * phipA

    hC = V - phiC - 1.0
    iC = ioC * (np.exp(f * hC) - np.exp(-f * hC))
    res1 = iC - kappa * phipC

    return np.array([res0, res1])


# 定义一个函数，用于计算非线性边界条件方程的解析解
def nonlin_bc_sol(x):
    # 返回非线性边界条件方程的解析解
    return -0.13426436116763119 - 1.1308709 * x


# 定义一个测试函数，用于测试修改网格的功能
def test_modify_mesh():
    x = np.array([0, 1, 3, 9], dtype=float)
    # 调用 modify_mesh 函数修改网格
    x_new = modify_mesh(x, np.array([0]), np.array([2]))
    # 断言修改后的网格与预期结果一致
    assert_array_equal(x_new, np.array([0, 0.5, 1, 3, 5, 7, 9]))

    x = np.array([-6, -3, 0, 3, 6], dtype=float)
    # 再次调用 modify_mesh 函数修改网格
    x_new = modify_mesh(x, np.array([1], dtype=int), np.array([0, 2, 3]))
    # 断言修改后的网格与预期结果一致
    assert_array_equal(x_new, [-6, -5, -4, -3, -1.5, 0, 1, 2, 3, 4, 5, 6])


# 定义一个测试函数，用于测试计算函数和雅可比矩阵的功能
def test_compute_fun_jac():
    x = np.linspace(0, 1, 5)
    y = np.empty((2, x.shape[0]))
    y[0] = 0.01
    y[1] = 0.02
    p = np.array([])
    # 调用 estimate_fun_jac 函数计算函数和雅可比矩阵
    df_dy, df_dp = estimate_fun_jac(lambda x, y, p: exp_fun(x, y), x, y, p)
    # 调用 exp_fun_jac 函数计算函数和雅可比矩阵的解析解
    df_dy_an = exp_fun_jac(x, y
    # 计算数组 x 中每个元素的正弦值，存储在 y 的第一个行中
    y[0] = np.sin(x)
    # 计算数组 x 中每个元素的余弦值，存储在 y 的第二行中
    y[1] = np.cos(x)
    # 创建一个包含单个浮点数 1.0 的 numpy 数组 p
    p = np.array([1.0])
    # 估算函数 sl_fun 的雅可比矩阵 df_dy 和 df_dp
    df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
    # 使用准确函数 sl_fun_jac 计算真实的 df_dy_an 和 df_dp_an
    df_dy_an, df_dp_an = sl_fun_jac(x, y, p)
    # 断言估算的 df_dy 和准确的 df_dy_an 很接近
    assert_allclose(df_dy, df_dy_an)
    # 断言估算的 df_dp 和准确的 df_dp_an 很接近
    assert_allclose(df_dp, df_dp_an)
    
    # 创建一个从 0 到 1 的等间距数字组成的数组 x，包含 10 个元素
    x = np.linspace(0, 1, 10)
    # 创建一个形状为 (2, 10) 的空 numpy 数组 y
    y = np.empty((2, x.shape[0]))
    # 将 y 的第一行填充为 (3/4) 的平方根
    y[0] = (3/4)**0.5
    # 将 y 的第二行填充为 1e-4
    y[1] = 1e-4
    # 创建一个空的 numpy 数组 p
    p = np.array([])
    # 估算函数 emden_fun 的雅可比矩阵 df_dy 和 df_dp
    df_dy, df_dp = estimate_fun_jac(lambda x, y, p: emden_fun(x, y), x, y, p)
    # 使用准确函数 emden_fun_jac 计算真实的 df_dy_an
    df_dy_an = emden_fun_jac(x, y)
    # 断言估算的 df_dy 和准确的 df_dy_an 很接近
    assert_allclose(df_dy, df_dy_an)
    # 断言 df_dp 为 None
    assert_(df_dp is None)
# 定义函数 test_compute_bc_jac，用于测试边界条件雅可比矩阵的计算
def test_compute_bc_jac():
    # 设置初始值 ya 和 yb 为 numpy 数组
    ya = np.array([-1.0, 2])
    yb = np.array([0.5, 3])
    # 设置 p 为空数组
    p = np.array([])
    # 调用 estimate_bc_jac 函数计算边界条件雅可比矩阵的近似值
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
        lambda ya, yb, p: exp_bc(ya, yb), ya, yb, p)
    # 计算真实的边界条件雅可比矩阵值
    dbc_dya_an, dbc_dyb_an = exp_bc_jac(ya, yb)
    # 使用 assert_allclose 检查 dbc_dya 和 dbc_dya_an 的近似程度
    assert_allclose(dbc_dya, dbc_dya_an)
    # 使用 assert_allclose 检查 dbc_dyb 和 dbc_dyb_an 的近似程度
    assert_allclose(dbc_dyb, dbc_dyb_an)
    # 检查 dbc_dp 是否为 None
    assert_(dbc_dp is None)

    # 更新 ya, yb, p 的值
    ya = np.array([0.0, 1])
    yb = np.array([0.0, -1])
    p = np.array([0.5])
    # 再次调用 estimate_bc_jac 函数计算边界条件雅可比矩阵的近似值
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, ya, yb, p)
    # 计算真实的边界条件雅可比矩阵值
    dbc_dya_an, dbc_dyb_an, dbc_dp_an = sl_bc_jac(ya, yb, p)
    # 使用 assert_allclose 检查 dbc_dya 和 dbc_dya_an 的近似程度
    assert_allclose(dbc_dya, dbc_dya_an)
    # 使用 assert_allclose 检查 dbc_dyb 和 dbc_dyb_an 的近似程度
    assert_allclose(dbc_dyb, dbc_dyb_an)
    # 使用 assert_allclose 检查 dbc_dp 和 dbc_dp_an 的近似程度
    assert_allclose(dbc_dp, dbc_dp_an)

    # 更新 ya, yb, p 的值
    ya = np.array([0.5, 100])
    yb = np.array([-1000, 10.5])
    p = np.array([])
    # 再次调用 estimate_bc_jac 函数计算边界条件雅可比矩阵的近似值
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
        lambda ya, yb, p: emden_bc(ya, yb), ya, yb, p)
    # 计算真实的边界条件雅可比矩阵值
    dbc_dya_an, dbc_dyb_an = emden_bc_jac(ya, yb)
    # 使用 assert_allclose 检查 dbc_dya 和 dbc_dya_an 的近似程度
    assert_allclose(dbc_dya, dbc_dya_an)
    # 使用 assert_allclose 检查 dbc_dyb 和 dbc_dyb_an 的近似程度
    assert_allclose(dbc_dyb, dbc_dyb_an)
    # 检查 dbc_dp 是否为 None


# 定义函数 test_compute_jac_indices，用于测试雅可比矩阵的索引计算
def test_compute_jac_indices():
    # 设置 n, m, k 的值
    n = 2
    m = 4
    k = 2
    # 调用 compute_jac_indices 函数计算雅可比矩阵的索引 i 和 j
    i, j = compute_jac_indices(n, m, k)
    # 使用 coo_matrix 创建稀疏矩阵 s，并将其转换为数组形式
    s = coo_matrix((np.ones_like(i), (i, j))).toarray()
    # 设置真实的稀疏矩阵 s_true
    s_true = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    # 使用 assert_array_equal 检查 s 和 s_true 是否相等
    assert_array_equal(s, s_true)


# 定义函数 test_compute_global_jac，用于测试全局雅可比矩阵的构建
def test_compute_global_jac():
    # 设置 n, m, k 的值
    n = 2
    m = 5
    k = 1
    # 调用 compute_jac_indices 函数计算雅可比矩阵的索引 i_jac 和 j_jac
    i_jac, j_jac = compute_jac_indices(2, 5, 1)
    # 创建线性空间 x
    x = np.linspace(0, 1, 5)
    # 计算步长 h
    h = np.diff(x)
    # 创建矩阵 y
    y = np.vstack((np.sin(np.pi * x), np.pi * np.cos(np.pi * x)))
    # 设置参数数组 p
    p = np.array([3.0])

    # 计算函数值 f
    f = sl_fun(x, y, p)

    # 计算中间值 x_middle 和 y_middle
    x_middle = x[:-1] + 0.5 * h
    y_middle = 0.5 * (y[:, :-1] + y[:, 1:]) - h/8 * (f[:, 1:] - f[:, :-1])

    # 计算函数的雅可比矩阵 df_dy 和 df_dp
    df_dy, df_dp = sl_fun_jac(x, y, p)
    # 计算中间点的函数雅可比矩阵 df_dy_middle 和 df_dp_middle
    df_dy_middle, df_dp_middle = sl_fun_jac(x_middle, y_middle, p)
    # 计算边界条件的雅可比矩阵 dbc_dya, dbc_dyb, dbc_dp
    dbc_dya, dbc_dyb, dbc_dp = sl_bc_jac(y[:, 0], y[:, -1], p)

    # 构建全局雅可比矩阵 J
    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
                             df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    # 将稀疏矩阵 J 转换为稠密数组形式
    J = J.toarray()

    # 定义 J_block 函数，计算块状雅可比矩阵的值
    def J_block(h, p):
        return np.array([
            [h**2*p**2/12 - 1, -0.5*h, -h**2*p**2/12 + 1, -0.5*h],
            [0.5*h*p**2, h**2*p**2/12 - 1, 0.5*h*p**2, 1 - h**2*p**2/12]
        ])

    # 初始化真实的全局雅可比矩阵 J_true
    J_true = np.zeros((m * n
    # 设置 J_true 矩阵中特定位置的值
    J_true[10, 1] = 1
    J_true[10, 10] = -1
    
    # 使用 assert_allclose 函数检查 J 矩阵与 J_true 矩阵的接近程度
    assert_allclose(J, J_true, rtol=1e-10)
    
    # 调用 estimate_fun_jac 函数计算目标函数关于 y 和 p 的雅可比矩阵
    df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
    df_dy_middle, df_dp_middle = estimate_fun_jac(sl_fun, x_middle, y_middle, p)
    
    # 调用 estimate_bc_jac 函数计算边界条件关于 y 和 p 的雅可比矩阵
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, y[:, 0], y[:, -1], p)
    
    # 调用 construct_global_jac 函数构建全局雅可比矩阵 J
    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
                             df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    # 将稀疏矩阵 J 转换为稠密数组
    J = J.toarray()
    
    # 使用 assert_allclose 函数检查 J 矩阵与 J_true 矩阵的接近程度
    assert_allclose(J, J_true, rtol=2e-8, atol=2e-8)
# 定义参数验证的测试函数
def test_parameter_validation():
    # 设置 x 为列表 [0, 1, 0.5]
    x = [0, 1, 0.5]
    # 创建一个 2x3 的零矩阵 y
    y = np.zeros((2, 3))
    # 断言调用 solve_bvp 函数时会引发 ValueError 异常
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)

    # 将 x 定义为从 0 到 1 等间距取 5 个点的数组
    x = np.linspace(0, 1, 5)
    # 创建一个 2x4 的零矩阵 y
    y = np.zeros((2, 4))
    # 断言调用 solve_bvp 函数时会引发 ValueError 异常
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)

    # 定义一个函数 fun，接受参数 x, y, p，调用 exp_fun 返回结果
    def fun(x, y, p):
        return exp_fun(x, y)
    # 定义一个函数 bc，接受参数 ya, yb, p，调用 exp_bc 返回结果
    def bc(ya, yb, p):
        return exp_bc(ya, yb)

    # 创建一个 2x5 的零矩阵 y
    y = np.zeros((2, x.shape[0]))
    # 断言调用 solve_bvp 函数时会引发 ValueError 异常，并带有额外参数 p=[1]
    assert_raises(ValueError, solve_bvp, fun, bc, x, y, p=[1])

    # 定义一个函数 wrong_shape_fun，接受参数 x, y，返回一个形状为 (3,) 的零数组
    def wrong_shape_fun(x, y):
        return np.zeros(3)

    # 断言调用 solve_bvp 函数时会引发 ValueError 异常
    assert_raises(ValueError, solve_bvp, wrong_shape_fun, bc, x, y)

    # 创建一个形状为 (1, 2) 的数组 S
    S = np.array([[0, 0]])
    # 断言调用 solve_bvp 函数时会引发 ValueError 异常，并带有额外参数 S=S
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y, S=S)


# 定义无参数情况下的测试函数
def test_no_params():
    # 将 x 定义为从 0 到 1 等间距取 5 个点的数组
    x = np.linspace(0, 1, 5)
    # 将 x_test 定义为从 0 到 1 等间距取 100 个点的数组
    x_test = np.linspace(0, 1, 100)
    # 创建一个 2x5 的零矩阵 y
    y = np.zeros((2, x.shape[0]))
    # 遍历两个可能的参数组合：fun_jac 和 bc_jac
    for fun_jac in [None, exp_fun_jac]:
        for bc_jac in [None, exp_bc_jac]:
            # 调用 solve_bvp 函数，求解常微分方程组
            sol = solve_bvp(exp_fun, exp_bc, x, y, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            # 断言解的状态为 0（成功）
            assert_equal(sol.status, 0)
            # 断言求解成功
            assert_(sol.success)

            # 断言解的 x 值数组大小为 5
            assert_equal(sol.x.size, 5)

            # 计算在 x_test 上的解 sol_test
            sol_test = sol.sol(x_test)

            # 断言 sol_test 的第一个分量与 exp_sol(x_test) 很接近
            assert_allclose(sol_test[0], exp_sol(x_test), atol=1e-5)

            # 计算在 x_test 上的函数值 f_test
            f_test = exp_fun(x_test, sol_test)
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res**2, axis=0)**0.5

            # 断言归一化的残差 norm_res 全部小于 1e-3
            assert_(np.all(norm_res < 1e-3))

            # 断言所有的均方残差 sol.rms_residuals 全部小于 1e-3
            assert_(np.all(sol.rms_residuals < 1e-3))

            # 断言在解的点上，sol.sol(sol.x) 很接近 sol.y
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)

            # 断言在解的点上，sol.sol(sol.x, 1) 很接近 sol.yp
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


# 定义带参数情况下的测试函数
def test_with_params():
    # 将 x 定义为从 0 到 π 等间距取 5 个点的数组
    x = np.linspace(0, np.pi, 5)
    # 将 x_test 定义为从 0 到 π 等间距取 100 个点的数组
    x_test = np.linspace(0, np.pi, 100)
    # 创建一个形状为 (2, 5) 的全为 1 的矩阵 y
    y = np.ones((2, x.shape[0]))

    # 遍历两个可能的参数组合：fun_jac 和 bc_jac
    for fun_jac in [None, sl_fun_jac]:
        for bc_jac in [None, sl_bc_jac]:
            # 调用 solve_bvp 函数，求解带参数的常微分方程组
            sol = solve_bvp(sl_fun, sl_bc, x, y, p=[0.5], fun_jac=fun_jac,
                            bc_jac=bc_jac)

            # 断言解的状态为 0（成功）
            assert_equal(sol.status, 0)
            # 断言求解成功
            assert_(sol.success)

            # 断言解的 x 值数组大小小于 10
            assert_(sol.x.size < 10)

            # 断言参数 sol.p 很接近 [1]
            assert_allclose(sol.p, [1], rtol=1e-4)

            # 计算在 x_test 上的解 sol_test
            sol_test = sol.sol(x_test)

            # 断言 sol_test 的第一个分量与 sl_sol(x_test, [1]) 很接近
            assert_allclose(sol_test[0], sl_sol(x_test, [1]),
                            rtol=1e-4, atol=1e-4)

            # 计算在 x_test 上的函数值 f_test
            f_test = sl_fun(x_test, sol_test, [1])
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

            # 断言归一化的残差 norm_res 全部小于 1e-3
            assert_(np.all(norm_res < 1e-3))

            # 断言所有的均方残差 sol.rms_residuals 全部小于 1e-3
            assert_(np.all(sol.rms_residuals < 1e-3))

            # 断言在解的点上，sol.sol(sol.x) 很接近 sol.y
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)

            # 断言在解的点上，sol.sol(sol.x, 1) 很接近 sol.yp
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


# 定义带奇异项的测试函数
def test_singular_term():
    # 将 x 定义为从 0 到 1 等间距取 10 个点的数组
    x = np.linspace(0, 1, 10)
    # 将 x_test 定义为从 0.05 到 1 等间距取 100 个点的数组
    # 遍历两个函数（fun_jac和bc_jac）的可能取值，分别为None和emden_fun_jac/emden_bc_jac
    for fun_jac in [None, emden_fun_jac]:
        # 同样遍历两个边界条件的雅可比矩阵的可能取值，分别为None和emden_bc_jac/emden_bc_jac
        for bc_jac in [None, emden_bc_jac]:
            # 使用solve_bvp函数求解边界值问题
            sol = solve_bvp(emden_fun, emden_bc, x, y, S=S, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            # 断言求解状态为0，表示求解成功
            assert_equal(sol.status, 0)
            # 断言求解成功
            assert_(sol.success)

            # 断言返回的解向量sol.x的大小为10
            assert_equal(sol.x.size, 10)

            # 对求解的结果进行测试，计算与emden_sol(x_test)的绝对误差，容差为1e-5
            sol_test = sol.sol(x_test)
            assert_allclose(sol_test[0], emden_sol(x_test), atol=1e-5)

            # 计算测试函数f_test，用于后续的残差计算
            f_test = emden_fun(x_test, sol_test) + S.dot(sol_test) / x_test
            # 计算残差r，相对残差rel_res和范数残差norm_res
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

            # 断言范数残差norm_res的所有元素都小于1e-3
            assert_(np.all(norm_res < 1e-3))
            
            # 对sol.sol(sol.x)与sol.y进行绝对误差和相对误差的断言检查
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            # 对sol.sol(sol.x, 1)与sol.yp进行绝对误差和相对误差的断言检查
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
def test_complex():
    # 测试与 test_no_params 类似，但边界条件被设置为复杂情况
    x = np.linspace(0, 1, 5)  # 在 [0, 1] 区间生成等间距的5个点作为横坐标
    x_test = np.linspace(0, 1, 100)  # 在 [0, 1] 区间生成等间距的100个点作为测试横坐标
    y = np.zeros((2, x.shape[0]), dtype=complex)  # 创建一个复数类型的二维数组，形状为 (2, 5)，用于存储解向量

    # 循环迭代不同的 fun_jac 和 bc_jac 函数
    for fun_jac in [None, exp_fun_jac]:
        for bc_jac in [None, exp_bc_jac]:
            # 调用 solve_bvp 函数求解边值问题
            sol = solve_bvp(exp_fun, exp_bc_complex, x, y, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            # 断言解的状态为 0，表示成功
            assert_equal(sol.status, 0)
            # 断言解的成功状态为真
            assert_(sol.success)

            # 对解在测试点 x_test 上进行评估
            sol_test = sol.sol(x_test)

            # 断言实部和虚部与期望解 exp_sol(x_test) 很接近
            assert_allclose(sol_test[0].real, exp_sol(x_test), atol=1e-5)
            assert_allclose(sol_test[0].imag, exp_sol(x_test), atol=1e-5)

            # 计算在测试点上的函数值并比较
            f_test = exp_fun(x_test, sol_test)
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(np.real(rel_res * np.conj(rel_res)),
                              axis=0) ** 0.5
            # 断言相对残差的范数小于 1e-3
            assert_(np.all(norm_res < 1e-3))

            # 断言所有的均方根残差小于 1e-3
            assert_(np.all(sol.rms_residuals < 1e-3))
            # 断言解函数在其内部节点处与解向量 sol.y 很接近
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            # 断言解函数在其内部节点处的导数与解向量的导数 sol.yp 很接近
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_failures():
    x = np.linspace(0, 1, 2)  # 在 [0, 1] 区间生成等间距的2个点作为横坐标
    y = np.zeros((2, x.size))  # 创建一个二维零数组，形状为 (2, 2)

    # 调用 solve_bvp 函数求解边值问题，设置容差和最大节点数
    res = solve_bvp(exp_fun, exp_bc, x, y, tol=1e-5, max_nodes=5)
    # 断言解的状态为 1，表示失败
    assert_equal(res.status, 1)
    # 断言解的成功状态为假
    assert_(not res.success)

    x = np.linspace(0, 1, 5)  # 在 [0, 1] 区间生成等间距的5个点作为横坐标
    y = np.zeros((2, x.size))  # 创建一个二维零数组，形状为 (2, 5)

    # 调用 solve_bvp 函数求解边值问题，使用未定义的函数 exp_fun 和 exp_bc
    res = solve_bvp(undefined_fun, undefined_bc, x, y)
    # 断言解的状态为 2，表示错误
    assert_equal(res.status, 2)
    # 断言解的成功状态为假
    assert_(not res.success)


def test_big_problem():
    n = 30  # 设置问题的规模参数
    x = np.linspace(0, 1, 5)  # 在 [0, 1] 区间生成等间距的5个点作为横坐标
    y = np.zeros((2 * n, x.size))  # 创建一个二维零数组，形状为 (60, 5)

    # 调用 solve_bvp 函数求解大规模边值问题
    sol = solve_bvp(big_fun, big_bc, x, y)

    # 断言解的状态为 0，表示成功
    assert_equal(sol.status, 0)
    # 断言解的成功状态为真
    assert_(sol.success)

    # 对解在横坐标 x 上的评估
    sol_test = sol.sol(x)

    # 断言解在第一个分量上与期望解 big_sol(x, n) 很接近
    assert_allclose(sol_test[0], big_sol(x, n))

    # 计算在横坐标 x 上的函数值并比较
    f_test = big_fun(x, sol_test)
    r = sol.sol(x, 1) - f_test
    rel_res = r / (1 + np.abs(f_test))
    norm_res = np.sum(np.real(rel_res * np.conj(rel_res)), axis=0) ** 0.5
    # 断言相对残差的范数小于 1e-3
    assert_(np.all(norm_res < 1e-3))

    # 断言所有的均方根残差小于 1e-3
    assert_(np.all(sol.rms_residuals < 1e-3))
    # 断言解函数在其内部节点处与解向量 sol.y 很接近
    assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
    # 断言解函数在其内部节点处的导数与解向量的导数 sol.yp 很接近
    assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_big_problem_with_parameters():
    n = 30  # 设置问题的规模参数
    x = np.linspace(0, np.pi, 5)  # 在 [0, π] 区间生成等间距的5个点作为横坐标
    x_test = np.linspace(0, np.pi, 100)  # 在 [0, π] 区间生成等间距的100个点作为测试横坐标
    y = np.ones((2 * n, x.size))  # 创建一个全为1的二维数组，形状为 (60, 5)
    # 遍历不同的函数雅可比矩阵选项和边界条件雅可比矩阵选项
    for fun_jac in [None, big_fun_with_parameters_jac]:
        for bc_jac in [None, big_bc_with_parameters_jac]:
            # 使用给定的函数和边界条件解决边界值问题
            sol = solve_bvp(big_fun_with_parameters, big_bc_with_parameters, x,
                            y, p=[0.5, 0.5], fun_jac=fun_jac, bc_jac=bc_jac)

            # 断言解决方案的状态为成功
            assert_equal(sol.status, 0)
            assert_(sol.success)

            # 断言解的参数与期望值非常接近
            assert_allclose(sol.p, [1, 1], rtol=1e-4)

            # 对解的测试进行评估
            sol_test = sol.sol(x_test)

            # 遍历解的每个分量，与带参数的大解函数的结果进行比较
            for isol in range(0, n, 4):
                assert_allclose(sol_test[isol],
                                big_sol_with_parameters(x_test, [1, 1])[0],
                                rtol=1e-4, atol=1e-4)
                assert_allclose(sol_test[isol + 2],
                                big_sol_with_parameters(x_test, [1, 1])[1],
                                rtol=1e-4, atol=1e-4)

            # 计算测试函数与解的相对残差
            f_test = big_fun_with_parameters(x_test, sol_test, [1, 1])
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

            # 断言所有残差的范数都小于给定的阈值
            assert_(np.all(norm_res < 1e-3))

            # 断言解的均方根残差都小于给定的阈值
            assert_(np.all(sol.rms_residuals < 1e-3))

            # 断言解在给定点的值与解向量的值非常接近
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)

            # 断言解在给定点的导数与解向量的导数值非常接近
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)
def test_verbose():
    # 定义一个测试函数，用于测试 solve_bvp 函数的 verbose 参数
    # Smoke test，检查打印是否正常运行且不崩溃

    x = np.linspace(0, 1, 5)  # 生成一个包含5个元素的等间距数组 x
    y = np.zeros((2, x.shape[0]))  # 创建一个2行5列的零矩阵 y

    # 遍历 verbose 参数的三种可能取值：0, 1, 2
    for verbose in [0, 1, 2]:
        old_stdout = sys.stdout  # 保存原始标准输出
        sys.stdout = StringIO()  # 将标准输出重定向到内存中的字符串

        try:
            # 调用 solve_bvp 函数进行求解
            sol = solve_bvp(exp_fun, exp_bc, x, y, verbose=verbose)
            text = sys.stdout.getvalue()  # 获取 solve_bvp 输出的文本
        finally:
            sys.stdout = old_stdout  # 恢复原始标准输出

        # 断言求解成功
        assert_(sol.success)

        # 根据 verbose 参数进行不同的断言
        if verbose == 0:
            assert_(not text, text)  # verbose=0 时，不应该有输出文本
        if verbose >= 1:
            assert_("Solved in" in text, text)  # verbose>=1 时，输出文本中应包含 "Solved in"
        if verbose >= 2:
            assert_("Max residual" in text, text)  # verbose>=2 时，输出文本中应包含 "Max residual"
```