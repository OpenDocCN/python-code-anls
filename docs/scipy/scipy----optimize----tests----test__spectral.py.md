# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__spectral.py`

```
import itertools  # 导入 itertools 模块，用于生成迭代器的函数

import numpy as np  # 导入 NumPy 库并重命名为 np，用于数值计算
from numpy import exp  # 从 NumPy 中导入 exp 函数，用于计算指数函数
from numpy.testing import assert_, assert_equal  # 导入 NumPy 测试模块中的 assert_ 和 assert_equal 函数

from scipy.optimize import root  # 从 SciPy 中导入 root 函数，用于求解方程

def test_performance():
    # 比较性能结果与 [Cheng & Li, IMA J. Num. An. 29, 814 (2008)] 和
    # [W. La Cruz, J.M. Martinez, M. Raydan, Math. Comp. 75, 1429 (2006)] 中的列表进行比较。
    # 和 M. Raydan 网站上的 dfsane.f 产生的结果。
    #
    # 若结果不一致，则取最大限度的结果。

    e_a = 1e-5  # 绝对误差容差
    e_r = 1e-4  # 相对误差容差

    table_1 = [
        dict(F=F_1, x0=x0_1, n=1000, nit=5, nfev=5),  # 第一个测试参数字典
        dict(F=F_1, x0=x0_1, n=10000, nit=2, nfev=2),  # 第二个测试参数字典
        dict(F=F_2, x0=x0_2, n=500, nit=11, nfev=11),  # 第三个测试参数字典
        dict(F=F_2, x0=x0_2, n=2000, nit=11, nfev=11),  # 第四个测试参数字典
        # dict(F=F_4, x0=x0_4, n=999, nit=243, nfev=1188) 已移除:
        # 对舍入误差过于敏感
        # dfsane.f 的结果；文献中 nit=3, nfev=3
        dict(F=F_6, x0=x0_6, n=100, nit=6, nfev=6),  # 第五个测试参数字典
        # 必须有 n%3==0，文献中有错字吗？
        dict(F=F_7, x0=x0_7, n=99, nit=23, nfev=29),  # 第六个测试参数字典
        # 必须有 n%3==0，文献中有错字吗？
        dict(F=F_7, x0=x0_7, n=999, nit=23, nfev=29),  # 第七个测试参数字典
        # dfsane.f 的结果；文献中 nit=nfev=6？
        dict(F=F_9, x0=x0_9, n=100, nit=12, nfev=18),  # 第八个测试参数字典
        dict(F=F_9, x0=x0_9, n=1000, nit=12, nfev=18),  # 第九个测试参数字典
        # dfsane.f 的结果；文献中 nit=2, nfev=12
        dict(F=F_10, x0=x0_10, n=1000, nit=5, nfev=5),  # 第十个测试参数字典
    ]

    # 检查尺度不变性
    for xscale, yscale, line_search in itertools.product(
        [1.0, 1e-10, 1e10], [1.0, 1e-10, 1e10], ['cruz', 'cheng']
    ):
        for problem in table_1:
            n = problem['n']
            def func(x, n):
                return yscale * problem['F'](x / xscale, n)
            args = (n,)
            x0 = problem['x0'](n) * xscale

            # 计算容差
            fatol = np.sqrt(n) * e_a * yscale + e_r * np.linalg.norm(func(x0, n))

            sigma_eps = 1e-10 * min(yscale/xscale, xscale/yscale)
            sigma_0 = xscale/yscale

            with np.errstate(over='ignore'):
                sol = root(func, x0, args=args,
                           options=dict(ftol=0, fatol=fatol, maxfev=problem['nfev'] + 1,
                                        sigma_0=sigma_0, sigma_eps=sigma_eps,
                                        line_search=line_search),
                           method='DF-SANE')

            # 错误消息
            err_msg = repr(
                [xscale, yscale, line_search, problem, np.linalg.norm(func(sol.x, n)),
                 fatol, sol.success, sol.nit, sol.nfev]
            )
            assert sol.success, err_msg
            # nfev+1: dfsane.f 不计算第一次评估
            assert sol.nfev <= problem['nfev'] + 1, err_msg
            assert sol.nit <= problem['nit'], err_msg
            assert np.linalg.norm(func(sol.x, n)) <= fatol, err_msg


def test_complex():
    def func(z):
        return z**2 - 1 + 2j
    x0 = 2.0j

    ftol = 1e-4  # 功能容差
    # 使用 DF-SANE 方法求解函数 func 在初始点 x0 处的最优解
    sol = root(func, x0, tol=ftol, method='DF-SANE')

    # 断言优化成功，即 sol 对象中的 success 属性为 True
    assert_(sol.success)

    # 计算初始点 x0 处函数值的范数 f0
    f0 = np.linalg.norm(func(x0))
    
    # 计算经过优化后的最优解 sol.x 处函数值的范数 fx
    fx = np.linalg.norm(func(sol.x))
    
    # 断言经过优化后的函数值范数 fx 小于等于 ftol 乘以初始函数值范数 f0
    assert_(fx <= ftol*f0)
def test_linear_definite():
    # DF-SANE paper证明了对于“强独立”解，收敛性成立。
    #
    # 对于线性系统 F(x) = A x - b = 0，其中 A 是正定或负定的，解是强独立的。

    def check_solvability(A, b, line_search='cruz'):
        # 定义函数 func(x)，返回 A*x - b
        def func(x):
            return A.dot(x) - b
        
        # 求解线性方程 A*x = b，得到精确解 xp
        xp = np.linalg.solve(A, b)
        
        # 计算误差容限 eps
        eps = np.linalg.norm(func(xp)) * 1e3
        
        # 使用 DF-SANE 方法求解非线性方程 func(x) = 0
        sol = root(
            func, b,
            options=dict(fatol=eps, ftol=0, maxfev=17523, line_search=line_search),
            method='DF-SANE',
        )
        
        # 断言求解成功
        assert_(sol.success)
        
        # 断言解的精度符合误差容限 eps
        assert_(np.linalg.norm(func(sol.x)) <= eps)

    n = 90

    # 测试线性正定系统
    np.random.seed(1234)
    A = np.arange(n*n).reshape(n, n)
    A = A + n*n * np.diag(1 + np.arange(n))
    assert_(np.linalg.eigvals(A).min() > 0)
    b = np.arange(n) * 1.0
    
    # 使用 'cruz' 和 'cheng' 两种线搜索方法测试解的可解性
    check_solvability(A, b, 'cruz')
    check_solvability(A, b, 'cheng')

    # 测试线性负定系统
    check_solvability(-A, b, 'cruz')
    check_solvability(-A, b, 'cheng')


def test_shape():
    def f(x, arg):
        return x - arg

    # 测试不同数据类型的 x 和 arg
    for dt in [float, complex]:
        x = np.zeros([2,2])
        arg = np.ones([2,2], dtype=dt)

        # 使用 DF-SANE 方法求解非线性方程 f(x, arg) = 0
        sol = root(f, x, args=(arg,), method='DF-SANE')
        
        # 断言求解成功
        assert_(sol.success)
        
        # 断言解的形状与 x 相同
        assert_equal(sol.x.shape, x.shape)


# Some of the test functions and initial guesses listed in
# [W. La Cruz, M. Raydan. Optimization Methods and Software, 18, 583 (2003)]

def F_1(x, n):
    g = np.zeros([n])
    i = np.arange(2, n+1)
    g[0] = exp(x[0] - 1) - 1
    g[1:] = i*(exp(x[1:] - 1) - x[1:])
    return g

def x0_1(n):
    x0 = np.empty([n])
    x0.fill(n/(n-1))
    return x0

def F_2(x, n):
    g = np.zeros([n])
    i = np.arange(2, n+1)
    g[0] = exp(x[0]) - 1
    g[1:] = 0.1*i*(exp(x[1:]) + x[:-1] - 1)
    return g

def x0_2(n):
    x0 = np.empty([n])
    x0.fill(1/n**2)
    return x0


def F_4(x, n):  # skip name check
    assert_equal(n % 3, 0)
    g = np.zeros([n])
    # 注意：一些文献中第一行存在拼写错误；
    # 在原文 [Gasparo, Optimization Meth. 13, 79 (2000)] 中是正确的
    g[::3] = 0.6 * x[::3] + 1.6 * x[1::3]**3 - 7.2 * x[1::3]**2 + 9.6 * x[1::3] - 4.8
    g[1::3] = (0.48 * x[::3] - 0.72 * x[1::3]**3 + 3.24 * x[1::3]**2 - 4.32 * x[1::3]
               - x[2::3] + 0.2 * x[2::3]**3 + 2.16)
    g[2::3] = 1.25 * x[2::3] - 0.25*x[2::3]**3
    return g


def x0_4(n):  # skip name check
    assert_equal(n % 3, 0)
    x0 = np.array([-1, 1/2, -1] * (n//3))
    return x0

def F_6(x, n):
    c = 0.9
    mu = (np.arange(1, n+1) - 0.5)/n
    return x - 1/(1 - c/(2*n) * (mu[:,None]*x / (mu[:,None] + mu)).sum(axis=1))

def x0_6(n):
    return np.ones([n])

def F_7(x, n):
    assert_equal(n % 3, 0)

    def phi(t):
        v = 0.5*t - 2
        v[t > -1] = ((-592*t**3 + 888*t**2 + 4551*t - 1924)/1998)[t > -1]
        v[t >= 2] = (0.5*t + 2)[t >= 2]
        return v
    g = np.zeros([n])
    # 将 g 数组中每隔三个元素进行赋值操作，计算结果为 1e4 * x 数组中每隔三个元素的平方减去 1
    g[::3] = 1e4 * x[1::3]**2 - 1
    # 将 g 数组中每隔三个元素后一个元素进行赋值操作，计算结果为 exp(-x 数组每隔三个元素) + exp(-x 数组每隔三个元素后一个元素) - 1.0001
    g[1::3] = exp(-x[::3]) + exp(-x[1::3]) - 1.0001
    # 将 g 数组中每隔三个元素后两个元素进行赋值操作，计算结果为调用 phi 函数对 x 数组中每隔三个元素后两个元素的计算结果
    g[2::3] = phi(x[2::3])
    # 返回计算后的 g 数组
    return g
# 确保输入的参数 n 是 3 的倍数，否则会触发断言错误
def x0_7(n):
    assert_equal(n % 3, 0)
    # 返回一个长度为 n 的 NumPy 数组，数组元素为 [1e-3, 18, 1] 的循环
    return np.array([1e-3, 18, 1] * (n//3))

# 计算给定的向量 x 和长度 n 的目标函数 F(x)
def F_9(x, n):
    # 初始化一个长度为 n 的零向量 g
    g = np.zeros([n])
    # 创建从 2 到 n-1 的整数数组 i
    i = np.arange(2, n)
    # 计算目标函数的各个分量
    g[0] = x[0]**3/3 + x[1]**2/2
    g[1:-1] = -x[1:-1]**2/2 + i*x[1:-1]**3/3 + x[2:]**2/2
    g[-1] = -x[-1]**2/2 + n*x[-1]**3/3
    return g

# 返回一个长度为 n 的全为 1 的 NumPy 数组
def x0_9(n):
    return np.ones([n])

# 计算给定的向量 x 和长度 n 的目标函数 F(x)
def F_10(x, n):
    # 返回对数函数 log(1 + x) 减去 x/n 的结果
    return np.log(1 + x) - x/n

# 返回一个长度为 n 的全为 1 的 NumPy 数组
def x0_10(n):
    return np.ones([n])
```