# `D:\src\scipysrc\sympy\sympy\polys\tests\test_solvers.py`

```
"""Tests for low-level linear systems solver. """

# 导入所需的模块和函数
from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix


# 定义测试函数：解一个2x2线性系统有解的情况
def test_solve_lin_sys_2x2_one():
    domain, x1,x2 = ring("x1,x2", QQ)
    # 定义线性方程组
    eqs = [x1 + x2 - 5,
           2*x1 - x2]
    # 预期解
    sol = {x1: QQ(5, 3), x2: QQ(10, 3)}
    # 调用求解函数
    _sol = solve_lin_sys(eqs, domain)
    # 断言结果与预期相符，并且所有解的类型都是 domain.dtype 类型
    assert _sol == sol and all(isinstance(s, domain.dtype) for s in _sol)


# 定义测试函数：解一个2x4线性系统无解的情况
def test_solve_lin_sys_2x4_none():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs = [x1 - 1,
           x1 - x2,
           x1 - 2*x2,
           x2 - 1]
    # 断言解不存在
    assert solve_lin_sys(eqs, domain) is None


# 定义测试函数：解一个3x4线性系统有唯一解的情况
def test_solve_lin_sys_3x4_one():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 + 2*x2 + 3*x3,
           2*x1 - x2 + x3,
           3*x1 + x2 + x3,
           5*x2 + 2*x3]
    # 预期解
    sol = {x1: 0, x2: 0, x3: 0}
    # 断言结果与预期相符
    assert solve_lin_sys(eqs, domain) == sol


# 定义测试函数：解一个3x3线性系统有无穷解的情况
def test_solve_lin_sys_3x3_inf():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 - x2 + 2*x3 - 1,
           2*x1 + x2 + x3 - 8,
           x1 + x2 - 5]
    # 预期解
    sol = {x1: -x3 + 3, x2: x3 + 2}
    # 断言结果与预期相符
    assert solve_lin_sys(eqs, domain) == sol


# 定义测试函数：解一个3x4线性系统无解的情况
def test_solve_lin_sys_3x4_none():
    domain, x1,x2,x3,x4 = ring("x1,x2,x3,x4", QQ)
    eqs = [2*x1 + x2 + 7*x3 - 7*x4 - 2,
           -3*x1 + 4*x2 - 5*x3 - 6*x4 - 3,
           x1 + x2 + 4*x3 - 5*x4 - 2]
    # 断言解不存在
    assert solve_lin_sys(eqs, domain) is None


# 定义测试函数：解一个4x7线性系统有无穷解的情况
def test_solve_lin_sys_4x7_inf():
    domain, x1,x2,x3,x4,x5,x6,x7 = ring("x1,x2,x3,x4,x5,x6,x7", QQ)
    eqs = [x1 + 4*x2 - x4 + 7*x6 - 9*x7 - 3,
           2*x1 + 8*x2 - x3 + 3*x4 + 9*x5 - 13*x6 + 7*x7 - 9,
           2*x3 - 3*x4 - 4*x5 + 12*x6 - 8*x7 - 1,
           -x1 - 4*x2 + 2*x3 + 4*x4 + 8*x5 - 31*x6 + 37*x7 - 4]
    # 预期解
    sol = {x1: 4 - 4*x2 - 2*x5 - x6 + 3*x7,
           x3: 2 - x5 + 3*x6 - 5*x7,
           x4: 1 - 2*x5 + 6*x6 - 6*x7}
    # 断言结果与预期相符
    assert solve_lin_sys(eqs, domain) == sol


# 定义测试函数：解一个5x5线性系统有无穷解的情况
def test_solve_lin_sys_5x5_inf():
    domain, x1,x2,x3,x4,x5 = ring("x1,x2,x3,x4,x5", QQ)
    eqs = [x1 - x2 - 2*x3 + x4 + 11*x5 - 13,
           x1 - x2 + x3 + x4 + 5*x5 - 16,
           2*x1 - 2*x2 + x4 + 10*x5 - 21,
           2*x1 - 2*x2 - x3 + 3*x4 + 20*x5 - 38,
           2*x1 - 2*x2 + x3 + x4 + 8*x5 - 22]
    # 预期解
    sol = {x1: 6 + x2 - 3*x5,
           x3: 1 + 2*x5,
           x4: 9 - 4*x5}
    # 断言结果与预期相符
    assert solve_lin_sys(eqs, domain) == sol


# 定义测试函数：解一个6x6线性系统有唯一解的情况
def test_solve_lin_sys_6x6_1():
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    eqs = [b + q/d - c/d,
           c*(1/d + 1/e + 1/g) - f/g - q/d,
           f*(1/g + 1/i + 1/j) - c/g - h/i,
           h*(1/i + 1/l + 1/m) - f/i - k/m,
           k*(1/m + 1/o + 1/p) - h/m - n/p,
           n/p - k/p]
    # 定义一个包含线性方程组解的字典 `sol`，键为变量名，值为对应的解表达式
    sol = {
         b: (e*i*l*q + e*i*m*q + e*i*o*q + e*j*l*q + e*j*m*q + e*j*o*q + e*l*m*q + e*l*o
# 定义一个测试函数，用于解决一个包含6个变量的线性方程组
def test_solve_lin_sys_6x6_2():
    # 在整数环中定义变量：d, r, e, g, i, j, l, o, m, p, q
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    # 在定义的整数环中再定义变量：c, f, h, k, n, b
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    # 定义线性方程组的表达式列表
    eqs = [
        b + r/d - c/d,
        c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p
    ]
    # 闭合花括号多余，应删除

    # 断言解线性方程组的函数返回的结果与预期解相等
    assert solve_lin_sys(eqs, domain) == sol

# 定义一个测试函数，用于将方程组转换为系数矩阵
def test_eqs_to_matrix():
    # 在有理数域中定义变量：x1, x2
    domain, x1,x2 = ring("x1,x2", QQ)
    # 定义方程组的系数字典列表
    eqs_coeff = [{x1: QQ(1), x2: QQ(1)}, {x1: QQ(2), x2: QQ(-1)}]
    # 定义方程组的右侧值列表
    eqs_rhs = [QQ(-5), QQ(0)]
    # 调用将方程组转换为矩阵的函数，指定有理数域
    M = eqs_to_matrix(eqs_coeff, eqs_rhs, [x1, x2], QQ)
    # 断言转换后的矩阵对象与预期的矩阵相等
    assert M.to_Matrix() == Matrix([[1, 1, 5], [2, -1, 0]])
```