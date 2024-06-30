# `D:\src\scipysrc\sympy\sympy\polys\matrices\tests\test_linsolve.py`

```
# 导入必要的模块和函数
from sympy.testing.pytest import raises  # 导入 raises 函数用于测试异常
from sympy.core.numbers import I  # 导入虚数单位 I
from sympy.core.relational import Eq  # 导入方程类 Eq
from sympy.core.singleton import S  # 导入 SymPy 中的单例 S
from sympy.abc import x, y, z  # 导入符号 x, y, z

from sympy.polys.matrices.linsolve import _linsolve  # 导入线性求解函数 _linsolve
from sympy.polys.solvers import PolyNonlinearError  # 导入多项式非线性异常类 PolyNonlinearError

# 定义测试函数 test__linsolve
def test__linsolve():
    # 测试空方程组的情况，期望返回 {x: x}
    assert _linsolve([], [x]) == {x:x}

    # 测试方程组只包含零的情况，期望返回 {x: x}
    assert _linsolve([S.Zero], [x]) == {x:x}

    # 测试无解的情况，期望返回 None
    assert _linsolve([x-1,x-2], [x]) is None

    # 测试单一方程的情况，期望返回 {x: 1}
    assert _linsolve([x-1], [x]) == {x:1}

    # 测试包含多个变量的方程组，期望返回 {x: 1, y: 0}
    assert _linsolve([x-1, y], [x, y]) == {x:1, y:S.Zero}

    # 测试包含复数的情况，期望返回 None
    assert _linsolve([2*I], [x]) is None

    # 测试引发多项式非线性异常的情况
    raises(PolyNonlinearError, lambda: _linsolve([x*(1 + x)], [x]))


def test__linsolve_float():
    # 测试精确解的情况
    eqs = [
        y - x,
        y - 0.0216 * x
    ]
    sol = {x:0.0, y:0.0}
    assert _linsolve(eqs, (x, y)) == sol

    # 测试接近 eps 的其他情况
    def all_close(sol1, sol2, eps=1e-15):
        close = lambda a, b: abs(a - b) < eps
        assert sol1.keys() == sol2.keys()
        return all(close(sol1[s], sol2[s]) for s in sol1)

    # 测试具体方程组的解
    eqs = [
        0.8*x + 0.8*z + 0.2,
        0.9*x + 0.7*y + 0.2*z + 0.9,
        0.7*x + 0.2*y + 0.2*z + 0.5
    ]
    sol_exact = {x:-29/42, y:-11/21, z:37/84}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    # 更多的方程组测试，检查数值解是否接近预期
    eqs = [
        0.9*x + 0.3*y + 0.4*z + 0.6,
        0.6*x + 0.9*y + 0.1*z + 0.7,
        0.4*x + 0.6*y + 0.9*z + 0.5
    ]
    sol_exact = {x:-88/175, y:-46/105, z:-1/25}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        0.4*x + 0.3*y + 0.6*z + 0.7,
        0.4*x + 0.3*y + 0.9*z + 0.9,
        0.7*x + 0.9*y,
    ]
    sol_exact = {x:-9/5, y:7/5, z:-2/3}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        x*(0.7 + 0.6*I) + y*(0.4 + 0.7*I) + z*(0.9 + 0.1*I) + 0.5,
        0.2*I*x + 0.2*I*y + z*(0.9 + 0.2*I) + 0.1,
        x*(0.9 + 0.7*I) + y*(0.9 + 0.7*I) + z*(0.9 + 0.4*I) + 0.4,
    ]
    sol_exact = {
        x:-6157/7995 - 411/5330*I,
        y:8519/15990 + 1784/7995*I,
        z:-34/533 + 107/1599*I,
    }
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    # 注释掉的方程组，包含有关 RR(z) 上 x 和 y 的系统
    #
    # eqs = [
    #     x*(0.2*z + 0.9) + y*(0.5*z + 0.8) + 0.6,
    #     0.1*x*z + y*(0.1*z + 0.6) + 0.9,
    # ]
    #
    # linsolve(eqs, [x, y])
    # 解为 x 的解为
    #
    #       -3.9e-5*z**2 - 3.6e-5*z - 8.67361737988404e-20
    #  x =  ----------------------------------------------
    #           3.0e-6*z**3 - 1.3e-5*z**2 - 5.4e-5*z
    #
    # 分子中的 8e-20 应为零，从而允许 z 取消
    # 顶部和底部。某种方式应该避免这种情况，因为
    #
    # 矩阵的逆矩阵在分母中仅有一个二次因子（行列式）。
# 定义一个名为 test__linsolve_deprecated 的测试函数，用于测试 _linsolve 函数在处理特定情况下是否会引发 PolyNonlinearError 异常。
def test__linsolve_deprecated():
    # 断言：调用 _linsolve 函数，传入一个方程列表和变量列表，预期会引发 PolyNonlinearError 异常
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq(x**2, x**2 + y)], [x, y]))
    # 断言：调用 _linsolve 函数，传入一个包含单个方程的列表和变量列表，预期会引发 PolyNonlinearError 异常
    raises(PolyNonlinearError, lambda:
        _linsolve([(x + y)**2 - x**2], [x]))
    # 断言：调用 _linsolve 函数，传入一个包含单个方程的列表和变量列表，预期会引发 PolyNonlinearError 异常
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq((x + y)**2, x**2)], [x]))
```