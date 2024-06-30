# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_lie_group.py`

```
# 导入 SymPy 库中的函数和符号类型
from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)

# 导入 SymPy 库中求解常微分方程的相关模块
from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)

# 导入 SymPy 库中用于检查常微分方程解的模块
from sympy.solvers.ode.subscheck import checkodesol

# 导入 SymPy 的测试框架中的 XFAIL 类
from sympy.testing.pytest import XFAIL

# 定义一个符号 C1
C1 = Symbol('C1')

# 定义符号 x, y，并声明它们为符号变量
x, y = symbols("x y")

# 定义函数 f, xi, eta
f = Function('f')
xi = Function('xi')
eta = Function('eta')

# 定义一个测试函数 test_heuristic1
def test_heuristic1():
    # 定义多个符号变量
    a, b, c, a4, a3, a2, a1, a0 = symbols("a b c a4 a3 a2 a1 a0")

    # 计算 f(x) 对 x 的导数
    df = f(x).diff(x)

    # 定义一系列常微分方程表达式
    eq = Eq(df, x**2*f(x))
    eq1 = f(x).diff(x) + a*f(x) - c*exp(b*x)
    eq2 = f(x).diff(x) + 2*x*f(x) - x*exp(-x**2)
    eq3 = (1 + 2*x)*df + 2 - 4*exp(-f(x))
    eq4 = f(x).diff(x) - (a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**Rational(-1, 2)
    eq5 = x**2*df - f(x) + x**2*exp(x - (1/x))

    # 将方程式放入列表中
    eqlist = [eq, eq1, eq2, eq3, eq4, eq5]

    # 对每个方程式应用 Lie 群方法求解微分方程
    i = infinitesimals(eq, hint='abaco1_simple')
    assert i == [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0},
        {eta(x, f(x)): f(x), xi(x, f(x)): 0},
        {eta(x, f(x)): 0, xi(x, f(x)): x**(-2)}]

    i1 = infinitesimals(eq1, hint='abaco1_simple')
    assert i1 == [{eta(x, f(x)): exp(-a*x), xi(x, f(x)): 0}]

    i2 = infinitesimals(eq2, hint='abaco1_simple')
    assert i2 == [{eta(x, f(x)): exp(-x**2), xi(x, f(x)): 0}]

    i3 = infinitesimals(eq3, hint='abaco1_simple')
    assert i3 == [{eta(x, f(x)): 0, xi(x, f(x)): 2*x + 1},
        {eta(x, f(x)): 0, xi(x, f(x)): 1/(exp(f(x)) - 2)}]

    i4 = infinitesimals(eq4, hint='abaco1_simple')
    assert i4 == [{eta(x, f(x)): 1, xi(x, f(x)): 0},
        {eta(x, f(x)): 0,
        xi(x, f(x)): sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)}]

    i5 = infinitesimals(eq5, hint='abaco1_simple')
    assert i5 == [{xi(x, f(x)): 0, eta(x, f(x)): exp(-1/x)}]

    # 将 Lie 群方法求解得到的结果与原方程列表逐个对应检查
    ilist = [i, i1, i2, i3, i4, i5]
    for eq, i in zip(eqlist, ilist):
        check = checkinfsol(eq, i)
        assert check[0]

    # 对特定的常微分方程使用 Lie 群方法求解，此方程需要更好的假设条件
    eq6 = df - (f(x)/x)*(x*log(x**2/f(x)) + 2)
    i = infinitesimals(eq6, hint='abaco1_product')
    assert i == [{eta(x, f(x)): f(x)*exp(-x), xi(x, f(x)): 0}]
    assert checkinfsol(eq6, i)[0]

    # 检查方程是否符合特定的假设条件
    eq7 = x*(f(x).diff(x)) + 1 - f(x)**2
    i = infinitesimals(eq7, hint='chi')
    assert checkinfsol(eq7, i)[0]


# 定义测试函数 test_heuristic3
def test_heuristic3():
    # 定义符号变量
    a, b = symbols("a b")

    # 计算 f(x) 对 x 的导数
    df = f(x).diff(x)

    # 定义常微分方程表达式
    eq = x**2*df + x*f(x) + f(x)**2 + x**2

    # 使用 Lie 群方法求解微分方程
    i = infinitesimals(eq, hint='bivariate')
    assert i == [{eta(x, f(x)): f(x), xi(x, f(x)): x}]
    assert checkinfsol(eq, i)[0]

    eq = x**2*(-f(x)**2 + df)- a*x**2*f(x) + 2 - a*x

    # 检查方程是否符合特定的假设条件
    i = infinitesimals(eq, hint='bivariate')
    assert checkinfsol(eq, i)[0]
    # 计算表达式的微分
    eq = f(x).diff(x) - (3*(1 + x**2/f(x)**2)*atan(f(x)/x) + (1 - 2*f(x))/x +
       (1 - 3*f(x))*(x/f(x)**2))
    
    # 计算微分方程的无穷小，使用函数求和的提示
    i = infinitesimals(eq, hint='function_sum')
    
    # 断言检查无穷小的解是否为指定的字典
    assert i == [{eta(x, f(x)): f(x)**(-2) + x**(-2), xi(x, f(x)): 0}]
    
    # 断言检查无穷小的解是否满足给定的微分方程
    assert checkinfsol(eq, i)[0]
# 定义测试函数 test_heuristic_abaco2_similar
def test_heuristic_abaco2_similar():
    # 使用 sympy 的 symbols 函数定义符号变量 a 和 b
    a, b = symbols("a b")
    # 使用 sympy 的 Function 函数定义函数 F
    F = Function('F')
    # 定义微分方程 eq，表示 f(x) 对 x 的导数等于 F(a*x + b*f(x))
    eq = f(x).diff(x) - F(a*x + b*f(x))
    # 调用 infinitesimals 函数，使用 hint='abaco2_similar' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='abaco2_similar')
    # 断言求解结果 i 应为 [{eta(x, f(x)): -a/b, xi(x, f(x)): 1}]
    assert i == [{eta(x, f(x)): -a/b, xi(x, f(x)): 1}]
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]

    # 修改微分方程 eq，表示 f(x) 对 x 的导数等于 f(x)^2 / (sin(f(x) - x) - x**2 + 2*x*f(x))
    eq = f(x).diff(x) - (f(x)**2 / (sin(f(x) - x) - x**2 + 2*x*f(x)))
    # 再次调用 infinitesimals 函数，使用 hint='abaco2_similar' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='abaco2_similar')
    # 断言求解结果 i 应为 [{eta(x, f(x)): f(x)**2, xi(x, f(x)): f(x)**2}]
    assert i == [{eta(x, f(x)): f(x)**2, xi(x, f(x)): f(x)**2}]
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]


# 定义测试函数 test_heuristic_abaco2_unique_unknown
def test_heuristic_abaco2_unique_unknown():
    # 使用 sympy 的 symbols 函数定义符号变量 a 和 b
    a, b = symbols("a b")
    # 使用 sympy 的 Function 函数定义函数 F
    F = Function('F')
    # 定义微分方程 eq，表示 f(x) 对 x 的导数等于 x**(a - 1)*(f(x)**(1 - b))*F(x**a/a + f(x)**b/b)
    eq = f(x).diff(x) - x**(a - 1)*(f(x)**(1 - b))*F(x**a/a + f(x)**b/b)
    # 调用 infinitesimals 函数，使用 hint='abaco2_unique_unknown' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    # 断言求解结果 i 应为 [{eta(x, f(x)): -f(x)*f(x)**(-b), xi(x, f(x)): x*x**(-a)}]
    assert i == [{eta(x, f(x)): -f(x)*f(x)**(-b), xi(x, f(x)): x*x**(-a)}]
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]

    # 修改微分方程 eq，表示 f(x) 对 x 的导数等于 tan(F(x**2 + f(x)**2) + atan(x/f(x)))
    eq = f(x).diff(x) + tan(F(x**2 + f(x)**2) + atan(x/f(x)))
    # 再次调用 infinitesimals 函数，使用 hint='abaco2_unique_unknown' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    # 断言求解结果 i 应为 [{eta(x, f(x)): x, xi(x, f(x)): -f(x)}]
    assert i == [{eta(x, f(x)): x, xi(x, f(x)): -f(x)}]
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]

    # 修改微分方程 eq，无法通过现有的函数库解决该微分方程
    eq = (x*f(x).diff(x) + f(x) + 2*x)**2 -4*x*f(x) -4*x**2 -4*a
    # 调用 infinitesimals 函数，使用 hint='abaco2_unique_unknown' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]


# 定义测试函数 test_heuristic_linear
def test_heuristic_linear():
    # 使用 sympy 的 symbols 函数定义符号变量 a, b, m, n
    a, b, m, n = symbols("a b m n")

    # 定义线性微分方程 eq
    eq = x**(n*(m + 1) - m)*(f(x).diff(x)) - a*f(x)**n -b*x**(n*(m + 1))
    # 调用 infinitesimals 函数，使用 hint='linear' 求解微分方程的无穷小
    i = infinitesimals(eq, hint='linear')
    # 断言使用求解结果检查微分方程是否满足无穷小条件
    assert checkinfsol(eq, i)[0]


# 定义测试函数 test_user_infinitesimals
def test_user_infinitesimals():
    # 使用 sympy 的 Symbol 函数定义符号变量 x
    x = Symbol("x")  # assuming x is real generates an error

    # 定义微分方程 eq
    eq = x*(f(x).diff(x)) + 1 - f(x)**2
    # 定义微分方程的解 sol
    sol = Eq(f(x), (C1 + x**2)/(C1 - x**2))
    # 定义无穷小字典 infinitesimals
    infinitesimals = {'xi':sqrt(f(x) - 1)/sqrt(f(x) + 1), 'eta':0}
    # 使用 dsolve 函数，使用 hint='lie_group' 和给定的无穷小解求解微分方程
    assert dsolve(eq, hint='lie_group', **infinitesimals) == sol
    # 使用 checkodesol 函数检查微分方程的解是否正确
    assert checkodesol(eq, sol) == (True, 0)


# 定义测试函数 test_lie_group_issue15219
@XFAIL
def test_lie_group_issue15219():
    # 使用 sympy 的 symbols 函数定义符号变量 a, b, alpha, c
    a, b, alpha, c = symbols("a b alpha c")
    # 定义微分方程 eqn
    eqn = exp(f(x).diff(x)-f(x))
    # 使用 classify_ode 函数检查微分方程 eqn 是否属于 'lie_group'
    assert 'lie_group' not in classify_ode(eqn, f(x))
```