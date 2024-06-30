# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_subscheck.py`

```
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.error_functions import (Ei, erf, erfi)
from sympy.integrals.integrals import Integral

from sympy.solvers.ode.subscheck import checkodesol, checksysodesol

from sympy.functions import besselj, bessely

from sympy.testing.pytest import raises, slow


# 定义一些符号变量
C0, C1, C2, C3, C4 = symbols('C0:5')
u, x, y, z = symbols('u,x:z', real=True)
# 定义函数符号
f = Function('f')
g = Function('g')
h = Function('h')


@slow
def test_checkodesol():
    # 对于大部分情况，checkodesol 在下面的测试中已经进行了良好的测试。
    # 这些测试仅处理下面未检查的情况。
    # 测试对于形如 f(x, y).diff(x) 的方程，检查其是否为 ODE，并引发 ValueError
    raises(ValueError, lambda: checkodesol(f(x, y).diff(x), Eq(f(x, y), x)))
    # 测试对于形如 f(x).diff(x) 的方程，检查其是否为 ODE，并引发 ValueError
    raises(ValueError, lambda: checkodesol(f(x).diff(x), Eq(f(x, y),
           x), f(x, y)))
    # 检查 f(x).diff(x) 是否为 ODE，并返回 False 及其余部分
    assert checkodesol(f(x).diff(x), Eq(f(x, y), x)) == \
        (False, -f(x).diff(x) + f(x, y).diff(x) - 1)
    # 检查 f(x).diff(x) 是否为 ODE，并返回不是 True 及其余部分
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) is not True
    # 检查 f(x).diff(x) 是否为 ODE，并返回 False 及其余部分
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) == (False, 1)
    # 检查对于给定的方程 sol1，其导数是否为 ODE，并返回 True 及其余部分
    sol1 = Eq(f(x)**5 + 11*f(x) - 2*f(x) + x, 0)
    assert checkodesol(diff(sol1.lhs, x), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其导数乘以 exp(f(x)) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(sol1.lhs, x)*exp(f(x)), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其二阶导数是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(sol1.lhs, x, 2), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其二阶导数乘以 exp(f(x)) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(sol1.lhs, x, 2)*exp(f(x)), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其三阶导数是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(sol1.lhs, x, 3), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其三阶导数乘以 exp(f(x)) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(sol1.lhs, x, 3)*exp(f(x)), sol1) == (True, 0)
    # 检查对于给定的方程 sol1，其三阶导数是否为 ODE，并返回 False 及其余部分
    assert checkodesol(diff(sol1.lhs, x, 3), Eq(f(x), x*log(x))) == \
        (False, 60*x**4*((log(x) + 1)**2 + log(x))*(
        log(x) + 1)*log(x)**2 - 5*x**4*log(x)**4 - 9)
    # 检查对于给定的方程 exp(f(x)) + x 的导数乘以 x 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0)) == \
        (True, 0)
    # 检查对于给定的方程 exp(f(x)) + x 的导数乘以 x 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0),
        solve_for_func=False) == (True, 0)
    # 检查对于给定的方程 f(x).diff(x, 2) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(f(x).diff(x, 2), [Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)]) == \
        [(True, 0), (True, 0), (False, C2)]
    # 检查对于给定的方程 f(x).diff(x, 2) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(f(x).diff(x, 2), {Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)}) == \
        {(True, 0), (True, 0), (False, C2)}
    # 检查对于给定的方程 f(x).diff(x) - 1/f(x)/2 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(f(x).diff(x) - 1/f(x)/2, Eq(f(x)**2, x)) == \
        [(True, 0), (True, 0)]
    # 检查对于给定的方程 f(x).diff(x) - f(x) 是否为 ODE，并返回 True 及其余部分
    assert checkodesol(f(x).diff(x) - f(x), Eq(C1*exp(x), f(x))) == (True, 0)
    # 根据 test_1st_homogeneous_coeff_ode2_eq3sol，确保 checkodesol 在可以时尝试反向替换 f(x)
    eq3 = x*exp(f(x)/x) + f(x) - x*f(x).diff(x)
    sol3 = Eq(f(x), log(log(C1/x)**(-x)))
    # 检查是否没有 f(x) 在 checkodesol 中尝试反向替换
    assert not checkodesol(eq3, sol3)[1].has(f(x))
    # 定义微分方程 Eq 对象，测试 Derivative 函数的解析性
    eqn = Eq(Derivative(x*Derivative(f(x), x), x)/x, exp(x))
    # 定义期望的解 sol，包含未知常数 C1、C2，以及特殊函数 Ei(x)
    sol = Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))
    # 使用 checkodesol 函数检查微分方程 eqn 的解 sol，不解出函数
    assert checkodesol(eqn, sol, order=2, solve_for_func=False)[0]
    
    # 定义二阶线性常系数微分方程 eq
    eq = x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (2*x**2 + 25)*f(x)
    # 定义期望的解 sol，包含未知常数 C1、C2，以及 besselj 和 bessely 函数
    sol = Eq(f(x), C1*besselj(5*I, sqrt(2)*x) + C2*bessely(5*I, sqrt(2)*x))
    # 使用 checkodesol 函数检查微分方程 eq 的解 sol
    assert checkodesol(eq, sol) == (True, 0)

    # 定义一组微分方程 eqs，包含 f(x) 和 g(x) 的关系
    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    # 定义期望的解 sol，包含未知常数 C1、C2，以及指数函数 exp(2*x)
    sol = [Eq(f(x), -C1 + C2*exp(2*x)), Eq(g(x), C1 + C2*exp(2*x))]
    # 使用 checkodesol 函数检查一组微分方程 eqs 的解 sol
    assert checkodesol(eqs, sol) == (True, [0, 0])
def test_checksysodesol():
    # 定义符号变量 x, y, z 为函数
    x, y, z = symbols('x, y, z', cls=Function)
    # 定义符号变量 t
    t = Symbol('t')

    # 定义第一个方程组
    eq = (Eq(diff(x(t),t), 9*y(t)), Eq(diff(y(t),t), 12*x(t)))
    # 解的期望值
    sol = [Eq(x(t), 9*C1*exp(-6*sqrt(3)*t) + 9*C2*exp(6*sqrt(3)*t)),
           Eq(y(t), -6*sqrt(3)*C1*exp(-6*sqrt(3)*t) + 6*sqrt(3)*C2*exp(6*sqrt(3)*t))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第二个方程组
    eq = (Eq(diff(x(t),t), 2*x(t) + 4*y(t)), Eq(diff(y(t),t), 12*x(t) + 41*y(t)))
    # 解的期望值
    sol = [Eq(x(t), 4*C1*exp(t*(-sqrt(1713)/2 + Rational(43, 2))) + 4*C2*exp(t*(sqrt(1713)/2 + Rational(43, 2)))),
           Eq(y(t), C1*(-sqrt(1713)/2 + Rational(39, 2))*exp(t*(-sqrt(1713)/2 + Rational(43, 2))) + C2*(Rational(39, 2) + sqrt(1713)/2)*exp(t*(sqrt(1713)/2 + Rational(43, 2))))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第三个方程组
    eq = (Eq(diff(x(t),t), x(t) + y(t)), Eq(diff(y(t),t), -2*x(t) + 2*y(t)))
    # 解的期望值
    sol = [Eq(x(t), (C1*sin(sqrt(7)*t/2) + C2*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2))),
           Eq(y(t), ((C1/2 - sqrt(7)*C2/2)*sin(sqrt(7)*t/2) + (sqrt(7)*C1/2 + C2/2)*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2)))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第四个方程组
    eq = (Eq(diff(x(t),t), x(t) + y(t) + 9), Eq(diff(y(t),t), 2*x(t) + 5*y(t) + 23))
    # 解的期望值
    sol = [Eq(x(t), C1*exp(t*(-sqrt(6) + 3)) + C2*exp(t*(sqrt(6) + 3)) - Rational(22, 3)),
           Eq(y(t), C1*(-sqrt(6) + 2)*exp(t*(-sqrt(6) + 3)) + C2*(2 + sqrt(6))*exp(t*(sqrt(6) + 3)) - Rational(5, 3))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第五个方程组
    eq = (Eq(diff(x(t),t), x(t) + y(t) + 81), Eq(diff(y(t),t), -2*x(t) + y(t) + 23))
    # 解的期望值
    sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - Rational(58, 3)),
           Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - Rational(185, 3))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第六个方程组
    eq = (Eq(diff(x(t),t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t),t), 2*x(t) + 5*t*y(t)))
    # 解的期望值
    sol = [Eq(x(t), (C1*exp(Integral(2, t).doit()) + C2*exp(-(Integral(2, t)).doit()))*exp((Integral(5*t, t)).doit())),
           Eq(y(t), (C1*exp((Integral(2, t)).doit()) - C2*exp(-(Integral(2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第七个方程组
    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    # 解的期望值
    sol = [Eq(x(t), (C1*cos((Integral(t**2, t)).doit()) + C2*sin((Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit())),
           Eq(y(t), (-C1*sin((Integral(t**2, t)).doit()) + C2*cos((Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    # 断言解是否符合方程组
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第八个方程组的第一部分
    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + (5*t+9*t**2)*y(t)))
    # 解的期望值的第一部分
    sol = [Eq(x(t), (C1*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + C2*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit())), \
    # 定义一个包含微分方程组的表达式
    eq = (Eq(y(t), (C1*(-sqrt(77)/2 + Rational(9, 2))*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + \
    C2*(sqrt(77)/2 + Rational(9, 2))*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第二组微分方程组的表达式
    eq = (Eq(diff(x(t),t,t), 5*x(t) + 43*y(t)), Eq(diff(y(t),t,t), x(t) + 9*y(t)))
    # 计算并定义方程的四个根
    root0 = -sqrt(-sqrt(47) + 7)
    root1 = sqrt(-sqrt(47) + 7)
    root2 = -sqrt(sqrt(47) + 7)
    root3 = sqrt(sqrt(47) + 7)
    # 定义方程组的解
    sol = [Eq(x(t), 43*C1*exp(t*root0) + 43*C2*exp(t*root1) + 43*C3*exp(t*root2) + 43*C4*exp(t*root3)), \
    Eq(y(t), C1*(root0**2 - 5)*exp(t*root0) + C2*(root1**2 - 5)*exp(t*root1) + \
    C3*(root2**2 - 5)*exp(t*root2) + C4*(root3**2 - 5)*exp(t*root3))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第三组微分方程组的表达式
    eq = (Eq(diff(x(t),t,t), 8*x(t)+3*y(t)+31), Eq(diff(y(t),t,t), 9*x(t)+7*y(t)+12))
    # 计算并定义方程的四个根
    root0 = -sqrt(-sqrt(109)/2 + Rational(15, 2))
    root1 = sqrt(-sqrt(109)/2 + Rational(15, 2))
    root2 = -sqrt(sqrt(109)/2 + Rational(15, 2))
    root3 = sqrt(sqrt(109)/2 + Rational(15, 2))
    # 定义方程组的解
    sol = [Eq(x(t), 3*C1*exp(t*root0) + 3*C2*exp(t*root1) + 3*C3*exp(t*root2) + 3*C4*exp(t*root3) - Rational(181, 29)), \
    Eq(y(t), C1*(root0**2 - 8)*exp(t*root0) + C2*(root1**2 - 8)*exp(t*root1) + \
    C3*(root2**2 - 8)*exp(t*root2) + C4*(root3**2 - 8)*exp(t*root3) + Rational(183, 29))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第四组微分方程组的表达式
    eq = (Eq(diff(x(t),t,t) - 9*diff(y(t),t) + 7*x(t),0), Eq(diff(y(t),t,t) + 9*diff(x(t),t) + 7*y(t),0))
    # 定义方程组的解
    sol = [Eq(x(t), C1*cos(t*(Rational(9, 2) + sqrt(109)/2)) + C2*sin(t*(Rational(9, 2) + sqrt(109)/2)) + \
    C3*cos(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*sin(t*(-sqrt(109)/2 + Rational(9, 2)))), Eq(y(t), -C1*sin(t*(Rational(9, 2) + sqrt(109)/2)) \
    + C2*cos(t*(Rational(9, 2) + sqrt(109)/2)) - C3*sin(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*cos(t*(-sqrt(109)/2 + Rational(9, 2))))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第五组微分方程组的表达式
    eq = (Eq(diff(x(t),t,t), 9*t*diff(y(t),t)-9*y(t)), Eq(diff(y(t),t,t),7*t*diff(x(t),t)-7*x(t)))
    # 定义方程组的两个积分表达式
    I1 = sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erfi(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(3*sqrt(7)*t**2/2)/t
    I2 = -sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erf(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(-3*sqrt(7)*t**2/2)/t
    # 定义方程组的解
    sol = [Eq(x(t), C3*t + t*(9*C1*I1 + 9*C2*I2)), Eq(y(t), C4*t + t*(3*sqrt(7)*C1*I1 - 3*sqrt(7)*C2*I2))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0])

    # 定义第六组微分方程组的表达式
    eq = (Eq(diff(x(t),t), 21*x(t)), Eq(diff(y(t),t), 17*x(t)+3*y(t)), Eq(diff(z(t),t), 5*x(t)+7*y(t)+9*z(t)))
    # 定义方程组的解
    sol = [Eq(x(t), C1*exp(21*t)), Eq(y(t), 17*C1*exp(21*t)/18 + C2*exp(3*t)), \
    Eq(z(t), 209*C1*exp(21*t)/216 - 7*C2*exp(3*t)/6 + C3*exp(9*t))]
    # 断言微分方程组的解符合预期
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    # 定义第七组微分方程组的表达式
    eq = (Eq(diff(x(t),t),3*y(t)-11*z(t)),Eq(diff(y(t),t),7*z(t)-3*x(t)),Eq(diff(z(t),t),11*x(t)-7*y(t)))
    sol = [Eq(x(t), 7*C0 + sqrt(179)*C1*cos(sqrt(179)*t) + (77*C1/3 + 130*C2/3)*sin(sqrt(179)*t)), \
    Eq(y(t), 11*C0 + sqrt(179)*C2*cos(sqrt(179)*t) + (-58*C1/3 - 77*C2/3)*sin(sqrt(179)*t)), \
    Eq(z(t), 3*C0 + sqrt(179)*(-7*C1/3 - 11*C2/3)*cos(sqrt(179)*t) + (11*C1 - 7*C2)*sin(sqrt(179)*t))]
    # 设置解的列表，包含三个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0, 0])
    
    eq = (Eq(3*diff(x(t),t),4*5*(y(t)-z(t))),Eq(4*diff(y(t),t),3*5*(z(t)-x(t))),Eq(5*diff(z(t),t),3*4*(x(t)-y(t))))
    sol = [Eq(x(t), C0 + 5*sqrt(2)*C1*cos(5*sqrt(2)*t) + (12*C1/5 + 164*C2/15)*sin(5*sqrt(2)*t)), \
    Eq(y(t), C0 + 5*sqrt(2)*C2*cos(5*sqrt(2)*t) + (-51*C1/10 - 12*C2/5)*sin(5*sqrt(2)*t)), \
    Eq(z(t), C0 + 5*sqrt(2)*(-9*C1/25 - 16*C2/25)*cos(5*sqrt(2)*t) + (12*C1/5 - 12*C2/5)*sin(5*sqrt(2)*t))]
    # 设置解的列表，包含三个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0, 0])
    
    eq = (Eq(diff(x(t),t),4*x(t) - z(t)),Eq(diff(y(t),t),2*x(t)+2*y(t)-z(t)),Eq(diff(z(t),t),3*x(t)+y(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t) + C3*exp(2*t)), \
    Eq(y(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t)), \
    Eq(z(t), 2*C1*exp(2*t) + 2*C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t) + C3*t*exp(2*t) + C3*exp(2*t))]
    # 设置解的列表，包含三个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0, 0])
    
    eq = (Eq(diff(x(t),t),4*x(t) - y(t) - 2*z(t)),Eq(diff(y(t),t),2*x(t) + y(t)- 2*z(t)),Eq(diff(z(t),t),5*x(t)-3*z(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), \
    Eq(y(t), C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), \
    Eq(z(t), C1*exp(2*t) + 5*C2*cos(t) + 5*C3*sin(t))]
    # 设置解的列表，包含三个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0, 0])
    
    eq = (Eq(diff(x(t),t),x(t)*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    sol = [Eq(x(t), C1*exp((-1/(4*C2 + 4*t))**(Rational(-1, 4)))), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-1/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    # 设置解的列表，包含两个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0])
    
    eq = (Eq(diff(x(t),t), exp(3*x(t))*y(t)**3),Eq(diff(y(t),t), y(t)**5))
    sol = [Eq(x(t), -log(C1 - 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 - 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    # 设置解的列表，包含两个微分方程的解表达式，每个表达式用符号方程（Eq）封装
    assert checksysodesol(eq, sol) == (True, [0, 0])
    # 使用 checksysodesol 函数检查解 sol 是否符合微分方程 eq 的解，期望结果是 (True, [0, 0])
    
    eq = (Eq(x(t),t*diff(x(t),t)+diff(x(t),t)*diff(y(t),t)), Eq(y(t),t*diff(y(t),t)+diff(y(t),t)**2))
    # 设置包含两个微分方程的方程组 eq
    # 创建一个包含两个方程的字典，表示解为 x(t) 和 y(t)
    sol = {Eq(x(t), C1*C2 + C1*t), Eq(y(t), C2**2 + C2*t)}
    # 断言检查给定的微分方程组 eq 在解 sol 下是否成立，期望返回结果为 True，并且在 t=0 时解的值为 [0, 0]
    assert checksysodesol(eq, sol) == (True, [0, 0])
```