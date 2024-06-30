# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_heurisch.py`

```
from sympy.concrete.summations import Sum  # 导入 Sum 类
from sympy.core.add import Add  # 导入 Add 类
from sympy.core.function import (Derivative, Function, diff)  # 导入 Derivative, Function, diff 函数和类
from sympy.core.numbers import (I, Rational, pi)  # 导入 I, Rational, pi 常数和类
from sympy.core.relational import Eq, Ne  # 导入 Eq, Ne 类
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol, symbols 符号和类
from sympy.functions.elementary.exponential import (LambertW, exp, log)  # 导入 LambertW, exp, log 函数
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)  # 导入 asinh, cosh, sinh, tanh 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 类
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)  # 导入 acos, asin, atan, cos, sin, tan 函数
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)  # 导入 besselj, besselk, bessely, jn 函数
from sympy.functions.special.error_functions import erf  # 导入 erf 函数
from sympy.integrals.integrals import Integral  # 导入 Integral 类
from sympy.logic.boolalg import And  # 导入 And 类
from sympy.matrices import Matrix  # 导入 Matrix 类
from sympy.simplify.ratsimp import ratsimp  # 导入 ratsimp 函数
from sympy.simplify.simplify import simplify  # 导入 simplify 函数
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper  # 导入 components, heurisch, heurisch_wrapper 函数
from sympy.testing.pytest import XFAIL, slow  # 导入 XFAIL, slow 装饰器
from sympy.integrals.integrals import integrate  # 导入 integrate 函数

x, y, z, nu = symbols('x,y,z,nu')  # 创建符号 x, y, z, nu
f = Function('f')  # 创建函数 f


def test_components():  # 定义测试函数 test_components
    assert components(x*y, x) == {x}  # 断言 components 函数对 x*y 关于 x 的成分是 {x}
    assert components(1/(x + y), x) == {x}  # 断言 components 函数对 1/(x + y) 关于 x 的成分是 {x}
    assert components(sin(x), x) == {sin(x), x}  # 断言 components 函数对 sin(x) 关于 x 的成分是 {sin(x), x}
    assert components(sin(x)*sqrt(log(x)), x) == \
        {log(x), sin(x), sqrt(log(x)), x}  # 断言 components 函数对 sin(x)*sqrt(log(x)) 关于 x 的成分是 {log(x), sin(x), sqrt(log(x)), x}
    assert components(x*sin(exp(x)*y), x) == \
        {sin(y*exp(x)), x, exp(x)}  # 断言 components 函数对 x*sin(exp(x)*y) 关于 x 的成分是 {sin(y*exp(x)), x, exp(x)}
    assert components(x**Rational(17, 54)/sqrt(sin(x)), x) == \
        {sin(x), x**Rational(1, 54), sqrt(sin(x)), x}  # 断言 components 函数对 x**Rational(17, 54)/sqrt(sin(x)) 关于 x 的成分是 {sin(x), x**Rational(1, 54), sqrt(sin(x)), x}

    assert components(f(x), x) == \
        {x, f(x)}  # 断言 components 函数对 f(x) 关于 x 的成分是 {x, f(x)}
    assert components(Derivative(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x)}  # 断言 components 函数对 Derivative(f(x), x) 关于 x 的成分是 {x, f(x), Derivative(f(x), x)}
    assert components(f(x)*diff(f(x), x), x) == \
        {x, f(x), Derivative(f(x), x), Derivative(f(x), x)}  # 断言 components 函数对 f(x)*diff(f(x), x) 关于 x 的成分是 {x, f(x), Derivative(f(x), x), Derivative(f(x), x)}


def test_issue_10680():  # 定义测试函数 test_issue_10680
    assert isinstance(integrate(x**log(x**log(x**log(x))),x), Integral)  # 断言 integrate(x**log(x**log(x**log(x))),x) 返回的对象是 Integral 类的实例


def test_issue_21166():  # 定义测试函数 test_issue_21166
    assert integrate(sin(x/sqrt(abs(x))), (x, -1, 1)) == 0  # 断言 integrate(sin(x/sqrt(abs(x))), (x, -1, 1)) 的结果等于 0


def test_heurisch_polynomials():  # 定义测试函数 test_heurisch_polynomials
    assert heurisch(1, x) == x  # 断言 heurisch(1, x) 的结果是 x
    assert heurisch(x, x) == x**2/2  # 断言 heurisch(x, x) 的结果是 x**2/2
    assert heurisch(x**17, x) == x**18/18  # 断言 heurisch(x**17, x) 的结果是 x**18/18
    # For coverage
    assert heurisch_wrapper(y, x) == y*x  # 断言 heurisch_wrapper(y, x) 的结果是 y*x


def test_heurisch_fractions():  # 定义测试函数 test_heurisch_fractions
    assert heurisch(1/x, x) == log(x)  # 断言 heurisch(1/x, x) 的结果是 log(x)
    assert heurisch(1/(2 + x), x) == log(x + 2)  # 断言 heurisch(1/(2 + x), x) 的结果是 log(x + 2)
    assert heurisch(1/(x + sin(y)), x) == log(x + sin(y))  # 断言 heurisch(1/(x + sin(y)), x) 的结果是 log(x + sin(y))

    # Up to a constant, where C = pi*I*Rational(5, 12), Mathematica gives identical
    # result in the first case. The difference is because SymPy changes
    # signs of expressions without any care.
    # XXX ^ ^ ^ is this still correct?
    assert heurisch(5*x**5/(
        2*x**6 - 5), x) in [5*log(2*x**6 - 5) / 12, 5*log(-2*x**6 + 5) / 12]  # 断言 heurisch(5*x**5/(2*x**6 - 5), x) 的结果在列表 [5*log(2*x**6 - 5) / 12, 5*log(-2*x**6 + 5) / 12] 中
    assert heurisch(5*x**5/(2*x**6 + 5), x) == 5*log(2*x**6 + 5) / 12  # 断言 heurisch(5*x**5/(2*x**6 + 5), x) 的结果是 5*log(2*x**6 + 5) / 12

    assert heurisch(1/x**2, x) == -1/x  # 断言 heurisch(1/x**2, x) 的结果是 -1/x
    assert heurisch(-1/x**5, x) == 1/(4*x**4)  # 断言 heurisch(-1/x**5, x) 的结果是 1/(4*x**4)
    # 断言：使用 heurisch 函数计算 log(x) 关于 x 的不定积分，应等于 x*log(x) - x
    assert heurisch(log(x), x) == x*log(x) - x
    
    # 断言：使用 heurisch 函数计算 log(3*x) 关于 x 的不定积分，应等于 -x + x*log(3) + x*log(x)
    assert heurisch(log(3*x), x) == -x + x*log(3) + x*log(x)
    
    # 断言：使用 heurisch 函数计算 log(x**2) 关于 x 的不定积分，结果应为下列其中之一：
    # x*log(x**2) - 2*x 或者 2*x*log(x) - 2*x
    assert heurisch(log(x**2), x) in [x*log(x**2) - 2*x, 2*x*log(x) - 2*x]
# 定义一个测试函数，用于测试 heurisch 函数的各种情况
def test_heurisch_exp():
    # 测试 exp(x) 的不定积分，预期结果是 exp(x)
    assert heurisch(exp(x), x) == exp(x)
    # 测试 exp(-x) 的不定积分，预期结果是 -exp(-x)
    assert heurisch(exp(-x), x) == -exp(-x)
    # 测试 exp(17*x) 的不定积分，预期结果是 exp(17*x) / 17
    assert heurisch(exp(17*x), x) == exp(17*x) / 17
    # 测试 x*exp(x) 的不定积分，预期结果是 x*exp(x) - exp(x)
    assert heurisch(x*exp(x), x) == x*exp(x) - exp(x)
    # 测试 x*exp(x**2) 的不定积分，预期结果是 exp(x**2) / 2
    assert heurisch(x*exp(x**2), x) == exp(x**2) / 2

    # 测试 exp(-x**2) 的不定积分，预期结果是 None（表示无法积分）
    assert heurisch(exp(-x**2), x) is None

    # 测试 2**x 的不定积分，预期结果是 2**x/log(2)
    assert heurisch(2**x, x) == 2**x/log(2)
    # 测试 x*2**x 的不定积分，预期结果是 x*2**x/log(2) - 2**x*log(2)**(-2)
    assert heurisch(x*2**x, x) == x*2**x/log(2) - 2**x*log(2)**(-2)

    # 测试包含积分的表达式的不定积分
    assert heurisch(Integral(x**z*y, (y, 1, 2), (z, 2, 3)).function, x) == (x*x**z*y)/(z+1)
    # 测试包含求和的表达式的不定积分
    assert heurisch(Sum(x**z, (z, 1, 2)).function, z) == x**z/log(x)

    # 对于给定链接的问题，验证 exp(z)*exp(-z*sqrt(x - y)) 的不定积分
    anti = -exp(z)/(sqrt(x - y)*exp(z*sqrt(x - y)) - exp(z*sqrt(x - y)))
    assert heurisch(exp(z)*exp(-z*sqrt(x - y)), z) == anti


# 定义测试三角函数的不定积分函数
def test_heurisch_trigonometric():
    # 测试 sin(x) 的不定积分，预期结果是 -cos(x)
    assert heurisch(sin(x), x) == -cos(x)
    # 测试 pi*sin(x) + 1 的不定积分，预期结果是 x - pi*cos(x)
    assert heurisch(pi*sin(x) + 1, x) == x - pi*cos(x)

    # 测试 cos(x) 的不定积分，预期结果是 sin(x)
    assert heurisch(cos(x), x) == sin(x)
    # 测试 tan(x) 的不定积分，预期结果有多个可能性
    assert heurisch(tan(x), x) in [
        log(1 + tan(x)**2)/2,
        log(tan(x) + I) + I*x,
        log(tan(x) - I) - I*x,
    ]

    # 测试 sin(x)*sin(y) 的不定积分关于 x 和 y
    assert heurisch(sin(x)*sin(y), x) == -cos(x)*sin(y)
    assert heurisch(sin(x)*sin(y), y) == -cos(y)*sin(x)

    # 测试 sin(x)*cos(x) 的不定积分，预期结果是 sin(x)**2 / 2 或者 -cos(x)**2 / 2
    assert heurisch(sin(x)*cos(x), x) in [sin(x)**2 / 2, -cos(x)**2 / 2]
    # 测试 cos(x)/sin(x) 的不定积分，预期结果是 log(sin(x))
    assert heurisch(cos(x)/sin(x), x) == log(sin(x))

    # 测试 x*sin(7*x) 的不定积分，预期结果是 sin(7*x) / 49 - x*cos(7*x) / 7
    assert heurisch(x*sin(7*x), x) == sin(7*x) / 49 - x*cos(7*x) / 7
    # 测试 1/pi/4 * x**2*cos(x) 的不定积分
    assert heurisch(1/pi/4 * x**2*cos(x), x) == 1/pi/4*(x**2*sin(x) - 2*sin(x) + 2*x*cos(x))

    # 测试 acos(x/4) * asin(x/4) 的不定积分
    assert heurisch(acos(x/4) * asin(x/4), x) == 2*x - (sqrt(16 - x**2))*asin(x/4) \
        + (sqrt(16 - x**2))*acos(x/4) + x*asin(x/4)*acos(x/4)

    # 测试 sin(x)/(cos(x)**2+1) 的不定积分，预期结果是 -atan(cos(x))
    assert heurisch(sin(x)/(cos(x)**2+1), x) == -atan(cos(x))  # 修复 issue 13723
    # 测试 1/(cos(x)+2) 的不定积分
    assert heurisch(1/(cos(x)+2), x) == 2*sqrt(3)*atan(sqrt(3)*tan(x/2)/3)/3
    # 测试 2*sin(x)*cos(x)/(sin(x)**4 + 1) 的不定积分
    assert heurisch(2*sin(x)*cos(x)/(sin(x)**4 + 1), x) == atan(sqrt(2)*sin(x) - 1) - atan(sqrt(2)*sin(x) + 1)

    # 测试 1/cosh(x) 的不定积分，预期结果是 2*atan(tanh(x/2))
    assert heurisch(1/cosh(x), x) == 2*atan(tanh(x/2))


# 定义测试双曲函数的不定积分函数
def test_heurisch_hyperbolic():
    # 测试 sinh(x) 的不定积分，预期结果是 cosh(x)
    assert heurisch(sinh(x), x) == cosh(x)
    # 测试 cosh(x) 的不定积分，预期结果是 sinh(x)
    assert heurisch(cosh(x), x) == sinh(x)

    # 测试 x*sinh(x) 的不定积分，预期结果是 x*cosh(x) - sinh(x)
    assert heurisch(x*sinh(x), x) == x*cosh(x) - sinh(x)
    # 测试 x*cosh(x) 的不定积分，预期结果是 x*sinh(x) - cosh(x)
    assert heurisch(x*cosh(x), x) == x*sinh(x) - cosh(x)

    # 测试 x*asinh(x/2) 的不定积分
    assert heurisch(x*asinh(x/2), x) == x**2*asinh(x/2)/2 + asinh(x/2) - x*sqrt(4 + x**2)/4


# 定义测试混合类型表达式的不定积分函数
def test_heurisch_mixed():
    # 测试 sin(x)*exp(x) 的不定积分，预期结果是 exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    assert heurisch(sin(x)*exp(x), x) == exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    # 测试 sin(x/sqrt(-x)) 的不定积分
    assert heurisch(sin(x/sqrt(-x)), x) == 2*x*cos(x/sqrt(-x))/sqrt(-x) - 2*sin(x/sqrt(-x))


# 定义测试根式表达式的不定积分函数
def test_heurisch_radicals():
    # 测试 1/sqrt(x) 的不定积分，预期结果是 2*sqrt(x)
    assert heurisch(1/sqrt(x), x) == 2*sqrt(x)
    # 测试 1/sqrt(x)**3 的不定积分，预期结果是 -2/sqrt(x)
    assert heur
    # 断言语句：验证 heurisch_wrapper 函数对 sin(y*sqrt(x)) 在 x 变量上的计算结果是否符合预期
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == Piecewise(
        # 如果 y 不等于 0，则返回以下表达式
        (-2*sqrt(x)*cos(sqrt(x)*y)/y + 2*sin(sqrt(x)*y)/y**2, Ne(y, 0)),
        # 否则返回 0
        (0, True))
    
    # 创建符号变量 y，并指定其为正数
    y = Symbol('y', positive=True)
    
    # 再次断言语句：验证 heurisch_wrapper 函数对 sin(y*sqrt(x)) 在 x 变量上的计算结果是否符合预期
    assert heurisch_wrapper(sin(y*sqrt(x)), x) == 2/y**2*sin(y*sqrt(x)) - \
        2*sqrt(x)*cos(y*sqrt(x))/y
def test_heurisch_special():
    # 测试特定的积分情况，验证 heurisch 函数的正确性
    assert heurisch(erf(x), x) == x*erf(x) + exp(-x**2)/sqrt(pi)
    assert heurisch(exp(-x**2)*erf(x), x) == sqrt(pi)*erf(x)**2 / 4


def test_heurisch_symbolic_coeffs():
    # 测试包含符号系数的积分情况
    assert heurisch(1/(x + y), x) == log(x + y)
    assert heurisch(1/(x + sqrt(2)), x) == log(x + sqrt(2))
    assert simplify(diff(heurisch(log(x + y + z), y), y)) == log(x + y + z)


def test_heurisch_symbolic_coeffs_1130():
    # 测试包含符号系数的积分情况，考虑特殊情况和正数情况
    y = Symbol('y')
    assert heurisch_wrapper(1/(x**2 + y), x) == Piecewise(
        (log(x - sqrt(-y))/(2*sqrt(-y)) - log(x + sqrt(-y))/(2*sqrt(-y)),
         Ne(y, 0)), (-1/x, True))
    y = Symbol('y', positive=True)
    assert heurisch_wrapper(1/(x**2 + y), x) == (atan(x/sqrt(y))/sqrt(y))


def test_heurisch_hacking():
    # 测试多种特定函数的积分计算，使用 hints=[] 参数以提高计算精度
    assert heurisch(sqrt(1 + 7*x**2), x, hints=[]) == \
        x*sqrt(1 + 7*x**2)/2 + sqrt(7)*asinh(sqrt(7)*x)/14
    assert heurisch(sqrt(1 - 7*x**2), x, hints=[]) == \
        x*sqrt(1 - 7*x**2)/2 + sqrt(7)*asin(sqrt(7)*x)/14

    assert heurisch(1/sqrt(1 + 7*x**2), x, hints=[]) == \
        sqrt(7)*asinh(sqrt(7)*x)/7
    assert heurisch(1/sqrt(1 - 7*x**2), x, hints=[]) == \
        sqrt(7)*asin(sqrt(7)*x)/7

    assert heurisch(exp(-7*x**2), x, hints=[]) == \
        sqrt(7*pi)*erf(sqrt(7)*x)/14

    assert heurisch(1/sqrt(9 - 4*x**2), x, hints=[]) == \
        asin(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(9 + 4*x**2), x, hints=[]) == \
        asinh(x*Rational(2, 3))/2

    assert heurisch(1/sqrt(3*x**2-4), x, hints=[]) == \
           sqrt(3)*log(3*x + sqrt(3)*sqrt(3*x**2 - 4))/3


def test_heurisch_function():
    # 测试对未知函数的积分，预期结果为 None
    assert heurisch(f(x), x) is None


@XFAIL
def test_heurisch_function_derivative():
    # 测试未知函数导数积分，这部分原本能正常工作，但由于实现问题暂时不能支持
    # TODO: 探索原因，尝试修复支持该功能

    df = diff(f(x), x)

    assert heurisch(f(x)*df, x) == f(x)**2/2
    assert heurisch(f(x)**2*df, x) == f(x)**3/3
    assert heurisch(df/f(x), x) == log(f(x))


def test_heurisch_wrapper():
    # 测试 heurisch_wrapper 函数的不同输入情况
    f = 1/(y + x)
    assert heurisch_wrapper(f, x) == log(x + y)
    f = 1/(y - x)
    assert heurisch_wrapper(f, x) == -log(x - y)
    f = 1/((y - x)*(y + x))
    assert heurisch_wrapper(f, x) == Piecewise(
        (-log(x - y)/(2*y) + log(x + y)/(2*y), Ne(y, 0)), (1/x, True))
    # issue 6926
    f = sqrt(x**2/((y - x)*(y + x)))
    assert heurisch_wrapper(f, x) == x*sqrt(-x**2/(x**2 - y**2)) \
    - y**2*sqrt(-x**2/(x**2 - y**2))/x


def test_issue_3609():
    # 测试特定问题下的积分计算
    assert heurisch(1/(x * (1 + log(x)**2)), x) == atan(log(x))

### These are examples from the Poor Man's Integrator
### http://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/examples/


def test_pmint_rat():
    # TODO: heurisch() 计算结果与期望值偏差为常数 -3/4，可能需要不同的排列组合来获得最优结果？
    # 定义函数 drop_const，用于简化表达式 expr 中不包含变量 x 的项
    def drop_const(expr, x):
        # 如果 expr 是加法表达式
        if expr.is_Add:
            # 返回一个新的加法表达式，仅包含包含变量 x 的项
            return Add(*[ arg for arg in expr.args if arg.has(x) ])
        else:
            # 如果 expr 不是加法表达式，则直接返回原始表达式
            return expr

    # 定义函数 f 和 g，分别为两个复杂的代数表达式
    f = (x**7 - 24*x**4 - 4*x**2 + 8*x - 8)/(x**8 + 6*x**6 + 12*x**4 + 8*x**2)
    g = (4 + 8*x**2 + 6*x + 3*x**3)/(x**5 + 4*x**3 + 4*x) + log(x)

    # 使用 assert 语句验证一个条件，如果条件为假，抛出异常 AssertionError
    # 调用 heurisch 函数对 f 进行不完全积分，然后对结果使用 ratsimp 函数简化，去除不含变量 x 的项
    # 最后与预期结果 g 进行比较
    assert drop_const(ratsimp(heurisch(f, x)), x) == g
def test_pmint_trig():
    # Define the function f(x) involving trigonometric functions
    f = (x - tan(x)) / tan(x)**2 + tan(x)
    # Define the expected antiderivative g(x)
    g = -x**2/2 - x/tan(x) + log(tan(x)**2 + 1)/2

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g


def test_pmint_logexp():
    # Define the function f(x) involving logarithmic and exponential functions
    f = (1 + x + x*exp(x))*(x + log(x) + exp(x) - 1)/(x + log(x) + exp(x))**2/x
    # Define the expected antiderivative g(x)
    g = log(x + exp(x) + log(x)) + 1/(x + exp(x) + log(x))

    # Assert that the simplified heuristic integration of f(x) equals g(x)
    assert ratsimp(heurisch(f, x)) == g


def test_pmint_erf():
    # Define the function f(x) involving error function and exponential function
    f = exp(-x**2)*erf(x)/(erf(x)**3 - erf(x)**2 - erf(x) + 1)
    # Define the expected antiderivative g(x)
    g = sqrt(pi)*log(erf(x) - 1)/8 - sqrt(pi)*log(erf(x) + 1)/8 - sqrt(pi)/(4*erf(x) - 4)

    # Assert that the simplified heuristic integration of f(x) equals g(x)
    assert ratsimp(heurisch(f, x)) == g


def test_pmint_LambertW():
    # Define the function f(x) as LambertW(x)
    f = LambertW(x)
    # Define the expected antiderivative g(x)
    g = x*LambertW(x) - x + x/LambertW(x)

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g


def test_pmint_besselj():
    # Define the function f(x) involving Bessel functions
    f = besselj(nu + 1, x)/besselj(nu, x)
    # Define the expected antiderivative g(x)
    g = nu*log(x) - log(besselj(nu, x))

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g

    # Define another function f(x) involving Bessel functions
    f = (nu*besselj(nu, x) - x*besselj(nu + 1, x))/x
    # Define the expected antiderivative g(x)
    g = besselj(nu, x)

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g

    # Define another function f(x) involving Bessel functions
    f = jn(nu + 1, x)/jn(nu, x)
    # Define the expected antiderivative g(x)
    g = nu*log(x) - log(jn(nu, x))

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g


@slow
def test_pmint_bessel_products():
    # Define the function f(x) involving products of Bessel functions
    f = x*besselj(nu, x)*bessely(nu, 2*x)
    # Define the expected antiderivative g(x)
    g = -2*x*besselj(nu, x)*bessely(nu - 1, 2*x)/3 + x*besselj(nu - 1, x)*bessely(nu, 2*x)/3

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g

    # Define another function f(x) involving products of Bessel functions
    f = x*besselj(nu, x)*besselk(nu, 2*x)
    # Define the expected antiderivative g(x)
    g = -2*x*besselj(nu, x)*besselk(nu - 1, 2*x)/5 - x*besselj(nu - 1, x)*besselk(nu, 2*x)/5

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g


def test_pmint_WrightOmega():
    # Define the Wright Omega function omega(x)
    def omega(x):
        return LambertW(exp(x))

    # Define the function f(x) involving Wright Omega function
    f = (1 + omega(x) * (2 + cos(omega(x)) * (x + omega(x))))/(1 + omega(x))/(x + omega(x))
    # Define the expected antiderivative g(x)
    g = log(x + LambertW(exp(x))) + sin(LambertW(exp(x)))

    # Assert that the heuristic integration of f(x) equals g(x)
    assert heurisch(f, x) == g


def test_RR():
    # Test for a specific case in the RR ring
    assert heurisch(sqrt(1 + 0.25*x**2), x, hints=[]) == \
        0.5*x*sqrt(0.25*x**2 + 1) + 1.0*asinh(0.5*x)


# TODO: convert the rest of PMINT tests:
# Airy functions
# f = (x - AiryAi(x)*AiryAi(1, x)) / (x**2 - AiryAi(x)**2)
# g = Rational(1,2)*ln(x + AiryAi(x)) + Rational(1,2)*ln(x - AiryAi(x))
# f = x**2 * AiryAi(x)
# g = -AiryAi(x) + AiryAi(1, x)*x
# Whittaker functions
# f = WhittakerW(mu + 1, nu, x) / (WhittakerW(mu, nu, x) * x)
# g = x/2 - mu*ln(x) - ln(WhittakerW(mu, nu, x))


def test_issue_22527():
    t, R = symbols(r't R')
    z = Function('z')(t)
    def f(x):
      return x/sqrt(R**2 - x**2)
    Uz = integrate(f(z), z)
    Ut = integrate(f(t), t)
    # Assert equality between integrals Ut and Uz
    assert Ut == Uz.subs(z, t)


def test_heurisch_complex_erf_issue_26338():
    r = symbols('r', real=True)
    a = exp(-r**2/(2*(2 - I)**2))
    # Assert that heuristic integration of a is None
    assert heurisch(a, r, hints=[]) is None  # None, not a wrong soln
    a = sqrt(pi)*erf((1 + I)/2)/2
    # Assert equality of the definite integral and a
    assert integrate(exp(-I*r**2/2), (r, 0, 1)) == a - I*a

    a = exp(-x**2/(2*(2 - I)**2))
    # Assert that heuristic integration of a is None
    assert heurisch(a, x, hints=[]) is None  # None, not a wrong soln
    a = sqrt(pi)*erf((1 + I)/2)/2
    # Assert equality of the definite integral and a
    assert integrate(exp(-I*x**2/2), (x, 0, 1)) == a - I*a
# 定义一个名为 test_issue_15498 的测试函数
def test_issue_15498():
    # 创建一个名为 Z0 的符号函数
    Z0 = Function('Z0')
    # 定义一些符号变量 k01, k10, t, s，都为实数且正数
    k01, k10, t, s = symbols('k01 k10 t s', real=True, positive=True)
    # 创建一个包含单个元素的 1x1 矩阵，元素为 exp(-k10*t)
    m = Matrix([[exp(-k10*t)]])
    # 定义一个有理数 83/100，也可以用 0.83 表示
    _83 = Rational(83, 100)  # 0.83 works, too
    # 定义一个包含七个元素的列表 [a, b, c, d, e, f, g]
    [a, b, c, d, e, f, g] = [100, 0.5, _83, 50, 0.6, 2, 120]
    # 定义血管输入函数的两个部分 AIF_btf 和 AIF_atf
    AIF_btf = a * (d * e * (1 - exp(-(t - b) / e)) + f * g * (1 - exp(-(t - b) / g)))
    AIF_atf = a * (d * e * exp(-(t - b) / e) * (exp((c - b) / e) - 1) + f * g * exp(-(t - b) / g) * (exp((c - b) / g) - 1))
    # 定义血管输入函数 AIF_sym，使用 Piecewise 函数根据不同的时间段返回不同的值
    AIF_sym = Piecewise((0, t < b), (AIF_btf, And(b <= t, t < c)), (AIF_atf, c <= t))
    # 定义一个方程 aif_eq，表示 Z0(t) 等于 AIF_sym
    aif_eq = Eq(Z0(t), AIF_sym)
    # 创建一个包含单个元素的 1x1 矩阵，元素为 k01*Z0(t)
    f_vec = Matrix([[k01 * Z0(t)]])
    # 定义积分被积函数 integrand，包括 m*m.subs(t, s)**-1*f_vec.subs(aif_eq.lhs, aif_eq.rhs).subs(t, s)
    integrand = m * m.subs(t, s)**-1 * f_vec.subs(aif_eq.lhs, aif_eq.rhs).subs(t, s)
    # 对被积函数 integrand[0] 进行从 s=0 到 s=t 的积分，求解结果赋给 solution
    solution = integrate(integrand[0], (s, 0, t))
    # 断言确保 solution 不为空，且执行时间不超过 10 秒
    assert solution is not None  # does not hang and takes less than 10 s
```