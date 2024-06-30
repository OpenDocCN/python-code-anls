# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_manual.py`

```
# 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.expr import Expr
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.function 模块导入 Derivative, Function, diff, expand 函数
from sympy.core.function import (Derivative, Function, diff, expand)
# 从 sympy.core.numbers 模块导入 I, Rational, pi 常数
from sympy.core.numbers import (I, Rational, pi)
# 从 sympy.core.relational 模块导入 Ne 类
from sympy.core.relational import Ne
# 从 sympy.core.singleton 模块导入 S 单例
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Dummy, Symbol, symbols 函数
from sympy.core.symbol import (Dummy, Symbol, symbols)
# 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
from sympy.functions.elementary.exponential import (exp, log)
# 从 sympy.functions.elementary.hyperbolic 模块导入 asinh, csch, cosh, coth, sech, sinh, tanh 函数
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise, piecewise_fold 函数
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
# 从 sympy.functions.elementary.trigonometric 模块导入 acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan 函数
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
# 从 sympy.functions.special.delta_functions 模块导入 Heaviside, DiracDelta 函数
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
# 从 sympy.functions.special.elliptic_integrals 模块导入 elliptic_e, elliptic_f 函数
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
# 从 sympy.functions.special.error_functions 模块导入 Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li 函数
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
# 从 sympy.functions.special.gamma_functions 模块导入 uppergamma 函数
from sympy.functions.special.gamma_functions import uppergamma
# 从 sympy.functions.special.polynomials 模块导入 assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre 函数
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
# 从 sympy.functions.special.zeta_functions 模块导入 polylog 函数
from sympy.functions.special.zeta_functions import polylog
# 从 sympy.integrals.integrals 模块导入 Integral, integrate 函数
from sympy.integrals.integrals import (Integral, integrate)
# 从 sympy.logic.boolalg 模块导入 And 类
from sympy.logic.boolalg import And
# 从 sympy.integrals.manualintegrate 模块导入 manualintegrate, find_substitutions,
# _parts_rule, integral_steps, manual_subs 函数
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
    _parts_rule, integral_steps, manual_subs)
# 从 sympy.testing.pytest 模块导入 raises, slow 函数
from sympy.testing.pytest import raises, slow

# 创建符号变量 x, y, z, u, n, a, b, c, d, e
x, y, z, u, n, a, b, c, d, e = symbols('x y z u n a b c d e')
# 创建函数符号 f
f = Function('f')

# 定义函数 assert_is_integral_of，用于验证 f 是否是 F 的积分，并且 F 是 f 的导数
def assert_is_integral_of(f: Expr, F: Expr):
    assert manualintegrate(f, x) == F
    assert F.diff(x).equals(f)

# 定义测试函数 test_find_substitutions，用于测试 find_substitutions 函数的行为
def test_find_substitutions():
    assert find_substitutions((cot(x)**2 + 1)**2*csc(x)**2*cot(x)**2, x, u) == \
        [(cot(x), 1, -u**6 - 2*u**4 - u**2)]
    assert find_substitutions((sec(x)**2 + tan(x) * sec(x)) / (sec(x) + tan(x)),
                              x, u) == [(sec(x) + tan(x), 1, 1/u)]
    assert (-x**2, Rational(-1, 2), exp(u)) in find_substitutions(x * exp(-x**2), x, u)
    assert not find_substitutions(Derivative(f(x), x)**2, x, u)

# 定义测试函数 test_manualintegrate_polynomials，用于测试 manualintegrate 函数在多项式上的行为
def test_manualintegrate_polynomials():
    assert manualintegrate(y, x) == x*y
    assert manualintegrate(exp(2), x) == x * exp(2)
    assert manualintegrate(x**2, x) == x**3 / 3
    assert manualintegrate(3 * x**2 + 4 * x**3, x) == x**3 + x**4

    assert manualintegrate((x + 2)**3, x) == (x + 2)**4 / 4
    assert manualintegrate((3*x + 4)**2, x) == (3*x + 4)**3 / 9

    assert manualintegrate((u + 2)**3, u) == (u + 2)**4 / 4
    assert manualintegrate((3*u + 4)**2, u) == (3*u + 4)**3 / 9

# 定义测试函数 test_manualintegrate_exponentials，用于测试 manualintegrate 函数在指数函数上的行为
def test_manualintegrate_exponentials():
    assert manualintegrate(exp(2*x), x) == exp(2*x) / 2
    assert manualintegrate(2**x, x) == (2 ** x) / log(2)
    assert_is_integral_of(1/sqrt(1-exp(2*x)),
                          log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2)
    # 断言：使用 manualintegrate 函数计算 1/x 的不定积分是否等于 log(x)
    assert manualintegrate(1 / x, x) == log(x)
    
    # 断言：使用 manualintegrate 函数计算 1 / (2*x + 3) 的不定积分是否等于 log(2*x + 3) / 2
    assert manualintegrate(1 / (2*x + 3), x) == log(2*x + 3) / 2
    
    # 断言：使用 manualintegrate 函数计算 log(x)**2 / x 的不定积分是否等于 log(x)**3 / 3
    assert manualintegrate(log(x)**2 / x, x) == log(x)**3 / 3
    
    # 断言：验证 x**x * (log(x) + 1) 是否是 x**x 的一个原函数
    assert_is_integral_of(x**x*(log(x)+1), x**x)
# 定义函数 test_manualintegrate_parts，用于测试 manualintegrate 函数的部分导数求解功能
def test_manualintegrate_parts():
    # 断言：对 exp(x) * sin(x) 进行手动积分应得到 (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert manualintegrate(exp(x) * sin(x), x) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    # 断言：对 2*x*cos(x) 进行手动积分应得到 2*x*sin(x) + 2*cos(x)
    assert manualintegrate(2*x*cos(x), x) == 2*x*sin(x) + 2*cos(x)
    # 断言：对 x * log(x) 进行手动积分应得到 x**2*log(x)/2 - x**2/4
    assert manualintegrate(x * log(x), x) == x**2*log(x)/2 - x**2/4
    # 断言：对 log(x) 进行手动积分应得到 x * log(x) - x
    assert manualintegrate(log(x), x) == x * log(x) - x
    # 断言：对 (3*x**2 + 5) * exp(x) 进行手动积分应得到 3*x**2*exp(x) - 6*x*exp(x) + 11*exp(x)
    assert manualintegrate((3*x**2 + 5) * exp(x), x) == \
        3*x**2*exp(x) - 6*x*exp(x) + 11*exp(x)
    # 断言：对 atan(x) 进行手动积分应得到 x*atan(x) - log(x**2 + 1)/2
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2

    # 确保 _parts_rule 函数不会选择 u = 常数，但必要时可以选择 dv = 常数，例如对 integrate(atan(x))
    # 断言：对 cos(x) 使用 _parts_rule 应返回 None
    assert _parts_rule(cos(x), x) == None
    # 断言：对 exp(x) 使用 _parts_rule 应返回 None
    assert _parts_rule(exp(x), x) == None
    # 断言：对 x**2 使用 _parts_rule 应返回 None
    assert _parts_rule(x**2, x) == None
    # 对 atan(x) 使用 _parts_rule 应返回 (atan(x), 1)
    result = _parts_rule(atan(x), x)
    assert result[0] == atan(x) and result[1] == 1


# 定义函数 test_manualintegrate_trigonometry，用于测试 manualintegrate 函数对三角函数的积分功能
def test_manualintegrate_trigonometry():
    # 断言：对 sin(x) 进行手动积分应得到 -cos(x)
    assert manualintegrate(sin(x), x) == -cos(x)
    # 断言：对 tan(x) 进行手动积分应得到 -log(cos(x))
    assert manualintegrate(tan(x), x) == -log(cos(x))

    # 断言：对 sec(x) 进行手动积分应得到 log(sec(x) + tan(x))
    assert manualintegrate(sec(x), x) == log(sec(x) + tan(x))
    # 断言：对 csc(x) 进行手动积分应得到 -log(csc(x) + cot(x))
    assert manualintegrate(csc(x), x) == -log(csc(x) + cot(x))

    # 断言：对 sin(x) * cos(x) 进行手动积分应得到 sin(x) ** 2 / 2 或 -cos(x)**2 / 2 中的一种
    assert manualintegrate(sin(x) * cos(x), x) in [sin(x) ** 2 / 2, -cos(x)**2 / 2]
    # 断言：对 -sec(x) * tan(x) 进行手动积分应得到 -sec(x)
    assert manualintegrate(-sec(x) * tan(x), x) == -sec(x)
    # 断言：对 csc(x) * cot(x) 进行手动积分应得到 -csc(x)
    assert manualintegrate(csc(x) * cot(x), x) == -csc(x)
    # 断言：对 sec(x)**2 进行手动积分应得到 tan(x)
    assert manualintegrate(sec(x)**2, x) == tan(x)
    # 断言：对 csc(x)**2 进行手动积分应得到 -cot(x)
    assert manualintegrate(csc(x)**2, x) == -cot(x)

    # 断言：对 x * sec(x**2) 进行手动积分应得到 log(tan(x**2) + sec(x**2))/2
    assert manualintegrate(x * sec(x**2), x) == log(tan(x**2) + sec(x**2))/2
    # 断言：对 cos(x)*csc(sin(x)) 进行手动积分应得到 -log(cot(sin(x)) + csc(sin(x)))
    assert manualintegrate(cos(x)*csc(sin(x)), x) == -log(cot(sin(x)) + csc(sin(x)))
    # 断言：对 cos(3*x)*sec(x) 进行手动积分应得到 -x + sin(2*x)
    assert manualintegrate(cos(3*x)*sec(x), x) == -x + sin(2*x)
    # 断言：对 sin(3*x)*sec(x) 进行手动积分应得到 -3*log(cos(x)) + 2*log(cos(x)**2) - 2*cos(x)**2
    assert manualintegrate(sin(3*x)*sec(x), x) == \
        -3*log(cos(x)) + 2*log(cos(x)**2) - 2*cos(x)**2

    # 断言：验证 sinh(2*x) 的手动积分结果为 cosh(2*x)/2
    assert_is_integral_of(sinh(2*x), cosh(2*x)/2)
    # 断言：验证 x*cosh(x**2) 的手动积分结果为 sinh(x**2)/2
    assert_is_integral_of(x*cosh(x**2), sinh(x**2)/2)
    # 断言：验证 tanh(x) 的手动积分结果为 log(cosh(x))
    assert_is_integral_of(tanh(x), log(cosh(x)))
    # 断言：验证 coth(x) 的手动积分结果为 log(sinh(x))
    assert_is_integral_of(coth(x), log(sinh(x)))
    # 断言：验证 sech(x) 的手动积分结果为 2*atan(tanh(x/2))
    f, F = sech(x), 2*atan(tanh(x/2))
    assert manualintegrate(f, x) == F
    # 断言：验证 (F.diff(x) - f).rewrite(exp).simplify() == 0
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0  # todo: equals returns None
    # 断言：验证 csch(x) 的手动积分结果为 log(tanh(x/2))
    f, F = csch(x), log(tanh(x/2))
    assert manualintegrate(f, x) == F
    # 断言：验证 (F.diff(x) - f).rewrite(exp).simplify() == 0
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0


# 定义函数 test_manualintegrate_trigpowers，用于测试 manualintegrate 函数对三角函数的幂次积分功能
@slow
def test_manualintegrate_trigpowers():
    # 断言：对 sin(x)**2 * cos(x) 进行手动积分应得到 sin(x)**3 / 3
    assert manualintegrate(sin(x)**2 * cos(x), x) == sin(x)**3 / 3
    # 断言：对 sin(x)**2 * cos(x)**2 进行手动积分应得到 x / 8 - sin(4*x) / 32
    assert manualintegrate(sin(x)**2 * cos(x)**2, x) == \
        x / 8 - sin(4*x) / 32
    # 断言：对 sin(x) * cos(x)**3 进行手动积分应得到 -cos(x)**4 / 4
    assert manualintegrate(sin(x) * cos(x)**
def test_manualintegrate_inversetrig():
    # 测试手动积分函数对反三角函数的积分计算

    # atan函数的测试用例
    assert manualintegrate(exp(x) / (1 + exp(2*x)), x) == atan(exp(x))
    assert manualintegrate(1 / (4 + 9 * x**2), x) == atan(3 * x/2) / 6
    assert manualintegrate(1 / (16 + 16 * x**2), x) == atan(x) / 16
    assert manualintegrate(1 / (4 + x**2), x) == atan(x / 2) / 2
    assert manualintegrate(1 / (1 + 4 * x**2), x) == atan(2*x) / 2

    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)

    # Piecewise条件分段函数的测试
    assert manualintegrate(1/(ra + rb*x**2), x) == \
        Piecewise((atan(x/sqrt(ra/rb))/(rb*sqrt(ra/rb)), ra/rb > 0),
                  ((log(x - sqrt(-ra/rb)) - log(x + sqrt(-ra/rb)))/(2*sqrt(rb)*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + rb*x**2), x) == \
        Piecewise((atan(x/(2*sqrt(1/rb)))/(2*rb*sqrt(1/rb)), 1/rb > 0),
                  (-I*(log(x - 2*sqrt(-1/rb)) - log(x + 2*sqrt(-1/rb)))/(4*sqrt(rb)), True))
    assert manualintegrate(1/(ra + 4*x**2), x) == \
        Piecewise((atan(2*x/sqrt(ra))/(2*sqrt(ra)), ra > 0),
                  ((log(x - sqrt(-ra)/2) - log(x + sqrt(-ra)/2))/(4*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + 4*x**2), x) == atan(x) / 4

    # 通用形式下的Piecewise条件测试
    assert manualintegrate(1/(a + b*x**2), x) == Piecewise((atan(x/sqrt(a/b))/(b*sqrt(a/b)), Ne(a, 0)),
                                                           (-1/(b*x), True))

    # asin函数的测试用例
    assert manualintegrate(1/sqrt(1-x**2), x) == asin(x)
    assert manualintegrate(1/sqrt(4-4*x**2), x) == asin(x)/2
    assert manualintegrate(3/sqrt(1-9*x**2), x) == asin(3*x)
    assert manualintegrate(1/sqrt(4-9*x**2), x) == asin(x*Rational(3, 2))/3

    # asinh函数的测试用例
    assert manualintegrate(1/sqrt(x**2 + 1), x) == \
        asinh(x)
    assert manualintegrate(1/sqrt(x**2 + 4), x) == \
        asinh(x/2)
    assert manualintegrate(1/sqrt(4*x**2 + 4), x) == \
        asinh(x)/2
    assert manualintegrate(1/sqrt(4*x**2 + 1), x) == \
        asinh(2*x)/2
    assert manualintegrate(1/sqrt(ra*x**2 + 1), x) == \
        Piecewise((asin(x*sqrt(-ra))/sqrt(-ra), ra < 0), (asinh(sqrt(ra)*x)/sqrt(ra), ra > 0), (x, True))
    assert manualintegrate(1/sqrt(ra + x**2), x) == \
        Piecewise((asinh(x*sqrt(1/ra)), ra > 0), (log(2*x + 2*sqrt(ra + x**2)), True))

    # log函数的测试用例
    assert manualintegrate(1/sqrt(x**2 - 1), x) == log(2*x + 2*sqrt(x**2 - 1))
    assert manualintegrate(1/sqrt(x**2 - 4), x) == log(2*x + 2*sqrt(x**2 - 4))
    assert manualintegrate(1/sqrt(4*x**2 - 4), x) == log(8*x + 4*sqrt(4*x**2 - 4))/2
    assert manualintegrate(1/sqrt(9*x**2 - 1), x) == log(18*x + 6*sqrt(9*x**2 - 1))/3
    assert manualintegrate(1/sqrt(ra*x**2 - 4), x) == \
           Piecewise((log(2*sqrt(ra)*sqrt(ra*x**2 - 4) + 2*ra*x)/sqrt(ra), Ne(ra, 0)), (-I*x/2, True))
    assert manualintegrate(1/sqrt(-ra + 4*x**2), x) == \
        Piecewise((asinh(2*x*sqrt(-1/ra))/2, ra < 0), (log(8*x + 4*sqrt(-ra + 4*x**2))/2, True))
    # 验证对 asin(x) 的手动积分是否正确
    assert manualintegrate(asin(x), x) == x*asin(x) + sqrt(1 - x**2)
    # 验证对 asin(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(asin(a*x), x) == Piecewise(((a*x*asin(a*x) + sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (0, True))
    # 验证对 x*asin(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(x*asin(a*x), x) == \
           -a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                         log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                        (x**3/3, True))/2 + x**2*asin(a*x)/2
    # 验证对 acos(x) 的手动积分是否正确
    assert manualintegrate(acos(x), x) == x*acos(x) - sqrt(1 - x**2)
    # 验证对 acos(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(acos(a*x), x) == Piecewise(((a*x*acos(a*x) - sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (pi*x/2, True))
    # 验证对 x*acos(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(x*acos(a*x), x) == \
           a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                        log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                       (x**3/3, True))/2 + x**2*acos(a*x)/2
    # 验证对 atan(x) 的手动积分是否正确
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2
    # 验证对 atan(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(atan(a*x), x) == Piecewise(((a*x*atan(a*x) - log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (0, True))
    # 验证对 x*atan(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(x*atan(a*x), x) == -a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*atan(a*x)/2
    # 验证对 acsc(x) 的手动积分是否正确
    assert manualintegrate(acsc(x), x) == x*acsc(x) + Integral(1/(x*sqrt(1 - 1/x**2)), x)
    # 验证对 acsc(a*x) 的手动积分是否正确，使用 Integral 表达
    assert manualintegrate(acsc(a*x), x) == x*acsc(a*x) + Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    # 验证对 x*acsc(a*x) 的手动积分是否正确，使用 Integral 表达
    assert manualintegrate(x*acsc(a*x), x) == x**2*acsc(a*x)/2 + Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # 验证对 asec(x) 的手动积分是否正确
    assert manualintegrate(asec(x), x) == x*asec(x) - Integral(1/(x*sqrt(1 - 1/x**2)), x)
    # 验证对 asec(a*x) 的手动积分是否正确，使用 Integral 表达
    assert manualintegrate(asec(a*x), x) == x*asec(a*x) - Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    # 验证对 x*asec(a*x) 的手动积分是否正确，使用 Integral 表达
    assert manualintegrate(x*asec(a*x), x) == x**2*asec(a*x)/2 - Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # 验证对 acot(x) 的手动积分是否正确
    assert manualintegrate(acot(x), x) == x*acot(x) + log(x**2 + 1)/2
    # 验证对 acot(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(acot(a*x), x) == Piecewise(((a*x*acot(a*x) + log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (pi*x/2, True))
    # 验证对 x*acot(a*x) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(x*acot(a*x), x) == a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*acot(a*x)/2

    # 验证对 1/sqrt(ra-rb*x**2) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(1/sqrt(ra-rb*x**2), x) == \
        Piecewise((asin(x*sqrt(rb/ra))/sqrt(rb), And(-rb < 0, ra > 0)),
                  (asinh(x*sqrt(-rb/ra))/sqrt(-rb), And(-rb > 0, ra > 0)),
                  (log(-2*rb*x + 2*sqrt(-rb)*sqrt(ra - rb*x**2))/sqrt(-rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))
    # 验证对 1/sqrt(ra+rb*x**2) 的手动积分是否正确，使用 Piecewise 表达
    assert manualintegrate(1/sqrt(ra+rb*x**2), x) == \
        Piecewise((asin(x*sqrt(-rb/ra))/sqrt(-rb), And(ra > 0, rb < 0)),
                  (asinh(x*sqrt(rb/ra))/sqrt(rb), And(ra > 0, rb > 0)),
                  (log(2*sqrt(rb)*sqrt(ra + rb*x**2) + 2*rb*x)/sqrt(rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))
# 定义测试函数 test_manualintegrate_trig_substitution，用于测试 manualintegrate 函数的三角代换情况
def test_manualintegrate_trig_substitution():
    # 断言语句，验证对于 sqrt(16*x**2 - 9)/x 的积分结果
    assert manualintegrate(sqrt(16*x**2 - 9)/x, x) == \
        Piecewise((sqrt(16*x**2 - 9) - 3*acos(3/(4*x)),
                   And(x < Rational(3, 4), x > Rational(-3, 4))))
    
    # 断言语句，验证对于 1/(x**4 * sqrt(25-x**2)) 的积分结果
    assert manualintegrate(1/(x**4 * sqrt(25-x**2)), x) == \
        Piecewise((-sqrt(-x**2/25 + 1)/(125*x) -
                   (-x**2/25 + 1)**(3*S.Half)/(15*x**3), And(x < 5, x > -5)))
    
    # 断言语句，验证对于 x**7/(49*x**2 + 1)**(3 * S.Half) 的积分结果
    assert manualintegrate(x**7/(49*x**2 + 1)**(3 * S.Half), x) == \
        ((49*x**2 + 1)**(5*S.Half)/28824005 -
         (49*x**2 + 1)**(3*S.Half)/5764801 +
         3*sqrt(49*x**2 + 1)/5764801 + 1/(5764801*sqrt(49*x**2 + 1)))

# 定义测试函数 test_manualintegrate_trivial_substitution，用于测试 manualintegrate 函数的平凡代换情况
def test_manualintegrate_trivial_substitution():
    # 断言语句，验证对于 (exp(x) - exp(-x))/x 的积分结果
    assert manualintegrate((exp(x) - exp(-x))/x, x) == -Ei(-x) + Ei(x)
    
    # 创建函数 f(x)
    f = Function('f')
    # 断言语句，验证对于 (f(x) - f(-x))/x 的积分结果
    assert manualintegrate((f(x) - f(-x))/x, x) == \
        -Integral(f(-x)/x, x) + Integral(f(x)/x, x)

# 定义测试函数 test_manualintegrate_rational，用于测试 manualintegrate 函数的有理函数情况
def test_manualintegrate_rational():
    # 断言语句，验证对于 1/(4 - x**2) 的积分结果
    assert manualintegrate(1/(4 - x**2), x) == -log(x - 2)/4 + log(x + 2)/4
    
    # 断言语句，验证对于 1/(-1 + x**2) 的积分结果
    assert manualintegrate(1/(-1 + x**2), x) == log(x - 1)/2 - log(x + 1)/2

# 定义测试函数 test_manualintegrate_special，用于测试 manualintegrate 函数的特殊函数情况
def test_manualintegrate_special():
    # 验证特殊函数 4*exp(-x**2/3) 的积分结果为 2*sqrt(3)*sqrt(pi)*erf(sqrt(3)*x/3)
    f, F = 4*exp(-x**2/3), 2*sqrt(3)*sqrt(pi)*erf(sqrt(3)*x/3)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 3*exp(4*x**2) 的积分结果为 3*sqrt(pi)*erfi(2*x)/4
    f, F = 3*exp(4*x**2), 3*sqrt(pi)*erfi(2*x)/4
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 x**Rational(1, 3)*exp(-x/8) 的积分结果为 -16*uppergamma(Rational(4, 3), x/8)
    f, F = x**Rational(1, 3)*exp(-x/8), -16*uppergamma(Rational(4, 3), x/8)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 exp(2*x)/x 的积分结果为 Ei(2*x)
    f, F = exp(2*x)/x, Ei(2*x)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 exp(1 + 2*x - x**2) 的积分结果为 sqrt(pi)*exp(2)*erf(x - 1)/2
    f, F = exp(1 + 2*x - x**2), sqrt(pi)*exp(2)*erf(x - 1)/2
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 sin(x**2 + 4*x + 1) 的积分结果
    f = sin(x**2 + 4*x + 1)
    F = (sqrt(2)*sqrt(pi)*(-sin(3)*fresnelc(sqrt(2)*(2*x + 4)/(2*sqrt(pi))) +
        cos(3)*fresnels(sqrt(2)*(2*x + 4)/(2*sqrt(pi))))/2)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 cos(4*x**2) 的积分结果为 sqrt(2)*sqrt(pi)*fresnelc(2*sqrt(2)*x/sqrt(pi))/4
    f, F = cos(4*x**2), sqrt(2)*sqrt(pi)*fresnelc(2*sqrt(2)*x/sqrt(pi))/4
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 sin(3*x + 2)/x 的积分结果为 sin(2)*Ci(3*x) + cos(2)*Si(3*x)
    f, F = sin(3*x + 2)/x, sin(2)*Ci(3*x) + cos(2)*Si(3*x)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 sinh(3*x - 2)/x 的积分结果为 -sinh(2)*Chi(3*x) + cosh(2)*Shi(3*x)
    f, F = sinh(3*x - 2)/x, -sinh(2)*Chi(3*x) + cosh(2)*Shi(3*x)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 5*cos(2*x - 3)/x 的积分结果为 5*cos(3)*Ci(2*x) + 5*sin(3)*Si(2*x)
    f, F = 5*cos(2*x - 3)/x, 5*cos(3)*Ci(2*x) + 5*sin(3)*Si(2*x)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 cosh(x/2)/x 的积分结果为 Chi(x/2)
    f, F = cosh(x/2)/x, Chi(x/2)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 cos(x**2)/x 的积分结果为 Ci(x**2)/2
    f, F = cos(x**2)/x, Ci(x**2)/2
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 1/log(2*x + 1) 的积分结果为 li(2*x + 1)/2
    f, F = 1/log(2*x + 1), li(2*x + 1)/2
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 polylog(2, 5*x)/x 的积分结果为 polylog(3, 5*x)
    f, F = polylog(2, 5*x)/x, polylog(3, 5*x)
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 5/sqrt(3 - 2*sin(x)**2) 的积分结果为 5*sqrt(3)*elliptic_f(x, Rational(2, 3))/3
    f, F = 5/sqrt(3 - 2*sin(x)**2), 5*sqrt(3)*elliptic_f(x, Rational(2, 3))/3
    assert_is_integral_of(f, F)
    
    # 验证特殊函数 sqrt(4 + 9*sin(x)**2) 的积分结果为 2*ell
# 定义测试函数 `test_manualintegrate_Heaviside`，用于测试手动积分函数对 Heaviside 和 DiracDelta 的处理
def test_manualintegrate_Heaviside():
    # 断言：DiracDelta(3*x+2) 的积分应为 Heaviside(3*x+2)/3
    assert_is_integral_of(DiracDelta(3*x+2), Heaviside(3*x+2)/3)
    # 断言：DiracDelta(3*x, 0) 的积分应为 Heaviside(3*x)/3
    assert_is_integral_of(DiracDelta(3*x, 0), Heaviside(3*x)/3)
    # 断言：DiracDelta(a+b*x, 1) 的积分应为 Piecewise((DiracDelta(a + b*x)/b, Ne(b, 0)), (x*DiracDelta(a, 1), True))
    assert manualintegrate(DiracDelta(a+b*x, 1), x) == \
        Piecewise((DiracDelta(a + b*x)/b, Ne(b, 0)), (x*DiracDelta(a, 1), True))
    # 断言：DiracDelta(x/3-1, 2) 的积分应为 3*DiracDelta(x/3-1, 1)
    assert_is_integral_of(DiracDelta(x/3-1, 2), 3*DiracDelta(x/3-1, 1))
    # 断言：Heaviside(x) 的积分应为 x*Heaviside(x)
    assert manualintegrate(Heaviside(x), x) == x*Heaviside(x)
    # 断言：x*Heaviside(2) 的积分应为 x**2/2
    assert manualintegrate(x*Heaviside(2), x) == x**2/2
    # 断言：x*Heaviside(-2) 的积分应为 0
    assert manualintegrate(x*Heaviside(-2), x) == 0
    # 断言：x*Heaviside(x) 的积分应为 x**2*Heaviside(x)/2
    assert manualintegrate(x*Heaviside(x), x) == x**2*Heaviside(x)/2
    # 断言：x*Heaviside(-x) 的积分应为 x**2*Heaviside(-x)/2
    assert manualintegrate(x*Heaviside(-x), x) == x**2*Heaviside(-x)/2
    # 断言：Heaviside(2*x + 4) 的积分应为 (x+2)*Heaviside(2*x + 4)
    assert manualintegrate(Heaviside(2*x + 4), x) == (x+2)*Heaviside(2*x + 4)
    # 断言：x*Heaviside(x) 的积分应为 x**2*Heaviside(x)/2
    assert manualintegrate(x*Heaviside(x), x) == x**2*Heaviside(x)/2
    # 断言：Heaviside(x + 1)*Heaviside(1 - x)*x**2 的积分应为 ((x**3/3 + Rational(1, 3))*Heaviside(x + 1) - Rational(2, 3))*Heaviside(-x + 1)
    assert manualintegrate(Heaviside(x + 1)*Heaviside(1 - x)*x**2, x) == \
        ((x**3/3 + Rational(1, 3))*Heaviside(x + 1) - Rational(2, 3))*Heaviside(-x + 1)

    # 创建符号变量 y
    y = Symbol('y')
    # 断言：sin(7 + x)*Heaviside(3*x - 7) 的积分应为 (- cos(x + 7) + cos(Rational(28, 3)))*Heaviside(3*x - S(7))
    assert manualintegrate(sin(7 + x)*Heaviside(3*x - 7), x) == \
            (- cos(x + 7) + cos(Rational(28, 3)))*Heaviside(3*x - S(7))
    # 断言：sin(y + x)*Heaviside(3*x - y) 的积分应为 (cos(y*Rational(4, 3)) - cos(x + y))*Heaviside(3*x - y)
    assert manualintegrate(sin(y + x)*Heaviside(3*x - y), x) == \
            (cos(y*Rational(4, 3)) - cos(x + y))*Heaviside(3*x - y)


# 定义测试函数 `test_manualintegrate_orthogonal_poly`，用于测试手动积分函数对多项式的处理
def test_manualintegrate_orthogonal_poly():
    # 创建符号变量 n
    n = symbols('n')
    # 设定常量 a, b
    a, b = 7, Rational(5, 3)
    # 定义多项式列表
    polys = [jacobi(n, a, b, x), gegenbauer(n, a, x), chebyshevt(n, x),
        chebyshevu(n, x), legendre(n, x), hermite(n, x), laguerre(n, x),
        assoc_laguerre(n, a, x)]
    # 遍历多项式列表
    for p in polys:
        # 计算多项式 p 对 x 的积分
        integral = manualintegrate(p, x)
        # 对于给定的阶数进行断言，确保积分的导数与原多项式的值相等
        for deg in [-2, -1, 0, 1, 3, 5, 8]:
            try:
                # 尝试用给定的阶数替换多项式中的 n
                p_subbed = p.subs(n, deg)
            except ValueError:
                continue
            # 断言：积分的导数与原多项式的值相等
            assert (integral.subs(n, deg).diff(x) - p_subbed).expand() == 0

        # 用这些多项式还可以积分简单的表达式
        q = x*p.subs(x, 2*x + 1)
        integral = manualintegrate(q, x)
        for deg in [2, 4, 7]:
            # 断言：积分的导数与原表达式的值相等
            assert (integral.subs(n, deg).diff(x) - q.subs(n, deg)).expand() == 0

        # 对于多项式的其他参数，不能进行积分
        t = symbols('t')
        for i in range(len(p.args) - 1):
            new_args = list(p.args)
            new_args[i] = t
            # 断言：使用其他参数进行积分会抛出 Integral 错误
            assert isinstance(manualintegrate(p.func(*new_args), t), Integral)


# 定义测试函数 `test_issue_6799`，用于测试特定问题的处理
@slow
def test_issue_6799():
    # 创建符号变量 r, x, phi
    r, x, phi = map(Symbol, 'r x phi'.split())
    # 创建整数且为正的符号变量 n
    n = Symbol('n', integer=True, positive=True)

    # 定义被积函数
    integrand = (cos(n*(x-phi))*cos(n*x))
    # 定义积分上下限
    limits = (x, -pi, pi)
    # 断言：积分结果应为 ((n*x/2 + sin(2*n*x)/4)*cos(n*phi) - sin(n*phi)*cos(n*x)**2/2)/n
    assert manualintegrate(integrand, x) == \
        ((n*x/2 + sin(2*n*x)/4)*cos(n*phi) - sin(n*phi)*cos(n*x)**2/2)/n
    # 断言：被积函数积分后与 integrate 函数的结果相等（已简化）
    assert r * integrate(integrand, limits).trigsimp() / pi == r * cos(n * phi)
    # 断言：被积函数积分后结果不含虚数单位 i
    assert not integrate(integrand, limits).has(Dummy)


# 定义测试函数 `test_issue_12251`，用于测试特定问题的处理
def test_issue_12251():
    # 断言语句，用于测试 manualintegrate 函数对于给定参数的返回值是否符合预期
    assert manualintegrate(x**y, x) == Piecewise(
        # 当 y 不等于 -1 时，返回 x^(y + 1) / (y + 1)
        (x**(y + 1)/(y + 1), Ne(y, -1)),
        # 当 y 等于 -1 时，返回 log(x)
        (log(x), True))
# 定义用于测试 issue 3796 的函数
def test_issue_3796():
    # 断言手动积分 diff(exp(x + x**2)) 后得到 exp(x + x**2)
    assert manualintegrate(diff(exp(x + x**2)), x) == exp(x + x**2)
    # 断言积分 x * exp(x**4) 后得到 -I*sqrt(pi)*erf(I*x**2)/4，使用 risch=False 禁用 Risch 算法
    assert integrate(x * exp(x**4), x, risch=False) == -I*sqrt(pi)*erf(I*x**2)/4


# 定义用于测试手动积分 manual=True 的函数
def test_manual_true():
    # 断言手动积分 exp(x) * sin(x) 后得到 (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert integrate(exp(x) * sin(x), x, manual=True) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    # 断言手动积分 sin(x) * cos(x) 后结果在 [sin(x)**2 / 2, -cos(x)**2 / 2] 中
    assert integrate(sin(x) * cos(x), x, manual=True) in \
        [sin(x)**2 / 2, -cos(x)**2 / 2]


# 定义用于测试 issue 6746 的函数
def test_issue_6746():
    y = Symbol('y')
    n = Symbol('n')
    # 断言手动积分 y**x 后得到 Piecewise((y**x/log(y), Ne(log(y), 0)), (x, True))
    assert manualintegrate(y**x, x) == Piecewise(
        (y**x/log(y), Ne(log(y), 0)), (x, True))
    # 断言手动积分 y**(n*x) 后得到 Piecewise((Piecewise((y**(n*x)/log(y), Ne(log(y), 0)), (n*x, True))/n, Ne(n, 0)), (x, True))
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (Piecewise(
            (y**(n*x)/log(y), Ne(log(y), 0)),
            (n*x, True)
        )/n, Ne(n, 0)),
        (x, True))
    # 断言手动积分 exp(n*x) 后得到 Piecewise((exp(n*x)/n, Ne(n, 0)), (x, True))
    assert manualintegrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))

    y = Symbol('y', positive=True)
    # 断言手动积分 (y + 1)**x 后得到 (y + 1)**x/log(y + 1)
    assert manualintegrate((y + 1)**x, x) == (y + 1)**x/log(y + 1)
    y = Symbol('y', zero=True)
    # 断言手动积分 (y + 1)**x 后得到 x
    assert manualintegrate((y + 1)**x, x) == x
    y = Symbol('y')
    n = Symbol('n', nonzero=True)
    # 断言手动积分 y**(n*x) 后得到 Piecewise((y**(n*x)/log(y), Ne(log(y), 0)), (n*x, True))/n
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (y**(n*x)/log(y), Ne(log(y), 0)), (n*x, True))/n
    y = Symbol('y', positive=True)
    # 断言手动积分 (y + 1)**(n*x) 后得到 (y + 1)**(n*x)/(n*log(y + 1))
    assert manualintegrate((y + 1)**(n*x), x) == \
        (y + 1)**(n*x)/(n*log(y + 1))
    a = Symbol('a', negative=True)
    b = Symbol('b')
    # 断言手动积分 1/(a + b*x**2) 后得到 atan(x/sqrt(a/b))/(b*sqrt(a/b))
    assert manualintegrate(1/(a + b*x**2), x) == atan(x/sqrt(a/b))/(b*sqrt(a/b))
    b = Symbol('b', negative=True)
    # 断言手动积分 1/(a + b*x**2) 后得到 atan(x/(sqrt(-a)*sqrt(-1/b)))/(b*sqrt(-a)*sqrt(-1/b))
    assert manualintegrate(1/(a + b*x**2), x) == \
        atan(x/(sqrt(-a)*sqrt(-1/b)))/(b*sqrt(-a)*sqrt(-1/b))
    # 断言手动积分 1/((x**a + y**b + 4)*sqrt(a*x**2 + 1)) 后得到 y**(-b)*Integral(x**(-a)/(y**(-b)*sqrt(a*x**2 + 1) + x**(-a)*sqrt(a*x**2 + 1) + 4*x**(-a)*y**(-b)*sqrt(a*x**2 + 1)), x)
    assert manualintegrate(1/((x**a + y**b + 4)*sqrt(a*x**2 + 1)), x) == \
        y**(-b)*Integral(x**(-a)/(y**(-b)*sqrt(a*x**2 + 1) +
        x**(-a)*sqrt(a*x**2 + 1) + 4*x**(-a)*y**(-b)*sqrt(a*x**2 + 1)), x)
    # 断言手动积分 1/((x**2 + 4)*sqrt(4*x**2 + 1)) 后得到 Integral(1/((x**2 + 4)*sqrt(4*x**2 + 1)), x)
    assert manualintegrate(1/(x**2 + 4)*sqrt(4*x**2 + 1), x) == \
        Integral(1/(x**2 + 4)*sqrt(4*x**2 + 1), x)
    # 断言手动积分 1/(x - a**x + x*b**2) 后得到 Integral(1/(-a**x + b**2*x + x), x)
    assert manualintegrate(1/(x - a**x + x*b**2), x) == \
        Integral(1/(-a**x + b**2*x + x), x)


# 标记为慢速测试的函数，用于测试 issue 2850
@slow
def test_issue_2850():
    # 断言手动积分 asin(x)*log(x) 后得到 -x*asin(x) - sqrt(-x**2 + 1) + (x*asin(x) + sqrt(-x**2 + 1))*log(x) - Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(asin(x)*log(x), x) == -x*asin(x) - sqrt(-x**2 + 1) \
            + (x*asin(x) + sqrt(-x**2 + 1))*log(x) - Integral(sqrt(-x**2 + 1)/x, x)
    # 断言手动积分 acos(x)*log(x) 后得到 -x*acos(x) + sqrt(-x**2 + 1) + (x*acos(x) - sqrt(-x**2 + 1))*log(x) + Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(acos(x)*log(x), x) == -x*acos(x) + sqrt(-x**2 + 1) + \
        (x*acos(x) - sqrt(-x**2 + 1))*log(x) + Integral(sqrt(-x**2 + 1)/x, x)
    # 断言手动积分 atan(x)*log(x) 后得到 -x*atan(x) + (x*atan(x) - log(x**2 + 1)/2)*log(x) + log(x**2 + 1)/2 + Integral(log(x**2 + 1)/x, x)/2
    assert manualintegrate(atan(x)*log(x), x) == -x*atan(x) + (x*atan(x) - \
            log(x**2 + 1)/2)*log(x) + log(x**2 + 1)/2 + Integral(log(x**2 + 1)/x, x)/2


# 定义用于测试 issue 9462 的函数
def test_issue_9462():
    # 断言手动积分 sin(2*x)*exp(x) 后得到 exp(x)*sin
# 定义一个测试函数，用于测试 manualintegrate 函数处理不同数学表达式的情况
def test_issue_10847():

    # 测试表达式：x**2 / (x**2 - c)
    assert manualintegrate(x**2 / (x**2 - c), x) == \
        c*Piecewise((atan(x/sqrt(-c))/sqrt(-c), Ne(c, 0)), (-1/x, True)) + x

    # 创建一个实数符号 rc，并测试表达式：x**2 / (x**2 - rc)
    rc = Symbol('c', real=True)
    assert manualintegrate(x**2 / (x**2 - rc), x) == \
        rc*Piecewise((atan(x/sqrt(-rc))/sqrt(-rc), rc < 0),
                     ((log(-sqrt(rc) + x) - log(sqrt(rc) + x))/(2*sqrt(rc)), True)) + x

    # 测试表达式：sqrt(x - y) * log(z / x)
    assert manualintegrate(sqrt(x - y) * log(z / x), x) == \
        4*y**2*Piecewise((atan(sqrt(x - y)/sqrt(y))/sqrt(y), Ne(y, 0)),
                         (-1/sqrt(x - y), True))/3 - 4*y*sqrt(x - y)/3 + \
        2*(x - y)**Rational(3, 2)*log(z/x)/3 + 4*(x - y)**Rational(3, 2)/9

    # 创建实数符号 ry 和 rz，并测试表达式：sqrt(x - ry) * log(rz / x)
    ry = Symbol('y', real=True)
    rz = Symbol('z', real=True)
    assert manualintegrate(sqrt(x - ry) * log(rz / x), x) == \
        4*ry**2*Piecewise((atan(sqrt(x - ry)/sqrt(ry))/sqrt(ry), ry > 0),
                         ((log(-sqrt(-ry) + sqrt(x - ry)) - log(sqrt(-ry) + sqrt(x - ry)))/(2*sqrt(-ry)), True))/3 \
                         - 4*ry*sqrt(x - ry)/3 + 2*(x - ry)**Rational(3, 2)*log(rz/x)/3 \
                         + 4*(x - ry)**Rational(3, 2)/9

    # 测试表达式：sqrt(x) * log(x)
    assert manualintegrate(sqrt(x) * log(x), x) == 2*x**Rational(3, 2)*log(x)/3 - 4*x**Rational(3, 2)/9

    # 测试表达式：sqrt(a*x + b) / x
    result = manualintegrate(sqrt(a*x + b) / x, x)
    assert result == Piecewise((-2*b*Piecewise(
        (-atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b), Ne(b, 0)),
        (1/sqrt(a*x + b), True)) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))
    assert piecewise_fold(result) == Piecewise(
        (2*b*atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b) + 2*sqrt(a*x + b), Ne(a, 0) & Ne(b, 0)),
        (-2*b/sqrt(a*x + b) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))

    # 创建实数符号 ra 和 rb，并测试表达式：sqrt(ra*x + rb) / x
    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(sqrt(ra*x + rb) / x, x) == \
        Piecewise(
            (-2*rb*Piecewise(
                (-atan(sqrt(ra*x + rb)/sqrt(-rb))/sqrt(-rb), rb < 0),
                (-I*(log(-sqrt(rb) + sqrt(ra*x + rb)) - log(sqrt(rb) + sqrt(ra*x + rb)))/(2*sqrt(-rb)), True)) +
             2*sqrt(ra*x + rb), Ne(ra, 0)),
            (sqrt(rb)*log(x), True))
    # 断言语句：验证手动积分函数对给定表达式的计算结果是否正确

    # 第一个断言：验证对 sqrt(ra*x + rb) / (x + rc) 的手动积分结果
    assert expand(manualintegrate(sqrt(ra*x + rb) / (x + rc), x)) == \
           Piecewise(
               # 分段函数的第一个分支
               (-2*ra*rc*Piecewise(
                   # Piecewise 条件判断：当 ra*rc - rb > 0 时的结果
                   (atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                   # Piecewise 条件判断：当 ra*rc - rb <= 0 时的结果
                   (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                    log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True))
               +
               # 分段函数的第二个分支
               2*rb*Piecewise(
                   # Piecewise 条件判断：当 ra*rc - rb > 0 时的结果
                   (atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                   # Piecewise 条件判断：当 ra*rc - rb <= 0 时的结果
                   (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                    log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True))
               +
               # 分段函数的第三个部分
               2*sqrt(ra*x + rb), Ne(ra, 0)),
               # 默认分支
               (sqrt(rb)*log(rc + x), True))

    # 第二个断言：验证对 sqrt(2*x + 3) / (x + 1) 的手动积分结果
    assert manualintegrate(sqrt(2*x + 3) / (x + 1), x) == 2*sqrt(2*x + 3) - log(sqrt(2*x + 3) + 1) + log(sqrt(2*x + 3) - 1)

    # 第三个断言：验证对 sqrt(2*x + 3) / 2 * x 的手动积分结果
    assert manualintegrate(sqrt(2*x + 3) / 2 * x, x) == (2*x + 3)**Rational(5, 2)/20 - (2*x + 3)**Rational(3, 2)/4

    # 第四个断言：验证对 x**Rational(3,2) * log(x) 的手动积分结果
    assert manualintegrate(x**Rational(3,2) * log(x), x) == 2*x**Rational(5,2)*log(x)/5 - 4*x**Rational(5,2)/25

    # 第五个断言：验证对 x**(-3) * log(x) 的手动积分结果
    assert manualintegrate(x**(-3) * log(x), x) == -log(x)/(2*x**2) - 1/(4*x**2)

    # 第六个断言：验证对 log(y)/(y**2*(1 - 1/y)) 的手动积分结果
    assert manualintegrate(log(y)/(y**2*(1 - 1/y)), y) == \
        log(y)*log(-1 + 1/y) - Integral(log(-1 + 1/y)/y, y)
# 测试处理 GitHub 问题 12899，检验对于 f(x,y) 对 x 的偏导数积分后是否等于 f(x,y) 对 x 的导数的积分
def test_issue_12899():
    # 断言积分求解函数对 f(x,y) 对 x 的偏导数的积分结果是否等于 f(x,y) 对 x 的导数
    assert manualintegrate(f(x,y).diff(x), y) == Integral(Derivative(f(x,y), x), y)

# 测试处理符号独立的常数情况，检验积分中的符号独立常数
def test_constant_independent_of_symbol():
    # 断言积分求解函数对 y 的 x 范围积分是否等于 x 乘以对 y 的积分
    assert manualintegrate(Integral(y, (x, 1, 2)), x) == \
        x * Integral(y, (x, 1, 2))

# 测试处理 GitHub 问题 12641，检验对 sin(2*x) 和 cos(x)*sin(2*x) 的积分求解
def test_issue_12641():
    # 断言积分求解函数对 sin(2*x) 的积分是否等于 -cos(2*x)/2
    assert manualintegrate(sin(2*x), x) == -cos(2*x)/2
    # 断言积分求解函数对 cos(x)*sin(2*x) 的积分是否等于 -2*cos(x)**3/3
    assert manualintegrate(cos(x)*sin(2*x), x) == -2*cos(x)**3/3
    # 断言积分求解函数对 (sin(2*x)*cos(x))/(1 + cos(x)) 的积分是否等于 -2*log(cos(x) + 1) - cos(x)**2 + 2*cos(x)
    assert manualintegrate((sin(2*x)*cos(x))/(1 + cos(x)), x) == \
        -2*log(cos(x) + 1) - cos(x)**2 + 2*cos(x)

# 缓慢测试处理 GitHub 问题 13297，检验对 sin(x) * cos(x)**5 的积分求解
@slow
def test_issue_13297():
    # 断言积分求解函数对 sin(x) * cos(x)**5 的积分是否等于 -cos(x)**6 / 6
    assert manualintegrate(sin(x) * cos(x)**5, x) == -cos(x)**6 / 6

# 测试处理 GitHub 问题 14470，检验对 1/(x*sqrt(x + 1)) 的积分求解
def test_issue_14470():
    # 断言是否为 log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1) 的原函数
    assert_is_integral_of(1/(x*sqrt(x + 1)), log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1))

# 缓慢测试处理 GitHub 问题 9858，检验对 exp(x)*cos(exp(x)) 等表达式的积分求解
@slow
def test_issue_9858():
    # 断言积分求解函数对 exp(x)*cos(exp(x)) 的积分是否等于 sin(exp(x))
    assert manualintegrate(exp(x)*cos(exp(x)), x) == sin(exp(x))
    # 断言积分求解函数对 exp(2*x)*cos(exp(x)) 的积分是否等于 exp(x)*sin(exp(x)) + cos(exp(x))
    assert manualintegrate(exp(2*x)*cos(exp(x)), x) == \
        exp(x)*sin(exp(x)) + cos(exp(x))
    # 检查对 exp(10*x)*sin(exp(x)) 的积分是否不含积分符号，并且其对 x 的导数是否等于 exp(10*x)*sin(exp(x))
    res = manualintegrate(exp(10*x)*sin(exp(x)), x)
    assert not res.has(Integral)
    assert res.diff(x) == exp(10*x)*sin(exp(x))
    # 检查对多个类似积分部分求解的例子
    assert manualintegrate(sum(x*exp(k*x) for k in range(1, 8)), x) == (
        x*exp(7*x)/7 + x*exp(6*x)/6 + x*exp(5*x)/5 + x*exp(4*x)/4 +
        x*exp(3*x)/3 + x*exp(2*x)/2 + x*exp(x) - exp(7*x)/49 -exp(6*x)/36 -
        exp(5*x)/25 - exp(4*x)/16 - exp(3*x)/9 - exp(2*x)/4 - exp(x))

# 测试处理 GitHub 问题 8520，检验对 x/(x**4 + 1) 和 x**2/(x**6 + 25) 的积分求解
def test_issue_8520():
    # 断言积分求解函数对 x/(x**4 + 1) 的积分是否等于 atan(x**2)/2
    assert manualintegrate(x/(x**4 + 1), x) == atan(x**2)/2
    # 断言积分求解函数对 x**2/(x**6 + 25) 的积分是否等于 atan(x**3/5)/15
    assert manualintegrate(x**2/(x**6 + 25), x) == atan(x**3/5)/15
    # 设置 f 为 x/(9*x**4 + 4)**2，检查其积分的导数化简后是否等于 f 本身
    f = x/(9*x**4 + 4)**2
    assert manualintegrate(f, x).diff(x).factor() == f

# 测试处理 GitHub 问题 15471，检验对 log(x)*cos(log(x))/x**Rational(3, 4) 的积分求解
@slow
def test_issue_15471():
    # 断言 log(x)*cos(log(x))/x**Rational(3, 4) 的原函数是否等于 -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 + (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x)
    assert_is_integral_of(log(x)*cos(log(x))/x**Rational(3, 4), -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 + (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x))

# 测试处理二次多项式的分母问题，检验对 (5*x + 2)/(3*x**2 - 2*x + 8) 和 3/(2*x**2 + 3*x + 1) 的积分求解
def test_quadratic_denom():
    # 断言积分求解函数对 (5*x + 2)/(3*x**2 - 2*x + 8) 的积分是否等于 5*log(3*x**2 - 2*x + 8)/6 + 11*sqrt(23)*atan(3*sqrt(23)*(x - Rational(1, 3))/23)/69
    f = (5*x + 2)/(3*x**2 - 2*x + 8)
    assert manualintegrate(f, x) == 5*log(3*x**2 - 2*x + 8)/6 + 11*sqrt(23)*atan(3*sqrt(23)*(x - Rational(1, 3))/23)/69
    # 断言积分求解函数对 3/(2*x**2 + 3*x + 1) 的积分是否等于 3*log(4*x + 2) - 3*log(4*x + 4)
    g = 3/(2*x**2 + 3*x + 1)
    assert manualintegrate(g, x) == 3*log(4*x + 2) - 3*log(4*x + 4)

# 测试处理 GitHub 问题 22757，检验对 sin(x) 的积分求解
def test_issue_22757():
    # 断言积分求解函数对 sin(x) 的积分是否等于 y * sin(x)
    assert manualintegrate(sin(x), y) == y * sin(x)

# 测试处理 GitHub 问题 23348，检验对 tan(x) 的积
    # 使用 SymPy 计算积分 ∫(1/√(x² - 1)) dx 在区间 [-2, -1] 上的值，手动指定进行计算
    i = Integral(1/sqrt(x**2 - 1), (x, -2, -1)).doit(manual=True)
    
    # 断言计算得到的积分值与预期结果相等，预期结果是 -log(4 - 2*sqrt(3)) + log(2)
    assert i == -log(4 - 2*sqrt(3)) + log(2)
    
    # 断言将积分值转换为数值，并将其字符串表示与预期字符串 '1.31695789692482' 相等
    assert str(i.n()) == '1.31695789692482'
# 定义测试函数，用于验证 Issue 25093
def test_issue_25093():
    # 定义符号变量 ap 和 an，分别具有正和负属性
    ap = Symbol('ap', positive=True)
    an = Symbol('an', negative=True)
    
    # 断言对指定的表达式进行手动积分后的结果
    assert manualintegrate(exp(a*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(a)*x)/(2*sqrt(a))
    assert manualintegrate(exp(ap*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(ap)*x)/(2*sqrt(ap))
    assert manualintegrate(exp(an*x**2 + b), x) == -sqrt(pi)*exp(b)*erf(an*x/sqrt(-an))/(2*sqrt(-an))
    assert manualintegrate(sin(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(sin(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))
    assert manualintegrate(cos(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(-sin(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))


# 定义测试函数，用于验证嵌套幂的积分
def test_nested_pow():
    # 断言各表达式是否是 sqrt(x**2) 的积分
    assert_is_integral_of(sqrt(x**2), x*sqrt(x**2)/2)
    assert_is_integral_of(sqrt(x**(S(5)/3)), 6*x*sqrt(x**(S(5)/3))/11)
    assert_is_integral_of(1/sqrt(x**2), x*log(x)/sqrt(x**2))
    assert_is_integral_of(x*sqrt(x**(-4)), x**2*sqrt(x**-4)*log(x))
    
    # 定义复杂的表达式 f 和其两种可能的积分结果 F1 和 F2
    f = (c*(a+b*x)**d)**e
    F1 = (c*(a + b*x)**d)**e*(a/b + x)/(d*e + 1)
    F2 = (c*(a + b*x)**d)**e*(a/b + x)*log(a/b + x)
    
    # 断言 f 的手动积分结果是否符合分段函数的形式
    assert manualintegrate(f, x) == \
        Piecewise((Piecewise((F1, Ne(d*e, -1)), (F2, True)), Ne(b, 0)), (x*(a**d*c)**e, True))
    
    # 断言 F1 对 x 的导数是否等于 f
    assert F1.diff(x).equals(f)
    
    # 断言当 d*e = -1 时，F2 对 x 的导数是否等于 f
    assert F2.diff(x).subs(d*e, -1).equals(f)


# 定义测试函数，用于验证 sqrt 线性表达式的积分
def test_manualintegrate_sqrt_linear():
    # 断言对给定的 sqrt 线性表达式进行手动积分的结果
    assert_is_integral_of((5*x**3+4)/sqrt(2+3*x),
                          10*(3*x + 2)**(S(7)/2)/567 - 4*(3*x + 2)**(S(5)/2)/27 +
                          40*(3*x + 2)**(S(3)/2)/81 + 136*sqrt(3*x + 2)/81)
    assert manualintegrate(x/sqrt(a+b*x)**3, x) == \
        Piecewise((Mul(2, b**-2, a/sqrt(a + b*x) + sqrt(a + b*x)), Ne(b, 0)), (x**2/(2*a**(S(3)/2)), True))
    assert_is_integral_of((sqrt(3*x+3)+1)/((2*x+2)**(1/S(3))+1),
                          3*sqrt(6)*(2*x + 2)**(S(7)/6)/14 - 3*sqrt(6)*(2*x + 2)**(S(5)/6)/10 -
                          3*sqrt(6)*(2*x + 2)**(S.One/6)/2 + 3*(2*x + 2)**(S(2)/3)/4 - 3*(2*x + 2)**(S.One/3)/2 +
                          sqrt(6)*sqrt(2*x + 2)/2 + 3*log((2*x + 2)**(S.One/3) + 1)/2 +
                          3*sqrt(6)*atan((2*x + 2)**(S.One/6))/2)
    assert_is_integral_of(sqrt(x+sqrt(x)),
                          2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8)
    assert_is_integral_of(sqrt(2*x+3+sqrt(4*x+5))**3,
                          sqrt(2*x + sqrt(4*x + 5) + 3) *
                          (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2)


# 定义测试函数，用于验证 sqrt 二次表达式的积分
def test_manualintegrate_sqrt_quadratic():
    # 断言对给定的 sqrt 二次表达式进行手动积分的结果
    assert_is_integral_of(1/sqrt((x - I)**2-1), log(2*x + 2*sqrt(x**2 - 2*I*x - 2) - 2*I))
    assert_is_integral_of(1/sqrt(3*x**2+4*x+5), sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3)
    assert_is_integral_of(1/sqrt(-3*x**2+4*x+5), sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3)
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of(1/sqrt(3*x**2+4*x-5), sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3)
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of(1/sqrt(4*x**2-4*x+1), (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2)))
    
    # 调用 manualintegrate 函数，计算给定表达式的积分，并与预期结果进行比较
    assert manualintegrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x+5),
                          7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9)
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of((7*x+6)/sqrt(-3*x**2+4*x+5),
                          -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9)
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x-5),
                          7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9)
    
    # 调用 manualintegrate 函数，计算给定表达式的积分，并与预期结果进行比较
    assert manualintegrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))
    
    # 调用 manualintegrate 函数，计算给定表达式的积分，并与预期结果进行比较
    assert manualintegrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2), x) == \
        sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of(sqrt(53225*x**2-66732*x+23013),
                          (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) +
                          111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250)
    
    # 调用 manualintegrate 函数，计算给定表达式的积分，并与预期结果进行比较
    assert manualintegrate(sqrt(a+c*x**2), x) == \
        Piecewise((a*Piecewise((log(2*sqrt(c)*sqrt(a + c*x**2) + 2*c*x)/sqrt(c), Ne(a, 0)),
                               (x*log(x)/sqrt(c*x**2), True))/2 + x*sqrt(a + c*x**2)/2, Ne(c, 0)),
                  (sqrt(a)*x, True))
    
    # 调用 manualintegrate 函数，计算给定表达式的积分，并与预期结果进行比较
    assert manualintegrate(sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((a/2 - b**2/(8*c)) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
                  (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
                  (sqrt(a)*x, True))
    
    # 调用自定义函数 assert_is_integral_of，验证给定表达式是否是所提供积分的确切解
    assert_is_integral_of(x*sqrt(x**2+2*x+4),
                          (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2)
# 定义一个测试函数，用于测试多项式乘积和导数的计算结果是否正确
def test_mul_pow_derivative():
    # 断言：x * sec(x) * tan(x) 的积分应该等于 x * sec(x) - log(tan(x) + sec(x))
    assert_is_integral_of(x*sec(x)*tan(x), x*sec(x) - log(tan(x) + sec(x)))
    
    # 断言：x * sec(x)**2 的积分应该等于 x * tan(x) + log(cos(x))
    assert_is_integral_of(x*sec(x)**2, x*tan(x) + log(cos(x)))
    
    # 断言：x**3 * Derivative(f(x), (x, 4)) 的积分应该等于 x**3 * Derivative(f(x), (x, 3))
    #       - 3 * x**2 * Derivative(f(x), (x, 2)) + 6 * x * Derivative(f(x), x) - 6 * f(x)
    assert_is_integral_of(x**3*Derivative(f(x), (x, 4)),
                          x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) +
                          6*x*Derivative(f(x), x) - 6*f(x))
```