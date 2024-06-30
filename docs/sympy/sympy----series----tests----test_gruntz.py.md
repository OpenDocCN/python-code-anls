# `D:\src\scipysrc\sympy\sympy\series\tests\test_gruntz.py`

```
"""
这个测试套件测试使用自底向上的方法的极限算法。
查看 limits2.py 中的文档。算法本身是高度递归的，
因此 "compare" 在逻辑上是算法的最底层部分，但在某种意义上，它是最复杂的部分，
因为它需要计算极限来返回结果。

然而，算法的其余部分依赖于 compare 正确运行。
"""

from sympy.core import EulerGamma                             # 导入 EulerGamma 符号
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)  # 导入常见的数学常数和对象
from sympy.core.singleton import S                            # 导入符号 S
from sympy.core.symbol import Symbol                          # 导入符号 Symbol
from sympy.functions.elementary.exponential import (exp, log) # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt    # 导入平方根函数
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)  # 导入三角函数
from sympy.functions.elementary.complexes import sign as _sign  # 导入复数函数
from sympy.functions.special.error_functions import (Ei, erf) # 导入错误函数
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)  # 导入 gamma 函数
from sympy.functions.special.zeta_functions import zeta      # 导入 zeta 函数
from sympy.polys.polytools import cancel                      # 导入多项式工具中的 cancel 函数
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh  # 导入双曲函数
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, sign  # 导入 Gruntz 等级函数
from sympy.testing.pytest import XFAIL, skip, slow             # 导入测试相关函数

x = Symbol('x', real=True)                                   # 创建实数符号 x
m = Symbol('m', real=True)                                   # 创建实数符号 m

runslow = False                                               # 是否运行慢速测试的标志

def _sskip():                                                 # 定义一个函数 _sskip，用于跳过慢速测试
    if not runslow:
        skip("slow")

@slow                                                          # 将以下函数标记为慢速测试
def test_gruntz_evaluation():                                  # 定义测试函数 test_gruntz_evaluation
    # Gruntz 的论文中的测试例子，页码 122 到 123
    # 8.1
    assert gruntz(exp(x)*(exp(1/x - exp(-x)) - exp(1/x)), x, oo) == -1  # 断言，验证 gruntz 函数的结果是否为 -1
    # 8.2
    assert gruntz(exp(x)*(exp(1/x + exp(-x) + exp(-x**2))
                  - exp(1/x - exp(-exp(x)))), x, oo) == 1            # 断言，验证 gruntz 函数的结果是否为 1
    # 8.3
    assert gruntz(exp(exp(x - exp(-x))/(1 - 1/x)) - exp(exp(x)), x, oo) is oo  # 断言，验证 gruntz 函数的结果是否为无穷大
    # 8.5
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(exp(x))), x, oo) is oo  # 断言，验证 gruntz 函数的结果是否为无穷大
    # 8.6
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(x))))),
                  x, oo) is oo                                     # 断言，验证 gruntz 函数的结果是否为无穷大
    # 8.7
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(exp(x)))))),
                  x, oo) == 1                                      # 断言，验证 gruntz 函数的结果是否为 1
    # 8.8
    assert gruntz(exp(exp(x)) / exp(exp(x - exp(-exp(exp(x))))), x, oo) == 1  # 断言，验证 gruntz 函数的结果是否为 1
    # 8.9
    assert gruntz(log(x)**2 * exp(sqrt(log(x))*(log(log(x)))**2
                  * exp(sqrt(log(log(x))) * (log(log(log(x))))**3)) / sqrt(x),
                  x, oo) == 0                                      # 断言，验证 gruntz 函数的结果是否为 0
    # 8.10
    assert gruntz((x*log(x)*(log(x*exp(x) - x**2))**2)
                  / (log(log(x**2 + 2*exp(exp(3*x**3*log(x)))))), x, oo) == Rational(1, 3)  # 断言，验证 gruntz 函数的结果是否为 1/3
    # 8.11
    assert gruntz((exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1)))) - exp(x))/x,
                  x, oo) == -exp(2)                                 # 断言，验证 gruntz 函数的结果是否为 -e^2
    # 8.12
    assert gruntz((3**x + 5**x)**(1/x), x, oo) == 5                 # 断言，验证 gruntz 函数的结果是否为 5
    # 8.13
    assert gruntz(x/log(x**(log(x**(log(2)/log(x))))), x, oo) is oo  # 断言，验证 gruntz 函数的结果是否为无穷大
    # 8.14
    assert gruntz(exp(exp(2*log(x**5 + x)*log(log(x))))
                  / exp(exp(10*log(x)*log(log(x)))), x, oo) is oo    # 断言，验证 gruntz 函数的结果是否为无穷大
    # 8.15 （这里缺少最后的测试用例，可能因为截断）
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否为正无穷
    assert gruntz(exp(exp(Rational(5, 2)*x**Rational(-5, 7) + Rational(21, 8)*x**Rational(6, 11)
                              + 2*x**(-8) + Rational(54, 17)*x**Rational(49, 45)))**8
                      / log(log(-log(Rational(4, 3)*x**Rational(-5, 14))))**Rational(7, 6), x, oo) is oo
    # 8.16
    
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否等于 1
    assert gruntz((exp(4*x*exp(-x)/(1/exp(x) + 1/exp(2*x**2/(x + 1)))) - exp(x))
                      / exp(x)**4, x, oo) == 1
    # 8.17
    
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否等于 1
    assert gruntz(exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1))))/exp(x), x, oo) \
            == 1
    # 8.19
    
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否等于 1
    assert gruntz(log(x)*(log(log(x) + log(log(x))) - log(log(x)))
                      / (log(log(x) + log(log(log(x))))), x, oo) == 1
    # 8.20
    
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否等于自然常数 e
    assert gruntz(exp((log(log(x + exp(log(x)*log(log(x))))))
                      / (log(log(log(exp(x) + x + log(x)))))), x, oo) == E
    # Another
    
    # 断言：验证当 x 趋向正无穷时，表达式的极限是否为正无穷
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(x)), x, oo) is oo
def test_gruntz_evaluation_slow():
    _sskip()
    # 跳过当前测试（这里的_sskip函数用于测试跳过）
    # 8.4
    assert gruntz(exp(exp(exp(x)/(1 - 1/x)))
                  - exp(exp(exp(x)/(1 - 1/x - log(x)**(-log(x))))), x, oo) is -oo
    # 断言测试Gruntz极限，验证结果是否为负无穷

    # 8.18
    assert gruntz((exp(exp(-x/(1 + exp(-x))))*exp(-x/(1 + exp(-x/(1 + exp(-x)))))
                   *exp(exp(-x + exp(-x/(1 + exp(-x))))))
                  / (exp(-x/(1 + exp(-x))))**2 - exp(x) + x, x, oo) == 2
    # 断言测试Gruntz极限，验证结果是否为2


@slow
def test_gruntz_eval_special():
    # Gruntz, p. 126
    # 针对特殊情况的Gruntz极限测试
    assert gruntz(exp(x)*(sin(1/x + exp(-x)) - sin(1/x + exp(-x**2))), x, oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1
    assert gruntz((erf(x - exp(-exp(x))) - erf(x)) * exp(exp(x)) * exp(x**2),
                  x, oo) == -2/sqrt(pi)
    # 断言测试Gruntz极限，验证结果是否为-2/sqrt(pi)
    assert gruntz(exp(exp(x)) * (exp(sin(1/x + exp(-exp(x)))) - exp(sin(1/x))),
                  x, oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1
    assert gruntz(exp(x)*(gamma(x + exp(-x)) - gamma(x)), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(exp(exp(digamma(digamma(x))))/x, x, oo) == exp(Rational(-1, 2))
    # 断言测试Gruntz极限，验证结果是否为exp(-1/2)
    assert gruntz(exp(exp(digamma(log(x))))/x, x, oo) == exp(Rational(-1, 2))
    # 断言测试Gruntz极限，验证结果是否为exp(-1/2)
    assert gruntz(digamma(digamma(digamma(x))), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(loggamma(loggamma(x)), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(((gamma(x + 1/gamma(x)) - gamma(x))/log(x) - cos(1/x))
                  * x*log(x), x, oo) == Rational(-1, 2)
    # 断言测试Gruntz极限，验证结果是否为-1/2
    assert gruntz(x * (gamma(x - 1/gamma(x)) - gamma(x) + log(x)), x, oo) \
        == S.Half
    # 断言测试Gruntz极限，验证结果是否为1/2
    assert gruntz((gamma(x + 1/gamma(x)) - gamma(x)) / log(x), x, oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1


def test_gruntz_eval_special_slow():
    _sskip()
    # 跳过当前测试（这里的_sskip函数用于测试跳过）
    assert gruntz(gamma(x + 1)/sqrt(2*pi)
                  - exp(-x)*(x**(x + S.Half) + x**(x - S.Half)/12), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(exp(exp(exp(digamma(digamma(digamma(x))))))/x, x, oo) == 0
    # 断言测试Gruntz极限，验证结果是否为0


@XFAIL
def test_grunts_eval_special_slow_sometimes_fail():
    _sskip()
    # 跳过当前测试（这里的_sskip函数用于测试跳过）
    # XXX This sometimes fails!!!
    # 这个有时会失败！！！
    assert gruntz(exp(gamma(x - exp(-x))*exp(1/x)) - exp(gamma(x)), x, oo) is oo


def test_gruntz_Ei():
    assert gruntz((Ei(x - exp(-exp(x))) - Ei(x)) *exp(-x)*exp(exp(x))*x, x, oo) == -1
    # 断言测试Gruntz极限，验证结果是否为-1


@XFAIL
def test_gruntz_eval_special_fail():
    # TODO zeta function series
    # TODO 费曼zeta函数级数
    assert gruntz(
        exp((log(2) + 1)*x) * (zeta(x + exp(-x)) - zeta(x)), x, oo) == -log(2)

    # TODO 8.35 - 8.37 (bessel, max-min)
    # TODO 8.35 - 8.37 (贝塞尔函数，最大-最小)


def test_gruntz_hyperbolic():
    assert gruntz(cosh(x), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(cosh(x), x, -oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(sinh(x), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(sinh(x), x, -oo) is -oo
    # 断言测试Gruntz极限，验证结果是否为负无穷
    assert gruntz(2*cosh(x)*exp(x), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(2*cosh(x)*exp(x), x, -oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1
    assert gruntz(2*sinh(x)*exp(x), x, oo) is oo
    # 断言测试Gruntz极限，验证结果是否为正无穷
    assert gruntz(2*sinh(x)*exp(x), x, -oo) == -1
    # 断言测试Gruntz极限，验证结果是否为-1
    assert gruntz(tanh(x), x, oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1
    assert gruntz(tanh(x), x, -oo) == -1
    # 断言测试Gruntz极限，验证结果是否为-1
    assert gruntz(coth(x), x, oo) == 1
    # 断言测试Gruntz极限，验证结果是否为1
    assert gruntz(coth(x), x, -oo) == -1
    # 断言测试Gruntz极限，验证结果是否为-1


def test_compare1():
    assert compare(2, x, x) == "<"
    # 断言测试compare函数，验证结果是否为"<"
    assert compare(x, exp(x), x) == "<"
    # 断言测试compare函数，验证结果是否为"<"
    assert compare(exp(x), exp(x**2), x) == "<"
    # 断言测试compare函数，验证结果是否为"<"
    assert compare(exp(x**2), exp(exp(x)), x) == "<"
    # 断言测试compare函数，验证结果是否为"<"
    assert compare(1, exp(exp(x)), x) == "<"
    # 断言测试compare函数，验证结果是否为"<"
    # 断言：比较函数 compare 的结果与预期值 ">" 是否相等，x 为变量
    assert compare(x, 2, x) == ">"
    # 断言：比较函数 compare 的结果与预期值 ">" 是否相等，exp(x) 为变量
    assert compare(exp(x), x, x) == ">"
    # 断言：比较函数 compare 的结果与预期值 ">" 是否相等，exp(x**2) 为变量
    assert compare(exp(x**2), exp(x), x) == ">"
    # 断言：比较函数 compare 的结果与预期值 ">" 是否相等，exp(exp(x)) 为变量
    assert compare(exp(exp(x)), exp(x**2), x) == ">"
    # 断言：比较函数 compare 的结果与预期值 ">" 是否相等，1 为变量
    assert compare(exp(exp(x)), 1, x) == ">"

    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(2, 3, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(3, -5, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(2, -5, x) == "="

    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(x, x**2, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(x**2, x**3, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量
    assert compare(x**3, 1/x, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量，m 为变量
    assert compare(1/x, x**m, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，x 为变量，m 为变量
    assert compare(x**m, -x, x) == "="

    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(x) 为变量
    assert compare(exp(x), exp(-x), x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(-x) 为变量
    assert compare(exp(-x), exp(2*x), x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(2*x) 为变量
    assert compare(exp(2*x), exp(x)**2, x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(x)**2 为变量
    assert compare(exp(x)**2, exp(x + exp(-x)), x) == "="
    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(x) 为变量
    assert compare(exp(x), exp(x + exp(-x)), x) == "="

    # 断言：比较函数 compare 的结果与预期值 "=" 是否相等，exp(x**2) 为变量
    assert compare(exp(x**2), 1/exp(x**2), x) == "="
def test_compare2():
    assert compare(exp(x), x**5, x) == ">"
    assert compare(exp(x**2), exp(x)**2, x) == ">"
    assert compare(exp(x), exp(x + exp(-x)), x) == "="
    assert compare(exp(x + exp(-x)), exp(x), x) == "="
    assert compare(exp(x + exp(-x)), exp(-x), x) == "="
    assert compare(exp(-x), x, x) == ">"
    assert compare(x, exp(-x), x) == "<"
    assert compare(exp(x + 1/x), x, x) == ">"
    assert compare(exp(-exp(x)), exp(x), x) == ">"
    assert compare(exp(exp(-exp(x)) + x), exp(-exp(x)), x) == "<"

# 测试函数 test_compare2，用于测试 compare 函数对指数表达式进行比较


def test_compare3():
    assert compare(exp(exp(x)), exp(x + exp(-exp(x))), x) == ">"
# 测试函数 test_compare3，用于测试 compare 函数对复杂指数表达式进行比较


def test_sign1():
    assert sign(Rational(0), x) == 0
    assert sign(Rational(3), x) == 1
    assert sign(Rational(-5), x) == -1
    assert sign(log(x), x) == 1
    assert sign(exp(-x), x) == 1
    assert sign(exp(x), x) == 1
    assert sign(-exp(x), x) == -1
    assert sign(3 - 1/x, x) == 1
    assert sign(-3 - 1/x, x) == -1
    assert sign(sin(1/x), x) == 1
    assert sign((x**Integer(2)), x) == 1
    assert sign(x**2, x) == 1
    assert sign(x**5, x) == 1

# 测试函数 test_sign1，用于测试 sign 函数对各种数学表达式的符号进行判断


def test_sign2():
    assert sign(x, x) == 1
    assert sign(-x, x) == -1
    y = Symbol("y", positive=True)
    assert sign(y, x) == 1
    assert sign(-y, x) == -1
    assert sign(y*x, x) == 1
    assert sign(-y*x, x) == -1

# 测试函数 test_sign2，用于测试 sign 函数对变量和符号的符号进行判断


def mmrv(a, b):
    return set(mrv(a, b)[0].keys())

# 定义函数 mmrv，用于计算表达式的主要可约变量集合


def test_mrv1():
    assert mmrv(x, x) == {x}
    assert mmrv(x + 1/x, x) == {x}
    assert mmrv(x**2, x) == {x}
    assert mmrv(log(x), x) == {x}
    assert mmrv(exp(x), x) == {exp(x)}
    assert mmrv(exp(-x), x) == {exp(-x)}
    assert mmrv(exp(x**2), x) == {exp(x**2)}
    assert mmrv(-exp(1/x), x) == {x}
    assert mmrv(exp(x + 1/x), x) == {exp(x + 1/x)}

# 测试函数 test_mrv1，用于测试 mmrv 函数对不同类型表达式的主要可约变量的计算


def test_mrv2a():
    assert mmrv(exp(x + exp(-exp(x))), x) == {exp(-exp(x))}
    assert mmrv(exp(x + exp(-x)), x) == {exp(x + exp(-x)), exp(-x)}
    assert mmrv(exp(1/x + exp(-x)), x) == {exp(-x)}

# 测试函数 test_mrv2a，用于测试 mmrv 函数对复杂表达式的主要可约变量的计算


def test_mrv2b():
    assert mmrv(exp(x + exp(-x**2)), x) == {exp(-x**2)}

# 测试函数 test_mrv2b，用于测试 mmrv 函数对特定复杂表达式的主要可约变量的计算


def test_mrv2c():
    assert mmrv(
        exp(-x + 1/x**2) - exp(x + 1/x), x) == {exp(x + 1/x), exp(1/x**2 - x)}

# 测试函数 test_mrv2c，用于测试 mmrv 函数对特定复杂表达式的主要可约变量的计算


def test_mrv3():
    assert mmrv(exp(x**2) + x*exp(x) + log(x)**x/x, x) == {exp(x**2)}
    assert mmrv(
        exp(x)*(exp(1/x + exp(-x)) - exp(1/x)), x) == {exp(x), exp(-x)}
    assert mmrv(log(
        x**2 + 2*exp(exp(3*x**3*log(x)))), x) == {exp(exp(3*x**3*log(x)))}
    assert mmrv(log(x - log(x))/log(x), x) == {x}
    assert mmrv(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == {exp(x), exp(-x)}
    assert mmrv(
        1/exp(-x + exp(-x)) - exp(x), x) == {exp(x), exp(-x), exp(x - exp(-x))}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)), x) == {exp(x*exp(x))}
    assert mmrv(exp(exp(log(log(x) + 1/x))), x) == {x}

# 测试函数 test_mrv3，用于测试 mmrv 函数对更复杂表达式的主要可约变量的计算


def test_mrv4():
    ln = log

# 定义函数 test_mrv4，用于初始化 log 函数
    # 断言：验证 mmrv 函数对给定表达式 x 的计算结果是否与集合 {x} 相等
    assert mmrv((ln(ln(x) + ln(ln(x))) - ln(ln(x)))/ln(ln(x) + ln(ln(ln(x))))*ln(x),
                x) == {x}
    
    # 断言：验证 mmrv 函数对给定表达式 x 的计算结果是否与集合 {exp(x*exp(x))} 相等
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)) - exp(exp(log(log(x) + 1/x))), x) == \
            {exp(x*exp(x))}
# 定义函数mrewrite，接受三个参数a, b, c，并调用函数rewrite
def mrewrite(a, b, c):
    # 调用rewrite函数，传入参数a[1], a[0], b, c，并返回结果
    return rewrite(a[1], a[0], b, c)


# 定义测试函数test_rewrite1，测试mrewrite函数的行为
def test_rewrite1():
    # 初始化变量e为exp(x)，并断言mrewrite(mrv(e, x), x, m)等于(1/m, -x)
    e = exp(x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x)
    
    # 初始化变量e为exp(x**2)，并断言mrewrite(mrv(e, x), x, m)等于(1/m, -x**2)
    e = exp(x**2)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x**2)
    
    # 初始化变量e为exp(x + 1/x)，并断言mrewrite(mrv(e, x), x, m)等于(1/m, -x - 1/x)
    e = exp(x + 1/x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x - 1/x)
    
    # 初始化变量e为1/exp(-x + exp(-x)) - exp(x)，并断言mrewrite(mrv(e, x), x, m)等于(1/(m*exp(m)) - 1/m, -x)
    e = 1/exp(-x + exp(-x)) - exp(x)
    assert mrewrite(mrv(e, x), x, m) == (1/(m*exp(m)) - 1/m, -x)


# 定义测试函数test_rewrite2，测试mrewrite函数的行为
def test_rewrite2():
    # 初始化变量e为exp(x)*log(log(exp(x)))，并断言mmrv(e, x)等于{exp(x)}
    e = exp(x)*log(log(exp(x)))
    assert mmrv(e, x) == {exp(x)}
    # 断言mrewrite(mrv(e, x), x, m)等于(1/m*log(x), -x)
    assert mrewrite(mrv(e, x), x, m) == (1/m*log(x), -x)


# 定义测试函数test_rewrite3，测试mrewrite函数的行为
def test_rewrite3():
    # 初始化变量e为exp(-x + 1/x**2) - exp(x + 1/x)，并断言mrewrite(mrv(e, x), x, m)在指定的列表中
    e = exp(-x + 1/x**2) - exp(x + 1/x)
    assert mrewrite(mrv(e, x), x, m) in [(-1/m + m*exp(1/x + 1/x**2), -x - 1/x), (m - 1/m*exp(1/x + x**(-2)), x**(-2) - x)]


# 定义测试函数test_mrv_leadterm1，测试mrv_leadterm函数的行为
def test_mrv_leadterm1():
    # 断言mrv_leadterm(-exp(1/x), x)等于(-1, 0)
    assert mrv_leadterm(-exp(1/x), x) == (-1, 0)
    # 断言mrv_leadterm(1/exp(-x + exp(-x)) - exp(x), x)等于(-1, 0)
    assert mrv_leadterm(1/exp(-x + exp(-x)) - exp(x), x) == (-1, 0)
    # 断言mrv_leadterm((exp(1/x - exp(-x)) - exp(1/x))*exp(x), x)等于(-exp(1/x), 0)
    assert mrv_leadterm((exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == (-exp(1/x), 0)


# 定义测试函数test_mrv_leadterm2，测试mrv_leadterm函数的行为
def test_mrv_leadterm2():
    # 断言mrv_leadterm((log(exp(x) + x) - x)/log(exp(x) + log(x))*exp(x), x)等于(1, 0)
    assert mrv_leadterm((log(exp(x) + x) - x)/log(exp(x) + log(x))*exp(x), x) == (1, 0)


# 定义测试函数test_mrv_leadterm3，测试mrv_leadterm函数的行为
def test_mrv_leadterm3():
    # 断言mmrv(exp(-x + exp(-x)*exp(-x*log(x))), x)等于{exp(-x - x*log(x))}
    assert mmrv(exp(-x + exp(-x)*exp(-x*log(x))), x) == {exp(-x - x*log(x))}
    # 断言mrv_leadterm(exp(-x + exp(-x)*exp(-x*log(x))), x)等于(exp(-x), 0)
    assert mrv_leadterm(exp(-x + exp(-x)*exp(-x*log(x))), x) == (exp(-x), 0)


# 定义测试函数test_limit1，测试gruntz函数的行为
def test_limit1():
    # 断言gruntz(x, x, oo)等于oo
    assert gruntz(x, x, oo) is oo
    # 断言gruntz(x, x, -oo)等于-oo
    assert gruntz(x, x, -oo) is -oo
    # 断言gruntz(-x, x, oo)等于-oo
    assert gruntz(-x, x, oo) is -oo
    # 断言gruntz(x**2, x, -oo)等于oo
    assert gruntz(x**2, x, -oo) is oo
    # 断言gruntz(-x**2, x, oo)等于-oo
    assert gruntz(-x**2, x, oo) is -oo
    # 断言gruntz(x*log(x), x, 0, dir="+")等于0
    assert gruntz(x*log(x), x, 0, dir="+") == 0
    # 断言gruntz(1/x, x, oo)等于0
    assert gruntz(1/x, x, oo) == 0
    # 断言gruntz(exp(x), x, oo)等于oo
    assert gruntz(exp(x), x, oo) is oo
    # 断言gruntz(-exp(x), x, oo)等于-oo
    assert gruntz(-exp(x), x, oo) is -oo
    # 断言gruntz(exp(x)/x, x, oo)等于oo
    assert gruntz(exp(x)/x, x, oo) is oo
    # 断言gruntz(1/x - exp(-x), x, oo)等于0
    assert gruntz(1/x - exp(-x), x, oo) == 0
    # 断言gruntz(x + 1/x, x, oo)等于oo
    assert gruntz(x + 1/x, x, oo) is oo


# 定义测试函数test_limit2，测试gruntz函数的行为
def test_limit2():
    # 断言gruntz(x**x, x, 0, dir="+")等于1
    assert gruntz(x**x, x, 0, dir="+") == 1
    # 断言gruntz((exp(x) - 1)/x, x, 0)等于1
    assert gruntz((exp(x) - 1)/x, x, 0) == 1
    # 断言gruntz(1 + 1/x, x, oo)等于1
    assert gruntz(1 + 1/x, x, oo) == 1
    # 断言gruntz(-exp(1/x), x, oo)等于-1
    assert gruntz(-exp(1/x), x, oo) == -1
    # 断言gruntz(x + exp(-x), x, oo)等于oo
    assert gruntz(x + exp(-x), x, oo) is oo
    # 断言gruntz(x + exp(-x**2), x, oo)等于oo
    assert gruntz(x + exp(-x**2), x, oo) is oo
    # 断言gruntz(x + exp(-exp(x)), x, oo)等于oo
    assert gruntz(x + exp(-exp(x)), x, oo) is oo
    # 断言gruntz(13 + 1/x - exp(-x), x, oo)等于13
    assert gruntz(13 + 1/x - exp(-x), x, oo) == 13


# 定义测试函数test_limit3，测试gruntz函数的行为
def test_limit3():
    # 初始化符号变量a
    a = Symbol
    # 创建符号变量 y
    y = Symbol("y")
    # 使用 gruntz 函数检查极限，验证 I*x 当 x 趋向无穷时的结果是否为 I*oo
    assert gruntz(I*x, x, oo) == I*oo
    # 使用 gruntz 函数检查极限，验证 y*I*x 当 x 趋向无穷时的结果是否为 y*I*oo
    assert gruntz(y*I*x, x, oo) == y*I*oo
    # 使用 gruntz 函数检查极限，验证 y*3*I*x 当 x 趋向无穷时的结果是否为 y*I*oo
    assert gruntz(y*3*I*x, x, oo) == y*I*oo
    # 使用 gruntz 函数检查极限，验证 y*3*sin(I)*x 当 x 趋向无穷时的结果经过简化且重新表达后是否为 _sign(y)*I*oo
    assert gruntz(y*3*sin(I)*x, x, oo).simplify().rewrite(_sign) == _sign(y)*I*oo
def test_issue_4814():
    assert gruntz((x + 1)**(1/log(x + 1)), x, oo) == E

# 测试函数，验证 gruntz 函数对 ((x + 1)**(1/log(x + 1))) 在 x 趋近无穷大时的极限是否等于 E


def test_intractable():
    assert gruntz(1/gamma(x), x, oo) == 0
    assert gruntz(1/loggamma(x), x, oo) == 0
    assert gruntz(gamma(x)/loggamma(x), x, oo) is oo
    assert gruntz(exp(gamma(x))/gamma(x), x, oo) is oo
    assert gruntz(gamma(x), x, 3) == 2
    assert gruntz(gamma(Rational(1, 7) + 1/x), x, oo) == gamma(Rational(1, 7))
    assert gruntz(log(x**x)/log(gamma(x)), x, oo) == 1
    assert gruntz(log(gamma(gamma(x)))/exp(x), x, oo) is oo

# 测试函数，验证 gruntz 函数对多个数学表达式在 x 趋近无穷大或者某个特定点时的极限情况


def test_aseries_trig():
    assert cancel(gruntz(1/log(atan(x)), x, oo)
           - 1/(log(pi) + log(S.Half))) == 0
    assert gruntz(1/acot(x), x, -oo) is -oo

# 测试函数，验证 gruntz 函数对对数函数和反三角函数在 x 趋近无穷大或负无穷大时的极限


def test_exp_log_series():
    assert gruntz(x/log(log(x*exp(x))), x, oo) is oo

# 测试函数，验证 gruntz 函数对指数和对数级数在 x 趋近无穷大时的极限


def test_issue_3644():
    assert gruntz(((x**7 + x + 1)/(2**x + x**2))**(-1/x), x, oo) == 2

# 测试函数，验证 gruntz 函数对复杂表达式 ((x**7 + x + 1)/(2**x + x**2))**(-1/x) 在 x 趋近无穷大时的极限是否等于 2


def test_issue_6843():
    n = Symbol('n', integer=True, positive=True)
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    assert gruntz(r, x, 1).simplify() == n/2

# 测试函数，验证 gruntz 函数对 r 在 x 趋近 1 时的极限，进一步简化后是否等于 n/2


def test_issue_4190():
    assert gruntz(x - gamma(1/x), x, oo) == S.EulerGamma

# 测试函数，验证 gruntz 函数对 x - gamma(1/x) 在 x 趋近无穷大时的极限是否等于 Euler's constant (S.EulerGamma)


@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c + \
        (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    expr = expr.subs(c, c + 1)
    assert gruntz(expr.subs(c, m), n, oo) == 1
    # fail:
    assert gruntz(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)

# 测试函数，验证 gruntz 函数对复杂表达式在特定条件下的极限情况，其中包括一个预期失败的情况（XFAIL）


def test_issue_4109():
    assert gruntz(1/gamma(x), x, 0) == 0
    assert gruntz(x*gamma(x), x, 0) == 1

# 测试函数，验证 gruntz 函数对 gamma 函数的极限在 x 趋近 0 时的情况


def test_issue_6682():
    assert gruntz(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)

# 测试函数，验证 gruntz 函数对指数积分 Ei 函数的极限在 x 趋近 0 时的情况


def test_issue_7096():
    from sympy.functions import sign
    assert gruntz(x**-pi, x, 0, dir='-') == oo*sign((-1)**(-pi))

# 测试函数，验证 gruntz 函数对 x**-pi 在 x 趋近 0 时，使用负方向（dir='-'）的极限


def test_issue_24210_25885():
    eq = exp(x)/(1+1/x)**x**2
    ans = sqrt(E)
    assert gruntz(eq, x, oo) == ans
    assert gruntz(1/eq, x, oo) == 1/ans

# 测试函数，验证 gruntz 函数对指数和幂次方函数在 x 趋近无穷大时的极限，验证两个相关的表达式
```