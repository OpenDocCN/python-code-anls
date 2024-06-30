# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_transforms.py`

```
# 导入 Sympy 库中的不同积分变换函数和相关错误处理类
from sympy.integrals.transforms import (
    mellin_transform, inverse_mellin_transform,
    fourier_transform, inverse_fourier_transform,
    sine_transform, inverse_sine_transform,
    cosine_transform, inverse_cosine_transform,
    hankel_transform, inverse_hankel_transform,
    FourierTransform, SineTransform, CosineTransform, InverseFourierTransform,
    InverseSineTransform, InverseCosineTransform, IntegralTransformError)

# 导入 Sympy 库中的拉普拉斯变换函数和其逆变换函数
from sympy.integrals.laplace import (
    laplace_transform, inverse_laplace_transform)

# 导入 Sympy 核心功能模块中的函数类和乘法展开函数
from sympy.core.function import Function, expand_mul

# 导入 Sympy 核心模块中的欧拉常数
from sympy.core import EulerGamma

# 导入 Sympy 核心数值模块中的虚数单位、有理数、无穷大和圆周率常量
from sympy.core.numbers import I, Rational, oo, pi

# 导入 Sympy 核心单例模块中的符号常量
from sympy.core.singleton import S

# 导入 Sympy 核心符号模块中的符号和符号集合
from sympy.core.symbol import Symbol, symbols

# 导入 Sympy 组合数学模块中的阶乘函数
from sympy.functions.combinatorial.factorials import factorial

# 导入 Sympy 复数函数模块中的实部和解极坐标
from sympy.functions.elementary.complexes import re, unpolarify

# 导入 Sympy 指数函数模块中的指数、极坐标指数和对数函数
from sympy.functions.elementary.exponential import exp, exp_polar, log

# 导入 Sympy 杂项函数模块中的平方根函数
from sympy.functions.elementary.miscellaneous import sqrt

# 导入 Sympy 三角函数模块中的反正切、余弦、正弦和正切函数
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan

# 导入 Sympy 贝塞尔函数模块中的贝塞尔函数
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely

# 导入 Sympy 特殊函数模块中的海维赛德单位阶跃函数
from sympy.functions.special.delta_functions import Heaviside

# 导入 Sympy 误差函数模块中的误差函数和指数积分函数
from sympy.functions.special.error_functions import erf, expint

# 导入 Sympy 伽玛函数模块中的伽玛函数
from sympy.functions.special.gamma_functions import gamma

# 导入 Sympy 超函模块中的迈耶格函数
from sympy.functions.special.hyper import meijerg

# 导入 Sympy 简化伽玛函数模块中的简化函数
from sympy.simplify.gammasimp import gammasimp

# 导入 Sympy 展开超函数模块中的展开函数
from sympy.simplify.hyperexpand import hyperexpand

# 导入 Sympy 简化三角函数模块中的简化函数
from sympy.simplify.trigsimp import trigsimp

# 导入 Sympy 测试模块中的测试修饰符
from sympy.testing.pytest import XFAIL, slow, skip, raises

# 导入 Sympy 符号模块中的预定义符号
from sympy.abc import x, s, a, b, c, d

# 定义符号变量 nu、beta、rho
nu, beta, rho = symbols('nu beta rho')


def test_undefined_function():
    # 导入 Sympy 积分变换模块中的MellinTransform类
    from sympy.integrals.transforms import MellinTransform
    # 定义函数 f(x)
    f = Function('f')
    # 断言使用 Mellin 变换计算 f(x) 的变换结果等于 MellinTransform 类的实例
    assert mellin_transform(f(x), x, s) == MellinTransform(f(x), x, s)
    # 断言使用 Mellin 变换计算 f(x) + exp(-x) 的变换结果等于 表达式加上 gamma(s + 1)/s
    assert mellin_transform(f(x) + exp(-x), x, s) == \
        (MellinTransform(f(x), x, s) + gamma(s + 1)/s, (0, oo), True)


def test_free_symbols():
    # 定义函数 f(x)
    f = Function('f')
    # 断言 f(x) 的 Mellin 变换的自由符号为 {s}
    assert mellin_transform(f(x), x, s).free_symbols == {s}
    # 断言 f(x)*a 的 Mellin 变换的自由符号为 {s, a}


def test_as_integral():
    # 导入 Sympy 积分模块中的Integral类
    from sympy.integrals.integrals import Integral
    # 定义函数 f(x)
    f = Function('f')
    # 断言 f(x) 的 Mellin 变换重写为积分形式
    assert mellin_transform(f(x), x, s).rewrite('Integral') == \
        Integral(x**(s - 1)*f(x), (x, 0, oo))
    # 断言 f(x) 的 Fourier 变换重写为积分形式
    assert fourier_transform(f(x), x, s).rewrite('Integral') == \
        Integral(f(x)*exp(-2*I*pi*s*x), (x, -oo, oo))
    # 断言 f(x) 的 Laplace 变换重写为积分形式，忽略条件
    assert laplace_transform(f(x), x, s, noconds=True).rewrite('Integral') == \
        Integral(f(x)*exp(-s*x), (x, 0, oo))
    # 断言 f(s) 的 inverse_mellin_transform 变换重写为积分形式
    assert str(2*pi*I*inverse_mellin_transform(f(s), s, x, (a, b)).rewrite('Integral')) \
        == "Integral(f(s)/x**s, (s, _c - oo*I, _c + oo*I))"
    # 断言 f(s) 的 inverse_laplace_transform 变换重写为积分形式
    assert str(2*pi*I*inverse_laplace_transform(f(s), s, x).rewrite('Integral')) == \
        "Integral(f(s)*exp(s*x), (s, _c - oo*I, _c + oo*I))"
    # 断言：验证傅里叶逆变换后的结果是否等于重写为积分形式的表达式
    assert inverse_fourier_transform(f(s), s, x).rewrite('Integral') == \
        Integral(f(s)*exp(2*I*pi*s*x), (s, -oo, oo))
@slow
@XFAIL
# 标记该测试函数为“慢速”和“预期失败”，即预计会失败
def test_mellin_transform_fail():
    # 跳过该测试，并注明原因为“Risch算法耗时太长”
    skip("Risch takes forever.")

    # 将mellin_transform函数赋值给变量MT
    MT = mellin_transform

    # 创建一个正数符号bpos
    bpos = symbols('b', positive=True)

    # 定义一个表达式，包含符号bpos，并注明此处的表达式不适用于负数b，需要进行匹配上的更改
    expr = (sqrt(x + b**2) + b)**a/sqrt(x + b**2)
    # 断言调用MT函数，对表达式中的b进行替换后的结果
    assert MT(expr.subs(b, -bpos), x, s) == \
        ((-1)**(a + 1)*2**(a + 2*s)*bpos**(a + 2*s - 1)*gamma(a + s)
         *gamma(1 - a - 2*s)/gamma(1 - s),
            (-re(a), -re(a)/2 + S.Half), True)

    # 定义另一个表达式，包含符号bpos，并注明此处的表达式不适用于负数b，需要进行匹配上的更改
    expr = (sqrt(x + b**2) + b)**a
    # 断言调用MT函数，对表达式中的b进行替换后的结果
    assert MT(expr.subs(b, -bpos), x, s) == \
        (
            2**(a + 2*s)*a*bpos**(a + 2*s)*gamma(-a - 2*
                   s)*gamma(a + s)/gamma(-s + 1),
            (-re(a), -re(a)/2), True)

    # 测试指数为1的情况
    assert MT(expr.subs({b: -bpos, a: 1}), x, s) == \
        (-bpos**(2*s + 1)*gamma(s)*gamma(-s - S.Half)/(2*sqrt(pi)),
            (-1, Rational(-1, 2)), True)


def test_mellin_transform():
    # 从sympy.functions.elementary.miscellaneous模块导入Max和Min函数，并将mellin_transform函数赋值给变量MT
    from sympy.functions.elementary.miscellaneous import (Max, Min)
    MT = mellin_transform

    # 创建一个正数符号bpos
    bpos = symbols('b', positive=True)

    # 8.4.2
    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    assert MT(x**nu*Heaviside(x - 1), x, s) == \
        (-1/(nu + s), (-oo, -re(nu)), True)
    assert MT(x**nu*Heaviside(1 - x), x, s) == \
        (1/(nu + s), (-re(nu), oo), True)

    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    assert MT((1 - x)**(beta - 1)*Heaviside(1 - x), x, s) == \
        (gamma(beta)*gamma(s)/gamma(beta + s), (0, oo), re(beta) > 0)
    assert MT((x - 1)**(beta - 1)*Heaviside(x - 1), x, s) == \
        (gamma(beta)*gamma(1 - beta - s)/gamma(1 - s),
            (-oo, 1 - re(beta)), re(beta) > 0)

    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    assert MT((1 + x)**(-rho), x, s) == \
        (gamma(s)*gamma(rho - s)/gamma(rho), (0, re(rho)), True)

    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    assert MT(abs(1 - x)**(-rho), x, s) == (
        2*sin(pi*rho/2)*gamma(1 - rho)*
        cos(pi*(s - rho/2))*gamma(s)*gamma(rho-s)/pi,
        (0, re(rho)), re(rho) < 1)
    
    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    mt = MT((1 - x)**(beta - 1)*Heaviside(1 - x)
            + a*(x - 1)**(beta - 1)*Heaviside(x - 1), x, s)
    assert mt[1], mt[2] == ((0, -re(beta) + 1), re(beta) > 0)

    # 断言调用MT函数，对给定的表达式进行Mellin变换，并进行验证
    assert MT((x**a - b**a)/(x - b), x, s)[0] == \
        pi*b**(a + s - 1)*sin(pi*a)/(sin(pi*s)*sin(pi*(a + s)))
    assert MT((x**a - bpos**a)/(x - bpos), x, s) == \
        (pi*bpos**(a + s - 1)*sin(pi*a)/(sin(pi*s)*sin(pi*(a + s))),
            (Max(0, -re(a)), Min(1, 1 - re(a))), True)

    # 定义一个表达式，包含符号bpos，并注明此处的表达式不适用于负数b，需要进行匹配上的更改
    expr = (sqrt(x + b**2) + b)**a
    # 断言调用MT函数，对表达式中的b进行替换后的结果
    assert MT(expr.subs(b, bpos), x, s) == \
        (-a*(2*bpos)**(a + 2*s)*gamma(s)*gamma(-a - 2*s)/gamma(-a - s + 1),
            (0, -re(a)/2), True)

    # 定义一个表达式，包含符号bpos，并注明此处的表达式不适用于负数b，需要进行匹配上的更改
    expr = (sqrt(x + b**2) + b)**a/sqrt(x + b**2)
    # 断言调用MT函数，对表达式中的b进行替换后的结果
    assert MT(expr.subs(b, bpos), x, s) == \
        (2**(a + 2*s)*bpos**(a + 2*s - 1)*gamma(s)
                                         *gamma(1 - a - 2*s)/gamma(1 - a - s),
            (0, -re(a)/2 + S.Half), True)

    # 8.4.2
    # 断言调用MT函数，对给定的指数函数进行Mellin变换，并进行验证
    assert MT(exp(-x), x, s) == (gamma(s), (0, oo), True)
    # 断言验证 MT 函数对于给定表达式和变量的变换结果是否正确
    assert MT(exp(-1/x), x, s) == (gamma(-s), (-oo, 0), True)

    # 8.4.5 验证以下表达式的 MT 变换结果
    assert MT(log(x)**4*Heaviside(1 - x), x, s) == (24/s**5, (0, oo), True)
    assert MT(log(x)**3*Heaviside(x - 1), x, s) == (6/s**4, (-oo, 0), True)
    assert MT(log(x + 1), x, s) == (pi/(s*sin(pi*s)), (-1, 0), True)
    assert MT(log(1/x + 1), x, s) == (pi/(s*sin(pi*s)), (0, 1), True)
    assert MT(log(abs(1 - x)), x, s) == (pi/(s*tan(pi*s)), (-1, 0), True)
    assert MT(log(abs(1 - 1/x)), x, s) == (pi/(s*tan(pi*s)), (0, 1), True)

    # 8.4.14 验证以下表达式的 MT 变换结果
    assert MT(erf(sqrt(x)), x, s) == \
        (-gamma(s + S.Half)/(sqrt(pi)*s), (Rational(-1, 2), 0), True)
def test_mellin_transform2():
    MT = mellin_transform  # 将 mellin_transform 函数赋值给变量 MT

    # TODO we cannot currently do these (needs summation of 3F2(-1))
    #      this also implies that they cannot be written as a single g-function
    #      (although this is possible)
    mt = MT(log(x)/(x + 1), x, s)
    # 断言测试 Mellin 变换后结果的性质
    assert mt[1:] == ((0, 1), True)
    # 断言 hyperexpand 函数应用在 mt[0] 上不包含 meijerg 函数
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)

    mt = MT(log(x)**2/(x + 1), x, s)
    assert mt[1:] == ((0, 1), True)
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)

    mt = MT(log(x)/(x + 1)**2, x, s)
    assert mt[1:] == ((0, 2), True)
    assert not hyperexpand(mt[0], allow_hyper=True).has(meijerg)


@slow
def test_mellin_transform_bessel():
    from sympy.functions.elementary.miscellaneous import Max
    MT = mellin_transform  # 将 mellin_transform 函数赋值给变量 MT

    # 8.4.19
    assert MT(besselj(a, 2*sqrt(x)), x, s) == \
        (gamma(a/2 + s)/gamma(a/2 - s + 1), (-re(a)/2, Rational(3, 4)), True)
    assert MT(sin(sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (2**a*gamma(-2*s + S.Half)*gamma(a/2 + s + S.Half)/(
        gamma(-a/2 - s + 1)*gamma(a - 2*s + 1)), (
        -re(a)/2 - S.Half, Rational(1, 4)), True)
    assert MT(cos(sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (2**a*gamma(a/2 + s)*gamma(-2*s + S.Half)/(
        gamma(-a/2 - s + S.Half)*gamma(a - 2*s + 1)), (
        -re(a)/2, Rational(1, 4)), True)
    assert MT(besselj(a, sqrt(x))**2, x, s) == \
        (gamma(a + s)*gamma(S.Half - s)
         / (sqrt(pi)*gamma(1 - s)*gamma(1 + a - s)),
            (-re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x))*besselj(-a, sqrt(x)), x, s) == \
        (gamma(s)*gamma(S.Half - s)
         / (sqrt(pi)*gamma(1 - a - s)*gamma(1 + a - s)),
            (0, S.Half), True)
    # NOTE: prudnikov gives the strip below as (1/2 - re(a), 1). As far as
    #       I can see this is wrong (since besselj(z) ~ 1/sqrt(z) for z large)
    assert MT(besselj(a - 1, sqrt(x))*besselj(a, sqrt(x)), x, s) == \
        (gamma(1 - s)*gamma(a + s - S.Half)
         / (sqrt(pi)*gamma(Rational(3, 2) - s)*gamma(a - s + S.Half)),
            (S.Half - re(a), S.Half), True)
    assert MT(besselj(a, sqrt(x))*besselj(b, sqrt(x)), x, s) == \
        (4**s*gamma(1 - 2*s)*gamma((a + b)/2 + s)
         / (gamma(1 - s + (b - a)/2)*gamma(1 - s + (a - b)/2)
            *gamma( 1 - s + (a + b)/2)),
            (-(re(a) + re(b))/2, S.Half), True)
    assert MT(besselj(a, sqrt(x))**2 + besselj(-a, sqrt(x))**2, x, s)[1:] == \
        ((Max(re(a), -re(a)), S.Half), True)

    # Section 8.4.20
    assert MT(bessely(a, 2*sqrt(x)), x, s) == \
        (-cos(pi*(a/2 - s))*gamma(s - a/2)*gamma(s + a/2)/pi,
            (Max(-re(a)/2, re(a)/2), Rational(3, 4)), True)
    assert MT(sin(sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-4**s*sin(pi*(a/2 - s))*gamma(S.Half - 2*s)
         * gamma((1 - a)/2 + s)*gamma((1 + a)/2 + s)
         / (sqrt(pi)*gamma(1 - s - a/2)*gamma(1 - s + a/2)),
            (Max(-(re(a) + 1)/2, (re(a) - 1)/2), Rational(1, 4)), True)
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(cos(sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-4**s*cos(pi*(a/2 - s))*gamma(s - a/2)*gamma(s + a/2)*gamma(S.Half - 2*s)
         / (sqrt(pi)*gamma(S.Half - s - a/2)*gamma(S.Half - s + a/2)),
            (Max(-re(a)/2, re(a)/2), Rational(1, 4)), True)
    
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besselj(a, sqrt(x))*bessely(a, sqrt(x)), x, s) == \
        (-cos(pi*s)*gamma(s)*gamma(a + s)*gamma(S.Half - s)
         / (pi**S('3/2')*gamma(1 + a - s)),
            (Max(-re(a), 0), S.Half), True)
    
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besselj(a, sqrt(x))*bessely(b, sqrt(x)), x, s) == \
        (-4**s*cos(pi*(a/2 - b/2 + s))*gamma(1 - 2*s)
         * gamma(a/2 - b/2 + s)*gamma(a/2 + b/2 + s)
         / (pi*gamma(a/2 - b/2 - s + 1)*gamma(a/2 + b/2 - s + 1)),
            (Max((-re(a) + re(b))/2, (-re(a) - re(b))/2), S.Half), True)
    
    # 注释：说明 bessely(a, sqrt(x))**2 和 bessely(a, sqrt(x))*bessely(b, sqrt(x)) 的复杂性问题
    # 是一个混乱的情况（无论从哪个角度看）
    assert MT(bessely(a, sqrt(x))**2, x, s)[1:] == \
             ((Max(-re(a), 0, re(a)), S.Half), True)

    # 注释：第 8.4.22 节
    # TODO：我们无法处理这些情况（细微的抵消）

    # 注释：第 8.4.23 节
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besselk(a, 2*sqrt(x)), x, s) == \
        (gamma(
         s - a/2)*gamma(s + a/2)/2, (Max(-re(a)/2, re(a)/2), oo), True)
    
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besselj(a, 2*sqrt(2*sqrt(x)))*besselk(
        a, 2*sqrt(2*sqrt(x))), x, s) == (4**(-s)*gamma(2*s)*
        gamma(a/2 + s)/(2*gamma(a/2 - s + 1)), (Max(0, -re(a)/2), oo), True)
    
    # TODO：bessely(a, x)*besselk(a, x) 的处理是一个混乱的情况

    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besseli(a, sqrt(x))*besselk(a, sqrt(x)), x, s) == \
        (gamma(s)*gamma(
        a + s)*gamma(-s + S.Half)/(2*sqrt(pi)*gamma(a - s + 1)),
        (Max(-re(a), 0), S.Half), True)
    
    # 断言：计算 MT 函数对于给定的数学表达式和变量的结果，并进行比较
    assert MT(besseli(b, sqrt(x))*besselk(a, sqrt(x)), x, s) == \
        (2**(2*s - 1)*gamma(-2*s + 1)*gamma(-a/2 + b/2 + s)* \
        gamma(a/2 + b/2 + s)/(gamma(-a/2 + b/2 - s + 1)* \
        gamma(a/2 + b/2 - s + 1)), (Max(-re(a)/2 - re(b)/2, \
        re(a)/2 - re(b)/2), S.Half), True)

    # TODO：besselk 的各种奇怪的乘积是一个混乱的情况

    # 变量赋值：计算 MT 函数对于给定表达式和变量的结果
    mt = MT(exp(-x/2)*besselk(a, x/2), x, s)
    # 变量赋值：简化数学表达式 mt[0] 的伽马函数
    mt0 = gammasimp(trigsimp(gammasimp(mt[0].expand(func=True))))
    # 断言：验证简化后的 mt0 是否等于给定的数学表达式
    assert mt0 == 2*pi**Rational(3, 2)*cos(pi*s)*gamma(S.Half - s)/(
        (cos(2*pi*a) - cos(2*pi*s))*gamma(-a - s + 1)*gamma(a - s + 1))
    # 断言：检查 MT 函数的其它结果是否与给定的数学表达式相符
    assert mt[1:] == ((Max(-re(a), re(a)), oo), True)
    # TODO：目前无法处理 exp(x/2)*besselk(a, x/2) 等表达式
    # TODO：各种奇怪的特殊阶乘积
@slow
# 定义一个测试函数，用于测试指数积分和相关的逆 Mellin 变换
def test_expint():
    # 导入所需的符号和函数模块
    from sympy.functions.elementary.miscellaneous import Max
    from sympy.functions.special.error_functions import Ci, E1, Si
    from sympy.simplify.simplify import simplify

    # 定义负数和极坐标符号
    aneg = Symbol('a', negative=True)
    u = Symbol('u', polar=True)

    # 断言：对 E1 函数进行 Mellin 变换，期望得到 gamma(s)/s
    assert mellin_transform(E1(x), x, s) == (gamma(s)/s, (0, oo), True)
    # 断言：对 gamma(s)/s 进行逆 Mellin 变换，期望得到 E1(x)
    assert inverse_mellin_transform(gamma(s)/s, s, x,
              (0, oo)).rewrite(expint).expand() == E1(x)
    # 断言：对 expint(a, x) 进行 Mellin 变换，期望得到 gamma(s)/(a + s - 1)
    assert mellin_transform(expint(a, x), x, s) == \
        (gamma(s)/(a + s - 1), (Max(1 - re(a), 0), oo), True)
    # XXX IMT has hickups with complicated strips ...
    # 断言：简化复杂条带下的逆 Mellin 变换结果，期望得到 expint(aneg, x)
    assert simplify(unpolarify(
                    inverse_mellin_transform(gamma(s)/(aneg + s - 1), s, x,
                  (1 - aneg, oo)).rewrite(expint).expand(func=True))) == \
        expint(aneg, x)

    # 断言：对 Si(x) 进行 Mellin 变换，期望得到 (-2**s*sqrt(pi)*gamma(s/2 + S.Half)/(2*s*gamma(-s/2 + 1)), (-1, 0), True)
    assert mellin_transform(Si(x), x, s) == \
        (-2**s*sqrt(pi)*gamma(s/2 + S.Half)/(
        2*s*gamma(-s/2 + 1)), (-1, 0), True)
    # 断言：对 (-2**s*sqrt(pi)*gamma((s + 1)/2)/(2*s*gamma(-s/2 + 1)) 进行逆 Mellin 变换，期望得到 Si(x)
    assert inverse_mellin_transform(-2**s*sqrt(pi)*gamma((s + 1)/2)
                                    /(2*s*gamma(-s/2 + 1)), s, x, (-1, 0)) \
        == Si(x)

    # 断言：对 Ci(sqrt(x)) 进行 Mellin 变换，期望得到 (-2**(2*s - 1)*sqrt(pi)*gamma(s)/(s*gamma(-s + S.Half)), (0, 1), True)
    assert mellin_transform(Ci(sqrt(x)), x, s) == \
        (-2**(2*s - 1)*sqrt(pi)*gamma(s)/(s*gamma(-s + S.Half)), (0, 1), True)
    # 断言：对 (-4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S.Half))) 进行逆 Mellin 变换，期望得到 Ci(sqrt(u))
    assert inverse_mellin_transform(
        -4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S.Half)),
        s, u, (0, 1)).expand() == Ci(sqrt(u))


@slow
# 定义测试逆 Mellin 变换的函数
def test_inverse_mellin_transform():
    # 导入所需的符号和函数模块
    from sympy.core.function import expand
    from sympy.functions.elementary.miscellaneous import (Max, Min)
    from sympy.functions.elementary.trigonometric import cot
    from sympy.simplify.powsimp import powsimp
    from sympy.simplify.simplify import simplify
    # 定义逆 Mellin 变换的缩写
    IMT = inverse_mellin_transform

    # 断言：对 gamma(s) 进行逆 Mellin 变换，期望得到 exp(-x)
    assert IMT(gamma(s), s, x, (0, oo)) == exp(-x)
    # 断言：对 gamma(-s) 进行逆 Mellin 变换，期望得到 exp(-1/x)
    assert IMT(gamma(-s), s, x, (-oo, 0)) == exp(-1/x)
    # 断言：简化 s/(2*s**2 - 2) 的逆 Mellin 变换结果，期望得到 (x**2 + 1)*Heaviside(1 - x)/(4*x)
    assert simplify(IMT(s/(2*s**2 - 2), s, x, (2, oo))) == \
        (x**2 + 1)*Heaviside(1 - x)/(4*x)

    # 测试传递 "None"
    # 断言：对 1/(s**2 - 1) 进行逆 Mellin 变换，期望得到 -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)
    assert IMT(1/(s**2 - 1), s, x, (-1, None)) == \
        -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)
    # 断言：对 1/(s**2 - 1) 进行逆 Mellin 变换，期望得到 -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)
    assert IMT(1/(s**2 - 1), s, x, (None, 1)) == \
        -x*Heaviside(-x + 1)/2 - Heaviside(x - 1)/(2*x)

    # 测试和的展开
    # 断言：对 gamma(s) + gamma(s - 1) 进行逆 Mellin 变换，期望得到 (x + 1)*exp(-x)/x
    assert IMT(gamma(s) + gamma(s - 1), s, x, (1, oo)) == (x + 1)*exp(-x)/x

    # 测试多项式的因式分解
    r = symbols('r', real=True)
    # 断言：对 1/(s**2 + 1) 进行逆 Mellin 变换，并将结果展开，期望得到 sin(r)*Heaviside(1 - exp(-r))
    assert IMT(1/(s**2 + 1), s, exp(-x), (None, oo)
              ).subs(x, r).rewrite(sin).simplify() \
        == sin(r)*Heaviside(1 - exp(-r))

    # 测试乘法替换
    _a, _b = symbols('a b', positive=True)
    # 断言：对 _b**(-s/_a)*factorial(s/_a)/s 进行逆 Mellin 变换，期望得到 exp(-_b*x**_a)
    assert IMT(_b**(-s/_a)*factorial(s/_a)/s, s, x, (0, oo)) == exp(-_b*x**_a)
    # 断言：对 factorial(_a/_b + s/_b)/(_a + s) 进行逆 Mellin 变换，期望得到 x**_a*exp(-x**_b)
    assert IMT(factorial(_a/_b + s/_b)/(_a + s), s, x, (-_a, oo)) == x**_a*exp(-x**_b)

    # 定义简化幂函数的函数
    def simp_pows(expr):
        return simplify(powsimp(expand_mul(expr, deep=False), force=True)).replace(exp_polar, exp)

    # 现在测试所有上面测试过的直接变换的逆变换

    # Section 8.4.2
    # 定义符号 nu，其为实数
    nu = symbols('nu', real=True)
    # 第一个断言：计算 IMT 函数在给定参数下的结果，断言结果与 x**nu*Heaviside(x - 1)相等
    assert IMT(-1/(nu + s), s, x, (-oo, None)) == x**nu*Heaviside(x - 1)
    
    # 第二个断言：计算 IMT 函数在给定参数下的结果，断言结果与 x**nu*Heaviside(1 - x)相等
    assert IMT(1/(nu + s), s, x, (None, oo)) == x**nu*Heaviside(1 - x)
    
    # 第三个断言：简化 IMT 函数的幂运算结果，断言结果与 (1 - x)**(beta - 1)*Heaviside(1 - x)相等
    assert simp_pows(IMT(gamma(beta)*gamma(s)/gamma(s + beta), s, x, (0, oo))) \
        == (1 - x)**(beta - 1)*Heaviside(1 - x)
    
    # 第四个断言：简化 IMT 函数的幂运算结果，断言结果与 (x - 1)**(beta - 1)*Heaviside(x - 1)相等
    assert simp_pows(IMT(gamma(beta)*gamma(1 - beta - s)/gamma(1 - s),
                         s, x, (-oo, None))) \
        == (x - 1)**(beta - 1)*Heaviside(x - 1)
    
    # 第五个断言：简化 IMT 函数的幂运算结果，断言结果与 (1/(x + 1))**rho 相等
    assert simp_pows(IMT(gamma(s)*gamma(rho - s)/gamma(rho), s, x, (0, None))) \
        == (1/(x + 1))**rho
    
    # 第六个断言：简化 IMT 函数的幂运算结果，断言结果与 (x**c - d**c)/(x - d) 相等
    assert simp_pows(IMT(d**c*d**(s - 1)*sin(pi*c)
                         *gamma(s)*gamma(s + c)*gamma(1 - s)*gamma(1 - s - c)/pi,
                         s, x, (Max(-re(c), 0), Min(1 - re(c), 1)))) \
        == (x**c - d**c)/(x - d)
    
    # 第七个断言：简化 IMT 函数的幂运算结果，断言结果与 (1 + sqrt(x + 1))**c 相等
    assert simplify(IMT(1/sqrt(pi)*(-c/2)*gamma(s)*gamma((1 - c)/2 - s)
                        *gamma(-c/2 - s)/gamma(1 - c - s),
                        s, x, (0, -re(c)/2))) == (1 + sqrt(x + 1))**c
    
    # 第八个断言：简化 IMT 函数的幂运算结果，断言结果与 b**(a - 1)*(b**2*(sqrt(1 + x/b**2) + 1)**a + x*(sqrt(1 + x/b**2) + 1)**(a - 1))/(b**2 + x) 相等
    assert simplify(IMT(2**(a + 2*s)*b**(a + 2*s - 1)*gamma(s)*gamma(1 - a - 2*s)
                        /gamma(1 - a - s), s, x, (0, (-re(a) + 1)/2))) == \
        b**(a - 1)*(b**2*(sqrt(1 + x/b**2) + 1)**a + x*(sqrt(1 + x/b**2) + 1)**(a - 1))/(b**2 + x)
    
    # 第九个断言：简化 IMT 函数的幂运算结果，断言结果与 b**c*(sqrt(1 + x/b**2) + 1)**c 相等
    assert simplify(IMT(-2**(c + 2*s)*c*b**(c + 2*s)*gamma(s)*gamma(-c - 2*s)
                        / gamma(-c - s + 1), s, x, (0, -re(c)/2))) == \
        b**c*(sqrt(1 + x/b**2) + 1)**c
    
    # 第十个断言：计算 IMT 函数的结果，断言结果与 log(x)**4*Heaviside(1 - x) 相等
    assert IMT(24/s**5, s, x, (0, oo)) == log(x)**4*Heaviside(1 - x)
    
    # 第十一个断言：扩展 IMT 函数的结果，断言结果与 log(x)**3*Heaviside(x - 1) 相等
    assert expand(IMT(6/s**4, s, x, (-oo, 0)), force=True) == \
        log(x)**3*Heaviside(x - 1)
    
    # 第十二个断言：计算 IMT 函数的结果，断言结果与 log(x + 1) 相等
    assert IMT(pi/(s*sin(pi*s)), s, x, (-1, 0)) == log(x + 1)
    
    # 第十三个断言：计算 IMT 函数的结果，断言结果与 log(x**2 + 1) 相等
    assert IMT(pi/(s*sin(pi*s/2)), s, x, (-2, 0)) == log(x**2 + 1)
    
    # 第十四个断言：计算 IMT 函数的结果，断言结果与 log(sqrt(x) + 1) 相等
    assert IMT(pi/(s*sin(2*pi*s)), s, x, (Rational(-1, 2), 0)) == log(sqrt(x) + 1)
    
    # 第十五个断言：计算 IMT 函数的结果，断言结果与 log(1 + 1/x) 相等
    assert IMT(pi/(s*sin(pi*s)), s, x, (0, 1)) == log(1 + 1/x)
    
    # 定义 mysimp 函数，对给定表达式进行简化
    def mysimp(expr):
        from sympy.core.function import expand
        from sympy.simplify.powsimp import powsimp
        from sympy.simplify.simplify import logcombine
        return expand(
            powsimp(logcombine(expr, force=True), force=True, deep=True),
            force=True).replace(exp_polar, exp)
    
    # 第十六个断言：断言 mysimp 函数对给定表达式的简化结果符合预期列表中的一种
    assert mysimp(mysimp(IMT(pi/(s*tan(pi*s)), s, x, (-1, 0)))) in [
        log(1 - x)*Heaviside(1 - x) + log(x - 1)*Heaviside(x - 1),
        log(x)*Heaviside(x - 1) + log(1 - 1/x)*Heaviside(x - 1) + log(-x + 1)*Heaviside(-x + 1)]
    
    # 第十七个断言：断言 mysimp 函数对给定表达式的简化结果符合预期列表中的一种
    assert mysimp(IMT(pi*cot(pi*s)/s, s, x, (0, 1))) in [
        log(1/x - 1)*Heaviside(1 - x) + log(1 - 1/x)*Heaviside(x - 1),
        -log(x)*Heaviside(-x + 1) + log(1 - 1/x)*Heaviside(x - 1) + log(-x + 1)*Heaviside(-x + 1)]
    
    # 第十八个断言：计算 IMT 函数的结果，断言结果与 erf(sqrt(x)) 相等
    assert IMT(-gamma(s + S.Half)/(sqrt(pi)*s), s, x, (Rational(-1, 2), 0)) == erf(sqrt(x))
    
    # 第十九个断言：留待补充
    # 使用断言检查表达式是否成立：简化后的特殊积分变换等式等于修正的贝塞尔函数
    assert simplify(IMT(gamma(a/2 + s)/gamma(a/2 - s + 1), s, x, (-re(a)/2, Rational(3, 4)))) \
        == besselj(a, 2*sqrt(x))
    
    # 使用断言检查表达式是否成立：简化后的特殊积分变换等式等于正弦函数乘以修正的贝塞尔函数
    assert simplify(IMT(2**a*gamma(S.Half - 2*s)*gamma(s + (a + 1)/2)
                      / (gamma(1 - s - a/2)*gamma(1 - 2*s + a)),
                      s, x, (-(re(a) + 1)/2, Rational(1, 4)))) == \
        sin(sqrt(x))*besselj(a, sqrt(x))
    
    # 使用断言检查表达式是否成立：简化后的特殊积分变换等式等于余弦函数乘以修正的贝塞尔函数
    assert simplify(IMT(2**a*gamma(a/2 + s)*gamma(S.Half - 2*s)
                      / (gamma(S.Half - s - a/2)*gamma(1 - 2*s + a)),
                      s, x, (-re(a)/2, Rational(1, 4)))) == \
        cos(sqrt(x))*besselj(a, sqrt(x))
    
    # TODO 待办事项：这个结果看起来混乱，但可以简化得很好
    assert simplify(IMT(gamma(a + s)*gamma(S.Half - s)
                      / (sqrt(pi)*gamma(1 - s)*gamma(1 + a - s)),
                      s, x, (-re(a), S.Half))) == \
        besselj(a, sqrt(x))**2
    
    # 使用断言检查表达式是否成立：简化后的特殊积分变换等式等于两个修正贝塞尔函数乘积
    assert simplify(IMT(gamma(s)*gamma(S.Half - s)
                      / (sqrt(pi)*gamma(1 - s - a)*gamma(1 + a - s)),
                      s, x, (0, S.Half))) == \
        besselj(-a, sqrt(x))*besselj(a, sqrt(x))
    
    # 使用断言检查表达式是否成立：简化后的特殊积分变换等式等于两个修正贝塞尔函数乘积
    assert simplify(IMT(4**s*gamma(-2*s + 1)*gamma(a/2 + b/2 + s)
                      / (gamma(-a/2 + b/2 - s + 1)*gamma(a/2 - b/2 - s + 1)
                         *gamma(a/2 + b/2 - s + 1)),
                      s, x, (-(re(a) + re(b))/2, S.Half))) == \
        besselj(a, sqrt(x))*besselj(b, sqrt(x))
    
    # Section 8.4.20
    # TODO 待办事项：这可以进一步简化！
    assert simplify(IMT(-2**(2*s)*cos(pi*a/2 - pi*b/2 + pi*s)*gamma(-2*s + 1) *
                    gamma(a/2 - b/2 + s)*gamma(a/2 + b/2 + s) /
                    (pi*gamma(a/2 - b/2 - s + 1)*gamma(a/2 + b/2 - s + 1)),
                    s, x,
                    (Max(-re(a)/2 - re(b)/2, -re(a)/2 + re(b)/2), S.Half))) == \
                    besselj(a, sqrt(x))*-(besselj(-b, sqrt(x)) -
                    besselj(b, sqrt(x))*cos(pi*b))/sin(pi*b)
    
    # TODO 更多待办事项

    # 用于测试覆盖率的断言：特殊积分变换等式等于表达式的平方根除以 x+1 的平方根
    assert IMT(pi/cos(pi*s), s, x, (0, S.Half)) == sqrt(x)/(x + 1)
def test_fourier_transform():
    # 导入需要的函数和类
    from sympy.core.function import (expand, expand_complex, expand_trig)
    from sympy.polys.polytools import factor
    from sympy.simplify.simplify import simplify
    # 导入 Fourier 变换和逆 Fourier 变换函数并赋值给 FT 和 IFT
    FT = fourier_transform
    IFT = inverse_fourier_transform

    def simp(x):
        # 简化函数，依次应用复杂化简、三角函数展开和复数展开
        return simplify(expand_trig(expand_complex(expand(x))))

    def sinc(x):
        # 定义 sinc 函数
        return sin(pi*x)/(pi*x)
    
    # 定义符号变量 k 和函数 f
    k = symbols('k', real=True)
    f = Function("f")

    # TODO for this to work with real a, need to expand abs(a*x) to abs(a)*abs(x)
    # 定义正数符号变量 a 和 b
    a = symbols('a', positive=True)
    b = symbols('b', positive=True)

    # 定义正数符号变量 posk
    posk = symbols('posk', positive=True)

    # Test unevaluated form
    # 测试 Fourier 变换和逆 Fourier 变换的未求值形式是否相等
    assert fourier_transform(f(x), x, k) == FourierTransform(f(x), x, k)
    assert inverse_fourier_transform(
        f(k), k, x) == InverseFourierTransform(f(k), k, x)

    # basic examples from wikipedia
    # 施氏阶跃函数的 Fourier 变换示例
    assert simp(FT(Heaviside(1 - abs(2*a*x)), x, k)) == sinc(k/a)/a
    assert simp(FT(Heaviside(1 - abs(a*x))*(1 - abs(a*x)), x, k)) == sinc(k/a)**2/a

    # factor 函数的使用示例
    assert factor(FT(exp(-a*x)*Heaviside(x), x, k), extension=I) == \
        1/(a + 2*pi*I*k)
    
    # NOTE: the ift comes out in pieces
    # 逆 Fourier 变换结果的注意事项
    assert IFT(1/(a + 2*pi*I*x), x, posk,
            noconds=False) == (exp(-a*posk), True)
    assert IFT(1/(a + 2*pi*I*x), x, -posk,
            noconds=False) == (0, True)
    assert IFT(1/(a + 2*pi*I*x), x, symbols('k', negative=True),
            noconds=False) == (0, True)

    # factor 函数的使用示例
    assert factor(FT(x*exp(-a*x)*Heaviside(x), x, k), extension=I) == \
        1/(a + 2*pi*I*k)**2
    
    # Fourier 变换的使用示例
    assert FT(exp(-a*x)*sin(b*x)*Heaviside(x), x, k) == \
        b/(b**2 + (a + 2*I*pi*k)**2)

    assert FT(exp(-a*x**2), x, k) == sqrt(pi)*exp(-pi**2*k**2/a)/sqrt(a)
    assert IFT(sqrt(pi/a)*exp(-(pi*k)**2/a), k, x) == exp(-a*x**2)
    assert FT(exp(-a*abs(x)), x, k) == 2*a/(a**2 + 4*pi**2*k**2)

    # test besselj(n, x), n an integer > 0 actually can be done...
    # 测试贝塞尔函数变换
    # are there other common transforms (no distributions!)?
    # 是否还有其他常见的变换（无分布函数）？

def test_sine_transform():
    # 定义符号变量 t 和 w
    t = symbols("t")
    w = symbols("w")
    a = symbols("a")
    f = Function("f")

    # Test unevaluated form
    # 测试正弦变换和逆正弦变换的未求值形式是否相等
    assert sine_transform(f(t), t, w) == SineTransform(f(t), t, w)
    assert inverse_sine_transform(
        f(w), w, t) == InverseSineTransform(f(w), w, t)

    # sine_transform 函数的使用示例
    assert sine_transform(1/sqrt(t), t, w) == 1/sqrt(w)
    assert inverse_sine_transform(1/sqrt(w), w, t) == 1/sqrt(t)

    assert sine_transform((1/sqrt(t))**3, t, w) == 2*sqrt(w)

    assert sine_transform(t**(-a), t, w) == 2**(
        -a + S.Half)*w**(a - 1)*gamma(-a/2 + 1)/gamma((a + 1)/2)
    assert inverse_sine_transform(2**(-a + S(
        1)/2)*w**(a - 1)*gamma(-a/2 + 1)/gamma(a/2 + S.Half), w, t) == t**(-a)

    assert sine_transform(
        exp(-a*t), t, w) == sqrt(2)*w/(sqrt(pi)*(a**2 + w**2))
    # 确保逆正弦变换公式的正确性
    assert inverse_sine_transform(
        sqrt(2)*w/(sqrt(pi)*(a**2 + w**2)), w, t) == exp(-a*t)
    
    # 确保正弦变换公式的正确性
    assert sine_transform(
        log(t)/t, t, w) == sqrt(2)*sqrt(pi)*-(log(w**2) + 2*EulerGamma)/4
    
    # 确保正弦变换公式的正确性
    assert sine_transform(
        t*exp(-a*t**2), t, w) == sqrt(2)*w*exp(-w**2/(4*a))/(4*a**Rational(3, 2))
    
    # 确保逆正弦变换公式的正确性
    assert inverse_sine_transform(
        sqrt(2)*w*exp(-w**2/(4*a))/(4*a**Rational(3, 2)), w, t) == t*exp(-a*t**2)
def test_cosine_transform():
    from sympy.functions.special.error_functions import (Ci, Si)

    t = symbols("t")  # 符号变量 t
    w = symbols("w")  # 符号变量 w
    a = symbols("a")  # 符号变量 a
    f = Function("f")  # 符号函数 f

    # 测试未简化形式
    assert cosine_transform(f(t), t, w) == CosineTransform(f(t), t, w)  # 断言余弦变换
    assert inverse_cosine_transform(f(w), w, t) == InverseCosineTransform(f(w), w, t)  # 断言反余弦变换

    assert cosine_transform(1/sqrt(t), t, w) == 1/sqrt(w)  # 断言余弦变换的结果
    assert inverse_cosine_transform(1/sqrt(w), w, t) == 1/sqrt(t)  # 断言反余弦变换的结果

    assert cosine_transform(1/(a**2 + t**2), t, w) == sqrt(2)*sqrt(pi)*exp(-a*w)/(2*a)  # 断言余弦变换的结果

    assert cosine_transform(t**(-a), t, w) == 2**(-a + S.Half)*w**(a - 1)*gamma((-a + 1)/2)/gamma(a/2)  # 断言余弦变换的结果
    assert inverse_cosine_transform(2**(-a + S(1)/2)*w**(a - 1)*gamma(-a/2 + S.Half)/gamma(a/2), w, t) == t**(-a)  # 断言反余弦变换的结果

    assert cosine_transform(exp(-a*t), t, w) == sqrt(2)*a/(sqrt(pi)*(a**2 + w**2))  # 断言余弦变换的结果
    assert inverse_cosine_transform(sqrt(2)*a/(sqrt(pi)*(a**2 + w**2)), w, t) == exp(-a*t)  # 断言反余弦变换的结果

    assert cosine_transform(exp(-a*sqrt(t))*cos(a*sqrt(t)), t, w) == a*exp(-a**2/(2*w))/(2*w**Rational(3, 2))  # 断言余弦变换的结果

    assert cosine_transform(1/(a + t), t, w) == sqrt(2)*((-2*Si(a*w) + pi)*sin(a*w)/2 - cos(a*w)*Ci(a*w))/sqrt(pi)  # 断言余弦变换的结果
    assert inverse_cosine_transform(sqrt(2)*meijerg(((S.Half, 0), ()), ((S.Half, 0, 0), (S.Half,)), a**2*w**2/4)/(2*pi), w, t) == 1/(a + t)  # 断言反余弦变换的结果

    assert cosine_transform(1/sqrt(a**2 + t**2), t, w) == sqrt(2)*meijerg(((S.Half,), ()), ((0, 0), (S.Half,)), a**2*w**2/4)/(2*sqrt(pi))  # 断言余弦变换的结果
    assert inverse_cosine_transform(sqrt(2)*meijerg(((S.Half,), ()), ((0, 0), (S.Half,)), a**2*w**2/4)/(2*sqrt(pi)), w, t) == 1/(t*sqrt(a**2/t**2 + 1))  # 断言反余弦变换的结果


def test_hankel_transform():
    r = Symbol("r")  # 符号变量 r
    k = Symbol("k")  # 符号变量 k
    nu = Symbol("nu")  # 符号变量 nu
    m = Symbol("m")  # 符号变量 m
    a = symbols("a")  # 符号变量 a

    assert hankel_transform(1/r, r, k, 0) == 1/k  # 断言 Hankel 变换的结果
    assert inverse_hankel_transform(1/k, k, r, 0) == 1/r  # 断言反 Hankel 变换的结果

    assert hankel_transform(1/r**m, r, k, 0) == 2**(-m + 1)*k**(m - 2)*gamma(-m/2 + 1)/gamma(m/2)  # 断言 Hankel 变换的结果
    assert inverse_hankel_transform(2**(-m + 1)*k**(m - 2)*gamma(-m/2 + 1)/gamma(m/2), k, r, 0) == r**(-m)  # 断言反 Hankel 变换的结果

    assert hankel_transform(1/r**m, r, k, nu) == 2*2**(-m)*k**(m - 2)*gamma(-m/2 + nu/2 + 1)/gamma(m/2 + nu/2)  # 断言 Hankel 变换的结果
    assert inverse_hankel_transform(2**(-m + 1)*k**(m - 2)*gamma(-m/2 + nu/2 + 1)/gamma(m/2 + nu/2), k, r, nu) == r**(-m)  # 断言反 Hankel 变换的结果

    assert hankel_transform(r**nu*exp(-a*r), r, k, nu) == 2**(nu + 1)*a*k**(-nu - 3)*(a**2/k**2 + 1)**(-nu - S(3)/2)*gamma(nu + Rational(3, 2))/sqrt(pi)  # 断言 Hankel 变换的结果
    assert inverse_hankel_transform(2**(nu + 1)*a*k**(-nu - 3)*(a**2/k**2 + 1)**(-nu - Rational(3, 2))*gamma(nu + Rational(3, 2))/sqrt(pi), k, r, nu) == r**nu*exp(-a*r)  # 断言反 Hankel 变换的结果


def test_issue_7181():
    assert mellin_transform(1/(1 - x), x, s) != None  # 断言 Mellin 变换的结果


def test_issue_8882():
    # 这是原始测试。
    pass  # 空语句，什么也不做
    # 导入 sympy 库中的差分、积分和积分求解函数
    from sympy import diff, Integral, integrate
    # 定义符号变量 r
    r = Symbol('r')
    # 定义波函数 psi
    psi = 1/r*sin(r)*exp(-(a0*r))
    # 定义哈密顿量 h
    h = -1/2*diff(psi, r, r) - 1/r*psi
    # 定义波函数 psi 和哈密顿量 h 的乘积 f
    f = 4*pi*psi*h*r**2
    # 断言积分结果是否包含 Integral，使用 meijerg 算法进行检查
    assert integrate(f, (r, -oo, 3), meijerg=True).has(Integral) == True

    # 为了节省时间，仅包含关键部分
    # 定义变量 F，包含复杂的数学表达式
    F = -a**(-s + 1)*(4 + 1/a**2)**(-s/2)*sqrt(1/a**2)*exp(-s*I*pi)* \
        sin(s*atan(sqrt(1/a**2)/2))*gamma(s)
    # 断言对逆 Mellin 变换的应用是否抛出 IntegralTransformError 异常
    raises(IntegralTransformError, lambda:
        inverse_mellin_transform(F, s, x, (-1, oo),
        **{'as_meijerg': True, 'needeval': True}))
# 定义名为 test_issue_12591 的函数，用于测试问题编号为 12591 的情况
def test_issue_12591():
    # 创建符号变量 x 和 y，并指定它们为实数
    x, y = symbols("x y", real=True)
    # 断言 Fourier 变换应用于 exp(x) 关于 x 的结果等于 FourierTransform 类的实例化对象，处理 exp(x) 关于 x 和 y 的变换
    assert fourier_transform(exp(x), x, y) == FourierTransform(exp(x), x, y)
```