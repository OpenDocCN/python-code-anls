# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_bessel.py`

```
from itertools import product  # 导入itertools模块中的product函数，用于计算笛卡尔积

from sympy.concrete.summations import Sum  # 导入Sum类，用于表示和式
from sympy.core.function import (diff, expand_func)  # 导入diff和expand_func函数，用于求导和展开函数
from sympy.core.numbers import (I, Rational, oo, pi)  # 导入I（虚数单位）、Rational（有理数类）、oo（无穷大）、pi（圆周率）
from sympy.core.singleton import S  # 导入S单例，用于表示单例符号
from sympy.core.symbol import (Symbol, symbols)  # 导入Symbol类和symbols函数，用于符号操作
from sympy.functions.elementary.complexes import (conjugate, polar_lift)  # 导入复数操作函数
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import (cosh, sinh)  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, hn1, hn2, jn, jn_zeros, yn)  # 导入贝塞尔函数及相关函数
from sympy.functions.special.gamma_functions import (gamma, uppergamma)  # 导入伽玛函数和上不完全伽玛函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.integrals.integrals import Integral  # 导入积分函数
from sympy.series.order import O  # 导入O记号函数，用于表示阶数
from sympy.series.series import series  # 导入级数展开函数
from sympy.functions.special.bessel import (airyai, airybi,
                                            airyaiprime, airybiprime, marcumq)  # 导入Airy函数和Marcum Q函数
from sympy.core.random import (random_complex_number as randcplx,
                               verify_numerically as tn,
                               test_derivative_numerically as td,
                               _randint)  # 导入随机数生成和数值验证函数
from sympy.simplify import besselsimp  # 导入简化贝塞尔函数的函数
from sympy.testing.pytest import raises, slow  # 导入用于测试的函数和修饰器

from sympy.abc import z, n, k, x  # 导入符号z, n, k, x

randint = _randint()  # 使用_randint函数生成一个随机整数


def test_bessel_rand():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2]:
        assert td(f(randcplx(), z), z)  # 对每个贝塞尔函数应用随机复数和符号z进行数值导数测试

    for f in [jn, yn, hn1, hn2]:
        assert td(f(randint(-10, 10), z), z)  # 对每个特殊函数应用在随机整数和符号z上进行数值导数测试


def test_bessel_twoinputs():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, jn, yn]:
        raises(TypeError, lambda: f(1))  # 测试每个贝塞尔函数和特殊函数在只有一个输入参数时是否引发TypeError异常
        raises(TypeError, lambda: f(1, 2, 3))  # 测试每个贝塞尔函数和特殊函数在有三个输入参数时是否引发TypeError异常


def test_besselj_leading_term():
    assert besselj(0, x).as_leading_term(x) == 1  # 验证贝塞尔函数J_0(x)在x的主导项为1
    assert besselj(1, sin(x)).as_leading_term(x) == x/2  # 验证贝塞尔函数J_1(sin(x))在x的主导项为x/2
    assert besselj(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)  # 验证贝塞尔函数J_1(2*sqrt(x))在x的主导项为sqrt(x)

    # https://github.com/sympy/sympy/issues/21701
    assert (besselj(z, x)/x**z).as_leading_term(x) == 1/(2**z*gamma(z + 1))  # 验证贝塞尔函数J_z(x)/x^z在x的主导项为1/(2^z*gamma(z + 1))


def test_bessely_leading_term():
    assert bessely(0, x).as_leading_term(x) == (2*log(x) - 2*log(2) + 2*S.EulerGamma)/pi  # 验证贝塞尔函数Y_0(x)在x的主导项
    assert bessely(1, sin(x)).as_leading_term(x) == -2/(pi*x)  # 验证贝塞尔函数Y_1(sin(x))在x的主导项
    assert bessely(1, 2*sqrt(x)).as_leading_term(x) == -1/(pi*sqrt(x))  # 验证贝塞尔函数Y_1(2*sqrt(x))在x的主导项


def test_besseli_leading_term():
    assert besseli(0, x).as_leading_term(x) == 1  # 验证修正贝塞尔函数I_0(x)在x的主导项为1
    assert besseli(1, sin(x)).as_leading_term(x) == x/2  # 验证修正贝塞尔函数I_1(sin(x))在x的主导项为x/2
    assert besseli(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)  # 验证修正贝塞尔函数I_1(2*sqrt(x))在x的主导项为sqrt(x)


def test_besselk_leading_term():
    assert besselk(0, x).as_leading_term(x) == -log(x) - S.EulerGamma + log(2)  # 验证修正贝塞尔函数K_0(x)在x的主导项
    assert besselk(1, sin(x)).as_leading_term(x) == 1/x  # 验证修正贝塞尔函数K_1(sin(x))在x的主导项
    assert besselk(1, 2*sqrt(x)).as_leading_term(x) == 1/(2*sqrt(x))  # 验证修正贝塞尔函数K_1(2*sqrt(x))在x的主导项


def test_besselj_series():
    # 断言：验证贝塞尔函数的零阶特定参数的级数展开结果是否符合预期
    assert besselj(0, x).series(x) == 1 - x**2/4 + x**4/64 + O(x**6)
    
    # 断言：验证贝塞尔函数的零阶在参数 x 的 1.1 次幂下的级数展开结果是否符合预期
    assert besselj(0, x**(1.1)).series(x) == 1 + x**4.4/64 - x**2.2/4 + O(x**6)
    
    # 断言：验证贝塞尔函数的零阶在参数 x**2 + x 的情况下的级数展开结果是否符合预期
    assert besselj(0, x**2 + x).series(x) == 1 - x**2/4 - x**3/2 \
        - 15*x**4/64 + x**5/16 + O(x**6)
    
    # 断言：验证贝塞尔函数的零阶在参数 sqrt(x) + x 的情况下的级数展开结果是否符合预期（截取到 n=4）
    assert besselj(0, sqrt(x) + x).series(x, n=4) == 1 - x/4 - 15*x**2/64 \
        + 215*x**3/2304 - x**Rational(3, 2)/2 + x**Rational(5, 2)/16 \
        + 23*x**Rational(7, 2)/384 + O(x**4)
    
    # 断言：验证贝塞尔函数的零阶在参数 x/(1 - x) 的情况下的级数展开结果是否符合预期
    assert besselj(0, x/(1 - x)).series(x) == 1 - x**2/4 - x**3/2 - 47*x**4/64 \
        - 15*x**5/16 + O(x**6)
    
    # 断言：验证贝塞尔函数的零阶在参数 log(1 + x) 的情况下的级数展开结果是否符合预期
    assert besselj(0, log(1 + x)).series(x) == 1 - x**2/4 + x**3/4 \
        - 41*x**4/192 + 17*x**5/96 + O(x**6)
    
    # 断言：验证贝塞尔函数的一阶在参数 sin(x) 的情况下的级数展开结果是否符合预期
    assert besselj(1, sin(x)).series(x) == x/2 - 7*x**3/48 + 73*x**5/1920 + O(x**6)
    
    # 断言：验证贝塞尔函数的一阶在参数 2*sqrt(x) 的情况下的级数展开结果是否符合预期
    assert besselj(1, 2*sqrt(x)).series(x) == sqrt(x) - x**Rational(3, 2)/2 \
        + x**Rational(5, 2)/12 - x**Rational(7, 2)/144 + x**Rational(9, 2)/2880 \
        - x**Rational(11, 2)/86400 + O(x**6)
    
    # 断言：验证贝塞尔函数的负二阶在参数 sin(x) 的情况下的级数展开结果是否与贝塞尔函数二阶在同参数情况下的级数展开结果相同
    assert besselj(-2, sin(x)).series(x, n=4) == besselj(2, sin(x)).series(x, n=4)
# 定义函数 test_bessely_series，用于测试贝塞尔函数 Y 的级数展开
def test_bessely_series():
    # 计算常数 const，这里涉及数学常数和对数函数
    const = 2*S.EulerGamma/pi - 2*log(2)/pi + 2*log(x)/pi
    # 断言 Y_0(x) 的级数展开是否等于预期表达式
    assert bessely(0, x).series(x, n=4) == const + x**2*(-log(x)/(2*pi)\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    # 断言 Y_1(x) 的级数展开是否等于预期表达式
    assert bessely(1, x).series(x, n=4) == -2/(pi*x) + x*(log(x)/pi - log(2)/pi - \
        (1 - 2*S.EulerGamma)/(2*pi)) + x**3*(-log(x)/(8*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(16*pi) + log(2)/(8*pi)) + O(x**4*log(x))
    # 断言 Y_2(x) 的级数展开是否等于预期表达式
    assert bessely(2, x).series(x, n=4) == -4/(pi*x**2) - 1/pi + x**2*(log(x)/(4*pi) - \
        log(2)/(4*pi) - (S(3)/2 - 2*S.EulerGamma)/(8*pi)) + O(x**4*log(x))
    # 断言 Y_3(x) 的级数展开是否等于预期表达式
    assert bessely(3, x).series(x, n=4) == -16/(pi*x**3) - 2/(pi*x) - \
        x/(4*pi) + x**3*(log(x)/(24*pi) - log(2)/(24*pi) - \
        (S(11)/6 - 2*S.EulerGamma)/(48*pi)) + O(x**4*log(x))
    # 断言 Y_0(x^1.1) 的级数展开是否等于预期表达式
    assert bessely(0, x**(1.1)).series(x, n=4) == 2*S.EulerGamma/pi\
        - 2*log(2)/pi + 2.2*log(x)/pi + x**2.2*(-0.55*log(x)/pi\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    # 断言 Y_0(x^2 + x) 的级数展开是否等于预期表达式
    assert bessely(0, x**2 + x).series(x, n=4) == \
        const - (2 - 2*S.EulerGamma)*(-x**3/(2*pi) - x**2/(4*pi)) + 2*x/pi\
        + x**2*(-log(x)/(2*pi) - 1/pi + log(2)/(2*pi))\
        + x**3*(-log(x)/pi + 1/(6*pi) + log(2)/pi) + O(x**4*log(x))
    # 断言 Y_0(x/(1 - x)) 的级数展开是否等于预期表达式
    assert bessely(0, x/(1 - x)).series(x, n=3) == const\
        + 2*x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 1/pi) + O(x**3*log(x))
    # 断言 Y_0(log(1 + x)) 的级数展开是否等于预期表达式
    assert bessely(0, log(1 + x)).series(x, n=3) == const\
        - x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 5/(12*pi)) + O(x**3*log(x))
    # 断言 Y_1(sin(x)) 的级数展开是否等于预期表达式
    assert bessely(1, sin(x)).series(x, n=4) == -1/(pi*(-x**3/12 + x/2)) - \
        (1 - 2*S.EulerGamma)*(-x**3/12 + x/2)/pi + x*(log(x)/pi - log(2)/pi) + \
        x**3*(-7*log(x)/(24*pi) - 1/(6*pi) + (S(5)/2 - 2*S.EulerGamma)/(16*pi) +
        7*log(2)/(24*pi)) + O(x**4*log(x))
    # 断言 Y_1(2*sqrt(x)) 的级数展开是否等于预期表达式
    assert bessely(1, 2*sqrt(x)).series(x, n=3) == -1/(pi*sqrt(x)) + \
        sqrt(x)*(log(x)/pi - (1 - 2*S.EulerGamma)/pi) + x**(S(3)/2)*(-log(x)/(2*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(2*pi)) + x**(S(5)/2)*(log(x)/(12*pi) - \
        (S(10)/3 - 2*S.EulerGamma)/(12*pi)) + O(x**3*log(x))
    # 断言 Y_{-2}(sin(x)) 的级数展开是否等于预期表达式，这里利用了 Y_{-2}(x) 和 Y_2(x) 的对称性质
    assert bessely(-2, sin(x)).series(x, n=4) == bessely(2, sin(x)).series(x, n=4)


# 定义函数 test_besseli_series，用于测试贝塞尔函数 I 的级数展开
def test_besseli_series():
    # 断言 I_0(x) 的级数展开是否等于预期表达式
    assert besseli(0, x).series(x) == 1 + x**2/4 + x**4/64 + O(x**6)
    # 断言 I_0(x^1.1) 的级数展开是否等于预期表达式
    assert besseli(0, x**(1.1)).series(x) == 1 + x**4.4/64 + x**2.2/4 + O(x**6)
    # 断言 I_0(x^2 + x) 的级数展开是否等于预期表达式
    assert besseli(0, x**2 + x).series(x) == 1 + x**2/4 + x**3/2 + 17*x**4/64 + \
        x**5/16 + O(x**6)
    # 断言 I_0(sqrt(x) + x) 的级数展开是否等于预期表达式
    assert besseli(0, sqrt(x) + x).series(x, n=4) == 1 + x/4 + 17*x**2/64 + \
        217*x**3/2304 + x**(S(3)/2)/2 + x**(S(5)/2)/16 + 25*x**(S(7)/2)/384 + O(x**4)
    # 断言 I_0(x/(1 - x)) 的级数展开是否等于预期表达式
    assert besseli(0, x/(1 - x)).series(x) == 1 + x**2/4 + x**3/2 + 49*x**4/64 + \
        17*x**5/16 + O(x**6)
    # 断言 I_0(log(1 + x)) 的级数展开是否等于预期表达式
    assert besseli(0, log(1 + x)).series(x) == 1 + x**2/4 - x**3/4 + 47*x**4/192 - \
        23*x**5/96 + O(x**6)
    # 断言：调用 besseli 函数计算修正贝塞尔函数 I_n(x) 的泰勒级数展开，验证结果是否与给定表达式相等
    assert besseli(1, sin(x)).series(x) == x/2 - x**3/48 - 47*x**5/1920 + O(x**6)
    
    # 断言：调用 besseli 函数计算修正贝塞尔函数 I_n(x) 的泰勒级数展开，验证结果是否与给定表达式相等
    assert besseli(1, 2*sqrt(x)).series(x) == sqrt(x) + x**(S(3)/2)/2 + x**(S(5)/2)/12 + \
        x**(S(7)/2)/144 + x**(S(9)/2)/2880 + x**(S(11)/2)/86400 + O(x**6)
    
    # 断言：调用 besseli 函数计算修正贝塞尔函数 I_n(x) 的泰勒级数展开，指定展开的阶数为4，验证结果是否与同阶数下 n=2 的修正贝塞尔函数的展开结果相等
    assert besseli(-2, sin(x)).series(x, n=4) == besseli(2, sin(x)).series(x, n=4)
def test_besselk_series():
    # 计算常量，对数值是基于 2 的自然对数减去欧拉常数再减去 x 的自然对数
    const = log(2) - S.EulerGamma - log(x)
    # 断言修正的贝塞尔函数 K 的级数展开的表达式
    assert besselk(0, x).series(x, n=4) == const + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + log(2)/4 + S(1)/4) + O(x**4*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式
    assert besselk(1, x).series(x, n=4) == 1/x + x*(log(x)/2 - log(2)/2 - \
        S(1)/4 + S.EulerGamma/2) + x**3*(log(x)/16 - S(5)/64 - log(2)/16 + \
        S.EulerGamma/16) + O(x**4*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式
    assert besselk(2, x).series(x, n=4) == 2/x**2 - S(1)/2 + x**2*(-log(x)/8 - \
        S.EulerGamma/8 + log(2)/8 + S(3)/32) + O(x**4*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 的指数是 1.1
    assert besselk(0, x**(1.1)).series(x, n=4) == log(2) - S.EulerGamma - \
        1.1*log(x) + x**2.2*(-0.275*log(x) - S.EulerGamma/4 + \
        log(2)/4 + S(1)/4) + O(x**4*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 是一个复杂的表达式
    assert besselk(0, x**2 + x).series(x, n=4) == const + \
        (2 - 2*S.EulerGamma)*(x**3/4 + x**2/8) - x + x**2*(-log(x)/4 + \
        log(2)/4 + S(1)/2) + x**3*(-log(x)/2 - S(7)/12 + log(2)/2) + O(x**4*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 是一个复杂的表达式
    assert besselk(0, x/(1 - x)).series(x, n=3) == const - x + x**2*(-log(x)/4 - \
        S(1)/4 - S.EulerGamma/4 + log(2)/4) + O(x**3*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 是一个复杂的表达式
    assert besselk(0, log(1 + x)).series(x, n=3) == const + x/2 + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + S(1)/24 + log(2)/4) + O(x**3*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 是一个复杂的表达式
    assert besselk(1, 2*sqrt(x)).series(x, n=3) == 1/(2*sqrt(x)) + \
        sqrt(x)*(log(x)/2 - S(1)/2 + S.EulerGamma) + x**(S(3)/2)*(log(x)/4 - S(5)/8 + \
        S.EulerGamma/2) + x**(S(5)/2)*(log(x)/24 - S(5)/36 + S.EulerGamma/12) + O(x**3*log(x))
    # 断言修正的贝塞尔函数 K 的级数展开的表达式，其中 x 是一个复杂的表达式
    assert besselk(-2, sin(x)).series(x, n=4) == besselk(2, sin(x)).series(x, n=4)


def test_diff():
    # 断言贝塞尔函数 J 的导数
    assert besselj(n, z).diff(z) == besselj(n - 1, z)/2 - besselj(n + 1, z)/2
    # 断言贝塞尔函数 Y 的导数
    assert bessely(n, z).diff(z) == bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    # 断言修正的贝塞尔函数 I 的导数
    assert besseli(n, z).diff(z) == besseli(n - 1, z)/2 + besseli(n + 1, z)/2
    # 断言修正的贝塞尔函数 K 的导数
    assert besselk(n, z).diff(z) == -besselk(n - 1, z)/2 - besselk(n + 1, z)/2
    # 断言汉克尔函数 H_1 的导数
    assert hankel1(n, z).diff(z) == hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2
    # 断言汉克尔函数 H_2 的导数
    assert hankel2(n, z).diff(z) == hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2


def test_rewrite():
    # 断言贝塞尔函数 J 的重写为第一类贝塞尔函数
    assert besselj(n, z).rewrite(jn) == sqrt(2*z/pi)*jn(n - S.Half, z)
    # 断言贝塞尔函数 Y 的重写为第二类贝塞尔函数
    assert bessely(n, z).rewrite(yn) == sqrt(2*z/pi)*yn(n - S.Half, z)
    # 断言贝塞尔函数 I 的重写为贝塞尔函数 J
    assert besseli(n, z).rewrite(besselj) == \
        exp(-I*n*pi/2)*besselj(n, polar_lift(I)*z)
    # 断言贝塞尔函数 J 的重写为贝塞尔函数 I
    assert besselj(n, z).rewrite(besseli) == \
        exp(I*n*pi/2)*besseli(n, polar_lift(-I)*z)

    # 定义一个随机复数 nu
    nu = randcplx()

    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besselj(nu, z), besselj(nu, z).rewrite(besseli), z)
    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besselj(nu, z), besselj(nu, z).rewrite(bessely), z)

    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besseli(nu, z), besseli(nu, z).rewrite(besselj), z)
    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besseli(nu, z), besseli(nu, z).rewrite(bessely), z)

    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besselj), z)
    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besseli), z)

    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besselj), z)
    # 断言通过比较两个函数的相等性，使用的是函数 tn
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besseli), z)
    # 使用 assert 语句来检验 besselk 函数是否与其用 bessely 重写后的结果相等
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(bessely), z)

    # 当阶数设置为通用符号 'nu' 时，检查是否触发了重写，以确保不与 jn 函数重写
    assert yn(nu, z) != yn(nu, z).rewrite(jn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(jn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(jn)
    assert jn(nu, z) != jn(nu, z).rewrite(yn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(yn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(yn)

    # 对于球贝塞尔函数（SBFs），当阶数使用通用符号 'nu' 时，不允许相对于 besselj, bessely 重写，
    # 避免不一致性（bessel[jy] 的阶数允许为复数，而 SBFs 只定义在整数阶数上）
    order = nu
    for f in (besselj, bessely):
        assert hn1(order, z) == hn1(order, z).rewrite(f)
        assert hn2(order, z) == hn2(order, z).rewrite(f)

    # 检验使用 besselj 重写后的球贝塞尔函数 jn 的结果是否正确
    assert jn(order, z).rewrite(besselj) == sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(order + S.Half, z)/2
    # 检验使用 bessely 重写后的球贝塞尔函数 jn 的结果是否正确
    assert jn(order, z).rewrite(bessely) == (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-order - S.Half, z)/2

    # 对于整数阶数，允许相对于 bessel[jy] 重写球贝塞尔函数
    N = Symbol('n', integer=True)
    ri = randint(-11, 10)
    for order in (ri, N):
        for f in (besselj, bessely):
            assert yn(order, z) != yn(order, z).rewrite(f)
            assert jn(order, z) != jn(order, z).rewrite(f)
            assert hn1(order, z) != hn1(order, z).rewrite(f)
            assert hn2(order, z) != hn2(order, z).rewrite(f)

    # 使用 product 函数生成组合，检验球贝塞尔函数在不同函数之间重写的结果是否正确
    for func, refunc in product((yn, jn, hn1, hn2),
                                (jn, yn, besselj, bessely)):
        assert tn(func(ri, z), func(ri, z).rewrite(refunc), z)
def test_expand():
    # 测试 expand_func 函数对 besselj(S.Half, z) 的展开结果是否正确
    assert expand_func(besselj(S.Half, z).rewrite(jn)) == \
        sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    # 测试 expand_func 函数对 bessely(S.Half, z) 的展开结果是否正确
    assert expand_func(bessely(S.Half, z).rewrite(yn)) == \
        -sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))

    # XXX: teach sin/cos to work around arguments like
    # x*exp_polar(I*pi*n/2).  Then change besselsimp -> expand_func
    # 测试 besselsimp 函数对 besselj(S.Half, z) 的简化结果是否正确
    assert besselsimp(besselj(S.Half, z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 besselj(Rational(-1, 2), z) 的简化结果是否正确
    assert besselsimp(besselj(Rational(-1, 2), z)) == sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 besselj(Rational(5, 2), z) 的简化结果是否正确
    assert besselsimp(besselj(Rational(5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))
    # 测试 besselsimp 函数对 besselj(Rational(-5, 2), z) 的简化结果是否正确
    assert besselsimp(besselj(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))

    # 测试 besselsimp 函数对 bessely(S.Half, z) 的简化结果是否正确
    assert besselsimp(bessely(S.Half, z)) == \
        -(sqrt(2)*cos(z))/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 bessely(Rational(-1, 2), z) 的简化结果是否正确
    assert besselsimp(bessely(Rational(-1, 2), z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 bessely(Rational(5, 2), z) 的简化结果是否正确
    assert besselsimp(bessely(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))
    # 测试 besselsimp 函数对 bessely(Rational(-5, 2), z) 的简化结果是否正确
    assert besselsimp(bessely(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))

    # 测试 besselsimp 函数对 besseli(S.Half, z) 的简化结果是否正确
    assert besselsimp(besseli(S.Half, z)) == sqrt(2)*sinh(z)/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 besseli(Rational(-1, 2), z) 的简化结果是否正确
    assert besselsimp(besseli(Rational(-1, 2), z)) == \
        sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    # 测试 besselsimp 函数对 besseli(Rational(5, 2), z) 的简化结果是否正确
    assert besselsimp(besseli(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*sinh(z) - 3*z*cosh(z) + 3*sinh(z))/(sqrt(pi)*z**Rational(5, 2))
    # 测试 besselsimp 函数对 besseli(Rational(-5, 2), z) 的简化结果是否正确
    assert besselsimp(besseli(Rational(-5, 2), z)) == \
        sqrt(2)*(z**2*cosh(z) - 3*z*sinh(z) + 3*cosh(z))/(sqrt(pi)*z**Rational(5, 2))

    # 测试 besselsimp 函数对 besselk(S.Half, z) 的简化结果是否正确
    assert besselsimp(besselk(S.Half, z)) == \
        besselsimp(besselk(Rational(-1, 2), z)) == sqrt(pi)*exp(-z)/(sqrt(2)*sqrt(z))
    # 测试 besselsimp 函数对 besselk(Rational(5, 2), z) 的简化结果是否正确
    assert besselsimp(besselk(Rational(5, 2), z)) == \
        besselsimp(besselk(Rational(-5, 2), z)) == \
        sqrt(2)*sqrt(pi)*(z**2 + 3*z + 3)*exp(-z)/(2*z**Rational(5, 2))

    n = Symbol('n', integer=True, positive=True)

    # 测试 expand_func 函数对 besseli(n + 2, z) 的展开结果是否正确
    assert expand_func(besseli(n + 2, z)) == \
        besseli(n, z) + (-2*n - 2)*(-2*n*besseli(n, z)/z + besseli(n - 1, z))/z
    # 测试 expand_func 函数对 besselj(n + 2, z) 的展开结果是否正确
    assert expand_func(besselj(n + 2, z)) == \
        -besselj(n, z) + (2*n + 2)*(2*n*besselj(n, z)/z - besselj(n - 1, z))/z
    # 测试 expand_func 函数对 besselk(n + 2, z) 的展开结果是否正确
    assert expand_func(besselk(n + 2, z)) == \
        besselk(n, z) + (2*n + 2)*(2*n*besselk(n, z)/z + besselk(n - 1, z))/z
    # 测试 expand_func 函数对 bessely(n + 2, z) 的展开结果是否正确
    assert expand_func(bessely(n + 2, z)) == \
        -bessely(n, z) + (2*n + 2)*(2*n*bessely(n, z)/z - bessely(n - 1, z))/z

    # 测试 expand_func 函数对 besseli(n + S.Half, z).rewrite(jn) 的展开结果是否正确
    assert expand_func(besseli(n + S.Half, z).rewrite(jn)) == \
        (sqrt(2)*sqrt(z)*exp(-I*pi*(n + S.Half)/2) *
         exp_polar(I*pi/4)*jn(n, z*exp_polar(I*pi/2))/sqrt(pi))
    # 测试 expand_func 函数对 besselj(n + S.Half, z).rewrite(jn) 的展开结果是否正确
    assert expand_func(besselj(n + S.Half, z).rewrite(jn)) == \
        sqrt(2)*sqrt(z)*jn(n, z)/sqrt(pi)

    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i = Symbol('i', integer=True)
    # 对于每个贝塞尔函数 besselx（其中 x 可以是 j, y, i, k 中的一个）
    for besselx in [besselj, bessely, besseli, besselk]:
        # 断言 besselx(i, p) 的结果是扩展实数（is_extended_real 属性为 True）
        assert besselx(i, p).is_extended_real is True
        # 断言 besselx(i, x) 的结果不是扩展实数（is_extended_real 属性为 None）
        assert besselx(i, x).is_extended_real is None
        # 断言 besselx(x, z) 的结果不是扩展实数（is_extended_real 属性为 None）
        assert besselx(x, z).is_extended_real is None

    # 对于部分贝塞尔函数 besselx（其中 x 可以是 j, i）
    for besselx in [besselj, besseli]:
        # 断言 besselx(i, r) 的结果是扩展实数（is_extended_real 属性为 True）
        assert besselx(i, r).is_extended_real is True
    # 对于另外的部分贝塞尔函数 besselx（其中 x 可以是 y, k）
    for besselx in [bessely, besselk]:
        # 断言 besselx(i, r) 的结果不是扩展实数（is_extended_real 属性为 None）
        assert besselx(i, r).is_extended_real is None

    # 对于每个贝塞尔函数 besselx（其中 x 可以是 j, y, i, k 中的一个）
    for besselx in [besselj, bessely, besseli, besselk]:
        # 断言对 besselx(oo, x) 进行函数展开后的结果等于未评估时的 besselx(oo, x)
        assert expand_func(besselx(oo, x)) == besselx(oo, x, evaluate=False)
        # 断言对 besselx(-oo, x) 进行函数展开后的结果等于未评估时的 besselx(-oo, x)
        assert expand_func(besselx(-oo, x)) == besselx(-oo, x, evaluate=False)
# 声明一个装饰器 @slow，用于标记下面的函数为“慢速”测试函数
@slow
# 定义一个名为 test_slow_expand 的函数，用于测试扩展函数的正确性
def test_slow_expand():
    # 定义一个内部函数 check，用于比较两个表达式是否相等
    def check(eq, ans):
        # 检查简化后的表达式 eq 和预期结果 ans 是否相等，返回布尔值
        return tn(eq, ans) and eq == ans

    # 生成一个随机复数 rn
    rn = randcplx(a=1, b=0, d=0, c=2)

    # 对于每个贝塞尔函数 besselx，进行以下测试
    for besselx in [besselj, bessely, besseli, besselk]:
        # 生成一个半整数 ri，在 [-21/2, 21/2] 范围内
        ri = S(2*randint(-11, 10) + 1) / 2  # half integer in [-21/2, 21/2]
        # 断言简化后的 besselx(ri, z) 等于原始的 besselx(ri, z)
        assert tn(besselsimp(besselx(ri, z)), besselx(ri, z))

    # 断言扩展后的 besseli(rn, x) 等于原始的 besseli(rn - 2, x) - 2*(rn - 1)*besseli(rn - 1, x)/x
    assert check(expand_func(besseli(rn, x)),
                 besseli(rn - 2, x) - 2*(rn - 1)*besseli(rn - 1, x)/x)
    # 断言扩展后的 besseli(-rn, x) 等于原始的 besseli(-rn + 2, x) + 2*(-rn + 1)*besseli(-rn + 1, x)/x
    assert check(expand_func(besseli(-rn, x)),
                 besseli(-rn + 2, x) + 2*(-rn + 1)*besseli(-rn + 1, x)/x)

    # 断言扩展后的 besselj(rn, x) 等于原始的 -besselj(rn - 2, x) + 2*(rn - 1)*besselj(rn - 1, x)/x
    assert check(expand_func(besselj(rn, x)),
                 -besselj(rn - 2, x) + 2*(rn - 1)*besselj(rn - 1, x)/x)
    # 断言扩展后的 besselj(-rn, x) 等于原始的 -besselj(-rn + 2, x) + 2*(-rn + 1)*besselj(-rn + 1, x)/x
    assert check(expand_func(besselj(-rn, x)),
                 -besselj(-rn + 2, x) + 2*(-rn + 1)*besselj(-rn + 1, x)/x)

    # 断言扩展后的 besselk(rn, x) 等于原始的 besselk(rn - 2, x) + 2*(rn - 1)*besselk(rn - 1, x)/x
    assert check(expand_func(besselk(rn, x)),
                 besselk(rn - 2, x) + 2*(rn - 1)*besselk(rn - 1, x)/x)
    # 断言扩展后的 besselk(-rn, x) 等于原始的 besselk(-rn + 2, x) - 2*(-rn + 1)*besselk(-rn + 1, x)/x
    assert check(expand_func(besselk(-rn, x)),
                 besselk(-rn + 2, x) - 2*(-rn + 1)*besselk(-rn + 1, x)/x)

    # 断言扩展后的 bessely(rn, x) 等于原始的 -bessely(rn - 2, x) + 2*(rn - 1)*bessely(rn - 1, x)/x
    assert check(expand_func(bessely(rn, x)),
                 -bessely(rn - 2, x) + 2*(rn - 1)*bessely(rn - 1, x)/x)
    # 断言扩展后的 bessely(-rn, x) 等于原始的 -bessely(-rn + 2, x) + 2*(-rn + 1)*bessely(-rn + 1, x)/x
    assert check(expand_func(bessely(-rn, x)),
                 -bessely(-rn + 2, x) + 2*(-rn + 1)*bessely(-rn + 1, x)/x)


# 定义一个名为 mjn 的函数，返回扩展后的贝塞尔函数 jn(n, z)
def mjn(n, z):
    return expand_func(jn(n, z))


# 定义一个名为 myn 的函数，返回扩展后的贝塞尔函数 yn(n, z)
def myn(n, z):
    return expand_func(yn(n, z))


# 定义一个名为 test_jn 的函数，用于测试贝塞尔函数 jn(n, z) 的特性
def test_jn():
    z = symbols("z")
    # 断言 jn(0, 0) 等于 1
    assert jn(0, 0) == 1
    # 断言 jn(1, 0) 等于 0
    assert jn(1, 0) == 0
    # 断言 jn(-1, 0) 等于 S.ComplexInfinity
    assert jn(-1, 0) == S.ComplexInfinity
    # 断言 jn(z, 0) 等于 jn(z, 0, evaluate=False)
    assert jn(z, 0) == jn(z, 0, evaluate=False)
    # 断言 jn(0, oo) 等于 0
    assert jn(0, oo) == 0
    # 断言 jn(0, -oo) 等于 0
    assert jn(0, -oo) == 0

    # 断言 mjn(0, z) 等于 sin(z)/z
    assert mjn(0, z) == sin(z)/z
    # 断言 mjn(1, z) 等于 sin(z)/z**2 - cos(z)/z
    assert mjn(1, z) == sin(z)/z**2 - cos(z)/z
    # 断言 mjn(2, z) 等于 (3/z**3 - 1/z)*sin(z) - (3/z**2) * cos(z)
    assert mjn(2, z) == (3/z**3 - 1/z)*sin(z) - (3/z**2) * cos(z)
    # 断言 mjn(3, z) 等于 (15/z**4 - 6/z**2)*sin(z) + (1/z - 15/z**3)*cos(z)
    assert mjn(3, z) == (15/z**4 - 6/z**2)*sin(z) + (1/z - 15/z**3)*cos(z)
    # 断言 mjn(4, z) 等于 (1/z + 105/z**5 - 45/z**3)*sin(z) + (-105/z**4 + 10/z**2)*cos(z)
    assert mjn(4, z) == (1/z + 105/z**5 - 45/z**3)*sin(z) + (-105/z**4 + 10/z**2)*cos(z)
    # 断言 mjn(5, z) 等于 (945/z**6 - 420/z**4 + 15/z**2)*sin(z) + (-1/z - 945/z**5 + 105/z**3)*cos(z)
    assert mjn(5, z) == (945/z**6 - 420/z**4 + 15/z**2)*sin(z) + (-1/z - 945/z**5 + 105/z**3)*cos(z)
    # 断言 mjn(6, z) 等于 (-1/z + 10395/z**7 - 4725/z**5 + 210/z**3)*sin(z) + (-10395/z**6 +
    # 断言检查 myn 函数的返回值是否符合预期
    assert myn(3, pi) == 15/pi**4 - 6/pi**2
# 定义一个函数 `eq`，用于比较两个数组 `a` 和 `b` 的元素是否在给定的容差 `tol` 范围内相等
def eq(a, b, tol=1e-6):
    # 使用 zip 函数同时迭代数组 a 和 b 中的元素
    for u, v in zip(a, b):
        # 如果任意两个元素的差的绝对值不小于容差 `tol`，返回 False
        if not (abs(u - v) < tol):
            return False
    # 如果所有元素的差的绝对值都小于容差 `tol`，返回 True
    return True


# 定义一个测试函数 `test_jn_zeros`，用于测试贝塞尔函数 `jn_zeros` 的返回值是否符合预期
def test_jn_zeros():
    # 断言 `jn_zeros(0, 4)` 返回的结果应与给定的值列表 [3.141592, 6.283185, 9.424777, 12.566370] 相等
    assert eq(jn_zeros(0, 4), [3.141592, 6.283185, 9.424777, 12.566370])
    # 断言 `jn_zeros(1, 4)` 返回的结果应与给定的值列表 [4.493409, 7.725251, 10.904121, 14.066193] 相等
    assert eq(jn_zeros(1, 4), [4.493409, 7.725251, 10.904121, 14.066193])
    # 断言 `jn_zeros(2, 4)` 返回的结果应与给定的值列表 [5.763459, 9.095011, 12.322940, 15.514603] 相等
    assert eq(jn_zeros(2, 4), [5.763459, 9.095011, 12.322940, 15.514603])
    # 断言 `jn_zeros(3, 4)` 返回的结果应与给定的值列表 [6.987932, 10.417118, 13.698023, 16.923621] 相等
    assert eq(jn_zeros(3, 4), [6.987932, 10.417118, 13.698023, 16.923621])
    # 断言 `jn_zeros(4, 4)` 返回的结果应与给定的值列表 [8.182561, 11.704907, 15.039664, 18.301255] 相等
    assert eq(jn_zeros(4, 4), [8.182561, 11.704907, 15.039664, 18.301255])


# 定义一个测试函数 `test_bessel_eval`，用于测试贝塞尔函数在不同参数下的计算结果是否符合预期
def test_bessel_eval():
    # 声明符号变量 n, m, k，分别具有整数属性或者非零的整数属性
    n, m, k = Symbol('n', integer=True), Symbol('m'), Symbol('k', integer=True, zero=False)

    # 对于函数 f 在 [besselj, besseli] 中的每一个，执行一系列断言以验证其返回值是否符合预期
    for f in [besselj, besseli]:
        assert f(0, 0) is S.One
        assert f(2.1, 0) is S.Zero
        assert f(-3, 0) is S.Zero
        assert f(-10.2, 0) is S.ComplexInfinity
        assert f(1 + 3*I, 0) is S.Zero
        assert f(-3 + I, 0) is S.ComplexInfinity
        assert f(-2*I, 0) is S.NaN
        assert f(n, 0) != S.One and f(n, 0) != S.Zero
        assert f(m, 0) != S.One and f(m, 0) != S.Zero
        assert f(k, 0) is S.Zero

    # 验证特定贝塞尔函数在参数为 0 时的特殊值
    assert bessely(0, 0) is S.NegativeInfinity
    assert besselk(0, 0) is S.Infinity

    # 对于函数 f 在 [bessely, besselk] 中的每一个，执行一系列断言以验证其返回值是否符合预期
    for f in [bessely, besselk]:
        assert f(1 + I, 0) is S.ComplexInfinity
        assert f(I, 0) is S.NaN

    # 验证贝塞尔函数在特定参数下的性质
    for f in [besselj, bessely]:
        assert f(m, S.Infinity) is S.Zero
        assert f(m, S.NegativeInfinity) is S.Zero

    # 验证贝塞尔函数在特定参数下的性质
    for f in [besseli, besselk]:
        assert f(m, I*S.Infinity) is S.Zero
        assert f(m, I*S.NegativeInfinity) is S.Zero

    # 对于函数 f 在 [besseli, besselk] 中的每一个，执行一系列断言以验证其对称性质
    for f in [besseli, besselk]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == f(3, z)
        assert f(-n, z) == f(n, z)
        assert f(-m, z) != f(m, z)

    # 对于函数 f 在 [besselj, bessely] 中的每一个，执行一系列断言以验证其对称性质
    for f in [besselj, bessely]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == -f(3, z)
        assert f(-n, z) == (-1)**n*f(n, z)
        assert f(-m, z) != (-1)**m*f(m, z)

    # 对于函数 f 在 [besselj, besseli] 中的每一个，执行一系列断言以验证其特定性质
    for f in [besselj, besseli]:
        assert f(m, -z) == (-z)**m*z**(-m)*f(m, z)

    # 验证特定贝塞尔函数在参数为负数时的特殊值
    assert besseli(2, -z) == besseli(2, z)
    assert besseli(3, -z) == -besseli(3, z)

    assert besselj(0, -z) == besselj(0, z)
    assert besselj(1, -z) == -besselj(1, z)

    assert besseli(0, I*z) == besselj(0, z)
    assert besseli(1, I*z) == I*besselj(1, z)
    assert besselj(3, I*z) == -I*besseli(3, z)


# 定义一个测试函数 `test_bessel_nan`，用于验证贝塞尔函数在参数为 NaN 时的行为
def test_bessel_nan():
    # 对于一系列贝塞尔函数 [besselj, bessely, besseli, besselk, hankel1, hankel2, yn, jn]，
    # 验证其在参数为 NaN 时返回的值是否符合预期（此处只返回参数也为 NaN）
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, yn, jn]:
        assert f(1, S.NaN) == f(1, S.NaN, evaluate=False)


# 定义一个测试函数 `test_meromorphic`，用于验证贝塞尔函数的亚纯性质是否符合预期
def test_meromorphic():
    # 验证贝塞尔函数在特定参数下的亚纯性质
    assert besselj(2, x).is_meromorphic(x, 1) == True
    assert besselj(2, x).is_meromorphic(x, 0) == True
    assert besselj(2, x).is_meromorphic(x, oo) == False
    assert besselj(S(2)/3, x).is_meromorphic(x, 1) == True
    assert besselj(S(2)/3, x).is_meromorphic(x
    # 断言贝塞尔函数的性质：0阶贝塞尔函数对应的 x 是否在 x = 1 处是亚黎曼函数
    assert besselk(0, x).is_meromorphic(x, 1) == True
    # 断言贝塞尔函数的性质：2阶贝塞尔函数对应的 x 是否在 x = 0 处是亚黎曼函数
    assert besselk(2, x).is_meromorphic(x, 0) == True
    # 断言贝塞尔函数的性质：0阶修正贝塞尔函数对应的 x 是否在 x = 1 处是亚黎曼函数
    assert besseli(0, x).is_meromorphic(x, 1) == True
    # 断言贝塞尔函数的性质：2阶修正贝塞尔函数对应的 x 是否在 x = 0 处是亚黎曼函数
    assert besseli(2, x).is_meromorphic(x, 0) == True
    # 断言贝塞尔函数的性质：0阶第二类贝塞尔函数对应的 x 是否在 x = 1 处是亚黎曼函数
    assert bessely(0, x).is_meromorphic(x, 1) == True
    # 断言贝塞尔函数的性质：0阶第二类贝塞尔函数对应的 x 是否在 x = 0 处不是亚黎曼函数
    assert bessely(0, x).is_meromorphic(x, 0) == False
    # 断言贝塞尔函数的性质：2阶第二类贝塞尔函数对应的 x 是否在 x = 0 处是亚黎曼函数
    assert bessely(2, x).is_meromorphic(x, 0) == True
    # 断言汉克尔函数Hankel第一类函数的性质：3阶Hankel函数对应的 x^2 + 2*x 是否在 x = 1 处是亚黎曼函数
    assert hankel1(3, x**2 + 2*x).is_meromorphic(x, 1) == True
    # 断言汉克尔函数Hankel第一类函数的性质：0阶Hankel函数对应的 x 是否在 x = 0 处不是亚黎曼函数
    assert hankel1(0, x).is_meromorphic(x, 0) == False
    # 断言汉克尔函数Hankel第二类函数的性质：11阶Hankel函数对应的 4 是否在 x = 5 处是亚黎曼函数
    assert hankel2(11, 4).is_meromorphic(x, 5) == True
    # 断言H_n^(1)函数的性质：6阶H_n^(1)函数对应的 7*x^3 + 4 是否在 x = 7 处是亚黎曼函数
    assert hn1(6, 7*x**3 + 4).is_meromorphic(x, 7) == True
    # 断言H_n^(2)函数的性质：3阶H_n^(2)函数对应的 2*x 是否在 x = 9 处是亚黎曼函数
    assert hn2(3, 2*x).is_meromorphic(x, 9) == True
    # 断言J_n函数（第一类贝塞尔函数）的性质：5阶J_n函数对应的 2*x + 7 是否在 x = 4 处是亚黎曼函数
    assert jn(5, 2*x + 7).is_meromorphic(x, 4) == True
    # 断言Y_n函数（第二类贝塞尔函数）的性质：8阶Y_n函数对应的 x^2 + 11 是否在 x = 6 处是亚黎曼函数
    assert yn(8, x**2 + 11).is_meromorphic(x, 6) == True
def test_conjugate():
    # 定义符号变量 n
    n = Symbol('n')
    # 定义符号变量 z，限制其为非扩展实数
    z = Symbol('z', extended_real=False)
    # 定义符号变量 x，限制其为扩展实数
    x = Symbol('x', extended_real=True)
    # 定义符号变量 y，限制其为正数
    y = Symbol('y', positive=True)
    # 定义符号变量 t，限制其为负数
    t = Symbol('t', negative=True)

    # 对于每个函数 f 在列表中执行以下断言
    for f in [besseli, besselj, besselk, bessely, hankel1, hankel2]:
        # 断言 f(n, -1) 的共轭不等于 f(conjugate(n), -1)
        assert f(n, -1).conjugate() != f(conjugate(n), -1)
        # 断言 f(n, x) 的共轭不等于 f(conjugate(n), x)
        assert f(n, x).conjugate() != f(conjugate(n), x)
        # 断言 f(n, t) 的共轭不等于 f(conjugate(n), t)
        assert f(n, t).conjugate() != f(conjugate(n), t)

    # 生成一个随机复数 rz
    rz = randcplx(b=0.5)

    # 对于每个函数 f 在列表中执行以下断言
    for f in [besseli, besselj, besselk, bessely]:
        # 断言 f(n, 1 + I) 的共轭等于 f(conjugate(n), 1 - I)
        assert f(n, 1 + I).conjugate() == f(conjugate(n), 1 - I)
        # 断言 f(n, 0) 的共轭等于 f(conjugate(n), 0)
        assert f(n, 0).conjugate() == f(conjugate(n), 0)
        # 断言 f(n, 1) 的共轭等于 f(conjugate(n), 1)
        assert f(n, 1).conjugate() == f(conjugate(n), 1)
        # 断言 f(n, z) 的共轭等于 f(conjugate(n), conjugate(z))
        assert f(n, z).conjugate() == f(conjugate(n), conjugate(z))
        # 断言 f(n, y) 的共轭等于 f(conjugate(n), y)
        assert f(n, y).conjugate() == f(conjugate(n), y)
        # 使用函数 tn 检查 f(n, rz) 的共轭和 f(conjugate(n), conjugate(rz)) 是否近似相等
        assert tn(f(n, rz).conjugate(), f(conjugate(n), conjugate(rz)))

    # 断言 hankel1(n, 1 + I) 的共轭等于 hankel2(conjugate(n), 1 - I)
    assert hankel1(n, 1 + I).conjugate() == hankel2(conjugate(n), 1 - I)
    # 断言 hankel1(n, 0) 的共轭等于 hankel2(conjugate(n), 0)
    assert hankel1(n, 0).conjugate() == hankel2(conjugate(n), 0)
    # 断言 hankel1(n, 1) 的共轭等于 hankel2(conjugate(n), 1)
    assert hankel1(n, 1).conjugate() == hankel2(conjugate(n), 1)
    # 断言 hankel1(n, y) 的共轭等于 hankel2(conjugate(n), y)
    assert hankel1(n, y).conjugate() == hankel2(conjugate(n), y)
    # 断言 hankel1(n, z) 的共轭等于 hankel2(conjugate(n), conjugate(z))
    assert hankel1(n, z).conjugate() == hankel2(conjugate(n), conjugate(z))
    # 使用函数 tn 检查 hankel1(n, rz) 的共轭和 hankel2(conjugate(n), conjugate(rz)) 是否近似相等
    assert tn(hankel1(n, rz).conjugate(), hankel2(conjugate(n), conjugate(rz)))

    # 断言 hankel2(n, 1 + I) 的共轭等于 hankel1(conjugate(n), 1 - I)
    assert hankel2(n, 1 + I).conjugate() == hankel1(conjugate(n), 1 - I)
    # 断言 hankel2(n, 0) 的共轭等于 hankel1(conjugate(n), 0)
    assert hankel2(n, 0).conjugate() == hankel1(conjugate(n), 0)
    # 断言 hankel2(n, 1) 的共轭等于 hankel1(conjugate(n), 1)
    assert hankel2(n, 1).conjugate() == hankel1(conjugate(n), 1)
    # 断言 hankel2(n, y) 的共轭等于 hankel1(conjugate(n), y)
    assert hankel2(n, y).conjugate() == hankel1(conjugate(n), y)
    # 断言 hankel2(n, z) 的共轭等于 hankel1(conjugate(n), conjugate(z))
    assert hankel2(n, z).conjugate() == hankel1(conjugate(n), conjugate(z))
    # 使用函数 tn 检查 hankel2(n, rz) 的共轭和 hankel1(conjugate(n), conjugate(rz)) 是否近似相等
    assert tn(hankel2(n, rz).conjugate(), hankel1(conjugate(n), conjugate(rz)))


def test_branching():
    # 断言 besselj(polar_lift(k), x) 等于 besselj(k, x)
    assert besselj(polar_lift(k), x) == besselj(k, x)
    # 断言 besseli(polar_lift(k), x) 等于 besseli(k, x)

    # 定义符号变量 n，限制其为整数
    n = Symbol('n', integer=True)
    # 断言 besselj(n, exp_polar(2*pi*I)*x) 等于 besselj(n, x)
    assert besselj(n, exp_polar(2*pi*I)*x) == besselj(n, x)
    # 断言 besselj(n, polar_lift(x)) 等于 besselj(n, x)
    assert besselj(n, polar_lift(x)) == besselj(n, x)
    # 断言 besseli(n, exp_polar(2*pi*I)*x) 等于 besseli(n, x)
    assert besseli(n, exp_polar(2*pi*I)*x) == besseli(n, x)
    # 断言 besseli(n, polar_lift(x)) 等于 besseli(n, x)

    # 定义函数 tn，参数为 func 和 s
    def tn(func, s):
        # 导入 uniform 函数从 sympy.core.random 模块
        from sympy.core.random import uniform
        # 生成一个随机复数 c，范围为 1 到 5
        c = uniform(1, 5)
        # 定义表达式 expr，计算 func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
        expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
        # 定义一个很小的误差值 eps
        eps = 1e-15
        # 定义表达式 expr2，计算 func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
        expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
        # 返回 expr 和 expr2 的绝对差是否小于 1e-10
        return abs(expr.n() - expr2.n()).n() < 1e-10

    # 定义符号变量 nu
    nu = Symbol('nu')
    # 断言 besselj(nu, exp_polar(2*pi
    # 断言语句：验证 airyai(x+I*y) 的实部和虚部是否等于以下值的元组
    assert airyai(x+I*y).as_real_imag() == (
            # airyai(x - I*y) 和 airyai(x + I*y) 的实部的平均值
            airyai(x - I*y)/2 + airyai(x + I*y)/2,
            # airyai(x - I*y) 和 airyai(x + I*y) 的虚部的差值的一半乘以虚数单位
            I*(airyai(x - I*y) - airyai(x + I*y))/2)
# 定义测试函数 test_airyai
def test_airyai():
    # 创建一个复数符号 z，实部为非实数
    z = Symbol('z', real=False)
    # 创建一个负数符号 t
    t = Symbol('t', negative=True)
    # 创建一个正数符号 p
    p = Symbol('p', positive=True)

    # 断言 airyai(z) 返回的类型是 airyai 类型
    assert isinstance(airyai(z), airyai)

    # 断言 airyai(0) 的值等于特定数学表达式
    assert airyai(0) == 3**Rational(1, 3)/(3*gamma(Rational(2, 3)))
    # 断言 airyai(oo) 的值为 0
    assert airyai(oo) == 0
    # 断言 airyai(-oo) 的值为 0
    assert airyai(-oo) == 0

    # 断言 airyai(z) 对 z 求导数的结果等于 airyaiprime(z)
    assert diff(airyai(z), z) == airyaiprime(z)

    # 断言 airyai(z) 在 z = 0 处展开成级数的前 3 项的结果
    assert series(airyai(z), z, 0, 3) == (
        3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) - 3**Rational(1, 6)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    # 断言将 airyai(z) 用超函换形式重写的结果
    assert airyai(z).rewrite(hyper) == (
        -3**Rational(2, 3)*z*hyper((), (Rational(4, 3),), z**3/9)/(3*gamma(Rational(1, 3))) +
         3**Rational(1, 3)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    # 断言 airyai(z) 重写为 besselj 函数的结果仍然是 airyai 类型
    assert isinstance(airyai(z).rewrite(besselj), airyai)
    # 断言 airyai(t) 重写为 besselj 函数的结果
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    # 断言 airyai(z) 重写为 besseli 函数的结果
    assert airyai(z).rewrite(besseli) == (
        -z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(1, 3)) +
         (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)
    # 断言 airyai(p) 重写为 besseli 函数的结果
    assert airyai(p).rewrite(besseli) == (
        sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) -
                 besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)

    # 断言对 airyai(2*(3*z**5)**Rational(1, 3)) 进行函数展开的结果
    assert expand_func(airyai(2*(3*z**5)**Rational(1, 3))) == (
        -sqrt(3)*(-1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
         (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


# 定义测试函数 test_airybi
def test_airybi():
    # 创建一个复数符号 z，实部为非实数
    z = Symbol('z', real=False)
    # 创建一个负数符号 t
    t = Symbol('t', negative=True)
    # 创建一个正数符号 p
    p = Symbol('p', positive=True)

    # 断言 airybi(z) 返回的类型是 airybi 类型
    assert isinstance(airybi(z), airybi)

    # 断言 airybi(0) 的值等于特定数学表达式
    assert airybi(0) == 3**Rational(5, 6)/(3*gamma(Rational(2, 3)))
    # 断言 airybi(oo) 的值为 oo
    assert airybi(oo) is oo
    # 断言 airybi(-oo) 的值为 0
    assert airybi(-oo) == 0

    # 断言 airybi(z) 对 z 求导数的结果等于 airybiprime(z)
    assert diff(airybi(z), z) == airybiprime(z)

    # 断言 airybi(z) 在 z = 0 处展开成级数的前 3 项的结果
    assert series(airybi(z), z, 0, 3) == (
        3**Rational(1, 3)*gamma(Rational(1, 3))/(2*pi) + 3**Rational(2, 3)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    # 断言将 airybi(z) 用超函换形式重写的结果
    assert airybi(z).rewrite(hyper) == (
        3**Rational(1, 6)*z*hyper((), (Rational(4, 3),), z**3/9)/gamma(Rational(1, 3)) +
        3**Rational(5, 6)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    # 断言 airybi(z) 重写为 besselj 函数的结果仍然是 airybi 类型
    assert isinstance(airybi(z).rewrite(besselj), airybi)
    # 断言 airyai(t) 重写为 besselj 函数的结果
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    # 断言 airybi(z) 重写为 besseli 函数的结果
    assert airybi(z).rewrite(besseli) == (
        sqrt(3)*(z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(1, 3) +
                 (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3))/3)
    # 断言：验证 airybi(p) 重写为 besseli 函数后的结果是否等于给定的表达式
    assert airybi(p).rewrite(besseli) == (
        sqrt(3)*sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) +
                         besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)
    
    # 断言：验证 airybi(2*(3*z**5)**Rational(1, 3)) 展开后是否等于给定的表达式
    assert expand_func(airybi(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(1 - (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/2)
# 定义测试函数 test_airyaiprime，用于测试 airyaiprime 函数的各种属性和方法
def test_airyaiprime():
    # 创建实数域外的符号 z
    z = Symbol('z', real=False)
    # 创建负数域内的符号 t
    t = Symbol('t', negative=True)
    # 创建正数域内的符号 p
    p = Symbol('p', positive=True)

    # 断言 airyaiprime(z) 返回的对象是 airyaiprime 类的实例
    assert isinstance(airyaiprime(z), airyaiprime)

    # 断言 airyaiprime(0) 的计算结果
    assert airyaiprime(0) == -3**Rational(2, 3)/(3*gamma(Rational(1, 3)))
    # 断言 airyaiprime(oo) 的计算结果
    assert airyaiprime(oo) == 0

    # 断言 airyaiprime(z) 对 z 的导数计算结果
    assert diff(airyaiprime(z), z) == z*airyai(z)

    # 断言 airyaiprime(z) 在 z=0 处展开到 3 阶的级数结果
    assert series(airyaiprime(z), z, 0, 3) == (
        -3**Rational(2, 3)/(3*gamma(Rational(1, 3))) +
        3**Rational(1, 3)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    # 断言 airyaiprime(z) 重写为超函 hyper 后的结果
    assert airyaiprime(z).rewrite(hyper) == (
        3**Rational(1, 3)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) -
        3**Rational(2, 3)*hyper((), (Rational(1, 3),), z**3/9)/(3*gamma(Rational(1, 3))))

    # 断言 airyaiprime(z) 重写为 besselj 函数后返回的对象是 airyaiprime 类的实例
    assert isinstance(airyaiprime(z).rewrite(besselj), airyaiprime)

    # 断言 airyai(t) 重写为 besselj 函数后的结果
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)

    # 断言 airyaiprime(z) 重写为 besseli 函数后的结果
    assert airyaiprime(z).rewrite(besseli) == (
        z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(2, 3)) -
        (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)

    # 断言 airyaiprime(p) 重写为 besseli 函数后的结果
    assert airyaiprime(p).rewrite(besseli) == (
        p*(-besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    # 断言 airyaiprime(2*(3*z**5)**Rational(1, 3)) 展开后的函数结果
    assert expand_func(airyaiprime(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


# 定义测试函数 test_airybiprime，用于测试 airybiprime 函数的各种属性和方法
def test_airybiprime():
    # 创建实数域外的符号 z
    z = Symbol('z', real=False)
    # 创建负数域内的符号 t
    t = Symbol('t', negative=True)
    # 创建正数域内的符号 p
    p = Symbol('p', positive=True)

    # 断言 airybiprime(z) 返回的对象是 airybiprime 类的实例
    assert isinstance(airybiprime(z), airybiprime)

    # 断言 airybiprime(0) 的计算结果
    assert airybiprime(0) == 3**Rational(1, 6)/gamma(Rational(1, 3))
    # 断言 airybiprime(oo) 的计算结果
    assert airybiprime(oo) is oo
    # 断言 airybiprime(-oo) 的计算结果
    assert airybiprime(-oo) == 0

    # 断言 airybiprime(z) 对 z 的导数计算结果
    assert diff(airybiprime(z), z) == z*airybi(z)

    # 断言 airybiprime(z) 在 z=0 处展开到 3 阶的级数结果
    assert series(airybiprime(z), z, 0, 3) == (
        3**Rational(1, 6)/gamma(Rational(1, 3)) + 3**Rational(5, 6)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    # 断言 airybiprime(z) 重写为超函 hyper 后的结果
    assert airybiprime(z).rewrite(hyper) == (
        3**Rational(5, 6)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) +
        3**Rational(1, 6)*hyper((), (Rational(1, 3),), z**3/9)/gamma(Rational(1, 3)))

    # 断言 airybiprime(z) 重写为 besselj 函数后返回的对象是 airybiprime 类的实例
    assert isinstance(airybiprime(z).rewrite(besselj), airybiprime)

    # 断言 airyai(t) 重写为 besselj 函数后的结果
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)

    # 断言 airybiprime(z) 重写为 besseli 函数后的结果
    assert airybiprime(z).rewrite(besseli) == (
        sqrt(3)*(z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(2, 3) +
                 (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-2, 3), 2*z**Rational(3, 2)/3))/3)
    # 断言：验证 airybiprime(p) 通过使用 besseli 重写后的结果是否等于给定的表达式
    assert airybiprime(p).rewrite(besseli) == (
        # 计算表达式的值：sqrt(3)*p*(besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3
        sqrt(3)*p*(besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    # 断言：验证 expand_func(airybiprime(2*(3*z**5)**Rational(1, 3))) 的结果是否等于给定的表达式
    assert expand_func(airybiprime(2*(3*z**5)**Rational(1, 3))) == (
        # 计算表达式的值：sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        #               (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)
# 定义一个测试函数 `test_marcumq`
def test_marcumq():
    # 创建符号变量 m, a, b
    m = Symbol('m')
    a = Symbol('a')
    b = Symbol('b')

    # 断言 marcumq(0, 0, 0) 的返回值为 0
    assert marcumq(0, 0, 0) == 0
    # 断言 marcumq(m, 0, b) 的返回值为 uppergamma(m, b**2/2)/gamma(m)
    assert marcumq(m, 0, b) == uppergamma(m, b**2/2)/gamma(m)
    # 断言 marcumq(2, 0, 5) 的返回值为 27*exp(Rational(-25, 2))/2
    assert marcumq(2, 0, 5) == 27*exp(Rational(-25, 2))/2
    # 断言 marcumq(0, a, 0) 的返回值为 1 - exp(-a**2/2)
    assert marcumq(0, a, 0) == 1 - exp(-a**2/2)
    # 断言 marcumq(0, pi, 0) 的返回值为 1 - exp(-pi**2/2)
    assert marcumq(0, pi, 0) == 1 - exp(-pi**2/2)
    # 断言 marcumq(1, a, a) 的返回值为 S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(1, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    # 断言 marcumq(2, a, a) 的返回值为 S.Half + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)
    assert marcumq(2, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    # 断言 marcumq(1, a, 3) 对 a 的导数为 a*(-marcumq(1, a, 3) + marcumq(2, a, 3))
    assert diff(marcumq(1, a, 3), a) == a*(-marcumq(1, a, 3) + marcumq(2, a, 3))
    # 断言 marcumq(2, 3, b) 对 b 的导数为 -b**2*exp(-b**2/2 - Rational(9, 2))*besseli(1, 3*b)/3
    assert diff(marcumq(2, 3, b), b) == -b**2*exp(-b**2/2 - Rational(9, 2))*besseli(1, 3*b)/3

    # 创建符号变量 x
    x = Symbol('x')
    # 断言 marcumq(2, 3, 4) 重写为积分形式，与 Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3 相等
    assert marcumq(2, 3, 4).rewrite(Integral, x=x) == \
           Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3
    # 断言 marcumq(5, -2, 3) 重写为积分后，计算数值近似等于 0.7905769565
    assert eq([marcumq(5, -2, 3).rewrite(Integral).evalf(10)],
              [0.7905769565])

    # 创建符号变量 k
    k = Symbol('k')
    # 断言 marcumq(-3, -5, -7) 重写为和形式，与 exp(-37)*Sum((Rational(5, 7))**k*besseli(k, 35), (k, 4, oo)) 相等
    assert marcumq(-3, -5, -7).rewrite(Sum, k=k) == \
           exp(-37)*Sum((Rational(5, 7))**k*besseli(k, 35), (k, 4, oo))
    # 断言 marcumq(1, 3, 1) 重写为和形式后，计算数值近似等于 0.9891705502
    assert eq([marcumq(1, 3, 1).rewrite(Sum).evalf(10)],
              [0.9891705502])

    # 断言 marcumq(1, a, a, evaluate=False) 重写为 besseli 函数形式，与 S.Half + exp(-a**2)*besseli(0, a**2)/2 相等
    assert marcumq(1, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    # 断言 marcumq(2, a, a, evaluate=False) 重写为 besseli 函数形式，与上述两项相加后相等
    assert marcumq(2, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + \
           exp(-a**2)*besseli(1, a**2)
    # 断言 marcumq(3, a, a) 重写为 besseli 函数形式，与 (besseli(1, a**2) + besseli(2, a**2))*exp(-a**2) + S.Half + exp(-a**2)*besseli(0, a**2)/2 相等
    assert marcumq(3, a, a).rewrite(besseli) == (besseli(1, a**2) + besseli(2, a**2))*exp(-a**2) + \
           S.Half + exp(-a**2)*besseli(0, a**2)/2
    # 断言 marcumq(5, 8, 8) 重写为 besseli 函数形式，与 exp(-64)*besseli(0, 64)/2 + (besseli(4, 64) + besseli(3, 64) + besseli(2, 64) + besseli(1, 64))*exp(-64) + S.Half 相等
    assert marcumq(5, 8, 8).rewrite(besseli) == exp(-64)*besseli(0, 64)/2 + \
           (besseli(4, 64) + besseli(3, 64) + besseli(2, 64) + besseli(1, 64))*exp(-64) + S.Half
    # 断言 marcumq(m, a, a) 重写为 besseli 函数形式，与原函数相等
    assert marcumq(m, a, a).rewrite(besseli) == marcumq(m, a, a)

    # 创建符号变量 x，且 x 是整数
    x = Symbol('x', integer=True)
    # 断言 marcumq(x, a, a) 重写为 besseli 函数形式，与原函数相等
    assert marcumq(x, a, a).rewrite(besseli) == marcumq(x, a, a)


# 定义一个测试函数 `test_issue_26134`
def test_issue_26134():
    # 创建符号变量 x
    x = Symbol('x')
    # 断言 marcumq(2, 3, 4) 重写为积分形式，与 Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3 近似等效
    assert marcumq(2, 3, 4).rewrite(Integral, x=x).dummy_eq(
        Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3)
```