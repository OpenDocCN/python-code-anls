# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_zeta_functions.py`

```
# 导入从Sympy库中引入的具体模块和函数

from sympy.concrete.summations import Sum  # 导入Sum类，用于处理求和运算
from sympy.core.function import expand_func  # 导入expand_func函数，用于展开函数表达式
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)  # 导入数值类型和常数
from sympy.core.singleton import S  # 导入Sympy中的单例模块S
from sympy.core.symbol import Symbol  # 导入Symbol类，用于定义符号变量
from sympy.functions.elementary.complexes import (Abs, polar_lift)  # 导入复数相关的函数
from sympy.functions.elementary.exponential import (exp, exp_polar, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)  # 导入ζ函数及相关函数
from sympy.series.order import O  # 导入O类，用于表示级数的阶数
from sympy.core.function import ArgumentIndexError  # 导入参数索引错误异常类
from sympy.functions.combinatorial.numbers import (bernoulli, factorial, genocchi, harmonic)  # 导入组合数学函数
from sympy.testing.pytest import raises  # 导入用于测试的raises函数
from sympy.core.random import (test_derivative_numerically as td,
                      random_complex_number as randcplx, verify_numerically)  # 导入用于随机数生成和数值验证的函数

# 定义符号变量
x = Symbol('x')
a = Symbol('a')
b = Symbol('b', negative=True)
z = Symbol('z')
s = Symbol('s')

# 定义测试函数test_zeta_eval，用于测试ζ函数的评估和计算
def test_zeta_eval():

    # ζ函数的特殊值和异常情况处理
    assert zeta(nan) is nan  # 当输入为nan时，返回nan
    assert zeta(x, nan) is nan  # 当输入为(x, nan)时，返回nan

    assert zeta(0) == Rational(-1, 2)  # ζ(0) = -1/2
    assert zeta(0, x) == S.Half - x  # ζ(0, x) = 1/2 - x
    assert zeta(0, b) == S.Half - b  # ζ(0, b) = 1/2 - b

    assert zeta(1) is zoo  # ζ(1) = ∞
    assert zeta(1, 2) is zoo  # ζ(1, 2) = ∞
    assert zeta(1, -7) is zoo  # ζ(1, -7) = ∞
    assert zeta(1, x) is zoo  # ζ(1, x) = ∞

    assert zeta(2, 1) == pi**2/6  # ζ(2, 1) = π²/6
    assert zeta(3, 1) == zeta(3)  # ζ(3, 1) = ζ(3)

    assert zeta(2) == pi**2/6  # ζ(2) = π²/6
    assert zeta(4) == pi**4/90  # ζ(4) = π⁴/90
    assert zeta(6) == pi**6/945  # ζ(6) = π⁶/945

    assert zeta(4, 3) == pi**4/90 - Rational(17, 16)  # ζ(4, 3) = π⁴/90 - 17/16
    assert zeta(7, 4) == zeta(7) - Rational(282251, 279936)  # ζ(7, 4) = ζ(7) - 282251/279936
    assert zeta(S.Half, 2).func == zeta  # ζ(1/2, 2)的函数类型为ζ
    assert expand_func(zeta(S.Half, 2)) == zeta(S.Half) - 1  # 展开ζ(1/2, 2) = ζ(1/2) - 1
    assert zeta(x, 3).func == zeta  # ζ(x, 3)的函数类型为ζ
    assert expand_func(zeta(x, 3)) == zeta(x) - 1 - 1/2**x  # 展开ζ(x, 3) = ζ(x) - 1 - 1/2**x

    assert zeta(2, 0) is nan  # ζ(2, 0) = nan
    assert zeta(3, -1) is nan  # ζ(3, -1) = nan
    assert zeta(4, -2) is nan  # ζ(4, -2) = nan

    assert zeta(oo) == 1  # ζ(∞) = 1

    assert zeta(-1) == Rational(-1, 12)  # ζ(-1) = -1/12
    assert zeta(-2) == 0  # ζ(-2) = 0
    assert zeta(-3) == Rational(1, 120)  # ζ(-3) = 1/120
    assert zeta(-4) == 0  # ζ(-4) = 0
    assert zeta(-5) == Rational(-1, 252)  # ζ(-5) = -1/252

    assert zeta(-1, 3) == Rational(-37, 12)  # ζ(-1, 3) = -37/12
    assert zeta(-1, 7) == Rational(-253, 12)  # ζ(-1, 7) = -253/12
    assert zeta(-1, -4) == Rational(-121, 12)  # ζ(-1, -4) = -121/12
    assert zeta(-1, -9) == Rational(-541, 12)  # ζ(-1, -9) = -541/12

    assert zeta(-4, 3) == -17  # ζ(-4, 3) = -17
    assert zeta(-4, -8) == 8772  # ζ(-4, -8) = 8772

    assert zeta(0, 1) == Rational(-1, 2)  # ζ(0, 1) = -1/2
    assert zeta(0, -1) == Rational(3, 2)  # ζ(0, -1) = 3/2

    assert zeta(0, 2) == Rational(-3, 2)  # ζ(0, 2) = -3/2
    assert zeta(0, -2) == Rational(5, 2)  # ζ(0, -2) = 5/2

    # 对ζ(3)进行数值计算，精确到20位小数，与给定值比较，允许误差为1e-19
    assert zeta(3).evalf(20).epsilon_eq(Float("1.2020569031595942854", 20), 1e-19)


# 定义测试函数test_zeta_series，用于测试ζ函数的级数展开
def test_zeta_series():
    assert zeta(x, a).series(a, z, 2) == \
        zeta(x, z) - x*(a-z)*zeta(x+1, z) + O((a-z)**2, (a, z))

# 定义测试函数test_dirichlet_eta_eval，用于测试Dirichlet η函数的评估和计算
def test_dirichlet_eta_eval():
    assert dirichlet_eta(0) == S.Half  # η(0) = 1/2
    assert dirichlet_eta(-1) == Rational(1, 4)  # η(-1) = 1/4
    assert dirichlet_eta(1) == log(2)  # η(1) = log(2)
    assert dirichlet_eta(1, S.Half).simplify() == pi/2  # η(1, 1/2)化简后为π/2
    assert dirichlet_eta(1, 2) == 1 - log(2)
    # 检查 dirichlet_eta 函数在输入 4 时是否返回指定的数学常数值
    assert dirichlet_eta(4) == pi**4*Rational(7, 720)
    
    # 检查 dirichlet_eta 函数在输入复数 I 时，对其进行数值评估并将结果转换为字符串
    assert str(dirichlet_eta(I).evalf(n=10)) == '0.5325931818 + 0.2293848577*I'
    
    # 检查 dirichlet_eta 函数在输入两个复数 I 时，对其进行数值评估并将结果转换为字符串
    assert str(dirichlet_eta(I, I).evalf(n=10)) == '3.462349253 + 0.220285771*I'
def test_riemann_xi_eval():
    # 检查 riemann_xi 函数对于特定输入的计算结果是否正确
    assert riemann_xi(2) == pi/6
    # 检查 riemann_xi 函数在输入为 0 时的计算结果是否为 1/2
    assert riemann_xi(0) == Rational(1, 2)
    # 检查 riemann_xi 函数在输入为 1 时的计算结果是否为 1/2
    assert riemann_xi(1) == Rational(1, 2)
    # 检查 riemann_xi 函数在输入为 3 时通过 zeta 函数重写后的计算结果是否正确
    assert riemann_xi(3).rewrite(zeta) == 3*zeta(3)/(2*pi)
    # 检查 riemann_xi 函数在输入为 4 时的计算结果是否为 pi^2/15
    assert riemann_xi(4) == pi**2/15


def test_rewriting():
    from sympy.functions.elementary.piecewise import Piecewise
    # 检查 dirichlet_eta 函数通过 rewrite 方法转换为 zeta 函数后是否返回 Piecewise 对象
    assert isinstance(dirichlet_eta(x).rewrite(zeta), Piecewise)
    # 检查 dirichlet_eta 函数通过 rewrite 方法转换为 genocchi 函数后是否返回 Piecewise 对象
    assert isinstance(dirichlet_eta(x).rewrite(genocchi), Piecewise)
    # 检查 zeta 函数通过 rewrite 方法转换为 dirichlet_eta 函数后的计算结果是否正确
    assert zeta(x).rewrite(dirichlet_eta) == dirichlet_eta(x)/(1 - 2**(1 - x))
    # 检查带有参数 a 的 zeta 函数通过 rewrite 方法转换为 dirichlet_eta 函数后的计算结果是否等于原始 zeta 函数
    assert zeta(x).rewrite(dirichlet_eta, a=2) == zeta(x)
    # 检查数值验证函数 verify_numerically 对 dirichlet_eta 函数和其通过 zeta 函数转换后的结果是否返回相似的数值
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(zeta), x)
    # 检查数值验证函数 verify_numerically 对 dirichlet_eta 函数和其通过 genocchi 函数转换后的结果是否返回相似的数值
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(genocchi), x)
    # 检查数值验证函数 verify_numerically 对 zeta 函数和其通过 dirichlet_eta 函数转换后的结果是否返回相似的数值
    assert verify_numerically(zeta(x), zeta(x).rewrite(dirichlet_eta), x)

    # 检查带有参数 a 的 zeta 函数通过 rewrite 方法转换为 lerchphi 函数后的计算结果是否正确
    assert zeta(x, a).rewrite(lerchphi) == lerchphi(1, x, a)
    # 检查带有参数 s 和 z 的 polylog 函数通过 rewrite 方法转换为 lerchphi 函数后的计算结果是否正确
    assert polylog(s, z).rewrite(lerchphi) == lerchphi(z, s, 1)*z

    # 检查 lerchphi 函数通过 rewrite 方法转换为 zeta 函数后的计算结果是否正确
    assert lerchphi(1, x, a).rewrite(zeta) == zeta(x, a)
    # 检查带有参数 z 和 s 的 lerchphi 函数通过 rewrite 方法转换为 polylog 函数后的计算结果是否正确
    assert z*lerchphi(z, s, 1).rewrite(polylog) == polylog(s, z)


def test_derivatives():
    from sympy.core.function import Derivative
    # 检查 zeta 函数关于 x 的偏导数是否等于 Derivative 对象
    assert zeta(x, a).diff(x) == Derivative(zeta(x, a), x)
    # 检查 zeta 函数关于 a 的偏导数是否等于 -x*zeta(x + 1, a)
    assert zeta(x, a).diff(a) == -x*zeta(x + 1, a)
    # 检查 lerchphi 函数关于 z 的偏导数是否等于 (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
    assert lerchphi(z, s, a).diff(z) == (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
    # 检查 lerchphi 函数关于 a 的偏导数是否等于 -s*lerchphi(z, s + 1, a)
    assert lerchphi(z, s, a).diff(a) == -s*lerchphi(z, s + 1, a)
    # 检查 polylog 函数关于 z 的偏导数是否等于 polylog(s - 1, z)/z
    assert polylog(s, z).diff(z) == polylog(s - 1, z)/z

    # 随机生成复数参数 b 和 c
    b = randcplx()
    c = randcplx()
    # 检查 td 函数对 zeta(b, x) 关于 x 的偏导数是否成立
    assert td(zeta(b, x), x)
    # 检查 td 函数对 polylog(b, z) 关于 z 的偏导数是否成立
    assert td(polylog(b, z), z)
    # 检查 td 函数对 lerchphi(c, b, x) 关于 x 的偏导数是否成立
    assert td(lerchphi(c, b, x), x)
    # 检查 td 函数对 lerchphi(x, b, c) 关于 x 的偏导数是否成立
    assert td(lerchphi(x, b, c), x)
    # 检查 lerchphi 函数的 fdiff 方法在索引为 2 时是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(2))
    # 检查 lerchphi 函数的 fdiff 方法在索引为 4 时是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(4))
    # 检查 polylog 函数的 fdiff 方法在索引为 1 时是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(1))
    # 检查 polylog 函数的 fdiff 方法在索引为 3 时是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(3))


def myexpand(func, target):
    # 将函数 func 进行展开
    expanded = expand_func(func)
    # 如果指定了目标 target，则检查展开后的结果是否等于目标
    if target is not None:
        return expanded == target
    # 如果展开后与原函数相同，则返回 False，表示未能展开
    if expanded == func:  # it didn't expand
        return False

    # 检查展开后的函数与原函数在随机复数取值下是否具有相同的数值
    subs = {}
    for a in func.free_symbols:
        subs[a] = randcplx()
    return abs(func.subs(subs).n()
               - expanded.replace(exp_polar, exp).subs(subs).n()) < 1e-10


def test_polylog_expansion():
    # 检查 polylog 函数在 s = 1, z = 0 时的计算结果是否为 0
    assert polylog(s, 0) == 0
    # 检查 polylog 函数在 s = 1, z = 1 时的计算结果是否为 zeta(s)
    assert polylog(s, 1) == zeta(s)
    # 检查 polylog 函数在 s = 1, z = -1 时的计算结果是否为 -dirichlet_eta(s)
    assert polylog(s, -1) == -dirichlet_eta(s)
    # 检查 polylog 函数在 s 和 exp_polar(I*pi*Rational(4, 3)) 时的计算结果是否与 exp(I*pi*Rational(4, 3)) 替换后的 polylog 函数
    # 断言：检查 polylog(1, z) 的泰勒级数展开是否等于 z + z**2/2 + z**3/3 + z**4/4 + O(z**5)
    assert polylog(1, z).series(z, n=5) == z + z**2/2 + z**3/3 + z**4/4 + O(z**5)
    
    # 断言：检查 polylog(1, sqrt(z)) 的泰勒级数展开是否等于 z/2 + z**2/4 + sqrt(z) + z**(S(3)/2)/3 + z**(S(5)/2)/5 + O(z**3)
    assert polylog(1, sqrt(z)).series(z, n=3) == z/2 + z**2/4 + sqrt(z) + z**(S(3)/2)/3 + z**(S(5)/2)/5 + O(z**3)

    # 断言：检查 polylog(S(3)/2, -z) 的泰勒级数展开是否等于 -z + sqrt(2)*z**2/4 - sqrt(3)*z**3/9 + z**4/8 + O(z**5)
    # 这个断言解决了 https://github.com/sympy/sympy/issues/9497 的问题
    assert polylog(S(3)/2, -z).series(z, 0, 5) == -z + sqrt(2)*z**2/4 - sqrt(3)*z**3/9 + z**4/8 + O(z**5)
# 测试一个特定问题，确保符号 'i' 是一个整数
def test_issue_8404():
    # 创建一个整数符号 'i'
    i = Symbol('i', integer=True)
    # 断言一个级数的绝对值与特定值的差异小于指定精度
    assert Abs(Sum(1/(3*i + 1)**2, (i, 0, S.Infinity)).doit().n(4)
        - 1.122) < 0.001


# 测试 polylog 函数的不同值
def test_polylog_values():
    # 断言 polylog 函数的值与预期的值相等
    assert polylog(2, 2) == pi**2/4 - I*pi*log(2)
    assert polylog(2, S.Half) == pi**2/12 - log(2)**2/2
    # 对一系列复数 z 进行测试，确保数值近似误差在给定阈值之内
    for z in [S.Half, 2, (sqrt(5)-1)/2, -(sqrt(5)-1)/2, -(sqrt(5)+1)/2, (3-sqrt(5))/2]:
        assert Abs(polylog(2, z).evalf() - polylog(2, z, evaluate=False).evalf()) < 1e-15
    # 对一个符号 'z' 进行测试，使用数值方法验证 polylog 函数的数值结果
    z = Symbol("z")
    for s in [-1, 0]:
        for _ in range(10):
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=-3, b=-2, c=S.Half, d=2)
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=2, b=-2, c=5, d=2)

    # 从 sympy.integrals.integrals 模块导入 Integral 类
    assert polylog(0, Integral(1, (x, 0, 1))) == -S.Half


# 测试 lerchphi 函数的展开
def test_lerchphi_expansion():
    # 断言 lerchphi 函数的展开结果符合预期
    assert myexpand(lerchphi(1, s, a), zeta(s, a))
    assert myexpand(lerchphi(z, s, 1), polylog(s, z)/z)

    # 直接求和方法
    assert myexpand(lerchphi(z, -1, a), a/(1 - z) + z/(1 - z)**2)
    assert myexpand(lerchphi(z, -3, a), None)
    # polylog 函数的约简结果
    assert myexpand(lerchphi(z, s, S.Half),
                    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z)
                              - polylog(s, polar_lift(-1)*sqrt(z))/sqrt(z)))
    assert myexpand(lerchphi(z, s, 2), -1/z + polylog(s, z)/z**2)
    assert myexpand(lerchphi(z, s, Rational(3, 2)), None)
    assert myexpand(lerchphi(z, s, Rational(7, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-1, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-5, 2)), None)

    # hurwitz zeta 函数的约简结果
    assert myexpand(lerchphi(-1, s, a),
                    2**(-s)*zeta(s, a/2) - 2**(-s)*zeta(s, (a + 1)/2))
    assert myexpand(lerchphi(I, s, a), None)
    assert myexpand(lerchphi(-I, s, a), None)
    assert myexpand(lerchphi(exp(I*pi*Rational(2, 5)), s, a), None)


# 测试 stieltjes 函数
def test_stieltjes():
    # 断言 stieltjes 函数返回的对象类型是 stieltjes
    assert isinstance(stieltjes(x), stieltjes)
    assert isinstance(stieltjes(x, a), stieltjes)

    # 零阶常数 EulerGamma
    assert stieltjes(0) == S.EulerGamma
    assert stieltjes(0, 1) == S.EulerGamma

    # 未定义的情况
    assert stieltjes(nan) is nan
    assert stieltjes(0, nan) is nan
    assert stieltjes(-1) is S.ComplexInfinity
    assert stieltjes(1.5) is S.ComplexInfinity
    assert stieltjes(z, 0) is S.ComplexInfinity
    assert stieltjes(z, -1) is S.ComplexInfinity


# 测试 stieltjes 函数的数值求解
def test_stieltjes_evalf():
    # 断言 stieltjes 函数的数值结果与预期的误差在给定精度之内
    assert abs(stieltjes(0).evalf() - 0.577215664) < 1E-9
    assert abs(stieltjes(0, 0.5).evalf() - 1.963510026) < 1E-9
    assert abs(stieltjes(1, 2).evalf() + 0.072815845) < 1E-9


# 测试一个特定问题，确保符号 'a', 'b', 's' 的属性满足预期
def test_issue_10475():
    a = Symbol('a', extended_real=True)
    b = Symbol('b', extended_positive=True)
    s = Symbol('s', zero=False)

    # 断言 zeta 函数的某些参数是有限的
    assert zeta(2 + I).is_finite
    assert zeta(1).is_finite is False
    assert zeta(x).is_finite is None
    # 断言检查 zeta(x + I) 的有限性，预期返回 None
    assert zeta(x + I).is_finite is None
    
    # 断言检查 zeta(a) 的有限性，预期返回 None
    assert zeta(a).is_finite is None
    
    # 断言检查 zeta(b) 的有限性，预期返回 None
    assert zeta(b).is_finite is None
    
    # 断言检查 zeta(-b) 的有限性，预期返回 True
    assert zeta(-b).is_finite is True
    
    # 断言检查 zeta(b**2 - 2*b + 1) 的有限性，预期返回 None
    assert zeta(b**2 - 2*b + 1).is_finite is None
    
    # 断言检查 zeta(a + I) 的有限性，预期返回 True
    assert zeta(a + I).is_finite is True
    
    # 断言检查 zeta(b + 1) 的有限性，预期返回 True
    assert zeta(b + 1).is_finite is True
    
    # 断言检查 zeta(s + 1) 的有限性，预期返回 True
    assert zeta(s + 1).is_finite is True
# 定义一个测试函数，用于测试与 issue 14177 相关的功能

def test_issue_14177():
    # 创建一个非负整数符号 n
    n = Symbol('n', nonnegative=True, integer=True)
    
    # 断言：关于 zeta 函数的变换为 Bernoulli 表示式
    assert zeta(-n).rewrite(bernoulli) == bernoulli(n+1) / (-n-1)
    # 断言：带有额外参数 a 的 zeta 函数的 Bernoulli 变换
    assert zeta(-n, a).rewrite(bernoulli) == bernoulli(n+1, a) / (-n-1)
    
    # 计算 zeta(2*n) 的 Bernoulli 变换
    z2n = -(2*I*pi)**(2*n)*bernoulli(2*n) / (2*factorial(2*n))
    assert zeta(2*n).rewrite(bernoulli) == z2n
    
    # 展开 zeta 函数中的 n+1 参数
    assert expand_func(zeta(s, n+1)) == zeta(s) - harmonic(n, s)
    
    # 展开具有负指数参数的 zeta 函数，应该返回 nan
    assert expand_func(zeta(-b, -n)) is nan
    
    # 展开具有负指数参数的 zeta 函数
    assert expand_func(zeta(-b, n)) == zeta(-b, n)
    
    # 重新定义 n 为一个一般符号（不限制为非负整数）
    n = Symbol('n')
    
    # 断言：对于 zeta(2*n)，左右两边应该相等，因为 z 的符号未确定
    assert zeta(2*n) == zeta(2*n) # As sign of z (= 2*n) is not determined
```