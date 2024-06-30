# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_trigonometric.py`

```
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
                                                      cos, cot, csc, sec, sin, sinc, tan)
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError, PoleError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises


# 定义符号变量
x, y, z = symbols('x y z')
# 定义实数符号变量
r = Symbol('r', real=True)
# 定义整数符号变量
k, m = symbols('k m', integer=True)
# 定义正数符号变量
p = Symbol('p', positive=True)
# 定义负数符号变量
n = Symbol('n', negative=True)
# 定义非正数符号变量
np = Symbol('p', nonpositive=True)
# 定义非负数符号变量
nn = Symbol('n', nonnegative=True)
# 定义非零符号变量
nz = Symbol('nz', nonzero=True)
# 定义扩展正数符号变量
ep = Symbol('ep', extended_positive=True)
# 定义扩展负数符号变量
en = Symbol('en', extended_negative=True)
# 定义扩展非正数符号变量
enp = Symbol('ep', extended_nonpositive=True)
# 定义扩展非负数符号变量
enn = Symbol('en', extended_nonnegative=True)
# 定义扩展非零符号变量
enz = Symbol('enz', extended_nonzero=True)
# 定义代数数符号变量
a = Symbol('a', algebraic=True)
# 定义非零代数数符号变量
na = Symbol('na', nonzero=True, algebraic=True)


# 定义测试函数 test_sin
def test_sin():
    # 定义符号变量
    x, y = symbols('x y')
    # 定义虚数符号变量
    z = symbols('z', imaginary=True)

    # 断言 sin 函数参数个数为 1
    assert sin.nargs == FiniteSet(1)
    # 断言 sin(nan) 返回 nan
    assert sin(nan) is nan
    # 断言 sin(zoo) 返回 nan
    assert sin(zoo) is nan

    # 断言 sin(oo) 返回累积界 [-1, 1]
    assert sin(oo) == AccumBounds(-1, 1)
    # 断言 sin(oo) - sin(oo) 返回累积界 [-2, 2]
    assert sin(oo) - sin(oo) == AccumBounds(-2, 2)
    # 断言 sin(oo*I) 返回 oo*I
    assert sin(oo*I) == oo*I
    # 断言 sin(-oo*I) 返回 -oo*I
    assert sin(-oo*I) == -oo*I
    # 断言 0*sin(oo) 返回 S.Zero
    assert 0*sin(oo) is S.Zero
    # 断言 0/sin(oo) 返回 S.Zero
    assert 0/sin(oo) is S.Zero
    # 断言 0 + sin(oo) 返回累积界 [-1, 1]
    assert 0 + sin(oo) == AccumBounds(-1, 1)
    # 断言 5 + sin(oo) 返回累积界 [4, 6]
    assert 5 + sin(oo) == AccumBounds(4, 6)

    # 断言 sin(0) 返回 0
    assert sin(0) == 0

    # 断言 sin(z*I) 返回 I*sinh(z)
    assert sin(z*I) == I*sinh(z)
    # 断言 sin(asin(x)) 返回 x
    assert sin(asin(x)) == x
    # 断言 sin(atan(x)) 返回 x / sqrt(1 + x**2)
    assert sin(atan(x)) == x / sqrt(1 + x**2)
    # 断言 sin(acos(x)) 返回 sqrt(1 - x**2)
    assert sin(acos(x)) == sqrt(1 - x**2)
    # 断言 sin(acot(x)) 返回 1 / (sqrt(1 + 1 / x**2) * x)
    assert sin(acot(x)) == 1 / (sqrt(1 + 1 / x**2) * x)
    # 断言 sin(acsc(x)) 返回 1 / x
    assert sin(acsc(x)) == 1 / x
    # 断言 sin(asec(x)) 返回 sqrt(1 - 1 / x**2)
    assert sin(asec(x)) == sqrt(1 - 1 / x**2)
    # 断言：根据正弦函数和反正切函数的关系验证
    assert sin(atan2(y, x)) == y / sqrt(x**2 + y**2)

    # 断言：正弦函数与复数π乘以虚数i的双曲正弦函数的关系验证
    assert sin(pi*I) == sinh(pi)*I
    assert sin(-pi*I) == -sinh(pi)*I
    assert sin(-2*I) == -sinh(2)*I

    # 断言：正弦函数的基本性质验证
    assert sin(pi) == 0
    assert sin(-pi) == 0
    assert sin(2*pi) == 0
    assert sin(-2*pi) == 0
    assert sin(-3*10**73*pi) == 0
    assert sin(7*10**103*pi) == 0

    # 断言：正弦函数在特定角度下的值验证
    assert sin(pi/2) == 1
    assert sin(-pi/2) == -1
    assert sin(pi*Rational(5, 2)) == 1
    assert sin(pi*Rational(7, 2)) == -1

    # 符号定义和正弦函数的关系验证
    ne = symbols('ne', integer=True, even=False)
    e = symbols('e', even=True)
    assert sin(pi*ne/2) == (-1)**(ne/2 - S.Half)
    assert sin(pi*k/2).func == sin
    assert sin(pi*e/2) == 0
    assert sin(pi*k) == 0
    assert sin(pi*k).subs(k, 3) == sin(pi*k/2).subs(k, 6)  # issue 8298

    # 正弦函数在特定角度下的值验证
    assert sin(pi/3) == S.Half*sqrt(3)
    assert sin(pi*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)

    assert sin(pi/4) == S.Half*sqrt(2)
    assert sin(-pi/4) == Rational(-1, 2)*sqrt(2)
    assert sin(pi*Rational(17, 4)) == S.Half*sqrt(2)
    assert sin(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert sin(pi/6) == S.Half
    assert sin(-pi/6) == Rational(-1, 2)
    assert sin(pi*Rational(7, 6)) == Rational(-1, 2)
    assert sin(pi*Rational(-5, 6)) == Rational(-1, 2)

    # 正弦函数在特定角度下的值验证
    assert sin(pi*Rational(1, 5)) == sqrt((5 - sqrt(5)) / 8)
    assert sin(pi*Rational(2, 5)) == sqrt((5 + sqrt(5)) / 8)
    assert sin(pi*Rational(3, 5)) == sin(pi*Rational(2, 5))
    assert sin(pi*Rational(4, 5)) == sin(pi*Rational(1, 5))
    assert sin(pi*Rational(6, 5)) == -sin(pi*Rational(1, 5))
    assert sin(pi*Rational(8, 5)) == -sin(pi*Rational(2, 5))

    assert sin(pi*Rational(-1273, 5)) == -sin(pi*Rational(2, 5))

    # 正弦函数在特定角度下的值验证
    assert sin(pi/8) == sqrt((2 - sqrt(2))/4)

    assert sin(pi/10) == Rational(-1, 4) + sqrt(5)/4

    assert sin(pi/12) == -sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(5, 12)) == sqrt(2)/4 + sqrt(6)/4
    assert sin(pi*Rational(-7, 12)) == -sqrt(2)/4 - sqrt(6)/4
    assert sin(pi*Rational(-11, 12)) == sqrt(2)/4 - sqrt(6)/4

    # 正弦函数在特定角度下的值验证
    assert sin(pi*Rational(104, 105)) == sin(pi/105)
    assert sin(pi*Rational(106, 105)) == -sin(pi/105)

    assert sin(pi*Rational(-104, 105)) == -sin(pi/105)
    assert sin(pi*Rational(-106, 105)) == sin(pi/105)

    # 断言：正弦函数与虚数乘积的关系验证
    assert sin(x*I) == sinh(x)*I

    # 断言：正弦函数在整数倍π时的值验证
    assert sin(k*pi) == 0
    assert sin(17*k*pi) == 0
    assert sin(2*k*pi + 4) == sin(4)
    assert sin(2*k*pi + m*pi + 1) == (-1)**(m + 2*k)*sin(1)

    # 断言：正弦函数在整数倍π乘以虚数i时的值验证
    assert sin(k*pi*I) == sinh(k*pi)*I

    # 断言：正弦函数的实部为实数
    assert sin(r).is_real is True

    # 断言：正弦函数的代数性质验证
    assert sin(0, evaluate=False).is_algebraic
    assert sin(a).is_algebraic is None
    assert sin(na).is_algebraic is False
    q = Symbol('q', rational=True)
    assert sin(pi*q).is_algebraic
    qn = Symbol('qn', rational=True, nonzero=True)
    assert sin(qn).is_rational is False
    assert sin(q).is_rational is None  # issue 8653

    # 断言：正弦函数与实部和虚部的关系验证
    assert isinstance(sin( re(x) - im(y)), sin) is True
    assert isinstance(sin(-re(x) + im(y)), sin) is False
    # 断言，验证 SetExpr(Interval(0, 1)) 的正弦值是否等于 ImageSet(Lambda(x, sin(x)), Interval(0, 1)) 中的表达式
    assert sin(SetExpr(Interval(0, 1))) == SetExpr(ImageSet(Lambda(x, sin(x)),
                       Interval(0, 1)))
    
    # 嵌套循环，对于每个 d 在列表 [1, 2, ..., 21, 60, 85] 中，以及每个 n 在范围 [0, 2*d] 中
    # 计算 x = n*pi/d，然后计算 sin(x) 的浮点值与 sin(float(x)) 的差的绝对值 e
    # 断言 e 小于 1e-12
    for d in list(range(1, 22)) + [60, 85]:
        for n in range(d*2 + 1):
            x = n*pi/d
            e = abs( float(sin(x)) - sin(float(x)) )
            assert e < 1e-12
    
    # 断言 sin(0, evaluate=False).is_zero 的值为 True
    assert sin(0, evaluate=False).is_zero is True
    
    # 断言 sin(k*pi, evaluate=False).is_zero 的值为 True
    assert sin(k*pi, evaluate=False).is_zero is True
    
    # 断言 sin(Add(1, -1, evaluate=False), evaluate=False).is_zero 的值为 True
    assert sin(Add(1, -1, evaluate=False), evaluate=False).is_zero is True
python
def test_sin_cos():
    # 循环遍历不同的角度值和范围，进行正弦和余弦的数学断言测试
    for d in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 60, 120]:  # list is not exhaustive...
        for n in range(-2*d, d*2):
            # 计算角度 x，并进行数学断言测试
            x = n*pi/d
            # 断言正弦函数加上 π/2 的角度等于余弦函数
            assert sin(x + pi/2) == cos(x), "fails for %d*pi/%d" % (n, d)
            # 断言正弦函数减去 π/2 的角度等于负余弦函数
            assert sin(x - pi/2) == -cos(x), "fails for %d*pi/%d" % (n, d)
            # 断言正弦函数等于余弦函数减去 π/2 的角度
            assert sin(x) == cos(x - pi/2), "fails for %d*pi/%d" % (n, d)
            # 断言负正弦函数等于余弦函数加上 π/2 的角度
            assert -sin(x) == cos(x + pi/2), "fails for %d*pi/%d" % (n, d)


def test_sin_series():
    # 断言正弦函数在级数展开中的表达式
    assert sin(x).series(x, 0, 9) == \
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**9)


def test_sin_rewrite():
    # 断言正弦函数按照不同的函数重写规则得到的结果
    assert sin(x).rewrite(exp) == -I*(exp(I*x) - exp(-I*x))/2
    assert sin(x).rewrite(tan) == 2*tan(x/2)/(1 + tan(x/2)**2)
    assert sin(x).rewrite(cot) == \
        Piecewise((0, Eq(im(x), 0) & Eq(Mod(x, pi), 0)),
                  (2*cot(x/2)/(cot(x/2)**2 + 1), True))
    assert sin(sinh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sinh(3)).n()
    assert sin(cosh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cosh(3)).n()
    assert sin(tanh(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tanh(3)).n()
    assert sin(coth(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, coth(3)).n()
    assert sin(sin(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, sin(3)).n()
    assert sin(cos(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cos(3)).n()
    assert sin(tan(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, tan(3)).n()
    assert sin(cot(x)).rewrite(
        exp).subs(x, 3).n() == sin(x).rewrite(exp).subs(x, cot(3)).n()
    assert sin(log(x)).rewrite(Pow) == I*x**-I / 2 - I*x**I /2
    assert sin(x).rewrite(csc) == 1/csc(x)
    assert sin(x).rewrite(cos) == cos(x - pi / 2, evaluate=False)
    assert sin(x).rewrite(sec) == 1 / sec(x - pi / 2, evaluate=False)
    assert sin(cos(x)).rewrite(Pow) == sin(cos(x))
    assert sin(x).rewrite(besselj) == sqrt(pi*x/2)*besselj(S.Half, x)
    assert sin(x).rewrite(besselj).subs(x, 0) == sin(0)


def _test_extrig(f, i, e):
    from sympy.core.function import expand_trig
    # 断言三角函数扩展不变性和展开后的结果
    assert unchanged(f, i)
    assert expand_trig(f(i)) == f(i)
    # 直接测试而不是使用 .expand(trig=True)，因为其他的展开方式会取消未评估的 Mul
    assert expand_trig(f(Mul(i, 1, evaluate=False))) == e
    assert abs(f(i) - e).n() < 1e-10


def test_sin_expansion():
    # 断言正弦函数在不同展开情况下的表达式
    # 注意：这些公式不是唯一的，这里使用的来自切比雪夫公式
    assert sin(x + y).expand(trig=True) == sin(x)*cos(y) + cos(x)*sin(y)
    assert sin(x - y).expand(trig=True) == sin(x)*cos(y) - cos(x)*sin(y)
    assert sin(y - x).expand(trig=True) == cos(x)*sin(y) - sin(x)*cos(y)
    assert sin(2*x).expand(trig=True) == 2*sin(x)*cos(x)
    assert sin(3*x).expand(trig=True) == -4*sin(x)**3 + 3*sin(x)
    # 对 sin(4*x) 进行展开，验证其等于 -8*sin(x)**3*cos(x) + 4*sin(x)*cos(x)
    assert sin(4*x).expand(trig=True) == -8*sin(x)**3*cos(x) + 4*sin(x)*cos(x)
    
    # 对 sin(2*pi/17) 进行展开，验证其等于 sin(2*pi/17, evaluate=False)
    assert sin(2*pi/17).expand(trig=True) == sin(2*pi/17, evaluate=False)
    
    # 对 sin(x+pi/17) 进行展开，验证其等于 sin(pi/17)*cos(x) + cos(pi/17)*sin(x)
    assert sin(x+pi/17).expand(trig=True) == sin(pi/17)*cos(x) + cos(pi/17)*sin(x)
    
    # 使用 _test_extrig 函数测试 sin 函数的额外三角函数性质
    _test_extrig(sin, 2, 2*sin(1)*cos(1))
    _test_extrig(sin, 3, -4*sin(1)**3 + 3*sin(1))
# 定义一个测试函数，用于验证 sin 函数在 AccumBounds 参数下的行为
def test_sin_AccumBounds():
    # 验证 sin 函数对无穷区间 [-oo, oo] 的计算结果是否为 [-1, 1]
    assert sin(AccumBounds(-oo, oo)) == AccumBounds(-1, 1)
    # 验证 sin 函数对区间 [0, oo] 的计算结果是否为 [-1, 1]
    assert sin(AccumBounds(0, oo)) == AccumBounds(-1, 1)
    # 验证 sin 函数对区间 [-oo, 0] 的计算结果是否为 [-1, 1]
    assert sin(AccumBounds(-oo, 0)) == AccumBounds(-1, 1)
    # 验证 sin 函数对区间 [0, 2*pi] 的计算结果是否为 [-1, 1]
    assert sin(AccumBounds(0, 2*S.Pi)) == AccumBounds(-1, 1)
    # 验证 sin 函数对区间 [0, pi*3/4] 的计算结果是否为 [0, 1]
    assert sin(AccumBounds(0, S.Pi*Rational(3, 4))) == AccumBounds(0, 1)
    # 验证 sin 函数对区间 [pi*3/4, pi*7/4] 的计算结果是否为 [-1, sin(pi*3/4)]
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(7, 4))) == AccumBounds(-1, sin(S.Pi*Rational(3, 4)))
    # 验证 sin 函数对区间 [pi/4, pi/3] 的计算结果是否为 [sin(pi/4), sin(pi/3)]
    assert sin(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(sin(S.Pi/4), sin(S.Pi/3))
    # 验证 sin 函数对区间 [pi*3/4, pi*5/6] 的计算结果是否为 [sin(pi*5/6), sin(pi*3/4)]
    assert sin(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 6))) == AccumBounds(sin(S.Pi*Rational(5, 6)), sin(S.Pi*Rational(3, 4)))


# 定义一个测试函数，用于验证 sin 函数的微分结果
def test_sin_fdiff():
    # 验证 sin 函数的一阶导数是否为 cos 函数
    assert sin(x).fdiff() == cos(x)
    # 验证 sin 函数的二阶及以上导数会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: sin(x).fdiff(2))


# 定义一个测试函数，用于验证三角函数的对称性质
def test_trig_symmetry():
    # 验证 sin 函数的奇函数性质
    assert sin(-x) == -sin(x)
    # 验证 cos 函数的偶函数性质
    assert cos(-x) == cos(x)
    # 验证 tan 函数的奇函数性质
    assert tan(-x) == -tan(x)
    # 验证 cot 函数的奇函数性质
    assert cot(-x) == -cot(x)
    # 验证 sin 函数的周期性质
    assert sin(x + pi) == -sin(x)
    assert sin(x + 2*pi) == sin(x)
    assert sin(x + 3*pi) == -sin(x)
    assert sin(x + 4*pi) == sin(x)
    assert sin(x - 5*pi) == -sin(x)
    # 验证 cos 函数的周期性质
    assert cos(x + pi) == -cos(x)
    assert cos(x + 2*pi) == cos(x)
    assert cos(x + 3*pi) == -cos(x)
    assert cos(x + 4*pi) == cos(x)
    assert cos(x - 5*pi) == -cos(x)
    # 验证 tan 函数的周期性质
    assert tan(x + pi) == tan(x)
    assert tan(x - 3*pi) == tan(x)
    # 验证 cot 函数的周期性质
    assert cot(x + pi) == cot(x)
    assert cot(x - 3*pi) == cot(x)
    # 验证 sin 函数的余角性质
    assert sin(pi/2 - x) == cos(x)
    assert sin(pi*Rational(3, 2) - x) == -cos(x)
    assert sin(pi*Rational(5, 2) - x) == cos(x)
    # 验证 cos 函数的余角性质
    assert cos(pi/2 - x) == sin(x)
    assert cos(pi*Rational(3, 2) - x) == -sin(x)
    assert cos(pi*Rational(5, 2) - x) == sin(x)
    # 验证 tan 函数的余角性质
    assert tan(pi/2 - x) == cot(x)
    assert tan(pi*Rational(3, 2) - x) == cot(x)
    assert tan(pi*Rational(5, 2) - x) == cot(x)
    # 验证 cot 函数的余角性质
    assert cot(pi/2 - x) == tan(x)
    assert cot(pi*Rational(3, 2) - x) == tan(x)
    assert cot(pi*Rational(5, 2) - x) == tan(x)
    # 验证 sin 函数的补角性质
    assert sin(pi/2 + x) == cos(x)
    # 验证 cos 函数的补角性质
    assert cos(pi/2 + x) == -sin(x)
    # 验证 tan 函数的补角性质
    assert tan(pi/2 + x) == -cot(x)
    # 验证 cot 函数的补角性质
    assert cot(pi/2 + x) == -tan(x)


# 定义一个测试函数，用于验证 cos 函数的各种输入情况
def test_cos():
    x, y = symbols('x y')

    # 验证 cos 函数的参数个数是否为 1
    assert cos.nargs == FiniteSet(1)
    # 验证 cos 函数对 nan 的处理结果为 nan
    assert cos(nan) is nan

    # 验证 cos 函数对无穷大的处理结果
    assert cos(oo) == AccumBounds(-1, 1)
    assert cos(oo) - cos(oo) == AccumBounds(-2, 2)
    assert cos(oo*I) is oo
    assert cos(-oo*I) is oo
    assert cos(zoo) is nan

    # 验证 cos 函数对特定角度的计算结果
    assert cos(0) == 1

    # 验证 cos 函数对反三角函数的性质
    assert cos(acos(x)) == x
    assert cos(atan(x)) == 1 / sqrt(1 + x**2)
    assert cos(asin(x)) == sqrt(1 - x**2)
    assert cos(acot(x)) == 1 / sqrt(1 + 1 / x**2)
    assert cos(acsc(x)) == sqrt(1 - 1 / x**2)
    assert cos(asec(x)) == 1 / x
    assert cos(atan2(y, x)) == x / sqrt(x**2 + y**2)

    # 验证 cos 函数对纯虚数的处理结果
    assert cos(pi*I) == cosh(pi)
    assert cos(-pi*I) == cosh(pi)
    assert cos(-2*I) == cosh(2)

    # 验证 cos 函数对常用角度的计算结果
    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos(pi/2) == 0
    assert cos(-pi/2) == 0
    assert cos((-3*10**73 + 1)*pi/2) == 0
    assert cos((7*10**103 + 1)*pi/2) == 0


这些注释提供了对每个测试函数及其语句的详细解释，确保了代码的可读性和理解性。
    # 定义一个整数且非偶数的符号变量 n
    n = symbols('n', integer=True, even=False)
    # 定义一个偶数的符号变量 e
    e = symbols('e', even=True)
    # 断言：当 n 是整数时，cos(pi*n/2) 应该等于 0
    assert cos(pi*n/2) == 0
    # 断言：当 e 是偶数时，cos(pi*e/2) 应该等于 (-1)**(e/2)
    assert cos(pi*e/2) == (-1)**(e/2)

    # 一些常见的角度值的余弦值断言
    assert cos(pi) == -1
    assert cos(-pi) == -1
    assert cos(2*pi) == 1
    assert cos(5*pi) == -1
    assert cos(8*pi) == 1

    # 对一些特定角度的余弦值进行断言
    assert cos(pi/3) == S.Half
    assert cos(pi*Rational(-2, 3)) == Rational(-1, 2)

    assert cos(pi/4) == S.Half*sqrt(2)
    assert cos(-pi/4) == S.Half*sqrt(2)
    assert cos(pi*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cos(pi*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert cos(pi/6) == S.Half*sqrt(3)
    assert cos(-pi/6) == S.Half*sqrt(3)
    assert cos(pi*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cos(pi*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    # 对一些有理倍数角度的余弦值进行断言
    assert cos(pi*Rational(1, 5)) == (sqrt(5) + 1)/4
    assert cos(pi*Rational(2, 5)) == (sqrt(5) - 1)/4
    assert cos(pi*Rational(3, 5)) == -cos(pi*Rational(2, 5))
    assert cos(pi*Rational(4, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(6, 5)) == -cos(pi*Rational(1, 5))
    assert cos(pi*Rational(8, 5)) == cos(pi*Rational(2, 5))

    # 对一个大的负有理数倍数角度的余弦值进行断言
    assert cos(pi*Rational(-1273, 5)) == -cos(pi*Rational(2, 5))

    assert cos(pi/8) == sqrt((2 + sqrt(2))/4)

    assert cos(pi/12) == sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(5, 12)) == -sqrt(2)/4 + sqrt(6)/4
    assert cos(pi*Rational(7, 12)) == sqrt(2)/4 - sqrt(6)/4
    assert cos(pi*Rational(11, 12)) == -sqrt(2)/4 - sqrt(6)/4

    # 对有理倍数角度的余弦值进行断言
    assert cos(pi*Rational(104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(106, 105)) == -cos(pi/105)

    # 对一个大的负有理数倍数角度的余弦值进行断言
    assert cos(pi*Rational(-104, 105)) == -cos(pi/105)
    assert cos(pi*Rational(-106, 105)) == -cos(pi/105)

    # 对虚数参数的余弦值进行断言
    assert cos(x*I) == cosh(x)
    assert cos(k*pi*I) == cosh(k*pi)

    # 断言：cos(r) 的结果应为实数
    assert cos(r).is_real is True

    # 断言：cos(0, evaluate=False) 的结果应为代数的
    assert cos(0, evaluate=False).is_algebraic
    # 断言：cos(a) 的结果应为未知
    assert cos(a).is_algebraic is None
    # 断言：cos(na) 的结果应为非代数的
    assert cos(na).is_algebraic is False
    # 定义一个有理数符号变量 q
    q = Symbol('q', rational=True)
    # 断言：cos(pi*q) 的结果应为代数的
    assert cos(pi*q).is_algebraic
    # 断言：cos(pi*Rational(2, 7)) 的结果应为代数的
    assert cos(pi*Rational(2, 7)).is_algebraic

    # 对整数倍数角度的余弦值进行断言
    assert cos(k*pi) == (-1)**k
    assert cos(2*k*pi) == 1
    # 断言：cos(0, evaluate=False) 的结果不应为零
    assert cos(0, evaluate=False).is_zero is False
    # 断言：cos(Rational(1, 2)) 的结果不应为零
    assert cos(Rational(1, 2)).is_zero is False
    # 断言：当 asin(-1, evaluate=False) 的结果为零时，cos(asin(-1, evaluate=False), evaluate=False) 的结果应为 None
    assert cos(asin(-1, evaluate=False), evaluate=False).is_zero is None

    # 对一些特定角度下余弦函数的数值计算进行断言
    for d in list(range(1, 22)) + [60, 85]:
        for n in range(2*d + 1):
            x = n*pi/d
            e = abs( float(cos(x)) - cos(float(x)) )
            # 断言：数值计算误差应小于 1e-12
            assert e < 1e-12
# 测试函数，用于验证问题 #6190
def test_issue_6190():
    # 创建一个浮点数对象 c，表示一个非常大的浮点数
    c = Float('123456789012345678901234567890.25', '')
    # 遍历数学函数 sin, cos, tan, cot
    for cls in [sin, cos, tan, cot]:
        # 断言：数学函数应用于 c*pi 后的结果等于应用于 pi/4 后的结果
        assert cls(c*pi) == cls(pi/4)
        # 断言：数学函数应用于 4.125*pi 后的结果等于应用于 pi/8 后的结果
        assert cls(4.125*pi) == cls(pi/8)
        # 断言：数学函数应用于 4.7*pi 后的结果等于应用于 (4.7 % 2)*pi 后的结果
        assert cls(4.7*pi) == cls((4.7 % 2)*pi)


# 测试函数，验证余弦函数的级数展开
def test_cos_series():
    # 断言：cos(x) 在 x=0 处展开到 9 阶的级数
    assert cos(x).series(x, 0, 9) == \
        1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**9)


# 测试函数，验证余弦函数的重写
def test_cos_rewrite():
    # 断言：cos(x) 通过指数重写成 exp(I*x)/2 + exp(-I*x)/2
    assert cos(x).rewrite(exp) == exp(I*x)/2 + exp(-I*x)/2
    # 断言：cos(x) 通过 tan(x) 重写成 (1 - tan(x/2)**2)/(1 + tan(x/2)**2)
    assert cos(x).rewrite(tan) == (1 - tan(x/2)**2)/(1 + tan(x/2)**2)
    # 断言：cos(x) 通过 cot(x) 重写成 Piecewise((1, Eq(im(x), 0) & Eq(Mod(x, 2*pi), 0)),
    #                                      ((cot(x/2)**2 - 1)/(cot(x/2)**2 + 1), True))
    assert cos(x).rewrite(cot) == \
        Piecewise((1, Eq(im(x), 0) & Eq(Mod(x, 2*pi), 0)),
                  ((cot(x/2)**2 - 1)/(cot(x/2)**2 + 1), True))
    # 断言：cos(sinh(x)) 通过 exp 重写后，取 x=3，再数值化的结果应等于 cos(x) 通过 exp 重写，取 x=sinh(3)，再数值化的结果
    assert cos(sinh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sinh(3)).n()
    # 同上，验证 cos(cosh(x)), cos(tanh(x)), cos(coth(x)), cos(sin(x)), cos(cos(x)), cos(tan(x)), cos(cot(x)) 的重写和数值化结果
    assert cos(cosh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cosh(3)).n()
    assert cos(tanh(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tanh(3)).n()
    assert cos(coth(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, coth(3)).n()
    assert cos(sin(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, sin(3)).n()
    assert cos(cos(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cos(3)).n()
    assert cos(tan(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, tan(3)).n()
    assert cos(cot(x)).rewrite(
        exp).subs(x, 3).n() == cos(x).rewrite(exp).subs(x, cot(3)).n()
    # 断言：cos(log(x)) 通过 Pow 重写成 x**I/2 + x**-I/2
    assert cos(log(x)).rewrite(Pow) == x**I/2 + x**-I/2
    # 断言：cos(x) 通过 sec(x) 重写成 1/sec(x)
    assert cos(x).rewrite(sec) == 1/sec(x)
    # 断言：cos(x) 通过 sin(x) 重写成 sin(x + pi/2, evaluate=False)
    assert cos(x).rewrite(sin) == sin(x + pi/2, evaluate=False)
    # 断言：cos(x) 通过 csc(x) 重写成 1/csc(-x + pi/2, evaluate=False)
    assert cos(x).rewrite(csc) == 1/csc(-x + pi/2, evaluate=False)
    # 断言：cos(sin(x)) 通过 Pow 重写后应与原始表达式相等
    assert cos(sin(x)).rewrite(Pow) == cos(sin(x))
    # 断言：cos(x) 通过 besselj 重写成 Piecewise(...), Ne(x, 0)
    assert cos(x).rewrite(besselj) == Piecewise(
                (sqrt(pi*x/2)*besselj(-S.Half, x), Ne(x, 0)),
                (1, True)
            )
    # 断言：cos(x) 通过 besselj 重写后，取 x=0 的结果应等于 cos(0)
    assert cos(x).rewrite(besselj).subs(x, 0) == cos(0)


# 测试函数，验证余弦函数的展开
def test_cos_expansion():
    # 断言：cos(x + y) 在 trig=True 情况下展开为 cos(x)*cos(y) - sin(x)*sin(y)
    assert cos(x + y).expand(trig=True) == cos(x)*cos(y) - sin(x)*sin(y)
    # 断言：cos(x - y) 展开为 cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(x - y).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    # 断言：cos(y - x) 展开为 cos(x)*cos(y) + sin(x)*sin(y)
    assert cos(y - x).expand(trig=True) == cos(x)*cos(y) + sin(x)*sin(y)
    # 断言：cos(2*x) 展开为 2*cos(x)**2 - 1
    assert cos(2*x).expand(trig=True) == 2*cos(x)**2 - 1
    # 断言：cos(3*x) 展开为 4*cos(x)**3 - 3*cos(x)
    assert cos(3*x).expand(trig=True) == 4*cos(x)**3 - 3*cos(x)
    # 断言：cos(4*x) 展开为 8*cos(x)**4 - 8*cos(x)**2 + 1
    assert cos(4*x).expand(trig=True) == 8*cos(x)**4 - 8*cos(x)**2 + 1
    # 断言：cos(2*pi/17) 在 trig=True 情况下展开为 cos(2*pi/17, evaluate=False)
    assert cos(2*pi/17).expand(trig=True) == cos(2*pi/17, evaluate=False)
    # 断言：cos(x+pi/17) 在 trig=True 情况下展开为 cos(pi/17)*cos(x) - sin(pi/17)*sin(x)
    assert cos(x+pi/17).expand(trig=True) == cos(pi/17)*cos(x) - sin(pi/17)*sin(x)
    # 辅助函数 _test_extrig 的测试，验证 cos(1) 的平方乘以 2，减去 1 的结果
    _test_extrig(cos, 2, 2*cos(1)**2 - 1)
    # 辅助函数 _test_extrig 的测试，验证 cos(1) 的立方乘以 4，减去 3*cos(1) 的结果
    _test_extrig(cos, 3, 4*cos(1)**3 - 3*cos(1))


# 测试函数，验证余弦函数在 AccumBounds 上的计算
def test_cos_AccumBounds():
    # 断言：cos(AccumBounds(-oo, oo)) 应等于 AccumBounds(-1, 1)
    assert cos(AccumBounds(-oo, oo)) == AccumBounds
    # 断言：对余弦函数在区间内的累积边界进行计算，并验证结果是否符合预期
    assert cos(AccumBounds(-S.Pi/3, S.Pi/4)) == AccumBounds(cos(-S.Pi/3), 1)
    
    # 断言：对余弦函数在区间内的累积边界进行计算，并验证结果是否符合预期
    assert cos(AccumBounds(S.Pi*Rational(3, 4), S.Pi*Rational(5, 4))) == AccumBounds(-1, cos(S.Pi*Rational(3, 4)))
    
    # 断言：对余弦函数在区间内的累积边界进行计算，并验证结果是否符合预期
    assert cos(AccumBounds(S.Pi*Rational(5, 4), S.Pi*Rational(4, 3))) == AccumBounds(cos(S.Pi*Rational(5, 4)), cos(S.Pi*Rational(4, 3)))
    
    # 断言：对余弦函数在区间内的累积边界进行计算，并验证结果是否符合预期
    assert cos(AccumBounds(S.Pi/4, S.Pi/3)) == AccumBounds(cos(S.Pi/3), cos(S.Pi/4))
# 定义一个函数用于测试余弦函数的一阶导数计算是否正确
def test_cos_fdiff():
    # 断言余弦函数的一阶导数应该等于负的正弦函数
    assert cos(x).fdiff() == -sin(x)
    # 断言调用余弦函数的第二阶导数会引发参数索引错误异常
    raises(ArgumentIndexError, lambda: cos(x).fdiff(2))

# 定义一个函数用于测试正切函数在不同参数下的结果
def test_tan():
    # 断言 tan(nan) 的结果应该是 nan
    assert tan(nan) is nan

    # 断言 tan(zoo) 的结果应该是 nan
    assert tan(zoo) is nan
    # 断言 tan(oo) 的结果应该是一个无穷区间 AccumBounds(-oo, oo)
    assert tan(oo) == AccumBounds(-oo, oo)
    # 断言 tan(oo) - tan(oo) 的结果应该是一个无穷区间 AccumBounds(-oo, oo)
    assert tan(oo) - tan(oo) == AccumBounds(-oo, oo)
    # 断言 tan.nargs 应该是一个有限集合 FiniteSet(1)
    assert tan.nargs == FiniteSet(1)
    # 断言 tan(oo*I) 的结果应该是复数单位虚数 I
    assert tan(oo*I) == I
    # 断言 tan(-oo*I) 的结果应该是负的复数单位虚数 -I

    assert tan(-oo*I) == -I

    # 断言 tan(0) 的结果应该是 0
    assert tan(0) == 0

    # 断言 tan(atan(x)) 的结果应该是 x
    assert tan(atan(x)) == x
    # 断言 tan(asin(x)) 的结果应该是 x / sqrt(1 - x**2)
    assert tan(asin(x)) == x / sqrt(1 - x**2)
    # 断言 tan(acos(x)) 的结果应该是 sqrt(1 - x**2) / x
    assert tan(acos(x)) == sqrt(1 - x**2) / x
    # 断言 tan(acot(x)) 的结果应该是 1 / x
    assert tan(acot(x)) == 1 / x
    # 断言 tan(acsc(x)) 的结果应该是 1 / (sqrt(1 - 1 / x**2) * x)
    assert tan(acsc(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    # 断言 tan(asec(x)) 的结果应该是 sqrt(1 - 1 / x**2) * x
    assert tan(asec(x)) == sqrt(1 - 1 / x**2) * x
    # 断言 tan(atan2(y, x)) 的结果应该是 y/x
    assert tan(atan2(y, x)) == y/x

    # 断言 tan(pi*I) 的结果应该是 tanh(pi)*I
    assert tan(pi*I) == tanh(pi)*I
    # 断言 tan(-pi*I) 的结果应该是 -tanh(pi)*I
    assert tan(-pi*I) == -tanh(pi)*I
    # 断言 tan(-2*I) 的结果应该是 -tanh(2)*I
    assert tan(-2*I) == -tanh(2)*I

    # 断言 tan(pi) 的结果应该是 0
    assert tan(pi) == 0
    # 断言 tan(-pi) 的结果应该是 0
    assert tan(-pi) == 0
    # 断言 tan(2*pi) 的结果应该是 0
    assert tan(2*pi) == 0
    # 断言 tan(-2*pi) 的结果应该是 0
    assert tan(-2*pi) == 0
    # 断言 tan(-3*10**73*pi) 的结果应该是 0
    assert tan(-3*10**73*pi) == 0

    # 断言 tan(pi/2) 的结果应该是 zoo
    assert tan(pi/2) is zoo
    # 断言 tan(pi*Rational(3, 2)) 的结果应该是 zoo
    assert tan(pi*Rational(3, 2)) is zoo

    # 断言 tan(pi/3) 的结果应该是 sqrt(3)
    assert tan(pi/3) == sqrt(3)
    # 断言 tan(pi*Rational(-2, 3)) 的结果应该是 sqrt(3)
    assert tan(pi*Rational(-2, 3)) == sqrt(3)

    # 断言 tan(pi/4) 的结果应该是 S.One
    assert tan(pi/4) is S.One
    # 断言 tan(-pi/4) 的结果应该是 S.NegativeOne
    assert tan(-pi/4) is S.NegativeOne
    # 断言 tan(pi*Rational(17, 4)) 的结果应该是 S.One
    assert tan(pi*Rational(17, 4)) is S.One
    # 断言 tan(pi*Rational(-3, 4)) 的结果应该是 S.One
    assert tan(pi*Rational(-3, 4)) is S.One

    # 断言 tan(pi/5) 的结果应该是 sqrt(5 - 2*sqrt(5))
    assert tan(pi/5) == sqrt(5 - 2*sqrt(5))
    # 断言 tan(pi*Rational(2, 5)) 的结果应该是 sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(2, 5)) == sqrt(5 + 2*sqrt(5))
    # 断言 tan(pi*Rational(18, 5)) 的结果应该是 -sqrt(5 + 2*sqrt(5))
    assert tan(pi*Rational(18, 5)) == -sqrt(5 + 2*sqrt(5))
    # 断言 tan(pi*Rational(-16, 5)) 的结果应该是 -sqrt(5 - 2*sqrt(5))
    assert tan(pi*Rational(-16, 5)) == -sqrt(5 - 2*sqrt(5))

    # 断言 tan(pi/6) 的结果应该是 1/sqrt(3)
    assert tan(pi/6) == 1/sqrt(3)
    # 断言 tan(-pi/6) 的结果应该是 -1/sqrt(3)
    assert tan(-pi/6) == -1/sqrt(3)
    # 断言 tan(pi*Rational(7, 6)) 的结果应该是 1/sqrt(3)
    assert tan(pi*Rational(7, 6)) == 1/sqrt(3)
    # 断言 tan(pi*Rational(-5, 6)) 的结果应该是 1/sqrt(3)
    assert tan(pi*Rational(-5, 6)) == 1/sqrt(3)

    # 断言 tan(pi/8) 的结果应该是 -1 + sqrt(2)
    assert tan(pi/8) == -1 + sqrt(2)
    # 断言 tan(pi*Rational(3, 8)) 的结果应该是 1 + sqrt(2)  # issue 15959
    assert tan(pi*Rational(3, 8)) == 1 + sqrt(2)
    # 断言 tan(pi*Rational(5, 8)) 的结果应该是 -1 - sqrt(2)
    assert tan(pi*Rational(5, 8)) == -1 - sqrt(2)
    # 断言 tan(pi*Rational(7, 8)) 的结果应该是 1 - sqrt(2)
    assert tan(pi*Rational(7, 8)) == 1 - sqrt(2)

    # 断言 tan(pi/10) 的结果应该是 sqrt(1 - 2*sqrt(5)/5)
    assert tan(pi/10) == sqrt(1 - 2*sqrt(5)/5)
    # 断言 tan(pi*Rational(3, 10)) 的结果应该是 sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(3, 10)) == sqrt(1 + 2*sqrt(5)/5)
    # 断言 tan(pi*Rational(17, 10)) 的结果应该是 -sqrt(1 + 2*sqrt(5)/5)
    assert tan(pi*Rational(17, 10)) == -sqrt(1 + 2*sqrt(5)/5)
    # 断言 tan(pi*Rational(-31, 10)) 的结果应该是 -sqrt(1 - 2*sqrt(5)/5)
    assert tan(pi*Rational(-31, 10)) == -sqrt(1 - 2*sqrt(5)/5)

    #
    # 断言：tan(k*pi*I) 应该等于 tanh(k*pi)*I
    assert tan(k*pi*I) == tanh(k*pi)*I

    # 断言：tan(r) 的实部为实数（不是确定的）
    assert tan(r).is_real is None
    # 断言：tan(r) 是扩展实数
    assert tan(r).is_extended_real is True

    # 断言：tan(0, evaluate=False) 是代数的
    assert tan(0, evaluate=False).is_algebraic
    # 断言：tan(a) 是否为代数的（不确定）
    assert tan(a).is_algebraic is None
    # 断言：tan(na) 不是代数的
    assert tan(na).is_algebraic is False

    # 断言：tan(pi*10/7) 应该等于 tan(pi*3/7)
    assert tan(pi*Rational(10, 7)) == tan(pi*Rational(3, 7))
    # 断言：tan(pi*11/7) 应该等于 -tan(pi*3/7)
    assert tan(pi*Rational(11, 7)) == -tan(pi*Rational(3, 7))
    # 断言：tan(pi*-11/7) 应该等于 tan(pi*3/7)
    assert tan(pi*Rational(-11, 7)) == tan(pi*Rational(3, 7))

    # 断言：tan(pi*15/14) 应该等于 tan(pi/14)
    assert tan(pi*Rational(15, 14)) == tan(pi/14)
    # 断言：tan(pi*-15/14) 应该等于 -tan(pi/14)
    assert tan(pi*Rational(-15, 14)) == -tan(pi/14)

    # 断言：tan(r) 是否是有限的（不确定）
    assert tan(r).is_finite is None
    # 断言：tan(I*r) 是有限的
    assert tan(I*r).is_finite is True

    # https://github.com/sympy/sympy/issues/21177
    # 创建函数 f，用于测试
    f = tan(pi*(x + S(3)/2))/(3*x)
    # 断言：f 的主导项在 x 上的系数应该为 -1/(3*pi*x**2)
    assert f.as_leading_term(x) == -1/(3*pi*x**2)
# 定义一个函数用于测试 tan 函数在不同条件下的级数展开是否正确
def test_tan_series():
    # 断言 tan(x) 在 x = 0 处展开到 x**9 的级数为 x + x**3/3 + 2*x**5/15 + 17*x**7/315 + O(x**9)
    assert tan(x).series(x, 0, 9) == \
        x + x**3/3 + 2*x**5/15 + 17*x**7/315 + O(x**9)


# 定义一个函数测试 tan 函数的重写操作
def test_tan_rewrite():
    # 定义复数指数和负数指数
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    # 断言 tan(x) 重写为 exp 函数后的结果
    assert tan(x).rewrite(exp) == I*(neg_exp - pos_exp)/(neg_exp + pos_exp)
    # 断言 tan(x) 重写为 sin 函数后的结果
    assert tan(x).rewrite(sin) == 2*sin(x)**2/sin(2*x)
    # 断言 tan(x) 重写为 cos 函数后的结果
    assert tan(x).rewrite(cos) == cos(x - S.Pi/2, evaluate=False)/cos(x)
    # 断言 tan(x) 重写为 cot 函数后的结果
    assert tan(x).rewrite(cot) == 1/cot(x)
    # 其余断言同理，依次测试 tan 函数重写为不同三角函数的结果
    assert tan(sinh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sinh(3)).n()
    assert tan(cosh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cosh(3)).n()
    assert tan(tanh(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tanh(3)).n()
    assert tan(coth(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, coth(3)).n()
    assert tan(sin(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, sin(3)).n()
    assert tan(cos(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cos(3)).n()
    assert tan(tan(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, tan(3)).n()
    assert tan(cot(x)).rewrite(exp).subs(x, 3).n() == tan(x).rewrite(exp).subs(x, cot(3)).n()
    # 断言 tan(x) 重写为 log 函数后的结果
    assert tan(log(x)).rewrite(Pow) == I*(x**-I - x**I)/(x**-I + x**I)
    # 断言 tan(x) 重写为 sec 函数后的结果
    assert tan(x).rewrite(sec) == sec(x)/sec(x - pi/2, evaluate=False)
    # 断言 tan(x) 重写为 csc 函数后的结果
    assert tan(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)/csc(x)
    # 断言 tan(sin(x)) 重写为 Pow 后的结果
    assert tan(sin(x)).rewrite(Pow) == tan(sin(x))
    # 断言 tan(x) 在给定条件下重写为 sqrt 函数后的结果
    assert tan(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == sqrt(sqrt(5)/8 +
               Rational(5, 8))/(Rational(-1, 4) + sqrt(5)/4)
    # 断言 tan(x) 重写为 besselj 函数后的结果
    assert tan(x).rewrite(besselj) == besselj(S.Half, x)/besselj(-S.Half, x)
    # 断言 tan(x) 在 x = 0 处重写为 besselj 函数的结果
    assert tan(x).rewrite(besselj).subs(x, 0) == tan(0)


# 定义一个函数测试 tan 函数的替换操作
def test_tan_subs():
    # 断言用 y 替换 tan(x) 后的结果应为 y
    assert tan(x).subs(tan(x), y) == y
    # 断言用 y 替换 x 后的 tan(x) 结果应为 tan(y)
    assert tan(x).subs(x, y) == tan(y)
    # 断言 tan(x) 在 x = pi/2 处的替换结果应为无穷大
    assert tan(x).subs(x, S.Pi/2) is zoo
    # 断言 tan(x) 在 x = 3*pi/2 处的替换结果应为无穷大
    assert tan(x).subs(x, S.Pi*Rational(3, 2)) is zoo


# 定义一个函数测试 tan 函数的展开操作
def test_tan_expansion():
    # 断言 tan(x+y) 在展开三角函数后的结果
    assert tan(x + y).expand(trig=True) == ((tan(x) + tan(y))/(1 - tan(x)*tan(y))).expand()
    # 断言 tan(x-y) 在展开三角函数后的结果
    assert tan(x - y).expand(trig=True) == ((tan(x) - tan(y))/(1 + tan(x)*tan(y))).expand()
    # 断言 tan(x+y+z) 在展开三角函数后的结果
    assert tan(x + y + z).expand(trig=True) == (
        (tan(x) + tan(y) + tan(z) - tan(x)*tan(y)*tan(z))/
        (1 - tan(x)*tan(y) - tan(x)*tan(z) - tan(y)*tan(z))).expand()
    # 断言 tan(2*x) 在展开三角函数后的结果重写为 tan 函数的结果
    assert 0 == tan(2*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 7))])*24 - 7
    # 断言 tan(3*x) 在展开三角函数后的结果重写为 tan 函数的结果
    assert 0 == tan(3*x).expand(trig=True).rewrite(tan).subs([(tan(x), Rational(1, 5))])*55 - 37
    # 断言：验证表达式结果为零
    assert 0 == (
        # 对 tan(4*x - pi/4) 进行三角恒等式展开和重新写为 tan 表达式
        tan(4*x - pi/4).expand(trig=True).rewrite(tan)
        # 将 tan(x) 替换为有理数 1/5
        .subs([(tan(x), Rational(1, 5))])
        # 乘以 239 并减去 1
        )*239 - 1
    
    # 调用测试函数 _test_extrig 进行测试，参数为 tan 函数、2 和 2*tan(1)/(1 - tan(1)**2)
    _test_extrig(tan, 2, 2*tan(1)/(1 - tan(1)**2))
    
    # 调用测试函数 _test_extrig 进行测试，参数为 tan 函数、3 和 (-tan(1)**3 + 3*tan(1))/(1 - 3*tan(1)**2)
    _test_extrig(tan, 3, (-tan(1)**3 + 3*tan(1))/(1 - 3*tan(1)**2))
def test_tan_AccumBounds():
    # 测试 tan 函数对 AccumBounds(-oo, oo) 的计算结果是否为 AccumBounds(-oo, oo)
    assert tan(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    # 测试 tan 函数对 AccumBounds(Pi/3, Pi*2/3) 的计算结果
    assert tan(AccumBounds(S.Pi/3, S.Pi*Rational(2, 3))) == AccumBounds(-oo, oo)
    # 测试 tan 函数对 AccumBounds(Pi/6, Pi/3) 的计算结果
    assert tan(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(tan(S.Pi/6), tan(S.Pi/3))


def test_tan_fdiff():
    # 断言 tan(x) 的一阶导数是否为 tan(x)^2 + 1
    assert tan(x).fdiff() == tan(x)**2 + 1
    # 断言 tan(x) 的二阶导数会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: tan(x).fdiff(2))


def test_cot():
    # 断言 cot(nan) 的结果是 nan
    assert cot(nan) is nan

    # 断言 cot 函数的参数个数是 1
    assert cot.nargs == FiniteSet(1)
    # 断言 cot(oo*I) 的结果是 -I
    assert cot(oo*I) == -I
    # 断言 cot(-oo*I) 的结果是 I
    assert cot(-oo*I) == I
    # 断言 cot(zoo) 的结果是 nan
    assert cot(zoo) is nan

    # 断言 cot(0) 的结果是 zoo
    assert cot(0) is zoo
    # 断言 cot(2*pi) 的结果是 zoo
    assert cot(2*pi) is zoo

    # 断言 cot(acot(x)) 等于 x
    assert cot(acot(x)) == x
    # 断言 cot(atan(x)) 等于 1 / x
    assert cot(atan(x)) == 1 / x
    # 断言 cot(asin(x)) 等于 sqrt(1 - x**2) / x
    assert cot(asin(x)) == sqrt(1 - x**2) / x
    # 断言 cot(acos(x)) 等于 x / sqrt(1 - x**2)
    assert cot(acos(x)) == x / sqrt(1 - x**2)
    # 断言 cot(acsc(x)) 等于 sqrt(1 - 1 / x**2) * x
    assert cot(acsc(x)) == sqrt(1 - 1 / x**2) * x
    # 断言 cot(asec(x)) 等于 1 / (sqrt(1 - 1 / x**2) * x)
    assert cot(asec(x)) == 1 / (sqrt(1 - 1 / x**2) * x)
    # 断言 cot(atan2(y, x)) 等于 x/y
    assert cot(atan2(y, x)) == x/y

    # 断言 cot(pi*I) 的结果是 -coth(pi)*I
    assert cot(pi*I) == -coth(pi)*I
    # 断言 cot(-pi*I) 的结果是 coth(pi)*I
    assert cot(-pi*I) == coth(pi)*I
    # 断言 cot(-2*I) 的结果是 coth(2)*I
    assert cot(-2*I) == coth(2)*I

    # 断言 cot(pi), cot(2*pi), cot(3*pi) 的结果相等
    assert cot(pi) == cot(2*pi) == cot(3*pi)
    # 断言 cot(-pi), cot(-2*pi), cot(-3*pi) 的结果相等
    assert cot(-pi) == cot(-2*pi) == cot(-3*pi)

    # 断言 cot(pi/2) 的结果是 0
    assert cot(pi/2) == 0
    # 断言 cot(-pi/2) 的结果是 0
    assert cot(-pi/2) == 0
    # 断言 cot(pi*5/2) 的结果是 0
    assert cot(pi*Rational(5, 2)) == 0
    # 断言 cot(pi*7/2) 的结果是 0
    assert cot(pi*Rational(7, 2)) == 0

    # 断言 cot(pi/3) 的结果是 1/sqrt(3)
    assert cot(pi/3) == 1/sqrt(3)
    # 断言 cot(pi*-2/3) 的结果是 1/sqrt(3)
    assert cot(pi*Rational(-2, 3)) == 1/sqrt(3)

    # 断言 cot(pi/4) 的结果是 S.One
    assert cot(pi/4) is S.One
    # 断言 cot(-pi/4) 的结果是 S.NegativeOne
    assert cot(-pi/4) is S.NegativeOne
    # 断言 cot(pi*17/4) 的结果是 S.One
    assert cot(pi*Rational(17, 4)) is S.One
    # 断言 cot(pi*-3/4) 的结果是 S.One
    assert cot(pi*Rational(-3, 4)) is S.One

    # 断言 cot(pi/6) 的结果是 sqrt(3)
    assert cot(pi/6) == sqrt(3)
    # 断言 cot(-pi/6) 的结果是 -sqrt(3)
    assert cot(-pi/6) == -sqrt(3)
    # 断言 cot(pi*7/6) 的结果是 sqrt(3)
    assert cot(pi*Rational(7, 6)) == sqrt(3)
    # 断言 cot(pi*-5/6) 的结果是 sqrt(3)
    assert cot(pi*Rational(-5, 6)) == sqrt(3)

    # 断言 cot(pi/8) 的结果是 1 + sqrt(2)
    assert cot(pi/8) == 1 + sqrt(2)
    # 断言 cot(pi*3/8) 的结果是 -1 + sqrt(2)
    assert cot(pi*Rational(3, 8)) == -1 + sqrt(2)
    # 断言 cot(pi*5/8) 的结果是 1 - sqrt(2)
    assert cot(pi*Rational(5, 8)) == 1 - sqrt(2)
    # 断言 cot(pi*7/8) 的结果是 -1 - sqrt(2)
    assert cot(pi*Rational(7, 8)) == -1 - sqrt(2)

    # 断言 cot(pi/12) 的结果是 sqrt(3) + 2
    assert cot(pi/12) == sqrt(3) + 2
    # 断言 cot(pi*5/12) 的结果是 -sqrt(3) + 2
    assert cot(pi*Rational(5, 12)) == -sqrt(3) + 2
    # 断言 cot(pi*7/12) 的结果是 sqrt(3) - 2
    assert cot(pi*Rational(7, 12)) == sqrt(3) - 2
    # 断言 cot(pi*11/12) 的结果是 -sqrt(3) - 2
    assert cot(pi*Rational(11, 12)) == -sqrt(3) - 2

    # 断言 cot(pi/24).radsimp() 的结果是 sqrt(2) + sqrt(3) + 2 + sqrt(6)
    assert cot(pi/24).radsimp() == sqrt(2) + sqrt(3) + 2 + sqrt(6)
    # 断言 cot(pi*5/24).radsimp() 的结果是 -sqrt(2) - sqrt(3) + 2 + sqrt(6)
    assert cot(pi*Rational(5, 24)).radsimp() == -sqrt(2) - sqrt(3) + 2 + sqrt(6)
    # 断言 cot(pi*7/24).radsimp() 的结果是 -sqrt(2) + sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(7, 24)).radsimp() == -sqrt(2) + sqrt(3) - 2 + sqrt(6)
    # 断言 cot(pi*11/24).radsimp() 的结果是 sqrt(2) - sqrt(3) - 2 + sqrt(6)
    assert cot(pi*Rational(11, 24)).radsimp() == sqrt(2) - sqrt(3) - 2 + sqrt(6)
    # 断言 cot(pi*13/24).radsimp() 的结果是 -sqrt(2) + sqrt(3) + 2 - sqrt(6)
    assert cot(pi*Rational(13, 24)).radsimp() == -sqrt
    # 检查余切函数在有理数倍数 π*Rational(-11/7) 和 π*Rational(3/7) 上的值是否相等
    assert cot(pi*Rational(-11, 7)) == cot(pi*Rational(3, 7))
    
    # 检查余切函数在有理数倍数 π*Rational(39/34) 和 π*Rational(5/34) 上的值是否相等
    assert cot(pi*Rational(39, 34)) == cot(pi*Rational(5, 34))
    
    # 检查余切函数在有理数倍数 π*Rational(-41/34) 和 π*Rational(7/34) 上的值是否满足负关系
    assert cot(pi*Rational(-41, 34)) == -cot(pi*Rational(7, 34))
    
    # 检查普通符号 x 和任意符号 r 的余切函数是否为无限大
    assert cot(x).is_finite is None
    assert cot(r).is_finite is None
    
    # 创建一个虚数符号 i，并检查其余切函数是否为有限的
    i = Symbol('i', imaginary=True)
    assert cot(i).is_finite is True
    
    # 检查符号 x 替换为 3*pi 后的余切函数是否为无穷大
    assert cot(x).subs(x, 3*pi) is zoo
    
    # 针对 issue #21177 (https://github.com/sympy/sympy/issues/21177) 进行验证
    # 创建函数 f，并检查其在 x + 4 处的余切函数除以 (3*x) 的主导项是否等于 1/(3*pi*x**2)
    f = cot(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == 1/(3*pi*x**2)
# 定义一个测试函数，用于测试正切、余切、正弦和余弦函数的数值评估
def test_tan_cot_sin_cos_evalf():
    # 断言判断 tan(pi*8/15)*cos(pi*8/15)/sin(pi*8/15) 是否接近于 1
    assert abs((tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15)) - 1).evalf()) < 1e-14
    # 断言判断 cot(pi*4/15)*sin(pi*4/15)/cos(pi*4/15) 是否接近于 1
    assert abs((cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15)) - 1).evalf()) < 1e-14

# 标记为预期失败的测试函数
@XFAIL
def test_tan_cot_sin_cos_ratsimp():
    # 断言判断 tan(pi*8/15)*cos(pi*8/15)/sin(pi*8/15) 是否能简化为 1
    assert 1 == (tan(pi*Rational(8, 15))*cos(pi*Rational(8, 15))/sin(pi*Rational(8, 15))).ratsimp()
    # 断言判断 cot(pi*4/15)*sin(pi*4/15)/cos(pi*4/15) 是否能简化为 1
    assert 1 == (cot(pi*Rational(4, 15))*sin(pi*Rational(4, 15))/cos(pi*Rational(4, 15))).ratsimp()

# 测试余切函数的级数展开
def test_cot_series():
    # 断言判断 cot(x).series(x, 0, 9) 的级数展开结果是否正确
    assert cot(x).series(x, 0, 9) == 1/x - x/3 - x**3/45 - 2*x**5/945 - x**7/4725 + O(x**9)
    # issue 6210
    # 断言判断 cot(x**4 + x**5).series(x, 0, 1) 的级数展开结果是否正确
    assert cot(x**4 + x**5).series(x, 0, 1) == x**(-4) - 1/x**3 + x**(-2) - 1/x + 1 + O(x)
    # 断言判断 cot(pi*(1-x)).series(x, 0, 3) 的级数展开结果是否正确
    assert cot(pi*(1-x)).series(x, 0, 3) == -1/(pi*x) + pi*x/3 + O(x**3)
    # 断言判断 cot(x).taylor_term(0, x) 的泰勒展开项是否为 1/x
    assert cot(x).taylor_term(0, x) == 1/x
    # 断言判断 cot(x).taylor_term(2, x) 的泰勒展开项是否为 S.Zero
    assert cot(x).taylor_term(2, x) is S.Zero
    # 断言判断 cot(x).taylor_term(3, x) 的泰勒展开项是否为 -x**3/45
    assert cot(x).taylor_term(3, x) == -x**3/45

# 测试余切函数的重写表达式
def test_cot_rewrite():
    # 定义负指数和正指数
    neg_exp, pos_exp = exp(-x*I), exp(x*I)
    # 断言判断 cot(x).rewrite(exp) 的重写结果是否正确
    assert cot(x).rewrite(exp) == I*(pos_exp + neg_exp)/(pos_exp - neg_exp)
    # 断言判断 cot(x).rewrite(sin) 的重写结果是否正确
    assert cot(x).rewrite(sin) == sin(2*x)/(2*(sin(x)**2))
    # 断言判断 cot(x).rewrite(cos) 的重写结果是否正确
    assert cot(x).rewrite(cos) == cos(x)/cos(x - pi/2, evaluate=False)
    # 断言判断 cot(x).rewrite(tan) 的重写结果是否正确
    assert cot(x).rewrite(tan) == 1/tan(x)
    
    # 定义一个检查函数，用于检查余切函数在不同函数下的重写表达式
    def check(func):
        z = cot(func(x)).rewrite(exp) - cot(x).rewrite(exp).subs(x, func(x))
        assert z.rewrite(exp).expand() == 0
    # 分别检查 sinh、cosh、tanh、coth、sin、cos、tan 函数下的重写表达式
    check(sinh)
    check(cosh)
    check(tanh)
    check(coth)
    check(sin)
    check(cos)
    check(tan)
    
    # 断言判断 cot(log(x)).rewrite(Pow) 的重写结果是否正确
    assert cot(log(x)).rewrite(Pow) == -I*(x**-I + x**I)/(x**-I - x**I)
    # 断言判断 cot(x).rewrite(sec) 的重写结果是否正确
    assert cot(x).rewrite(sec) == sec(x - pi / 2, evaluate=False) / sec(x)
    # 断言判断 cot(x).rewrite(csc) 的重写结果是否正确
    assert cot(x).rewrite(csc) == csc(x) / csc(- x + pi / 2, evaluate=False)
    # 断言判断 cot(sin(x)).rewrite(Pow) 的重写结果是否正确
    assert cot(sin(x)).rewrite(Pow) == cot(sin(x))
    # 断言判断 cot(pi*2/5).rewrite(sqrt) 的重写结果是否正确
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == (Rational(-1, 4) + sqrt(5)/4)/\
                                                        sqrt(sqrt(5)/8 + Rational(5, 8))
    # 断言判断 cot(x).rewrite(besselj) 的重写结果是否正确
    assert cot(x).rewrite(besselj) == besselj(-S.Half, x)/besselj(S.Half, x)
    # 断言判断 cot(x).rewrite(besselj).subs(x, 0) 的重写结果是否正确
    assert cot(x).rewrite(besselj).subs(x, 0) == cot(0)

# 标记为慢速测试的测试函数
@slow
def test_cot_rewrite_slow():
    # 断言判断 cot(pi*4/34).rewrite(pow).ratsimp() 的重写结果是否正确
    assert cot(pi*Rational(4, 34)).rewrite(pow).ratsimp() == \
        (cos(pi*Rational(4, 34))/sin(pi*Rational(4, 34))).rewrite(pow).ratsimp()
    # 断言判断 cot(pi*4/17).rewrite(pow) 的重写结果是否正确
    assert cot(pi*Rational(4, 17)).rewrite(pow) == \
        (cos(pi*Rational(4, 17))/sin(pi*Rational(4, 17))).rewrite(pow)
    # 断言判断 cot(pi/19).rewrite(pow) 的重写结果是否正确
    assert cot(pi/19).rewrite(pow) == cot(pi/19)
    # 断言判断 cot(pi/19).rewrite(sqrt) 的重写结果是否正确
    assert cot(pi/19).rewrite(sqrt) == cot(pi/19)
    # 断言判断 cot(pi*2/5).rewrite(sqrt) 的重写结果是否正确
    assert cot(pi*Rational(2, 5), evaluate=False).rewrite(sqrt) == \
        (Rational(-1, 4) + sqrt(5)/4) / sqrt(sqrt(5)/8 + Rational(5, 8))

# 测试余切函数的替换
def test_cot_subs():
    # 断言判断 cot(x).subs(cot(x), y) 的替换结果是否为 y
    assert cot(x).subs(cot(x), y) == y
    # 断言判断 cot(x).subs(x, y) 的替换结果是否为 cot(y)
    assert cot(x).subs(x, y) == cot(y)
    # 断言判断 cot(x).subs(x, 0) 的替换结果是否为 zoo (无穷大)
    assert cot(x).subs(x, 0) is zoo
    # 断言判断 cot(x).subs(x, pi) 的替换结果是否为 zoo (无穷大)
    assert cot(x).subs(x, S.Pi) is zoo

# 测试余切函数的展开
def test_cot_expansion():
    # 断言判断 cot(x + y).expand(trig=True).to
    # 断言：验证 cot(x - y) 的展开结果是否等于给定表达式
    assert cot(x - y).expand(trig=True).together() == (
        cot(x)*cot(-y) - 1)/(cot(x) + cot(-y))
    
    # 断言：验证 cot(x + y + z) 的展开结果是否等于给定表达式
    assert cot(x + y + z).expand(trig=True).together() == (
        (cot(x)*cot(y)*cot(z) - cot(x) - cot(y) - cot(z))/
        (-1 + cot(x)*cot(y) + cot(x)*cot(z) + cot(y)*cot(z)))
    
    # 断言：验证 cot(3*x) 的展开结果是否等于给定表达式
    assert cot(3*x).expand(trig=True).together() == (
        (cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1))
    
    # 断言：验证 cot(2*x) 的展开结果是否等于给定表达式
    assert cot(2*x).expand(trig=True) == cot(x)/2 - 1/(2*cot(x))
    
    # 断言：验证 cot(3*x) 的展开结果是否等于给定表达式
    assert cot(3*x).expand(trig=True).together() == (
        cot(x)**2 - 3)*cot(x)/(3*cot(x)**2 - 1)
    
    # 断言：验证 cot(4*x - pi/4) 的展开结果是否等于给定表达式
    assert cot(4*x - pi/4).expand(trig=True).cancel() == (
        -tan(x)**4 + 4*tan(x)**3 + 6*tan(x)**2 - 4*tan(x) - 1
        )/(tan(x)**4 + 4*tan(x)**3 - 6*tan(x)**2 - 4*tan(x) + 1)
    
    # 调用函数 _test_extrig 进行额外的三角函数测试
    _test_extrig(cot, 2, (-1 + cot(1)**2)/(2*cot(1)))
    _test_extrig(cot, 3, (-3*cot(1) + cot(1)**3)/(-1 + 3*cot(1)**2))
# 定义一个名为 test_cot_AccumBounds 的测试函数，用于测试 cot 函数在 AccumBounds 对象上的行为
def test_cot_AccumBounds():
    # 断言 cot 函数对于区间 [-∞, ∞] 返回 [-∞, ∞]
    assert cot(AccumBounds(-oo, oo)) == AccumBounds(-oo, oo)
    # 断言 cot 函数对于区间 [-π/3, π/3] 返回 [-∞, ∞]
    assert cot(AccumBounds(-S.Pi/3, S.Pi/3)) == AccumBounds(-oo, oo)
    # 断言 cot 函数对于区间 [π/6, π/3] 返回 [cot(π/3), cot(π/6)]
    assert cot(AccumBounds(S.Pi/6, S.Pi/3)) == AccumBounds(cot(S.Pi/3), cot(S.Pi/6))


# 定义一个名为 test_cot_fdiff 的测试函数，用于测试 cot 函数的导数计算
def test_cot_fdiff():
    # 断言 cot(x) 的一阶导数为 -cot(x)^2 - 1
    assert cot(x).fdiff() == -cot(x)**2 - 1
    # 断言尝试计算 cot(x) 的二阶导数会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: cot(x).fdiff(2))


# 定义一个名为 test_sinc 的测试函数，用于测试 sinc 函数的各种性质
def test_sinc():
    # 断言 sinc(x) 返回的对象类型是 sinc
    assert isinstance(sinc(x), sinc)

    # 定义一个零值符号 s，并断言 sinc(s) 返回 S.One
    s = Symbol('s', zero=True)
    assert sinc(s) is S.One
    # 断言 sinc(∞) 返回 S.Zero
    assert sinc(S.Infinity) is S.Zero
    # 断言 sinc(-∞) 返回 S.Zero
    assert sinc(S.NegativeInfinity) is S.Zero
    # 断言 sinc(NaN) 返回 S.NaN
    assert sinc(S.NaN) is S.NaN
    # 断言 sinc(ComplexInfinity) 返回 S.NaN
    assert sinc(S.ComplexInfinity) is S.NaN

    # 定义一个非零整数符号 n，并断言 sinc(n*pi) 返回 S.Zero
    n = Symbol('n', integer=True, nonzero=True)
    assert sinc(n*pi) is S.Zero
    # 断言 sinc(-n*pi) 返回 S.Zero
    assert sinc(-n*pi) is S.Zero
    # 断言 sinc(pi/2) 返回 2/pi
    assert sinc(pi/2) == 2 / pi
    # 断言 sinc(-pi/2) 返回 2/pi
    assert sinc(-pi/2) == 2 / pi
    # 断言 sinc(5*pi/2) 返回 2 / (5*pi)
    assert sinc(pi*Rational(5, 2)) == 2 / (5*pi)
    # 断言 sinc(7*pi/2) 返回 -2 / (7*pi)
    assert sinc(pi*Rational(7, 2)) == -2 / (7*pi)

    # 断言 sinc(-x) 等于 sinc(x)
    assert sinc(-x) == sinc(x)

    # 断言 sinc(x) 对 x 的一阶导数为 cos(x)/x - sin(x)/x^2
    assert sinc(x).diff(x) == cos(x)/x - sin(x)/x**2
    # 断言 sinc(x) 对 x 的一阶导数可以用 (sin(x)/x).diff(x) 表示
    assert sinc(x).diff(x) == (sin(x)/x).diff(x)
    # 断言 sinc(x) 对 x 的二阶导数为 (-sin(x) - 2*cos(x)/x + 2*sin(x)/x**2)/x
    assert sinc(x).diff(x, x) == (-sin(x) - 2*cos(x)/x + 2*sin(x)/x**2)/x
    # 断言 sinc(x) 对 x 的二阶导数可以用 (sin(x)/x).diff(x, x) 表示
    assert sinc(x).diff(x, x) == (sin(x)/x).diff(x, x)
    # 断言 sinc(x) 在 x 趋近于 0 时的极限为 0
    assert limit(sinc(x).diff(x), x, 0) == 0
    # 断言 sinc(x) 在 x 趋近于 0 时的二阶导数极限为 -1/3
    assert limit(sinc(x).diff(x, x), x, 0) == -S(1)/3

    # 针对已知的 issue 进行验证
    #
    # assert sinc(x).diff(x) == Piecewise(((x*cos(x) - sin(x)) / x**2, Ne(x, 0)), (0, True))
    #
    # assert sinc(x).diff(x).equals(sinc(x).rewrite(sin).diff(x))
    #
    # assert sinc(x).diff(x).subs(x, 0) is S.Zero

    # 断言 sinc(x) 的泰勒级数展开为 1 - x^2/6 + x^4/120 + O(x^6)
    assert sinc(x).series() == 1 - x**2/6 + x**4/120 + O(x**6)

    # 断言 sinc(x) 通过 jn 函数重写为 jn(0, x)
    assert sinc(x).rewrite(jn) == jn(0, x)
    # 断言 sinc(x) 通过 sin 函数重写为 Piecewise((sin(x)/x, Ne(x, 0)), (1, True))
    assert sinc(x).rewrite(sin) == Piecewise((sin(x)/x, Ne(x, 0)), (1, True))
    # 断言 sinc(pi, evaluate=False) 返回 is_zero 为 True
    assert sinc(pi, evaluate=False).is_zero is True
    # 断言 sinc(0, evaluate=False) 返回 is_zero 为 False
    assert sinc(0, evaluate=False).is_zero is False
    # 断言 sinc(n*pi, evaluate=False) 返回 is_zero 为 True
    assert sinc(n*pi, evaluate=False).is_zero is True
    # 断言 sinc(x) 的 is_zero 返回 None
    assert sinc(x).is_zero is None
    # 定义一个实数非零符号 xr，并断言 sinc(x) 的 is_real 返回 None
    xr = Symbol('xr', real=True, nonzero=True)
    assert sinc(x).is_real is None
    # 断言 sinc(xr) 的 is_real 返回 True
    assert sinc(xr).is_real is True
    # 断言 sinc(I*xr) 的 is_real 返回 True
    assert sinc(I*xr).is_real is True
    # 断言 sinc(I*100) 的 is_real 返回 True
    assert sinc(I*100).is_real is True
    # 断言 sinc(x) 的 is_finite 返回 None
    assert sinc(x).is_finite is None
    # 断言 sinc(xr) 的 is_finite 返回 True
    assert sinc(xr).is_finite is True
    # 断言：根据给定的表达式，验证反正弦函数的计算结果
    assert asin((sqrt(3) - 1)/sqrt(2**3)) == pi/12
    assert asin(-(sqrt(3) - 1)/sqrt(2**3)) == -pi/12

    # 检查对于精确值的往返测试：
    for d in [5, 6, 8, 10, 12]:
        for n in range(-(d//2), d//2 + 1):
            # 如果 n 和 d 互质，则进行下列断言
            if gcd(n, d) == 1:
                # 断言：对于正弦值，验证反正弦函数的计算结果
                assert asin(sin(n*pi/d)) == n*pi/d

    # 断言：验证反正弦函数对自变量 x 的导数
    assert asin(x).diff(x) == 1/sqrt(1 - x**2)

    # 断言：验证反正弦函数在特定条件下的实部性质
    assert asin(0.2, evaluate=False).is_real is True
    assert asin(-2).is_real is False
    assert asin(r).is_real is None

    # 断言：验证反正弦函数在复数域内的计算结果
    assert asin(-2*I) == -I*asinh(2)

    # 断言：验证反正弦函数在有理数域内的正性质
    assert asin(Rational(1, 7), evaluate=False).is_positive is True
    assert asin(Rational(-1, 7), evaluate=False).is_positive is False
    assert asin(p).is_positive is None

    # 断言：验证反正弦函数的反函数性质
    assert asin(sin(Rational(7, 2))) == Rational(-7, 2) + pi
    assert asin(sin(Rational(-7, 4))) == Rational(7, 4) - pi

    # 断言：验证反正弦函数不受余弦函数影响
    assert unchanged(asin, cos(x))
# 定义测试函数 test_asin_series，用于测试 asin(x) 的级数展开的准确性
def test_asin_series():
    # 断言 asin(x).series(x, 0, 9) 的结果与预期相等
    assert asin(x).series(x, 0, 9) == \
        x + x**3/6 + 3*x**5/40 + 5*x**7/112 + O(x**9)
    # 计算 asin(x).taylor_term(5, x) 并将结果赋给 t5
    t5 = asin(x).taylor_term(5, x)
    # 断言 t5 的值与预期的 3*x**5/40 相等
    assert t5 == 3*x**5/40
    # 断言 asin(x).taylor_term(7, x, t5, 0) 的结果与预期的 5*x**7/112 相等
    assert asin(x).taylor_term(7, x, t5, 0) == 5*x**7/112


# 定义测试函数 test_asin_leading_term，测试 asin(x) 的主导项计算
def test_asin_leading_term():
    # 断言 asin(x).as_leading_term(x) 的结果与预期的 x 相等
    assert asin(x).as_leading_term(x) == x
    # 测试分支点的情况
    assert asin(x + 1).as_leading_term(x) == pi/2
    assert asin(x - 1).as_leading_term(x) == -pi/2
    assert asin(1/x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert asin(1/x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    # 测试位于分支切线上的情况
    assert asin(I*x + 2).as_leading_term(x, cdir=1) == pi - asin(2)
    assert asin(-I*x + 2).as_leading_term(x, cdir=1) == asin(2)
    assert asin(I*x - 2).as_leading_term(x, cdir=1) == -asin(2)
    assert asin(-I*x - 2).as_leading_term(x, cdir=1) == -pi + asin(2)
    # 测试 im(ndir) == 0 的情况
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -pi/2 + I*log(2 - sqrt(3))
    assert asin(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(2 - sqrt(3))


# 定义测试函数 test_asin_rewrite，测试 asin(x) 的重写方法
def test_asin_rewrite():
    assert asin(x).rewrite(log) == -I*log(I*x + sqrt(1 - x**2))
    assert asin(x).rewrite(atan) == 2*atan(x/(1 + sqrt(1 - x**2)))
    assert asin(x).rewrite(acos) == S.Pi/2 - acos(x)
    assert asin(x).rewrite(acot) == 2*acot((sqrt(-x**2 + 1) + 1)/x)
    assert asin(x).rewrite(asec) == -asec(1/x) + pi/2
    assert asin(x).rewrite(acsc) == acsc(1/x)


# 定义测试函数 test_asin_fdiff，测试 asin(x) 的一阶导数
def test_asin_fdiff():
    # 断言 asin(x).fdiff() 的结果与预期的 1/sqrt(1 - x**2) 相等
    assert asin(x).fdiff() == 1/sqrt(1 - x**2)
    # 使用 lambda 函数测试 asin(x).fdiff(2) 是否会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: asin(x).fdiff(2))


# 定义测试函数 test_acos，测试 acos(x) 的各种情况
def test_acos():
    assert acos(nan) is nan  # 断言 acos(nan) 的结果为 nan
    assert acos(zoo) is zoo  # 断言 acos(zoo) 的结果为 zoo

    # 断言 acos.nargs 的结果为 FiniteSet(1)
    assert acos.nargs == FiniteSet(1)
    assert acos(oo) == I*oo  # 断言 acos(oo) 的结果为 I*oo
    assert acos(-oo) == -I*oo  # 断言 acos(-oo) 的结果为 -I*oo

    # 注意：acos(-x) = pi - acos(x)
    assert acos(0) == pi/2
    assert acos(S.Half) == pi/3
    assert acos(Rational(-1, 2)) == pi*Rational(2, 3)
    assert acos(1) == 0
    assert acos(-1) == pi
    assert acos(sqrt(2)/2) == pi/4
    assert acos(-sqrt(2)/2) == pi*Rational(3, 4)

    # 检查精确值的来回转换：
    for d in [5, 6, 8, 10, 12]:
        for num in range(d):
            if gcd(num, d) == 1:
                assert acos(cos(num*pi/d)) == num*pi/d

    assert acos(2*I) == pi/2 - asin(2*I)

    # 断言 acos(x).diff(x) 的结果与预期的 -1/sqrt(1 - x**2) 相等
    assert acos(x).diff(x) == -1/sqrt(1 - x**2)

    assert acos(0.2).is_real is True
    assert acos(-2).is_real is False
    assert acos(r).is_real is None

    assert acos(Rational(1, 7), evaluate=False).is_positive is True
    assert acos(Rational(-1, 7), evaluate=False).is_positive is True
    assert acos(Rational(3, 2), evaluate=False).is_positive is False
    assert acos(p).is_positive is None

    assert acos(2 + p).conjugate() != acos(10 + p)
    assert acos(-3 + n).conjugate() != acos(-3 + n)
    assert acos(Rational(1, 3)).conjugate() == acos(Rational(1, 3))
    assert acos(Rational(-1, 3)).conjugate() == acos(Rational(-1, 3))
    # 断言：对复数进行反余弦运算后取共轭，应该等于对其共轭后再进行反余弦运算。
    assert acos(p + n*I).conjugate() == acos(p - n*I)
    
    # 断言：对复数 z 进行反余弦运算后取共轭，不应该等于对 z 的共轭进行反余弦运算。
    assert acos(z).conjugate() != acos(conjugate(z))
# 定义测试函数，测试 acos 函数的 leading term 方法
def test_acos_leading_term():
    # 断言 acos(x) 的 leading term 是 pi/2
    assert acos(x).as_leading_term(x) == pi/2

    # 测试分支点
    assert acos(x + 1).as_leading_term(x) == sqrt(2)*sqrt(-x)
    assert acos(x - 1).as_leading_term(x) == pi
    assert acos(1/x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert acos(1/x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)

    # 测试落在分支切线上的点
    assert acos(I*x + 2).as_leading_term(x, cdir=1) == -acos(2)
    assert acos(-I*x + 2).as_leading_term(x, cdir=1) == acos(2)
    assert acos(I*x - 2).as_leading_term(x, cdir=1) == acos(-2)
    assert acos(-I*x - 2).as_leading_term(x, cdir=1) == 2*pi - acos(-2)

    # 测试 im(ndir) == 0 的点
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == pi + I*log(sqrt(3) + 2)
    assert acos(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == pi + I*log(sqrt(3) + 2)


# 定义测试函数，测试 acos 函数的 series 方法
def test_acos_series():
    # 测试 x=0 处到 8 阶的级数展开
    assert acos(x).series(x, 0, 8) == \
        pi/2 - x - x**3/6 - 3*x**5/40 - 5*x**7/112 + O(x**8)
    # 使用 asin 函数的级数展开验证 acos 函数的级数展开
    assert acos(x).series(x, 0, 8) == pi/2 - asin(x).series(x, 0, 8)
    
    # 测试 taylor_term 方法
    t5 = acos(x).taylor_term(5, x)
    assert t5 == -3*x**5/40
    assert acos(x).taylor_term(7, x, t5, 0) == -5*x**7/112
    assert acos(x).taylor_term(0, x) == pi/2
    assert acos(x).taylor_term(2, x) is S.Zero


# 定义测试函数，测试 acos 函数的 rewrite 方法
def test_acos_rewrite():
    assert acos(x).rewrite(log) == pi/2 + I*log(I*x + sqrt(1 - x**2))
    assert acos(x).rewrite(atan) == pi*(-x*sqrt(x**(-2)) + 1)/2 + atan(sqrt(1 - x**2)/x)
    assert acos(0).rewrite(atan) == S.Pi/2
    assert acos(0.5).rewrite(atan) == acos(0.5).rewrite(log)
    assert acos(x).rewrite(asin) == S.Pi/2 - asin(x)
    assert acos(x).rewrite(acot) == -2*acot((sqrt(-x**2 + 1) + 1)/x) + pi/2
    assert acos(x).rewrite(asec) == asec(1/x)
    assert acos(x).rewrite(acsc) == -acsc(1/x) + pi/2


# 定义测试函数，测试 acos 函数的 fdiff 方法
def test_acos_fdiff():
    assert acos(x).fdiff() == -1/sqrt(1 - x**2)
    raises(ArgumentIndexError, lambda: acos(x).fdiff(2))


# 定义测试函数，测试 atan 函数的各种情况
def test_atan():
    assert atan(nan) is nan

    assert atan.nargs == FiniteSet(1)
    assert atan(oo) == pi/2
    assert atan(-oo) == -pi/2
    assert atan(zoo) == AccumBounds(-pi/2, pi/2)

    assert atan(0) == 0
    assert atan(1) == pi/4
    assert atan(sqrt(3)) == pi/3
    assert atan(-(1 + sqrt(2))) == pi*Rational(-3, 8)
    assert atan(sqrt(5 - 2 * sqrt(5))) == pi/5
    assert atan(-sqrt(1 - 2 * sqrt(5)/ 5)) == -pi/10
    assert atan(sqrt(1 + 2 * sqrt(5) / 5)) == pi*Rational(3, 10)
    assert atan(-2 + sqrt(3)) == -pi/12
    assert atan(2 + sqrt(3)) == pi*Rational(5, 12)
    assert atan(-2 - sqrt(3)) == pi*Rational(-5, 12)

    # 检查精确值的往返：
    for d in [5, 6, 8, 10, 12]:
        for num in range(-(d//2), d//2 + 1):
            if gcd(num, d) == 1:
                assert atan(tan(num*pi/d)) == num*pi/d

    assert atan(oo) == pi/2
    assert atan(x).diff(x) == 1/(1 + x**2)

    assert atan(r).is_real is True

    assert atan(-2*I) == -I*atanh(2)
    assert unchanged(atan, cot(x))
    # 断言，验证 arctan(cot(1/4)) 是否等于 -1/4 + pi/2
    assert atan(cot(Rational(1, 4))) == Rational(-1, 4) + pi/2
    
    # 断言，验证 arccot(1/4) 是否为非有理数
    assert acot(Rational(1, 4)).is_rational is False
    
    # 对于变量集合 (x, p, n, np, nn, nz, ep, en, enp, enn, enz) 中的每一个变量 s 进行断言验证
    for s in (x, p, n, np, nn, nz, ep, en, enp, enn, enz):
        # 如果 s 是实数或者扩展实数但不是空值
        if s.is_real or s.is_extended_real is None:
            # 断言：s 是否非零与 atan(s) 是否非零相同
            assert s.is_nonzero is atan(s).is_nonzero
            # 断言：s 是否为正与 atan(s) 是否为正相同
            assert s.is_positive is atan(s).is_positive
            # 断言：s 是否为负与 atan(s) 是否为负相同
            assert s.is_negative is atan(s).is_negative
            # 断言：s 是否为非正与 atan(s) 是否为非正相同
            assert s.is_nonpositive is atan(s).is_nonpositive
            # 断言：s 是否为非负与 atan(s) 是否为非负相同
            assert s.is_nonnegative is atan(s).is_nonnegative
        else:
            # 断言：s 是否扩展非零与 atan(s) 是否非零相同
            assert s.is_extended_nonzero is atan(s).is_nonzero
            # 断言：s 是否扩展正与 atan(s) 是否为正相同
            assert s.is_extended_positive is atan(s).is_positive
            # 断言：s 是否扩展负与 atan(s) 是否为负相同
            assert s.is_extended_negative is atan(s).is_negative
            # 断言：s 是否扩展非正与 atan(s) 是否为非正相同
            assert s.is_extended_nonpositive is atan(s).is_nonpositive
            # 断言：s 是否扩展非负与 atan(s) 是否为非负相同
            assert s.is_extended_nonnegative is atan(s).is_nonnegative
    
        # 断言：s 是否扩展非零与 atan(s) 是否扩展非零相同
        assert s.is_extended_nonzero is atan(s).is_extended_nonzero
        # 断言：s 是否扩展正与 atan(s) 是否扩展正相同
        assert s.is_extended_positive is atan(s).is_extended_positive
        # 断言：s 是否扩展负与 atan(s) 是否扩展负相同
        assert s.is_extended_negative is atan(s).is_extended_negative
        # 断言：s 是否扩展非正与 atan(s) 是否扩展非正相同
        assert s.is_extended_nonpositive is atan(s).is_extended_nonpositive
        # 断言：s 是否扩展非负与 atan(s) 是否扩展非负相同
        assert s.is_extended_nonnegative is atan(s).is_extended_nonnegative
# 定义测试函数 test_atan_rewrite，用于测试反正切函数的重写方法
def test_atan_rewrite():
    # 断言：使用对数重写，验证 atan(x) 的结果
    assert atan(x).rewrite(log) == I*(log(1 - I*x)-log(1 + I*x))/2
    # 断言：使用反正弦重写，验证 atan(x) 的结果
    assert atan(x).rewrite(asin) == (-asin(1/sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x
    # 断言：使用反余弦重写，验证 atan(x) 的结果
    assert atan(x).rewrite(acos) == sqrt(x**2)*acos(1/sqrt(x**2 + 1))/x
    # 断言：使用反余切重写，验证 atan(x) 的结果
    assert atan(x).rewrite(acot) == acot(1/x)
    # 断言：使用反正割重写，验证 atan(x) 的结果
    assert atan(x).rewrite(asec) == sqrt(x**2)*asec(sqrt(x**2 + 1))/x
    # 断言：使用反余割重写，验证 atan(x) 的结果
    assert atan(x).rewrite(acsc) == (-acsc(sqrt(x**2 + 1)) + pi/2)*sqrt(x**2)/x

    # 断言：对复数 -5*I 进行数值计算，验证使用对数重写的 atan(-5*I) 的结果
    assert atan(-5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:-5*I})
    # 断言：对复数 5*I 进行数值计算，验证使用对数重写的 atan(5*I) 的结果
    assert atan(5*I).evalf() == atan(x).rewrite(log).evalf(subs={x:5*I})


# 定义测试函数 test_atan_fdiff，用于测试反正切函数的导数
def test_atan_fdiff():
    # 断言：验证 atan(x) 的导数计算结果
    assert atan(x).fdiff() == 1/(x**2 + 1)
    # 断言：验证对不存在的导数序数进行调用时抛出异常
    raises(ArgumentIndexError, lambda: atan(x).fdiff(2))


# 定义测试函数 test_atan_leading_term，用于测试反正切函数的主导项
def test_atan_leading_term():
    # 断言：验证 atan(x) 的主导项计算结果
    assert atan(x).as_leading_term(x) == x
    # 断言：验证 atan(1/x) 的主导项计算结果（正方向）
    assert atan(1/x).as_leading_term(x, cdir=1) == pi/2
    # 断言：验证 atan(1/x) 的主导项计算结果（负方向）
    assert atan(1/x).as_leading_term(x, cdir=-1) == -pi/2
    # 断言：验证存在分支点情况下 atan(x + I) 的主导项计算结果（正方向）
    assert atan(x + I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    # 断言：验证存在分支点情况下 atan(x + I) 的主导项计算结果（负方向）
    assert atan(x + I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    # 断言：验证存在分支点情况下 atan(x - I) 的主导项计算结果（正方向）
    assert atan(x - I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    # 断言：验证存在分支点情况下 atan(x - I) 的主导项计算结果（负方向）
    assert atan(x - I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    # 断言：验证存在分支切割线上情况下 atan(x + 2*I) 的主导项计算结果（正方向）
    assert atan(x + 2*I).as_leading_term(x, cdir=1) == I*atanh(2)
    # 断言：验证存在分支切割线上情况下 atan(x + 2*I) 的主导项计算结果（负方向）
    assert atan(x + 2*I).as_leading_term(x, cdir=-1) == -pi + I*atanh(2)
    # 断言：验证存在分支切割线上情况下 atan(x - 2*I) 的主导项计算结果（正方向）
    assert atan(x - 2*I).as_leading_term(x, cdir=1) == pi - I*atanh(2)
    # 断言：验证存在分支切割线上情况下 atan(x - 2*I) 的主导项计算结果（负方向）
    assert atan(x - 2*I).as_leading_term(x, cdir=-1) == -I*atanh(2)
    # 断言：验证存在 re(ndir) == 0 情况下 atan(2*I - I*x - x**2) 的主导项计算结果（正方向）
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 + I*log(3)/2
    # 断言：验证存在 re(ndir) == 0 情况下 atan(2*I - I*x - x**2) 的主导项计算结果（负方向）
    assert atan(2*I - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(3)/2


# 定义测试函数 test_atan2，用于测试二元反正切函数 atan2 的不同情况
def test_atan2():
    # 断言：验证 atan2 的参数个数
    assert atan2.nargs == FiniteSet(2)
    # 断言：验证 atan2(0, 0) 的结果
    assert atan2(0, 0) is S.NaN
    # 断言：验证 atan2(0, 1) 的结果
    assert atan2(0, 1) == 0
    # 断言：验证 atan2(1, 1) 的结果
    assert atan2(1, 1) == pi/4
    # 断言：验证 atan2(1, 0) 的结果
    assert atan2(1, 0) == pi/2
    # 断言：验证 atan2(1, -1) 的结果
    assert atan2(1, -1) == pi*Rational(3, 4)
    # 断言：验证 atan2(0, -1) 的结果
    assert atan2(0, -1) == pi
    # 断言：验证 atan2(-1, -1) 的结果
    assert atan2(-1, -1) == pi*Rational(-3, 4)
    # 断言：验证 atan2(-1, 0) 的结果
    assert atan2(-1, 0) == -pi/2
    # 断言：验证 atan2(-1, 1) 的结果
    assert atan2(-1, 1) == -pi/4
    # 符号定义
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    # 断言：验证 atan2(r, i) 的结果
    eq = atan2(r, i)
    ans = -I*log((i + I*r)/sqrt(i**2 + r**2))
    reps = ((r, 2), (i, I))
    assert eq.subs(reps) == ans.subs(reps)
    # 断言：验证 atan2(y, x) 在 x 和 y 为负数时的结果
    x = Symbol('x', negative=True)
    y = Symbol('y', negative=True)
    assert atan2(y, x) == atan(y/x) - pi
    # 断言：验证 atan2(y, x) 在 y 为非负数时的结果
    y = Symbol('y', nonnegative=True)
    assert atan2(y, x) == atan(y/x) + pi
    # 断言：验证 atan2(y, x) 的基本结果
    y = Symbol('y')
    # 计算 atan2 函数的结果，其中 atan2 是反正切函数，能处理复数参数
    ex = atan2(y, x) - arg(x + I*y)
    
    # 断言：对特定的 x 和 y 值计算 ex 的 arg 重写结果应该为 0
    assert ex.subs({x:2, y:3}).rewrite(arg) == 0
    
    # 断言：对特定的 x 和 y*I 值计算 ex 的 arg 重写结果应该为 -pi - I*log(sqrt(5)*I/5)
    assert ex.subs({x:2, y:3*I}).rewrite(arg) == -pi - I*log(sqrt(5)*I/5)
    
    # 断言：对特定的 2*I 和 3 值计算 ex 的 arg 重写结果应该为 -pi/2 - I*log(sqrt(5)*I)
    assert ex.subs({x:2*I, y:3}).rewrite(arg) == -pi/2 - I*log(sqrt(5)*I)
    
    # 断言：对特定的 2*I 和 3*I 值计算 ex 的 arg 重写结果应该为 -pi + atan(Rational(2, 3)) + atan(Rational(3, 2))
    assert ex.subs({x:2*I, y:3*I}).rewrite(arg) == -pi + atan(Rational(2, 3)) + atan(Rational(3, 2))
    
    # 定义虚数符号 i 和实数符号 r
    i = symbols('i', imaginary=True)
    r = symbols('r', real=True)
    
    # 计算 atan2(i, r)
    e = atan2(i, r)
    
    # 将 e 的 arg 重写结果存储到 rewrite 变量中
    rewrite = e.rewrite(arg)
    
    # 定义替换字典 reps，将 i 替换为虚数单位 I，将 r 替换为 -2
    reps = {i: I, r: -2}
    
    # 断言：e - rewrite 在应用替换 reps 后应该等于 0
    assert (e - rewrite).subs(reps).equals(0)
    
    # 断言：计算 atan2(0, x) 的 atan 重写结果应为 Piecewise 类型，根据 x 的实部进行条件判断
    assert atan2(0, x).rewrite(atan) == Piecewise((pi, re(x) < 0),
                                            (0, Ne(x, 0)),
                                            (nan, True))
    
    # 断言：计算 atan2(0, r) 的 atan 重写结果应为 Piecewise 类型，根据 r 的实部进行条件判断
    assert atan2(0, r).rewrite(atan) == Piecewise((pi, r < 0), (0, Ne(r, 0)), (S.NaN, True))
    
    # 断言：计算 atan2(0, i) 的 atan 重写结果应为 0
    assert atan2(0, i).rewrite(atan) == 0
    
    # 断言：计算 atan2(0, r + i) 的 atan 重写结果应为 Piecewise 类型，根据 r 的实部进行条件判断
    assert atan2(0, r + i).rewrite(atan) == Piecewise((pi, r < 0), (0, True))
    
    # 断言：计算 atan2(y, x) 的 atan 重写结果应为 Piecewise 类型，根据 x 和 y 的值进行条件判断
    assert atan2(y, x).rewrite(atan) == Piecewise(
            (2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
            (pi, re(x) < 0),
            (0, (re(x) > 0) | Ne(im(x), 0)),
            (nan, True))
    
    # 断言：atan2(x, y) 的共轭应该等于 atan2(x 的共轭, y 的共轭)
    assert conjugate(atan2(x, y)) == atan2(conjugate(x), conjugate(y))
    
    # 断言：计算 atan2(y, x) 对 x 的偏导数应为 -y/(x**2 + y**2)
    assert diff(atan2(y, x), x) == -y/(x**2 + y**2)
    
    # 断言：计算 atan2(y, x) 对 y 的偏导数应为 x/(x**2 + y**2)
    assert diff(atan2(y, x), y) == x/(x**2 + y**2)
    
    # 断言：简化 atan2(y, x) 的 log 重写后对 x 的偏导数应为 -y/(x**2 + y**2)
    assert simplify(diff(atan2(y, x).rewrite(log), x)) == -y/(x**2 + y**2)
    
    # 断言：简化 atan2(y, x) 的 log 重写后对 y 的偏导数应为 x/(x**2 + y**2)
    assert simplify(diff(atan2(y, x).rewrite(log), y)) == x/(x**2 + y**2)
    
    # 断言：计算 atan2(1, 2) 的数值结果并保留 5 位小数应为 '0.46365'
    assert str(atan2(1, 2).evalf(5)) == '0.46365'
    
    # 断言：对于超出参数索引范围的 atan2(x, y)，应该引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: atan2(x, y).fdiff(3))
def test_issue_17461():
    # 定义符号类 A，继承自 Symbol 类
    class A(Symbol):
        # 设置 is_extended_real 属性为 True
        is_extended_real = True

        # 定义 _eval_evalf 方法，返回浮点数 5.0
        def _eval_evalf(self, prec):
            return Float(5.0)

    # 创建符号对象 x 和 y，分别用字符串 'X' 和 'Y' 初始化
    x = A('X')
    y = A('Y')
    # 断言计算 atan2(x, y) 的数值近似与 0.785398163397448 之差的绝对值小于等于 1e-10
    assert abs(atan2(x, y).evalf() - 0.785398163397448) <= 1e-10

def test_acot():
    # 断言 acot(nan) 返回 nan
    assert acot(nan) is nan

    # 断言 acot 函数的参数个数为 FiniteSet(1)
    assert acot.nargs == FiniteSet(1)
    # 断言 acot(-oo) 返回 0
    assert acot(-oo) == 0
    # 断言 acot(oo) 返回 0
    assert acot(oo) == 0
    # 断言 acot(zoo) 返回 0
    assert acot(zoo) == 0
    # 断言 acot(1) 返回 pi/4
    assert acot(1) == pi/4
    # 断言 acot(0) 返回 pi/2
    assert acot(0) == pi/2
    # 断言 acot(sqrt(3)/3) 返回 pi/3
    assert acot(sqrt(3)/3) == pi/3
    # 断言 acot(1/sqrt(3)) 返回 pi/3
    assert acot(1/sqrt(3)) == pi/3
    # 断言 acot(-1/sqrt(3)) 返回 -pi/3
    assert acot(-1/sqrt(3)) == -pi/3
    # 断言 acot(x).diff(x) 返回 -1/(1 + x**2)
    assert acot(x).diff(x) == -1/(1 + x**2)

    # 断言 acot(r).is_extended_real 属性为 True
    assert acot(r).is_extended_real is True

    # 断言 acot(I*pi) 返回 -I*acoth(pi)
    assert acot(I*pi) == -I*acoth(pi)
    # 断言 acot(-2*I) 返回 I*acoth(2)
    assert acot(-2*I) == I*acoth(2)
    # 断言 acot(x).is_positive 返回 None
    assert acot(x).is_positive is None
    # 断言 acot(n).is_positive 返回 False
    assert acot(n).is_positive is False
    # 断言 acot(p).is_positive 返回 True
    assert acot(p).is_positive is True
    # 断言 acot(I).is_positive 返回 False
    assert acot(I).is_positive is False
    # 断言 acot(Rational(1, 4)).is_rational 返回 False
    assert acot(Rational(1, 4)).is_rational is False
    # 断言 unchanged(acot, cot(x))
    assert unchanged(acot, cot(x))
    # 断言 unchanged(acot, tan(x))
    assert unchanged(acot, tan(x))
    # 断言 acot(cot(Rational(1, 4))) 返回 Rational(1, 4)
    assert acot(cot(Rational(1, 4))) == Rational(1, 4)
    # 断言 acot(tan(Rational(-1, 4))) 返回 Rational(1, 4) - pi/2
    assert acot(tan(Rational(-1, 4))) == Rational(1, 4) - pi/2

def test_acot_rewrite():
    # 断言 acot(x).rewrite(log) 返回复数形式表达式
    assert acot(x).rewrite(log) == I*(log(1 - I/x)-log(1 + I/x))/2
    # 断言 acot(x).rewrite(asin) 返回表达式
    assert acot(x).rewrite(asin) == x*(-asin(sqrt(-x**2)/sqrt(-x**2 - 1)) + pi/2)*sqrt(x**(-2))
    # 断言 acot(x).rewrite(acos) 返回表达式
    assert acot(x).rewrite(acos) == x*sqrt(x**(-2))*acos(sqrt(-x**2)/sqrt(-x**2 - 1))
    # 断言 acot(x).rewrite(atan) 返回 atan(1/x)
    assert acot(x).rewrite(atan) == atan(1/x)
    # 断言 acot(x).rewrite(asec) 返回表达式
    assert acot(x).rewrite(asec) == x*sqrt(x**(-2))*asec(sqrt((x**2 + 1)/x**2))
    # 断言 acot(x).rewrite(acsc) 返回表达式
    assert acot(x).rewrite(acsc) == x*(-acsc(sqrt((x**2 + 1)/x**2)) + pi/2)*sqrt(x**(-2))

    # 断言 acot(-I/5).evalf() 等于 acot(x).rewrite(log).evalf(subs={x:-I/5})
    assert acot(-I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:-I/5})
    # 断言 acot(I/5).evalf() 等于 acot(x).rewrite(log).evalf(subs={x:I/5})
    assert acot(I/5).evalf() == acot(x).rewrite(log).evalf(subs={x:I/5})

def test_acot_fdiff():
    # 断言 acot(x).fdiff() 返回 -1/(x**2 + 1)
    assert acot(x).fdiff() == -1/(x**2 + 1)
    # 断言调用 acot(x).fdiff(2) 抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: acot(x).fdiff(2))

def test_acot_leading_term():
    # 断言 acot(1/x).as_leading_term(x) 返回 x
    assert acot(1/x).as_leading_term(x) == x
    # 测试关于分支点的情况
    assert acot(x + I).as_leading_term(x, cdir=1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x + I).as_leading_term(x, cdir=-1) == I*log(x)/2 + pi/4 - I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=1) == -I*log(x)/2 + pi/4 + I*log(2)/2
    assert acot(x - I).as_leading_term(x, cdir=-1) == -I*log(x)/2 - 3*pi/4 + I*log(2)/2
    # 测试关于位于分支切口上的点的情况
    assert acot(x).as_leading_term(x, cdir=1) == pi/2
    assert acot(x).as_leading_term(x, cdir=-1) == -pi/2
    assert acot(x + I/2).as_leading_term(x, cdir=1) == pi - I*acoth(S(1)/2)
    assert acot(x + I/2).as_leading_term(x, cdir=-1) == -I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=1) == I*acoth(S(1)/2)
    assert acot(x - I/2).as_leading_term(x, cdir=-1) == -pi + I*acoth(S(1)/2)
    # 测试关于 re(ndir) == 0 的情况
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=1) == -pi/2 - I*log(3)/2
    # 断言：验证表达式是否为真
    assert acot(I/2 - I*x - x**2).as_leading_term(x, cdir=-1) == -pi/2 - I*log(3)/2
    # 检查 acot 函数返回的表达式的主导项（leading term），在 x 接近负无穷方向（cdir=-1）时是否等于 -pi/2 - I*log(3)/2
def test_attributes():
    # 断言 sin(x) 的参数应该是 (x,)
    assert sin(x).args == (x,)


def test_sincos_rewrite():
    # 检查 sin(pi/2 - x) 是否等于 cos(x)
    assert sin(pi/2 - x) == cos(x)
    # 检查 sin(pi - x) 是否等于 sin(x)
    assert sin(pi - x) == sin(x)
    # 检查 cos(pi/2 - x) 是否等于 sin(x)
    assert cos(pi/2 - x) == sin(x)
    # 检查 cos(pi - x) 是否等于 -cos(x)
    assert cos(pi - x) == -cos(x)


def _check_even_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> f(x)
    arg : -x
    """
    # 返回检查结果，确保表达式已经使用 f(-x) -> f(x) 重写
    return func(arg).args[0] == -arg


def _check_odd_rewrite(func, arg):
    """Checks that the expr has been rewritten using f(-x) -> -f(x)
    arg : -x
    """
    # 返回检查结果，确保表达式已经使用 f(-x) -> -f(x) 重写
    return func(arg).func.is_Mul


def _check_no_rewrite(func, arg):
    """Checks that the expr is not rewritten"""
    # 返回检查结果，确保表达式没有被重写
    return func(arg).args[0] == arg


def test_evenodd_rewrite():
    # 设置测试用例
    a = cos(2)  # negative
    b = sin(1)  # positive
    even = [cos]
    odd = [sin, tan, cot, asin, atan, acot]
    with_minus = [-1, -2**1024 * E, -pi/105, -x*y, -x - y]
    # 针对每个偶函数进行测试
    for func in even:
        for expr in with_minus:
            assert _check_even_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        # 断言函数在 x-y 和 y-x 两种形式下结果相同，规范形式不影响结果
        assert func(
            x - y) == func(y - x)  # it doesn't matter which form is canonical
    # 针对每个奇函数进行测试
    for func in odd:
        for expr in with_minus:
            assert _check_odd_rewrite(func, expr)
        assert _check_no_rewrite(func, a*b)
        # 断言函数在 x-y 和 y-x 两种形式下结果为相反数，规范形式不影响结果
        assert func(
            x - y) == -func(y - x)  # it doesn't matter which form is canonical


def test_as_leading_term_issue_5272():
    # 断言 sin(x) 的主导项为 x
    assert sin(x).as_leading_term(x) == x
    # 断言 cos(x) 的主导项为 1
    assert cos(x).as_leading_term(x) == 1
    # 断言 tan(x) 的主导项为 x
    assert tan(x).as_leading_term(x) == x
    # 断言 cot(x) 的主导项为 1/x
    assert cot(x).as_leading_term(x) == 1/x


def test_leading_terms():
    # 断言 sin(1/x) 的主导项为 AccumBounds(-1, 1)
    assert sin(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    # 断言 sin(S.Half) 的主导项为 sin(S.Half)
    assert sin(S.Half).as_leading_term(x) == sin(S.Half)
    # 断言 cos(1/x) 的主导项为 AccumBounds(-1, 1)
    assert cos(1/x).as_leading_term(x) == AccumBounds(-1, 1)
    # 断言 cos(S.Half) 的主导项为 cos(S.Half)
    assert cos(S.Half).as_leading_term(x) == cos(S.Half)
    # 断言 sec(1/x) 的主导项为 AccumBounds(-∞, ∞)
    assert sec(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    # 断言 csc(1/x) 的主导项为 AccumBounds(-∞, ∞)
    assert csc(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    # 断言 tan(1/x) 的主导项为 AccumBounds(-∞, ∞)
    assert tan(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    # 断言 cot(1/x) 的主导项为 AccumBounds(-∞, ∞)
    assert cot(1/x).as_leading_term(x) == AccumBounds(S.NegativeInfinity, S.Infinity)
    # 断言给定函数的主导项满足 https://github.com/sympy/sympy/issues/21038
    f = sin(pi*(x + 4))/(3*x)
    assert f.as_leading_term(x) == pi/3


def test_atan2_expansion():
    # 检查 atan2(x**2, x + 1) 的导数是否等于 atan(x**2/(x + 1)) 的导数
    assert cancel(atan2(x**2, x + 1).diff(x) - atan(x**2/(x + 1)).diff(x)) == 0
    # 检查 atan(y/x) 在 y=0 的级数展开是否等于 atan2(y, x) 在 y=0 的级数展开
    assert cancel(atan(y/x).series(y, 0, 5) - atan2(y, x).series(y, 0, 5)
                  + atan2(0, x) - atan(0)) == O(y**5)
    # 检查 atan(y/x) 在 x=1 的级数展开是否等于 atan2(y, x) 在 x=1 的级数展开
    assert cancel(atan(y/x).series(x, 1, 4) - atan2(y, x).series(x, 1, 4)
                  + atan2(y, 1) - atan(y)) == O((x - 1)**4, (x, 1))
    # 检查 atan((y + x)/x) 在 x=1 的级数展开是否等于 atan2(y + x, x) 在 x=1 的级数展开
    assert cancel(atan((y + x)/x).series(x, 1, 3) - atan2(y + x, x).series(x, 1, 3)
                  + atan2(1 + y, 1) - atan(1 + y)) == O((x - 1)**3, (x, 1))
    # 断言 Matrix([atan2(y, x)]).jacobian([y, x]) 结果为 Matrix([[x/(y**2 + x**2), -y/(y**2 + x**2)]])
    assert Matrix([atan2(y, x)]).jacobian([y, x]) == \
        Matrix([[x/(y**2 + x**2), -y/(y**2 + x**2)]])


def test_aseries():
    # 这个测试函数没有实现内容，保留空白
    pass
    # 定义函数 t，用于测试数值函数的级数展开近似精度
    def t(n, v, d, e):
        # 使用断言确保以下条件成立：
        # 对于反正切函数 n(1/v) 的数值评估结果，
        # 与在点 v 处展开级数并移除高阶项后再代入 v 的结果之差的绝对值小于 e
        assert abs(
            n(1/v).evalf() - n(1/x).series(x, dir=d).removeO().subs(x, v)) < e

    # 调用 t 函数进行四组测试：
    # 1. 测试反正切函数 atan 在 v=0.1 处，向正方向（'+') 的级数展开精度是否满足要求 e=1e-5
    t(atan, 0.1, '+', 1e-5)
    # 2. 测试反正切函数 atan 在 v=-0.1 处，向负方向（'-'） 的级数展开精度是否满足要求 e=1e-5
    t(atan, -0.1, '-', 1e-5)
    # 3. 测试反余切函数 acot 在 v=0.1 处，向正方向（'+') 的级数展开精度是否满足要求 e=1e-5
    t(acot, 0.1, '+', 1e-5)
    # 4. 测试反余切函数 acot 在 v=-0.1 处，向负方向（'-'） 的级数展开精度是否满足要求 e=1e-5
    t(acot, -0.1, '-', 1e-5)
def test_issue_4420():
    # 定义整数符号变量 i
    i = Symbol('i', integer=True)
    # 定义偶数符号变量 e
    e = Symbol('e', even=True)
    # 定义奇数符号变量 o
    o = Symbol('o', odd=True)

    # 对于未知奇偶性的变量，验证余弦值应为1
    assert cos(4*i*pi) == 1
    # 对于未知奇偶性的变量，验证正弦值应为0
    assert sin(4*i*pi) == 0
    # 对于未知奇偶性的变量，验证正切值应为0
    assert tan(4*i*pi) == 0
    # 对于未知奇偶性的变量，验证余切值应为无穷
    assert cot(4*i*pi) is zoo

    # 对于未知奇偶性的变量，验证余弦值应等于余弦(pi*i)的值，即+/-1
    assert cos(3*i*pi) == cos(pi*i)  # +/-1
    # 对于未知奇偶性的变量，验证正弦值应为0
    assert sin(3*i*pi) == 0
    # 对于未知奇偶性的变量，验证正切值应为0
    assert tan(3*i*pi) == 0
    # 对于未知奇偶性的变量，验证余切值应为无穷
    assert cot(3*i*pi) is zoo

    # 对于未知奇偶性的浮点数变量，验证余弦值应为1
    assert cos(4.0*i*pi) == 1
    # 对于未知奇偶性的浮点数变量，验证正弦值应为0
    assert sin(4.0*i*pi) == 0
    # 对于未知奇偶性的浮点数变量，验证正切值应为0
    assert tan(4.0*i*pi) == 0
    # 对于未知奇偶性的浮点数变量，验证余切值应为无穷
    assert cot(4.0*i*pi) is zoo

    # 对于未知奇偶性的浮点数变量，验证余弦值应等于余弦(pi*i)的值，即+/-1
    assert cos(3.0*i*pi) == cos(pi*i)  # +/-1
    # 对于未知奇偶性的浮点数变量，验证正弦值应为0
    assert sin(3.0*i*pi) == 0
    # 对于未知奇偶性的浮点数变量，验证正切值应为0
    assert tan(3.0*i*pi) == 0
    # 对于未知奇偶性的浮点数变量，验证余切值应为无穷
    assert cot(3.0*i*pi) is zoo

    # 对于未知奇偶性的浮点数变量，验证余弦值应等于余弦(0.5*pi*i)的值
    assert cos(4.5*i*pi) == cos(0.5*pi*i)
    # 对于未知奇偶性的浮点数变量，验证正弦值应等于正弦(0.5*pi*i)的值
    assert sin(4.5*i*pi) == sin(0.5*pi*i)
    # 对于未知奇偶性的浮点数变量，验证正切值应等于正切(0.5*pi*i)的值
    assert tan(4.5*i*pi) == tan(0.5*pi*i)
    # 对于未知奇偶性的浮点数变量，验证余切值应等于余切(0.5*pi*i)的值
    assert cot(4.5*i*pi) == cot(0.5*pi*i)

    # 对于已知偶数的变量，验证余弦值应为1
    assert cos(4*e*pi) == 1
    # 对于已知偶数的变量，验证正弦值应为0
    assert sin(4*e*pi) == 0
    # 对于已知偶数的变量，验证正切值应为0
    assert tan(4*e*pi) == 0
    # 对于已知偶数的变量，验证余切值应为无穷
    assert cot(4*e*pi) is zoo

    # 对于已知偶数的变量，验证余弦值应为1
    assert cos(3*e*pi) == 1
    # 对于已知偶数的变量，验证正弦值应为0
    assert sin(3*e*pi) == 0
    # 对于已知偶数的变量，验证正切值应为0
    assert tan(3*e*pi) == 0
    # 对于已知偶数的变量，验证余切值应为无穷
    assert cot(3*e*pi) is zoo

    # 对于已知偶数的浮点数变量，验证余弦值应为1
    assert cos(4.0*e*pi) == 1
    # 对于已知偶数的浮点数变量，验证正弦值应为0
    assert sin(4.0*e*pi) == 0
    # 对于已知偶数的浮点数变量，验证正切值应为0
    assert tan(4.0*e*pi) == 0
    # 对于已知偶数的浮点数变量，验证余切值应为无穷
    assert cot(4.0*e*pi) is zoo

    # 对于已知偶数的浮点数变量，验证余弦值应为1
    assert cos(3.0*e*pi) == 1
    # 对于已知偶数的浮点数变量，验证正弦值应为0
    assert sin(3.0*e*pi) == 0
    # 对于已知偶数的浮点数变量，验证正切值应为0
    assert tan(3.0*e*pi) == 0
    # 对于已知偶数的浮点数变量，验证余切值应为无穷
    assert cot(3.0*e*pi) is zoo

    # 对于已知偶数的浮点数变量，验证余弦值应等于余弦(0.5*pi*e)的值
    assert cos(4.5*e*pi) == cos(0.5*pi*e)
    # 对于已知偶数的浮点数变量，验证正弦值应等于正弦(0.5*pi*e)的值
    assert sin(4.5*e*pi) == sin(0.5*pi*e)
    # 对于已知偶数的浮点数变量，验证正切值应等于正切(0.5*pi*e)的值
    assert tan(4.5*e*pi) == tan(0.5*pi*e)
    # 对于已知偶数的浮点数变量，验证余切值应等于余切(0.5*pi*e)的值
    assert cot(4.5*e*pi) == cot(0.5*pi*e)

    # 对于已知奇数的变量，验证余弦值应为1
    assert cos(4*o*pi) == 1
    # 对于已知奇数的变量，验证正弦值应为0
    assert sin(4*o*pi) == 0
    # 对于已知奇数的变量，验证正切值应为0
    assert tan(4*o*pi) == 0
    # 对于已知奇数的变量，验证余切值应为无穷
    assert cot(4*o*pi) is zoo

    # 对于已知奇数的变量，验证余弦值应为-1
    assert cos(3*o*pi) == -1
    # 对于已知奇数的变量，验证正弦值应为0
    assert sin(3*o*pi) == 0
    # 对于已知奇数的变量，验证正切值应为0
    assert tan(3*o
# 定义一个函数用于测试反函数是否正确
def test_inverses():
    # 断言调用 sin(x) 的 inverse() 方法会引发 AttributeError 异常
    raises(AttributeError, lambda: sin(x).inverse())
    # 断言调用 cos(x) 的 inverse() 方法会引发 AttributeError 异常
    raises(AttributeError, lambda: cos(x).inverse())
    # 断言 tan(x) 的反函数为 atan
    assert tan(x).inverse() == atan
    # 断言 cot(x) 的反函数为 acot
    assert cot(x).inverse() == acot
    # 断言调用 csc(x) 的 inverse() 方法会引发 AttributeError 异常
    raises(AttributeError, lambda: csc(x).inverse())
    # 断言调用 sec(x) 的 inverse() 方法会引发 AttributeError 异常
    raises(AttributeError, lambda: sec(x).inverse())
    # 断言 asin(x) 的反函数为 sin
    assert asin(x).inverse() == sin
    # 断言 acos(x) 的反函数为 cos
    assert acos(x).inverse() == cos
    # 断言 atan(x) 的反函数为 tan
    assert atan(x).inverse() == tan
    # 断言 acot(x) 的反函数为 cot
    assert acot(x).inverse() == cot


# 定义一个函数用于测试实部和虚部的计算
def test_real_imag():
    # 定义符号变量 a 和 b，且设定它们为实数
    a, b = symbols('a b', real=True)
    # 创建复数 z，其中实部为 a，虚部为 b
    z = a + b*I
    # 遍历深度为 True 和 False 的情况
    for deep in [True, False]:
        # 断言 sin(z) 的实部和虚部分解深度为 deep 时分别为 (sin(a)*cosh(b), cos(a)*sinh(b))
        assert sin(z).as_real_imag(deep=deep) == (sin(a)*cosh(b), cos(a)*sinh(b))
        # 断言 cos(z) 的实部和虚部分解深度为 deep 时分别为 (cos(a)*cosh(b), -sin(a)*sinh(b))
        assert cos(z).as_real_imag(deep=deep) == (cos(a)*cosh(b), -sin(a)*sinh(b))
        # 断言 tan(z) 的实部和虚部分解深度为 deep 时分别为 (sin(2*a)/(cos(2*a) + cosh(2*b)), sinh(2*b)/(cos(2*a) + cosh(2*b)))
        assert tan(z).as_real_imag(deep=deep) == (sin(2*a)/(cos(2*a) + cosh(2*b)), sinh(2*b)/(cos(2*a) + cosh(2*b)))
        # 断言 cot(z) 的实部和虚部分解深度为 deep 时分别为 (-sin(2*a)/(cos(2*a) - cosh(2*b)), sinh(2*b)/(cos(2*a) - cosh(2*b)))
        assert cot(z).as_real_imag(deep=deep) == (-sin(2*a)/(cos(2*a) - cosh(2*b)), sinh(2*b)/(cos(2*a) - cosh(2*b)))
        # 断言 sin(a) 的实部和虚部分解深度为 deep 时分别为 (sin(a), 0)
        assert sin(a).as_real_imag(deep=deep) == (sin(a), 0)
        # 断言 cos(a) 的实部和虚部分解深度为 deep 时分别为 (cos(a), 0)
        assert cos(a).as_real_imag(deep=deep) == (cos(a), 0)
        # 断言 tan(a) 的实部和虚部分解深度为 deep 时分别为 (tan(a), 0)
        assert tan(a).as_real_imag(deep=deep) == (tan(a), 0)
        # 断言 cot(a) 的实部和虚部分解深度为 deep 时分别为 (cot(a), 0)
        assert cot(a).as_real_imag(deep=deep) == (cot(a), 0)


# 标记为 XFAIL 的测试函数，测试带有无穷大的 sin 和 cos 函数
@XFAIL
def test_sin_cos_with_infinity():
    # 测试 issue 5196：sin(oo) 应该返回 S.NaN
    assert sin(oo) is S.NaN
    # 测试 issue 5196：cos(oo) 应该返回 S.NaN
    assert cos(oo) is S.NaN


# 标记为 slow 的测试函数，测试 sincos_rewrite_sqrt 函数
def test_sincos_rewrite_sqrt():
    # 遍历不同的 p 和 t 组合
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            # 对于一个正则 n 边形的顶点 exp(i*pi/n)，如果 n 是 Fermat 素数 p 和 2 的幂 t 的乘积，
            # 则可以通过嵌套的平方根表达。这段代码旨在检查不属于 m 边形的顶点，其中 m < n 且 gcd(i, n) == 1。
            # 对于大的 n，这使得测试变得太慢，因此限制顶点到 index `i < 10`。
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    s1 = sin(x).rewrite(sqrt)
                    c1 = cos(x).rewrite(sqrt)
                    # 断言 s1 不含有 sin 和 cos 函数，否则失败，打印失败信息
                    assert not s1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    # 断言 c1 不含有 sin 和 cos 函数，否则失败，打印失败信息
                    assert not c1.has(cos, sin), "fails for %d*pi/%d" % (i, n)
                    # 断言 sin(x) 的数值近似值与 s1 的数值近似值相差小于 1e-3，否则失败，打印失败信息
                    assert 1e-3 > abs(sin(x.evalf(5)) - s1.evalf(2)), "fails for %d*pi/%d" % (i, n)
                    # 断言 cos(x) 的数值近似值与 c1 的数值近似值相差小于 1e-3，否则失败，打印失败信息
                    assert 1e-3 > abs(cos(x.evalf(5)) - c1.evalf(2)), "fails for %d*pi/%d" % (i, n)
    # 断言 cos(pi/14) 的 rewrite(sqrt) 结果为 sqrt(cos(pi/7)/2 + S.Half)
    assert cos(pi/14).rewrite(sqrt) == sqrt(cos(pi/7)/2 + S.Half)
    # 断言 cos(pi*Rational(-15, 2)/11, evaluate=False) 的 rewrite(sqrt) 结果为 -sqrt(-cos(pi*Rational(4, 11))/2 + S.Half)
    assert cos(pi*Rational(-15, 2)/11, evaluate=False).rewrite(sqrt) == -sqrt(-cos(pi*Rational(4, 11))/2 + S.Half)
    # 断言 cos(2*pi/2) 的 rewrite(sqrt) 结果为 -1
    assert cos(Mul(2, pi, S.Half, evaluate=False), evaluate=False).rewrite(sqrt) == -1
    # 定义 e = cos(pi/3/17)，不使用 pi/15 因为在实例化时会被捕获
    e = cos(pi/3/17)
    # 断言：使用 e 对象的 rewrite 方法计算 sqrt 的结果，并与 a 进行比较
    assert e.rewrite(sqrt) == a
    # 断言：使用 e 对象的 n 方法获取结果，并与 a 对象的 n 方法返回的结果进行比较
    assert e.n() == a.n()
    # 断言：测试 fermatCoords 方法覆盖情况，此处计算 cos(pi/9/17) 的平方根重写结果，与复杂的表达式进行比较
    assert cos(pi/9/17).rewrite(sqrt) == \
        sin(pi/9)*sin(pi*Rational(2, 17)) + cos(pi/9)*cos(pi*Rational(2, 17))
@slow
# 定义一个测试函数，测试 sincos_rewrite_sqrt_257 函数
def test_sincos_rewrite_sqrt_257():
    # 断言：计算 cos(pi/257) 并对其应用 sqrt 重写，精确到小数点后64位，与原始计算结果进行比较
    assert cos(pi/257).rewrite(sqrt).evalf(64) == cos(pi/257).evalf(64)


@slow
# 定义一个测试函数，测试 tancot_rewrite_sqrt 函数
def test_tancot_rewrite_sqrt():
    # 等价于测试 rewrite(pow)
    # 遍历不同的参数 p 和 t 的组合
    for p in [1, 3, 5, 17]:
        for t in [1, 8]:
            n = t*p
            # 遍历满足条件的整数 i
            for i in range(1, min((n + 1)//2 + 1, 10)):
                if 1 == gcd(i, n):
                    x = i*pi/n
                    # 检查是否满足特定条件，如果满足，则进行以下断言
                    if  2*i != n and 3*i != 2*n:
                        # 计算 tan(x) 并对其应用 sqrt 重写
                        t1 = tan(x).rewrite(sqrt)
                        # 断言：tan(x) 的结果不含有 cot 或 tan，同时近似误差小于 1e-3
                        assert not t1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( tan(x.evalf(7)) - t1.evalf(4) ), "fails for %d*pi/%d" % (i, n)
                    if  i != 0 and i != n:
                        # 计算 cot(x) 并对其应用 sqrt 重写
                        c1 = cot(x).rewrite(sqrt)
                        # 断言：cot(x) 的结果不含有 cot 或 tan，同时近似误差小于 1e-3
                        assert not c1.has(cot, tan), "fails for %d*pi/%d" % (i, n)
                        assert 1e-3 > abs( cot(x.evalf(7)) - c1.evalf(4) ), "fails for %d*pi/%d" % (i, n)


# 定义一个测试函数，测试 sec 函数
def test_sec():
    # 声明符号变量 x 和 z
    x = symbols('x', real=True)
    z = symbols('z')

    # 断言：sec 函数的参数个数为 1
    assert sec.nargs == FiniteSet(1)

    # 一系列特定输入的预期输出断言
    assert sec(zoo) is nan
    assert sec(0) == 1
    assert sec(pi) == -1
    assert sec(pi/2) is zoo
    assert sec(-pi/2) is zoo
    assert sec(pi/6) == 2*sqrt(3)/3
    assert sec(pi/3) == 2
    assert sec(pi*Rational(5, 2)) is zoo
    assert sec(pi*Rational(9, 7)) == -sec(pi*Rational(2, 7))
    assert sec(pi*Rational(3, 4)) == -sqrt(2)  # issue 8421
    assert sec(I) == 1/cosh(1)
    assert sec(x*I) == 1/cosh(x)
    assert sec(-x) == sec(x)

    assert sec(asec(x)) == x

    assert sec(z).conjugate() == sec(conjugate(z))

    assert (sec(z).as_real_imag() ==
    (cos(re(z))*cosh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2),
     sin(re(z))*sinh(im(z))/(sin(re(z))**2*sinh(im(z))**2 +
                             cos(re(z))**2*cosh(im(z))**2)))

    assert sec(x).expand(trig=True) == 1/cos(x)
    assert sec(2*x).expand(trig=True) == 1/(2*cos(x)**2 - 1)

    assert sec(x).is_extended_real == True
    assert sec(z).is_real == None

    assert sec(a).is_algebraic is None
    assert sec(na).is_algebraic is False

    assert sec(x).as_leading_term() == sec(x)

    assert sec(0, evaluate=False).is_finite == True
    assert sec(x).is_finite == None
    assert sec(pi/2, evaluate=False).is_finite == False

    assert series(sec(x), x, x0=0, n=6) == 1 + x**2/2 + 5*x**4/24 + O(x**6)

    # https://github.com/sympy/sympy/issues/7166
    assert series(sqrt(sec(x))) == 1 + x**2/4 + 7*x**4/96 + O(x**6)

    # https://github.com/sympy/sympy/issues/7167
    assert (series(sqrt(sec(x)), x, x0=pi*3/2, n=4) ==
            1/sqrt(x - pi*Rational(3, 2)) + (x - pi*Rational(3, 2))**Rational(3, 2)/12 +
            (x - pi*Rational(3, 2))**Rational(7, 2)/160 + O((x - pi*Rational(3, 2))**4, (x, pi*Rational(3, 2))))

    assert sec(x).diff(x) == tan(x)*sec(x)

    # Taylor Term checks
    assert sec(z).taylor_term(4, z) == 5*z**4/24
    # 断言：验证 sec(z) 的泰勒展开中 z 的六阶项是否等于 61*z**6/720
    assert sec(z).taylor_term(6, z) == 61*z**6/720
    # 断言：验证 sec(z) 的泰勒展开中 z 的五阶项是否等于 0
    assert sec(z).taylor_term(5, z) == 0
def test_sec_rewrite():
    # 检查 sec(x) 在使用 exp 重写后的结果是否正确
    assert sec(x).rewrite(exp) == 1/(exp(I*x)/2 + exp(-I*x)/2)
    # 检查 sec(x) 在使用 cos 重写后的结果是否正确
    assert sec(x).rewrite(cos) == 1/cos(x)
    # 检查 sec(x) 在使用 tan 重写后的结果是否正确
    assert sec(x).rewrite(tan) == (tan(x/2)**2 + 1)/(-tan(x/2)**2 + 1)
    # 检查 sec(x) 在使用 pow 重写后的结果是否正确
    assert sec(x).rewrite(pow) == sec(x)
    # 检查 sec(x) 在使用 sqrt 重写后的结果是否正确
    assert sec(x).rewrite(sqrt) == sec(x)
    # 检查 sec(z) 在使用 cot 重写后的结果是否正确
    assert sec(z).rewrite(cot) == (cot(z/2)**2 + 1)/(cot(z/2)**2 - 1)
    # 检查 sec(x) 在使用 sin 重写后的结果是否正确，其中 evaluate=False 表示不自动计算
    assert sec(x).rewrite(sin) == 1 / sin(x + pi / 2, evaluate=False)
    # 再次检查 sec(x) 在使用 tan 重写后的结果是否正确
    assert sec(x).rewrite(tan) == (tan(x / 2)**2 + 1) / (-tan(x / 2)**2 + 1)
    # 检查 sec(x) 在使用 csc 重写后的结果是否正确，其中 evaluate=False 表示不自动计算
    assert sec(x).rewrite(csc) == csc(-x + pi/2, evaluate=False)
    # 检查 sec(x) 在使用 besselj 重写后的结果是否正确，使用 Piecewise 处理不同情况
    assert sec(x).rewrite(besselj) == Piecewise(
                (sqrt(2)/(sqrt(pi*x)*besselj(-S.Half, x)), Ne(x, 0)),
                (1, True)
            )
    # 检查 sec(x) 在使用 besselj 重写后，再代入 x=0 后的结果是否正确
    assert sec(x).rewrite(besselj).subs(x, 0) == sec(0)


def test_sec_fdiff():
    # 检查 sec(x) 的一阶导数是否正确
    assert sec(x).fdiff() == tan(x)*sec(x)
    # 检查调用 sec(x) 的二阶导数是否会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: sec(x).fdiff(2))


def test_csc():
    x = symbols('x', real=True)
    z = symbols('z')

    # https://github.com/sympy/sympy/issues/6707
    # 检查 cosecant 对象是否等于其备选形式 alternate
    cosecant = csc('x')
    alternate = 1/sin('x')
    assert cosecant.equals(alternate) == True
    assert alternate.equals(cosecant) == True

    # 检查 csc 函数的参数个数
    assert csc.nargs == FiniteSet(1)

    # 检查一些特殊情况下 csc 函数的返回值
    assert csc(0) is zoo
    assert csc(pi) is zoo
    assert csc(zoo) is nan

    # 检查 csc 在一些常见角度下的返回值是否正确
    assert csc(pi/2) == 1
    assert csc(-pi/2) == -1
    assert csc(pi/6) == 2
    assert csc(pi/3) == 2*sqrt(3)/3
    assert csc(pi*Rational(5, 2)) == 1
    assert csc(pi*Rational(9, 7)) == -csc(pi*Rational(2, 7))
    assert csc(pi*Rational(3, 4)) == sqrt(2)  # issue 8421
    assert csc(I) == -I/sinh(1)
    assert csc(x*I) == -I/sinh(x)
    assert csc(-x) == -csc(x)

    # 检查 csc 函数的反函数 acsc 是否正确
    assert csc(acsc(x)) == x

    # 检查 csc 函数的共轭是否正确
    assert csc(z).conjugate() == csc(conjugate(z))

    # 检查 csc 函数在展开三角函数时的结果是否正确
    assert (csc(z).as_real_imag() ==
            (sin(re(z))*cosh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                                     cos(re(z))**2*sinh(im(z))**2),
             -cos(re(z))*sinh(im(z))/(sin(re(z))**2*cosh(im(z))**2 +
                          cos(re(z))**2*sinh(im(z))**2)))

    # 检查 csc 函数在展开 trig=True 时的结果是否正确
    assert csc(x).expand(trig=True) == 1/sin(x)
    assert csc(2*x).expand(trig=True) == 1/(2*sin(x)*cos(x))

    # 检查 csc 函数是否为扩展实数
    assert csc(x).is_extended_real == True
    assert csc(z).is_real == None

    # 检查 csc 函数是否为代数函数
    assert csc(a).is_algebraic is None
    assert csc(na).is_algebraic is False

    # 检查 csc 函数的主导项是否正确
    assert csc(x).as_leading_term() == csc(x)

    # 检查 csc 函数在 evaluate=False 时是否为有限值
    assert csc(0, evaluate=False).is_finite == False
    assert csc(x).is_finite == None
    assert csc(pi/2, evaluate=False).is_finite == True

    # 检查 csc 函数在泰勒展开时的结果是否正确
    assert series(csc(x), x, x0=pi/2, n=6) == \
        1 + (x - pi/2)**2/2 + 5*(x - pi/2)**4/24 + O((x - pi/2)**6, (x, pi/2))
    assert series(csc(x), x, x0=0, n=6) == \
            1/x + x/6 + 7*x**3/360 + 31*x**5/15120 + O(x**6)

    # 检查 csc 函数的导数是否正确
    assert csc(x).diff(x) == -cot(x)*csc(x)

    # 检查 csc 函数的泰勒项是否正确
    assert csc(x).taylor_term(2, x) == 0
    assert csc(x).taylor_term(3, x) == 7*x**3/360
    assert csc(x).taylor_term(5, x) == 31*x**5/15120
    raises(ArgumentIndexError, lambda: csc(x).fdiff(2))


def test_asec():
    # 待补充
    # 创建一个符号变量 z，使其等于零
    z = Symbol('z', zero=True)
    # 断言函数 asec(z) 返回 zoo
    assert asec(z) is zoo
    # 断言函数 asec(nan) 返回 nan
    assert asec(nan) is nan
    # 断言函数 asec(1) 返回 0
    assert asec(1) == 0
    # 断言函数 asec(-1) 返回 pi
    assert asec(-1) == pi
    # 断言函数 asec(oo) 返回 pi/2
    assert asec(oo) == pi/2
    # 断言函数 asec(-oo) 返回 pi/2
    assert asec(-oo) == pi/2
    # 断言函数 asec(zoo) 返回 pi/2
    assert asec(zoo) == pi/2

    # 断言函数 asec(sec(pi*Rational(13, 4))) 返回 pi*Rational(3, 4)
    assert asec(sec(pi*Rational(13, 4))) == pi*Rational(3, 4)
    # 断言函数 asec(1 + sqrt(5)) 返回 pi*Rational(2, 5)
    assert asec(1 + sqrt(5)) == pi*Rational(2, 5)
    # 断言函数 asec(2/sqrt(3)) 返回 pi/6
    assert asec(2/sqrt(3)) == pi/6
    # 断言函数 asec(sqrt(4 - 2*sqrt(2))) 返回 pi/8
    assert asec(sqrt(4 - 2*sqrt(2))) == pi/8
    # 断言函数 asec(-sqrt(4 + 2*sqrt(2))) 返回 pi*Rational(5, 8)
    assert asec(-sqrt(4 + 2*sqrt(2))) == pi*Rational(5, 8)
    # 断言函数 asec(sqrt(2 + 2*sqrt(5)/5)) 返回 pi*Rational(3, 10)
    assert asec(sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(3, 10)
    # 断言函数 asec(-sqrt(2 + 2*sqrt(5)/5)) 返回 pi*Rational(7, 10)
    assert asec(-sqrt(2 + 2*sqrt(5)/5)) == pi*Rational(7, 10)
    # 断言函数 asec(sqrt(2) - sqrt(6)) 返回 pi*Rational(11, 12)
    assert asec(sqrt(2) - sqrt(6)) == pi*Rational(11, 12)

    # 断言函数 asec(x).diff(x) 的导数等于 1/(x**2*sqrt(1 - 1/x**2))
    assert asec(x).diff(x) == 1/(x**2*sqrt(1 - 1/x**2))

    # 断言函数 asec(x).rewrite(log) 重写为对数形式的表达式
    assert asec(x).rewrite(log) == I*log(sqrt(1 - 1/x**2) + I/x) + pi/2
    # 断言函数 asec(x).rewrite(asin) 重写为 arcsin 形式的表达式
    assert asec(x).rewrite(asin) == -asin(1/x) + pi/2
    # 断言函数 asec(x).rewrite(acos) 重写为 arccos 形式的表达式
    assert asec(x).rewrite(acos) == acos(1/x)
    # 断言函数 asec(x).rewrite(atan) 重写为 arctan 形式的表达式
    assert asec(x).rewrite(atan) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*atan(sqrt(x**2 - 1))/x
    # 断言函数 asec(x).rewrite(acot) 重写为 arccot 形式的表达式
    assert asec(x).rewrite(acot) == \
        pi*(1 - sqrt(x**2)/x)/2 + sqrt(x**2)*acot(1/sqrt(x**2 - 1))/x
    # 断言函数 asec(x).rewrite(acsc) 重写为 arccsc 形式的表达式
    assert asec(x).rewrite(acsc) == -acsc(x) + pi/2
    # 断言函数 asec(x).fdiff(2) 引发参数索引错误异常
    raises(ArgumentIndexError, lambda: asec(x).fdiff(2))
def test_asec_is_real():
    # 检查 asec(S.Half) 是否为实数，预期为 False
    assert asec(S.Half).is_real is False
    # 创建一个正整数符号 n
    n = Symbol('n', positive=True, integer=True)
    # 检查 asec(n) 是否是扩展实数，预期为 True
    assert asec(n).is_extended_real is True
    # 检查 asec(x) 是否为实数，预期为 None
    assert asec(x).is_real is None
    # 检查 asec(r) 是否为实数，预期为 None
    assert asec(r).is_real is None
    # 创建一个名为 t 的符号，其实部为非实数，但有限
    t = Symbol('t', real=False, finite=True)
    # 检查 asec(t) 是否为实数，预期为 False
    assert asec(t).is_real is False


def test_asec_leading_term():
    # 检查 asec(1/x) 的主导项，预期为 pi/2
    assert asec(1/x).as_leading_term(x) == pi/2
    # 关于分支点的测试
    assert asec(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert asec(x - 1).as_leading_term(x) == pi
    # 关于位于分支切割线上的点的测试
    assert asec(x).as_leading_term(x, cdir=1) == -I*log(x) + I*log(2)
    assert asec(x).as_leading_term(x, cdir=-1) == I*log(x) + 2*pi - I*log(2)
    assert asec(I*x + 1/2).as_leading_term(x, cdir=1) == asec(1/2)
    assert asec(-I*x + 1/2).as_leading_term(x, cdir=1) == -asec(1/2)
    assert asec(I*x - 1/2).as_leading_term(x, cdir=1) == 2*pi - asec(-1/2)
    assert asec(-I*x - 1/2).as_leading_term(x, cdir=1) == asec(-1/2)
    # 关于 im(ndir) == 0 的测试
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == pi + I*log(2 - sqrt(3))
    assert asec(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == pi + I*log(2 - sqrt(3))


def test_asec_series():
    # 检查 asec(x) 的级数展开，到 x^9 项
    assert asec(x).series(x, 0, 9) == \
        I*log(2) - I*log(x) - I*x**2/4 - 3*I*x**4/32 \
        - 5*I*x**6/96 - 35*I*x**8/1024 + O(x**9)
    # 获取 asec(x) 的 x^4 项
    t4 = asec(x).taylor_term(4, x)
    assert t4 == -3*I*x**4/32
    # 检查 asec(x) 的 x^6 项
    assert asec(x).taylor_term(6, x, t4, 0) == -5*I*x**6/96


def test_acsc():
    # 检查 acsc(nan)，预期为 nan
    assert acsc(nan) is nan
    # 检查 acsc(1)，预期为 pi/2
    assert acsc(1) == pi/2
    # 检查 acsc(-1)，预期为 -pi/2
    assert acsc(-1) == -pi/2
    # 检查 acsc(oo)，预期为 0
    assert acsc(oo) == 0
    # 检查 acsc(-oo)，预期为 0
    assert acsc(-oo) == 0
    # 检查 acsc(zoo)，预期为 0
    assert acsc(zoo) == 0
    # 检查 acsc(0)，预期为 zoo
    assert acsc(0) is zoo

    # 关于 csc 的逆操作的测试
    assert acsc(csc(3)) == -3 + pi
    assert acsc(csc(4)) == -4 + pi
    assert acsc(csc(6)) == 6 - 2*pi
    assert unchanged(acsc, csc(x))
    assert unchanged(acsc, sec(x))

    # 具体数值的测试
    assert acsc(2/sqrt(3)) == pi/3
    assert acsc(csc(pi*Rational(13, 4))) == -pi/4
    assert acsc(sqrt(2 + 2*sqrt(5)/5)) == pi/5
    assert acsc(-sqrt(2 + 2*sqrt(5)/5)) == -pi/5
    assert acsc(-2) == -pi/6
    assert acsc(-sqrt(4 + 2*sqrt(2))) == -pi/8
    assert acsc(sqrt(4 - 2*sqrt(2))) == pi*Rational(3, 8)
    assert acsc(1 + sqrt(5)) == pi/10
    assert acsc(sqrt(2) - sqrt(6)) == pi*Rational(-5, 12)

    # 检查 acsc(x) 对 x 的导数
    assert acsc(x).diff(x) == -1/(x**2*sqrt(1 - 1/x**2))

    # 使用不同函数重写 acsc(x) 的测试
    assert acsc(x).rewrite(log) == -I*log(sqrt(1 - 1/x**2) + I/x)
    assert acsc(x).rewrite(asin) == asin(1/x)
    assert acsc(x).rewrite(acos) == -acos(1/x) + pi/2
    assert acsc(x).rewrite(atan) == \
        (-atan(sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(acot) == (-acot(1/sqrt(x**2 - 1)) + pi/2)*sqrt(x**2)/x
    assert acsc(x).rewrite(asec) == -asec(x) + pi/2
    raises(ArgumentIndexError, lambda: acsc(x).fdiff(2))


def test_csc_rewrite():
    # 检查 csc(x) 的重写，使用幂函数返回自身
    assert csc(x).rewrite(pow) == csc(x)
    # 检查 csc(x) 的重写，使用平方根函数返回自身
    assert csc(x).rewrite(sqrt) == csc(x)
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下的重写表达式是否符合预期
    assert csc(x).rewrite(exp) == 2*I/(exp(I*x) - exp(-I*x))
    
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下使用 sin 函数重写的表达式是否符合预期
    assert csc(x).rewrite(sin) == 1/sin(x)
    
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下使用 tan 函数重写的表达式是否符合预期
    assert csc(x).rewrite(tan) == (tan(x/2)**2 + 1)/(2*tan(x/2))
    
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下使用 cot 函数重写的表达式是否符合预期
    assert csc(x).rewrite(cot) == (cot(x/2)**2 + 1)/(2*cot(x/2))
    
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下使用 cos 函数重写的表达式是否符合预期
    assert csc(x).rewrite(cos) == 1/cos(x - pi/2, evaluate=False)
    
    # 使用 SymPy 中的 csc 函数，测试其在不同情况下使用 sec 函数重写的表达式是否符合预期
    assert csc(x).rewrite(sec) == sec(-x + pi/2, evaluate=False)
    
    # issue 17349 的特定测试情况
    assert csc(1 - exp(-besselj(I, I))).rewrite(cos) == \
           -1/cos(-pi/2 - 1 + cos(I*besselj(I, I)) +
                  I*cos(-pi/2 + I*besselj(I, I), evaluate=False), evaluate=False)
    
    # 使用 SymPy 中的 csc 函数，测试其在 besselj 函数下的重写表达式是否符合预期
    assert csc(x).rewrite(besselj) == sqrt(2)/(sqrt(pi*x)*besselj(S.Half, x))
    
    # 将 x 替换为 0 后，使用 SymPy 中的 csc 函数，检查其在 besselj 重写后表达式是否符合预期
    assert csc(x).rewrite(besselj).subs(x, 0) == csc(0)
def test_acsc_leading_term():
    assert acsc(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acsc(x + 1).as_leading_term(x) == pi/2
    assert acsc(x - 1).as_leading_term(x) == -pi/2
    # Tests concerning points lying on branch cuts
    assert acsc(x).as_leading_term(x, cdir=1) == I*log(x) + pi/2 - I*log(2)
    assert acsc(x).as_leading_term(x, cdir=-1) == -I*log(x) - 3*pi/2 + I*log(2)
    assert acsc(I*x + 1/2).as_leading_term(x, cdir=1) == acsc(1/2)
    assert acsc(-I*x + 1/2).as_leading_term(x, cdir=1) == pi - acsc(1/2)
    assert acsc(I*x - 1/2).as_leading_term(x, cdir=1) == -pi - acsc(-1/2)
    assert acsc(-I*x - 1/2).as_leading_term(x, cdir=1) == -acsc(1/2)
    # Tests concerning im(ndir) == 0
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=1) == -pi/2 + I*log(sqrt(3) + 2)
    assert acsc(-I*x**2 + x - S(1)/2).as_leading_term(x, cdir=-1) == -pi/2 + I*log(sqrt(3) + 2)


def test_acsc_series():
    assert acsc(x).series(x, 0, 9) == \
        -I*log(2) + pi/2 + I*log(x) + I*x**2/4 \
        + 3*I*x**4/32 + 5*I*x**6/96 + 35*I*x**8/1024 + O(x**9)
    # Extracts the 6th-order Taylor term for arcsine of cosecant
    t6 = acsc(x).taylor_term(6, x)
    assert t6 == 5*I*x**6/96
    # Evaluates the 8th-order Taylor term for arcsine of cosecant with specific known terms
    assert acsc(x).taylor_term(8, x, t6, 0) == 35*I*x**8/1024


def test_asin_nseries():
    # Evaluates the asymptotic series expansion of arcsine function
    assert asin(x + 2)._eval_nseries(x, 4, None, I) == -asin(2) + pi + \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x + 2)._eval_nseries(x, 4, None, -I) == asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, I) == -asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asin(x - 2)._eval_nseries(x, 4, None, -I) == asin(2) - pi + \
    sqrt(3)*I*x/3 + sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # testing nseries for asin at branch points
    assert asin(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/12 - 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    assert asin(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    assert asin(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(-x) + \
    sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    assert asin(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
    sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)


def test_acos_nseries():
    assert acos(x + 2)._eval_nseries(x, 4, None, I) == -acos(2) - sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x + 2)._eval_nseries(x, 4, None, -I) == acos(2) + sqrt(3)*I*x/3 - \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, I) == acos(-2) + sqrt(3)*I*x/3 + \
    sqrt(3)*I*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acos(x - 2)._eval_nseries(x, 4, None, -I) == -acos(-2) + 2*pi - \
    ```
    # 计算表达式 sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    sqrt(3)*I*x/3 - sqrt(3)*I*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    
    # 对 acos 在分支点处进行 nseries 测试
    assert acos(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) + \
        sqrt(2)*(-x)**(S(3)/2)/12 + 3*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    
    # 对 acos 在分支点处进行 nseries 测试
    assert acos(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) - \
        sqrt(2)*x**(S(3)/2)/12 - 3*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    
    # 对 acos(exp(x)) 在分支点处进行 nseries 测试
    assert acos(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(-x) - \
        sqrt(2)*(-x)**(S(3)/2)/6 + sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
    
    # 对 acos(-exp(x)) 在分支点处进行 nseries 测试
    assert acos(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
        sqrt(2)*(-x)**(S(3)/2)/6 - sqrt(2)*(-x)**(S(5)/2)/120 + O(x**3)
def test_atan_nseries():
    # Assert statement testing the Taylor series expansion of atan(x + 2*I) around x, with specified terms and direction
    assert atan(x + 2*I)._eval_nseries(x, 4, None, 1) == I*atanh(2) - x/3 - \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of atan(x + 2*I) around x, with specified terms and direction
    assert atan(x + 2*I)._eval_nseries(x, 4, None, -1) == I*atanh(2) - pi - \
    x/3 - 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of atan(x - 2*I) around x, with specified terms and direction
    assert atan(x - 2*I)._eval_nseries(x, 4, None, 1) == -I*atanh(2) + pi - \
    x/3 + 2*I*x**2/9 + 13*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of atan(x - 2*I) around x, with specified terms and direction
    assert atan(x - 2*I)._eval_nseries(x, 4, None, -1) == -I*atanh(2) - x/3 + \
    2*I*x**2/9 + 13*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of atan(1/x) around x, with specified terms and direction
    assert atan(1/x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    # Assert statement testing the Taylor series expansion of atan(1/x) around x, with specified terms and direction
    assert atan(1/x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # Assert statement testing the Taylor series expansion of atan(x + I) around x, with specified terms (no direction specified)
    assert atan(x + I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 + x/4 + I*x**2/16 - x**3/48 + O(x**4)
    # Assert statement testing the Taylor series expansion of atan(x - I) around x, with specified terms (no direction specified)
    assert atan(x - I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 + x/4 - I*x**2/16 - x**3/48 + O(x**4)


def test_acot_nseries():
    # Assert statement testing the Taylor series expansion of acot(x + S(1)/2*I) around x, with specified terms and direction
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, 1) == -I*acoth(S(1)/2) + \
    pi - 4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of acot(x + S(1)/2*I) around x, with specified terms and direction
    assert acot(x + S(1)/2*I)._eval_nseries(x, 4, None, -1) == -I*acoth(S(1)/2) - \
    4*x/3 + 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of acot(x - S(1)/2*I) around x, with specified terms and direction
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, 1) == I*acoth(S(1)/2) - \
    4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of acot(x - S(1)/2*I) around x, with specified terms and direction
    assert acot(x - S(1)/2*I)._eval_nseries(x, 4, None, -1) == I*acoth(S(1)/2) - \
    pi - 4*x/3 - 8*I*x**2/9 + 112*x**3/81 + O(x**4)
    # Assert statement testing the Taylor series expansion of acot(x) around x, with specified terms and direction
    assert acot(x)._eval_nseries(x, 2, None, 1) == pi/2 - x + O(x**2)
    # Assert statement testing the Taylor series expansion of acot(x) around x, with specified terms and direction
    assert acot(x)._eval_nseries(x, 2, None, -1) == -pi/2 - x + O(x**2)
    # Assert statement testing the Taylor series expansion of acot(x + I) around x, with specified terms (no direction specified)
    assert acot(x + I)._eval_nseries(x, 4, None) == -I*log(2)/2 + pi/4 + \
    I*log(x)/2 - x/4 - I*x**2/16 + x**3/48 + O(x**4)
    # Assert statement testing the Taylor series expansion of acot(x - I) around x, with specified terms (no direction specified)
    assert acot(x - I)._eval_nseries(x, 4, None) == I*log(2)/2 + pi/4 - \
    I*log(x)/2 - x/4 + I*x**2/16 + x**3/48 + O(x**4)


def test_asec_nseries():
    # Assert statement testing the Taylor series expansion of asec(x + S(1)/2) around x, with specified terms and direction (I)
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, I) == asec(S(1)/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # Assert statement testing the Taylor series expansion of asec(x + S(1)/2) around x, with specified terms and direction (-I)
    assert asec(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -asec(S(1)/2) + \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # Assert statement testing the Taylor series expansion of asec(x - S(1)/2) around x, with specified terms and direction (I)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, I) == -asec(-S(1)/2) + \
    2*pi + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    # Assert statement testing the Taylor series expansion of asec(x - S(1)/2) around x, with specified terms and direction (-I)
    assert asec(x - S(1)/2)._eval_nseries(x, 4, None, -I) == asec(-S(1)/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # Assert statement testing the Taylor series expansion of asec(1 + x) around x, with specified terms (no direction specified)
    assert asec(1 + x)._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)
    # Assert statement testing the Taylor series expansion of asec(-1 + x) around x, with specified terms (no direction specified)
    assert asec(-1 + x)._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(-x) + \
    5*sqrt(2)*(-x)**(S(3)/2)/12 - 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)
    # 断言，验证 asec(exp(x)) 的前三项泰勒级数展开是否等于预期表达式
    assert asec(exp(x))._eval_nseries(x, 3, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    # 断言，验证 asec(-exp(x)) 的前三项泰勒级数展开是否等于预期表达式
    assert asec(-exp(x))._eval_nseries(x, 3, None) == pi - sqrt(2)*sqrt(x) + \
    sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)
# 测试 acsc 函数的 nseries 方法，计算 x + S(1)/2 的 nseries 展开结果
assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) + \
4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)

# 测试 acsc 函数的 nseries 方法，计算 x + S(1)/2 的 nseries 展开结果（负虚数单位）
assert acsc(x + S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
pi - 4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)

# 测试 acsc 函数的 nseries 方法，计算 x - S(1)/2 的 nseries 展开结果
assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, I) == acsc(S(1)/2) - pi -\
4*sqrt(3)*I*x/3 - 8*sqrt(3)*I*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)

# 测试 acsc 函数的 nseries 方法，计算 x - S(1)/2 的 nseries 展开结果（负虚数单位）
assert acsc(x - S(1)/2)._eval_nseries(x, 4, None, -I) == -acsc(S(1)/2) + \
4*sqrt(3)*I*x/3 + 8*sqrt(3)*I*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)

# 在分支点处测试 acsc 函数的 nseries 方法
assert acsc(1 + x)._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
5*sqrt(2)*x**(S(3)/2)/12 - 43*sqrt(2)*x**(S(5)/2)/160 + O(x**3)

# 在分支点处测试 acsc 函数的 nseries 方法
assert acsc(-1 + x)._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(-x) - \
5*sqrt(2)*(-x)**(S(3)/2)/12 + 43*sqrt(2)*(-x)**(S(5)/2)/160 + O(x**3)

# 测试 acsc 函数的 nseries 方法，对 exp(x) 的结果进行展开
assert acsc(exp(x))._eval_nseries(x, 3, None) == pi/2 - sqrt(2)*sqrt(x) + \
sqrt(2)*x**(S(3)/2)/6 - sqrt(2)*x**(S(5)/2)/120 + O(x**3)

# 测试 acsc 函数的 nseries 方法，对 -exp(x) 的结果进行展开
assert acsc(-exp(x))._eval_nseries(x, 3, None) == -pi/2 + sqrt(2)*sqrt(x) - \
sqrt(2)*x**(S(3)/2)/6 + sqrt(2)*x**(S(5)/2)/120 + O(x**3)
    # 断言：对给定的正弦余割函数进行调用，检查其返回的实部属性是否为 None
    assert acsc(p).is_real is None
    # 断言：对给定的正弦余割函数进行调用，检查其返回的实部属性是否为 None
    assert acsc(n).is_real is None
    # 断言：对给定的正切函数进行调用，检查其返回的正属性是否为 True
    assert atan(p).is_positive is True
    # 断言：对给定的正切函数进行调用，检查其返回的负属性是否为 True
    assert atan(n).is_negative is True
    # 断言：对给定的余切函数进行调用，检查其返回的正属性是否为 True
    assert acot(p).is_positive is True
    # 断言：对给定的余切函数进行调用，检查其返回的负属性是否为 True
    assert acot(n).is_negative is True
`
# 定义一个测试函数，用于检验 issue 14320 的数学函数的正确性
def test_issue_14320():
    # 检验 asin 和 sin 的关系
    assert asin(sin(2)) == -2 + pi and (-pi/2 <= -2 + pi <= pi/2) and sin(2) == sin(-2 + pi)
    # 检验 asin 和 cos 的关系
    assert asin(cos(2)) == -2 + pi/2 and (-pi/2 <= -2 + pi/2 <= pi/2) and cos(2) == sin(-2 + pi/2)
    # 检验 acos 和 sin 的关系
    assert acos(sin(2)) == -pi/2 + 2 and (0 <= -pi/2 + 2 <= pi) and sin(2) == cos(-pi/2 + 2)
    # 检验 acos 和 cos 的关系
    assert acos(cos(20)) == -6*pi + 20 and (0 <= -6*pi + 20 <= pi) and cos(20) == cos(-6*pi + 20)
    # 检验 acos 和 cos 的关系
    assert acos(cos(30)) == -30 + 10*pi and (0 <= -30 + 10*pi <= pi) and cos(30) == cos(-30 + 10*pi)

    # 检验 atan 和 tan 的关系
    assert atan(tan(17)) == -5*pi + 17 and (-pi/2 < -5*pi + 17 < pi/2) and tan(17) == tan(-5*pi + 17)
    # 检验 atan 和 tan 的关系
    assert atan(tan(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 < pi/2) and tan(15) == tan(-5*pi + 15)
    # 检验 atan 和 cot 的关系
    assert atan(cot(12)) == -12 + pi*Rational(7, 2) and (-pi/2 < -12 + pi*Rational(7, 2) < pi/2) and cot(12) == tan(-12 + pi*Rational(7, 2))
    # 检验 acot 和 cot 的关系
    assert acot(cot(15)) == -5*pi + 15 and (-pi/2 < -5*pi + 15 <= pi/2) and cot(15) == cot(-5*pi + 15)
    # 检验 acot 和 tan 的关系
    assert acot(tan(19)) == -19 + pi*Rational(13, 2) and (-pi/2 < -19 + pi*Rational(13, 2) <= pi/2) and tan(19) == cot(-19 + pi*Rational(13, 2))

    # 检验 asec 和 sec 的关系
    assert asec(sec(11)) == -11 + 4*pi and (0 <= -11 + 4*pi <= pi) and cos(11) == cos(-11 + 4*pi)
    # 检验 asec 和 csc 的关系
    assert asec(csc(13)) == -13 + pi*Rational(9, 2) and (0 <= -13 + pi*Rational(9, 2) <= pi) and sin(13) == cos(-13 + pi*Rational(9, 2))
    # 检验 acsc 和 csc 的关系
    assert acsc(csc(14)) == -4*pi + 14 and (-pi/2 <= -4*pi + 14 <= pi/2) and sin(14) == sin(-4*pi + 14)
    # 检验 acsc 和 sec 的关系
    assert acsc(sec(10)) == pi*Rational(-7, 2) + 10 and (-pi/2 <= pi*Rational(-7, 2) + 10 <= pi/2) and cos(10) == sin(pi*Rational(-7, 2) + 10)

# 定义一个测试函数，用于检验 issue 14543 的三角函数的关系
def test_issue_14543():
    # 检验 sec 的周期性质
    assert sec(2*pi + 11) == sec(11)
    assert sec(2*pi - 11) == sec(11)
    assert sec(pi + 11) == -sec(11)
    assert sec(pi - 11) == -sec(11)

    # 检验 csc 的周期性质
    assert csc(2*pi + 17) == csc(17)
    assert csc(2*pi - 17) == -csc(17)
    assert csc(pi + 17) == -csc(17)
    assert csc(pi - 17) == csc(17)

    # 检验 csc 和 sec 的关系
    x = Symbol('x')
    assert csc(pi/2 + x) == sec(x)
    assert csc(pi/2 - x) == sec(x)
    assert csc(pi*Rational(3, 2) + x) == -sec(x)
    assert csc(pi*Rational(3, 2) - x) == -sec(x)

    # 检验 sec 和 csc 的关系
    assert sec(pi/2 - x) == csc(x)
    assert sec(pi/2 + x) == -csc(x)
    assert sec(pi*Rational(3, 2) + x) == csc(x)
    assert sec(pi*Rational(3, 2) - x) == -csc(x)


# 定义一个测试函数，用于检验 issue 17142 的数学函数的特殊情况
def test_as_real_imag():
    # 检验特定表达式的实部和虚部
    # 如果在不相关的构建或主分支中重新出现问题，请重新开放该问题。
    expr = atan(I/(I + I*tan(1)))
    assert expr.as_real_imag() == (expr, 0)


# 定义一个测试函数，用于检验 issue 18746 的数学函数的周期性
def test_issue_18746():
    # 检验特定表达式的周期
    e3 = cos(S.Pi*(x/= pi/2


def test_issue_25847():
    # 测试反正切函数
    assert atan(sin(x)/x).as_leading_term(x) == pi/4
    raises(PoleError, lambda: atan(exp(1/x)).as_leading_term(x))

    # 测试反正弦函数
    # 检查 asin 函数对 sin(x)/x 的反正弦是否等于 pi/2 的主导项
    assert asin(sin(x)/x).as_leading_term(x) == pi/2
    # 检查 asin 函数对 exp(1/x) 的反正弦是否会引发 PoleError 异常
    raises(PoleError, lambda: asin(exp(1/x)).as_leading_term(x))

    # 检查 acos 函数对 sin(x)/x 的反余弦是否等于 0 的主导项
    assert acos(sin(x)/x).as_leading_term(x) == 0
    # 检查 acos 函数对 exp(1/x) 的反余弦是否会引发 PoleError 异常
    raises(PoleError, lambda: acos(exp(1/x)).as_leading_term(x))

    # 检查 acot 函数对 sin(x)/x 的反余切是否等于 pi/4 的主导项
    assert acot(sin(x)/x).as_leading_term(x) == pi/4
    # 检查 acot 函数对 exp(1/x) 的反余切是否会引发 PoleError 异常
    raises(PoleError, lambda: acot(exp(1/x)).as_leading_term(x))

    # 检查 asec 函数对 sin(x)/x 的反正割是否等于 0 的主导项
    assert asec(sin(x)/x).as_leading_term(x) == 0
    # 检查 asec 函数对 exp(1/x) 的反正割是否会引发 PoleError 异常
    raises(PoleError, lambda: asec(exp(1/x)).as_leading_term(x))

    # 检查 acsc 函数对 sin(x)/x 的反余割是否等于 pi/2 的主导项
    assert acsc(sin(x)/x).as_leading_term(x) == pi/2
    # 检查 acsc 函数对 exp(1/x) 的反余割是否会引发 PoleError 异常
    raises(PoleError, lambda: acsc(exp(1/x)).as_leading_term(x))
def test_issue_23843():
    # 计算 atan(x + I) 的级数展开，x 趋向正无穷时的结果
    assert atan(x + I).series(x, oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    # 计算 atan(x + I) 的级数展开，x 趋向负无穷时的结果
    assert atan(x + I).series(x, -oo) == -16/(5*x**5) - 2*I/x**4 + 4/(3*x**3) + I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))
    # 计算 atan(x - I) 的级数展开，x 趋向正无穷时的结果
    assert atan(x - I).series(x, oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x + pi/2 + O(x**(-6), (x, oo))
    # 计算 atan(x - I) 的级数展开，x 趋向负无穷时的结果
    assert atan(x - I).series(x, -oo) == -16/(5*x**5) + 2*I/x**4 + 4/(3*x**3) - I/x**2 - 1/x - pi/2 + O(x**(-6), (x, -oo))

    # 计算 acot(x + I) 的级数展开，x 趋向正无穷时的结果
    assert acot(x + I).series(x, oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, oo))
    # 计算 acot(x + I) 的级数展开，x 趋向负无穷时的结果
    assert acot(x + I).series(x, -oo) == 16/(5*x**5) + 2*I/x**4 - 4/(3*x**3) - I/x**2 + 1/x + O(x**(-6), (x, -oo))
    # 计算 acot(x - I) 的级数展开，x 趋向正无穷时的结果
    assert acot(x - I).series(x, oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, oo))
    # 计算 acot(x - I) 的级数展开，x 趋向负无穷时的结果
    assert acot(x - I).series(x, -oo) == 16/(5*x**5) - 2*I/x**4 - 4/(3*x**3) + I/x**2 + 1/x + O(x**(-6), (x, -oo))
```