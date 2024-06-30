# `D:\src\scipysrc\sympy\sympy\calculus\tests\test_util.py`

```
# 导入需要的符号、函数和模块
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import frac
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    cos, cot, csc, sec, sin, tan, asin, acos, atan, acot, asec, acsc)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
                                 periodicity, lcim, is_convex,
                                 stationary_points, minimum, maximum)
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow, slow
from sympy.abc import x, y

# 定义符号变量 a，其为实数
a = Symbol('a', real=True)

# 定义测试函数 test_function_range
def test_function_range():
    # 断言 sin(x) 在区间 [-pi/2, pi/2] 上的函数值范围是 [-1, 1]
    assert function_range(sin(x), x, Interval(-pi/2, pi/2)) == Interval(-1, 1)
    # 断言 sin(x) 在区间 [0, pi] 上的函数值范围是 [0, 1]
    assert function_range(sin(x), x, Interval(0, pi)) == Interval(0, 1)
    # 断言 tan(x) 在区间 [0, pi] 上的函数值范围是 (-oo, oo)
    assert function_range(tan(x), x, Interval(0, pi)) == Interval(-oo, oo)
    # 断言 tan(x) 在区间 [pi/2, pi] 上的函数值范围是 (-oo, 0)
    assert function_range(tan(x), x, Interval(pi/2, pi)) == Interval(-oo, 0)
    # 断言 (x + 3)/(x - 2) 在区间 [-5, 5] 上的函数值范围是 (-oo, 2/7) U (8/3, oo)
    assert function_range((x + 3)/(x - 2), x, Interval(-5, 5)) == Union(Interval(-oo, Rational(2, 7)), Interval(Rational(8, 3), oo))
    # 断言 1/(x**2) 在区间 [-1, 1] 上的函数值范围是 [1, oo)
    assert function_range(1/(x**2), x, Interval(-1, 1)) == Interval(1, oo)
    # 断言 exp(x) 在区间 [-1, 1] 上的函数值范围是 [exp(-1), exp(1)]
    assert function_range(exp(x), x, Interval(-1, 1)) == Interval(exp(-1), exp(1))
    # 断言 log(x) - x 在实数集上的函数值范围是 (-oo, -1)
    assert function_range(log(x) - x, x, S.Reals) == Interval(-oo, -1)
    # 断言 sqrt(3*x - 1) 在区间 [0, 2] 上的函数值范围是 [0, sqrt(5)]
    assert function_range(sqrt(3*x - 1), x, Interval(0, 2)) == Interval(0, sqrt(5))
    # 断言 x*(x - 1) - (x**2 - x) 在实数集上的函数值范围是 {0}
    assert function_range(x*(x - 1) - (x**2 - x), x, S.Reals) == FiniteSet(0)
    # 断言 x*(x - 1) - (x**2 - x) + y 在实数集上的函数值范围是 {y}
    assert function_range(x*(x - 1) - (x**2 - x) + y, x, S.Reals) == FiniteSet(y)
    # 断言 sin(x) 在联合区间 [-5, -3] U {4} 上的函数值范围是 [-sin(3), 1] U {sin(4)}
    assert function_range(sin(x), x, Union(Interval(-5, -3), FiniteSet(4))) == Union(Interval(-sin(3), 1), FiniteSet(sin(4)))
    # 断言 cos(x) 在区间 (-oo, -4] 上的函数值范围是 [-1, 1]
    assert function_range(cos(x), x, Interval(-oo, -4)) == Interval(-1, 1)
    # 断言 cos(x) 在空集上的函数值范围是 空集
    assert function_range(cos(x), x, S.EmptySet) == S.EmptySet
    # 断言 x/sqrt(x**2+1) 在实数集上的函数值范围是 (-1, 1)
    assert function_range(x/sqrt(x**2+1), x, S.Reals) == Interval.open(-1, 1)
    # 断言调用未实现的函数时会引发 NotImplementedError
    raises(NotImplementedError, lambda: function_range(exp(x)*(sin(x) - cos(x))/2 - x, x, S.Reals))
    # 调用 raises 函数来测试 function_range 函数在特定条件下是否抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda : function_range(
        sin(x) + x, x, S.Reals)) # issue 13273
    
    # 调用 raises 函数来测试 function_range 函数在特定条件下是否抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda : function_range(
        log(x), x, S.Integers))
    
    # 调用 raises 函数来测试 function_range 函数在特定条件下是否抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda : function_range(
        sin(x)/2, x, S.Naturals))
@slow
# 定义一个标记为 @slow 的测试函数，用于测试 function_range 函数
def test_function_range1():
    # 断言 function_range 函数对于 tan(x)**2 + tan(3*x)**2 + 1 在 x 属于实数集合中的结果为区间 [1, ∞)
    assert function_range(tan(x)**2 + tan(3*x)**2 + 1, x, S.Reals) == Interval(1,oo)


# 测试 continuous_domain 函数的不同输入情况
def test_continuous_domain():
    # 断言 sin(x) 在区间 [0, 2*pi] 上的连续定义域是区间 [0, 2*pi]
    assert continuous_domain(sin(x), x, Interval(0, 2*pi)) == Interval(0, 2*pi)
    # 断言 tan(x) 在区间 [0, 2*pi] 上的连续定义域是区间的并集
    assert continuous_domain(tan(x), x, Interval(0, 2*pi)) == \
        Union(Interval(0, pi/2, False, True), Interval(pi/2, pi*Rational(3, 2), True, True),
              Interval(pi*Rational(3, 2), 2*pi, True, False))
    # 断言 cot(x) 在区间 [0, 2*pi] 上的连续定义域是区间的并集
    assert continuous_domain(cot(x), x, Interval(0, 2*pi)) == Union(
        Interval.open(0, pi), Interval.open(pi, 2*pi))
    # 断言 (x - 1)/((x - 1)**2) 在实数集合上的连续定义域是区间的并集
    assert continuous_domain((x - 1)/((x - 1)**2), x, S.Reals) == \
        Union(Interval(-oo, 1, True, True), Interval(1, oo, True, True))
    # 断言 log(x) + log(4*x - 1) 在实数集合上的连续定义域是区间 [1/4, ∞)
    assert continuous_domain(log(x) + log(4*x - 1), x, S.Reals) == \
        Interval(Rational(1, 4), oo, True, True)
    # 断言 1/sqrt(x - 3) 在实数集合上的连续定义域是区间 [3, ∞)
    assert continuous_domain(1/sqrt(x - 3), x, S.Reals) == Interval(3, oo, True, True)
    # 断言 1/x - 2 在实数集合上的连续定义域是区间的并集
    assert continuous_domain(1/x - 2, x, S.Reals) == \
        Union(Interval.open(-oo, 0), Interval.open(0, oo))
    # 断言 1/(x**2 - 4) + 2 在实数集合上的连续定义域是区间的并集
    assert continuous_domain(1/(x**2 - 4) + 2, x, S.Reals) == \
        Union(Interval.open(-oo, -2), Interval.open(-2, 2), Interval.open(2, oo))
    # 断言 (x+1)**pi 在实数集合上的连续定义域是区间 [-1, ∞)
    assert continuous_domain((x+1)**pi, x, S.Reals) == Interval(-1, oo)
    # 断言 (x+1)**(pi/2) 在实数集合上的连续定义域是区间 [-1, ∞)
    assert continuous_domain((x+1)**(pi/2), x, S.Reals) == Interval(-1, oo)
    # 断言 x**x 在实数集合上的连续定义域是区间 [0, ∞)
    assert continuous_domain(x**x, x, S.Reals) == Interval(0, oo)
    # 断言 (x+1)**log(x**2) 在实数集合上的连续定义域是区间的并集
    assert continuous_domain((x+1)**log(x**2), x, S.Reals) == Union(
        Interval.Ropen(-1, 0), Interval.open(0, oo))
    # 计算 log(tan(x)**2 + 1) 在实数集合上的连续定义域，并进行相关断言
    domain = continuous_domain(log(tan(x)**2 + 1), x, S.Reals)
    assert not domain.contains(3*pi/2)
    assert domain.contains(5)
    # 断言 x**(1/d) 在实数集合上的连续定义域是区间 [0, ∞)
    d = Symbol('d', even=True, zero=False)
    assert continuous_domain(x**(1/d), x, S.Reals) == Interval(0, oo)
    # 断言 1/sin(x) 在实数集合上的连续定义域是全体实数减去一些点的集合
    n = Dummy('n')
    assert continuous_domain(1/sin(x), x, S.Reals).dummy_eq(Complement(
        S.Reals, Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                       ImageSet(Lambda(n, 2*n*pi), S.Integers))))
    # 断言 sin(x) + cos(x) 在实数集合上的连续定义域是全体实数
    assert continuous_domain(sin(x) + cos(x), x, S.Reals) == S.Reals
    # 断言 asin(x) 在实数集合上的连续定义域是区间 [-1, 1]
    assert continuous_domain(asin(x), x, S.Reals) == Interval(-1, 1) # issue #21786
    # 断言 1/acos(log(x)) 在实数集合上的连续定义域是区间 (exp(-1), e]
    assert continuous_domain(1/acos(log(x)), x, S.Reals) == Interval.Ropen(exp(-1), E)
    # 断言 sinh(x)+cosh(x) 在实数集合上的连续定义域是全体实数
    assert continuous_domain(sinh(x)+cosh(x), x, S.Reals) == S.Reals
    # 断言 tanh(x)+sech(x) 在实数集合上的连续定义域是全体实数
    assert continuous_domain(tanh(x)+sech(x), x, S.Reals) == S.Reals
    # 断言 atan(x)+asinh(x) 在实数集合上的连续定义域是全体实数
    assert continuous_domain(atan(x)+asinh(x), x, S.Reals) == S.Reals
    # 断言 acosh(x) 在实数集合上的连续定义域是区间 [1, ∞)
    assert continuous_domain(acosh(x), x, S.Reals) == Interval(1, oo)
    # 断言 atanh(x) 在实数集合上的连续定义域是区间 (-1, 1)
    assert continuous_domain(atanh(x), x, S.Reals) == Interval.open(-1, 1)
    # 断言 atanh(x)+acosh(x) 在实数集合上的连续定义域是空集
    assert continuous_domain(atanh(x)+acosh(x), x, S.Reals) == S.EmptySet
    # 断言 asech(x) 在实数集合上的连续定义域是区间 (0, 1]
    assert continuous_domain(asech(x), x, S.Reals) == Interval.Lopen(0, 1)
    # 断言 acoth(x) 在实数集合上的连续定义域是区间的并集
    assert continuous_domain(acoth(x), x, S.Reals) == Union(
        Interval.open(-oo, -1), Interval.open(1, oo))
    # 断言 asec(x) 在实数集合上的连续定义域是区间的并集
    assert continuous_domain(asec(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    # 断言：对 acsc(x) 的连续定义域为实数集的负无穷到-1、1到正无穷的并集
    assert continuous_domain(acsc(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    
    # 遍历函数集合 (coth, acsch, csch)，断言每个函数在实数集上的连续定义域为负无穷到0、0到正无穷的开区间的并集
    for f in (coth, acsch, csch):
        assert continuous_domain(f(x), x, S.Reals) == Union(
            Interval.open(-oo, 0), Interval.open(0, oo))
    
    # 断言 acot(x) 在实数集上的连续定义域不包含0
    assert continuous_domain(acot(x), x, S.Reals).contains(0) == False
    
    # 断言 1/(exp(x) - x) 在实数集上的连续定义域为实数集减去满足条件 Eq(-x + exp(x), 0) 的集合的补集
    assert continuous_domain(1/(exp(x) - x), x, S.Reals) == Complement(
        S.Reals, ConditionSet(x, Eq(-x + exp(x), 0), S.Reals))
    
    # 断言 frac(x**2) 在区间 [-2,-1] 上的连续定义域为三个开区间的并集
    assert continuous_domain(frac(x**2), x, Interval(-2,-1)) == Union(
        Interval.open(-2, -sqrt(3)), Interval.open(-sqrt(2), -1),
        Interval.open(-sqrt(3), -sqrt(2)))
    
    # 断言 frac(x) 在实数集上的连续定义域为实数集减去整数集的补集
    assert continuous_domain(frac(x), x, S.Reals) == Complement(
        S.Reals, S.Integers)
    
    # 使用 lambda 函数抛出 NotImplementedError，断言 1/(x**2+1) 在复数集上的连续定义域未实现
    raises(NotImplementedError, lambda: continuous_domain(
        1/(x**2+1), x, S.Complexes))
    
    # 使用 lambda 函数抛出 NotImplementedError，断言 gamma(x) 在区间 [-5,0] 上的连续定义域未实现
    raises(NotImplementedError, lambda: continuous_domain(
        gamma(x), x, Interval(-5,0)))
    
    # 断言 x + gamma(pi) 在实数集上的连续定义域为实数集
    assert continuous_domain(x + gamma(pi), x, S.Reals) == S.Reals
@XFAIL
# 标记为预期失败的测试函数，不期望当前测试通过

def test_continuous_domain_acot():
    # 创建一个分段函数acot_cont，当x<0时返回pi+acot(x)，否则返回acot(x)
    acot_cont = Piecewise((pi+acot(x), x<0), (acot(x), True))
    # 断言对于acot_cont函数在实数域上的连续性为实数集合S.Reals
    assert continuous_domain(acot_cont, x, S.Reals) == S.Reals

@XFAIL
# 标记为预期失败的测试函数，不期望当前测试通过

def test_continuous_domain_gamma():
    # 断言gamma(x)函数在实数域上不包含-1
    assert continuous_domain(gamma(x), x, S.Reals).contains(-1) == False

@XFAIL
# 标记为预期失败的测试函数，不期望当前测试通过

def test_continuous_domain_neg_power():
    # 断言(x-2)**(1-x)函数在实数域上的连续性为开区间(2, oo)
    assert continuous_domain((x-2)**(1-x), x, S.Reals) == Interval.open(2, oo)


def test_not_empty_in():
    # 断言FiniteSet(x, 2*x)与区间(1, 2)的交集在变量x上的非空部分为区间(S.Half, 2)
    assert not_empty_in(FiniteSet(x, 2*x).intersect(Interval(1, 2, True, False)), x) == \
        Interval(S.Half, 2, True, False)
    # 断言FiniteSet(x, x**2)与区间(1, 2)的交集在变量x上的非空部分为两个区间的并集
    assert not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x) == \
        Union(Interval(-sqrt(2), -1), Interval(1, 2))
    # 断言FiniteSet(x**2 + x, x)与区间(2, 4)的交集在变量x上的非空部分为三个区间的并集
    assert not_empty_in(FiniteSet(x**2 + x, x).intersect(Interval(2, 4)), x) == \
        Union(Interval(-sqrt(17)/2 - S.Half, -2),
              Interval(1, Rational(-1, 2) + sqrt(17)/2), Interval(2, 4))
    # 断言FiniteSet(x/(x - 1))与实数集合S.Reals的交集在变量x上的非空部分为实数集合S.Reals减去有限集{1}
    assert not_empty_in(FiniteSet(x/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    # 断言FiniteSet(a/(a - 1))与实数集合S.Reals的交集在变量a上的非空部分为实数集合S.Reals减去有限集{1}
    assert not_empty_in(FiniteSet(a/(a - 1)).intersect(S.Reals), a) == \
        Complement(S.Reals, FiniteSet(1))
    # 断言FiniteSet((x**2 - 3*x + 2)/(x - 1))与实数集合S.Reals的交集在变量x上的非空部分为实数集合S.Reals减去有限集{1}
    assert not_empty_in(FiniteSet((x**2 - 3*x + 2)/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    # 断言FiniteSet(3, 4, x/(x - 1))与区间(2, 3)的交集在变量x上的非空部分为实数轴上的全集
    assert not_empty_in(FiniteSet(3, 4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(-oo, oo)
    # 断言FiniteSet(4, x/(x - 1))与区间(2, 3)的交集在变量x上的非空部分为区间(S(3)/2, 2)
    assert not_empty_in(FiniteSet(4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(S(3)/2, 2)
    # 断言FiniteSet(x/(x**2 - 1))与实数集合S.Reals的交集在变量x上的非空部分为实数集合S.Reals减去有限集{-1, 1}
    assert not_empty_in(FiniteSet(x/(x**2 - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(-1, 1))
    # 断言FiniteSet(x, x**2)与区间并集(Interval(1, 3, True, True), Interval(4, 5))的交集在变量x上的非空部分为四个区间的并集
    assert not_empty_in(FiniteSet(x, x**2).intersect(Union(Interval(1, 3, True, True),
                                                           Interval(4, 5))), x) == \
        Union(Interval(-sqrt(5), -2), Interval(-sqrt(3), -1, True, True),
              Interval(1, 3, True, True), Interval(4, 5))
    # 断言FiniteSet(1)与区间(3, 4)的交集在变量x上的非空部分为空集
    assert not_empty_in(FiniteSet(1).intersect(Interval(3, 4)), x) == S.EmptySet
    # 断言FiniteSet(x**2/(x + 2))与区间(1, oo)的交集在变量x上的非空部分为两个区间的并集
    assert not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x) == \
        Union(Interval(-2, -1, True, False), Interval(2, oo))
    # 断言调用函数not_empty_in(x)时会抛出ValueError异常
    raises(ValueError, lambda: not_empty_in(x))
    # 断言调用函数not_empty_in(Interval(0, 1), x)时会抛出ValueError异常
    raises(ValueError, lambda: not_empty_in(Interval(0, 1), x))
    # 断言调用函数not_empty_in(FiniteSet(x).intersect(S.Reals), x, a)时会抛出NotImplementedError异常
    raises(NotImplementedError,
           lambda: not_empty_in(FiniteSet(x).intersect(S.Reals), x, a))


@_both_exp_pow
# 带有装饰器_both_exp_pow的测试函数，对指数和幂函数都进行测试

def test_periodicity():
    # 断言sin(2*x)函数在变量x上的周期为pi
    assert periodicity(sin(2*x), x) == pi
    # 断言(-2)*tan(4*x)函数在变量x上的周期为pi/4
    assert periodicity((-2)*tan(4*x), x) == pi/4
    # 断言sin(x)**2函数在变量x上的周期为2*pi
    assert periodicity(sin(x)**2, x) == 2*pi
    # 断言3**tan(3*x)函数在变量x上的周期为pi/3
    assert periodicity(3**tan(3*x), x) == pi/3
    # 断言tan(x)*cos(x)函数在变量x上的周期为2*pi
    assert periodicity(tan(x)*cos(x), x) == 2*pi
    # 断言sin(x)**(tan(x))函数在变量x上的周期为2*pi
    assert periodicity(sin(x)**(tan(x)), x) == 2*pi
    # 断言tan(x)*sec(x)函数在变量x上的周期为2*pi
    assert periodicity(tan(x)*sec(x), x) == 2*pi
    # 断言sin(2*x)*cos(2*x) - y函数在变量x上的周期为pi/2
    assert periodicity(sin(2*x)*cos(2*x) - y, x) == pi/2
    # 断言tan(x) + cot(x)函数在变量x上的周期为pi
    assert periodicity(tan(x) + cot(x), x) == pi
    # 断言sin(x) - cos
    # 断言 sin(x) 的指数函数 exp(sin(x)) 在变量 x 上的周期性为 2*pi
    assert periodicity(exp(sin(x)), x) == 2*pi
    # 断言 log(cot(2*x)) - sin(cos(2*x)) 在变量 x 上的周期性为 pi
    assert periodicity(log(cot(2*x)) - sin(cos(2*x)), x) == pi
    # 断言 sin(2*x)*exp(tan(x) - csc(2*x)) 在变量 x 上的周期性为 pi
    assert periodicity(sin(2*x)*exp(tan(x) - csc(2*x)), x) == pi
    # 断言 cos(sec(x) - csc(2*x)) 在变量 x 上的周期性为 2*pi
    assert periodicity(cos(sec(x) - csc(2*x)), x) == 2*pi
    # 断言 tan(sin(2*x)) 在变量 x 上的周期性为 pi
    assert periodicity(tan(sin(2*x)), x) == pi
    # 断言 2*tan(x)**2 在变量 x 上的周期性为 pi
    assert periodicity(2*tan(x)**2, x) == pi
    # 断言 sin(x%4) 在变量 x 上的周期性为 4
    assert periodicity(sin(x%4), x) == 4
    # 断言 sin(x)%4 在变量 x 上的周期性为 2*pi
    assert periodicity(sin(x)%4, x) == 2*pi
    # 断言 tan((3*x-2)%4) 在变量 x 上的周期性为 4/3
    assert periodicity((tan((3*x-2)%4)), x) == Rational(4, 3)
    # 断言 (sqrt(2)*(x+1)+x) % 3 在变量 x 上的周期性为 3 / (sqrt(2)+1)
    assert periodicity((sqrt(2)*(x+1)+x) % 3, x) == 3 / (sqrt(2)+1)
    # 断言 (x**2+1) % x 在变量 x 上的周期性为 None
    assert periodicity((x**2+1) % x, x) is None
    # 断言 sin(re(x)) 在变量 x 上的周期性为 2*pi
    assert periodicity(sin(re(x)), x) == 2*pi
    # 断言 sin(x)**2 + cos(x)**2 在变量 x 上的周期性为 0
    assert periodicity(sin(x)**2 + cos(x)**2, x) is S.Zero
    # 断言 tan(x) 在变量 y 上的周期性为 0
    assert periodicity(tan(x), y) is S.Zero
    # 断言 sin(x) + I*cos(x) 在变量 x 上的周期性为 2*pi
    assert periodicity(sin(x) + I*cos(x), x) == 2*pi
    # 断言 x - sin(2*y) 在变量 y 上的周期性为 pi
    assert periodicity(x - sin(2*y), y) == pi

    # 断言 exp(x) 在变量 x 上的周期性为 None
    assert periodicity(exp(x), x) is None
    # 断言 exp(I*x) 在变量 x 上的周期性为 2*pi
    assert periodicity(exp(I*x), x) == 2*pi
    # 断言 exp(I*a) 在变量 a 上的周期性为 2*pi
    assert periodicity(exp(I*a), a) == 2*pi
    # 断言 exp(a) 在变量 a 上的周期性为 None
    assert periodicity(exp(a), a) is None
    # 断言 exp(log(sin(a) + I*cos(2*a)), evaluate=False) 在变量 a 上的周期性为 2*pi
    assert periodicity(exp(log(sin(a) + I*cos(2*a)), evaluate=False), a) == 2*pi
    # 断言 exp(log(sin(2*a) + I*cos(a)), evaluate=False) 在变量 a 上的周期性为 2*pi
    assert periodicity(exp(log(sin(2*a) + I*cos(a)), evaluate=False), a) == 2*pi
    # 断言 exp(sin(a)) 在变量 a 上的周期性为 2*pi
    assert periodicity(exp(sin(a)), a) == 2*pi
    # 断言 exp(2*I*a) 在变量 a 上的周期性为 pi
    assert periodicity(exp(2*I*a), a) == pi
    # 断言 exp(a + I*sin(a)) 在变量 a 上的周期性为 None
    assert periodicity(exp(a + I*sin(a)), a) is None
    # 断言 exp(cos(a/2) + sin(a)) 在变量 a 上的周期性为 4*pi
    assert periodicity(exp(cos(a/2) + sin(a)), a) == 4*pi
    # 断言 log(x) 在变量 x 上的周期性为 None
    assert periodicity(log(x), x) is None
    # 断言 exp(x)**sin(x) 在变量 x 上的周期性为 None
    assert periodicity(exp(x)**sin(x), x) is None
    # 断言 sin(x)**y 在变量 y 上的周期性为 None
    assert periodicity(sin(x)**y, y) is None

    # 断言 Abs(sin(Abs(sin(x)))) 在变量 x 上的周期性为 pi
    assert periodicity(Abs(sin(Abs(sin(x)))), x) == pi
    # 断言 对于 cos, sin, sec, csc, tan, cot 函数，Abs(f(x)) 在变量 x 上的周期性均为 pi
    assert all(periodicity(Abs(f(x)), x) == pi for f in (cos, sin, sec, csc, tan, cot))
    # 断言 Abs(sin(tan(x))) 在变量 x 上的周期性为 pi
    assert periodicity(Abs(sin(tan(x))), x) == pi
    # 断言 Abs(sin(sin(x) + tan(x))) 在变量 x 上的周期性为 2*pi
    assert periodicity(Abs(sin(sin(x) + tan(x))), x) == 2*pi
    # 断言 sin(x) > S.Half 在变量 x 上的周期性为 2*pi
    assert periodicity(sin(x) > S.Half, x) == 2*pi

    # 断言 x > 2 在变量 x 上的周期性为 None
    assert periodicity(x > 2, x) is None
    # 断言 x**3 - x**2 + 1 在变量 x 上的周期性为 None
    assert periodicity(x**3 - x**2 + 1, x) is None
    # 断言 Abs(x) 在变量 x 上的周期性为 None
    assert periodicity(Abs(x), x) is None
    # 断言 Abs(x**2 - 1) 在变量 x 上的周期性为 None
    assert periodicity(Abs(x**2 - 1), x) is None

    # 断言 (x**2 + 4)%2 在变量 x 上的周期性为 None
    assert periodicity((x**2 + 4)%2, x) is None
    # 断言 (E**x)%3 在变量 x 上的周期性为 None
    assert periodicity((E**x)%3, x) is None

    # 断言 sin(expint(1, x))/expint(1, x) 在变量 x 上的周期性为 None
    assert periodicity(sin(expint(1, x))/expint(1, x), x) is None
    # 断言 对于任何 Piecewise 结果返回 None
    p = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    assert periodicity(p, x) is None

    # 断言对于矩阵符号 m，sin(m) 和 sin(m[0, 0]) 的周期性未实现
    m = MatrixSymbol('m', 3, 3)
    raises(NotImplementedError, lambda: periodicity(sin(m), m))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m))
    raises(NotImplementedError, lambda: periodicity(sin(m), m[0, 0]))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m[0, 0]))
# 定义测试函数 test_periodicity_check，用于测试 periodicity 函数的周期性检查功能
def test_periodicity_check():
    # 断言 tan(x) 的周期性，应返回 pi
    assert periodicity(tan(x), x, check=True) == pi
    # 断言 sin(x) + cos(x) 的周期性，应返回 2*pi
    assert periodicity(sin(x) + cos(x), x, check=True) == 2*pi
    # 断言 sec(x) 的周期性，应返回 2*pi
    assert periodicity(sec(x), x) == 2*pi
    # 断言 sin(x*y) 的周期性，应返回 2*pi/|y|
    assert periodicity(sin(x*y), x) == 2*pi/abs(y)
    # 断言 Abs(sec(sec(x))) 的周期性，应返回 pi
    assert periodicity(Abs(sec(sec(x))), x) == pi


# 定义测试函数 test_lcim，用于测试 lcim 函数的最小公倍数计算功能
def test_lcim():
    # 断言计算 [S.Half, S(2), S(3)] 的最小公倍数，应返回 6
    assert lcim([S.Half, S(2), S(3)]) == 6
    # 断言计算 [pi/2, pi/4, pi] 的最小公倍数，应返回 pi
    assert lcim([pi/2, pi/4, pi]) == pi
    # 断言计算 [2*pi, pi/2] 的最小公倍数，应返回 2*pi
    assert lcim([2*pi, pi/2]) == 2*pi
    # 断言计算 [S.One, 2*pi] 的最小公倍数，应返回 None
    assert lcim([S.One, 2*pi]) is None
    # 断言计算 [S(2) + 2*E, E/3 + Rational(1, 3), S.One + E] 的最小公倍数，应返回 S(2) + 2*E
    assert lcim([S(2) + 2*E, E/3 + Rational(1, 3), S.One + E]) == S(2) + 2*E


# 定义测试函数 test_is_convex，用于测试 is_convex 函数的凸性检查功能
def test_is_convex():
    # 断言 1/x 在区间 (0, oo) 上是凸的，应返回 True
    assert is_convex(1/x, x, domain=Interval.open(0, oo)) == True
    # 断言 1/x 在区间 (-oo, 0) 上不是凸的，应返回 False
    assert is_convex(1/x, x, domain=Interval(-oo, 0)) == False
    # 断言 x**2 在区间 (0, oo) 上是凸的，应返回 True
    assert is_convex(x**2, x, domain=Interval(0, oo)) == True
    # 断言 1/x**3 在区间 (0, oo) 上是凸的，应返回 True
    assert is_convex(1/x**3, x, domain=Interval.Lopen(0, oo)) == True
    # 断言 -1/x**3 在区间 (-oo, 0) 上是凸的，应返回 True
    assert is_convex(-1/x**3, x, domain=Interval.Ropen(-oo, 0)) == True
    # 断言 log(x) 不是凸函数，应返回 False
    assert is_convex(log(x), x) == False
    # 断言 x**2 + y**2 在变量 x 上是凸的，应返回 True
    assert is_convex(x**2 + y**2, x, y) == True
    # 断言 cos(x) + cos(y) 在变量 x 上不是凸的，应返回 False
    assert is_convex(cos(x) + cos(y), x) == False
    # 断言 8*x**2 - 2*y**2 在变量 x, y 上不是凸的，应返回 False
    assert is_convex(8*x**2 - 2*y**2, x, y) == False


# 定义测试函数 test_stationary_points，用于测试 stationary_points 函数的稳定点计算功能
def test_stationary_points():
    # 断言 sin(x) 在区间 (-pi/2, pi/2) 上的稳定点，应返回 {-pi/2, pi/2}
    assert stationary_points(sin(x), x, Interval(-pi/2, pi/2)) == {-pi/2, pi/2}
    # 断言 sin(x) 在区间 (0, pi/4) 上的稳定点，应返回 EmptySet
    assert stationary_points(sin(x), x, Interval.Ropen(0, pi/4)) is S.EmptySet
    # 断言 tan(x) 的稳定点，应返回 EmptySet
    assert stationary_points(tan(x), x) is S.EmptySet
    # 断言 sin(x)*cos(x) 在区间 (0, pi) 上的稳定点，应返回 {pi/4, 3*pi/4}
    assert stationary_points(sin(x)*cos(x), x, Interval(0, pi)) == {pi/4, 3*pi/4}
    # 断言 sec(x) 在区间 (0, pi) 上的稳定点，应返回 {0, pi}
    assert stationary_points(sec(x), x, Interval(0, pi)) == {0, pi}
    # 断言 (x+3)*(x-2) 的稳定点，应返回 {-1/2}
    assert stationary_points((x+3)*(x-2), x) == FiniteSet(Rational(-1, 2))
    # 断言 (x + 3)/(x - 2) 在区间 (-5, 5) 上的稳定点，应返回 EmptySet
    assert stationary_points((x + 3)/(x - 2), x, Interval(-5, 5)) is S.EmptySet
    # 断言 (x**2+3)/(x-2) 的稳定点，应返回 {2 - sqrt(7), 2 + sqrt(7)}
    assert stationary_points((x**2+3)/(x-2), x) == {2 - sqrt(7), 2 + sqrt(7)}
    # 断言 (x**2+3)/(x-2) 在区间 (0, 5) 上的稳定点，应返回 {2 + sqrt(7)}
    assert stationary_points((x**2+3)/(x-2), x, Interval(0, 5)) == {2 + sqrt(7)}
    # 断言 x**4 + x**3 - 5*x**2 在实数集上的稳定点，应返回 {-2, 0, 5/4}
    assert stationary_points(x**4 + x**3 - 5*x**2, x, S.Reals) == FiniteSet(-2, 0, Rational(5, 4))
    # 断言 exp(x) 的稳定点，应返回 EmptySet
    assert stationary_points(exp(x), x) is S.EmptySet
    # 断言 log(x) - x 在实数集上的稳定点，应返回 {1}
    assert stationary_points(log(x) - x, x, S.Reals) == {1}
    # 断言 cos(x) 在区间 (0, 5) 和区间 (-6, -3) 的并集上的稳定点，应返回 {0, -pi, pi}
    assert stationary_points(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))) == {0, -pi, pi}
    # 断言 y 在变量 x 上的稳定点，应返回 实数集
    assert stationary_points(y, x, S.Reals) == S.Reals
    # 断言 y 在空集上的稳定点，应返回 EmptySet
    assert stationary_points(y, x, S.EmptySet) == S.EmptySet


# 定义测试函数 test_maximum，用于测试 maximum 函数的最大值计算功能
def test_maximum():
    # 断言 sin(x) 的最大值，应返回 1
    assert maximum(sin(x), x) is S.One
    # 断言 sin(x) 在区间 (0, 1) 上的最大值，应返回 sin(1)
    assert maximum(sin(x), x, Interval(0, 1)) == sin(1)
    # 断言 tan(x) 的最大值，应返回 oo (无穷大)
    assert maximum(tan(x), x) is oo
    # 断言 tan(x) 在区间 (-pi/4, pi/4) 上的最
    # 断言：对于表达式 -x**4 - x**3 + x**2 + 10，找到其在 x 上的最大值，应该等于 41*sqrt(41)/512 + Rational(5419, 512)
    assert simplify(maximum(-x**4 - x**3 + x**2 + 10, x)) == 41*sqrt(41)/512 + Rational(5419, 512)
    
    # 断言：对于指数函数 exp(x)，在区间 (-oo, 2) 上找到其最大值，应该等于 exp(2)
    assert maximum(exp(x), x, Interval(-oo, 2)) == exp(2)
    
    # 断言：对于对数函数 log(x) - x，在实数范围内 (S.Reals) 找到其最大值，应该是 -1
    assert maximum(log(x) - x, x, S.Reals) is S.NegativeOne
    
    # 断言：对于余弦函数 cos(x)，在区间 [0, 5] 和 [-6, -3] 的并集内找到其最大值，应该是 1
    assert maximum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))) is S.One
    
    # 断言：对于 cos(x) - sin(x)，在所有实数范围内 (S.Reals) 找到其最大值，应该是 sqrt(2)
    assert maximum(cos(x) - sin(x), x, S.Reals) == sqrt(2)
    
    # 断言：对于表达式 y，找到其在 x 上的最大值，应该等于 y，这里假设 y 是一个参数
    assert maximum(y, x, S.Reals) == y
    
    # 断言：对于表达式 abs(a**3 + a)，在区间 [0, 2] 内找到其最大值，应该等于 10
    assert maximum(abs(a**3 + a), a, Interval(0, 2)) == 10
    
    # 断言：对于表达式 abs(60*a**3 + 24*a)，在区间 [0, 2] 内找到其最大值，应该等于 528
    assert maximum(abs(60*a**3 + 24*a), a, Interval(0, 2)) == 528
    
    # 断言：对于表达式 abs(12*a*(5*a**2 + 2))，在区间 [0, 2] 内找到其最大值，应该等于 528
    assert maximum(abs(12*a*(5*a**2 + 2)), a, Interval(0, 2)) == 528
    
    # 断言：对于表达式 x/sqrt(x**2+1)，在所有实数范围内 (S.Reals) 找到其最大值，应该等于 1
    assert maximum(x/sqrt(x**2+1), x, S.Reals) == 1
    
    # 断言：使用 lambda 函数，预期在空集合 S.EmptySet 上调用 maximum(sin(x), x) 会引发 ValueError 异常
    raises(ValueError, lambda: maximum(sin(x), x, S.EmptySet))
    
    # 断言：使用 lambda 函数，预期在空集合 S.EmptySet 上调用 maximum(log(cos(x)), x) 会引发 ValueError 异常
    raises(ValueError, lambda: maximum(log(cos(x)), x, S.EmptySet))
    
    # 断言：使用 lambda 函数，预期在空集合 S.EmptySet 上调用 maximum(1/(x**2 + y**2 + 1), x) 会引发 ValueError 异常
    raises(ValueError, lambda: maximum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    
    # 断言：使用 lambda 函数，预期在 sin(x) 函数中同时传递 sin(x) 作为 x 和 y 参数会引发 ValueError 异常
    raises(ValueError, lambda: maximum(sin(x), sin(x)))
    
    # 断言：使用 lambda 函数，预期在空集合 S.EmptySet 上调用 maximum(sin(x), x*y) 会引发 ValueError 异常
    raises(ValueError, lambda: maximum(sin(x), x*y, S.EmptySet))
    
    # 断言：使用 lambda 函数，预期在单一参数 S.One 上调用 maximum(sin(x)) 会引发 ValueError 异常
    raises(ValueError, lambda: maximum(sin(x), S.One))
# 定义测试函数，用于测试 minimum 函数的各种情况
def test_minimum():
    # 断言对于 sin(x)，在整个实数范围内的最小值为 -1
    assert minimum(sin(x), x) is S.NegativeOne
    # 断言对于 sin(x)，在区间 [1, 4] 内的最小值为 sin(4)
    assert minimum(sin(x), x, Interval(1, 4)) == sin(4)
    # 断言对于 tan(x)，在整个实数范围内的最小值为负无穷
    assert minimum(tan(x), x) is -oo
    # 断言对于 tan(x)，在区间 [-pi/4, pi/4] 内的最小值为 -1
    assert minimum(tan(x), x, Interval(-pi/4, pi/4)) is S.NegativeOne
    # 断言对于 sin(x)*cos(x)，在整个实数范围内的最小值为 -1/2
    assert minimum(sin(x)*cos(x), x, S.Reals) == Rational(-1, 2)
    # 断言对于 sin(x)*cos(x)，在区间 [3*pi/8, 5*pi/8] 内的最小值化简为 -sqrt(2)/4
    assert simplify(minimum(sin(x)*cos(x), x, Interval(pi*Rational(3, 8), pi*Rational(5, 8)))
        ) == -sqrt(2)/4
    # 断言对于 (x+3)*(x-2)，在整个实数范围内的最小值为 -25/4
    assert minimum((x+3)*(x-2), x) == Rational(-25, 4)
    # 断言对于 (x+3)/(x-2)，在区间 [-5, 0] 内的最小值为 -3/2
    assert minimum((x+3)/(x-2), x, Interval(-5, 0)) == Rational(-3, 2)
    # 断言对于 x**4-x**3+x**2+10，在整个实数范围内的最小值为 10
    assert minimum(x**4-x**3+x**2+10, x) == S(10)
    # 断言对于 exp(x)，在区间 [-2, oo) 内的最小值为 exp(-2)
    assert minimum(exp(x), x, Interval(-2, oo)) == exp(-2)
    # 断言对于 log(x) - x，在整个实数范围内的最小值为负无穷
    assert minimum(log(x) - x, x, S.Reals) is -oo
    # 断言对于 cos(x)，在区间 [0, 5] 和 [-6, -3] 的并集内的最小值为 -1
    assert minimum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) is S.NegativeOne
    # 断言对于 cos(x)-sin(x)，在整个实数范围内的最小值为 -sqrt(2)
    assert minimum(cos(x)-sin(x), x, S.Reals) == -sqrt(2)
    # 断言对于 y，关于 x 在整个实数范围内的最小值为 y
    assert minimum(y, x, S.Reals) == y
    # 断言对于 x/sqrt(x**2+1)，在整个实数范围内的最小值为 -1

    raises(ValueError, lambda : minimum(sin(x), x, S.EmptySet))
    # 断言当区间为 EmptySet 时会引发 ValueError
    raises(ValueError, lambda : minimum(log(cos(x)), x, S.EmptySet))
    # 断言当区间为 EmptySet 时会引发 ValueError
    raises(ValueError, lambda : minimum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    # 断言当区间为 EmptySet 时会引发 ValueError
    raises(ValueError, lambda : minimum(sin(x), sin(x)))
    # 断言当自变量不是变量时会引发 ValueError
    raises(ValueError, lambda : minimum(sin(x), x*y, S.EmptySet))
    # 断言当区间为 EmptySet 时会引发 ValueError
    raises(ValueError, lambda : minimum(sin(x), S.One))
    # 断言当域为 S.One 时会引发 ValueError


def test_issue_19869():
    # 断言关于 sqrt(3)*(x - 1)/(3*sqrt(x**2 + 1)) 的最大值为 sqrt(3)/3
    assert (maximum(sqrt(3)*(x - 1)/(3*sqrt(x**2 + 1)), x)
        ) == sqrt(3)/3


def test_issue_16469():
    # 定义 f = abs(a)，断言函数 f 在实数范围内的取值范围为 [0, oo)
    f = abs(a)
    assert function_range(f, a, S.Reals) == Interval(0, oo, False, True)


@_both_exp_pow
def test_issue_18747():
    # 断言 exp(pi*I*(x/4 + S.Half/2)) 的周期性为 8
    assert periodicity(exp(pi*I*(x/4 + S.Half/2)), x) == 8


def test_issue_25942():
    # 断言 acos(x) > pi/3 的解集为开区间 (-1, 1/2]
    assert (acos(x) > pi/3).as_set() == Interval.Ropen(-1, S(1)/2)
```