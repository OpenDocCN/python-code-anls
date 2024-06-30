# `D:\src\scipysrc\sympy\sympy\core\tests\test_power.py`

```
# 导入 SymPy 库中的各种模块和函数

from sympy.core import (
    Basic, Rational, Symbol, S, Float, Integer, Mul, Number, Pow,
    Expr, I, nan, pi, symbols, oo, zoo, N)
# 从 sympy.core.parameters 模块导入全局参数
from sympy.core.parameters import global_parameters
# 从 sympy.core.tests.test_evalf 模块导入 NS 函数
from sympy.core.tests.test_evalf import NS
# 从 sympy.core.function 模块导入 expand_multinomial 函数
from sympy.core.function import expand_multinomial
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt, cbrt 函数
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
# 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.special.error_functions 模块导入 erf 函数
from sympy.functions.special.error_functions import erf
# 从 sympy.functions.elementary.trigonometric 模块导入 sin, cos, tan, sec, csc, atan 函数
from sympy.functions.elementary.trigonometric import (
    sin, cos, tan, sec, csc, atan)
# 从 sympy.functions.elementary.hyperbolic 模块导入 cosh, sinh, tanh 函数
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
# 从 sympy.polys 模块导入 Poly 类
from sympy.polys import Poly
# 从 sympy.series.order 模块导入 O 类
from sympy.series.order import O
# 从 sympy.sets 模块导入 FiniteSet 类
from sympy.sets import FiniteSet
# 从 sympy.core.power 模块导入 power 函数
from sympy.core.power import power
# 从 sympy.core.intfunc 模块导入 integer_nthroot 函数
from sympy.core.intfunc import integer_nthroot
# 从 sympy.testing.pytest 模块导入 warns, _both_exp_pow 函数
from sympy.testing.pytest import warns, _both_exp_pow
# 从 sympy.utilities.exceptions 模块导入 SymPyDeprecationWarning 异常
from sympy.utilities.exceptions import SymPyDeprecationWarning
# 从 sympy.abc 模块导入 a, b, c, x, y 符号变量
from sympy.abc import a, b, c, x, y

# 定义测试函数 test_rational
def test_rational():
    # 定义有理数 a = 1/5
    a = Rational(1, 5)

    # 计算 sqrt(5)/5 并赋值给 r
    r = sqrt(5)/5
    # 断言 sqrt(a) 等于 r
    assert sqrt(a) == r
    # 断言 2*sqrt(a) 等于 2*r
    assert 2*sqrt(a) == 2*r

    # 计算 a*a**(1/2) 并赋值给 r
    r = a*a**S.Half
    # 断言 a**(3/2) 等于 r
    assert a**Rational(3, 2) == r
    # 断言 2*a**(3/2) 等于 2*r
    assert 2*a**Rational(3, 2) == 2*r

    # 计算 a**5 * a**(2/3) 并赋值给 r
    r = a**5*a**Rational(2, 3)
    # 断言 a**(17/3) 等于 r
    assert a**Rational(17, 3) == r
    # 断言 2*a**(17/3) 等于 2*r
    assert 2 * a**Rational(17, 3) == 2*r


# 定义测试函数 test_large_rational
def test_large_rational():
    # 计算复杂的有理数表达式 e 并赋值给 e
    e = (Rational(123712**12 - 1, 7) + Rational(1, 7))**Rational(1, 3)
    # 断言 e 等于 234232585392159195136 * (1/7)**(1/3)
    assert e == 234232585392159195136 * (Rational(1, 7)**Rational(1, 3))


# 定义测试函数 test_negative_real
def test_negative_real():
    # 定义 feq 函数，用于判断两个数值是否在一定精度内相等
    def feq(a, b):
        return abs(a - b) < 1E-10

    # 断言 1 / (-0.5) 等于 -2
    assert feq(S.One / Float(-0.5), -Integer(2))


# 定义测试函数 test_expand
def test_expand():
    # 断言 (2**(-1 - x)).expand() 等于 1/2 * 2**(-x)
    assert (2**(-1 - x)).expand() == S.Half*2**(-x)


# 定义测试函数 test_issue_3449
def test_issue_3449():
    # 断言 ((x**(1/3))**2) 等于 x**(2/3)
    assert ((x**Rational(1, 3))**Rational(2)) == x**Rational(2, 3)
    # 断言 ((x**3)**(2/5)) 等于 (x**3)**(2/5)
    assert ((x**Rational(3))**Rational(2, 5)) == (x**Rational(3))**Rational(2, 5)

    # 定义符号变量 a, b，并指定其为实数
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    # 断言 (a**2)**b 等于 (abs(a)**b)**2
    assert (a**2)**b == (abs(a)**b)**2
    # 断言 sqrt(1/a) 不等于 1/sqrt(a)，例如当 a = -1 时
    assert sqrt(1/a) != 1/sqrt(a)
    # 断言 (a**3)**(1/3) 不等于 a
    assert (a**3)**Rational(1, 3) != a
    # 断言 (x**a)**b 不等于 x**(a*b)，例如当 x = -1, a = 2, b = 1/2 时
    assert (x**a)**b != x**(a*b)
    # 断言 (x**0.5)**b 等于 x**(0.5*b)
    assert (x**.5)**b == x**(.5*b)
    # 断言 (x**0.5)**0.5 等于 x**0.25
    assert (x**.5)**.5 == x**.25
    # 断言 (x**2.5)**0.5 不等于 x**1.25，例如当 x = 5*I 时
    assert (x**2.5)**.5 != x**1.25

    # 定义符号变量 k, m，并指定其为整数
    k = Symbol('k', integer=True)
    m = Symbol('m', integer=True)
    # 断言 (x**k)**m 等于 x**(k*m)
    assert (x**k)**m == x**(k*m)
    # 断言 Number(5)**(2/3) 等于 Number(25)**(1/3)
    assert Number(5)**Rational(2, 3) == Number(25)**Rational(1, 3)

    # 断言 (x**0.5)**2 等于 x**1.0
    assert (x**.5)**2 == x**1.0
    # 断言 (x**2)**k 等于 (x**k)**2 等于 x**(2*k)
    assert (x**2)**k == (x**k)**2 == x**(2*k)

    # 定义符号变量 a，并指定其为正数
    a = Symbol('a', positive=True)
    # 断言 (a**3)**(2/5) 等于 a**(6/5)
    assert (a**3)**Rational(2, 5) == a**Rational(6, 5)
    # 断言 (a**2)**b 等于 (a**b
    # 创建一个名为 nonneg 的符号，其值为非负数
    nonneg = Symbol('nonneg', nonnegative=True)
    # 创建一个名为 any 的符号，其值不做限制
    any = Symbol('any')
    # 计算 sqrt(1/neg) 的分子和分母
    num, den = sqrt(1/neg).as_numer_denom()
    # 断言分子应为 sqrt(-1)
    assert num == sqrt(-1)
    # 断言分母应为 sqrt(-neg)
    assert den == sqrt(-neg)
    # 计算 sqrt(1/nonneg) 的分子和分母
    num, den = sqrt(1/nonneg).as_numer_denom()
    # 断言分子应为 1
    assert num == 1
    # 断言分母应为 sqrt(nonneg)
    assert den == sqrt(nonneg)
    # 计算 sqrt(1/any) 的分子和分母
    num, den = sqrt(1/any).as_numer_denom()
    # 断言分子应为 sqrt(1/any)
    assert num == sqrt(1/any)
    # 断言分母应为 1

    # 定义一个函数 eqn，用于计算 num/den 的 pow 次方
    def eqn(num, den, pow):
        return (num/den)**pow
    # 初始化一些正数和负数用于测试
    npos = 1
    nneg = -1
    dpos = 2 - sqrt(3)
    dneg = 1 - sqrt(3)
    # 断言：确保 dpos 大于 0，dneg 小于 0，npos 大于 0，nneg 小于 0
    assert dpos > 0 and dneg < 0 and npos > 0 and nneg < 0
    # 测试不同组合的 eqn 函数调用结果
    # 正数或负数整数
    eq = eqn(npos, dpos, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dpos**2)
    eq = eqn(npos, dneg, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dneg**2)
    eq = eqn(nneg, dpos, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dpos**2)
    eq = eqn(nneg, dneg, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dneg**2)
    eq = eqn(npos, dpos, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**2, 1)
    eq = eqn(npos, dneg, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dneg**2, 1)
    eq = eqn(nneg, dpos, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**2, 1)
    eq = eqn(nneg, dneg, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dneg**2, 1)
    # 正数或负数有理数
    pow = S.Half
    eq = eqn(npos, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (npos**pow, dpos**pow)
    eq = eqn(npos, dneg, pow)
    assert eq.is_Pow is False and eq.as_numer_denom() == ((-npos)**pow, (-dneg)**pow)
    eq = eqn(nneg, dpos, pow)
    assert not eq.is_Pow or eq.as_numer_denom() == (nneg**pow, dpos**pow)
    eq = eqn(nneg, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-nneg)**pow, (-dneg)**pow)
    eq = eqn(npos, dpos, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**pow, npos**pow)
    eq = eqn(npos, dneg, -pow)
    assert eq.is_Pow is False and eq.as_numer_denom() == (-(-npos)**pow*(-dneg)**pow, npos)
    eq = eqn(nneg, dpos, -pow)
    assert not eq.is_Pow or eq.as_numer_denom() == (dpos**pow, nneg**pow)
    eq = eqn(nneg, dneg, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-nneg)**pow)
    # 未知指数
    pow = 2*any
    eq = eqn(npos, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (npos**pow, dpos**pow)
    eq = eqn(npos, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-npos)**pow, (-dneg)**pow)
    eq = eqn(nneg, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (nneg**pow, dpos**pow)
    eq = eqn(nneg, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-nneg)**pow, (-dneg)**pow)
    eq = eqn(npos, dpos, -pow)
    assert eq.as_numer_denom() == (dpos**pow, npos**pow)
    eq = eqn(npos, dneg, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-npos)**pow)
    eq = eqn(nneg, dpos, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**pow, nneg**pow)
    eq = eqn(nneg, dneg, -pow)
    # 确保等式左侧为幂运算，并且分子分母形式等于((-dneg)**pow, (-nneg)**pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-nneg)**pow)

    # 确保等式左侧的分子分母形式为((1/(1 + x/3))**(-S.One))，等于(3 + x, 3)
    assert ((1/(1 + x/3))**(-S.One)).as_numer_denom() == (3 + x, 3)

    # 创建一个符号变量notp，其值为非正，即positive=False
    notp = Symbol('notp', positive=False)  # not positive does not imply real

    # 计算b的值，b为表达式((1 + x/notp)**-2)
    b = ((1 + x/notp)**-2)

    # 确保等式左侧的分子分母形式为(b**(-y))，等于(1, b**y)
    assert (b**(-y)).as_numer_denom() == (1, b**y)

    # 确保等式左侧的分子分母形式为(b**(-S.One))，等于((notp + x)**2, notp**2)
    assert (b**(-S.One)).as_numer_denom() == ((notp + x)**2, notp**2)

    # 创建一个符号变量nonp，其值为非正，即nonpositive=True
    nonp = Symbol('nonp', nonpositive=True)

    # 确保等式左侧的分子分母形式为(((1 + x/nonp)**-2)**(-S.One))，等于((-nonp - x)**2, nonp**2)
    assert (((1 + x/nonp)**-2)**(-S.One)).as_numer_denom() == ((-nonp - x)**2, nonp**2)

    # 创建一个符号变量n，其值为负数，即negative=True
    n = Symbol('n', negative=True)

    # 确保等式左侧的分子分母形式为(x**n)，等于(1, x**-n)
    assert (x**n).as_numer_denom() == (1, x**-n)

    # 确保等式左侧的分子分母形式为(sqrt(1/n))，等于(S.ImaginaryUnit, sqrt(-n))
    assert sqrt(1/n).as_numer_denom() == (S.ImaginaryUnit, sqrt(-n))

    # 创建一个符号变量n，其值为非正，即nonpositive=True
    n = Symbol('0 or neg', nonpositive=True)
    # 对于表达式(1/sqrt(x/n))，确保等式左侧的分子分母形式等于(sqrt(-n), sqrt(-x))
    # 注释部分解释了当x和n没有经过否定处理时，并且n为负数时可能会得到错误的结果
    # 这段注释还讨论了n为0时不会有问题，因为1/oo和1/zoo都是零，sqrt(0)/sqrt(-x)也是零，除非x也是零（在这种情况下，负号无关紧要）
    # 例如：1/sqrt(1/-1) = -I，但sqrt(-1)/sqrt(1) = I
    assert (1/sqrt(x/n)).as_numer_denom() == (sqrt(-n), sqrt(-x))

    # 创建一个符号变量c，其值为复数，即complex=True
    c = Symbol('c', complex=True)
    # 计算e的值为sqrt(1/c)
    e = sqrt(1/c)

    # 确保等式左侧的分子分母形式为(e)，等于(e, 1)
    assert e.as_numer_denom() == (e, 1)

    # 创建一个符号变量i，其值为整数，即integer=True
    i = Symbol('i', integer=True)
    # 确保等式左侧的分子分母形式为((1 + x/y)**i)，等于((x + y)**i, y**i)
    assert ((1 + x/y)**i).as_numer_denom() == ((x + y)**i, y**i)
def test_Pow_Expr_args():
    bases = [Basic(), Poly(x, x), FiniteSet(x)]
    for base in bases:
        # 缓存可能会影响堆栈级别测试
        with warns(SymPyDeprecationWarning, test_stacklevel=False):
            # 测试 Pow 函数在给定基础上的行为
            Pow(base, S.One)


def test_Pow_signs():
    """Cf. issues 4595 and 5250"""
    n = Symbol('n', even=True)
    # 检查指数运算中符号的影响
    assert (3 - y)**2 != (y - 3)**2
    assert (3 - y)**n != (y - 3)**n
    assert (-3 + y - x)**2 != (3 - y + x)**2
    assert (y - 3)**3 != -(3 - y)**3


def test_power_with_noncommutative_mul_as_base():
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    # 检查非交换乘法作为基础的幂运算
    assert not (x*y)**3 == x**3*y**3
    assert (2*x*y)**3 == 8*(x*y)**3


@_both_exp_pow
def test_power_rewrite_exp():
    # 测试幂函数重写为指数函数的情况
    assert (I**I).rewrite(exp) == exp(-pi/2)

    expr = (2 + 3*I)**(4 + 5*I)
    assert expr.rewrite(exp) == exp((4 + 5*I)*(log(sqrt(13)) + I*atan(Rational(3, 2))))
    assert expr.rewrite(exp).expand() == \
        169*exp(5*I*log(13)/2)*exp(4*I*atan(Rational(3, 2)))*exp(-5*atan(Rational(3, 2)))

    assert ((6 + 7*I)**5).rewrite(exp) == 7225*sqrt(85)*exp(5*I*atan(Rational(7, 6)))

    expr = 5**(6 + 7*I)
    assert expr.rewrite(exp) == exp((6 + 7*I)*log(5))
    assert expr.rewrite(exp).expand() == 15625*exp(7*I*log(5))

    assert Pow(123, 789, evaluate=False).rewrite(exp) == 123**789
    assert (1**I).rewrite(exp) == 1**I
    assert (0**I).rewrite(exp) == 0**I

    expr = (-2)**(2 + 5*I)
    assert expr.rewrite(exp) == exp((2 + 5*I)*(log(2) + I*pi))
    assert expr.rewrite(exp).expand() == 4*exp(-5*pi)*exp(5*I*log(2))

    assert ((-2)**S(-5)).rewrite(exp) == (-2)**S(-5)

    x, y = symbols('x y')
    assert (x**y).rewrite(exp) == exp(y*log(x))
    if global_parameters.exp_is_pow:
        assert (7**x).rewrite(exp) == Pow(S.Exp1, x*log(7), evaluate=False)
    else:
        assert (7**x).rewrite(exp) == exp(x*log(7), evaluate=False)
    assert ((2 + 3*I)**x).rewrite(exp) == exp(x*(log(sqrt(13)) + I*atan(Rational(3, 2))))
    assert (y**(5 + 6*I)).rewrite(exp) == exp(log(y)*(5 + 6*I))

    assert all((1/func(x)).rewrite(exp) == 1/(func(x).rewrite(exp)) for func in
                    (sin, cos, tan, sec, csc, sinh, cosh, tanh))


def test_zero():
    assert 0**x != 0
    assert 0**(2*x) == 0**x
    assert 0**(1.0*x) == 0**x
    assert 0**(2.0*x) == 0**x
    assert (0**(2 - x)).as_base_exp() == (0, 2 - x)
    assert 0**(x - 2) != S.Infinity**(2 - x)
    assert 0**(2*x*y) == 0**(x*y)
    assert 0**(-2*x*y) == S.ComplexInfinity**(x*y)
    assert Float(0)**2 is not S.Zero
    assert Float(0)**2 == 0.0
    assert Float(0)**-2 is zoo
    assert Float(0)**oo is S.Zero

    # 测试问题 19572
    assert 0 ** -oo is zoo
    assert power(0, -oo) is zoo
    assert Float(0)**-oo is zoo


def test_pow_as_base_exp():
    assert (S.Infinity**(2 - x)).as_base_exp() == (S.Infinity, 2 - x)
    assert (S.Infinity**(x - 2)).as_base_exp() == (S.Infinity, x - 2)
    p = S.Half**x
    assert p.base, p.exp == p.as_base_exp() == (S(2), -x)
    # 创建一个指数表达式 p = (3/2)**x
    p = (S(3)/2)**x
    # 使用 assert 断言验证以下条件：
    # p.base 是指数表达式的基数部分，p.exp 是指数表达式的指数部分
    # p.as_base_exp() 返回基数和指数的元组
    assert p.base, p.exp == p.as_base_exp() == (3*S.Half, x)
    
    # 更新指数表达式为 p = (2/3)**x
    p = (S(2)/3)**x
    # 使用 assert 断言验证以下条件：
    # p.as_base_exp() 返回基数和指数的元组
    assert p.as_base_exp() == (S(3)/2, -x)
    # 使用 assert 断言验证以下条件：
    # p.base 是指数表达式的基数部分，p.exp 是指数表达式的指数部分
    assert p.base, p.exp == (S(2)/3, x)
    
    # issue 8344:
    # 使用 assert 断言验证 Pow(1, 2, evaluate=False) 的 as_base_exp() 返回 (1, 2)
    assert Pow(1, 2, evaluate=False).as_base_exp() == (S.One, S(2))
# 定义测试函数 test_nseries，用于测试 _eval_nseries 方法的数学表达式近似级数计算
def test_nseries():
    # 断言平方根表达式的 x 的级数展开结果与预期相等
    assert sqrt(I*x - 1)._eval_nseries(x, 4, None, 1) == I + x/2 + I*x**2/8 - x**3/16 + O(x**4)
    # 断言平方根表达式的 x 的级数展开结果与预期相等（使用不同的符号参数）
    assert sqrt(I*x - 1)._eval_nseries(x, 4, None, -1) == -I - x/2 - I*x**2/8 + x**3/16 + O(x**4)
    # 断言立方根表达式的 x 的级数展开结果与预期相等
    assert cbrt(I*x - 1)._eval_nseries(x, 4, None, 1) == (-1)**(S(1)/3) - (-1)**(S(5)/6)*x/3 + \
        (-1)**(S(1)/3)*x**2/9 + 5*(-1)**(S(5)/6)*x**3/81 + O(x**4)
    # 断言立方根表达式的 x 的级数展开结果与预期相等（使用不同的符号参数）
    assert cbrt(I*x - 1)._eval_nseries(x, 4, None, -1) == -(-1)**(S(2)/3) - (-1)**(S(1)/6)*x/3 - \
        (-1)**(S(2)/3)*x**2/9 + 5*(-1)**(S(1)/6)*x**3/81 + O(x**4)
    # 断言给定表达式的级数展开结果与预期相等
    assert (1 / (exp(-1/x) + 1/x))._eval_nseries(x, 2, None) == x + O(x**2)
    # test issue 23752
    # 断言平方根表达式的 x 的级数展开结果与预期相等（针对特定问题）
    assert sqrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, 1) == -sqrt(3)*I + sqrt(3)*I*x/6 - \
        sqrt(3)*I*x**2*(-S(1)/72 + I/6) - sqrt(3)*I*x**3*(-S(1)/432 + I/36) + O(x**4)
    # 断言平方根表达式的 x 的级数展开结果与预期相等（使用不同的符号参数，针对特定问题）
    assert sqrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, -1) == -sqrt(3)*I + sqrt(3)*I*x/6 - \
        sqrt(3)*I*x**2*(-S(1)/72 + I/6) - sqrt(3)*I*x**3*(-S(1)/432 + I/36) + O(x**4)
    # 断言立方根表达式的 x 的级数展开结果与预期相等（针对特定问题）
    assert cbrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, 1) == -(-1)**(S(2)/3)*3**(S(1)/3) + \
        (-1)**(S(2)/3)*3**(S(1)/3)*x/9 - (-1)**(S(2)/3)*3**(S(1)/3)*x**2*(-S(1)/81 + I/9) - \
        (-1)**(S(2)/3)*3**(S(1)/3)*x**3*(-S(5)/2187 + 2*I/81) + O(x**4)
    # 断言立方根表达式的 x 的级数展开结果与预期相等（使用不同的符号参数，针对特定问题）
    assert cbrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, -1) == -(-1)**(S(2)/3)*3**(S(1)/3) + \
        (-1)**(S(2)/3)*3**(S(1)/3)*x/9 - (-1)**(S(2)/3)*3**(S(1)/3)*x**2*(-S(1)/81 + I/9) - \
        (-1)**(S(2)/3)*3**(S(1)/3)*x**3*(-S(5)/2187 + 2*I/81) + O(x**4)


# 定义测试函数 test_issue_6100_12942_4473，用于测试不同表达式的符号和数值操作
def test_issue_6100_12942_4473():
    # 断言 x 的浮点次方不等于 x
    assert x**1.0 != x
    # 断言 x 不等于 x 的浮点次方
    assert x != x**1.0
    # 断言 True 不等于 x 的浮点次方
    assert True != x**1.0
    # 断言 x 的浮点次方不等于 True
    assert x**1.0 is not True
    # 断言 x 不等于 True
    assert x is not True
    # 断言 x*y 不等于 (x*y) 的浮点次方
    assert x*y != (x*y)**1.0
    # 断言 (x 的浮点次方的浮点次方) 不等于 x
    assert (x**1.0)**1.0 != x
    # 断言 (x 的浮点次方的浮点次方) 不等于 x 的平方
    assert (x**1.0)**2.0 != x**2
    # 创建一个空的表达式对象 b
    b = Expr()
    # 断言使用 evaluate=False 创建的 Pow 对象不等于 b
    assert Pow(b, 1.0, evaluate=False) != b
    # 断言 ((x*y) 的浮点次方) 的函数不是 Pow
    assert ((x*y)**1.0).func is Pow


# 定义测试函数 test_issue_6208，用于测试数学函数和表达式的特定问题
def test_issue_6208():
    # 导入根函数并进行特定问题的断言测试
    from sympy.functions.elementary.miscellaneous import root
    # 断言 33 的 I*9/10 次方的平方根等于 -33 的 I*9/20 次方
    assert sqrt(33**(I*9/10)) == -33**(I*9/20)
    # 断言 (6*I) 的 2*I 次方的根数为 3 的基数部分等于有理数 1/3
    assert root((6*I)**(2*I), 3).as_base_exp()[1] == Rational(1, 3)  # != 2*I/3
    # 断言 (6*I) 的 I/3 次方的根数为 3 的基数部分等于 I/9
    assert root((6*I)**(I/3), 3).as_base_exp()[1] == I/9
    # 断言 exp(3*I) 的平方根等于 exp(3*I/2)
    assert sqrt(exp(3*I)) == exp(3*I/2)
    # 断言 -sqrt(3)*(1 + 2*I) 的平方根等于 sqrt(3)*sqrt(-1 - 2*I)
    assert sqrt(-sqrt(3)*(1 + 2*I)) == sqrt(sqrt(3))*sqrt(-1 - 2*I)
    # 断言 exp(5*I) 的平方根等于 -exp(5*I/2)
    assert sqrt(exp(5*I)) == -exp(5*I/2)
    # 断言 exp(5*I) 的根数为 3 的指数部分等于有理数 1/3
    assert root(exp(5*I), 3).exp == Rational(1, 3)


# 定义测试函数 test_issue_6990，用于测试数学函数中的表达式级数展开
def test_issue_6990():
    # 断言 sqrt(a + b*x + x**2) 的 x 的级数展开去除高阶项后与预期相等
    assert (sqrt(a + b*x
    # 断言：验证 sin(x) 的平方根在 x = 0 处展开的前 9 项系数是否等于指定的表达式
    assert sqrt(sin(x)).series(x, 0, 9) == \
        sqrt(x) - x**Rational(5, 2)/12 + x**Rational(9, 2)/1440 - \
        x**Rational(13, 2)/24192 - 67*x**Rational(17, 2)/29030400 + O(x**9)
    
    # 断言：验证 sin(x**3) 的平方根在 x = 0 处展开的前 19 项系数是否等于指定的表达式
    assert sqrt(sin(x**3)).series(x, 0, 19) == \
        x**Rational(3, 2) - x**Rational(15, 2)/12 + x**Rational(27, 2)/1440 + O(x**19)
    
    # 断言：验证 sin(x**3) 的平方根在 x = 0 处展开的前 20 项系数是否等于指定的表达式
    assert sqrt(sin(x**3)).series(x, 0, 20) == \
        x**Rational(3, 2) - x**Rational(15, 2)/12 + x**Rational(27, 2)/1440 - \
        x**Rational(39, 2)/24192 + O(x**20)
# 测试函数，验证在特定数学表达式上的符号计算
def test_issue_6782():
    # 验证在 x**3 的正弦平方根的级数展开
    assert sqrt(sin(x**3)).series(x, 0, 7) == x**Rational(3, 2) + O(x**7)
    # 验证在 x**4 的正弦平方根的级数展开
    assert sqrt(sin(x**4)).series(x, 0, 3) == x**2 + O(x**3)


# 测试函数，验证在特定数学表达式上的级数展开
def test_issue_6653():
    # 验证在 1/(sqrt(1 + sin(x**2))) 的级数展开
    assert (1 / sqrt(1 + sin(x**2))).series(x, 0, 3) == 1 - x**2/2 + O(x**3)


# 测试函数，验证在特定表达式上的泰勒项计算
def test_issue_6429():
    # 定义表达式 f
    f = (c**2 + x)**(0.5)
    # 验证 f 在 x0=0, n=1 时的泰勒展开项
    assert f.series(x, x0=0, n=1) == (c**2)**0.5 + O(x)
    # 验证 f 的一阶泰勒项
    assert f.taylor_term(0, x) == (c**2)**0.5
    # 验证 f 的二阶泰勒项
    assert f.taylor_term(1, x) == 0.5*x*(c**2)**(-0.5)
    # 验证 f 的三阶泰勒项
    assert f.taylor_term(2, x) == -0.125*x**2*(c**2)**(-1.5)


# 测试函数，验证在特定数学表达式上的幂运算和指数运算
def test_issue_7638():
    # 定义常数 f
    f = pi/log(sqrt(2))
    # 验证复数幂次方的计算
    assert ((1 + I)**(I*f/2))**0.3 == (1 + I)**(0.15*I*f)
    # 验证复数幂次方的计算
    assert (1 + I)**(4*I*f) == ((1 + I)**(12*I*f))**Rational(1, 3)
    # 验证指数运算的结果是否为 1/3
    assert (((1 + I)**(I*(1 + 7*f)))**Rational(1, 3)).exp == Rational(1, 3)
    # 验证平方根运算，等于绝对值
    r = symbols('r', real=True)
    assert sqrt(r**2) == abs(r)
    # 验证立方根运算
    assert cbrt(r**3) != r
    # 验证复数的平方根计算
    assert sqrt(Pow(2*I, 5*S.Half)) != (2*I)**Rational(5, 4)
    # 验证正数的立方根计算
    p = symbols('p', positive=True)
    assert cbrt(p**2) == p**Rational(2, 3)
    # 验证复数幂次方的数值结果
    assert NS(((0.2 + 0.7*I)**(0.7 + 1.0*I))**(0.5 - 0.1*I), 1) == '0.4 + 0.2*I'
    # 验证复数的平方根计算
    assert sqrt(1/(1 + I)) == sqrt(1 - I)/sqrt(2)  # or 1/sqrt(1 + I)
    # 定义常数 e
    e = 1/(1 - sqrt(2))
    # 验证复数的平方根计算
    assert sqrt(e) == I/sqrt(-1 + sqrt(2))
    # 验证复数的负幂次方计算
    assert e**Rational(-1, 2) == -I*sqrt(-1 + sqrt(2))
    # 验证带有三角函数的复数幂次方计算
    assert sqrt((cos(1)**2 + sin(1)**2 - 1)**(3 + I)).exp in [S.Half, Rational(3, 2) + I/2]
    # 验证正数的平方根计算
    assert sqrt(r**Rational(4, 3)) != r**Rational(2, 3)
    # 验证复数的立方根计算
    assert sqrt((p + I)**Rational(4, 3)) == (p + I)**Rational(2, 3)
    # 循环验证正负虚数的平方根计算
    for q in 1+I, 1-I:
        assert sqrt(q**2) == q
    for q in -1+I, -1-I:
        assert sqrt(q**2) == -q
    # 验证复数的平方根计算
    assert sqrt((p + r*I)**2) != p + r*I
    # 定义常数 e
    e = (1 + I/5)
    # 验证复数的五次方根计算
    assert sqrt(e**5) == e**(5*S.Half)
    # 验证复数的六次方根计算
    assert sqrt(e**6) == e**3
    # 验证复数的六次方根计算
    assert sqrt((1 + I*r)**6) != (1 + I*r)**3


# 测试函数，验证幂运算的结果
def test_issue_8582():
    # 验证无穷大的幂运算结果为 NaN
    assert 1**oo is nan
    assert 1**(-oo) is nan
    assert 1**zoo is nan
    assert 1**(oo + I) is nan
    assert 1**(1 + I*oo) is nan
    assert 1**(oo + I*oo) is nan


# 测试函数，验证在特定表达式上的数学运算
def test_issue_8650():
    # 定义符号 n 为整数且非负
    n = Symbol('n', integer=True, nonnegative=True)
    # 验证 n**n 是正数
    assert (n**n).is_positive is True
    # 定义表达式 x
    x = 5*n + 5
    # 验证 x**(5*(n + 1)) 是正数
    assert (x**(5*(n + 1))).is_positive is True


# 测试函数，验证在特定表达式上的幂运算结果
def test_issue_13914():
    # 定义符号 b
    b = Symbol('b')
    # 验证幂运算的结果为 NaN
    assert (-1)**zoo is nan
    assert 2**zoo is nan
    assert (S.Half)**(1 + zoo) is nan
    assert I**(zoo + I) is nan
    assert b**(I + zoo) is nan


# 测试函数，验证在特定复数表达式上的平方根计算
def test_better_sqrt():
    # 定义符号 n 为整数且非负
    n = Symbol('n', integer=True, nonnegative=True)
    # 验证复数的平方根计算
    assert sqrt(3 + 4*I) == 2 + I
    assert sqrt(3 - 4*I) == 2 - I
    assert sqrt(-3 - 4*I) == 1 - 2*I
    assert sqrt(-3 + 4*I) == 1 + 2*I
    assert sqrt(32 + 24*I) == 6 + 2*I
    assert sqrt(32 - 24*I) == 6 - 2*I
    assert sqrt(-32 - 24*I) == 2 - 6*I
    assert sqrt(-32 + 24*I) == 2 + 6*I
    # triple (3, 4, 5):
    # 3的奇偶性与5的奇偶性匹配，而4是一个平方数
    assert sqrt((3 + 4*I)/4) == 1 + I/2

    # triple (8, 15, 17)
    # 8的奇偶性与17的奇偶性不匹配，但8除以2是一个平方数
    assert sqrt((8 + 15*I)/8) == (5 + 3*I)/4

    # 处理分母
    assert sqrt((3 - 4*I)/25) == (2 - I)/5
    assert sqrt((3 - 4*I)/26) == (2 - I)/sqrt(26)

    # mul
    # 问题 #12739
    assert sqrt((3 + 4*I)/(3 - 4*I)) == (3 + 4*I)/5
    assert sqrt(2/(3 + 4*I)) == sqrt(2)/5*(2 - I)
    assert sqrt(n/(3 + 4*I)).subs(n, 2) == sqrt(2)/5*(2 - I)
    assert sqrt(-2/(3 + 4*I)) == sqrt(2)/5*(1 + 2*I)
    assert sqrt(-n/(3 + 4*I)).subs(n, 2) == sqrt(2)/5*(1 + 2*I)

    # power
    assert sqrt(1/(3 + I*4)) == (2 - I)/5
    assert sqrt(1/(3 - I)) == sqrt(10)*sqrt(3 + I)/10

    # symbolic
    i = symbols('i', imaginary=True)
    assert sqrt(3/i) == Mul(sqrt(3), 1/sqrt(i), evaluate=False)

    # multiples of 1/2; don't make this too automatic
    assert sqrt(3 + 4*I)**3 == (2 + I)**3
    assert Pow(3 + 4*I, Rational(3, 2)) == 2 + 11*I
    assert Pow(6 + 8*I, Rational(3, 2)) == 2*sqrt(2)*(2 + 11*I)

    # 对a进行定义和相关断言
    n, d = (3 + 4*I), (3 - 4*I)**3
    a = n/d
    assert a.args == (1/d, n)
    eq = sqrt(a)
    assert eq.args == (a, S.Half)
    assert expand_multinomial(eq) == sqrt((-117 + 44*I)*(3 + 4*I))/125
    assert eq.expand() == (7 - 24*I)/125

    # issue 12775
    # 正虚部分
    assert sqrt(2*I) == (1 + I)
    assert sqrt(2*9*I) == Mul(3, 1 + I, evaluate=False)
    assert Pow(2*I, 3*S.Half) == (1 + I)**3

    # 负虚部分
    assert sqrt(-I/2) == Mul(S.Half, 1 - I, evaluate=False)

    # 分数虚部分
    assert Pow(Rational(-9, 2)*I, Rational(3, 2)) == 27*(1 - I)**3/8
# 测试 issue 2993
def test_issue_2993():
    # 断言表达式的字符串表示是否等于给定字符串
    assert str((2.3*x - 4)**0.3) == '1.5157165665104*(0.575*x - 1)**0.3'
    assert str((2.3*x + 4)**0.3) == '1.5157165665104*(0.575*x + 1)**0.3'
    assert str((-2.3*x + 4)**0.3) == '1.5157165665104*(1 - 0.575*x)**0.3'
    assert str((-2.3*x - 4)**0.3) == '1.5157165665104*(-0.575*x - 1)**0.3'
    assert str((2.3*x - 2)**0.3) == '1.28386201800527*(x - 0.869565217391304)**0.3'
    assert str((-2.3*x - 2)**0.3) == '1.28386201800527*(-x - 0.869565217391304)**0.3'
    assert str((-2.3*x + 2)**0.3) == '1.28386201800527*(0.869565217391304 - x)**0.3'
    assert str((2.3*x + 2)**0.3) == '1.28386201800527*(x + 0.869565217391304)**0.3'
    assert str((2.3*x - 4)**Rational(1, 3)) == '2**(2/3)*(0.575*x - 1)**(1/3)'
    # 将表达式赋值给变量 eq
    eq = (2.3*x + 4)
    # 断言平方是否等于给定表达式的平方
    assert eq**2 == 16*(0.575*x + 1)**2
    # 断言分母的 args 属性是否为元组，包含 eq 和 -1
    assert (1/eq).args == (eq, -1)  # 不改变平凡的幂
    # issue 17735
    # 计算 q 的值
    q = .5*exp(x) - .5*exp(-x) + 0.1
    # 断言将 x 替换为 1 后，q 的平方的整数部分是否等于 1
    assert int((q**2).subs(x, 1)) == 1
    # issue 17756
    # 声明符号 y
    y = Symbol('y')
    # 断言将 y 替换为 pi.n(25) 后，sqrt(...) 的浮点数原子数是否为 2
    assert len(sqrt(x/(x + y)**2 + Float('0.008', 30)).subs(y, pi.n(25)).atoms(Float)) == 2
    # issue 17756
    # 声明符号 a, b, c, d, e, f, g
    a, b, c, d, e, f, g = symbols('a:g')
    # 定义表达式 expr
    expr = sqrt(1 + a*(c**4 + g*d - 2*g*e - f*(-g + d))**2 /
        (c**3*b**2*(d - 3*e + 2*f)**2))/2
    # 定义符号变量的数值列表 r
    r = [
        (a, N('0.0170992456333788667034850458615', 30)),
        (b, N('0.0966594956075474769169134801223', 30)),
        (c, N('0.390911862903463913632151616184', 30)),
        (d, N('0.152812084558656566271750185933', 30)),
        (e, N('0.137562344465103337106561623432', 30)),
        (f, N('0.174259178881496659302933610355', 30)),
        (g, N('0.220745448491223779615401870086', 30))]
    # 计算 expr 在数值替换 r 下的数值，赋给 tru
    tru = expr.n(30, subs=dict(r))
    # 计算 expr 在数值替换 r 下的表达式，赋给 seq
    seq = expr.subs(r)
    # 虽然 tru 是使用数值计算 expr 的正确方式，但如果系数提取错误，seq 的精度将显著降低
    # 断言 seq 是否等于 tru
    assert seq == tru

# 测试 issue 17450
def test_issue_17450():
    # 断言一些表达式的实部是否为 None
    assert (erf(cosh(1)**7)**I).is_real is None
    assert (erf(cosh(1)**7)**I).is_imaginary is False
    assert (Pow(exp(1+sqrt(2)), ((1-sqrt(2))*I*pi), evaluate=False)).is_real is None
    assert ((-10)**(10*I*pi/3)).is_real is False
    assert ((-5)**(4*I*pi)).is_real is False

# 测试 issue 18190
def test_issue_18190():
    # 断言一些表达式的平方根是否满足特定关系
    assert sqrt(1 / tan(1 + I)) == 1 / sqrt(tan(1 + I))

# 测试 issue 14815
def test_issue_14815():
    # 断言不同类型 x 的平方根是否具有预期的 is_extended_negative 值
    x = Symbol('x', real=True)
    assert sqrt(x).is_extended_negative is False
    x = Symbol('x', real=False)
    assert sqrt(x).is_extended_negative is None
    x = Symbol('x', complex=True)
    assert sqrt(x).is_extended_negative is False
    x = Symbol('x', extended_real=True)
    assert sqrt(x).is_extended_negative is False
    assert sqrt(zoo, evaluate=False).is_extended_negative is False
    assert sqrt(nan, evaluate=False).is_extended_negative is None

# 测试 issue 18509
def test_issue_18509():
    x = Symbol('x', prime=True)
    # 断言一些表达式的幂是否符合预期的数学定义
    assert x**oo is oo
    assert (1/x)**oo is S.Zero
    assert (-1/x)**oo is S.Zero
    assert (-x)**oo is zoo
    # 断言：负无穷的负一次方乘以负一虚数单位是零
    assert (-oo)**(-1 + I) is S.Zero
    # 断言：负无穷的正一次方乘以负一虚数单位是正无穷
    assert (-oo)**(1 + I) is zoo
    # 断言：正无穷的负一次方乘以正一虚数单位是零
    assert (oo)**(-1 + I) is S.Zero
    # 断言：正无穷的正一次方乘以正一虚数单位是正无穷
    assert (oo)**(1 + I) is zoo
def test_issue_18762():
    # 定义符号变量 e 和 p
    e, p = symbols('e p')
    # 计算 g0 函数表达式，包含符号操作和数学函数
    g0 = sqrt(1 + e**2 - 2*e*cos(p))
    # 断言 g0 在 e 展开为级数后的参数个数为 4
    assert len(g0.series(e, 1, 3).args) == 4


def test_issue_21860():
    # 定义符号表达式 e，涉及有理数运算和符号替换
    e = 3*2**Rational(66666666667,200000000000)*3**Rational(16666666667,50000000000)*x**Rational(66666666667, 200000000000)
    # 定义预期结果 ans，包含有理数运算和幂函数
    ans = Mul(Rational(3, 2),
              Pow(Integer(2), Rational(33333333333, 100000000000)),
              Pow(Integer(3), Rational(26666666667, 40000000000)))
    # 断言在将 x 替换为有理数 Rational(3,8) 后，表达式 e 等于 ans
    assert e.xreplace({x: Rational(3,8)}) == ans


def test_issue_21647():
    # 定义符号表达式 e，包含对数运算和大整数幂次
    e = log((Integer(567)/500)**(811*(Integer(567)/500)**x/100))
    # 定义预期结果 ans，包含有理数和幂函数的组合
    ans = log(Mul(Rational(64701150190720499096094005280169087619821081527,
                           76293945312500000000000000000000000000000000000),
                  Pow(Integer(2), Rational(396204892125479941, 781250000000000000)),
                  Pow(Integer(3), Rational(385045107874520059, 390625000000000000)),
                  Pow(Integer(5), Rational(407364676376439823, 1562500000000000000)),
                  Pow(Integer(7), Rational(385045107874520059, 1562500000000000000))))
    # 断言在将 x 替换为整数 6 后，表达式 e 等于 ans
    assert e.xreplace({x: 6}) == ans


def test_issue_21762():
    # 定义符号表达式 e，包含幂次和有理数的乘法
    e = (x**2 + 6)**(Integer(33333333333333333)/50000000000000000)
    # 定义预期结果 ans，包含有理数和幂函数的组合
    ans = Mul(Rational(5, 4),
              Pow(Integer(2), Rational(16666666666666667, 25000000000000000)),
              Pow(Integer(5), Rational(8333333333333333, 25000000000000000)))
    # 断言在将 x 替换为 S.Half 后，表达式 e 等于 ans
    assert e.xreplace({x: S.Half}) == ans


def test_issue_14704():
    # 定义整数 a 的幂次运算
    a = 144**144
    # 计算 a 的整数根和精确根
    x, xexact = integer_nthroot(a,a)
    # 断言 x 等于 1 并且 xexact 为 False
    assert x == 1 and xexact is False


def test_rational_powers_larger_than_one():
    # 断言有理数的幂次运算结果
    assert Rational(2, 3)**Rational(3, 2) == 2*sqrt(6)/9
    assert Rational(1, 6)**Rational(9, 4) == 6**Rational(3, 4)/216
    assert Rational(3, 7)**Rational(7, 3) == 9*3**Rational(1, 3)*7**Rational(2, 3)/343


def test_power_dispatcher():
    # 定义自定义类 NewBase 和 NewPow
    class NewBase(Expr):
        pass
    class NewPow(NewBase, Pow):
        pass
    # 定义符号变量 a 和自定义类 NewBase 的实例 b
    a, b = Symbol('a'), NewBase()

    # 注册不同类型组合的幂次函数
    @power.register(Expr, NewBase)
    @power.register(NewBase, Expr)
    @power.register(NewBase, NewBase)
    def _(a, b):
        return NewPow(a, b)

    # 断言使用默认 Pow 函数计算 2 的 3 次方等于 8*S.One
    assert power(2, 3) == 8*S.One
    # 断言使用默认 Pow 函数计算 a 的 2 次方等于 Pow(a, 2)
    assert power(a, 2) == Pow(a, 2)
    # 断言使用自定义 NewPow 函数计算 a 的 a 次方等于 NewPow(a, a)
    assert power(a, a) == NewPow(a, a)

    # 断言使用自定义 NewPow 函数计算 a 和 b 的幂次等于 NewPow(a, b)
    assert power(a, b) == NewPow(a, b)
    assert power(b, a) == NewPow(b, a)
    assert power(b, b) == NewPow(b, b)


def test_powers_of_I():
    # 断言计算复数单位 I 的幂次结果
    assert [sqrt(I)**i for i in range(13)] == [
        1, sqrt(I), I, sqrt(I)**3, -1, -sqrt(I), -I, -sqrt(I)**3,
        1, sqrt(I), I, sqrt(I)**3, -1]
    # 断言计算复数单位 I 的 S(9)/2 次方结果
    assert sqrt(I)**(S(9)/2) == -I**(S(1)/4)


def test_issue_23918():
    # 定义有理数 b
    b = S(2)/3
    # 断言 b 的 x 次方的基数和指数部分
    assert (b**x).as_base_exp() == (1/b, -x)


def test_issue_26546():
    # 定义实数符号变量 x
    x = Symbol('x', real=True)
    # 断言 x 是扩展实数域的一部分
    assert x.is_extended_real is True
    # 断言表达式 sqrt(x+I) 不属于扩展实数域
    assert sqrt(x+I).is_extended_real is False
    # 断言表达式 (x+I)^(1/2) 不属于扩展实数域
    assert Pow(x+I, S.Half).is_extended_real is False
    # 断言表达式 (x+I)^(1/2) 不属于扩展实数域
    assert Pow(x+I, Rational(1,2)).is_extended_real is False
    # 断言：检查复数 x + I 的 1/13 次方是否为非扩展实数
    assert Pow(x+I, Rational(1,13)).is_extended_real is False
    
    # 断言：检查复数 x + I 的 2/3 次方是否为未定义状态（即不确定是否为扩展实数）
    assert Pow(x+I, Rational(2,3)).is_extended_real is None
```