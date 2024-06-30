# `D:\src\scipysrc\sympy\sympy\core\tests\test_numbers.py`

```
# 导入名为 nums 的 numbers 模块
import numbers as nums
# 导入 decimal 模块
import decimal
# 从 sympy.concrete.summations 模块导入 Sum 类
from sympy.concrete.summations import Sum
# 从 sympy.core 模块导入 EulerGamma, Catalan, TribonacciConstant, GoldenRatio 类/常数
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
    GoldenRatio)
# 从 sympy.core.containers 模块导入 Tuple 类
from sympy.core.containers import Tuple
# 从 sympy.core.expr 模块导入 unchanged 函数
from sympy.core.expr import unchanged
# 从 sympy.core.logic 模块导入 fuzzy_not 函数
from sympy.core.logic import fuzzy_not
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.numbers 模块导入 mpf_norm, seterr, Integer, I, pi, comp, Rational, E, nan, oo, AlgebraicNumber, Number, Float, zoo, equal_valued, int_valued, all_close 函数/常数
from sympy.core.numbers import (mpf_norm, seterr,
    Integer, I, pi, comp, Rational, E, nan,
    oo, AlgebraicNumber, Number, Float, zoo, equal_valued,
    int_valued, all_close)
# 从 sympy.core.intfunc 模块导入 igcd, igcdex, igcd2, igcd_lehmer, ilcm, integer_nthroot, isqrt, integer_log, mod_inverse 函数
from sympy.core.intfunc import (igcd, igcdex, igcd2, igcd_lehmer,
    ilcm, integer_nthroot, isqrt, integer_log, mod_inverse)
# 从 sympy.core.power 模块导入 Pow 类
from sympy.core.power import Pow
# 从 sympy.core.relational 模块导入 Ge, Gt, Le, Lt 类
from sympy.core.relational import Ge, Gt, Le, Lt
# 从 sympy.core.singleton 模块导入 S 类
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Dummy, Symbol 类
from sympy.core.symbol import Dummy, Symbol
# 从 sympy.core.sympify 模块导入 sympify 函数
from sympy.core.sympify import sympify
# 从 sympy.functions.combinatorial.factorials 模块导入 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 从 sympy.functions.elementary.integers 模块导入 floor 函数
from sympy.functions.elementary.integers import floor
# 从 sympy.functions.combinatorial.numbers 模块导入 fibonacci 函数
from sympy.functions.combinatorial.numbers import fibonacci
# 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt, cbrt 函数
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
# 从 sympy.functions.elementary.trigonometric 模块导入 cos, sin 函数
from sympy.functions.elementary.trigonometric import cos, sin
# 从 sympy.polys.domains.realfield 模块导入 RealField 类
from sympy.polys.domains.realfield import RealField
# 从 sympy.printing.latex 模块导入 latex 函数
from sympy.printing.latex import latex
# 从 sympy.printing.repr 模块导入 srepr 函数
from sympy.printing.repr import srepr
# 从 sympy.simplify 模块导入 simplify 函数
from sympy.simplify import simplify
# 从 sympy.polys.domains.groundtypes 模块导入 PythonRational 类
from sympy.polys.domains.groundtypes import PythonRational
# 从 sympy.utilities.decorator 模块导入 conserve_mpmath_dps 装饰器
from sympy.utilities.decorator import conserve_mpmath_dps
# 从 sympy.utilities.iterables 模块导入 permutations 函数
from sympy.utilities.iterables import permutations
# 从 sympy.testing.pytest 模块导入 XFAIL, raises, _both_exp_pow 函数/装饰器
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
# 从 sympy 模块导入 Add 类
from sympy import Add

# 从 mpmath 模块导入 mpf 函数
from mpmath import mpf
# 导入 mpmath 模块
import mpmath
# 从 sympy.core 模块导入 numbers 模块
from sympy.core import numbers

# 创建一个名为 t 的符号，不是实数
t = Symbol('t', real=False)

# 设置 _ninf 为负无穷大的浮点数表示
_ninf = float(-oo)
# 设置 _inf 为正无穷大的浮点数表示
_inf = float(oo)


def same_and_same_prec(a, b):
    # 比较两个 Float 对象是否相等且精度相同
    return a == b and a._prec == b._prec


def test_seterr():
    # 设置错误处理，允许除法错误
    seterr(divide=True)
    # 断言除以零会引发 ValueError 异常
    raises(ValueError, lambda: S.Zero/S.Zero)
    # 恢复除法错误处理，断言零除以零得到 NaN
    seterr(divide=False)
    assert S.Zero / S.Zero is S.NaN


def test_mod():
    # 定义有理数 x, y, z
    x = S.Half
    y = Rational(3, 4)
    z = Rational(5, 18043)

    # 各种数值的模运算断言
    assert x % x == 0
    assert x % y == S.Half
    assert x % z == Rational(3, 36086)
    assert y % x == Rational(1, 4)
    assert y % y == 0
    assert y % z == Rational(9, 72172)
    assert z % x == Rational(5, 18043)
    assert z % y == Rational(5, 18043)
    assert z % z == 0

    # 定义浮点数 a
    a = Float(2.6)

    # 浮点数和浮点数的模运算断言
    assert (a % .2) == 0.0
    assert (a % 2).round(15) == 0.6
    assert (a % 0.5).round(15) == 0.1

    # 定义一个无穷大符号 p
    p = Symbol('p', infinite=True)

    # 各种无穷大数值的模运算断言
    assert oo % oo is nan
    assert zoo % oo is nan
    assert 5 % oo is nan
    assert p % 5 is nan

    # 在这两个测试中，如果 m 的精度与 ans 的精度不匹配，
    # 那么现在所做的更改可能会导致精度降低的答案。
    r = Rational(500, 41)
    f = Float('.36', 3)
    m = r % f
    ans = Float(r % Rational(f), 3)
    assert m == ans and m._prec == ans._prec
    f = Float('8.36', 3)
    m = f % r
    # 创建一个 Float 类型对象，表示有理数 f 对 r 取模的结果，精度保留到小数点后三位
    ans = Float(Rational(f) % r, 3)
    # 断言 m 等于 ans 且它们的精度相同
    assert m == ans and m._prec == ans._prec

    # 初始化一个 S.Zero 的对象 s，表示零值
    s = S.Zero

    # 断言 s 对 1.0 取模的结果为浮点数 0.0
    assert s % float(1) == 0.0

    # 这些数字可以精确表示，无需四舍五入。
    # 断言有理数 3/4 对浮点数 1.1 取模的结果为 0.75
    assert Rational(3, 4) % Float(1.1) == 0.75
    # 断言浮点数 1.5 对有理数 5/4 取模的结果为 0.25
    assert Float(1.5) % Rational(5, 4) == 0.25
    # 断言有理数 5/4 对浮点数 '1.5' 取模的结果为 0.25
    assert Rational(5, 4).__rmod__(Float('1.5')) == 0.25
    # 断言浮点数 '1.5' 对浮点数 '2.75' 取模的结果为浮点数 '1.25'
    assert Float('1.5').__rmod__(Float('2.75')) == Float('1.25')
    # 断言浮点数 2.75 对浮点数 '1.5' 取模的结果为浮点数 '1.25'
    assert 2.75 % Float('1.5') == Float('1.25')

    # 初始化两个整数对象 a 和 b 分别为 7 和 4
    a = Integer(7)
    b = Integer(4)

    # 断言 a 对 b 取模的结果是整数类型 Integer
    assert type(a % b) == Integer
    # 断言 a 对 b 取模的结果为整数 3
    assert a % b == Integer(3)
    # 断言整数 1 对有理数 2/3 取模的结果为有理数 1/3
    assert Integer(1) % Rational(2, 3) == Rational(1, 3)
    # 断言有理数 7/5 对整数 1 取模的结果为有理数 2/5
    assert Rational(7, 5) % Integer(1) == Rational(2, 5)
    # 断言整数 2 对浮点数 1.5 取模的结果为浮点数 0.5
    assert Integer(2) % 1.5 == 0.5

    # 断言整数 3 对整数 10 取模的结果为整数 1
    assert Integer(3).__rmod__(Integer(10)) == Integer(1)
    # 断言整数 10 对整数 4 取模的结果为整数 2
    assert Integer(10) % 4 == Integer(2)
    # 断言整数 15 对整数 4 取模的结果为整数 3
    assert 15 % Integer(4) == Integer(3)
# 定义测试函数 test_divmod
def test_divmod():
    # 创建符号变量 x
    x = Symbol("x")
    # 断言 divmod(12, 8) 的结果为 Tuple(1, 4)
    assert divmod(S(12), S(8)) == Tuple(1, 4)
    # 断言 divmod(-12, 8) 的结果为 Tuple(-2, 4)
    assert divmod(-S(12), S(8)) == Tuple(-2, 4)
    # 断言 divmod(0, 1) 的结果为 Tuple(0, 0)
    assert divmod(S.Zero, S.One) == Tuple(0, 0)
    # 测试除零异常，期望抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: divmod(S.Zero, S.Zero))
    # 测试除零异常，期望抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: divmod(S.One, S.Zero))
    # 断言 divmod(12, 8) 的结果为 Tuple(1, 4)
    assert divmod(S(12), 8) == Tuple(1, 4)
    # 断言 divmod(12, 8) 的结果为 Tuple(1, 4)
    assert divmod(12, S(8)) == Tuple(1, 4)
    # 断言 S(1024)//x 的结果与 1024//x 的结果都等于 floor(1024/x)
    assert S(1024)//x == 1024//x == floor(1024/x)

    # 以下为多个复杂数据类型和分数的除法取余运算的断言
    assert divmod(S("2"), S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), S("2")) == Tuple(S("0"), S("3/2"))
    assert divmod(S("2"), S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), S("2")) == Tuple(S("1"), S("1.5"))
    assert divmod(S("2"), S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), S("2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("2"), S("1/10")) == Tuple(S("20"), S("0"))
    assert divmod(S("2"), S(".1"))[0] == 19
    assert divmod(S("0.1"), S("2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("2"), 2) == Tuple(S("1"), S("0"))
    assert divmod(2, S("2")) == Tuple(S("1"), S("0"))
    assert divmod(S("2"), 1.5) == Tuple(S("1"), S("0.5"))
    assert divmod(1.5, S("2")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("3/2"), S("3.5")) == Tuple(S("0"), S(3/2))
    assert divmod(S("3.5"), S("3/2")) == Tuple(S("2"), S("0.5"))
    assert divmod(S("3/2"), S("1/3")) == Tuple(S("4"), S("1/6"))
    assert divmod(S("1/3"), S("3/2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("3/2"), S("0.1"))[0] == 14
    assert divmod(S("0.1"), S("3/2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3/2"), 2) == Tuple(S("0"), S("3/2"))
    assert divmod(2, S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), 1.5) == Tuple(S("1"), S("0."))
    assert divmod(1.5, S("3/2")) == Tuple(S("1"), S("0."))
    assert divmod(S("3/2"), 0.3) == Tuple(S("5"), S("0."))
    assert divmod(0.3, S("3/2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("1/3"), S("3.5")) == (0, 1/3)
    assert divmod(S("3.5"), S("0.1")) == Tuple(S("35"), S("0."))
    assert divmod(S("0.1"), S("3.5")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3.5"), 2) == Tuple(S("1"), S("1.5"))
    assert divmod(2, S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), 1.5) == Tuple(S("2"), S("0.5"))
    assert divmod(1.5, S("3.5")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("3.5")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("0.1"), S("1/3")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("1/3"), 2) == Tuple(S("0"), S("1/3"))
    assert divmod(2, S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), 1.5) == (0, 1/3)
    assert divmod(0.3, S("1/3")) == (0, 0.3)
    assert divmod(S("0.1"), 2) == (0, 0.1)
    assert divmod(2, S("0.1"))[0] == 19
    assert divmod(S("0.1"), 1.5) == (0, 0.1)
    assert divmod(1.5, S("0.1")) == Tuple(S("15"), S("0."))
    assert divmod(S("0.1"), 0.3) == Tuple(S("0"), S("0.1"))
    # 使用 SymPy 的 S 函数将字符串转换为符号对象，进行 divmod 操作
    assert str(divmod(S("2"), 0.3)) == '(6, 0.2)'
    assert str(divmod(S("3.5"), S("1/3"))) == '(10, 0.166666666666667)'
    assert str(divmod(S("3.5"), 0.3)) == '(11, 0.2)'
    assert str(divmod(S("1/3"), S("0.1"))) == '(3, 0.0333333333333333)'
    assert str(divmod(1.5, S("1/3"))) == '(4, 0.166666666666667)'
    assert str(divmod(S("1/3"), 0.3)) == '(1, 0.0333333333333333)'
    assert str(divmod(0.3, S("0.1"))) == '(2, 0.1)'

    # 使用 SymPy 的 S 函数处理负数，确保 divmod 的行为正确
    assert divmod(-3, S(2)) == (-2, 1)
    assert divmod(S(-3), S(2)) == (-2, 1)
    assert divmod(S(-3), 2) == (-2, 1)

    # 处理特殊情况：无穷大和非数字的 divmod 结果
    assert divmod(oo, 1) == (S.NaN, S.NaN)
    assert divmod(S.NaN, 1) == (S.NaN, S.NaN)
    assert divmod(1, S.NaN) == (S.NaN, S.NaN)

    # 使用 SymPy 的 S 函数和 Python 的 float 处理无穷大和负无穷大的 divmod 结果
    ans = [(-1, oo), (-1, oo), (0, 0), (0, 1), (0, 2)]
    OO = float('inf')
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, oo) for i in range(-2, 3)] == ans

    # 使用 SymPy 的 S 函数和 Python 的 float 处理负无穷大的 divmod 结果
    ans = [(0, -2), (0, -1), (0, 0), (-1, -oo), (-1, -oo)]
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, -oo) for i in range(-2, 3)] == ans
    assert [divmod(i, -OO) for i in range(-2, 3)] == ANS

    # 自定义 lambda 函数 dmod，确保 SymPy 的 divmod 返回整数商而不是浮点数
    dmod = lambda a, b: tuple([j if i else int(j) for i, j in enumerate(divmod(a, b))])
    # 遍历多种数据类型的 a 和 b 进行 divmod 操作，并与预期结果进行断言
    for a in (4, 4., 4.25, 0, 0., -4, -4., -4.25):
        for b in (2, 2., 2.5, -2, -2., -2.5):
            assert divmod(S(a), S(b)) == dmod(a, b)
def test_igcd():
    # 检查 igcd 函数对于各种输入的正确性
    assert igcd(0, 0) == 0
    assert igcd(0, 1) == 1
    assert igcd(1, 0) == 1
    assert igcd(0, 7) == 7
    assert igcd(7, 0) == 7
    assert igcd(7, 1) == 1
    assert igcd(1, 7) == 1
    assert igcd(-1, 0) == 1
    assert igcd(0, -1) == 1
    assert igcd(-1, -1) == 1
    assert igcd(-1, 7) == 1
    assert igcd(7, -1) == 1
    assert igcd(8, 2) == 2
    assert igcd(4, 8) == 4
    assert igcd(8, 16) == 8
    assert igcd(7, -3) == 1
    assert igcd(-7, 3) == 1
    assert igcd(-7, -3) == 1
    assert igcd(*[10, 20, 30]) == 10
    raises(TypeError, lambda: igcd())
    raises(TypeError, lambda: igcd(2))
    raises(ValueError, lambda: igcd(0, None))
    raises(ValueError, lambda: igcd(1, 2.2))
    for args in permutations((45.1, 1, 30)):
        raises(ValueError, lambda: igcd(*args))
    for args in permutations((1, 2, None)):
        raises(ValueError, lambda: igcd(*args))


def test_igcd_lehmer():
    # 检查 igcd_lehmer 函数的不同情况下的正确性
    a, b = fibonacci(10001), fibonacci(10000)
    # len(str(a)) == 2090
    # small divisors, long Euclidean sequence
    assert igcd_lehmer(a, b) == 1
    c = fibonacci(100)
    assert igcd_lehmer(a*c, b*c) == c
    # big divisor
    assert igcd_lehmer(a, 10**1000) == 1
    # swapping argument
    assert igcd_lehmer(1, 2) == igcd_lehmer(2, 1)


def test_igcd2():
    # 检查 igcd2 函数在不同输入下的正确性
    # short loop
    assert igcd2(2**100 - 1, 2**99 - 1) == 1
    # Lehmer's algorithm
    a, b = int(fibonacci(10001)), int(fibonacci(10000))
    assert igcd2(a, b) == 1


def test_ilcm():
    # 检查 ilcm 函数对于各种输入的正确性
    assert ilcm(0, 0) == 0
    assert ilcm(1, 0) == 0
    assert ilcm(0, 1) == 0
    assert ilcm(1, 1) == 1
    assert ilcm(2, 1) == 2
    assert ilcm(8, 2) == 8
    assert ilcm(8, 6) == 24
    assert ilcm(8, 7) == 56
    assert ilcm(*[10, 20, 30]) == 60
    raises(ValueError, lambda: ilcm(8.1, 7))
    raises(ValueError, lambda: ilcm(8, 7.1))
    raises(TypeError, lambda: ilcm(8))


def test_igcdex():
    # 检查 igcdex 函数在不同输入下的正确性
    assert igcdex(2, 3) == (-1, 1, 1)
    assert igcdex(10, 12) == (-1, 1, 2)
    assert igcdex(100, 2004) == (-20, 1, 4)
    assert igcdex(0, 0) == (0, 1, 0)
    assert igcdex(1, 0) == (1, 0, 1)


def _strictly_equal(a, b):
    return (a.p, a.q, type(a.p), type(a.q)) == \
           (b.p, b.q, type(b.p), type(b.q))


def _test_rational_new(cls):
    """
    Tests that are common between Integer and Rational.
    """
    # 检查 Integer 和 Rational 构造函数的通用测试
    assert cls(0) is S.Zero
    assert cls(1) is S.One
    assert cls(-1) is S.NegativeOne
    # These look odd, but are similar to int():
    assert cls('1') is S.One
    assert cls('-1') is S.NegativeOne

    i = Integer(10)
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls(int(10)))
    assert _strictly_equal(i, cls(i))

    raises(TypeError, lambda: cls(Symbol('x')))


def test_Integer_new():
    """
    Test for Integer constructor
    """
    # 检查 Integer 构造函数的不同情况下的正确性
    _test_rational_new(Integer)

    assert _strictly_equal(Integer(0.9), S.Zero)
    assert _strictly_equal(Integer(10.5), Integer(10))
    # 使用 raises 函数验证在执行 lambda 函数时是否会引发 ValueError 异常，lambda 函数尝试将字符串 "10.5" 转换为 Integer 类型
    raises(ValueError, lambda: Integer("10.5"))
    
    # 使用断言语句确保将包含很多 '9' 的字符串与 Rational 类型相加后转换为 Integer 类型时结果等于 1
    assert Integer(Rational('1.' + '9'*20)) == 1
# 定义测试函数 test_Rational_new
def test_Rational_new():
    """"
    Test for Rational constructor
    """
    # 调用 _test_rational_new 函数，测试 Rational 构造函数
    _test_rational_new(Rational)

    # 定义并测试 n1 变量为 S.Half，确保与 Rational(Integer(1), 2) 相等
    n1 = S.Half
    assert n1 == Rational(Integer(1), 2)
    # 测试不同参数形式的 Rational 构造方法
    assert n1 == Rational(Integer(1), Integer(2))
    assert n1 == Rational(1, Integer(2))
    assert n1 == Rational(S.Half)
    assert 1 == Rational(n1, n1)
    assert Rational(3, 2) == Rational(S.Half, Rational(1, 3))
    assert Rational(3, 1) == Rational(1, Rational(1, 3))
    
    # 定义 n3_4 为 Rational(3, 4)，并进行多种方式的比较和断言
    n3_4 = Rational(3, 4)
    assert Rational('3/4') == n3_4
    assert -Rational('-3/4') == n3_4
    assert Rational('.76').limit_denominator(4) == n3_4
    assert Rational(19, 25).limit_denominator(4) == n3_4
    assert Rational('19/25').limit_denominator(4) == n3_4
    assert Rational(1.0, 3) == Rational(1, 3)
    assert Rational(1, 3.0) == Rational(1, 3)
    assert Rational(Float(0.5)) == S.Half
    assert Rational('1e2/1e-2') == Rational(10000)
    assert Rational('1 234') == Rational(1234)
    assert Rational('1/1 234') == Rational(1, 1234)
    assert Rational(-1, 0) is S.ComplexInfinity
    assert Rational(1, 0) is S.ComplexInfinity

    # 确保在 Float 上 Rational 不会失去精度
    assert Rational(pi.evalf(100)).evalf(100) == pi.evalf(100)
    
    # 检查错误类型是否会被正确引发
    raises(TypeError, lambda: Rational('3**3'))
    raises(TypeError, lambda: Rational('1/2 + 2/3'))

    # 处理 fractions.Fraction 实例的情况（如果模块可用）
    try:
        import fractions
        assert Rational(fractions.Fraction(1, 2)) == S.Half
    except ImportError:
        pass

    # 断言使用 PythonRational 类的 Rational 实例等于 Rational(1, 3)
    assert Rational(PythonRational(2, 6)) == Rational(1, 3)

    # 检查 Rational 对象的构造方式和 gcd 参数
    assert Rational(2, 4, gcd=1).q == 4
    n = Rational(2, -4, gcd=1)
    assert n.q == 4
    assert n.p == -2

# 定义测试函数 test_issue_24543
def test_issue_24543():
    for p in ('1.5', 1.5, 2):
        for q in ('1.5', 1.5, 2):
            # 确保 Rational(p, q) 与 Rational('%s/%s'%(p,q)) 的分子和分母相同
            assert Rational(p, q).as_numer_denom() == Rational('%s/%s'%(p,q)).as_numer_denom()

    # 确保 Rational('0.5', '100') 等于 Rational(1, 200)
    assert Rational('0.5', '100') == Rational(1, 200)

# 定义测试函数 test_Number_new
def test_Number_new():
    """"
    Test for Number constructor
    """
    # 数字和字符串的预期行为
    assert Number(1) is S.One
    assert Number(2).__class__ is Integer
    assert Number(-622).__class__ is Integer
    assert Number(5, 3).__class__ is Rational
    assert Number(5.3).__class__ is Float
    assert Number('1') is S.One
    assert Number('2').__class__ is Integer
    assert Number('-622').__class__ is Integer
    assert Number('5/3').__class__ is Rational
    assert Number('5.3').__class__ is Float
    # 确保输入 'cos' 引发 ValueError，输入 cos 引发 TypeError
    raises(ValueError, lambda: Number('cos'))
    raises(TypeError, lambda: Number(cos))
    a = Rational(3, 5)
    assert Number(a) is a  # 确保 Number 的幂等性
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        assert Number(i) is a, (i, Number(i), a)

# 定义测试函数 test_Number_cmp
def test_Number_cmp():
    n1 = Number(1)
    n2 = Number(2)
    n3 = Number(-3)

    # 比较不同 Number 实例之间的大小关系
    assert n1 < n2
    assert n1 <= n2
    assert n3 < n1
    assert n2 > n3
    assert n2 >= n3

    # 确保与 S.NaN 比较时引发 TypeError
    raises(TypeError, lambda: n1 < S.NaN)
    # 检查是否会引发 TypeError 异常，因为 n1 <= S.NaN 操作中的 S.NaN 是无效的数学运算
    raises(TypeError, lambda: n1 <= S.NaN)
    # 检查是否会引发 TypeError 异常，因为 n1 > S.NaN 操作中的 S.NaN 是无效的数学运算
    raises(TypeError, lambda: n1 > S.NaN)
    # 检查是否会引发 TypeError 异常，因为 n1 >= S.NaN 操作中的 S.NaN 是无效的数学运算
    raises(TypeError, lambda: n1 >= S.NaN)
# 定义了一个测试函数 test_Rational_cmp，用于测试 Rational 类的比较方法
def test_Rational_cmp():
    # 创建 Rational 对象，分别初始化不同的分数
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n6 = Rational(1)
    n7 = Rational(3)
    n8 = Rational(-3)

    # 断言：小于关系的比较
    assert n8 < n5  # -3 < 0
    assert n5 < n6  # 0 < 1
    assert n6 < n7  # 1 < 3
    assert n8 < n7  # -3 < 3
    assert n7 > n8  # 3 > -3
    assert (n1 + 1)**n2 < 2  # (1/4 + 1)**(1/3) < 2
    assert ((n1 + n6)/n7) < 1  # (1/4 + 1)/3 < 1

    # 断言：小于关系的比较
    assert n4 < n3  # (2/-4) < (2/4), 即 -1/2 < 1/2
    assert n2 < n3  # 1/3 < 1/2
    assert n1 < n2  # 1/4 < 1/3
    assert n3 > n1  # 1/2 > 1/4
    assert not n3 < n1  # 1/2 不小于 1/4
    assert not (Rational(-1) > 0)  # -1 不大于 0
    assert Rational(-1) < 0  # -1 小于 0

    # 断言：抛出 TypeError 异常
    raises(TypeError, lambda: n1 < S.NaN)
    raises(TypeError, lambda: n1 <= S.NaN)
    raises(TypeError, lambda: n1 > S.NaN)
    raises(TypeError, lambda: n1 >= S.NaN)


# 定义了一个测试函数 test_Float
def test_Float():
    # 定义一个辅助函数 eq，用于比较浮点数 a 和 b 的接近程度
    def eq(a, b):
        t = Float("1.0E-15")
        return (-t < a - b < t)

    # zeros 包含多种零值的组合形式
    zeros = (0, S.Zero, 0., Float(0))
    # 使用 permutations 对 zeros 中的元素进行排列组合
    for i, j in permutations(zeros[:-1], 2):
        assert i == j
    for i, j in permutations(zeros[-2:], 2):
        assert i == j
    for z in zeros:
        assert z in zeros
    assert S.Zero.is_zero  # 断言：S.Zero 是零

    # 计算 2 的 3 次方，并使用 eq 函数比较其与 8 的接近程度
    a = Float(2) ** Float(3)
    assert eq(a.evalf(), Float(8))
    # 计算 pi 的倒数，并使用 eq 函数比较其与 0.31830988618379067 的接近程度
    assert eq((pi ** -1).evalf(), Float("0.31830988618379067"))
    # 计算 2 的 4 次方，并使用 eq 函数比较其与 16 的接近程度
    a = Float(2) ** Float(4)
    assert eq(a.evalf(), Float(16))
    # 断言：S(.3) 等于 S(.5) 应为 False
    assert (S(.3) == S(.5)) is False

    # 定义不同形式的浮点数，并进行比较
    mpf = (0, 5404319552844595, -52, 53)
    x_str = Float((0, '13333333333333', -52, 53))
    x_0xstr = Float((0, '0x13333333333333', -52, 53))
    x2_str = Float((0, '26666666666666', -53, 54))
    x_hex = Float((0, int(0x13333333333333), -52, 53))
    x_dec = Float(mpf)
    assert x_str == x_0xstr == x_hex == x_dec == Float(1.2)
    # 断言：x2_str 的内部表示应与 mpf 相等
    # x2_str 被输入时略微格式不正确，但通过规范化解决了这个问题
    assert Float(1.2)._mpf_ == mpf
    assert x2_str._mpf_ == mpf

    # 断言：使用不完整签名的浮点数表示为特殊值
    assert Float((0, int(0), -123, -1)) is S.NaN
    assert Float((0, int(0), -456, -2)) is S.Infinity
    assert Float((1, int(0), -789, -3)) is S.NegativeInfinity
    # 断言：如果不提供完整签名，应为普通浮点数值
    assert Float((0, int(0), -123)) == Float(0)
    assert Float((0, int(0), -456)) == Float(0)
    assert Float((1, int(0), -789)) == Float(0)

    # 断言：抛出 ValueError 异常
    raises(ValueError, lambda: Float((0, 7, 1, 3), ''))

    # 断言：Float('0.0') 的一些属性
    assert Float('0.0').is_finite is True
    assert Float('0.0').is_negative is False
    assert Float('0.0').is_positive is False
    assert Float('0.0').is_infinite is False
    assert Float('0.0').is_zero is True

    # 断言：有理数属性
    assert Float(1).is_integer is None
    assert Float(1).is_rational is None
    assert Float(1).is_irrational is None
    assert sqrt(2).n(15).is_rational is None
    assert sqrt(2).n(15).is_irrational is None
    # 不自动评估表达式的函数定义
    def teq(a):
        # 断言，a 的数值化评估结果不等于 a 自身
        assert (a.evalf() == a) is False
        # 断言，a 的数值化评估结果不等于 a 自身
        assert (a.evalf() != a) is True
        # 断言，a 等于其数值化评估结果不成立
        assert (a == a.evalf()) is False
        # 断言，a 不等于其数值化评估结果成立
        assert (a != a.evalf()) is True

    # 使用 teq 函数测试常数 pi
    teq(pi)
    # 使用 teq 函数测试常数 2*pi
    teq(2*pi)
    # 使用 teq 函数测试 cos(0.1) 的评估，但不自动进行评估
    teq(cos(0.1, evaluate=False))

    # 长整数赋值
    i = 12345678901234567890
    # 断言，使用相同精度和值的 Float 对象
    assert same_and_same_prec(Float(12, ''), Float('12', ''))
    # 断言，使用相同精度和值的 Float 对象
    assert same_and_same_prec(Float(Integer(i), ''), Float(i, ''))
    # 断言，使用相同精度和值的 Float 对象
    assert same_and_same_prec(Float(i, ''), Float(str(i), 20))
    # 断言，使用相同精度和值的 Float 对象
    assert same_and_same_prec(Float(str(i)), Float(i, ''))
    # 断言，使用相同精度和值的 Float 对象
    assert same_and_same_prec(Float(i), Float(i, ''))

    # 不精确的浮点数（重复的二进制，分母不是 2 的倍数）
    # 不能具有大于 15 的精度
    assert Float(.125, 22)._prec == 76
    assert Float(2.0, 22)._prec == 76
    # 只有默认精度相等，即使是可精确表示的浮点数
    assert Float(.125, 22) != .125
    #assert Float(2.0, 22) == 2
    # 测试浮点数的精确值
    assert float(Float('.12500000000000001', '')) == .125
    # 断言，浮点数不支持的值引发 ValueError
    raises(ValueError, lambda: Float(.12500000000000001, ''))

    # 允许使用空格
    assert Float('123 456.123 456') == Float('123456.123456')
    assert Integer('123 456') == Integer('123456')
    assert Rational('123 456.123 456') == Rational('123456.123456')
    assert Float(' .3e2') == Float('0.3e2')
    # 但是，空格在数字之间必须严格按照下划线处理：只允许一个
    raises(ValueError, lambda: Float('1  2'))

    # 允许数字之间使用下划线
    assert Float('1_23.4_56') == Float('123.456')
    # 在所有情况下都不允许（基于 Python 3.6）
    raises(ValueError, lambda: Float('1_'))
    raises(ValueError, lambda: Float('1__2'))
    raises(ValueError, lambda: Float('_1'))
    raises(ValueError, lambda: Float('_inf'))

    # 允许自动精度检测
    assert Float('.1', '') == Float(.1, 1)
    assert Float('.125', '') == Float(.125, 3)
    assert Float('.100', '') == Float(.1, 3)
    assert Float('2.0', '') == Float('2', 2)

    raises(ValueError, lambda: Float("12.3d-4", ""))
    raises(ValueError, lambda: Float(12.3, ""))
    raises(ValueError, lambda: Float('.'))
    raises(ValueError, lambda: Float('-.'))

    zero = Float('0.0')
    assert Float('-0') == zero
    assert Float('.0') == zero
    assert Float('-.0') == zero
    assert Float('-0.0') == zero
    assert Float(0.0) == zero
    assert Float(0) == zero
    assert Float(0, '') == Float('0', '')
    assert Float(1) == Float(1.0)
    assert Float(S.Zero) == zero
    assert Float(S.One) == Float(1.0)

    assert Float(decimal.Decimal('0.1'), 3) == Float('.1', 3)
    assert Float(decimal.Decimal('nan')) is S.NaN
    assert Float(decimal.Decimal('Infinity')) is S.Infinity
    assert Float(decimal.Decimal('-Infinity')) is S.NegativeInfinity

    assert '{:.3f}'.format(Float(4.236622)) == '4.237'
    assert '{:.35f}'.format(Float(pi.n(40), 40)) == \
        '3.14159265358979323846264338327950288'

    # unicode
    # 断言，验证两个浮点数对象相等
    assert Float('0.73908513321516064100000000') == \
        Float('0.73908513321516064100000000')
    
    # 断言，验证两个指定精度的浮点数对象相等
    assert Float('0.73908513321516064100000000', 28) == \
        Float('0.73908513321516064100000000', 28)
    
    # binary precision
    # 创建四个不同精度的浮点数对象
    # 注意：十进制值 0.1 无法精确表示为二进制分数
    a = Float(S.One/10, dps=15)
    b = Float(S.One/10, dps=16)
    p = Float(S.One/10, precision=53)
    q = Float(S.One/10, precision=54)
    # 断言，验证两个对象内部的 MPF 值是否相等
    assert a._mpf_ == p._mpf_
    assert not a._mpf_ == q._mpf_
    assert not b._mpf_ == q._mpf_
    
    # Precision specifying errors
    # 验证在指定精度和小数位数参数时，会引发 ValueError 异常
    raises(ValueError, lambda: Float("1.23", dps=3, precision=10))
    raises(ValueError, lambda: Float("1.23", dps="", precision=10))
    raises(ValueError, lambda: Float("1.23", dps=3, precision=""))
    raises(ValueError, lambda: Float("1.23", dps="", precision=""))
    
    # from NumberSymbol
    # 断言，验证从数学常数转换而来的浮点数对象和数学库计算结果相同且精度相同
    assert same_and_same_prec(Float(pi, 32), pi.evalf(32))
    assert same_and_same_prec(Float(Catalan), Catalan.evalf())
    
    # oo and nan
    # 测试创建特殊浮点数对象（无穷大和非数）
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        # 断言，验证字符串转换的特殊浮点数和预期的特殊浮点数对象相等
        assert Float(i) is a
def test_zero_not_false():
    # 检查 SymPy 中的 S 对象，确保 0.0 不等于 S.false
    assert (S(0.0) == S.false) is False
    # 检查 SymPy 中的 S 对象，确保 S.false 不等于 0.0
    assert (S.false == S(0.0)) is False
    # 检查 SymPy 中的 S 对象，确保 0 不等于 S.false
    assert (S(0) == S.false) is False
    # 检查 SymPy 中的 S 对象，确保 S.false 不等于 0
    assert (S.false == S(0)) is False


@conserve_mpmath_dps
def test_float_mpf():
    import mpmath
    # 设置 mpmath 精度为 100
    mpmath.mp.dps = 100
    # 计算 mpmath 中的 pi，并赋值给 mp_pi
    mp_pi = mpmath.pi()

    # 检查浮点数对象 Float 的比较
    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)

    # 恢复 mpmath 精度为默认值 15
    mpmath.mp.dps = 15

    # 再次检查浮点数对象 Float 的比较
    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)


def test_Float_RealElement():
    # 使用 RealField 设置精度为 100，并将 pi 的数值赋给 repi
    repi = RealField(dps=100)(pi.evalf(100))
    # 确保 Float 对象保持从结果中完全保留精度，需要手动传入精度
    assert Float(repi, 100) == pi.evalf(100)


def test_Float_default_to_highprec_from_str():
    # 将 pi 的数值计算结果转换为字符串
    s = str(pi.evalf(128))
    # 测试通过字符串创建 Float 对象的相等性和精度一致性
    assert same_and_same_prec(Float(s), Float(s, ''))


def test_Float_eval():
    # 创建一个浮点数对象 a，并检查其平方是否仍然为浮点数对象
    a = Float(3.2)
    assert (a**2).is_Float


def test_Float_issue_2107():
    # 创建浮点数对象 a 和 b，确保浮点数运算的准确性
    a = Float(0.1, 10)
    b = Float("0.1", 10)

    # 检查浮点数对象 a 和 b 的加减法结果
    assert a - a == 0
    assert a + (-a) == 0
    assert S.Zero + a - a == 0
    assert S.Zero + a + (-a) == 0

    assert b - b == 0
    assert b + (-b) == 0
    assert S.Zero + b - b == 0
    assert S.Zero + b + (-b) == 0


def test_issue_14289():
    from sympy.polys.numberfields import to_number_field

    # 创建一个数值并将其转换为数域对象，验证转换的正确性
    a = 1 - sqrt(2)
    b = to_number_field(a)
    assert b.as_expr() == a
    assert b.minpoly(a).expand() == 0


def test_Float_from_tuple():
    # 使用元组创建浮点数对象 a 和 b，并验证它们的相等性
    a = Float((0, '1L', 0, 1))
    b = Float((0, '1', 0, 1))
    assert a == b


def test_Infinity():
    # 检查 oo（无穷大）的各种数学运算和关系
    assert oo != 1
    assert 1*oo is oo
    assert 1 != oo
    assert oo != -oo
    assert oo != Symbol("x")**3
    assert oo + 1 is oo
    assert 2 + oo is oo
    assert 3*oo + 2 is oo
    assert S.Half**oo == 0
    assert S.Half**(-oo) is oo
    assert -oo*3 is -oo
    assert oo + oo is oo
    assert -oo + oo*(-5) is -oo
    assert 1/oo == 0
    assert 1/(-oo) == 0
    assert 8/oo == 0
    assert oo % 2 is nan
    assert 2 % oo is nan
    assert oo/oo is nan
    assert oo/-oo is nan
    assert -oo/oo is nan
    assert -oo/-oo is nan
    assert oo - oo is nan
    assert oo - -oo is oo
    assert -oo - oo is -oo
    assert -oo - -oo is nan
    assert oo + -oo is nan
    assert -oo + oo is nan
    assert oo + oo is oo
    assert -oo + oo is nan
    assert oo + -oo is nan
    assert -oo + -oo is -oo
    assert oo*oo is oo
    assert -oo*oo is -oo
    assert oo*-oo is -oo
    assert -oo*-oo is oo
    assert oo/0 is oo
    assert -oo/0 is -oo
    assert 0/oo == 0
    assert 0/-oo == 0
    assert oo*0 is nan
    assert -oo*0 is nan
    assert 0*oo is nan
    assert 0*-oo is nan
    assert oo + 0 is oo
    assert -oo + 0 is -oo
    assert 0 + oo is oo
    assert 0 + -oo is -oo
    assert oo - 0 is oo
    assert -oo - 0 is -oo
    assert 0 - oo is -oo
    assert 0 - -oo is oo
    assert oo/2 is oo
    assert -oo/2 is -oo
    assert oo/-2 is -oo
    # 断言负无穷除以负二是正无穷
    assert -oo/-2 is oo
    # 断言正无穷乘以二是正无穷
    assert oo*2 is oo
    # 断言负无穷乘以二是负无穷
    assert -oo*2 is -oo
    # 断言正无穷乘以负二是负无穷
    assert oo*-2 is -oo
    # 断言二除以正无穷是零
    assert 2/oo == 0
    # 断言二除以负无穷是零
    assert 2/-oo == 0
    # 断言负二除以正无穷是零
    assert -2/oo == 0
    # 断言负二除以负无穷是零
    assert -2/-oo == 0
    # 断言二乘以正无穷是正无穷
    assert 2*oo is oo
    # 断言二乘以负无穷是负无穷
    assert 2*-oo is -oo
    # 断言负二乘以正无穷是负无穷
    assert -2*oo is -oo
    # 断言负二乘以负无穷是正无穷
    assert -2*-oo is oo
    # 断言二加正无穷是正无穷
    assert 2 + oo is oo
    # 断言二减去正无穷是负无穷
    assert 2 - oo is -oo
    # 断言负二加正无穷是正无穷
    assert -2 + oo is oo
    # 断言负二减去正无穷是负无穷
    assert -2 - oo is -oo
    # 断言二加负无穷是负无穷
    assert 2 + -oo is -oo
    # 断言二减去负无穷是正无穷
    assert 2 - -oo is oo
    # 断言负二加负无穷是负无穷
    assert -2 + -oo is -oo
    # 断言负二减去负无穷是正无穷
    assert -2 - -oo is oo
    # 断言S(2)加正无穷是正无穷
    assert S(2) + oo is oo
    # 断言S(2)减去正无穷是负无穷
    assert S(2) - oo is -oo
    # 断言正无穷除以虚数I的结果是负无穷乘以虚数I
    assert oo/I == -oo*I
    # 断言负无穷除以虚数I的结果是正无穷乘以虚数I
    assert -oo/I == oo*I
    # 断言正无穷乘以浮点数1的结果是正无穷，并且其与正无穷相等
    assert oo*float(1) == _inf and (oo*float(1)) is oo
    # 断言负无穷乘以浮点数1的结果是负无穷，并且其与负无穷相等
    assert -oo*float(1) == _ninf and (-oo*float(1)) is -oo
    # 断言正无穷除以浮点数1的结果是正无穷，并且其与正无穷相等
    assert oo/float(1) == _inf and (oo/float(1)) is oo
    # 断言负无穷除以浮点数1的结果是负无穷，并且其与负无穷相等
    assert -oo/float(1) == _ninf and (-oo/float(1)) is -oo
    # 断言正无穷乘以浮点数-1的结果是负无穷，并且其与负无穷相等
    assert oo*float(-1) == _ninf and (oo*float(-1)) is -oo
    # 断言负无穷乘以浮点数-1的结果是正无穷，并且其与正无穷相等
    assert -oo*float(-1) == _inf and (-oo*float(-1)) is oo
    # 断言正无穷除以浮点数-1的结果是负无穷，并且其与负无穷相等
    assert oo/float(-1) == _ninf and (oo/float(-1)) is -oo
    # 断言负无穷除以浮点数-1的结果是正无穷，并且其与正无穷相等
    assert -oo/float(-1) == _inf and (-oo/float(-1)) is oo
    # 断言正无穷加上浮点数1的结果是正无穷，并且其与正无穷相等
    assert oo + float(1) == _inf and (oo + float(1)) is oo
    # 断言负无穷加上浮点数1的结果是负无穷，并且其与负无穷相等
    assert -oo + float(1) == _ninf and (-oo + float(1)) is -oo
    # 断言正无穷减去浮点数1的结果是正无穷，并且其与正无穷相等
    assert oo - float(1) == _inf and (oo - float(1)) is oo
    # 断言负无穷减去浮点数1的结果是负无穷，并且其与负无穷相等
    assert -oo - float(1) == _ninf and (-oo - float(1)) is -oo
    # 断言浮点数1乘以正无穷的结果是正无穷，并且其与正无穷相等
    assert float(1)*oo == _inf and (float(1)*oo) is oo
    # 断言浮点数1乘以负无穷的结果是负无穷，并且其与负无穷相等
    assert float(1)*-oo == _ninf and (float(1)*-oo) is -oo
    # 断言浮点数1除以正无穷的结果是零
    assert float(1)/oo == 0
    # 断言浮点数1除以负无穷的结果是零
    assert float(1)/-oo == 0
    # 断言浮点数-1乘以正无穷的结果是负无穷，并且其与负无穷相等
    assert float(-1)*oo == _ninf and (float(-1)*oo) is -oo
    # 断言浮点数-1乘以负无穷的结果是正无穷，并且其与正无穷相等
    assert float(-1)*-oo == _inf and (float(-1)*-oo) is oo
    # 断言浮点数-1除以正无穷的结果是零
    assert float(-1)/oo == 0
    # 断言浮点数-1除以负无穷的结果是零
    assert float(-1)/-oo == 0
    # 断言浮点数1加上正无穷的结果是正无穷，并且其与正无穷相等
    assert float(1) + oo is oo
    # 断言浮点数1加上负无穷的结果是负无穷，并且其与负无穷相等
    assert float(1) + -oo is -oo
    # 断言浮点数1减去正无穷的结果是负无穷，并且其与负无穷相等
    assert float(1) - oo is -oo
    # 断言浮点数1减去负无穷的结果是正无穷，并且其与正无穷相等
    assert float(1) - -oo is oo
    # 断言检查：nan 除以 S.One（sympy 中的 1）结果为 nan
    assert nan/S.One is nan
    
    # 断言检查：负无穷减去 S.One（sympy 中的 1）结果为负无穷
    assert -oo - S.One is -oo
# 测试无穷大与符号 x 的乘法，验证不同情况下的结果
def test_Infinity_2():
    x = Symbol('x')
    # 无穷大乘以 x 不等于无穷大
    assert oo*x != oo
    # 无穷大乘以 (π - 1) 仍然是无穷大
    assert oo*(pi - 1) is oo
    # 无穷大乘以 (1 - π) 是负无穷大
    assert oo*(1 - pi) is -oo

    # 负无穷大乘以 x 不等于负无穷大
    assert (-oo)*x != -oo
    # 负无穷大乘以 (π - 1) 是负无穷大
    assert (-oo)*(pi - 1) is -oo
    # 负无穷大乘以 (1 - π) 是无穷大
    assert (-oo)*(1 - pi) is oo

    # 负一的 NaN 次方是 NaN
    assert (-1)**S.NaN is S.NaN
    # 无穷大减去正无穷大是 NaN
    assert oo - _inf is S.NaN
    # 无穷大加上负无穷大是 NaN
    assert oo + _ninf is S.NaN
    # 无穷大乘以零是 NaN
    assert oo*0 is S.NaN
    # 无穷大除以正无穷大是 NaN
    assert oo/_inf is S.NaN
    # 无穷大除以负无穷大是 NaN
    assert oo/_ninf is S.NaN
    # 无穷大的 NaN 次方是 NaN
    assert oo**S.NaN is S.NaN
    # 负无穷大加上正无穷大是 NaN
    assert -oo + _inf is S.NaN
    # 负无穷大减去负无穷大是 NaN
    assert -oo - _ninf is S.NaN
    # 负无穷大乘以 NaN 是 NaN
    assert -oo*S.NaN is S.NaN
    # 负无穷大乘以零是 NaN
    assert -oo*0 is S.NaN
    # 负无穷大除以正无穷大是 NaN
    assert -oo/_inf is S.NaN
    # 负无穷大除以负无穷大是 NaN
    assert -oo/_ninf is S.NaN
    # 负无穷大除以 NaN 是 NaN
    assert -oo/S.NaN is S.NaN
    # 负无穷大的绝对值是无穷大
    assert abs(-oo) is oo
    # 所有形式的负无穷大的任意次方是 NaN
    assert all((-oo)**i is S.NaN for i in (oo, -oo, S.NaN))
    # 负无穷大的立方是负无穷大
    assert (-oo)**3 is -oo
    # 负无穷大的平方是无穷大
    assert (-oo)**2 is oo
    # 无穷大的复数无穷大的绝对值是无穷大
    assert abs(S.ComplexInfinity) is oo


# 测试无穷大与零的乘法，验证不同情况下的结果
def test_Mul_Infinity_Zero():
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan


# 测试除以零的情况，验证不同情况下的结果
def test_Div_By_Zero():
    # 1 除以零是正无穷大
    assert 1/S.Zero is zoo
    # 1.0 除以零是正无穷大
    assert 1/Float(0) is zoo
    # 0 除以零是 NaN
    assert 0/S.Zero is nan
    # 0.0 除以零是 NaN
    assert 0/Float(0) is nan
    # 零除以零是 NaN
    assert S.Zero/0 is nan
    # 0.0 除以零是 NaN
    assert Float(0)/0 is nan
    # -1 除以零是负无穷大
    assert -1/S.Zero is zoo
    # -1.0 除以零是负无穷大
    assert -1/Float(0) is zoo


# 测试无穷大的不等式，验证不同情况下的结果
@_both_exp_pow
def test_Infinity_inequations():
    # 无穷大大于π
    assert oo > pi
    # 无穷大不小于π
    assert not (oo < pi)
    # e 的 -3 次方小于无穷大
    assert exp(-3) < oo

    # 负无穷大大于π
    assert _inf > pi
    # 负无穷大不小于π
    assert not (_inf < pi)
    # e 的 -3 次方小于负无穷大
    assert exp(-3) < _inf

    # 无法将无穷大与复数单位 i 比较
    raises(TypeError, lambda: oo < I)
    raises(TypeError, lambda: oo <= I)
    raises(TypeError, lambda: oo > I)
    raises(TypeError, lambda: oo >= I)
    raises(TypeError, lambda: -oo < I)
    raises(TypeError, lambda: -oo <= I)
    raises(TypeError, lambda: -oo > I)
    raises(TypeError, lambda: -oo >= I)

    # 无法将复数单位 i 与无穷大比较
    raises(TypeError, lambda: I < oo)
    raises(TypeError, lambda: I <= oo)
    raises(TypeError, lambda: I > oo)
    raises(TypeError, lambda: I >= oo)
    raises(TypeError, lambda: I < -oo)
    raises(TypeError, lambda: I <= -oo)
    raises(TypeError, lambda: I > -oo)
    raises(TypeError, lambda: I >= -oo)

    # 无穷大大于负无穷大且不小于负无穷大
    assert oo > -oo and oo >= -oo
    # 无穷大不小于自身且不小于自身
    assert (oo < -oo) == False and (oo <= -oo) == False
    # 负无穷大小于无穷大且小于无穷大
    assert -oo < oo and -oo <= oo
    # 负无穷大不大于自身且不大于自身
    assert (-oo > oo) == False and (-oo >= oo) == False

    # 无穷大不小于自身，问题 7775
    assert (oo < oo) == False
    # 无穷大不大于自身
    assert (oo > oo) == False
    # 负无穷大不大于自身且不小于自身
    assert (-oo > -oo) == False and (-oo < -oo) == False
    # 无穷大不小于自身且不大于自身，负无穷大不小于自身且不大于自身
    assert oo >= oo and oo <= oo and -oo >= -oo and -oo <= -oo
    # 负无穷大不小于负无穷大
    assert (-oo < -_inf) ==  False
    # 无穷大不大于正无穷大
    assert (oo > _inf) == False
    # 负无穷大不小于负无穷大
    assert -oo >= -_inf
    # 无穷大不大于正无穷大
    assert oo <= _inf

    # 符号 x
    x = Symbol('x')
    # 符号 b 有限，实数，与无穷大和负无穷大的比较
    b = Symbol('b', finite=True, real=True)
    # 符号 x 小于无穷大，问题 7775
    assert (x < oo)
    # 断言：检查oo是否小于x，并验证其等效性为Lt(oo, x)
    # 断言：检查oo是否大于x，并验证其等效性为Gt(oo, x)
    assert (oo < x) == Lt(oo, x) and (oo > x) == Gt(oo, x)
    
    # 断言：检查oo是否小于等于x，并验证其等效性为Le(oo, x)
    # 断言：检查oo是否大于等于x，并验证其等效性为Ge(oo, x)
    assert (oo <= x) == Le(oo, x) and (oo >= x) == Ge(oo, x)
    
    # 断言：检查-oo是否小于x，并验证其等效性为Lt(-oo, x)
    # 断言：检查-oo是否大于x，并验证其等效性为Gt(-oo, x)
    assert (-oo < x) == Lt(-oo, x) and (-oo > x) == Gt(-oo, x)
    
    # 断言：检查-oo是否小于等于x，并验证其等效性为Le(-oo, x)
    # 断言：检查-oo是否大于等于x，并验证其等效性为Ge(-oo, x)
    assert (-oo <= x) == Le(-oo, x) and (-oo >= x) == Ge(-oo, x)
# 定义一个函数，用于测试 NaN（Not a Number）的特性和行为
def test_NaN():
    # 断言 NaN 等于自身（根据 IEEE 754 规范）
    assert nan is nan
    # 断言 NaN 不等于 1
    assert nan != 1
    # 断言 1 乘以 NaN 仍然是 NaN
    assert 1*nan is nan
    # 断言 1 不等于 NaN
    assert 1 != nan
    # 断言 -NaN 仍然是 NaN
    assert -nan is nan
    # 断言 无穷大不等于符号表达式 "x" 的三次方
    assert oo != Symbol("x")**3
    # 断言 2 加上 NaN 仍然是 NaN
    assert 2 + nan is nan
    # 断言 3 乘以 NaN 加上 2 仍然是 NaN
    assert 3*nan + 2 is nan
    # 断言 -NaN 乘以 3 仍然是 NaN
    assert -nan*3 is nan
    # 断言 NaN 加上 NaN 仍然是 NaN
    assert nan + nan is nan
    # 断言 -NaN 加上 NaN 乘以 -5 仍然是 NaN
    assert -nan + nan*(-5) is nan
    # 断言 8 除以 NaN 仍然是 NaN
    assert 8/nan is nan
    # 断言会引发 TypeError 异常，因为 NaN 不能与数字进行大小比较
    raises(TypeError, lambda: nan > 0)
    raises(TypeError, lambda: nan < 0)
    raises(TypeError, lambda: nan >= 0)
    raises(TypeError, lambda: nan <= 0)
    raises(TypeError, lambda: 0 < nan)
    raises(TypeError, lambda: 0 > nan)
    raises(TypeError, lambda: 0 <= nan)
    raises(TypeError, lambda: 0 >= nan)
    # 断言 NaN 的零次方等于 1（根据 IEEE 754 规范）
    assert nan**0 == 1
    # 断言 1 的 NaN 次方仍然是 NaN
    assert 1**nan is nan
    # 测试 Pow._eval_power 如何处理 NaN
    assert Pow(nan, 0, evaluate=False)**2 == 1
    # 遍历数值 n 的集合，验证加法、减法、除法运算中 NaN 的行为
    for n in (1, 1., S.One, S.NegativeOne, Float(1)):
        assert n + nan is nan
        assert n - nan is nan
        assert nan + n is nan
        assert nan - n is nan
        assert n/nan is nan
        assert nan/n is nan


# 定义一个函数，用于测试特殊数值（NaN、无穷大等）的属性和行为
def test_special_numbers():
    # 断言 S.NaN、S.Infinity、S.NegativeInfinity 是 Number 类型的实例
    assert isinstance(S.NaN, Number) is True
    assert isinstance(S.Infinity, Number) is True
    assert isinstance(S.NegativeInfinity, Number) is True

    # 断言 S.NaN、S.Infinity、S.NegativeInfinity 是数值类型
    assert S.NaN.is_number is True
    assert S.Infinity.is_number is True
    assert S.NegativeInfinity.is_number is True
    assert S.ComplexInfinity.is_number is True

    # 断言 S.NaN、S.Infinity、S.NegativeInfinity 不是 Rational 类型的实例
    assert isinstance(S.NaN, Rational) is False
    assert isinstance(S.Infinity, Rational) is False
    assert isinstance(S.NegativeInfinity, Rational) is False

    # 断言 S.NaN、S.Infinity、S.NegativeInfinity 不是有理数
    assert S.NaN.is_rational is not True
    assert S.Infinity.is_rational is not True
    assert S.NegativeInfinity.is_rational is not True


# 定义一个函数，用于测试整数的 n 次方根的计算
def test_powers():
    # 测试整数的平方根
    assert integer_nthroot(1, 2) == (1, True)
    assert integer_nthroot(1, 5) == (1, True)
    assert integer_nthroot(2, 1) == (2, True)
    assert integer_nthroot(2, 2) == (1, False)
    assert integer_nthroot(2, 5) == (1, False)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(123**25, 25) == (123, True)
    assert integer_nthroot(123**25 + 1, 25) == (123, False)
    assert integer_nthroot(123**25 - 1, 25) == (122, False)
    assert integer_nthroot(1, 1) == (1, True)
    assert integer_nthroot(0, 1) == (0, True)
    assert integer_nthroot(0, 3) == (0, True)
    assert integer_nthroot(10000, 1) == (10000, True)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(16, 2) == (4, True)
    assert integer_nthroot(26, 2) == (5, False)
    assert integer_nthroot(1234567**7, 7) == (1234567, True)
    assert integer_nthroot(1234567**7 + 1, 7) == (1234567, False)
    assert integer_nthroot(1234567**7 - 1, 7) == (1234566, False)
    
    # 测试大整数的 n 次方根
    b = 25**1000
    assert integer_nthroot(b, 1000) == (25, True)
    assert integer_nthroot(b + 1, 1000) == (25, False)
    assert integer_nthroot(b - 1, 1000) == (24, False)
    
    # 测试大数的平方根
    c = 10**400
    c2 = c**2
    assert integer_nthroot(c2, 2) == (c, True)
    # 确保整数的平方根函数返回正确的结果
    assert integer_nthroot(c2 + 1, 2) == (c, False)
    
    # 确保整数的平方根函数返回正确的结果
    assert integer_nthroot(c2 - 1, 2) == (c - 1, False)
    
    # 确保整数的平方根函数处理大数时不会出错
    assert integer_nthroot(2, 10**10) == (1, False)
    
    # 计算 10000 的阶乘的整数平方根，验证部分结果
    p, r = integer_nthroot(int(factorial(10000)), 100)
    assert p % (10**10) == 5322420655
    assert not r
    
    # 测试整数的平方根函数的性能
    assert integer_nthroot(2, 10**10) == (1, False)
    
    # 确保整数的平方根函数在可能时返回整数类型的结果
    assert type(integer_nthroot(2**61, 2)[0]) is int
# 定义测试函数，用于测试整数的 n 次根函数
def test_integer_nthroot_overflow():
    # 断言对于非常大的数，求其 50 次方根，返回结果应为 (10**50, True)
    assert integer_nthroot(10**(50*50), 50) == (10**50, True)
    # 断言对于非常大的数，求其 10000 次方根，返回结果应为 (10**10, True)
    assert integer_nthroot(10**100000, 10000) == (10**10, True)


# 定义测试函数，用于测试整数对数函数
def test_integer_log():
    # 对于无效输入，期望引发 ValueError 异常
    raises(ValueError, lambda: integer_log(2, 1))
    raises(ValueError, lambda: integer_log(0, 2))
    raises(ValueError, lambda: integer_log(1.1, 2))
    raises(ValueError, lambda: integer_log(1, 2.2))

    # 断言对数值为 1，任何基数的对数应为 (0, True)
    assert integer_log(1, 2) == (0, True)
    assert integer_log(1, 3) == (0, True)
    # 断言对数值为 2，以 3 为基数的对数应为 (0, False)
    assert integer_log(2, 3) == (0, False)
    # 断言对数值为 3，以 3 为基数的对数应为 (1, True)
    assert integer_log(3, 3) == (1, True)
    assert integer_log(3*2, 3) == (1, False)
    assert integer_log(3**2, 3) == (2, True)
    assert integer_log(3*4, 3) == (2, False)
    assert integer_log(3**3, 3) == (3, True)
    assert integer_log(27, 5) == (2, False)
    assert integer_log(2, 3) == (0, False)
    assert integer_log(-4, 2) == (2, False)
    assert integer_log(-16, 4) == (0, False)
    assert integer_log(-4, -2) == (2, False)
    assert integer_log(4, -2) == (2, True)
    assert integer_log(-8, -2) == (3, True)
    assert integer_log(8, -2) == (3, False)
    assert integer_log(-9, 3) == (0, False)
    assert integer_log(-9, -3) == (2, False)
    assert integer_log(9, -3) == (2, True)
    assert integer_log(-27, -3) == (3, True)
    assert integer_log(27, -3) == (3, False)


# 定义测试函数，用于测试整数平方根函数
def test_isqrt():
    # 导入 math 模块中的 sqrt 函数
    from math import sqrt as _sqrt
    # 设置一个较大的限制值
    limit = 4503599761588223
    # 断言 isqrt 函数的结果应与 math.sqrt 函数对 limit 的结果相等
    assert int(_sqrt(limit)) == integer_nthroot(limit, 2)[0]
    # 断言 isqrt 函数的结果应与 integer_nthroot 函数对 limit + 1 的结果相等
    assert int(_sqrt(limit + 1)) != integer_nthroot(limit + 1, 2)[0]
    # 断言 isqrt 函数对 limit + 1 的结果应与 integer_nthroot 函数对 limit + 1 的结果相等
    assert isqrt(limit + 1) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + S.Half) == integer_nthroot(limit, 2)[0]
    assert isqrt(limit + 1 + S.Half) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 2 + S.Half) == integer_nthroot(limit + 2, 2)[0]

    # 对于 GitHub 问题编号为 17034 的回归测试
    assert isqrt(4503599761588224) == 67108864
    assert isqrt(9999999999999999) == 99999999

    # 其他边界情况的测试，特别是涉及非整数的情况
    raises(ValueError, lambda: isqrt(-1))
    raises(ValueError, lambda: isqrt(-10**1000))
    raises(ValueError, lambda: isqrt(Rational(-1, 2)))

    tiny = Rational(1, 10**1000)
    raises(ValueError, lambda: isqrt(-tiny))
    assert isqrt(1-tiny) == 0
    assert isqrt(4503599761588224-tiny) == 67108864
    assert isqrt(10**100 - tiny) == 10**50 - 1


# 定义测试函数，用于测试 Integer 类的幂函数
def test_powers_Integer():
    """Test Integer._eval_power"""
    # 检查无穷大的情况
    assert S.One ** S.Infinity is S.NaN
    assert S.NegativeOne** S.Infinity is S.NaN
    assert S(2) ** S.Infinity is S.Infinity
    assert S(-2)** S.Infinity == zoo
    assert S(0) ** S.Infinity is S.Zero

    # 检查 NaN 的情况
    assert S.One ** S.NaN is S.NaN
    assert S.NegativeOne ** S.NaN is S.NaN

    # 检查精确根的情况
    assert S.NegativeOne ** Rational(6, 5) == - (-1)**(S.One/5)
    assert sqrt(S(4)) == 2
    assert sqrt(S(-4)) == I * 2
    assert S(16) ** Rational(1, 4) == 2
    # 测试负数的有理指数幂
    assert S(-16) ** Rational(1, 4) == 2 * (-1)**Rational(1, 4)
    assert S(9) ** Rational(3, 2) == 27
    assert S(-9) ** Rational(3, 2) == -27*I
    assert S(27) ** Rational(2, 3) == 9
    assert S(-27) ** Rational(2, 3) == 9 * (S.NegativeOne ** Rational(2, 3))
    assert (-2) ** Rational(-2, 1) == Rational(1, 4)

    # 非精确的根
    assert sqrt(-3) == I*sqrt(3)
    assert (3) ** (Rational(3, 2)) == 3 * sqrt(3)
    assert (-3) ** (Rational(3, 2)) == - 3 * sqrt(-3)
    assert (-3) ** (Rational(5, 2)) == 9 * I * sqrt(3)
    assert (-3) ** (Rational(7, 2)) == - I * 27 * sqrt(3)
    assert (2) ** (Rational(3, 2)) == 2 * sqrt(2)
    assert (2) ** (Rational(-3, 2)) == sqrt(2) / 4
    assert (81) ** (Rational(2, 3)) == 9 * (S(3) ** (Rational(2, 3)))
    assert (-81) ** (Rational(2, 3)) == 9 * (S(-3) ** (Rational(2, 3)))
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3

    # 合并根
    assert sqrt(6) + sqrt(24) == 3*sqrt(6)
    assert sqrt(2) * sqrt(3) == sqrt(6)

    # 分离符号和常数
    x = Symbol("x")
    assert sqrt(49 * x) == 7 * sqrt(x)
    assert sqrt((3 - sqrt(pi)) ** 2) == 3 - sqrt(pi)

    # 检查大数值时的速度
    assert (2**64 + 1) ** Rational(4, 3)
    assert (2**64 + 1) ** Rational(17, 25)

    # 负有理指数幂和负基数
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3
    assert (-2) ** Rational(-10, 3) == \
        (-1)**Rational(2, 3)*2**Rational(2, 3)/16
    assert abs(Pow(-2, Rational(-10, 3)).n() -
        Pow(-2, Rational(-10, 3), evaluate=False).n()) < 1e-16

    # 负基数和有理指数幂以及一些简化
    assert (-8) ** Rational(2, 5) == \
        2*(-1)**Rational(2, 5)*2**Rational(1, 5)
    assert (-4) ** Rational(9, 5) == \
        -8*(-1)**Rational(4, 5)*2**Rational(3, 5)

    # 测试因式分解
    assert S(1234).factors() == {617: 1, 2: 1}
    assert Rational(2*3, 3*5*7).factors() == {2: 1, 5: -1, 7: -1}

    # 测试 eval_power 对大于当前限制（2**15）的因数化的影响
    from sympy.ntheory.generate import nextprime
    n = nextprime(2**15)
    assert sqrt(n**2) == n
    assert sqrt(n**3) == n*sqrt(n)
    assert sqrt(4*n) == 2*sqrt(n)

    # 检查具有共享 gcd 的基数与幂的因数化
    assert (2**4*3)**Rational(1, 6) == 2**Rational(2, 3)*3**Rational(1, 6)
    assert (2**4*3)**Rational(5, 6) == 8*2**Rational(1, 3)*3**Rational(5, 6)

    # 检查共享 gcd 的基数的因数化
    assert 2**Rational(1, 3)*3**Rational(1, 4)*6**Rational(1, 5) == \
        2**Rational(8, 15)*3**Rational(9, 20)
    assert sqrt(8)*24**Rational(1, 3)*6**Rational(1, 5) == \
        4*2**Rational(7, 10)*3**Rational(8, 15)
    # 使用 sympy 库的数学函数和符号计算进行断言测试
    
    # 断言一：验证复杂表达式是否相等
    assert sqrt(8)*(-24)**Rational(1, 3)*(-6)**Rational(1, 5) == \
        4*(-3)**Rational(8, 15)*2**Rational(7, 10)
    
    # 断言二：验证幂运算的等式
    assert 2**Rational(1, 3)*2**Rational(8, 9) == 2*2**Rational(2, 9)
    
    # 断言三：验证幂运算的等式
    assert 2**Rational(2, 3)*6**Rational(1, 3) == 2*3**Rational(1, 3)
    
    # 断言四：验证复杂幂运算的等式
    assert 2**Rational(2, 3)*6**Rational(8, 9) == \
        2*2**Rational(5, 9)*3**Rational(8, 9)
    
    # 断言五：验证带负数幂运算的等式
    assert (-2)**Rational(2, S(3))*(-4)**Rational(1, S(3)) == -2*2**Rational(1, 3)
    
    # 断言六：验证指数运算的等式
    assert 3*Pow(3, 2, evaluate=False) == 3**3
    
    # 断言七：验证有理指数运算的等式
    assert 3*Pow(3, Rational(-1, 3), evaluate=False) == 3**Rational(2, 3)
    
    # 断言八：验证复杂的幂运算和符号运算的等式
    assert (-2)**Rational(1, 3)*(-3)**Rational(1, 4)*(-5)**Rational(5, 6) == \
        -(-1)**Rational(5, 12)*2**Rational(1, 3)*3**Rational(1, 4) * \
        5**Rational(5, 6)
    
    # 断言九：验证符号计算中的偶数幂次方等式
    assert Integer(-2)**Symbol('', even=True) == \
        Integer(2)**Symbol('', even=True)
    
    # 断言十：验证复数幂次方的等式
    assert (-1)**Float(.5) == 1.0*I
# 测试 Rational 类的 _eval_power 方法
def test_powers_Rational():
    """Test Rational._eval_power"""

    # 检查无穷大的幂运算
    assert S.Half ** S.Infinity == 0
    assert Rational(3, 2) ** S.Infinity is S.Infinity
    assert Rational(-1, 2) ** S.Infinity == 0
    assert Rational(-3, 2) ** S.Infinity == zoo  # 这里的 zoo 可能是某个特殊值或全局变量

    # 检查 NaN 的幂运算
    assert Rational(3, 4) ** S.NaN is S.NaN
    assert Rational(-2, 3) ** S.NaN is S.NaN

    # 测试分子为整数的精确根
    assert sqrt(Rational(4, 3)) == 2 * sqrt(3) / 3
    assert Rational(4, 3) ** Rational(3, 2) == 8 * sqrt(3) / 9
    assert sqrt(Rational(-4, 3)) == I * 2 * sqrt(3) / 3
    assert Rational(-4, 3) ** Rational(3, 2) == - I * 8 * sqrt(3) / 9
    assert Rational(27, 2) ** Rational(1, 3) == 3 * (2 ** Rational(2, 3)) / 2
    assert Rational(5**3, 8**3) ** Rational(4, 3) == Rational(5**4, 8**4)

    # 测试分母为整数的精确根
    assert sqrt(Rational(1, 4)) == S.Half
    assert sqrt(Rational(1, -4)) == I * S.Half
    assert sqrt(Rational(3, 4)) == sqrt(3) / 2
    assert sqrt(Rational(3, -4)) == I * sqrt(3) / 2
    assert Rational(5, 27) ** Rational(1, 3) == (5 ** Rational(1, 3)) / 3

    # 测试非精确根的情况
    assert sqrt(S.Half) == sqrt(2) / 2
    assert sqrt(Rational(-4, 7)) == I * sqrt(Rational(4, 7))
    assert Rational(-3, 2)**Rational(-7, 3) == \
        -4*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/27
    assert Rational(-3, 2)**Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*2**Rational(2, 3)*3**Rational(1, 3)/3
    assert Rational(-3, 2)**Rational(-10, 3) == \
        8*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/81
    assert abs(Pow(Rational(-2, 3), Rational(-7, 4)).n() -
        Pow(Rational(-2, 3), Rational(-7, 4), evaluate=False).n()) < 1e-16

    # 负整数幂和负有理数底数
    assert Rational(-2, 3) ** Rational(-2, 1) == Rational(9, 4)

    # 测试浮点数的幂运算
    a = Rational(1, 10)
    assert a**Float(a, 2) == Float(a, 2)**Float(a, 2)
    assert Rational(-2, 3)**Symbol('', even=True) == \
        Rational(2, 3)**Symbol('', even=True)


# 测试 Float 类的幂运算
def test_powers_Float():
    assert str((S('-1/10')**S('3/10')).n()) == str(Float(-.1)**(.3))


# 测试 Integer 类的左移运算
def test_lshift_Integer():
    assert Integer(0) << Integer(2) == Integer(0)
    assert Integer(0) << 2 == Integer(0)
    assert 0 << Integer(2) == Integer(0)

    assert Integer(0b11) << Integer(0) == Integer(0b11)
    assert Integer(0b11) << 0 == Integer(0b11)
    assert 0b11 << Integer(0) == Integer(0b11)

    assert Integer(0b11) << Integer(2) == Integer(0b11 << 2)
    assert Integer(0b11) << 2 == Integer(0b11 << 2)
    assert 0b11 << Integer(2) == Integer(0b11 << 2)

    assert Integer(-0b11) << Integer(2) == Integer(-0b11 << 2)
    assert Integer(-0b11) << 2 == Integer(-0b11 << 2)
    assert -0b11 << Integer(2) == Integer(-0b11 << 2)

    raises(TypeError, lambda: Integer(2) << 0.0)  # 不能将浮点数用作左移运算的位数
    raises(TypeError, lambda: 0.0 << Integer(2))  # 不能将浮点数用作左移运算的被移位数
    raises(ValueError, lambda: Integer(1) << Integer(-1))  # 不能用负数作为左移运算的位数


# 测试 Integer 类的右移运算
def test_rshift_Integer():
    assert Integer(0) >> Integer(2) == Integer(0)
    # 断言：使用整数对象进行按位右移操作，预期结果是整数对象 0
    assert Integer(0) >> 2 == Integer(0)
    # 断言：使用整数 0 进行按位右移操作，预期结果是整数对象 0
    assert 0 >> Integer(2) == Integer(0)
    
    # 断言：使用整数对象 0b11 进行按位右移 0 位，预期结果是整数对象 0b11
    assert Integer(0b11) >> Integer(0) == Integer(0b11)
    # 断言：使用整数对象 0b11 进行按位右移操作，整数 0 位，预期结果是整数对象 0b11
    assert Integer(0b11) >> 0 == Integer(0b11)
    # 断言：使用整数 0b11 进行按位右移 0 位，预期结果是整数对象 0b11
    assert 0b11 >> Integer(0) == Integer(0b11)
    
    # 断言：使用整数对象 0b11 进行按位右移 2 位，预期结果是整数对象 0
    assert Integer(0b11) >> Integer(2) == Integer(0)
    # 断言：使用整数 0b11 进行按位右移 2 位，预期结果是整数对象 0
    assert Integer(0b11) >> 2 == Integer(0)
    # 断言：使用整数 0b11 进行按位右移 2 位，预期结果是整数对象 0
    assert 0b11 >> Integer(2) == Integer(0)
    
    # 断言：使用整数对象 -0b11 进行按位右移 2 位，预期结果是整数对象 -1
    assert Integer(-0b11) >> Integer(2) == Integer(-1)
    # 断言：使用整数对象 -0b11 进行按位右移 2 位，预期结果是整数对象 -1
    assert Integer(-0b11) >> 2 == Integer(-1)
    # 断言：使用整数 -0b11 进行按位右移 2 位，预期结果是整数对象 -1
    assert -0b11 >> Integer(2) == Integer(-1)
    
    # 断言：使用整数对象 0b1100 进行按位右移 2 位，预期结果是整数对象 0b0011
    assert Integer(0b1100) >> Integer(2) == Integer(0b1100 >> 2)
    # 断言：使用整数对象 0b1100 进行按位右移 2 位，预期结果是整数对象 0b0011
    assert Integer(0b1100) >> 2 == Integer(0b1100 >> 2)
    # 断言：使用整数 0b1100 进行按位右移 2 位，预期结果是整数对象 0b0011
    assert 0b1100 >> Integer(2) == Integer(0b1100 >> 2)
    
    # 断言：使用整数对象 -0b1100 进行按位右移 2 位，预期结果是整数对象 -0b0011
    assert Integer(-0b1100) >> Integer(2) == Integer(-0b1100 >> 2)
    # 断言：使用整数对象 -0b1100 进行按位右移 2 位，预期结果是整数对象 -0b0011
    assert Integer(-0b1100) >> 2 == Integer(-0b1100 >> 2)
    # 断言：使用整数 -0b1100 进行按位右移 2 位，预期结果是整数对象 -0b0011
    assert -0b1100 >> Integer(2) == Integer(-0b1100 >> 2)
    
    # 断言：使用整数对象 0b10 进行按位右移浮点数 0.0 位，预期引发 TypeError 异常
    raises(TypeError, lambda: Integer(0b10) >> 0.0)
    # 断言：使用浮点数 0.0 进行按位右移整数对象 2 位，预期引发 TypeError 异常
    raises(TypeError, lambda: 0.0 >> Integer(2))
    # 断言：使用整数对象 1 进行按位右移整数对象 -1 位，预期引发 ValueError 异常
    raises(ValueError, lambda: Integer(1) >> Integer(-1))
`
def test_and_Integer():
    # 测试整数与位运算的与操作
    assert Integer(0b01010101) & Integer(0b10101010) == Integer(0)
    # 测试整数与二进制字面量的与操作
    assert Integer(0b01010101) & 0b10101010 == Integer(0)
    # 测试二进制字面量与整数的与操作
    assert 0b01010101 & Integer(0b10101010) == Integer(0)

    # 测试整数与整数的与操作
    assert Integer(0b01010101) & Integer(0b11011011) == Integer(0b01010001)
    # 测试整数与二进制字面量的与操作
    assert Integer(0b01010101) & 0b11011011 == Integer(0b01010001)
    # 测试二进制字面量与整数的与操作
    assert 0b01010101 & Integer(0b11011011) == Integer(0b01010001)

    # 测试负整数与整数的与操作
    assert -Integer(0b01010101) & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)
    # 测试负整数与二进制字面量的与操作
    assert Integer(-0b01010101) & 0b11011011 == Integer(-0b01010101 & 0b11011011)
    # 测试负二进制字面量与整数的与操作
    assert -0b01010101 & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)

    # 测试整数与负整数的与操作
    assert Integer(0b01010101) & -Integer(0b11011011) == Integer(0b01010101 & -0b11011011)
    # 测试整数与负二进制字面量的与操作
    assert Integer(0b01010101) & -0b11011011 == Integer(0b01010101 & -0b11011011)
    # 测试二进制字面量与负整数的与操作
    assert 0b01010101 & Integer(-0b11011011) == Integer(0b01010101 & -0b11011011)

    # 测试类型错误抛出
    raises(TypeError, lambda: Integer(2) & 0.0)
    raises(TypeError, lambda: 0.0 & Integer(2))


def test_xor_Integer():
    # 测试整数与位运算的异或操作
    assert Integer(0b01010101) ^ Integer(0b11111111) == Integer(0b10101010)
    # 测试整数与二进制字面量的异或操作
    assert Integer(0b01010101) ^ 0b11111111 == Integer(0b10101010)
    # 测试二进制字面量与整数的异或操作
    assert 0b01010101 ^ Integer(0b11111111) == Integer(0b10101010)

    # 测试整数与整数的异或操作
    assert Integer(0b01010101) ^ Integer(0b11011011) == Integer(0b10001110)
    # 测试整数与二进制字面量的异或操作
    assert Integer(0b01010101) ^ 0b11011011 == Integer(0b10001110)
    # 测试二进制字面量与整数的异或操作
    assert 0b01010101 ^ Integer(0b11011011) == Integer(0b10001110)

    # 测试负整数与整数的异或操作
    assert -Integer(0b01010101) ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)
    # 测试负整数与二进制字面量的异或操作
    assert Integer(-0b01010101) ^ 0b11011011 == Integer(-0b01010101 ^ 0b11011011)
    # 测试负二进制字面量与整数的异或操作
    assert -0b01010101 ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)

    # 测试整数与负整数的异或操作
    assert Integer(0b01010101) ^ -Integer(0b11011011) == Integer(0b01010101 ^ -0b11011011)
    # 测试整数与负二进制字面量的异或操作
    assert Integer(0b01010101) ^ -0b11011011 == Integer(0b01010101 ^ -0b11011011)
    # 测试二进制字面量与负整数的异或操作
    assert 0b01010101 ^ Integer(-0b11011011) == Integer(0b01010101 ^ -0b11011011)

    # 测试类型错误抛出
    raises(TypeError, lambda: Integer(2) ^ 0.0)
    raises(TypeError, lambda: 0.0 ^ Integer(2))


def test_or_Integer():
    # 测试整数与位运算的或操作
    assert Integer(0b01010101) | Integer(0b10101010) == Integer(0b11111111)
    # 测试整数与二进制字面量的或操作
    assert Integer(0b01010101) | 0b10101010 == Integer(0b11111111)
    # 测试二进制字面量与整数的或操作
    assert 0b01010101 | Integer(0b10101010) == Integer(0b11111111)

    # 测试整数与整数的或操作
    assert Integer(0b01010101) | Integer(0b11011011) == Integer(0b11011111)
    # 测试整数与二进制字面量的或操作
    assert Integer(0b01010101) | 0b11011011 == Integer(0b11011111)
    # 测试二进制字面量与整数的或操作
    assert 0b01010101 | Integer(0b11011011) == Integer(0b11011111)

    # 测试负整数与整数的或操作
    assert -Integer(0b01010101) | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)
    # 测试负整数与二进制字面量的或操作
    assert Integer(-0b01010101) | 0b11011011 == Integer(-0b01010101 | 0b11011011)
    # 测试负二进制字面量与整数的或操作
    assert -0b01010101 | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)

    # 测试整数与负整数的或操作
    assert Integer(0b01010101) | -Integer(0b11011011) == Integer(0b01010101 | -0b11011011)
    # 测试整数与负二进制字面量的或操作
    assert Integer(0b01010101) | -0b11011011 == Integer(0b01010101 | -0b11011011)
    # 测试二进制字面量与负整数的或操作
    assert 0b01010101 | Integer(-0b11011011) == Integer(0b01010101 | -0b11011011)
    # 断言：对整数 0b01010101 和整数 -0b11011011 进行按位或运算，预期结果是整数 0b01010101 | -0b11011011
    assert 0b01010101 | Integer(-0b11011011) == Integer(0b01010101 | -0b11011011)
    
    # 断言：尝试对整数 2 和浮点数 0.0 进行按位或运算，预期引发 TypeError 异常
    raises(TypeError, lambda: Integer(2) | 0.0)
    
    # 断言：尝试对浮点数 0.0 和整数 2 进行按位或运算，预期引发 TypeError 异常
    raises(TypeError, lambda: 0.0 | Integer(2))
def test_invert_Integer():
    # 测试按位取反操作符 ~ 对整数类型 Integer 的影响
    assert ~Integer(0b01010101) == Integer(-0b01010110)
    # 测试按位取反操作符 ~ 对整数类型 Integer 的影响
    assert ~Integer(0b01010101) == Integer(~0b01010101)
    # 测试按位取反操作符 ~ 的连续应用
    assert ~(~Integer(0b01010101)) == Integer(0b01010101)


def test_abs1():
    # 测试有理数类型 Rational 的绝对值函数 abs
    assert Rational(1, 6) != Rational(-1, 6)
    # 测试有理数类型 Rational 的绝对值函数 abs
    assert abs(Rational(1, 6)) == abs(Rational(-1, 6))


def test_accept_int():
    # 测试将整数转换为浮点数 Float 的情况
    assert not Float(4) == 4
    # 测试将整数转换为浮点数 Float 的情况
    assert Float(4) != 4
    # 测试将整数转换为浮点数 Float 的情况
    assert Float(4) == 4.0


def test_dont_accept_str():
    # 测试不接受字符串形式的输入，要求输入为浮点数 Float
    assert Float("0.2") != "0.2"
    # 测试不接受字符串形式的输入，要求输入为浮点数 Float
    assert not (Float("0.2") == "0.2")


def test_int():
    # 测试有理数类型 Rational 转换为整数类型 int
    a = Rational(5)
    assert int(a) == 5
    a = Rational(9, 10)
    # 测试有理数类型 Rational 转换为整数类型 int
    assert int(a) == int(-a) == 0
    assert 1/(-1)**Rational(2, 3) == -(-1)**Rational(1, 3)
    # issue 10368
    # 测试特定问题的解决方案
    a = Rational(32442016954, 78058255275)
    assert type(int(a)) is type(int(-a)) is int


def test_int_NumberSymbols():
    # 测试数学常数（例如 Catalan、EulerGamma、pi 等）转换为整数
    assert int(Catalan) == 0
    assert int(EulerGamma) == 0
    assert int(pi) == 3
    assert int(E) == 2
    assert int(GoldenRatio) == 1
    assert int(TribonacciConstant) == 1
    for i in [Catalan, E, EulerGamma, GoldenRatio, TribonacciConstant, pi]:
        a, b = i.approximation_interval(Integer)
        # 验证转换后的整数值等于其近似值的下限
        ia = int(i)
        assert ia == a
        assert isinstance(ia, int)
        # 验证转换后的整数值等于其近似值的上限
        assert b == a + 1
        assert a.is_Integer and b.is_Integer


def test_real_bug():
    x = Symbol("x")
    # 测试字符串表示的问题
    assert str(2.0*x*x) in ["(2.0*x)*x", "2.0*x**2", "2.00000000000000*x**2"]
    assert str(2.1*x*x) != "(2.0*x)*x"


def test_bug_sqrt():
    # 测试平方根函数的问题
    assert ((sqrt(Rational(2)) + 1)*(sqrt(Rational(2)) - 1)).expand() == 1


def test_pi_Pi():
    "Test that pi (instance) is imported, but Pi (class) is not"
    from sympy import pi  # noqa
    with raises(ImportError):
        from sympy import Pi  # noqa


def test_no_len():
    # 测试不支持长度操作的情况
    raises(TypeError, lambda: len(Rational(2)))
    raises(TypeError, lambda: len(Rational(2, 3)))
    raises(TypeError, lambda: len(Integer(2)))


def test_issue_3321():
    # 测试特定问题的解决方案
    assert sqrt(Rational(1, 5)) == Rational(1, 5)**S.Half
    assert 5 * sqrt(Rational(1, 5)) == sqrt(5)


def test_issue_3692():
    # 测试特定问题的解决方案
    assert ((-1)**Rational(1, 6)).expand(complex=True) == I/2 + sqrt(3)/2
    assert ((-5)**Rational(1, 6)).expand(complex=True) == \
        5**Rational(1, 6)*I/2 + 5**Rational(1, 6)*sqrt(3)/2
    assert ((-64)**Rational(1, 6)).expand(complex=True) == I + sqrt(3)


def test_issue_3423():
    x = Symbol("x")
    # 测试平方根函数的问题
    assert sqrt(x - 1).as_base_exp() == (x - 1, S.Half)
    assert sqrt(x - 1) != I*sqrt(1 - x)


def test_issue_3449():
    x = Symbol("x")
    # 测试平方根函数的问题
    assert sqrt(x - 1).subs(x, 5) == 2


def test_issue_13890():
    x = Symbol("x")
    e = (-x/4 - S.One/12)**x - 1
    f = simplify(e)
    a = Rational(9, 5)
    # 测试特定问题的解决方案
    assert abs(e.subs(x,a).evalf() - f.subs(x,a).evalf()) < 1e-15


def test_Integer_factors():
    def F(i):
        return Integer(i).factors()

    # 测试整数的因子函数
    assert F(1) == {}
    assert F(2) == {2: 1}
    assert F(3) == {3: 1}
    assert F(4) == {2: 2}
    assert F(5) == {5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(6) == {2: 1, 3: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(7) == {7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(8) == {2: 3}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(9) == {3: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(10) == {2: 1, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(11) == {11: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(12) == {2: 2, 3: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(13) == {13: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(14) == {2: 1, 7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(15) == {3: 1, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(16) == {2: 4}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(17) == {17: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(18) == {2: 1, 3: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(19) == {19: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(20) == {2: 2, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(21) == {3: 1, 7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(22) == {2: 1, 11: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(23) == {23: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(24) == {2: 3, 3: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(25) == {5: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(26) == {2: 1, 13: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(27) == {3: 3}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(28) == {2: 2, 7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(29) == {29: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(30) == {2: 1, 3: 1, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(31) == {31: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(32) == {2: 5}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(33) == {3: 1, 11: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(34) == {2: 1, 17: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(35) == {5: 1, 7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(36) == {2: 2, 3: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(37) == {37: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(38) == {2: 1, 19: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(39) == {3: 1, 13: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(40) == {2: 3, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(41) == {41: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(42) == {2: 1, 3: 1, 7: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(43) == {43: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(44) == {2: 2, 11: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(45) == {3: 2, 5: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(46) == {2: 1, 23: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(47) == {47: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(48) == {2: 4, 3: 1}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(49) == {7: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(50) == {2: 1, 5: 2}
    # 断言语句，验证函数 F 的返回值是否符合预期结果
    assert F(51) == {3: 1, 17: 1}
def test_Rational_factors():
    # 嵌套函数，计算有理数 p/q 的因子
    def F(p, q, visual=None):
        return Rational(p, q).factors(visual=visual)

    # 断言不同输入下计算得到的因子字典是否符合预期
    assert F(2, 3) == {2: 1, 3: -1}
    assert F(2, 9) == {2: 1, 3: -2}
    assert F(2, 15) == {2: 1, 3: -1, 5: -1}
    assert F(6, 10) == {3: 1, 5: -1}


def test_issue_4107():
    # 断言表达式是否不等于零
    assert pi*(E + 10) + pi*(-E - 10) != 0
    assert pi*(E + 10**10) + pi*(-E - 10**10) != 0
    assert pi*(E + 10**20) + pi*(-E - 10**20) != 0
    assert pi*(E + 10**80) + pi*(-E - 10**80) != 0

    # 断言扩展后的表达式是否等于零
    assert (pi*(E + 10) + pi*(-E - 10)).expand() == 0
    assert (pi*(E + 10**10) + pi*(-E - 10**10)).expand() == 0
    assert (pi*(E + 10**20) + pi*(-E - 10**20)).expand() == 0
    assert (pi*(E + 10**80) + pi*(-E - 10**80)).expand() == 0


def test_IntegerInteger():
    # 创建整数对象并进行断言比较
    a = Integer(4)
    b = Integer(a)

    assert a == b


def test_Rational_gcd_lcm_cofactors():
    # 断言整数对象的最大公约数和最小公倍数计算是否正确
    assert Integer(4).gcd(2) == Integer(2)
    assert Integer(4).lcm(2) == Integer(4)
    assert Integer(4).gcd(Integer(2)) == Integer(2)
    assert Integer(4).lcm(Integer(2)) == Integer(4)
    
    # 断言有理数对象的最大公约数和最小公倍数计算是否正确
    a, b = 720**99911, 480**12342
    assert Integer(a).lcm(b) == a*b/Integer(a).gcd(b)

    assert Integer(4).gcd(3) == Integer(1)
    assert Integer(4).lcm(3) == Integer(12)
    assert Integer(4).gcd(Integer(3)) == Integer(1)
    assert Integer(4).lcm(Integer(3)) == Integer(12)

    assert Rational(4, 3).gcd(2) == Rational(2, 3)
    assert Rational(4, 3).lcm(2) == Integer(4)
    assert Rational(4, 3).gcd(Integer(2)) == Rational(2, 3)
    assert Rational(4, 3).lcm(Integer(2)) == Integer(4)

    assert Integer(4).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Integer(4).lcm(Rational(2, 9)) == Integer(4)

    assert Rational(4, 3).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Rational(4, 3).lcm(Rational(2, 9)) == Rational(4, 3)
    assert Rational(4, 5).gcd(Rational(2, 9)) == Rational(2, 45)
    assert Rational(4, 5).lcm(Rational(2, 9)) == Integer(4)
    assert Rational(5, 9).lcm(Rational(3, 7)) == Rational(Integer(5).lcm(3), Integer(9).gcd(7))

    assert Integer(4).cofactors(2) == (Integer(2), Integer(2), Integer(1))
    assert Integer(4).cofactors(Integer(2)) == (Integer(2), Integer(2), Integer(1))

    assert Integer(4).gcd(Float(2.0)) == Float(1.0)
    assert Integer(4).lcm(Float(2.0)) == Float(8.0)
    assert Integer(4).cofactors(Float(2.0)) == (Float(1.0), Float(4.0), Float(2.0))

    assert S.Half.gcd(Float(2.0)) == Float(1.0)
    assert S.Half.lcm(Float(2.0)) == Float(1.0)
    assert S.Half.cofactors(Float(2.0)) == (Float(1.0), Float(0.5), Float(2.0))


def test_Float_gcd_lcm_cofactors():
    # 断言浮点数对象和其他对象的最大公约数、最小公倍数及互质因子计算是否正确
    assert Float(2.0).gcd(Integer(4)) == Float(1.0)
    assert Float(2.0).lcm(Integer(4)) == Float(8.0)
    assert Float(2.0).cofactors(Integer(4)) == (Float(1.0), Float(2.0), Float(4.0))

    assert Float(2.0).gcd(S.Half) == Float(1.0)
    assert Float(2.0).lcm(S.Half) == Float(1.0)
    assert Float(2.0).cofactors(S.Half) == (Float(1.0), Float(2.0), Float(0.5))


def test_issue_4611():
    # 待实现
    # 使用 assert 断言来验证 pi 常数的数值近似是否在给定精度内
    assert abs(pi._evalf(50) - 3.14159265358979) < 1e-10
    # 使用 assert 断言来验证 E 常数的数值近似是否在给定精度内
    assert abs(E._evalf(50) - 2.71828182845905) < 1e-10
    # 使用 assert 断言来验证 Catalan 常数的数值近似是否在给定精度内
    assert abs(Catalan._evalf(50) - 0.915965594177219) < 1e-10
    # 使用 assert 断言来验证 EulerGamma 常数的数值近似是否在给定精度内
    assert abs(EulerGamma._evalf(50) - 0.577215664901533) < 1e-10
    # 使用 assert 断言来验证 GoldenRatio 常数的数值近似是否在给定精度内
    assert abs(GoldenRatio._evalf(50) - 1.61803398874989) < 1e-10
    # 使用 assert 断言来验证 TribonacciConstant 常数的数值近似是否在给定精度内
    assert abs(TribonacciConstant._evalf(50) - 1.83928675521416) < 1e-10

    # 创建一个符号变量 x
    x = Symbol("x")
    # 使用 assert 断言来验证 pi + x 的数值近似是否等于 pi 的数值近似加上 x 的值
    assert (pi + x).evalf() == pi.evalf() + x
    # 使用 assert 断言来验证 E + x 的数值近似是否等于 E 的数值近似加上 x 的值
    assert (E + x).evalf() == E.evalf() + x
    # 使用 assert 断言来验证 Catalan + x 的数值近似是否等于 Catalan 的数值近似加上 x 的值
    assert (Catalan + x).evalf() == Catalan.evalf() + x
    # 使用 assert 断言来验证 EulerGamma + x 的数值近似是否等于 EulerGamma 的数值近似加上 x 的值
    assert (EulerGamma + x).evalf() == EulerGamma.evalf() + x
    # 使用 assert 断言来验证 GoldenRatio + x 的数值近似是否等于 GoldenRatio 的数值近似加上 x 的值
    assert (GoldenRatio + x).evalf() == GoldenRatio.evalf() + x
    # 使用 assert 断言来验证 TribonacciConstant + x 的数值近似是否等于 TribonacciConstant 的数值近似加上 x 的值
    assert (TribonacciConstant + x).evalf() == TribonacciConstant.evalf() + x
@conserve_mpmath_dps
def test_conversion_to_mpmath():
    # 测试将不同类型的数学对象转换为 mpmath 对象
    assert mpmath.mpmathify(Integer(1)) == mpmath.mpf(1)
    assert mpmath.mpmathify(S.Half) == mpmath.mpf(0.5)
    assert mpmath.mpmathify(Float('1.23', 15)) == mpmath.mpf('1.23')

    # 测试复数转换为 mpmath 复数对象
    assert mpmath.mpmathify(I) == mpmath.mpc(1j)

    # 测试带实部和虚部的复数转换
    assert mpmath.mpmathify(1 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(S.Half + S.Half*I) == mpmath.mpc(0.5 + 0.5j)

    # 测试纯虚数的转换
    assert mpmath.mpmathify(2*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(2.0*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(S.Half*I) == mpmath.mpc(0.5j)

    # 设置更高的精度，然后测试高精度数值的转换
    mpmath.mp.dps = 100
    assert mpmath.mpmathify(pi.evalf(100) + pi.evalf(100)*I) == mpmath.pi + mpmath.pi*mpmath.j
    assert mpmath.mpmathify(pi.evalf(100)*I) == mpmath.pi*mpmath.j


def test_relational():
    # 测试不同类型数值的关系比较
    # 测试实数
    x = S(.1)
    assert (x != cos) is True
    assert (x == cos) is False

    # 测试有理数
    x = Rational(1, 3)
    assert (x != cos) is True
    assert (x == cos) is False

    # 整数会被当作有理数处理，因此这里省略整数的测试

    # 测试数学常数符号
    x = pi
    assert (x != cos) is True
    assert (x == cos) is False


def test_Integer_as_index():
    # 测试将整数对象用作索引
    assert 'hello'[Integer(2):] == 'llo'


def test_Rational_int():
    # 测试有理数对象转换为整数
    assert int(Rational(7, 5)) == 1
    assert int(S.Half) == 0
    assert int(Rational(-1, 2)) == 0
    assert int(-Rational(7, 5)) == -1


def test_zoo():
    # 创建不同属性的符号对象
    b = Symbol('b', finite=True)
    nz = Symbol('nz', nonzero=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    im = Symbol('i', imaginary=True)
    c = Symbol('c', complex=True)
    pb = Symbol('pb', positive=True)
    nb = Symbol('nb', negative=True)
    imb = Symbol('ib', imaginary=True, finite=True)
    for i in [I, S.Infinity, S.NegativeInfinity, S.Zero, S.One, S.Pi, S.Half, S(3), log(3),
              b, nz, p, n, im, pb, nb, imb, c]:
        # 对于每个数值或表达式 i，检查其是否有限且为实数或虚数部分
        if i.is_finite and (i.is_real or i.is_imaginary):
            # 断言 i + 无穷大（zoo）仍为无穷大
            assert i + zoo is zoo
            # 断言 i - 无穷大（zoo）仍为无穷大
            assert i - zoo is zoo
            # 断言 无穷大（zoo） + i 仍为无穷大
            assert zoo + i is zoo
            # 断言 无穷大（zoo） - i 仍为无穷大
            assert zoo - i is zoo
        # 如果 i 不是有限数
        elif i.is_finite is not False:
            # 断言 i + 无穷大（zoo）为加法表达式
            assert (i + zoo).is_Add
            # 断言 i - 无穷大（zoo）为加法表达式
            assert (i - zoo).is_Add
            # 断言 无穷大（zoo） + i 为加法表达式
            assert (zoo + i).is_Add
            # 断言 无穷大（zoo） - i 为加法表达式
            assert (zoo - i).is_Add
        else:
            # 断言 i + 无穷大（zoo）为非数值（NaN）
            assert (i + zoo) is S.NaN
            # 断言 i - 无穷大（zoo）为非数值（NaN）
            assert (i - zoo) is S.NaN
            # 断言 无穷大（zoo） + i 为非数值（NaN）
            assert (zoo + i) is S.NaN
            # 断言 无穷大（zoo） - i 为非数值（NaN）
            assert (zoo - i) is S.NaN

        # 如果 i 非零且为实数或虚数部分
        if fuzzy_not(i.is_zero) and (i.is_extended_real or i.is_imaginary):
            # 断言 i * 无穷大（zoo）为无穷大
            assert i*zoo is zoo
            # 断言 无穷大（zoo）* i 为无穷大
            assert zoo*i is zoo
        # 如果 i 为零
        elif i.is_zero:
            # 断言 i * 无穷大（zoo）为非数值（NaN）
            assert i*zoo is S.NaN
            # 断言 无穷大（zoo）* i 为非数值（NaN）
            assert zoo*i is S.NaN
        else:
            # 断言 i * 无穷大（zoo）为乘法表达式
            assert (i*zoo).is_Mul
            # 断言 无穷大（zoo）* i 为乘法表达式
            assert (zoo*i).is_Mul

        # 如果 1/i 非零且为实数或虚数部分
        if fuzzy_not((1/i).is_zero) and (i.is_real or i.is_imaginary):
            # 断言 无穷大（zoo）/ i 为无穷大
            assert zoo/i is zoo
        # 如果 1/i 为零
        elif (1/i).is_zero:
            # 断言 无穷大（zoo）/ i 为非数值（NaN）
            assert zoo/i is S.NaN
        # 如果 i 为零
        elif i.is_zero:
            # 断言 无穷大（zoo）/ i 为无穷大
            assert zoo/i is zoo
        else:
            # 断言 无穷大（zoo）/ i 为乘法表达式
            assert (zoo/i).is_Mul

    # 断言 I * 无穷大（oo）为乘法表达式，允许有向无穷大
    assert (I*oo).is_Mul  # allow directed infinity
    # 断言 无穷大（zoo） + 无穷大（zoo）为非数值（NaN）
    assert zoo + zoo is S.NaN
    # 断言 无穷大（zoo） * 无穷大（zoo）为无穷大
    assert zoo * zoo is zoo
    # 断言 无穷大（zoo） - 无穷大（zoo）为非数值（NaN）
    assert zoo - zoo is S.NaN
    # 断言 无穷大（zoo）/ 无穷大（zoo）为非数值（NaN）
    assert zoo/zoo is S.NaN
    # 断言 无穷大（zoo）** 无穷大（zoo）为非数值（NaN）
    assert zoo**zoo is S.NaN
    # 断言 无穷大（zoo）** 0 为 1
    assert zoo**0 is S.One
    # 断言 无穷大（zoo）** 2 为无穷大
    assert zoo**2 is zoo
    # 断言 1 / 无穷大（zoo）为零
    assert 1/zoo is S.Zero

    # 断言 Mul.flatten([S.NegativeOne, oo, S(0)]) 返回结果符合预期
    assert Mul.flatten([S.NegativeOne, oo, S(0)]) == ([S.NaN], [], None)
def test_issue_4122():
    # 创建符号对象 'x'，设定其属性为 nonpositive=True
    x = Symbol('x', nonpositive=True)
    # 断言无穷大加上 'x' 等于无穷大
    assert oo + x is oo

    # 修改 'x' 的属性为 extended_nonpositive=True
    x = Symbol('x', extended_nonpositive=True)
    # 断言无穷大加上 'x' 是一个加法表达式
    assert (oo + x).is_Add

    # 修改 'x' 的属性为 finite=True
    x = Symbol('x', finite=True)
    # 断言无穷大加上 'x' 是一个加法表达式，'x' 可能为虚数
    assert (oo + x).is_Add  # x could be imaginary

    # 修改 'x' 的属性为 nonnegative=True
    x = Symbol('x', nonnegative=True)
    # 断言无穷大加上 'x' 等于无穷大
    assert oo + x is oo

    # 修改 'x' 的属性为 extended_nonnegative=True
    x = Symbol('x', extended_nonnegative=True)
    # 断言无穷大加上 'x' 等于无穷大
    assert oo + x is oo

    # 修改 'x' 的属性为 finite=True, real=True
    x = Symbol('x', finite=True, real=True)
    # 断言无穷大加上 'x' 等于无穷大
    assert oo + x is oo

    # 类似地，对于负无穷大的情况
    # 修改 'x' 的属性为 nonnegative=True
    x = Symbol('x', nonnegative=True)
    # 断言负无穷大加上 'x' 等于负无穷大
    assert -oo + x is -oo

    # 修改 'x' 的属性为 extended_nonnegative=True
    x = Symbol('x', extended_nonnegative=True)
    # 断言负无穷大加上 'x' 是一个加法表达式
    assert (-oo + x).is_Add

    # 修改 'x' 的属性为 finite=True
    x = Symbol('x', finite=True)
    # 断言负无穷大加上 'x' 是一个加法表达式
    assert (-oo + x).is_Add

    # 修改 'x' 的属性为 nonpositive=True
    x = Symbol('x', nonpositive=True)
    # 断言负无穷大加上 'x' 等于负无穷大
    assert -oo + x is -oo

    # 修改 'x' 的属性为 extended_nonpositive=True
    x = Symbol('x', extended_nonpositive=True)
    # 断言负无穷大加上 'x' 等于负无穷大
    assert -oo + x is -oo

    # 修改 'x' 的属性为 finite=True, real=True
    x = Symbol('x', finite=True, real=True)
    # 断言负无穷大加上 'x' 等于负无穷大
    assert -oo + x is -oo
    # 断言语句，用于验证 Catalan.rewrite() 的返回值是否等于 Catalan 对象本身
    assert Catalan.rewrite() == Catalan
def test_bool_eq():
    # 测试布尔值相等性
    assert 0 == False
    # 测试 SymPy 对象和布尔值的比较
    assert S(0) == False
    # 测试 SymPy 对象和 SymPy 的 false 常量的不相等性
    assert S(0) != S.false
    # 测试整数和布尔值的比较
    assert 1 == True
    # 测试 SymPy 对象和布尔值的比较
    assert S.One == True
    # 测试 SymPy 对象和 SymPy 的 true 常量的不相等性
    assert S.One != S.true


def test_Float_eq():
    # 测试不同精度的浮点数不相等
    assert Float(.5, 10) != Float(.5, 11) != Float(.5, 1)
    # 测试不同精度的浮点数不相等
    assert Float(.12, 3) != Float(.12, 4)
    # 测试浮点数和数值常量不相等
    assert Float(.12, 3) != .12
    # 测试数值常量和浮点数不相等
    assert 0.12 != Float(.12, 3)
    # 测试字符串表示的浮点数和数值常量不相等
    assert Float('.12', 22) != .12
    # issue 11707
    # 测试浮点数和有理数的不相等性，除了 0 外，Float/Rational 是精确的
    assert Float('1.1') != Rational(11, 10)
    assert Rational(11, 10) != Float('1.1')
    # 覆盖更多情况
    assert not Float(3) == 2
    assert not Float(3) == Float(2)
    assert not Float(3) == 3
    assert not Float(2**2) == S.Half
    assert Float(2**2) == 4.0
    assert not Float(2**-2) == 1
    assert Float(2**-1) == 0.5
    assert not Float(2*3) == 3
    assert not Float(2*3) == 0.5
    assert Float(2*3) == 6.0
    assert not Float(2*3) == 6
    assert not Float(2*3) == 8
    assert not Float(.75) == Rational(3, 4)
    assert Float(.75) == 0.75
    assert Float(5/18) == 5/18
    # 4473
    assert Float(2.) != 3
    assert not Float((0,1,-3)) == S.One/8
    assert Float((0,1,-3)) == 1/8
    assert Float((0,1,-3)) != S.One/9
    # 16196
    # 测试整数和浮点数的比较，与 Python 不同
    assert not 2 == Float(2)
    assert t**2 != t**2.0


def test_issue_6640():
    from mpmath.libmp.libmpf import finf, fninf
    # fnan 不包含在内，因为 Float 不再返回 fnan，
    # 但是可以应用相同类型的测试
    assert Float(finf).is_zero is False
    assert Float(fninf).is_zero is False
    assert bool(Float(0)) is False


def test_issue_6349():
    # 测试不同字符串表示形式的浮点数的精度
    assert Float('23.e3', '')._prec == 10
    assert Float('23e3', '')._prec == 20
    assert Float('23000', '')._prec == 20
    assert Float('-23000', '')._prec == 20


def test_mpf_norm():
    # 测试 mpf_norm 函数的返回结果
    assert mpf_norm((1, 0, 1, 0), 10) == mpf('0')._mpf_
    assert Float._new((1, 0, 1, 0), 10)._mpf_ == mpf('0')._mpf_


def test_latex():
    # 测试 SymPy 对象到 LaTeX 字符串的转换
    assert latex(pi) == r"\pi"
    assert latex(E) == r"e"
    assert latex(GoldenRatio) == r"\phi"
    assert latex(TribonacciConstant) == r"\text{TribonacciConstant}"
    assert latex(EulerGamma) == r"\gamma"
    assert latex(oo) == r"\infty"
    assert latex(-oo) == r"-\infty"
    assert latex(zoo) == r"\tilde{\infty}"
    assert latex(nan) == r"\text{NaN}"
    assert latex(I) == r"i"


def test_issue_7742():
    # 测试负无穷取模 1 的结果是否为 NaN
    assert -oo % 1 is nan


def test_simplify_AlgebraicNumber():
    A = AlgebraicNumber
    e = 3**(S.One/6)*(3 + (135 + 78*sqrt(3))**Rational(2, 3))/(45 + 26*sqrt(3))**(S.One/3)
    # 测试 AlgebraicNumber 对象的简化
    assert simplify(A(e)) == A(12)  # wester test_C20

    e = (41 + 29*sqrt(2))**(S.One/5)
    # 测试 AlgebraicNumber 对象的简化
    assert simplify(A(e)) == A(1 + sqrt(2))  # wester test_C21
    # 创建复数表达式 e，其值为 (3 + 4*I) 的 3/2 次幂
    e = (3 + 4*I)**Rational(3, 2)
    # 断言简化后的 A(e) 等于 A(2 + 11*I)，用于验证问题 4401
    assert simplify(A(e)) == A(2 + 11*I)  # issue 4401
# 测试浮点数类型的幂等性
def test_Float_idempotence():
    # 创建一个浮点数对象 x，值为 '1.23'
    x = Float('1.23', '')
    # 创建另一个浮点数对象 y，值为 x 的复制
    y = Float(x)
    # 创建第三个浮点数对象 z，值为 x 的复制，并设置精度为 15
    z = Float(x, 15)
    # 断言 y 与 x 具有相同的值和相同的精度
    assert same_and_same_prec(y, x)
    # 断言 z 与 x 的值不同
    assert not same_and_same_prec(z, x)
    # 将 x 的值设为 10 的 20 次方
    x = Float(10**20)
    # 创建浮点数对象 y，值为 x 的复制
    y = Float(x)
    # 创建浮点数对象 z，值为 x 的复制，并设置精度为 15
    z = Float(x, 15)
    # 断言 y 与 x 具有相同的值和相同的精度
    assert same_and_same_prec(y, x)
    # 断言 z 与 x 的值不同
    assert not same_and_same_prec(z, x)


# 测试浮点数比较函数
def test_comp1():
    # 计算根号 2 的近似值，精确到小数点后 7 位
    a = sqrt(2).n(7)
    # 断言 a 与 1.4142129 不相等
    assert comp(a, 1.4142129) is False
    # 断言 a 与 1.4142130 相等
    assert comp(a, 1.4142130)
    # 断言 a 与 1.4142141 不相等
    assert comp(a, 1.4142141)
    # 断言 a 与 1.4142142 不相等
    assert comp(a, 1.4142142) is False
    # 断言根号 2 的近似值，精确到小数点后 2 位，与 '1.4' 相等
    assert comp(sqrt(2).n(2), '1.4')
    # 断言根号 2 的近似值，精确到小数点后 2 位，与浮点数对象 Float(1.4, 2) 相等
    assert comp(sqrt(2).n(2), Float(1.4, 2), '')
    # 断言根号 2 的近似值，精确到小数点后 2 位，与 1.4 相等
    assert comp(sqrt(2).n(2), 1.4, '')
    # 断言根号 2 的近似值，精确到小数点后 2 位，与浮点数对象 Float(1.4, 3) 不相等
    assert comp(sqrt(2).n(2), Float(1.4, 3), '') is False
    # 断言根号 2 加上根号 3 乘以虚数单位 I，与 1.4 加上 1.7*I 相等，精度为 0.1
    assert comp(sqrt(2) + sqrt(3)*I, 1.4 + 1.7*I, .1)
    # 断言根号 2 加上根号 3 乘以虚数单位 I，与 (1.5 + 1.7*I)*0.89 不相等，精度为 0.1
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.89, .1)
    # 断言根号 2 加上根号 3 乘以虚数单位 I，与 (1.5 + 1.7*I)*0.90 相等，精度为 0.1
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.90, .1)
    # 断言根号 2 加上根号 3 乘以虚数单位 I，与 (1.5 + 1.7*I)*1.07 相等，精度为 0.1
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.07, .1)
    # 断言根号 2 加上根号 3 乘以虚数单位 I，与 (1.5 + 1.7*I)*1.08 不相等，精度为 0.1
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.08, .1)
    # 返回一个列表，包含所有满足条件的 (i, j) 对，其中 i 在 130 到 149 之间，j 在 170 到 179 之间
    assert [(i, j)
            for i in range(130, 150)
            for j in range(170, 180)
            if comp((sqrt(2)+ I*sqrt(3)).n(3), i/100. + I*j/100.)] == [
        (141, 173), (142, 173)]
    # 断言调用 comp 函数时传递 t 会引发 ValueError 异常
    raises(ValueError, lambda: comp(t, '1'))
    # 断言调用 comp 函数时传递 t 会引发 ValueError 异常
    raises(ValueError, lambda: comp(t, 1))
    # 断言 0 与 0.0 相等
    assert comp(0, 0.0)
    # 断言 0.5 与 S.Half 相等
    assert comp(.5, S.Half)
    # 断言 2 加上根号 2 与 2.0 加上根号 2 相等
    assert comp(2 + sqrt(2), 2.0 + sqrt(2))
    # 断言 0 与 1 不相等
    assert not comp(0, 1)
    # 断言 2 与根号 2 不相等
    assert not comp(2, sqrt(2))
    # 断言 2 加上虚数单位 I 与 2.0 加上根号 2 不相等
    assert not comp(2 + I, 2.0 + sqrt(2))
    # 断言 2.0 加上根号 2 与 2 加上虚数单位 I 不相等
    assert not comp(2.0 + sqrt(2), 2 + I)
    # 断言 2.0 加上根号 2 与根号 3 不相等
    assert not comp(2.0 + sqrt(2), sqrt(3))
    # 断言根号 2 的近似值，精确到小数点后 4 位，与 0.3183 相等，精度为 1e-5
    assert comp(1/pi.n(4), 0.3183, 1e-5)
    # 断言根号 2 的近似值，精确到小数点后 4 位，与 0.3183 不相等，精度为 8e-6
    assert not comp(1/pi.n(4), 0.3183, 8e-6)


# 测试特定问题：oo 的次方运算
def test_issue_9491():
    # 断言 oo 的 zoo 次方结果为 nan（不是一个数字）
    assert oo**zoo is nan


# 测试特定问题：浮点数次方运算
def test_issue_10063():
    # 断言 2 的 Float(3) 次方等于 Float(8)
    assert 2**Float(3) == Float(8)


# 测试特定问题：oo 的次方运算（复数结果）
def test_issue_10020():
    # 断言 oo 的 I 次方结果为 S.NaN
    assert oo**I is S.NaN
    # 断言 oo 的 (1 + I) 次方结果为 S.ComplexInfinity
    assert oo**(1 + I) is S.ComplexInfinity
    # 断言 oo 的 (-1 + I) 次方结果为 S.Zero
    assert oo**(-1 + I) is
    # 调用 raises 函数来验证 mod_inverse 函数的异常处理功能，预期会抛出 TypeError 异常
    raises(TypeError, lambda: mod_inverse(2, x))
    
    # 调用 raises 函数来验证 mod_inverse 函数的异常处理功能，预期会抛出 ValueError 异常，因为 S.Half 不是整数
    raises(ValueError, lambda: mod_inverse(2, S.Half))
    
    # 调用 raises 函数来验证 mod_inverse 函数的异常处理功能，预期会抛出 ValueError 异常，因为 cos(1)**2 + sin(1)**2 不是整数
    raises(ValueError, lambda: mod_inverse(2, cos(1)**2 + sin(1)**2))
def test_golden_ratio_rewrite_as_sqrt():
    # 断言黄金比例重写为平方根的表达式
    assert GoldenRatio.rewrite(sqrt) == S.Half + sqrt(5)*S.Half


def test_tribonacci_constant_rewrite_as_sqrt():
    # 断言第三类斐波那契常数重写为平方根的表达式
    assert TribonacciConstant.rewrite(sqrt) == \
      (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3


def test_comparisons_with_unknown_type():
    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.
        """
        
    # 创建整数、浮点数、有理数对象及未知类型对象Foo
    ni, nf, nr = Integer(3), Float(1.0), Rational(1, 3)
    foo = Foo()

    # 针对不同类型对象进行多种比较操作，预期引发TypeError异常
    for n in ni, nf, nr, oo, -oo, zoo, nan:
        assert n != foo
        assert foo != n
        assert not n == foo
        assert not foo == n
        raises(TypeError, lambda: n < foo)
        raises(TypeError, lambda: foo > n)
        raises(TypeError, lambda: n > foo)
        raises(TypeError, lambda: foo < n)
        raises(TypeError, lambda: n <= foo)
        raises(TypeError, lambda: foo >= n)
        raises(TypeError, lambda: n >= foo)
        raises(TypeError, lambda: foo <= n)

    class Bar:
        """
        Class that considers itself equal to any instance of Number except
        infinities and nans, and relies on SymPy types returning the
        NotImplemented singleton for symmetric equality relations.
        """
        
        # 自定义Bar类的等于和不等于操作
        def __eq__(self, other):
            if other in (oo, -oo, zoo, nan):
                return False
            if isinstance(other, Number):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    # 针对整数、浮点数、有理数对象进行等于和不等于操作
    for n in ni, nf, nr:
        assert n == bar
        assert bar == n
        assert not n != bar
        assert not bar != n

    # 针对无穷大、负无穷大、未定义数、NaN进行等于和不等于操作
    for n in oo, -oo, zoo, nan:
        assert n != bar
        assert bar != n
        assert not n == bar
        assert not bar == n

    # 针对整数、浮点数、有理数对象及无穷大、负无穷大、未定义数、NaN进行比较操作，预期引发TypeError异常
    for n in ni, nf, nr, oo, -oo, zoo, nan:
        raises(TypeError, lambda: n < bar)
        raises(TypeError, lambda: bar > n)
        raises(TypeError, lambda: n > bar)
        raises(TypeError, lambda: bar < n)
        raises(TypeError, lambda: n <= bar)
        raises(TypeError, lambda: bar >= n)
        raises(TypeError, lambda: n >= bar)
        raises(TypeError, lambda: bar <= n)


def test_NumberSymbol_comparison():
    from sympy.core.tests.test_relational import rel_check
    # 检查NumberSymbol类型对象的比较
    rpi = Rational('905502432259640373/288230376151711744')
    fpi = Float(float(pi))
    assert rel_check(rpi, fpi)


def test_Integer_precision():
    # 确保关键字参数中的整数输入正常工作
    assert Float('1.0', dps=Integer(15))._prec == 53
    assert Float('1.0', precision=Integer(15))._prec == 15
    assert type(Float('1.0', precision=Integer(15))._prec) == int
    assert sympify(srepr(Float('1.0', precision=15))) == Float('1.0', precision=15)


def test_numpy_to_float():
    from sympy.testing.pytest import skip
    from sympy.external import import_module
    np = import_module('numpy')
    # 检查是否导入了 numpy 模块，如果没有则跳过测试并输出消息
    if not np:
        skip('numpy not installed. Abort numpy tests.')

    # 定义一个函数，用于检查浮点数精度和相对误差
    def check_prec_and_relerr(npval, ratval):
        # 确定 numpy 浮点数的精度
        prec = np.finfo(npval).nmant + 1
        # 将 npval 转换为自定义的 Float 类型
        x = Float(npval)
        # 断言 x 对象的精度与预期的一致
        assert x._prec == prec
        # 使用指定精度创建一个 Float 对象 y，并比较相对误差
        y = Float(ratval, precision=prec)
        assert abs((x - y)/y) < 2**(-(prec + 1))

    # 分别调用 check_prec_and_relerr 函数，检查不同精度下的计算结果
    check_prec_and_relerr(np.float16(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float32(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float64(2.0/3), Rational(2, 3))
    
    # 在某些体系结构/编译器上，使用扩展精度 longdouble 进行计算
    x = np.longdouble(2)/3
    check_prec_and_relerr(x, Rational(2, 3))
    # 使用精度为 10 创建 Float 对象 y，并断言其与指定精度下的 Rational 对象相同
    y = Float(x, precision=10)
    assert same_and_same_prec(y, Float(Rational(2, 3), precision=10))

    # 测试在复数输入时是否会引发 TypeError 异常
    raises(TypeError, lambda: Float(np.complex64(1+2j)))
    raises(TypeError, lambda: Float(np.complex128(1+2j)))
# 定义测试函数 test_Integer_ceiling_floor，用于测试 Integer 类的 floor 和 ceiling 方法
def test_Integer_ceiling_floor():
    # 创建一个 Integer 对象 a，值为 4
    a = Integer(4)

    # 断言 a 的 floor 方法返回自身
    assert a.floor() == a
    # 断言 a 的 ceiling 方法返回自身
    assert a.ceiling() == a


# 定义测试函数 test_ComplexInfinity，用于测试 zoo（复数无穷大）对象的 floor 和 ceiling 方法以及乘方操作
def test_ComplexInfinity():
    # 断言 zoo 的 floor 方法返回自身
    assert zoo.floor() is zoo
    # 断言 zoo 的 ceiling 方法返回自身
    assert zoo.ceiling() is zoo
    # 断言 zoo 的乘方操作 zoo**zoo 返回 S.NaN（不是一个数字）
    assert zoo**zoo is S.NaN


# 定义测试函数 test_Infinity_floor_ceiling_power，用于测试 oo（正无穷大）对象的 floor 和 ceiling 方法以及乘方操作
def test_Infinity_floor_ceiling_power():
    # 断言 oo 的 floor 方法返回自身
    assert oo.floor() is oo
    # 断言 oo 的 ceiling 方法返回自身
    assert oo.ceiling() is oo
    # 断言 oo 的乘方操作 oo**S.NaN 返回 S.NaN
    assert oo**S.NaN is S.NaN
    # 断言 oo 的乘方操作 oo**zoo 返回 S.NaN
    assert oo**zoo is S.NaN


# 定义测试函数 test_One_power，用于测试 S.One（符号 1）对象的乘方操作
def test_One_power():
    # 断言 S.One 的乘方操作 S.One**12 返回 S.One
    assert S.One**12 is S.One
    # 断言 S.NegativeOne 的乘方操作 S.NegativeOne**S.NaN 返回 S.NaN
    assert S.NegativeOne**S.NaN is S.NaN


# 定义测试函数 test_NegativeInfinity，用于测试 -oo（负无穷大）对象的 floor、ceiling 和乘方操作
def test_NegativeInfinity():
    # 断言 -oo 的 floor 方法返回自身
    assert (-oo).floor() is -oo
    # 断言 -oo 的 ceiling 方法返回自身
    assert (-oo).ceiling() is -oo
    # 断言 -oo 的乘方操作 (-oo)**11 返回 -oo
    assert (-oo)**11 is -oo
    # 断言 -oo 的乘方操作 (-oo)**12 返回 oo
    assert (-oo)**12 is oo


# 定义测试函数 test_issue_6133，测试与特定问题相关的一些比较操作
def test_issue_6133():
    # 断言对 (-oo < None) 的类型错误异常抛出
    raises(TypeError, lambda: (-oo < None))
    # 断言对 (S(-2) < None) 的类型错误异常抛出
    raises(TypeError, lambda: (S(-2) < None))
    # 断言对 (oo < None) 的类型错误异常抛出
    raises(TypeError, lambda: (oo < None))
    # 断言对 (oo > None) 的类型错误异常抛出
    raises(TypeError, lambda: (oo > None))
    # 断言对 (S(2) < None) 的类型错误异常抛出
    raises(TypeError, lambda: (S(2) < None))


# 定义测试函数 test_abc，测试 numbers 模块中的各种数值类型及其类型检查
def test_abc():
    # 创建一个浮点数对象 x，值为 5
    x = numbers.Float(5)
    # 断言 x 是 nums.Number 类型的实例
    assert(isinstance(x, nums.Number))
    # 断言 x 是 numbers.Number 类型的实例
    assert(isinstance(x, numbers.Number))
    # 断言 x 是 nums.Real 类型的实例
    assert(isinstance(x, nums.Real))
    # 创建一个有理数对象 y，分子为 1，分母为 3
    y = numbers.Rational(1, 3)
    # 断言 y 是 nums.Number 类型的实例
    assert(isinstance(y, nums.Number))
    # 断言 y 的分子为 1
    assert(y.numerator == 1)
    # 断言 y 的分母为 3
    assert(y.denominator == 3)
    # 断言 y 是 nums.Rational 类型的实例
    assert(isinstance(y, nums.Rational))
    # 创建一个整数对象 z，值为 3
    z = numbers.Integer(3)
    # 断言 z 是 nums.Number 类型的实例
    assert(isinstance(z, nums.Number))
    # 断言 z 是 numbers.Number 类型的实例
    assert(isinstance(z, numbers.Number))
    # 断言 z 是 nums.Rational 类型的实例
    assert(isinstance(z, nums.Rational))
    # 断言 z 是 numbers.Rational 类型的实例
    assert(isinstance(z, numbers.Rational))
    # 断言 z 是 nums.Integral 类型的实例
    assert(isinstance(z, nums.Integral))


# 定义测试函数 test_floordiv，测试 S(2) 对象与 S.Half（符号 1/2）的整数除法
def test_floordiv():
    # 断言 S(2) 除以 S.Half 的整数结果为 4
    assert S(2)//S.Half == 4


# 定义测试函数 test_negation，测试符号的取反操作
def test_negation():
    # 断言 -S.Zero 的结果是 S.Zero
    assert -S.Zero is S.Zero
    # 断言 -Float(0) 不是 S.Zero，且其值为 0.0
    assert -Float(0) is not S.Zero and -Float(0) == 0.0


# 定义测试函数 test_exponentiation_of_0，测试 0 的指数运算
def test_exponentiation_of_0():
    # 创建一个符号 x
    x = Symbol('x')
    # 断言 0 的 -x 次方等于 zoo 的 x 次方
    assert 0**-x == zoo**x
    # 断言在 Pow 操作下，0 的 x 次方保持不变
    assert unchanged(Pow, 0, x)
    # 创建一个零值符号 x
    x = Symbol('x', zero=True)
    # 断言 0 的 -x 次方等于 S.One
    assert 0**-x == S.One
    # 断言 0 的 x 次方等于 S.One
    assert 0**x == S.One


# 定义测试函数 test_int_valued，测试符号的整数值特性
def test_int_valued():
    # 创建一个符号 x
    x = Symbol('x')
    # 断言 x 不是整数值
    assert int_valued(x) == False
    # 断言 S.Half 不是整数值
    assert int_valued(S.Half) == False
    # 断言 S.One 是整数值
    assert int_valued(S.One) == True
    # 断言 Float(1) 是整数值
    assert int_valued(Float(1)) == True
    # 断言 Float(1.1) 不是整数值
    assert int_valued(Float(1.1)) == False
    # 断言 pi 不是整数值
    assert int_valued(pi) == False


# 定义测试函数 test_equal_valued，测试符号的相等值特性
def test_equal_valued():
    # 创建一个符号 x
    x = Symbol('x')

    # 创建一个相等值的列表
    equal_values = [
        [1, 1.0, S(1), S(1.0), S(1).n(5)],
        [2, 2.0, S(2), S(2.0), S(2).n(5)],
        [-1, -1.0, -S(1), -S(1.0), -S(1).n(5)],
        [0.5, S(0.5), S(1)/2],
        [-0.5, -S(0.5), -S(1)/2],
        [0, 0.0, S(0), S(0.0), S(0).n()],
        [pi], [pi.n()],           # <-- 不相等
        [S(1)/10], [0.1, S(0.1)], # <-- 不相等
        [S(0.1).n(5)],
        [oo],
        [cos(x/2)], [cos(0.5*x)], #
    # 遍历 equal_values 列表，m 是索引，values_m 是对应索引处的子列表
    for m, values_m in enumerate(equal_values):
        # 遍历当前子列表 values_m 中的每个值 value_i
        for value_i in values_m:

            # 断言：当前子列表 values_m 中的任意两个值都应该相等
            for value_j in values_m:
                assert equal_valued(value_i, value_j) is True

            # 检查当前值 value_i 是否不等于其它任何子列表 values_n 中的任意值
            for n, values_n in enumerate(equal_values):
                # 跳过与当前列表相同的列表
                if n == m:
                    continue
                # 在每个不同的列表 values_n 中，遍历每个值 value_j
                for value_j in values_n:
                    # 断言：当前值 value_i 与 values_n 中的任何值 value_j 应该不相等
                    assert equal_valued(value_i, value_j) is False
# 定义测试函数，用于测试数学表达式是否“接近”
def test_all_close():
    # 定义符号变量 x, y, z
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    
    # 断言两个数是否“接近”，期望为 True
    assert all_close(2, 2) is True
    # 断言两个数是否“接近”，期望为 True
    assert all_close(2, 2.0000) is True
    # 断言两个数是否“接近”，期望为 False
    assert all_close(2, 2.0001) is False
    # 断言两个数是否“接近”，期望为 False
    assert all_close(1/3, 1/3.0001) is False
    # 断言两个数是否“接近”，期望为 True
    assert all_close(1/3, 1/3.0001, 1e-3, 1e-3) is True
    # 断言两个数是否“接近”，期望为 True
    assert all_close(1/3, Rational(1, 3)) is True
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(0.1*exp(0.2*x), exp(x/5)/10) is True
    
    # 断言两个数是否“接近”，期望为 False
    assert all_close(1.4142135623730951, sqrt(2)) is False
    # 断言两个数是否“接近”，期望为 True
    assert all_close(1.4142135623730951, sqrt(2).evalf()) is True
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(x + 1e-20, x) is True
    
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(Add(1, 2, evaluate=False), Add(2, 1, evaluate=False))
    
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(2*x, 3*x)
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(2*x, 3*x, 1)
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(2*x, 3*x, 0, 0.5)
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(2*x, 3*x, 0, 1)
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(y*x, z*x)
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(2*x*exp(1.0*x), 2.0*x*exp(x))
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(2*x*exp(1.0*x), 2.0*x*exp(2.*x))
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(x + 2.*y, 1.*x + 2*y)
    # 断言两个表达式是否“接近”，期望为 True
    assert all_close(x + exp(2.*x)*y, 1.*x + exp(2*x)*y)
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(x + exp(2.*x)*y, 1.*x + 2*exp(2*x)*y)
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(x + exp(2.*x)*y, 1.*x + exp(3*x)*y)
    # 断言两个表达式是否“接近”，期望为 False
    assert not all_close(x + 2.*y, 1.*x + 3*y)
```