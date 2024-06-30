# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_inequalities.py`

```
"""Tests for tools for solving inequalities and systems of inequalities. """

# 导入必要的模块和类
from sympy.concrete.summations import Sum  # 导入求和符号类
from sympy.core.function import Function  # 导入函数类
from sympy.core.numbers import I, Rational, oo, pi  # 导入虚数单位、有理数、无穷大和圆周率常量
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Ne  # 导入相等、大于等于、大于、小于等于、小于和不等于关系运算类
from sympy.core.singleton import S  # 导入符号类
from sympy.core.symbol import Dummy, Symbol  # 导入虚拟符号和符号类
from sympy.functions.elementary.complexes import Abs  # 导入绝对值函数
from sympy.functions.elementary.exponential import exp, log  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import root, sqrt  # 导入根号和平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import cos, sin, tan  # 导入余弦、正弦和正切函数
from sympy.integrals.integrals import Integral  # 导入积分类
from sympy.logic.boolalg import And, Or  # 导入逻辑与和逻辑或运算类
from sympy.polys.polytools import Poly, PurePoly  # 导入多项式和纯多项式类
from sympy.sets.sets import FiniteSet, Interval, Union  # 导入有限集合、区间和并集类
from sympy.solvers.inequalities import (reduce_inequalities,  # 导入简化不等式、求解多项式不等式、简化有理数不等式、求解一元不等式、简化绝对值不等式和解不等式函数
                                        solve_poly_inequality as psolve,
                                        reduce_rational_inequalities,
                                        solve_univariate_inequality as isolve,
                                        reduce_abs_inequality,
                                        _solve_inequality)
from sympy.polys.rootoftools import rootof  # 导入根式工具类
from sympy.solvers.solvers import solve  # 导入解方程类
from sympy.solvers.solveset import solveset  # 导入解集类
from sympy.core.mod import Mod  # 导入模运算类
from sympy.abc import x, y  # 导入符号变量 x 和 y

from sympy.testing.pytest import raises, XFAIL  # 导入测试相关的函数

# 将无穷大转换为浮点数
inf = oo.evalf()

# 测试解多项式不等式函数
def test_solve_poly_inequality():
    assert psolve(Poly(0, x), '==') == [S.Reals]  # 解 0 == x 的多项式不等式应为全体实数集合
    assert psolve(Poly(1, x), '==') == [S.EmptySet]  # 解 1 == x 的多项式不等式应为空集
    assert psolve(PurePoly(x + 1, x), ">") == [Interval(-1, oo, True, False)]  # 解 x + 1 > 0 的多项式不等式应为区间 (-1, ∞)

# 测试简化有理数不等式函数在实数区间上的应用
def test_reduce_poly_inequalities_real_interval():
    assert reduce_rational_inequalities(
        [[Eq(x**2, 0)]], x, relational=False) == FiniteSet(0)  # 简化 x**2 == 0 的有理数不等式应为有限集 {0}
    assert reduce_rational_inequalities(
        [[Le(x**2, 0)]], x, relational=False) == FiniteSet(0)  # 简化 x**2 <= 0 的有理数不等式应为有限集 {0}
    assert reduce_rational_inequalities(
        [[Lt(x**2, 0)]], x, relational=False) == S.EmptySet  # 简化 x**2 < 0 的有理数不等式应为空集
    assert reduce_rational_inequalities(
        [[Ge(x**2, 0)]], x, relational=False) == \
        S.Reals if x.is_real else Interval(-oo, oo)  # 简化 x**2 >= 0 的有理数不等式应为实数集合（若 x 是实数）或者区间 (-∞, ∞)
    assert reduce_rational_inequalities(
        [[Gt(x**2, 0)]], x, relational=False) == \
        FiniteSet(0).complement(S.Reals)  # 简化 x**2 > 0 的有理数不等式应为全体实数集合的补集
    assert reduce_rational_inequalities(
        [[Ne(x**2, 0)]], x, relational=False) == \
        FiniteSet(0).complement(S.Reals)  # 简化 x**2 != 0 的有理数不等式应为全体实数集合的补集

    assert reduce_rational_inequalities(
        [[Eq(x**2, 1)]], x, relational=False) == FiniteSet(-1, 1)  # 简化 x**2 == 1 的有理数不等式应为有限集 {-1, 1}
    assert reduce_rational_inequalities(
        [[Le(x**2, 1)]], x, relational=False) == Interval(-1, 1)  # 简化 x**2 <= 1 的有理数不等式应为区间 [-1, 1]
    assert reduce_rational_inequalities(
        [[Lt(x**2, 1)]], x, relational=False) == Interval(-1, 1, True, True)  # 简化 x**2 < 1 的有理数不等式应为区间 (-1, 1)
    assert reduce_rational_inequalities(
        [[Ge(x**2, 1)]], x, relational=False) == \
        Union(Interval(-oo, -1), Interval(1, oo))  # 简化 x**2 >= 1 的有理数不等式应为区间 (-∞, -1) ∪ (1, ∞)
    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 > 1，并验证结果是否为区间 (-1, 1) 的补集
    assert reduce_rational_inequalities(
        [[Gt(x**2, 1)]], x, relational=False) == \
        Interval(-1, 1).complement(S.Reals)

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 != 1，并验证结果是否为有限集合 {-1, 1} 的补集
    assert reduce_rational_inequalities(
        [[Ne(x**2, 1)]], x, relational=False) == \
        FiniteSet(-1, 1).complement(S.Reals)

    # 断言：使用 reduce_rational_inequalities 函数处理等式 x**2 == 1.0，并验证结果是否为有限集合 {-1.0, 1.0} 的数值化
    assert reduce_rational_inequalities([[Eq(
        x**2, 1.0)]], x, relational=False) == FiniteSet(-1.0, 1.0).evalf()

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 <= 1.0，并验证结果是否为区间 [-1.0, 1.0]
    assert reduce_rational_inequalities(
        [[Le(x**2, 1.0)]], x, relational=False) == Interval(-1.0, 1.0)

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 < 1.0，并验证结果是否为区间 (-1.0, 1.0)
    assert reduce_rational_inequalities([[Lt(
        x**2, 1.0)]], x, relational=False) == Interval(-1.0, 1.0, True, True)

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 >= 1.0，并验证结果是否为联合区间 (-∞, -1.0) ∪ (1.0, ∞)
    assert reduce_rational_inequalities(
        [[Ge(x**2, 1.0)]], x, relational=False) == \
        Union(Interval(-inf, -1.0), Interval(1.0, inf))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 > 1.0，并验证结果是否为联合区间 (-∞, -1.0) ∪ (1.0, ∞)
    assert reduce_rational_inequalities(
        [[Gt(x**2, 1.0)]], x, relational=False) == \
        Union(Interval(-inf, -1.0, right_open=True),
        Interval(1.0, inf, left_open=True))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 != 1.0，并验证结果是否为集合 {-1.0, 1.0} 的补集
    assert reduce_rational_inequalities([[Ne(
        x**2, 1.0)]], x, relational=False) == \
        FiniteSet(-1.0, 1.0).complement(S.Reals)

    # 计算 sqrt(2) 并赋值给 s
    s = sqrt(2)

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 1 < 0 和 x**2 - 1 > 0，并验证结果是否为空集
    assert reduce_rational_inequalities([[Lt(
        x**2 - 1, 0), Gt(x**2 - 1, 0)]], x, relational=False) == S.EmptySet

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 1 <= 0 和 x**2 - 1 >= 0，并验证结果是否为集合 {-1, 1}
    assert reduce_rational_inequalities([[Le(x**2 - 1, 0), Ge(
        x**2 - 1, 0)]], x, relational=False) == FiniteSet(-1, 1)

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 2 <= 0 和 x**2 - 1 >= 0，并验证结果是否为联合区间 (-sqrt(2), -1) ∪ (1, sqrt(2))
    assert reduce_rational_inequalities(
        [[Le(x**2 - 2, 0), Ge(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, False, False), Interval(1, s, False, False))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 2 <= 0 和 x**2 - 1 > 0，并验证结果是否为联合区间 (-sqrt(2), -1] ∪ (1, sqrt(2))
    assert reduce_rational_inequalities(
        [[Le(x**2 - 2, 0), Gt(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, False, True), Interval(1, s, True, False))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 2 < 0 和 x**2 - 1 >= 0，并验证结果是否为联合区间 (-sqrt(2), -1) ∪ (1, sqrt(2))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Ge(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, False), Interval(1, s, False, True))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 2 < 0 和 x**2 - 1 > 0，并验证结果是否为联合区间 (-sqrt(2), -1) ∪ (1, sqrt(2))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, True), Interval(1, s, True, True))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 - 2 < 0 和 x**2 - 1 != 0，并验证结果是否为联合区间 (-sqrt(2), -1) ∪ (-1, 1) ∪ (1, sqrt(2))
    assert reduce_rational_inequalities(
        [[Lt(x**2 - 2, 0), Ne(x**2 - 1, 0)]], x, relational=False
        ) == Union(Interval(-s, -1, True, True), Interval(-1, 1, True, True),
        Interval(1, s, True, True))

    # 断言：使用 reduce_rational_inequalities 函数处理不等式 x**2 < -1.0，并验证结果是否为逻辑假 (S.false)
    assert reduce_rational_inequalities([[Lt(x**2, -1.)]], x) is S.false
# 定义一个测试函数，用于测试 reduce_rational_inequalities 函数处理复杂关系的情况
def test_reduce_poly_inequalities_complex_relational():
    # 测试当 x^2 等于 0 时，reduce_rational_inequalities 是否正确返回 Eq(x, 0)
    assert reduce_rational_inequalities([[Eq(x**2, 0)]], x, relational=True) == Eq(x, 0)
    
    # 测试当 x^2 小于等于 0 时，reduce_rational_inequalities 是否正确返回 Eq(x, 0)
    assert reduce_rational_inequalities([[Le(x**2, 0)]], x, relational=True) == Eq(x, 0)
    
    # 测试当 x^2 小于 0 时，reduce_rational_inequalities 是否正确返回 False
    assert reduce_rational_inequalities([[Lt(x**2, 0)]], x, relational=True) == False
    
    # 测试当 x^2 大于等于 0 时，reduce_rational_inequalities 是否正确返回 And(Lt(-oo, x), Lt(x, oo))
    assert reduce_rational_inequalities([[Ge(x**2, 0)]], x, relational=True) == And(Lt(-oo, x), Lt(x, oo))
    
    # 测试当 x^2 大于 0 时，reduce_rational_inequalities 是否正确返回 And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))
    assert reduce_rational_inequalities([[Gt(x**2, 0)]], x, relational=True) == \
        And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))
    
    # 测试当 x^2 不等于 0 时，reduce_rational_inequalities 是否正确返回 And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))
    assert reduce_rational_inequalities([[Ne(x**2, 0)]], x, relational=True) == \
        And(Gt(x, -oo), Lt(x, oo), Ne(x, 0))

    # 针对 S.One 和 S(1.0) 进行循环测试
    for one in (S.One, S(1.0)):
        inf = one*oo
        
        # 测试当 x^2 等于 one 时，reduce_rational_inequalities 是否正确返回 Or(Eq(x, -one), Eq(x, one))
        assert reduce_rational_inequalities([[Eq(x**2, one)]], x, relational=True) == \
            Or(Eq(x, -one), Eq(x, one))
        
        # 测试当 x^2 小于等于 one 时，reduce_rational_inequalities 是否正确返回 And(And(Le(-one, x), Le(x, one)))
        assert reduce_rational_inequalities([[Le(x**2, one)]], x, relational=True) == \
            And(And(Le(-one, x), Le(x, one)))
        
        # 测试当 x^2 小于 one 时，reduce_rational_inequalities 是否正确返回 And(And(Lt(-one, x), Lt(x, one)))
        assert reduce_rational_inequalities([[Lt(x**2, one)]], x, relational=True) == \
            And(And(Lt(-one, x), Lt(x, one)))
        
        # 测试当 x^2 大于等于 one 时，reduce_rational_inequalities 是否正确返回 And(Or(And(Le(one, x), Lt(x, inf)), And(Le(x, -one), Lt(-inf, x))))
        assert reduce_rational_inequalities([[Ge(x**2, one)]], x, relational=True) == \
            And(Or(And(Le(one, x), Lt(x, inf)), And(Le(x, -one), Lt(-inf, x))))
        
        # 测试当 x^2 大于 one 时，reduce_rational_inequalities 是否正确返回 And(Or(And(Lt(-inf, x), Lt(x, -one)), And(Lt(one, x), Lt(x, inf))))
        assert reduce_rational_inequalities([[Gt(x**2, one)]], x, relational=True) == \
            And(Or(And(Lt(-inf, x), Lt(x, -one)), And(Lt(one, x), Lt(x, inf))))
        
        # 测试当 x^2 不等于 one 时，reduce_rational_inequalities 是否正确返回 Or(And(Lt(-inf, x), Lt(x, -one)), And(Lt(-one, x), Lt(x, one)), And(Lt(one, x), Lt(x, inf)))
        assert reduce_rational_inequalities([[Ne(x**2, one)]], x, relational=True) == \
            Or(And(Lt(-inf, x), Lt(x, -one)),
               And(Lt(-one, x), Lt(x, one)),
               And(Lt(one, x), Lt(x, inf)))


# 定义一个测试函数，用于测试 reduce_rational_inequalities 函数处理实数关系的情况
def test_reduce_rational_inequalities_real_relational():
    # 测试空条件时，reduce_rational_inequalities 是否正确返回 False
    assert reduce_rational_inequalities([], x) == False
    
    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Union(Interval.open(-oo, -4), Interval(-2, -1), Interval.open(4, oo))
    assert reduce_rational_inequalities(
        [[(x**2 + 3*x + 2)/(x**2 - 16) >= 0]], x, relational=False) == \
        Union(Interval.open(-oo, -4), Interval(-2, -1), Interval.open(4, oo))

    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Union(Interval.open(-5, 2), Interval.open(2, 3))
    assert reduce_rational_inequalities([[( (-2*x - 10)*(3 - x))/((x**2 + 5)*(x - 2)**2) < 0]], x,
        relational=False) == \
        Union(Interval.open(-5, 2), Interval.open(2, 3))

    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Interval.Ropen(-1, 5)
    assert reduce_rational_inequalities([[(x + 1)/(x - 5) <= 0]], x,
        relational=False) == \
        Interval.Ropen(-1, 5)

    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Union(Interval.open(-3, -1), Interval.open(1, oo))
    assert reduce_rational_inequalities([[(x**2 + 4*x + 3)/(x - 1) > 0]], x,
        relational=False) == \
        Union(Interval.open(-3, -1), Interval.open(1, oo))

    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Union(Interval.open(-4, 1), Interval.open(1, 4))
    assert reduce_rational_inequalities([[(x**2 - 16)/(x - 1)**2 < 0]], x,
        relational=False) == \
        Union(Interval.open(-4, 1), Interval.open(1, 4))

    # 测试给定一个复杂的不等式条件时，reduce_rational_inequalities 是否正确返回 Union(Interval.open(-oo, -4), Interval.Ropen(Rational(3, 2), oo))
    assert reduce_rational_inequalities([[(3*x + 1)/(x + 4) >= 1]], x,
        relational=False) == \
        Union(Interval.open(-oo, -4), Interval.Ropen(Rational(3, 2), oo))
    # 断言语句，验证有理不等式的缩写结果是否符合预期
    assert reduce_rational_inequalities([[(x - 8)/x <= 3 - x]], x,
        relational=False) == \
        Union(Interval.Lopen(-oo, -2), Interval.Lopen(0, 4))
    
    # 这是一个与 sympy 问题追踪编号 sympy/sympy#10237 相关的断言
    assert reduce_rational_inequalities(
        [[x < oo, x >= 0, -oo < x]], x, relational=False) == Interval(0, oo)
# 定义一个名为 test_reduce_abs_inequalities 的测试函数
def test_reduce_abs_inequalities():
    # 创建一个表示 abs(x - 5) < 3 的表达式 e
    e = abs(x - 5) < 3
    # 用于表示 2 < x < 8 的表达式 ans
    ans = And(Lt(2, x), Lt(x, 8))
    # 断言 reduce_inequalities(e) 的结果与 ans 相等
    assert reduce_inequalities(e) == ans
    # 断言 reduce_inequalities(e, x) 的结果与 ans 相等
    assert reduce_inequalities(e, x) == ans
    # 断言 reduce_inequalities(abs(x - 5)) 的结果与 Eq(x, 5) 相等
    assert reduce_inequalities(abs(x - 5)) == Eq(x, 5)
    # 断言 reduce_inequalities(abs(2*x + 3) >= 8) 的结果与 Or(...) 相等
    assert reduce_inequalities(
        abs(2*x + 3) >= 8) == Or(And(Le(Rational(5, 2), x), Lt(x, oo)),
        And(Le(x, Rational(-11, 2)), Lt(-oo, x)))
    # 断言 reduce_inequalities(abs(x - 4) + abs(3*x - 5) < 7) 的结果与 And(...) 相等
    assert reduce_inequalities(abs(x - 4) + abs(
        3*x - 5) < 7) == And(Lt(S.Half, x), Lt(x, 4))
    # 断言 reduce_inequalities(abs(x - 4) + abs(3*abs(x) - 5) < 7) 的结果与 Or(...) 相等
    assert reduce_inequalities(abs(x - 4) + abs(3*abs(x) - 5) < 7) == \
        Or(And(S(-2) < x, x < -1), And(S.Half < x, x < 4))

    # 创建一个非实数符号 nr
    nr = Symbol('nr', extended_real=False)
    # 使用 lambda 函数测试 reduce_inequalities(abs(nr - 5) < 3) 是否引发 TypeError
    raises(TypeError, lambda: reduce_inequalities(abs(nr - 5) < 3))
    # 断言 reduce_inequalities(x < 3, symbols=[x, nr]) 的结果与 And(-oo < x, x < 3) 相等
    assert reduce_inequalities(x < 3, symbols=[x, nr]) == And(-oo < x, x < 3)


# 定义一个名为 test_reduce_inequalities_general 的测试函数
def test_reduce_inequalities_general():
    # 断言 reduce_inequalities(Ge(sqrt(2)*x, 1)) 的结果与 And(sqrt(2)/2 <= x, x < oo) 相等
    assert reduce_inequalities(Ge(sqrt(2)*x, 1)) == And(sqrt(2)/2 <= x, x < oo)
    # 断言 reduce_inequalities(x + 1 > 0) 的结果与 And(S.NegativeOne < x, x < oo) 相等
    assert reduce_inequalities(x + 1 > 0) == And(S.NegativeOne < x, x < oo)


# 定义一个名为 test_reduce_inequalities_boolean 的测试函数
def test_reduce_inequalities_boolean():
    # 断言 reduce_inequalities([Eq(x**2, 0), True]) 的结果与 Eq(x, 0) 相等
    assert reduce_inequalities([Eq(x**2, 0), True]) == Eq(x, 0)
    # 断言 reduce_inequalities([Eq(x**2, 0), False]) 的结果为 False
    assert reduce_inequalities([Eq(x**2, 0), False]) == False
    # 断言 reduce_inequalities(x**2 >= 0) 的结果为 S.true
    assert reduce_inequalities(x**2 >= 0) is S.true  # issue 10196


# 定义一个名为 test_reduce_inequalities_multivariate 的测试函数
def test_reduce_inequalities_multivariate():
    # 断言 reduce_inequalities([Ge(x**2, 1), Ge(y**2, 1)]) 的结果与 And(...) 相等
    assert reduce_inequalities([Ge(x**2, 1), Ge(y**2, 1)]) == And(
        Or(And(Le(S.One, x), Lt(x, oo)), And(Le(x, -1), Lt(-oo, x))),
        Or(And(Le(S.One, y), Lt(y, oo)), And(Le(y, -1), Lt(-oo, y))))


# 定义一个名为 test_reduce_inequalities_errors 的测试函数
def test_reduce_inequalities_errors():
    # 使用 lambda 函数测试 reduce_inequalities(Ge(sin(x) + x, 1)) 是否引发 NotImplementedError
    raises(NotImplementedError, lambda: reduce_inequalities(Ge(sin(x) + x, 1)))
    # 使用 lambda 函数测试 reduce_inequalities(Ge(x**2*y + y, 1)) 是否引发 NotImplementedError
    raises(NotImplementedError, lambda: reduce_inequalities(Ge(x**2*y + y, 1)))


# 定义一个名为 test__solve_inequalities 的测试函数
def test__solve_inequalities():
    # 断言 reduce_inequalities(x + y < 1, symbols=[x]) 的结果与 (x < 1 - y) 相等
    assert reduce_inequalities(x + y < 1, symbols=[x]) == (x < 1 - y)
    # 断言 reduce_inequalities(x + y >= 1, symbols=[x]) 的结果与 (x < oo) & (x >= -y + 1) 相等
    assert reduce_inequalities(x + y >= 1, symbols=[x]) == (x < oo) & (x >= -y + 1)
    # 断言 reduce_inequalities(Eq(0, x - y), symbols=[x]) 的结果与 Eq(x, y) 相等
    assert reduce_inequalities(Eq(0, x - y), symbols=[x]) == Eq(x, y)
    # 断言 reduce_inequalities(Ne(0, x - y), symbols=[x]) 的结果与 Ne(x, y) 相等
    assert reduce_inequalities(Ne(0, x - y), symbols=[x]) == Ne(x, y)


# 定义一个名为 test_issue_6343 的测试函数
def test_issue_6343():
    # 定义一个复杂的不等式表达式 eq
    eq = -3*x**2/2 - x*Rational(45, 4) + Rational(33, 2) > 0
    # 断言 reduce_inequalities(eq) 的结果与 And(...) 相等
    assert reduce_inequalities(eq) == \
        And(x < Rational(-15, 4) + sqrt(401)/4, -sqrt(401)/4 - Rational(15, 4) < x)


# 定义一个名为 test_issue_8235 的测试函数
def test_issue_8235():
    # 断言 reduce_inequalities(x**2 - 1 < 0) 的结果与 And(S.NegativeOne < x, x < 1) 相等
    assert reduce_inequalities(x**2 - 1 < 0) == \
        And(S.NegativeOne < x, x < 1)
    # 断言 reduce_inequalities(x**2 - 1 <= 0) 的结果与 And(S.NegativeOne <= x, x <= 1) 相等
    assert reduce_inequalities(x**2 - 1 <= 0) == \
        And(S.NegativeOne <= x, x <= 1)
    # 断言 reduce_inequalities(x**2 - 1 > 0) 的结果与 Or(...) 相等
    assert reduce_inequalities(x**2 - 1 > 0) == \
        Or(And(-oo < x, x < -1), And(x < oo, S.One < x))
    # 断言 reduce_inequalities(x**2 - 1 >= 0) 的结果与 Or(...) 相等
    assert reduce_inequalities(x**2 - 1 >= 0) == \
        Or(And(-oo < x, x <= -1), And(S.One <= x, x < oo))

    # 定义一个等式表达式 eq
    eq = x**8 + x - 9  # we want CRootOf solns here
    # 解方程 eq >= 0 的结果
    sol = solve(eq >= 0)
    # 预期的结果 tru
    tru = Or(And(rootof(eq, 1) <= x, x < oo), And(-oo < x, x <= rootof(eq, 0)))
    #
    # 使用 SymPy 库中的 reduce_inequalities 函数，对不等式进行简化和比较
    assert reduce_inequalities(0 <=
        x + Integral(y**2, (y, 1, 3)) - 1, [x]) == \
        (x >= -Integral(y**2, (y, 1, 3)) + 1)
    
    # 创建一个函数 'f' 的符号对象
    f = Function('f')
    
    # 计算函数 f(x) 在 x 从 1 到 3 的和
    e = Sum(f(x), (x, 1, 3))
    
    # 使用 SymPy 库中的 reduce_inequalities 函数，对不等式进行简化和比较
    assert reduce_inequalities(0 <= x + e + y**2, [x]) == \
        (x >= -y**2 - Sum(f(x), (x, 1, 3)))
def test_solve_univariate_inequality():
    # 检查 x^2 >= 4 的解，返回包含所有解的并集
    assert isolve(x**2 >= 4, x, relational=False) == Union(Interval(-oo, -2),
        Interval(2, oo))
    # 检查 x^2 >= 4 的解，返回每个子表达式的或逻辑表达式
    assert isolve(x**2 >= 4, x) == Or(And(Le(2, x), Lt(x, oo)), And(Le(x, -2),
        Lt(-oo, x)))
    # 检查 (x-1)*(x-2)*(x-3) >= 0 的解，返回包含所有解的并集
    assert isolve((x - 1)*(x - 2)*(x - 3) >= 0, x, relational=False) == \
        Union(Interval(1, 2), Interval(3, oo))
    # 检查 (x-1)*(x-2)*(x-3) >= 0 的解，返回每个子表达式的或逻辑表达式
    assert isolve((x - 1)*(x - 2)*(x - 3) >= 0, x) == \
        Or(And(Le(1, x), Le(x, 2)), And(Le(3, x), Lt(x, oo)))
    # 在给定域内，检查 (x-1)*(x-2)*(x-4) < 0 的解，返回或逻辑表达式
    assert isolve((x - 1)*(x - 2)*(x - 4) < 0, x, domain = FiniteSet(0, 3)) == \
        Or(Eq(x, 0), Eq(x, 3))
    # 检查 x^3 - 2*x - 1 > 0 的解，返回包含所有解的并集
    assert isolve(x**3 - 2*x - 1 > 0, x, relational=False) == \
        Union(Interval(-1, -sqrt(5)/2 + S.Half, True, True),
              Interval(S.Half + sqrt(5)/2, oo, True, True))
    # 检查 x^3 - x^2 + x - 1 > 0 的解，返回一个区间
    assert isolve(x**3 - x**2 + x - 1 > 0, x, relational=False) == \
        Interval(1, oo, True)
    # 检查 (x + I)*(x + 2*I) < 0 的解，返回等式 x = 0
    assert isolve((x + I)*(x + 2*I) < 0, x) == Eq(x, 0)
    # 检查 (((x-1)*(x-2)+I)*((x-1)*(x-2)+2*I)) / (x-2) > 0 的解，返回等式 x = 1
    assert isolve((((x - 1)*(x - 2) + I)*((x - 1)*(x - 2) + 2*I))/(x - 2) > 0, x) == Eq(x, 1)
    # 抛出 ValueError 异常，因为 isolve((x**2 - 3*x*I + 2)/x < 0, x) 尚未实现
    raises (ValueError, lambda: isolve((x**2 - 3*x*I + 2)/x < 0, x))

    # 数值测试在 valid() 中是必要的
    # 检查 x^7 - x - 2 > 0 的解，返回一个逻辑与表达式
    assert isolve(x**7 - x - 2 > 0, x) == \
        And(rootof(x**7 - x - 2, 0) < x, x < oo)

    # 处理分子和分母；尽管这些将被视为有理不等式，这些测试确认在域是 EX 时（例如用 sqrt(2) 替换 2）正确处理
    # 检查 1/(x-2) > 0 的解，返回一个逻辑与表达式
    assert isolve(1/(x - 2) > 0, x) == And(S(2) < x, x < oo)
    # 展开分母后，检查 (x-1)/den <= 0 的解，返回一个逻辑与表达式
    den = ((x - 1)*(x - 2)).expand()
    assert isolve((x - 1)/den <= 0, x) == \
        (x > -oo) & (x < 2) & Ne(x, 1)

    n = Dummy('n')
    # 抛出 NotImplementedError 异常，因为尚未实现对 Abs(x) <= n 的求解
    raises(NotImplementedError, lambda: isolve(Abs(x) <= n, x, relational=False))
    c1 = Dummy("c1", positive=True)
    # 抛出 NotImplementedError 异常，因为尚未实现对 n/c1 < 0 的求解
    raises(NotImplementedError, lambda: isolve(n/c1 < 0, c1))
    n = Dummy('n', negative=True)
    # 检查 n/c1 > -2 的解，返回一个逻辑表达式
    assert isolve(n/c1 > -2, c1) == (-n/2 < c1)
    # 检查 n/c1 < 0 的解，返回 True
    assert isolve(n/c1 < 0, c1) == True
    # 检查 n/c1 > 0 的解，返回 False
    assert isolve(n/c1 > 0, c1) == False

    zero = cos(1)**2 + sin(1)**2 - 1
    # 抛出 NotImplementedError 异常，因为尚未实现对 x^2 < zero 的求解
    raises(NotImplementedError, lambda: isolve(x**2 < zero, x))
    # 抛出 NotImplementedError 异常，因为尚未实现对 x^2 < zero*I 的求解
    raises(NotImplementedError, lambda: isolve(
        x**2 < zero*I, x))
    # 抛出 NotImplementedError 异常，因为尚未实现对 1/(x-y) < 2 的求解
    raises(NotImplementedError, lambda: isolve(1/(x - y) < 2, x))
    # 抛出 NotImplementedError 异常，因为尚未实现对 1/(x-y) < 0 的求解
    raises(NotImplementedError, lambda: isolve(1/(x - y) < 0, x))
    # 抛出 TypeError 异常，因为无法解析 x - I < 0 的比较
    raises(TypeError, lambda: isolve(x - I < 0, x))

    zero = x**2 + x - x*(x + 1)
    # 检查 zero < 0 的解，返回 S.EmptySet 空集
    assert isolve(zero < 0, x, relational=False) is S.EmptySet
    # 检查 zero <= 0 的解，返回 S.Reals 实数集
    assert isolve(zero <= 0, x, relational=False) is S.Reals

    # 确保 iter_solutions 得到一个默认值
    raises(NotImplementedError, lambda: isolve(
        Eq(cos(x)**2 + sin(x)**2, 1), x))


def test_trig_inequalities():
    # 所有三角不等式在一个周期区间内求解。
    # 断言：求解 sin(x) < 1/2 的不等式，返回结果为区间并集
    assert isolve(sin(x) < S.Half, x, relational=False) == \
        Union(Interval(0, pi/6, False, True), Interval.open(pi*Rational(5, 6), 2*pi))
    
    # 断言：求解 sin(x) > 1/2 的不等式，返回结果为单个区间
    assert isolve(sin(x) > S.Half, x, relational=False) == \
        Interval(pi/6, pi*Rational(5, 6), True, True)
    
    # 断言：求解 cos(x) < 0 的不等式，返回结果为单个区间
    assert isolve(cos(x) < S.Zero, x, relational=False) == \
        Interval(pi/2, pi*Rational(3, 2), True, True)
    
    # 断言：求解 cos(x) >= 0 的不等式，返回结果为区间并集
    assert isolve(cos(x) >= S.Zero, x, relational=False) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    
    # 断言：求解 tan(x) < 1 的不等式，返回结果为区间并集
    assert isolve(tan(x) < S.One, x, relational=False) == \
        Union(Interval.Ropen(0, pi/4), Interval.open(pi/2, pi))
    
    # 断言：求解 sin(x) <= 0 的不等式，返回结果为区间并集和有限集的并集
    assert isolve(sin(x) <= S.Zero, x, relational=False) == \
        Union(FiniteSet(S.Zero), Interval.Ropen(pi, 2*pi))
    
    # 断言：求解 sin(x) <= 1 的不等式，返回结果为全体实数集合
    assert isolve(sin(x) <= S.One, x, relational=False) == S.Reals
    
    # 断言：求解 cos(x) < -2 的不等式，返回结果为空集
    assert isolve(cos(x) < S(-2), x, relational=False) == S.EmptySet
    
    # 断言：求解 sin(x) >= -1 的不等式，返回结果为全体实数集合
    assert isolve(sin(x) >= S.NegativeOne, x, relational=False) == S.Reals
    
    # 断言：求解 cos(x) > 1 的不等式，返回结果为空集
    assert isolve(cos(x) > S.One, x, relational=False) == S.EmptySet
# 定义测试函数 test_issue_9954，验证对不同形式的不等式求解结果是否正确
def test_issue_9954():
    # 检查非关系型求解 x**2 >= 0，期望结果是实数集 S.Reals
    assert isolve(x**2 >= 0, x, relational=False) == S.Reals
    # 检查关系型求解 x**2 >= 0，期望结果是实数集的关系形式
    assert isolve(x**2 >= 0, x, relational=True) == S.Reals.as_relational(x)
    # 检查非关系型求解 x**2 < 0，期望结果是空集 S.EmptySet
    assert isolve(x**2 < 0, x, relational=False) == S.EmptySet
    # 检查关系型求解 x**2 < 0，期望结果是空集的关系形式
    assert isolve(x**2 < 0, x, relational=True) == S.EmptySet.as_relational(x)


# 标记为预期失败的测试函数，测试解决非常慢的一般单变量问题
@XFAIL
def test_slow_general_univariate():
    r = rootof(x**5 - x**2 + 1, 0)
    # 检查求解 sqrt(x) + 1/root(x, 3) > 1 的结果
    assert solve(sqrt(x) + 1/root(x, 3) > 1) == \
        Or(And(0 < x, x < r**6), And(r**6 < x, x < oo))


# 测试函数 test_issue_8545，验证对绝对值和平方根不等式简化的正确性
def test_issue_8545():
    eq = 1 - x - abs(1 - x)
    ans = And(Lt(1, x), Lt(x, oo))
    # 验证对绝对值不等式 -1 + 1/abs(1/x - 1) < 0 的简化结果
    assert reduce_abs_inequality(eq, '<', x) == ans
    eq = 1 - x - sqrt((1 - x)**2)
    # 验证对 sqrt((1 - x)**2) - x < 0 的简化结果
    assert reduce_inequalities(eq < 0) == ans


# 测试函数 test_issue_8974，验证对无穷大不等式求解的正确性
def test_issue_8974():
    # 检查求解 -oo < x 的结果，期望结果是 -oo < x < oo
    assert isolve(-oo < x, x) == And(-oo < x, x < oo)
    # 检查求解 oo > x 的结果，期望结果是 -oo < x < oo
    assert isolve(oo > x, x) == And(-oo < x, x < oo)


# 测试函数 test_issue_10198，验证对复杂不等式简化的正确性
def test_issue_10198():
    # 验证对 -1 + 1/abs(1/x - 1) < 0 的简化结果
    assert reduce_inequalities(
        -1 + 1/abs(1/x - 1) < 0) == (x > -oo) & (x < S(1)/2) & Ne(x, 0)

    # 验证对 abs(1/sqrt(x)) - 1 的简化结果
    assert reduce_inequalities(abs(1/sqrt(x)) - 1, x) == Eq(x, 1)

    # 验证对 -3 + 1/abs(1 - 1/x) < 0 的简化结果
    assert reduce_abs_inequality(-3 + 1/abs(1 - 1/x), '<', x) == \
        Or(And(-oo < x, x < 0),
           And(S.Zero < x, x < Rational(3, 4)), And(Rational(3, 2) < x, x < oo))
    
    # 预期抛出 ValueError 的情况，验证对 -3 + 1/abs(1 - 1/sqrt(x)) < 0 的处理
    raises(ValueError, lambda: reduce_abs_inequality(-3 + 1/abs(
        1 - 1/sqrt(x)), '<', x))


# 测试函数 test_issue_10047，验证对 sin 不等式求解的正确性
def test_issue_10047():
    # 验证对 sin(x) < 2 的求解结果
    assert solve(sin(x) < 2) == True
    # 验证对 sin(x) < 2 的求解结果，限定在实数集内
    assert solveset(sin(x) < 2, domain=S.Reals) == S.Reals


# 测试函数 test_issue_10268，验证对对数不等式求解的正确性
def test_issue_10268():
    # 验证对 log(x) < 1000 的求解结果
    assert solve(log(x) < 1000) == And(S.Zero < x, x < exp(1000))


# 标记为预期失败的测试函数，测试绝对值不等式求解
@XFAIL
def test_isolve_Sets():
    n = Dummy('n')
    # 验证对 Abs(x) <= n 的求解结果
    assert isolve(Abs(x) <= n, x, relational=False) == \
        Piecewise((S.EmptySet, n < 0), (Interval(-n, n), True))


# 测试函数 test_integer_domain_relational_isolve，验证对整数域关系型不等式求解的正确性
def test_integer_domain_relational_isolve():

    dom = FiniteSet(0, 3)
    x = Symbol('x', zero=False)
    # 验证对 (x - 1)*(x - 2)*(x - 4) < 0 在整数集合 {0, 3} 上的求解结果
    assert isolve((x - 1)*(x - 2)*(x - 4) < 0, x, domain=dom) == Eq(x, 3)

    x = Symbol('x')
    # 验证对 x + 2 < 0 在整数集合上的求解结果
    assert isolve(x + 2 < 0, x, domain=S.Integers) == \
           (x <= -3) & (x > -oo) & Eq(Mod(x, 1), 0)
    # 验证对 2 * x + 3 > 0 在整数集合上的求解结果
    assert isolve(2 * x + 3 > 0, x, domain=S.Integers) == \
           (x >= -1) & (x < oo)  & Eq(Mod(x, 1), 0)
    # 验证对 x ** 2 + 3 * x - 2 < 0 在整数集合上的求解结果
    assert isolve((x ** 2 + 3 * x - 2) < 0, x, domain=S.Integers) == \
           (x >= -3) & (x <= 0)  & Eq(Mod(x, 1), 0)
    # 验证对 x ** 2 + 3 * x - 2 > 0 在整数集合上的求解结果
    assert isolve((x ** 2 + 3 * x - 2) > 0, x, domain=S.Integers) == \
           ((x >= 1) & (x < oo)  & Eq(Mod(x, 1), 0)) | (
               (x <= -4) & (x > -oo)  & Eq(Mod(x, 1), 0))


# 测试函数 test_issue_10671_12466，验证对 solve 函数的应用
def test_issue_10671_12466():
    # 验证对 sin(y) 在区间 [0, pi] 上的求解结果
    assert solveset(sin(y), y, Interval(0, pi)) == FiniteSet(0, pi)
    i = Interval(1, 10)
    # 验证对 (1/x).diff(x) < 0 在区间 [1, 10] 上的求解结果
    assert solveset((1/x).diff(x) < 0, x, i) == i
    # 验证对 (log(x - 6)/x) <= 0 在实数集上的求解结果
    assert solveset((log(x - 6)/x) <= 0, x, S.Reals) == Interval.Lopen(6, 7)
def test__solve_inequality():
    # 对于每个比较操作符 op 进行测试
    for op in (Gt, Lt, Le, Ge, Eq, Ne):
        # 断言：解决不等式 op(x, 1)，并验证左手边是否为 x
        assert _solve_inequality(op(x, 1), x).lhs == x
        # 断言：解决不等式 op(S.One, x)，并验证左手边是否为 x
        assert _solve_inequality(op(S.One, x), x).lhs == x
    # 断言：解决方程 2*x - 1 = x，预期结果为解 x = 1
    assert _solve_inequality(Eq(2*x - 1, x), x) == Eq(x, 1)
    # ie 是等式 S.One = y
    ie = Eq(S.One, y)
    # 断言：解决不等式 ie，预期结果应该与 ie 相同
    assert _solve_inequality(ie, x) == ie
    # 对于 fx 中的每个表达式进行测试
    for fx in (x**2, exp(x), sin(x) + cos(x), x*(1 + x)):
        # 对于每个常数 c 进行测试
        for c in (0, 1):
            # 构造不等式 e = 2*fx - c > 0
            e = 2*fx - c > 0
            # 断言：解决不等式 e，当 linear=True 时，预期结果应为 fx > c/2
            assert _solve_inequality(e, x, linear=True) == (fx > c/S(2))
    # 断言：解决不等式 2*x**2 + 2*x - 1 < 0，当 linear=True 时，预期结果应为 x*(x + 1) < 1/2
    assert _solve_inequality(2*x**2 + 2*x - 1 < 0, x, linear=True) == (x*(x + 1) < S.Half)
    # 断言：解决方程 Eq(x*y, 1)，预期结果应与原方程相同
    assert _solve_inequality(Eq(x*y, 1), x) == Eq(x*y, 1)
    # nz 是非零的符号变量
    nz = Symbol('nz', nonzero=True)
    # 断言：解决方程 Eq(x*nz, 1)，预期结果为 Eq(x, 1/nz)
    assert _solve_inequality(Eq(x*nz, 1), x) == Eq(x, 1/nz)
    # 断言：解决不等式 x*nz < 1，预期结果应为 x*nz < 1
    assert _solve_inequality(x*nz < 1, x) == (x*nz < 1)
    # a 是正的符号变量
    a = Symbol('a', positive=True)
    # 断言：解决不等式 a/x > 1，预期结果应为 (0 < x) & (x < a)
    assert _solve_inequality(a/x > 1, x) == (S.Zero < x) & (x < a)
    # 断言：解决不等式 a/x > 1，当 linear=True 时，预期结果应为 1/x > 1/a
    assert _solve_inequality(a/x > 1, x, linear=True) == (1/x > 1/a)
    # 构造等式 e = Eq(1 - x, x*(1/x - 1))
    e = Eq(1 - x, x*(1/x - 1))
    # 断言：解决不等式 e，预期结果应为 x ≠ 0
    assert _solve_inequality(e, x) == Ne(x, 0)
    # 断言：解决不等式 x < x*(1/x - 1)，预期结果应为 (x < 1/2) & (x ≠ 0)
    assert _solve_inequality(x < x*(1/x - 1), x) == (x < S.Half) & Ne(x, 0)


def test__pt():
    # 导入 _pt 函数并进行测试
    from sympy.solvers.inequalities import _pt
    assert _pt(-oo, oo) == 0
    assert _pt(S.One, S(3)) == 2
    assert _pt(S.One, oo) == _pt(oo, S.One) == 2
    assert _pt(S.One, -oo) == _pt(-oo, S.One) == S.Half
    assert _pt(S.NegativeOne, oo) == _pt(oo, S.NegativeOne) == Rational(-1, 2)
    assert _pt(S.NegativeOne, -oo) == _pt(-oo, S.NegativeOne) == -2
    assert _pt(x, oo) == _pt(oo, x) == x + 1
    assert _pt(x, -oo) == _pt(-oo, x) == x - 1
    # 断言：调用 _pt(Dummy('i', infinite=True), S.One) 时应引发 ValueError 异常
    raises(ValueError, lambda: _pt(Dummy('i', infinite=True), S.One))


def test_issue_25697():
    # 断言：解决不等式 log(x, 3) <= 2，预期结果应为 (x <= 9) & (0 < x)
    assert _solve_inequality(log(x, 3) <= 2, x) == (x <= 9) & (S.Zero < x)


def test_issue_25738():
    # 断言：调用 reduce_inequalities(3 < abs(x)) 的结果应与 reduce_inequalities(pi < abs(x)).subs(pi, 3) 的结果相同
    assert reduce_inequalities(3 < abs(x)) == reduce_inequalities(pi < abs(x)).subs(pi, 3)


def test_issue_25983():
    # 断言：解决不等式 pi/Abs(x) <= 1 的结果应为 ((pi <= x) & (x < oo)) | ((-oo < x) & (x <= -pi))
    assert(reduce_inequalities(pi/Abs(x) <= 1) == ((pi <= x) & (x < oo)) | ((-oo < x) & (x <= -pi)))
```