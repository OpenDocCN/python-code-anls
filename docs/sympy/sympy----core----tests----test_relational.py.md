# `D:\src\scipysrc\sympy\sympy\core\tests\test_relational.py`

```
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.multipledispatch import dispatch
from sympy.testing.pytest import XFAIL, raises
from sympy.assumptions.ask import Q
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr, unchanged
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign, Abs
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (And, Implies, Not, Or, Xor)
from sympy.sets import Reals
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.core.relational import (Relational, Equality, Unequality,
                                   GreaterThan, LessThan, StrictGreaterThan,
                                   StrictLessThan, Rel, Eq, Lt, Le,
                                   Gt, Ge, Ne, is_le, is_gt, is_ge, is_lt, is_eq, is_neq)
from sympy.sets.sets import Interval, FiniteSet

from itertools import combinations

x, y, z, t = symbols('x,y,z,t')

# 定义函数 rel_check，用于检查两个数学表达式 a 和 b 的关系
def rel_check(a, b):
    from sympy.testing.pytest import raises
    # 断言 a 和 b 必须是数值类型
    assert a.is_number and b.is_number
    # 对于指定次数的循环
    for do in range(len({type(a), type(b)})):
        # 如果 a 或 b 中存在 NaN，执行下面的断言
        if S.NaN in (a, b):
            v = [(a == b), (a != b)]
            # 断言 v 中所有值相同且为 False
            assert len(set(v)) == 1 and v[0] == False
            # 断言 a 不等于 b，同时 b 也不等于 a
            assert not (a != b) and not (a == b)
            # 断言比较 a 和 b 会引发 TypeError
            assert raises(TypeError, lambda: a < b)
            assert raises(TypeError, lambda: a <= b)
            assert raises(TypeError, lambda: a > b)
            assert raises(TypeError, lambda: a >= b)
        else:
            E = [(a == b), (a != b)]
            # 断言 E 中的两个值不相同
            assert len(set(E)) == 2
            v = [
            (a < b), (a <= b), (a > b), (a >= b)]
            i = [
            [True,    True,     False,   False],
            [False,   True,     False,   True], # <-- i == 1
            [False,   False,    True,    True]].index(v)
            # 根据 v 的值确定 i 的索引位置，执行相应的断言
            if i == 1:
                assert E[0] or (a.is_Float != b.is_Float) # ugh
            else:
                assert E[1]
        # 交换 a 和 b 的值，进行下一次循环的比较
        a, b = b, a
    # 函数返回 True 表示比较成功
    return True


# 定义函数 test_rel_ne，用于测试不等关系的符号处理
def test_rel_ne():
    # 断言 Relational(x, y, '!=') 等同于 Ne(x, y)
    assert Relational(x, y, '!=') == Ne(x, y)

    # 对 issue 6116 的特殊情况进行测试
    p = Symbol('p', positive=True)
    assert Ne(p, 0) is S.true


# 定义函数 test_rel_subs，用于测试关系符号的替换操作
def test_rel_subs():
    # 创建关系表达式 Relational(x, y, '==')
    e = Relational(x, y, '==')
    # 对 x 进行替换操作，生成新的表达式 e
    e = e.subs(x, z)

    # 断言 e 是 Equality 类型的对象
    assert isinstance(e, Equality)
    # 断言 e 的左操作数是 z
    assert e.lhs == z
    # 断言 e 的右操作数是 y

    e = Relational(x, y, '>=')
    e = e.subs(x, z)

    # 断言 e 是 GreaterThan 类型的对象
    assert isinstance(e, GreaterThan)
    # 断言 e 的左操作数是 z
    assert e.lhs == z
    # 断言 e 的右操作数是 y
    # 断言左手边(lhs)表达式等于 z
    assert e.lhs == z
    # 断言右手边(rhs)表达式等于 y
    assert e.rhs == y
    
    # 创建一个关系表达式 e，表示 x <= y
    e = Relational(x, y, '<=')
    # 用 z 替换 e 中的变量 x
    e = e.subs(x, z)
    
    # 断言 e 是 LessThan 类型的实例
    assert isinstance(e, LessThan)
    # 断言 e 的左手边(lhs)表达式等于 z
    assert e.lhs == z
    # 断言 e 的右手边(rhs)表达式等于 y
    assert e.rhs == y
    
    # 创建一个关系表达式 e，表示 x > y
    e = Relational(x, y, '>')
    # 用 z 替换 e 中的变量 x
    e = e.subs(x, z)
    
    # 断言 e 是 StrictGreaterThan 类型的实例
    assert isinstance(e, StrictGreaterThan)
    # 断言 e 的左手边(lhs)表达式等于 z
    assert e.lhs == z
    # 断言 e 的右手边(rhs)表达式等于 y
    assert e.rhs == y
    
    # 创建一个关系表达式 e，表示 x < y
    e = Relational(x, y, '<')
    # 用 z 替换 e 中的变量 x
    e = e.subs(x, z)
    
    # 断言 e 是 StrictLessThan 类型的实例
    assert isinstance(e, StrictLessThan)
    # 断言 e 的左手边(lhs)表达式等于 z
    assert e.lhs == z
    # 断言 e 的右手边(rhs)表达式等于 y
    assert e.rhs == y
    
    # 创建一个等式表达式 e，表示 x == 0
    e = Eq(x, 0)
    # 断言用 x 替换 e 中的变量 x 后结果为 S.true
    assert e.subs(x, 0) is S.true
    # 断言用 x 替换 e 中的变量 x 后结果为 S.false
    assert e.subs(x, 1) is S.false
# 定义一个测试函数，用于测试关系运算符的行为
def test_wrappers():
    # 计算表达式 e = x + x**2
    e = x + x**2

    # 测试 Relational 类的 '==' 操作符，比较 y 和 e 是否相等
    res = Relational(y, e, '==')
    assert Rel(y, x + x**2, '==') == res
    # 同样测试 Eq 类的 '==' 操作符，期望结果与上一行相同
    assert Eq(y, x + x**2) == res

    # 测试 Relational 类的 '<' 操作符，比较 y 是否小于 e
    res = Relational(y, e, '<')
    assert Lt(y, x + x**2) == res

    # 测试 Relational 类的 '<=' 操作符，比较 y 是否小于等于 e
    res = Relational(y, e, '<=')
    assert Le(y, x + x**2) == res

    # 测试 Relational 类的 '>' 操作符，比较 y 是否大于 e
    res = Relational(y, e, '>')
    assert Gt(y, x + x**2) == res

    # 测试 Relational 类的 '>=' 操作符，比较 y 是否大于等于 e
    res = Relational(y, e, '>=')
    assert Ge(y, x + x**2) == res

    # 测试 Relational 类的 '!=' 操作符，比较 y 是否不等于 e
    res = Relational(y, e, '!=')
    assert Ne(y, x + x**2) == res


def test_Eq_Ne():
    # 测试 Eq 类的 '==' 操作符，比较 x 是否等于自身
    assert Eq(x, x)  # issue 5719

    # 测试 Eq 类的 '==' 操作符，比较正数符号变量 p 是否等于 0，预期结果为 S.false
    p = Symbol('p', positive=True)
    assert Eq(p, 0) is S.false

    # 测试 Eq 类的 '==' 操作符，比较 True 和 1 是否相等，预期结果为 S.false
    assert Eq(True, 1) is S.false
    # 测试 Eq 类的 '==' 操作符，比较 False 和 0 是否相等，预期结果为 S.false
    assert Eq(False, 0) is S.false
    # 测试 Eq 类的 '==' 操作符，比较 ~x 和 0 是否相等，预期结果为 S.false
    assert Eq(~x, 0) is S.false
    # 测试 Eq 类的 '==' 操作符，比较 ~x 和 1 是否相等，预期结果为 S.false
    assert Eq(~x, 1) is S.false
    # 测试 Ne 类的 '!=' 操作符，比较 True 和 1 是否不等，预期结果为 S.true
    assert Ne(True, 1) is S.true
    # 测试 Ne 类的 '!=' 操作符，比较 False 和 0 是否不等，预期结果为 S.true
    assert Ne(False, 0) is S.true
    # 测试 Ne 类的 '!=' 操作符，比较 ~x 和 0 是否不等，预期结果为 S.true
    assert Ne(~x, 0) is S.true
    # 测试 Ne 类的 '!=' 操作符，比较 ~x 和 1 是否不等，预期结果为 S.true
    assert Ne(~x, 1) is S.true

    # 测试 Eq 类的 '==' 操作符，比较空元组 () 和 1 是否相等，预期结果为 S.false
    assert Eq((), 1) is S.false
    # 测试 Ne 类的 '!=' 操作符，比较空元组 () 和 1 是否不等，预期结果为 S.true
    assert Ne((), 1) is S.true


def test_as_poly():
    from sympy.polys.polytools import Poly
    # 只有 Eq 类应该具有 as_poly 方法，测试其返回结果是否符合预期
    assert Eq(x, 1).as_poly() == Poly(x - 1, x, domain='ZZ')
    # 测试 Ne 类没有 as_poly 方法是否会引发 AttributeError
    raises(AttributeError, lambda: Ne(x, 1).as_poly())
    # 测试 Ge 类没有 as_poly 方法是否会引发 AttributeError
    raises(AttributeError, lambda: Ge(x, 1).as_poly())
    # 测试 Gt 类没有 as_poly 方法是否会引发 AttributeError
    raises(AttributeError, lambda: Gt(x, 1).as_poly())
    # 测试 Le 类没有 as_poly 方法是否会引发 AttributeError
    raises(AttributeError, lambda: Le(x, 1).as_poly())
    # 测试 Lt 类没有 as_poly 方法是否会引发 AttributeError
    raises(AttributeError, lambda: Lt(x, 1).as_poly())


def test_rel_Infinity():
    # 注意：这些比较实际上由 sympy.core.Number 处理，并不创建 Relational 对象
    # 测试无穷大与自身比较是否为 False
    assert (oo > oo) is S.false
    # 测试无穷大与负无穷大比较是否为 True
    assert (oo > -oo) is S.true
    # 测试无穷大与常数 1 比较是否为 True
    assert (oo > 1) is S.true
    # 测试无穷大与自身比较是否为 False
    assert (oo < oo) is S.false
    # 测试无穷大与负无穷大比较是否为 False
    assert (oo < -oo) is S.false
    # 测试无穷大与常数 1 比较是否为 False
    assert (oo < 1) is S.false
    # 测试无穷大与自身比较是否为 True
    assert (oo >= oo) is S.true
    # 测试无穷大与负无穷大比较是否为 True
    assert (oo >= -oo) is S.true
    # 测试无穷大与常数 1 比较是否为 True
    assert (oo >= 1) is S.true
    # 测试无穷大与自身比较是否为 True
    assert (oo <= oo) is S.true
    # 测试无穷大与负无穷大比较是否为 False
    assert (oo <= -oo) is S.false
    # 测试无穷大与常数 1 比较是否为 False
    assert (oo <= 1) is S.false
    # 测试负无穷大与无穷大比较是否为 False
    assert (-oo > oo) is S.false
    # 测试负无穷大与自身比较是否为 False
    assert (-oo > -oo) is S.false
    # 测试负无穷大与常数 1 比较是否为 False
    assert (-oo > 1) is S.false
    # 测试负无穷大与无穷大比较是否为 True
    assert (-oo < oo) is S.true
    # 测试负无穷大与自身比较是否为 False
    assert (-oo < -oo) is S.false
    # 测试负无穷大与常数 1 比较是否为 True
    assert (-oo < 1) is S.true
    # 测试负无穷大与无穷大比较是否为 False
    assert (-oo >= oo) is S.false
    # 测试负无穷大与自身比较是否为 True
    assert (-oo >= -oo) is S.true
    # 测试负无穷大与常数 1 比较是否为 False
    assert (-oo >= 1) is S.false
    # 测试负无穷大与无穷大比较是否为 True
    assert (-oo <= oo) is S.true
    # 测试负无穷大与自身比较是否为 True
    assert (-oo <= -oo) is S.true
    # 测试负无穷大与常数 1 比较是否为 True
    assert (-oo <= 1) is S.true


def test_infinite_symbol_inequalities():
    # 定义符号 x，y 为具有扩展正无穷大属性的符号变量
    x = Symbol('x', extended_positive=True, infinite=True)
    # 对于集合中的每个正无穷大值进行断言
    for inf1 in inf_set:
        # 断言正无穷大值小于 1 是假的
        assert (inf1 < 1) is S.false
        # 断言正无穷大值大于 1 是真的
        assert (inf1 > 1) is S.true
        # 断言正无穷大值小于等于 1 是假的
        assert (inf1 <= 1) is S.false
        # 断言正无穷大值大于等于 1 是真的
        assert (inf1 >= 1) is S.true

        # 对于集合中的每个正无穷大值，再次遍历集合进行断言
        for inf2 in inf_set:
            # 断言一个正无穷大值小于另一个正无穷大值 是假的
            assert (inf1 < inf2) is S.false
            # 断言一个正无穷大值大于另一个正无穷大值 是假的
            assert (inf1 > inf2) is S.false
            # 断言一个正无穷大值小于等于另一个正无穷大值 是真的
            assert (inf1 <= inf2) is S.true
            # 断言一个正无穷大值大于等于另一个正无穷大值 是真的
            assert (inf1 >= inf2) is S.true

        # 对于集合中的每个负无穷大值进行断言
        for ninf1 in ninf_set:
            # 断言正无穷大值小于负无穷大值 是假的
            assert (inf1 < ninf1) is S.false
            # 断言正无穷大值大于负无穷大值 是真的
            assert (inf1 > ninf1) is S.true
            # 断言正无穷大值小于等于负无穷大值 是假的
            assert (inf1 <= ninf1) is S.false
            # 断言正无穷大值大于等于负无穷大值 是真的
            assert (inf1 >= ninf1) is S.true
            # 断言负无穷大值小于正无穷大值 是真的
            assert (ninf1 < inf1) is S.true
            # 断言负无穷大值大于正无穷大值 是假的
            assert (ninf1 > inf1) is S.false
            # 断言负无穷大值小于等于正无穷大值 是真的
            assert (ninf1 <= inf1) is S.true
            # 断言负无穷大值大于等于正无穷大值 是假的
            assert (ninf1 >= inf1) is S.false

    # 对于集合中的每个负无穷大值进行断言
    for ninf1 in ninf_set:
        # 断言负无穷大值小于 1 是真的
        assert (ninf1 < 1) is S.true
        # 断言负无穷大值大于 1 是假的
        assert (ninf1 > 1) is S.false
        # 断言负无穷大值小于等于 1 是真的
        assert (ninf1 <= 1) is S.true
        # 断言负无穷大值大于等于 1 是假的
        assert (ninf1 >= 1) is S.false

        # 对于集合中的每个负无穷大值，再次遍历集合进行断言
        for ninf2 in ninf_set:
            # 断言一个负无穷大值小于另一个负无穷大值 是假的
            assert (ninf1 < ninf2) is S.false
            # 断言一个负无穷大值大于另一个负无穷大值 是假的
            assert (ninf1 > ninf2) is S.false
            # 断言一个负无穷大值小于等于另一个负无穷大值 是真的
            assert (ninf1 <= ninf2) is S.true
            # 断言一个负无穷大值大于等于另一个负无穷大值 是真的
            assert (ninf1 >= ninf2) is S.true
# 定义测试函数 test_bool，用于测试符号逻辑运算的正确性
def test_bool():
    # 断言 0 等于 0 返回真值 S.true
    assert Eq(0, 0) is S.true
    # 断言 1 等于 0 返回假值 S.false
    assert Eq(1, 0) is S.false
    # 断言 0 不等于 0 返回假值 S.false
    assert Ne(0, 0) is S.false
    # 断言 1 不等于 0 返回真值 S.true
    assert Ne(1, 0) is S.true
    # 断言 0 小于 1 返回真值 S.true
    assert Lt(0, 1) is S.true
    # 断言 1 小于 0 返回假值 S.false
    assert Lt(1, 0) is S.false
    # 断言 0 小于等于 1 返回真值 S.true
    assert Le(0, 1) is S.true
    # 断言 1 小于等于 0 返回假值 S.false
    assert Le(1, 0) is S.false
    # 断言 0 小于等于 0 返回真值 S.true
    assert Le(0, 0) is S.true
    # 断言 1 大于 0 返回真值 S.true
    assert Gt(1, 0) is S.true
    # 断言 0 大于 1 返回假值 S.false
    assert Gt(0, 1) is S.false
    # 断言 1 大于等于 0 返回真值 S.true
    assert Ge(1, 0) is S.true
    # 断言 0 大于等于 1 返回假值 S.false
    assert Ge(0, 1) is S.false
    # 断言 1 大于等于 1 返回真值 S.true
    assert Ge(1, 1) is S.true
    # 断言 I 等于 2 返回假值 S.false
    assert Eq(I, 2) is S.false
    # 断言 I 不等于 2 返回真值 S.true
    assert Ne(I, 2) is S.true
    # 断言对于 Gt(I, 2) 抛出 TypeError 异常
    raises(TypeError, lambda: Gt(I, 2))
    # 断言对于 Ge(I, 2) 抛出 TypeError 异常
    raises(TypeError, lambda: Ge(I, 2))
    # 断言对于 Lt(I, 2) 抛出 TypeError 异常
    raises(TypeError, lambda: Lt(I, 2))
    # 断言对于 Le(I, 2) 抛出 TypeError 异常
    raises(TypeError, lambda: Le(I, 2))
    # 创建浮点数 a 和 b
    a = Float('.000000000000000000001', '')
    b = Float('.0000000000000000000001', '')
    # 断言 pi + a 等于 pi + b 返回假值 S.false
    assert Eq(pi + a, pi + b) is S.false


# 定义测试函数 test_rich_cmp，用于测试富比较运算符的等效性
def test_rich_cmp():
    # 断言 x 小于 y 等同于 Lt(x, y)
    assert (x < y) == Lt(x, y)
    # 断言 x 小于等于 y 等同于 Le(x, y)
    assert (x <= y) == Le(x, y)
    # 断言 x 大于 y 等同于 Gt(x, y)
    assert (x > y) == Gt(x, y)
    # 断言 x 大于等于 y 等同于 Ge(x, y)
    assert (x >= y) == Ge(x, y)


# 定义测试函数 test_doit，用于测试 doit 方法在不同条件下的返回结果
def test_doit():
    # 导入符号 p 和 n，并设置其属性为正和负
    from sympy.core.symbol import Symbol
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    np = Symbol('np', nonpositive=True)
    nn = Symbol('nn', nonnegative=True)
    
    # 断言 Gt(p, 0).doit() 返回真值 S.true
    assert Gt(p, 0).doit() is S.true
    # 断言 Gt(p, 1).doit() 等于 Gt(p, 1)
    assert Gt(p, 1).doit() == Gt(p, 1)
    # 断言 Ge(p, 0).doit() 返回真值 S.true
    assert Ge(p, 0).doit() is S.true
    # 断言 Le(p, 0).doit() 返回假值 S.false
    assert Le(p, 0).doit() is S.false
    # 断言 Lt(n, 0).doit() 返回真值 S.true
    assert Lt(n, 0).doit() is S.true
    # 断言 Le(np, 0).doit() 返回真值 S.true
    assert Le(np, 0).doit() is S.true
    # 断言 Gt(nn, 0).doit() 等于 Gt(nn, 0)
    assert Gt(nn, 0).doit() == Gt(nn, 0)
    # 断言 Lt(nn, 0).doit() 返回假值 S.false
    
    assert Lt(nn, 0).doit() is S.false

    # 断言 Eq(x, 0).doit() 等于 Eq(x, 0)
    assert Eq(x, 0).doit() == Eq(x, 0)


# 定义测试函数 test_new_relational，用于测试新建关系表达式的等效性
def test_new_relational():
    # 定义符号 x
    x = Symbol('x')

    # 断言 Eq(x, 0) 等同于 Relational(x, 0)（使用默认关系 '=='）
    assert Eq(x, 0) == Relational(x, 0)       # None ==> Equality
    assert Eq(x, 0) == Relational(x, 0, '==')
    assert Eq(x, 0) == Relational(x, 0, 'eq')
    assert Eq(x, 0) == Equality(x, 0)

    # 断言 Eq(x, 0) 不等于 Relational(x, 1)（使用默认关系 '=='）
    assert Eq(x, 0) != Relational(x, 1)       # None ==> Equality
    assert Eq(x, 0) != Relational(x, 1, '==')
    assert Eq(x, 0) != Relational(x, 1, 'eq')
    assert Eq(x, 0) != Equality(x, 1)

    # 断言 Eq(x, -1) 等同于 Relational(x, -1)（使用默认关系 '=='）
    assert Eq(x, -1) == Relational(x, -1)       # None ==> Equality
    assert Eq(x, -1) == Relational(x, -1, '==')
    assert Eq(x, -1) == Relational(x, -1, 'eq')
    assert Eq(x, -1) == Equality(x, -1)
    # 断言 Eq(x, -1) 不等于 Relational(x, 1)（使用默认关系 '=='）
    assert Eq(x, -1) != Relational(x, 1)       # None ==> Equality
    assert Eq(x, -1) != Relational(x, 1, '==')
    assert Eq(x, -1) != Relational(x, 1, 'eq')
    assert Eq(x, -1) != Equality(x, 1)

    # 断言 Ne(x, 0) 等同于 Relational(x, 0, '!=')（使用关系 '!='）
    assert Ne(x, 0) == Relational(x, 0, '!=')
    assert Ne(x, 0) == Relational(x, 0, '<>')
    assert Ne(x, 0) == Relational(x, 0, 'ne')
    assert Ne(x, 0) == Unequality(x, 0)
    # 断言 Ne(x, 0) 不等于 Relational(x, 1, '!=')（使用关系 '!='）
    assert Ne(x, 0) != Relational(x, 1, '!=')
    assert Ne(x, 0) != Relational(x, 1, '<>')
    assert Ne(x, 0) != Relational(x, 1, 'ne')
    assert Ne(x, 0) != Unequality
    # 断言 x 是否大于等于 1，使用 Relational 类检查关系
    assert (x >= 1) == Relational(x, 1, '>=')
    # 断言 x 是否大于等于 1，使用 Relational 类检查关系（使用 'ge' 作为运算符）
    assert (x >= 1) == Relational(x, 1, 'ge')
    # 断言 x 是否大于等于 1，使用 GreaterThan 类检查严格大于关系
    assert (x >= 1) == GreaterThan(x, 1)
    # 断言 x 是否大于等于 0，预期结果与 Relational 类的结果不相等
    assert (x >= 0) != Relational(x, 1, '>=')
    # 断言 x 是否大于等于 0，预期结果与 Relational 类的结果不相等（使用 'ge' 作为运算符）
    assert (x >= 0) != Relational(x, 1, 'ge')
    # 断言 x 是否大于等于 0，预期结果与 GreaterThan 类的结果不相等
    assert (x >= 0) != GreaterThan(x, 1)

    # 断言 x 是否小于等于 0，使用 Relational 类检查关系
    assert Le(x, 0) == Relational(x, 0, '<=')
    # 断言 x 是否小于等于 0，使用 Relational 类检查关系（使用 'le' 作为运算符）
    assert Le(x, 0) == Relational(x, 0, 'le')
    # 断言 x 是否小于等于 0，使用 LessThan 类检查严格小于关系
    assert Le(x, 0) == LessThan(x, 0)
    # 断言 x 是否小于等于 1，预期结果与 Relational 类的结果不相等
    assert Le(x, 1) != Relational(x, 0, '<=')
    # 断言 x 是否小于等于 1，预期结果与 Relational 类的结果不相等（使用 'le' 作为运算符）
    assert Le(x, 1) != Relational(x, 0, 'le')
    # 断言 x 是否小于等于 1，预期结果与 LessThan 类的结果不相等
    assert Le(x, 1) != LessThan(x, 0)

    # 断言 x 是否大于 0，使用 Relational 类检查关系
    assert Gt(x, 0) == Relational(x, 0, '>')
    # 断言 x 是否大于 0，使用 Relational 类检查关系（使用 'gt' 作为运算符）
    assert Gt(x, 0) == Relational(x, 0, 'gt')
    # 断言 x 是否大于 0，使用 StrictGreaterThan 类检查严格大于关系
    assert Gt(x, 0) == StrictGreaterThan(x, 0)
    # 断言 x 是否大于 1，预期结果与 Relational 类的结果不相等
    assert Gt(x, 1) != Relational(x, 0, '>')
    # 断言 x 是否大于 1，预期结果与 Relational 类的结果不相等（使用 'gt' 作为运算符）
    assert Gt(x, 1) != Relational(x, 0, 'gt')
    # 断言 x 是否大于 1，预期结果与 StrictGreaterThan 类的结果不相等
    assert Gt(x, 1) != StrictGreaterThan(x, 0)

    # 断言 x 是否小于 0，使用 Relational 类检查关系
    assert Lt(x, 0) == Relational(x, 0, '<')
    # 断言 x 是否小于 0，使用 Relational 类检查关系（使用 'lt' 作为运算符）
    assert Lt(x, 0) == Relational(x, 0, 'lt')
    # 断言 x 是否小于 0，使用 StrictLessThan 类检查严格小于关系
    assert Lt(x, 0) == StrictLessThan(x, 0)
    # 断言 x 是否小于 1，预期结果与 Relational 类的结果不相等
    assert Lt(x, 1) != Relational(x, 0, '<')
    # 断言 x 是否小于 1，预期结果与 Relational 类的结果不相等（使用 'lt' 作为运算符）
    assert Lt(x, 1) != Relational(x, 0, 'lt')
    # 断言 x 是否小于 1，预期结果与 StrictLessThan 类的结果不相等
    assert Lt(x, 1) != StrictLessThan(x, 0)

    # 最后进行模糊测试
    # 导入 randint 函数用于生成随机整数
    from sympy.core.random import randint
    # 循环 100 次进行测试
    for i in range(100):
        # 生成随机的字符类型和长度
        while 1:
            strtype, length = (chr, 65535) if randint(0, 1) else (chr, 255)
            relation_type = strtype(randint(0, length))
            if randint(0, 1):
                relation_type += strtype(randint(0, length))
            # 如果生成的关系类型不在指定的列表中，则跳出循环
            if relation_type not in ('==', 'eq', '!=', '<>', 'ne', '>=', 'ge',
                                     '<=', 'le', '>', 'gt', '<', 'lt', ':=',
                                     '+=', '-=', '*=', '/=', '%='):
                break

        # 使用 lambda 函数和 Relational 类断言会抛出 ValueError 异常
        raises(ValueError, lambda: Relational(x, 1, relation_type))

    # 断言所有使用 Relational 类生成的等于关系的 rel_op 属性都等于 '=='
    assert all(Relational(x, 0, op).rel_op == '==' for op in ('eq', '=='))
    # 断言所有使用 Relational 类生成的不等于关系的 rel_op 属性都等于 '!='
    assert all(Relational(x, 0, op).rel_op == '!='
               for op in ('ne', '<>', '!='))
    # 断言所有使用 Relational 类生成的大于关系的 rel_op 属性都等于 '>'
    assert all(Relational(x, 0, op).rel_op == '>' for op in ('gt', '>'))
    # 断言所有使用 Relational 类生成的小于关系的 rel_op 属性都等于 '<'
    assert all(Relational(x, 0, op).rel_op == '<' for op in ('lt', '<'))
    # 断言所有使用 Relational 类生成的大于等于关系的 rel_op 属性都等于 '>='
    assert all(Relational(x, 0, op).rel_op == '>=' for op in ('ge', '>='))
    # 断言所有使用 Relational 类生成的小于等于关系的 rel_op 属性都等于 '<='
    assert all(Relational(x, 0, op).rel_op == '<=' for op in ('le', '<='))
# 定义一个测试函数，用于测试关系运算符的行为
def test_relational_arithmetic():
    # 遍历关系运算符类列表
    for cls in [Eq, Ne, Le, Lt, Ge, Gt]:
        # 创建指定关系运算符的实例
        rel = cls(x, y)
        # 断言以下操作会引发 TypeError 异常
        raises(TypeError, lambda: 0+rel)
        raises(TypeError, lambda: 1*rel)
        raises(TypeError, lambda: 1**rel)
        raises(TypeError, lambda: rel**1)
        raises(TypeError, lambda: Add(0, rel))
        raises(TypeError, lambda: Mul(1, rel))
        raises(TypeError, lambda: Pow(1, rel))
        raises(TypeError, lambda: Pow(rel, 1))


# 定义一个测试函数，用于测试关系运算符的布尔输出
def test_relational_bool_output():
    # 引用 GitHub 上的问题链接，详细说明测试内容
    raises(TypeError, lambda: bool(x > 3))
    raises(TypeError, lambda: bool(x >= 3))
    raises(TypeError, lambda: bool(x < 3))
    raises(TypeError, lambda: bool(x <= 3))
    raises(TypeError, lambda: bool(Eq(x, 3)))
    raises(TypeError, lambda: bool(Ne(x, 3)))


# 定义一个测试函数，测试关系逻辑符号的行为
def test_relational_logic_symbols():
    # 验证逻辑与运算符的行为
    assert (x < y) & (z < t) == And(x < y, z < t)
    # 验证逻辑或运算符的行为
    assert (x < y) | (z < t) == Or(x < y, z < t)
    # 验证逻辑非运算符的行为
    assert ~(x < y) == Not(x < y)
    # 验证蕴含运算符的行为
    assert (x < y) >> (z < t) == Implies(x < y, z < t)
    # 验证双向蕴含运算符的行为
    assert (x < y) << (z < t) == Implies(z < t, x < y)
    # 验证异或运算符的行为
    assert (x < y) ^ (z < t) == Xor(x < y, z < t)

    # 断言返回值的类型是否符合预期
    assert isinstance((x < y) & (z < t), And)
    assert isinstance((x < y) | (z < t), Or)
    assert isinstance(~(x < y), GreaterThan)
    assert isinstance((x < y) >> (z < t), Implies)
    assert isinstance((x < y) << (z < t), Implies)
    assert isinstance((x < y) ^ (z < t), (Or, Xor))


# 定义一个测试函数，测试单变量关系运算转换为集合的行为
def test_univariate_relational_as_set():
    # 验证大于零的关系转换为集合
    assert (x > 0).as_set() == Interval(0, oo, True, True)
    # 验证大于等于零的关系转换为集合
    assert (x >= 0).as_set() == Interval(0, oo)
    # 验证小于零的关系转换为集合
    assert (x < 0).as_set() == Interval(-oo, 0, True, True)
    # 验证小于等于零的关系转换为集合
    assert (x <= 0).as_set() == Interval(-oo, 0)
    # 验证等于零的关系转换为集合
    assert Eq(x, 0).as_set() == FiniteSet(0)
    # 验证不等于零的关系转换为集合
    assert Ne(x, 0).as_set() == Interval(-oo, 0, True, True) + \
        Interval(0, oo, True, True)

    # 验证平方大于等于 4 的关系转换为集合
    assert (x**2 >= 4).as_set() == Interval(-oo, -2) + Interval(2, oo)


# 定义一个测试函数，测试逻辑非运算符的行为
def test_Not():
    assert Not(Equality(x, y)) == Unequality(x, y)
    assert Not(Unequality(x, y)) == Equality(x, y)
    assert Not(StrictGreaterThan(x, y)) == LessThan(x, y)
    assert Not(StrictLessThan(x, y)) == GreaterThan(x, y)
    assert Not(GreaterThan(x, y)) == StrictLessThan(x, y)
    assert Not(LessThan(x, y)) == StrictGreaterThan(x, y)


# 定义一个测试函数，测试关系运算符的求值行为
def test_evaluate():
    assert str(Eq(x, x, evaluate=False)) == 'Eq(x, x)'
    assert Eq(x, x, evaluate=False).doit() == S.true
    assert str(Ne(x, x, evaluate=False)) == 'Ne(x, x)'
    assert Ne(x, x, evaluate=False).doit() == S.false

    assert str(Ge(x, x, evaluate=False)) == 'x >= x'
    assert str(Le(x, x, evaluate=False)) == 'x <= x'
    assert str(Gt(x, x, evaluate=False)) == 'x > x'
    assert str(Lt(x, x, evaluate=False)) == 'x < x'


# 定义一个函数，验证所有不等式运算符是否会引发 TypeError 异常
def assert_all_ineq_raise_TypeError(a, b):
    raises(TypeError, lambda: a > b)
    # 抛出 TypeError 异常，如果尝试进行 a >= b 的比较
    raises(TypeError, lambda: a >= b)
    # 抛出 TypeError 异常，如果尝试进行 a < b 的比较
    raises(TypeError, lambda: a < b)
    # 抛出 TypeError 异常，如果尝试进行 a <= b 的比较
    raises(TypeError, lambda: a <= b)
    # 抛出 TypeError 异常，如果尝试进行 b > a 的比较
    raises(TypeError, lambda: b > a)
    # 抛出 TypeError 异常，如果尝试进行 b >= a 的比较
    raises(TypeError, lambda: b >= a)
    # 抛出 TypeError 异常，如果尝试进行 b < a 的比较
    raises(TypeError, lambda: b < a)
    # 抛出 TypeError 异常，如果尝试进行 b <= a 的比较
    raises(TypeError, lambda: b <= a)
# 确保所有的不等式操作结果都是类 Inequality 的实例
def assert_all_ineq_give_class_Inequality(a, b):
    """All inequality operations on `a` and `b` result in class Inequality."""
    # 导入符号运算模块中的不等式类 _Inequality
    from sympy.core.relational import _Inequality as Inequality
    # 断言所有的比较操作返回的对象是 Inequality 类的实例
    assert isinstance(a > b,  Inequality)
    assert isinstance(a >= b, Inequality)
    assert isinstance(a < b,  Inequality)
    assert isinstance(a <= b, Inequality)
    assert isinstance(b > a,  Inequality)
    assert isinstance(b >= a, Inequality)
    assert isinstance(b < a,  Inequality)
    assert isinstance(b <= a, Inequality)


def test_imaginary_compare_raises_TypeError():
    # 查看问题编号 #5724
    assert_all_ineq_raise_TypeError(I, x)


def test_complex_compare_not_real():
    # 两种不是实数的情况
    y = Symbol('y', imaginary=True)
    z = Symbol('z', complex=True, extended_real=False)
    for w in (y, z):
        assert_all_ineq_raise_TypeError(2, w)
    # 一些应该保持未评估状态的情况
    t = Symbol('t')
    x = Symbol('x', real=True)
    z = Symbol('z', complex=True)
    for w in (x, z, t):
        assert_all_ineq_give_class_Inequality(2, w)


def test_imaginary_and_inf_compare_raises_TypeError():
    # 查看拉取请求 #7835
    y = Symbol('y', imaginary=True)
    assert_all_ineq_raise_TypeError(oo, y)
    assert_all_ineq_raise_TypeError(-oo, y)


def test_complex_pure_imag_not_ordered():
    raises(TypeError, lambda: 2*I < 3*I)

    # 更一般地
    x = Symbol('x', real=True, nonzero=True)
    y = Symbol('y', imaginary=True)
    z = Symbol('z', complex=True)
    assert_all_ineq_raise_TypeError(I, y)

    t = I*x   # 一个虚数，应该引发错误
    assert_all_ineq_raise_TypeError(2, t)

    t = -I*y   # 一个实数，因此不应有错误
    assert_all_ineq_give_class_Inequality(2, t)

    t = I*z   # 未知，应保持未评估状态
    assert_all_ineq_give_class_Inequality(2, t)


def test_x_minus_y_not_same_as_x_lt_y():
    """
    由于拉取请求 #7792 的后果，`x - y < 0` 和 `x < y` 不等同。
    """
    x = I + 2
    y = I + 3
    raises(TypeError, lambda: x < y)
    assert x - y < 0

    ineq = Lt(x, y, evaluate=False)
    raises(TypeError, lambda: ineq.doit())
    assert ineq.lhs - ineq.rhs < 0

    t = Symbol('t', imaginary=True)
    x = 2 + t
    y = 3 + t
    ineq = Lt(x, y, evaluate=False)
    raises(TypeError, lambda: ineq.doit())
    assert ineq.lhs - ineq.rhs < 0

    # 这个例子两种方法都应该报错
    x = I + 2
    y = 2*I + 3
    raises(TypeError, lambda: x < y)
    raises(TypeError, lambda: x - y < 0)


def test_nan_equality_exceptions():
    # 查看问题编号 #7774
    import random
    assert Equality(nan, nan) is S.false
    assert Unequality(nan, nan) is S.true

    # 查看问题编号 #7773
    A = (x, S.Zero, S.One/3, pi, oo, -oo)
    assert Equality(nan, random.choice(A)) is S.false
    assert Equality(random.choice(A), nan) is S.false
    assert Unequality(nan, random.choice(A)) is S.true
    assert Unequality(random.choice(A), nan) is S.true
def test_nan_inequality_raise_errors():
    # 在 PR #7776 的讨论中提到，我们测试包含各种类别示例的集合中的不等式。
    for q in (x, S.Zero, S(10), S.One/3, pi, S(1.3), oo, -oo, nan):
        assert_all_ineq_raise_TypeError(q, nan)


def test_nan_complex_inequalities():
    # NaN 与非实数的比较会引发错误，我们不太在意是 NaN 错误还是复数错误。
    for r in (I, zoo, Symbol('z', imaginary=True)):
        assert_all_ineq_raise_TypeError(r, nan)


def test_complex_infinity_inequalities():
    raises(TypeError, lambda: zoo > 0)
    raises(TypeError, lambda: zoo >= 0)
    raises(TypeError, lambda: zoo < 0)
    raises(TypeError, lambda: zoo <= 0)


def test_inequalities_symbol_name_same():
    """Using the operator and functional forms should give same results."""
    # 我们测试来自集合的所有组合
    # FIXME: 在测试通过后可以用随机选择替换
    A = (x, y, S.Zero, S.One/3, pi, oo, -oo)
    for a in A:
        for b in A:
            assert Gt(a, b) == (a > b)
            assert Lt(a, b) == (a < b)
            assert Ge(a, b) == (a >= b)
            assert Le(a, b) == (a <= b)

    for b in (y, S.Zero, S.One/3, pi, oo, -oo):
        assert Gt(x, b, evaluate=False) == (x > b)
        assert Lt(x, b, evaluate=False) == (x < b)
        assert Ge(x, b, evaluate=False) == (x >= b)
        assert Le(x, b, evaluate=False) == (x <= b)

    for b in (y, S.Zero, S.One/3, pi, oo, -oo):
        assert Gt(b, x, evaluate=False) == (b > x)
        assert Lt(b, x, evaluate=False) == (b < x)
        assert Ge(b, x, evaluate=False) == (b >= x)
        assert Le(b, x, evaluate=False) == (b <= x)


def test_inequalities_symbol_name_same_complex():
    """Using the operator and functional forms should give same results.
    With complex non-real numbers, both should raise errors.
    """
    # FIXME: 在测试通过后可以用随机选择替换
    for a in (x, S.Zero, S.One/3, pi, oo, Rational(1, 3)):
        raises(TypeError, lambda: Gt(a, I))
        raises(TypeError, lambda: a > I)
        raises(TypeError, lambda: Lt(a, I))
        raises(TypeError, lambda: a < I)
        raises(TypeError, lambda: Ge(a, I))
        raises(TypeError, lambda: a >= I)
        raises(TypeError, lambda: Le(a, I))
        raises(TypeError, lambda: a <= I)


def test_inequalities_cant_sympify_other():
    # 参见问题 7833
    from operator import gt, lt, ge, le

    bar = "foo"

    for a in (x, S.Zero, S.One/3, pi, I, zoo, oo, -oo, nan, Rational(1, 3)):
        for op in (lt, gt, le, ge):
            raises(TypeError, lambda: op(a, bar))


def test_ineq_avoid_wild_symbol_flip():
    # 参见问题 #7951，我们尝试在内部避免这种情况，例如通过使用 __lt__ 而不是 "<"。
    from sympy.core.symbol import Wild
    p = symbols('p', cls=Wild)
    # x > p 可能会翻转，但 Gt 不应该：
    assert Gt(x, p) == Gt(x, p, evaluate=False)
    # 使用符号数学库中的 Lt 表达式来替换 y 为 p，创建新的表达式 e
    e = Lt(x, y).subs({y: p})
    # 使用断言检查新表达式 e 是否等于给定的 Lt 表达式，不进行求值
    assert e == Lt(x, p, evaluate=False)
    
    # 使用符号数学库中的 Ge 表达式创建 x >= p 的表达式 e，并进行求值
    e = Ge(x, p).doit()
    # 使用断言检查新表达式 e 是否等于给定的 Ge 表达式，不进行求值
    assert e == Ge(x, p, evaluate=False)
# 定义一个名为 test_issue_8245 的测试函数
def test_issue_8245():
    # 创建一个符号对象 a，表示一个大整数分数
    a = S("6506833320952669167898688709329/5070602400912917605986812821504")
    # 断言：使用 rel_check 函数验证 a 和 a 的 n(10) 的数值在相对误差允许范围内
    assert rel_check(a, a.n(10))
    # 断言：使用 rel_check 函数验证 a 和 a 的 n(20) 的数值在相对误差允许范围内
    assert rel_check(a, a.n(20))
    # 断言：使用 rel_check 函数验证 a 和 a 的数值在默认精度下的数值在相对误差允许范围内
    assert rel_check(a, a.n())
    # 断言：使用 Float 函数将 a 转换为浮点数，精度为 31，以便完全捕获 a 作为 mpf
    assert Float(a, 31) == Float(str(a.p), '')/Float(str(a.q), '')
    
    # 对于 i 在范围 [0, 31) 内的每个整数，进行以下操作
    for i in range(31):
        # 将 a 转换为浮点有理数
        r = Rational(Float(a, i))
        # 将浮点有理数 r 转换为浮点数 f
        f = Float(r)
        # 断言：比较 f < a 的结果是否等于 Rational(f) < a 的结果
        assert (f < a) == (Rational(f) < a)
    
    # 测试符号的符号处理
    assert (-f < -a) == (Rational(-f) < -a)
    
    # 将 a 表示为浮点数 a.p / a.q
    isa = Float(a.p,'')/Float(a.q,'')
    # 断言：验证 isa 小于等于 a
    assert isa <= a
    # 断言：验证 isa 不小于 a
    assert not isa < a
    # 断言：验证 isa 大于等于 a
    assert isa >= a
    # 断言：验证 isa 不大于 a
    assert not isa > a
    # 断言：验证 isa 大于 0
    assert isa > 0
    
    # 将 a 设置为 sqrt(2)
    a = sqrt(2)
    # 将 a 的数值转换为有理数，精度为 30
    r = Rational(str(a.n(30)))
    # 断言：使用 rel_check 函数验证 a 和 r 的数值在相对误差允许范围内
    assert rel_check(a, r)
    
    # 将 a 设置为 sqrt(2)
    a = sqrt(2)
    # 将 a 的数值转换为有理数，精度为 29
    r = Rational(str(a.n(29)))
    # 断言：使用 rel_check 函数验证 a 和 r 的数值在相对误差允许范围内
    assert rel_check(a, r)
    
    # 断言：验证 log(cos(2)^2 + sin(2)^2) 等于 0
    assert Eq(log(cos(2)**2 + sin(2)**2), 0) is S.true


# 定义一个名为 test_issue_8449 的测试函数
def test_issue_8449():
    # 创建一个非负符号对象 p
    p = Symbol('p', nonnegative=True)
    # 断言：验证 -oo 小于 p
    assert Lt(-oo, p)
    # 断言：验证 -oo 不大于 p
    assert Ge(-oo, p) is S.false
    # 断言：验证 oo 大于 -p
    assert Gt(oo, -p)
    # 断言：验证 oo 不小于 -p
    assert Le(oo, -p) is S.false


# 定义一个名为 test_simplify_relational 的测试函数
def test_simplify_relational():
    # 断言：简化表达式 x*(y + 1) - x*y - x + 1 < x 的结果为 x > 1
    assert simplify(x*(y + 1) - x*y - x + 1 < x) == (x > 1)
    # 断言：简化表达式 x*(y + 1) - x*y - x - 1 < x 的结果为 x > -1
    assert simplify(x*(y + 1) - x*y - x - 1 < x) == (x > -1)
    # 断言：简化表达式 x < x*(y + 1) - x*y - x + 1 的结果为 x < 1
    assert simplify(x < x*(y + 1) - x*y - x + 1) == (x < 1)
    
    # 创建符号 q 和 r
    q, r = symbols("q r")
    # 断言：简化表达式 ((-q + r) - (q - r)) <= 0 的结果为 q >= r
    assert (((-q + r) - (q - r)) <= 0).simplify() == (q >= r)
    
    # 计算 sqrt(2)
    root2 = sqrt(2)
    # 断言：简化表达式 ((root2 * (-q + r) - root2 * (q - r)) <= 0).simplify() 的结果为 q >= r
    equation = ((root2 * (-q + r) - root2 * (q - r)) <= 0).simplify()
    assert equation == (q >= r)
    
    # 断言：简化表达式 Eq(log(cos(2)^2 + sin(2)^2), 0) 的结果为 S.true
    assert Eq(log(cos(2)**2 + sin(2)**2), 0) is S.true
    
    # 创建一个关系表达式 r，表示 S.One < x
    r = S.One < x
    # 断言：简化 r 的结果为 r.canonical
    assert simplify(r) == r.canonical
    # 断言：使用 ratio=0 简化 r 的结果不等于 r.canonical
    assert simplify(r, ratio=0) != r.canonical
    
    # 断言：简化表达式 -(2**(pi*Rational(3, 2)) + 6**pi)**(1/pi) + 2*(2**(pi/2) + 3**pi)**(1/pi) < 0 的结果为 S.false
    assert simplify(-(2**(pi*Rational(3, 2)) + 6**pi)**(1/pi) + 2*(2**(pi/2) + 3**pi)**(1/pi) < 0) is S.false
    
    # 断言：简化表达式 Eq(y, x) 的结果为 Eq(x, y)
    assert Eq(y, x).simplify() == Eq(x, y)
    # 断言：简化表达式 Eq(x - 1, 0) 的结果为 Eq(x, 1)
    assert Eq(x - 1, 0).simplify() == Eq(x, 1)
    # 断言：简化表达式 Eq(x - 1, x) 的结果为 S.false
    assert Eq(x - 1, x).simplify() == S.false
    # 断言：简化表达式 Eq(2*x - 1, x) 的结果为 Eq(x, 1)
    assert Eq(2*x - 1, x).simplify() == Eq(x, 1)
    # 断言：简化表达式 Eq(2*x, 4) 的结果为 Eq(x, 2)
    assert Eq(2*x, 4).simplify() == Eq(x, 2)
    
    # 计算 z = cos(1)^2 + sin(1)^2 - 1
    z = cos(1)**2 + sin(1)**2 - 1
    # 断言：简化表达式 Eq(z*x, 0) 的结果为 S.true
    assert Eq(z*x, 0).simplify() == S.true
    
    # 断言：简化表达式 Ne(y, x) 的结果为 Ne(x, y)
    assert Ne(y, x).simplify() == Ne(x, y)
    # 断言：简化表达式 Ne(x - 1, 0) 的结果为 Ne(x, 1)
    assert Ne(x - 1, 0).simplify() == Ne(x, 1)
    # 断言：简化表达式 Ne(x - 1, x
    # 断言：检查 Ge(z*x, 0) 简化后是否等于真值 S.true
    assert Ge(z*x, 0).simplify() == S.true
    # 断言：检查 Ge(x, -2) 简化后是否等于 Ge(x, -2)
    assert Ge(x, -2).simplify() == Ge(x, -2)
    # 断言：检查 Ge(-x, -2) 简化后是否等于 Le(x, 2)
    assert Ge(-x, -2).simplify() == Le(x, 2)
    # 断言：检查 Ge(x, 2) 简化后是否等于 Ge(x, 2)
    assert Ge(x, 2).simplify() == Ge(x, 2)
    # 断言：检查 Ge(-x, 2) 简化后是否等于 Le(x, -2)
    assert Ge(-x, 2).simplify() == Le(x, -2)

    # 断言：检查 Le(y, x) 简化后是否等于 Ge(x, y)
    assert Le(y, x).simplify() == Ge(x, y)
    # 断言：检查 Le(x - 1, 0) 简化后是否等于 Le(x, 1)
    assert Le(x - 1, 0).simplify() == Le(x, 1)
    # 断言：检查 Le(x - 1, x) 简化后是否等于真值 S.true
    assert Le(x - 1, x).simplify() == S.true
    # 断言：检查 Le(2*x - 1, x) 简化后是否等于 Le(x, 1)
    assert Le(2*x - 1, x).simplify() == Le(x, 1)
    # 断言：检查 Le(2*x, 4) 简化后是否等于 Le(x, 2)
    assert Le(2*x, 4).simplify() == Le(x, 2)
    # 断言：检查 Le(z*x, 0) 简化后是否等于真值 S.true
    assert Le(z*x, 0).simplify() == S.true
    # 断言：检查 Le(x, -2) 简化后是否等于 Le(x, -2)
    assert Le(x, -2).simplify() == Le(x, -2)
    # 断言：检查 Le(-x, -2) 简化后是否等于 Ge(x, 2)
    assert Le(-x, -2).simplify() == Ge(x, 2)
    # 断言：检查 Le(x, 2) 简化后是否等于 Le(x, 2)
    assert Le(x, 2).simplify() == Le(x, 2)
    # 断言：检查 Le(-x, 2) 简化后是否等于 Ge(x, -2)
    assert Le(-x, 2).simplify() == Ge(x, -2)

    # 断言：检查 Gt(y, x) 简化后是否等于 Lt(x, y)
    assert Gt(y, x).simplify() == Lt(x, y)
    # 断言：检查 Gt(x - 1, 0) 简化后是否等于 Gt(x, 1)
    assert Gt(x - 1, 0).simplify() == Gt(x, 1)
    # 断言：检查 Gt(x - 1, x) 简化后是否等于假值 S.false
    assert Gt(x - 1, x).simplify() == S.false
    # 断言：检查 Gt(2*x - 1, x) 简化后是否等于 Gt(x, 1)
    assert Gt(2*x - 1, x).simplify() == Gt(x, 1)
    # 断言：检查 Gt(2*x, 4) 简化后是否等于 Gt(x, 2)
    assert Gt(2*x, 4).simplify() == Gt(x, 2)
    # 断言：检查 Gt(z*x, 0) 简化后是否等于假值 S.false
    assert Gt(z*x, 0).simplify() == S.false
    # 断言：检查 Gt(x, -2) 简化后是否等于 Gt(x, -2)
    assert Gt(x, -2).simplify() == Gt(x, -2)
    # 断言：检查 Gt(-x, -2) 简化后是否等于 Lt(x, 2)
    assert Gt(-x, -2).simplify() == Lt(x, 2)
    # 断言：检查 Gt(x, 2) 简化后是否等于 Gt(x, 2)
    assert Gt(x, 2).simplify() == Gt(x, 2)
    # 断言：检查 Gt(-x, 2) 简化后是否等于 Lt(x, -2)
    assert Gt(-x, 2).simplify() == Lt(x, -2)

    # 断言：检查 Lt(y, x) 简化后是否等于 Gt(x, y)
    assert Lt(y, x).simplify() == Gt(x, y)
    # 断言：检查 Lt(x - 1, 0) 简化后是否等于 Lt(x, 1)
    assert Lt(x - 1, 0).simplify() == Lt(x, 1)
    # 断言：检查 Lt(x - 1, x) 简化后是否等于真值 S.true
    assert Lt(x - 1, x).simplify() == S.true
    # 断言：检查 Lt(2*x - 1, x) 简化后是否等于 Lt(x, 1)
    assert Lt(2*x - 1, x).simplify() == Lt(x, 1)
    # 断言：检查 Lt(2*x, 4) 简化后是否等于 Lt(x, 2)
    assert Lt(2*x, 4).simplify() == Lt(x, 2)
    # 断言：检查 Lt(z*x, 0) 简化后是否等于假值 S.false
    assert Lt(z*x, 0).simplify() == S.false
    # 断言：检查 Lt(x, -2) 简化后是否等于 Lt(x, -2)
    assert Lt(x, -2).simplify() == Lt(x, -2)
    # 断言：检查 Lt(-x, -2) 简化后是否等于 Ge(x, 2)
    assert Lt(-x, -2).simplify() == Ge(x, 2)
    # 断言：检查 Lt(x, 2) 简化后是否等于 Lt(x, 2)
    assert Lt(x, 2).simplify() == Lt(x, 2)
    # 断言：检查 Lt(-x, 2) 简化后是否等于 Ge(x, -2)
    assert Lt(-x, 2).simplify() == Ge(x, -2)

    # 测试 _eval_simplify 的特定分支
    m = exp(1) - exp_polar(1)
    # 断言：检查 simplify(m*x > 1) 简化后是否等于假值 S.false
    assert simplify(m*x > 1) is S.false
    # 断言：检查 simplify(m*x + 2*m*y > 1) 简化后是否等于假值 S.false（与下一行测试相同的分支）
    assert simplify(m*x + 2*m*y > 1) is S.false
    # 断言：检查 simplify(m*x + y > 1 + y) 简化后是否等于假值 S.false
    assert simplify(m*x + y > 1 + y) is S.false
# 定义一个测试函数，用于检查表达式是否相等
def test_equals():
    # 使用 sympy 的 symbols 函数创建符号变量 w, x, y, z
    w, x, y, z = symbols('w:z')
    # 使用 sympy 的 Function 函数创建函数 f
    f = Function('f')
    
    # 断言第一个表达式是否相等，返回 True
    assert Eq(x, 1).equals(Eq(x*(y + 1) - x*y - x + 1, x))
    # 断言第二个表达式是否相等，返回 False
    assert Eq(x, y).equals(x < y, True) == False
    # 断言第三个表达式是否相等，返回 f(1) - f(2)
    assert Eq(x, f(1)).equals(Eq(x, f(2)), True) == f(1) - f(2)
    # 断言第四个表达式是否相等，返回 f(1) - f(2)
    assert Eq(f(1), y).equals(Eq(f(2), y), True) == f(1) - f(2)
    # 断言第五个表达式是否相等，返回 f(1) - f(2)
    assert Eq(x, f(1)).equals(Eq(f(2), x), True) == f(1) - f(2)
    # 断言第六个表达式是否相等，返回 f(1) - f(2)
    assert Eq(f(1), x).equals(Eq(x, f(2)), True) == f(1) - f(2)
    # 断言第七个表达式是否相等，返回 False
    assert Eq(w, x).equals(Eq(y, z), True) == False
    # 断言第八个表达式是否相等，返回 f(1) - f(3)
    assert Eq(f(1), f(2)).equals(Eq(f(3), f(4)), True) == f(1) - f(3)
    # 断言第九个表达式是否相等，返回 True
    assert (x < y).equals(y > x, True) == True
    # 断言第十个表达式是否相等，返回 False
    assert (x < y).equals(y >= x, True) == False
    # 断言第十一个表达式是否相等，返回 False
    assert (x < y).equals(z < y, True) == False
    # 断言第十二个表达式是否相等，返回 False
    assert (x < y).equals(x < z, True) == False
    # 断言第十三个表达式是否相等，返回 f(1) - f(2)
    assert (x < f(1)).equals(x < f(2), True) == f(1) - f(2)
    # 断言第十四个表达式是否相等，返回 f(1) - f(2)
    assert (f(1) < x).equals(f(2) < x, True) == f(1) - f(2)


# 定义一个测试函数，用于检查逆向关系
def test_reversed():
    # 断言逆向操作是否正确
    assert (x < y).reversed == (y > x)
    assert (x <= y).reversed == (y >= x)
    assert Eq(x, y, evaluate=False).reversed == Eq(y, x, evaluate=False)
    assert Ne(x, y, evaluate=False).reversed == Ne(y, x, evaluate=False)
    assert (x >= y).reversed == (y <= x)
    assert (x > y).reversed == (y < x)


# 定义一个测试函数，用于检查规范化表达式
def test_canonical():
    # 创建一个包含不同表达式的列表 c
    c = [i.canonical for i in (
        x + y < z,
        x + 2 > 3,
        x < 2,
        S(2) > x,
        x**2 > -x/y,
        Gt(3, 2, evaluate=False)
        )]
    # 断言列表中每个元素的 canonical 属性是否等于自身
    assert [i.canonical for i in c] == c
    # 断言列表中每个元素的逆向 canonical 属性是否等于自身
    assert [i.reversed.canonical for i in c] == c
    # 断言列表中是否不存在 lhs 是数值而 rhs 不是数值的元素
    assert not any(i.lhs.is_Number and not i.rhs.is_Number for i in c)

    # 对列表中每个元素执行反向操作，并获取其 canonical 属性
    c = [i.reversed.func(i.rhs, i.lhs, evaluate=False).canonical for i in c]
    # 断言列表中每个元素的 canonical 属性是否等于自身
    assert [i.canonical for i in c] == c
    # 断言列表中每个元素的逆向 canonical 属性是否等于自身
    assert [i.reversed.canonical for i in c] == c
    # 断言列表中是否不存在 lhs 是数值而 rhs 不是数值的元素
    assert not any(i.lhs.is_Number and not i.rhs.is_Number for i in c)
    # 断言 Eq(y < x, x > y) 的 canonical 属性是否为 S.true


# 标记为测试失败的函数
@XFAIL
def test_issue_8444_nonworkingtests():
    # 创建实数符号变量 x
    x = symbols('x', real=True)
    # 断言无穷大是否等于无穷大
    assert (x <= oo) == (x >= -oo) == True

    # 重新创建符号变量 x
    x = symbols('x')
    # 断言 x 是否大于等于其下取整
    assert x >= floor(x)
    # 断言 x 是否小于其下取整，预期为 False
    assert (x < floor(x)) == False
    # 断言 x 是否小于等于其上取整
    assert x <= ceiling(x)
    # 断言 x 是否大于其上取整，预期为 False
    assert (x > ceiling(x)) == False


# 定义一个测试函数，用于检查修复的问题
def test_issue_8444_workingtests():
    # 创建符号变量 x
    x = symbols('x')
    # 断言两个表达式是否相等，evaluate=False 表示不进行求值
    assert Gt(x, floor(x)) == Gt(x, floor(x), evaluate=False)
    assert Ge(x, floor(x)) == Ge(x, floor(x), evaluate=False)
    assert Lt(x, ceiling(x)) == Lt(x, ceiling(x), evaluate=False)
    assert Le(x, ceiling(x)) == Le(x, ceiling(x), evaluate=False)
    # 创建整数符号变量 i
    i = symbols('i', integer=True)
    # 断言 i 是否大于其下取整，预期为 False
    assert (i > floor(i)) == False
    # 断言 i 是否小于其上取整，预期为 False
    assert (i < ceiling(i)) == False


# 定义一个测试函数，用于检查修复的问题
def test_issue_10304():
    # 创建一个表达式 d
    d = cos(1)**2 + sin(1)**2 - 1
    # 断言 d 是否不可比较
    assert d.is_comparable is False  # if this fails, find a new d
    # 创建一个表达式 e
    e = 1 + d*I
    # 断言简化后的 e 是否为 S.false
    assert simplify(Eq(e, 0)) is S.false


# 定义一个测试函数，用于检查修复的问题
def test_issue_18412():
    # 创建一个表达式 d
    d = (Rational(1, 6) + z / 4 / y)
    # 断言替换表达式中的 y**3 为 z 后是否相等
    assert Eq(x, pi * y**3 * d).replace(y**3, z) == Eq(x, pi * z * d)


# 定义一个测试函数，用于检查修复的问题
def test_issue_10401():
    # 创建符号变量 x
    x = symbols('x')
    # 创建有限的符号变量 fin 和无穷的符号变量 inf
    fin = symbols('inf', finite=True)
    inf = symbols('inf', infinite=True)
    inf2 = symbols('inf2', infinite=True)
    infx = symbols('infx', infinite=True, extended_real=True)
    # 在下面的注释测试中使用：
    infnx = symbols('inf~x', infinite=True, extended_real=False)
    infnx2 = symbols('inf~x2', infinite=True, extended_real=False)
    infp = symbols('infp', infinite=True, extended_positive=True)
    infp1 = symbols('infp1', infinite=True, extended_positive=True)
    infn = symbols('infn', infinite=True, extended_negative=True)
    zero = symbols('z', zero=True)
    nonzero = symbols('nz', zero=False, finite=True)

    # 断言以下表达式的函数类型是等式
    assert Eq(1/(1/x + 1), 1).func is Eq
    # 断言在将 x 替换为 S.ComplexInfinity 后，等式成立
    assert Eq(1/(1/x + 1), 1).subs(x, S.ComplexInfinity) is S.true
    # 断言在将 fin 替换为 inf 后，等式不成立
    assert Eq(1/(1/fin + 1), 1) is S.false

    T, F = S.true, S.false
    # 断言 fin 和 inf 不相等
    assert Eq(fin, inf) is F
    # 断言 inf 和 inf2 不相等，并且不在 (True, False) 中
    assert Eq(inf, inf2) not in (T, F) and inf != inf2
    # 断言 1 + inf 不等于 2 + inf2，并且不在 (True, False) 中
    assert Eq(1 + inf, 2 + inf2) not in (T, F) and inf != inf2
    # 断言 infp 和 infp1 相等
    assert Eq(infp, infp1) is T
    # 断言 infp 和 infn 不相等
    assert Eq(infp, infn) is F
    # 断言 1 + I*oo 不等于 I*oo
    assert Eq(1 + I*oo, I*oo) is F
    # 断言 I*oo 和 1 + I*oo 不等
    assert Eq(I*oo, 1 + I*oo) is F
    # 断言 1 + I*oo 和 2 + I*oo 不等
    assert Eq(1 + I*oo, 2 + I*oo) is F
    # 断言 1 + I*oo 和 2 + I*infx 不等
    assert Eq(1 + I*oo, 2 + I*infx) is F
    # 断言 1 + I*oo 和 2 + infx 不等
    assert Eq(1 + I*oo, 2 + infx) is F
    # FIXME: 下面的测试失败，因为 (-infx).is_extended_positive 为 True（应为 None）
    #assert Eq(1 + I*infx, 1 + I*infx2) not in (T, F) and infx != infx2
    #
    # 断言 zoo 等于 sqrt(2) + I*oo 不成立
    assert Eq(zoo, sqrt(2) + I*oo) is F
    # 断言 zoo 等于 oo 不成立
    assert Eq(zoo, oo) is F
    # 定义实数符号 r 和虚数符号 i
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    # 断言 i*I 等于 r 不成立
    assert Eq(i*I, r) not in (T, F)
    # 断言 infx 和 infnx 不相等
    assert Eq(infx, infnx) is F
    # 断言 infnx 和 infnx2 不相等，并且不在 (True, False) 中
    assert Eq(infnx, infnx2) not in (T, F) and infnx != infnx2
    # 断言 zoo 等于 oo 不成立
    assert Eq(zoo, oo) is F
    # 断言 inf/inf2 等于 0 不成立
    assert Eq(inf/inf2, 0) is F
    # 断言 inf/fin 等于 0 不成立
    assert Eq(inf/fin, 0) is F
    # 断言 fin/inf 等于 0 成立
    assert Eq(fin/inf, 0) is T
    # 断言 zero/nonzero 等于 0 成立，并且 (zero/nonzero) 不等于 0
    assert Eq(zero/nonzero, 0) is T and ((zero/nonzero) != 0)
    # 下面的注释测试被注释掉是不正确的，因为：
    #assert zoo == -zoo
    # 断言 zoo 等于 -zoo 成立
    assert Eq(zoo, -zoo) is T
    # 断言 oo 等于 -oo 不成立
    assert Eq(oo, -oo) is F
    # 断言 inf 等于 -inf 不在 (True, False) 中
    assert Eq(inf, -inf) not in (T, F)

    # 断言 fin/(fin + 1) 等于 1 不成立
    assert Eq(fin/(fin + 1), 1) is S.false

    # 定义奇数符号 o
    o = symbols('o', odd=True)
    # 断言 o 等于 2*o 不成立
    assert Eq(o, 2*o) is S.false

    # 定义正数符号 p
    p = symbols('p', positive=True)
    # 断言 p/(p - 1) 等于 1 不成立
    assert Eq(p/(p - 1), 1) is F
def test_issue_10633():
    # 第一个断言：比较 True 和 False 是否相等，期望结果为 False
    assert Eq(True, False) == False
    # 第二个断言：比较 False 和 True 是否相等，期望结果为 False
    assert Eq(False, True) == False
    # 第三个断言：比较 True 和 True 是否相等，期望结果为 True
    assert Eq(True, True) == True
    # 第四个断言：比较 False 和 False 是否相等，期望结果为 True
    assert Eq(False, False) == True


def test_issue_10927():
    # 定义符号变量 x
    x = symbols('x')
    # 断言将 x 和正无穷 (∞) 进行相等性字符串表示，期望结果为 'Eq(x, oo)'
    assert str(Eq(x, oo)) == 'Eq(x, oo)'
    # 断言将 x 和负无穷 (-∞) 进行相等性字符串表示，期望结果为 'Eq(x, -oo)'
    assert str(Eq(x, -oo)) == 'Eq(x, -oo)'


def test_issues_13081_12583_12534():
    # 13081
    # 定义有理数 r，分子为 905502432259640373，分母为 288230376151711744
    r = Rational('905502432259640373/288230376151711744')
    # 断言 r 是否小于 π，期望结果为 False
    assert (r < pi) is S.false
    # 断言 r 是否大于 π，期望结果为 True
    assert (r > pi) is S.true

    # 12583
    # 计算平方根值
    v = sqrt(2)
    # 使用复杂的表达式计算 u 的值
    u = sqrt(v) + 2/sqrt(10 - 8/sqrt(2 - v) + 4*v*(1/sqrt(2 - v) - 1))
    # 断言 u 是否大于等于 0，期望结果为 True
    assert (u >= 0) is S.true

    # 12534; Rational vs NumberSymbol
    # 精度比较，确认 Rational 形式和 π 的数值在不同精度下的关系
    assert [p for p in range(20, 50) if
            (Rational(pi.n(p)) < pi) and
            (pi < Rational(pi.n(p + 1)))] == [20, 24, 27, 33, 37, 43, 48]
    
    # 选取一个精度进行确认相反的操作结果
    for i in (20, 21):
        v = pi.n(i)
        assert rel_check(Rational(v), pi)
        assert rel_check(v, pi)
    assert rel_check(pi.n(20), pi.n(21))
    
    # Float vs Rational
    # Rational 形式是否小于浮点数表示的 π
    assert [i for i in range(15, 50) if Rational(pi.n(i)) > pi.n(i)] == []
    # 反转操作后结果相同的断言
    assert [i for i in range(15, 50) if pi.n(i) < Rational(pi.n(i))] == []


def test_issue_18188():
    # 导入符号变量 x 的条件集
    from sympy.sets.conditionset import ConditionSet
    # 创建结果 result1，包含 x*cos(x) - 3*sin(x) 的方程
    result1 = Eq(x*cos(x) - 3*sin(x), 0)
    # 断言 result1 转化为集合形式后是否与 ConditionSet(x, Eq(x*cos(x) - 3*sin(x), 0), Reals) 相等
    assert result1.as_set() == ConditionSet(x, Eq(x*cos(x) - 3*sin(x), 0), Reals)

    # 创建结果 result2，包含 x**2 + sqrt(x*2) + sin(x) 的方程
    result2 = Eq(x**2 + sqrt(x*2) + sin(x), 0)
    # 断言 result2 转化为集合形式后是否与 ConditionSet(x, Eq(sqrt(2)*sqrt(x) + x**2 + sin(x), 0), Reals) 相等
    assert result2.as_set() == ConditionSet(x, Eq(sqrt(2)*sqrt(x) + x**2 + sin(x), 0), Reals)


def test_binary_symbols():
    # 创建集合 ans，包含符号变量 x
    ans = {x}
    # 遍历 Eq 和 Ne 函数
    for f in Eq, Ne:
        # 遍历 S.true 和 S.false
        for t in S.true, S.false:
            # 创建等式 eq，比较 x 和 S.true 的结果
            eq = f(x, S.true)
            # 断言 eq 的二元符号是否与 ans 相等
            assert eq.binary_symbols == ans
            # 断言反转后的 eq 的二元符号是否与 ans 相等
            assert eq.reversed.binary_symbols == ans
        # 断言 f(x, 1) 的二元符号集合为空集
        assert f(x, 1).binary_symbols == set()


def test_rel_args():
    # 遍历操作符列表
    for op in ['<', '<=', '>', '>=']:
        # 遍历布尔值 b 的集合
        for b in (S.true, x < 1, And(x, y)):
            # 遍历值 v 的集合
            for v in (0.1, 1, 2**32, t, S.One):
                # 使用 lambda 函数验证 Relational(b, v, op) 是否引发 TypeError 异常
                raises(TypeError, lambda: Relational(b, v, op))


def test_nothing_happens_to_Eq_condition_during_simplify():
    # issue 25701
    # 定义实数符号变量 r
    r = symbols('r', real=True)
    # 使用断言验证表达式是否简化为零
    assert Eq(2*sign(r + 3)/(5*Abs(r + 3)**Rational(3, 5)), 0
        ).simplify() == Eq(
        # 如果 r 等于 -3，则表达式等于 0
        Piecewise(
            (0, Eq(r, -3)),
            # 否则根据给定的表达式计算值
            ((r + 3)/(5*Abs((r + 3)**Rational(8, 5)))*2, True)), 0)
def test_issue_15847():
    # 创建一个表达式 Ne(x*(x + y), x**2 + x*y)
    a = Ne(x*(x + y), x**2 + x*y)
    # 断言简化后的结果为 False
    assert simplify(a) == False


def test_negated_property():
    # 创建一个等式 Eq(x, y)
    eq = Eq(x, y)
    # 断言其否定形式为 Ne(x, y)
    assert eq.negated == Ne(x, y)

    # 创建一个不等式 Ne(x, y)
    eq = Ne(x, y)
    # 断言其否定形式为 Eq(x, y)
    assert eq.negated == Eq(x, y)

    # 创建一个大于等于不等式 Ge(x + y, y - x)
    eq = Ge(x + y, y - x)
    # 断言其否定形式为 Lt(x + y, y - x)
    assert eq.negated == Lt(x + y, y - x)

    # 遍历所有比较运算函数，验证双重否定的效果
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).negated.negated == f(x, y)


def test_reversedsign_property():
    # 创建一个等式 Eq(x, y)
    eq = Eq(x, y)
    # 断言其反号形式为 Eq(-x, -y)
    assert eq.reversedsign == Eq(-x, -y)

    # 创建一个不等式 Ne(x, y)
    eq = Ne(x, y)
    # 断言其反号形式为 Ne(-x, -y)
    assert eq.reversedsign == Ne(-x, -y)

    # 创建一个大于等于不等式 Ge(x + y, y - x)
    eq = Ge(x + y, y - x)
    # 断言其反号形式为 Le(-x - y, x - y)
    assert eq.reversedsign == Le(-x - y, x - y)

    # 遍历所有比较运算函数，验证双重反号的效果
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).reversedsign.reversedsign == f(x, y)

    # 验证对 x 进行反号的情况
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, y).reversedsign.reversedsign == f(-x, y)

    # 验证对 y 进行反号的情况
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, -y).reversedsign.reversedsign == f(x, -y)

    # 验证对 x, y 同时进行反号的情况
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, -y).reversedsign.reversedsign == f(-x, -y)


def test_reversed_reversedsign_property():
    # 遍历所有比较运算函数，验证反向和双重反号的对称性
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).reversed.reversedsign == f(x, y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, y).reversed.reversedsign == f(-x, y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, -y).reversed.reversedsign == f(x, -y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, -y).reversed.reversedsign == \
            f(-x, -y).reversedsign.reversed


def test_improved_canonical():
    # 定义一个函数，测试不同形式的表达式是否具有相同的 canonical 形式
    def test_different_forms(listofforms):
        for form1, form2 in combinations(listofforms, 2):
            assert form1.canonical == form2.canonical

    # 生成不同的表达式形式列表
    def generate_forms(expr):
        return [expr, expr.reversed, expr.reversedsign,
                expr.reversed.reversedsign]

    # 对各种形式的比较表达式进行测试
    test_different_forms(generate_forms(x > -y))
    test_different_forms(generate_forms(x >= -y))
    test_different_forms(generate_forms(Eq(x, -y)))
    test_different_forms(generate_forms(Ne(x, -y)))
    test_different_forms(generate_forms(pi < x))
    test_different_forms(generate_forms(pi - 5*y < -x + 2*y**2 - 7))

    # 断言 pi >= x 的 canonical 形式与 x <= pi 相同
    assert (pi >= x).canonical == (x <= pi)


def test_set_equality_canonical():
    # 定义符号变量
    a, b, c = symbols('a b c')

    # 创建两个集合的等式和不等式
    A = Eq(FiniteSet(a, b, c), FiniteSet(1, 2, 3))
    B = Ne(FiniteSet(a, b, c), FiniteSet(4, 5, 6))

    # 断言集合等式和其反向形式相同
    assert A.canonical == A.reversed
    # 断言集合不等式和其反向形式相同
    assert B.canonical == B.reversed


def test_trigsimp():
    # issue 16736
    # 定义正弦和余弦函数
    s, c = sin(2*x), cos(2*x)
    # 创建一个等式 Eq(s, c)
    eq = Eq(s, c)
    # 断言对三角函数进行简化后结果不变
    assert trigsimp(eq) == eq  # no rearrangement of sides

    # 对 sides 进行简化可能导致未评估的等式
    changed = trigsimp(Eq(s + c, sqrt(2)))
    # 断言结果为 Eq 对象
    assert isinstance(changed, Eq)
    # 断言代入 pi/8 后的结果为 S.true
    assert changed.subs(x, pi/8) is S.true

    # 或者结果可能是已评估的等式
    assert trigsimp(Eq(cos(x)**2 + sin(x)**2, 1)) is S.true


def test_polynomial_relation_simplification():
    # 略，未提供具体实现
    # 断言：验证表达式 Ge(3*x*(x + 1) + 4, 3*x).simplify() 是否在指定列表中
    assert Ge(3*x*(x + 1) + 4, 3*x).simplify() in [Ge(x**2, -Rational(4,3)), Le(-x**2, Rational(4, 3))]
    
    # 断言：验证表达式 Le(-(3*x*(x + 1) + 4), -3*x).simplify() 是否在指定列表中
    assert Le(-(3*x*(x + 1) + 4), -3*x).simplify() in [Ge(x**2, -Rational(4,3)), Le(-x**2, Rational(4, 3))]
    
    # 断言：验证表达式 ((x**2+3)*(x**2-1)+3*x >= 2*x**2).simplify() 是否在指定列表中
    assert ((x**2+3)*(x**2-1)+3*x >= 2*x**2).simplify() in [(x**4 + 3*x >= 3), (-x**4 - 3*x <= -3)]
def test_multivariate_linear_function_simplification():
    # 检查 Ge 表达式的简化结果是否正确
    assert Ge(x + y, x - y).simplify() == Ge(y, 0)
    # 检查 Le 表达式的简化结果是否正确
    assert Le(-x + y, -x - y).simplify() == Le(y, 0)
    # 检查 Eq 表达式的简化结果是否为 False
    assert Eq(2*x + y, 2*x + y - 3).simplify() == False
    # 检查大于表达式的简化结果是否为 True
    assert (2*x + y > 2*x + y - 3).simplify() == True
    # 检查小于表达式的简化结果是否为 False
    assert (2*x + y < 2*x + y - 3).simplify() == False
    # 检查小于表达式的简化结果是否为 True
    assert (2*x + y < 2*x + y + 3).simplify() == True
    # 声明符号变量 a, b, c, d, e, f, g
    a, b, c, d, e, f, g = symbols('a b c d e f g')
    # 检查 Lt 表达式的简化结果是否正确
    assert Lt(a + b + c + 2*d, 3*d - f + g).simplify() == Lt(a, -b - c + d - f + g)


def test_nonpolymonial_relations():
    # 检查 cos(x) == 0 表达式的简化结果是否正确
    assert Eq(cos(x), 0).simplify() == Eq(cos(x), 0)


def test_18778():
    # 检查 is_le 函数调用是否引发 TypeError 异常
    raises(TypeError, lambda: is_le(Basic(), Basic()))
    # 检查 is_gt 函数调用是否引发 TypeError 异常
    raises(TypeError, lambda: is_gt(Basic(), Basic()))
    # 检查 is_ge 函数调用是否引发 TypeError 异常
    raises(TypeError, lambda: is_ge(Basic(), Basic()))
    # 检查 is_lt 函数调用是否引发 TypeError 异常
    raises(TypeError, lambda: is_lt(Basic(), Basic()))


def test_EvalEq():
    """
    This test exists to ensure backwards compatibility.
    The method to use is _eval_is_eq
    """
    # 导入所需的符号表达式类
    from sympy.core.expr import Expr

    # 定义 PowTest 类，继承自 Expr
    class PowTest(Expr):
        def __new__(cls, base, exp):
           return Basic.__new__(PowTest, _sympify(base), _sympify(exp))

        # 定义 _eval_Eq 方法，用于比较两个 PowTest 对象是否相等
        def _eval_Eq(lhs, rhs):
            if type(lhs) == PowTest and type(rhs) == PowTest:
                return lhs.args[0] == rhs.args[0] and lhs.args[1] == rhs.args[1]

    # 检查 is_eq 函数对 PowTest 对象的比较是否正确
    assert is_eq(PowTest(3, 4), PowTest(3,4))
    # 检查 is_eq 函数对 PowTest 对象和 _sympify(4) 的比较是否正确
    assert is_eq(PowTest(3, 4), _sympify(4)) is None
    # 检查 is_neq 函数对 PowTest 对象的比较是否正确
    assert is_neq(PowTest(3, 4), PowTest(3,7))


def test_is_eq():
    # 检查 is_eq 函数在给定的假设条件下的行为是否符合预期
    assert is_eq(x, y, Q.infinite(x) & Q.finite(y)) is False
    assert is_eq(x, y, Q.infinite(x) & Q.infinite(y) & Q.extended_real(x) & ~Q.extended_real(y)) is False
    assert is_eq(x, y, Q.infinite(x) & Q.infinite(y) & Q.extended_positive(x) & Q.extended_negative(y)) is False

    assert is_eq(x+I, y+I, Q.infinite(x) & Q.finite(y)) is False
    assert is_eq(1+x*I, 1+y*I, Q.infinite(x) & Q.finite(y)) is False

    assert is_eq(x, S(0), assumptions=Q.zero(x))
    assert is_eq(x, S(0), assumptions=~Q.zero(x)) is False
    assert is_eq(x, S(0), assumptions=Q.nonzero(x)) is False
    assert is_neq(x, S(0), assumptions=Q.zero(x)) is False
    assert is_neq(x, S(0), assumptions=~Q.zero(x))
    assert is_neq(x, S(0), assumptions=Q.nonzero(x))

    # 定义 PowTest 类，继承自 Expr
    class PowTest(Expr):
        def __new__(cls, base, exp):
            return Basic.__new__(cls, _sympify(base), _sympify(exp))

    # 注册 _eval_is_eq 方法用于 PowTest 类对象的比较
    @dispatch(PowTest, PowTest)
    def _eval_is_eq(lhs, rhs):
        if type(lhs) == PowTest and type(rhs) == PowTest:
            return fuzzy_and([is_eq(lhs.args[0], rhs.args[0]), is_eq(lhs.args[1], rhs.args[1])])

    # 检查 is_eq 函数对 PowTest 对象的比较是否正确
    assert is_eq(PowTest(3, 4), PowTest(3,4))
    # 检查 is_eq 函数对 PowTest 对象和 _sympify(4) 的比较是否正确
    assert is_eq(PowTest(3, 4), _sympify(4)) is None
    # 检查 is_neq 函数对 PowTest 对象的比较是否正确
    assert is_neq(PowTest(3, 4), PowTest(3,7))


def test_is_ge_le():
    # 检查 is_ge 函数在给定的假设条件下的行为是否符合预期
    assert is_ge(x, S(0), Q.nonnegative(x)) is True
    assert is_ge(x, S(0), Q.negative(x)) is False

    # 注册相关函数或方法
    # 定义 PowTest 类，继承自 Expr 类
    class PowTest(Expr):
        # 定义 __new__ 方法，用于创建 PowTest 实例
        def __new__(cls, base, exp):
            # 调用 Basic 类的 __new__ 方法，确保 base 和 exp 被 sympify 处理
            return Basic.__new__(cls, _sympify(base), _sympify(exp))

    # 使用 dispatch 装饰器定义 _eval_is_ge 函数，处理 PowTest 类的实例
    @dispatch(PowTest, PowTest)
    def _eval_is_ge(lhs, rhs):
        # 检查 lhs 和 rhs 是否都是 PowTest 类的实例
        if type(lhs) == PowTest and type(rhs) == PowTest:
            # 返回 lhs 和 rhs 的各自成员的不等式关系
            return fuzzy_and([is_ge(lhs.args[0], rhs.args[0]), is_ge(lhs.args[1], rhs.args[1])])

    # 断言 PowTest(3, 9) >= PowTest(3, 2)
    assert is_ge(PowTest(3, 9), PowTest(3,2))
    # 断言 PowTest(3, 9) > PowTest(3, 2)
    assert is_gt(PowTest(3, 9), PowTest(3,2))
    # 断言 PowTest(3, 2) <= PowTest(3, 9)
    assert is_le(PowTest(3, 2), PowTest(3,9))
    # 断言 PowTest(3, 2) < PowTest(3, 9)
    assert is_lt(PowTest(3, 2), PowTest(3,9))
# 定义一个测试函数，用于测试符号表达式的比较操作符的行为
def test_weak_strict():
    # 遍历 Eq 和 Ne 函数
    for func in (Eq, Ne):
        # 创建一个符号表达式，比较 x 是否等于 1
        eq = func(x, 1)
        # 断言其 strict、weak 和其本身相等
        assert eq.strict == eq.weak == eq
    # 创建一个大于表达式，比较 x 是否大于 1
    eq = Gt(x, 1)
    # 断言其 weak 等价于大于等于表达式 Ge(x, 1)
    assert eq.weak == Ge(x, 1)
    # 断言其 strict 等于其本身
    assert eq.strict == eq
    # 创建一个小于表达式，比较 x 是否小于 1
    eq = Lt(x, 1)
    # 断言其 weak 等价于小于等于表达式 Le(x, 1)
    assert eq.weak == Le(x, 1)
    # 断言其 strict 等于其本身
    assert eq.strict == eq
    # 创建一个大于等于表达式，比较 x 是否大于等于 1
    eq = Ge(x, 1)
    # 断言其 strict 等价于大于表达式 Gt(x, 1)
    assert eq.strict == Gt(x, 1)
    # 断言其 weak 等于其本身
    assert eq.weak == eq
    # 创建一个小于等于表达式，比较 x 是否小于等于 1
    eq = Le(x, 1)
    # 断言其 strict 等价于小于表达式 Lt(x, 1)
    assert eq.strict == Lt(x, 1)
    # 断言其 weak 等于其本身
    assert eq.weak == eq


# 定义一个测试函数，用于测试符号表达式在不同情况下的保持性
def test_issue_23731():
    # 创建一个整数符号变量 i
    i = symbols('i', integer=True)
    # 断言符号表达式 Eq(i, 1.0) 的保持性
    assert unchanged(Eq, i, 1.0)
    # 断言符号表达式 Eq(i/2, 0.5) 的保持性
    assert unchanged(Eq, i/2, 0.5)
    # 创建一个非整数符号变量 ni
    ni = symbols('ni', integer=False)
    # 断言非整数符号表达式 Eq(ni, 1) 的结果为 False
    assert Eq(ni, 1) == False
    # 断言符号表达式 Eq(ni, .1) 的保持性
    assert unchanged(Eq, ni, .1)
    # 断言非整数符号表达式 Eq(ni, 1.0) 的结果为 False
    assert Eq(ni, 1.0) == False
    # 创建一个有理数非符号变量 nr
    nr = symbols('nr', rational=False)
    # 断言非有理数符号表达式 Eq(nr, .1) 的结果为 False
    assert Eq(nr, .1) == False


# 定义一个测试函数，测试符号表达式在重写 Add 操作时的行为
def test_rewrite_Add():
    # 导入警告函数，以便捕获符号表达式重写 Add 操作的警告
    from sympy.testing.pytest import warns_deprecated_sympy
    # 在捕获到符号表达式重写 Add 操作的警告时进行断言，确保重写结果为 x - y
    with warns_deprecated_sympy():
        assert Eq(x, y).rewrite(Add) == x - y
```