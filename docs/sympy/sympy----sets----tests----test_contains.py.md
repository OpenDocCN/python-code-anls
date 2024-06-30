# `D:\src\scipysrc\sympy\sympy\sets\tests\test_contains.py`

```
from sympy.core.expr import unchanged  # 导入 unchanged 函数，用于测试对象是否未更改
from sympy.core.numbers import oo  # 导入 oo（无穷大）常数
from sympy.core.relational import Eq  # 导入关系运算符 Eq
from sympy.core.singleton import S  # 导入单例对象 S
from sympy.core.symbol import Symbol  # 导入符号对象 Symbol
from sympy.sets.contains import Contains  # 导入 Contains 类，用于表示包含关系
from sympy.sets.sets import (FiniteSet, Interval)  # 导入有限集合和区间对象
from sympy.testing.pytest import raises  # 导入 raises 函数，用于测试是否引发异常


def test_contains_basic():
    raises(TypeError, lambda: Contains(S.Integers, 1))  # 断言引发 TypeError 异常，因为 S.Integers 不包含整数 1
    assert Contains(2, S.Integers) is S.true  # 断言整数 2 在 S.Integers 中为真
    assert Contains(-2, S.Naturals) is S.false  # 断言整数 -2 不在 S.Naturals 中为假

    i = Symbol('i', integer=True)
    assert Contains(i, S.Naturals) == Contains(i, S.Naturals, evaluate=False)  # 断言符号 i 在 S.Naturals 中的包含关系，关闭评估


def test_issue_6194():
    x = Symbol('x')
    assert unchanged(Contains, x, Interval(0, 1))  # 断言 unchanged 函数对 Contains(x, Interval(0, 1)) 的结果
    assert Interval(0, 1).contains(x) == (S.Zero <= x) & (x <= 1)  # 断言 Interval(0, 1) 是否包含符号 x，使用逻辑与运算符表示包含关系
    assert Contains(x, FiniteSet(0)) != S.false  # 断言符号 x 是否在 FiniteSet(0) 中，不等于假
    assert Contains(x, Interval(1, 1)) != S.false  # 断言符号 x 是否在区间 Interval(1, 1) 中，不等于假
    assert Contains(x, S.Integers) != S.false  # 断言符号 x 是否在 S.Integers 中，不等于假


def test_issue_10326():
    assert Contains(oo, Interval(-oo, oo)) == False  # 断言无穷大 oo 是否在区间 Interval(-oo, oo) 中为假
    assert Contains(-oo, Interval(-oo, oo)) == False  # 断言负无穷大 -oo 是否在区间 Interval(-oo, oo) 中为假


def test_binary_symbols():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert Contains(x, FiniteSet(y, Eq(z, True))  # 断言 x 是否在包含 y 和 Eq(z, True) 的 FiniteSet 中
        ).binary_symbols == {y, z}  # 断言包含关系对象的二元符号集合为 {y, z}


def test_as_set():
    x = Symbol('x')
    y = Symbol('y')
    assert Contains(x, FiniteSet(y)).as_set() == FiniteSet(y)  # 断言包含关系 x 是否与 FiniteSet(y) 的集合形式相等
    assert Contains(x, S.Integers).as_set() == S.Integers  # 断言包含关系 x 是否与 S.Integers 的集合形式相等
    assert Contains(x, S.Reals).as_set() == S.Reals  # 断言包含关系 x 是否与 S.Reals 的集合形式相等


def test_type_error():
    raises(TypeError, lambda: Contains(2, None))  # 断言传递给 Contains 的第一个参数不是集合类型时会引发 TypeError 异常
```