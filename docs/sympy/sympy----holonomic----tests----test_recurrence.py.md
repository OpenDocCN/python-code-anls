# `D:\src\scipysrc\sympy\sympy\holonomic\tests\test_recurrence.py`

```
# 导入必要的库函数和模块
from sympy.holonomic.recurrence import RecurrenceOperators, RecurrenceOperator
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ

# 定义测试函数，用于验证 RecurrenceOperator 类的功能
def test_RecurrenceOperator():
    # 声明符号变量 n 为整数
    n = symbols('n', integer=True)
    # 创建一个 RecurrenceOperators 对象 R 和一个 RecurrenceOperator 对象 Sn
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    # 第一个断言，验证 Sn * n 是否等于 (n + 1) * Sn
    assert Sn*n == (n + 1)*Sn
    # 第二个断言，验证 Sn * n**2 是否等于 (n**2 + 1 + 2*n) * Sn
    assert Sn*n**2 == (n**2 + 1 + 2*n)*Sn
    # 第三个断言，验证 Sn**2 * n**2 是否等于 (n**2 + 4*n + 4) * Sn**2
    assert Sn**2*n**2 == (n**2 + 4*n + 4)*Sn**2
    # 定义一个多项式 p，验证 Sn**3 * n**2 + Sn * n 的平方是否等于 q
    p = (Sn**3*n**2 + Sn*n)**2
    q = (n**2 + 3*n + 2)*Sn**2 + (2*n**3 + 19*n**2 + 57*n + 52)*Sn**4 + (n**4 + 18*n**3 + \
        117*n**2 + 324*n + 324)*Sn**6
    assert p == q


# 定义另一个测试函数，用于验证 RecurrenceOperator 对象的相等性
def test_RecurrenceOperatorEqPoly():
    # 声明符号变量 n 为整数
    n = symbols('n', integer=True)
    # 创建一个 RecurrenceOperators 对象 R 和一个 RecurrenceOperator 对象 Sn
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    # 创建两个 RecurrenceOperator 对象 rr 和 rr2，验证它们的不等性
    rr = RecurrenceOperator([n**2, 0, 0], R)
    rr2 = RecurrenceOperator([n**2, 1, n], R)
    assert not rr == rr2

    # 由于多项式比较问题，暂时禁用以下断言，详见 https://github.com/sympy/sympy/pull/15799
    # d = rr.listofpoly[0]
    # assert rr == d

    # 验证 rr2 和其内部的多项式对象 d2 是否相等
    d2 = rr2.listofpoly[0]
    assert not rr2 == d2


# 定义第三个测试函数，验证 RecurrenceOperator 对象的乘幂运算
def test_RecurrenceOperatorPow():
    # 声明符号变量 n 为整数
    n = symbols('n', integer=True)
    # 创建一个 RecurrenceOperators 对象 R 和一个 RecurrenceOperator 对象 _
    R, _ = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    # 创建一个 RecurrenceOperator 对象 rr，用于测试乘幂操作
    rr = RecurrenceOperator([n**2, 0, 0], R)
    # 创建一个 RecurrenceOperator 对象 a，初始化为 R.base.one
    a = RecurrenceOperator([R.base.one], R)
    # 对 rr 进行连续的乘幂操作，验证是否与 a 相等
    for m in range(10):
        assert a == rr**m
        a *= rr
```