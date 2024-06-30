# `D:\src\scipysrc\sympy\sympy\polys\domains\tests\test_quotientring.py`

```
"""Tests for quotient rings."""

# 导入整数环 ZZ 和有理数域 QQ 的相关定义
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
# 导入符号 x 和 y
from sympy.abc import x, y
# 导入异常类 NotReversible
from sympy.polys.polyerrors import NotReversible
# 导入 pytest 的 raises 函数用于测试异常
from sympy.testing.pytest import raises


# 定义测试函数 test_QuotientRingElement
def test_QuotientRingElement():
    # 创建一个关于 x 的多项式环，并构造模 [x**10] 的商环 R
    R = QQ.old_poly_ring(x)/[x**10]
    # 将符号 x 转换为 R 中的元素 X
    X = R.convert(x)

    # 断言测试
    assert X*(X + 1) == R.convert(x**2 + x)
    assert X*x == R.convert(x**2)
    assert x*X == R.convert(x**2)
    assert X + x == R.convert(2*x)
    assert x + X == 2*X
    assert X**2 == R.convert(x**2)
    assert 1/(1 - X) == R.convert(sum(x**i for i in range(10)))
    assert X**10 == R.zero
    assert X != x

    # 断言引发异常 NotReversible
    raises(NotReversible, lambda: 1/X)


# 定义测试函数 test_QuotientRing
def test_QuotientRing():
    # 创建一个关于 x 的多项式环，并构造理想 I = [x**2 + 1]
    I = QQ.old_poly_ring(x).ideal(x**2 + 1)
    # 构造模 [x**2 + 1] 的商环 R
    R = QQ.old_poly_ring(x)/I

    # 断言测试
    assert R == QQ.old_poly_ring(x)/[x**2 + 1]
    assert R == QQ.old_poly_ring(x)/QQ.old_poly_ring(x).ideal(x**2 + 1)
    assert R != QQ.old_poly_ring(x)

    # 断言测试
    assert R.convert(1)/x == -x + I
    assert -1 + I == x**2 + I
    assert R.convert(ZZ(1), ZZ) == 1 + I
    assert R.convert(R.convert(x), R) == R.convert(x)

    # 将符号 x 转换为 R 中的元素 X 和 QQ.old_poly_ring(x) 中的元素 Y 进行比较
    X = R.convert(x)
    Y = QQ.old_poly_ring(x).convert(x)
    assert -1 + I == X**2 + I
    assert -1 + I == Y**2 + I
    assert R.to_sympy(X) == x

    # 断言引发异常 ValueError
    raises(ValueError, lambda: QQ.old_poly_ring(x)/QQ.old_poly_ring(x, y).ideal(x))

    # 创建关于 x 和 y 的多项式环 R，并构造理想 I = [x]
    R = QQ.old_poly_ring(x, order="ilex")
    I = R.ideal(x)
    # 断言测试
    assert R.convert(1) + I == (R/I).convert(1)
```