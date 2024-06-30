# `D:\src\scipysrc\sympy\sympy\polys\domains\tests\test_polynomialring.py`

```
"""Tests for the PolynomialRing classes. """

# 导入必要的模块和函数
from sympy.polys.domains import QQ, ZZ  # 导入 QQ 和 ZZ 域
from sympy.polys.polyerrors import ExactQuotientFailed, CoercionFailed, NotReversible  # 导入异常类

from sympy.abc import x, y  # 导入符号变量 x 和 y

from sympy.testing.pytest import raises  # 导入测试函数 raises


def test_build_order():
    # 创建一个多项式环 R，包含变量 x 和 y，使用 lex 和 ilex 排序
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y)))
    # 断言排序后的结果符合预期
    assert R.order((1, 5)) == ((1,), (-5,))


def test_globalring():
    # 创建有理函数域 Qxy 和多项式环 R，包含变量 x 和 y
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y)
    X = R.convert(x)
    Y = R.convert(y)

    # 断言 x 在 R 中
    assert x in R
    # 断言 1/x 不在 R 中
    assert 1/x not in R
    # 断言 1/(1 + x) 不在 R 中
    assert 1/(1 + x) not in R
    # 断言 Y 在 R 中
    assert Y in R
    # 断言 X * (Y**2 + 1) 等于 R.convert(x * (y**2 + 1))
    assert X * (Y**2 + 1) == R.convert(x * (y**2 + 1))
    # 断言 X + 1 等于 R.convert(x + 1)
    assert X + 1 == R.convert(x + 1)
    # 断言 X/Y 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: X/Y)
    # 断言 x/Y 抛出 TypeError 异常
    raises(TypeError, lambda: x/Y)
    # 断言 X/y 抛出 TypeError 异常
    raises(TypeError, lambda: X/y)
    # 断言 X**2 / X 等于 X
    assert X**2 / X == X

    # 断言 R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) 等于 X
    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    # 断言 R.from_FractionField(Qxy.convert(x), Qxy) 等于 X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    # 断言 R.from_FractionField(Qxy.convert(x/y), Qxy) 为 None
    assert R.from_FractionField(Qxy.convert(x/y), Qxy) is None

    # 断言 R._sdm_to_vector(R._vector_to_sdm([X, Y], R.order), 2) 等于 [X, Y]
    assert R._sdm_to_vector(R._vector_to_sdm([X, Y], R.order), 2) == [X, Y]


def test_localring():
    # 创建有理函数域 Qxy 和多项式环 R，包含变量 x 和 y，使用 ilex 排序
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y, order="ilex")
    X = R.convert(x)
    Y = R.convert(y)

    # 断言 x 在 R 中
    assert x in R
    # 断言 1/x 不在 R 中
    assert 1/x not in R
    # 断言 1/(1 + x) 在 R 中
    assert 1/(1 + x) in R
    # 断言 Y 在 R 中
    assert Y in R
    # 断言 X*(Y**2 + 1)/(1 + X) 等于 R.convert(x*(y**2 + 1)/(1 + x))
    assert X*(Y**2 + 1)/(1 + X) == R.convert(x*(y**2 + 1)/(1 + x))
    # 断言 x/Y 抛出 TypeError 异常
    raises(TypeError, lambda: x/Y)
    # 断言 X/y 抛出 TypeError 异常
    raises(TypeError, lambda: X/y)
    # 断言 X + 1 等于 R.convert(x + 1)
    assert X + 1 == R.convert(x + 1)
    # 断言 X**2 / X 等于 X
    assert X**2 / X == X

    # 断言 R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) 等于 X
    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    # 断言 R.from_FractionField(Qxy.convert(x), Qxy) 等于 X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    # 断言 R.from_FractionField(Qxy.convert(x/y), Qxy) 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: R.from_FractionField(Qxy.convert(x/y), Qxy))
    # 断言 R.exquo(X, Y) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: R.exquo(X, Y))
    # 断言 R.revert(X) 抛出 NotReversible 异常
    raises(NotReversible, lambda: R.revert(X))

    # 断言 R._sdm_to_vector(R._vector_to_sdm([X/(X + 1), Y/(1 + X*Y)], R.order), 2) 等于 [X*(1 + X*Y), Y*(1 + X)]
    assert R._sdm_to_vector(
        R._vector_to_sdm([X/(X + 1), Y/(1 + X*Y)], R.order), 2) == \
        [X*(1 + X*Y), Y*(1 + X)]


def test_conversion():
    # 创建多项式环 L 和 G，包含变量 x 和 y
    L = QQ.old_poly_ring(x, y, order="ilex")
    G = QQ.old_poly_ring(x, y)

    # 断言 L.convert(x) 等于 L.convert(G.convert(x), G)
    assert L.convert(x) == L.convert(G.convert(x), G)
    # 断言 G.convert(x) 等于 G.convert(L.convert(x), L)
    assert G.convert(x) == G.convert(L.convert(x), L)
    # 断言 G.convert(L.convert(1/(1 + x)), L) 抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: G.convert(L.convert(1/(1 + x)), L))


def test_units():
    # 创建整数多项式环 R，包含变量 x
    R = QQ.old_poly_ring(x)
    # 断言 R.is_unit(R.convert(1)) 为 True
    assert R.is_unit(R.convert(1))
    # 断言 R.is_unit(R.convert(2)) 为 True
    assert R.is_unit(R.convert(2))
    # 断言 R.is_unit(R.convert(x)) 为 False
    assert not R.is_unit(R.convert(x))
    # 断言 R.is_unit(R.convert(1 + x)) 为 False
    assert not R.is_unit(R.convert(1 + x))

    # 创建整数多项式环 R，包含变量 x，使用 ilex 排序
    R = QQ.old_poly_ring(x, order='ilex')
    # 断言 R.is_unit(R.convert(1)) 为 True
    assert R.is_unit(R.convert(1))
    # 断言 R.is_unit(R.convert(2)) 为 True
    assert R.is_unit(R.convert(2))
    # 断言 R.is_unit(R.convert(x)) 为 False
    assert not R.is_unit(R.convert(x))
    # 断言 R.is_unit(R.convert(1 + x)) 为 True
    assert R.is_unit(R.convert(1 + x))

    # 创建整数多项式环 R，包含变量 x
    R = ZZ.old_poly_ring(x)
    # 断言 R.is_unit(R.convert(1)) 为 True
    assert R.is_unit(R.convert(1))
    # 断言 R.is_unit(R.convert(2)) 为 False
    assert not R.is_unit(R.convert(2))
    # 断言 R.is_unit(R.convert(x)) 为 False
    assert not R.is_unit(R.convert(x))
    # 断言 R.is_unit(R.convert(
```