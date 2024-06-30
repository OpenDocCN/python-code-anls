# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_subfield.py`

```
# 导入所需的模块和函数

from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
    is_isomorphism_possible,
    field_isomorphism_pslq,
    field_isomorphism,
    primitive_element,
    to_number_field,
)
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises

from sympy.abc import x

Q = Rational  # 定义 Rational 类型的简写 Q

# 定义测试函数 test_field_isomorphism_pslq，用于测试 field_isomorphism_pslq 函数
def test_field_isomorphism_pslq():
    # 创建 AlgebraicNumber 实例 a 和 b
    a = AlgebraicNumber(I)
    b = AlgebraicNumber(I * sqrt(3))

    # 检查在这些数域上的同构是否可能引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: field_isomorphism_pslq(a, b))

    # 创建多个 AlgebraicNumber 实例 a, b, c, d, e
    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))
    d = AlgebraicNumber(sqrt(2) + sqrt(3))
    e = AlgebraicNumber(sqrt(2) + sqrt(3) + sqrt(7))

    # 断言检查不同数域之间的同构关系
    assert field_isomorphism_pslq(a, a) == [1, 0]
    assert field_isomorphism_pslq(a, b) is None
    assert field_isomorphism_pslq(a, c) is None
    assert field_isomorphism_pslq(a, d) == [Q(1, 2), 0, -Q(9, 2), 0]
    assert field_isomorphism_pslq(a, e) == [Q(1, 80), 0, -Q(1, 2), 0, Q(59, 20), 0]

    assert field_isomorphism_pslq(b, a) is None
    assert field_isomorphism_pslq(b, b) == [1, 0]
    assert field_isomorphism_pslq(b, c) is None
    assert field_isomorphism_pslq(b, d) == [-Q(1, 2), 0, Q(11, 2), 0]
    assert field_isomorphism_pslq(b, e) == [-Q(3, 640), 0, Q(67, 320), 0, -Q(297, 160), 0, Q(313, 80), 0]

    assert field_isomorphism_pslq(c, a) is None
    assert field_isomorphism_pslq(c, b) is None
    assert field_isomorphism_pslq(c, c) == [1, 0]
    assert field_isomorphism_pslq(c, d) is None
    assert field_isomorphism_pslq(c, e) == [Q(3, 640), 0, -Q(71, 320), 0, Q(377, 160), 0, -Q(469, 80), 0]

    assert field_isomorphism_pslq(d, a) is None
    assert field_isomorphism_pslq(d, b) is None
    assert field_isomorphism_pslq(d, c) is None
    assert field_isomorphism_pslq(d, d) == [1, 0]
    assert field_isomorphism_pslq(d, e) == [-Q(3, 640), 0, Q(71, 320), 0, -Q(377, 160), 0, Q(549, 80), 0]

    assert field_isomorphism_pslq(e, a) is None
    assert field_isomorphism_pslq(e, b) is None
    assert field_isomorphism_pslq(e, c) is None
    assert field_isomorphism_pslq(e, d) is None
    assert field_isomorphism_pslq(e, e) == [1, 0]

    # 创建 AlgebraicNumber 实例 f
    f = AlgebraicNumber(3 * sqrt(2) + 8 * sqrt(7) - 5)

    # 断言检查 f 和 e 之间的同构关系
    assert field_isomorphism_pslq(f, e) == [Q(3, 80), 0, -Q(139, 80), 0, Q(347, 20), 0, -Q(761, 20), -5]


# 定义测试函数 test_field_isomorphism，用于测试 field_isomorphism 函数
def test_field_isomorphism():
    # 断言检查不同数域之间的同构关系
    assert field_isomorphism(3, sqrt(2)) == [3]

    assert field_isomorphism(I * sqrt(3), I * sqrt(3) / 2) == [2, 0]
    assert field_isomorphism(-I * sqrt(3), I * sqrt(3) / 2) == [-2, 0]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism( I*sqrt(3), -I*sqrt(3)/2) == [-2, 0]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(-I*sqrt(3), -I*sqrt(3)/2) == [ 2, 0]

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism( 2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [ Rational(6, 35), 0]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(-2*I*sqrt(3)/7, 5*I*sqrt(3)/3) == [Rational(-6, 35), 0]

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism( 2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [Rational(-6, 35), 0]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(-2*I*sqrt(3)/7, -5*I*sqrt(3)/3) == [ Rational(6, 35), 0]

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [ Rational(6, 35), 27]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, 5*I*sqrt(3)/3) == [Rational(-6, 35), 27]

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(
        2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [Rational(-6, 35), 27]
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期
    assert field_isomorphism(
        -2*I*sqrt(3)/7 + 27, -5*I*sqrt(3)/3) == [ Rational(6, 35), 27]

    # 创建代数数对象 p, q, r, s，分别表示 sqrt(2) + sqrt(3), -sqrt(2) + sqrt(3), sqrt(2) - sqrt(3), -sqrt(2) - sqrt(3)
    p = AlgebraicNumber( sqrt(2) + sqrt(3))
    q = AlgebraicNumber(-sqrt(2) + sqrt(3))
    r = AlgebraicNumber( sqrt(2) - sqrt(3))
    s = AlgebraicNumber(-sqrt(2) - sqrt(3))

    # 定义正系数列表和负系数列表
    pos_coeffs = [ S.Half, S.Zero, Rational(-9, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(9, 2), S.Zero]

    # 创建代数数对象 a，表示 sqrt(2)
    a = AlgebraicNumber(sqrt(2))

    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, p) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, q) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, r) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, s) is True

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    # 重新定义代数数对象 a，表示 -sqrt(2)
    a = AlgebraicNumber(-sqrt(2))

    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, p) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, q) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, r) is True
    # 断言：验证 is_isomorphism_possible 函数对给定参数的返回结果是否为 True
    assert is_isomorphism_possible(a, s) is True

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，使用 fast 模式
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    # 断言：验证 field_isomorphism 函数对给定参数的返回结果是否符合预期，不使用 fast 模式
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    # 更新正系数列表和负系数列表
    pos_coeffs = [ S.Half, S.Zero, Rational(-11, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(11,
    # 使用快速模式检查域同构性，并断言结果与预期的负系数列表相等
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    # 使用非快速模式检查域同构性，并断言结果与预期的负系数列表相等
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    # 创建一个代数数对象，表示-algebraic.sqrt(3)
    a = AlgebraicNumber(-sqrt(3))

    # 断言在给定域上是否可能存在同构映射
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    # 使用快速模式检查域同构性，并断言结果与预期的正系数列表相等
    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs

    # 使用非快速模式检查域同构性，并断言结果与预期的正系数列表相等
    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs

    # 定义正负系数列表，分别表示正系数和负系数的多项式系数
    pos_coeffs = [ Rational(3, 2), S.Zero, Rational(-33, 2), -S(8)]
    neg_coeffs = [Rational(-3, 2), S.Zero, Rational(33, 2), -S(8)]

    # 创建一个代数数对象，表示3*algebraic.sqrt(3) - 8
    a = AlgebraicNumber(3*sqrt(3) - 8)

    # 断言在给定域上是否可能存在同构映射
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    # 使用快速模式检查域同构性，并断言结果与预期的负系数列表相等
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs

    # 使用非快速模式检查域同构性，并断言结果与预期的负系数列表相等
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs

    # 创建一个代数数对象，表示3*algebraic.sqrt(2) + 2*algebraic.sqrt(3) + 1
    a = AlgebraicNumber(3*sqrt(2) + 2*sqrt(3) + 1)

    # 定义不同情况下的正负系数列表
    pos_1_coeffs = [ S.Half, S.Zero, Rational(-5, 2), S.One]
    neg_5_coeffs = [Rational(-5, 2), S.Zero, Rational(49, 2), S.One]
    pos_5_coeffs = [ Rational(5, 2), S.Zero, Rational(-49, 2), S.One]
    neg_1_coeffs = [Rational(-1, 2), S.Zero, Rational(5, 2), S.One]

    # 断言在给定域上是否可能存在同构映射
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True

    # 使用快速模式检查域同构性，并断言结果与预期的系数列表相等
    assert field_isomorphism(a, p, fast=True) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_1_coeffs

    # 使用非快速模式检查域同构性，并断言结果与预期的系数列表相等
    assert field_isomorphism(a, p, fast=False) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_5_coeffs
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，期望返回 `pos_5_coeffs`
    assert field_isomorphism(a, r, fast=False) == pos_5_coeffs
    
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，期望返回 `neg_1_coeffs`
    assert field_isomorphism(a, s, fast=False) == neg_1_coeffs
    
    # 创建三个代数数对象，分别对应 sqrt(2), sqrt(3), sqrt(7)
    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))
    
    # 断言：检查是否可以在代数数之间建立同构映射，期望返回 True
    assert is_isomorphism_possible(a, b) is True
    # 断言：检查是否可以在代数数之间建立同构映射，期望返回 True
    assert is_isomorphism_possible(b, a) is True
    
    # 断言：检查是否可以在代数数之间建立同构映射，期望返回 False
    assert is_isomorphism_possible(c, p) is False
    
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，快速模式下期望返回 None
    assert field_isomorphism(sqrt(2), sqrt(3), fast=True) is None
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，快速模式下期望返回 None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=True) is None
    
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，非快速模式下期望返回 None
    assert field_isomorphism(sqrt(2), sqrt(3), fast=False) is None
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，非快速模式下期望返回 None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=False) is None
    
    # 重新定义代数数对象 a 和 b，分别对应 sqrt(2) 和 2^(1/3)
    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(2 ** (S(1) / 3))
    
    # 断言：检查是否可以在代数数之间建立同构映射，期望返回 False
    assert is_isomorphism_possible(a, b) is False
    # 断言：使用 `field_isomorphism` 函数检查是否满足非同构条件，期望返回 None
    assert field_isomorphism(a, b) is None
# 定义测试函数 `test_primitive_element()`，用于测试 `primitive_element` 函数的不同用法和参数组合
def test_primitive_element():
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2)] 和变量 x，期望返回结果为 (x**2 - 2, [1])
    assert primitive_element([sqrt(2)], x) == (x**2 - 2, [1])
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), sqrt(3)] 和变量 x，期望返回结果为 (x**4 - 10*x**2 + 1, [1, 1])
    assert primitive_element([sqrt(2), sqrt(3)], x) == (x**4 - 10*x**2 + 1, [1, 1])

    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2)]、变量 x，并设置 polys=True，期望返回结果为 (Poly(x**2 - 2, domain='QQ'), [1])
    assert primitive_element([sqrt(2)], x, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1])
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), sqrt(3)]、变量 x，并设置 polys=True，期望返回结果为 (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1])
    assert primitive_element([sqrt(2), sqrt(3)], x, polys=True) == (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1])

    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2)]、变量 x，并设置 ex=True，期望返回结果为 (x**2 - 2, [1], [[1, 0]])
    assert primitive_element([sqrt(2)], x, ex=True) == (x**2 - 2, [1], [[1, 0]])
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), sqrt(3)]、变量 x，并设置 ex=True，期望返回结果为 (x**4 - 10*x**2 + 1, [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True) == \
        (x**4 - 10*x**2 + 1, [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])

    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2)]、变量 x，并设置 ex=True、polys=True，期望返回结果为 (Poly(x**2 - 2, domain='QQ'), [1], [[1, 0]])
    assert primitive_element([sqrt(2)], x, ex=True, polys=True) == (Poly(x**2 - 2, domain='QQ'), [1], [[1, 0]])
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), sqrt(3)]、变量 x，并设置 ex=True、polys=True，期望返回结果为 (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True, polys=True) == \
        (Poly(x**4 - 10*x**2 + 1, domain='QQ'), [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])

    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2)]，并设置 polys=True，期望返回结果为 (Poly(x**2 - 2), [1])
    assert primitive_element([sqrt(2)], polys=True) == (Poly(x**2 - 2), [1])

    # 检查是否引发 ValueError 异常，lambda 表达式调用 `primitive_element` 函数，传入空列表、变量 x 和 ex=False，预期会引发异常
    raises(ValueError, lambda: primitive_element([], x, ex=False))
    # 检查是否引发 ValueError 异常，lambda 表达式调用 `primitive_element` 函数，传入空列表、变量 x 和 ex=True，预期会引发异常
    raises(ValueError, lambda: primitive_element([], x, ex=True))

    # Issue 14117 的测试：定义变量 a 和 b，分别为 I*sqrt(2*sqrt(2) + 3) 和 I*sqrt(-2*sqrt(2) + 3)
    a, b = I*sqrt(2*sqrt(2) + 3), I*sqrt(-2*sqrt(2) + 3)
    # 断言检查：调用 `primitive_element` 函数，传入 [a, b, I] 和变量 x，期望返回结果为 (x**4 + 6*x**2 + 1, [1, 0, 0])
    assert primitive_element([a, b, I], x) == (x**4 + 6*x**2 + 1, [1, 0, 0])

    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), 0] 和变量 x，期望返回结果为 (x**2 - 2, [1, 0])
    assert primitive_element([sqrt(2), 0], x) == (x**2 - 2, [1, 0])
    # 断言检查：调用 `primitive_element` 函数，传入 [0, sqrt(2)] 和变量 x，期望返回结果为 (x**2 - 2, [1, 1])
    assert primitive_element([0, sqrt(2)], x) == (x**2 - 2, [1, 1])
    
    # 断言检查：调用 `primitive_element` 函数，传入 [sqrt(2), 0]、变量 x，并设置 ex=True，期望返回结果为 (x**2 - 2, [1, 0], [[MPQ(1,1), MPQ(0,1)], []])
    assert primitive_element([sqrt(2), 0], x, ex=True) == (x**2 - 2, [1, 0], [[MPQ(1,1), MPQ(0,1)], []])
    # 断言检查：调用 `primitive_element` 函数，传入 [0, sqrt(2)]、变量 x，并设置 ex=True，期望返回结果为 (x**2 - 2, [1, 1], [[], [MPQ(1,1), MPQ(0,1)]])
    assert primitive_element([0, sqrt(2)], x, ex=True) == (x**2 - 2, [1, 1], [[], [MPQ(1,1), MPQ(0,1)]])

# 定义测试函数 `test_to_number_field()`，用于测试 `to_number_field` 函数的不同用法和参数组合
def test_to_number_field():
    # 断言检查：调用 `to_number_field` 函数，传入 sqrt(2)，期望返回 AlgebraicNumber(sqrt(2))
    assert to_number_field(sqrt(2)) == AlgebraicNumber(sqrt(2))
    # 断言检查：调用 `to_number_field` 函数，传入 [sqrt(2), sqrt(3)]，期望返回 AlgebraicNumber(sqrt(2) + sqrt(3))
    assert to_number_field([sqrt(2), sqrt(3)]) == AlgebraicNumber(sqrt(2) + sqrt(3))

    # 定义变量 a，为 AlgebraicNumber(sqrt(2) + sqrt(3), [S.Half, S.Zero, Rational(-9, 2), S.Zero])
    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S.Half, S.Zero, Rational(-9, 2), S.Zero])
    # 断言检查：调用 `to_number_field` 函数，传入 sqrt(2) 和 sqrt(2) + sqrt(3)，期望返回结果为 a
```