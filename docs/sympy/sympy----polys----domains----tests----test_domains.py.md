# `D:\src\scipysrc\sympy\sympy\polys\domains\tests\test_domains.py`

```
"""Tests for classes defining properties of ground domains, e.g. ZZ, QQ, ZZ[x] ... """

# 导入符号计算相关模块和类
from sympy.external.gmpy import GROUND_TYPES

from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
    Rational, oo, pi, _illegal)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z

# 导入符号计算中的多项式和域相关模块和类
from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
    ZZ_python, QQ_gmpy, QQ_python)
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField

# 导入符号计算中的数域相关模块和类
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field

# 导入符号计算中的多项式错误相关模块和类
from sympy.polys.polyerrors import (
    UnificationFailed,
    GeneratorsError,
    CoercionFailed,
    NotInvertible,
    DomainError)

# 导入符号计算中的测试工具和函数
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 导入 itertools 中的 product 函数
from itertools import product

# 创建代数域 QQ(sqrt(2), sqrt(3))
ALG = QQ.algebraic_field(sqrt(2), sqrt(3))

# 定义 unify 函数，用于尝试统一两个域对象
def unify(K0, K1):
    return K0.unify(K1)

# 定义测试函数 test_Domain_unify
def test_Domain_unify():
    # 创建有限域 F3 和 F5
    F3 = GF(3)
    F5 = GF(5)

    # 测试有限域 F3 的统一
    assert unify(F3, F3) == F3
    raises(UnificationFailed, lambda: unify(F3, ZZ))
    raises(UnificationFailed, lambda: unify(F3, QQ))
    raises(UnificationFailed, lambda: unify(F3, ZZ_I))
    raises(UnificationFailed, lambda: unify(F3, QQ_I))
    raises(UnificationFailed, lambda: unify(F3, ALG))
    raises(UnificationFailed, lambda: unify(F3, RR))
    raises(UnificationFailed, lambda: unify(F3, CC))
    raises(UnificationFailed, lambda: unify(F3, ZZ[x]))
    raises(UnificationFailed, lambda: unify(F3, ZZ.frac_field(x)))
    raises(UnificationFailed, lambda: unify(F3, EX))

    # 测试有限域 F5 的统一
    assert unify(F5, F5) == F5
    raises(UnificationFailed, lambda: unify(F5, F3))
    raises(UnificationFailed, lambda: unify(F5, F3[x]))
    raises(UnificationFailed, lambda: unify(F5, F3.frac_field(x)))

    # 测试整数环 ZZ 的统一
    raises(UnificationFailed, lambda: unify(ZZ, F3))
    assert unify(ZZ, ZZ) == ZZ
    assert unify(ZZ, QQ) == QQ
    assert unify(ZZ, ALG) == ALG
    assert unify(ZZ, RR) == RR
    assert unify(ZZ, CC) == CC
    assert unify(ZZ, ZZ[x]) == ZZ[x]
    assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ, EX) == EX

    # 测试有理数域 QQ 的统一
    raises(UnificationFailed, lambda: unify(QQ, F3))
    assert unify(QQ, ZZ) == QQ
    assert unify(QQ, QQ) == QQ
    assert unify(QQ, ALG) == ALG
    assert unify(QQ, RR) == RR
    assert unify(QQ, CC) == CC
    assert unify(QQ, ZZ[x]) == QQ[x]
    assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ, EX) == EX
    # 使用 lambda 函数调用 unify(ZZ_I, F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(ZZ_I, F3))
    # 验证 unify(ZZ_I, ZZ) 是否等于 ZZ_I
    assert unify(ZZ_I, ZZ) == ZZ_I
    # 验证 unify(ZZ_I, ZZ_I) 是否等于 ZZ_I
    assert unify(ZZ_I, ZZ_I) == ZZ_I
    # 验证 unify(ZZ_I, QQ) 是否等于 QQ_I
    assert unify(ZZ_I, QQ) == QQ_I
    # 验证 unify(ZZ_I, ALG) 是否等于 QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(ZZ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    # 验证 unify(ZZ_I, RR) 是否等于 CC
    assert unify(ZZ_I, RR) == CC
    # 验证 unify(ZZ_I, CC) 是否等于 CC
    assert unify(ZZ_I, CC) == CC
    # 验证 unify(ZZ_I, ZZ[x]) 是否等于 ZZ_I[x]
    assert unify(ZZ_I, ZZ[x]) == ZZ_I[x]
    # 验证 unify(ZZ_I, ZZ_I[x]) 是否等于 ZZ_I[x]
    assert unify(ZZ_I, ZZ_I[x]) == ZZ_I[x]
    # 验证 unify(ZZ_I, ZZ.frac_field(x)) 是否等于 ZZ_I.frac_field(x)
    assert unify(ZZ_I, ZZ.frac_field(x)) == ZZ_I.frac_field(x)
    # 验证 unify(ZZ_I, ZZ_I.frac_field(x)) 是否等于 ZZ_I.frac_field(x)
    assert unify(ZZ_I, ZZ_I.frac_field(x)) == ZZ_I.frac_field(x)
    # 验证 unify(ZZ_I, EX) 是否等于 EX
    assert unify(ZZ_I, EX) == EX

    # 使用 lambda 函数调用 unify(QQ_I, F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(QQ_I, F3))
    # 验证 unify(QQ_I, ZZ) 是否等于 QQ_I
    assert unify(QQ_I, ZZ) == QQ_I
    # 验证 unify(QQ_I, ZZ_I) 是否等于 QQ_I
    assert unify(QQ_I, ZZ_I) == QQ_I
    # 验证 unify(QQ_I, QQ) 是否等于 QQ_I
    assert unify(QQ_I, QQ) == QQ_I
    # 验证 unify(QQ_I, ALG) 是否等于 QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(QQ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    # 验证 unify(QQ_I, RR) 是否等于 CC
    assert unify(QQ_I, RR) == CC
    # 验证 unify(QQ_I, CC) 是否等于 CC
    assert unify(QQ_I, CC) == CC
    # 验证 unify(QQ_I, ZZ[x]) 是否等于 QQ_I[x]
    assert unify(QQ_I, ZZ[x]) == QQ_I[x]
    # 验证 unify(QQ_I, ZZ_I[x]) 是否等于 QQ_I[x]
    assert unify(QQ_I, ZZ_I[x]) == QQ_I[x]
    # 验证 unify(QQ_I, QQ[x]) 是否等于 QQ_I[x]
    assert unify(QQ_I, QQ[x]) == QQ_I[x]
    # 验证 unify(QQ_I, QQ_I[x]) 是否等于 QQ_I[x]
    assert unify(QQ_I, QQ_I[x]) == QQ_I[x]
    # 验证 unify(QQ_I, ZZ.frac_field(x)) 是否等于 QQ_I.frac_field(x)
    assert unify(QQ_I, ZZ.frac_field(x)) == QQ_I.frac_field(x)
    # 验证 unify(QQ_I, ZZ_I.frac_field(x)) 是否等于 QQ_I.frac_field(x)
    assert unify(QQ_I, ZZ_I.frac_field(x)) == QQ_I.frac_field(x)
    # 验证 unify(QQ_I, QQ.frac_field(x)) 是否等于 QQ_I.frac_field(x)
    assert unify(QQ_I, QQ.frac_field(x)) == QQ_I.frac_field(x)
    # 验证 unify(QQ_I, QQ_I.frac_field(x)) 是否等于 QQ_I.frac_field(x)
    assert unify(QQ_I, QQ_I.frac_field(x)) == QQ_I.frac_field(x)
    # 验证 unify(QQ_I, EX) 是否等于 EX
    assert unify(QQ_I, EX) == EX

    # 使用 lambda 函数调用 unify(RR, F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(RR, F3))
    # 验证 unify(RR, ZZ) 是否等于 RR
    assert unify(RR, ZZ) == RR
    # 验证 unify(RR, QQ) 是否等于 RR
    assert unify(RR, QQ) == RR
    # 验证 unify(RR, ALG) 是否等于 RR
    assert unify(RR, ALG) == RR
    # 验证 unify(RR, RR) 是否等于 RR
    assert unify(RR, RR) == RR
    # 验证 unify(RR, CC) 是否等于 CC
    assert unify(RR, CC) == CC
    # 验证 unify(RR, ZZ[x]) 是否等于 RR[x]
    assert unify(RR, ZZ[x]) == RR[x]
    # 验证 unify(RR, ZZ.frac_field(x)) 是否等于 RR.frac_field(x)
    assert unify(RR, ZZ.frac_field(x)) == RR.frac_field(x)
    # 验证 unify(RR, EX) 是否等于 EX
    assert unify(RR, EX) == EX
    # 验证 RR[x].unify(ZZ.frac_field(y)) 是否等于 RR.frac_field(x, y)
    assert RR[x].unify(ZZ.frac_field(y)) == RR.frac_field(x, y)

    # 使用 lambda 函数调用 unify(CC, F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(CC, F3))
    # 验证 unify(CC, ZZ) 是否等于 CC
    assert unify(CC, ZZ) == CC
    # 验证 unify(CC, QQ) 是否等于 CC
    assert unify(CC, QQ) == CC
    # 验证 unify(CC, ALG) 是否等于 CC
    assert unify(CC, ALG) == CC
    # 验证 unify(CC, RR) 是否等于 CC
    assert unify(CC, RR) == CC
    # 验证 unify(CC, CC) 是否等于 CC
    assert unify(CC, CC) == CC
    # 验证 unify(CC, ZZ[x]) 是否等于 CC[x]
    assert unify(CC, ZZ[x]) == CC[x]
    # 验证 unify(CC, ZZ.frac_field(x)) 是否等于 CC.frac_field(x)
    assert unify(CC, ZZ.frac_field(x)) == CC.frac_field(x)
    # 验证 unify(CC, EX) 是否等于 EX
    assert unify(CC, EX) == EX

    # 使用 lambda 函数调用 unify(ZZ[x], F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(ZZ[x], F3))
    # 验证 unify(ZZ[x], ZZ) 是否等于 ZZ[x]
    assert unify(ZZ[x], ZZ) == ZZ[x]
    # 验证 unify(ZZ[x], QQ) 是否等于 QQ[x]
    assert unify(ZZ[x], QQ) == QQ[x]
    # 验证 unify(ZZ[x], ALG) 是否等于 ALG[x]
    assert unify(ZZ[x], ALG) == ALG[x]
    # 验证 unify(ZZ[x], RR) 是否等于 RR[x]
    assert unify(ZZ[x], RR) == RR[x]
    # 验证 unify(ZZ[x], CC) 是否等于 CC[x]
    assert unify(ZZ[x], CC) == CC[x]
    # 验证 unify(ZZ[x], ZZ[x]) 是否等于 ZZ[x]
    assert unify(ZZ[x], ZZ[x]) == ZZ[x]
    # 验证 unify(ZZ[x], ZZ.frac_field(x)) 是否等于 ZZ.frac_field(x)
    assert unify(ZZ[x], ZZ.frac_field(x)) == ZZ.frac_field(x)
    # 验证 unify(ZZ[x], EX) 是否等于 EX
    assert unify(ZZ[x], EX) == EX

    # 使用 lambda 函数调用 unify(ZZ.frac_field(x), F3)，验证是否引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(ZZ.frac_field(x), F3))
    # 验证 unify(ZZ.frac_field(x), ZZ) 是否等于 ZZ.frac_field(x)
    assert unify(Z
    # 调用 lambda 函数测试 unify(EX, F3)，验证是否会抛出 UnificationFailed 异常
    raises(UnificationFailed, lambda: unify(EX, F3))
    # 检查 unify(EX, ZZ) 是否等于 EX
    assert unify(EX, ZZ) == EX
    # 检查 unify(EX, QQ) 是否等于 EX
    assert unify(EX, QQ) == EX
    # 检查 unify(EX, ALG) 是否等于 EX
    assert unify(EX, ALG) == EX
    # 检查 unify(EX, RR) 是否等于 EX
    assert unify(EX, RR) == EX
    # 检查 unify(EX, CC) 是否等于 EX
    assert unify(EX, CC) == EX
    # 检查 unify(EX, ZZ[x]) 是否等于 EX
    assert unify(EX, ZZ[x]) == EX
    # 检查 unify(EX, ZZ.frac_field(x)) 是否等于 EX
    assert unify(EX, ZZ.frac_field(x)) == EX
    # 检查 unify(EX, EX) 是否等于 EX
    assert unify(EX, EX) == EX
# 定义测试函数 test_Domain_unify_composite，用于测试 unify 函数的各种情况

assert unify(ZZ.poly_ring(x), ZZ) == ZZ.poly_ring(x)
# 测试 unify 函数：将整数环上的多项式环与整数环进行统一，预期结果为整数环上的相同多项式环

assert unify(ZZ.poly_ring(x), QQ) == QQ.poly_ring(x)
# 测试 unify 函数：将整数环上的多项式环与有理数环进行统一，预期结果为有理数环上的相同多项式环

assert unify(QQ.poly_ring(x), ZZ) == QQ.poly_ring(x)
# 测试 unify 函数：将有理数环上的多项式环与整数环进行统一，预期结果为有理数环上的相同多项式环

assert unify(QQ.poly_ring(x), QQ) == QQ.poly_ring(x)
# 测试 unify 函数：将有理数环上的多项式环与有理数环进行统一，预期结果为有理数环上的相同多项式环

# 各种情况的类似测试，包括整数环和有理数环的统一，多项式环和分式域的统一等等，均按照相同的模式进行测试和注释
# 以下的每一行代码都是在测试 unify 函数，比较不同数学结构之间的统一结果

assert unify(ZZ, ZZ.poly_ring(x)) == ZZ.poly_ring(x)
assert unify(QQ, ZZ.poly_ring(x)) == QQ.poly_ring(x)
assert unify(ZZ, QQ.poly_ring(x)) == QQ.poly_ring(x)
assert unify(QQ, QQ.poly_ring(x)) == QQ.poly_ring(x)

assert unify(ZZ.poly_ring(x, y), ZZ) == ZZ.poly_ring(x, y)
assert unify(ZZ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x, y), ZZ) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)

assert unify(ZZ, ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
assert unify(QQ, ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
assert unify(ZZ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
assert unify(QQ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

assert unify(ZZ.frac_field(x), ZZ) == ZZ.frac_field(x)
assert unify(ZZ.frac_field(x), QQ) == QQ.frac_field(x)
assert unify(QQ.frac_field(x), ZZ) == QQ.frac_field(x)
assert unify(QQ.frac_field(x), QQ) == QQ.frac_field(x)

assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
assert unify(ZZ, QQ.frac_field(x)) == QQ.frac_field(x)
assert unify(QQ, QQ.frac_field(x)) == QQ.frac_field(x)

assert unify(ZZ.frac_field(x, y), ZZ) == ZZ.frac_field(x, y)
assert unify(ZZ.frac_field(x, y), QQ) == QQ.frac_field(x, y)
assert unify(QQ.frac_field(x, y), ZZ) == QQ.frac_field(x, y)
assert unify(QQ.frac_field(x, y), QQ) == QQ.frac_field(x, y)

assert unify(ZZ, ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
assert unify(QQ, ZZ.frac_field(x, y)) == QQ.frac_field(x, y)
assert unify(ZZ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)
assert unify(QQ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)

assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x)) == ZZ.poly_ring(x)
assert unify(ZZ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)
assert unify(QQ.poly_ring(x), ZZ.poly_ring(x)) == QQ.poly_ring(x)
assert unify(QQ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)

assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x)) == ZZ.poly_ring(x, y)
assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x)) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)

assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
assert unify(ZZ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x), ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
assert unify(QQ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x, z)) == ZZ.poly_ring(x, y, z)
    # 断言：将多项式环 ZZ.poly_ring(x, y) 和 QQ.poly_ring(x, z) 统一化，预期结果为 QQ.poly_ring(x, y, z)
    assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)

    # 断言：将多项式环 QQ.poly_ring(x, y) 和 ZZ.poly_ring(x, z) 统一化，预期结果为 QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)

    # 断言：将多项式环 QQ.poly_ring(x, y) 和 QQ.poly_ring(x, z) 统一化，预期结果为 QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)

    # 断言：将分式域 ZZ.frac_field(x) 与自身统一化，预期结果为 ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), ZZ.frac_field(x)) == ZZ.frac_field(x)

    # 断言：将分式域 ZZ.frac_field(x) 与 QQ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)

    # 断言：将分式域 QQ.frac_field(x) 与 ZZ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x)) == QQ.frac_field(x)

    # 断言：将分式域 QQ.frac_field(x) 与自身统一化，预期结果为 QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)

    # 断言：将分式域 ZZ.frac_field(x, y) 与 ZZ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)

    # 断言：将分式域 ZZ.frac_field(x, y) 与 QQ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    # 断言：将分式域 QQ.frac_field(x, y) 与 ZZ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x)) == QQ.frac_field(x, y)

    # 断言：将分式域 QQ.frac_field(x, y) 与自身统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    # 断言：将分式域 ZZ.frac_field(x) 与 ZZ.frac_field(x, y) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)

    # 断言：将分式域 ZZ.frac_field(x) 与 QQ.frac_field(x, y) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    # 断言：将分式域 QQ.frac_field(x) 与 ZZ.frac_field(x, y) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x, y)) == QQ.frac_field(x, y)

    # 断言：将分式域 QQ.frac_field(x) 与自身 QQ.frac_field(x, y) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    # 断言：将分式域 ZZ.frac_field(x, y) 与 ZZ.frac_field(x, z) 统一化，预期结果为 ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)

    # 断言：将分式域 ZZ.frac_field(x, y) 与 QQ.frac_field(x, z) 统一化，预期结果为 QQ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    # 断言：将分式域 QQ.frac_field(x, y) 与 ZZ.frac_field(x, z) 统一化，预期结果为 QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    # 断言：将分式域 QQ.frac_field(x, y) 与 QQ.frac_field(x, z) 统一化，预期结果为 QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    # 断言：将多项式环 ZZ.poly_ring(x) 与 ZZ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x)
    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)

    # 断言：将多项式环 ZZ.poly_ring(x) 与 QQ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x)) == ZZ.frac_field(x)

    # 断言：将多项式环 QQ.poly_ring(x) 与 ZZ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x)
    assert unify(QQ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)

    # 断言：将多项式环 QQ.poly_ring(x) 与 QQ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x)
    assert unify(QQ.poly_ring(x), QQ.frac_field(x)) == QQ.frac_field(x)

    # 断言：将多项式环 ZZ.poly_ring(x, y) 与 ZZ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)

    # 断言：将多项式环 ZZ.poly_ring(x, y) 与 QQ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ.frac_field(x)) == ZZ.frac_field(x, y)

    # 断言：将多项式环 QQ.poly_ring(x, y) 与 ZZ.frac_field(x) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)

    # 断言：将多项式环 QQ.poly_ring(x, y) 与 QQ.frac_field(x) 统一化，预期结果为 QQ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    # 断言：将多项式环 ZZ.poly_ring(x) 与 ZZ.frac_field(x, y) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)

    # 断言：将多项式环 ZZ.poly_ring(x) 与 QQ.frac_field(x, y) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x, y)) == ZZ.frac_field(x, y)

    # 断言：将多项式环 QQ.poly_ring(x) 与 ZZ.frac_field(x, y) 统一化，预期结果为 ZZ.frac_field(x, y)
    assert unify(Q
    # 断言：将 QQ.frac_field(x) 和 QQ.poly_ring(x) 统一后应该还是 QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x)) == QQ.frac_field(x)

    # 断言：将 ZZ.frac_field(x, y) 和 ZZ.poly_ring(x) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    # 断言：将 ZZ.frac_field(x, y) 和 QQ.poly_ring(x) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x)) == ZZ.frac_field(x, y)
    # 断言：将 QQ.frac_field(x, y) 和 ZZ.poly_ring(x) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    # 断言：将 QQ.frac_field(x, y) 和 QQ.poly_ring(x) 统一后应该还是 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x)) == QQ.frac_field(x, y)

    # 断言：将 ZZ.frac_field(x) 和 ZZ.poly_ring(x, y) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    # 断言：将 ZZ.frac_field(x) 和 QQ.poly_ring(x, y) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    # 断言：将 QQ.frac_field(x) 和 ZZ.poly_ring(x, y) 统一后应该得到 ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    # 断言：将 QQ.frac_field(x) 和 QQ.poly_ring(x, y) 统一后应该还是 QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x, y)) == QQ.frac_field(x, y)

    # 断言：将 ZZ.frac_field(x, y) 和 ZZ.poly_ring(x, z) 统一后应该得到 ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    # 断言：将 ZZ.frac_field(x, y) 和 QQ.poly_ring(x, z) 统一后应该得到 ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    # 断言：将 QQ.frac_field(x, y) 和 ZZ.poly_ring(x, z) 统一后应该得到 ZZ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    # 断言：将 QQ.frac_field(x, y) 和 QQ.poly_ring(x, z) 统一后应该还是 QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x, z)) == QQ.frac_field(x, y, z)
# 定义测试函数，用于测试 Domain 类的 unify_algebraic 方法
def test_Domain_unify_algebraic():
    # 创建包含 sqrt(5) 的有理数域对象
    sqrt5 = QQ.algebraic_field(sqrt(5))
    # 创建包含 sqrt(7) 的有理数域对象
    sqrt7 = QQ.algebraic_field(sqrt(7))
    # 创建包含 sqrt(5) 和 sqrt(7) 的有理数域对象
    sqrt57 = QQ.algebraic_field(sqrt(5), sqrt(7))

    # 断言 unify 方法正确合并 sqrt5 和 sqrt7 为 sqrt57
    assert sqrt5.unify(sqrt7) == sqrt57

    # 断言 unify 方法正确处理相同域的情况
    assert sqrt5.unify(sqrt5[x, y]) == sqrt5[x, y]
    assert sqrt5[x, y].unify(sqrt5) == sqrt5[x, y]

    # 断言 unify 方法正确处理有理函数域的情况
    assert sqrt5.unify(sqrt5.frac_field(x, y)) == sqrt5.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt5) == sqrt5.frac_field(x, y)

    # 断言 unify 方法正确处理包含不同变量的有理数域的情况
    assert sqrt5.unify(sqrt7[x, y]) == sqrt57[x, y]
    assert sqrt5[x, y].unify(sqrt7) == sqrt57[x, y]

    # 断言 unify 方法正确处理包含不同变量的有理函数域的情况
    assert sqrt5.unify(sqrt7.frac_field(x, y)) == sqrt57.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt7) == sqrt57.frac_field(x, y)

# 定义测试函数，用于测试 Domain 类的 unify_FiniteExtension 方法
def test_Domain_unify_FiniteExtension():
    # 创建有限扩展域对象，定义多个不同的对象
    KxZZ = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    KxQQ = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    KxZZy = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    KxQQy = FiniteExtension(Poly(x**2 - 2, x, domain=QQ[y]))

    # 断言 unify 方法正确处理相同对象的情况
    assert KxZZ.unify(KxZZ) == KxZZ
    assert KxQQ.unify(KxQQ) == KxQQ
    assert KxZZy.unify(KxZZy) == KxZZy
    assert KxQQy.unify(KxQQy) == KxQQy

    # 断言 unify 方法正确处理与整数和有理数域的合并情况
    assert KxZZ.unify(ZZ) == KxZZ
    assert KxZZ.unify(QQ) == KxQQ
    assert KxQQ.unify(ZZ) == KxQQ
    assert KxQQ.unify(QQ) == KxQQ

    # 断言 unify 方法正确处理与整数域和有理数域的扩展合并情况
    assert KxZZ.unify(ZZ[y]) == KxZZy
    assert KxZZ.unify(QQ[y]) == KxQQy
    assert KxQQ.unify(ZZ[y]) == KxQQy
    assert KxQQ.unify(QQ[y]) == KxQQy

    # 断言 unify 方法正确处理与多变量整数和有理数域的情况
    assert KxZZy.unify(ZZ) == KxZZy
    assert KxZZy.unify(QQ) == KxQQy
    assert KxQQy.unify(ZZ) == KxQQy
    assert KxQQy.unify(QQ) == KxQQy

    # 断言 unify 方法正确处理与多变量整数域的扩展情况
    assert KxZZy.unify(ZZ[y]) == KxZZy
    assert KxZZy.unify(QQ[y]) == KxQQy
    assert KxQQy.unify(ZZ[y]) == KxQQy
    assert KxQQy.unify(QQ[y]) == KxQQy

    # 创建新的有限扩展域对象
    K = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    # 断言 unify 方法正确处理与整数域的情况
    assert K.unify(ZZ) == K
    assert K.unify(ZZ[x]) == K
    assert K.unify(ZZ[y]) == K
    assert K.unify(ZZ[x, y]) == K

    # 创建新的有限扩展域对象，定义多个不同的对象
    Kz = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y, z]))
    # 断言 unify 方法正确处理与整数域和多变量整数域的扩展情况
    assert K.unify(ZZ[z]) == Kz
    assert K.unify(ZZ[x, z]) == Kz
    assert K.unify(ZZ[y, z]) == Kz
    assert K.unify(ZZ[x, y, z]) == Kz

    # 创建新的有限扩展域对象，定义多个不同的对象
    Kx = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    Ky = FiniteExtension(Poly(y**2 - 2, y, domain=ZZ))
    Kxy = FiniteExtension(Poly(y**2 - 2, y, domain=Kx))
    # 断言 unify 方法正确处理与整数域和多变量整数域的合并情况
    assert Kx.unify(Kx) == Kx
    assert Ky.unify(Ky) == Ky
    assert Kx.unify(Ky) == Kxy
    assert Ky.unify(Kx) == Kxy

# 定义测试函数，用于测试 Domain 类的 __contains__ 方法
def test_Domain__contains__():
    # 断言包含关系操作返回 True 的情况
    assert (0 in EX) is True
    assert (0 in ZZ) is True
    assert (0 in QQ) is True
    assert (0 in RR) is True
    assert (0 in CC) is True
    assert (0 in ALG) is True
    assert (0 in ZZ[x, y]) is True
    assert (0 in QQ[x, y]) is True
    assert (0 in RR[x, y]) is True

    # 断言包含关系操作返回 True 的情况
    assert (-7 in EX) is True
    assert (-7 in ZZ) is True
    # 检查整数 -7 是否存在于 QQ 集合中，并断言结果为 True
    assert (-7 in QQ) is True
    # 检查整数 -7 是否存在于 RR 集合中，并断言结果为 True
    assert (-7 in RR) is True
    # 检查整数 -7 是否存在于 CC 集合中，并断言结果为 True
    assert (-7 in CC) is True
    # 检查整数 -7 是否存在于 ALG 集合中，并断言结果为 True
    assert (-7 in ALG) is True
    # 检查整数 -7 是否存在于 ZZ[x, y] 集合中，并断言结果为 True
    assert (-7 in ZZ[x, y]) is True
    # 检查整数 -7 是否存在于 QQ[x, y] 集合中，并断言结果为 True
    assert (-7 in QQ[x, y]) is True
    # 检查整数 -7 是否存在于 RR[x, y] 集合中，并断言结果为 True
    assert (-7 in RR[x, y]) is True
    
    # 检查整数 17 是否存在于 EX 集合中，并断言结果为 True
    assert (17 in EX) is True
    # 检查整数 17 是否存在于 ZZ 集合中，并断言结果为 True
    assert (17 in ZZ) is True
    # 检查整数 17 是否存在于 QQ 集合中，并断言结果为 True
    assert (17 in QQ) is True
    # 检查整数 17 是否存在于 RR 集合中，并断言结果为 True
    assert (17 in RR) is True
    # 检查整数 17 是否存在于 CC 集合中，并断言结果为 True
    assert (17 in CC) is True
    # 检查整数 17 是否存在于 ALG 集合中，并断言结果为 True
    assert (17 in ALG) is True
    # 检查整数 17 是否存在于 ZZ[x, y] 集合中，并断言结果为 True
    assert (17 in ZZ[x, y]) is True
    # 检查整数 17 是否存在于 QQ[x, y] 集合中，并断言结果为 True
    assert (17 in QQ[x, y]) is True
    # 检查整数 17 是否存在于 RR[x, y] 集合中，并断言结果为 True
    assert (17 in RR[x, y]) is True
    
    # 检查有理数 -1/7 是否存在于 EX 集合中，并断言结果为 True
    assert (Rational(-1, 7) in EX) is True
    # 检查有理数 -1/7 是否存在于 ZZ 集合中，并断言结果为 False
    assert (Rational(-1, 7) in ZZ) is False
    # 检查有理数 -1/7 是否存在于 QQ 集合中，并断言结果为 True
    assert (Rational(-1, 7) in QQ) is True
    # 检查有理数 -1/7 是否存在于 RR 集合中，并断言结果为 True
    assert (Rational(-1, 7) in RR) is True
    # 检查有理数 -1/7 是否存在于 CC 集合中，并断言结果为 True
    assert (Rational(-1, 7) in CC) is True
    # 检查有理数 -1/7 是否存在于 ALG 集合中，并断言结果为 True
    assert (Rational(-1, 7) in ALG) is True
    # 检查有理数 -1/7 是否存在于 ZZ[x, y] 集合中，并断言结果为 False
    assert (Rational(-1, 7) in ZZ[x, y]) is False
    # 检查有理数 -1/7 是否存在于 QQ[x, y] 集合中，并断言结果为 True
    assert (Rational(-1, 7) in QQ[x, y]) is True
    # 检查有理数 -1/7 是否存在于 RR[x, y] 集合中，并断言结果为 True
    assert (Rational(-1, 7) in RR[x, y]) is True
    
    # 检查有理数 3/5 是否存在于 EX 集合中，并断言结果为 True
    assert (Rational(3, 5) in EX) is True
    # 检查有理数 3/5 是否存在于 ZZ 集合中，并断言结果为 False
    assert (Rational(3, 5) in ZZ) is False
    # 检查有理数 3/5 是否存在于 QQ 集合中，并断言结果为 True
    assert (Rational(3, 5) in QQ) is True
    # 检查有理数 3/5 是否存在于 RR 集合中，并断言结果为 True
    assert (Rational(3, 5) in RR) is True
    # 检查有理数 3/5 是否存在于 CC 集合中，并断言结果为 True
    assert (Rational(3, 5) in CC) is True
    # 检查有理数 3/5 是否存在于 ALG 集合中，并断言结果为 True
    assert (Rational(3, 5) in ALG) is True
    # 检查有理数 3/5 是否存在于 ZZ[x, y] 集合中，并断言结果为 False
    assert (Rational(3, 5) in ZZ[x, y]) is False
    # 检查有理数 3/5 是否存在于 QQ[x, y] 集合中，并断言结果为 True
    assert (Rational(3, 5) in QQ[x, y]) is True
    # 检查有理数 3/5 是否存在于 RR[x, y] 集合中，并断言结果为 True
    assert (Rational(3, 5) in RR[x, y]) is True
    
    # 检查浮点数 3.0 是否存在于 EX 集合中，并断言结果为 True
    assert (3.0 in EX) is True
    # 检查浮点数 3.0 是否存在于 ZZ 集合中，并断言结果为 True
    assert (3.0 in ZZ) is True
    # 检查浮点数 3.0 是否存在于 QQ 集合中，并断言结果为 True
    assert (3.0 in QQ) is True
    # 检查浮点数 3.0 是否存在于 RR 集合中，并断言结果为 True
    assert (3.0 in RR) is True
    # 检查浮点数 3.0 是否存在于 CC 集合中，并断言结果为 True
    assert (3.0 in CC) is True
    # 检查浮点数 3.0 是否存在于 ALG 集合中，并断言结果为 True
    assert (3.0 in ALG) is True
    # 检查浮点数 3.0 是否存在于 ZZ[x, y] 集合中，并断言结果为 True
    assert (3.0 in ZZ[x, y]) is True
    # 检查浮点数 3.0 是否存在于 QQ[x, y] 集合中，并断言结果为 True
    assert (3.0 in QQ[x, y]) is True
    # 检查浮点数 3.0 是否存在于 RR[x, y] 集合中，并断言结果为 True
    assert (3.0 in RR[x, y]) is True
    
    # 检查浮点数 3.14 是否存在于 EX 集合中，并断言结果为 True
    assert (3.14 in EX) is True
    # 检查浮点数 3.14 是否存在于 ZZ 集合中，并断言结果为 False
    assert (3.14 in ZZ) is False
    # 检查浮点数 3.14 是否存在于 QQ 集合中，并断言结果为 True
    assert (3.14 in QQ) is True
    # 检查浮点数 3.14 是否存在于 RR 集合中，并断言结果为 True
    assert (3.14 in RR) is True
    # 检查浮点数 3.14 是否存在于 CC 集合中，并断言结果为 True
    assert (3.14 in CC) is True
    # 检查浮点数 3.14 是否存在于 ALG 集合中，并断言结果为 True
    assert (3
    # 断言：检查 sin(1) 是否在整数环 ZZ 中，应该为 False
    assert (sin(1) in ZZ) is False
    # 断言：检查 sin(1) 是否在有理数环 QQ 中，应该为 False
    assert (sin(1) in QQ) is False
    # 断言：检查 sin(1) 是否在实数域 RR 中，应该为 True
    assert (sin(1) in RR) is True
    # 断言：检查 sin(1) 是否在复数域 CC 中，应该为 True
    assert (sin(1) in CC) is True
    # 断言：检查 sin(1) 是否在代数域 ALG 中，应该为 False
    assert (sin(1) in ALG) is False
    # 断言：检查 sin(1) 是否在多项式环 ZZ[x, y] 中，应该为 False
    assert (sin(1) in ZZ[x, y]) is False
    
    # 断言：检查 x**2 + 1 是否在表达式 EX 中，应该为 True
    assert (x**2 + 1 in EX) is True
    # 断言：检查 x**2 + 1 是否在整数环 ZZ 中，应该为 False
    assert (x**2 + 1 in ZZ) is False
    # 断言：检查 x**2 + 1 是否在有理数域 QQ 中，应该为 False
    assert (x**2 + 1 in QQ) is False
    # 断言：检查 x**2 + 1 是否在实数域 RR 中，应该为 False
    assert (x**2 + 1 in RR) is False
    # 断言：检查 x**2 + 1 是否在复数域 CC 中，应该为 False
    assert (x**2 + 1 in CC) is False
    # 断言：检查 x**2 + 1 是否在代数域 ALG 中，应该为 False
    assert (x**2 + 1 in ALG) is False
    # 断言：检查 x**2 + 1 是否在整数多项式环 ZZ[x] 中，应该为 True
    assert (x**2 + 1 in ZZ[x]) is True
    # 断言：检查 x**2 + 1 是否在有理数多项式环 QQ[x] 中，应该为 True
    assert (x**2 + 1 in QQ[x]) is True
    # 断言：检查 x**2 + 1 是否在实数多项式环 RR[x] 中，应该为 True
    assert (x**2 + 1 in RR[x]) is True
    # 断言：检查 x**2 + 1 是否在整数多项式环 ZZ[x, y] 中，应该为 True
    assert (x**2 + 1 in ZZ[x, y]) is True
    # 断言：检查 x**2 + 1 是否在有理数多项式环 QQ[x, y] 中，应该为 True
    assert (x**2 + 1 in QQ[x, y]) is True
    # 断言：检查 x**2 + 1 是否在实数多项式环 RR[x, y] 中，应该为 True
    assert (x**2 + 1 in RR[x, y]) is True
    
    # 断言：检查 x**2 + y**2 是否在表达式 EX 中，应该为 True
    assert (x**2 + y**2 in EX) is True
    # 断言：检查 x**2 + y**2 是否在整数环 ZZ 中，应该为 False
    assert (x**2 + y**2 in ZZ) is False
    # 断言：检查 x**2 + y**2 是否在有理数域 QQ 中，应该为 False
    assert (x**2 + y**2 in QQ) is False
    # 断言：检查 x**2 + y**2 是否在实数域 RR 中，应该为 False
    assert (x**2 + y**2 in RR) is False
    # 断言：检查 x**2 + y**2 是否在复数域 CC 中，应该为 False
    assert (x**2 + y**2 in CC) is False
    # 断言：检查 x**2 + y**2 是否在代数域 ALG 中，应该为 False
    assert (x**2 + y**2 in ALG) is False
    # 断言：检查 x**2 + y**2 是否在整数多项式环 ZZ[x] 中，应该为 False
    assert (x**2 + y**2 in ZZ[x]) is False
    # 断言：检查 x**2 + y**2 是否在有理数多项式环 QQ[x] 中，应该为 False
    assert (x**2 + y**2 in QQ[x]) is False
    # 断言：检查 x**2 + y**2 是否在实数多项式环 RR[x] 中，应该为 False
    assert (x**2 + y**2 in RR[x]) is False
    # 断言：检查 x**2 + y**2 是否在整数多项式环 ZZ[x, y] 中，应该为 True
    assert (x**2 + y**2 in ZZ[x, y]) is True
    # 断言：检查 x**2 + y**2 是否在有理数多项式环 QQ[x, y] 中，应该为 True
    assert (x**2 + y**2 in QQ[x, y]) is True
    # 断言：检查 x**2 + y**2 是否在实数多项式环 RR[x, y] 中，应该为 True
    assert (x**2 + y**2 in RR[x, y]) is True
    
    # 断言：检查有理数表达式 Rational(3, 2)*x/(y + 1) - z 是否在有理数多项式环 QQ[x, y, z] 中，应该为 False
    assert (Rational(3, 2)*x/(y + 1) - z in QQ[x, y, z]) is False
def test_issue_14433():
    # 检查有理数域中的运算：2/3 * x 是否在 1/x 的有理数域中
    assert (Rational(2, 3)*x in QQ.frac_field(1/x)) is True
    # 检查有理数域中的运算：1/x 是否在 x 的有理数域中
    assert (1/x in QQ.frac_field(x)) is True
    # 检查有理数域中的运算：x^2 + y^2 是否在 1/x, 1/y 的有理数域中
    assert ((x**2 + y**2) in QQ.frac_field(1/x, 1/y)) is True
    # 检查有理数域中的运算：x + y 是否在 1/x, y 的有理数域中
    assert ((x + y) in QQ.frac_field(1/x, y)) is True
    # 检查有理数域中的运算：x - y 是否在 x, 1/y 的有理数域中
    assert ((x - y) in QQ.frac_field(x, 1/y)) is True


def test_Domain_get_ring():
    # 检查整数环是否具有关联环
    assert ZZ.has_assoc_Ring is True
    # 检查有理数环是否具有关联环
    assert QQ.has_assoc_Ring is True
    # 检查多项式环 ZZ[x] 是否具有关联环
    assert ZZ[x].has_assoc_Ring is True
    # 检查分式域 QQ[x] 是否具有关联环
    assert QQ[x].has_assoc_Ring is True
    # 检查多变量多项式环 ZZ[x, y] 是否具有关联环
    assert ZZ[x, y].has_assoc_Ring is True
    # 检查多变量分式域 QQ[x, y] 是否具有关联环
    assert QQ[x, y].has_assoc_Ring is True
    # 检查带有单个变量的有理数分式域 ZZ.frac_field(x) 是否具有关联环
    assert ZZ.frac_field(x).has_assoc_Ring is True
    # 检查带有单个变量的有理数分式域 QQ.frac_field(x) 是否具有关联环
    assert QQ.frac_field(x).has_assoc_Ring is True
    # 检查带有两个变量的有理数分式域 ZZ.frac_field(x, y) 是否具有关联环
    assert ZZ.frac_field(x, y).has_assoc_Ring is True
    # 检查带有两个变量的有理数分式域 QQ.frac_field(x, y) 是否具有关联环
    assert QQ.frac_field(x, y).has_assoc_Ring is True

    # 检查表达式域 EX 是否具有关联环
    assert EX.has_assoc_Ring is False
    # 检查实数域 RR 是否具有关联环
    assert RR.has_assoc_Ring is False
    # 检查代数域 ALG 是否具有关联环
    assert ALG.has_assoc_Ring is False

    # 检查整数环的环应该是自身
    assert ZZ.get_ring() == ZZ
    # 检查有理数环的环应该是整数环
    assert QQ.get_ring() == ZZ
    # 检查多项式环 ZZ[x] 的环应该是自身
    assert ZZ[x].get_ring() == ZZ[x]
    # 检查分式域 QQ[x] 的环应该是自身
    assert QQ[x].get_ring() == QQ[x]
    # 检查多变量多项式环 ZZ[x, y] 的环应该是自身
    assert ZZ[x, y].get_ring() == ZZ[x, y]
    # 检查多变量分式域 QQ[x, y] 的环应该是自身
    assert QQ[x, y].get_ring() == QQ[x, y]
    # 检查带有单个变量的有理数分式域 ZZ.frac_field(x) 的环应该是 ZZ[x]
    assert ZZ.frac_field(x).get_ring() == ZZ[x]
    # 检查带有单个变量的有理数分式域 QQ.frac_field(x) 的环应该是 QQ[x]
    assert QQ.frac_field(x).get_ring() == QQ[x]
    # 检查带有两个变量的有理数分式域 ZZ.frac_field(x, y) 的环应该是 ZZ[x, y]
    assert ZZ.frac_field(x, y).get_ring() == ZZ[x, y]
    # 检查带有两个变量的有理数分式域 QQ.frac_field(x, y) 的环应该是 QQ[x, y]

    assert QQ.frac_field(x, y).get_ring() == QQ[x, y]


def test_Domain_get_field():
    # 检查表达式域 EX 是否具有关联域
    assert EX.has_assoc_Field is True
    # 检查整数环是否具有关联域
    assert ZZ.has_assoc_Field is True
    # 检查有理数环是否具有关联域
    assert QQ.has_assoc_Field is True
    # 检查实数域是否具有关联域
    assert RR.has_assoc_Field is True
    # 检查代数域是否具有关联域
    assert ALG.has_assoc_Field is True
    # 检查多项式环 ZZ[x] 是否具有关联域
    assert ZZ[x].has_assoc_Field is True
    # 检查分式域 QQ[x] 是否具有关联域
    assert QQ[x].has_assoc_Field is True
    # 检查多变量多项式环 ZZ[x, y] 是否具有关联域
    assert ZZ[x, y].has_assoc_Field is True
    # 检查多变量分式域 QQ[x, y] 是否具有关联域
    assert QQ[x, y].has_assoc_Field is True

    # 检查表达式域 EX 的域应该是自身
    assert EX.get_field() == EX
    # 检查整数环的域应该是有理数环 QQ
    assert ZZ.get_field() == QQ
    # 检查有理数环的域应该是有理数环 QQ
    assert QQ.get_field() == QQ
    # 检查实数域的域应该是实数域 RR
    assert RR.get_field() == RR
    # 检查代数域的域应该是代数域 ALG
    assert ALG.get_field() == ALG
    # 检查多项式环 ZZ[x] 的域应该是带有变量 x 的有理数分式域 QQ.frac_field(x)
    assert ZZ[x].get_field() == ZZ.frac_field(x)
    # 检查分式域 QQ[x] 的域应该是带有变量 x 的有理数分式域 QQ.frac_field(x)
    assert QQ[x].get_field() == QQ.frac_field(x)
    # 检查多变量多项式环 ZZ[x, y] 的域应该是带有变量 x, y 的有理数分式域 ZZ.frac_field(x, y)
    assert ZZ[x, y].get_field() == ZZ.frac_field(x, y)
    # 检查多变量分式域 QQ[x, y] 的域应该是带有变量 x, y 的有理数分式域 QQ.frac_field(x, y)


def test_Domain_set_domain():
    # 创建一个包含不同域的列表
    doms = [GF(5), ZZ, QQ, ALG, RR, CC, EX, ZZ[z], QQ[z], RR[z], CC[z], EX[z]]
    # 遍历所有可能的组合
    for D1 in doms:
        for D2 in doms:
            # 检查在 D2 的环中设置变量 x 后，D1[x] 的环应该是 D2[x]
            assert D1[x].set_domain(D2) == D2[x]
            # 检查在 D2 的环中设置变量 x, y 后，D1[x, y] 的环应该是 D2[x, y]
            assert D1[x, y].set_domain
    # 定义精确类型的对象列表，包括 GF(5), ZZ, QQ, ALG, EX
    exact = [GF(5), ZZ, QQ, ALG, EX]
    # 定义非精确类型的对象列表，包括 RR, CC
    inexact = [RR, CC]
    
    # 遍历精确类型和非精确类型的对象列表
    for D in exact + inexact:
        # 遍历每个对象的相关环境：D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x)
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            # 如果当前对象属于精确类型列表
            if D in exact:
                # 断言当前环境 R 是精确的
                assert R.is_Exact is True
            else:
                # 断言当前环境 R 是非精确的
                assert R.is_Exact is False
# 定义一个测试函数，用于测试域对象的精确性获取功能
def test_Domain_get_exact():
    # 断言各域对象的精确性获取结果是否与预期相同
    assert EX.get_exact() == EX
    assert ZZ.get_exact() == ZZ
    assert QQ.get_exact() == QQ
    assert RR.get_exact() == QQ
    assert CC.get_exact() == QQ_I
    assert ALG.get_exact() == ALG
    assert ZZ[x].get_exact() == ZZ[x]
    assert QQ[x].get_exact() == QQ[x]
    assert RR[x].get_exact() == QQ[x]
    assert CC[x].get_exact() == QQ_I[x]
    assert ZZ[x, y].get_exact() == ZZ[x, y]
    assert QQ[x, y].get_exact() == QQ[x, y]
    assert RR[x, y].get_exact() == QQ[x, y]
    assert CC[x, y].get_exact() == QQ_I[x, y]
    assert ZZ.frac_field(x).get_exact() == ZZ.frac_field(x)
    assert QQ.frac_field(x).get_exact() == QQ.frac_field(x)
    assert RR.frac_field(x).get_exact() == QQ.frac_field(x)
    assert CC.frac_field(x).get_exact() == QQ_I.frac_field(x)
    assert ZZ.frac_field(x, y).get_exact() == ZZ.frac_field(x, y)
    assert QQ.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert RR.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert CC.frac_field(x, y).get_exact() == QQ_I.frac_field(x, y)
    assert ZZ.old_poly_ring(x).get_exact() == ZZ.old_poly_ring(x)
    assert QQ.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert RR.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert CC.old_poly_ring(x).get_exact() == QQ_I.old_poly_ring(x)
    assert ZZ.old_poly_ring(x, y).get_exact() == ZZ.old_poly_ring(x, y)
    assert QQ.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert RR.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert CC.old_poly_ring(x, y).get_exact() == QQ_I.old_poly_ring(x, y)
    assert ZZ.old_frac_field(x).get_exact() == ZZ.old_frac_field(x)
    assert QQ.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert RR.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert CC.old_frac_field(x).get_exact() == QQ_I.old_frac_field(x)
    assert ZZ.old_frac_field(x, y).get_exact() == ZZ.old_frac_field(x, y)
    assert QQ.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert RR.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert CC.old_frac_field(x, y).get_exact() == QQ_I.old_frac_field(x, y)


# 定义一个测试函数，用于测试域对象的特征函数
def test_Domain_characteristic():
    # 对于每个特定的有限域和其特征值，执行如下断言
    for F, c in [(FF(3), 3), (FF(5), 5), (FF(7), 7)]:
        # 遍历 F, F[x], F.frac_field(x), F.old_poly_ring(x), F.old_frac_field(x) 这些环
        for R in F, F[x], F.frac_field(x), F.old_poly_ring(x), F.old_frac_field(x):
            # 断言环 R 的零特征是否为假，特征数是否为 c
            assert R.has_CharacteristicZero is False
            assert R.characteristic() == c
    # 对于整数环 ZZ, 有理数域 QQ, 整数环扩展 ZZ_I, 有理数域扩展 QQ_I 和代数域 ALG，执行如下断言
    for D in ZZ, QQ, ZZ_I, QQ_I, ALG:
        # 遍历 D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x) 这些环
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            # 断言环 R 的零特征是否为真，特征数是否为 0
            assert R.has_CharacteristicZero is True
            assert R.characteristic() == 0


# 定义一个测试函数，用于测试域对象的单位判断功能
def test_Domain_is_unit():
    # 定义一组测试数据：[-2, -1, 0, 1, 2]，以及预期的整数环和有理数域中的单位和非单位判断结果
    nums = [-2, -1, 0, 1, 2]
    invring = [False, True, False, True, False]
    invfield = [True, True, False, True, True]
    ZZx, QQx, QQxf = ZZ[x], QQ[x], QQ.frac_field(x)
    # 断言整数环 ZZ 和有理数域 QQ 中给定数值是否为单位和非单位
    assert [ZZ.is_unit(ZZ(n)) for n in nums] == invring
    assert [QQ.is_unit(QQ(n)) for n in nums] == invfield
    # 对于给定的整数列表 nums，检查它们是否为整数环 ZZx 中的单位元素，并与预期结果 invring 进行断言比较
    assert [ZZx.is_unit(ZZx(n)) for n in nums] == invring
    # 对于给定的有理数列表 nums，检查它们是否为有理数域 QQx 中的单位元素，并与预期结果 invfield 进行断言比较
    assert [QQx.is_unit(QQx(n)) for n in nums] == invfield
    # 对于给定的有理数域扩展列表 nums，检查它们是否为有理数域 QQxf 中的单位元素，并与预期结果 invfield 进行断言比较
    assert [QQxf.is_unit(QQxf(n)) for n in nums] == invfield
    # 检查整数 x 是否为整数环 ZZx 中的非单位元素，并断言结果为 False
    assert ZZx.is_unit(ZZx(x)) is False
    # 检查有理数 x 是否为有理数域 QQx 中的非单位元素，并断言结果为 False
    assert QQx.is_unit(QQx(x)) is False
    # 检查有理数域扩展 x 是否为有理数域 QQxf 中的单位元素，并断言结果为 True
    assert QQxf.is_unit(QQxf(x)) is True
def test_Domain_convert():

    def check_element(e1, e2, K1, K2, K3):
        # 检查两个元素的类型是否相同，如果不同则抛出断言错误，显示相关信息
        assert type(e1) is type(e2), '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)
        # 检查两个元素是否相等，如果不相等则抛出断言错误，显示相关信息
        assert e1 == e2, '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)

    def check_domains(K1, K2):
        # 将 K1 和 K2 合一得到 K3
        K3 = K1.unify(K2)
        # 检查 K3 转换回 K1 后是否等于 K1 的单位元素
        check_element(K3.convert_from(K1.one, K1),  K3.one,  K1, K2, K3)
        # 检查 K3 转换回 K2 后是否等于 K2 的单位元素
        check_element(K3.convert_from(K2.one, K2),  K3.one,  K1, K2, K3)
        # 检查 K3 转换回 K1 的零元素是否等于 K1 的零元素
        check_element(K3.convert_from(K1.zero, K1), K3.zero, K1, K2, K3)
        # 检查 K3 转换回 K2 的零元素是否等于 K2 的零元素
        check_element(K3.convert_from(K2.zero, K2), K3.zero, K1, K2, K3)

    def composite_domains(K):
        # 创建 K 的不同组合域的列表
        domains = [
            K,
            K[y], K[z], K[y, z],
            K.frac_field(y), K.frac_field(z), K.frac_field(y, z),
            # XXX: 这些应该被测试并使其工作...
            # K.old_poly_ring(y), K.old_frac_field(y),
        ]
        return domains

    # 创建有理数域的平方根扩展 QQ2 和 QQ3
    QQ2 = QQ.algebraic_field(sqrt(2))
    QQ3 = QQ.algebraic_field(sqrt(3))
    # 所有的域的列表
    doms = [ZZ, QQ, QQ2, QQ3, QQ_I, ZZ_I, RR, CC]

    # 对每对域进行组合域的测试
    for i, K1 in enumerate(doms):
        for K2 in doms[i:]:
            for K3 in composite_domains(K1):
                for K4 in composite_domains(K2):
                    check_domains(K3, K4)

    # 检查有理数域 QQ 对小数的转换
    assert QQ.convert(10e-52) == QQ(1684996666696915, 1684996666696914987166688442938726917102321526408785780068975640576)

    # 创建关于整数环的多项式环 R
    R, xr = ring("x", ZZ)
    # 检查整数环 ZZ 对于多项式变量 xr - xr 的转换是否等于 0
    assert ZZ.convert(xr - xr) == 0
    # 使用 R 的定义域检查整数环 ZZ 对于多项式变量 xr - xr 的转换是否等于 0
    assert ZZ.convert(xr - xr, R.to_domain()) == 0

    # 检查复数域 CC 对于有理数域整数 ZZ_I(1, 2) 的转换
    assert CC.convert(ZZ_I(1, 2)) == CC(1, 2)
    # 检查复数域 CC 对于有理数域有理数 QQ_I(1, 2) 的转换
    assert CC.convert(QQ_I(1, 2)) == CC(1, 2)

    # 检查有理数域 QQ 对于实数域 RR(0.5) 的转换
    assert QQ.convert_from(RR(0.5), RR) == QQ(1, 2)
    # 检查实数域 RR 对于有理数域 QQ(1/2) 的转换
    assert RR.convert_from(QQ(1, 2), QQ) == RR(0.5)
    # 检查有理数域 QQ_I 对于复数域 CC(0.5, 0.75) 的转换
    assert QQ_I.convert_from(CC(0.5, 0.75), CC) == QQ_I(QQ(1, 2), QQ(3, 4))
    # 检查复数域 CC 对于有理数域 QQ_I(QQ(1/2), QQ(3/4)) 的转换
    assert CC.convert_from(QQ_I(QQ(1, 2), QQ(3, 4)), QQ_I) == CC(0.5, 0.75)

    # 创建关于 x 的有理数域分式域 K1 和整数环的有理数域分式域 K2
    K1 = QQ.frac_field(x)
    K2 = ZZ.frac_field(x)
    K3 = QQ[x]
    K4 = ZZ[x]
    Ks = [K1, K2, K3, K4]
    # 对于 Ks 中的每对域 Ka, Kb 进行测试，检查 Ka 对于 Kb.from_sympy(x) 的转换是否等于 Ka.from_sympy(x)
    for Ka, Kb in product(Ks, Ks):
        assert Ka.convert_from(Kb.from_sympy(x), Kb) == Ka.from_sympy(x)

    # 检查整数环的有理数域对于有理数 QQ(1/2) 的转换
    assert K2.convert_from(QQ(1, 2), QQ) == K2(QQ(1, 2))


def test_EX_convert():

    elements = [
        (ZZ, ZZ(3)),
        (QQ, QQ(1,2)),
        (ZZ_I, ZZ_I(1,2)),
        (QQ_I, QQ_I(1,2)),
        (RR, RR(3)),
        (CC, CC(1,2)),
        (EX, EX(3)),
        (EXRAW, EXRAW(3)),
        (ALG, ALG.from_sympy(sqrt(2))),
    ]

    # 对于 elements 中的每对 R, e 进行测试
    for R, e in elements:
        for EE in EX, EXRAW:
            # 使用 R 的符号表示 e 创建 EE 的元素 elem
            elem = EE.from_sympy(R.to_sympy(e))
            # 检查 EE 对 e 的转换是否等于 elem
            assert EE.convert_from(e, R) == elem
            # 检查 R 对 elem 的转换是否等于 e
            assert R.convert_from(elem, EE) == e


def test_GlobalPolynomialRing_convert():
    # 创建关于 x 的整数环的旧多项式环 K1 和有理数域 QQ[x]
    K1 = QQ.old_poly_ring(x)
    K2 = QQ[x]
    # 检查 K1 对 x 的转换是否等于 K1 对 K2.convert(x) 转换后的结果
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    # 检查 K2 对 x 的转换是否等于 K2 对 K1.convert(x) 转换后的结果
    assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    # 创建关于 x, y 的整数环的旧多项式环 K1 和有理数域 QQ[x]
    K1 = QQ.old_poly_ring(x, y)
    K2 = QQ[x]
    # 检查 K1 对 x 的转换是否等于 K1 对 K2.convert(x) 转换后的结果
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    # 注释这行代码，因为它会导致语法错误
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    # 创建关于 x, y 的整数环的旧多项式环 K1 和整数环 ZZ[x]
    K1 = ZZ.old_poly_ring(x, y)
    K2 = QQ[x]
    # 断言：使用 K1 的转换方法将 x 转换后应与使用 K1 将 x 先转换为 K2 的类型，再转换回 K1 的结果相等
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    # 下一行代码被注释掉了，它断言使用 K2 的转换方法将 x 转换后应与使用 K2 将 x 先转换为 K1 的类型，再转换回 K2 的结果相等
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)
# 定义测试函数，初始化多项式环的测试
def test_PolynomialRing__init():
    # 调用 ring 函数创建一个环 R，使其等于一个空字符串和 ZZ 组成的元组中的第一个元素
    R, = ring("", ZZ)
    # 断言 ZZ 的多项式环函数等于 R 的域函数
    assert ZZ.poly_ring() == R.to_domain()


# 定义测试函数，初始化分式域的测试
def test_FractionField__init():
    # 调用 field 函数创建一个域 F，使其等于一个空字符串和 ZZ 组成的元组中的第一个元素
    F, = field("", ZZ)
    # 断言 ZZ 的分式域函数等于 F 的域函数
    assert ZZ.frac_field() == F.to_domain()


# 定义测试函数，转换分式域的测试
def test_FractionField_convert():
    # 创建 QQ 上关于变量 x 的分式域 K
    K = QQ.frac_field(x)
    # 断言 K 将 QQ 的有理数 2/3 转换为 QQ 类型的结果等于 K 中来自 SymPy 有理数 2/3 的转换结果
    assert K.convert(QQ(2, 3), QQ) == K.from_sympy(Rational(2, 3))
    # 再次创建 QQ 上关于变量 x 的分式域 K
    K = QQ.frac_field(x)
    # 断言 K 将 ZZ 的整数 2 转换为 ZZ 类型的结果等于 K 中来自 SymPy 整数 2 的转换结果
    assert K.convert(ZZ(2), ZZ) == K.from_sympy(Integer(2))


# 定义测试函数，注入生成元的测试
def test_inject():
    # 断言在 ZZ 中使用变量 x, y, z 注入得到的环等于 ZZ[x, y, z]
    assert ZZ.inject(x, y, z) == ZZ[x, y, z]
    # 断言在 ZZ[x] 中使用变量 y, z 注入得到的环等于 ZZ[x, y, z]
    assert ZZ[x].inject(y, z) == ZZ[x, y, z]
    # 断言在 QQ 中关于变量 x 的分式域，使用变量 y, z 注入得到的分式域等于 QQ 中关于变量 x, y, z 的分式域
    assert ZZ.frac_field(x).inject(y, z) == ZZ.frac_field(x, y, z)
    # 使用 lambda 表达式调用 raises 函数，断言在 ZZ[x] 中注入 x 会抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: ZZ[x].inject(x))


# 定义测试函数，丢弃生成元的测试
def test_drop():
    # 断言在 ZZ 中丢弃变量 x 得到的环等于 ZZ
    assert ZZ.drop(x) == ZZ
    # 断言在 ZZ[x] 中丢弃变量 x 得到的环等于 ZZ
    assert ZZ[x].drop(x) == ZZ
    # 断言在 ZZ[x, y] 中丢弃变量 x 得到的环等于 ZZ[y]
    assert ZZ[x, y].drop(x) == ZZ[y]
    # 断言在 QQ 中关于变量 x 的分式域，丢弃变量 x 得到的分式域等于 QQ
    assert ZZ.frac_field(x).drop(x) == ZZ
    # 断言在 QQ 中关于变量 x, y 的分式域，丢弃变量 x 得到的分式域等于 QQ 中关于变量 y 的分式域
    assert ZZ.frac_field(x, y).drop(x) == ZZ.frac_field(y)
    # 断言在 ZZ[x][y] 中丢弃变量 y 得到的环等于 ZZ[x]
    assert ZZ[x][y].drop(y) == ZZ[x]
    # 断言在 ZZ[x][y] 中丢弃变量 x 得到的环等于 ZZ[y]
    assert ZZ[x][y].drop(x) == ZZ[y]
    # 断言在 QQ 中关于变量 x 的分式域，再关于变量 y 丢弃变量 x 得到的分式域等于 QQ 关于变量 y 的分式域
    assert ZZ.frac_field(x)[y].drop(x) == ZZ[y]
    # 断言在 QQ 中关于变量 x 的分式域，再关于变量 y 丢弃变量 y 得到的分式域等于 QQ 关于变量 x 的分式域
    assert ZZ.frac_field(x)[y].drop(y) == ZZ.frac_field(x)
    # 创建 Ky 和 K 作为有限扩展对象，使 Ky 为 Poly(x**2-1, x, domain=ZZ[y]) 的有限扩展，K 为 Poly(x**2-1, x, domain=ZZ) 的有限扩展
    Ky = FiniteExtension(Poly(x**2-1, x, domain=ZZ[y]))
    K = FiniteExtension(Poly(x**2-1, x, domain=ZZ))
    # 断言在 Ky 中丢弃变量 y 得到 K
    assert Ky.drop(y) == K
    # 使用 lambda 表达式调用 raises 函数，断言在 Ky 中丢弃 x 会抛出 GeneratorsError 异常
    raises(GeneratorsError, lambda: Ky.drop(x))


# 定义测试函数，域映射的测试
def test_Domain_map():
    # 调用 ZZ 的 map 方法映射 [1, 2, 3, 4] 序列
    seq = ZZ.map([1, 2, 3, 4])
    # 断言 seq 中的所有元素都属于 ZZ 类型
    assert all(ZZ.of_type(elt) for elt in seq)
    # 调用 ZZ 的 map 方法映射 [[1, 2, 3, 4]] 序列
    seq = ZZ.map([[1, 2, 3, 4]])
    # 断言 seq 的第一个元素中的所有元素都属于 ZZ 类型，且 seq 的长度为 1
    assert all(ZZ.of_type(elt) for elt in seq[0]) and len(seq) == 1


# 定义测试函数，域相等性的测试
def test_Domain___eq__():
    # 断言 ZZ[x, y] 等于 ZZ[x, y]
    assert (ZZ[x, y] == ZZ[x, y]) is True
    # 断言 QQ[x, y] 等于 QQ[x, y]
    assert (QQ[x, y] == QQ[x, y]) is True

    # 断言 ZZ[x, y] 不等于 QQ[x, y]
    assert (ZZ[x, y] == QQ[x, y]) is False
    # 断言 QQ[x, y] 不等于 ZZ[x, y]
    assert (QQ[x, y] == ZZ[x, y]) is False

    # 断言 ZZ 关于变量 x, y 的分式域等于 ZZ 关于变量 x, y 的分式域
    assert (ZZ.frac_field(x, y) == ZZ.frac_field(x, y)) is True
    # 断言 QQ 关于变量 x, y 的分式域等于 QQ 关于变量 x, y 的分式域
    assert (QQ.frac_field(x, y) == QQ.frac_field(x, y)) is True

    # 断言 ZZ 关于变量 x, y 的分式域不等于 QQ 关于变量 x, y 的分式域
    assert (ZZ.frac_field(x, y) == QQ.frac_field(x, y)) is False
    # 断言 QQ 关于变量 x, y 的分式域不等于 ZZ 关于变量 x, y 的分式域
    assert (QQ.frac_field(x, y) == ZZ.frac_field(x, y)) is False

    # 断言 RealField() 关于变量 x 等于 RR 关于变量 x
    assert RealField()[x] == RR[x]


# 定义测试函数，代数域的测试
def test_Domain__algebraic_field():
    # 在 ZZ 中使用 sqrt(2) 创建代数域 alg
    alg = ZZ.algebraic_field(sqrt(2))
    # 断言 alg 的扩展的最小多项式为 x**2 - 2
    # 断言：验证 E.ext 和 K.ext 之间的字段同构性，如果不为 None，则断言通过
    assert field_isomorphism(E.ext, K.ext) is not None
    # 断言：验证 E 的定义域是否为有理数域 QQ
    assert E.dom == QQ
def test_PolynomialRing_from_FractionField():
    # 创建有理数域和多项式环，并定义变量
    F, x,y = field("x,y", ZZ)
    R, X,Y = ring("x,y", ZZ)

    # 定义分数和多项式
    f = (x**2 + y**2)/(x + 1)  # 分数 f
    g = (x**2 + y**2)/4        # 分数 g
    h =  x**2 + y**2           # 多项式 h

    # 断言从有理数域到多项式环的转换
    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2

    # 切换有理数域为 QQ
    F, x,y = field("x,y", QQ)
    R, X,Y = ring("x,y", QQ)

    # 重新定义分数和多项式
    f = (x**2 + y**2)/(x + 1)  # 分数 f
    g = (x**2 + y**2)/4        # 分数 g
    h =  x**2 + y**2           # 多项式 h

    # 断言从有理数域到多项式环的转换
    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2


def test_FractionField_from_PolynomialRing():
    # 创建多项式环和有理数域，并定义变量
    R, x,y = ring("x,y", QQ)
    F, X,Y = field("x,y", ZZ)

    # 定义多项式 f 和 g
    f = 3*x**2 + 5*y**2       # 多项式 f
    g = x**2/3 + y**2/5       # 多项式 g

    # 断言从多项式环到有理数域的转换
    assert F.to_domain().from_PolynomialRing(f, R.to_domain()) == 3*X**2 + 5*Y**2
    assert F.to_domain().from_PolynomialRing(g, R.to_domain()) == (5*X**2 + 3*Y**2)/15


def test_FF_of_type():
    # XXX: of_type 在这里并不是非常有用，因为在基础类型 flint 中所有元素都是 nmod 类型。
    assert FF(3).of_type(FF(3)(1)) is True
    assert FF(5).of_type(FF(5)(3)) is True


def test___eq__():
    # 断言 QQ[x] 和 ZZ[x] 不相等
    assert not QQ[x] == ZZ[x]
    # 断言 QQ.frac_field(x) 和 ZZ.frac_field(x) 不相等
    assert not QQ.frac_field(x) == ZZ.frac_field(x)


def test_RealField_from_sympy():
    # 断言转换 0 的结果
    assert RR.convert(S.Zero) == RR.dtype(0)
    # 断言转换 0.0 的结果
    assert RR.convert(S(0.0)) == RR.dtype(0.0)
    # 断言转换 1 的结果
    assert RR.convert(S.One) == RR.dtype(1)
    # 断言转换 1.0 的结果
    assert RR.convert(S(1.0)) == RR.dtype(1.0)
    # 断言转换 sin(1) 的结果
    assert RR.convert(sin(1)) == RR.dtype(sin(1).evalf())


def test_not_in_any_domain():
    # 检查非法列表中的元素是否不在任何域中
    check = list(_illegal) + [x] + [
        float(i) for i in _illegal[:3]]
    for dom in (ZZ, QQ, RR, CC, EX):
        for i in check:
            if i == x and dom == EX:
                continue
            assert i not in dom, (i, dom)
            raises(CoercionFailed, lambda: dom.convert(i))


def test_ModularInteger():
    # 创建模整数域 F3
    F3 = FF(3)

    # 测试不同元素在模整数域中的行为
    a = F3(0)
    assert F3.of_type(a) and a == 0
    a = F3(1)
    assert F3.of_type(a) and a == 1
    a = F3(2)
    assert F3.of_type(a) and a == 2
    a = F3(3)
    assert F3.of_type(a) and a == 0
    a = F3(4)
    assert F3.of_type(a) and a == 1

    # 测试模整数域中的嵌套行为
    a = F3(F3(0))
    assert F3.of_type(a) and a == 0
    a = F3(F3(1))
    assert F3.of_type(a) and a == 1
    a = F3(F3(2))
    assert F3.of_type(a) and a == 2
    a = F3(F3(3))
    assert F3.of_type(a) and a == 0
    a = F3(F3(4))
    assert F3.of_type(a) and a == 1

    # 测试负数和加法
    a = -F3(1)
    assert F3.of_type(a) and a == 2
    a = -F3(2)
    assert F3.of_type(a) and a == 1

    # 测试加法操作
    a = 2 + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + 2
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1

    # 测试减法操作
    a = 3 - F3(2)
    assert F3.of_type(a) and a == 1
    # 使用 F3 类创建一个对象，并计算其值减去 2
    a = F3(3) - 2
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的差
    a = F3(3) - F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的差
    a = F3(3) - F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值乘以 2
    a = 2 * F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值乘以 2
    a = F3(2) * 2
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的乘积
    a = F3(2) * F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的乘积
    a = F3(2) * F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值除以 2
    a = 2 / F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值除以 2
    a = F3(2) / 2
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的商
    a = F3(2) / F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类分别创建两个对象，并计算它们的商
    a = F3(2) / F3(2)
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值的 0 次方
    a = F3(2) ** 0
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 F3 类创建一个对象，并计算其值的 1 次方
    a = F3(2) ** 1
    # 断言对象 a 属于 F3 类型，并且其值等于 2
    assert F3.of_type(a) and a == 2

    # 使用 F3 类创建一个对象，并计算其值的 2 次方
    a = F3(2) ** 2
    # 断言对象 a 属于 F3 类型，并且其值等于 1
    assert F3.of_type(a) and a == 1

    # 使用 FF 类创建一个对象，其基础数为 7
    F7 = FF(7)

    # 使用 F7 类创建一个对象，并计算其值的 100000000000 次方
    a = F7(3) ** 100000000000
    # 断言对象 a 属于 F7 类型，并且其值等于 4
    assert F7.of_type(a) and a == 4

    # 使用 F7 类创建一个对象，并计算其值的 -100000000000 次方
    a = F7(3) ** -100000000000
    # 断言对象 a 属于 F7 类型，并且其值等于 2
    assert F7.of_type(a) and a == 2

    # 断言 F3 类创建的对象为假
    assert bool(F3(3)) is False
    # 断言 F3 类创建的对象为真
    assert bool(F3(4)) is True

    # 使用 FF 类创建一个对象，其基础数为 5
    F5 = FF(5)

    # 使用 F5 类创建一个对象，并计算其值的 -1 次方
    a = F5(1) ** (-1)
    # 断言对象 a 属于 F5 类型，并且其值等于 1
    assert F5.of_type(a) and a == 1

    # 使用 F5 类创建一个对象，并计算其值的 -1 次方
    a = F5(2) ** (-1)
    # 断言对象 a 属于 F5 类型，并且其值等于 3
    assert F5.of_type(a) and a == 3

    # 使用 F5 类创建一个对象，并计算其值的 -1 次方
    a = F5(3) ** (-1)
    # 断言对象 a 属于 F5 类型，并且其值等于 2
    assert F5.of_type(a) and a == 2

    # 使用 F5 类创建一个对象，并计算其值的 -1 次方
    a = F5(4) ** (-1)
    # 断言对象 a 属于 F5 类型，并且其值等于 4
    assert F5.of_type(a) and a == 4

    # 如果 GROUND_TYPES 不等于 'flint'，则执行下列操作
    if GROUND_TYPES != 'flint':
        # XXX: 在使用 python-flint 时会导致核心转储…
        raises(NotInvertible, lambda: F5(0) ** (-1))
        raises(NotInvertible, lambda: F5(5) ** (-1))

    # 断言创建 FF 类对象时基础数为 0 会引发 ValueError
    raises(ValueError, lambda: FF(0))
    # 断言创建 FF 类对象时基础数为 2.1 会引发 ValueError
    raises(ValueError, lambda: FF(2.1))

    # 遍历范围在 0 到 5 之间的数值
    for n1 in range(5):
        # 再次遍历范围在 0 到 5 之间的数值
        for n2 in range(5):
            # 如果 GROUND_TYPES 不等于 'flint'，则执行下列操作
            if GROUND_TYPES != 'flint':
                # 使用 warns_deprecated_sympy 函数上下文管理器来断言两个对象之间的比较结果
                with warns_deprecated_sympy():
                    assert (F5(n1) < F5(n2)) is (n1 < n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) <= F5(n2)) is (n1 <= n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) > F5(n2)) is (n1 > n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) >= F5(n2)) is (n1 >= n2)
            else:
                # 如果 GROUND_TYPES 等于 'flint'，则断言两个对象的比较会引发 TypeError
                raises(TypeError, lambda: F5(n1) < F5(n2))
                raises(TypeError, lambda: F5(n1) <= F5(n2))
                raises(TypeError, lambda: F5(n1) >
def test_QQ_int():
    # 确定 QQ 类型对大整数的处理正确性
    assert int(QQ(2**2000, 3**1250)) == 455431
    # 确定 QQ 类型对小整数的处理正确性
    assert int(QQ(2**100, 3)) == 422550200076076467165567735125

def test_RR_double():
    # 确定 RR 类型对双精度浮点数的处理正确性
    assert RR(3.14) > 1e-50
    assert RR(1e-13) > 1e-50
    assert RR(1e-14) > 1e-50
    assert RR(1e-15) > 1e-50
    assert RR(1e-20) > 1e-50
    assert RR(1e-40) > 1e-50

def test_RR_Float():
    # 测试 Float 类型的精度设置
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    assert f1._prec == 53
    assert f2._prec == 80
    # 确定 RR 类型与 Float 实例之间的计算正确性
    assert RR(f1)-1 > 1e-50
    assert RR(f2)-1 < 1e-50  # RR 的精度低于 f2 的精度

    RR2 = RealField(prec=f2._prec)
    # 确定自定义精度 RR2 与 Float 实例之间的计算正确性
    assert RR2(f1)-1 > 1e-50
    assert RR2(f2)-1 > 1e-50  # RR 的精度与 f2 的精度相同


def test_CC_double():
    # 确定 CC 类型对双精度浮点数实部的处理正确性
    assert CC(3.14).real > 1e-50
    assert CC(1e-13).real > 1e-50
    assert CC(1e-14).real > 1e-50
    assert CC(1e-15).real > 1e-50
    assert CC(1e-20).real > 1e-50
    assert CC(1e-40).real > 1e-50
    # 确定 CC 类型对双精度浮点数虚部的处理正确性
    assert CC(3.14j).imag > 1e-50
    assert CC(1e-13j).imag > 1e-50
    assert CC(1e-14j).imag > 1e-50
    assert CC(1e-15j).imag > 1e-50
    assert CC(1e-20j).imag > 1e-50
    assert CC(1e-40j).imag > 1e-50


def test_gaussian_domains():
    I = S.ImaginaryUnit
    a, b, c, d = [ZZ_I.convert(x) for x in (5, 2 + I, 3 - I, 5 - 5*I)]
    # 确定 ZZ_I 类型的 gcd 函数正确性
    assert ZZ_I.gcd(a, b) == b
    assert ZZ_I.gcd(a, c) == b
    # 确定 ZZ_I 类型的 lcm 函数正确性
    assert ZZ_I.lcm(a, b) == a
    assert ZZ_I.lcm(a, c) == d
    # 确定 ZZ_I 与 QQ_I 类型比较的正确性
    assert ZZ_I(3, 4) != QQ_I(3, 4)  # 这个是否正确或者应该转换为 ZZ->QQ？
    assert ZZ_I(3, 0) != 3           # 这个应该转换为 Integer 吗？
    assert QQ_I(S(3)/4, 0) != S(3)/4 # 这个应该转换为 Rational 吗？
    # 确定 ZZ_I 实例的象限函数正确性
    assert ZZ_I(0, 0).quadrant() == 0
    assert ZZ_I(-1, 0).quadrant() == 2

    assert QQ_I.convert(QQ(3, 2)) == QQ_I(QQ(3, 2), QQ(0))
    assert QQ_I.convert(QQ(3, 2), QQ) == QQ_I(QQ(3, 2), QQ(0))

def test_EX_EXRAW():
    # 确定 EXRAW 类型的 zero 和 one 常量
    assert EXRAW.zero is S.Zero
    assert EXRAW.one is S.One

    assert EX(1) == EX.Expression(1)
    assert EX(1).ex is S.One
    assert EXRAW(1) is S.One

    # EX 类型有取消操作，而 EXRAW 类型没有
    assert 2*EX((x + y*x)/x) == EX(2 + 2*y) != 2*((x + y*x)/x)
    assert 2*EXRAW((x + y*x)/x) == 2*((x + y*x)/x) != (1 + y)

    # 确定从 EX 类型转换为 EXRAW 类型的正确性
    assert EXRAW.convert_from(EX(1), EX) is EXRAW.one
    assert EX.convert_from(EXRAW(1), EXRAW) == EX.one

    # 确定 EXRAW 类型与 sympy 对象之间的转换正确性
    assert EXRAW.from_sympy(S.One) is S.One
    assert EXRAW.to_sympy(EXRAW.one) is S.One
    raises(CoercionFailed, lambda: EXRAW.from_sympy([]))

    # 确定获取 EXRAW 类型的域正确性
    assert EXRAW.get_field() == EXRAW

    # 确定 EXRAW 类型与 EX 类型的统一性
    assert EXRAW.unify(EX) == EXRAW
    assert EX.unify(EXRAW) == EXRAW


def test_EX_ordering():
    elements = [EX(1), EX(x), EX(3)]
    # 确定 EX 类型的排序正确性
    assert sorted(elements) == [EX(1), EX(3), EX(x)]


def test_canonical_unit():

    for K in [ZZ, QQ, RR]: # CC?
        # 确定 K 类型的 canonical_unit 函数的正确性
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(-2)) == K(-1)
    # 对于每个环（域）K，使用给定的元素进行单元归一化测试
    for K in [ZZ_I, QQ_I]:
        # 将符号表示的虚数单位转换为环（域）K中的对象
        i = K.from_sympy(I)
        # 断言：对于元素K(2)，其单元归一化后应为K(1)
        assert K.canonical_unit(K(2)) == K(1)
        # 断言：对于元素K(2)*i，其单元归一化后应为-i
        assert K.canonical_unit(K(2)*i) == -i
        # 断言：对于元素-K(2)，其单元归一化后应为-K(1)
        assert K.canonical_unit(-K(2)) == K(-1)
        # 断言：对于元素-K(2)*i，其单元归一化后应为i
        assert K.canonical_unit(-K(2)*i) == i
    
    # 使用环ZZ[x]，进行单元归一化测试
    K = ZZ[x]
    # 断言：对于元素K(x + 1)，其单元归一化后应为K(1)
    assert K.canonical_unit(K(x + 1)) == K(1)
    # 断言：对于元素K(-x + 1)，其单元归一化后应为-K(1)
    assert K.canonical_unit(K(-x + 1)) == K(-1)
    
    # 使用环ZZ_I[x]，进行单元归一化测试
    K = ZZ_I[x]
    # 断言：对于从符号I*x转换而来的元素，其单元归一化后应为ZZ_I(0, -1)
    assert K.canonical_unit(K.from_sympy(I*x)) == ZZ_I(0, -1)
    
    # 使用域ZZ_I上的分式域，进行单元归一化测试
    K = ZZ_I.frac_field(x, y)
    # 将符号表示的虚数单位转换为分式域K中的对象
    i = K.from_sympy(I)
    # 断言：i除以i应等于分式域K中的单位元K.one
    assert i / i == K.one
    # 断言：(K.one + i)/(i - K.one)应等于-i
    assert (K.one + i)/(i - K.one) == -i
def test_issue_18278():
    # 检查RR类实例的父类是否为字符串'RR'
    assert str(RR(2).parent()) == 'RR'
    # 检查CC类实例的父类是否为字符串'CC'
    assert str(CC(2).parent()) == 'CC'


def test_Domain_is_negative():
    # 获取虚数单位
    I = S.ImaginaryUnit
    # 将复数转换为复数域中的复数实例
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    # 检查a是否为负数
    assert CC.is_negative(a) == False
    # 检查b是否为负数
    assert CC.is_negative(b) == False


def test_Domain_is_positive():
    # 获取虚数单位
    I = S.ImaginaryUnit
    # 将复数转换为复数域中的复数实例
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    # 检查a是否为正数
    assert CC.is_positive(a) == False
    # 检查b是否为正数
    assert CC.is_positive(b) == False


def test_Domain_is_nonnegative():
    # 获取虚数单位
    I = S.ImaginaryUnit
    # 将复数转换为复数域中的复数实例
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    # 检查a是否为非负数
    assert CC.is_nonnegative(a) == False
    # 检查b是否为非负数
    assert CC.is_nonnegative(b) == False


def test_Domain_is_nonpositive():
    # 获取虚数单位
    I = S.ImaginaryUnit
    # 将复数转换为复数域中的复数实例
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    # 检查a是否为非正数
    assert CC.is_nonpositive(a) == False
    # 检查b是否为非正数
    assert CC.is_nonpositive(b) == False


def test_exponential_domain():
    # 创建包含E的整数环
    K = ZZ[E]
    # 从SymPy中创建eK
    eK = K.from_sympy(E)
    # 检查从SymPy中创建exp(3)是否等于eK的3次方
    assert K.from_sympy(exp(3)) == eK ** 3
    # 检查转换exp(3)是否等于eK的3次方
    assert K.convert(exp(3)) == eK ** 3


def test_AlgebraicField_alias():
    # 没有默认别名:
    k = QQ.algebraic_field(sqrt(2))
    # 断言k的扩展没有别名
    assert k.ext.alias is None

    # 对于单个扩展，使用其别名:
    alpha = AlgebraicNumber(sqrt(2), alias='alpha')
    k = QQ.algebraic_field(alpha)
    # 断言k的扩展别名为'alpha'
    assert k.ext.alias.name == 'alpha'

    # 可以覆盖单个扩展的别名:
    k = QQ.algebraic_field(alpha, alias='theta')
    # 断言k的扩展别名为'theta'
    assert k.ext.alias.name == 'theta'

    # 对于多个扩展，没有默认别名:
    k = QQ.algebraic_field(sqrt(2), sqrt(3))
    # 断言k的扩展没有别名
    assert k.ext.alias is None

    # 对于多个扩展，即使其中一个扩展有别名，也没有默认别名:
    k = QQ.algebraic_field(alpha, sqrt(3))
    # 断言k的扩展没有别名
    assert k.ext.alias is None

    # 对于多个扩展，可以设置一个别名:
    k = QQ.algebraic_field(sqrt(2), sqrt(3), alias='theta')
    # 断言k的扩展别名为'theta'
    assert k.ext.alias.name == 'theta'

    # 别名传递给构造的域元素:
    k = QQ.algebraic_field(alpha)
    beta = k.to_alg_num(k([1, 2, 3]))
    # 断言beta的别名与alpha的别名相同
    assert beta.alias is alpha.alias


def test_exsqrt():
    # 检查整数环中的数是否是完全平方数
    assert ZZ.is_square(ZZ(4)) is True
    # 检查整数环中完全平方数的平方根
    assert ZZ.exsqrt(ZZ(4)) == ZZ(2)
    # 检查整数环中的数是否是完全平方数
    assert ZZ.is_square(ZZ(42)) is False
    # 检查整数环中非完全平方数的平方根
    assert ZZ.exsqrt(ZZ(42)) is None
    # 检查整数环中的数是否是完全平方数
    assert ZZ.is_square(ZZ(0)) is True
    # 检查整数环中完全平方数的平方根
    assert ZZ.exsqrt(ZZ(0)) == ZZ(0)
    # 检查整数环中的数是否是完全平方数
    assert ZZ.is_square(ZZ(-1)) is False
    # 检查整数环中非完全平方数的平方根
    assert ZZ.exsqrt(ZZ(-1)) is None

    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(9, 4)) is True
    # 检查有理数环中完全平方数的平方根
    assert QQ.exsqrt(QQ(9, 4)) == QQ(3, 2)
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(18, 8)) is True
    # 检查有理数环中完全平方数的平方根
    assert QQ.exsqrt(QQ(18, 8)) == QQ(3, 2)
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(-9, -4)) is True
    # 检查有理数环中完全平方数的平方根
    assert QQ.exsqrt(QQ(-9, -4)) == QQ(3, 2)
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(11, 4)) is False
    # 检查有理数环中非完全平方数的平方根
    assert QQ.exsqrt(QQ(11, 4)) is None
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(9, 5)) is False
    # 检查有理数环中非完全平方数的平方根
    assert QQ.exsqrt(QQ(9, 5)) is None
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(4)) is True
    # 检查有理数环中完全平方数的平方根
    assert QQ.exsqrt(QQ(4)) == QQ(2)
    # 检查有理数环中的数是否是完全平方数
    assert QQ.is_square(QQ(0)) is True
    # 检查有理数环中完全平方数的平方根
    assert QQ.exsqrt(QQ(0)) == QQ(0)
    # 使用 QQ 对象检查给定的有理数是否为平方数，返回布尔值 False
    assert QQ.is_square(QQ(-16, 9)) is False
    
    # 使用 QQ 对象尝试从给定的有理数中提取平方根，返回 None
    assert QQ.exsqrt(QQ(-16, 9)) is None
    
    # 使用 RR 对象检查给定的实数是否为平方数，返回布尔值 True
    assert RR.is_square(RR(6.25)) is True
    
    # 使用 RR 对象从给定的实数中提取平方根，并断言其值为 2.5
    assert RR.exsqrt(RR(6.25)) == RR(2.5)
    
    # 使用 RR 对象检查给定的实数是否为平方数，返回布尔值 True
    assert RR.is_square(RR(2)) is True
    
    # 使用 RR 对象从给定的实数中提取平方根，并使用容差断言其接近于 1.4142135623730951
    assert RR.almosteq(RR.exsqrt(RR(2)), RR(1.4142135623730951), tolerance=1e-15)
    
    # 使用 RR 对象检查给定的实数是否为平方数，返回布尔值 True
    assert RR.is_square(RR(0)) is True
    
    # 使用 RR 对象从给定的实数中提取平方根，并断言其值为 0
    assert RR.exsqrt(RR(0)) == RR(0)
    
    # 使用 RR 对象检查给定的实数是否为平方数，返回布尔值 False
    assert RR.is_square(RR(-1)) is False
    
    # 使用 RR 对象尝试从给定的实数中提取平方根，返回 None
    assert RR.exsqrt(RR(-1)) is None
    
    # 使用 CC 对象检查给定的复数是否为平方数，返回布尔值 True
    assert CC.is_square(CC(2)) is True
    
    # 使用 CC 对象从给定的复数中提取平方根，并使用容差断言其接近于 1.4142135623730951
    assert CC.almosteq(CC.exsqrt(CC(2)), CC(1.4142135623730951), tolerance=1e-15)
    
    # 使用 CC 对象检查给定的复数是否为平方数，返回布尔值 True
    assert CC.is_square(CC(0)) is True
    
    # 使用 CC 对象从给定的复数中提取平方根，并断言其值为 0
    assert CC.exsqrt(CC(0)) == CC(0)
    
    # 使用 CC 对象检查给定的复数是否为平方数，返回布尔值 True
    assert CC.is_square(CC(-1)) is True
    
    # 使用 CC 对象从给定的复数中提取平方根，并断言其值为复数单位虚部
    assert CC.exsqrt(CC(-1)) == CC(0, 1)
    
    # 使用 CC 对象检查给定的复数是否为平方数，返回布尔值 True
    assert CC.is_square(CC(0, 2)) is True
    
    # 使用 CC 对象从给定的复数中提取平方根，并断言其值为复数 1 + i
    assert CC.exsqrt(CC(0, 2)) == CC(1, 1)
    
    # 使用 CC 对象检查给定的复数是否为平方数，返回布尔值 True
    assert CC.is_square(CC(-3, -4)) is True
    
    # 使用 CC 对象从给定的复数中提取平方根，并断言其值为复数 1 - 2i
    assert CC.exsqrt(CC(-3, -4)) == CC(1, -2)
    
    # 创建有限域 F2，检查给定的有限域元素是否为平方数，返回布尔值 True
    F2 = FF(2)
    assert F2.is_square(F2(1)) is True
    
    # 在有限域 F2 中尝试从给定的元素中提取平方根，并断言其值为 1
    assert F2.exsqrt(F2(1)) == F2(1)
    
    # 创建有限域 F2，检查给定的有限域元素是否为平方数，返回布尔值 True
    assert F2.is_square(F2(0)) is True
    
    # 在有限域 F2 中从给定的元素中提取平方根，并断言其值为 0
    assert F2.exsqrt(F2(0)) == F2(0)
    
    # 创建有限域 F7，检查给定的有限域元素是否为平方数，返回布尔值 True
    F7 = FF(7)
    assert F7.is_square(F7(2)) is True
    
    # 在有限域 F7 中从给定的元素中提取平方根，并断言其值为 3
    assert F7.exsqrt(F7(2)) == F7(3)
    
    # 创建有限域 F7，检查给定的有限域元素是否为平方数，返回布尔值 False
    assert F7.is_square(F7(3)) is False
    
    # 在有限域 F7 中尝试从给定的元素中提取平方根，返回 None
    assert F7.exsqrt(F7(3)) is None
    
    # 创建有限域 F7，检查给定的有限域元素是否为平方数，返回布尔值 True
    assert F7.is_square(F7(0)) is True
    
    # 在有限域 F7 中从给定的元素中提取平方根，并断言其值为 0
    assert F7.exsqrt(F7(0)) == F7(0)
```