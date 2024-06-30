# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_cartesian.py`

```
"""Tests for cartesian.py"""

# 导入需要的模块和符号
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval

# 导入量子物理相关模块和类
from sympy.physics.quantum import qapply, represent, L2, Dagger
from sympy.physics.quantum import Commutator, hbar
from sympy.physics.quantum.cartesian import (
    XOp, YOp, ZOp, PxOp, X, Y, Z, Px, XKet, XBra, PxKet, PxBra,
    PositionKet3D, PositionBra3D
)
from sympy.physics.quantum.operator import DifferentialOperator

# 定义符号变量
x, y, z, x_1, x_2, x_3, y_1, z_1 = symbols('x,y,z,x_1,x_2,x_3,y_1,z_1')
px, py, px_1, px_2 = symbols('px py px_1 px_2')


# 测试函数 test_x 开始
def test_x():
    # 断言X算符的希尔伯特空间
    assert X.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    # 断言X和Px的对易子结果
    assert Commutator(X, Px).doit() == I*hbar
    # 断言对XKet(x)应用X算符的结果
    assert qapply(X*XKet(x)) == x*XKet(x)
    # 断言XKet(x)的对偶类为XBra
    assert XKet(x).dual_class() == XBra
    # 断言XBra(x)的对偶类为XKet
    assert XBra(x).dual_class() == XKet
    # 断言Dagger(XKet(y)) * XKet(x)的结果
    assert (Dagger(XKet(y))*XKet(x)).doit() == DiracDelta(x - y)
    # 断言PxBra(px) * XKet(x)的结果
    assert (PxBra(px)*XKet(x)).doit() == exp(-I*x*px/hbar)/sqrt(2*pi*hbar)
    # 断言XKet(x)的表示
    assert represent(XKet(x)) == DiracDelta(x - x_1)
    # 断言XBra(x)的表示
    assert represent(XBra(x)) == DiracDelta(-x + x_1)
    # 断言XBra(x)的position属性
    assert XBra(x).position == x
    # 断言XOp() * XKet()的表示
    assert represent(XOp()*XKet()) == x*DiracDelta(x - x_2)
    # 断言XOp() * XKet() * XBra('y')的表示
    assert represent(XOp()*XKet()*XBra('y')) == \
        x*DiracDelta(x - x_3)*DiracDelta(x_1 - y)
    # 断言XBra("y") * XKet()的表示
    assert represent(XBra("y")*XKet()) == DiracDelta(x - y)
    # 断言XKet() * XBra()的表示
    assert represent(
        XKet()*XBra()) == DiracDelta(x - x_2) * DiracDelta(x_1 - x)

    # 使用不同的基础运算符PxOp计算XOp()的表示
    rep_p = represent(XOp(), basis=PxOp)
    assert rep_p == hbar*I*DiracDelta(px_1 - px_2)*DifferentialOperator(px_1)
    assert rep_p == represent(XOp(), basis=PxOp())
    assert rep_p == represent(XOp(), basis=PxKet)
    assert rep_p == represent(XOp(), basis=PxKet())

    # 断言XOp() * PxKet()的表示
    assert represent(XOp()*PxKet(), basis=PxKet) == \
        hbar*I*DiracDelta(px - px_2)*DifferentialOperator(px)


# 测试函数 test_p 开始
def test_p():
    # 断言Px算符的希尔伯特空间
    assert Px.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    # 断言对Px * PxKet(px)应用qapply的结果
    assert qapply(Px*PxKet(px)) == px*PxKet(px)
    # 断言PxKet(px)的对偶类为PxBra
    assert PxKet(px).dual_class() == PxBra
    # 断言PxBra(x)的对偶类为PxKet
    assert PxBra(x).dual_class() == PxKet
    # 断言Dagger(PxKet(py)) * PxKet(px)的结果
    assert (Dagger(PxKet(py))*PxKet(px)).doit() == DiracDelta(px - py)
    # 断言XBra(x) * PxKet(px)的结果
    assert (XBra(x)*PxKet(px)).doit() == \
        exp(I*x*px/hbar)/sqrt(2*pi*hbar)
    # 断言PxKet(px)的表示
    assert represent(PxKet(px)) == DiracDelta(px - px_1)

    # 使用不同的基础运算符XOp计算PxOp()的表示
    rep_x = represent(PxOp(), basis=XOp)
    assert rep_x == -hbar*I*DiracDelta(x_1 - x_2)*DifferentialOperator(x_1)
    assert rep_x == represent(PxOp(), basis=XOp())
    assert rep_x == represent(PxOp(), basis=XKet)
    assert rep_x == represent(PxOp(), basis=XKet())

    # 断言PxOp() * XKet()的表示
    assert represent(PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - x_2)*DifferentialOperator(x)
    # 断言XBra("y") * PxOp() * XKet()的表示
    assert represent(XBra("y")*PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - y)*DifferentialOperator(x)
# 定义名为 test_3dpos 的测试函数
def test_3dpos():
    # 断言 Y 的希尔伯特空间为区间负无穷到正无穷的 L2 空间
    assert Y.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    # 断言 Z 的希尔伯特空间为区间负无穷到正无穷的 L2 空间

    assert Z.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))

    # 创建一个三维位置态 PositionKet3D 对象，使用变量 x, y, z 作为参数
    test_ket = PositionKet3D(x, y, z)
    # 断言对 X 算符作用于 test_ket 后的结果为 x * test_ket
    assert qapply(X*test_ket) == x*test_ket
    # 断言对 Y 算符作用于 test_ket 后的结果为 y * test_ket
    assert qapply(Y*test_ket) == y*test_ket
    # 断言对 Z 算符作用于 test_ket 后的结果为 z * test_ket
    assert qapply(Z*test_ket) == z*test_ket
    # 断言对 X*Y 算符作用于 test_ket 后的结果为 x * y * test_ket
    assert qapply(X*Y*test_ket) == x*y*test_ket
    # 断言对 X*Y*Z 算符作用于 test_ket 后的结果为 x * y * z * test_ket
    assert qapply(X*Y*Z*test_ket) == x*y*z*test_ket
    # 断言对 Y*Z 算符作用于 test_ket 后的结果为 y * z * test_ket
    assert qapply(Y*Z*test_ket) == y*z*test_ket

    # 断言 PositionKet3D 类的默认构造函数返回的对象与 test_ket 相等
    assert PositionKet3D() == test_ket
    # 断言 YOp() 返回 Y 算符
    assert YOp() == Y
    # 断言 ZOp() 返回 Z 算符
    assert ZOp() == Z

    # 断言 PositionKet3D 的对偶类为 PositionBra3D
    assert PositionKet3D.dual_class() == PositionBra3D
    # 断言 PositionBra3D 的对偶类为 PositionKet3D
    assert PositionBra3D.dual_class() == PositionKet3D

    # 创建另一个三维位置态对象 other_ket，使用变量 x_1, y_1, z_1 作为参数
    other_ket = PositionKet3D(x_1, y_1, z_1)
    # 断言 Dagger(other_ket) * test_ket 的结果使用 doit() 方法后等于三个 DiracDelta 函数的乘积
    assert (Dagger(other_ket)*test_ket).doit() == \
        DiracDelta(x - x_1)*DiracDelta(y - y_1)*DiracDelta(z - z_1)

    # 断言 test_ket 对象的 position_x 属性等于 x
    assert test_ket.position_x == x
    # 断言 test_ket 对象的 position_y 属性等于 y
    assert test_ket.position_y == y
    # 断言 test_ket 对象的 position_z 属性等于 z
    assert test_ket.position_z == z
    # 断言 other_ket 对象的 position_x 属性等于 x_1
    assert other_ket.position_x == x_1
    # 断言 other_ket 对象的 position_y 属性等于 y_1
    assert other_ket.position_y == y_1
    # 断言 other_ket 对象的 position_z 属性等于 z_1

    # TODO: Add tests for representations
```