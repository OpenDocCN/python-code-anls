# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_operatorset.py`

```
# 导入 SymPy 库中的特定模块和类

from sympy.core.singleton import S  # 导入 SymPy 核心库中的 S 单例对象

from sympy.physics.quantum.operatorset import (  # 导入 SymPy 量子物理库中的操作符转换函数
    operators_to_state, state_to_operators
)

from sympy.physics.quantum.cartesian import (  # 导入 SymPy 量子物理库中的笛卡尔运算符和态
    XOp, XKet, PxOp, PxKet, XBra, PxBra
)

from sympy.physics.quantum.state import Ket, Bra  # 导入 SymPy 量子物理库中的态和双态
from sympy.physics.quantum.operator import Operator  # 导入 SymPy 量子物理库中的操作符
from sympy.physics.quantum.spin import (  # 导入 SymPy 量子物理库中的自旋态和自旋操作符
    JxKet, JyKet, JzKet, JxBra, JyBra, JzBra,
    JxOp, JyOp, JzOp, J2Op
)

from sympy.testing.pytest import raises  # 导入 SymPy 测试库中的异常抛出函数


def test_spin():
    # 断言：将自旋操作符 {J2Op, JxOp} 转换为自旋态 JxKet
    assert operators_to_state({J2Op, JxOp}) == JxKet
    # 断言：将自旋操作符 {J2Op, JyOp} 转换为自旋态 JyKet
    assert operators_to_state({J2Op, JyOp}) == JyKet
    # 断言：将自旋操作符 {J2Op, JzOp} 转换为自旋态 JzKet
    assert operators_to_state({J2Op, JzOp}) == JzKet
    # 断言：将自旋操作符 {J2Op(), JxOp()} 转换为自旋态 JxKet
    assert operators_to_state({J2Op(), JxOp()}) == JxKet
    # 断言：将自旋操作符 {J2Op(), JyOp()} 转换为自旋态 JyKet
    assert operators_to_state({J2Op(), JyOp()}) == JyKet
    # 断言：将自旋操作符 {J2Op(), JzOp()} 转换为自旋态 JzKet
    assert operators_to_state({J2Op(), JzOp()}) == JzKet

    # 断言：将自旋态 JxKet 转换为自旋操作符集合 {J2Op, JxOp}
    assert state_to_operators(JxKet) == {J2Op, JxOp}
    # 断言：将自旋态 JyKet 转换为自旋操作符集合 {J2Op, JyOp}
    assert state_to_operators(JyKet) == {J2Op, JyOp}
    # 断言：将自旋态 JzKet 转换为自旋操作符集合 {J2Op, JzOp}
    assert state_to_operators(JzKet) == {J2Op, JzOp}
    # 断言：将自旋态 JxBra 转换为自旋操作符集合 {J2Op, JxOp}
    assert state_to_operators(JxBra) == {J2Op, JxOp}
    # 断言：将自旋态 JyBra 转换为自旋操作符集合 {J2Op, JyOp}
    assert state_to_operators(JyBra) == {J2Op, JyOp}
    # 断言：将自旋态 JzBra 转换为自旋操作符集合 {J2Op, JzOp}
    assert state_to_operators(JzBra) == {J2Op, JzOp}

    # 断言：将自旋态 JxKet(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JxOp()}
    assert state_to_operators(JxKet(S.Half, S.Half)) == {J2Op(), JxOp()}
    # 断言：将自旋态 JyKet(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JyOp()}
    assert state_to_operators(JyKet(S.Half, S.Half)) == {J2Op(), JyOp()}
    # 断言：将自旋态 JzKet(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JzOp()}
    assert state_to_operators(JzKet(S.Half, S.Half)) == {J2Op(), JzOp()}
    # 断言：将自旋态 JxBra(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JxOp()}
    assert state_to_operators(JxBra(S.Half, S.Half)) == {J2Op(), JxOp()}
    # 断言：将自旋态 JyBra(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JyOp()}
    assert state_to_operators(JyBra(S.Half, S.Half)) == {J2Op(), JyOp()}
    # 断言：将自旋态 JzBra(S.Half, S.Half) 转换为自旋操作符集合 {J2Op(), JzOp()}
    assert state_to_operators(JzBra(S.Half, S.Half)) == {J2Op(), JzOp()}


def test_op_to_state():
    # 断言：将 XOp 转换为 XKet()，即笛卡尔坐标系下的位置算符 XOp 对应的态 XKet()
    assert operators_to_state(XOp) == XKet()
    # 断言：将 PxOp 转换为 PxKet()，即笛卡尔坐标系下的动量算符 PxOp 对应的态 PxKet()
    assert operators_to_state(PxOp) == PxKet()
    # 断言：将 Operator 转换为 Ket()，即一般算符 Operator 对应的态 Ket()
    assert operators_to_state(Operator) == Ket()

    # 断言：将 XKet 转换为其对应的操作符集合
    assert state_to_operators(operators_to_state(XOp("Q"))) == XOp("Q")
    # 断言：将 XKet() 转换为其对应的操作符集合
    assert state_to_operators(operators_to_state(XOp())) == XOp()

    # 断言：使用 lambda 表达式检测未实现的错误，期望抛出 NotImplementedError
    raises(NotImplementedError, lambda: operators_to_state(XKet))


def test_state_to_op():
    # 断言：将 XKet 转换为 XOp()，即 XKet 对应的笛卡尔坐标系下的位置算符 XOp()
    assert state_to_operators(XKet) == XOp()
    # 断言：将 PxKet 转换为 PxOp()，即 PxKet 对应的笛卡尔坐标系下的动量算符 PxOp()
    assert state_to_operators(PxKet) == PxOp()
    # 断言：将 XBra 转换为 XOp()，即 XBra 对应的笛卡尔坐标系下的位置算符 XOp()
    assert state_to_operators(XBra) == XOp()
    # 断言：将 PxBra 转换为 PxOp()，即 PxBra 对应的笛卡尔坐标系下的动量算符 PxOp()
    assert state_to_operators(PxBra) == PxOp()
    # 断言：将 Ket 转换为 Operator()，即 Ket 对应的一般算符 Operator()
    assert state_to_operators(Ket) == Operator()
    # 断言：将 Bra 转换为 Operator()，即 Bra 对应的一般算符 Operator()

    # 断言：将 XKet("test") 转换为对应的操作符集合
    assert operators_to_state(state_to_operators(X
```