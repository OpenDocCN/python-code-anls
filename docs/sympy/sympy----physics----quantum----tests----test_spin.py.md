# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_spin.py`

```
# 导入所需的符号、函数和类
from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
    Jx, Jy, Jz, Jplus, Jminus, J2,
    JxBra, JyBra, JzBra,
    JxKet, JyKet, JzKet,
    JxKetCoupled, JyKetCoupled, JzKetCoupled,
    couple, uncouple,
    Rotation, WignerD
)

from sympy.testing.pytest import raises, slow

# 定义符号变量
j1, j2, j3, j4, m1, m2, m3, m4 = symbols('j1:5 m1:5')
j12, j13, j24, j34, j123, j134, mi, mi1, mp = symbols(
    'j12 j13 j24 j34 j123 j134 mi mi1 mp')

# 测试函数：测试自旋算符的表示
def test_represent_spin_operators():
    assert represent(Jx) == hbar*Matrix([[0, 1], [1, 0]])/2  # 表示 Jx 自旋算符
    assert represent(
        Jx, j=1) == hbar*sqrt(2)*Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/2  # 在 j=1 自旋空间中表示 Jx 自旋算符
    assert represent(Jy) == hbar*I*Matrix([[0, -1], [1, 0]])/2  # 表示 Jy 自旋算符
    assert represent(
        Jy, j=1) == hbar*I*sqrt(2)*Matrix([[0, -1, 0], [1, 0, -1], [0, 1, 0]])/2  # 在 j=1 自旋空间中表示 Jy 自旋算符
    assert represent(Jz) == hbar*Matrix([[1, 0], [0, -1]])/2  # 表示 Jz 自旋算符
    assert represent(
        Jz, j=1) == hbar*Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])  # 在 j=1 自旋空间中表示 Jz 自旋算符

# 测试函数：测试自旋态的表示
def test_represent_spin_states():
    # Jx 基下的表示
    assert represent(JxKet(S.Half, S.Half), basis=Jx) == Matrix([1, 0])  # 表示 Jx 基下的 |1/2, 1/2⟩ 自旋态
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, 1])  # 表示 Jx 基下的 |1/2, -1/2⟩ 自旋态
    assert represent(JxKet(1, 1), basis=Jx) == Matrix([1, 0, 0])  # 表示 Jx 基下的 |1, 1⟩ 自旋态
    assert represent(JxKet(1, 0), basis=Jx) == Matrix([0, 1, 0])  # 表示 Jx 基下的 |1, 0⟩ 自旋态
    assert represent(JxKet(1, -1), basis=Jx) == Matrix([0, 0, 1])  # 表示 Jx 基下的 |1, -1⟩ 自旋态
    assert represent(
        JyKet(S.Half, S.Half), basis=Jx) == Matrix([exp(-I*pi/4), 0])  # 表示 Jx 基下的 |1/2, 1/2⟩ 自旋态
    assert represent(
        JyKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, exp(I*pi/4)])  # 表示 Jx 基下的 |1/2, -1/2⟩ 自旋态
    assert represent(JyKet(1, 1), basis=Jx) == Matrix([-I, 0, 0])  # 表示 Jx 基下的 |1, 1⟩ 自旋态
    assert represent(JyKet(1, 0), basis=Jx) == Matrix([0, 1, 0])  # 表示 Jx 基下的 |1, 0⟩ 自旋态
    assert represent(JyKet(1, -1), basis=Jx) == Matrix([0, 0, I])  # 表示 Jx 基下的 |1, -1⟩ 自旋态
    assert represent(
        JzKet(S.Half, S.Half), basis=Jx) == sqrt(2)*Matrix([-1, 1])/2  # 表示 Jx 基下的 |1/2, 1/2⟩ 自旋态
    assert represent(
        JzKet(S.Half, Rational(-1, 2)), basis=Jx) == sqrt(2)*Matrix([-1, -1])/2  # 表示 Jx 基下的 |1/2, -1/2⟩ 自旋态
    assert represent(JzKet(1, 1), basis=Jx) == Matrix([1, -sqrt(2), 1])/2  # 表示 Jx 基下的 |1, 1⟩ 自旋态
    assert represent(JzKet(1, 0), basis=Jx) == sqrt(2)*Matrix([1, 0, -1])/2  # 表示 Jx 基下的 |1, 0⟩ 自旋态
    assert represent(JzKet(1, -1), basis=Jx) == Matrix([1, sqrt(2), 1])/2  # 表示 Jx 基下的 |1, -1⟩ 自旋态
    # Jy 基下的表示
    assert represent(
        JxKet(S.Half, S.Half), basis=Jy) == Matrix([exp(I*pi*Rational(-3, 4)), 0])  # 表示 Jy 基下的 |1/2, 1/2⟩ 自旋态
    # 断言语句，验证 JxKet(S.Half, Rational(-1, 2)) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, exp(I*pi*Rational(3, 4))])
    
    # 断言语句，验证 JxKet(1, 1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, 1), basis=Jy) == Matrix([I, 0, 0])
    
    # 断言语句，验证 JxKet(1, 0) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    
    # 断言语句，验证 JxKet(1, -1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, -1), basis=Jy) == Matrix([0, 0, -I])
    
    # 断言语句，验证 JyKet(S.Half, S.Half) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(S.Half, S.Half), basis=Jy) == Matrix([1, 0])
    
    # 断言语句，验证 JyKet(S.Half, Rational(-1, 2)) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, 1])
    
    # 断言语句，验证 JyKet(1, 1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, 1), basis=Jy) == Matrix([1, 0, 0])
    
    # 断言语句，验证 JyKet(1, 0) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    
    # 断言语句，验证 JyKet(1, -1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, -1), basis=Jy) == Matrix([0, 0, 1])
    
    # 断言语句，验证 JzKet(S.Half, S.Half) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(S.Half, S.Half), basis=Jy) == sqrt(2)*Matrix([-1, I])/2
    
    # 断言语句，验证 JzKet(S.Half, Rational(-1, 2)) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jy) == sqrt(2)*Matrix([I, -1])/2
    
    # 断言语句，验证 JzKet(1, 1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(1, 1), basis=Jy) == Matrix([1, -I*sqrt(2), -1])/2
    
    # 断言语句，验证 JzKet(1, 0) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(1, 0), basis=Jy) == Matrix([-sqrt(2)*I, 0, -sqrt(2)*I])/2
    
    # 断言语句，验证 JzKet(1, -1) 在 Jy 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(1, -1), basis=Jy) == Matrix([-1, -sqrt(2)*I, 1])/2
    
    # 断言语句，验证 JxKet(S.Half, S.Half) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(S.Half, S.Half), basis=Jz) == sqrt(2)*Matrix([1, 1])/2
    
    # 断言语句，验证 JxKet(S.Half, Rational(-1, 2)) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2)*Matrix([-1, 1])/2
    
    # 断言语句，验证 JxKet(1, 1) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, 1), basis=Jz) == Matrix([1, sqrt(2), 1])/2
    
    # 断言语句，验证 JxKet(1, 0) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, 0), basis=Jz) == sqrt(2)*Matrix([-1, 0, 1])/2
    
    # 断言语句，验证 JxKet(1, -1) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JxKet(1, -1), basis=Jz) == Matrix([1, -sqrt(2), 1])/2
    
    # 断言语句，验证 JyKet(S.Half, S.Half) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(S.Half, S.Half), basis=Jz) == sqrt(2)*Matrix([-1, -I])/2
    
    # 断言语句，验证 JyKet(S.Half, Rational(-1, 2)) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2)*Matrix([-I, -1])/2
    
    # 断言语句，验证 JyKet(1, 1) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, 1), basis=Jz) == Matrix([1, sqrt(2)*I, -1])/2
    
    # 断言语句，验证 JyKet(1, 0) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, 0), basis=Jz) == sqrt(2)*Matrix([I, 0, I])/2
    
    # 断言语句，验证 JyKet(1, -1) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JyKet(1, -1), basis=Jz) == Matrix([-1, sqrt(2)*I, 1])/2
    
    # 断言语句，验证 JzKet(S.Half, S.Half) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(S.Half, S.Half), basis=Jz) == Matrix([1, 0])
    
    # 断言语句，验证 JzKet(S.Half, Rational(-1, 2)) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jz) == Matrix([0, 1])
    
    # 断言语句，验证 JzKet(1, 1) 在 Jz 基础上的表示是否等于指定的矩阵
    assert represent(JzKet(1, 1), basis=Jz) == Matrix([1, 0, 0])
    
    # 断言语句，验证 JzKet(
# 定义测试函数 test_represent_uncoupled_states，用于验证表示不耦合态的函数
def test_represent_uncoupled_states():
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, S.Half) 和 JxKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, S.Half) 和 JxKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, Rational(-1, 2)) 和 JxKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, Rational(-1, 2)) 和 JxKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 0, 0, 1])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, S.Half) 和 JyKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([-I, 0, 0, 0])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, S.Half) 和 JyKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, Rational(-1, 2)) 和 JyKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, Rational(-1, 2)) 和 JyKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([0, 0, 0, I])
    # Jz 基础下的断言：表示张量积 JzKet(S.Half, S.Half) 和 JzKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    # Jz 基础下的断言：表示张量积 JzKet(S.Half, S.Half) 和 JzKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([S.Half, S.Half, Rational(-1, 2), Rational(-1, 2)])
    # Jz 基础下的断言：表示张量积 JzKet(S.Half, Rational(-1, 2)) 和 JzKet(S.Half, S.Half) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jx) == \
        Matrix([S.Half, Rational(-1, 2), S.Half, Rational(-1, 2)])
    # Jz 基础下的断言：表示张量积 JzKet(S.Half, Rational(-1, 2)) 和 JzKet(S.Half, Rational(-1, 2)) 的结果，使用 Jx 作为基础
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == \
        Matrix([S.Half, S.Half, S.Half, S.Half])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, S.Half) 和 JxKet(S.Half, S.Half) 的结果，使用 Jy 作为基础
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([I, 0, 0, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, S.Half) 和 JxKet(S.Half, Rational(-1, 2)) 的结果，使用 Jy 作为基础
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 1, 0, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, Rational(-1, 2)) 和 JxKet(S.Half, S.Half) 的结果，使用 Jy 作为基础
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    # Jx 基础下的断言：表示张量积 JxKet(S.Half, Rational(-1, 2)) 和 JxKet(S.Half, Rational(-1, 2)) 的结果，使用 Jy 作为基础
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([0, 0, 0, -I])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, S.Half) 和 JyKet(S.Half, S.Half) 的结果，使用 Jy 作为基础
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    # Jy 基础下的断言：表示张量积 JyKet(S.Half, S.Half) 和 JyKet(S.Half, Rational
    # 断言：验证张量积（TensorProduct）中两个JzKet对象的表示与给定基底Jy相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([S.Half, -I/2, -I/2, Rational(-1, 2)])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jy相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([-I/2, S.Half, Rational(-1, 2), -I/2])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jy相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jy) == \
        Matrix([-I/2, Rational(-1, 2), S.Half, -I/2])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jy相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == \
        Matrix([Rational(-1, 2), -I/2, -I/2, S.Half])
    
    # Jz基底
    # 断言：验证张量积中两个JxKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([S.Half, S.Half, S.Half, S.Half])
    
    # 断言：验证张量积中两个JxKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([Rational(-1, 2), S.Half, Rational(-1, 2), S.Half])
    
    # 断言：验证张量积中两个JxKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([Rational(-1, 2), Rational(-1, 2), S.Half, S.Half])
    
    # 断言：验证张量积中两个JxKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    
    # 断言：验证张量积中两个JyKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([S.Half, I/2, I/2, Rational(-1, 2)])
    
    # 断言：验证张量积中两个JyKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([I/2, S.Half, Rational(-1, 2), I/2])
    
    # 断言：验证张量积中两个JyKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([I/2, Rational(-1, 2), S.Half, I/2])
    
    # 断言：验证张量积中两个JyKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([Rational(-1, 2), I/2, I/2, S.Half])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([0, 1, 0, 0])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 1, 0])
    
    # 断言：验证张量积中两个JzKet对象的表示与给定基底Jz相关的矩阵表示是否正确
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == \
        Matrix([0, 0, 0, 1])
def test_represent_coupled_states():
    # Jx basis
    # 断言：使用 represent 函数计算 JxKetCoupled(0, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [1, 0, 0, 0]
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, 1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, 1, 0, 0]
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 1, 0, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, 0, 1, 0]
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, -1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, 0, 0, 1]
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 0, 1])
    # 断言：使用 represent 函数计算 JyKetCoupled(0, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [1, 0, 0, 0]
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    # 断言：使用 represent 函数计算 JyKetCoupled(1, 1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, -I, 0, 0]
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, -I, 0, 0])
    # 断言：使用 represent 函数计算 JyKetCoupled(1, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, 0, 1, 0]
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 1, 0])
    # 断言：使用 represent 函数计算 JyKetCoupled(1, -1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, 0, 0, I]
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, 0, 0, I])
    # 断言：使用 represent 函数计算 JzKetCoupled(0, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [1, 0, 0, 0]
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([1, 0, 0, 0])
    # 断言：使用 represent 函数计算 JzKetCoupled(1, 1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, S.Half, -sqrt(2)/2, S.Half]
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, S.Half, -sqrt(2)/2, S.Half])
    # 断言：使用 represent 函数计算 JzKetCoupled(1, 0, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, sqrt(2)/2, 0, -sqrt(2)/2]
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, sqrt(2)/2, 0, -sqrt(2)/2])
    # 断言：使用 represent 函数计算 JzKetCoupled(1, -1, (S.Half, S.Half)) 在 Jx 基下的表示矩阵，应为 [0, S.Half, sqrt(2)/2, S.Half]
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == \
        Matrix([0, S.Half, sqrt(2)/2, S.Half])
    # Jy basis
    # 断言：使用 represent 函数计算 JxKetCoupled(0, 0, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [1, 0, 0, 0]
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, 1, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [0, I, 0, 0]
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, I, 0, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, 0, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [0, 0, 1, 0]
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 1, 0])
    # 断言：使用 represent 函数计算 JxKetCoupled(1, -1, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [0, 0, 0, -I]
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 0, 0, -I])
    # 断言：使用 represent 函数计算 JyKetCoupled(0, 0, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [1, 0, 0, 0]
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == \
        Matrix([1, 0, 0, 0])
    # 断言：使用 represent 函数计算 JyKetCoupled(1, 1, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [0, 1, 0, 0]
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == \
        Matrix([0, 1, 0, 0])
    # 断言：使用 represent 函数计算 JyKetCoupled(1, 0, (S.Half, S.Half)) 在 Jy 基下的表示矩阵，应为 [0, 0, 1, 0]
    assert represent(Jy
    # 断言：验证 JxKetCoupled(1, 0, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, -sqrt(2)/2, 0, sqrt(2)/2])
    
    # 断言：验证 JxKetCoupled(1, -1, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, S.Half, -sqrt(2)/2, S.Half])
    
    # 断言：验证 JyKetCoupled(0, 0, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    
    # 断言：验证 JyKetCoupled(1, 1, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, S.Half, I*sqrt(2)/2, Rational(-1, 2)])
    
    # 断言：验证 JyKetCoupled(1, 0, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, I*sqrt(2)/2, 0, I*sqrt(2)/2])
    
    # 断言：验证 JyKetCoupled(1, -1, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, Rational(-1, 2), I*sqrt(2)/2, S.Half])
    
    # 断言：验证 JzKetCoupled(0, 0, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([1, 0, 0, 0])
    
    # 断言：验证 JzKetCoupled(1, 1, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 1, 0, 0])
    
    # 断言：验证 JzKetCoupled(1, 0, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 1, 0])
    
    # 断言：验证 JzKetCoupled(1, -1, (S.Half, S.Half)) 在 Jz 基础上的表示是否等于特定的矩阵
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == \
        Matrix([0, 0, 0, 1])
def test_represent_rotation():
    assert represent(Rotation(0, pi/2, 0)) == \
        Matrix(
            [[WignerD(
                S(
                    1)/2, S(
                        1)/2, S(
                            1)/2, 0, pi/2, 0), WignerD(
                                S.Half, S.Half, Rational(-1, 2), 0, pi/2, 0)],
                [WignerD(S.Half, Rational(-1, 2), S.Half, 0, pi/2, 0), WignerD(S.Half, Rational(-1, 2), Rational(-1, 2), 0, pi/2, 0)]])
    assert represent(Rotation(0, pi/2, 0), doit=True) == \
        Matrix([[sqrt(2)/2, -sqrt(2)/2],
                [sqrt(2)/2, sqrt(2)/2]])

# 测试表示旋转的函数
def test_represent_rotation():
    # 检查旋转矩阵的表示是否正确
    assert represent(Rotation(0, pi/2, 0)) == \
        Matrix(
            [[WignerD(
                S(1)/2, S(1)/2, S(1)/2, 0, pi/2, 0), WignerD(
                    S.Half, S.Half, Rational(-1, 2), 0, pi/2, 0)],
             [WignerD(S.Half, Rational(-1, 2), S.Half, 0, pi/2, 0), WignerD(S.Half, Rational(-1, 2), Rational(-1, 2), 0, pi/2, 0)]])
    # 检查进行 'doit' 操作后的旋转矩阵表示是否正确
    assert represent(Rotation(0, pi/2, 0), doit=True) == \
        Matrix([[sqrt(2)/2, -sqrt(2)/2],
                [sqrt(2)/2, sqrt(2)/2]])

# 测试重写到相同基础的函数
def test_rewrite_same():
    # 重写到相同基础
    assert JxBra(1, 1).rewrite('Jx') == JxBra(1, 1)
    assert JxBra(j, m).rewrite('Jx') == JxBra(j, m)
    assert JxKet(1, 1).rewrite('Jx') == JxKet(1, 1)
    assert JxKet(j, m).rewrite('Jx') == JxKet(j, m)

# 测试重写到 'Bra' 的函数
def test_rewrite_Bra():
    # 数值计算
    assert JxBra(1, 1).rewrite('Jy') == -I*JyBra(1, 1)
    assert JxBra(1, 0).rewrite('Jy') == JyBra(1, 0)
    assert JxBra(1, -1).rewrite('Jy') == I*JyBra(1, -1)
    assert JxBra(1, 1).rewrite('Jz') == JzBra(1, 1)/2 + JzBra(1, 0)/sqrt(2) + JzBra(1, -1)/2
    assert JxBra(1, 0).rewrite('Jz') == -sqrt(2)*JzBra(1, 1)/2 + sqrt(2)*JzBra(1, -1)/2
    assert JxBra(1, -1).rewrite('Jz') == JzBra(1, 1)/2 - JzBra(1, 0)/sqrt(2) + JzBra(1, -1)/2
    assert JyBra(1, 1).rewrite('Jx') == I*JxBra(1, 1)
    assert JyBra(1, 0).rewrite('Jx') == JxBra(1, 0)
    assert JyBra(1, -1).rewrite('Jx') == -I*JxBra(1, -1)
    assert JyBra(1, 1).rewrite('Jz') == JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, 0)/2 - JzBra(1, -1)/2
    assert JyBra(1, 0).rewrite('Jz') == -sqrt(2)*I*JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, -1)/2
    assert JyBra(1, -1).rewrite('Jz') == -JzBra(1, 1)/2 - sqrt(2)*I*JzBra(1, 0)/2 + JzBra(1, -1)/2
    assert JzBra(1, 1).rewrite('Jx') == JxBra(1, 1)/2 - sqrt(2)*JxBra(1, 0)/2 + JxBra(1, -1)/2
    assert JzBra(1, 0).rewrite('Jx') == sqrt(2)*JxBra(1, 1)/2 - sqrt(2)*JxBra(1, -1)/2
    assert JzBra(1, -1).rewrite('Jx') == JxBra(1, 1)/2 + sqrt(2)*JxBra(1, 0)/2 + JxBra(1, -1)/2
    assert JzBra(1, 1).rewrite('Jy') == JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, 0)/2 - JyBra(1, -1)/2
    assert JzBra(1, 0).rewrite('Jy') == sqrt(2)*I*JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, -1)/2
    assert JzBra(1, -1).rewrite('Jy') == -JyBra(1, 1)/2 + sqrt(2)*I*JyBra(1, 0)/2 + JyBra(1, -1)/2
    # 符号计算
    assert JxBra(j, m).rewrite('Jy') == Sum(WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyBra(j, mi), (mi, -j, j))
    assert JxBra(j, m).rewrite('Jz') == Sum(WignerD(j, mi, m, 0, pi/2, 0) * JzBra(j, mi), (mi, -j, j))
    assert JyBra(j, m).rewrite('Jx') == Sum(WignerD(j, mi, m, 0, 0, pi/2) * JxBra(j, mi), (mi, -j, j))
    assert JyBra(j, m).rewrite('Jz') == Sum(WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzBra(j, mi), (mi, -j, j))
    # 使用 JzBra 类的 rewrite 方法重写为 'Jx' 的表达式，并断言其结果与 Sum 对象相等
    assert JzBra(j, m).rewrite('Jx') == Sum(
        # 计算 Wigner-D 函数的值乘以 JxBra(j, mi)，并对 mi 在 -j 到 j 的范围内求和
        WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxBra(j, mi), (mi, -j, j))
    
    # 使用 JzBra 类的 rewrite 方法重写为 'Jy' 的表达式，并断言其结果与 Sum 对象相等
    assert JzBra(j, m).rewrite('Jy') == Sum(
        # 计算 Wigner-D 函数的值乘以 JyBra(j, mi)，并对 mi 在 -j 到 j 的范围内求和
        WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyBra(j, mi), (mi, -j, j))
    
    
    这段代码包含两个断言语句，分别测试 `JzBra(j, m)` 对象调用 `rewrite` 方法后返回的结果是否与预期的 `Sum` 对象相等。
def test_rewrite_Ket():
    # Numerical tests for rewriting spin operators
    assert JxKet(1, 1).rewrite('Jy') == I*JyKet(1, 1)  # Rewrite JxKet(1, 1) in terms of Jy
    assert JxKet(1, 0).rewrite('Jy') == JyKet(1, 0)    # JxKet(1, 0) remains unchanged under Jy rewrite
    assert JxKet(1, -1).rewrite('Jy') == -I*JyKet(1, -1)  # Rewrite JxKet(1, -1) in terms of Jy with a sign change
    assert JxKet(1, 1).rewrite('Jz') == JzKet(1, 1)/2 + JzKet(1, 0)/sqrt(2) + JzKet(1, -1)/2  # Expand JxKet(1, 1) in terms of Jz
    assert JxKet(1, 0).rewrite('Jz') == -sqrt(2)*JzKet(1, 1)/2 + sqrt(2)*JzKet(1, -1)/2  # Expand JxKet(1, 0) in terms of Jz
    assert JxKet(1, -1).rewrite('Jz') == JzKet(1, 1)/2 - JzKet(1, 0)/sqrt(2) + JzKet(1, -1)/2  # Expand JxKet(1, -1) in terms of Jz
    assert JyKet(1, 1).rewrite('Jx') == -I*JxKet(1, 1)  # Rewrite JyKet(1, 1) in terms of Jx
    assert JyKet(1, 0).rewrite('Jx') == JxKet(1, 0)    # JyKet(1, 0) remains unchanged under Jx rewrite
    assert JyKet(1, -1).rewrite('Jx') == I*JxKet(1, -1)  # Rewrite JyKet(1, -1) in terms of Jx with a sign change
    assert JyKet(1, 1).rewrite('Jz') == JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, 0)/2 - JzKet(1, -1)/2  # Expand JyKet(1, 1) in terms of Jz
    assert JyKet(1, 0).rewrite('Jz') == sqrt(2)*I*JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, -1)/2  # Expand JyKet(1, 0) in terms of Jz
    assert JyKet(1, -1).rewrite('Jz') == -JzKet(1, 1)/2 + sqrt(2)*I*JzKet(1, 0)/2 + JzKet(1, -1)/2  # Expand JyKet(1, -1) in terms of Jz
    assert JzKet(1, 1).rewrite('Jx') == JxKet(1, 1)/2 - sqrt(2)*JxKet(1, 0)/2 + JxKet(1, -1)/2  # Expand JzKet(1, 1) in terms of Jx
    assert JzKet(1, 0).rewrite('Jx') == sqrt(2)*JxKet(1, 1)/2 - sqrt(2)*JxKet(1, -1)/2  # Expand JzKet(1, 0) in terms of Jx
    assert JzKet(1, -1).rewrite('Jx') == JxKet(1, 1)/2 + sqrt(2)*JxKet(1, 0)/2 + JxKet(1, -1)/2  # Expand JzKet(1, -1) in terms of Jx
    assert JzKet(1, 1).rewrite('Jy') == JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, 0)/2 - JyKet(1, -1)/2  # Expand JzKet(1, 1) in terms of Jy
    assert JzKet(1, 0).rewrite('Jy') == -sqrt(2)*I*JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, -1)/2  # Expand JzKet(1, 0) in terms of Jy
    assert JzKet(1, -1).rewrite('Jy') == -JyKet(1, 1)/2 - sqrt(2)*I*JyKet(1, 0)/2 + JyKet(1, -1)/2  # Expand JzKet(1, -1) in terms of Jy
    # Symbolic tests for rewriting using Wigner-D functions
    assert JxKet(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyKet(j, mi), (mi, -j, j))
    assert JxKet(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, 0, pi/2, 0) * JzKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, 0, pi/2) * JxKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jz') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jx') == Sum(
        WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jy') == Sum(
        WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j, mi), (mi, -j, j))


def test_rewrite_uncoupled_state():
    # Numerical tests for rewriting tensor product states
    assert TensorProduct(JyKet(1, 1), JxKet(1, 1)).rewrite('Jx') == -I*TensorProduct(JxKet(1, 1), JxKet(1, 1))  # Rewrite tensor product of JyKet(1, 1) and JxKet(1, 1) in terms of Jx
    assert TensorProduct(JyKet(1, 0), JxKet(1, 1)).rewrite('Jx') == TensorProduct(JxKet(1, 0), JxKet(1, 1))  # Rewrite tensor product of JyKet(1, 0) and JxKet(1, 1) in terms of Jx
    assert TensorProduct(JyKet(1, -1), JxKet(1, 1)).rewrite('Jx') == I*TensorProduct(JxKet(1, -1), JxKet(1, 1))  # Rewrite tensor product of JyKet(1, -1) and JxKet(1, 1) in terms of Jx
    assert TensorProduct(JzKet(1, 1), JxKet(1, 1)).rewrite('Jx') == \
        TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 - sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, 1))/2 + TensorProduct(JxKet(1, 1), JxKet(1, 1))/2  # Expand tensor product of JzKet(1, 1) and JxKet(1, 1) in terms of Jx
    # 断言语句，验证张量积（TensorProduct）对象经过重写('Jx')操作后的结果是否与期望相等
    assert TensorProduct(JzKet(1, 0), JxKet(1, 1)).rewrite('Jx') == \
        -sqrt(2)*TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 + sqrt(
            2)*TensorProduct(JxKet(1, 1), JxKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jx')操作下的结果是否与期望相等
    assert TensorProduct(JzKet(1, -1), JxKet(1, 1)).rewrite('Jx') == \
        TensorProduct(JxKet(1, -1), JxKet(1, 1))/2 + sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, 1))/2 + TensorProduct(JxKet(1, 1), JxKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, 1), JyKet(
        1, 1)).rewrite('Jy') == I*TensorProduct(JyKet(1, 1), JyKet(1, 1))

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, 0), JyKet(
        1, 1)).rewrite('Jy') == TensorProduct(JyKet(1, 0), JyKet(1, 1))

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, -1), JyKet(
        1, 1)).rewrite('Jy') == -I*TensorProduct(JyKet(1, -1), JyKet(1, 1))

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JzKet(1, 1), JyKet(1, 1)).rewrite('Jy') == \
        -TensorProduct(JyKet(1, -1), JyKet(1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2 + TensorProduct(JyKet(1, 1), JyKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JzKet(1, 0), JyKet(1, 1)).rewrite('Jy') == \
        -sqrt(2)*I*TensorProduct(JyKet(1, -1), JyKet(
            1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 1), JyKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等
    assert TensorProduct(JzKet(1, -1), JyKet(1, 1)).rewrite('Jy') == \
        TensorProduct(JyKet(1, -1), JyKet(1, 1))/2 - sqrt(2)*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2 - TensorProduct(JyKet(1, 1), JyKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, 1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, 0), JzKet(1, 1)).rewrite('Jz') == \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(
            1, 1))/2 - sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JxKet(1, -1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 - sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JyKet(1, 1), JzKet(1, 1)).rewrite('Jz') == \
        -TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 + TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JyKet(1, 0), JzKet(1, 1)).rewrite('Jz') == \
        sqrt(2)*I*TensorProduct(JzKet(1, -1), JzKet(
            1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jz')操作下的结果是否与期望相等
    assert TensorProduct(JyKet(1, -1), JzKet(1, 1)).rewrite('Jz') == \
        TensorProduct(JzKet(1, -1), JzKet(1, 1))/2 + sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2 - TensorProduct(JzKet(1, 1), JzKet(1, 1))/2

    # 断言语句，验证张量积对象在重写('Jy')操作下的结果是否与期望相等，使用符号
    assert TensorProduct(JyKet(j1, m1), JxKet(j2, m2)).rewrite('Jy') == \
        TensorProduct(JyKet(j1, m1), Sum(
            WignerD(j2, mi, m2, pi*Rational(3, 2), 0, 0) * JyKet(j2, mi), (mi, -j2, j2)))
    # 断言：使用张量积和 JzKet(j1, m1) 重写后的表达式与张量积和 JxKet(j2, m2) 的表达式相等
    assert TensorProduct(JzKet(j1, m1), JxKet(j2, m2)).rewrite('Jz') == \
        TensorProduct(JzKet(j1, m1), Sum(
            # 计算 Wigner-D 函数和 JzKet(j2, mi) 的乘积，并对 mi 进行求和
            WignerD(j2, mi, m2, 0, pi/2, 0) * JzKet(j2, mi), (mi, -j2, j2)))
    
    # 断言：使用张量积和 JxKet(j1, m1) 重写后的表达式与张量积和 JyKet(j2, m2) 的表达式相等
    assert TensorProduct(JxKet(j1, m1), JyKet(j2, m2)).rewrite('Jx') == \
        TensorProduct(JxKet(j1, m1), Sum(
            # 计算 Wigner-D 函数和 JxKet(j2, mi) 的乘积，并对 mi 进行求和
            WignerD(j2, mi, m2, 0, 0, pi/2) * JxKet(j2, mi), (mi, -j2, j2)))
    
    # 断言：使用张量积和 JzKet(j1, m1) 重写后的表达式与张量积和 JyKet(j2, m2) 的表达式相等
    assert TensorProduct(JzKet(j1, m1), JyKet(j2, m2)).rewrite('Jz') == \
        TensorProduct(JzKet(j1, m1), Sum(
            # 计算 Wigner-D 函数和 JzKet(j2, mi) 的乘积，并对 mi 进行求和
            WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) * JzKet(j2, mi), (mi, -j2, j2)))
    
    # 断言：使用张量积和 JxKet(j1, m1) 重写后的表达式与张量积和 JzKet(j2, m2) 的表达式相等
    assert TensorProduct(JxKet(j1, m1), JzKet(j2, m2)).rewrite('Jx') == \
        TensorProduct(JxKet(j1, m1), Sum(
            # 计算 Wigner-D 函数和 JxKet(j2, mi) 的乘积，并对 mi 进行求和
            WignerD(j2, mi, m2, 0, pi*Rational(3, 2), 0) * JxKet(j2, mi), (mi, -j2, j2)))
    
    # 断言：使用张量积和 JyKet(j1, m1) 重写后的表达式与张量积和 JzKet(j2, m2) 的表达式相等
    assert TensorProduct(JyKet(j1, m1), JzKet(j2, m2)).rewrite('Jy') == \
        TensorProduct(JyKet(j1, m1), Sum(
            # 计算 Wigner-D 函数和 JyKet(j2, mi) 的乘积，并对 mi 进行求和
            WignerD(j2, mi, m2, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j2, mi), (mi, -j2, j2)))
def test_rewrite_coupled_state():
    # 测试函数：test_rewrite_coupled_state()

    # 数值断言，检查 JyKetCoupled(0, 0, (S.Half, S.Half)) 重写为 JxKetCoupled(0, 0, (S.Half, S.Half))
    assert JyKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(0, 0, (S.Half, S.Half))

    # 数值断言，检查 JyKetCoupled(1, 1, (S.Half, S.Half)) 重写为 -I*JxKetCoupled(1, 1, (S.Half, S.Half))
    assert JyKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jx') == \
        -I*JxKetCoupled(1, 1, (S.Half, S.Half))

    # 数值断言，检查 JyKetCoupled(1, 0, (S.Half, S.Half)) 重写为 JxKetCoupled(1, 0, (S.Half, S.Half))
    assert JyKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 0, (S.Half, S.Half))

    # 数值断言，检查 JyKetCoupled(1, -1, (S.Half, S.Half)) 重写为 I*JxKetCoupled(1, -1, (S.Half, S.Half))
    assert JyKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jx') == \
        I*JxKetCoupled(1, -1, (S.Half, S.Half))

    # 数值断言，检查 JzKetCoupled(0, 0, (S.Half, S.Half)) 重写为 JxKetCoupled(0, 0, (S.Half, S.Half))
    assert JzKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(0, 0, (S.Half, S.Half))

    # 数值断言，检查 JzKetCoupled(1, 1, (S.Half, S.Half)) 重写为 JxKetCoupled(1, 1, (S.Half, S.Half))/2 - sqrt(2)*JxKetCoupled(1, 0, (S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 1, (S.Half, S.Half))/2 - sqrt(2)*JxKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2

    # 数值断言，检查 JzKetCoupled(1, 0, (S.Half, S.Half)) 重写为 sqrt(2)*JxKetCoupled(1, 1, (S(1)/2, S.Half))/2 - sqrt(2)*JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jx') == \
        sqrt(2)*JxKetCoupled(1, 1, (S(1)/2, S.Half))/2 - sqrt(2)*JxKetCoupled(1, -1, (S.Half, S.Half))/2

    # 数值断言，检查 JzKetCoupled(1, -1, (S.Half, S.Half)) 重写为 JxKetCoupled(1, 1, (S.Half, S.Half))/2 + sqrt(2)*JxKetCoupled(1, 0, (S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jx') == \
        JxKetCoupled(1, 1, (S.Half, S.Half))/2 + sqrt(2)*JxKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JxKetCoupled(1, -1, (S.Half, S.Half))/2

    # 数值断言，检查 JxKetCoupled(0, 0, (S.Half, S.Half)) 重写为 JyKetCoupled(0, 0, (S.Half, S.Half))
    assert JxKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(0, 0, (S.Half, S.Half))

    # 数值断言，检查 JxKetCoupled(1, 1, (S.Half, S.Half)) 重写为 I*JyKetCoupled(1, 1, (S.Half, S.Half))
    assert JxKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jy') == \
        I*JyKetCoupled(1, 1, (S.Half, S.Half))

    # 数值断言，检查 JxKetCoupled(1, 0, (S.Half, S.Half)) 重写为 JyKetCoupled(1, 0, (S.Half, S.Half))
    assert JxKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(1, 0, (S.Half, S.Half))

    # 数值断言，检查 JxKetCoupled(1, -1, (S.Half, S.Half)) 重写为 -I*JyKetCoupled(1, -1, (S.Half, S.Half))
    assert JxKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jy') == \
        -I*JyKetCoupled(1, -1, (S.Half, S.Half))

    # 数值断言，检查 JzKetCoupled(0, 0, (S.Half, S.Half)) 重写为 JyKetCoupled(0, 0, (S.Half, S.Half))
    assert JzKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(0, 0, (S.Half, S.Half))

    # 数值断言，检查 JzKetCoupled(1, 1, (S.Half, S.Half)) 重写为 JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(2)*JyKetCoupled(1, 0, (S.Half, S.Half))/2 - JyKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jy') == \
        JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(2)*JyKetCoupled(1, 0, (
            S.Half, S.Half))/2 - JyKetCoupled(1, -1, (S.Half, S.Half))/2

    # 数值断言，检查 JzKetCoupled(1, 0, (S.Half, S.Half)) 重写为 -I*sqrt(2)*JyKetCoupled(1, 1, (S.Half, S.Half))/2 - I*sqrt(2)*JyKetCoupled(1, -1, (S.Half, S.Half))/2
    assert JzKetCoupled(1, 0, (S.Half, S.Half)).rewrite('J
    # 断言：验证 JxKetCoupled 类的 rewrite 方法是否返回预期的表达式
    assert JxKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(1, 1, (S.Half, S.Half))/2 - sqrt(2)*JzKetCoupled(1, 0, (
            S.Half, S.Half))/2 + JzKetCoupled(1, -1, (S.Half, S.Half))/2
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式
    assert JyKetCoupled(0, 0, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(0, 0, (S.Half, S.Half))
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式
    assert JyKetCoupled(1, 1, (S.Half, S.Half)).rewrite('Jz') == \
        JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(2)*JzKetCoupled(1, 0, (
            S.Half, S.Half))/2 - JzKetCoupled(1, -1, (S.Half, S.Half))/2
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式
    assert JyKetCoupled(1, 0, (S.Half, S.Half)).rewrite('Jz') == \
        I*sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(
            2)*JzKetCoupled(1, -1, (S.Half, S.Half))/2
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式
    assert JyKetCoupled(1, -1, (S.Half, S.Half)).rewrite('Jz') == \
        -JzKetCoupled(1, 1, (S.Half, S.Half))/2 + I*sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2 + JzKetCoupled(1, -1, (S.Half, S.Half))/2
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JyKetCoupled(j, m, (j1, j2)).rewrite('Jx') == \
        Sum(WignerD(j, mi, m, 0, 0, pi/2) * JxKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    
    # 断言：验证 JzKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JzKetCoupled(j, m, (j1, j2)).rewrite('Jx') == \
        Sum(WignerD(j, mi, m, 0, pi*Rational(3, 2), 0) * JxKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    
    # 断言：验证 JxKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JxKetCoupled(j, m, (j1, j2)).rewrite('Jy') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), 0, 0) * JyKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    
    # 断言：验证 JzKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JzKetCoupled(j, m, (j1, j2)).rewrite('Jy') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), pi/2, pi/2) * JyKetCoupled(j,
            mi, (j1, j2)), (mi, -j, j))
    
    # 断言：验证 JxKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JxKetCoupled(j, m, (j1, j2)).rewrite('Jz') == \
        Sum(WignerD(j, mi, m, 0, pi/2, 0) * JzKetCoupled(j, mi, (
            j1, j2)), (mi, -j, j))
    
    # 断言：验证 JyKetCoupled 类的 rewrite 方法是否返回预期的表达式（符号化变量）
    assert JyKetCoupled(j, m, (j1, j2)).rewrite('Jz') == \
        Sum(WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) * JzKetCoupled(
            j, mi, (j1, j2)), (mi, -j, j))
def test_innerproducts_of_rewritten_states():
    # 数值验证内积结果是否为1
    assert qapply(JxBra(1, 1)*JxKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 0)*JxKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, -1)*JxKet(1, -1).rewrite('Jy')).doit() == 1
    assert qapply(JxBra(1, 1)*JxKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, 0)*JxKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JxBra(1, -1)*JxKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jx')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 1)*JyKet(1, 1).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, 0)*JyKet(1, 0).rewrite('Jz')).doit() == 1
    assert qapply(JyBra(1, -1)*JyKet(1, -1).rewrite('Jz')).doit() == 1
    assert qapply(JzBra(1, 1)*JzKet(1, 1).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, 0)*JzKet(1, 0).rewrite('Jy')).doit() == 1
    assert qapply(JzBra(1, -1)*JzKet(1, -1).rewrite('Jy')).doit() == 1
    # 验证内积结果是否为0
    assert qapply(JxBra(1, 1)*JxKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JxBra(1, 1)*JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 1)*JxKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JxBra(1, 1)*JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 1)*JyKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JyBra(1, 1)*JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 1)*JyKet(1, 0).rewrite('Jz')).doit() == 0
    assert qapply(JyBra(1, 1)*JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 1)*JzKet(1, 0).rewrite('Jx')).doit() == 0
    assert qapply(JzBra(1, 1)*JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 1)*JzKet(1, 0).rewrite('Jy')).doit() == 0
    assert qapply(JzBra(1, 1)*JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JxBra(1, 0)*JxKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, 1).rewrite('Jz')) == 0
    assert qapply(JyBra(1, 0)*JyKet(1, -1).rewrite('Jz')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, 1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, -1).rewrite('Jx')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JzBra(1, 0)*JzKet(1, -1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 1).rewrite('Jy')) == 0
    assert qapply(JxBra(1, -1)*JxKet(1, 0).rewrite('Jy')).doit() == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JxBra(1, -1)*JxKet(1, 1).rewrite('Jz')) == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JxBra(1, -1)*JxKet(1, 0).rewrite('Jz')).doit() == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JyBra(1, -1)*JyKet(1, 1).rewrite('Jx')) == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JyBra(1, -1)*JyKet(1, 0).rewrite('Jx')).doit() == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JyBra(1, -1)*JyKet(1, 1).rewrite('Jz')) == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JyBra(1, -1)*JyKet(1, 0).rewrite('Jz')).doit() == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JzBra(1, -1)*JzKet(1, 1).rewrite('Jx')) == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JzBra(1, -1)*JzKet(1, 0).rewrite('Jx')).doit() == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JzBra(1, -1)*JzKet(1, 1).rewrite('Jy')) == 0
    # 断言：对应的量子力学运算结果应为零
    assert qapply(JzBra(1, -1)*JzKet(1, 0).rewrite('Jy')).doit() == 0
def test_uncouple_2_coupled_states():
    # 测试函数：验证解耦后与耦合前状态张量积的展开结果是否相同
    
    # j1=1/2, j2=1/2 的情况
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    
    # j1=1/2, j2=-1/2 的情况
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    
    # j1=1/2, j2=-1/2 的情况（顺序颠倒）
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    
    # j1=1/2, j2=-1/2 的情况
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(
            TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    
    # j1=1/2, j2=1 的情况
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1)) )))
    
    # j1=1/2, j2=0 的情况
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0)) )))
    
    # j1=1/2, j2=-1 的情况
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1)) )))
    
    # j1=1/2, j2=1 的情况
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)) )))
    
    # j1=1/2, j2=0 的情况
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)) )))
    
    # j1=1/2, j2=-1 的情况
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)) == \
        expand(uncouple(
            couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)) )))
    
    # j1=1, j2=1 的情况
    assert TensorProduct(JzKet(1, 1), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, 1)) )))
    
    # j1=1, j2=0 的情况
    assert TensorProduct(JzKet(1, 1), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, 0)) )))
    
    # j1=1, j2=-1 的情况
    assert TensorProduct(JzKet(1, 1), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 1), JzKet(1, -1)) )))
    
    # j1=1, j2=1 的情况
    assert TensorProduct(JzKet(1, 0), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, 1)) )))
    
    # j1=1, j2=0 的情况
    assert TensorProduct(JzKet(1, 0), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, 0)) )))
    
    # j1=1, j2=-1 的情况
    assert TensorProduct(JzKet(1, 0), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, 0), JzKet(1, -1)) )))
    
    # j1=1, j2=1 的情况
    assert TensorProduct(JzKet(1, -1), JzKet(1, 1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, 1)) )))
    # 断言：对于张量积 JzKet(1, -1) ⊗ JzKet(1, 0)，应当与展开后的结果相等
    assert TensorProduct(JzKet(1, -1), JzKet(1, 0)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, 0)) )))
    
    # 断言：对于张量积 JzKet(1, -1) ⊗ JzKet(1, -1)，应当与展开后的结果相等
    assert TensorProduct(JzKet(1, -1), JzKet(1, -1)) == \
        expand(uncouple(couple( TensorProduct(JzKet(1, -1), JzKet(1, -1)) )))
def test_uncouple_3_coupled_states():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    # Assertion 1: Tests the result of uncoupling and then recoupling the given coupled state.

    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    # Assertion 2: Tests the result of uncoupling and then recoupling another coupled state.

    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    # Assertion 3: Tests the result of uncoupling and then recoupling yet another coupled state.

    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    # Assertion 4: Tests the result of uncoupling and then recoupling a different coupled state.

    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    # Assertion 5: Tests the result of uncoupling and then recoupling another coupled state.

    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    # Assertion 6: Tests the result of uncoupling and then recoupling yet another coupled state.

    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    # Assertion 7: Tests the result of uncoupling and then recoupling a different coupled state.

    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.NegativeOne/
               2), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) )))
    # Assertion 8: Tests the result of uncoupling and then recoupling another coupled state.

    # j1=1/2, j2=1, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    # Assertion 9: Tests the result of uncoupling and then recoupling a state with different quantum numbers.

    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    # Assertion 10: Tests the result of uncoupling and then recoupling another state with different quantum numbers.

    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    # Assertion 11: Tests the result of uncoupling and then recoupling a state with different quantum numbers.
    # 断言：张量积的耦合与解耦操作结果应该相等

    # 断言1
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    
    # 断言3
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言4
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    
    # 断言5
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言6
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    
    # 断言7
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言8
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    
    # 断言9
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言10：耦合 j1+j3=j13, j13+j2=j
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    
    # 断言11
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    
    # 断言12
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    # 断言：验证张量积的耦合和解耦结果是否与展开后的结果相等

    # 断言1
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))

    # 断言2
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))

    # 断言3
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))

    # 断言4
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))

    # 断言5
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))

    # 断言6
    # j1=1/2, j2=1, j3=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))

    # 断言7
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))

    # 断言8
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))

    # 断言9
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))

    # 断言10
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))

    # 断言11
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            1)/2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S(
            -1)/2), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) )))
    # 断言语句：验证两个张量积是否相等，这里使用了张量积对象的特定构造和操作
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.NegativeOne/
               2), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) )))
@slow
# 定义测试函数，测试对于四个耦合态的解耦过程
def test_uncouple_4_coupled_states():
    # 断言：对于所有四个自旋量子数均为1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)))))
    
    # 断言：对于三个自旋量子数为1/2和一个为-1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))))))
    
    # 断言：对于两个自旋量子数为1/2，一个为-1/2，一个为1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)))))
    
    # 断言：对于两个自旋量子数为1/2，一个为-1/2，一个为-1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))))))
    
    # 断言：对于一个自旋量子数为1/2，一个为-1/2，两个为1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)))))
    
    # 断言：对于一个自旋量子数为1/2，一个为-1/2，两个为-1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S(
            1)/2, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)))))
    
    # 断言：对于一个自旋量子数为1/2，一个为-1/2，两个为-1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))))))
    
    # 断言：对于一个自旋量子数为-1/2，三个为1/2的情况，张量积操作后解耦结果与扩展后的耦合解耦过程结果相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)))))
    # 断言：验证张量积的结果是否与扩展、解耦和耦合后的张量积结果相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) )))
    
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(1)/2, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)) )))
    
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) )))
    
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S(1)/2, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)) )))
    
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) )))
    
    # 断言：验证张量积的结果是否与扩展、解耦和耦合后的张量积结果相等，其中 j1=1/2, j2=1/2, j3=1, j4=1/2
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)) ) )))
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    
    # 检验两个张量积的等式是否成立，张量积中包含四个JzKet对象
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        # 对张量积进行解耦合并扩展
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)),
               JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) )))
    # 验证张量积表达式是否等于展开后的结果，使用断言进行验证
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) )))
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(
            S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的组合结果是否等于展开和非耦合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积运算后的结果是否等于解耦后再组合的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    
    # 断言：验证张量积的结果是否等于展开和解耦后的结果
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言语句，用于验证两个表达式是否相等
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)) == \
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    # 断言，验证两个张量积的等式是否成立
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))) == \
        # 扩展和解耦张量积的组合结果
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)) == \
        # 扩展和解耦张量积的组合结果
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))) == \
        # 扩展和解耦张量积的组合结果
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)) == \
        # 扩展和解耦张量积的组合结果
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, S.Half)), ((1, 3), (2, 4), (1, 2)) )))
    assert TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))) == \
        # 扩展和解耦张量积的组合结果
        expand(uncouple(couple( TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (2, 4), (1, 2)) )))
def test_uncouple_2_coupled_states_numerical():
    # j1=1/2, j2=1/2
    # 断言两个 JzKetCoupled(0, 0, (S.Half, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(0, 0, (S.Half, S.Half))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/2

    # 断言两个 JzKetCoupled(1, 1, (S.Half, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half))) == \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))

    # 断言两个 JzKetCoupled(1, 0, (S.Half, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/2 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/2

    # 断言两个 JzKetCoupled(1, -1, (S.Half, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))

    # j1=1, j2=1/2
    # 断言两个 JzKetCoupled(S.Half, S.Half, (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(S.Half, S.Half, (1, S.Half))) == \
        -sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))/3

    # 断言两个 JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))/3 - \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))/3

    # 断言两个 JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half))) == \
        TensorProduct(JzKet(1, 1), JzKet(S.Half, S.Half))

    # 断言两个 JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))/3

    # 断言两个 JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))) == \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))/3

    # 断言两个 JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half))) == \
        TensorProduct(JzKet(1, -1), JzKet(S.Half, Rational(-1, 2)))

    # j1=1, j2=1
    # 断言两个 JzKetCoupled(0, 0, (1, 1)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(0, 0, (1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/3 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/3

    # 断言两个 JzKetCoupled(1, 1, (1, 1)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2

    # 断言两个 JzKetCoupled(1, 0, (1, 1)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, 0, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/2

    # 断言两个 JzKetCoupled(1, -1, (1, 1)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(1, -1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 0))/2

    # 断言两个 JzKetCoupled(2, 2, (1, 1)) 解耦后的结果是否正确
    assert uncouple(JzKetCoupled(2, 2, (1, 1))) == \
        TensorProduct(JzKet(1, 1), JzKet(1, 1))
    # 第一个断言语句，验证 uncouple 函数对 JzKetCoupled(2, 1, (1, 1)) 的返回结果是否正确
    assert uncouple(JzKetCoupled(2, 1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    
    # 第二个断言语句，验证 uncouple 函数对 JzKetCoupled(2, 0, (1, 1)) 的返回结果是否正确
    assert uncouple(JzKetCoupled(2, 0, (1, 1))) == \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, 1))/6
    
    # 第三个断言语句，验证 uncouple 函数对 JzKetCoupled(2, -1, (1, 1)) 的返回结果是否正确
    assert uncouple(JzKetCoupled(2, -1, (1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2 + \
        sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, 0))/2
    
    # 第四个断言语句，验证 uncouple 函数对 JzKetCoupled(2, -2, (1, 1)) 的返回结果是否正确
    assert uncouple(JzKetCoupled(2, -2, (1, 1))) == \
        TensorProduct(JzKet(1, -1), JzKet(1, -1))
def test_uncouple_3_coupled_states_numerical():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2
    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half))) == \
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/3

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))/3 + \
        sqrt(3)*TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))/3

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half))) == \
        TensorProduct(JzKet(
            S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))

    # j1=1/2, j2=1/2, j3=1
    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1))) == \
        TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/2

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1))) == \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/6

    # 断言：解耦并返回解耦后的张量积态
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/2 + \
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))/2
    # 断言语句，用于验证 uncouple 函数返回的结果是否符合预期
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1))) == \
        # 根据给定的 JzKetCoupled 对象创建张量积，包含三个 JzKet 对象
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))

    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1))) == \
        # 计算表达式，涉及 JzKet 对象的张量积运算以及数学函数 sqrt
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/2

    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1))) == \
        # 计算表达式，包含 JzKet 对象的张量积运算以及数学函数 sqrt
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/2

    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1))) == \
        # 计算表达式，包含 JzKet 对象的张量积运算以及数学函数 sqrt
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))/2 + \
        TensorProduct(
            JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/2

    # j1=1/2, j2=1, j3=1 的特殊情况
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, 1, 1))) == \
        # 计算表达式，涉及 JzKet 对象的张量积运算
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))

    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, 1, 1))) == \
        # 计算表达式，涉及 JzKet 对象的张量积运算以及数学函数 sqrt
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/5

    assert uncouple(JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1))) == \
        # 计算表达式，涉及 JzKet 对象的张量积运算以及数学函数 sqrt
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10

    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1))) == \
        # 计算表达式，涉及 JzKet 对象的张量积运算以及数学函数 sqrt
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/5 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/5
    # 断言语句，用于验证 uncouple 函数对给定 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/5 + \
        sqrt(5)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/5
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1))) == \
        -sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 - \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, 0))/5
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1))) == \
        -4*sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 - \
        2*sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, -1))/5
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1))) == \
        -sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/5 - \
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        2*sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 - \
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        4*sqrt(5)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/15
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1))) == \
        -sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/5 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(
            JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/15
    # 断言语句，验证 uncouple 函数对另一个 JzKetCoupled 实例的返回值是否符合预期
    assert uncouple(JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1))) == \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/3 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1),
             JzKet(1, -1))/2
    # 验证第一个等式，验证结果应与sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 -
    # sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 +
    # sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 -
    # TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/3 +
    # TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/3相等。
    assert uncouple(JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1))) == \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/3 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/3
    
    # j1=1, j2=1, j3=1时的解耦合函数结果应为TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1))。
    assert uncouple(JzKetCoupled(3, 3, (1, 1, 1))) == \
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1))
    
    # j1=1, j2=1, j3=1时的解耦合函数结果应与sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/3 +
    # sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))/3 +
    # sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/3相等。
    assert uncouple(JzKetCoupled(3, 2, (1, 1, 1))) == \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/3
    
    # j1=1, j2=1, j3=1时的解耦合函数结果应与sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/15 +
    # 2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/15 +
    # 2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/15 +
    # sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/15 +
    # 2*sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/15 +
    # sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/15相等。
    assert uncouple(JzKetCoupled(3, 1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/15
    
    # j1=1, j2=1, j3=1时的解耦合函数结果应与sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/10 +
    # sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/10 +
    # sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/10 +
    # sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))/5 +
    # sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/10 +
    # sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/10 +
    # sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/10相等。
    assert uncouple(JzKetCoupled(3, 0, (1, 1, 1))) == \
        sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))/5 + \
        sqrt(10)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(10)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/10
    
    # j1=1, j2=1, j3=1时的解耦合函数结果应与sqrt(15)*Tensor
    # 断言：验证 uncouple 函数对 JzKetCoupled(2, 2, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(2, 2, (1, 1, 1))) == \
        -sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/3
    
    # 断言：验证 uncouple 函数对 JzKetCoupled(2, 1, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(2, 1, (1, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/3
    
    # 断言：验证 uncouple 函数对 JzKetCoupled(2, 0, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(2, 0, (1, 1, 1))) == \
        -TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/2 + \
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/2
    
    # 断言：验证 uncouple 函数对 JzKetCoupled(2, -1, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(2, -1, (1, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))/6 - \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/6
    
    # 断言：验证 uncouple 函数对 JzKetCoupled(2, -2, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(2, -2, (1, 1, 1))) == \
        -sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(6)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))/6
    
    # 断言：验证 uncouple 函数对 JzKetCoupled(1, 1, (1, 1, 1)) 的计算结果是否符合预期
    assert uncouple(JzKetCoupled(1, 1, (1, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/30 + \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/15 - \
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/5
    # 断言，验证 uncouple 函数对于 JzKetCoupled(1, 0, (1, 1, 1)) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(1, 0, (1, 1, 1))) == \
        # 第一项
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/10 - \
        # 第二项
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/15 + \
        # 第三项
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/10 - \
        # 第四项
        2*sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0))/15 + \
        # 第五项
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/10 - \
        # 第六项
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/15 + \
        # 第七项
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/10

    # 断言，验证 uncouple 函数对于 JzKetCoupled(1, -1, (1, 1, 1)) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(1, -1, (1, 1, 1))) == \
        # 第一项
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/5 - \
        # 第二项
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/10 + \
        # 第三项
        sqrt(15)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))/30 - \
        # 第四项
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))/10 + \
        # 第五项
        sqrt(15)*TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/15 + \
        # 第六项
        sqrt(15)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/30

    # Defined j13
    # j1=1/2, j2=1/2, j3=1, j13=1/2
    # 断言，验证 uncouple 函数对于 JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) ) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)))) == \
        # 第一项
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))/3 + \
        # 第二项
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))/3

    # 断言，验证 uncouple 函数对于 JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) ) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)))) == \
        # 第一项
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))/3 - \
        # 第二项
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))/6 + \
        # 第三项
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/6 + \
        # 第四项
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))/3

    # 断言，验证 uncouple 函数对于 JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) ) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)))) == \
        # 第一项
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))/3 + \
        # 第二项
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))/3

    # j1=1/2, j2=1, j3=1, j13=1/2
    # 断言，验证 uncouple 函数对于 JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2)))) 的返回值是否符合预期
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        # 第一项
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 + \
        # 第二项
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3
    # 断言：对于给定的 JzKetCoupled 对象，进行解耦操作并验证结果是否正确
    assert uncouple(JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        # 计算解耦结果的第一项
        -2*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/3 - \
        # 计算解耦结果的第二项
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/3 + \
        # 计算解耦结果的第三项
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        # 计算解耦结果的第四项
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        # 计算解耦结果的第一项
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/3 - \
        # 计算解耦结果的第二项
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 + \
        # 计算解耦结果的第三项
        TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/3 + \
        # 计算解耦结果的第四项
        2*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))))) == \
        # 计算解耦结果的第一项
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        # 计算解耦结果的第二项
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/3
    # 断言：j1=1, j2=1, j3=1, j13=1 的情况下的解耦结果验证
    assert uncouple(JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        # 计算解耦结果的第一项
        -sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))/2 + \
        # 计算解耦结果的第二项
        sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))/2
    assert uncouple(JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        # 计算解耦结果的第一项
        -TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/2 - \
        # 计算解耦结果的第二项
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/2 + \
        # 计算解耦结果的第三项
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/2 + \
        # 计算解耦结果的第四项
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        # 计算解耦结果的第一项
        -sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))/3 - \
        # 计算解耦结果的第二项
        sqrt(3)*TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/6 - \
        # 计算解耦结果的第三项
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/6 + \
        # 计算解耦结果的第四项
        sqrt(3)*TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/6 + \
        # 计算解耦结果的第五项
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/6 + \
        # 计算解耦结果的第六项
        sqrt(3)*TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))/3
    assert uncouple(JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        # 计算解耦结果的第一项
        -TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/2 - \
        # 计算解耦结果的第二项
        TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/2 + \
        # 计算解耦结果的第三项
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/2 + \
        # 计算解耦结果的第四项
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/2
    assert uncouple(JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)))) == \
        # 计算解耦结果的第一项
        -sqrt(2)*TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))/2 + \
        # 计算解耦结果的第二项
        sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))/2
    # 断言1：验证对给定的 JzKetCoupled 对象执行 uncouple 函数的结果
    assert uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        # 计算第一项：JzKet(1, -1), JzKet(1, 1), JzKet(1, 1) 的张量积除以2
        TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))/2 - \
        # 计算第二项：JzKet(1, 0), JzKet(1, 0), JzKet(1, 1) 的张量积除以2
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))/2 + \
        # 计算第三项：JzKet(1, 1), JzKet(1, 0), JzKet(1, 0) 的张量积除以2
        TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))/2 - \
        # 计算第四项：JzKet(1, 1), JzKet(1, 1), JzKet(1, -1) 的张量积除以2
        TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))/2
    
    # 断言2：验证对给定的 JzKetCoupled 对象执行 uncouple 函数的结果
    assert uncouple(JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        # 计算第一项：JzKet(1, -1), JzKet(1, 1), JzKet(1, 0) 的张量积除以2
        TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))/2 - \
        # 计算第二项：JzKet(1, 0), JzKet(1, -1), JzKet(1, 1) 的张量积除以2
        TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))/2 - \
        # 计算第三项：JzKet(1, 0), JzKet(1, 1), JzKet(1, -1) 的张量积除以2
        TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))/2 + \
        # 计算第四项：JzKet(1, 1), JzKet(1, -1), JzKet(1, 0) 的张量积除以2
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))/2
    
    # 断言3：验证对给定的 JzKetCoupled 对象执行 uncouple 函数的结果
    assert uncouple(JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)))) == \
        # 计算第一项：-JzKet(1, -1), JzKet(1, -1), JzKet(1, 1) 的张量积除以2
        -TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))/2 + \
        # 计算第二项：JzKet(1, -1), JzKet(1, 0), JzKet(1, 0) 的张量积除以2
        TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))/2 - \
        # 计算第三项：JzKet(1, 0), JzKet(1, 0), JzKet(1, -1) 的张量积除以2
        TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))/2 + \
        # 计算第四项：JzKet(1, 1), JzKet(1, -1), JzKet(1, -1) 的张量积除以2
        TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))/2
def test_uncouple_4_coupled_states_numerical():
    # 定义测试函数，用于验证数值解耦函数对四个耦合态的计算

    # 第一个断言：j1=1/2, j2=1/2, j3=1, j4=1，默认耦合
    assert uncouple(JzKetCoupled(3, 3, (S.Half, S.Half, 1, 1))) == \
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))

    # 第二个断言：j1=1/2, j2=1/2, j3=1, j4=1，数值解耦计算
    assert uncouple(JzKetCoupled(3, 2, (S.Half, S.Half, 1, 1))) == \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3

    # 第三个断言：j1=1/2, j2=1/2, j3=1, j4=1，数值解耦计算
    assert uncouple(JzKetCoupled(3, 1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, 1), JzKet(1, -1))/15


这段代码是一个测试函数，用于验证数值解耦函数对不同耦合态的计算结果是否正确。每个断言都包含了一个耦合态对象作为输入，并使用 `uncouple` 函数进行计算，然后与预期结果进行比较，以确保计算的准确性。
    # 断言语句，验证函数 uncouple 对 JzKetCoupled(3, 0, (S.Half, S.Half, 1, 1)) 的返回值是否等于指定的表达式
    assert uncouple(JzKetCoupled(3, 0, (S.Half, S.Half, 1, 1))) == \
        # 第一项结果计算：根号10乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 0) ⊗ JzKet(1, 1)，除以10
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        # 第二项结果计算：根号10乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 1) ⊗ JzKet(1, 0)，除以10
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 + \
        # 第三项结果计算：根号5乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, -1) ⊗ JzKet(1, 1)，除以10
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        # 第四项结果计算：根号5乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, 0) ⊗ JzKet(1, 0)，除以5
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        # 第五项结果计算：根号5乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, 1) ⊗ JzKet(1, -1)，除以10
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10 + \
        # 第六项结果计算：根号5乘以张量积 JzKet(S.Half, S.Half) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, -1) ⊗ JzKet(1, 1)，除以10
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        # 第七项结果计算：根号5乘以张量积 JzKet(S.Half, S.Half) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 0) ⊗ JzKet(1, 0)，除以5
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        # 第八项结果计算：根号5乘以张量积 JzKet(S.Half, S.Half) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 1) ⊗ JzKet(1, -1)，除以10
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        # 第九项结果计算：根号10乘以张量积 JzKet(S.Half, S.Half) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, -1) ⊗ JzKet(1, 0)，除以10
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        # 第十项结果计算：根号10乘以张量积 JzKet(S.Half, S.Half) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, 0) ⊗ JzKet(1, -1)，除以10
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/10
    
    # 断言语句，验证函数 uncouple 对 JzKetCoupled(3, -1, (S.Half, S.Half, 1, 1)) 的返回值是否等于指定的表达式
    assert uncouple(JzKetCoupled(3, -1, (S.Half, S.Half, 1, 1))) == \
        # 第一项结果计算：根号15乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, -1) ⊗ JzKet(1, 1)，除以15
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/15 + \
        # 第二项结果计算：2乘以根号15乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 0) ⊗ JzKet(1, 0)，除以15
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        # 第三项结果计算：根号15乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(1, 1) ⊗ JzKet(1, -1)，除以15
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 + \
        # 第四项结果计算：根号30乘以张量积 JzKet(S.Half, Rational(-1, 2)) ⊗ JzKet(S.Half, S.Half) ⊗ JzKet(1, -1) ⊗ JzKet(1, 0)，除以15
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.H
    # 确定表达式结果等于给定的张量积
    assert uncouple(JzKetCoupled(3, -3, (S.Half, S.Half, 1, 1))) == \
        # 计算张量积并验证结果
        TensorProduct(JzKet(S.Half, -S(1)/2), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))

    # 确定表达式结果等于给定的张量积和系数
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3

    # 确定表达式结果等于给定的张量积和系数
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/6 + \
        sqrt(3)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/3

    # 确定表达式结果等于给定的张量积和系数
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/4 + \
        TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    # 第一个断言：计算 uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1))) 的值并进行断言
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1))) == \
        # 第一项
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/3 - \
        # 第二项
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/6 + \
        # 第三项
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        # 第四项
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/12 + \
        # 第五项
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/6 - \
        # 第六项
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/12 + \
        # 第七项
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        # 第八项
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6
    
    # 第二个断言：计算 uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1))) 的值并进行断言
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1))) == \
        # 第一项
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        # 第二项
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        # 第三项
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6 + \
        # 第四项
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/6
    
    # 第三个断言：计算 uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1))) 的值并进行断言
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1))) == \
        # 第一项
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/30 + \
        # 第二项
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/30 - \
        # 第三项
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/20 + \
        # 第四项
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/30 - \
        # 第五项
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/20 + \
        # 第六项
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 - \
        # 第七项
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/10 + \
        # 第八项
        sqrt(15)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/5
    # 断言语句：验证uncouple函数对给定JzKetCoupled对象的计算结果是否正确

    # 第一个断言：计算JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1))的解耦结果
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/20 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/10

    # 第二个断言：计算JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1))的解耦结果
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/5 - \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/20 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/30

    # 第三个断言：计算JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))的解耦结果
    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        -sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/2 + \
        sqrt(2)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/2
    # 断言：验证 uncouple 函数对给定的 JzKetCoupled 对象返回的结果是否符合预期
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        # 第一项结果的解耦合
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/4 + \
        # 第二项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/4 - \
        # 第三项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/4 + \
        # 第四项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/4 - \
        # 第五项结果的解耦合
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/2 + \
        # 第六项结果的解耦合
        TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/2

    # 断言：验证 uncouple 函数对给定的 JzKetCoupled 对象返回的结果是否符合预期
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        # 第一项结果的解耦合
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/6 + \
        # 第二项结果的解耦合
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/6 - \
        # 第三项结果的解耦合
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        # 第四项结果的解耦合
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/6 - \
        # 第五项结果的解耦合
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/6 + \
        # 第六项结果的解耦合
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 - \
        # 第七项结果的解耦合
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/6 + \
        # 第八项结果的解耦合
        sqrt(3)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/6

    # 断言：验证 uncouple 函数对给定的 JzKetCoupled 对象返回的结果是否符合预期
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        # 第一项结果的解耦合
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 + \
        # 第二项结果的解耦合
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/2 - \
        # 第三项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/4 + \
        # 第四项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/4 - \
        # 第五项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/4 + \
        # 第六项结果的解耦合
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/4
    # 断言语句，用于测试 uncouple 函数的返回结果是否符合预期
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 2)))) == \
        # 第一个返回值的计算公式
        -sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/2 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/2
    
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        # 第二个返回值的计算公式
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/4 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/2
    
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        # 第三个返回值的计算公式
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/2 - \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 1), (1, 3, 1)))) == \
        # 第四个返回值的计算公式
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/2 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/4 - \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/4 + \
        sqrt(2)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/4
    
    # j1=1/2, j2=1/2, j3=1, j4=1, j12=1, j34=2
    assert uncouple(JzKetCoupled(3, 3, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        # 第五个返回值的计算公式
        TensorProduct(JzKet(
            S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))
    # 断言：验证 uncouple 函数对于给定 JzKetCoupled 对象的返回值是否与预期相等
    assert uncouple(JzKetCoupled(3, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        # 第一项结果
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/6 + \
        # 第二项结果
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/6 + \
        # 第三项结果
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/3 + \
        # 第四项结果
        sqrt(3)*TensorProduct(JzKet(S(1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/3
    
    # 断言：验证 uncouple 函数对于给定 JzKetCoupled 对象的返回值是否与预期相等
    assert uncouple(JzKetCoupled(3, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        # 第一项结果
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/15 + \
        # 第二项结果
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/15 + \
        # 第三项结果
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/15 + \
        # 第四项结果
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/15 + \
        # 第五项结果
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/15 + \
        # 第六项结果
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/15 + \
        # 第七项结果
        2*sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        # 第八项结果
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/15
    
    # 断言：验证 uncouple 函数对于给定 JzKetCoupled 对象的返回值是否与预期相等
    assert uncouple(JzKetCoupled(3, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        # 第一项结果
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        # 第二项结果
        sqrt(10)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 + \
        # 第三项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/10 + \
        # 第四项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/5 + \
        # 第五项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/10 + \
        # 第六项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/10 + \
        # 第七项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/5 + \
        # 第八项结果
        sqrt(5)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/10 + \
        # 第九项结果
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        # 第十项结果
        sqrt(10)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/10
    assert uncouple(JzKetCoupled(3, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/15 + \
        2*sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/15 + \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/15
    # 断言：验证 uncouple 函数对给定 JzKetCoupled 参数的计算结果是否等于预期结果

    assert uncouple(JzKetCoupled(3, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/6
    # 断言：验证 uncouple 函数对给定 JzKetCoupled 参数的计算结果是否等于预期结果

    assert uncouple(JzKetCoupled(3, -3, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 3)))) == \
        TensorProduct(JzKet(S.Half, -S(
            1)/2), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))
    # 断言：验证 uncouple 函数对给定 JzKetCoupled 参数的计算结果是否等于预期结果

    assert uncouple(JzKetCoupled(2, 2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))/3 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/6
    # 断言：验证 uncouple 函数对给定 JzKetCoupled 参数的计算结果是否等于预期结果
    # 断言语句：验证 JzKetCoupled 对象的解耦结果是否正确
    assert uncouple(JzKetCoupled(2, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/3 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/12 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/12 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/3 + \
        sqrt(3)*TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/6
    
    # 断言语句：验证 JzKetCoupled 对象的解耦结果是否正确
    assert uncouple(JzKetCoupled(2, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/2 - \
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/2 + \
        TensorProduct(JzKet(S(
            1)/2, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/2
    
    # 断言语句：验证 JzKetCoupled 对象的解耦结果是否正确
    assert uncouple(JzKetCoupled(2, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/6 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/3 - \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/6 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/12 + \
        sqrt(6)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/12 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half,
             S.Half), JzKet(1, -1), JzKet(1, -1))/3
    # 断言：验证 uncouple 函数对于给定的 JzKetCoupled 对象返回的值是否等于以下表达式
    
    assert uncouple(JzKetCoupled(2, -2, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 2)))) == \
        -sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/6 - \
        sqrt(6)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/6 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/3 + \
        sqrt(3)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))/3
    
    # 断言：验证 uncouple 函数对于给定的 JzKetCoupled 对象返回的值是否等于以下表达式
    
    assert uncouple(JzKetCoupled(1, 1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))/5 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/20 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/20 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/30
    
    # 断言：验证 uncouple 函数对于给定的 JzKetCoupled 对象返回的值是否等于以下表达式
    
    assert uncouple(JzKetCoupled(1, 0, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))/10 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))/15 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/30 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 - \
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/10 + \
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/10
    # 断言语句，用于检查表达式的正确性
    assert uncouple(JzKetCoupled(1, -1, (S.Half, S.Half, 1, 1), ((1, 2, 1), (3, 4, 2), (1, 3, 1)))) == \
        # 第一项的计算结果
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))/30 + \
        # 第二项的计算结果
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))/15 + \
        # 第三项的计算结果
        sqrt(15)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))/30 - \
        # 第四项的计算结果
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))/20 - \
        # 第五项的计算结果
        sqrt(30)*TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))/20 - \
        # 第六项的计算结果
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))/20 - \
        # 第七项的计算结果
        sqrt(30)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))/20 + \
        # 第八项的计算结果
        sqrt(15)*TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))/5
def test_uncouple_symbolic():
    # 测试函数 test_uncouple_symbolic，用于验证 uncouple 函数的符号处理功能

    # 第一个 assert 语句，验证对两个角动量的耦合情况
    assert uncouple(JzKetCoupled(j, m, (j1, j2) )) == \
        Sum(CG(j1, m1, j2, m2, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2)),
            (m1, -j1, j1), (m2, -j2, j2))
    
    # 第二个 assert 语句，验证对三个角动量的耦合情况
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3) )) == \
        Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    
    # 第三个 assert 语句，验证对四个角动量的耦合情况，并加入其他参数的情况
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3), ((1, 3, j13), (1, 2, j)) )) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m) *
            TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    
    # 第四个 assert 语句，验证对四个角动量的复杂耦合情况，并加入其他参数的情况
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4) )) == \
        Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j1 + j2 + j3, m1 + m2 + m3) * CG(j1 + j2 + j3, m1 + m2 + m3, j4, m4, j, m) *
            TensorProduct(
                JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))
    
    # 第五个 assert 语句，验证对四个角动量的复杂耦合情况，并加入多组其他参数的情况
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4), ((1, 3, j13), (2, 4, j24), (1, 2, j)) )) ==  \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j2, m2, j4, m4, j24, m2 + m4) * CG(j13, m1 + m3, j24, m2 + m4, j, m) *
            TensorProduct(
                JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)),
            (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))


def test_couple_2_states():
    # 测试函数 test_couple_2_states，用于验证 couple 和 expand 函数对状态耦合的影响

    # j1=1/2, j2=1/2 的耦合状态
    assert JzKetCoupled(0, 0, (S.Half, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, S.Half)) )))
    
    # j1=1, j2=1/2 的耦合状态
    assert JzKetCoupled(S.Half, S.Half, (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (1, S.Half)) )))
    # 断言：验证JzKetCoupled函数对于特定参数的输出是否符合预期
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half)) )))
    # j1=1, j2=1
    assert JzKetCoupled(0, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (1, 1)) )))
    assert JzKetCoupled(1, 1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (1, 1)) )))
    assert JzKetCoupled(1, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (1, 1)) )))
    assert JzKetCoupled(1, -1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (1, 1)) )))
    assert JzKetCoupled(2, 2, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (1, 1)) )))
    assert JzKetCoupled(2, 1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (1, 1)) )))
    assert JzKetCoupled(2, 0, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (1, 1)) )))
    assert JzKetCoupled(2, -1, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (1, 1)) )))
    assert JzKetCoupled(2, -2, (1, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (1, 1)) )))
    # j1=1/2, j2=3/2
    assert JzKetCoupled(1, 1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(1, 0, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(1, -1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 2, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, 0, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, -1, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (S.Half, Rational(3, 2))) )))
    assert JzKetCoupled(2, -2, (S.Half, Rational(3, 2))) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (S.Half, Rational(3, 2))) )))
# 定义一个测试函数 test_couple_3_states，用于验证 JzKetCoupled 函数的不同输入情况下的行为
def test_couple_3_states():
    # Default coupling
    # 验证默认情况下的耦合行为
    # j1=1/2, j2=1/2, j3=1/2
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half)) )))
    
    # 验证不同 j2 值时的耦合行为
    # j1=1/2, j2=-1/2, j3=1/2
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half)) )))
    
    # 验证不同 j1 和 j2 值时的耦合行为
    # j1=3/2, j2=3/2, j3=1/2
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half)) )))
    
    # j1=3/2, j2=1/2, j3=1/2
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half)) )))
    
    # j1=3/2, j2=-1/2, j3=1/2
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half)) )))
    
    # j1=3/2, j2=-3/2, j3=1/2
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1
    # 验证 j3=1 时不同 j1 和 j2 值的耦合行为
    assert JzKetCoupled(0, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(1, 1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(1, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(1, -1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(2, 2, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 2, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(2, 1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 1, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(2, 0, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, 0, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(2, -1, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -1, (S.Half, S.Half, 1)) )))
    
    assert JzKetCoupled(2, -2, (S.Half, S.Half, 1)) == \
        expand(couple(uncouple( JzKetCoupled(2, -2, (S.Half, S.Half, 1)) )))
    
    # Couple j1+j3=j13, j13+j2=j
    # 验证带有额外耦合关系的情况
    # j1=1/2, j2=1/2, j3=1/2, j13=0
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S(
            1)/2, S.Half), ((1, 3, 0), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S(
            1)/2, S.Half), ((1, 3, 0), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=1/2, j2=-1/2, j3=1/2, j13=0 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(S.Half, S.Half, (1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (
            1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=1/2, j2=1/2, j3=1, j13=1 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, S.Half))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=1/2, j2=-1/2, j3=1, j13=1 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(3, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=3/2, j2=3/2, j3=1, j13=3/2 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), S.Half, (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=3/2, j2=1/2, j3=1, j13=3/2 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-1, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=3/2, j2=-1/2, j3=1, j13=3/2 的情况下的耦合角动量本征态计算是否正确

    # 断言语句：验证 JzKetCoupled 函数的输出是否符合预期
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (
            1, S.Half, 1), ((1, 3, 1), (1, 2, Rational(3, 2)))) ), ((1, 3), (1, 2)) ))
    # 注释：验证 j1=3/2, j2=-3/2, j3=1, j13=3/2 的情况下的耦合角动量本征态计算是否正确
def test_couple_4_states():
    # 定义一个测试函数，用于验证四个态耦合的情况

    # 默认的耦合情况
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1/2, j4=0
    assert JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1/2, j4=-1
    assert JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1, j4=1
    assert JzKetCoupled(2, 2, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(2, 2, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1, j4=0
    assert JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(
            uncouple( JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1, j4=-1
    assert JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1, j4=-2
    assert JzKetCoupled(2, -2, (S.Half, S.Half, S.Half, S.Half)) == \
        expand(couple(uncouple(
            JzKetCoupled(2, -2, (S.Half, S.Half, S.Half, S.Half)) )))
    
    # j1=1/2, j2=1/2, j3=1/2, j4=1
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=1/2, j2=-1/2, j3=1/2, j4=1
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=3/2, j2=3/2, j3=1/2, j4=1
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=3/2, j2=1/2, j3=1/2, j4=1
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=3/2, j2=-1/2, j3=1/2, j4=1
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=3/2, j2=-3/2, j3=1/2, j4=1
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) )))
    
    # j1=5/2, j2=5/2, j3=1/2, j4=1
    assert JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S.Half, S.Half, 1)) )))
    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    assert JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1)) )))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    assert JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1)) )))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    assert JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1)) )))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    assert JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1)) )))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    assert JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S.Half, S.Half, 1)) == \
        expand(couple(uncouple(
            JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S.Half, S.Half, 1)) )))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：Coupling j1+j3=j13, j2+j4=j24, j13+j24=j，以及各参数的具体值
    assert JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：j1=1/2, j2=1/2, j3=1/2, j4=1, j13=1, j24=0
    assert JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：j1=1/2, j2=1/2, j3=1/2, j4=1, j13=1, j24=0
    assert JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 3, 1), (2, 4, 0), (1, 2, 1)) ) ), ((1, 3), (2, 4), (1, 2)) ))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：j1=1/2, j2=1/2, j3=1/2, j4=1, j13=1, j24=1/2
    assert JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) )), ((1, 3), (2, 4), (1, 2)) ))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：j1=1/2, j2=1/2, j3=1/2, j4=1, j13=1, j24=1/2
    assert JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) == \
        expand(couple(uncouple( JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, S.Half)) ) ), ((1, 3), (2, 4), (1, 2)) ))

    # 断言语句，验证 JzKetCoupled 函数的返回值与 expand(couple(uncouple(JzKetCoupled(...)))) 的结果是否相等
    # 同时注释了一组约束条件：j1=3/2, j2=3/2, j3=1/2, j4=1, j13=1, j24=3/2
    assert JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3,
    # 断言1：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言2：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言3：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) == \
        expand(couple(uncouple( JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 3, 1), (2, 4, S.Half), (1, 2, Rational(3, 2))) ) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言4：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等，包含特定的注释说明
    # j1=1/2, j2=1, j3=1/2, j4=1, j13=0, j24=1
    assert JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言5：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言6：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), ((1, 3, 0), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), (
            (1, 3, 0), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言7：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等，包含特定的注释说明
    # j1=1/2, j2=1, j3=1/2, j4=1, j13=1, j24=1
    assert JzKetCoupled(0, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 0)) ) == \
        expand(couple(uncouple( JzKetCoupled(0, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 0))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言8：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言9：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, 0, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    
    # 断言10：验证 JzKetCoupled 函数在特定参数下的计算结果是否与 expand(couple(uncouple(...))) 相等
    assert JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 1)) ) == \
        expand(couple(uncouple( JzKetCoupled(1, -1, (S.Half, 1, S.Half, 1), (
            (1, 3, 1), (2, 4, 1), (1, 2, 1))) ), ((1, 3), (2, 4), (1, 2)) ))
    # 断言：验证 JzKetCoupled 函数的返回值是否等于 expand(couple(uncouple(...)))
    assert JzKetCoupled(2, 2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))) == \
        expand(couple(uncouple(JzKetCoupled(2, 2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))))), ((1, 3), (2, 4), (1, 2)))
    
    # 断言：验证 JzKetCoupled 函数的返回值是否等于 expand(couple(uncouple(...)))
    assert JzKetCoupled(2, 1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))) == \
        expand(couple(uncouple(JzKetCoupled(2, 1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))))), ((1, 3), (2, 4), (1, 2)))
    
    # 断言：验证 JzKetCoupled 函数的返回值是否等于 expand(couple(uncouple(...)))
    assert JzKetCoupled(2, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))) == \
        expand(couple(uncouple(JzKetCoupled(2, 0, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))))), ((1, 3), (2, 4), (1, 2)))
    
    # 断言：验证 JzKetCoupled 函数的返回值是否等于 expand(couple(uncouple(...)))
    assert JzKetCoupled(2, -1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))) == \
        expand(couple(uncouple(JzKetCoupled(2, -1, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))))), ((1, 3), (2, 4), (1, 2)))
    
    # 断言：验证 JzKetCoupled 函数的返回值是否等于 expand(couple(uncouple(...)))
    assert JzKetCoupled(2, -2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))) == \
        expand(couple(uncouple(JzKetCoupled(2, -2, (S.Half, 1, S.Half, 1), ((1, 3, 1), (2, 4, 1), (1, 2, 2))))), ((1, 3), (2, 4), (1, 2)))
def test_couple_2_states_numerical():
    # j1=1/2, j2=1/2
    # 断言：耦合两个 JzKet(S.Half, S.Half) 张量积得到 JzKetCoupled(1, 1, (S.Half, S.Half))
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(1, 1, (S.Half, S.Half))

    # 断言：耦合 JzKet(S.Half, S.Half) 和 JzKet(S.Half, Rational(-1, 2)) 得到的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(0, 0, (S(1)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2

    # 断言：耦合 JzKet(S.Half, Rational(-1, 2)) 和 JzKet(S.Half, S.Half) 得到的结果
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(0, 0, (S(1)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half))/2

    # 断言：耦合两个 JzKet(S.Half, Rational(-1, 2)) 张量积得到 JzKetCoupled(1, -1, (S.Half, S.Half))
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(1, -1, (S.Half, S.Half))

    # j1=1, j2=1/2
    # 断言：耦合 JzKet(1, 1) 和 JzKet(S.Half, S.Half) 得到的结果
    assert couple(TensorProduct(JzKet(1, 1), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(Rational(3, 2), Rational(3, 2), (1, S.Half))

    # 断言：耦合 JzKet(1, 1) 和 JzKet(S.Half, Rational(-1, 2)) 得到的结果
    assert couple(TensorProduct(JzKet(1, 1), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (1, S.Half))/3 + sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))/3

    # 断言：耦合 JzKet(1, 0) 和 JzKet(S.Half, S.Half) 得到的结果
    assert couple(TensorProduct(JzKet(1, 0), JzKet(S.Half, S.Half))) == \
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (1, S.Half))/3 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (1, S.Half))/3

    # 断言：耦合 JzKet(1, 0) 和 JzKet(S.Half, Rational(-1, 2)) 得到的结果
    assert couple(TensorProduct(JzKet(1, 0), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (1, S.Half))/3 + \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))/3

    # 断言：耦合 JzKet(1, -1) 和 JzKet(S.Half, S.Half) 得到的结果
    assert couple(TensorProduct(JzKet(1, -1), JzKet(S.Half, S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (1, S(1)/2))/3 + sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (1, S.Half))/3

    # 断言：耦合 JzKet(1, -1) 和 JzKet(S.Half, Rational(-1, 2)) 得到的结果
    assert couple(TensorProduct(JzKet(1, -1), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(Rational(3, 2), Rational(-3, 2), (1, S.Half))

    # j1=1, j2=1
    # 断言：耦合 JzKet(1, 1) 和 JzKet(1, 1) 得到的结果
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(2, 2, (1, 1))

    # 断言：耦合 JzKet(1, 1) 和 JzKet(1, 0) 得到的结果
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(1, 1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, 1, (1, 1))/2

    # 断言：耦合 JzKet(1, 1) 和 JzKet(1, -1) 得到的结果
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (1, 1))/3 + sqrt(2)*JzKetCoupled(1, 0, (1, 1))/2 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/6

    # 断言：耦合 JzKet(1, 0) 和 JzKet(1, 1) 得到的结果
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(1, 1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, 1, (1, 1))/2

    # 断言：耦合 JzKet(1, 0) 和 JzKet(1, 0) 得到的结果
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0))) == \
        -sqrt(3)*JzKetCoupled(0, 0, (1, 1))/3 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/3

    # 断言：耦合 JzKet(1, 0) 和 JzKet(1, -1) 得到的结果
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(1, -1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, -1, (1, 1))/2
    # 使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (1, 1))/3 - sqrt(2)*JzKetCoupled(
            1, 0, (1, 1))/2 + sqrt(6)*JzKetCoupled(2, 0, (1, 1))/6
    # 使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(
            1, -1, (1, 1))/2 + sqrt(2)*JzKetCoupled(2, -1, (1, 1))/2
    # 使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(2, -2, (1, 1))
    # j1=3/2, j2=1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(3, 2)), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(2, 2, (Rational(3, 2), S.Half))
    # j1=3/2, j2=-1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(3, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(3)*JzKetCoupled(
            1, 1, (Rational(3, 2), S.Half))/2 + JzKetCoupled(2, 1, (Rational(3, 2), S.Half))/2
    # j1=3/2, j2=1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), S.Half), JzKet(S.Half, S.Half))) == \
        -JzKetCoupled(1, 1, (S(
            3)/2, S.Half))/2 + sqrt(3)*JzKetCoupled(2, 1, (Rational(3, 2), S.Half))/2
    # j1=3/2, j2=-1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(1, 0, (S(
            3)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(2, 0, (Rational(3, 2), S.Half))/2
    # j1=3/2, j2=-1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(1, 0, (S(
            3)/2, S.Half))/2 + sqrt(2)*JzKetCoupled(2, 0, (Rational(3, 2), S.Half))/2
    # j1=3/2, j2=-1/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(1, -1, (S(
            3)/2, S.Half))/2 + sqrt(3)*JzKetCoupled(2, -1, (Rational(3, 2), S.Half))/2
    # j1=3/2, j2=-3/2，使用 couple 函数计算两个 JzKet 对象张量积的耦合结果，并进行断言验证
    assert couple(TensorProduct(JzKet(Rational(3, 2), Rational(-3, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(3)*JzKetCoupled(1, -1, (Rational(3, 2), S.Half))/2 + \
        JzKetCoupled(2, -1, (Rational(3, 2), S.Half))/2
def test_couple_3_states_numerical():
    # 默认情况下的耦合
    # j1=1/2, j2=1/2, j3=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        JzKetCoupled(Rational(3, 2), S(3)/2, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2))))
    # 断言：对三个状态进行数值耦合的测试
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half))) / 3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/2), ((1, 2, 1), (1, 3, Rational(3, 2)))) / 3
    # 断言：对两个半整数自旋和一个负一半自旋进行数值耦合的测试
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half))) / 2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half))) / 6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/2), ((1, 2, 1), (1, 3, Rational(3, 2)))) / 3
    # 断言：对两个半整数自旋和一个负一半自旋进行数值耦合的测试
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half))) / 2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half))) / 6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One/2), ((1, 2, 1), (1, 3, Rational(3, 2)))) / 3
    # 断言：对两个半整数自旋和一个负一半自旋进行数值耦合的测试
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half))) / 2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half))) / 6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/2), ((1, 2, 1), (1, 3, Rational(3, 2)))) / 3
    # 断言：对一个负一半自旋和两个半整数自旋进行数值耦合的测试
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half))) / 3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One/2), ((1, 2, 1), (1, 3, Rational(3, 2)))) / 3
    # 断言：对给定的张量积使用偶极耦合函数进行计算，并进行比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        JzKetCoupled(Rational(3, 2), -S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2))) )
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=S.Half, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        JzKetCoupled(2, 2, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=S.Half, j3=0
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(2)*JzKetCoupled(
            2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=S.Half, j3=-1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/3 + \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/6
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=-1/2, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=-1/2, j3=0
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/3
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=S.Half, j2=-1/2, j3=-1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=-1/2, j2=S.Half, j3=1
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 - \
        JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言：对于给定的张量积，使用偶极耦合函数计算并进行比较，其中 j1=-1/2, j2=S.Half, j3=0
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/3
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 0)) )/3 - \
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/6
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(2)*JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )/2
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        JzKetCoupled(2, -2, (S.Half, S.Half, 1), ((1, 2, 1), (1, 3, 2)) )
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(
            Rational(5, 2), Rational(5, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言，验证耦合函数的计算结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 断言1：验证耦合函数对于给定的张量积JzKet对象的计算结果是否正确
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0))) == \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(10)*JzKetCoupled(S(5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    # 断言2：验证耦合函数对于给定的张量积JzKet对象的计算结果是否正确
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1))) == \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    # 断言3：验证耦合函数对于给定的张量积JzKet对象的计算结果是否正确
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1))) == \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        -2*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    
    # 断言4：验证耦合函数对于给定的张量积JzKet对象的计算结果是否正确
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0))) == \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算JzKetCoupled函数的结果，并进行标准化处理
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 使用 assert 语句来验证四个数学表达式是否相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1))) == \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    # 使用 assert 语句来验证四个数学表达式是否相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1))) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    # 使用 assert 语句来验证五个数学表达式是否相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0))) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    # 使用 assert 语句来验证五个数学表达式是否相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1))) == \
        -2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 验证表达式，断言两个张量积的耦合结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1))) == \
        # 第一项耦合结果
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 + \
        # 第二项耦合结果
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        # 第三项耦合结果
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 第四项耦合结果
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 第五项耦合结果
        sqrt(5)*JzKetCoupled(S(5)/2, S.Half, (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0))) == \
        # 第一项耦合结果
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, S.Half)) )/3 - \
        # 第二项耦合结果
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        # 第三项耦合结果
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 第四项耦合结果
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 第五项耦合结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1))) == \
        # 第一项耦合结果
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 第二项耦合结果
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 第三项耦合结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1))) == \
        # 第一项耦合结果
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, S.Half)) )/2 - \
        # 第二项耦合结果
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        # 第三项耦合结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0))) == \
        # 第一项耦合结果
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        # 第二项耦合结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(S(
            5)/2, Rational(-5, 2), (S.Half, 1, 1), ((1, 2, Rational(3, 2)), (1, 3, Rational(5, 2))) )
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1))) == \
        JzKetCoupled(3, 3, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/5 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    
    # 断言：将三个自旋态进行耦合计算，并断言结果与预期的耦合态相等
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    # 使用 assert 语句来验证 couple 函数的输出是否符合预期
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        # 第二个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        # 第三个项的系数和 JzKetCoupled 对象的组合
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 + \
        # 第四个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/3 + \
        # 第五个项的系数和 JzKetCoupled 对象的组合
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        # 第二个项的系数和 JzKetCoupled 对象的组合
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        # 第三个项的系数和 JzKetCoupled 对象的组合
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 + \
        # 第四个项的系数和 JzKetCoupled 对象的组合
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        # 第五个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        # 第六个项的系数和 JzKetCoupled 对象的组合
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 1))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        -sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        # 第二个项的系数和 JzKetCoupled 对象的组合
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        # 第三个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        -JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        # 第二个项的系数和 JzKetCoupled 对象的组合
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        # 第三个项的系数和 JzKetCoupled 对象的组合
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        # 第四个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        # 第五个项的系数和 JzKetCoupled 对象的组合
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        # 第二个项的系数和 JzKetCoupled 对象的组合
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        # 第三个项的系数和 JzKetCoupled 对象的组合
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        # 第四个项的系数和 JzKetCoupled 对象的组合
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 + \
        # 第五个项的系数和 JzKetCoupled 对象的组合
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        # 第六个项的系数和 JzKetCoupled 对象的组合
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1))) == \
        # 第一个项的系数和 JzKetCoupled 对象的组合
        -sqrt(3)*JzKetCoupled(1, 1, (1, 1,
    # 断言：计算偶极耦合算符作用于给定张量积态的结果，验证其是否与预期相等

    # 第一个断言
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1))) == \
        -sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15

    # 第二个断言
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10

    # 第三个断言
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0))) == \
        -JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15

    # 第四个断言
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1))) == \
        sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3

    # 第五个断言
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1))) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 + \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15

    # 第六个断言
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0))) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/15 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 0), (1, 3, 1)) )/3 - \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/30 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1))) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/10
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0))) == \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/10 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1))) == \
        -sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 1), (1, 3, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1))) == \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 1)) )/5 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/15
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0))) == \
        -sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )/3
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, -1))) == \
        JzKetCoupled(3, -3, (1, 1, 1), ((1, 2, 2), (1, 3, 3)) )
    
    # 确认耦合算符在给定张量积状态上的作用结果与预期值相等，其中量子数 j1=S.Half, j2=S.Half, j3=Rational(3, 2)
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        JzKetCoupled(Rational(5, 2), S(5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )
    # 断言：验证与给定耦合值相符的张量积耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half))) == \
        # 计算张量积耦合后的表达式，分为两部分相加
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5

    # 断言：验证与给定耦合值相符的张量积耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        # 计算张量积耦合后的表达式，分为三部分相加
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        2*sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/15 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10

    # 断言：验证与给定耦合值相符的张量积耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        # 计算张量积耦合后的表达式，分为三部分相加
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/2 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10

    # 断言：验证与给定耦合值相符的张量积耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        # 计算张量积耦合后的表达式，分为三部分相加
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5

    # 断言：验证与给定耦合值相符的张量积耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half))) == \
        # 计算张量积耦合后的表达式，分为四部分相加
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    # 第一个 assert 语句，验证一个函数调用的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        # 表达式中的第一项
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        # 表达式中的第二项
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        # 表达式中的第三项
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        # 表达式中的第四项
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    
    # 第二个 assert 语句，验证一个函数调用的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        # 表达式中的第一项
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        # 表达式中的第二项
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        # 表达式中的第三项
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)
             /2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    
    # 第三个 assert 语句，验证一个函数调用的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        # 表达式中的第一项
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        # 表达式中的第二项
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        # 表达式中的第三项
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/
             2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    
    # 第四个 assert 语句，验证一个函数调用的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half))) == \
        # 表达式中的第一项
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        # 表达式中的第二项
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 - \
        # 表达式中的第三项
        sqrt(30)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        # 表达式中的第四项
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(
            1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    # 第一个断言：计算张量积的耦合系数，并进行断言比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        # 计算第一项的耦合系数
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        # 计算第二项的耦合系数
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        # 计算第三项的耦合系数
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/30 + \
        # 计算第四项的耦合系数
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    
    # 第二个断言：计算张量积的耦合系数，并进行断言比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        # 计算第一项的耦合系数
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 0), (1, 3, Rational(3, 2))) )/2 + \
        # 计算第二项的耦合系数
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/10 + \
        # 计算第三项的耦合系数
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)/2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    
    # 第三个断言：计算张量积的耦合系数，并进行断言比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2)))) == \
        # 计算第一项的耦合系数
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/2 - \
        # 计算第二项的耦合系数
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        # 计算第三项的耦合系数
        sqrt(10)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    
    # 第四个断言：计算张量积的耦合系数，并进行断言比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half))) == \
        # 计算第一项的耦合系数
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, S.Half)) )/6 - \
        # 计算第二项的耦合系数
        2*sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第三项的耦合系数
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )/10
    
    # 第五个断言：计算张量积的耦合系数，并进行断言比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2)))) == \
        # 计算第一项的耦合系数
        -sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(3, 2))) )/5 + \
        # 计算第二项的耦合系数
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)/2), ((1, 2, 1), (1, 3, Rational(5, 2))) )/5
    # 确认耦合运算结果与预期结果相同
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2)))) == \
        JzKetCoupled(Rational(5, 2), -S(
            5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 2, 1), (1, 3, Rational(5, 2))) )
    # 确认 j1=1/2, j2=1/2, j3=1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(3, 2), S(
            3)/2, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2))) )
    # 确认 j1=1/2, j2=1/2, j3=-1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    # 确认 j1=1/2, j2=-1/2, j3=1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    # 确认 j1=1/2, j2=-1/2, j3=-1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 + \
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    # 确认 j1=-1/2, j2=1/2, j3=1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half)) )/2 - \
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.One/
             2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    # 确认 j1=-1/2, j2=1/2, j3=-1/2 的耦合结果
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half)) )/3 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One
             /2), ((1, 2, 1), (1, 3, Rational(3, 2))) )/3
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 3), (1, 2)) ) == \
        # 第一项计算
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 0), (1, 2, S.Half)) )/2 + \
        # 第二项计算
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        # 第三项计算
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.One/2), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3
    
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        # 计算 JzKetCoupled 的值
        JzKetCoupled(Rational(3, 2), -S(3)/2, (S.Half, S.Half, S.Half), ((1, 3, 1), (1, 2, Rational(3, 2))) )
    
    # 断言，验证耦合函数的输出是否与期望值相等
    # j1=1/2, j2=1/2, j3=1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 计算 JzKetCoupled 的值
        JzKetCoupled(2, 2, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )
    
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项计算
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        # 第二项计算
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        # 第三项计算
        sqrt(2)*JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算
        -sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/3 + \
        # 第二项计算
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        # 第三项计算
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        # 第四项计算
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/6
    
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/2 + \
        # 第二项计算
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2
    
    # 断言，验证耦合函数的输出是否与期望值相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项计算
        sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/6 + \
        # 第二项计算
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/6 + \
        # 第三项计算
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/3 + \
        # 第四项计算
        sqrt(3)*JzKetCoupled(2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/3
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/6 - \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/6 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/3 + \
        sqrt(3)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/3

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/2 + \
        JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 0)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(
            2, 0, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/6

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, S.Half), (1, 2, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 1)) )/6 + \
        sqrt(2)*JzKetCoupled(
            2, -1, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )/2

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(2, -2, (S.Half, S.Half, 1), ((1, 3, Rational(3, 2)), (1, 2, 2)) )

# 断言：验证耦合函数对给定张量积的结果是否与预期相等，用于计算量子态耦合系数


    # j 1=1/2, j 2=1, j 3=1

# 注释：定义了量子数 j1、j2 和 j3 的具体值，用于上述量子态耦合系数的计算
    # 断言，验证 couple 函数对于给定的张量积和耦合方式是否返回预期的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(
            Rational(5, 2), Rational(5, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )
    
    # 断言，验证 couple 函数对于给定的张量积和耦合方式是否返回预期的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    
    # 断言，验证 couple 函数对于给定的张量积和耦合方式是否返回预期的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -2*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    
    # 断言，验证 couple 函数对于给定的张量积和耦合方式是否返回预期的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    
    # 断言，验证 couple 函数对于给定的张量积和耦合方式是否返回预期的结果
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(S(
            5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    # 确认特定耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 计算第一组耦合项的值
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        # 计算第二组耦合项的值
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        # 计算第三组耦合项的值
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        # 计算第四组耦合项的值
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 计算第五组耦合项的值
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5

    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 计算第一组耦合项的值
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/2 + \
        # 计算第二组耦合项的值
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        # 计算第三组耦合项的值
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10

    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 计算第一组耦合项的值
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        # 计算第二组耦合项的值
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 + \
        # 计算第三组耦合项的值
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        # 计算第四组耦合项的值
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 计算第五组耦合项的值
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5

    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 计算第一组耦合项的值
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        # 计算第二组耦合项的值
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 计算第三组耦合项的值
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5

    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 计算第一组耦合项的值
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        # 计算第二组耦合项的值
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 计算第三组耦合项的值
        sqrt(5)*JzKetCoupled(S(
            5)/2, Rational(3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    # 验证第一个表达式是否成立
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        # 第二项
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        # 第三项
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        # 第四项
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 第五项
        sqrt(5)*JzKetCoupled(S(5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    
    # 验证第二个表达式是否成立
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 第一项
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/2 - \
        # 第二项
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        # 第三项
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    
    # 验证第三个表达式是否成立
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 第一项
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        # 第二项
        JzKetCoupled(S.Half, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        # 第三项
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        # 第四项
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 第五项
        sqrt(5)*JzKetCoupled(S(5)/2, S.Half, (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    
    # 验证第四个表达式是否成立
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 - \
        # 第二项
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/3 - \
        # 第三项
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 - \
        # 第四项
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        # 第五项
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    # 断言1：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    # 断言2：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/10
    # 断言3：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, S.Half), (1, 2, Rational(3, 2))) )/3 + \
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, 1, 1), ((1,
             3, Rational(3, 2)), (1, 2, Rational(5, 2))) )/5
    # 断言4：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(S(
            5)/2, Rational(-5, 2), (S.Half, 1, 1), ((1, 3, Rational(3, 2)), (1, 2, Rational(5, 2))) )
    # 断言5：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(3, 3, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )
    # 断言6：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(2)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    # 断言7：计算给定的张量积和耦合系数，验证结果是否符合预期
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    # 第一条断言：计算张量积JzKet(1, 1), JzKet(1, 0), JzKet(1, 1)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(2, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, 2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    
    # 第二条断言：计算张量积JzKet(1, 1), JzKet(1, 0), JzKet(1, 0)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    # 第三条断言：计算张量积JzKet(1, 1), JzKet(1, 0), JzKet(1, -1)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/3 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    
    # 第四条断言：计算张量积JzKet(1, 1), JzKet(1, -1), JzKet(1, 1)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/5 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    # 第五条断言：计算张量积JzKet(1, 1), JzKet(1, -1), JzKet(1, 0)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    
    # 第六条断言：计算张量积JzKet(1, 1), JzKet(1, -1), JzKet(1, -1)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct(JzKet(1, 1), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    # 第七条断言：计算张量积JzKet(1, 0), JzKet(1, 1), JzKet(1, 1)的耦合结果，与给定的耦合系数进行比较
    assert couple(TensorProduct
    # 使用 couple 函数计算两个张量积态的耦合系数，并断言其结果等于以下表达式
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项耦合系数计算
        -sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        # 第二项耦合系数计算
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 - \
        # 第三项耦合系数计算
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        # 第四项耦合系数计算
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15

    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 - \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10

    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15

    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        2*sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/5

    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 + \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15

    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 - \
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 + \
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    # 断言语句，验证调用 couple 函数后返回的结果是否符合预期
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        -sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        # 第二项计算结果
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 + \
        # 第三项计算结果
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        # 第四项计算结果
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    assert couple(TensorProduct(JzKet(1, 0), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        # 第二项计算结果
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        # 第三项计算结果
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        sqrt(3)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 + \
        # 第二项计算结果
        JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        # 第三项计算结果
        sqrt(15)*JzKetCoupled(1, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 - \
        # 第四项计算结果
        JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        # 第五项计算结果
        sqrt(3)*JzKetCoupled(2, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        # 第六项计算结果
        sqrt(15)*JzKetCoupled(3, 1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        -sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        # 第二项计算结果
        JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        # 第三项计算结果
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        # 第四项计算结果
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/6 - \
        # 第五项计算结果
        JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/2 + \
        # 第六项计算结果
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/5 - \
        # 第二项计算结果
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        # 第三项计算结果
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        # 第一项计算结果
        sqrt(6)*JzKetCoupled(0, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 0)) )/6 + \
        # 第二项计算结果
        sqrt(3)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        # 第三项计算结果
        sqrt(15)*JzKetCoupled(1, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/15 - \
        # 第四项计算结果
        sqrt(3)*JzKetCoupled(2, 0, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/3 + \
        # 第五项计算结果
        sqrt(10)*JzKetCoupled(3, 0, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/10
    # 断言语句：验证计算出的耦合系数是否符合预期值
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 - \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/10 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 - \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        2*sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, 0), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        -sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/3 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 1)), ((1, 3), (1, 2)) ) == \
        sqrt(3)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 0), (1, 2, 1)) )/3 - \
        JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 1)) )/2 + \
        sqrt(15)*JzKetCoupled(1, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 1)) )/30 - \
        JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(3)*JzKetCoupled(2, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(15)*JzKetCoupled(3, -1, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/15
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, 0)), ((1, 3), (1, 2)) ) == \
        -sqrt(2)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 1), (1, 2, 2)) )/2 + \
        sqrt(6)*JzKetCoupled(2, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 2)) )/6 + \
        sqrt(3)*JzKetCoupled(3, -2, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )/3
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(1, -1), JzKet(1, -1), JzKet(1, -1)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(3, -3, (1, 1, 1), ((1, 3, 2), (1, 2, 3)) )
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    # 验证耦合函数对给定态的计算结果，与预期的耦合系数进行比较
    # 第一个断言：计算两个JzKet张量积的耦合系数，并进行断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        # 计算各项耦合系数并求和
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    
    # 第二个断言：计算两个JzKet张量积的耦合系数，并进行断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        # 计算各项耦合系数并求和
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/2 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    
    # 第三个断言：计算三个JzKet张量积的耦合系数，并进行断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        # 计算各项耦合系数并求和
        2*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    
    # 第四个断言：计算三个JzKet张量积的耦合系数，并进行断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        # 计算各项耦合系数并求和
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/6 + \
        3*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    # 断言：验证复合算符 `couple` 的结果是否等于下面的表达式
    
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2))), ((1, 3), (1, 2)) ) == \
        # 计算第一项结果
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 + \
        # 计算第二项结果
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 + \
        # 计算第三项结果
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        # 计算第四项结果
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2))), ((1, 3), (1, 2)) ) == \
        # 计算第一项结果
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        # 计算第二项结果
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        # 计算第三项结果
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(3, 2))), ((1, 3), (1, 2)) ) == \
        # 计算第一项结果
        -sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 - \
        # 计算第二项结果
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        # 计算第三项结果
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), S.Half)), ((1, 3), (1, 2)) ) == \
        # 计算第一项结果
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        # 计算第二项结果
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 - \
        # 计算第三项结果
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        # 计算第四项结果
        sqrt(30)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    # 断言1：计算与对偶自旋算符耦合的张量积的耦合系数，验证其等于以下表达式
    assert couple(
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-1, 2))),
        ((1, 3), (1, 2))
    ) == \
        # 计算第一项的值
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        # 计算第二项的值
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/6 - \
        # 计算第三项的值
        3*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        # 计算第四项的值
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    
    # 断言2：计算与对偶自旋算符耦合的张量积的耦合系数，验证其等于以下表达式
    assert couple(
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(Rational(3, 2), Rational(-3, 2))),
        ((1, 3), (1, 2))
    ) == \
        # 计算第一项的值
        -2*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        # 计算第二项的值
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    
    # 断言3：计算与对偶自旋算符耦合的张量积的耦合系数，验证其等于以下表达式
    assert couple(
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(3, 2))),
        ((1, 3), (1, 2))
    ) == \
        # 计算第一项的值
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/2 - \
        # 计算第二项的值
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        # 计算第三项的值
        sqrt(15)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        # 计算第四项的值
        sqrt(10)*JzKetCoupled(Rational(5, 2), S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    
    # 断言4：计算与对偶自旋算符耦合的张量积的耦合系数，验证其等于以下表达式
    assert couple(
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), S.Half)),
        ((1, 3), (1, 2))
    ) == \
        # 计算第一项的值
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, S.Half)) )/6 - \
        # 计算第二项的值
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/3 + \
        # 计算第三项的值
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/5 + \
        # 计算第四项的值
        sqrt(30)*JzKetCoupled(Rational(5, 2), -S(1)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )/10
    # 断言：验证两个量子态耦合后的结果是否等于给定表达式
    assert couple(
        # 计算张量积后的耦合态，参数为三个自旋态
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-1, 2))),
        # 给定的耦合模式，((1, 3), (1, 2))
        ((1, 3), (1, 2))
        ) == \
        # 表达式右侧的项，包含三个部分的线性组合
        -JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 1), (1, 2, Rational(3, 2))) )/2 + \
        # 第二项的线性组合，乘以 sqrt(15)
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(3, 2))) )/10 + \
        # 第三项的线性组合，乘以 sqrt(15)
        sqrt(15)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S(3)/2), ((1, 3, 2), (1, 2, Rational(5, 2))) )/5
    
    # 断言：验证两个量子态耦合后的结果是否等于给定表达式
    assert couple(
        # 计算张量积后的耦合态，参数为三个自旋态
        TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(Rational(3, 2), Rational(-3, 2))),
        # 给定的耦合模式，((1, 3), (1, 2))
        ((1, 3), (1, 2))
        ) == \
        # 计算得到的耦合态，参数为 (5/2, -5/2, (1/2, 1/2, 3/2))，给定的耦合模式 ((1, 3, 2), (1, 2, 5/2))
        JzKetCoupled(Rational(5, 2), -S(5)/2, (S.Half, S.Half, Rational(3, 2)), ((1, 3, 2), (1, 2, Rational(5, 2))) )
def test_couple_4_states_numerical():
    # Default coupling
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    # 断言：对四个角动量为 1/2 的态进行耦合
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        # 期望得到的耦合态为 JzKetCoupled(2, 2, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 2)))
        JzKetCoupled(2, 2, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        # 期望得到的耦合态为 sqrt(3)*JzKetCoupled(1, 1, (1/2, 1/2, 1/2, -1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 1)) )/2 + JzKetCoupled(2, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 2)) )/2
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        # 期望得到的耦合态为 sqrt(6)*JzKetCoupled(1, 1, (1/2, 1/2, -1/2, 1/2), ((1, 2, 1), (1, 3, 1/2), (1, 4, 1)) )/3 - sqrt(3)*JzKetCoupled(1, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 1)) )/6 + JzKetCoupled(2, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 2)) )/2
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 - \
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        JzKetCoupled(2, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        # 期望得到的耦合态为 sqrt(3)*JzKetCoupled(0, 0, (1/2, 1/2, -1/2, -1/2), ((1, 2, 1), (1, 3, 1/2), (1, 4, 0)) )/3 + sqrt(3)*JzKetCoupled(1, 0, (1/2, 1/2, 1/2, -1/2), ((1, 2, 1), (1, 3, 1/2), (1, 4, 1)) )/3 + sqrt(6)*JzKetCoupled(1, 0, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 1)) )/6 + sqrt(6)*JzKetCoupled(2, 0, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 2)) )/6
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/3 + \
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 + \
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        # 期望得到的耦合态为 sqrt(2)*JzKetCoupled(1, 1, (1/2, -1/2, 1/2, 1/2), ((1, 2, 0), (1, 3, 1/2), (1, 4, 1)) )/2 - sqrt(6)*JzKetCoupled(1, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 1/2), (1, 4, 1)) )/6 - sqrt(3)*JzKetCoupled(1, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4, 1)) )/6 + JzKetCoupled(2, 1, (1/2, 1/2, 1/2, 1/2), ((1, 2, 1), (1, 3, 3/2), (1, 4
    # 断言语句：验证耦合函数 couple 的输出是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)),
                        JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        # 第一项耦合结果，返回 JzKetCoupled 对象
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)))/2 - \
        # 第二项耦合结果，乘以常数 sqrt(3)
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)))/6 + \
        # 第三项耦合结果
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)))/2 - \
        # 第四项耦合结果，乘以常数 sqrt(3)
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)))/6 + \
        # 第五项耦合结果，乘以常数 sqrt(6)
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)))/6 + \
        # 第六项耦合结果，乘以常数 sqrt(6)
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half),
                ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)))/6
    
    # 断言语句：验证耦合函数 couple 的输出是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        # 第一项耦合结果，乘以常数 -1/2
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        # 第二项耦合结果，乘以常数 -sqrt(3)/6
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 + \
        # 第三项耦合结果，乘以常数 1/2
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        # 第四项耦合结果，乘以常数 sqrt(3)/6
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        # 第五项耦合结果，乘以常数 -sqrt(6)/6
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第六项耦合结果，乘以常数 sqrt(6)/6
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    
    # 断言语句：验证耦合函数 couple 的输出是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        # 第一项耦合结果，乘以常数 sqrt(2)/2
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        # 第二项耦合结果，乘以常数 sqrt(6)/6
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        # 第三项耦合结果，乘以常数 sqrt(3)/6
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第四项耦合结果，乘以常数 1/2
        JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    # 断言：验证复合函数 `couple` 对于给定的张量积的值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        # 第一项
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 - \
        # 第二项
        sqrt(6)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        # 第三项
        sqrt(3)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第四项
        JzKetCoupled(2, 1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    
    # 断言：验证复合函数 `couple` 对于另一个给定的张量积的值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        # 第一项
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        # 第二项
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 - \
        # 第三项
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 - \
        # 第四项
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        # 第五项
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第六项
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    
    # 断言：验证复合函数 `couple` 对于另一个给定的张量积的值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        # 第一项
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 0)) )/2 - \
        # 第二项
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/6 - \
        # 第三项
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        # 第四项
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 - \
        # 第五项
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第六项
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6
    
    # 断言：验证复合函数 `couple` 对于另一个给定的张量积的值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        # 第一项
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (1, 3, S.Half), (1, 4, 1)) )/2 + \
        # 第二项
        sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/6 + \
        # 第三项
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 第四项
        JzKetCoupled(2, -1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2
    # 验证第一个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half))) == \
        # 计算第一项结果
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 0)) )/3 - \
        # 计算第二项结果
        sqrt(3)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 - \
        # 计算第三项结果
        sqrt(6)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 计算第四项结果
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/6

    # 验证第二个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))) == \
        # 计算第一项结果
        -sqrt(6)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, S.Half), (1, 4, 1)) )/3 + \
        # 计算第二项结果
        sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/6 + \
        # 计算第三项结果
        JzKetCoupled(2, -1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2

    # 验证第三个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half))) == \
        # 计算第一项结果
        -sqrt(3)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 1)) )/2 + \
        # 计算第二项结果
        JzKetCoupled(2, -1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )/2

    # 验证第四个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)))) == \
        # 计算唯一一项结果
        JzKetCoupled(2, -2, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, 2)) )

    # 验证第五个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        # 计算唯一一项结果
        JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )

    # 验证第六个等式，对四个 JzKet 进行张量积，然后对其耦合
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        # 计算第一项结果
        sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        # 计算第二项结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    # 第一个断言：计算耦合算符对张量积的作用结果是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        # 第一项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(2)/2
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/2 + \
        # 第二项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(10)/5
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        # 第三项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(10)/10
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    
    # 第二个断言：计算耦合算符对张量积的作用结果是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        # 第一项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(6)/3
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        # 第二项结果：根据特定耦合的JzKetCoupled计算，乘以系数-sqrt(30)/15
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 第三项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(5)/5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    
    # 第三个断言：计算耦合算符对张量积的作用结果是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        # 第一项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(2)/3
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        # 第二项结果：根据特定耦合的JzKetCoupled计算，乘以系数-1/3
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 + \
        # 第三项结果：根据特定耦合的JzKetCoupled计算，乘以系数2/3
        2*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        # 第四项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(5)/15
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 第五项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(5)/5
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    
    # 第四个断言：计算耦合算符对张量积的作用结果是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        # 第一项结果：根据特定耦合的JzKetCoupled计算，乘以系数2/3
        2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        # 第二项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(2)/6
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        # 第三项结果：根据特定耦合的JzKetCoupled计算，乘以系数sqrt(2)/3
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        # 第四项结果：根据特定耦合的JzKetCoupled计算，乘以系数2*sqrt(10)/15
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 第
    # 第一个断言语句，验证耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        # 计算结果的第一项，使用了 sqrt(2)*JzKetCoupled 函数
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))))/2 - \
        # 计算结果的第二项，使用了 sqrt(6)*JzKetCoupled 函数
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))))/6 - \
        # 计算结果的第三项，使用了 sqrt(30)*JzKetCoupled 函数
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))))/15 + \
        # 计算结果的第四项，使用了 sqrt(5)*JzKetCoupled 函数
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))))/5
    
    # 第二个断言语句，验证耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        # 计算结果的第一项，使用了 sqrt(6)*JzKetCoupled 函数
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)))/6 - \
        # 计算结果的第二项，使用了 sqrt(2)*JzKetCoupled 函数
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)))/6 - \
        # 计算结果的第三项，使用了 JzKetCoupled 函数
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)))/3 + \
        # 计算结果的第四项，使用了 sqrt(3)*JzKetCoupled 函数
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))))/3 - \
        # 计算结果的第五项，使用了 JzKetCoupled 函数
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))))/3 + \
        # 计算结果的第六项，使用了 sqrt(5)*JzKetCoupled 函数
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))))/15 + \
        # 计算结果的第七项，使用了 sqrt(5)*JzKetCoupled 函数
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))))/5
    # 断言，验证耦合函数计算结果是否符合预期值
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        # 第一项计算结果
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        # 第二项计算结果
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        # 第三项计算结果
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        # 第四项计算结果
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        # 第五项计算结果
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        # 第六项计算结果
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 第七项计算结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        # 第一项计算结果
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        # 第二项计算结果
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        # 第三项计算结果
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 + \
        # 第四项计算结果
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        # 第五项计算结果
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        # 第六项计算结果
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 第七项计算结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    # 断言第一个计算结果，验证与预期值的匹配
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        # 计算第一个耦合系数，结果为负平方根6的耦合态除以6
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        # 计算第二个耦合系数，结果为负平方根2的耦合态除以6
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        # 计算第三个耦合系数，结果为负平方根3的耦合态除以3
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 + \
        # 计算第四个耦合系数，结果为正平方根3的耦合态除以3
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        # 计算第五个耦合系数，结果为正平方根3的耦合态除以3
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        # 计算第六个耦合系数，结果为负平方根5的耦合态除以15
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第七个耦合系数，结果为正平方根5的耦合态除以5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5

    # 断言第二个计算结果，验证与预期值的匹配
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        # 计算第一个耦合系数，结果为正平方根2的耦合态除以2
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 + \
        # 计算第二个耦合系数，结果为正平方根6的耦合态除以6
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        # 计算第三个耦合系数，结果为正平方根30的耦合态除以15
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第四个耦合系数，结果为正平方根5的耦合态除以5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5

    # 断言第三个计算结果，验证与预期值的匹配
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        # 计算第一个耦合系数，结果为负平方根2的耦合态除以2
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 - \
        # 计算第二个耦合系数，结果为负平方根6的耦合态除以6
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        # 计算第三个耦合系数，结果为负平方根30的耦合态除以15
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第四个耦合系数，结果为正平方根5的耦合态除以5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    # 第一个断言：计算耦合函数 `couple` 的结果，断言其与给定的表达式相等
    assert couple(
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),        # 创建一个 JzKet 对象，参数为 S.Half 和 Rational(-1, 2)
            JzKet(S.Half, S.Half),                # 创建一个 JzKet 对象，参数为 S.Half 和 S.Half
            JzKet(S.Half, S.Half),                # 创建一个 JzKet 对象，参数为 S.Half 和 S.Half
            JzKet(1, 0)                           # 创建一个 JzKet 对象，参数为 1 和 0
        )
    ) == \
        -sqrt(6) * JzKetCoupled(
            S.Half, S.Half,                      # 创建一个 JzKetCoupled 对象，参数为 S.Half, S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half))
        ) / 6 - \
        sqrt(2) * JzKetCoupled(
            S.Half, S.Half,                      # 创建一个 JzKetCoupled 对象，参数为 S.Half, S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half))
        ) / 6 - \
        JzKetCoupled(
            S.Half, S.Half,                      # 创建一个 JzKetCoupled 对象，参数为 S.Half, S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half))
        ) / 3 - \
        sqrt(3) * JzKetCoupled(
            Rational(3, 2), S.Half,               # 创建一个 JzKetCoupled 对象，参数为 Rational(3, 2), S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2)))   # 另一个元组参数 ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2)))
        ) / 3 - \
        JzKetCoupled(
            Rational(3, 2), S.Half,               # 创建一个 JzKetCoupled 对象，参数为 Rational(3, 2), S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2)))   # 另一个元组参数 ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2)))
        ) / 3 + \
        sqrt(5) * JzKetCoupled(
            Rational(3, 2), S.Half,               # 创建一个 JzKetCoupled 对象，参数为 Rational(3, 2), S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2)))   # 另一个元组参数 ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2)))
        ) / 15 + \
        sqrt(5) * JzKetCoupled(
            Rational(5, 2), S.Half,               # 创建一个 JzKetCoupled 对象，参数为 Rational(5, 2), S.Half
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2)))   # 另一个元组参数 ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2)))
        ) / 5
    
    # 第二个断言：计算耦合函数 `couple` 的结果，断言其与给定的表达式相等
    assert couple(
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),        # 创建一个 JzKet 对象，参数为 S.Half 和 Rational(-1, 2)
            JzKet(S.Half, S.Half),                # 创建一个 JzKet 对象，参数为 S.Half 和 S.Half
            JzKet(S.Half, S.Half),                # 创建一个 JzKet 对象，参数为 S.Half 和 S.Half
            JzKet(1, -1)                           # 创建一个 JzKet 对象，参数为 1 和 -1
        )
    ) == \
        -sqrt(3) * JzKetCoupled(
            S.Half, Rational(-1, 2),             # 创建一个 JzKetCoupled 对象，参数为 S.Half 和 Rational(-1, 2)
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half))
        ) / 3 - \
        JzKetCoupled(
            S.Half, Rational(-1, 2),             # 创建一个 JzKetCoupled 对象，参数为 S.Half 和 Rational(-1, 2)
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half))
        ) / 3 + \
        sqrt(2) * JzKetCoupled(
            S.Half, Rational(-1, 2),             # 创建一个 JzKetCoupled 对象，参数为 S.Half 和 Rational(-1, 2)
            (S.Half, S.Half, S.Half, 1),          # 一个元组参数 (S.Half, S.Half, S.Half, 1)
            ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half))   # 另一个元组参数 ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half))
        ) / 6 - \
        sqrt(6) * JzKetCoupled(
    # 第一个断言语句，测试函数 `couple` 的输出是否与预期值相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        # 计算复杂的表达式，包括几个 JzKetCoupled 的线性组合
        sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 - \
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    
    # 第二个断言语句，测试函数 `couple` 的输出是否与预期值相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        # 计算复杂的表达式，包括几个 JzKetCoupled 的线性组合
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/6 - \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 - \
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    # 断言：验证耦合函数的返回值是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        # 计算第一项的耦合值，结果除以2
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/2 + \
        # 计算第二项的耦合值，结果除以6
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/6 + \
        # 计算第三项的耦合值，结果除以15
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第四项的耦合值，结果除以5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1))) == \
        # 计算第一项的耦合值乘以2，结果除以3
        2*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 + \
        # 计算第二项的耦合值乘以sqrt(2)，结果除以6
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/6 - \
        # 计算第三项的耦合值乘以-sqrt(2)，结果除以3
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        # 计算第四项的耦合值乘以-2*sqrt(10)，结果除以15
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第五项的耦合值乘以sqrt(10)，结果除以10
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0))) == \
        # 计算第一项的耦合值乘以sqrt(2)，结果除以3
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, S.Half)) )/3 - \
        # 计算第二项的耦合值乘以1/3，结果除以3
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/3 - \
        # 计算第三项的耦合值乘以2/3，结果除以3
        2*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 - \
        # 计算第四项的耦合值乘以-sqrt(5)，结果除以15
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        # 计算第五项的耦合值乘以sqrt(5)，结果除以5
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5
    # 断言语句，用于验证耦合函数的计算结果是否符合预期

    # 计算第一个耦合函数，将四个JzKet对象进行张量积，并对其进行耦合
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1))) == \
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, S.Half), (1, 4, Rational(3, 2))) )/3 + \
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/15 + \
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5

    # 计算第二个耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1))) == \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, S.Half)) )/2 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/10

    # 计算第三个耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0))) == \
        -sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(3, 2))) )/5 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )/5

    # 计算第四个耦合函数的结果是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1))) == \
        JzKetCoupled(Rational(5, 2), Rational(-5, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (1, 3, Rational(3, 2)), (1, 4, Rational(5, 2))) )

    # 断言语句，验证最后一个计算是否符合预期
    # 将四个S.Half的JzKet对象进行张量积，并耦合j1到j2，j3到j4
    # j1=1/2, j2=1/2, j3=1/2, j4=1/2
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        JzKetCoupled(2, 2, (S(
            1)/2, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )

    # 断言语句，验证倒数第二个计算是否符合预期
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        JzKetCoupled(2, 1, (S.Half, S(
            1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    # 断言：对给定的张量积和耦合参数，进行耦合计算并断言结果是否等于下面的表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一项：计算耦合系数为 -sqrt(2)*JzKetCoupled(...)，并除以2
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        # 第二项：计算耦合系数为 JzKetCoupled(...)，并除以2
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 第三项：计算耦合系数为 JzKetCoupled(...)，并除以2
        JzKetCoupled(2, 1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2

    # 断言：对给定的张量积和耦合参数，进行耦合计算并断言结果是否等于下面的表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一项：计算耦合系数为 sqrt(3)*JzKetCoupled(...)，并除以3
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/3 + \
        # 第二项：计算耦合系数为 sqrt(2)*JzKetCoupled(...)，并除以2
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 第三项：计算耦合系数为 sqrt(6)*JzKetCoupled(...)，并除以6
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6

    # 断言：对给定的张量积和耦合参数，进行耦合计算并断言结果是否等于下面的表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一项：计算耦合系数为 sqrt(2)*JzKetCoupled(...)，并除以2
        sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        # 第二项：计算耦合系数为 -JzKetCoupled(...)，并除以2
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 第三项：计算耦合系数为 JzKetCoupled(...)，并除以2
        JzKetCoupled(2, 1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2

    # 断言：对给定的张量积和耦合参数，进行耦合计算并断言结果是否等于下面的表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一项：计算耦合系数为 -JzKetCoupled(...)，并除以2
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        # 第二项：计算耦合系数为 sqrt(3)*JzKetCoupled(...)，并除以6
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 + \
        # 第三项：计算耦合系数为 JzKetCoupled(...)，并除以2
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 第四项：计算耦合系数为 JzKetCoupled(...)，并除以2
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        # 第五项：计算耦合系数为 sqrt(6)*JzKetCoupled(...)，并除以6
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    # 第一个断言：计算耦合的张量积并进行比较
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个项的贡献
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第二个项的贡献
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个项的贡献
        JzKetCoupled(2, -1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    
    # 第二个断言：计算耦合的张量积并进行比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个项的贡献
        -sqrt(2)*JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        # 计算第二个项的贡献
        JzKetCoupled(1, 1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个项的贡献
        JzKetCoupled(2, 1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    
    # 第三个断言：计算耦合的张量积并进行比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个项的贡献
        -JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        # 计算第二个项的贡献
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 - \
        # 计算第三个项的贡献
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第四个项的贡献
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        # 计算第五个项的贡献
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    
    # 第四个断言：计算耦合的张量积并进行比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个项的贡献
        JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 0), (1, 3, 0)) )/2 - \
        # 计算第二个项的贡献
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/6 - \
        # 计算第三个项的贡献
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 - \
        # 计算第四个项的贡献
        JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 + \
        # 计算第五个项的贡献
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6
    
    # 第五个断言：计算耦合的张量积并进行比较
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个项的贡献
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 0), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第二个项的贡献
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个项的贡献
        JzKetCoupled(2, -1, (S.Half, S(1)/2, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2
    # 断言第一个耦合系数函数的返回值是否等于指定的表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        sqrt(3)*JzKetCoupled(0, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 0)) )/3 - \
        # 计算第二个耦合项的值
        sqrt(2)*JzKetCoupled(1, 0, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个耦合项的值
        sqrt(6)*JzKetCoupled(2, 0, (S.Half, S.Half, S.Half, S.One/2), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/6

    # 断言第二个耦合系数函数的返回值是否等于指定的表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 - \
        # 计算第二个耦合项的值
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个耦合项的值
        JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2

    # 断言第三个耦合系数函数的返回值是否等于指定的表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        -sqrt(2)*JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 0), (1, 3, 1)) )/2 - \
        # 计算第二个耦合项的值
        JzKetCoupled(1, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 1)) )/2 + \
        # 计算第三个耦合项的值
        JzKetCoupled(2, -1, (S.Half, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )/2

    # 断言第四个耦合系数函数的返回值是否等于指定的表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        JzKetCoupled(2, -2, (S(1)/2, S.Half, S.Half, S.Half), ((1, 2, 1), (3, 4, 1), (1, 3, 2)) )

    # 断言第五个耦合系数函数的返回值是否等于指定的表达式
    # j1=S.Half, S.Half, S.Half, 1
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        JzKetCoupled(Rational(5, 2), Rational(5, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )

    # 断言第六个耦合系数函数的返回值是否等于指定的表达式
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合项的值
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第二个耦合项的值
        2*sqrt(15)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第三个耦合项的值
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 断言：验证通过耦合函数计算得到的值是否等于预期值
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项耦合系数
        2*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算第二项耦合系数
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        # 计算第三项耦合系数
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第四项耦合系数
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第五项耦合系数
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项耦合系数
        -sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第二项耦合系数
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第三项耦合系数
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项耦合系数
        -sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算第二项耦合系数
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        # 计算第三项耦合系数
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第四项耦合系数
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第五项耦合系数
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项耦合系数
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/2 + \
        # 计算第二项耦合系数
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        # 计算第三项耦合系数
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言：计算两个张量积的耦合系数，并进行相等性断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项的耦合系数
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 - \
        # 计算第二项的耦合系数
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        # 计算第三项的耦合系数
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 断言：计算两个张量积的耦合系数，并进行相等性断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项的耦合系数
        sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 计算第二项的耦合系数
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 计算第三项的耦合系数
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        # 计算第四项的耦合系数
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第五项的耦合系数
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算第六项的耦合系数
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第七项的耦合系数
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 断言：计算两个张量积的耦合系数，并进行相等性断言
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一项的耦合系数
        sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算第二项的耦合系数
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        # 计算第三项的耦合系数
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 + \
        # 计算第四项的耦合系数
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 + \
        # 计算第五项的耦合系数
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 计算第六项的耦合系数
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        # 计算第七项的耦合系数
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言语句，验证调用 couple 函数后返回的结果是否符合预期值
    assert couple(
        # 调用 TensorProduct 函数，将四个 JzKet 实例对象作为参数，生成一个张量积
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)),
        # 第二个参数是一个元组，包含三个元组，指定了如何耦合这些 JzKet 对象
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 第一个耦合项
        -sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half))) / 3 + \
        # 第二个耦合项
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half))) / 3 - \
        # 第三个耦合项
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half))) / 6 + \
        # 第四个耦合项
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2)))) / 6 - \
        # 第五个耦合项
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2)))) / 3 - \
        # 第六个耦合项
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2)))) / 30 + \
        # 第七个耦合项
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2)))) / 10
    
    assert couple(
        # 调用 TensorProduct 函数，将四个 JzKet 实例对象作为参数，生成一个张量积
        TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)),
        # 第二个参数是一个元组，包含三个元组，指定了如何耦合这些 JzKet 对象
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 第一个耦合项
        -sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half))) / 6 - \
        # 第二个耦合项
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half))) / 6 - \
        # 第三个耦合项
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half))) / 3 + \
        # 第四个耦合项
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2)))) / 3 - \
        # 第五个耦合项
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2)))) / 3 + \
        # 第六个耦合项
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2)))) / 15 + \
        # 第七个耦合项
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2)))) / 5
    # 第一个断言：计算两个张量积的耦合系数，并验证其与给定表达式相等
    assert couple(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个项
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 + \
        # 第二个项
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        # 第三个项
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 第二个断言：计算两个张量积的耦合系数，并验证其与给定表达式相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个项
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 - \
        # 第二个项
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        # 第三个项
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(3, 2), (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 第三个断言：计算两个张量积的耦合系数，并验证其与给定表达式相等
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个项
        -sqrt(6)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 第二个项
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 第三个项
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        # 第四个项
        sqrt(3)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 + \
        # 第五个项
        JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 第六个项
        sqrt(5)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 第七个项
        sqrt(5)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S(1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    # 第一个断言：调用 couple 函数进行比较
    assert couple(
        # 使用 TensorProduct 函数创建张量积，包括四个 JzKet 对象作为参数
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),   # 第一个 JzKet 对象
            JzKet(S.Half, S.Half),            # 第二个 JzKet 对象
            JzKet(S.Half, S.Half),            # 第三个 JzKet 对象
            JzKet(1, -1)                      # 第四个 JzKet 对象
        ),
        # 指定耦合的索引对列表
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 预期的结果表达式，包含多项 JzKetCoupled 的加权和
        -sqrt(3)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 + \
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    
    # 第二个断言：调用 couple 函数进行比较
    assert couple(
        # 使用 TensorProduct 函数创建张量积，包括四个 JzKet 对象作为参数
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),   # 第一个 JzKet 对象
            JzKet(S.Half, S.Half),            # 第二个 JzKet 对象
            JzKet(S.Half, Rational(-1, 2)),   # 第三个 JzKet 对象
            JzKet(1, 1)                       # 第四个 JzKet 对象
        ),
        # 指定耦合的索引对列表
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 预期的结果表达式，包含多项 JzKetCoupled 的加权和
        sqrt(3)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 - \
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        sqrt(6)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/6 - \
        sqrt(2)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/30 + \
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言：验证三个耦合函数的返回值是否等于特定的数学表达式
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个耦合函数的结果
        sqrt(6)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 第二个耦合函数的结果
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/6 - \
        # 第三个耦合函数的结果
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 - \
        # 第四个耦合函数的结果
        sqrt(3)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/3 - \
        # 第五个耦合函数的结果
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 + \
        # 第六个耦合函数的结果
        sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 第七个耦合函数的结果
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个耦合函数的结果
        -sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 0), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/2 + \
        # 第二个耦合函数的结果
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/10 + \
        # 第三个耦合函数的结果
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5
    
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 第一个耦合函数的结果
        sqrt(2)*JzKetCoupled(S.Half, S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/2 - \
        # 第二个耦合函数的结果
        sqrt(10)*JzKetCoupled(Rational(3, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/5 + \
        # 第三个耦合函数的结果
        sqrt(10)*JzKetCoupled(Rational(5, 2), S.Half, (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言第一组耦合函数的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, 0)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合态的表达式
        -sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算第二个耦合态的表达式
        JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/3 + \
        # 计算第三个耦合态的表达式
        JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算第四个耦合态的表达式
        4*sqrt(5)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第五个耦合态的表达式
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 断言第二组耦合函数的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half), JzKet(1, -1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合态的表达式
        sqrt(6)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算第二个耦合态的表达式
        sqrt(30)*JzKetCoupled(Rational(3, 2), Rational(-3, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第三个耦合态的表达式
        sqrt(5)*JzKetCoupled(Rational(5, 2), Rational(-3, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/5

    # 断言第三组耦合函数的返回值是否等于特定表达式
    assert couple(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2)), JzKet(1, 1)), ((1, 2), (3, 4), (1, 3)) ) == \
        # 计算第一个耦合态的表达式
        2*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, S.Half)) )/3 + \
        # 计算第二个耦合态的表达式
        sqrt(2)*JzKetCoupled(S.Half, Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, S.Half)) )/6 - \
        # 计算第三个耦合态的表达式
        sqrt(2)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2))) )/3 - \
        # 计算第四个耦合态的表达式
        2*sqrt(10)*JzKetCoupled(Rational(3, 2), Rational(-1, 2), (S.Half, S.Half, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2))) )/15 + \
        # 计算第五个耦合态的表达式
        sqrt(10)*JzKetCoupled(Rational(5, 2), Rational(-1, 2), (S.Half, S(
            1)/2, S.Half, 1), ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2))) )/10
    # 断言语句，验证特定的耦合函数的计算结果是否符合预期值
    assert couple(
        # 调用 TensorProduct 函数，生成多个 JzKet 对象的张量积，并传入耦合函数
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),        # 第一个 JzKet 对象
            JzKet(S.Half, Rational(-1, 2)),        # 第二个 JzKet 对象
            JzKet(S.Half, Rational(-1, 2)),        # 第三个 JzKet 对象
            JzKet(1, 0)                            # 第四个 JzKet 对象
        ),
        # 指定耦合函数的耦合方式
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 计算结果的第一项
        -sqrt(3) * JzKetCoupled(
            Rational(3, 2),                      # 第一个角动量量子数
            Rational(-3, 2),                     # 第二个角动量量子数
            (S.Half, S.Half, S.Half, 1),          # 参与耦合的角动量量子数列表
            ((1, 2, 1), (3, 4, S.Half), (1, 3, Rational(3, 2)))  # 耦合方式的描述
        ) / 3 - \
        # 计算结果的第二项
        2 * sqrt(15) * JzKetCoupled(
            Rational(3, 2),                      # 第一个角动量量子数
            Rational(-3, 2),                     # 第二个角动量量子数
            (S.Half, S.Half, S.Half, 1),          # 参与耦合的角动量量子数列表
            ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(3, 2)))  # 耦合方式的描述
        ) / 15 + \
        # 计算结果的第三项
        sqrt(10) * JzKetCoupled(
            Rational(5, 2),                      # 第一个角动量量子数
            Rational(-3, 2),                     # 第二个角动量量子数
            (S.Half, S.Half, S.Half, 1),          # 参与耦合的角动量量子数列表
            ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2)))  # 耦合方式的描述
        ) / 5
    
    # 断言语句，验证特定的耦合函数的计算结果是否符合预期值
    assert couple(
        # 调用 TensorProduct 函数，生成多个 JzKet 对象的张量积，并传入耦合函数
        TensorProduct(
            JzKet(S.Half, Rational(-1, 2)),        # 第一个 JzKet 对象
            JzKet(S.Half, Rational(-1, 2)),        # 第二个 JzKet 对象
            JzKet(S.Half, Rational(-1, 2)),        # 第三个 JzKet 对象
            JzKet(1, -1)                           # 第四个 JzKet 对象
        ),
        # 指定耦合函数的耦合方式
        ((1, 2), (3, 4), (1, 3))
    ) == \
        # 计算结果的单项
        JzKetCoupled(
            Rational(5, 2),                      # 第一个角动量量子数
            Rational(-5, 2),                     # 第二个角动量量子数
            (S.Half, S.Half, S.Half, 1),          # 参与耦合的角动量量子数列表
            ((1, 2, 1), (3, 4, Rational(3, 2)), (1, 3, Rational(5, 2)))  # 耦合方式的描述
        )
def test_couple_symbolic():
    # 第一个断言：对于两个张量积的情况进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        Sum(CG(j1, m1, j2, m2, j, m1 + m2) * JzKetCoupled(j, m1 + m2, (
            j1, j2)), (j, m1 + m2, j1 + j2))

    # 第二个断言：对于三个张量积的情况进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3))) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j, m1 + m2 + m3) *
            JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 2, j12), (1, 3, j)) ),
            (j12, m1 + m2, j1 + j2), (j, m1 + m2 + m3, j12 + j3))

    # 第三个断言：对于三个张量积并指定耦合顺序进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)), ((1, 3), (1, 2)) ) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m1 + m2 + m3) *
            JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 3, j13), (1, 2, j)) ),
            (j13, m1 + m3, j1 + j3), (j, m1 + m2 + m3, j13 + j2))

    # 第四个断言：对于四个张量积的情况进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4))) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j123, m1 + m2 + m3) * CG(j123, m1 + m2 + m3, j4, m4, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 2, j12), (1, 3, j123), (1, 4, j)) ),
            (j12, m1 + m2, j1 + j2), (j123, m1 + m2 + m3, j12 + j3), (j, m1 + m2 + m3 + m4, j123 + j4))

    # 第五个断言：对于四个张量积并指定耦合顺序进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 2), (3, 4), (1, 3)) ) == \
        Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j3, m3, j4, m4, j34, m3 + m4) * CG(j12, m1 + m2, j34, m3 + m4, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 2, j12), (3, 4, j34), (1, 3, j)) ),
            (j12, m1 + m2, j1 + j2), (j34, m3 + m4, j3 + j4), (j, m1 + m2 + m3 + m4, j12 + j34))

    # 第六个断言：对于四个张量积并指定不同的耦合顺序进行耦合
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 3), (1, 4), (1, 2)) ) == \
        Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j4, m4, j134, m1 + m3 + m4) * CG(j134, m1 + m3 + m4, j2, m2, j, m1 + m2 + m3 + m4) *
            JzKetCoupled(j, m1 + m2 + m3 + m4, (
                j1, j2, j3, j4), ((1, 3, j13), (1, 4, j134), (1, 2, j)) ),
            (j13, m1 + m3, j1 + j3), (j134, m1 + m3 + m4, j13 + j4), (j, m1 + m2 + m3 + m4, j134 + j2))


def test_innerproduct():
    # 第一个断言：计算Jz Bra和Jz Ket的内积，结果应为1
    assert InnerProduct(JzBra(1, 1), JzKet(1, 1)).doit() == 1

    # 第二个断言：计算半整数角动量的Bra和Ket的内积，结果应为0
    assert InnerProduct(
        JzBra(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))).doit() == 0

    # 第三个断言：计算任意角动量的Bra和Ket的内积，结果应为1
    assert InnerProduct(JzBra(j, m), JzKet(j, m)).doit() == 1

    # 第四个断言：计算Jx Bra和Jy Ket的内积，结果应为I/sqrt(2)
    assert InnerProduct(JzBra(1, 0), JyKet(1, 1)).doit() == I/sqrt(2)

    # 第五个断言：计算Jx Bra和Jz Ket的内积，结果应为-sqrt(2)/2
    assert InnerProduct(
        JxBra(S.Half, S.Half), JzKet(S.Half, S.Half)).doit() == -sqrt(2)/2

    # 第六个断言：计算Jy Bra和Jz Ket的内积，结果应为1/2
    assert InnerProduct(JyBra(1, 1), JzKet(1, 1)).doit() == S.Half

    # 第七个断言：计算Jx Bra和Jy Ket的内积，结果应为0
    assert InnerProduct(JxBra(1, -1), JyKet(1, 1)).doit() == 0


def test_rotation_small_d():
    # 符号测试
    # j = 1/2
    # 验证旋转矩阵的特定值，确保其结果与预期的余弦函数和正弦函数值匹配
    assert Rotation.d(S.Half, S.Half, S.Half, beta).doit() == cos(beta/2)
    # 验证旋转矩阵的特定值，确保其结果与预期的余弦函数和正弦函数值匹配
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), beta).doit() == -sin(beta/2)
    # 验证旋转矩阵的特定值，确保其结果与预期的余弦函数和正弦函数值匹配
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, beta).doit() == sin(beta/2)
    # 验证旋转矩阵的特定值，确保其结果与预期的余弦函数和正弦函数值匹配
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), beta).doit() == cos(beta/2)
    
    # j = 1 时的旋转矩阵特定值验证
    assert Rotation.d(1, 1, 1, beta).doit() == (1 + cos(beta))/2
    assert Rotation.d(1, 1, 0, beta).doit() == -sin(beta)/sqrt(2)
    assert Rotation.d(1, 1, -1, beta).doit() == (1 - cos(beta))/2
    assert Rotation.d(1, 0, 1, beta).doit() == sin(beta)/sqrt(2)
    assert Rotation.d(1, 0, 0, beta).doit() == cos(beta)
    assert Rotation.d(1, 0, -1, beta).doit() == -sin(beta)/sqrt(2)
    assert Rotation.d(1, -1, 1, beta).doit() == (1 - cos(beta))/2
    assert Rotation.d(1, -1, 0, beta).doit() == sin(beta)/sqrt(2)
    assert Rotation.d(1, -1, -1, beta).doit() == (1 + cos(beta))/2
    
    # j = 3/2 时的旋转矩阵特定值验证
    assert Rotation.d(S(3)/2, Rational(3, 2), Rational(3, 2), beta).doit() == (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(3)/2, S.Half, beta).doit() == -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(3)/2, Rational(-1, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(3)/2, Rational(-3, 2), beta).doit() == (-3*sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(1)/2, Rational(3, 2), beta).doit() == sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(S(3)/2, S.Half, S.Half, beta).doit() == (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4
    assert Rotation.d(S(3)/2, S.Half, Rational(-1, 2), beta).doit() == (sin(beta/2) - 3*sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), S(1)/2, Rational(-3, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(1)/2, Rational(3, 2), beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(1)/2, S.Half, beta).doit() == (-sin(beta/2) + 3*sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(1)/2, Rational(-1, 2), beta).doit() == (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(1)/2, Rational(-3, 2), beta).doit() == -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    assert Rotation.d(S(3)/2, Rational(-3, 2), Rational(3, 2), beta).doit() == (3*sin(beta/2) - sin(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(3)/2, S.Half, beta).doit() == sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4
    assert Rotation.d(Rational(3, 2), -S(3)/2, Rational(-1, 2), beta).doit() == sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(Rational(3, 2), -S(
        3)/2, Rational(-3, 2), beta).doit() == (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 2, 2, beta).doit() == (3 + 4*cos(beta) + cos(2*beta))/8
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 2, 1, beta).doit() == -((cos(beta) + 1)*sin(beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 2, 0, beta).doit() == sqrt(6)*sin(beta)**2/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 2, -1, beta).doit() == (cos(beta) - 1)*sin(beta)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 2, -2, beta).doit() == (3 - 4*cos(beta) + cos(2*beta))/8
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 1, 2, beta).doit() == (cos(beta) + 1)*sin(beta)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 1, 1, beta).doit() == (cos(beta) + cos(2*beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 1, 0, beta).doit() == -sqrt(6)*sin(2*beta)/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 1, -1, beta).doit() == (cos(beta) - cos(2*beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 1, -2, beta).doit() == (cos(beta) - 1)*sin(beta)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 0, 2, beta).doit() == sqrt(6)*sin(beta)**2/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 0, 1, beta).doit() == sqrt(6)*sin(2*beta)/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 0, 0, beta).doit() == (1 + 3*cos(2*beta))/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 0, -1, beta).doit() == -sqrt(6)*sin(2*beta)/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, 0, -2, beta).doit() == sqrt(6)*sin(beta)**2/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -1, 2, beta).doit() == (2*sin(beta) - sin(2*beta))/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -1, 1, beta).doit() == (cos(beta) - cos(2*beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -1, 0, beta).doit() == sqrt(6)*sin(2*beta)/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -1, -1, beta).doit() == (cos(beta) + cos(2*beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -1, -2, beta).doit() == -((cos(beta) + 1)*sin(beta))/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -2, 2, beta).doit() == (3 - 4*cos(beta) + cos(2*beta))/8
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -2, 1, beta).doit() == (2*sin(beta) - sin(2*beta))/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -2, 0, beta).doit() == sqrt(6)*sin(beta)**2/4
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -2, -1, beta).doit() == (cos(beta) + 1)*sin(beta)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(2, -2, -2, beta).doit() == (3 + 4*cos(beta) + cos(2*beta))/8
    # 数值测试
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(S.Half, S.Half, S.Half, pi/2).doit() == sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(S.Half, S.Half, Rational(-1, 2), pi/2).doit() == -sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(S.Half, Rational(-1, 2), S.Half, pi/2).doit() == sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(S.Half, Rational(-1, 2), Rational(-1, 2), pi/2).doit() == sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(1, 1, 1, pi/2).doit() == S.Half
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(1, 1, 0, pi/2).doit() == -sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(1, 1, -1, pi/2).doit() == S.Half
    # 断言：对于 Rotation.d 方法调用，验证其返回值是否等于给定表达式
    assert Rotation.d(1, 0, 1, pi/2).doit() == sqrt(2)/2
    # 断言：对于 Rotation.d 方法调用，验证其返回值
    # 以下代码用于测试 Rotation 类的不同参数组合下，执行 doit() 方法的结果是否符合预期
    
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(3, 2), pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), S.Half, pi/2).doit() == -sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-1, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(3, 2), Rational(-3, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), S.Half, S.Half, pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-1, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), S.Half, Rational(-3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(3, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), S.Half, pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-1, 2), pi/2).doit() == -sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-1, 2), Rational(-3, 2), pi/2).doit() == -sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(3, 2), pi/2).doit() == sqrt(2)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), S.Half, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-1, 2), pi/2).doit() == sqrt(6)/4
    assert Rotation.d(Rational(3, 2), Rational(-3, 2), Rational(-3, 2), pi/2).doit() == sqrt(2)/4
    
    # 测试特殊情况，当参数为整数时的结果
    # j = 2
    assert Rotation.d(2, 2, 2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, 2, 1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, 0, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, 2, -1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 2, -2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, 1, 2, pi/2).doit() == S.Half
    assert Rotation.d(2, 1, 1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 1, 0, pi/2).doit() == 0
    assert Rotation.d(2, 1, -1, pi/2).doit() == S.Half
    assert Rotation.d(2, 1, -2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, 2, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, 0, 1, pi/2).doit() == 0
    assert Rotation.d(2, 0, 0, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, 0, -1, pi/2).doit() == 0
    assert Rotation.d(2, 0, -2, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, -1, 2, pi/2).doit() == S.Half
    assert Rotation.d(2, -1, 1, pi/2).doit() == S.Half
    assert Rotation.d(2, -1, 0, pi/2).doit() == 0
    assert Rotation.d(2, -1, -1, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -1, -2, pi/2).doit() == Rational(-1, 2)
    assert Rotation.d(2, -2, 2, pi/2).doit() == Rational(1, 4)
    assert Rotation.d(2, -2, 1, pi/2).doit() == S.Half
    assert Rotation.d(2, -2, 0, pi/2).doit() == sqrt(6)/4
    assert Rotation.d(2, -2, -1, pi/2).doit() == S.Half
    assert Rotation.d(2, -2, -2, pi/2).doit() == Rational(1, 4)
# 定义一个名为 test_rotation_d 的函数，用于测试旋转 D 函数的符号表达式
def test_rotation_d():
    # 符号测试 j = 1/2
    # 验证旋转 D 函数对于给定的参数返回正确的符号表达式结果
    assert Rotation.D(S.Half, S.Half, S.Half, alpha, beta, gamma).doit() == \
        cos(beta/2)*exp(-I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(S.Half, S.Half, Rational(-1, 2), alpha, beta, gamma).doit() == \
        -sin(beta/2)*exp(-I*alpha/2)*exp(I*gamma/2)
    assert Rotation.D(S.Half, Rational(-1, 2), S.Half, alpha, beta, gamma).doit() == \
        sin(beta/2)*exp(I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(S.Half, Rational(-1, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        cos(beta/2)*exp(I*alpha/2)*exp(I*gamma/2)
    
    # 符号测试 j = 1
    assert Rotation.D(1, 1, 1, alpha, beta, gamma).doit() == \
        (1 + cos(beta))/2*exp(-I*alpha)*exp(-I*gamma)
    assert Rotation.D(1, 1, 0, alpha, beta, gamma).doit() == -sin(
        beta)/sqrt(2)*exp(-I*alpha)
    assert Rotation.D(1, 1, -1, alpha, beta, gamma).doit() == \
        (1 - cos(beta))/2*exp(-I*alpha)*exp(I*gamma)
    assert Rotation.D(1, 0, 1, alpha, beta, gamma).doit() == \
        sin(beta)/sqrt(2)*exp(-I*gamma)
    assert Rotation.D(1, 0, 0, alpha, beta, gamma).doit() == cos(beta)
    assert Rotation.D(1, 0, -1, alpha, beta, gamma).doit() == \
        -sin(beta)/sqrt(2)*exp(I*gamma)
    assert Rotation.D(1, -1, 1, alpha, beta, gamma).doit() == \
        (1 - cos(beta))/2*exp(I*alpha)*exp(-I*gamma)
    assert Rotation.D(1, -1, 0, alpha, beta, gamma).doit() == \
        sin(beta)/sqrt(2)*exp(I*alpha)
    assert Rotation.D(1, -1, -1, alpha, beta, gamma).doit() == \
        (1 + cos(beta))/2*exp(I*alpha)*exp(I*gamma)
    
    # 符号测试 j = 3/2
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), Rational(3, 2), S.Half, alpha, beta, gamma).doit() == \
        -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma/2)
    assert Rotation.D(Rational(3, 2), Rational(3, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        (-3*sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(-3, 2))*exp(I*gamma*Rational(3, 2))
    assert Rotation.D(Rational(3, 2), S.Half, Rational(3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma*Rational(-3, 2))
    assert Rotation.D(Rational(3, 2), S.Half, S.Half, alpha, beta, gamma).doit() == \
        (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(-I*gamma/2)
    assert Rotation.D(Rational(3, 2), S.Half, Rational(-1, 2), alpha, beta, gamma).doit() == \
        (sin(beta/2) - 3*sin(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma/2)
    # 第一个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), S.Half, Rational(-3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(-I*alpha/2)*exp(I*gamma*Rational(3, 2))
    
    # 第二个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma*Rational(-3, 2))
    
    # 第三个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), S.Half, alpha, beta, gamma).doit() == \
        (-sin(beta/2) + 3*sin(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(-I*gamma/2)
    
    # 第四个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        (cos(beta/2) + 3*cos(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma/2)
    
    # 第五个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-1, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        -sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha/2)*exp(I*gamma*Rational(3, 2))
    
    # 第六个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(3, 2), alpha, beta, gamma).doit() == \
        (3*sin(beta/2) - sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma*Rational(-3, 2))
    
    # 第七个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), S.Half, alpha, beta, gamma).doit() == \
        sqrt(3)*(cos(beta/2) - cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(-I*gamma/2)
    
    # 第八个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(-1, 2), alpha, beta, gamma).doit() == \
        sqrt(3)*(sin(beta/2) + sin(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma/2)
    
    # 第九个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(Rational(3, 2), Rational(-3, 2), Rational(-3, 2), alpha, beta, gamma).doit() == \
        (3*cos(beta/2) + cos(beta*Rational(3, 2)))/4*exp(I*alpha*Rational(3, 2))*exp(I*gamma*Rational(3, 2))
    
    # 第十个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 2, 2, alpha, beta, gamma).doit() == \
        (3 + 4*cos(beta) + cos(2*beta))/8*exp(-2*I*alpha)*exp(-2*I*gamma)
    
    # 第十一个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 2, 1, alpha, beta, gamma).doit() == \
        -((cos(beta) + 1)*exp(-2*I*alpha)*exp(-I*gamma)*sin(beta))/2
    
    # 第十二个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 2, 0, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(-2*I*alpha)
    
    # 第十三个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 2, -1, alpha, beta, gamma).doit() == \
        (cos(beta) - 1)*sin(beta)/2*exp(-2*I*alpha)*exp(I*gamma)
    
    # 第十四个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 2, -2, alpha, beta, gamma).doit() == \
        (3 - 4*cos(beta) + cos(2*beta))/8*exp(-2*I*alpha)*exp(2*I*gamma)
    
    # 第十五个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 1, 2, alpha, beta, gamma).doit() == \
        (cos(beta) + 1)*sin(beta)/2*exp(-I*alpha)*exp(-2*I*gamma)
    
    # 第十六个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 1, 1, alpha, beta, gamma).doit() == \
        (cos(beta) + cos(2*beta))/2*exp(-I*alpha)*exp(-I*gamma)
    
    # 第十七个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 1, 0, alpha, beta, gamma).doit() == -sqrt(6)* \
        sin(2*beta)/4*exp(-I*alpha)
    
    # 第十八个断言：验证 Rotation.D 对象的计算结果是否等于特定表达式
    assert Rotation.D(2, 1, -1, alpha, beta, gamma).doit() == \
        (cos(beta) - cos(2*beta))/2*exp(-I*alpha)*exp(I*gamma)
    # 断言：验证 Rotation.D(2, 1, -2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, 1, -2, alpha, beta, gamma).doit() == \
        (cos(beta) - 1)*sin(beta)/2*exp(-I*alpha)*exp(2*I*gamma)
    
    # 断言：验证 Rotation.D(2, 0, 2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, 0, 2, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(-2*I*gamma)
    
    # 断言：验证 Rotation.D(2, 0, 1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, 0, 1, alpha, beta, gamma).doit() == sqrt(6)* \
        sin(2*beta)/4*exp(-I*gamma)
    
    # 断言：验证 Rotation.D(2, 0, 0, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(
        2, 0, 0, alpha, beta, gamma).doit() == (1 + 3*cos(2*beta))/4
    
    # 断言：验证 Rotation.D(2, 0, -1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, 0, -1, alpha, beta, gamma).doit() == -sqrt(6)* \
        sin(2*beta)/4*exp(I*gamma)
    
    # 断言：验证 Rotation.D(2, 0, -2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, 0, -2, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(2*I*gamma)
    
    # 断言：验证 Rotation.D(2, -1, 2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -1, 2, alpha, beta, gamma).doit() == \
        (2*sin(beta) - sin(2*beta))/4*exp(I*alpha)*exp(-2*I*gamma)
    
    # 断言：验证 Rotation.D(2, -1, 1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -1, 1, alpha, beta, gamma).doit() == \
        (cos(beta) - cos(2*beta))/2*exp(I*alpha)*exp(-I*gamma)
    
    # 断言：验证 Rotation.D(2, -1, 0, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -1, 0, alpha, beta, gamma).doit() == sqrt(6)* \
        sin(2*beta)/4*exp(I*alpha)
    
    # 断言：验证 Rotation.D(2, -1, -1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -1, -1, alpha, beta, gamma).doit() == \
        (cos(beta) + cos(2*beta))/2*exp(I*alpha)*exp(I*gamma)
    
    # 断言：验证 Rotation.D(2, -1, -2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -1, -2, alpha, beta, gamma).doit() == \
        -((cos(beta) + 1)*sin(beta))/2*exp(I*alpha)*exp(2*I*gamma)
    
    # 断言：验证 Rotation.D(2, -2, 2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -2, 2, alpha, beta, gamma).doit() == \
        (3 - 4*cos(beta) + cos(2*beta))/8*exp(2*I*alpha)*exp(-2*I*gamma)
    
    # 断言：验证 Rotation.D(2, -2, 1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -2, 1, alpha, beta, gamma).doit() == \
        (2*sin(beta) - sin(2*beta))/4*exp(2*I*alpha)*exp(-I*gamma)
    
    # 断言：验证 Rotation.D(2, -2, 0, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -2, 0, alpha, beta, gamma).doit() == \
        sqrt(6)*sin(beta)**2/4*exp(2*I*alpha)
    
    # 断言：验证 Rotation.D(2, -2, -1, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -2, -1, alpha, beta, gamma).doit() == \
        (cos(beta) + 1)*sin(beta)/2*exp(2*I*alpha)*exp(I*gamma)
    
    # 断言：验证 Rotation.D(2, -2, -2, alpha, beta, gamma) 的结果是否等于给定表达式
    assert Rotation.D(2, -2, -2, alpha, beta, gamma).doit() == \
        (3 + 4*cos(beta) + cos(2*beta))/8*exp(2*I*alpha)*exp(2*I*gamma)
    
    # 数值测试
    # 断言：验证 Rotation.D(S.Half, S.Half, S.Half, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(
        S.Half, S.Half, S.Half, pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/2
    
    # 断言：验证 Rotation.D(S.Half, S.Half, Rational(-1, 2), pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(
        S.Half, S.Half, Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -sqrt(2)/2
    
    # 断言：验证 Rotation.D(S.Half, Rational(-1, 2), S.Half, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(
        S.Half, Rational(-1, 2), S.Half, pi/2, pi/2, pi/2).doit() == sqrt(2)/2
    
    # 断言：验证 Rotation.D(S.Half, Rational(-1, 2), Rational(-1, 2), pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(
        S.Half, Rational(-1, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == I*sqrt(2)/2
    
    # 断言：验证 Rotation.D(1, 1, 1, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(1, 1, 1, pi/2, pi/2, pi/2).doit() == Rational(-1, 2)
    
    # 断言：验证 Rotation.D(1, 1, 0, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(1, 1, 0, pi/2, pi/2, pi/2).doit() == I*sqrt(2)/2
    
    # 断言：验证 Rotation.D(1, 1, -1, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(1, 1, -1, pi/2, pi/2, pi/2).doit() == S.Half
    
    # 断言：验证 Rotation.D(1, 0, 1, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(1, 0, 1, pi/2, pi/2, pi/2).doit() == -I*sqrt(2)/2
    
    # 断言：验证 Rotation.D(1, 0, 0, pi/2, pi/2, pi/2) 的结果是否等于给定表达式
    assert Rotation.D(1, 0, 0, pi/2, pi/2, pi/2).doit() == 0
    
    # 断言：验证 Rotation.D(1, 0, -1, pi/2, pi/2
    # 验证旋转 D 表达式的结果是否等于 -1/2
    assert Rotation.D(1, -1, -1, pi/2, pi/2, pi/2).doit() == Rational(-1, 2)
    
    # j = 3/2 时的旋转 D 表达式验证
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), Rational(3, 2), pi/2, pi/2, pi/2).doit() == I*sqrt(2)/4
    
    # j = 3/2, m = 1/2, k = 1/2 时的旋转 D 表达式验证
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), S.Half, pi/2, pi/2, pi/2).doit() == sqrt(6)/4
    
    # j = 3/2, m = 1/2, k = -1/2 时的旋转 D 表达式验证
    assert Rotation.D(
        Rational(3, 2), Rational(3, 2), Rational(-1, 2), pi/2, pi/2, pi/2).doit() == -I*sqrt(6)/4
    
    # 其他类似的旋转 D 表达式验证，依次类推
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -1, 2, pi/2, pi/2, pi/2).doit() == -I/2
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -1, 1, pi/2, pi/2, pi/2).doit() == S.Half
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -1, 0, pi/2, pi/2, pi/2).doit() == 0
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -1, -1, pi/2, pi/2, pi/2).doit() == S.Half
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -1, -2, pi/2, pi/2, pi/2).doit() == I/2
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -2, 2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -2, 1, pi/2, pi/2, pi/2).doit() == I/2
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -2, 0, pi/2, pi/2, pi/2).doit() == -sqrt(6)/4
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -2, -1, pi/2, pi/2, pi/2).doit() == -I/2
    # 断言：验证 Rotation.D 函数对特定输入返回预期值是否成立
    assert Rotation.D(2, -2, -2, pi/2, pi/2, pi/2).doit() == Rational(1, 4)
def test_wignerd():
    # 断言旋转 D 函数与 WignerD 函数的结果是否相等
    assert Rotation.D(
        j, m, mp, alpha, beta, gamma) == WignerD(j, m, mp, alpha, beta, gamma)
    # 断言旋转 d 函数与 WignerD 函数的结果是否相等
    assert Rotation.d(j, m, mp, beta) == WignerD(j, m, mp, 0, beta, 0)

def test_wignerD():
    # 定义符号变量 i 和 j
    i,j=symbols('i j')
    # 各种参数下，旋转 D 函数与 WignerD 函数的结果是否相等
    assert Rotation.D(1, 1, 1, 0, 0, 0) == WignerD(1, 1, 1, 0, 0, 0)
    assert Rotation.D(1, 1, 2, 0, 0, 0) == WignerD(1, 1, 2, 0, 0, 0)
    assert Rotation.D(1, i**2 - j**2, i**2 - j**2, 0, 0, 0) == WignerD(1, i**2 - j**2, i**2 - j**2, 0, 0, 0)
    assert Rotation.D(1, i, i, 0, 0, 0) == WignerD(1, i, i, 0, 0, 0)
    assert Rotation.D(1, i, i+1, 0, 0, 0) == WignerD(1, i, i+1, 0, 0, 0)
    assert Rotation.D(1, 0, 0, 0, 0, 0) == WignerD(1, 0, 0, 0, 0, 0)

def test_jplus():
    # 断言 Jplus 和 Jminus 的对易子的结果是否为 2*hbar*Jz
    assert Commutator(Jplus, Jminus).doit() == 2*hbar*Jz
    # 断言 Jplus 对角元素的结果是否为 0
    assert Jplus.matrix_element(1, 1, 1, 1) == 0
    # 断言 Jplus 在 'xyz' 坐标系中的重写结果是否为 Jx + I*Jy
    assert Jplus.rewrite('xyz') == Jx + I*Jy
    # 对 Jplus 乘以 JxKet(1, 1) 的量子应用，数值计算结果
    assert qapply(Jplus*JxKet(1, 1)) == \
        -hbar*sqrt(2)*JxKet(1, 0)/2 + hbar*JxKet(1, 1)
    # 对 Jplus 乘以 JyKet(1, 1) 的量子应用，数值计算结果
    assert qapply(Jplus*JyKet(1, 1)) == \
        hbar*sqrt(2)*JyKet(1, 0)/2 + I*hbar*JyKet(1, 1)
    # 对 Jplus 乘以 JzKet(1, 1) 的量子应用，数值计算结果
    assert qapply(Jplus*JzKet(1, 1)) == 0
    # 对 Jplus 乘以 JxKet(j, m) 的量子应用，符号计算结果
    assert qapply(Jplus*JxKet(j, m)) == \
        Sum(hbar * sqrt(-mi**2 - mi + j**2 + j) * WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    # 对 Jplus 乘以 JyKet(j, m) 的量子应用，符号计算结果
    assert qapply(Jplus*JyKet(j, m)) == \
        Sum(hbar * sqrt(j**2 + j - mi**2 - mi) * WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    # 对 Jplus 乘以 JzKet(j, m) 的量子应用，符号计算结果
    assert qapply(Jplus*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKet(j, m + 1)
    # 对 Jplus 乘以 JxKetCoupled(j, m, (j1, j2)) 的量子应用，符号计算结果
    assert qapply(Jplus*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar * sqrt(-mi**2 - mi + j**2 + j) * WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(
            WignerD(
                j, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    # 对 Jplus 乘以 JyKetCoupled(j, m, (j1, j2)) 的量子应用，符号计算结果
    assert qapply(Jplus*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar * sqrt(j**2 + j - mi**2 - mi) * WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(
            WignerD(j, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) *
            JyKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    # 对 Jplus 乘以 JzKetCoupled(j, m, (j1, j2)) 的量子应用，符号计算结果
    assert qapply(Jplus*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2 + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2 + \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1))

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(
        TensorProduct(Jplus, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))

    # 符号化处理的量子算符应用，施加 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar * sqrt(-mi**2 - mi + j1**2 + j1) * WignerD(j1, mi, m1, 0, pi/2, 0) *
        Sum(WignerD(j1, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))

    # 符号化处理的量子算符应用，施加 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar * sqrt(-mi**2 - mi + j2**2 + j2) * WignerD(j2, mi, m2, 0, pi/2, 0) *
        Sum(WignerD(j2, mi1, mi + 1, 0, pi*Rational(3, 2), 0) * JxKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))

    # 符号化处理的量子算符应用，施加 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar * sqrt(j1**2 + j1 - mi**2 - mi) * WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j1, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))

    # 符号化处理的量子算符应用，施加 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar * sqrt(j2**2 + j2 - mi**2 - mi) * WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j2, mi1, mi + 1, pi*Rational(3, 2), pi/2, pi/2) * JyKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(Jplus, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))

    # 施加量子算符 Jplus 到两个张量积态的乘积，并验证结果
    assert qapply(TensorProduct(1, Jplus)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))
# 定义名为 test_jminus 的函数，用于测试 Jminus 算符的功能
def test_jminus():
    # 断言：Jminus 作用在 JzKet(1, -1) 上结果应为 0
    assert qapply(Jminus*JzKet(1, -1)) == 0
    # 断言：计算 Jminus 的矩阵元素 ⟨1, 0|Jminus|1, 1⟩ 结果为 sqrt(2)*hbar
    assert Jminus.matrix_element(1, 0, 1, 1) == sqrt(2)*hbar
    # 断言：Jminus 的重写为 'xyz' 表示 Jx - I*Jy
    assert Jminus.rewrite('xyz') == Jx - I*Jy
    # Normal operators, normal states
    # Numerical
    # 断言：Jminus 作用在 JxKet(1, 1) 上的数值计算结果
    assert qapply(Jminus*JxKet(1, 1)) == \
        hbar*sqrt(2)*JxKet(1, 0)/2 + hbar*JxKet(1, 1)
    # 断言：Jminus 作用在 JyKet(1, 1) 上的数值计算结果
    assert qapply(Jminus*JyKet(1, 1)) == \
        hbar*sqrt(2)*JyKet(1, 0)/2 - hbar*I*JyKet(1, 1)
    # 断言：Jminus 作用在 JzKet(1, 1) 上的数值计算结果
    assert qapply(Jminus*JzKet(1, 1)) == sqrt(2)*hbar*JzKet(1, 0)
    # Symbolic
    # 断言：Jminus 作用在 JxKet(j, m) 上的符号计算结果
    assert qapply(Jminus*JxKet(j, m)) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    # 断言：Jminus 作用在 JyKet(j, m) 上的符号计算结果
    assert qapply(Jminus*JyKet(j, m)) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j, mi1),
        (mi1, -j, j)), (mi, -j, j))
    # 断言：Jminus 作用在 JzKet(j, m) 上的符号计算结果
    assert qapply(Jminus*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 + m)*JzKet(j, m - 1)
    # Normal operators, coupled states
    # Numerical
    # 断言：Jminus 作用在 JxKetCoupled(1, 1, (1, 1)) 上的数值计算结果
    assert qapply(Jminus*JxKetCoupled(1, 1, (1, 1))) == \
        hbar*sqrt(2)*JxKetCoupled(1, 0, (1, 1))/2 + \
        hbar*JxKetCoupled(1, 1, (1, 1))
    # 断言：Jminus 作用在 JyKetCoupled(1, 1, (1, 1)) 上的数值计算结果
    assert qapply(Jminus*JyKetCoupled(1, 1, (1, 1))) == \
        hbar*sqrt(2)*JyKetCoupled(1, 0, (1, 1))/2 - \
        hbar*I*JyKetCoupled(1, 1, (1, 1))
    # 断言：Jminus 作用在 JzKetCoupled(1, 1, (1, 1)) 上的数值计算结果
    assert qapply(Jminus*JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*hbar*JzKetCoupled(1, 0, (1, 1))
    # Symbolic
    # 断言：Jminus 作用在 JxKetCoupled(j, m, (j1, j2)) 上的符号计算结果
    assert qapply(Jminus*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, 0, pi/2, 0) *
        Sum(WignerD(j, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    # 断言：Jminus 作用在 JyKetCoupled(j, m, (j1, j2)) 上的符号计算结果
    assert qapply(Jminus*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*sqrt(j**2 + j - mi**2 + mi)*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(
            WignerD(j, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*
            JyKetCoupled(j, mi1, (j1, j2)),
        (mi1, -j, j)), (mi, -j, j))
    # 断言：Jminus 作用在 JzKetCoupled(j, m, (j1, j2)) 上的符号计算结果
    assert qapply(Jminus*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    # 断言：Jminus 作用在 TensorProduct(Jminus, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1)) 上的数值计算结果
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2 + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    # 断言：Jminus 作用在 TensorProduct(1, Jminus)*TensorProduct(JxKet(1, 1), JxKet(1, -1)) 上的数值计算结果
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1)) - \
        hbar*sqrt(2)*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2
    # 断言：Jminus 作用在 TensorProduct(Jminus, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1)) 上的数值计算结果
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2 - \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, -1)) + \
        hbar*sqrt(2)*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        sqrt(2)*hbar*TensorProduct(JzKet(1, 0), JzKet(1, -1))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(
        1, Jminus)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    
    # Symbolic
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式（符号表示）
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar*sqrt(j1**2 + j1 - mi**2 + mi)*WignerD(j1, mi, m1, 0, pi/2, 0) *
        Sum(WignerD(j1, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*sqrt(j2**2 + j2 - mi**2 + mi)*WignerD(j2, mi, m2, 0, pi/2, 0) *
        Sum(WignerD(j2, mi1, mi - 1, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*sqrt(j1**2 + j1 - mi**2 + mi)*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j1, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1),
        (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*sqrt(j2**2 + j2 - mi**2 + mi)*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2) *
        Sum(WignerD(j2, mi1, mi - 1, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1),
        (mi1, -j2, j2)), (mi, -j2, j2)))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(Jminus, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))
    
    # 断言：应用量子操作到张量积态上，检查结果是否等于给定表达式
    assert qapply(TensorProduct(1, Jminus)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*sqrt(
            j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))
def test_j2():
    assert Commutator(J2, Jz).doit() == 0
    assert J2.matrix_element(1, 1, 1, 1) == 2*hbar**2
    # Normal operators, normal states
    # Numerical
    assert qapply(J2*JxKet(1, 1)) == 2*hbar**2*JxKet(1, 1)
    assert qapply(J2*JyKet(1, 1)) == 2*hbar**2*JyKet(1, 1)
    assert qapply(J2*JzKet(1, 1)) == 2*hbar**2*JzKet(1, 1)
    # Symbolic
    assert qapply(J2*JxKet(j, m)) == \
        hbar**2*j**2*JxKet(j, m) + hbar**2*j*JxKet(j, m)
    assert qapply(J2*JyKet(j, m)) == \
        hbar**2*j**2*JyKet(j, m) + hbar**2*j*JyKet(j, m)
    assert qapply(J2*JzKet(j, m)) == \
        hbar**2*j**2*JzKet(j, m) + hbar**2*j*JzKet(j, m)
    # Normal operators, coupled states
    # Numerical
    assert qapply(J2*JxKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JxKetCoupled(1, 1, (1, 1))
    assert qapply(J2*JyKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JyKetCoupled(1, 1, (1, 1))
    assert qapply(J2*JzKetCoupled(1, 1, (1, 1))) == \
        2*hbar**2*JzKetCoupled(1, 1, (1, 1))
    # Symbolic
    assert qapply(J2*JxKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JxKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JxKetCoupled(j, m, (j1, j2))
    assert qapply(J2*JyKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JyKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JyKetCoupled(j, m, (j1, j2))
    assert qapply(J2*JzKetCoupled(j, m, (j1, j2))) == \
        hbar**2*j**2*JzKetCoupled(j, m, (j1, j2)) + \
        hbar**2*j*JzKetCoupled(j, m, (j1, j2))
    # Uncoupled operators, uncoupled states
    # Numerical
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        2*hbar**2*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        2*hbar**2*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        2*hbar**2*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        2*hbar**2*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        2*hbar**2*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        2*hbar**2*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    # Symbolic
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)))
    assert qapply(TensorProduct(1, J2)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2)))


注释：

# 定义了一个名为 test_j2 的测试函数
def test_j2():
    # 断言：J2 和 Jz 的对易子等于 0
    assert Commutator(J2, Jz).doit() == 0
    # 断言：J2 的矩阵元素 (1, 1, 1, 1) 等于 2*hbar**2
    assert J2.matrix_element(1, 1, 1, 1) == 2*hbar**2
    # Normal operators, normal states
    # Numerical
    # 断言：对于数值状态，应用 J2 到 JxKet(1, 1) 得到 2*hbar**2*JxKet(1, 1)
    assert qapply(J2*JxKet(1, 1)) == 2*hbar**2*JxKet(1, 1)
    # 同上，应用 J2 到 JyKet(1, 1) 得到 2*hbar**2*JyKet(1, 1)
    assert qapply(J2*JyKet(1, 1)) == 2*hbar**2*JyKet(1, 1)
    # 同上，应用 J2 到 JzKet(1, 1) 得到 2*hbar**2*JzKet(1, 1)
    assert qapply(J2*JzKet(1, 1)) == 2*hbar**2*JzKet(1, 1)
    # Symbolic
    # 断言：对于符号状态，应用 J2 到 JxKet(j, m) 得到对应的符号表达式
    assert qapply(J2*JxKet(j, m)) == \
        hbar**2*j**2*JxKet(j, m) + hbar**2*j*JxKet(j, m)
    # 同上，应用 J2 到 JyKet(j, m) 得到对应的符号表达式
    assert qapply(J2*JyKet(j, m)) == \
        hbar**2*j**2*JyKet(j, m) + hbar**2*j*JyKet(j, m)
    # 同上，应用 J2 到 JzKet(j, m) 得到对应的符号表达式
    assert qapply(J2*JzKet(j, m)) == \
        hbar**2*j**2*JzKet(j, m) + hbar**2*j*JzKet(j, m)
    # Normal operators, coupled states
    # Numerical
    # 断言：对于耦合态，应用 J2 到 JxKetCoupled(1, 1, (1, 1)) 得到对
    # 断言：应用量子操作到张量积态的运算结果是否符合预期
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    
    # 断言：应用量子操作到张量积态的运算结果是否符合预期
    assert qapply(TensorProduct(1, J2)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    
    # 断言：应用量子操作到张量积态的运算结果是否符合预期
    assert qapply(TensorProduct(J2, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar**2*j1**2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + \
        hbar**2*j1*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    
    # 断言：应用量子操作到张量积态的运算结果是否符合预期
    assert qapply(TensorProduct(1, J2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar**2*j2**2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + \
        hbar**2*j2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
def test_jx():
    # 断言：计算 Jx 和 Jz 的对易子，并验证结果是否等于 -i*hbar*Jy
    assert Commutator(Jx, Jz).doit() == -I*hbar*Jy

    # 断言：使用 'plusminus' 形式重写 Jx，并验证结果是否等于 (Jminus + Jplus)/2
    assert Jx.rewrite('plusminus') == (Jminus + Jplus)/2

    # 断言：使用指定基底 Jz 和量子数 j 表示 Jx，并验证结果是否等于 (represent(Jplus, basis=Jz, j=1) + represent(Jminus, basis=Jz, j=1))/2
    assert represent(Jx, basis=Jz, j=1) == (represent(Jplus, basis=Jz, j=1) + represent(Jminus, basis=Jz, j=1))/2

    # Normal operators, normal states
    # Numerical
    # 断言：应用 Jx 作用在 JxKet(1, 1) 上，并验证结果是否等于 hbar*JxKet(1, 1)
    assert qapply(Jx*JxKet(1, 1)) == hbar*JxKet(1, 1)

    # 断言：应用 Jx 作用在 JyKet(1, 1) 上，并验证结果是否等于 hbar*JyKet(1, 1)
    assert qapply(Jx*JyKet(1, 1)) == hbar*JyKet(1, 1)

    # 断言：应用 Jx 作用在 JzKet(1, 1) 上，并验证结果是否等于 sqrt(2)*hbar*JzKet(1, 0)/2
    assert qapply(Jx*JzKet(1, 1)) == sqrt(2)*hbar*JzKet(1, 0)/2

    # Symbolic
    # 断言：应用 Jx 作用在 JxKet(j, m) 上，并验证结果是否等于 hbar*m*JxKet(j, m)
    assert qapply(Jx*JxKet(j, m)) == hbar*m*JxKet(j, m)

    # 断言：应用 Jx 作用在 JyKet(j, m) 上，并验证结果是否等于 复杂的符号表达式
    assert qapply(Jx*JyKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, 0, pi/2)*Sum(WignerD(j, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))

    # 断言：应用 Jx 作用在 JzKet(j, m) 上，并验证结果是否等于 复杂的符号表达式
    assert qapply(Jx*JzKet(j, m)) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKet(j, m + 1)/2 + hbar*sqrt(j**2 + j - m**2 + m)*JzKet(j, m - 1)/2

    # Normal operators, coupled states
    # Numerical
    # 断言：应用 Jx 作用在 JxKetCoupled(1, 1, (1, 1)) 上，并验证结果是否等于 hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jx*JxKetCoupled(1, 1, (1, 1))) == hbar*JxKetCoupled(1, 1, (1, 1))

    # 断言：应用 Jx 作用在 JyKetCoupled(1, 1, (1, 1)) 上，并验证结果是否等于 hbar*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jx*JyKetCoupled(1, 1, (1, 1))) == hbar*JyKetCoupled(1, 1, (1, 1))

    # 断言：应用 Jx 作用在 JzKetCoupled(1, 1, (1, 1)) 上，并验证结果是否等于 sqrt(2)*hbar*JzKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jx*JzKetCoupled(1, 1, (1, 1))) == sqrt(2)*hbar*JzKetCoupled(1, 0, (1, 1))/2

    # Symbolic
    # 断言：应用 Jx 作用在 JxKetCoupled(j, m, (j1, j2)) 上，并验证结果是否等于 hbar*m*JxKetCoupled(j, m, (j1, j2))
    assert qapply(Jx*JxKetCoupled(j, m, (j1, j2))) == hbar*m*JxKetCoupled(j, m, (j1, j2))

    # 断言：应用 Jx 作用在 JyKetCoupled(j, m, (j1, j2)) 上，并验证结果是否等于 复杂的符号表达式
    assert qapply(Jx*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, 0, pi/2)*Sum(WignerD(j, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))

    # 断言：应用 Jx 作用在 JzKetCoupled(j, m, (j1, j2)) 上，并验证结果是否等于 复杂的符号表达式
    assert qapply(Jx*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))/2 + hbar*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))/2

    # Normal operators, uncoupled states
    # Numerical
    # 断言：应用 Jx 作用在 TensorProduct(JxKet(1, 1), JxKet(1, 1)) 上，并验证结果是否等于 2*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jx*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == 2*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))

    # 断言：应用 Jx 作用在 TensorProduct(JyKet(1, 1), JyKet(1, 1)) 上，并验证结果是否等于 hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1)) + hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jx*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1)) + hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))

    # 断言：应用 Jx 作用在 TensorProduct(JzKet(1, 1), JzKet(1, 1)) 上，并验证结果是否等于 sqrt(2)*hbar*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + sqrt(2)*hbar*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert qapply(Jx*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == sqrt(2)*hbar*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2
    # 施加算符 Jx 到张量积态 JyKet(j1, m1) 和 JyKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(Jx*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算结果：第一项
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, 0, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2)) + \
        # 计算结果：第二项
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, 0, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))

    # 施加算符 Jx 到张量积态 JzKet(j1, m1) 和 JzKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(Jx*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        # 计算结果：第一项
        hbar*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        # 计算结果：第二项
        hbar*sqrt(j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2 + \
        # 计算结果：第三项
        hbar*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        # 计算结果：第四项
        hbar*sqrt(j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2

    # 施加算符 Jx 到张量积态 JxKet(1, 1) 和 JxKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        # 计算结果
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))

    # 施加算符 Jx 到张量积态 JxKet(1, 1) 和 JxKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        # 计算结果
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))

    # 施加算符 Jx 到张量积态 JyKet(1, 1) 和 JyKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        # 计算结果
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))

    # 施加算符 Jx 到张量积态 JyKet(1, 1) 和 JyKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        # 计算结果
        -hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))

    # 施加算符 Jx 到张量积态 JzKet(1, 1) 和 JzKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        # 计算结果
        hbar*sqrt(2)*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2

    # 施加算符 Jx 到张量积态 JzKet(1, 1) 和 JzKet(1, -1)，并应用量子求解器进行计算
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        # 计算结果
        hbar*sqrt(2)*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2

    # 施加算符 Jx 到张量积态 JxKet(j1, m1) 和 JxKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        # 计算结果
        hbar*m1*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))

    # 施加算符 Jx 到张量积态 JxKet(j1, m1) 和 JxKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        # 计算结果
        hbar*m2*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))

    # 施加算符 Jx 到张量积态 JyKet(j1, m1) 和 JyKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算结果
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, 0, pi/2) * Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))

    # 施加算符 Jx 到张量积态 JyKet(j1, m1) 和 JyKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算结果
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, 0, pi/2) * Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), 0, 0)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))

    # 施加算符 Jx 到张量积态 JzKet(j1, m1) 和 JzKet(j2, m2)，并应用量子求解器进行计算
    assert qapply(TensorProduct(Jx, 1)*TensorProduct(JzKet(j1
    # 断言：验证张量积运算的应用是否正确
    assert qapply(TensorProduct(1, Jx)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        # 计算第一项：角动量算符 Jx 作用于张量积态 JzKet(j1, m1) ⊗ JzKet(j2, m2)，乘以常数 hbar 和 sqrt(j2**2 + j2 - m2**2 - m2)，结果除以 2
        hbar*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        # 计算第二项：角动量算符 Jx 作用于张量积态 JzKet(j1, m1) ⊗ JzKet(j2, m2)，乘以常数 hbar 和 sqrt(j2**2 + j2 - m2**2 + m2)，结果除以 2
        hbar*sqrt(j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2
def test_jy():
    # 断言：计算 Jy 和 Jz 的对易子，并应用 doit 方法，期望结果为 I*hbar*Jx
    assert Commutator(Jy, Jz).doit() == I*hbar*Jx
    # 断言：使用 'plusminus' 重写 Jy 表达式，期望结果为 (Jplus - Jminus)/(2*I)
    assert Jy.rewrite('plusminus') == (Jplus - Jminus)/(2*I)
    # 断言：使用 Jz 作为基础，表示 Jy 在该基础下的表示，期望结果为 (represent(Jplus) - represent(Jminus))/(2*I)
    assert represent(Jy, basis=Jz) == (
        represent(Jplus, basis=Jz) - represent(Jminus, basis=Jz))/(2*I)
    
    # Normal operators, normal states
    # 数值计算
    # 断言：对 Jy 作用于 JxKet(1, 1)，期望结果为 hbar*JxKet(1, 1)
    assert qapply(Jy*JxKet(1, 1)) == hbar*JxKet(1, 1)
    # 断言：对 Jy 作用于 JyKet(1, 1)，期望结果为 hbar*JyKet(1, 1)
    assert qapply(Jy*JyKet(1, 1)) == hbar*JyKet(1, 1)
    # 断言：对 Jy 作用于 JzKet(1, 1)，期望结果为 sqrt(2)*hbar*I*JzKet(1, 0)/2
    assert qapply(Jy*JzKet(1, 1)) == sqrt(2)*hbar*I*JzKet(1, 0)/2
    
    # 符号计算
    # 断言：对 Jy 作用于 JxKet(j, m)，期望结果为由符号求和和乘积的表达式
    assert qapply(Jy*JxKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), 0, 0)*Sum(WignerD(
            j, mi1, mi, 0, 0, pi/2)*JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    # 断言：对 Jy 作用于 JyKet(j, m)，期望结果为 hbar*m*JyKet(j, m)
    assert qapply(Jy*JyKet(j, m)) == hbar*m*JyKet(j, m)
    # 断言：对 Jy 作用于 JzKet(j, m)，期望结果为由符号表达式组成
    assert qapply(Jy*JzKet(j, m)) == \
        -hbar*I*sqrt(j**2 + j - m**2 - m)*JzKet(
            j, m + 1)/2 + hbar*I*sqrt(j**2 + j - m**2 + m)*JzKet(j, m - 1)/2
    
    # Normal operators, coupled states
    # 数值计算
    # 断言：对 Jy 作用于 JxKetCoupled(1, 1, (1, 1))，期望结果为 hbar*JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jy*JxKetCoupled(1, 1, (1, 1))) == \
        hbar*JxKetCoupled(1, 1, (1, 1))
    # 断言：对 Jy 作用于 JyKetCoupled(1, 1, (1, 1))，期望结果为 hbar*JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jy*JyKetCoupled(1, 1, (1, 1))) == \
        hbar*JyKetCoupled(1, 1, (1, 1))
    # 断言：对 Jy 作用于 JzKetCoupled(1, 1, (1, 1))，期望结果为 sqrt(2)*hbar*I*JzKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jy*JzKetCoupled(1, 1, (1, 1))) == \
        sqrt(2)*hbar*I*JzKetCoupled(1, 0, (1, 1))/2
    
    # 符号计算
    # 断言：对 Jy 作用于 JxKetCoupled(j, m, (j1, j2))，期望结果为由符号求和和乘积的表达式
    assert qapply(Jy*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), 0, 0)*Sum(WignerD(j, mi1, mi, 0, 0, pi/2)*JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    # 断言：对 Jy 作用于 JyKetCoupled(j, m, (j1, j2))，期望结果为 hbar*m*JyKetCoupled(j, m, (j1, j2))
    assert qapply(Jy*JyKetCoupled(j, m, (j1, j2))) == \
        hbar*m*JyKetCoupled(j, m, (j1, j2))
    # 断言：对 Jy 作用于 JzKetCoupled(j, m, (j1, j2))，期望结果为由符号表达式组成
    assert qapply(Jy*JzKetCoupled(j, m, (j1, j2))) == \
        -hbar*I*sqrt(j**2 + j - m**2 - m)*JzKetCoupled(j, m + 1, (j1, j2))/2 + \
        hbar*I*sqrt(j**2 + j - m**2 + m)*JzKetCoupled(j, m - 1, (j1, j2))/2
    
    # Normal operators, uncoupled states
    # 数值计算
    # 断言：对 Jy 作用于 TensorProduct(JxKet(1, 1), JxKet(1, 1))，期望结果为 hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1)) + hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jy*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1)) + \
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, 1))
    # 断言：对 Jy 作用于 TensorProduct(JyKet(1, 1), JyKet(1, 1))，期望结果为 2*hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jy*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == \
        2*hbar*TensorProduct(JyKet(1, 1), JyKet(1, 1))
    # 断言：对 Jy 作用于 TensorProduct(JzKet(1, 1), JzKet(1, 1))，期望结果为 sqrt(2)*hbar*I*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2 + sqrt(2)*hbar*I*TensorProduct(JzKet(1, 0), JzKet(1, 1))/2
    assert qapply(Jy*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        sqrt(2)*hbar*I*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2
    # 施加 Jy 操作符到 JyKet(j1, m1) 和 JyKet(j2, m2) 的张量积上，并应用 qapply 函数进行量子态演化
    assert qapply(Jy*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算得到的结果应当为两个量子态之间的耦合项之和，包括对应的角动量和哈密顿量因子
        hbar*m1*TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + hbar*m2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))

    # 施加 Jy 操作符到 JzKet(j1, m1) 和 JzKet(j2, m2) 的张量积上，并应用 qapply 函数进行量子态演化
    assert qapply(Jy*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        # 计算得到的结果应当为两个量子态之间的耦合项之和，包括对应的角动量和哈密顿量因子
        -hbar*I*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*I*sqrt(j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2 + \
        -hbar*I*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        hbar*I*sqrt(j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2

    # 未耦合的操作符，未耦合的态
    # 数值计算
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        -hbar*TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        -hbar*TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        hbar*sqrt(2)*I*TensorProduct(JzKet(1, 0), JzKet(1, -1))/2
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        # 计算得到的结果应当为两个态之间的哈密顿量作用结果
        -hbar*sqrt(2)*I*TensorProduct(JzKet(1, 1), JzKet(1, 0))/2

    # 符号计算
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        # 计算得到的结果应当为两个态之间的符号哈密顿量作用结果，包括 WignerD 函数的调用
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), 0, 0) * Sum(WignerD(j1, mi1, mi, 0, 0, pi/2)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        # 计算得到的结果应当为两个态之间的符号哈密顿量作用结果，包括 WignerD 函数的调用
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), 0, 0) * Sum(WignerD(j2, mi1, mi, 0, 0, pi/2)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算得到的结果应当为两个态之间的符号哈密顿量作用结果
        hbar*m1*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        # 计算得到的结果应当为两个态之间的符号哈密顿量作用结果
        hbar*m2*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(Jy, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        # 计算得到的结果应当为两个态之间的符号哈密顿量作用结果
        -hbar*I*sqrt(j1**2 + j1 - m1**2 - m1)*TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))/2 + \
        hbar*I*sqrt(j1**2 + j1 - m1**2 + m1)*TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2))/2
    # 断言语句，验证 qapply 函数对 TensorProduct(1, Jy)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) 的处理结果
    assert qapply(TensorProduct(1, Jy)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        # 计算第一项的结果：负号、虚数单位、以及 j2、m2 的平方和差对应的张量积结果
        -hbar*I*sqrt(j2**2 + j2 - m2**2 - m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))/2 + \
        # 计算第二项的结果：虚数单位、以及 j2、m2 的平方和和对应的张量积结果
        hbar*I*sqrt(j2**2 + j2 - m2**2 + m2)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1))/2
def test_jz():
    assert Commutator(Jz, Jminus).doit() == -hbar*Jminus
    # 计算Jz和Jminus的对易子，返回结果应为-hbar乘以Jminus

    # Normal operators, normal states
    # 数值计算
    assert qapply(Jz*JxKet(1, 1)) == -sqrt(2)*hbar*JxKet(1, 0)/2
    assert qapply(Jz*JyKet(1, 1)) == -sqrt(2)*hbar*I*JyKet(1, 0)/2
    assert qapply(Jz*JzKet(2, 1)) == hbar*JzKet(2, 1)

    # 符号计算
    assert qapply(Jz*JxKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, pi/2, 0)*Sum(WignerD(j,
            mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JyKet(j, m)) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j, mi1,
            mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JzKet(j, m)) == hbar*m*JzKet(j, m)

    # Normal operators, coupled states
    # 数值计算
    assert qapply(Jz*JxKetCoupled(1, 1, (1, 1))) == \
        -sqrt(2)*hbar*JxKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jz*JyKetCoupled(1, 1, (1, 1))) == \
        -sqrt(2)*hbar*I*JyKetCoupled(1, 0, (1, 1))/2
    assert qapply(Jz*JzKetCoupled(1, 1, (1, 1))) == \
        hbar*JzKetCoupled(1, 1, (1, 1))

    # 符号计算
    assert qapply(Jz*JxKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, 0, pi/2, 0)*Sum(WignerD(j, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JyKetCoupled(j, m, (j1, j2))) == \
        Sum(hbar*mi*WignerD(j, mi, m, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jz*JzKetCoupled(j, m, (j1, j2))) == \
        hbar*m*JzKetCoupled(j, m, (j1, j2))

    # Normal operators, uncoupled states
    # 数值计算
    assert qapply(Jz*TensorProduct(JxKet(1, 1), JxKet(1, 1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2 - \
        sqrt(2)*hbar*TensorProduct(JxKet(1, 0), JxKet(1, 1))/2
    assert qapply(Jz*TensorProduct(JyKet(1, 1), JyKet(1, 1))) == \
        -sqrt(2)*hbar*I*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2 - \
        sqrt(2)*hbar*I*TensorProduct(JyKet(1, 0), JyKet(1, 1))/2
    assert qapply(Jz*TensorProduct(JzKet(1, 1), JzKet(1, 1))) == \
        2*hbar*TensorProduct(JzKet(1, 1), JzKet(1, 1))
    assert qapply(Jz*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0

    # 符号计算
    assert qapply(Jz*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, pi/2, 0)*Sum(WignerD(j2, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, pi/2, 0)*Sum(WignerD(j1, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
   `
    # 断言 qapply 函数应用于 Jz 和 TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(Jz*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    
    # 断言 qapply 函数应用于 Jz 和 TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(Jz*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m1*TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) + hbar*m2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JxKet(1, 1), JxKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 0), JxKet(1, -1))/2
    
    # 断言 qapply 函数应用于 TensorProduct(1, Jz) 和 TensorProduct(JxKet(1, 1), JxKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JxKet(1, 1), JxKet(1, -1))) == \
        -sqrt(2)*hbar*TensorProduct(JxKet(1, 1), JxKet(1, 0))/2
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JyKet(1, 1), JyKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        -sqrt(2)*I*hbar*TensorProduct(JyKet(1, 0), JyKet(1, -1))/2
    
    # 断言 qapply 函数应用于 TensorProduct(1, Jz) 和 TensorProduct(JyKet(1, 1), JyKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JyKet(1, 1), JyKet(1, -1))) == \
        sqrt(2)*I*hbar*TensorProduct(JyKet(1, 1), JyKet(1, 0))/2
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JzKet(1, 1), JzKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        hbar*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    
    # 断言 qapply 函数应用于 TensorProduct(1, Jz) 和 TensorProduct(JzKet(1, 1), JzKet(1, -1)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JzKet(1, 1), JzKet(1, -1))) == \
        -hbar*TensorProduct(JzKet(1, 1), JzKet(1, -1))
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, 0, pi/2, 0)*Sum(WignerD(j1, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    
    # 断言 qapply 函数应用于 TensorProduct(1, Jz) 和 TensorProduct(JxKet(j1, m1), JxKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == \
        TensorProduct(JxKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, 0, pi/2, 0)*Sum(WignerD(j2, mi1, mi, 0, pi*Rational(3, 2), 0)*JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(Sum(hbar*mi*WignerD(j1, mi, m1, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j1, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    
    # 断言 qapply 函数应用于 TensorProduct(1, Jz) 和 TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == \
        TensorProduct(JyKet(j1, m1), Sum(hbar*mi*WignerD(j2, mi, m2, pi*Rational(3, 2), -pi/2, pi/2)*Sum(WignerD(j2, mi1, mi, pi*Rational(3, 2), pi/2, pi/2)*JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    
    # 断言 qapply 函数应用于 TensorProduct(Jz, 1) 和 TensorProduct(JzKet(j1, m1), JzKet(j2, m2)) 的结果，应该等于表达式右侧的 TensorProduct
    assert qapply(TensorProduct(Jz, 1)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m1*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
    # 使用断言验证张量积算子作用在两个 JzKet 状态向量上的结果是否正确
    assert qapply(TensorProduct(1, Jz)*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == \
        hbar*m2*TensorProduct(JzKet(j1, m1), JzKet(j2, m2))
def test_rotation():
    a, b, g = symbols('a b g')  # 定义符号变量 a, b, g
    j, m = symbols('j m')  # 定义符号变量 j, m
    # Uncoupled
    answ = [JxKet(1,-1)/2 - sqrt(2)*JxKet(1,0)/2 + JxKet(1,1)/2 ,  # 创建答案列表
            JyKet(1,-1)/2 - sqrt(2)*JyKet(1,0)/2 + JyKet(1,1)/2 ,
            JzKet(1,-1)/2 - sqrt(2)*JzKet(1,0)/2 + JzKet(1,1)/2]
    fun = [state(1, 1) for state in (JxKet, JyKet, JzKet)]  # 创建状态函数列表
    for state in fun:  # 对于每个状态函数
        got = qapply(Rotation(0, pi/2, 0)*state)  # 应用旋转操作并计算状态
        assert got in answ  # 断言结果在答案列表中
        answ.remove(got)  # 移除已经检查过的结果
    assert not answ  # 断言答案列表已空
    arg = Rotation(a, b, g)*fun[0]  # 构造旋转后的第一个状态函数参数
    assert qapply(arg) == (
        -exp(-I*a)*exp(I*g)*cos(b)*JxKet(1,-1)/2 +  # 断言旋转后的结果与预期相等
        exp(-I*a)*exp(I*g)*JxKet(1,-1)/2 - sqrt(2)*exp(-I*a)*sin(b)*JxKet(1,0)/2 +
        exp(-I*a)*exp(-I*g)*cos(b)*JxKet(1,1)/2 + exp(-I*a)*exp(-I*g)*JxKet(1,1)/2)
    # dummy effective
    assert str(qapply(Rotation(a, b, g)*JzKet(j, m), dummy=False)) == str(
        qapply(Rotation(a, b, g)*JzKet(j, m), dummy=True)).replace('_','')
    # Coupled
    ans = [JxKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JxKetCoupled(1,0,(1,1))/2 +
           JxKetCoupled(1,1,(1,1))/2 ,
           JyKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JyKetCoupled(1,0,(1,1))/2 +
           JyKetCoupled(1,1,(1,1))/2 ,
           JzKetCoupled(1,-1,(1,1))/2 - sqrt(2)*JzKetCoupled(1,0,(1,1))/2 +
           JzKetCoupled(1,1,(1,1))/2]
    fun = [state(1, 1, (1,1)) for state in (JxKetCoupled, JyKetCoupled, JzKetCoupled)]
    for state in fun:  # 对于每个状态函数
        got = qapply(Rotation(0, pi/2, 0)*state)  # 应用旋转操作并计算状态
        assert got in ans  # 断言结果在答案列表中
        ans.remove(got)  # 移除已经检查过的结果
    assert not ans  # 断言答案列表已空
    arg = Rotation(a, b, g)*fun[0]  # 构造旋转后的第一个状态函数参数
    assert qapply(arg) == (
        -exp(-I*a)*exp(I*g)*cos(b)*JxKetCoupled(1,-1,(1,1))/2 +
        exp(-I*a)*exp(I*g)*JxKetCoupled(1,-1,(1,1))/2 -
        sqrt(2)*exp(-I*a)*sin(b)*JxKetCoupled(1,0,(1,1))/2 +
        exp(-I*a)*exp(-I*g)*cos(b)*JxKetCoupled(1,1,(1,1))/2 +
        exp(-I*a)*exp(-I*g)*JxKetCoupled(1,1,(1,1))/2)
    # dummy effective
    assert str(qapply(Rotation(a,b,g)*JzKetCoupled(j,m,(j1,j2)), dummy=False)) == str(
        qapply(Rotation(a,b,g)*JzKetCoupled(j,m,(j1,j2)), dummy=True)).replace('_','')


def test_jzket():
    j, m = symbols('j m')  # 定义符号变量 j, m
    # j not integer or half integer
    raises(ValueError, lambda: JzKet(Rational(2, 3), Rational(-1, 3)))  # 断言引发 ValueError 异常
    raises(ValueError, lambda: JzKet(Rational(2, 3), m))  # 断言引发 ValueError 异常
    # j < 0
    raises(ValueError, lambda: JzKet(-1, 1))  # 断言引发 ValueError 异常
    raises(ValueError, lambda: JzKet(-1, m))  # 断言引发 ValueError 异常
    # m not integer or half integer
    raises(ValueError, lambda: JzKet(j, Rational(-1, 3)))  # 断言引发 ValueError 异常
    # abs(m) > j
    raises(ValueError, lambda: JzKet(1, 2))  # 断言引发 ValueError 异常
    raises(ValueError, lambda: JzKet(1, -2))  # 断言引发 ValueError 异常
    # j-m not integer
    raises(ValueError, lambda: JzKet(1, S.Half))  # 断言引发 ValueError 异常


def test_jzketcoupled():
    j, m = symbols('j m')  # 定义符号变量 j, m
    # j not integer or half integer
    raises(ValueError, lambda: JzKetCoupled(Rational(2, 3), Rational(-1, 3), (1,)))  # 断言引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(Rational(2, 3), m, (1,)))  # 断言引发 ValueError 异常
    # j < 0
    raises(ValueError, lambda: JzKetCoupled(-1, 1, (1,)))  # 断言引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(-1, m, (1,)))  # 断言引发 ValueError 异常
    # 如果 m 不是整数或半整数，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(j, Rational(-1, 3), (1,)))
    # 如果 abs(m) 大于 j，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 2, (1,)))
    raises(ValueError, lambda: JzKetCoupled(1, -2, (1,)))
    # 如果 j-m 不是整数，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, S.Half, (1,)))
    # 如果对耦合方案的类型检查失败，则应引发 TypeError 异常
    raises(TypeError, lambda: JzKetCoupled(1, 1, 1))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1,), 1))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1, 1), (1,)))
    raises(TypeError, lambda: JzKetCoupled(1, 1, (1, 1, 1), (1, 2, 1),
           (1, 3, 1)))
    # 如果耦合项的长度不符合预期，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1,), ((1, 2, 1),)))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 2),)))
    # 如果所有 jn 均不是整数或半整数，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (Rational(1, 3), Rational(2, 3))))
    # 如果耦合方案中的索引不是整数，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((S.Half, 1, 2),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, S.Half, 2),) ))
    # 如果耦合方案中的索引超出了预期范围，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((0, 2, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((3, 2, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 0, 1),) ))
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 3, 1),) ))
    # 如果耦合方案中所有 j 值不是整数或半整数，则应引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1, 1), ((1, 2, S(4)/3), (1, 3, 1)) ))
    # 每对耦合必须满足 |j1-j2| <= j3 <= j1+j2 的条件，否则引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 5)))
    raises(ValueError, lambda: JzKetCoupled(5, 1, (1, 1)))
    # 最后一个耦合的 j 值必须与状态的 j 值相等，否则引发 ValueError 异常
    raises(ValueError, lambda: JzKetCoupled(1, 1, (1, 1), ((1, 2, 2),) ))
```