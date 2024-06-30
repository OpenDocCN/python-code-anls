# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_qapply.py`

```
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import H, XGate, IdentityGate
from sympy.physics.quantum.operator import Operator, IdentityOperator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.spin import Jx, Jy, Jz, Jplus, Jminus, J2, JzKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.state import Ket
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.boson import BosonOp, BosonFockKet, BosonFockBra

# 定义符号变量
j, jp, m, mp = symbols("j j' m m'")

# 创建 JzKet 的实例
z = JzKet(1, 0)
po = JzKet(1, 1)
mo = JzKet(1, -1)

# 创建算子 A 的实例
A = Operator('A')

# 定义一个自定义类 Foo，继承自 Operator
class Foo(Operator):
    # 覆盖父类的 _apply_operator_JzKet 方法
    def _apply_operator_JzKet(self, ket, **options):
        return ket

# 测试函数 test_basic，验证基本量子操作的应用
def test_basic():
    # 验证 Jz 作用在 po 上的量子态变换
    assert qapply(Jz*po) == hbar*po
    # 验证 Jx 作用在 z 上的量子态变换
    assert qapply(Jx*z) == hbar*po/sqrt(2) + hbar*mo/sqrt(2)
    # 验证 (Jplus + Jminus) 作用在 z 上的量子态变换
    assert qapply((Jplus + Jminus)*z/sqrt(2)) == hbar*po + hbar*mo
    # 验证 Jz 作用在 (po + mo) 上的量子态变换
    assert qapply(Jz*(po + mo)) == hbar*po - hbar*mo
    # 验证 Jz 分别作用在 po 和 mo 上的量子态变换之和
    assert qapply(Jz*po + Jz*mo) == hbar*po - hbar*mo
    # 验证连续两次作用 Jminus 在 po 上的量子态变换
    assert qapply(Jminus*Jminus*po) == 2*hbar**2*mo
    # 验证 Jplus 的平方作用在 mo 上的量子态变换
    assert qapply(Jplus**2*mo) == 2*hbar**2*po
    # 验证 Jplus 的平方和 Jminus 的平方作用在 po 上的量子态变换
    assert qapply(Jplus**2*Jminus**2*po) == 4*hbar**4*po

# 测试函数 test_extra，验证额外的量子操作应用
def test_extra():
    # 计算 z 的对偶态与 A 和 z 的量子态外积
    extra = z.dual*A*z
    assert qapply(Jz*po*extra) == hbar*po*extra
    assert qapply(Jx*z*extra) == (hbar*po/sqrt(2) + hbar*mo/sqrt(2))*extra
    assert qapply((Jplus + Jminus)*z/sqrt(2)*extra) == hbar*po*extra + hbar*mo*extra
    assert qapply(Jz*(po + mo)*extra) == hbar*po*extra - hbar*mo*extra
    assert qapply(Jz*po*extra + Jz*mo*extra) == hbar*po*extra - hbar*mo*extra
    assert qapply(Jminus*Jminus*po*extra) == 2*hbar**2*mo*extra
    assert qapply(Jplus**2*mo*extra) == 2*hbar**2*po*extra
    assert qapply(Jplus**2*Jminus**2*po*extra) == 4*hbar**4*po*extra

# 测试函数 test_innerproduct，验证量子态的内积计算
def test_innerproduct():
    assert qapply(po.dual*Jz*po, ip_doit=False) == hbar*(po.dual*po)
    assert qapply(po.dual*Jz*po) == hbar

# 测试函数 test_zero，验证零量子态的应用
def test_zero():
    assert qapply(0) == 0
    assert qapply(Integer(0)) == 0

# 测试函数 test_commutator，验证量子力学的对易子操作
def test_commutator():
    assert qapply(Commutator(Jx, Jy)*Jz*po) == I*hbar**3*po
    assert qapply(Commutator(J2, Jz)*Jz*po) == 0
    assert qapply(Commutator(Jz, Foo('F'))*po) == 0
    assert qapply(Commutator(Foo('F'), Jz)*po) == 0

# 测试函数 test_anticommutator，验证量子力学的反对易子操作
def test_anticommutator():
    assert qapply(AntiCommutator(Jz, Foo('F'))*po) == 2*hbar*po
    assert qapply(AntiCommutator(Foo('F'), Jz)*po) == 2*hbar*po

# 测试函数 test_outerproduct，验证量子态的外积操作
def test_outerproduct():
    e = Jz*(mo*po.dual)*Jz*po
    assert qapply(e) == -hbar**2*mo
    assert qapply(e, ip_doit=False) == -hbar**2*(po.dual*po)*mo
    # 使用断言验证 qapply(e).doit() 的计算结果是否等于 -hbar**2*mo
    assert qapply(e).doit() == -hbar**2*mo
def test_tensorproduct():
    # 创建玻色子算符"a"和"b"
    a = BosonOp("a")
    b = BosonOp("b")
    # 创建张量积态，例如 |1⟩⨂|2⟩
    ket1 = TensorProduct(BosonFockKet(1), BosonFockKet(2))
    # 创建另一个张量积态，例如 |0⟩⨂|0⟩
    ket2 = TensorProduct(BosonFockKet(0), BosonFockKet(0))
    # 创建第三个张量积态，例如 |0⟩⨂|2⟩
    ket3 = TensorProduct(BosonFockKet(0), BosonFockKet(2))
    # 创建张量积态的共轭转置，例如 ⟨0|⨂⟨0|
    bra1 = TensorProduct(BosonFockBra(0), BosonFockBra(0))
    # 创建另一个张量积态的共轭转置，例如 ⟨1|⨂⟨2|
    bra2 = TensorProduct(BosonFockBra(1), BosonFockBra(2))
    # 断言验证量子操作在张量积态上的作用结果
    assert qapply(TensorProduct(a, b ** 2) * ket1) == sqrt(2) * ket2
    assert qapply(TensorProduct(a, Dagger(b) * b) * ket1) == 2 * ket3
    assert qapply(bra1 * TensorProduct(a, b * b),
                  dagger=True) == sqrt(2) * bra2
    assert qapply(bra2 * ket1).doit() == TensorProduct(1, 1)
    assert qapply(TensorProduct(a, b * b) * ket1) == sqrt(2) * ket2
    assert qapply(Dagger(TensorProduct(a, b * b) * ket1),
                  dagger=True) == sqrt(2) * Dagger(ket2)


def test_dagger():
    # 创建双态的共轭转置操作
    lhs = Dagger(Qubit(0))*Dagger(H(0))
    # 创建双态的线性组合
    rhs = Dagger(Qubit(1))/sqrt(2) + Dagger(Qubit(0))/sqrt(2)
    # 断言验证量子操作在共轭转置下的作用结果
    assert qapply(lhs, dagger=True) == rhs


def test_issue_6073():
    # 创建符号变量
    x, y = symbols('x y', commutative=False)
    # 创建符号态和算符
    A = Ket(x, y)
    B = Operator('B')
    # 断言验证量子操作的应用结果
    assert qapply(A) == A
    assert qapply(A.dual*B) == A.dual*B


def test_density():
    # 创建密度矩阵
    d = Density([Jz*mo, 0.5], [Jz*po, 0.5])
    # 断言验证密度矩阵的量子操作结果
    assert qapply(d) == Density([-hbar*mo, 0.5], [hbar*po, 0.5])


def test_issue3044():
    # 创建张量积态表达式
    expr1 = TensorProduct(Jz*JzKet(S(2),S.NegativeOne)/sqrt(2), Jz*JzKet(S.Half,S.Half))
    # 创建预期的乘积表达式
    result = Mul(S.NegativeOne, Rational(1, 4), 2**S.Half, hbar**2)
    result *= TensorProduct(JzKet(2,-1), JzKet(S.Half,S.Half))
    # 断言验证量子操作在张量积态表达式上的应用结果
    assert qapply(expr1) == result


# Issue 24158: Tests whether qapply incorrectly evaluates some ket*op as op*ket
def test_issue24158_ket_times_op():
    # 创建不明确的项
    P = BosonFockKet(0) * BosonOp("a") # undefined term
    # 断言验证在修复之前的量子操作结果
    assert qapply(P) == P   # qapply(P) -> BosonOp("a")*BosonFockKet(0) = 0 before fix
    P = Qubit(1) * XGate(0) # undefined term
    # 断言验证在修复之前的量子操作结果
    assert qapply(P) == P   # qapply(P) -> Qubit(0) before fix
    P1 = Mul(QubitBra(0), Mul(QubitBra(0), Qubit(0)), XGate(0)) # legal expr <0| * (<1|*|1>) * X
    # 断言验证在修复之前的量子操作结果
    assert qapply(P1) == QubitBra(0) * XGate(0)     # qapply(P1) -> 0 before fix
    P1 = qapply(P1, dagger = True)  # unsatisfactorily -> <0|*X(0), expect <1| since dagger=True
    # 断言验证在修复之前的量子操作结果
    assert qapply(P1, dagger = True) == QubitBra(1) # qapply(P1, dagger=True) -> 0 before fix
    P2 = QubitBra(0) * QubitBra(0) * Qubit(0) * XGate(0) # 'forgot' to set brackets
    P2 = qapply(P2, dagger = True) # unsatisfactorily -> <0|*X(0), expect <1| since dagger=True
    # 断言验证在修复之前的量子操作结果
    assert qapply(P2, dagger = True) == QubitBra(1) # qapply(P1) -> 0 before fix
    # Pull Request 24237: IdentityOperator from the right without dagger=True option
    assert qapply(QubitBra(1)*IdentityOperator()) == QubitBra(1)
    assert qapply(IdentityGate(0)*(Qubit(0) + Qubit(1))) == Qubit(0) + Qubit(1)
```