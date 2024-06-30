# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_pauli.py`

```
# 从 sympy 库导入特定模块和类
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
                                   Operator, represent)
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
                                         SigmaMinus, SigmaPlus,
                                         qsimplify_pauli)
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises

# 定义 SigmaX, SigmaY, SigmaZ 算符对象
sx, sy, sz = SigmaX(), SigmaY(), SigmaZ()
sx1, sy1, sz1 = SigmaX(1), SigmaY(1), SigmaZ(1)
sx2, sy2, sz2 = SigmaX(2), SigmaY(2), SigmaZ(2)

# 定义 SigmaMinus 和 SigmaPlus 算符对象
sm, sp = SigmaMinus(), SigmaPlus()
sm1, sp1 = SigmaMinus(1), SigmaPlus(1)
A, B = Operator("A"), Operator("B")

# 定义测试函数：验证 Pauli 算符的类型
def test_pauli_operators_types():
    assert isinstance(sx, SigmaOpBase) and isinstance(sx, SigmaX)
    assert isinstance(sy, SigmaOpBase) and isinstance(sy, SigmaY)
    assert isinstance(sz, SigmaOpBase) and isinstance(sz, SigmaZ)
    assert isinstance(sm, SigmaOpBase) and isinstance(sm, SigmaMinus)
    assert isinstance(sp, SigmaOpBase) and isinstance(sp, SigmaPlus)

# 定义测试函数：验证 Pauli 算符的对易子
def test_pauli_operators_commutator():
    assert Commutator(sx, sy).doit() == 2 * I * sz
    assert Commutator(sy, sz).doit() == 2 * I * sx
    assert Commutator(sz, sx).doit() == 2 * I * sy

# 定义测试函数：验证带有标签的 Pauli 算符的对易子
def test_pauli_operators_commutator_with_labels():
    assert Commutator(sx1, sy1).doit() == 2 * I * sz1
    assert Commutator(sy1, sz1).doit() == 2 * I * sx1
    assert Commutator(sz1, sx1).doit() == 2 * I * sy1

    assert Commutator(sx2, sy2).doit() == 2 * I * sz2
    assert Commutator(sy2, sz2).doit() == 2 * I * sx2
    assert Commutator(sz2, sx2).doit() == 2 * I * sy2

    assert Commutator(sx1, sy2).doit() == 0
    assert Commutator(sy1, sz2).doit() == 0
    assert Commutator(sz1, sx2).doit() == 0

# 定义测试函数：验证 Pauli 算符的反对易子
def test_pauli_operators_anticommutator():
    assert AntiCommutator(sy, sz).doit() == 0
    assert AntiCommutator(sz, sx).doit() == 0
    assert AntiCommutator(sx, sm).doit() == 1
    assert AntiCommutator(sx, sp).doit() == 1

# 定义测试函数：验证 Pauli 算符的共轭转置
def test_pauli_operators_adjoint():
    assert Dagger(sx) == sx
    assert Dagger(sy) == sy
    assert Dagger(sz) == sz

# 定义测试函数：验证带有标签的 Pauli 算符的共轭转置
def test_pauli_operators_adjoint_with_labels():
    assert Dagger(sx1) == sx1
    assert Dagger(sy1) == sy1
    assert Dagger(sz1) == sz1

    assert Dagger(sx1) != sx2
    assert Dagger(sy1) != sy2
    assert Dagger(sz1) != sz2

# 定义测试函数：验证 Pauli 算符的乘法
def test_pauli_operators_multiplication():
    assert qsimplify_pauli(sx * sx) == 1
    assert qsimplify_pauli(sy * sy) == 1
    assert qsimplify_pauli(sz * sz) == 1

    assert qsimplify_pauli(sx * sy) == I * sz
    assert qsimplify_pauli(sy * sz) == I * sx
    assert qsimplify_pauli(sz * sx) == I * sy

    assert qsimplify_pauli(sy * sx) == - I * sz
    assert qsimplify_pauli(sz * sy) == - I * sx
    assert qsimplify_pauli(sx * sz) == - I * sy
def test_pauli_operators_multiplication_with_labels():

    # 断言：简化保利算符的乘积 sx1 * sx1 结果为 1
    assert qsimplify_pauli(sx1 * sx1) == 1
    # 断言：简化保利算符的乘积 sy1 * sy1 结果为 1
    assert qsimplify_pauli(sy1 * sy1) == 1
    # 断言：简化保利算符的乘积 sz1 * sz1 结果为 1
    assert qsimplify_pauli(sz1 * sz1) == 1

    # 断言：检查 sx1 * sx2 的乘积是否为 Mul 类型
    assert isinstance(sx1 * sx2, Mul)
    # 断言：检查 sy1 * sy2 的乘积是否为 Mul 类型
    assert isinstance(sy1 * sy2, Mul)
    # 断言：检查 sz1 * sz2 的乘积是否为 Mul 类型
    assert isinstance(sz1 * sz2, Mul)

    # 断言：简化保利算符的乘积 sx1 * sy1 * sx2 * sy2 的结果是否为 - sz1 * sz2
    assert qsimplify_pauli(sx1 * sy1 * sx2 * sy2) == - sz1 * sz2
    # 断言：简化保利算符的乘积 sy1 * sz1 * sz2 * sx2 的结果是否为 - sx1 * sy2
    assert qsimplify_pauli(sy1 * sz1 * sz2 * sx2) == - sx1 * sy2


def test_pauli_states():
    sx, sz = SigmaX(), SigmaZ()

    up = SigmaZKet(0)
    down = SigmaZKet(1)

    # 断言：应用 sx * up 后得到 down
    assert qapply(sx * up) == down
    # 断言：应用 sx * down 后得到 up
    assert qapply(sx * down) == up
    # 断言：应用 sz * up 后得到 up
    assert qapply(sz * up) == up
    # 断言：应用 sz * down 后得到 - down
    assert qapply(sz * down) == - down

    up = SigmaZBra(0)
    down = SigmaZBra(1)

    # 断言：应用 up * sx 的共轭转置后得到 down
    assert qapply(up * sx, dagger=True) == down
    # 断言：应用 down * sx 的共轭转置后得到 up
    assert qapply(down * sx, dagger=True) == up
    # 断言：应用 up * sz 的共轭转置后得到 up
    assert qapply(up * sz, dagger=True) == up
    # 断言：应用 down * sz 的共轭转置后得到 - down
    assert qapply(down * sz, dagger=True) == - down

    # 断言：SigmaZKet(0) 的共轭转置应为 SigmaZBra(0)
    assert Dagger(SigmaZKet(0)) == SigmaZBra(0)
    # 断言：SigmaZBra(1) 的共轭转置应为 SigmaZKet(1)
    assert Dagger(SigmaZBra(1)) == SigmaZKet(1)
    # 断言：SigmaZBra(2) 应引发 ValueError 异常
    raises(ValueError, lambda: SigmaZBra(2))
    # 断言：SigmaZKet(2) 应引发 ValueError 异常
    raises(ValueError, lambda: SigmaZKet(2))


def test_use_name():
    # 断言：sm.use_name 应为 False
    assert sm.use_name is False
    # 断言：sm1.use_name 应为 True
    assert sm1.use_name is True
    # 断言：sx.use_name 应为 False
    assert sx.use_name is False
    # 断言：sx1.use_name 应为 True
    assert sx1.use_name is True


def test_printing():
    # 断言：latex(sx) 应为 r'{\sigma_x}'
    assert latex(sx) == r'{\sigma_x}'
    # 断言：latex(sx1) 应为 r'{\sigma_x^{(1)}}'
    assert latex(sx1) == r'{\sigma_x^{(1)}}'
    # 断言：latex(sy) 应为 r'{\sigma_y}'
    assert latex(sy) == r'{\sigma_y}'
    # 断言：latex(sy1) 应为 r'{\sigma_y^{(1)}}'
    assert latex(sy1) == r'{\sigma_y^{(1)}}'
    # 断言：latex(sz) 应为 r'{\sigma_z}'
    assert latex(sz) == r'{\sigma_z}'
    # 断言：latex(sz1) 应为 r'{\sigma_z^{(1)}}'
    assert latex(sz1) == r'{\sigma_z^{(1)}}'
    # 断言：latex(sm) 应为 r'{\sigma_-}'
    assert latex(sm) == r'{\sigma_-}'
    # 断言：latex(sm1) 应为 r'{\sigma_-^{(1)}}'
    assert latex(sm1) == r'{\sigma_-^{(1)}}'
    # 断言：latex(sp) 应为 r'{\sigma_+}'
    assert latex(sp) == r'{\sigma_+}'
    # 断言：latex(sp1) 应为 r'{\sigma_+^{(1)}}'
    assert latex(sp1) == r'{\sigma_+^{(1)}}'


def test_represent():
    # 断言：represent(sx) 应为 Matrix([[0, 1], [1, 0]])
    assert represent(sx) == Matrix([[0, 1], [1, 0]])
    # 断言：represent(sy) 应为 Matrix([[0, -I], [I, 0]])
    assert represent(sy) == Matrix([[0, -I], [I, 0]])
    # 断言：represent(sz) 应为 Matrix([[1, 0], [0, -1]])
    assert represent(sz) == Matrix([[1, 0], [0, -1]])
    # 断言：represent(sm) 应为 Matrix([[0, 0], [1, 0]])
    assert represent(sm) == Matrix([[0, 0], [1, 0]])
    # 断言：represent(sp) 应为 Matrix([[0, 1], [0, 0]])
    assert represent(sp) == Matrix([[0, 1], [0, 0]])
```