# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_qft.py`

```
# 从 sympy 库中导入各种符号和函数
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix

# 从 sympy.physics.quantum.qft 模块导入量子傅里叶变换相关的类和函数
from sympy.physics.quantum.qft import QFT, IQFT, RkGate
# 从 sympy.physics.quantum.gate 模块导入各种量子门类
from sympy.physics.quantum.gate import (ZGate, SwapGate, HadamardGate, CGate,
                                        PhaseGate, TGate)
# 从 sympy.physics.quantum.qubit 模块导入量子比特相关的类
from sympy.physics.quantum.qubit import Qubit
# 从 sympy.physics.quantum.qapply 模块导入 qapply 函数
from sympy.physics.quantum.qapply import qapply
# 从 sympy.physics.quantum.represent 模块导入 represent 函数
from sympy.physics.quantum.represent import represent

# 从 sympy.functions.elementary.complexes 模块导入 sign 函数
from sympy.functions.elementary.complexes import sign


# 定义测试函数 test_RkGate，用于测试 RkGate 类的功能
def test_RkGate():
    # 定义符号变量 x
    x = Symbol('x')
    # 断言语句，测试 RkGate 对象的 k 属性是否为 x
    assert RkGate(1, x).k == x
    # 断言语句，测试 RkGate 对象的 targets 属性是否为 (1,)
    assert RkGate(1, x).targets == (1,)
    # 断言语句，测试 RkGate(1, 1) 是否等于 ZGate(1)
    assert RkGate(1, 1) == ZGate(1)
    # 断言语句，测试 RkGate(2, 2) 是否等于 PhaseGate(2)
    assert RkGate(2, 2) == PhaseGate(2)
    # 断言语句，测试 RkGate(3, 3) 是否等于 TGate(3)
    assert RkGate(3, 3) == TGate(3)

    # 断言语句，测试 represent 函数对 RkGate(0, x) 的表达式
    assert represent(
        RkGate(0, x), nqubits=1) == Matrix([[1, 0], [0, exp(sign(x)*2*pi*I/(2**abs(x)))]])


# 定义测试函数 test_quantum_fourier，用于测试量子傅里叶变换相关函数的功能
def test_quantum_fourier():
    # 断言语句，测试 QFT(0, 3) 的分解结果
    assert QFT(0, 3).decompose() == \
        SwapGate(0, 2)*HadamardGate(0)*CGate((0,), PhaseGate(1)) * \
        HadamardGate(1)*CGate((0,), TGate(2))*CGate((1,), PhaseGate(2)) * \
        HadamardGate(2)

    # 断言语句，测试 IQFT(0, 3) 的分解结果
    assert IQFT(0, 3).decompose() == \
        HadamardGate(2)*CGate((1,), RkGate(2, -2))*CGate((0,), RkGate(2, -3)) * \
        HadamardGate(1)*CGate((0,), RkGate(1, -2))*HadamardGate(0)*SwapGate(0, 2)

    # 断言语句，测试 represent 函数对 QFT(0, 3) 的表示
    assert represent(QFT(0, 3), nqubits=3) == \
        Matrix([[exp(2*pi*I/8)**(i*j % 8)/sqrt(8) for i in range(8)] for j in range(8)])

    # 断言语句，测试 QFT(0, 4) 的分解结果，表示其具有非平凡的分解方式
    assert QFT(0, 4).decompose()

    # 断言语句，测试 qapply 函数对量子态的作用
    assert qapply(QFT(0, 3).decompose()*Qubit(0, 0, 0)).expand() == qapply(
        HadamardGate(0)*HadamardGate(1)*HadamardGate(2)*Qubit(0, 0, 0)
    ).expand()


# 定义测试函数 test_qft_represent，用于测试量子傅里叶变换的表示函数的功能
def test_qft_represent():
    # 创建 QFT(0, 3) 的对象 c
    c = QFT(0, 3)
    # 计算 c 和其分解形式的表示矩阵 a 和 b，并进行断言比较它们的数值计算结果
    a = represent(c, nqubits=3)
    b = represent(c.decompose(), nqubits=3)
    assert a.evalf(n=10) == b.evalf(n=10)
```