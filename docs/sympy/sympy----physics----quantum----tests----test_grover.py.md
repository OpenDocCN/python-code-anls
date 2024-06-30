# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_grover.py`

```
# 导入从Sympy库中引入特定函数和类

from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (
    apply_grover,              # 导入Grover算法中的应用函数
    superposition_basis,       # 导入Grover算法中的超position基函数
    OracleGate,                # 导入Grover算法中的Oracle门类
    grover_iteration,          # 导入Grover算法中的迭代函数
    WGate                      # 导入Grover算法中的W门类
)


def return_one_on_two(qubits):
    return qubits == IntQubit(2, qubits.nqubits)


def return_one_on_one(qubits):
    return qubits == IntQubit(1, nqubits=qubits.nqubits)


def test_superposition_basis():
    # 测试超position_basis函数的行为，第一个测试用例是nbits=2的情况
    nbits = 2
    first_half_state = IntQubit(0, nqubits=nbits)/2 + IntQubit(1, nqubits=nbits)/2
    second_half_state = IntQubit(2, nbits)/2 + IntQubit(3, nbits)/2
    assert first_half_state + second_half_state == superposition_basis(nbits)

    # 第二个测试用例是nbits=3的情况
    nbits = 3
    firstq = (1/sqrt(8))*IntQubit(0, nqubits=nbits) + (1/sqrt(8))*IntQubit(1, nqubits=nbits)
    secondq = (1/sqrt(8))*IntQubit(2, nbits) + (1/sqrt(8))*IntQubit(3, nbits)
    thirdq = (1/sqrt(8))*IntQubit(4, nbits) + (1/sqrt(8))*IntQubit(5, nbits)
    fourthq = (1/sqrt(8))*IntQubit(6, nbits) + (1/sqrt(8))*IntQubit(7, nbits)
    assert firstq + secondq + thirdq + fourthq == superposition_basis(nbits)


def test_OracleGate():
    # 测试OracleGate类的行为
    v = OracleGate(1, lambda qubits: qubits == IntQubit(0))
    assert qapply(v*IntQubit(0)) == -IntQubit(0)
    assert qapply(v*IntQubit(1)) == IntQubit(1)

    # 使用nbits=2测试返回函数return_one_on_two的OracleGate行为
    nbits = 2
    v = OracleGate(2, return_one_on_two)
    assert qapply(v*IntQubit(0, nbits)) == IntQubit(0, nqubits=nbits)
    assert qapply(v*IntQubit(1, nbits)) == IntQubit(1, nqubits=nbits)
    assert qapply(v*IntQubit(2, nbits)) == -IntQubit(2, nbits)
    assert qapply(v*IntQubit(3, nbits)) == IntQubit(3, nbits)

    # 验证OracleGate(1, lambda qubits: qubits == IntQubit(0))的矩阵表示
    assert represent(OracleGate(1, lambda qubits: qubits == IntQubit(0)), nqubits=1) == \
           Matrix([[-1, 0], [0, 1]])
    # 验证返回函数return_one_on_two的OracleGate(2)的矩阵表示
    assert represent(v, nqubits=2) == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def test_WGate():
    # 测试WGate类的行为
    nqubits = 2
    basis_states = superposition_basis(nqubits)
    assert qapply(WGate(nqubits)*basis_states) == basis_states

    # 使用nqubits=2测试WGate的行为
    expected = ((2/sqrt(pow(2, nqubits)))*basis_states) - IntQubit(1, nqubits=nqubits)
    assert qapply(WGate(nqubits)*IntQubit(1, nqubits=nqubits)) == expected


def test_grover_iteration_1():
    # 测试第一个Grover迭代函数的行为
    numqubits = 2
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_one)
    expected = IntQubit(1, nqubits=numqubits)
    assert qapply(grover_iteration(basis_states, v)) == expected


def test_grover_iteration_2():
    # 测试第二个Grover迭代函数的行为
    numqubits = 4
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_two)
    # 在大约pi倍sqrt(2^n)次迭代后，IntQubit(2)应具有最高的概率
    # 在此情况下，大约在pi倍（3或4）次迭代后
    iterated = grover_iteration(basis_states, v)
    iterated = qapply(iterated)
    iterated = grover_iteration(iterated, v)
    iterated = qapply(iterated)
    # 使用 grover_iteration 函数对 iterated 进行一次 Grover 迭代操作，并将结果重新赋给 iterated
    iterated = grover_iteration(iterated, v)
    # 使用 qapply 函数对 iterated 应用量子操作
    iterated = qapply(iterated)
    # 在这种情况下，经过3次迭代后，Qubit('0010')的概率最高
    # Qubit('0010')的概率为 251/256 (3次迭代) vs 781/1024 (4次迭代)
    # 询问有关测量的问题
    # 计算期望值，这里基于量子位的数学运算
    expected = (-13*basis_states)/64 + 264*IntQubit(2, numqubits)/256
    # 断言应用量子操作后的结果等于 iterated
    assert qapply(expected) == iterated
# 定义测试函数 test_grover
def test_grover():
    # 设置量子比特数为2，验证应用 Grover 算法后返回值为 IntQubit(1, nqubits=2)
    nqubits = 2
    assert apply_grover(return_one_on_one, nqubits) == IntQubit(1, nqubits=nqubits)

    # 设置量子比特数为4，生成超位置基态
    nqubits = 4
    basis_states = superposition_basis(nqubits)
    # 计算期望值，使用超位置基态和常数值 264 乘以 IntQubit(2, nqubits) 的比例
    expected = (-13 * basis_states) / 64 + 264 * IntQubit(2, nqubits) / 256
    # 验证应用 Grover 算法后返回值等于期望的量子态
    assert apply_grover(return_one_on_two, 4) == qapply(expected)
```