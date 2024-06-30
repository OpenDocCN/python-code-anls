# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_qubit.py`

```
import random  # 导入随机数模块

from sympy.core.numbers import (Integer, Rational)  # 导入整数和有理数类
from sympy.core.singleton import S  # 导入单例模块中的 S 对象
from sympy.core.symbol import symbols  # 导入符号变量类
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类
from sympy.physics.quantum.qubit import (measure_all, measure_partial,  # 导入量子比特相关函数和类
                                         matrix_to_qubit, matrix_to_density,
                                         qubit_to_matrix, IntQubit,
                                         IntQubitBra, QubitBra)
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,  # 导入量子门类
                                        ZGate, PhaseGate)
from sympy.physics.quantum.qapply import qapply  # 导入量子操作函数
from sympy.physics.quantum.represent import represent  # 导入量子态表示函数
from sympy.physics.quantum.shor import Qubit  # 导入量子比特类
from sympy.testing.pytest import raises  # 导入用于测试的 pytest 函数
from sympy.physics.quantum.density import Density  # 导入密度矩阵类
from sympy.physics.quantum.trace import Tr  # 导入迹运算函数

x, y = symbols('x,y')  # 创建符号变量 x 和 y

epsilon = .000001  # 定义一个非常小的数作为 epsilon


def test_Qubit():
    array = [0, 0, 1, 1, 0]  # 创建一个整数数组
    qb = Qubit('00110')  # 创建一个量子比特对象
    assert qb.flip(0) == Qubit('00111')  # 测试翻转量子比特的功能
    assert qb.flip(1) == Qubit('00100')  # 测试翻转量子比特的功能
    assert qb.flip(4) == Qubit('10110')  # 测试翻转量子比特的功能
    assert qb.qubit_values == (0, 0, 1, 1, 0)  # 检查量子比特的值
    assert qb.dimension == 5  # 检查量子比特的维度
    for i in range(5):
        assert qb[i] == array[4 - i]  # 检查量子比特中每个位置的值
    assert len(qb) == 5  # 检查量子比特的长度
    qb = Qubit('110')  # 创建另一个量子比特对象


def test_QubitBra():
    qb = Qubit(0)  # 创建一个量子比特对象
    qb_bra = QubitBra(0)  # 创建一个量子比特的布拉对象
    assert qb.dual_class() == QubitBra  # 检查量子比特的对偶类是否正确
    assert qb_bra.dual_class() == Qubit  # 检查量子比特的布拉的对偶类是否正确

    qb = Qubit(1, 1, 0)  # 创建一个包含多个值的量子比特对象
    qb_bra = QubitBra(1, 1, 0)  # 创建一个对应的量子比特布拉对象
    assert represent(qb, nqubits=3).H == represent(qb_bra, nqubits=3)  # 检查表示的厄米共轭是否相等

    qb = Qubit(0, 1)  # 创建另一个量子比特对象
    qb_bra = QubitBra(1, 0)  # 创建对应的量子比特布拉对象
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(0)  # 计算内积并检查结果是否为整数

    qb_bra = QubitBra(0, 1)  # 创建另一个量子比特布拉对象
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(1)  # 计算内积并检查结果是否为整数


def test_IntQubit():
    # issue 9136
    iqb = IntQubit(0, nqubits=1)  # 创建一个整数量子比特对象
    assert qubit_to_matrix(Qubit('0')) == qubit_to_matrix(iqb)  # 检查量子比特转换为矩阵的功能是否正确

    qb = Qubit('1010')  # 创建一个量子比特对象
    assert qubit_to_matrix(IntQubit(qb)) == qubit_to_matrix(qb)  # 检查量子比特转换为矩阵的功能是否正确

    iqb = IntQubit(1, nqubits=1)  # 创建另一个整数量子比特对象
    assert qubit_to_matrix(Qubit('1')) == qubit_to_matrix(iqb)  # 检查量子比特转换为矩阵的功能是否正确
    assert qubit_to_matrix(IntQubit(1)) == qubit_to_matrix(iqb)  # 检查量子比特转换为矩阵的功能是否正确

    iqb = IntQubit(7, nqubits=4)  # 创建另一个整数量子比特对象
    assert qubit_to_matrix(Qubit('0111')) == qubit_to_matrix(iqb)  # 检查量子比特转换为矩阵的功能是否正确
    assert qubit_to_matrix(IntQubit(7, 4)) == qubit_to_matrix(iqb)  # 检查量子比特转换为矩阵的功能是否正确

    iqb = IntQubit(8)  # 创建一个整数量子比特对象
    assert iqb.as_int() == 8  # 检查整数量子比特对象的整数值
    assert iqb.qubit_values == (1, 0, 0, 0)  # 检查整数量子比特对象的值

    iqb = IntQubit(7, 4)  # 创建另一个整数量子比特对象
    assert iqb.qubit_values == (0, 1, 1, 1)  # 检查整数量子比特对象的值
    assert IntQubit(3) == IntQubit(3, 2)  # 检查两个整数量子比特对象是否相等

    # test Dual Classes
    iqb = IntQubit(3)  # 创建一个整数量子比特对象
    iqb_bra = IntQubitBra(3)  # 创建对应的整数量子比特布拉对象
    assert iqb.dual_class() == IntQubitBra  # 检查整数量子比特对象的对偶类是否正确
    assert iqb_bra.dual_class() == IntQubit  # 检查整数量子比特布拉对象的对偶类是否正确

    iqb = IntQubit(5)  # 创建一个整数量子比特对象
    iqb_bra = IntQubitBra(5)  # 创建对应的整数量子比特布拉对象
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(1)  # 计算内积并检查结果是否为整数

    iqb = IntQubit(4)  # 创建一个整数量子比特对象
    iqb_bra = IntQubitBra(5)  # 创建另一个整数量子比特布拉对象
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(0)  # 计算内积并检查结果是否为整数
    # 使用 raises 函数测试 IntQubit 类的异常情况，期望引发 ValueError 异常
    raises(ValueError, lambda: IntQubit(4, 1))
    
    # 使用 raises 函数测试 IntQubit 类的异常情况，期望引发 ValueError 异常
    raises(ValueError, lambda: IntQubit('5'))
    
    # 使用 raises 函数测试 IntQubit 类的异常情况，期望引发 ValueError 异常
    raises(ValueError, lambda: IntQubit(5, '5'))
    
    # 使用 raises 函数测试 IntQubit 类的异常情况，期望引发 ValueError 异常
    raises(ValueError, lambda: IntQubit(5, nqubits='5'))
    
    # 使用 raises 函数测试 IntQubit 类的异常情况，期望引发 TypeError 异常
    raises(TypeError, lambda: IntQubit(5, bad_arg=True))
def test_superposition_of_states():
    # 创建一个量子态，表示叠加态 1/sqrt(2) * |01⟩ + 1/sqrt(2) * |10⟩
    state = 1/sqrt(2)*Qubit('01') + 1/sqrt(2)*Qubit('10')
    # 应用门操作到量子态上，例如 CNOT(0, 1)*HadamardGate(0)*state
    state_gate = CNOT(0, 1)*HadamardGate(0)*state
    # 定义一个展开后的期望量子态，表示叠加态的展开形式
    state_expanded = Qubit('01')/2 + Qubit('00')/2 - Qubit('11')/2 + Qubit('10')/2
    # 断言应用门操作后的量子态展开结果与预期的展开量子态相等
    assert qapply(state_gate).expand() == state_expanded
    # 断言用矩阵表示门操作后的量子态与预期展开量子态相等
    assert matrix_to_qubit(represent(state_gate, nqubits=2)) == state_expanded


#test apply methods
def test_apply_represent_equality():
    # 随机选择多个量子门
    gates = [HadamardGate(int(3*random.random())),
             XGate(int(3*random.random())), ZGate(int(3*random.random())),
             YGate(int(3*random.random())), ZGate(int(3*random.random())),
             PhaseGate(int(3*random.random()))]

    # 随机生成一个初始量子比特电路
    circuit = Qubit(int(random.random()*2), int(random.random()*2),
                    int(random.random()*2), int(random.random()*2), int(random.random()*2),
                    int(random.random()*2))
    
    # 随机选择并应用多个量子门到电路上
    for i in range(int(random.random()*6)):
        circuit = gates[int(random.random()*6)]*circuit

    # 将量子电路表示为一个矩阵
    mat = represent(circuit, nqubits=6)
    # 应用电路操作后得到的量子态
    states = qapply(circuit)
    # 将矩阵表示的量子态转换为标准的量子态表示
    state_rep = matrix_to_qubit(mat)
    # 展开量子态
    states = states.expand()
    state_rep = state_rep.expand()
    # 断言标准量子态与展开后的量子态相等
    assert state_rep == states


def test_matrix_to_qubits():
    # 创建一个初始量子比特
    qb = Qubit(0, 0, 0, 0)
    # 创建一个给定矩阵的量子态表示
    mat = Matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 断言将矩阵转换为量子态后与预期量子态相等
    assert matrix_to_qubit(mat) == qb
    # 断言将量子态转换为矩阵后与预期矩阵相等
    assert qubit_to_matrix(qb) == mat

    # 创建一个复杂的量子态
    state = 2*sqrt(2)*(Qubit(0, 0, 0) + Qubit(0, 0, 1) + Qubit(0, 1, 0) +
                       Qubit(0, 1, 1) + Qubit(1, 0, 0) + Qubit(1, 0, 1) +
                       Qubit(1, 1, 0) + Qubit(1, 1, 1))
    # 创建一个预期的展开后的量子态
    ones = sqrt(2)*2*Matrix([1, 1, 1, 1, 1, 1, 1, 1])
    # 断言将矩阵转换为量子态后与预期量子态相等
    assert matrix_to_qubit(ones) == state.expand()
    # 断言将量子态转换为矩阵后与预期矩阵相等
    assert qubit_to_matrix(state) == ones


def test_measure_normalize():
    a, b = symbols('a b')
    # 创建一个含有符号参数的量子态
    state = a*Qubit('110') + b*Qubit('111')
    # 断言部分测量量子态后，不归一化时的结果与预期相等
    assert measure_partial(state, (0,), normalize=False) == \
        [(a*Qubit('110'), a*a.conjugate()), (b*Qubit('111'), b*b.conjugate())]
    # 断言完全测量量子态后，不归一化时的结果与预期相等
    assert measure_all(state, normalize=False) == \
        [(Qubit('110'), a*a.conjugate()), (Qubit('111'), b*b.conjugate())]


def test_measure_partial():
    #Basic test of collapse of entangled two qubits (Bell States)
    # 创建一个贝尔态的基本测试
    state = Qubit('01') + Qubit('10')
    # 断言部分测量量子态后的结果与预期相等
    assert measure_partial(state, (0,)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    # 断言部分测量量子态后的结果与预期相等（整数索引）
    assert measure_partial(state, int(0)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    # 断言部分测量量子态后的结果与按照不同顺序测量的结果相等
    assert measure_partial(state, (0,)) == \
        measure_partial(state, (1,))[::-1]

    #Test of more complex collapse and probability calculation
    # 创建一个更复杂的量子态的测试，包括概率计算
    state1 = sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111')
    # 断言部分测量量子态后的结果与预期相等
    assert measure_partial(state1, (0,)) == \
        [(sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111'), 1)]
    # 断言部分测量量子态后的结果与预期相等（不同位的测量）
    assert measure_partial(state1, (1, 2)) == measure_partial(state1, (3, 4))
    # 断言部分测量量子态后的结果与预期相等（多位测量）
    assert measure_partial(state1, (1, 2, 3)) == \
        [(Qubit('00001'), Rational(2, 3)), (Qubit('11111'), Rational(1, 3))]

    #test of measuring multiple bits at once
    # 创建一个四量子比特系统的初始状态
    state2 = Qubit('1111') + Qubit('1101') + Qubit('1011') + Qubit('1000')
    # 断言对部分量子态进行测量的结果
    assert measure_partial(state2, (0, 1, 3)) == \
        [(Qubit('1000'), Rational(1, 4)), (Qubit('1101'), Rational(1, 4)),
         (Qubit('1011')/sqrt(2) + Qubit('1111')/sqrt(2), S.Half)]
    # 断言对部分量子态进行仅在第0量子比特上的测量结果
    assert measure_partial(state2, (0,)) == \
        [(Qubit('1000'), Rational(1, 4)),
         (Qubit('1111')/sqrt(3) + Qubit('1101')/sqrt(3) +
          Qubit('1011')/sqrt(3), Rational(3, 4))]
# 测试函数，用于测试 measure_all 函数的不同输入情况
def test_measure_all():
    # 断言检查 measure_all 函数对 '11' 量子比特的测量结果是否为 [(Qubit('11'), 1)]
    assert measure_all(Qubit('11')) == [(Qubit('11'), 1)]
    # 创建两个量子态的叠加态 '11' + '10'
    state = Qubit('11') + Qubit('10')
    # 断言检查 measure_all 函数对叠加态的测量结果是否为 [(Qubit('10'), S.Half), (Qubit('11'), S.Half)]
    assert measure_all(state) == [(Qubit('10'), S.Half),
                                  (Qubit('11'), S.Half)]
    # 创建另一个叠加态 '11'/sqrt(5) + 2*'00'/sqrt(5)
    state2 = Qubit('11')/sqrt(5) + 2*Qubit('00')/sqrt(5)
    # 断言检查 measure_all 函数对另一个叠加态的测量结果是否为 [(Qubit('00'), Rational(4, 5)), (Qubit('11'), Rational(1, 5))]
    assert measure_all(state2) == \
        [(Qubit('00'), Rational(4, 5)), (Qubit('11'), Rational(1, 5))]

    # from issue #12585
    # 断言检查 measure_all 函数对单量子态 '0' 的测量结果是否为 [(Qubit('0'), 1)]
    assert measure_all(qapply(Qubit('0'))) == [(Qubit('0'), 1)]


# 测试函数，用于测试 eval_trace 函数的不同输入情况
def test_eval_trace():
    # 创建两个量子态 q1 和 q2
    q1 = Qubit('10110')
    q2 = Qubit('01010')
    # 创建密度矩阵 d，其中包含 q1 和 q2，并指定各自的权重
    d = Density([q1, 0.6], [q2, 0.4])

    # 计算密度矩阵的迹 t，并断言其结果为 1.0
    t = Tr(d)
    assert t.doit() == 1.0

    # 极端位的迹计算，只保留第 0 位
    t = Tr(d, 0)
    assert t.doit() == (0.4*Density([Qubit('0101'), 1]) +
                        0.6*Density([Qubit('1011'), 1]))
    # 极端位的迹计算，只保留第 4 位
    t = Tr(d, 4)
    assert t.doit() == (0.4*Density([Qubit('1010'), 1]) +
                        0.6*Density([Qubit('0110'), 1]))
    # 中间某位的迹计算，只保留第 2 位
    t = Tr(d, 2)
    assert t.doit() == (0.4*Density([Qubit('0110'), 1]) +
                        0.6*Density([Qubit('1010'), 1]))
    # 迹计算所有位
    t = Tr(d, [0, 1, 2, 3, 4])
    assert t.doit() == 1.0

    # 迹计算部分位，以非规范顺序初始化
    t = Tr(d, [2, 1, 3])
    assert t.doit() == (0.4*Density([Qubit('00'), 1]) +
                        0.6*Density([Qubit('10'), 1]))

    # 混合态的迹计算，量子态 q 的 (1/sqrt(2)) * ('00' + '11')
    q = (1/sqrt(2)) * (Qubit('00') + Qubit('11'))
    d = Density( [q, 1.0] )
    # 只保留第 0 位的迹计算
    t = Tr(d, 0)
    assert t.doit() == (0.5*Density([Qubit('0'), 1]) +
                        0.5*Density([Qubit('1'), 1]))


# 测试函数，用于测试 matrix_to_density 函数的不同输入情况
def test_matrix_to_density():
    # 创建一个 2x2 矩阵 mat
    mat = Matrix([[0, 0], [0, 1]])
    # 断言检查 matrix_to_density 函数对矩阵 mat 的转换结果是否为 Density([Qubit('1'), 1])
    assert matrix_to_density(mat) == Density([Qubit('1'), 1])

    # 创建一个 2x2 矩阵 mat
    mat = Matrix([[1, 0], [0, 0]])
    # 断言检查 matrix_to_density 函数对矩阵 mat 的转换结果是否为 Density([Qubit('0'), 1])
    assert matrix_to_density(mat) == Density([Qubit('0'), 1])

    # 创建一个 2x2 矩阵 mat
    mat = Matrix([[0, 0], [0, 0]])
    # 断言检查 matrix_to_density 函数对矩阵 mat 的转换结果是否为 0
    assert matrix_to_density(mat) == 0

    # 创建一个 4x4 矩阵 mat
    mat = Matrix([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])
    # 断言检查 matrix_to_density 函数对矩阵 mat 的转换结果是否为 Density([Qubit('10'), 1])
    assert matrix_to_density(mat) == Density([Qubit('10'), 1])

    # 创建一个 4x4 矩阵 mat
    mat = Matrix([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    # 断言检查 matrix_to_density 函数对矩阵 mat 的转换结果是否为 Density([Qubit('00'), 1])
    assert matrix_to_density(mat) == Density([Qubit('00'), 1])
```