# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_gate.py`

```
from sympy.core.mul import Mul  # 导入 sympy 中的 Mul 类
from sympy.core.numbers import (I, Integer, Rational, pi)  # 导入 sympy 中的数学常数和数字类型
from sympy.core.symbol import (Wild, symbols)  # 导入 sympy 中的通配符和符号类型
from sympy.functions.elementary.exponential import exp  # 导入 sympy 中的指数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sympy 中的平方根函数
from sympy.matrices import Matrix, ImmutableMatrix  # 导入 sympy 中的矩阵类型和不可变矩阵类型

from sympy.physics.quantum.gate import (XGate, YGate, ZGate, random_circuit,  # 导入量子门类和函数
        CNOT, IdentityGate, H, X, Y, S, T, Z, SwapGate, gate_simp, gate_sort,
        CNotGate, TGate, HadamardGate, PhaseGate, UGate, CGate)
from sympy.physics.quantum.commutator import Commutator  # 导入量子力学中的对易子类
from sympy.physics.quantum.anticommutator import AntiCommutator  # 导入量子力学中的反对易子类
from sympy.physics.quantum.represent import represent  # 导入表示函数
from sympy.physics.quantum.qapply import qapply  # 导入量子应用函数
from sympy.physics.quantum.qubit import Qubit, IntQubit, qubit_to_matrix, \  # 导入量子比特类和相关函数
    matrix_to_qubit
from sympy.physics.quantum.matrixutils import matrix_to_zero  # 导入矩阵工具函数
from sympy.physics.quantum.matrixcache import sqrt2_inv  # 导入矩阵缓存函数
from sympy.physics.quantum import Dagger  # 导入伴随操作类


def test_gate():
    """测试基本的量子门功能."""
    # 创建一个 Hadamard 门实例 h
    h = HadamardGate(1)
    # 断言 Hadamard 门的最小量子比特数为 2
    assert h.min_qubits == 2
    # 断言 Hadamard 门的量子比特数为 1
    assert h.nqubits == 1

    # 创建通配符实例 i0 和 i1
    i0 = Wild('i0')
    i1 = Wild('i1')
    # 创建两个相同的 Hadamard 门实例
    h0_w1 = HadamardGate(i0)
    h0_w2 = HadamardGate(i0)
    # 创建一个不同的 Hadamard 门实例
    h1_w1 = HadamardGate(i1)

    # 断言两个相同的 Hadamard 门实例相等
    assert h0_w1 == h0_w2
    # 断言一个 Hadamard 门实例与另一个不同的 Hadamard 门实例不相等
    assert h0_w1 != h1_w1
    # 断言另外两个 Hadamard 门实例也不相等
    assert h1_w1 != h0_w2

    # 创建 CNOT 门实例 cnot_10_w1 和 cnot_10_w2
    cnot_10_w1 = CNOT(i1, i0)
    cnot_10_w2 = CNOT(i1, i0)
    # 创建一个不同的 CNOT 门实例 cnot_01_w1
    cnot_01_w1 = CNOT(i0, i1)

    # 断言两个相同的 CNOT 门实例相等
    assert cnot_10_w1 == cnot_10_w2
    # 断言一个 CNOT 门实例与另一个不同的 CNOT 门实例不相等
    assert cnot_10_w1 != cnot_01_w1
    # 断言另外两个 CNOT 门实例也不相等
    assert cnot_10_w2 != cnot_01_w1


def test_UGate():
    """测试 U 门的功能."""
    # 创建符号 a, b, c, d
    a, b, c, d = symbols('a,b,c,d')
    # 创建一个 2x2 的矩阵 uMat
    uMat = Matrix([[a, b], [c, d]])

    # 在 1 量子比特空间中测试基本情况下门存在的情况
    u1 = UGate((0,), uMat)
    # 断言在 nqubits=1 的情况下表示 u1 的矩阵与 uMat 相等
    assert represent(u1, nqubits=1) == uMat
    # 断言对 Qubit('0') 应用 u1 门后的结果
    assert qapply(u1*Qubit('0')) == a*Qubit('0') + c*Qubit('1')
    # 断言对 Qubit('1') 应用 u1 门后的结果
    assert qapply(u1*Qubit('1')) == b*Qubit('0') + d*Qubit('1')

    # 在更大空间中测试门存在的情况
    u2 = UGate((1,), uMat)
    # 计算表示 u2 在 2 量子比特空间中的矩阵
    u2Rep = represent(u2, nqubits=2)
    # 断言对所有 4 种初始量子比特状态的应用结果
    for i in range(4):
        assert u2Rep*qubit_to_matrix(IntQubit(i, 2)) == \
            qubit_to_matrix(qapply(u2*IntQubit(i, 2)))


def test_cgate():
    """测试通用的 CGate."""
    # 测试单控制比特功能
    CNOTMatrix = Matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert represent(CGate(1, XGate(0)), nqubits=2) == CNOTMatrix

    # 测试多控制比特功能
    ToffoliGate = CGate((1, 2), XGate(0))
    assert represent(ToffoliGate, nqubits=3) == \
        Matrix(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0,
        1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]])

    ToffoliGate = CGate((3, 0), XGate(1))
    assert qapply(ToffoliGate*Qubit('1001')) == \
        matrix_to_qubit(represent(ToffoliGate*Qubit('1001'), nqubits=4))
    # 断言测试 Toffoli 门作用于 Qubit('0000') 后的结果等于其在矩阵表示下的量子态
    assert qapply(ToffoliGate*Qubit('0000')) == \
        matrix_to_qubit(represent(ToffoliGate*Qubit('0000'), nqubits=4))

    # 创建一个控制-Y门，控制位为第1位，目标为YGate(0)，并生成其矩阵表示
    CYGate = CGate(1, YGate(0))
    CYGate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, -I), (0, 0, I, 0)))
    # 断言测试 2量子比特控制-Y门的分解方法得到的矩阵表示与预期的CYGate_matrix相等
    assert represent(CYGate.decompose(), nqubits=2) == CYGate_matrix

    # 创建一个控制-Z门，控制位为第0位，目标为ZGate(1)，并生成其矩阵表示
    CZGate = CGate(0, ZGate(1))
    CZGate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1)))
    # 断言测试控制-Z门作用于 Qubit('11') 后的结果等于其取负值
    assert qapply(CZGate*Qubit('11')) == -Qubit('11')
    # 断言测试 2量子比特控制-Z门的分解方法得到的矩阵表示与预期的CZGate_matrix相等
    assert represent(CZGate.decompose(), nqubits=2) == CZGate_matrix

    # 创建一个控制相位门，控制位为第0位，目标为PhaseGate(1)
    CPhaseGate = CGate(0, PhaseGate(1))
    # 断言测试控制相位门作用于 Qubit('11') 后的结果等于其乘以虚数单位
    assert qapply(CPhaseGate*Qubit('11')) == \
        I*Qubit('11')
    # 断言测试 2量子比特控制相位门的分解方法得到的矩阵表示与其自身的矩阵表示不相等（因为它不是自逆门）
    assert matrix_to_qubit(represent(CPhaseGate*Qubit('11'), nqubits=2)) == \
        I*Qubit('11')

    # 断言测试控制-Z门的共轭转置（Dagger）、逆运算和幂运算是否正确评估
    assert Dagger(CZGate) == CZGate
    assert pow(CZGate, 1) == Dagger(CZGate)
    assert Dagger(CZGate) == CZGate.inverse()
    # 断言测试控制相位门的共轭转置（Dagger）与逆运算是否相等
    assert Dagger(CPhaseGate) != CPhaseGate
    assert Dagger(CPhaseGate) == CPhaseGate.inverse()
    assert Dagger(CPhaseGate) == pow(CPhaseGate, -1)
    assert pow(CPhaseGate, -1) == CPhaseGate.inverse()
def test_UGate_CGate_combo():
    # 导入符号变量库，定义符号变量
    a, b, c, d = symbols('a,b,c,d')
    # 创建一个 2x2 的矩阵 uMat
    uMat = Matrix([[a, b], [c, d]])
    # 创建一个 4x4 的矩阵 cMat
    cMat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, c, d]])

    # 创建 UGate 对象 u1 和 CGate 对象 cu1，针对 1-qubit 空间的基本测试
    u1 = UGate((0,), uMat)
    cu1 = CGate(1, u1)
    # 断言 cu1 的表示与预期的 cMat 相等
    assert represent(cu1, nqubits=2) == cMat
    # 断言应用 cu1 到 Qubit('10') 的结果
    assert qapply(cu1*Qubit('10')) == a*Qubit('10') + c*Qubit('11')
    assert qapply(cu1*Qubit('11')) == b*Qubit('10') + d*Qubit('11')
    assert qapply(cu1*Qubit('01')) == Qubit('01')
    assert qapply(cu1*Qubit('00')) == Qubit('00')

    # 创建 UGate 对象 u2，针对更大空间的测试
    u2 = UGate((1,), uMat)
    u2Rep = represent(u2, nqubits=2)
    # 遍历并断言 u2 的表示与 qapply 的结果对 IntQubit(i, 2) 相等
    for i in range(4):
        assert u2Rep * qubit_to_matrix(IntQubit(i, 2)) == \
            qubit_to_matrix(qapply(u2 * IntQubit(i, 2)))


def test_UGate_OneQubitGate_combo():
    # 导入符号变量库，定义符号变量
    v, w, f, g = symbols('v w f g')
    # 创建一个不可变的 2x2 矩阵 uMat1
    uMat1 = ImmutableMatrix([[v, w], [f, g]])
    # 创建一个 4x4 的矩阵 cMat1
    cMat1 = Matrix([[v, w + 1, 0, 0], [f + 1, g, 0, 0], [0, 0, v, w + 1], [0, 0, f + 1, g]])
    # 创建 UGate 对象 u1
    u1 = X(0) + UGate(0, uMat1)
    # 断言 u1 的表示与预期的 cMat1 相等
    assert represent(u1, nqubits=2) == cMat1

    # 创建一个不可变的 2x2 矩阵 uMat2
    uMat2 = ImmutableMatrix([[1/sqrt(2), 1/sqrt(2)], [I/sqrt(2), -I/sqrt(2)]])
    # 创建两个预期的 2x2 矩阵 cMat2_1 和 cMat2_2
    cMat2_1 = Matrix([[Rational(1, 2) + I/2, Rational(1, 2) - I/2],
                      [Rational(1, 2) - I/2, Rational(1, 2) + I/2]])
    cMat2_2 = Matrix([[1, 0], [0, I]])
    # 创建 UGate 对象 u2
    u2 = UGate(0, uMat2)
    # 断言 H(0)*u2 和 u2*H(0) 的表示分别与预期的 cMat2_1 和 cMat2_2 相等
    assert represent(H(0)*u2, nqubits=1) == cMat2_1
    assert represent(u2*H(0), nqubits=1) == cMat2_2


def test_represent_hadamard():
    """Test the representation of the hadamard gate."""
    # 创建 HadamardGate 对象和初始量子比特状态
    circuit = HadamardGate(0) * Qubit('00')
    # 计算表示结果
    answer = represent(circuit, nqubits=2)
    # 断言表示结果与预期的矩阵相等，使用了 sqrt2_inv 变量
    assert answer == Matrix([sqrt2_inv, sqrt2_inv, 0, 0])


def test_represent_xgate():
    """Test the representation of the X gate."""
    # 创建 XGate 对象和初始量子比特状态
    circuit = XGate(0) * Qubit('00')
    # 计算表示结果
    answer = represent(circuit, nqubits=2)
    # 断言表示结果与预期的矩阵相等
    assert Matrix([0, 1, 0, 0]) == answer


def test_represent_ygate():
    """Test the representation of the Y gate."""
    # 创建 YGate 对象和初始量子比特状态
    circuit = YGate(0) * Qubit('00')
    # 计算表示结果
    answer = represent(circuit, nqubits=2)
    # 断言表示结果中的元素与预期的值相等，使用了复数单位 I
    assert answer[0] == 0 and answer[1] == I and \
        answer[2] == 0 and answer[3] == 0


def test_represent_zgate():
    """Test the representation of the Z gate."""
    # 创建 ZGate 对象和初始量子比特状态
    circuit = ZGate(0) * Qubit('00')
    # 计算表示结果
    answer = represent(circuit, nqubits=2)
    # 断言表示结果与预期的矩阵相等
    assert Matrix([1, 0, 0, 0]) == answer


def test_represent_phasegate():
    """Test the representation of the S gate."""
    # 创建 PhaseGate 对象和初始量子比特状态
    circuit = PhaseGate(0) * Qubit('01')
    # 计算表示结果
    answer = represent(circuit, nqubits=2)
    # 断言表示结果与预期的矩阵相等，使用了复数单位 I
    assert Matrix([0, I, 0, 0]) == answer


def test_represent_tgate():
    """Test the representation of the T gate."""
    # 创建 TGate 对象和初始量子比特状态
    circuit = TGate(0) * Qubit('01')
    # 断言表示结果与预期的矩阵相等，使用了 e^(iπ/4) 的表达式
    assert Matrix([0, exp(I*pi/4), 0, 0]) == represent(circuit, nqubits=2)


def test_compound_gates():
    """Test a compound gate representation."""
    # 创建复合门的量子电路
    circuit = YGate(0) * ZGate(0) * XGate(0) * HadamardGate(0) * Qubit('00')
    # 使用 represent 函数计算电路的表示，指定电路的量子比特数为 2
    answer = represent(circuit, nqubits=2)
    # 使用断言检查返回的矩阵是否与预期矩阵相等
    assert Matrix([I/sqrt(2), I/sqrt(2), 0, 0]) == answer
# 定义测试 CNOT 门的函数
def test_cnot_gate():
    """Test the CNOT gate."""
    # 创建一个 CNOT 门电路对象，作用在第 1 和第 0 量子比特上
    circuit = CNotGate(1, 0)
    # 断言该电路的表示与给定的矩阵表示相等
    assert represent(circuit, nqubits=2) == \
        Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # 将电路作用在 '111' 这个初始态上，并断言其与 Qapply 结果相等
    circuit = circuit*Qubit('111')
    assert matrix_to_qubit(represent(circuit, nqubits=3)) == \
        qapply(circuit)

    # 重新创建一个 CNOT 门电路对象
    circuit = CNotGate(1, 0)
    # 断言 CNOT 门的共轭等于自身
    assert Dagger(circuit) == circuit
    # 断言 CNOT 门的两次共轭等于自身
    assert Dagger(Dagger(circuit)) == circuit
    # 断言 CNOT 门的平方等于单位矩阵
    assert circuit*circuit == 1


# 定义测试 gate_sort 函数
def test_gate_sort():
    """Test gate_sort."""
    # 遍历 X, Y, Z, H, S, T 这些门，并断言对应的门排序结果正确
    for g in (X, Y, Z, H, S, T):
        assert gate_sort(g(2)*g(1)*g(0)) == g(0)*g(1)*g(2)
    # 测试复杂的门排序情况，断言结果正确
    e = gate_sort(X(1)*H(0)**2*CNOT(0, 1)*X(1)*X(0))
    assert e == H(0)**2*CNOT(0, 1)*X(0)*X(1)**2
    assert gate_sort(Z(0)*X(0)) == -X(0)*Z(0)
    assert gate_sort(Z(0)*X(0)**2) == X(0)**2*Z(0)
    assert gate_sort(Y(0)*H(0)) == -H(0)*Y(0)
    assert gate_sort(Y(0)*X(0)) == -X(0)*Y(0)
    assert gate_sort(Z(0)*Y(0)) == -Y(0)*Z(0)
    assert gate_sort(T(0)*S(0)) == S(0)*T(0)
    assert gate_sort(Z(0)*S(0)) == S(0)*Z(0)
    assert gate_sort(Z(0)*T(0)) == T(0)*Z(0)
    assert gate_sort(Z(0)*CNOT(0, 1)) == CNOT(0, 1)*Z(0)
    assert gate_sort(S(0)*CNOT(0, 1)) == CNOT(0, 1)*S(0)
    assert gate_sort(T(0)*CNOT(0, 1)) == CNOT(0, 1)*T(0)
    assert gate_sort(X(1)*CNOT(0, 1)) == CNOT(0, 1)*X(1)
    # 下面的代码需要花费很长时间，只在偶尔情况下取消注释
    # nqubits = 5
    # ngates = 10
    # trials = 10
    # for i in range(trials):
    #     c = random_circuit(ngates, nqubits)
    #     assert represent(c, nqubits=nqubits) == \
    #            represent(gate_sort(c), nqubits=nqubits)


# 定义测试 gate_simp 函数
def test_gate_simp():
    """Test gate_simp."""
    # 断言 gate_simp 对各种门的简化结果正确
    e = H(0)*X(1)*H(0)**2*CNOT(0, 1)*X(1)**3*X(0)*Z(3)**2*S(4)**3
    assert gate_simp(e) == H(0)*CNOT(0, 1)*S(4)*X(0)*Z(4)
    assert gate_simp(X(0)*X(0)) == 1
    assert gate_simp(Y(0)*Y(0)) == 1
    assert gate_simp(Z(0)*Z(0)) == 1
    assert gate_simp(H(0)*H(0)) == 1
    assert gate_simp(T(0)*T(0)) == S(0)
    assert gate_simp(S(0)*S(0)) == Z(0)
    assert gate_simp(Integer(1)) == Integer(1)
    assert gate_simp(X(0)**2 + Y(0)**2) == Integer(2)


# 定义测试 SWAP 门的函数
def test_swap_gate():
    """Test the SWAP gate."""
    # 定义 SWAP 门的矩阵表示
    swap_gate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    # 断言 SWAP 门的分解表示与预期的矩阵表示相等
    assert represent(SwapGate(1, 0).decompose(), nqubits=2) == swap_gate_matrix
    # 断言 SWAP 门作用在特定量子态上的结果正确
    assert qapply(SwapGate(1, 3)*Qubit('0010')) == Qubit('1000')
    # 遍历所有可能的量子比特数和交换的组合，断言 SWAP 门的表示正确
    nqubits = 4
    for i in range(nqubits):
        for j in range(i):
            assert represent(SwapGate(i, j), nqubits=nqubits) == \
                represent(SwapGate(i, j).decompose(), nqubits=nqubits)


# 定义测试单量子比特门的对易关系的函数
def test_one_qubit_commutators():
    """Test single qubit gate commutation relations."""
    # 对于 g1 和 g2 在给定门集合中的每一对进行迭代
    for g1 in (IdentityGate, X, Y, Z, H, T, S):
        # 对于第二个门 g2 在给定门集合中的每一个门进行迭代
        for g2 in (IdentityGate, X, Y, Z, H, T, S):
            # 计算 g1(0) 和 g2(0) 的对易子
            e = Commutator(g1(0), g2(0))
            # 将对易子表示为符号表达式，并将其转换为零矩阵
            a = matrix_to_zero(represent(e, nqubits=1, format='sympy'))
            # 将 do-it 方法应用于对易子，并将其表示为零矩阵
            b = matrix_to_zero(represent(e.doit(), nqubits=1, format='sympy'))
            # 断言两个矩阵相等
            assert a == b
    
            # 计算 g1(0) 和 g2(1) 的对易子
            e = Commutator(g1(0), g2(1))
            # 断言对易子的 do-it 方法结果为零矩阵
            assert e.doit() == 0
# 测试单量子比特门的反对易关系
def test_one_qubit_anticommutators():
    """Test single qubit gate anticommutation relations."""
    # 遍历单量子比特门的所有组合
    for g1 in (IdentityGate, X, Y, Z, H):
        for g2 in (IdentityGate, X, Y, Z, H):
            # 创建 g1(0) 和 g2(0) 的反对易子
            e = AntiCommutator(g1(0), g2(0))
            # 将反对易子表示为 SymPy 格式的矩阵，并化简为零矩阵
            a = matrix_to_zero(represent(e, nqubits=1, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=1, format='sympy'))
            # 断言两个化简后的矩阵相等
            assert a == b
            # 创建 g1(0) 和 g2(1) 的反对易子
            e = AntiCommutator(g1(0), g2(1))
            # 将反对易子表示为 SymPy 格式的矩阵，并化简为零矩阵
            a = matrix_to_zero(represent(e, nqubits=2, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=2, format='sympy'))
            # 断言两个化简后的矩阵相等
            assert a == b


# 测试涉及 CNOT 门的交换子关系
def test_cnot_commutators():
    """Test commutators of involving CNOT gates."""
    # 断言 CNOT(0, 1) 与 Z(0) 的交换子为零
    assert Commutator(CNOT(0, 1), Z(0)).doit() == 0
    # 断言 CNOT(0, 1) 与 T(0) 的交换子为零
    assert Commutator(CNOT(0, 1), T(0)).doit() == 0
    # 断言 CNOT(0, 1) 与 S(0) 的交换子为零
    assert Commutator(CNOT(0, 1), S(0)).doit() == 0
    # 断言 CNOT(0, 1) 与 X(1) 的交换子为零
    assert Commutator(CNOT(0, 1), X(1)).doit() == 0
    # 断言 CNOT(0, 1) 与 CNOT(0, 1) 的交换子为零
    assert Commutator(CNOT(0, 1), CNOT(0, 1)).doit() == 0
    # 断言 CNOT(0, 1) 与 CNOT(0, 2) 的交换子为零
    assert Commutator(CNOT(0, 1), CNOT(0, 2)).doit() == 0
    # 断言 CNOT(0, 2) 与 CNOT(0, 1) 的交换子为零
    assert Commutator(CNOT(0, 2), CNOT(0, 1)).doit() == 0
    # 断言 CNOT(1, 2) 与 CNOT(1, 0) 的交换子为零
    assert Commutator(CNOT(1, 2), CNOT(1, 0)).doit() == 0


# 测试随机电路生成器的功能
def test_random_circuit():
    """Test random circuit generator."""
    # 生成包含 10 个量子门和 3 个量子比特的随机电路
    c = random_circuit(10, 3)
    # 断言 c 是 Mul 类型的对象
    assert isinstance(c, Mul)
    # 将电路表示为 3 量子比特的矩阵
    m = represent(c, nqubits=3)
    # 断言矩阵 m 的形状为 (8, 8)
    assert m.shape == (8, 8)
    # 断言 m 是 Matrix 类型的对象
    assert isinstance(m, Matrix)


# 测试 Hermite 共轭 X 门
def test_hermitian_XGate():
    """Test Hermitian conjugate of XGate."""
    # 创建 XGate(1, 2) 门
    x = XGate(1, 2)
    # 计算 XGate(1, 2) 的 Hermite 共轭
    x_dagger = Dagger(x)

    # 断言 XGate(1, 2) 与其 Hermite 共轭相等
    assert (x == x_dagger)


# 测试 Hermite 共轭 Y 门
def test_hermitian_YGate():
    """Test Hermitian conjugate of YGate."""
    # 创建 YGate(1, 2) 门
    y = YGate(1, 2)
    # 计算 YGate(1, 2) 的 Hermite 共轭
    y_dagger = Dagger(y)

    # 断言 YGate(1, 2) 与其 Hermite 共轭相等
    assert (y == y_dagger)


# 测试 Hermite 共轭 Z 门
def test_hermitian_ZGate():
    """Test Hermitian conjugate of ZGate."""
    # 创建 ZGate(1, 2) 门
    z = ZGate(1, 2)
    # 计算 ZGate(1, 2) 的 Hermite 共轭
    z_dagger = Dagger(z)

    # 断言 ZGate(1, 2) 与其 Hermite 共轭相等
    assert (z == z_dagger)


# 测试 X 门的单位性质
def test_unitary_XGate():
    """Test unitary property of XGate."""
    # 创建 XGate(1, 2) 门
    x = XGate(1, 2)
    # 计算 XGate(1, 2) 的 Hermite 共轭
    x_dagger = Dagger(x)

    # 断言 XGate(1, 2) 与其 Hermite 共轭的乘积为单位矩阵
    assert (x*x_dagger == 1)


# 测试 Y 门的单位性质
def test_unitary_YGate():
    """Test unitary property of YGate."""
    # 创建 YGate(1, 2) 门
    y = YGate(1, 2)
    # 计算 YGate(1, 2) 的 Hermite 共轭
    y_dagger = Dagger(y)

    # 断言 YGate(1, 2) 与其 Hermite 共轭的乘积为单位矩阵
    assert (y*y_dagger == 1)


# 测试 Z 门的单位性质
def test_unitary_ZGate():
    """Test unitary property of ZGate."""
    # 创建 ZGate(1, 2) 门
    z = ZGate(1, 2)
    # 计算 ZGate(1, 2) 的 Hermite 共轭
    z_dagger = Dagger(z)

    # 断言 ZGate(1, 2) 与其 Hermite 共轭的乘积为单位矩阵
    assert (z*z_dagger == 1)
```