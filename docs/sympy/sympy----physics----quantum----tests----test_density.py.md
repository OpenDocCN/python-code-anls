# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_density.py`

```
from sympy.core.numbers import Rational  # 导入 Rational 类，用于处理有理数
from sympy.core.singleton import S  # 导入 S 单例，表示特殊符号
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.functions.elementary.exponential import log  # 导入 log 函数，对数函数
from sympy.external import import_module  # 导入 import_module 函数，用于动态导入模块
from sympy.physics.quantum.density import Density, entropy, fidelity  # 导入量子力学相关模块
from sympy.physics.quantum.state import Ket, TimeDepKet  # 导入量子态相关类
from sympy.physics.quantum.qubit import Qubit  # 导入量子比特类
from sympy.physics.quantum.represent import represent  # 导入表示相关函数
from sympy.physics.quantum.dagger import Dagger  # 导入伴随算符类
from sympy.physics.quantum.cartesian import XKet, PxKet, PxOp, XOp  # 导入量子力学中的一些相关类
from sympy.physics.quantum.spin import JzKet  # 导入自旋态相关类
from sympy.physics.quantum.operator import OuterProduct  # 导入外积算符类
from sympy.physics.quantum.trace import Tr  # 导入迹运算函数
from sympy.functions import sqrt  # 导入平方根函数
from sympy.testing.pytest import raises  # 导入 raises 函数，用于检查是否抛出异常
from sympy.physics.quantum.matrixutils import scipy_sparse_matrix  # 导入矩阵工具函数
from sympy.physics.quantum.tensorproduct import TensorProduct  # 导入张量积类

def test_eval_args():
    # 检查 Density 实例是否创建成功
    assert isinstance(Density([Ket(0), 0.5], [Ket(1), 0.5]), Density)
    assert isinstance(Density([Qubit('00'), 1/sqrt(2)],
                              [Qubit('11'), 1/sqrt(2)]), Density)

    # 测试 Qubit 对象类型是否保留
    d = Density([Qubit('00'), 1/sqrt(2)], [Qubit('11'), 1/sqrt(2)])
    for (state, prob) in d.args:
        assert isinstance(state, Qubit)

    # 检查是否会抛出值错误，当未提供概率时
    raises(ValueError, lambda: Density([Ket(0)], [Ket(1)]))


def test_doit():

    x, y = symbols('x y')  # 创建符号变量 x 和 y
    A, B, C, D, E, F = symbols('A B C D E F', commutative=False)  # 创建非交换符号变量 A, B, C, D, E, F
    d = Density([XKet(), 0.5], [PxKet(), 0.5])  # 创建 Density 实例 d，包含 XKet 和 PxKet
    assert (0.5*(PxKet()*Dagger(PxKet())) +
            0.5*(XKet()*Dagger(XKet()))) == d.doit()

    # 检查带有符号表达式的 kets
    d_with_sym = Density([XKet(x*y), 0.5], [PxKet(x*y), 0.5])
    assert (0.5*(PxKet(x*y)*Dagger(PxKet(x*y))) +
            0.5*(XKet(x*y)*Dagger(XKet(x*y)))) == d_with_sym.doit()

    d = Density([(A + B)*C, 1.0])  # 创建包含符号表达式的 Density 实例 d
    assert d.doit() == (1.0*A*C*Dagger(C)*Dagger(A) +
                        1.0*A*C*Dagger(C)*Dagger(B) +
                        1.0*B*C*Dagger(C)*Dagger(A) +
                        1.0*B*C*Dagger(C)*Dagger(B))

    # 使用张量积作为参数
    # 创建包含简单张量积的 Density 实例
    t = TensorProduct(A, B, C)
    d = Density([t, 1.0])
    assert d.doit() == \
        1.0 * TensorProduct(A*Dagger(A), B*Dagger(B), C*Dagger(C))

    # 创建包含多个张量积状态的 Density 实例
    t2 = TensorProduct(A, B)
    t3 = TensorProduct(C, D)

    d = Density([t2, 0.5], [t3, 0.5])
    assert d.doit() == (0.5 * TensorProduct(A*Dagger(A), B*Dagger(B)) +
                        0.5 * TensorProduct(C*Dagger(C), D*Dagger(D)))

    # 创建包含混合态的 Density 实例
    d = Density([t2 + t3, 1.0])
    # 使用 assert 断言验证等式是否成立
    assert d.doit() == (1.0 * TensorProduct(A*Dagger(A), B*Dagger(B)) +
                        1.0 * TensorProduct(A*Dagger(C), B*Dagger(D)) +
                        1.0 * TensorProduct(C*Dagger(A), D*Dagger(B)) +
                        1.0 * TensorProduct(C*Dagger(C), D*Dagger(D)))

    # 创建具有自旋状态的密度算子
    tp1 = TensorProduct(JzKet(1, 1), JzKet(1, -1))
    d = Density([tp1, 1])

    # 对密度算子进行全迹运算
    t = Tr(d)
    # 使用 assert 断言验证全迹运算结果是否为1
    assert t.doit() == 1

    # 对具有自旋状态的密度算子进行部分迹运算
    t = Tr(d, [0])
    # 使用 assert 断言验证部分迹运算结果是否符合预期
    assert t.doit() == JzKet(1, -1) * Dagger(JzKet(1, -1))
    t = Tr(d, [1])
    assert t.doit() == JzKet(1, 1) * Dagger(JzKet(1, 1))

    # 使用另一个自旋状态创建密度算子
    tp2 = TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))
    d = Density([tp2, 1])

    # 对新的密度算子进行全迹运算
    t = Tr(d)
    assert t.doit() == 1

    # 对具有自旋状态的新密度算子进行部分迹运算
    t = Tr(d, [0])
    assert t.doit() == JzKet(S.Half, Rational(-1, 2)) * Dagger(JzKet(S.Half, Rational(-1, 2)))
    t = Tr(d, [1])
    assert t.doit() == JzKet(S.Half, S.Half) * Dagger(JzKet(S.Half, S.Half))
# 定义测试函数，用于测试 Density 类的 apply_op 方法
def test_apply_op():
    # 创建 Density 对象 d，包含两个元素：Ket(0) 和 0.5，以及 Ket(1) 和 0.5
    d = Density([Ket(0), 0.5], [Ket(1), 0.5])
    # 断言 apply_op(XOp()) 方法的结果应与给定的 Density 对象相等
    assert d.apply_op(XOp()) == Density([XOp()*Ket(0), 0.5],
                                        [XOp()*Ket(1), 0.5])


# 定义测试函数，用于测试 Density 类的 represent 方法
def test_represent():
    # 定义符号变量 x 和 y
    x, y = symbols('x y')
    # 创建 Density 对象 d，包含两个元素：XKet() 和 0.5，以及 PxKet() 和 0.5
    d = Density([XKet(), 0.5], [PxKet(), 0.5])
    # 断言 represent 方法的结果应与给定表达式的结果相等
    assert (represent(0.5*(PxKet()*Dagger(PxKet()))) +
            represent(0.5*(XKet()*Dagger(XKet())))) == represent(d)

    # 创建包含符号表达式的 Density 对象 d_with_sym
    d_with_sym = Density([XKet(x*y), 0.5], [PxKet(x*y), 0.5])
    # 断言 represent 方法的结果应与给定表达式的结果相等
    assert (represent(0.5*(PxKet(x*y)*Dagger(PxKet(x*y)))) +
            represent(0.5*(XKet(x*y)*Dagger(XKet(x*y))))) == \
        represent(d_with_sym)

    # 断言 represent 方法在给定 PxOp() 作为基础时的结果应与给定表达式的结果相等
    assert (represent(0.5*(XKet()*Dagger(XKet())), basis=PxOp()) +
            represent(0.5*(PxKet()*Dagger(PxKet())), basis=PxOp())) == \
        represent(d, basis=PxOp())


# 定义测试函数，用于测试 Density 类的 states 方法
def test_states():
    # 创建 Density 对象 d，包含两个元素：Ket(0) 和 0.5，以及 Ket(1) 和 0.5
    d = Density([Ket(0), 0.5], [Ket(1), 0.5])
    # 调用 states 方法，返回状态数组 states
    states = d.states()
    # 断言 states 数组中的第一个元素应为 Ket(0)，第二个元素应为 Ket(1)
    assert states[0] == Ket(0) and states[1] == Ket(1)


# 定义测试函数，用于测试 Density 类的 probs 方法
def test_probs():
    # 创建 Density 对象 d，包含两个元素：Ket(0) 和 0.75，以及 Ket(1) 和 0.25
    d = Density([Ket(0), .75], [Ket(1), 0.25])
    # 调用 probs 方法，返回概率数组 probs
    probs = d.probs()
    # 断言 probs 数组中的第一个元素应为 0.75，第二个元素应为 0.25

    # 稀疏矩阵 scipy.sparse
    x, y = symbols('x y')
    d = Density([Ket(0), x], [Ket(1), y])
    probs = d.probs()
    assert probs[0] == x and probs[1] == y


# 定义测试函数，用于测试 Density 类的 get_state 方法
def test_get_state():
    # 定义符号变量 x 和 y
    x, y = symbols('x y')
    # 创建 Density 对象 d，包含两个元素：Ket(0) 和 x，以及 Ket(1) 和 y
    d = Density([Ket(0), x], [Ket(1), y])
    # 调用 get_state 方法，返回状态元组 states
    states = (d.get_state(0), d.get_state(1))
    # 断言 states 元组中的第一个元素应为 Ket(0)，第二个元素应为 Ket(1)
    assert states[0] == Ket(0) and states[1] == Ket(1)


# 定义测试函数，用于测试 Density 类的 get_prob 方法
def test_get_prob():
    # 定义符号变量 x 和 y
    x, y = symbols('x y')
    # 创建 Density 对象 d，包含两个元素：Ket(0) 和 x，以及 Ket(1) 和 y
    d = Density([Ket(0), x], [Ket(1), y])
    # 调用 get_prob 方法，返回概率元组 probs
    probs = (d.get_prob(0), d.get_prob(1))
    # 断言 probs 元组中的第一个元素应为 x，第二个元素应为 y
    assert probs[0] == x and probs[1] == y


# 定义测试函数，用于测试 Density 类的 entropy 方法
def test_entropy():
    # 创建 JzKet 对象 up 和 down
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    # 创建 Density 对象 d，包含两个元素：up 和 0.5，以及 down 和 0.5
    d = Density((up, S.Half), (down, S.Half))

    # 测试 density 对象的 entropy 方法
    ent = entropy(d)
    assert entropy(d) == log(2)/2
    assert d.entropy() == log(2)/2

    # 导入 numpy 并测试 numpy 格式的 represent 方法
    np = import_module('numpy', min_module_version='1.4.0')
    if np:
        np_mat = represent(d, format='numpy')
        ent = entropy(np_mat)
        assert isinstance(np_mat, np.ndarray)
        assert ent.real == 0.69314718055994529
        assert ent.imag == 0

    # 导入 scipy.sparse 并测试 scipy.sparse 格式的 represent 方法
    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if scipy and np:
        mat = represent(d, format="scipy.sparse")
        assert isinstance(mat, scipy_sparse_matrix)
        assert ent.real == 0.69314718055994529
        assert ent.imag == 0


# 定义测试函数，用于测试 Tr 类的 doit 方法
def test_eval_trace():
    # 创建 JzKet 对象 up 和 down
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    # 创建 Density 对象 d，包含两个元素：up 和 0.5，以及 down 和 0.5
    d = Density((up, 0.5), (down, 0.5))

    # 测试 Tr 类的 doit 方法
    t = Tr(d)
    assert t.doit() == 1.0

    # 测试时间依赖态的 dummy test
    class TestTimeDepKet(TimeDepKet):
        def _eval_trace(self, bra, **options):
            return 1

    x, t = symbols('x t')
    # 创建一个 TestTimeDepKet 对象 k1，时间参数为 0，系数为 0.5
    k1 = TestTimeDepKet(0, 0.5)
    # 创建一个 TestTimeDepKet 对象 k2，时间参数为 0，系数为 1
    k2 = TestTimeDepKet(0, 1)
    # 使用 k1 和 k2 创建一个 Density 对象 d，分别以 0.5 的权重
    d = Density([k1, 0.5], [k2, 0.5])
    # 断言 Density 对象 d 的计算结果，应当等于 0.5 * OuterProduct(k1, k1.dual) + 0.5 * OuterProduct(k2, k2.dual)
    assert d.doit() == (0.5 * OuterProduct(k1, k1.dual) +
                        0.5 * OuterProduct(k2, k2.dual))
    
    # 创建一个 Tr 对象 t，用于计算给定对象的迹
    t = Tr(d)
    # 断言 Tr 对象 t 的计算结果，应当等于 1.0
    assert t.doit() == 1.0
def test_fidelity():
    #test with kets
    # 创建两个 JzKet 对象，表示量子态 |up> 和 |down>
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    # 创建叠加态 |updown> = (1/sqrt(2)) * |up> + (1/sqrt(2)) * |down>

    #check with matrices
    # 计算各个量子态的密度矩阵
    up_dm = represent(up * Dagger(up))
    down_dm = represent(down * Dagger(down))
    updown_dm = represent(updown * Dagger(updown))

    # 断言：验证量子态之间的保真度
    assert abs(fidelity(up_dm, up_dm) - 1) < 1e-3
    assert fidelity(up_dm, down_dm) < 1e-3
    assert abs(fidelity(up_dm, updown_dm) - (S.One/sqrt(2))) < 1e-3
    assert abs(fidelity(updown_dm, down_dm) - (S.One/sqrt(2))) < 1e-3

    #check with density
    # 使用 Density 对象创建量子态的混合态的密度矩阵
    up_dm = Density([up, 1.0])
    down_dm = Density([down, 1.0])
    updown_dm = Density([updown, 1.0])

    # 断言：验证密度矩阵之间的保真度
    assert abs(fidelity(up_dm, up_dm) - 1) < 1e-3
    assert abs(fidelity(up_dm, down_dm)) < 1e-3
    assert abs(fidelity(up_dm, updown_dm) - (S.One/sqrt(2))) < 1e-3
    assert abs(fidelity(updown_dm, down_dm) - (S.One/sqrt(2))) < 1e-3

    #check mixed states with density
    # 创建新的混合态 |updown2> = sqrt(3)/2 * |up> + 1/2 * |down>
    updown2 = sqrt(3)/2*up + S.Half*down
    # 创建两个混合态的密度矩阵
    d1 = Density([updown, 0.25], [updown2, 0.75])
    d2 = Density([updown, 0.75], [updown2, 0.25])
    # 断言：验证混合态密度矩阵之间的保真度
    assert abs(fidelity(d1, d2) - 0.991) < 1e-3
    assert abs(fidelity(d2, d1) - fidelity(d1, d2)) < 1e-3

    #using qubits/density(pure states)
    # 创建 Qubit 对象表示量子态 |0> 和 |1>
    state1 = Qubit('0')
    state2 = Qubit('1')
    # 创建叠加态 |state3> = (1/sqrt(2)) * |0> + (1/sqrt(2)) * |1>
    state3 = S.One/sqrt(2)*state1 + S.One/sqrt(2)*state2
    # 创建混合态 |state4>
    state4 = sqrt(Rational(2, 3))*state1 + S.One/sqrt(3)*state2

    # 创建纯态的密度矩阵
    state1_dm = Density([state1, 1])
    state2_dm = Density([state2, 1])
    state3_dm = Density([state3, 1])

    # 断言：验证纯态密度矩阵之间的保真度
    assert fidelity(state1_dm, state1_dm) == 1
    assert fidelity(state1_dm, state2_dm) == 0
    assert abs(fidelity(state1_dm, state3_dm) - 1/sqrt(2)) < 1e-3
    assert abs(fidelity(state3_dm, state2_dm) - 1/sqrt(2)) < 1e-3

    #using qubits/density(mixed states)
    # 创建混合态的密度矩阵
    d1 = Density([state3, 0.70], [state4, 0.30])
    d2 = Density([state3, 0.20], [state4, 0.80])
    # 断言：验证混合态密度矩阵之间的保真度
    assert abs(fidelity(d1, d1) - 1) < 1e-3
    assert abs(fidelity(d1, d2) - 0.996) < 1e-3
    assert abs(fidelity(d1, d2) - fidelity(d2, d1)) < 1e-3

    #TODO: test for invalid arguments
    # non-square matrix
    # 创建非方阵 mat1 和 mat2，应引发 ValueError 异常
    mat1 = [[0, 0],
            [0, 0],
            [0, 0]]

    mat2 = [[0, 0],
            [0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))

    # unequal dimensions
    # 创建维度不相等的矩阵 mat1 和 mat2，应引发 ValueError 异常
    mat1 = [[0, 0],
            [0, 0]]
    mat2 = [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))

    # unsupported data-type
    # 创建不支持的数据类型 x 和 y，应引发 ValueError 异常
    x, y = 1, 2  # random values that is not a matrix
    raises(ValueError, lambda: fidelity(x, y))
```