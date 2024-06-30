# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_qasm.py`

```
# 导入需要的 QASM 相关模块和函数
from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
     get_index, nonblank, fullsplit, fixcommand, stripquotes, read_qasm
# 导入量子门操作符和测量函数
from sympy.physics.quantum.gate import X, Z, H, S, T
# 导入特殊控制门和量子线路绘图函数
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz

# 测试读取 QASM 语句的功能
def test_qasm_readqasm():
    qasm_lines = """\
    qubit q_0
    qubit q_1
    h q_0
    cnot q_0,q_1
    """
    # 解析 QASM 语句并返回量子线路对象，断言其等于特定的量子门序列
    q = read_qasm(qasm_lines)
    assert q.get_circuit() == CNOT(1,0)*H(1)

# 测试通过 Qasm 类直接创建并操作量子线路的功能
def test_qasm_ex1():
    q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    # 断言生成的量子线路对象等于特定的量子门序列
    assert q.get_circuit() == CNOT(1,0)*H(1)

# 测试使用 Qasm 类方法调用创建和操作量子线路的功能
def test_qasm_ex1_methodcalls():
    q = Qasm()
    q.qubit('q_0')
    q.qubit('q_1')
    q.h('q_0')
    q.cnot('q_0', 'q_1')
    # 断言生成的量子线路对象等于特定的量子门序列
    assert q.get_circuit() == CNOT(1,0)*H(1)

# 测试 SWAP 门操作的功能
def test_qasm_swap():
    q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    # 断言生成的量子线路对象等于特定的量子门序列
    assert q.get_circuit() == CNOT(1,0)*CNOT(0,1)*CNOT(1,0)

# 测试包含多个量子门操作和测量的 QASM 语句的功能
def test_qasm_ex2():
    q = Qasm('qubit q_0', 'qubit q_1', 'qubit q_2', 'h  q_1',
             'cnot q_1,q_2', 'cnot q_0,q_1', 'h q_0',
             'measure q_1', 'measure q_0',
             'c-x q_1,q_2', 'c-z q_0,q_2')
    # 断言生成的量子线路对象等于特定的量子门序列
    assert q.get_circuit() == CGate(2,Z(0))*CGate(1,X(0))*Mz(2)*Mz(1)*H(2)*CNOT(2,1)*CNOT(1,0)*H(1)

# 测试单量子比特门操作的功能
def test_qasm_1q():
    for symbol, gate in [('x', X), ('z', Z), ('h', H), ('s', S), ('t', T), ('measure', Mz)]:
        q = Qasm('qubit q_0', '%s q_0' % symbol)
        # 断言生成的量子线路对象等于特定的单量子门操作
        assert q.get_circuit() == gate(0)

# 测试双量子比特门操作的功能
def test_qasm_2q():
    for symbol, gate in [('cnot', CNOT), ('swap', SWAP), ('cphase', CPHASE)]:
        q = Qasm('qubit q_0', 'qubit q_1', '%s q_0,q_1' % symbol)
        # 断言生成的量子线路对象等于特定的双量子门操作
        assert q.get_circuit() == gate(1,0)

# 测试三量子比特门操作的功能
def test_qasm_3q():
    q = Qasm('qubit q0', 'qubit q1', 'qubit q2', 'toffoli q2,q1,q0')
    # 断言生成的量子线路对象等于特定的三量子门操作
    assert q.get_circuit() == CGateS((0,1),X(2))

# 测试翻转索引的功能
def test_qasm_flip_index():
    assert flip_index(0, 2) == 1
    assert flip_index(1, 2) == 0

# 测试修剪字符串的功能
def test_qasm_trim():
    assert trim('nothing happens here') == 'nothing happens here'
    assert trim("Something #happens here") == "Something "

# 测试获取索引的功能
def test_qasm_get_index():
    assert get_index('q0', ['q0', 'q1']) == 1
    assert get_index('q1', ['q0', 'q1']) == 0

# 测试移除空白字符的功能
def test_qasm_nonblank():
    assert list(nonblank('abcd')) == list('abcd')
    assert list(nonblank('abc ')) == list('abc')

# 测试全分割函数的功能
def test_qasm_fullsplit():
    assert fullsplit('g q0,q1,q2,  q3') == ('g', ['q0', 'q1', 'q2', 'q3'])

# 测试修复命令字符串的功能
def test_qasm_fixcommand():
    assert fixcommand('foo') == 'foo'
    assert fixcommand('def') == 'qdef'

# 测试移除引号的功能
def test_qasm_stripquotes():
    assert stripquotes("'S'") == 'S'
    assert stripquotes('"S"') == 'S'
    assert stripquotes('S') == 'S'

# 测试 QASM 中定义量子门的功能
def test_qasm_qdef():
    # 弱测试条件 (str)，因为没有实际类的访问权限
    q = Qasm("def Q,0,Q",'qubit q0','Q q0')
    assert str(q.get_circuit()) == 'Q(0)'

    q = Qasm("def CQ,1,Q", 'qubit q0', 'qubit q1', 'CQ q0,q1')
    assert str(q.get_circuit()) == 'C((1),Q(0))'
```