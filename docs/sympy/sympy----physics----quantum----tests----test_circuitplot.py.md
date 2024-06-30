# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_circuitplot.py`

```
# 导入所需的函数和模块
from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
     CreateCGate
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip

# 尝试导入 matplotlib 模块
mpl = import_module('matplotlib')

# 定义测试函数 test_render_label，测试 render_label 函数的不同输入
def test_render_label():
    assert render_label('q0') == r'$\left|q0\right\rangle'
    assert render_label('q0', {'q0': '0'}) == r'$\left|q0\right\rangle=\left|0\right\rangle$'

# 定义测试函数 test_Mz，测试 Mz 函数的输出
def test_Mz():
    assert str(Mz(0)) == 'Mz(0)'

# 定义测试函数 test_create1，测试 CreateOneQubitGate 函数的输出
def test_create1():
    Qgate = CreateOneQubitGate('Q')
    assert str(Qgate(0)) == 'Q(0)'

# 定义测试函数 test_createc，测试 CreateCGate 函数的输出
def test_createc():
    Qgate = CreateCGate('Q')
    assert str(Qgate([1],0)) == 'C((1),Q(0))'

# 定义测试函数 test_labeller，测试 labeller 函数的不同输入
def test_labeller():
    """Test the labeller utility"""
    assert labeller(2) == ['q_1', 'q_0']
    assert labeller(3,'j') == ['j_2', 'j_1', 'j_0']

# 定义测试函数 test_cnot，测试简单的 CNOT 电路
def test_cnot():
    """Test a simple cnot circuit. Right now this only makes sure the code doesn't
    raise an exception, and some simple properties
    """
    # 如果没有安装 matplotlib，则跳过测试
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    # 创建包含 CNOT 门的电路图，并验证一些简单的属性
    c = CircuitPlot(CNOT(1,0),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

    # 创建另一个电路图，不带标签
    c = CircuitPlot(CNOT(1,0),2)
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == []

# 定义测试函数 test_ex1，测试复杂的电路图，包含 CNOT 和 H 门
def test_ex1():
    """Test a more complex circuit with CNOT and H gates"""
    # 如果没有安装 matplotlib，则跳过测试
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    # 创建包含 CNOT 和 H 门的电路图，并验证一些属性
    c = CircuitPlot(CNOT(1,0)*H(1),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

# 定义测试函数 test_ex4，测试更复杂的电路图，包含多个门的组合
def test_ex4():
    """Test a very complex circuit with SWAP, H, CGate, S, and T gates"""
    # 如果没有安装 matplotlib，则跳过测试
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    # 创建包含多个门的复杂电路图，并验证一些属性
    c = CircuitPlot(SWAP(0,2)*H(0)* CGate((0,),S(1)) *H(1)*CGate((0,),T(2))\
                    *CGate((1,),S(2))*H(2),3,labels=labeller(3,'j'))
    assert c.ngates == 7
    assert c.nqubits == 3
    assert c.labels == ['j_2', 'j_1', 'j_0']
```