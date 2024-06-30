# `D:\src\scipysrc\sympy\sympy\physics\quantum\qasm.py`

```
# 文件说明和导入模块列表
__all__ = [
    'Qasm',
    ]

# 导入必要的数学函数
from math import prod

# 导入量子计算库中定义的门操作
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T, CPHASE
# 导入量子电路绘图相关模块
from sympy.physics.quantum.circuitplot import Mz

# 将输入的 Qasm 指令行解析为 Qasm 对象的函数
def read_qasm(lines):
    return Qasm(*lines.splitlines())

# 从文件中读取 Qasm 指令并解析为 Qasm 对象的函数
def read_qasm_file(filename):
    return Qasm(*open(filename).readlines())

# 将量子比特索引从大到小重新排序的函数
def flip_index(i, n):
    """Reorder qubit indices from largest to smallest.

    >>> from sympy.physics.quantum.qasm import flip_index
    >>> flip_index(0, 2)
    1
    >>> flip_index(1, 2)
    0
    """
    return n-i-1

# 去除 Qasm 指令行中的注释部分的函数
def trim(line):
    """Remove everything following comment # characters in line.

    >>> from sympy.physics.quantum.qasm import trim
    >>> trim('nothing happens here')
    'nothing happens here'
    >>> trim('something #happens here')
    'something '
    """
    if '#' not in line:
        return line
    return line.split('#')[0]

# 获取量子比特标签在标签列表中对应的索引的函数
def get_index(target, labels):
    """Get qubit labels from the rest of the line,and return indices

    >>> from sympy.physics.quantum.qasm import get_index
    >>> get_index('q0', ['q0', 'q1'])
    1
    >>> get_index('q1', ['q0', 'q1'])
    0
    """
    nq = len(labels)
    return flip_index(labels.index(target), nq)

# 获取多个量子比特标签对应的索引列表的函数
def get_indices(targets, labels):
    return [get_index(t, labels) for t in targets]

# 过滤掉空白行和注释行的生成器函数
def nonblank(args):
    for line in args:
        line = trim(line)
        if line.isspace():
            continue
        yield line
    return

# 将 Qasm 指令行拆分为命令和参数的函数
def fullsplit(line):
    words = line.split()
    rest = ' '.join(words[1:])
    return fixcommand(words[0]), [s.strip() for s in rest.split(',')]

# 修正 Qasm 命令名称的函数
def fixcommand(c):
    """Fix Qasm command names.

    Remove all of forbidden characters from command c, and
    replace 'def' with 'qdef'.
    """
    forbidden_characters = ['-']
    c = c.lower()
    for char in forbidden_characters:
        c = c.replace(char, '')
    if c == 'def':
        return 'qdef'
    return c

# 去除字符串中的显式引号的函数
def stripquotes(s):
    """Replace explicit quotes in a string.

    >>> from sympy.physics.quantum.qasm import stripquotes
    >>> stripquotes("'S'") == 'S'
    True
    >>> stripquotes('"S"') == 'S'
    True
    >>> stripquotes('S') == 'S'
    True
    """
    s = s.replace('"', '') # Remove second set of quotes?
    s = s.replace("'", '')
    return s

# Qasm 类，用于从 Qasm 指令行形成对象
class Qasm:
    """Class to form objects from Qasm lines

    >>> from sympy.physics.quantum.qasm import Qasm
    >>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    """
    # 调用实例方法 `q.get_circuit()`，返回量子电路的字符串表示
    >>> q.get_circuit()
    CNOT(1,0)*H(1)
    # 创建一个新的 Qasm 对象 `q`，初始化时传入一系列参数和关键字参数
    >>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    # 调用实例方法 `q.get_circuit()`，返回量子电路的字符串表示
    >>> q.get_circuit()
    CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
    """
    # Qasm 类的初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 定义一个空字典 `defs`，用于存储定义的命令
        self.defs = {}
        # 初始化一个空的电路列表 `circuit`
        self.circuit = []
        # 初始化一个空的标签列表 `labels`
        self.labels = []
        # 初始化一个空的初始化信息字典 `inits`
        self.inits = {}
        # 调用 add 方法，将传入的位置参数添加到对象中
        self.add(*args)
        # 将关键字参数存储在实例变量 `kwargs` 中
        self.kwargs = kwargs

    # 实例方法 add，用于添加量子指令到电路中
    def add(self, *lines):
        # 遍历非空的行
        for line in nonblank(lines):
            # 分割命令和其余部分
            command, rest = fullsplit(line)
            # 如果命令已经被定义过，则调用定义的函数处理
            if self.defs.get(command): #defs come first, since you can override built-in
                function = self.defs.get(command)
                # 获取参数的索引
                indices = self.indices(rest)
                # 如果参数数量为1，则调用函数并将结果添加到电路中
                if len(indices) == 1:
                    self.circuit.append(function(indices[0]))
                # 如果参数数量大于1，则调用函数并将结果添加到电路中
                else:
                    self.circuit.append(function(indices[:-1], indices[-1]))
            # 如果对象有命令对应的方法，则直接调用该方法处理命令
            elif hasattr(self, command):
                function = getattr(self, command)
                function(*rest)
            # 如果命令既不是已定义的函数也没有对应的方法，则打印错误信息
            else:
                print("Function %s not defined. Skipping" % command)

    # 实例方法 get_circuit，返回反转后的电路列表的乘积
    def get_circuit(self):
        return prod(reversed(self.circuit))

    # 实例方法 get_labels，返回反转后的标签列表
    def get_labels(self):
        return list(reversed(self.labels))

    # 实例方法 plot，绘制量子电路图
    def plot(self):
        # 导入绘图模块
        from sympy.physics.quantum.circuitplot import CircuitPlot
        # 获取电路和标签
        circuit, labels = self.get_circuit(), self.get_labels()
        # 创建电路图
        CircuitPlot(circuit, len(labels), labels=labels, inits=self.inits)

    # 实例方法 qubit，用于添加量子比特标签和初始化信息
    def qubit(self, arg, init=None):
        self.labels.append(arg)
        if init: self.inits[arg] = init

    # 实例方法 indices，根据参数获取标签对应的索引
    def indices(self, args):
        return get_indices(args, self.labels)

    # 实例方法 index，根据参数获取标签对应的索引
    def index(self, arg):
        return get_index(arg, self.labels)

    # 实例方法 nop，空操作，用于跳过不支持的命令
    def nop(self, *args):
        pass

    # 实例方法 x，添加单量子比特 X 门操作到电路
    def x(self, arg):
        self.circuit.append(X(self.index(arg)))

    # 实例方法 z，添加单量子比特 Z 门操作到电路
    def z(self, arg):
        self.circuit.append(Z(self.index(arg)))

    # 实例方法 h，添加单量子比特 H 门操作到电路
    def h(self, arg):
        self.circuit.append(H(self.index(arg)))

    # 实例方法 s，添加单量子比特 S 门操作到电路
    def s(self, arg):
        self.circuit.append(S(self.index(arg)))

    # 实例方法 t，添加单量子比特 T 门操作到电路
    def t(self, arg):
        self.circuit.append(T(self.index(arg)))

    # 实例方法 measure，添加单量子比特测量操作到电路
    def measure(self, arg):
        self.circuit.append(Mz(self.index(arg)))

    # 实例方法 cnot，添加 CNOT 门操作到电路
    def cnot(self, a1, a2):
        self.circuit.append(CNOT(*self.indices([a1, a2])))

    # 实例方法 swap，添加 SWAP 门操作到电路
    def swap(self, a1, a2):
        self.circuit.append(SWAP(*self.indices([a1, a2])))

    # 实例方法 cphase，添加 CPHASE 门操作到电路
    def cphase(self, a1, a2):
        self.circuit.append(CPHASE(*self.indices([a1, a2])))

    # 实例方法 toffoli，添加 Toffoli 门操作到电路
    def toffoli(self, a1, a2, a3):
        i1, i2, i3 = self.indices([a1, a2, a3])
        self.circuit.append(CGateS((i1, i2), X(i3)))

    # 实例方法 cx，添加 CX 门操作到电路
    def cx(self, a1, a2):
        fi, fj = self.indices([a1, a2])
        self.circuit.append(CGate(fi, X(fj)))

    # 实例方法 cz，添加 CZ 门操作到电路
    def cz(self, a1, a2):
        fi, fj = self.indices([a1, a2])
        self.circuit.append(CGate(fi, Z(fj)))

    # 实例方法 defbox，暂未支持的方法，打印跳过信息
    def defbox(self, *args):
        print("defbox not supported yet. Skipping: ", args)
    # 定义一个方法 qdef，用于定义量子门操作
    def qdef(self, name, ncontrols, symbol):
        # 导入量子电路绘图所需的函数
        from sympy.physics.quantum.circuitplot import CreateOneQubitGate, CreateCGate
        # 将 ncontrols 转换为整数类型
        ncontrols = int(ncontrols)
        # 调用 fixcommand 函数修正命令名
        command = fixcommand(name)
        # 调用 stripquotes 函数去除符号中的引号
        symbol = stripquotes(symbol)
        # 根据 ncontrols 的值决定使用单量子门或控制量子门
        if ncontrols > 0:
            # 如果 ncontrols 大于 0，则使用 CreateCGate 创建控制量子门，并存储到 self.defs 中
            self.defs[command] = CreateCGate(symbol)
        else:
            # 如果 ncontrols 不大于 0，则使用 CreateOneQubitGate 创建单量子门，并存储到 self.defs 中
            self.defs[command] = CreateOneQubitGate(symbol)
```