# `D:\src\scipysrc\sympy\sympy\physics\quantum\shor.py`

```
import math  # 导入数学库
import random  # 导入随机数库

from sympy.core.mul import Mul  # 导入 SymPy 的乘法运算模块 Mul
from sympy.core.singleton import S  # 导入 SymPy 的单例模块 S
from sympy.functions.elementary.exponential import log  # 导入 SymPy 的对数函数模块 log
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 SymPy 的平方根函数模块 sqrt
from sympy.core.intfunc import igcd  # 导入 SymPy 的最大公约数函数模块 igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction  # 导入 SymPy 的连分数周期模块 continued_fraction
from sympy.utilities.iterables import variations  # 导入 SymPy 的排列组合函数模块 variations

from sympy.physics.quantum.gate import Gate  # 导入 SymPy 的量子门模块 Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot  # 导入 SymPy 的量子比特和部分测量模块
from sympy.physics.quantum.qapply import qapply  # 导入 SymPy 的量子操作模块 qapply
from sympy.physics.quantum.qft import QFT  # 导入 SymPy 的量子傅里叶变换模块 QFT
from sympy.physics.quantum.qexpr import QuantumError  # 导入 SymPy 的量子错误模块 QuantumError


class OrderFindingException(QuantumError):  # 定义一个继承自 QuantumError 的订单查找异常类
    pass


class CMod(Gate):
    """A controlled mod gate.

    This is black box controlled Mod function for use by shor's algorithm.
    TODO: implement a decompose property that returns how to do this in terms
    of elementary gates
    """

    @classmethod
    def _eval_args(cls, args):
        # t = args[0]
        # a = args[1]
        # N = args[2]
        raise NotImplementedError('The CMod gate has not been completed.')

    @property
    def t(self):
        """Size of 1/2 input register.  First 1/2 holds output."""
        return self.label[0]  # 返回标签中的第一个元素作为 t

    @property
    def a(self):
        """Base of the controlled mod function."""
        return self.label[1]  # 返回标签中的第二个元素作为 a

    @property
    def N(self):
        """N is the type of modular arithmetic we are doing."""
        return self.label[2]  # 返回标签中的第三个元素作为 N

    def _apply_operator_Qubit(self, qubits, **options):
        """
        This directly calculates the controlled mod of the second half of
        the register and puts it in the second
        This will look pretty when we get Tensor Symbolically working
        """
        n = 1  # 初始化 n 为 1
        k = 0  # 初始化 k 为 0
        # Determine the value stored in high memory.
        for i in range(self.t):  # 循环次数为 t
            k += n*qubits[self.t + i]  # 计算 k 的值
            n *= 2  # n 乘以 2

        # The value to go in low memory will be out.
        out = int(self.a**k % self.N)  # 计算模控制函数的结果

        # Create array for new qbit-ket which will have high memory unaffected
        outarray = list(qubits.args[0][:self.t])  # 创建一个新的量子位数组，高内存不受影响

        # Place out in low memory
        for i in reversed(range(self.t)):  # 反向循环 t 次
            outarray.append((out >> i) & 1)  # 在低内存中放置 out 的值

        return Qubit(*outarray)


def shor(N):
    """This function implements Shor's factoring algorithm on the Integer N

    The algorithm starts by picking a random number (a) and seeing if it is
    coprime with N. If it is not, then the gcd of the two numbers is a factor
    and we are done. Otherwise, it begins the period_finding subroutine which
    finds the period of a in modulo N arithmetic. This period, if even, can
    be used to calculate factors by taking a**(r/2)-1 and a**(r/2)+1.
    These values are returned.
    """
    # 从范围 [2, N-1] 中随机选择一个整数 a
    a = random.randrange(N - 2) + 2
    # 如果选取的 a 不与 N 互质（最大公约数不为1），则返回 igcd(N, a)
    if igcd(N, a) != 1:
        return igcd(N, a)
    # 使用选取的 a 和 N 寻找其周期 r
    r = period_find(a, N)
    # 如果 r 是奇数，则重新调用 shor(N) 函数
    if r % 2 == 1:
        shor(N)
    # 计算两个数的最大公约数，作为答案返回
    answer = (igcd(a**(r/2) - 1, N), igcd(a**(r/2) + 1, N))
    # 返回答案
    return answer
# 计算 x/y 的连分数
def getr(x, y, N):
    fraction = continued_fraction(x, y)
    # 调用 ratioize 函数计算连分数的总和，得到 r
    total = ratioize(fraction, N)
    return total


# 计算给定列表的总和，直到遇到大于 N 的第一个元素则返回零
def ratioize(list, N):
    if list[0] > N:
        return S.Zero
    if len(list) == 1:
        return list[0]
    return list[0] + ratioize(list[1:], N)


# 寻找在模 N 算术下，整数 a 的周期
def period_find(a, N):
    """Finds the period of a in modulo N arithmetic

    This is quantum part of Shor's algorithm. It takes two registers,
    puts first in superposition of states with Hadamards so: ``|k>|0>``
    with k being all possible choices. It then does a controlled mod and
    a QFT to determine the order of a.
    """
    epsilon = .5
    # 选取 t，使得在 epsilon 范围内保持准确性
    t = int(2*math.ceil(log(N, 2)))
    # 将第一半寄存器设为全 0 的状态 |000...000>
    start = [0 for x in range(t)]
    # 将第二半寄存器置于叠加态，即 |1>x|0> + |2>x|0> + ... + |k>x>|0> + ... + |2**n-1>x|0>
    factor = 1/sqrt(2**t)
    qubits = 0
    for arr in variations(range(2), t, repetition=True):
        qbitArray = list(arr) + start
        qubits = qubits + Qubit(*qbitArray)
    circuit = (factor*qubits).expand()
    # 控制第二半寄存器，使得状态变为：|1>x|a**1 %N> + |2>x|a**2 %N> + ... + |k>x|a**k %N >+ ... + |2**n-1=k>x|a**k % n>
    circuit = CMod(t, a, N)*circuit
    # 测量第一半寄存器，得到其中的一个 a**k % N

    circuit = qapply(circuit)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i)
    # 在第二半寄存器上应用逆量子傅里叶变换（Inverse Quantum Fourier Transform）

    circuit = qapply(QFT(t, t*2).decompose()*circuit, floatingPoint=True)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i + t)
    if isinstance(circuit, Qubit):
        register = circuit
    elif isinstance(circuit, Mul):
        register = circuit.args[-1]
    else:
        register = circuit.args[-1].args[-1]

    n = 1
    answer = 0
    for i in range(len(register)/2):
        answer += n*register[i + t]
        n = n << 1
    if answer == 0:
        raise OrderFindingException(
            "Order finder returned 0. Happens with chance %f" % epsilon)
    # 使用连分数将 answer 转换为 r
    g = getr(answer, 2**t, N)
    return g
```