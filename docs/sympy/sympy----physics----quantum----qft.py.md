# `D:\src\scipysrc\sympy\sympy\physics\quantum\qft.py`

```
# 导入必要的模块和类
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt

# 量子计算的相关模块
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product

# 量子门相关类的导入
from sympy.physics.quantum.gate import (
    Gate, HadamardGate, SwapGate, OneQubitGate, CGate, PhaseGate, TGate, ZGate
)
from sympy.functions.elementary.complexes import sign

# 指定模块中公开的类名
__all__ = [
    'QFT',
    'IQFT',
    'RkGate',
    'Rk'
]

#-----------------------------------------------------------------------------
# Fourier stuff
#-----------------------------------------------------------------------------


class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name = 'Rk'
    gate_name_latex = 'R'

    def __new__(cls, *args):
        # 检查参数个数是否为两个，如果不是则抛出异常
        if len(args) != 2:
            raise QuantumError(
                'Rk gates only take two arguments, got: %r' % args
            )
        # 对于小的 k 值，Rk 门简化为其他门，使用这些替换得到对小量子比特数的熟悉结果
        target = args[0]
        k = args[1]
        if k == 1:
            return ZGate(target)
        elif k == 2:
            return PhaseGate(target)
        elif k == 3:
            return TGate(target)
        # 对参数进行评估
        args = cls._eval_args(args)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _eval_args(cls, args):
        # 回退到这里，因为 Gate._eval_args 假定 args 是所有目标，不能包含重复项
        return QExpr._eval_args(args)

    @property
    def k(self):
        return self.label[1]

    @property
    def targets(self):
        return self.label[:1]

    @property
    def gate_name_plot(self):
        # 返回用于绘图的门名称，使用 LaTeX 格式
        return r'$%s_%s$' % (self.gate_name_latex, str(self.k))

    def get_target_matrix(self, format='sympy'):
        # 根据指定的格式返回目标矩阵
        if format == 'sympy':
            return Matrix([[1, 0], [0, exp(sign(self.k)*Integer(2)*pi*I/(Integer(2)**abs(self.k)))]])
        # 如果格式无效，则抛出未实现错误
        raise NotImplementedError(
            'Invalid format for the R_k gate: %r' % format)


Rk = RkGate


class Fourier(Gate):
    """Superclass of Quantum Fourier and Inverse Quantum Fourier Gates."""

    @classmethod
    # 检查参数列表长度是否为2，如果不是则抛出量子错误异常
    def _eval_args(self, args):
        if len(args) != 2:
            raise QuantumError(
                'QFT/IQFT only takes two arguments, got: %r' % args
            )
        # 检查起始参数是否小于结束参数，否则抛出量子错误异常
        if args[0] >= args[1]:
            raise QuantumError("Start must be smaller than finish")
        # 调用 Gate 类的 _eval_args 方法，返回其结果
        return Gate._eval_args(args)

    # 使用默认基础上表示 Z 门的表达式
    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    # 在 Z 基础上表示 (I)QFT
    def _represent_ZGate(self, basis, **options):
        """
            Represents the (I)QFT In the Z Basis
        """
        # 获取选项中的量子比特数，如果为0则抛出量子错误异常
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')
        # 如果量子比特数小于最小量子比特数，则抛出量子错误异常
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )
        # 获取 QFT 矩阵的尺寸和角频率
        size = self.size
        omega = self.omega

        # 创建基本傅里叶变换矩阵的数组
        arrayFT = [[omega**(
            i*j % size)/sqrt(size) for i in range(size)] for j in range(size)]
        # 转换为矩阵对象
        matrixFT = Matrix(arrayFT)

        # 如果门标签的第一个元素不为0，则在高维空间中嵌入傅里叶变换矩阵
        if self.label[0] != 0:
            matrixFT = matrix_tensor_product(eye(2**self.label[0]), matrixFT)
        # 如果最小量子比特数小于给定量子比特数，则在矩阵FT后面嵌入单位矩阵
        if self.min_qubits < nqubits:
            matrixFT = matrix_tensor_product(
                matrixFT, eye(2**(nqubits - self.min_qubits)))

        # 返回傅里叶变换矩阵
        return matrixFT

    # 返回目标量子比特的范围
    @property
    def targets(self):
        return range(self.label[0], self.label[1])

    # 返回最小量子比特数
    @property
    def min_qubits(self):
        return self.label[1]

    # 返回 QFT 矩阵的尺寸，即2的(结束标签-开始标签)次方
    @property
    def size(self):
        """Size is the size of the QFT matrix"""
        return 2**(self.label[1] - self.label[0])

    # 返回角频率 omega 的符号表示
    @property
    def omega(self):
        return Symbol('omega')
class QFT(Fourier):
    """The forward quantum Fourier transform."""

    gate_name = 'QFT'
    gate_name_latex = 'QFT'

    def decompose(self):
        """Decomposes QFT into elementary gates."""
        # 获取QFT标签的起始和结束位置
        start = self.label[0]
        finish = self.label[1]
        # 初始化量子电路为单位矩阵
        circuit = 1
        # 从后向前遍历标签范围内的每一级
        for level in reversed(range(start, finish)):
            # 添加Hadamard门到电路
            circuit = HadamardGate(level)*circuit
            # 在每一级内部，添加CGate和RkGate组成的门到电路
            for i in range(level - start):
                circuit = CGate(level - i - 1, RkGate(level, i + 2))*circuit
        # 对标签范围内的前一半进行SwapGate操作
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        # 返回构建好的量子电路
        return circuit

    def _apply_operator_Qubit(self, qubits, **options):
        # 将量子电路分解后作用于量子比特
        return qapply(self.decompose()*qubits)

    def _eval_inverse(self):
        # 返回QFT的逆操作
        return IQFT(*self.args)

    @property
    def omega(self):
        # 返回QFT的角频率
        return exp(2*pi*I/self.size)


class IQFT(Fourier):
    """The inverse quantum Fourier transform."""

    gate_name = 'IQFT'
    gate_name_latex = '{QFT^{-1}}'

    def decompose(self):
        """Decomposes IQFT into elementary gates."""
        # 获取IQFT参数的起始和结束位置
        start = self.args[0]
        finish = self.args[1]
        # 初始化量子电路为单位矩阵
        circuit = 1
        # 对参数范围内的前一半进行SwapGate操作
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        # 从起始到结束遍历每一级
        for level in range(start, finish):
            # 在每一级内部，从后向前添加CGate和RkGate组成的门到电路
            for i in reversed(range(level - start)):
                circuit = CGate(level - i - 1, RkGate(level, -i - 2))*circuit
            # 在每一级最后添加HadamardGate门到电路
            circuit = HadamardGate(level)*circuit
        # 返回构建好的量子电路
        return circuit

    def _eval_inverse(self):
        # 返回IQFT的逆操作
        return QFT(*self.args)

    @property
    def omega(self):
        # 返回IQFT的角频率
        return exp(-2*pi*I/self.size)
```