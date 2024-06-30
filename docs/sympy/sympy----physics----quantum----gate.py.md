# `D:\src\scipysrc\sympy\sympy\physics\quantum\gate.py`

```
# 导入必要的库
from itertools import chain
import random

# 导入 SymPy 的各种符号和函数
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict

# 导入量子力学相关的符号和函数
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
                                            HermitianOperator)
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache

# 导入 SymPy 的矩阵基类
from sympy.matrices.matrixbase import MatrixBase

# 导入 SymPy 的辅助函数
from sympy.utilities.iterables import is_sequence

# 导出的公开符号列表
__all__ = [
    'Gate',
    'CGate',
    'UGate',
    'OneQubitGate',
    'TwoQubitGate',
    'IdentityGate',
    'HadamardGate',
    'XGate',
    'YGate',
    'ZGate',
    'TGate',
    'PhaseGate',
    'SwapGate',
    'CNotGate',
    # 别名门名称
    'CNOT',
    'SWAP',
    'H',
    'X',
    'Y',
    'Z',
    'T',
    'S',
    'Phase',
    'normalized',
    'gate_sort',
    'gate_simp',
    'random_circuit',
    'CPHASE',
    'CGateS',
]

#-----------------------------------------------------------------------------
# 门类的超类定义
#-----------------------------------------------------------------------------

# 是否进行规范化的全局标志，默认为 True
_normalized = True


def _max(*args, **kwargs):
    # 如果未指定排序关键字，则使用默认排序键
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    # 返回最大值
    return max(*args, **kwargs)


def _min(*args, **kwargs):
    # 如果未指定排序关键字，则使用默认排序键
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    # 返回最小值
    return min(*args, **kwargs)


def normalized(normalize):
    r"""设置控制哈达玛门通过 `1/\sqrt{2}` 规范化的标志。

    这是一个全局设置，用于简化各种表达式的外观，通过去除哈达玛门的前导 `1/\sqrt{2}`。

    参数
    ----------
    # normalize : bool
    # 布尔型参数，指定是否归一化哈达玛门的系数 `1/\sqrt{2}`。
    # 当值为 True 时，哈达玛门包含 `1/\sqrt{2}` 归一化因子。
    # 当值为 False 时，哈达玛门不包含此因子。
    """
    设置全局变量 `_normalized` 为传入的 `normalize` 参数的值。
    """
    global _normalized
    _normalized = normalize
def _validate_targets_controls(tandc):
    # 将输入参数 tandc 转换为列表形式
    tandc = list(tandc)
    
    # 检查每个元素是否为整数或符号
    for bit in tandc:
        if not bit.is_Integer and not bit.is_Symbol:
            raise TypeError('Integer expected, got: %r' % tandc[bit])
    
    # 检测是否有重复的元素
    if len(set(tandc)) != len(tandc):
        raise QuantumError(
            'Target/control qubits in a gate cannot be duplicated'
        )


class Gate(UnitaryOperator):
    """Non-controlled unitary gate operator that acts on qubits.

    This is a general abstract gate that needs to be subclassed to do anything
    useful.

    Parameters
    ----------
    label : tuple, int
        A list of the target qubits (as ints) that the gate will apply to.

    Examples
    ========


    """

    _label_separator = ','

    gate_name = 'G'
    gate_name_latex = 'G'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # 将输入参数 args 转换为元组形式
        args = Tuple(*UnitaryOperator._eval_args(args))
        # 调用 _validate_targets_controls 函数检查参数有效性
        _validate_targets_controls(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        # 返回一个最小可能的 Hilbert 空间，作用于参数 args
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        # 返回门作用的总量子比特数，包括目标和控制量子比特
        return len(self.targets)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        # 返回门需要作用的最小量子比特数
        return _max(self.targets) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        # 返回目标量子比特的元组
        return self.label

    @property
    def gate_name_plot(self):
        # 返回用于绘图的门名称 LaTeX 表示
        return r'$%s$' % self.gate_name_latex

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix representation of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        # 抛出未实现的错误，表示该方法在 Gate 类中尚未实现
        raise NotImplementedError(
            'get_target_matrix is not implemented in Gate.')

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------
    # 重定向对 IntQubit 的操作到 Qubit
    def _apply_operator_IntQubit(self, qubits, **options):
        """Redirect an apply from IntQubit to Qubit"""
        return self._apply_operator_Qubit(qubits, **options)

    # 将门作用于 Qubit
    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this gate to a Qubit."""

        # 检查门作用的量子比特数目
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )

        # 如果控制条件不满足，则直接返回
        if isinstance(self, CGate):
            if not self.eval_controls(qubits):
                return qubits

        targets = self.targets
        target_matrix = self.get_target_matrix(format='sympy')

        # 确定目标矩阵中应用的列索引
        column_index = 0
        n = 1
        for target in targets:
            column_index += n*qubits[target]
            n = n << 1
        column = target_matrix[:, int(column_index)]

        # 对每个列元素应用到量子比特上
        result = 0
        for index in range(column.rows):
            # TODO: 可以优化以减少量子比特对象的创建次数。
            # 应当直接操作量子比特的原始值列表，然后一次性构建新的量子比特对象。
            # 复制传入的量子比特
            new_qubit = qubits.__class__(*qubits.args)
            # 翻转需要翻转的比特
            for bit, target in enumerate(targets):
                if new_qubit[target] != (index >> bit) & 1:
                    new_qubit = new_qubit.flip(target)
            # 列中该行的值乘以翻转后的比特，作为结果的一部分
            result += column[index]*new_qubit
        return result

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    # 使用默认基础表示
    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    # 使用 Z 门表示
    def _represent_ZGate(self, basis, **options):
        format = options.get('format', 'sympy')
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')

        # 确保门所需的量子比特数足够
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )

        target_matrix = self.get_target_matrix(format)
        targets = self.targets
        if isinstance(self, CGate):
            controls = self.controls
        else:
            controls = []
        
        # 生成 Z 基础表示
        m = represent_zbasis(
            controls, targets, target_matrix, nqubits, format
        )
        return m
    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------
    
    # 返回符号表达式形式的字符串表示，包括门的名称和标签
    def _sympystr(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s(%s)' % (self.gate_name, label)
    
    # 返回美观的打印形式，使用字符串图形和标签的美化输出
    def _pretty(self, printer, *args):
        a = stringPict(self.gate_name)
        b = self._print_label_pretty(printer, *args)
        return self._print_subscript_pretty(a, b)
    
    # 返回 LaTeX 格式的字符串表示，包括门的 LaTeX 名称和标签
    def _latex(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s_{%s}' % (self.gate_name_latex, label)
    
    # 绘制门的图形表示，抛出未实现错误
    def plot_gate(self, axes, gate_idx, gate_grid, wire_grid):
        raise NotImplementedError('plot_gate is not implemented.')
class CGate(Gate):
    """A general unitary gate with control qubits.

    A general control gate applies a target gate to a set of targets if all
    of the control qubits have a particular values (set by
    ``CGate.control_value``).

    Parameters
    ----------
    label : tuple
        The label in this case has the form (controls, gate), where controls
        is a tuple/list of control qubits (as ints) and gate is a ``Gate``
        instance that is the target operator.

    Examples
    ========

    """

    gate_name = 'C'
    gate_name_latex = 'C'

    # The values this class controls for.
    control_value = _S.One

    simplify_cgate = False

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # _eval_args has the right logic for the controls argument.
        controls = args[0]
        gate = args[1]
        if not is_sequence(controls):
            controls = (controls,)
        # Ensure controls are correctly formatted and validated
        controls = UnitaryOperator._eval_args(controls)
        # Validate that the targets and controls are appropriate
        _validate_targets_controls(chain(controls, gate.targets))
        return (Tuple(*controls), gate)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        # Compute the Hilbert space required for the gate operation
        return ComplexSpace(2)**_max(_max(args[0]) + 1, args[1].min_qubits)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        # Calculate the total number of qubits affected by this gate
        return len(self.targets) + len(self.controls)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        # Determine the minimum number of qubits required for the gate
        return _max(_max(self.controls), _max(self.targets)) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        # Retrieve the target qubits for this gate
        return self.gate.targets

    @property
    def controls(self):
        """A tuple of control qubits."""
        # Retrieve the control qubits from the gate's label
        return tuple(self.label[0])

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        # Retrieve the gate object from the label
        return self.label[1]

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        # Obtain the matrix representation of the gate for the targets
        return self.gate.get_target_matrix(format)

    def eval_controls(self, qubit):
        """Return True/False to indicate if the controls are satisfied."""
        # Evaluate if the control conditions are met for the given qubit state
        return all(qubit[bit] == self.control_value for bit in self.controls)
    # 将控制门分解为 CNOT 和单量子比特门
    def decompose(self, **options):
        """Decompose the controlled gate into CNOT and single qubits gates."""
        # 如果控制量子比特的数量为1
        if len(self.controls) == 1:
            # 取第一个控制量子比特和目标量子比特
            c = self.controls[0]
            t = self.gate.targets[0]
            # 如果门是 Y 门类型
            if isinstance(self.gate, YGate):
                # 分解为相位门、CNOT门、再次相位门和 Z 门的乘积
                g1 = PhaseGate(t)
                g2 = CNotGate(c, t)
                g3 = PhaseGate(t)
                g4 = ZGate(t)
                return g1 * g2 * g3 * g4
            # 如果门是 Z 门类型
            if isinstance(self.gate, ZGate):
                # 分解为Hadamard门、CNOT门和再次Hadamard门的乘积
                g1 = HadamardGate(t)
                g2 = CNotGate(c, t)
                g3 = HadamardGate(t)
                return g1 * g2 * g3
        else:
            # 如果控制量子比特的数量不为1，直接返回自身
            return self

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _print_label(self, printer, *args):
        # 打印标签方法，格式为"(controls),gate"
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return '(%s),%s' % (controls, gate)

    def _pretty(self, printer, *args):
        # 打印美化输出方法
        controls = self._print_sequence_pretty(
            self.controls, ',', printer, *args)
        gate = printer._print(self.gate)
        gate_name = stringPict(self.gate_name)
        first = self._print_subscript_pretty(gate_name, controls)
        gate = self._print_parens_pretty(gate)
        final = prettyForm(*first.right(gate))
        return final

    def _latex(self, printer, *args):
        # 打印 LaTeX 格式方法
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return r'%s_{%s}{\left(%s\right)}' % \
            (self.gate_name_latex, controls, gate)

    def plot_gate(self, circ_plot, gate_idx):
        """
        Plot the controlled gate. If *simplify_cgate* is true, simplify
        C-X and C-Z gates into their more familiar forms.
        """
        # 计算最小和最大的量子线
        min_wire = int(_min(chain(self.controls, self.targets)))
        max_wire = int(_max(chain(self.controls, self.targets)))
        # 在电路图上绘制控制线
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        # 绘制控制点
        for c in self.controls:
            circ_plot.control_point(gate_idx, int(c))
        # 如果需要简化控制门
        if self.simplify_cgate:
            # 如果门是 X 门类型，绘制简化的加门
            if self.gate.gate_name == 'X':
                self.gate.plot_gate_plus(circ_plot, gate_idx)
            # 如果门是 Z 门类型，绘制控制点
            elif self.gate.gate_name == 'Z':
                circ_plot.control_point(gate_idx, self.targets[0])
            else:
                # 其他情况下，绘制原始门
                self.gate.plot_gate(circ_plot, gate_idx)
        else:
            # 如果不需要简化，绘制原始门
            self.gate.plot_gate(circ_plot, gate_idx)

    #-------------------------------------------------------------------------
    # Miscellaneous
    #-------------------------------------------------------------------------

    def _eval_dagger(self):
        # 如果门是 HermitianOperator 类型，返回自身
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            # 否则，调用 Gate 的 _eval_dagger 方法
            return Gate._eval_dagger(self)
    # 定义一个方法 `_eval_inverse`，用于计算逆操作
    def _eval_inverse(self):
        # 如果当前门是 HermitianOperator 类型，则返回自身
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            # 否则调用父类 Gate 的 `_eval_inverse` 方法
            return Gate._eval_inverse(self)

    # 定义一个方法 `_eval_power`，用于计算门的指数幂操作
    def _eval_power(self, exp):
        # 如果当前门是 HermitianOperator 类型
        if isinstance(self.gate, HermitianOperator):
            # 如果指数 exp 等于 -1，返回当前门的逆操作
            if exp == -1:
                return Gate._eval_power(self, exp)
            # 如果指数 exp 是偶数，则返回当前门与其逆的乘积
            elif abs(exp) % 2 == 0:
                return self * (Gate._eval_inverse(self))
            # 如果指数 exp 是奇数，则返回当前门自身
            else:
                return self
        else:
            # 如果当前门不是 HermitianOperator 类型，则调用父类 Gate 的 `_eval_power` 方法
            return Gate._eval_power(self, exp)
class CGateS(CGate):
    """Version of CGate that allows gate simplifications.
    I.e. cnot looks like an oplus, cphase has dots, etc.
    """
    # CGateS 类继承自 CGate，支持门的简化表示
    simplify_cgate=True


class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    # UGate 类，表示通用的量子门，由目标和目标矩阵定义

    gate_name = 'U'
    gate_name_latex = 'U'

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        targets = args[0]
        if not is_sequence(targets):
            targets = (targets,)
        targets = Gate._eval_args(targets)
        _validate_targets_controls(targets)
        mat = args[1]
        if not isinstance(mat, MatrixBase):
            raise TypeError('Matrix expected, got: %r' % mat)
        # 确保矩阵类型为 Basic
        mat = _sympify(mat)
        dim = 2**len(targets)
        if not all(dim == shape for shape in mat.shape):
            raise IndexError(
                'Number of targets must match the matrix size: %r %r' %
                (targets, mat)
            )
        return (targets, mat)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        # 返回可能的最小希尔伯特空间
        return ComplexSpace(2)**(_max(args[0]) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        """A tuple of target qubits."""
        # 返回目标量子比特的元组
        return tuple(self.label[0])

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        # 返回门目标部分的矩阵表示
        return self.label[1]

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _pretty(self, printer, *args):
        targets = self._print_sequence_pretty(
            self.targets, ',', printer, *args)
        gate_name = stringPict(self.gate_name)
        return self._print_subscript_pretty(gate_name, targets)

    def _latex(self, printer, *args):
        targets = self._print_sequence(self.targets, ',', printer, *args)
        return r'%s_{%s}' % (self.gate_name_latex, targets)
    # 定义一个方法用于在图形化界面中绘制量子门
    def plot_gate(self, circ_plot, gate_idx):
        # 调用图形化界面的方法，绘制单量子门框
        circ_plot.one_qubit_box(
            # 使用量子门的名称来标识绘制的门
            self.gate_name_plot,
            # 门的索引位置，用于确定门的位置
            gate_idx,
            # 将目标量子比特的索引转换为整数，并作为绘制门的参数之一
            int(self.targets[0])
        )
class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""

    nqubits = _S.One  # 指定门作用的量子比特数为1

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,  # 在量子电路图中绘制单量子比特门的图形
            gate_idx, int(self.targets[0])  # gate_idx是门的索引，self.targets[0]是目标量子比特的索引
        )

    def _eval_commutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return _S.Zero  # 如果与另一个单量子比特门的目标或类别不同，则返回零算符
        return Operator._eval_commutator(self, other, **hints)  # 否则调用父类的算符对易子计算方法

    def _eval_anticommutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return Integer(2)*self*other  # 如果与另一个单量子比特门的目标或类别不同，则返回2倍的反对易子
        return Operator._eval_anticommutator(self, other, **hints)  # 否则调用父类的算符反对易子计算方法


class TwoQubitGate(Gate):
    """A two qubit unitary gate base class."""

    nqubits = Integer(2)  # 指定门作用的量子比特数为2

#-----------------------------------------------------------------------------
# Single Qubit Gates
#-----------------------------------------------------------------------------


class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = True  # 表示该门是自共轭的
    gate_name = '1'  # 门的名称
    gate_name_latex = '1'  # LaTeX显示的门的名称

    # Short cut version of gate._apply_operator_Qubit
    def _apply_operator_Qubit(self, qubits, **options):
        # 检查门作用的量子比特数是否符合要求
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )
        return qubits  # 对于IdentityGate，不需要进行任何计算，直接返回量子比特对象

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('eye2', format)  # 获取单位矩阵的表示

    def _eval_commutator(self, other, **hints):
        return _S.Zero  # IdentityGate与任何门的对易子都为零

    def _eval_anticommutator(self, other, **hints):
        return Integer(2)*other  # IdentityGate与其他门的反对易子为2倍的other（另一个门的表示）


class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name = 'H'  # 门的名称
    gate_name_latex = 'H'  # LaTeX显示的门的名称
    # 获取目标矩阵的方法，根据给定格式返回矩阵
    def get_target_matrix(self, format='sympy'):
        # 如果_normalized为真，则返回'H'对应的格式化矩阵
        if _normalized:
            return matrix_cache.get_matrix('H', format)
        else:
            # 否则返回'Hsqrt2'对应的格式化矩阵
            return matrix_cache.get_matrix('Hsqrt2', format)

    # 计算X门与另一个门的对易子
    def _eval_commutator_XGate(self, other, **hints):
        # 返回I*sqrt(2)*YGate(self.targets[0])的结果
        return I*sqrt(2)*YGate(self.targets[0])

    # 计算Y门与另一个门的对易子
    def _eval_commutator_YGate(self, other, **hints):
        # 返回I*sqrt(2)*(ZGate(self.targets[0]) - XGate(self.targets[0]))的结果
        return I*sqrt(2)*(ZGate(self.targets[0]) - XGate(self.targets[0]))

    # 计算Z门与另一个门的对易子
    def _eval_commutator_ZGate(self, other, **hints):
        # 返回-I*sqrt(2)*YGate(self.targets[0])的结果
        return -I*sqrt(2)*YGate(self.targets[0])

    # 计算X门与另一个门的反对易子
    def _eval_anticommutator_XGate(self, other, **hints):
        # 返回sqrt(2)*IdentityGate(self.targets[0])的结果
        return sqrt(2)*IdentityGate(self.targets[0])

    # 计算Y门与另一个门的反对易子
    def _eval_anticommutator_YGate(self, other, **hints):
        # 返回零矩阵的结果
        return _S.Zero

    # 计算Z门与另一个门的反对易子
    def _eval_anticommutator_ZGate(self, other, **hints):
        # 返回sqrt(2)*IdentityGate(self.targets[0])的结果
        return sqrt(2)*IdentityGate(self.targets[0])
class XGate(HermitianOperator, OneQubitGate):
    """The single qubit X, or NOT, gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'X'  # 定义门的名称为'X'
    gate_name_latex = 'X'  # 定义门的 LaTeX 显示名称为'X'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('X', format)  # 获取目标矩阵 'X' 的表示

    def plot_gate(self, circ_plot, gate_idx):
        OneQubitGate.plot_gate(self, circ_plot, gate_idx)  # 绘制门的图形

    def plot_gate_plus(self, circ_plot, gate_idx):
        circ_plot.not_point(
            gate_idx, int(self.label[0])
        )  # 绘制带有指定标签的 NOT 门图形

    def _eval_commutator_YGate(self, other, **hints):
        return Integer(2)*I*ZGate(self.targets[0])  # 计算与 Y 门的对易子

    def _eval_anticommutator_XGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])  # 计算与另一 X 门的反对易子

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与另一 Y 门的反对易子为零

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与 Z 门的反对易子为零


class YGate(HermitianOperator, OneQubitGate):
    """The single qubit Y gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Y'  # 定义门的名称为'Y'
    gate_name_latex = 'Y'  # 定义门的 LaTeX 显示名称为'Y'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Y', format)  # 获取目标矩阵 'Y' 的表示

    def _eval_commutator_ZGate(self, other, **hints):
        return Integer(2)*I*XGate(self.targets[0])  # 计算与 Z 门的对易子

    def _eval_anticommutator_YGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])  # 计算与另一 Y 门的反对易子

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与 Z 门的反对易子为零


class ZGate(HermitianOperator, OneQubitGate):
    """The single qubit Z gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Z'  # 定义门的名称为'Z'
    gate_name_latex = 'Z'  # 定义门的 LaTeX 显示名称为'Z'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Z', format)  # 获取目标矩阵 'Z' 的表示

    def _eval_commutator_XGate(self, other, **hints):
        return Integer(2)*I*YGate(self.targets[0])  # 计算与 X 门的对易子

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与 Y 门的反对易子为零


class PhaseGate(OneQubitGate):
    """The single qubit phase, or S, gate.

    This gate rotates the phase of the state by pi/2 if the state is ``|1>``
    and does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian =  False  # 标记这个门不是自共轭的
    gate_name = 'S'  # 定义门的名称为'S'
    gate_name_latex = 'S'  # 定义门的 LaTeX 显示名称为'S'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('S', format)  # 获取目标矩阵 'S' 的表示

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与 Z 门的对易子为零

    def _eval_commutator_TGate(self, other, **hints):
        return _S.Zero  # 返回零，表示与 T 门的对易子为零


class TGate(OneQubitGate):
    """The single qubit pi/8 gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    """
    This gate rotates the phase of the state by pi/4 if the state is |1> and
    does nothing if the state is |0>.
    
    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.
    
    Examples
    ========
    
    """
    
    # 是否是厄米矩阵，这里默认为 False
    is_hermitian = False
    # 门的名称为 'T'
    gate_name = 'T'
    # LaTeX 格式下的门的名称也是 'T'
    gate_name_latex = 'T'
    
    # 获取目标矩阵的函数，根据指定格式返回矩阵
    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('T', format)
    
    # 计算本门与 Z 门的对易子，结果为零
    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero
    
    # 计算本门与 PhaseGate 的对易子，结果为零
    def _eval_commutator_PhaseGate(self, other, **hints):
        return _S.Zero
# 别名用于门的名称。
H = HadamardGate  # 将 'H' 作为 Hadamard 门的别名
X = XGate  # 将 'X' 作为 X 门的别名
Y = YGate  # 将 'Y' 作为 Y 门的别名
Z = ZGate  # 将 'Z' 作为 Z 门的别名
T = TGate  # 将 'T' 作为 T 门的别名
Phase = S = PhaseGate  # 将 'Phase' 和 'S' 都作为 PhaseGate 的别名

#-----------------------------------------------------------------------------
# 2 Qubit Gates
#-----------------------------------------------------------------------------

class CNotGate(HermitianOperator, CGate, TwoQubitGate):
    """Two qubit controlled-NOT.

    This gate performs the NOT or X gate on the target qubit if the control
    qubits all have the value 1.

    Parameters
    ----------
    label : tuple
        A tuple of the form (control, target).

    Examples
    ========

    >>> from sympy.physics.quantum.gate import CNOT
    >>> from sympy.physics.quantum.qapply import qapply
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> c = CNOT(1,0)
    >>> qapply(c*Qubit('10')) # note that qubits are indexed from right to left
    |11>

    """
    gate_name = 'CNOT'  # 门的名称为 'CNOT'
    gate_name_latex = r'\text{CNOT}'  # LaTeX 表示为 '\text{CNOT}'
    simplify_cgate = True  # 启用简化 C 门操作

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Gate._eval_args(args)  # 调用 Gate 类的 _eval_args 方法处理参数
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)  # 返回最小可能的 Hilbert 空间大小

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.label) + 1  # 返回此门需要作用的最小量子比特数目

    @property
    def targets(self):
        """A tuple of target qubits."""
        return (self.label[1],)  # 返回目标量子比特的元组

    @property
    def controls(self):
        """A tuple of control qubits."""
        return (self.label[0],)  # 返回控制量子比特的元组

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return XGate(self.label[1])  # 返回将应用于目标的非受控门 XGate

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    # The default printing of Gate works better than those of CGate, so we
    # go around the overridden methods in CGate.

    def _print_label(self, printer, *args):
        return Gate._print_label(self, printer, *args)  # 使用 Gate 类的 _print_label 方法打印标签

    def _pretty(self, printer, *args):
        return Gate._pretty(self, printer, *args)  # 使用 Gate 类的 _pretty 方法美化打印

    def _latex(self, printer, *args):
        return Gate._latex(self, printer, *args)  # 使用 Gate 类的 _latex 方法生成 LaTeX 表示

    #-------------------------------------------------------------------------
    # Commutator/AntiCommutator
    #-------------------------------------------------------------------------
    # 定义一个方法用于计算 CNOT 门与其他量子门的对易子
    def _eval_commutator_ZGate(self, other, **hints):
        """[CNOT(i, j), Z(i)] == 0."""
        # 如果第一个控制位与目标位相同，则对易子结果为零
        if self.controls[0] == other.targets[0]:
            return _S.Zero
        else:
            # 否则抛出未实现错误
            raise NotImplementedError('Commutator not implemented: %r' % other)

    # 定义计算 CNOT 门与 T 门对易子的方法
    def _eval_commutator_TGate(self, other, **hints):
        """[CNOT(i, j), T(i)] == 0."""
        # 调用 _eval_commutator_ZGate 方法计算对易子，结果为零
        return self._eval_commutator_ZGate(other, **hints)

    # 定义计算 CNOT 门与 PhaseGate 对易子的方法
    def _eval_commutator_PhaseGate(self, other, **hints):
        """[CNOT(i, j), S(i)] == 0."""
        # 调用 _eval_commutator_ZGate 方法计算对易子，结果为零
        return self._eval_commutator_ZGate(other, **hints)

    # 定义计算 CNOT 门与 X 门对易子的方法
    def _eval_commutator_XGate(self, other, **hints):
        """[CNOT(i, j), X(j)] == 0."""
        # 如果第一个目标位与其他门的目标位相同，则对易子结果为零
        if self.targets[0] == other.targets[0]:
            return _S.Zero
        else:
            # 否则抛出未实现错误
            raise NotImplementedError('Commutator not implemented: %r' % other)

    # 定义计算 CNOT 门与 CNOT 门对易子的方法
    def _eval_commutator_CNotGate(self, other, **hints):
        """[CNOT(i, j), CNOT(i,k)] == 0."""
        # 如果第一个控制位与其他门的控制位相同，则对易子结果为零
        if self.controls[0] == other.controls[0]:
            return _S.Zero
        else:
            # 否则抛出未实现错误
            raise NotImplementedError('Commutator not implemented: %r' % other)
class SwapGate(TwoQubitGate):
    """Two qubit SWAP gate.

    This gate swaps the values of the two qubits.

    Parameters
    ----------
    label : tuple
        A tuple of the form (target1, target2).

    Examples
    ========

    """
    # 设置门是否为共轭转置
    is_hermitian = True
    # 设置门的名称
    gate_name = 'SWAP'
    # 设置门的 LaTeX 表示
    gate_name_latex = r'\text{SWAP}'

    def get_target_matrix(self, format='sympy'):
        # 返回缓存中 'SWAP' 的矩阵表示
        return matrix_cache.get_matrix('SWAP', format)

    def decompose(self, **options):
        """Decompose the SWAP gate into CNOT gates."""
        # 获取目标量子比特的索引
        i, j = self.targets[0], self.targets[1]
        # 创建两个 CNOT 门来分解 SWAP 门
        g1 = CNotGate(i, j)
        g2 = CNotGate(j, i)
        return g1*g2*g1

    def plot_gate(self, circ_plot, gate_idx):
        # 计算最小和最大的目标量子比特索引
        min_wire = int(_min(self.targets))
        max_wire = int(_max(self.targets))
        # 在电路图上绘制控制线
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        # 在电路图上标注 SWAP 点
        circ_plot.swap_point(gate_idx, min_wire)
        circ_plot.swap_point(gate_idx, max_wire)

    def _represent_ZGate(self, basis, **options):
        """Represent the SWAP gate in the computational basis.

        The following representation is used to compute this:

        SWAP = |1><1|x|1><1| + |0><0|x|0><0| + |1><0|x|0><1| + |0><1|x|1><0|
        """
        # 获取格式选项（默认为 'sympy'）
        format = options.get('format', 'sympy')
        # 将目标量子比特索引转换为整数类型
        targets = [int(t) for t in self.targets]
        # 计算最小和最大的目标量子比特索引
        min_target = _min(targets)
        max_target = _max(targets)
        # 获取选项中的量子比特数或者使用最小量子比特数
        nqubits = options.get('nqubits', self.min_qubits)

        # 从缓存中获取四种操作矩阵
        op01 = matrix_cache.get_matrix('op01', format)
        op10 = matrix_cache.get_matrix('op10', format)
        op11 = matrix_cache.get_matrix('op11', format)
        op00 = matrix_cache.get_matrix('op00', format)
        eye2 = matrix_cache.get_matrix('eye2', format)

        result = None
        # 对四种不同的操作矩阵进行迭代
        for i, j in ((op01, op10), (op10, op01), (op00, op00), (op11, op11)):
            # 创建包含单位矩阵的列表
            product = nqubits*[eye2]
            # 替换列表中的特定位置为对应的操作矩阵
            product[nqubits - min_target - 1] = i
            product[nqubits - max_target - 1] = j
            # 计算张量积
            new_result = matrix_tensor_product(*product)
            if result is None:
                result = new_result
            else:
                result = result + new_result

        return result


# 门名称的别名
CNOT = CNotGate
SWAP = SwapGate

def CPHASE(a,b): return CGateS((a,),Z(b))


#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def represent_zbasis(controls, targets, target_matrix, nqubits, format='sympy'):
    """Represent a gate with controls, targets and target_matrix.

    This function does the low-level work of representing gates as matrices
    in the standard computational basis (ZGate). Currently, we support two
    main cases:

    1. One target qubit and no control qubits.
    2. One target qubits and multiple control qubits.

    For the base of multiple controls, we use the following expression [1]:

    1_{2**n} + (|1><1|)^{(n-1)} x (target-matrix - 1_{2})

    """
    # 表示具有控制、目标和目标矩阵的门的矩阵表示
    # 此函数负责以标准计算基础（ZGate）中的矩阵形式表示门的低级工作
    # 目前我们支持两种主要情况：
    # 1. 一个目标量子比特且没有控制量子比特。
    # 2. 一个目标量子比特和多个控制量子比特。
    # 对于多个控制的基础，我们使用以下表达式 [1]：
    # 1_{2**n} + (|1><1|)^{(n-1)} x (target-matrix - 1_{2})
    pass
    # 将控制量子位列表转换为整数列表
    controls = [int(x) for x in controls]
    # 将目标量子位列表转换为整数列表
    targets = [int(x) for x in targets]
    # 将量子比特数转换为整数
    nqubits = int(nqubits)

    # 获取格式化后的操作矩阵 'op11'
    op11 = matrix_cache.get_matrix('op11', format)
    # 获取格式化后的单位矩阵 'eye2'
    eye2 = matrix_cache.get_matrix('eye2', format)

    # 单控制量子位，单目标量子位情况
    if len(controls) == 0 and len(targets) == 1:
        product = []
        bit = targets[0]
        # 填充 product 列表为 [I1,Gate,I2]，使得单位矩阵 I 能正确应用于 Gate
        if bit != nqubits - 1:
            product.append(matrix_eye(2**(nqubits - bit - 1), format=format))
        product.append(target_matrix)
        if bit != 0:
            product.append(matrix_eye(2**bit, format=format))
        # 返回张量积结果
        return matrix_tensor_product(*product)

    # 单目标量子位，多控制量子位情况
    elif len(targets) == 1 and len(controls) >= 1:
        target = targets[0]

        # 构建非平凡部分
        product2 = []
        for i in range(nqubits):
            product2.append(matrix_eye(2, format=format))
        for control in controls:
            product2[nqubits - 1 - control] = op11
        product2[nqubits - 1 - target] = target_matrix - eye2

        # 返回张量积结果
        return matrix_eye(2**nqubits, format=format) + \
            matrix_tensor_product(*product2)

    # 尚未实现多目标量子位，多控制量子位的情况
    else:
        raise NotImplementedError(
            'The representation of multi-target, multi-control gates '
            'is not implemented.'
        )
#-----------------------------------------------------------------------------
# Gate manipulation functions.
#-----------------------------------------------------------------------------


def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """

    # 使用 gate_sort 对电路中的门进行排序。
    circuit = gate_sort(circuit)

    # 如果电路是加法表达式，对每个项递归调用 gate_simp 并求和。
    if isinstance(circuit, Add):
        return sum(gate_simp(t) for t in circuit.args)
    # 如果电路是乘法表达式，获取乘法表达式的参数列表。
    elif isinstance(circuit, Mul):
        circuit_args = circuit.args
    # 如果电路是幂次表达式，分解为基数和指数，对基数递归调用 gate_simp。
    elif isinstance(circuit, Pow):
        b, e = circuit.as_base_exp()
        circuit_args = (gate_simp(b)**e,)
    else:
        # 如果电路不是复合表达式，则直接返回。
        return circuit

    # 逐个遍历电路中的每个元素，进行简化处理（如果可能）。
    # 对于传入的 circuit_args 列表进行遍历
    for i in range(len(circuit_args)):
        # 检查当前元素是否是 Pow 对象（指数运算）
        if isinstance(circuit_args[i], Pow):
            # 如果底数是 HGate、XGate、YGate 或者 ZGate，并且指数是数字类型
            if isinstance(circuit_args[i].base,
                (HadamardGate, XGate, YGate, ZGate)) \
                    and isinstance(circuit_args[i].exp, Number):
                # 构建一个新的电路，将 HGate、XGate、YGate 或者 ZGate 的平方替换为 1
                newargs = (circuit_args[:i] +
                          (circuit_args[i].base**(circuit_args[i].exp % 2),) +
                           circuit_args[i + 1:])
                # 递归简化新电路
                circuit = gate_simp(Mul(*newargs))
                # 中断循环，返回简化后的电路
                break
            # 如果底数是 PhaseGate
            elif isinstance(circuit_args[i].base, PhaseGate):
                # 构建一个新的电路，将 PhaseGate 的平方替换为 ZGate
                newargs = circuit_args[:i]
                newargs = newargs + (ZGate(circuit_args[i].base.args[0])**
                (Integer(circuit_args[i].exp/2)), circuit_args[i].base**
                (circuit_args[i].exp % 2))
                newargs = newargs + circuit_args[i + 1:]
                # 递归简化新电路
                circuit = gate_simp(Mul(*newargs))
                # 中断循环，返回简化后的电路
                break
            # 如果底数是 TGate
            elif isinstance(circuit_args[i].base, TGate):
                # 构建一个新的电路，将 TGate 的平方替换为 PhaseGate
                newargs = circuit_args[:i]
                newargs = newargs + (PhaseGate(circuit_args[i].base.args[0])**
                Integer(circuit_args[i].exp/2), circuit_args[i].base**
                    (circuit_args[i].exp % 2))
                newargs = newargs + circuit_args[i + 1:]
                # 递归简化新电路
                circuit = gate_simp(Mul(*newargs))
                # 中断循环，返回简化后的电路
                break
    # 返回简化后的电路
    return circuit
# 定义一个函数，用于对量子门进行排序，并保持交换关系的跟踪
def gate_sort(circuit):
    """Sorts the gates while keeping track of commutation relations

    This function uses a bubble sort to rearrange the order of gate
    application. Keeps track of Quantum computations special commutation
    relations (e.g. things that apply to the same Qubit do not commute with
    each other)

    circuit is the Mul of gates that are to be sorted.
    """
    # 如果 circuit 是 Add 类型，则对其内部的每个元素递归调用 gate_sort，并返回它们的和
    if isinstance(circuit, Add):
        return sum(gate_sort(t) for t in circuit.args)
    # 如果 circuit 是 Pow 类型，则对其 base 进行 gate_sort 后对 exp 次方
    if isinstance(circuit, Pow):
        return gate_sort(circuit.base)**circuit.exp
    # 如果 circuit 是 Gate 类型，则直接返回它
    elif isinstance(circuit, Gate):
        return circuit
    # 如果 circuit 不是 Mul 类型，则直接返回它
    if not isinstance(circuit, Mul):
        return circuit

    # 初始化变量，用于控制循环
    changes = True
    while changes:
        changes = False
        # 获取 circuit 的所有子元素
        circ_array = circuit.args
        # 遍历子元素数组
        for i in range(len(circ_array) - 1):
            # 如果当前元素和下一个元素都是 Gate 或 Pow 类型
            if isinstance(circ_array[i], (Gate, Pow)) and \
                    isinstance(circ_array[i + 1], (Gate, Pow)):
                # 如果 circ_array[i] 和 circ_array[i + 1] 是 Pow 类型，获取它们的 base 和 exp
                first_base, first_exp = circ_array[i].as_base_exp()
                second_base, second_exp = circ_array[i + 1].as_base_exp()

                # 使用 SymPy 的基于哈希值的排序机制。这不是数学上的排序，而是基于对象哈希值的比较
                # 参见 Basic.compare 的详细说明
                if first_base.compare(second_base) > 0:
                    # 如果两者的基相互 commute，则交换它们的顺序
                    if Commutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                    (circuit.args[i],) + circuit.args[i + 2:])
                        circuit = Mul(*new_args)
                        changes = True
                        break
                    # 如果两者的基相互 anti-commute，则交换它们的顺序并添加符号
                    if AntiCommutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                    (circuit.args[i],) + circuit.args[i + 2:])
                        sign = _S.NegativeOne**(first_exp*second_exp)
                        circuit = sign*Mul(*new_args)
                        changes = True
                        break
    return circuit


#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------


def random_circuit(ngates, nqubits, gate_space=(X, Y, Z, S, T, H, CNOT, SWAP)):
    """Return a random circuit of ngates and nqubits.

    This uses an equally weighted sample of (X, Y, Z, S, T, H, CNOT, SWAP)
    gates.

    Parameters
    ----------
    ngates : int
        The number of gates in the circuit.
    nqubits : int
        The number of qubits in the circuit.
    # gate_space 是一个元组，包含了电路中将要使用的门类。
    # 多次重复相同的门类会增加它们在随机电路中出现的频率。
    """
    qubit_space = range(nqubits)
    result = []
    for i in range(ngates):
        # 从 gate_space 中随机选择一个门类 g
        g = random.choice(gate_space)
        # 如果选择的门类是 CNotGate 或 SwapGate，则需要两个随机选取的量子比特
        if g == CNotGate or g == SwapGate:
            qubits = random.sample(qubit_space, 2)
            # 创建 g 的实例，传入选取的两个量子比特
            g = g(*qubits)
        else:
            # 否则，选择一个随机的量子比特
            qubit = random.choice(qubit_space)
            # 创建 g 的实例，传入选取的量子比特
            g = g(qubit)
        # 将创建的门添加到结果列表中
        result.append(g)
    # 返回结果列表中所有门类的乘积
    return Mul(*result)
# 从 Z 到 X 基础的转换矩阵函数
def zx_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to X basis."""
    # 使用 matrix_cache 中的函数获取 'ZX' 格式的转换矩阵，并以指定格式返回
    return matrix_cache.get_matrix('ZX', format)


# 从 Z 到 Y 基础的转换矩阵函数
def zy_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to Y basis."""
    # 使用 matrix_cache 中的函数获取 'ZY' 格式的转换矩阵，并以指定格式返回
    return matrix_cache.get_matrix('ZY', format)
```