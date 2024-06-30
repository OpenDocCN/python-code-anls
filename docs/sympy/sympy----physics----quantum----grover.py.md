# `D:\src\scipysrc\sympy\sympy\physics\quantum\grover.py`

```
# 导入必要的模块和函数
from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit

# 定义导出的模块列表
__all__ = [
    'OracleGate',
    'WGate',
    'superposition_basis',
    'grover_iteration',
    'apply_grover'
]

# 创建一个相等叠加态的函数，表示计算基的相等叠加态
def superposition_basis(nqubits):
    """Creates an equal superposition of the computational basis.

    Parameters
    ==========

    nqubits : int
        The number of qubits.

    Returns
    =======

    state : Qubit
        An equal superposition of the computational basis with nqubits.

    Examples
    ========

    Create an equal superposition of 2 qubits::

        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> superposition_basis(2)
        |0>/2 + |1>/2 + |2>/2 + |3>/2
    """

    amp = 1/sqrt(2**nqubits)
    return sum(amp*IntQubit(n, nqubits=nqubits) for n in range(2**nqubits))

# 定义一个包装器类，用于包装在 OracleGate 中使用的 Python 函数
class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        return type(self), self.function

    def __call__(self, *args):
        return self.function(*args)

# 定义一个 OracleGate 类，代表一个黑匣子门
class OracleGate(Gate):
    """A black box gate.

    The gate marks the desired qubits of an unknown function by flipping
    the sign of the qubits.  The unknown function returns true when it
    finds its desired qubits and false otherwise.

    Parameters
    ==========

    qubits : int
        Number of qubits.

    oracle : callable
        A callable function that returns a boolean on a computational basis.

    Examples
    ========

    Apply an Oracle gate that flips the sign of ``|2>`` on different qubits::

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(2, f)
        >>> qapply(v*IntQubit(2))
        -|2>
        >>> qapply(v*IntQubit(3))
        |3>
    """
    gate_name = 'V'
    gate_name_latex = 'V'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # 检查参数数量是否正确，必须为两个参数
        if len(args) != 2:
            raise QuantumError(
                'Insufficient/excessive arguments to Oracle.  Please ' +
                'supply the number of qubits and an unknown function.'
            )
        sub_args = (args[0],)
        # 对第一个参数进行进一步验证
        sub_args = UnitaryOperator._eval_args(sub_args)
        # 确保第一个参数为整数
        if not sub_args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % sub_args[0])

        function = args[1]
        # 如果第二个参数不是 OracleGateFunction 类型，则将其转换为 OracleGateFunction 对象
        if not isinstance(function, OracleGateFunction):
            function = OracleGateFunction(function)

        return (sub_args[0], function)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        # 根据参数创建最小可能的 Hilbert 空间
        return ComplexSpace(2)**args[0]

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def search_function(self):
        """The unknown function that helps find the sought after qubits."""
        # 返回用于搜索目标量子比特的未知函数
        return self.label[1]

    @property
    def targets(self):
        """A tuple of target qubits."""
        # 返回一个包含目标量子比特的元组
        return sympify(tuple(range(self.args[0])))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this operator to a Qubit subclass.

        Parameters
        ==========

        qubits : Qubit
            The qubit subclass to apply this operator to.

        Returns
        =======

        state : Expr
            The resulting quantum state.
        """
        # 检查传入的量子比特数量是否与操作符所需数量一致
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'OracleGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )
        # 如果搜索函数在给定的量子比特上返回 1，则返回量子比特的负数（翻转符号）
        if self.search_function(qubits):
            return -qubits
        else:
            return qubits

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------
    def _represent_ZGate(self, basis, **options):
        """
        Represent the OracleGate in the computational basis.
        """
        nbasis = 2**self.nqubits  # 计算基数，仅计算一次

        # 创建一个单位矩阵作为 OracleGate 的初始表示
        matrixOracle = eye(nbasis)

        # 根据 Oracle 函数的输出，翻转对应位置的矩阵元素的符号
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()

        # 返回表示后的 OracleGate 矩阵
        return matrixOracle
class WGate(Gate):
    """General n qubit W Gate in Grover's algorithm.

    The gate performs the operation ``2|phi><phi| - 1`` on some qubits.
    ``|phi> = (tensor product of n Hadamards)*(|0> with n qubits)``

    Parameters
    ==========

    nqubits : int
        The number of qubits to operate on

    """

    gate_name = 'W'
    gate_name_latex = 'W'

    @classmethod
    def _eval_args(cls, args):
        # 检查参数列表长度是否为1
        if len(args) != 1:
            raise QuantumError(
                'Insufficient/excessive arguments to W gate.  Please ' +
                'supply the number of qubits to operate on.'
            )
        # 对参数进行进一步评估和处理
        args = UnitaryOperator._eval_args(args)
        # 确保参数是整数类型
        if not args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % args[0])
        return args

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        # 返回目标qubits，反向排列的元组
        return sympify(tuple(reversed(range(self.args[0]))))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """
        qubits: a set of qubits (Qubit)
        Returns: quantum object (quantum expression - QExpr)
        """
        # 检查操作的qubits数量是否与预期一致
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'WGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )

        # 根据 Mermin 的书籍，应用 WGate 操作的结果
        # 返回 (2/(sqrt(2^n)))|phi> - |a>，其中|a>是当前基态，phi是基态的叠加
        basis_states = superposition_basis(self.nqubits)
        change_to_basis = (2/sqrt(2**self.nqubits))*basis_states
        return change_to_basis - qubits


def grover_iteration(qstate, oracle):
    """Applies one application of the Oracle and W Gate, WV.

    Parameters
    ==========

    qstate : Qubit
        A superposition of qubits.
    oracle : OracleGate
        The black box operator that flips the sign of the desired basis qubits.

    Returns
    =======

    Qubit : The qubits after applying the Oracle and W gate.

    Examples
    ========
    """
    Perform one iteration of grover's algorithm to see a phase change::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> from sympy.physics.quantum.grover import grover_iteration
        >>> numqubits = 2
        >>> basis_states = superposition_basis(numqubits)  # 生成包含所有叠加基础状态的列表
        >>> f = lambda qubits: qubits == IntQubit(2)  # 定义一个 lambda 函数，表示目标状态
        >>> v = OracleGate(numqubits, f)  # 创建一个包含目标状态的 Oracle 门
        >>> qapply(grover_iteration(basis_states, v))  # 应用 Grover 迭代算法，计算新的量子状态
        |2>  # 返回最终的量子状态

    """
    wgate = WGate(oracle.nqubits)  # 根据 Oracle 的量子比特数创建一个 W 门
    return wgate*oracle*qstate  # 返回经过 W 门和 Oracle 门作用后的量子状态
def apply_grover(oracle, nqubits, iterations=None):
    """Applies Grover's algorithm to search for a marked state.

    Parameters
    ==========

    oracle : callable
        The unknown function that marks the desired state.

    nqubits : int
        Number of qubits involved in the algorithm.

    iterations : int or None, optional
        Number of iterations of Grover's algorithm to perform. If None,
        calculated as floor(sqrt(2**nqubits) * (pi/4)).

    Returns
    =======

    state : Expr
        The resulting state after applying Grover's algorithm.

    Examples
    ========

    Apply Grover's algorithm to find |2> from an even superposition of 2 qubits::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import apply_grover
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> qapply(apply_grover(f, 2))
        |2>

    """
    if nqubits <= 0:
        raise QuantumError(
            'Grover\'s algorithm needs nqubits > 0, received %r qubits'
            % nqubits
        )
    if iterations is None:
        iterations = floor(sqrt(2**nqubits)*(pi/4))
    
    # Create the OracleGate object for the given oracle function
    v = OracleGate(nqubits, oracle)
    # Start with an even superposition of states
    iterated = superposition_basis(nqubits)
    # Perform Grover's algorithm for the specified number of iterations
    for iter in range(iterations):
        # Apply the Grover iteration step
        iterated = grover_iteration(iterated, v)
        # Apply the quantum state transformation using qapply
        iterated = qapply(iterated)

    # Return the final state after all iterations
    return iterated
```