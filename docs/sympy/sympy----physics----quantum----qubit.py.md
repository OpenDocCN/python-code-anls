# `D:\src\scipysrc\sympy\sympy\physics\quantum\qubit.py`

```
# Qubits for quantum computing.
# 
# Todo:
# * Finish implementing measurement logic. This should include POVM.
# * Update docstrings.
# * Update tests.

import math  # 导入数学库

from sympy.core.add import Add  # 导入 SymPy 的加法运算
from sympy.core.mul import Mul  # 导入 SymPy 的乘法运算
from sympy.core.numbers import Integer  # 导入 SymPy 的整数类型
from sympy.core.power import Pow  # 导入 SymPy 的指数运算
from sympy.core.singleton import S  # 导入 SymPy 的单例模块
from sympy.functions.elementary.complexes import conjugate  # 导入 SymPy 的复数共轭函数
from sympy.functions.elementary.exponential import log  # 导入 SymPy 的对数函数
from sympy.core.basic import _sympify  # 导入 SymPy 的类型转换函数
from sympy.external.gmpy import SYMPY_INTS  # 导入 SymPy 对外部 GMPY 的整数支持
from sympy.matrices import Matrix, zeros  # 导入 SymPy 的矩阵和零矩阵生成函数
from sympy.printing.pretty.stringpict import prettyForm  # 导入 SymPy 的漂亮字符串形式

from sympy.physics.quantum.hilbert import ComplexSpace  # 导入 SymPy 的复数空间
from sympy.physics.quantum.state import Ket, Bra, State  # 导入 SymPy 的量子态相关模块

from sympy.physics.quantum.qexpr import QuantumError  # 导入 SymPy 的量子表达式异常
from sympy.physics.quantum.represent import represent  # 导入 SymPy 的表示模块
from sympy.physics.quantum.matrixutils import (  # 导入 SymPy 的矩阵工具函数
    numpy_ndarray, scipy_sparse_matrix
)
from mpmath.libmp.libintmath import bitcount  # 导入 mpmath 的整数位计数函数

__all__ = [  # 导出模块的公开接口列表
    'Qubit',
    'QubitBra',
    'IntQubit',
    'IntQubitBra',
    'qubit_to_matrix',
    'matrix_to_qubit',
    'matrix_to_density',
    'measure_all',
    'measure_partial',
    'measure_partial_oneshot',
    'measure_all_oneshot'
]

#-----------------------------------------------------------------------------
# Qubit Classes
#-----------------------------------------------------------------------------


class QubitState(State):
    """Base class for Qubit and QubitBra."""
    
    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # If we are passed a QubitState or subclass, we just take its qubit
        # values directly.
        if len(args) == 1 and isinstance(args[0], QubitState):
            return args[0].qubit_values

        # Turn strings into tuple of strings
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple( S.Zero if qb == "0" else S.One for qb in args[0])
        else:
            args = tuple( S.Zero if qb == "0" else S.One if qb == "1" else qb for qb in args)
        args = tuple(_sympify(arg) for arg in args)

        # Validate input (must have 0 or 1 input)
        for element in args:
            if element not in (S.Zero, S.One):
                raise ValueError(
                    "Qubit values must be 0 or 1, got: %r" % element)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        return ComplexSpace(2)**len(args)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def dimension(self):
        """The number of Qubits in the state."""
        return len(self.qubit_values)

    @property
    def nqubits(self):
        """Alias for dimension, returns the number of qubits."""
        return self.dimension
    # 返回该量子比特对象的标签值作为元组
    def qubit_values(self):
        """Returns the values of the qubits as a tuple."""
        return self.label

    #-------------------------------------------------------------------------
    # Special methods
    #-------------------------------------------------------------------------

    # 返回该量子比特对象的维度作为长度
    def __len__(self):
        return self.dimension

    # 根据索引获取量子比特对象中特定比特的值
    def __getitem__(self, bit):
        return self.qubit_values[int(self.dimension - bit - 1)]

    #-------------------------------------------------------------------------
    # Utility methods
    #-------------------------------------------------------------------------

    # 翻转指定位置的比特值
    def flip(self, *bits):
        """Flip the bit(s) given."""
        # 复制当前比特值列表
        newargs = list(self.qubit_values)
        # 遍历传入的比特位置参数
        for i in bits:
            # 计算实际比特索引（由高到低位）
            bit = int(self.dimension - i - 1)
            # 执行翻转操作
            if newargs[bit] == 1:
                newargs[bit] = 0
            else:
                newargs[bit] = 1
        # 返回一个新的量子比特对象，其比特值已被翻转
        return self.__class__(*tuple(newargs))
class Qubit(QubitState, Ket):
    """A multi-qubit ket in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    Examples
    ========

    Create a qubit in a couple of different ways and look at their attributes:

        >>> from sympy.physics.quantum.qubit import Qubit
        >>> Qubit(0,0,0)
        |000>
        >>> q = Qubit('0101')
        >>> q
        |0101>

        >>> q.nqubits
        4
        >>> len(q)
        4
        >>> q.dimension
        4
        >>> q.qubit_values
        (0, 1, 0, 1)

    We can flip the value of an individual qubit:

        >>> q.flip(1)
        |0111>

    We can take the dagger of a Qubit to get a bra:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> Dagger(q)
        <0101|
        >>> type(Dagger(q))
        <class 'sympy.physics.quantum.qubit.QubitBra'>

    Inner products work as expected:

        >>> ip = Dagger(q)*q
        >>> ip
        <0101|0101>
        >>> ip.doit()
        1
    """

    @classmethod
    def dual_class(self):
        """Returns the dual class associated with Qubit, which is QubitBra."""
        return QubitBra

    def _eval_innerproduct_QubitBra(self, bra, **hints):
        """Evaluate the inner product between self and a QubitBra instance.

        Parameters
        ----------
        bra : QubitBra
            The bra to compute the inner product with.
        **hints :
            Additional hints or options for evaluation.

        Returns
        -------
        sympy.core.numbers.One or sympy.core.numbers.Zero
            Returns S.One if labels match, otherwise S.Zero.
        """
        if self.label == bra.label:
            return S.One
        else:
            return S.Zero

    def _represent_default_basis(self, **options):
        """Represent this qubits in the default basis.

        This method calls _represent_ZGate with default options.

        Parameters
        ----------
        **options :
            Options for representation.

        Returns
        -------
        Matrix or numpy.array or scipy.sparse.csr_matrix
            Representation of qubits in the default basis.
        """
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """Represent this qubits in the computational basis (ZGate).

        Parameters
        ----------
        basis :
            Basis information (not used in this implementation).
        **options :
            Options for representation format ('sympy', 'numpy', 'scipy.sparse').

        Returns
        -------
        Matrix or numpy.array or scipy.sparse.csr_matrix
            Representation of qubits in the computational basis.
        """
        _format = options.get('format', 'sympy')
        n = 1
        definite_state = 0
        for it in reversed(self.qubit_values):
            definite_state += n*it
            n = n*2
        result = [0]*(2**self.dimension)
        result[int(definite_state)] = 1
        if _format == 'sympy':
            return Matrix(result)
        elif _format == 'numpy':
            import numpy as np
            return np.array(result, dtype='complex').transpose()
        elif _format == 'scipy.sparse':
            from scipy import sparse
            return sparse.csr_matrix(result, dtype='complex').transpose()
    def _eval_trace(self, bra, **kwargs):
        # 获取indices参数，如果不存在则为空列表
        indices = kwargs.get('indices', [])

        # 将索引列表按升序排序，以便从最高位量子比特开始跟踪
        sorted_idx = list(indices)
        if len(sorted_idx) == 0:
            sorted_idx = list(range(0, self.nqubits))
        sorted_idx.sort()

        # 对每个索引进行追踪
        new_mat = self * bra
        for i in range(len(sorted_idx) - 1, -1, -1):
            # 从最左侧的量子比特开始追踪
            new_mat = self._reduced_density(new_mat, int(sorted_idx[i]))

        if len(sorted_idx) == self.nqubits:
            # 如果请求了完全追踪，则返回密度矩阵的第一个元素
            return new_mat[0]
        else:
            # 否则将新矩阵转换为密度矩阵并返回
            return matrix_to_density(new_mat)

    def _reduced_density(self, matrix, qubit, **options):
        """计算通过追踪一个量子比特得到的约化密度矩阵。
           qubit参数应为Python整数类型，因为它用于位操作。
        """
        def find_index_that_is_projected(j, k, qubit):
            # 创建一个位掩码，用于定位和清除特定的量子比特
            bit_mask = 2**qubit - 1
            return ((j >> qubit) << (1 + qubit)) + (j & bit_mask) + (k << qubit)

        # 将输入矩阵表示为特定的矩阵表示形式
        old_matrix = represent(matrix, **options)
        old_size = old_matrix.cols
        # 我们期望old_size是偶数
        new_size = old_size // 2
        new_matrix = Matrix().zeros(new_size)

        # 对新矩阵进行迭代，计算约化密度矩阵的元素
        for i in range(new_size):
            for j in range(new_size):
                for k in range(2):
                    col = find_index_that_is_projected(j, k, qubit)
                    row = find_index_that_is_projected(i, k, qubit)
                    new_matrix[i, j] += old_matrix[row, col]

        return new_matrix
class QubitBra(QubitState, Bra):
    """A multi-qubit bra in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    See also
    ========

    Qubit: Examples using qubits

    """

    @classmethod
    def dual_class(self):
        # 返回 Qubit 类作为双类
        return Qubit


class IntQubitState(QubitState):
    """A base class for qubits that work with binary representations."""

    @classmethod
    def _eval_args(cls, args, nqubits=None):
        # 对传入参数进行评估和处理

        # 如果 args 只有一个元素且是 QubitState 的实例
        if len(args) == 1 and isinstance(args[0], QubitState):
            return QubitState._eval_args(args)
        # 否则，args 应该是整数类型或者 Integer 类型的元组
        elif not all(isinstance(a, (int, Integer)) for a in args):
            raise ValueError('values must be integers, got (%s)' % (tuple(type(a) for a in args),))
        
        # 如果指定了 nqubits
        if nqubits is not None:
            # 确保 nqubits 是整数类型
            if not isinstance(nqubits, (int, Integer)):
                raise ValueError('nqubits must be an integer, got (%s)' % type(nqubits))
            # 如果 args 的长度不是 1，抛出异常
            if len(args) != 1:
                raise ValueError(
                    'too many positional arguments (%s). should be (number, nqubits=n)' % (args,))
            # 使用指定的 nqubits 处理参数
            return cls._eval_args_with_nqubits(args[0], nqubits)
        
        # 对于单个参数，构建其最小位数的二进制表示
        if len(args) == 1 and args[0] > 1:
            # 计算表示该整数所需的最小位数
            rvalues = reversed(range(bitcount(abs(args[0]))))
            qubit_values = [(args[0] >> i) & 1 for i in rvalues]
            return QubitState._eval_args(qubit_values)
        
        # 对于两个参数，第二个参数表示位数，例如 IntQubit(0,5) == |00000>
        elif len(args) == 2 and args[1] > 1:
            return cls._eval_args_with_nqubits(args[0], args[1])
        
        # 其他情况，继续使用 QubitState 的处理方法
        else:
            return QubitState._eval_args(args)

    @classmethod
    def _eval_args_with_nqubits(cls, number, nqubits):
        # 根据指定的 nqubits 构建 number 的二进制表示

        # 计算 number 表示所需的位数
        need = bitcount(abs(number))
        # 如果 nqubits 小于所需的位数，抛出异常
        if nqubits < need:
            raise ValueError(
                'cannot represent %s with %s bits' % (number, nqubits))
        # 构建二进制表示的列表
        qubit_values = [(number >> i) & 1 for i in reversed(range(nqubits))]
        return QubitState._eval_args(qubit_values)

    def as_int(self):
        """Return the numerical value of the qubit."""
        number = 0
        n = 1
        # 根据 qubit_values 计算 qubit 的数值表示
        for i in reversed(self.qubit_values):
            number += n*i
            n = n << 1
        return number

    def _print_label(self, printer, *args):
        # 返回 qubit 的数值表示的字符串形式
        return str(self.as_int())
    # 定义一个方法 _print_label_pretty，接受一个打印机对象和任意数量的参数
    def _print_label_pretty(self, printer, *args):
        # 调用 self._print_label 方法生成标签
        label = self._print_label(printer, *args)
        # 使用 prettyForm 类将标签转换为漂亮格式，并返回
        return prettyForm(label)

    # 创建 _print_label_repr 方法，其与 _print_label 方法相同
    _print_label_repr = _print_label

    # 创建 _print_label_latex 方法，其与 _print_label 方法相同
    _print_label_latex = _print_label
class IntQubit(IntQubitState, Qubit):
    """A qubit ket that store integers as binary numbers in qubit values.

    The differences between this class and ``Qubit`` are:

    * The form of the constructor.
    * The qubit values are printed as their corresponding integer, rather
      than the raw qubit values. The internal storage format of the qubit
      values in the same as ``Qubit``.

    Parameters
    ==========

    values : int, tuple
        If a single argument, the integer we want to represent in the qubit
        values. This integer will be represented using the fewest possible
        number of qubits.
        If a pair of integers and the second value is more than one, the first
        integer gives the integer to represent in binary form and the second
        integer gives the number of qubits to use.
        List of zeros and ones is also accepted to generate qubit by bit pattern.

    nqubits : int
        The integer that represents the number of qubits.
        This number should be passed with keyword ``nqubits=N``.
        You can use this in order to avoid ambiguity of Qubit-style tuple of bits.
        Please see the example below for more details.

    Examples
    ========

    Create a qubit for the integer 5:

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qubit import Qubit
        >>> q = IntQubit(5)
        >>> q
        |5>

    We can also create an ``IntQubit`` by passing a ``Qubit`` instance.

        >>> q = IntQubit(Qubit('101'))
        >>> q
        |5>
        >>> q.as_int()
        5
        >>> q.nqubits
        3
        >>> q.qubit_values
        (1, 0, 1)

    We can go back to the regular qubit form.

        >>> Qubit(q)
        |101>

    Please note that ``IntQubit`` also accepts a ``Qubit``-style list of bits.
    So, the code below yields qubits 3, not a single bit ``1``.

        >>> IntQubit(1, 1)
        |3>

    To avoid ambiguity, use ``nqubits`` parameter.
    Use of this keyword is recommended especially when you provide the values by variables.

        >>> IntQubit(1, nqubits=1)
        |1>
        >>> a = 1
        >>> IntQubit(a, nqubits=1)
        |1>
    """
    @classmethod
    def dual_class(self):
        # 返回与当前类(IntQubit)对偶的类(IntQubitBra)
        return IntQubitBra

    def _eval_innerproduct_IntQubitBra(self, bra, **hints):
        # 调用 Qubit 类的方法计算与给定 bra 的内积
        return Qubit._eval_innerproduct_QubitBra(self, bra)

class IntQubitBra(IntQubitState, QubitBra):
    """A qubit bra that store integers as binary numbers in qubit values."""

    @classmethod
    def dual_class(self):
        # 返回与当前类(IntQubitBra)对偶的类(IntQubit)
        return IntQubit


#-----------------------------------------------------------------------------
# Qubit <---> Matrix conversion functions
#-----------------------------------------------------------------------------


def matrix_to_qubit(matrix):
    """Convert from the matrix repr. to a sum of Qubit objects.

    Parameters
    ----------
    # matrix: Matrix, numpy.matrix, scipy.sparse
    #     要构建量子比特表示的矩阵。可以是 SymPy 矩阵、numpy 矩阵或者 scipy.sparse 稀疏矩阵。

    # 确定输入矩阵的类型，以确定格式
    format = 'sympy'
    if isinstance(matrix, numpy_ndarray):
        format = 'numpy'
    if isinstance(matrix, scipy_sparse_matrix):
        format = 'scipy.sparse'

    # 确保矩阵具有正确的维度，适合量子比特矩阵表示
    if matrix.shape[0] == 1:
        mlistlen = matrix.shape[1]
        nqubits = log(mlistlen, 2)
        ket = False
        cls = QubitBra
    elif matrix.shape[1] == 1:
        mlistlen = matrix.shape[0]
        nqubits = log(mlistlen, 2)
        ket = True
        cls = Qubit
    else:
        # 抛出量子错误，矩阵必须是行向量或列向量
        raise QuantumError(
            'Matrix must be a row/column vector, got %r' % matrix
        )

    # 如果 nqubits 不是整数，则抛出量子错误
    if not isinstance(nqubits, Integer):
        raise QuantumError('Matrix must be a row/column vector of size '
                           '2**nqubits, got: %r' % matrix)

    # 遍历矩阵中的每个元素，如果非零则构建量子比特项
    result = 0
    for i in range(mlistlen):
        if ket:
            element = matrix[i, 0]
        else:
            element = matrix[0, i]

        # 如果格式是 numpy 或 scipy.sparse，则将元素转换为复数类型
        if format in ('numpy', 'scipy.sparse'):
            element = complex(element)

        # 如果元素不为零，则构建量子比特数组
        if element != 0.0:
            # 构建量子比特数组；在 i 为 1 的比特位置上为 1，其它位置为 0
            qubit_array = [int(i & (1 << x) != 0) for x in range(nqubits)]
            qubit_array.reverse()
            result = result + element * cls(*qubit_array)

    # 如果 result 是 SymPy 的 Mul、Add 或 Pow 类型，则扩展化简结果
    if isinstance(result, (Mul, Add, Pow)):
        result = result.expand()

    # 返回构建好的量子比特表示
    return result
def matrix_to_density(mat):
    """
    Works by finding the eigenvectors and eigenvalues of the matrix.
    We know we can decompose rho by doing:
    sum(EigenVal*|Eigenvect><Eigenvect|)
    """
    # 导入所需的密度矩阵类
    from sympy.physics.quantum.density import Density
    # 计算矩阵的特征向量和特征值
    eigen = mat.eigenvects()
    # 生成量子态和其对应概率的列表
    args = [[matrix_to_qubit(Matrix(
        [vector, ])), x[0]] for x in eigen for vector in x[2] if x[0] != 0]
    if (len(args) == 0):
        # 如果没有量子态，则返回零密度矩阵
        return S.Zero
    else:
        # 根据生成的量子态和概率列表构建密度矩阵
        return Density(*args)


def qubit_to_matrix(qubit, format='sympy'):
    """Converts an Add/Mul of Qubit objects into it's matrix representation

    This function is the inverse of ``matrix_to_qubit`` and is a shorthand
    for ``represent(qubit)``.
    """
    # 调用 represent 函数将量子态转换为矩阵表示
    return represent(qubit, format=format)


#-----------------------------------------------------------------------------
# Measurement
#-----------------------------------------------------------------------------


def measure_all(qubit, format='sympy', normalize=True):
    """Perform an ensemble measurement of all qubits.

    Parameters
    ==========

    qubit : Qubit, Add
        The qubit to measure. This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_all
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_all(q)
        [(|00>, 1/4), (|01>, 1/4), (|10>, 1/4), (|11>, 1/4)]
    """
    # 将量子态转换为对应的矩阵表示
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        results = []

        if normalize:
            # 如果需要归一化，则对矩阵进行归一化处理
            m = m.normalized()

        size = max(m.shape)  # Max of shape to account for bra or ket
        nqubits = int(math.log(size)/math.log(2))
        for i in range(size):
            if m[i] != 0.0:
                # 将非零元素对应的量子态和概率添加到结果列表中
                results.append(
                    (Qubit(IntQubit(i, nqubits=nqubits)), m[i]*conjugate(m[i]))
                )
        return results
    else:
        # 抛出未实现错误，因为目前仅支持 'sympy' 格式的矩阵
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial(qubit, bits, format='sympy', normalize=True):
    """Perform a partial ensemble measure on the specified qubits.

    Parameters
    ==========

    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    # 将量子比特表示转换为指定格式的矩阵表示
    m = qubit_to_matrix(qubit, format)

    # 如果 bits 是整数，转换为元组形式
    if isinstance(bits, (SYMPY_INTS, Integer)):
        bits = (int(bits),)

    # 如果选择使用 'sympy' 格式
    if format == 'sympy':
        # 如果需要归一化，对矩阵 m 进行归一化处理
        if normalize:
            m = m.normalized()

        # 获取可能的测量结果列表
        possible_outcomes = _get_possible_outcomes(m, bits)

        # 初始化输出结果列表
        output = []
        # 遍历每个可能的测量结果
        for outcome in possible_outcomes:
            # 计算测量结果的概率
            prob_of_outcome = 0
            prob_of_outcome += (outcome.H * outcome)[0]

            # 如果概率不为零，构建下一个量子比特表示并添加到输出列表中
            if prob_of_outcome != 0:
                if normalize:
                    next_matrix = matrix_to_qubit(outcome.normalized())
                else:
                    next_matrix = matrix_to_qubit(outcome)

                output.append((
                    next_matrix,
                    prob_of_outcome
                ))

        # 返回输出结果列表
        return output
    else:
        # 如果选择的格式不是 'sympy'，抛出未实现错误
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )
# 执行部分单次测量，测量指定量子比特的部分单次测量结果
def measure_partial_oneshot(qubit, bits, format='sympy'):
    """Perform a partial oneshot measurement on the specified qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    # 将量子比特转换为指定格式的矩阵表示
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        # 对于 sympy 格式的矩阵，进行归一化处理
        m = m.normalized()
        # 获取可能的测量结果列表
        possible_outcomes = _get_possible_outcomes(m, bits)

        # 从可能的结果中随机选择一个状态
        random_number = random.random()
        total_prob = 0
        for outcome in possible_outcomes:
            # 计算找到指定比特位的概率
            total_prob += (outcome.H*outcome)[0]
            if total_prob >= random_number:
                # 将矩阵形式的结果转换为量子比特表示，并返回
                return matrix_to_qubit(outcome.normalized())
    else:
        # 对于非 sympy 格式的矩阵，抛出未实现的错误
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


# 获取可以在测量中产生的可能状态列表
def _get_possible_outcomes(m, bits):
    """Get the possible states that can be produced in a measurement.

    Parameters
    ----------
    m : Matrix
        The matrix representing the state of the system.
    bits : tuple, list
        Which bits will be measured.

    Returns
    -------
    result : list
        The list of possible states which can occur given this measurement.
        These are un-normalized so we can derive the probability of finding
        this state by taking the inner product with itself
    """

    # This is filled with loads of dirty binary tricks...You have been warned

    # 获取矩阵的大小，考虑到 bra 或 ket 的情况取最大值
    size = max(m.shape)
    # 可能的量子比特数
    nqubits = int(math.log2(size) + .1)

    # 创建输出状态列表，每个状态表示测量的可能结果
    output_matrices = []
    for i in range(1 << len(bits)):
        output_matrices.append(zeros(2**nqubits, 1))

    # 位掩码将帮助确定可能的测量结果
    # 当位掩码与矩阵索引进行与运算时，
    # 生成一个空列表，用于存储位掩码（bit masks）
    bit_masks = []
    # 遍历给定的位列表（bits），生成对应的位掩码并添加到bit_masks中
    for bit in bits:
        bit_masks.append(1 << bit)

    # 生成可能的输出状态矩阵
    for i in range(2**nqubits):
        trueness = 0  # 用于确定这个值属于哪个输出矩阵
        # 查找 trueness 的值
        for j in range(len(bit_masks)):
            # 检查是否当前位掩码在 i 中存在，如果存在则增加对应的值到 trueness
            if i & bit_masks[j]:
                trueness += j + 1
        # 将值放入正确的输出矩阵中
        output_matrices[trueness][i] = m[i]

    # 返回填充好的输出矩阵列表
    return output_matrices
def measure_all_oneshot(qubit, format='sympy'):
    """Perform a oneshot ensemble measurement on all qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    # 导入随机模块
    import random
    # 将 qubit 转换为矩阵
    m = qubit_to_matrix(qubit)

    # 如果指定使用 'sympy' 格式
    if format == 'sympy':
        # 对矩阵进行归一化处理
        m = m.normalized()
        # 生成一个随机数
        random_number = random.random()
        total = 0
        result = 0
        # 遍历矩阵中的元素
        for i in m:
            total += i*i.conjugate()
            # 如果累积概率超过随机数，则确定返回的结果
            if total > random_number:
                break
            result += 1
        # 构造并返回量子比特对象，表示系统在测量后坍缩到的状态
        return Qubit(IntQubit(result, int(math.log2(max(m.shape)) + .1)))
    else:
        # 如果指定格式不是 'sympy'，则抛出未实现错误
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )
```