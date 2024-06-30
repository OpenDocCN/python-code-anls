# `D:\src\scipysrc\sympy\sympy\physics\quantum\density.py`

```
from itertools import product  # 导入 itertools 模块中的 product 函数

from sympy.core.add import Add  # 导入 sympy 中的 Add 类
from sympy.core.containers import Tuple  # 导入 sympy 中的 Tuple 类
from sympy.core.function import expand  # 导入 sympy 中的 expand 函数
from sympy.core.mul import Mul  # 导入 sympy 中的 Mul 类
from sympy.core.singleton import S  # 导入 sympy 中的 S 单例
from sympy.functions.elementary.exponential import log  # 导入 sympy 中的 log 函数
from sympy.matrices.dense import MutableDenseMatrix as Matrix  # 导入 sympy 中的 MutableDenseMatrix 类并重命名为 Matrix
from sympy.printing.pretty.stringpict import prettyForm  # 导入 sympy 中的 prettyForm 类
from sympy.physics.quantum.dagger import Dagger  # 导入 sympy 中的 Dagger 类
from sympy.physics.quantum.operator import HermitianOperator  # 导入 sympy 中的 HermitianOperator 类
from sympy.physics.quantum.represent import represent  # 导入 sympy 中的 represent 函数
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy  # 导入 sympy 中的多个函数
from sympy.physics.quantum.tensorproduct import TensorProduct, tensor_product_simp  # 导入 sympy 中的 TensorProduct 类和 tensor_product_simp 函数
from sympy.physics.quantum.trace import Tr  # 导入 sympy 中的 Tr 函数

class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """
    @classmethod
    def _eval_args(cls, args):
        # 调用此方法来对参数进行符号化处理
        args = super()._eval_args(args)

        for arg in args:
            # 检查参数是否为元组形式
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError("Each argument should be of form [state,prob]"
                                 " or ( state, prob )")

        return args

    def states(self):
        """Return list of all states.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()
        (|0>, |1>)

        """
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        """Return list of all probabilities.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()
        (0.5, 0.5)

        """
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        """Return specific state by index.

        Parameters
        ==========

        index : index of state to be returned

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()[1]
        |1>

        """
        state = self.args[index][0]
        return state
    def get_prob(self, index):
        """Return probability of specific state by index.

        Parameters
        ===========
        index : index of states whose probability is returned.

        Examples
        ========
        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()[1]
        0.500000000000000
        """
        # 获取指定索引处状态的概率值
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        """op will operate on each individual state.

        Parameters
        ==========
        op : Operator

        Examples
        ========
        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.apply_op(A)
        Density((A*|0>, 0.5),(A*|1>, 0.5))
        """
        # 对每个单独的状态应用操作符 op
        new_args = [(op*state, prob) for (state, prob) in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        """Expand the density operator into an outer product format.

        Examples
        ========
        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.doit()
        0.5*|0><0| + 0.5*|1><1|
        """
        # 将密度算符展开为外积形式
        terms = []
        for (state, prob) in self.args:
            state = state.expand()  # 需要展开以分解 (a+b)*c
            if (isinstance(state, Add)):
                for arg in product(state.args, repeat=2):
                    terms.append(prob*self._generate_outer_prod(arg[0],
                                                                arg[1]))
            else:
                terms.append(prob*self._generate_outer_prod(state, state))

        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        c_part1, nc_part1 = arg1.args_cnc()
        c_part2, nc_part2 = arg2.args_cnc()

        if (len(nc_part1) == 0 or len(nc_part2) == 0):
            raise ValueError('Atleast one-pair of Non-commutative instance required for outer product.')

        # Muls of Tensor Products should be expanded before this function is called
        if (isinstance(nc_part1[0], TensorProduct) and len(nc_part1) == 1
                and len(nc_part2) == 1):
            op = tensor_product_simp(nc_part1[0]*Dagger(nc_part2[0]))
        else:
            op = Mul(*nc_part1)*Dagger(Mul(*nc_part2))

        return Mul(*c_part1)*Mul(*c_part2) * op

    def _represent(self, **options):
        return represent(self.doit(), **options)
    # 返回 LaTeX 格式的算子名称 "\rho"
    def _print_operator_name_latex(self, printer, *args):
        return r'\rho'

    # 返回漂亮打印格式的算子名称 "ρ"（希腊字母 rho）
    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('\N{GREEK SMALL LETTER RHO}')

    # 计算跟踪（迹）操作的结果
    def _eval_trace(self, **kwargs):
        # 从关键字参数中获取指标列表，默认为空列表
        indices = kwargs.get('indices', [])
        # 执行跟踪操作并返回结果
        return Tr(self.doit(), indices).doit()

    # 计算密度矩阵的熵
    def entropy(self):
        """ Compute the entropy of a density matrix.

        Refer to density.entropy() method for examples.
        """
        # 调用外部函数 entropy 计算并返回熵
        return entropy(self)
# 计算矩阵或密度对象的熵

def entropy(density):
    """Compute the entropy of a matrix/density object.

    This computes -Tr(density*ln(density)) using the eigenvalue decomposition
    of density, which is given as either a Density instance or a matrix
    (numpy.ndarray, sympy.Matrix or scipy.sparse).

    Parameters
    ==========

    density : density matrix of type Density, SymPy matrix,
    scipy.sparse or numpy.ndarray

    Examples
    ========

    >>> from sympy.physics.quantum.density import Density, entropy
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy import S
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> d = Density((up,S(1)/2),(down,S(1)/2))
    >>> entropy(d)
    log(2)/2

    """

    # 如果输入是 Density 实例，则转换为对应的矩阵
    if isinstance(density, Density):
        density = represent(density)  # 在矩阵中表示

    # 如果输入是 scipy 稀疏矩阵，则转换为 numpy 数组
    if isinstance(density, scipy_sparse_matrix):
        density = to_numpy(density)

    # 如果输入是 SymPy 矩阵，则计算其特征值并计算熵
    if isinstance(density, Matrix):
        eigvals = density.eigenvals().keys()
        return expand(-sum(e*log(e) for e in eigvals))
    # 如果输入是 numpy 数组，则使用 numpy 函数计算特征值和熵
    elif isinstance(density, numpy_ndarray):
        import numpy as np
        eigvals = np.linalg.eigvals(density)
        return -np.sum(eigvals*np.log(eigvals))
    else:
        # 如果输入类型不符合预期，则抛出 ValueError 异常
        raise ValueError(
            "numpy.ndarray, scipy.sparse or SymPy matrix expected")


# 计算两个量子态之间的保真度

def fidelity(state1, state2):
    """ Computes the fidelity [1]_ between two quantum states

    The arguments provided to this function should be a square matrix or a
    Density object. If it is a square matrix, it is assumed to be diagonalizable.

    Parameters
    ==========

    state1, state2 : a density matrix or Matrix


    Examples
    ========

    >>> from sympy import S, sqrt
    >>> from sympy.physics.quantum.dagger import Dagger
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy.physics.quantum.density import fidelity
    >>> from sympy.physics.quantum.represent import represent
    >>>
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> amp = 1/sqrt(2)
    >>> updown = (amp*up) + (amp*down)
    >>>
    >>> # represent turns Kets into matrices
    >>> up_dm = represent(up*Dagger(up))
    >>> down_dm = represent(down*Dagger(down))
    >>> updown_dm = represent(updown*Dagger(updown))
    >>>
    >>> fidelity(up_dm, up_dm)
    1
    >>> fidelity(up_dm, down_dm) # orthogonal states
    0
    >>> fidelity(up_dm, updown_dm).evalf().round(3)
    0.707

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    """
    
    # 如果 state1 是 Density 实例，则转换为对应的矩阵
    state1 = represent(state1) if isinstance(state1, Density) else state1
    # 如果 state2 是 Density 实例，则转换为对应的矩阵
    state2 = represent(state2) if isinstance(state2, Density) else state2
    # 检查 state1 和 state2 是否为 Matrix 类型的对象，如果不是则抛出数值错误异常
    if not isinstance(state1, Matrix) or not isinstance(state2, Matrix):
        raise ValueError("state1 and state2 must be of type Density or Matrix "
                         "received type=%s for state1 and type=%s for state2" %
                         (type(state1), type(state2)))

    # 检查 state1 和 state2 的形状是否相同且 state1 是否为方阵，如果不满足条件则抛出数值错误异常
    if state1.shape != state2.shape and state1.is_square:
        raise ValueError("The dimensions of both args should be equal and the "
                         "matrix obtained should be a square matrix")

    # 计算 state1 的平方根
    sqrt_state1 = state1**S.Half
    # 计算表达式 Tr(sqrt_state1 * state2 * sqrt_state1)**(1/2) 的值并返回结果
    return Tr((sqrt_state1 * state2 * sqrt_state1)**S.Half).doit()
```