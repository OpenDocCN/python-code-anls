# `D:\src\scipysrc\sympy\sympy\physics\quantum\sho1d.py`

```
# 导入必要的符号和函数库
from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros

#------------------------------------------------------------------------------

class SHOOp(Operator):
    """Simple Harmonic Oscillator Operator base class.

    This class serves as a base for SHO operators, restricting them to one argument.

    """

    @classmethod
    def _eval_args(cls, args):
        # Evaluate arguments as quantum expressions
        args = QExpr._eval_args(args)
        # Ensure only one argument is provided
        if len(args) == 1:
            return args
        else:
            raise ValueError("Too many arguments")

    @classmethod
    def _eval_hilbert_space(cls, label):
        # Evaluate the Hilbert space associated with the operator
        return ComplexSpace(S.Infinity)

class RaisingOp(SHOOp):
    """Raising Operator (a^dagger) for the Simple Harmonic Oscillator.

    The Raising Operator raises the state by one. Its adjoint is the Lowering Operator 'a'.
    It can be represented in terms of position and momentum and is typically expressed as a matrix.

    Parameters
    ==========

    args : tuple
        List of numbers or parameters uniquely identifying the operator.

    Examples
    ========

    Create a Raising Operator, rewrite it in terms of position and momentum,
    and demonstrate that its adjoint is 'a':

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum import Dagger

        >>> ad = RaisingOp('a')
        >>> ad.rewrite('xp').doit()
        sqrt(2)*(m*omega*X - I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(ad)
        a

    Compute the commutator of a^dagger with other operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(ad, a).doit()
        -1
        >>> Commutator(ad, N).doit()
        -RaisingOp(a)

    Apply the Raising Operator to a quantum state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import RaisingOp, SHOKet

        >>> ad = RaisingOp('a')
        >>> k = SHOKet('k')
        >>> qapply(ad*k)
        sqrt(k + 1)*|k + 1>

    """

    def __init__(self, *args):
        super().__init__(*args)
    def _eval_rewrite_as_xp(self, *args, **kwargs):
        # 重写为位置和动量算符的表达式
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            S.NegativeOne*I*Px + m*omega*X)

    def _eval_adjoint(self):
        # 返回当前算符的共轭算符（降算符）
        return LoweringOp(*self.args)

    def _eval_commutator_LoweringOp(self, other):
        # 计算当前算符与降算符之间的对易子，结果为 -1
        return S.NegativeOne

    def _eval_commutator_NumberOp(self, other):
        # 计算当前算符与数值算符之间的对易子，结果为 -self
        return S.NegativeOne*self

    def _apply_operator_SHOKet(self, ket, **options):
        # 对给定的 SHO Ket 应用当前算符
        temp = ket.n + S.One
        return sqrt(temp)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        # 用默认基表示当前算符（使用数值算符表示）
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # 报错，未实现位置表示
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        # 用数值算符表示当前算符，默认为 4x4 的矩阵
        ndim_info = options.get('ndim', 4)
        format = options.get('format','sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i + 1, i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix

    #--------------------------------------------------------------------------
    # Printing Methods
    #--------------------------------------------------------------------------

    def _print_contents(self, printer, *args):
        # 打印当前算符的内容
        arg0 = printer._print(self.args[0], *args)
        return '%s(%s)' % (self.__class__.__name__, arg0)

    def _print_contents_pretty(self, printer, *args):
        # 使用漂亮的形式打印当前算符的内容，加上 '†' 符号
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        pform = pform**prettyForm('\N{DAGGER}')
        return pform

    def _print_contents_latex(self, printer, *args):
        # 使用 LaTeX 打印当前算符的内容，加上 '^{\dagger}' 符号
        arg = printer._print(self.args[0])
        return '%s^{\\dagger}' % arg
class LoweringOp(SHOOp):
    """The Lowering Operator or 'a'.

    When 'a' acts on a state it lowers the state up by one. Taking
    the adjoint of 'a' returns a^dagger, the Raising Operator. 'a'
    can be rewritten in terms of position and momentum. We can
    represent 'a' as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Lowering Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns a^dagger:

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum import Dagger

        >>> a = LoweringOp('a')
        >>> a.rewrite('xp').doit()
        sqrt(2)*(m*omega*X + I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(a)
        RaisingOp(a)

    Taking the commutator of 'a' with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import LoweringOp, RaisingOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> a = LoweringOp('a')
        >>> ad = RaisingOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(a, ad).doit()
        1
        >>> Commutator(a, N).doit()
        a

    Apply 'a' to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet('k')
        >>> qapply(a*k)
        sqrt(k)*|k - 1>

    Taking 'a' of the lowest state will return 0:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet(0)
        >>> qapply(a*k)
        0

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum.represent import represent
        >>> a = LoweringOp('a')
        >>> represent(a, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 1,       0,       0],
        [0, 0, sqrt(2),       0],
        [0, 0,       0, sqrt(3)],
        [0, 0,       0,       0]])

    """

    # 将操作符 'a' 重写为关于位置和动量的表达式
    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            I*Px + m*omega*X)

    # 返回 'a' 的伴随操作符 RaisingOp
    def _eval_adjoint(self):
        return RaisingOp(*self.args)

    # 计算 'a' 与 RaisingOp 的对易子结果为 1
    def _eval_commutator_RaisingOp(self, other):
        return S.One

    # 计算 'a' 与 NumberOp 的对易子结果为 'a'
    def _eval_commutator_NumberOp(self, other):
        return self

    # 应用 'a' 操作符到 SHOKet 态上，降低其量子数
    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n - Integer(1)
        if ket.n is S.Zero:
            return S.Zero
        else:
            return sqrt(ket.n)*SHOKet(temp)

    # 使用默认的基底表示 'a' 操作符
    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)
    # 定义一个方法来表示某种操作 `XOp`，接受 `basis` 和其他选项参数
    def _represent_XOp(self, basis, **options):
        # 这段逻辑目前是有效的，但是基于位置的表示逻辑存在问题。
        
        # 暂时注释掉以下两行，因为位置表示的逻辑有问题
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        
        # 抛出未实现错误，因为位置表示尚未实现
        raise NotImplementedError('Position representation is not implemented')

    # 定义一个方法来表示数字操作 `NumberOp`，接受 `basis` 和其他选项参数
    def _represent_NumberOp(self, basis, **options):
        # 获取选项中的 `ndim` 参数，默认为 4
        ndim_info = options.get('ndim', 4)
        # 获取选项中的 `format` 参数，默认为 'sympy'
        format = options.get('format', 'sympy')
        
        # 创建一个 ndim_info x ndim_info 的零矩阵，使用 `matrix_zeros` 函数
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        
        # 遍历矩阵的每一行（总共 ndim_info - 1 行）
        for i in range(ndim_info - 1):
            # 计算当前位置的值，通常是 sqrt(i + 1)
            value = sqrt(i + 1)
            # 如果格式为 'scipy.sparse'，则将值转换为浮点数
            if format == 'scipy.sparse':
                value = float(value)
            # 将值设置到矩阵中对角线的下一个元素位置
            matrix[i, i + 1] = value
        
        # 如果格式为 'scipy.sparse'，则将矩阵转换为 CSR 格式
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        
        # 返回生成的矩阵
        return matrix
class NumberOp(SHOOp):
    """The Number Operator is simply a^dagger*a

    It is often useful to write a^dagger*a as simply the Number Operator
    because the Number Operator commutes with the Hamiltonian. And can be
    expressed using the Number Operator. Also the Number Operator can be
    applied to states. We can represent the Number Operator as a matrix,
    which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Number Operator and rewrite it in terms of the ladder
    operators, position and momentum operators, and Hamiltonian:

        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> N = NumberOp('N')
        >>> N.rewrite('a').doit()
        RaisingOp(a)*a
        >>> N.rewrite('xp').doit()
        -1/2 + (m**2*omega**2*X**2 + Px**2)/(2*hbar*m*omega)
        >>> N.rewrite('H').doit()
        -1/2 + H/(hbar*omega)

    Take the Commutator of the Number Operator with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import NumberOp, Hamiltonian
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp

        >>> N = NumberOp('N')
        >>> H = Hamiltonian('H')
        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> Commutator(N,H).doit()
        0
        >>> Commutator(N,ad).doit()
        RaisingOp(a)
        >>> Commutator(N,a).doit()
        -a

    Apply the Number Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import NumberOp, SHOKet

        >>> N = NumberOp('N')
        >>> k = SHOKet('k')
        >>> qapply(N*k)
        k*|k>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import NumberOp
        >>> from sympy.physics.quantum.represent import represent
        >>> N = NumberOp('N')
        >>> represent(N, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])

    """

    def _eval_rewrite_as_a(self, *args, **kwargs):
        # 将操作符重写为阶梯操作符 a^dagger 和 a 的乘积
        return ad*a

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        # 将操作符重写为位置和动量算符的表达式
        return (S.One/(Integer(2)*m*hbar*omega))*(Px**2 + (
            m*omega*X)**2) - S.Half

    def _eval_rewrite_as_H(self, *args, **kwargs):
        # 将操作符重写为哈密顿算符的表达式
        return H/(hbar*omega) - S.Half

    def _apply_operator_SHOKet(self, ket, **options):
        # 应用操作符到态矢量
        return ket.n*ket

    def _eval_commutator_Hamiltonian(self, other):
        # 计算与哈密顿算符的对易子
        return S.Zero

    def _eval_commutator_RaisingOp(self, other):
        # 计算与阶梯操作符的对易子
        return other

    def _eval_commutator_LoweringOp(self, other):
        # 计算与降阶梯操作符的对易子
        return S.NegativeOne*other

    def _represent_default_basis(self, **options):
        # 使用默认基底表示操作符
        return self._represent_NumberOp(None, **options)
    # 这个方法用于生成某个操作符在指定基底下的表示形式
    def _represent_XOp(self, basis, **options):
        # 抛出未实现的错误，因为位置表示的逻辑尚未实现
        raise NotImplementedError('Position representation is not implemented')

    # 这个方法用于生成某个数值操作符在指定基底下的表示形式
    def _represent_NumberOp(self, basis, **options):
        # 获取维度信息，默认为4
        ndim_info = options.get('ndim', 4)
        # 获取格式信息，默认为'sympy'
        format = options.get('format', 'sympy')
        # 创建一个大小为ndim_info x ndim_info的零矩阵
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        # 填充矩阵对角线上的值
        for i in range(ndim_info):
            value = i
            # 如果格式为'scipy.sparse'，将值转换为浮点数
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i] = value
        # 如果格式为'scipy.sparse'，将矩阵转换为压缩稀疏行格式
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        # 返回生成的矩阵表示形式
        return matrix
# 定义 Hamiltonian 类，继承自 SHOOp 类，表示谐振子的哈密顿算符。
class Hamiltonian(SHOOp):
    """The Hamiltonian Operator.

    The Hamiltonian is used to solve the time-independent Schrodinger
    equation. The Hamiltonian can be expressed using the ladder operators,
    as well as by position and momentum. We can represent the Hamiltonian
    Operator as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Hamiltonian Operator and rewrite it in terms of the ladder
    operators, position and momentum, and the Number Operator:

        >>> from sympy.physics.quantum.sho1d import Hamiltonian

        >>> H = Hamiltonian('H')
        >>> H.rewrite('a').doit()
        hbar*omega*(1/2 + RaisingOp(a)*a)
        >>> H.rewrite('xp').doit()
        (m**2*omega**2*X**2 + Px**2)/(2*m)
        >>> H.rewrite('N').doit()
        hbar*omega*(1/2 + N)

    Take the Commutator of the Hamiltonian and the Number Operator:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, NumberOp

        >>> H = Hamiltonian('H')
        >>> N = NumberOp('N')
        >>> Commutator(H,N).doit()
        0

    Apply the Hamiltonian Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, SHOKet

        >>> H = Hamiltonian('H')
        >>> k = SHOKet('k')
        >>> qapply(H*k)
        hbar*k*omega*|k> + hbar*omega*|k>/2

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import Hamiltonian
        >>> from sympy.physics.quantum.represent import represent

        >>> H = Hamiltonian('H')
        >>> represent(H, basis=N, ndim=4, format='sympy')
        Matrix([
        [hbar*omega/2,              0,              0,              0],
        [           0, 3*hbar*omega/2,              0,              0],
        [           0,              0, 5*hbar*omega/2,              0],
        [           0,              0,              0, 7*hbar*omega/2]])

    """

    # 将 Hamiltonian 表达为阶梯算符 'a' 的重写函数
    def _eval_rewrite_as_a(self, *args, **kwargs):
        return hbar*omega*(ad*a + S.Half)

    # 将 Hamiltonian 表达为位置和动量 'xp' 的重写函数
    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)

    # 将 Hamiltonian 表达为数算符 'N' 的重写函数
    def _eval_rewrite_as_N(self, *args, **kwargs):
        return hbar*omega*(N + S.Half)

    # 应用 Hamiltonian 算符到 SHOKet 状态的函数
    def _apply_operator_SHOKet(self, ket, **options):
        return (hbar*omega*(ket.n + S.Half))*ket

    # Hamiltonian 与数算符的对易子的计算函数
    def _eval_commutator_NumberOp(self, other):
        return S.Zero

    # 默认基础下的 Hamiltonian 矩阵表示函数
    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)
    # 定义一个方法 `_represent_XOp`，用于处理某种操作符的表达式在特定基础上的表示
    def _represent_XOp(self, basis, **options):
        # 抛出未实现错误，因为位置表示逻辑尚未实现
        raise NotImplementedError('Position representation is not implemented')

    # 定义一个方法 `_represent_NumberOp`，用于处理某种数值操作符在特定基础上的表示
    def _represent_NumberOp(self, basis, **options):
        # 获取维度信息，默认为4维
        ndim_info = options.get('ndim', 4)
        # 获取输出格式，默认为'sympy'
        format = options.get('format', 'sympy')
        
        # 创建一个维度为 ndim_info x ndim_info 的零矩阵
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        
        # 遍历矩阵的每一行和每一列
        for i in range(ndim_info):
            # 计算当前位置的数值，加上 0.5
            value = i + S.Half
            
            # 如果输出格式为 'scipy.sparse'，将值转换为浮点数
            if format == 'scipy.sparse':
                value = float(value)
            
            # 将计算得到的值放入矩阵的对角线位置
            matrix[i,i] = value
        
        # 如果输出格式为 'scipy.sparse'，将矩阵转换为 CSR 格式（压缩稀疏行）
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        
        # 返回计算得到的矩阵乘以常数 hbar * omega
        return hbar * omega * matrix
#------------------------------------------------------------------------------

class SHOState(State):
    """State class for SHO states"""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # 返回具有无穷大空间的复数空间
        return ComplexSpace(S.Infinity)

    @property
    def n(self):
        # 返回该状态的第一个参数作为量子数
        return self.args[0]


class SHOKet(SHOState, Ket):
    """1D eigenket.

    Inherits from SHOState and Ket.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Ket's know about their associated bra:

        >>> from sympy.physics.quantum.sho1d import SHOKet

        >>> k = SHOKet('k')
        >>> k.dual
        <k|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOBra'>

    Take the Inner Product with a bra:

        >>> from sympy.physics.quantum import InnerProduct
        >>> from sympy.physics.quantum.sho1d import SHOKet, SHOBra

        >>> k = SHOKet('k')
        >>> b = SHOBra('b')
        >>> InnerProduct(b,k).doit()
        KroneckerDelta(b, k)

    Vector representation of a numerical state ket:

        >>> from sympy.physics.quantum.sho1d import SHOKet, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> k = SHOKet(3)
        >>> N = NumberOp('N')
        >>> represent(k, basis=N, ndim=4)
        Matrix([
        [0],
        [0],
        [0],
        [1]])

    """

    @classmethod
    def dual_class(self):
        # 返回这个类的双类，即 SHOBra
        return SHOBra

    def _eval_innerproduct_SHOBra(self, bra, **hints):
        # 返回 KroneckerDelta(bra.n, self.n) 的结果
        result = KroneckerDelta(self.n, bra.n)
        return result

    def _represent_default_basis(self, **options):
        # 使用默认基底来表示该态的向量表示
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        # 获取选项中的维度信息，默认为4
        ndim_info = options.get('ndim', 4)
        # 获取选项中的格式信息，默认为 sympy
        format = options.get('format', 'sympy')
        # 设置选项中的稀疏矩阵类型为 'lil'
        options['spmatrix'] = 'lil'
        # 创建一个大小为 ndim_info x 1 的零矩阵
        vector = matrix_zeros(ndim_info, 1, **options)
        # 如果量子数是整数类型
        if isinstance(self.n, Integer):
            # 如果量子数超过了给定的维度信息，则抛出异常
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            # 根据选项中的格式，设置相应位置的值为 1.0 或 S.One
            if format == 'scipy.sparse':
                vector[int(self.n), 0] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[int(self.n), 0] = 1.0
            else:
                vector[self.n, 0] = S.One
            return vector
        else:
            # 如果量子数不是数值类型，则抛出异常
            return ValueError("Not Numerical State")


class SHOBra(SHOState, Bra):
    """A time-independent Bra in SHO.

    Inherits from SHOState and Bra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========
    @classmethod
    def dual_class(self):
        # 返回关联的 SHOKet 类
        return SHOKet

    def _represent_default_basis(self, **options):
        # 使用默认基础的表示法来表示对象
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        # 获取选项中的 ndim 信息，默认为 4
        ndim_info = options.get('ndim', 4)
        # 获取选项中的 format 信息，默认为 'sympy'
        format = options.get('format', 'sympy')
        # 指定稀疏矩阵的类型为 'lil'
        options['spmatrix'] = 'lil'
        # 创建一个大小为 1 x ndim_info 的零矩阵，格式由选项决定
        vector = matrix_zeros(1, ndim_info, **options)
        # 如果 self.n 是整数类型
        if isinstance(self.n, Integer):
            # 如果 self.n 大于等于 ndim_info，则报错
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            # 根据 format 的不同类型，设定向量的值
            if format == 'scipy.sparse':
                vector[0, int(self.n)] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[0, int(self.n)] = 1.0
            else:
                vector[0, self.n] = S.One
            return vector
        else:
            # 如果 self.n 不是整数类型，则报错
            return ValueError("Not Numerical State")
# 创建一个 RaisingOp 对象 'ad'，操作符是 'a'
ad = RaisingOp('a')
# 创建一个 LoweringOp 对象 'a'
a = LoweringOp('a')
# 创建一个 Hamiltonian 对象 'H'
H = Hamiltonian('H')
# 创建一个 NumberOp 对象 'N'
N = NumberOp('N')
# 创建一个 Symbol 对象 'omega'
omega = Symbol('omega')
# 创建一个 Symbol 对象 'm'
m = Symbol('m')
```