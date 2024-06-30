# `D:\src\scipysrc\sympy\sympy\physics\quantum\state.py`

```
# 引入 sympy 库中必要的模块和类
"""Dirac notation for states."""

from sympy.core.cache import cacheit  # 导入 cacheit 函数
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.core.expr import Expr  # 导入 Expr 类
from sympy.core.function import Function  # 导入 Function 类
from sympy.core.numbers import oo, equal_valued  # 导入 oo 和 equal_valued 常量
from sympy.core.singleton import S  # 导入 S 单例
from sympy.functions.elementary.complexes import conjugate  # 导入 conjugate 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.integrals.integrals import integrate  # 导入 integrate 函数
from sympy.printing.pretty.stringpict import stringPict  # 导入 stringPict 类
from sympy.physics.quantum.qexpr import QExpr, dispatch_method  # 导入 QExpr 和 dispatch_method 函数

__all__ = [  # 模块的公开接口列表
    'KetBase',
    'BraBase',
    'StateBase',
    'State',
    'Ket',
    'Bra',
    'TimeDepState',
    'TimeDepBra',
    'TimeDepKet',
    'OrthogonalKet',
    'OrthogonalBra',
    'OrthogonalState',
    'Wavefunction'
]

#-----------------------------------------------------------------------------
# States, bras and kets.
#-----------------------------------------------------------------------------

# ASCII brackets
_lbracket = "<"  # 左尖括号的 ASCII 表示
_rbracket = ">"  # 右尖括号的 ASCII 表示
_straight_bracket = "|"  # 竖直线的 ASCII 表示


# Unicode brackets
# MATHEMATICAL ANGLE BRACKETS
_lbracket_ucode = "\N{MATHEMATICAL LEFT ANGLE BRACKET}"  # 左数学角括号的 Unicode 表示
_rbracket_ucode = "\N{MATHEMATICAL RIGHT ANGLE BRACKET}"  # 右数学角括号的 Unicode 表示
# LIGHT VERTICAL BAR
_straight_bracket_ucode = "\N{LIGHT VERTICAL BAR}"  # 竖直条的 Unicode 表示

# Other options for unicode printing of <, > and | for Dirac notation.

# LEFT-POINTING ANGLE BRACKET
# _lbracket = "\u2329"
# _rbracket = "\u232A"

# LEFT ANGLE BRACKET
# _lbracket = "\u3008"
# _rbracket = "\u3009"

# VERTICAL LINE
# _straight_bracket = "\u007C"


class StateBase(QExpr):
    """Abstract base class for general abstract states in quantum mechanics.

    All other state classes defined will need to inherit from this class. It
    carries the basic structure for all other states such as dual, _eval_adjoint
    and label.

    This is an abstract base class and you should not instantiate it directly,
    instead use State.
    """

    @classmethod
    def _operators_to_state(self, ops, **options):
        """ Returns the eigenstate instance for the passed operators.

        This method should be overridden in subclasses. It will handle being
        passed either an Operator instance or set of Operator instances. It
        should return the corresponding state INSTANCE or simply raise a
        NotImplementedError. See cartesian.py for an example.
        """

        raise NotImplementedError("Cannot map operators to states in this class. Method not implemented!")
    #-------------------------------------------------------------------------
    # Dagger/dual
    #-------------------------------------------------------------------------

    @property
    def dual(self):
        """Return the dual state of this one."""
        # 返回这个状态的对偶状态
        return self.dual_class()._new_rawargs(self.hilbert_space, *self.args)

    @classmethod
    def dual_class(self):
        """Return the class used to construct the dual."""
        # 返回用于构造对偶的类
        raise NotImplementedError(
            'dual_class must be implemented in a subclass'
        )

    def _eval_adjoint(self):
        """Compute the dagger of this state using the dual."""
        # 使用对偶计算这个状态的伴随（共轭转置）
        return self.dual

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------
    def _pretty_brackets(self, height, use_unicode=True):
        # 返回格式化美化的括号用于状态
        # 理想情况下，可以通过 pform.parens 实现，但它不支持尖角 < 和 >

        # 设置用于 Unicode 和 ASCII 的括号字符
        if use_unicode:
            lbracket, rbracket = getattr(self, 'lbracket_ucode', ""), getattr(self, 'rbracket_ucode', "")
            slash, bslash, vert = '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT}', \
                                  '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT}', \
                                  '\N{BOX DRAWINGS LIGHT VERTICAL}'
        else:
            lbracket, rbracket = getattr(self, 'lbracket', ""), getattr(self, 'rbracket', "")
            slash, bslash, vert = '/', '\\', '|'

        # 如果高度为1，直接返回括号字符串
        if height == 1:
            return stringPict(lbracket), stringPict(rbracket)
        # 将高度设为偶数
        height += (height % 2)

        brackets = []
        for bracket in lbracket, rbracket:
            # 创建左括号
            if bracket in {_lbracket, _lbracket_ucode}:
                bracket_args = [ ' ' * (height//2 - i - 1) +
                                 slash for i in range(height // 2)]
                bracket_args.extend(
                    [' ' * i + bslash for i in range(height // 2)])
            # 创建右括号
            elif bracket in {_rbracket, _rbracket_ucode}:
                bracket_args = [ ' ' * i + bslash for i in range(height // 2)]
                bracket_args.extend([ ' ' * (
                    height//2 - i - 1) + slash for i in range(height // 2)])
            # 创建竖直直线形式的括号
            elif bracket in {_straight_bracket, _straight_bracket_ucode}:
                bracket_args = [vert] * height
            else:
                raise ValueError(bracket)
            brackets.append(
                stringPict('\n'.join(bracket_args), baseline=height//2))
        return brackets

    def _sympystr(self, printer, *args):
        # 获取打印内容
        contents = self._print_contents(printer, *args)
        # 返回内容及其左右括号的组合字符串
        return '%s%s%s' % (getattr(self, 'lbracket', ""), contents, getattr(self, 'rbracket', ""))

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        # 获取打印内容的 prettyForm
        pform = self._print_contents_pretty(printer, *args)
        # 获取左右括号
        lbracket, rbracket = self._pretty_brackets(
            pform.height(), printer._use_unicode)
        # 组合状态的 prettyForm
        pform = prettyForm(*pform.left(lbracket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        # 获取 LaTeX 格式的打印内容
        contents = self._print_contents_latex(printer, *args)
        # 返回用于 matplotlib 渲染的正确 LaTeX 格式字符串
        return '{%s%s%s}' % (getattr(self, 'lbracket_latex', ""), contents, getattr(self, 'rbracket_latex', ""))
class KetBase(StateBase):
    """Base class for Kets.

    This class defines the dual property and the brackets for printing. This is
    an abstract base class and you should not instantiate it directly, instead
    use Ket.
    """

    # Define left and right brackets for kets in different formats
    lbracket = _straight_bracket
    rbracket = _rbracket
    lbracket_ucode = _straight_bracket_ucode
    rbracket_ucode = _rbracket_ucode
    lbracket_latex = r'\left|'
    rbracket_latex = r'\right\rangle '

    @classmethod
    def default_args(self):
        """Return default arguments for instantiation."""
        return ("psi",)

    @classmethod
    def dual_class(self):
        """Return the dual class associated with this ket."""
        return BraBase

    def __mul__(self, other):
        """Define multiplication operation for KetBase * other."""
        from sympy.physics.quantum.operator import OuterProduct
        if isinstance(other, BraBase):
            return OuterProduct(self, other)
        else:
            return Expr.__mul__(self, other)

    def __rmul__(self, other):
        """Define right multiplication operation for other * KetBase."""
        from sympy.physics.quantum.innerproduct import InnerProduct
        if isinstance(other, BraBase):
            return InnerProduct(other, self)
        else:
            return Expr.__rmul__(self, other)

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_innerproduct(self, bra, **hints):
        """Evaluate the inner product between this ket and a bra.

        This is called to compute <bra|ket>, where the ket is ``self``.

        This method will dispatch to sub-methods having the format::

            ``def _eval_innerproduct_BraClass(self, **hints):``

        Subclasses should define these methods (one for each BraClass) to
        teach the ket how to take inner products with bras.
        """
        return dispatch_method(self, '_eval_innerproduct', bra, **hints)

    def _apply_from_right_to(self, op, **options):
        """Apply an Operator to this Ket as Operator*Ket

        This method will dispatch to methods having the format::

            ``def _apply_from_right_to_OperatorName(op, **options):``

        Subclasses should define these methods (one for each OperatorName) to
        teach the Ket how to implement OperatorName*Ket

        Parameters
        ==========

        op : Operator
            The Operator that is acting on the Ket as op*Ket
        options : dict
            A dict of key/value pairs that control how the operator is applied
            to the Ket.
        """
        return dispatch_method(self, '_apply_from_right_to', op, **options)


class BraBase(StateBase):
    """Base class for Bras.

    This class defines the dual property and the brackets for printing. This
    is an abstract base class and you should not instantiate it directly,
    instead use Bra.
    """

    # Define left and right brackets for bras in different formats
    lbracket = _lbracket
    rbracket = _straight_bracket
    lbracket_ucode = _lbracket_ucode
    rbracket_ucode = _straight_bracket_ucode
    lbracket_latex = r'\left\langle '
    # 定义 LaTeX 中右边的分隔符字符串
    rbracket_latex = r'\right|'

    @classmethod
    def _operators_to_state(self, ops, **options):
        # 调用当前类的双态类方法 _operators_to_state 将操作符转换为状态
        state = self.dual_class()._operators_to_state(ops, **options)
        # 返回双态的对偶
        return state.dual

    def _state_to_operators(self, op_classes, **options):
        # 调用对偶的 _state_to_operators 方法将状态转换为操作符类
        return self.dual._state_to_operators(op_classes, **options)

    def _enumerate_state(self, num_states, **options):
        # 调用对偶的 _enumerate_state 方法枚举对偶状态
        dual_states = self.dual._enumerate_state(num_states, **options)
        # 返回每个对偶状态的对偶
        return [x.dual for x in dual_states]

    @classmethod
    def default_args(self):
        # 调用双态类的 default_args 方法获取默认参数
        return self.dual_class().default_args()

    @classmethod
    def dual_class(self):
        # 返回双态类的基类 KetBase
        return KetBase

    def __mul__(self, other):
        """BraBase*other"""
        from sympy.physics.quantum.innerproduct import InnerProduct
        # 如果 other 是 KetBase 类的实例，返回 BraBase 和 other 的内积
        if isinstance(other, KetBase):
            return InnerProduct(self, other)
        else:
            # 否则调用父类 Expr 的 __mul__ 方法
            return Expr.__mul__(self, other)

    def __rmul__(self, other):
        """other*BraBase"""
        from sympy.physics.quantum.operator import OuterProduct
        # 如果 other 是 KetBase 类的实例，返回 other 和 BraBase 的外积
        if isinstance(other, KetBase):
            return OuterProduct(other, self)
        else:
            # 否则调用父类 Expr 的 __rmul__ 方法
            return Expr.__rmul__(self, other)

    def _represent(self, **options):
        """A default represent that uses the Ket's version."""
        from sympy.physics.quantum.dagger import Dagger
        # 调用对偶的 _represent 方法返回其表示
        return Dagger(self.dual._represent(**options))
class State(StateBase):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass



class Ket(State, KetBase):
    """A general time-independent Ket in quantum mechanics.

    Inherits from State and KetBase. This class should be used as the base
    class for all physical, time-independent Kets in a system. This class
    and its subclasses will be the main classes that users will use for
    expressing Kets in Dirac notation [1]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Ket and looking at its properties::

        >>> from sympy.physics.quantum import Ket
        >>> from sympy import symbols, I
        >>> k = Ket('psi')
        >>> k
        |psi>
        >>> k.hilbert_space
        H
        >>> k.is_commutative
        False
        >>> k.label
        (psi,)

    Ket's know about their associated bra::

        >>> k.dual
        <psi|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.Bra'>

    Take a linear combination of two kets::

        >>> k0 = Ket(0)
        >>> k1 = Ket(1)
        >>> 2*I*k0 - 4*k1
        2*I*|0> - 4*|1>

    Compound labels are passed as tuples::

        >>> n, m = symbols('n,m')
        >>> k = Ket(n,m)
        >>> k
        |nm>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """

    @classmethod
    def dual_class(self):
        return Bra



class Bra(State, BraBase):
    """A general time-independent Bra in quantum mechanics.

    Inherits from State and BraBase. A Bra is the dual of a Ket [1]_. This
    class and its subclasses will be the main classes that users will use for
    expressing Bras in Dirac notation.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Bra and look at its properties::

        >>> from sympy.physics.quantum import Bra
        >>> from sympy import symbols, I
        >>> b = Bra('psi')
        >>> b
        <psi|
        >>> b.hilbert_space
        H
        >>> b.is_commutative
        False

    Bra's know about their dual Ket's::

        >>> b.dual
        |psi>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.state.Ket'>

    Like Kets, Bras can have compound labels and be manipulated in a similar
    manner::

        >>> n, m = symbols('n,m')
        >>> b = Bra(n,m) - I*Bra(m,n)
        >>> b
        -I*<mn| + <nm|

    Symbols in a Bra can be substituted using ``.subs``::

        >>> b.subs(n,m)
        <mm| - I*<mm|

    References
    ==========
    # 定义一个类方法，用于返回类的另一个类，这里返回的是 Ket 类
    @classmethod
    def dual_class(self):
        # 返回 Ket 类作为结果
        return Ket
#-----------------------------------------------------------------------------
# Time dependent states, bras and kets.
#-----------------------------------------------------------------------------


class TimeDepState(StateBase):
    """Base class for a general time-dependent quantum state.

    This class is used as a base class for any time-dependent state. The main
    difference between this class and the time-independent state is that this
    class takes a second argument that is the time in addition to the usual
    label argument.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.
    """

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def default_args(self):
        """Returns default arguments for the state."""
        return ("psi", "t")

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """The label of the state."""
        return self.args[:-1]

    @property
    def time(self):
        """The time of the state."""
        return self.args[-1]

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print_time(self, printer, *args):
        """Prints the time representation."""
        return printer._print(self.time, *args)

    _print_time_repr = _print_time
    _print_time_latex = _print_time

    def _print_time_pretty(self, printer, *args):
        """Returns a pretty-printed representation of the time."""
        pform = printer._print(self.time, *args)
        return pform

    def _print_contents(self, printer, *args):
        """Prints the label and time contents."""
        label = self._print_label(printer, *args)
        time = self._print_time(printer, *args)
        return '%s;%s' % (label, time)

    def _print_label_repr(self, printer, *args):
        """Returns the representation of label and time in a printable format."""
        label = self._print_sequence(self.label, ',', printer, *args)
        time = self._print_time_repr(printer, *args)
        return '%s,%s' % (label, time)

    def _print_contents_pretty(self, printer, *args):
        """Returns a pretty-printed representation of label and time."""
        label = self._print_label_pretty(printer, *args)
        time = self._print_time_pretty(printer, *args)
        return printer._print_seq((label, time), delimiter=';')

    def _print_contents_latex(self, printer, *args):
        """Returns a LaTeX representation of label and time."""
        label = self._print_sequence(
            self.label, self._label_separator, printer, *args)
        time = self._print_time_latex(printer, *args)
        return '%s;%s' % (label, time)


class TimeDepKet(TimeDepState, KetBase):
    """General time-dependent Ket in quantum mechanics.

    This inherits from ``TimeDepState`` and ``KetBase`` and is the main class
    responsible for representing time-dependent quantum ket states.
    """
    @classmethod
    # 声明一个类方法，用于获取该类的对偶态类对象
    def dual_class(self):
        # 返回对偶态类 TimeDepBra
        return TimeDepBra
class TimeDepBra(TimeDepState, BraBase):
    """General time-dependent Bra in quantum mechanics.

    This inherits from TimeDepState and BraBase and is the main class that
    should be used for Bras that vary with time. Its dual is a TimeDepKet.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

        >>> from sympy.physics.quantum import TimeDepBra
        >>> b = TimeDepBra('psi', 't')
        >>> b
        <psi;t|
        >>> b.time
        t
        >>> b.label
        (psi,)
        >>> b.hilbert_space
        H
        >>> b.dual
        |psi;t>
    """

    @classmethod
    def dual_class(cls):
        # 返回该类的对偶类，TimeDepKet
        return TimeDepKet


class OrthogonalState(State, StateBase):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass

class OrthogonalKet(OrthogonalState, KetBase):
    """Orthogonal Ket in quantum mechanics.

    The inner product of two states with different labels will give zero,
    states with the same label will give one.

        >>> from sympy.physics.quantum import OrthogonalBra, OrthogonalKet
        >>> from sympy.abc import m, n
        >>> (OrthogonalBra(n)*OrthogonalKet(n)).doit()
        1
        >>> (OrthogonalBra(n)*OrthogonalKet(n+1)).doit()
        0
        >>> (OrthogonalBra(n)*OrthogonalKet(m)).doit()
        <n|m>
    """

    @classmethod
    def dual_class(cls):
        # 返回该类的对偶类，OrthogonalBra
        return OrthogonalBra

    def _eval_innerproduct(self, bra, **hints):
        # 求解内积，即两个量子态的乘积

        # 检查态的标签数是否一致
        if len(self.args) != len(bra.args):
            raise ValueError('Cannot multiply a ket that has a different number of labels.')

        # 遍历态的参数和对偶态的参数，计算差异
        for arg, bra_arg in zip(self.args, bra.args):
            diff = arg - bra_arg
            diff = diff.expand()

            is_zero = diff.is_zero

            # 如果差异不为零，则返回零（内积为零）
            if is_zero is False:
                return S.Zero  # i.e. Integer(0)

            # 如果差异为未确定，则返回空
            if is_zero is None:
                return None

        # 差异全部为零，返回一（内积为一）
        return S.One  # i.e. Integer(1)


class OrthogonalBra(OrthogonalState, BraBase):
    """Orthogonal Bra in quantum mechanics.
    """

    @classmethod
    def dual_class(cls):
        # 返回该类的对偶类，OrthogonalKet
        return OrthogonalKet


class Wavefunction(Function):
    """Class for representations in continuous bases

    This class takes an expression and coordinates in its constructor. It can
    be used to easily calculate normalizations and probabilities.

    Parameters
    ==========

    expr : Expr
           The expression representing the functional form of the w.f.

    coords : Symbol or tuple
           The coordinates to be integrated over, and their bounds

    Examples
    ========

    Particle in a box, specifying bounds in the more primitive way of using
    """

    # 该类未完成，需要继续添加代码实现其功能
    """
    # 定义一个新的类构造函数，处理传入的参数和选项
    # args 是传入的参数列表，options 是传入的关键字参数字典
    def __new__(cls, *args, **options):
        # 初始化一个与 args 相同长度的新参数列表 new_args
        new_args = [None for i in args]
        # 计数器，用于迭代 args 中的每个参数
        ct = 0
        # 遍历 args 中的每个参数 arg
        for arg in args:
            # 如果当前参数是一个 tuple 类型
            if isinstance(arg, tuple):
                # 将该 tuple 转换为 Tuple 对象，并存储在 new_args 中对应位置
                new_args[ct] = Tuple(*arg)
            else:
                # 如果当前参数不是 tuple 类型，则直接存储在 new_args 中对应位置
                new_args[ct] = arg
            # 计数器加一，以便处理下一个参数
            ct += 1

        # 调用父类的构造函数来创建新的实例，传入处理后的参数 new_args 和 options
        return super().__new__(cls, *new_args, **options)
    ```
    def __call__(self, *args, **options):
        # 获取波函数对象的自由变量
        var = self.variables

        # 检查传入参数个数与变量个数是否一致，如果不一致则抛出异常
        if len(args) != len(var):
            raise NotImplementedError(
                "Incorrect number of arguments to function!")

        ct = 0
        # 对每个变量进行限制条件检查，如果超出限制范围则返回0
        for v in var:
            lower, upper = self.limits[v]

            # 如果参数是表达式且不包含在限制符号中，则跳过比较
            if isinstance(args[ct], Expr) and \
                not (lower in args[ct].free_symbols
                     or upper in args[ct].free_symbols):
                continue

            # 如果参数超出上下限，则返回零
            if (args[ct] < lower) == True or (args[ct] > upper) == True:
                return S.Zero

            ct += 1

        expr = self.expr

        # 允许用户使用关键字调用，如 f(2, 4, m=1, n=1)
        for symbol in list(expr.free_symbols):
            if str(symbol) in options.keys():
                val = options[str(symbol)]
                expr = expr.subs(symbol, val)

        # 替换波函数中的变量，并返回计算结果
        return expr.subs(zip(var, args))

    def _eval_derivative(self, symbol):
        # 计算波函数关于指定符号的导数
        expr = self.expr
        deriv = expr._eval_derivative(symbol)

        # 返回新的波函数对象，表示导数
        return Wavefunction(deriv, *self.args[1:])

    def _eval_conjugate(self):
        # 返回波函数的共轭
        return Wavefunction(conjugate(self.expr), *self.args[1:])

    def _eval_transpose(self):
        # 返回波函数的转置，这里波函数是自身
        return self

    @property
    def free_symbols(self):
        # 返回波函数表达式中的自由符号集合
        return self.expr.free_symbols

    @property
    def is_commutative(self):
        """
        重写函数的 is_commutative 方法，以保持表达式中的顺序
        """
        return False

    @classmethod
    def eval(self, *args):
        # 类方法，用于评估波函数，此处总是返回 None
        return None

    @property
    def variables(self):
        """
        返回波函数依赖的坐标

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x,y = symbols('x,y')
            >>> f = Wavefunction(x*y, x, y)
            >>> f.variables
            (x, y)
            >>> g = Wavefunction(x*y, x)
            >>> g.variables
            (x,)

        """
        # 解析并返回波函数依赖的变量元组
        var = [g[0] if isinstance(g, Tuple) else g for g in self._args[1:]]
        return tuple(var)
    def limits(self):
        """
        Return the limits of the coordinates which the w.f. depends on If no
        limits are specified, defaults to ``(-oo, oo)``.

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, (x, 0, 1))
            >>> f.limits
            {x: (0, 1)}
            >>> f = Wavefunction(x**2, x)
            >>> f.limits
            {x: (-oo, oo)}
            >>> f = Wavefunction(x**2 + y**2, x, (y, -1, 2))
            >>> f.limits
            {x: (-oo, oo), y: (-1, 2)}

        """
        # 从 Wavefunction 的参数中提取限制条件，如果没有指定则使用默认的 (-oo, oo)
        limits = [(g[1], g[2]) if isinstance(g, Tuple) else (-oo, oo)
                  for g in self._args[1:]]
        # 将限制条件转换成字典形式并返回
        return dict(zip(self.variables, tuple(limits)))

    @property
    def expr(self):
        """
        Return the expression which is the functional form of the Wavefunction

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, x)
            >>> f.expr
            x**2

        """
        # 返回 Wavefunction 的功能形式表达式
        return self._args[0]

    @property
    def is_normalized(self):
        """
        Returns true if the Wavefunction is properly normalized

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.is_normalized
            True

        """
        # 判断 Wavefunction 是否已经被正确归一化
        return equal_valued(self.norm, 1)

    @property  # type: ignore
    @cacheit
    def norm(self):
        """
        Return the normalization of the specified functional form.

        This function integrates over the coordinates of the Wavefunction, with
        the bounds specified.

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            1
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            sqrt(2)*sqrt(L)/2

        """

        # 计算表达式的模的平方，即 \psi(x) * \bar{\psi}(x)
        exp = self.expr * conjugate(self.expr)
        # 获取变量和积分限制
        var = self.variables
        limits = self.limits

        # 对每个变量进行积分
        for v in var:
            curr_limits = limits[v]
            # 对表达式 exp 进行积分，积分变量 v 的范围是 curr_limits[0] 到 curr_limits[1]
            exp = integrate(exp, (v, curr_limits[0], curr_limits[1]))

        # 返回表达式 exp 的平方根作为标准化系数
        return sqrt(exp)

    def normalize(self):
        """
        Return a normalized version of the Wavefunction

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x = symbols('x', real=True)
            >>> L = symbols('L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.normalize()
            Wavefunction(sqrt(2)*sin(pi*n*x/L)/sqrt(L), (x, 0, L))

        """
        # 获取归一化常数
        const = self.norm

        # 如果常数为无穷大，则函数不可归一化，抛出异常
        if const is oo:
            raise NotImplementedError("The function is not normalizable!")
        else:
            # 返回归一化后的波函数对象
            return Wavefunction((const)**(-1) * self.expr, *self.args[1:])

    def prob(self):
        r"""
        Return the absolute magnitude of the w.f., `|\psi(x)|^2`

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', real=True)
            >>> n = symbols('n', integer=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.prob()
            Wavefunction(sin(pi*n*x/L)**2, x)

        """

        # 返回波函数的模的平方，即 |ψ(x)|^2
        return Wavefunction(self.expr * conjugate(self.expr), *self.variables)
```