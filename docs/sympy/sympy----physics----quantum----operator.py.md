# `D:\src\scipysrc\sympy\sympy\physics\quantum\operator.py`

```
"""
Quantum mechanical operators.

TODO:

* Fix early 0 in apply_operators.
* Debug and test apply_operators.
* Get cse working with classes in this file.
* Doctests and documentation of special methods for InnerProduct, Commutator,
  AntiCommutator, represent, apply_operators.
"""
from typing import Optional

from sympy.core.add import Add  # 导入加法表达式类
from sympy.core.expr import Expr  # 导入表达式基类
from sympy.core.function import (Derivative, expand)  # 导入导数和表达式展开函数
from sympy.core.mul import Mul  # 导入乘法表达式类
from sympy.core.numbers import oo  # 导入无穷大常量
from sympy.core.singleton import S  # 导入单例类
from sympy.printing.pretty.stringpict import prettyForm  # 导入美化字符串表示形式类
from sympy.physics.quantum.dagger import Dagger  # 导入共轭转置运算符类
from sympy.physics.quantum.qexpr import QExpr, dispatch_method  # 导入量子表达式和分派方法函数
from sympy.matrices import eye  # 导入单位矩阵类

__all__ = [
    'Operator',
    'HermitianOperator',
    'UnitaryOperator',
    'IdentityOperator',
    'OuterProduct',
    'DifferentialOperator'
]

#-----------------------------------------------------------------------------
# Operators and outer products
#-----------------------------------------------------------------------------


class Operator(QExpr):
    """Base class for non-commuting quantum operators.

    An operator maps between quantum states [1]_. In quantum mechanics,
    observables (including, but not limited to, measured physical values) are
    represented as Hermitian operators [2]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    Create an operator and examine its attributes::

        >>> from sympy.physics.quantum import Operator
        >>> from sympy import I
        >>> A = Operator('A')
        >>> A
        A
        >>> A.hilbert_space
        H
        >>> A.label
        (A,)
        >>> A.is_commutative
        False

    Create another operator and do some arithmetic operations::

        >>> B = Operator('B')
        >>> C = 2*A*A + I*B
        >>> C
        2*A**2 + I*B

    Operators do not commute::

        >>> A.is_commutative
        False
        >>> B.is_commutative
        False
        >>> A*B == B*A
        False

    Polymonials of operators respect the commutation properties::

        >>> e = (A+B)**3
        >>> e.expand()
        A*B*A + A*B**2 + A**2*B + A**3 + B*A*B + B*A**2 + B**2*A + B**3

    Operator inverses are handle symbolically::

        >>> A.inv()
        A**(-1)
        >>> A*A.inv()
        1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Operator_%28physics%29
    .. [2] https://en.wikipedia.org/wiki/Observable
    """
    is_hermitian: Optional[bool] = None  # 是否为厄米算符的标志，初始为None
    is_unitary: Optional[bool] = None  # 是否为单位算符的标志，初始为None

    @classmethod
    def default_args(self):
        return ("O",)  # 默认参数返回元组 ("O",)

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    _label_separator = ','  # 标签分隔符设为逗号
    # 返回当前对象的类名
    def _print_operator_name(self, printer, *args):
        return self.__class__.__name__

    # 将 _print_operator_name 方法赋值给 _print_operator_name_latex
    _print_operator_name_latex = _print_operator_name

    # 返回当前对象的类名，用于 PrettyForm 对象
    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm(self.__class__.__name__)

    # 打印对象的内容，如果标签长度为1，则调用 _print_label 方法；否则返回格式化字符串
    def _print_contents(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label(printer, *args)
        else:
            return '%s(%s)' % (
                self._print_operator_name(printer, *args),
                self._print_label(printer, *args)
            )

    # 打印对象的内容，如果标签长度为1，则调用 _print_label_pretty 方法；否则返回 PrettyForm 对象
    def _print_contents_pretty(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_pretty(printer, *args)
        else:
            pform = self._print_operator_name_pretty(printer, *args)
            label_pform = self._print_label_pretty(printer, *args)
            label_pform = prettyForm(
                *label_pform.parens(left='(', right=')')
            )
            pform = prettyForm(*pform.right(label_pform))
            return pform

    # 打印对象的内容，如果标签长度为1，则调用 _print_label_latex 方法；否则返回 LaTeX 格式的字符串
    def _print_contents_latex(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_latex(printer, *args)
        else:
            return r'%s\left(%s\right)' % (
                self._print_operator_name_latex(printer, *args),
                self._print_label_latex(printer, *args)
            )

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    # 评估对易子 [self, other]，如果能够评估则返回结果，否则返回 None
    def _eval_commutator(self, other, **options):
        """Evaluate [self, other] if known, return None if not known."""
        return dispatch_method(self, '_eval_commutator', other, **options)

    # 评估反对易子 {self, other}，如果能够评估则返回结果
    def _eval_anticommutator(self, other, **options):
        """Evaluate {self, other} if known."""
        return dispatch_method(self, '_eval_anticommutator', other, **options)

    #-------------------------------------------------------------------------
    # Operator application
    #-------------------------------------------------------------------------

    # 应用操作符 self 到 ket 上，返回结果
    def _apply_operator(self, ket, **options):
        return dispatch_method(self, '_apply_operator', ket, **options)

    # 从右侧应用到 bra，但这里返回 None
    def _apply_from_right_to(self, bra, **options):
        return None

    # 抛出未实现的错误，因为 matrix_element 方法没有定义
    def matrix_element(self, *args):
        raise NotImplementedError('matrix_elements is not defined')

    # 返回对象的逆操作
    def inverse(self):
        return self._eval_inverse()

    # 将 _eval_inverse 方法赋值给 inv 属性
    inv = inverse

    # 评估对象的逆操作，返回 self 的负一次方
    def _eval_inverse(self):
        return self**(-1)

    # 定义乘法操作，如果 other 是 IdentityOperator，则返回 self；否则返回 Mul(self, other)
    def __mul__(self, other):
        if isinstance(other, IdentityOperator):
            return self
        return Mul(self, other)
class HermitianOperator(Operator):
    """A Hermitian operator that satisfies H == Dagger(H).

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, HermitianOperator
    >>> H = HermitianOperator('H')
    >>> Dagger(H)
    H
    """

    is_hermitian = True  # 设置属性表明这是一个厄米算符

    def _eval_inverse(self):
        if isinstance(self, UnitaryOperator):  # 检查是否是单位算符
            return self  # 如果是，返回自身
        else:
            return Operator._eval_inverse(self)  # 否则调用父类的逆运算方法

    def _eval_power(self, exp):
        if isinstance(self, UnitaryOperator):  # 检查是否是单位算符
            if exp.is_even:
                from sympy.core.singleton import S
                return S.One  # 如果指数是偶数，返回单位矩阵
            elif exp.is_odd:
                return self  # 如果指数是奇数，返回自身
        # 在其他情况下不进行简化
        return Operator._eval_power(self, exp)  # 调用父类的幂运算方法


class UnitaryOperator(Operator):
    """A unitary operator that satisfies U*Dagger(U) == 1.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, UnitaryOperator
    >>> U = UnitaryOperator('U')
    >>> U*Dagger(U)
    1
    """
    is_unitary = True  # 设置属性表明这是一个单位算符

    def _eval_adjoint(self):
        return self._eval_inverse()  # 返回自身的逆运算结果


class IdentityOperator(Operator):
    """An identity operator I that satisfies op * I == I * op == op for any
    operator op.

    Parameters
    ==========

    N : Integer
        Optional parameter that specifies the dimension of the Hilbert space
        of operator. This is used when generating a matrix representation.

    Examples
    ========

    >>> from sympy.physics.quantum import IdentityOperator
    >>> IdentityOperator()
    I
    """
    is_hermitian = True  # 设置属性表明这是一个厄米算符
    is_unitary = True  # 设置属性表明这是一个单位算符

    @property
    def dimension(self):
        return self.N  # 返回算符的维度信息

    @classmethod
    def default_args(self):
        return (oo,)  # 返回默认参数无穷大

    def __init__(self, *args, **hints):
        if not len(args) in (0, 1):
            raise ValueError('0 or 1 parameters expected, got %s' % args)

        self.N = args[0] if (len(args) == 1 and args[0]) else oo  # 初始化算符的维度信息，默认为无穷大

    def _eval_commutator(self, other, **hints):
        return S.Zero  # 返回零，表示这是一个单位算符，与其他算符的对易子为零

    def _eval_anticommutator(self, other, **hints):
        return 2 * other  # 返回2乘以另一个算符，表示反对易子的计算

    def _eval_inverse(self):
        return self  # 返回自身，单位算符的逆运算是自身

    def _eval_adjoint(self):
        return self  # 返回自身，单位算符的伴随运算是自身

    def _apply_operator(self, ket, **options):
        return ket  # 应用算符到右侧态矢的操作，返回态矢本身

    def _apply_from_right_to(self, bra, **options):
        return bra  # 应用算符到左侧态矢的操作，返回态矢本身

    def _eval_power(self, exp):
        return self  # 返回自身，单位算符的任意幂次方仍然是自身

    def _print_contents(self, printer, *args):
        return 'I'  # 返回打印时的内容，表示为I，即单位算符的打印输出
    # 返回一个表示空格的美观形式的对象
    def _print_contents_pretty(self, printer, *args):
        return prettyForm('I')

    # 返回一个表示空格的 LaTeX 形式的对象
    def _print_contents_latex(self, printer, *args):
        return r'{\mathcal{I}}'

    # 定义乘法运算符的重载方法，用于乘法操作
    def __mul__(self, other):
        # 如果乘数是 Operator 或者 Dagger 类型，则直接返回乘数本身
        if isinstance(other, (Operator, Dagger)):
            return other
        # 否则返回当前对象和乘数的乘积
        return Mul(self, other)

    # 返回默认基础的表示，通常是单位矩阵
    def _represent_default_basis(self, **options):
        # 如果维度 N 不存在或为无穷大，则抛出未实现错误
        if not self.N or self.N == oo:
            raise NotImplementedError('Cannot represent infinite dimensional' +
                                      ' identity operator as a matrix')

        # 获取表示格式，默认为 'sympy'
        format = options.get('format', 'sympy')
        # 如果表示格式不是 'sympy'，则抛出未实现错误
        if format != 'sympy':
            raise NotImplementedError('Representation in format ' +
                                      '%s not implemented.' % format)

        # 返回一个维度为 self.N 的单位矩阵作为默认基础表示
        return eye(self.N)
class OuterProduct(Operator):
    """An unevaluated outer product between a ket and bra.

    This constructs an outer product between any subclass of ``KetBase`` and
    ``BraBase`` as ``|a><b|``. An ``OuterProduct`` inherits from Operator as they act as
    operators in quantum expressions.  For reference see [1]_.

    Parameters
    ==========

    ket : KetBase
        The ket on the left side of the outer product.
    bar : BraBase
        The bra on the right side of the outer product.

    Examples
    ========

    Create a simple outer product by hand and take its dagger::

        >>> from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger
        >>> from sympy.physics.quantum import Operator

        >>> k = Ket('k')  # 创建一个名为 'k' 的 ket 对象
        >>> b = Bra('b')  # 创建一个名为 'b' 的 bra 对象
        >>> op = OuterProduct(k, b)  # 创建一个 ket 和 bra 的外积对象
        >>> op  # 打印外积对象
        |k><b|
        >>> op.hilbert_space  # 获取外积对象的 Hilbert 空间
        H
        >>> op.ket  # 获取外积对象中的 ket 部分
        |k>
        >>> op.bra  # 获取外积对象中的 bra 部分
        <b|
        >>> Dagger(op)  # 对外积对象进行 dagger 操作，即转置并取复共轭
        |b><k|

    In simple products of kets and bras outer products will be automatically
    identified and created::

        >>> k*b  # 创建 k 和 b 的外积
        |k><b|

    But in more complex expressions, outer products are not automatically
    created::

        >>> A = Operator('A')
        >>> A*k*b  # 复合表达式中，外积不会自动创建
        A*|k>*<b|

    A user can force the creation of an outer product in a complex expression
    by using parentheses to group the ket and bra::

        >>> A*(k*b)  # 使用括号强制创建外积
        A*|k><b|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Outer_product
    """
    is_commutative = False  # 设置 OuterProduct 类不可交换
    # 定义一个特殊方法 __new__，用于创建新的对象实例
    def __new__(cls, *args, **old_assumptions):
        # 导入必要的类：KetBase 和 BraBase
        from sympy.physics.quantum.state import KetBase, BraBase

        # 检查参数数量是否为2个，否则引发 ValueError 异常
        if len(args) != 2:
            raise ValueError('2 parameters expected, got %d' % len(args))

        # 对第一个参数和第二个参数进行展开
        ket_expr = expand(args[0])
        bra_expr = expand(args[1])

        # 检查第一个参数是否是 KetBase 或 Mul 类型，第二个参数是否是 BraBase 或 Mul 类型
        if (isinstance(ket_expr, (KetBase, Mul)) and
                isinstance(bra_expr, (BraBase, Mul))):
            # 分别提取系数和表达式列表
            ket_c, kets = ket_expr.args_cnc()
            bra_c, bras = bra_expr.args_cnc()

            # 如果表达式列表中不是单一的 KetBase 子类，则引发 TypeError 异常
            if len(kets) != 1 or not isinstance(kets[0], KetBase):
                raise TypeError('KetBase subclass expected'
                                ', got: %r' % Mul(*kets))

            # 如果表达式列表中不是单一的 BraBase 子类，则引发 TypeError 异常
            if len(bras) != 1 or not isinstance(bras[0], BraBase):
                raise TypeError('BraBase subclass expected'
                                ', got: %r' % Mul(*bras))

            # 如果 ket 和 bra 不是对偶类，则引发 TypeError 异常
            if not kets[0].dual_class() == bras[0].__class__:
                raise TypeError(
                    'ket and bra are not dual classes: %r, %r' %
                    (kets[0].__class__, bras[0].__class__)
                    )

            # 创建一个新的 Expr 对象实例，传入 ket 和 bra，同时传入旧的假设参数
            obj = Expr.__new__(cls, *(kets[0], bras[0]), **old_assumptions)
            # 设置 hilbert_space 属性为 ket 的 hilbert_space
            obj.hilbert_space = kets[0].hilbert_space
            # 返回乘积结果：ket_c + bra_c 乘以 obj
            return Mul(*(ket_c + bra_c)) * obj

        # 初始化操作项列表
        op_terms = []
        # 如果 ket_expr 和 bra_expr 都是 Add 类型
        if isinstance(ket_expr, Add) and isinstance(bra_expr, Add):
            # 分别迭代 ket_expr 和 bra_expr 中的项，生成 OuterProduct 对象并加入 op_terms
            for ket_term in ket_expr.args:
                for bra_term in bra_expr.args:
                    op_terms.append(OuterProduct(ket_term, bra_term,
                                                 **old_assumptions))
        # 如果 ket_expr 是 Add 类型，bra_expr 不是 Add 类型
        elif isinstance(ket_expr, Add):
            # 迭代 ket_expr 中的项，生成 OuterProduct 对象并加入 op_terms
            for ket_term in ket_expr.args:
                op_terms.append(OuterProduct(ket_term, bra_expr,
                                             **old_assumptions))
        # 如果 bra_expr 是 Add 类型，ket_expr 不是 Add 类型
        elif isinstance(bra_expr, Add):
            # 迭代 bra_expr 中的项，生成 OuterProduct 对象并加入 op_terms
            for bra_term in bra_expr.args:
                op_terms.append(OuterProduct(ket_expr, bra_term,
                                             **old_assumptions))
        else:
            # 如果 ket_expr 和 bra_expr 都不是 Add 类型，则引发 TypeError 异常
            raise TypeError(
                'Expected ket and bra expression, got: %r, %r' %
                (ket_expr, bra_expr)
                )

        # 返回所有操作项的加和结果
        return Add(*op_terms)

    @property
    def ket(self):
        """Return the ket on the left side of the outer product."""
        # 返回当前对象的第一个参数，即 ket
        return self.args[0]

    @property
    def bra(self):
        """Return the bra on the right side of the outer product."""
        # 返回当前对象的第二个参数，即 bra
        return self.args[1]

    def _eval_adjoint(self):
        # 返回当前对象的共轭转置，即将 bra 和 ket 互换并取 Dagger
        return OuterProduct(Dagger(self.bra), Dagger(self.ket))

    def _sympystr(self, printer, *args):
        # 返回当前对象的字符串表示，使用 printer 打印 ket 和 bra
        return printer._print(self.ket) + printer._print(self.bra)

    def _sympyrepr(self, printer, *args):
        # 返回当前对象的 repr 表示，使用 printer 打印 ket 和 bra
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.ket, *args), printer._print(self.bra, *args))
    # 调用对象的 _pretty 方法，获取格式化后的表示形式
    def _pretty(self, printer, *args):
        pform = self.ket._pretty(printer, *args)
        # 调用对象的 _pretty 方法，获取格式化后的表示形式，并构造新的 prettyForm 对象
        return prettyForm(*pform.right(self.bra._pretty(printer, *args)))

    # 调用对象的 _latex 方法，获取 LaTeX 格式的表示形式
    def _latex(self, printer, *args):
        k = printer._print(self.ket, *args)
        b = printer._print(self.bra, *args)
        # 返回表示两个对象 LaTeX 格式的字符串拼接结果
        return k + b

    # 调用对象的 _represent 方法，获取对象的数学表示
    def _represent(self, **options):
        k = self.ket._represent(**options)
        b = self.bra._represent(**options)
        # 返回两个对象数学表示的乘积
        return k*b

    # 对对象进行求迹运算的评估
    def _eval_trace(self, **kwargs):
        # 如果操作数是张量积，则可能会有不同的处理方式
        # TODO 如果操作数是张量积，则此处的处理方式可能不同，需要进一步确认
        return self.ket._eval_trace(self.bra, **kwargs)
    @property
    def variables(self):
        """
        返回用于评估指定任意表达式中函数的变量

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Symbol, Function, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
        >>> d.variables
        (x,)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.variables
        (x, y)
        """

        return self.args[-1].args

    @property
    def function(self):
        """
        返回将要用波函数替换的函数

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.function
        f(x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.function
        f(x, y)
        """

        return self.args[-1]
    def expr(self):
        """
        返回将波函数替换为其中的任意表达式

        示例
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.expr
        Derivative(f(x), x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.expr
        Derivative(f(x, y), x) + Derivative(f(x, y), y)
        """

        return self.args[0]

    @property
    def free_symbols(self):
        """
        返回表达式的自由符号集合。
        """

        return self.expr.free_symbols

    def _apply_operator_Wavefunction(self, func, **options):
        """
        将波函数操作符应用于指定的函数，并返回新的波函数对象。

        参数：
        func : callable
            应用波函数操作符的函数对象。
        **options : dict
            其他选项参数，暂时未使用。

        返回：
        Wavefunction
            应用操作后的新波函数对象。

        """

        from sympy.physics.quantum.state import Wavefunction
        var = self.variables
        wf_vars = func.args[1:]

        f = self.function
        new_expr = self.expr.subs(f, func(*var))
        new_expr = new_expr.doit()

        return Wavefunction(new_expr, *wf_vars)

    def _eval_derivative(self, symbol):
        """
        返回对给定符号的导数作为新的微分操作符对象。

        参数：
        symbol : Symbol
            要对其求导的符号。

        返回：
        DifferentialOperator
            新的微分操作符对象，表示对给定符号的导数。

        """

        new_expr = Derivative(self.expr, symbol)
        return DifferentialOperator(new_expr, self.args[-1])

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print(self, printer, *args):
        """
        生成一个字符串表示，用于在打印输出时显示操作符和标签。

        参数：
        printer : Printer
            打印机对象，用于生成打印输出的字符串。
        *args : tuple
            其他可能的参数，暂时未使用。

        返回：
        str
            表示操作符和标签的字符串。

        """

        return '%s(%s)' % (
            self._print_operator_name(printer, *args),
            self._print_label(printer, *args)
        )

    def _print_pretty(self, printer, *args):
        """
        生成一个漂亮的打印输出形式，用于显示操作符和标签。

        参数：
        printer : PrettyPrinter
            漂亮打印机对象，用于生成漂亮的打印输出。
        *args : tuple
            其他可能的参数，暂时未使用。

        返回：
        prettyForm
            表示操作符和标签的漂亮打印输出形式。

        """

        pform = self._print_operator_name_pretty(printer, *args)
        label_pform = self._print_label_pretty(printer, *args)
        label_pform = prettyForm(
            *label_pform.parens(left='(', right=')')
        )
        pform = prettyForm(*pform.right(label_pform))
        return pform
```