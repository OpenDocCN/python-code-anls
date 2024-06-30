# `D:\src\scipysrc\sympy\sympy\physics\quantum\qexpr.py`

```
# 从 sympy 库中导入所需的类和函数
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence

# 从 sympy.physics.quantum.dagger 模块中导入 Dagger 类
from sympy.physics.quantum.dagger import Dagger
# 从 sympy.physics.quantum.matrixutils 模块中导入多个函数
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix,
    to_sympy, to_numpy, to_scipy_sparse
)

# __all__ 列表指定了该模块中公开的对象，即可被导入的对象名
__all__ = [
    'QuantumError',  # 导出 QuantumError 类
    'QExpr'           # 导出 QExpr 类
]


#-----------------------------------------------------------------------------
# Error handling
#-----------------------------------------------------------------------------

# 定义一个 QuantumError 异常类，继承自 Python 内置的 Exception 类
class QuantumError(Exception):
    pass


def _qsympify_sequence(seq):
    """Convert elements of a sequence to standard form.

    This is like sympify, but it performs special logic for arguments passed
    to QExpr. The following conversions are done:

    * (list, tuple, Tuple) => _qsympify_sequence each element and convert
      sequence to a Tuple.
    * basestring => Symbol
    * Matrix => Matrix
    * other => sympify

    Strings are passed to Symbol, not sympify to make sure that variables like
    'pi' are kept as Symbols, not the SymPy built-in number subclasses.

    Examples
    ========

    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence
    >>> _qsympify_sequence((1,2,[3,4,[1,]]))
    (1, 2, (3, 4, (1,)))

    """
    # 将序列中的每个元素转换为标准形式，根据元素类型进行不同的转换逻辑
    return tuple(__qsympify_sequence_helper(seq))


def __qsympify_sequence_helper(seq):
    """
       Helper function for _qsympify_sequence
       This function does the actual work.
    """
    # 如果 seq 不是序列，则进行符号化（sympify）
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq  # 如果是 Matrix 对象，直接返回
        elif isinstance(seq, str):
            return Symbol(seq)  # 如果是字符串，转换为 Symbol 对象
        else:
            return sympify(seq)  # 否则调用 sympify 进行通用转换

    # 如果 seq 是 QExpr 类的实例，则直接返回
    if isinstance(seq, QExpr):
        return seq

    # 如果是列表，则递归地对列表中的每个元素调用 __qsympify_sequence_helper 函数
    result = [__qsympify_sequence_helper(item) for item in seq]

    return Tuple(*result)


#-----------------------------------------------------------------------------
# Basic Quantum Expression from which all objects descend
#-----------------------------------------------------------------------------

# 定义 QExpr 类，继承自 Expr 类
class QExpr(Expr):
    """A base class for all quantum object like operators and states."""

    # 在 sympy 中，slots 是由 __new__ 方法动态计算的实例属性，不包含在 args 中，但由 args 派生。
    
    # 量子对象所属的希尔伯特空间
    __slots__ = ('hilbert_space', )

    # 量子对象不可交换
    is_commutative = False

    # 在打印标签时使用的分隔符
    _label_separator = ''

    @property
    def free_symbols(self):
        return {self}  # 返回一个集合，包含自身作为自由符号的 QExpr 对象
    def __new__(cls, *args, **kwargs):
        """创建一个新的量子对象。

        Parameters
        ==========

        args : tuple
            描述量子对象的一组唯一标识符或参数。对于状态而言，这可能是其符号或一组量子数。

        Examples
        ========

        >>> from sympy.physics.quantum.qexpr import QExpr
        >>> q = QExpr(0)
        >>> q
        0
        >>> q.label
        (0,)
        >>> q.hilbert_space
        H
        >>> q.args
        (0,)
        >>> q.is_commutative
        False
        """

        # 首先计算 args，并调用 Expr.__new__ 来创建实例
        args = cls._eval_args(args, **kwargs)
        if len(args) == 0:
            args = cls._eval_args(tuple(cls.default_args()), **kwargs)
        inst = Expr.__new__(cls, *args)
        # 设置实例的 slots
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        """使用给定的 hilbert_space 和 args 创建此类的新实例。

        当已经确信这些参数是有效的、处于最终适当形式时，可用此方法绕过 ``__new__``
        方法中更复杂的逻辑进行对象创建优化。
        """

        obj = Expr.__new__(cls, *args, **old_assumptions)
        obj.hilbert_space = hilbert_space
        return obj

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """标签是对象的唯一标识符集合。

        通常情况下，这将包括关于状态的所有信息，*除了*时间（对于时间相关对象）。

        必须是一个 tuple，而不是 Tuple。
        """
        if len(self.args) == 0:  # 如果没有指定标签，返回默认值
            return self._eval_args(list(self.default_args()))
        else:
            return self.args

    @property
    def is_symbolic(self):
        return True

    @classmethod
    def default_args(self):
        """如果未指定参数，则返回要通过构造函数运行的默认参数集。

        注意: 任何覆盖此方法的类都必须返回参数的元组。子类应覆盖此方法以指定 kets 和 operators 的默认参数。
        """
        raise NotImplementedError("No default arguments for this class!")

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------
    def _eval_adjoint(self):
        # 调用父类 Expr 的 _eval_adjoint 方法，获取其返回的对象
        obj = Expr._eval_adjoint(self)
        # 如果返回对象为 None，则创建一个新的 Dagger 类型对象
        if obj is None:
            obj = Expr.__new__(Dagger, self)
        # 如果返回对象是 QExpr 类型，则设置其 hilbert_space 属性为当前对象的 hilbert_space
        if isinstance(obj, QExpr):
            obj.hilbert_space = self.hilbert_space
        # 返回处理后的对象
        return obj

    @classmethod
    def _eval_args(cls, args):
        """处理传递给 __new__ 方法的参数。

        这个方法简单地通过 _qsympify_sequence 处理 args。
        """
        return _qsympify_sequence(args)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """从参数中计算 Hilbert 空间的实例。
        """
        from sympy.physics.quantum.hilbert import HilbertSpace
        return HilbertSpace()

    #-------------------------------------------------------------------------
    # 打印
    #-------------------------------------------------------------------------

    # 打印工具：这些操作于原始 SymPy 对象上

    def _print_sequence(self, seq, sep, printer, *args):
        result = []
        for item in seq:
            result.append(printer._print(item, *args))
        return sep.join(result)

    def _print_sequence_pretty(self, seq, sep, printer, *args):
        pform = printer._print(seq[0], *args)
        for item in seq[1:]:
            pform = prettyForm(*pform.right(sep))
            pform = prettyForm(*pform.right(printer._print(item, *args)))
        return pform

    def _print_subscript_pretty(self, a, b):
        top = prettyForm(*b.left(' '*a.width()))
        bot = prettyForm(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    def _print_superscript_pretty(self, a, b):
        return a**b

    def _print_parens_pretty(self, pform, left='(', right=')'):
        return prettyForm(*pform.parens(left=left, right=right))

    # 打印标签（即参数）

    def _print_label(self, printer, *args):
        """打印 QExpr 的标签

        这个方法打印 self.label，使用 self._label_separator 分隔元素。不应覆盖此方法，而是覆盖 _print_contents 来更改打印行为。
        """
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_repr(self, printer, *args):
        return self._print_sequence(
            self.label, ',', printer, *args
        )

    def _print_label_pretty(self, printer, *args):
        return self._print_sequence_pretty(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    # 打印内容（默认为标签）
    def _print_contents(self, printer, *args):
        """Printer for contents of QExpr

        Handles the printing of any unique identifying contents of a QExpr to
        print as its contents, such as any variables or quantum numbers. The
        default is to print the label, which is almost always the args. This
        should not include printing of any brackets or parentheses.
        """
        # 使用_print_label方法打印QExpr对象的内容
        return self._print_label(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        """Pretty printer for contents of QExpr

        Handles the pretty printing of QExpr contents. This method delegates
        to _print_label_pretty for printing.
        """
        # 使用_print_label_pretty方法美化打印QExpr对象的内容
        return self._print_label_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        """Latex printer for contents of QExpr

        Handles the latex formatting of QExpr contents. This method uses
        _print_label_latex to produce the LaTeX representation.
        """
        # 使用_print_label_latex方法以Latex格式打印QExpr对象的内容
        return self._print_label_latex(printer, *args)

    # Main printing methods

    def _sympystr(self, printer, *args):
        """Default string representation of QExpr objects

        Handles the default string representation of a QExpr object. Calls
        _print_contents to include any additional content in the string output.
        """
        # 使用_print_contents方法生成QExpr对象的默认字符串表示
        return self._print_contents(printer, *args)

    def _sympyrepr(self, printer, *args):
        """Default representation of QExpr objects

        Generates the default representation of a QExpr object by combining
        its class name and printed label.
        """
        classname = self.__class__.__name__
        # 使用_print_label_repr方法生成包含类名的QExpr对象的默认表示形式
        label = self._print_label_repr(printer, *args)
        return '%s(%s)' % (classname, label)

    def _pretty(self, printer, *args):
        """Pretty printing of QExpr objects

        Generates a pretty printed representation of a QExpr object by calling
        _print_contents_pretty.
        """
        # 使用_print_contents_pretty方法生成QExpr对象的美化打印形式
        pform = self._print_contents_pretty(printer, *args)
        return pform

    def _latex(self, printer, *args):
        """Latex representation of QExpr objects

        Generates a LaTeX representation of a QExpr object by calling
        _print_contents_latex.
        """
        # 使用_print_contents_latex方法生成QExpr对象的Latex表示
        return self._print_contents_latex(printer, *args)

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        """Representation of QExpr default basis

        Raises NotImplementedError indicating that this object does not have
        a default basis representation.
        """
        # 抛出NotImplementedError，表明该对象没有默认基表示
        raise NotImplementedError('This object does not have a default basis')
    def _represent(self, *, basis=None, **options):
        """Represent this object in a given basis.

        This method dispatches to the actual methods that perform the
        representation. Subclases of QExpr should define various methods to
        determine how the object will be represented in various bases. The
        format of these methods is::

            def _represent_BasisName(self, basis, **options):

        Thus to define how a quantum object is represented in the basis of
        the operator Position, you would define::

            def _represent_Position(self, basis, **options):

        Usually, basis object will be instances of Operator subclasses, but
        there is a chance we will relax this in the future to accommodate other
        types of basis sets that are not associated with an operator.

        If the ``format`` option is given it can be ("sympy", "numpy",
        "scipy.sparse"). This will ensure that any matrices that result from
        representing the object are returned in the appropriate matrix format.

        Parameters
        ==========

        basis : Operator
            The Operator whose basis functions will be used as the basis for
            representation.
        options : dict
            A dictionary of key/value pairs that give options and hints for
            the representation, such as the number of basis functions to
            be used.
        """
        if basis is None:
            # 如果未指定基础算符，则使用默认基础进行对象的表示
            result = self._represent_default_basis(**options)
        else:
            # 否则，根据基础算符分派到相应的表示方法
            result = dispatch_method(self, '_represent', basis, **options)

        # 如果结果是矩阵表示，则根据指定的格式进行转换
        format = options.get('format', 'sympy')
        result = self._format_represent(result, format)
        return result

    def _format_represent(self, result, format):
        """Format the representation result into the desired matrix format.

        Depending on the specified format ('sympy', 'numpy', 'scipy.sparse'),
        convert the result to the appropriate matrix type if it is not already
        in that format.

        Parameters
        ==========
        
        result : object
            The representation result that needs formatting.
        format : str
            The desired format for the result ('sympy', 'numpy', 'scipy.sparse').

        Returns
        =======
        
        object
            The formatted representation result.

        """
        if format == 'sympy' and not isinstance(result, Matrix):
            # 如果格式为 'sympy' 并且结果不是 SymPy 的 Matrix 对象，则转换为 SymPy 的 Matrix
            return to_sympy(result)
        elif format == 'numpy' and not isinstance(result, numpy_ndarray):
            # 如果格式为 'numpy' 并且结果不是 NumPy 的 ndarray 对象，则转换为 NumPy 的 ndarray
            return to_numpy(result)
        elif format == 'scipy.sparse' and \
                not isinstance(result, scipy_sparse_matrix):
            # 如果格式为 'scipy.sparse' 并且结果不是 SciPy 的 sparse matrix 对象，则转换为 SciPy 的 sparse matrix
            return to_scipy_sparse(result)

        # 如果格式已经匹配或未知格式，则直接返回结果
        return result
# 将表达式 e 拆分为可交换部分和不可交换部分
def split_commutative_parts(e):
    # 调用 e 的 args_cnc 方法，返回可交换部分和不可交换部分
    c_part, nc_part = e.args_cnc()
    # 将可交换部分转换为列表
    c_part = list(c_part)
    return c_part, nc_part


# 将表达式 e 拆分为表达式部分和非可交换 QExpr 部分
def split_qexpr_parts(e):
    # 初始化空列表来存储表达式部分和非可交换 QExpr 部分
    expr_part = []
    qexpr_part = []
    # 遍历 e 的所有参数
    for arg in e.args:
        # 如果参数不是 QExpr 类型，则加入表达式部分列表
        if not isinstance(arg, QExpr):
            expr_part.append(arg)
        else:
            # 如果参数是 QExpr 类型，则加入非可交换 QExpr 部分列表
            qexpr_part.append(arg)
    return expr_part, qexpr_part


# 分发方法给适当的处理器
def dispatch_method(self, basename, arg, **options):
    # 构建方法名，格式为 'basename_arg的类名'
    method_name = '%s_%s' % (basename, arg.__class__.__name__)
    # 检查是否存在对应的方法
    if hasattr(self, method_name):
        # 获取对应方法的引用
        f = getattr(self, method_name)
        # 调用方法处理参数，并传入选项
        # 可能会引发异常，允许异常传播
        result = f(arg, **options)
        # 如果方法返回结果不为 None，则返回结果
        if result is not None:
            return result
    # 如果找不到对应的方法，则抛出未实现错误
    raise NotImplementedError(
        "%s.%s cannot handle: %r" %
        (self.__class__.__name__, basename, arg)
    )
```