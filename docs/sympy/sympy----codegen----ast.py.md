# `D:\src\scipysrc\sympy\sympy\codegen\ast.py`

```
"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------
::

  *Basic*
       |
       |
   CodegenAST
       |
       |--->AssignmentBase
       |             |--->Assignment
       |             |--->AugmentedAssignment
       |                                    |--->AddAugmentedAssignment
       |                                    |--->SubAugmentedAssignment
       |                                    |--->MulAugmentedAssignment
       |                                    |--->DivAugmentedAssignment
       |                                    |--->ModAugmentedAssignment
       |
       |--->CodeBlock
       |
       |
       |--->Token
                |--->Attribute
                |--->For
                |--->String
                |       |--->QuotedString
                |       |--->Comment
                |--->Type
                |       |--->IntBaseType
                |       |              |--->_SizedIntType
                |       |                               |--->SignedIntType
                |       |                               |--->UnsignedIntType
                |       |--->FloatBaseType
                |                        |--->FloatType
                |                        |--->ComplexBaseType
                |                                           |--->ComplexType
                |--->Node
                |       |--->Variable
                |       |           |---> Pointer
                |       |--->FunctionPrototype
                |                            |--->FunctionDefinition
                |--->Element
                |--->Declaration
                |--->While
                |--->Scope
                |--->Stream
                |--->Print
                |--->FunctionCall
                |--->BreakToken
                |--->ContinueToken
                |--->NoneToken
                |--->Return


Predefined types
----------------

A number of ``Type`` instances are provided in the ``sympy.codegen.ast`` module
for convenience. Perhaps the two most common ones for code-generation (of numeric
codes) are ``float32`` and ``float64`` (known as single and double precision respectively).
There are also precision generic versions of Types (for which the codeprinters selects the
underlying data type at time of printing): ``real``, ``integer``, ``complex_``, ``bool_``.

The other ``Type`` instances defined are:

- ``intc``: Integer type used by C's "int".
- ``intp``: Integer type used by C's "unsigned".
- ``int8``, ``int16``, ``int32``, ``int64``: n-bit integers.
- ``uint8``, ``uint16``, ``uint32``, ``uint64``: n-bit unsigned integers.
- ``float80``: known as "extended precision" on modern x86/amd64 hardware.
- ``complex64``: Complex number represented by two ``float32`` numbers

"""

# 以下是需要注释的代码
from __future__ import annotations
# 导入未来版本的类型注解支持

from typing import Any
# 导入类型提示模块，用于声明函数参数和返回值的类型

from collections import defaultdict
# 导入 defaultdict，用于创建默认值为列表的字典

from sympy.core.relational import (Ge, Gt, Le, Lt)
# 导入 Sympy 中的关系运算符类：Ge（大于等于）、Gt（大于）、Le（小于等于）、Lt（小于）

from sympy.core import Symbol, Tuple, Dummy
# 导入 Sympy 中的符号、元组和虚拟变量类

from sympy.core.basic import Basic
# 导入 Sympy 中的基础类 Basic

from sympy.core.expr import Expr, Atom
# 导入 Sympy 中的表达式和原子表达式类

from sympy.core.numbers import Float, Integer, oo
# 导入 Sympy 中的浮点数、整数和无穷大常量 oo

from sympy.core.sympify import _sympify, sympify, SympifyError
# 导入 Sympy 中的符号化函数和相关异常

from sympy.utilities.iterables import (iterable, topological_sort,
                                       numbered_symbols, filter_symbols)
# 导入 Sympy 中的迭代工具函数：可迭代性判断、拓扑排序、符号编号生成器、符号过滤器

def _mk_Tuple(args):
    """
    Create a SymPy Tuple object from an iterable, converting Python strings to
    AST strings.

    Parameters
    ==========

    args: iterable
        Arguments to :class:`sympy.Tuple`.

    Returns
    =======

    sympy.Tuple
    """
    args = [String(arg) if isinstance(arg, str) else arg for arg in args]
    # 如果参数是字符串，则转换为 AST 字符串
    return Tuple(*args)
    # 返回一个 SymPy Tuple 对象，由参数 args 构成

class CodegenAST(Basic):
    __slots__ = ()
    # CodegenAST 类没有额外的实例属性，只使用基类 Basic 的属性和方法

class Token(CodegenAST):
    """ Base class for the AST types.

    Explanation
    ===========

    Defining fields are set in ``_fields``. Attributes (defined in _fields)
    are only allowed to contain instances of Basic (unless atomic, see
    ``String``). The arguments to ``__new__()`` correspond to the attributes in
    the order defined in ``_fields`. The ``defaults`` class attribute is a
    dictionary mapping attribute names to their default values.

    Subclasses should not need to override the ``__new__()`` method. They may
    """
    # Token 类是 AST 类型的基类，用于表示抽象语法树节点
    # 提供了对 AST 类型的基本说明和类属性的设置
    # 属性应仅包含 Basic 实例（除非是原子的，见 String）
    """
    define a class or static method named ``_construct_<attr>`` for each
    attribute to process the value passed to ``__new__()``. Attributes listed
    in the class attribute ``not_in_args`` are not passed to :class:`~.Basic`.
    """

    # 定义一个只包含字符串的元组，用于限制实例的动态属性
    __slots__: tuple[str, ...] = ()
    # _fields 是一个类属性，包含所有实例属性的名称
    _fields = __slots__
    # defaults 是一个类属性，存储各个属性的默认值
    defaults: dict[str, Any] = {}
    # not_in_args 是一个类属性，列出不传递给 :class:`~.Basic` 的属性列表
    not_in_args: list[str] = []
    # indented_args 是一个类属性，包含需要缩进处理的属性名称列表
    indented_args = ['body']

    @property
    def is_Atom(self):
        # 返回实例是否为原子对象的布尔值
        return len(self._fields) == 0

    @classmethod
    def _get_constructor(cls, attr):
        """ Get the constructor function for an attribute by name. """
        # 返回指定属性的构造函数，如果不存在则返回一个默认函数
        return getattr(cls, '_construct_%s' % attr, lambda x: x)

    @classmethod
    def _construct(cls, attr, arg):
        """ Construct an attribute value from argument passed to ``__new__()``. """
        # 根据传递给 ``__new__()`` 的参数构造属性值
        # 如果 arg 是 None，则返回默认值或 none
        if arg == None:
            return cls.defaults.get(attr, none)
        else:
            # 如果 arg 是 Dummy 类的实例，则直接返回
            if isinstance(arg, Dummy):  # SymPy's replace uses Dummy instances
                return arg
            else:
                # 否则根据属性名称获取对应的构造函数并应用于 arg
                return cls._get_constructor(attr)(arg)

    def __new__(cls, *args, **kwargs):
        # Pass through existing instances when given as sole argument
        # 如果作为唯一参数给出并且是现有实例，则直接返回该实例
        if len(args) == 1 and not kwargs and isinstance(args[0], cls):
            return args[0]

        # 检查传入的位置参数数量是否超过定义的属性数量
        if len(args) > len(cls._fields):
            raise ValueError("Too many arguments (%d), expected at most %d" % (len(args), len(cls._fields)))

        attrvals = []

        # 处理位置参数
        for attrname, argval in zip(cls._fields, args):
            if attrname in kwargs:
                raise TypeError('Got multiple values for attribute %r' % attrname)

            # 调用 _construct 方法构造属性值并添加到 attrvals 列表中
            attrvals.append(cls._construct(attrname, argval))

        # 处理关键字参数
        for attrname in cls._fields[len(args):]:
            if attrname in kwargs:
                argval = kwargs.pop(attrname)

            elif attrname in cls.defaults:
                argval = cls.defaults[attrname]

            else:
                raise TypeError('No value for %r given and attribute has no default' % attrname)

            # 调用 _construct 方法构造属性值并添加到 attrvals 列表中
            attrvals.append(cls._construct(attrname, argval))

        # 检查是否还有未知的关键字参数
        if kwargs:
            raise ValueError("Unknown keyword arguments: %s" % ' '.join(kwargs))

        # 调用父类的 __new__ 方法创建实例对象
        basic_args = [
            val for attr, val in zip(cls._fields, attrvals)
            if attr not in cls.not_in_args
        ]
        obj = CodegenAST.__new__(cls, *basic_args)

        # 设置实例的属性
        for attr, arg in zip(cls._fields, attrvals):
            setattr(obj, attr, arg)

        return obj

    def __eq__(self, other):
        # 检查当前实例是否等于另一个实例
        if not isinstance(other, self.__class__):
            return False
        for attr in self._fields:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True
    # 返回一个包含实例所有字段值的元组，用于哈希
    def _hashable_content(self):
        return tuple([getattr(self, attr) for attr in self._fields])

    # 哈希函数，调用父类的哈希方法
    def __hash__(self):
        return super().__hash__()

    # 返回适合打印的连接器，用于连接不同的参数
    def _joiner(self, k, indent_level):
        return (',\n' + ' '*indent_level) if k in self.indented_args else ', '

    # 根据打印器和参数进行适当的缩进，处理嵌套情况
    def _indented(self, printer, k, v, *args, **kwargs):
        il = printer._context['indent_level']
        def _print(arg):
            if isinstance(arg, Token):
                return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
            else:
                return printer._print(arg, *args, **kwargs)

        # 如果值是元组，则根据情况进行格式化和缩进处理
        if isinstance(v, Tuple):
            joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
            if k in self.indented_args:
                return '(\n' + ' '*il + joined + ',\n' + ' '*(il - 4) + ')'
            else:
                return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
        else:
            return _print(v)

    # 使用打印器和适当的格式化输出实例的字符串表示形式
    def _sympyrepr(self, printer, *args, joiner=', ', **kwargs):
        from sympy.printing.printer import printer_context
        exclude = kwargs.get('exclude', ())
        values = [getattr(self, k) for k in self._fields]
        indent_level = printer._context.get('indent_level', 0)

        arg_reprs = []

        # 遍历实例的字段及其值，根据需要排除默认值和处理缩进
        for i, (attr, value) in enumerate(zip(self._fields, values)):
            if attr in exclude:
                continue

            # 跳过具有默认值的属性
            if attr in self.defaults and value == self.defaults[attr]:
                continue

            ilvl = indent_level + 4 if attr in self.indented_args else 0
            with printer_context(printer, indent_level=ilvl):
                indented = self._indented(printer, attr, value, *args, **kwargs)
            arg_reprs.append(('{1}' if i == 0 else '{0}={1}').format(attr, indented.lstrip()))

        return "{}({})".format(self.__class__.__name__, joiner.join(arg_reprs))

    # _sympystr 是 _sympyrepr 的别名
    _sympystr = _sympyrepr

    # 返回实例的字符串表示形式，使用 sympy.printing.srepr 方法
    def __repr__(self):  # sympy.core.Basic.__repr__ uses sstr
        from sympy.printing import srepr
        return srepr(self)

    # 将实例的字段作为关键字参数的字典返回
    def kwargs(self, exclude=(), apply=None):
        """ Get instance's attributes as dict of keyword arguments.

        Parameters
        ==========

        exclude : collection of str
            Collection of keywords to exclude.

        apply : callable, optional
            Function to apply to all values.
        """
        kwargs = {k: getattr(self, k) for k in self._fields if k not in exclude}
        if apply is not None:
            return {k: apply(v) for k, v in kwargs.items()}
        else:
            return kwargs
class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """

# 创建一个表示 'break' 的类，继承自 Token 类
class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """

# 创建一个预定义的 'break' 实例，用于表示 'break' 语句
break_ = BreakToken()


class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """

# 创建一个表示 'continue' 的类，继承自 Token 类
class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """

# 创建一个预定义的 'continue' 实例，用于表示 'continue' 语句
continue_ = ContinueToken()


class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """
    def __eq__(self, other):
        return other is None or isinstance(other, NoneToken)

    def _hashable_content(self):
        return ()

    def __hash__(self):
        return super().__hash__()

# 创建一个表示 Python NoneType 的类
class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """
    # 重写相等性比较运算符，用于与 None 或 NoneToken 实例进行比较
    def __eq__(self, other):
        return other is None or isinstance(other, NoneToken)

    # 返回一个空的可哈希内容，用于支持哈希操作
    def _hashable_content(self):
        return ()

    # 重写哈希方法，保证实例可以正确地哈希化
    def __hash__(self):
        return super().__hash__()

# 创建一个预定义的 None 实例，用于表示 Python 中的 None
none = NoneToken()


class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)

        cls._check_args(lhs, rhs)

        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @classmethod


注释：
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        # 导入需要的类和模块
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement

        # 可以出现在赋值左侧的类型的元组
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable,
                ArrayElement)
        # 检查 lhs 是否属于可赋值类型之一，如果不是则抛出类型错误异常
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        # 检查 lhs 是否定义了 shape 属性且不是 Indexed 类型，用于判断是否为矩阵
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

        # 如果 lhs 和 rhs 具有相同的结构，则赋值是合法的
        if lhs_is_mat:
            # 如果 rhs 不是矩阵，则抛出值错误异常
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            # 如果 lhs 和 rhs 的形状不匹配，则抛出值错误异常
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs do not align.")
        # 如果 rhs 是矩阵且 lhs 不是，则抛出值错误异常
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")
class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation.

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.codegen.ast import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """

    op = ':='  # 设置操作符为赋值符号


class AugmentedAssignment(AssignmentBase):
    """
    Base class for augmented assignments.

    Attributes:
    ===========

    binop : str
       Symbol for binary operation being applied in the assignment, such as "+",
       "*", etc.
    """
    binop = None  # type: str  # 初始化二元操作符为空字符串

    @property
    def op(self):
        return self.binop + '='  # 返回增强赋值操作符，例如'+='、'-='等


class AddAugmentedAssignment(AugmentedAssignment):
    binop = '+'  # 设置增强赋值操作符为加法


class SubAugmentedAssignment(AugmentedAssignment):
    binop = '-'  # 设置增强赋值操作符为减法


class MulAugmentedAssignment(AugmentedAssignment):
    binop = '*'  # 设置增强赋值操作符为乘法


class DivAugmentedAssignment(AugmentedAssignment):
    binop = '/'  # 设置增强赋值操作符为除法


class ModAugmentedAssignment(AugmentedAssignment):
    binop = '%'  # 设置增强赋值操作符为取模


# Mapping from binary op strings to AugmentedAssignment subclasses
augassign_classes = {
    cls.binop: cls for cls in [
        AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
        DivAugmentedAssignment, ModAugmentedAssignment
    ]
}


def aug_assign(lhs, op, rhs):
    """
    Create 'lhs op= rhs'.

    Explanation
    ===========

    Represents augmented variable assignment for code generation. This is a
    convenience function. You can also use the AugmentedAssignment classes
    directly, like AddAugmentedAssignment(x, y).

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : str
        Operator (+, -, /, *, %).
    """
    # 创建增强赋值表达式，如 x += y
    pass  # 在这里可以添加代码来实现增强赋值的具体逻辑
    # rhs : Expr
    # 表示表达式的右侧的 SymPy 对象。可以是任何类型，只要其形状与 lhs 对应。例如，
    # Matrix 类型可以赋给 MatrixSymbol，但不能赋给 Symbol，因为维度不对应。

    # Examples
    # ========
    # 示例用法

    # >>> from sympy import symbols
    # >>> from sympy.codegen.ast import aug_assign
    # >>> x, y = symbols('x, y')
    # >>> aug_assign(x, '+', y)
    # AddAugmentedAssignment(x, y)
    """
    如果操作符 op 不在 augassign_classes 中，抛出 ValueError 异常，提示操作符未识别。
    """
    if op not in augassign_classes:
        raise ValueError("Unrecognized operator %s" % op)
    # 根据操作符 op 返回相应的增强赋值操作类，并执行赋值操作。
    return augassign_classes[op](lhs, rhs)
    """
    Represents a block of code.

    Explanation
    ===========

    For now only assignments are supported. This restriction will be lifted in
    the future.

    Useful attributes on this object are:

    ``left_hand_sides``:
        Tuple of left-hand sides of assignments, in order.
    ``left_hand_sides``:
        Tuple of right-hand sides of assignments, in order.
    ``free_symbols``: Free symbols of the expressions in the right-hand sides
        which do not appear in the left-hand side of an assignment.

    Useful methods on this object are:

    ``topological_sort``:
        Class method. Return a CodeBlock with assignments
        sorted so that variables are assigned before they
        are used.
    ``cse``:
        Return a new CodeBlock with common subexpressions eliminated and
        pulled out as assignments.

    Examples
    ========

    >>> from sympy import symbols, ccode
    >>> from sympy.codegen.ast import CodeBlock, Assignment
    >>> x, y = symbols('x y')
    >>> c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    >>> print(ccode(c))
    x = 1;
    y = x + 1;

    """
    def __new__(cls, *args):
        left_hand_sides = []
        right_hand_sides = []
        # 遍历输入的参数列表
        for i in args:
            if isinstance(i, Assignment):  # 检查参数是否为Assignment类型
                lhs, rhs = i.args  # 解构赋值表达式的左右两边
                left_hand_sides.append(lhs)  # 将左边添加到左边变量列表中
                right_hand_sides.append(rhs)  # 将右边添加到右边表达式列表中

        obj = CodegenAST.__new__(cls, *args)  # 调用父类构造函数生成对象

        obj.left_hand_sides = Tuple(*left_hand_sides)  # 设置对象的左边变量元组属性
        obj.right_hand_sides = Tuple(*right_hand_sides)  # 设置对象的右边表达式元组属性
        return obj

    def __iter__(self):
        return iter(self.args)  # 返回迭代器，以便迭代对象的参数列表

    def _sympyrepr(self, printer, *args, **kwargs):
        il = printer._context.get('indent_level', 0)  # 获取打印器的缩进级别
        joiner = ',\n' + ' '*il  # 设置连接器和缩进空格
        joined = joiner.join(map(printer._print, self.args))  # 将参数列表中的每个参数打印成字符串并连接起来
        return ('{}(\n'.format(' '*(il-4) + self.__class__.__name__,) +  # 格式化打印对象名称和左括号
                ' '*il + joined + '\n' + ' '*(il - 4) + ')')  # 打印拼接后的参数列表和右括号，保持正确的缩进格式

    _sympystr = _sympyrepr

    @property
    def free_symbols(self):
        return super().free_symbols - set(self.left_hand_sides)  # 返回自由符号集合减去左边变量集合的结果

    @classmethod
    def topological_sort(cls, assignments):
        """
        Return a CodeBlock with topologically sorted assignments so that
        variables are assigned before they are used.

        Examples
        ========

        The existing order of assignments is preserved as much as possible.

        This function assumes that variables are assigned to only once.

        This is a class constructor so that the default constructor for
        CodeBlock can error when variables are used before they are assigned.

        >>> from sympy import symbols
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> assignments = [
        ...     Assignment(x, y + z),
        ...     Assignment(y, z + 1),
        ...     Assignment(z, 2),
        ... ]
        >>> CodeBlock.topological_sort(assignments)
        CodeBlock(
            Assignment(z, 2),
            Assignment(y, z + 1),
            Assignment(x, y + z)
        )

        """

        if not all(isinstance(i, Assignment) for i in assignments):
            # 如果 assignments 中不是全部都是 Assignment 对象，则抛出异常
            raise NotImplementedError("CodeBlock.topological_sort only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in assignments):
            # 如果 assignments 中包含 AugmentedAssignment 对象，则抛出异常
            raise NotImplementedError("CodeBlock.topological_sort does not yet work with AugmentedAssignments")

        # Create a graph where the nodes are assignments and there is a directed edge
        # between nodes that use a variable and nodes that assign that
        # variable, like

        # [(x := 1, y := x + 1), (x := 1, z := y + z), (y := x + 1, z := y + z)]

        # If we then topologically sort these nodes, they will be in
        # assignment order, like

        # x := 1
        # y := x + 1
        # z := y + z

        # A = The nodes
        #
        # enumerate keeps nodes in the same order they are already in if
        # possible. It will also allow us to handle duplicate assignments to
        # the same variable when those are implemented.
        A = list(enumerate(assignments))

        # var_map = {variable: [nodes for which this variable is assigned to]}
        # like {x: [(1, x := y + z), (4, x := 2 * w)], ...}
        var_map = defaultdict(list)
        for node in A:
            i, a = node
            # 将每个变量的赋值节点添加到 var_map 中
            var_map[a.lhs].append(node)

        # E = Edges in the graph
        E = []
        for dst_node in A:
            i, a = dst_node
            for s in a.rhs.free_symbols:
                # 找到每个节点的自由符号（被使用的变量），并建立边
                for src_node in var_map[s]:
                    E.append((src_node, dst_node))

        # ordered_assignments 是经过拓扑排序后的赋值节点
        ordered_assignments = topological_sort([A, E])

        # De-enumerate the result
        # 将排序后的节点去除枚举，返回拓扑排序后的赋值节点列表
        return cls(*[a for i, a in ordered_assignments])
    def cse(self, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
        """
        Return a new code block with common subexpressions eliminated.

        Explanation
        ===========

        See the docstring of :func:`sympy.simplify.cse_main.cse` for more
        information.

        Examples
        ========

        >>> from sympy import symbols, sin
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> c = CodeBlock(
        ...     Assignment(x, 1),
        ...     Assignment(y, sin(x) + 1),
        ...     Assignment(z, sin(x) - 1),
        ... )
        ...
        >>> c.cse()
        CodeBlock(
            Assignment(x, 1),
            Assignment(x0, sin(x)),
            Assignment(y, x0 + 1),
            Assignment(z, x0 - 1)
        )

        """
        from sympy.simplify.cse_main import cse

        # 检查代码块是否仅包含对唯一变量的赋值操作
        if not all(isinstance(i, Assignment) for i in self.args):
            # 后续可能支持更多类型
            raise NotImplementedError("CodeBlock.cse only supports Assignments")

        # 如果存在增强赋值语句，暂时不支持
        if any(isinstance(i, AugmentedAssignment) for i in self.args):
            raise NotImplementedError("CodeBlock.cse does not yet work with AugmentedAssignments")

        # 检查是否存在重复赋值到同一变量的情况
        for i, lhs in enumerate(self.left_hand_sides):
            if lhs in self.left_hand_sides[:i]:
                raise NotImplementedError("Duplicate assignments to the same "
                    "variable are not yet supported (%s)" % lhs)

        # 确保子表达式的新符号不与现有符号冲突
        existing_symbols = self.atoms(Symbol)
        if symbols is None:
            symbols = numbered_symbols()
        symbols = filter_symbols(symbols, existing_symbols)

        # 调用公共子表达式消除函数，获取替换和简化后的表达式列表
        replacements, reduced_exprs = cse(list(self.right_hand_sides),
            symbols=symbols, optimizations=optimizations, postprocess=postprocess,
            order=order)

        # 构造新的代码块，包括替换后的赋值语句和简化后的表达式
        new_block = [Assignment(var, expr) for var, expr in
            zip(self.left_hand_sides, reduced_exprs)]
        new_assignments = [Assignment(var, expr) for var, expr in replacements]
        
        # 对新的赋值语句和表达式进行拓扑排序后返回
        return self.topological_sort(new_assignments + new_block)
class For(Token):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ==========

    target : symbol
        The symbol representing the loop variable.
    iter : iterable
        The iterable object over which the loop iterates.
    body : CodeBlock or iterable
        The body of the loop, represented either as a CodeBlock or an iterable of statements.

    Examples
    ========

    >>> from sympy import symbols, Range
    >>> from sympy.codegen.ast import aug_assign, For
    >>> x, i, j, k = symbols('x i j k')
    >>> for_i = For(i, Range(10), [aug_assign(x, '+', i*j*k)])
    >>> for_i  # doctest: -NORMALIZE_WHITESPACE
    For(i, iterable=Range(0, 10, 1), body=CodeBlock(
        AddAugmentedAssignment(x, i*j*k)
    ))
    >>> for_ji = For(j, Range(7), [for_i])
    >>> for_ji  # doctest: -NORMALIZE_WHITESPACE
    For(j, iterable=Range(0, 7, 1), body=CodeBlock(
        For(i, iterable=Range(0, 10, 1), body=CodeBlock(
            AddAugmentedAssignment(x, i*j*k)
        ))
    ))
    >>> for_kji = For(k, Range(5), [for_ji])
    >>> for_kji  # doctest: -NORMALIZE_WHITESPACE
    For(k, iterable=Range(0, 5, 1), body=CodeBlock(
        For(j, iterable=Range(0, 7, 1), body=CodeBlock(
            For(i, iterable=Range(0, 10, 1), body=CodeBlock(
                AddAugmentedAssignment(x, i*j*k)
            ))
        ))
    ))
    """
    __slots__ = _fields = ('target', 'iterable', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        """Constructs the body of the for-loop.

        Parameters
        ==========
        itr : CodeBlock or iterable
            The iterable or CodeBlock representing the body of the loop.

        Returns
        =======
        CodeBlock
            The constructed CodeBlock object.

        Notes
        =====
        When passed an iterable, it is used to instantiate a CodeBlock.
        """
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_iterable(cls, itr):
        """Constructs the iterable object for the for-loop.

        Parameters
        ==========
        itr : iterable
            The iterable object over which the loop iterates.

        Returns
        =======
        object
            The constructed iterable object.

        Raises
        ======
        TypeError
            If itr is not an iterable.

        Notes
        =====
        - Raises TypeError if the iterable is not iterable.
        - Converts lists to tuples to avoid mutability issues with _sympify.
        """
        if not iterable(itr):
            raise TypeError("iterable must be an iterable")
        if isinstance(itr, list):  # _sympify errors on lists because they are mutable
            itr = tuple(itr)
        return _sympify(itr)


class String(Atom, Token):
    """ SymPy object representing a string.

    Atomic object which is not an expression (as opposed to Symbol).

    Parameters
    ==========

    text : str
        The string content represented by this object.

    Examples
    ========

    >>> from sympy.codegen.ast import String
    >>> f = String('foo')
    >>> f
    foo
    >>> str(f)
    'foo'
    >>> f.text
    'foo'
    >>> print(repr(f))
    String('foo')

    """
    __slots__ = _fields = ('text',)
    not_in_args = ['text']
    is_Atom = True

    @classmethod
    def _construct_text(cls, text):
        """Constructs the text attribute of the String object.

        Parameters
        ==========
        text : str
            The string content to be assigned.

        Returns
        =======
        str
            The constructed string.

        Raises
        ======
        TypeError
            If text is not of type str.

        Notes
        =====
        Raises TypeError if the argument is not a string type.
        """
        if not isinstance(text, str):
            raise TypeError("Argument text is not a string type.")
        return text

    def _sympystr(self, printer, *args, **kwargs):
        """Returns a string representation of the String object.

        Parameters
        ==========
        printer : object
            The printer object responsible for string conversion.
        args : tuple
            Additional positional arguments for formatting.
        kwargs : dict
            Additional keyword arguments for formatting.

        Returns
        =======
        str
            The string representation of the String object.
        """
        return self.text

    def kwargs(self, exclude = (), apply = None):
        """Returns keyword arguments for the String object.

        Parameters
        ==========
        exclude : tuple, optional
            Arguments to exclude from the keyword dictionary.
        apply : object, optional
            Function or object to apply to the keyword arguments.

        Returns
        =======
        dict
            The dictionary of keyword arguments for the String object.
        """
        return {}

    #to be removed when Atom is given a suitable func
    @property
    def func(self):
        """Returns a lambda function for the String object."""
        return lambda: self

    def _latex(self, printer):
        """Returns a LaTeX representation of the String object.

        Parameters
        ==========
        printer : object
            The printer object responsible for LaTeX conversion.

        Returns
        =======
        str
            The LaTeX representation of the String object.
        """
        from sympy.printing.latex import latex_escape
        return r'\texttt{{"{}"}}'.format(latex_escape(self.text))

class QuotedString(String):
    """SymPy object representing a quoted string.

    Inherits from String and represents a string enclosed in quotes.

    This class inherits all attributes and methods from String.

    """
    # 定义一个类，表示带引号打印的字符串
    class QuotedString(str):
        # 覆盖字符串类的 __repr__() 方法，返回带引号的字符串表示
        def __repr__(self):
            return f'"{self}"'
class Comment(String):
    """ Represents a comment. """

class Node(Token):
    """ Subclass of Token, carrying the attribute 'attrs' (Tuple)

    Examples
    ========

    >>> from sympy.codegen.ast import Node, value_const, pointer_const
    >>> n1 = Node([value_const])
    >>> n1.attr_params('value_const')  # get the parameters of attribute (by name)
    ()
    >>> from sympy.codegen.fnodes import dimension
    >>> n2 = Node([value_const, dimension(5, 3)])
    >>> n2.attr_params(value_const)  # get the parameters of attribute (by Attribute instance)
    ()
    >>> n2.attr_params('dimension')  # get the parameters of attribute (by name)
    (5, 3)
    >>> n2.attr_params(pointer_const) is None
    True

    """

    __slots__: tuple[str, ...] = ('attrs',)
    _fields = __slots__

    defaults: dict[str, Any] = {'attrs': Tuple()}

    _construct_attrs = staticmethod(_mk_Tuple)

    def attr_params(self, looking_for):
        """ Returns the parameters of the Attribute with name ``looking_for`` in self.attrs """
        for attr in self.attrs:
            if str(attr.name) == str(looking_for):
                return attr.parameters


class Type(Token):
    """ Represents a type.

    Explanation
    ===========

    The naming is a super-set of NumPy naming. Type has a classmethod
    ``from_expr`` which offer type deduction. It also has a method
    ``cast_check`` which casts the argument to its type, possibly raising an
    exception if rounding error is not within tolerances, or if the value is not
    representable by the underlying data type (e.g. unsigned integers).

    Parameters
    ==========

    name : str
        Name of the type, e.g. ``object``, ``int16``, ``float16`` (where the latter two
        would use the ``Type`` sub-classes ``IntType`` and ``FloatType`` respectively).
        If a ``Type`` instance is given, the said instance is returned.

    Examples
    ========

    >>> from sympy.codegen.ast import Type
    >>> t = Type.from_expr(42)
    >>> t
    integer
    >>> print(repr(t))
    IntBaseType(String('integer'))
    >>> from sympy.codegen.ast import uint8
    >>> uint8.cast_check(-1)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Minimum value for data type bigger than new value.
    >>> from sympy.codegen.ast import float32
    >>> v6 = 0.123456
    >>> float32.cast_check(v6)
    0.123456
    >>> v10 = 12345.67894
    >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Casting gives a significantly different value.
    >>> boost_mp50 = Type('boost::multiprecision::cpp_dec_float_50')
    >>> from sympy import cxxcode
    >>> from sympy.codegen.ast import Declaration, Variable
    >>> cxxcode(Declaration(Variable('x', type=boost_mp50)))
    'boost::multiprecision::cpp_dec_float_50 x'

    References
    ==========

    .. [1] https://numpy.org/doc/stable/user/basics.types.html

    """
    # 定义只允许包含'name'属性的类插槽
    __slots__: tuple[str, ...] = ('name',)
    # 将插槽作为类的字段
    _fields = __slots__
    
    # 定义类属性_construct_name为字符串类型
    _construct_name = String
    
    # 定义一个私有方法，用于返回对象名称的字符串表示
    def _sympystr(self, printer, *args, **kwargs):
        return str(self.name)
    
    # 定义一个类方法，根据表达式或符号推断类型
    @classmethod
    def from_expr(cls, expr):
        """ Deduces type from an expression or a ``Symbol``.
    
        Parameters
        ==========
    
        expr : number or SymPy object
            The type will be deduced from type or properties.
    
        Examples
        ========
    
        >>> from sympy.codegen.ast import Type, integer, complex_
        >>> Type.from_expr(2) == integer
        True
        >>> from sympy import Symbol
        >>> Type.from_expr(Symbol('z', complex=True)) == complex_
        True
        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Could not deduce type from expr.
    
        Raises
        ======
    
        ValueError when type deduction fails.
    
        """
        # 根据表达式的类型或属性推断返回对应的类型对象
        if isinstance(expr, (float, Float)):
            return real
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return integer
        if getattr(expr, 'is_real', False):
            return real
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return complex_
        if isinstance(expr, bool) or getattr(expr, 'is_Relational', False):
            return bool_
        else:
            raise ValueError("Could not deduce type from expr.")
    
    # 定义一个私有方法，用于检查值，但实际上该方法没有实现内容
    def _check(self, value):
        pass
    def cast_check(self, value, rtol=None, atol=0, precision_targets=None):
        """ 
        Casts a value to the data type of the instance.

        Parameters
        ==========

        value : number
            The value to be casted to the data type of the instance.
        rtol : floating point number
            Relative tolerance. If not provided, it will be deduced based on `exp10`.
        atol : floating point number
            Absolute tolerance added to `rtol` for tolerance comparison.
        precision_targets : dict
            Dictionary mapping type aliases for substitutions, e.g., {integer: int64, real: float32}.

        Examples
        ========

        >>> from sympy.codegen.ast import integer, float32, int8
        >>> integer.cast_check(3.0) == 3
        True
        >>> float32.cast_check(1e-40)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Minimum value for data type bigger than new value.
        >>> int8.cast_check(256)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Maximum value for data type smaller than new value.
        >>> v10 = 12345.67894
        >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float64
        >>> float64.cast_check(v10)
        12345.67894
        >>> from sympy import Float
        >>> v18 = Float('0.123456789012345646')
        >>> float64.cast_check(v18)
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float80
        >>> float80.cast_check(v18)
        0.123456789012345649

        """
        val = sympify(value)  # Convert `value` to a SymPy expression if it isn't already.

        ten = Integer(10)
        exp10 = getattr(self, 'decimal_dig', None)  # Get the number of decimal digits from instance attributes.

        if rtol is None:
            rtol = 1e-15 if exp10 is None else 2.0 * ten**(-exp10)  # Set relative tolerance if not provided based on `exp10`.

        def tol(num):
            return atol + rtol * abs(num)  # Compute tolerance based on `atol` and `rtol`.

        new_val = self.cast_nocheck(value)  # Cast `value` to the instance's data type without checking.
        self._check(new_val)  # Check if the new value is valid for the data type.

        delta = new_val - val  # Calculate the difference between the original value and the new value.
        if abs(delta) > tol(val):  # Check if the difference exceeds the tolerance.
            raise ValueError("Casting gives a significantly different value.")  # Raise an error if the difference is significant.

        return new_val  # Return the casted value.

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        type_name = latex_escape(self.__class__.__name__)  # Get the LaTeX-escaped class name of the instance.
        name = latex_escape(self.name.text)  # Get the LaTeX-escaped name attribute of the instance.
        return r"\text{{{}}}\left(\texttt{{{}}}\right)".format(type_name, name)  # Return a LaTeX representation of the instance.
class IntBaseType(Type):
    """ Integer base type, contains no size information. """
    __slots__ = ()
    # 无大小信息的整数基础类型

    # Lambda函数，将输入强制转换为整数
    cast_nocheck = lambda self, i: Integer(int(i))


class _SizedIntType(IntBaseType):
    """ Integer type with size information. """
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__

    # 构造函数，用于设定整数类型的大小
    _construct_nbits = Integer

    def _check(self, value):
        """ Check if the value fits within the specified range. """
        # 检查值是否在指定范围内
        if value < self.min:
            raise ValueError("Value is too small: %d < %d" % (value, self.min))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))


class SignedIntType(_SizedIntType):
    """ Represents a signed integer type. """
    __slots__ = ()
    
    @property
    def min(self):
        """ Minimum value representable by this signed integer type. """
        # 此有符号整数类型能表示的最小值
        return -2**(self.nbits-1)

    @property
    def max(self):
        """ Maximum value representable by this signed integer type. """
        # 此有符号整数类型能表示的最大值
        return 2**(self.nbits-1) - 1


class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()
    
    @property
    def min(self):
        """ Minimum value representable by this unsigned integer type. """
        # 此无符号整数类型能表示的最小值
        return 0

    @property
    def max(self):
        """ Maximum value representable by this unsigned integer type. """
        # 此无符号整数类型能表示的最大值
        return 2**self.nbits - 1

two = Integer(2)

class FloatBaseType(Type):
    """ Represents a floating point number type. """
    __slots__ = ()
    # 浮点数类型

    # Lambda函数，将输入强制转换为浮点数
    cast_nocheck = Float

class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the exponent.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """

    __slots__ = ('nbits', 'nmant', 'nexp',)
    _fields = Type._fields + __slots__

    # 构造函数，用于设定浮点数类型的大小、尾数和指数
    _construct_nbits = _construct_nmant = _construct_nexp = Integer

    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        # 最大的正指数，使得 2**(n - 1) 是可表示的有限值
        return two**(self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        # 最小的负指数，使得 2**(n - 1) 是一个有效的归一化数
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        # 可表示的最大值
        return (1 - two**-(self.nmant+1))*two**self.max_exponent

    @property
    def min(self):
        """ Minimum value representable. """
        # 可表示的最小值

        return -self.max


注释：


        """ Minimum value representable. """
        # 可表示的最小值
        return -self.max


这样，每一行代码都有了相应的注释解释其作用。
    def tiny(self):
        """ The minimum positive normalized value. """
        # 返回一个值，这个值是2的(self.min_exponent - 1)次方
        return two**(self.min_exponent - 1)


    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        # 返回一个值，这个值是2的(-self.nmant)次方
        return two**(-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        # 导入必要的函数并返回一个整数值
        from sympy.functions import floor, log
        return floor(self.nmant * log(2)/log(10))

    @property
    def decimal_dig(self):
        """ Number of digits needed to store & load without loss.

        Explanation
        ===========

        Number of decimal digits needed to guarantee that two consecutive conversions
        (float -> text -> float) to be idempotent. This is useful when one do not want
        to loose precision due to rounding errors when storing a floating point value
        as text.
        """
        # 导入必要的函数并返回一个整数值
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2)/log(10) + 1)

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        # 如果值等于正无穷大，则返回正无穷大的浮点表示
        if value == oo:  # float(oo) or oo
            return float(oo)
        # 如果值等于负无穷大，则返回负无穷大的浮点表示
        elif value == -oo:  # float(-oo) or -oo
            return float(-oo)
        # 将输入值转换成符号表达式，计算其浮点值，并返回指定精度的浮点数表示
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        # 如果值小于-self.max，则抛出值过小的异常
        if value < -self.max:
            raise ValueError("Value is too small: %d < %d" % (value, -self.max))
        # 如果值大于self.max，则抛出值过大的异常
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))
        # 如果值的绝对值小于self.tiny，则抛出最小值异常
        if abs(value) < self.tiny:
            raise ValueError("Smallest (absolute) value for data type bigger than new value.")
class ComplexBaseType(FloatBaseType):
    """ A complex number base type inheriting from FloatBaseType. """

    __slots__ = ()  # Empty slot tuple for optimization

    def cast_nocheck(self, value):
        """ Casts 'value' to complex without checking bounds or subnormality. """
        from sympy.functions import re, im
        return (
            super().cast_nocheck(re(value)) +  # Cast real part
            super().cast_nocheck(im(value)) * 1j  # Cast imaginary part
        )

    def _check(self, value):
        """ Checks the real and imaginary parts of 'value'. """
        from sympy.functions import re, im
        super()._check(re(value))  # Check real part
        super()._check(im(value))  # Check imaginary part


class ComplexType(ComplexBaseType, FloatType):
    """ Represents a complex floating point number. """
    __slots__ = ()  # Empty slot tuple for optimization


# NumPy types:
intc = IntBaseType('intc')  # 32-bit integer type
intp = IntBaseType('intp')  # Integer type used for indexing
int8 = SignedIntType('int8', 8)  # Signed 8-bit integer
int16 = SignedIntType('int16', 16)  # Signed 16-bit integer
int32 = SignedIntType('int32', 32)  # Signed 32-bit integer
int64 = SignedIntType('int64', 64)  # Signed 64-bit integer
uint8 = UnsignedIntType('uint8', 8)  # Unsigned 8-bit integer
uint16 = UnsignedIntType('uint16', 16)  # Unsigned 16-bit integer
uint32 = UnsignedIntType('uint32', 32)  # Unsigned 32-bit integer
uint64 = UnsignedIntType('uint64', 64)  # Unsigned 64-bit integer
float16 = FloatType('float16', 16, nexp=5, nmant=10)  # IEEE 754 binary16, Half precision
float32 = FloatType('float32', 32, nexp=8, nmant=23)  # IEEE 754 binary32, Single precision
float64 = FloatType('float64', 64, nexp=11, nmant=52)  # IEEE 754 binary64, Double precision
float80 = FloatType('float80', 80, nexp=15, nmant=63)  # x86 extended precision, "long double"
float128 = FloatType('float128', 128, nexp=15, nmant=112)  # IEEE 754 binary128, Quadruple precision
float256 = FloatType('float256', 256, nexp=19, nmant=236)  # IEEE 754 binary256, Octuple precision

complex64 = ComplexType('complex64', nbits=64, **float32.kwargs(exclude=('name', 'nbits')))
complex128 = ComplexType('complex128', nbits=128, **float64.kwargs(exclude=('name', 'nbits')))

# Generic types (precision may be chosen by code printers):
untyped = Type('untyped')  # Untyped generic type
real = FloatBaseType('real')  # Real number base type
integer = IntBaseType('integer')  # Integer base type
complex_ = ComplexBaseType('complex')  # Complex number base type
bool_ = Type('bool')  # Boolean type


class Attribute(Token):
    """ Represents an attribute, possibly parametrized.

    Used with :class:`sympy.codegen.ast.Node` (instances of ``Attribute``
    are used as ``attrs``).

    Parameters
    ==========

    name : str
        The name of the attribute.
    parameters : Tuple
        Parameters associated with the attribute.

    Examples
    ========

    >>> from sympy.codegen.ast import Attribute
    >>> volatile = Attribute('volatile')
    >>> volatile
    volatile
    >>> print(repr(volatile))
    Attribute(String('volatile'))
    >>> a = Attribute('foo', [1, 2, 3])
    >>> a
    foo(1, 2, 3)
    >>> a.parameters == (1, 2, 3)
    True
    """
    __slots__ = _fields = ('name', 'parameters')
    defaults = {'parameters': Tuple()}  # Default parameters as an empty tuple

    _construct_name = String  # Name constructor as a String
    _construct_parameters = staticmethod(_mk_Tuple)  # Parameters constructor as a tuple maker

    def _sympystr(self, printer, *args, **kwargs):
        """ String representation of the Attribute. """
        result = str(self.name)
        if self.parameters:
            result += '(%s)' % ', '.join((printer._print(
                arg, *args, **kwargs) for arg in self.parameters))
        return result


value_const = Attribute('value_const')  # Attribute representing a constant value
pointer_const = Attribute('pointer_const')

# 创建一个名为 `pointer_const` 的 `Attribute` 实例，并将其赋值给 `pointer_const` 变量。


class Variable(Node):
    """ Represents a variable.

    Parameters
    ==========

    symbol : Symbol
        变量的符号表示。
    type : Type (optional)
        变量的类型，可选参数。
    attrs : iterable of Attribute instances
        作为 Tuple 存储的属性实例列表。

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, float32, integer
    >>> x = Symbol('x')
    >>> v = Variable(x, type=float32)
    >>> v.attrs
    ()
    >>> v == Variable('x')
    False
    >>> v == Variable('x', type=float32)
    True
    >>> v
    Variable(x, type=float32)

    One may also construct a ``Variable`` instance with the type deduced from
    assumptions about the symbol using the ``deduced`` classmethod:

    >>> i = Symbol('i', integer=True)
    >>> v = Variable.deduced(i)
    >>> v.type == integer
    True
    >>> v == Variable('i')
    False
    >>> from sympy.codegen.ast import value_const
    >>> value_const in v.attrs
    False
    >>> w = Variable('w', attrs=[value_const])
    >>> w
    Variable(w, attrs=(value_const,))
    >>> value_const in w.attrs
    True
    >>> w.as_Declaration(value=42)
    Declaration(Variable(w, value=42, attrs=(value_const,)))

    """

    __slots__ = ('symbol', 'type', 'value')
    _fields = __slots__ + Node._fields

    defaults = Node.defaults.copy()
    defaults.update({'type': untyped, 'value': none})

    _construct_symbol = staticmethod(sympify)
    _construct_value = staticmethod(sympify)

# 定义了一个名为 `Variable` 的类，表示一个变量，具有符号、类型和属性等特征。包括多种构造方法和示例，以及静态方法和类属性的定义。


@classmethod
def deduced(cls, symbol, value=None, attrs=Tuple(), cast_check=True):
    """ Alt. constructor with type deduction from ``Type.from_expr``.

    Deduces type primarily from ``symbol``, secondarily from ``value``.

    Parameters
    ==========

    symbol : Symbol
        表示变量的符号。
    value : expr
        （可选）变量的值。
    attrs : iterable of Attribute instances
        属性实例的可迭代对象。
    cast_check : bool
        是否对值应用 `Type.cast_check`。

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, complex_
    >>> n = Symbol('n', integer=True)
    >>> str(Variable.deduced(n).type)
    'integer'
    >>> x = Symbol('x', real=True)
    >>> v = Variable.deduced(x)
    >>> v.type
    real
    >>> z = Symbol('z', complex=True)
    >>> Variable.deduced(z).type == complex_
    True

    """
    # 如果 symbol 已经是 Variable 的实例，则直接返回 symbol
    if isinstance(symbol, Variable):
        return symbol

    # 尝试从 symbol 推断类型，如果失败，则从 value 推断类型
    try:
        type_ = Type.from_expr(symbol)
    except ValueError:
        type_ = Type.from_expr(value)

    # 如果 value 不为 None 并且 cast_check 为真，则对 value 进行类型检查
    if value is not None and cast_check:
        value = type_.cast_check(value)

    # 返回一个新的 Variable 实例，包括 symbol、type、value 和 attrs
    return cls(symbol, type=type_, value=value, attrs=attrs)
    # Convenience method for creating a Declaration instance from the current object.
    def as_Declaration(self, **kwargs):
        """ Convenience method for creating a Declaration instance.

        Explanation
        ===========

        If the variable of the Declaration needs to wrap a modified
        variable, keyword arguments may be passed (overriding e.g.
        the ``value`` of the Variable instance).

        Examples
        ========

        >>> from sympy.codegen.ast import Variable, NoneToken
        >>> x = Variable('x')
        >>> decl1 = x.as_Declaration()
        >>> # value is special NoneToken() which must be tested with == operator
        >>> decl1.variable.value is None  # won't work
        False
        >>> decl1.variable.value == None  # not PEP-8 compliant
        True
        >>> decl1.variable.value == NoneToken()  # OK
        True
        >>> decl2 = x.as_Declaration(value=42.0)
        >>> decl2.variable.value == 42.0
        True

        """
        # Retrieve keyword arguments and update with any additional kwargs passed
        kw = self.kwargs()
        kw.update(kwargs)
        # Create a Declaration instance using updated keyword arguments
        return Declaration(self.func(**kw))

    # Private method for handling relational operations with another object.
    def _relation(self, rhs, op):
        try:
            # Attempt to convert rhs to a sympy expression
            rhs = _sympify(rhs)
        except SympifyError:
            # Raise an error if conversion fails
            raise TypeError("Invalid comparison %s < %s" % (self, rhs))
        # Perform the specified comparison operation
        return op(self, rhs, evaluate=False)

    # Define less than (__lt__) operator for the current object
    __lt__ = lambda self, other: self._relation(other, Lt)
    # Define less than or equal (__le__) operator for the current object
    __le__ = lambda self, other: self._relation(other, Le)
    # Define greater than or equal (__ge__) operator for the current object
    __ge__ = lambda self, other: self._relation(other, Ge)
    # Define greater than (__gt__) operator for the current object
    __gt__ = lambda self, other: self._relation(other, Gt)
class Pointer(Variable):
    """ Represents a pointer. See ``Variable``.

    Examples
    ========

    Can create instances of ``Element``:

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Pointer
    >>> i = Symbol('i', integer=True)
    >>> p = Pointer('x')
    >>> p[i+1]
    Element(x, indices=(i + 1,))

    """
    __slots__ = ()

    def __getitem__(self, key):
        # 尝试创建一个 Element 实例，使用给定的符号和键
        try:
            return Element(self.symbol, key)
        except TypeError:
            # 如果出现 TypeError，将键作为元组传递给 Element 实例
            return Element(self.symbol, (key,))


class Element(Token):
    """ Element in (a possibly N-dimensional) array.

    Examples
    ========

    >>> from sympy.codegen.ast import Element
    >>> elem = Element('x', 'ijk')
    >>> elem.symbol.name == 'x'
    True
    >>> elem.indices
    (i, j, k)
    >>> from sympy import ccode
    >>> ccode(elem)
    'x[i][j][k]'
    >>> ccode(Element('x', 'ijk', strides='lmn', offset='o'))
    'x[i*l + j*m + k*n + o]'

    """
    __slots__ = _fields = ('symbol', 'indices', 'strides', 'offset')
    defaults = {'strides': none, 'offset': none}
    _construct_symbol = staticmethod(sympify)
    _construct_indices = staticmethod(lambda arg: Tuple(*arg))
    _construct_strides = staticmethod(lambda arg: Tuple(*arg))
    _construct_offset = staticmethod(sympify)


class Declaration(Token):
    """ Represents a variable declaration

    Parameters
    ==========

    variable : Variable

    Examples
    ========

    >>> from sympy.codegen.ast import Declaration, NoneToken, untyped
    >>> z = Declaration('z')
    >>> z.variable.type == untyped
    True
    >>> # value is special NoneToken() which must be tested with == operator
    >>> z.variable.value is None  # won't work
    False
    >>> z.variable.value == None  # not PEP-8 compliant
    True
    >>> z.variable.value == NoneToken()  # OK
    True
    """
    __slots__ = _fields = ('variable',)
    _construct_variable = Variable


class While(Token):
    """ Represents a 'for-loop' in the code.

    Expressions are of the form:
        "while condition:
             body..."

    Parameters
    ==========

    condition : expression convertible to Boolean
    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Gt, Abs
    >>> from sympy.codegen import aug_assign, Assignment, While
    >>> x, dx = symbols('x dx')
    >>> expr = 1 - x**2
    >>> whl = While(Gt(Abs(dx), 1e-9), [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx)
    ... ])

    """
    __slots__ = _fields = ('condition', 'body')
    _construct_condition = staticmethod(lambda cond: _sympify(cond))

    @classmethod
    def _construct_body(cls, itr):
        # 如果传入的 itr 是 CodeBlock 的实例，则直接返回
        if isinstance(itr, CodeBlock):
            return itr
        else:
            # 否则，使用 itr 中的元素创建一个 CodeBlock 实例并返回
            return CodeBlock(*itr)


class Scope(Token):
    """ Represents a scope in the code.

    Parameters
    ==========

    """
    # body 是一个属性，可以是 CodeBlock 类型或者可迭代对象
    body : CodeBlock or iterable
        当传入一个可迭代对象时，它会用来实例化一个 CodeBlock 对象。

    """
    __slots__ = _fields = ('body',)

    # 定义只允许存在 body 属性的类，以优化内存使用和访问速度
    @classmethod
    def _construct_body(cls, itr):
        # 如果传入的 itr 已经是 CodeBlock 类型，则直接返回它
        if isinstance(itr, CodeBlock):
            return itr
        else:
            # 否则，用传入的可迭代对象 itr 来构造一个新的 CodeBlock 对象
            return CodeBlock(*itr)
class Stream(Token):
    """ Represents a stream.

    There are two predefined Stream instances ``stdout`` & ``stderr``.

    Parameters
    ==========

    name : str
        The name of the stream.

    Examples
    ========

    >>> from sympy import pycode, Symbol
    >>> from sympy.codegen.ast import Print, stderr, QuotedString
    >>> print(pycode(Print(['x'], file=stderr)))
    print(x, file=sys.stderr)
    >>> x = Symbol('x')
    >>> print(pycode(Print([QuotedString('x')], file=stderr)))  # print literally "x"
    print("x", file=sys.stderr)

    """
    __slots__ = _fields = ('name',)
    _construct_name = String

stdout = Stream('stdout')  # 创建名为 'stdout' 的 Stream 实例
stderr = Stream('stderr')  # 创建名为 'stderr' 的 Stream 实例


class Print(Token):
    r""" Represents print command in the code.

    Parameters
    ==========

    formatstring : str
        The format string for the print statement.
    *args : Basic instances (or convertible to such through sympify)
        The arguments to be printed.

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy import pycode
    >>> print(pycode(Print('x y'.split(), "coordinate: %12.5g %12.5g\\n")))
    print("coordinate: %12.5g %12.5g\n" % (x, y), end="")

    """
    __slots__ = _fields = ('print_args', 'format_string', 'file')
    defaults = {'format_string': none, 'file': none}

    _construct_print_args = staticmethod(_mk_Tuple)
    _construct_format_string = QuotedString
    _construct_file = Stream


class FunctionPrototype(Node):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
        The return type of the function.
    name : str
        The name of the function.
    parameters: iterable of Variable instances
        The parameters of the function.
    attrs : iterable of Attribute instances
        Additional attributes of the function.

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """
    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields

    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError("func_def is not an instance of FunctionDefinition")
        return cls(**func_def.kwargs(exclude=('body',)))


class FunctionDefinition(FunctionPrototype):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
        The return type of the function.
    name : str
        The name of the function.
    parameters: iterable of Variable instances
        The parameters of the function.
    body : CodeBlock or iterable
        The body of the function.
    attrs : iterable of Attribute instances
        Additional attributes of the function.

    """
    __slots__ = ('body', )
    _slots = ('body', )  # 定义一个只允许包含 'body' 属性的特殊类属性 __slots__
    
    _fields = FunctionPrototype._fields[:-1] + __slots__ + Node._fields
    # 从 FunctionPrototype._fields 和 Node._fields 构建 _fields，除了最后一个元素外，添加 __slots__ 中的元素
    
    @classmethod
    def _construct_body(cls, itr):
        # 定义一个类方法 _construct_body，用于根据输入构建代码块对象
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)
    
    @classmethod
    def from_FunctionPrototype(cls, func_proto, body):
        # 定义一个类方法 from_FunctionPrototype，从 FunctionPrototype 对象创建 FunctionDefinition 对象
        if not isinstance(func_proto, FunctionPrototype):
            raise TypeError("func_proto is not an instance of FunctionPrototype")
        return cls(body=body, **func_proto.kwargs())
        # 返回一个新的 FunctionDefinition 实例，传入参数 body 和从 func_proto 获取的其他关键字参数
# 表示代码中的返回命令的类
class Return(Token):
    """ Represents a return command in the code.

    Parameters
    ==========

    return : Basic  # 表示返回的基本内容，可以是表达式或变量

    Examples
    ========

    >>> from sympy.codegen.ast import Return
    >>> from sympy.printing.pycode import pycode
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> print(pycode(Return(x)))
    return x  # 打印为 return x 的代码

    """
    __slots__ = _fields = ('return',)  # 定义类的特殊属性，表示仅有 'return' 这一个属性
    _construct_return = staticmethod(_sympify)  # 用于构造 return 属性的静态方法


# 表示代码中函数调用的类
class FunctionCall(Token, Expr):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str  # 函数名
    function_args : Tuple  # 函数的参数列表，以元组形式给出

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)  # 打印为 foo(bar, baz) 的代码

    """
    __slots__ = _fields = ('name', 'function_args')  # 定义类的特殊属性，包括函数名和参数列表

    _construct_name = String  # 用于构造函数名的属性，期望是一个字符串
    _construct_function_args = staticmethod(lambda args: Tuple(*args))  # 构造函数参数的静态方法


# 表示在代码中抛出异常的类
class Raise(Token):
    """ Prints as 'raise ...' in Python, 'throw ...' in C++"""
    __slots__ = _fields = ('exception',)  # 定义类的特殊属性，仅有异常信息

    # 没有额外的构造方法或属性


# 表示在代码中表示运行时错误的类
class RuntimeError_(Token):
    """ Represents 'std::runtime_error' in C++ and 'RuntimeError' in Python.

    Note that the latter is uncommon, and you might want to use e.g. ValueError.
    """
    __slots__ = _fields = ('message',)  # 定义类的特殊属性，仅有错误消息

    _construct_message = String  # 用于构造错误消息的属性
```