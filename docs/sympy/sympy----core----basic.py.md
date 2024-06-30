# `D:\src\scipysrc\sympy\sympy\core\basic.py`

```
# 导入所有 SymPy 中对象的基类
"""Base class for all the objects in SymPy"""
# 引入在将来版本中可能会用到的注解
from __future__ import annotations

# 引入默认字典和映射类
from collections import defaultdict
from collections.abc import Mapping
# 引入链和zip_longest函数
from itertools import chain, zip_longest
# 引入比较函数
from functools import cmp_to_key

# 引入 SymPy 内部模块
from .assumptions import _prepare_class_assumptions
# 缓存装饰器
from .cache import cacheit
# 引入符号化函数
from .sympify import _sympify, sympify, SympifyError, _external_converter
# 引入排序函数
from .sorting import ordered
# 引入类型定义
from .kind import Kind, UndefinedKind
# 引入打印助手类
from ._print_helpers import Printable

# 引入废弃装饰器
from sympy.utilities.decorator import deprecated
# 引入 SymPy 废弃警告
from sympy.utilities.exceptions import sympy_deprecation_warning
# 引入可迭代函数和编号符号函数
from sympy.utilities.iterables import iterable, numbered_symbols
# 引入填充格式化字符串的函数和函数名称获取函数
from sympy.utilities.misc import filldedent, func_name

# 引入获取方法解析顺序的函数
from inspect import getmro


def as_Basic(expr):
    """Return expr as a Basic instance using strict sympify
    or raise a TypeError; this is just a wrapper to _sympify,
    raising a TypeError instead of a SympifyError."""
    try:
        # 尝试将表达式转换为 Basic 实例
        return _sympify(expr)
    except SympifyError:
        # 如果转换失败，则抛出类型错误
        raise TypeError(
            'Argument must be a Basic object, not `%s`' % func_name(
            expr))


# 用于按照名称排序可交换参数的关键字
# 用于 Add 和 Mul 类的规范顺序，如果类名都出现在此处
# 此列表中的一些条目与其名称不同拼写，因此它们实际上不会出现在这里。参见 Basic.compare.
ordering_of_classes = [
    # 单例数字
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity',
    # 数字
    'Integer', 'Rational', 'Float',
    # 单例符号
    'Exp1', 'Pi', 'ImaginaryUnit',
    # 符号
    'Symbol', 'Wild',
    # 算术运算
    'Pow', 'Mul', 'Add',
    # 函数值
    'Derivative', 'Integral',
    # 定义的单例函数
    'Abs', 'Sign', 'Sqrt',
    'Floor', 'Ceiling',
    'Re', 'Im', 'Arg',
    'Conjugate',
    'Exp', 'Log',
    'Sin', 'Cos', 'Tan', 'Cot', 'ASin', 'ACos', 'ATan', 'ACot',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'ASinh', 'ACosh', 'ATanh', 'ACoth',
    'RisingFactorial', 'FallingFactorial',
    'factorial', 'binomial',
    'Gamma', 'LowerGamma', 'UpperGamma', 'PolyGamma',
    'Erf',
    # 特殊多项式
    'Chebyshev', 'Chebyshev2',
    # 未定义函数
    'Function', 'WildFunction',
    # 匿名函数
    'Lambda',
    # Landau O 符号
    'Order',
    # 关系运算
    'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan',
]

def _cmp_name(x: type, y: type) -> int:
    """return -1, 0, 1 if the name of x is before that of y.
    A string comparison is done if either name does not appear
    in `ordering_of_classes`. This is the helper for
    ``Basic.compare``

    Examples
    ========

    >>> from sympy import cos, tan, sin
    >>> from sympy.core import basic
    >>> save = basic.ordering_of_classes
    >>> basic.ordering_of_classes = ()
    # 使用 basic 模块中的 _cmp_name 函数比较两个类的名称
    >>> basic._cmp_name(cos, tan)
    # 若第一个类的名称在预定义的类顺序列表中排在第二个类之前，则返回 -1
    -1
    # 修改类的顺序列表为 ["tan", "sin", "cos"]
    >>> basic.ordering_of_classes = ["tan", "sin", "cos"]
    # 重新调用 _cmp_name 函数比较两个类的名称
    >>> basic._cmp_name(cos, tan)
    # 若第一个类的名称在新的类顺序列表中排在第二个类之后，则返回 1
    1
    # 再次调用 _cmp_name 函数比较两个类的名称
    >>> basic._cmp_name(sin, cos)
    # 若第一个类的名称在新的类顺序列表中排在第二个类之前，则返回 -1
    -1
    # 恢复原来的类顺序列表
    >>> basic.ordering_of_classes = save
# 定义 Basic 类，作为所有 SymPy 对象的基类
class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """

    # __slots__ 定义类实例允许的属性，提高内存利用效率
    __slots__ = ('_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'         # assumptions about the instance
                )

    # _args 是元组，存储 Basic 对象的参数
    _args: tuple[Basic, ...]
    # _mhash 是对象的哈希值，初始为 None，将由 __hash__ 方法设置
    _mhash: int | None

    # @property 装饰器，定义 __sympy__ 方法为属性
    @property
    def __sympy__(self):
        return True

    # __init_subclass__ 方法，在每个子类初始化时调用，设置默认假设
    def __init_subclass__(cls):
        super().__init_subclass__()
        # 初始化默认假设 FactKB 和任何假设属性方法
        _prepare_class_assumptions(cls)

    # 下列属性声明为 False，应在相应的子类中覆盖为 True
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False
    is_real: bool | None
    is_extended_real: bool | None
    is_zero: bool | None
    is_negative: bool | None
    is_commutative: bool | None

    # kind 属性，类型为 Kind，默认为 UndefinedKind
    kind: Kind = UndefinedKind

    # __new__ 方法，创建类实例时调用，返回一个新的 Basic 对象
    def __new__(cls, *args):
        obj = object.__new__(cls)
        # 设置默认假设
        obj._assumptions = cls.default_assumptions
        # _mhash 初始为 None，将由 __hash__ 方法设置
        obj._mhash = None

        # _args 存储传入的参数 args，每个元素必须是 Basic 对象
        obj._args = args
        return obj

    # 复制实例的方法，返回一个新的相同类型的实例
    def copy(self):
        return self.func(*self.args)

    # 返回用于创建对象的参数元组
    def __getnewargs__(self):
        return self.args

    # 返回对象状态的方法，这里返回 None
    def __getstate__(self):
        return None
    # 定义对象的 __setstate__ 方法，用于设置对象的状态
    def __setstate__(self, state):
        # 遍历状态字典的键值对
        for name, value in state.items():
            # 使用 setattr 方法设置对象的属性
            setattr(self, name, value)

    # 定义对象的 __reduce_ex__ 方法，用于支持 pickle 协议的序列化
    def __reduce_ex__(self, protocol):
        # 如果协议小于 2，则抛出 NotImplementedError 异常
        if protocol < 2:
            msg = "Only pickle protocol 2 or higher is supported by SymPy"
            raise NotImplementedError(msg)
        # 调用父类的 __reduce_ex__ 方法
        return super().__reduce_ex__(protocol)

    # 定义对象的 __hash__ 方法，返回对象的哈希值
    def __hash__(self) -> int:
        # 如果哈希值未缓存，则计算并缓存哈希值
        h = self._mhash
        if h is None:
            # 计算哈希值，包括对象类型和可哈希内容
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    # 定义对象的 _hashable_content 方法，返回用于计算哈希的元组信息
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    # 定义 assumptions0 属性的 getter 方法，返回对象的初始假设信息
    @property
    def assumptions0(self):
        """
        Return object `type` assumptions.

        For example:

          Symbol('x', real=True)
          Symbol('x', integer=True)

        are different objects. In other words, besides Python type (Symbol in
        this case), the initial assumptions are also forming their typeinfo.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.abc import x
        >>> x.assumptions0
        {'commutative': True}
        >>> x = Symbol("x", positive=True)
        >>> x.assumptions0
        {'commutative': True, 'complex': True, 'extended_negative': False,
         'extended_nonnegative': True, 'extended_nonpositive': False,
         'extended_nonzero': True, 'extended_positive': True, 'extended_real':
         True, 'finite': True, 'hermitian': True, 'imaginary': False,
         'infinite': False, 'negative': False, 'nonnegative': True,
         'nonpositive': False, 'nonzero': True, 'positive': True, 'real':
         True, 'zero': False}
        """
        return {}  # 返回空字典作为对象的初始假设信息
    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        # 如果 self 和 other 是同一个对象，则返回 0
        if self is other:
            return 0
        # 获取 self 和 other 的类名
        n1 = self.__class__
        n2 = other.__class__
        # 调用 _cmp_name 函数比较类名的顺序
        c = _cmp_name(n1, n2)
        if c:
            return c
        # 获取 self 和 other 的可哈希内容
        st = self._hashable_content()
        ot = other._hashable_content()
        # 比较可哈希内容的长度
        c = (len(st) > len(ot)) - (len(st) < len(ot))
        if c:
            return c
        # 逐个比较相同长度的内容
        for l, r in zip(st, ot):
            # 如果 l 或 r 是 frozenset 类型，则转换为 Basic 对象
            l = Basic(*l) if isinstance(l, frozenset) else l
            r = Basic(*r) if isinstance(r, frozenset) else r
            # 如果 l 是 Basic 类型，则调用其 compare 方法进行比较
            if isinstance(l, Basic):
                c = l.compare(r)
            else:
                # 否则直接比较 l 和 r 的大小
                c = (l > r) - (l < r)
            if c:
                return c
        # 如果所有比较都没有差异，则返回 0
        return 0

    @staticmethod
    def _compare_pretty(a, b):
        """return -1, 0, 1 if a is canonically less, equal or
        greater than b. This is used when 'order=old' is selected
        for printing. This puts Order last, orders Rationals
        according to value, puts terms in order wrt the power of
        the last power appearing in a term. Ties are broken using
        Basic.compare.
        """
        # 导入 Order 类
        from sympy.series.order import Order
        # 如果 a 是 Order 类而 b 不是，则 a 大于 b
        if isinstance(a, Order) and not isinstance(b, Order):
            return 1
        # 如果 a 不是 Order 类而 b 是，则 a 小于 b
        if not isinstance(a, Order) and isinstance(b, Order):
            return -1

        # 如果 a 和 b 都是有理数，则按照值的大小比较
        if a.is_Rational and b.is_Rational:
            l = a.p * b.q
            r = b.p * a.q
            return (l > r) - (l < r)
        else:
            # 导入 Wild 类
            from .symbol import Wild
            p1, p2, p3 = Wild("p1"), Wild("p2"), Wild("p3")
            # 使用正则表达式匹配 a 和 b，获取指数部分 p3
            r_a = a.match(p1 * p2**p3)
            if r_a and p3 in r_a:
                a3 = r_a[p3]
                r_b = b.match(p1 * p2**p3)
                if r_b and p3 in r_b:
                    b3 = r_b[p3]
                    # 递归调用 Basic.compare 方法比较 a3 和 b3
                    c = Basic.compare(a3, b3)
                    if c != 0:
                        return c

        # 处理完所有情况后，使用 Basic.compare 方法再次比较 a 和 b
        return Basic.compare(a, b)

    @classmethod
    def fromiter(cls, args, **assumptions):
        """
        Create a new object from an iterable.

        This is a convenience function that allows one to create objects from
        any iterable, without having to convert to a list or tuple first.

        Examples
        ========

        >>> from sympy import Tuple
        >>> Tuple.fromiter(i for i in range(5))
        (0, 1, 2, 3, 4)

        """
        # 使用类方法创建一个新对象，从给定的可迭代对象 args 中构造参数列表，并传递额外的假设参数
        return cls(*tuple(args), **assumptions)

    @classmethod
    def class_key(cls):
        """Nice order of classes."""
        # 返回一个排序键，这里返回一个元组，用于排序类的顺序
        return 5, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        """
        Return a sort key.

        Examples
        ========

        >>> from sympy import S, I

        >>> sorted([S(1)/2, I, -I], key=lambda x: x.sort_key())
        [1/2, -I, I]

        >>> S("[x, 1/x, 1/x**2, x**2, x**(1/2), x**(1/4), x**(3/2)]")
        [x, 1/x, x**(-2), x**2, sqrt(x), x**(1/4), x**(3/2)]
        >>> sorted(_, key=lambda x: x.sort_key())
        [x**(-2), 1/x, x**(1/4), sqrt(x), x, x**(3/2), x**2]

        """

        # XXX: remove this when issue 5169 is fixed
        # 定义一个内部函数 inner_key，用于获取对象的排序键
        def inner_key(arg):
            if isinstance(arg, Basic):
                return arg.sort_key(order)
            else:
                return arg

        # 获取对象的已排序参数
        args = self._sorted_args
        # 构造排序键的元组，包括类键、参数个数和参数的 inner_key
        args = len(args), tuple([inner_key(arg) for arg in args])
        return self.class_key(), args, S.One.sort_key(), S.One

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        # 检查 other 是否可以转换为 Basic 类型，并进行相等性比较
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        # 检查 other 是否定义了 _sympy_ 方法，如果有则使用它进行比较
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented
    def __eq__(self, other):
        """
        Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        # Check if 'self' and 'other' refer to the same object in memory
        if self is other:
            return True

        # If 'other' is not an instance of Basic class, delegate comparison to _do_eq_sympify method
        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)

        # Check for cases where both self and other are not pure number expressions
        # and their types are different
        if not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        
        # Get hashable content of self and other
        a, b = self._hashable_content(), other._hashable_content()
        
        # If hashable content of self and other are not equal, return False
        if a != b:
            return False
        
        # Check if there are numbers embedded within expressions
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        
        # If all checks pass, return True indicating equality
        return True

    def __ne__(self, other):
        """
        ``a != b``  -> Compare two symbolic trees and see whether they are different

        this is the same as:

        ``a.compare(b) != 0``

        but faster
        """
        # Return the negation of the result of __eq__() method
        return not self == other

    def dummy_eq(self, other, symbol=None):
        """
        Compare two expressions and handle dummy symbols.

        Examples
        ========

        >>> from sympy import Dummy
        >>> from sympy.abc import x, y

        >>> u = Dummy('u')

        >>> (u**2 + 1).dummy_eq(x**2 + 1)
        True
        >>> (u**2 + 1) == (x**2 + 1)
        False

        >>> (u**2 + y).dummy_eq(x**2 + y, x)
        True
        >>> (u**2 + y).dummy_eq(x**2 + y, y)
        False

        """
        # Convert self into a form suitable for comparison with other
        s = self.as_dummy()
        # Ensure other is sympified
        o = _sympify(other)
        # Convert other into a form suitable for comparison with self
        o = o.as_dummy()

        # Extract dummy symbols from self
        dummy_symbols = [i for i in s.free_symbols if i.is_Dummy]

        # If there is exactly one dummy symbol in self, assign it to dummy
        if len(dummy_symbols) == 1:
            dummy = dummy_symbols.pop()
        else:
            # If there are no or more than one dummy symbols, directly compare self and other
            return s == o

        # If symbol is not provided, extract symbols from other
        if symbol is None:
            symbols = o.free_symbols

            # If there is exactly one symbol in other, assign it to symbol
            if len(symbols) == 1:
                symbol = symbols.pop()
            else:
                # If there are no or more than one symbols, directly compare self and other
                return s == o

        # Create a new dummy symbol of the same class as the extracted dummy symbol
        tmp = dummy.__class__()

        # Replace dummy with tmp in self and symbol with tmp in other, then compare
        return s.xreplace({dummy: tmp}) == o.xreplace({symbol: tmp})
    # 定义一个方法 atoms，用于返回当前对象中的原子。

    """Returns the atoms that form the current object.

    By default, only objects that are truly atomic and cannot
    be divided into smaller pieces are returned: symbols, numbers,
    and number symbols like I and pi. It is possible to request
    atoms of any type, however, as demonstrated below.

    Examples
    ========

    >>> from sympy import I, pi, sin
    >>> from sympy.abc import x, y
    >>> (1 + x + 2*sin(y + I*pi)).atoms()
    {1, 2, I, pi, x, y}

    If one or more types are given, the results will contain only
    those types of atoms.

    >>> from sympy import Number, NumberSymbol, Symbol
    >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
    {x, y}

    >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
    {1, 2}

    >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
    {1, 2, pi}

    >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
    {1, 2, I, pi}

    Note that I (imaginary unit) and zoo (complex infinity) are special
    types of number symbols and are not part of the NumberSymbol class.

    The type can be given implicitly, too:

    >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
    {x, y}

    Be careful to check your assumptions when using the implicit option
    since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
    of SymPy atom, while ``type(S(2))`` is type ``Integer`` and will find all
    integers in an expression:

    >>> from sympy import S
    >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
    {1}

    >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
    {1, 2}

    Finally, arguments to atoms() can select more than atomic atoms: any
    SymPy type (loaded in core/__init__.py) can be listed as an argument
    and those types of "atoms" as found in scanning the arguments of the
    expression recursively:

    >>> from sympy import Function, Mul
    >>> from sympy.core.function import AppliedUndef
    >>> f = Function('f')
    >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
    {f(x), sin(y + I*pi)}
    >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
    {f(x)}

    >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
    {I*pi, 2*sin(y + I*pi)}

    """

    # 如果 types 参数被传递，将其转换为类型元组
    if types:
        types = tuple(
            [t if isinstance(t, type) else type(t) for t in types])

    # 使用先序遍历函数 _preorder_traversal 获取所有节点
    nodes = _preorder_traversal(self)

    # 如果 types 非空，则筛选出符合指定类型的节点
    if types:
        result = {node for node in nodes if isinstance(node, types)}
    else:
        # 否则，筛选出没有子节点的节点（即原子节点）
        result = {node for node in nodes if not node.args}

    # 返回结果集合
    return result
    def free_symbols(self) -> set[Basic]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol``. Eg: IndexedBase('I')[0].free_symbols

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method."""
        # 初始化一个空集合用于存储自由符号
        empty: set[Basic] = set()
        # 使用生成器表达式获取所有子表达式的自由符号集合，并将其并集到空集合中
        return empty.union(*(a.free_symbols for a in self.args))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        # 返回一个空集合，表示不再支持该属性
        return set()

    def as_dummy(self):
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True. When applied to a
        symbol a new symbol having only the same commutativity will be
        returned.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True
        >>> r.as_dummy()
        _r

        Notes
        =====

        Any object that has structurally bound variables should have
        a property, `bound_symbols` that returns those symbols
        appearing in the object.
        """
        # 导入需要的模块
        from .symbol import Dummy, Symbol
        # 定义一个函数，用于替换结构上绑定的符号
        def can(x):
            # 获取自由符号和绑定符号的交集，并为每个交集元素创建一个虚拟符号
            free = x.free_symbols
            bound = set(x.bound_symbols)
            d = {i: Dummy() for i in bound & free}
            # 使用虚拟符号替换自由符号和绑定符号的交集
            x = x.subs(d)
            # 使用规范变量名替换绑定符号
            x = x.xreplace(x.canonical_variables)
            # 撤销虚拟符号的替换，将原始符号恢复
            return x.xreplace({v: k for k, v in d.items()})
        # 如果表达式不包含 Symbol，则直接返回自身
        if not self.has(Symbol):
            return self
        # 替换具有 bound_symbols 属性的对象中的符号
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            can,
            simultaneous=False)
    def canonical_variables(self):
        """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
        # 检查对象是否具有属性 'bound_symbols'，如果没有则返回空字典
        if not hasattr(self, 'bound_symbols'):
            return {}
        # 创建以 '_' 开头的编号符号生成器
        dums = numbered_symbols('_')
        # 初始化替换字典
        reps = {}
        # 获取自由符号集合减去绑定符号集合中的符号名称
        names = {i.name for i in self.free_symbols - set(self.bound_symbols)}
        # 遍历绑定符号集合
        for b in self.bound_symbols:
            # 从编号符号生成器中获取下一个符号
            d = next(dums)
            # 如果当前绑定符号是一个符号对象
            if b.is_Symbol:
                # 确保生成的新符号名字不会与自由符号集合中的任何符号名称冲突
                while d.name in names:
                    d = next(dums)
            # 将原始绑定符号映射到新生成的符号
            reps[b] = d
        # 返回符号替换字典
        return reps

    def rcall(self, *args):
        """Apply on the argument recursively through the expression tree.

        This method is used to simulate a common abuse of notation for
        operators. For instance, in SymPy the following will not work:

        ``(x+Lambda(y, 2*y))(z) == x+2*z``,

        however, you can use:

        >>> from sympy import Lambda
        >>> from sympy.abc import x, y, z
        >>> (x + Lambda(y, 2*y)).rcall(z)
        x + 2*z
        """
        # 使用 Basic 类的静态方法 _recursive_call 对表达式树递归应用操作
        return Basic._recursive_call(self, args)

    @staticmethod
    def _recursive_call(expr_to_call, on_args):
        """Helper for rcall method."""
        from .symbol import Symbol
        # 判断表达式是否被重载了 '__call__' 方法
        def the_call_method_is_overridden(expr):
            for cls in getmro(type(expr)):
                if '__call__' in cls.__dict__:
                    return cls != Basic

        # 如果表达式可调用且其 '__call__' 方法已被重载
        if callable(expr_to_call) and the_call_method_is_overridden(expr_to_call):
            # 如果表达式是符号对象
            if isinstance(expr_to_call, Symbol):
                # 当调用一个符号时，它会被转换成一个未定义函数
                return expr_to_call
            else:
                # 否则对表达式进行参数替换调用
                return expr_to_call(*on_args)
        # 如果表达式有子表达式
        elif expr_to_call.args:
            # 递归地对每个子表达式应用 _recursive_call 方法
            args = [Basic._recursive_call(
                sub, on_args) for sub in expr_to_call.args]
            # 返回应用了相同类型的新表达式对象
            return type(expr_to_call)(*args)
        else:
            # 如果表达式没有子表达式，直接返回它自身
            return expr_to_call

    def is_hypergeometric(self, k):
        from sympy.simplify.simplify import hypersimp
        from sympy.functions.elementary.piecewise import Piecewise
        # 如果表达式包含分段函数，则返回 None
        if self.has(Piecewise):
            return None
        # 判断是否是超几何函数，返回结果为布尔值
        return hypersimp(self, k) is not None
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        # Check if self is an extended real number
        is_extended_real = self.is_extended_real
        if is_extended_real is False:
            return False
        # Check if self is a number
        if not self.is_number:
            return False
        # Extract real and imaginary parts, evaluating if necessary
        n, i = [p.evalf(2) if not p.is_Number else p
                for p in self.as_real_imag()]
        # If either part is not a number, return False
        if not (i.is_Number and n.is_Number):
            return False
        # If self has a non-zero imaginary part, it cannot be compared
        if i:
            return False
        else:
            # Return True if the precision of the real part is not 1
            return n._prec != 1

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        # Return the class of self (typically the type of the object)
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        # Return the internal tuple of arguments of the object
        return self._args

    @property
    def _sorted_args(self):
        """
        The same as ``args``.  Derived classes which do not fix an
        order on their arguments should override this method to
        produce the sorted representation.
        """
        # Return the sorted version of the arguments, if applicable
        return self.args
    # 定义一个方法，用于计算表达式的内容和原始组成部分，返回一个默认为 S.One，自身为 self 的元组
    def as_content_primitive(self, radical=False, clear=True):
        """A stub to allow Basic args (like Tuple) to be skipped when computing
        the content and primitive components of an expression.

        See Also
        ========

        sympy.core.expr.Expr.as_content_primitive
        """
        return S.One, self

    # 使用缓存装饰器对 _eval_subs 方法进行装饰，用于替换表达式中的 old 对象为 new 对象
    @cacheit
    def _eval_subs(self, old, new):
        """Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        return None

    # 替换表达式中的对象，使用 rule 参数指定的替换规则
    def xreplace(self, rule):
        """
        Replace occurrences of objects within the expression.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule

        Returns
        =======

        xreplace : the result of the replacement

        Examples
        ========

        >>> from sympy import symbols, pi, exp
        >>> x, y, z = symbols('x y z')
        >>> (1 + x*y).xreplace({x: pi})
        pi*y + 1
        >>> (1 + x*y).xreplace({x: pi, y: 2})
        1 + 2*pi

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> (x*y + z).xreplace({x*y: pi})
        z + pi
        >>> (x*y*z).xreplace({x*y: pi})
        x*y*z
        >>> (2*x).xreplace({2*x: y, x: z})
        y
        >>> (2*2*x).xreplace({2*x: y, x: z})
        4*z
        >>> (x + y + 2).xreplace({x + y: 2})
        x + y + 2
        >>> (x + 2 + exp(x + 2)).xreplace({x + 2: y})
        x + exp(y) + 2

        xreplace does not differentiate between free and bound symbols. In the
        following, subs(x, y) would not change x since it is a bound symbol,
        but xreplace does:

        >>> from sympy import Integral
        >>> Integral(x, (x, 1, 2*x)).xreplace({x: y})
        Integral(y, (y, 1, 2*y))

        Trying to replace x with an expression raises an error:

        >>> Integral(x, (x, 1, 2*x)).xreplace({x: 2*y}) # doctest: +SKIP
        ValueError: Invalid limits given: ((2*y, 1, 4*y),)

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        subs: substitution of subexpressions as defined by the objects
              themselves.

        """
        # 调用内部方法 _xreplace 处理替换逻辑，并返回替换后的值和原始值
        value, _ = self._xreplace(rule)
        return value
    def _xreplace(self, rule):
        """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
        # 检查当前对象是否在替换规则中，如果是，则返回替换后的结果和 True 标志
        if self in rule:
            return rule[self], True
        # 如果当前对象不在替换规则中，但规则非空，则尝试对当前对象的子表达式进行替换
        elif rule:
            args = []
            changed = False
            # 遍历当前对象的所有子表达式
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                # 如果子表达式支持 _xreplace 方法，则尝试替换
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            # 如果有任何子表达式发生了替换，则返回替换后的结果和 True 标志
            if changed:
                return self.func(*args), True
        # 如果没有发生替换，则返回当前对象和 False 标志
        return self, False

    @cacheit
    def has(self, *patterns):
        """
        Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y, z
        >>> (x**2 + sin(x*y)).has(z)
        False
        >>> (x**2 + sin(x*y)).has(x, y, z)
        True
        >>> x.has(x)
        True

        Note ``has`` is a structural algorithm with no knowledge of
        mathematics. Consider the following half-open interval:

        >>> from sympy import Interval
        >>> i = Interval.Lopen(0, 5); i
        Interval.Lopen(0, 5)
        >>> i.args
        (0, 5, True, False)
        >>> i.has(4)  # there is no "4" in the arguments
        False
        >>> i.has(0)  # there *is* a "0" in the arguments
        True

        Instead, use ``contains`` to determine whether a number is in the
        interval or not:

        >>> i.contains(4)
        True
        >>> i.contains(0)
        False


        Note that ``expr.has(*patterns)`` is exactly equivalent to
        ``any(expr.has(p) for p in patterns)``. In particular, ``False`` is
        returned when the list of patterns is empty.

        >>> x.has()
        False

        """
        # 调用内部方法 _has 进行实际的匹配检查
        return self._has(iterargs, *patterns)

    def has_xfree(self, s: set[Basic]):
        """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
        # 确保参数 s 的类型是 set，然后检查当前对象的自由参数是否与给定模式集合 s 中的任何一个匹配
        if type(s) is not set:
            raise TypeError('expecting set argument')
        return any(a in s for a in iterfreeargs(self))

    @cacheit
    def has_free(self, *patterns):
        """Return True if self has object(s) ``x`` as a free expression
        else False.

        Examples
        ========

        >>> from sympy import Integral, Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> g = Function('g')
        >>> expr = Integral(f(x), (f(x), 1, g(y)))
        >>> expr.free_symbols
        {y}
        >>> expr.has_free(g(y))
        True
        >>> expr.has_free(*(x, f(x)))
        False

        This works for subexpressions and types, too:

        >>> expr.has_free(g)
        True
        >>> (x + y + 1).has_free(y + 1)
        True
        """
        # 如果未提供任何模式，则直接返回 False
        if not patterns:
            return False
        # 取第一个模式
        p0 = patterns[0]
        # 如果只有一个模式并且它是可迭代的，并且不是 Basic 类型，则抛出 TypeError
        if len(patterns) == 1 and iterable(p0) and not isinstance(p0, Basic):
            raise TypeError(filldedent('''
                Expecting 1 or more Basic args, not a single
                non-Basic iterable. Don't forget to unpack
                iterables: `eq.has_free(*patterns)`'''))
        
        # 尝试进行快速测试
        s = set(patterns)
        # 调用 self.has_xfree 方法检查是否有自由变量
        rv = self.has_xfree(s)
        if rv:
            return rv
        # 否则通过较慢的 _has 方法匹配模式
        return self._has(iterfreeargs, *patterns)

    def _has(self, iterargs, *patterns):
        # 将类型和不可哈希对象分离出来
        type_set = set()  # 只包含类型
        p_set = set()  # 可哈希的非类型对象
        for p in patterns:
            # 如果模式是类型且是 Basic 的子类，则添加到 type_set
            if isinstance(p, type) and issubclass(p, Basic):
                type_set.add(p)
                continue
            # 如果模式不是 Basic 类型，则尝试将其转换为 SymPy 表达式
            if not isinstance(p, Basic):
                try:
                    p = _sympify(p)
                except SympifyError:
                    continue  # Basic 类型不会包含此类对象
            p_set.add(p)  # 如果对象定义了 __eq__ 但未定义 __hash__，则会失败
        
        types = tuple(type_set)   # 将类型集合转换为元组
        for i in iterargs(self):  # 对于 self 的迭代参数
            if i in p_set:        # 如果 i 在 p_set 中
                return True
            # 如果 i 是 types 中的实例
            if isinstance(i, types):
                return True

        # 如果定义了匹配器，则使用匹配器，例如操作定义了检查精确子集包含的匹配器
        for i in p_set - type_set:  # 类型不具有匹配器
            if not hasattr(i, '_has_matcher'):
                continue
            match = i._has_matcher()
            # 如果任何一个匹配成功，则返回 True
            if any(match(arg) for arg in iterargs(self)):
                return True

        # 没有匹配成功，则返回 False
        return False
    # 寻找所有匹配查询的子表达式。
    def find(self, query, group=False):
        """Find all subexpressions matching a query."""
        # 将查询转换为查找查询对象
        query = _make_find_query(query)
        # 对当前表达式树进行前序遍历，并过滤出与查询匹配的结果列表
        results = list(filter(query, _preorder_traversal(self)))

        # 如果不需要分组，返回结果集合
        if not group:
            return set(results)
        else:
            groups = {}

            # 统计每个匹配结果出现的次数
            for result in results:
                if result in groups:
                    groups[result] += 1
                else:
                    groups[result] = 1

            # 返回结果字典，键为匹配的子表达式，值为出现次数
            return groups

    # 计算匹配查询的子表达式的数量。
    def count(self, query):
        """Count the number of matching subexpressions."""
        # 将查询转换为查找查询对象
        query = _make_find_query(query)
        # 对当前表达式树进行前序遍历，统计满足查询条件的子表达式数量
        return sum(bool(query(sub)) for sub in _preorder_traversal(self))

    # 检查当前表达式是否与给定表达式匹配，返回匹配的替换字典。
    def matches(self, expr, repl_dict=None, old=False):
        """
        Helper method for match() that looks for a match between Wild symbols
        in self and expressions in expr.

        Examples
        ========

        >>> from sympy import symbols, Wild, Basic
        >>> a, b, c = symbols('a b c')
        >>> x = Wild('x')
        >>> Basic(a + x, x).matches(Basic(a + b, c)) is None
        True
        >>> Basic(a + x, x).matches(Basic(a + b + c, b + c))
        {x_: b + c}
        """
        # 将给定表达式转换为Sympy表达式
        expr = sympify(expr)
        # 如果给定表达式不是当前对象的实例，则返回None
        if not isinstance(expr, self.__class__):
            return None

        # 如果替换字典为空，则初始化为空字典；否则复制输入的替换字典
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # 如果当前对象与给定表达式相等，则直接返回替换字典
        if self == expr:
            return repl_dict

        # 如果当前对象与给定表达式的参数个数不同，则返回None
        if len(self.args) != len(expr.args):
            return None

        # 逐个比较当前对象的参数与给定表达式的参数，进行递归匹配
        d = repl_dict  # already a copy
        for arg, other_arg in zip(self.args, expr.args):
            if arg == other_arg:
                continue
            # 如果参数是关系表达式，则尝试进行替换并继续匹配
            if arg.is_Relational:
                try:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
                except TypeError: # Should be InvalidComparisonError when introduced
                    d = None
            else:
                d = arg.xreplace(d).matches(other_arg, d, old=old)
            if d is None:
                return None
        return d
    # 定义一个方法用于模式匹配，用于查找符合指定模式的表达式结构
    def match(self, pattern, old=False):
        """
        Pattern matching.

        Wild symbols match all.

        Return ``None`` when expression (self) does not match
        with pattern. Otherwise return a dictionary such that::

          pattern.xreplace(self.match(pattern)) == self

        Examples
        ========

        >>> from sympy import Wild, Sum
        >>> from sympy.abc import x, y
        >>> p = Wild("p")
        >>> q = Wild("q")
        >>> r = Wild("r")
        >>> e = (x+y)**(x+y)
        >>> e.match(p**p)
        {p_: x + y}
        >>> e.match(p**q)
        {p_: x + y, q_: x + y}
        >>> e = (2*x)**2
        >>> e.match(p*q**r)
        {p_: 4, q_: x, r_: 2}
        >>> (p*q**r).xreplace(e.match(p*q**r))
        4*x**2

        Structurally bound symbols are ignored during matching:

        >>> Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, p)))
        {p_: 2}

        But they can be identified if desired:

        >>> Sum(x, (x, 1, 2)).match(Sum(q, (q, 1, p)))
        {p_: 2, q_: x}

        The ``old`` flag will give the old-style pattern matching where
        expressions and patterns are essentially solved to give the
        match. Both of the following give None unless ``old=True``:

        >>> (x - 2).match(p - x, old=True)
        {p_: 2*x - 2}
        >>> (2/x).match(p*x, old=True)
        {p_: 2/x**2}

        """
        # 将模式转换为 Sympy 表达式
        pattern = sympify(pattern)
        # 匹配非绑定符号
        canonical = lambda x: x if x.is_Symbol else x.as_dummy()
        # 使用模式的匹配函数进行匹配，返回匹配结果字典
        m = canonical(pattern).matches(canonical(self), old=old)
        # 如果没有匹配到，则返回 None
        if m is None:
            return m
        # 导入可能的通配符类型
        from .symbol import Wild
        from .function import WildFunction
        from ..tensor.tensor import WildTensor, WildTensorIndex, WildTensorHead
        # 获取模式中的所有通配符（Wild）
        wild = pattern.atoms(Wild, WildFunction, WildTensor, WildTensorIndex, WildTensorHead)
        # 进行健全性检查，确保匹配结果中不包含意外的符号
        if set(m) - wild:
            raise ValueError(filldedent('''
            Some `matches` routine did not use a copy of repl_dict
            and injected unexpected symbols. Report this as an
            error at https://github.com/sympy/sympy/issues'''))
        # 检查是否需要处理绑定符号
        bwild = wild - set(m)
        # 如果不需要处理绑定符号，则直接返回匹配结果字典
        if not bwild:
            return m
        # 将模式中自由通配符替换为匹配结果，以便下一轮匹配
        wpat = pattern.xreplace(m)
        # 识别剩余的绑定通配符
        w = wpat.matches(self, old=old)
        # 将这些绑定通配符加入到匹配结果中
        if w:
            m.update(w)
        # 返回最终的匹配结果字典
        return m

    # 定义一个方法用于统计操作数量，返回操作计数
    def count_ops(self, visual=None):
        """Wrapper for count_ops that returns the operation count."""
        # 导入用于操作计数的函数
        from .function import count_ops
        # 调用操作计数函数，返回结果
        return count_ops(self, visual)
    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
        integrals, sums and products. All objects of this kind will be
        evaluated recursively, unless some species were excluded via 'hints'
        or unless the 'deep' hint was set to 'False'.

        >>> from sympy import Integral
        >>> from sympy.abc import x

        >>> 2*Integral(x, x)
        2*Integral(x, x)

        >>> (2*Integral(x, x)).doit()
        x**2

        >>> (2*Integral(x, x)).doit(deep=False)
        2*Integral(x, x)

        """
        # 根据 'deep' 提示参数的设置，递归地对不被默认计算的对象进行评估
        if hints.get('deep', True):
            # 如果对象是 Basic 类型，对其中的每个项递归地进行 doit 操作
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            # 用处理后的项重新构造对象并返回
            return self.func(*terms)
        else:
            # 如果 deep 参数设置为 False，则直接返回对象本身
            return self

    def simplify(self, **kwargs):
        """See the simplify function in sympy.simplify"""
        # 调用 sympy.simplify 中的 simplify 函数进行简化操作
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        # 调用 sympy.assumptions 中的 refine 函数进行细化操作
        from sympy.assumptions.refine import refine
        return refine(self, assumption)

    def _eval_derivative_n_times(self, s, n):
        # 这是导数的默认评估器（由 `diff` 和 `Derivative` 调用），它尝试通过调用相应的 `_eval_derivative` 方法
        # 进行 n 次求导，如果 n 是符号化的则保持导数未评估。如果对象有符号 n 次导数的闭合形式，则应覆盖此方法。
        from .numbers import Integer
        if isinstance(n, (int, Integer)):
            obj = self
            for i in range(n):
                obj2 = obj._eval_derivative(s)
                if obj == obj2 or obj2 is None:
                    break
                obj = obj2
            return obj2
        else:
            return None

    def _rewrite(self, pattern, rule, method, **hints):
        deep = hints.pop('deep', True)
        if deep:
            # 如果 deep 参数为 True，则对每个参数递归调用 _rewrite 方法
            args = [a._rewrite(pattern, rule, method, **hints)
                    for a in self.args]
        else:
            # 如果 deep 参数为 False，则直接使用原始参数
            args = self.args
        if not pattern or any(isinstance(self, p) for p in pattern):
            # 如果没有规则模式或者 self 是 pattern 中的任何一个实例，则尝试重写对象
            meth = getattr(self, method, None)
            if meth is not None:
                rewritten = meth(*args, **hints)
            else:
                rewritten = self._eval_rewrite(rule, args, **hints)
            if rewritten is not None:
                return rewritten
        if not args:
            # 如果参数为空，则返回对象本身
            return self
        # 用处理后的参数重新构造对象并返回
        return self.func(*args)

    def _eval_rewrite(self, rule, args, **hints):
        # 默认的重写评估器，返回 None 表示不进行具体重写操作
        return None

    _constructor_postprocessor_mapping = {}  # type: ignore

    @classmethod
    # 定义一个静态方法，用于执行构造器后处理器
    def _exec_constructor_postprocessors(cls, obj):
        # 警告：这是一个实验性的API。

        # 这是一个实验性的API，为SymPy核心元素引入构造器后处理器。
        # 如果SymPy表达式的参数具有 `_constructor_postprocessor_mapping` 属性，
        # 则它将被解释为包含匹配表达式节点名称的后处理函数列表的字典。

        # 获取对象的类名
        clsname = obj.__class__.__name__
        # 创建一个默认字典来存储后处理器
        postprocessors = defaultdict(list)
        # 遍历对象的参数
        for i in obj.args:
            try:
                # 检索构造器后处理器映射，根据参数类型的方法解析
                postprocessor_mappings = (
                    Basic._constructor_postprocessor_mapping[cls].items()
                    for cls in type(i).mro()
                    if cls in Basic._constructor_postprocessor_mapping
                )
                # 将找到的后处理函数添加到相应的后处理器列表中
                for k, v in chain.from_iterable(postprocessor_mappings):
                    postprocessors[k].extend([j for j in v if j not in postprocessors[k]])
            except TypeError:
                pass

        # 对于当前类名的后处理器，依次应用于对象
        for f in postprocessors.get(clsname, []):
            obj = f(obj)

        # 返回处理后的对象
        return obj

    def _sage_(self):
        """
        将 *self* 转换为 SageMath 的符号表达式。

        这个版本的方法只是一个占位符。
        """
        # 保存旧的 _sage_ 方法
        old_method = self._sage_
        # 导入 SageMath 的 sympy_init 函数
        from sage.interfaces.sympy import sympy_init
        sympy_init()  # 可能会在 self 的类或超类中猴子补丁 _sage_ 方法
        # 如果方法没有被修改，则抛出未实现异常
        if old_method == self._sage_:
            raise NotImplementedError('conversion to SageMath is not implemented')
        else:
            # 调用新添加的猴子补丁方法
            return self._sage_()

    def could_extract_minus_sign(self):
        # 始终返回 False，参见 Expr.could_extract_minus_sign
        return False  # see Expr.could_extract_minus_sign
    def is_same(a, b, approx=None):
        """Check if two objects a and b are structurally the same.

        This function compares two objects for structural equality,
        considering their types and values. Optionally, an approximation
        function can be provided to handle numerical comparisons.

        Parameters:
        - a : object
            The first object to compare.
        - b : object
            The second object to compare.
        - approx : function, optional
            A function used to approximate equality between numerical values.

        Returns:
        - bool
            True if the objects are structurally the same, False otherwise.

        Notes:
        - This function checks equality based on types and values. Numerical
          comparisons can be customized using the `approx` function.
        - Objects of different types are considered not equal unless handled
          by the approximation function.
        - It uses SymPy's internal functions like `Number` and `postorder_traversal`.

        Examples:
        >>> from sympy import S
        >>> 2.0 == S(2)
        False
        >>> 0.5 == S.Half
        False
        >>> from sympy import Float
        >>> from sympy.core.numbers import equal_valued
        >>> (S.Half/4).is_same(Float(0.125, 1), equal_valued)
        True
        >>> Float(1, 2).is_same(Float(1, 10), equal_valued)
        True
        >>> Float(0.1, 9).is_same(Float(0.1, 10), equal_valued)
        False
        >>> import math
        >>> Float(0.1, 9).is_same(Float(0.1, 10), math.isclose)
        True
        >>> from sympy import eye, Basic
        >>> eye(1) == S(eye(1))  # mutable vs immutable
        True
        >>> Basic.is_same(eye(1), S(eye(1)))
        False

        """
        from .numbers import Number  # Importing Number class from sympy.numbers
        from .traversal import postorder_traversal as pot  # Importing postorder_traversal function from sympy.traversal

        # Iterate over the zip_longest iterator of postorder_traversal results of a and b
        for t in zip_longest(pot(a), pot(b)):
            # If any pair in t contains None, return False
            if None in t:
                return False
            # Unpack a and b from t
            a, b = t
            # If a is an instance of Number
            if isinstance(a, Number):
                # If b is not an instance of Number, return False
                if not isinstance(b, Number):
                    return False
                # If approx function is provided, use it to compare a and b
                if approx:
                    return approx(a, b)
            # If a and b are not equal or their types are not identical, return False
            if not (a == b and a.__class__ == b.__class__):
                return False
        # If all comparisons pass, return True
        return True
# 将 Basic.is_same 赋值给 _aresame，以便于其他地方导入使用
_aresame = Basic.is_same  # for sake of others importing this

# 使用 Basic.compare 函数创建一个用于排序参数的比较函数
_args_sortkey = cmp_to_key(Basic.compare)

# 对所有 Basic 的子类调用 _prepare_class_assumptions 方法，
# 但 Basic 类本身的 __init_subclass__ 方法不会调用，因此在此处手动调用该函数。
_prepare_class_assumptions(Basic)

# Atom 类，继承自 Basic 类，表示原子表达式，即没有子表达式的表达式。
class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    # 如果 self 与 expr 相等，则返回替换字典（如果提供），否则返回 None
    def matches(self, expr, repl_dict=None, old=False):
        if self == expr:
            if repl_dict is None:
                return {}
            return repl_dict.copy()

    # 使用规则字典 rule 进行替换，如果没有找到匹配项则返回自身
    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    # 对原子表达式调用 doit 方法返回自身
    def doit(self, **hints):
        return self

    # 类方法，返回用于排序的关键字，优先级为 (2, 0, 类名)
    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    # 使用缓存装饰器 cacheit 包装的 sort_key 方法，返回排序关键字
    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    # 对于原子表达式，简化操作直接返回自身
    def _eval_simplify(self, **kwargs):
        return self

    # 属性方法，对于原子表达式调用 _sorted_args 会引发 AttributeError
    # 提示不应在原子表达式上使用 _sorted_args
    @property
    def _sorted_args(self):
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')

# 函数 _atomic(e, recursive=False) 返回与替换相关的原子量
def _atomic(e, recursive=False):
    """Return atom-like quantities as far as substitution is
    concerned: Derivatives, Functions and Symbols. Do not
    return any 'atoms' that are inside such quantities unless
    they also appear outside, too, unless `recursive` is True.

    Examples
    ========

    >>> from sympy import Derivative, Function, cos
    >>> from sympy.abc import x, y
    >>> from sympy.core.basic import _atomic
    >>> f = Function('f')
    >>> _atomic(x + y)
    {x, y}
    >>> _atomic(x + f(y))
    {x, f(y)}
    >>> _atomic(Derivative(f(x), x) + cos(x) + y)
    {y, cos(x), Derivative(f(x), x)}

    """
    # 使用前序遍历 pot 来遍历表达式 e 的节点
    pot = _preorder_traversal(e)
    seen = set()
    if isinstance(e, Basic):
        free = getattr(e, "free_symbols", None)
        if free is None:
            return {e}
    else:
        return set()
    from .symbol import Symbol
    from .function import Derivative, Function
    atoms = set()
    for p in pot:
        if p in seen:
            pot.skip()
            continue
        seen.add(p)
        # 如果 p 是 Symbol 并且在 free_symbols 中，则将其加入原子集合
        if isinstance(p, Symbol) and p in free:
            atoms.add(p)
        # 如果 p 是 Derivative 或 Function，则将其加入原子集合
        elif isinstance(p, (Derivative, Function)):
            if not recursive:
                pot.skip()
            atoms.add(p)
    return atoms

# 函数 _make_find_query(query) 的定义未提供，因此无法添加注释
    """将 Basic.find() 的参数转换为可调用对象"""

    # 尝试将 query 参数转换为符号表达式
    try:
        query = _sympify(query)
    # 如果无法转换，捕获 SympifyError 异常并忽略
    except SympifyError:
        pass

    # 如果 query 是一个类型（class），返回一个 lambda 函数，检查表达式是否为该类型的实例
    if isinstance(query, type):
        return lambda expr: isinstance(expr, query)
    
    # 如果 query 是 Basic 类的实例，返回一个 lambda 函数，检查表达式是否与 query 匹配
    elif isinstance(query, Basic):
        return lambda expr: expr.match(query) is not None
    
    # 如果 query 不是类型也不是 Basic 类的实例，则直接返回 query 本身
    return query
# 从当前包的singleton模块中导入S对象，使用延迟导入避免循环导入问题
from .singleton import S
# 从当前包的traversal模块中导入以下函数并重命名：preorder_traversal为_preorder_traversal，
# iterargs，iterfreeargs保持原名不变
from .traversal import (preorder_traversal as _preorder_traversal,
   iterargs, iterfreeargs)

# 使用deprecated装饰器将_preorder_traversal函数标记为已弃用
preorder_traversal = deprecated(
    """
    Using preorder_traversal from the sympy.core.basic submodule is
    deprecated.

    Instead, use preorder_traversal from the top-level sympy namespace, like

        sympy.preorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_preorder_traversal)
```