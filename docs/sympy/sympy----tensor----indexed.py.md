# `D:\src\scipysrc\sympy\sympy\tensor\indexed.py`

```
r"""Module that defines indexed objects.

The classes ``IndexedBase``, ``Indexed``, and ``Idx`` represent a
matrix element ``M[i, j]`` as in the following diagram::

       1) The Indexed class represents the entire indexed object.
                  |
               ___|___
              '       '
               M[i, j]
              /   \__\______
              |             |
              |             |
              |     2) The Idx class represents indices; each Idx can
              |        optionally contain information about its range.
              |
        3) IndexedBase represents the 'stem' of an indexed object, here `M`.
           The stem used by itself is usually taken to represent the entire
           array.

There can be any number of indices on an Indexed object.  No
transformation properties are implemented in these Base objects, but
implicit contraction of repeated indices is supported.

Note that the support for complicated (i.e. non-atomic) integer
expressions as indices is limited.  (This should be improved in
future releases.)

Examples
========

To express the above matrix element example you would write:

>>> from sympy import symbols, IndexedBase, Idx
>>> M = IndexedBase('M')
>>> i, j = symbols('i j', cls=Idx)
>>> M[i, j]
M[i, j]

Repeated indices in a product implies a summation, so to express a
matrix-vector product in terms of Indexed objects:

>>> x = IndexedBase('x')
>>> M[i, j]*x[j]
M[i, j]*x[j]

If the indexed objects will be converted to component based arrays, e.g.
with the code printers or the autowrap framework, you also need to provide
(symbolic or numerical) dimensions.  This can be done by passing an
optional shape parameter to IndexedBase upon construction:

>>> dim1, dim2 = symbols('dim1 dim2', integer=True)
>>> A = IndexedBase('A', shape=(dim1, 2*dim1, dim2))
>>> A.shape
(dim1, 2*dim1, dim2)
>>> A[i, j, 3].shape
(dim1, 2*dim1, dim2)

If an IndexedBase object has no shape information, it is assumed that the
array is as large as the ranges of its indices:

>>> n, m = symbols('n m', integer=True)
>>> i = Idx('i', m)
>>> j = Idx('j', n)
>>> M[i, j].shape
(m, n)
>>> M[i, j].ranges
[(0, m - 1), (0, n - 1)]

The above can be compared with the following:

>>> A[i, 2, j].shape
(dim1, 2*dim1, dim2)
>>> A[i, 2, j].ranges
[(0, m - 1), None, (0, n - 1)]

To analyze the structure of indexed expressions, you can use the methods
get_indices() and get_contraction_structure():

>>> from sympy.tensor import get_indices, get_contraction_structure
>>> get_indices(A[i, j, j])
({i}, {})
>>> get_contraction_structure(A[i, j, j])
{(j,): {A[i, j, j]}}

See the appropriate docstrings for a detailed explanation of the output.
"""

#   TODO:  (some ideas for improvement)
#
#   o test and guarantee numpy compatibility
#      - implement full support for broadcasting
#      - strided arrays
#
#   o more functions to analyze indexed expressions
#      - identify standard constructs, e.g matrix-vector product in a subexpression
#
#   o functions to generate component based arrays (numpy and sympy.Matrix)
#      - generate a single array directly from Indexed
#      - convert simple sub-expressions
#
#   o sophisticated indexing (possibly in subclasses to preserve simplicity)
#      - Idx with range smaller than dimension of Indexed
#      - Idx with stepsize != 1
#      - Idx with step determined by function call
from collections.abc import Iterable  # 导入 collections.abc 模块中的 Iterable 类

from sympy.core.numbers import Number  # 导入 sympy 核心模块中的 Number 类
from sympy.core.assumptions import StdFactKB  # 导入 sympy 核心模块中的 StdFactKB 类
from sympy.core import Expr, Tuple, sympify, S  # 导入 sympy 核心模块中的 Expr, Tuple, sympify, S 类/函数
from sympy.core.symbol import _filter_assumptions, Symbol  # 导入 sympy 核心模块中的 _filter_assumptions, Symbol 类
from sympy.core.logic import fuzzy_bool, fuzzy_not  # 导入 sympy 核心模块中的 fuzzy_bool, fuzzy_not 函数
from sympy.core.sympify import _sympify  # 导入 sympy 核心模块中的 _sympify 函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入 sympy 函数模块中的 KroneckerDelta 函数
from sympy.multipledispatch import dispatch  # 导入 sympy.multipledispatch 模块中的 dispatch 函数
from sympy.utilities.iterables import is_sequence, NotIterable  # 导入 sympy.utilities.iterables 模块中的 is_sequence, NotIterable 函数
from sympy.utilities.misc import filldedent  # 导入 sympy.utilities.misc 模块中的 filldedent 函数


class IndexException(Exception):  # 定义 IndexException 类，继承自 Exception 类
    pass


class Indexed(Expr):  # 定义 Indexed 类，继承自 Expr 类，表示带有索引的数学对象
    """Represents a mathematical object with indices.

    >>> from sympy import Indexed, IndexedBase, Idx, symbols
    >>> i, j = symbols('i j', cls=Idx)
    >>> Indexed('A', i, j)
    A[i, j]

    It is recommended that ``Indexed`` objects be created by indexing ``IndexedBase``:
    ``IndexedBase('A')[i, j]`` instead of ``Indexed(IndexedBase('A'), i, j)``.

    >>> A = IndexedBase('A')
    >>> a_ij = A[i, j]           # Prefer this,
    >>> b_ij = Indexed(A, i, j)  # over this.
    >>> a_ij == b_ij
    True

    """
    is_Indexed = True  # 设置类属性 is_Indexed 为 True，表示该类是 Indexed 类
    is_symbol = True  # 设置类属性 is_symbol 为 True，表示该类是符号
    is_Atom = True  # 设置类属性 is_Atom 为 True，表示该类是原子

    def __new__(cls, base, *args, **kw_args):
        from sympy.tensor.array.ndim_array import NDimArray  # 从 sympy.tensor.array.ndim_array 模块导入 NDimArray 类
        from sympy.matrices.matrixbase import MatrixBase  # 从 sympy.matrices.matrixbase 模块导入 MatrixBase 类

        if not args:
            raise IndexException("Indexed needs at least one index.")  # 如果参数 args 为空，则抛出 IndexException 异常
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)  # 如果 base 是字符串或符号，则将其转换为 IndexedBase 对象
        elif not hasattr(base, '__getitem__') and not isinstance(base, IndexedBase):
            raise TypeError(filldedent("""
                The base can only be replaced with a string, Symbol,
                IndexedBase or an object with a method for getting
                items (i.e. an object with a `__getitem__` method).
                """))  # 如果 base 不具有 __getitem__ 方法且不是 IndexedBase 类型，则抛出 TypeError 异常
        args = list(map(sympify, args))  # 对参数 args 中的每个元素应用 sympify 函数转换为符号表达式
        if isinstance(base, (NDimArray, Iterable, Tuple, MatrixBase)) and all(i.is_number for i in args):
            # 如果 base 是 NDimArray, Iterable, Tuple 或 MatrixBase 类型，且所有 args 元素都是数字类型
            if len(args) == 1:
                return base[args[0]]  # 如果 args 只有一个元素，则返回 base 对应该索引的值
            else:
                return base[args]  # 如果 args 有多个元素，则返回 base 对应这些索引的值

        base = _sympify(base)  # 对 base 进行符号化处理

        obj = Expr.__new__(cls, base, *args, **kw_args)  # 调用父类 Expr 的构造函数创建对象

        IndexedBase._set_assumptions(obj, base.assumptions0)  # 设置 IndexedBase 的假设属性

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))  # 返回对象的可哈希内容

    @property
    def name(self):
        return str(self)  # 返回对象的名称字符串

    @property
    def _diff_wrt(self):
        """允许相对于“Indexed”对象进行导数计算。"""
        return True

    def _eval_derivative(self, wrt):
        """计算对象的导数。

        Args:
            wrt: 导数的目标对象

        Returns:
            如果目标对象是“Indexed”且与当前对象的基础相同，则返回导数结果；如果当前对象的基础是NDimArray，则返回基于数组的导数；否则返回S.NaN或S.Zero。

        Raises:
            IndexException: 如果“Indexed”对象与目标对象的索引数量不匹配。

        """
        from sympy.tensor.array.ndim_array import NDimArray

        if isinstance(wrt, Indexed) and wrt.base == self.base:
            if len(self.indices) != len(wrt.indices):
                msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                       wrt)
                raise IndexException(msg)
            result = S.One
            for index1, index2 in zip(self.indices, wrt.indices):
                result *= KroneckerDelta(index1, index2)
            return result
        elif isinstance(self.base, NDimArray):
            from sympy.tensor.array import derive_by_array
            return Indexed(derive_by_array(self.base, wrt), *self.args[1:])
        else:
            if Tuple(self.indices).has(wrt):
                return S.NaN
            return S.Zero

    @property
    def assumptions0(self):
        """返回不为None的假设字典项。"""
        return {k: v for k, v in self._assumptions.items() if v is not None}

    @property
    def base(self):
        """返回“Indexed”对象的“IndexedBase”。

        Examples
        ========

        >>> from sympy import Indexed, IndexedBase, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).base
        A
        >>> B = IndexedBase('B')
        >>> B == B[i, j].base
        True

        """
        return self.args[0]

    @property
    def indices(self):
        """
        返回“Indexed”对象的索引列表。

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).indices
        (i, j)

        """
        return self.args[1:]

    @property
    def rank(self):
        """
        返回“Indexed”对象的秩（维度数量）。

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j, k, l, m = symbols('i:m', cls=Idx)
        >>> Indexed('A', i, j).rank
        2
        >>> q = Indexed('A', i, j, k, l, m)
        >>> q.rank
        5
        >>> q.rank == len(q.indices)
        True

        """
        return len(self.args) - 1
    def shape(self):
        """Returns a list with dimensions of each index.

        Dimensions is a property of the array, not of the indices.  Still, if
        the ``IndexedBase`` does not define a shape attribute, it is assumed
        that the ranges of the indices correspond to the shape of the array.

        >>> from sympy import IndexedBase, Idx, symbols
        >>> n, m = symbols('n m', integer=True)
        >>> i = Idx('i', m)
        >>> j = Idx('j', m)
        >>> A = IndexedBase('A', shape=(n, n))
        >>> B = IndexedBase('B')
        >>> A[i, j].shape
        (n, n)
        >>> B[i, j].shape
        (m, m)
        """

        # Check if the base object has a defined shape attribute
        if self.base.shape:
            return self.base.shape
        
        # If shape is not defined, infer it from the ranges of indices
        sizes = []
        for i in self.indices:
            upper = getattr(i, 'upper', None)
            lower = getattr(i, 'lower', None)
            # Ensure that both upper and lower bounds are defined
            if None in (upper, lower):
                raise IndexException(filldedent("""
                    Range is not defined for all indices in: %s""" % self))
            try:
                size = upper - lower + 1
            except TypeError:
                raise IndexException(filldedent("""
                    Shape cannot be inferred from Idx with
                    undefined range: %s""" % self))
            sizes.append(size)
        return Tuple(*sizes)

    @property
    def ranges(self):
        """Returns a list of tuples with lower and upper range of each index.

        If an index does not define the data members upper and lower, the
        corresponding slot in the list contains ``None`` instead of a tuple.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> Indexed('A', Idx('i', 2), Idx('j', 4), Idx('k', 8)).ranges
        [(0, 1), (0, 3), (0, 7)]
        >>> Indexed('A', Idx('i', 3), Idx('j', 3), Idx('k', 3)).ranges
        [(0, 2), (0, 2), (0, 2)]
        >>> x, y, z = symbols('x y z', integer=True)
        >>> Indexed('A', x, y, z).ranges
        [None, None, None]

        """
        ranges = []
        sentinel = object()
        for i in self.indices:
            upper = getattr(i, 'upper', sentinel)
            lower = getattr(i, 'lower', sentinel)
            if sentinel not in (upper, lower):
                ranges.append((lower, upper))
            else:
                ranges.append(None)
        return ranges

    def _sympystr(self, p):
        indices = list(map(p.doprint, self.indices))
        return "%s[%s]" % (p.doprint(self.base), ", ".join(indices))

    @property
    def free_symbols(self):
        base_free_symbols = self.base.free_symbols
        indices_free_symbols = {
            fs for i in self.indices for fs in i.free_symbols}
        if base_free_symbols:
            return {self} | base_free_symbols | indices_free_symbols
        else:
            return indices_free_symbols
    # 定义一个方法 expr_free_symbols，用于获取表达式中的自由符号
    def expr_free_symbols(self):
        # 导入 sympy 库中的异常处理函数 sympy_deprecation_warning
        from sympy.utilities.exceptions import sympy_deprecation_warning
        # 发出 sympy_deprecation_warning 警告，提示 expr_free_symbols 属性已被弃用，建议使用 free_symbols 方法获取表达式的自由符号
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",  # 指定从版本 1.9 开始被弃用
            active_deprecations_target="deprecated-expr-free-symbols")  # 指定目标为 deprecated-expr-free-symbols

        # 返回一个集合，包含当前对象 self 的内容，表示该方法返回表达式中的自由符号
        return {self}
class IndexedBase(Expr, NotIterable):
    """Represent the base or stem of an indexed object

    The IndexedBase class represent an array that contains elements. The main purpose
    of this class is to allow the convenient creation of objects of the Indexed
    class.  The __getitem__ method of IndexedBase returns an instance of
    Indexed.  Alone, without indices, the IndexedBase class can be used as a
    notation for e.g. matrix equations, resembling what you could do with the
    Symbol class.  But, the IndexedBase class adds functionality that is not
    available for Symbol instances:

      -  An IndexedBase object can optionally store shape information.  This can
         be used in to check array conformance and conditions for numpy
         broadcasting.  (TODO)
      -  An IndexedBase object implements syntactic sugar that allows easy symbolic
         representation of array operations, using implicit summation of
         repeated indices.
      -  The IndexedBase object symbolizes a mathematical structure equivalent
         to arrays, and is recognized as such for code generation and automatic
         compilation and wrapping.

    >>> from sympy.tensor import IndexedBase, Idx
    >>> from sympy import symbols
    >>> A = IndexedBase('A'); A
    A
    >>> type(A)
    <class 'sympy.tensor.indexed.IndexedBase'>

    When an IndexedBase object receives indices, it returns an array with named
    axes, represented by an Indexed object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'sympy.tensor.indexed.Indexed'>

    The IndexedBase constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedBase('B', shape=(o, p))
    >>> B[i, j].shape
    (o, p)

    Assumptions can be specified with keyword arguments the same way as for Symbol:

    >>> A_real = IndexedBase('A', real=True)
    >>> A_real.is_real
    True
    >>> A != A_real
    True

    Assumptions can also be inherited if a Symbol is used to initialize the IndexedBase:

    >>> I = symbols('I', integer=True)
    >>> C_inherit = IndexedBase(I)
    >>> C_explicit = IndexedBase('I', integer=True)
    >>> C_inherit == C_explicit
    True
    """
    # 声明类属性 is_symbol 和 is_Atom，表示该类是符号类并且是原子类
    is_symbol = True
    is_Atom = True

    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        # 复制一份假设信息到临时变量
        tmp_asm_copy = assumptions.copy()
        # 根据 'commutative' 关键字在假设信息中获取是否可交换的信息，并转换为布尔值
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        # 将 'commutative' 假设设置为获取到的布尔值
        assumptions['commutative'] = is_commutative
        # 将假设信息作为标准事实知识库对象赋值给 obj 的假设属性
        obj._assumptions = StdFactKB(assumptions)
        # 将原始假设信息赋值回 obj 的生成器属性
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873
    def __new__(cls, label, shape=None, *, offset=S.Zero, strides=None, **kw_args):
        # 导入必要的模块和类
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array.ndim_array import NDimArray

        # 过滤并获取假设信息
        assumptions, kw_args = _filter_assumptions(kw_args)

        # 根据不同的输入类型处理标签(label)
        if isinstance(label, str):
            # 如果标签是字符串，则创建一个符号对象(Symbol)
            label = Symbol(label, **assumptions)
        elif isinstance(label, Symbol):
            # 如果标签是符号(Symbol)，则合并假设信息
            assumptions = label._merge(assumptions)
        elif isinstance(label, (MatrixBase, NDimArray)):
            # 如果标签是矩阵基类或多维数组类，则直接返回该对象
            return label
        elif isinstance(label, Iterable):
            # 如果标签是可迭代对象，则转换成符号化的形式
            return _sympify(label)
        else:
            # 否则，将标签符号化
            label = _sympify(label)

        # 如果shape是序列，则转换成元组(Tuple)
        if is_sequence(shape):
            shape = Tuple(*shape)
        elif shape is not None:
            shape = Tuple(shape)

        # 根据是否提供了shape信息，创建新的对象
        if shape is not None:
            obj = Expr.__new__(cls, label, shape)
        else:
            obj = Expr.__new__(cls, label)

        # 设置对象的形状(shape)，偏移(offset)，步长(strides)和名称(name)
        obj._shape = shape
        obj._offset = offset
        obj._strides = strides
        obj._name = str(label)

        # 设置对象的假设信息(assumptions)
        IndexedBase._set_assumptions(obj, assumptions)
        return obj

    @property
    def name(self):
        # 返回对象的名称(name)
        return self._name

    def _hashable_content(self):
        # 返回可哈希内容，包括父类的哈希内容和假设信息的排序元组
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def assumptions0(self):
        # 返回对象的假设信息(假设值不为None的部分)
        return {k: v for k, v in self._assumptions.items() if v is not None}

    def __getitem__(self, indices, **kw_args):
        # 如果indices是序列，则创建Indexed对象
        if is_sequence(indices):
            # 特殊情况：M[*my_tuple] 是语法错误，因此需要特别处理
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return Indexed(self, *indices, **kw_args)
        else:
            # 如果indices不是序列，则创建Indexed对象
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return Indexed(self, indices, **kw_args)

    @property
    def shape(self):
        """Returns the shape of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase, Idx
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).shape
        (x, y)

        Note: If the shape of the ``IndexedBase`` is specified, it will override
        any shape information given by the indices.

        >>> A = IndexedBase('A', shape=(x, y))
        >>> B = IndexedBase('B')
        >>> i = Idx('i', 2)
        >>> j = Idx('j', 1)
        >>> A[i, j].shape
        (x, y)
        >>> B[i, j].shape
        (2, 1)

        """
        # 返回对象的形状(shape)
        return self._shape

    @property
    def strides(self):
        """Returns the strided scheme for the ``IndexedBase`` object.

        Normally this is a tuple denoting the number of
        steps to take in the respective dimension when traversing
        an array. For code generation purposes strides='C' and
        strides='F' can also be used.

        strides='C' would mean that code printer would unroll
        in row-major order and 'F' means unroll in column major
        order.

        """

        # 返回该 IndexedBase 对象的步进方案（strides）
        return self._strides

    @property
    def offset(self):
        """Returns the offset for the ``IndexedBase`` object.

        This is the value added to the resulting index when the
        2D Indexed object is unrolled to a 1D form. Used in code
        generation.

        Examples
        ==========
        >>> from sympy.printing import ccode
        >>> from sympy.tensor import IndexedBase, Idx
        >>> from sympy import symbols
        >>> l, m, n, o = symbols('l m n o', integer=True)
        >>> A = IndexedBase('A', strides=(l, m, n), offset=o)
        >>> i, j, k = map(Idx, 'ijk')
        >>> ccode(A[i, j, k])
        'A[l*i + m*j + n*k + o]'

        """
        # 返回该 IndexedBase 对象的偏移量（offset）
        return self._offset

    @property
    def label(self):
        """Returns the label of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).label
        A

        """
        # 返回该 IndexedBase 对象的标签（label）
        return self.args[0]

    def _sympystr(self, p):
        # 返回该 IndexedBase 对象的字符串表示形式，使用给定的打印机 p
        return p.doprint(self.label)
class Idx(Expr):
    """Represents an integer index as an ``Integer`` or integer expression.

    There are a number of ways to create an ``Idx`` object.  The constructor
    takes two arguments:

    ``label``
        An integer or a symbol that labels the index.
    ``range``
        Optionally you can specify a range as either

        * ``Symbol`` or integer: This is interpreted as a dimension. Lower and
          upper bounds are set to ``0`` and ``range - 1``, respectively.
        * ``tuple``: The two elements are interpreted as the lower and upper
          bounds of the range, respectively.

    Note: bounds of the range are assumed to be either integer or infinite (oo
    and -oo are allowed to specify an unbounded range). If ``n`` is given as a
    bound, then ``n.is_integer`` must not return false.

    For convenience, if the label is given as a string it is automatically
    converted to an integer symbol.  (Note: this conversion is not done for
    range or dimension arguments.)

    Examples
    ========

    >>> from sympy import Idx, symbols, oo
    >>> n, i, L, U = symbols('n i L U', integer=True)

    If a string is given for the label an integer ``Symbol`` is created and the
    bounds are both ``None``:

    >>> idx = Idx('qwerty'); idx
    qwerty
    >>> idx.lower, idx.upper
    (None, None)

    Both upper and lower bounds can be specified:

    >>> idx = Idx(i, (L, U)); idx
    i
    >>> idx.lower, idx.upper
    (L, U)

    When only a single bound is given it is interpreted as the dimension
    and the lower bound defaults to 0:

    >>> idx = Idx(i, n); idx.lower, idx.upper
    (0, n - 1)
    >>> idx = Idx(i, 4); idx.lower, idx.upper
    (0, 3)
    >>> idx = Idx(i, oo); idx.lower, idx.upper
    (0, oo)

    """

    # Indicates that this index represents an integer
    is_integer = True
    # Indicates that the index is finite
    is_finite = True
    # Indicates that the index is real
    is_real = True
    # Indicates that the index is a symbol
    is_symbol = True
    # Indicates that the index is an atom
    is_Atom = True
    # Indicates that differentiation can be done with respect to this index
    _diff_wrt = True
    # 定义一个特殊方法 __new__，用于创建 Idx 类的新实例
    def __new__(cls, label, range=None, **kw_args):

        # 如果 label 是字符串，则将其转换为符号对象，确保是整数
        if isinstance(label, str):
            label = Symbol(label, integer=True)
        
        # 使用 sympify 函数将 label 和 range 转换为 SymPy 对象
        label, range = list(map(sympify, (label, range)))

        # 如果 label 是一个数字
        if label.is_Number:
            # 如果不是整数，则抛出类型错误异常
            if not label.is_integer:
                raise TypeError("Index is not an integer number.")
            # 返回 label 本身
            return label

        # 如果 label 不是整数，则抛出类型错误异常
        if not label.is_integer:
            raise TypeError("Idx object requires an integer label.")

        # 如果 range 是一个序列
        elif is_sequence(range):
            # 如果 range 的长度不是 2，则抛出值错误异常
            if len(range) != 2:
                raise ValueError(filldedent("""
                    Idx range tuple must have length 2, but got %s""" % len(range)))
            # 检查 range 中的边界值是否为整数，不是则抛出类型错误异常
            for bound in range:
                if (bound.is_integer is False and bound is not S.Infinity
                        and bound is not S.NegativeInfinity):
                    raise TypeError("Idx object requires integer bounds.")
            # 构建参数 args
            args = label, Tuple(*range)
        
        # 如果 range 是 Expr 类型
        elif isinstance(range, Expr):
            # 如果 range 不是整数或者不是正无穷大，则抛出类型错误异常
            if range is not S.Infinity and fuzzy_not(range.is_integer):
                raise TypeError("Idx object requires an integer dimension.")
            # 构建参数 args
            args = label, Tuple(0, range - 1)
        
        # 如果 range 存在且不为空，则抛出类型错误异常
        elif range:
            raise TypeError(filldedent("""
                The range must be an ordered iterable or
                integer SymPy expression."""))
        
        # 如果没有特别处理的情况，则仅传递 label 参数
        else:
            args = label,

        # 调用 Expr 类的 __new__ 方法创建对象
        obj = Expr.__new__(cls, *args, **kw_args)
        # 设置对象的假设为有限和实数
        obj._assumptions["finite"] = True
        obj._assumptions["real"] = True
        # 返回创建的对象
        return obj

    @property
    def label(self):
        """返回 Idx 对象的标签（整数或整数表达式）。

        Examples
        ========

        >>> from sympy import Idx, Symbol
        >>> x = Symbol('x', integer=True)
        >>> Idx(x).label
        x
        >>> j = Symbol('j', integer=True)
        >>> Idx(j).label
        j
        >>> Idx(j + 1).label
        j + 1

        """
        # 返回参数元组中的第一个元素作为标签
        return self.args[0]

    @property
    def lower(self):
        """返回 Idx 的下限。

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).lower
        0
        >>> Idx('j', 5).lower
        0
        >>> Idx('j').lower is None
        True

        """
        try:
            # 返回参数元组中的第二个元素的第一个元素作为下限
            return self.args[1][0]
        except IndexError:
            return

    @property
    def upper(self):
        """返回 Idx 的上限。

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).upper
        1
        >>> Idx('j', 5).upper
        4
        >>> Idx('j').upper is None
        True

        """
        try:
            # 返回参数元组中的第二个元素的第二个元素作为上限
            return self.args[1][1]
        except IndexError:
            return

    def _sympystr(self, p):
        # 返回标签的打印字符串形式
        return p.doprint(self.label)

    @property
    def name(self):
        # 如果标签是符号，则返回其名称；否则返回其字符串表示
        return self.label.name if self.label.is_Symbol else str(self.label)

    @property
    def free_symbols(self):
        # 返回一个集合，其中包含对象自身
        return {self}
# 定义一个特定签名的函数 _eval_is_ge，用于比较两个 Idx 类型的对象
@dispatch(Idx, Idx)
def _eval_is_ge(lhs, rhs): # noqa:F811
    # 如果 rhs 的上界为 None，则 other_upper 为 rhs 自身，否则为 rhs 的上界
    other_upper = rhs if rhs.upper is None else rhs.upper
    # 如果 rhs 的下界为 None，则 other_lower 为 rhs 自身，否则为 rhs 的下界
    other_lower = rhs if rhs.lower is None else rhs.lower

    # 如果 lhs 的下界不为 None，并且 lhs 的下界大于等于 other_upper，则返回 True
    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    # 如果 lhs 的上界不为 None，并且 lhs 的上界小于 other_lower，则返回 False
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    # 否则返回 None
    return None


# 定义一个特定签名的函数 _eval_is_ge，用于比较 Idx 和 Number 类型的对象
@dispatch(Idx, Number)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    # other_upper 和 other_lower 均为 rhs 的值，因为 rhs 是一个数字
    other_upper = rhs
    other_lower = rhs

    # 如果 lhs 的下界不为 None，并且 lhs 的下界大于等于 other_upper，则返回 True
    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    # 如果 lhs 的上界不为 None，并且 lhs 的上界小于 other_lower，则返回 False
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    # 否则返回 None
    return None


# 定义一个特定签名的函数 _eval_is_ge，用于比较 Number 和 Idx 类型的对象
@dispatch(Number, Idx)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    # other_upper 和 other_lower 均为 lhs 的值，因为 lhs 是一个数字
    other_upper = lhs
    other_lower = lhs

    # 如果 rhs 的上界不为 None，并且 rhs 的上界小于等于 other_lower，则返回 True
    if rhs.upper is not None and (rhs.upper <= other_lower) == True:
        return True
    # 如果 rhs 的下界不为 None，并且 rhs 的下界大于 other_upper，则返回 False
    if rhs.lower is not None and (rhs.lower > other_upper) == True:
        return False
    # 否则返回 None
    return None
```