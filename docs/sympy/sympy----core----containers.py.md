# `D:\src\scipysrc\sympy\sympy\core\containers.py`

```
"""
Module for SymPy containers

    (SymPy objects that store other SymPy objects)

    The containers implemented in this module are subclassed to Basic.
    They are supposed to work seamlessly within the SymPy framework.
"""

from collections import OrderedDict  # 导入 OrderedDict 类
from collections.abc import MutableSet  # 导入 MutableSet 抽象基类
from typing import Any, Callable  # 导入类型提示 Any 和 Callable

from .basic import Basic  # 从 basic 模块导入 Basic 类
from .sorting import default_sort_key, ordered  # 从 sorting 模块导入 default_sort_key 和 ordered 函数
from .sympify import _sympify, sympify, _sympy_converter, SympifyError  # 从 sympify 模块导入 _sympify, sympify, _sympy_converter, SympifyError
from sympy.core.kind import Kind  # 从 sympy.core.kind 导入 Kind 类
from sympy.utilities.iterables import iterable  # 从 sympy.utilities.iterables 导入 iterable 函数
from sympy.utilities.misc import as_int  # 从 sympy.utilities.misc 导入 as_int 函数

class Tuple(Basic):
    """
    Wrapper around the builtin tuple object.

    Explanation
    ===========

    The Tuple is a subclass of Basic, so that it works well in the
    SymPy framework.  The wrapped tuple is available as self.args, but
    you can also access elements or slices with [:] syntax.

    Parameters
    ==========

    sympify : bool
        If ``False``, ``sympify`` is not called on ``args``. This
        can be used for speedups for very large tuples where the
        elements are known to already be SymPy objects.

    Examples
    ========

    >>> from sympy import Tuple, symbols
    >>> a, b, c, d = symbols('a b c d')
    >>> Tuple(a, b, c)[1:]
    (b, c)
    >>> Tuple(a, b, c).subs(a, d)
    (d, b, c)

    """

    def __new__(cls, *args, **kwargs):
        if kwargs.get('sympify', True):
            # 如果 sympify 参数为 True，则对 args 中的每个元素进行 sympify 转换
            args = (sympify(arg) for arg in args)
        # 调用 Basic 类的 __new__ 方法创建新对象
        obj = Basic.__new__(cls, *args)
        return obj

    def __getitem__(self, i):
        if isinstance(i, slice):
            # 如果 i 是 slice 对象，获取索引范围并返回新的 Tuple 对象
            indices = i.indices(len(self))
            return Tuple(*(self.args[j] for j in range(*indices)))
        # 否则直接返回 args 中的元素
        return self.args[i]

    def __len__(self):
        # 返回 Tuple 对象中 args 的长度
        return len(self.args)

    def __contains__(self, item):
        # 检查 item 是否在 Tuple 对象的 args 中
        return item in self.args

    def __iter__(self):
        # 返回 Tuple 对象 args 的迭代器
        return iter(self.args)

    def __add__(self, other):
        # 实现 Tuple 对象与其他 Tuple 或 tuple 的加法操作
        if isinstance(other, Tuple):
            return Tuple(*(self.args + other.args))
        elif isinstance(other, tuple):
            return Tuple(*(self.args + other))
        else:
            return NotImplemented

    def __radd__(self, other):
        # 实现其他对象与 Tuple 对象的右加操作
        if isinstance(other, Tuple):
            return Tuple(*(other.args + self.args))
        elif isinstance(other, tuple):
            return Tuple(*(other + self.args))
        else:
            return NotImplemented

    def __mul__(self, other):
        # 实现 Tuple 对象与整数的乘法操作
        try:
            n = as_int(other)
        except ValueError:
            raise TypeError("Can't multiply sequence by non-integer of type '%s'" % type(other))
        # 返回乘法结果的新 Tuple 对象
        return self.func(*(self.args * n))

    __rmul__ = __mul__  # 右乘与左乘相同

    def __eq__(self, other):
        # 实现 Tuple 对象的相等比较操作
        if isinstance(other, Basic):
            return super().__eq__(other)
        return self.args == other

    def __ne__(self, other):
        # 实现 Tuple 对象的不等比较操作
        if isinstance(other, Basic):
            return super().__ne__(other)
        return self.args != other
    # 返回对象自身的哈希值，基于其参数 args 的哈希值
    def __hash__(self):
        return hash(self.args)

    # 将对象转换为 mpmath 库中的数据类型，并返回一个元组
    def _to_mpmath(self, prec):
        return tuple(a._to_mpmath(prec) for a in self.args)

    # 比较对象是否小于另一个对象
    def __lt__(self, other):
        return _sympify(self.args < other.args)

    # 比较对象是否小于等于另一个对象
    def __le__(self, other):
        return _sympify(self.args <= other.args)

    # XXX: Basic defines count() as something different, so we can't
    # redefine it here. Originally this lead to cse() test failure.
    # 返回对象中特定值出现的次数
    def tuple_count(self, value) -> int:
        """Return number of occurrences of value."""
        return self.args.count(value)

    # 在对象中搜索并返回值第一次出现的索引位置
    def index(self, value, start=None, stop=None):
        """Searches and returns the first index of the value."""
        # XXX: One would expect:
        #
        # return self.args.index(value, start, stop)
        #
        # here. Any trouble with that? Yes:
        #
        # >>> (1,).index(1, None, None)
        # Traceback (most recent call last):
        #   File "<stdin>", line 1, in <module>
        # TypeError: slice indices must be integers or None or have an __index__ method
        #
        # See: http://bugs.python.org/issue13340

        # 如果 start 和 stop 均为 None，则使用原始的 index 方法搜索值
        if start is None and stop is None:
            return self.args.index(value)
        # 如果 stop 为 None，则使用原始的 index 方法搜索值，同时指定起始位置 start
        elif stop is None:
            return self.args.index(value, start)
        # 否则，使用原始的 index 方法搜索值，同时指定起始位置 start 和结束位置 stop
        else:
            return self.args.index(value, start, stop)

    @property
    # 返回 Tuple 实例的类型，总是为 TupleKind，根据每个元素的种类进行参数化
    def kind(self):
        """
        The kind of a Tuple instance.

        The kind of a Tuple is always of :class:`TupleKind` but
        parametrised by the number of elements and the kind of each element.

        Examples
        ========

        >>> from sympy import Tuple, Matrix
        >>> Tuple(1, 2).kind
        TupleKind(NumberKind, NumberKind)
        >>> Tuple(Matrix([1, 2]), 1).kind
        TupleKind(MatrixKind(NumberKind), NumberKind)
        >>> Tuple(1, 2).kind.element_kind
        (NumberKind, NumberKind)

        See Also
        ========

        sympy.matrices.kind.MatrixKind
        sympy.core.kind.NumberKind
        """
        # 返回每个元素的种类构成的元组，作为 TupleKind 的参数
        return TupleKind(*(i.kind for i in self.args))
# 将 lambda 函数注册到 _sympy_converter 字典中，用于将 tuple 转换为 Tuple 对象
_sympy_converter[tuple] = lambda tup: Tuple(*tup)

# 定义装饰器函数 tuple_wrapper，用于将函数参数中的 tuple 转换为 Tuple 对象
def tuple_wrapper(method):
    """
    Decorator that converts any tuple in the function arguments into a Tuple.

    Explanation
    ===========

    The motivation for this is to provide simple user interfaces.  The user can
    call a function with regular tuples in the argument, and the wrapper will
    convert them to Tuples before handing them to the function.

    Explanation
    ===========

    >>> from sympy.core.containers import tuple_wrapper
    >>> def f(*args):
    ...    return args
    >>> g = tuple_wrapper(f)

    The decorated function g sees only the Tuple argument:

    >>> g(0, (1, 2), 3)
    (0, (1, 2), 3)

    """
    def wrap_tuples(*args, **kw_args):
        newargs = []
        for arg in args:
            if isinstance(arg, tuple):
                newargs.append(Tuple(*arg))
            else:
                newargs.append(arg)
        return method(*newargs, **kw_args)
    return wrap_tuples


class Dict(Basic):
    """
    Wrapper around the builtin dict object.

    Explanation
    ===========

    The Dict is a subclass of Basic, so that it works well in the
    SymPy framework.  Because it is immutable, it may be included
    in sets, but its values must all be given at instantiation and
    cannot be changed afterwards.  Otherwise it behaves identically
    to the Python dict.

    Examples
    ========

    >>> from sympy import Dict, Symbol

    >>> D = Dict({1: 'one', 2: 'two'})
    >>> for key in D:
    ...    if key == 1:
    ...        print('%s %s' % (key, D[key]))
    1 one

    The args are sympified so the 1 and 2 are Integers and the values
    are Symbols. Queries automatically sympify args so the following work:

    >>> 1 in D
    True
    >>> D.has(Symbol('one')) # searches keys and values
    True
    >>> 'one' in D # not in the keys
    False
    >>> D[1]
    one

    """

    def __new__(cls, *args):
        # 根据传入的参数类型不同，创建不同类型的 Dict 对象
        if len(args) == 1 and isinstance(args[0], (dict, Dict)):
            items = [Tuple(k, v) for k, v in args[0].items()]
        elif iterable(args) and all(len(arg) == 2 for arg in args):
            items = [Tuple(k, v) for k, v in args]
        else:
            raise TypeError('Pass Dict args as Dict((k1, v1), ...) or Dict({k1: v1, ...})')
        elements = frozenset(items)
        obj = Basic.__new__(cls, *ordered(items))
        obj.elements = elements
        obj._dict = dict(items)  # 以防 Tuple 决定要 sympify
        return obj

    def __getitem__(self, key):
        """x.__getitem__(y) <==> x[y]"""
        try:
            key = _sympify(key)
        except SympifyError:
            raise KeyError(key)

        return self._dict[key]

    def __setitem__(self, key, value):
        # 不支持修改值，抛出异常
        raise NotImplementedError("SymPy Dicts are Immutable")

    def items(self):
        '''Returns a set-like object providing a view on dict's items.
        '''
        return self._dict.items()
    # 返回字典中所有键的列表
    def keys(self):
        '''Returns the list of the dict's keys.'''
        return self._dict.keys()

    # 返回字典中所有值的列表
    def values(self):
        '''Returns the list of the dict's values.'''
        return self._dict.values()

    # 返回字典的迭代器，使得对象可迭代
    def __iter__(self):
        '''x.__iter__() <==> iter(x)'''
        return iter(self._dict)

    # 返回字典中键值对的数量
    def __len__(self):
        '''x.__len__() <==> len(x)'''
        return self._dict.__len__()

    # 获取字典中指定键的值，如果键不存在则返回默认值
    def get(self, key, default=None):
        '''Returns the value for key if the key is in the dictionary.'''
        try:
            key = _sympify(key)  # 尝试将键转换为符号表达式
        except SympifyError:
            return default  # 转换失败时返回默认值
        return self._dict.get(key, default)

    # 检查字典中是否包含指定键
    def __contains__(self, key):
        '''D.__contains__(k) -> True if D has a key k, else False'''
        try:
            key = _sympify(key)  # 尝试将键转换为符号表达式
        except SympifyError:
            return False  # 转换失败时返回 False
        return key in self._dict  # 返回键是否在字典中的结果

    # 比较当前字典对象与另一个对象的大小关系
    def __lt__(self, other):
        return _sympify(self.args < other.args)

    # 返回排序后的字典键列表的元组
    @property
    def _sorted_args(self):
        return tuple(sorted(self.args, key=default_sort_key))

    # 比较当前字典对象与另一个对象是否相等
    def __eq__(self, other):
        if isinstance(other, dict):
            return self == Dict(other)  # 如果其他对象是字典，则比较与当前对象的相等性
        return super().__eq__(other)  # 否则调用父类的相等比较方法

    # 定义哈希方法，用于计算对象的哈希值
    __hash__ : Callable[[Basic], Any] = Basic.__hash__
# this handles dict, defaultdict, OrderedDict
# 定义一个处理 dict、defaultdict、OrderedDict 的 sympy 转换器
_sympy_converter[dict] = lambda d: Dict(*d.items())

# 定义一个 OrderedSet 类，继承自 MutableSet
class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        # 如果提供了可迭代对象，使用 OrderedDict 创建有序集合；否则创建空的有序集合
        if iterable:
            self.map = OrderedDict((item, None) for item in iterable)
        else:
            self.map = OrderedDict()

    def __len__(self):
        # 返回有序集合中元素的数量
        return len(self.map)

    def __contains__(self, key):
        # 检查指定的键是否在有序集合中
        return key in self.map

    def add(self, key):
        # 向有序集合中添加元素
        self.map[key] = None

    def discard(self, key):
        # 从有序集合中移除指定的元素
        self.map.pop(key)

    def pop(self, last=True):
        # 弹出并返回有序集合中的最后一个元素或者第一个元素
        return self.map.popitem(last=last)[0]

    def __iter__(self):
        # 返回一个迭代器，迭代有序集合中的所有元素
        yield from self.map.keys()

    def __repr__(self):
        # 返回有序集合的字符串表示形式
        if not self.map:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.map.keys()))

    def intersection(self, other):
        # 返回当前有序集合与另一个集合的交集
        return self.__class__([val for val in self if val in other])

    def difference(self, other):
        # 返回当前有序集合与另一个集合的差集
        return self.__class__([val for val in self if val not in other])

    def update(self, iterable):
        # 向当前有序集合中添加多个元素
        for val in iterable:
            self.add(val)

# 定义一个 TupleKind 类，继承自 Kind
class TupleKind(Kind):
    """
    TupleKind is a subclass of Kind, which is used to define Kind of ``Tuple``.

    Parameters of TupleKind will be kinds of all the arguments in Tuples, for
    example

    Parameters
    ==========

    args : tuple(element_kind)
       element_kind is kind of element.
       args is tuple of kinds of element

    Examples
    ========

    >>> from sympy import Tuple
    >>> Tuple(1, 2).kind
    TupleKind(NumberKind, NumberKind)
    >>> Tuple(1, 2).kind.element_kind
    (NumberKind, NumberKind)

    See Also
    ========

    sympy.core.kind.NumberKind
    MatrixKind
    sympy.sets.sets.SetKind
    """
    def __new__(cls, *args):
        # 创建一个新的 TupleKind 实例，初始化其参数，并设置 element_kind 属性
        obj = super().__new__(cls, *args)
        obj.element_kind = args
        return obj

    def __repr__(self):
        # 返回 TupleKind 对象的字符串表示形式，包括其 element_kind 属性
        return "TupleKind{}".format(self.element_kind)
```