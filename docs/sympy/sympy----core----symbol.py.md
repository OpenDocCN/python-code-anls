# `D:\src\scipysrc\sympy\sympy\core\symbol.py`

```
# 导入未来版本的注解支持
from __future__ import annotations

# 导入必要的模块和类
from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent

# 导入标准库模块
import string
import re as _re
import random
from itertools import product
from typing import Any

# 定义字符串类，继承自Atom类
class Str(Atom):
    """
    Represents string in SymPy.

    Explanation
    ===========

    Previously, ``Symbol`` was used where string is needed in ``args`` of SymPy
    objects, e.g. denoting the name of the instance. However, since ``Symbol``
    represents mathematical scalar, this class should be used instead.
    """
    __slots__ = ('name',)

    # 构造函数，参数name必须为字符串类型
    def __new__(cls, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))
        # 调用父类Expr的构造函数创建对象
        obj = Expr.__new__(cls, **kwargs)
        obj.name = name
        return obj

    # 返回用于序列化的参数
    def __getnewargs__(self):
        return (self.name,)

    # 返回用于哈希的内容
    def _hashable_content(self):
        return (self.name,)


# 分离给定字典中的假设和非假设项
def _filter_assumptions(kwargs):
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    assumptions, nonassumptions = map(dict, sift(kwargs.items(),
        lambda i: i[0] in _assume_defined,
        binary=True))
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions


# 返回符号s，如果s是符号，则直接返回；如果s是字符串，则返回匹配符号或根据给定的假设创建一个新符号
def _symbol(s, matching_symbol=None, **assumptions):
    """Return s if s is a Symbol, else if s is a string, return either
    the matching_symbol if the names are the same or else a new symbol
    with the same assumptions as the matching symbol (or the
    assumptions as provided).
    """
    # 示例和说明见函数内部注释
    pass  # 实际函数实现未提供，暂时忽略
    # 如果输入参数 s 是字符串类型
    if isinstance(s, str):
        # 如果存在 matching_symbol 并且其名称与 s 相同，则返回 matching_symbol
        if matching_symbol and matching_symbol.name == s:
            return matching_symbol
        # 否则，使用给定的假设创建一个新的 Symbol 对象，并返回
        return Symbol(s, **assumptions)
    # 如果输入参数 s 是 Symbol 类型，则直接返回该 Symbol 对象
    elif isinstance(s, Symbol):
        return s
    # 如果输入参数 s 不是合法的类型（既不是字符串也不是 Symbol），则抛出 ValueError 异常
    else:
        raise ValueError('symbol must be string for symbol name or Symbol')
# 定义一个函数用于生成唯一命名的符号，确保其名字在给定的表达式集合中是唯一的
def uniquely_named_symbol(xname, exprs=(), compare=str, modify=None, **assumptions):
    """
    Return a symbol whose name is derivated from *xname* but is unique
    from any other symbols in *exprs*.

    *xname* and symbol names in *exprs* are passed to *compare* to be
    converted to comparable forms. If ``compare(xname)`` is not unique,
    it is recursively passed to *modify* until unique name is acquired.

    Parameters
    ==========

    xname : str or Symbol
        Base name for the new symbol.

    exprs : Expr or iterable of Expr
        Expressions whose symbols are compared to *xname*.

    compare : function
        Unary function which transforms *xname* and symbol names from
        *exprs* to comparable form.

    modify : function
        Unary function which modifies the string. Default is appending
        the number, or increasing the number if exists.

    Examples
    ========

    By default, a number is appended to *xname* to generate unique name.
    If the number already exists, it is recursively increased.

    >>> from sympy.core.symbol import uniquely_named_symbol, Symbol
    >>> uniquely_named_symbol('x', Symbol('x'))
    x0
    >>> uniquely_named_symbol('x', (Symbol('x'), Symbol('x0')))
    x1
    >>> uniquely_named_symbol('x0', (Symbol('x1'), Symbol('x0')))
    x2

    Name generation can be controlled by passing *modify* parameter.

    >>> from sympy.abc import x
    >>> uniquely_named_symbol('x', x, modify=lambda s: 2*s)
    xx

    """
    # 定义一个函数，用于增加数字以确保字符串唯一
    def numbered_string_incr(s, start=0):
        if not s:
            return str(start)
        i = len(s) - 1
        while i != -1:
            if not s[i].isdigit():
                break
            i -= 1
        n = str(int(s[i + 1:] or start - 1) + 1)
        return s[:i + 1] + n

    # 处理传入的 xname 参数，如果是一个序列，则解包；然后通过 compare 函数转换为可比较的形式
    default = None
    if is_sequence(xname):
        xname, default = xname
    x = compare(xname)
    # 如果 exprs 为空，则直接返回一个符号对象
    if not exprs:
        return _symbol(x, default, **assumptions)
    # 如果 exprs 不是序列，则转换为列表
    if not is_sequence(exprs):
        exprs = [exprs]
    # 收集所有表达式中的符号名和未定义函数名，放入集合 names 中
    names = set().union(
        [i.name for e in exprs for i in e.atoms(Symbol)] +
        [i.func.name for e in exprs for i in e.atoms(AppliedUndef)])
    # 如果 modify 参数为 None，则默认使用 numbered_string_incr 函数
    if modify is None:
        modify = numbered_string_incr
    # 循环直到找到一个唯一的名字，确保 x 不在 names 中
    while any(x == compare(s) for s in names):
        x = modify(x)
    # 返回一个符号对象，名字为 x，带有给定的 assumptions
    return _symbol(x, default, **assumptions)
_uniquely_named_symbol = uniquely_named_symbol

class Symbol(AtomicExpr, Boolean):
    """
    Symbol class is used to create symbolic variables.

    Explanation
    ===========

    Symbolic variables are placeholders for mathematical symbols that can represent numbers, constants, or any other mathematical entities and can be used in mathematical expressions and to perform symbolic computations.

    Assumptions:

    commutative = True
    positive = True
    real = True
    imaginary = True
    complex = True
    complete list of more assumptions- :ref:`predicates`

    You can override the default assumptions in the constructor.


    """
    # 初始化一个类变量，标识是否可比较，默认为 False
    is_comparable = False
    
    # 使用 __slots__ 定义该类实例允许的属性，这里包括 'name', '_assumptions_orig', '_assumptions0'
    __slots__ = ('name', '_assumptions_orig', '_assumptions0')
    
    # 类属性 name，指定为字符串类型
    name: str
    
    # 类属性 is_Symbol 和 is_symbol 均设为 True，表示该类为符号类的实例
    is_Symbol = True
    is_symbol = True
    
    # 定义类属性 kind 的 getter 方法，根据是否可交换性返回相应的类别
    @property
    def kind(self):
        if self.is_commutative:
            return NumberKind
        return UndefinedKind
    
    # 定义类属性 _diff_wrt 的 getter 方法，返回 True，允许针对符号进行导数计算
    @property
    def _diff_wrt(self):
        """Allow derivatives wrt Symbols.
    
        Examples
        ========
    
        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> x._diff_wrt
        True
        """
        return True
    
    # 静态方法 _sanitize，用于清理符号的假设信息
    @staticmethod
    def _sanitize(assumptions, obj=None):
        """Remove None, convert values to bool, check commutativity *in place*.
        """
    
        # 严格处理交换性：不能为 None
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if is_commutative is None:
            whose = '%s ' % obj.__name__ if obj else ''
            raise ValueError(
                '%scommutativity must be True or False.' % whose)
    
        # 清理其他假设，将 None 转换为 bool 值
        for key in list(assumptions.keys()):
            v = assumptions[key]
            if v is None:
                assumptions.pop(key)
                continue
            assumptions[key] = bool(v)
    
    # 定义 _merge 方法，用于合并符号的假设信息
    def _merge(self, assumptions):
        base = self.assumptions0
        for k in set(assumptions) & set(base):
            if assumptions[k] != base[k]:
                raise ValueError(filldedent('''
                    non-matching assumptions for %s: existing value
                    is %s and new value is %s''' % (
                    k, base[k], assumptions[k])))
        base.update(assumptions)
        return base
    
    # 重写 __new__ 方法，创建符号对象，确保每个符号唯一对应其假设信息
    def __new__(cls, name, **assumptions):
        """Symbols are identified by name and assumptions::
    
        >>> from sympy import Symbol
        >>> Symbol("x") == Symbol("x")
        True
        >>> Symbol("x", real=True) == Symbol("x", real=False)
        False
    
        """
        cls._sanitize(assumptions, cls)
        return Symbol.__xnew_cached_(cls, name, **assumptions)
    def __xnew__(cls, name, **assumptions):  # never cached (e.g. dummy)
        # 如果name不是字符串类型，则抛出类型错误
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))

        # 以下代码段的目的是为了确保srepr函数能够根据是否显式指定commutative=True而选择是否包含该假设。
        # 理想情况下，srepr不应区分这些情况，因为这些符号在其他方面相等并被视为等效。
        #
        # 参考：https://github.com/sympy/sympy/issues/8873
        #
        # 复制传入的假设参数到新的字典中
        assumptions_orig = assumptions.copy()

        # 默认情况下，唯一的假设是commutative=True
        assumptions.setdefault('commutative', True)

        # 使用标准事实知识库创建一个新的假设字典
        assumptions_kb = StdFactKB(assumptions)
        assumptions0 = dict(assumptions_kb)

        # 创建一个新的Expr对象
        obj = Expr.__new__(cls)
        obj.name = name

        # 分别设置对象的假设属性
        obj._assumptions = assumptions_kb
        obj._assumptions_orig = assumptions_orig
        obj._assumptions0 = assumptions0

        # 以下三个假设字典有一些差异：
        #
        #   >>> from sympy import Symbol
        #   >>> x = Symbol('x', finite=True)
        #   >>> x.is_positive  # 查询一个假设
        #   >>> x._assumptions
        #   {'finite': True, 'infinite': False, 'commutative': True, 'positive': None}
        #   >>> x._assumptions0
        #   {'finite': True, 'infinite': False, 'commutative': True}
        #   >>> x._assumptions_orig
        #   {'finite': True}
        #
        # 如果两个符号具有相同的名称，它们的_assumptions0相同，则它们相等。可以说应该比较_assumptions_orig，
        # 因为这对用户更透明（这是通过构造函数传递的除了由_sanitize进行的更改之外的值）。
        
        return obj

    @staticmethod
    @cacheit
    def __xnew_cached_(cls, name, **assumptions):  # symbols are always cached
        # 调用Symbol.__xnew__方法创建并返回一个新的Symbol对象，由于符号总是缓存的
        return Symbol.__xnew__(cls, name, **assumptions)

    def __getnewargs_ex__(self):
        # 返回一个元组，包含Symbol对象的名称和初始假设的原始字典
        return ((self.name,), self._assumptions_orig)

    # 注意：__setstate__方法在由__getnewargs_ex__创建的pickle中不需要，
    # 但在v1.9中Symbol更改为在v1.9中使用__getnewargs_ex__时使用。
    # 以前SymPy版本中创建的pickle仍然需要__setstate__，以便它们可以在SymPy > v1.9中解pickle。

    def __setstate__(self, state):
        # 将状态字典中的每个项目设置为self对象的属性
        for name, value in state.items():
            setattr(self, name, value)

    def _hashable_content(self):
        # 注意：仅对用户指定的假设进行哈希，而不是所有的派生假设
        return (self.name,) + tuple(sorted(self.assumptions0.items()))

    def _eval_subs(self, old, new):
        # 如果旧表达式是幂运算，则使用幂类创建一个新的幂对象，并继续进行替换操作
        if old.is_Pow:
            from sympy.core.power import Pow
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    def _eval_refine(self, assumptions):
        # 返回自身对象，因为在这个方法中不执行任何具体的细化操作
        return self
    # 返回 self._assumptions0 的副本
    def assumptions0(self):
        return self._assumptions0.copy()

    # 使用缓存装饰器缓存函数结果，根据给定的 order 返回排序键
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    # 转换为虚拟变量，如果 self.is_commutative 不是 False，则默认为可交换的虚拟变量
    def as_dummy(self):
        # 只有当 self.is_commutative 不为 False 时，显式地将可交换性加入
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

    # 将对象转换为实部和虚部的元组表示
    def as_real_imag(self, deep=True, **hints):
        # 如果 hints 中有 'ignore' 并且其值等于 self，则返回 None
        if hints.get('ignore') == self:
            return None
        else:
            # 导入实部和虚部函数并返回其元组
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    # 判断对象是否是常数，*wrt 是其它对象，**flags 是附加标志
    def is_constant(self, *wrt, **flags):
        # 如果没有给定 wrt，则返回 False
        if not wrt:
            return False
        # 否则检查 self 是否不在 wrt 中，返回结果
        return self not in wrt

    # 返回包含自身的集合作为其自由符号
    @property
    def free_symbols(self):
        return {self}

    # 在这种情况下，将 binary_symbols 定义为 free_symbols
    binary_symbols = free_symbols  # in this case, not always

    # 将对象视为一个集合，返回整个 UniversalSet
    def as_set(self):
        return S.UniversalSet
# 定义一个名为 Dummy 的类，该类继承自 Symbol 类
class Dummy(Symbol):
    """Dummy symbols are each unique, even if they have the same name:

    Examples
    ========

    >>> from sympy import Dummy
    >>> Dummy("x") == Dummy("x")
    False

    If a name is not supplied then a string value of an internal count will be
    used. This is useful when a temporary variable is needed and the name
    of the variable used in the expression is not important.

    >>> Dummy() #doctest: +SKIP
    _Dummy_10

    """

    # 用于计数 Dummy 对象的实例数
    _count = 0
    # 生成伪随机数的随机数生成器对象
    _prng = random.Random()
    # 基础 dummy_index 的起始值，用于保证 Dummy 对象的唯一性
    _base_dummy_index = _prng.randint(10**6, 9*10**6)

    # __slots__ 是用来限制实例的属性，只允许有 'dummy_index' 这一个属性
    __slots__ = ('dummy_index',)

    # 标识这是一个 Dummy 对象
    is_Dummy = True

    # 构造函数，用于创建 Dummy 对象的实例
    def __new__(cls, name=None, dummy_index=None, **assumptions):
        # 如果指定了 dummy_index，则需要同时提供 name
        if dummy_index is not None:
            assert name is not None, "If you specify a dummy_index, you must also provide a name"

        # 如果未指定 name，则使用内部计数生成一个默认的名字
        if name is None:
            name = "Dummy_" + str(Dummy._count)

        # 如果未指定 dummy_index，则使用基础 dummy_index 加上当前计数值
        if dummy_index is None:
            dummy_index = Dummy._base_dummy_index + Dummy._count
            Dummy._count += 1

        # 调用 _sanitize 方法对 assumptions 进行处理
        cls._sanitize(assumptions, cls)
        # 使用 Symbol 类的构造方法创建对象实例
        obj = Symbol.__xnew__(cls, name, **assumptions)

        # 设置实例的 dummy_index 属性
        obj.dummy_index = dummy_index

        return obj

    # 返回一个元组，用于对象的重建
    def __getnewargs_ex__(self):
        return ((self.name, self.dummy_index), self._assumptions_orig)

    # 返回用于排序的关键字
    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (
            2, (self.name, self.dummy_index)), S.One.sort_key(), S.One

    # 返回可散列内容，用于哈希计算
    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.dummy_index,)


# 定义一个名为 Wild 的类，该类也继承自 Symbol 类
class Wild(Symbol):
    """
    A Wild symbol matches anything, or anything
    without whatever is explicitly excluded.

    Parameters
    ==========

    name : str
        Name of the Wild instance.

    exclude : iterable, optional
        Instances in ``exclude`` will not be matched.

    properties : iterable of functions, optional
        Functions, each taking an expressions as input
        and returns a ``bool``. All functions in ``properties``
        need to return ``True`` in order for the Wild instance
        to match the expression.

    Examples
    ========

    >>> from sympy import Wild, WildFunction, cos, pi
    >>> from sympy.abc import x, y, z
    >>> a = Wild('a')
    >>> x.match(a)
    {a_: x}
    >>> pi.match(a)
    {a_: pi}
    >>> (3*x**2).match(a*x)
    {a_: 3*x}
    >>> cos(x).match(a)
    {a_: cos(x)}
    >>> b = Wild('b', exclude=[x])
    >>> (3*x**2).match(b*x)
    >>> b.match(a)
    {a_: b_}
    >>> A = WildFunction('A')
    >>> A.match(a)
    {a_: A_}

    Tips
    ====


    """

    # Wild 类用于表示可以匹配任何表达式的符号
    # 它可以排除指定的实例，或者要求匹配特定的属性函数集合

    # 注释结束
    """
    When using Wild, be sure to use the exclude
    keyword to make the pattern more precise.
    Without the exclude pattern, you may get matches
    that are technically correct, but not what you
    wanted. For example, using the above without
    exclude:

    >>> from sympy import symbols
    >>> a, b = symbols('a b', cls=Wild)
    >>> (2 + 3*y).match(a*x + b*y)
    {a_: 2/x, b_: 3}

    This is technically correct, because
    (2/x)*x + 3*y == 2 + 3*y, but you probably
    wanted it to not match at all. The issue is that
    you really did not want a and b to include x and y,
    and the exclude parameter lets you specify exactly
    this.  With the exclude parameter, the pattern will
    not match.

    >>> a = Wild('a', exclude=[x, y])
    >>> b = Wild('b', exclude=[x, y])
    >>> (2 + 3*y).match(a*x + b*y)

    Exclude also helps remove ambiguity from matches.

    >>> E = 2*x**3*y*z
    >>> a, b = symbols('a b', cls=Wild)
    >>> E.match(a*b)
    {a_: 2*y*z, b_: x**3}
    >>> a = Wild('a', exclude=[x, y])
    >>> E.match(a*b)
    {a_: z, b_: 2*x**3*y}
    >>> a = Wild('a', exclude=[x, y, z])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    Wild also accepts a ``properties`` parameter:

    >>> a = Wild('a', properties=[lambda k: k.is_Integer])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    """

    is_Wild = True  # 设置一个标志变量 is_Wild 为 True

    __slots__ = ('exclude', 'properties')  # 定义 Wild 类的槽（slots），包括 exclude 和 properties

    def __new__(cls, name, exclude=(), properties=(), **assumptions):
        exclude = tuple([sympify(x) for x in exclude])  # 将 exclude 转换为元组，并用 sympify 处理
        properties = tuple(properties)  # 将 properties 转换为元组
        cls._sanitize(assumptions, cls)  # 调用 _sanitize 方法，处理 cls 的假设
        return Wild.__xnew__(cls, name, exclude, properties, **assumptions)  # 调用 Wild 类的 __xnew__ 方法

    def __getnewargs__(self):
        return (self.name, self.exclude, self.properties)  # 返回一个元组，包含 name、exclude 和 properties

    @staticmethod
    @cacheit
    def __xnew__(cls, name, exclude, properties, **assumptions):
        obj = Symbol.__xnew__(cls, name, **assumptions)  # 调用 Symbol 类的 __xnew__ 方法创建对象 obj
        obj.exclude = exclude  # 设置 obj 的 exclude 属性
        obj.properties = properties  # 设置 obj 的 properties 属性
        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self.exclude, self.properties)  # 返回可哈希内容，包括 exclude 和 properties

    # TODO add check against another Wild
    def matches(self, expr, repl_dict=None, old=False):
        if any(expr.has(x) for x in self.exclude):  # 如果 expr 包含 self.exclude 中的任意项，则返回 None
            return None
        if not all(f(expr) for f in self.properties):  # 如果不是所有的 properties 函数都返回 True，则返回 None
            return None
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        repl_dict[self] = expr  # 将 self 和 expr 加入 repl_dict
        return repl_dict  # 返回 repl_dict
# 编译正则表达式模式，用于匹配符号名称中的范围表示
_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

# 定义函数 symbols，将字符串转换为 Symbol 类的实例
def symbols(names, *, cls=Symbol, **args) -> Any:
    r"""
    Transform strings into instances of :class:`Symbol` class.

    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::

        >>> from sympy import symbols, Function

        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')

    The type of output is dependent on the properties of input arguments::

        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}

    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::

        >>> symbols('x', seq=True)
        (x,)

    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::

        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)

        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)

        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))

    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::

        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)

        >>> symbols(':c')
        (a, b, c)

        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)

        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))

    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::

        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)

    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    """
    创建一个空列表，用于存储 symbols 函数返回的符号对象

    如果 names 为空，则返回空列表
    否则，对于每个 name 在 names 中：
        使用 symbols 函数创建一个符号对象，将其添加到 result 列表中，可以通过 cls 关键字参数指定对象类型，args 用于设置对象的假设条件

    返回与输入 names 相同类型的列表，其中包含创建的符号对象
    """
    result = []

    else:
        for name in names:
            result.append(symbols(name, cls=cls, **args))

        return type(names)(result)
def disambiguate(*iter):
    """
    Return a Tuple containing the passed expressions with symbols
    that appear the same when printed replaced with numerically
    subscripted symbols, and all Dummy symbols replaced with Symbols.

    Parameters
    ==========

    iter: list of symbols or expressions.

    Examples
    ========

    >>> from sympy.core.symbol import disambiguate
    >>> from sympy import Dummy, Symbol, Tuple
    >>> from sympy.abc import y

    >>> tup = Symbol('_x'), Dummy('x'), Dummy('x')
    >>> disambiguate(*tup)
    (x_2, x, x_1)

    >>> eqs = Tuple(Symbol('x')/y, Dummy('x')/y)
    >>> disambiguate(*eqs)
    (x_1/y, x/y)

    >>> ix = Symbol('x', integer=True)
    >>> vx = Symbol('x')
    >>> disambiguate(vx + ix)
    (x + x_1,)

    To make your own mapping of symbols to use, pass only the free symbols
    of the expressions and create a dictionary:

    >>> free = eqs.free_symbols
    >>> mapping = dict(zip(free, disambiguate(*free)))
    >>> eqs.xreplace(mapping)
    (x_1/y, x/y)

    """
    # 将输入的参数转换为一个有序的 Tuple
    new_iter = Tuple(*iter)
    # 按照符号的属性进行排序，以确保每个符号的替代顺序一致
    key = lambda x:tuple(sorted(x.assumptions0.items()))
    # 获取所有自由符号，并按照指定的键排序
    syms = ordered(new_iter.free_symbols, keys=key)
    # 初始化一个空的映射字典
    mapping = {}
    # 遍历所有符号
    for s in syms:
        # 将符号名称去掉下划线，并将符号加入到映射字典中
        mapping.setdefault(str(s).lstrip('_'), []).append(s)
    # 初始化一个空的替代字典
    reps = {}
    # 遍历映射中的每个键 k
    for k in mapping:
        # 第一个或唯一的符号不带下标，但需要确保它是一个符号（Symbol），而不是一个虚拟符号（Dummy）
        mapk0 = Symbol("%s" % (k), **mapping[k][0].assumptions0)
        
        # 如果映射中第一个符号不等于 mapk0，则将其映射关系存入 reps 字典
        if mapping[k][0] != mapk0:
            reps[mapping[k][0]] = mapk0
        
        # 对于其余的符号，添加下标并转换为符号对象
        skip = 0
        for i in range(1, len(mapping[k])):
            # 确保新生成的符号名 name 不在映射中已经存在，若存在则增加 skip 直到找到可用的 name
            while True:
                name = "%s_%i" % (k, i + skip)
                if name not in mapping:
                    break
                skip += 1
            
            # 取出映射中的符号 ki，并创建带有新名字的符号对象，保留其原始属性
            ki = mapping[k][i]
            reps[ki] = Symbol(name, **ki.assumptions0)
    
    # 使用 reps 字典替换 new_iter 中的符号，并返回替换后的结果
    return new_iter.xreplace(reps)
```