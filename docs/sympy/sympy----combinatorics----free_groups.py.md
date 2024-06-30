# `D:\src\scipysrc\sympy\sympy\combinatorics\free_groups.py`

```
# 导入必要的模块和类
from __future__ import annotations  # 导入未来版本的 annotations 特性

from sympy.core import S  # 导入 Sympy 核心模块中的 S
from sympy.core.expr import Expr  # 导入 Sympy 核心表达式类 Expr
from sympy.core.symbol import Symbol, symbols as _symbols  # 导入 Sympy 核心符号类 Symbol 和 symbols 函数
from sympy.core.sympify import CantSympify  # 导入 Sympy 核心 sympify 异常类 CantSympify
from sympy.printing.defaults import DefaultPrinting  # 导入 Sympy 默认打印设置类 DefaultPrinting
from sympy.utilities import public  # 导入 Sympy 公共函数修饰器 public
from sympy.utilities.iterables import flatten, is_sequence  # 导入 Sympy 迭代工具函数 flatten 和 is_sequence
from sympy.utilities.magic import pollute  # 导入 Sympy 魔术功能 pollute
from sympy.utilities.misc import as_int  # 导入 Sympy 杂项函数 as_int


@public
def free_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)
        符号或表达式的字符串、Symbol/Expr 对象，或者它们的序列（可以为空）

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> F, x, y, z = free_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> x**2*y**-1
    x**2*y**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)  # 创建自由群对象
    return (_free_group,) + tuple(_free_group.generators)  # 返回自由群对象及其生成器元组


@public
def xfree_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1)))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)
        符号或表达式的字符串、Symbol/Expr 对象，或者它们的序列（可以为空）

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import xfree_group
    >>> F, (x, y, z) = xfree_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> y**2*x**-2*z**-1
    y**2*x**-2*z**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)  # 创建自由群对象
    return (_free_group, _free_group.generators)  # 返回自由群对象及其生成器元组


@public
def vfree_group(symbols):
    """Construct a free group and inject ``f_0, f_1, ..., f_(n-1)`` as symbols
    into the global namespace.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)
        符号或表达式的字符串、Symbol/Expr 对象，或者它们的序列（可以为空）

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import vfree_group
    >>> vfree_group("x, y, z")
    <free group on the generators (x, y, z)>
    >>> x**2*y**-2*z # noqa: F821
    x**2*y**-2*z
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)  # 创建自由群对象
    pollute([sym.name for sym in _free_group.symbols], _free_group.generators)  # 将生成器注入全局命名空间
    return _free_group


def _parse_symbols(symbols):
    """Parse symbols into a sequence of Symbol/Expr objects.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr
        符号或表达式的字符串、Symbol/Expr 对象，或者它们的序列

    Raises
    ======

    ValueError
        如果 `symbols` 类型不符合预期，则抛出异常

    """
    if not symbols:
        return ()  # 如果 symbols 为空，则返回空元组
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True)  # 如果 symbols 是字符串，则调用 _symbols 函数处理并返回
    elif isinstance(symbols, (Expr, FreeGroupElement)):
        return (symbols,)  # 如果 symbols 是 Expr 或 FreeGroupElement 对象，则返回包含该对象的元组
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)  # 如果 symbols 是字符串组成的序列，则调用 _symbols 函数处理并返回
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols  # 如果 symbols 是 Expr 对象组成的序列，则直接返回
    raise ValueError("The type of `symbols` must be one of the following: "
                     "a str, Symbol/Expr or a sequence of "
                     "one of these types")  # 如果 symbols 类型不符合预期，则抛出 ValueError 异常
##############################################################################
#                          FREE GROUP                                        #
##############################################################################

# 缓存已创建的自由群对象，以哈希映射形式存储
_free_group_cache: dict[int, FreeGroup] = {}

# 定义自由群类，继承自 DefaultPrinting 类
class FreeGroup(DefaultPrinting):
    """
    Free group with finite or infinite number of generators. Its input API
    is that of a str, Symbol/Expr or a sequence of one of
    these types (which may be empty)

    See Also
    ========

    sympy.polys.rings.PolyRing

    References
    ==========

    .. [1] https://www.gap-system.org/Manuals/doc/ref/chap37.html

    .. [2] https://en.wikipedia.org/wiki/Free_group

    """
    # 标志：该类实例是一个可交换的群
    is_associative = True
    # 标志：该类实例是一个群
    is_group = True
    # 标志：该类实例是一个自由群
    is_FreeGroup = True
    # 标志：该类实例不是一个置换群
    is_PermutationGroup = False
    # 自由群的关系式列表，初始化为空列表
    relators: list[Expr] = []

    # 构造函数，创建自由群对象
    def __new__(cls, symbols):
        # 解析输入的符号，转换成元组形式
        symbols = tuple(_parse_symbols(symbols))
        # 计算符号的个数，即自由群的秩
        rank = len(symbols)
        # 计算对象的哈希值，用于缓存查找
        _hash = hash((cls.__name__, symbols, rank))
        # 从缓存中获取已存在的自由群对象
        obj = _free_group_cache.get(_hash)

        if obj is None:
            # 如果缓存中不存在，则创建新对象
            obj = object.__new__(cls)
            obj._hash = _hash
            obj._rank = rank
            # 创建 FreeGroupElement 的数据类型方法
            obj.dtype = type("FreeGroupElement", (FreeGroupElement,), {"group": obj})
            obj.symbols = symbols
            # 计算生成器列表
            obj.generators = obj._generators()
            # 生成生成器集合
            obj._gens_set = set(obj.generators)
            # 将符号与生成器对应起来，并设置为对象的属性
            for symbol, generator in zip(obj.symbols, obj.generators):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if hasattr(obj, name):
                        setattr(obj, name, generator)

            # 将新创建的自由群对象存入缓存
            _free_group_cache[_hash] = obj

        return obj

    # 返回自由群的生成器列表
    def _generators(group):
        """Returns the generators of the FreeGroup.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> F.generators
        (x, y, z)

        """
        gens = []
        for sym in group.symbols:
            elm = ((sym, 1),)
            gens.append(group.dtype(elm))
        return tuple(gens)

    # 克隆自由群对象，可选择更改符号集合
    def clone(self, symbols=None):
        return self.__class__(symbols or self.symbols)

    # 判断某个元素是否属于自由群
    def __contains__(self, i):
        """Return True if ``i`` is contained in FreeGroup."""
        if not isinstance(i, FreeGroupElement):
            return False
        group = i.group
        return self == group

    # 返回对象的哈希值
    def __hash__(self):
        return self._hash

    # 返回自由群的秩，即生成器的数量
    def __len__(self):
        return self.rank

    # 返回自由群对象的字符串表示形式
    def __str__(self):
        if self.rank > 30:
            str_form = "<free group with %s generators>" % self.rank
        else:
            str_form = "<free group on the generators "
            gens = self.generators
            str_form += str(gens) + ">"
        return str_form

    # 返回对象的字符串表示形式，用于调试
    __repr__ = __str__
    def __getitem__(self, index):
        symbols = self.symbols[index]  # 获取索引处的符号列表
        return self.clone(symbols=symbols)  # 返回一个克隆对象，使用新的符号列表

    def __eq__(self, other):
        """No ``FreeGroup`` is equal to any "other" ``FreeGroup``.
        比较当前的 FreeGroup 对象和其他对象是否相等，始终返回 False
        """
        return self is other  # 比较对象的引用是否相同，即判断对象是否相等

    def index(self, gen):
        """Return the index of the generator `gen` from ``(f_0, ..., f_(n-1))``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.index(y)
        1
        >>> F.index(x)
        0

        返回生成器 `gen` 在当前 FreeGroup 对象的生成器列表中的索引
        """
        if isinstance(gen, self.dtype):  # 检查 gen 是否属于当前 FreeGroup 的生成器类型
            return self.generators.index(gen)  # 返回生成器 gen 在列表 self.generators 中的索引
        else:
            raise ValueError("expected a generator of Free Group %s, got %s" % (self, gen))  # 抛出值错误异常

    def order(self):
        """Return the order of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.order()
        oo

        >>> free_group("")[0].order()
        1

        返回自由群的阶数
        """
        if self.rank == 0:  # 如果当前 FreeGroup 的 rank 为 0
            return S.One  # 返回 SymPy 中的表示 1 的对象
        else:
            return S.Infinity  # 返回 SymPy 中的表示无穷大的对象

    @property
    def elements(self):
        """
        Return the elements of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> (z,) = free_group("")
        >>> z.elements
        {<identity>}

        返回自由群的元素集合，如果 rank 为 0，则只包含一个表示单位元素的对象
        """
        if self.rank == 0:
            return {self.identity}  # 返回包含单位元素的集合
        else:
            raise ValueError("Group contains infinitely many elements"
                            ", hence cannot be represented")  # 如果 rank 不为 0，则抛出值错误异常

    @property
    def rank(self):
        r"""
        In group theory, the `rank` of a group `G`, denoted `G.rank`,
        can refer to the smallest cardinality of a generating set
        for G, that is

        \operatorname{rank}(G)=\min\{ |X|: X\subseteq G, \left\langle X\right\rangle =G\}.

        返回自由群的 rank，即最小的生成集的基数
        """
        return self._rank  # 返回私有属性 _rank 的值

    @property
    def is_abelian(self):
        """Returns if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.is_abelian
        False

        返回当前自由群是否为阿贝尔群（交换群）
        """
        return self.rank in (0, 1)  # 如果 rank 为 0 或 1，则返回 True，否则返回 False

    @property
    def identity(self):
        """Returns the identity element of free group."""
        return self.dtype()  # 返回自由群的单位元素，通过调用自身的 dtype 方法
    def contains(self, g):
        """
        Tests if Free Group element ``g`` belongs to self, ``G``.

        In mathematical terms, checks if ``g`` is an element of the current Free Group ``G``.
        This method verifies if ``g`` is an instance of FreeGroupElement and belongs to the same group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.contains(x**3*y**2)
        True
        """
        # 检查参数 g 是否是 FreeGroupElement 类型，如果不是则返回 False
        if not isinstance(g, FreeGroupElement):
            return False
        # 如果 g 的群组与当前 Free Group 不同，则返回 False
        elif self != g.group:
            return False
        else:
            # 如果以上条件都不满足，则返回 True，表示 g 属于当前 Free Group
            return True

    def center(self):
        """
        Returns the center of the free group `self`.

        The center of a group consists of elements that commute with all other elements in the group.
        For a free group, the center is typically the trivial group containing only the identity element.

        Returns
        =======
        set
            A set containing only the identity element of the free group `self`.
        """
        # 返回一个包含自由群 `self` 的中心元素的集合，通常只包含群的单位元素
        return {self.identity}
############################################################################
#                          FreeGroupElement                                #
############################################################################

# 定义一个类 FreeGroupElement，继承自 CantSympify、DefaultPrinting 和 tuple
class FreeGroupElement(CantSympify, DefaultPrinting, tuple):
    """Used to create elements of FreeGroup. It cannot be used directly to
    create a free group element. It is called by the `dtype` method of the
    `FreeGroup` class.

    用于创建 FreeGroup 的元素。不能直接用来创建自由群元素，而是通过 FreeGroup 类的 `dtype` 方法调用。
    """

    # 设置类属性 is_assoc_word 为 True
    is_assoc_word = True

    # 定义方法 new，用于创建新的 FreeGroupElement 对象
    def new(self, init):
        return self.__class__(init)

    # 初始化属性 _hash 为 None
    _hash = None

    # 定义方法 __hash__，计算对象的哈希值
    def __hash__(self):
        _hash = self._hash
        if _hash is None:
            # 如果 _hash 为 None，则计算哈希值并缓存
            self._hash = _hash = hash((self.group, frozenset(tuple(self))))
        return _hash

    # 定义方法 copy，返回当前对象的副本
    def copy(self):
        return self.new(self)

    # 定义属性 is_identity，判断当前元素是否为单位元（即是否为空元组）
    @property
    def is_identity(self):
        return not self.array_form

    # 定义属性 array_form，返回当前元素的数组形式表示
    @property
    def array_form(self):
        """
        SymPy provides two different internal kinds of representation
        of associative words. The first one is called the `array_form`
        which is a tuple containing `tuples` as its elements, where the
        size of each tuple is two. At the first position the tuple
        contains the `symbol-generator`, while at the second position
        of tuple contains the exponent of that generator at the position.
        Since elements (i.e. words) do not commute, the indexing of tuple
        makes that property to stay.

        The structure in ``array_form`` of ``FreeGroupElement`` is of form:

        ``( ( symbol_of_gen, exponent ), ( , ), ... ( , ) )``

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> (x*z).array_form
        ((x, 1), (z, 1))
        >>> (x**2*z*y*x**2).array_form
        ((x, 2), (z, 1), (y, 1), (x, 2))

        See Also
        ========

        letter_repr

        """
        return tuple(self)

    # 定义属性 letter_form，返回当前元素的字母形式表示
    @property
    def letter_form(self):
        """
        The letter representation of a ``FreeGroupElement`` is a tuple
        of generator symbols, with each entry corresponding to a group
        generator. Inverses of the generators are represented by
        negative generator symbols.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b, c, d = free_group("a b c d")
        >>> (a**3).letter_form
        (a, a, a)
        >>> (a**2*d**-2*a*b**-4).letter_form
        (a, a, -d, -d, a, -b, -b, -b, -b)
        >>> (a**-2*b**3*d).letter_form
        (-a, -a, b, b, b, d)

        See Also
        ========

        array_form

        """
        return tuple(flatten([(i,)*j if j > 0 else (-i,)*(-j)
                    for i, j in self.array_form]))
    # 获取元素在自由群中的指定索引的元素
    def __getitem__(self, i):
        # 获取所属的自由群
        group = self.group
        # 获取第 i 个字母形式的元素
        r = self.letter_form[i]
        # 如果 r 是符号元素，则返回其对应的数据类型
        if r.is_Symbol:
            return group.dtype(((r, 1),))
        else:
            # 否则返回其负值对应的数据类型
            return group.dtype(((-r, -1),))

    # 返回给定生成元的索引值
    def index(self, gen):
        # 如果生成元的数量不是1，则引发值错误异常
        if len(gen) != 1:
            raise ValueError()
        # 返回生成元在字母形式列表中的索引
        return (self.letter_form).index(gen.letter_form[0])

    @property
    def letter_form_elm(self):
        """
        """
        # 获取所属的自由群
        group = self.group
        # 获取字母形式列表
        r = self.letter_form
        # 返回由字母形式列表中元素转换而来的数据类型列表
        return [group.dtype(((elm,1),)) if elm.is_Symbol \
                else group.dtype(((-elm,-1),)) for elm in r]

    @property
    def ext_rep(self):
        """This is called the External Representation of ``FreeGroupElement``
        """
        # 返回数组形式扁平化后的元组作为外部表示
        return tuple(flatten(self.array_form))

    # 检查生成元是否在数组形式的首个元素中
    def __contains__(self, gen):
        return gen.array_form[0][0] in tuple([r[0] for r in self.array_form])

    # 将对象转换为字符串表示形式
    def __str__(self):
        # 如果是单位元素，则返回 "<identity>"
        if self.is_identity:
            return "<identity>"

        # 否则逐个处理数组形式的元素，构建字符串表示形式
        str_form = ""
        array_form = self.array_form
        for i in range(len(array_form)):
            if i == len(array_form) - 1:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0])
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1])
            else:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0]) + "*"
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1]) + "*"
        return str_form

    __repr__ = __str__

    # 实现自由群元素的乘法操作
    def __mul__(self, other):
        """Returns the product of elements belonging to the same ``FreeGroup``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x*y**2*y**-4
        x*y**-2
        >>> z*y**-2
        z*y**-2
        >>> x**2*y*y**-1*x**-2
        <identity>

        """
        # 获取所属的自由群
        group = self.group
        # 检查是否为同一自由群的元素
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        # 如果自身是单位元素，则返回另一个元素
        if self.is_identity:
            return other
        # 如果另一个元素是单位元素，则返回自身
        if other.is_identity:
            return self
        # 否则将两个数组形式的元素合并，并化简
        r = list(self.array_form + other.array_form)
        zero_mul_simp(r, len(self.array_form) - 1)
        return group.dtype(tuple(r))
    # 实现自定义除法运算符 `__truediv__`，返回当前元素与另一个元素的乘法的逆运算
    def __truediv__(self, other):
        # 获取当前元素所属的自由群
        group = self.group
        # 如果 `other` 不是相同自由群的 `FreeGroup` 元素，则抛出类型错误
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                            "be multiplied")
        # 返回当前元素与 `other` 的逆元素的乘积
        return self * (other.inverse())

    # 实现反向自定义除法运算符 `__rtruediv__`
    def __rtruediv__(self, other):
        # 获取当前元素所属的自由群
        group = self.group
        # 如果 `other` 不是相同自由群的 `FreeGroup` 元素，则抛出类型错误
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                            "be multiplied")
        # 返回 `other` 与当前元素的逆元素的乘积
        return other * (self.inverse())

    # 实现自定义加法运算符 `__add__`，返回 `NotImplemented` 表示不支持加法操作
    def __add__(self, other):
        return NotImplemented

    # 返回当前元素的逆元素
    def inverse(self):
        """
        返回 ``FreeGroupElement`` 元素的逆元素

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x.inverse()
        x**-1
        >>> (x*y).inverse()
        y**-1*x**-1

        """
        # 获取当前元素所属的自由群
        group = self.group
        # 将当前元素表示的数组形式取反，然后构造成相同类型的 `FreeGroup` 元素并返回
        r = tuple([(i, -j) for i, j in self.array_form[::-1]])
        return group.dtype(r)

    # 计算当前元素的阶数
    def order(self):
        """Find the order of a ``FreeGroupElement``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> (x**2*y*y**-1*x**-2).order()
        1

        """
        # 如果当前元素是单位元素，则返回 `1`
        if self.is_identity:
            return S.One
        else:
            # 否则返回无穷大表示无限阶
            return S.Infinity

    # 计算当前元素与另一个元素的交换子
    def commutator(self, other):
        """
        返回 `self` 和 `x` 的交换子: ``~x*~self*x*self``

        """
        # 获取当前元素所属的自由群
        group = self.group
        # 如果 `other` 不是相同自由群的 `FreeGroup` 元素，则抛出值错误
        if not isinstance(other, group.dtype):
            raise ValueError("commutator of only FreeGroupElement of the same "
                             "FreeGroup exists")
        else:
            # 返回交换子的计算结果
            return self.inverse() * other.inverse() * self * other

    # 用给定的字典 `words` 替换元素中的子词
    def eliminate_words(self, words, _all=False, inverse=True):
        '''
        用字典 `words` 中的值替换元素中的每个子词。
        如果 `words` 是一个列表，则用单位元素替换每个子词。

        '''
        # 设定循环标志和新元素为当前元素
        again = True
        new = self
        # 如果 `words` 是字典，则循环直到不再有变化
        if isinstance(words, dict):
            while again:
                again = False
                # 遍历 `words` 中的每个子词
                for sub in words:
                    prev = new
                    # 调用 `eliminate_word` 方法替换子词
                    new = new.eliminate_word(sub, words[sub], _all=_all, inverse=inverse)
                    # 如果替换后的元素与之前不同，则继续循环
                    if new != prev:
                        again = True
        else:
            # 如果 `words` 是列表，则循环直到不再有变化
            while again:
                again = False
                # 遍历 `words` 中的每个子词
                for sub in words:
                    prev = new
                    # 调用 `eliminate_word` 方法替换子词
                    new = new.eliminate_word(sub, _all=_all, inverse=inverse)
                    # 如果替换后的元素与之前不同，则继续循环
                    if new != prev:
                        again = True
        # 返回替换后的新元素
        return new
    def eliminate_word(self, gen, by=None, _all=False, inverse=True):
        """
        对于一个关联词 `self`，一个子词 `gen`，和一个关联词 `by`（默认为标识元素），返回通过将 `self` 中每个 `gen` 替换为 `by` 而得到的关联词。
        如果 `_all = True`，则会替换第一次替换后可能出现的 `gen`，依此类推，直到找不到更多出现。这可能不会总是终止（例如 `(x).eliminate_word(x, x**2, _all=True)`）。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> w = x**5*y*x**2*y**-4*x
        >>> w.eliminate_word( x, x**2 )
        x**10*y*x**4*y**-4*x**2
        >>> w.eliminate_word( x, y**-1 )
        y**-11
        >>> w.eliminate_word(x**5)
        y*x**2*y**-4*x
        >>> w.eliminate_word(x*y, y)
        x**4*y*x**2*y**-4*x

        See Also
        ========
        substituted_word

        """
        # 如果 `by` 未指定，则使用自身的标识元素
        if by is None:
            by = self.group.identity
        # 如果 `gen` 是独立的或等于 `by`，则直接返回自身
        if self.is_independent(gen) or gen == by:
            return self
        # 如果 `gen` 等于 `self`，则返回 `by`
        if gen == self:
            return by
        # 如果 `gen` 的逆元素等于 `by`，并且 `_all = False`，则将 `_all` 设为 `False`
        if gen**-1 == by:
            _all = False
        # 初始化操作词为 `self`
        word = self
        # 计算子词 `gen` 的长度
        l = len(gen)

        try:
            # 尝试获取 `gen` 在 `self` 中的索引
            i = word.subword_index(gen)
            k = 1
        except ValueError:
            # 如果找不到 `gen`，且不考虑逆元素，则直接返回 `word`
            if not inverse:
                return word
            try:
                # 尝试获取 `gen` 的逆元素在 `self` 中的索引
                i = word.subword_index(gen**-1)
                k = -1
            except ValueError:
                return word

        # 递归地替换子词 `gen` 为 `by`，并组装新的关联词
        word = word.subword(0, i)*by**k*word.subword(i+l, len(word)).eliminate_word(gen, by)

        # 如果 `_all=True`，则递归地进行全部替换操作
        if _all:
            return word.eliminate_word(gen, by, _all=True, inverse=inverse)
        else:
            return word

    def __len__(self):
        """
        对于一个关联词 `self`，返回其包含的字母数。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> len(w)
        13
        >>> len(a**17)
        17
        >>> len(w**0)
        0

        """
        # 返回关联词中所有字母的绝对值之和
        return sum(abs(j) for (i, j) in self)
    def __eq__(self, other):
        """
        Two associative words are equal if they are words over the
        same alphabet and if they are sequences of the same letters.
        This is equivalent to saying that the external representations
        of the words are equal.
        There is no "universal" empty word, every alphabet has its own
        empty word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")
        >>> f
        <free group on the generators (swapnil0, swapnil1)>
        >>> g, swap0, swap1 = free_group("swap0 swap1")
        >>> g
        <free group on the generators (swap0, swap1)>

        >>> swapnil0 == swapnil1
        False
        >>> swapnil0*swapnil1 == swapnil1/swapnil1*swapnil0*swapnil1
        True
        >>> swapnil0*swapnil1 == swapnil1*swapnil0
        False
        >>> swapnil1**0 == swap0**0
        False

        """
        group = self.group  # 获取当前对象所属的组
        if not isinstance(other, group.dtype):  # 如果other不是相同组的数据类型
            return False  # 返回False
        return tuple.__eq__(self, other)  # 否则调用tuple的__eq__方法比较self和other

    def __lt__(self, other):
        """
        The ordering of associative words is defined by length and
        lexicography (this ordering is called short-lex ordering), that
        is, shorter words are smaller than longer words, and words of the
        same length are compared w.r.t. the lexicographical ordering induced
        by the ordering of generators. Generators are sorted according
        to the order in which they were created. If the generators are
        invertible then each generator `g` is larger than its inverse `g^{-1}`,
        and `g^{-1}` is larger than every generator that is smaller than `g`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> b < a
        False
        >>> a < a.inverse()
        False

        """
        group = self.group  # 获取当前对象所属的组
        if not isinstance(other, group.dtype):  # 如果other不是相同组的数据类型
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                            "be compared")  # 抛出类型错误异常
        l = len(self)  # 获取self的长度
        m = len(other)  # 获取other的长度
        # implement lenlex order
        if l < m:  # 如果self比other短
            return True  # 返回True
        elif l > m:  # 如果self比other长
            return False  # 返回False
        for i in range(l):  # 遍历self的长度
            a = self[i].array_form[0]  # 获取self的第i个元素的数组形式的第一个元素
            b = other[i].array_form[0]  # 获取other的第i个元素的数组形式的第一个元素
            p = group.symbols.index(a[0])  # 获取a的第一个元素在组符号中的索引
            q = group.symbols.index(b[0])  # 获取b的第一个元素在组符号中的索引
            if p < q:  # 如果p小于q
                return True  # 返回True
            elif p > q:  # 如果p大于q
                return False  # 返回False
            elif a[1] < b[1]:  # 如果a的第二个元素小于b的第二个元素
                return True  # 返回True
            elif a[1] > b[1]:  # 如果a的第二个元素大于b的第二个元素
                return False  # 返回False
        return False  # 默认返回False

    def __le__(self, other):
        return (self == other or self < other)  # 如果self等于other或者self小于other，则返回True，否则返回False
    def __gt__(self, other):
        """
        Override greater-than comparison for FreeGroup elements.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> y**2 > x**2
        True
        >>> y*z > z*y
        False
        >>> x > x.inverse()
        True

        """
        # 获取当前元素所属的群对象
        group = self.group
        # 检查比较对象是否与当前对象类型相同，如果不同则抛出类型错误
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                             "be compared")
        # 返回是否不小于（即大于）比较对象的结果
        return not self <= other

    def __ge__(self, other):
        """
        Override greater-than-or-equal comparison for FreeGroup elements.

        """
        # 直接返回是否不小于比较对象的结果
        return not self < other

    def exponent_sum(self, gen):
        """
        Calculate the exponent sum of a generator in the associative word.

        For an associative word `self` and a generator or inverse of generator
        `gen`, ``exponent_sum`` returns the number of times `gen` appears in
        `self` minus the number of times its inverse appears in `self`. If
        neither `gen` nor its inverse occur in `self` then 0 is returned.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.exponent_sum(x)
        2
        >>> w.exponent_sum(x**-1)
        -2
        >>> w = x**2*y**4*x**-3
        >>> w.exponent_sum(x)
        -1

        See Also
        ========

        generator_count

        """
        # 检查 gen 是否是单个生成元或生成元的逆，否则抛出值错误
        if len(gen) != 1:
            raise ValueError("gen must be a generator or inverse of a generator")
        # 获取生成元的符号和指数
        s = gen.array_form[0]
        # 计算生成元在自由群元素中的出现次数之差
        return s[1]*sum(i[1] for i in self.array_form if i[0] == s[0])

    def generator_count(self, gen):
        """
        Count the occurrences of a generator in the associative word.

        For an associative word `self` and a generator `gen`,
        ``generator_count`` returns the multiplicity of generator
        `gen` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.generator_count(x)
        2
        >>> w = x**2*y**4*x**-3
        >>> w.generator_count(x)
        5

        See Also
        ========

        exponent_sum

        """
        # 检查 gen 是否是单个生成元且指数大于等于零，否则抛出值错误
        if len(gen) != 1 or gen.array_form[0][1] < 0:
            raise ValueError("gen must be a generator")
        # 获取生成元的符号和指数
        s = gen.array_form[0]
        # 计算生成元在自由群元素中的出现次数
        return s[1]*sum(abs(i[1]) for i in self.array_form if i[0] == s[0])
    def subword(self, from_i, to_j, strict=True):
        """
        For an associative word `self` and two positive integers `from_i` and
        `to_j`, `subword` returns the subword of `self` that begins at position
        `from_i` and ends at `to_j - 1`, indexing is done with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.subword(2, 6)
        a**3*b

        """
        # 获取关联词组
        group = self.group
        # 如果非严格模式，调整起始和结束位置以确保在合法范围内
        if not strict:
            from_i = max(from_i, 0)
            to_j = min(len(self), to_j)
        # 检查起始和结束位置是否有效
        if from_i < 0 or to_j > len(self):
            raise ValueError("`from_i`, `to_j` must be positive and no greater than "
                    "the length of associative word")
        # 如果结束位置小于等于起始位置，返回词组的单位元素
        if to_j <= from_i:
            return group.identity
        else:
            # 提取字母形式的子词
            letter_form = self.letter_form[from_i: to_j]
            # 将字母形式转换为数组形式
            array_form = letter_form_to_array_form(letter_form, group)
            # 返回数组形式对应的词组元素
            return group.dtype(array_form)

    def subword_index(self, word, start=0):
        '''
        Find the index of `word` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**2*b*a*b**3
        >>> w.subword_index(a*b*a*b)
        1

        '''
        # 获取词的长度
        l = len(word)
        # 获取自身和给定词的字母形式
        self_lf = self.letter_form
        word_lf = word.letter_form
        index = None
        # 从指定位置开始，搜索自身中是否包含给定词的字母形式
        for i in range(start, len(self_lf) - l + 1):
            if self_lf[i:i+l] == word_lf:
                index = i
                break
        # 如果找到索引，返回索引值；否则，抛出值错误异常
        if index is not None:
            return index
        else:
            raise ValueError("The given word is not a subword of self")

    def is_dependent(self, word):
        """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**4*y**-3).is_dependent(x**4*y**-2)
        True
        >>> (x**2*y**-1).is_dependent(x*y)
        False
        >>> (x*y**2*x*y**2).is_dependent(x*y**2)
        True
        >>> (x**12).is_dependent(x**-4)
        True

        See Also
        ========

        is_independent

        """
        # 尝试查找给定词或其逆在自身中的索引，若成功则返回 True，否则返回 False
        try:
            return self.subword_index(word) is not None
        except ValueError:
            pass
        try:
            return self.subword_index(word**-1) is not None
        except ValueError:
            return False

    def is_independent(self, word):
        """

        See Also
        ========

        is_dependent

        """
        # 判断给定词是否不依赖于自身，若是则返回 True
        return not self.is_dependent(word)
    def contains_generators(self):
        """
        返回词所包含的生成元素。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> (x**2*y**-1).contains_generators()
        {x, y}
        >>> (x**3*z).contains_generators()
        {x, z}

        """
        # 获取关联的群
        group = self.group
        # 从词的数组形式中提取出生成元素
        gens = {group.dtype(((syllable[0], 1),)) for syllable in self.array_form}
        return gens

    def cyclic_subword(self, from_i, to_j):
        """
        返回从索引 `from_i` 到 `to_j` 的循环子词。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x*y*x
        >>> w.cyclic_subword(1, 4)
        x*y

        """
        # 获取关联的群
        group = self.group
        # 获取词的长度
        l = len(self)
        # 获取词的字母形式
        letter_form = self.letter_form
        # 计算起始点所在的周期
        period1 = int(from_i / l)
        if from_i >= l:
            from_i -= l * period1
            to_j -= l * period1
        # 计算子词的长度差
        diff = to_j - from_i
        # 提取指定范围内的子词
        word = letter_form[from_i: to_j]
        # 计算结束点所在的周期
        period2 = int(to_j / l) - 1
        # 若跨越了周期边界，需要补充额外的部分
        word += letter_form * period2 + letter_form[:diff - l + from_i - l * period2]
        # 将子词转换为数组形式
        word = letter_form_to_array_form(word, group)
        return group.dtype(word)

    def cyclic_conjugates(self):
        """
        返回与词 `self` 循环共轭的所有词。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x*y*x
        >>> w.cyclic_conjugates()
        {x*y*x, y*x*y}

        """
        return {self.cyclic_subword(i, i + len(self)) for i in range(len(self))}

    def is_cyclic_conjugate(self, w):
        """
        检查词 `self` 和 `w` 是否是循环共轭的。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w1 = x*y*x
        >>> w2 = x**-1*y*x**-1
        >>> w1.is_cyclic_conjugate(w2)
        True

        """
        l1 = len(self)
        l2 = len(w)
        if l1 != l2:
            return False
        w1 = self.identity_cyclic_reduction()
        w2 = w.identity_cyclic_reduction()
        letter1 = w1.letter_form
        letter2 = w2.letter_form
        str1 = ' '.join(map(str, letter1))
        str2 = ' '.join(map(str, letter2))
        if len(str1) != len(str2):
            return False

        return str1 in str2 + ' ' + str2

    def number_syllables(self):
        """
        返回关联词 `self` 的音节数。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**-1).number_syllables()
        3

        """
        return len(self.array_form)
    def exponent_syllable(self, i):
        """
        返回关联词 `self` 的第 `i` 个音节的指数。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.exponent_syllable( 2 )
        2

        """
        return self.array_form[i][1]

    def generator_syllable(self, i):
        """
        返回参与关联词 `self` 第 `i` 个音节的生成器符号。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.generator_syllable( 3 )
        b

        """
        return self.array_form[i][0]

    def sub_syllables(self, from_i, to_j):
        """
        返回由关联词 `self` 的第 `from_i` 到 `to_j` 位置的音节组成的子词。
        `from_i` 和 `to_j` 必须为正整数，索引从0开始。

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a, b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.sub_syllables(1, 2)
        b
        >>> w.sub_syllables(3, 3)
        <identity>

        """
        if not isinstance(from_i, int) or not isinstance(to_j, int):
            raise ValueError("both arguments should be integers")
        group = self.group
        if to_j <= from_i:
            return group.identity
        else:
            r = tuple(self.array_form[from_i: to_j])
            return group.dtype(r)

    def substituted_word(self, from_i, to_j, by):
        """
        返回通过用关联词 `by` 替换从位置 `from_i` 到 `to_j - 1` 的子词而得到的关联词。
        `from_i` 和 `to_j` 必须为正整数，索引从0开始。换句话说，
        `w.substituted_word(w, from_i, to_j, by)` 是三个词的乘积：
        `w.subword(0, from_i)`, `by`, 和 `w.subword(to_j, len(w))`。

        See Also
        ========

        eliminate_word

        """
        lw = len(self)
        if from_i >= to_j or from_i > lw or to_j > lw:
            raise ValueError("values should be within bounds")

        # otherwise there are four possibilities

        # first if from=1 and to=lw then
        if from_i == 0 and to_j == lw:
            return by
        elif from_i == 0:  # second if from_i=1 (and to_j < lw) then
            return by*self.subword(to_j, lw)
        elif to_j == lw:   # third if to_j=1 (and from_i > 1) then
            return self.subword(0, from_i)*by
        else:              # finally
            return self.subword(0, from_i)*by*self.subword(to_j, lw)
    def is_cyclically_reduced(self):
        r"""Returns whether the word is cyclically reduced or not.
        A word is cyclically reduced if by forming the cycle of the
        word, the word is not reduced, i.e a word w = `a_1 ... a_n`
        is called cyclically reduced if `a_1 \ne a_n^{-1}`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**-1*x**-1).is_cyclically_reduced()
        False
        >>> (y*x**2*y**2).is_cyclically_reduced()
        True

        """
        # 检查是否为空词，空词被认为是环简化的
        if not self:
            return True
        # 返回第一个符号与最后一个符号的逆是否相等的布尔值，用来判断环简化
        return self[0] != self[-1]**-1

    def identity_cyclic_reduction(self):
        """Return a unique cyclically reduced version of the word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).identity_cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).identity_cyclic_reduction()
        x**2*y**-1

        References
        ==========

        .. [1] https://planetmath.org/cyclicallyreduced

        """
        # 复制当前词
        word = self.copy()
        # 获取词所属的自由群
        group = self.group
        # 循环直到词变为环简化形式
        while not word.is_cyclically_reduced():
            # 获取词的首部和尾部的指数
            exp1 = word.exponent_syllable(0)
            exp2 = word.exponent_syllable(-1)
            # 计算首尾合并的指数和
            r = exp1 + exp2
            # 根据合并结果选择新的表示形式
            if r == 0:
                rep = word.array_form[1: word.number_syllables() - 1]
            else:
                rep = ((word.generator_syllable(0), exp1 + exp2),) + \
                        word.array_form[1: word.number_syllables() - 1]
            # 构建新的词对象
            word = group.dtype(rep)
        # 返回环简化后的词对象
        return word
    def cyclic_reduction(self, removed=False):
        '''
        Return a cyclically reduced version of the word. Unlike
        `identity_cyclic_reduction`, this will not cyclically permute
        the reduced word - just remove the "unreduced" bits on either
        side of it. Compare the examples with those of
        `identity_cyclic_reduction`.

        When `removed` is `True`, return a tuple `(word, r)` where
        self `r` is such that before the reduction the word was either
        `r*word*r**-1`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction()
        y**-1*x**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction(removed=True)
        (y**-1*x**2, x**-3)

        '''
        # 创建一个副本以便操作
        word = self.copy()
        # 初始化一个单位元素
        g = self.group.identity
        # 循环直到单词变为循环简化形式
        while not word.is_cyclically_reduced():
            # 获取第一个和最后一个音节的指数绝对值
            exp1 = abs(word.exponent_syllable(0))
            exp2 = abs(word.exponent_syllable(-1))
            # 取两者中较小的一个作为简化的指数
            exp = min(exp1, exp2)
            # 构造起始和结束元素
            start = word[0]**abs(exp)
            end = word[-1]**abs(exp)
            # 对单词进行简化操作
            word = start**-1*word*end**-1
            # 计算简化过程中的右乘因子
            g = g*start
        # 如果需要返回简化前的单词和右乘因子
        if removed:
            return word, g
        # 否则，只返回简化后的单词
        return word

    def power_of(self, other):
        '''
        Check if `self == other**n` for some integer n.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> ((x*y)**2).power_of(x*y)
        True
        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)
        True

        '''
        # 如果 self 是单位元素，则返回 True
        if self.is_identity:
            return True

        # 获取 other 的长度
        l = len(other)
        # 如果 other 只包含一个生成元素
        if l == 1:
            # self 必须是某个生成元素的幂次方
            gens = self.contains_generators()
            s = other in gens or other**-1 in gens
            return len(gens) == 1 and s

        # 如果 self 不是循环简化的，并且是 other 的幂次方
        reduced, r1 = self.cyclic_reduction(removed=True)
        if not r1.is_identity:
            other, r2 = other.cyclic_reduction(removed=True)
            # 如果简化时的右乘因子相等，则继续检查简化后的部分
            if r1 == r2:
                return reduced.power_of(other)
            return False

        # 如果 self 的长度小于 other 或者不能整除 other 的长度
        if len(self) < l or len(self) % l:
            return False

        # 获取 self 的前缀
        prefix = self.subword(0, l)
        # 检查前缀是否等于 other 或者其逆
        if prefix == other or prefix**-1 == other:
            # 获取 self 的剩余部分
            rest = self.subword(l, len(self))
            # 继续检查剩余部分是否是 other 的幂次方
            return rest.power_of(other)
        return False
# 将数组形式的列表转换为新的数组形式列表
def letter_form_to_array_form(array_form, group):
    """
    This method converts a list given with possible repetitions of elements in
    it. It returns a new list such that repetitions of consecutive elements is
    removed and replace with a tuple element of size two such that the first
    index contains `value` and the second index contains the number of
    consecutive repetitions of `value`.
    """

    # 复制输入的数组形式列表
    a = list(array_form[:])
    # 初始化一个空的新数组
    new_array = []
    # 计数器，用于统计连续重复的次数
    n = 1
    # 获取符号组的符号列表
    symbols = group.symbols

    # 遍历数组形式列表
    for i in range(len(a)):
        # 处理最后一个元素的情况
        if i == len(a) - 1:
            # 如果当前元素与前一个相同
            if a[i] == a[i - 1]:
                # 如果当前元素的相反数存在于符号列表中
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -n))  # 添加负数表示的元素及其重复次数
                else:
                    new_array.append((a[i], n))  # 添加正数表示的元素及其重复次数
            else:
                # 如果当前元素的相反数存在于符号列表中
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -1))  # 添加负数表示的元素及其重复次数为1
                else:
                    new_array.append((a[i], 1))  # 添加正数表示的元素及其重复次数为1
            return new_array  # 返回处理后的新数组
        # 处理非最后一个元素的情况
        elif a[i] == a[i + 1]:
            n += 1  # 如果当前元素与下一个相同，增加重复次数
        else:
            # 如果当前元素的相反数存在于符号列表中
            if (-a[i]) in symbols:
                new_array.append((-a[i], -n))  # 添加负数表示的元素及其重复次数
            else:
                new_array.append((a[i], n))  # 添加正数表示的元素及其重复次数
            n = 1  # 重置重复次数为1


# 用于合并两个简化单词
def zero_mul_simp(l, index):
    """Used to combine two reduced words."""
    # 当索引在有效范围内，并且当前元素与下一个元素相同时执行循环
    while index >= 0 and index < len(l) - 1 and l[index][0] == l[index + 1][0]:
        # 计算新的指数
        exp = l[index][1] + l[index + 1][1]
        # 获取基础值
        base = l[index][0]
        l[index] = (base, exp)  # 更新列表中的元素为新的基础值和指数
        del l[index + 1]  # 删除下一个元素
        if l[index][1] == 0:
            del l[index]  # 如果指数为0，则删除当前元素
            index -= 1  # 更新索引位置
```