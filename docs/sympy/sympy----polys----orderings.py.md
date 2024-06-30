# `D:\src\scipysrc\sympy\sympy\polys\orderings.py`

```
# 引入将来的语法以支持类型注解
from __future__ import annotations

# 导出的符号列表
__all__ = ["lex", "grlex", "grevlex", "ilex", "igrlex", "igrevlex"]

# 导入符号类和可迭代工具
from sympy.core import Symbol
from sympy.utilities.iterables import iterable

# 定义单项式排序的基类
class MonomialOrder:
    """Base class for monomial orderings. """

    alias: str | None = None  # 别名，可能为字符串或None
    is_global: bool | None = None  # 是否为全局排序，可能为布尔值或None
    is_default = False  # 是否为默认排序，始终为False

    def __repr__(self):
        return self.__class__.__name__ + "()"  # 返回类名的字符串表示形式

    def __str__(self):
        return self.alias  # 返回排序的别名字符串表示形式

    def __call__(self, monomial):
        raise NotImplementedError  # 调用实例时引发未实现错误

    def __eq__(self, other):
        return self.__class__ == other.__class__  # 判断是否相等

    def __hash__(self):
        return hash(self.__class__)  # 返回哈希值

    def __ne__(self, other):
        return not (self == other)  # 判断是否不相等

# 定义词典排序的子类：字典序
class LexOrder(MonomialOrder):
    """Lexicographic order of monomials. """

    alias = 'lex'  # 别名为'lex'
    is_global = True  # 全局排序为True
    is_default = True  # 默认排序为True

    def __call__(self, monomial):
        return monomial  # 返回单项式本身

# 定义词典排序的子类：分级词典序
class GradedLexOrder(MonomialOrder):
    """Graded lexicographic order of monomials. """

    alias = 'grlex'  # 别名为'grlex'
    is_global = True  # 全局排序为True

    def __call__(self, monomial):
        return (sum(monomial), monomial)  # 返回单项式的加权和及单项式本身的元组

# 定义词典排序的子类：反向分级词典序
class ReversedGradedLexOrder(MonomialOrder):
    """Reversed graded lexicographic order of monomials. """

    alias = 'grevlex'  # 别名为'grevlex'
    is_global = True  # 全局排序为True

    def __call__(self, monomial):
        return (sum(monomial), tuple(reversed([-m for m in monomial])))  # 返回单项式的加权和及反向单项式的元组

# 定义词典排序的子类：乘积排序
class ProductOrder(MonomialOrder):
    """
    A product order built from other monomial orders.

    Given (not necessarily total) orders O1, O2, ..., On, their product order
    P is defined as M1 > M2 iff there exists i such that O1(M1) = O2(M2),
    ..., Oi(M1) = Oi(M2), O{i+1}(M1) > O{i+1}(M2).

    Product orders are typically built from monomial orders on different sets
    of variables.

    ProductOrder is constructed by passing a list of pairs
    [(O1, L1), (O2, L2), ...] where Oi are MonomialOrders and Li are callables.
    Upon comparison, the Li are passed the total monomial, and should filter
    out the part of the monomial to pass to Oi.

    Examples
    ========

    We can use a lexicographic order on x_1, x_2 and also on
    y_1, y_2, y_3, and their product on {x_i, y_i} as follows:

    >>> from sympy.polys.orderings import lex, grlex, ProductOrder
    >>> P = ProductOrder(
    ...     (lex, lambda m: m[:2]), # lex order on x_1 and x_2 of monomial
    ...     (grlex, lambda m: m[2:]) # grlex on y_1, y_2, y_3
    ... )
    >>> P((2, 1, 1, 0, 0)) > P((1, 10, 0, 2, 0))
    True

    Here the exponent `2` of `x_1` in the first monomial
    (`x_1^2 x_2 y_1`) is bigger than the exponent `1` of `x_1` in the
    second monomial (`x_1 x_2^10 y_2^2`), so the first monomial is greater
    in the product ordering.

    >>> P((2, 1, 1, 0, 0)) < P((2, 1, 0, 2, 0))
    True

    Here the exponents of `x_1` and `x_2` agree, so the grlex order on
    `y_1, y_2, y_3` is used to decide the ordering. In this case the monomial
    """
    `y_2^2` is ordered larger than `y_1`, since for the grlex order the degree
    of the monomial is most important.
    """

    # 定义一个类 ProductOrder，用于处理多项式的排序规则
    def __init__(self, *args):
        self.args = args

    # 在实例被调用时，返回一个元组，其中每个元素是对应参数经过 lambda 函数处理后的结果
    def __call__(self, monomial):
        return tuple(O(lamda(monomial)) for (O, lamda) in self.args)

    # 返回对象的规范字符串表示形式，包括类名和每个参数的规范字符串
    def __repr__(self):
        contents = [repr(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    # 返回对象的非正式字符串表示形式，包括类名和每个参数的字符串表示
    def __str__(self):
        contents = [str(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    # 检查两个 ProductOrder 对象是否相等
    def __eq__(self, other):
        if not isinstance(other, ProductOrder):
            return False
        return self.args == other.args

    # 返回对象的哈希值，用于在集合中比较和存储对象
    def __hash__(self):
        return hash((self.__class__, self.args))

    # 返回是否全局排序的属性，如果所有参数都是全局排序，则返回 True；如果都不是，则返回 False；否则返回 None
    @property
    def is_global(self):
        if all(o.is_global is True for o, _ in self.args):
            return True
        if all(o.is_global is False for o, _ in self.args):
            return False
        return None
class InverseOrder(MonomialOrder):
    """
    The "inverse" of another monomial order.

    If O is any monomial order, we can construct another monomial order iO
    such that `A >_{iO} B` if and only if `B >_O A`. This is useful for
    constructing local orders.

    Note that many algorithms only work with *global* orders.

    For example, in the inverse lexicographic order on a single variable `x`,
    high powers of `x` count as small:

    >>> from sympy.polys.orderings import lex, InverseOrder
    >>> ilex = InverseOrder(lex)
    >>> ilex((5,)) < ilex((0,))
    True
    """

    def __init__(self, O):
        self.O = O  # 存储传入的原始 monomial order

    def __str__(self):
        return "i" + str(self.O)  # 返回当前逆序 monomial order 的字符串表示形式，以'i'开头

    def __call__(self, monomial):
        def inv(l):
            if iterable(l):  # 如果l是可迭代对象
                return tuple(inv(x) for x in l)  # 递归地对l中的每个元素应用inv函数
            return -l  # 返回-l，即取负数
        return inv(self.O(monomial))  # 对传入的单项式应用原始 monomial order，并对结果取负数

    @property
    def is_global(self):
        if self.O.is_global is True:  # 如果原始 monomial order 是全局的
            return False  # 则当前逆序 monomial order 不是全局的
        if self.O.is_global is False:  # 如果原始 monomial order 不是全局的
            return True  # 则当前逆序 monomial order 是全局的
        return None  # 如果无法确定原始 monomial order 的全局性，则返回None

    def __eq__(self, other):
        return isinstance(other, InverseOrder) and other.O == self.O  # 比较两个逆序 monomial order 是否相等

    def __hash__(self):
        return hash((self.__class__, self.O))  # 返回当前逆序 monomial order 的哈希值

lex = LexOrder()  # 创建 lexicographic order
grlex = GradedLexOrder()  # 创建 graded lexicographic order
grevlex = ReversedGradedLexOrder()  # 创建 reversed graded lexicographic order
ilex = InverseOrder(lex)  # 创建 lexicographic order 的逆序
igrlex = InverseOrder(grlex)  # 创建 graded lexicographic order 的逆序
igrevlex = InverseOrder(grevlex)  # 创建 reversed graded lexicographic order 的逆序

_monomial_key = {
    'lex': lex,  # 将字符串 'lex' 映射到 lexicographic order
    'grlex': grlex,  # 将字符串 'grlex' 映射到 graded lexicographic order
    'grevlex': grevlex,  # 将字符串 'grevlex' 映射到 reversed graded lexicographic order
    'ilex': ilex,  # 将字符串 'ilex' 映射到 lexicographic order 的逆序
    'igrlex': igrlex,  # 将字符串 'igrlex' 映射到 graded lexicographic order 的逆序
    'igrevlex': igrevlex  # 将字符串 'igrevlex' 映射到 reversed graded lexicographic order 的逆序
}

def monomial_key(order=None, gens=None):
    """
    Return a function defining admissible order on monomials.

    The result of a call to :func:`monomial_key` is a function which should
    be used as a key to :func:`sorted` built-in function, to provide order
    in a set of monomials of the same length.

    Currently supported monomial orderings are:

    1. lex       - lexicographic order (default)
    2. grlex     - graded lexicographic order
    3. grevlex   - reversed graded lexicographic order
    4. ilex, igrlex, igrevlex - the corresponding inverse orders

    If the ``order`` input argument is not a string but has ``__call__``
    attribute, then it will pass through with an assumption that the
    callable object defines an admissible order on monomials.

    If the ``gens`` input argument contains a list of generators, the
    resulting key function can be used to sort SymPy ``Expr`` objects.
    """
    if order is None:
        order = lex  # 默认使用 lexicographic order

    if isinstance(order, Symbol):
        order = str(order)  # 如果order是符号，则转换为字符串表示

    if isinstance(order, str):
        try:
            order = _monomial_key[order]  # 尝试从_monomial_key中获取对应的 monomial order
        except KeyError:
            raise ValueError("supported monomial orderings are 'lex', 'grlex' and 'grevlex', got %r" % order)
    # 检查对象 order 是否具有 __call__ 属性，即是否为可调用对象（函数）
    if hasattr(order, '__call__'):
        # 如果 gens 不为 None，则定义一个内部函数 _order，用于计算表达式的多项式在给定生成元下的次数列表，并返回其次数列表经 order 函数处理后的结果
        if gens is not None:
            def _order(expr):
                return order(expr.as_poly(*gens).degree_list())
            # 返回内部函数 _order
            return _order
        # 如果 gens 为 None，则直接返回 order 函数本身
        return order
    else:
        # 如果 order 不是可调用对象，则抛出 ValueError 异常，指明单项式排序规范必须是字符串或可调用对象，实际传入的类型是 order
        raise ValueError("monomial ordering specification must be a string or a callable, got %s" % order)
class _ItemGetter:
    """Helper class to return a subsequence of values."""

    def __init__(self, seq):
        # 将传入的序列转换为元组并存储在实例中
        self.seq = tuple(seq)

    def __call__(self, m):
        # 对象被调用时，返回给定索引序列对应的元组值
        return tuple(m[idx] for idx in self.seq)

    def __eq__(self, other):
        # 检查是否与另一个 _ItemGetter 对象相等
        if not isinstance(other, _ItemGetter):
            return False
        return self.seq == other.seq

def build_product_order(arg, gens):
    """
    Build a monomial order on ``gens``.

    ``arg`` should be a tuple of iterables. The first element of each iterable
    should be a string or monomial order (will be passed to monomial_key),
    the others should be subsets of the generators. This function will build
    the corresponding product order.

    For example, build a product of two grlex orders:

    >>> from sympy.polys.orderings import build_product_order
    >>> from sympy.abc import x, y, z, t

    >>> O = build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
    >>> O((1, 2, 3, 4))
    ((3, (1, 2)), (7, (3, 4)))

    """
    gens2idx = {}
    # 将生成器列表映射到它们的索引
    for i, g in enumerate(gens):
        gens2idx[g] = i
    order = []
    # 遍历参数列表
    for expr in arg:
        name = expr[0]
        var = expr[1:]

        def makelambda(var):
            # 创建一个 lambda 函数，该函数返回给定变量对应的索引序列
            return _ItemGetter(gens2idx[g] for g in var)
        # 将每个参数的排序关键字和对应的 lambda 函数添加到顺序列表中
        order.append((monomial_key(name), makelambda(var)))
    # 使用排序顺序列表构建 ProductOrder 对象并返回
    return ProductOrder(*order)
```