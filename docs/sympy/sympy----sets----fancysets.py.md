# `D:\src\scipysrc\sympy\sympy\sets\fancysets.py`

```
from functools import reduce  # 导入 reduce 函数，用于对可迭代对象应用函数
from itertools import product  # 导入 product 函数，用于迭代生成多个可迭代对象的笛卡尔积

from sympy.core.basic import Basic  # 导入基本符号操作的基类
from sympy.core.containers import Tuple  # 导入元组类，用于处理符号表达式中的元组
from sympy.core.expr import Expr  # 导入符号表达式基类
from sympy.core.function import Lambda  # 导入 Lambda 类，用于表示匿名函数
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and  # 导入模糊逻辑运算函数
from sympy.core.mod import Mod  # 导入模运算类
from sympy.core.intfunc import igcd  # 导入整数最大公约数函数
from sympy.core.numbers import oo, Rational  # 导入无穷大和有理数类
from sympy.core.relational import Eq, is_eq  # 导入等式类和等式判断函数
from sympy.core.kind import NumberKind  # 导入数值类型类
from sympy.core.singleton import Singleton, S  # 导入单例模式和全局符号对象 S
from sympy.core.symbol import Dummy, symbols, Symbol  # 导入符号类和符号创建函数
from sympy.core.sympify import _sympify, sympify, _sympy_converter  # 导入符号化函数和转换函数
from sympy.functions.elementary.integers import ceiling, floor  # 导入向上取整和向下取整函数
from sympy.functions.elementary.trigonometric import sin, cos  # 导入正弦和余弦函数
from sympy.logic.boolalg import And, Or  # 导入布尔代数的与和或操作类
from .sets import tfn, Set, Interval, Union, FiniteSet, ProductSet, SetKind  # 导入自定义集合相关类
from sympy.utilities.misc import filldedent  # 导入填充字符串的函数

class Rationals(Set, metaclass=Singleton):
    """
    代表有理数集合。这个集合也作为单例模式下的 S.Rationals 可用。

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True  # 表示集合可迭代
    _inf = S.NegativeInfinity  # 集合的负无穷
    _sup = S.Infinity  # 集合的正无穷
    is_empty = False  # 集合非空
    is_finite_set = False  # 集合非有限集合

    def _contains(self, other):
        """
        判断是否包含某个元素。

        Parameters
        ==========

        other : Expr
            待检测的元素。

        Returns
        =======

        Boolean
            如果元素是有理数则返回 True，否则返回 False。
        """
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_rational]  # 利用 is_rational 判断是否为有理数的函数结果

    def __iter__(self):
        """
        遍历有理数集合。

        Yields
        ======

        Expr
            生成有理数集合中的元素。
        """
        yield S.Zero  # 生成 0
        yield S.One  # 生成 1
        yield S.NegativeOne  # 生成 -1
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)  # 生成 n/d 的有理数
                    yield Rational(d, n)  # 生成 d/n 的有理数
                    yield Rational(-n, d)  # 生成 -n/d 的有理数
                    yield Rational(-d, n)  # 生成 -d/n 的有理数
            d += 1

    @property
    def _boundary(self):
        """
        返回有理数集合的边界。

        Returns
        =======

        S.Reals
            实数集合。
        """
        return S.Reals  # 返回实数集合作为边界

    def _kind(self):
        """
        返回集合类型。

        Returns
        =======

        SetKind
            数值类型集合。
        """
        return SetKind(NumberKind)  # 返回数值类型集合的类型
    # 检查对象是否包含另一个表达式对象
    def _contains(self, other):
        # 如果other不是Expr类型，则返回逻辑假值S.false
        if not isinstance(other, Expr):
            return S.false
        # 如果other是正整数类型的表达式，则返回逻辑真值S.true
        elif other.is_positive and other.is_integer:
            return S.true
        # 如果other不是整数或不是正整数，则返回逻辑假值S.false
        elif other.is_integer is False or other.is_positive is False:
            return S.false

    # 判断当前对象是否为给定范围的子集
    def _eval_is_subset(self, other):
        return Range(1, oo).is_subset(other)

    # 判断当前对象是否为给定范围的超集
    def _eval_is_superset(self, other):
        return Range(1, oo).is_superset(other)

    # 迭代器方法，从_inf开始无限生成整数序列
    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    # 属性方法，返回当前对象作为边界的表示
    @property
    def _boundary(self):
        return self

    # 将当前对象表示为关系表达式，要求x是整数且在当前对象的边界内
    def as_relational(self, x):
        return And(Eq(floor(x), x), x >= self.inf, x < oo)

    # 返回当前对象的类型为NumberKind的集合类型
    def _kind(self):
        return SetKind(NumberKind)
class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero  # 设置无限下界为零

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false  # 如果参数不是表达式，则返回假
        elif other.is_integer and other.is_nonnegative:
            return S.true  # 如果参数是非负整数，则返回真
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false  # 如果参数不是整数或者不是非负数，则返回假

    def _eval_is_subset(self, other):
        return Range(oo).is_subset(other)  # 判断该集合是否为另一个集合的子集

    def _eval_is_superset(self, other):
        return Range(oo).is_superset(other)  # 判断该集合是否为另一个集合的超集


class Integers(Set, metaclass=Singleton):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the singleton ``S.Integers``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True  # 设置该集合可以迭代
    is_empty = False  # 设置该集合非空
    is_finite_set = False  # 设置该集合为无限集合

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false  # 如果参数不是表达式，则返回假
        return tfn[other.is_integer]  # 判断参数是否为整数，并返回相应结果

    def __iter__(self):
        yield S.Zero  # 生成器的起始点为0
        i = S.One  # 设置计数器从1开始
        while True:
            yield i  # 产生正整数
            yield -i  # 产生负整数
            i = i + 1  # 计数器自增

    @property
    def _inf(self):
        return S.NegativeInfinity  # 返回负无穷

    @property
    def _sup(self):
        return S.Infinity  # 返回正无穷

    @property
    def _boundary(self):
        return self  # 返回集合本身作为边界

    def _kind(self):
        return SetKind(NumberKind)  # 返回该集合的类型

    def as_relational(self, x):
        return And(Eq(floor(x), x), -oo < x, x < oo)  # 将实数 x 转换为与之对应的关系表达式

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)  # 判断该集合是否为另一个集合的子集

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)  # 判断该集合是否为另一个集合的超集


class Reals(Interval, metaclass=Singleton):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the singleton ``S.Reals``.


    Examples
    ========

    >>> from sympy import S, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    @property
    def start(self):
        return S.NegativeInfinity  # 返回负无穷

    @property
    def end(self):
        return S.Infinity  # 返回正无穷

    @property
    def left_open(self):
        return True  # 返回左开区间
    # 定义一个属性方法，返回固定值 True
    @property
    def right_open(self):
        return True

    # 定义特殊方法 __eq__，用于比较对象是否等于一个特定的 Interval 对象
    def __eq__(self, other):
        return other == Interval(S.NegativeInfinity, S.Infinity)

    # 定义特殊方法 __hash__，返回一个哈希值，使得对象在集合中可以被唯一标识
    def __hash__(self):
        return hash(Interval(S.NegativeInfinity, S.Infinity))
class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from ``imageset``.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy import FiniteSet, ImageSet, Interval

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    {1, 4, 9}

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in ``base_set`` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    {0}

    See Also
    ========

    sympy.sets.sets.imageset
    """

    def __new__(cls, flambda, *sets):
        # 检查第一个参数是否为 Lambda 函数
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        # 获取 Lambda 函数的签名
        signature = flambda.signature

        # 检查 Lambda 函数参数数量与集合数量是否匹配
        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        # 将集合转换为 Sympy 对象
        sets = [_sympify(s) for s in sets]

        # 检查集合是否都是 Set 类型
        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        # 检查 Lambda 函数的签名与集合的参数类型是否匹配
        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        # 如果 Lambda 函数是恒等函数且只有一个集合作为参数，则返回该集合
        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        # 检查 Lambda 函数的自由符号是否与集合中的任何集合元素交集为空
        if not set(flambda.variables) & flambda.expr.free_symbols:
            # 如果集合中的所有集合都是空的，则返回空集
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                # 如果集合不是空的，则返回包含 Lambda 函数表达式的有限集
                return FiniteSet(flambda.expr)

        # 如果以上条件都不满足，则调用基类构造函数
        return Basic.__new__(cls, flambda, *sets)

    # 返回 Lambda 函数
    lamda = property(lambda self: self.args[0])
    # 返回基本集合
    base_sets = property(lambda self: self.args[1:])

    @property
    def base_set(self):
        # XXX: Maybe deprecate this? It is poorly defined in handling
        # the multivariate case...
        # 获取基本集合
        sets = self.base_sets
        if len(sets) == 1:
            return sets[0]
        else:
            # 返回基本集合的笛卡尔积并展平
            return ProductSet(*sets).flatten()

    @property
    def base_pset(self):
        # 返回基本集合的笛卡尔积
        return ProductSet(*self.base_sets)

    @classmethod
    # 检查签名是否符号类型，若是则返回True
    def _check_sig(cls, sig_i, set_i):
        if sig_i.is_symbol:
            return True
        # 若集合类型是 ProductSet，则递归检查签名是否匹配其中的各个集合
        elif isinstance(set_i, ProductSet):
            sets = set_i.sets
            # 检查签名和集合数是否匹配
            if len(sig_i) != len(sets):
                return False
            # 逐个递归检查每个元组签名和对应集合是否匹配
            return all(cls._check_sig(ts, ps) for ts, ps in zip(sig_i, sets))
        else:
            # XXX: 需要一种更好的方法来检查集合是否为元组集合
            # 例如 FiniteSet 可能包含元组，ImageSet 或 ConditionSet 也是如此
            # 其他如整数、实数等不能包含元组。在这里列出可能性...
            # 目前的代码可能仅适用于 ProductSet
            return True  # 给予容错的好处

    # 迭代器方法，返回基础集合经过 lambda 函数处理后的独特值
    def __iter__(self):
        # 存储已经看到的值的集合
        already_seen = set()
        # 遍历基础集合中的每个元素
        for i in self.base_pset:
            # 应用 lambda 函数到当前元素，得到结果值
            val = self.lamda(*i)
            # 如果结果值已经在已见集合中，跳过此值
            if val in already_seen:
                continue
            else:
                # 将新的结果值加入已见集合，并 yield 返回该值
                already_seen.add(val)
                yield val

    # 检查 lambda 函数是否多变量的方法
    def _is_multivariate(self):
        # 如果 lambda 函数的变量数大于1，则认为是多变量的
        return len(self.lamda.variables) > 1
    # 检查当前对象是否包含另一个对象的方法定义
    def _contains(self, other):
        # 导入解决器相关模块
        from sympy.solvers.solveset import _solveset_multi

        # 定义函数：尝试生成符号到基础集合的映射
        def get_symsetmap(signature, base_sets):
            '''Attempt to get a map of symbols to base_sets'''
            queue = list(zip(signature, base_sets))
            symsetmap = {}
            for sig, base_set in queue:
                if sig.is_symbol:
                    symsetmap[sig] = base_set
                elif base_set.is_ProductSet:
                    sets = base_set.sets
                    if len(sig) != len(sets):
                        raise ValueError("Incompatible signature")
                    # 递归处理
                    queue.extend(zip(sig, sets))
                else:
                    # 如果出现此情况，如 sig = (x, y) 和 base_set = {(1, 2), (3, 4)}，暂时放弃处理
                    return None

            return symsetmap

        # 定义函数：查找与表达式和候选对象相关的方程式
        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # 将 other 转换为 sympy 对象
        other = _sympify(other)
        # 获取 Lambda 表达式和签名
        expr = self.lamda.expr
        sig = self.lamda.signature
        # 获取 Lambda 变量
        variables = self.lamda.variables
        # 获取基础集合
        base_sets = self.base_sets

        # 使用虚拟符号替换 ImageSet 参数，以确保它们不会与 other 中的任何内容匹配
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # 映射 other 的部分到 Lambda 表达式中的对应部分，生成方程式列表
        equations = []
        for eq in get_equations(expr, other):
            # 如果方程式无法满足？
            if eq is False:
                return S.false
            equations.append(eq)

        # 将签名中的符号映射到对应的域
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # 无法将基础集合分解为 ProductSet
            return None

        # 确定 Lambda 签名中需要解决的变量
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # 使用内部多变量解集求解
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return tfn[fuzzy_not(solnset.is_empty)]

    # 属性：检查对象是否可迭代
    @property
    def is_iterable(self):
        return all(s.is_iterable for s in self.base_sets)
    # 定义一个方法 `doit`，接受任意数量的关键字参数 `hints`
    def doit(self, **hints):
        # 从 sympy.sets.setexpr 模块导入 SetExpr 类
        from sympy.sets.setexpr import SetExpr
        # 获取 self 对象的 lamda 属性，并赋值给变量 f
        f = self.lamda
        # 获取 f 的签名信息，并赋值给变量 sig
        sig = f.signature
        # 如果签名长度为 1，且签名的第一个元素是符号，并且 f 的表达式是 Expr 类型
        if len(sig) == 1 and sig[0].is_symbol and isinstance(f.expr, Expr):
            # 获取 self 对象的 base_sets 属性的第一个元素，并赋值给 base_set
            base_set = self.base_sets[0]
            # 构造 SetExpr 对象，调用 _eval_func 方法并返回其 set 属性
            return SetExpr(base_set)._eval_func(f).set
        # 如果 self 的所有 base_sets 元素都是有限集合
        if all(s.is_FiniteSet for s in self.base_sets):
            # 使用 self.base_sets 的笛卡尔积作为参数，应用 f 函数，并返回 FiniteSet 对象
            return FiniteSet(*(f(*a) for a in product(*self.base_sets)))
        # 如果以上条件都不满足，返回 self 对象本身
        return self

    # 定义一个私有方法 `_kind`
    def _kind(self):
        # 返回一个 SetKind 对象，使用 self.lamda.expr.kind 作为参数
        return SetKind(self.lamda.expr.kind)
class Range(Set):
    """
    Represents a range of integers. Can be called as ``Range(stop)``,
    ``Range(start, stop)``, or ``Range(start, stop, step)``; when ``step`` is
    not given it defaults to 1.

    ``Range(stop)`` is the same as ``Range(0, stop, 1)`` and the stop value
    (just as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although ``Range`` is a :class:`Set` (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where ``range`` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 18}
    """


注释：

Represents a range of integers. Supports creation with different start, stop, and step parameters.
Defines behavior for infinite ranges, set operations, and symbolic ranges.
    # 定义特殊方法 __new__，用于创建新对象，参数包括类本身和其他参数
    def __new__(cls, *args):
        # 如果参数个数为1
        if len(args) == 1:
            # 如果参数是 range 类型，则引发类型错误
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # 将参数解包为 slice 对象
        slc = slice(*args)

        # 如果步长为0，则抛出值错误
        if slc.step == 0:
            raise ValueError("step cannot be 0")

        # 提取起始、终止和步长，若未指定起始则默认为0，步长默认为1
        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        
        try:
            ok = []
            # 对于起始、终止和步长，使用 sympify 转换为符号表达式
            for w in (start, stop, step):
                w = sympify(w)
                # 如果是特定的无穷大或者包含符号并且为整数
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                # 如果不是整数，则抛出值错误
                elif not w.is_Integer:
                    if w.is_infinite:
                        raise ValueError('infinite symbols not allowed')
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            # 捕获值错误，抛出详细的值错误说明
            raise ValueError(filldedent('''
    Finite arguments to Range must be integers; `imageset` can define
    other cases, e.g. use `imageset(i, i/10, Range(3))` to give
    [0, 1/10, 1/5].'''))
        
        # 重新赋值起始、终止和步长为验证后的值
        start, stop, step = ok

        # 初始化 null 标志为 False
        null = False
        # 如果任意一个参数包含符号
        if any(i.has(Symbol) for i in (start, stop, step)):
            # 计算差值
            dif = stop - start
            # 计算元素个数 n
            n = dif/step
            # 如果 n 是有理数
            if n.is_Rational:
                # 如果差值为0，则置 null 为 True
                if dif == 0:
                    null = True
                else:  # 处理其他情况
                    n = floor(n)
                    end = start + n*step
                    # 如果差值为有理数且终止值小于结束值，则结束值增加步长
                    if dif.is_Rational:
                        if (end - stop).is_negative:
                            end += step
                    else:  # 处理非有理数情况
                        if (end/stop - 1).is_negative:
                            end += step
            # 如果 n 是负无穷大，则置 null 为 True
            elif n.is_extended_negative:
                null = True
            else:
                end = stop  # 其他方法如 sup 和 reversed 必须失败
        # 如果起始值为无穷大
        elif start.is_infinite:
            # 计算跨度
            span = step*(stop - start)
            # 如果跨度为 NaN 或小于等于0，则置 null 为 True
            if span is S.NaN or span <= 0:
                null = True
            # 如果步长为整数且终止值为无穷大且步长绝对值不为1，则引发值错误
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            # 处理步长为无穷大的情况
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            # 计算元素个数 n
            n = ceiling((stop - start)/step)
            # 如果 n 小于等于0，则置 null 为 True
            if n <= 0:
                null = True
            elif oostep:
                step = S.One  # 将步长置为规范的值
                end = start + step
            else:
                end = start + n*step
        
        # 如果 null 为 True，则将起始和结束值置为 0，步长置为 1
        if null:
            start = end = S.Zero
            step = S.One
        
        # 调用父类 Basic 的 __new__ 方法创建新的对象并返回
        return Basic.__new__(cls, start, end, step)

    # 定义 start 属性，返回对象的起始值
    start = property(lambda self: self.args[0])
    # 定义 stop 属性，返回对象的终止值
    stop = property(lambda self: self.args[1])
    # 定义一个属性 step，返回 self.args[2] 的值
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        # 如果 Range 包含符号，则检查参数是否符合条件
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            # 如果 n 不是正数或者 self.args 中存在非整数或无限大的元素，则抛出异常
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                raise ValueError('invalid method for symbolic range')
        # 如果起始值等于结束值，则返回自身
        if self.start == self.stop:
            return self
        # 否则返回一个以相反顺序的等价 Range 对象
        return self.func(
            self.stop - self.step, self.start - self.step, -self.step)

    def _kind(self):
        # 返回 NumberKind 类型的 SetKind
        return SetKind(NumberKind)

    def _contains(self, other):
        # 如果起始值等于结束值，则返回 False
        if self.start == self.stop:
            return S.false
        # 如果 other 是无限大数，则返回 False
        if other.is_infinite:
            return S.false
        # 如果 other 不是整数，则返回 tfn[other.is_integer]
        if not other.is_integer:
            return tfn[other.is_integer]
        # 如果 Range 包含符号，则检查参数是否符合条件
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            # 如果 n 不是正数或者 self.args 中存在非整数或无限大的元素，则返回空
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                return
        else:
            n = self.size
        # 如果起始值是有限的，则参考值为起始值
        if self.start.is_finite:
            ref = self.start
        # 如果结束值是有限的，则参考值为结束值
        elif self.stop.is_finite:
            ref = self.stop
        else:  # 如果两者都是无限的，则 step 被 __new__ 强制为 +/- 1
            return S.true
        # 如果 n 等于 1，则返回 other 等于 self[0] 的相等关系
        if n == 1:
            return Eq(other, self[0])
        # 否则计算 (ref - other) % self.step 的结果
        res = (ref - other) % self.step
        # 如果余数为零
        if res == S.Zero:
            # 如果 Range 包含符号，则创建一个虚拟符号并用其替换 other
            if self.has(Symbol):
                d = Dummy('i')
                return self.as_relational(d).subs(d, other)
            # 否则返回 other 在 inf 和 sup 之间的关系
            return And(other >= self.inf, other <= self.sup)
        # 如果余数是整数，说明偏离了序列，返回 False
        elif res.is_Integer:
            return S.false
        # 否则返回空，表示符号化或未简化的余数模 step
        else:
            return None

    def __iter__(self):
        # 验证 Range 的大小 n
        n = self.size
        # 如果 n 不是无穷大、负无穷大或整数，则抛出类型错误
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            raise TypeError("Cannot iterate over symbolic Range")
        # 如果起始值为负无穷大或正无穷大，则抛出类型错误
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise TypeError("Cannot iterate over Range with infinite start")
        # 如果起始值不等于结束值
        elif self.start != self.stop:
            i = self.start
            # 如果 n 是无穷大，则循环生成值直到无限
            if n.is_infinite:
                while True:
                    yield i
                    i += self.step
            # 否则循环生成 n 次值
            else:
                for _ in range(n):
                    yield i
                    i += self.step
    def is_iterable(self):
        # 检查能否确定大小，用于 __iter__
        dif = self.stop - self.start
        n = dif/self.step
        # 检查 n 是否为无穷大、负无穷大或整数，如果不是则返回 False
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            return False
        # 如果起始点为负无穷大或正无穷大，则返回 False
        if self.start in [S.NegativeInfinity, S.Infinity]:
            return False
        # 检查 n 是否为扩展非负并且所有步长参数是否为整数，如果不是则返回 False
        if not (n.is_extended_nonnegative and all(i.is_integer for i in self.args)):
            return False
        # 其他情况返回 True
        return True

    def __len__(self):
        # 获取范围的长度
        rv = self.size
        # 如果长度为无穷大，则引发 ValueError 异常
        if rv is S.Infinity:
            raise ValueError('Use .size to get the length of an infinite Range')
        # 返回长度的整数值
        return int(rv)

    @property
    def size(self):
        # 获取范围的大小
        if self.start == self.stop:
            return S.Zero
        # 计算范围的差值
        dif = self.stop - self.start
        n = dif/self.step
        # 如果 n 是无限大，则返回无穷大
        if n.is_infinite:
            return S.Infinity
        # 如果 n 是扩展非负并且所有步长参数是整数，则返回 n 的绝对值的向下取整
        if n.is_extended_nonnegative and all(i.is_integer for i in self.args):
            return abs(floor(n))
        # 其他情况引发 ValueError 异常
        raise ValueError('Invalid method for symbolic Range')

    @property
    def is_finite_set(self):
        # 检查范围是否是有限集合
        if self.start.is_integer and self.stop.is_integer:
            return True
        # 检查范围的大小是否为有限值
        return self.size.is_finite

    @property
    def is_empty(self):
        # 检查范围是否为空
        try:
            return self.size.is_zero
        except ValueError:
            return None

    def __bool__(self):
        # 仅区分确定的空范围和非空/未知的空范围；返回 True 不意味着范围实际上不是空的
        b = is_eq(self.start, self.stop)
        # 如果无法确定范围是否为空，则引发 ValueError 异常
        if b is None:
            raise ValueError('cannot tell if Range is null or not')
        # 返回范围是否非空
        return not bool(b)

    @property
    def _inf(self):
        # 获取范围的下确界
        if not self:
            return S.EmptySet.inf
        # 如果范围包含符号，则检查所有参数是否为整数或无穷大
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                # 根据步长的正负性和差值的正负性返回相应的下确界
                if self.step.is_positive and dif.is_positive:
                    return self.start
                elif self.step.is_negative and dif.is_negative:
                    return self.stop - self.step
            # 如果条件不满足，则引发 ValueError 异常
            raise ValueError('invalid method for symbolic range')
        # 如果步长为正，则返回起始点；否则返回停止点减去步长
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step

    @property
    def _sup(self):
        # 获取范围的上确界
        if not self:
            return S.EmptySet.sup
        # 如果范围包含符号，则检查所有参数是否为整数或无穷大
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                # 根据步长的正负性和差值的正负性返回相应的上确界
                if self.step.is_positive and dif.is_positive:
                    return self.stop - self.step
                elif self.step.is_negative and dif.is_negative:
                    return self.start
            # 如果条件不满足，则引发 ValueError 异常
            raise ValueError('invalid method for symbolic range')
        # 如果步长为正，则返回停止点减去步长；否则返回起始点
        if self.step > 0:
            return self.stop - self.step
        else:
            return self.start
    # 返回当前对象本身，用于边界处理
    def _boundary(self):
        return self

    # 将范围表示为等式和逻辑运算符的形式
    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        
        # 如果起始值是无限的，根据实例化条件来确定 a 的值
        if self.start.is_infinite:
            assert not self.stop.is_infinite  # 通过实例化确定
            a = self.reversed.start
        else:
            a = self.start
        
        # 获取步长值
        step = self.step
        
        # 判断 x 是否符合等式 Mod(x - a, step) == 0
        in_seq = Eq(Mod(x - a, step), 0)
        
        # 判断起始值和步长是否是整数
        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
        
        # 计算范围内的元素个数 n
        n = (self.stop - self.start) / self.step
        
        # 如果 n 等于 0，返回空集合 S.EmptySet.as_relational(x)
        if n == 0:
            return S.EmptySet.as_relational(x)
        
        # 如果 n 等于 1，返回 And(Eq(x, a), ints)
        if n == 1:
            return And(Eq(x, a), ints)
        
        # 尝试获取无穷大值 inf 和 sup
        try:
            a, b = self.inf, self.sup
        except ValueError:
            a = None
        
        # 根据 a 是否为 None 来确定范围条件
        if a is not None:
            range_cond = And(
                x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b)
        else:
            a, b = self.start, self.stop - self.step
            range_cond = Or(
                And(self.step >= 1, x > a if a.is_infinite else x >= a,
                    x < b if b.is_infinite else x <= b),
                And(self.step <= -1, x < a if a.is_infinite else x <= a,
                    x > b if b.is_infinite else x >= b))
        
        # 返回最终的逻辑表达式，包括等式、整数条件和范围条件
        return And(in_seq, ints, range_cond)
# 将 range 类型映射到 sympy 中的 Range 对象的 lambda 函数定义
_sympy_converter[range] = lambda r: Range(r.start, r.stop, r.step)

# 标准化实数集合 `theta` 到区间 `[0, 2\pi)` 中的值。返回标准化后的 theta 集合。
# 对于 Interval，返回一个最多包含一个周期 `$[0, 2\pi]$` 的区间。例如，对于 theta 为 `$[0, 10\pi]$`，返回的标准化值为 `$[0, 2\pi)$`。
# 目前不支持端点非 `pi` 的倍数的区间。
def normalize_theta_set(theta):
    from sympy.functions.elementary.trigonometric import _pi_coeff

    # 如果 theta 是一个区间
    if theta.is_Interval:
        interval_len = theta.measure
        
        # 完整的一个圆
        if interval_len >= 2*S.Pi:
            # 如果长度正好是一个完整的 $2\pi$ 并且两端都开放
            if interval_len == 2*S.Pi and theta.left_open and theta.right_open:
                k = _pi_coeff(theta.start)
                return Union(Interval(0, k*S.Pi, False, True),
                             Interval(k*S.Pi, 2*S.Pi, True, True))
            # 否则返回标准化后的 $[0, 2\pi)$ 区间
            return Interval(0, 2*S.Pi, False, True)

        # 计算 theta 起始点和结束点的 pi 系数
        k_start, k_end = _pi_coeff(theta.start), _pi_coeff(theta.end)

        # 如果起始点或结束点没有 pi 系数，抛出未实现的异常
        if k_start is None or k_end is None:
            raise NotImplementedError("Normalizing theta without pi as coefficient is "
                                      "not yet implemented")
        
        # 计算新的起始点和结束点
        new_start = k_start*S.Pi
        new_end = k_end*S.Pi

        # 如果新起始点大于新结束点，则返回两个区间的联合
        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open),
                         Interval(new_start, 2*S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)

    # 如果 theta 是一个有限集合
    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = _pi_coeff(element)
            # 如果元素没有 pi 系数，抛出未实现的异常
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as '
                                          'coefficient, is not Implemented.')
            else:
                new_theta.append(k*S.Pi)
        return FiniteSet(*new_theta)
    # 如果 theta 是并集类型
    elif theta.is_Union:
        # 返回对 theta 中每个区间进行规范化后的并集
        return Union(*[normalize_theta_set(interval) for interval in theta.args])

    # 如果 theta 是实数集的子集
    elif theta.is_subset(S.Reals):
        # 抛出未实现错误，提示无法规范化这种类型的 theta
        raise NotImplementedError("Normalizing theta when, it is of type %s is not "
                                  "implemented" % type(theta))
    else:
        # 抛出值错误，提示 theta 不是一个实数集
        raise ValueError(" %s is not a real set" % (theta))
class ComplexRegion(Set):
    r"""
    Represents the Set of all Complex Numbers. It can represent a
    region of Complex Plane in both the standard forms Polar and
    Rectangular coordinates.

    * Polar Form
      Input is in the form of the ProductSet or Union of ProductSets
      of the intervals of ``r`` and ``theta``, and use the flag ``polar=True``.

      .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    * Rectangular Form
      Input is in the form of the ProductSet or Union of ProductSets
      of interval of x and y, the real and imaginary parts of the Complex numbers in a plane.
      Default input type is in rectangular form.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, S, I, Union
    >>> a = Interval(2, 3)
    >>> b = Interval(4, 6)
    >>> c1 = ComplexRegion(a*b)  # Rectangular Form
    >>> c1
    CartesianComplexRegion(ProductSet(Interval(2, 3), Interval(4, 6)))

    * c1 represents the rectangular region in complex plane
      surrounded by the coordinates (2, 4), (3, 4), (3, 6) and
      (2, 6), of the four vertices.

    >>> c = Interval(1, 8)
    >>> c2 = ComplexRegion(Union(a*b, b*c))
    >>> c2
    CartesianComplexRegion(Union(ProductSet(Interval(2, 3), Interval(4, 6)), ProductSet(Interval(4, 6), Interval(1, 8))))

    * c2 represents the Union of two rectangular regions in complex
      plane. One of them surrounded by the coordinates of c1 and
      other surrounded by the coordinates (4, 1), (6, 1), (6, 8) and
      (4, 8).

    >>> 2.5 + 4.5*I in c1
    True
    >>> 2.5 + 6.5*I in c1
    False

    >>> r = Interval(0, 1)
    >>> theta = Interval(0, 2*S.Pi)
    >>> c2 = ComplexRegion(r*theta, polar=True)  # Polar Form
    >>> c2  # unit Disk
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval.Ropen(0, 2*pi)))

    * c2 represents the region in complex plane inside the
      Unit Disk centered at the origin.

    >>> 0.5 + 0.5*I in c2
    True
    >>> 1 + 2*I in c2
    False

    >>> unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    >>> upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    >>> intersection = unit_disk.intersect(upper_half_unit_disk)
    >>> intersection
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval(0, pi)))
    >>> intersection == upper_half_unit_disk
    True

    See Also
    ========

    CartesianComplexRegion
    PolarComplexRegion
    Complexes

    """
    is_ComplexRegion = True

    def __new__(cls, sets, polar=False):
        # 根据参数 polar 的值，确定创建 CartesianComplexRegion 还是 PolarComplexRegion 对象
        if polar is False:
            return CartesianComplexRegion(sets)
        elif polar is True:
            return PolarComplexRegion(sets)
        else:
            raise ValueError("polar should be either True or False")

    @property
    def sets(self):
        """
        Return raw input sets to the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.sets
        ProductSet(Interval(2, 3), Interval(4, 5))
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.sets
        Union(ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        return self.args[0]

    @property
    def psets(self):
        """
        Return a tuple of sets (ProductSets) input of the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)),)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        if self.sets.is_ProductSet:
            # 如果 self.sets 是一个 ProductSet，初始化空元组 psets
            psets = ()
            # 将 self.sets 加入到 psets 中
            psets = psets + (self.sets, )
        else:
            # 如果 self.sets 不是 ProductSet，则获取其 args 属性
            psets = self.sets.args
        return psets

    @property
    def a_interval(self):
        """
        Return the union of intervals of `x` when, self is in
        rectangular form, or the union of intervals of `r` when
        self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.a_interval
        Interval(2, 3)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.a_interval
        Union(Interval(2, 3), Interval(4, 5))

        """
        a_interval = []
        # 遍历 self.psets 中的每个元素，获取其第一个参数，并加入到 a_interval 列表中
        for element in self.psets:
            a_interval.append(element.args[0])

        # 将 a_interval 列表中的元素进行并集操作，得到并集结果
        a_interval = Union(*a_interval)
        return a_interval

    @property
    def b_interval(self):
        """
        Return the union of intervals of `y` when, self is in
        rectangular form, or the union of intervals of `theta`
        when self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.b_interval
        Interval(4, 5)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.b_interval
        Interval(1, 7)

        """
        b_interval = []
        # 遍历 self.psets 中的每个元素，获取其第二个参数，并加入到 b_interval 列表中
        for element in self.psets:
            b_interval.append(element.args[1])

        # 将 b_interval 列表中的元素进行并集操作，得到并集结果
        b_interval = Union(*b_interval)
        return b_interval
    # 返回 self.sets 的测度（即复杂区域的测度）
    def _measure(self):
        return self.sets._measure

    # 返回 self.args[0] 的 kind 属性，表示对象类型
    def _kind(self):
        return self.args[0].kind

    @classmethod
    def from_real(cls, sets):
        """
        将给定的实数子集转换为复杂区域对象。

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion
        >>> unit = Interval(0,1)
        >>> ComplexRegion.from_real(unit)
        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))

        """
        if not sets.is_subset(S.Reals):
            raise ValueError("sets must be a subset of the real line")

        # 返回一个以 sets 和 {0} 为因子的 CartesianComplexRegion 对象
        return CartesianComplexRegion(sets * FiniteSet(0))

    def _contains(self, other):
        from sympy.functions import arg, Abs

        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

        # 如果 other 不是 Expr 或 Tuple，则返回 S.false
        if not isinstance(other, (Expr, Tuple)):
            return S.false

        # 如果是直角坐标形式
        if not self.polar:
            # 将 other 拆分为实部和虚部 re, im
            re, im = other if isTuple else other.as_real_imag()
            # 检查是否满足任意一个 pset 的条件
            return tfn[fuzzy_or(fuzzy_and([
                pset.args[0]._contains(re),
                pset.args[1]._contains(im)])
                for pset in self.psets)]

        # 如果是极坐标形式
        elif self.polar:
            if other.is_zero:
                # 忽略未定义的复数参数
                return tfn[fuzzy_or(pset.args[0]._contains(S.Zero)
                    for pset in self.psets)]
            if isTuple:
                r, theta = other
            else:
                r, theta = Abs(other), arg(other)
            if theta.is_real and theta.is_number:
                # 角度 theta 规范化到 [0, 2pi)
                theta %= 2*S.Pi
                # 检查是否满足任意一个 pset 的条件
                return tfn[fuzzy_or(fuzzy_and([
                    pset.args[0]._contains(r),
                    pset.args[1]._contains(theta)])
                    for pset in self.psets)]
class CartesianComplexRegion(ComplexRegion):
    r"""
    Set representing a square region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, I, Interval
    >>> region = ComplexRegion(Interval(1, 3) * Interval(4, 6))
    >>> 2 + 5*I in region
    True
    >>> 5*I in region
    False

    See also
    ========

    ComplexRegion
    PolarComplexRegion
    Complexes
    """

    polar = False
    variables = symbols('x, y', cls=Dummy)

    def __new__(cls, sets):
        # Check if the Cartesian product of real sets is the entire complex plane
        if sets == S.Reals*S.Reals:
            return S.Complexes

        # Check if sets consist of two FiniteSets
        if all(_a.is_FiniteSet for _a in sets.args) and (len(sets.args) == 2):

            # ** ProductSet of FiniteSets in the Complex Plane. **
            # For Cases like ComplexRegion({2, 4}*{3}), It
            # would return {2 + 3*I, 4 + 3*I}

            # FIXME: This should probably be handled with something like:
            # return ImageSet(Lambda((x, y), x+I*y), sets).rewrite(FiniteSet)

            # Generate list of complex numbers from Cartesian product of FiniteSets
            complex_num = []
            for x in sets.args[0]:
                for y in sets.args[1]:
                    complex_num.append(x + S.ImaginaryUnit*y)
            return FiniteSet(*complex_num)
        else:
            return Set.__new__(cls, sets)

    @property
    def expr(self):
        x, y = self.variables
        return x + S.ImaginaryUnit*y


class PolarComplexRegion(ComplexRegion):
    r"""
    Set representing a polar region of the complex plane.

    .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, oo, pi, I
    >>> rset = Interval(0, oo)
    >>> thetaset = Interval(0, pi)
    >>> upper_half_plane = ComplexRegion(rset * thetaset, polar=True)
    >>> 1 + I in upper_half_plane
    True
    >>> 1 - I in upper_half_plane
    False

    See also
    ========

    ComplexRegion
    CartesianComplexRegion
    Complexes

    """

    polar = True
    variables = symbols('r, theta', cls=Dummy)

    def __new__(cls, sets):
        new_sets = []
        # If sets is not a ProductSet, convert each element to a list
        if not sets.is_ProductSet:
            for k in sets.args:
                new_sets.append(k)
        # If sets is a ProductSet, directly append it to new_sets
        else:
            new_sets.append(sets)
        
        # Normalize theta input
        for k, v in enumerate(new_sets):
            new_sets[k] = ProductSet(v.args[0],
                                     normalize_theta_set(v.args[1]))
        
        # Unionize the sets and return a new instance of the class
        sets = Union(*new_sets)
        return Set.__new__(cls, sets)

    @property
    def expr(self):
        r, theta = self.variables
        return r*(cos(theta) + S.ImaginaryUnit*sin(theta))


class Complexes(CartesianComplexRegion, metaclass=Singleton):
    """
    The :class:`Set` of all complex numbers

    Examples
    ========
    
    # No examples provided in the snippet for this class
    # 导入 sympy 库中的 S 和 I 符号
    >>> from sympy import S, I
    # 访问 S.Complexes 表示复数集合
    >>> S.Complexes
    Complexes
    # 检查复数 1 + I 是否属于 S.Complexes
    >>> 1 + I in S.Complexes
    True

    # 下面的部分是类的文档字符串，描述了该类的作用和相关的内容
    # 参考资料还包括 Reals 和 ComplexRegion

    is_empty = False  # 设定 is_empty 属性为 False，表示集合非空
    is_finite_set = False  # 设定 is_finite_set 属性为 False，表示集合不是有限集合

    # 由于 Complexes 类没有参数，因此覆盖了超类的属性
    @property
    def sets(self):
        # 返回一个 ProductSet，其中元素来自 S.Reals 的笛卡尔积
        return ProductSet(S.Reals, S.Reals)

    def __new__(cls):
        # 创建 Set 类的新实例并返回
        return Set.__new__(cls)
```