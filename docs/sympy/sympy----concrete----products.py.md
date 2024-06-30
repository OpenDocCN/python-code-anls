# `D:\src\scipysrc\sympy\sympy\concrete\products.py`

```
from typing import Tuple as tTuple  # 导入类型别名Tuple作为tTuple

from .expr_with_intlimits import ExprWithIntLimits  # 导入自定义模块expr_with_intlimits中的ExprWithIntLimits类
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete  # 导入自定义模块summations中的Sum, summation, _dummy_with_inherited_properties_concrete函数
from sympy.core.expr import Expr  # 从sympy核心模块中导入Expr类
from sympy.core.exprtools import factor_terms  # 从sympy核心模块exprtools中导入factor_terms函数
from sympy.core.function import Derivative  # 从sympy核心模块function中导入Derivative类
from sympy.core.mul import Mul  # 从sympy核心模块mul中导入Mul类
from sympy.core.singleton import S  # 从sympy核心模块singleton中导入S类
from sympy.core.symbol import Dummy, Symbol  # 从sympy核心模块symbol中导入Dummy, Symbol类
from sympy.functions.combinatorial.factorials import RisingFactorial  # 从sympy组合数学模块中导入RisingFactorial类
from sympy.functions.elementary.exponential import exp, log  # 从sympy初等函数模块中导入exp, log函数
from sympy.functions.special.tensor_functions import KroneckerDelta  # 从sympy张量函数模块中导入KroneckerDelta类
from sympy.polys import quo, roots  # 从sympy多项式模块中导入quo, roots函数


class Product(ExprWithIntLimits):
    r"""
    Represents unevaluated products.

    Explanation
    ===========

    ``Product`` represents a finite or infinite product, with the first
    argument being the general form of terms in the series, and the second
    argument being ``(dummy_variable, start, end)``, with ``dummy_variable``
    taking all integer values from ``start`` through ``end``. In accordance
    with long-standing mathematical convention, the end term is included in
    the product.

    Finite products
    ===============

    For finite products (and products with symbolic limits assumed to be finite)
    we follow the analogue of the summation convention described by Karr [1],
    especially definition 3 of section 1.4. The product:

    .. math::

        \prod_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \prod_{m \leq i < n} f(i) = f(m) f(m+1) \cdot \ldots \cdot f(n-2) f(n-1)

    with the upper limit value `f(n)` excluded. The product over an empty set is
    one if and only if `m = n`:

    .. math::

        \prod_{m \leq i < n} f(i) = 1  \quad \mathrm{for} \quad  m = n

    Finally, for all other products over empty sets we assume the following
    definition:

    .. math::

        \prod_{m \leq i < n} f(i) = \frac{1}{\prod_{n \leq i < m} f(i)}  \quad \mathrm{for} \quad  m > n

    It is important to note that above we define all products with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the product convention. Indeed we have:

    .. math::

        \prod_{m \leq i < n} f(i) = \prod_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import a, b, i, k, m, n, x
    >>> from sympy import Product, oo
    >>> Product(k, (k, 1, m))
    Product(k, (k, 1, m))
    >>> Product(k, (k, 1, m)).doit()
    factorial(m)
    >>> Product(k**2,(k, 1, m))
    Product(k**2, (k, 1, m))
    >>> Product(k**2,(k, 1, m)).doit()
    factorial(m)**2

    Wallis' product for pi:

    >>> W = Product(2*i/(2*i-1) * 2*i/(2*i+1), (i, 1, oo))
    >>> W
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    Direct computation currently fails:
    # 定义一个符号类，表示一个带有整数限制的表达式
    class Product(ExprWithIntLimits):
        # 初始化方法，创建一个新的 Product 对象
        def __new__(cls, function, *symbols, **assumptions):
            # 调用父类的初始化方法来创建对象
            obj = ExprWithIntLimits.__new__(cls, function, *symbols, **assumptions)
            return obj
    
    # 类的__slots__属性，用于限制对象实例可以拥有的属性，这里为空
    # 通过使用 __slots__ 可以节省内存，避免动态添加属性
    __slots__ = ()
    
    # 类型提示，表示一个元组，包含元组内部每个元素都是符号、表达式、表达式的元组
    limits: tTuple[tTuple[Symbol, Expr, Expr]]
    # 将当前对象表示为 Sum 类的重写表达式
    def _eval_rewrite_as_Sum(self, *args, **kwargs):
        # 返回 exp(Sum(log(self.function), *self.limits)) 表达式
        return exp(Sum(log(self.function), *self.limits))

    @property
    # 返回此 Product 对象的第一个参数作为 term 属性
    def term(self):
        return self._args[0]
    # 将 term 属性设置为 function 属性的别名
    function = term

    # 判断当前 Product 对象是否为零
    def _eval_is_zero(self):
        if self.has_empty_sequence:
            return False

        # 检查 term 是否为零
        z = self.term.is_zero
        if z is True:
            return True
        if self.has_finite_limits:
            # 如果有有限的极限，只有当 term 为零时，Product 才为零
            return z

    # 判断当前 Product 对象是否为扩展实数
    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True

        # 返回 function 是否为扩展实数
        return self.function.is_extended_real

    # 判断当前 Product 对象是否为正数
    def _eval_is_positive(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_positive and self.has_finite_limits:
            return True

    # 判断当前 Product 对象是否为非负数
    def _eval_is_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_nonnegative and self.has_finite_limits:
            return True

    # 判断当前 Product 对象是否为扩展非负数
    def _eval_is_extended_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_extended_nonnegative:
            return True

    # 判断当前 Product 对象是否为扩展非正数
    def _eval_is_extended_nonpositive(self):
        if self.has_empty_sequence:
            return True

    # 判断当前 Product 对象是否为有限数
    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    # 计算 Product 对象的值
    def doit(self, **hints):
        # 首先确保任何确定的限制都有与之匹配的变量假设
        reps = {}
        for xab in self.limits:
            # 使用具有继承属性的具体虚拟变量
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        # 如果存在替换，则进行替换并递归调用 doit 方法
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # 当 separate=True 时
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        # 导入 powsimp 函数
        from sympy.simplify.powsimp import powsimp
        f = self.function
        for index, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            # 如果差值为整数且为负数，则调整限制和函数
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = 1 / f

            # 计算带有给定限制的 Product 的值
            g = self._eval_product(f, (i, a, b))
            if g in (None, S.NaN):
                # 如果计算结果为 None 或 NaN，则返回简化后的表达式
                return self.func(powsimp(f), *self.limits[index:])
            else:
                f = g

        # 如果 hints 中指定了深度处理，则继续递归调用 doit 方法
        if hints.get('deep', True):
            return f.doit(**hints)
        else:
            return powsimp(f)

    # 计算当前 Product 对象的伴随
    def _eval_adjoint(self):
        if self.is_commutative:
            return self.func(self.function.adjoint(), *self.limits)
        return None

    # 计算当前 Product 对象的共轭
    def _eval_conjugate(self):
        return self.func(self.function.conjugate(), *self.limits)
    # 使用kwargs参数传递各种参数给简化函数，导入symphy简化库中的product_simplify函数
    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import product_simplify
        # 调用product_simplify函数简化当前对象self
        rv = product_simplify(self, **kwargs)
        # 如果kwargs中包含'doit'参数并且其值为True，则执行rv的doit方法，否则返回rv本身
        return rv.doit() if kwargs['doit'] else rv

    # 检查当前对象是否为可交换对象，如果是，则返回其转置后的函数形式，否则返回None
    def _eval_transpose(self):
        if self.is_commutative:
            # 返回一个新的对象，其函数为self.function的转置，限制条件为self.limits的参数列表
            return self.func(self.function.transpose(), *self.limits)
        return None

    # 对直接乘积进行评估，使用term和limits参数
    def _eval_product_direct(self, term, limits):
        (k, a, n) = limits
        # 返回一个乘积对象，其元素为term在不同k值下的求值结果，k的范围是从a到n
        return Mul(*[term.subs(k, a + i) for i in range(n - a + 1)])

    # 对导数进行评估，使用参数x
    def _eval_derivative(self, x):
        # 如果x是符号并且不在self自由符号集合中，则返回S.Zero
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero
        f, limits = self.function, list(self.limits)
        # 弹出限制条件中的最后一个限制
        limit = limits.pop(-1)
        if limits:
            # 如果还有其他限制条件，则将self.function扩展为包含这些限制条件
            f = self.func(f, *limits)
        i, a, b = limit
        # 如果x在a或b的自由符号集合中，则返回None
        if x in a.free_symbols or x in b.free_symbols:
            return None
        # 创建一个虚拟变量h
        h = Dummy()
        # 计算导数的表达式
        rv = Sum( Product(f, (i, a, h - 1)) * Product(f, (i, h + 1, b)) * Derivative(f, x, evaluate=True).subs(i, h), (h, a, b))
        return rv

    # 判断当前对象是否收敛
    def is_convergent(self):
        r"""
        See docs of :obj:`.Sum.is_convergent()` for explanation of convergence
        in SymPy.

        Explanation
        ===========

        The infinite product:

        .. math::

            \prod_{1 \leq i < \infty} f(i)

        is defined by the sequence of partial products:

        .. math::

            \prod_{i=1}^{n} f(i) = f(1) f(2) \cdots f(n)

        as n increases without bound. The product converges to a non-zero
        value if and only if the sum:

        .. math::

            \sum_{1 \leq i < \infty} \log{f(n)}

        converges.

        Examples
        ========

        >>> from sympy import Product, Symbol, cos, pi, exp, oo
        >>> n = Symbol('n', integer=True)
        >>> Product(n/(n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Product(1/n**2, (n, 1, oo)).is_convergent()
        False
        >>> Product(cos(pi/n), (n, 1, oo)).is_convergent()
        True
        >>> Product(exp(-n**2), (n, 1, oo)).is_convergent()
        False

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Infinite_product
        """
        sequence_term = self.function
        log_sum = log(sequence_term)
        lim = self.limits
        try:
            # 尝试计算log_sum在给定限制条件lim下的求和是否收敛
            is_conv = Sum(log_sum, *lim).is_convergent()
        except NotImplementedError:
            # 如果求和过程抛出NotImplementedError，则检查序列项是否绝对收敛
            if Sum(sequence_term - 1, *lim).is_absolutely_convergent() is S.true:
                return S.true
            # 抛出NotImplementedError，指出无法找到当前序列项的收敛算法
            raise NotImplementedError("The algorithm to find the product convergence of %s "
                                      "is not yet implemented" % (sequence_term))
        # 返回收敛结果
        return is_conv
    def reverse_order(expr, *indices):
        """
        Reverse the order of a limit in a Product.

        Explanation
        ===========

        ``reverse_order(expr, *indices)`` reverses some limits in the expression
        ``expr`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import gamma, Product, simplify, Sum
        >>> from sympy.abc import x, y, a, b, c, d
        >>> P = Product(x, (x, a, b))
        >>> Pr = P.reverse_order(x)
        >>> Pr
        Product(1/x, (x, b + 1, a - 1))
        >>> Pr = Pr.doit()
        >>> Pr
        1/RisingFactorial(b + 1, a - b - 1)
        >>> simplify(Pr.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), b > -1), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))
        >>> P = P.doit()
        >>> P
        RisingFactorial(a, -a + b + 1)
        >>> simplify(P.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), a > 0), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x*y, (x, a, b), (y, c, d))
        >>> S
        Sum(x*y, (x, a, b), (y, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x*y, (x, b + 1, a - 1), (y, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x*y, (x, b + 1, a - 1), (y, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255

        """
        # 将可变参数 indices 转换为列表形式
        l_indices = list(indices)

        # 将非整数类型的索引转换为相应的位置索引
        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = expr.index(indx)

        # 初始化指数和限制列表
        e = 1
        limits = []

        # 遍历表达式的限制条件
        for i, limit in enumerate(expr.limits):
            l = limit
            # 如果当前索引在要反转的列表中
            if i in l_indices:
                e = -e
                # 将限制条件反转
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        # 返回反转限制后的乘积表达式
        return Product(expr.function ** e, *limits)
# 定义一个函数用于计算乘积
def product(*args, **kwargs):
    """
    计算乘积。

    Explanation
    ===========

    符号的表示类似于求和或积分中使用的表示法。product(f, (i, a, b)) 计算函数 f
    关于变量 i 从 a 到 b 的乘积，即，

    ::

                                     b
                                   _____
        product(f(n), (i, a, b)) = |   | f(n)
                                   |   |
                                   i = a

    如果无法计算乘积，则返回一个未计算的 Product 对象。
    可以通过引入额外的符号元组来计算重复的乘积::

    Examples
    ========

    >>> from sympy import product, symbols
    >>> i, n, m, k = symbols('i n m k', integer=True)

    >>> product(i, (i, 1, k))
    factorial(k)
    >>> product(m, (i, 1, k))
    m**k
    >>> product(i, (i, 1, k), (k, 1, n))
    Product(factorial(k), (k, 1, n))

    """

    # 创建一个 Product 对象，传入参数和关键字参数
    prod = Product(*args, **kwargs)

    # 如果返回的对象是 Product 类型，则计算其值
    if isinstance(prod, Product):
        return prod.doit(deep=False)
    else:
        # 否则直接返回计算结果
        return prod
```