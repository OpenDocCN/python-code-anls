# `D:\src\scipysrc\sympy\sympy\integrals\integrals.py`

```
# 导入必要的类型注解模块，别名为 tTuple
from typing import Tuple as tTuple

# 导入 sympy 库中的各个具体模块和类
from sympy.concrete.expr_with_limits import AddWithLimits  # 导入带限制的表达式求和类
from sympy.core.add import Add  # 导入加法运算类
from sympy.core.basic import Basic  # 导入基本对象类
from sympy.core.containers import Tuple  # 导入元组容器类
from sympy.core.expr import Expr  # 导入表达式基类
from sympy.core.exprtools import factor_terms  # 导入表达式工具中的因式分解函数
from sympy.core.function import diff  # 导入函数求导类
from sympy.core.logic import fuzzy_bool  # 导入模糊布尔逻辑类
from sympy.core.mul import Mul  # 导入乘法运算类
from sympy.core.numbers import oo, pi  # 导入无穷大和圆周率常数
from sympy.core.relational import Ne  # 导入不等式关系类
from sympy.core.singleton import S  # 导入单例类
from sympy.core.symbol import (Dummy, Symbol, Wild)  # 导入符号类、虚拟符号类、通配符类
from sympy.core.sympify import sympify  # 导入将对象转换为 sympy 对象的函数
from sympy.functions import (  # 导入数学函数
    Piecewise,  # 分段函数
    sqrt,  # 平方根函数
    piecewise_fold,  # 折叠分段函数
    tan, cot, atan  # 正切函数、余切函数、反正切函数
)
from sympy.functions.elementary.exponential import log  # 导入对数函数
from sympy.functions.elementary.integers import floor  # 导入向下取整函数
from sympy.functions.elementary.complexes import Abs, sign  # 导入复数函数的绝对值函数和符号函数
from sympy.functions.elementary.miscellaneous import Min, Max  # 导入最小值和最大值函数
from sympy.functions.special.singularity_functions import Heaviside  # 导入海维赛德阶跃函数
from .rationaltools import ratint  # 导入有理数工具中的有理函数积分函数
from sympy.matrices import MatrixBase  # 导入矩阵基类
from sympy.polys import Poly, PolynomialError  # 导入多项式类和多项式错误类
from sympy.series.formal import FormalPowerSeries  # 导入正式幂级数类
from sympy.series.limits import limit  # 导入极限计算函数
from sympy.series.order import Order  # 导入级数展开顺序类
from sympy.tensor.functions import shape  # 导入张量形状函数
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 sympy 废弃警告异常
from sympy.utilities.iterables import is_sequence  # 导入判断是否为序列的函数
from sympy.utilities.misc import filldedent  # 导入填充去除缩进文本的函数

# 定义 Integral 类，继承自 AddWithLimits 类，表示未求值积分
class Integral(AddWithLimits):
    """Represents unevaluated integral."""

    __slots__ = ()  # 使用 __slots__ 机制，防止动态添加属性

    args: tTuple[Expr, Tuple]  # 类型注解，args 属性为元组，包含表达式和元组对象
    def __new__(cls, function, *symbols, **assumptions):
        """
        Create an unevaluated integral.

        Explanation
        ===========

        Arguments are an integrand followed by one or more limits.

        If no limits are given and there is only one free symbol in the
        expression, that symbol will be used, otherwise an error will be
        raised.

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x)
        Integral(x, x)
        >>> Integral(y)
        Integral(y, y)

        When limits are provided, they are interpreted as follows (using
        ``x`` as though it were the variable of integration):

            (x,) or x - indefinite integral
            (x, a) - "evaluate at" integral is an abstract antiderivative
            (x, a, b) - definite integral

        The ``as_dummy`` method can be used to see which symbols cannot be
        targeted by subs: those with a prepended underscore cannot be
        changed with ``subs``. (Also, the integration variables themselves --
        the first element of a limit -- can never be changed by subs.)

        >>> i = Integral(x, x)
        >>> at = Integral(x, (x, x))
        >>> i.as_dummy()
        Integral(x, x)
        >>> at.as_dummy()
        Integral(_0, (_0, x))

        """

        # This checks if the `function` object has a method `_eval_Integral` and
        # calls it if available, allowing other classes to define their own
        # behavior for Integral evaluation.
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)

        # Deprecated warning when `function` is an instance of `Poly`.
        if isinstance(function, Poly):
            sympy_deprecation_warning(
                """
                integrate(Poly) and Integral(Poly) are deprecated. Instead,
                use the Poly.integrate() method, or convert the Poly to an
                Expr first with the Poly.as_expr() method.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-integrate-poly")

        # Create a new instance of the class AddWithLimits using the provided
        # arguments and return the object.
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def __getnewargs__(self):
        # Return a tuple representing the arguments needed to recreate the
        # current Integral object.
        return (self.function,) + tuple([tuple(xab) for xab in self.limits])

    @property
    def free_symbols(self):
        """
        This method returns the symbols that will exist when the
        integral is evaluated. This is useful if one is trying to
        determine whether an integral depends on a certain
        symbol or not.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x, (x, y, 1)).free_symbols
        {y}

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.function
        sympy.concrete.expr_with_limits.ExprWithLimits.limits
        sympy.concrete.expr_with_limits.ExprWithLimits.variables
        """
        # Call the `free_symbols` method of the superclass to retrieve the set
        # of symbols that will exist when the integral is evaluated.
        return super().free_symbols
    def _eval_is_zero(self):
        # 这是一个非常简单和快速的测试，不打算对积分进行操作，只是判断是否为零。
        # 例如，Integral(sin(x), (x, 0, 2*pi)) 结果为零，但这个函数应该返回 None。
        # 类似 Mul，对于某些显而易见的情况，积分将是零，因此我们检查这些情况。
        
        # 检查函数是否本身为零
        if self.function.is_zero:
            return True
        
        got_none = False
        
        # 遍历限制条件
        for l in self.limits:
            if len(l) == 3:
                # 检查是否是一个平凡的情况，即积分上下限相等或者上下限差为零
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        
        # 获取自由符号
        free = self.function.free_symbols
        
        # 再次遍历限制条件
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                # 如果积分变量不在自由符号中，则检查第二个元素是否为零
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            
            # 将积分符号从自由符号中移除，因为它将被限制条件中的自由符号替代
            free.discard(xab[0])
            
            # 添加新的符号进入自由符号集合
            for i in xab[1:]:
                free.update(i.free_symbols)
        
        # 如果函数本身不是零且没有得到 None 的情况，则返回 False
        if self.function.is_zero is False and got_none is False:
            return False

    def _eval_lseries(self, x, logx=None, cdir=0):
        # 将积分表达式转换为虚拟变量形式
        expr = self.as_dummy()
        symb = x
        
        # 遍历限制条件，确定符号 x 是否在其中
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        
        # 对函数的级数进行求和
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        symb = x
        
        # 遍历限制条件，确定符号 x 是否在其中
        for l in self.limits:
            if x in l[1:]:
                symb = l[0]
                break
        
        # 对函数的 n 级数进行求和
        terms, order = self.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        
        # 对阶数进行符号替换
        order = [o.subs(symb, x) for o in order]
        
        # 返回积分结果加上阶数项的和乘以 x
        return integrate(terms, *self.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取序列的生成器
        series_gen = self.args[0].lseries(x)
        
        # 遍历序列的主导项
        for leading_term in series_gen:
            if leading_term != 0:
                break
        
        # 返回主导项的积分结果
        return integrate(leading_term, *self.args[1:])

    def _eval_simplify(self, **kwargs):
        # 对表达式进行因式分解
        expr = factor_terms(self)
        
        # 如果结果是积分类型，则调用 simplify 函数进行简化
        if isinstance(expr, Integral):
            from sympy.simplify.simplify import simplify
            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
        
        # 否则直接调用表达式的 simplify 方法
        return expr.simplify(**kwargs)
    def principal_value(self, **kwargs):
        """
        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval
        on the real axis.

        Explanation
        ===========

        In mathematics, the Cauchy principal value is a method for assigning values to certain improper
        integrals which would otherwise be undefined.

        Examples
        ========

        >>> from sympy import Integral, oo
        >>> from sympy.abc import x
        >>> Integral(x+1, (x, -oo, oo)).principal_value()
        oo
        >>> f = 1 / (x**3)
        >>> Integral(f, (x, -oo, oo)).principal_value()
        0
        >>> Integral(f, (x, -10, 10)).principal_value()
        0
        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()
        0

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value
        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html
        """
        # 检查限制条件是否符合要求，确保变量、下限和上限均存在
        if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
            raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate "
                             "Cauchy's principal value")

        x, a, b = self.limits[0]
        # 检查上下限是否可比较且下限小于等于上限
        if not (a.is_comparable and b.is_comparable and a <= b):
            raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate "
                             "Cauchy's principal value. Also, a and b need to be comparable.")

        # 如果下限等于上限，直接返回零
        if a == b:
            return S.Zero

        # 导入计算奇点的函数
        from sympy.calculus.singularities import singularities

        # 设定虚拟变量
        r = Dummy('r')
        f = self.function

        # 找出函数在给定区间内的所有奇点
        singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]

        # 遍历奇点列表，如果奇点在区间端点上，抛出异常
        for i in singularities_list:
            if i in (a, b):
                raise ValueError(
                    'The principal value is not defined in the given interval due to singularity at %d.' % (i))

        # 计算函数的积分
        F = integrate(f, x, **kwargs)

        # 如果积分结果中还包含积分符号，则返回自身
        if F.has(Integral):
            return self

        # 如果区间为负无穷到正无穷，应用积分的对称性质
        if a is -oo and b is oo:
            I = limit(F - F.subs(x, -x), x, oo)
        else:
            # 否则按照常规方式计算积分的主值
            I = limit(F, x, b, '-') - limit(F, x, a, '+')

        # 对于每个奇点，加上其周围的主值贡献
        for s in singularities_list:
            I += limit(((F.subs(x, s - r)) - F.subs(x, s + r)), r, 0, '+')

        # 返回计算得到的积分的主值
        return I
# 定义 integrate 函数，接受任意位置参数和关键字参数，以及一些特定参数用于控制积分行为
def integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs):
    """integrate(f, var, ...)

    .. deprecated:: 1.6

       Using ``integrate()`` with :class:`~.Poly` is deprecated. Use
       :meth:`.Poly.integrate` instead. See :ref:`deprecated-integrate-poly`.

    Explanation
    ===========

    Compute definite or indefinite integral of one or more variables
    using Risch-Norman algorithm and table lookup. This procedure is
    able to handle elementary algebraic and transcendental functions
    and also a huge class of special functions, including Airy,
    Bessel, Whittaker and Lambert.

    var can be:

    - a symbol                   -- indefinite integration
    - a tuple (symbol, a)        -- indefinite integration with result
                                    given with ``a`` replacing ``symbol``
    - a tuple (symbol, a, b)     -- definite integration

    Several variables can be specified, in which case the result is
    multiple integration. (If var is omitted and the integrand is
    univariate, the indefinite integral in that variable will be performed.)

    Indefinite integrals are returned without terms that are independent
    of the integration variables. (see examples)

    Definite improper integrals often entail delicate convergence
    conditions. Pass conds='piecewise', 'separate' or 'none' to have
    these returned, respectively, as a Piecewise function, as a separate
    result (i.e. result will be a tuple), or not at all (default is
    'piecewise').

    **Strategy**

    SymPy uses various approaches to definite integration. One method is to
    find an antiderivative for the integrand, and then use the fundamental
    theorem of calculus. Various functions are implemented to integrate
    polynomial, rational and trigonometric functions, and integrands
    containing DiracDelta terms.

    SymPy also implements the part of the Risch algorithm, which is a decision
    procedure for integrating elementary functions, i.e., the algorithm can
    either find an elementary antiderivative, or prove that one does not
    exist.  There is also a (very successful, albeit somewhat slow) general
    implementation of the heuristic Risch algorithm.  This algorithm will
    eventually be phased out as more of the full Risch algorithm is
    implemented. See the docstring of Integral._eval_integral() for more
    details on computing the antiderivative using algebraic methods.

    The option risch=True can be used to use only the (full) Risch algorithm.
    This is useful if you want to know if an elementary function has an
    elementary antiderivative.  If the indefinite Integral returned by this
    function is an instance of NonElementaryIntegral, that means that the
    Risch algorithm has proven that integral to be non-elementary.  Note that
    by default, additional methods (such as the Meijer G method outlined
    below) may be used to find an antiderivative.

    Parameters
    ==========
    *args : expressions
        The integrand(s) to integrate.
    meijerg : boolean or None, optional
        Use the Meijer G method, default is None.
    conds : string, optional
        How to treat conditions, default is 'piecewise'.
    risch : boolean or None, optional
        Use only the Risch algorithm, default is None.
    heurisch : boolean or None, optional
        Use the heuristic Risch algorithm, default is None.
    manual : boolean or None, optional
        Whether the integration should only apply manual methods, default is None.
    **kwargs : dict
        Additional keyword arguments passed to underlying integration methods.

    Returns
    =======
    Expr or Tuple
        The result of the integration.

    Raises
    ======
    ValueError
        If integration fails or is not supported.

    See Also
    ========
    sympy.integrals.integrals.Integral
    sympy.integrals.manualintegrate
    sympy.integrals.trigonometry

    Examples
    ========
    >>> from sympy import integrate, Symbol
    >>> x = Symbol('x')
    >>> integrate(x**2, x)
    x**3/3
    >>> integrate(x**2, (x, 0, 1))
    1/3

    Notes
    =====
    Currently, the integration of the Meijer G function is experimental.

    """
    # Deprecated warning about using integrate() with Poly, suggesting Poly.integrate()
    """integrate() 使用 Poly 已经被弃用，请使用 Poly.integrate() 替代。参见 :ref:`deprecated-integrate-poly`。"""
    
    # 计算一元或多元变量的定积分或不定积分，使用 Risch-Norman 算法和查表方法
    """Compute definite or indefinite integral of one or more variables
    using Risch-Norman algorithm and table lookup."""
    
    # 处理特定的积分条件，如 piecewise、separate 或 none
    """Handle special integration conditions like piecewise, separate, or none."""
    
    # 选择使用全 Risch 算法进行积分
    """Use the full Risch algorithm for integration."""
    
    # 使用启发式的 Risch 算法进行积分
    """Use the heuristic Risch algorithm for integration."""
    
    # 如果需要，只应用手动方法进行积分
    """Apply manual methods only for integration if necessary."""
    
    # 其他参数传递给底层的积分方法
    """Pass additional keyword arguments to underlying integration methods."""
    below) are tried on these integrals, as they may be expressible in terms
    of special functions, so if you only care about elementary answers, use
    risch=True.  Also note that an unevaluated Integral returned by this
    function is not necessarily a NonElementaryIntegral, even with risch=True,
    as it may just be an indication that the particular part of the Risch
    algorithm needed to integrate that function is not yet implemented.

    Another family of strategies comes from re-writing the integrand in
    terms of so-called Meijer G-functions. Indefinite integrals of a
    single G-function can always be computed, and the definite integral
    of a product of two G-functions can be computed from zero to
    infinity. Various strategies are implemented to rewrite integrands
    as G-functions, and use this information to compute integrals (see
    the ``meijerint`` module).

    The option manual=True can be used to use only an algorithm that tries
    to mimic integration by hand. This algorithm does not handle as many
    integrands as the other algorithms implemented but may return results in
    a more familiar form. The ``manualintegrate`` module has functions that
    return the steps used (see the module docstring for more information).

    In general, the algebraic methods work best for computing
    antiderivatives of (possibly complicated) combinations of elementary
    functions. The G-function methods work best for computing definite
    integrals from zero to infinity of moderately complicated
    combinations of special functions, or indefinite integrals of very
    simple combinations of special functions.

    The strategy employed by the integration code is as follows:

    - If computing a definite integral, and both limits are real,
      and at least one limit is +- oo, try the G-function method of
      definite integration first.

    - Try to find an antiderivative, using all available methods, ordered
      by performance (that is try fastest method first, slowest last; in
      particular polynomial integration is tried first, Meijer
      G-functions second to last, and heuristic Risch last).

    - If still not successful, try G-functions irrespective of the
      limits.

    The option meijerg=True, False, None can be used to, respectively:
    always use G-function methods and no others, never use G-function
    methods, or use all available methods (in order as described above).
    It defaults to None.

    Examples
    ========

    >>> from sympy import integrate, log, exp, oo
    >>> from sympy.abc import a, x, y

    >>> integrate(x*y, x)
    x**2*y/2

    >>> integrate(log(x), x)
    x*log(x) - x

    >>> integrate(log(x), (x, 1, a))
    a*log(a) - a + 1

    >>> integrate(x)
    x**2/2

    Terms that are independent of x are dropped by indefinite integration:

    >>> from sympy import sqrt
    >>> integrate(sqrt(1 + x), (x, 0, x))
    2*(x + 1)**(3/2)/3 - 2/3
    >>> integrate(sqrt(1 + x), x)
    2*(x + 1)**(3/2)/3

# 对 sqrt(1 + x) 进行 x 的积分，结果是 2*(x + 1)**(3/2)/3

    >>> integrate(x*y)
    Traceback (most recent call last):
    ...
    ValueError: specify integration variables to integrate x*y

# 尝试对 x*y 进行积分但未指定积分变量，抛出 ValueError 异常

    Note that ``integrate(x)`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

# 注意 ``integrate(x)`` 语法只在交互式会话中方便使用，在库代码中应避免使用

    >>> integrate(x**a*exp(-x), (x, 0, oo)) # same as conds='piecewise'
    Piecewise((gamma(a + 1), re(a) > -1),
        (Integral(x**a*exp(-x), (x, 0, oo)), True))

# 对 x**a*exp(-x) 进行从 0 到无穷大的积分，结果相当于设置 conds='piecewise'，返回 Piecewise((gamma(a + 1), re(a) > -1), (Integral(x**a*exp(-x), (x, 0, oo)), True))

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='none')
    gamma(a + 1)

# 对 x**a*exp(-x) 进行从 0 到无穷大的积分，设置 conds='none'，直接返回 gamma(a + 1)

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='separate')
    (gamma(a + 1), re(a) > -1)

# 对 x**a*exp(-x) 进行从 0 到无穷大的积分，设置 conds='separate'，返回 (gamma(a + 1), re(a) > -1)

    See Also
    ========

    Integral, Integral.doit

# 参见 Integral, Integral.doit

    """
    doit_flags = {
        'deep': False,
        'meijerg': meijerg,
        'conds': conds,
        'risch': risch,
        'heurisch': heurisch,
        'manual': manual
        }

# 定义 doit 方法的标志参数字典，包括 'deep': False, 'meijerg': meijerg, 'conds': conds, 'risch': risch, 'heurisch': heurisch, 'manual': manual

    integral = Integral(*args, **kwargs)

# 创建 Integral 类的实例 integral，传入的位置参数为 *args，关键字参数为 **kwargs

    if isinstance(integral, Integral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, Integral) else a
            for a in integral.args]
        return integral.func(*new_args)

# 如果 integral 是 Integral 类的实例，则调用其 doit 方法并传入 doit_flags；否则，对 integral 的每个参数进行 doit 调用（如果参数是 Integral 类型的实例），然后重新组合参数调用 integral.func
# 定义线积分函数，计算给定场和曲线的线积分
def line_integrate(field, curve, vars):
    """line_integrate(field, Curve, variables)

    Compute the line integral.

    Examples
    ========

    >>> from sympy import Curve, line_integrate, E, ln
    >>> from sympy.abc import x, y, t
    >>> C = Curve([E**t + 1, E**t - 1], (t, 0, ln(2)))
    >>> line_integrate(x + y, C, [x, y])
    3*sqrt(2)

    See Also
    ========

    sympy.integrals.integrals.integrate, Integral
    """
    # 导入 Curve 类
    from sympy.geometry import Curve
    # 将 field 转换为 SymPy 表达式
    F = sympify(field)
    # 如果 F 为空，则抛出值错误异常
    if not F:
        raise ValueError(
            "Expecting function specifying field as first argument.")
    # 如果 curve 不是 Curve 实例，则抛出值错误异常
    if not isinstance(curve, Curve):
        raise ValueError("Expecting Curve entity as second argument.")
    # 如果 vars 不是可迭代的序列，则抛出值错误异常
    if not is_sequence(vars):
        raise ValueError("Expecting ordered iterable for variables.")
    # 如果 curve 的函数数量与 vars 的长度不匹配，则抛出值错误异常
    if len(curve.functions) != len(vars):
        raise ValueError("Field variable size does not match curve dimension.")
    # 如果 curve 的参数与 vars 中的某个参数重复，则抛出值错误异常
    if curve.parameter in vars:
        raise ValueError("Curve parameter clashes with field parameters.")

    # 计算线参数函数的导数
    Ft = F
    dldt = 0
    for i, var in enumerate(vars):
        _f = curve.functions[i]
        _dn = diff(_f, curve.parameter)
        # 计算弧长的平方和
        dldt = dldt + (_dn * _dn)
        # 替换 Ft 中的 var 变量为 _f 函数
        Ft = Ft.subs(var, _f)
    # 计算最终的积分表达式，乘以弧长的平方根
    Ft = Ft * sqrt(dldt)

    # 对积分进行求解
    integral = Integral(Ft, curve.limits).doit(deep=False)
    return integral
```