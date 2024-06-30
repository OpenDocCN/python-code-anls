# `D:\src\scipysrc\sympy\sympy\calculus\singularities.py`

```
"""
Singularities
=============

This module implements algorithms for finding singularities for a function
and identifying types of functions.

The differential calculus methods in this module include methods to identify
the following function types in the given ``Interval``:
- Increasing
- Strictly Increasing
- Decreasing
- Strictly Decreasing
- Monotonic

"""

# 导入 sympy 库中必要的模块
from sympy.core.power import Pow  # 导入 Pow 类，用于处理幂运算
from sympy.core.singleton import S  # 导入 S 单例对象，表示数学中的特殊常数
from sympy.core.symbol import Symbol  # 导入 Symbol 类，用于处理符号
from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 sympy 表达式
from sympy.functions.elementary.exponential import log  # 导入 log 函数，处理对数运算
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos  # 导入三角函数
from sympy.functions.elementary.hyperbolic import (  # 导入双曲函数及其反函数
    sech, csch, coth, tanh, cosh, asech, acsch, atanh, acoth)
from sympy.utilities.misc import filldedent  # 导入 filldedent 函数，用于去除字符串缩进

# 定义函数 singularities，用于寻找给定函数的奇点
def singularities(expression, symbol, domain=None):
    """
    Find singularities of a given function.

    Parameters
    ==========

    expression : Expr
        The target function in which singularities need to be found.
    symbol : Symbol
        The symbol over the values of which the singularity in
        expression in being searched for.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``expression`` has a
        singularity. An ``EmptySet`` is returned if ``expression`` has no
        singularities for any given value of ``Symbol``.

    Raises
    ======

    NotImplementedError
        Methods for determining the singularities of this function have
        not been developed.

    Notes
    =====

    This function does not find non-isolated singularities
    nor does it find branch points of the expression.

    Currently supported functions are:
        - univariate continuous (real or complex) functions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathematical_singularity

    Examples
    ========

    >>> from sympy import singularities, Symbol, log
    >>> x = Symbol('x', real=True)
    >>> y = Symbol('y', real=False)
    >>> singularities(x**2 + x + 1, x)
    EmptySet
    >>> singularities(1/(x + 1), x)
    {-1}
    >>> singularities(1/(y**2 + 1), y)
    {-I, I}
    >>> singularities(1/(y**3 + 1), y)
    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
    >>> singularities(log(x), x)
    {0}

    """
    from sympy.solvers.solveset import solveset  # 导入 solveset 函数，用于解方程

    # 如果未指定 domain，则根据 symbol 的属性选择实数域或复数域
    if domain is None:
        domain = S.Reals if symbol.is_real else S.Complexes
    try:
        # 初始化空集合以存储奇点
        sings = S.EmptySet
        # 对表达式进行重写，将 sec, csc, cot, tan 替换为 cos
        e = expression.rewrite([sec, csc, cot, tan], cos)
        # 对表达式进行重写，将 sech, csch, coth, tanh 替换为 cosh
        e = e.rewrite([sech, csch, coth, tanh], cosh)
        # 遍历表达式中的幂运算对象
        for i in e.atoms(Pow):
            # 如果幂的指数是无穷大，则抛出未实现错误
            if i.exp.is_infinite:
                raise NotImplementedError
            # 如果幂的指数是负数
            if i.exp.is_negative:
                # XXX: 未处理指数变化符号的情况
                # 将解集添加到奇点集合中
                sings += solveset(i.base, symbol, domain)
        # 遍历表达式中的对数和反双曲函数对象
        for i in expression.atoms(log, asech, acsch):
            # 将解集添加到奇点集合中
            sings += solveset(i.args[0], symbol, domain)
        # 遍历表达式中的反双曲正切和反双曲余切对象
        for i in expression.atoms(atanh, acoth):
            # 将解集添加到奇点集合中
            sings += solveset(i.args[0] - 1, symbol, domain)
            sings += solveset(i.args[0] + 1, symbol, domain)
        # 返回所有找到的奇点集合
        return sings
    except NotImplementedError:
        # 抛出未实现错误并提供解释信息
        raise NotImplementedError(filldedent('''
            Methods for determining the singularities
            of this function have not been developed.'''))
# 定义一个帮助函数，用于检查函数单调性的辅助函数
def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    """
    Helper function for functions checking function monotonicity.

    Parameters
    ==========

    expression : Expr
        被检查的目标函数
    predicate : function
        正在测试的属性。函数接受一个整数作为导数输入，并返回一个布尔值。
        如果属性被满足则返回True，否则返回False。
    interval : Set, optional
        我们正在测试的值的范围，默认为所有实数。
    symbol : Symbol, optional
        表达式中的符号，在给定范围内变化。

    它返回一个布尔值，指示函数导数满足给定谓词的区间是否是给定区间的超集。

    Returns
    =======

    Boolean
        如果 ``predicate`` 在 ``symbol`` 在 ``range`` 上变化时对所有导数都为真，则返回True，否则返回False。

    """
    from sympy.solvers.solveset import solveset

    # 将表达式转换为Sympy表达式
    expression = sympify(expression)
    # 找到表达式中的自由符号
    free = expression.free_symbols

    if symbol is None:
        # 如果没有指定符号，且自由符号数量大于1，则抛出未实现错误
        if len(free) > 1:
            raise NotImplementedError(
                'The function has not yet been implemented'
                ' for all multivariate expressions.'
            )

    # 确定变量是指定的符号或者是唯一的自由符号
    variable = symbol or (free.pop() if free else Symbol('x'))
    # 计算表达式关于变量的导数
    derivative = expression.diff(variable)
    # 计算谓词对导数的解集
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    # 返回区间是否是解集的子集的布尔值
    return interval.is_subset(predicate_interval)


def is_increasing(expression, interval=S.Reals, symbol=None):
    """
    返回函数在给定区间内是否递增。

    Parameters
    ==========

    expression : Expr
        被检查的目标函数。
    interval : Set, optional
        我们正在测试的值的范围（默认为所有实数）。
    symbol : Symbol, optional
        表达式中的符号，在给定范围内变化。

    Returns
    =======

    Boolean
        如果 ``expression`` 在给定的 ``interval`` 内是递增的（严格递增或常数），则返回True；否则返回False。

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    """
    # 调用名为 `monotonicity_helper` 的函数，传入参数 `expression`、lambda 函数 `lambda x: x >= 0`、`interval` 和 `symbol`
    # 这个 lambda 函数用于检查 x 是否大于等于 0
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)
# 判断给定表达式在指定区间内是否严格递增的函数
def is_strictly_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly increasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import Interval, oo
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    False
    >>> is_strictly_increasing(-x**2, Interval(0, oo))
    False
    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x > 0, interval, symbol)


# 判断给定表达式在指定区间内是否递减（可以是严格递减或常数）的函数
def is_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is decreasing (either strictly decreasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x <= 0, interval, symbol)


# 判断给定表达式在指定区间内是否严格递减的函数
def is_strictly_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly decreasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    True
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)
    # 定义函数 is_strictly_decreasing，判断表达式在给定区间内是否严格递减
    interval : Set, optional
        # interval 表示测试的数值范围，默认为所有实数的集合。
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        # symbol 表示表达式中变化的符号。
        The symbol present in expression which gets varied over the given range.

    Returns
    =======
    # 返回布尔值，表示表达式是否在给定的 interval 内严格递减。
    Boolean
        True if ``expression`` is strictly decreasing in the given ``interval``,
        False otherwise.

    Examples
    =========
    # 示例展示了不同情况下 is_strictly_decreasing 函数的使用方法。

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    # 调用 monotonicity_helper 函数，使用 lambda 表达式检查是否严格小于零，传入表达式、检查条件、区间和符号。
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)
# 定义了一个函数，用于检查给定表达式在指定区间内是否单调
def is_monotonic(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is monotonic in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is monotonic in the given ``interval``,
        False otherwise.

    Raises
    ======

    NotImplementedError
        Monotonicity check has not been implemented for the queried function.

    Examples
    ========

    >>> from sympy import is_monotonic
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_monotonic(-x**2, S.Reals)
    False
    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    True

    """
    # 导入解方程的函数
    from sympy.solvers.solveset import solveset
    
    # 将输入的表达式转换为 SymPy 的表达式对象
    expression = sympify(expression)
    
    # 获取表达式中的自由符号
    free = expression.free_symbols
    
    # 如果未指定符号，并且表达式中的自由符号超过一个，则抛出未实现错误
    if symbol is None and len(free) > 1:
        raise NotImplementedError(
            'is_monotonic has not yet been implemented'
            ' for all multivariate expressions.'
        )
    
    # 确定变量为指定符号或者表达式中唯一的自由符号
    variable = symbol or (free.pop() if free else Symbol('x'))
    
    # 计算表达式的导数，并求解其在指定区间内的零点（可能的转折点）
    turning_points = solveset(expression.diff(variable), variable, interval)
    
    # 判断区间与可能的转折点的交集是否为空集
    return interval.intersection(turning_points) is S.EmptySet
```