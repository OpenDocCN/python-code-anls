# `D:\src\scipysrc\sympy\sympy\stats\error_prop.py`

```
"""Tools for arithmetic error propagation."""

# 从 itertools 导入 repeat 和 combinations 函数
from itertools import repeat, combinations

# 从 sympy 库中导入各个类和函数
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.simplify.simplify import simplify
from sympy.stats.symbolic_probability import RandomSymbol, Variance, Covariance
from sympy.stats.rv import is_random

# 定义一个 lambda 函数，用于获取变量的第一个参数或者变量本身
_arg0_or_var = lambda var: var.args[0] if len(var.args) > 0 else var

# 定义函数 variance_prop，用于符号化地传播表达式的方差
def variance_prop(expr, consts=(), include_covar=False):
    r"""Symbolically propagates variance (`\sigma^2`) for expressions.
    This is computed as as seen in [1]_.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to compute the variance for.
    consts : sequence of Symbols, optional
        Represents symbols that are known constants in the expr,
        and thus have zero variance. All symbols not in consts are
        assumed to be variant.
    include_covar : bool, optional
        Flag for whether or not to include covariances, default=False.

    Returns
    =======

    var_expr : Expr
        An expression for the total variance of the expr.
        The variance for the original symbols (e.g. x) are represented
        via instance of the Variance symbol (e.g. Variance(x)).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.stats.error_prop import variance_prop
    >>> x, y = symbols('x y')

    >>> variance_prop(x + y)
    Variance(x) + Variance(y)

    >>> variance_prop(x * y)
    x**2*Variance(y) + y**2*Variance(x)

    >>> variance_prop(exp(2*x))
    4*exp(4*x)*Variance(x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    """
    # 获取表达式的参数
    args = expr.args
    # 如果参数个数为0
    if len(args) == 0:
        # 如果表达式在常数集合中，返回零
        if expr in consts:
            return S.Zero
        # 如果表达式是随机变量，返回其方差
        elif is_random(expr):
            return Variance(expr).doit()
        # 如果表达式是符号变量，返回其随机符号的方差
        elif isinstance(expr, Symbol):
            return Variance(RandomSymbol(expr)).doit()
        else:
            return S.Zero
    # 计算每个参数的方差传播
    nargs = len(args)
    var_args = list(map(variance_prop, args, repeat(consts, nargs),
                        repeat(include_covar, nargs)))
    # 如果表达式是加法
    if isinstance(expr, Add):
        # 合并所有参数的方差
        var_expr = Add(*var_args)
        # 如果包括协方差
        if include_covar:
            # 计算所有参数之间的协方差
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() \
                     for x, y in combinations(var_args, 2)]
            var_expr += Add(*terms)
    # 如果表达式是乘法
    elif isinstance(expr, Mul):
        # 计算每个参数的贡献
        terms = [v/a**2 for a, v in zip(args, var_args)]
        var_expr = simplify(expr**2 * Add(*terms))
        # 如果包括协方差
        if include_covar:
            # 计算每对参数之间的协方差
            terms = [2*Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand()/(a*b) \
                     for (a, b), (x, y) in zip(combinations(args, 2),
                                               combinations(var_args, 2))]
            var_expr += Add(*terms)
    elif isinstance(expr, Pow):
        # 如果表达式是幂运算（Pow对象）
        b = args[1]  # 取幂运算的第二个参数
        v = var_args[0] * (expr * b / args[0])**2  # 计算方差表达式的数学运算
        var_expr = simplify(v)  # 简化计算得到的表达式
    elif isinstance(expr, exp):
        # 如果表达式是指数函数（exp对象）
        var_expr = simplify(var_args[0] * expr**2)  # 计算指数函数的方差表达式，并简化
    else:
        # 如果表达式类型不明确，无法继续处理，返回整个表达式的方差
        var_expr = Variance(expr)
    return var_expr
```