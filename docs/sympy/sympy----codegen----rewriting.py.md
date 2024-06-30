# `D:\src\scipysrc\sympy\sympy\codegen\rewriting.py`

```
"""
Classes and functions useful for rewriting expressions for optimized code
generation. Some languages (or standards thereof), e.g. C99, offer specialized
math functions for better performance and/or precision.

Using the ``optimize`` function in this module, together with a collection of
rules (represented as instances of ``Optimization``), one can rewrite the
expressions for this purpose::

    >>> from sympy import Symbol, exp, log
    >>> from sympy.codegen.rewriting import optimize, optims_c99
    >>> x = Symbol('x')
    >>> optimize(3*exp(2*x) - 3, optims_c99)
    3*expm1(2*x)
    >>> optimize(exp(2*x) - 1 - exp(-33), optims_c99)
    expm1(2*x) - exp(-33)
    >>> optimize(log(3*x + 3), optims_c99)
    log1p(x) + log(3)
    >>> optimize(log(2*x + 3), optims_c99)
    log(2*x + 3)

The ``optims_c99`` imported above is tuple containing the following instances
(which may be imported from ``sympy.codegen.rewriting``):

- ``expm1_opt``
- ``log1p_opt``
- ``exp2_opt``
- ``log2_opt``
- ``log2const_opt``

"""

from sympy.core.function import expand_log          # 导入函数 expand_log 用于对数表达式的展开
from sympy.core.singleton import S                  # 导入单例对象 S
from sympy.core.symbol import Wild                  # 导入通配符 Wild
from sympy.functions.elementary.complexes import sign  # 导入复数函数 sign
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数函数 exp 和 对数函数 log
from sympy.functions.elementary.miscellaneous import (Max, Min)  # 导入杂项函数 Max 和 Min
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)  # 导入三角函数 cos, sin, sinc
from sympy.assumptions import Q, ask                 # 导入 Q 和 ask 用于处理假设
from sympy.codegen.cfunctions import log1p, log2, exp2, expm1  # 导入 C 函数库中的对数和指数相关函数
from sympy.codegen.matrix_nodes import MatrixSolve   # 导入矩阵求解相关节点
from sympy.core.expr import UnevaluatedExpr          # 导入未求值表达式
from sympy.core.power import Pow                     # 导入幂函数 Pow
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2  # 导入 NumPy 相关节点
from sympy.codegen.scipy_nodes import cosm1, powm1   # 导入 SciPy 相关节点
from sympy.core.mul import Mul                       # 导入乘法运算 Mul
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号表示
from sympy.utilities.iterables import sift           # 导入 sift 函数用于迭代操作

class Optimization:
    """ Abstract base class for rewriting optimization.

    Subclasses should implement ``__call__`` taking an expression
    as argument.

    Parameters
    ==========
    cost_function : callable returning number
    priority : number

    """
    def __init__(self, cost_function=None, priority=1):
        self.cost_function = cost_function         # 设置成本函数用于优化
        self.priority = priority                   # 设置优先级

    def cheapest(self, *args):
        return min(args, key=self.cost_function)    # 返回具有最低成本函数值的参数

class ReplaceOptim(Optimization):
    """ Rewriting optimization calling replace on expressions.

    Explanation
    ===========

    The instance can be used as a function on expressions for which
    it will apply the ``replace`` method (see
    :meth:`sympy.core.basic.Basic.replace`).

    Parameters
    ==========

    query :
        First argument passed to replace.
    value :
        Second argument passed to replace.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.rewriting import ReplaceOptim
    >>> from sympy.codegen.cfunctions import exp2
    >>> x = Symbol('x')

"""
    >>> exp2_opt = ReplaceOptim(lambda p: p.is_Pow and p.base == 2,
    ...     lambda p: exp2(p.exp))

# 创建一个 ReplaceOptim 对象，用于优化表达式，将符合特定查询条件的表达式替换为指定的值表达式
# 查询条件为：判断表达式是否为幂运算且底数为2
# 替换值为：将符合条件的表达式替换为 exp2(p.exp) 形式的表达式


    >>> exp2_opt(2**x)

# 使用 exp2_opt 对象处理 2**x 这个表达式
# 结果应为 exp2(x)


    """

    def __init__(self, query, value, **kwargs):

# ReplaceOptim 类的初始化方法
# 参数：
# - query: 用于查询的函数或 Lambda 表达式，判断是否符合替换条件
# - value: 用于替换的函数或 Lambda 表达式，执行替换操作
# - **kwargs: 其他可选参数，传递给父类的初始化方法


        super().__init__(**kwargs)

# 调用父类的初始化方法，传递任何额外的关键字参数


        self.query = query
        self.value = value

# 将传入的查询函数和替换函数保存为对象的属性，以备后续调用使用


    def __call__(self, expr):

# 定义 ReplaceOptim 对象的调用方法，使得对象可以像函数一样被调用


        return expr.replace(self.query, self.value)

# 使用对象内部保存的查询函数（self.query）和替换函数（self.value）对传入的表达式 expr 进行替换操作，并返回替换后的结果表达式
def optimize(expr, optimizations):
    """ Apply optimizations to an expression.

    Parameters
    ==========

    expr : expression
        The mathematical expression to be optimized.
    optimizations : iterable of ``Optimization`` instances
        The optimizations to be applied, sorted by ``priority`` (highest first).

    Returns
    =======

    expr : expression
        The optimized expression.

    Examples
    ========

    >>> from sympy import log, Symbol
    >>> from sympy.codegen.rewriting import optims_c99, optimize
    >>> x = Symbol('x')
    >>> optimize(log(x+3)/log(2) + log(x**2 + 1), optims_c99)
    log1p(x**2) + log2(x + 3)

    """

    # Iterate through optimizations sorted by priority (highest first)
    for optim in sorted(optimizations, key=lambda opt: opt.priority, reverse=True):
        # Apply the current optimization to the expression
        new_expr = optim(expr)
        # Update the expression with the optimized result based on cost consideration
        if optim.cost_function is None:
            expr = new_expr
        else:
            expr = optim.cheapest(expr, new_expr)
    # Return the fully optimized expression
    return expr


exp2_opt = ReplaceOptim(
    lambda p: p.is_Pow and p.base == 2,
    lambda p: exp2(p.exp)
)


_d = Wild('d', properties=[lambda x: x.is_Dummy])
_u = Wild('u', properties=[lambda x: not x.is_number and not x.is_Add])
_v = Wild('v')
_w = Wild('w')
_n = Wild('n', properties=[lambda x: x.is_number])

sinc_opt1 = ReplaceOptim(
    sin(_w)/_w, sinc(_w)
)
sinc_opt2 = ReplaceOptim(
    sin(_n*_w)/_w, _n*sinc(_n*_w)
)
sinc_opts = (sinc_opt1, sinc_opt2)

log2_opt = ReplaceOptim(
    _v*log(_w)/log(2), _v*log2(_w),
    cost_function=lambda expr: expr.count(
        lambda e: (
            e.is_Pow and e.exp.is_negative  # Counting divisions
            or (isinstance(e, (log, log2)) and not e.args[0].is_number)  # Counting transcendental functions
        )
    )
)

log2const_opt = ReplaceOptim(
    log(2)*log2(_w), log(_w)
)

logsumexp_2terms_opt = ReplaceOptim(
    lambda l: (
        isinstance(l, log)
        and l.args[0].is_Add
        and len(l.args[0].args) == 2
        and all(isinstance(t, exp) for t in l.args[0].args)
    ),
    lambda l: (
        Max(*[e.args[0] for e in l.args[0].args]) +  # Finding maximum exponent
        log1p(exp(Min(*[e.args[0] for e in l.args[0].args])))  # Computing log1p(exp(minimum exponent))
    )
)


class FuncMinusOneOptim(ReplaceOptim):
    """Specialization of ReplaceOptim for functions evaluating "f(x) - 1".

    Explanation
    ===========

    Functions that evaluate to one when x approaches zero often have better numerical stability
    when computed as a single function instead of subtracting one later.

    Parameters
    ==========

    func :
        The original function.
    func_m_1 :
        The specialized function evaluating ``func(x) - 1``.
    opportunistic : bool
        When ``True``, applies the transformation as long as it decreases the magnitude of the terms.
        When ``False``, applies the transformation only if it completely eliminates the number term.

    Examples
    ========

    """
    """
    This class optimizes expressions containing calls to a specified function `func_m_1` by replacing
    them with more efficient forms when possible.

    """
    
    def __init__(self, func, func_m_1, opportunistic=True):
        """
        Initialize the optimizer.

        Parameters:
        - func: The primary function in the expressions to optimize.
        - func_m_1: The function to replace with, for optimization.
        - opportunistic: Boolean flag to enable opportunistic substitution.

        """
        # Arbitrary weight to balance the cost function
        weight = 10  # <-- this is an arbitrary number (heuristic)
        # Initialize the superclass with conditions for optimization
        super().__init__(lambda e: e.is_Add, self.replace_in_Add,
                         cost_function=lambda expr: expr.count_ops() - weight*expr.count(func_m_1))
        self.func = func
        self.func_m_1 = func_m_1
        self.opportunistic = opportunistic

    def _group_Add_terms(self, add):
        """
        Group terms in an addition expression into categories for optimization.

        Parameters:
        - add: The addition expression to analyze.

        Returns:
        - numsum: Sum of numeric terms.
        - terms_with_func: Terms containing the function `func`.
        - other_non_num_terms: Non-numeric terms without `func`.

        """
        numbers, non_num = sift(add.args, lambda arg: arg.is_number, binary=True)
        numsum = sum(numbers)
        terms_with_func, other = sift(non_num, lambda arg: arg.has(self.func), binary=True)
        return numsum, terms_with_func, other

    def replace_in_Add(self, e):
        """
        Replace function calls in an addition expression for optimization.

        Parameters:
        - e: The expression to optimize.

        Returns:
        - e: The optimized expression after substitution.

        """
        numsum, terms_with_func, other_non_num_terms = self._group_Add_terms(e)
        if numsum == 0:
            return e
        substituted, untouched = [], []
        for with_func in terms_with_func:
            if with_func.is_Mul:
                func, coeff = sift(with_func.args, lambda arg: arg.func == self.func, binary=True)
                if len(func) == 1 and len(coeff) == 1:
                    func, coeff = func[0], coeff[0]
                else:
                    coeff = None
            elif with_func.func == self.func:
                func, coeff = with_func, S.One
            else:
                coeff = None

            if coeff is not None and coeff.is_number and sign(coeff) == -sign(numsum):
                if self.opportunistic:
                    do_substitute = abs(coeff+numsum) < abs(numsum)
                else:
                    do_substitute = coeff+numsum == 0

                if do_substitute:  # advantageous substitution
                    numsum += coeff
                    substituted.append(coeff*self.func_m_1(*func.args))
                    continue
            untouched.append(with_func)

        return e.func(numsum, *substituted, *untouched, *other_non_num_terms)

    def __call__(self, expr):
        """
        Optimize the given expression by attempting two different transformations.

        Parameters:
        - expr: The expression to optimize.

        Returns:
        - The optimized expression.

        """
        alt1 = super().__call__(expr)
        alt2 = super().__call__(expr.factor())
        return self.cheapest(alt1, alt2)
# 创建对 exp 函数的 FuncMinusOneOptim 优化对象，以用于优化 expm1 函数
expm1_opt = FuncMinusOneOptim(exp, expm1)
# 创建对 cos 函数的 FuncMinusOneOptim 优化对象，以用于优化 cosm1 函数
cosm1_opt = FuncMinusOneOptim(cos, cosm1)
# 创建对 Pow 函数的 FuncMinusOneOptim 优化对象，以用于优化 powm1 函数
powm1_opt = FuncMinusOneOptim(Pow, powm1)

# 创建用于对 log 函数进行替换优化的 ReplaceOptim 对象
log1p_opt = ReplaceOptim(
    lambda e: isinstance(e, log),  # 判断表达式是否为 log 类型
    lambda l: expand_log(l.replace(
        log, lambda arg: log(arg.factor())  # 替换 log 函数中的参数
    )).replace(log(_u+1), log1p(_u))  # 进行特定替换
)

# 创建用于扩展 Pow 函数优化的 ReplaceOptim 对象
def create_expand_pow_optimization(limit, *, base_req=lambda b: b.is_symbol):
    """ Creates an instance of :class:`ReplaceOptim` for expanding ``Pow``.

    Explanation
    ===========

    The requirements for expansions are that the base needs to be a symbol
    and the exponent needs to be an Integer (and be less than or equal to
    ``limit``).

    Parameters
    ==========

    limit : int
         The highest power which is expanded into multiplication.
    base_req : function returning bool
         Requirement on base for expansion to happen, default is to return
         the ``is_symbol`` attribute of the base.

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.codegen.rewriting import create_expand_pow_optimization
    >>> x = Symbol('x')
    >>> expand_opt = create_expand_pow_optimization(3)
    >>> expand_opt(x**5 + x**3)
    x**5 + x*x*x
    >>> expand_opt(x**5 + x**3 + sin(x)**3)
    x**5 + sin(x)**3 + x*x*x
    >>> opt2 = create_expand_pow_optimization(3, base_req=lambda b: not b.is_Function)
    >>> opt2((x+1)**2 + sin(x)**2)
    sin(x)**2 + (x + 1)*(x + 1)

    """
    return ReplaceOptim(
        lambda e: e.is_Pow and base_req(e.base) and e.exp.is_Integer and abs(e.exp) <= limit,
        lambda p: (
            UnevaluatedExpr(Mul(*([p.base]*+p.exp), evaluate=False)) if p.exp > 0 else
            1/UnevaluatedExpr(Mul(*([p.base]*-p.exp), evaluate=False))
        ))

# 创建对 MatMul 表达式进行矩阵求逆优化的 ReplaceOptim 对象
matinv_opt = ReplaceOptim(_matinv_predicate, _matinv_transform)

# 创建用于优化 log(exp(_v)+exp(_w)) 表达式的 ReplaceOptim 对象
logaddexp_opt = ReplaceOptim(log(exp(_v)+exp(_w)), logaddexp(_v, _w))
# 创建用于优化 log(Pow(2, _v)+Pow(2, _w)) 表达式的 ReplaceOptim 对象
logaddexp2_opt = ReplaceOptim(log(Pow(2, _v)+Pow(2, _w)), logaddexp2(_v, _w)*log(2))

# 创建 C99 标准的优化集合
optims_c99 = (expm1_opt, log1p_opt, exp2_opt, log2_opt, log2const_opt)
# 创建 NumPy 标准的优化集合，包含 C99 标准优化和额外的 sinc_opts 优化
optims_numpy = optims_c99 + (logaddexp_opt, logaddexp2_opt,) + sinc_opts
# 创建 SciPy 标准的优化集合，包含 cosm1_opt 和 powm1_opt 两个优化
optims_scipy = (cosm1_opt, powm1_opt)
```