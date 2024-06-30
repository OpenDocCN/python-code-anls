# `D:\src\scipysrc\sympy\sympy\concrete\delta.py`

```
"""
This module implements sums and products containing the Kronecker Delta function.

References
==========

.. [1] https://mathworld.wolfram.com/KroneckerDelta.html

"""
from .products import product  # 导入 product 模块，用于处理乘积
from .summations import Sum, summation  # 导入 Sum 和 summation 模块，用于处理求和
from sympy.core import Add, Mul, S, Dummy  # 导入 SymPy 核心模块的一些类和函数
from sympy.core.cache import cacheit  # 导入缓存装饰器 cacheit
from sympy.core.sorting import default_sort_key  # 导入默认排序关键字函数 default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold  # 导入 KroneckerDelta 和 Piecewise 等函数
from sympy.polys.polytools import factor  # 导入因式分解函数 factor
from sympy.sets.sets import Interval  # 导入 Interval 类
from sympy.solvers.solvers import solve  # 导入 solve 函数


@cacheit
def _expand_delta(expr, index):
    """
    Expand the first Add containing a simple KroneckerDelta.
    """
    if not expr.is_Mul:
        return expr
    delta = None
    func = Add
    terms = [S.One]
    for h in expr.args:
        if delta is None and h.is_Add and _has_simple_delta(h, index):
            delta = True
            func = h.func
            terms = [terms[0]*t for t in h.args]
        else:
            terms = [t*h for t in terms]
    return func(*terms)


@cacheit
def _extract_delta(expr, index):
    """
    Extract a simple KroneckerDelta from the expression.

    Explanation
    ===========

    Returns the tuple ``(delta, newexpr)`` where:

      - ``delta`` is a simple KroneckerDelta expression if one was found,
        or ``None`` if no simple KroneckerDelta expression was found.

      - ``newexpr`` is a Mul containing the remaining terms; ``expr`` is
        returned unchanged if no simple KroneckerDelta expression was found.

    Examples
    ========

    >>> from sympy import KroneckerDelta
    >>> from sympy.concrete.delta import _extract_delta
    >>> from sympy.abc import x, y, i, j, k
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), i)
    (KroneckerDelta(i, j), 4*x*y)
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), k)
    (None, 4*x*y*KroneckerDelta(i, j))

    See Also
    ========

    sympy.functions.special.tensor_functions.KroneckerDelta
    deltaproduct
    deltasummation
    """
    if not _has_simple_delta(expr, index):
        return (None, expr)
    if isinstance(expr, KroneckerDelta):
        return (expr, S.One)
    if not expr.is_Mul:
        raise ValueError("Incorrect expr")
    delta = None
    terms = []

    for arg in expr.args:
        if delta is None and _is_simple_delta(arg, index):
            delta = arg
        else:
            terms.append(arg)
    return (delta, expr.func(*terms))


@cacheit
def _has_simple_delta(expr, index):
    """
    Returns True if ``expr`` is an expression that contains a KroneckerDelta
    that is simple in the index ``index``, meaning that this KroneckerDelta
    is nonzero for a single value of the index ``index``.
    """
    if expr.has(KroneckerDelta):
        if _is_simple_delta(expr, index):
            return True
        if expr.is_Add or expr.is_Mul:
            return any(_has_simple_delta(arg, index) for arg in expr.args)
    return False


@cacheit
def _is_simple_delta(delta, index):
    """
    Check if ``delta`` is a simple KroneckerDelta in the index ``index``.
    """
    # Implementation of _is_simple_delta function would go here, if provided in the original code.
    """
    如果 ``delta`` 是一个 KroneckerDelta 对象，并且其涉及到指数 ``index``，
    则返回 True。
    """
    # 检查 delta 是否是 KroneckerDelta 对象，并且是否包含指数 index
    if isinstance(delta, KroneckerDelta) and delta.has(index):
        # 提取 KroneckerDelta 对象的参数，并转化为关于 index 的多项式
        p = (delta.args[0] - delta.args[1]).as_poly(index)
        # 如果成功转化为多项式，则检查其是否是一次多项式
        if p:
            return p.degree() == 1
    # 如果不满足条件，则返回 False
    return False
# 使用缓存装饰器来优化函数性能
@cacheit
# 移除包含多个 KroneckerDelta 的乘积项
def _remove_multiple_delta(expr):
    # 如果表达式是加法，则递归处理每个参数
    if expr.is_Add:
        return expr.func(*list(map(_remove_multiple_delta, expr.args)))
    # 如果不是乘法，则直接返回表达式
    if not expr.is_Mul:
        return expr
    # 初始化空列表存放等式和新的参数列表
    eqs = []
    newargs = []
    # 遍历表达式中的每个参数
    for arg in expr.args:
        # 如果是 KroneckerDelta 类型，提取其参数并加入等式列表
        if isinstance(arg, KroneckerDelta):
            eqs.append(arg.args[0] - arg.args[1])
        else:
            newargs.append(arg)
    # 如果没有等式，则返回原表达式
    if not eqs:
        return expr
    # 解等式得到解
    solns = solve(eqs, dict=True)
    # 如果没有解，则返回零
    if len(solns) == 0:
        return S.Zero
    # 如果只有一个解，则更新参数列表并构造新的表达式
    elif len(solns) == 1:
        newargs += [KroneckerDelta(k, v) for k, v in solns[0].items()]
        expr2 = expr.func(*newargs)
        # 如果新表达式与原表达式不同，则递归处理新表达式
        if expr != expr2:
            return _remove_multiple_delta(expr2)
    # 返回处理后的表达式
    return expr


# 使用缓存装饰器来优化函数性能
@cacheit
# 将 KroneckerDelta 的索引重写为最简形式
def _simplify_delta(expr):
    # 如果表达式是 KroneckerDelta 类型
    if isinstance(expr, KroneckerDelta):
        try:
            # 解等式得到解
            slns = solve(expr.args[0] - expr.args[1], dict=True)
            # 如果有解且只有一个解，则构造新的 KroneckerDelta 对象
            if slns and len(slns) == 1:
                return Mul(*[KroneckerDelta(*(key, value))
                             for key, value in slns[0].items()])
        except NotImplementedError:
            pass
    # 返回原表达式
    return expr


# 处理包含 KroneckerDelta 的乘积项
def deltaproduct(f, limit):
    # 如果上下限的差小于零，则返回 1
    if ((limit[2] - limit[1]) < 0) == True:
        return S.One

    # 如果 f 中不含有 KroneckerDelta，则直接返回乘积
    if not f.has(KroneckerDelta):
        return product(f, limit)

    # 如果 f 是加法
    if f.is_Add:
        # 在加法中识别包含简单 KroneckerDelta 的项
        delta = None
        terms = []
        for arg in sorted(f.args, key=default_sort_key):
            if delta is None and _has_simple_delta(arg, limit[0]):
                delta = arg
            else:
                terms.append(arg)
        newexpr = f.func(*terms)
        k = Dummy("kprime", integer=True)
        # 如果上下限是整数，则直接计算结果
        if isinstance(limit[1], int) and isinstance(limit[2], int):
            result = deltaproduct(newexpr, limit) + sum(deltaproduct(newexpr, (limit[0], limit[1], ik - 1)) *
                delta.subs(limit[0], ik) *
                deltaproduct(newexpr, (limit[0], ik + 1, limit[2])) for ik in range(int(limit[1]), int(limit[2] + 1))
            )
        else:
            # 如果上下限不是整数，则进行求和计算
            result = deltaproduct(newexpr, limit) + deltasummation(
                deltaproduct(newexpr, (limit[0], limit[1], k - 1)) *
                delta.subs(limit[0], k) *
                deltaproduct(newexpr, (limit[0], k + 1, limit[2])),
                (k, limit[1], limit[2]),
                no_piecewise=_has_simple_delta(newexpr, limit[0])
            )
        # 移除多余的 KroneckerDelta
        return _remove_multiple_delta(result)

    # 提取 f 中的 KroneckerDelta 和它的第一个参数
    delta, _ = _extract_delta(f, limit[0])
    # 如果 delta 为空（即 delta 为假值），执行以下操作
    if not delta:
        # 根据给定的函数 f 和限制条件中的第一个值，扩展 delta
        g = _expand_delta(f, limit[0])
        # 如果 f 不等于 g，则尝试执行以下操作
        if f != g:
            try:
                # 尝试计算 deltaproduct(g, limit)，并返回其结果
                return factor(deltaproduct(g, limit))
            except AssertionError:
                # 如果计算中出现断言错误，则返回 deltaproduct(g, limit) 的结果
                return deltaproduct(g, limit)
        # 如果 f 等于 g，则返回 product(f, limit) 的结果
        return product(f, limit)

    # 如果 delta 不为空（即 delta 为真值），执行以下操作
    return _remove_multiple_delta(f.subs(limit[0], limit[1])*KroneckerDelta(limit[2], limit[1])) + \
        # 返回 S.One 乘以简化后的 KroneckerDelta(limit[2], limit[1] - 1) 的结果
        S.One*_simplify_delta(KroneckerDelta(limit[2], limit[1] - 1))
# 定义一个装饰器函数，用于缓存函数的计算结果，提高函数执行效率
@cacheit
# 定义名为 deltasummation 的函数，处理包含 KroneckerDelta 的求和操作
def deltasummation(f, limit, no_piecewise=False):
    """
    Handle summations containing a KroneckerDelta.

    Explanation
    ===========

    The idea for summation is the following:

    - If we are dealing with a KroneckerDelta expression, i.e. KroneckerDelta(g(x), j),
      we try to simplify it.

      If we could simplify it, then we sum the resulting expression.
      We already know we can sum a simplified expression, because only
      simple KroneckerDelta expressions are involved.

      If we could not simplify it, there are two cases:

      1) The expression is a simple expression: we return the summation,
         taking care if we are dealing with a Derivative or with a proper
         KroneckerDelta.

      2) The expression is not simple (i.e. KroneckerDelta(cos(x))): we can do
         nothing at all.

    - If the expr is a multiplication expr having a KroneckerDelta term:

      First we expand it.

      If the expansion did work, then we try to sum the expansion.

      If not, we try to extract a simple KroneckerDelta term, then we have two
      cases:

      1) We have a simple KroneckerDelta term, so we return the summation.

      2) We did not have a simple term, but we do have an expression with
         simplified KroneckerDelta terms, so we sum this expression.

    Examples
    ========

    >>> from sympy import oo, symbols
    >>> from sympy.abc import k
    >>> i, j = symbols('i, j', integer=True, finite=True)
    >>> from sympy.concrete.delta import deltasummation
    >>> from sympy import KroneckerDelta
    >>> deltasummation(KroneckerDelta(i, k), (k, -oo, oo))
    1
    >>> deltasummation(KroneckerDelta(i, k), (k, 0, oo))
    Piecewise((1, i >= 0), (0, True))
    >>> deltasummation(KroneckerDelta(i, k), (k, 1, 3))
    Piecewise((1, (i >= 1) & (i <= 3)), (0, True))
    >>> deltasummation(k*KroneckerDelta(i, j)*KroneckerDelta(j, k), (k, -oo, oo))
    j*KroneckerDelta(i, j)
    >>> deltasummation(j*KroneckerDelta(i, j), (j, -oo, oo))
    i
    >>> deltasummation(i*KroneckerDelta(i, j), (i, -oo, oo))
    j

    See Also
    ========

    deltaproduct
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.sums.summation
    """
    # 如果上限小于下限，返回零
    if ((limit[2] - limit[1]) < 0) == True:
        return S.Zero

    # 如果函数 f 中不包含 KroneckerDelta，直接对其进行求和操作
    if not f.has(KroneckerDelta):
        return summation(f, limit)

    # 将变量 x 设为求和的变量
    x = limit[0]

    # 尝试对 f 中的 KroneckerDelta 表达式进行展开
    g = _expand_delta(f, x)
    # 如果展开后的 g 是一个加法表达式，对其中的每个项递归调用 deltasummation，并使用 piecewise_fold 处理结果
    if g.is_Add:
        return piecewise_fold(
            g.func(*[deltasummation(h, limit, no_piecewise) for h in g.args]))

    # 尝试提取一个简单的 KroneckerDelta 项
    delta, expr = _extract_delta(g, x)

    # 如果成功提取到 delta 并且 delta 的 delta_range 不为 None
    if (delta is not None) and (delta.delta_range is not None):
        dinf, dsup = delta.delta_range
        # 如果上限小于等于 delta 的下限，并且下限大于等于 delta 的上限，设定 no_piecewise 为 True
        if (limit[1] - dinf <= 0) == True and (limit[2] - dsup >= 0) == True:
            no_piecewise = True

    # 如果没有提取到有效的 delta，直接对 f 进行求和操作
    if not delta:
        return summation(f, limit)

    # 解方程 delta.args[0] - delta.args[1] 关于变量 x，并返回解的列表
    solns = solve(delta.args[0] - delta.args[1], x)
    # 如果解的数量为0，返回零值
    if len(solns) == 0:
        return S.Zero
    # 如果解的数量不为1，返回函数求和结果
    elif len(solns) != 1:
        return Sum(f, limit)
    # 获取唯一的解
    value = solns[0]
    # 如果禁用分段函数（Piecewise），直接用解值替换表达式中的变量 x
    if no_piecewise:
        return expr.subs(x, value)
    # 如果允许使用分段函数，根据条件创建分段函数
    return Piecewise(
        # 如果解 value 在给定的区间内，则返回替换后的表达式值
        (expr.subs(x, value), Interval(*limit[1:3]).as_relational(value)),
        # 如果解 value 不在给定的区间内，则返回零值
        (S.Zero, True)
    )
```