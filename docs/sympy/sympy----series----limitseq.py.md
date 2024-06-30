# `D:\src\scipysrc\sympy\sympy\series\limitseq.py`

```
"""Limits of sequences"""

# 导入必要的库和模块
from sympy.calculus.accumulationbounds import AccumulationBounds
from sympy.core.add import Add
from sympy.core.function import PoleError
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import factorial, subfactorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.limits import Limit


def difference_delta(expr, n=None, step=1):
    """Difference Operator.

    Explanation
    ===========

    Discrete analog of differential operator. Given a sequence x[n],
    returns the sequence x[n + step] - x[n].

    Examples
    ========

    >>> from sympy import difference_delta as dd
    >>> from sympy.abc import n
    >>> dd(n*(n + 1), n)
    2*n + 2
    >>> dd(n*(n + 1), n, 2)
    4*n + 6

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/DifferenceDelta.html
    """
    expr = sympify(expr)  # 将输入表达式转换为Sympy表达式对象

    if n is None:
        f = expr.free_symbols
        if len(f) == 1:
            n = f.pop()  # 如果表达式只有一个自由变量，将其赋给n
        elif len(f) == 0:
            return S.Zero  # 如果表达式没有自由变量，返回零
        else:
            raise ValueError("Since there is more than one variable in the"
                             " expression, a variable must be supplied to"
                             " take the difference of %s" % expr)
    step = sympify(step)  # 将步长转换为Sympy表达式对象
    if step.is_number is False or step.is_finite is False:
        raise ValueError("Step should be a finite number.")  # 如果步长不是有限数值，抛出异常

    if hasattr(expr, '_eval_difference_delta'):
        result = expr._eval_difference_delta(n, step)
        if result:
            return result

    # 返回差分结果
    return expr.subs(n, n + step) - expr


def dominant(expr, n):
    """Finds the dominant term in a sum, that is a term that dominates
    every other term.

    Explanation
    ===========

    If limit(a/b, n, oo) is oo then a dominates b.
    If limit(a/b, n, oo) is 0 then b dominates a.
    Otherwise, a and b are comparable.

    If there is no unique dominant term, then returns ``None``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.series.limitseq import dominant
    >>> from sympy.abc import n, k
    >>> dominant(5*n**3 + 4*n**2 + n + 1, n)
    5*n**3
    >>> dominant(2**n + Sum(k, (k, 0, n)), n)
    2**n

    See Also
    ========

    sympy.series.limitseq.dominant
    """
    terms = Add.make_args(expr.expand(func=True))  # 将表达式展开并获取所有项
    term0 = terms[-1]  # 获取最后一项作为基准项
    comp = [term0]  # 可比较的项
    # 遍历列表 terms 中除了最后一个元素的所有元素
    for t in terms[:-1]:
        # 计算 term0 除以当前 t 的结果
        r = term0/t
        # 对结果 r 进行 Gamma 函数简化
        e = r.gammasimp()
        # 如果简化后的结果 e 等于原始结果 r，则对 e 进行因式分解
        if e == r:
            e = r.factor()
        # 计算 e 的极限，n 是指定的参数
        l = limit_seq(e, n)
        # 如果极限为 None，则返回 None
        if l is None:
            return None
        # 如果极限为零
        elif l.is_zero:
            # 更新 term0 的值为当前的 t
            term0 = t
            # 重置 comp 列表，只包含当前的 term0
            comp = [term0]
        # 如果极限不是正无穷也不是负无穷
        elif l not in [S.Infinity, S.NegativeInfinity]:
            # 将当前的 t 添加到 comp 列表中
            comp.append(t)
    # 如果 comp 列表的长度大于 1，则返回 None
    if len(comp) > 1:
        return None
    # 否则返回最后计算得到的 term0
    return term0
# 定义一个函数，用于计算在指定条件下的极限序列
def _limit_inf(expr, n):
    try:
        # 尝试计算表达式 expr 当 n 趋向无穷时的极限
        return Limit(expr, n, S.Infinity).doit(deep=False)
    except (NotImplementedError, PoleError):
        # 处理极限计算中可能出现的未实现错误或极点错误
        return None


# 定义一个函数，用于计算在指定条件下的极限序列，考虑多次尝试
def _limit_seq(expr, n, trials):
    from sympy.concrete.summations import Sum

    # 进行指定次数的尝试
    for i in range(trials):
        # 如果表达式中不包含和符号 Sum
        if not expr.has(Sum):
            # 尝试计算表达式 expr 当 n 趋向无穷时的极限
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        # 将表达式拆分为分子和分母
        num, den = expr.as_numer_denom()
        # 如果分母或分子中不含有 n
        if not den.has(n) or not num.has(n):
            # 尝试计算表达式在进行深度计算后当 n 趋向无穷时的极限
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None

        # 对分子和分母分别应用差分增量，并简化表达式
        num, den = (difference_delta(t.expand(), n) for t in [num, den])
        expr = (num / den).gammasimp()

        # 如果表达式中不包含和符号 Sum
        if not expr.has(Sum):
            # 尝试计算表达式 expr 当 n 趋向无穷时的极限
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        # 将表达式拆分为分子和分母
        num, den = expr.as_numer_denom()

        # 应用 dominant 函数获取主导项的分子
        num = dominant(num, n)
        if num is None:
            return None

        # 应用 dominant 函数获取主导项的分母
        den = dominant(den, n)
        if den is None:
            return None

        # 简化表达式为分子除以分母，并进行 gamma 简化
        expr = (num / den).gammasimp()


# 定义一个函数，用于计算序列的极限值，当 n 趋向无穷时
def limit_seq(expr, n=None, trials=5):
    """Finds the limit of a sequence as index ``n`` tends to infinity.

    Parameters
    ==========

    expr : Expr
        SymPy expression for the ``n-th`` term of the sequence
    n : Symbol, optional
        The index of the sequence, an integer that tends to positive
        infinity. If None, inferred from the expression unless it has
        multiple symbols.
    trials: int, optional
        The algorithm is highly recursive. ``trials`` is a safeguard from
        infinite recursion in case the limit is not easily computed by the
        algorithm. Try increasing ``trials`` if the algorithm returns ``None``.

    Admissible Terms
    ================

    The algorithm is designed for sequences built from rational functions,
    indefinite sums, and indefinite products over an indeterminate n. Terms of
    alternating sign are also allowed, but more complex oscillatory behavior is
    not supported.

    Examples
    ========

    >>> from sympy import limit_seq, Sum, binomial
    >>> from sympy.abc import n, k, m
    >>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
    5/3
    >>> limit_seq(binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n)), n)
    3/4
    >>> limit_seq(Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n), n)
    4

    See Also
    ========

    sympy.series.limitseq.dominant

    References
    ==========

    .. [1] Computing Limits of Sequences - Manuel Kauers
    """

    from sympy.concrete.summations import Sum
    if n is None:
        free = expr.free_symbols
        # 如果表达式只有一个自由符号，则将其视为 n
        if len(free) == 1:
            n = free.pop()
        # 如果没有自由符号，则返回表达式本身
        elif not free:
            return expr
        # 如果表达式有多个自由符号，则报错
        else:
            raise ValueError("Expression has more than one variable. "
                             "Please specify a variable.")
    # 如果变量 n 不在表达式的自由符号中，则直接返回表达式，不做处理
    elif n not in expr.free_symbols:
        return expr

    # 将表达式中的 fibonacci 函数重写为 GoldenRatio，将 factorial 函数重写为 subfactorial 和 gamma
    expr = expr.rewrite(fibonacci, S.GoldenRatio)
    expr = expr.rewrite(factorial, subfactorial, gamma)

    # 创建三个虚拟变量：n_ 为正整数，n1 为正奇数，n2 为正偶数
    n_ = Dummy("n", integer=True, positive=True)
    n1 = Dummy("n", odd=True, positive=True)
    n2 = Dummy("n", even=True, positive=True)

    # 如果表达式中存在负指数与 n 相关的项，或者包含三角函数，则分别考虑 n 为偶数和奇数的情况
    powers = (p.as_base_exp() for p in expr.atoms(Pow))
    if (any(b.is_negative and e.has(n) for b, e in powers) or
            expr.has(cos, sin)):
        # 对于 n 为正奇数的情况，计算极限
        L1 = _limit_seq(expr.xreplace({n: n1}), n1, trials)
        if L1 is not None:
            # 对于 n 为正偶数的情况，计算极限
            L2 = _limit_seq(expr.xreplace({n: n2}), n2, trials)
            # 如果两者不相等，则返回它们的累积边界
            if L1 != L2:
                if L1.is_comparable and L2.is_comparable:
                    return AccumulationBounds(Min(L1, L2), Max(L1, L2))
                else:
                    return None
    else:
        # 如果表达式不包含负指数与 n 相关的项或三角函数，则计算 n 为正整数时的极限
        L1 = _limit_seq(expr.xreplace({n: n_}), n_, trials)

    # 如果计算得到的极限值不为 None，则返回该值
    if L1 is not None:
        return L1
    else:
        # 如果表达式是一个加法表达式，分别计算每个加数的极限并进行合并
        if expr.is_Add:
            limits = [limit_seq(term, n, trials) for term in expr.args]
            # 如果任何一个加数的极限为 None，则整个表达式的极限也为 None
            if any(result is None for result in limits):
                return None
            else:
                return Add(*limits)
        # 如果表达式不包含 Sum（求和）操作，则考虑其绝对值是否更容易处理，如果趋向于 0，则极限为 0
        elif not expr.has(Sum):
            lim = _limit_seq(Abs(expr.xreplace({n: n_})), n_, trials)
            if lim is not None and lim.is_zero:
                return S.Zero
```