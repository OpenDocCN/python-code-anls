# `D:\src\scipysrc\sympy\sympy\concrete\summations.py`

```
from typing import Tuple as tTuple  # 导入类型提示，给 Tuple 别名 tTuple

from sympy.calculus.singularities import is_decreasing  # 导入函数 is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds  # 导入类 AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits  # 从当前包导入模块 expr_with_intlimits 中的 ExprWithIntLimits 类
from .expr_with_limits import AddWithLimits  # 从当前包导入模块 expr_with_limits 中的 AddWithLimits 类
from .gosper import gosper_sum  # 从当前包导入模块 gosper 中的 gosper_sum 函数
from sympy.core.expr import Expr  # 导入类 Expr
from sympy.core.add import Add  # 导入类 Add
from sympy.core.containers import Tuple  # 导入类 Tuple
from sympy.core.function import Derivative, expand  # 导入函数 Derivative 和 expand
from sympy.core.mul import Mul  # 导入类 Mul
from sympy.core.numbers import Float, _illegal  # 导入类 Float 和 _illegal
from sympy.core.relational import Eq  # 导入类 Eq
from sympy.core.singleton import S  # 导入符号 S
from sympy.core.sorting import ordered  # 导入函数 ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols  # 导入类 Dummy, Wild, Symbol 和 symbols
from sympy.functions.combinatorial.factorials import factorial  # 导入函数 factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic  # 导入函数 bernoulli 和 harmonic
from sympy.functions.elementary.exponential import exp, log  # 导入函数 exp 和 log
from sympy.functions.elementary.piecewise import Piecewise  # 导入类 Piecewise
from sympy.functions.elementary.trigonometric import cot, csc  # 导入函数 cot 和 csc
from sympy.functions.special.hyper import hyper  # 导入函数 hyper
from sympy.functions.special.tensor_functions import KroneckerDelta  # 导入函数 KroneckerDelta
from sympy.functions.special.zeta_functions import zeta  # 导入函数 zeta
from sympy.integrals.integrals import Integral  # 导入类 Integral
from sympy.logic.boolalg import And  # 导入函数 And
from sympy.polys.partfrac import apart  # 导入函数 apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed  # 导入异常类 PolynomialError 和 PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor  # 导入函数 parallel_poly_from_expr, Poly 和 factor
from sympy.polys.rationaltools import together  # 导入函数 together
from sympy.series.limitseq import limit_seq  # 导入函数 limit_seq
from sympy.series.order import O  # 导入类 O
from sympy.series.residues import residue  # 导入函数 residue
from sympy.sets.sets import FiniteSet, Interval  # 导入类 FiniteSet 和 Interval
from sympy.utilities.iterables import sift  # 导入函数 sift
import itertools  # 导入 itertools 库


class Sum(AddWithLimits, ExprWithIntLimits):
    r"""
    Represents unevaluated summation.

    Explanation
    ===========

    ``Sum`` represents a finite or infinite series, with the first argument
    being the general form of terms in the series, and the second argument
    being ``(dummy_variable, start, end)``, with ``dummy_variable`` taking
    all integer values from ``start`` through ``end``. In accordance with
    long-standing mathematical convention, the end term is included in the
    summation.

    Finite sums
    ===========

    For finite sums (and sums with symbolic limits assumed to be finite) we
    follow the summation convention described by Karr [1], especially
    definition 3 of section 1.4. The sum:

    .. math::

        \sum_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \sum_{m \leq i < n} f(i) = f(m) + f(m+1) + \ldots + f(n-2) + f(n-1)

    with the upper limit value `f(n)` excluded. The sum over an empty set is
    zero if and only if `m = n`:

    .. math::

        \sum_{m \leq i < n} f(i) = 0  \quad \mathrm{for} \quad  m = n

    Finally, for all other sums over empty sets we assume the following
    definition:
    """
    # 定义一个空类，用于表示一个数学上的求和符号
    __slots__ = ()
    
    # 类型标注：定义一个名为 `limits` 的类变量，其类型为元组，元组元素为三元组，包含符号、表达式和表达式
    limits: tTuple[tTuple[Symbol, Expr, Expr]]
    # 使用特殊方法__new__来创建新的对象实例，继承自AddWithLimits类
    def __new__(cls, function, *symbols, **assumptions):
        # 调用父类AddWithLimits的__new__方法创建对象实例
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        # 如果对象没有属性'limits'，直接返回对象实例
        if not hasattr(obj, 'limits'):
            return obj
        # 检查是否每个限制条件都是长度为3的元组且不含有None值，若不是则引发ValueError异常
        if any(len(l) != 3 or None in l for l in obj.limits):
            raise ValueError('Sum requires values for lower and upper bounds.')

        return obj

    # 判断该求和对象是否为零，仅当其函数为零或所有项相互抵消时才为零
    def _eval_is_zero(self):
        if self.function.is_zero or self.has_empty_sequence:
            return True

    # 判断该求和对象是否为扩展实数（包括无穷大和无穷小）
    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True
        return self.function.is_extended_real

    # 判断该求和对象是否为正数
    def _eval_is_positive(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_positive

    # 判断该求和对象是否为负数
    def _eval_is_negative(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_negative

    # 判断该求和对象是否为有限的（不是无穷大或无穷小）
    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True
    def doit(self, **hints):
        # 如果 hints 字典中有 'deep' 键且其值为 True，则递归调用 self.function.doit()
        if hints.get('deep', True):
            f = self.function.doit(**hints)
        else:
            f = self.function

        # 确保任何明确的限制都有匹配的假设变量
        reps = {}
        for xab in self.limits:
            # 为限制元组 xab 中的第一个元素创建具有继承属性的虚拟变量
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        # 如果有替换的情况，则创建反向替换的字典 undo
        if reps:
            undo = {v: k for k, v in reps.items()}
            # 将替换应用到当前对象 self 上，并递归调用 doit 方法
            did = self.xreplace(reps).doit(**hints)
            # 处理 separate=True 的情况
            if isinstance(did, tuple):
                did = tuple([i.xreplace(undo) for i in did])
            elif did is not None:
                did = did.xreplace(undo)
            else:
                did = self
            return did

        # 如果 self.function 是矩阵类型，则展开矩阵并进行相应的处理
        if self.function.is_Matrix:
            expanded = self.expand()
            if self != expanded:
                return expanded.doit()
            return _eval_matrix_sum(self)

        # 遍历限制列表 self.limits
        for n, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            # 如果差值为 -1，则任何空集的求和结果为零
            if dif == -1:
                return S.Zero
            # 如果差值为整数且为负数，则调整 a 和 b，并反转 f 的符号
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = -f

            # 对 f 进行求和处理，传入变量 i、起始值 a、结束值 b
            newf = eval_sum(f, (i, a, b))
            # 如果 newf 为 None
            if newf is None:
                # 如果 f 与 self.function 相等，则尝试计算 zeta 函数
                if f == self.function:
                    zeta_function = self.eval_zeta_function(f, (i, a, b))
                    if zeta_function is not None:
                        return zeta_function
                    return self
                else:
                    # 否则返回当前对象 self 的 func 方法调用结果，传入参数 f 和后续的限制
                    return self.func(f, *self.limits[n:])
            f = newf

        # 如果 hints 字典中有 'deep' 键且其值为 True，则检查是否需要递归处理 f
        if hints.get('deep', True):
            # 如果 eval_sum 返回的结果部分未求值化，例如 Piecewise 类型，则不递归调用 doit()
            if not isinstance(f, Piecewise):
                return f.doit(**hints)

        # 返回处理后的结果 f
        return f

    def eval_zeta_function(self, f, limits):
        """
        检查函数是否匹配 zeta 函数。

        如果匹配，则返回一个 `Piecewise` 表达式，因为 zeta 函数仅在 `s > 1` 且 `q > 0` 时收敛。
        """
        i, a, b = limits
        w, y, z = Wild('w', exclude=[i]), Wild('y', exclude=[i]), Wild('z', exclude=[i])
        # 尝试匹配函数 f 是否为形式 (w * i + y) ** (-z)
        result = f.match((w * i + y) ** (-z))
        # 如果匹配成功且 b 为正无穷，则计算系数和 zeta 函数
        if result is not None and b is S.Infinity:
            coeff = 1 / result[w] ** result[z]
            s = result[z]
            q = result[y] / result[w] + a
            return Piecewise((coeff * zeta(s, q), And(q > 0, s > 1)), (self, True))
    def _eval_derivative(self, x):
        """
        Differentiate wrt x as long as x is not in the free symbols of any of
        the upper or lower limits.

        Explanation
        ===========

        Sum(a*b*x, (x, 1, a)) can be differentiated wrt x or b but not `a`
        since the value of the sum is discontinuous in `a`. In a case
        involving a limit variable, the unevaluated derivative is returned.
        """

        # 如果 x 是符号，并且不在当前对象的自由符号集合中，则返回零
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero

        # 获取函数和极限
        f, limits = self.function, list(self.limits)

        # 弹出最后一个极限
        limit = limits.pop(-1)

        # 如果还有其它极限，则 f 是一个 Sum 的参数
        if limits:
            f = self.func(f, *limits)

        # 解构极限元组，获取上下限和变量
        _, a, b = limit

        # 如果 x 在上限或下限的自由符号中，则返回 None
        if x in a.free_symbols or x in b.free_symbols:
            return None
        
        # 否则，计算函数 f 对 x 的导数
        df = Derivative(f, x, evaluate=True)
        
        # 构造新的对象并返回
        rv = self.func(df, limit)
        return rv

    def _eval_difference_delta(self, n, step):
        k, _, upper = self.args[-1]

        # 替换上限的变量 n 为 n + step
        new_upper = upper.subs(n, n + step)

        # 确定函数 f
        if len(self.args) == 2:
            f = self.args[0]
        else:
            f = self.func(*self.args[:-1])

        # 返回计算后的求和
        return Sum(f, (k, upper + 1, new_upper)).doit()

    def _eval_simplify(self, **kwargs):
        # 获取函数
        function = self.function

        # 如果 deep 参数为真，则深度简化函数
        if kwargs.get('deep', True):
            function = function.simplify(**kwargs)

        # 将函数展开并拆分成加法项
        terms = Add.make_args(expand(function))
        s_t = [] # Sum Terms，用于存储求和项
        o_t = [] # Other Terms，用于存储其它项

        # 遍历每一个加法项
        for term in terms:
            if term.has(Sum):
                # 如果加法项中包含求和符号
                # 将其展开并简化内部求和项
                subterms = Mul.make_args(expand(term))
                out_terms = []
                for subterm in subterms:
                    # 遍历每个子项
                    if isinstance(subterm, Sum):
                        # 如果是求和项，则进一步简化
                        out_terms.append(subterm._eval_simplify(**kwargs))
                    else:
                        # 否则直接添加到输出项中
                        out_terms.append(subterm)

                # 构造乘积项并添加到求和项列表中
                s_t.append(Mul(*out_terms))
            else:
                # 如果加法项中不包含求和符号，则直接添加到其它项列表中
                o_t.append(term)

        # 尝试结合所有内部的求和项以进一步简化结果
        from sympy.simplify.simplify import factor_sum, sum_combine
        result = Add(sum_combine(s_t), *o_t)

        # 对结果进行因式分解并返回
        return factor_sum(result, limits=self.limits)
    # 检查一个无穷级数的绝对收敛性。
    # 相当于检查无穷级数项绝对值的收敛性。
    def is_absolutely_convergent(self):
        """
        Checks for the absolute convergence of an infinite series.

        Same as checking convergence of absolute value of sequence_term of
        an infinite series.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absolute_convergence

        Examples
        ========

        >>> from sympy import Sum, Symbol, oo
        >>> n = Symbol('n', integer=True)
        >>> Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent()
        False
        >>> Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent()
        True

        See Also
        ========

        Sum.is_convergent
        """
        # 返回绝对值序列项的收敛性结果
        return Sum(abs(self.function), self.limits).is_convergent()
    def reverse_order(self, *indices):
        """
        Reverse the order of a limit in a Sum.

        Explanation
        ===========
        
        ``reverse_order(self, *indices)`` reverses some limits in the expression
        ``self`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y, a, b, c, d

        >>> Sum(x, (x, 0, 3)).reverse_order(x)
        Sum(-x, (x, 4, -1))
        >>> Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(x, y)
        Sum(x*y, (x, 6, 0), (y, 7, -1))
        >>> Sum(x, (x, a, b)).reverse_order(x)
        Sum(-x, (x, b + 1, a - 1))
        >>> Sum(x, (x, a, b)).reverse_order(0)
        Sum(-x, (x, b + 1, a - 1))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x**2, (x, a, b), (x, c, d))
        >>> S
        Sum(x**2, (x, a, b), (x, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x**2, (x, b + 1, a - 1), (x, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x**2, (x, b + 1, a - 1), (x, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index, reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255
        """
        l_indices = list(indices)  # 将可变参数 indices 转换为列表形式

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):  # 如果索引不是整数，将其转换为对应的索引值
                l_indices[i] = self.index(indx)

        e = 1  # 初始化符号因子为正
        limits = []
        for i, limit in enumerate(self.limits):
            l = limit
            if i in l_indices:  # 如果当前索引在需要反转的索引列表中
                e = -e  # 切换符号因子
                l = (limit[0], limit[2] + 1, limit[1] - 1)  # 反转限制的上下界
            limits.append(l)  # 将处理后的限制添加到列表中

        return Sum(e * self.function, *limits)  # 返回反转后的表达式和限制

    def _eval_rewrite_as_Product(self, *args, **kwargs):
        from sympy.concrete.products import Product
        if self.function.is_extended_real:
            return log(Product(exp(self.function), *self.limits))
def summation(f, *symbols, **kwargs):
    r"""
    Compute the summation of f with respect to symbols.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Integral.
    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,
    i.e.,

    ::

                                    b
                                  ____
                                  \   `
        summation(f, (i, a, b)) =  )    f
                                  /___,
                                  i = a

    If it cannot compute the sum, it returns an unevaluated Sum object.
    Repeated sums can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import summation, oo, symbols, log
    >>> i, n, m = symbols('i n m', integer=True)

    >>> summation(2*i - 1, (i, 1, n))
    n**2
    >>> summation(1/2**i, (i, 0, oo))
    2
    >>> summation(1/log(n)**n, (n, 2, oo))
    Sum(log(n)**(-n), (n, 2, oo))
    >>> summation(i, (i, 0, n), (n, 0, m))
    m**3/6 + m**2/2 + m/3

    >>> from sympy.abc import x
    >>> from sympy import factorial
    >>> summation(x**n/factorial(n), (n, 0, oo))
    exp(x)

    See Also
    ========

    Sum
    Product, sympy.concrete.products.product

    """
    return Sum(f, *symbols, **kwargs).doit(deep=False)


def telescopic_direct(L, R, n, limits):
    """
    Returns the direct summation of the terms of a telescopic sum

    Explanation
    ===========

    L is the term with lower index
    R is the term with higher index
    n difference between the indexes of L and R

    Examples
    ========

    >>> from sympy.concrete.summations import telescopic_direct
    >>> from sympy.abc import k, a, b
    >>> telescopic_direct(1/k, -1/(k+2), 2, (k, a, b))
    -1/(b + 2) - 1/(b + 1) + 1/(a + 1) + 1/a

    """
    (i, a, b) = limits
    # Direct summation of telescopic terms over specified range
    return Add(*[L.subs(i, a + m) + R.subs(i, b - m) for m in range(n)])


def telescopic(L, R, limits):
    '''
    Tries to perform the summation using the telescopic property.

    Return None if not possible.
    '''
    (i, a, b) = limits
    if L.is_Add or R.is_Add:
        return None

    # Attempt to find a solution for telescopic cancellation
    k = Wild("k")
    sol = (-R).match(L.subs(i, i + k))
    s = None
    if sol and k in sol:
        s = sol[k]
        if not (s.is_Integer and L.subs(i, i + s) + R == 0):
            # If the match isn't valid or doesn't cancel properly, reset s to None
            s = None

    # Additional checks or refinements could go here
    # 如果 s 为 None，则执行以下操作
    if s is None:
        # 创建一个名为 m 的虚拟变量
        m = Dummy('m')
        try:
            # 尝试导入 sympy.solvers.solvers 模块中的 solve 函数，并求解方程 L.subs(i, i + m) + R
            sol = solve(L.subs(i, i + m) + R, m) or []
        except NotImplementedError:
            # 如果 solve 函数抛出 NotImplementedError 异常，则返回 None
            return None
        # 筛选出符合条件的整数解 si，并且使得 (L.subs(i, i + si) + R).expand().is_zero 成立
        sol = [si for si in sol if si.is_Integer and
               (L.subs(i, i + si) + R).expand().is_zero]
        # 如果符合条件的解不唯一，则返回 None
        if len(sol) != 1:
            return None
        # 将求解得到的唯一解赋给 s
        s = sol[0]

    # 如果 s 小于 0，则调用 telescopic_direct 函数，交换 L 和 R，并求绝对值后的 s
    if s < 0:
        return telescopic_direct(R, L, abs(s), (i, a, b))
    # 如果 s 大于 0，则调用 telescopic_direct 函数，直接使用当前的 L 和 R，并传递 s
    elif s > 0:
        return telescopic_direct(L, R, s, (i, a, b))
# 定义函数 eval_sum，计算给定函数 f 在指定上下限 limits 范围内的求和
def eval_sum(f, limits):
    # 解包 limits 元组，分别得到求和变量 i，上限 a，下限 b
    (i, a, b) = limits
    # 如果 f 是零函数，直接返回零
    if f.is_zero:
        return S.Zero
    # 如果求和变量 i 不在 f 的自由符号中，说明 f 不含有求和变量 i，直接返回 f 乘以区间宽度
    if i not in f.free_symbols:
        return f*(b - a + 1)
    # 如果上限等于下限，即求和范围只有一个值，返回 f 在该点的取值
    if a == b:
        return f.subs(i, a)
    # 如果 f 是 Piecewise 类型的函数
    if isinstance(f, Piecewise):
        # 检查 Piecewise 中的每一个分支，如果所有分支的条件不依赖于求和变量 i，
        # 则可以将整个求和替换为对每个分支求和的 Piecewise 结果
        if not any(i in arg.args[1].free_symbols for arg in f.args):
            newargs = []
            # 对 Piecewise 的每个分支，递归调用 eval_sum 计算求和结果，并更新分支
            for arg in f.args:
                newexpr = eval_sum(arg.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, arg.cond))
            return f.func(*newargs)

    # 如果 f 包含 KroneckerDelta
    if f.has(KroneckerDelta):
        from .delta import deltasummation, _has_simple_delta
        # 替换 f 中的 Sum 函数，确保 delta 函数简化后可以直接求和
        f = f.replace(
            lambda x: isinstance(x, Sum),
            lambda x: x.factor()
        )
        if _has_simple_delta(f, limits[0]):
            # 使用 deltasummation 函数计算包含 delta 函数的求和
            return deltasummation(f, limits)

    # 计算上下限差值 dif
    dif = b - a
    definite = dif.is_Integer
    # 如果 dif 是整数并且较小，则直接使用 eval_sum_direct 进行求和计算
    if definite and (dif < 100):
        return eval_sum_direct(f, (i, a, b))
    # 如果 f 是 Piecewise 类型，则无法用简单方法求和，返回 None
    if isinstance(f, Piecewise):
        return None
    # 尝试用符号方法求和计算，这在求和项较多时可以节省时间
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    # 如果 dif 是整数，则尝试直接计算求和
    if definite:
        return eval_sum_direct(f, (i, a, b))


# 定义函数 eval_sum_direct，直接计算给定表达式 expr 在指定上下限 limits 范围内的求和
def eval_sum_direct(expr, limits):
    """
    Evaluate expression directly, but perform some simple checks first
    to possibly result in a smaller expression and faster execution.
    """
    # 解包 limits 元组，分别得到求和变量 i，上限 a，下限 b
    (i, a, b) = limits

    # 计算上下限差值 dif
    dif = b - a
    # 如果表达式是乘法类型，尝试因式分解不包含求和变量 i 的部分，并进行优化计算
    if expr.is_Mul:
        without_i, with_i = expr.as_independent(i)
        if without_i != 1:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            L, R = expr.as_two_terms()
            # 尝试单独计算每一项，以优化计算速度
            if not L.has(i):
                sR = eval_sum_direct(R, (i, a, b))
                if sR:
                    return L*sR
            if not R.has(i):
                sL = eval_sum_direct(L, (i, a, b))
                if sL:
                    return sL*R

    # 对于 Add 或 Mul 类型的表达式，进行进一步计算
    try:
        expr = apart(expr, i)  # 查看是否变成 Add 类型
    except PolynomialError:
        pass
    # 检查表达式是否是加法表达式
    if expr.is_Add:
        # 尝试分离不包含变量 i 的部分和包含变量 i 的部分
        without_i, with_i = expr.as_independent(i)
        # 如果不包含变量 i 的部分不为零
        if without_i != 0:
            # 对不包含变量 i 的部分求和
            s = eval_sum_direct(with_i, (i, a, b))
            # 如果求和结果不为空
            if s:
                # 计算最终结果
                r = without_i*(dif + 1) + s
                # 如果结果不是 NaN，则返回结果
                if r is not S.NaN:
                    return r
        else:
            # 尝试按项计算求和
            L, R = expr.as_two_terms()
            # 分别对左右两部分求和
            lsum = eval_sum_direct(L, (i, a, b))
            rsum = eval_sum_direct(R, (i, a, b))

            # 如果左右两部分的求和结果都不为空
            if None not in (lsum, rsum):
                # 计算最终结果
                r = lsum + rsum
                # 如果结果不是 NaN，则返回结果
                if r is not S.NaN:
                    return r

    # 如果表达式不是加法表达式或以上方法无法得到有效结果，则进行备选计算
    return Add(*[expr.subs(i, a + j) for j in range(dif + 1)])
# 定义函数 eval_sum_symbolic，计算符号表达式 f 的和，给定求和的上下限 limits
def eval_sum_symbolic(f, limits):
    # 备份原始表达式 f
    f_orig = f
    # 解包上下限元组 (i, a, b)
    (i, a, b) = limits
    
    # 如果 f 不包含变量 i，则直接返回 f 乘以 (b - a + 1) 的结果
    if not f.has(i):
        return f*(b - a + 1)

    # 线性性质处理
    if f.is_Mul:
        # 尝试因子分解，分为不包含 i 的部分和包含 i 的部分
        without_i, with_i = f.as_independent(i)
        if without_i != 1:
            # 递归计算包含 i 的部分的和
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            # 尝试分为两项求和
            L, R = f.as_two_terms()

            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L*sR

            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL*R

    # 如果 f 是 Add 类型的表达式
    if f.is_Add:
        L, R = f.as_two_terms()
        # 尝试使用 telescopical 方法求解
        lrsum = telescopic(L, R, (i, a, b))

        if lrsum:
            return lrsum

        # 尝试因子分解，分为不包含 i 的部分和包含 i 的部分
        without_i, with_i = f.as_independent(i)
        if without_i != 0:
            # 递归计算包含 i 的部分的和
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*(b - a + 1) + s
                if r is not S.NaN:
                    return r
        else:
            # 尝试分为两项逐项求和
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))

            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r

    # 多项式项处理，使用 Faulhaber's formula
    n = Wild('n')
    result = f.match(i**n)

    if result is not None:
        n = result[n]

        if n.is_Integer:
            if n >= 0:
                # 处理指数为非负整数的情况
                if (b is S.Infinity and a is not S.NegativeInfinity) or \
                   (a is S.NegativeInfinity and b is not S.Infinity):
                    return S.Infinity
                # 计算 Bernoulli 数列的差值
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a))/(n + 1)).expand()
            elif a.is_Integer and a >= 1:
                # 处理指数为负整数且 a 为整数且大于等于 1 的情况
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))
    # 检查 a 和 b 是否都不包含无穷大值，否则跳过
    if not (a.has(S.Infinity, S.NegativeInfinity) or
            b.has(S.Infinity, S.NegativeInfinity)):
        
        # 定义用于匹配的通配符
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])
        wexp = Wild('wexp')

        # 尝试对 f 进行 powsimp，以便更容易与指数模式匹配，并对指数进行展开，以便更容易与线性模式匹配
        e = f.powsimp().match(c1 ** wexp)
        if e is not None:
            # 如果成功匹配指数模式，则尝试匹配线性模式
            e_exp = e.pop(wexp).expand().match(c2*i + c3)
            if e_exp is not None:
                e.update(e_exp)

                # 计算 p 和 q
                p = (c1**c3).subs(e)
                q = (c1**c2).subs(e)

                # 计算求和表达式的分式部分
                r = p*(q**a - q**(b + 1))/(1 - q)
                l = p*(b - a + 1)

                # 返回分段函数，根据 q 是否等于 1 来决定返回 l 或者 r
                return Piecewise((l, Eq(q, S.One)), (r, True))

        # 如果无法使用指数模式匹配，尝试使用高斯-戈斯珀算法求和
        r = gosper_sum(f, (i, a, b))

        # 如果求和结果是乘法或加法表达式
        if isinstance(r, (Mul, Add)):
            # 导入所需的函数和模块
            from sympy.simplify.radsimp import denom
            from sympy.solvers.solvers import solve

            # 找到非限制符号和分母
            non_limit = r.free_symbols - Tuple(*limits[1:]).free_symbols
            den = denom(together(r))
            den_sym = non_limit & den.free_symbols
            args = []

            # 对每个符号进行处理
            for v in ordered(den_sym):
                try:
                    s = solve(den, v)
                    m = Eq(v, s[0]) if s else S.false
                    if m != False:
                        args.append((Sum(f_orig.subs(*m.args), limits).doit(), m))
                    break
                except NotImplementedError:
                    continue

            # 添加剩余的表达式到分段函数的参数中
            args.append((r, True))
            return Piecewise(*args)

        # 如果 r 是有效值但不是 None 或 S.NaN，则直接返回
        if r not in (None, S.NaN):
            return r

    # 尝试使用超几何求和函数
    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h

    # 尝试使用残余求和函数
    r = eval_sum_residue(f_orig, (i, a, b))
    if r is not None:
        return r

    # 如果能够因式分解 f_orig，则尝试使用符号求和函数
    factored = f_orig.factor()
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))
def _eval_sum_hyper(f, i, a):
    """ Returns (res, cond). Sums from a to oo. """
    # 如果 a 不等于 0，递归调用 _eval_sum_hyper 函数，将 i 替换为 i + a，直到 a 为 0
    if a != 0:
        return _eval_sum_hyper(f.subs(i, i + a), i, 0)

    # 如果 f 在 i = 0 处为 0
    if f.subs(i, 0) == 0:
        # 导入 simplify 函数并检查在整数且为正的虚拟符号 i 下，f 是否简化为 0
        from sympy.simplify.simplify import simplify
        if simplify(f.subs(i, Dummy('i', integer=True, positive=True))) == 0:
            # 返回和为 0，并设置条件为 True
            return S.Zero, True
        # 否则，递归调用 _eval_sum_hyper 函数，将 i 替换为 i + 1
        return _eval_sum_hyper(f.subs(i, i + 1), i, 0)

    # 导入 hypersimp 函数并尝试对 f 进行超级简化
    from sympy.simplify.simplify import hypersimp
    hs = hypersimp(f, i)
    if hs is None:
        return None

    # 如果 hs 是 Float 类型，导入 nsimplify 函数并将其简化为可数值化的形式
    if isinstance(hs, Float):
        from sympy.simplify.simplify import nsimplify
        hs = nsimplify(hs)

    # 导入 combsimp, hyperexpand, fraction 函数，准备对超几何级数进行进一步处理
    from sympy.simplify.combsimp import combsimp
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.simplify.radsimp import fraction
    numer, denom = fraction(factor(hs))
    top, topl = numer.as_coeff_mul(i)
    bot, botl = denom.as_coeff_mul(i)
    ab = [top, bot]
    factors = [topl, botl]
    params = [[], []]
    for k in range(2):
        for fac in factors[k]:
            mul = 1
            if fac.is_Pow:
                mul = fac.exp
                fac = fac.base
                if not mul.is_Integer:
                    return None
            p = Poly(fac, i)
            if p.degree() != 1:
                return None
            m, n = p.all_coeffs()
            ab[k] *= m**mul
            params[k] += [n/m]*mul

    # 在分子参数列表中添加 "1"，以考虑超几何级数中隐含的 n! 因子
    ap = params[0] + [1]
    bq = params[1]
    x = ab[0]/ab[1]
    h = hyper(ap, bq, x)
    # 对 f 进行组合简化
    f = combsimp(f)
    # 返回结果：f 在 i = 0 处的值乘以 hyper 函数的展开，以及 hyper 函数的收敛性声明
    return f.subs(i, 0)*hyperexpand(h), h.convergence_statement


def eval_sum_hyper(f, i_a_b):
    i, a, b = i_a_b

    # 如果 f 不是超几何函数，则返回 None
    if f.is_hypergeometric(i) is False:
        return

    # 如果 (b - a) 是整数，直接返回 None
    if (b - a).is_Integer:
        # 我们无法比直接按照显式方式进行求和更好
        return None

    # 创建一个 Sum 对象，表示从 a 到 b 对 f 求和
    old_sum = Sum(f, (i, a, b))

    # 如果 b 不是无穷大
    if b != S.Infinity:
        # 如果 a 是负无穷
        if a is S.NegativeInfinity:
            # 尝试求解 f 在 -i 下从 -b 到 -无穷大 的和
            res = _eval_sum_hyper(f.subs(i, -i), i, -b)
            if res is not None:
                # 如果成功求解，返回 Piecewise 表达式，条件为 True
                return Piecewise(res, (old_sum, True))
        else:
            # 定义一个 lambda 函数 n_illegal，用于统计 f 中非法项的数量
            n_illegal = lambda x: sum(x.count(_) for _ in _illegal)
            had = n_illegal(f)
            # 检查是否引入了额外的非法项
            res1 = _eval_sum_hyper(f, i, a)
            if res1 is None or n_illegal(res1) > had:
                return
            res2 = _eval_sum_hyper(f, i, b + 1)
            if res2 is None or n_illegal(res2) > had:
                return
            # 获取结果和条件
            (res1, cond1), (res2, cond2) = res1, res2
            cond = And(cond1, cond2)
            if cond == False:
                return None
            # 返回 Piecewise 表达式，如果条件为 False，返回 None
            return Piecewise((res1 - res2, cond), (old_sum, True))
    # 如果 a 是负无穷，处理特殊情况
    if a is S.NegativeInfinity:
        # 计算替换 i 为 -i 后的超级和
        res1 = _eval_sum_hyper(f.subs(i, -i), i, 1)
        # 计算原始超级和
        res2 = _eval_sum_hyper(f, i, 0)
        # 如果任一结果为 None，则返回 None
        if res1 is None or res2 is None:
            return None
        # 解包结果和条件
        res1, cond1 = res1
        res2, cond2 = res2
        # 组合条件
        cond = And(cond1, cond2)
        # 如果条件为 False 或结果为空集，返回 None
        if cond == False or cond.as_set() == S.EmptySet:
            return None
        # 返回一个分段函数，根据条件选择 res1 + res2 或 old_sum
        return Piecewise((res1 + res2, cond), (old_sum, True))

    # 现在 b == oo，且 a != -oo
    # 计算超级和 f 在 i 从 a 到无穷的情况
    res = _eval_sum_hyper(f, i, a)
    # 如果结果不为 None
    if res is not None:
        r, c = res
        # 如果条件为 False
        if c == False:
            # 如果结果是数值
            if r.is_number:
                # 替换 f 中的 i 为 Dummy 变量 i+a，该变量为正整数
                f = f.subs(i, Dummy('i', integer=True, positive=True) + a)
                # 如果 f 是正数或零，返回正无穷
                if f.is_positive or f.is_zero:
                    return S.Infinity
                # 如果 f 是负数，返回负无穷
                elif f.is_negative:
                    return S.NegativeInfinity
            # 其他情况返回 None
            return None
        # 返回一个分段函数，根据条件选择 res 或 old_sum
        return Piecewise(res, (old_sum, True))
# 定义函数 eval_sum_residue，用于计算具有无穷级数的残余求和
def eval_sum_residue(f, i_a_b):
    r"""Compute the infinite summation with residues

    Notes
    =====

    If $f(n), g(n)$ are polynomials with $\deg(g(n)) - \deg(f(n)) \ge 2$,
    some infinite summations can be computed by the following residue
    evaluations.

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\cot(\pi x) \frac{f(x)}{g(x)}, \alpha)

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} (-1)^n \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\csc(\pi x) \frac{f(x)}{g(x)}, \alpha)

    Examples
    ========

    >>> from sympy import Sum, oo, Symbol
    >>> x = Symbol('x')

    Doubly infinite series of rational functions.

    >>> Sum(1 / (x**2 + 1), (x, -oo, oo)).doit()
    pi/tanh(pi)

    Doubly infinite alternating series of rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, -oo, oo)).doit()
    pi/sinh(pi)

    Infinite series of even rational functions.

    >>> Sum(1 / (x**2 + 1), (x, 0, oo)).doit()
    1/2 + pi/(2*tanh(pi))

    Infinite series of alternating even rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, 0, oo)).doit()
    pi/(2*sinh(pi)) + 1/2

    This also have heuristics to transform arbitrarily shifted summand or
    arbitrarily shifted summation range to the canonical problem the
    formula can handle.

    >>> Sum(1 / (x**2 + 2*x + 2), (x, -1, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 4*x + 5), (x, -2, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 1, oo)).doit()
    -1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 2, oo)).doit()
    -1 + pi/(2*tanh(pi))

    References
    ==========

    .. [#] http://www.supermath.info/InfiniteSeriesandtheResidueTheorem.pdf

    .. [#] Asmar N.H., Grafakos L. (2018) Residue Theory.
           In: Complex Analysis with Applications.
           Undergraduate Texts in Mathematics. Springer, Cham.
           https://doi.org/10.1007/978-3-319-94063-2_5
    """
    # 解构元组 i_a_b 到 i, a, b
    i, a, b = i_a_b

    def is_even_function(numer, denom):
        """Test if the rational function is an even function"""
        # 检查分子和分母的所有单项是否都是偶数次或奇数次
        numer_even = all(i % 2 == 0 for (i,) in numer.monoms())
        denom_even = all(i % 2 == 0 for (i,) in denom.monoms())
        numer_odd = all(i % 2 == 1 for (i,) in numer.monoms())
        denom_odd = all(i % 2 == 1 for (i,) in denom.monoms())
        return (numer_even and denom_even) or (numer_odd and denom_odd)

    def match_rational(f, i):
        # 将有理函数 f 分解成分子和分母，并尝试并行多项式转换
        numer, denom = f.as_numer_denom()
        try:
            (numer, denom), opt = parallel_poly_from_expr((numer, denom), i)
        except (PolificationFailed, PolynomialError):
            return None
        return numer, denom
    # 获取分母的所有根
    roots = denom.sqf_part().all_roots()
    # 筛选出所有整数根
    roots = sift(roots, lambda x: x.is_integer)
    # 如果根中包含 None，则返回 None
    if None in roots:
        return None
    # 将整数根和非整数根分别存储
    int_roots, nonint_roots = roots[True], roots[False]
    return int_roots, nonint_roots

    # 获取位移值
    n = denom.degree(i)
    a = denom.coeff_monomial(i**n)
    b = denom.coeff_monomial(i**(n-1))
    shift = - b / a / n
    return shift

# 创建一个没有附加假设的虚拟符号 z，以供 get_residue_factor 使用
z = Dummy('z')

# 计算残留因子
def get_residue_factor(numer, denom, alternating):
    # 将数值表达式转换为符号表达式，并用 z 替换 i
    residue_factor = (numer.as_expr() / denom.as_expr()).subs(i, z)
    # 根据 alternating 的值选择乘以 cot 或者 csc 函数
    if not alternating:
        residue_factor *= cot(S.Pi * z)
    else:
        residue_factor *= csc(S.Pi * z)
    return residue_factor

# 如果表达式 f 中除了 i 外还包含其它自由符号，则返回 None
if f.free_symbols - {i}:
    return None

# 如果 a 不是整数或者正负无穷，则返回 None
if not (a.is_Integer or a in (S.Infinity, S.NegativeInfinity)):
    return None
# 如果 b 不是整数或者正负无穷，则返回 None
if not (b.is_Integer or b in (S.Infinity, S.NegativeInfinity)):
    return None

# 如果范围不是无限的，则快速退出，返回 None
if a != S.NegativeInfinity and b != S.Infinity:
    return None

# 尝试匹配 f 是否为有理函数
match = match_rational(f, i)
if match:
    alternating = False
    numer, denom = match
else:
    # 如果匹配失败，则尝试匹配 f / (-1)**i 是否为有理函数
    match = match_rational(f / S.NegativeOne**i, i)
    if match:
        alternating = True
        numer, denom = match
    else:
        return None

# 如果分母的次数减去分子的次数小于 2，则返回 None
if denom.degree(i) - numer.degree(i) < 2:
    return None

# 如果范围是负无穷到正无穷，则处理以下逻辑
if (a, b) == (S.NegativeInfinity, S.Infinity):
    # 获取分母的极点
    poles = get_poles(denom)
    if poles is None:
        return None
    int_roots, nonint_roots = poles

    # 如果存在整数根，则返回 None
    if int_roots:
        return None

    # 计算残留因子
    residue_factor = get_residue_factor(numer, denom, alternating)
    # 计算所有非整数根处的残留并返回
    residues = [residue(residue_factor, z, root) for root in nonint_roots]
    return -S.Pi * sum(residues)

# 如果 b 不是正无穷，则返回 None
if not (a.is_finite and b is S.Infinity):
    return None

# 如果数值函数不是偶函数，则尝试位移和检查
if not is_even_function(numer, denom):
    # 获取位移值
    shift = get_shift(denom)

    # 如果位移值不是整数，则返回 None
    if not shift.is_Integer:
        return None
    # 如果位移值为 0，则返回 None
    if shift == 0:
        return None

    # 对分子和分母进行位移
    numer = numer.shift(shift)
    denom = denom.shift(shift)

    # 如果位移后函数仍不是偶函数，则返回 None
    if not is_even_function(numer, denom):
        return None

    # 根据 alternating 的值计算 f，并返回计算结果
    if alternating:
        f = S.NegativeOne**i * (S.NegativeOne**shift * numer.as_expr() / denom.as_expr())
    else:
        f = numer.as_expr() / denom.as_expr()
    return eval_sum_residue(f, (i, a-shift, b-shift))

# 获取分母的极点
poles = get_poles(denom)
if poles is None:
    return None
int_roots, nonint_roots = poles
    # 如果存在整数根，则将每个根转换为整数类型
    if int_roots:
        int_roots = [int(root) for root in int_roots]
        # 计算整数根的最大值和最小值
        int_roots_max = max(int_roots)
        int_roots_min = min(int_roots)
        # 整数值的极点必须相邻且关于原点对称（因为函数是偶函数）
        if not len(int_roots) == int_roots_max - int_roots_min + 1:
            return None

        # 检查极点的最大值是否小于等于参数 a
        if a <= max(int_roots):
            return None

    # 计算残余因子
    residue_factor = get_residue_factor(numer, denom, alternating)
    # 计算所有极点（整数和非整数）的残余
    residues = [residue(residue_factor, z, root) for root in int_roots + nonint_roots]
    # 计算所有残余的总和
    full_sum = -S.Pi * sum(residues)

    # 如果不存在整数根
    if not int_roots:
        # 计算 Sum(f, (i, 0, oo)) 通过在原点添加一个额外的评估
        half_sum = (full_sum + f.xreplace({i: 0})) / 2

        # 添加和减去额外的评估
        extraneous_neg = [f.xreplace({i: i0}) for i0 in range(int(a), 0)]
        extraneous_pos = [f.xreplace({i: i0}) for i0 in range(0, int(a))]
        result = half_sum + sum(extraneous_neg) - sum(extraneous_pos)

        return result

    # 计算 Sum(f, (i, min(poles) + 1, oo))
    half_sum = full_sum / 2

    # 减去额外的评估
    extraneous = [f.xreplace({i: i0}) for i0 in range(max(int_roots) + 1, int(a))]
    result = half_sum - sum(extraneous)

    return result
# 计算矩阵求和表达式的结果
def _eval_matrix_sum(expression):
    # 获取表达式的函数部分
    f = expression.function
    # 遍历表达式的限制条件
    for limit in expression.limits:
        # 解析限制条件，i, a, b 分别表示索引变量、起始值、结束值
        i, a, b = limit
        # 计算起始值和结束值的差值
        dif = b - a
        # 检查差值是否为整数
        if dif.is_Integer:
            # 如果差值小于0，则交换起始值和结束值，并反转函数 f 的符号
            if (dif < 0) == True:
                a, b = b + 1, a - 1
                f = -f

            # 使用直接求和函数 eval_sum_direct 计算求和结果
            newf = eval_sum_direct(f, (i, a, b))
            # 如果计算结果不为 None，则进行计算并返回结果
            if newf is not None:
                return newf.doit()


# 返回一个 Dummy 符号，该符号尽可能继承给定符号和限制条件的所有假设
def _dummy_with_inherited_properties_concrete(limits):
    """
    Return a Dummy symbol that inherits as many assumptions as possible
    from the provided symbol and limits.

    If the symbol already has all True assumption shared by the limits
    then return None.
    """
    # 解析限制条件，x, a, b 分别表示符号变量、起始值、结束值
    x, a, b = limits
    # 构建限制值列表
    l = [a, b]

    # 考虑要继承的假设列表
    assumptions_to_consider = ['extended_nonnegative', 'nonnegative',
                               'extended_nonpositive', 'nonpositive',
                               'extended_positive', 'positive',
                               'extended_negative', 'negative',
                               'integer', 'rational', 'finite',
                               'zero', 'real', 'extended_real']

    # 初始化要保留和要添加的假设字典
    assumptions_to_keep = {}
    assumptions_to_add = {}
    
    # 遍历假设列表，检查是否符号变量已具备相应假设或限制值均满足
    for assum in assumptions_to_consider:
        assum_true = x._assumptions.get(assum, None)
        if assum_true:
            assumptions_to_keep[assum] = True
        elif all(getattr(i, 'is_' + assum) for i in l):
            assumptions_to_add[assum] = True
    
    # 如果有需要添加的假设，则更新保留的假设字典并返回一个带有这些假设的 Dummy 符号
    if assumptions_to_add:
        assumptions_to_keep.update(assumptions_to_add)
        return Dummy('d', **assumptions_to_keep)
```