# `D:\src\scipysrc\sympy\sympy\series\formal.py`

```
"""Formal Power Series"""

from collections import defaultdict  # 导入 defaultdict 类，用于创建默认字典

from sympy.core.numbers import (nan, oo, zoo)  # 导入数学常量 nan, oo, zoo
from sympy.core.add import Add  # 导入加法表达式类 Add
from sympy.core.expr import Expr  # 导入表达式基类 Expr
from sympy.core.function import Derivative, Function, expand  # 导入导数、函数、展开函数
from sympy.core.mul import Mul  # 导入乘法表达式类 Mul
from sympy.core.numbers import Rational  # 导入有理数类 Rational
from sympy.core.relational import Eq  # 导入等式类 Eq
from sympy.sets.sets import Interval  # 导入区间类 Interval
from sympy.core.singleton import S  # 导入单例类 S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol  # 导入符号相关类和函数
from sympy.core.sympify import sympify  # 导入 sympify 函数
from sympy.discrete.convolutions import convolution  # 导入卷积函数
from sympy.functions.combinatorial.factorials import binomial, factorial, rf  # 导入组合数学函数
from sympy.functions.combinatorial.numbers import bell  # 导入贝尔数函数
from sympy.functions.elementary.integers import floor, frac, ceiling  # 导入整数相关函数
from sympy.functions.elementary.miscellaneous import Min, Max  # 导入最小值和最大值函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数类 Piecewise
from sympy.series.limits import Limit  # 导入极限类 Limit
from sympy.series.order import Order  # 导入阶数类 Order
from sympy.series.sequences import sequence  # 导入序列类 sequence
from sympy.series.series_class import SeriesBase  # 导入级数基类 SeriesBase
from sympy.utilities.iterables import iterable  # 导入可迭代工具函数

def rational_algorithm(f, x, k, order=4, full=False):
    """
    Rational algorithm for computing
    formula of coefficients of Formal Power Series
    of a function.

    Explanation
    ===========

    Applicable when f(x) or some derivative of f(x)
    is a rational function in x.

    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction
    decomposition. :func:`~.apart` by default uses 'undetermined coefficients
    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used
    instead.

    Looks for derivative of a function up to 4'th order (by default).
    This can be overridden using order option.

    Parameters
    ==========

    x : Symbol
        符号变量 x
    order : int, optional
        求导阶数，缺省为 4
    full : bool
        是否完全计算

    Returns
    =======

    formula : Expr
        表达式
    ind : Expr
        独立项
    order : int
        求导阶数
    full : bool
        是否完全计算

    Examples
    ========

    >>> from sympy import log, atan
    >>> from sympy.series.formal import rational_algorithm as ra
    >>> from sympy.abc import x, k

    >>> ra(1 / (1 - x), x, k)
    (1, 0, 0)
    >>> ra(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)

    >>> ra(atan(x), x, k, full=True)
    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)

    Notes
    =====

    By setting ``full=True``, range of admissible functions to be solved using
    ``rational_algorithm`` can be increased. This option should be used
    carefully as it can significantly slow down the computation as ``doit`` is
    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`
    function. Use ``full=False`` whenever possible.

    See Also
    ========

    sympy.polys.partfrac.apart

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf

    """
    # 导入 sympy 库中的 RootSum 和 apart 函数
    from sympy.polys import RootSum, apart
    # 导入 sympy 库中的 integrate 函数
    from sympy.integrals import integrate

    # 将 f 赋值给 diff
    diff = f
    # 初始化一个空列表 ds，用于存储 diff 变量
    ds = []  # list of diff

    # 循环从 0 到 order，共执行 order + 1 次
    for i in range(order + 1):
        # 如果 i 不为 0，则对 diff 关于变量 x 求偏导数
        if i:
            diff = diff.diff(x)

        # 检查 diff 是否为有理函数
        if diff.is_rational_function(x):
            # 初始化 coeff 和 sep 为 0
            coeff, sep = S.Zero, S.Zero

            # 将 diff 分解成部分分式
            terms = apart(diff, x, full=full)
            # 如果 terms 中包含 RootSum，则对其进行计算
            if terms.has(RootSum):
                terms = terms.doit()

            # 遍历 terms 中的每一项
            for t in Add.make_args(terms):
                # 将每一项 t 分离为分子和分母
                num, den = t.as_numer_denom()
                # 如果分母中不包含 x，则将 t 添加到 sep 中
                if not den.has(x):
                    sep += t
                else:
                    # 如果分母是 Mul 类型，则将其转化为独立部分
                    if isinstance(den, Mul):
                        ind = den.as_independent(x)
                        den = ind[1]
                        num /= ind[0]

                    # 将分母转化为基底和指数
                    den, j = den.as_base_exp()
                    a, xterm = den.as_coeff_add(x)

                    # 如果 a 为 0，则将 t 添加到 sep 中并继续下一次循环
                    if not a:
                        sep += t
                        continue

                    # 计算系数 a 的相反数和 num 的修正值
                    xc = xterm[0].coeff(x)
                    a /= -xc
                    num /= xc**j

                    # 计算 ak 的值
                    ak = ((-1)**j * num *
                          binomial(j + k - 1, k).rewrite(factorial) /
                          a**(j + k))
                    coeff += ak

            # 如果 coeff 为零，则返回 None
            if coeff.is_zero:
                return None
            # 如果 coeff 中包含特定的符号，则返回 None
            if (coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or
                    coeff.has(nan)):
                return None

            # 对于 j 从 0 到 i-1 的每一个值
            for j in range(i):
                # 计算 coeff 和 sep 的修正值
                coeff = (coeff / (k + j + 1))
                sep = integrate(sep, x)
                sep += (ds.pop() - sep).limit(x, 0)  # 积分常数
            # 返回包含 coeff、sep 和 i 的元组
            return (coeff.subs(k, k - i), sep, i)

        else:
            # 将 diff 添加到 ds 列表中
            ds.append(diff)

    # 如果未找到有理函数的情况下返回 None
    return None
def rational_independent(terms, x):
    """
    Returns a list of all the rationally independent terms.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.series.formal import rational_independent
    >>> from sympy.abc import x

    >>> rational_independent([cos(x), sin(x)], x)
    [cos(x), sin(x)]
    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)
    [x**3 + x**2, x*sin(x) + sin(x)]
    """
    if not terms:
        return []

    # Initialize the list of rationally independent terms with the first term
    ind = terms[0:1]

    # Iterate over the remaining terms
    for t in terms[1:]:
        # Extract the coefficient (denoted as 'n') of the term 't'
        n = t.as_independent(x)[1]
        # Try to find a rational function combination in 'ind' for 't'
        for i, term in enumerate(ind):
            # Extract the coefficient (denoted as 'd') of the term 'term' in 'ind'
            d = term.as_independent(x)[1]
            # Compute the quotient of coefficients and simplify
            q = (n / d).cancel()
            # Check if 'q' is a rational function in 'x'
            if q.is_rational_function(x):
                # If so, add 't' to the corresponding term in 'ind'
                ind[i] += t
                break
        else:
            # If no rational function combination found, add 't' as a new term in 'ind'
            ind.append(t)

    # Return the list of rationally independent terms
    return ind


def simpleDE(f, x, g, order=4):
    r"""
    Generates simple DE.

    Explanation
    ===========

    DE is of the form

    .. math::
        f^k(x) + \sum\limits_{j=0}^{k-1} A_j f^j(x) = 0

    where :math:`A_j` should be rational function in x.

    Generates DE's upto order 4 (default). DE's can also have free parameters.

    By increasing order, higher order DE's can be found.

    Yields a tuple of (DE, order).
    """
    from sympy.solvers.solveset import linsolve

    # Define symbols a_0, a_1, ..., a_(order-1)
    a = symbols('a:%d' % (order))

    # Function to generate a differential equation of order 'k'
    def _makeDE(k):
        # Construct the differential equation for 'f(x)'
        eq = f.diff(x, k) + Add(*[a[i]*f.diff(x, i) for i in range(0, k)])
        # Construct the differential equation for 'g(x)'
        DE = g(x).diff(x, k) + Add(*[a[i]*g(x).diff(x, i) for i in range(0, k)])
        return eq, DE

    found = False
    # Iterate over orders from 1 to 'order'
    for k in range(1, order + 1):
        # Generate the differential equation 'eq' and its corresponding 'DE' for order 'k'
        eq, DE = _makeDE(k)
        # Expand 'eq' into an ordered list of terms
        eq = eq.expand()
        terms = eq.as_ordered_terms()
        # Find the rationally independent terms in 'eq'
        ind = rational_independent(terms, x)
        # If a solution is found or the number of independent terms equals 'k'
        if found or len(ind) == k:
            # Solve the linear system defined by 'ind' for 'a'
            sol = dict(zip(a, (i for s in linsolve(ind, a[:k]) for i in s)))
            if sol:
                found = True
                # Substitute the solution 'sol' into 'DE'
                DE = DE.subs(sol)
            # Extract the numerator of 'DE'
            DE = DE.as_numer_denom()[0]
            # Factorize 'DE' and extract the coefficient of the highest derivative
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            # Yield the differential equation and its order
            yield DE.collect(Derivative(g(x))), k


def exp_re(DE, r, k):
    """Converts a DE with constant coefficients (explike) into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        f^j(x) \\to r(k + j)

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import exp_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> exp_re(-f(x) + Derivative(f(x)), r, k)
    -r(k) + r(k + 1)
    >>> exp_re(Derivative(f(x), x) + Derivative(f(x), (x, 2)), r, k)
    r(k) + r(k + 1)

    See Also
    ========

    sympy.series.formal.hyper_re
    """
    # Initialize RE as zero
    RE = S.Zero

    # Extract the function 'g' from differential equation 'DE'
    g = DE.atoms(Function).pop()

    # 'mini' is not initialized in the provided code snippet, leaving as is
    # 遍历 Add.make_args(DE) 返回的迭代器，对每个元素执行以下操作
    for t in Add.make_args(DE):
        # 将 t 表示为以 g 为独立变量的系数和导数的元组
        coeff, d = t.as_independent(g)
        
        # 如果 d 是 Derivative 类型的实例，则获取其导数次数
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            # 否则导数次数 j 设为 0
            j = 0
        
        # 如果 mini 为 None 或者 j 比 mini 小，则更新 mini
        if mini is None or j < mini:
            mini = j
        
        # 将 RE 增加 coeff * r(k + j) 的值
        RE += coeff * r(k + j)
    
    # 如果 mini 不为 0，则对 RE 中的 k 进行替换，将 k 减去 mini
    if mini:
        RE = RE.subs(k, k - mini)
    
    # 返回最终计算结果 RE
    return RE
# 递归求解超几何级数的函数
def _rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Recursive wrapper to rsolve_hypergeometric.

    Explanation
    ===========

    This function recursively solves the hypergeometric series equation.

    Parameters
    ==========
    f : Expression
        The function expression to solve.
    x : Symbol
        The independent variable of the function.
    P : Expression
        The numerator polynomial in terms of k.
    Q : Expression
        The denominator polynomial in terms of k.
    k : Symbol
        The parameter symbol in the hypergeometric series.
    m : Integer
        Parameter for the hypergeometric series.

    Returns
    =======
    Tuple
        A tuple containing:
        - Formula: The solved hypergeometric formula as a list of tuples.
        - Series independent terms: The series terms independent of the parameter k.
        - Series specific terms: Terms specific to the hypergeometric series.

    Notes
    =====
    This function handles the recursive resolution of the hypergeometric series
    equation, utilizing polynomial roots and factorial calculations.

    Examples
    ========
    >>> from sympy import Symbol, S
    >>> k = Symbol('k')
    >>> x = Symbol('x')
    >>> P = S(1)
    >>> Q = S(1)
    >>> f = S(1)
    >>> m = 1
    >>> _rsolve_hypergeometric(f, x, P, Q, k, m)
    ([], [], [])

    See Also
    ========
    sympy.series.formal.exp_re
    """
    from sympy.polys import roots

    sol = []
    # 循环计算超几何级数的解
    for i in range(k_max + 1, k_max + m + 1):
        # 如果 i 小于 0，继续下一个循环
        if (i < 0) == True:
            continue
        # 计算 f 在 x 处对 x 的 i 次导数的极限
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        # 如果 r 是零，则跳过当前循环
        if r.is_zero:
            continue

        kterm = m*k + i
        res = r

        # 替换 P 和 Q 中的 kterm，并计算其主导项
        p = P.subs(k, kterm)
        q = Q.subs(k, kterm)
        c1 = p.subs(k, 1/k).leadterm(k)[0]
        c2 = q.subs(k, 1/k).leadterm(k)[0]
        res *= (-c1 / c2)**k

        # 计算 P 和 Q 的根，并分别乘以其指数
        res *= Mul(*[rf(-r, k)**mul for r, mul in roots(p, k).items()])
        res /= Mul(*[rf(-r, k)**mul for r, mul in roots(q, k).items()])

        # 将结果添加到解的列表中
        sol.append((res, kterm))

    return sol
    # 导入所需的符号计算模块中的函数
    from sympy.polys import lcm, roots
    from sympy.integrals import integrate

    # 变换 - c
    # 计算多项式 P 和 Q 的根
    proots, qroots = roots(P, k), roots(Q, k)
    # 将所有根合并到一个字典中
    all_roots = dict(proots)
    all_roots.update(qroots)
    # 计算所有根的最小公倍数的分母，用于后续的尺度变换
    scale = lcm([r.as_numer_denom()[1] for r, t in all_roots.items()
                 if r.is_rational])
    # 应用变换 c，并更新相关参数
    f, P, Q, m = _transformation_c(f, x, P, Q, k, m, scale)

    # 变换 - a
    # 计算多项式 Q 的根
    qroots = roots(Q, k)
    if qroots:
        k_min = Min(*qroots.keys())
    else:
        k_min = S.Zero
    # 计算移动量 shift，并更新相关参数
    shift = k_min + m
    f, P, Q, m = _transformation_a(f, x, P, Q, k, m, shift)

    # 计算 f 在 x 趋向于 0 时的极限值
    l = (x*f).limit(x, 0)
    # 如果极限值不是 Limit 对象且不等于 0，则返回 None
    if not isinstance(l, Limit) and l != 0:  # Ideally should only be l != 0
        return None

    # 计算多项式 Q 的根
    qroots = roots(Q, k)
    if qroots:
        k_max = Max(*qroots.keys())
    else:
        k_max = S.Zero

    # 初始化指标 ind 和最大幂次 mp
    ind, mp = S.Zero, -oo
    # 遍历范围为 k_max + m + 1 的整数 i
    for i in range(k_max + m + 1):
        # 计算 f 关于 x 的 i 阶导数在 x 趋向于 0 时的极限值，并除以 i 的阶乘
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        # 如果 r 不是有限数值，则进行变换并递归求解
        if r.is_finite is False:
            # 保存旧的 f
            old_f = f
            # 应用变换 a，并更新相关参数
            f, P, Q, m = _transformation_a(f, x, P, Q, k, m, i)
            # 应用变换 e，并更新相关参数
            f, P, Q, m = _transformation_e(f, x, P, Q, k, m)
            # 递归求解超几何级数的解 sol，指标 ind，最大幂次 mp
            sol, ind, mp = _rsolve_hypergeometric(f, x, P, Q, k, m)
            # 对解应用积分操作，并更新指标 ind
            sol = _apply_integrate(sol, x, k)
            sol = _apply_shift(sol, i)
            ind = integrate(ind, x)
            ind += (old_f - ind).limit(x, 0)  # 积分常数项
            mp += 1
            return sol, ind, mp
        # 如果 r 存在且不为空，则更新指标 ind 和最大幂次 mp
        elif r:
            ind += r*x**(i + shift)
            pow_x = Rational((i + shift), scale)
            # 如果当前幂次大于 mp，则更新 mp
            if pow_x > mp:
                mp = pow_x  # 最大的幂次 x
    # 对 ind 进行 x 的 scale 次方根的替换
    ind = ind.subs(x, x**(1/scale))

    # 计算公式的解 sol
    sol = _compute_formula(f, x, P, Q, k, m, k_max)
    sol = _apply_shift(sol, shift)
    sol = _apply_scale(sol, scale)

    # 返回计算结果 sol，指标 ind，最大幂次 mp
    return sol, ind, mp
# 解决超几何类型的递归方程(RE)
def rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Solves RE of hypergeometric type.

    Explanation
    ===========

    Attempts to solve RE of the form

    Q(k)*a(k + m) - P(k)*a(k)

    Transformations that preserve Hypergeometric type:

        a. x**n*f(x): b(k + m) = R(k - n)*b(k)
        b. f(A*x): b(k + m) = A**m*R(k)*b(k)
        c. f(x**n): b(k + n*m) = R(k/n)*b(k)
        d. f(x**(1/m)): b(k + 1) = R(k*m)*b(k)
        e. f'(x): b(k + m) = ((k + m + 1)/(k + 1))*R(k + 1)*b(k)

    Some of these transformations have been used to solve the RE.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import exp, ln, S
    >>> from sympy.series.formal import rsolve_hypergeometric as rh
    >>> from sympy.abc import x, k

    >>> rh(exp(x), x, -S.One, (k + 1), k, 1)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> rh(ln(1 + x), x, k**2, k*(k + 1), k, 1)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf
    """
    # 调用内部函数 _rsolve_hypergeometric 处理递归方程
    result = _rsolve_hypergeometric(f, x, P, Q, k, m)

    # 如果未能解出结果，则返回 None
    if result is None:
        return None

    sol_list, ind, mp = result

    # 创建默认字典，用于存储解的条件
    sol_dict = defaultdict(lambda: S.Zero)
    # 遍历解的列表
    for res, cond in sol_list:
        # 将条件 cond 分解为 j 和 mk 的线性组合
        j, mk = cond.as_coeff_Add()
        c = mk.coeff(k)

        # 如果 j 不是整数，则乘以 x 的 j 次幂并向下取整
        if j.is_integer is False:
            res *= x**frac(j)
            j = floor(j)

        # 替换 k 为 (k - j) / c，并用条件 Eq(k % c, j % c) 来表示
        res = res.subs(k, (k - j) / c)
        cond = Eq(k % c, j % c)
        # 将结果 res 根据条件 cond 分组存储在 sol_dict 中
        sol_dict[cond] += res  # 将具有相同条件的公式组合在一起

    # 将 sol_dict 转换为 [(res, cond), ...] 的列表形式，并添加一个默认条件 (S.Zero, True)
    sol = [(res, cond) for cond, res in sol_dict.items()]
    sol.append((S.Zero, True))
    # 创建分段函数 Piecewise
    sol = Piecewise(*sol)

    # 如果 mp 是负无穷，则 s 为 S.Zero；否则根据 mp 的整数性进行不同的计算
    if mp is -oo:
        s = S.Zero
    elif mp.is_integer is False:
        s = ceiling(mp)
    else:
        s = mp + 1

    # 如果 s 小于 0，则将所有形如 sol * x**k 的项添加到 ind 中
    if s < 0:
        ind += sum(sequence(sol * x**k, (k, s, -1)))
        s = S.Zero

    # 返回结果元组 (sol, ind, s)
    return (sol, ind, s)


def _solve_hyper_RE(f, x, RE, g, k):
    """See docstring of :func:`rsolve_hypergeometric` for details."""
    # 将递归方程 RE 转换为项列表
    terms = Add.make_args(RE)

    # 如果项的数量为 2，则尝试解递归方程
    if len(terms) == 2:
        gs = list(RE.atoms(Function))
        P, Q = map(RE.coeff, gs)
        m = gs[1].args[0] - gs[0].args[0]
        if m < 0:
            P, Q = Q, P
            m = abs(m)
        # 调用 rsolve_hypergeometric 函数解递归方程
        return rsolve_hypergeometric(f, x, P, Q, k, m)


def _solve_explike_DE(f, x, DE, g, k):
    """Solves DE with constant coefficients."""
    from sympy.solvers import rsolve

    # 遍历方程 DE 的各项
    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        # 如果某项包含自由符号，则返回 None
        if coeff.free_symbols:
            return

    # 将差分方程 DE 转换为指数型递归方程 RE
    RE = exp_re(DE, g, k)

    # 初始化字典
    init = {}
    # 遍历 range(len(Add.make_args(RE))) 返回的索引范围
    for i in range(len(Add.make_args(RE))):
        # 如果 i 不为 0
        if i:
            # 对函数 f 关于变量 x 求导数
            f = f.diff(x)
        # 用 g(k) 替换 k 后初始化 init 字典
        init[g(k).subs(k, i)] = f.limit(x, 0)

    # 使用 rsolve 求解差分方程 RE，初始条件为 g(k) 和 init 字典
    sol = rsolve(RE, g(k), init)

    # 如果求解成功
    if sol:
        # 返回解 sol 除以 k 的阶乘，以及 S.Zero 两次
        return (sol / factorial(k), S.Zero, S.Zero)
def solve_de(f, x, DE, order, g, k):
    """
    Solves the differential equation (DE).

    Explanation
    ===========

    Tries to solve the differential equation by either converting it into a recurrence equation (RE)
    containing two terms or converting it into a differential equation (DE) with constant coefficients.

    Returns
    =======

    formula : Expr
        The solution formula for the differential equation.
    ind : Expr
        Independent terms or conditions under which the solution applies.
    order : int
        Order of the differential equation.

    Examples
    ========

    >>> from sympy import Derivative as D, Function
    >>> from sympy import exp, ln
    >>> from sympy.series.formal import solve_de
    >>> from sympy.abc import x, k
    >>> f = Function('f')

    >>> solve_de(exp(x), x, D(f(x), x) - f(x), 1, f, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> solve_de(ln(1 + x), x, (x + 1)*D(f(x), x, 2) + D(f(x)), 2, f, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)
    """

    sol = None
    # Determine symbols in the differential equation other than g and x
    syms = DE.free_symbols.difference({g, x})

    if syms:
        # Transform the differential equation into a recurrence equation of hypergeometric type
        RE = _transform_DE_RE(DE, g, k, order, syms)
    else:
        # Use the hypergeometric representation of the differential equation
        RE = hyper_re(DE, g, k)

    # Return the recurrence equation (RE)
    return RE
    # 如果 RE 的自由符号集合与 {k} 的差集为空，则执行下面的代码块
    if not RE.free_symbols.difference({k}):
        # 调用 _solve_hyper_RE 函数，解决超越方程 RE 关于变量 x 的值，得到解 sol
        sol = _solve_hyper_RE(f, x, RE, g, k)

    # 如果 sol 非空，则返回解 sol
    if sol:
        return sol

    # 如果 syms 非空，则对于类似显式形式的微分方程 DE，使用 _transform_explike_DE 函数进行变换
    if syms:
        # 对显式形式的微分方程 DE 进行变换，使用 g、x、order、syms 作为参数
        DE = _transform_explike_DE(DE, g, x, order, syms)
    
    # 如果 DE 的自由符号集合与 {x} 的差集为空，则执行下面的代码块
    if not DE.free_symbols.difference({x}):
        # 调用 _solve_explike_DE 函数，解决显式形式的微分方程 DE 关于变量 x 的值，得到解 sol
        sol = _solve_explike_DE(f, x, DE, g, k)

    # 如果 sol 非空，则返回解 sol
    if sol:
        return sol
# 定义一个函数，使用超几何算法计算形式幂级数的解
def hyper_algorithm(f, x, k, order=4):
    """
    Hypergeometric algorithm for computing Formal Power Series.

    Explanation
    ===========

    Steps:
        * Generates DE 生成微分方程
        * Convert the DE into RE 将微分方程转换为递推方程
        * Solves the RE 解递推方程

    Examples
    ========

    >>> from sympy import exp, ln
    >>> from sympy.series.formal import hyper_algorithm

    >>> from sympy.abc import x, k

    >>> hyper_algorithm(exp(x), x, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> hyper_algorithm(ln(1 + x), x, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    See Also
    ========

    sympy.series.formal.simpleDE
    sympy.series.formal.solve_de
    """
    # 定义一个符号函数 g
    g = Function('g')

    # 初始化微分方程列表
    des = []  # list of DE's
    sol = None

    # 使用简单微分方程函数 simpleDE 进行迭代
    for DE, i in simpleDE(f, x, g, order):
        # 如果存在微分方程，则尝试解微分方程
        if DE is not None:
            sol = solve_de(f, x, DE, i, g, k)
        # 如果找到解，则返回解
        if sol:
            return sol
        # 如果微分方程中只包含变量 x，则将其添加到待处理的微分方程列表中
        if not DE.free_symbols.difference({x}):
            des.append(DE)

    # 如果以上方法都无法求解，则尝试直接使用 rsolve 函数
    for DE in des:
        sol = _solve_simple(f, x, DE, g, k)
        if sol:
            return sol


# 定义一个递归包装函数，用于计算形式幂级数
def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
    # 处理 x0 为无穷大或负无穷大的情况
    if x0 in [S.Infinity, S.NegativeInfinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1/x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1/x), result[2].subs(x, 1/x))
    # 处理 x0 不为零或 dir 为 -1 的情况
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b),
                result[2].subs(x, rep2 + rep2b))

    # 如果 f 是多项式，则直接返回系数序列和幂级数
    if f.is_polynomial(x):
        k = Dummy('k')
        ak = sequence(Coeff(f, x, k), (k, 1, oo))
        xk = sequence(x**k, (k, 0, oo))
        ind = f.coeff(x, 0)
        return ak, xk, ind

    # 打破 Add 实例，允许不同项使用不同的算法
    # 增加可接受函数范围
    # 如果 f 是 Add 类型的表达式
    if isinstance(f, Add):
        # 初始化结果为 False
        result = False
        # 创建序列 ak = 0, 1, 2, ...
        ak = sequence(S.Zero, (0, oo))
        # 初始化索引和 xk
        ind, xk = S.Zero, None
        # 遍历 f 中的每个子项 t
        for t in Add.make_args(f):
            # 调用 _compute_fps 计算 t 关于 x 的 Fourier 系数
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            # 如果 res 非空
            if res:
                # 如果结果为 False，则更新为 True，并设置 xk
                if not result:
                    result = True
                    xk = res[1]
                # 比较 res 的起始点和 ak 的起始点，选择合适的序列
                if res[0].start > ak.start:
                    seq = ak
                    s, f = ak.start, res[0].start
                else:
                    seq = res[0]
                    s, f = res[0].start, ak.start
                # 计算保存的项并更新 ak 和 ind
                save = Add(*[z[0]*z[1] for z in zip(seq[0:(f - s)], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                # 如果 res 为空，则直接将 t 加到 ind 中
                ind += t
        # 如果找到了结果，则返回 ak, xk, ind；否则返回 None
        if result:
            return ak, xk, ind
        return None
    
    # 从 f 中提取自由符号（除了 x），并将 f 展开，使得 symb 与函数分离
    syms = f.free_symbols.difference({x})
    (f, symb) = expand(f).as_independent(*syms)

    result = None

    # 如果 rational 为 True，则使用 rational_algorithm 处理 f
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)

    # 如果 result 仍为 None 且 hyper 为 True，则使用 hyper_algorithm 处理 f
    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)

    # 如果 result 仍为 None，则返回 None
    if result is None:
        return None

    # 导入 powsimp 函数，并对 symb 进行处理
    from sympy.simplify.powsimp import powsimp
    # 如果 symb 是零，则将其设为 1；否则进行 powsimp 处理
    if symb.is_zero:
        symb = S.One
    else:
        symb = powsimp(symb)

    # 创建序列 ak 和 xk，并对结果进行 powsimp 处理
    ak = sequence(result[0], (k, result[2], oo))
    xk_formula = powsimp(x**k * symb)
    xk = sequence(xk_formula, (k, 0, oo))
    ind = powsimp(result[1] * symb)

    # 返回结果 ak, xk, ind
    return ak, xk, ind
# 定义一个函数 compute_fps，用于计算函数的形式幂级数公式
def compute_fps(f, x, x0=0, dir=1, hyper=True, order=4, rational=True,
                full=False):
    """
    Computes the formula for Formal Power Series of a function.

    Explanation
    ===========

    Tries to compute the formula by applying the following techniques
    (in order):

    * rational_algorithm
    * Hypergeometric algorithm

    Parameters
    ==========

    x : Symbol
        符号变量，表示自变量
    x0 : number, optional
        级数展开的中心点，默认为0
    dir : {1, -1, '+', '-'}, optional
        dir为1或'+'时，从右侧计算级数；为-1或'-'时，从左侧计算级数。
        对于光滑函数，此标志不会改变结果。默认为1。
    hyper : {True, False}, optional
        是否使用超几何算法。设置为False则跳过超几何算法，默认为True。
    order : int, optional
        函数f的导数阶数，默认为4。
    rational : {True, False}, optional
        是否使用有理算法。设置为False则跳过有理算法，默认为True。
    full : {True, False}, optional
        是否使用全范围的有理算法。详见 rational_algorithm 的说明。
        默认为False。

    Returns
    =======

    ak : sequence
        系数的序列
    xk : sequence
        x的幂次序列
    ind : Expr
        独立项
    mul : Pow
        公共项

    See Also
    ========

    sympy.series.formal.rational_algorithm
    sympy.series.formal.hyper_algorithm
    """
    # 将输入的函数f和自变量x转换为Sympy表达式
    f = sympify(f)
    x = sympify(x)

    # 如果函数f中不包含自变量x，则返回None
    if not f.has(x):
        return None

    # 将x0转换为Sympy表达式
    x0 = sympify(x0)

    # 处理dir参数，将其转换为Sympy表达式
    if dir == '+':
        dir = S.One
    elif dir == '-':
        dir = -S.One
    elif dir not in [S.One, -S.One]:
        raise ValueError("Dir must be '+' or '-'")
    else:
        dir = sympify(dir)

    # 调用内部函数 _compute_fps 来计算形式幂级数
    return _compute_fps(f, x, x0, dir, hyper, order, rational, full)


class Coeff(Function):
    """
    Coeff(p, x, n) represents the nth coefficient of the polynomial p in x
    """
    @classmethod
    def eval(cls, p, x, n):
        # 如果p是关于x的多项式，并且n是整数，则返回p在x中的第n个系数
        if p.is_polynomial(x) and n.is_integer:
            return p.coeff(x, n)


class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """
    def __new__(cls, *args):
        # 将输入的所有参数转换为Sympy表达式
        args = map(sympify, args)
        # 调用父类Expr的构造函数来创建一个新的实例
        return Expr.__new__(cls, *args)
    def __init__(self, *args):
        # 从传入参数中获取第五个元素的第一个变量
        ak = args[4][0]
        # 获取第一个变量的第一个符号
        k = ak.variables[0]
        # 创建一个从1到正无穷的序列，序列的表达式是第五个参数的公式
        self.ak_seq = sequence(ak.formula, (k, 1, oo))
        # 创建一个阶乘序列，阶乘的变量从1到正无穷
        self.fact_seq = sequence(factorial(k), (k, 1, oo))
        # 计算阿凯序列和阶乘序列的乘积，生成贝尔多项式系数序列
        self.bell_coeff_seq = self.ak_seq * self.fact_seq
        # 创建一个符号序列，序列值为 (-1, 1)，变量从1到正无穷
        self.sign_seq = sequence((-1, 1), (k, 1, oo))

    @property
    def function(self):
        # 返回对象的第一个参数，表示函数
        return self.args[0]

    @property
    def x(self):
        # 返回对象的第二个参数，表示 x
        return self.args[1]

    @property
    def x0(self):
        # 返回对象的第三个参数，表示 x0
        return self.args[2]

    @property
    def dir(self):
        # 返回对象的第四个参数，表示 dir 方向
        return self.args[3]

    @property
    def ak(self):
        # 返回对象的第五个参数的第一个元素，表示 ak
        return self.args[4][0]

    @property
    def xk(self):
        # 返回对象的第五个参数的第二个元素，表示 xk
        return self.args[4][1]

    @property
    def ind(self):
        # 返回对象的第五个参数的第三个元素，表示 ind
        return self.args[4][2]

    @property
    def interval(self):
        # 返回一个区间对象，范围从 0 到正无穷
        return Interval(0, oo)

    @property
    def start(self):
        # 返回区间对象的下界，即 0
        return self.interval.inf

    @property
    def stop(self):
        # 返回区间对象的上界，即正无穷
        return self.interval.sup

    @property
    def length(self):
        # 返回正无穷
        return oo

    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
        from sympy.concrete import Sum
        ak, xk = self.ak, self.xk
        k = ak.variables[0]
        # 构建一个求和对象，求和表达式为 ak.formula * xk.formula
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))

        return self.ind + inf_sum

    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
        # 将 term 表达式中独立于 x 的部分提取出来，并返回其指数部分
        xterm, pow_x = term.as_independent(self.x)[1].as_base_exp()
        # 如果 xterm 中不包含 x，则返回 0
        if not xterm.has(self.x):
            return S.Zero
        return pow_x

    def polynomial(self, n=6):
        """
        Truncated series as polynomial.

        Explanation
        ===========

        Returns series expansion of ``f`` upto order ``O(x**n)``
        as a polynomial(without ``O`` term).
        """
        terms = []
        sym = self.free_symbols
        # 遍历序列中的项，将符合条件的项添加到 terms 列表中
        for i, t in enumerate(self):
            xp = self._get_pow_x(t)
            # 如果指数包含任何自由符号，则将其转换为整数
            if xp.has(*sym):
                xp = xp.as_coeff_add(*sym)[0]
            # 如果指数大于等于 n，则结束循环
            if xp >= n:
                break
            # 如果指数是整数且 i 等于 n+1，则结束循环
            elif xp.is_integer is True and i == n + 1:
                break
            # 如果 t 不为零，则将其添加到 terms 中
            elif t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def truncate(self, n=6):
        """
        Truncated series.

        Explanation
        ===========

        Returns truncated series expansion of f upto
        order ``O(x**n)``.

        If n is ``None``, returns an infinite iterator.
        """
        if n is None:
            return iter(self)

        x, x0 = self.x, self.x0
        # 获取 xk 的第 n 项系数
        pt_xk = self.xk.coeff(n)
        # 如果 x0 是负无穷，则将其替换为正无穷
        if x0 is S.NegativeInfinity:
            x0 = S.Infinity

        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def zero_coeff(self):
        # 返回零阶系数的计算结果
        return self._eval_term(0)
    # 计算给定表达式的项的值
    def _eval_term(self, pt):
        try:
            # 获取 xk 中 pt 位置的系数
            pt_xk = self.xk.coeff(pt)
            # 获取 ak 中 pt 位置的系数，并简化其表达式
            pt_ak = self.ak.coeff(pt).simplify()  # 简化系数
        except IndexError:
            # 如果索引错误，则将 term 设为零
            term = S.Zero
        else:
            # 计算 term 的值为 ak * xk
            term = (pt_ak * pt_xk)

        # 如果存在 self.ind，计算其值
        if self.ind:
            ind = S.Zero
            # 获取自由符号
            sym = self.free_symbols
            # 对于 self.ind 中的每个项 t
            for t in Add.make_args(self.ind):
                # 获取 t 中 x 的幂次
                pow_x = self._get_pow_x(t)
                # 如果 pow_x 中包含自由符号
                if pow_x.has(*sym):
                    # 将 pow_x 表达式拆解为系数和其余部分，取第一个系数作为 pow_x
                    pow_x = pow_x.as_coeff_add(*sym)[0]
                # 根据条件累加 ind 的值
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            # 将 ind 加到 term 上
            term += ind

        # 返回计算得到的 term，并将结果按照 self.x 进行整理和收集
        return term.collect(self.x)

    # 如果旧表达式中包含 x，则进行替换操作
    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    # 计算作为主导项的值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 对于自身的每一个项 t
        for t in self:
            # 如果 t 不是零，则返回 t 作为主导项
            if t is not S.Zero:
                return t

    # 对于函数的 x 变量进行导数计算
    def _eval_derivative(self, x):
        # 计算函数的导数 f
        f = self.function.diff(x)
        # 计算 self.ind 的导数 ind
        ind = self.ind.diff(x)

        # 获取 xk 中 x 的幂次
        pow_xk = self._get_pow_x(self.xk.formula)
        # 获取 ak 和 k
        ak = self.ak
        k = ak.variables[0]
        # 如果 ak.formula 中包含 x
        if ak.formula.has(x):
            # 对于 ak.formula 的每一个项 (e, c)
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                # 对于 e 中的每一个项 t
                for t in Add.make_args(e):
                    # 获取 t 中 x 的幂次
                    pow_x = self._get_pow_x(t)
                    # 计算 temp 的值为 t * (pow_xk + pow_x)
                    temp += t * (pow_xk + pow_x)
                # 将 (temp, c) 添加到 form 中
                form.append((temp, c))
            # 使用 Piecewise 构造 form
            form = Piecewise(*form)
            # 更新 ak 为 form.subs(k, k + 1)，并使用序列构造
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            # 更新 ak 为 (ak.formula * pow_xk).subs(k, k + 1)，并使用序列构造
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1),
                          (k, ak.start - 1, ak.stop))

        # 返回使用函数构造器构造的导数结果
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))
    def integrate(self, x=None, **kwargs):
        """
        Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin, integrate
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> integrate(f, (x, 0, 1))
        1 - cos(1)
        """
        # 导入积分函数
        from sympy.integrals import integrate

        # 如果未指定积分变量 x，则使用对象自身的 x
        if x is None:
            x = self.x
        # 如果 x 是可迭代的，则对函数使用多重积分
        elif iterable(x):
            return integrate(self.function, x)

        # 对函数 f 和指标 ind 分别进行积分
        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        # 加上积分常数 (f - ind).limit(x, 0)
        ind += (f - ind).limit(x, 0)

        # 获取序列 ak 的公式和变量
        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        # 如果 ak 公式中包含 x，则按公式进行处理
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        # 否则按简单的公式进行处理
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1),
                          (k, ak.start + 1, ak.stop))

        # 返回积分后的 Formal Power Series 对象
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def product(self, other, x=None, n=6):
        """
        Multiplies two Formal Power Series, using discrete convolution and
        return the truncated terms upto specified order.

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))

        >>> f1.product(f2, x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)

        See Also
        ========

        sympy.discrete.convolutions
        sympy.series.formal.FormalPowerSeriesProduct

        """
        # 如果 n 为 None，则返回迭代器
        if n is None:
            return iter(self)

        # 确保 other 是 FormalPowerSeries 类型
        other = sympify(other)
        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        # 确保两个序列具有相同的方向、起始点和符号
        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")
        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        # 返回两个序列的乘积对象
        return FormalPowerSeriesProduct(self, other)
    def coeff_bell(self, n):
        r"""
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
                \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        """

        # 计算内部的 Bell 多项式系数列表，使用 bell 函数进行计算
        inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n-j+1])) for j in range(1, n+1)]

        # 创建一个虚拟符号 k
        k = Dummy('k')
        # 返回一个符号序列，包含计算出的 Bell 多项式系数
        return sequence(tuple(inner_coeffs), (k, 1, oo))
    def compose(self, other, x=None, n=6):
        r"""
        Returns the truncated terms of the formal power series of the composed function,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series `fp`
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))

        >>> f1.compose(f2, x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

        >>> f1.compose(f2, x).truncate(8)
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesCompose

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        # 如果未指定 n，则返回迭代器形式的自身
        if n is None:
            return iter(self)

        # 确保 other 是一个符号表达式
        other = sympify(other)

        # 确保 other 是 FormalPowerSeries 类的实例
        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        # 检查两个序列的计算方向是否相同
        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        # 检查两个序列的展开点是否相同
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")
        # 检查两个序列的符号变量是否相同
        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        # 检查内部函数的常数项系数是否为零，应为形式幂级数的内部函数不应有常数项
        if other._eval_term(0).as_coeff_mul(other.x)[0] is not S.Zero:
            raise ValueError("The formal power series of the inner function should not have any "
                "constant coefficient term.")

        # 返回组合后的形式幂级数对象
        return FormalPowerSeriesCompose(self, other)
    def inverse(self, x=None, n=6):
        r"""
        Returns the truncated terms of the inverse of the formal power series,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series ``fp``
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))

        >>> f1.inverse(x).truncate()
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)

        >>> f2.inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesInverse

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        # 如果 n 参数为 None，则返回迭代器
        if n is None:
            return iter(self)

        # 检查零次系数是否为零，若为零则抛出 ValueError
        if self._eval_term(0).is_zero:
            raise ValueError("Constant coefficient should exist for an inverse of a formal"
                " power series to exist.")

        # 返回 FormalPowerSeriesInverse 对象
        return FormalPowerSeriesInverse(self)

    # 重载加法运算符 +
    def __add__(self, other):
        # 将 other 转换为 Sympy 符号表达式
        other = sympify(other)

        # 如果 other 是 FormalPowerSeries 类型
        if isinstance(other, FormalPowerSeries):
            # 检查两个序列的方向是否一致
            if self.dir != other.dir:
                raise ValueError("Both series should be calculated from the"
                                 " same direction.")
            # 检查两个序列的展开点是否一致
            elif self.x0 != other.x0:
                raise ValueError("Both series should be calculated about the"
                                 " same point.")

            # 获取两个序列的自变量
            x, y = self.x, other.x
            # 计算两个序列的函数表达式之和
            f = self.function + other.function.subs(y, x)

            # 如果自变量不在函数表达式中，则直接返回 f
            if self.x not in f.free_symbols:
                return f

            # 计算两个序列的系数 ak 之和
            ak = self.ak + other.ak
            # 根据 ak 的起始点确定序列的范围
            if self.ak.start > other.ak.start:
                seq = other.ak
                s, e = other.ak.start, self.ak.start
            else:
                seq = self.ak
                s, e = self.ak.start, other.ak.start
            # 计算序列的乘积并保存
            save = Add(*[z[0]*z[1] for z in zip(seq[0:(e - s)], self.xk[s:e])])
            # 计算序列的指标 ind
            ind = self.ind + other.ind + save

            # 返回更新后的 FormalPowerSeries 对象
            return self.func(f, x, self.x0, self.dir, (ak, self.xk, ind))

        # 如果 other 不包含自变量 x，则将其加到函数表达式中
        elif not other.has(self.x):
            f = self.function + other
            ind = self.ind + other

            # 返回更新后的 FormalPowerSeries 对象
            return self.func(f, self.x, self.x0, self.dir,
                             (self.ak, self.xk, ind))

        # 其他情况下返回两个对象的和
        return Add(self, other)
    # 实现右加法运算符重载，将其委托给 __add__ 方法处理
    def __radd__(self, other):
        return self.__add__(other)

    # 实现取负运算符重载
    def __neg__(self):
        return self.func(-self.function, self.x, self.x0, self.dir,
                         (-self.ak, self.xk, -self.ind))

    # 实现减法运算符重载，将其委托给 __add__ 方法处理
    def __sub__(self, other):
        return self.__add__(-other)

    # 实现右减法运算符重载
    def __rsub__(self, other):
        return (-self).__add__(other)

    # 实现乘法运算符重载
    def __mul__(self, other):
        # 将 other 转换为 sympy 表达式
        other = sympify(other)

        # 如果 other 中包含 self.x，则返回 Mul(self, other)
        if other.has(self.x):
            return Mul(self, other)

        # 计算新的函数表达式、系数 ak 和指数 ind
        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other

        # 返回经过乘法操作后的新对象
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    # 实现右乘法运算符重载
    def __rmul__(self, other):
        return self.__mul__(other)
class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""

    def __init__(self, *args):
        pass
    # 初始化方法，接受任意数量的参数，但不执行任何操作

    @property
    def ffps(self):
        return self.args[0]
    # 返回第一个参数作为属性 ffps

    @property
    def gfps(self):
        return self.args[1]
    # 返回第二个参数作为属性 gfps

    @property
    def f(self):
        return self.ffps.function
    # 返回 ffps 的 function 属性作为属性 f

    @property
    def g(self):
        return self.gfps.function
    # 返回 gfps 的 function 属性作为属性 g

    @property
    def infinite(self):
        raise NotImplementedError("No infinite version for an object of"
                                  " FiniteFormalPowerSeries class.")
    # 抛出未实现错误，表明 FiniteFormalPowerSeries 类没有无限版本

    def _eval_terms(self, n):
        raise NotImplementedError("(%s)._eval_terms()" % self)
    # 抛出未实现错误，表明 _eval_terms 方法尚未实现

    def _eval_term(self, pt):
        raise NotImplementedError("By the current logic, one can get terms"
                                  "upto a certain order, instead of getting term by term.")
    # 抛出未实现错误，表明 _eval_term 方法尚未实现

    def polynomial(self, n):
        return self._eval_terms(n)
    # 调用 _eval_terms 方法返回多项式的前 n 项

    def truncate(self, n=6):
        ffps = self.ffps
        pt_xk = ffps.xk.coeff(n)
        x, x0 = ffps.x, ffps.x0

        return self.polynomial(n) + Order(pt_xk, (x, x0))
    # 截断级数至第 n 阶，并返回结果与 Order 对象的和

    def _eval_derivative(self, x):
        raise NotImplementedError
    # 抛出未实现错误，表明 _eval_derivative 方法尚未实现

    def integrate(self, x):
        raise NotImplementedError
    # 抛出未实现错误，表明 integrate 方法尚未实现


class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        ffps, gfps = self.ffps, self.gfps
        # 获取 ffps 和 gfps 作为初始化参数

        k = ffps.ak.variables[0]
        self.coeff1 = sequence(ffps.ak.formula, (k, 0, oo))
        # 计算 ffps 的系数序列并赋值给 coeff1

        k = gfps.ak.variables[0]
        self.coeff2 = sequence(gfps.ak.formula, (k, 0, oo))
        # 计算 gfps 的系数序列并赋值给 coeff2

    @property
    def function(self):
        """Function of the product of two formal power series."""
        return self.f * self.g
    # 返回两个形式幂级数的乘积作为 function 属性
    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the product formal power series.
        Term by term logic is implemented here.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))
        >>> fprod = f1.product(f2, x)

        >>> fprod._eval_terms(4)
        x**3/3 + x**2 + x

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.product

        """

        # 获取两个系数列表 coeff1 和 coeff2
        coeff1, coeff2 = self.coeff1, self.coeff2

        # 计算前 n 项的卷积结果
        aks = convolution(coeff1[:n], coeff2[:n])

        # 初始化一个空列表用于存放每一项的结果
        terms = []
        # 遍历范围从 0 到 n-1
        for i in range(0, n):
            # 计算每一项的结果并添加到 terms 列表中
            terms.append(aks[i] * self.ffps.xk.coeff(i))

        # 将所有项相加得到最终的结果
        return Add(*terms)
class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """
    Represents the composed formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesCompose` object. The first argument contains the outer
    function and the inner function involved in the composition. Also, the
    coefficient sequence contains the generic sequence which is to be multiplied
    by a custom ``bell_seq`` finite sequence. The finite terms will then be added up to
    get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    @property
    def function(self):
        """Function for the composed formal power series."""
        # 返回组合形式幂级数的函数
        f, g, x = self.f, self.g, self.ffps.x
        return f.subs(x, g)

    def _eval_terms(self, n):
        """
        Returns the first `n` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the :obj:`FormalPowerSeriesCompose` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))
        >>> fcomp = f1.compose(f2, x)

        >>> fcomp._eval_terms(6)
        -x**5/15 - x**4/8 + x**2/2 + x + 1

        >>> fcomp._eval_terms(8)
        x**7/90 - x**6/240 - x**5/15 - x**4/8 + x**2/2 + x + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.compose
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """

        ffps, gfps = self.ffps, self.gfps
        terms = [ffps.zero_coeff()]

        for i in range(1, n):
            bell_seq = gfps.coeff_bell(i)
            seq = (ffps.bell_coeff_seq * bell_seq)
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))

        return Add(*terms)


class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """
    Represents the Inverse of a formal power series.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There is a single difference between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesInverse` object. The coefficient sequence contains the
    generic sequence which is to be multiplied by a custom ``bell_seq`` finite sequence.
    The finite terms will then be added up to get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries

    """
    # 导入 sympy 库中的 formal 模块下的 FiniteFormalPowerSeries 类
    sympy.series.formal.FiniteFormalPowerSeries
    
    """
    FormalPowerSeriesInverse 类的构造函数，接受任意数量的参数
    """
    def __init__(self, *args):
        # 获取 self.ffps 的引用
        ffps = self.ffps
        # 从 ffps.xk 中获取变量列表的第一个变量
        k = ffps.xk.variables[0]
    
        # 计算 ffps 的零次系数，得到 inv
        inv = ffps.zero_coeff()
        # 生成一个序列，其中每一项为 inv 的负 (k + 1) 次幂
        inv_seq = sequence(inv ** (-(k + 1)), (k, 1, oo))
        # 计算并存储辅助序列，由 ffps 的符号序列、阶乘序列和 inv_seq 组成
        self.aux_seq = ffps.sign_seq * ffps.fact_seq * inv_seq
    
    @property
    def function(self):
        """返回正式幂级数的倒数的函数。"""
        # 获取 self.f 的引用
        f = self.f
        # 返回 f 的倒数
        return 1 / f
    
    @property
    def g(self):
        # 抛出 ValueError 异常，说明只考虑执行正式幂级数的倒数时的一个函数
        raise ValueError("Only one function is considered while performing"
                         "inverse of a formal power series.")
    
    @property
    def gfps(self):
        # 抛出 ValueError 异常，说明只考虑执行正式幂级数的倒数时的一个函数
        raise ValueError("Only one function is considered while performing"
                         "inverse of a formal power series.")
    
    def _eval_terms(self, n):
        """
        返回组合形式幂级数的前 ``n`` 项。
        逐项逻辑在此实现。
    
        Explanation
        ===========
    
        `FormalPowerSeriesInverse` 对象的系数序列是通用序列。
        它与 ``bell_seq`` 相乘得到一个序列，其项相加得到多项式的最终项。
    
        Examples
        ========
    
        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))
        >>> finv1, finv2 = f1.inverse(), f2.inverse()
    
        >>> finv1._eval_terms(6)
        -x**5/120 + x**4/24 - x**3/6 + x**2/2 - x + 1
    
        >>> finv2._eval_terms(8)
        61*x**6/720 + 5*x**4/24 + x**2/2 + 1
    
        See Also
        ========
    
        sympy.series.formal.FormalPowerSeries.inverse
        sympy.series.formal.FormalPowerSeries.coeff_bell
        """
        # 获取 self.ffps 的引用
        ffps = self.ffps
        # 初始化 terms 列表，包含 ffps 的零次系数
        terms = [ffps.zero_coeff()]
    
        # 遍历 1 到 n-1 的范围
        for i in range(1, n):
            # 获取 ffps 的 i 阶贝尔多项式
            bell_seq = ffps.coeff_bell(i)
            # 计算 seq，为 self.aux_seq 与 bell_seq 的乘积
            seq = (self.aux_seq * bell_seq)
            # 将 seq 的前 i 项相加，然后除以 ffps 的阶乘序列的第 i-1 项，再乘以 ffps 的 xk 系数 i
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))
    
        # 返回 terms 的总和
        return Add(*terms)
def fps(f, x=None, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    """
    Generates Formal Power Series of ``f``.

    Explanation
    ===========

    Returns the formal series expansion of ``f`` around ``x = x0``
    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.

    Formal Power Series is represented using an explicit formula
    computed using different algorithms.

    See :func:`compute_fps` for the more details regarding the computation
    of formula.

    Parameters
    ==========

    x : Symbol, optional
        If x is None and ``f`` is univariate, the univariate symbols will be
        supplied, otherwise an error will be raised.
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Examples
    ========

    >>> from sympy import fps, ln, atan, sin
    >>> from sympy.abc import x, n

    Rational Functions

    >>> fps(ln(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)

    Symbolic Functions

    >>> fps(x**n*sin(x**2), x).truncate(8)
    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.compute_fps
    """
    # 将输入的函数转换为符号表达式
    f = sympify(f)

    # 如果未提供 x，则尝试从函数的自由符号中推断
    if x is None:
        free = f.free_symbols
        # 如果函数只有一个自由符号，将其设为 x
        if len(free) == 1:
            x = free.pop()
        # 如果没有自由符号，直接返回原函数
        elif not free:
            return f
        # 对于多变量函数，暂时不支持
        else:
            raise NotImplementedError("multivariate formal power series")

    # 调用 compute_fps 函数计算形式幂级数的结果
    result = compute_fps(f, x, x0, dir, hyper, order, rational, full)

    # 如果计算结果为空，返回原函数
    if result is None:
        return f

    # 返回形式幂级数对象 FormalPowerSeries，将计算结果作为其参数
    return FormalPowerSeries(f, x, x0, dir, result)
```