# `D:\src\scipysrc\sympy\sympy\concrete\guess.py`

```
"""Various algorithms for helping identifying numbers and sequences."""


# 从 sympy.concrete.products 模块导入 Product 和 product 函数
from sympy.concrete.products import (Product, product)
# 从 sympy.core 模块导入 Function, S
from sympy.core import Function, S
# 从 sympy.core.add 模块导入 Add
from sympy.core.add import Add
# 从 sympy.core.numbers 模块导入 Integer, Rational
from sympy.core.numbers import Integer, Rational
# 从 sympy.core.symbol 模块导入 Symbol, symbols
from sympy.core.symbol import Symbol, symbols
# 从 sympy.core.sympify 模块导入 sympify
from sympy.core.sympify import sympify
# 从 sympy.functions.elementary.exponential 模块导入 exp
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.integers 模块导入 floor
from sympy.functions.elementary.integers import floor
# 从 sympy.integrals.integrals 模块导入 integrate
from sympy.integrals.integrals import integrate
# 从 sympy.polys.polyfuncs 模块导入 rational_interpolate 作为 rinterp
from sympy.polys.polyfuncs import rational_interpolate as rinterp
# 从 sympy.polys.polytools 模块导入 lcm
from sympy.polys.polytools import lcm
# 从 sympy.simplify.radsimp 模块导入 denom
from sympy.simplify.radsimp import denom
# 从 sympy.utilities 模块导入 public
from sympy.utilities import public


@public
def find_simple_recurrence_vector(l):
    """
    This function is used internally by other functions from the
    sympy.concrete.guess module. While most users may want to rather use the
    function find_simple_recurrence when looking for recurrence relations
    among rational numbers, the current function may still be useful when
    some post-processing has to be done.

    Explanation
    ===========

    The function returns a vector of length n when a recurrence relation of
    order n is detected in the sequence of rational numbers v.

    If the returned vector has a length 1, then the returned value is always
    the list [0], which means that no relation has been found.

    While the functions is intended to be used with rational numbers, it should
    work for other kinds of real numbers except for some cases involving
    quadratic numbers; for that reason it should be used with some caution when
    the argument is not a list of rational numbers.

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence_vector
    >>> from sympy import fibonacci
    >>> find_simple_recurrence_vector([fibonacci(k) for k in range(12)])
    [1, -1, -1]

    See Also
    ========

    See the function sympy.concrete.guess.find_simple_recurrence which is more
    user-friendly.

    """
    # 初始化 q1 为 [0]
    q1 = [0]
    # 初始化 q2 为 [1]
    q2 = [1]
    # 初始化 b 和 z
    b, z = 0, len(l) >> 1
    # 开始循环，直到 q2 的长度大于 z
    while len(q2) <= z:
        # 当 l[b] 等于 0 时，递增 b 直到找到第一个非零值
        while l[b]==0:
            b += 1
            # 如果 b 等于 l 的长度，则计算结果向量并返回
            if b == len(l):
                # 初始化 c 为 1
                c = 1
                # 计算 c 为 q2 中元素的最小公倍数的分母
                for x in q2:
                    c = lcm(c, denom(x))
                # 如果 q2 的第一个元素乘以 c 小于 0，则取相反数
                if q2[0]*c < 0: c = -c
                # 更新 q2 中的每个元素
                for k in range(len(q2)):
                    q2[k] = int(q2[k]*c)
                # 返回结果向量 q2
                return q2
        # 计算 a 为 S.One / l[b]
        a = S.One/l[b]
        # 初始化 m 为 [a]
        m = [a]
        # 开始循环，计算 m 的值
        for k in range(b+1, len(l)):
            m.append(-sum(l[j+1]*m[b-j-1] for j in range(b, k))*a)
        # 更新 l 和 m 的值
        l, m = m, [0] * max(len(q2), b+len(q1))
        for k, q in enumerate(q2):
            m[k] = a*q
        for k, q in enumerate(q1):
            m[k+b] += q
        # 去除末尾的零元素
        while m[-1]==0: m.pop() # because trailing zeros can occur
        # 更新 q1, q2 和 b 的值
        q1, q2, b = q2, m, 1
    # 返回结果向量 [0]
    return [0]

@public
def find_simple_recurrence(v, A=Function('a'), N=Symbol('n')):
    """
    Detects and returns a recurrence relation from a sequence of several integer
    """
    # 使用给定向量 `v` 找到其简单递推关系的多项式表示
    p = find_simple_recurrence_vector(v)
    # 计算向量 `p` 的长度
    n = len(p)
    # 如果向量长度 `n` 小于等于 1，返回零
    if n <= 1: return S.Zero
    
    # 构造递推关系的表达式，其中 `A` 是函数，`N` 是符号，从 `n-1` 到 `0` 遍历多项式 `p` 的系数
    return Add(*[A(N+n-1-k)*p[k] for k in range(n)])
@public
def rationalize(x, maxcoeff=10000):
    """
    Helps identifying a rational number from a float (or mpmath.mpf) value by
    using a continued fraction. The algorithm stops as soon as a large partial
    quotient is detected (greater than 10000 by default).

    Examples
    ========

    >>> from sympy.concrete.guess import rationalize
    >>> from mpmath import cos, pi
    >>> rationalize(cos(pi/3))
    1/2

    >>> from mpmath import mpf
    >>> rationalize(mpf("0.333333333333333"))
    1/3

    While the function is rather intended to help 'identifying' rational
    values, it may be used in some cases for approximating real numbers.
    (Though other functions may be more relevant in that case.)

    >>> rationalize(pi, maxcoeff = 250)
    355/113

    See Also
    ========

    Several other methods can approximate a real number as a rational, like:

      * fractions.Fraction.from_decimal
      * fractions.Fraction.from_float
      * mpmath.identify
      * mpmath.pslq by using the following syntax: mpmath.pslq([x, 1])
      * mpmath.findpoly by using the following syntax: mpmath.findpoly(x, 1)
      * sympy.simplify.nsimplify (which is a more general function)

    The main difference between the current function and all these variants is
    that control focuses on magnitude of partial quotients here rather than on
    global precision of the approximation. If the real is "known to be" a
    rational number, the current function should be able to detect it correctly
    with the default settings even when denominator is great (unless its
    expansion contains unusually big partial quotients) which may occur
    when studying sequences of increasing numbers. If the user cares more
    on getting simple fractions, other methods may be more convenient.

    """
    p0, p1 = 0, 1
    q0, q1 = 1, 0
    a = floor(x)
    while a < maxcoeff or q1==0:
        # Compute the next terms in the continued fraction expansion
        p = a*p1 + p0
        q = a*q1 + q0
        p0, p1 = p1, p
        q0, q1 = q1, q
        if x==a: break
        x = 1/(x-a)
        a = floor(x)
    return sympify(p) / q


@public
def guess_generating_function_rational(v, X=Symbol('x')):
    """
    Tries to "guess" a rational generating function for a sequence of rational
    numbers v.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function_rational
    >>> from sympy import fibonacci
    >>> l = [fibonacci(k) for k in range(5,15)]
    >>> guess_generating_function_rational(l)
    (3*x + 5)/(-x**2 - x + 1)

    See Also
    ========

    sympy.series.approximants
    mpmath.pade

    """
    # Compute the denominator coefficients for the recurrence relation
    q = find_simple_recurrence_vector(v)
    n = len(q)
    if n <= 1: return None
    # Compute the numerator coefficients for the recurrence relation
    p = [sum(v[i-k]*q[k] for k in range(min(i+1, n)))
            for i in range(len(v)>>1)]
    # Return the rational generating function as a SymPy expression
    return (sum(p[k]*X**k for k in range(len(p)))
            / sum(q[k]*X**k for k in range(n)))
def guess_generating_function(v, X=Symbol('x'), types=['all'], maxsqrtn=2):
    """
    Tries to "guess" a generating function for a sequence of rational numbers v.
    Only a few patterns are implemented yet.

    Explanation
    ===========

    The function returns a dictionary where keys are the name of a given type of
    generating function. Six types are currently implemented:

         type  |  formal definition
        -------+----------------------------------------------------------------
        ogf    | f(x) = Sum(            a_k * x^k       ,  k: 0..infinity )
        egf    | f(x) = Sum(            a_k * x^k / k!  ,  k: 0..infinity )
        lgf    | f(x) = Sum( (-1)^(k+1) a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        hlgf   | f(x) = Sum(            a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        lgdogf | f(x) = derivate( log(Sum( a_k * x^k, k: 0..infinity )), x)
        lgdegf | f(x) = derivate( log(Sum( a_k * x^k / k!, k: 0..infinity )), x)

    In order to spare time, the user can select only some types of generating
    functions (default being ['all']). While forgetting to use a list in the
    case of a single type may seem to work most of the time as in: types='ogf'
    this (convenient) syntax may lead to unexpected extra results in some cases.

    Discarding a type when calling the function does not mean that the type will
    not be present in the returned dictionary; it only means that no extra
    computation will be performed for that type, but the function may still add
    it in the result when it can be easily converted from another type.

    Two generating functions (lgdogf and lgdegf) are not even computed if the
    initial term of the sequence is 0; it may be useful in that case to try
    again after having removed the leading zeros.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function as ggf
    >>> ggf([k+1 for k in range(12)], types=['ogf', 'lgf', 'hlgf'])
    {'hlgf': 1/(1 - x), 'lgf': 1/(x + 1), 'ogf': 1/(x**2 - 2*x + 1)}

    >>> from sympy import sympify
    >>> l = sympify("[3/2, 11/2, 0, -121/2, -363/2, 121]")
    >>> ggf(l)
    {'ogf': (x + 3/2)/(11*x**2 - 3*x + 1)}

    >>> from sympy import fibonacci
    >>> ggf([fibonacci(k) for k in range(5, 15)], types=['ogf'])
    {'ogf': (3*x + 5)/(-x**2 - x + 1)}

    >>> from sympy import factorial
    >>> ggf([factorial(k) for k in range(12)], types=['ogf', 'egf', 'lgf'])
    {'egf': 1/(1 - x)}

    >>> ggf([k+1 for k in range(12)], types=['egf'])
    {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}

    N-th root of a rational function can also be detected (below is an example
    coming from the sequence A108626 from https://oeis.org).
    The greatest n-th root to be tested is specified as maxsqrtn (default 2).
    """

    # Initialize an empty dictionary to store the results
    result = {}

    # Loop over each type of generating function specified
    for gftype in types:
        if gftype == 'ogf':
            # Compute ordinary generating function
            result['ogf'] = RationalGeneratingFunction(v, X)
        elif gftype == 'egf':
            # Compute exponential generating function
            result['egf'] = ExponentialGeneratingFunction(v, X)
        elif gftype == 'lgf':
            # Compute Laguerre generating function
            result['lgf'] = LaguerreGeneratingFunction(v, X)
        elif gftype == 'hlgf':
            # Compute modified Laguerre generating function
            result['hlgf'] = ModifiedLaguerreGeneratingFunction(v, X)
        elif gftype == 'lgdogf':
            # Compute logarithmic derivative of ordinary generating function
            if v[0] != 0:
                result['lgdogf'] = LogDerivativeOrdinaryGeneratingFunction(v, X)
        elif gftype == 'lgdegf':
            # Compute logarithmic derivative of exponential generating function
            if v[0] != 0:
                result['lgdegf'] = LogDerivativeExponentialGeneratingFunction(v, X)

    # Return the dictionary containing generating function results
    return result
    # 如果 types 包含 'all'，则将 types 重新定义为所有生成函数的类型
    if 'all' in types:
        types = ('ogf', 'egf', 'lgf', 'hlgf', 'lgdogf', 'lgdegf')

    result = {}  # 初始化结果字典，用于存储生成函数的结果

    # 普通生成函数 (ogf)
    if 'ogf' in types:
        # 对序列进行一些自卷积操作
        t = [1] + [0]*(len(v) - 1)
        for d in range(max(1, maxsqrtn)):
            # 计算序列的自卷积
            t = [sum(t[n-i]*v[i] for i in range(n+1)) for n in range(len(v))]
            # 猜测序列的有理生成函数
            g = guess_generating_function_rational(t, X=X)
            if g:
                # 将结果存入结果字典中，对生成函数进行 d+1 次根运算
                result['ogf'] = g**Rational(1, d+1)
                break

    # 指数生成函数 (egf)
    if 'egf' in types:
        # 对序列进行变换（除以阶乘）
        w, f = [], S.One
        for i, k in enumerate(v):
            f *= i if i else 1
            w.append(k/f)
        # 对序列进行一些自卷积操作
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            # 计算序列的自卷积
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            # 猜测序列的有理生成函数
            g = guess_generating_function_rational(t, X=X)
            if g:
                # 将结果存入结果字典中，对生成函数进行 d+1 次根运算
                result['egf'] = g**Rational(1, d+1)
                break

    # 对数生成函数 (lgf)
    if 'lgf' in types:
        # 对序列进行变换（乘以 (-1)^(n+1) / n）
        w, f = [], S.NegativeOne
        for i, k in enumerate(v):
            f = -f
            w.append(f*k/Integer(i+1))
        # 对序列进行一些自卷积操作
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            # 计算序列的自卷积
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            # 猜测序列的有理生成函数
            g = guess_generating_function_rational(t, X=X)
            if g:
                # 将结果存入结果字典中，对生成函数进行 d+1 次根运算
                result['lgf'] = g**Rational(1, d+1)
                break

    # 双对数生成函数 (lgdogf)
    if 'lgdogf' in types:
        # TODO: 缺少实现部分，暂时无法解释

    # 对数导数生成函数 (lgdegf)
    if 'lgdegf' in types:
        # TODO: 缺少实现部分，暂时无法解释
    # 检查序列的第一个元素是否不为零，且满足以下条件之一：
    # 1. types 中包含 'lgdogf'
    # 2. types 中包含 'ogf' 但不包含 'ogf' 在 result 中不存在时
    if v[0] != 0 and ('lgdogf' in types
                       or ('ogf' in types and 'ogf' not in result)):
        # 对序列进行转换，计算 f'(x)/f(x)
        # 因为 log(f(x)) = integrate( f'(x)/f(x) )
        
        # 将 v[0] 转换为 sympy 符号对象 a，初始化空列表 w
        a, w = sympify(v[0]), []
        # 遍历 v 的剩余元素
        for n in range(len(v)-1):
            # 计算序列 w 的每个元素
            w.append(
               (v[n+1]*(n+1) - sum(w[-i-1]*v[i+1] for i in range(n)))/a)
        
        # 执行序列的一些卷积操作
        t = [1] + [0]*(len(w) - 1)
        # 对于最大为 max(1, maxsqrtn) 的范围内的每个 d
        for d in range(max(1, maxsqrtn)):
            # 计算序列 t 的每个元素
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            # 猜测生成函数的有理解
            g = guess_generating_function_rational(t, X=X)
            # 如果猜测成功
            if g:
                # 将结果放入字典 result 中的 'lgdogf' 键
                result['lgdogf'] = g**Rational(1, d+1)
                # 如果 result 中不存在 'ogf' 键
                if 'ogf' not in result:
                    # 计算结果的指数函数
                    result['ogf'] = exp(integrate(result['lgdogf'], X))
                break

    # 对于指数生成函数的对数导数（lgdegf）
    if v[0] != 0 and ('lgdegf' in types
                       or ('egf' in types and 'egf' not in result)):
        # 转换序列 / 步骤 1 （除以阶乘）
        z, f = [], S.One
        # 遍历序列的索引 i 和元素 k
        for i, k in enumerate(v):
            # 计算阶乘 f
            f *= i if i else 1
            # 将 k/f 加入列表 z
            z.append(k/f)
        
        # 转换序列 / 步骤 2 通过计算 f'(x)/f(x)
        # 因为 log(f(x)) = integrate( f'(x)/f(x) )
        a, w = z[0], []
        # 遍历 z 的长度减一
        for n in range(len(z)-1):
            # 计算序列 w 的每个元素
            w.append(
               (z[n+1]*(n+1) - sum(w[-i-1]*z[i+1] for i in range(n)))/a)
        
        # 执行序列的一些卷积操作
        t = [1] + [0]*(len(w) - 1)
        # 对于最大为 max(1, maxsqrtn) 的范围内的每个 d
        for d in range(max(1, maxsqrtn)):
            # 计算序列 t 的每个元素
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            # 猜测生成函数的有理解
            g = guess_generating_function_rational(t, X=X)
            # 如果猜测成功
            if g:
                # 将结果放入字典 result 中的 'lgdegf' 键
                result['lgdegf'] = g**Rational(1, d+1)
                # 如果 result 中不存在 'egf' 键
                if 'egf' not in result:
                    # 计算结果的指数函数
                    result['egf'] = exp(integrate(result['lgdegf'], X))
                break

    # 返回结果字典
    return result
@public
def guess(l, all=False, evaluate=True, niter=2, variables=None):
    """
    This function is adapted from the Rate.m package for Mathematica
    written by Christian Krattenthaler.
    It tries to guess a formula from a given sequence of rational numbers.

    Explanation
    ===========

    In order to speed up the process, the 'all' variable is set to False by
    default, stopping the computation as some results are returned during an
    iteration; the variable can be set to True if more iterations are needed
    (other formulas may be found; however they may be equivalent to the first
    ones).

    Another option is the 'evaluate' variable (default is True); setting it
    to False will leave the involved products unevaluated.

    By default, the number of iterations is set to 2 but a greater value (up
    to len(l)-1) can be specified with the optional 'niter' variable.
    More and more convoluted results are found when the order of the
    iteration gets higher:

      * first iteration returns polynomial or rational functions;
      * second iteration returns products of rising factorials and their
        inverses;
      * third iteration returns products of products of rising factorials
        and their inverses;
      * etc.

    The returned formulas contain symbols i0, i1, i2, ... where the main
    variable is i0 (and auxiliary variables are i1, i2, ...). A list of
    other symbols can be provided in the 'variables' option; the length of
    the list should be the value of 'niter' (more is acceptable but only
    the first symbols will be used); in this case, the main variable will be
    the first symbol in the list.

    Examples
    ========

    >>> from sympy.concrete.guess import guess
    >>> guess([1,2,6,24,120], evaluate=False)
    [Product(i1 + 1, (i1, 1, i0 - 1))]

    >>> from sympy import symbols
    >>> r = guess([1,2,7,42,429,7436,218348,10850216], niter=4)
    >>> i0 = symbols("i0")
    >>> [r[0].subs(i0,n).doit() for n in range(1,10)]
    [1, 2, 7, 42, 429, 7436, 218348, 10850216, 911835460]
    """
    # 如果序列中有任何元素为0，则直接返回空列表
    if any(a == 0 for a in l[:-1]):
        return []
    # 计算序列的长度
    N = len(l)
    # 确定迭代次数不超过序列长度-1和指定的niter之间的较小值
    niter = min(N - 1, niter)
    # 根据evaluate参数选择使用普通乘积或符号表达式的乘积
    myprod = product if evaluate else Product
    # 初始化空列表g和res
    g = []
    res = []
    # 如果未指定变量symbols，则使用'i0', 'i1', ...的符号序列
    if variables is None:
        symb = symbols('i:' + str(niter))
    else:
        symb = variables
    # 对每个符号进行迭代
    for k, s in enumerate(symb):
        g.append(l)  # 将序列l添加到g列表中
        n, r = len(l), []
        # 从n-2到0逆向迭代序列
        for i in range(n - 2 - 1, -1, -1):
            # 使用rinterp函数计算插值ri
            ri = rinterp(enumerate(g[k][:-1], start=1), i, X=s)
            # 检查ri是否符合条件，将符合条件的ri添加到r列表中
            if (denom(ri).subs({s: n}) != 0
                    and (ri.subs({s: n}) - g[k][-1] == 0)
                    and ri not in r):
                r.append(ri)
        # 如果r列表非空
        if r:
            # 逆向迭代符号列表
            for i in range(k - 1, -1, -1):
                # 计算结果表达式r的乘积
                r = [g[i][0]
                     * myprod(v, (symb[i + 1], 1, symb[i] - 1)) for v in r]
            # 如果all参数为False，则返回结果表达式r
            if not all:
                return r
            # 否则将结果表达式r添加到res列表中
            res += r
        # 更新序列l，使用Rational函数生成新的有理数序列
        l = [Rational(l[i + 1], l[i]) for i in range(N - k - 1)]
    # 返回结果表达式列表res
    return res
```