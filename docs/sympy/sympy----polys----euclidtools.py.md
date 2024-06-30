# `D:\src\scipysrc\sympy\sympy\polys\euclidtools.py`

```
"""Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """

# 从 sympy.polys.densearith 导入一系列函数
from sympy.polys.densearith import (
    dup_sub_mul,    # 多项式的减法乘法
    dup_neg, dmp_neg,   # 多项式的负数运算
    dmp_add,    # 多项式的加法
    dmp_sub,    # 多项式的减法
    dup_mul, dmp_mul,   # 多项式的乘法
    dmp_pow,    # 多项式的幂运算
    dup_div, dmp_div,   # 多项式的除法
    dup_rem,    # 多项式的求余
    dup_quo, dmp_quo,   # 多项式的商
    dup_prem, dmp_prem, # 多项式的部分余数
    dup_mul_ground, dmp_mul_ground, # 多项式与常数的乘法
    dmp_mul_term,   # 多项式与单项式的乘法
    dup_quo_ground, dmp_quo_ground, # 多项式除以常数的商
    dup_max_norm, dmp_max_norm    # 多项式的最大范数
)

# 从 sympy.polys.densebasic 导入一系列函数
from sympy.polys.densebasic import (
    dup_strip,  # 移除多项式的高次零项
    dmp_raise,  # 提升多项式的阶数
    dmp_zero, dmp_one, dmp_ground,   # 创建各种类型的多项式
    dmp_one_p, dmp_zero_p,   # 检查多项式是否为单位多项式或零多项式
    dmp_zeros,  # 创建指定长度的零多项式列表
    dup_degree, dmp_degree, dmp_degree_in,   # 计算多项式的次数
    dup_LC, dmp_LC, dmp_ground_LC,   # 多项式的主系数
    dmp_multi_deflate, dmp_inflate, # 多项式的通解系数缩放与展开
    dup_convert, dmp_convert,   # 多项式的系数转换
    dmp_apply_pairs    # 应用函数对多项式列表进行操作
)

# 从 sympy.polys.densetools 导入一系列函数
from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,   # 清除多项式的分母
    dup_diff, dmp_diff,   # 多项式的微分运算
    dup_eval, dmp_eval, dmp_eval_in,   # 多项式的求值运算
    dup_trunc, dmp_ground_trunc,   # 多项式的截断操作
    dup_monic, dmp_ground_monic,   # 多项式的首一化
    dup_primitive, dmp_ground_primitive,   # 多项式的原始部分分解
    dup_extract, dmp_ground_extract   # 多项式的系数提取
)

# 从 sympy.polys.galoistools 导入一系列函数
from sympy.polys.galoistools import (
    gf_int, gf_crt   # 有限域的整数与中国剩余定理
)

# 导入查询函数 query 和异常类
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,    # 多元多项式错误
    HeuristicGCDFailed, HeuristicGCDFailed,   # 启发式 GCD 失败
    HomomorphismFailed,  # 同态映射失败
    NotInvertible,   # 非可逆
    DomainError   # 域错误
)

def dup_half_gcdex(f, g, K):
    """
    Half extended Euclidean algorithm in `F[x]`.

    Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_half_gcdex(f, g)
    (-1/5*x + 3/5, x + 1)

    """
    if not K.is_Field:
        raise DomainError("Cannot compute half extended GCD over %s" % K)

    a, b = [K.one], []    # 初始化多项式系数列表

    while g:
        q, r = dup_div(f, g, K)   # 计算 f 除以 g 的商与余数
        f, g = g, r    # 更新 f 和 g
        a, b = b, dup_sub_mul(a, q, b, K)   # 更新系数列表

    a = dup_quo_ground(a, dup_LC(f, K), K)   # 对系数列表进行额外的运算
    f = dup_monic(f, K)   # 首一化 f

    return a, f


def dmp_half_gcdex(f, g, u, K):
    """
    Half extended Euclidean algorithm in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_half_gcdex(f, g, K)   # 如果 u 为空，则调用一元多项式的半扩展欧几里得算法
    else:
        raise MultivariatePolynomialError(f, g)   # 否则抛出多元多项式错误异常


def dup_gcdex(f, g, K):
    """
    Extended Euclidean algorithm in `F[x]`.

    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_gcdex(f, g)
    (-1/5*x + 3/5, 1/5*x**2 - 6/5*x + 2, x + 1)

    """
    s, h = dup_half_gcdex(f, g, K)   # 调用一元多项式的半扩展欧几里得算法得到 s 和 h

    F = dup_sub_mul(h, s, f, K)   # 计算 F = h - s*f
    t = dup_quo(F, g, K)   # 计算 t = F / g

    return s, t, h   # 返回结果
    # 创建一个多项式环 R，并定义变量 x 和 y，使用整数环 ZZ
    >>> R, x, y = ring("x,y", ZZ)
    
    """
    如果 u 为空（假设 u 是一个空值或者假值），则调用 dup_gcdex 函数来计算多项式 f 和 g 的扩展最大公因子。
    扩展最大公因子通常是指在多项式环中，计算 f 和 g 的最大公因子以及关于 f 和 g 的 Bézout 系数的过程。
    
    如果 u 不为空（假设 u 是一个非空值），则抛出 MultivariatePolynomialError 异常。
    MultivariatePolynomialError 异常通常用于指示在多项式计算中发生了某种错误或不支持的操作。
    
    """
    if not u:
        return dup_gcdex(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)
# 在环 R 中计算多项式 f 对于多项式 g 的乘法逆元
def dup_invert(f, g, K):
    # 使用 dup_half_gcdex 函数计算 f 对 g 的乘法逆元，返回 s 作为结果
    s, h = dup_half_gcdex(f, g, K)

    # 如果 h 是 [K.one]，即表示 h 是单位元，则返回 s 对 g 取模的余数
    if h == [K.one]:
        return dup_rem(s, g, K)
    else:
        # 否则，抛出 NotInvertible 异常，表示 g 是零除子，无法求逆
        raise NotInvertible("zero divisor")


# 在环 R 中计算多变量多项式 f 对于多变量多项式 g 的乘法逆元
def dmp_invert(f, g, u, K):
    # 如果 u 是空集，即单变量多项式情况，则调用 dup_invert 函数计算结果
    if not u:
        return dup_invert(f, g, K)
    else:
        # 否则，抛出 MultivariatePolynomialError 异常，多变量多项式不支持求逆操作
        raise MultivariatePolynomialError(f, g)


# 在域 K[x] 中计算多项式 f 和 g 的欧几里德多项式余序列（PRS）
def dup_euclidean_prs(f, g, K):
    # 初始化余序列列表，将 f 和 g 加入其中
    prs = [f, g]
    # 计算 f 对 g 取模的余数，并赋值给 h
    h = dup_rem(f, g, K)

    # 循环计算直到 h 为零
    while h:
        # 将 h 加入余序列列表
        prs.append(h)
        # 更新 f 和 g 的值，继续计算下一个余数 h
        f, g = g, h
        h = dup_rem(f, g, K)

    # 返回完整的多项式余序列列表
    return prs


# 在域 K[X] 中计算多变量多项式 f 和 g 的欧几里德多项式余序列（PRS）
def dmp_euclidean_prs(f, g, u, K):
    # 如果 u 是空集，即单变量多项式情况，则调用 dup_euclidean_prs 函数计算结果
    if not u:
        return dup_euclidean_prs(f, g, K)
    else:
        # 否则，抛出 MultivariatePolynomialError 异常，多变量多项式不支持欧几里德算法
        raise MultivariatePolynomialError(f, g)


# 在域 K[x] 中计算多项式 f 和 g 的原始多项式余序列（PRS）
def dup_primitive_prs(f, g, K):
    # 初始化原始多项式余序列列表，将 f 和 g 加入其中
    prs = [f, g]
    # 使用 dup_prem 函数计算 f 对 g 的预处理结果，然后取其原始部分，并赋值给 h
    _, h = dup_primitive(dup_prem(f, g, K), K)

    # 循环计算直到 h 为零
    while h:
        # 将 h 加入原始多项式余序列列表
        prs.append(h)
        # 更新 f 和 g 的值，继续计算下一个余数 h
        _, h = dup_primitive(dup_prem(f, g, K), K)

    # 返回完整的原始多项式余序列列表
    return prs


# 在域 K[X] 中计算多变量多项式 f 和 g 的原始多项式余序列（PRS）
def dmp_primitive_prs(f, g, u, K):
    # 如果 u 是空集，即单变量多项式情况，则调用 dup_primitive_prs 函数计算结果
    if not u:
        return dup_primitive_prs(f, g, K)
    else:
        # 否则，抛出 MultivariatePolynomialError 异常，多变量多项式不支持原始多项式算法
        raise MultivariatePolynomialError(f, g)
    # 创建一个多项式环，变量为 'x' 和 'y'，系数环为整数环 ZZ
    >>> R, x, y = ring("x,y", ZZ)
    
    """
    如果输入的参数 u 为假值（如 None 或者 False），则调用 dup_primitive_prs 函数处理多项式 f 和 g 在环 K 上的原始部分剖析
    否则，抛出多变量多项式错误，指定的多项式为 f 和 g
    """
    if not u:
        return dup_primitive_prs(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)
# 定义函数 `dup_inner_subresultants`，计算多项式在 `K[x]` 上的子结果多项式余序列 (PRS)
# 和非零标量子结果。根据 [1] 定理 3，这些是常数 '-c' (负号优化计算符号)。
def dup_inner_subresultants(f, g, K):
    # 计算多项式 `f` 和 `g` 的次数
    n = dup_degree(f)
    m = dup_degree(g)

    # 如果 `f` 的次数小于 `g` 的次数，交换 `f` 和 `g`
    if n < m:
        f, g = g, f
        n, m = m, n

    # 如果 `f` 是零多项式，返回空列表
    if not f:
        return [], []

    # 如果 `g` 是零多项式，返回 `[f]` 和 `[K.one]`
    if not g:
        return [f], [K.one]

    # 初始化结果列表 `R`，包含 `f` 和 `g`
    R = [f, g]

    # 计算 `d`，即 `n - m`
    d = n - m

    # 计算 `-1^(d + 1)`，作为标量 `b`
    b = (-K.one)**(d + 1)

    # 计算 `h = prem(f, g)`，并乘以标量 `b`
    h = dup_prem(f, g, K)
    h = dup_mul_ground(h, b, K)

    # 计算 `lc = LC(g)`，并计算 `c = lc^d`
    lc = dup_LC(g, K)
    c = lc**d

    # 初始化标量子结果列表 `S`，第一个子行列式为 1
    S = [K.one, c]
    c = -c

    # 进入循环，直到 `h` 为零多项式
    while h:
        # 计算 `k = deg(h)`，并将 `h` 添加到结果列表 `R`
        k = dup_degree(h)
        R.append(h)

        # 更新 `f, g, m, d`
        f, g, m, d = g, h, k, m - k

        # 计算 `-lc * c^d` 作为标量 `b`
        b = -lc * c**d

        # 计算 `h = prem(f, g)`，并将其除以标量 `b`
        h = dup_prem(f, g, K)
        h = dup_quo_ground(h, b, K)

        # 更新 `lc = LC(g)`
        lc = dup_LC(g, K)

        # 处理特殊情况，当 `d > 1` 时，计算 `c^(d - 1)`
        if d > 1:        # 异常情况
            q = c**(d - 1)
            c = K.quo((-lc)**d, q)
        else:
            c = -lc

        # 将 `-c` 添加到标量子结果列表 `S`
        S.append(-c)

    # 返回结果列表 `R` 和标量子结果列表 `S`
    return R, S
    # 如果输入参数 u 是空（即假值），则调用 dup_inner_subresultants 函数，并返回其结果
    if not u:
        return dup_inner_subresultants(f, g, K)
    
    # 计算多项式 f 和 g 在变量 u 上的次数
    n = dmp_degree(f, u)
    m = dmp_degree(g, u)
    
    # 如果 f 的次数小于 g 的次数，则交换 f 和 g，确保 n >= m
    if n < m:
        f, g = g, f
        n, m = m, n
    
    # 如果 f 是零多项式，则返回空列表
    if dmp_zero_p(f, u):
        return [], []
    
    # 设置变量 v 为 u - 1
    v = u - 1
    
    # 如果 g 是零多项式，则返回包含 f 和 1 的列表，1 是在变量 v 上的 K.one
    if dmp_zero_p(g, u):
        return [f], [dmp_ground(K.one, v)]
    
    # 初始化 R 为包含 f 和 g 的列表
    R = [f, g]
    # 计算 d = n - m，即 f 和 g 的次数差
    d = n - m
    
    # 计算 b = (-1)^d * v^(d + 1)，其中 v 是 u - 1
    b = dmp_pow(dmp_ground(-K.one, v), d + 1, v, K)
    
    # 计算 h = dmp_prem(f, g, u, K)，即 f 除以 g 的余式
    h = dmp_prem(f, g, u, K)
    # 计算 h = h * b，即 h 乘以 b
    h = dmp_mul_term(h, b, 0, u, K)
    
    # 计算 lc = dmp_LC(g, K)，即 g 的主导系数
    lc = dmp_LC(g, K)
    # 计算 c = lc^d，即 lc 的 d 次幂
    c = dmp_pow(lc, d, v, K)
    
    # 初始化 S 为包含 1 和 c 的列表
    S = [dmp_ground(K.one, v), c]
    # 计算 c = -c
    c = dmp_neg(c, v, K)
    
    # 循环直到 h 是零多项式
    while not dmp_zero_p(h, u):
        # 计算 k = dmp_degree(h, u)，即 h 的次数
        k = dmp_degree(h, u)
        # 将 h 添加到 R 列表中
        R.append(h)
    
        # 更新 f, g, m, d
        f, g, m, d = g, h, k, m - k
    
        # 计算 b = -lc * c^d，其中 b 是在变量 v 上的多项式
        b = dmp_mul(dmp_neg(lc, v, K),
                    dmp_pow(c, d, v, K), v, K)
    
        # 计算 h = dmp_prem(f, g, u, K)，即 f 除以 g 的余式
        h = dmp_prem(f, g, u, K)
        # 计算 h = [ ch / b for ch in h ]，即 h 中每个项除以 b
        h = [ dmp_quo(ch, b, v, K) for ch in h ]
    
        # 更新 lc = dmp_LC(g, K)，即 g 的主导系数
        lc = dmp_LC(g, K)
    
        # 如果 d > 1，则计算 p = (-lc)^d 和 q = c^(d-1)，然后计算 c = p / q
        if d > 1:
            p = dmp_pow(dmp_neg(lc, v, K), d, v, K)
            q = dmp_pow(c, d - 1, v, K)
            c = dmp_quo(p, q, v, K)
        else:
            # 否则，计算 c = -lc
            c = dmp_neg(lc, v, K)
    
        # 将 -c 添加到 S 中
        S.append(dmp_neg(c, v, K))
    
    # 返回结果列表 R 和 S
    return R, S
# 计算两个多项式在 `K[X]` 中的子结果式 PRS（Primitive Recursive Subresultants）。
def dmp_subresultants(f, g, u, K):
    """
    Computes subresultant PRS of two polynomials in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> R.dmp_subresultants(f, g) == [f, g, a, b]
    True

    """
    # 调用内部函数计算子结果式 PRS，返回结果中的第一个多项式
    return dmp_inner_subresultants(f, g, u, K)[0]


# 在 `K[X]` 中使用子结果式 PRS 计算结果式。
def dmp_prs_resultant(f, g, u, K):
    """
    Resultant algorithm in `K[X]` using subresultant PRS.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> res, prs = R.dmp_prs_resultant(f, g)

    >>> res == b             # resultant has n-1 variables
    False
    >>> res == b.drop(x)
    True
    >>> prs == [f, g, a, b]
    True

    """
    # 如果多项式度为零，返回其最高次系数；否则返回子结果式 PRS 结果中的第一个多项式
    if not u:
        return dup_prs_resultant(f, g, K)

    # 如果其中一个多项式为零多项式，则返回相应的零多项式
    if dmp_zero_p(f, u) or dmp_zero_p(g, u):
        return (dmp_zero(u - 1), [])

    # 计算两个多项式的内部子结果式 PRS
    R, S = dmp_inner_subresultants(f, g, u, K)

    # 如果最后一个结果的多项式度大于零，返回零多项式和子结果式 PRS 列表
    if dmp_degree(R[-1], u) > 0:
        return (dmp_zero(u - 1), R)

    # 否则返回最后一个子结果式 PRS 和完整的子结果式 PRS 列表
    return S[-1], R


# 使用模素数 `p` 计算 `K[X]` 中多项式 `f` 和 `g` 的结果式。
def dmp_zz_modular_resultant(f, g, p, u, K):
    """
    Compute resultant of `f` and `g` modulo a prime `p`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x + y + 2
    >>> g = 2*x*y + x + 3

    >>> R.dmp_zz_modular_resultant(f, g, 5)
    -2*y**2 + 1

    """
    # 如果多项式度为零，计算其在有限域中的结果式并返回
    if not u:
        return gf_int(dup_prs_resultant(f, g, K)[0] % p, p)

    # 计算多项式 `f` 和 `g` 的度
    v = u - 1
    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    # 计算 `f` 和 `g` 在首项上的度
    N = dmp_degree_in(f, 1, u)
    M = dmp_degree_in(g, 1, u)

    # 计算辅助参数 B
    B = n*M + m*N

    # 初始化 D 和 a
    D, a = [K.one], -K.one
    r = dmp_zero(v)

    # 循环直到多项式 D 的度大于 B
    while dup_degree(D) <= B:
        while True:
            a += K.one

            if a == p:
                raise HomomorphismFailed('no luck')

            # 计算在有限域中的多项式 `F` 和 `G`
            F = dmp_eval_in(f, gf_int(a, p), 1, u, K)

            if dmp_degree(F, v) == n:
                G = dmp_eval_in(g, gf_int(a, p), 1, u, K)

                if dmp_degree(G, v) == m:
                    break

        # 递归调用计算模素数 `p` 下的结果式
        R = dmp_zz_modular_resultant(F, G, p, v, K)
        e = dmp_eval(r, a, v, K)

        if not v:
            R = dup_strip([R])
            e = dup_strip([e])
        else:
            R = [R]
            e = [e]

        # 计算系数和结果
        d = K.invert(dup_eval(D, a, K), p)
        d = dup_mul_ground(D, d, K)
        d = dmp_raise(d, v, 0, K)

        c = dmp_mul(d, dmp_sub(R, e, v, K), v, K)
        r = dmp_add(r, c, v, K)

        r = dmp_ground_trunc(r, p, v, K)

        D = dup_mul(D, [K.one, -a], K)
        D = dup_trunc(D, p, K)

    return r


# Collins 结果算法的 CRT 包装器函数。
def _collins_crt(r, R, P, p, K):
    """Wrapper of CRT for Collins's resultant algorithm. """
    # 调用 gf_crt 函数，传入参数 [r, R] 和 [P, p]，以及 K
    # 再将 gf_crt 的返回值作为参数传入 gf_int 函数
    # 返回 gf_int 函数的结果
    return gf_int(gf_crt([r, R], [P, p], K), P*p)
# 计算多项式 f 在自然数集合中的次数
n = dmp_degree(f, u)
# 计算多项式 g 在自然数集合中的次数
m = dmp_degree(g, u)

# 如果任何一个多项式的次数小于 0，则返回 u-1 次的零多项式
if n < 0 or m < 0:
    return dmp_zero(u - 1)

# 计算多项式 f 在环 K 中的最大范数
A = dmp_max_norm(f, u, K)
# 计算多项式 g 在环 K 中的最大范数
B = dmp_max_norm(g, u, K)

# 计算多项式 f 在环 K 中的首项系数
a = dmp_ground_LC(f, u, K)
# 计算多项式 g 在环 K 中的首项系数
b = dmp_ground_LC(g, u, K)

# 设置 v 为 u-1
v = u - 1

# 计算 B 的值，包含多项式 f 和 g 的信息
B = K(2) * K.factorial(K(n + m)) * A**m * B**n
# 初始化 r 为 v 次的零多项式，p 和 P 为 K 中的单位元素
r, p, P = dmp_zero(v), K.one, K.one

# 导入下一个素数函数
from sympy.ntheory import nextprime

# 当 P 小于等于 B 时，执行以下循环
while P <= B:
    # 获取下一个素数 p
    p = K(nextprime(p))

    # 如果 a 不可除以 p 或者 b 不可除以 p，则获取下一个素数 p
    while not (a % p) or not (b % p):
        p = K(nextprime(p))

    # 在模 p 下获取多项式 f 和 g 的截断
    F = dmp_ground_trunc(f, p, u, K)
    G = dmp_ground_trunc(g, p, u, K)

    try:
        # 计算在模 p 下的 ZZ 余式结果
        R = dmp_zz_modular_resultant(F, G, p, u, K)
    except HomomorphismFailed:
        # 如果计算失败，继续下一个素数的尝试
        continue

    # 如果 P 是单位元素，则 r 为 R
    # 否则，r 更新为 r 和 R 的合并结果
    if K.is_one(P):
        r = R
    else:
        r = dmp_apply_pairs(r, R, _collins_crt, (P, p, K), v, K)

    # 更新 P 为 P 乘以 p
    P *= p

# 返回计算结果 r
return r
    # 定义环 R 和变量 x，使用整数环 ZZ
    >>> R, x = ring("x", ZZ)

    # 计算多项式 f 的重数（即最高次数）
    >>> R.dup_discriminant(x**2 + 2*x + 3)
    # 返回多项式 x**2 + 2*x + 3 的判别式的值，这里结果是 -8

    """
    # 计算多项式 f 的次数
    d = dup_degree(f)

    # 如果多项式 f 的次数 d 小于等于 0，返回环 K 中的零元素
    if d <= 0:
        return K.zero
    else:
        # 计算 (-1) 的幂，幂次为 (d*(d - 1)) // 2
        s = (-1)**((d*(d - 1)) // 2)
        
        # 计算多项式 f 的首项系数
        c = dup_LC(f, K)
        
        # 计算多项式 f 和其一阶导数的结果式子
        r = dup_resultant(f, dup_diff(f, 1, K), K)
        
        # 返回结果式 r 除以 c*K(s) 的商，其中 K(s) 是环 K 中的元素 s
        return K.quo(r, c*K(s))
# 多项式在环 `K[X]` 中计算判别式
def dmp_discriminant(f, u, K):
    # 如果 `u` 为空（即零维多项式），则调用 `dup_discriminant` 函数计算判别式
    if not u:
        return dup_discriminant(f, K)

    # 计算多项式 `f` 在 `u` 维度下的次数 `d`，以及 `v = u - 1`
    d, v = dmp_degree(f, u), u - 1

    # 如果多项式 `f` 的次数小于等于 0，则返回 `v` 维度下的零多项式
    if d <= 0:
        return dmp_zero(v)
    else:
        # 计算 `-1` 的 `(d*(d - 1)) // 2` 次幂，并确定符号 `s`
        s = (-1)**((d*(d - 1)) // 2)
        # 计算多项式 `f` 的领导系数 `c`
        c = dmp_LC(f, K)

        # 计算多项式 `f` 和其导数的结果式的最大公因式 `r`
        r = dmp_resultant(f, dmp_diff(f, 1, u, K), u, K)
        # 将 `c` 乘以 `K(s)`，得到乘以常数 `s` 后的多项式 `c`
        c = dmp_mul_ground(c, K(s), v, K)

        # 返回 `r` 除以 `c` 的商，结果在 `v` 维度下的多项式
        return dmp_quo(r, c, v, K)


# 在环中处理 GCD 算法中的平凡情况
def _dup_rr_trivial_gcd(f, g, K):
    # 如果 `f` 和 `g` 都是零多项式，则返回空列表作为结果
    if not (f or g):
        return [], [], []
    # 如果 `f` 是零多项式
    elif not f:
        # 如果 `g` 的领导系数非负，则返回 `g` 作为 GCD 的结果，同时返回空列表和 `[K.one]`
        if K.is_nonnegative(dup_LC(g, K)):
            return g, [], [K.one]
        else:
            # 否则返回 `g` 的负数作为 GCD 的结果，同时返回空列表和 `[-K.one]`
            return dup_neg(g, K), [], [-K.one]
    # 如果 `g` 是零多项式
    elif not g:
        # 如果 `f` 的领导系数非负，则返回 `f` 作为 GCD 的结果，同时返回 `[K.one]` 和空列表
        if K.is_nonnegative(dup_LC(f, K)):
            return f, [K.one], []
        else:
            # 否则返回 `f` 的负数作为 GCD 的结果，同时返回 `[-K.one]` 和空列表
            return dup_neg(f, K), [-K.one], []

    # 如果以上情况均不满足，则返回 `None`
    return None


# 在域中处理 GCD 算法中的平凡情况
def _dup_ff_trivial_gcd(f, g, K):
    # 如果 `f` 和 `g` 都是零多项式，则返回空列表作为结果
    if not (f or g):
        return [], [], []
    # 如果 `f` 是零多项式，则返回 `g` 的首一多项式作为 GCD 的结果，同时返回空列表和 `g` 的领导系数
    elif not f:
        return dup_monic(g, K), [], [dup_LC(g, K)]
    # 如果 `g` 是零多项式，则返回 `f` 的首一多项式作为 GCD 的结果，同时返回 `f` 的领导系数和空列表
    elif not g:
        return dup_monic(f, K), [dup_LC(f, K)], []
    else:
        # 如果以上情况均不满足，则返回 `None`
        return None


# 在环中处理多变量 GCD 算法中的平凡情况
def _dmp_rr_trivial_gcd(f, g, u, K):
    # 判断多项式 `f` 和 `g` 是否都是零多项式
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    # 判断多项式 `f` 和 `g` 是否包含单位元 `1`
    if_contain_one = dmp_one_p(f, u, K) or dmp_one_p(g, u, K)

    # 如果 `f` 和 `g` 都是零多项式，则返回三个零多项式组成的元组
    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    # 如果 `f` 是零多项式
    elif zero_f:
        # 如果 `g` 的领导系数非负，则返回 `g` 作为 GCD 的结果，同时返回 `u` 维度下的零多项式和 `u` 维度下的单位元 `1`
        if K.is_nonnegative(dmp_ground_LC(g, u, K)):
            return g, dmp_zero(u), dmp_one(u, K)
        else:
            # 否则返回 `g` 的负多项式作为 GCD 的结果，同时返回 `u` 维度下的零多项式和 `-1` 的 `u` 维度下的常数多项式
            return dmp_neg(g, u, K), dmp_zero(u), dmp_ground(-K.one, u)
    # 如果 `g` 是零多项式
    elif zero_g:
        # 如果 `f` 的领导系数非负，则返回 `f` 作为 GCD 的结果，同时返回 `u` 维度下的单位元 `1` 和 `u` 维度下的零多项式
        if K.is_nonnegative(dmp_ground_LC(f, u, K)):
            return f, dmp_one(u, K), dmp_zero(u)
        else:
            # 否则返回 `f` 的负多项式作为 GCD 的结果，同时返回 `-1` 的 `u` 维度下的常数多项式和 `u` 维度下的零多项式
            return dmp_neg(f, u, K), dmp_ground(-K.one, u), dmp_zero(u)
    # 如果 `f` 或 `g` 中包含单位元 `1`
    elif if_contain_one:
        # 返回 `u` 维度下的单位元 `1`，以及多项式 `f` 和 `g`
        return dmp_one(u, K), f, g
    # 如果启用了简化 GCD 计算
    elif query('USE_SIMPLIFY_GCD'):
        # 调用 `_dmp_simplify_gcd` 函数处理 `f` 和 `g` 的 GCD 计算
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        # 如果以上情况均不满足，则返回 `None`
        return None


# 在域中处理多变量 GCD 算法中的平凡情况
def _dmp_ff_trivial_gcd(f, g, u, K):
    # 判断多项式 `f` 和 `g` 是否都是零多项式
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)

    # 如果 `f` 和 `g` 都是零多项式，则返回三个零多项式组成的元组
    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    # 如果 `f` 是零多项式
    elif zero_f:
        # 返回 `g` 的首一多项式，`u` 维度下的零多项式，以及 `g` 的领导系数所对应的常数多项式
        return (dmp_ground_monic(g, u, K),
                dmp_zero(u),
                dmp_ground(dmp_ground_LC(g, u, K), u))
    # 如果 `g` 是零多项式
    elif zero_g:
        # 返回 `f` 的首一多项式，`f` 的领导系数所对应的常数多项式，以及 `u` 维
    # 计算多项式 f 和 g 的度数，相对于变量 u
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    # 如果 f 和 g 的最高次数都大于 0，则返回 None
    if df > 0 and dg > 0:
        return None

    # 如果 f 和 g 的最高次数都为 0
    if not (df or dg):
        # 计算 f 和 g 的领头系数
        F = dmp_LC(f, K)
        G = dmp_LC(g, K)
    else:
        # 如果 f 的最高次数为 0，g 的最高次数不为 0
        if not df:
            # 计算 f 的领头系数，g 的内容
            F = dmp_LC(f, K)
            G = dmp_content(g, u, K)
        else:
            # 如果 f 的最高次数不为 0，计算 f 的内容，g 的领头系数
            F = dmp_content(f, u, K)
            G = dmp_LC(g, K)

    # 设置变量 v 为 u - 1
    v = u - 1
    # 计算多项式 F 和 G 在变量 v 下的最大公因子
    h = dmp_gcd(F, G, v, K)

    # 计算 f 和 g 除以 h 的商多项式列表
    cff = [ dmp_quo(cf, h, v, K) for cf in f ]
    cfg = [ dmp_quo(cg, h, v, K) for cg in g ]

    # 返回最大公因子 h，以及 f 和 g 除以 h 的商多项式列表
    return [h], cff, cfg
# 计算多项式的最大公因式（GCD），使用子结果式方法，适用于环（整数环）。
# 返回 ``(h, cff, cfg)``，其中 ``a = gcd(f, g)``, ``cff = quo(f, h)``, ``cfg = quo(g, h)``。
def dup_rr_prs_gcd(f, g, K):
    result = _dup_rr_trivial_gcd(f, g, K)  # 尝试使用简单方法计算 GCD

    if result is not None:  # 如果简单方法成功找到 GCD，则直接返回结果
        return result

    fc, F = dup_primitive(f, K)  # 将 f 简化为本原多项式 fc 和对应的 F
    gc, G = dup_primitive(g, K)  # 将 g 简化为本原多项式 gc 和对应的 G

    c = K.gcd(fc, gc)  # 计算 fc 和 gc 的最大公约数

    h = dup_subresultants(F, G, K)[-1]  # 计算 F 和 G 的子结果式的最后一个结果
    _, h = dup_primitive(h, K)  # 将 h 简化为本原多项式

    c *= K.canonical_unit(dup_LC(h, K))  # 乘以 h 的首项系数的规范单位

    h = dup_mul_ground(h, c, K)  # 将 h 乘以常数 c

    cff = dup_quo(f, h, K)  # 计算 f 除以 h 的商
    cfg = dup_quo(g, h, K)  # 计算 g 除以 h 的商

    return h, cff, cfg


# 计算多项式的最大公因式（GCD），使用子结果式方法，适用于域（有理数域）。
# 返回 ``(h, cff, cfg)``，其中 ``a = gcd(f, g)``, ``cff = quo(f, h)``, ``cfg = quo(g, h)``。
def dup_ff_prs_gcd(f, g, K):
    result = _dup_ff_trivial_gcd(f, g, K)  # 尝试使用简单方法计算 GCD

    if result is not None:  # 如果简单方法成功找到 GCD，则直接返回结果
        return result

    h = dup_subresultants(f, g, K)[-1]  # 计算 f 和 g 的子结果式的最后一个结果
    h = dup_monic(h, K)  # 将 h 转换为首一多项式

    cff = dup_quo(f, h, K)  # 计算 f 除以 h 的商
    cfg = dup_quo(g, h, K)  # 计算 g 除以 h 的商

    return h, cff, cfg


# 计算多变量多项式的最大公因式（GCD），使用子结果式方法，适用于环（整数环）。
# 返回 ``(h, cff, cfg)``，其中 ``a = gcd(f, g)``, ``cff = quo(f, h)``, ``cfg = quo(g, h)``。
def dmp_rr_prs_gcd(f, g, u, K):
    if not u:  # 如果变量数为 0，则调用 dup_rr_prs_gcd 处理
        return dup_rr_prs_gcd(f, g, K)

    result = _dmp_rr_trivial_gcd(f, g, u, K)  # 尝试使用简单方法计算 GCD

    if result is not None:  # 如果简单方法成功找到 GCD，则直接返回结果
        return result

    fc, F = dmp_primitive(f, u, K)  # 将 f 简化为本原多项式 fc 和对应的 F
    gc, G = dmp_primitive(g, u, K)  # 将 g 简化为本原多项式 gc 和对应的 G

    h = dmp_subresultants(F, G, u, K)[-1]  # 计算 F 和 G 的子结果式的最后一个结果
    c, _, _ = dmp_rr_prs_gcd(fc, gc, u - 1, K)  # 递归计算 fc 和 gc 的 GCD

    _, h = dmp_primitive(h, u, K)  # 将 h 简化为本原多项式
    h = dmp_mul_term(h, c, 0, u, K)  # 将 h 乘以常数 c

    unit = K.canonical_unit(dmp_ground_LC(h, u, K))  # 获取 h 的首项系数的规范单位

    if unit != K.one:  # 如果单位不是 1，则将 h 乘以单位
        h = dmp_mul_ground(h, unit, u, K)

    cff = dmp_quo(f, h, u, K)  # 计算 f 除以 h 的商
    cfg = dmp_quo(g, h, u, K)  # 计算 g 除以 h 的商

    return h, cff, cfg


# 计算多变量多项式的最大公因式（GCD），使用子结果式方法，适用于域（有理数域）。
# 返回 ``(h, cff, cfg)``，其中 ``a = gcd(f, g)``, ``cff = quo(f, h)``, ``cfg = quo(g, h)``。
def dmp_ff_prs_gcd(f, g, u, K):
    if not u:  # 如果变量数为 0，则调用 dup_ff_prs_gcd 处理
        return dup_ff_prs_gcd(f, g, K)
    # 调用 _dmp_ff_trivial_gcd 函数尝试计算简单的多项式最大公因子
    result = _dmp_ff_trivial_gcd(f, g, u, K)

    # 如果 result 不为 None，则直接返回计算得到的最大公因子
    if result is not None:
        return result

    # 对 f 和 g 进行本原多项式分解，返回本原因子 fc 和 F
    fc, F = dmp_primitive(f, u, K)
    # 对 g 进行本原多项式分解，返回本原因子 gc 和 G
    gc, G = dmp_primitive(g, u, K)

    # 计算 F 和 G 的子结果式，选择最后一个结果作为 h
    h = dmp_subresultants(F, G, u, K)[-1]
    # 使用本原因子 fc 和 gc 计算 f 和 g 的最大公因子 c
    c, _, _ = dmp_ff_prs_gcd(fc, gc, u - 1, K)

    # 对 h 进行本原多项式分解，返回本原因子和本原化的多项式 h
    _, h = dmp_primitive(h, u, K)
    # 将 h 乘以常数因子 c
    h = dmp_mul_term(h, c, 0, u, K)
    # 将 h 本原化（将 h 的首项系数调整为单位元素）
    h = dmp_ground_monic(h, u, K)

    # 计算 f 除以 h 的商，返回商 cff
    cff = dmp_quo(f, h, u, K)
    # 计算 g 除以 h 的商，返回商 cfg
    cfg = dmp_quo(g, h, u, K)

    # 返回计算得到的最大公因子 h，以及 f 和 g 除以 h 的商 cff 和 cfg
    return h, cff, cfg
# 最大迭代次数，用于控制启发式算法的迭代次数上限
HEU_GCD_MAX = 6

# 从整数的最大公约数插值出多项式的最大迭代次数
def _dup_zz_gcd_interpolate(h, x, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []

    # 当 h 不为零时循环执行以下操作
    while h:
        # 计算 h 对 x 取模的结果
        g = h % x

        # 若 g 大于 x 的一半，则减去 x
        if g > x // 2:
            g -= x

        # 将 g 插入到 f 的最前面
        f.insert(0, g)

        # 更新 h 为 (h - g) // x
        h = (h - g) // x

    return f


# 在整数环 Z[x] 中的启发式多项式最大公约数算法
def dup_zz_heu_gcd(f, g, K):
    """
    Heuristic polynomial GCD in `Z[x]`.

    Given univariate polynomials `f` and `g` in `Z[x]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation.  The final step is to verify if the result is the
    correct GCD. This gives cofactors as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_zz_heu_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    References
    ==========

    .. [1] [Liao95]_

    """
    # 尝试使用简单多项式最大公约数算法
    result = _dup_rr_trivial_gcd(f, g, K)

    # 如果简单算法成功找到结果，则直接返回
    if result is not None:
        return result

    # 获取多项式 f 和 g 的次数
    df = dup_degree(f)
    dg = dup_degree(g)

    # 提取 f 和 g 的最大公约数和较小的多项式
    gcd, f, g = dup_extract(f, g, K)

    # 如果其中一个多项式的次数为零，则直接返回
    if df == 0 or dg == 0:
        return [gcd], f, g

    # 计算 f 和 g 的最大范数
    f_norm = dup_max_norm(f, K)
    g_norm = dup_max_norm(g, K)

    # 计算 B 的值，用于多项式 GCD 的计算
    B = K(2*min(f_norm, g_norm) + 29)

    # 计算 x 的值，用于多项式 GCD 的计算
    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dup_LC(f, K)),
                  g_norm // abs(dup_LC(g, K))) + 4)

    # 迭代启发式算法的主循环
    for i in range(0, HEU_GCD_MAX):
        # 在点 x 处计算多项式 f 和 g 的值
        ff = dup_eval(f, x, K)
        gg = dup_eval(g, x, K)

        # 如果 ff 和 gg 都不为零，则计算它们的最大公约数 h
        if ff and gg:
            h = K.gcd(ff, gg)

            # 计算 ff 和 gg 的商
            cff = ff // h
            cfg = gg // h

            # 从整数的最大公约数插值出多项式的最大公约数 h
            h = _dup_zz_gcd_interpolate(h, x, K)
            h = dup_primitive(h, K)[1]

            # 验证 h 是否是正确的多项式最大公约数
            cff_, r = dup_div(f, h, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg_

            # 从整数的最大公约数插值出多项式的商 cff
            cff = _dup_zz_gcd_interpolate(cff, x, K)

            h, r = dup_div(f, cff, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff, cfg_

            # 从整数的最大公约数插值出多项式的商 cfg
            cfg = _dup_zz_gcd_interpolate(cfg, x, K)

            h, r = dup_div(g, cfg, K)

            if not r:
                cff_, r = dup_div(f, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg

        # 更新 x 的值
        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    # 如果多项式最大公约数计算失败，则抛出异常
    raise HeuristicGCDFailed('no luck')
# 如果 u 为零，调用 dup_zz_heu_gcd 函数求解 f 和 g 的最大公因数
def dmp_zz_heu_gcd(f, g, u, K):
    """
    Heuristic polynomial GCD in `Z[X]`.

    Given univariate polynomials `f` and `g` in `Z[X]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation. The evaluation process reduces f and g variable by
    variable into a large integer.  The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_zz_heu_gcd(f, g)
    (x + y, x + y, x)

    References
    ==========

    .. [1] [Liao95]_

    """
    # 如果 u 为零，则使用 dup_zz_heu_gcd 函数求解 f 和 g 的最大公因数
    if not u:
        return dup_zz_heu_gcd(f, g, K)

    # 调用 _dmp_rr_trivial_gcd 函数尝试使用简单的方法计算 f 和 g 的最大公因数
    result = _dmp_rr_trivial_gcd(f, g, u, K)

    # 如果结果不为 None，则直接返回计算结果
    if result is not None:
        return result

    # 提取 f 和 g 的常数部分，并计算它们的最大范数
    gcd, f, g = dmp_ground_extract(f, g, u, K)

    # 计算 f 和 g 的最大范数
    f_norm = dmp_max_norm(f, u, K)
    g_norm = dmp_max_norm(g, u, K)

    # 计算 B 的值，其中 B 是一个常数
    B = K(2*min(f_norm, g_norm) + 29)

    # 计算 x 的值，其中 x 是用于插值的变量
    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dmp_ground_LC(f, u, K)),
                  g_norm // abs(dmp_ground_LC(g, u, K))) + 4)
    # 循环迭代范围为0到HEU_GCD_MAX，尝试执行以下操作
    for i in range(0, HEU_GCD_MAX):
        # 使用多项式f在变量x和u上的求值，结果存储在ff中
        ff = dmp_eval(f, x, u, K)
        # 使用多项式g在变量x和u上的求值，结果存储在gg中
        gg = dmp_eval(g, x, u, K)

        # 变量v赋值为u-1
        v = u - 1

        # 如果ff在变量v上不为零且gg在变量v上不为零
        if not (dmp_zero_p(ff, v) or dmp_zero_p(gg, v)):
            # 对ff和gg在变量v上进行heu_gcd算法，结果存储在h, cff, cfg中
            h, cff, cfg = dmp_zz_heu_gcd(ff, gg, v, K)

            # 使用插值技术对h进行重新插值，基于变量x和v，结果赋给h
            h = _dmp_zz_gcd_interpolate(h, x, v, K)
            # 对h进行去整，返回的第二个值赋给h
            h = dmp_ground_primitive(h, u, K)[1]

            # 使用f除以h，商赋给cff_，余数赋给r
            cff_, r = dmp_div(f, h, u, K)

            # 如果余数r在变量u上为零
            if dmp_zero_p(r, u):
                # 使用g除以h，商赋给cfg_，余数赋给r
                cfg_, r = dmp_div(g, h, u, K)

                # 如果余数r在变量u上为零
                if dmp_zero_p(r, u):
                    # 将h乘以gcd，并返回结果h, cff_, cfg_
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg_

            # 使用cff除以f，商赋给h，余数赋给r
            h, r = dmp_div(f, cff, u, K)

            # 如果余数r在变量u上为零
            if dmp_zero_p(r, u):
                # 使用g除以h，商赋给cfg_，余数赋给r
                cfg_, r = dmp_div(g, h, u, K)

                # 如果余数r在变量u上为零
                if dmp_zero_p(r, u):
                    # 将h乘以gcd，并返回结果h, cff, cfg_
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff, cfg_

            # 使用cfg除以g，商赋给h，余数赋给r
            h, r = dmp_div(g, cfg, u, K)

            # 如果余数r在变量u上为零
            if dmp_zero_p(r, u):
                # 使用f除以h，商赋给cff_，余数赋给r
                cff_, r = dmp_div(f, h, u, K)

                # 如果余数r在变量u上为零
                if dmp_zero_p(r, u):
                    # 将h乘以gcd，并返回结果h, cff_, cfg
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg

        # 更新变量x的值
        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    # 如果循环内没有成功返回，则触发异常HeuristicGCDFailed
    raise HeuristicGCDFailed('no luck')
def dup_inner_gcd(f, g, K):
    """
    Computes polynomial GCD and cofactors of `f` and `g` in `K[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_inner_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    # XXX: This used to check for K.is_Exact but leads to awkward results when
    # the domain is something like RR[z] e.g.:
    #
    # >>> g, p, q = Poly(1, x).cancel(Poly(51.05*x*y - 1.0, x))
    # >>> g
    # 1.0
    # >>> p
    # Poly(17592186044421.0, x, domain='RR[y]')
    # >>> q
    # Poly(898081097567692.0*y*x - 17592186044421.0, x, domain='RR[y]'))
    #
    # Maybe it would be better to flatten into multivariate polynomials first.

    # 这里的注释提醒了一个曾经的实现问题，即在处理非精确域时可能导致奇怪的结果。
    # 建议先将多变量多项式展开，以避免这种情况的发生。
    pass



def dup_qq_heu_gcd(f, g, K0):
    """
    Heuristic polynomial GCD in `Q[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x

    >>> R.dup_qq_heu_gcd(f, g)
    (x + 2, 1/2*x + 3/4, 1/2*x)

    """
    result = _dup_ff_trivial_gcd(f, g, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dup_clear_denoms(f, K0, K1)
    cg, g = dup_clear_denoms(g, K0, K1)

    f = dup_convert(f, K0, K1)
    g = dup_convert(g, K0, K1)

    h, cff, cfg = dup_zz_heu_gcd(f, g, K1)

    h = dup_convert(h, K1, K0)

    c = dup_LC(h, K0)
    h = dup_monic(h, K0)

    cff = dup_convert(cff, K1, K0)
    cfg = dup_convert(cfg, K1, K0)

    cff = dup_mul_ground(cff, K0.quo(c, cf), K0)
    cfg = dup_mul_ground(cfg, K0.quo(c, cg), K0)

    return h, cff, cfg



def dmp_qq_heu_gcd(f, g, u, K0):
    """
    Heuristic polynomial GCD in `Q[X]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x, y = ring("x,y", QQ)

    >>> f = QQ(1,4)*x**2 + x*y + y**2
    >>> g = QQ(1,2)*x**2 + x*y

    >>> R.dmp_qq_heu_gcd(f, g)
    (x + 2*y, 1/4*x + 1/2*y, 1/2*x)

    """
    result = _dmp_ff_trivial_gcd(f, g, u, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dmp_clear_denoms(f, u, K0, K1)
    cg, g = dmp_clear_denoms(g, u, K0, K1)

    f = dmp_convert(f, u, K0, K1)
    g = dmp_convert(g, u, K0, K1)

    h, cff, cfg = dmp_zz_heu_gcd(f, g, u, K1)

    h = dmp_convert(h, u, K1, K0)

    c = dmp_ground_LC(h, u, K0)
    h = dmp_ground_monic(h, u, K0)

    cff = dmp_convert(cff, u, K1, K0)
    cfg = dmp_convert(cfg, u, K1, K0)

    cff = dmp_mul_ground(cff, K0.quo(c, cf), u, K0)
    cfg = dmp_mul_ground(cfg, K0.quo(c, cg), u, K0)

    return h, cff, cfg
    # 如果域 K 是实数域或复数域
    if K.is_RR or K.is_CC:
        try:
            # 尝试获取 K 的精确表示
            exact = K.get_exact()
        except DomainError:
            # 如果获取精确表示时出现域错误，返回 [K.one] 作为结果以及 f 和 g 本身
            return [K.one], f, g

        # 将 f 和 g 转换为精确表示 exact 的系数类型
        f = dup_convert(f, K, exact)
        g = dup_convert(g, K, exact)

        # 对转换后的 f 和 g 计算内部最大公因式 h 和相应的系数
        h, cff, cfg = dup_inner_gcd(f, g, exact)

        # 将计算得到的 h、cff 和 cfg 转换回原始的域 K 的系数类型
        h = dup_convert(h, exact, K)
        cff = dup_convert(cff, exact, K)
        cfg = dup_convert(cfg, exact, K)

        # 返回计算得到的 h、cff 和 cfg
        return h, cff, cfg
    # 如果域 K 是一个域（Field），进一步判断其类型
    elif K.is_Field:
        # 如果 K 是有理数域 QQ，并且使用启发式算法 USE_HEU_GCD
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                # 尝试使用 QQ 域的启发式 GCD 算法进行计算
                return dup_qq_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                # 如果启发式 GCD 算法失败，继续执行下面的操作
                pass

        # 对于其他类型的域 K，或者 QQ 域没有启用 USE_HEU_GCD
        # 使用有限域或多项式环的快速幂算法进行 GCD 计算
        return dup_ff_prs_gcd(f, g, K)
    else:
        # 如果 K 是整数环 ZZ，并且使用启发式算法 USE_HEU_GCD
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                # 尝试使用 ZZ 域的启发式 GCD 算法进行计算
                return dup_zz_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                # 如果启发式 GCD 算法失败，继续执行下面的操作
                pass

        # 对于其他类型的环或整数环没有启用 USE_HEU_GCD
        # 使用有理数环或复数环的快速幂算法进行 GCD 计算
        return dup_rr_prs_gcd(f, g, K)
def _dmp_inner_gcd(f, g, u, K):
    """Helper function for `dmp_inner_gcd()`. """
    # 如果域 K 不是精确的，则尝试获取其精确表示
    if not K.is_Exact:
        try:
            exact = K.get_exact()
        except DomainError:
            # 如果获取精确表示失败，则返回默认结果
            return dmp_one(u, K), f, g

        # 将输入的多项式 f 和 g 转换为精确域 exact 中的多项式
        f = dmp_convert(f, u, K, exact)
        g = dmp_convert(g, u, K, exact)

        # 在精确域 exact 中执行 _dmp_inner_gcd
        h, cff, cfg = _dmp_inner_gcd(f, g, u, exact)

        # 将结果转换回原始域 K
        h = dmp_convert(h, u, exact, K)
        cff = dmp_convert(cff, u, exact, K)
        cfg = dmp_convert(cfg, u, exact, K)

        return h, cff, cfg
    elif K.is_Field:
        # 如果 K 是一个域，并且是有理数域 QQ 且启用了启发式 GCD
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                # 尝试使用启发式算法计算 GCD
                return dmp_qq_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        # 否则使用有限域或扩展域中的最小多项式算法计算 GCD
        return dmp_ff_prs_gcd(f, g, u, K)
    else:
        # 如果 K 不是域，但是是整数环 ZZ 且启用了启发式 GCD
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                # 尝试使用启发式算法计算 GCD
                return dmp_zz_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        # 否则使用实数环或有理数环中的最小多项式算法计算 GCD
        return dmp_rr_prs_gcd(f, g, u, K)
    # 导入 sympy.polys 中的 ring 和 QQ 模块，创建有理数域上的多项式环 R 和变量 x
    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)
    
    # 使用有理数 QQ 创建多项式 f 和 g
    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x
    
    # 计算 f 和 g 的最小公倍式（least common multiple）
    >>> R.dup_ff_lcm(f, g)
    x**3 + 7/2*x**2 + 3*x
    
    """
    h = dup_quo(dup_mul(f, g, K),
                dup_gcd(f, g, K), K)
    
    return dup_monic(h, K)
    """
def dup_cancel(f, g, K, include=True):
    """
    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_cancel(x**2 - 1, x + 1)
    x - 1

    """
    # 导入 sympy.polys 中的 ring 和 ZZ 模块
    from sympy.polys import ring, ZZ
    # 使用 ring 函数创建一个多项式环 R，并创建一个未定元 x
    R, x = ring("x", ZZ)
    
    # 在环 R 上调用 dup_cancel 方法，对两个多项式进行因式分解和约简操作
    # 第一个参数是被约简的多项式，第二个参数是用于约简的多项式
    # 返回一个包含两个多项式的元组，分别是约简后的商和余数
    R.dup_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    """
    (2*x + 2, x - 1)
    """
# 取消有理函数 `f/g` 中的公因子。
def dmp_cancel(f, g, u, K, include=True):
    # 初始化 K0 为 None
    K0 = None

    # 如果 K 是一个域（Field）且具有关联环（Ring），则进行以下操作
    if K.is_Field and K.has_assoc_Ring:
        # 将 K0 设置为当前 K，然后将 K 转换为其关联环
        K0, K = K, K.get_ring()

        # 清除 f 和 g 的分母，返回清除后的结果及清除的常数部分
        cq, f = dmp_clear_denoms(f, u, K0, K, convert=True)
        cp, g = dmp_clear_denoms(g, u, K0, K, convert=True)
    else:
        # 如果 K 不是域或没有关联环，则将 cp 和 cq 初始化为 K 的单位元素
        cp, cq = K.one, K.one

    # 计算 f 和 g 的最内部 GCD（最大公因子）
    _, p, q = dmp_inner_gcd(f, g, u, K)

    # 如果 K0 不为 None，则进行以下操作
    if K0 is not None:
        # 计算 cp 和 cq 的系数
        _, cp, cq = K.cofactors(cp, cq)

        # 将 p 和 q 转换为 K0 环上的多项式
        p = dmp_convert(p, u, K, K0)
        q = dmp_convert(q, u, K, K0)

        # 将 K 设置为 K0
        K = K0

    # 检查 p 和 q 的首项系数是否为负数
    p_neg = K.is_negative(dmp_ground_LC(p, u, K))
    q_neg = K.is_negative(dmp_ground_LC(q, u, K))

    # 如果 p 和 q 的首项系数都为负数，则将它们都取反
    if p_neg and q_neg:
        p, q = dmp_neg(p, u, K), dmp_neg(q, u, K)
    elif p_neg:
        cp, p = -cp, dmp_neg(p, u, K)
    elif q_neg:
        cp, q = -cp, dmp_neg(q, u, K)

    # 如果不包含额外信息，则返回 cp, cq, p, q
    if not include:
        return cp, cq, p, q

    # 将 p 和 q 分别乘以 cp 和 cq
    p = dmp_mul_ground(p, cp, u, K)
    q = dmp_mul_ground(q, cq, u, K)

    # 返回处理后的多项式 p 和 q
    return p, q
```