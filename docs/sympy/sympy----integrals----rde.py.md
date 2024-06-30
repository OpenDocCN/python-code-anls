# `D:\src\scipysrc\sympy\sympy\integrals\rde.py`

```
# 导入乘法操作符和reduce函数
from operator import mul
from functools import reduce

# 导入无穷大和虚拟符号
from sympy.core import oo
from sympy.core.symbol import Dummy

# 导入多项式相关函数和整数多项式
from sympy.polys import Poly, gcd, ZZ, cancel

# 导入复数函数和平方根函数
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt

# 导入risch.py中的部分函数和异常
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative)

# TODO: 在NonElementaryIntegralException异常中添加错误消息

def order_at(a, p, t):
    """
    计算a在p处相对于t的阶数。

    说明
    ===========

    对于k[t]中的a、p，a在p处相对于t的阶数定义为nu_p(a) = max({n ∈ Z+ | p**n|a}), 其中a ≠ 0。如果a == 0，则nu_p(a) = +oo。

    要计算有理函数a/b的阶数，在a/b的情况下使用nu_p(a) == nu_p(a) - nu_p(b)。
    """
    if a.is_zero:
        return oo
    if p == Poly(t, t):
        return a.as_poly(t).ET()[0][0]

    # 使用二分查找计算幂。power_list收集元组(p^k,k)，其中k是2的某个幂次方。确定最大的k后，
    # p^k|a并迭代计算实际幂。
    power_list = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1,tracks_power))
        p1 = p1*p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = Poly(1, t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product*final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n


def order_at_oo(a, d, t):
    """
    计算a/d在oo（无穷大）处相对于t的阶数。

    对于k(t)中的f，f在oo处的阶数定义为deg(d) - deg(a)，其中f == a/d。
    """
    if a.is_zero:
        return oo
    return d.degree(t) - a.degree(t)
def weak_normalizer(a, d, DE, z=None):
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)
    """
    z = z or Dummy('z')  # 如果 z 为 None，则创建一个虚拟符号 'z'
    dn, ds = splitfactor(d, DE)

    # 计算 d1，其中 dn == d1*d2**2*...*dn**n 是 d 的无平方因子分解
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))

    # 使用 Diophantine 解法计算 a/d1 在 DE.t 下的 gcd 扩展
    a1, b = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t),
        a.as_poly(DE.t))
    r = (a - Poly(z, DE.t)*derivation(d1, DE)).as_poly(DE.t).resultant(
        d1.as_poly(DE.t))
    r = Poly(r, z)

    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))

    N = [i for i in r.real_roots() if i in ZZ and i > 0]

    # 计算 q，使得 f - Dq/q 弱正规化
    q = reduce(mul, [gcd(a - Poly(n, DE.t)*derivation(d1, DE), d1) for n in N],
        Poly(1, DE.t))

    dq = derivation(q, DE)
    sn = q*a - d*dq
    sd = q*d
    sn, sd = sn.cancel(sd, include=True)

    return (q, (sn, sd))


def normal_denom(fa, fd, ga, gd, DE):
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.
    """
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h
    if c.div(en)[1]:
        # en does not divide dn*h**2
        raise NonElementaryIntegralException

    ca = c*ga
    ca, cd = ca.cancel(gd, include=True)

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    # 返回 quadruplet (dn*h, dn*h*f - dn*Dh, dn*h**2*g, h)
    return (a, (ba, bd), (ca, cd), h)


def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    # 根据情况设置 case 的值为 DE.case（假设 DE 是某个上下文的对象）
    if case == 'auto':
        case = DE.case

    # 根据不同的 case 值，选择不同的多项式 p
    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        # 对于 'primitive' 或 'base' 情况，直接返回 (a, B, C, Poly(1, DE.t)) 四元组
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        # 抛出值错误的异常，如果 case 不在预期的值中
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    # 计算 nb 和 nc 的值
    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)

    # 计算 n 的值
    n = min(0, nc - min(0, nb))

    # 如果 nb 为假（即为0），则可能存在取消项
    if not nb:
        # 导入 parametric_log_deriv 函数
        from .prde import parametric_log_deriv
        if case == 'exp':
            # 对于 'exp' 情况，计算相关系数
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            # 使用 DecrementLevel 上下文，计算 parametric_log_deriv
            with DecrementLevel(DE):  # 因为 case != 'base'，所以可以保证不会有问题
                # 计算 alphaa 和 alphad
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                # 调用 parametric_log_deriv 函数并检查结果
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            # 对于 'tan' 情况，计算相关系数
            dcoeff = DE.d.quo(Poly(DE.t**2+1, DE.t))
            with DecrementLevel(DE):  # 因为 case != 'base'，所以可以保证不会有问题
                # 计算 alphaa 和 alphad
                alphaa, alphad = frac_in(im(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                betaa, betad = frac_in(re(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                # 检查是否为对数导数
                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    # 调用 parametric_log_deriv 函数并检查结果
                    A = parametric_log_deriv(alphaa*Poly(sqrt(-1), DE.t)*betad+alphad*betaa, alphad*betad, etaa, etad, DE)
                    if A is not None:
                       Q, m, z = A
                       if Q == 1:
                           n = min(n, m)
    # 计算 N 和 pN 的值
    N = max(0, -nb, n - nc)
    pN = p**N
    pn = p**-n

    # 计算 A、B、C 和 h 的值
    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    C = (ca*pN*pn).quo(cd)
    h = pn

    # 返回结果四元组
    # (a*p**N, (b + n*a*Dp/p)*p**N, c*p**(N - n), p**-n)
    return (A, B, C, h)
# 根据给定的参数和条件，计算多项式解的上界。

def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of Polys.

    This constitutes step 3 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    # 根据情况设定参数值
    if case == 'auto':
        case = DE.case

    # 计算多项式 a, b 在给定导数 D 下的次数
    da = a.degree(DE.t)
    db = b.degree(DE.t)

    # 根据 parametric 的不同取值，计算 cQ 的最高次数
    if parametric:
        dc = max(i.degree(DE.t) for i in cQ)
    else:
        dc = cQ.degree(DE.t)

    # 计算 alpha 值，这是一个关键的参数
    alpha = cancel(-b.as_poly(DE.t).LC().as_expr() /
        a.as_poly(DE.t).LC().as_expr())

    # 根据 case 的值进行不同情况的处理
    if case == 'base':
        # 计算多项式解的上界 n
        n = max(0, dc - max(db, da - 1))
        # 在特定条件下进一步调整 n 的值
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)
    # 如果 case 是 'primitive'
    elif case == 'primitive':
        # 如果 db 大于 da
        if db > da:
            # 计算 n，取 0 和 dc - db 中较大的值
            n = max(0, dc - db)
        else:
            # 计算 n，取 0 和 dc - da + 1 中较大的值
            n = max(0, dc - da + 1)

        # 计算 DE.d / DE.T[DE.level - 1] 的分数形式
        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])

        # 将 DE.t 赋值给 t1
        t1 = DE.t

        # 使用 DecrementLevel(DE) 这个上下文管理器
        with DecrementLevel(DE):
            # 将 alpha / DE.t 的分数形式赋值给 alphaa, alphad
            alphaa, alphad = frac_in(alpha, DE.t)

            # 如果 db == da - 1
            if db == da - 1:
                # 从 .prde 模块导入 limited_integrate 函数
                from .prde import limited_integrate
                # 尝试调用 limited_integrate 函数
                try:
                    # 调用 limited_integrate 函数，传入参数 alphaa, alphad, [(etaa, etad)], DE
                    (za, zd), m = limited_integrate(alphaa, alphad, [(etaa, etad)], DE)
                # 捕获 NonElementaryIntegralException 异常
                except NonElementaryIntegralException:
                    # 如果捕获到异常，什么也不做
                    pass
                else:
                    # 如果 m 的长度不为 1，抛出 ValueError 异常
                    if len(m) != 1:
                        raise ValueError("Length of m should be 1")
                    # 更新 n，取 n 和 m[0] 中较大的值
                    n = max(n, m[0])

            # 如果 db == da
            elif db == da:
                # 从 .prde 模块导入 is_log_deriv_k_t_radical_in_field 函数
                from .prde import is_log_deriv_k_t_radical_in_field
                # 调用 is_log_deriv_k_t_radical_in_field 函数，传入参数 alphaa, alphad, DE
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                # 如果 A 不为 None
                if A is not None:
                    # 将 A 解包为 aa, z
                    aa, z = A
                    # 如果 aa 等于 1
                    if aa == 1:
                        # 计算 beta，并赋值给 beta
                        beta = -(a*derivation(z, DE).as_poly(t1) +
                                 b*z.as_poly(t1)).LC()/(z.as_expr()*a.LC())
                        # 将 beta / DE.t 的分数形式赋值给 betaa, betad
                        betaa, betad = frac_in(beta, DE.t)
                        # 从 .prde 模块导入 limited_integrate 函数
                        from .prde import limited_integrate
                        # 尝试调用 limited_integrate 函数
                        try:
                            # 调用 limited_integrate 函数，传入参数 betaa, betad, [(etaa, etad)], DE
                            (za, zd), m = limited_integrate(betaa, betad, [(etaa, etad)], DE)
                        # 捕获 NonElementaryIntegralException 异常
                        except NonElementaryIntegralException:
                            # 如果捕获到异常，什么也不做
                            pass
                        else:
                            # 如果 m 的长度不为 1，抛出 ValueError 异常
                            if len(m) != 1:
                                raise ValueError("Length of m should be 1")
                            # 更新 n，取 n 和 m[0].as_expr() 中较大的值
                            n = max(n, m[0].as_expr())

    # 如果 case 是 'exp'
    elif case == 'exp':
        # 从 .prde 模块导入 parametric_log_deriv 函数
        from .prde import parametric_log_deriv

        # 计算 n，取 0 和 dc - max(db, da) 中较大的值
        n = max(0, dc - max(db, da))
        # 如果 da 等于 db
        if da == db:
            # 计算 DE.d / Poly(DE.t, DE.t) 的分数形式，并赋值给 etaa, etad
            etaa, etad = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            # 使用 DecrementLevel(DE) 这个上下文管理器
            with DecrementLevel(DE):
                # 将 alpha / DE.t 的分数形式赋值给 alphaa, alphad
                alphaa, alphad = frac_in(alpha, DE.t)
                # 调用 parametric_log_deriv 函数，传入参数 alphaa, alphad, etaa, etad, DE
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                # 如果 A 不为 None
                if A is not None:
                    # 将 A 解包为 a, m, z
                    a, m, z = A
                    # 如果 a 等于 1
                    if a == 1:
                        # 更新 n，取 n 和 m 中较大的值
                        n = max(n, m)

    # 如果 case 是 'tan' 或 'other_nonlinear'
    elif case in ('tan', 'other_nonlinear'):
        # 计算 DE.d 相对于 DE.t 的次数，并赋值给 delta
        delta = DE.d.degree(DE.t)
        # 计算 DE.d 的首项系数，并赋值给 lam
        lam = DE.d.LC()
        # 将 alpha / lam 简化，并赋值给 alpha
        alpha = cancel(alpha/lam)
        # 计算 n，取 0 和 dc - max(da + delta - 1, db) 中较大的值
        n = max(0, dc - max(da + delta - 1, db))
        # 如果 db 等于 da + delta - 1，并且 alpha 是整数
        if db == da + delta - 1 and alpha.is_Integer:
            # 计算 n，取 0 和 alpha, dc - db 中较大的值
            n = max(0, alpha, dc - db)
    else:
        # 如果输入的 case 参数不在指定的选项内，抛出 ValueError 异常
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'other_nonlinear', 'base'}, not %s." % case)

    # 返回计算得到的 n 值
    return n
# 定义了一个函数 spde，实现了 Rothstein 的特殊多项式微分方程算法
def spde(a, b, c, n, DE):
    # 初始化零多项式
    zero = Poly(0, DE.t)

    # 初始化 alpha 为常数多项式 1，beta 为常数多项式 0
    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)

    # 进入主循环
    while True:
        # 如果 c 是零多项式，返回结果元组 (zero, zero, 0, zero, beta)
        if c.is_zero:
            return (zero, zero, 0, zero, beta)  # -1 is more to the point
        # 如果 n 小于 0，抛出 NonElementaryIntegralException 异常
        if (n < 0) is True:
            raise NonElementaryIntegralException

        # 计算 a 和 b 的最大公因数 g
        g = a.gcd(b)
        # 如果 c 对 g 取模不为零，即 g 不整除 c，则抛出 NonElementaryIntegralException 异常
        if not c.rem(g).is_zero:  # g does not divide c
            raise NonElementaryIntegralException

        # 更新 a, b, c 为它们分别对 g 取商的结果
        a, b, c = a.quo(g), b.quo(g), c.quo(g)

        # 如果 a 在 DE.t 变量上的次数为 0
        if a.degree(DE.t) == 0:
            # 将 b 转换为域上的多项式，并对 a 取商，同时对 c 也做类似的处理
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            # 返回结果元组 (b, c, n, alpha, beta)
            return (b, c, n, alpha, beta)

        # 使用 gcdex_diophantine 函数计算 r 和 z
        r, z = gcdex_diophantine(b, a, c)
        # 更新 b 为 b 加上 a 的导数在 DE 上的结果
        b += derivation(a, DE)
        # 更新 c 为 z 减去 r 在 DE 上的导数的结果
        c = z - derivation(r, DE)
        # 减少 n，减去 a 在 DE.t 变量上的次数
        n -= a.degree(DE.t)

        # 更新 beta 为 beta 加上 alpha 乘以 r 的结果
        beta += alpha * r
        # 更新 alpha 为 alpha 乘以 a
        alpha *= a


# 定义了一个函数 no_cancel_b_large，实现了 Poly Risch Differential Equation - No cancellation 的算法，其中 deg(b) 大于等于 deg(D) - 1
def no_cancel_b_large(b, c, n, DE):
    # 初始化 q 为零多项式
    q = Poly(0, DE.t)

    # 进入主循环，直到 c 是零多项式
    while not c.is_zero:
        # 计算 m，即 c 在 DE.t 变量上的次数减去 b 在 DE.t 变量上的次数
        m = c.degree(DE.t) - b.degree(DE.t)
        # 如果 m 不在 0 到 n 之间，抛出 NonElementaryIntegralException 异常
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        # 计算 p，即 c 的首项系数除以 b 的首项系数乘以 DE.t 的 m 次幂，形成多项式
        p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()*DE.t**m, DE.t, expand=False)
        # 更新 q 为 q 加上 p
        q = q + p
        # 更新 n 为 m - 1
        n = m - 1
        # 更新 c 为 c 减去 p 在 DE 上的导数和 b 乘以 p 的结果
        c = c - derivation(p, DE) - b * p

    # 返回最终计算得到的 q
    return q


# 定义了一个函数 no_cancel_b_small，实现了 Poly Risch Differential Equation - No cancellation 的算法，其中 deg(b) 小于 deg(D) - 1
def no_cancel_b_small(b, c, n, DE):
    # 初始化 q 为零多项式
    q = Poly(0, DE.t)

    # 进入主循环，直到 c 是零多项式
    while not c.is_zero:
        # 计算 m，即 c 在 DE.t 变量上的次数减去 b 在 DE.t 变量上的次数
        m = c.degree(DE.t) - b.degree(DE.t)
        # 如果 m 不在 0 到 n 之间，或者 DE.t 的次数小于 2，则抛出 NonElementaryIntegralException 异常
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        # 计算 p，即 c 的首项系数除以 b 的首项系数乘以 DE.t 的 m 次幂，形成多项式
        p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()*DE.t**m, DE.t, expand=False)
        # 更新 q 为 q 加上 p
        q = q + p
        # 更新 n 为 m - 1
        n = m - 1
        # 更新 c 为 c 减去 p 在 DE 上的导数和 b 乘以 p 的结果
        c = c - derivation(p, DE) - b * p

    # 返回最终计算得到的 q
    return q
    """
    Return a polynomial `q` in `k[t]` such that `q` satisfies `Dq + bq == c`,
    where `D` is the derivation, `b` and `c` are polynomials over `k`, and `n`
    is the maximum degree of `q`. Additionally, ensure `y == q - h` satisfies
    `Dy + b0*y == c0` in `k`.
    """
    # Initialize `q` as the zero polynomial in `k[t]`
    q = Poly(0, DE.t)

    # Loop until `c` becomes zero
    while not c.is_zero:
        # Calculate the degree condition `m`
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1

        # Check if the degree condition `m` is within bounds
        if not 0 <= m <= n:
            # Raise exception if `m` is out of bounds
            raise NonElementaryIntegralException

        # Handle the case when `m > 0`
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC() / (m * DE.d.as_poly(DE.t).LC()) * DE.t**m,
                     DE.t, expand=False)
        else:
            # Handle the case when `m == 0`
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                # Return a tuple if degrees are zero
                return (q, b.as_poly(DE.T[DE.level - 1]), c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC(), DE.t, expand=False)

        # Update `q` by adding `p`
        q = q + p
        # Update `n` to `m - 1`
        n = m - 1
        # Update `c` using the derivation function and `b`
        c = c - derivation(p, DE) - b * p

    # Return the computed polynomial `q`
    return q
# TODO: better name for this function
def no_cancel_equal(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with deg(D) >= 2, n either an integer
    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c has
    no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n, or the tuple (h, m, C) such that h
    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of
    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]
    of degree at most m of Dy + b*y == C.
    """
    q = Poly(0, DE.t)  # Initialize q as the zero polynomial in variable DE.t
    lc = cancel(-b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC())
    # Compute lc as the cancellation of leading coefficients of -b and DE.d
    if lc.is_Integer and lc.is_positive:
        M = lc  # Set M to lc if lc is an Integer and positive
    else:
        M = -1  # Otherwise, set M to -1

    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)
        # Compute m as the maximum of M and the difference in degrees plus 1

        if not 0 <= m <= n:  # Check if m is within the valid range [0, n]
            raise NonElementaryIntegralException  # Raise exception if out of range

        u = cancel(m*DE.d.as_poly(DE.t).LC() + b.as_poly(DE.t).LC())
        # Compute u using cancellation of terms involving m and coefficients
        if u.is_zero:
            return (q, m, c)  # Return (q, m, c) if u is zero

        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/u*DE.t**m, DE.t, expand=False)
            # Compute p as a polynomial using cancellation and multiplication
        else:
            if c.degree(DE.t) != DE.d.degree(DE.t) - 1:
                raise NonElementaryIntegralException
                # Raise exception if degrees do not match expected condition
            else:
                p = c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()
                # Compute p as a ratio of leading coefficients

        q = q + p  # Update q by adding p
        n = m - 1  # Update n to m - 1
        c = c - derivation(p, DE) - b*p  # Update c based on derivation and operations

    return q  # Return the final polynomial q


def cancel_primitive(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    # Delayed imports
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)  # Compute fractions of b in terms of DE.t
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        # Check if ba and bd satisfy specific conditions using is_log_deriv_k_t_radical_in_field
        if A is not None:
            n, z = A
            if n == 1:  # If n equals 1, handle specific case (not implemented)
                raise NotImplementedError("is_deriv_in_field() is required to "
                    " solve this problem.")
                # Specific case handling if z*c == Dp for p in k[t] and deg(p) <= n
                #     return p/z
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # Return 0 if c is zero

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException
        # Raise exception if n is less than degree of c in terms of DE.t

    q = Poly(0, DE.t)  # Initialize q as the zero polynomial in variable DE.t
    # 当 c 不为零时执行循环，直到 c 为零
    while not c.is_zero:
        # 计算 c 的最高次数
        m = c.degree(DE.t)
        # 如果 n 小于 m，则抛出非初等积分异常
        if n < m:
            raise NonElementaryIntegralException
        # 在减少积分级别的上下文中执行以下操作
        with DecrementLevel(DE):
            # 计算 c 的首项的分数表示形式
            a2a, a2d = frac_in(c.LC(), DE.t)
            # 使用 Risch 算法求解微分方程 DE 的解
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        # 构造多项式 stm，其值为 sa/sd * DE.t^m
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        # 将 stm 加入到商 q 中
        q += stm
        # 更新 n 的值为 m - 1
        n = m - 1
        # 更新 c 的值为 c - b*stm - derivation(stm, DE)
        c -= b*stm + derivation(stm, DE)

    # 返回求得的商 q
    return q
# 定义一个函数，用于解决多项式Risch微分方程中的取消问题，针对超指数情况

def cancel_exp(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Hyperexponential case.

    Explanation
    ===========

    给定在 k[t] 上的导数 D，其中 n 是整数或 +oo，b 是 k 中的常数，c 是 k[t] 中的多项式，
    且 Dt/t 是 k 中的元素，b != 0。如果没有以最多度数 n 的解 q 满足方程 Dq + b*q == c，
    则引发 NonElementaryIntegralException；否则返回 k[t] 中满足条件的解 q，使得 deg(q) <= n。
    """

    # 导入需要的模块和函数
    from .prde import parametric_log_deriv

    # 计算 eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()
    eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()

    # 使用 DecrementLevel(DE) 上下文管理器
    with DecrementLevel(DE):
        # 将 eta 分数化
        etaa, etad = frac_in(eta, DE.t)
        # 将 b 分数化
        ba, bd = frac_in(b, DE.t)
        # 计算 parametric_log_deriv 函数返回的 A
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)

        # 如果 A 不为 None，则继续处理
        if A is not None:
            a, m, z = A
            # 如果 a == 1，目前尚未实现 is_deriv_in_field() 函数，因此抛出 NotImplementedError
            if a == 1:
                raise NotImplementedError("is_deriv_in_field() is required to solve this problem.")
                # 如果存在 k<t> 中的多项式 p，使得 c*z*t**m == Dp，并且 q = p/(z*t**m) 在 k[t] 中
                # 且 deg(q) <= n，则返回 q；否则引发 NonElementaryIntegralException

    # 如果 c 是零多项式，则直接返回 c（即返回 0）
    if c.is_zero:
        return c

    # 如果 n 小于 c 在 DE.t 上的最高次数，则引发 NonElementaryIntegralException
    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    # 初始化 q 为 k[t] 中的零多项式
    q = Poly(0, DE.t)

    # 当 c 不是零多项式时进行循环处理
    while not c.is_zero:
        # 计算 c 在 DE.t 上的最高次数
        m = c.degree(DE.t)
        # 如果 n 小于 m，则引发 NonElementaryIntegralException
        if n < m:
            raise NonElementaryIntegralException

        # 计算 a1 = b + m*Dt/t
        a1 = b.as_expr()
        with DecrementLevel(DE):
            # TODO: 编写一个虚拟函数，执行这种习惯用法
            # 将 a1 分数化
            a1a, a1d = frac_in(a1, DE.t)
            # 计算 a1a 和 a1d 的乘积
            a1a = a1a*etad + etaa*a1d*Poly(m, DE.t)
            # 更新 a1d
            a1d = a1d*etad

            # 将 c 的首项系数转为分数形式
            a2a, a2d = frac_in(c.LC(), DE.t)

            # 调用 rischDE 函数，计算 sa 和 sd
            sa, sd = rischDE(a1a, a1d, a2a, a2d, DE)

        # 计算 stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        # 更新 q
        q += stm
        # 更新 n
        n = m - 1
        # 更新 c，减去 b*stm + derivation(stm, DE)
        c -= b*stm + derivation(stm, DE)  # deg(c) becomes smaller

    # 返回结果 q
    return q
    # 如果 b 是零或者 b 相对于 DE.t 的阶数小于 DE.d 相对于 DE.t 的阶数减一，并且
    # DE.case 是 'base' 或者 DE.d 相对于 DE.t 的阶数大于等于 2
    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and \
            (DE.case == 'base' or DE.d.degree(DE.t) >= 2):
        
        # 如果 parametric 为真，导入 prde_no_cancel_b_small 并返回其结果
        if parametric:
            from .prde import prde_no_cancel_b_small
            return prde_no_cancel_b_small(b, cQ, n, DE)

        # 否则调用 no_cancel_b_small 并赋值给 R
        R = no_cancel_b_small(b, cQ, n, DE)

        # 如果 R 是 Poly 类型，则直接返回 R
        if isinstance(R, Poly):
            return R
        else:
            # 否则解包 R 到 h, b0, c0，并在 DE 的减少级别下进行处理
            h, b0, c0 = R
            with DecrementLevel(DE):
                # 将 b0, c0 转换为 DE.t 的多项式
                b0, c0 = b0.as_poly(DE.t), c0.as_poly(DE.t)
                # 如果 b0 是 None，则抛出 ValueError 异常
                if b0 is None:  # See above comment
                    raise ValueError("b0 should be a non-Null value")
                # 如果 c0 是 None，则抛出 ValueError 异常
                if c0 is None:
                    raise ValueError("c0 should be a non-Null value")
                # 解决多项式 RDE b0, c0, n, DE，并将结果转换为 DE.t 的多项式 y
                y = solve_poly_rde(b0, c0, n, DE).as_poly(DE.t)
            # 返回 h + y
            return h + y

    # 如果 DE.d 相对于 DE.t 的阶数大于等于 2，并且 b 相对于 DE.t 的阶数等于 DE.d 相对于 DE.t 的阶数减一，并且
    # n 大于 -b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC()
    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and \
            n > -b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC():

        # TODO: 这个检查是否必要，如果必要，失败时应该执行什么操作？
        # b 来自 spde() 返回的第一个元素
        if not b.as_poly(DE.t).LC().is_number:
            raise TypeError("Result should be a number")

        # 如果 parametric 为真，抛出 NotImplementedError
        if parametric:
            raise NotImplementedError("prde_no_cancel_b_equal() is not yet "
                "implemented.")

        # 否则调用 no_cancel_equal 并赋值给 R
        R = no_cancel_equal(b, cQ, n, DE)

        # 如果 R 是 Poly 类型，则直接返回 R
        if isinstance(R, Poly):
            return R
        else:
            # 否则解包 R 到 h, m, C
            h, m, C = R
            # 解决多项式 RDE b, C, m, DE，并将结果赋给 y
            y = solve_poly_rde(b, C, m, DE)
            # 返回 h + y
            return h + y

    else:
        # 取消
        # 如果 b 是零，抛出 NotImplementedError 异常
        if b.is_zero:
            raise NotImplementedError("Remaining cases for Poly (P)RDE are "
            "not yet implemented (is_deriv_in_field() required).")
        else:
            # 如果 DE.case 是 'exp'
            if DE.case == 'exp':
                # 如果 parametric 为真，抛出 NotImplementedError
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "hyperexponential case is not yet implemented.")
                # 否则调用 cancel_exp 并返回结果
                return cancel_exp(b, cQ, n, DE)

            # 如果 DE.case 是 'primitive'
            elif DE.case == 'primitive':
                # 如果 parametric 为真，抛出 NotImplementedError
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "primitive case is not yet implemented.")
                # 否则调用 cancel_primitive 并返回结果
                return cancel_primitive(b, cQ, n, DE)

            # 其他情况抛出 NotImplementedError 异常
            else:
                raise NotImplementedError("Other Poly (P)RDE cancellation "
                    "cases are not yet implemented (%s)." % DE.case)

        # 如果 parametric 为真，抛出 NotImplementedError
        if parametric:
            raise NotImplementedError("Remaining cases for Poly PRDE not yet "
                "implemented.")
        # 抛出 NotImplementedError
        raise NotImplementedError("Remaining cases for Poly RDE not yet "
            "implemented.")
# 定义一个函数，解决 Risch 微分方程：Dy + f*y == g。
def rischDE(fa, fd, ga, gd, DE):
    """
    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.
    """

    # 对输入的 f 和 f' 进行弱归一化处理
    _, (fa, fd) = weak_normalizer(fa, fd, DE)
    
    # 计算 fa 和 fd 的正常分母以及其他参数
    a, (ba, bd), (ca, cd), hn = normal_denom(fa, fd, ga, gd, DE)
    
    # 计算特殊分母 A, B, C 以及 hs
    A, B, C, hs = special_denom(a, ba, bd, ca, cd, DE)
    
    try:
        # 计算 A, B, C 的度的上界
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        # 如果计算上界不可行，则设置 n 为无穷大
        # 用于调试信息：
        # import warnings
        # warnings.warn("rischDE: Proceeding with n = oo; may cause "
        #     "non-termination.")
        n = oo

    # 解特殊分母方程 spde(A, B, C, n, DE)，返回 B, C, m, alpha, beta
    B, C, m, alpha, beta = spde(A, B, C, n, DE)
    
    if C.is_zero:
        # 如果 C 为零，则 y = C
        y = C
    else:
        # 否则，解多项式 RDE 方程 solve_poly_rde(B, C, m, DE)，得到 y
        y = solve_poly_rde(B, C, m, DE)

    # 返回解的形式为 (alpha*y + beta, hn*hs)
    return (alpha*y + beta, hn*hs)
```