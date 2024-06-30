# `D:\src\scipysrc\sympy\sympy\integrals\rationaltools.py`

```
"""This module implements tools for integrating rational functions. """

from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ


def ratint(f, x, **flags):
    """
    Performs indefinite integration of rational functions.

    Explanation
    ===========

    Given a field :math:`K` and a rational function :math:`f = p/q`,
    where :math:`p` and :math:`q` are polynomials in :math:`K[x]`,
    returns a function :math:`g` such that :math:`f = g'`.

    Examples
    ========

    >>> from sympy.integrals.rationaltools import ratint
    >>> from sympy.abc import x

    >>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)
    (12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)

    References
    ==========

    .. [1] M. Bronstein, Symbolic Integration I: Transcendental
       Functions, Second Edition, Springer-Verlag, 2005, pp. 35-70

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.rationaltools.ratint_logpart
    sympy.integrals.rationaltools.ratint_ratpart

    """
    # Check if f is given as a tuple (p, q); if not, extract numerator and denominator
    if isinstance(f, tuple):
        p, q = f
    else:
        p, q = f.as_numer_denom()

    # Create polynomial objects for numerator p and denominator q
    p, q = Poly(p, x, composite=False, field=True), Poly(q, x, composite=False, field=True)

    # Cancel common factors in p and q and retrieve the quotient and remainder
    coeff, p, q = p.cancel(q)
    poly, p = p.div(q)

    # Integrate the resulting polynomial poly with respect to x
    result = poly.integrate(x).as_expr()

    # If p is zero after division, return the integrated result multiplied by coeff
    if p.is_zero:
        return coeff*result

    # Decompose the rational part p/q into g and h such that p = g*q + h
    g, h = ratint_ratpart(p, q, x)

    # Separate h into numerator P and denominator Q
    P, Q = h.as_numer_denom()

    # Convert P and Q to polynomial objects
    P = Poly(P, x)
    Q = Poly(Q, x)

    # Divide P by Q to obtain quotient q and remainder r
    q, r = P.div(Q)

    # Integrate q with respect to x and add to the overall result
    result += g + q.integrate(x).as_expr()
    # 如果 r 不为零，则执行以下代码块
    if not r.is_zero:
        # 从 flags 字典获取 'symbol' 键对应的值，如果不存在则默认为 't'
        symbol = flags.get('symbol', 't')

        # 如果 symbol 不是 Symbol 类的实例，则创建一个 Dummy 对象 t
        if not isinstance(symbol, Symbol):
            t = Dummy(symbol)
        else:
            # 否则将 symbol 转换为 Dummy 对象并赋值给 t
            t = symbol.as_dummy()

        # 调用 ratint_logpart 函数计算 r 关于 Q, x, t 的对数部分，返回结果赋值给 L
        L = ratint_logpart(r, Q, x, t)

        # 从 flags 字典获取 'real' 键对应的值
        real = flags.get('real')

        # 如果 real 为 None，则根据 f 的类型判断 atoms 集合是否为扩展实数集
        if real is None:
            if isinstance(f, tuple):
                p, q = f
                atoms = p.atoms() | q.atoms()
            else:
                atoms = f.atoms()

            # 遍历 atoms 中的元素，检查是否除了 x 外都是扩展实数
            for elt in atoms - {x}:
                if not elt.is_extended_real:
                    real = False
                    break
            else:
                # 如果所有元素都是扩展实数，则 real 设为 True
                real = True

        # 初始化 eps 为零
        eps = S.Zero

        # 如果 real 为 False，则执行以下代码块
        if not real:
            # 遍历 L 中的每对 (h, q)，计算 h 的原始函数并将结果加到 eps 上
            for h, q in L:
                _, h = h.primitive()
                eps += RootSum(
                    q, Lambda(t, t*log(h.as_expr())), quadratic=True)
        else:
            # 否则，遍历 L 中的每对 (h, q)，尝试将 h 转换为实数并将结果加到 eps 上
            for h, q in L:
                _, h = h.primitive()
                R = log_to_real(h, q, x, t)

                # 如果转换成功，则将结果加到 eps 上；否则计算 h 的原始函数并将结果加到 eps 上
                if R is not None:
                    eps += R
                else:
                    eps += RootSum(
                        q, Lambda(t, t*log(h.as_expr())), quadratic=True)

        # 将 eps 加到 result 上
        result += eps

    # 返回 coeff 乘以 result 的结果
    return coeff*result
def ratint_logpart(f, g, x, t=None):
    """
    Lazard-Rioboo-Trager algorithm.

    Explanation
    ===========

    Given a field K and polynomials f and g in K[x], such that f and g
    are coprime, deg(f) < deg(g) and g is square-free, returns a list
    of tuples (s_i, q_i) of polynomials, for i = 1..n, such that s_i
    in K[t, x] and q_i in K[t], and::

                           ___    ___
                 d  f   d  \  `   \  `
                 -- - = --  )      )   a log(s_i(a, x))
                 dx g   dx /__,   /__,
                          i=1..n a | q_i(a) = 0

    Examples
    ========

    >>> from sympy.integrals.rationaltools import ratint_logpart
    >>> from sympy.abc import x
    >>> from sympy import Poly
    >>> ratint_logpart(Poly(1, x, domain='ZZ'),
    ... Poly(x**2 + x + 1, x, domain='ZZ'), x)
    [(Poly(x + 3*_t/2 + 1/2, x, domain='QQ[_t]'),
    ...Poly(3*_t**2 + 1, _t, domain='ZZ'))]
    >>> ratint_logpart(Poly(12, x, domain='ZZ'),
    ... Poly(x**2 - x - 2, x, domain='ZZ'), x)
    [(Poly(x - 3*_t/8 - 1/2, x, domain='QQ[_t]'),
    ...Poly(-_t**2 + 16, _t, domain='ZZ'))]

    See Also
    ========

    ratint, ratint_ratpart
    """
    # 将输入的 f 和 g 转换为多项式
    f, g = Poly(f, x), Poly(g, x)

    # 如果未提供 t 的值，则使用一个虚拟变量作为 t
    t = t or Dummy('t')

    # 定义 b 为 g 和 f 的导数的乘积
    a, b = g, f - g.diff()*Poly(t, x)
    # 调用 resultant 函数计算多项式 a 和 b 的结果多项式及剩余集合 R
    res, R = resultant(a, b, includePRS=True)
    # 将结果多项式 res 转换为 Poly 类型的对象，使用变量 t，不合并同类项
    res = Poly(res, t, composite=False)

    # 检查结果多项式是否非零，如果是零则抛出断言异常
    assert res, "BUG: resultant(%s, %s) cannot be zero" % (a, b)

    # 初始化 R_map 为一个空字典，H 为一个空列表
    R_map, H = {}, []

    # 遍历剩余集合 R 中的多项式 r，将 r 的次数作为键存入 R_map 字典
    for r in R:
        R_map[r.degree()] = r

    # 定义内部函数 _include_sign，用于处理扩展实数域中的符号问题
    def _include_sign(c, sqf):
        if c.is_extended_real and (c < 0) == True:
            h, k = sqf[0]
            c_poly = c.as_poly(h.gens)
            sqf[0] = h*c_poly, k

    # 对结果多项式 res 进行平方因式分解，得到基本的单位部分 C 和平方因式列表 res_sqf
    C, res_sqf = res.sqf_list()
    # 调用 _include_sign 处理 C 和 res_sqf 中的符号问题
    _include_sign(C, res_sqf)

    # 遍历平方因式列表 res_sqf 中的每对 (q, i)
    for q, i in res_sqf:
        # 对 q 进行原始多项式的原始部分处理
        _, q = q.primitive()

        # 如果 g 的次数与 i 相等，则将 (g, q) 加入列表 H 中；否则：
        if g.degree() == i:
            H.append((g, q))
        else:
            # 从 R_map 中取出次数为 i 的多项式 h
            h = R_map[i]
            # 计算 h 的首项系数
            h_lc = Poly(h.LC(), t, field=True)

            # 对 h_lc 进行全平方因式分解，得到系数 c 和平方因式列表 h_lc_sqf
            c, h_lc_sqf = h_lc.sqf_list(all=True)
            # 调用 _include_sign 处理 c 和 h_lc_sqf 中的符号问题
            _include_sign(c, h_lc_sqf)

            # 遍历 h_lc_sqf 中的每对 (a, j)
            for a, j in h_lc_sqf:
                # 将 h 除以 Poly(a.gcd(q)**j, x) 的结果赋给 h
                h = h.quo(Poly(a.gcd(q)**j, x))

            # 计算 h_lc 关于 q 的乘法逆元 inv，并初始化 coeffs 列表
            inv, coeffs = h_lc.invert(q), [S.One]

            # 遍历 h 的系数列表，计算并添加每个系数的 T 值到 coeffs 列表中
            for coeff in h.coeffs()[1:]:
                coeff = coeff.as_poly(inv.gens)
                T = (inv*coeff).rem(q)
                coeffs.append(T.as_expr())

            # 使用 coeffs 列表重新构造 h 多项式，并加入到列表 H 中
            h = Poly(dict(list(zip(h.monoms(), coeffs))), x)
            H.append((h, q))

    # 返回结果列表 H
    return H
# 将复数对数转换为实数反正切

def log_to_atan(f, g):
    """
    Convert complex logarithms to real arctangents.

    Explanation
    ===========
    
    给定一个实数域 K 和 K[x] 中的多项式 f 和 g，其中 g ≠ 0，
    返回 K[x] 中多项式 arctan 的和 h，满足以下条件：

                   dh   d         f + I g
                   -- = -- I log( ------- )
                   dx   dx        f - I g

    Examples
    ========

        >>> from sympy.integrals.rationaltools import log_to_atan
        >>> from sympy.abc import x
        >>> from sympy import Poly, sqrt, S
        >>> log_to_atan(Poly(x, x, domain='ZZ'), Poly(1, x, domain='ZZ'))
        2*atan(x)
        >>> log_to_atan(Poly(x + S(1)/2, x, domain='QQ'),
        ... Poly(sqrt(3)/2, x, domain='EX'))
        2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)

    See Also
    ========

    log_to_real
    """
    # 如果 f 的次数小于 g 的次数，则交换 f 和 g
    if f.degree() < g.degree():
        f, g = -g, f

    # 将 f 和 g 转换为域上的多项式
    f = f.to_field()
    g = g.to_field()

    # 计算 f 除以 g 的商和余数
    p, q = f.div(g)

    # 如果余数 q 为零，则返回 2*atan(p.as_expr())
    if q.is_zero:
        return 2*atan(p.as_expr())
    else:
        # 否则，计算 g 和 -f 的最大公因式，以及商 u
        s, t, h = g.gcdex(-f)
        u = (f*s + g*t).quo(h)
        # 计算 2*atan(u.as_expr())
        A = 2*atan(u.as_expr())

        # 返回 A + log_to_atan(s, t)
        return A + log_to_atan(s, t)


def log_to_real(h, q, x, t):
    r"""
    Convert complex logarithms to real functions.

    Explanation
    ===========

    给定实数域 K 和 K[t,x] 中的多项式 h，以及 K[t] 中的多项式 q，
    返回实函数 f，满足以下条件：

                          ___
                  df   d  \  `
                  -- = --  )  a log(h(a, x))
                  dx   dx /__,
                         a | q(a) = 0

    Examples
    ========

        >>> from sympy.integrals.rationaltools import log_to_real
        >>> from sympy.abc import x, y
        >>> from sympy import Poly, S
        >>> log_to_real(Poly(x + 3*y/2 + S(1)/2, x, domain='QQ[y]'),
        ... Poly(3*y**2 + 1, y, domain='ZZ'), x, y)
        2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
        >>> log_to_real(Poly(x**2 - 1, x, domain='ZZ'),
        ... Poly(-2*y + 1, y, domain='ZZ'), x, y)
        log(x**2 - 1)/2

    See Also
    ========

    log_to_atan
    """
    from sympy.simplify.radsimp import collect
    u, v = symbols('u,v', cls=Dummy)

    # 将 h 和 q 中的 t 替换为 u + I*v，并展开表达式
    H = h.as_expr().xreplace({t: u + I*v}).expand()
    Q = q.as_expr().xreplace({t: u + I*v}).expand()

    # 收集 H 和 Q 中关于 I 的项
    H_map = collect(H, I, evaluate=False)
    Q_map = collect(Q, I, evaluate=False)

    # 获取 H 和 Q 中 1 和 I 的系数
    a, b = H_map.get(S.One, S.Zero), H_map.get(I, S.Zero)
    c, d = Q_map.get(S.One, S.Zero), Q_map.get(I, S.Zero)

    # 计算 Q(a) = 0 的结果
    R = Poly(resultant(c, d, v), u)

    # 对 R 求根，过滤得到实数解
    R_u = roots(R, filter='R')

    # 如果根的数量不等于 R 的所有根数，则返回 None
    if len(R_u) != R.count_roots():
        return None

    result = S.Zero
    # 遍历字典 R_u 的键值
    for r_u in R_u.keys():
        # 使用 r_u 替换多项式 c 中的变量 u，并生成关于 v 的多项式 C
        C = Poly(c.xreplace({u: r_u}), v)
        # 如果 C 为零多项式
        if not C:
            # 提示：t 被分割成实部和虚部，并且分母 Q(u, v) = c + I*d。我们刚刚发现 c(r_u) 等于 0，所以根在虚部 d 中。
            # 使用 r_u 替换多项式 d 中的变量 u，得到多项式 C
            C = Poly(d.xreplace({u: r_u}), v)
            # 提示：我们原本打算拒绝不使 d 等于零的 C 的根，但是因为现在我们使用了 C = d 而且 c 已经是零，所以不需要检查任何内容。
            # 将 d 设为零
            d = S.Zero

        # 求解多项式 C 的实数根 R_v
        R_v = roots(C, filter='R')

        # 如果实数根的数量不等于 C 的根的总数，返回空
        if len(R_v) != C.count_roots():
            return None

        # 初始化用于存储一对共轭根中的一个根的列表
        R_v_paired = []
        # 遍历实数根 R_v
        for r_v in R_v:
            # 如果 r_v 和其相反数不在 R_v_paired 中
            if r_v not in R_v_paired and -r_v not in R_v_paired:
                # 如果 r_v 是负数或者可能提取负号
                if r_v.is_negative or r_v.could_extract_minus_sign():
                    # 添加 -r_v 到 R_v_paired 中
                    R_v_paired.append(-r_v)
                # 否则如果 r_v 不为零
                elif not r_v.is_zero:
                    # 添加 r_v 到 R_v_paired 中
                    R_v_paired.append(r_v)

        # 遍历 R_v_paired 中的每个根 r_v
        for r_v in R_v_paired:
            # 使用 r_u 和 r_v 替换多项式 d 中的变量 u 和 v，生成多项式 D
            D = d.xreplace({u: r_u, v: r_v})

            # 如果 D 在数值评估后不等于零，继续下一个循环
            if D.evalf(chop=True) != 0:
                continue

            # 使用 r_u 和 r_v 替换多项式 a 中的变量 u 和 v，生成多项式 A
            A = Poly(a.xreplace({u: r_u, v: r_v}), x)
            # 使用 r_u 和 r_v 替换多项式 b 中的变量 u 和 v，生成多项式 B
            B = Poly(b.xreplace({u: r_u, v: r_v}), x)

            # 计算 A^2 + B^2，并转换为表达式 AB
            AB = (A**2 + B**2).as_expr()

            # 将 r_u*log(AB) + r_v*log_to_atan(A, B) 添加到结果中
            result += r_u*log(AB) + r_v*log_to_atan(A, B)

    # 求解多项式 q 的实数根 R_q
    R_q = roots(q, filter='R')

    # 如果实数根的数量不等于 q 的根的总数，返回空
    if len(R_q) != q.count_roots():
        return None

    # 遍历字典 R_q 的键值
    for r in R_q.keys():
        # 将 r*log(h.as_expr().subs(t, r)) 添加到结果中
        result += r*log(h.as_expr().subs(t, r))

    # 返回最终的结果
    return result
```