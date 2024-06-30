# `D:\src\scipysrc\sympy\sympy\polys\groebnertools.py`

```
# 导入所需模块和函数
from sympy.core.symbol import Dummy  # 导入 Dummy 符号
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div  # 导入单项式操作函数
from sympy.polys.orderings import lex  # 导入词典序排列函数
from sympy.polys.polyerrors import DomainError  # 导入域错误异常
from sympy.polys.polyconfig import query  # 导入配置查询函数

# 定义函数 groebner
def groebner(seq, ring, method=None):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Wrapper around the (default) improved Buchberger and the other algorithms
    for computing Groebner bases. The choice of algorithm can be changed via
    ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where
    ``method`` can be either ``buchberger`` or ``f5b``.

    """
    # 如果未指定算法方法，则使用配置中的默认方法
    if method is None:
        method = query('groebner')

    # 定义不同 Groebner 算法对应的函数
    _groebner_methods = {
        'buchberger': _buchberger,
        'f5b': _f5b,
    }

    # 尝试根据指定的方法选择相应的 Groebner 算法函数
    try:
        _groebner = _groebner_methods[method]
    except KeyError:
        # 如果指定的方法不在允许的列表中，则抛出值错误异常
        raise ValueError("'%s' is not a valid Groebner bases algorithm (valid are 'buchberger' and 'f5b')" % method)

    # 获取环的域和原始环
    domain, orig = ring.domain, None

    # 如果环不是域或者没有关联域，尝试转换为域
    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, ring = ring, ring.clone(domain=domain.get_field())
        except DomainError:
            # 如果转换失败，抛出域错误异常
            raise DomainError("Cannot compute a Groebner basis over %s" % domain)
        else:
            # 对于序列中的每个多项式，设置新的环
            seq = [ s.set_ring(ring) for s in seq ]

    # 使用选定的 Groebner 算法计算 Groebner 基
    G = _groebner(seq, ring)

    # 如果有原始环，则将结果还原到原始环
    if orig is not None:
        G = [ g.clear_denoms()[1].set_ring(orig) for g in G ]

    # 返回计算得到的 Groebner 基
    return G

# 定义函数 _buchberger
def _buchberger(f, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Given a set of multivariate polynomials `F`, finds another
    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced
    Groebner basis.

    The resulting basis is unique and has monic generators if the
    ground domains is a field. Otherwise the result is non-unique
    but Groebner bases over e.g. integers can be computed (if the
    input polynomials are monic).

    Groebner bases can be used to choose specific generators for a
    polynomial ideal. Because these bases are unique you can check
    for ideal equality by comparing the Groebner bases.  To see if
    one polynomial lies in an ideal, divide by the elements in the
    base and see if the remainder vanishes.

    They can also be used to solve systems of polynomial equations
    as,  by choosing lexicographic ordering,  you can eliminate one
    variable at a time, provided that the ideal is zero-dimensional
    (finite number of solutions).

    Notes
    =====

    Algorithm used: an improved version of Buchberger's algorithm
    as presented in T. Becker, V. Weispfenning, Groebner Bases: A
    Computational Approach to Commutative Algebra, Springer, 1993,
    page 232.

    References
    ==========

    .. [1] [Bose03]_
    .. [2] [Giovini91]_
    .. [3] [Ajwa95]_
    .. [4] [Cox97]_

    """
    # 获取环的排序方式
    order = ring.order

    # 多项式乘法函数
    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm



    def select(P):
        # 普通的选择策略
        # 选择具有最小 LCM(LM(f), LM(g)) 的一对
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        return pr



    def normal(g, J):
        # 标准化函数，用于计算 g 除以 f[J] 的余数 h
        h = g.rem([ f[j] for j in J ])

        if not h:
            return None
        else:
            h = h.monic()

            if h not in I:
                I[h] = len(f)
                f.append(h)

            return h.LM, I[h]



    def update(G, B, ih):
        # 使用临界对集合 B 和 h 更新 G
        # [BW] 第 230 页
        h = f[ih]
        mh = h.LM

        # 过滤新的对 (h, g)，其中 g 属于 G
        C = G.copy()
        D = set()

        while C:
            # 从 C 中弹出一个元素选择一对 (h, g)
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                # LCM(LM(h), LM(p)) 能整除 LCM(LM(h), LM(g))
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)

            # HT(h) 和 HT(g) 互不相交：mh*mg == LCMhg
            if monomial_mul(mh, mg) == LCMhg or (
                not any(lcm_divides(ipx) for ipx in C) and
                    not any(lcm_divides(pr[1]) for pr in D)):
                D.add((ih, ig))

        E = set()

        while D:
            # 从 D 中选择 h, g (h 与上述相同)
            ih, ig = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)

            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))

        # 过滤旧的对
        B_new = set()

        while B:
            # 从 B 中选择 g1, g2 (-> CP)
            ig1, ig2 = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)

            # 如果 HT(h) 不整除 lcm(HT(g1), HT(g2))
            if not monomial_div(LCM12, mh) or \
                monomial_lcm(mg1, mh) == LCM12 or \
                    monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))

        B_new |= E

        # 过滤多项式
        G_new = set()

        while G:
            ig = G.pop()
            mg = f[ig].LM

            if not monomial_div(mg, mh):
                G_new.add(ig)

        G_new.add(ih)

        return G_new, B_new
        # update 结束 ################################

    if not f:
        return []

    # 用初始多项式的简化列表替换 f；参见 [BW] 第 203 页
    f1 = f[:]

    while True:
        f = f1[:]
        f1 = []

        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])

            if r:
                f1.append(r.monic())

        if f == f1:
            break

    I = {}            # ip = I[p]; p = f[ip]
    F = set()         # 多项式索引集合
    G = set()         # 初始化空集合 G，用于存储中间可能成为格罗布纳基的索引
    CP = set()        # 初始化空集合 CP，用于存储临界对的索引对

    for i, h in enumerate(f):
        I[h] = i     # 将 h 对应的索引 i 存储在字典 I 中
        F.add(i)     # 将索引 i 添加到集合 F 中

    #####################################
    # 算法 GROEBNERNEWS2 参见 [BW] 页码 232

    while F:
        # 根据单项式顺序选择具有最小单项式的 p
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]    # 获取 h 对应的索引 ih
        F.remove(ih)  # 从集合 F 中移除索引 ih
        G, CP = update(G, CP, ih)  # 调用 update 函数更新集合 G 和 CP

    # 统计可约化为零的临界对数目
    reductions_to_zero = 0

    while CP:
        ig1, ig2 = select(CP)  # 从 CP 中选择一对临界对的索引 ig1 和 ig2
        CP.remove((ig1, ig2))  # 从集合 CP 中移除该临界对

        h = spoly(f[ig1], f[ig2], ring)  # 计算 f[ig1] 和 f[ig2] 的 S-多项式 h
        G1 = sorted(G, key=lambda g: order(f[g].LM))  # 根据单项式顺序对 G 进行排序得到 G1
        ht = normal(h, G1)  # 使用 G1 对 h 进行规范化得到 ht

        if ht:
            G, CP = update(G, CP, ht[1])  # 如果 ht 存在，则更新集合 G 和 CP
        else:
            reductions_to_zero += 1  # 否则，将 reductions_to_zero 计数加一

    ######################################
    # 现在 G 是格罗布纳基；对其进行约化
    Gr = set()

    for ig in G:
        ht = normal(f[ig], G - {ig})  # 使用 G - {ig} 对 f[ig] 进行规范化得到 ht

        if ht:
            Gr.add(ht[1])  # 如果 ht 存在，则将其添加到集合 Gr 中

    Gr = [f[ig] for ig in Gr]  # 将集合 Gr 转换为列表

    # 根据单项式顺序对 Gr 进行降序排序
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)

    return Gr  # 返回排序后的 Gr
# 计算 S-多项式，假设 p1 和 p2 是首项为 1 的多项式
def spoly(p1, p2, ring):
    # 获取 p1 和 p2 的首项
    LM1 = p1.LM
    LM2 = p2.LM
    # 计算 p1 和 p2 的最小公倍数
    LCM12 = ring.monomial_lcm(LM1, LM2)
    # 计算 p1 和 p2 的系数的商，得到 m1 和 m2
    m1 = ring.monomial_div(LCM12, LM1)
    m2 = ring.monomial_div(LCM12, LM2)
    # 计算 S-多项式的两部分
    s1 = p1.mul_monom(m1)
    s2 = p2.mul_monom(m2)
    # 计算并返回 S-多项式
    s = s1 - s2
    return s

# F5B

# 便捷函数

# 返回给定元组的第一个元素
def Sign(f):
    return f[0]

# 返回给定元组的第二个元素
def Polyn(f):
    return f[1]

# 返回给定元组的第三个元素
def Num(f):
    return f[2]

# 创建一个带有给定单项式和索引的签名元组
def sig(monomial, index):
    return (monomial, index)

# 创建一个带有给定签名、多项式和数字的带标签的多项式元组
def lbp(signature, polynomial, number):
    return (signature, polynomial, number)

# 签名函数

# 按照给定的序列比较两个签名元组 u 和 v
def sig_cmp(u, v, order):
    """
    Compare two signatures by extending the term order to K[X]^n.

    u < v iff
        - the index of v is greater than the index of u
    or
        - the index of v is equal to the index of u and u[0] < v[0] w.r.t. order

    u > v otherwise
    """
    if u[1] > v[1]:
        return -1
    if u[1] == v[1]:
        if order(u[0]) < order(v[0]):
            return -1
    return 1

# 创建一个用于比较两个签名的键
def sig_key(s, order):
    """
    Key for comparing two signatures.

    s = (m, k), t = (n, l)

    s < t iff [k > l] or [k == l and m < n]
    s > t otherwise
    """
    return (-s[1], order(s[0]))

# 将签名乘以单项式 m
def sig_mult(s, m):
    """
    Multiply a signature by a monomial.

    The product of a signature (m, i) and a monomial n is defined as
    (m * t, i).
    """
    return sig(monomial_mul(s[0], m), s[1])

# 带标签的多项式函数

# 从 f 中减去带标签的多项式 g
def lbp_sub(f, g):
    """
    Subtract labeled polynomial g from f.

    The signature and number of the difference of f and g are signature
    and number of the maximum of f and g, w.r.t. lbp_cmp.
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) < 0:
        max_poly = g
    else:
        max_poly = f

    ret = Polyn(f) - Polyn(g)

    return lbp(Sign(max_poly), ret, Num(max_poly))

# 将带标签的多项式 f 乘以项 cx
def lbp_mul_term(f, cx):
    """
    Multiply a labeled polynomial with a term.

    The product of a labeled polynomial (s, p, k) by a monomial is
    defined as (m * s, m * p, k).
    """
    return lbp(sig_mult(Sign(f), cx[0]), Polyn(f).mul_term(cx), Num(f))

# 比较两个带标签的多项式
def lbp_cmp(f, g):
    """
    Compare two labeled polynomials.

    f < g iff
        - Sign(f) < Sign(g)
    or
        - Sign(f) == Sign(g) and Num(f) > Num(g)

    f > g otherwise
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) == -1:
        return -1
    if Sign(f) == Sign(g):
        if Num(f) > Num(g):
            return -1
    return 1

# 创建一个用于比较带标签的多项式的键
def lbp_key(f):
    """
    Key for comparing two labeled polynomials.
    """
    return (sig_key(Sign(f), Polyn(f).ring.order), -Num(f))

# 算法和辅助函数

# 计算两个带标签的多项式对应的临界对
def critical_pair(f, g, ring):
    """
    Compute the critical pair corresponding to two labeled polynomials.
    
    # 一个关键对是一个元组 (um, f, vm, g)，其中 um 和 vm 是项，使得 um * f - vm * g 是 f 和 g 的 S-多项式（所以假设 um * f > vm * g）。
    # 为了性能考虑，一个关键对被表示为元组 (Sign(um * f), um, f, Sign(vm * g), vm, g)，因为 um * f 会创建一个新的相对昂贵的内存对象，而 Sign(um * f) 和 um 是轻量级的，而 f（在元组中）是对内存中已存在的对象的引用。
    """
    # 获取环的域
    domain = ring.domain
    
    # 获取多项式 f 的领先项
    ltf = Polyn(f).LT
    # 获取多项式 g 的领先项
    ltg = Polyn(g).LT
    # 计算领先项的最小公倍数和单位元，构成元组 lt
    lt = (monomial_lcm(ltf[0], ltg[0]), domain.one)
    
    # 使用 lt 和 ltf 计算 um，这里 domain 是一个环
    um = term_div(lt, ltf, domain)
    # 使用 lt 和 ltg 计算 vm，这里 domain 是一个环
    vm = term_div(lt, ltg, domain)
    
    # 由于不需要完整的信息（现在），因此只考虑与领先项的乘积：
    # 计算 fr，包括 Sign(f)，Polyn(f).leading_term() 和 Num(f)
    fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    # 计算 gr，包括 Sign(g)，Polyn(g).leading_term() 和 Num(g)
    gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)
    
    # 返回适当的顺序，使得 S-多项式只需 u_first * f_first - u_second * f_second：
    if lbp_cmp(fr, gr) == -1:
        return (Sign(gr), vm, g, Sign(fr), um, f)
    else:
        return (Sign(fr), um, f, Sign(gr), vm, g)
def cp_cmp(c, d):
    """
    Compare two critical pairs c and d.

    c < d iff
        - lbp(c[0], _, Num(c[2])) < lbp(d[0], _, Num(d[2])) (this
        corresponds to um_c * f_c and um_d * f_d)
    or
        - lbp(c[0], _, Num(c[2])) == lbp(d[0], _, Num(d[2])) and
        lbp(c[3], _, Num(c[5])) < lbp(d[3], _, Num(d[5])) (this
        corresponds to vm_c * g_c and vm_d * g_d)

    c > d otherwise
    """
    zero = Polyn(c[2]).ring.zero  # 初始化一个零多项式

    c0 = lbp(c[0], zero, Num(c[2]))  # 计算 c 的第一个元素的最低主单项
    d0 = lbp(d[0], zero, Num(d[2]))  # 计算 d 的第一个元素的最低主单项

    r = lbp_cmp(c0, d0)  # 比较 c0 和 d0 的大小

    if r == -1:  # 如果 c0 < d0
        return -1  # 返回 -1 表示 c < d
    if r == 0:  # 如果 c0 == d0
        c1 = lbp(c[3], zero, Num(c[5]))  # 计算 c 的第二个元素的最低主单项
        d1 = lbp(d[3], zero, Num(d[5]))  # 计算 d 的第二个元素的最低主单项

        r = lbp_cmp(c1, d1)  # 比较 c1 和 d1 的大小

        if r == -1:  # 如果 c1 < d1
            return -1  # 返回 -1 表示 c < d
    return 1  # 默认情况返回 1，表示 c > d


def cp_key(c, ring):
    """
    Key for comparing critical pairs.
    """
    return (lbp_key(lbp(c[0], ring.zero, Num(c[2]))), lbp_key(lbp(c[3], ring.zero, Num(c[5]))))
    # 返回一个元组，包含两个用于比较关键对的键值


def s_poly(cp):
    """
    Compute the S-polynomial of a critical pair.

    The S-polynomial of a critical pair cp is cp[1] * cp[2] - cp[4] * cp[5].
    """
    return lbp_sub(lbp_mul_term(cp[2], cp[1]), lbp_mul_term(cp[5], cp[4]))
    # 计算关键对 cp 的 S-多项式，即 cp[1] * cp[2] - cp[4] * cp[5]


def is_rewritable_or_comparable(sign, num, B):
    """
    Check if a labeled polynomial is redundant by checking if its
    signature and number imply rewritability or comparability.

    (sign, num) is comparable if there exists a labeled polynomial
    h in B, such that sign[1] (the index) is less than Sign(h)[1]
    and sign[0] is divisible by the leading monomial of h.

    (sign, num) is rewritable if there exists a labeled polynomial
    h in B, such that sign[1] is equal to Sign(h)[1], num < Num(h)
    and sign[0] is divisible by Sign(h)[0].
    """
    for h in B:
        # comparable
        if sign[1] < Sign(h)[1]:  # 如果 sign[1] < Sign(h)[1]
            if monomial_divides(Polyn(h).LM, sign[0]):  # 如果 sign[0] 可以被 h 的领先单项整除
                return True  # 返回 True，说明可比较

        # rewritable
        if sign[1] == Sign(h)[1]:  # 如果 sign[1] == Sign(h)[1]
            if num < Num(h):  # 如果 num < Num(h)
                if monomial_divides(Sign(h)[0], sign[0]):  # 如果 sign[0] 可以被 h 的符号整除
                    return True  # 返回 True，说明可重写
    return False  # 默认返回 False，说明既不可重写也不可比较


def f5_reduce(f, B):
    """
    F5-reduce a labeled polynomial f by B.

    Continuously searches for non-zero labeled polynomial h in B, such
    that the leading term lt_h of h divides the leading term lt_f of
    f and Sign(lt_h * h) < Sign(f). If such a labeled polynomial h is
    found, f gets replaced by f - lt_f / lt_h * h. If no such h can be
    found or f is 0, f is no further F5-reducible and f gets returned.

    A polynomial that is reducible in the usual sense need not be
    F5-reducible, e.g.:

    >>> from sympy.polys.groebnertools import lbp, sig, f5_reduce, Polyn
    >>> from sympy.polys import ring, QQ, lex

    >>> R, x,y,z = ring("x,y,z", QQ, lex)

    >>> f = lbp(sig((1, 1, 1), 4), x, 3)
    >>> g = lbp(sig((0, 0, 0), 2), x, 2)

    >>> Polyn(f).rem([Polyn(g)])
    0
    >>> f5_reduce(f, [g])
    (((1, 1, 1), 4), x, 3)

    """
    # 获取多项式 f 的环的次序（order）
    order = Polyn(f).ring.order
    # 获取多项式 f 的环的定义域（domain）
    domain = Polyn(f).ring.domain

    # 如果 f 不是多项式，则直接返回 f
    if not Polyn(f):
        return f

    # 进入无限循环，直到某个条件触发返回
    while True:
        # 将当前 f 赋值给 g
        g = f

        # 遍历集合 B 中的每个元素 h
        for h in B:
            # 如果 h 是多项式
            if Polyn(h):
                # 如果 h 的主导单项式 LM 能整除 f 的主导单项式 LM
                if monomial_divides(Polyn(h).LM, Polyn(f).LM):
                    # 计算 t 为 f 的首项除以 h 的首项的商
                    t = term_div(Polyn(f).LT, Polyn(h).LT, domain)
                    # 如果 h 的符号乘以 t 的第一个元素与 f 的符号比较结果小于 0
                    if sig_cmp(sig_mult(Sign(h), t[0]), Sign(f), order) < 0:
                        # 使用 h 的首项乘以 t 更新 hp
                        hp = lbp_mul_term(h, t)
                        # 更新 f 为 f 减去 hp
                        f = lbp_sub(f, hp)
                        # 跳出当前循环，继续外层循环
                        break

        # 如果经过一轮循环后 f 没有变化，或者 f 不是多项式，则返回 f
        if g == f or not Polyn(f):
            return f
    # 使用 F5B 算法计算生成由 F 中多项式生成的理想的简化 Groebner 基础
    """
    Computes a reduced Groebner basis for the ideal generated by F.

    f5b is an implementation of the F5B algorithm by Yao Sun and
    Dingkang Wang. Similarly to Buchberger's algorithm, the algorithm
    proceeds by computing critical pairs, computing the S-polynomial,
    reducing it and adjoining the reduced S-polynomial if it is not 0.

    Unlike Buchberger's algorithm, each polynomial contains additional
    information, namely a signature and a number. The signature
    specifies the path of computation (i.e. from which polynomial in
    the original basis was it derived and how), the number says when
    the polynomial was added to the basis.  With this information it
    is (often) possible to decide if an S-polynomial will reduce to
    0 and can be discarded.

    Optimizations include: Reducing the generators before computing
    a Groebner basis, removing redundant critical pairs when a new
    polynomial enters the basis and sorting the critical pairs and
    the current basis.

    Once a Groebner basis has been found, it gets reduced.

    References
    ==========

    .. [1] Yao Sun, Dingkang Wang: "A New Proof for the Correctness of F5
           (F5-Like) Algorithm", https://arxiv.org/abs/1004.0084 (specifically
           v4)

    .. [2] Thomas Becker, Volker Weispfenning, Groebner bases: A computational
           approach to commutative algebra, 1993, p. 203, 216
    """
    # 获取环的排序规则
    order = ring.order

    # reduce polynomials (like in Mario Pernici's implementation) (Becker, Weispfenning, p. 203)
    # 使用 Mario Pernici 实现中的方法，对多项式进行简化
    B = F
    while True:
        F = B
        B = []

        for i in range(len(F)):
            p = F[i]
            # 对 F 中的多项式 p 求余，得到余式 r
            r = p.rem(F[:i])

            if r:
                # 如果余式 r 非零，则将其添加到 B 中
                B.append(r)

        # 如果没有新的多项式添加到 B 中，则退出循环
        if F == B:
            break

    # basis
    # 创建基础 B，每个元素是由 lbp 函数生成的对象，具有签名和编号
    B = [lbp(sig(ring.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    # 根据多项式的首项的排序规则对基础 B 进行降序排序
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)

    # critical pairs
    # 创建关键对 CP，其中每对是通过 critical_pair 函数计算得到的
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    # 根据关键对的排序规则对 CP 进行降序排序
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

    # 初始化计数器 k，用于跟踪基础 B 中的多项式数量
    k = len(B)

    # 初始化将多项式约减为零的计数器
    reductions_to_zero = 0
    # 当 CP 非空时循环执行以下操作
    while len(CP):
        # 弹出 CP 中的一个元素作为 cp
        cp = CP.pop()

        # 丢弃重复的关键对：
        # 如果 cp 的第一个元素与 cp[2] 的数值是可重写或可比较的，并且 cp[2] 在 B 中
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        # 如果 cp 的第四个元素与 cp[5] 的数值是可重写或可比较的，并且 cp[5] 在 B 中
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue

        # 计算 s 多项式
        s = s_poly(cp)

        # 使用 B 中的元素对 s 进行约简
        p = f5_reduce(s, B)

        # 将 p 转化为本原多项式，并使用其符号进行比较
        p = lbp(Sign(p), Polyn(p).monic(), k + 1)

        # 如果 p 不为空
        if Polyn(p):
            # 移除旧的关键对，这些关键对在添加 p 后变得多余：
            indices = []
            for i, cp in enumerate(CP):
                if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                    indices.append(i)
                elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                    indices.append(i)

            # 倒序移除多余的关键对
            for i in reversed(indices):
                del CP[i]

            # 只添加不会被 p 变得多余的新关键对：
            for g in B:
                if Polyn(g):
                    cp = critical_pair(p, g, ring)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue

                    CP.append(cp)

            # 根据 cp_key 函数对 CP 进行排序（使用降序）
            CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

            # 将 p 插入到 B 中：
            m = Polyn(p).LM
            if order(m) <= order(Polyn(B[-1]).LM):
                B.append(p)
            else:
                for i, q in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break

            # 增加 k 的值
            k += 1

            # 输出一些信息，显示移除了多少个关键对
            #print(len(B), len(CP), "%d critical pairs removed" % len(indices))
        else:
            # 如果 p 为空，则将 reductions_to_zero 加一
            reductions_to_zero += 1

    # 将 Groebner 基约化：
    # 将 B 中的每个元素转化为本原多项式
    H = [Polyn(g).monic() for g in B]
    # 使用 red_groebner 函数对 H 进行约化
    H = red_groebner(H, ring)

    # 按照 LM 的次序对 H 进行排序（使用降序）
    return sorted(H, key=lambda f: order(f.LM), reverse=True)
def groebner_lcm(f, g):
    """
    Computes LCM of two polynomials using Groebner bases.

    The LCM is computed as the unique generator of the intersection
    of the two ideals generated by `f` and `g`. The approach is to
    compute a Groebner basis with respect to lexicographic ordering
    of `t*f` and `(1 - t)*g`, where `t` is an unrelated variable and
    then filtering out the solution that does not contain `t`.

    References
    ==========

    .. [1] [Cox97]_

    """
    # 检查输入的多项式是否属于同一个环
    if f.ring != g.ring:
        raise ValueError("Values should be equal")

    ring = f.ring
    domain = ring.domain

    # 如果其中一个多项式为空，则返回环的零元素
    if not f or not g:
        return ring.zero

    # 如果两个多项式只有一个单项式，则计算它们的最小公倍数
    if len(f) <= 1 and len(g) <= 1:
        # 计算单项式的最小公倍数和系数的最小公倍数
        monom = monomial_lcm(f.LM, g.LM)
        coeff = domain.lcm(f.LC, g.LC)
        return ring.term_new(monom, coeff)

    # 将多项式化为原始形式
    fc, f = f.primitive()
    gc, g = g.primitive()

    # 计算系数的最小公倍数
    lcm = domain.lcm(fc, gc)

    # 对多项式的项进行转换，以便于计算格罗布纳基
    f_terms = [ ((1,) + monom, coeff) for monom, coeff in f.terms() ]
    g_terms = [ ((0,) + monom, coeff) for monom, coeff in g.terms() ] \
            + [ ((1,) + monom,-coeff) for monom, coeff in g.terms() ]

    # 创建一个虚拟变量 t
    t = Dummy("t")
    # 使用现有的 ring 对象克隆一个新的对象 t_ring，同时添加一个新符号 t，并按 lex 排序
    t_ring = ring.clone(symbols=(t,) + ring.symbols, order=lex)

    # 从给定的 f_terms 创建 t_ring 对象 F
    F = t_ring.from_terms(f_terms)

    # 从给定的 g_terms 创建 t_ring 对象 G
    G = t_ring.from_terms(g_terms)

    # 计算 F 和 G 的格罗布纳基基底，结果存储在 basis 中
    basis = groebner([F, G], t_ring)

    # 定义一个函数 is_independent，判断 h 中第 j 个单项式是否独立
    def is_independent(h, j):
        return not any(monom[j] for monom in h.monoms())

    # 从 basis 中选择所有满足 is_independent(h, 0) 的 h 元素，存储在 H 中
    H = [ h for h in basis if is_independent(h, 0) ]

    # 从 H 中的第一个元素 H[0] 提取单项式和系数的乘积，组成 h_terms 列表
    h_terms = [ (monom[1:], coeff*lcm) for monom, coeff in H[0].terms() ]

    # 从 h_terms 创建一个新的 ring 对象 h
    h = ring.from_terms(h_terms)

    # 返回计算得到的 h 对象
    return h
# 使用 Groebner 基础计算两个多项式的最大公因式（GCD）。
def groebner_gcd(f, g):
    """Computes GCD of two polynomials using Groebner bases. """
    # 检查多项式 f 和 g 是否属于同一个环（环上的多项式系数相同）
    if f.ring != g.ring:
        raise ValueError("Values should be equal")
    
    # 获取多项式的定义域
    domain = f.ring.domain

    # 如果定义域不是域（即不是一个字段），则进行原始多项式提取
    if not domain.is_Field:
        # 将 f 和 g 提取为其原始多项式和其系数
        fc, f = f.primitive()
        gc, g = g.primitive()
        # 计算系数的最大公因数
        gcd = domain.gcd(fc, gc)

    # 计算 f 和 g 的乘积，并对其使用最小公倍数生成的 Groebner 基础
    H = (f * g).quo([groebner_lcm(f, g)])

    # 如果生成的 H 的长度不为 1，则抛出异常
    if len(H) != 1:
        raise ValueError("Length should be 1")
    
    # 获取生成的 H 中的第一个多项式 h
    h = H[0]

    # 如果定义域不是域，返回 gcd*h
    if not domain.is_Field:
        return gcd * h
    else:
        # 如果定义域是域，返回 h 的首一多项式
        return h.monic()
```