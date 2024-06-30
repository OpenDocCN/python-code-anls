# `D:\src\scipysrc\sympy\sympy\polys\factortools.py`

```
# 多项式因式分解的程序，适用于特征为零的情况。

# 从 sympy.external.gmpy 导入 GROUND_TYPES，用于确定底层数据类型
from sympy.external.gmpy import GROUND_TYPES

# 从 sympy.core.random 模块导入 _randint 函数，用于生成随机整数
from sympy.core.random import _randint

# 从 sympy.polys.galoistools 导入各种 Galois 域工具函数，用于有限域计算
from sympy.polys.galoistools import (
    gf_from_int_poly, gf_to_int_poly,
    gf_lshift, gf_add_mul, gf_mul,
    gf_div, gf_rem,
    gf_gcdex,
    gf_sqf_p,
    gf_factor_sqf, gf_factor)

# 从 sympy.polys.densebasic 导入稠密多项式的基本操作函数
from sympy.polys.densebasic import (
    dup_LC, dmp_LC, dmp_ground_LC,  # 提取多项式的领头系数
    dup_TC,  # 提取多项式的领导系数
    dup_convert, dmp_convert,  # 多项式之间的转换
    dup_degree, dmp_degree,  # 计算多项式的次数
    dmp_degree_in, dmp_degree_list,  # 计算多项式在特定变量下的次数
    dmp_from_dict,  # 从字典创建多项式
    dmp_zero_p,  # 判断多项式是否为零多项式
    dmp_one,  # 创建单位多项式
    dmp_nest, dmp_raise,  # 对多项式进行升阶或者降阶
    dup_strip,  # 去除多项式中的零系数项
    dmp_ground,  # 创建多变量多项式的常数项多项式
    dup_inflate,  # 对多项式进行扩充
    dmp_exclude, dmp_include,  # 排除或者包含多项式的指定变量
    dmp_inject, dmp_eject,  # 注入或者驱逐多项式的指定变量
    dup_terms_gcd, dmp_terms_gcd)  # 计算多项式的最大公因式

# 从 sympy.polys.densearith 导入稠密多项式的算术运算函数
from sympy.polys.densearith import (
    dup_neg, dmp_neg,  # 多项式取负
    dup_add, dmp_add,  # 多项式加法
    dup_sub, dmp_sub,  # 多项式减法
    dup_mul, dmp_mul,  # 多项式乘法
    dup_sqr,  # 多项式的平方
    dmp_pow,  # 多项式的幂次运算
    dup_div, dmp_div,  # 多项式的除法
    dup_quo, dmp_quo,  # 多项式的商
    dmp_expand,  # 多项式的展开
    dmp_add_mul,  # 多项式加倍
    dup_sub_mul, dmp_sub_mul,  # 多项式减倍
    dup_lshift,  # 多项式左移
    dup_max_norm, dmp_max_norm,  # 多项式的最大范数
    dup_l1_norm,  # 多项式的L1范数
    dup_mul_ground, dmp_mul_ground,  # 多项式乘以常数
    dup_quo_ground, dmp_quo_ground)  # 多项式除以常数

# 从 sympy.polys.densetools 导入稠密多项式的工具函数
from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,  # 清除多项式的有理数分母
    dup_trunc, dmp_ground_trunc,  # 截断多项式的高阶项
    dup_content,  # 计算多项式的内容
    dup_monic, dmp_ground_monic,  # 归一化多项式
    dup_primitive, dmp_ground_primitive,  # 多项式的基本分解
    dmp_eval_tail,  # 对多项式进行尾部求值
    dmp_eval_in, dmp_diff_eval_in,  # 在指定点对多项式求值或者求导
    dup_shift, dmp_shift,  # 对多项式进行移位
    dup_mirror)  # 多项式的镜像

# 从 sympy.polys.euclidtools 导入欧几里得算法相关函数
from sympy.polys.euclidtools import (
    dmp_primitive,  # 多项式的基本分解
    dup_inner_gcd, dmp_inner_gcd)  # 多项式的内部最大公因式

# 从 sympy.polys.sqfreetools 导入平方因式分解相关函数
from sympy.polys.sqfreetools import (
    dup_sqf_p,  # 判断多项式是否为平方自由的
    dup_sqf_norm, dmp_sqf_norm,  # 计算多项式的标准形式
    dup_sqf_part, dmp_sqf_part,  # 计算多项式的平方部分
    _dup_check_degrees, _dmp_check_degrees)  # 检查多项式的次数

# 从 sympy.polys.polyutils 导入多项式工具函数
from sympy.polys.polyutils import _sort_factors  # 对因子进行排序

# 从 sympy.polys.polyconfig 导入配置查询函数
from sympy.polys.polyconfig import query

# 从 sympy.polys.polyerrors 导入多项式相关的错误类型
from sympy.polys.polyerrors import (
    ExtraneousFactors, DomainError, CoercionFailed, EvaluationFailed)

# 从 sympy.utilities 导入 subsets 函数，用于生成子集
from sympy.utilities import subsets

# 从 math 模块导入 ceil, log, log2 函数，并重新命名为 _ceil, _log, _log2
from math import ceil as _ceil, log as _log, log2 as _log2

# 根据 GROUND_TYPES 的值选择合适的底层数据类型进行多项式运算
if GROUND_TYPES == 'flint':
    from flint import fmpz_poly  # 导入 flint 库中的多项式操作函数
else:
    fmpz_poly = None  # 如果 GROUND_TYPES 不为 'flint'，则置为 None

def dup_trial_division(f, factors, K):
    """
    使用试除法确定单变量多项式的因子的重数。

    如果任何因子不能整除 ``f``，则会引发错误。
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dup_div(f, factor, K)

            if not r:
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)

def dmp_trial_division(f, factors, u, K):
    """
    使用试除法确定多变量多项式的因子的重数。

    如果任何因子不能整除 ``f``，则会引发错误。
    """
    result = []
    # 对每个因子进行循环处理
    for factor in factors:
        # 初始化计数器 k
        k = 0

        # 进行试除法操作，直到不能再整除为止
        while True:
            # 使用 dmp_div 函数对 f 和 factor 进行除法运算，返回商 q 和余数 r
            q, r = dmp_div(f, factor, u, K)

            # 如果余数 r 是零，表示能整除
            if dmp_zero_p(r, u):
                # 更新 f 为商 q，增加 k 计数
                f, k = q, k + 1
            else:
                # 如果有余数 r，则退出循环
                break

        # 如果 k 为零，表示试除失败，抛出运行时错误
        if k == 0:
            raise RuntimeError("trial division failed")

        # 将当前因子 factor 和其出现的次数 k 添加到结果列表中
        result.append((factor, k))

    # 对结果列表进行排序并返回
    return _sort_factors(result)
def dup_zz_hensel_step(m, f, g, h, s, t, K):
    """
    One step in Hensel lifting in `Z[x]`.

    Given positive integer `m` and `Z[x]` polynomials `f`, `g`, `h`, `s`
    and `t` such that::

        f = g*h (mod m)
        s*g + t*h = 1 (mod m)

        lc(f) is not a zero divisor (mod m)
        lc(h) = 1

        deg(f) = deg(g) + deg(h)
        deg(s) < deg(h)
        deg(t) < deg(g)

    returns polynomials `G`, `H`, `S` and `T`, such that::

        f = G*H (mod m**2)
        S*G + T*H = 1 (mod m**2)

    References
    ==========

    .. [1] [Gathen99]_

    """
    M = m**2  # 计算 m 的平方

    e = dup_sub_mul(f, g, h, K)  # 计算 f - g * h
    e = dup_trunc(e, M, K)  # 截断多项式 e，使其次数不超过 M

    q, r = dup_div(dup_mul(s, e, K), h, K)  # 计算 (s * e) // h 的商和余数

    q = dup_trunc(q, M, K)  # 截断多项式 q，使其次数不超过 M
    r = dup_trunc(r, M, K)  # 截断多项式 r，使其次数不超过 M

    u = dup_add(dup_mul(t, e, K), dup_mul(q, g, K), K)  # 计算 t * e + q * g
    G = dup_trunc(dup_add(g, u, K), M, K)  # 计算 G = g + u 并截断使其次数不超过 M
    H = dup_trunc(dup_add(h, r, K), M, K)  # 计算 H = h + r 并截断使其次数不超过 M

    u = dup_add(dup_mul(s, G, K), dup_mul(t, H, K), K)  # 计算 s * G + t * H
    b = dup_trunc(dup_sub(u, [K.one], K), M, K)  # 计算 u - 1 并截断使其次数不超过 M

    c, d = dup_div(dup_mul(s, b, K), H, K)  # 计算 (s * b) // H 的商和余数

    c = dup_trunc(c, M, K)  # 截断多项式 c，使其次数不超过 M
    d = dup_trunc(d, M, K)  # 截断多项式 d，使其次数不超过 M
    # 调用 dup_mul 函数两次，分别对 t 和 b 进行乘法，并对结果进行加法
    u = dup_add(dup_mul(t, b, K), dup_mul(c, G, K), K)
    
    # 调用 dup_sub 函数，对 s 和 d 进行减法，并对结果进行截断
    S = dup_trunc(dup_sub(s, d, K), M, K)
    
    # 调用 dup_sub 函数，对 t 和 u 进行减法，并对结果进行截断
    T = dup_trunc(dup_sub(t, u, K), M, K)
    
    # 返回变量 G, H, S, T 的值作为函数结果
    return G, H, S, T
    # 多因子 Hensel 提升在 `Z[x]` 中的实现
    # 给定素数 `p`，多项式 `f` 在 `Z[x]` 上，使得 `lc(f)` 在模 `p` 下是单位，
    # 并且互为首的单位系列多项式 `f_i` 在 `Z[x]` 上，满足：
    # f = lc(f) f_1 ... f_r (mod p)
    # 和正整数 `l`，返回一系列满足下述条件的单位系列多项式 `F_1,\ F_2,\ \dots,\ F_r`：
    # f = lc(f) F_1 ... F_r (mod p**l)
    # F_i = f_i (mod p)，i = 1..r
    def dup_zz_hensel_lift(p, f, f_list, l, K):
        r"""
        Multifactor Hensel lifting in `Z[x]`.

        Given a prime `p`, polynomial `f` over `Z[x]` such that `lc(f)`
        is a unit modulo `p`, monic pair-wise coprime polynomials `f_i`
        over `Z[x]` satisfying::

            f = lc(f) f_1 ... f_r (mod p)

        and a positive integer `l`, returns a list of monic polynomials
        `F_1,\ F_2,\ \dots,\ F_r` satisfying::

           f = lc(f) F_1 ... F_r (mod p**l)

           F_i = f_i (mod p), i = 1..r

        References
        ==========

        .. [1] [Gathen99]_

        """
        r = len(f_list)
        lc = dup_LC(f, K)

        if r == 1:
            F = dup_mul_ground(f, K.gcdex(lc, p**l)[0], K)
            return [ dup_trunc(F, p**l, K) ]

        m = p
        k = r // 2
        d = int(_ceil(_log2(l)))

        g = gf_from_int_poly([lc], p)

        for f_i in f_list[:k]:
            g = gf_mul(g, gf_from_int_poly(f_i, p), p, K)

        h = gf_from_int_poly(f_list[k], p)

        for f_i in f_list[k + 1:]:
            h = gf_mul(h, gf_from_int_poly(f_i, p), p, K)

        s, t, _ = gf_gcdex(g, h, p, K)

        g = gf_to_int_poly(g, p)
        h = gf_to_int_poly(h, p)
        s = gf_to_int_poly(s, p)
        t = gf_to_int_poly(t, p)

        for _ in range(1, d + 1):
            (g, h, s, t), m = dup_zz_hensel_step(m, f, g, h, s, t, K), m**2

        return dup_zz_hensel_lift(p, g, f_list[:k], l, K) \
            + dup_zz_hensel_lift(p, h, f_list[k:], l, K)

    # 测试函数 `_test_pl`，检查 `fc` 是否为 `pl` 的非正数的约束
    def _test_pl(fc, q, pl):
        if q > pl // 2:
            q = q - pl
        if not q:
            return True
        return fc % q == 0

    # 在 `Z[x]` 中因式分解原始的平方自由多项式
    def dup_zz_zassenhaus(f, K):
        """Factor primitive square-free polynomials in `Z[x]`. """
        n = dup_degree(f)

        if n == 1:
            return [f]

        from sympy.ntheory import isprime

        fc = f[-1]
        A = dup_max_norm(f, K)
        b = dup_LC(f, K)
        B = int(abs(K.sqrt(K(n + 1))*2**n*A*b))
        C = int((n + 1)**(2*n)*A**(2*n - 1))
        gamma = int(_ceil(2*_log2(C)))
        bound = int(2*gamma*_log(gamma))
        a = []
        # 选择素数 `p`，使得 `f` 在 `Z_p` 中是平方自由的
        # 如果在 `Z_p` 中有多个因子，选择其中少数几个不同的 `p`
        # 其中因子较少
        for px in range(3, bound + 1):
            if not isprime(px) or b % px == 0:
                continue

            px = K.convert(px)

            F = gf_from_int_poly(f, px)

            if not gf_sqf_p(F, px, K):
                continue
            fsqfx = gf_factor_sqf(F, px, K)[1]
            a.append((px, fsqfx))
            if len(fsqfx) < 15 or len(a) > 4:
                break
        p, fsqf = min(a, key=lambda x: len(x[1]))

        l = int(_ceil(_log(2*B + 1, p)))

        modular = [gf_to_int_poly(ff, p) for ff in fsqf]

        g = dup_zz_hensel_lift(p, f, modular, l, K)

        sorted_T = range(len(g))
        T = set(sorted_T)
        factors, s = [], 1
        pl = p**l
    # 当 `2*s` 小于等于列表 `T` 的长度时，执行循环
    while 2*s <= len(T):
        # 遍历排序后的列表 `sorted_T` 的子集 `S`
        for S in subsets(sorted_T, s):
            # 如果常数系数不在 `fc` 中，则在子集 `S` 的因子乘积 `G` 也不会在输入多项式中出现
            if b == 1:
                # 如果 `b` 等于 1，则初始化 `q` 为 1，计算乘积 `q` 的常数系数
                q = 1
                for i in S:
                    q = q * g[i][-1]
                q = q % pl
                # 如果 `q` 不能整除 `fc`，则继续下一个子集的处理
                if not _test_pl(fc, q, pl):
                    continue
            else:
                # 否则，初始化 `G` 为包含 `b` 的列表，计算子集 `S` 的因子乘积 `G`
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)
                G = dup_primitive(G, K)[1]
                q = G[-1]
                # 如果 `q` 存在且 `fc` 不能整除 `q`，则继续下一个子集的处理
                if q and fc % q != 0:
                    continue

            # 初始化 `H` 为包含 `b` 的列表，设置 `S` 为集合形式，计算 `T` 与 `S` 的差集 `T_S`
            H = [b]
            S = set(S)
            T_S = T - S

            # 如果 `b` 等于 1，则重新计算 `G` 为子集 `S` 的因子乘积
            if b == 1:
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)

            # 计算 `T_S` 中所有索引对应的因子乘积 `H`
            for i in T_S:
                H = dup_mul(H, g[i], K)

            H = dup_trunc(H, pl, K)

            # 计算 `G` 和 `H` 的 L1 范数
            G_norm = dup_l1_norm(G, K)
            H_norm = dup_l1_norm(H, K)

            # 如果 `G_norm` 和 `H_norm` 的乘积小于等于 `B`，更新 `T` 和 `sorted_T` 的值
            if G_norm * H_norm <= B:
                T = T_S
                sorted_T = [i for i in sorted_T if i not in S]

                # 提取 `G` 和 `H` 的原始多项式
                G = dup_primitive(G, K)[1]
                f = dup_primitive(H, K)[1]

                # 将 `G` 添加到因子列表 `factors` 中，更新 `b` 的值为 `f` 的主导系数
                factors.append(G)
                b = dup_LC(f, K)

                # 跳出当前循环
                break
        else:
            # 如果内层循环未中断，则增加 `s` 的值
            s += 1

    # 返回因子列表 `factors` 加上最后一个多项式 `f`
    return factors + [f]
# 使用 Eisenstein 准则测试多项式是否是旋群不可约的
def dup_zz_irreducible_p(f, K):
    # 计算多项式 f 的首项系数
    lc = dup_LC(f, K)
    # 计算多项式 f 的首项
    tc = dup_TC(f, K)

    # 计算多项式 f 的除去首项的内容
    e_fc = dup_content(f[1:], K)

    # 如果除去首项内容不为空，则执行以下操作
    if e_fc:
        # 导入 sympy.ntheory 中的 factorint 函数
        from sympy.ntheory import factorint
        # 对多项式 f 的除去首项内容进行因数分解
        e_ff = factorint(int(e_fc))

        # 遍历除去首项内容的因数
        for p in e_ff.keys():
            # 如果首项系数 lc 不能被 p 整除且首项 tc 不能被 p^2 整除，则返回 True
            if (lc % p) and (tc % p**2):
                return True


# 高效地测试多项式是否是旋群多项式
def dup_cyclotomic_p(f, K, irreducible=False):
    """
    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(f)
    False

    >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(g)
    True

    References
    ==========

    Bradford, Russell J., and James H. Davenport. "Effective tests for
    cyclotomic polynomials." In International Symposium on Symbolic and
    Algebraic Computation, pp. 244-251. Springer, Berlin, Heidelberg, 1988.

    """
    # 如果 K 是有理数域，则尝试将多项式 f 转换为 K 的环
    if K.is_QQ:
        try:
            K0, K = K, K.get_ring()
            f = dup_convert(f, K0, K)
        except CoercionFailed:
            return False
    # 如果 K 不是整数环，则返回 False
    elif not K.is_ZZ:
        return False

    # 计算多项式 f 的首项系数
    lc = dup_LC(f, K)
    # 计算多项式 f 的首项
    tc = dup_TC(f, K)

    # 如果首项系数不为 1 或者首项不为 -1 或 1，则返回 False
    if lc != 1 or (tc != -1 and tc != 1):
        return False

    # 如果不要求多项式 f 不可约，则执行以下操作
    if not irreducible:
        # 计算多项式 f 的因子列表
        coeff, factors = dup_factor_list(f, K)

        # 如果系数不为 K 的单位元或者因子列表不为 [(f, 1)]，则返回 False
        if coeff != K.one or factors != [(f, 1)]:
            return False

    # 计算多项式 f 的次数
    n = dup_degree(f)
    g, h = [], []

    # 将多项式 f 的偶次项系数和奇次项系数分别存入 g 和 h 中
    for i in range(n, -1, -2):
        g.insert(0, f[i])

    for i in range(n - 1, -1, -2):
        h.insert(0, f[i])

    # 对 g 和 h 进行平方处理，并赋值给 F
    g = dup_sqr(dup_strip(g), K)
    h = dup_sqr(dup_strip(h), K)

    F = dup_sub(g, dup_lshift(h, 1, K), K)

    # 如果 F 的首项系数为 K 中的负数，则将 F 取反
    if K.is_negative(dup_LC(F, K)):
        F = dup_neg(F, K)

    # 如果 F 等于 f，则返回 True
    if F == f:
        return True

    # 将 f 翻转，赋值给 g
    g = dup_mirror(f, K)

    # 如果 g 的首项系数为 K 中的负数，则将 g 取反
    if K.is_negative(dup_LC(g, K)):
        g = dup_neg(g, K)

    # 如果 F 等于 g 并且 g 是旋群多项式，则返回 True
    if F == g and dup_cyclotomic_p(g, K):
        return True

    # 计算 F 的平方自由部分，并赋值给 G
    G = dup_sqf_part(F, K)

    # 如果 G 的平方等于 F 并且 G 是旋群多项式，则返回 True
    if dup_sqr(G, K) == F and dup_cyclotomic_p(G, K):
        return True

    return False


# 高效生成第 n 个旋群多项式
def dup_zz_cyclotomic_poly(n, K):
    """
    Examples
    ========

    """
    from sympy.ntheory import factorint
    h = [K.one, -K.one]

    # 对 n 进行因数分解，并生成对应的旋群多项式
    for p, k in factorint(n).items():
        h = dup_quo(dup_inflate(h, p, K), h, K)
        h = dup_inflate(h, p**(k - 1), K)

    return h


# 分解旋群多项式
def _dup_cyclotomic_decompose(n, K):
    from sympy.ntheory import factorint

    H = [[K.one, -K.one]]

    # 对 n 进行因数分解，并生成对应的旋群多项式列表
    for p, k in factorint(n).items():
        Q = [ dup_quo(dup_inflate(h, p, K), h, K) for h in H ]
        H.extend(Q)

        for i in range(1, k):
            Q = [ dup_inflate(q, p, K) for q in Q ]
            H.extend(Q)

    return H


# 高效因式分解多项式 `x**n - 1` 和 `x**n + 1` 在 `Z[x]` 中
def dup_zz_cyclotomic_factor(f, K):
    """
    """
    # 计算多项式 f 的首项系数 lc_f 和尾项系数 tc_f
    lc_f, tc_f = dup_LC(f, K)

    # 如果 f 的次数小于等于 0，返回 None
    if dup_degree(f) <= 0:
        return None

    # 如果 f 的首项系数不为 1，或者尾项系数不为 -1 或 1，返回 None
    if lc_f != 1 or tc_f not in [-1, 1]:
        return None

    # 如果 f 中除了首项和尾项系数外还有其他非零项，返回 None
    if any(bool(cf) for cf in f[1:-1]):
        return None

    # 计算多项式 f 的次数 n
    n = dup_degree(f)

    # 使用分解旋轮多项式的方法计算多项式 f 的分解 F
    F = _dup_cyclotomic_decompose(n, K)

    # 如果 f 的尾项系数为 1，则返回计算得到的分解 F
    if not K.is_one(tc_f):
        return F
    else:
        # 否则，初始化空列表 H
        H = []

        # 遍历计算 2*n 次旋轮多项式的分解，将不在 F 中的项添加到 H 列表中
        for h in _dup_cyclotomic_decompose(2*n, K):
            if h not in F:
                H.append(h)

        # 返回列表 H，其中包含不在 F 中的分解项
        return H
# 在整数环中，对多项式 `f` 进行因式分解，不保证多项式是平方自由的（非原始的）
def dup_zz_factor_sqf(f, K):
    """Factor square-free (non-primitive) polynomials in `Z[x]`. """
    # 计算多项式 `f` 的内容和其原始多项式
    cont, g = dup_primitive(f, K)

    # 计算多项式 `g` 的次数
    n = dup_degree(g)

    # 如果 `g` 的首项系数小于零，则取相反数并更新内容 `cont`
    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    # 如果 `g` 的次数小于等于零，返回其内容和空因子列表
    if n <= 0:
        return cont, []
    # 如果 `g` 的次数为1，返回其内容和包含 `g` 的因子列表
    elif n == 1:
        return cont, [g]

    # 如果启用了使用不可约多项式的选项，并且 `g` 是不可约的，则返回其内容和单一的不可约因子
    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [g]

    factors = None

    # 如果启用了使用周期因子分解的选项，则使用周期因子分解 `g`
    if query('USE_CYCLOTOMIC_FACTOR'):
        factors = dup_zz_cyclotomic_factor(g, K)

    # 如果周期因子分解没有成功，使用 Zassenhaus 算法进行因式分解
    if factors is None:
        factors = dup_zz_zassenhaus(g, K)

    # 返回内容和排序后的因子列表
    return cont, _sort_factors(factors, multiple=False)


# 在整数环中，对多项式 `f` 进行因式分解
def dup_zz_factor(f, K):
    """
    Factor (non square-free) polynomials in `Z[x]`.

    Given a univariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, ..., f_n` into irreducibles over integers::

                f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Zassenhaus algorithm. Trial division is used to recover the
    multiplicities of factors.

    The result is returned as a tuple consisting of::

              (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Examples
    ========

    Consider the polynomial `f = 2*x**4 - 2`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x = ring("x", ZZ)

        >>> R.dup_zz_factor(2*x**4 - 2)
        (2, [(x - 1, 1), (x + 1, 1), (x**2 + 1, 1)])

    In result we got the following factorization::

                 f = 2 (x - 1) (x + 1) (x**2 + 1)

    Note that this is a complete factorization over integers,
    however over Gaussian integers we can factor the last term.

    By default, polynomials `x**n - 1` and `x**n + 1` are factored
    using cyclotomic decomposition to speedup computations. To
    disable this behaviour set cyclotomic=False.

    References
    ==========

    .. [1] [Gathen99]_

    """
    # 如果使用的底层类型是 'flint'
    if GROUND_TYPES == 'flint':
        # 将多项式 `f` 转换为 `fmpz_poly` 类型，然后进行因式分解
        f_flint = fmpz_poly(f[::-1])
        cont, factors = f_flint.factor()
        # 调整因子的顺序和系数，然后返回内容和排序后的因子列表
        factors = [(fac.coeffs()[::-1], exp) for fac, exp in factors]
        return cont, _sort_factors(factors)

    # 计算多项式 `f` 的内容和其原始多项式
    cont, g = dup_primitive(f, K)

    # 计算多项式 `g` 的次数
    n = dup_degree(g)

    # 如果 `g` 的首项系数小于零，则取相反数并更新内容 `cont`
    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    # 如果 `g` 的次数小于等于零，返回其内容和空因子列表
    if n <= 0:
        return cont, []
    # 如果 `g` 的次数为1，返回其内容和包含 `g` 的因子列表
    elif n == 1:
        return cont, [(g, 1)]

    # 如果启用了使用不可约多项式的选项，并且 `g` 是不可约的，则返回其内容和单一的不可约因子
    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [(g, 1)]

    # 计算 `g` 的平方自由部分
    g = dup_sqf_part(g, K)
    H = None

    # 如果启用了使用周期因子分解的选项，则使用周期因子分解 `g`
    if query('USE_CYCLOTOMIC_FACTOR'):
        H = dup_zz_cyclotomic_factor(g, K)

    # 如果周期因子分解没有成功，使用 Zassenhaus 算法进行因式分解
    if H is None:
        H = dup_zz_zassenhaus(g, K)

    # 使用试除法恢复因子的重数
    factors = dup_trial_division(f, H, K)

    # 检查多项式 `f` 和因子列表的次数是否匹配
    _dup_check_degrees(f, factors)

    # 返回内容和因子列表
    return cont, factors


# Wang/EEZ: 计算有效除数的集合
def dmp_zz_wang_non_divisors(E, cs, ct, K):
    """Wang/EEZ: Compute a set of valid divisors.  """
    # 返回一个列表，包含 `cs*ct` 的结果
    result = [ cs*ct ]
    # 对于集合 E 中的每个元素 q，取其绝对值
    for q in E:
        q = abs(q)

        # 遍历结果列表 result 的逆序
        for r in reversed(result):
            # 当 r 不等于 1 时，持续计算 r 和 q 的最大公约数，并更新 q
            while r != 1:
                r = K.gcd(r, q)
                q = q // r

            # 如果此时 q 等于 1，说明在当前 r 的情况下，q 已经被整除至 1，无需继续
            if K.is_one(q):
                # 返回 None，表示无法继续分解
                return None

        # 将处理后的 q 添加到结果列表中
        result.append(q)

    # 返回结果列表中除第一个元素外的所有元素
    return result[1:]
# 解决多变量丢番图方程，基于 Wang/EEZ 算法
def dmp_zz_diophantine(F, c, A, d, p, u, K):
    """Wang/EEZ: Solve multivariate Diophantine equations. """
    # 如果 F 中只有两个多项式
    if len(F) == 2:
        a, b = F
        # 将整数多项式转换为有限域中的多项式
        f = gf_from_int_poly(a, p)
        g = gf_from_int_poly(b, p)
        # 计算在有限域中的扩展最大公因式
        s, t, G = gf_gcdex(g, f, p, K)
        # 将 s 和 t 向左移动 m 位
        s = gf_lshift(s, m, K)
        t = gf_lshift(t, m, K)
        # 计算 s 除以 f 的商 q 和余数 s
        q, s = gf_div(s, f, p, K)
        # 计算 t 加上 q 乘以 g 的结果
        t = gf_add_mul(t, q, g, p, K)
        # 将 s 和 t 转换为整数多项式
        s = gf_to_int_poly(s, p)
        t = gf_to_int_poly(t, p)
        # 返回解的列表
        result = [s, t]
    else:
        # 初始化 G 列表
        G = [F[-1]]
        # 从倒数第二个多项式开始向前遍历 F 中的多项式
        for f in reversed(F[1:-1]):
            # 计算多项式乘积
            G.insert(0, dup_mul(f, G[0], K))
        # 初始化 S 和 T 列表
        S, T = [], [[1]]
        # 对 F 和 G 中的每对多项式执行迭代
        for f, g in zip(F, G):
            # 解决单变量丢番图方程
            t, s = dmp_zz_diophantine([g, f], T[-1], [], 0, p, 1, K)
            T.append(t)
            S.append(s)
        # 将 S 和 T 的结果合并
        result, S = [], S + [T[-1]]
        # 对于每个 S 中的多项式 s 和 F 中的多项式 f
        for s, f in zip(S, F):
            # 将 s 转换为有限域中的多项式
            s = gf_from_int_poly(s, p)
            f = gf_from_int_poly(f, p)
            # 计算 s 向左移动 m 位后对 f 求余数 r
            r = gf_rem(gf_lshift(s, m, K), f, p, K)
            # 将 r 转换为整数多项式
            s = gf_to_int_poly(r, p)
            # 将解添加到结果列表中
            result.append(s)
    # 返回最终的解列表
    return result
    # 如果 A 是空的，则初始化 S 为一个空列表的列表，长度为 F 的长度，并计算多项式 c 的重复度
    if not A:
        S = [ [] for _ in F ]
        n = dup_degree(c)

        # 遍历多项式 c 的系数，跳过系数为零的情况
        for i, coeff in enumerate(c):
            if not coeff:
                continue

            # 计算 dup_zz_diophantine 函数的结果，将结果存储在 T 中
            T = dup_zz_diophantine(F, n - i, p, K)

            # 将每个 T 中的元素与对应的 S[j] 相乘，并截断结果，存储在 S[j] 中
            for j, (s, t) in enumerate(zip(S, T)):
                t = dup_mul_ground(t, coeff, K)
                S[j] = dup_trunc(dup_add(s, t, K), p, K)
    else:
        # 如果 A 不为空，则进行以下操作：
        n = len(A)
        e = dmp_expand(F, u, K)

        # 获取 A 的最后一个元素，并将其作为 a，其余元素作为 A 的剩余部分
        a, A = A[-1], A[:-1]
        B, G = [], []

        # 对每个 F 中的元素 f，计算 B 和 G 的值
        for f in F:
            B.append(dmp_quo(e, f, u, K))
            G.append(dmp_eval_in(f, a, n, u, K))

        # 计算多项式 c 在 a 处的值
        C = dmp_eval_in(c, a, n, u, K)

        # 设置 v 为 u - 1
        v = u - 1

        # 计算 dmp_zz_diophantine 函数的结果 S，并对每个结果执行 dmp_raise 操作
        S = dmp_zz_diophantine(G, C, A, d, p, v, K)
        S = [ dmp_raise(s, 1, v, K) for s in S ]

        # 对每个 S 和 B 中的元素执行 dmp_sub_mul 操作，并截断结果
        for s, b in zip(S, B):
            c = dmp_sub_mul(c, s, b, u, K)

        # 对 c 进行 dmp_ground_trunc 操作，截断结果
        c = dmp_ground_trunc(c, p, u, K)

        # 计算 m 和 M 的值，其中 m 是 [K.one, -a] 的 n 次嵌套，M 初始化为单位元素
        m = dmp_nest([K.one, -a], n, K)
        M = dmp_one(n, K)

        # 对 k 从 0 到 d 进行迭代
        for k in range(0, d):
            # 如果 c 是零多项式，则跳出循环
            if dmp_zero_p(c, u):
                break

            # 计算 M 与 m 的乘积，并计算 c 在 k+1 处的偏导数 C
            M = dmp_mul(M, m, u, K)
            C = dmp_diff_eval_in(c, k + 1, a, n, u, K)

            # 如果 C 不是零多项式，则对 C 执行 dmp_quo_ground 操作，并计算 dmp_zz_diophantine 的结果 T
            if not dmp_zero_p(C, v):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), v, K)
                T = dmp_zz_diophantine(G, C, A, d, p, v, K)

                # 对每个 T 中的元素执行 dmp_raise 和 dmp_mul 操作，并存储结果到 S 中
                for i, t in enumerate(T):
                    T[i] = dmp_mul(dmp_raise(t, 1, v, K), M, u, K)

                for i, (s, t) in enumerate(zip(S, T)):
                    S[i] = dmp_add(s, t, u, K)

                # 对每个 T 和 B 中的元素执行 dmp_sub_mul 操作，并截断结果
                for t, b in zip(T, B):
                    c = dmp_sub_mul(c, t, b, u, K)

                # 对 c 进行 dmp_ground_trunc 操作，截断结果
                c = dmp_ground_trunc(c, p, u, K)

        # 对每个 S 中的元素执行 dmp_ground_trunc 操作，截断结果
        S = [ dmp_ground_trunc(s, p, u, K) for s in S ]

    # 返回结果列表 S
    return S
def dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K):
    """Wang/EEZ: Parallel Hensel lifting algorithm. """
    # 初始化 S 为包含 f 的列表，n 为 A 的长度，v 为 u - 1
    S, n, v = [f], len(A), u - 1

    # 复制 H 到列表中
    H = list(H)

    # 对 A 的倒数第二个元素开始的每个元素 a 执行循环
    for i, a in enumerate(reversed(A[1:])):
        # 计算 S[0] 在 a 处的评估，生成 n - i, u - i 次数的结果
        s = dmp_eval_in(S[0], a, n - i, u - i, K)
        # 将结果的地面截断并插入到 S 的开头
        S.insert(0, dmp_ground_trunc(s, p, v - i, K))

    # 计算 f 在 u 次数下的最高次数
    d = max(dmp_degree_list(f, u)[1:])

    # 对于范围从 2 到 n+1 的每个 j，以及对应的 S, a 执行循环
    for j, s, a in zip(range(2, n + 2), S, A):
        # 复制 H 到 G 中，并设置 w 为 j-1
        G, w = list(H), j - 1

        # I 和 J 分别是 A 的前 j-2 项和后 j-1 项
        I, J = A[:j - 2], A[j - 1:]

        # 对于 H 和 LC 的每一对 h, lc 执行循环
        for i, (h, lc) in enumerate(zip(H, LC)):
            # 在 J 处评估 lc，并对结果进行地面截断
            lc = dmp_ground_trunc(dmp_eval_tail(lc, J, v, K), p, w - 1, K)
            # 将地面截断后的 lc 作为新的首项与 h[1:] 进行提升
            H[i] = [lc] + dmp_raise(h[1:], 1, w - 1, K)

        # 构建 m 和 M
        m = dmp_nest([K.one, -a], w, K)
        M = dmp_one(w, K)

        # 计算 s 和 H 的展开之差，并对结果进行地面截断
        c = dmp_sub(s, dmp_expand(H, w, K), w, K)

        # 计算 s 在 w 次数下的次数
        dj = dmp_degree_in(s, w, w)

        # 对于范围从 0 到 dj 的每个 k 执行循环
        for k in range(0, dj):
            # 如果 c 是零多项式则退出循环
            if dmp_zero_p(c, w):
                break

            # 计算 M 与 m 的乘积
            M = dmp_mul(M, m, w, K)
            # 在 a 处对 c 进行差分评估，并对结果进行地面截断
            C = dmp_diff_eval_in(c, k + 1, a, w, w, K)

            # 如果 C 在 w-1 次数下不是零多项式
            if not dmp_zero_p(C, w - 1):
                # 对 C 进行 K(k + 1) 的阶乘地面除法
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), w - 1, K)
                # 计算 DMP ZZ Diophantine
                T = dmp_zz_diophantine(G, C, I, d, p, w - 1, K)

                # 对于 H 和 T 的每一对 h, t 执行循环
                for i, (h, t) in enumerate(zip(H, T)):
                    # 计算 h + t * M，并对结果进行地面截断
                    h = dmp_add_mul(h, dmp_raise(t, 1, w - 1, K), M, w, K)
                    H[i] = dmp_ground_trunc(h, p, w, K)

                # 计算 s 和 H 的展开之差，并对结果进行地面截断
                h = dmp_sub(s, dmp_expand(H, w, K), w, K)
                c = dmp_ground_trunc(h, p, w, K)

    # 如果 H 在 u 次数下的展开不等于 f，则抛出 ExtraneousFactors 异常
    if dmp_expand(H, u, K) != f:
        raise ExtraneousFactors  # pragma: no cover
    else:
        # 返回 H
        return H


def dmp_zz_wang(f, u, K, mod=None, seed=None):
    r"""
    Factor primitive square-free polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x_1,...,x_n]`, which is
    primitive and square-free in `x_1`, computes factorization of `f` into
    irreducibles over integers.

    The procedure is based on Wang's Enhanced Extended Zassenhaus
    algorithm. The algorithm works by viewing `f` as a univariate polynomial
    in `Z[x_2,...,x_n][x_1]`, for which an evaluation mapping is computed::

                      x_2 -> a_2, ..., x_n -> a_n

    where `a_i`, for `i = 2, \dots, n`, are carefully chosen integers.  The
    mapping is used to transform `f` into a univariate polynomial in `Z[x_1]`,
    which can be factored efficiently using Zassenhaus algorithm. The last
    step is to lift univariate factors to obtain true multivariate
    factors. For this purpose a parallel Hensel lifting procedure is used.

    The parameter ``seed`` is passed to _randint and can be used to seed randint
    (when an integer) or (for testing purposes) can be a sequence of numbers.

    References
    ==========

    .. [1] [Wang78]_
    .. [2] [Geddes92]_

    """
    # 导入 nextprime 函数
    from sympy.ntheory import nextprime

    # 使用 seed 初始化 _randint 函数
    randint = _randint(seed)

    # 计算 f 的首项系数和次数
    ct, T = dmp_zz_factor(dmp_LC(f, K), u - 1, K)

    # 计算 f 的 Mignotte 边界并生成一个素数 p
    b = dmp_zz_mignotte_bound(f, u, K)
    p = K(nextprime(b))
    # 如果模数 mod 为 None，则根据 u 的值确定 mod 的初始值
    if mod is None:
        if u == 1:
            mod = 2
        else:
            mod = 1

    # 历史记录、配置列表、系数列表 A 和 r 的初始化
    history, configs, A, r = set(), [], [K.zero]*u, None

    try:
        # 使用 dmp_zz_wang_test_points 函数获取测试点 cs, s 和误差 E
        cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)

        # 对 s 进行平方自然形式的因式分解，得到因子列表 H
        _, H = dup_zz_factor_sqf(s, K)

        # 记录因子的数量
        r = len(H)

        # 如果只有一个因子，直接返回原多项式 f 的列表形式
        if r == 1:
            return [f]

        # 将当前配置加入配置列表中
        configs = [(s, cs, E, H, A)]
    except EvaluationFailed:
        pass

    # 获取 EEZ_NUMBER_OF_CONFIGS, EEZ_NUMBER_OF_TRIES 和 EEZ_MODULUS_STEP 的值
    eez_num_configs = query('EEZ_NUMBER_OF_CONFIGS')
    eez_num_tries = query('EEZ_NUMBER_OF_TRIES')
    eez_mod_step = query('EEZ_MODULUS_STEP')

    # 当配置列表长度小于指定的配置数量时，执行以下循环
    while len(configs) < eez_num_configs:
        for _ in range(eez_num_tries):
            # 生成随机系数列表 A，每个系数的范围在 -mod 和 mod 之间
            A = [ K(randint(-mod, mod)) for _ in range(u) ]

            # 如果生成的 A 已经在历史记录中，则重新生成
            if tuple(A) not in history:
                history.add(tuple(A))
            else:
                continue

            try:
                # 使用 dmp_zz_wang_test_points 函数获取测试点 cs, s 和误差 E
                cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)
            except EvaluationFailed:
                continue

            # 对 s 进行平方自然形式的因式分解，得到因子列表 H
            _, H = dup_zz_factor_sqf(s, K)

            # 记录因子的数量
            rr = len(H)

            # 如果已经有一个记录的因子数 r，则根据条件比较 rr 和 r 的大小
            if r is not None:
                if rr != r:  # pragma: no cover
                    if rr < r:
                        configs, r = [], rr
                    else:
                        continue
            else:
                r = rr

            # 如果只有一个因子，直接返回原多项式 f 的列表形式
            if r == 1:
                return [f]

            # 将当前配置加入配置列表中
            configs.append((s, cs, E, H, A))

            # 如果配置列表长度达到指定的配置数量，退出循环
            if len(configs) == eez_num_configs:
                break
        else:
            # 如果内循环完成但配置列表长度仍未达到指定数量，则增加 mod 的值
            mod += eez_mod_step

    # 初始化最小范数、对应的索引和计数器
    s_norm, s_arg, i = None, 0, 0

    # 遍历配置列表，计算每个 s 的最大范数，并找出最小范数对应的索引
    for s, _, _, _, _ in configs:
        _s_norm = dup_max_norm(s, K)

        if s_norm is not None:
            if _s_norm < s_norm:
                s_norm = _s_norm
                s_arg = i
        else:
            s_norm = _s_norm

        i += 1

    # 根据最小范数对应的索引，获取相应的 s, cs, E, H 和 A
    _, cs, E, H, A = configs[s_arg]
    orig_f = f

    try:
        # 使用 dmp_zz_wang_lead_coeffs 函数计算领导系数和新的多项式
        f, H, LC = dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K)

        # 使用 dmp_zz_wang_hensel_lifting 函数进行 Hensel 提升
        factors = dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K)
    except ExtraneousFactors:  # pragma: no cover
        # 如果出现额外的因子，根据配置选择是否重新开始算法
        if query('EEZ_RESTART_IF_NEEDED'):
            return dmp_zz_wang(orig_f, u, K, mod + 1)
        else:
            raise ExtraneousFactors(
                "we need to restart algorithm with better parameters")

    result = []

    # 对每个因子进行处理，将其化为首项系数为正数的形式，并添加到结果列表中
    for f in factors:
        _, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)

        result.append(f)

    # 返回因子列表作为最终结果
    return result
def dmp_zz_factor(f, u, K):
    r"""
    Factor (non square-free) polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, \dots, f_n` into irreducibles over integers::

                 f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Enhanced Extended Zassenhaus (EEZ) algorithm. Trial division
    is used to recover the multiplicities of factors.

    The result is returned as a tuple consisting of::

             (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Consider polynomial `f = 2*(x**2 - y**2)`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> R.dmp_zz_factor(2*x**2 - 2*y**2)
        (2, [(x - y, 1), (x + y, 1)])

    In result we got the following factorization::

                    f = 2 (x - y) (x + y)

    References
    ==========

    .. [1] [Gathen99]_

    """
    # 如果 `u` 为假值（例如空列表），调用 `dup_zz_factor` 函数对 `f` 进行因式分解
    if not u:
        return dup_zz_factor(f, K)

    # 如果 `f` 是零多项式，返回 `(0, [])`
    if dmp_zero_p(f, u):
        return K.zero, []

    # 将 `f` 转换为其原始形式，并提取其内容和原始多项式
    cont, g = dmp_ground_primitive(f, u, K)

    # 如果原始多项式的首项系数小于零，调整为正值
    if dmp_ground_LC(g, u, K) < 0:
        cont, g = -cont, dmp_neg(g, u, K)

    # 如果所有分量的次数都小于等于零，返回内容和空的因子列表
    if all(d <= 0 for d in dmp_degree_list(g, u)):
        return cont, []

    # 将原始多项式转化为其基本部分和原始多项式
    G, g = dmp_primitive(g, u, K)

    factors = []

    # 如果原始多项式的次数大于零，提取其平方因子部分并应用 EEZ 算法进行分解
    if dmp_degree(g, u) > 0:
        g = dmp_sqf_part(g, u, K)
        H = dmp_zz_wang(g, u, K)
        factors = dmp_trial_division(f, H, u, K)

    # 对于 G 的每个因子，递归调用 `dmp_zz_factor` 函数，并将结果插入到因子列表开头
    for g, k in dmp_zz_factor(G, u - 1, K)[1]:
        factors.insert(0, ([g], k))

    # 检查因子的次数与原始多项式是否一致
    _dmp_check_degrees(f, u, factors)

    # 返回内容和已排序的因子列表
    return cont, _sort_factors(factors)


def dup_qq_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `QQ_I[x]`. """
    # 在 QQ<I> 中进行因式分解
    K1 = K0.as_AlgebraicField()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_factor_list(f, K1)
    factors = [(dup_convert(fac, K1, K0), i) for fac, i in factors]
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_zz_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `ZZ_I[x]`. """
    # 首先在 QQ_I 中进行因式分解
    K1 = K0.get_field()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_qq_i_factor(f, K1)

    new_factors = []
    for fac, i in factors:
        # 提取内容
        fac_denom, fac_num = dup_clear_denoms(fac, K1)
        fac_num_ZZ_I = dup_convert(fac_num, K1, K0)
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, 0, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dmp_qq_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `QQ_I[X]`. """
    # 在 QQ<I> 中进行因式分解
    K1 = K0.as_AlgebraicField()
    f = dmp_convert(f, u, K0, K1)
    coeff, factors = dmp_factor_list(f, u, K1)
    # 使用 dmp_convert 函数将 factors 列表中的每个 fac 元素转换为 K1 环中的多项式，并保留其索引 i
    factors = [(dmp_convert(fac, u, K1, K0), i) for fac, i in factors]
    
    # 使用 K0.convert 方法将 coeff 从 K0 环转换为 K1 环中的元素
    coeff = K0.convert(coeff, K1)
    
    # 返回转换后的 coeff 和 factors
    return coeff, factors
def dmp_zz_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `ZZ_I[X]`. """
    # 首先转换到 `QQ_I` 上的多项式
    K1 = K0.get_field()
    f = dmp_convert(f, u, K0, K1)
    # 使用 `dmp_qq_i_factor` 函数进行因式分解
    coeff, factors = dmp_qq_i_factor(f, u, K1)

    new_factors = []
    for fac, i in factors:
        # 提取内容
        fac_denom, fac_num = dmp_clear_denoms(fac, u, K1)
        # 将分子转换为 `ZZ_I` 上的多项式
        fac_num_ZZ_I = dmp_convert(fac_num, u, K1, K0)
        # 计算内容和本原部分
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, u, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_ext_factor(f, K):
    r"""Factor univariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dup_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 - 2` over `K`:

    >>> p = [K(1), K(0), K(-2)] # x^2 - 2
    >>> p1 = [K(1), -K.unit]    # x - sqrt(2)
    >>> p2 = [K(1), +K.unit]    # x + sqrt(2)
    >>> dup_ext_factor(p, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))

    Explanation
    ===========

    Uses Trager's algorithm. In particular this function is algorithm
    ``alg_factor`` from [Trager76]_.

    If `f` is a polynomial in `k(a)[x]` then its norm `g(x)` is a polynomial in
    `k[x]`. If `g(x)` is square-free and has irreducible factors `g_1(x)`,
    `g_2(x)`, `\cdots` then the irreducible factors of `f` in `k(a)[x]` are
    given by `f_i(x) = \gcd(f(x), g_i(x))` where the GCD is computed in
    `k(a)[x]`.

    The first step in Trager's algorithm is to find an integer shift `s` so
    that `f(x-sa)` has square-free norm. Then the norm is factorized in `k[x]`
    and the GCD of (shifted) `f` with each factor gives the shifted factors of
    `f`. At the end the shift is undone to recover the unshifted factors of `f`
    in `k(a)[x]`.

    The algorithm reduces the problem of factorization in `k(a)[x]` to
    factorization in `k[x]` with the main additional steps being to compute the
    norm (a resultant calculation in `k[x,y]`) and some polynomial GCDs in
    `k(a)[x]`.

    In practice in SymPy the base field `k` will be the rationals :ref:`QQ` and
    this function factorizes a polynomial with coefficients in an algebraic
    number field  like `\mathbb{Q}(\sqrt{2})`.

    See Also
    ========

    dmp_ext_factor:
        Analogous function for multivariate polynomials over ``k(a)``.
    dup_sqf_norm:
        Subroutine ``sqfr_norm`` also from [Trager76]_.
    """
    # sympy.polys.polytools.factor 函数的具体实现，用于处理需要的情况。
    """
    高级函数，根据需要最终使用此函数。
    """
    # 提取多项式 f 的次数和主导系数
    n, lc = dup_degree(f), dup_LC(f, K)
    
    # 将多项式 f 转换为首一形式
    f = dup_monic(f, K)
    
    # 如果多项式次数小于等于 0，返回主导系数和空因子列表
    if n <= 0:
        return lc, []
    # 如果多项式次数为 1，返回主导系数和包含 (f, 1) 的列表
    if n == 1:
        return lc, [(f, 1)]
    
    # 求取多项式 f 的平方因式部分及其原始形式
    f, F = dup_sqf_part(f, K), f
    s, g, r = dup_sqf_norm(f, K)
    
    # 对剩余部分 r 进行因式分解
    factors = dup_factor_list_include(r, K.dom)
    
    # 如果分解后的因子个数为 1，返回主导系数和 [(f, n//dup_degree(f))] 的列表
    if len(factors) == 1:
        return lc, [(f, n//dup_degree(f))]
    
    # 计算 H = s*K.unit
    H = s*K.unit
    
    # 对每个因子进行处理，转换为整数环 K 中的多项式，并执行内部 gcd 和移位操作
    for i, (factor, _) in enumerate(factors):
        h = dup_convert(factor, K.dom, K)
        h, _, g = dup_inner_gcd(h, g, K)
        h = dup_shift(h, H, K)
        factors[i] = h
    
    # 使用试除法对原始多项式 F 和因子列表 factors 进行因式分解
    factors = dup_trial_division(F, factors, K)
    
    # 检查分解后的因子列表与原始多项式的次数匹配情况
    _dup_check_degrees(F, factors)
    
    # 返回主导系数和因子列表作为结果
    return lc, factors
# 定义一个函数，用于在代数数域上因式分解多变量多项式。
def dmp_ext_factor(f, u, K):
    r"""Factor multivariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).
    域 `K` 必须是一个代数数域 `k(a)`（参见 :ref:`QQ(a)`）。

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dmp_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 y^2 - 2` over `K`:

    >>> p = [[K(1),K(0),K(0)], [], [K(-2)]] # x**2*y**2 - 2
    >>> p1 = [[K(1),K(0)], [-K.unit]]       # x*y - sqrt(2)
    >>> p2 = [[K(1),K(0)], [+K.unit]]       # x*y + sqrt(2)
    >>> dmp_ext_factor(p, 1, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x, y
    >>> factor(x**2*y**2 - 2, extension=sqrt(2))
    (x*y - sqrt(2))*(x*y + sqrt(2))

    Explanation
    ===========

    This is Trager's algorithm for multivariate polynomials. In particular this
    function is algorithm ``alg_factor`` from [Trager76]_.
    这是用于多变量多项式的 Trager 算法。特别是，此函数是来自 [Trager76]_ 的算法 ``alg_factor``。

    See :func:`dup_ext_factor` for explanation.
    参见 :func:`dup_ext_factor` 进行解释。

    See Also
    ========

    dup_ext_factor:
        Analogous function for univariate polynomials over ``k(a)``.
        用于 ``k(a)`` 上的单变量多项式的类似函数。
    dmp_sqf_norm:
        Multivariate version of subroutine ``sqfr_norm`` also from [Trager76]_.
        也来自 [Trager76]_ 的子程序 ``sqfr_norm`` 的多变量版本。
    sympy.polys.polytools.factor:
        The high-level function that ultimately uses this function as needed.
        最终根据需要使用此函数的高级函数。

    """
    if not u:
        return dup_ext_factor(f, K)

    # 计算多项式 `f` 的领导系数
    lc = dmp_ground_LC(f, u, K)
    # 将多项式 `f` 变为首一多项式
    f = dmp_ground_monic(f, u, K)

    # 如果多项式 `f` 的所有变量的次数都小于等于0，则返回领导系数和空因子列表
    if all(d <= 0 for d in dmp_degree_list(f, u)):
        return lc, []

    # 求 `f` 的平方因式部分
    f, F = dmp_sqf_part(f, u, K), f
    # 计算 `f` 的标准型
    s, g, r = dmp_sqf_norm(f, u, K)

    # 将 `r` 分解成因子列表
    factors = dmp_factor_list_include(r, u, K.dom)

    # 如果只有一个因子，则将其置于列表中
    if len(factors) == 1:
        factors = [f]
    else:
        # 对每个因子进行处理
        for i, (factor, _) in enumerate(factors):
            # 将因子转换为 `K` 域上的多项式
            h = dmp_convert(factor, u, K.dom, K)
            # 计算 `h` 和 `g` 的内部最大公因式
            h, _, g = dmp_inner_gcd(h, g, u, K)
            # 计算移位因子 `a`
            a = [si*K.unit for si in s]
            # 将 `h` 进行移位
            h = dmp_shift(h, a, u, K)
            factors[i] = h

    # 进行试除法
    result = dmp_trial_division(F, factors, u, K)

    # 检查结果的次数
    _dmp_check_degrees(F, u, result)

    return lc, result


def dup_gf_factor(f, K):
    """Factor univariate polynomials over finite fields. """
    # 将多项式 `f` 转换为有限域 `K` 上的多项式
    f = dup_convert(f, K, K.dom)

    # 使用有限域因式分解 `gf_factor` 对 `f` 进行因式分解
    coeff, factors = gf_factor(f, K.mod, K.dom)

    # 将因子列表中的每个因子转换为 `K` 域上的多项式
    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    return K.convert(coeff, K.dom), factors


def dmp_gf_factor(f, u, K):
    """Factor multivariate polynomials over finite fields. """
    # 抛出未实现的错误，因为多变量多项式在有限域上的因式分解尚未实现
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_factor_list(f, K0):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    # 计算 `f` 的最大公因式的系数和 `f` 的原始多项式
    j, f = dup_terms_gcd(f, K0)
    cont, f = dup_primitive(f, K0)

    # 如果 `K0` 是有限域，则使用 `dup_gf_factor` 对 `f` 进行因式分解
    if K0.is_FiniteField:
        coeff, factors = dup_gf_factor(f, K0)
    # 如果K0是代数环，则使用dup_ext_factor函数进行因式分解
    elif K0.is_Algebraic:
        coeff, factors = dup_ext_factor(f, K0)
    # 如果K0是高斯整数环，则使用dup_zz_i_factor函数进行因式分解
    elif K0.is_GaussianRing:
        coeff, factors = dup_zz_i_factor(f, K0)
    # 如果K0是高斯有理数域，则使用dup_qq_i_factor函数进行因式分解
    elif K0.is_GaussianField:
        coeff, factors = dup_qq_i_factor(f, K0)
    else:
        # 如果K0不是精确的，将K0转换为精确的，并将f从K0_inexact转换为K0
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dup_convert(f, K0_inexact, K0)
        else:
            K0_inexact = None

        # 如果K0是域（字段）
        if K0.is_Field:
            # 获取K0的环，并消除f的分母
            K = K0.get_ring()
            denom, f = dup_clear_denoms(f, K0, K)
            # 将f从K0转换为K
            f = dup_convert(f, K0, K)
        else:
            K = K0

        # 如果K是整数环（ZZ），则使用dup_zz_factor函数进行因式分解
        if K.is_ZZ:
            coeff, factors = dup_zz_factor(f, K)
        # 如果K是多项式环（Poly）
        elif K.is_Poly:
            # 将f从多项式转换为分别乘法表示，并使用dmp_factor_list函数进行因式分解
            f, u = dmp_inject(f, 0, K.dom)
            coeff, factors = dmp_factor_list(f, u, K.dom)

            # 将每个因子重新转换为多项式，并更新factors列表
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, u, K), k)

            # 将coeff从K转换为K.dom
            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            # 如果不支持给定的K0，则抛出DomainError异常
            raise DomainError('factorization not supported over %s' % K0)

        # 如果K0是域（字段）
        if K0.is_Field:
            # 将每个因子从K转换为K0，并更新coeff
            for i, (f, k) in enumerate(factors):
                factors[i] = (dup_convert(f, K, K0), k)

            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            # 如果K0_inexact存在，对因子进行调整以保持精确性
            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dup_max_norm(f, K0)
                    f = dup_quo_ground(f, max_norm, K0)
                    f = dup_convert(f, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    # 如果j非零，则在factors列表的开头插入([K0.one, K0.zero], j)
    if j:
        factors.insert(0, ([K0.one, K0.zero], j))

    # 返回coeff乘以cont和按照某种规则排序后的factors列表
    return coeff * cont, _sort_factors(factors)
# 将单变量多项式在域 `K[x]` 中分解为不可约因子。
def dup_factor_list_include(f, K):
    # 调用 `dup_factor_list` 函数，返回多项式 `f` 的系数和因子列表
    coeff, factors = dup_factor_list(f, K)

    # 如果因子列表为空，则返回由 `coeff` 组成的元组列表
    if not factors:
        return [(dup_strip([coeff]), 1)]
    else:
        # 将首个因子乘以 `coeff` 后添加到结果列表中
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, factors[0][1])] + factors[1:]


# 将多变量多项式在环 `K[X]` 中分解为不可约因子。
def dmp_factor_list(f, u, K0):
    # 如果 `u` 为假（即为零），则调用 `dup_factor_list` 函数进行处理
    if not u:
        return dup_factor_list(f, K0)

    # 使用 `dmp_terms_gcd` 函数计算 `f` 和 `u` 的最大公因式，并更新 `f`
    J, f = dmp_terms_gcd(f, u, K0)
    # 使用 `dmp_ground_primitive` 函数将 `f` 转换为原始的多项式形式，并返回首项和更新后的 `f`
    cont, f = dmp_ground_primitive(f, u, K0)

    # 根据 `K0` 的类型选择合适的分解函数，得到系数 `coeff` 和因子列表 `factors`
    if K0.is_FiniteField:  # 如果 `K0` 是有限域
        coeff, factors = dmp_gf_factor(f, u, K0)
    elif K0.is_Algebraic:  # 如果 `K0` 是代数域
        coeff, factors = dmp_ext_factor(f, u, K0)
    elif K0.is_GaussianRing:  # 如果 `K0` 是高斯整环
        coeff, factors = dmp_zz_i_factor(f, u, K0)
    elif K0.is_GaussianField:  # 如果 `K0` 是高斯域
        coeff, factors = dmp_qq_i_factor(f, u, K0)
    else:
        # 如果 `K0` 不是精确的，将 `K0` 转换为精确域
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dmp_convert(f, u, K0_inexact, K0)
        else:
            K0_inexact = None

        # 根据 `K0` 的类型进行处理
        if K0.is_Field:  # 如果 `K0` 是域
            K = K0.get_ring()

            # 清除 `f` 的分母，并将其转换为 `K` 域中的多项式
            denom, f = dmp_clear_denoms(f, u, K0, K)
            f = dmp_convert(f, u, K0, K)
        else:
            K = K0

        # 根据 `K` 的类型进行处理
        if K.is_ZZ:  # 如果 `K` 是整数环
            levels, f, v = dmp_exclude(f, u, K)
            coeff, factors = dmp_zz_factor(f, v, K)

            # 将因子列表中的每个多项式重新包含到 `levels` 级别中
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_include(f, levels, v, K), k)
        elif K.is_Poly:  # 如果 `K` 是多项式环
            f, v = dmp_inject(f, u, K)

            # 调用 `dmp_factor_list` 函数处理 `f` 和 `v`
            coeff, factors = dmp_factor_list(f, v, K.dom)

            # 将因子列表中的每个多项式 `f` 从 `v` 中弹出，并将其重新注入
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, v, K), k)

            # 将 `coeff` 从 `K.dom` 转换为 `K` 中的元素
            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            # 如果以上情况都不符合，抛出域错误异常
            raise DomainError('factorization not supported over %s' % K0)

        # 如果 `K0` 是域，将因子列表中的每个多项式 `f` 转换为 `K0` 域中的元素
        if K0.is_Field:
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_convert(f, u, K, K0), k)

            # 将 `coeff` 从 `K` 转换为 `K0` 中的元素，并将其除以 `denom`
            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            # 如果 `K0_inexact` 不为空，对因子列表中的每个多项式 `f` 进行处理
            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dmp_max_norm(f, u, K0)
                    f = dmp_quo_ground(f, max_norm, u, K0)
                    f = dmp_convert(f, u, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                # 将 `coeff` 从 `K0_inexact` 转换为 `K0` 中的元素
                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    # 对 `J` 的反向枚举，并将非零项添加到结果因子列表的开头
    for i, j in enumerate(reversed(J)):
        if not j:
            continue

        term = {(0,)*(u - i) + (1,) + (0,)*i: K0.one}
        factors.insert(0, (dmp_from_dict(term, u, K0), j))

    # 返回 `coeff*cont` 和排序后的因子列表 `_sort_factors(factors)`
    return coeff * cont, _sort_factors(factors)


# 将多变量多项式在环 `K[X]` 中分解为不可约因子，并在结果中包含 `K`。
def dmp_factor_list_include(f, u, K):
    # 如果 `u` 为假（即为零），则调用 `dup_factor_list_include` 函数进行处理
    if not u:
        return dup_factor_list_include(f, K)

    # 调用 `dmp_factor_list` 函数，得到系数 `coeff` 和因子列表 `factors`
    coeff, factors = dmp_factor_list(f, u, K)
    # 如果 factors 列表为空
    if not factors:
        # 返回一个包含单个元组的列表，元组由 dmp_ground(coeff, u) 和 1 组成
        return [(dmp_ground(coeff, u), 1)]
    else:
        # 否则，从 factors 列表中取出第一个元素的第一个元素，并与 coeff 相乘，使用 dmp_mul_ground 函数
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        # 返回一个新列表，其中第一个元素是 (g, factors[0][1]) 组成的元组，后续元素与 factors 列表的剩余部分相同
        return [(g, factors[0][1])] + factors[1:]
# 判断给定的一元多项式 f 在域 K 上是否为不可约多项式，即没有在其域上的因子
def dup_irreducible_p(f, K):
    """
    Returns ``True`` if a univariate polynomial ``f`` has no factors
    over its domain.
    """
    # 调用 dmp_irreducible_p 函数，其中 u=0 表示一元多项式，返回其判断结果
    return dmp_irreducible_p(f, 0, K)


# 判断给定的多元多项式 f 在域 K 上是否为不可约多项式，即没有在其域上的因子
def dmp_irreducible_p(f, u, K):
    """
    Returns ``True`` if a multivariate polynomial ``f`` has no factors
    over its domain.
    """
    # 使用 dmp_factor_list 函数计算多项式 f 的因子列表
    _, factors = dmp_factor_list(f, u, K)

    # 如果因子列表为空，则多项式 f 是不可约的
    if not factors:
        return True
    # 如果因子列表长度大于 1，则多项式 f 有多个因子，因此可约
    elif len(factors) > 1:
        return False
    else:
        # 否则，因子列表长度为 1，检查唯一的因子的指数 k 是否为 1，判断多项式 f 是否为不可约的
        _, k = factors[0]
        return k == 1
```