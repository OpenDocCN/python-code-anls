# `D:\src\scipysrc\sympy\sympy\polys\galoistools.py`

```
"""Dense univariate polynomials with coefficients in Galois fields. """

# 从 math 模块中导入 ceil 和 sqrt 函数，并且导入 prod 函数
from math import ceil as _ceil, sqrt as _sqrt, prod

# 从 sympy.core.random 模块中导入 uniform 和 _randint 函数
from sympy.core.random import uniform, _randint
# 从 sympy.external.gmpy 模块中导入 SYMPY_INTS, MPZ 和 invert 函数
from sympy.external.gmpy import SYMPY_INTS, MPZ, invert
# 从 sympy.polys.polyconfig 模块中导入 query 函数
from sympy.polys.polyconfig import query
# 从 sympy.polys.polyerrors 模块中导入 ExactQuotientFailed 异常类
from sympy.polys.polyerrors import ExactQuotientFailed
# 从 sympy.polys.polyutils 模块中导入 _sort_factors 函数
from sympy.polys.polyutils import _sort_factors


def gf_crt(U, M, K=None):
    """
    Chinese Remainder Theorem.

    Given a set of integer residues ``u_0,...,u_n`` and a set of
    co-prime integer moduli ``m_0,...,m_n``, returns an integer
    ``u``, such that ``u = u_i mod m_i`` for ``i = ``0,...,n``.

    Examples
    ========

    Consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_crt

       >>> gf_crt([49, 76, 65], [99, 97, 95], ZZ)
       639985

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    Note: this is a low-level routine with no error checking.

    See Also
    ========

    sympy.ntheory.modular.crt : a higher level crt routine
    sympy.ntheory.modular.solve_congruence

    """
    # 计算所有模数的乘积
    p = prod(M, start=K.one)
    # 初始化结果变量为零
    v = K.zero

    # 对于每一对 (u, m)，其中 u 是余数，m 是模数
    for u, m in zip(U, M):
        # 计算 p/m
        e = p // m
        # 计算 e 和 m 的最大公约数的扩展欧几里得算法
        s, _, _ = K.gcdex(e, m)
        # 更新 v，根据中国剩余定理的公式计算
        v += e*(u*s % m)

    # 返回最终结果 v 模 p 的结果
    return v % p


def gf_crt1(M, K):
    """
    First part of the Chinese Remainder Theorem.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]

    The following two codes have the same result.

    >>> gf_crt(U, M, ZZ)
    639985

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    However, it is faster when we want to fix ``M`` and
    compute for multiple U, i.e. the following cases:

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> Us = [[49, 76, 65], [23, 42, 67]]
    >>> for U in Us:
    ...     print(gf_crt2(U, M, p, E, S, ZZ))
    639985
    236237

    See Also
    ========

    sympy.ntheory.modular.crt1 : a higher level crt routine
    sympy.polys.galoistools.gf_crt
    sympy.polys.galoistools.gf_crt2

    """
    # 初始化空列表 E 和 S
    E, S = [], []
    # 计算所有模数的乘积
    p = prod(M, start=K.one)

    # 对于每一个模数 m
    for m in M:
        # 计算 p/m
        E.append(p // m)
        # 计算模数 m 和 E[-1] 的扩展欧几里得算法的结果
        S.append(K.gcdex(E[-1], m)[0] % m)

    # 返回 p, E, S 这三个值
    return p, E, S


def gf_crt2(U, M, p, E, S, K):
    """
    Second part of the Chinese Remainder Theorem.

    See ``gf_crt1`` for usage.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt2

    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]
    >>> p = 912285
    >>> E = [9215, 9405, 9603]
    >>> S = [62, 24, 12]

    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    See Also
    ========

    sympy.ntheory.modular.crt2 : a higher level crt routine
    sympy.polys.galoistools.gf_crt

    """
    # 初始化结果变量为零
    u = K.zero

    # 对于每一对 (u_i, m_i)，其中 u_i 是余数，m_i 是模数
    for u_i, m_i, e_i, s_i in zip(U, M, E, S):
        # 计算 u_i * s_i * e_i % m_i 的结果，并加到 u 上
        u += u_i * s_i * e_i % m_i

    # 返回 u 模 p 的结果
    return u % p
    sympy.polys.galoistools.gf_crt1
    """
        多项式运算库 sympy.polys 中的 gf_crt1 函数，执行 Galois 域上的 Chinese Remainder Theorem (CRT) 算法。
        输入参数：
        - U: 一组整数列表，代表余数
        - M: 一组整数列表，代表模数
        - E: 一组整数列表，代表对应的权重系数
        - S: 一组整数列表，代表加权后的乘数
    
        输出：
        - 返回通过 CRT 计算出的结果 v % p，其中 p 是一个给定的素数
    
        算法流程：
        - 初始化 v 为零元素
        - 遍历 U, M, E, S 四个列表，分别表示余数、模数、权重系数和加权乘数
        - 对于每组对应的 u, m, e, s，计算 u * s 在模 m 下的乘积，并乘以权重系数 e
        - 将所有计算得到的乘积累加到 v 中
        - 最后将 v 对素数 p 取模并返回结果
    """
def gf_int(a, p):
    """
    Coerce ``a mod p`` to an integer in the range ``[-p/2, p/2]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_int

    >>> gf_int(2, 7)
    2
    >>> gf_int(5, 7)
    -2

    """
    # 如果 a 小于等于 p//2，则直接返回 a
    if a <= p // 2:
        return a
    else:
        # 否则返回 a - p，确保返回值在 [-p/2, p/2] 的范围内
        return a - p


def gf_degree(f):
    """
    Return the leading degree of ``f``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_degree

    >>> gf_degree([1, 1, 2, 0])
    3
    >>> gf_degree([])
    -1

    """
    # 返回多项式 f 的最高次数，空列表表示 -1 次多项式
    return len(f) - 1


def gf_LC(f, K):
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_LC

    >>> gf_LC([3, 0, 1], ZZ)
    3

    """
    # 如果 f 是空的，则返回 K 的零元素
    if not f:
        return K.zero
    else:
        # 否则返回 f 的首项系数
        return f[0]


def gf_TC(f, K):
    """
    Return the trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_TC

    >>> gf_TC([3, 0, 1], ZZ)
    1

    """
    # 如果 f 是空的，则返回 K 的零元素
    if not f:
        return K.zero
    else:
        # 否则返回 f 的末项系数
        return f[-1]


def gf_strip(f):
    """
    Remove leading zeros from ``f``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_strip

    >>> gf_strip([0, 0, 0, 3, 0, 1])
    [3, 0, 1]

    """
    # 如果 f 是空的或者首项不是零，则直接返回 f
    if not f or f[0]:
        return f

    k = 0

    # 找到第一个非零项的索引 k
    for coeff in f:
        if coeff:
            break
        else:
            k += 1

    # 返回从索引 k 开始到末尾的部分
    return f[k:]


def gf_trunc(f, p):
    """
    Reduce all coefficients modulo ``p``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_trunc

    >>> gf_trunc([7, -2, 3], 5)
    [2, 3, 3]

    """
    # 对 f 中的每个系数取模 p，并去除首部的零系数
    return gf_strip([ a % p for a in f ])


def gf_normal(f, p, K):
    """
    Normalize all coefficients in ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_normal

    >>> gf_normal([5, 10, 21, -3], 5, ZZ)
    [1, 2]

    """
    # 将 f 中的每个系数映射为 K 类型后，再取模 p，并去除首部的零系数
    return gf_trunc(list(map(K, f)), p)


def gf_from_dict(f, p, K):
    """
    Create a ``GF(p)[x]`` polynomial from a dict.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_from_dict

    >>> gf_from_dict({10: ZZ(4), 4: ZZ(33), 0: ZZ(-1)}, 5, ZZ)
    [4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4]

    """
    # 找到 f 字典中最大的键 n
    n, h = max(f.keys()), []

    # 根据 n 的类型，选择不同的字典键访问方式
    if isinstance(n, SYMPY_INTS):
        # 如果 n 是整数，按照整数方式处理
        for k in range(n, -1, -1):
            # 将每个系数取模 p 后添加到 h 中
            h.append(f.get(k, K.zero) % p)
    else:
        # 否则，n 应该是元组，按元组方式处理
        (n,) = n

        for k in range(n, -1, -1):
            # 将每个系数取模 p 后添加到 h 中
            h.append(f.get((k,), K.zero) % p)

    # 返回处理后的多项式，并去除首部的零系数
    return gf_trunc(h, p)


def gf_to_dict(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to a dict.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_dict

    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5)
    {0: -1, 4: -2, 10: -1}

    """
    # 创建一个空字典用于存储多项式的非零系数
    d = {}

    # 遍历多项式 f 的每个非零系数
    for i, coeff in enumerate(f):
        if coeff:
            # 如果系数不为零，则将其存入字典，键为 i，并取模 p
            d[i] = coeff % p

    # 如果 symmetric 为 True，则额外存储 f 的末项系数
    if symmetric:
        d[gf_degree(f)] = gf_TC(f, ZZ) % p

    return d
    # 调用 gf_degree 函数计算多项式 f 的次数 n 和空字典 result
    n, result = gf_degree(f), {}

    # 遍历多项式 f 的系数，从高次到低次，k 从 0 到 n
    for k in range(0, n + 1):
        # 如果 symmetric 为 True，则将系数 f[n - k] 转换为有限域上的整数 a
        if symmetric:
            a = gf_int(f[n - k], p)
        # 如果 symmetric 为 False，则直接取系数 f[n - k]
        else:
            a = f[n - k]

        # 如果系数 a 不为零
        if a:
            # 将非零系数 a 存入结果字典 result，键为 k
            result[k] = a

    # 返回最终的结果字典 result
    return result
# 创建一个在有限域 GF(p) 上的多项式，其系数来自整数环 Z
def gf_from_int_poly(f, p):
    return gf_trunc(f, p)


# 将一个在有限域 GF(p) 上的多项式转换为整数环 Z 上的多项式
def gf_to_int_poly(f, p, symmetric=True):
    if symmetric:
        # 对于对称模式，将多项式中每个系数转换为整数环 Z 中的数值
        return [ gf_int(c, p) for c in f ]
    else:
        return f


# 在有限域 GF(p) 上对一个多项式取负
def gf_neg(f, p, K):
    return [ -coeff % p for coeff in f ]


# 在有限域 GF(p) 上计算一个多项式和一个元素 a 的加法
def gf_add_ground(f, a, p, K):
    if not f:
        # 如果多项式 f 为空，则直接返回 a 对 p 取模的结果
        a = a % p
    else:
        # 否则，将多项式的最高次项与 a 相加，并对 p 取模
        a = (f[-1] + a) % p
        
        # 如果多项式长度大于 1，则将新的最高次项加入多项式并返回
        if len(f) > 1:
            return f[:-1] + [a]

    # 如果结果为 0，则返回空列表；否则返回包含结果的列表
    if not a:
        return []
    else:
        return [a]


# 在有限域 GF(p) 上计算一个多项式和一个元素 a 的减法
def gf_sub_ground(f, a, p, K):
    if not f:
        # 如果多项式 f 为空，则直接返回 -a 对 p 取模的结果
        a = -a % p
    else:
        # 否则，将多项式的最高次项与 a 相减，并对 p 取模
        a = (f[-1] - a) % p
        
        # 如果多项式长度大于 1，则将新的最高次项加入多项式并返回
        if len(f) > 1:
            return f[:-1] + [a]

    # 如果结果为 0，则返回空列表；否则返回包含结果的列表
    if not a:
        return []
    else:
        return [a]


# 在有限域 GF(p) 上计算一个多项式和一个元素 a 的乘法
def gf_mul_ground(f, a, p, K):
    if not a:
        # 如果 a 为 0，则直接返回空列表
        return []
    else:
        # 否则，对多项式中的每个系数乘以 a 并对 p 取模
        return [ (a*b) % p for b in f ]


# 在有限域 GF(p) 上计算一个多项式除以一个元素 a 的结果
def gf_quo_ground(f, a, p, K):
    # 使用 K.invert(a, p) 计算 a 在有限域 GF(p) 中的乘法逆元素，然后进行乘法操作
    return gf_mul_ground(f, K.invert(a, p), p, K)


# 在有限域 GF(p) 上计算两个多项式的加法
def gf_add(f, g, p, K):
    pass  # 此函数未实现完整，预期功能是在有限域 GF(p) 上计算两个多项式的加法
    # 如果 f 为空，则返回 g
    if not f:
        return g
    # 如果 g 为空，则返回 f
    if not g:
        return f
    
    # 计算多项式 f 和 g 的次数
    df = gf_degree(f)  # 获取多项式 f 的次数
    dg = gf_degree(g)  # 获取多项式 g 的次数
    
    # 如果 f 和 g 的次数相同，则进行多项式相加并取模运算
    if df == dg:
        # 逐项相加并取模，生成新的多项式
        return gf_strip([ (a + b) % p for a, b in zip(f, g) ])
    else:
        # 计算次数差值
        k = abs(df - dg)
    
        # 根据次数差值调整多项式的长度
        if df > dg:
            h, f = f[:k], f[k:]  # 将多项式 f 拆分为 h 和 f
        else:
            h, g = g[:k], g[k:]  # 将多项式 g 拆分为 h 和 g
    
        # 对剩余部分的多项式进行逐项相加并取模
        return h + [ (a + b) % p for a, b in zip(f, g) ]
# 在有限域GF(p)[x]中，实现多项式的减法
def gf_sub(f, g, p, K):
    # 如果g是空多项式，直接返回f
    if not g:
        return f
    # 如果f是空多项式，返回-g乘以1的结果，即负多项式函数gf_neg的应用
    if not f:
        return gf_neg(g, p, K)

    # 计算多项式f和g的次数
    df = gf_degree(f)
    dg = gf_degree(g)

    # 如果f和g次数相同，进行对应系数相减并取模运算，返回结果的多项式
    if df == dg:
        return gf_strip([ (a - b) % p for a, b in zip(f, g) ])
    else:
        # 计算两个多项式次数的差
        k = abs(df - dg)

        # 根据f和g的次数差分别处理
        if df > dg:
            # 分别取出多项式h和f的前k个系数，其余部分为f
            h, f = f[:k], f[k:]
        else:
            # 对g取前k个系数求负多项式，并赋值给h，g取k后的系数
            h, g = gf_neg(g[:k], p, K), g[k:]

        # 返回结果多项式h与对应系数相减并取模运算的结果
        return h + [ (a - b) % p for a, b in zip(f, g) ]


# 在有限域GF(p)[x]中，实现多项式的乘法
def gf_mul(f, g, p, K):
    # 计算多项式f和g的次数
    df = gf_degree(f)
    dg = gf_degree(g)

    # 计算结果多项式h的最高次数
    dh = df + dg
    # 初始化结果多项式h的系数为0
    h = [0]*(dh + 1)

    # 逐项计算乘积多项式h的系数
    for i in range(0, dh + 1):
        coeff = K.zero

        # 遍历所有可能的乘积项并求和
        for j in range(max(0, i - dg), min(i, df) + 1):
            coeff += f[j]*g[i - j]

        # 对乘积结果取模p，并赋值给结果多项式h的第i个系数
        h[i] = coeff % p

    # 返回处理后的结果多项式h
    return gf_strip(h)


# 在有限域GF(p)[x]中，实现多项式的平方
def gf_sqr(f, p, K):
    # 计算多项式f的次数
    df = gf_degree(f)

    # 计算结果多项式h的最高次数
    dh = 2 * df
    # 初始化结果多项式h的系数为0
    h = [0]*(dh + 1)

    # 逐项计算平方多项式h的系数
    for i in range(0, dh + 1):
        coeff = K.zero

        # 计算j的范围，使得i - j在0到df之间
        jmin = max(0, i - df)
        jmax = min(i, df)

        # 计算元素数目n，并相应调整jmax
        n = jmax - jmin + 1
        jmax = jmin + n // 2 - 1

        # 计算jmin到jmax的平方乘积之和
        for j in range(jmin, jmax + 1):
            coeff += f[j]*f[i - j]

        # 若n为奇数，计算元素elem的平方并合并coeff
        coeff += coeff if n & 1 else f[jmax + 1] ** 2

        # 对乘积结果取模p，并赋值给结果多项式h的第i个系数
        h[i] = coeff % p

    # 返回处理后的结果多项式h
    return gf_strip(h)


# 在有限域GF(p)[x]中，实现多项式的加法乘法组合操作
def gf_add_mul(f, g, h, p, K):
    # 返回f与g乘以h的结果，使用函数gf_add和gf_mul
    return gf_add(f, gf_mul(g, h, p, K), p, K)


# 在有限域GF(p)[x]中，实现多项式的减法乘法组合操作
def gf_sub_mul(f, g, h, p, K):
    # 返回f与g乘以h的结果，使用函数gf_sub和gf_mul
    return gf_sub(f, gf_mul(g, h, p, K), p, K)


# 在有限域GF(p)[x]中，扩展分解结果的展开操作
def gf_expand(F, p, K):
    # 扩展 :func:`~.factor` 在GF(p)[x]中的结果
    return NotImplemented  # 未实现的部分，暂时返回NotImplemented
    # 如果变量 F 是一个元组，则解构元组，将其第一个元素赋给 lc，第二个元素赋给 F
    if isinstance(F, tuple):
        lc, F = F
    # 否则，将变量 K.one 赋给 lc
    else:
        lc = K.one
    
    # 将 lc 放入列表中，形成列表 g，作为初始值
    g = [lc]
    
    # 遍历元组 F 中的每对元素 (f, k)
    for f, k in F:
        # 使用函数 gf_pow 对 f 进行模 p 的幂运算，得到结果 f
        f = gf_pow(f, k, p, K)
        # 使用函数 gf_mul 对列表 g 和 f 进行模 p 的乘法运算，更新列表 g
        g = gf_mul(g, f, p, K)
    
    # 返回最终的列表 g 作为结果
    return g
```python`
# 计算多项式在有限域 GF(p)[x] 上的除法余数
def gf_div(f, g, p, K):
    """
    Division with remainder in ``GF(p)[x]``.

    Given univariate polynomials ``f`` and ``g`` with coefficients in a
    finite field with ``p`` elements, returns polynomials ``q`` and ``r``
    (quotient and remainder) such that ``f = q*g + r``.

    Consider polynomials ``x**3 + x + 1`` and ``x**2 + x`` in GF(2)::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_div, gf_add_mul

       >>> gf_div(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       ([1, 1], [1])

    As result we obtained quotient ``x + 1`` and remainder ``1``, thus::

       >>> gf_add_mul(ZZ.map([1]), ZZ.map([1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       [1, 0, 1, 1]

    References
    ==========

    .. [1] [Monagan93]_
    .. [2] [Gathen99]_

    """
    # 计算 f 和 g 的最高次数
    df = gf_degree(f)
    dg = gf_degree(g)

    # 如果除数 g 是零多项式，则抛出异常
    if not g:
        raise ZeroDivisionError("polynomial division")
    # 如果被除多项式 f 的次数小于除数 g 的次数，则直接返回 ([], f)
    elif df < dg:
        return [], f

    # 计算 g[0] 在有限域 GF(p) 中的乘法逆元
    inv = K.invert(g[0], p)

    # 初始化 h 为 f 的副本，dq 为商的次数，dr 为余数的次数
    h, dq, dr = list(f), df - dg, dg - 1

    # 开始多项式除法的主循环
    for i in range(0, df + 1):
        coeff = h[i]

        # 更新余数的每一项
        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        # 如果当前位置在商的次数内，则乘以 g[0] 的乘法逆元
        if i <= dq:
            coeff *= inv

        # 对结果取模 p
        h[i] = coeff % p

    # 返回商和剩余项
    return h[:dq + 1], gf_strip(h[dq + 1:])


def gf_rem(f, g, p, K):
    """
    Compute polynomial remainder in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rem

    >>> gf_rem(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1]

    """
    # 直接调用 gf_div 函数并返回余数部分
    return gf_div(f, g, p, K)[1]


def gf_quo(f, g, p, K):
    """
    Compute exact quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo

    >>> gf_quo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1, 1]
    >>> gf_quo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    """
    # 计算 f 和 g 的最高次数
    df = gf_degree(f)
    dg = gf_degree(g)

    # 如果除数 g 是零多项式，则抛出异常
    if not g:
        raise ZeroDivisionError("polynomial division")
    # 如果被除多项式 f 的次数小于除数 g 的次数，则直接返回空列表
    elif df < dg:
        return []

    # 计算 g[0] 在有限域 GF(p) 中的乘法逆元
    inv = K.invert(g[0], p)

    # 初始化 h 为 f 的副本，dq 为商的次数，dr 为余数的次数
    h, dq, dr = f[:], df - dg, dg - 1

    # 开始多项式除法的主循环
    for i in range(0, dq + 1):
        coeff = h[i]

        # 更新余数的每一项
        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        # 对结果取模 p，并乘以 g[0] 的乘法逆元
    else:
        # 如果上述条件都不满足，则抛出 ExactQuotientFailed 异常
        raise ExactQuotientFailed(f, g)
# 计算多项式 ``f`` 与 ``g`` 的次数，主要用于计算 Frobenius 映射
m = gf_degree(g)

# 如果多项式 ``f`` 的次数大于等于 ``g`` 的次数，则将 ``f`` 对 ``g`` 取模
if gf_degree(f) >= m:
    f = gf_rem(f, g, p, K)

# 如果 ``f`` 为空多项式，则返回空列表
if not f:
    return []

# 计算多项式 ``f`` 的次数
n = gf_degree(f)

# 初始化结果列表，并将多项式 ``f`` 的最高次系数加入到结果列表中
sf = [f[-1]]
    # 对于整数范围内的每一个 i，执行以下操作，范围从 1 到 n+1（不包括 n+1）
    for i in range(1, n + 1):
        # 计算 b[i] 与 f[n-i] 的乘积，再对结果执行模 p 运算，使用环 K 上的多项式乘法 gf_mul_ground
        v = gf_mul_ground(b[i], f[n - i], p, K)
        # 将计算结果 sf 与 v 相加，结果也在环 K 上进行模 p 运算，使用环 K 上的多项式加法 gf_add
        sf = gf_add(sf, v, p, K)
    # 返回最终结果 sf，这是多项式 f 在域 K 上对 b 的求和结果
    return sf
def _gf_pow_pnm1d2(f, n, g, b, p, K):
    """
    utility function for ``gf_edf_zassenhaus``
    Compute ``f**((p**n - 1) // 2)`` in ``GF(p)[x]/(g)``
    ``f**((p**n - 1) // 2) = (f*f**p*...*f**(p**n - 1))**((p - 1) // 2)``
    """
    # 对输入的多项式 f 进行模 g 的余数运算
    f = gf_rem(f, g, p, K)
    # 初始化 h 和 r 为 f
    h = f
    r = f
    # 迭代 n-1 次，计算 h 和 r 的乘积
    for i in range(1, n):
        # 应用 Frobenius 映射到 h 上
        h = gf_frobenius_map(h, g, b, p, K)
        # 计算 r = r * h 并模 g
        r = gf_mul(r, h, p, K)
        r = gf_rem(r, g, p, K)

    # 计算 r 的 (p-1)//2 次幂，并返回结果
    res = gf_pow_mod(r, (p - 1)//2, g, p, K)
    return res


def gf_pow_mod(f, n, g, p, K):
    """
    Compute ``f**n`` in ``GF(p)[x]/(g)`` using repeated squaring.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]`` and a non-negative
    integer ``n``, efficiently computes ``f**n (mod g)`` i.e. the remainder
    of ``f**n`` from division by ``g``, using the repeated squaring algorithm.
    """
    # 如果 n 为 0，则返回 1
    if not n:
        return [K.one]
    # 如果 n 为 1，则返回 f 对 g 的模余数
    elif n == 1:
        return gf_rem(f, g, p, K)
    # 如果 n 为 2，则返回 f 的平方对 g 的模余数
    elif n == 2:
        return gf_rem(gf_sqr(f, p, K), g, p, K)

    # 初始化 h 为 1
    h = [K.one]

    while True:
        # 如果 n 是奇数，则更新 h 为 h * f 对 g 的模余数，并将 n 减 1
        if n & 1:
            h = gf_mul(h, f, p, K)
            h = gf_rem(h, g, p, K)
            n -= 1

        # 将 n 右移 1 位
        n >>= 1

        # 如果 n 变为 0，则跳出循环
        if not n:
            break

        # 更新 f 为 f 的平方对 g 的模余数
        f = gf_sqr(f, p, K)
        f = gf_rem(f, g, p, K)

    return h


def gf_gcd(f, g, p, K):
    """
    Euclidean Algorithm in ``GF(p)[x]``.
    """
    # 使用欧几里得算法计算多项式 f 和 g 在 GF(p)[x] 上的最大公因数
    while g:
        f, g = g, gf_rem(f, g, p, K)

    return gf_monic(f, p, K)[1]


def gf_lcm(f, g, p, K):
    """
    Compute polynomial LCM in ``GF(p)[x]``.
    """
    # 如果 f 或 g 为空，则返回空列表
    if not f or not g:
        return []

    # 计算 f 和 g 的最小公倍数并返回
    h = gf_quo(gf_mul(f, g, p, K),
               gf_gcd(f, g, p, K), p, K)

    return gf_monic(h, p, K)[1]


def gf_cofactors(f, g, p, K):
    """
    Compute polynomial GCD and cofactors in ``GF(p)[x]``.
    """
    # 如果 f 和 g 同时为空，则返回空元组
    if not f and not g:
        return ([], [], [])

    # 计算 f 和 g 的最大公因数 h
    h = gf_gcd(f, g, p, K)

    # 返回 f 除以 h 的商，以及 g 除以 h 的商，作为 cofactors
    return (h, gf_quo(f, h, p, K),
            gf_quo(g, h, p, K))


def gf_gcdex(f, g, p, K):
    """
    Extended Euclidean Algorithm in ``GF(p)[x]``.
    """
    # 使用扩展欧几里得算法计算 f 和 g 在 GF(p)[x] 上的最大公因数
    while g:
        f, g = g, gf_rem(f, g, p, K)

    return gf_monic(f, p, K)[1]
    # 如果 f 和 g 都为空，则返回单位元素列表和空列表作为结果
    if not (f or g):
        return [K.one], [], []

    # 将 f 和 g 分别转化为首一形式，返回首一形式的结果以及剩余项 r0 和 r1
    p0, r0 = gf_monic(f, p, K)
    p1, r1 = gf_monic(g, p, K)

    # 如果 f 为空，则返回空列表、g 的逆元素以及剩余项 r1
    if not f:
        return [], [K.invert(p1, p)], r1
    # 如果 g 为空，则返回 f 的逆元素、空列表以及剩余项 r0
    if not g:
        return [K.invert(p0, p)], [], r0

    # 初始化 s0 和 t1 为 f 的逆元素和 g 的逆元素，s1 和 t0 为空列表
    s0, s1 = [K.invert(p0, p)], []
    t0, t1 = [], [K.invert(p1, p)]

    # 进入循环，执行扩展欧几里得算法
    while True:
        # 用 r0 除以 r1 得到商 Q 和余数 R
        Q, R = gf_div(r0, r1, p, K)

        # 如果余数 R 为空，则退出循环
        if not R:
            break

        # 将 R 转化为首一形式，lc 是其首项系数，更新 r0 为 r1，r1 为 R
        (lc, r1), r0 = gf_monic(R, p, K), r1

        # 计算 lc 在有限域 p 上的乘法逆元素
        inv = K.invert(lc, p)

        # 更新 s 和 t 的值
        s = gf_sub_mul(s0, s1, Q, p, K)
        t = gf_sub_mul(t0, t1, Q, p, K)

        # 计算 s 的乘以 inv 在有限域 p 上的结果，更新 s1 和 s0
        s1, s0 = gf_mul_ground(s, inv, p, K), s1
        # 计算 t 的乘以 inv 在有限域 p 上的结果，更新 t1 和 t0
        t1, t0 = gf_mul_ground(t, inv, p, K), t1

    # 返回最终计算得到的 s1, t1 和 r1，分别代表多项式 f 和 g 的最大公因式的系数以及余数
    return s1, t1, r1
# 计算多项式 f 在有限域 GF(p) 上的导数
def gf_diff(f, p, K):
    # 计算多项式 f 的次数
    df = gf_degree(f)
    
    # 初始化结果多项式 h 为次数 df 的零多项式
    h, n = [K.zero]*df, df
    
    # 遍历 f 的系数，从高次到低次
    for coeff in f[:-1]:
        # 将系数乘以当前的次数 n，并在有限域 GF(p) 上取模
        coeff *= K(n)
        coeff %= p
        
        # 如果系数不为零，则将其放入结果多项式 h 的对应位置
        if coeff:
            h[df - n] = coeff
        
        # 减少次数 n
        n -= 1
    
    # 去除结果多项式 h 的高次零系数，并返回结果
    return gf_strip(h)
    # 在有限域 GF(p)[x]/(f) 的商环中，找到满足 b = c**t (mod f) 的元素 c，其中 t 是正整数次幂，并且一个正整数 n，
    # 返回一个映射：
    #   a -> a**t**n, a + a**t + a**t**2 + ... + a**t**n (mod f)
    # 在因式分解的上下文中，b = x**p mod f，c = x mod f。
    # 这样我们可以在等次因式分解程序中高效地计算迹多项式，比使用迭代的Frobenius算法在大次数情况下更快。
    u = gf_compose_mod(a, b, f, p, K)  # 计算 gf_compose_mod(a, b, f, p, K)，返回结果存入 u
    v = b  # 将 b 赋值给 v

    if n & 1:
        U = gf_add(a, u, p, K)  # 如果 n 是奇数，计算 gf_add(a, u, p, K)，返回结果存入 U
        V = b  # 将 b 赋值给 V
    else:
        U = a  # 如果 n 是偶数，将 a 赋值给 U
        V = c  # 将 c 赋值给 V

    n >>= 1  # 将 n 右移一位（相当于除以 2）

    while n:
        u = gf_add(u, gf_compose_mod(u, v, f, p, K), p, K)  # 计算 gf_add(u, gf_compose_mod(u, v, f, p, K))，更新 u
        v = gf_compose_mod(v, v, f, p, K)  # 计算 gf_compose_mod(v, v, f, p, K)，更新 v

        if n & 1:
            U = gf_add(U, gf_compose_mod(u, V, f, p, K), p, K)  # 如果 n 是奇数，计算 gf_add(U, gf_compose_mod(u, V, f, p, K))，更新 U
            V = gf_compose_mod(v, V, f, p, K)  # 计算 gf_compose_mod(v, V, f, p, K)，更新 V

        n >>= 1  # 将 n 右移一位（相当于除以 2）

    return gf_compose_mod(a, V, f, p, K), U  # 返回 gf_compose_mod(a, V, f, p, K) 和 U
def _gf_trace_map(f, n, g, b, p, K):
    """
    utility for ``gf_edf_shoup``
    """
    # 计算 f mod g 在有限域 GF(p) 上的余数
    f = gf_rem(f, g, p, K)
    h = f  # 将 h 初始化为 f
    r = f  # 将 r 初始化为 f
    for i in range(1, n):
        # 计算 Frobenius 映射后的 h
        h = gf_frobenius_map(h, g, b, p, K)
        # 将 h 添加到 r 上
        r = gf_add(r, h, p, K)
        # 计算 r mod g 在有限域 GF(p) 上的余数
        r = gf_rem(r, g, p, K)
    return r  # 返回计算得到的 r


def gf_random(n, p, K):
    """
    Generate a random polynomial in ``GF(p)[x]`` of degree ``n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_random
    >>> gf_random(10, 5, ZZ) #doctest: +SKIP
    [1, 2, 3, 2, 1, 1, 1, 2, 0, 4, 2]

    """
    pi = int(p)
    # 生成一个随机的 GF(p) 上的多项式，其次数为 n
    return [K.one] + [ K(int(uniform(0, pi))) for i in range(0, n) ]


def gf_irreducible(n, p, K):
    """
    Generate random irreducible polynomial of degree ``n`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible
    >>> gf_irreducible(10, 5, ZZ) #doctest: +SKIP
    [1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]

    """
    while True:
        # 生成一个随机的 GF(p) 上的多项式
        f = gf_random(n, p, K)
        # 检查这个多项式是否是不可约的
        if gf_irreducible_p(f, p, K):
            return f


def gf_irred_p_ben_or(f, p, K):
    """
    Ben-Or's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_ben_or

    >>> gf_irred_p_ben_or(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_ben_or(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)
    if n < 5:
        H = h = gf_pow_mod([K.one, K.zero], p, f, p, K)

        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)

            # 判断 f 和 g 的最大公因式是否为 1
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_compose_mod(h, H, f, p, K)
            else:
                return False
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            # 判断 f 和 g 的最大公因式是否为 1
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_frobenius_map(h, f, b, p, K)
            else:
                return False

    return True


def gf_irred_p_rabin(f, p, K):
    """
    Rabin's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_rabin

    >>> gf_irred_p_rabin(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_rabin(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)

    x = [K.one, K.zero]

    from sympy.ntheory import factorint

    indices = { n//d for d in factorint(n) }

    b = gf_frobenius_monomial_base(f, p, K)
    h = b[1]
    # 判断 f 和 h 的最大公因式是否为 1
    # 这里没有完整的代码，需要补全
    # 对于范围从1到n-1的每个整数i进行迭代
    for i in range(1, n):
        # 如果i在给定的索引集合indices中
        if i in indices:
            # 计算多项式g，作为h除以x的结果，使用域K中的p进行操作
            g = gf_sub(h, x, p, K)
            
            # 如果多项式f和g在域K中的p上的最大公因数不是1，则返回False
            if gf_gcd(f, g, p, K) != [K.one]:
                return False
    
        # 应用Frobenius映射到多项式h上，使用多项式f、常数b和域K中的p进行操作
        h = gf_frobenius_map(h, f, b, p, K)
    
    # 返回判断结果，即判断多项式h是否等于多项式x
    return h == x
# 定义一个字典，将字符串方法名映射到对应的不可约多项式判定函数
_irred_methods = {
    'ben-or': gf_irred_p_ben_or,
    'rabin': gf_irred_p_rabin,
}


def gf_irreducible_p(f, p, K):
    """
    在 ``GF(p)[x]`` 中测试多项式 ``f`` 是否为不可约多项式。

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible_p

    >>> gf_irreducible_p(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irreducible_p(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    # 查询设置中的不可约多项式判定方法
    method = query('GF_IRRED_METHOD')

    # 根据查询到的方法选择对应的函数来判断多项式是否不可约
    if method is not None:
        irred = _irred_methods[method](f, p, K)
    else:
        irred = gf_irred_p_rabin(f, p, K)

    return irred


def gf_sqf_p(f, p, K):
    """
    如果 ``f`` 在 ``GF(p)[x]`` 中是平方自由的则返回 ``True``。

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_p

    >>> gf_sqf_p(ZZ.map([3, 2, 4]), 5, ZZ)
    True
    >>> gf_sqf_p(ZZ.map([2, 4, 4, 2, 2, 1, 4]), 5, ZZ)
    False

    """
    # 规范化 f，确保其首项系数为单位元
    _, f = gf_monic(f, p, K)

    # 如果 f 是零多项式，则返回 True
    if not f:
        return True
    else:
        # 否则，检查 f 和其导数在模 p 的意义下的最大公因子是否为单位元
        return gf_gcd(f, gf_diff(f, p, K), p, K) == [K.one]


def gf_sqf_part(f, p, K):
    """
    返回 ``GF(p)[x]`` 多项式的平方自由部分。

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_part

    >>> gf_sqf_part(ZZ.map([1, 1, 3, 0, 1, 0, 2, 2, 1]), 5, ZZ)
    [1, 4, 3]

    """
    # 获取多项式的平方自由列表，并将其合并为一个多项式
    _, sqf = gf_sqf_list(f, p, K)

    g = [K.one]

    # 将所有平方自由部分的乘积合并为一个多项式
    for f, _ in sqf:
        g = gf_mul(g, f, p, K)

    return g


def gf_sqf_list(f, p, K, all=False):
    """
    返回 ``GF(p)[x]`` 多项式的平方自由分解。

    给定 ``GF(p)[x]`` 中的多项式 ``f``，返回其首项系数和平方自由分解
    ``f_1**e_1 f_2**e_2 ... f_k**e_k``，其中所有 ``f_i`` 都是首项系数为1的
    单位多项式，对于任意的 ``i != j``，``(f_i, f_j)`` 是互素的，且 ``e_1 ... e_k``
    按升序排列。所有平凡项（即 ``f_i = 1``）不包含在输出中。

    考虑在 ``GF(11)[x]`` 中多项式 ``f = x**11 + 1`` 的情况::

       >>> from sympy.polys.domains import ZZ

       >>> from sympy.polys.galoistools import (
       ...     gf_from_dict, gf_diff, gf_sqf_list, gf_pow,
       ... )
       ... # doctest: +NORMALIZE_WHITESPACE

       >>> f = gf_from_dict({11: ZZ(1), 0: ZZ(1)}, 11, ZZ)

    注意到 ``f'(x) = 0``::

       >>> gf_diff(f, 11, ZZ)
       []

    这种现象不会在特征为零的情况下发生。然而，我们仍然可以使用 ``gf_sqf()`` 计算 ``f`` 的平方自由分解::

       >>> gf_sqf_list(f, 11, ZZ)
       (1, [([1, 1], 11)])

    我们得到了因子分解 ``f = (x + 1)**11``。这是正确的因为::

       >>> gf_pow([1, 1], 11, 11, ZZ) == f
       True

    References
    ==========

    .. [1] [Geddes92]_

    """
    n, sqf, factors, r = 1, False, [], int(p)

    # 规范化 f，确保其首项系数为单位元
    lc, f = gf_monic(f, p, K)
    # 如果多项式 f 的次数小于 1，则返回 lc 和空列表
    if gf_degree(f) < 1:
        return lc, []

    # 进入一个无限循环，直到 break 被执行
    while True:
        # 对多项式 f 求关于 p 的微分，结果存储在 F 中
        F = gf_diff(f, p, K)

        # 如果微分结果不为空列表
        if F != []:
            # 计算 f 和 F 的最大公因数，并将结果存储在 g 中
            g = gf_gcd(f, F, p, K)
            # 计算 f 除以 g 的商，并将结果存储在 h 中
            h = gf_quo(f, g, p, K)

            # 初始化指数 i 为 1
            i = 1

            # 进入一个循环，直到 h 等于 [K.one]
            while h != [K.one]:
                # 计算 g 和 h 的最大公因数，并将结果存储在 G 中
                G = gf_gcd(g, h, p, K)
                # 计算 h 除以 G 的商，并将结果存储在 H 中
                H = gf_quo(h, G, p, K)

                # 如果 H 的次数大于 0，则将 (H, i*n) 添加到 factors 列表中
                if gf_degree(H) > 0:
                    factors.append((H, i*n))

                # 更新 g, h, i 的值
                g, h, i = gf_quo(g, G, p, K), G, i + 1

            # 如果 g 等于 [K.one]，则设置 sqf 为 True；否则更新 f 的值为 g
            if g == [K.one]:
                sqf = True
            else:
                f = g

        # 如果 sqf 不为 True
        if not sqf:
            # 计算多项式 f 的次数除以 r 的结果，存储在 d 中
            d = gf_degree(f) // r

            # 更新多项式 f 的系数，每隔 r 个取一个
            for i in range(0, d + 1):
                f[i] = f[i*r]

            # 更新多项式 f 的值为前 d+1 项，更新 n 为 n*r
            f, n = f[:d + 1], n*r
        else:
            # 如果 sqf 为 True，跳出当前循环
            break

    # 如果 all 为 True，则抛出 ValueError 异常
    if all:
        raise ValueError("'all=True' is not supported yet")

    # 返回 lc 和 factors
    return lc, factors
# 计算 Berlekamp 的 Q 矩阵
def gf_Qmatrix(f, p, K):
    """
    计算 Berlekamp 的 Q 矩阵。

    Parameters
    ==========
    f : list
        输入多项式的系数列表，代表 GF(p)[x] 中的多项式。
    p : int
        有限域 GF(p) 中的素数。
    K : object
        多项式系数的域，如 ZZ（整数环）。

    Returns
    =======
    Q : list of lists
        Q 矩阵，其中 Q[i][j] 表示第 i 行第 j 列的值。

    Examples
    ========
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix

    >>> gf_Qmatrix([3, 2, 4], 5, ZZ)
    [[1, 0],
     [3, 4]]

    >>> gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 0, 0],
     [0, 4, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 4]]

    """
    # 计算多项式的次数和域的大小
    n, r = gf_degree(f), int(p)

    # 初始化 Q 矩阵
    q = [K.one] + [K.zero]*(n - 1)
    Q = [list(q)] + [[]]*(n - 1)

    # 计算 Q 矩阵的每个元素
    for i in range(1, (n - 1)*r + 1):
        # 计算新的行 qq 和常数 c
        qq, c = [(-q[-1]*f[-1]) % p], q[-1]

        # 计算 qq 的每个元素
        for j in range(1, n):
            qq.append((q[j - 1] - c*f[-j - 1]) % p)

        # 如果符合条件则更新 Q 矩阵
        if not (i % r):
            Q[i//r] = list(qq)

        # 更新 q 为新的 qq 行
        q = qq

    return Q


# 计算 Q 矩阵的核的基
def gf_Qbasis(Q, p, K):
    """
    计算 Q 矩阵的核的基。

    Parameters
    ==========
    Q : list of lists
        Q 矩阵，其中 Q[i][j] 表示第 i 行第 j 列的值。
    p : int
        有限域 GF(p) 中的素数。
    K : object
        多项式系数的域，如 ZZ（整数环）。

    Returns
    =======
    basis : list of lists
        核的基，其中每个元素是 GF(p) 上的一个向量。

    Examples
    ========
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix, gf_Qbasis

    >>> gf_Qbasis(gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ), 5, ZZ)
    [[1, 0, 0, 0], [0, 0, 1, 0]]

    >>> gf_Qbasis(gf_Qmatrix([3, 2, 4], 5, ZZ), 5, ZZ)
    [[1, 0]]

    """
    # 复制 Q 矩阵的内容到新的变量 Q
    Q, n = [ list(q) for q in Q ], len(Q)

    # 调整 Q 矩阵的对角线元素
    for k in range(0, n):
        Q[k][k] = (Q[k][k] - K.one) % p

    # 利用高斯消元法计算 Q 矩阵的逆
    for k in range(0, n):
        for i in range(k, n):
            if Q[k][i]:
                break
        else:
            continue

        # 计算 Q[k][i] 的逆元素
        inv = K.invert(Q[k][i], p)

        # 更新 Q 矩阵的每一行
        for j in range(0, n):
            Q[j][i] = (Q[j][i]*inv) % p

        # 交换 Q 矩阵的列
        for j in range(0, n):
            t = Q[j][k]
            Q[j][k] = Q[j][i]
            Q[j][i] = t

        # 使用 Gauss 消元法更新 Q 矩阵的每一行
        for i in range(0, n):
            if i != k:
                q = Q[k][i]

                for j in range(0, n):
                    Q[j][i] = (Q[j][i] - Q[j][k]*q) % p

    # 调整 Q 矩阵的对角线和非对角线元素
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Q[i][j] = (K.one - Q[i][j]) % p
            else:
                Q[i][j] = (-Q[i][j]) % p

    # 提取非零行作为基
    basis = []

    for q in Q:
        if any(q):
            basis.append(q)

    return basis


# 计算多项式的贝尔勒坎普因式分解
def gf_berlekamp(f, p, K):
    """
    在小素数 p 的有限域 GF(p)[x] 中分解平方自由多项式 f。

    Parameters
    ==========
    f : list
        输入多项式的系数列表，代表 GF(p)[x] 中的多项式。
    p : int
        有限域 GF(p) 中的素数。
    K : object
        多项式系数的域，如 ZZ（整数环）。

    Returns
    =======
    factors : list of lists
        f 的因式分解，每个因子是一个系数列表。

    Examples
    ========
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_berlekamp

    >>> gf_berlekamp([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 2], [1, 0, 3]]

    """
    # 计算 Q 矩阵
    Q = gf_Qmatrix(f, p, K)
    # 计算 Q 矩阵的基
    V = gf_Qbasis(Q, p, K)

    # 对基向量进行逆序和去除零元素处理
    for i, v in enumerate(V):
        V[i] = gf_strip(list(reversed(v)))

    # 初始化因子列表
    factors = [f]
    # 对于 V 中索引从 1 到 len(V)-1 的每个元素 k 进行循环处理
    for k in range(1, len(V)):
        # 遍历 factors 列表的副本，以便在循环中修改 factors
        for f in list(factors):
            # 初始化 s 为零元素
            s = K.zero

            # 当 s 小于 p 时循环执行以下操作
            while s < p:
                # 计算 V[k] 减去 s 对 p 取模的结果，并返回结果多项式 g
                g = gf_sub_ground(V[k], s, p, K)
                # 计算 f 与 g 对 p 取模的最大公因数，并返回结果多项式 h
                h = gf_gcd(f, g, p, K)

                # 如果 h 不是 [K.one] 且不等于 f，则执行以下操作
                if h != [K.one] and h != f:
                    # 从 factors 中移除 f
                    factors.remove(f)
                    # 用 f 除以 h 对 p 取模的商，更新 f
                    f = gf_quo(f, h, p, K)
                    # 将新的多项式 f 和 h 添加到 factors 中
                    factors.extend([f, h])

                # 如果 factors 的长度等于 V 的长度，则返回 factors 的排序版本
                if len(factors) == len(V):
                    return _sort_factors(factors, multiple=False)

                # s 自增 K.one
                s += K.one

    # 如果循环完成后仍未返回，则返回 factors 的排序版本
    return _sort_factors(factors, multiple=False)
def gf_ddf_zassenhaus(f, p, K):
    """
    Cantor-Zassenhaus: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1 ... f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    Consider the polynomial ``x**15 - 1`` in ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_from_dict

       >>> f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)

    Distinct degree factorization gives::

       >>> from sympy.polys.galoistools import gf_ddf_zassenhaus

       >>> gf_ddf_zassenhaus(f, 11, ZZ)
       [([1, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    which means ``x**15 - 1 = (x**5 - 1) (x**10 + x**5 + 1)``. To obtain
    factorization into irreducibles, use equal degree factorization
    procedure (EDF) with each of the factors.

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_

    """
    # 初始化变量 i 为 1，g 为 [1, 0]，factors 为一个空列表
    i, g, factors = 1, [K.one, K.zero], []

    # 计算多项式 f 的 Frobenius 基
    b = gf_frobenius_monomial_base(f, p, K)
    # 当 2*i 小于等于 f 的次数时，执行循环
    while 2*i <= gf_degree(f):
        # 计算 g 的 Frobenius 映射
        g = gf_frobenius_map(g, f, b, p, K)
        # 计算 h 为 g 和 (x - 1) 的最大公因子
        h = gf_gcd(f, gf_sub(g, [K.one, K.zero], p, K), p, K)

        # 如果 h 不是 [1]，说明找到了一个因子
        if h != [K.one]:
            factors.append((h, i))

            # 更新 f 为 f 除以 h 的商
            f = gf_quo(f, h, p, K)
            # 更新 g 为 g 除以 f 的余数
            g = gf_rem(g, f, p, K)
            # 更新 Frobenius 基为新的 f 的 Frobenius 基
            b = gf_frobenius_monomial_base(f, p, K)

        # 增加 i 的值
        i += 1

    # 如果最终 f 不等于 [1]，将剩余的 f 加入因子列表中
    if f != [K.one]:
        return factors + [(f, gf_degree(f))]
    else:
        return factors


def gf_edf_zassenhaus(f, n, p, K):
    """
    Cantor-Zassenhaus: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and
    an integer ``n``, such that ``n`` divides ``deg(f)``, returns all
    irreducible factors ``f_1,...,f_d`` of ``f``, each of degree ``n``.
    EDF procedure gives complete factorization over Galois fields.

    Consider the square-free polynomial ``f = x**3 + x**2 + x + 1`` in
    ``GF(5)[x]``. Let's compute its irreducible factors of degree one::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_edf_zassenhaus

       >>> gf_edf_zassenhaus([1,1,1,1], 1, 5, ZZ)
       [[1, 1], [1, 2], [1, 3]]

    Notes
    =====

    The case p == 2 is handled by Cohen's Algorithm 3.4.8. The case p odd is
    as in Geddes Algorithm 8.9 (or Cohen's Algorithm 3.4.6).

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_ Algorithm 8.9
    .. [3] [Cohen93]_ Algorithm 3.4.8

    """
    # 初始化 factors 为包含 f 的列表
    factors = [f]

    # 如果 f 的次数小于等于 n，则直接返回 factors
    if gf_degree(f) <= n:
        return factors

    # 计算 N 为 f 的次数除以 n 的商
    N = gf_degree(f) // n
    # 如果 p 不等于 2，则计算多项式 f 的 Frobenius 基
    if p != 2:
        b = gf_frobenius_monomial_base(f, p, K)

    # 初始化 t 为 [1, 0]
    t = [K.one, K.zero]
    # 当因子列表长度小于 N 时循环执行以下代码块
    while len(factors) < N:
        # 如果 p 等于 2，则执行以下代码块
        if p == 2:
            # 初始化 h、r、t 为相同的值
            h = r = t

            # 对于 i 在范围 [0, n-2] 的循环
            for i in range(n - 1):
                # 计算 r 的平方模 f 在有限域 p 上的结果，并更新 r
                r = gf_pow_mod(r, 2, f, p, K)
                # 计算 h 与 r 的和，并更新 h
                h = gf_add(h, r, p, K)

            # 计算 f 与 h 在有限域 p 上的最大公约数
            g = gf_gcd(f, h, p, K)
            # t 增加两个 K.zero 元素
            t += [K.zero, K.zero]
        # 如果 p 不等于 2，则执行以下代码块
        else:
            # 生成一个长度为 2*n-1 的随机元素 r
            r = gf_random(2 * n - 1, p, K)
            # 计算 _gf_pow_pnm1d2(r, n, f, b, p, K) 的结果赋给 h
            h = _gf_pow_pnm1d2(r, n, f, b, p, K)
            # 计算 f 与 h-1 在有限域 p 上的最大公约数
            g = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)

        # 如果 g 不等于 [K.one] 并且 g 不等于 f，则执行以下代码块
        if g != [K.one] and g != f:
            # 将 g 的因子分解结果与 gf_quo(f, g, p, K) 的因子分解结果连接起来，赋给 factors
            factors = gf_edf_zassenhaus(g, n, p, K) \
                + gf_edf_zassenhaus(gf_quo(f, g, p, K), n, p, K)

    # 返回经过排序的因子列表 factors，multiple 参数为 False
    return _sort_factors(factors, multiple=False)
# 给定的多项式 ``f`` 在有限域 ``GF(p)[x]`` 中，执行Kaltofen-Shoup算法的确定性分别度因子分解。
# 算法返回多项式 ``f`` 的部分分别度因子分解 ``f_1,...,f_d``，其中 ``deg(f_i) != deg(f_j)`` 对于 ``i != j``。
# 结果以列表形式返回，每个元素是一个二元组 ``(f_i, e_i)``，其中 ``deg(f_i) > 0`` 且 ``e_i > 0`` 是等次分解例程的参数。

def gf_ddf_shoup(f, p, K):
    """
    Kaltofen-Shoup: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1,...,f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_ddf_shoup, gf_from_dict

    >>> f = gf_from_dict({6: ZZ(1), 5: ZZ(-1), 4: ZZ(1), 3: ZZ(1), 1: ZZ(-1)}, 3, ZZ)

    >>> gf_ddf_shoup(f, 3, ZZ)
    [([1, 1, 0], 1), ([1, 1, 0, 1, 2], 2)]

    References
    ==========

    .. [1] [Kaltofen98]_
    .. [2] [Shoup95]_
    .. [3] [Gathen92]_

    """
    # 计算多项式 ``f`` 的次数
    n = gf_degree(f)
    # 计算参数 ``k``，向上取整 ``_ceil`` 是一个函数
    k = int(_ceil(_sqrt(n//2)))
    # 计算 Frobenius monomial base
    b = gf_frobenius_monomial_base(f, p, K)
    # 计算 Frobenius map
    h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
    # U[i] = x**(p**i)，初始化 U 列表
    U = [[K.one, K.zero], h] + [K.zero]*(k - 1)

    # 计算 U 列表的值
    for i in range(2, k + 1):
        U[i] = gf_frobenius_map(U[i-1], f, b, p, K)

    # 更新 h 和 U
    h, U = U[k], U[:k]
    # V[i] = x**(p**(k*(i+1)))，初始化 V 列表
    V = [h] + [K.zero]*(k - 1)

    # 计算 V 列表的值
    for i in range(1, k):
        V[i] = gf_compose_mod(V[i - 1], h, f, p, K)

    # 初始化因子列表
    factors = []

    # 遍历 V 列表，计算因子
    for i, v in enumerate(V):
        h, j = [K.one], k - 1

        for u in U:
            g = gf_sub(v, u, p, K)
            h = gf_mul(h, g, p, K)
            h = gf_rem(h, f, p, K)

        g = gf_gcd(f, h, p, K)
        f = gf_quo(f, g, p, K)

        for u in reversed(U):
            h = gf_sub(v, u, p, K)
            F = gf_gcd(g, h, p, K)

            if F != [K.one]:
                factors.append((F, k*(i + 1) - j))

            g, j = gf_quo(g, F, p, K), j - 1

    # 如果 f 不是单项式，则将其添加到因子列表中
    if f != [K.one]:
        factors.append((f, gf_degree(f)))

    # 返回计算得到的因子列表
    return factors


# 给定的多项式 ``f`` 在有限域 ``GF(p)[x]`` 中，执行Gathen-Shoup算法的概率等次分解。
# 算法返回多项式 ``f`` 的所有不可约因子 ``f_1,...,f_d``，每个因子的次数为 ``n``。
# 这是在 Galois 域上的完全分解。

def gf_edf_shoup(f, n, p, K):
    """
    Gathen-Shoup: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and integer
    ``n`` such that ``n`` divides ``deg(f)``, returns all irreducible factors
    ``f_1,...,f_d`` of ``f``, each of degree ``n``. This is a complete
    factorization over Galois fields.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_edf_shoup

    >>> gf_edf_shoup(ZZ.map([1, 2837, 2277]), 1, 2917, ZZ)
    [[1, 852], [1, 1985]]

    References
    ==========

    .. [1] [Shoup91]_
    .. [2] [Gathen92]_

    """
    # 计算多项式 ``f`` 的次数和有限域 ``GF(p)`` 中的模数 ``q``
    N, q = gf_degree(f), int(p)

    # 如果多项式 ``f`` 的次数为零，返回空列表
    if not N:
        return []
    # 如果 ``n`` 大于等于多项式 ``f`` 的次数，返回多项式 ``f`` 的列表形式
    if N <= n:
        return [f]
    factors, x = [f], [K.one, K.zero]
    # 初始化因子列表和变量 x，其中 factors 初始包含 f，x 初始包含 K.one 和 K.zero

    r = gf_random(N - 1, p, K)
    # 生成一个随机元素 r，该元素是有限域上的一个随机数

    if p == 2:
        # 如果 p 等于 2，执行以下操作：

        h = gf_pow_mod(x, q, f, p, K)
        # 计算 x^q 模 f 在有限域 p 上的幂，得到 h

        H = gf_trace_map(r, h, x, n - 1, f, p, K)[1]
        # 计算 r 在 h 上的迹映射的第二个返回值 H

        h1 = gf_gcd(f, H, p, K)
        # 计算 f 和 H 在有限域 p 上的最大公因子 h1

        h2 = gf_quo(f, h1, p, K)
        # 计算 f 除以 h1 在有限域 p 上的商 h2

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K)
        # 使用 Shoup 分解算法分解 h1 和 h2，并将结果累加到 factors 中
    else:
        # 如果 p 不等于 2，执行以下操作：

        b = gf_frobenius_monomial_base(f, p, K)
        # 计算 f 的 Frobenius 单项基础 b

        H = _gf_trace_map(r, n, f, b, p, K)
        # 计算 r 在 f 上的迹映射，基于 Frobenius 单项基础 b，得到 H

        h = gf_pow_mod(H, (q - 1)//2, f, p, K)
        # 计算 H 的 (q-1)/2 次幂 模 f 在有限域 p 上的幂，得到 h

        h1 = gf_gcd(f, h, p, K)
        # 计算 f 和 h 在有限域 p 上的最大公因子 h1

        h2 = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)
        # 计算 f 和 h-1 在有限域 p 上的最大公因子 h2

        h3 = gf_quo(f, gf_mul(h1, h2, p, K), p, K)
        # 计算 f 除以 h1*h2 在有限域 p 上的商 h3

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K) \
            + gf_edf_shoup(h3, n, p, K)
        # 使用 Shoup 分解算法分解 h1, h2 和 h3，并将结果累加到 factors 中

    return _sort_factors(factors, multiple=False)
    # 对 factors 进行排序并返回，不允许返回多个因子
def gf_zassenhaus(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for medium ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_zassenhaus

    >>> gf_zassenhaus(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    # 使用 gf_ddf_zassenhaus 函数对 f 进行因式分解，得到因式和它们的次数 n
    for factor, n in gf_ddf_zassenhaus(f, p, K):
        # 使用 gf_edf_zassenhaus 函数对每个因式再进行进一步的因式分解，得到新的因式列表
        factors += gf_edf_zassenhaus(factor, n, p, K)

    # 对因式列表进行排序，multiple=False 表示只返回一次的因式，不包括重复因式
    return _sort_factors(factors, multiple=False)


def gf_shoup(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for large ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_shoup

    >>> gf_shoup(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    # 使用 gf_ddf_shoup 函数对 f 进行因式分解，得到因式和它们的次数 n
    for factor, n in gf_ddf_shoup(f, p, K):
        # 使用 gf_edf_shoup 函数对每个因式再进行进一步的因式分解，得到新的因式列表
        factors += gf_edf_shoup(factor, n, p, K)

    # 对因式列表进行排序，multiple=False 表示只返回一次的因式，不包括重复因式
    return _sort_factors(factors, multiple=False)


_factor_methods = {
    'berlekamp': gf_berlekamp,  # ``p`` : small
    'zassenhaus': gf_zassenhaus,  # ``p`` : medium
    'shoup': gf_shoup,      # ``p`` : large
}


def gf_factor_sqf(f, p, K, method=None):
    """
    Factor a square-free polynomial ``f`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_factor_sqf

    >>> gf_factor_sqf(ZZ.map([3, 2, 4]), 5, ZZ)
    (3, [[1, 1], [1, 3]])

    """
    # 将 f 转化为首一多项式
    lc, f = gf_monic(f, p, K)

    # 如果 f 是度小于 1 的多项式，则返回其首一系数和空因式列表
    if gf_degree(f) < 1:
        return lc, []

    # 根据指定的方法或默认方法，使用对应的函数进行因式分解
    method = method or query('GF_FACTOR_METHOD')

    if method is not None:
        factors = _factor_methods[method](f, p, K)
    else:
        factors = gf_zassenhaus(f, p, K)

    return lc, factors


def gf_factor(f, p, K):
    """
    Factor (non square-free) polynomials in ``GF(p)[x]``.

    Given a possibly non square-free polynomial ``f`` in ``GF(p)[x]``,
    returns its complete factorization into irreducibles::

                 f_1(x)**e_1 f_2(x)**e_2 ... f_d(x)**e_d

    where each ``f_i`` is a monic polynomial and ``gcd(f_i, f_j) == 1``,
    for ``i != j``.  The result is given as a tuple consisting of the
    leading coefficient of ``f`` and a list of factors of ``f`` with
    their multiplicities.

    The algorithm proceeds by first computing square-free decomposition
    of ``f`` and then iteratively factoring each of square-free factors.

    Consider a non square-free polynomial ``f = (7*x + 1) (x + 2)**2`` in
    ``GF(11)[x]``. We obtain its factorization into irreducibles as follows::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_factor

       >>> gf_factor(ZZ.map([5, 2, 7, 2]), 11, ZZ)
       (5, [([1, 2], 1), ([1, 8], 2)])

    We arrived with factorization ``f = 5 (x + 2) (x + 8)**2``. We did not
    recover the exact form of the input polynomial because we requested to
    get monic factors of ``f`` and its leading coefficient separately.

    """
    Square-free factors of ``f`` can be factored into irreducibles over
    ``GF(p)`` using three very different methods:

    Berlekamp
        efficient for very small values of ``p`` (usually ``p < 25``)
    Cantor-Zassenhaus
        efficient on average input and with "typical" ``p``
    Shoup-Kaltofen-Gathen
        efficient with very large inputs and modulus

    If you want to use a specific factorization method, instead of the default
    one, set ``GF_FACTOR_METHOD`` with one of ``berlekamp``, ``zassenhaus`` or
    ``shoup`` values.

    References
    ==========

    .. [1] [Gathen99]_

    """
    # 计算 f 在有限域 GF(p) 上的首一形式和系数列表
    lc, f = gf_monic(f, p, K)

    # 如果 f 是零次多项式（即常数），直接返回首一系数和空因子列表
    if gf_degree(f) < 1:
        return lc, []

    # 初始化因子列表
    factors = []

    # 获取 f 的平方因子分解列表，并遍历每个平方因子及其重数
    for g, n in gf_sqf_list(f, p, K)[1]:
        # 对每个平方因子 g 进行平方自由因子分解，将分解结果添加到因子列表
        for h in gf_factor_sqf(g, p, K)[1]:
            factors.append((h, n))

    # 返回首一系数和排序后的因子列表
    return lc, _sort_factors(factors)
def gf_value(f, a):
    """
    计算有限域 R 上多项式 'f' 在 'a' 处的值。

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_value

    >>> gf_value([1, 7, 2, 4], 11)
    2204

    """
    result = 0  # 初始化结果为0
    for c in f:  # 遍历多项式的系数列表
        result *= a  # 将当前结果乘以 a
        result += c  # 加上当前系数 c
    return result  # 返回计算得到的多项式在 a 处的值


def linear_congruence(a, b, m):
    """
    求解形如 a*x ≡ b (mod m) 的方程在模 m 下的所有解。

    这里 m 是正整数，a, b 是自然数。函数返回模 m 下唯一的解集合。

    Examples
    ========

    >>> from sympy.polys.galoistools import linear_congruence

    >>> linear_congruence(3, 12, 15)
    [4, 9, 14]

    在模 15 下有 3 个不同的解，因为 gcd(a, m) = gcd(3, 15) = 3。

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Linear_congruence_theorem

    """
    from sympy.polys.polytools import gcdex
    if a % m == 0:  # 如果 a 对 m 取模为0
        if b % m == 0:  # 并且 b 对 m 取模也为0
            return list(range(m))  # 返回整个模 m 的所有可能解
        else:
            return []  # 否则无解
    r, _, g = gcdex(a, m)  # 使用扩展欧几里得算法计算 gcd(a, m)
    if b % g != 0:  # 如果 b 不是 gcd(a, m) 的倍数
        return []  # 则无解
    return [(r * b // g + t * m // g) % m for t in range(g)]  # 返回模 m 下所有解的集合


def _raise_mod_power(x, s, p, f):
    """
    用于 gf_csolve 函数，生成满足 f(x) ≡ 0 (mod p**(s + 1)) 的解，
    基于满足 f(x) ≡ 0 (mod p**s) 的解。

    Examples
    ========

    >>> from sympy.polys.galoistools import _raise_mod_power
    >>> from sympy.polys.galoistools import csolve_prime

    这是 f(x) = x**2 + x + 7 在模 3 下的解：

    >>> f = [1, 1, 7]
    >>> csolve_prime(f, 3)
    [1]
    >>> [ i for i in range(3) if not (i**2 + i + 7) % 3]
    [1]

    通过 _raise_mod_power 返回的值构建了在模 9 下的解集合：

    >>> x, s, p = 1, 1, 3
    >>> V = _raise_mod_power(x, s, p, f)
    >>> [x + v * p**s for v in V]
    [1, 4, 7]

    可以用以下代码确认这些解的正确性：

    >>> [ i for i in range(3**2) if not (i**2 + i + 7) % 3**2]
    [1, 4, 7]

    """
    from sympy.polys.domains import ZZ
    f_f = gf_diff(f, p, ZZ)  # 计算 f(x) 对 p 的微分
    alpha = gf_value(f_f, x)  # 计算 f_f 在 x 处的值
    beta = - gf_value(f, x) // p**s  # 计算 f 在 x 处的值除以 p**s 后的商的负数
    return linear_congruence(alpha, beta, p)  # 返回满足条件的模 p 的解集合


def _csolve_prime_las_vegas(f, p, seed=None):
    r""" f(x) ≡ 0 (mod p) 的解，其中 f(0) \not\equiv 0 (mod p)。

    Explanation
    ===========

    这个算法属于拉斯维加斯方法。
    它总是返回正确答案并在许多情况下解决问题很快，但如果不幸，它可能永远不会给出答案。

    假设多项式 f 不是零多项式。进一步假设它的次数最多为 p-1，并且 f(0) \not\equiv 0 (mod p)。
    这些假设不是算法的基本部分，只是调用这个函数更方便来解决这些问题。

    注意，x^{p-1} - 1 \equiv \prod_{a=1}^{p-1}(x - a) \pmod{p}。

    """
    """
    Thus, the greatest common divisor with f is `\prod_{s \in S}(x - s)`,
    with S being the set of solutions to f. Furthermore,
    when a is randomly determined, `(x+a)^{(p-1)/2}-1` is
    a polynomial with (p-1)/2 randomly chosen solutions.
    The greatest common divisor of f may be a nontrivial factor of f.

    When p is large and the degree of f is small,
    it is faster than naive solution methods.

    Parameters
    ==========

    f : polynomial
        A polynomial over a finite field GF(p) defined by its coefficients.
    p : prime number
        A prime number defining the finite field GF(p) over which calculations are performed.

    Returns
    =======

    list[int]
        A list of solutions, sorted in ascending order by integers in the range [1, p).
        Each solution represents a root of the polynomial f in the finite field GF(p).
        If no solution exists, returns an empty list [].

    Examples
    ========

    >>> from sympy.polys.galoistools import _csolve_prime_las_vegas
    >>> _csolve_prime_las_vegas([1, 4, 3], 7) # x^2 + 4x + 3 = 0 (mod 7)
    [4, 6]
    >>> _csolve_prime_las_vegas([5, 7, 1, 9], 11) # 5x^3 + 7x^2 + x + 9 = 0 (mod 11)
    [1, 5, 8]

    References
    ==========

    .. [1] R. Crandall and C. Pomerance "Prime Numbers", 2nd Ed., Algorithm 2.3.10

    """
    # Import necessary functions and classes from sympy
    from sympy.polys.domains import ZZ
    from sympy.ntheory import sqrt_mod
    # Initialize a random integer generator based on seed
    randint = _randint(seed)
    # Initialize an empty set to store roots
    root = set()
    # Compute g(x) = x^(p-1) - 1 (mod f(x)) using Galois field arithmetic
    g = gf_pow_mod([1, 0], p - 1, f, p, ZZ)
    g = gf_sub_ground(g, 1, p, ZZ)
    
    # List to store factors of f(x) obtained during computation
    factors = [gf_gcd(f, g, p, ZZ)]
    while factors:
        # Pop the last factor from the list
        f = factors.pop()
        
        # If degree of f is small (<= 1), continue to next iteration
        if len(f) <= 1:
            continue
        
        # If degree of f is 2, solve directly for roots
        if len(f) == 2:
            root.add(-invert(f[0], p) * f[1] % p)
            continue
        
        # If degree of f is 3, solve for roots using specific method
        if len(f) == 3:
            inv = invert(f[0], p)
            b = f[1] * inv % p
            b = (b + p * (b % 2)) // 2
            root.update((r - b) % p for r in
                        sqrt_mod(b**2 - f[2] * inv, p, all_roots=True))
            continue
        
        # If degree of f is greater than 3, enter main loop
        while True:
            # Choose a random integer a in the range [0, p-1]
            a = randint(0, p - 1)
            # Compute g(x) = (x+a)^((p-1)/2) - 1 (mod f(x)) using Galois field arithmetic
            g = gf_pow_mod([1, a], (p - 1) // 2, f, p, ZZ)
            g = gf_sub_ground(g, 1, p, ZZ)
            # Compute gcd(f(x), g(x))
            g = gf_gcd(f, g, p, ZZ)
            # If the computed g(x) is a nontrivial factor of f(x), add factors to the list
            if 1 < len(g) < len(f):
                factors.append(g)
                factors.append(gf_div(f, g, p, ZZ)[0])
                break
    
    # Return sorted list of roots
    return sorted(root)
`
def gf_csolve(f, n):
    """
    To solve f(x) congruent 0 mod(n).

    n is divided into canonical factors and f(x) cong 0 mod(p**e) will be
    solved for each factor. Applying the Chinese Remainder Theorem to the
    results returns the final answers.

    Examples
    ========

    Solve [1, 1, 7] congruent 0 mod(189):

    >>> from sympy.polys.galoistools import gf_csolve
    >>> gf_csolve([1, 1, 7], 189)
    [13, 49, 76, 112, 139, 175]

    See Also
    ========

    sympy.ntheory.residue_ntheory.polynomial_congruence : a higher level solving routine

    References
    ==========

    .. [1] 'An introduction to the Theory of Numbers' 5th Edition by Ivan Niven,
           Zuckerman and Montgomery.

    """

    # Import the ZZ domain from sympy.polys.domains
    from sympy.polys.domains import ZZ
    # Import the factorint function from sympy.ntheory
    from sympy.ntheory import factorint

    # Factorize n into its prime factors
    P = factorint(n)

    # Solve the polynomial congruences f(x) mod p**e for each prime factor p and its exponent e
    X = [csolve_prime(f, p, e) for p, e in P.items()]

    # Convert each solution set to a tuple and store in pools
    pools = list(map(tuple, X))

    # Initialize permutations list with an empty list
    perms = [[]]

    # Generate permutations of solutions from pools
    for pool in pools:
        perms = [x + [y] for x in perms for y in pool]

    # Calculate distinct factors from prime factorization P
    dist_factors = [pow(p, e) for p, e in P.items()]

    # Use Chinese Remainder Theorem to find final solutions for each permutation in perms
    return sorted([gf_crt(per, dist_factors, ZZ) for per in perms])


这段代码的主要作用是解决多项式 f(x) 在模 n 的意义下的同余问题。它首先将 n 分解成其素因子，然后对每个素因子和其指数求解 f(x) 在模 p**e 的情况下的同余问题。最后，利用中国剩余定理组合所有素因子的解，给出最终的解集合。
```