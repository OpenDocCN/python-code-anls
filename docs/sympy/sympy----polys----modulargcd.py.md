# `D:\src\scipysrc\sympy\sympy\polys\modulargcd.py`

```
from sympy.core.symbol import Dummy  # 导入 Dummy 符号
from sympy.ntheory import nextprime  # 导入找下一个质数的函数
from sympy.ntheory.modular import crt  # 导入中国剩余定理函数
from sympy.polys.domains import PolynomialRing  # 导入多项式环
from sympy.polys.galoistools import (  # 导入 Galois 域工具函数
    gf_gcd, gf_from_dict, gf_gcdex, gf_div, gf_lcm)
from sympy.polys.polyerrors import ModularGCDFailed  # 导入多项式异常类

from mpmath import sqrt  # 导入平方根函数
import random  # 导入随机数生成函数


def _trivial_gcd(f, g):
    """
    Compute the GCD of two polynomials in trivial cases, i.e. when one
    or both polynomials are zero.
    """
    ring = f.ring  # 获取多项式环

    if not (f or g):  # 如果 f 和 g 都是零
        return ring.zero, ring.zero, ring.zero  # 返回环的零元三次
    elif not f:  # 如果 f 是零
        if g.LC < ring.domain.zero:  # 如果 g 的首项系数小于零
            return -g, ring.zero, -ring.one  # 返回 -g, 零元, -1
        else:
            return g, ring.zero, ring.one  # 返回 g, 零元, 1
    elif not g:  # 如果 g 是零
        if f.LC < ring.domain.zero:  # 如果 f 的首项系数小于零
            return -f, -ring.one, ring.zero  # 返回 -f, -1, 零元
        else:
            return f, ring.one, ring.zero  # 返回 f, 1, 零元
    return None  # 其他情况返回空


def _gf_gcd(fp, gp, p):
    r"""
    Compute the GCD of two univariate polynomials in `\mathbb{Z}_p[x]`.
    """
    dom = fp.ring.domain  # 获取多项式环的定义域

    while gp:  # 当 gp 不为空时循环
        rem = fp  # 余数为 fp
        deg = gp.degree()  # 计算 gp 的次数
        lcinv = dom.invert(gp.LC, p)  # 计算 gp 的首项系数的模 p 的逆元

        while True:
            degrem = rem.degree()  # 计算余数的次数
            if degrem < deg:  # 如果余数的次数小于 gp 的次数
                break
            # 更新余数为减去 gp 乘以特定单项式的结果
            rem = (rem - gp.mul_monom((degrem - deg,)).mul_ground(lcinv * rem.LC)).trunc_ground(p)

        fp = gp  # 更新 fp 为 gp
        gp = rem  # 更新 gp 为余数

    # 返回 fp 乘以首项系数的模 p 的逆元，然后再模 p 的结果
    return fp.mul_ground(dom.invert(fp.LC, p)).trunc_ground(p)


def _degree_bound_univariate(f, g):
    r"""
    Compute an upper bound for the degree of the GCD of two univariate
    integer polynomials `f` and `g`.

    The function chooses a suitable prime `p` and computes the GCD of
    `f` and `g` in `\mathbb{Z}_p[x]`. The choice of `p` guarantees that
    the degree in `\mathbb{Z}_p[x]` is greater than or equal to the degree
    in `\mathbb{Z}[x]`.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    """
    gamma = f.ring.domain.gcd(f.LC, g.LC)  # 计算 f 和 g 的首项系数的最大公约数
    p = 1  # 初始化 p

    p = nextprime(p)  # 找下一个质数
    while gamma % p == 0:  # 当 gamma 能整除 p 时
        p = nextprime(p)  # 继续找下一个质数

    fp = f.trunc_ground(p)  # 将 f 模 p 截断
    gp = g.trunc_ground(p)  # 将 g 模 p 截断
    hp = _gf_gcd(fp, gp, p)  # 在 \mathbb{Z}_p[x] 中计算 f 和 g 的 GCD
    deghp = hp.degree()  # 计算结果的次数
    return deghp  # 返回结果的次数


def _chinese_remainder_reconstruction_univariate(hp, hq, p, q):
    r"""
    Construct a polynomial `h_{pq}` in `\mathbb{Z}_{p q}[x]` such that

    .. math ::

        h_{pq} = h_p \; \mathrm{mod} \, p

        h_{pq} = h_q \; \mathrm{mod} \, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\mathbb{Z}_p[x]` and `\mathbb{Z}_q[x]`
    respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\mathbb{Z}_p[x]`, `\mathbb{Z}_q[x]` and `\mathbb{Z}_{p q}[x]` is used.
    It is assumed that `h_p` and `h_q` have the same degree.

    Parameters
    ==========

    hp : PolyElement
        univariate polynomial in `\mathbb{Z}_p[x]`
    hq : PolyElement
        univariate polynomial in `\mathbb{Z}_q[x]`
    p : Integer
        prime integer
    q : Integer
        prime integer

    """
    # hp : PolyElement
    #     univariate integer polynomial with coefficients in `\mathbb{Z}_p`
    # hq : PolyElement
    #     univariate integer polynomial with coefficients in `\mathbb{Z}_q`
    # p : Integer
    #     modulus of `hp`, relatively prime to `q`
    # q : Integer
    #     modulus of `hq`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = -x**3 - 1
    >>> hq = 2*x**3 - 2*x**2 + x

    # Compute hpq using Chinese remainder theorem reconstruction
    >>> hpq = _chinese_remainder_reconstruction_univariate(hp, hq, p, q)
    >>> hpq
    2*x**3 + 3*x**2 + 6*x + 5

    # Check if hpq truncated modulo p equals hp
    >>> hpq.trunc_ground(p) == hp
    True
    # Check if hpq truncated modulo q equals hq
    >>> hpq.trunc_ground(q) == hq
    True

    """
    # Get the degree of hp
    n = hp.degree()
    # Get the generator (x) of the ring of hp
    x = hp.ring.gens[0]
    # Initialize hpq as the zero polynomial in the ring of hp
    hpq = hp.ring.zero

    # Iterate over the range of degrees of hp, reconstructing hpq using CRT
    for i in range(n+1):
        # Use the Chinese remainder theorem to reconstruct coefficients of hpq
        hpq[(i,)] = crt([p, q], [hp.coeff(x**i), hq.coeff(x**i)], symmetric=True)[0]

    # Remove zero coefficients from hpq
    hpq.strip_zero()
    # Return the reconstructed polynomial hpq
    return hpq
# 定义一个函数，计算两个整系数一元多项式在模算法下的最大公因数
def modgcd_univariate(f, g):
    # 文档字符串，描述了该算法的作用和实现细节
    r"""
    Computes the GCD of two polynomials in `\mathbb{Z}[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two univariate integer polynomials
    `f` and `g` by computing the GCD in `\mathbb{Z}_p[x]` for suitable
    primes `p` and then reconstructing the coefficients with the Chinese
    Remainder Theorem. Trial division is only made for candidates which
    are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)

    >>> f = x**5 - 1
    >>> g = x - 1

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = 6*x**2 - 6
    >>> g = 2*x**2 + 4*x + 2

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (2*x + 2, 3*x - 3, x + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    # 断言条件，确保 f 和 g 是在同一个环上的多项式，并且系数在整数集合中
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    # 利用 _trivial_gcd 函数快速计算如果可能的话返回的最大公因数
    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    # 获取多项式的环
    ring = f.ring

    # 将 f 和 g 转化为其原始部分并返回系数
    cf, f = f.primitive()
    cg, g = g.primitive()
    # 计算 f 和 g 的系数最大公因数
    ch = ring.domain.gcd(cf, cg)

    # 计算多项式的最大次数
    bound = _degree_bound_univariate(f, g)
    if bound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    # 计算 f 和 g 的最低次数系数的最大公因数
    gamma = ring.domain.gcd(f.LC, g.LC)
    m = 1
    p = 1

    while True:
        # 获取下一个素数
        p = nextprime(p)
        while gamma % p == 0:
            p = nextprime(p)

        # 将 f 和 g 限制为 p
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        # 计算 f 和 g 的最大公因数
        hp = _gf_gcd(fp, gp, p)
        deghp = hp.degree()

        # 如果 deghp 比 bound 大则继续循环
        if deghp > bound:
            continue
        elif deghp < bound:
            m = 1
            bound = deghp
            continue

        # 重新构造 hp
        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        # 重建多项式 hp
        hm = _chinese_remainder_reconstruction_univariate(hp, hlastm, p, m)
        m *= p

        # 如果不同则重新构建 hm
        if not hm == hlastm:
            hlastm = hm
            continue

        # 如果 h 是负的则返回 ch 乘以 h
        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


# 定义一个私有函数，计算具有一个函数的系数
def _primitive(f, p):
    r"""
    # 获取多项式 `f` 所在的环
    ring = f.ring
    # 确定环的定义域
    dom = ring.domain
    # 获取环中生成元的数量
    k = ring.ngens

    # 初始化空字典，用于存储每个单项式的系数
    coeffs = {}
    # 遍历多项式 `f` 中的每个单项式及其系数
    for monom, coeff in f.iterterms():
        # 如果单项式去掉最后一个变量后不在 coeffs 字典中，就添加一个空字典作为其值
        if monom[:-1] not in coeffs:
            coeffs[monom[:-1]] = {}
        # 将当前单项式的最后一个变量作为键，系数作为值存入 coeffs 字典中
        coeffs[monom[:-1]][monom[-1]] = coeff

    # 初始化一个空列表，用于存储计算出的内容多项式
    cont = []
    # 遍历 coeffs 字典中的每个系数字典，计算每个系数字典的最大公因子
    for coeff in iter(coeffs.values()):
        # 使用 gf_gcd 函数计算当前系数字典的最大公因子，并将结果追加到 cont 列表中
        cont = gf_gcd(cont, gf_from_dict(coeff, p, dom), p, dom)

    # 在原环中为最后一个变量创建一个新的环
    yring = ring.clone(symbols=ring.symbols[k-1])
    # 将计算出的内容多项式转换为 yring 环中的多项式，并将其限制在模 p 下
    contf = yring.from_dense(cont).trunc_ground(p)

    # 返回计算出的内容多项式以及原多项式 `f` 除以内容多项式后的商
    return contf, f.quo(contf.set_ring(ring))
def _degree_bound_bivariate(f, g):
    r"""
    Compute upper degree bounds for the GCD of two bivariate
    integer polynomials `f` and `g`.

    The GCD is viewed as a polynomial in `\mathbb{Z}[y][x]` and the
    function returns an upper bound for its degree and one for the degree
    of its content. This is done by choosing a suitable prime `p` and
    computing the GCD of the contents of `f \; \mathrm{mod} \, p` and
    `g \; \mathrm{mod} \, p`. The choice of `p` guarantees that the degree
    of the content in `\mathbb{Z}_p[y]` is greater than or equal to the
    degree in `\mathbb{Z}[y]`. To obtain the degree bound in the variable
    `x`, the polynomials are evaluated at `y = a` for a suitable
    `a \in \mathbb{Z}_p` and then their GCD in `\mathbb{Z}_p[x]` is computed.

    Parameters
    ==========

    f, g : PolyElement
        bivariate integer polynomials

    Returns
    =======

    deg_bound : Tuple
        upper degree bound for the GCD of `f` and `g` in `x`

    """
    # 获取多项式 `f` 的环（环境）
    ring = f.ring

    # 计算 `f` 和 `g` 的领导系数的最大公约数
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(_swap(f, 1).LC, _swap(g, 1).LC)
    # 计算导致问题的素数乘积
    badprimes = gamma1 * gamma2
    p = 1

    # 找到比 `p` 大的下一个素数
    p = nextprime(p)
    while badprimes % p == 0:
        p = nextprime(p)

    # 在模 `p` 下截断 `f` 和 `g` 的系数
    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    # 计算 `fp` 和 `gp` 的原始部分和系数调整后的多项式
    contfp, fp = _primitive(fp, p)
    contgp, gp = _primitive(gp, p)
    # 计算 `fp` 和 `gp` 的最大公因式的原始部分，得到在 `Z_p[y]` 中的多项式
    conthp = _gf_gcd(contfp, contgp, p)
    # 计算在 `y` 变量中最大公因式内容的次数上界
    ycontbound = conthp.degree()

    # 计算在 `Z_p[y]` 中 `fp` 和 `gp` 的领导系数的最大公因式
    delta = _gf_gcd(_LC(fp), _LC(gp), p)

    # 对于每个 `a` 在 `Z_p` 范围内进行迭代
    for a in range(p):
        # 如果 `delta` 在 `(0, a)` 处的值对 `p` 取模为零，则继续下一个 `a`
        if not delta.evaluate(0, a) % p:
            continue
        # 对 `fp` 和 `gp` 在 `x=1, y=a` 处的值进行 `p` 模截断
        fpa = fp.evaluate(1, a).trunc_ground(p)
        gpa = gp.evaluate(1, a).trunc_ground(p)
        # 计算在 `Z_p` 中 `fpa` 和 `gpa` 的最大公因式
        hpa = _gf_gcd(fpa, gpa, p)
        # 计算 `hpa` 在 `x` 变量中的次数上界
        xbound = hpa.degree()
        # 返回 `xbound` 和 `ycontbound` 作为结果
        return xbound, ycontbound

    # 如果循环未找到合适的 `a`，返回 `fp` 和 `gp` 在 `x` 变量中的次数的最小值，以及 `ycontbound`
    return min(fp.degree(), gp.degree()), ycontbound
# 构建一个多变量整数多项式 `h_{pq}`，其系数在 `\mathbb{Z}_{p q}[x_0, \ldots, x_{k-1}]` 上
# 使得以下条件成立：
#   - `h_{pq} = h_p \; \mathrm{mod} \, p`
#   - `h_{pq} = h_q \; \mathrm{mod} \, q`
# 其中 `p` 和 `q` 是互质的整数，`h_p` 和 `h_q` 分别是在 `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` 和 `\mathbb{Z}_q[x_0, \ldots, x_{k-1}]` 上的多项式。

def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    # 获取 `hp` 和 `hq` 的单项式集合
    hpmonoms = set(hp.monoms())
    hqmonoms = set(hq.monoms())
    # 找到 `hp` 和 `hq` 公共的单项式
    monoms = hpmonoms.intersection(hqmonoms)
    # 从 `hp` 和 `hq` 的单项式集合中移除公共的单项式
    hpmonoms.difference_update(monoms)
    hqmonoms.difference_update(monoms)
    
    # 在环 `hp.ring` 中定义零元素
    zero = hp.ring.domain.zero
    # 创建一个零多项式 `hpq`
    hpq = hp.ring.zero
    
    # 如果 `hp.ring.domain` 是多项式环，则使用 `_chinese_remainder_reconstruction_multivariate` 函数本身
    if isinstance(hp.ring.domain, PolynomialRing):
        crt_ = _chinese_remainder_reconstruction_multivariate
    else:
        # 否则定义一个内部函数 `crt_`，使用 `_chinese_remainder_reconstruction_multivariate` 函数来计算
        def crt_(cp, cq, p, q):
            return crt([p, q], [cp, cq], symmetric=True)[0]
    
    # 对于公共的单项式，计算 `hpq` 的系数，使用 `crt_` 函数
    for monom in monoms:
        hpq[monom] = crt_(hp[monom], hq[monom], p, q)
    # 对于 `hp` 独有的单项式，使用 `crt_` 函数和零元素来计算 `hpq` 的系数
    for monom in hpmonoms:
        hpq[monom] = crt_(hp[monom], zero, p, q)
    # 对于 `hq` 独有的单项式，使用 `crt_` 函数和零元素来计算 `hpq` 的系数
    for monom in hqmonoms:
        hpq[monom] = crt_(zero, hq[monom], p, q)
    
    # 返回重构的多项式 `hpq`
    return hpq
    """
    Interpolates a polynomial `h_p` over `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` from its evaluations.

    It is possible to reconstruct a parameter of the ground domain if `h_p` is a polynomial over `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.
    In such cases, set ``ground=True``.

    Parameters
    ==========

    evalpoints : list of Integer objects
        List of evaluation points in `\mathbb{Z}_p`.
    hpeval : list of PolyElement objects
        List of polynomials in `\mathbb{Z}_p[x_0, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{k-1}]`,
        representing the images of `h_p` evaluated in the variable `x_i`.
    ring : PolyRing
        The ring in which `h_p` resides.
    i : Integer
        Index of the variable to reconstruct.
    p : Integer
        Prime number, modulus of `h_p`.
    ground : Boolean
        Indicates if `x_i` is in the ground domain (`True`) or not (`False`).

    Returns
    =======

    hp : PolyElement
        Interpolated polynomial in `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`
    """
    # Initialize hp to zero polynomial in the ring
    hp = ring.zero

    # Determine the domain and the variable y based on the ground flag
    if ground:
        domain = ring.domain.domain
        y = ring.domain.gens[i]
    else:
        domain = ring.domain
        y = ring.gens[i]

    # Loop over evaluation points and corresponding polynomial evaluations
    for a, hpa in zip(evalpoints, hpeval):
        numer = ring.one  # Numerator initialized to multiplicative identity in the ring
        denom = domain.one  # Denominator initialized to multiplicative identity in the domain

        # Compute the product of linear factors (y - b) for b != a
        for b in evalpoints:
            if b == a:
                continue
            numer *= y - b
            denom *= a - b

        # Invert the denominator in the domain modulo p
        denom = domain.invert(denom, p)

        # Compute the coefficient for hpa and add to hp
        coeff = numer.mul_ground(denom)
        hp += hpa.set_ring(ring) * coeff

    # Reduce hp modulo p and return
    return hp.trunc_ground(p)
    # 断言两个多项式 f 和 g 在同一个环上，并且环的系数域为整数环
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    # 对 f 和 g 进行简单的 GCD 计算，如果可以直接确定结果则返回
    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    # 获取多项式的环
    ring = f.ring

    # 将 f 和 g 化为本原多项式，并获取系数的最大公因子
    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    # 计算双变量多项式 f 和 g 的度限制
    xbound, ycontbound = _degree_bound_bivariate(f, g)
    if xbound == ycontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    # 对交换后的多项式进行处理，计算其度限制
    fswap = _swap(f, 1)
    gswap = _swap(g, 1)
    degyf = fswap.degree()
    degyg = gswap.degree()

    ybound, xcontbound = _degree_bound_bivariate(fswap, gswap)
    if ybound == xcontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    # TODO: 为了提高性能，在这里选择主要变量

    # 计算 f 和 g 的主导系数的最大公因子，以及交换后的多项式的主导系数的最大公因子
    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(fswap.LC, gswap.LC)
    # 计算不良质数的乘积
    badprimes = gamma1 * gamma2
    m = 1
    p = 1
    # 进入无限循环，直到条件中断循环
    while True:
        # 寻找下一个素数
        p = nextprime(p)
        # 当 badprimes 对 p 可整除时，继续寻找下一个素数
        while badprimes % p == 0:
            p = nextprime(p)

        # 对 f 和 g 模 p 截断，得到 fp 和 gp
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        
        # 计算 fp 和 gp 的本原部分，并更新 fp 和 gp
        contfp, fp = _primitive(fp, p)
        contgp, gp = _primitive(gp, p)
        
        # 计算 fp 和 gp 的最大公因式，结果为 Z_p[y] 中的首一多项式
        conthp = _gf_gcd(contfp, contgp, p)
        # 计算 conthp 的次数
        degconthp = conthp.degree()

        # 如果 degconthp 大于 ycontbound，则继续下一轮循环
        if degconthp > ycontbound:
            continue
        # 如果 degconthp 小于 ycontbound，则更新 m 为 1，并更新 ycontbound
        elif degconthp < ycontbound:
            m = 1
            ycontbound = degconthp
            continue

        # 计算 Z_p[y] 中的多项式 delta
        delta = _gf_gcd(_LC(fp), _LC(gp), p)

        # 计算 fp 和 gp 的本原部分的次数
        degcontfp = contfp.degree()
        degcontgp = contgp.degree()
        # 计算 delta 的次数
        degdelta = delta.degree()

        # 计算 N 的值，为几个边界值的最小值加 1
        N = min(degyf - degcontfp, degyg - degcontgp,
            ybound - ycontbound + degdelta) + 1

        # 如果 p 小于 N，则继续下一轮循环
        if p < N:
            continue

        # 初始化计数器 n 和评估点列表 evalpoints，以及 hpeval 列表
        n = 0
        evalpoints = []
        hpeval = []
        # 初始化 unlucky 为 False
        unlucky = False

        # 遍历从 0 到 p-1 的所有整数 a
        for a in range(p):
            # 计算 delta 在 (0, a) 处的值
            deltaa = delta.evaluate(0, a)
            # 如果 deltaa 对 p 可整除，则继续下一轮循环
            if not deltaa % p:
                continue

            # 计算 fp 在 (1, a) 处的值，并模 p 截断
            fpa = fp.evaluate(1, a).trunc_ground(p)
            # 计算 gp 在 (1, a) 处的值，并模 p 截断
            gpa = gp.evaluate(1, a).trunc_ground(p)
            # 计算 fpa 和 gpa 的最大公因式，结果为 Z_p[x] 中的首一多项式
            hpa = _gf_gcd(fpa, gpa, p)
            # 计算 hpa 的次数
            deghpa = hpa.degree()

            # 如果 deghpa 大于 xbound，则继续下一轮循环
            if deghpa > xbound:
                continue
            # 如果 deghpa 小于 xbound，则更新 m 为 1，更新 xbound，设置 unlucky 为 True，并跳出循环
            elif deghpa < xbound:
                m = 1
                xbound = deghpa
                unlucky = True
                break

            # 计算 hpa 乘以 deltaa，并模 p 截断
            hpa = hpa.mul_ground(deltaa).trunc_ground(p)
            # 将 a 添加到 evalpoints 列表中
            evalpoints.append(a)
            # 将 hpa 添加到 hpeval 列表中
            hpeval.append(hpa)
            # 计数器 n 自增 1
            n += 1

            # 如果 n 等于 N，则跳出循环
            if n == N:
                break

        # 如果 unlucky 为 True，则继续下一轮循环
        if unlucky:
            continue
        # 如果 n 小于 N，则继续下一轮循环
        if n < N:
            continue

        # 对 evalpoints 和 hpeval 使用多变量插值，得到 hp，使用环 ring 和模 p
        hp = _interpolate_multivariate(evalpoints, hpeval, ring, 1, p)

        # 对 hp 取本原部分，并更新 hp
        hp = _primitive(hp, p)[1]
        # hp 乘以 conthp 的环
        hp = hp * conthp.set_ring(ring)
        # 计算 hp 关于第一个变量的次数
        degyhp = hp.degree(1)

        # 如果 degyhp 大于 ybound，则继续下一轮循环
        if degyhp > ybound:
            continue
        # 如果 degyhp 小于 ybound，则更新 m 为 1，更新 ybound，并继续下一轮循环
        if degyhp < ybound:
            m = 1
            ybound = degyhp
            continue

        # hp 乘以 gamma1，并模 p 截断
        hp = hp.mul_ground(gamma1).trunc_ground(p)
        
        # 如果 m 等于 1，则更新 m 为 p，更新 hlastm 为 hp，并继续下一轮循环
        if m == 1:
            m = p
            hlastm = hp
            continue

        # 使用多变量中国剩余定理重构 hp 和 hlastm，使用模 p 和 m
        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        # m 乘以 p
        m *= p

        # 如果 hm 不等于 hlastm，则更新 hlastm 为 hm，并继续下一轮循环
        if not hm == hlastm:
            hlastm = hm
            continue

        # 对 hm 进行 quo_ground(hm.content())，得到 h
        h = hm.quo_ground(hm.content())
        # 使用 f.div(h) 得到 fquo 和 frem
        fquo, frem = f.div(h)
        # 使用 g.div(h) 得到 gquo 和 grem
        gquo, grem = g.div(h)

        # 如果 frem 和 grem 均为零
        if not frem and not grem:
            # 如果 h 的首项系数小于 0，则更新 ch 为 -ch
            if h.LC < 0:
                ch = -ch
            # 将 h 乘以 ch
            h = h.mul_ground(ch)
            # 将 fquo 乘以 cf // ch，得到 cff
            cff = fquo.mul_ground(cf // ch)
            # 将 gquo 乘以 cg // ch，得到 cfg
            cfg = gquo.mul_ground(cg // ch)
            # 返回 h、cff 和 cfg
            return h, cff, cfg
# 定义一个函数，计算多变量多项式在有限域上的最大公约数
def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    r"""
    Compute the GCD of two polynomials in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `x_{k-1} = a` for suitable
    `a \in \mathbb{Z}_p` and then calls itself recursively to compute the GCD
    in `\mathbb{Z}_p[x_0, \ldots, x_{k-2}]`. If these recursive calls are
    successful for enough evaluation points, the GCD in `k` variables is
    interpolated, otherwise the algorithm returns ``None``. Every time a GCD
    or a content is computed, their degrees are compared with the bounds. If
    a degree greater then the bound is encountered, then the current call
    returns ``None`` and a new evaluation point has to be chosen. If at some
    point the degree is smaller, the correspondent bound is updated and the
    algorithm fails.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    g : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    p : Integer
        prime number, modulus of `f` and `g`
    degbound : list of Integer objects
        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`
        and `g` in the variable `x_i`
    contbound : list of Integer objects
        ``contbound[i]`` is an upper bound for the degree of the content of
        the GCD in `\mathbb{Z}_p[x_i][x_0, \ldots, x_{i-1}]`,
        ``contbound[0]`` is not used can therefore be chosen
        arbitrarily.

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g` or ``None``

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    """
    # 获取多项式所在环的信息
    ring = f.ring
    k = ring.ngens

    # 如果只有一个变量，则直接计算其GCD
    if k == 1:
        h = _gf_gcd(f, g, p).trunc_ground(p)  # 计算单变量多项式的GCD，并对结果进行p次幂截断
        degh = h.degree()  # 计算GCD的次数

        # 如果GCD的次数超过了给定的上界，则返回None
        if degh > degbound[0]:
            return None
        # 如果GCD的次数小于给定的上界，则更新上界，并抛出异常
        if degh < degbound[0]:
            degbound[0] = degh
            raise ModularGCDFailed

        return h  # 返回计算得到的GCD

    # 否则，进行多变量情况下的处理
    degyf = f.degree(k-1)  # 计算f在最高变量方向上的次数
    degyg = g.degree(k-1)  # 计算g在最高变量方向上的次数

    contf, f = _primitive(f, p)  # 对f进行原始多项式和内容分解
    contg, g = _primitive(g, p)  # 对g进行原始多项式和内容分解

    conth = _gf_gcd(contf, contg, p)  # 计算内容的GCD，得到一个多项式在Z_p[y]中

    degcontf = contf.degree()  # 计算内容的次数
    degcontg = contg.degree()  # 计算内容的次数
    degconth = conth.degree()  # 计算内容GCD的次数

    # 如果内容GCD的次数超过了给定的上界，则返回None
    if degconth > contbound[k-1]:
        return None
    # 如果内容GCD的次数小于给定的上界，则更新上界，并抛出异常
    if degconth < contbound[k-1]:
        contbound[k-1] = degconth
        raise ModularGCDFailed

    lcf = _LC(f)  # 计算f的最低次数项系数
    lcg = _LC(g)  # 计算g的最低次数项系数

    delta = _gf_gcd(lcf, lcg, p)  # 计算最低次数项系数的GCD，得到一个多项式在Z_p[y]中

    evaltest = delta

    # 对每个变量进行处理，计算最低次数项系数的GCD，并更新evaltest
    for i in range(k-1):
        evaltest *= _gf_gcd(_LC(_swap(f, i)), _LC(_swap(g, i)), p)

    degdelta = delta.degree()  # 计算delta的次数

    # 计算N的值，为后续处理提供上界
    N = min(degyf - degcontf, degyg - degcontg,
            degbound[k-1] - contbound[k-1] + degdelta) + 1

    # 如果p小于N，则返回None
    if p < N:
        return None

    n = 0
    d = 0
    evalpoints = []
    heval = []
    points = list(range(p))

    # 代码未完，继续进行处理...
    # 当还有未处理的点时执行循环
    while points:
        # 从 points 中随机选择一个点 a
        a = random.sample(points, 1)[0]
        # 移除已选的点 a
        points.remove(a)

        # 如果 evaltest 对 a 的评估结果模 p 等于 0，则继续下一轮循环
        if not evaltest.evaluate(0, a) % p:
            continue

        # 计算 deltaa，即 delta 在 a 处的评估结果模 p
        deltaa = delta.evaluate(0, a) % p

        # 计算 fa 和 ga，即 f 和 g 在 k-1 处对 a 的评估结果，并对 p 取模
        fa = f.evaluate(k-1, a).trunc_ground(p)
        ga = g.evaluate(k-1, a).trunc_ground(p)

        # 在 Z_p[x_0, ..., x_{k-2}] 上计算多元多项式 ha 的模 p 的最大公因子
        ha = _modgcd_multivariate_p(fa, ga, p, degbound, contbound)

        # 如果 ha 为 None，则增加 d，并检查是否超过 n，超过则返回 None
        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        # 如果 ha 是常数多项式，则返回 conth 在 ring 上的模 p 截断结果
        if ha.is_ground:
            h = conth.set_ring(ring).trunc_ground(p)
            return h

        # 将 ha 乘以 deltaa 并对 p 取模，然后截断到 p
        ha = ha.mul_ground(deltaa).trunc_ground(p)

        # 将当前处理的点 a 和其对应的多项式 ha 加入评估点列表和多项式列表
        evalpoints.append(a)
        heval.append(ha)
        n += 1

        # 如果 n 等于 N，则进行多元插值
        if n == N:
            # 对评估点和多项式列表执行多元插值，得到多项式 h
            h = _interpolate_multivariate(evalpoints, heval, ring, k-1, p)

            # 对 h 进行原始化，并乘以 conth 在 ring 上的结果，得到 h
            h = _primitive(h, p)[1] * conth.set_ring(ring)
            # 计算 h 在 k-1 上的次数
            degyh = h.degree(k-1)

            # 如果 h 在 k-1 上的次数大于 degbound[k-1]，返回 None
            if degyh > degbound[k-1]:
                return None
            # 如果 h 在 k-1 上的次数小于 degbound[k-1]，更新 degbound[k-1] 并引发 ModularGCDFailed 异常
            if degyh < degbound[k-1]:
                degbound[k-1] = degyh
                raise ModularGCDFailed

            # 返回多项式 h
            return h

    # 如果 points 已处理完毕仍未找到合适的多项式，则返回 None
    return None
    """
    计算 `\mathbb{Z}[x_0, \ldots, x_{k-1}]` 中两个多项式的最大公因式（GCD），
    使用模算法。

    该算法通过在适当的素数 `p` 下计算 `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` 中的
    GCD，然后利用中国剩余定理重构系数来计算多变量整数多项式 `f` 和 `g` 的GCD。
    为了在 `\mathbb{Z}[x_0, \ldots, x_{k-1}]` 中验证结果，进行试除，
    但仅针对可能是所需GCD的候选项。

    Parameters
    ==========

    f : PolyElement
        多变量整数多项式
    g : PolyElement
        多变量整数多项式

    Returns
    =======

    h : PolyElement
        多项式 `f` 和 `g` 的最大公因式（GCD）
    cff : PolyElement
        `f` 的余因子，即 `\frac{f}{h}`
    cfg : PolyElement
        `g` 的余因子，即 `\frac{g}{h}`
    """

    assert f.ring == g.ring and f.ring.domain.is_ZZ

    # 使用 `_trivial_gcd` 函数尝试计算简单情况下的GCD
    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring
    k = ring.ngens

    # 整数内容的除法
    cf, f = f.primitive()
    cg, g = g.primitive()

    # 计算整数内容的最大公因数
    ch = ring.domain.gcd(cf, cg)

    # 计算导数的最大公因数
    gamma = ring.domain.gcd(f.LC, g.LC)

    # 计算不良素数
    badprimes = ring.domain.one
    for i in range(k):
        badprimes *= ring.domain.gcd(_swap(f, i).LC, _swap(g, i).LC)

    # 计算度和内容的边界
    degbound = [min(fdeg, gdeg) for fdeg, gdeg in zip(f.degrees(), g.degrees())]
    contbound = list(degbound)

    m = 1
    p = 1
    # 进入无限循环，寻找下一个素数
    while True:
        # 获取下一个素数
        p = nextprime(p)
        # 如果当前素数是坏素数badprimes的因子，则继续找下一个素数
        while badprimes % p == 0:
            p = nextprime(p)

        # 对 f 和 g 在 p 下的截断多项式
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)

        try:
            # 在 Z_p[x_0, ..., x_{k-2}, y] 环中计算 fp 和 gp 的首一最大公因式
            hp = _modgcd_multivariate_p(fp, gp, p, degbound, contbound)
        except ModularGCDFailed:
            # 如果计算失败，重置 m 并继续下一轮循环
            m = 1
            continue

        # 如果 hp 为 None，则继续下一轮循环
        if hp is None:
            continue

        # 将 hp 乘以 gamma 并在 p 下截断
        hp = hp.mul_ground(gamma).trunc_ground(p)

        # 如果 m 等于 1，更新 m 和 hlastm，并继续下一轮循环
        if m == 1:
            m = p
            hlastm = hp
            continue

        # 使用中国剩余定理在多变量环中重构 hp 和 hlastm
        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        # 更新 m 为 m * p
        m *= p

        # 如果 hm 不等于 hlastm，则更新 hlastm 并继续下一轮循环
        if not hm == hlastm:
            hlastm = hm
            continue

        # 取 hm 的原始部分，并在 f 和 g 上进行整除运算
        h = hm.primitive()[1]
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)

        # 如果 f 和 g 都能整除 h，则调整 h 的符号以使 LC < 0
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            # 计算最终的返回值
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg
# 定义函数 `_gf_div`，用于计算两个一元多项式 `f` 和 `g` 在整数环 `\mathbb Z_p` 上的模 `p` 除法
def _gf_div(f, g, p):
    # 获取 f 的环
    ring = f.ring
    # 调用 gf_div 函数计算 f 和 g 的密集表示的模 p 除法
    densequo, denserem = gf_div(f.to_dense(), g.to_dense(), p, ring.domain)
    # 从密集表示中恢复多项式，并返回商和余数
    return ring.from_dense(densequo), ring.from_dense(denserem)

# 定义函数 `_rational_function_reconstruction`，用于从多项式 `c` 和 `m` 在整数环 `\mathbb Z_p[t]` 上的模 `m` 同余条件中重建有理函数 `\frac a b`
def _rational_function_reconstruction(c, p, m):
    # 获取 c 的环
    ring = c.ring
    # 获取 c 的定义域
    domain = ring.domain
    # 计算模 m 的次数 M 和相关参数
    M = m.degree()
    N = M // 2
    D = M - N - 1

    # 初始化算法中的变量 r0, s0, r1, s1
    r0, s0 = m, ring.zero
    r1, s1 = c, ring.one

    # 使用欧几里得算法进行迭代，直到 r1 的次数小于等于 N
    while r1.degree() > N:
        # 计算当前迭代的商
        quo = _gf_div(r0, r1, p)[0]
        # 更新 r0, r1 和 s0, s1
        r0, r1 = r1, (r0 - quo*r1).trunc_ground(p)
        s0, s1 = s1, (s0 - quo*s1).trunc_ground(p)

    # 最终得到的多项式 a 和 b
    a, b = r1, s1
    # 检查 b 的次数和是否可逆性，若不符合条件则返回 None
    if b.degree() > D or _gf_gcd(b, m, p) != 1:
        return None

    # 若 b 的主导系数不为 1，则调整 a 和 b
    lc = b.LC
    if lc != 1:
        lcinv = domain.invert(lc, p)
        a = a.mul_ground(lcinv).trunc_ground(p)
        b = b.mul_ground(lcinv).trunc_ground(p)

    # 将 a 和 b 转换为域元素，并返回它们的比值作为结果
    field = ring.to_field()
    return field(a) / field(b)

# 定义函数 `_rational_reconstruction_func_coeffs`，用于从多项式 `hm` 的系数 `c_{h_m}` 中重建多项式 `h` 的系数 `c_h`
def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    # 获取多项式 hm 的环
    r"""
    Reconstruct every coefficient `c_h` of a polynomial `h` in
    `\mathbb Z_p(t_k)[t_1, \ldots, t_{k-1}][x, z]` from the corresponding
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\mathbb Z_p[t_1, \ldots, t_k][x, z] \cong \mathbb Z_p[t_k][t_1, \ldots, t_{k-1}][x, z]`
    such that

    .. math::

        c_{h_m} = c_h \; \mathrm{mod} \, m,

    where `m \in \mathbb Z_p[t]`.

    The reconstruction is based on the Euclidean Algorithm. In general, `m`
    is not irreducible, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    p : Integer
        prime number, modulus of `\mathbb Z_p`
    m : PolyElement
        modulus, polynomial in `\mathbb Z[t]`, not necessarily irreducible
    ring : PolyRing
        `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]`, `h` will be an
        element of this ring
    k : Integer
        index of the parameter `t_k` which will be reconstructed

    Returns
    =======
    None
    """
    # 返回多项式 `h` 的系数 `c_h` 的重建结果或 None
    return
    # 定义变量 h 为 PolyElement 类型，表示重构的多项式，属于 `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]` 或者为 None
    h = ring.zero
    
    # 迭代 hm 中的每一个项，monom 为单项式，coeff 为对应的系数
    for monom, coeff in hm.iterterms():
        # 如果 k 等于 0，则使用 _rational_function_reconstruction 函数重构系数 coeff
        if k == 0:
            coeffh = _rational_function_reconstruction(coeff, p, m)
            
            # 如果 coeffh 为 None，则返回 None
            if not coeffh:
                return None
        
        # 如果 k 不等于 0
        else:
            # 初始化 coeffh 为 ring.domain 的零元素
            coeffh = ring.domain.zero
            
            # 对 coeff 进行 k 降维操作，然后迭代每一个降维后的 mon 和 c
            for mon, c in coeff.drop_to_ground(k).iterterms():
                # 使用 _rational_function_reconstruction 函数重构 c
                ch = _rational_function_reconstruction(c, p, m)
                
                # 如果 ch 为 None，则返回 None
                if not ch:
                    return None
                
                # 将重构后的 ch 赋值给 coeffh 的 mon 位置
                coeffh[mon] = ch
        
        # 将 coeffh 赋值给 h 的 monom 位置
        h[monom] = coeffh
    
    # 返回重构后的多项式 h
    return h
# 定义一个函数 `_gf_gcdex`，实现在环 `\mathbb Z_p[z]` 上两个一元多项式的扩展欧几里得算法
def _gf_gcdex(f, g, p):
    # 获取多项式 f 的环
    ring = f.ring
    # 调用 `_gf_gcdex` 函数计算两个多项式的扩展欧几里得算法结果，返回 s, t, h
    s, t, h = gf_gcdex(f.to_dense(), g.to_dense(), p, ring.domain)
    # 将结果转换为原环的多项式，并返回
    return ring.from_dense(s), ring.from_dense(t), ring.from_dense(h)


# 定义一个函数 `_trunc`，用于计算多项式在环 `\mathbb Z_p[z] / (\check m_{\alpha}(z))[x]` 中的约简表示
def _trunc(f, minpoly, p):
    # 获取多项式 f 的环
    ring = f.ring
    # 设置 minpoly 的环为当前环
    minpoly = minpoly.set_ring(ring)
    # 创建 p_，作为环的新元素 p
    p_ = ring.ground_new(p)

    # 计算多项式 f 在 mod minpoly 和 p 下的约简形式，并返回
    return f.trunc_ground(p).rem([minpoly, p_]).trunc_ground(p)


# 定义一个函数 `_euclidean_algorithm`，用于在 `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x]` 中计算两个多项式的单参数 GCD
def _euclidean_algorithm(f, g, minpoly, p):
    # 获取多项式 f 的环
    ring = f.ring

    # 对 f 和 g 应用 _trunc 函数，得到它们的约简形式
    f = _trunc(f, minpoly, p)
    g = _trunc(g, minpoly, p)

    # 使用欧几里得算法计算多项式的 GCD
    while g:
        rem = f
        deg = g.degree(0)  # g 在 x 的次数

        # 计算 g 的首项系数的逆
        lcinv, _, gcd = _gf_gcdex(ring.dmp_LC(g), minpoly, p)

        # 如果 lcinv 不等于 1，返回 None
        if not gcd == 1:
            return None

        # 进行一系列步骤来减少 rem
        while True:
            degrem = rem.degree(0)  # rem 在 x 的次数
            if degrem < deg:
                break
            quo = (lcinv * ring.dmp_LC(rem)).set_ring(ring)
            rem = _trunc(rem - g.mul_monom((degrem - deg, 0))*quo, minpoly, p)

        # 交换 f 和 g
        f = g
        g = rem

    # 计算 f 的首项系数的逆 lcfinv
    lcfinv = _gf_gcdex(ring.dmp_LC(f), minpoly, p)[0].set_ring(ring)

    # 返回 f * lcfinv 在 mod minpoly 和 p 下的约简形式
    return _trunc(f * lcfinv, minpoly, p)


# 定义一个函数 `_trial_division`，用于检查多项式 h 是否能整除多项式 f
def _trial_division(f, h, minpoly, p=None):
    # 获取多项式 f 的环
    ring = f.ring

    # 函数文档字符串描述了算法基于伪除法，并且可以在 `\mathbb K` 是 `\mathbb Q` 或 `\mathbb Z_p` 时工作
    # 获取多项式 f 的环
    ring = f.ring

    # 在环的符号顺序中交换变量顺序，生成一个新的环
    zxring = ring.clone(symbols=(ring.symbols[1], ring.symbols[0]))

    # 将 minpoly 的环设定为与 f 相同的环
    minpoly = minpoly.set_ring(ring)

    # 初始化余式为 f
    rem = f

    # 计算余式的最高次数
    degrem = rem.degree()
    # 计算 h 的最高次数
    degh = h.degree()
    # 计算 minpoly 关于 z 的最高次数
    degm = minpoly.degree(1)

    # 计算 h 的首项系数，并将其设定为 f 环的元素
    lch = _LC(h).set_ring(ring)
    # 计算 minpoly 的首项系数
    lcm = minpoly.LC

    # 当余式不为零且余式的次数大于等于 h 的次数时循环执行以下内容
    while rem and degrem >= degh:
        # 计算余式的首项系数，并将其设定为 f 环的元素
        lcrem = _LC(rem).set_ring(ring)
        # 更新余式为 rem * lch - h * t^(degrem - degh) * lcrem
        rem = rem*lch - h.mul_monom((degrem - degh, 0))*lcrem
        # 如果给定了 p，则对余式进行 p 模操作
        if p:
            rem = rem.trunc_ground(p)
        # 更新余式的次数
        degrem = rem.degree(1)

        # 当余式不为零且余式的次数大于等于 minpoly 关于 z 的次数时循环执行以下内容
        while rem and degrem >= degm:
            # 将余式设定为关于 zxring 的环的元素后计算其首项系数
            lcrem = _LC(rem.set_ring(zxring)).set_ring(ring)
            # 更新余式为 rem * minpoly.LC - minpoly * x^degrem-degm * lcrem
            rem = rem.mul_ground(lcm) - minpoly.mul_monom((0, degrem - degm))*lcrem
            # 如果给定了 p，则对余式进行 p 模操作
            if p:
                rem = rem.trunc_ground(p)
            # 更新余式的次数
            degrem = rem.degree(1)

        # 更新余式的次数
        degrem = rem.degree()

    # 返回计算得到的余式
    return rem
def _evaluate_ground(f, i, a):
    r"""
    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground
    domain.
    """
    # 克隆环，并丢弃第 `i` 变量的域
    ring = f.ring.clone(domain=f.ring.domain.ring.drop(i))
    # 初始化结果多项式为零多项式
    fa = ring.zero

    # 遍历多项式 `f` 的每一项
    for monom, coeff in f.iterterms():
        # 在第 `i` 个变量上对多项式 `f` 求值，并将结果添加到结果多项式中
        fa[monom] = coeff.evaluate(i, a)

    # 返回结果多项式
    return fa


def _func_field_modgcd_p(f, g, minpoly, p):
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Z_p(t_1, \ldots, t_k)[z]/(\check m_\alpha(z))[x]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `t_k = a` for suitable `a \in \mathbb Z_p`
    and then calls itself recursively to compute the GCD in
    `\mathbb Z_p(t_1, \ldots, t_{k-1})[z]/(\check m_\alpha(z))[x]`. If these
    recursive calls are successful, the GCD over `k` variables is
    interpolated, otherwise the algorithm returns ``None``. After
    interpolation, Rational Function Reconstruction is used to obtain the
    correct coefficients. If this fails, a new evaluation point has to be
    chosen, otherwise the desired polynomial is obtained by clearing
    denominators. The result is verified with a fraction free trial
    division.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        primitive associate in `\mathbb Z[t_1, \ldots, t_k][x, z]` of the
        GCD of the polynomials `f` and `g`  or ``None``, coefficients are
        in `\left[ -\frac{p-1} 2, \frac{p-1} 2 \right]`

    References
    ==========

    1. [Hoeij04]_

    """
    # 获取多项式 `f` 的环
    ring = f.ring
    # 获取环的域，即 Z[t_1, ..., t_k]
    domain = ring.domain

    # 如果域是多项式环，则获取变量个数 k
    if isinstance(domain, PolynomialRing):
        k = domain.ngens
    else:
        # 如果域不是多项式环，则调用默认的欧几里得算法
        return _euclidean_algorithm(f, g, minpoly, p)

    # 如果 k 等于 1，设置 Q[t_1, ..., t_1] 的域
    if k == 1:
        qdomain = domain.ring.to_field()
    else:
        # 否则，将 t_k 丢弃到地面，并转换为域
        qdomain = domain.ring.drop_to_ground(k - 1)
        qdomain = qdomain.clone(domain=qdomain.domain.ring.to_field())

    # 克隆环，设定域为 qdomain，即 Z(t_k)[t_1, ..., t_{k-1}][x, z]
    qring = ring.clone(domain=qdomain)

    # 初始化 gamma 和 delta
    # gamma 是 Z_p[t_1, ..., t_k][z] 中的多项式
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # delta 是 Z_p[t_1, ..., t_k] 中的多项式
    delta = minpoly.LC

    # 初始化评估点列表和结果列表
    evalpoints = []
    heval = []

    # 初始化 LMlist 和 points
    LMlist = []
    points = list(range(p))
    # 当还有点集合非空时执行循环
    while points:
        # 从点集合中随机选择一个点a并移除它
        a = random.sample(points, 1)[0]
        points.remove(a)

        # 根据k的值选择不同的测试条件
        if k == 1:
            # 计算 delta.evaluate(k-1, a) 是否能整除 p
            test = delta.evaluate(k-1, a) % p == 0
        else:
            # 使用 delta.evaluate(k-1, a) 的地板除法进行测试
            test = delta.evaluate(k-1, a).trunc_ground(p) == 0

        # 如果测试通过则继续下一次循环
        if test:
            continue

        # 计算 gamma 在点 a 处的值
        gammaa = _evaluate_ground(gamma, k-1, a)
        # 计算 minpoly 在点 a 处的值
        minpolya = _evaluate_ground(minpoly, k-1, a)

        # 如果 gamma(a) 对 minpoly(a) 和 gammaa.ring(p) 的余数为零，则继续下一次循环
        if gammaa.rem([minpolya, gammaa.ring(p)]) == 0:
            continue

        # 计算 f 在点 a 处的值
        fa = _evaluate_ground(f, k-1, a)
        # 计算 g 在点 a 处的值
        ga = _evaluate_ground(g, k-1, a)

        # 在 Z_p[x, t_1, ..., t_{k-1}, z]/(minpoly) 中计算 ha = gcd(fa, ga) 的多项式
        ha = _func_field_modgcd_p(fa, ga, minpolya, p)

        # 如果 ha 为 None，则增加 d 的计数并检查是否超过 n，如果超过则返回 None
        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        # 如果 ha 等于 1，则返回 1
        if ha == 1:
            return ha

        # 初始化 LM 为包含 ha 的主导单项式的列表
        LM = [ha.degree()] + [0]*(k-1)
        # 如果 k > 1，则遍历 ha 的每个单项式，更新 LM
        if k > 1:
            for monom, coeff in ha.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        # 将当前点 a 添加到评估点列表中，并将其对应的 ha 添加到 ha 列表中
        evalpoints_a = [a]
        heval_a = [ha]

        # 根据 k 的值设置 m 的初值
        if k == 1:
            m = qring.domain.get_ring().one
        else:
            m = qring.domain.domain.get_ring().one

        # 取得 m 的生成元 t
        t = m.ring.gens[0]

        # 对于 evalpoints 和 heval 中的每一对 b, hb, LMhb，根据 LMhb 是否等于 LM 来更新 evalpoints_a 和 heval_a，并更新 m
        for b, hb, LMhb in zip(evalpoints, heval, LMlist):
            if LMhb == LM:
                evalpoints_a.append(b)
                heval_a.append(hb)
                m *= (t - b)

        # 对 m 进行地板除法并更新 evalpoints, heval 和 LMlist
        m = m.trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        LMlist.append(LM)
        n += 1

        # 在 Z_p[t_1, ..., t_k][x, z] 中插值多变量 h_a
        h = _interpolate_multivariate(evalpoints_a, heval_a, ring, k-1, p, ground=True)

        # 在 Z_p(t_k)[t_1, ..., t_{k-1}][x, z] 中对 h_a 进行有理重建
        h = _rational_reconstruction_func_coeffs(h, p, m, qring, k-1)

        # 如果 h 为 None，则继续下一次循环
        if h is None:
            continue

        # 根据 k 的值选择不同的域和分母初值
        if k == 1:
            dom = qring.domain.field
            den = dom.ring.one
        else:
            dom = qring.domain.domain.field
            den = dom.ring.one

        # 遍历 h 的系数，并使用 gf_lcm 函数计算其分母的最小公倍数
        for coeff in h.itercoeffs():
            if k == 1:
                den = dom.ring.from_dense(gf_lcm(den.to_dense(), coeff.denom.to_dense(), p, dom.domain))
            else:
                for c in coeff.itercoeffs():
                    den = dom.ring.from_dense(gf_lcm(den.to_dense(), c.denom.to_dense(), p, dom.domain))

        # 对 den 进行地板除法并创建新的 qring.domain
        den = qring.domain_new(den.trunc_ground(p))
        # 将 h 乘以 den，并将其转换为 ring 类型，并进行地板除法
        h = ring(h.mul_ground(den).as_expr()).trunc_ground(p)

        # 如果 _trial_division(f, h, minpoly, p) 和 _trial_division(g, h, minpoly, p) 都为 False，则返回 h
        if not _trial_division(f, h, minpoly, p) and not _trial_division(g, h, minpoly, p):
            return h

    # 若循环结束仍未找到合适的 h，则返回 None
    return None
# 从整数 `c` 和模数 `m` 重构有理数 `\frac{a}{b}`。
# 这里的 `c` 和 `m` 是整数。
def _integer_rational_reconstruction(c, m, domain):
    # 如果 `c` 是负数，将其转为非负数以便计算
    if c < 0:
        c += m

    # 初始化 Euclidean 算法所需的变量
    r0, s0 = m, domain.zero
    r1, s1 = c, domain.one

    # 计算边界，作为停止条件，使用平方根函数计算 `m/2` 的平方根
    bound = sqrt(m / 2)  # 如果替换为 `ZZ.sqrt(m // 2)` 仍然正确吗？

    # 执行 Euclidean 算法，直到 `r1` 小于边界值
    while int(r1) >= bound:
        quo = r0 // r1
        r0, r1 = r1, r0 - quo * r1
        s0, s1 = s1, s0 - quo * s1

    # 如果 `s1` 的绝对值大于等于边界值，返回 `None`
    if abs(int(s1)) >= bound:
        return None

    # 根据 `s1` 的符号确定 `a` 和 `b` 的值
    if s1 < 0:
        a, b = -r1, -s1
    elif s1 > 0:
        a, b = r1, s1
    else:
        return None

    # 获取域的字段并返回有理数 `a/b`
    field = domain.get_field()
    return field(a) / field(b)


# 从整数 `c_h_m` 重构多项式 `h` 中每个有理数系数 `c_h`。
# 这里 `h` 是多项式，`c_h_m` 是与 `h` 相对应的整数系数的多项式，
# 满足 `c_h_m = c_h \; \mathrm{mod} \, m`。
def _rational_reconstruction_int_coeffs(hm, m, ring):
    # 初始化零多项式 `h`
    h = ring.zero

    # 根据 `ring` 的类型选择不同的重构方法和域
    if isinstance(ring.domain, PolynomialRing):
        reconstruction = _rational_reconstruction_int_coeffs
        domain = ring.domain.ring
    else:
        reconstruction = _integer_rational_reconstruction
        domain = hm.ring.domain

    # 遍历 `hm` 中的每个单项式和系数
    for monom, coeff in hm.iterterms():
        # 使用指定的重构函数重构系数 `coeff`
        coeffh = reconstruction(coeff, m, domain)

        # 如果重构失败，返回 `None`
        if not coeffh:
            return None

        # 将重构后的系数 `coeffh` 加入到多项式 `h` 中
        h[monom] = coeffh

    # 返回重构后的多项式 `h`
    return h


# 在函数字段上，使用模算法计算两个多项式的最大公因式。
# 这里的多项式位于 `\mathbb{Q}(t_1, \ldots, t_k)[z]/(m_{\alpha}(z))[x]` 中。
def _func_field_modgcd_m(f, g, minpoly):
    # 此函数尚未完整提供，不需要进一步注释。
    pass
    # 获取多项式的环和定义域
    ring = f.ring
    domain = ring.domain

    # 如果定义域是多项式环
    if isinstance(domain, PolynomialRing):
        # 确定变量数量 k
        k = domain.ngens
        # 创建一个有理数域的多项式环 QQdomain
        QQdomain = domain.ring.clone(domain=domain.domain.get_field())
        # 在有理数域上克隆环 QQring
        QQring = ring.clone(domain=QQdomain)
    else:
        # 没有变量时，k 置为 0
        k = 0
        # 在定义域的有理数域上克隆环 QQring
        QQring = ring.clone(domain=ring.domain.get_field())

    # 分解 f 和 g 成为系数和主部分
    cf, f = f.primitive()
    cg, g = g.primitive()

    # 计算 f 和 g 的首项系数乘积
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # 获取 minpoly 的首项系数
    delta = minpoly.LC

    # 初始化 p 为 1， primes 为空列表，hplist 和 LMlist 也初始化为空列表
    p = 1
    primes = []
    hplist = []
    LMlist = []
    # 进入无限循环，直到条件 break 才会跳出循环
    while True:
        # 找到下一个素数
        p = nextprime(p)

        # 使用 gamma.trunc_ground(p) 函数检查 gamma 在 p 下的截断值是否为 0
        if gamma.trunc_ground(p) == 0:
            # 如果为 0，继续下一轮循环
            continue

        # 根据 k 的值选择不同的条件来测试 delta 对 p 的整除性
        if k == 0:
            test = (delta % p == 0)
        else:
            test = (delta.trunc_ground(p) == 0)

        # 如果 test 为真，则继续下一轮循环
        if test:
            continue

        # 计算 f, g, minpoly 在 p 下的截断值
        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        minpolyp = minpoly.trunc_ground(p)

        # 调用 _func_field_modgcd_p 函数计算 fp, gp, minpolyp 对应的结果 hp
        hp = _func_field_modgcd_p(fp, gp, minpolyp, p)

        # 如果 hp 为 None，则继续下一轮循环
        if hp is None:
            continue

        # 如果 hp 等于 1，返回环的单位元素 ring.one
        if hp == 1:
            return ring.one

        # 初始化 LM 列表，首元素为 hp 的次数，其余元素为 0，长度为 k+1
        LM = [hp.degree()] + [0]*k

        # 如果 k 大于 0，对 hp 的每一个项进行迭代
        if k > 0:
            for monom, coeff in hp.iterterms():
                # 如果 monom 的首元素等于 LM 的首元素，并且 coeff.LM 大于 LM[1:]，更新 LM[1:]
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        # 将 hp 赋值给 hm，m 赋值给 p
        hm = hp
        m = p

        # 对于 primes, hplist, LMlist 中的每个 q, hq, LMhq，如果 LMhq 等于 LM
        for q, hq, LMhq in zip(primes, hplist, LMlist):
            if LMhq == LM:
                # 调用 _chinese_remainder_reconstruction_multivariate 函数
                hm = _chinese_remainder_reconstruction_multivariate(hq, hm, q, m)
                # 更新 m 为 m * q
                m *= q

        # 将 p, hp, LM 添加到 primes, hplist, LMlist 中
        primes.append(p)
        hplist.append(hp)
        LMlist.append(LM)

        # 调用 _rational_reconstruction_int_coeffs 函数对 hm 进行有理数重建
        hm = _rational_reconstruction_int_coeffs(hm, m, QQring)

        # 如果 hm 为 None，则继续下一轮循环
        if hm is None:
            continue

        # 根据 k 的值处理 hm
        if k == 0:
            h = hm.clear_denoms()[1]
        else:
            den = domain.domain.one
            for coeff in hm.itercoeffs():
                den = domain.domain.lcm(den, coeff.clear_denoms()[0])
            h = hm.mul_ground(den)

        # 将 h 设置为 ring 环下的元素，并将其返回原始形式的第二个元素
        h = h.set_ring(ring)
        h = h.primitive()[1]

        # 如果不满足 _trial_division(f.mul_ground(cf), h, minpoly) 或 _trial_division(g.mul_ground(cg), h, minpoly) 的条件，则返回 h
        if not (_trial_division(f.mul_ground(cf), h, minpoly) or
                _trial_division(g.mul_ground(cg), h, minpoly)):
            return h
# 将多项式 `f` 转换为 `ring` 环中的关联多项式，`ring` 是多项式环
def _to_ZZ_poly(f, ring):
    r"""
    Compute an associate of a polynomial
    `f \in \mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` in
    `\mathbb Z[x_1, \ldots, x_{n-1}][z] / (\check m_{\alpha}(z))[x_0]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`
    ring : PolyRing
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    Returns
    =======

    f_ : PolyElement
        associate of `f` in
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    """
    # 初始化关联多项式 `f_` 为 `ring` 的零元素
    f_ = ring.zero

    # 确定多项式环的定义域
    if isinstance(ring.domain, PolynomialRing):
        domain = ring.domain.domain
    else:
        domain = ring.domain

    # 计算多项式系数的公共分母
    den = domain.one

    # 迭代 `f` 的每个系数
    for coeff in f.itercoeffs():
        # 迭代系数列表
        for c in coeff.to_list():
            if c:
                # 计算系数的公共分母
                den = domain.lcm(den, c.denominator)

    # 迭代 `f` 的每个单项式和系数
    for monom, coeff in f.iterterms():
        # 将系数转换为列表
        coeff = coeff.to_list()
        m = ring.domain.one

        # 若环为多项式环，则计算单项式的乘积
        if isinstance(ring.domain, PolynomialRing):
            m = m.mul_monom(monom[1:])

        n = len(coeff)

        # 迭代系数列表
        for i in range(n):
            if coeff[i]:
                # 计算调整后的系数
                c = domain.convert(coeff[i] * den) * m

                # 更新关联多项式 `f_` 的系数
                if (monom[0], n-i-1) not in f_:
                    f_[(monom[0], n-i-1)] = c
                else:
                    f_[(monom[0], n-i-1)] += c

    # 返回关联多项式 `f_`
    return f_


# 将多项式 `f` 转换为 `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` 中的多项式
def _to_ANP_poly(f, ring):
    r"""
    Convert a polynomial
    `f \in \mathbb Z[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha}(z))[x_0]`
    to a polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`
    ring : PolyRing
        `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    f_ : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    """
    # 确定环的定义域
    domain = ring.domain

    # 初始化多项式 `f_` 为环 `ring` 的零元素
    f_ = ring.zero

    # 若环为多项式环，则迭代每个单项式和系数
    if isinstance(f.ring.domain, PolynomialRing):
        for monom, coeff in f.iterterms():
            for mon, coef in coeff.iterterms():
                # 构造新的单项式 `m` 和系数 `c`
                m = (monom[0],) + mon
                c = domain([domain.domain(coef)] + [0]*monom[1])

                # 更新多项式 `f_` 的系数
                if m not in f_:
                    f_[m] = c
                else:
                    f_[m] += c

    # 若环为整数环，则迭代每个单项式和系数
    else:
        for monom, coeff in f.iterterms():
            m = (monom[0],)
            c = domain([domain.domain(coeff)] + [0]*monom[1])

            # 更新多项式 `f_` 的系数
            if m not in f_:
                f_[m] = c
            else:
                f_[m] += c

    # 返回多项式 `f_`
    return f_


# 将最小多项式的表示从 `DMP` 改变为给定环的 `PolyElement`
def _minpoly_from_dense(minpoly, ring):
    r"""
    Change representation of the minimal polynomial from ``DMP`` to
    ``PolyElement`` for a given ring.
    """
    # 初始化一个空多项式 minpoly_，用于存储最小多项式的结果
    minpoly_ = ring.zero
    
    # 遍历给定多项式 minpoly 的每一个项，其中 monom 为单项式，coeff 为系数
    for monom, coeff in minpoly.terms():
        # 将单项式 monom 添加到 minpoly_ 中，并且使用 ring 的域(domain)方法获取系数 coeff 的定义域
        minpoly_[monom] = ring.domain(coeff)
    
    # 返回处理后的最小多项式 minpoly_
    return minpoly_
# 计算多项式 `f` 在环 `Q(alpha)[x_0, ..., x_{n-1}]` 中的内容和原始部分
def _primitive_in_x0(f):
    # 获取多项式 `f` 的环
    fring = f.ring
    # 将多项式环降至地域 `Q(alpha)[x_1, ..., x_{n-1}]`
    ring = fring.drop_to_ground(*range(1, fring.ngens))
    # 获取环的定义域
    dom = ring.domain.ring
    # 将 `f` 转换为在新环 `ring` 中的表示
    f_ = ring(f.as_expr())
    # 初始化内容为零元素
    cont = dom.zero

    # 遍历 `f_` 的所有系数
    for coeff in f_.itercoeffs():
        # 使用模块化算法计算内容的原始关联
        cont = func_field_modgcd(cont, coeff)[0]
        # 如果内容已经是单位元，则直接返回内容和多项式 `f`
        if cont == dom.one:
            return cont, f

    # 返回计算得到的内容和 `f` 除以内容后的结果
    return cont, f.quo(cont.set_ring(fring))


# TODO: add support for algebraic function fields
# 计算 `f` 和 `g` 在环 `Q(alpha)[x_0, ..., x_{n-1}]` 中的最大公因式
def func_field_modgcd(f, g):
    # 计算两个多项式 `f` 和 `g` 在特定算法下的最大公因式
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` using a modular algorithm.

    The algorithm first computes the primitive associate
    `\check m_{\alpha}(z)` of the minimal polynomial `m_{\alpha}` in
    `\mathbb{Z}[z]` and the primitive associates of `f` and `g` in
    `\mathbb{Z}[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha})[x_0]`. Then it
    computes the GCD in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]`.
    This is done by calculating the GCD in
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem and Rational Reconstuction. The GCD over
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` is
    computed with a recursive subroutine, which evaluates the polynomials at
    `x_{n-1} = a` for suitable evaluation points `a \in \mathbb Z_p` and
    then calls itself recursively until the ground domain does no longer
    contain any parameters. For
    `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x_0]` the Euclidean Algorithm is
    used. The results of those recursive calls are then interpolated and
    Rational Function Reconstruction is used to obtain the correct
    coefficients. The results, both in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]` and
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]`, are
    verified by a fraction free trial division.

    Apart from the above GCD computation some GCDs in
    `\mathbb Q(\alpha)[x_1, \ldots, x_{n-1}]` have to be calculated,
    because treating the polynomials as univariate ones can result in
    a spurious content of the GCD. For this ``func_field_modgcd`` is
    called recursively.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    h : PolyElement
        monic GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac f h`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac g h`

    Examples
    ========

    >>> from sympy.polys.modulargcd import func_field_modgcd
    >>> from sympy.polys import AlgebraicField, QQ, ring
    >>> from sympy import sqrt
    # 初始化一个代数域 QQ 上的代数扩展域，基础元素是 sqrt(2)
    >>> A = AlgebraicField(QQ, sqrt(2))
    # 在环 R 上定义一个变量 x，R 和 x 是返回的元组
    >>> R, x = ring('x', A)

    # 定义多项式 f = x^2 - 2
    >>> f = x**2 - 2
    # 定义多项式 g = x + sqrt(2)
    >>> g = x + sqrt(2)

    # 计算 f 和 g 的函数场最大公因式 h，以及对应的系数 cff 和 cfg
    >>> h, cff, cfg = func_field_modgcd(f, g)

    # 验证 h 是否等于 x + sqrt(2)
    >>> h == x + sqrt(2)
    True
    # 验证 cff * h 是否等于 f
    >>> cff * h == f
    True
    # 验证 cfg * h 是否等于 g
    >>> cfg * h == g
    True

    # 在环 R 上同时定义两个变量 x 和 y
    >>> R, x, y = ring('x, y', A)

    # 定义多项式 f = x^2 + 2*sqrt(2)*x*y + 2*y^2
    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2
    # 定义多项式 g = x + sqrt(2)*y
    >>> g = x + sqrt(2)*y

    # 计算 f 和 g 的函数场最大公因式 h，以及对应的系数 cff 和 cfg
    >>> h, cff, cfg = func_field_modgcd(f, g)

    # 验证 h 是否等于 x + sqrt(2)*y
    >>> h == x + sqrt(2)*y
    True
    # 验证 cff * h 是否等于 f
    >>> cff * h == f
    True
    # 验证 cfg * h 是否等于 g
    >>> cfg * h == g
    True

    # 定义多项式 f = x + sqrt(2)*y
    >>> f = x + sqrt(2)*y
    # 定义多项式 g = x + y
    >>> g = x + y

    # 计算 f 和 g 的函数场最大公因式 h，以及对应的系数 cff 和 cfg
    >>> h, cff, cfg = func_field_modgcd(f, g)

    # 验证 h 是否等于 R.one，即环 R 的单位元
    >>> h == R.one
    True
    # 验证 cff * h 是否等于 f
    >>> cff * h == f
    True
    # 验证 cfg * h 是否等于 g
    >>> cfg * h == g
    True

    # 引用外部函数 _trivial_gcd，确定输入的 f 和 g 是否有显而易见的最大公因式
    ring = f.ring
    domain = ring.domain
    n = ring.ngens

    # 确保环 ring 和 g 的环相同，并且 domain 是代数域
    assert ring == g.ring and domain.is_Algebraic

    # 如果 _trivial_gcd 返回非空结果，则直接返回该结果
    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    # 创建一个虚拟变量 z
    z = Dummy('z')

    # 在环 ZZring 中创建一个新的多项式环，增加 z 变量，使用 domain 的环作为基础
    ZZring = ring.clone(symbols=ring.symbols + (z,), domain=domain.domain.get_ring())

    # 根据变量数量 n 的不同情况处理多项式 f 和 g
    if n == 1:
        # 将 f 和 g 转换为整数环 ZZring 上的多项式
        f_ = _to_ZZ_poly(f, ZZring)
        g_ = _to_ZZ_poly(g, ZZring)
        # 从 domain.mod 转换为整数环 ZZring 的最小多项式
        minpoly = ZZring.drop(0).from_dense(domain.mod.to_list())

        # 计算函数场上的最大公因式 h
        h = _func_field_modgcd_m(f_, g_, minpoly)
        # 将 h 转换回原始的多项式环 ring
        h = _to_ANP_poly(h, ring)

    else:
        # 将 f 和 g 的原始形式转换为 x_0 的原始形式 contx0f 和 contx0g
        contx0f, f = _primitive_in_x0(f)
        contx0g, g = _primitive_in_x0(g)
        # 计算 x_0 上的函数场最大公因式 contx0h
        contx0h = func_field_modgcd(contx0f, contx0g)[0]

        # 在整数环 ZZring_ 上丢弃第一个变量，并处理剩余的变量
        ZZring_ = ZZring.drop_to_ground(*range(1, n))

        # 将 f 和 g 转换为整数环 ZZring_ 上的多项式
        f_ = _to_ZZ_poly(f, ZZring_)
        g_ = _to_ZZ_poly(g, ZZring_)
        # 从 domain.mod 转换为整数环 ZZring_ 的最小多项式
        minpoly = _minpoly_from_dense(domain.mod, ZZring_.drop(0))

        # 计算函数场上的最大公因式 h
        h = _func_field_modgcd_m(f_, g_, minpoly)
        # 将 h 转换回原始的多项式环 ring
        h = _to_ANP_poly(h, ring)

        # 将 x_0 上的最大公因式 contx0h 转换为 ANP 多项式并与 h 相乘
        contx0h_, h = _primitive_in_x0(h)
        h *= contx0h.set_ring(ring)
        f *= contx0f.set_ring(ring)
        g *= contx0g.set_ring(ring)

    # 对 h 进行地板除以其首项系数 h.LC
    h = h.quo_ground(h.LC)

    # 返回最大公因式 h，以及 f 除以 h 和 g 除以 h 的结果
    return h, f.quo(h), g.quo(h)
```