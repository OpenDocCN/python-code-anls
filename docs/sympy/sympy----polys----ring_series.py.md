# `D:\src\scipysrc\sympy\sympy\polys\ring_series.py`

```
# 导入必要的模块和函数
from sympy.polys.domains import QQ, EX  # 导入QQ和EX域
from sympy.polys.rings import PolyElement, ring, sring  # 导入多项式元素、环和弱环
from sympy.polys.polyerrors import DomainError  # 导入多项式相关的域错误
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,  # 导入单项式操作函数
                                   monomial_ldiv)
from mpmath.libmp.libintmath import ifac  # 导入mpmath中的整数阶乘函数
from sympy.core import PoleError, Function, Expr  # 导入核心函数、表达式和极点错误
from sympy.core.numbers import Rational  # 导入有理数
from sympy.core.intfunc import igcd  # 导入整数最大公约数函数
from sympy.functions import (sin, cos, tan, atan, exp, atanh, tanh, log, ceiling)  # 导入数学函数
from sympy.utilities.misc import as_int  # 导入转换为整数的函数
from mpmath.libmp.libintmath import giant_steps  # 导入mpmath中的巨大步骤函数
import math  # 导入数学库

# 定义一个函数，用于计算一个一元多项式在1/x处的倒数
def _invert_monoms(p1):
    """
    Compute ``x**n * p1(1/x)`` for a univariate polynomial ``p1`` in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _invert_monoms
    >>> R, x = ring('x', ZZ)
    >>> p = x**2 + 2*x + 3
    >>> _invert_monoms(p)
    3*x**2 + 2*x + 1

    See Also
    ========

    sympy.polys.densebasic.dup_reverse
    """
    # 将多项式p1的项按照次数排序
    terms = list(p1.items())
    terms.sort()
    # 计算多项式p1的最高次数
    deg = p1.degree()
    # 从多项式 p1 中获取其环 R（环上的运算对象）
    R = p1.ring
    # 创建一个零多项式 p，使用环 R 的零元素
    p = R.zero
    # 获取多项式 p1 的系数列表
    cv = p1.listcoeffs()
    # 获取多项式 p1 的单项式列表
    mv = p1.listmonoms()
    # 遍历单项式列表 mv 和系数列表 cv，并将每对单项式 mvi 和系数 cvi 插入到 p 中
    for mvi, cvi in zip(mv, cv):
        # 将系数 cvi 插入到多项式 p 的对应单项式位置，计算方式是将总次数 deg 减去单项式 mvi 的第一个元素作为键
        p[(deg - mvi[0],)] = cvi
    # 返回构建好的多项式 p
    return p
# 返回一个精度步长列表，用于牛顿法
def _giant_steps(target):
    res = giant_steps(2, target)  # 调用名为giant_steps的函数，计算从2到目标值的步长列表
    if res[0] != 2:
        res = [2] + res  # 如果步长列表的第一个元素不是2，则在列表开头添加2
    return res  # 返回步长列表

# 在变量x处以精度prec截断系列，即模O(x**prec)
def rs_trunc(p1, x, prec):
    R = p1.ring  # 获取多项式p1的环
    p = R.zero  # 初始化结果多项式p为零多项式
    i = R.gens.index(x)  # 获取变量x在环中的索引
    for exp1 in p1:
        if exp1[i] >= prec:
            continue  # 如果p1中当前项的x的指数大于或等于prec，则跳过该项
        p[exp1] = p1[exp1]  # 否则将p1中当前项复制到结果多项式p中
    return p  # 返回截断后的多项式p

# 检测p是否为x的Puiseux级数
# 如果出现x的负幂次，则引发异常
def rs_is_puiseux(p, x):
    index = p.ring.gens.index(x)  # 获取变量x在环中的索引
    for k in p:
        if k[index] != int(k[index]):
            return True  # 如果某项的x的指数不是整数，则返回True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)  # 如果某项的x的指数为负数，则引发异常
    return False  # 如果所有项的x的指数都是整数且非负数，则返回False

# 返回函数f(p, x, prec)的Puiseux级数
# 仅当函数f仅适用于正规级数时使用
def rs_puiseux(f, p, x, prec):
    index = p.ring.gens.index(x)  # 获取变量x在环中的索引
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = int(n * den // igcd(n, den))  # 计算x的所有指数的公分母
        elif power != int(power):
            den = power.denominator
            n = int(n * den // igcd(n, den))  # 计算x的所有指数的公分母
    if n != 1:
        p1 = pow_xin(p, index, n)  # 将p中所有的x的指数乘以n
        r = f(p1, x, prec * n)  # 调用函数f，传入乘以n后的多项式p1和相应的精度
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])  # 将结果中所有的x的指数除以n
        else:
            r = pow_xin(r, index, n1)  # 将结果中所有的x的指数除以n
    else:
        r = f(p, x, prec)  # 如果n为1，则直接调用函数f，传入原始多项式p和相应的精度
    return r  # 返回Puiseux级数

# 返回函数f(p, q, x, prec)的Puiseux级数
# 仅当函数f仅适用于正规级数时使用
def rs_puiseux2(f, p, q, x, prec):
    index = p.ring.gens.index(x)  # 获取变量x在环中的索引
    n = 1
    # 遍历列表 p 中的每个元素 k
    for k in p:
        # 获取 k 元素中索引为 index 的值，赋给变量 power
        power = k[index]
        # 如果 power 是 Rational 类型的实例
        if isinstance(power, Rational):
            # 将 power 转换为分子和分母的形式
            num, den = power.as_numer_denom()
            # 计算 n 与 den 的最大公约数，更新 n 的值
            n = n * den // igcd(n, den)
        # 如果 power 不是整数
        elif power != int(power):
            # 获取 power 的分母
            den = power.denominator
            # 计算 n 与 den 的最大公约数，更新 n 的值
            n = n * den // igcd(n, den)
    # 如果 n 不等于 1
    if n != 1:
        # 将 p 中的每个元素 k 都乘以 n，得到新的列表 p1
        p1 = pow_xin(p, index, n)
        # 调用函数 f 处理 p1、q、x、prec*n，得到结果 r
        r = f(p1, q, x, prec * n)
        # 创建有理数 QQ(1, n)，赋给 n1
        n1 = QQ(1, n)
        # 将 r 中的每个元素 k 都乘以 n1，得到新的结果 r
        r = pow_xin(r, index, n1)
    else:
        # 如果 n 等于 1，则直接调用函数 f 处理 p、q、x、prec，得到结果 r
        r = f(p, q, x, prec)
    # 返回最终的结果 r
    return r
# 返回两个级数的乘积，取模于 ``O(x**prec)``。
# ``x`` 是级数变量或其在生成器中的位置。
def rs_mul(p1, p2, x, prec):
    R = p1.ring  # 获取第一个多项式的环
    p = R.zero  # 初始化结果多项式为零多项式
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')  # 如果两个多项式的环不同，抛出错误
    iv = R.gens.index(x)  # 获取变量 `x` 在生成器中的索引
    if not isinstance(p2, PolyElement):
        raise ValueError('p2 must be a polynomial')  # 如果 `p2` 不是多项式，抛出错误
    if R == p2.ring:
        get = p.get  # 获取多项式 `p` 的元素
        items2 = list(p2.items())  # 获取 `p2` 的所有项并转换为列表
        items2.sort(key=lambda e: e[0][iv])  # 根据项的 `iv` 索引排序
        if R.ngens == 1:
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    exp = exp1[0] + exp2[0]  # 计算指数的总和
                    if exp < prec:
                        exp = (exp, )
                        p[exp] = get(exp, 0) + v1*v2  # 计算乘积并添加到结果多项式 `p` 中
                    else:
                        break  # 如果指数总和超过 `prec`，终止循环
        else:
            monomial_mul = R.monomial_mul  # 获取多项式环的单项式乘法
            for exp1, v1 in p1.items():
                for exp2, v2 in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)  # 计算单项式乘积
                        p[exp] = get(exp, 0) + v1*v2  # 计算乘积并添加到结果多项式 `p` 中
                    else:
                        break  # 如果指数总和超过 `prec`，终止循环

    p.strip_zero()  # 移除多项式 `p` 中系数为零的项
    return p  # 返回乘积多项式

# 对给定的级数进行平方操作，取模于 ``O(x**prec)``
def rs_square(p1, x, prec):
    R = p1.ring  # 获取多项式 `p1` 的环
    p = R.zero  # 初始化结果多项式为零多项式
    iv = R.gens.index(x)  # 获取变量 `x` 在生成器中的索引
    get = p.get  # 获取多项式 `p` 的元素
    items = list(p1.items())  # 获取 `p1` 的所有项并转换为列表
    items.sort(key=lambda e: e[0][iv])  # 根据项的 `iv` 索引排序
    monomial_mul = R.monomial_mul  # 获取多项式环的单项式乘法
    for i in range(len(items)):
        exp1, v1 = items[i]
        for j in range(i):
            exp2, v2 = items[j]
            if exp1[iv] + exp2[iv] < prec:
                exp = monomial_mul(exp1, exp2)  # 计算单项式乘积
                p[exp] = get(exp, 0) + v1*v2  # 计算乘积并添加到结果多项式 `p` 中
            else:
                break  # 如果指数总和超过 `prec`，终止循环
    p = p.imul_num(2)  # 结果乘以 2
    get = p.get  # 获取多项式 `p` 的元素
    for expv, v in p1.items():
        if 2*expv[iv] < prec:
            e2 = monomial_mul(expv, expv)  # 计算单项式的平方
            p[e2] = get(e2, 0) + v**2  # 计算平方并添加到结果多项式 `p` 中
    p.strip_zero()  # 移除多项式 `p` 中系数为零的项
    return p  # 返回结果多项式

# 返回 `p1` 的 `n` 次幂，取模于 ``O(x**prec)``
def rs_pow(p1, n, x, prec):
    R = p1.ring  # 获取多项式 `p1` 的环
    # 如果 n 是 Rational 类型的实例，处理有理数幂运算
    if isinstance(n, Rational):
        # 将 n 的分子和分母转换为整数
        np = int(n.p)
        nq = int(n.q)
        # 如果 n 的分母不为 1，先计算其 nq 次根号，再计算 np 次幂
        if nq != 1:
            res = rs_nth_root(p1, nq, x, prec)
            if np != 1:
                res = rs_pow(res, np, x, prec)
        else:
            # 如果 n 的分母为 1，直接计算 p1 的 np 次幂
            res = rs_pow(p1, np, x, prec)
        # 返回计算结果
        return res

    # 将 n 转换为整数
    n = as_int(n)
    # 处理特殊情况：如果 n 等于 0
    if n == 0:
        # 如果 p1 不为零，返回 1；否则抛出异常，因为 0**0 未定义
        if p1:
            return R(1)
        else:
            raise ValueError('0**0 is undefined')
    # 处理负指数 n 的情况
    if n < 0:
        # 计算 p1 的 -n 次幂，然后调用逆序级数函数处理
        p1 = rs_pow(p1, -n, x, prec)
        return rs_series_inversion(p1, x, prec)
    # 处理 n 等于 1 的情况，返回 p1 的截断函数值
    if n == 1:
        return rs_trunc(p1, x, prec)
    # 处理 n 等于 2 的情况，返回 p1 的平方函数值
    if n == 2:
        return rs_square(p1, x, prec)
    # 处理 n 等于 3 的情况，返回 p1 的平方和函数值
    if n == 3:
        p2 = rs_square(p1, x, prec)
        return rs_mul(p1, p2, x, prec)
    
    # 处理一般的指数 n 的情况，使用二进制展开的快速幂算法
    p = R(1)
    while 1:
        if n & 1:
            # 如果 n 是奇数，累乘 p1 到 p
            p = rs_mul(p1, p, x, prec)
            n -= 1
            if not n:
                break
        # 将 p1 平方，同时将 n 除以 2
        p1 = rs_square(p1, x, prec)
        n = n // 2
    
    # 返回计算结果 p
    return p
# 按照给定规则进行替换和截断，返回带有精度 ``prec`` 的生成器 ``x`` 中的级数
def rs_subs(p, rules, x, prec):
    R = p.ring  # 获取多项式环
    ngens = R.ngens  # 获取生成器数量
    d = R(0)  # 初始化一个零多项式
    # 为每个生成器设置默认的一阶幂次项
    for i in range(ngens):
        d[(i, 1)] = R.gens[i]
    # 根据给定规则更新每个变量的对应幂次项
    for var in rules:
        d[(R.index(var), 1)] = rules[var]
    p1 = R(0)  # 初始化结果多项式
    p_keys = sorted(p.keys())  # 对多项式的指数进行排序
    # 对于每个指数进行计算
    for expv in p_keys:
        p2 = R(1)  # 初始化一个单项式
        # 对每个生成器进行计算
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            # 如果幂次项不在字典中，根据不同情况进行计算
            if (i, power) not in d:
                q, r = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[(i, power)] = rs_square(d[(i, q)], x, prec)
                elif (i, power - 1) in d:
                    d[(i, power)] = rs_mul(d[(i, power - 1)], d[(i, 1)],
                                           x, prec)
                else:
                    d[(i, power)] = rs_pow(d[(i, 1)], power, x, prec)
            # 计算当前生成器的幂次项对应的多项式
            p2 = rs_mul(p2, d[(i, power)], x, prec)
        # 将计算得到的单项式加入到结果多项式中
        p1 += p2 * p[expv]
    return p1  # 返回最终计算结果的多项式

# 检查多项式 ``p`` 是否在变量 ``x`` 中有常数项
def _has_constant_term(p, x):
    R = p.ring  # 获取多项式环
    iv = R.gens.index(x)  # 获取变量在生成器列表中的索引
    zm = R.zero_monom  # 获取零单项式
    a = [0] * R.ngens  # 初始化生成器系数列表
    a[iv] = 1  # 将变量对应的生成器系数设置为1，其余为0
    miv = tuple(a)  # 转换为元组形式
    # 检查是否存在与零单项式相同的最小单项式
    return any(monomial_min(expv, miv) == zm for expv in p)

# 返回多项式 ``p`` 关于变量 ``x`` 的常数项
def _get_constant_term(p, x):
    """Return constant term in p with respect to x

    Note that it is not simply `p[R.zero_monom]` as there might be multiple
    generators in the ring R. We want the `x`-free term which can contain other
    generators.
    """
    R = p.ring  # 获取多项式环
    # 获取元素 x 在 R.gens 中的索引
    i = R.gens.index(x)
    # 获取 R.zero_monom 的值，通常表示零单项式
    zm = R.zero_monom
    # 创建一个长度为 R.ngens 的列表 a，并将索引 i 处的元素设置为 1
    a = [0]*R.ngens
    a[i] = 1
    # 将列表 a 转换为元组 miv
    miv = tuple(a)
    # 初始化计数器 c
    c = 0
    # 遍历 p 中的每个指数 expv
    for expv in p:
        # 检查 expv 和 miv 的最小单项式是否为 zm
        if monomial_min(expv, miv) == zm:
            # 如果是，则将 R({expv: p[expv]}) 加到 c 上
            c += R({expv: p[expv]})
    # 返回计数器 c
    return c
# 计算变量 x 在环中的索引位置
index = p.ring.gens.index(x)

# 找到包含最小索引值的项
m = min(p, key=lambda k: k[index])[index]

# 如果找到的最小值小于 0，则抛出异常，指明尚未实现围绕 [oo] 的渐近展开
if m < 0:
    raise PoleError("Asymptotic expansion of %s around [oo] not "
                    "implemented." % name)

# 返回变量 x 的索引和最小值 m
return index, m



# 对 p 进行单变量级数求逆，模 O(x**prec)
# 使用牛顿法进行求逆

if rs_is_puiseux(p, x):
    # 如果 p 是 Puiseux 级数，则调用 rs_puiseux 进行处理
    return rs_puiseux(_series_inversion1, p, x, prec)

R = p.ring
zm = R.zero_monom
c = p[zm]

# 处理特殊情况，当 prec 是整数时，转换为整型
if prec == int(prec):
    prec = int(prec)

# 检查 p 中是否存在常数项
if zm not in p:
    raise ValueError("No constant term in series")

# 检查 p-c 是否包含与参数有关的常数项
if _has_constant_term(p - c, x):
    raise ValueError("p cannot contain a constant term depending on "
                     "parameters")

# 设置 R.domain 的单位元为 1
one = R(1)
if R.domain is EX:
    one = 1

# 如果 p 的常数项不是 1，则计算其倒数
if c != one:
    p1 = R(1)/c
else:
    p1 = R(1)

# 使用巨大步长计算求逆过程
for precx in _giant_steps(prec):
    t = 1 - rs_mul(p1, p, x, precx)
    p1 = p1 + rs_mul(p1, t, x, precx)

# 返回计算得到的 p 的倒数
return p1



# 多变量级数求逆 ``1/p`` 模 ``O(x**prec)``
R = p.ring

# 如果 p 是零多项式，则抛出除零异常
if p == R.zero:
    raise ZeroDivisionError

zm = R.zero_monom
index = R.gens.index(x)

# 找到 p 中所有变量的最小指数
m = min(p, key=lambda k: k[index])[index]

# 如果存在非零最小指数，则调整 p 和 prec
if m:
    p = mul_xin(p, index, -m)
    prec = prec + m

# 检查 p 中是否存在常数项
if zm not in p:
    raise NotImplementedError("No constant term in series")

# 检查 p - p[0] 是否有常数项
if _has_constant_term(p - p[zm], x):
    raise NotImplementedError("p - p[0] must not have a constant term in "
                              "the series variables")

# 调用单变量级数求逆函数 _series_inversion1 计算 p 的逆
r = _series_inversion1(p, x, prec)

# 如果 m 不为零，则将结果乘以 x^(-m) 进行修正
if m != 0:
    r = mul_xin(r, index, -m)

# 返回计算得到的多变量级数求逆结果
return r



# 计算多项式 p 中给定指数 t = (i, j) 的系数，其中 i 是索引，j 是指数
i, j = t
R = p.ring

# 创建一个包含 j 的列表，并将其作为指数向量 expv1
expv1 = [0]*R.ngens
expv1[i] = j
expv1 = tuple(expv1)

# 初始化系数 p1 为零
p1 = R(0)
    # 遍历列表 p 中的每个指数 expv
    for expv in p:
        # 检查当前指数 expv 的第 i 个位置是否等于 j
        if expv[i] == j:
            # 如果条件成立，将 p1 字典中的键设置为 expv 除以 expv1 的结果，对应值为 p 中 expv 的值
            p1[monomial_div(expv, expv1)] = p[expv]
    # 返回更新后的字典 p1
    return p1
def rs_series_from_list(p, c, x, prec, concur=1):
    """
    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.

    It reduces the number of multiplications by summing concurrently.

    `ax = [1, p, p**2, .., p**(J - 1)]`
    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`
    with `K >= (n + 1)/J`

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> c = [1, 2, 3]
    >>> rs_series_from_list(p, c, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> pc = R.from_list(list(reversed(c)))
    >>> rs_trunc(pc.compose(x, p), x, 4)
    6*x**3 + 11*x**2 + 8*x + 6

    """

    # TODO: Add this when it is documented in Sphinx
    """
    See Also
    ========

    sympy.polys.rings.PolyRing.compose

    """
    R = p.ring
    # 计算输入列表 c 的长度
    n = len(c)
    # 如果 concur 为 False，执行以下操作
    if not concur:
        # 使用 R(1) 初始化 q
        q = R(1)
        # 初始化 s 为 c[0] 乘以 q
        s = c[0]*q
        # 对于列表 c 中的每个元素（从索引 1 到 n-1）
        for i in range(1, n):
            # 计算 q = rs_mul(q, p, x, prec)
            q = rs_mul(q, p, x, prec)
            # 计算 s += c[i]*q
            s += c[i]*q
        # 返回累加结果 s
        return s
    # 计算 J 为 n 的平方根加 1 的整数部分
    J = int(math.sqrt(n) + 1)
    # 计算 K 和 r 分别为 n 除以 J 的商和余数
    K, r = divmod(n, J)
    # 如果有余数 r，则 K 值加 1
    if r:
        K += 1
    # 初始化列表 ax，第一个元素为 R(1)
    ax = [R(1)]
    # 初始化 q 为 R(1)
    q = R(1)
    # 如果 p 的长度小于 20，执行以下操作
    if len(p) < 20:
        # 对于列表 ax 中的每个索引 i（从 1 到 J-1）
        for i in range(1, J):
            # 计算 q = rs_mul(q, p, x, prec)
            q = rs_mul(q, p, x, prec)
            # 将 q 添加到列表 ax 中
            ax.append(q)
    else:
        # 对于列表 ax 中的每个索引 i（从 1 到 J-1）
        for i in range(1, J):
            # 如果 i 是偶数
            if i % 2 == 0:
                # 计算 q = rs_square(ax[i//2], x, prec)
                q = rs_square(ax[i//2], x, prec)
            else:
                # 计算 q = rs_mul(q, p, x, prec)
                q = rs_mul(q, p, x, prec)
            # 将 q 添加到列表 ax 中
            ax.append(q)
    # 使用 rs_mul 优化计算 pj
    pj = rs_mul(ax[-1], p, x, prec)
    # 初始化 b 为 R(1)
    b = R(1)
    # 初始化 s 为 R(0)
    s = R(0)
    # 对于每个 k 在 K-1 范围内的整数
    for k in range(K - 1):
        # 计算 r = J*k
        r = J*k
        # 初始化 s1 为 c[r]
        s1 = c[r]
        # 对于每个 j 在 1 到 J-1 范围内的整数
        for j in range(1, J):
            # 计算 s1 += c[r + j]*ax[j]
            s1 += c[r + j]*ax[j]
        # 计算 s1 = rs_mul(s1, b, x, prec)
        s1 = rs_mul(s1, b, x, prec)
        # 计算 s += s1
        s += s1
        # 计算 b = rs_mul(b, pj, x, prec)
        b = rs_mul(b, pj, x, prec)
        # 如果 b 为假值，则跳出循环
        if not b:
            break
    # 设置 k 为 K-1
    k = K - 1
    # 计算 r = J*k
    r = J*k
    # 如果 r 小于 n
    if r < n:
        # 初始化 s1 为 c[r]*R(1)
        s1 = c[r]*R(1)
        # 对于每个 j 在 1 到 J-1 范围内的整数
        for j in range(1, J):
            # 如果 r + j 大于等于 n，则跳出循环
            if r + j >= n:
                break
            # 计算 s1 += c[r + j]*ax[j]
            s1 += c[r + j]*ax[j]
        # 计算 s1 = rs_mul(s1, b, x, prec)
        s1 = rs_mul(s1, b, x, prec)
        # 计算 s += s1
        s += s1
    # 返回累加结果 s
    return s
def rs_fun(p, f, *args):
    r"""
    Function of a multivariate series computed by substitution.

    The case with f method name is used to compute `rs\_tan` and `rs\_nth\_root`
    of a multivariate series:

        `rs\_fun(p, tan, iv, prec)`

        tan series is first computed for a dummy variable _x,
        i.e, `rs\_tan(\_x, iv, prec)`. Then we substitute _x with p to get the
        desired series

    Parameters
    ==========

    p : :class:`~.PolyElement` The multivariate series to be expanded.
    f : `ring\_series` function to be applied on `p`.
    args[-2] : :class:`~.PolyElement` with respect to which, the series is to be expanded.
    args[-1] : Required order of the expanded series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_fun, _tan1
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x*y + x**2*y + x**3*y**2
    >>> rs_fun(p, _tan1, x, 4)
    1/3*x**3*y**3 + 2*x**3*y**2 + x**3*y + 1/3*x**3 + x**2*y + x*y + x

    """
    _R = p.ring  # 获取多项式 p 所在的环 _R
    R1, _x = ring('_x', _R.domain)  # 创建一个只包含 _x 的环 R1
    h = int(args[-1])  # 获取最后一个参数作为扩展系列的所需阶数 h
    args1 = args[:-2] + (_x, h)  # 准备用于调用函数 f 的参数列表
    zm = _R.zero_monom  # 获取环 _R 中的零单项式
    # 分离系列的常数项
    # 计算单变量系列 f(_x, .., 'x', sum(nv))
    if zm in p:
        x1 = _x + p[zm]  # 创建包含常数项的新变量 x1
        p1 = p - p[zm]   # 从 p 中减去常数项，得到新系列 p1
    else:
        x1 = _x  # 如果没有常数项，直接使用 _x
        p1 = p
    if isinstance(f, str):
        q = getattr(x1, f)(*args1)  # 如果 f 是字符串，则调用 _x 的对应方法
    else:
        q = f(x1, *args1)  # 否则直接调用函数 f
    a = sorted(q.items())  # 对 q 中的项按照键排序，返回一个列表
    c = [0]*h  # 创建一个长度为 h 的零列表
    # 对列表 a 中的每个元素 x 进行遍历
    for x in a:
        # 将 x 的第一个元素的第一个字符作为键，第二个元素作为值，存入字典 c 中
        c[x[0][0]] = x[1]
    # 调用 rs_series_from_list 函数，传入参数 p1, c, args[-2], args[-1]，并将结果赋给变量 p1
    p1 = rs_series_from_list(p1, c, args[-2], args[-1])
    # 返回变量 p1 的值作为函数的返回结果
    return p1
# 返回多项式 p 乘以 x_i 的 n 次方后的结果
def mul_xin(p, i, n):
    R = p.ring  # 获取多项式 p 的环
    q = R(0)    # 创建一个环 R 中的零元素
    for k, v in p.items():
        k1 = list(k)   # 将多项式 p 中的指数列表转换为可变列表 k1
        k1[i] += n     # 将第 i 个变量的指数增加 n
        q[tuple(k1)] = v  # 将修改后的指数列表 k1 和对应的系数 v 添加到结果多项式 q 中
    return q

# 返回多项式 p 的第 i 个变量 x_i 的指数乘以 n 次方后的结果
def pow_xin(p, i, n):
    R = p.ring  # 获取多项式 p 的环
    q = R(0)    # 创建一个环 R 中的零元素
    for k, v in p.items():
        k1 = list(k)   # 将多项式 p 中的指数列表转换为可变列表 k1
        k1[i] *= n     # 将第 i 个变量的指数乘以 n
        q[tuple(k1)] = v  # 将修改后的指数列表 k1 和对应的系数 v 添加到结果多项式 q 中
    return q

# 计算多项式 p 的 x 变量的 n 次根的单变量级数展开
def _nth_root1(p, n, x, prec):
    if rs_is_puiseux(p, x):  # 检查多项式 p 是否为 x 的普斯代数级数
        return rs_puiseux2(_nth_root1, p, n, x, prec)  # 若是，则调用普斯代数级数的根计算函数
    R = p.ring  # 获取多项式 p 的环
    zm = R.zero_monom  # 获取环 R 的零单项
    if zm not in p:  # 如果 p 中不包含零单项
        raise NotImplementedError('No constant term in series')  # 抛出未实现错误
    n = as_int(n)  # 将 n 转换为整数
    assert p[zm] == 1  # 断言多项式 p 的常数项为 1
    p1 = R(1)   # 创建环 R 中的单位元素
    if p == 1:  # 如果 p 等于 1
        return p   # 返回 1
    if n == 0:  # 如果 n 等于 0
        return R(1)  # 返回环 R 中的单位元素
    if n == 1:  # 如果 n 等于 1
        return p   # 返回多项式 p
    if n < 0:   # 如果 n 小于 0
        n = -n   # 取 n 的相反数
        sign = 1  # 设置符号标志为 1
    else:   # 否则
        sign = 0  # 设置符号标志为 0
    for precx in _giant_steps(prec):  # 遍历精度步骤迭代器
        tmp = rs_pow(p1, n + 1, x, precx)  # 计算 p1 的 n+1 次幂
        tmp = rs_mul(tmp, p, x, precx)     # 计算 tmp 与 p 的乘积
        p1 += p1/n - tmp/n   # 更新 p1 的值
    if sign:  # 如果符号标志为真
        return p1   # 返回 p1
    else:   # 否则
        return _series_inversion1(p1, x, prec)  # 返回 p1 的级数倒数

# 计算多项式 p 的 x 变量的 n 次根的多变量级数展开
def rs_nth_root(p, n, x, prec):
    if n == 0:   # 如果 n 等于 0
        if p == 0:   # 如果 p 等于 0
            raise ValueError('0**0 expression')  # 抛出值错误
        else:   # 否则
            return p.ring(1)   # 返回环中的单位元素
    if n == 1:   # 如果 n 等于 1
        return rs_trunc(p, x, prec)   # 返回 p 对 x 的截断级数展开
    R = p.ring   # 获取多项式 p 的环
    index = R.gens.index(x)   # 获取变量 x 在环中的索引
    m = min(p, key=lambda k: k[index])[index]   # 计算 p 中变量 x 的最小指数
    p = mul_xin(p, index, -m)   # 将多项式 p 中的变量 x 的指数乘以 -m
    prec -= m   # 更新精度值
    # 如果 p-1 中存在常数项，则执行以下代码块
    if _has_constant_term(p - 1, x):
        # 获取零次单项式
        zm = R.zero_monom
        # 获取多项式 p 的常数项
        c = p[zm]
        
        # 如果环 R 是 EX，即表达式环
        if R.domain is EX:
            # 将常数项转换为表达式
            c_expr = c.as_expr()
            # 计算常数项的 n 次根
            const = c_expr**QQ(1, n)
        
        # 如果常数项是多项式元素
        elif isinstance(c, PolyElement):
            try:
                # 将常数项转换为表达式
                c_expr = c.as_expr()
                # 计算常数项的 n 次根，并将结果转换为环 R 中的元素
                const = R(c_expr**(QQ(1, n)))
            except ValueError:
                # 抛出域错误异常，指明在该域中无法展开给定的序列
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        
        # 如果常数项不是表达式或多项式元素
        else:
            try:
                # 使用有理数作为指数，计算常数项的 n 次根，并将结果转换为环 R 中的元素
                const = R(c**Rational(1, n))
            except ValueError:
                # 抛出域错误异常，指明在该域中无法展开给定的序列
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        
        # 将 p 除以常数项 c 的 n 次根，计算结果并乘以常数项的 n 次根
        res = rs_nth_root(p/c, n, x, prec)*const
    
    # 如果 p-1 中不存在常数项，则执行以下代码块
    else:
        # 调用 _nth_root1 函数计算 p 的 n 次根
        res = _nth_root1(p, n, x, prec)
    
    # 如果 m 不为零
    if m:
        # 将 m 转换为有理数
        m = QQ(m, n)
        # 对结果 res 进行乘法操作，乘以 x^index 以及 m
        res = mul_xin(res, index, m)
    
    # 返回计算结果 res
    return res
# 计算给定多项式 p 对于变量 x 和精度 prec 的对数的级数展开
def rs_log(p, x, prec):
    """
    The Logarithm of ``p`` modulo ``O(x**prec)``.

    Notes
    =====

    Truncation of ``integral dx p**-1*d p/dx`` is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_log
    >>> R, x = ring('x', QQ)
    >>> rs_log(1 + x, x, 8)
    1/7*x**7 - 1/6*x**6 + 1/5*x**5 - 1/4*x**4 + 1/3*x**3 - 1/2*x**2 + x
    >>> rs_log(x**QQ(3, 2) + 1, x, 5)
    1/3*x**(9/2) - 1/2*x**3 + x**(3/2)
    """
    # 检查多项式 p 是否是 Puiseux 级数
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    
    # 获取多项式的环 R
    R = p.ring
    
    # 如果 p 是常数多项式 1，直接返回 R 的零元素
    if p == 1:
        return R.zero
    
    # 获取多项式 p 的常数项 c
    c = _get_constant_term(p, x)
    
    # 如果存在常数项 c
    if c:
        const = 0
        
        # 如果常数项 c 是 1，pass
        if c == 1:
            pass
        else:
            c_expr = c.as_expr()
            
            # 根据环的定义进行对数运算
            if R.domain is EX:
                const = log(c_expr)
            elif isinstance(c, PolyElement):
                try:
                    const = R(log(c_expr))
                except ValueError:
                    # 尝试添加新的生成元到环 R 中
                    R = R.add_gens([log(c_expr)])
                    p = p.set_ring(R)
                    x = x.set_ring(R)
                    c = c.set_ring(R)
                    const = R(log(c_expr))
            else:
                try:
                    const = R(log(c))
                except ValueError:
                    raise DomainError("The given series cannot be expanded in "
                                      "this domain.")
        
        # 计算 p 对 x 的微分
        dlog = p.diff(x)
        
        # 计算 rs_mul 函数的结果，对 p 的逆序列进行级数展开
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        
        # 返回积分和常数项之和
        return rs_integrate(dlog, x) + const
    else:
        # 如果不存在常数项 c，抛出未实现的错误
        raise NotImplementedError


# 计算主枝的 Lambert W 函数的级数展开
def rs_LambertW(p, x, prec):
    """
    Calculate the series expansion of the principal branch of the Lambert W
    function.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_LambertW
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_LambertW(x + x*y, x, 3)
    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x

    See Also
    ========

    LambertW
    """
    # 检查多项式 p 是否是 Puiseux 级数
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    
    # 获取多项式的环 R
    R = p.ring
    p1 = R(0)
    
    # 如果多项式 p 含有常数项，则抛出未实现的错误
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "the series variables")
    
    # 如果变量 x 在多项式的生成元中
    if x in R.gens:
        # 使用巨大步长算法进行级数展开
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        
        # 返回级数展开的结果
        return p1
    else:
        # 如果变量 x 不在生成元中，抛出未实现的错误
        raise NotImplementedError


# rs_exp 函数的辅助函数，计算级数展开的结果
def _exp1(p, x, prec):
    r"""Helper function for `rs\_exp`. """
    # 获取多项式的环 R
    R = p.ring
    p1 = R(1)
    
    # 使用巨大步长算法进行级数展开
    for precx in _giant_steps(prec):
        pt = p - rs_log(p1, x, precx)
        tmp = rs_mul(pt, p1, x, precx)
        p1 += tmp
    
    # 返回级数展开的结果
    return p1
# 如果输入的级数是普韦苏级数，则使用普韦苏级数函数进行计算，并返回结果
if rs_is_puiseux(p, x):
    return rs_puiseux(rs_atan, p, x, prec)

# 获取级数环
R = p.ring

# 获取级数中的常数项
c = _get_constant_term(p, x)

# 如果存在常数项
if c:
    # 如果环为 EX 域，将常数项转换为表达式，并计算其指数函数
    if R.domain is EX:
        c_expr = c.as_expr()
        const = exp(c_expr)
    # 如果常数项是多项式元素
    elif isinstance(c, PolyElement):
        try:
            c_expr = c.as_expr()
            const = R(exp(c_expr))
        except ValueError:
            # 如果计算出错，扩展环并重新计算常数项的指数函数
            R = R.add_gens([exp(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(exp(c_expr))
    else:
        try:
            # 计算常数项的指数函数
            const = R(exp(c))
        except ValueError:
            # 如果在此环中不能扩展级数，抛出域错误
            raise DomainError("The given series cannot be expanded in "
                "this domain.")

    # 将常数项从级数中移除
    p1 = p - c

    # 递归调用 rs_exp 函数，将常数项的指数函数乘以剩余的级数并返回结果
    return const * rs_exp(p1, x, prec)

# 如果级数的长度大于 20，调用 _exp1 函数处理
if len(p) > 20:
    return _exp1(p, x, prec)

# 初始化单位元素和计数变量
one = R(1)
n = 1
c = []

# 生成级数的系数列表，用于构造级数
for k in range(prec):
    c.append(one / n)
    k += 1
    n *= k

# 使用系数列表生成级数，并返回结果
r = rs_series_from_list(p, c, x, prec)
return r
    # 如果多项式 p 中包含常数项
    if _has_constant_term(p, x):
        # 获取零单项式
        zm = R.zero_monom
        # 提取常数项 c
        c = p[zm]
        # 如果域是 EX（表达式域），将常数项转换为表达式并计算其反正切值
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atan(c_expr)
        # 如果常数项是多项式元素，则尝试将其转换为表达式并计算反正切值
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atan(c_expr))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        # 否则直接计算常数项的反正切值
        else:
            try:
                const = R(atan(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")

    # 不使用闭式公式，而是对 atan(p) 进行微分得到 `1/(1+p**2) * dp`，其级数展开更容易计算。
    # 最后进行积分以恢复到 atan 函数
    dp = p.diff(x)
    # 计算 `1 + p**2` 的逆级数展开
    p1 = rs_square(p, x, prec) + R(1)
    # 对逆级数展开进行级数求逆
    p1 = rs_series_inversion(p1, x, prec - 1)
    # 计算 `dp * p1` 的级数乘积
    p1 = rs_mul(dp, p1, x, prec - 1)
    # 对结果级数进行积分，并加上之前计算的常数项 const
    return rs_integrate(p1, x) + const
# 切割方法，对于输入的精度进行切割
def _giant_steps(prec):
    # 返回一个生成器，该生成器提供与输入精度相关的切割值
    return (1 << i for i in range(1, prec + 1).bit_length())

# rs_asin函数：对于给定的多项式p，关于变量x进行asin函数的级数展开
def rs_asin(p, x, prec):
    # 如果p是Puisieux级数，则调用Puisieux级数版本的rs_asin函数
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asin, p, x, prec)
    # 如果p中有常数项，则抛出未实现的错误
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "series variables")
    # 获取多项式p所在的环R
    R = p.ring
    # 如果x是环R的生成元之一
    if x in R.gens:
        # 获取一个好的值
        if len(p) > 20:
            # 对p进行微分，得到dp
            dp = rs_diff(p, x)
            # 计算1 - p^2的级数展开，精度为prec-1
            p1 = 1 - rs_square(p, x, prec - 1)
            # 计算(-2)次根的级数展开，精度为prec-1
            p1 = rs_nth_root(p1, -2, x, prec - 1)
            # 计算dp与p1的乘积的级数展开，精度为prec-1
            p1 = rs_mul(dp, p1, x, prec - 1)
            # 返回p1的积分的级数展开
            return rs_integrate(p1, x)
        # 设置一个单位值
        one = R(1)
        # 初始化系数c为[0, 1, 0]
        c = [0, one, 0]
        # 生成奇数次数的级数展开系数，范围为3到prec，步长为2
        for k in range(3, prec, 2):
            c.append((k - 2)**2*c[-2]/(k*(k - 1)))
            c.append(0)
        # 使用系数c对p进行级数展开，精度为prec
        return rs_series_from_list(p, c, x, prec)

    else:
        # 抛出未实现的错误
        raise NotImplementedError

# _tan1函数：rs_tan函数的辅助函数，使用牛顿法计算单变量级数的tan函数的级数展开
def _tan1(p, x, prec):
    # 获取多项式p所在的环R
    R = p.ring
    # 初始化p1为R的零元素
    p1 = R(0)
    # 对于精度prec的巨大步长序列中的每个precx
    for precx in _giant_steps(prec):
        # tmp为p - rs_atan(p1, x, precx)的级数展开
        tmp = p - rs_atan(p1, x, precx)
        # tmp为tmp与1 + rs_square(p1, x, precx)的乘积的级数展开
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        # p1为p1与tmp的级数展开的和
        p1 += tmp
    # 返回p1的级数展开
    return p1

# rs_tan函数：对于给定的多项式p，关于变量x进行tan函数的级数展开
def rs_tan(p, x, prec):
    # 如果p是Puisieux级数，则调用Puisieux级数版本的rs_tan函数
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    # 获取多项式p所在的环R
    R = p.ring
    # 初始化常数为0
    const = 0
    # 获取p中关于x的常数项
    c = _get_constant_term(p, x)
    # 如果常量 c 存在
    if c:
        # 如果 R 的域是 EX（假设是一个特定的域）
        if R.domain is EX:
            # 将常量 c 转换为表达式形式
            c_expr = c.as_expr()
            # 计算 tan(c_expr) 的值作为常量
            const = tan(c_expr)
        # 如果 c 是多项式元素的实例
        elif isinstance(c, PolyElement):
            try:
                # 将常量 c 转换为表达式形式
                c_expr = c.as_expr()
                # 将 tan(c_expr) 转换为 R 环中的元素
                const = R(tan(c_expr))
            except ValueError:
                # 如果转换失败，则向 R 中添加 tan(c_expr) 作为新的生成元素
                R = R.add_gens([tan(c_expr, )])
                # 更新相关变量的环
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                # 重新计算常量 const
                const = R(tan(c_expr))
        else:
            try:
                # 将常量 c 转换为 R 环中的元素
                const = R(tan(c))
            except ValueError:
                # 如果转换失败，抛出域错误
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        # 计算多项式 p 与常量 c 的差
        p1 = p - c

    # 使用 SymPy 函数计算常量项的 tan 函数值，并进行进一步的数值计算
        # 使用 rs_tan 函数计算 p1 在 x 处的 tan 值
        t2 = rs_tan(p1, x, prec)
        # 使用 rs_series_inversion 函数计算级数的反函数
        t = rs_series_inversion(1 - const*t2, x, prec)
        # 返回乘积结果 const + t2 与 t 的乘积
        return rs_mul(const + t2, t, x, prec)

    # 如果 R 的生成元素个数为 1，则调用 _tan1 函数
    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        # 否则调用 rs_fun 函数，使用 rs_tan 函数进行计算
        return rs_fun(p, rs_tan, x, prec)
# 计算级数的余切

Return the series expansion of the cotangent of ``p``, about 0.

Examples
========

>>> from sympy.polys.domains import QQ
>>> from sympy.polys.rings import ring
>>> from sympy.polys.ring_series import rs_cot
>>> R, x, y = ring('x, y', QQ)
>>> rs_cot(x, x, 6)
-2/945*x**5 - 1/45*x**3 - 1/3*x + x**(-1)

See Also
========

cot
"""
def rs_cot(p, x, prec):
    # 如果级数 ``p`` 含有符号系数的线性项，则无法处理类似 `p = x + x*y` 的级数。
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_cot, p, x, prec)
        return r
    # 检查级数变量 ``p`` 是否符合余切函数要求，返回变量和次数信息。
    i, m = _check_series_var(p, x, 'cot')
    # 扩展精度以保证计算精确性。
    prec1 = prec + 2*m
    # 计算级数的余弦和正弦。
    c, s = rs_cos_sin(p, x, prec1)
    # 将正弦级数乘以系数和指数以匹配余切级数。
    s = mul_xin(s, i, -m)
    # 对正弦级数进行级数反转。
    s = rs_series_inversion(s, x, prec1)
    # 计算余切级数。
    res = rs_mul(c, s, x, prec1)
    # 对结果级数进行指数和系数匹配。
    res = mul_xin(res, i, -m)
    # 截断结果级数以匹配所需的精度。
    res = rs_trunc(res, x, prec)
    return res

# 计算级数的正弦函数
def rs_sin(p, x, prec):
    """
    Sine of a series

    Return the series expansion of the sin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sin
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sin(x + x*y, x, 4)
    -1/6*x**3*y**3 - 1/2*x**3*y**2 - 1/2*x**3*y - 1/6*x**3 + x*y + x
    >>> rs_sin(x**QQ(3, 2) + x*y**QQ(7, 5), x, 4)
    -1/2*x**(7/2)*y**(14/5) - 1/6*x**3*y**(21/5) + x**(3/2) + x*y**(7/5)

    See Also
    ========

    sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sin, p, x, prec)
    R = x.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            t1, t2 = sin(c_expr), cos(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                t1, t2 = R(sin(c_expr)), R(cos(c_expr))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        else:
            try:
                t1, t2 = R(sin(c)), R(cos(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        p1 = p - c

        # 利用SymPy的cos和sin函数计算常数项的cos和sin值。
        return rs_sin(p1, x, prec)*t2 + rs_cos(p1, x, prec)*t1

    # 如果级数长度超过20且环只有一个生成元，通过tan计算级数。
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 2*t, x, prec)
    one = R(1)
    n = 1
    c = [0]
    # 遍历从2到prec+2（不包括prec+2），步长为2的整数序列
    for k in range(2, prec + 2, 2):
        # 将one除以n的结果追加到列表c中
        c.append(one/n)
        # 向列表c中追加0
        c.append(0)
        # 更新n的值，乘以-k*(k + 1)
        n *= -k*(k + 1)
    
    # 返回通过给定参数p, c, x, prec计算得到的级数结果
    return rs_series_from_list(p, c, x, prec)
# 计算给定多项式 p 的级数展开中的余弦值
def rs_cos(p, x, prec):
    """
    Cosine of a series

    Return the series expansion of the cos of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cos
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cos(x + x*y, x, 4)
    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1
    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)
    -1/2*x**(3/5)*y**2 - x**(3/5)*y - 1/2*x**(3/5) + x**(-7/5)

    See Also
    ========

    cos
    """
    # 如果多项式 p 是 Puiseux 级数，则调用 rs_puiseux 处理
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    
    # 获取多项式环 R 和常数项 c
    R = p.ring
    c = _get_constant_term(p, x)
    
    # 如果存在常数项
    if c:
        # 处理不同域的情况
        if R.domain is EX:
            c_expr = c.as_expr()
            _, _ = sin(c_expr), cos(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                _, _ = R(sin(c_expr)), R(cos(c_expr))
            except ValueError:
                # 如果在当前环中无法展开，则添加 sin 和 cos 生成器并重设环
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
        else:
            try:
                _, _ = R(sin(c)), R(cos(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        
        # 将常数项从多项式 p 中移除
        p1 = p - c

        # 计算剩余多项式的余弦和正弦级数展开
        p_cos = rs_cos(p1, x, prec)
        p_sin = rs_sin(p1, x, prec)
        
        # 合并环并应用余弦和正弦值的系数
        R = R.compose(p_cos.ring).compose(p_sin.ring)
        p_cos.set_ring(R)
        p_sin.set_ring(R)
        t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        
        # 返回余弦乘以余弦项、正弦乘以正弦项之差
        return p_cos*t2 - p_sin*t1
    
    # 如果多项式长度大于 20 并且变量个数为 1，则利用 tan 展开
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    
    # 否则，计算 cos 的级数展开
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k - 1)
    
    # 返回基于系数 c 的级数展开
    return rs_series_from_list(p, c, x, prec)


# 返回 rs_cos 和 rs_sin 的元组
def rs_cos_sin(p, x, prec):
    r"""
    Return the tuple ``(rs_cos(p, x, prec)`, `rs_sin(p, x, prec))``.

    Is faster than calling rs_cos and rs_sin separately
    """
    # 如果多项式 p 是 Puiseux 级数，则调用 rs_puiseux 处理
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    
    # 否则，计算 tan 的级数展开并返回 cos 和 sin 的乘积
    t = rs_tan(p/2, x, prec)
    t2 = rs_square(t, x, prec)
    p1 = rs_series_inversion(1 + t2, x, prec)
    return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2*t, x, prec))


# 使用公式展开计算 rs_atanh
def _atanh(p, x, prec):
    """
    Expansion using formula

    Faster for very small and univariate series
    """
    R = p.ring
    one = R(1)
    c = [one]
    p2 = rs_square(p, x, prec)
    for k in range(1, prec):
        c.append(one/(2*k + 1))
    
    # 返回基于系数 c 的级数展开
    s = rs_series_from_list(p2, c, x, prec)
    s = rs_mul(s, p, x, prec)
    return s


# 计算 rs_atanh 的级数展开
def rs_atanh(p, x, prec):
    """
    # 如果 p 是 Puiseux 序列，直接返回 Puiseux 展开后的结果
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atanh, p, x, prec)
    
    # 获取 p 的环
    R = p.ring
    # 初始化常数项为 0
    const = 0
    
    # 检查 p 是否有常数项
    if _has_constant_term(p, x):
        # 获取零单项式和其系数
        zm = R.zero_monom
        c = p[zm]
        
        # 如果环为 EX（表达式环），将常数项转换为表达式并计算其反双曲正切
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atanh(c_expr)
        # 如果常数是多项式元素，则尝试计算其反双曲正切
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atanh(c_expr))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        # 否则，直接计算其反双曲正切
        else:
            try:
                const = R(atanh(c))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
    
    # 计算 atanh(p) 的导数 dp
    dp = rs_diff(p, x)
    # 计算 1/(1-p**2) * dp，这个表达式的级数展开较容易计算
    p1 = - rs_square(p, x, prec) + 1
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    # 对上述结果积分，加上之前计算得到的常数项
    return rs_integrate(p1, x) + const
# Helper function to compute the hyperbolic tangent of a series using Newton's method
def _tanh(p, x, prec):
    """
    Helper function of :func:`rs_tanh`

    Return the series expansion of tanh of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atanh is
    easier than that of tanh.

    See Also
    ========

    _tanh
    """
    # Get the ring of the series
    R = p.ring
    # Initialize p1 to zero series
    p1 = R(0)
    # Iterate using giant steps algorithm for precision
    for precx in _giant_steps(prec):
        # Compute atanh of p1
        tmp = p - rs_atanh(p1, x, precx)
        # Multiply tmp by 1 - p1^2
        tmp = rs_mul(tmp, 1 - rs_square(p1, x, prec), x, precx)
        # Update p1
        p1 += tmp
    return p1

def rs_tanh(p, x, prec):
    """
    Hyperbolic tangent of a series

    Return the series expansion of the tanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tanh(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    tanh
    """
    # Check if the series is a Puiseux series
    if rs_is_puiseux(p, x):
        # If so, return the Puiseux series expansion of rs_tanh
        return rs_puiseux(rs_tanh, p, x, prec)
    # Get the ring of the series
    R = p.ring
    # Initialize constant to zero
    const = 0
    # 检查多项式 p 是否有常数项
    if _has_constant_term(p, x):
        # 获取零次单项式的系数
        zm = R.zero_monom
        c = p[zm]
        # 如果环 R 的域是 EX，则将系数转换为表达式并应用双曲正切函数
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tanh(c_expr)
        # 如果系数 c 是多项式元素，则尝试将其转换为表达式并应用双曲正切函数
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tanh(c_expr))
            except ValueError:
                # 如果转换失败，抛出域错误异常
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            # 直接将系数应用双曲正切函数
            try:
                const = R(tanh(c))
            except ValueError:
                # 如果转换失败，抛出域错误异常
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        # 从多项式 p 中减去常数项 c 得到新的多项式 p1
        p1 = p - c
        # 计算 rs_tanh 函数应用于 p1 的结果
        t1 = rs_tanh(p1, x, prec)
        # 计算级数倒数的级数展开，使用 const*t1 作为参数
        t = rs_series_inversion(1 + const*t1, x, prec)
        # 返回 const + t1 与 t 的乘积，即最终结果
        return rs_mul(const + t1, t, x, prec)

    # 如果环 R 中的生成元个数为 1，则调用 _tanh 函数处理 p
    if R.ngens == 1:
        return _tanh(p, x, prec)
    else:
        # 否则调用 rs_fun 函数，其中函数为 _tanh
        return rs_fun(p, _tanh, x, prec)
# 计算多项式 `p` 的截断牛顿和
def rs_newton(p, x, prec):
    # 获取多项式 `p` 的次数
    deg = p.degree()
    # 将 `p` 中的单项式反转
    p1 = _invert_monoms(p)
    # 对反转后的 `p1` 进行环级数反转，精度为 `prec`
    p2 = rs_series_inversion(p1, x, prec)
    # 计算 `p1` 对 `x` 的导数与 `p2` 的乘积，环级数乘法，精度为 `prec`
    p3 = rs_mul(p1.diff(x), p2, x, prec)
    # 计算结果，为多项式 `deg - p3*x`
    res = deg - p3*x
    return res

# 计算多项式 `p1` 的哈达玛尔指数展开
def rs_hadamard_exp(p1, inverse=False):
    # 获取多项式环
    R = p1.ring
    # 检查环是否为有理数域 QQ，否则抛出未实现异常
    if R.domain != QQ:
        raise NotImplementedError
    # 创建多项式 `p`，初始化为零
    p = R.zero
    # 根据 `inverse` 参数选择不同的操作
    if not inverse:
        # 对于每个单项式 `exp1, v1` 在 `p1` 中，设置 `p` 的值为 `v1/int(ifac(exp1[0]))`
        for exp1, v1 in p1.items():
            p[exp1] = v1/int(ifac(exp1[0]))
    else:
        # 对于每个单项式 `exp1, v1` 在 `p1` 中，设置 `p` 的值为 `v1*int(ifac(exp1[0]))`
        for exp1, v1 in p1.items():
            p[exp1] = v1*int(ifac(exp1[0]))
    return p

# 计算多项式 `p1` 与 `p2` 的复合和
def rs_compose_add(p1, p2):
    # 获取多项式环
    R = p1.ring
    # 获取变量 `x`
    x = R.gens[0]
    # 计算所需的精度
    prec = p1.degree()*p2.degree() + 1
    # 计算 `p1` 的牛顿和展开
    np1 = rs_newton(p1, x, prec)
    # 计算 `np1` 的哈达玛尔指数展开
    np1e = rs_hadamard_exp(np1)
    # 计算 `p2` 的牛顿和展开
    np2 = rs_newton(p2, x, prec)
    # 计算 `np2` 的哈达玛尔指数展开
    np2e = rs_hadamard_exp(np2)
    # 计算 `np1e` 与 `np2e` 的环级数乘法
    np3e = rs_mul(np1e, np2e, x, prec)
    # 计算 `np3e` 的哈达玛尔指数展开，反转标志为真
    np3 = rs_hadamard_exp(np3e, True)
    # 计算 `(np3[(0,)] - np3)/x`
    np3a = (np3[(0,)] - np3)/x
    # 对 `np3a` 进行积分
    q = rs_integrate(np3a, x)
    # 对 `q` 进行指数函数展开
    q = rs_exp(q, x, prec)
    # 将 `q` 中单项式反转
    q = _invert_monoms(q)
    # 返回 `q` 的原始部分
    q = q.primitive()[1]
    # 计算 `p1` 和 `p2` 的次数乘积与 `q` 的次数差
    dp = p1.degree()*p2.degree() - q.degree()
    # `dp` 是结果的根的重数；
    # 这些根在这个计算中被忽略，所以将它们放在这里。
    # 如果 `p1` 和 `p2` 是首一不可约多项式，
    # 结果的根会存在，当且仅当 `p1 = p2`；实际上在这种情况下 `p1` 和 `p2` 有共同的根，
    # 所以 `gcd(p1, p2) != 1`；由于 `p1` 和 `p2` 是不可约的，这意味着 `p1 = p2`
    if dp:
        q = q*x**dp
    return q
_convert_func = {
    'sin': 'rs_sin',  # 将'sin'函数映射到'rs_sin'函数
    'cos': 'rs_cos',  # 将'cos'函数映射到'rs_cos'函数
    'exp': 'rs_exp',  # 将'exp'函数映射到'rs_exp'函数
    'tan': 'rs_tan',  # 将'tan'函数映射到'rs_tan'函数
    'log': 'rs_log'   # 将'log'函数映射到'rs_log'函数
}

def rs_min_pow(expr, series_rs, a):
    """Find the minimum power of `a` in the series expansion of expr"""
    series = 0  # 初始化级数为0
    n = 2  # 初始步长为2
    while series == 0:
        series = _rs_series(expr, series_rs, a, n)  # 计算级数展开
        n *= 2  # 增加步长
    R = series.ring  # 获取级数的环
    a = R(a)  # 将参数a转换为环R中的元素
    i = R.gens.index(a)  # 获取a在环中的索引
    return min(series, key=lambda t: t[i])[i]  # 返回级数中最小幂次的索引


def _rs_series(expr, series_rs, a, prec):
    # TODO Use _parallel_dict_from_expr instead of sring as sring is
    # inefficient. For details, read the todo in sring.
    args = expr.args  # 获取表达式的参数列表
    R = series_rs.ring  # 获取级数环

    # expr does not contain any function to be expanded
    if not any(arg.has(Function) for arg in args) and not expr.is_Function:
        return series_rs  # 如果表达式没有需要展开的函数，则返回原始级数

    if not expr.has(a):
        return series_rs  # 如果表达式不包含变量a，则返回原始级数

    elif expr.is_Function:
        arg = args[0]  # 获取函数的参数
        if len(args) > 1:
            raise NotImplementedError  # 如果参数大于1个，则抛出未实现错误
        R1, series = sring(arg, domain=QQ, expand=False, series=True)  # 对参数进行级数展开
        series_inner = _rs_series(arg, series, a, prec)  # 递归地对函数参数进行级数展开

        # Why do we need to compose these three rings?
        #
        # We want to use a simple domain (like ``QQ`` or ``RR``) but they don't
        # support symbolic coefficients. We need a ring that for example lets
        # us have `sin(1)` and `cos(1)` as coefficients if we are expanding
        # `sin(x + 1)`. The ``EX`` domain allows all symbolic coefficients, but
        # that makes it very complex and hence slow.
        #
        # To solve this problem, we add only those symbolic elements as
        # generators to our ring, that we need. Here, series_inner might
        # involve terms like `sin(4)`, `exp(a)`, etc, which are not there in
        # R1 or R. Hence, we compose these three rings to create one that has
        # the generators of all three.
        R = R.compose(R1).compose(series_inner.ring)  # 将三个环组合成一个环
        series_inner = series_inner.set_ring(R)  # 将内部级数设置为新的环R
        series = eval(_convert_func[str(expr.func)])(series_inner,
            R(a), prec)  # 根据函数类型对内部级数进行计算
        return series

    elif expr.is_Mul:
        n = len(args)
        for arg in args:    # XXX Looks redundant
            if not arg.is_Number:
                R1, _ = sring(arg, expand=False, series=True)  # 对乘法项进行级数展开
                R = R.compose(R1)  # 将展开后的环组合到当前环中
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args],
            [a]*len(args)))  # 计算各乘法项的最小幂次
        sum_pows = sum(min_pows)  # 计算最小幂次的总和
        series = R(1)  # 初始化级数为1

        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec - sum_pows +
                min_pows[i])  # 递归地对每个乘法项进行级数展开
            R = R.compose(_series.ring)  # 将展开后的环组合到当前环中
            _series = _series.set_ring(R)  # 将内部级数设置为新的环R
            series = series.set_ring(R)  # 将当前级数设置为新的环R
            series *= _series  # 级数相乘
        series = rs_trunc(series, R(a), prec)  # 对级数进行截断
        return series
    # 如果表达式是加法表达式
    elif expr.is_Add:
        # 获取加法操作数的数量
        n = len(args)
        # 初始化一个零元素的环序列
        series = R(0)
        # 遍历每个操作数
        for i in range(n):
            # 计算当前操作数的环级数
            _series = _rs_series(args[i], R(args[i]), a, prec)
            # 将环序列的环组合到总环中
            R = R.compose(_series.ring)
            # 将当前操作数的环级数设置为总环中的环
            _series = _series.set_ring(R)
            # 设置总序列的环
            series = series.set_ring(R)
            # 累加当前操作数的环级数到总序列中
            series += _series
        # 返回累加后的总序列
        return series

    # 如果表达式是幂次方表达式
    elif expr.is_Pow:
        # 将表达式基数转换为环序列
        R1, _ = sring(expr.base, domain=QQ, expand=False, series=True)
        # 将基数的环组合到总环中
        R = R.compose(R1)
        # 计算基数的环级数
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        # 调用幂次方函数计算序列
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)

    # 最后检查表达式是否为常数，因为 `is_constant` 方法存在问题，需要在最后检查
    # 参见问题编号 #9786 以获取更多细节
    elif isinstance(expr, Expr) and expr.is_constant():
        # 将表达式转换为环序列并返回其序列部分
        return sring(expr, domain=QQ, expand=False, series=True)[1]

    # 如果以上条件均不满足，则抛出未实现错误
    else:
        raise NotImplementedError
def rs_series(expr, a, prec):
    """Return the series expansion of an expression about 0.

    Parameters
    ==========

    expr : :class:`Expr`
        The expression for which the series expansion is calculated.
    a : :class:`Symbol`
        The symbol with respect to which `expr` is expanded.
    prec : int
        The desired order of the series expansion.

    Currently supports multivariate Taylor series expansion. This is much
    faster than SymPy's series method as it uses sparse polynomial operations.

    It automatically creates the simplest ring required to represent the series
    expansion through repeated calls to sring.

    Examples
    ========

    >>> from sympy.polys.ring_series import rs_series
    >>> from sympy import sin, cos, exp, tan, symbols, QQ
    >>> a, b, c = symbols('a, b, c')
    >>> rs_series(sin(a) + exp(a), a, 5)
    1/24*a**4 + 1/2*a**2 + 2*a + 1
    >>> series = rs_series(tan(a + b)*cos(a + c), a, 2)
    >>> series.as_expr()
    -a*sin(c)*tan(b) + a*cos(c)*tan(b)**2 + a*cos(c) + cos(c)*tan(b)
    >>> series = rs_series(exp(a**QQ(1,3) + a**QQ(2, 5)), a, 1)
    >>> series.as_expr()
    a**(11/15) + a**(4/5)/2 + a**(2/5) + a**(2/3)/2 + a**(1/3) + 1

    """
    # Create a ring R and initial series using sring function
    R, series = sring(expr, domain=QQ, expand=False, series=True)
    # Add the symbol 'a' to the ring if it's not already there
    if a not in R.symbols:
        R = R.add_gens([a, ])
    # Set the ring for the series
    series = series.set_ring(R)
    # Compute the series expansion using _rs_series
    series = _rs_series(expr, series, a, prec)
    # Get the ring from the series
    R = series.ring
    gen = R(a)
    # Calculate the degree of the series expansion obtained
    prec_got = series.degree(gen) + 1

    # Check if the obtained precision is sufficient
    if prec_got >= prec:
        # If yes, truncate the series to the desired precision
        return rs_trunc(series, gen, prec)
    else:
        # If not, iteratively increase the requested number of terms
        # until the desired precision is achieved
        for more in range(1, 9):
            p1 = _rs_series(expr, series, a, prec=prec + more)
            gen = gen.set_ring(p1.ring)
            new_prec = p1.degree(gen) + 1
            if new_prec != prec_got:
                # Adjust the precision dynamically to approach the desired precision
                prec_do = ceiling(prec + (prec - prec_got)*more/(new_prec - prec_got))
                p1 = _rs_series(expr, series, a, prec=prec_do)
                while p1.degree(gen) + 1 < prec:
                    p1 = _rs_series(expr, series, a, prec=prec_do)
                    gen = gen.set_ring(p1.ring)
                    prec_do *= 2
                break
            else:
                break
        else:
            # Raise an error if the desired precision cannot be achieved
            raise ValueError('Could not calculate %s terms for %s' % (str(prec), expr))
        return rs_trunc(p1, gen, prec)
```