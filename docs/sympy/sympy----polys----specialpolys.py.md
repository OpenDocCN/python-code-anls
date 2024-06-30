# `D:\src\scipysrc\sympy\sympy\polys\specialpolys.py`

```
"""
Functions for generating interesting polynomials, e.g. for benchmarking.
"""

# 从 sympy 库导入需要的模块和函数
from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
    dmp_add_term, dmp_neg, dmp_mul, dmp_sqr
)
from sympy.polys.densebasic import (
    dmp_zero, dmp_one, dmp_ground,
    dup_from_raw_dict, dmp_raise, dup_random
)
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent

# 定义函数 swinnerton_dyer_poly
@public
def swinnerton_dyer_poly(n, x=None, polys=False):
    """
    Generates n-th Swinnerton-Dyer polynomial in `x`.

    Parameters
    ----------
    n : int
        `n` decides the order of polynomial
    x : optional
    polys : bool, optional
        ``polys=True`` returns a Poly object, otherwise
        (default) returns an expression.
    """
    # 如果 n 小于等于 0，则抛出 ValueError 异常
    if n <= 0:
        raise ValueError(
            "Cannot generate Swinnerton-Dyer polynomial of order %s" % n)

    # 如果给定了 x，则将其转换为符号对象；否则创建一个新的符号对象 x
    if x is not None:
        sympify(x)
    else:
        x = Dummy('x')

    # 如果 n 大于 3，则生成一个特定类型的多项式
    if n > 3:
        from sympy.functions.elementary.miscellaneous import sqrt
        from .numberfields import minimal_polynomial
        p = 2
        a = [sqrt(2)]
        for i in range(2, n + 1):
            p = nextprime(p)
            a.append(sqrt(p))
        return minimal_polynomial(Add(*a), x, polys=polys)

    # 根据不同的 n 值生成预定义的多项式表达式
    if n == 1:
        ex = x**2 - 2
    elif n == 2:
        ex = x**4 - 10*x**2 + 1
    elif n == 3:
        ex = x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576

    # 如果 polys=True，则返回 Poly 对象，否则返回多项式表达式
    return PurePoly(ex, x) if polys else ex


# 定义函数 cyclotomic_poly
@public
def cyclotomic_poly(n, x=None, polys=False):
    """
    Generates cyclotomic polynomial of order `n` in `x`.

    Parameters
    ----------
    n : int
        `n` decides the order of polynomial
    x : optional
    polys : bool, optional
        ``polys=True`` returns a Poly object, otherwise
        (default) returns an expression.
    """
    # 如果 n 小于等于 0，则抛出 ValueError 异常
    if n <= 0:
        raise ValueError(
            "Cannot generate cyclotomic polynomial of order %s" % n)

    # 使用 dup_zz_cyclotomic_poly 函数生成指定阶数 n 的环周期多项式
    poly = DMP(dup_zz_cyclotomic_poly(int(n), ZZ), ZZ)

    # 如果给定了 x，则将多项式 poly 转换为关于 x 的 Poly 对象；否则创建一个新的关于 x 的 PurePoly 对象
    if x is not None:
        poly = Poly.new(poly, x)
    else:
        poly = PurePoly.new(poly, Dummy('x'))

    # 如果 polys=True，则返回 Poly 对象，否则返回多项式表达式
    return poly if polys else poly.as_expr()


# 定义函数 symmetric_poly
@public
def symmetric_poly(n, *gens, polys=False):
    """
    Generates symmetric polynomial of order `n`.

    Parameters
    ==========

    polys: bool, optional (default: False)
        Returns a Poly object when ``polys=True``, otherwise
        (default) returns an expression.
    """
    # 分析生成器 gens
    gens = _analyze_gens(gens)

    # 如果 n 小于 0 或大于生成器数目或没有给定生成器，则抛出 ValueError 异常
    if n < 0 or n > len(gens) or not gens:
        raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, gens))
    # 如果 n 等于 0，则返回单位元 S.One
    elif not n:
        poly = S.One
    else:
        # 如果不是上述特殊情况，则执行以下代码块
        # 使用生成器表达式生成所有可能的子集乘积，并将它们加和
        poly = Add(*[Mul(*s) for s in subsets(gens, int(n))])

    # 如果 polys 为真（非空），返回一个多项式对象，否则返回 poly
    return Poly(poly, *gens) if polys else poly
# 定义一个公共函数 `random_poly`，生成指定次数、指定范围内系数的多项式
@public
def random_poly(x, n, inf, sup, domain=ZZ, polys=False):
    """Generates a polynomial of degree ``n`` with coefficients in
    ``[inf, sup]``.

    Parameters
    ----------
    x
        `x` is the independent term of polynomial
    n : int
        `n` decides the order of polynomial
    inf
        Lower limit of range in which coefficients lie
    sup
        Upper limit of range in which coefficients lie
    domain : optional
         Decides what ring the coefficients are supposed
         to belong. Default is set to Integers.
    polys : bool, optional
        ``polys=True`` returns an expression, otherwise
        (default) returns an expression.
    """
    # 调用 `dup_random` 生成指定范围、指定次数的随机系数列表，创建多项式对象 `Poly`
    poly = Poly(dup_random(n, inf, sup, domain), x, domain=domain)

    # 如果 `polys` 为真返回多项式对象，否则返回多项式表达式
    return poly if polys else poly.as_expr()


# 定义一个公共函数 `interpolating_poly`，构造拉格朗日插值多项式
@public
def interpolating_poly(n, x, X='x', Y='y'):
    """Construct Lagrange interpolating polynomial for ``n``
    data points. If a sequence of values are given for ``X`` and ``Y``
    then the first ``n`` values will be used.
    """
    # 检查 `x` 是否具有 `free_symbols` 属性
    ok = getattr(x, 'free_symbols', None)

    # 如果 `X` 是字符串，则创建对应的符号列表，否则检查是否包含重复符号
    if isinstance(X, str):
        X = symbols("%s:%s" % (X, n))
    elif ok and ok & Tuple(*X).free_symbols:
        ok = False

    # 如果 `Y` 是字符串，则创建对应的符号列表，否则检查是否包含重复符号
    if isinstance(Y, str):
        Y = symbols("%s:%s" % (Y, n))
    elif ok and ok & Tuple(*Y).free_symbols:
        ok = False

    # 如果有重复符号或者 `ok` 不为真，则抛出值错误
    if not ok:
        raise ValueError(filldedent('''
            Expecting symbol for x that does not appear in X or Y.
            Use `interpolate(list(zip(X, Y)), x)` instead.'''))

    coeffs = []
    numert = Mul(*[x - X[i] for i in range(n)])

    # 计算拉格朗日插值多项式的系数
    for i in range(n):
        numer = numert/(x - X[i])
        denom = Mul(*[(X[i] - X[j]) for j in range(n) if i != j])
        coeffs.append(numer/denom)

    # 返回拉格朗日插值多项式表达式
    return Add(*[coeff*y for coeff, y in zip(coeffs, Y)])


# 定义一个函数 `fateman_poly_F_1`，实现 Fateman 的 GCD 基准测试：平凡的最大公约数
def fateman_poly_F_1(n):
    """Fateman's GCD benchmark: trivial GCD """
    # 创建符号列表 `Y`
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    # 定义符号 `y_0` 和 `y_1`
    y_0, y_1 = Y[0], Y[1]

    # 计算 `u` 和 `v`
    u = y_0 + Add(*Y[1:])
    v = y_0**2 + Add(*[y**2 for y in Y[1:]])

    # 创建多项式对象 `F` 和 `G`
    F = ((u + 1)*(u + 2)).as_poly(*Y)
    G = ((v + 1)*(-3*y_1*y_0**2 + y_1**2 - 1)).as_poly(*Y)

    # 创建多项式对象 `H`
    H = Poly(1, *Y)

    # 返回多项式 `F`, `G`, `H`
    return F, G, H


# 定义一个函数 `dmp_fateman_poly_F_1`，实现 Fateman 的 GCD 基准测试：平凡的最大公约数
def dmp_fateman_poly_F_1(n, K):
    """Fateman's GCD benchmark: trivial GCD """
    # 初始化 `u` 列表
    u = [K(1), K(0)]

    # 循环生成 `u` 列表
    for i in range(n):
        u = [dmp_one(i, K), u]

    # 初始化 `v` 列表
    v = [K(1), K(0), K(0)]

    # 循环生成 `v` 列表
    for i in range(0, n):
        v = [dmp_one(i, K), dmp_zero(i), v]

    # 计算 `m`
    m = n - 1

    # 计算 `U`, `V`
    U = dmp_add_term(u, dmp_ground(K(1), m), 0, n, K)
    V = dmp_add_term(u, dmp_ground(K(2), m), 0, n, K)

    # 初始化 `f`
    f = [[-K(3), K(0)], [], [K(1), K(0), -K(1)]]

    # 计算 `W`, `Y`
    W = dmp_add_term(v, dmp_ground(K(1), m), 0, n, K)
    Y = dmp_raise(f, m, 1, K)

    # 计算 `F`, `G`
    F = dmp_mul(U, V, n, K)
    G = dmp_mul(W, Y, n, K)

    # 创建多项式对象 `H`
    H = dmp_one(n, K)

    # 返回多项式 `F`, `G`, `H`
    return F, G, H


# 定义一个函数 `fateman_poly_F_2`，实现 Fateman 的 GCD 基准测试：线性密集的四次输入
def fateman_poly_F_2(n):
    """Fateman's GCD benchmark: linearly dense quartic inputs """
    # 创建符号列表 `Y`
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    # 定义符号 `y_0`
    y_0 = Y[0]

    # 计算 `u`
    u = Add(*Y[1:])
    # 创建多项式 H，其表达式为 (y_0 + u + 1)**2，使用 Y 中的符号作为变量
    H = Poly((y_0 + u + 1)**2, *Y)

    # 创建多项式 F，其表达式为 (y_0 - u - 2)**2，使用 Y 中的符号作为变量
    F = Poly((y_0 - u - 2)**2, *Y)
    # 创建多项式 G，其表达式为 (y_0 + u + 2)**2，使用 Y 中的符号作为变量
    G = Poly((y_0 + u + 2)**2, *Y)

    # 返回三个多项式的结果：H*F, H*G, H
    return H*F, H*G, H
# 定义函数 dmp_fateman_poly_F_2，实现Fateman的GCD基准测试：线性密集的四次输入
def dmp_fateman_poly_F_2(n, K):
    """Fateman's GCD benchmark: linearly dense quartic inputs """
    # 初始化列表 u，包含 K(1) 和 K(0)
    u = [K(1), K(0)]

    # 循环 n - 1 次，生成列表 u
    for i in range(n - 1):
        # 将 dmp_one(i, K) 添加到列表 u 的开头
        u = [dmp_one(i, K), u]

    # m 等于 n - 1
    m = n - 1

    # 计算 v，调用 dmp_add_term 函数
    v = dmp_add_term(u, dmp_ground(K(2), m - 1), 0, n, K)

    # 计算 f，调用 dmp_sqr 函数
    f = dmp_sqr([dmp_one(m, K), dmp_neg(v, m, K)], n, K)
    # 计算 g，调用 dmp_sqr 函数
    g = dmp_sqr([dmp_one(m, K), v], n, K)

    # 更新 v，调用 dmp_add_term 函数
    v = dmp_add_term(u, dmp_one(m - 1, K), 0, n, K)

    # 计算 h，调用 dmp_sqr 函数
    h = dmp_sqr([dmp_one(m, K), v], n, K)

    # 返回结果，调用 dmp_mul 函数
    return dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h


# 定义函数 fateman_poly_F_3，实现Fateman的GCD基准测试：稀疏输入（f 的次数 ~ 变量数）
def fateman_poly_F_3(n):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    # 创建符号列表 Y，包含 n + 1 个符号
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    # 设置 y_0 为 Y[0]
    y_0 = Y[0]

    # 计算 u，使用 Add 函数生成
    u = Add(*[y**(n + 1) for y in Y[1:]])

    # 构建多项式 H
    H = Poly((y_0**(n + 1) + u + 1)**2, *Y)

    # 构建多项式 F
    F = Poly((y_0**(n + 1) - u - 2)**2, *Y)
    # 构建多项式 G
    G = Poly((y_0**(n + 1) + u + 2)**2, *Y)

    # 返回结果
    return H*F, H*G, H


# 定义函数 dmp_fateman_poly_F_3，实现Fateman的GCD基准测试：稀疏输入（f 的次数 ~ 变量数）
def dmp_fateman_poly_F_3(n, K):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    # 使用 dup_from_raw_dict 函数创建 u
    u = dup_from_raw_dict({n + 1: K.one}, K)

    # 循环 n - 1 次，更新 u
    for i in range(0, n - 1):
        # 调用 dmp_add_term 函数，更新 u
        u = dmp_add_term([u], dmp_one(i, K), n + 1, i + 1, K)

    # 计算 v，调用 dmp_add_term 函数
    v = dmp_add_term(u, dmp_ground(K(2), n - 2), 0, n, K)

    # 计算 f，调用 dmp_sqr 函数
    f = dmp_sqr(
        dmp_add_term([dmp_neg(v, n - 1, K)], dmp_one(n - 1, K), n + 1, n, K), n, K)
    # 计算 g，调用 dmp_sqr 函数
    g = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)

    # 更新 v，调用 dmp_add_term 函数
    v = dmp_add_term(u, dmp_one(n - 2, K), 0, n - 1, K)

    # 计算 h，调用 dmp_sqr 函数
    h = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)

    # 返回结果，调用 dmp_mul 函数
    return dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h
    # 返回一个多项式表达式的计算结果，结果为整数
    return -x**9*y**8*z - x**8*y**5*z**3 - x**7*y**12*z**2 - 5*x**7*y**8 - x**6*y**9*z**4 + \
           x**6*y**7*z**3 + 3*x**6*y**7*z - 5*x**6*y**5*z**2 - x**6*y**4*z**3 + \
           x**5*y**4*z**5 + 3*x**5*y**4*z**3 - x**5*y*z**5 + x**4*y**11*z**4 + \
           3*x**4*y**11*z**2 - x**4*y**8*z**4 + 5*x**4*y**7*z**2 + 15*x**4*y**7 - \
           5*x**4*y**4*z**2 + x**3*y**8*z**6 + 3*x**3*y**8*z**4 - x**3*y**5*z**6 + \
           5*x**3*y**4*z**4 + 15*x**3*y**4*z**2 + x**3*y**3*z**5 + 3*x**3*y**3*z**3 - \
           5*x**3*y*z**4 + x**2*z**7 + 3*x**2*z**5 + x*y**7*z**6 + 3*x*y**7*z**4 + \
           5*x*y**3*z**4 + 15*x*y**3*z**2 + y**4*z**8 + 3*y**4*z**6 + 5*z**6 + 15*z**4
# 定义一个多项式环 R，包含变量 x, y, z，使用整数环 ZZ
def _f_5():
    R, x, y, z = ring("x,y,z", ZZ)
    # 返回一个特定的多项式表达式
    return -x**3 - 3*x**2*y + 3*x**2*z - 3*x*y**2 + 6*x*y*z - 3*x*z**2 - y**3 + 3*y**2*z - 3*y*z**2 + z**3

# 定义一个多项式环 R，包含变量 x, y, z, t，使用整数环 ZZ
def _f_6():
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    # 返回一个特定的多项式表达式
    return 2115*x**4*y + 45*x**3*z**3*t**2 - 45*x**3*t**2 - 423*x*y**4 - 47*x*y**3 + 141*x*y*z**3 + 94*x*y*z*t - 9*y**3*z**3*t**2 + 9*y**3*t**2 - y**2*z**3*t**2 + y**2*t**2 + 3*z**6*t**2 + 2*z**4*t**3 - 3*z**3*t**2 - 2*z*t**3

# 定义一个多项式环 R，包含变量 x, y, z，使用整数环 ZZ
def _w_1():
    R, x, y, z = ring("x,y,z", ZZ)
    # 返回一个特定的多项式表达式
    return 4*x**6*y**4*z**2 + 4*x**6*y**3*z**3 - 4*x**6*y**2*z**4 - 4*x**6*y*z**5 + x**5*y**4*z**3 + 12*x**5*y**3*z - x**5*y**2*z**5 + 12*x**5*y**2*z**2 - 12*x**5*y*z**3 - 12*x**5*z**4 + 8*x**4*y**4 + 6*x**4*y**3*z**2 + 8*x**4*y**3*z - 4*x**4*y**2*z**4 + 4*x**4*y**2*z**3 - 8*x**4*y**2*z**2 - 4*x**4*y*z**5 - 2*x**4*y*z**4 - 8*x**4*y*z**3 + 2*x**3*y**4*z + x**3*y**3*z**3 - x**3*y**2*z**5 - 2*x**3*y**2*z**3 + 9*x**3*y**2*z - 12*x**3*y*z**3 + 12*x**3*y*z**2 - 12*x**3*z**4 + 3*x**3*z**3 + 6*x**2*y**3 - 6*x**2*y**2*z**2 + 8*x**2*y**2*z - 2*x**2*y*z**4 - 8*x**2*y*z**3 + 2*x**2*y*z**2 + 2*x*y**3*z - 2*x*y**2*z**3 - 3*x*y*z + 3*x*z**3 - 2*y**2 + 2*y*z**2

# 定义一个多项式环 R，包含变量 x, y，使用整数环 ZZ
def _w_2():
    R, x, y = ring("x,y", ZZ)
    # 返回一个特定的多项式表达式
    return 24*x**8*y**3 + 48*x**8*y**2 + 24*x**7*y**5 - 72*x**7*y**2 + 25*x**6*y**4 + 2*x**6*y**3 + 4*x**6*y + 8*x**6 + x**5*y**6 + x**5*y**3 - 12*x**5 + x**4*y**5 - x**4*y**4 - 2*x**4*y**3 + 292*x**4*y**2 - x**3*y**6 + 3*x**3*y**3 - x**2*y**5 + 12*x**2*y**3 + 48*x**2 - 12*y**3

# 返回多项式 _f_0 到 _f_4 的元组
def f_polys():
    return _f_0(), _f_1(), _f_2(), _f_3(), _f_4(), _f_5(), _f_6()

# 返回多项式 _w_1 和 _w_2 的元组
def w_polys():
    return _w_1(), _w_2()
```