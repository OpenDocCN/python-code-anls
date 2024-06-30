# `D:\src\scipysrc\sympy\sympy\polys\sqfreetools.py`

```
"""Square-free decomposition algorithms and related tools. """

# 导入从 sympy.polys.densearith 模块中的函数
from sympy.polys.densearith import (
    dup_neg, dmp_neg,            # 导入多项式的负值函数
    dup_sub, dmp_sub,            # 导入多项式的减法函数
    dup_mul, dmp_mul,            # 导入多项式的乘法函数
    dup_quo, dmp_quo,            # 导入多项式的除法函数
    dup_mul_ground, dmp_mul_ground)  # 导入多项式与地面元素的乘法函数

# 导入从 sympy.polys.densebasic 模块中的函数
from sympy.polys.densebasic import (
    dup_strip,                  # 导入多项式的去除首部零系数函数
    dup_LC, dmp_ground_LC,      # 导入多项式的首项系数函数
    dmp_zero_p,                 # 导入判断多项式是否为零函数
    dmp_ground,                 # 导入创建多项式的地面元素函数
    dup_degree, dmp_degree,     # 导入计算多项式的次数函数
    dmp_degree_in, dmp_degree_list,  # 导入计算多项式的次数函数
    dmp_raise, dmp_inject,      # 导入多项式的提升和注入函数
    dup_convert)                # 导入多项式的类型转换函数

# 导入从 sympy.polys.densetools 模块中的函数
from sympy.polys.densetools import (
    dup_diff, dmp_diff, dmp_diff_in,  # 导入多项式的微分函数
    dup_shift, dmp_shift,        # 导入多项式的移位函数
    dup_monic, dmp_ground_monic,  # 导入多项式的首项归一化函数
    dup_primitive, dmp_ground_primitive)  # 导入多项式的原始多项式函数

# 导入从 sympy.polys.euclidtools 模块中的函数
from sympy.polys.euclidtools import (
    dup_inner_gcd, dmp_inner_gcd,  # 导入多项式的内部最大公因数函数
    dup_gcd, dmp_gcd,            # 导入多项式的最大公因数函数
    dmp_resultant, dmp_primitive)  # 导入多项式的结果函数

# 导入从 sympy.polys.galoistools 模块中的函数
from sympy.polys.galoistools import (
    gf_sqf_list, gf_sqf_part)    # 导入有限域中多项式的平方自由因子列表和部分函数

# 导入从 sympy.polys.polyerrors 模块中的异常类
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,  # 导入多元多项式错误类
    DomainError)                 # 导入域错误类


def _dup_check_degrees(f, result):
    """Sanity check the degrees of a computed factorization in K[x]."""
    deg = sum(k * dup_degree(fac) for (fac, k) in result)
    assert deg == dup_degree(f)


def _dmp_check_degrees(f, u, result):
    """Sanity check the degrees of a computed factorization in K[X]."""
    degs = [0] * (u + 1)
    for fac, k in result:
        degs_fac = dmp_degree_list(fac, u)
        degs = [d1 + k * d2 for d1, d2 in zip(degs, degs_fac)]
    assert tuple(degs) == dmp_degree_list(f, u)


def dup_sqf_p(f, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    """
    if not f:
        return True
    else:
        return not dup_degree(dup_gcd(f, dup_diff(f, 1, K), K))


def dmp_sqf_p(f, u, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_p(x**2 + 2*x*y + y**2)
    False
    >>> R.dmp_sqf_p(x**2 + y**2)
    True

    """
    if dmp_zero_p(f, u):
        return True

    for i in range(u+1):

        fp = dmp_diff_in(f, 1, i, u, K)

        if dmp_zero_p(fp, u):
            continue

        gcd = dmp_gcd(f, fp, u, K)

        if dmp_degree_in(gcd, i, u) != 0:
            return False

    return True


def dup_sqf_norm(f, K):
    r"""
    Find a shift of `f` in `K[x]` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x)=f(x-sa)`, `r(x)=\text{Norm}(g(x))` and
    `r` is a square-free polynomial over `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(\sqrt{3})`
    and rings `K[x]` and `k[x]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    # 检查基础域是否为代数域，否则抛出域错误异常
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    # 初始化变量 s 和 g
    s, g = 0, dmp_raise(K.mod.to_list(), 1, 0, K.dom)

    # 进入主循环，执行 Trager 算法的一部分
    while True:
        # 将多项式 f 提升到 K 中，并获得 h
        h, _ = dmp_inject(f, 0, K, front=True)
        # 计算 g 和 h 的结果式 resultant(g, h)
        r = dmp_resultant(g, h, 1, K.dom)

        # 如果 r 是平方自由的多项式，则结束循环
        if dup_sqf_p(r, K.dom):
            break
        else:
            # 否则，通过将 f 向左移动 -K.unit 单位，并更新 s
            f, s = dup_shift(f, -K.unit, K), s + 1

    # 返回 s（偏移量）、f（变换后的多项式）和 r（结果式）
    return s, f, r
def _dmp_sqf_norm_shifts(f, u, K):
    """Generate a sequence of candidate shifts for dmp_sqf_norm."""
    #
    # We want to find a minimal shift if possible because shifting high degree
    # variables can be expensive e.g. x**10 -> (x + 1)**10. We try a few easy
    # cases first before the final infinite loop that is guaranteed to give
    # only finitely many bad shifts (see Trager76 for proof of this in the
    # univariate case).
    #

    # First the trivial shift [0, 0, ...]
    n = u + 1
    s0 = [0] * n
    # Yield the trivial shift s0 and the input polynomial f
    yield s0, f

    # Shift in multiples of the generator of the extension field K
    a = K.unit

    # Variables of degree > 0 ordered by increasing degree
    d = dmp_degree_list(f, u)
    var_indices = [i for di, i in sorted(zip(d, range(u+1))) if di > 0]

    # Now try [1, 0, 0, ...], [0, 1, 0, ...]
    for i in var_indices:
        s1 = s0.copy()
        s1[i] = 1
        a1 = [-a*s1i for s1i in s1]
        # Shift the polynomial f by a1 and yield the shifted polynomial f1
        f1 = dmp_shift(f, a1, u, K)
        yield s1, f1

    # Now try [1, 1, 1, ...], [2, 2, 2, ...]
    j = 0
    while True:
        j += 1
        sj = [j] * n
        aj = [-a*j] * n
        # Shift the polynomial f by aj and yield the shifted polynomial fj
        fj = dmp_shift(f, aj, u, K)
        yield sj, fj


def dmp_sqf_norm(f, u, K):
    r"""
    Find a shift of ``f`` in ``K[X]`` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x_1,x_2,\cdots)=f(x_1-s_1 a, x_2 - s_2 a,
    \cdots)`, `r(x)=\text{Norm}(g(x))` and `r` is a square-free polynomial over
    `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(i)` and rings
    `K[x,y]` and `k[x,y]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x, y = ring("x,y", K)
    >>> _, X, Y = ring("x,y", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x*y + y**2
    >>> s, g, r = R.dmp_sqf_norm(f)

    The choice of shifts ``s`` is arbitrary and the particular values returned
    for ``g`` and ``r`` are determined by ``s``.

    >>> s
    [0, 1]
    >>> g == x*y - I*x + y**2 - 2*I*y - 1
    True
    >>> r == X**2*Y**2 + X**2 + 2*X*Y**3 + 2*X*Y + Y**4 + 2*Y**2 + 1
    True

    The required invariants are:

    >>> g == f.shift_list([-si*K.unit for si in s])
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is a multivariate
    generalization of algorithm ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dup_sqf_norm:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dmp_ext_factor:
        Function implementing Trager's algorithm that uses this.

    """
    # 如果输入参数 u 为空（假值），则调用 dup_sqf_norm 函数计算 f 的平方根形式规范，并返回结果列表 [s], g, r
    if not u:
        s, g, r = dup_sqf_norm(f, K)
        return [s], g, r

    # 如果输入参数 K 不是代数域，则抛出 DomainError 异常，提示“地面域必须是代数的”
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    # 将 K.mod 转换为列表，并升维到 u+1 维，使用 K.dom 作为元素类型，创建 g
    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)

    # 迭代 _dmp_sqf_norm_shifts(f, u, K) 生成器产生的每个 (s, f) 对
    for s, f in _dmp_sqf_norm_shifts(f, u, K):

        # 将 f 提升到 u 维，并在前面注入 K，得到 h
        h, _ = dmp_inject(f, u, K, front=True)
        
        # 计算 g 和 h 的 u+1 维度下的结果式，使用 K.dom 作为系数环
        r = dmp_resultant(g, h, u + 1, K.dom)

        # 如果 r 是 u 维下的平方根形式多项式，则跳出循环
        if dmp_sqf_p(r, u, K.dom):
            break

    # 返回最终结果 s, f, r
    return s, f, r
# 计算多项式 ``f`` 在环 ``K[X]`` 中的范数，通常不是平方自由的。

# 检查域 ``K`` 是否是代数数域 ``k(a)`` 的实例（参见 :ref:`QQ(a)`）。
# 如果不是代数数域，则抛出域错误异常。
if not K.is_Algebraic:
    raise DomainError("ground domain must be algebraic")

# 提升 ``K.mod`` 到一个多项式，其系数在 ``K.dom`` 中，其次数为 ``u + 1``。
g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)

# 将多项式 ``f`` 注入到域 ``K`` 中，并返回注入后的结果和可能的余数。
h, _ = dmp_inject(f, u, K, front=True)

# 计算多项式 ``g`` 和 ``h`` 的结果式子。
# 这里的结果式子是 ``g`` 和 ``h`` 的结果式子，其次数为 ``u + 1``，系数在 ``K.dom`` 中。
return dmp_resultant(g, h, u + 1, K.dom)
    # 抛出未实现错误，提示多变量有限域上的多项式功能尚未完成
    raise NotImplementedError('multivariate polynomials over finite fields')
# 返回 ``K[x]`` 中多项式的平方自由部分
def dup_sqf_part(f, K):
    if K.is_FiniteField:  # 如果 K 是有限域
        return dup_gf_sqf_part(f, K)  # 调用有限域中多项式的平方自由部分函数

    if not f:  # 如果 f 是空的多项式（零多项式）
        return f  # 直接返回 f

    if K.is_negative(dup_LC(f, K)):  # 如果 f 的领头系数是负数
        f = dup_neg(f, K)  # 取 f 的相反数

    gcd = dup_gcd(f, dup_diff(f, 1, K), K)  # 计算 f 和其导数的最大公因式
    sqf = dup_quo(f, gcd, K)  # 将 f 除以 gcd 得到平方自由部分

    if K.is_Field:  # 如果 K 是域
        return dup_monic(sqf, K)  # 返回 sqf 的首项系数为1的多项式
    else:  # 如果 K 不是域
        return dup_primitive(sqf, K)[1]  # 返回 sqf 的原始形式的第二个返回值


# 返回 ``K[X]`` 中多项式的平方自由部分
def dmp_sqf_part(f, u, K):
    if not u:  # 如果 u 是零
        return dup_sqf_part(f, K)  # 返回 ``K[x]`` 中多项式的平方自由部分

    if K.is_FiniteField:  # 如果 K 是有限域
        return dmp_gf_sqf_part(f, u, K)  # 调用有限域中多项式的平方自由部分函数

    if dmp_zero_p(f, u):  # 如果 ``K[X]`` 中的多项式是零
        return f  # 直接返回 f

    if K.is_negative(dmp_ground_LC(f, u, K)):  # 如果 ``K[X]`` 中的多项式的领头系数是负数
        f = dmp_neg(f, u, K)  # 取 ``K[X]`` 中多项式的相反数

    gcd = f
    for i in range(u+1):  # 对于每一个 i 从 0 到 u
        gcd = dmp_gcd(gcd, dmp_diff_in(f, 1, i, u, K), u, K)  # 计算 ``K[X]`` 中多项式和其 i 阶偏导数的最大公因式
    sqf = dmp_quo(f, gcd, u, K)  # 将 ``K[X]`` 中的多项式除以 gcd 得到平方自由部分

    if K.is_Field:  # 如果 K 是域
        return dmp_ground_monic(sqf, u, K)  # 返回 ``K[X]`` 中多项式的首项系数为1的多项式
    else:  # 如果 K 不是域
        return dmp_ground_primitive(sqf, u, K)[1]  # 返回 ``K[X]`` 中多项式的原始形式的第二个返回值


# 在 ``GF(p)[x]`` 中计算 ``f`` 的平方自由分解
def dup_gf_sqf_list(f, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
    f_orig = f  # 保存原始多项式

    f = dup_convert(f, K, K.dom)  # 将多项式转换为 ``GF(p)[x]`` 中的形式

    coeff, factors = gf_sqf_list(f, K.mod, K.dom, all=all)  # 计算 ``f`` 的平方自由分解

    for i, (f, k) in enumerate(factors):  # 对于每一个分解得到的因子
        factors[i] = (dup_convert(f, K.dom, K), k)  # 将因子转换为 ``K[x]`` 中的形式

    _dup_check_degrees(f_orig, factors)  # 检查分解后的因子的次数

    return K.convert(coeff, K.dom), factors  # 返回系数和因子的列表


# 在 ``GF(p)[X]`` 中计算 ``f`` 的平方自由分解
def dmp_gf_sqf_list(f, u, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


# 返回 ``K[x]`` 中多项式的平方自由分解
def dup_sqf_list(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Uses Yun's algorithm from [Yun76]_.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    See Also
    ========

    dmp_sqf_list:
        Corresponding function for multivariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.

    References
    ==========

    [Yun76]_
    """
    if K.is_FiniteField:  # 如果 K 是有限域
        return dup_gf_sqf_list(f, K, all=all)  # 调用 ``GF(p)[x]`` 中多项式的平方自由分解函数

    f_orig = f  # 保存原始多项式

    if K.is_Field:  # 如果 K 是域
        coeff = dup_LC(f, K)  # 计算多项式的领头系数
        f = dup_monic(f, K)  # 将多项式转换为首项系数为1的形式
    else:
        coeff, f = dup_primitive(f, K)
        # 如果多项式 f 不是原始的，将其转换为原始形式，并返回系数 coeff 和新的多项式 f

        if K.is_negative(dup_LC(f, K)):
            f = dup_neg(f, K)
            coeff = -coeff
            # 如果 f 的首项系数为负数，则取负数的 f，同时系数 coeff 取相反数

    if dup_degree(f) <= 0:
        return coeff, []
        # 如果多项式 f 的次数小于等于 0，直接返回 coeff 和空列表作为结果

    result, i = [], 1
    # 初始化结果列表 result 和计数器 i

    h = dup_diff(f, 1, K)
    # 计算多项式 f 的一阶导数 h

    g, p, q = dup_inner_gcd(f, h, K)
    # 计算 f 和 h 的内部最大公因式 g，以及对应的 p 和 q

    while True:
        d = dup_diff(p, 1, K)
        # 计算 p 的一阶导数 d
        h = dup_sub(q, d, K)
        # 计算 q 与 d 的差 h

        if not h:
            result.append((p, i))
            break
            # 如果 h 为零，则将 (p, i) 添加到结果列表 result 中，并结束循环

        g, p, q = dup_inner_gcd(p, h, K)
        # 更新 g，p，q，继续计算 p 和 h 的内部最大公因式

        if all or dup_degree(g) > 0:
            result.append((g, i))
            # 如果 all 为真或者 g 的次数大于 0，则将 (g, i) 添加到结果列表 result 中

        i += 1
        # 计数器 i 自增

    _dup_check_degrees(f_orig, result)
    # 检查原始多项式 f_orig 和结果列表 result 的次数情况

    return coeff, result
    # 返回系数 coeff 和最终的结果列表 result
def dmp_sqf_list(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    Explanation
    ===========

    Uses Yun's algorithm for univariate polynomials from [Yun76]_ recursively.
    The multivariate polynomial is treated as a univariate polynomial in its
    leading variable. Then Yun's algorithm computes the square-free
    factorization of the primitive and the content is factored recursively.

    It would be better to use a dedicated algorithm for multivariate
    polynomials instead.

    See Also
    ========

    dup_sqf_list:
        Corresponding function for univariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.
    """
    # If the degrees of freedom is 0, return the coeff and an empty list.
    if not u:
        return dup_sqf_list(f, K, all=all)

    # If the coefficient field is finite, use a specific factorization function.
    if K.is_FiniteField:
        return dmp_gf_sqf_list(f, u, K, all=all)

    # Save the original polynomial.
    f_orig = f

    # If the field of the coefficients is a field, normalize the leading coefficient of the polynomial and, if necessary, the monic polynomial.
    if K.is_Field:
        coeff = dmp_ground_LC(f, u, K)
        f = dmp_ground_monic(f, u, K)
    else:
        coeff, f = dmp_ground_primitive(f, u, K)

        # If the leading coefficient of the polynomial is negative, change the polynomial's sign and adjust the monic polynomial.
        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)
            coeff = -coeff

    # Define the polynomial's degree and return the coefficients and an empty list if the degree is less than zero.
    deg = dmp_degree(f, u)
    if deg < 0:
        return coeff, []

    # The Yun algorithm needs the polynomial to be a monovariate polynomial in its main variable.
    content, f = dmp_primitive(f, u, K)

    # Set an empty list for the result.
    result = {}

    # If the degree is different from 0, the h variable is the first differentiation, and the inner variable is the GCD
 يجد
    # 将系数乘以系数内容
    coeff *= coeff_content

    # 将内容部分和原始部分具有相同重数的因子组合起来，以产生一个按重数升序排列的列表。
    for fac, i in result_content:
        # 将因子 fac 包装成列表形式
        fac = [fac]
        # 如果在 result 中存在重数 i 的条目，则使用 dmp_mul 函数将 fac 与 result[i] 相乘并更新结果
        if i in result:
            result[i] = dmp_mul(result[i], fac, u, K)
        # 否则，将 fac 直接作为结果的新条目
        else:
            result[i] = fac

    # 将 result 转换为 (result[i], i) 形式的元组列表，并按照 i 的值进行排序
    result = [(result[i], i) for i in sorted(result)]

    # 检查原始多项式 f_orig 和结果 result 中的多项式的次数，并作出必要的调整
    _dmp_check_degrees(f_orig, u, result)

    # 返回计算得到的系数和结果
    return coeff, result
# 返回在 K[x] 中多项式的平方自由分解列表，其中 K 是系数环
def dmp_sqf_list_include(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list_include(f)
    [(1, 1), (x + y, 2), (x, 3)]
    >>> R.dmp_sqf_list_include(f, all=True)
    [(1, 1), (x + y, 2), (x, 3)]

    """
    # 如果 u 为空（单变量情况），调用 dup_sqf_list_include 函数处理
    if not u:
        return dup_sqf_list_include(f, K, all=all)

    # 计算多项式 f 在多重指数 u 下的平方自由分解
    coeff, factors = dmp_sqf_list(f, u, K, all=all)

    # 如果存在因子并且第一个因子的次数为 1，则计算 g 并返回结果
    if factors and factors[0][1] == 1:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, 1)] + factors[1:]
    else:
        # 否则，计算 g 并返回结果
        g = dmp_ground(coeff, u)
        return [(g, 1)] + factors


# 计算在 K[x] 中多项式的最大阶乘因子分解
def dup_gff_list(f, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gff_list(x**5 + 2*x**4 - x**3 - 2*x**2)
    [(x, 1), (x + 2, 4)]

    """
    # 如果 f 是零多项式，抛出 ValueError
    if not f:
        raise ValueError("greatest factorial factorization doesn't exist for a zero polynomial")

    # 将 f 转化为首一多项式
    f = dup_monic(f, K)

    # 如果 f 的次数为 0，则返回空列表
    if not dup_degree(f):
        return []
    else:
        # 计算 f 和 f 在 K.one 处偏移后的最大公因式 g
        g = dup_gcd(f, dup_shift(f, K.one, K), K)
        # 递归计算 g 的最大阶乘因子分解 H
        H = dup_gff_list(g, K)

        # 更新 H 中每个因子的次数并计算新的 g
        for i, (h, k) in enumerate(H):
            g = dup_mul(g, dup_shift(h, -K(k), K), K)
            H[i] = (h, k + 1)

        # 计算 f 除以 g 的商
        f = dup_quo(f, g, K)

        # 如果 f 的次数为 0，则返回 H
        if not dup_degree(f):
            return H
        else:
            # 否则返回 (f, 1) 和 H 的合并结果
            return [(f, 1)] + H


# 抛出多变量多项式错误
def dmp_gff_list(f, u, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    # 如果 u 为空（单变量情况），调用 dup_gff_list 函数处理
    if not u:
        return dup_gff_list(f, K)
    else:
        # 否则抛出多变量多项式错误
        raise MultivariatePolynomialError(f)
```