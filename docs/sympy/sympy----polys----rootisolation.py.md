# `D:\src\scipysrc\sympy\sympy\polys\rootisolation.py`

```
"""Real and complex root isolation and refinement algorithms. """

# 导入 SymPy 中的多项式操作函数
from sympy.polys.densearith import (
    dup_neg, dup_rshift, dup_rem,
    dup_l2_norm_squared)
from sympy.polys.densebasic import (
    dup_LC, dup_TC, dup_degree,
    dup_strip, dup_reverse,
    dup_convert,
    dup_terms_gcd)
from sympy.polys.densetools import (
    dup_clear_denoms,
    dup_mirror, dup_scale, dup_shift,
    dup_transform,
    dup_diff,
    dup_eval, dmp_eval_in,
    dup_sign_variations,
    dup_real_imag)
from sympy.polys.euclidtools import (
    dup_discriminant)
from sympy.polys.factortools import (
    dup_factor_list)
from sympy.polys.polyerrors import (
    RefinementFailed,
    DomainError,
    PolynomialError)
from sympy.polys.sqfreetools import (
    dup_sqf_part, dup_sqf_list)


def dup_sturm(f, K):
    """
    Computes the Sturm sequence of ``f`` in ``F[x]``.

    Given a univariate, square-free polynomial ``f(x)`` returns the
    associated Sturm sequence ``f_0(x), ..., f_n(x)`` defined by::

       f_0(x), f_1(x) = f(x), f'(x)
       f_n = -rem(f_{n-2}(x), f_{n-1}(x))

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_sturm(x**3 - 2*x**2 + x - 3)
    [x**3 - 2*x**2 + x - 3, 3*x**2 - 4*x + 1, 2/9*x + 25/9, -2079/4]

    References
    ==========

    .. [1] [Davenport88]_

    """
    # 检查域 K 是否为一个域，如果不是则抛出异常
    if not K.is_Field:
        raise DomainError("Cannot compute Sturm sequence over %s" % K)

    # 计算多项式 f 的平方因子部分
    f = dup_sqf_part(f, K)

    # 初始化 Sturm 序列，包含 f 和 f'，即 f_0 和 f_1
    sturm = [f, dup_diff(f, 1, K)]

    # 使用辗转相除法构建 Sturm 序列直到最后一个非零项
    while sturm[-1]:
        s = dup_rem(sturm[-2], sturm[-1], K)
        sturm.append(dup_neg(s, K))

    # 返回 Sturm 序列去除最后一个零项后的结果
    return sturm[:-1]

def dup_root_upper_bound(f, K):
    """Compute the LMQ upper bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

    References
    ==========
    .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
        Values of the Positive Roots of Polynomials"
        Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    # 获取多项式 f 的长度和空列表 P
    n, P = len(f), []
    # 初始化 t 为长度为 n 的单位元素列表
    t = n * [K.one]
    # 如果 f 的领头系数小于 0，则将 f 取反
    if dup_LC(f, K) < 0:
        f = dup_neg(f, K)
    # 将 f 列表反转
    f = list(reversed(f))

    # 遍历多项式的系数
    for i in range(0, n):
        # 如果系数为非正数，继续下一次循环
        if f[i] >= 0:
            continue

        # 计算系数的对数值 a
        a, QL = K.log(-f[i], 2), []

        # 遍历下一阶系数
        for j in range(i + 1, n):

            # 如果系数为非正数，继续下一次循环
            if f[j] <= 0:
                continue

            # 计算 Q 值，并加入 QL 列表
            q = t[j] + a - K.log(f[j], 2)
            QL.append([q // (j - i), j])

        # 如果 QL 为空，继续下一次循环
        if not QL:
            continue

        # 计算最小 Q 值
        q = min(QL)

        # 将 t[q[1]] 增加 1
        t[q[1]] = t[q[1]] + 1

        # 将 q[0] 添加到 P 列表
        P.append(q[0])

    # 如果 P 为空，返回 None；否则返回最大 P 值加 1 的 2 的幂值
    if not P:
        return None
    else:
        return K.get_field()(2)**(max(P) + 1)

def dup_root_lower_bound(f, K):
    # To be continued
    """
    Compute the LMQ lower bound for the positive roots of `f`;
    LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

    References
    ==========
    .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
           Values of the Positive Roots of Polynomials"
           Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    # 计算 `f` 的反向多项式，并计算其根的上界
    bound = dup_root_upper_bound(dup_reverse(f), K)

    # 如果计算得到了根的上界，则返回其倒数作为 LMQ 下界
    if bound is not None:
        return 1/bound
    # 如果未能计算出根的上界，则返回 None
    else:
        return None
# 计算多项式 f 的次数
n = dup_degree(f)
if n < 1:
    # 如果多项式没有根，则抛出异常
    raise PolynomialError('Polynomial has no roots.')

# 检查 K 是否为整数环
if K.is_ZZ:
    # 获取整数环的域
    L = K.get_field()
    # 将 f 转换为整数环 L 上的多项式，并更新 K
    f, K = dup_convert(f, K, L), L
# 如果 K 不是有理数环、实数环或复数环，则抛出域错误异常
elif not K.is_QQ or K.is_RR or K.is_CC:
    raise DomainError('Cauchy bound not supported over %s' % K)
else:
    # 复制多项式 f
    f = f[:]

# 移除多项式末尾的零系数
while K.is_zero(f[-1]):
    f.pop()
if len(f) == 1:
    # 如果多项式只有一个单项式，则所有根都为零
    return K.zero

# 取多项式的首项系数
lc = f[0]
# 计算 Cauchy 上界
return K.one + max(abs(n / lc) for n in f[1:])



# 计算多项式 f 的反向多项式 g
g = dup_reverse(f)
if len(g) < 2:
    # 如果反向多项式只有一个项，则多项式没有非零根
    raise PolynomialError('Polynomial has no non-zero roots.')
# 如果 K 是整数环，则将其转换为域
if K.is_ZZ:
    K = K.get_field()
# 计算 Cauchy 上界 b
b = dup_cauchy_upper_bound(g, K)
# 返回 Cauchy 下界的倒数
return K.one / b



# 返回多项式 f 中不同根之间的 Mignotte 下界的平方
n = dup_degree(f)
if n < 2:
    # 如果多项式次数小于 2，则不存在不同的根
    raise PolynomialError('Polynomials of degree < 2 have no distinct roots.')

# 如果 K 是整数环，则获取其域
if K.is_ZZ:
    L = K.get_field()
    f, K = dup_convert(f, K, L), L
# 如果 K 不是有理数环、实数环或复数环，则抛出域错误异常
elif not K.is_QQ or K.is_RR or K.is_CC:
    raise DomainError('Mignotte bound not supported over %s' % K)

# 计算多项式的判别式 D 和 l2 范数的平方 l2sq
D = dup_discriminant(f, K)
l2sq = dup_l2_norm_squared(f, K)
# 返回 Mignotte 下界的平方
return K(3)*K.abs(D) / ( K(n)**(n+1) * l2sq**(n-1) )



# 将开区间 I 转换为一个 Mobius 变换
s, t = I

# 分别获取 s 和 t 的分子和分母
a, c = field.numer(s), field.denom(s)
b, d = field.numer(t), field.denom(t)

# 返回 Mobius 变换的四个参数
return a, b, c, d



# 将 Mobius 变换 M 转换回开区间
a, b, c, d = M

# 分别计算 s 和 t
s, t = field(a, c), field(b, d)

# 确保返回的开区间是有序的
if s <= t:
    return (s, t)
else:
    return (t, s)



# 正实数根细化算法的一步
a, b, c, d = M

# 如果 Mobius 变换的参数相等，则返回原多项式和 Mobius 变换
if a == b and c == d:
    return f, (a, b, c, d)
    # 使用 dup_root_lower_bound 函数计算 A 的值
    A = dup_root_lower_bound(f, K)

    # 如果 A 不是 None，则将 A 转换为整数类型 K(int(A))
    if A is not None:
        A = K(int(A))
    else:
        # 如果 A 是 None，则将 A 设置为 K 的零元素
        A = K.zero

    # 如果 fast 为真且 A 大于 16
    if fast and A > 16:
        # 对 f 进行缩放，使用 A 作为参数调用 dup_scale 函数
        f = dup_scale(f, A, K)
        # 更新 a, c, A 的值
        a, c, A = A*a, A*c, K.one

    # 如果 A 大于等于 K 的一元素
    if A >= K.one:
        # 对 f 进行位移，使用 A 作为参数调用 dup_shift 函数
        f = dup_shift(f, A, K)
        # 更新 b, d 的值
        b, d = A*a + b, A*c + d

        # 如果在零点处对 f 进行求值返回 False
        if not dup_eval(f, K.zero, K):
            # 返回结果 f 和元组 (b, b, d, d)
            return f, (b, b, d, d)

    # 对 f 进行位移，使用 K 的一元素作为参数调用 dup_shift 函数
    f, g = dup_shift(f, K.one, K), f

    # 更新 a1, b1, c1, d1 的值
    a1, b1, c1, d1 = a, a + b, c, c + d

    # 如果在零点处对 f 进行求值返回 False
    if not dup_eval(f, K.zero, K):
        # 返回结果 f 和元组 (b1, b1, d1, d1)
        return f, (b1, b1, d1, d1)

    # 计算 f 的符号变化数目，赋值给 k
    k = dup_sign_variations(f, K)

    # 如果 k 等于 1
    if k == 1:
        # 更新 a, b, c, d 的值
        a, b, c, d = a1, b1, c1, d1
    else:
        # 对 g 进行逆位移，然后进行位移，使用 K 的一元素作为参数
        f = dup_shift(dup_reverse(g), K.one, K)

        # 如果在零点处对 f 进行求值返回 False
        if not dup_eval(f, K.zero, K):
            # 对 f 进行右位移，参数为 1，使用 K 作为参数
            f = dup_rshift(f, 1, K)

        # 更新 a, b, c, d 的值
        a, b, c, d = b, a + b, d, c + d

    # 返回最终结果 f 和元组 (a, b, c, d)
    return f, (a, b, c, d)
# 给定多项式 f 和一个区间 (s, t)，使用 K 域的 Mobius 变换生成对应的 a, b, c, d 值
a, b, c, d = _mobius_from_interval((s, t), K.get_field())

# 使用 Mobius 变换将多项式 f 转换为新的多项式，以适应新的区间 (a/b, c/d)
f = dup_transform(f, dup_strip([a, b]),
                     dup_strip([c, d]), K)

# 检查转换后的多项式 f 在区间 (s, t) 内是否有且仅有一个正根，若不是则引发异常
if dup_sign_variations(f, K) != 1:
    raise RefinementFailed("there should be exactly one root in (%s, %s) interval" % (s, t))

# 调用内部函数 dup_inner_refine_real_root 进一步细化正根的区间
return dup_inner_refine_real_root(f, (a, b, c, d), K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
# 使用给定的域 K 中的单位元素和零元素初始化参数 a, b, c, d
a, b, c, d = K.one, K.zero, K.zero, K.one

# 计算多项式 f 在域 K 中的符号变化次数
k = dup_sign_variations(f, K)

# 如果多项式 f 在域 K 中没有符号变化，则返回空列表
if k == 0:
    return []

# 如果多项式 f 在域 K 中只有一次符号变化，则使用内部函数 dup_inner_refine_real_root
# 对实根进行精细化处理，并返回处理后的实根列表
if k == 1:
    roots = [dup_inner_refine_real_root(
        f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True)]
    else:
        roots, stack = [], [(a, b, c, d, f, k)]

        while stack:
            a, b, c, d, f, k = stack.pop()

            # 计算多项式 f 的下界根 A
            A = dup_root_lower_bound(f, K)

            # 如果 A 不为 None，则将 A 转换为 K 类型；否则，将 A 设为 K 类型的零
            if A is not None:
                A = K(int(A))
            else:
                A = K.zero

            # 如果 fast 标志为真且 A 大于 16，则对多项式 f 进行缩放操作
            if fast and A > 16:
                f = dup_scale(f, A, K)
                a, c, A = A*a, A*c, K.one

            # 如果 A 大于等于 1，则对多项式 f 进行位移操作
            if A >= K.one:
                f = dup_shift(f, A, K)
                b, d = A*a + b, A*c + d

                # 如果多项式 f 不是常量，则记录根的信息，并将 f 右移一位
                if not dup_TC(f, K):
                    roots.append((f, (b, b, d, d)))
                    f = dup_rshift(f, 1, K)

                # 计算多项式 f 的符号变化次数
                k = dup_sign_variations(f, K)

                # 如果符号变化次数为 0，则继续下一轮循环
                if k == 0:
                    continue
                # 如果符号变化次数为 1，则对单根进行精细化处理并记录
                if k == 1:
                    roots.append(dup_inner_refine_real_root(
                        f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True))
                    continue

            # 对多项式 f 进行单位位移
            f1 = dup_shift(f, K.one, K)

            # 计算新的系数
            a1, b1, c1, d1, r = a, a + b, c, c + d, 0

            # 如果多项式 f1 不是常量，则记录根的信息，并将 f1 右移一位
            if not dup_TC(f1, K):
                roots.append((f1, (b1, b1, d1, d1)))
                f1, r = dup_rshift(f1, 1, K), 1

            # 计算多项式 f1 的符号变化次数
            k1 = dup_sign_variations(f1, K)
            k2 = k - k1 - r

            # 计算新的系数
            a2, b2, c2, d2 = b, a + b, d, c + d

            # 如果 k2 大于 1，则对多项式 f2 进行单位位移并计算其符号变化次数
            if k2 > 1:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                # 如果多项式 f2 不是常量，则将 f2 右移一位
                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

                # 计算多项式 f2 的符号变化次数
                k2 = dup_sign_variations(f2, K)
            else:
                f2 = None

            # 如果 k1 小于 k2，则交换系数顺序
            if k1 < k2:
                a1, a2, b1, b2 = a2, a1, b2, b1
                c1, c2, d1, d2 = c2, c1, d2, d1
                f1, f2, k1, k2 = f2, f1, k2, k1

            # 如果 k1 为 0，则继续下一轮循环
            if not k1:
                continue

            # 如果 f1 为 None，则对多项式 f 进行单位位移并处理
            if f1 is None:
                f1 = dup_shift(dup_reverse(f), K.one, K)

                # 如果多项式 f1 不是常量，则将 f1 右移一位
                if not dup_TC(f1, K):
                    f1 = dup_rshift(f1, 1, K)

            # 如果 k1 为 1，则对单根进行精细化处理并记录
            if k1 == 1:
                roots.append(dup_inner_refine_real_root(
                    f1, (a1, b1, c1, d1), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a1, b1, c1, d1, f1, k1))

            # 如果 k2 为 0，则继续下一轮循环
            if not k2:
                continue

            # 如果 f2 为 None，则对多项式 f 进行单位位移并处理
            if f2 is None:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                # 如果多项式 f2 不是常量，则将 f2 右移一位
                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

            # 如果 k2 为 1，则对单根进行精细化处理并记录
            if k2 == 1:
                roots.append(dup_inner_refine_real_root(
                    f2, (a2, b2, c2, d2), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a2, b2, c2, d2, f2, k2))

    # 返回计算得到的根列表
    return roots
# 当 isolating interval 超出 ``(inf, sup)`` 范围时丢弃它
def _discard_if_outside_interval(f, M, inf, sup, K, negative, fast, mobius):
    """Discard an isolating interval if outside ``(inf, sup)``. """
    F = K.get_field()  # 获取域 F

    while True:
        u, v = _mobius_to_interval(M, F)  # 使用 Mobius 变换得到区间 (u, v)

        if negative:
            u, v = -v, -u  # 如果是负数，取反区间

        # 检查是否在所需区间内，如果是则返回相应的值
        if (inf is None or u >= inf) and (sup is None or v <= sup):
            if not mobius:
                return u, v  # 不使用 Mobius 变换，返回区间 (u, v)
            else:
                return f, M  # 使用 Mobius 变换，返回函数 f 和区间 M
        elif (sup is not None and u > sup) or (inf is not None and v < inf):
            return None  # 区间超出了 (inf, sup) 范围，返回 None
        else:
            f, M = dup_step_refine_real_root(f, M, K, fast=fast)  # 对实根进行精细化处理

# 迭代计算不相交的正实根隔离区间
def dup_inner_isolate_positive_roots(f, K, eps=None, inf=None, sup=None, fast=False, mobius=False):
    """Iteratively compute disjoint positive root isolation intervals. """
    if sup is not None and sup < 0:
        return []  # 如果 sup 存在且小于 0，返回空列表

    roots = dup_inner_isolate_real_roots(f, K, eps=eps, fast=fast)  # 计算实根隔离区间

    F, results = K.get_field(), []  # 获取域 F 和结果列表

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, False, fast, mobius)  # 检查区间是否在 (inf, sup) 范围内

            if result is not None:
                results.append(result)  # 将符合条件的区间结果添加到列表中
    elif not mobius:
        results.extend(_mobius_to_interval(M, F) for _, M in roots)  # 使用 Mobius 变换计算区间并扩展到结果列表中
    else:
        results = roots  # 如果使用 Mobius 变换，则直接使用 roots 作为结果

    return results  # 返回最终结果列表

# 迭代计算不相交的负实根隔离区间
def dup_inner_isolate_negative_roots(f, K, inf=None, sup=None, eps=None, fast=False, mobius=False):
    """Iteratively compute disjoint negative root isolation intervals. """
    if inf is not None and inf >= 0:
        return []  # 如果 inf 存在且大于等于 0，返回空列表

    roots = dup_inner_isolate_real_roots(dup_mirror(f, K), K, eps=eps, fast=fast)  # 计算镜像后的实根隔离区间

    F, results = K.get_field(), []  # 获取域 F 和结果列表

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, True, fast, mobius)  # 检查区间是否在 (inf, sup) 范围内

            if result is not None:
                results.append(result)  # 将符合条件的区间结果添加到列表中
    elif not mobius:
        for f, M in roots:
            u, v = _mobius_to_interval(M, F)  # 使用 Mobius 变换计算区间
            results.append((-v, -u))  # 添加负区间到结果列表中
    else:
        results = roots  # 如果使用 Mobius 变换，则直接使用 roots 作为结果

    return results  # 返回最终结果列表

# 处理 CF 算法的特殊情况，当 f 是齐次时
def _isolate_zero(f, K, inf, sup, basis=False, sqf=False):
    """Handle special case of CF algorithm when ``f`` is homogeneous. """
    j, f = dup_terms_gcd(f, K)  # 计算 f 和 K 的最大公因式

    if j > 0:
        F = K.get_field()  # 获取域 F

        # 如果在所需区间内，返回相应的结果
        if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
            if not sqf:
                if not basis:
                    return [((F.zero, F.zero), j)], f  # 返回相应的根和函数 f
                else:
                    return [((F.zero, F.zero), j, [K.one, K.zero])], f  # 返回相应的根、函数 f 和基础数据
            else:
                return [(F.zero, F.zero)], f  # 返回零多项式的根和函数 f

    return [], f  # 否则返回空列表和函数 f

# 对方程进行 SQF 算法处理，计算实根隔离区间
def dup_isolate_real_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real roots of a square-free polynomial using the Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========
       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots. Nonlinear Analysis:
              Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    # 检查域是否为有理数域 QQ
    if K.is_QQ:
        # 清除多项式 f 的分母，得到清理后的多项式
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    # 如果域不是整数环 ZZ，则抛出域错误异常
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    # 如果多项式 f 的次数小于等于 0，则返回空列表
    if dup_degree(f) <= 0:
        return []

    # 使用 _isolate_zero 函数分离多项式 f 的实根区间 I_zero，并更新多项式 f
    I_zero, f = _isolate_zero(f, K, inf, sup, basis=False, sqf=True)

    # 分离多项式 f 的负实根区间 I_neg
    I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
    # 分离多项式 f 的正实根区间 I_pos
    I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

    # 将所有实根区间按照大小排序，并合并到 roots 列表中
    roots = sorted(I_neg + I_zero + I_pos)

    # 如果 blackbox 不为真，则返回根列表
    if not blackbox:
        return roots
    # 否则，返回实根区间的列表
    else:
        return [ RealInterval((a, b), f, K) for (a, b) in roots ]
# 利用 Vincent-Akritas-Strzebonski (VAS) 连分数方法，隔离多项式 f 的实根
def dup_isolate_real_roots(f, K, eps=None, inf=None, sup=None, basis=False, fast=False):
    """Isolate real roots using Vincent-Akritas-Strzebonski (VAS) continued fractions approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    # 如果 K 是有理数域 QQ
    if K.is_QQ:
        # 清除 f 的分母，并转换为整数系数多项式
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    # 如果 K 不是整数环 ZZ，则抛出域错误
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    # 如果 f 的次数小于等于 0，则返回空列表
    if dup_degree(f) <= 0:
        return []

    # 隔离多项式 f 的零点
    I_zero, f = _isolate_zero(f, K, inf, sup, basis=basis, sqf=False)

    # 对 f 进行平方因式分解
    _, factors = dup_sqf_list(f, K)

    # 如果只有一个因子
    if len(factors) == 1:
        # 解别多项式 f 的负实根
        I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
        # 解别多项式 f 的正实根
        I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        # 转换负实根的格式为 ((u, v), k)
        I_neg = [ ((u, v), k) for u, v in I_neg ]
        # 转换正实根的格式为 ((u, v), k)
        I_pos = [ ((u, v), k) for u, v in I_pos ]
    else:
        # 处理多个因子的情况，隔离实根并分离
        I_neg, I_pos = _real_isolate_and_disjoin(factors, K,
            eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)

    # 返回排序后的负实根、零点和正实根列表
    return sorted(I_neg + I_zero + I_pos)

# 隔离多项式列表 polys 中各多项式的实根
def dup_isolate_real_roots_list(polys, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of square-free polynomial using Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    # 如果 K 是有理数域 QQ
    if K.is_QQ:
        # 获取 K 的整数环，同时转换 polys 中的每个多项式为整数系数多项式
        K, F, polys = K.get_ring(), K, polys[:]

        for i, p in enumerate(polys):
            polys[i] = dup_clear_denoms(p, F, K, convert=True)[1]
    # 如果 K 不是整数环 ZZ，则抛出域错误
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    # 是否找到了零点和因子字典初始化为 False 和 空字典
    zeros, factors_dict = False, {}

    # 如果 inf 为 None 或 inf <= 0，且 sup 为 None 或 sup >= 0，则设置 zeros 为 True 和 zero_indices 为空字典
    if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
        zeros, zero_indices = True, {}
    # 遍历多项式列表，获取索引和多项式
    for i, p in enumerate(polys):
        # 对当前多项式 p 进行因式分解并返回最大公约数
        j, p = dup_terms_gcd(p, K)

        # 如果存在零点且 j 大于 0，则记录该多项式在 zero_indices 中的索引
        if zeros and j > 0:
            zero_indices[i] = j

        # 对 p 进行因式分解，返回分解后的因子列表
        for f, k in dup_factor_list(p, K)[1]:
            # 将因子 f 转换为元组形式
            f = tuple(f)

            # 如果因子 f 不在 factors_dict 中，则将其加入，并初始化对应的字典
            if f not in factors_dict:
                factors_dict[f] = {i: k}
            else:
                # 如果因子 f 已经存在于 factors_dict 中，则更新其对应的字典
                factors_dict[f][i] = k

    # 将 factors_dict 中的数据转换为列表形式，并存入 factors_list 中
    factors_list = [(list(f), indices) for f, indices in factors_dict.items()]

    # 对实数多项式进行隔离与分离，返回负根和正根的区间列表
    I_neg, I_pos = _real_isolate_and_disjoin(factors_list, K, eps=eps,
        inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)

    # 获取域 K 的字段
    F = K.get_field()

    # 如果没有零点或者 zero_indices 为空，则初始化 I_zero 为空列表
    if not zeros or not zero_indices:
        I_zero = []
    else:
        # 如果存在零点且基不为空，则创建包含零点信息的列表
        if not basis:
            I_zero = [((F.zero, F.zero), zero_indices)]
        else:
            # 如果基不为空，则包含额外的基信息
            I_zero = [((F.zero, F.zero), zero_indices, [K.one, K.zero])]

    # 将负根、零点和正根的列表合并并进行排序，然后返回结果
    return sorted(I_neg + I_zero + I_pos)
def _disjoint_p(M, N, strict=False):
    """Check if Mobius transforms define disjoint intervals. """
    # 解构输入的 Mobius 变换 M 和 N
    a1, b1, c1, d1 = M
    a2, b2, c2, d2 = N

    # 计算交叉乘积 a1*d1 和 b1*c1，以及 a2*d2 和 b2*c2
    a1d1, b1c1 = a1*d1, b1*c1
    a2d2, b2c2 = a2*d2, b2*c2

    # 如果两对变换的交叉乘积相等，则视为不交叠
    if a1d1 == b1c1 and a2d2 == b2c2:
        return True

    # 如果第一对变换的交叉乘积大于第二对，交换变换顺序
    if a1d1 > b1c1:
        a1, c1, b1, d1 = b1, d1, a1, c1

    # 如果第二对变换的交叉乘积大于第一对，交换变换顺序
    if a2d2 > b2c2:
        a2, c2, b2, d2 = b2, d2, a2, c2

    # 根据 strict 参数判断是否严格不交叠
    if not strict:
        return a2*d1 >= c2*b1 or b2*c1 <= d2*a1
    else:
        return a2*d1 > c2*b1 or b2*c1 < d2*a1

def _real_isolate_and_disjoin(factors, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of polynomials and disjoin intervals. """
    # 初始化正根和负根的列表
    I_pos, I_neg = [], []

    # 对每个多项式和其次数进行处理
    for i, (f, k) in enumerate(factors):
        # 针对每个多项式找出正根的区间，并添加到正根列表中
        for F, M in dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_pos.append((F, M, k, f))

        # 针对每个多项式找出负根的区间，并添加到负根列表中
        for G, N in dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_neg.append((G, N, k, f))

    # 对正根列表中的每一项，确保它们与其他项不交叠
    for i, (f, M, k, F) in enumerate(I_pos):
        for j, (g, N, m, G) in enumerate(I_pos[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                # 如果存在交叠，对根进行细化，直到不交叠为止
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_pos[i + j + 1] = (g, N, m, G)

        I_pos[i] = (f, M, k, F)

    # 对负根列表中的每一项，确保它们与其他项不交叠
    for i, (f, M, k, F) in enumerate(I_neg):
        for j, (g, N, m, G) in enumerate(I_neg[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                # 如果存在交叠，对根进行细化，直到不交叠为止
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_neg[i + j + 1] = (g, N, m, G)

        I_neg[i] = (f, M, k, F)

    # 如果 strict 参数为真，则确保负根和正根列表中的第一项不交叠
    if strict:
        for i, (f, M, k, F) in enumerate(I_neg):
            if not M[0]:
                while not M[0]:
                    # 对根进行细化，直到不交叠为止
                    f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)

                I_neg[i] = (f, M, k, F)
                break

        for j, (g, N, m, G) in enumerate(I_pos):
            if not N[0]:
                while not N[0]:
                    # 对根进行细化，直到不交叠为止
                    g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

                I_pos[j] = (g, N, m, G)

    # 获取域信息
    field = K.get_field()

    # 转换负根列表中的 Mobius 变换为区间表示
    I_neg = [(_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_neg]
    # 转换正根列表中的 Mobius 变换为区间表示
    I_pos = [(_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_pos]

    # 如果 basis 参数为假，则只返回区间和次数，不返回多项式
    if not basis:
        I_neg = [((u, v), k) for ((u, v), k, _) in I_neg]
        I_pos = [((u, v), k) for ((u, v), k, _) in I_pos]

    # 返回负根和正根的列表
    return I_neg, I_pos

def dup_count_real_roots(f, K, inf=None, sup=None):
    """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
    # 检查多项式 f 的次数，若次数小于等于 0，则没有实数根
    if dup_degree(f) <= 0:
        return 0

    # 如果 K 不是一个域，则转换 K 成为其域
    if not K.is_Field:
        R, K = K, K.get_field()
        f = dup_convert(f, R, K)

    # 使用 Sturm 序列计算多项式 f 在域 K 上的 Sturm 序列
    sturm = dup_sturm(f, K)

    # 计算在 inf 点的 Sturm 变号数
    if inf is None:
        signs_inf = dup_sign_variations([ dup_LC(s, K)*(-1)**dup_degree(s) for s in sturm ], K)
    else:
        signs_inf = dup_sign_variations([ dup_eval(s, inf, K) for s in sturm ], K)

    # 计算在 sup 点的 Sturm 变号数
    if sup is None:
        signs_sup = dup_sign_variations([ dup_LC(s, K) for s in sturm ], K)
    else:
        signs_sup = dup_sign_variations([ dup_eval(s, sup, K) for s in sturm ], K)

    # 计算在区间 [inf, sup] 内多项式 f 实根的个数
    count = abs(signs_inf - signs_sup)

    # 如果 inf 不为 None 且 f(inf) == 0，则多项式 f 在 inf 处有一个额外的实根
    if inf is not None and not dup_eval(f, inf, K):
        count += 1

    # 返回实根的个数
    return count
OO = 'OO'  # 定义坐标系原点（re, im）

Q1 = 'Q1'  # 第一象限 (++): re > 0 and im > 0
Q2 = 'Q2'  # 第二象限 (-+): re < 0 and im > 0
Q3 = 'Q3'  # 第三象限 (--): re < 0 and im < 0
Q4 = 'Q4'  # 第四象限 (+-): re > 0 and im < 0

A1 = 'A1'  # 轴1 (+0): re > 0 and im = 0
A2 = 'A2'  # 轴2 (0+): re = 0 and im > 0
A3 = 'A3'  # 轴3 (-0): re < 0 and im = 0
A4 = 'A4'  # 轴4 (0-): re = 0 and im < 0

_rules_simple = {
    # Q --> Q (相同象限) => 无变化
    (Q1, Q1): 0,
    (Q2, Q2): 0,
    (Q3, Q3): 0,
    (Q4, Q4): 0,

    # A -- 逆时针旋转 --> Q => +1/4 (逆时针)
    (A1, Q1): 1,
    (A2, Q2): 1,
    (A3, Q3): 1,
    (A4, Q4): 1,

    # A -- 顺时针旋转 --> Q => -1/4 (逆时针)
    (A1, Q4): 2,
    (A2, Q1): 2,
    (A3, Q2): 2,
    (A4, Q3): 2,

    # Q -- 逆时针旋转 --> A => +1/4 (逆时针)
    (Q1, A2): 3,
    (Q2, A3): 3,
    (Q3, A4): 3,
    (Q4, A1): 3,

    # Q -- 顺时针旋转 --> A => -1/4 (逆时针)
    (Q1, A1): 4,
    (Q2, A2): 4,
    (Q3, A3): 4,
    (Q4, A4): 4,

    # Q -- 逆时针旋转 --> Q => +1/2 (逆时针)
    (Q1, Q2): +5,
    (Q2, Q3): +5,
    (Q3, Q4): +5,
    (Q4, Q1): +5,

    # Q -- 顺时针旋转 --> Q => -1/2 (顺时针)
    (Q1, Q4): -5,
    (Q2, Q1): -5,
    (Q3, Q2): -5,
    (Q4, Q3): -5,
}

_rules_ambiguous = {
    # A -- 逆时针旋转 --> Q => { +1/4 (逆时针), -9/4 (顺时针) }
    (A1, OO, Q1): -1,
    (A2, OO, Q2): -1,
    (A3, OO, Q3): -1,
    (A4, OO, Q4): -1,

    # A -- 顺时针旋转 --> Q => { -1/4 (逆时针), +7/4 (顺时针) }
    (A1, OO, Q4): -2,
    (A2, OO, Q1): -2,
    (A3, OO, Q2): -2,
    (A4, OO, Q3): -2,

    # Q -- 逆时针旋转 --> A => { +1/4 (逆时针), -9/4 (顺时针) }
    (Q1, OO, A2): -3,
    (Q2, OO, A3): -3,
    (Q3, OO, A4): -3,
    (Q4, OO, A1): -3,

    # Q -- 顺时针旋转 --> A => { -1/4 (逆时针), +7/4 (顺时针) }
    (Q1, OO, A1): -4,
    (Q2, OO, A2): -4,
    (Q3, OO, A3): -4,
    (Q4, OO, A4): -4,

    # A --  OO --> A => { +1 (逆时针), -1 (顺时针) }
    (A1, A3): 7,
    (A2, A4): 7,
    (A3, A1): 7,
    (A4, A2): 7,

    (A1, OO, A3): 7,
    (A2, OO, A4): 7,
    (A3, OO, A1): 7,
    (A4, OO, A2): 7,

    # Q -- 对角线 --> Q => { +1 (逆时针), -1 (顺时针) }
    (Q1, Q3): 8,
    (Q2, Q4): 8,
    (Q3, Q1): 8,
    (Q4, Q2): 8,

    (Q1, OO, Q3): 8,
    (Q2, OO, Q4): 8,
    (Q3, OO, Q1): 8,
    (Q4, OO, Q2): 8,

    # A -- 右转 ---> A => { +1/2 (逆时针), -3/2 (顺时针) }
    (A1, A2): 9,
    (A2, A3): 9,
    (A3, A4): 9,
    (A4, A1): 9,

    (A1, OO, A2): 9,
    (A2, OO, A3): 9,
    (A3, OO, A4): 9,
    (A4, OO, A1): 9,

    # A -- 左转 ---> A => { +3/2 (逆时针), -1/2 (顺时针) }
    (A1, A4): 10,
    (A2, A1): 10,
    (A3, A2): 10,
    (A4, A3): 10,

    (A1, OO, A4): 10,
    (A2, OO, A1): 10,
    (A3, OO, A2): 10,
    (A4, OO, A3): 10,

    # Q -- 1 ---> A => { +3/4 (逆时针), -5/4 (顺时针) }
    (Q1, A3): 11,
    (Q2, A4): 11,
    (Q3, A1): 11,
    (Q4, A2): 11,

    (Q1, OO, A3): 11,
    (Q2, OO, A4): 11,
    (Q3, OO, A1): 11,
    (Q4, OO, A2): 11,

    # Q -- 2 ---> A => { +5/4 (逆时针), -3/4 (顺时针) }
    (Q1, A4): 12,
    (Q2, A1): 12,
    (Q3, A2): 12,
    (Q4, A3): 12,

    (Q1, OO, A4): 12,
    (Q2, OO, A1): 12,
    (Q3, OO, A2): 12,
    (Q4, OO, A3): 12,
}
    # 定义字典中的键值对，表示从状态 A 到状态 Q 的转移，每个转移的权重为 13
    (A1, Q3): 13,
    (A2, Q4): 13,
    (A3, Q1): 13,
    (A4, Q2): 13,

    # 定义字典中的键值对，表示从状态 A 到状态 Q 经过 OO 的转移，每个转移的权重为 13
    (A1, OO, Q3): 13,
    (A2, OO, Q4): 13,
    (A3, OO, Q1): 13,
    (A4, OO, Q2): 13,

    # 定义字典中的键值对，表示从状态 A 到状态 Q 的另一种转移方式，每个转移的权重为 14
    (A1, Q2): 14,
    (A2, Q3): 14,
    (A3, Q4): 14,
    (A4, Q1): 14,

    # 定义字典中的键值对，表示从状态 A 到状态 Q 经过 OO 的另一种转移方式，每个转移的权重为 14
    (A1, OO, Q2): 14,
    (A2, OO, Q3): 14,
    (A3, OO, Q4): 14,
    (A4, OO, Q1): 14,

    # 定义字典中的键值对，表示从状态 Q 到状态 Q 经过 OO 的转移，每个转移的权重为 15
    (Q1, OO, Q2): 15,
    (Q2, OO, Q3): 15,
    (Q3, OO, Q4): 15,
    (Q4, OO, Q1): 15,

    # 定义字典中的键值对，表示从状态 Q 到状态 Q 经过 OO 的另一种转移方式，每个转移的权重为 16
    (Q1, OO, Q4): 16,
    (Q2, OO, Q1): 16,
    (Q3, OO, Q2): 16,
    (Q4, OO, Q3): 16,

    # 定义字典中的键值对，表示从状态 A 到状态 A 经过 OO 的转移，每个转移的权重为 17
    (A1, OO, A1): 17,
    (A2, OO, A2): 17,
    (A3, OO, A3): 17,
    (A4, OO, A4): 17,

    # 定义字典中的键值对，表示从状态 Q 到状态 Q 经过 OO 的转移，每个转移的权重为 18
    (Q1, OO, Q1): 18,
    (Q2, OO, Q2): 18,
    (Q3, OO, Q3): 18,
    (Q4, OO, Q4): 18,
_values = {
    0: [( 0, 1)],  # 键为0，值为一个包含元组的列表，元组为(0, 1)
    1: [(+1, 4)],  # 键为1，值为一个包含元组的列表，元组为(+1, 4)
    2: [(-1, 4)],  # 键为2，值为一个包含元组的列表，元组为(-1, 4)
    3: [(+1, 4)],  # 键为3，值为一个包含元组的列表，元组为(+1, 4)
    4: [(-1, 4)],  # 键为4，值为一个包含元组的列表，元组为(-1, 4)
    -1: [(+9, 4), (+1, 4)],  # 键为-1，值为一个包含两个元组的列表，元组分别为(+9, 4)和(+1, 4)
    -2: [(+7, 4), (-1, 4)],  # 键为-2，值为一个包含两个元组的列表，元组分别为(+7, 4)和(-1, 4)
    -3: [(+9, 4), (+1, 4)],  # 键为-3，值为一个包含两个元组的列表，元组分别为(+9, 4)和(+1, 4)
    -4: [(+7, 4), (-1, 4)],  # 键为-4，值为一个包含两个元组的列表，元组分别为(+7, 4)和(-1, 4)
    +5: [(+1, 2)],  # 键为+5，值为一个包含元组的列表，元组为(+1, 2)
    -5: [(-1, 2)],  # 键为-5，值为一个包含元组的列表，元组为(-1, 2)
    7: [(+1, 1), (-1, 1)],  # 键为7，值为一个包含两个元组的列表，元组分别为(+1, 1)和(-1, 1)
    8: [(+1, 1), (-1, 1)],  # 键为8，值为一个包含两个元组的列表，元组分别为(+1, 1)和(-1, 1)
    9: [(+1, 2), (-3, 2)],  # 键为9，值为一个包含两个元组的列表，元组分别为(+1, 2)和(-3, 2)
    10: [(+3, 2), (-1, 2)],  # 键为10，值为一个包含两个元组的列表，元组分别为(+3, 2)和(-1, 2)
    11: [(+3, 4), (-5, 4)],  # 键为11，值为一个包含两个元组的列表，元组分别为(+3, 4)和(-5, 4)
    12: [(+5, 4), (-3, 4)],  # 键为12，值为一个包含两个元组的列表，元组分别为(+5, 4)和(-3, 4)
    13: [(+5, 4), (-3, 4)],  # 键为13，值为一个包含两个元组的列表，元组分别为(+5, 4)和(-3, 4)
    14: [(+3, 4), (-5, 4)],  # 键为14，值为一个包含两个元组的列表，元组分别为(+3, 4)和(-5, 4)
    15: [(+1, 2), (-3, 2)],  # 键为15，值为一个包含两个元组的列表，元组分别为(+1, 2)和(-3, 2)
    16: [(+3, 2), (-1, 2)],  # 键为16，值为一个包含两个元组的列表，元组分别为(+3, 2)和(-1, 2)
    17: [(+2, 1), ( 0, 1)],  # 键为17，值为一个包含两个元组的列表，元组分别为(+2, 1)和( 0, 1)
    18: [(+2, 1), ( 0, 1)],  # 键为18，值为一个包含两个元组的列表，元组分别为(+2, 1)和( 0, 1)
}

def _classify_point(re, im):
    """Return the half-axis (or origin) on which (re, im) point is located. """
    if not re and not im:  # 如果实部和虚部都为0
        return OO  # 返回OO

    if not re:  # 如果实部为0
        if im > 0:
            return A2  # 如果虚部大于0，返回A2
        else:
            return A4  # 否则返回A4
    elif not im:  # 如果虚部为0
        if re > 0:
            return A1  # 如果实部大于0，返回A1
        else:
            return A3  # 否则返回A3

def _intervals_to_quadrants(intervals, f1, f2, s, t, F):
    """Generate a sequence of extended quadrants from a list of critical points. """
    if not intervals:  # 如果intervals为空列表
        return []  # 返回空列表

    Q = []  # 初始化空列表Q

    if not f1:  # 如果f1为假值（空、0等）
        (a, b), _, _ = intervals[0]  # 获取intervals列表中第一个元素的第一个元组(a, b)，忽略后两个元素

        if a == b == s:  # 如果a、b都等于s
            if len(intervals) == 1:  # 如果intervals列表长度为1
                if dup_eval(f2, t, F) > 0:  # 如果dup_eval(f2, t, F)大于0
                    return [OO, A2]  # 返回包含OO和A2的列表
                else:
                    return [OO, A4]  # 否则返回包含OO和A4的列表
            else:
                (a, _), _, _ = intervals[1]  # 获取intervals列表中第二个元素的第一个元组(a, _)，忽略后两个元素

                if dup_eval(f2, (s + a)/2, F) > 0:  # 如果dup_eval(f2, (s + a)/2, F)大于0
                    Q.extend([OO, A2])  # 扩展Q列表添加OO和A2
                    f2_sgn = +1  # 设置f2_sgn为+1
                else:
                    Q.extend([OO, A4])  # 否则扩展Q列表添加OO和A4
                    f2_sgn = -1  # 设置f2_sgn为-1

                intervals = intervals[1:]  # 更新intervals列表为去掉第一个元素后的部分
        else:
            if dup_eval(f2, s, F) > 0:  # 如果dup_eval(f2, s, F)大于0
                Q.append(A2)  # 添加A2到Q列表
                f2_sgn = +1  # 设置f2_sgn为+1
            else:
                Q.append(A4)  # 否则添加A4到Q列表
                f2_sgn = -1  # 设置f2_sgn为-1

        for (a, _), indices, _ in intervals:  # 遍历intervals列表中的元素，每个元素是一个元组(a, _)，忽略后两个元素
            Q.append(OO)  # 添加OO到Q列表

            if indices[1] % 2 == 1:  # 如果indices列表中第二个元素对2取模等于1
                f2_sgn = -f2_sgn  # 更新f2_sgn为其相反数

            if a != t:  # 如果a不等于t
                if f2_sgn > 0:  # 如果f2_sgn大于0
                    Q.append(A2)  # 添加A2到Q列表
                else:
                    Q.append(A4
    # 如果 f2 为空
    if not f2:
        # 从 intervals 列表中获取第一个元素的第一个子元素 (a, b)，并丢弃其余两个子元素
        (a, b), _, _ = intervals[0]

        # 如果 a 等于 b 等于 s
        if a == b == s:
            # 如果 intervals 只有一个元素
            if len(intervals) == 1:
                # 计算 f1 在 t 处的重复计算值
                if dup_eval(f1, t, F) > 0:
                    # 返回结果列表 [OO, A1]
                    return [OO, A1]
                else:
                    # 返回结果列表 [OO, A3]
                    return [OO, A3]
            else:
                # 获取 intervals 中的第二个元素的第一个子元素 (a, _)
                (a, _), _, _ = intervals[1]

                # 计算 f1 在 (s + a)/2 处的重复计算值
                if dup_eval(f1, (s + a)/2, F) > 0:
                    # 扩展结果列表 Q，加入 [OO, A1]
                    Q.extend([OO, A1])
                    # 设置 f1_sgn 为 +1
                    f1_sgn = +1
                else:
                    # 扩展结果列表 Q，加入 [OO, A3]
                    Q.extend([OO, A3])
                    # 设置 f1_sgn 为 -1
                    f1_sgn = -1

                # 丢弃 intervals 中的第一个元素
                intervals = intervals[1:]
        else:
            # 计算 f1 在 s 处的重复计算值
            if dup_eval(f1, s, F) > 0:
                # 将 A1 添加到结果列表 Q
                Q.append(A1)
                # 设置 f1_sgn 为 +1
                f1_sgn = +1
            else:
                # 将 A3 添加到结果列表 Q
                Q.append(A3)
                # 设置 f1_sgn 为 -1
                f1_sgn = -1

        # 遍历 intervals 列表中的元素
        for (a, _), indices, _ in intervals:
            # 添加 OO 到结果列表 Q
            Q.append(OO)

            # 如果 indices 中的第一个元素是奇数
            if indices[0] % 2 == 1:
                # 反转 f1_sgn 的符号
                f1_sgn = -f1_sgn

            # 如果 a 不等于 t
            if a != t:
                # 如果 f1_sgn 大于 0
                if f1_sgn > 0:
                    # 将 A1 添加到结果列表 Q
                    Q.append(A1)
                else:
                    # 将 A3 添加到结果列表 Q
                    Q.append(A3)

        # 返回结果列表 Q
        return Q

    # 计算 f1 在 s 处的重复计算值
    re = dup_eval(f1, s, F)
    # 计算 f2 在 s 处的重复计算值
    im = dup_eval(f2, s, F)

    # 如果 re 或者 im 为空
    if not re or not im:
        # 将 _classify_point(re, im) 的结果添加到结果列表 Q
        Q.append(_classify_point(re, im))

        # 如果 intervals 只有一个元素
        if len(intervals) == 1:
            # 计算 f1 在 t 处的重复计算值
            re = dup_eval(f1, t, F)
            # 计算 f2 在 t 处的重复计算值
            im = dup_eval(f2, t, F)
        else:
            # 获取 intervals 中的第二个元素的第一个子元素 (a, _)
            (a, _), _, _ = intervals[1]

            # 计算 f1 在 (s + a)/2 处的重复计算值
            re = dup_eval(f1, (s + a)/2, F)
            # 计算 f2 在 (s + a)/2 处的重复计算值
            im = dup_eval(f2, (s + a)/2, F)

        # 丢弃 intervals 中的第一个元素
        intervals = intervals[1:]

    # 如果 re 大于 0
    if re > 0:
        # 设置 f1_sgn 为 +1
        f1_sgn = +1
    else:
        # 设置 f1_sgn 为 -1
        f1_sgn = -1

    # 如果 im 大于 0
    if im > 0:
        # 设置 f2_sgn 为 +1
        f2_sgn = +1
    else:
        # 设置 f2_sgn 为 -1
        f2_sgn = -1

    # 定义一个字典 sgn，映射 (f1_sgn, f2_sgn) 到相应的结果标签
    sgn = {
        (+1, +1): Q1,
        (-1, +1): Q2,
        (-1, -1): Q3,
        (+1, -1): Q4,
    }

    # 将 sgn[(f1_sgn, f2_sgn)] 添加到结果列表 Q
    Q.append(sgn[(f1_sgn, f2_sgn)])

    # 遍历 intervals 列表中的元素
    for (a, b), indices, _ in intervals:
        # 如果 a 等于 b
        if a == b:
            # 计算 f1 在 a 处的重复计算值
            re = dup_eval(f1, a, F)
            # 计算 f2 在 a 处的重复计算值
            im = dup_eval(f2, a, F)

            # 对点 (re, im) 进行分类
            cls = _classify_point(re, im)

            # 如果分类结果不为 None
            if cls is not None:
                # 将 cls 添加到结果列表 Q
                Q.append(cls)

        # 如果 indices 中包含 0
        if 0 in indices:
            # 如果 indices 中的第一个元素是奇数
            if indices[0] % 2 == 1:
                # 反转 f1_sgn 的符号
                f1_sgn = -f1_sgn

        # 如果 indices 中包含 1
        if 1 in indices:
            # 如果 indices 中的第二个元素是奇数
            if indices[1] % 2 == 1:
                # 反转 f2_sgn 的符号
                f2_sgn = -f2_sgn

        # 如果 a 不等于 b 且不等于 t
        if not (a == b and b == t):
            # 将 sgn[(f1_sgn, f2_sgn)] 添加到结果列表 Q
            Q.append(sgn[(f1_sgn, f2_sgn)])

    # 返回结果列表 Q
    return Q
# 将四个象限序列转换为规则序列的函数
def _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=None):
    """Transform sequences of quadrants to a sequence of rules. """
    # 如果 exclude 为 True，则设置边缘为 [1, 1, 0, 0]
    if exclude is True:
        edges = [1, 1, 0, 0]
        # 设置角落字典，指定每个角落的值
        corners = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 0,
            (3, 0): 1,
        }
    else:
        edges = [0, 0, 0, 0]
        # 设置角落字典，所有角落的值为 0
        corners = {
            (0, 1): 0,
            (1, 2): 0,
            (2, 3): 0,
            (3, 0): 0,
        }

    # 如果 exclude 不为 None 且不为 True，则将 exclude 转换为集合类型
    if exclude is not None and exclude is not True:
        exclude = set(exclude)

        # 遍历边缘列表，根据 exclude 设置相应边缘的值为 1
        for i, edge in enumerate(['S', 'E', 'N', 'W']):
            if edge in exclude:
                edges[i] = 1

        # 遍历角落列表，根据 exclude 设置相应角落的值为 1
        for i, corner in enumerate(['SW', 'SE', 'NE', 'NW']):
            if corner in exclude:
                corners[((i - 1) % 4, i)] = 1

    # 将四个象限列表组成 QQ 列表
    QQ, rules = [Q_L1, Q_L2, Q_L3, Q_L4], []

    # 遍历 QQ 列表中的象限
    for i, Q in enumerate(QQ):
        # 如果 Q 为空列表，则跳过
        if not Q:
            continue

        # 如果象限最后一个元素为 OO，则去除最后一个元素
        if Q[-1] == OO:
            Q = Q[:-1]

        # 如果象限第一个元素为 OO
        if Q[0] == OO:
            j, Q = (i - 1) % 4, Q[1:]
            qq = (QQ[j][-2], OO, Q[0])

            # 如果 qq 在 _rules_ambiguous 中，则添加规则和对应角落的值到 rules 列表中
            if qq in _rules_ambiguous:
                rules.append((_rules_ambiguous[qq], corners[(j, i)]))
            else:
                raise NotImplementedError("3 element rule (corner): " + str(qq))

        q1, k = Q[0], 1

        # 遍历象限 Q 中的元素
        while k < len(Q):
            q2, k = Q[k], k + 1

            # 如果 q2 不为 OO
            if q2 != OO:
                qq = (q1, q2)

                # 如果 qq 在 _rules_simple 中，则添加规则和边缘的值 0 到 rules 列表中
                if qq in _rules_simple:
                    rules.append((_rules_simple[qq], 0))
                # 如果 qq 在 _rules_ambiguous 中，则添加规则和对应边缘的值到 rules 列表中
                elif qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("2 element rule (inside): " + str(qq))
            else:
                # 如果 q2 为 OO，则将 qq 设为 (q1, q2, Q[k])，并递增 k
                qq, k = (q1, q2, Q[k]), k + 1

                # 如果 qq 在 _rules_ambiguous 中，则添加规则和对应边缘的值到 rules 列表中
                if qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("3 element rule (edge): " + str(qq))

            q1 = qq[-1]

    # 返回计算得到的 rules 列表
    return rules

# 反转间隔，使得遍历方向从右到左、从上到下
def _reverse_intervals(intervals):
    """Reverse intervals for traversal from right to left and from top to bottom. """
    return [ ((b, a), indices, f) for (a, b), indices, f in reversed(intervals) ]

# 计算输入多项式的绕数，即其根的数量
def _winding_number(T, field):
    """Compute the winding number of the input polynomial, i.e. the number of roots. """
    return int(sum(field(*_values[t][i]) for t, i in T) / field(2))

# 使用 Collins-Krandick 算法在 [u + v*I, s + t*I] 矩形中计算复数根的数量
def dup_count_complex_roots(f, K, inf=None, sup=None, exclude=None):
    """Count all roots in [u + v*I, s + t*I] rectangle using Collins-Krandick algorithm. """
    # 如果 K 不是整数环也不是有理数环，则抛出 DomainError
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("complex root counting is not supported over %s" % K)

    # 如果 K 是整数环
    if K.is_ZZ:
        R, F = K, K.get_field()
    else:
        R, F = K.get_ring(), K

    # 将多项式 f 转换为域 F 上的多项式
    f = dup_convert(f, K, F)

    # 如果 inf 或 sup 有任何一个为 None，则分别计算多项式的度和其最高系数绝对值的两倍，并取其最大值
    if inf is None or sup is None:
        _, lc = dup_degree(f), abs(dup_LC(f, F))
        B = 2*max(F.quo(abs(c), lc) for c in f)
    # 如果输入参数 inf 为 None，则将 u 和 v 初始化为 -B
    if inf is None:
        (u, v) = (-B, -B)
    else:
        # 否则将 u 和 v 设置为 inf 参数的值
        (u, v) = inf

    # 如果输入参数 sup 为 None，则将 s 和 t 初始化为 +B
    if sup is None:
        (s, t) = (+B, +B)
    else:
        # 否则将 s 和 t 设置为 sup 参数的值
        (s, t) = sup

    # 将函数 f 拆分为实部和虚部，分别存储在 f1 和 f2 中
    f1, f2 = dup_real_imag(f, F)

    # 在 v 处对 f1 和 f2 进行求值，考虑一阶偏导数，结果存储在 f1L1F 和 f2L1F 中
    f1L1F = dmp_eval_in(f1, v, 1, 1, F)
    f2L1F = dmp_eval_in(f2, v, 1, 1, F)

    # 清除 f1L1F 和 f2L1F 的分母，结果存储在 f1L1R 和 f2L1R 中
    _, f1L1R = dup_clear_denoms(f1L1F, F, R, convert=True)
    _, f2L1R = dup_clear_denoms(f2L1F, F, R, convert=True)

    # 在 s 处对 f1 和 f2 进行求值，考虑常数项，结果存储在 f1L2F 和 f2L2F 中
    f1L2F = dmp_eval_in(f1, s, 0, 1, F)
    f2L2F = dmp_eval_in(f2, s, 0, 1, F)

    # 清除 f1L2F 和 f2L2F 的分母，结果存储在 f1L2R 和 f2L2R 中
    _, f1L2R = dup_clear_denoms(f1L2F, F, R, convert=True)
    _, f2L2R = dup_clear_denoms(f2L2F, F, R, convert=True)

    # 在 t 处对 f1 和 f2 进行求值，考虑一阶偏导数，结果存储在 f1L3F 和 f2L3F 中
    f1L3F = dmp_eval_in(f1, t, 1, 1, F)
    f2L3F = dmp_eval_in(f2, t, 1, 1, F)

    # 清除 f1L3F 和 f2L3F 的分母，结果存储在 f1L3R 和 f2L3R 中
    _, f1L3R = dup_clear_denoms(f1L3F, F, R, convert=True)
    _, f2L3R = dup_clear_denoms(f2L3F, F, R, convert=True)

    # 在 u 处对 f1 和 f2 进行求值，考虑常数项，结果存储在 f1L4F 和 f2L4F 中
    f1L4F = dmp_eval_in(f1, u, 0, 1, F)
    f2L4F = dmp_eval_in(f2, u, 0, 1, F)

    # 清除 f1L4F 和 f2L4F 的分母，结果存储在 f1L4R 和 f2L4R 中
    _, f1L4R = dup_clear_denoms(f1L4F, F, R, convert=True)
    _, f2L4R = dup_clear_denoms(f2L4F, F, R, convert=True)

    # 将计算得到的结果组合成四个子集合 S_L1, S_L2, S_L3, S_L4
    S_L1 = [f1L1R, f2L1R]
    S_L2 = [f1L2R, f2L2R]
    S_L3 = [f1L3R, f2L3R]
    S_L4 = [f1L4R, f2L4R]

    # 分别对四个子集合进行实根的隔离，结果存储在 I_L1, I_L2, I_L3, I_L4 中
    I_L1 = dup_isolate_real_roots_list(S_L1, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, R, inf=v, sup=t, fast=True, basis=True, strict=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, R, inf=v, sup=t, fast=True, basis=True, strict=True)

    # 翻转 I_L3 和 I_L4 中的区间顺序
    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    # 将隔离得到的区间转换为象限，结果存储在 Q_L1, Q_L2, Q_L3, Q_L4 中
    Q_L1 = _intervals_to_quadrants(I_L1, f1L1F, f2L1F, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2F, f2L2F, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3F, f2L3F, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4F, f2L4F, t, v, F)

    # 根据象限信息进行遍历，排除 exclude 中的象限，结果存储在 T 中
    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=exclude)

    # 计算最终的绕数，结果返回
    return _winding_number(T, F)
# 定义垂直二分法的步骤，用于Collins-Krandick根隔离算法中的垂直二分步骤
def _vertical_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Vertical bisection step in Collins-Krandick root isolation algorithm. """
    # 解包参数a和b为(u, v)和(s, t)
    (u, v), (s, t) = a, b

    # 解包参数I为四个元组I_L1, I_L2, I_L3, I_L4
    I_L1, I_L2, I_L3, I_L4 = I
    # 解包参数Q为四个元组Q_L1, Q_L2, Q_L3, Q_L4
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    # 解包参数F1为四个元组f1L1F, f1L2F, f1L3F, f1L4F
    f1L1F, f1L2F, f1L3F, f1L4F = F1
    # 解包参数F2为四个元组f2L1F, f2L2F, f2L3F, f2L4F
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    # 计算垂直中点x
    x = (u + s) / 2

    # 在点x处评估多项式f1和f2的值
    f1V = dmp_eval_in(f1, x, 0, 1, F)
    f2V = dmp_eval_in(f2, x, 0, 1, F)

    # 在区间[v, t]上隔离多项式f1V和f2V的实根，返回一个区间列表I_V
    I_V = dup_isolate_real_roots_list([f1V, f2V], F, inf=v, sup=t, fast=True, strict=True, basis=True)

    # 根据x将I_L1分成左右两部分
    I_L1_L, I_L1_R = [], []
    for I in I_L1:
        (a, b), indices, h = I

        if a == b:
            if a == x:
                I_L1_L.append(I)
                I_L1_R.append(I)
            elif a < x:
                I_L1_L.append(I)
            else:
                I_L1_R.append(I)
        else:
            if b <= x:
                I_L1_L.append(I)
            elif a >= x:
                I_L1_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)
                if b <= x:
                    I_L1_L.append(((a, b), indices, h))
                if a >= x:
                    I_L1_R.append(((a, b), indices, h))

    # 根据x将I_L3分成左右两部分
    I_L3_L, I_L3_R = [], []
    for I in I_L3:
        (b, a), indices, h = I

        if a == b:
            if a == x:
                I_L3_L.append(I)
                I_L3_R.append(I)
            elif a < x:
                I_L3_L.append(I)
            else:
                I_L3_R.append(I)
        else:
            if b <= x:
                I_L3_L.append(I)
            elif a >= x:
                I_L3_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)
                if b <= x:
                    I_L3_L.append(((b, a), indices, h))
                if a >= x:
                    I_L3_R.append(((b, a), indices, h))

    # 根据左侧子区间和函数值f1L1F, f2L1F, u, x生成左侧四象限Q_L1_L
    Q_L1_L = _intervals_to_quadrants(I_L1_L, f1L1F, f2L1F, u, x, F)
    # 根据左侧子区间和函数值f1V, f2V, v, t生成左侧四象限Q_L2_L
    Q_L2_L = _intervals_to_quadrants(I_L2_L, f1V, f2V, v, t, F)
    # 根据左侧子区间和函数值f1L3F, f2L3F, x, u生成左侧四象限Q_L3_L
    Q_L3_L = _intervals_to_quadrants(I_L3_L, f1L3F, f2L3F, x, u, F)
    # Q_L4_L保持不变
    Q_L4_L = Q_L4

    # 根据右侧子区间和函数值f1L1F, f2L1F, x, s生成右侧四象限Q_L1_R
    Q_L1_R = _intervals_to_quadrants(I_L1_R, f1L1F, f2L1F, x, s, F)
    # Q_L2_R保持不变
    Q_L2_R = Q_L2
    # 根据右侧子区间和函数值f1L3F, f2L3F, s, x生成右侧四象限Q_L3_R
    Q_L3_R = _intervals_to_quadrants(I_L3_R, f1L3F, f2L3F, s, x, F)
    # 根据右侧子区间和函数值f1V, f2V, t, v生成右侧四象限Q_L4_R
    Q_L4_R = _intervals_to_quadrants(I_L4_R, f1V, f2V, t, v, F)

    # 根据左侧四象限Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L进行象限遍历，排除左侧
    T_L = _traverse_quadrants(Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L, exclude=True)
    # 根据右侧四象限Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R进行象限遍历，排除右侧
    T_R = _traverse_quadrants(Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R, exclude=True)

    # 计算左侧和右侧象限遍历得到的绕行数
    N_L = _winding_number(T_L, F)
    N_R = _winding_number(T_R, F)

    # 左侧区间列表和四象限列表
    I_L = (I_L1_L, I_L2_L, I_L3_L, I_L4_L)
    Q_L = (Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L)

    # 右侧区间列表和四象限列表
    I_R = (I_L1_R, I_L2_R, I_L3_R, I_L4_R)
    Q_R = (Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R)

    # 左侧函数列表F1_L和F2_L
    F1_L = (f1L1F, f1V, f1L3F, f1L4F)
    F2_L = (f2L1F, f2V, f2L3F, f2L4F)

    # 右侧函数列表F1_R和F2_R
    F1_R = (f1L1F, f1L2F, f1L3F, f1V)
    F2_R = (f2L1F, f2L2F, f2L3F, f2V)

    # 更新a和b为(u, v)和(x, t)
    a, b = (u, v), (x, t)
    # 将元组 (x, v) 赋值给变量 c 和 d
    c, d = (x, v), (s, t)

    # 创建元组 D_L 包含 N_L, a, b, I_L, Q_L, F1_L, F2_L，并赋值给变量 D_L
    D_L = (N_L, a, b, I_L, Q_L, F1_L, F2_L)
    
    # 创建元组 D_R 包含 N_R, c, d, I_R, Q_R, F1_R, F2_R，并赋值给变量 D_R
    D_R = (N_R, c, d, I_R, Q_R, F1_R, F2_R)

    # 返回元组 D_L 和 D_R
    return D_L, D_R
def _horizontal_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Horizontal bisection step in Collins-Krandick root isolation algorithm. """
    # 解构元组 a, b，并分别赋值给 (u, v), (s, t)
    (u, v), (s, t) = a, b

    # 解构元组 I，并分别赋值给 I_L1, I_L2, I_L3, I_L4
    I_L1, I_L2, I_L3, I_L4 = I
    # 解构元组 Q，并分别赋值给 Q_L1, Q_L2, Q_L3, Q_L4
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    # 解构元组 F1，并分别赋值给 f1L1F, f1L2F, f1L3F, f1L4F
    f1L1F, f1L2F, f1L3F, f1L4F = F1
    # 解构元组 F2，并分别赋值给 f2L1F, f2L2F, f2L3F, f2L4F
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    # 计算中点 y
    y = (v + t) / 2

    # 在点 y 处计算 f1 和 f2 的值
    f1H = dmp_eval_in(f1, y, 1, 1, F)
    f2H = dmp_eval_in(f2, y, 1, 1, F)

    # 使用 f1H 和 f2H 列表计算实根的区间列表 I_H
    I_H = dup_isolate_real_roots_list([f1H, f2H], F, inf=u, sup=s, fast=True, strict=True, basis=True)

    # 根据 y 对 I_L1 进行分割，更新区间的下界和上界
    I_L1_B, I_L1_U = I_L1, I_H

    # 初始化空列表用于存储分割后的区间
    I_L2_B, I_L2_U = [], []
    I_L3_B, I_L3_U = _reverse_intervals(I_H), I_L3
    I_L4_B, I_L4_U = [], []

    # 遍历 I_L2 中的区间 I，根据 y 更新区间的下界和上界
    for I in I_L2:
        (a, b), indices, h = I

        if a == b:
            if a == y:
                I_L2_B.append(I)
                I_L2_U.append(I)
            elif a < y:
                I_L2_B.append(I)
            else:
                I_L2_U.append(I)
        else:
            if b <= y:
                I_L2_B.append(I)
            elif a >= y:
                I_L2_U.append(I)
            else:
                # 对于不在 y 处的区间，精细化实根的计算
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L2_B.append(((a, b), indices, h))
                if a >= y:
                    I_L2_U.append(((a, b), indices, h))

    # 类似地，遍历 I_L4 中的区间 I，根据 y 更新区间的下界和上界
    for I in I_L4:
        (b, a), indices, h = I

        if a == b:
            if a == y:
                I_L4_B.append(I)
                I_L4_U.append(I)
            elif a < y:
                I_L4_B.append(I)
            else:
                I_L4_U.append(I)
        else:
            if b <= y:
                I_L4_B.append(I)
            elif a >= y:
                I_L4_U.append(I)
            else:
                # 对于不在 y 处的区间，精细化实根的计算
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L4_B.append(((b, a), indices, h))
                if a >= y:
                    I_L4_U.append(((b, a), indices, h))

    # 初始化 Q_L1_B 为 Q_L1
    Q_L1_B = Q_L1
    # 将 I_L2_B 转换为相应的象限列表 Q_L2_B
    Q_L2_B = _intervals_to_quadrants(I_L2_B, f1L2F, f2L2F, v, y, F)
    # 将 I_L3_B 转换为相应的象限列表 Q_L3_B
    Q_L3_B = _intervals_to_quadrants(I_L3_B, f1H, f2H, s, u, F)
    # 将 I_L4_B 转换为相应的象限列表 Q_L4_B
    Q_L4_B = _intervals_to_quadrants(I_L4_B, f1L4F, f2L4F, y, v, F)

    # 将 I_L1_U 转换为相应的象限列表 Q_L1_U
    Q_L1_U = _intervals_to_quadrants(I_L1_U, f1H, f2H, u, s, F)
    # 将 I_L2_U 转换为相应的象限列表 Q_L2_U
    Q_L2_U = _intervals_to_quadrants(I_L2_U, f1L2F, f2L2F, y, t, F)
    # 初始化 Q_L3_U 为 Q_L3
    Q_L3_U = Q_L3
    # 将 I_L4_U 转换为相应的象限列表 Q_L4_U
    Q_L4_U = _intervals_to_quadrants(I_L4_U, f1L4F, f2L4F, t, y, F)

    # 使用 _traverse_quadrants 函数遍历象限列表，排除特定象限后返回结果 T_B
    T_B = _traverse_quadrants(Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B, exclude=True)
    # 使用 _traverse_quadrants 函数遍历象限列表，排除特定象限后返回结果 T_U
    T_U = _traverse_quadrants(Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U, exclude=True)

    # 计算 T_B 的绕行数 N_B
    N_B = _winding_number(T_B, F)
    # 计算 T_U 的绕行数 N_U
    N_U = _winding_number(T_U, F)

    # 组装并返回结果元组
    I_B = (I_L1_B, I_L2_B, I_L3_B, I_L4_B)
    Q_B = (Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B)

    I_U = (I_L1_U, I_L2_U, I_L3_U, I_L4_U)
    Q_U = (Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U)

    F1_B = (f1L1F, f1L2F, f1H, f1L4F)
    F2_B = (f2L1F, f2L2F, f2H, f2L4F)

    F1_U = (f1H, f1L2F, f1L3F, f1L4F)
    F2_U = (f2H, f2L2F, f2L3F, f2L4F)
    # 将元组 (u, v) 和 (s, y) 分别赋值给变量 a, b
    a, b = (u, v), (s, y)
    
    # 将元组 (u, y) 和 (s, t) 分别赋值给变量 c, d
    c, d = (u, y), (s, t)
    
    # 创建元组 D_B，包含 N_B, a, b, I_B, Q_B, F1_B, F2_B
    D_B = (N_B, a, b, I_B, Q_B, F1_B, F2_B)
    
    # 创建元组 D_U，包含 N_U, c, d, I_U, Q_U, F1_U, F2_U
    D_U = (N_U, c, d, I_U, Q_U, F1_U, F2_U)
    
    # 返回元组 D_B 和 D_U
    return D_B, D_U
def dup_isolate_complex_roots_sqf(f, K, eps=None, inf=None, sup=None, blackbox=False):
    """Isolate complex roots of a square-free polynomial using Collins-Krandick algorithm. """
    # 检查多项式系数环是否是整数环或有理数环，否则抛出域错误
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of complex roots is not supported over %s" % K)

    # 如果多项式的次数小于等于0，直接返回空列表
    if dup_degree(f) <= 0:
        return []

    # 将多项式转换为指定的域 F
    if K.is_ZZ:
        F = K.get_field()
    else:
        F = K

    f = dup_convert(f, K, F)

    # 计算多项式的首项系数的绝对值
    lc = abs(dup_LC(f, F))
    # 计算 B 的值
    B = 2 * max(F.quo(abs(c), lc) for c in f)

    # 初始化矩形区域的边界
    (u, v), (s, t) = (-B, F.zero), (B, B)

    # 根据参数 inf 和 sup 更新矩形的水平边界
    if inf is not None:
        u = inf

    if sup is not None:
        s = sup

    # 检查矩形的有效性，如果不是有效的复数隔离矩形则抛出值错误
    if v < 0 or t <= v or s <= u:
        raise ValueError("not a valid complex isolation rectangle")

    # 将多项式分成实部和虚部，并在四个边界点计算对应的值
    f1, f2 = dup_real_imag(f, F)

    f1L1 = dmp_eval_in(f1, v, 1, 1, F)
    f2L1 = dmp_eval_in(f2, v, 1, 1, F)

    f1L2 = dmp_eval_in(f1, s, 0, 1, F)
    f2L2 = dmp_eval_in(f2, s, 0, 1, F)

    f1L3 = dmp_eval_in(f1, t, 1, 1, F)
    f2L3 = dmp_eval_in(f2, t, 1, 1, F)

    f1L4 = dmp_eval_in(f1, u, 0, 1, F)
    f2L4 = dmp_eval_in(f2, u, 0, 1, F)

    # 构建四个区间列表
    S_L1 = [f1L1, f2L1]
    S_L2 = [f1L2, f2L2]
    S_L3 = [f1L3, f2L3]
    S_L4 = [f1L4, f2L4]

    # 在四个区间上分别隔离实数根
    I_L1 = dup_isolate_real_roots_list(S_L1, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, F, inf=v, sup=t, fast=True, strict=True, basis=True)

    # 对第三和第四个区间列表进行反转
    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    # 将区间列表转换为象限列表
    Q_L1 = _intervals_to_quadrants(I_L1, f1L1, f2L1, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2, f2L2, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3, f2L3, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4, f2L4, t, v, F)

    # 遍历四个象限并获取遍历路径
    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4)
    # 计算遍历路径的绕行数
    N = _winding_number(T, F)

    # 如果绕行数为零，则返回空列表
    if not N:
        return []

    # 组合得到最终的结果
    I = (I_L1, I_L2, I_L3, I_L4)
    Q = (Q_L1, Q_L2, Q_L3, Q_L4)

    F1 = (f1L1, f1L2, f1L3, f1L4)
    F2 = (f2L1, f2L2, f2L3, f2L4)

    rectangles, roots = [(N, (u, v), (s, t), I, Q, F1, F2)], []
    # 当还有矩形未处理时循环执行以下操作
    while rectangles:
        # 从未处理的矩形中选择一个矩形进行深度优先搜索，并获取其参数
        N, (u, v), (s, t), I, Q, F1, F2 = _depth_first_select(rectangles)

        # 判断矩形的宽度是否大于高度
        if s - u > t - v:
            # 对选定的矩形进行垂直方向的二分法分割
            D_L, D_R = _vertical_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            # 解析左右两个分割后的矩形数据
            N_L, a, b, I_L, Q_L, F1_L, F2_L = D_L
            N_R, c, d, I_R, Q_R, F1_R, F2_R = D_R

            # 如果左边的矩形数量大于等于1
            if N_L >= 1:
                # 如果左边只有一个小矩形并且满足小矩形的条件
                if N_L == 1 and _rectangle_small_p(a, b, eps):
                    # 将找到的根添加到根列表中
                    roots.append(ComplexInterval(a, b, I_L, Q_L, F1_L, F2_L, f1, f2, F))
                else:
                    # 将左边的矩形加入待处理矩形列表中
                    rectangles.append(D_L)

            # 如果右边的矩形数量大于等于1
            if N_R >= 1:
                # 如果右边只有一个小矩形并且满足小矩形的条件
                if N_R == 1 and _rectangle_small_p(c, d, eps):
                    # 将找到的根添加到根列表中
                    roots.append(ComplexInterval(c, d, I_R, Q_R, F1_R, F2_R, f1, f2, F))
                else:
                    # 将右边的矩形加入待处理矩形列表中
                    rectangles.append(D_R)
        else:
            # 对选定的矩形进行水平方向的二分法分割
            D_B, D_U = _horizontal_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            # 解析下方和上方两个分割后的矩形数据
            N_B, a, b, I_B, Q_B, F1_B, F2_B = D_B
            N_U, c, d, I_U, Q_U, F1_U, F2_U = D_U

            # 如果下方的矩形数量大于等于1
            if N_B >= 1:
                # 如果下方只有一个小矩形并且满足小矩形的条件
                if N_B == 1 and _rectangle_small_p(a, b, eps):
                    # 将找到的根添加到根列表中
                    roots.append(ComplexInterval(
                        a, b, I_B, Q_B, F1_B, F2_B, f1, f2, F))
                else:
                    # 将下方的矩形加入待处理矩形列表中
                    rectangles.append(D_B)

            # 如果上方的矩形数量大于等于1
            if N_U >= 1:
                # 如果上方只有一个小矩形并且满足小矩形的条件
                if N_U == 1 and _rectangle_small_p(c, d, eps):
                    # 将找到的根添加到根列表中
                    roots.append(ComplexInterval(
                        c, d, I_U, Q_U, F1_U, F2_U, f1, f2, F))
                else:
                    # 将上方的矩形加入待处理矩形列表中
                    rectangles.append(D_U)

    # 对找到的所有根进行排序，按照指定的排序规则
    _roots, roots = sorted(roots, key=lambda r: (r.ax, r.ay)), []

    # 将每个根的共轭和原根添加到最终的根列表中
    for root in _roots:
        roots.extend([root.conjugate(), root])

    # 如果需要黑盒子模式，则直接返回根列表
    if blackbox:
        return roots
    else:
        # 否则返回根的元组表示组成的列表
        return [ r.as_tuple() for r in roots ]
# 定义函数 dup_isolate_all_roots_sqf，用于分离多项式 f 的实数和复数根，其中 f 是方阵。
def dup_isolate_all_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real and complex roots of a square-free polynomial ``f``. """
    # 调用函数 dup_isolate_real_roots_sqf 分离 f 的实数根
    return (
        dup_isolate_real_roots_sqf( f, K, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox),
        # 调用函数 dup_isolate_complex_roots_sqf 分离 f 的复数根
        dup_isolate_complex_roots_sqf(f, K, eps=eps, inf=inf, sup=sup, blackbox=blackbox))

# 定义函数 dup_isolate_all_roots，用于分离非方阵多项式 f 的实数和复数根。
def dup_isolate_all_roots(f, K, eps=None, inf=None, sup=None, fast=False):
    """Isolate real and complex roots of a non-square-free polynomial ``f``. """
    # 如果 K 不是整数环也不是有理数环，则引发 DomainError 异常
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of real and complex roots is not supported over %s" % K)

    # 调用函数 dup_sqf_list 获取 f 的平方自由因式分解结果
    _, factors = dup_sqf_list(f, K)

    # 如果 f 只有一个因式
    if len(factors) == 1:
        # 解构因式列表中的唯一元素
        ((f, k),) = factors

        # 调用函数 dup_isolate_all_roots_sqf 分离 f 的实数和复数根
        real_part, complex_part = dup_isolate_all_roots_sqf(
            f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        # 转换实数根和复数根的表示方式
        real_part = [ ((a, b), k) for (a, b) in real_part ]
        complex_part = [ ((a, b), k) for (a, b) in complex_part ]

        # 返回转换后的实数根和复数根列表
        return real_part, complex_part
    else:
        # 如果 f 有多个因式，则抛出 NotImplementedError 异常
        raise NotImplementedError( "only trivial square-free polynomials are supported")

# 定义类 RealInterval，表示实数隔离区间的全面描述
class RealInterval:
    """A fully qualified representation of a real isolation interval. """

    # 初始化方法，用于创建新的实数隔离区间，并使用完整信息初始化
    def __init__(self, data, f, dom):
        """Initialize new real interval with complete information. """
        # 如果 data 中包含两个元素
        if len(data) == 2:
            s, t = data

            self.neg = False

            # 如果 s 小于 0
            if s < 0:
                # 如果 t 小于等于 0，则反转 f 并调整 s 和 t 的值
                if t <= 0:
                    f, s, t, self.neg = dup_mirror(f, dom), -t, -s, True
                else:
                    # 否则，引发 ValueError 异常，指示无法在区间 (s, t) 中细化实数根
                    raise ValueError("Cannot refine a real root in (%s, %s)" % (s, t))

            # 根据区间 (s, t) 和 dom 的域属性，计算 Mobius 变换的系数
            a, b, c, d = _mobius_from_interval((s, t), dom.get_field())

            # 对多项式 f 进行 Mobius 变换
            f = dup_transform(f, dup_strip([a, b]),
                                 dup_strip([c, d]), dom)

            self.mobius = a, b, c, d
        else:
            # 否则，直接使用 data 的前面部分作为 Mobius 变换的系数，最后一个元素作为 neg 属性
            self.mobius = data[:-1]
            self.neg = data[-1]

        self.f, self.dom = f, dom

    # 返回 func 属性，表示该实例的类类型
    @property
    def func(self):
        return RealInterval

    # 返回 args 属性，表示实例的构造参数
    @property
    def args(self):
        i = self
        return (i.mobius + (i.neg,), i.f, i.dom)

    # 实现相等性比较方法，检查两个 RealInterval 实例是否相等
    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    # 返回 a 属性，表示实数隔离区间的左端位置
    @property
    def a(self):
        """Return the position of the left end. """
        # 获取域的属性
        field = self.dom.get_field()
        a, b, c, d = self.mobius

        # 如果非负变量为真
        if not self.neg:
            # 如果 a*d 小于 b*c，则返回 field(a, c)，否则返回 field(b, d)
            if a*d < b*c:
                return field(a, c)
            return field(b, d)
        else:
            # 否则，如果 a*d 大于 b*c，则返回 -field(a, c)，否则返回 -field(b, d)
            if a*d > b*c:
                return -field(a, c)
            return -field(b, d)

    # 返回 b 属性，表示实数隔离区间的右端位置
    @property
    def b(self):
        """Return the position of the right end. """
        # 保存 neg 属性的当前状态
        was = self.neg
        # 将 neg 属性设为其相反值
        self.neg = not was
        # 计算 -a 的值并保存为 rv
        rv = -self.a
        # 恢复 neg 属性原来的状态
        self.neg = was
        return rv

    # 返回 dx 属性，表示实数隔离区间的宽度
    @property
    def dx(self):
        """Return width of the real isolating interval. """
        # 返回 b 减去 a 的结果
        return self.b - self.a
    # 返回实隔离区间的中心点
    @property
    def center(self):
        """Return the center of the real isolating interval. """
        return (self.a + self.b)/2

    # 返回在端点中出现的最大分母
    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.a.denominator, self.b.denominator)

    # 返回实隔离区间的元组表示形式
    def as_tuple(self):
        """Return tuple representation of real isolating interval. """
        return (self.a, self.b)

    # 返回对象的字符串表示形式
    def __repr__(self):
        return "(%s, %s)" % (self.a, self.b)

    # 判断复数是否属于实区间
    def __contains__(self, item):
        """
        Say whether a complex number belongs to this real interval.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return im == 0 and self.a <= re <= self.b

    # 返回True，如果两个隔离区间不相交
    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return (self.b < other.a or other.b < self.a)
        assert isinstance(other, ComplexInterval)
        return (self.b < other.ax or other.bx < self.a
            or other.ay*other.by > 0)

    # 内部的一步实根细化过程
    def _inner_refine(self):
        """Internal one step real root refinement procedure. """
        if self.mobius is None:
            return self

        f, mobius = dup_inner_refine_real_root(
            self.f, self.mobius, self.dom, steps=1, mobius=True)

        return RealInterval(mobius + (self.neg,), f, self.dom)

    # 对隔离区间进行细化，直到与另一个区间不相交
    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = expr._inner_refine(), other._inner_refine()

        return expr, other

    # 对隔离区间进行细化，直到其大小足够小
    def refine_size(self, dx):
        """Refine an isolating interval until it is of sufficiently small size. """
        expr = self
        while not (expr.dx < dx):
            expr = expr._inner_refine()

        return expr

    # 执行多步实根细化算法
    def refine_step(self, steps=1):
        """Perform several steps of real root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()

        return expr

    # 执行一步实根细化算法
    def refine(self):
        """Perform one step of real root refinement algorithm. """
        return self._inner_refine()
class ComplexInterval:
    """A fully qualified representation of a complex isolation interval.
    The printed form is shown as (ax, bx) x (ay, by) where (ax, ay)
    and (bx, by) are the coordinates of the southwest and northeast
    corners of the interval's rectangle, respectively.

    Examples
    ========

    >>> from sympy import CRootOf, S
    >>> from sympy.abc import x
    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> root = CRootOf(x**10 - 2*x + 3, 9)
    >>> i = root._get_interval(); i
    (3/64, 3/32) x (9/8, 75/64)

    The real part of the root lies within the range [0, 3/4] while
    the imaginary part lies within the range [9/8, 3/2]:

    >>> root.n(3)
    0.0766 + 1.14*I

    The width of the ranges in the x and y directions on the complex
    plane are:

    >>> i.dx, i.dy
    (3/64, 3/64)

    The center of the range is

    >>> i.center
    (9/128, 147/128)

    The northeast coordinate of the rectangle bounding the root in the
    complex plane is given by attribute b and the x and y components
    are accessed by bx and by:

    >>> i.b, i.bx, i.by
    ((3/32, 75/64), 3/32, 75/64)

    The southwest coordinate is similarly given by i.a

    >>> i.a, i.ax, i.ay
    ((3/64, 9/8), 3/64, 9/8)

    Although the interval prints to show only the real and imaginary
    range of the root, all the information of the underlying root
    is contained as properties of the interval.

    For example, an interval with a nonpositive imaginary range is
    considered to be the conjugate. Since the y values of y are in the
    range [0, 1/4] it is not the conjugate:

    >>> i.conj
    False

    The conjugate's interval is

    >>> ic = i.conjugate(); ic
    (3/64, 3/32) x (-75/64, -9/8)

        NOTE: the values printed still represent the x and y range
        in which the root -- conjugate, in this case -- is located,
        but the underlying a and b values of a root and its conjugate
        are the same:

        >>> assert i.a == ic.a and i.b == ic.b

        What changes are the reported coordinates of the bounding rectangle:

        >>> (i.ax, i.ay), (i.bx, i.by)
        ((3/64, 9/8), (3/32, 75/64))
        >>> (ic.ax, ic.ay), (ic.bx, ic.by)
        ((3/64, -75/64), (3/32, -9/8))

    The interval can be refined once:

    >>> i  # for reference, this is the current interval
    (3/64, 3/32) x (9/8, 75/64)

    >>> i.refine()
    (3/64, 3/32) x (9/8, 147/128)

    Several refinement steps can be taken:

    >>> i.refine_step(2)  # 2 steps
    (9/128, 3/32) x (9/8, 147/128)

    It is also possible to refine to a given tolerance:

    >>> tol = min(i.dx, i.dy)/2
    >>> i.refine_size(tol)
    (9/128, 21/256) x (9/8, 291/256)

    A disjoint interval is one whose bounding rectangle does not
    overlap with another. An interval, necessarily, is not disjoint with
    itself, but any interval is disjoint with a conjugate since the
    conjugate rectangle will always be in the lower half of the complex
    """

    def __init__(self, a, b):
        """Initialize the ComplexInterval with southwest (a) and northeast (b) coordinates."""
        self.a = a  # southwest coordinate (ax, ay)
        self.b = b  # northeast coordinate (bx, by)

    @property
    def ax(self):
        """Get the x-coordinate of the southwest (a) corner."""
        return self.a[0]

    @property
    def ay(self):
        """Get the y-coordinate of the southwest (a) corner."""
        return self.a[1]

    @property
    def bx(self):
        """Get the x-coordinate of the northeast (b) corner."""
        return self.b[0]

    @property
    def by(self):
        """Get the y-coordinate of the northeast (b) corner."""
        return self.b[1]

    @property
    def dx(self):
        """Calculate the width of the interval in the x direction."""
        return self.bx - self.ax

    @property
    def dy(self):
        """Calculate the height of the interval in the y direction."""
        return self.by - self.ay

    @property
    def center(self):
        """Calculate the center of the interval."""
        cx = (self.ax + self.bx) / 2
        cy = (self.ay + self.by) / 2
        return (cx, cy)

    def conjugate(self):
        """Return the conjugate interval (flipping the y-coordinates)."""
        return ComplexInterval((self.ax, -self.by), (self.bx, -self.ay))

    def refine(self):
        """Refine the interval once, improving precision."""
        return ComplexInterval((self.ax, self.ax + self.dx/2), (self.ay, self.ay + self.dy/2))

    def refine_step(self, steps):
        """Refine the interval by the given number of steps."""
        for _ in range(steps):
            self = self.refine()
        return self

    def refine_size(self, tol):
        """Refine the interval to a given tolerance."""
        dx_new = min(tol, self.dx)
        dy_new = min(tol, self.dy)
        return ComplexInterval((self.ax, self.ax + dx_new), (self.ay, self.ay + dy_new))

    @property
    def conj(self):
        """Check if the interval is a conjugate."""
        return self.ay <= 0 and self.by <= 0
    def __init__(self, a, b, I, Q, F1, F2, f1, f2, dom, conj=False):
        """Initialize new complex interval with complete information. """
        # 初始化复数区间对象，使用给定的参数
        # a 和 b 是复数区间的西南角和东北角坐标
        # 对于非共轭根（具有正虚部的根），a 和 b 分别表示（ax, ay）和（bx, by）
        # 当处理共轭根时，a 和 b 的值仍然为非负，但 ay 和 by 的符号相反
        self.a, self.b = a, b
        # I 和 Q 是复数区间的额外信息
        self.I, self.Q = I, Q

        # f1, F1, f2, F2 是额外的函数信息
        self.f1, self.F1 = f1, F1
        self.f2, self.F2 = f2, F2

        # dom 是复数区间的定义域
        self.dom = dom
        # conj 表示是否为共轭根，默认为 False
        self.conj = conj

    @property
    def func(self):
        """Return the function type of the complex interval. """
        return ComplexInterval

    @property
    def args(self):
        """Return a tuple of arguments used to initialize the complex interval. """
        i = self
        return (i.a, i.b, i.I, i.Q, i.F1, i.F2, i.f1, i.f2, i.dom, i.conj)

    def __eq__(self, other):
        """Check if two complex intervals are equal. """
        # 检查两个复数区间是否相等
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def ax(self):
        """Return ``x`` coordinate of south-western corner. """
        # 返回复数区间西南角的 x 坐标
        return self.a[0]

    @property
    def ay(self):
        """Return ``y`` coordinate of south-western corner. """
        # 返回复数区间西南角的 y 坐标
        if not self.conj:
            return self.a[1]
        else:
            return -self.b[1]

    @property
    def bx(self):
        """Return ``x`` coordinate of north-eastern corner. """
        # 返回复数区间东北角的 x 坐标
        return self.b[0]

    @property
    def by(self):
        """Return ``y`` coordinate of north-eastern corner. """
        # 返回复数区间东北角的 y 坐标
        if not self.conj:
            return self.b[1]
        else:
            return -self.a[1]

    @property
    def dx(self):
        """Return width of the complex isolating interval. """
        # 返回复数隔离区间的宽度
        return self.b[0] - self.a[0]

    @property
    def dy(self):
        """Return height of the complex isolating interval. """
        # 返回复数隔离区间的高度
        return self.b[1] - self.a[1]
    def center(self):
        """Return the center of the complex isolating interval. """
        # 计算复数隔离区间的中心点坐标
        return ((self.ax + self.bx)/2, (self.ay + self.by)/2)

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        # 返回在任一端点中出现的最大分母
        return max(self.ax.denominator, self.bx.denominator,
                   self.ay.denominator, self.by.denominator)

    def as_tuple(self):
        """Return tuple representation of the complex isolating
        interval's SW and NE corners, respectively. """
        # 返回复数隔离区间的 SW 和 NE 角的元组表示
        return ((self.ax, self.ay), (self.bx, self.by))

    def __repr__(self):
        # 返回复数隔离区间的字符串表示形式
        return "(%s, %s) x (%s, %s)" % (self.ax, self.bx, self.ay, self.by)

    def conjugate(self):
        """This complex interval really is located in lower half-plane. """
        # 返回当前复数区间在下半平面的共轭
        return ComplexInterval(self.a, self.b, self.I, self.Q,
            self.F1, self.F2, self.f1, self.f2, self.dom, conj=True)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this complex rectangular
        region.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        # 判断复数是否属于此复数矩形区域
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return self.ax <= re <= self.bx and self.ay <= im <= self.by

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        # 如果两个隔离区间不相交，则返回 True
        if isinstance(other, RealInterval):
            return other.is_disjoint(self)
        if self.conj != other.conj:  # above and below real axis
            return True
        re_distinct = (self.bx < other.ax or other.bx < self.ax)
        if re_distinct:
            return True
        im_distinct = (self.by < other.ay or other.by < self.ay)
        return im_distinct

    def _inner_refine(self):
        """Internal one step complex root refinement procedure. """
        # 内部复数根细化过程的一步
        (u, v), (s, t) = self.a, self.b

        I, Q = self.I, self.Q

        f1, F1 = self.f1, self.F1
        f2, F2 = self.f2, self.F2

        dom = self.dom

        if s - u > t - v:
            D_L, D_R = _vertical_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_L[0] == 1:
                _, a, b, I, Q, F1, F2 = D_L
            else:
                _, a, b, I, Q, F1, F2 = D_R
        else:
            D_B, D_U = _horizontal_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_B[0] == 1:
                _, a, b, I, Q, F1, F2 = D_B
            else:
                _, a, b, I, Q, F1, F2 = D_U

        return ComplexInterval(a, b, I, Q, F1, F2, f1, f2, dom, self.conj)
    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self  # 将当前对象赋值给变量expr，表示当前的隔离区间
        while not expr.is_disjoint(other):  # 循环直到当前区间与另一个区间不相交
            expr, other = expr._inner_refine(), other._inner_refine()  # 分别对当前区间和另一个区间进行内部细化

        return expr, other  # 返回经过细化后的两个区间对象

    def refine_size(self, dx, dy=None):
        """Refine an isolating interval until it is of sufficiently small size. """
        if dy is None:
            dy = dx  # 如果dy未指定，则设为dx的值
        expr = self  # 将当前对象赋值给变量expr，表示当前的隔离区间
        while not (expr.dx < dx and expr.dy < dy):  # 循环直到区间的宽度和高度均小于给定的dx和dy
            expr = expr._inner_refine()  # 对当前区间进行内部细化

        return expr  # 返回经过细化后的区间对象

    def refine_step(self, steps=1):
        """Perform several steps of complex root refinement algorithm. """
        expr = self  # 将当前对象赋值给变量expr，表示当前的复根细化对象
        for _ in range(steps):  # 执行给定步数的复根细化算法步骤
            expr = expr._inner_refine()  # 对当前复根对象进行内部细化

        return expr  # 返回经过指定步数细化后的复根对象

    def refine(self):
        """Perform one step of complex root refinement algorithm. """
        return self._inner_refine()  # 执行一步复根细化算法，并返回细化后的对象
```