# `D:\src\scipysrc\sympy\sympy\integrals\prde.py`

```
"""
Algorithms for solving Parametric Risch Differential Equations.

The methods used for solving Parametric Risch Differential Equations parallel
those for solving Risch Differential Equations.  See the outline in the
docstring of rde.py for more information.

The Parametric Risch Differential Equation problem is, given f, g1, ..., gm in
K(t), to determine if there exist y in K(t) and c1, ..., cm in Const(K) such
that Dy + f*y == Sum(ci*gi, (i, 1, m)), and to find such y and ci if they exist.

For the algorithms here G is a list of tuples of factions of the terms on the
right hand side of the equation (i.e., gi in k(t)), and Q is a list of terms on
the right hand side of the equation (i.e., qi in k[t]).  See the docstring of
each function for more information.
"""
import itertools
from functools import reduce

from sympy.core.intfunc import ilcm
from sympy.core import Dummy, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
    bound_degree)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    residue_reduce, splitfactor, residue_reduce_derivation, DecrementLevel,
    recognize_log_derivative)
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve

zeros = Matrix.zeros
eye = Matrix.eye


def prde_normal_denom(fa, fd, G, DE):
    """
    Parametric Risch Differential Equation - Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g1, ..., gm in k(t) with f weakly
    normalized with respect to t, return the tuple (a, b, G, h) such that
    a, h in k[t], b in k<t>, G = [g1, ..., gm] in k(t)^m, and for any solution
    c1, ..., cm in Const(k) and y in k(t) of Dy + f*y == Sum(ci*gi, (i, 1, m)),
    q == y*h in k<t> satisfies a*Dq + b*q == Sum(ci*Gi, (i, 1, m)).
    """
    # Split the given denominator fd into numerator dn and denominator ds
    dn, ds = splitfactor(fd, DE)
    # Unzip G into Gas (numerator) and Gds (denominator) lists
    Gas, Gds = list(zip(*G))
    # Compute the least common multiple of all elements in Gds
    gd = reduce(lambda i, j: i.lcm(j), Gds, Poly(1, DE.t))
    # Split the computed gd into numerator en and denominator es
    en, es = splitfactor(gd, DE)

    # Compute p as the gcd of dn and en
    p = dn.gcd(en)
    # Compute h as the gcd of en and its derivative with respect to DE.t
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    # Compute a as dn multiplied by h
    a = dn * h
    # Compute c as a multiplied by h
    c = a * h

    # Compute ba as a * fa minus dn times the derivation of h with respect to DE.t times fd
    ba = a * fa - dn * derivation(h, DE) * fd
    # Cancel ba with fd to get ba and bd (numerator and denominator)
    ba, bd = ba.cancel(fd, include=True)

    # For each pair (A, D) in G, cancel c * A with D to get the new list G
    G = [(c * A).cancel(D, include=True) for A, D in G]

    return (a, (ba, bd), G, h)


def real_imag(ba, bd, gen):
    """
    Helper function, to get the real and imaginary part of a rational function
    evaluated at sqrt(-1) without actually evaluating it at sqrt(-1).

    Explanation
    ===========

    Separates the even and odd power terms by checking the degree of terms wrt
    mod 4. Returns a tuple (ba[0], ba[1], bd) where ba[0] is real part
    of the numerator ba[1] is the imaginary part and bd is the denominator
    of the rational function.
    """
    # Convert bd and ba to dictionaries of their polynomial terms
    bd = bd.as_poly(gen).as_dict()
    ba = ba.as_poly(gen).as_dict()
    # Determine the real and imaginary parts of the rational function
    denom_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in bd.items()]
    # Return the real and imaginary parts along with the denominator bd
    return (ba[0], ba[1], bd)
    # 计算分子的虚部系数列表，根据键的第一个元素的模 4 的不同值来决定系数的正负
    denom_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in bd.items()]
    
    # 计算分母的实部总和
    bd_real = sum(r for r in denom_real)
    
    # 计算分母的虚部总和
    bd_imag = sum(r for r in denom_imag)
    
    # 计算分子的实部系数列表，根据键的第一个元素的模 4 的不同值来决定系数的正负
    num_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in ba.items()]
    
    # 计算分子的虚部系数列表，根据键的第一个元素的模 4 的不同值来决定系数的正负
    num_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in ba.items()]
    
    # 计算分子的实部总和
    ba_real = sum(r for r in num_real)
    
    # 计算分子的虚部总和
    ba_imag = sum(r for r in num_imag)
    
    # 计算分数的商，结果为一个包含两个元素的元组，每个元素是一个多项式对象
    # 第一个元素是商的实部，第二个元素是商的虚部
    ba = ((ba_real*bd_real + ba_imag*bd_imag).as_poly(gen), (ba_imag*bd_real - ba_real*bd_imag).as_poly(gen))
    
    # 计算分母的多项式表示
    bd = (bd_real*bd_real + bd_imag*bd_imag).as_poly(gen)
    
    # 返回结果元组，包含分子的实部、虚部以及分母的多项式表示
    return (ba[0], ba[1], bd)
# 定义一个函数用于处理参数化的 Risch 微分方程中的特殊分母部分
def prde_special_denom(a, ba, bd, G, DE, case='auto'):
    """
    Parametric Risch Differential Equation - Special part of the denominator.

    Explanation
    ===========

    Case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the hyperexponential
    (resp. hypertangent) case, given a derivation D on k[t] and a in k[t],
    b in k<t>, and g1, ..., gm in k(t) with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the tuple (A, B, GG, h) such that A, B, h in
    k[t], GG = [gg1, ..., ggm] in k(t)^m, and for any solution c1, ..., cm in
    Const(k) and q in k<t> of a*Dq + b*q == Sum(ci*gi, (i, 1, m)), r == q*h in
    k[t] satisfies A*Dr + B*r == Sum(ci*ggi, (i, 1, m)).

    For case == 'primitive', k<t> == k[t], so it returns (a, b, G, 1) in this
    case.
    """
    # TODO: Merge this with the very similar special_denom() in rde.py

    # 如果 case 为 'auto'，则使用 DE 对象中的 case 属性
    if case == 'auto':
        case = DE.case

    # 根据 case 的不同情况，选择不同的多项式 p
    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        # 计算 B = ba / bd，并返回元组 (a, B, G, Poly(1, DE.t))
        B = ba.quo(bd)
        return (a, B, G, Poly(1, DE.t))
    else:
        # 抛出异常，说明 case 参数不合法
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    # 计算 nb 和 nc
    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = min(order_at(Ga, p, DE.t) - order_at(Gd, p, DE.t) for Ga, Gd in G)

    # 计算 n
    n = min(0, nc - min(0, nb))

    # 如果 nb 为零，考虑可能的取消情况
    if not nb:
        # 只有在 case 为 'exp' 时才会执行以下代码块
        if case == 'exp':
            # 计算 dcoeff = DE.d / Poly(DE.t)
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))

            # 计算 alphaa, alphad，其中 alphaa = -ba(0) / bd(0) / a(0)
            alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)

            # 计算 etaa, etad
            etaa, etad = frac_in(dcoeff, DE.t)

            # 调用 parametric_log_deriv 函数计算 A
            A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)

            # 如果 A 不为 None，则进一步处理
            if A is not None:
                Q, m, z = A
                if Q == 1:
                    n = min(n, m)

        # 只有在 case 为 'tan' 时才会执行以下代码块
        elif case == 'tan':
            # 计算 dcoeff = DE.d / Poly(DE.t**2 + 1)
            dcoeff = DE.d.quo(Poly(DE.t**2 + 1, DE.t))

            # 计算 betaa, alphaa, alphad
            betaa, alphaa, alphad =  real_imag(ba, bd*a, DE.t)
            betad = alphad

            # 计算 etaa, etad
            etaa, etad = frac_in(dcoeff, DE.t)

            # 调用 recognize_log_derivative 函数判断是否是对数导数
            if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                B = parametric_log_deriv(betaa, betad, etaa, etad, DE)
                if A is not None and B is not None:
                    Q, s, z = A
                    if Q == 1:
                        n = min(n, s/2)

    # 计算 N
    N = max(0, -nb)
    pN = p**N
    pn = p**-n  # 计算 p 的 N 次幂和 -n 次幂，即 p 的倒数

    A = a*pN  # 计算 a 乘以 p 的 N 次幂
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN  # 计算多项式 ba 除以 bd，加上一些乘积和导数操作
    G = [(Ga*pN*pn).cancel(Gd, include=True) for Ga, Gd in G]  # 对列表 G 中的每对元素应用 cancel 方法
    h = pn  # 将 h 设置为 p 的 -n 次幂，即 p 的倒数

    # 返回一个元组，包含以下元素：
    # (a*p**N, (b + n*a*Dp/p)*p**N, g1*p**(N - n), ..., gm*p**(N - n), p**-n)
    return (A, B, G, h)
def prde_linear_constraints(a, b, G, DE):
    """
    Parametric Risch Differential Equation - Generate linear constraints on the constants.

    Explanation
    ===========

    Given a derivation D on k[t], a, b, in k[t] with gcd(a, b) == 1, and
    G = [g1, ..., gm] in k(t)^m, return Q = [q1, ..., qm] in k[t]^m and a
    matrix M with entries in k(t) such that for any solution c1, ..., cm in
    Const(k) and p in k[t] of a*Dp + b*p == Sum(ci*gi, (i, 1, m)),
    (c1, ..., cm) is a solution of Mx == 0, and p and the ci satisfy
    a*Dp + b*p == Sum(ci*qi, (i, 1, m)).

    Because M has entries in k(t), and because Matrix does not play well with
    Poly, M will be a Matrix of Basic expressions.
    """
    m = len(G)  # 获取 G 的长度，即向量的数量

    Gns, Gds = list(zip(*G))  # 分离 G 中每个向量的分子和分母部分
    d = reduce(lambda i, j: i.lcm(j), Gds)  # 计算 G 的分母的最小公倍数
    d = Poly(d, field=True)  # 将最小公倍数转换为多项式对象
    Q = [(ga*(d).quo(gd)).div(d) for ga, gd in G]  # 计算每个 gi 对应的 qi

    if not all(ri.is_zero for _, ri in Q):  # 检查所有 qi 是否均为零
        N = max(ri.degree(DE.t) for _, ri in Q)  # 计算 Q 中每个元素对 DE.t 的最高次数
        M = Matrix(N + 1, m, lambda i, j: Q[j][1].nth(i), DE.t)  # 创建基于 Q 的 Matrix 对象
    else:
        M = Matrix(0, m, [], DE.t)  # 如果所有 qi 都为零，则返回空的 Matrix，无约束条件

    qs, _ = list(zip(*Q))  # 提取 Q 中的所有 qi
    return (qs, M)  # 返回 qi 列表和约束矩阵 M

def poly_linear_constraints(p, d):
    """
    Given p = [p1, ..., pm] in k[t]^m and d in k[t], return
    q = [q1, ..., qm] in k[t]^m and a matrix M with entries in k such
    that Sum(ci*pi, (i, 1, m)), for c1, ..., cm in k, is divisible
    by d if and only if (c1, ..., cm) is a solution of Mx = 0, in
    which case the quotient is Sum(ci*qi, (i, 1, m)).
    """
    m = len(p)  # 获取 p 的长度

    q, r = zip(*[pi.div(d) for pi in p])  # 计算每个 pi 除以 d 的商和余数

    if not all(ri.is_zero for ri in r):  # 检查所有余数 ri 是否均为零
        n = max(ri.degree() for ri in r)  # 计算所有余数 ri 的最高次数
        M = Matrix(n + 1, m, lambda i, j: r[j].nth(i), d.gens)  # 创建基于余数 r 的 Matrix 对象
    else:
        M = Matrix(0, m, [], d.gens)  # 如果所有余数 ri 都为零，则返回空的 Matrix，无约束条件

    return q, M  # 返回 qi 列表和约束矩阵 M

def constant_system(A, u, DE):
    """
    Generate a system for the constant solutions.

    Explanation
    ===========

    Given a differential field (K, D) with constant field C = Const(K), a Matrix
    A, and a vector (Matrix) u with coefficients in K, returns the tuple
    (B, v, s), where B is a Matrix with coefficients in C and v is a vector
    (Matrix) such that either v has coefficients in C, in which case s is True
    and the solutions in C of Ax == u are exactly all the solutions of Bx == v,
    or v has a non-constant coefficient, in which case s is False Ax == u has no
    constant solution.

    This algorithm is used both in solving parametric problems and in
    determining if an element a of K is a derivative of an element of K or the
    logarithmic derivative of a K-radical using the structure theorem approach.

    Because Poly does not play well with Matrix yet, this algorithm assumes that
    all matrix entries are Basic expressions.
    """
    if not A:  # 检查矩阵 A 是否为空
        return A, u

    Au = A.row_join(u)  # 将矩阵 A 和向量 u 水平连接成一个新矩阵 Au
    Au, _ = Au.rref()  # 对增广矩阵 Au 进行行简化阶梯形式处理

    # Warning: This will NOT return correct results if cancel() cannot reduce
    # 警告：如果 cancel() 无法简化，则此处不会返回正确的结果
    # 定义变量 A 和 u，分别为矩阵 Au 的前 m 列和最后一列
    A, u = Au[:, :-1], Au[:, -1]

    # 定义函数 D，用于对表达式进行微分，DE 是微分方程的变量列表
    D = lambda x: derivation(x, DE, basic=True)

    # 使用 itertools.product 遍历矩阵 A 的所有行列组合
    for j, i in itertools.product(range(A.cols), range(A.rows)):
        # 检查 A[i, j] 的表达式是否包含微分方程变量
        if A[i, j].expr.has(*DE.T):
            # 计算 Ri = A[i, :]
            Ri = A[i, :]
            # 计算 DAij = D(A[i, j])
            DAij = D(A[i, j])
            # 计算 Rm1 = Ri 中每个元素除以 DAij
            Rm1 = Ri.applyfunc(lambda x: D(x) / DAij)
            # 计算 um1 = u[i] 除以 DAij
            um1 = D(u[i]) / DAij

            # 计算 Aj = A[:, j]
            Aj = A[:, j]

            # 更新矩阵 A 和向量 u
            A = A - Aj * Rm1
            u = u - Aj * um1

            # 将 Rm1 作为 A 的新列添加
            A = A.col_join(Rm1)
            # 将 um1 作为 u 的新行添加
            u = u.col_join(Matrix([um1], u.gens))

    # 返回更新后的矩阵 A 和向量 u
    return (A, u)
# 特殊多项式微分方程算法：参数化版本
# 给定一个在 k[t] 上的导数 D，一个整数 n，以及 k[t] 中的多项式 a, b, q1, ..., qm，
# 其中 deg(a) > 0 且 gcd(a, b) == 1。返回 (A, B, Qq, R, n1)，其中 Qq = [q1, ..., qm]，
# R = [r1, ..., rm]，使得对于任意常数解 c1, ..., cm 和 k[t] 中次数不超过 n 的 q，
# 满足 a*Dq + b*q == Sum(ci*gi, (i, 1, m))，p = (q - Sum(ci*ri, (i, 1, m)))/a 的次数不超过 n1，
# 且满足 A*Dp + B*p == Sum(ci*qi, (i, 1, m))。
def prde_spde(a, b, Q, n, DE):
    R, Z = list(zip(*[gcdex_diophantine(b, a, qi) for qi in Q]))

    # A 和 B 的定义
    A = a
    B = b + derivation(a, DE)

    # 计算 Qq
    Qq = [zi - derivation(ri, DE) for ri, zi in zip(R, Z)]

    # 将 R 转换为列表
    R = list(R)

    # 计算 n1
    n1 = n - a.degree(DE.t)

    return (A, B, Qq, R, n1)


# 参数化多项式 Risch 微分方程 - 无取消项：当 b 的次数足够大时
# 给定在 k[t] 上的导数 D，整数 n，以及 k[t] 中的多项式 b, q1, ..., qm，
# 其中 b != 0 并且 D == d/dt 或者 deg(b) > max(0, deg(D) - 1)。返回 h1, ..., hr 和一个系数为 Const(k) 的矩阵 A，
# 如果 c1, ..., cm 是 Const(k) 中的常数，并且 q 在 k[t] 满足 deg(q) <= n 且 Dq + b*q == Sum(ci*qi, (i, 1, m))，
# 那么 q = Sum(dj*hj, (j, 1, r))，其中 d1, ..., dr 在 Const(k) 中，且 A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0。
def prde_no_cancel_b_large(b, Q, n, DE):
    db = b.degree(DE.t)
    m = len(Q)
    H = [Poly(0, DE.t)]*m

    # 嵌套循环遍历 n 到 0 的范围和 Q 中的索引 i
    for N, i in itertools.product(range(n, -1, -1), range(m)):  # [n, ..., 0]
        si = Q[i].nth(N + db)/b.LC()
        sitn = Poly(si*DE.t**N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    # 检查 Q 中的每个多项式是否为零
    if all(qi.is_zero for qi in Q):
        dc = -1
    else:
        dc = max(qi.degree(DE.t) for qi in Q)

    # 构建矩阵 M
    M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
    A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
    c = eye(m, DE.t)

    # 构造返回的矩阵 A
    A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))

    return (H, A)


# 参数化多项式 Risch 微分方程 - 无取消项：当 b 的次数足够小时
# 给定在 k[t] 上的导数 D，整数 n，以及 k[t] 中的多项式 b, q1, ..., qm，
# 其中 deg(b) < deg(D) - 1 并且 D == d/dt 或者 deg(D) >= 2。返回 h1, ..., hr 和一个系数为 Const(k) 的矩阵 A，
# 如果 c1, ..., cm 是 Const(k) 中的常数，并且 q 在 k[t] 满足 deg(q) <= n 且 Dq + b*q == Sum(ci*qi, (i, 1, m))，
# 那么 q = Sum(dj*hj, (j, 1, r))，其中 d1, ..., dr 在 Const(k) 中，且 A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0。
def prde_no_cancel_b_small(b, Q, n, DE):
    m = len(Q)
    H = [Poly(0, DE.t)]*m
    # 使用 itertools.product 生成器生成两个范围的笛卡尔积，其中第一个范围是 [n, ..., 1]，第二个范围是 [0, ..., m-1]
    for N, i in itertools.product(range(n, 0, -1), range(m)):  # [n, ..., 1]
        # 计算 si = Q[i] 中的第 N + DE.d.degree(DE.t) - 1 项除以 N*DE.d.LC()
        si = Q[i].nth(N + DE.d.degree(DE.t) - 1)/(N*DE.d.LC())
        # 构造新的多项式 sitn = si*DE.t**N
        sitn = Poly(si*DE.t**N, DE.t)
        # 更新 H[i]，累加 sitn
        H[i] = H[i] + sitn
        # 更新 Q[i]，减去 sitn 的导数 derivation(sitn, DE)，再减去 b*sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    # 如果 b 关于 DE.t 的次数大于 0
    if b.degree(DE.t) > 0:
        # 对每个 i 循环
        for i in range(m):
            # 计算 si，即 Q[i] 的第 b.degree(DE.t) 项的系数除以 b 的首项系数
            si = Poly(Q[i].nth(b.degree(DE.t))/b.LC(), DE.t)
            # 更新 H[i]，累加 si
            H[i] = H[i] + si
            # 更新 Q[i]，减去 si 的导数 derivation(si, DE)，再减去 b*si
            Q[i] = Q[i] - derivation(si, DE) - b*si
        # 如果 Q 中的所有项都是零多项式
        if all(qi.is_zero for qi in Q):
            # 设置 dc = -1
            dc = -1
        else:
            # 否则，取 Q 中每个多项式关于 DE.t 的最高次数的最大值
            dc = max(qi.degree(DE.t) for qi in Q)
        # 构造矩阵 M，其大小为 (dc + 1) × m，每个元素为 Q[j] 的第 i 项系数
        M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
        # 调用 constant_system 函数，返回 A 和 u，其中 A 是方程组的系数矩阵
        A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
        # 构造单位矩阵和负单位矩阵，并将它们合并成 A
        c = eye(m, DE.t)
        A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))
        return (H, A)

    # 如果 b 的次数为 0，则执行以下代码块
    # else: b is in k, deg(qi) < deg(Dt)

    # 设置 t = DE.t
    t = DE.t
    # 如果 DE.case 不是 'base'
    if DE.case != 'base':
        # 使用 DecrementLevel 上下文管理器减少 DE 的级别
        with DecrementLevel(DE):
            # 将 b 转化为 t0 变量的有理函数形式 ba/bd
            t0 = DE.t  # k = k0(t0)
            ba, bd = frac_in(b, t0, field=True)
            # 将 Q 中每个多项式的首项转化为 t0 变量的有理函数形式 Q0
            Q0 = [frac_in(qi.TC(), t0, field=True) for qi in Q]
            # 调用 param_rischDE 函数计算 ba, bd, Q0, DE，并返回 f 和 B
            f, B = param_rischDE(ba, bd, Q0, DE)

            # 将 f 转化为 k[t] 中的常数多项式列表
            f = [Poly(fa.as_expr()/fd.as_expr(), t, field=True)
                 for fa, fd in f]
            # 将 B 转化为 k[t] 中的矩阵
            B = Matrix.from_Matrix(B.to_Matrix(), t)
    else:
        # 如果 DE.case 是 'base'，则执行以下代码块
        # Base case. Dy == 0 for all y in k and b == 0.
        # Dy + b*y = Sum(ci*qi) is solvable if and only if
        # Sum(ci*qi) == 0 in which case the solutions are
        # y = d1*f1 for f1 = 1 and any d1 in Const(k) = k.

        # 设置 f 为包含单个常数多项式 1 的列表
        f = [Poly(1, t, field=True)]  # r = 1
        # 构造 B 矩阵，其大小为 m × (m + 1)，最后一列为零
        B = Matrix([[qi.TC() for qi in Q] + [S.Zero]], DE.t)
        # 求解条件为 B*Matrix([c1, ..., cm, d1]) == 0
        # 对于 d1 没有任何约束

    # 计算 Q 中每个多项式的次数关于 DE.t 的最大值
    d = max(qi.degree(DE.t) for qi in Q)
    # 如果 d 大于 0
    if d > 0:
        # 构造矩阵 M，其大小为 d × m，每个元素为 Q[j] 的第 i+1 项系数
        M = Matrix(d, m, lambda i, j: Q[j].nth(i + 1), DE.t)
        # 调用 constant_system 函数，返回 A 和一个未使用的变量 _
        A, _ = constant_system(M, zeros(d, 1, DE.t), DE)
    else:
        # 如果 d 等于 0，则没有对 hj 的约束
        A = Matrix(0, m, [], DE.t)

    # 原方程的解为 y = Sum(dj*fj, (j, 1, r) + Sum(ei*hi, (i, 1, m))
    # 其中 ei == ci (i = 1, ..., m)，当 A*Matrix([c1, ..., cm]) == 0
    # 且 B*Matrix([c1, ..., cm, d1, ..., dr]) == 0 时成立

    # 构造综合约束矩阵，其大小为 m + r + m 列
    r = len(f)
    I = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, r + m, DE.t))
    # 将矩阵 B 扩展为 B.row_join(zeros(B.rows, m, DE.t)) 的形式
    B = B.row_join(zeros(B.rows, m, DE.t))
    # 将单位矩阵 I 扩展为 I.row_join(zeros(m, r, DE.t)) 的形式，并与 -I 拼接
    C = I.row_join(zeros(m, r, DE.t)).row_join(-I)

    # 返回结果 f + H 与矩阵 A、B、C 按列连接的结果
    return f + H, A.col_join(B).col_join(C)
def prde_cancel_liouvillian(b, Q, n, DE):
    """
    Pg, 237.
    """
    H = []  # 初始化空列表 H，用于存储结果

    # 如果 DE.case 是 'primitive'，则执行以下操作：
    # 使用 DecrementLevel 上下文管理器处理 DE
    # 假设我们可以在 'k' 上解决这类问题（而不是 k[t]）
    if DE.case == 'primitive':
        with DecrementLevel(DE):
            # 将 b 转化为 DE.t 的分数形式
            ba, bd = frac_in(b, DE.t, field=True)

    # 从 n 到 0 遍历，步长为 -1
    for i in range(n, -1, -1):
        # 如果 DE.case 是 'exp'，则执行以下操作：
        # 使用 DecrementLevel 上下文管理器处理 DE
        # 计算 ba 和 bd，使得 ba 是 b 加上一些项的分数形式
        if DE.case == 'exp':
            with DecrementLevel(DE):
                ba, bd = frac_in(b + (i*(derivation(DE.t, DE)/DE.t)).as_poly(b.gens),
                                DE.t, field=True)
        
        with DecrementLevel(DE):
            # 计算 Q 中第 i 个元素的分数形式 Qy
            Qy = [frac_in(q.nth(i), DE.t, field=True) for q in Q]
            # 调用 param_rischDE 函数，计算参数 Risch 微分方程的结果
            fi, Ai = param_rischDE(ba, bd, Qy, DE)
        
        # 将 fi 中的每个分子分母转化为 k[t] 中的多项式
        fi = [Poly(fa.as_expr()/fd.as_expr(), DE.t, field=True)
                for fa, fd in fi]
        # 设置 Ai 的生成元为 DE.t
        Ai = Ai.set_gens(DE.t)

        ri = len(fi)

        if i == n:
            M = Ai
        else:
            # 将 Ai 作为 M 的列连接
            M = Ai.col_join(M.row_join(zeros(M.rows, ri, DE.t)))

        Fi, hi = [None]*ri, [None]*ri

        # 对 ri 个元素进行循环，构建 hji 和 Fi[j]
        # hji 是 fi[j] 乘以 DE.t**i 的 k[t] 多项式
        # Fi[j] 是 -(D(hji, DE) - b*hji)
        for j in range(ri):
            hji = fi[j] * (DE.t**i).as_poly(fi[j].gens)
            hi[j] = hji
            Fi[j] = -(derivation(hji, DE) - b*hji)

        # 将 hi 添加到 H 中
        H += hi
        # 将 Fi 添加到 Q 中
        Q = Q + Fi

    return (H, M)


def param_poly_rischDE(a, b, q, n, DE):
    """Polynomial solutions of a parametric Risch differential equation.

    Explanation
    ===========

    Given a derivation D in k[t], a, b in k[t] relatively prime, and q
    = [q1, ..., qm] in k[t]^m, return h = [h1, ..., hr] in k[t]^r and
    a matrix A with m + r columns and entries in Const(k) such that
    a*Dp + b*p = Sum(ci*qi, (i, 1, m)) has a solution p of degree <= n
    in k[t] with c1, ..., cm in Const(k) if and only if p = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.
    """
    m = len(q)
    if n < 0:
        # 只有平凡的零解是可能的。
        # 找到 qi 之间的关系。
        if all(qi.is_zero for qi in q):
            return [], zeros(1, m, DE.t)  # 没有约束条件。

        # 构建一个矩阵 M，其元素为 q[j].nth(i)
        N = max(qi.degree(DE.t) for qi in q)
        M = Matrix(N + 1, m, lambda i, j: q[j].nth(i), DE.t)
        # 调用 constant_system 函数，求解常系数线性方程组
        A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)

        return [], A
    # 如果 a 是地面项
    if a.is_ground:
        # 标准化：令 a = 1
        a = a.LC()
        # b 除以 a，q 中每个元素也除以 a
        b, q = b.quo_ground(a), [qi.quo_ground(a) for qi in q]

        # 如果 b 非零并且（DE 的 case 是 'base' 或者 b 的次数大于 max(0, DE.d 的次数 - 1)）
        if not b.is_zero and (DE.case == 'base' or
                b.degree() > max(0, DE.d.degree() - 1)):
            return prde_no_cancel_b_large(b, q, n, DE)

        # 如果（b 是零或者 b 的次数小于 DE.d 的次数 - 1）
        # 并且（DE 的 case 是 'base' 或者 DE.d 的次数至少为 2）
        elif ((b.is_zero or b.degree() < DE.d.degree() - 1)
                and (DE.case == 'base' or DE.d.degree() >= 2)):
            return prde_no_cancel_b_small(b, q, n, DE)

        # 如果 DE.d 的次数至少为 2，且 b 的次数等于 DE.d 的次数 - 1，
        # 且 n 大于 -b.as_poly().LC() / DE.d.as_poly().LC()
        elif (DE.d.degree() >= 2 and
              b.degree() == DE.d.degree() - 1 and
              n > -b.as_poly().LC() / DE.d.as_poly().LC()):
            raise NotImplementedError("prde_no_cancel_b_equal() is "
                "not yet implemented.")

        # 否则，对于 Liouvillian 情况
        else:
            # 如果 DE.case 在 ('primitive', 'exp') 中
            if DE.case in ('primitive', 'exp'):
                return prde_cancel_liouvillian(b, q, n, DE)
            else:
                raise NotImplementedError("non-linear and hypertangent "
                        "cases have not yet been implemented")

    # 否则：deg(a) > 0

    # 当 n >= 0 时迭代 SPDE，累积系数和项以恢复原始解
    alpha, beta = a.one, [a.zero]*m
    while n >= 0:  # 并且 a, b 互素
        a, b, q, r, n = prde_spde(a, b, q, n, DE)
        beta = [betai + alpha*ri for betai, ri in zip(beta, r)]
        alpha *= a
        # 解 p 满足 a*Dp + b*p = Sum(ci*qi) 对应于初始方程的解 alpha*p + Sum(ci*betai)
        d = a.gcd(b)
        if not d.is_ground:
            break

    # a*Dp + b*p = Sum(ci*qi) 只有在和为 d 的情况下才可能有多项式解

    qq, M = poly_linear_constraints(q, d)
    # qq = [qq1, ..., qqm]，其中 qqi = qi.quo(d)
    # M 是一个 m 列矩阵，元素在 k 中
    # Sum(fi*qi, (i, 1, m))，其中 f1, ..., fm 是 k 中的元素，如果且仅如果 M*Matrix([f1, ..., fm]) == 0，那么其可被 d 整除，其中 M*Matrix([f1, ..., fm]) 是 Sum(fi*qqi)

    A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # A 是一个 m 列矩阵，元素在 Const(k) 中
    # 如果 A*Matrix([c1, ..., cm]) == 0，则 Sum(ci*qqi) 是 Sum(ci*qi).quo(d)，并且余数为零

    V = A.nullspace()
    # V = [v1, ..., vu]，其中每个 vj 是一个列矩阵，元素为 Const(k) 中的 aj1, ..., ajm
    # Sum(aji*qi) 如果且仅如果 ci = Sum(dj*aji)（i = 1, ..., m）对于一些 Const(k) 中的 d1, ..., du
    # 在这种情况下，a*Dp + b*p = Sum(ci*qi) 的解是 (a/d)*Dp + (b/d)*p = Sum(dj*rj)，其中 rj = Sum(aji*qqi)

    if not V:  # 没有非平凡解
        return [], eye(m, DE.t)  # 可以返回 A，但这是具有最少行数的选择
    Mqq = Matrix([qq])  # 将列表 qq 转换为一个行向量的矩阵 Mqq

    # 计算向量 r = [r1, ..., ru]，其中每个元素为 Mqq 与 V 中向量 vj 的点积的第一个元素
    r = [(Mqq*vj)[0] for vj in V]

    # 解释如下三行注释：
    # (a/d)*Dp + (b/d)*p = Sum(dj*rj) 的解对应于初始方程的解 alpha*p + Sum(Sum(dj*aji)*betai)
    # 这些解等于 alpha*p + Sum(dj*fj)，其中 fj = Sum(aji*betai)。
    Mbeta = Matrix([beta])
    # 计算向量 f = [f1, ..., fu]，其中每个元素为 Mbeta 与 V 中向量 vj 的点积的第一个元素
    f = [(Mbeta*vj)[0] for vj in V]

    #
    # 递归解决减少的方程。
    #
    # 使用参数化Risch-Differential方程求解函数 param_poly_rischDE 求解
    g, B = param_poly_rischDE(a.quo(d), b.quo(d), r, n, DE)

    # g = [g1, ..., gv] 属于 k[t]^v，B 是一个矩阵，具有 u + v 列，其元素属于 Const(k)
    # 使得 (a/d)*Dp + (b/d)*p = Sum(dj*rj) 在 k[t] 中存在度数 <= n 的解 p
    # 当且仅当 p = Sum(ek*gk)，其中 e1, ..., ev 属于 Const(k) 并且 B*Matrix([d1, ..., du, e1, ..., ev]) == 0。
    # 原方程的解为 Sum(dj*fj, (j, 1, u)) + alpha*Sum(ek*gk, (k, 1, v))。

    # 收集解的组成部分。
    h = f + [alpha*gk for gk in g]

    # 构建组合关系矩阵。
    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(g), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    # 返回 h 和 A 作为结果。
    return h, A
def param_rischDE(fa, fd, G, DE):
    """
    Solve a Parametric Risch Differential Equation: Dy + f*y == Sum(ci*Gi, (i, 1, m)).

    Explanation
    ===========

    Given a derivation D in k(t), f in k(t), and G
    = [G1, ..., Gm] in k(t)^m, return h = [h1, ..., hr] in k(t)^r and
    a matrix A with m + r columns and entries in Const(k) such that
    Dy + f*y = Sum(ci*Gi, (i, 1, m)) has a solution y
    in k(t) with c1, ..., cm in Const(k) if and only if y = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    Elements of k(t) are tuples (a, d) with a and d in k[t].
    """
    m = len(G)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    # 计算弱标准化方程的解，得到标准化系数 q，并更新 fa 和 fd

    gamma = q
    G = [(q*ga).cancel(gd, include=True) for ga, gd in G]
    # 根据标准化系数 q 对 G 中的每个项进行标准化处理

    a, (ba, bd), G, hn = prde_normal_denom(fa, fd, G, DE)
    # 计算标准化分母的解，并更新 fa, fd, G，并返回标准化分母 hn

    gamma *= hn
    # 更新 gamma，乘以标准化分母 hn

    A, B, G, hs = prde_special_denom(a, ba, bd, G, DE)
    # 计算特殊分母的解，并更新 a, b, G，并返回特殊分母 hs

    gamma *= hs
    # 更新 gamma，乘以特殊分母 hs

    g = A.gcd(B)
    a, b, g = A.quo(g), B.quo(g), [gia.cancel(gid*g, include=True) for
        gia, gid in G]
    # 对 A*Dp + B*p = Sum(ci*gi) 进行处理，计算其解并更新参数

    q, M = prde_linear_constraints(a, b, g, DE)
    # 计算线性约束方程的解 q 和矩阵 M

    M, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # 计算常数系统的解 M

    ## Reduce number of constants at this point

    V = M.nullspace()
    # 计算 M 的零空间 V，得到非平凡解的集合

    if not V:  # No non-trivial solution
        return [], eye(m, DE.t)
    # 如果 V 为空集，返回空列表和单位矩阵

    Mq = Matrix([q])  # A single row.
    r = [(Mq*vj)[0] for vj in V]  # [r1, ..., ru]
    # 计算 q 与 V 中每个向量 vj 的乘积，并将结果存储在列表 r 中

    # Solutions of a*Dp + b*p = Sum(dj*rj) correspond to solutions
    # 满足 a*Dp + b*p = Sum(dj*rj) 的解对应于解
    # 计算初始方程中 y = p/gamma，其中 ci = Sum(dj*aji)。

    try:
        # 尝试使用 n=5。对于 prde_spde，无论 n 的取值如何，都将终止。
        n = bound_degree(a, b, r, DE, parametric=True)
    except NotImplementedError:
        # 抛出 NotImplementedError 异常时，暂时设定 n=5。
        # 最终将删除此临时绑定。
        # 当前添加的测试用例即使在 n=5 时也需要大量时间，
        # 在较大的 n 值下需要更长的时间。
        n = 5

    # 调用 param_poly_rischDE 函数，计算得到 h 和 B。
    h, B = param_poly_rischDE(a, b, r, n, DE)

    # 在注释中给出的公式中：
    # h = [h1, ..., hv] 是 k[t]^v 中的多项式向量，
    # B 是一个矩阵，其列数为 u + v，条目在 Const(k) 中，
    # 满足方程 a*Dp + b*p = Sum(dj*rj) 的解 p 的次数 <= n，
    # 当且仅当 p = Sum(ek*hk)，其中 e1, ..., ev 在 Const(k) 中，
    # 且 B*Matrix([d1, ..., du, e1, ..., ev]) == 0。
    # 原方程的解 y = Sum(ek*hk, (k, 1, v))/gamma，其中 gamma 是常数。

    ## 构建具有 m + u + v 列的组合关系矩阵。

    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(h), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    ## 消除 d1, ..., du。

    W = A.nullspace()

    # W = [w1, ..., wt]，每个 wl 是一个列矩阵，
    # 其中条目为 Const(k) 中的块 (k = 1, ..., m + u + v)。
    # 向量 (bl1, ..., blm) 生成方程 Dy + f*y == Sum(ci*Gi) 的常数族 (c1, ..., cm) 的空间。
    # 它们生成空间并形成一个基础，除非在 k(t} 中方程 Dy + f*y == 0 有解。
    # 相应的解为 y = Sum(blk'*hk, (k, 1, v))/gamma，其中 k' = k + m + u。

    v = len(h)
    shape = (len(W), m+v)
    elements = [wl[:m] + wl[-v:] for wl in W] # 删去 dj。
    items = [e for row in elements for e in row]

    # 如果 W 是空的，则需要设置 shape。
    M = Matrix(*shape, items, DE.t)
    N = M.nullspace()

    # N = [n1, ..., ns]，其中 ni 属于 Const(k)^(m + v)，
    # 是生成 c1, ..., cm, e1, ..., ev 之间线性关系空间的列向量。

    C = Matrix([ni[:] for ni in N], DE.t)  # 行 n1, ..., ns。

    return [hk.cancel(gamma, include=True) for hk in h], C
def limited_integrate_reduce(fa, fd, G, DE):
    """
    Simpler version of step 1 & 2 for the limited integration problem.

    Explanation
    ===========

    Given a derivation D on k(t) and f, g1, ..., gn in k(t), return
    (a, b, h, N, g, V) such that a, b, h in k[t], N is a non-negative integer,
    g in k(t), V == [v1, ..., vm] in k(t)^m, and for any solution v in k(t),
    c1, ..., cm in C of f == Dv + Sum(ci*wi, (i, 1, m)), p = v*h is in k<t>, and
    p and the ci satisfy a*Dp + b*p == g + Sum(ci*vi, (i, 1, m)).  Furthermore,
    if S1irr == Sirr, then p is in k[t], and if t is nonlinear or Liouvillian
    over k, then deg(p) <= N.

    So that the special part is always computed, this function calls the more
    general prde_special_denom() automatically if it cannot determine that
    S1irr == Sirr.  Furthermore, it will automatically call bound_degree() when
    t is linear and non-Liouvillian, which for the transcendental case, implies
    that Dt == a*t + b with for some a, b in k*.
    """
    # Split the factor of fd with respect to DE
    dn, ds = splitfactor(fd, DE)
    # Split each factor of G with respect to DE
    E = [splitfactor(gd, DE) for _, gd in G]
    # Separate numerators and denominators of E
    En, Es = list(zip(*E))
    # Compute the least common multiple of dn and En
    c = reduce(lambda i, j: i.lcm(j), (dn,) + En)  # lcm(dn, en1, ..., enm)
    # Compute the gcd of c with its derivative with respect to DE.t
    hn = c.gcd(c.diff(DE.t))
    # Set a = hn and compute b = -derivation(hn, DE)
    a = hn
    b = -derivation(hn, DE)
    # Initialize N to 0

    # Determine behavior based on DE.case
    if DE.case in ('base', 'primitive', 'exp', 'tan'):
        # Compute the least common multiple of ds and Es
        hs = reduce(lambda i, j: i.lcm(j), (ds,) + Es)  # lcm(ds, es1, ..., esm)
        # Update a to be hn * hs
        a = hn * hs
        # Update b by subtracting (hn * derivation(hs, DE)) / hs
        b -= (hn * derivation(hs, DE)).quo(hs)
        # Calculate mu as the minimum of certain orders
        mu = min(order_at_oo(fa, fd, DE.t), min(order_at_oo(ga, gd, DE.t) for ga, gd in G))
        # Update N based on degrees and orders
        N = hn.degree(DE.t) + hs.degree(DE.t) + max(0, 1 - DE.d.degree(DE.t) - mu)
    else:
        # If DE.case is not handled, raise NotImplementedError
        raise NotImplementedError

    # Compute V as a list of simplified expressions
    V = [(-a * hn * ga).cancel(gd, include=True) for ga, gd in G]
    return (a, b, a, N, (a * hn * fa).cancel(fd, include=True), V)


def limited_integrate(fa, fd, G, DE):
    """
    Solves the limited integration problem:  f = Dv + Sum(ci*wi, (i, 1, n))
    """
    # Normalize fa and fd
    fa, fd = fa * Poly(1 / fd.LC(), DE.t), fd.monic()
    # Initialize Fa and Fd
    Fa = Poly(0, DE.t)
    Fd = Poly(1, DE.t)
    # Include (fa, fd) in G
    G = [(fa, fd)] + G
    # Call param_rischDE to solve the parametric Risch DE problem
    h, A = param_rischDE(Fa, Fd, G, DE)
    # Compute the nullspace of A
    V = A.nullspace()
    # Filter out zero vectors from V
    V = [v for v in V if v[0] != 0]
    # If V is empty, return None
    if not V:
        return None
    else:
        # 从向量空间 V 中任选一个向量，这里选择 V 中的第一个向量 V[0]
        c0 = V[0][0]
        # 构造向量 v = [-1, c1, ..., cm, d1, ..., dr]
        v = V[0] / (-c0)
        r = len(h)  # 计算列表 h 的长度，赋值给 r
        m = len(v) - r - 1  # 计算向量 v 的长度减去 r 和 1后的结果，赋值给 m
        C = list(v[1: m + 1])  # 从向量 v 中提取出索引为 1 到 m 的元素（即 c1 到 cm），构成列表 C
        # 计算 y = -sum(v[m + 1 + i]*h[i][0].as_expr()/h[i][1].as_expr() for i in range(r))
        y = -sum(v[m + 1 + i] * h[i][0].as_expr() / h[i][1].as_expr() \
                for i in range(r))
        y_num, y_den = y.as_numer_denom()  # 将 y 表达式转换为分子和分母形式
        Ya, Yd = Poly(y_num, DE.t), Poly(y_den, DE.t)  # 使用 DE.t 构造多项式对象 Ya 和 Yd
        Y = Ya * Poly(1 / Yd.LC(), DE.t), Yd.monic()  # 计算 Y = Ya * (1/Yd.LC()), Yd.monic()
        return Y, C  # 返回结果 Y 和列表 C
def parametric_log_deriv_heu(fa, fd, wa, wd, DE, c1=None):
    """
    Parametric logarithmic derivative heuristic.

    Explanation
    ===========

    Given a derivation D on k[t], f in k(t), and a hyperexponential monomial
    theta over k(t), raises either NotImplementedError, in which case the
    heuristic failed, or returns None, in which case it has proven that no
    solution exists, or returns a solution (n, m, v) of the equation
    n*f == Dv/v + m*Dtheta/theta, with v in k(t)* and n, m in ZZ with n != 0.

    If this heuristic fails, the structure theorem approach will need to be
    used.

    The argument w == Dtheta/theta
    """
    # TODO: finish writing this and write tests

    # Initialize c1 if not provided
    c1 = c1 or Dummy('c1')

    # Compute p = fa / fd and a = fa % fd
    p, a = fa.div(fd)
    
    # Compute q = wa / wd and b = wa % wd
    q, b = wa.div(wd)

    # Determine degrees B and C
    B = max(0, derivation(DE.t, DE).degree(DE.t) - 1)
    C = max(p.degree(DE.t), q.degree(DE.t))

    # Check if degree of q exceeds B
    if q.degree(DE.t) > B:
        # Construct equations for solve function
        eqs = [p.nth(i) - c1*q.nth(i) for i in range(B + 1, C + 1)]
        s = solve(eqs, c1)
        
        # Check if no solutions or non-rational solution for c1
        if not s or not s[c1].is_Rational:
            # deg(q) > B, no solution for c.
            return None
        
        # Extract M and N from rational solution of c1
        M, N = s[c1].as_numer_denom()
        M_poly = M.as_poly(q.gens)
        N_poly = N.as_poly(q.gens)

        # Calculate expressions for n*f*wd - m*w*fd and fd*wd
        nfmwa = N_poly * fa * wd - M_poly * wa * fd
        nfmwd = fd * wd
        
        # Check if (N*f - M*w) is a logarithmic derivative of a k(t)-radical
        Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE, 'auto')
        if Qv is None:
            # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
            return None

        Q, v = Qv

        # Check if Q or v is zero
        if Q.is_zero or v.is_zero:
            return None

        return (Q * N, Q * M, v)

    # Check if degree of p exceeds B
    if p.degree(DE.t) > B:
        return None

    # Compute least common multiple of leading coefficients of fd and wd
    c = lcm(fd.as_poly(DE.t).LC(), wd.as_poly(DE.t).LC())
    
    # Construct l = lcm of fd and wd, multiplied by a polynomial in DE.t
    l = fd.monic().lcm(wd.monic()) * Poly(c, DE.t)
    ln, ls = splitfactor(l, DE)
    
    # Compute z as product of ls and gcd of ln with its derivative
    z = ls * ln.gcd(ln.diff(DE.t))

    # Check if z contains DE.t
    if not z.has(DE.t):
        # TODO: We treat this as 'no solution', until the structure
        # theorem version of parametric_log_deriv is implemented.
        return None

    # Compute u1, r1 = (fa*l.quo(fd)).div(z) and u2, r2 = (wa*l.quo(wd)).div(z)
    u1, r1 = (fa * l.quo(fd)).div(z)  # (l*f).div(z)
    u2, r2 = (wa * l.quo(wd)).div(z)  # (l*w).div(z)

    # Construct equations for solve function
    eqs = [r1.nth(i) - c1 * r2.nth(i) for i in range(z.degree(DE.t))]
    s = solve(eqs, c1)
    
    # Check if no solutions or non-rational solution for c1
    if not s or not s[c1].is_Rational:
        # deg(q) <= B, no solution for c.
        return None

    # Extract M and N from rational solution of c1
    M, N = s[c1].as_numer_denom()

    # Calculate expressions for n*f*wd - m*w*fd and fd*wd
    nfmwa = N.as_poly(DE.t) * fa * wd - M.as_poly(DE.t) * wa * fd
    nfmwd = fd * wd
    
    # Check if (N*f - M*w) is a logarithmic derivative of a k(t)-radical
    Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
    if Qv is None:
        # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
        return None

    Q, v = Qv

    # Check if Q or v is zero
    if Q.is_zero or v.is_zero:
        return None

    return (Q * N, Q * M, v)


def parametric_log_deriv(fa, fd, wa, wd, DE):
    # TODO: Write the full algorithm using the structure theorems.
    A = parametric_log_deriv_heu(fa, fd, wa, wd, DE)
#    except NotImplementedError:
        # 如果启发式失败，则必须使用完整的方法。
        # TODO: 这可以更有效地实现。
        # 这并不太令人担忧，因为启发式方法处理大多数困难情况。
    return A
    To handle the case where we are given Df/f, not f, use is_deriv_k_in_field().

    See also
    ========
    is_log_deriv_k_t_radical_in_field, is_log_deriv_k_t_radical

    """
    # 计算导数 Df/f
    dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)), fd*fa
    # 对计算结果进行化简
    dfa, dfd = dfa.cancel(dfd, include=True)

    # 我们假设每个单项式都是递归超越的
    if len(DE.exts) != len(DE.D):
        # 如果存在 DE 中的 'tan' 或者不包含在 'log' 索引中的 'primitive' 集合
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: 在这种情况下真正应该做些什么？
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    # 提取表达式中的 'exp' 和 'log' 部分
    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # 由于表达式 dfa/dfd 可能不是其符号中的任何多项式，因此我们使用 Dummy 作为 PolyMatrix 的生成器
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    # 求解常数系统 A, u
    A, u = constant_system(lhs, rhs, DE)

    # 将 u 转换为矩阵表达式
    u = u.to_Matrix()  # Poly to Expr

    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # 如果 u 的所有元素不全为常数
        # 注意: 见 constant_system 中的注释

        return None
    else:
        if not all(i.is_Rational for i in u):
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            # 构造结果表达式 ans, result, const
            terms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            result = Add(*[Mul(i, j) for i, j in ans])
            argterms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            l = []
            ld = []
            for i, j in zip(argterms, u):
                # 处理诸如 sqrt(x**2) != x 和 sqrt(x**2 + 2*x + 1) != x + 1 等问题
                # Issue 10798: i 不必是多项式
                i, d = i.as_numer_denom()
                icoeff, iterms = sqf_list(i)
                l.append(Mul(*([Pow(icoeff, j)] + [Pow(b, e*j) for b, e in iterms])))
                dcoeff, dterms = sqf_list(d)
                ld.append(Mul(*([Pow(dcoeff, j)] + [Pow(b, e*j) for b, e in dterms])))
            # 计算常数项 const
            const = cancel(fa.as_expr()/fd.as_expr()/Mul(*l)*Mul(*ld))

            return (ans, result, const)
# 定义一个函数，用于检查给定的 Df 是否是 k(t)-根式的对数导数
def is_log_deriv_k_t_radical(fa, fd, DE, Df=True):
    r"""
    检查 Df 是否是 k(t)-根式的对数导数。

    Explanation
    ===========

    b in k(t) 可以被写成 k(t) 根式的对数导数，如果存在 n 属于 ZZ 和 u 属于 k(t)，且 n, u != 0，使得 n*b == Du/u。
    函数返回 (ans, u, n, const) 或者 None，其中 ans 是一个元组列表，满足 Mul(*[i**j for i, j in ans]) == u。
    这对于精确查看哪些 k(t) 中的元素产生了 u 是有用的。

    此函数使用结构定理方法，该定理表明对于任何 f 属于 K，如果 Df 是 K-根式的对数导数，则存在 ri 属于 QQ，使得::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i
            /     i     i     /     i   ---   =  Df.
            ---               ---        t
         i in L            i in E         i
               K/C(x)            K/C(x)

    其中 C = Const(K)，L_K/C(x) = { i in {1, ..., n}，使得 t_i 在 C(x)(t_1, ..., t_i-1) 上超越，且 Dt_i = Da_i/a_i，对于某些 a_i 属于 C(x)(t_1, ..., t_i-1)* }（即 K 上所有对数单项式的指标集合），E_K/C(x) = { i in {1, ..., n}，使得 t_i 在 C(x)(t_1, ..., t_i-1) 上超越，且 Dt_i/t_i = Da_i，对于某些 a_i 属于 C(x)(t_1, ..., t_i-1) }（即 K 上所有超指数单项式的指标集合）。如果 K 是 C(x) 的基本扩展，则 L_K/C(x) U E_K/C(x) 的基数恰好是 K 关于 C(x) 的超越次数。此外，因为 Const_D(K) == Const_D(C(x)) == C，当 t_i 在 E_K/C(x) 中时，deg(Dt_i) == 1，当 t_i 在 L_K/C(x) 中时，deg(Dt_i) == 0，特别是意味着 E_K/C(x) 和 L_K/C(x) 是不相交的。

    由于其本质，集合 L_K/C(x) 和 E_K/C(x) 必须使用相同的函数递归计算。因此，需要将它们作为 D（或 T）的索引传递。L_args 是由 L_K 索引的对数参数（即如果 i 在 L_K 中，则 T[i] == log(L_args[i])）。这对于计算最终答案 u，使得 n*f == Du/u 是必要的。

    exp(f) 与 u 在乘法常数上相同。这是因为它们在单项式方面行为相同。例如，exp(x) 和 exp(x + 1) == E*exp(x) 满足 Dt == t。因此，返回项 const。const 是这样的，即 exp(const)*f == u。这通过从一个指数项的参数中减去另一个指数项的参数来计算。因此，需要传递 E_args 中的指数项参数。

    处理给定 Df 而不是 f 的情况，请使用 is_log_deriv_k_t_radical_in_field()。

    See also
    ========

    is_log_deriv_k_t_radical_in_field, is_deriv_k

    """
    # 如果 Df 为真，则进行以下操作，否则跳到 else 分支
    if Df:
        # 计算导数 dfa 和 dfd
        dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)).cancel(fd**2,
            include=True)
    else:
        # 如果 Df 不为真，则直接将 fa 和 fd 赋给 dfa 和 dfd
        dfa, dfd = fa, fd

    # 假设每个单项式都是递归超越的
    if len(DE.exts) != len(DE.D):
        # 检查是否有超越函数为 'tan'，或者是否存在非 'log' 的基本超越函数
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            # 抛出未实现错误，表示不支持具有超双曲正切支持的结构定理的真实版本
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: 在这种情况下真正应该怎么做？
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    # 提取所有指数型 DE.D[i].quo(Poly(DE.T[i], DE.T[i])) 的表达式作为 E_part
    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    # 提取所有对数型 DE.D[i].as_expr() 的表达式作为 L_part
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # 表达式 dfa/dfd 可能不是其任何符号的多项式，因此我们使用 Dummy 作为 PolyMatrix 的生成器
    dum = Dummy()
    # 构建左侧矩阵 lhs 和右侧矩阵 rhs
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    # 调用 constant_system 函数，解出常数系统 A 和 u
    A, u = constant_system(lhs, rhs, DE)

    # 将 u 转换为矩阵类型，Poly 转为 Expr
    u = u.to_Matrix()

    # 如果 A 为空或者 u 的所有元素的派生为零，则返回 None
    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # 如果 u 的元素不全为常数
        # 注意：见 constant_system 的注释
        # 另外注意：derivation(basic=True) 调用了 cancel()
        return None
    else:
        # 如果 u 的所有元素不全为有理数
        if not all(i.is_Rational for i in u):
            # TODO: 但也许我们可以判断它们是否为非有理数，比如 log(2)/log(3)。
            # 另外，应该有一个选项，即使结果可能潜在错误，也可以继续。
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            # 计算 u 的分母的最小公倍数，并将 u 扩大 n 倍
            n = S.One*reduce(ilcm, [i.as_numer_denom()[1] for i in u])
            u *= n
            # 构建 terms 列表和 ans 列表
            terms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            # 构建结果 result，乘积项为 Pow(i, j)
            result = Mul(*[Pow(i, j) for i, j in ans])

            # exp(f) 将与 result 相同，只是乘以一个乘法常数。现在找到该常数的对数。
            argterms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            const = cancel(fa.as_expr()/fd.as_expr() -
                Add(*[Mul(i, j/n) for i, j in zip(argterms, u)]))

            # 返回结果元组 (ans, result, n, const)
            return (ans, result, n, const)
# 检查函数 fa 和 fd 的公共因子并进行化简，保留结果
fa, fd = fa.cancel(fd, include=True)

# 拆分 fd 成其公因式和简化后的部分，n 为公因式，s 为简化后的部分
n, s = splitfactor(fd, DE)

# 如果简化后的部分 s 不是单位元，则函数 f 必须是简单的，否则返回 None
if not s.is_one:
    pass

# 如果 z 为 None，则创建一个虚拟符号 z
z = z or Dummy('z')

# 对于给定的 fa, fd, DE，计算其残余 H 和标志 b
H, b = residue_reduce(fa, fd, DE, z=z)

# 如果标志 b 为 False，则返回 None，表示函数 f 无法表示为 k(t)-radical 的对数导数
if not b:
    return None

# 检查 H 中每个多项式的根是否都是有理数，如果有不是有理数的情况，则返回 None
roots = [(i, i.real_roots()) for i, _ in H]
if not all(len(j) == i.degree() and all(k.is_Rational for k in j) for i, j in roots):
    return None

# 将每个 H 中的根值及其对应的多项式残余组成元组列表
respolys, residues = list(zip(*roots)) or [[], []]

# 取得每个根值对应的多项式残余项，形成列表 residueterms
residueterms = [(H[j][1].subs(z, i), i) for j in range(len(H)) for i in residues[j]]

# TODO: 完成这部分的编写，并编写测试代码

# 计算 p = fa/fd - residue_reduce_derivation(H, DE, z)，并进行化简
p = cancel(fa.as_expr() / fd.as_expr() - residue_reduce_derivation(H, DE, z))

# 将化简后的表达式 p 转换为 DE.t 的多项式
p = p.as_poly(DE.t)

# 如果 p 为 None，则返回 None，表示函数 f - Dg 不在 k[t] 中
if p is None:
    return None

# 如果 p 关于 DE.t 的次数大于等于 max(1, DE.d.degree(DE.t))，则返回 None
if p.degree(DE.t) >= max(1, DE.d.degree(DE.t)):
    return None

# 如果 case 参数为 'auto'，则将其设为 DE.case
if case == 'auto':
    case = DE.case
    # 如果情况为 'exp'，则进行指数情况的处理
    if case == 'exp':
        # 计算导数的求导结果 (wa, wd)，并进行部分化简和取消操作
        wa, wd = derivation(DE.t, DE).cancel(Poly(DE.t, DE.t), include=True)
        # 减少处理级别，使用带分数的形式处理 p 关于 DE.t 的分数
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t, cancel=True)
            # 对 (wa, wd) 进行 DE.t 的分数处理
            wa, wd = frac_in((wa, wd), DE.t)
            # 计算参数化对数导数
            A = parametric_log_deriv(pa, pd, wa, wd, DE)
        # 如果 A 为 None，则返回 None
        if A is None:
            return None
        # 解析 A 的三个元素 n, e, u，并将 u 乘以 DE.t 的 e 次幂
        n, e, u = A
        u *= DE.t**e

    # 如果情况为 'primitive'，则进行原始情况的处理
    elif case == 'primitive':
        # 减少处理级别，使用带分数的形式处理 p 关于 DE.t 的分数
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t)
            # 判断是否满足自动情况下是对数导数的条件
            A = is_log_deriv_k_t_radical_in_field(pa, pd, DE, case='auto')
        # 如果 A 为 None，则返回 None
        if A is None:
            return None
        # 解析 A 的两个元素 n, u
        n, u = A

    # 如果情况为 'base'，则进行基础情况的处理
    elif case == 'base':
        # 提示可以从 ratint() 中使用更高效的余数减少
        # 如果 fd 不是平方自由或者 fa 的次数大于等于 fd 的次数
        if not fd.is_sqf or fa.degree() >= fd.degree():
            # 返回 None，因为在这种情况下 f 不是对数导数
            return None
        # 注意：如果 residueterms 为空列表，返回 (1, 1)
        # 在这种情况下 f 最好是 0。
        n = S.One * reduce(ilcm, [i.as_numer_denom()[1] for _, i in residueterms], 1)
        # 计算 u 作为 residueterms 中各项的乘积
        u = Mul(*[Pow(i, j*n) for i, j in residueterms])
        return (n, u)

    # 如果情况为 'tan'，则抛出未实现错误
    elif case == 'tan':
        raise NotImplementedError("The hypertangent case is "
        "not yet implemented for is_log_deriv_k_t_radical_in_field()")

    # 如果情况为 'other_linear' 或 'other_nonlinear'，则抛出值错误
    elif case in ('other_linear', 'other_nonlinear'):
        # 如果这些情况受到结构定理支持，将错误改为 NotImplementedError
        raise ValueError("The %s case is not supported in this function." % case)

    # 如果情况不是上述任何一种，则抛出值错误
    else:
        raise ValueError("case must be one of {'primitive', 'exp', 'tan', "
        "'base', 'auto'}, not %s" % case)

    # 计算公共分母作为 residueterms 中各项分母的最小公倍数
    common_denom = S.One * reduce(ilcm, [i.as_numer_denom()[1] for i in [j for _, j in
        residueterms]] + [n], 1)
    # 将 residueterms 中的每个项乘以公共分母
    residueterms = [(i, j * common_denom) for i, j in residueterms]
    # 计算 m 作为公共分母与 n 的整除结果
    m = common_denom // n
    # 如果公共分母不等于 n*m，则抛出值错误，验证是否精确除法
    if common_denom != n * m:
        raise ValueError("Inexact division")
    # 对 u 进行 m 次方并使用 cancel 函数处理
    u = cancel(u ** m * Mul(*[Pow(i, j) for i, j in residueterms]))

    return (common_denom, u)
```