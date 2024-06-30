# `D:\src\scipysrc\sympy\sympy\polys\fglmtools.py`

```
"""Implementation of matrix FGLM Groebner basis conversion algorithm. """

# 导入所需模块和函数
from sympy.polys.monomials import monomial_mul, monomial_div

# 定义函数 matrix_fglm，实现 FGLM 矩阵格罗布纳基变换算法
def matrix_fglm(F, ring, O_to):
    """
    Converts the reduced Groebner basis ``F`` of a zero-dimensional
    ideal w.r.t. ``O_from`` to a reduced Groebner basis
    w.r.t. ``O_to``.

    References
    ==========

    .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
           Computation of Zero-dimensional Groebner Bases by Change of
           Ordering
    """

    # 获取环的基本属性
    domain = ring.domain
    ngens = ring.ngens

    # 使用新的排序创建环 ring_to
    ring_to = ring.clone(order=O_to)

    # 计算旧基底的表示矩阵 M
    old_basis = _basis(F, ring)
    M = _representing_matrices(old_basis, F, ring)

    # 初始化数据结构 S, V, G, P 和 L
    S = [ring.zero_monom]
    V = [[domain.one] + [domain.zero] * (len(old_basis) - 1)]
    G = []
    L = [(i, 0) for i in range(ngens)]  # (i, j) corresponds to x_i * S[j]
    L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)
    t = L.pop()

    # 初始化单位矩阵 P
    P = _identity_matrix(len(old_basis), domain)

    # 开始主循环
    while True:
        s = len(S)
        v = _matrix_mul(M[t[0]], V[t[1]])
        _lambda = _matrix_mul(P, v)

        # 检查线性组合是否为零向量
        if all(_lambda[i] == domain.zero for i in range(s, len(old_basis))):
            # 存在向量 v 可以被 V 的线性组合表示
            lt = ring.term_new(_incr_k(S[t[1]], t[0]), domain.one)
            rest = ring.from_dict({S[i]: _lambda[i] for i in range(s)})

            # 构造新的多项式 g，并加入到 G 中
            g = (lt - rest).set_ring(ring_to)
            if g:
                G.append(g)
        else:
            # v 与 V 线性无关
            P = _update(s, _lambda, P)
            S.append(_incr_k(S[t[1]], t[0]))
            V.append(v)

            # 更新列表 L
            L.extend([(i, s) for i in range(ngens)])
            L = list(set(L))
            L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)

        # 根据条件更新列表 L
        L = [(k, l) for (k, l) in L if all(monomial_div(_incr_k(S[l], k), g.LM) is None for g in G)]

        # 如果 L 为空，则返回标准化的 G
        if not L:
            G = [ g.monic() for g in G ]
            return sorted(G, key=lambda g: O_to(g.LM), reverse=True)

        # 选择下一个 t
        t = L.pop()


# 辅助函数，将 m 中第 k 个元素增加 1
def _incr_k(m, k):
    return tuple(list(m[:k]) + [m[k] + 1] + list(m[k + 1:]))


# 辅助函数，生成域 domain 上的 n 阶单位矩阵
def _identity_matrix(n, domain):
    M = [[domain.zero]*n for _ in range(n)]

    for i in range(n):
        M[i][i] = domain.one

    return M


# 辅助函数，计算矩阵 M 与向量 v 的乘积
def _matrix_mul(M, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in M]


# 辅助函数，更新矩阵 P，使得 P v = e_s
def _update(s, _lambda, P):
    k = min(j for j in range(s, len(_lambda)) if _lambda[j] != 0)

    for r in range(len(_lambda)):
        if r != k:
            P[r] = [P[r][j] - (P[k][j] * _lambda[r]) / _lambda[k] for j in range(len(P[r]))]

    P[k] = [P[k][j] / _lambda[k] for j in range(len(P[k]))]
    P[k], P[s] = P[s], P[k]

    return P


# 辅助函数，计算基底 basis 与集合 G 之间的表示矩阵
def _representing_matrices(basis, G, ring):
    r"""
    Compute the matrices corresponding to the linear maps `m \mapsto
    x_i m` for all variables `x_i`.
    """

# 获取环的定义域
domain = ring.domain
# 变量的数量减一
u = ring.ngens - 1

# 定义一个函数，返回表示第 i 个变量的元组
def var(i):
    return tuple([0] * i + [1] + [0] * (u - i))

# 定义一个函数，返回一个表示给定输入 m 的线性映射的矩阵
def representing_matrix(m):
    # 创建一个空的矩阵 M，维度为基向量的数量乘以基向量的数量
    M = [[domain.zero] * len(basis) for _ in range(len(basis))]

    # 遍历基向量集合
    for i, v in enumerate(basis):
        # 计算 m 与当前基向量 v 的乘积，然后对环 G 取余
        r = ring.term_new(monomial_mul(m, v), domain.one).rem(G)

        # 遍历余项的每一项，包括单项式和系数
        for monom, coeff in r.terms():
            # 获取单项式在基向量列表中的索引，并将系数放入矩阵 M 的相应位置
            j = basis.index(monom)
            M[j][i] = coeff

    return M

# 返回一个列表，其中每个元素是一个表示对应变量 x_i 的线性映射的矩阵
return [representing_matrix(var(i)) for i in range(u + 1)]
# 计算一组单项式列表，这些单项式在与 ``G`` 的首单项式关于 ``O`` 的除法中不可被整除。这些单项式是 `K[X_1, \ldots, X_n]/(G)` 的基础。
def _basis(G, ring):
    # 获取环的排序规则
    order = ring.order

    # 提取所有输入多项式 `G` 的首单项式
    leading_monomials = [g.LM for g in G]
    
    # 初始化候选单项式列表，包含零单项式
    candidates = [ring.zero_monom]
    
    # 初始化基础单项式列表
    basis = []

    # 循环直到候选单项式列表为空
    while candidates:
        # 弹出一个候选单项式并添加到基础单项式列表中
        t = candidates.pop()
        basis.append(t)

        # 生成新的候选单项式列表，这些单项式在所有 `leading_monomials` 中的首单项式除法中都不可整除
        new_candidates = [_incr_k(t, k) for k in range(ring.ngens)
                          if all(monomial_div(_incr_k(t, k), lmg) is None
                                 for lmg in leading_monomials)]
        
        # 将新的候选单项式列表加入到候选单项式列表中，并按照 `order` 规则进行降序排序
        candidates.extend(new_candidates)
        candidates.sort(key=order, reverse=True)

    # 去重基础单项式列表并按照 `order` 规则进行排序
    basis = list(set(basis))
    return sorted(basis, key=order)
```