# `D:\src\scipysrc\sympy\sympy\polys\distributedmodules.py`

```
# 导入 permutations 函数，用于生成可迭代对象的所有排列
from itertools import permutations

# 导入 monomial 相关函数
from sympy.polys.monomials import (
    monomial_mul, monomial_lcm, monomial_div, monomial_deg
)

# 导入多项式相关类和函数
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify

# 附加的单项式工具函数


def sdm_monomial_mul(M, X):
    """
    将表示 `K[X]` 单项式的元组 `X` 乘到表示 `F` 单项式的元组 `M` 上。

    Examples
    ========

    将 `xy^3` 乘到 `x f_1` 上得到 `x^2 y^3 f_1`:

    >>> from sympy.polys.distributedmodules import sdm_monomial_mul
    >>> sdm_monomial_mul((1, 1, 0), (1, 3))
    (1, 2, 3)
    """
    return (M[0],) + monomial_mul(X, M[1:])


def sdm_monomial_deg(M):
    """
    返回单项式 `M` 的总次数。

    Examples
    ========

    例如，`x^2 y f_5` 的总次数为 3:

    >>> from sympy.polys.distributedmodules import sdm_monomial_deg
    >>> sdm_monomial_deg((5, 2, 1))
    3
    """
    return monomial_deg(M[1:])


def sdm_monomial_lcm(A, B):
    """
    返回单项式 `A` 和 `B` 的最小公倍数。

    如果 `A = M e_j` 和 `B = N e_j`，其中 `M` 和 `N` 是多项式单项式，
    则返回 `\lcm(M, N) e_j`。注意 `A` 和 `B` 包含不同的单项式。

    否则结果是未定义的。

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_monomial_lcm
    >>> sdm_monomial_lcm((1, 2, 3), (1, 0, 5))
    (1, 2, 5)
    """
    return (A[0],) + monomial_lcm(A[1:], B[1:])


def sdm_monomial_divides(A, B):
    """
    判断单项式 `A` 是否整除 `B`。

    ```
    # 检查是否存在一个多项式单项式 X，使得 XA = B
    # 即检查给定的两个元组 A 和 B，是否满足条件：A 的每个元素都不大于 B 对应元素，并且 A 的第一个元素等于 B 的第一个元素
    
    return A[0] == B[0] and all(a <= b for a, b in zip(A[1:], B[1:]))
# 实际分布模块的代码。

# 返回多项式 `f` 的首项系数。
def sdm_LC(f, K):
    if not f:  # 如果 `f` 是空的，则返回零元素
        return K.zero
    else:
        return f[0][1]  # 否则返回首项的系数

# 将分布多项式转换为字典形式。
def sdm_to_dict(f):
    return dict(f)

# 根据字典 `d` 创建一个分布模块 `sdm`。
# 这里的 `O` 是要使用的单项式顺序。
def sdm_from_dict(d, O):
    return sdm_strip(sdm_sort(list(d.items()), O))

# 使用给定的单项式顺序 `O` 对 `f` 中的项进行排序。
def sdm_sort(f, O):
    return sorted(f, key=lambda term: O(term[0]), reverse=True)

# 从 `f` 中去除系数为零的项。
def sdm_strip(f):
    return [ (monom, coeff) for monom, coeff in f if coeff ]

# 在地方环境 `K[X]` 中，将两个模块元素 `f` 和 `g` 相加。
# 使用地方域 `K`，按照单项式顺序 `O` 进行排序。
def sdm_add(f, g, O, K):
    h = dict(f)

    for monom, c in g:
        if monom in h:
            coeff = h[monom] + c

            if not coeff:
                del h[monom]
            else:
                h[monom] = coeff
        else:
            h[monom] = c

    return sdm_from_dict(h, O)

# 返回多项式 `f` 的首项单项式。
# 仅在 `f \ne 0` 时有效。
def sdm_LM(f):
    return f[0][0]

# 返回多项式 `f` 的首项。
# 仅在 `f \ne 0` 时有效。
def sdm_LT(f):
    return f[0]

# 将分布模块 `f` 乘以单项式 `term`。
# 使用单项式顺序 `O`，地方域 `K`。
    # 将分布模块元素列表 f 按照多项式项 term 进行乘法运算。
    
    # 系数的乘法在有限域 K 上进行，多项式项按照排序规则 O 排序。
    
    # 如果 f 或者 c 为空，则返回空列表。
    # 否则：
    # - 如果 c 是单位元素（1），则对 f 中的每个元素 (f_M, f_c)，返回乘积 (sdm_monomial_mul(f_M, X), f_c)。
    # - 否则，对 f 中的每个元素 (f_M, f_c)，返回乘积 (sdm_monomial_mul(f_M, X), f_c * c)。
    
    def sdm_mul_term(f, term, O, K):
        X, c = term
        
        if not f or not c:
            return []
        else:
            if K.is_one(c):
                return [ (sdm_monomial_mul(f_M, X), f_c) for f_M, f_c in f ]
            else:
                return [ (sdm_monomial_mul(f_M, X), f_c * c) for f_M, f_c in f ]
# 返回一个空列表，表示零模块元素
def sdm_zero():
    return []


# 计算多项式 f 的次数
# 这是所有单项式次数的最大值
# 如果 f 是零，则无效
def sdm_deg(f):
    return max(sdm_monomial_deg(M[0]) for M in f)


# 从表达式向量创建一个分布模块（sdm）
# 系数在地域域 K 中创建，项按照单项式顺序 O 排序
# 其他参数传递给 polys 转换代码，可以用来指定生成器等
def sdm_from_vector(vec, O, K, **opts):
    dics, gens = parallel_dict_from_expr(sympify(vec), **opts)
    dic = {}
    for i, d in enumerate(dics):
        for k, v in d.items():
            dic[(i,) + k] = K.convert(v)
    return sdm_from_dict(dic, O)


# 将分布模块（sdm）转换为多项式表达式列表
# 多项式环的生成器通过 gens 指定，模块的秩通过 n 猜测或传递
# 地域域假定为 K
def sdm_to_vector(f, gens, K, n=None):
    dic = sdm_to_dict(f)
    dics = {}
    for k, v in dic.items():
        dics.setdefault(k[0], []).append((k[1:], v))
    n = n or len(dics)
    res = []
    for k in range(n):
        if k in dics:
            res.append(Poly(dict(dics[k]), gens=gens, domain=K).as_expr())
        else:
            res.append(S.Zero)
    return res


# 计算多项式 f 和 g 的广义 s-多项式
# 假定地域域为 K，并且单项式按照 O 排序
# 如果 f 或 g 是零，则无效
# 如果 f 和 g 的主导项涉及 F 的不同基础元素，则它们的 s-多项式定义为零
# 否则，它是 f 和 g 的某种线性组合，其中主导项相互抵消
# 参见 [SCA, defn 2.3.6] 获取详细信息
# 如果 phantom 不为 None，则应该是要在其上执行相同操作的模块元素对，此时返回两个结果
def sdm_spoly(f, g, O, K, phantom=None):
    # 如果输入的多项式列表 f 或 g 为空，返回零多项式
    if not f or not g:
        return sdm_zero()
    
    # 计算多项式 f 和 g 的主导单项式（Leading Monomial）
    LM1 = sdm_LM(f)
    LM2 = sdm_LM(g)
    
    # 如果 f 和 g 的主导单项式的首项系数不相等，返回零多项式
    if LM1[0] != LM2[0]:
        return sdm_zero()
    
    # 从主导单项式中去除首项系数，得到剩余的部分
    LM1 = LM1[1:]
    LM2 = LM2[1:]
    
    # 计算 f 和 g 主导单项式的最小公倍单项式（Least Common Multiple）
    lcm = monomial_lcm(LM1, LM2)
    
    # 计算将 lcm 分别除以 LM1 和 LM2 得到的单项式
    m1 = monomial_div(lcm, LM1)
    m2 = monomial_div(lcm, LM2)
    
    # 计算 f 和 g 的领导系数之商
    c = K.quo(-sdm_LC(f, K), sdm_LC(g, K))
    
    # 计算组合多项式 r1
    r1 = sdm_add(sdm_mul_term(f, (m1, K.one), O, K),
                 sdm_mul_term(g, (m2, c), O, K), O, K)
    
    # 如果没有给定 phantom 参数，返回 r1
    if phantom is None:
        return r1
    
    # 计算带有 phantom 参数的组合多项式 r2
    r2 = sdm_add(sdm_mul_term(phantom[0], (m1, K.one), O, K),
                 sdm_mul_term(phantom[1], (m2, c), O, K), O, K)
    
    # 返回 r1 和 r2
    return r1, r2
def sdm_nf_buchberger(f, G, O, K, phantom=None):
    r"""
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is the standard Buchberger algorithm for computing weak normal forms with
    respect to *global* monomial orders [SCA, algorithm 1.6.10].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    # Importing itertools repeat function for creating infinite iterators
    from itertools import repeat
    # Initializing h to be the input polynomial f
    h = f
    # Initializing T as a list containing elements of G
    T = list(G)
    # Checking if phantom variable is provided
    if phantom is not None:
        # Assigning phantom[0] to hp and converting phantom[1] to a list Tp
        hp = phantom[0]
        Tp = list(phantom[1])
        # Setting phantom to True indicating phantom variables are used
        phantom = True
    else:
        # Setting Tp to repeat an empty list, indicating no phantom variables
        Tp = repeat([])
        # Setting phantom to False indicating no phantom variables are used
        phantom = False
    # Looping until h is not zero
    while h:
        # Creating list Th with tuples (g, sdm_ecart(g), gp) for g in T and gp in Tp
        Th = [(g, sdm_ecart(g), gp) for g, gp in zip(T, Tp)
              if sdm_monomial_divides(sdm_LM(g), sdm_LM(h))]
        # If Th is empty, exit the loop
        if not Th:
            break
        # Finding the tuple with the minimum sdm_ecart(g)
        g, _, gp = min(Th, key=lambda x: x[1])
        # Comparing sdm_ecart(g) with sdm_ecart(h)
        if sdm_ecart(g) > sdm_ecart(h):
            # Adding h to T and hp to Tp if phantom variables are used
            T.append(h)
            if phantom:
                Tp.append(hp)
        # Updating h and hp by computing sdm_spoly(h, g, O, K, phantom=(hp, gp)) if phantom variables are used
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            # Updating h by computing sdm_spoly(h, g, O, K) if no phantom variables are used
            h = sdm_spoly(h, g, O, K)
    # Returning (h, hp) if phantom variables are used, otherwise returning h
    if phantom:
        return h, hp
    return h
    """
    # 导入repeat函数从itertools模块
    from itertools import repeat
    # 将参数f赋值给变量h
    h = f
    # 将G转换为列表，并赋值给变量T
    T = list(G)
    # 如果phantom不为None，则执行以下代码块
    if phantom is not None:
        # 使用phantom的第一个元素赋值给变量hp
        hp = phantom[0]
        # 将phantom的第二个元素转换为列表，并赋值给变量Tp
        Tp = list(phantom[1])
        # 将phantom设置为True
        phantom = True
    else:
        # 使用repeat函数创建一个空列表的迭代器，并赋值给变量Tp
        Tp = repeat([])
        # 将phantom设置为False
        phantom = False
    # 当h不为空时执行以下循环
    while h:
        # 尝试获取符合条件的(g, gp)对，使得sdm_monomial_divides(sdm_LM(g), sdm_LM(h))成立
        try:
            # 使用zip函数和生成器表达式获取下一个符合条件的(g, gp)对，并分别赋值给g和gp
            g, gp = next((g, gp) for g, gp in zip(T, Tp)
                         if sdm_monomial_divides(sdm_LM(g), sdm_LM(h)))
        # 如果StopIteration异常被触发则退出循环
        except StopIteration:
            break
        # 如果phantom为True，则执行以下代码块
        if phantom:
            # 使用sdm_spoly函数计算h和g的S多项式，并赋值给h和hp
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            # 使用sdm_spoly函数计算h和g的S多项式，并更新h的值
            h = sdm_spoly(h, g, O, K)
    # 如果phantom为True，则返回(h, hp)
    if phantom:
        return h, hp
    # 否则返回h
    return h
def sdm_nf_buchberger_reduced(f, G, O, K):
    r"""
    Compute a reduced normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    In contrast to weak normal forms, reduced normal forms *are* unique, but
    their computation is more expensive.

    This is the standard Buchberger algorithm for computing reduced normal forms
    with respect to *global* monomial orders [SCA, algorithm 1.6.11].

    The ``pantom`` option is not supported, so this normal form cannot be used
    as a normal form for the "extended" groebner algorithm.
    """
    h = sdm_zero()  # Initialize h as the zero element of the module
    g = f  # Set g to the input polynomial f
    while g:
        g = sdm_nf_buchberger(g, G, O, K)  # Compute normal form of g with respect to G and O
        if g:
            h = sdm_add(h, [sdm_LT(g)], O, K)  # Add the leading term of g to h
            g = g[1:]  # Remove the leading term from g
    return h  # Return the reduced normal form h


def sdm_groebner(G, NF, O, K, extended=False):
    """
    Compute a minimal standard basis of ``G`` with respect to order ``O``.

    The algorithm uses a normal form ``NF``, for example ``sdm_nf_mora``.
    The ground field is assumed to be ``K``, and monomials ordered according
    to ``O``.

    Let `N` denote the submodule generated by elements of `G`. A standard
    basis for `N` is a subset `S` of `N`, such that `in(S) = in(N)`, where for
    any subset `X` of `F`, `in(X)` denotes the submodule generated by the
    initial forms of elements of `X`. [SCA, defn 2.3.2]

    A standard basis is called minimal if no subset of it is a standard basis.

    One may show that standard bases are always generating sets.

    Minimal standard bases are not unique. This algorithm computes a
    deterministic result, depending on the particular order of `G`.

    If ``extended=True``, also compute the transition matrix from the initial
    generators to the groebner basis. That is, return a list of coefficient
    vectors, expressing the elements of the groebner basis in terms of the
    elements of ``G``.

    This functions implements the "sugar" strategy, see

    Giovini et al: "One sugar cube, please" OR Selection strategies in
    Buchberger algorithm.
    """

    # The critical pair set.
    # A critical pair is stored as (i, j, s, t) where (i, j) defines the pair
    # (by indexing S), s is the sugar of the pair, and t is the lcm of their
    # leading monomials.
    P = []  # Initialize an empty list for storing critical pairs

    # The eventual standard basis.
    S = []  # Initialize an empty list for the standard basis
    Sugars = []  # Initialize an empty list for storing sugars of S-polynomials

    def Ssugar(i, j):
        """Compute the sugar of the S-poly corresponding to (i, j)."""
        LMi = sdm_LM(S[i])  # Compute the leading monomial of S[i]
        LMj = sdm_LM(S[j])  # Compute the leading monomial of S[j]
        return max(Sugars[i] - sdm_monomial_deg(LMi),
                   Sugars[j] - sdm_monomial_deg(LMj)) \
            + sdm_monomial_deg(sdm_monomial_lcm(LMi, LMj))  # Compute and return the sugar of the S-polynomial

    ourkey = lambda p: (p[2], O(p[3]), p[1])  # Define a key function for sorting critical pairs
    def update(f, sugar, P):
        """Add f with sugar ``sugar`` to S, update P."""
        # 如果 f 为空，则直接返回 P
        if not f:
            return P
        # 计算当前 S 的长度作为新元素的索引，将 f 添加到 S 中
        k = len(S)
        S.append(f)
        # 将 sugar 添加到 Sugars 列表中，与新添加的 f 对应
        Sugars.append(sugar)

        # 计算 f 的主导单项式
        LMf = sdm_LM(f)

        def removethis(pair):
            # 解构 pair 中的元素
            i, j, s, t = pair
            # 如果 LMf 的第一个元素与 t 的第一个元素不相等，则返回 False
            if LMf[0] != t[0]:
                return False
            # 计算两个主导单项式的最小公倍单项式
            tik = sdm_monomial_lcm(LMf, sdm_LM(S[i]))
            tjk = sdm_monomial_lcm(LMf, sdm_LM(S[j]))
            # 如果条件符合，则返回 True，表示应该移除该 pair
            return tik != t and tjk != t and sdm_monomial_divides(tik, t) and \
                sdm_monomial_divides(tjk, t)

        # 应用链条件，更新 P
        P = [p for p in P if not removethis(p)]

        # 新的 pair 集合 N
        N = [(i, k, Ssugar(i, k), sdm_monomial_lcm(LMf, sdm_LM(S[i])))
             for i in range(k) if LMf[0] == sdm_LM(S[i])[0]]
        # TODO 是否应用乘积条件？
        N.sort(key=ourkey)
        remove = set()
        for i, p in enumerate(N):
            for j in range(i + 1, len(N)):
                if sdm_monomial_divides(p[3], N[j][3]):
                    remove.add(j)

        # TODO 是否应用归并排序？
        P.extend(reversed([p for i, p in enumerate(N) if i not in remove]))
        P.sort(key=ourkey, reverse=True)
        # 注意：逆向排序，因为我们希望从末尾弹出
        return P

    # 计算环中生成器的数量
    try:
        # 查找第一个非零向量，并取其第一个单项式
        # 环中生成器的数量比长度少一（因为第零个条目是模生成器）
        numgens = len(next(x[0] for x in G if x)[0]) - 1
    except StopIteration:
        # 如果 G 中没有非零元素...
        if extended:
            # 如果扩展标志为真，则返回空列表
            return [], []
        # 否则返回空列表
        return []

    # coefficients 列表将存储 S 中元素相对于初始生成器的表达式
    coefficients = []

    # 将所有 G 中的元素添加到 S 中
    for i, f in enumerate(G):
        # 更新 P，并根据需要将新的系数添加到 coefficients 中
        P = update(f, sdm_deg(f), P)
        if extended and f:
            coefficients.append(sdm_from_dict({(i,) + (0,)*numgens: K(1)}, O))

    # 现在执行 Buchberger 算法
    while P:
        i, j, s, t = P.pop()
        f, g = S[i], S[j]
        if extended:
            # 如果扩展标志为真，则计算 S 中两个元素的 S 多项式
            sp, coeff = sdm_spoly(f, g, O, K,
                                  phantom=(coefficients[i], coefficients[j]))
            # 使用 NF 函数计算 sp 的规范形式 h，同时更新 coefficients
            h, hcoeff = NF(sp, S, O, K, phantom=(coeff, coefficients))
            if h:
                coefficients.append(hcoeff)
        else:
            # 否则计算 S 多项式 sp，并使用 NF 函数计算其规范形式 h
            h = NF(sdm_spoly(f, g, O, K), S, O, K)
        # 更新 P
        P = update(h, Ssugar(i, j), P)

    # 最后对标准基进行互约化
    # （TODO：使用更好的数据结构）
    S = {(tuple(f), i) for i, f in enumerate(S)}
    for (a, ai), (b, bi) in permutations(S, 2):
        A = sdm_LM(a)
        B = sdm_LM(b)
        if sdm_monomial_divides(A, B) and (b, bi) in S and (a, ai) in S:
            S.remove((b, bi))
    # 对列表 S 中的每个元素 (f, i)，应用函数 sdm_LM(p[0]) 计算其排序依据 O(sdm_LM(p[0]))，
    # 将结果以降序排序，并将结果存储在列表 L 中，每个元素是形如 (list(f), i) 的元组
    L = sorted(((list(f), i) for f, i in S), key=lambda p: O(sdm_LM(p[0])), reverse=True)
    
    # 从排序后的列表 L 中提取每个元素的第一个元素 x[0]，构成新的列表 res
    res = [x[0] for x in L]
    
    # 如果 extended 参数为真，则同时返回 res 和根据排序后的索引提取的 coefficients 列表
    if extended:
        return res, [coefficients[i] for _, i in L]
    
    # 如果 extended 参数为假，则只返回 res
    return res
```