# `D:\src\scipysrc\sympy\sympy\concrete\gosper.py`

```
# 导入必要的模块和函数
from sympy.core import S, Dummy, symbols
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.utilities.iterables import is_sequence

# 定义函数 gosper_normal，计算 Gosper 标准形式
def gosper_normal(f, g, n, polys=True):
    r"""
    Compute the Gosper's normal form of ``f`` and ``g``.

    Explanation
    ===========

    Given relatively prime univariate polynomials ``f`` and ``g``,
    rewrite their quotient to a normal form defined as follows:

    .. math::
        \frac{f(n)}{g(n)} = Z \cdot \frac{A(n) C(n+1)}{B(n) C(n)}

    where ``Z`` is an arbitrary constant and ``A``, ``B``, ``C`` are
    monic polynomials in ``n`` with the following properties:

    1. `\gcd(A(n), B(n+h)) = 1 \forall h \in \mathbb{N}`
    2. `\gcd(B(n), C(n+1)) = 1`
    3. `\gcd(A(n), C(n)) = 1`

    This normal form, or rational factorization in other words, is a
    crucial step in Gosper's algorithm and in solving of difference
    equations. It can be also used to decide if two hypergeometric
    terms are similar or not.

    This procedure will return a tuple containing elements of this
    factorization in the form ``(Z*A, B, C)``.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_normal
    >>> from sympy.abc import n

    >>> gosper_normal(4*n+5, 2*(4*n+1)*(2*n+3), n, polys=False)
    (1/4, n + 3/2, n + 1/4)

    """
    # 将输入的多项式 f 和 g 转换为并行多项式 p 和 q
    (p, q), opt = parallel_poly_from_expr((f, g), n, field=True, extension=True)

    # 获取 p 和 q 的最高系数和首项单项式
    a, A = p.LC(), p.monic()
    b, B = q.LC(), q.monic()

    # 初始化 C 和 Z
    C, Z = A.one, a/b

    # 创建虚拟变量 h
    h = Dummy('h')

    # 创建多项式 D = n + h
    D = Poly(n + h, n, h, domain=opt.domain)

    # 计算 A 与 B 组合的结果式 R
    R = A.resultant(B.compose(D))

    # 找到 R 的所有非负整数根
    roots = {r for r in R.ground_roots().keys() if r.is_Integer and r >= 0}

    # 对根进行排序并处理
    for i in sorted(roots):
        # 计算 A 与 B.shift(+i) 的最大公因数
        d = A.gcd(B.shift(+i))

        # 更新 A 和 B
        A = A.quo(d)
        B = B.quo(d.shift(-i))

        # 更新 C
        for j in range(1, i + 1):
            C *= d.shift(-j)

    # 将 A 乘以 Z
    A = A.mul_ground(Z)

    # 如果不需要多项式形式，则转换为表达式形式
    if not polys:
        A = A.as_expr()
        B = B.as_expr()
        C = C.as_expr()

    # 返回结果元组 (A, B, C)
    return A, B, C


# 定义函数 gosper_term，计算 Gosper 的超几何项
def gosper_term(f, n):
    r"""
    Compute Gosper's hypergeometric term for ``f``.

    Explanation
    ===========

    Suppose ``f`` is a hypergeometric term such that:

    .. math::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f_k` does not depend on `n`. Returns a hypergeometric
    term `g_n` such that `g_{n+1} - g_n = f_n`.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_term
    >>> from sympy import factorial
    >>> from sympy.abc import n

    >>> gosper_term((4*n + 1)*factorial(n)/factorial(2*n + 1), n)
    (-n - 1/2)/(n + 1/4)

    """
    # 导入 hypersimp 函数
    from sympy.simplify import hypersimp

    # 对输入的 f 进行超几何简化
    r = hypersimp(f, n)

    # 如果简化结果为 None，则 f 不是超几何项
    if r is None:
        return None

    # 将简化后的结果 r 拆分为分子 p 和分母 q
    p, q = r.as_numer_denom()

    # 调用 gosper_normal 函数，获取 A, B, C
    A, B, C = gosper_normal(p, q, n)

    # 对 B 进行移位处理
    B = B.shift(-1)

    # 获取 A, B, C 的次数
    N = S(A.degree())
    M = S(B.degree())
    K = S(C.degree())

    # 如果 A 和 B 的次数不同，或者 A 和 B 的首项系数不同，则返回 None
    if (N != M) or (A.LC() != B.LC()):
        D = {K - max(N, M)}
    elif not N:
        # 如果 N 为 0，设置 D 为 {K - N + 1, 0}
        D = {K - N + 1, S.Zero}
    else:
        # 否则，设置 D 为 {K - N + 1, (B.nth(N - 1) - A.nth(N - 1))/A.LC()}
        D = {K - N + 1, (B.nth(N - 1) - A.nth(N - 1))/A.LC()}

    for d in set(D):
        # 遍历集合 D 中的每个元素 d
        if not d.is_Integer or d < 0:
            # 如果 d 不是整数或者小于 0，则从集合 D 中移除 d
            D.remove(d)

    if not D:
        # 如果集合 D 为空集
        return None    # 'f(n)' 不是 Gosper 可求和的

    d = max(D)

    coeffs = symbols('c:%s' % (d + 1), cls=Dummy)
    # 生成长度为 d+1 的符号列表 coeffs

    domain = A.get_domain().inject(*coeffs)
    # 在 A 的定义域中注入符号 coeffs

    x = Poly(coeffs, n, domain=domain)
    # 使用 coeffs 创建关于 n 的多项式 x

    H = A*x.shift(1) - B*x - C
    # 计算 H = A*x(n+1) - B*x(n) - C

    from sympy.solvers.solvers import solve
    solution = solve(H.coeffs(), coeffs)
    # 求解多项式 H 的系数对 coeffs 的解

    if solution is None:
        # 如果未找到解
        return None    # 'f(n)' 不是 Gosper 可求和的

    x = x.as_expr().subs(solution)
    # 使用解 solution 替换 x 中的系数

    for coeff in coeffs:
        # 遍历 coeffs 中的每个系数 coeff
        if coeff not in solution:
            # 如果 coeff 不在解 solution 中
            x = x.subs(coeff, 0)
            # 将 x 中的 coeff 替换为 0

    if x.is_zero:
        # 如果 x 等于零
        return None    # 'f(n)' 不是 Gosper 可求和的
    else:
        return B.as_expr()*x/C.as_expr()
        # 返回 B 的表达式乘以 x 除以 C 的表达式的结果
# 利用 Gosper 的超几何级数求和算法，计算给定超几何项 f 的求和结果。
def gosper_sum(f, k):
    r"""
    Gosper's hypergeometric summation algorithm.

    Explanation
    ===========

    Given a hypergeometric term ``f`` such that:

    .. math ::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f(n)` does not depend on `n`, returns `g_{n} - g(0)` where
    `g_{n+1} - g_n = f_n`, or ``None`` if `s_n` cannot be expressed
    in closed form as a sum of hypergeometric terms.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_sum
    >>> from sympy import factorial
    >>> from sympy.abc import n, k

    >>> f = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    >>> gosper_sum(f, (k, 0, n))
    (-factorial(n) + 2*factorial(2*n + 1))/factorial(2*n + 1)
    >>> _.subs(n, 2) == sum(f.subs(k, i) for i in [0, 1, 2])
    True
    >>> gosper_sum(f, (k, 3, n))
    (-60*factorial(n) + factorial(2*n + 1))/(60*factorial(2*n + 1))
    >>> _.subs(n, 5) == sum(f.subs(k, i) for i in [3, 4, 5])
    True

    References
    ==========

    .. [1] Marko Petkovsek, Herbert S. Wilf, Doron Zeilberger, A = B,
           AK Peters, Ltd., Wellesley, MA, USA, 1997, pp. 73--100

    """
    # 判断 k 是否为序列类型，若是，则解析出 k, a, b 的值；否则标记为不定积分
    indefinite = False
    if is_sequence(k):
        k, a, b = k
    else:
        indefinite = True

    # 计算 gosper_term(f, k)，得到 g_k
    g = gosper_term(f, k)

    # 如果 g_k 为 None，则返回 None
    if g is None:
        return None

    # 根据不定积分标记，计算相应的结果
    if indefinite:
        result = f*g
    else:
        result = (f*(g + 1)).subs(k, b) - (f*g).subs(k, a)

        # 如果结果为 NaN，则尝试计算极限值
        if result is S.NaN:
            try:
                result = (f*(g + 1)).limit(k, b) - (f*g).limit(k, a)
            except NotImplementedError:
                result = None

    # 对结果进行因式分解处理后返回
    return factor(result)
```