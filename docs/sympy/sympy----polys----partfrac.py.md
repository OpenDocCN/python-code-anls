# `D:\src\scipysrc\sympy\sympy\polys\partfrac.py`

```
# 导入必要的模块和函数
from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public

# 使用装饰器将函数标记为支持并行执行和公共接口
@xthreaded
@public
def apart(f, x=None, full=False, **options):
    """
    计算有理函数的偏分数分解。

    给定有理函数 ``f``，计算其偏分数分解。有两种算法可选：一种基于未知系数法，
    另一种是 Bronstein 的完全偏分数分解算法。

    当 ``full=False`` 时选择未知系数法，该方法使用多项式因式分解（因此接受与
    factor 相同的选项）处理分母。默认情况下它工作在有理数域上，因此不支持具有
    非有理根的分母的分解（例如无理数、复数根），详见 factor 的选项。

    当设置 ``full=True`` 时可以选择 Bronstein 的算法，允许处理具有非有理根的
    分母。可以通过 ``doit()`` 获得人类可读的结果（见下面的示例）。

    Examples
    ========

    >>> from sympy.polys.partfrac import apart
    >>> from sympy.abc import x, y

    默认情况下使用未知系数法：

    >>> apart(y/(x + 2)/(x + 1), x)
    -y/(x + 2) + y/(x + 1)

    当分母的根不是有理数时，未知系数法无法提供结果：

    >>> apart(y/(x**2 + x + 1), x)
    y/(x**2 + x + 1)

    通过设置 ``full=True`` 可选择 Bronstein 的算法：

    >>> apart(y/(x**2 + x + 1), x, full=True)
    RootSum(_w**2 + _w + 1, Lambda(_a, (-2*_a*y/3 - y/3)/(-_a + x)))

    调用 ``doit()`` 可以得到人类可读的结果：

    >>> apart(y/(x**2 + x + 1), x, full=True).doit()
    (-y/3 - 2*y*(-1/2 - sqrt(3)*I/2)/3)/(x + 1/2 + sqrt(3)*I/2) + (-y/3 -
        2*y*(-1/2 + sqrt(3)*I/2)/3)/(x + 1/2 - sqrt(3)*I/2)

    See Also
    ========

    apart_list, assemble_partfrac_list
    """
    # 检查选项参数是否合法
    allowed_flags(options, [])

    # 将输入的函数转换为 SymPy 表达式
    f = sympify(f)

    # 如果输入是原子表达式，则直接返回
    if f.is_Atom:
        return f
    else:
        # 将函数 f 分离成分子 P 和分母 Q
        P, Q = f.as_numer_denom()

    # 复制选项，以确保不影响原始选项
    _options = options.copy()
    # 设置默认选项
    options = set_defaults(options, extension=True)

    # 尝试并行处理 P 和 Q，返回处理后的多项式元组及选项信息
    try:
        (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)
    # 处理多项式错误，捕获并处理 PolynomialError 异常
    except PolynomialError as msg:
        # 如果多项式是可交换的，重新抛出 PolynomialError 异常
        if f.is_commutative:
            raise PolynomialError(msg)
        # 处理非可交换的情况
        # 如果多项式是乘法形式
        if f.is_Mul:
            # 将多项式分解为可交换部分 c 和非可交换部分 nc
            c, nc = f.args_cnc(split_1=False)
            # 将非可交换部分 nc 组装成一个新的多项式对象
            nc = f.func(*nc)
            # 如果存在可交换部分 c，则对其进行局部分式分解
            if c:
                c = apart(f.func._from_args(c), x=x, full=full, **_options)
                return c * nc
            else:
                return nc
        # 如果多项式是加法形式
        elif f.is_Add:
            c = []
            nc = []
            # 遍历多项式的每一个子项
            for i in f.args:
                # 如果子项是可交换的，则加入可交换列表 c
                if i.is_commutative:
                    c.append(i)
                else:
                    # 尝试对非可交换子项进行局部分式分解，如果不支持则保持原样
                    try:
                        nc.append(apart(i, x=x, full=full, **_options))
                    except NotImplementedError:
                        nc.append(i)
            # 对可交换部分 c 和非可交换部分 nc 分别进行局部分式分解，并相加
            return apart(f.func(*c), x=x, full=full, **_options) + f.func(*nc)
        else:
            # 对多项式进行先序遍历，尝试对每个子表达式进行局部分式分解
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                try:
                    # 尝试局部分式分解，并记录替换对
                    reps.append((e, apart(e, x=x, full=full, **_options)))
                    pot.skip()  # 标记当前处理成功
                except NotImplementedError:
                    pass
            # 使用记录的替换对对多项式进行替换
            return f.xreplace(dict(reps))

    # 如果多项式是多变量的
    if P.is_multivariate:
        # 对多项式进行取消公因式处理
        fc = f.cancel()
        # 如果取消公因式后与原多项式不同，则再次尝试局部分式分解
        if fc != f:
            return apart(fc, x=x, full=full, **_options)

        # 抛出未实现异常，因为不支持多元局部分式分解
        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    # 对多项式 P 和 Q 进行取消公因式处理，获取最简形式
    common, P, Q = P.cancel(Q)

    # 将多项式 P 除以 Q，得到商和余式
    poly, P = P.div(Q, auto=True)
    # 清除分母并返回新的 P 和 Q
    P, Q = P.rat_clear_denoms(Q)

    # 如果余式 Q 的次数小于等于 1，则直接进行局部分式分解
    if Q.degree() <= 1:
        partial = P / Q
    else:
        # 如果需要完全展开，则进行完全分解
        if not full:
            partial = apart_undetermined_coeffs(P, Q)
        else:
            partial = apart_full_decomposition(P, Q)

    # 初始化结果项
    terms = S.Zero

    # 遍历局部分式分解后的每一个项
    for term in Add.make_args(partial):
        # 如果项包含 RootSum，则直接添加到结果中
        if term.has(RootSum):
            terms += term
        else:
            # 对非 RootSum 的项进行因式分解，并添加到结果中
            terms += factor(term)

    # 返回最终的局部分式分解结果
    return common * (poly.as_expr() + terms)
# 使用未定系数法进行部分分式分解
def apart_undetermined_coeffs(P, Q):
    """Partial fractions via method of undetermined coefficients. """
    # 创建符号生成器X，用于生成虚拟变量
    X = numbered_symbols(cls=Dummy)
    # 初始化部分分式列表和符号列表
    partial, symbols = [], []

    # 对Q进行因式分解，返回(f, k)元组列表
    _, factors = Q.factor_list()

    # 遍历每个因子及其重数
    for f, k in factors:
        # 获取因子的次数n和Q的副本q
        n, q = f.degree(), Q

        # 迭代生成k个未知系数列表，同时更新q为q除以f的商
        for i in range(1, k + 1):
            coeffs, q = take(X, n), q.quo(f)
            partial.append((coeffs, q, f, i))
            symbols.extend(coeffs)

    # 将symbols注入Q的域中，构建域dom
    dom = Q.get_domain().inject(*symbols)
    # 用Q的生成元和域dom创建多项式F，初始化为0
    F = Poly(0, Q.gen, domain=dom)

    # 遍历partial列表，为每个未知系数列表coeffs创建多项式h，并与q相乘后加到F上
    for i, (coeffs, q, f, k) in enumerate(partial):
        h = Poly(coeffs, Q.gen, domain=dom)
        partial[i] = (h, f, k)
        q = q.set_domain(dom)
        F += h*q

    # 初始化系统和结果列表
    system, result = [], S.Zero

    # 遍历F中的每个(k,)项，将其系数与P的第k项做差，加到系统列表中
    for (k,), coeff in F.terms():
        system.append(coeff - P.nth(k))

    # 导入solve函数，解系统方程得到solution
    from sympy.solvers import solve
    solution = solve(system, symbols)

    # 对每个partial中的多项式h，用解solution替换未知系数，然后加到结果result中
    for h, f, k in partial:
        h = h.as_expr().subs(solution)
        result += h/f.as_expr()**k

    # 返回最终结果result
    return result


# 对给定有理函数P/Q进行Bronstein完全部分分解
def apart_full_decomposition(P, Q):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    # 调用apart_list函数，对P/Q进行部分分式分解，并返回结果
    return assemble_partfrac_list(apart_list(P/Q, P.gens[0]))


# 公共函数：计算有理函数f的部分分式分解，返回结构化的结果
def apart_list(f, x=None, dummies=None, **options):
    """
    Compute partial fraction decomposition of a rational function
    and return the result in structured form.

    Given a rational function ``f`` compute the partial fraction decomposition
    of ``f``. Only Bronstein's full partial fraction decomposition algorithm
    is supported by this method. The return value is highly structured and
    perfectly suited for further algorithmic treatment rather than being
    human-readable. The function returns a tuple holding three elements:

    * The first item is the common coefficient, free of the variable `x` used
      for decomposition. (It is an element of the base field `K`.)

    * The second item is the polynomial part of the decomposition. This can be
      the zero polynomial. (It is an element of `K[x].`)
    """
    # 调用函数 `allowed_flags`，传入 `options` 和一个空列表作为参数
    allowed_flags(options, [])

    # 将输入的表达式 `f` 转换为 SymPy 符号对象
    f = sympify(f)

    # 如果 `f` 是原子（Atom）表达式，则直接返回 `f`
    if f.is_Atom:
        return f
    else:
        # 否则，将 `f` 分解为分子 `P` 和分母 `Q`
        P, Q = f.as_numer_denom()

    # 将选项设置为默认值，其中扩展(extension)选项设置为True，并返回更新后的选项和新的分子分母对
    options = set_defaults(options, extension=True)
    (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)
    # 如果多元分式部分被标记为多变量，则抛出未实现的错误
    if P.is_multivariate:
        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    # 对多项式 P 和 Q 进行通分和约简，返回约简后的多项式和通分后的多项式
    common, P, Q = P.cancel(Q)

    # 将 P 除以 Q，返回商和余式，自动选择通分的方式
    poly, P = P.div(Q, auto=True)
    
    # 清除 P 和 Q 的有理数分母，返回清除分母后的 P 和 Q
    P, Q = P.rat_clear_denoms(Q)

    # 将多项式部分赋值给 polypart
    polypart = poly

    # 如果未提供虚拟变量生成器函数，则定义一个生成器函数以生成虚拟变量名
    if dummies is None:
        def dummies(name):
            d = Dummy(name)
            while True:
                yield d

        # 使用生成器函数创建以"w"为前缀的虚拟变量名生成器
        dummies = dummies("w")

    # 对 P 和 Q 进行全分解的部分分式分解，使用给定的虚拟变量名生成器
    rationalpart = apart_list_full_decomposition(P, Q, dummies)

    # 返回包含通分部分的常数、多项式部分和有理部分的元组
    return (common, polypart, rationalpart)
# 定义一个函数，实现 Bronstein 的完全偏分解算法，用于对有理函数进行处理。
# 函数接受三个参数 P, Q, dummygen，分别代表被分解的分子多项式 P，分母多项式 Q，以及一个用作虚拟生成器的变量 dummygen。
def apart_list_full_decomposition(P, Q, dummygen):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    # 保存 P, Q, P 的生成器和空列表 U 的原始值
    P_orig, Q_orig, x, U = P, Q, P.gen, []

    # 定义一个函数 u(x)，并将其赋值给 u
    u = Function('u')(x)
    # 创建一个虚拟变量 a，并赋值给 a
    a = Dummy('a')

    # 初始化一个空列表 partial
    partial = []

    # 遍历 Q 的平方因式列表，包括所有项
    for d, n in Q.sqf_list_include(all=True):
        # 将 d 转换为表达式，并赋值给 b
        b = d.as_expr()
        # 计算 u 对 x 的 (n-1) 阶导数，并将结果添加到 U 列表中
        U += [ u.diff(x, n - 1) ]

        # 计算 h = P_orig / (Q_orig 除以 d**n) / u**n，并简化
        h = cancel(P_orig / Q_orig.quo(d**n)) / u**n

        # 初始化 H 和 subs 列表
        H, subs = [h], []

        # 计算 H 列表中各项的导数，并将结果添加到 subs 列表中
        for j in range(1, n):
            H += [ H[-1].diff(x) / j ]

        # 遍历每个导数项，更新 P 和 Q，并进行一些替换操作
        for j in range(1, n + 1):
            subs += [ (U[j - 1], b.diff(x, j) / j) ]

        # 遍历每个导数项，将 H[j] 简化为 P 和 Q 的形式，并进行约分
        for j in range(0, n):
            P, Q = cancel(H[j]).as_numer_denom()

            # 根据 subs 列表对 P 进行替换操作
            for i in range(0, j + 1):
                P = P.subs(*subs[j - i])

            # 根据 subs 列表对 Q 进行替换操作
            Q = Q.subs(*subs[0])

            # 将 P 和 Q 转换为多项式对象
            P = Poly(P, x)
            Q = Poly(Q, x)

            # 计算 P 和 d 的最大公约数，并计算出 D = d / G
            G = P.gcd(d)
            D = d.quo(G)

            # 计算 Q 的半扩展最大公因式
            B, g = Q.half_gcdex(D)
            # 计算 b = (P * B / g) % D
            b = (P * B.quo(g)).rem(D)

            # 在 D 中用 dummygen 的下一个元素进行替换，并创建 Lambda 表达式
            Dw = D.subs(x, next(dummygen))
            numer = Lambda(a, b.as_expr().subs(x, a))
            denom = Lambda(a, (x - a))
            exponent = n - j

            # 将结果以元组形式添加到 partial 列表中
            partial.append((Dw, numer, denom, exponent))

    # 返回 partial 列表作为函数的结果
    return partial


# 标记函数 assemble_partfrac_list 为公共函数
@public
def assemble_partfrac_list(partial_list):
    r"""Reassemble a full partial fraction decomposition
    from a structured result obtained by the function ``apart_list``.

    Examples
    ========

    This example is taken from Bronstein's original paper:

    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list
    >>> from sympy.abc import x

    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    If we happen to know some roots we can provide them easily inside the structure:

    >>> pfd = apart_list(2/(x**2-2))
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w**2 - 2, _w, domain='ZZ'),
    Lambda(_a, _a/2),
    Lambda(_a, -_a + x),
    1)])

    >>> pfda = assemble_partfrac_list(pfd)
    >>> pfda
    RootSum(_w**2 - 2, Lambda(_a, _a/(-_a + x)))/2

    >>> pfda.doit()
    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))


    """
    # 函数重新组装一个完整的偏分解列表，根据 apart_list 函数的结构化结果得到。
    # 返回重新组装后的偏分解列表作为函数的结果
    return partial_list
    # 导入 sympy 库中的 Dummy, Poly, Lambda, sqrt 函数
    from sympy import Dummy, Poly, Lambda, sqrt
    
    # 创建一个虚拟符号 a
    a = Dummy("a")
    
    # 定义一个包含三个元素的元组 pfd
    pfd = (
        1,                                  # 第一个元素为常数 1
        Poly(0, x, domain='ZZ'),            # 第二个元素为一个多项式对象，多项式系数为 0，定义域为整数环 'ZZ'
        [                                   # 第三个元素为一个列表，包含以下内容：
            (                               # 元组内部的第一个元素包含以下内容：
                [sqrt(2), -sqrt(2)],         # - 根号 2 和负根号 2 的列表
                Lambda(a, a/2),              # - Lambda 函数，参数为 a，函数体为 a/2
                Lambda(a, -a + x),           # - Lambda 函数，参数为 a，函数体为 -a + x
                1                           # - 常数 1
            )
        ]
    )
    
    # 调用 assemble_partfrac_list 函数处理 pfd 变量，并返回结果
    assemble_partfrac_list(pfd)
    
    # 返回一个表达式，结果如下：
    # -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))
    
    # 查看相关函数 apart 和 apart_list
    # See Also
    # ========
    # apart, apart_list
```