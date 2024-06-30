# `D:\src\scipysrc\sympy\sympy\polys\dispersion.py`

```
    # 导入 SymPy 的 S 和 Poly 类
    from sympy.core import S
    from sympy.polys import Poly

    # 定义计算两个多项式的离散集合的函数
    def dispersionset(p, q=None, *gens, **args):
        r"""Compute the *dispersion set* of two polynomials.

        For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
        and `\deg g > 0` the dispersion set `\operatorname{J}(f, g)` is defined as:

        .. math::
            \operatorname{J}(f, g)
            & := \{a \in \mathbb{N}_0 | \gcd(f(x), g(x+a)) \neq 1\} \\
            &  = \{a \in \mathbb{N}_0 | \deg \gcd(f(x), g(x+a)) \geq 1\}

        For a single polynomial one defines `\operatorname{J}(f) := \operatorname{J}(f, f)`.

        Examples
        ========

        >>> from sympy import poly
        >>> from sympy.polys.dispersion import dispersion, dispersionset
        >>> from sympy.abc import x

        Dispersion set and dispersion of a simple polynomial:

        >>> fp = poly((x - 3)*(x + 3), x)
        >>> sorted(dispersionset(fp))
        [0, 6]
        >>> dispersion(fp)
        6

        Note that the definition of the dispersion is not symmetric:

        >>> fp = poly(x**4 - 3*x**2 + 1, x)
        >>> gp = fp.shift(-3)
        >>> sorted(dispersionset(fp, gp))
        [2, 3, 4]
        >>> dispersion(fp, gp)
        4
        >>> sorted(dispersionset(gp, fp))
        []
        >>> dispersion(gp, fp)
        -oo

        Computing the dispersion also works over field extensions:

        >>> from sympy import sqrt
        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
        >>> sorted(dispersionset(fp, gp))
        [2]
        >>> sorted(dispersionset(gp, fp))
        [1, 4]

        We can even perform the computations for polynomials
        having symbolic coefficients:

        >>> from sympy.abc import a
        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
        >>> sorted(dispersionset(fp))
        [0, 1]

        See Also
        ========

        dispersion

        References
        ==========

        .. [1] [ManWright94]_
        .. [2] [Koepf98]_
        .. [3] [Abramov71]_
        .. [4] [Man93]_
        """
        # 检查输入是否合法，如果 q 未提供，则与 p 相同
        same = False if q is not None else True
        if same:
            q = p

        # 将 p 和 q 转换为多项式对象
        p = Poly(p, *gens, **args)
        q = Poly(q, *gens, **args)

        # 如果 p 或 q 不是单变量多项式，则引发异常
        if not p.is_univariate or not q.is_univariate:
            raise ValueError("Polynomials need to be univariate")

        # 检查两个多项式是否使用相同的生成器
        if not p.gen == q.gen:
            raise ValueError("Polynomials must have the same generator")
        gen = p.gen

        # 对于常数多项式，定义其离散集合为 {0}
        if p.degree() < 1 or q.degree() < 1:
            return {0}

        # 分解 p 和 q 在有理数域上的因式
        fp = p.factor_list()
        fq = q.factor_list() if not same else fp

        # 初始化离散集合为空集合
        J = set()
    # 对于 fp[1] 中的每个元组 (s, unused)，其中 s 是一个多项式对象
    for s, unused in fp[1]:
        # 对于 fq[1] 中的每个元组 (t, unused)，其中 t 是一个多项式对象
        for t, unused in fq[1]:
            # 计算多项式 s 和 t 的次数
            m = s.degree()
            n = t.degree()
            # 如果多项式的次数不相等，则继续下一个循环
            if n != m:
                continue
            # 获取多项式 s 和 t 的领头系数
            an = s.LC()
            bn = t.LC()
            # 如果 s 的领头系数减去 t 的领头系数不为零，则继续下一个循环
            if not (an - bn).is_zero:
                continue
            # 根据论文中的描述，下面的 s 和 t 在角色上是互换的
            # 这是为了与 W. Koepf 的书中的描述保持一致
            # 计算 gen**(m-1) 和 gen**(n-1) 的系数
            anm1 = s.coeff_monomial(gen**(m-1))
            bnm1 = t.coeff_monomial(gen**(n-1))
            # 计算 alpha，即 (anm1 - bnm1) / (n * bn)
            alpha = (anm1 - bnm1) / S(n * bn)
            # 如果 alpha 不是整数，则继续下一个循环
            if not alpha.is_integer:
                continue
            # 如果 alpha 小于 0 或者已经在集合 J 中，则继续下一个循环
            if alpha < 0 or alpha in J:
                continue
            # 如果 n 大于 1，并且 s 和 t.shift(alpha) 的差不为零，则继续下一个循环
            if n > 1 and not (s - t.shift(alpha)).is_zero:
                continue
            # 将 alpha 添加到集合 J 中
            J.add(alpha)

    # 返回集合 J
    return J
def dispersion(p, q=None, *gens, **args):
    r"""Compute the *dispersion* of polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion `\operatorname{dis}(f, g)` is defined as:

    .. math::
        \operatorname{dis}(f, g)
        & := \max\{ J(f,g) \cup \{0\} \} \\
        &  = \max\{ \{a \in \mathbb{N} | \gcd(f(x), g(x+a)) \neq 1\} \cup \{0\} \}

    and for a single polynomial `\operatorname{dis}(f) := \operatorname{dis}(f, f)`.
    Note that we make the definition `\max\{\} := -\infty`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    The maximum of an empty set is defined to be `-\infty`
    as seen in this example.

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersionset

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """

    # Compute the dispersion set J using dispersionset function
    J = dispersionset(p, q, *gens, **args)
    
    # Check if J is an empty set
    if not J:
        # If J is empty, set j to negative infinity as per definition
        j = S.NegativeInfinity
    else:
        # Otherwise, calculate the maximum value in J
        j = max(J)
    
    # Return the computed dispersion value
    return j
```