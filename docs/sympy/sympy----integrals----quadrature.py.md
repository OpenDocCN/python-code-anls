# `D:\src\scipysrc\sympy\sympy\integrals\quadrature.py`

```
# 从 sympy.core 模块导入 S（符号）、Dummy（虚拟变量）、pi（π常数）
# 从 sympy.functions.combinatorial.factorials 模块导入 factorial（阶乘函数）
# 从 sympy.functions.elementary.trigonometric 模块导入 sin（正弦函数）、cos（余弦函数）
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt（平方根函数）
# 从 sympy.functions.special.gamma_functions 模块导入 gamma（Gamma 函数）
# 从 sympy.polys.orthopolys 模块导入 legendre_poly（勒让德多项式）、laguerre_poly（拉盖尔多项式）、
# hermite_poly（埃尔米特多项式）、jacobi_poly（雅可比多项式）
# 从 sympy.polys.rootoftools 模块导入 RootOf（多项式的根对象）

def gauss_legendre(n, n_digits):
    r"""
    Computes the Gauss-Legendre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Legendre quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2}{\left(1-x_i^2\right) \left(P'_n(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_legendre
    >>> x, w = gauss_legendre(3, 5)
    >>> x
    [-0.7746, 0, 0.7746]
    >>> w
    [0.55556, 0.88889, 0.55556]
    >>> x, w = gauss_legendre(4, 5)
    >>> x
    [-0.86114, -0.33998, 0.33998, 0.86114]
    >>> w
    [0.34785, 0.65215, 0.65215, 0.34785]

    See Also
    ========

    gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule/legendre_rule.html
    """
    # 创建一个虚拟变量 x
    x = Dummy("x")
    # 计算勒让德多项式 P_n，并标记其为多项式类型
    p = legendre_poly(n, x, polys=True)
    # 计算勒让德多项式 P_n 对 x 的导数
    pd = p.diff(x)
    # 初始化空列表用于存放节点 xi 和权重 w
    xi = []
    w = []
    # 遍历勒让德多项式 P_n 的实根
    for r in p.real_roots():
        # 如果 r 是 RootOf 对象，将其转换为有理数形式
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        # 计算节点 xi，并将其保留指定位数的有效数字
        xi.append(r.n(n_digits))
        # 计算权重 w_i，根据公式：w_i = 2 / ((1 - x_i^2) * (P'_n(x_i))^2)，并保留指定位数的有效数字
        w.append((2/((1-r**2) * pd.subs(x, r)**2)).n(n_digits))
    # 返回节点和权重的列表
    return xi, w


def gauss_laguerre(n, n_digits):
    r"""
    Computes the Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_0^{\infty} e^{-x} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)


    The nodes `x_i` of an order `n` quadrature rule are the roots of `L_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{x_i}{(n+1)^2 \left(L_{n+1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (xi, w) : the ``xi`` and ``w`` are lists of points and weights as Floats.
              The points `x_i` and weights `w_i` are returned as ``(xi, w)``
              tuple of lists.
    """
    # 这里需要继续补充注释
    # 创建一个符号变量 `x`
    x = Dummy("x")
    # 计算 n 阶拉盖尔多项式，返回多项式对象
    p = laguerre_poly(n, x, polys=True)
    # 计算 n+1 阶拉盖尔多项式，返回多项式对象
    p1 = laguerre_poly(n+1, x, polys=True)
    # 初始化空列表，用于存储计算得到的积分点
    xi = []
    # 初始化空列表，用于存储计算得到的权重值
    w = []
    # 对 n 阶拉盖尔多项式的实根进行迭代
    for r in p.real_roots():
        # 如果实根是 RootOf 类型，则进行有理数评估
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        # 计算实根的数值并添加到 xi 列表中
        xi.append(r.n(n_digits))
        # 计算实根对应的权重值并添加到 w 列表中
        w.append((r/((n+1)**2 * p1.subs(x, r)**2)).n(n_digits))
    # 返回计算得到的积分点和权重值
    return xi, w
# 导入必要的库函数
from sympy import Dummy, hermite_poly, RootOf, S, factorial, sqrt, pi

def gauss_hermite(n, n_digits):
    r"""
    Computes the Gauss-Hermite quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Hermite quadrature approximates the integral:

    .. math::
        \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `H_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2^{n-1} n! \sqrt{\pi}}{n^2 \left(H_{n-1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_hermite
    >>> x, w = gauss_hermite(3, 5)
    >>> x
    [-1.2247, 0, 1.2247]
    >>> w
    [0.29541, 1.1816, 0.29541]

    >>> x, w = gauss_hermite(6, 5)
    >>> x
    [-2.3506, -1.3358, -0.43608, 0.43608, 1.3358, 2.3506]
    >>> w
    [0.00453, 0.15707, 0.72463, 0.72463, 0.15707, 0.00453]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_gen_laguerre, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss-Hermite_Quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_hermite_rule/gen_hermite_rule.html
    """
    # 使用符号变量 x
    x = Dummy("x")
    # 计算 Hermite 多项式 H_n(x)
    p = hermite_poly(n, x, polys=True)
    # 计算 Hermite 多项式 H_{n-1}(x)
    p1 = hermite_poly(n-1, x, polys=True)
    # 初始化空列表存储节点 x_i 和权重 w_i
    xi = []
    w = []
    # 遍历 H_n(x) 的实根
    for r in p.real_roots():
        # 如果是 RootOf 类型，计算有理数近似
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        # 计算节点 x_i 的值，并保留指定有效数字
        xi.append(r.n(n_digits))
        # 计算权重 w_i 的值
        w.append(((2**(n-1) * factorial(n) * sqrt(pi)) /
                 (n**2 * p1.subs(x, r)**2)).n(n_digits))
    # 返回节点列表 xi 和权重列表 w
    return xi, w
    # 定义一个虚拟变量 x
    x = Dummy("x")
    # 使用拉盖尔多项式生成器，生成拉盖尔多项式 p，用于计算拉盖尔积分的节点和权重
    p = laguerre_poly(n, x, alpha=alpha, polys=True)
    # 生成次数比 n 小 1 的拉盖尔多项式 p1，用于计算拉盖尔积分的节点和权重
    p1 = laguerre_poly(n-1, x, alpha=alpha, polys=True)
    # 生成次数比 n 小 1 的拉盖尔多项式 p2，用于计算拉盖尔积分的节点和权重
    p2 = laguerre_poly(n-1, x, alpha=alpha+1, polys=True)
    # 初始化节点列表 xi 和权重列表 w
    xi = []
    w = []
    # 遍历 p 的实根（拉盖尔多项式的根）
    for r in p.real_roots():
        # 如果 r 是 RootOf 类型，则将其有理评估为 10**(n_digits+2) 的倒数
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        # 将 r 的数值结果加入节点列表 xi
        xi.append(r.n(n_digits))
        # 计算权重并加入权重列表 w
        w.append((gamma(alpha+n) / (n*gamma(n)*p1.subs(x, r)*p2.subs(x, r))).n(n_digits))
    # 返回节点列表 xi 和权重列表 w 作为结果
    return xi, w
# 定义一个函数，计算第一类 Gauss-Chebyshev 乘积的节点和权重
def gauss_chebyshev_t(n, n_digits):
    xi = []  # 初始化存储节点的空列表
    w = []   # 初始化存储权重的空列表
    for i in range(1, n+1):  # 循环计算每个节点和权重
        # 计算节点 xi_i，根据公式 xi_i = cos((2*i-1)/(2*n)*π)，并将结果保留 n_digits 位小数
        xi.append((cos((2*i-S.One)/(2*n)*S.Pi)).n(n_digits))
        # 计算权重 w_i，根据公式 w_i = π/n，并将结果保留 n_digits 位小数
        w.append((S.Pi/n).n(n_digits))
    return xi, w  # 返回节点列表和权重列表的元组


# 定义一个函数，计算第二类 Gauss-Chebyshev 乘积的节点和权重
def gauss_chebyshev_u(n, n_digits):
    xi = []  # 初始化存储节点的空列表
    w = []   # 初始化存储权重的空列表
    for i in range(1, n+1):  # 循环计算每个节点和权重
        # 计算节点 xi_i，根据公式 xi_i = cos((2*i)/(2*n+2)*π)，并将结果保留 n_digits 位小数
        xi.append((cos((2*i)/(2*n+2)*S.Pi)).n(n_digits))
        # 计算权重 w_i，根据公式 w_i = π/(n+1) * sin^2(i/(n+1)*π)，并将结果保留 n_digits 位小数
        w.append((S.Pi/(n+1) * sin(i/(n+1)*S.Pi)**2).n(n_digits))
    return xi, w  # 返回节点列表和权重列表的元组
    # 初始化空列表，用于存储 Gauss-Chebyshev 积分的节点 xi
    xi = []
    # 初始化空列表，用于存储 Gauss-Chebyshev 积分的权重 w
    w = []
    
    # 使用循环生成 Gauss-Chebyshev 积分的节点和权重
    for i in range(1, n+1):
        # 计算第 i 个节点 xi_i，其中 n_digits 是精度参数
        xi.append((cos(i/(n+S.One)*S.Pi)).n(n_digits))
        # 计算第 i 个权重 w_i，其中 n_digits 是精度参数
        w.append((S.Pi/(n+S.One)*sin(i*S.Pi/(n+S.One))**2).n(n_digits))
    
    # 返回计算得到的节点列表 xi 和权重列表 w
    return xi, w
# 计算 Gauss-Jacobi 积分的节点和权重
def gauss_jacobi(n, alpha, beta, n_digits):
    r"""
    Computes the Gauss-Jacobi quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Jacobi quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^1 (1-x)^\alpha (1+x)^\beta f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `P^{(\alpha,\beta)}_n` and the weights `w_i` are given by:

    .. math::
        w_i = -\frac{2n+\alpha+\beta+2}{n+\alpha+\beta+1}
              \frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}
              {\Gamma(n+\alpha+\beta+1)(n+1)!}
              \frac{2^{\alpha+\beta}}{P'_n(x_i)
              P^{(\alpha,\beta)}_{n+1}(x_i)}

    Parameters
    ==========

    n : the order of quadrature

    alpha : the first parameter of the Jacobi Polynomial, `\alpha > -1`

    beta : the second parameter of the Jacobi Polynomial, `\beta > -1`

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_jacobi
    >>> x, w = gauss_jacobi(3, S.Half, -S.Half, 5)
    >>> x
    [-0.90097, -0.22252, 0.62349]
    >>> w
    [1.7063, 1.0973, 0.33795]

    >>> x, w = gauss_jacobi(6, 1, 1, 5)
    >>> x
    [-0.87174, -0.5917, -0.2093, 0.2093, 0.5917, 0.87174]
    >>> w
    [0.050584, 0.22169, 0.39439, 0.39439, 0.22169, 0.050584]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre,
    gauss_chebyshev_t, gauss_chebyshev_u, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_rule/jacobi_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gegenbauer_rule/gegenbauer_rule.html
    """
    # 使用 SymPy 的 Dummy 变量创建符号变量 x
    x = Dummy("x")
    # 计算 Jacobi 多项式 P^(alpha,beta)_n(x)
    p = jacobi_poly(n, alpha, beta, x, polys=True)
    # 计算 Jacobi 多项式 P'_n(x)
    pd = p.diff(x)
    # 计算 Jacobi 多项式 P^(alpha,beta)_{n+1}(x)
    pn = jacobi_poly(n+1, alpha, beta, x, polys=True)
    # 存储节点 x_i 和权重 w_i 的空列表
    xi = []
    w = []
    # 遍历 Jacobi 多项式的实根
    for r in p.real_roots():
        # 如果根是 RootOf 类型，转换为有理数并保留指定精度的近似值
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        # 将节点 x_i 加入列表，并保留指定精度的近似值
        xi.append(r.n(n_digits))
        # 计算权重 w_i 的表达式并加入列表，保留指定精度的近似值
        w.append((
            - (2*n+alpha+beta+2) / (n+alpha+beta+S.One) *
            (gamma(n+alpha+1)*gamma(n+beta+1)) /
            (gamma(n+alpha+beta+S.One)*gamma(n+2)) *
            2**(alpha+beta) / (pd.subs(x, r) * pn.subs(x, r))).n(n_digits))
    # 返回节点列表和权重列表的元组
    return xi, w


# 计算 Gauss-Lobatto 积分的节点和权重
def gauss_lobatto(n, n_digits):
    r"""
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)
    x = Dummy("x")
    # 创建一个符号变量 x

    p = legendre_poly(n-1, x, polys=True)
    # 计算 Legendre 多项式 P_{n-1}(x)，polys=True 表示返回多项式对象

    pd = p.diff(x)
    # 对 P_{n-1}(x) 求导数，得到 P'_{n-1}(x)

    xi = []
    # 初始化存储节点 x_i 的空列表

    w = []
    # 初始化存储权重 w_i 的空列表

    for r in pd.real_roots():
        # 遍历 P'_{n-1}(x) 的实根 r

        if isinstance(r, RootOf):
            # 如果 r 是 RootOf 对象，则通过有理估算计算其数值
            r = r.eval_rational(S.One/10**(n_digits+2))

        xi.append(r.n(n_digits))
        # 将节点 r 加入到节点列表 xi 中，精确到 n_digits 位小数

        w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))
        # 计算节点 r 对应的权重 w_i，并加入到权重列表 w 中，精确到 n_digits 位小数

    xi.insert(0, -1)
    # 将 -1 插入到节点列表 xi 的开头

    xi.append(1)
    # 将 1 添加到节点列表 xi 的末尾

    w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    # 计算 x=-1 处的权重 w_i 并插入到权重列表 w 的开头，精确到 n_digits 位小数

    w.append((S(2)/(n*(n-1))).n(n_digits))
    # 计算 x=1 处的权重 w_i 并添加到权重列表 w 的末尾，精确到 n_digits 位小数

    return xi, w
    # 返回节点列表 xi 和权重列表 w 的元组
```