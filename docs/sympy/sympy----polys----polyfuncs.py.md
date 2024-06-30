# `D:\src\scipysrc\sympy\sympy\polys\polyfuncs.py`

```
# 导入所需的模块和函数
from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
    PolificationFailed, ComputationFailed,
    MultivariatePolynomialError, OptionError)
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
    symmetric_poly, interpolating_poly)
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public

# 将函数 symmetrize 声明为公共函数
@public
def symmetrize(F, *gens, **args):
    r"""
    Rewrite a polynomial in terms of elementary symmetric polynomials.

    A symmetric polynomial is a multivariate polynomial that remains invariant
    under any variable permutation, i.e., if `f = f(x_1, x_2, \dots, x_n)`,
    then `f = f(x_{i_1}, x_{i_2}, \dots, x_{i_n})`, where
    `(i_1, i_2, \dots, i_n)` is a permutation of `(1, 2, \dots, n)` (an
    element of the group `S_n`).

    Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that
    ``f = f1 + f2 + ... + fn``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import symmetrize
    >>> from sympy.abc import x, y

    >>> symmetrize(x**2 + y**2)
    (-2*x*y + (x + y)**2, 0)

    >>> symmetrize(x**2 + y**2, formal=True)
    (s1**2 - 2*s2, 0, [(s1, x + y), (s2, x*y)])

    >>> symmetrize(x**2 - y**2)
    (-2*x*y + (x + y)**2, -2*y**2)

    >>> symmetrize(x**2 - y**2, formal=True)
    (s1**2 - 2*s2, -2*y**2, [(s1, x + y), (s2, x*y)])

    """
    # 检查并允许指定的标志
    allowed_flags(args, ['formal', 'symbols'])

    # 检查输入是否可迭代
    iterable = True
    if not hasattr(F, '__iter__'):
        iterable = False
        F = [F]

    # 创建多项式环 R 和 F
    R, F = sring(F, *gens, **args)
    gens = R.symbols

    # 构建选项 opt
    opt = build_options(gens, args)
    symbols = opt.symbols
    symbols = [next(symbols) for i in range(len(gens))]

    # 初始化结果列表
    result = []

    # 对于每个多项式 f，进行对称化处理
    for f in F:
        p, r, m = f.symmetrize()
        result.append((p.as_expr(*symbols), r.as_expr(*gens)))

    # 构建多项式替换列表 polys
    polys = [(s, g.as_expr()) for s, (_, g) in zip(symbols, m)]

    # 如果不是形式化处理，则应用多项式替换
    if not opt.formal:
        for i, (sym, non_sym) in enumerate(result):
            result[i] = (sym.subs(polys), non_sym)

    # 如果输入不可迭代，则返回单个结果
    if not iterable:
        result, = result

    # 根据是否形式化处理返回结果
    if not opt.formal:
        return result
    else:
        if iterable:
            return result, polys
        else:
            return result + (polys,)


# 将函数 horner 声明为公共函数
@public
def horner(f, *gens, **args):
    """
    Rewrite a polynomial in Horner form.

    Among other applications, evaluation of a polynomial at a point is optimal
    when it is applied using the Horner scheme ([1]).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import horner
    >>> from sympy.abc import x, y, a, b, c, d, e

    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)
    x*(x*(x*(9*x + 8) + 7) + 6) + 5

    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    e + x*(d + x*(c + x*(a*x + b)))

    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y

    >>> horner(f, wrt=x)

    """
    # Horner形式的多项式重写函数

    # 确定参数中允许的标志
    allowed_flags(args, [])

    # 输出结果
    return Poly(f, *gens, **args).horner()
    x*(x*y*(4*y + 2) + y*(2*y + 1))
    """
    计算多项式表达式在 Horner 方案下的展开形式

    >>> horner(f, wrt=y)
    返回多项式在 Horner 方案下按照指定变量 y 展开的结果

    References
    ==========
    [1] - https://en.wikipedia.org/wiki/Horner_scheme

    """
    allowed_flags(args, [])
    """
    检查参数 args 是否包含合法的标志，此处为空列表表示不允许任何标志

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr
    """
    尝试将输入的表达式 f 转换为多项式 F，使用给定的生成器 gens 和参数 args，处理转换失败的异常情况并返回异常表达式

    form, gen = S.Zero, F.gen
    """
    初始化变量 form 为零，gen 为多项式 F 的生成器

    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form*gen + coeff
    else:
        F, gens = Poly(F, gen), gens[1:]
        """
        如果多项式 F 是单变量的，则遍历所有系数，使用 Horner 方案将其展开为形式 form
        否则，将 F 和生成器列表 gens 转换为多项式对象 Poly，并从 gens 中移除第一个生成器

        for coeff in F.all_coeffs():
            form = form*gen + horner(coeff, *gens, **args)
    """
    否则，对于 F 的所有系数，递归使用 Horner 方案展开每个系数，并将其结果加入 form 中

    return form
    """
    返回最终展开的多项式形式 form
    """
# 定义一个公共函数，用于根据给定的数据点构建插值多项式，以在点 x 处进行求值（x 可以是符号或数值）。
@public
def interpolate(data, x):
    """
    Construct an interpolating polynomial for the data points
    evaluated at point x (which can be symbolic or numeric).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import interpolate
    >>> from sympy.abc import a, b, x

    A list is interpreted as though it were paired with a range starting
    from 1:

    >>> interpolate([1, 4, 9, 16], x)
    x**2

    This can be made explicit by giving a list of coordinates:

    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)
    x**2

    The (x, y) coordinates can also be given as keys and values of a
    dictionary (and the points need not be equispaced):

    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)
    x**2 + 1
    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
    x**2 + 1

    If the interpolation is going to be used only once then the
    value of interest can be passed instead of passing a symbol:

    >>> interpolate([1, 4, 9], 5)
    25

    Symbolic coordinates are also supported:

    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
    [(1, a), (2, b), (3, -a + 2*b)]
    """
    # 获取数据点的数量
    n = len(data)

    # 如果数据是字典形式
    if isinstance(data, dict):
        # 如果 x 存在于字典中，直接返回对应的值
        if x in data:
            return S(data[x])
        # 否则将字典的键和值分别存储在 X 和 Y 中
        X, Y = list(zip(*data.items()))
    else:
        # 如果数据的第一个元素是元组
        if isinstance(data[0], tuple):
            # 将数据中的坐标分别存储在 X 和 Y 中
            X, Y = list(zip(*data))
            # 如果 x 存在于 X 中，则返回对应的 Y 值
            if x in X:
                return S(Y[X.index(x)])
        else:
            # 如果 x 在 1 到 n 之间，则返回对应索引处的值
            if x in range(1, n + 1):
                return S(data[x - 1])
            # 否则将数据转换为列表 Y，并创建索引范围 1 到 n 的列表 X
            Y = list(data)
            X = list(range(1, n + 1))

    try:
        # 尝试生成并展开插值多项式
        return interpolating_poly(n, x, X, Y).expand()
    except ValueError:
        # 如果出现 ValueError，则创建一个虚拟符号 d，并用 x 替换插值多项式中的 d
        d = Dummy()
        return interpolating_poly(n, d, X, Y).expand().subs(d, x)


# 定义一个公共函数，用于返回有理插值，其中数据点属于任何整数域，num 为有理函数的分子的最高次数，默认符号为 x
@public
def rational_interpolate(data, degnum, X=symbols('x')):
    """
    Returns a rational interpolation, where the data points are element of
    any integral domain.

    The first argument  contains the data (as a list of coordinates). The
    ``degnum`` argument is the degree in the numerator of the rational
    function. Setting it too high will decrease the maximal degree in the
    denominator for the same amount of data.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import rational_interpolate

    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]
    >>> rational_interpolate(data, 2)
    (105*x**2 - 525)/(x + 1)

    Values do not need to be integers:

    >>> from sympy import sympify
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")
    >>> rational_interpolate(zip(x, y), 2)
    (3*x**2 - 7*x + 2)/(x + 1)

    The symbol for the variable can be changed if needed:
    >>> from sympy import symbols
    >>> z = symbols('z')
    >>> rational_interpolate(data, 2, X=z)
    (105*z**2 - 525)/(z + 1)

    References
    ==========

    .. [1] Algorithm is adapted from:
           http://axiom-wiki.newsynthesis.org/RationalInterpolation

    """
    # 导入 sympy 库中的矩阵模块中的 ones 函数
    from sympy.matrices.dense import ones
    
    # 将 data 中的 x 和 y 数据分别解压为两个列表 xdata 和 ydata
    xdata, ydata = list(zip(*data))
    
    # 计算 k 的值，k 为 xdata 列表长度减去 degnum 和 1的差值
    k = len(xdata) - degnum - 1
    
    # 如果 k 小于 0，则抛出 OptionError 异常，表示数值不足以支持所需的阶数
    if k < 0:
        raise OptionError("Too few values for the required degree.")
    
    # 创建一个 degnum + k + 1 行，degnum + k + 2 列的矩阵 c，矩阵元素初始化为 1
    c = ones(degnum + k + 1, degnum + k + 2)
    
    # 循环遍历 max(degnum, k) 次，对矩阵 c 进行填充
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j] * xdata[i]
    
    # 再次循环遍历 k + 1 次，继续填充矩阵 c
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j] * ydata[i]
    
    # 求解矩阵 c 的零空间，取第一个解 r
    r = c.nullspace()[0]
    
    # 返回多项式的表达式，其中分子是多项式的系数乘以 X 的幂次和，分母是多项式的系数乘以 X 的幂次和
    return (sum(r[i] * X**i for i in range(degnum + 1))
            / sum(r[i + degnum + 1] * X**i for i in range(k + 1)))
# 定义一个公共函数 viete，用于生成给定多项式 f 的 Viète 公式
@public
def viete(f, roots=None, *gens, **args):
    """
    Generate Viete's formulas for ``f``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import viete
    >>> from sympy import symbols

    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')

    >>> viete(a*x**2 + b*x + c, [r1, r2], x)
    [(r1 + r2, -b/a), (r1*r2, c/a)]

    """
    # 检查参数中是否有未知的标志位，但这里没有指定任何允许的标志位
    allowed_flags(args, [])

    # 如果 roots 是 Basic 类型的对象，则将其作为生成器加入 gens 中，并将 roots 设为 None
    if isinstance(roots, Basic):
        gens, roots = (roots,) + gens, None

    try:
        # 将 f 转换为多项式对象，并获取转换选项
        f, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出 ComputationFailed 异常
        raise ComputationFailed('viete', 1, exc)

    # 如果 f 是多变量多项式，则抛出异常 MultivariatePolynomialError
    if f.is_multivariate:
        raise MultivariatePolynomialError(
            "multivariate polynomials are not allowed")

    # 获取多项式 f 的次数
    n = f.degree()

    # 如果多项式次数小于 1，则无法应用 Viète 公式，抛出 ValueError 异常
    if n < 1:
        raise ValueError(
            "Cannot derive Viete's formulas for a constant polynomial")

    # 如果 roots 为 None，则生成一个以 'r' 开头的符号生成器，并从中取出 n 个符号
    roots = take(roots, n)

    # 如果生成的 roots 数量与多项式的次数不符，则抛出 ValueError 异常
    if n != len(roots):
        raise ValueError("required %s roots, got %s" % (n, len(roots)))

    # 获取 f 的首项系数和所有系数
    lc, coeffs = f.LC(), f.all_coeffs()
    result, sign = [], -1

    # 遍历多项式的各项系数，计算对应的对称多项式，并生成 Viète 公式的一部分
    for i, coeff in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign*(coeff/lc)
        result.append((poly, coeff))
        sign = -sign

    # 返回生成的 Viète 公式的列表
    return result
```