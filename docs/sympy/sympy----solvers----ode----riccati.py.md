# `D:\src\scipysrc\sympy\sympy\solvers\ode\riccati.py`

```
r"""
This module contains :py:meth:`~sympy.solvers.ode.riccati.solve_riccati`,
a function which gives all rational particular solutions to first order
Riccati ODEs. A general first order Riccati ODE is given by -

.. math:: y' = b_0(x) + b_1(x)w + b_2(x)w^2

where `b_0, b_1` and `b_2` can be arbitrary rational functions of `x`
with `b_2 \ne 0`. When `b_2 = 0`, the equation is not a Riccati ODE
anymore and becomes a Linear ODE. Similarly, when `b_0 = 0`, the equation
is a Bernoulli ODE. The algorithm presented below can find rational
solution(s) to all ODEs with `b_2 \ne 0` that have a rational solution,
or prove that no rational solution exists for the equation.

Background
==========

A Riccati equation can be transformed to its normal form

.. math:: y' + y^2 = a(x)

using the transformation

.. math:: y = -b_2(x) - \frac{b'_2(x)}{2 b_2(x)} - \frac{b_1(x)}{2}

where `a(x)` is given by

.. math:: a(x) = \frac{1}{4}\left(\frac{b_2'}{b_2} + b_1\right)^2 - \frac{1}{2}\left(\frac{b_2'}{b_2} + b_1\right)' - b_0 b_2

Thus, we can develop an algorithm to solve for the Riccati equation
in its normal form, which would in turn give us the solution for
the original Riccati equation.

Algorithm
=========

The algorithm implemented here is presented in the Ph.D thesis
"Rational and Algebraic Solutions of First-Order Algebraic ODEs"
by N. Thieu Vo. The entire thesis can be found here -
https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf

We have only implemented the Rational Riccati solver (Algorithm 11,
Pg 78-82 in Thesis). Before we proceed towards the implementation
of the algorithm, a few definitions to understand are -

1. Valuation of a Rational Function at `\infty`:
    The valuation of a rational function `p(x)` at `\infty` is equal
    to the difference between the degree of the denominator and the
    numerator of `p(x)`.

    NOTE: A general definition of valuation of a rational function
    at any value of `x` can be found in Pg 63 of the thesis, but
    is not of any interest for this algorithm.

2. Zeros and Poles of a Rational Function:
    Let `a(x) = \frac{S(x)}{T(x)}, T \ne 0` be a rational function
    of `x`. Then -

    a. The Zeros of `a(x)` are the roots of `S(x)`.
    b. The Poles of `a(x)` are the roots of `T(x)`. However, `\infty`
    can also be a pole of a(x). We say that `a(x)` has a pole at
    `\infty` if `a(\frac{1}{x})` has a pole at 0.

Every pole is associated with an order that is equal to the multiplicity
of its appearance as a root of `T(x)`. A pole is called a simple pole if
it has an order 1. Similarly, a pole is called a multiple pole if it has
an order `\ge` 2.

Necessary Conditions
====================

For a Riccati equation in its normal form,

.. math:: y' + y^2 = a(x)

we can define

a. A pole is called a movable pole if it is a pole of `y(x)` and is not
a pole of `a(x)`.
b. Similarly, a pole is called a non-movable pole if it is a pole of both
`y(x)` and `a(x)`.


"""
# 算法要求有理解存在的条件如下：

# a. `a(x)`的每个极点必须是简单极点或偶数阶的多重极点。
# b. `a(x)`在`\infty`处的估值必须是偶数或`\geq 2`。

# 该算法用于寻找 Riccati 微分方程的所有可能有理解。如果找不到有理解，则表示不存在有理解。

# 该算法适用于 Riccati 微分方程，其中系数是独立变量`x`的有理函数，系数为有理数，即在`Q(x)`中。这些有理函数的系数不能是浮点数、无理数、符号或其他类型的表达式。原因如下：

# 1. 使用符号时，不同符号可能取相同的值，这会影响极点的重数，如果这里存在符号的话。
# 2. 需要一个整数次数的界限来计算辅助微分方程的多项式解，从而为原始微分方程提供特解。如果存在符号/浮点数/无理数，则无法确定界限表达式是否为整数。

# 解法
# ======

# 根据这些定义，我们可以陈述方程的解的一般形式。`y(x)`必须具有以下形式 -

# .. math:: y(x) = \sum_{i=1}^{n} \sum_{j=1}^{r_i} \frac{c_{ij}}{(x - x_i)^j} + \sum_{i=1}^{m} \frac{1}{x - \chi_i} + \sum_{i=0}^{N} d_i x^i

# 其中`x_1, x_2, \dots, x_n`是`a(x)`的不动极点，
# `\chi_1, \chi_2, \dots, \chi_m`是`a(x)`的可移动极点，
# 而`N, n, r_1, r_2, \dots, r_n`的值可以从`a(x)`中确定。系数向量`(d_0, d_1, \dots, d_N)`和`(c_{i1}, c_{i2}, \dots, c_{i r_i})`可以从`a(x)`中确定。我们对这些向量中的每一个有两种选择，部分步骤是确定应该使用哪个向量以正确获取解。

# 实现
# ==============

# 在此实现中，我们使用``Poly``来表示有理函数，而不是使用``Expr``，因为``Poly``速度更快。由于不能直接使用``Poly``表示有理函数，我们将有理函数表示为两个``Poly``对象 - 一个用于分子，另一个用于分母。

# 代码编写遵循论文（第82页）中给出的步骤。

# 第0步：匹配方程 -
# 找到`b_0, b_1`和`b_2`。如果`b_2 = 0`或不存在这样的函数，则引发错误。

# 第1步：按理论部分解释的方式将方程转换为其正常形式。

# 第2步：初始化一个空的解集合``sol``。

# 第3步：如果`a(x) = 0`，则将`\frac{1}/{(x - C1)}`追加到``sol``中。

# 第4步：如果`a(x)`是一个非零有理数，将`\pm \sqrt{a}`追加到``sol``中。

# 第5步：找到`a(x)`的极点及其重数。设极点数为`n`。同时使用``val_at_inf``找到`a(x)`在`\infty`处的估值。

# 注意：尽管算法将`\infty`视为一个极点，但它是
# 给定一个解 `w(x)`，转换成其对应的正常形式 Riccati ODE 的解 `y(x)`
# `w(x)` 满足如下方程：
# w'(x) = b_0(x) + b_1(x)*w(x) + b_2(x)*w(x)^2
# `b1` 和 `b2` 是有理函数系数

def riccati_normal(w, x, b1, b2):
    """
    给定一个解 `w(x)`，满足以下方程：

    .. math:: w'(x) = b_0(x) + b_1(x)*w(x) + b_2(x)*w(x)^2

    和有理函数系数 `b_1(x)` 和 `b_2(x)`，这个函数将 `w(x)` 转换为
    其对应的正常 Riccati ODE 解 `y(x)`：

    .. math:: y'(x) + y(x)^2 = a(x)

    使用如下变换：

    .. math:: y(x) = -b_2(x)*w(x) - b_2'(x)/(2*b_2(x)) - b_1(x)/2
    """
    return -b2*w - b2.diff(x)/(2*b2) - b1/2


def riccati_inverse_normal(y, x, b1, b2, bp=None):
    """
    将正常 Riccati ODE 的解逆转换为 Riccati ODE 的解。
    """
    # 如果变量 bp 尚未被计算过，则进行计算
    if bp is None:
        # 计算 bp，其表达式为 -b2.diff(x)/(2*b2**2) - b1/(2*b2)
        bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)
    # 返回表达式 -y/b2 + bp，该表达式代表 w(x) 的计算结果
    return -y/b2 + bp
def riccati_reduced(eq, f, x):
    """
    Convert a Riccati ODE into its corresponding
    normal Riccati ODE.
    """
    # 调用 match_riccati 函数匹配 Riccati ODE 的模式并获取函数 funcs
    match, funcs = match_riccati(eq, f, x)
    # 如果方程不是 Riccati ODE，则返回 False
    if not match:
        return False
    # 使用有理函数计算并返回 a(x) 的表达式
    b0, b1, b2 = funcs
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    # 返回正常形式的 Riccati ODE：f'(x) + f(x)^2 = a(x)
    return f(x).diff(x) + f(x)**2 - a

def linsolve_dict(eq, syms):
    """
    Get the output of linsolve as a dict
    """
    # 将 linsolve 的元组类型返回值转换为字典以便于使用
    sol = linsolve(eq, syms)
    if not sol:
        return {}
    return dict(zip(syms, list(sol)[0]))


def match_riccati(eq, f, x):
    """
    A function that matches and returns the coefficients
    if an equation is a Riccati ODE

    Parameters
    ==========

    eq: Equation to be matched
    f: Dependent variable
    x: Independent variable

    Returns
    =======

    match: True if equation is a Riccati ODE, False otherwise
    funcs: [b0, b1, b2] if match is True, [] otherwise. Here,
    b0, b1 and b2 are rational functions which match the equation.
    """
    # 根据 f(x) 对方程进行分组
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    eq = eq.expand().collect(f(x))
    cf = eq.coeff(f(x).diff(x))

    # 必须存在 f(x).diff(x) 项
    # eq 必须是一个 Add 对象，因为我们使用了扩展的方程，并且它必须至少有两项 (b2 != 0)
    if cf != 0 and isinstance(eq, Add):

        # 将所有系数除以 f(x).diff(x) 的系数，并重新添加项以获得相同的方程
        eq = Add(*((x/cf).cancel() for x in eq.args)).collect(f(x))

        # 与模式匹配方程
        b1 = -eq.coeff(f(x))
        b2 = -eq.coeff(f(x)**2)
        b0 = (f(x).diff(x) - b1*f(x) - b2*f(x)**2 - eq).expand()
        funcs = [b0, b1, b2]

        # 检查系数不是符号和浮点数
        if any(len(x.atoms(Symbol)) > 1 or len(x.atoms(Float)) for x in funcs):
            return False, []

        # 如果 b_0(x) 包含 f(x)，则它不是 Riccati ODE
        if len(b0.atoms(f)) or not all((b2 != 0, b0.is_rational_function(x),
            b1.is_rational_function(x), b2.is_rational_function(x))):
            return False, []
        return True, funcs
    return False, []


def val_at_inf(num, den, x):
    # 计算有理函数在无穷远处的值
    return den.degree(x) - num.degree(x)


def check_necessary_conds(val_inf, muls):
    """
    The necessary conditions for a rational solution
    to exist are as follows -

    i) Every pole of a(x) must be either a simple pole
    or a multiple pole of even order.

    ii) The valuation of a(x) at infinity must be even
    or be greater than or equal to 2.
    """
    # 检查是否所有的极点都是简单极点（其重数为1），或者是否存在至少一个重数大于1的多重极点
    return (val_inf >= 2 or (val_inf <= 0 and val_inf % 2 == 0)) and \
        # 检查每个极点的重数是否为1，或者重数是否大于等于2且为偶数
        all(mul == 1 or (mul % 2 == 0 and mul >= 2) for mul in muls)
# 定义一个函数，用于在有理函数中进行变换 x -> 1/x，其中分子和分母分别使用 Poly 对象表示
def inverse_transform_poly(num, den, x):
    """
    A function to make the substitution
    x -> 1/x in a rational function that
    is represented using Poly objects for
    numerator and denominator.
    """
    # 声明一个 Poly 对象表示常数 1
    one = Poly(1, x)
    # 声明一个 Poly 对象表示变量 x
    xpoly = Poly(x, x)

    # 检查分子的次数是否与分母相同
    pwr = val_at_inf(num, den, x)
    if pwr >= 0:
        # 如果分母次数大于等于分子，将 x 替换为 1/x 会使得多余的次数转移到分子中
        if num.expr != 0:
            num = num.transform(one, xpoly) * x**pwr
            den = den.transform(one, xpoly)
    else:
        # 如果分子次数大于分母，将 x 替换为 1/x 会使得多余的次数转移到分母中
        num = num.transform(one, xpoly)
        den = den.transform(one, xpoly) * x**(-pwr)
    # 返回经过化简后的有理函数结果
    return num.cancel(den, include=True)


# 定义一个函数，用于计算有理函数在无穷远处的极限
def limit_at_inf(num, den, x):
    """
    Find the limit of a rational function
    at oo
    """
    # 计算分子和分母的次数差
    pwr = -val_at_inf(num, den, x)
    # 如果分子次数大于分母，极限将依赖于分子和分母的首项系数的符号
    if pwr > 0:
        return oo * sign(num.LC() / den.LC())
    # 如果分子和分母次数相同，极限将是分子和分母的首项系数的比值
    elif pwr == 0:
        return num.LC() / den.LC()
    # 如果分子次数小于分母，极限将是 0
    else:
        return 0


# 定义一个函数，用于构造特定情况下的系数向量 c
def construct_c_case_1(num, den, x, pole):
    # 在关于极点 pole 的 Laurent 级数展开中，找到 1/(x - pole)**2 的系数
    num1, den1 = (num * Poly((x - pole)**2, x, extension=True)).cancel(den, include=True)
    r = (num1.subs(x, pole)) / (den1.subs(x, pole))

    # 如果极点的重数是 2，系数向量 c 将包含两个值，分别是 (1 +- sqrt(1 + 4*r))/2
    if r != -S(1)/4:
        return [[(1 + sqrt(1 + 4*r))/2], [(1 - sqrt(1 + 4*r))/2]]
    # 如果不是，只返回一个值为 1/2 的向量
    return [[S.Half]]


# 定义一个函数，用于构造特定情况下的系数向量 c
def construct_c_case_2(num, den, x, pole, mul):
    # 使用论文中第 5.14 节中提到的递推关系生成系数向量 c

    # r_i = mul/2
    ri = mul // 2

    # 在极点 pole 的 Laurent 级数展开中，找到对应的系数
    ser = rational_laurent_series(num, den, x, pole, mul, 6)

    # 初始化一个空的存储系数的 memo，这是为了加号情况
    cplus = [0 for i in range(ri)]

    # 基础情况
    cplus[ri-1] = sqrt(ser[2*ri])

    # 反向迭代以找到所有系数
    s = ri - 1
    sm = 0
    for s in range(ri-1, 0, -1):
        sm = 0
        for j in range(s+1, ri):
            sm += cplus[j-1] * cplus[ri+s-j-1]
        if s != 1:
            cplus[s-1] = (ser[ri+s] - sm) / (2 * cplus[ri-1])

    # 为减号情况存储 memo
    cminus = [-x for x in cplus]

    # 找到递推中的第零个系数
    # 计算 cplus[0] 的系数，使用当前索引 ri 和先前计算的 sm、cplus[ri-1] 进行计算
    cplus[0] = (ser[ri+s] - sm - ri*cplus[ri-1])/(2*cplus[ri-1])
    
    # 计算 cminus[0] 的系数，使用当前索引 ri 和先前计算的 sm、cminus[ri-1] 进行计算
    cminus[0] = (ser[ri+s] - sm  - ri*cminus[ri-1])/(2*cminus[ri-1])

    # 如果 cplus 和 cminus 不相等，则返回一个包含 cplus 和 cminus 的列表
    if cplus != cminus:
        return [cplus, cminus]
    
    # 如果 cplus 等于 cminus，则只返回 cplus
    return cplus
# 如果多重性为1，添加到c向量中的系数为1（没有选择余地）
def construct_c_case_3():
    # 返回一个包含[[1]]的列表，表示系数为1的情况
    return [[1]]


# 辅助函数：计算每个极点在c向量中的系数
def construct_c(num, den, x, poles, muls):
    # 初始化一个空列表来存储结果
    c = []
    
    # 遍历极点和它们的多重性
    for pole, mul in zip(poles, muls):
        # 对于每个极点创建一个空列表
        c.append([])

        # Case 3
        if mul == 1:
            # 将Case 3的系数添加到当前极点的系数列表中
            c[-1].extend(construct_c_case_3())

        # Case 1
        elif mul == 2:
            # 将Case 1的系数添加到当前极点的系数列表中
            c[-1].extend(construct_c_case_1(num, den, x, pole))

        # Case 2
        else:
            # 将Case 2的系数添加到当前极点的系数列表中
            c[-1].extend(construct_c_case_2(num, den, x, pole, mul))

    # 返回计算得到的c向量
    return c


# Case 4的辅助函数：计算d向量的系数
def construct_d_case_4(ser, N):
    # 初始化一个长度为N+2的零向量
    dplus = [0 for i in range(N+2)]
    # d_N = sqrt(a_{2*N})
    dplus[N] = sqrt(ser[2*N])

    # 使用递推关系计算d_s的值
    for s in range(N-1, -2, -1):
        sm = 0
        for j in range(s+1, N):
            sm += dplus[j]*dplus[N+s-j]
        if s != -1:
            dplus[s] = (ser[N+s] - sm)/(2*dplus[N])

    # 对于d_N = -sqrt(a_{2*N})的情况，计算相应的系数
    dminus = [-x for x in dplus]

    # 在论文5.15节中，第三个方程是错误的！
    # 在该方程中，d_N必须替换为N*d_N。
    dplus[-1] = (ser[N+s] - N*dplus[N] - sm)/(2*dplus[N])
    dminus[-1] = (ser[N+s] - N*dminus[N] - sm)/(2*dminus[N])

    # 如果dplus不等于dminus，返回两者；否则返回dplus
    if dplus != dminus:
        return [dplus, dminus]
    return dplus


# Case 5的辅助函数：计算d向量的系数
def construct_d_case_5(ser):
    # 初始化一个长度为2的零向量
    dplus = [0, 0]

    # d_0 = sqrt(a_0)
    dplus[0] = sqrt(ser[0])

    # d_(-1) = a_(-1)/(2*d_0)
    dplus[-1] = ser[-1]/(2*dplus[0])

    # 对于负案例，系数与正案例的相反数相同。
    dminus = [-x for x in dplus]

    # 如果dplus不等于dminus，返回两者；否则返回dplus
    if dplus != dminus:
        return [dplus, dminus]
    return dplus


# Case 6的辅助函数：计算d向量的系数
def construct_d_case_6(num, den, x):
    # 计算s_oo，即x->0时的极限，相当于x->oo时的极限
    s_inf = limit_at_inf(Poly(x**2, x)*num, den, x)

    # 计算d_(-1) = (1 +- sqrt(1 + 4*s_oo))/2
    if s_inf != -S(1)/4:
        return [[(1 + sqrt(1 + 4*s_inf))/2], [(1 - sqrt(1 + 4*s_inf))/2]]
    return [[S.Half]]


# 辅助函数：计算d向量的系数，基于在无穷远处的函数的评估
def construct_d(num, den, x, val_inf):
    # 计算极点的多重性N
    N = -val_inf//2
    # 多重性作为一个极点的值
    mul = -val_inf if val_inf < 0 else 0
    # 计算有理Laurent级数的系列
    ser = rational_laurent_series(num, den, x, oo, mul, 1)

    # Case 4
    if val_inf < 0:
        d = construct_d_case_4(ser, N)

    # Case 5
    elif val_inf == 0:
        d = construct_d_case_5(ser)

    # Case 6
    else:
        d = construct_d_case_6(num, den, x)

    # 返回计算得到的d向量
    return d
# 定义一个函数，计算有理函数的 Laurent 级数展开系数
def rational_laurent_series(num, den, x, r, m, n):
    r"""
    The function computes the Laurent series coefficients
    of a rational function.

    Parameters
    ==========

    num: A Poly object that is the numerator of `f(x)`.
    den: A Poly object that is the denominator of `f(x)`.
    x: The variable of expansion of the series.
    r: The point of expansion of the series.
    m: Multiplicity of r if r is a pole of `f(x)`. Should
    be zero otherwise.
    n: Order of the term upto which the series is expanded.

    Returns
    =======

    series: A dictionary that has power of the term as key
    and coefficient of that term as value.

    Below is a basic outline of how the Laurent series of a
    rational function `f(x)` about `x_0` is being calculated -

    1. Substitute `x + x_0` in place of `x`. If `x_0`
    is a pole of `f(x)`, multiply the expression by `x^m`
    where `m` is the multiplicity of `x_0`. Denote the
    the resulting expression as g(x). We do this substitution
    so that we can now find the Laurent series of g(x) about
    `x = 0`.

    2. We can then assume that the Laurent series of `g(x)`
    takes the following form -

    .. math:: g(x) = \frac{num(x)}{den(x)} = \sum_{m = 0}^{\infty} a_m x^m

    where `a_m` denotes the Laurent series coefficients.

    3. Multiply the denominator to the RHS of the equation
    and form a recurrence relation for the coefficients `a_m`.
    """
    # 创建一个多项式对象，表示常数 1
    one = Poly(1, x, extension=True)

    if r == oo:
        # 如果展开点为无穷远，则先对函数进行变换：x -> 1/x，然后再在 x = 0 处找到级数
        num, den = inverse_transform_poly(num, den, x)
        r = S(0)

    if r:
        # 对于非零展开点，需要进行变换：x -> x + r
        num = num.transform(Poly(x + r, x, extension=True), one)
        den = den.transform(Poly(x + r, x, extension=True), one)

    # 如果展开点是一个极点，则从分母中去除 x^m
    num, den = (num*x**m).cancel(den, include=True)

    # 计算第一个项的系数差异（基本情况）
    maxdegree = 1 + max(num.degree(), den.degree())
    # 创建一组虚拟符号以解决线性方程组
    syms = symbols(f'a:{maxdegree}', cls=Dummy)
    # 计算分子与分母乘积的差异，获取系数
    diff = num - den * Poly(syms[::-1], x)
    coeff_diffs = diff.all_coeffs()[::-1][:maxdegree]
    # 解线性方程组以获取系数
    (coeffs, ) = linsolve(coeff_diffs, syms)

    # 使用递推关系计算其余项
    recursion = den.all_coeffs()[::-1]
    div, rec_rhs = recursion[0], recursion[1:]
    series = list(coeffs)
    while len(series) < n:
        # 计算下一个系数
        next_coeff = Add(*(c*series[-1-n] for n, c in enumerate(rec_rhs))) / div
        series.append(-next_coeff)
    # 将结果转换为字典形式，以幂次为键，系数为值
    series = {m - i: val for i, val in enumerate(series)}
    return series

def compute_m_ybar(x, poles, choice, N):
    """
    Helper function to calculate -

    1. m - The degree bound for the polynomial
    solution that must be found for the auxiliary
    differential equation.
    """
    2. ybar - 解决方案的一部分，可以使用极点、c 和 d 向量计算。
    """
    ybar = 0  # 初始化 ybar 变量为 0
    m = Poly(choice[-1][-1], x, extension=True)  # 使用选择列表中最后一个元素创建多项式 m

    # 计算 ybar 的第一个（嵌套的）求和，根据论文第 9 步骤 (第 82 页) 给出
    dybar = []
    for i, polei in enumerate(poles):  # 遍历极点列表
        for j, cij in enumerate(choice[i]):  # 遍历选择列表中的子列表
            dybar.append(cij/(x - polei)**(j + 1))  # 计算每个项并添加到 dybar 列表中
        m -= Poly(choice[i][0], x, extension=True)  # 从多项式 m 中减去选择列表中每个子列表的第一个元素
    ybar += Add(*dybar)  # 将 dybar 中的所有项相加，并加到 ybar 上

    # 计算 ybar 的第二个求和
    for i in range(N+1):  # 循环迭代 N+1 次
        ybar += choice[-1][i] * x**i  # 将选择列表中最后一个子列表的每个元素与 x 的幂相乘，并加到 ybar 上
    return (m.expr, ybar)  # 返回 m 的表达式和计算得到的 ybar
# Helper function to find a polynomial solution
# of degree m for the auxiliary differential
# equation.
def solve_aux_eq(numa, dena, numy, deny, x, m):
    # Assume that the solution is of the type
    # p(x) = C_0 + C_1*x + ... + C_{m-1}*x**(m-1) + x**m
    psyms = symbols(f'C0:{m}', cls=Dummy)
    K = ZZ[psyms]
    psol = Poly(K.gens, x, domain=K) + Poly(x**m, x, domain=K)

    # Eq (5.16) in Thesis - Pg 81
    auxeq = (dena*(numy.diff(x)*deny - numy*deny.diff(x) + numy**2) - numa*deny**2)*psol
    if m >= 1:
        px = psol.diff(x)
        auxeq += px*(2*numy*deny*dena)
    if m >= 2:
        auxeq += px.diff(x)*(deny**2*dena)
    if m != 0:
        # m is a non-zero integer. Find the constant terms using undetermined coefficients
        return psol, linsolve_dict(auxeq.all_coeffs(), psyms), True
    else:
        # m == 0 . Check if 1 (x**0) is a solution to the auxiliary equation
        return S.One, auxeq, auxeq == 0


# Helper function to remove redundant
# solutions to the differential equation.
def remove_redundant_sols(sol1, sol2, x):
    # If y1 and y2 are redundant solutions, there is
    # some value of the arbitrary constant for which
    # they will be equal

    syms1 = sol1.atoms(Symbol, Dummy)
    syms2 = sol2.atoms(Symbol, Dummy)
    num1, den1 = [Poly(e, x, extension=True) for e in sol1.together().as_numer_denom()]
    num2, den2 = [Poly(e, x, extension=True) for e in sol2.together().as_numer_denom()]
    # Cross multiply
    e = num1*den2 - den1*num2
    # Check if there are any constants
    syms = list(e.atoms(Symbol, Dummy))
    if len(syms):
        # Find values of constants for which solutions are equal
        redn = linsolve(e.all_coeffs(), syms)
        if len(redn):
            # Return the general solution over a particular solution
            if len(syms1) > len(syms2):
                return sol2
            # If both have constants, return the lesser complex solution
            elif len(syms1) == len(syms2):
                return sol1 if count_ops(syms1) >= count_ops(syms2) else sol2
            else:
                return sol1


# Helper function which computes the general
# solution for a Riccati ODE from its particular
# solutions.
#
# There are 3 cases to find the general solution
# from the particular solutions for a Riccati ODE
# depending on the number of particular solution(s)
# we have - 1, 2 or 3.
#
# For more information, see Section 6 of
# "Methods of Solution of the Riccati Differential Equation"
# by D. R. Haaheim and F. M. Stein
def get_gen_sol_from_part_sol(part_sols, a, x):
    # If no particular solutions are found, a general
    # solution cannot be found
    if len(part_sols) == 0:
        return []

    # In case of a single particular solution, the general
    # solution can be found by using the substitution
    # y = y1 + 1/z and solving a Bernoulli ODE to find z.
    # 如果特解列表的长度为1，表示只有一个特解
    elif len(part_sols) == 1:
        # 取出唯一的特解
        y1 = part_sols[0]
        # 构造被积函数并求积分
        i = exp(Integral(2*y1, x))
        # 计算第一个通解
        z = i * Integral(a/i, x)
        # 求解积分
        z = z.doit()
        # 如果 a 为零或者 z 为零，则返回特解 y1
        if a == 0 or z == 0:
            return y1
        # 否则返回通解 y1 + 1/z
        return y1 + 1/z

    # 如果特解列表的长度为2，表示有两个特解
    # 大多数 Riccati 方程都有两个有理数特解，因此这是最常见的情况
    elif len(part_sols) == 2:
        # 分别取出两个特解
        y1, y2 = part_sols
        # 判断是否已经有一个带有常数
        if len(y1.atoms(Dummy)) + len(y2.atoms(Dummy)) > 0:
            # 若有一个带有常数，则计算指数部分并求积分
            u = exp(Integral(y2 - y1, x)).doit()
        else:
            # 否则引入一个新的常数 C1
            C1 = Dummy('C1')
            # 计算指数部分并引入常数 C1
            u = C1*exp(Integral(y2 - y1, x)).doit()
        # 如果 u 等于 1，则返回特解 y2
        if u == 1:
            return y2
        # 否则返回通解 (y2*u - y1)/(u - 1)
        return (y2*u - y1)/(u - 1)

    # 如果特解列表的长度为3，表示有三个特解
    # 可以直接得到通解的封闭形式
    else:
        # 取出前三个特解和引入一个常数 C1
        y1, y2, y3 = part_sols[:3]
        C1 = Dummy('C1')
        # 返回通解 (C1 + 1)*y2*(y1 - y3)/(C1*y1 + y2 - (C1 + 1)*y3)
        return (C1 + 1)*y2*(y1 - y3)/(C1*y1 + y2 - (C1 + 1)*y3)
# 定义解决 Riccati 方程的主函数，可以给出具体或通用解
def solve_riccati(fx, x, b0, b1, b2, gensol=False):
    """
    The main function that gives particular/general
    solutions to Riccati ODEs that have atleast 1
    rational particular solution.
    """

    # Step 1 : Convert to Normal Form
    # 计算并转换为正常形式
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    # 将表达式 a 合并分子分母
    a_t = a.together()
    # 将合并后的表达式 a_t 分别转化为多项式的形式，进行约分
    num, den = [Poly(e, x, extension=True) for e in a_t.as_numer_denom()]
    num, den = num.cancel(den, include=True)

    # Step 2
    presol = []

    # Step 3 : a(x) is 0
    # 如果分子为零，添加一个特定解
    if num == 0:
        presol.append(1/(x + Dummy('C1')))

    # Step 4 : a(x) is a non-zero constant
    # 如果分子不含有 x 的符号，添加两个解：平方根和负平方根
    elif x not in num.free_symbols.union(den.free_symbols):
        presol.extend([sqrt(a), -sqrt(a)])

    # Step 5 : Find poles and valuation at infinity
    # 计算分母的根（极点）和在无穷远处的评估
    poles = roots(den, x)
    poles, muls = list(poles.keys()), list(poles.values())
    val_inf = val_at_inf(num, den, x)
    if len(poles):
        # 如果有奇点存在，则执行以下操作

        # 检查必要条件（在模块文档字符串中有说明）
        if not check_necessary_conds(val_inf, muls):
            raise ValueError("Rational Solution doesn't exist")
            # 如果不满足必要条件，则抛出值错误异常

        # Step 6
        # 为每个奇点构造 c 向量
        c = construct_c(num, den, x, poles, muls)

        # 为每个奇点构造 d 向量
        d = construct_d(num, den, x, val_inf)

        # Step 7 : Iterate over all possible combinations and return solutions
        # 对所有可能的组合进行迭代并返回解决方案
        # 对于每个可能的组合，生成一个由 0 和 1 组成的数组，
        # 其中 0 表示选择第一个选项，1 表示选择第二个选项。

        # 注意：如果我们找到了3个特定解，可以退出循环，
        # 但这里没有实现 -
        #   a. 找到3个特定解非常罕见。大多数情况下，只找到2个特定解。
        #   b. 如果我们在找到3个特定解后退出，可能会发生1或2个冗余解的情况。
        #      因此，与其在计算特定解上花费更多时间，
        #      我们将从单个特定解计算一般解，这通常比从2或3个特定解计算一般解更慢。
        c.append(d)
        choices = product(*c)
        for choice in choices:
            m, ybar = compute_m_ybar(x, poles, choice, -val_inf//2)
            numy, deny = [Poly(e, x, extension=True) for e in ybar.together().as_numer_denom()]

            # Step 10 : Check if a valid solution exists. If yes, also check
            # if m is a non-negative integer
            # 步骤10：检查是否存在有效解。如果是，则检查 m 是否为非负整数
            if m.is_nonnegative == True and m.is_integer == True:

                # Step 11 : Find polynomial solutions of degree m for the auxiliary equation
                # 步骤11：为辅助方程找到 m 阶多项式解
                psol, coeffs, exists = solve_aux_eq(num, den, numy, deny, x, m)

                # Step 12 : If valid polynomial solution exists, append solution.
                # 步骤12：如果存在有效的多项式解，则追加解决方案
                if exists:
                    # m == 0 case
                    # m == 0 的情况
                    if psol == 1 and coeffs == 0:
                        # p(x) = 1，因此不需要添加 p'(x)/p(x) 项
                        presol.append(ybar)
                    # m 是正整数且存在有效系数
                    elif len(coeffs):
                        # 使用有效系数替换以获取 p(x)
                        psol = psol.xreplace(coeffs)
                        # y(x) = ybar(x) + p'(x)/p(x)
                        presol.append(ybar + psol.diff(x)/psol)

    # 从现有解列表中删除冗余解
    remove = set()
    for i in range(len(presol)):
        for j in range(i+1, len(presol)):
            rem = remove_redundant_sols(presol[i], presol[j], x)
            if rem is not None:
                remove.add(rem)
    sols = [x for x in presol if x not in remove]
    # Step 15 : Inverse transform the solutions of the equation in normal form
    # 计算在正常形式下方程的解的反变换
    bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)
    
    # If general solution is required, compute it from the particular solutions
    # 如果需要一般解，从特解计算它
    if gensol:
        sols = [get_gen_sol_from_part_sol(sols, a, x)]
    
    # Inverse transform the particular solutions
    # 对特定解进行反变换
    presol = [Eq(fx, riccati_inverse_normal(y, x, b1, b2, bp).cancel(extension=True)) for y in sols]
    # 返回处理后的解方程组
    return presol
```