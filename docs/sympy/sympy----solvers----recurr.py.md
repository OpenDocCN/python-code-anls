# `D:\src\scipysrc\sympy\sympy\solvers\recurr.py`

```
r"""
This module is intended for solving recurrences or, in other words,
difference equations. Currently supported are linear, inhomogeneous
equations with polynomial or rational coefficients.

The solutions are obtained among polynomials, rational functions,
hypergeometric terms, or combinations of hypergeometric term which
are pairwise dissimilar.

``rsolve_X`` functions were meant as a low level interface
for ``rsolve`` which would use Mathematica's syntax.

Given a recurrence relation:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              ... + a_{0}(n) y(n) = f(n)

where `k > 0` and `a_{i}(n)` are polynomials in `n`. To use
``rsolve_X`` we need to put all coefficients in to a list ``L`` of
`k+1` elements the following way:

    ``L = [a_{0}(n), ..., a_{k-1}(n), a_{k}(n)]``

where ``L[i]``, for `i=0, \ldots, k`, maps to
`a_{i}(n) y(n+i)` (`y(n+i)` is implicit).

For example if we would like to compute `m`-th Bernoulli polynomial
up to a constant (example was taken from rsolve_poly docstring),
then we would use `b(n+1) - b(n) = m n^{m-1}` recurrence, which
has solution `b(n) = B_m + C`.

Then ``L = [-1, 1]`` and `f(n) = m n^(m-1)` and finally for `m=4`:

>>> from sympy import Symbol, bernoulli, rsolve_poly
>>> n = Symbol('n', integer=True)

>>> rsolve_poly([-1, 1], 4*n**3, n)
C0 + n**4 - 2*n**3 + n**2

>>> bernoulli(4, n)
n**4 - 2*n**3 + n**2 - 1/30

For the sake of completeness, `f(n)` can be:

    [1] a polynomial               -> rsolve_poly
    [2] a rational function        -> rsolve_ratio
    [3] a hypergeometric function  -> rsolve_hyper
"""
from collections import defaultdict  # 导入 defaultdict 类用于创建默认字典

from sympy.concrete import product  # 导入 product 函数用于计算连乘积
from sympy.core.singleton import S  # 导入 S 单例对象，表示数学中的单例
from sympy.core.numbers import Rational, I  # 导入 Rational 和 I 用于处理有理数和虚数
from sympy.core.symbol import Symbol, Wild, Dummy  # 导入符号、通配符和占位符类
from sympy.core.relational import Equality  # 导入 Equality 类用于表示数学中的等式
from sympy.core.add import Add  # 导入 Add 类用于处理数学中的加法表达式
from sympy.core.mul import Mul  # 导入 Mul 类用于处理数学中的乘法表达式
from sympy.core.sorting import default_sort_key  # 导入 default_sort_key 函数用于排序
from sympy.core.sympify import sympify  # 导入 sympify 函数用于将输入转换为 SymPy 对象

from sympy.simplify import simplify, hypersimp, hypersimilar  # 导入简化函数和超几何函数处理函数
from sympy.solvers import solve, solve_undetermined_coeffs  # 导入求解函数和处理未定系数的函数
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant  # 导入多项式操作相关函数
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial  # 导入数学函数
from sympy.matrices import Matrix, casoratian  # 导入矩阵和 Casoratian 行列式函数
from sympy.utilities.iterables import numbered_symbols  # 导入用于生成序号符号的函数


def rsolve_poly(coeffs, f, n, shift=0, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order
    `k` with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek for
    all polynomial solutions over field `K` of characteristic zero.

    The algorithm performs two basic steps:

        (1) Compute degree `N` of the general polynomial solution.
        (2) Find all polynomials of degree `N` or less
            of `\operatorname{L} y = f`.

    There are two methods for computing the polynomial solutions.
    """
    """
    使用 sympify 函数将输入的 f 转换为 sympy 的表达式对象。

    如果 f 不是关于 n 的多项式，则返回 None。

    检查 f 是否为零多项式，确定是否为齐次方程。

    设置变量 r 为 coeffs 列表的长度减去 1。

    将 coeffs 列表中的每个元素转换为 sympy 的多项式对象。

    初始化 polys 和 terms 列表，长度均为 r + 1，用于存储多项式和其最高项的系数和指数。

    遍历计算多项式 polys[i]，通过线性组合构造出多项式的每一项。

    计算每个 polys[i] 的最高项系数和指数，并存储在 terms[i] 中。

    初始化 d 和 b 分别为 terms[0] 的指数。

    遍历 terms 列表，更新 d 和 b 分别为所有 terms[i] 的最高指数和次高指数。

    将 d 和 b 转换为整数类型。

    创建符号变量 x。

    初始化 degree_poly 为零多项式，用于存储计算出的次数多项式。

    遍历 terms 列表，将所有 terms[i] 指数减去 i 等于 b 的项相加，构造次数多项式 degree_poly。

    计算 degree_poly 的根，并存储为非负整数的列表 nni_roots。

    根据计算出的根更新 N 的值，确保 N 包含所有相关的根和指数。

    如果是齐次方程，则根据 hints 中的 symbols 值返回对应的结果或零多项式。

    否则，根据多项式 f 的度和 b 的值更新 N。

    最终确定 N 的最大整数值。

    如果 N 小于 0，则根据齐次或非齐次方程返回相应的结果或 None。
    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    homogeneous = f.is_zero

    r = len(coeffs) - 1

    coeffs = [Poly(coeff, n) for coeff in coeffs]

    polys = [Poly(0, n)]*(r + 1)
    terms = [(S.Zero, S.NegativeInfinity)]*(r + 1)

    for i in range(r + 1):
        for j in range(i, r + 1):
            polys[i] += coeffs[j]*(binomial(j, i).as_poly(n))

        if not polys[i].is_zero:
            (exp,), coeff = polys[i].LT()
            terms[i] = (coeff, exp)

    d = b = terms[0][1]

    for i in range(1, r + 1):
        if terms[i][1] > d:
            d = terms[i][1]

        if terms[i][1] - i > b:
            b = terms[i][1] - i

    d, b = int(d), int(b)

    x = Dummy('x')

    degree_poly = S.Zero

    for i in range(r + 1):
        if terms[i][1] - i == b:
            degree_poly += terms[i][0]*FallingFactorial(x, i)

    nni_roots = list(roots(degree_poly, x, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if nni_roots:
        N = [max(nni_roots)]
    else:
        N = []

    if homogeneous:
        N += [-b - 1]
    else:
        N += [f.as_poly(n).degree() - b, -b - 1]

    N = int(max(N))

    if N < 0:
        if homogeneous:
            if hints.get('symbols', False):
                return (S.Zero, [])
            else:
                return S.Zero
        else:
            return None
    # 如果给定的 N 小于等于 r，则执行以下操作
    if N <= r:
        # 初始化空列表 C，以及变量 y 和 E 设为零
        C = []
        y = E = S.Zero

        # 循环生成 C 列表中的符号对象，并计算 y 的表达式
        for i in range(N + 1):
            C.append(Symbol('C' + str(i + shift)))
            y += C[i] * n**i

        # 计算表达式 E，通过求和 coeffs[i] * y(n+i) 的值
        for i in range(r + 1):
            E += coeffs[i].as_expr() * y.subs(n, n + i)

        # 解未定系数的方程 E - f = 0，并获取解
        solutions = solve_undetermined_coeffs(E - f, C, n)

        # 如果找到解，则更新 C 和计算结果 result
        if solutions is not None:
            _C = C
            C = [c for c in C if (c not in solutions)]
            result = y.subs(solutions)
        else:
            return None  # 如果找不到解则返回 None，表示待定

    # 如果 C 不等于 _C，执行以下操作
    if C != _C:
        # 重新编号 C 列表，使其连续
        result = result.xreplace(dict(zip(C, _C)))
        C = _C[:len(C)]

    # 如果 hints 中包含 'symbols' 键并且值为 True，则返回结果和 C 列表
    if hints.get('symbols', False):
        return (result, C)
    else:
        return result
# 定义函数 rsolve_ratio，用于解决具有多项式系数和不齐次方程的线性递推算子 L 的有理解
# 给定线性递推算子 L 的 k 阶多项式系数和不齐次方程 L y = f，其中 f 是一个多项式，本函数寻找在特征为零的域 K 上的所有有理解。

    r"""
    # 多项式 `v(n)` 被计算为任何有理解 `y(n) = u(n)/v(n)` 的通用分母。
    # 构造新的线性差分方程通过替换 `y(n) = u(n)/v(n)` 并解出 `u(n)`，找到其所有多项式解。如果没有找到解则返回 `None`。
    
    # 此处实现的算法是 Abramov 的经过修订的版本，最初于 1989 年开发。
    # 新方法更易于实现且整体效率更好。此方法可以轻松适应 q-差分方程的情况。

    # 除了单独找到有理解外，这个函数也是 Hyper 算法的重要部分，用于找到递推不齐次部分的特解。

    Examples
    ========

    # 从 sympy.abc 导入 x
    # 从 sympy.solvers.recurr 导入 rsolve_ratio
    # 使用 rsolve_ratio 解决如下示例中的线性递推方程

    References
    ==========

    .. [1] S. A. Abramov, Rational solutions of linear difference
           and q-difference equations with polynomial coefficients,
           in: T. Levelt, ed., Proc. ISSAC '95, ACM Press, New York,
           1995, 285-289

    See Also
    ========

    rsolve_hyper
    """

    # 将 f 转换为符号表达式
    f = sympify(f)

    # 如果 f 不是关于 n 的多项式，则返回 None
    if not f.is_polynomial(n):
        return None

    # 将 coeffs 列表中的每个元素转换为符号表达式
    coeffs = list(map(sympify, coeffs))

    # 计算线性递推系数的阶数 r
    r = len(coeffs) - 1

    # 提取出 A 和 B 的系数
    A, B = coeffs[r], coeffs[0]

    # 将 A 中的 n 替换为 n-r 并展开
    A = A.subs(n, n - r).expand()

    # 计算 A 和 B.subs(n, n + h) 的 resultan
    h = Dummy('h')
    res = resultant(A, B.subs(n, n + h), n)

    # 如果 res 不是关于 h 的多项式，则将其视为分子分母的商，并简化为多项式
    if not res.is_polynomial(h):
        p, q = res.as_numer_denom()
        res = quo(p, q, h)

    # 找到 res 在非负整数 h 上的根，并转换为列表 nni_roots
    nni_roots = list(roots(res, h, filter='Z',
        predicate=lambda r: r >= 0).keys())

    # 如果没有找到非负整数的根，则调用 rsolve_poly 函数处理 coeffs, f, n，**hints
    if not nni_roots:
        return rsolve_poly(coeffs, f, n, **hints)
    else:
        # 初始化 C 为 S.One，numers 列表长度为 r + 1，每个元素初始化为 S.Zero
        C, numers = S.One, [S.Zero]*(r + 1)

        # 从最大的 nni_roots 开始向 0 遍历
        for i in range(int(max(nni_roots)), -1, -1):
            # 计算 A 和 B.subs(n, n + i) 的最大公约数
            d = gcd(A, B.subs(n, n + i), n)

            # 更新 A 和 B
            A = quo(A, d, n)
            B = quo(B, d.subs(n, n - i), n)

            # 计算乘积 C *= d.subs(n, n - j) 对于 j 从 0 到 i 的所有值
            C *= Mul(*[d.subs(n, n - j) for j in range(i + 1)])

        # 计算 denoms 列表，包含 C.subs(n, n + i) 对于 i 从 0 到 r 的所有值
        denoms = [C.subs(n, n + i) for i in range(r + 1)]

        # 对于每个 i 从 0 到 r
        for i in range(r + 1):
            # 计算 coeffs[i] 和 denoms[i] 的最大公约数
            g = gcd(coeffs[i], denoms[i], n)

            # 更新 numers[i] 和 denoms[i]
            numers[i] = quo(coeffs[i], g, n)
            denoms[i] = quo(denoms[i], g, n)

        # 对于每个 i 从 0 到 r
        for i in range(r + 1):
            # 计算 numers[i] *= Mul(*(denoms[:i] + denoms[i + 1:])) 的结果
            numers[i] *= Mul(*(denoms[:i] + denoms[i + 1:]))

        # 调用 rsolve_poly 函数解决 numers, f * Mul(*denoms), n 的问题，使用 hints 作为额外参数
        result = rsolve_poly(numers, f * Mul(*denoms), n, **hints)

        # 如果 result 不为 None
        if result is not None:
            # 如果 hints 中包含 'symbols' 键并且其值为 True
            if hints.get('symbols', False):
                # 返回简化后的结果 (simplify(result[0] / C), result[1])
                return (simplify(result[0] / C), result[1])
            else:
                # 返回简化后的结果 simplify(result / C)
                return simplify(result / C)
        else:
            # 如果 result 是 None，则返回 None
            return None
    # 将系数列表中的每个元素转换为符号表达式
    coeffs = list(map(sympify, coeffs))

    # 将输入的不定项表达式转换为符号表达式
    f = sympify(f)

    # 计算线性递推算子的阶数，即系数列表长度减一
    r, kernel, symbols = len(coeffs) - 1, [], set()
    # 如果 f 不是零多项式
    if not f.is_zero:
        # 如果 f 是加法表达式
        if f.is_Add:
            # 初始化一个空字典 similar，用于存储类似的超几何函数
            similar = {}

            # 对 f 展开后的每个子表达式 g 进行处理
            for g in f.expand().args:
                # 如果 g 不是超几何函数则返回 None
                if not g.is_hypergeometric(n):
                    return None

                # 遍历 similar 字典中的键 h
                for h in similar.keys():
                    # 如果 g 和 h 是类似的超几何函数，则合并到 h 中
                    if hypersimilar(g, h, n):
                        similar[h] += g
                        break
                else:
                    # 如果 g 与所有的 h 都不类似，则将 g 加入 similar 字典中
                    similar[g] = S.Zero

            # 生成不齐次项列表，将 similar 中的每对 g 和 h 相加
            inhomogeneous = [g + h for g, h in similar.items()]
        # 如果 f 是超几何函数
        elif f.is_hypergeometric(n):
            # 将 f 放入不齐次项列表中
            inhomogeneous = [f]
        else:
            # 如果 f 不是加法表达式也不是超几何函数，则返回 None
            return None

        # 遍历不齐次项列表中的每个元素
        for i, g in enumerate(inhomogeneous):
            # 初始化系数、多项式和分母列表
            coeff, polys = S.One, coeffs[:]
            denoms = [S.One]*(r + 1)

            # 对 g 进行超几何简化，得到 s
            s = hypersimp(g, n)

            # 计算多项式的系数和分母
            for j in range(1, r + 1):
                coeff *= s.subs(n, n + j - 1)
                p, q = coeff.as_numer_denom()
                polys[j] *= p
                denoms[j] = q

            # 对每个多项式进行乘积操作
            for j in range(r + 1):
                polys[j] *= Mul(*(denoms[:j] + denoms[j + 1:]))

            # FIXME: 下面调用 rsolve_ratio 应该足够 (可以移除 rsolve_poly 的调用)，但必须先修复 XFAIL 测试 test_rsolve_ratio_missed。
            # 使用 rsolve_ratio 函数求解差分方程
            R = rsolve_ratio(polys, Mul(*denoms), n, symbols=True)
            if R is not None:
                R, syms = R
                # 如果存在符号变量，则将其替换为 0
                if syms:
                    R = R.subs(zip(syms, [0]*len(syms)))
            else:
                # 否则，使用 rsolve_poly 函数求解差分方程
                R = rsolve_poly(polys, Mul(*denoms), n)

            # 如果成功求解出 R，则更新不齐次项列表中的元素
            if R:
                inhomogeneous[i] *= R
            else:
                # 如果无法求解，则返回 None
                return None

            # 计算结果的和并简化
            result = Add(*inhomogeneous)
            result = simplify(result)
    else:
        # 如果 f 是零多项式，则结果为零
        result = S.Zero

    # 创建一个虚拟变量 Z
    Z = Dummy('Z')

    # 提取 coeffs 列表的首尾元素，并在其中的 r 位置用 n 替换
    p, q = coeffs[0], coeffs[r].subs(n, n - r + 1)

    # 对 p 和 q 求解根并获取其键值
    p_factors = list(roots(p, n).keys())
    q_factors = list(roots(q, n).keys())

    # 初始化因子列表，包含单位元组
    factors = [(S.One, S.One)]

    # 遍历 p_factors 和 q_factors 列表中的元素对
    for p in p_factors:
        for q in q_factors:
            # 如果 p 和 q 都是整数且 p <= q，则跳过
            if p.is_integer and q.is_integer and p <= q:
                continue
            else:
                # 否则将 (n - p, n - q) 添加到因子列表中
                factors += [(n - p, n - q)]

    # 分别创建 p 和 q 列表
    p = [(n - p, S.One) for p in p_factors]
    q = [(S.One, n - q) for q in q_factors]

    # 将 p、factors 和 q 合并为最终的因子列表
    factors = p + factors + q
    # 遍历 factors 列表中的每对 A, B
    for A, B in factors:
        # 初始化空列表 polys 和 degrees
        polys, degrees = [], []
        # 计算 D = A * B.subs(n, n + r - 1)
        D = A * B.subs(n, n + r - 1)

        # 对于 i 从 0 到 r 的每一个值
        for i in range(r + 1):
            # 计算 a = A.subs(n, n + j) 的乘积，其中 j 从 0 到 i-1
            a = Mul(*[A.subs(n, n + j) for j in range(i)])
            # 计算 b = B.subs(n, n + j) 的乘积，其中 j 从 i 到 r-1
            b = Mul(*[B.subs(n, n + j) for j in range(i, r)])

            # 计算 poly = quo(coeffs[i] * a * b, D, n)，并将其转换为多项式
            poly = quo(coeffs[i] * a * b, D, n)
            polys.append(poly.as_poly(n))

            # 如果 poly 不为零，则记录其次数到 degrees 中
            if not poly.is_zero:
                degrees.append(polys[i].degree())

        # 如果 degrees 列表非空
        if degrees:
            # 计算 degrees 中的最大值 d，同时初始化 poly 为 S.Zero
            d, poly = max(degrees), S.Zero
        else:
            # 若 degrees 列表为空，则返回 None
            return None

        # 对于 i 从 0 到 r 的每一个值
        for i in range(r + 1):
            # 获取 polys[i] 在次数 d 处的系数
            coeff = polys[i].nth(d)

            # 如果 coeff 不为 S.Zero
            if coeff is not S.Zero:
                # 计算 poly += coeff * Z**i
                poly += coeff * Z**i

        # 对 poly 求根，获取根的字典，并遍历其键值对
        for z in roots(poly, Z).keys():
            # 如果 z 是零，则跳过此次循环
            if z.is_zero:
                continue

            # 计算递推系数 recurr_coeffs = [polys[i].as_expr() * z**i for i in range(r + 1)]
            recurr_coeffs = [polys[i].as_expr() * z**i for i in range(r + 1)]

            # 如果 d == 0 并且 0 不等于 Add(*[recurr_coeffs[j] * j for j in range(1, r + 1)])
            if d == 0 and 0 != Add(*[recurr_coeffs[j] * j for j in range(1, r + 1)]):
                # 使用符号 "C" 后接 symbols 的长度构建解的列表 sol
                sol = [Symbol("C" + str(len(symbols)))]
            else:
                # 调用 rsolve_poly 函数求解递推关系，返回解 sol 和符号列表 syms
                sol, syms = rsolve_poly(recurr_coeffs, 0, n, len(symbols), symbols=True)
                # 将 sol 收集关于符号的系数
                sol = sol.collect(syms)
                # 将 sol 转换为列表，包含每个符号的系数
                sol = [sol.coeff(s) for s in syms]

            # 对于解 sol 中的每个 C
            for C in sol:
                # 计算 ratio = z * A * C.subs(n, n + 1) / B / C，并简化 ratio
                ratio = z * A * C.subs(n, n + 1) / B / C
                ratio = simplify(ratio)

                # 获取 ratio 的分母部分，并计算其所有非负根的最小整数根 n0
                n0 = 0
                for n_root in roots(ratio.as_numer_denom()[1], n).keys():
                    # 如果 n_root 含有虚数部分，则返回 None
                    if n_root.has(I):
                        return None
                    # 如果 n0 小于 n_root + 1，则更新 n0
                    elif (n0 < (n_root + 1)) == True:
                        n0 = n_root + 1

                # 计算 K = product(ratio, (n, n0, n - 1))
                K = product(ratio, (n, n0, n - 1))
                # 如果 K 包含 factorial、FallingFactorial 或 RisingFactorial 函数，则简化 K
                if K.has(factorial, FallingFactorial, RisingFactorial):
                    K = simplify(K)

                # 如果 casoratian(kernel + [K], n, zero=False) 不为零，则将 K 添加到 kernel 中
                if casoratian(kernel + [K], n, zero=False) != 0:
                    kernel.append(K)

    # 根据默认排序键对 kernel 列表进行排序
    kernel.sort(key=default_sort_key)
    # 使用 numbered_symbols 函数为 kernel 列表中的每个元素生成 C 符号，并将其与 kernel 组成列表 sk
    sk = list(zip(numbered_symbols('C'), kernel))

    # 对于 sk 列表中的每个 C, ker 组合
    for C, ker in sk:
        # 计算 result += C * ker
        result += C * ker

    # 如果 hints 字典中的 'symbols' 键值为 True
    if hints.get('symbols', False):
        # XXX: 返回 sk 中的符号列表，但不按确定性顺序排列
        symbols |= {s for s, k in sk}
        # 返回 result 和符号列表 symbols 的元组
        return (result, list(symbols))
    else:
        # 如果 hints 字典中的 'symbols' 键值为 False，则仅返回 result
        return result
    r"""
    Solve univariate recurrence with rational coefficients.

    Given `k`-th order linear recurrence `\operatorname{L} y = f`,
    or equivalently:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              \cdots + a_{0}(n) y(n) = f(n)

    where `a_{i}(n)`, for `i=0, \ldots, k`, are polynomials or rational
    functions in `n`, and `f` is a hypergeometric function or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms in
    `n`, finds all solutions or returns ``None``, if none were found.

    Initial conditions can be given as a dictionary in two forms:

        (1) ``{  n_0  : v_0,   n_1  : v_1, ...,   n_m  : v_m}``
        (2) ``{y(n_0) : v_0, y(n_1) : v_1, ..., y(n_m) : v_m}``

    or as a list ``L`` of values:

        ``L = [v_0, v_1, ..., v_m]``

    where ``L[i] = v_i``, for `i=0, \ldots, m`, maps to `y(n_i)`.

    Examples
    ========

    Lets consider the following recurrence:

    .. math:: (n - 1) y(n + 2) - (n^2 + 3 n - 2) y(n + 1) +
              2 n (n + 1) y(n) = 0

    >>> from sympy import Function, rsolve
    >>> from sympy.abc import n
    >>> y = Function('y')

    >>> f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)

    >>> rsolve(f, y(n))
    2**n*C0 + C1*factorial(n)

    >>> rsolve(f, y(n), {y(0):0, y(1):3})
    3*2**n - 3*factorial(n)

    See Also
    ========

    rsolve_poly, rsolve_ratio, rsolve_hyper

    """
    # 如果输入的 f 是等式，则转换成左右两侧差的形式
    if isinstance(f, Equality):
        f = f.lhs - f.rhs

    # 获取 y 函数的自变量，即 n
    n = y.args[0]
    # 排除 n 的 Wildcard，用于匹配不同的 n+k 表达式
    k = Wild('k', exclude=(n,))

    # 对 f 进行展开并按 y 函数进行收集处理，以支持如 y(n) + a*(y(n + 1) + y(n - 1))/2 的输入
    f = f.expand().collect(y.func(Wild('m', integer=True)))

    # 初始化部分
    h_part = defaultdict(list)  # 存储与不同 n+k 匹配的系数
    i_part = []  # 存储独立项的系数

    # 分析并处理每个加法项
    for g in Add.make_args(f):
        coeff, dep = g.as_coeff_mul(y.func)
        if not dep:
            i_part.append(coeff)  # 如果没有 y 函数的依赖，将其作为独立项处理
            continue
        for h in dep:
            if h.is_Function and h.func == y.func:
                result = h.args[0].match(n + k)
                if result is not None:
                    h_part[int(result[k])].append(coeff)  # 将符合 n+k 匹配的系数添加到对应的字典项中
                    continue
            raise ValueError(
                "'%s(%s + k)' expected, got '%s'" % (y.func, n, h))  # 抛出异常，如果匹配的形式不正确

    # 确保所有匹配的 k 部分都被处理
    for k in h_part:
        h_part[k] = Add(*h_part[k])
    h_part.default_factory = lambda: 0  # 默认值设为 0
    i_part = Add(*i_part)  # 将独立项的系数合并

    # 简化处理后的系数
    for k, coeff in h_part.items():
        h_part[k] = simplify(coeff)

    # 确保独立项符合要求，应为一组超几何函数的和
    common = S.One
    if not i_part.is_zero and not i_part.is_hypergeometric(n) and \
       not (i_part.is_Add and all((x.is_hypergeometric(n) for x in i_part.expand().args))):
        raise ValueError("The independent term should be a sum of hypergeometric functions, got '%s'" % i_part)
    # 对于 h_part 中的每个系数进行遍历
    for coeff in h_part.values():
        # 检查系数是否是关于 n 的有理函数
        if coeff.is_rational_function(n):
            # 如果是有理函数，检查是否为多项式
            if not coeff.is_polynomial(n):
                # 如果不是多项式，计算通分后的分母与当前最小公倍数的最小公倍数
                common = lcm(common, coeff.as_numer_denom()[1], n)
        else:
            # 如果不是有理函数，抛出异常
            raise ValueError(
                "Polynomial or rational function expected, got '%s'" % coeff)

    # 将 i_part 分离为分子和分母
    i_numer, i_denom = i_part.as_numer_denom()

    # 检查 i_denom 是否为关于 n 的多项式
    if i_denom.is_polynomial(n):
        # 如果是多项式，计算其与当前最小公倍数的最小公倍数
        common = lcm(common, i_denom, n)

    # 如果 common 不是单位元（S.One）
    if common is not S.One:
        # 对 h_part 中的每个系数进行处理，使其乘以 common 除以其分母
        for k, coeff in h_part.items():
            numer, denom = coeff.as_numer_denom()
            h_part[k] = numer * quo(common, denom, n)

        # 对 i_part 进行处理，使其乘以 common 除以其分母
        i_part = i_numer * quo(common, i_denom, n)

    # 计算 h_part 中键的最小值
    K_min = min(h_part.keys())

    # 如果 K_min 小于 0
    if K_min < 0:
        # 计算 K 值
        K = abs(K_min)

        # 创建一个默认值为零的 defaultdict H_part
        H_part = defaultdict(lambda: S.Zero)

        # 替换 i_part 和 common 中的 n 为 n + K，并展开结果
        i_part = i_part.subs(n, n + K).expand()
        common = common.subs(n, n + K).expand()

        # 对 h_part 中的每个系数进行处理，使其键加上 K，同时替换 n 为 n + K 并展开结果
        for k, coeff in h_part.items():
            H_part[k + K] = coeff.subs(n, n + K).expand()
    else:
        # 如果 K_min 不小于 0，则直接使用 h_part
        H_part = h_part

    # 计算 H_part 中键的最大值
    K_max = max(H_part.keys())

    # 生成 coeffs 列表，包含 H_part 中每个键对应的值
    coeffs = [H_part[i] for i in range(K_max + 1)]

    # 使用 rsolve_hyper 函数求解递推关系方程，得到结果
    result = rsolve_hyper(coeffs, -i_part, n, symbols=True)

    # 如果结果为 None，则返回 None
    if result is None:
        return None

    # 解析结果和符号
    solution, symbols = result

    # 如果 init 为空字典或空列表，则将其置为 None
    if init in ({}, []):
        init = None

    # 如果 symbols 为真且 init 不为空
    if symbols and init is not None:
        # 如果 init 是列表，则转换为字典
        if isinstance(init, list):
            init = {i: init[i] for i in range(len(init))}

        # 初始化方程列表
        equations = []

        # 对 init 中的每个项进行处理
        for k, v in init.items():
            try:
                # 尝试将 k 转换为整数
                i = int(k)
            except TypeError:
                # 如果转换失败，检查 k 是否是函数并与 y.func 相同，若是则提取参数并转换为整数
                if k.is_Function and k.func == y.func:
                    i = int(k.args[0])
                else:
                    # 如果无法转换，则抛出异常
                    raise ValueError("Integer or term expected, got '%s'" % k)

            # 计算方程并添加到方程列表中
            eq = solution.subs(n, i) - v
            if eq.has(S.NaN):
                eq = solution.limit(n, i) - v
            equations.append(eq)

        # 解决方程组，并用结果替换 solution
        result = solve(equations, *symbols)

        # 如果解为空，则返回 None
        if not result:
            return None
        else:
            # 否则用解来替换 solution
            solution = solution.subs(result)

    # 返回最终解 solution
    return solution
```