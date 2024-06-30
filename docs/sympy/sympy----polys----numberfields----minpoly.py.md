# `D:\src\scipysrc\sympy\sympy\polys\numberfields\minpoly.py`

```
# 导入必要的模块和函数
from functools import reduce
# 导入符号计算相关的模块和类
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
# 导入数学函数，如指数函数、平方根、立方根、三角函数等
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
# 导入多项式相关的类和函数
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
    NotAlgebraic,
    GeneratorsError,
)
from sympy.polys.polytools import (
    Poly, PurePoly, invert, factor_list, groebner, resultant,
    degree, poly_from_expr, parallel_poly_from_expr, lcm
)
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
    numbered_symbols, public, sift
)

# 定义一个函数，根据给定的因子列表和根 v，选择一个具有根 v 的因子
def _choose_factor(factors, x, v, dom=QQ, prec=200, bound=5):
    """
    Return a factor having root ``v``
    It is assumed that one of the factors has root ``v``.
    """

    # 如果因子列表中的元素是元组，则取出元组的第一个元素作为因子
    if isinstance(factors[0], tuple):
        factors = [f[0] for f in factors]
    
    # 如果因子列表只有一个元素，直接返回该因子
    if len(factors) == 1:
        return factors[0]

    # 初始化精度和点集
    prec1 = 10
    points = {}
    # 如果域对象 dom 有 symbols 属性，则获取符号列表
    symbols = dom.symbols if hasattr(dom, 'symbols') else []

    # 当前精度小于等于给定的最大精度时执行循环
    while prec1 <= prec:
        # 将因子列表中每个因子的表达式替换为将 x 替换为 v 后的表达式
        fe = [f.as_expr().xreplace({x:v}) for f in factors]
        
        # 如果 v 是数值，则对每个因子进行数值计算以给定精度
        if v.is_number:
            fe = [f.n(prec) for f in fe]

        # 给符号赋值 [0, bound) 的整数值
        for n in subsets(range(bound), k=len(symbols), repetition=True):
            for s, i in zip(symbols, n):
                points[s] = i

            # 在这些点上计算表达式的值
            candidates = [(abs(f.subs(points).n(prec1)), i)
                           for i, f in enumerate(fe)]

            # 如果得到无效数值（如由于除零而导致的错误），则重新尝试
            if any(i in _illegal for i, _ in candidates):
                continue

            # 找到最小的两个候选值，如果它们显著不同，则假设找到了当 v 替换后变为 0 的因子
            can = sorted(candidates)
            (a, ix), (b, _) = can[:2]
            if b > a * 10**6:  # XXX what to use?
                return factors[ix]

        # 增加精度
        prec1 *= 2
    # 抛出未实现错误，提示多个候选项作为 %s 的最小多项式
    raise NotImplementedError("multiple candidates for the minimal polynomial of %s" % v)
def _is_sum_surds(p):
    # 如果 p 是一个加法表达式，则取其参数列表；否则将 p 包装成列表
    args = p.args if p.is_Add else [p]
    # 遍历参数列表
    for y in args:
        # 如果参数的平方不是有理数或者参数不是扩展实数，则返回 False
        if not ((y**2).is_Rational and y.is_extended_real):
            return False
    # 若所有参数满足条件，则返回 True
    return True


def _separate_sq(p):
    """
    ``_minimal_polynomial_sq`` 的辅助函数

    选择一个有理数 ``g``，使得多项式 ``p`` 包含一组平方根项，这些项的最大公约数等于 ``g``，
    并且包含一组与 ``g`` 互质的平方根项；然后进行域范数以消除 ``sqrt(g)``。

    参见 simplify.simplify.split_surds 和 polytools.sqf_norm。

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.minpoly import _separate_sq
    >>> p= -x + sqrt(2) + sqrt(3) + sqrt(7)
    >>> p = _separate_sq(p); p
    -x**2 + 2*sqrt(3)*x + 2*sqrt(7)*x - 2*sqrt(21) - 8
    >>> p = _separate_sq(p); p
    -x**4 + 4*sqrt(7)*x**3 - 32*x**2 + 8*sqrt(7)*x + 20
    >>> p = _separate_sq(p); p
    -x**8 + 48*x**6 - 536*x**4 + 1728*x**2 - 400

    """
    def is_sqrt(expr):
        # 判断表达式是否是平方根表达式
        return expr.is_Pow and expr.exp is S.Half
    # p = c1*sqrt(q1) + ... + cn*sqrt(qn) -> a = [(c1, q1), .., (cn, qn)]
    a = []
    for y in p.args:
        if not y.is_Mul:
            if is_sqrt(y):
                a.append((S.One, y**2))
            elif y.is_Atom:
                a.append((y, S.One))
            elif y.is_Pow and y.exp.is_integer:
                a.append((y, S.One))
            else:
                raise NotImplementedError
        else:
            T, F = sift(y.args, is_sqrt, binary=True)
            a.append((Mul(*F), Mul(*T)**2))
    a.sort(key=lambda z: z[1])
    if a[-1][1] is S.One:
        # 如果没有平方根项，则直接返回 p
        return p
    surds = [z for y, z in a]
    # 找到第一个不是 1 的项的索引
    for i in range(len(surds)):
        if surds[i] != 1:
            break
    # 导入 _split_gcd 函数来计算最大公约数
    from sympy.simplify.radsimp import _split_gcd
    g, b1, b2 = _split_gcd(*surds[i:])
    a1 = []
    a2 = []
    for y, z in a:
        if z in b1:
            a1.append(y*z**S.Half)
        else:
            a2.append(y*z**S.Half)
    p1 = Add(*a1)
    p2 = Add(*a2)
    # 计算差分表达式，去除平方根
    p = _mexpand(p1**2) - _mexpand(p2**2)
    return p

def _minimal_polynomial_sq(p, n, x):
    """
    返回一个包含一组平方根之和的``n``次根的最小多项式，如果失败则返回 ``None``。

    Parameters
    ==========

    p : 平方根的和
    n : 正整数
    x : 返回多项式的变量

    Examples
    ========

    >>> from sympy.polys.numberfields.minpoly import _minimal_polynomial_sq
    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> q = 1 + sqrt(2) + sqrt(3)
    >>> _minimal_polynomial_sq(q, 3, x)
    x**12 - 4*x**9 - 4*x**6 + 16*x**3 - 8

    """
    p = sympify(p)
    n = sympify(n)
    # 检查参数是否合法，如果不合法则返回 None
    if not n.is_Integer or not n > 0 or not _is_sum_surds(p):
        return None
    pn = p**Rational(1, n)
    # 消除平方根项
    p -= x
    # 进入无限循环，直到条件不再满足
    while 1:
        # 对多项式 p 调用 _separate_sq 函数，将其分离成 p1
        p1 = _separate_sq(p)
        # 如果分离后的 p1 与原 p 相同，则退出循环
        if p1 is p:
            # 将 p 替换为 p1 中变量 x 的 n 次幂，并结束循环
            p = p1.subs({x:x**n})
            break
        else:
            # 否则，将 p 替换为 p1 继续循环
            p = p1

    # 如果 n 等于 1
    if n == 1:
        # 将 p 转换为多项式对象
        p1 = Poly(p)
        # 如果 p 中 x^p1.degree(x) 的系数小于 0，则将 p 取反
        if p.coeff(x**p1.degree(x)) < 0:
            p = -p
        # 计算 p 的原始部分，并返回结果
        p = p.primitive()[1]
        return p
    
    # 根据构造，p 有根 pn
    # 找出 p 的因子分解列表中的因子
    factors = factor_list(p)[1]

    # 从 factors 中选择合适的因子，使其在 x = pn 处为零
    result = _choose_factor(factors, x, pn)
    return result
# 返回对 ``ex`` 的 ``pw`` 次幂的最小多项式

def _minpoly_pow(ex, pw, x, dom, mp=None):
    """
    Returns ``minpoly(ex**pw, x)``

    Parameters
    ==========

    ex : algebraic element     # 代数元素
    pw : rational number       # 有理数，表示幂次
    x : indeterminate of the polynomial   # 多项式的不定元
    dom: ground domain         # 域，表示域的类型
    mp : minimal polynomial of ``p``    # ``p`` 的最小多项式
    """
    y = Dummy(str(x))   # 创建一个新的符号变量 y，用于代换

    if mp is None:
        mp = _minpoly_compose(ex, x, dom)   # 计算 ex 的最小多项式

    mp = mp.subs({x: y})    # 将 x 替换为 y

    if pw == 1:
        # 如果幂次为 1，直接返回最小多项式 mp
        return mp

    # 计算 ex^pw 的最小多项式
    r = resultant(mp, pw * y - ex, gens=[y, x]) if dom != QQ else rs_compose_add(pw * y - ex, x)
    r = expr_from_dict(r.as_expr_dict(), x)   # 将结果转换为表达式

    # 计算 mp 的次数和 pw 的次数
    deg_mp = degree(mp, x)
    deg_pw = Rational(pw).q

    if deg_pw == 1 or deg_mp == 1:
        # 如果 pw 或者 mp 的次数为 1，则 r 是不可约的，直接返回
        return r

    r = Poly(r, x, domain=dom)   # 将结果转换为多项式
    _, factors = r.factor_list()    # 对 r 进行因式分解
    res = _choose_factor(factors, x, ex**pw, dom)   # 选择符合条件的因子
    return res.as_expr()    # 返回结果表达式
    pw = sympify(pw)

将输入的参数 `pw` 转换为 SymPy 符号表达式，确保可以进行符号计算。


    if not mp:

如果参数 `mp` 为假值（例如 `None`），执行以下操作。


        mp = _minpoly_compose(ex, x, dom)

调用 `_minpoly_compose` 函数，用给定的 `ex`、`x` 和 `dom` 参数计算最小多项式。


    if not pw.is_rational:

如果 `pw` 不是有理数，抛出 `NotAlgebraic` 异常，表示该参数不是一个代数元素。


        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)

抛出异常，指示 `ex` 参数似乎不是一个代数元素。


    if pw < 0:

如果 `pw` 小于零，执行以下操作。


        if mp == x:

如果 `mp` 等于 `x`，抛出 `ZeroDivisionError` 异常，表示 `ex` 是零。


            raise ZeroDivisionError('%s is zero' % ex)

抛出异常，指示 `ex` 是零。


        mp = _invertx(mp, x)

调用 `_invertx` 函数，对 `mp` 和 `x` 进行反转。


        if pw == -1:

如果 `pw` 等于 `-1`，返回 `mp`。


            return mp

直接返回 `mp`。


        pw = -pw
        ex = 1/ex

将 `pw` 取反，同时将 `ex` 取倒数。


    y = Dummy(str(x))

创建一个新的虚拟符号 `y`，其名称为 `x` 的字符串形式。


    mp = mp.subs({x: y})

用字典 `{x: y}` 替换 `mp` 中的 `x`，得到新的 `mp`。


    n, d = pw.as_numer_denom()

将 `pw` 转换为分子 `n` 和分母 `d`。


    res = Poly(resultant(mp, x**d - y**n, gens=[y]), x, domain=dom)

计算 `resultant(mp, x**d - y**n, gens=[y])` 的结果，并将其作为 `x` 的多项式，使用给定的域 `dom`。


    _, factors = res.factor_list()

将 `res` 的因式分解结果保存在 `factors` 中。


    res = _choose_factor(factors, x, ex**pw, dom)

调用 `_choose_factor` 函数，根据给定的因子列表 `factors`、`x` 和 `ex**pw`，在给定的域 `dom` 中选择因子。


    return res.as_expr()

将 `res` 转换为表达式并返回。
def _minpoly_add(x, dom, *a):
    """
    返回 ``minpoly(Add(*a), dom, x)``
    """
    # 调用 _minpoly_op_algebraic_element 函数计算 Add(*a) 的最小多项式
    mp = _minpoly_op_algebraic_element(Add, a[0], a[1], x, dom)
    # 计算前两个参数的和
    p = a[0] + a[1]
    for px in a[2:]:
        # 更新 mp 为当前 p 和 px 的最小多项式
        mp = _minpoly_op_algebraic_element(Add, p, px, x, dom, mp1=mp)
        # 更新 p 为 p 和 px 的和
        p = p + px
    return mp


def _minpoly_mul(x, dom, *a):
    """
    返回 ``minpoly(Mul(*a), dom, x)``
    """
    # 调用 _minpoly_op_algebraic_element 函数计算 Mul(*a) 的最小多项式
    mp = _minpoly_op_algebraic_element(Mul, a[0], a[1], x, dom)
    # 计算前两个参数的乘积
    p = a[0] * a[1]
    for px in a[2:]:
        # 更新 mp 为当前 p 和 px 的最小多项式
        mp = _minpoly_op_algebraic_element(Mul, p, px, x, dom, mp1=mp)
        # 更新 p 为 p 和 px 的乘积
        p = p * px
    return mp


def _minpoly_sin(ex, x):
    """
    返回 ``sin(ex)`` 的最小多项式
    参见 https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    # 将 ex 的第一个参数表示为 c * a，其中 a 是 pi
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            n = c.q
            q = sympify(n)
            if q.is_prime:
                # 对于 a = pi * p / q，其中 q 是奇素数，使用 chebyshevt
                # 写作 sin(q*a) = mp(sin(a))*sin(a)
                # mp(x) 的根是 sin(pi*p/q)，p = 1,...,q-1
                a = dup_chebyshevt(n, ZZ)
                return Add(*[x**(n - i - 1) * a[i] for i in range(n)])
            if c.p == 1:
                if q == 9:
                    return 64*x**6 - 96*x**4 + 36*x**2 - 3

            if n % 2 == 1:
                # 对于 a = pi * p / q，其中 q 是奇数，使用
                # sin(q*a) = 0 可以看出最小多项式必须是 dup_chebyshevt(n, ZZ) 的因子
                a = dup_chebyshevt(n, ZZ)
                a = [x**(n - i) * a[i] for i in range(n + 1)]
                r = Add(*a)
                _, factors = factor_list(r)
                res = _choose_factor(factors, x, ex)
                return res

            expr = ((1 - cos(2*c*pi))/2)**S.Half
            res = _minpoly_compose(expr, x, QQ)
            return res

    raise NotAlgebraic("%s 不是一个代数元素" % ex)


def _minpoly_cos(ex, x):
    """
    返回 ``cos(ex)`` 的最小多项式
    参见 https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    # 将 ex 的第一个参数表示为 c * a，其中 a 是 pi
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            if c.p == 1:
                if c.q == 7:
                    return 8*x**3 - 4*x**2 - 4*x + 1
                if c.q == 9:
                    return 8*x**3 - 6*x - 1
            elif c.p == 2:
                q = sympify(c.q)
                if q.is_prime:
                    s = _minpoly_sin(ex, x)
                    return _mexpand(s.subs({x: sqrt((1 - x)/2)}))

            # 对于 a = pi * p / q，cos(q*a) = T_q(cos(a)) = (-1)**p
            n = int(c.q)
            a = dup_chebyshevt(n, ZZ)
            a = [x**(n - i) * a[i] for i in range(n + 1)]
            r = Add(*a) - (-1)**c.p
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    raise NotAlgebraic("%s 不是一个代数元素" % ex)
# 返回 ``tan(ex)`` 的最小多项式
def _minpoly_tan(ex, x):
    # 提取表达式 ``ex`` 的首项系数和乘积项
    c, a = ex.args[0].as_coeff_Mul()
    # 如果乘积项是 pi
    if a is pi:
        # 如果首项系数是有理数
        if c.is_rational:
            # 将系数乘以 2
            c = c * 2
            # 将 c 转换为整数
            n = int(c.q)
            # 根据 c.p 的奇偶性决定初始值 a
            a = n if c.p % 2 == 0 else 1
            terms = []
            # 遍历范围从 ((c.p+1)%2) 到 n+1，步长为 2
            for k in range((c.p+1)%2, n+1, 2):
                # 向 terms 列表添加 a*x**k
                terms.append(a*x**k)
                # 更新 a 的值
                a = -(a*(n-k-1)*(n-k)) // ((k+1)*(k+2))

            # 将 terms 中的项求和得到 r
            r = Add(*terms)
            # 对 r 进行因式分解并选择合适的因子
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    # 抛出 NotAlgebraic 异常，表明 ex 不是代数元素
    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


# 返回 ``exp(ex)`` 的最小多项式
def _minpoly_exp(ex, x):
    # 提取表达式 ``ex`` 的首项系数和乘积项
    c, a = ex.args[0].as_coeff_Mul()
    # 如果乘积项是 I*pi
    if a == I*pi:
        # 如果首项系数是有理数
        if c.is_rational:
            # 将 c.q 转换为符号
            q = sympify(c.q)
            # 如果 c.p 是 1 或 -1
            if c.p == 1 or c.p == -1:
                # 根据 q 的值返回对应的多项式
                if q == 3:
                    return x**2 - x + 1
                if q == 4:
                    return x**4 + 1
                if q == 6:
                    return x**4 - x**2 + 1
                if q == 8:
                    return x**8 + 1
                if q == 9:
                    return x**6 - x**3 + 1
                if q == 10:
                    return x**8 - x**6 + x**4 - x**2 + 1
                if q.is_prime:
                    s = 0
                    for i in range(q):
                        s += (-x)**i
                    return s

            # 如果不满足上述条件，则按照 x**(2*q) = product(factors) 进行处理
            factors = [cyclotomic_poly(i, x) for i in divisors(2*q)]
            mp = _choose_factor(factors, x, ex)
            return mp
        else:
            # 抛出 NotAlgebraic 异常，表明 ex 不是代数元素
            raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    # 抛出 NotAlgebraic 异常，表明 ex 不是代数元素
    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


# 返回 ``CRootOf`` 对象的最小多项式
def _minpoly_rootof(ex, x):
    # 提取表达式 p，并替换 p 中的 ex.poly.gens[0] 为 x
    p = ex.expr
    p = p.subs({ex.poly.gens[0]:x})
    # 对 p 进行因式分解，并选择合适的因子
    _, factors = factor_list(p, x)
    result = _choose_factor(factors, x, ex)
    return result


# 计算代数元素的最小多项式，使用最小多项式上的操作
def _minpoly_compose(ex, x, dom):
    # 如果 ex 是有理数，则返回对应的多项式
    if ex.is_Rational:
        return ex.q*x - ex.p
    # 如果 ex 是虚数单位 I
    if ex is I:
        # 对 x**2 + 1 进行因式分解，并根据结果长度返回 x**2 + 1 或 x - I
        _, factors = factor_list(x**2 + 1, x, domain=dom)
        return x**2 + 1 if len(factors) == 1 else x - I
    if ex is S.GoldenRatio:
        # 如果表达式 ex 是黄金比例常数
        _, factors = factor_list(x**2 - x - 1, x, domain=dom)
        # 对 x^2 - x - 1 进行因式分解
        if len(factors) == 1:
            # 如果只有一个因子
            return x**2 - x - 1
            # 返回 x^2 - x - 1
        else:
            # 如果有多个因子，则选择合适的因子
            return _choose_factor(factors, x, (1 + sqrt(5))/2, dom=dom)

    if ex is S.TribonacciConstant:
        # 如果表达式 ex 是Tribonacci常数
        _, factors = factor_list(x**3 - x**2 - x - 1, x, domain=dom)
        # 对 x^3 - x^2 - x - 1 进行因式分解
        if len(factors) == 1:
            # 如果只有一个因子
            return x**3 - x**2 - x - 1
            # 返回 x^3 - x^2 - x - 1
        else:
            # 如果有多个因子，则选择合适的因子
            fac = (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3
            return _choose_factor(factors, x, fac, dom=dom)

    if hasattr(dom, 'symbols') and ex in dom.symbols:
        # 如果 dom 具有 'symbols' 属性且表达式 ex 是其符号之一
        return x - ex
        # 返回 x - ex

    if dom.is_QQ and _is_sum_surds(ex):
        # 如果 dom 是有理数域且 ex 是根式的和
        # 消除平方根
        ex -= x
        while 1:
            ex1 = _separate_sq(ex)
            # 将 ex 中的平方项分离
            if ex1 is ex:
                return ex
                # 如果无法再分离，则返回原始表达式
            else:
                ex = ex1
                # 否则继续分离

    if ex.is_Add:
        # 如果 ex 是加法表达式
        res = _minpoly_add(x, dom, *ex.args)
        # 对加法表达式使用 _minpoly_add 函数
    elif ex.is_Mul:
        # 如果 ex 是乘法表达式
        f = Factors(ex).factors
        # 获取 ex 的因子
        r = sift(f.items(), lambda itx: itx[0].is_Rational and itx[1].is_Rational)
        # 将因子按是否为有理数分为两组
        if r[True] and dom == QQ:
            # 如果存在有理数因子并且 dom 是有理数域
            ex1 = Mul(*[bx**ex for bx, ex in r[False] + r[None]])
            # 将非有理数因子和空因子组合成乘积
            r1 = dict(r[True])
            dens = [y.q for y in r1.values()]
            lcmdens = reduce(lcm, dens, 1)
            neg1 = S.NegativeOne
            expn1 = r1.pop(neg1, S.Zero)
            nums = [base**(y.p*lcmdens // y.q) for base, y in r1.items()]
            ex2 = Mul(*nums)
            mp1 = minimal_polynomial(ex1, x)
            # 计算 ex1 的最小多项式
            mp2 = ex2.q*x**lcmdens - ex2.p*neg1**(expn1*lcmdens)
            ex2 = neg1**expn1 * ex2**Rational(1, lcmdens)
            res = _minpoly_op_algebraic_element(Mul, ex1, ex2, x, dom, mp1=mp1, mp2=mp2)
            # 使用 _minpoly_op_algebraic_element 处理代数元素
        else:
            res = _minpoly_mul(x, dom, *ex.args)
            # 否则使用 _minpoly_mul 处理乘法表达式
    elif ex.is_Pow:
        # 如果 ex 是幂次表达式
        res = _minpoly_pow(ex.base, ex.exp, x, dom)
        # 使用 _minpoly_pow 处理幂次表达式
    elif ex.__class__ is sin:
        # 如果 ex 是正弦函数
        res = _minpoly_sin(ex, x)
        # 使用 _minpoly_sin 处理正弦函数
    elif ex.__class__ is cos:
        # 如果 ex 是余弦函数
        res = _minpoly_cos(ex, x)
        # 使用 _minpoly_cos 处理余弦函数
    elif ex.__class__ is tan:
        # 如果 ex 是正切函数
        res = _minpoly_tan(ex, x)
        # 使用 _minpoly_tan 处理正切函数
    elif ex.__class__ is exp:
        # 如果 ex 是指数函数
        res = _minpoly_exp(ex, x)
        # 使用 _minpoly_exp 处理指数函数
    elif ex.__class__ is CRootOf:
        # 如果 ex 是 CRootOf 类型的对象
        res = _minpoly_rootof(ex, x)
        # 使用 _minpoly_rootof 处理 CRootOf 对象
    else:
        # 如果 ex 是其他类型的对象，则抛出异常
        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
        # 抛出 NotAlgebraic 异常，提示 ex 不是代数元素
    return res
    # 返回处理后的结果
# 定义一个公共函数，计算代数元素的最小多项式
@public
def minimal_polynomial(ex, x=None, compose=True, polys=False, domain=None):
    """
    Computes the minimal polynomial of an algebraic element.

    Parameters
    ==========

    ex : Expr
        Element or expression whose minimal polynomial is to be calculated.

    x : Symbol, optional
        Independent variable of the minimal polynomial

    compose : boolean, optional (default=True)
        Method to use for computing minimal polynomial. If ``compose=True``
        (default) then ``_minpoly_compose`` is used, if ``compose=False`` then
        groebner bases are used.

    polys : boolean, optional (default=False)
        If ``True`` returns a ``Poly`` object else an ``Expr`` object.

    domain : Domain, optional
        Ground domain

    Notes
    =====

    By default ``compose=True``, the minimal polynomial of the subexpressions of ``ex``
    are computed, then the arithmetic operations on them are performed using the resultant
    and factorization.
    If ``compose=False``, a bottom-up algorithm is used with ``groebner``.
    The default algorithm stalls less frequently.

    If no ground domain is given, it will be generated automatically from the expression.

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, solve, QQ
    >>> from sympy.abc import x, y

    >>> minimal_polynomial(sqrt(2), x)
    x**2 - 2
    >>> minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2)))
    x - sqrt(2)
    >>> minimal_polynomial(sqrt(2) + sqrt(3), x)
    x**4 - 10*x**2 + 1
    >>> minimal_polynomial(solve(x**3 + x + 3)[0], x)
    x**3 + x + 3
    >>> minimal_polynomial(sqrt(y), x)
    x**2 - y

    """

    # 将输入元素转换为符号表达式
    ex = sympify(ex)
    if ex.is_number:
        # 对于数字，尝试进行扩展（issue 8354）
        ex = _mexpand(ex, recursive=True)
    
    # 使用前序遍历检查是否有代数数
    for expr in preorder_traversal(ex):
        if expr.is_AlgebraicNumber:
            compose = False
            break

    # 如果提供了独立变量 x，则将其转换为符号，否则创建一个虚拟符号 x
    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    # 如果没有指定域，则根据表达式自动生成
    if not domain:
        if ex.free_symbols:
            domain = FractionField(QQ, list(ex.free_symbols))
        else:
            domain = QQ
    
    # 检查是否在地面域中存在符号 x
    if hasattr(domain, 'symbols') and x in domain.symbols:
        raise GeneratorsError("the variable %s is an element of the ground "
                              "domain %s" % (x, domain))

    # 如果使用 compose 方法，则调用 _minpoly_compose 函数计算最小多项式
    if compose:
        result = _minpoly_compose(ex, x, domain)
        # 提取最高次项系数，确保结果为原始形式
        result = result.primitive()[1]
        c = result.coeff(x**degree(result, x))
        if c.is_negative:
            result = expand_mul(-result)
        return cls(result, x, field=True) if polys else result.collect(x)

    # 如果不使用 compose 方法，则使用 groebner 方法计算最小多项式
    if not domain.is_QQ:
        raise NotImplementedError("groebner method only works for QQ")

    result = _minpoly_groebner(ex, x, cls)
    return cls(result, x, field=True) if polys else result.collect(x)


def _minpoly_groebner(ex, x, cls):
    """
    Computes minimal polynomial using Groebner bases.

    Parameters
    ==========

    ex : Expr
        Element or expression whose minimal polynomial is to be calculated.

    x : Symbol
        Independent variable of the minimal polynomial

    cls : Class
        Polynomial class to use for representation

    Returns
    =======

    Poly or Expr
        Minimal polynomial of ``ex``

    """
    # 计算代数数的最小多项式，使用Groebner基方法
    
    generator = numbered_symbols('a', cls=Dummy)
    # 创建一个生成器，生成以'a'开头的符号
    
    mapping, symbols = {}, {}
    # 初始化两个空字典，用于存储映射关系和符号集合
    
    def update_mapping(ex, exp, base=None):
        # 更新映射函数，将表达式ex映射到一个新的符号a上，并存储映射关系到mapping中
        a = next(generator)
        symbols[ex] = a
    
        if base is not None:
            mapping[ex] = a**exp + base
        else:
            mapping[ex] = exp.as_expr(a)
    
        return a
    
    def simpler_inverse(ex):
        """
        如果倒数更有可能使最小多项式算法更好地工作，则返回True
        """
        if ex.is_Pow:
            if (1/ex.exp).is_integer and ex.exp < 0:
                if ex.base.is_Add:
                    return True
        if ex.is_Mul:
            hit = True
            for p in ex.args:
                if p.is_Add:
                    return False
                if p.is_Pow:
                    if p.base.is_Add and p.exp > 0:
                        return False
    
            if hit:
                return True
        return False
    
    inverted = False
    # 初始化倒数标志为False
    ex = expand_multinomial(ex)
    # 将表达式ex展开成多项式
    
    if ex.is_AlgebraicNumber:
        # 如果ex是代数数，直接返回其元素的最小多项式表达式
        return ex.minpoly_of_element().as_expr(x)
    elif ex.is_Rational:
        # 如果ex是有理数，计算其有理数表达式的多项式形式
        result = ex.q*x - ex.p
    else:
        inverted = simpler_inverse(ex)
        # 判断是否使用倒数来简化表达式
        if inverted:
            ex = ex**-1
    
        res = None
        if ex.is_Pow and (1/ex.exp).is_Integer:
            # 如果ex是幂次方，并且指数的倒数是整数
            n = 1/ex.exp
            res = _minimal_polynomial_sq(ex.base, n, x)
    
        elif _is_sum_surds(ex):
            # 如果ex是有理数根的和
            res = _minimal_polynomial_sq(ex, S.One, x)
    
        if res is not None:
            result = res
    
        if res is None:
            # 对ex进行自底向上的扫描
            bus = bottom_up_scan(ex)
            F = [x - bus] + list(mapping.values())
            G = groebner(F, list(symbols.values()) + [x], order='lex')
    
            _, factors = factor_list(G[-1])
            # 按照构造方式，G[-1]具有根`ex`
            result = _choose_factor(factors, x, ex)
    
    if inverted:
        # 如果进行了倒数操作，则对结果进行反转处理
        result = _invertx(result, x)
        if result.coeff(x**degree(result, x)) < 0:
            result = expand_mul(-result)
    
    return result
# 定义了一个公共函数装饰器，使函数可以被外部访问
@public
# 定义了一个函数 minpoly，用于计算给定表达式的最小多项式
def minpoly(ex, x=None, compose=True, polys=False, domain=None):
    """This is a synonym for :py:func:`~.minimal_polynomial`."""
    # 调用 minimal_polynomial 函数来计算给定表达式的最小多项式，并返回结果
    return minimal_polynomial(ex, x=x, compose=compose, polys=polys, domain=domain)
```