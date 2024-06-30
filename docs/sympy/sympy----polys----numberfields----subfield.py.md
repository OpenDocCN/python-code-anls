# `D:\src\scipysrc\sympy\sympy\polys\numberfields\subfield.py`

```
"""
Functions in ``polys.numberfields.subfield`` solve the "Subfield Problem" and
allied problems, for algebraic number fields.

Following Cohen (see [Cohen93]_ Section 4.5), we can define the main problem as
follows:

* **Subfield Problem:**

  Given two number fields $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$
  via the minimal polynomials for their generators $\alpha$ and $\beta$, decide
  whether one field is isomorphic to a subfield of the other.

From a solution to this problem flow solutions to the following problems as
well:

* **Primitive Element Problem:**

  Given several algebraic numbers
  $\alpha_1, \ldots, \alpha_m$, compute a single algebraic number $\theta$
  such that $\mathbb{Q}(\alpha_1, \ldots, \alpha_m) = \mathbb{Q}(\theta)$.

* **Field Isomorphism Problem:**

  Decide whether two number fields
  $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$ are isomorphic.

* **Field Membership Problem:**

  Given two algebraic numbers $\alpha$,
  $\beta$, decide whether $\alpha \in \mathbb{Q}(\beta)$, and if so write
  $\alpha = f(\beta)$ for some $f(x) \in \mathbb{Q}[x]$.
"""

# 导入所需模块和函数
from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public

from mpmath import MPContext


def is_isomorphism_possible(a, b):
    """Necessary but not sufficient test for isomorphism. """
    # 获取两个代数数字段的最小多项式的次数
    n = a.minpoly.degree()
    m = b.minpoly.degree()

    # 如果两个字段的最小多项式的次数不相等，则它们不可能同构
    if m % n != 0:
        return False

    # 如果两个字段的最小多项式的次数相等，则可能同构，但需进一步检查
    if n == m:
        return True

    # 获取两个字段的最小多项式的判别式
    da = a.minpoly.discriminant()
    db = b.minpoly.discriminant()

    i, k, half = 1, m//n, db//2

    # 使用素数筛法检查是否可能同构
    while True:
        p = sieve[i]
        P = p**k

        if P > half:
            break

        if ((da % p) % 2) and not (db % P):
            return False

        i += 1

    return True


def field_isomorphism_pslq(a, b):
    """Construct field isomorphism using PSLQ algorithm. """
    # 检查是否支持复数系数，如果不支持则抛出异常
    if not a.root.is_real or not b.root.is_real:
        raise NotImplementedError("PSLQ doesn't support complex coefficients")

    # 获取两个代数数字段的最小多项式
    f = a.minpoly
    g = b.minpoly.replace(f.gen)

    n, m, prev = 100, b.minpoly.degree(), None
    ctx = MPContext()
    # 循环迭代，计算给定范围内的值
    for i in range(1, 5):
        # 使用a对象的root属性计算精确到n位的值，并赋给A
        A = a.root.evalf(n)
        # 使用b对象的root属性计算精确到n位的值，并赋给B
        B = b.root.evalf(n)

        # 构建基础列表，包括1、B以及B的高次幂，以及-A
        basis = [1, B] + [ B**i for i in range(2, m) ] + [-A]

        # 设置上下文的精度为n
        ctx.dps = n
        # 使用pslq方法求解整数线性组合，使得系数尽可能小
        coeffs = ctx.pslq(basis, maxcoeff=10**10, maxsteps=1000)

        # 如果找不到整数线性组合，结束循环
        if coeffs is None:
            # PSLQ无法找到整数线性组合。放弃。
            break

        # 如果当前系数与上一次迭代的系数相同，结束循环
        if coeffs != prev:
            prev = coeffs
        else:
            # 增加精度未产生新的结果。放弃。
            break

        # 将等式转换为 A 作为 B 的多项式的近似表示
        coeffs = [S(c)/coeffs[-1] for c in coeffs[:-1]]

        # 去除多项式系数中的前导零
        while not coeffs[-1]:
            coeffs.pop()

        # 系数逆序排列，构建多项式对象h
        coeffs = list(reversed(coeffs))
        h = Poly(coeffs, f.gen, domain='QQ')

        # 检查 A 是否精确等于 h(B) 组合的余项
        if f.compose(h).rem(g).is_zero:
            # 现在我们知道 h(b) 实际上等于某个a的共轭。
            # 但是根据 A 精确到 h(B) 的近似，我们可以假设共轭就是 a 本身。
            return coeffs
        else:
            # 增加 n 的值继续迭代
            n *= 2

    # 若循环结束仍未找到满足条件的多项式，返回None
    return None
# 定义一个函数，用于计算字段同构的因子
def field_isomorphism_factor(a, b):
    """Construct field isomorphism via factorization. """
    # 对 a 的极小多项式进行因子分解，找到与 b 相关的因子列表
    _, factors = factor_list(a.minpoly, extension=b)
    # 遍历因子列表
    for f, _ in factors:
        # 如果因子是一次因子
        if f.degree() == 1:
            # 任何线性因子 f(x) 表示 a 在 QQ(b) 的某个共轭
            # 我们想知道这个线性因子是否表示 a 本身
            # 设 f = x - c
            c = -f.rep.TC()
            # 将 c 表示为关于 b 的多项式
            coeffs = c.to_sympy_list()
            d, terms = len(coeffs) - 1, []
            # 构造多项式 terms
            for i, coeff in enumerate(coeffs):
                terms.append(coeff*b.root**(d - i))
            r = Add(*terms)
            # 检查是否得到了数值 a
            if a.minpoly.same_root(r, a):
                return coeffs

    # 如果没有线性因子表示 a 在 QQ(b) 中，则 a 实际上不是 QQ(b) 的元素
    return None


# 定义一个公共函数，用于找到一个数域到另一个数域的嵌入
@public
def field_isomorphism(a, b, *, fast=True):
    r"""
    Find an embedding of one number field into another.

    Explanation
    ===========

    This function looks for an isomorphism from $\mathbb{Q}(a)$ onto some
    subfield of $\mathbb{Q}(b)$. Thus, it solves the Subfield Problem.

    Examples
    ========

    >>> from sympy import sqrt, field_isomorphism, I
    >>> print(field_isomorphism(3, sqrt(2)))  # doctest: +SKIP
    [3]
    >>> print(field_isomorphism( I*sqrt(3), I*sqrt(3)/2))  # doctest: +SKIP
    [2, 0]

    Parameters
    ==========

    a : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    b : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    fast : boolean, optional (default=True)
        If ``True``, we first attempt a potentially faster way of computing the
        isomorphism, falling back on a slower method if this fails. If
        ``False``, we go directly to the slower method, which is guaranteed to
        return a result.

    Returns
    =======

    List of rational numbers, or None
        If $\mathbb{Q}(a)$ is not isomorphic to some subfield of
        $\mathbb{Q}(b)$, then return ``None``. Otherwise, return a list of
        rational numbers representing an element of $\mathbb{Q}(b)$ to which
        $a$ may be mapped, in order to define a monomorphism, i.e. an
        isomorphism from $\mathbb{Q}(a)$ to some subfield of $\mathbb{Q}(b)$.
        The elements of the list are the coefficients of falling powers of $b$.

    """
    # 将 a 和 b 转换为符号表达式
    a, b = sympify(a), sympify(b)

    # 如果 a 不是代数数，则将其转换为代数数
    if not a.is_AlgebraicNumber:
        a = AlgebraicNumber(a)

    # 如果 b 不是代数数，则将其转换为代数数
    if not b.is_AlgebraicNumber:
        b = AlgebraicNumber(b)

    # 将 a 和 b 转换为其原始元素
    a = a.to_primitive_element()
    b = b.to_primitive_element()

    # 如果 a 等于 b，则返回 a 的系数
    if a == b:
        return a.coeffs()

    # 计算 a 和 b 的极小多项式的次数
    n = a.minpoly.degree()
    m = b.minpoly.degree()

    # 如果 a 的次数为 1，则返回 a 的根
    if n == 1:
        return [a.root]

    # 如果 m 对 n 取模不为 0，则返回 None
    if m % n != 0:
        return None
    # 如果 fast 参数为 True，则尝试使用 PSLQ 算法进行域同构检测
    if fast:
        try:
            # 调用 field_isomorphism_pslq 函数，尝试找到域同构的结果
            result = field_isomorphism_pslq(a, b)
            
            # 如果找到了非 None 的结果，则直接返回该结果
            if result is not None:
                return result
        # 捕获 NotImplementedError 异常，忽略它
        except NotImplementedError:
            pass

    # 如果 fast 参数为 False 或者 PSLQ 算法未找到结果，则调用 field_isomorphism_factor 函数
    return field_isomorphism_factor(a, b)
def _switch_domain(g, K):
    # 将多项式 g(a, b) = 0 转化为 g(b) = 0，其中 g 属于 Q(a)[x]，h(a) = 0，h 属于 Q(b)[x]。
    # 此函数将多项式 g 转换为 h，其中 Q(b) = K。
    frep = g.rep.inject()  # 将 g 转化为 Q(a) 上的多项式
    hrep = frep.eject(K, front=True)  # 将 Q(a) 上的多项式 h 转化为 Q(b) 上的多项式
    
    return g.new(hrep, g.gens[0])  # 返回转换后的多项式 h


def _linsolve(p):
    # 计算线性多项式的根。
    c, d = p.rep.to_list()  # 将多项式 p 转化为列表形式的系数
    return -d/c  # 返回线性多项式的根


@public
def primitive_element(extension, x=None, *, ex=False, polys=False):
    r"""
    找到一个数域的单一生成元，给定多个生成元。

    Explanation
    ===========

    基本问题是：给定多个代数数 $\alpha_1, \alpha_2, \ldots, \alpha_n$，找到一个代数数 $\theta$，
    使得 $\mathbb{Q}(\alpha_1, \alpha_2, \ldots, \alpha_n) = \mathbb{Q}(\theta)$。

    此函数保证 $\theta$ 是 $\alpha_i$ 的一个非负整数系数的线性组合。

    此外，如果需要，此函数可以告诉你如何将每个 $\alpha_i$ 表示为 $\theta$ 的 $\mathbb{Q}$ 线性组合的幂次。

    Examples
    ========

    >>> from sympy import primitive_element, sqrt, S, minpoly, simplify
    >>> from sympy.abc import x
    >>> f, lincomb, reps = primitive_element([sqrt(2), sqrt(3)], x, ex=True)

    然后 ``lincomb`` 告诉我们用给定生成元 ``sqrt(2)`` 和 ``sqrt(3)`` 的线性组合表示的原始元素。

    >>> print(lincomb)
    [1, 1]

    这意味着原始元素是 $\sqrt{2} + \sqrt{3}$。
    同时 ``f`` 是这个原始元素的最小多项式。

    >>> print(f)
    x**4 - 10*x**2 + 1
    >>> print(minpoly(sqrt(2) + sqrt(3), x))
    x**4 - 10*x**2 + 1

    最后，``reps``（只有在设置了关键字参数 ``ex=True`` 时返回）告诉我们如何将每个生成元 $\sqrt{2}$ 和 $\sqrt{3}$
    表示为原始元素 $\sqrt{2} + \sqrt{3}$ 的 $\mathbb{Q}$ 线性组合的幂次。

    >>> print([S(r) for r in reps[0]])
    [1/2, 0, -9/2, 0]
    >>> theta = sqrt(2) + sqrt(3)
    >>> print(simplify(theta**3/2 - 9*theta/2))
    sqrt(2)
    >>> print([S(r) for r in reps[1]])
    [-1/2, 0, 11/2, 0]
    >>> print(simplify(-theta**3/2 + 11*theta/2))
    sqrt(3)

    Parameters
    ==========

    extension : list of :py:class:`~.Expr`
        每个表达式必须表示一个代数数 $\alpha_i$。
    x : :py:class:`~.Symbol`, optional (default=None)
        期望出现在原始元素 $\theta$ 的计算最小多项式中的符号。如果为 ``None``，使用一个虚拟符号。
    ex : boolean, optional (default=False)
        如果为 ``True``，计算每个 $\alpha_i$ 在 $\theta$ 的幂次表示下的 $\mathbb{Q}$ 线性组合。
    polys : boolean, optional (default=False)
        如果为 ``True``，返回原始元素的最小多项式和每个 $\alpha_i$ 的表示。

    """
    polys : boolean, optional (default=False)
        如果为 ``True``，则返回最小多项式作为 :py:class:`~.Poly` 对象。
        否则返回作为 :py:class:`~.Expr` 对象。

    Returns
    =======

    Pair (f, coeffs) or triple (f, coeffs, reps), where:
        ``f`` 是原始元素的最小多项式。
        ``coeffs`` 给出原始元素作为给定生成元的线性组合。
        ``reps`` 存在当且仅当参数 ``ex=True`` 被传递时，
                是一个列表的列表，每个列表给出原始元素的降幂系数，以恢复一个原始的生成元。

    """
    # 如果 extension 为空，则抛出 ValueError 异常
    if not extension:
        raise ValueError("Cannot compute primitive element for empty extension")
    # 对 extension 中的每个元素进行符号化处理
    extension = [_sympify(ext) for ext in extension]

    # 如果给定了 x，则将其转换为符号化表示，并选择合适的类
    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    # 如果 ex=False，则计算原始元素的最小多项式和其系数
    if not ex:
        gen, coeffs = extension[0], [1]
        g = minimal_polynomial(gen, x, polys=True)
        for ext in extension[1:]:
            # 如果 ext 是有理数，则直接添加 0 到 coeffs 中，并继续下一个循环
            if ext.is_Rational:
                coeffs.append(0)
                continue
            _, factors = factor_list(g, extension=ext)
            g = _choose_factor(factors, x, gen)
            [s], _, g = g.sqf_norm()
            gen += s*ext
            coeffs.append(s)

        # 根据 polys 的值返回不同类型的结果
        if not polys:
            return g.as_expr(), coeffs
        else:
            return cls(g), coeffs

    # 否则，计算原始元素的最小多项式和其系数，并创建代数域 K
    gen, coeffs = extension[0], [1]
    f = minimal_polynomial(gen, x, polys=True)
    K = QQ.algebraic_field((f, gen))  # incrementally constructed field
    reps = [K.unit]  # representations of extension elements in K
    for ext in extension[1:]:
        if ext.is_Rational:
            coeffs.append(0)    # rational ext is not included in the expression of a primitive element
            reps.append(K.convert(ext))    # but it is included in reps
            continue
        p = minimal_polynomial(ext, x, polys=True)
        L = QQ.algebraic_field((p, ext))
        _, factors = factor_list(f, domain=L)
        f = _choose_factor(factors, x, gen)
        [s], g, f = f.sqf_norm()
        gen += s*ext
        coeffs.append(s)
        K = QQ.algebraic_field((f, gen))
        h = _switch_domain(g, K)
        erep = _linsolve(h.gcd(p))  # ext as element of K
        ogen = K.unit - s*erep  # old gen as element of K
        reps = [dup_eval(_.to_list(), ogen, K) for _ in reps] + [erep]

    # 如果所有扩展都是有理数，则处理成员 H
    if K.ext.root.is_Rational:  # all extensions are rational
        H = [K.convert(_).rep for _ in extension]
        coeffs = [0]*len(extension)
        f = cls(x, domain=QQ)
    else:
        H = [_.to_list() for _ in reps]
    
    # 根据 polys 的值返回不同类型的结果
    if not polys:
        return f.as_expr(), coeffs, H
    else:
        return f, coeffs, H
# 声明一个公共函数，用于将一个代数数在由另一个代数数生成的域中表达
@public
def to_number_field(extension, theta=None, *, gen=None, alias=None):
    r"""
    Express one algebraic number in the field generated by another.

    Explanation
    ===========

    Given two algebraic numbers $\eta, \theta$, this function either expresses
    $\eta$ as an element of $\mathbb{Q}(\theta)$, or else raises an exception
    if $\eta \not\in \mathbb{Q}(\theta)$.

    This function is essentially just a convenience, utilizing
    :py:func:`~.field_isomorphism` (our solution of the Subfield Problem) to
    solve this, the Field Membership Problem.

    As an additional convenience, this function allows you to pass a list of
    algebraic numbers $\alpha_1, \alpha_2, \ldots, \alpha_n$ instead of $\eta$.
    It then computes $\eta$ for you, as a solution of the Primitive Element
    Problem, using :py:func:`~.primitive_element` on the list of $\alpha_i$.

    Examples
    ========

    >>> from sympy import sqrt, to_number_field
    >>> eta = sqrt(2)
    >>> theta = sqrt(2) + sqrt(3)
    >>> a = to_number_field(eta, theta)
    >>> print(type(a))
    <class 'sympy.core.numbers.AlgebraicNumber'>
    >>> a.root
    sqrt(2) + sqrt(3)
    >>> print(a)
    sqrt(2)
    >>> a.coeffs()
    [1/2, 0, -9/2, 0]

    We get an :py:class:`~.AlgebraicNumber`, whose ``.root`` is $\theta$, whose
    value is $\eta$, and whose ``.coeffs()`` show how to write $\eta$ as a
    $\mathbb{Q}$-linear combination in falling powers of $\theta$.

    Parameters
    ==========

    extension : :py:class:`~.Expr` or list of :py:class:`~.Expr`
        Either the algebraic number that is to be expressed in the other field,
        or else a list of algebraic numbers, a primitive element for which is
        to be expressed in the other field.
    theta : :py:class:`~.Expr`, None, optional (default=None)
        If an :py:class:`~.Expr` representing an algebraic number, behavior is
        as described under **Explanation**. If ``None``, then this function
        reduces to a shorthand for calling :py:func:`~.primitive_element` on
        ``extension`` and turning the computed primitive element into an
        :py:class:`~.AlgebraicNumber`.
    gen : :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the generator symbol for the minimal
        polynomial in the returned :py:class:`~.AlgebraicNumber`.
    alias : str, :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the alias symbol for the returned
        :py:class:`~.AlgebraicNumber`.

    Returns
    =======

    AlgebraicNumber
        Belonging to $\mathbb{Q}(\theta)$ and equaling $\eta$.

    Raises
    ======

    IsomorphismFailed
        If $\eta \not\in \mathbb{Q}(\theta)$.

    See Also
    ========

    field_isomorphism
    primitive_element

    """
    # 如果 extension 可迭代，转换为列表；否则将其作为单个元素的列表
    if hasattr(extension, '__iter__'):
        extension = list(extension)
    else:
        extension = [extension]
    # 如果 extension 的长度为 1，并且其唯一元素是一个元组，则返回一个代数数对象
    if len(extension) == 1 and isinstance(extension[0], tuple):
        # 调用 AlgebraicNumber 类创建一个代数数对象，使用给定的元组作为参数，并可能提供别名
        return AlgebraicNumber(extension[0], alias=alias)

    # 使用 primitive_element 函数计算 extension 的最小多项式和系数
    minpoly, coeffs = primitive_element(extension, gen, polys=True)

    # 计算代数数的根，根据系数和 extension 中的元素相乘求和
    root = sum(coeff * ext for coeff, ext in zip(coeffs, extension))

    # 如果没有提供 theta 参数，则返回一个代数数对象，其表示为 (minpoly, root)，并可能提供别名
    if theta is None:
        return AlgebraicNumber((minpoly, root), alias=alias)
    else:
        # 将 theta 转换为符号表达式
        theta = sympify(theta)

        # 如果 theta 不是代数数对象，则尝试创建一个代数数对象，使用 theta 作为参数，并可能提供别名
        if not theta.is_AlgebraicNumber:
            theta = AlgebraicNumber(theta, gen=gen, alias=alias)

        # 使用 field_isomorphism 函数计算 root 和 theta 之间的系数映射
        coeffs = field_isomorphism(root, theta)

        # 如果计算得到了系数映射，则返回一个代数数对象，其表示为 (theta, coeffs)，并可能提供别名
        if coeffs is not None:
            return AlgebraicNumber(theta, coeffs, alias=alias)
        else:
            # 如果未能计算出系数映射，则抛出 IsomorphismFailed 异常，指示计算失败的原因
            raise IsomorphismFailed(
                "%s is not in a subfield of %s" % (root, theta.root))
```