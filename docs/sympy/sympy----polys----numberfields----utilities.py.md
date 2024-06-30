# `D:\src\scipysrc\sympy\sympy\polys\numberfields\utilities.py`

```
"""Utilities for algebraic number theory. """

# 导入必要的模块和函数
from sympy.core.sympify import sympify
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.minpoly import minpoly
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.utilities.decorator import public
from sympy.utilities.lambdify import lambdify

# 导入 mpmath 中的 mp 模块
from mpmath import mp


# 判断一个参数是否为合适的有理数类型
def is_rat(c):
    r"""
    Test whether an argument is of an acceptable type to be used as a rational
    number.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int``, :ref:`ZZ`, or :ref:`QQ`.

    See Also
    ========

    is_int

    """
    # 检查参数是否是 int 类型，或者 ZZ 类型，或者 QQ 类型
    # ``c in QQ`` 过于宽容（例如 ``3.14 in QQ`` 是 ``True``）
    # ``QQ.of_type(c)`` 过于严格（例如 ``QQ.of_type(3)`` 是 ``False``）
    # 如果安装了 gmpy2，则 ``ZZ.of_type()`` 只接受 ``mpz``，而不接受 ``int``，因此需要额外的条件确保接受 ``int``
    return isinstance(c, int) or ZZ.of_type(c) or QQ.of_type(c)


# 判断一个参数是否为合适的整数类型
def is_int(c):
    r"""
    Test whether an argument is of an acceptable type to be used as an integer.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int`` or :ref:`ZZ`.

    See Also
    ========

    is_rat

    """
    # 如果安装了 gmpy2，则 ``ZZ.of_type()`` 只接受 ``mpz``，而不接受 ``int``，因此需要额外的条件确保接受 ``int``
    return isinstance(c, int) or ZZ.of_type(c)


# 获取一个有理数参数的分子和分母
def get_num_denom(c):
    r"""
    Given any argument on which :py:func:`~.is_rat` is ``True``, return the
    numerator and denominator of this number.

    See Also
    ========

    is_rat

    """
    # 将参数 c 转换为有理数对象 r
    r = QQ(c)
    # 返回有理数 r 的分子和分母
    return r.numerator, r.denominator


# 提取整数参数的基本判别元
@public
def extract_fundamental_discriminant(a):
    r"""
    Extract a fundamental discriminant from an integer *a*.

    Explanation
    ===========

    Given any rational integer *a* that is 0 or 1 mod 4, write $a = d f^2$,
    where $d$ is either 1 or a fundamental discriminant, and return a pair
    of dictionaries ``(D, F)`` giving the prime factorizations of $d$ and $f$
    respectively, in the same format returned by :py:func:`~.factorint`.

    A fundamental discriminant $d$ is different from unity, and is either
    1 mod 4 and squarefree, or is 0 mod 4 and such that $d/4$ is squarefree
    and 2 or 3 mod 4. This is the same as being the discriminant of some
    quadratic field.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import extract_fundamental_discriminant
    >>> print(extract_fundamental_discriminant(-432))
    ({3: 1, -1: 1}, {2: 2, 3: 1})

    For comparison:

    >>> from sympy import factorint
    >>> print(factorint(-432))
    {2: 4, 3: 3, -1: 1}

    Parameters
    ==========

    a: int, must be 0 or 1 mod 4

    Returns
    =======

    """
    # 这里应该有函数体，但在提供的代码中缺失了
    pass
    Pair ``(D, F)``  of dictionaries.

    Raises
    ======

    ValueError
        If *a* is not 0 or 1 mod 4.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Prop. 5.1.3)

    """
    # 检查 a 是否为 0 或 1 模 4
    if a % 4 not in [0, 1]:
        raise ValueError('To extract fundamental discriminant, number must be 0 or 1 mod 4.')

    # 如果 a 等于 0，返回空的 D 和 {0: 1} 的 F
    if a == 0:
        return {}, {0: 1}

    # 如果 a 等于 1，返回空的 D 和 F
    if a == 1:
        return {}, {}

    # 计算 a 的因子分解
    a_factors = factorint(a)
    D = {}  # 初始化 D 字典
    F = {}  # 初始化 F 字典

    # 第一遍循环：确保 d 是无平方因子的，a/d 是一个完全平方数。
    # 同时计算 d 中的模 4 余 3 的素数及其个数。
    num_3_mod_4 = 0
    for p, e in a_factors.items():
        if e % 2 == 1:
            D[p] = 1
            if p % 4 == 3:
                num_3_mod_4 += 1
            if e >= 3:
                F[p] = (e - 1) // 2
        else:
            F[p] = e // 2

    # 第二遍循环：如果 d 余 2 或 3 模 4，那么我们需要从 f**2 中偷一个因子 4 给 d。
    even = 2 in D
    if even or num_3_mod_4 % 2 == 1:
        e2 = F[2]
        assert e2 > 0
        if e2 == 1:
            del F[2]
        else:
            F[2] = e2 - 1
        D[2] = 3 if even else 2

    # 返回计算结果的 D 和 F 字典
    return D, F
@public
class AlgIntPowers:
    r"""
    Compute the powers of an algebraic integer.

    Explanation
    ===========

    Given an algebraic integer $\theta$ by its monic irreducible polynomial
    ``T`` over :ref:`ZZ`, this class computes representations of arbitrarily
    high powers of $\theta$, as :ref:`ZZ`-linear combinations over
    $\{1, \theta, \ldots, \theta^{n-1}\}$, where $n = \deg(T)$.

    The representations are computed using the linear recurrence relations for
    powers of $\theta$, derived from the polynomial ``T``. See [1], Sec. 4.2.2.

    Optionally, the representations may be reduced with respect to a modulus.

    Examples
    ========

    >>> from sympy import Poly, cyclotomic_poly
    >>> from sympy.polys.numberfields.utilities import AlgIntPowers
    >>> T = Poly(cyclotomic_poly(5))
    >>> zeta_pow = AlgIntPowers(T)
    >>> print(zeta_pow[0])
    [1, 0, 0, 0]
    >>> print(zeta_pow[1])
    [0, 1, 0, 0]
    >>> print(zeta_pow[4])  # doctest: +SKIP
    [-1, -1, -1, -1]
    >>> print(zeta_pow[24])  # doctest: +SKIP
    [-1, -1, -1, -1]

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """

    def __init__(self, T, modulus=None):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`
            The monic irreducible polynomial over :ref:`ZZ` defining the
            algebraic integer.

        modulus : int, None, optional
            If not ``None``, all representations will be reduced w.r.t. this.

        """
        self.T = T  # 存储代表代数整数的单项不可约多项式 T
        self.modulus = modulus  # 存储用于约化的模数，如果为 None 则不约化
        self.n = T.degree()  # 计算多项式 T 的次数 n
        self.powers_n_and_up = [[-c % self for c in reversed(T.rep.to_list())][:-1]]  # 初始化代数整数的高次幂的表示
        self.max_so_far = self.n  # 记录已经计算的最高次幂

    def red(self, exp):
        """
        Helper function to reduce an exponent modulo self.modulus.

        Parameters
        ==========

        exp : int
            Exponent to be reduced.

        Returns
        =======

        int
            Reduced exponent if self.modulus is not None, otherwise exp itself.

        """
        return exp if self.modulus is None else exp % self.modulus

    def __rmod__(self, other):
        """
        Redefines the modulo operation for instances of AlgIntPowers.

        Parameters
        ==========

        other : int
            Operand to be reduced modulo self.modulus.

        Returns
        =======

        int
            Reduced value of other modulo self.modulus.

        """
        return self.red(other)

    def compute_up_through(self, e):
        """
        Compute representations for exponents up to e using recurrence relations.

        Parameters
        ==========

        e : int
            Target exponent to compute representations up to.

        """
        m = self.max_so_far
        if e <= m: return
        n = self.n
        r = self.powers_n_and_up
        c = r[0]
        for k in range(m+1, e+1):
            b = r[k-1-n][n-1]
            r.append(
                [c[0]*b % self] + [
                    (r[k-1-n][i-1] + c[i]*b) % self for i in range(1, n)
                ]
            )
        self.max_so_far = e

    def get(self, e):
        """
        Retrieve the representation of theta^e as a linear combination.

        Parameters
        ==========

        e : int
            Exponent for which the representation is desired.

        Returns
        =======

        list
            List representing theta^e as a linear combination of {1, theta, ..., theta^(n-1)}.

        Raises
        ======

        ValueError
            If e is negative.

        """
        n = self.n
        if e < 0:
            raise ValueError('Exponent must be non-negative.')
        elif e < n:
            return [1 if i == e else 0 for i in range(n)]
        else:
            self.compute_up_through(e)
            return self.powers_n_and_up[e - n]

    def __getitem__(self, item):
        """
        Override method to retrieve the representation of theta^item.

        Parameters
        ==========

        item : int
            Exponent for which the representation is desired.

        Returns
        =======

        list
            List representing theta^item as a linear combination.

        """
        return self.get(item)


@public
def coeff_search(m, R):
    r"""
    Generate coefficients for searching through polynomials.

    Explanation
    ===========

    Lead coeff is always non-negative. Explore all combinations with coeffs
    bounded in absolute value before increasing the bound. Skip the all-zero
    """
    # 将初始的最大绝对值系数保存在 R0 中，以便后续比较使用
    R0 = R
    # 创建一个长度为 m 的列表 c，初始值为 R，用于存储系数
    c = [R] * m
    # 无限循环，生成系数列表的生成器
    while True:
        # 如果 R 等于初始的 R0，或者 R 在列表 c 中，或者 -R 在列表 c 中，则生成当前系数列表的副本并返回
        if R == R0 or R in c or -R in c:
            yield c[:]
        # 找到 c 中最右边不为 -R 的元素的索引 j
        j = m - 1
        while c[j] == -R:
            j -= 1
        # 将 c[j] 减 1
        c[j] -= 1
        # 将索引 j 后面的所有元素设置为 R
        for i in range(j + 1, m):
            c[i] = R
        # 找到第一个不为 0 的元素的索引 j
        for j in range(m):
            if c[j] != 0:
                break
        else:
            # 如果所有元素都为 0，则将 R 值增加 1，并重新初始化 c 为长度为 m，值为 R 的列表
            R += 1
            c = [R] * m
def supplement_a_subspace(M):
    r"""
    Extend a basis for a subspace to a basis for the whole space.

    Explanation
    ===========

    Given an $n \times r$ matrix *M* of rank $r$ (so $r \leq n$), this function
    computes an invertible $n \times n$ matrix $B$ such that the first $r$
    columns of $B$ equal *M*.

    This operation can be interpreted as a way of extending a basis for a
    subspace, to give a basis for the whole space.

    To be precise, suppose you have an $n$-dimensional vector space $V$, with
    basis $\{v_1, v_2, \ldots, v_n\}$, and an $r$-dimensional subspace $W$ of
    $V$, spanned by a basis $\{w_1, w_2, \ldots, w_r\}$, where the $w_j$ are
    given as linear combinations of the $v_i$. If the columns of *M* represent
    the $w_j$ as such linear combinations, then the columns of the matrix $B$
    computed by this function give a new basis $\{u_1, u_2, \ldots, u_n\}$ for
    $V$, again relative to the $\{v_i\}$ basis, and such that $u_j = w_j$
    for $1 \leq j \leq r$.

    Examples
    ========

    Note: The function works in terms of columns, so in these examples we
    print matrix transposes in order to make the columns easier to inspect.

    >>> from sympy.polys.matrices import DM
    >>> from sympy import QQ, FF
    >>> from sympy.polys.numberfields.utilities import supplement_a_subspace
    >>> M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    >>> print(supplement_a_subspace(M).to_Matrix().transpose())
    Matrix([[1, 7, 0], [2, 3, 4], [1, 0, 0]])

    >>> M2 = M.convert_to(FF(7))
    >>> print(M2.to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3]])
    >>> print(supplement_a_subspace(M2).to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3], [0, 1, 0]])

    Parameters
    ==========

    M : :py:class:`~.DomainMatrix`
        The columns give the basis for the subspace.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        This matrix is invertible and its first $r$ columns equal *M*.

    Raises
    ======

    DMRankError
        If *M* was not of maximal rank.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*
       (See Sec. 2.3.2.)

    """
    n, r = M.shape
    # Let In be the n x n identity matrix.
    # Form the augmented matrix [M | In] and compute RREF.
    Maug = M.hstack(M.eye(n, M.domain))  # 形成增广矩阵 [M | In]
    R, pivots = Maug.rref()  # 计算增广矩阵的行简化阶梯形式（RREF）
    if pivots[:r] != tuple(range(r)):
        raise DMRankError('M was not of maximal rank')  # 如果 M 不是最大秩则抛出异常
    # Let J be the n x r matrix equal to the first r columns of In.
    # Since M is of rank r, RREF reduces [M | In] to [J | A], where A is the product of
    # elementary matrices Ei corresp. to the row ops performed by RREF. Since the Ei are
    # invertible, so is A. Let B = A^(-1).
    A = R[:, r:]  # 提取出增广矩阵 RREF 后的右侧矩阵 A
    B = A.inv()  # 计算 A 的逆矩阵 B
    # Then B is the desired matrix. It is invertible, since B^(-1) == A.
    # And A * [M | In] == [J | A]
    #  => A * M == J
    #  => M == B * J == the first r columns of B.
    # 返回变量 B 的值作为函数的返回结果
    return B
# 定义一个公共函数，用于找到实数代数数的有理隔离区间
@public
def isolate(alg, eps=None, fast=False):
    """
    Find a rational isolating interval for a real algebraic number.

    Examples
    ========

    >>> from sympy import isolate, sqrt, Rational
    >>> print(isolate(sqrt(2)))  # doctest: +SKIP
    (1, 2)
    >>> print(isolate(sqrt(2), eps=Rational(1, 100)))
    (24/17, 17/12)

    Parameters
    ==========

    alg : str, int, :py:class:`~.Expr`
        The algebraic number to be isolated. Must be a real number, to use this
        particular function. However, see also :py:meth:`.Poly.intervals`,
        which isolates complex roots when you pass ``all=True``.
    eps : positive element of :ref:`QQ`, None, optional (default=None)
        Precision to be passed to :py:meth:`.Poly.refine_root`
    fast : boolean, optional (default=False)
        Say whether fast refinement procedure should be used.
        (Will be passed to :py:meth:`.Poly.refine_root`.)

    Returns
    =======

    Pair of rational numbers defining an isolating interval for the given
    algebraic number.

    See Also
    ========

    .Poly.intervals

    """
    # 将输入的代数数转换为 sympy 的表达式
    alg = sympify(alg)

    # 如果代数数已经是有理数，则直接返回它自己构成的区间
    if alg.is_Rational:
        return (alg, alg)
    # 如果代数数不是实数，则抛出 NotImplementedError
    elif not alg.is_real:
        raise NotImplementedError(
            "complex algebraic numbers are not supported")

    # 使用 mpmath 中的 lambdify 函数创建一个函数 func，用于计算代数数的值
    func = lambdify((), alg, modules="mpmath", printer=IntervalPrinter())

    # 使用 minpoly 函数计算代数数 alg 的最小多项式，并调用其 intervals 方法得到区间列表
    poly = minpoly(alg, polys=True)
    intervals = poly.intervals(sqf=True)

    # 保存当前 mpmath 的精度设置和完成状态
    dps, done = mp.dps, False

    try:
        # 不断尝试增加精度，直到找到包含 alg 值的区间
        while not done:
            alg = func()

            for a, b in intervals:
                if a <= alg.a and alg.b <= b:
                    done = True
                    break
            else:
                mp.dps *= 2
    finally:
        # 恢复之前保存的 mpmath 精度设置
        mp.dps = dps

    # 如果提供了 eps 参数，则使用 poly 的 refine_root 方法进一步细化根的区间
    if eps is not None:
        a, b = poly.refine_root(a, b, eps=eps, fast=fast)

    # 返回找到的有理数区间的对
    return (a, b)
```