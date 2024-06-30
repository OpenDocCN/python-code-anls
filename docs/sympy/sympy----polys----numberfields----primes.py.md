# `D:\src\scipysrc\sympy\sympy\polys\numberfields\primes.py`

```
"""Prime ideals in number fields. """

# 导入所需的模块和类
from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace

# 定义一个函数，用于检查给定子模块是否符合最大秩条件
def _check_formal_conditions_for_maximal_order(submodule):
    r"""
    Several functions in this module accept an argument which is to be a
    :py:class:`~.Submodule` representing the maximal order in a number field,
    such as returned by the :py:func:`~sympy.polys.numberfields.basis.round_two`
    algorithm.

    We do not attempt to check that the given ``Submodule`` actually represents
    a maximal order, but we do check a basic set of formal conditions that the
    ``Submodule`` must satisfy, at a minimum. The purpose is to catch an
    obviously ill-formed argument.
    """
    prefix = 'The submodule representing the maximal order should '
    cond = None
    # 检查子模块是否为幂基子模块
    if not submodule.is_power_basis_submodule():
        cond = 'be a direct submodule of a power basis.'
    # 检查子模块的第一个生成元是否为单位元
    elif not submodule.starts_with_unity():
        cond = 'have 1 as its first generator.'
    # 检查子模块是否为方阵，且处于Hermite正规形式下的最大秩
    elif not submodule.is_sq_maxrank_HNF():
        cond = 'have square matrix, of maximal rank, in Hermite Normal Form.'
    # 如果有条件不满足，则引发结构错误异常
    if cond is not None:
        raise StructureError(prefix + cond)

# 定义一个类，表示数域中的素理想
class PrimeIdeal(IntegerPowerable):
    r"""
    A prime ideal in a ring of algebraic integers.
    """

    def __init__(self, ZK, p, alpha, f, e=None):
        """
        Parameters
        ==========

        ZK : :py:class:`~.Submodule`
            The maximal order where this ideal lives.
        p : int
            The rational prime this ideal divides.
        alpha : :py:class:`~.PowerBasisElement`
            Such that the ideal is equal to ``p*ZK + alpha*ZK``.
        f : int
            The inertia degree.
        e : int, ``None``, optional
            The ramification index, if already known. If ``None``, we will
            compute it here.

        """
        # 检查给定的最大秩子模块是否符合形式条件
        _check_formal_conditions_for_maximal_order(ZK)
        # 初始化素理想的属性
        self.ZK = ZK
        self.p = p
        self.alpha = alpha
        self.f = f
        self._test_factor = None
        # 如果未提供分歧指数e，则计算其值
        self.e = e if e is not None else self.valuation(p * ZK)

    def __str__(self):
        # 返回素理想的字符串表示形式
        if self.is_inert:
            return f'({self.p})'
        return f'({self.p}, {self.alpha.as_expr()})'

    @property
    def is_inert(self):
        """
        Say whether the rational prime we divide is inert, i.e. stays prime in
        our ring of integers.
        """
        # 判断该素理想对应的有理素数是否为惰性素理想
        return self.f == self.ZK.n
    def repr(self, field_gen=None, just_gens=False):
        """
        Print a representation of this prime ideal.

        Examples
        ========

        >>> from sympy import cyclotomic_poly, QQ
        >>> from sympy.abc import x, zeta
        >>> T = cyclotomic_poly(7, x)
        >>> K = QQ.algebraic_field((T, zeta))
        >>> P = K.primes_above(11)
        >>> print(P[0].repr())
        [ (11, x**3 + 5*x**2 + 4*x - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta))
        [ (11, zeta**3 + 5*zeta**2 + 4*zeta - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta, just_gens=True))
        (11, zeta**3 + 5*zeta**2 + 4*zeta - 1)

        Parameters
        ==========

        field_gen : :py:class:`~.Symbol`, ``None``, optional (default=None)
            The symbol to use for the generator of the field. This will appear
            in our representation of ``self.alpha``. If ``None``, we use the
            variable of the defining polynomial of ``self.ZK``.
        just_gens : bool, optional (default=False)
            If ``True``, just print the "(p, alpha)" part, showing "just the
            generators" of the prime ideal. Otherwise, print a string of the
            form "[ (p, alpha) e=..., f=... ]", giving the ramification index
            and inertia degree, along with the generators.

        """
        # 如果未提供 field_gen，则使用 self.ZK.parent.T.gen 作为域生成元
        field_gen = field_gen or self.ZK.parent.T.gen
        # 从对象的属性中获取 p, alpha, e, f 四个值
        p, alpha, e, f = self.p, self.alpha, self.e, self.f
        # 将 alpha 表示为字符串形式，分子部分使用 field_gen 作为变量
        alpha_rep = str(alpha.numerator(x=field_gen).as_expr())
        # 如果 alpha 的分母大于 1，则将 alpha_rep 表示为带分数形式
        if alpha.denom > 1:
            alpha_rep = f'({alpha_rep})/{alpha.denom}'
        # 格式化生成器部分的字符串，形如 "(p, alpha_rep)"
        gens = f'({p}, {alpha_rep})'
        # 如果 just_gens 为 True，则返回生成器部分的字符串
        if just_gens:
            return gens
        # 否则返回完整的表示形式字符串，包括 ramification index 和 inertia degree
        return f'[ {gens} e={e}, f={f} ]'

    def __repr__(self):
        # 直接调用 repr 方法的结果作为对象的字符串表示形式
        return self.repr()
    def as_submodule(self):
        r"""
        Represent this prime ideal as a :py:class:`~.Submodule`.

        Explanation
        ===========

        The :py:class:`~.PrimeIdeal` class serves to bundle information about
        a prime ideal, such as its inertia degree, ramification index, and
        two-generator representation, as well as to offer helpful methods like
        :py:meth:`~.PrimeIdeal.valuation` and
        :py:meth:`~.PrimeIdeal.test_factor`.

        However, in order to be added and multiplied by other ideals or
        rational numbers, it must first be converted into a
        :py:class:`~.Submodule`, which is a class that supports these
        operations.

        In many cases, the user need not perform this conversion deliberately,
        since it is automatically performed by the arithmetic operator methods
        :py:meth:`~.PrimeIdeal.__add__` and :py:meth:`~.PrimeIdeal.__mul__`.

        Raising a :py:class:`~.PrimeIdeal` to a non-negative integer power is
        also supported.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, prime_decomp
        >>> T = Poly(cyclotomic_poly(7))
        >>> P0 = prime_decomp(7, T)[0]
        >>> print(P0**6 == 7*P0.ZK)
        True

        Note that, on both sides of the equation above, we had a
        :py:class:`~.Submodule`. In the next equation we recall that adding
        ideals yields their GCD. This time, we need a deliberate conversion
        to :py:class:`~.Submodule` on the right:

        >>> print(P0 + 7*P0.ZK == P0.as_submodule())
        True

        Returns
        =======

        :py:class:`~.Submodule`
            Will be equal to ``self.p * self.ZK + self.alpha * self.ZK``.

        See Also
        ========

        __add__
        __mul__

        """
        # Calculate the submodule corresponding to this prime ideal
        M = self.p * self.ZK + self.alpha * self.ZK
        # Pre-set expensive boolean properties whose value we already know:
        M._starts_with_unity = False
        M._is_sq_maxrank_HNF = True
        # Return the submodule representation
        return M

    def __eq__(self, other):
        # Check equality with another PrimeIdeal by comparing their submodule representations
        if isinstance(other, PrimeIdeal):
            return self.as_submodule() == other.as_submodule()
        return NotImplemented

    def __add__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and add to another
        :py:class:`~.Submodule`.

        See Also
        ========

        as_submodule

        """
        # Convert both operands to submodules and perform addition
        return self.as_submodule() + other

    __radd__ = __add__  # Right addition is equivalent to left addition

    def __mul__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and multiply by another
        :py:class:`~.Submodule` or a rational number.

        See Also
        ========

        as_submodule

        """
        # Convert both operands to submodules and perform multiplication
        return self.as_submodule() * other

    __rmul__ = __mul__  # Right multiplication is equivalent to left multiplication

    def _zeroth_power(self):
        # Return the zeroth power of the prime ideal, which is itself (identity in multiplication)
        return self.ZK

    def _first_power(self):
        # Return the first power of the prime ideal, which is itself (identity in multiplication)
        return self
    def test_factor(self):
        r"""
        Compute a test factor for this prime ideal.

        Explanation
        ===========

        Write $\mathfrak{p}$ for this prime ideal, $p$ for the rational prime
        it divides. Then, for computing $\mathfrak{p}$-adic valuations it is
        useful to have a number $\beta \in \mathbb{Z}_K$ such that
        $p/\mathfrak{p} = p \mathbb{Z}_K + \beta \mathbb{Z}_K$.

        Essentially, this is the same as the number $\Psi$ (or the "reagent")
        from Kummer's 1847 paper (*Ueber die Zerlegung...*, Crelle vol. 35) in
        which ideal divisors were invented.
        """
        # 如果尚未计算测试因子，则调用_compute_test_factor计算并存储结果
        if self._test_factor is None:
            self._test_factor = _compute_test_factor(self.p, [self.alpha], self.ZK)
        return self._test_factor

    def valuation(self, I):
        r"""
        Compute the $\mathfrak{p}$-adic valuation of integral ideal I at this
        prime ideal.

        Parameters
        ==========

        I : :py:class:`~.Submodule`

        See Also
        ========

        prime_valuation

        """
        # 调用prime_valuation函数计算给定整数理想I在该素理想处的$\mathfrak{p}$-adic赋值
        return prime_valuation(I, self)

    def reduce_element(self, elt):
        """
        Reduce a :py:class:`~.PowerBasisElement` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.PowerBasisElement`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.PowerBasisElement`
            The reduced element.

        See Also
        ========

        reduce_ANP
        reduce_alg_num
        .Submodule.reduce_element

        """
        # 将给定的PowerBasisElement元素elt按照该素理想模约为其"小代表"
        return self.as_submodule().reduce_element(elt)

    def reduce_ANP(self, a):
        """
        Reduce an :py:class:`~.ANP` to a "small representative" modulo this
        prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.ANP`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.ANP`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_alg_num
        .Submodule.reduce_element

        """
        # 从给定的ANP元素a创建相应的元素elt，然后按照该素理想模约为其"小代表"
        elt = self.ZK.parent.element_from_ANP(a)
        red = self.reduce_element(elt)
        return red.to_ANP()

    def reduce_alg_num(self, a):
        """
        Reduce an :py:class:`~.AlgebraicNumber` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.AlgebraicNumber`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_ANP
        .Submodule.reduce_element

        """
        # 从给定的AlgebraicNumber元素a创建相应的元素elt，然后按照该素理想模约为其"小代表"
        elt = self.ZK.parent.element_from_alg_num(a)
        red = self.reduce_element(elt)
        return a.field_element(list(reversed(red.QQ_col.flat())))
    # 检查最大秩的形式条件是否满足
    _check_formal_conditions_for_maximal_order(ZK)
    # 获取模 ZK 的自同态环
    E = ZK.endomorphism_ring()
    # 生成表示每个生成元 g 在模 p 下的内自同态的矩阵列表
    matrices = [E.inner_endomorphism(g).matrix(modulus=p) for g in gens]
    # 创建一个域矩阵 B，将所有生成元 g 的乘法表示堆叠起来
    B = DomainMatrix.zeros((0, ZK.n), FF(p)).vstack(*matrices)
    # B 的零空间中的非零元素将表示对 omegas 的线性组合，满足：
    # (i) 不是 p 的倍数（因为在 FF(p) 上非零），且
    # (ii) 与每个生成元 g 的乘积是 p 的倍数（因为 B 表示这些生成元的乘法）
    # 根据理论预测，这样的元素必定存在，因此零空间应该是非平凡的
    x = B.nullspace()[0, :].transpose()
    # 计算 beta，表示为 ZK 的一个元素，由 ZK 的矩阵乘以 x 而来，除以 ZK 的分母
    beta = ZK.parent(ZK.matrix * x.convert_to(ZZ), denom=ZK.denom)
    # 返回计算得到的 beta
    return beta


@public
def prime_valuation(I, P):
    r"""
    计算整数理想 I 的 *P*-adic 估值。

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.polys.numberfields import prime_valuation
    >>> K = QQ.cyclotomic_field(5)
    >>> P = K.primes_above(5)
    >>> ZK = K.maximal_order()
    >>> print(prime_valuation(25*ZK, P[0]))
    8

    Parameters
    ==========

    I : :py:class:`~.Submodule`
        所需估值的整数理想。

    P : :py:class:`~.PrimeIdeal`
        要计算估值的素数理想。

    Returns
    =======

    int

    See Also
    ========

    .PrimeIdeal.valuation

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 4.8.17.)

    """
    # 获取素数 p 和它所在的 ZK
    p, ZK = P.p, P.ZK
    # 获取 ZK 的维数、矩阵和分母
    n, W, d = ZK.n, ZK.matrix, ZK.denom

    # 计算 A = W^(-1) * I * d / I.denom，转换成 QQ 后再转换回 ZZ
    A = W.convert_to(QQ).inv() * I.matrix * d / I.denom
    A = A.convert_to(ZZ)
    # 计算 A 的行列式 D
    D = A.det()
    # 如果 D 对 p 取模不为 0，则返回估值为 0
    if D % p != 0:
        return 0

    # 计算 P 的测试因子 beta
    beta = P.test_factor()

    # 计算 f = d^n // W.det()，并检查是否需要进行完整测试
    f = d ** n // W.det()
    need_complete_test = (f % p == 0)
    # 初始化估值 v 为 0
    v = 0
    # 进入循环，A的列表示omegas的线性组合。
    # 将它们转换为thetas的线性组合：
    A = W * A
    # 逐列处理...
    for j in range(n):
        # 将A的第j列视为有理整数的元素，denom=d
        c = ZK.parent(A[:, j], denom=d)
        # 乘以beta后，转换回omegas的线性组合：
        c *= beta
        # 将c表示为omegas的线性组合，并展平成一维数组
        c = ZK.represent(c).flat()
        for i in range(n):
            # 更新A的元素为c[i]
            A[i, j] = c[i]
    # 如果A的最后一个元素不是p的倍数，则跳出循环
    if A[n - 1, n - 1].element % p != 0:
        break
    # 将A除以p
    A = A / p
    # 如上所述，即使除法能整除，domain也会转换为QQ。
    # 因此即使我们不需要完全测试，也必须转换回来。
    if need_complete_test:
        # 在这种情况下，非整数条目实际上是我们的停止条件。
        try:
            # 将A转换为整数
            A = A.convert_to(ZZ)
        except CoercionFailed:
            break
    else:
        # 在这种情况下，理论上不应该有任何非整数条目。
        A = A.convert_to(ZZ)
    # 增加计数器v
    v += 1
return v
def _two_elt_rep(gens, ZK, p, f=None, Np=None):
    r"""
    Given a set of *ZK*-generators of a prime ideal, compute a set of just two
    *ZK*-generators for the same ideal, one of which is *p* itself.

    Parameters
    ==========

    gens : list of :py:class:`PowerBasisElement`
        Generators for the prime ideal over *ZK*, the ring of integers of the
        field $K$.

    ZK : :py:class:`~.Submodule`
        The maximal order in $K$.

    p : int
        The rational prime divided by the prime ideal.

    f : int, optional
        The inertia degree of the prime ideal, if known.

    Np : int, optional
        The norm $p^f$ of the prime ideal, if known.
        NOTE: There is no reason to supply both *f* and *Np*. Either one will
        save us from having to compute the norm *Np* ourselves. If both are known,
        *Np* is preferred since it saves one exponentiation.

    Returns
    =======

    :py:class:`~.PowerBasisElement` representing a single algebraic integer
    alpha such that the prime ideal is equal to ``p*ZK + alpha*ZK``.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Algorithm 4.7.10.)

    """
    # Ensure that ZK is indeed a maximal order in its field
    _check_formal_conditions_for_maximal_order(ZK)
    
    # Get the parent of ZK, which is the `PowerBasis` object
    pb = ZK.parent
    
    # Get the underlying algebraic number field T associated with ZK
    T = pb.T
    
    # Detect special cases where all generators are multiples of p
    # or where there are no generators (all conditions in the `all` statement are true)
    if all((g % p).equiv(0) for g in gens):
        return pb.zero()
    
    # Calculate the norm Np if not provided explicitly
    if Np is None:
        if f is not None:
            Np = p**f
        else:
            # Compute the determinant of the submodule generated by gens
            Np = abs(pb.submodule_from_gens(gens).matrix.det())
    
    # Compute pullbacks of the basis elements of ZK
    omega = ZK.basis_element_pullbacks()
    
    # Construct beta, where each element is p times an element of omega except the first
    beta = [p*om for om in omega[1:]]  # note: we omit omega[0] == 1
    
    # Extend beta with the given generators gens
    beta += gens
    
    # Perform coefficient search for coefficients `c` of length 1
    search = coeff_search(len(beta), 1)
    
    # Iterate over possible coefficient combinations `c`
    for c in search:
        # Compute alpha as a linear combination of beta with coefficients `c`
        alpha = sum(ci*betai for ci, betai in zip(c, beta))
        
        # Compute the norm of alpha divided by Np
        n = alpha.norm(T) // Np
        
        # Check if n mod p is not zero, then reduce alpha mod p
        if n % p != 0:
            # Return alpha reduced modulo p
            return alpha % p


def _prime_decomp_easy_case(p, ZK):
    r"""
    Compute the decomposition of rational prime *p* in the ring of integers
    *ZK* (given as a :py:class:`~.Submodule`), in the "easy case", i.e. the
    case where *p* does not divide the index of $\theta$ in *ZK*, where
    $\theta$ is the generator of the ``PowerBasis`` of which *ZK* is a
    ``Submodule``.
    """
    # Get the underlying algebraic number field T associated with ZK
    T = ZK.parent.T
    
    # Create the polynomial ring T_bar over T modulo p
    T_bar = Poly(T, modulus=p)
    
    # Factorize T_bar into its linear factors
    lc, fl = T_bar.factor_list()
    
    # Check if there is exactly one irreducible factor of degree 1
    if len(fl) == 1 and fl[0][1] == 1:
        # Return a list containing the prime ideal decomposition
        return [PrimeIdeal(ZK, p, ZK.parent.zero(), ZK.n, 1)]
    
    # Return a list of PrimeIdeal objects for each factor t with exponent e
    return [PrimeIdeal(ZK, p,
                       ZK.parent.element_from_poly(Poly(t, domain=ZZ)),
                       t.degree(), e)
            for t, e in fl]
def _prime_decomp_compute_kernel(I, p, ZK):
    r"""
    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``ZK/pZK``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    Pair ``(N, G)``, where:

        ``N`` is a :py:class:`~.Module` representing the kernel of the map
        ``a |--> a**p - a`` on ``(O/pO)/I``, guaranteed to be a module with
        unity.

        ``G`` is a :py:class:`~.Module` representing a basis for the separable
        algebra ``A = O/I`` (see Cohen).

    """
    W = I.matrix
    n, r = W.shape
    # Want to take the Fp-basis given by the columns of I, adjoin (1, 0, ..., 0)
    # (which we know is not already in there since I is a basis for a prime ideal)
    # and then supplement this with additional columns to make an invertible n x n
    # matrix. This will then represent a full basis for ZK, whose first r columns
    # are pullbacks of the basis for I.
    if r == 0:
        B = W.eye(n, ZZ)
    else:
        B = W.hstack(W.eye(n, ZZ)[:, 0])
    if B.shape[1] < n:
        B = supplement_a_subspace(B.convert_to(FF(p))).convert_to(ZZ)

    G = ZK.submodule_from_matrix(B)
    # Must compute G's multiplication table _before_ discarding the first r
    # columns. (See Step 9 in Alg 6.2.9 in Cohen, where the betas are actually
    # needed in order to represent each product of gammas. However, once we've
    # found the representations, then we can ignore the betas.)
    G.compute_mult_tab()
    G = G.discard_before(r)

    phi = ModuleEndomorphism(G, lambda x: x**p - x)
    N = phi.kernel(modulus=p)
    assert N.starts_with_unity()
    return N, G


def _prime_decomp_maximal_ideal(I, p, ZK):
    r"""
    We have reached the case where we have a maximal (hence prime) ideal *I*,
    which we know because the quotient ``O/I`` is a field.

    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``O/pO``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    :py:class:`~.PrimeIdeal` instance representing this prime

    """
    m, n = I.matrix.shape
    f = m - n
    G = ZK.matrix * I.matrix
    gens = [ZK.parent(G[:, j], denom=ZK.denom) for j in range(G.shape[1])]
    alpha = _two_elt_rep(gens, ZK, p, f=f)
    return PrimeIdeal(ZK, p, alpha, f)


def _prime_decomp_split_ideal(I, p, N, G, ZK):
    r"""
    Perform the step in the prime decomposition algorithm where we have determined
    the quotient ``ZK/I`` is _not_ a field, and we want to perform a non-trivial
    factorization of *I* by locating an idempotent element of ``ZK/I``.
    """
    assert I.parent == ZK and G.parent is ZK and N.parent is G
    # Since ZK/I is not a field, the kernel computed in the previous step contains
    # more than just the prime field Fp, and our basis N for the nullspace therefore
    # includes elements that are not simply in Fp.
    # 让 alpha 成为域 Fp 上的元素，使得 alpha 不属于 G 的子模块
    alpha = N(1).to_parent()
    # 确保 alpha 的模块是 G
    assert alpha.module is G

    # 初始化 alpha 的幂列表
    alpha_powers = []
    # 找到 alpha 的最小多项式 m，使用域 FF(p)，并将 alpha 的幂添加到 alpha_powers 列表中
    m = find_min_poly(alpha, FF(p), powers=alpha_powers)

    # 分解 m，得到其首项系数 lc 和因子列表 fl
    lc, fl = m.factor_list()
    # 取第一个非常数因子作为 m1
    m1 = fl[0][0]
    # 计算 m 除以 m1 得到的商作为 m2
    m2 = m.quo(m1)
    # 使用扩展欧几里得算法计算 m1 和 m2 的最大公因数，同时返回其系数的线性组合 U 和 V，以及最大公因数 g
    U, V, g = m1.gcdex(m2)

    # 理论上 m 应为无平方因子多项式，因此 m1 和 m2 应该互质
    assert g == 1

    # 构造 E 列表，其中每个元素是 U * m1 的反序系数列表的一部分
    E = list(reversed(Poly(U * m1, domain=ZZ).rep.to_list()))

    # 计算 eps1 和 eps2，它们是 alpha_powers 的线性组合，使得 eps1 + eps2 = 1
    eps1 = sum(E[i]*alpha_powers[i] for i in range(len(E)))
    eps2 = 1 - eps1

    # 构建 idemps 列表，其中包含 eps1 和 eps2
    idemps = [eps1, eps2]

    # 初始化 factors 列表，用于存储子模块 H
    factors = []
    for eps in idemps:
        # 将 eps 视为 ZK 的元素
        e = eps.to_parent()
        # 确保 e 的模块是 ZK
        assert e.module is ZK

        # 构造矩阵 D，其列为 e 与 ZK 的基元素 om 乘积的列向量
        D = I.matrix.convert_to(FF(p)).hstack(*[
            (e * om).column(domain=FF(p)) for om in ZK.basis_elements()
        ])

        # 计算 D 的列空间，并将结果转换为整数环 ZZ 上的矩阵
        W = D.columnspace().convert_to(ZZ)

        # 从 W 构建 ZK 的子模块 H
        H = ZK.submodule_from_matrix(W)

        # 将 H 添加到 factors 列表中
        factors.append(H)

    # 返回 factors 列表，其中包含 eps1 和 eps2 对应的 ZK 子模块
    return factors
@public
def prime_decomp(p, T=None, ZK=None, dK=None, radical=None):
    r"""
    Compute the decomposition of rational prime *p* in a number field.

    Explanation
    ===========

    Ordinarily this should be accessed through the
    :py:meth:`~.AlgebraicField.primes_above` method of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x, theta
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.algebraic_field((T, theta))
    >>> print(K.primes_above(2))
    [[ (2, x**2 + 1) e=1, f=1 ], [ (2, (x**2 + 3*x + 2)/2) e=1, f=1 ],
     [ (2, (3*x**2 + 3*x)/2) e=1, f=1 ]]

    Parameters
    ==========

    p : int
        The rational prime whose decomposition is desired.

    T : :py:class:`~.Poly`, optional
        Monic irreducible polynomial defining the number field $K$ in which to
        factor. NOTE: at least one of *T* or *ZK* must be provided.

    ZK : :py:class:`~.Submodule`, optional
        The maximal order for $K$, if already known.
        NOTE: at least one of *T* or *ZK* must be provided.

    dK : int, optional
        The discriminant of the field $K$, if already known.

    radical : :py:class:`~.Submodule`, optional
        The nilradical mod *p* in the integers of $K$, if already known.

    Returns
    =======

    List of :py:class:`~.PrimeIdeal` instances.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 6.2.9.)

    """
    # 检查输入参数，至少需要 T 或 ZK 中的一个
    if T is None and ZK is None:
        raise ValueError('At least one of T or ZK must be provided.')
    # 如果给定 ZK，则检查其是否为最大秩量
    if ZK is not None:
        _check_formal_conditions_for_maximal_order(ZK)
    # 如果 T 为空，则使用 ZK 的父域中的 T
    if T is None:
        T = ZK.parent.T
    # 初始化 radicals 字典
    radicals = {}
    # 如果 dK 或 ZK 为空，则通过 round_two 函数计算 ZK 和 dK
    if dK is None or ZK is None:
        ZK, dK = round_two(T, radicals=radicals)
    # 计算 T 的判别式
    dT = T.discriminant()
    # 计算 f_squared
    f_squared = dT // dK
    # 如果 f_squared 不能整除 p，则调用 _prime_decomp_easy_case 处理简单情况
    if f_squared % p != 0:
        return _prime_decomp_easy_case(p, ZK)
    # 如果 radical 未提供，则从 radicals 字典或通过 nilradical_mod_p 函数计算
    radical = radical or radicals.get(p) or nilradical_mod_p(ZK, p)
    # 初始化栈，添加 radical
    stack = [radical]
    # 初始化 primes 列表
    primes = []
    # 进行主循环，直到栈为空
    while stack:
        # 弹出栈顶元素作为当前处理的理想 I
        I = stack.pop()
        # 计算 N 和 G
        N, G = _prime_decomp_compute_kernel(I, p, ZK)
        # 如果 N 的阶数为 1，则调用 _prime_decomp_maximal_ideal 函数处理
        if N.n == 1:
            P = _prime_decomp_maximal_ideal(I, p, ZK)
            primes.append(P)
        else:
            # 否则，调用 _prime_decomp_split_ideal 函数分裂理想 I
            I1, I2 = _prime_decomp_split_ideal(I, p, N, G, ZK)
            stack.extend([I1, I2])
    # 返回分解得到的素理想列表 primes
    return primes
```