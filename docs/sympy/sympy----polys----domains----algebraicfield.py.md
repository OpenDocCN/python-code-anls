# `D:\src\scipysrc\sympy\sympy\polys\domains\algebraicfield.py`

```
# 导入需要的模块和类
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public

# 定义一个公共的代数域类，继承自Field、CharacteristicZero和SimpleDomain
@public
class AlgebraicField(Field, CharacteristicZero, SimpleDomain):
    # 代数数域 QQ(a) 的描述
    r"""Algebraic number field :ref:`QQ(a)`

    A :ref:`QQ(a)` domain represents an `algebraic number field`_
    `\mathbb{Q}(a)` as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression involving `algebraic
    numbers`_ will treat the algebraic numbers as generators if the generators
    argument is not specified.

    >>> from sympy import Poly, Symbol, sqrt
    >>> x = Symbol('x')
    >>> Poly(x**2 + sqrt(2))
    Poly(x**2 + (sqrt(2)), x, sqrt(2), domain='ZZ')

    That is a multivariate polynomial with ``sqrt(2)`` treated as one of the
    generators (variables). If the generators are explicitly specified then
    ``sqrt(2)`` will be considered to be a coefficient but by default the
    :ref:`EX` domain is used. To make a :py:class:`~.Poly` with a :ref:`QQ(a)`
    domain the argument ``extension=True`` can be given.

    >>> Poly(x**2 + sqrt(2), x)
    Poly(x**2 + sqrt(2), x, domain='EX')
    >>> Poly(x**2 + sqrt(2), x, extension=True)
    Poly(x**2 + sqrt(2), x, domain='QQ<sqrt(2)>')

    A generator of the algebraic field extension can also be specified
    explicitly which is particularly useful if the coefficients are all
    rational but an extension field is needed (e.g. to factor the
    polynomial).

    >>> Poly(x**2 + 1)
    Poly(x**2 + 1, x, domain='ZZ')
    >>> Poly(x**2 + 1, extension=sqrt(2))
    Poly(x**2 + 1, x, domain='QQ<sqrt(2)>')

    It is possible to factorise a polynomial over a :ref:`QQ(a)` domain using
    the ``extension`` argument to :py:func:`~.factor` or by specifying the domain
    explicitly.

    >>> from sympy import factor, QQ
    >>> factor(x**2 - 2)
    x**2 - 2
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain='QQ<sqrt(2)>')
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain=QQ.algebraic_field(sqrt(2)))
    (x - sqrt(2))*(x + sqrt(2))

    The ``extension=True`` argument can be used but will only create an
    extension that contains the coefficients which is usually not enough to
    factorise the polynomial.

    >>> p = x**3 + sqrt(2)*x**2 - 2*x - 2*sqrt(2)
    >>> factor(p)                         # treats sqrt(2) as a symbol
    (x + sqrt(2))*(x**2 - 2)
    >>> factor(p, extension=True)
    (x - sqrt(2))*(x + sqrt(2))**2
    >>> factor(x**2 - 2, extension=True)  # all rational coefficients
    ```
    # 计算表达式 x**2 - 2
    x**2 - 2
    
    # 使用 :ref:`QQ(a)` 可以与 :py:func:`~.cancel` 和 :py:func:`~.gcd` 函数一起使用
    # 通过导入 cancel 和 gcd 函数来使用它们
    It is also possible to use :ref:`QQ(a)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.
    
    # 使用 cancel 函数对表达式 (x**2 - 2)/(x - sqrt(2)) 进行化简
    >>> from sympy import cancel, gcd
    >>> cancel((x**2 - 2)/(x - sqrt(2)))
    (x**2 - 2)/(x - sqrt(2))
    
    # 使用带有 extension 参数的 cancel 函数来对表达式进行更进一步的化简
    >>> cancel((x**2 - 2)/(x - sqrt(2)), extension=sqrt(2))
    x + sqrt(2)
    
    # 使用 gcd 函数计算 x**2 - 2 和 x - sqrt(2) 的最大公约数
    >>> gcd(x**2 - 2, x - sqrt(2))
    1
    
    # 使用带有 extension 参数的 gcd 函数来计算 x**2 - 2 和 x - sqrt(2) 的最大公约数
    >>> gcd(x**2 - 2, x - sqrt(2), extension=sqrt(2))
    x - sqrt(2)
    
    # 当直接使用域 :ref:`QQ(a)` 作为构造函数时，可以用来创建支持操作 ``+,-,*,**,/`` 的实例
    # 使用 :py:meth:`~.Domain.algebraic_field` 方法构造特定的 :ref:`QQ(a)` 域
    # 使用 :py:meth:`~.Domain.from_sympy` 方法从普通的 SymPy 表达式创建域元素
    >>> K = QQ.algebraic_field(sqrt(2))
    >>> K
    QQ<sqrt(2)>
    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> xk  # doctest: +SKIP
    ANP([4, 3], [1, 0, -2], QQ)
    
    # :ref:`QQ(a)` 的元素是 :py:class:`~.ANP` 的实例，具有有限的打印支持
    # 原始显示展示了元素的内部表示作为列表 ``[4, 3]``，表示在形式 ``a * sqrt(2) + b * 1`` 中的元素，
    # 其中 ``a`` 和 ``b`` 是 :ref:`QQ` 的元素。生成器的最小多项式 ``(x**2 - 2)`` 也显示为列表 ``[1, 0, -2]``
    # 可以使用 :py:meth:`~.Domain.to_sympy` 来获取更好的打印形式和查看操作结果
    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> yk = K.from_sympy(2 + 3*sqrt(2))
    >>> xk * yk  # doctest: +SKIP
    ANP([17, 30], [1, 0, -2], QQ)
    >>> K.to_sympy(xk * yk)
    17*sqrt(2) + 30
    >>> K.to_sympy(xk + yk)
    5 + 7*sqrt(2)
    >>> K.to_sympy(xk ** 2)
    24*sqrt(2) + 41
    >>> K.to_sympy(xk / yk)
    sqrt(2)/14 + 9/7
    
    # 任何表示代数数的表达式都可以用来生成 :ref:`QQ(a)` 域，前提是可以计算其最小多项式
    # 函数 :py:func:`~.minpoly` 用于此目的
    >>> from sympy import exp, I, pi, minpoly
    >>> g = exp(2*I*pi/3)
    >>> g
    exp(2*I*pi/3)
    >>> g.is_algebraic
    True
    >>> minpoly(g, x)
    x**2 + x + 1
    >>> factor(x**3 - 1, extension=g)
    (x - 1)*(x - exp(2*I*pi/3))*(x + 1 + exp(2*I*pi/3))
    
    # 也可以从多个扩展元素创建一个代数域
    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    >>> K
    QQ<sqrt(2) + sqrt(3)>
    >>> p = x**4 - 5*x**2 + 6
    >>> factor(p)
    (x**2 - 3)*(x**2 - 2)
    >>> factor(p, domain=K)
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))
    >>> factor(p, extension=[sqrt(2), sqrt(3)])
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))
    
    # 多个扩展元素总是组合在一起形成一个单一的
    # 导入 sympy 中的 primitive_element 函数
    >>> from sympy import primitive_element
    # 使用 primitive_element 函数计算给定扩展域的原始元和其最小多项式
    >>> primitive_element([sqrt(2), sqrt(3)], x)
    # 返回原始元的最小多项式和系数列表
    (x**4 - 10*x**2 + 1, [1, 1])
    # 使用 minpoly 函数计算给定原始元的最小多项式
    >>> minpoly(sqrt(2) + sqrt(3), x)
    # 返回给定原始元的最小多项式
    x**4 - 10*x**2 + 1

    # 可以通过域的 ext 和 orig_ext 属性访问扩展域的扩展元素
    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    # 输出扩展域的字符串表示形式
    >>> K
    # 打印扩展域的原始元
    QQ<sqrt(2) + sqrt(3)>
    # 访问扩展元素
    >>> K.ext
    # 返回扩展域的原始元
    sqrt(2) + sqrt(3)
    # 返回原始元的组成部分
    >>> K.orig_ext
    # 返回扩展域的原始元组
    (sqrt(2), sqrt(3))
    # 访问扩展元素的最小多项式
    >>> K.mod  # doctest: +SKIP
    # 返回扩展域的原始元的最小多项式，使用 DMP_Python 类型表示

    # 可以通过 discriminant 方法获取域的判别式，通过 integral_basis 方法获取整数基
    >>> zeta5 = exp(2*I*pi/5)
    # 使用 zeta5 创建一个域 K
    >>> K = QQ.algebraic_field(zeta5)
    # 打印域 K 的字符串表示形式
    >>> K
    # 返回域 K 的原始元
    QQ<exp(2*I*pi/5)>
    # 计算域 K 的判别式
    >>> K.discriminant()
    # 返回域 K 的判别式
    125
    # 使用 sqrt(5) 创建域 K
    >>> K = QQ.algebraic_field(sqrt(5))
    # 打印域 K 的字符串表示形式
    >>> K
    # 返回域 K 的原始元
    QQ<sqrt(5)>
    # 以 sympy 格式返回域 K 的整数基
    >>> K.integral_basis(fmt='sympy')
    # 返回域 K 的整数基的列表
    [1, 1/2 + sqrt(5)/2]
    # 返回域 K 的最大秩，作为 Submodule[[2, 0], [1, 1]]/2 类型
    >>> K.maximal_order()
    # 返回域 K 的最大秩，作为 Submodule[[2, 0], [1, 1]]/2 类型

    # 通过 primes_above 方法计算有理素数在域中的素理想分解
    >>> zeta7 = exp(2*I*pi/7)
    # 使用 zeta7 创建一个域 K
    >>> K = QQ.algebraic_field(zeta7)
    # 打印域 K 的字符串表示形式
    >>> K
    # 返回域 K 的原始元
    QQ<exp(2*I*pi/7)>
    # 计算有理素数 11 在域 K 中的素理想分解
    >>> K.primes_above(11)
    # 返回有理素数 11 在域 K 中的素理想分解列表

    # 可以通过 galois_group 方法计算域的 Galois 闭包的 Galois 群
    >>> K.galois_group(by_name=True)[0]
    # 返回域 K 的 Galois 闭包的 Galois 群的第一个元素 S6TransitiveSubgroups.C6

    # 注意事项部分提到目前只能在 QQ 域上生成代数扩展
    # 理想情况下，希望能够在其他域上生成扩展，例如 QQ(x)(sqrt(x**2 - 2))
    # 这等效于 QQ(x)[y]/(y**2 - x**2 + 2)，并且有两种这类商环/扩展的实现
    # 设置变量 `dtype` 为 `ANP`，表示代数数域的数据类型
    dtype = ANP
    
    # 标记该类为代数数域的实现，同时也是代数数域的数值类型
    is_AlgebraicField = is_Algebraic = True
    
    # 标记该类为数值类型
    is_Numerical = True
    
    # 标记该类没有关联的环
    has_assoc_Ring = False
    
    # 标记该类有关联的域
    has_assoc_Field = True
    def __init__(self, dom, *ext, alias=None):
        r"""
        Parameters
        ==========

        dom : :py:class:`~.Domain`
            定义基域，作为扩展域的基础域。
            目前只接受 :ref:`QQ`（有理数域）。

        *ext : One or more :py:class:`~.Expr`
            扩展域的生成元。这些应该是关于 `\mathbb{Q}`（有理数集合）的代数表达式。

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            如果提供，将用作 :py:class:`~.AlgebraicField` 的原始元素的别名符号。
            如果为 ``None``，并且 ``ext`` 正好包含一个 :py:class:`~.AlgebraicNumber`，则将使用其别名（如果有）。
        """
        if not dom.is_QQ:
            raise DomainError("ground domain must be a rational field")

        from sympy.polys.numberfields import to_number_field
        if len(ext) == 1 and isinstance(ext[0], tuple):
            orig_ext = ext[0][1:]
        else:
            orig_ext = ext

        if alias is None and len(ext) == 1:
            alias = getattr(ext[0], 'alias', None)

        self.orig_ext = orig_ext
        """
        扩展域的原始元素列表。

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.orig_ext
        (sqrt(2), sqrt(3))
        """

        self.ext = to_number_field(ext, alias=alias)
        """
        用于扩展域的原始元素。

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.ext
        sqrt(2) + sqrt(3)
        """

        self.mod = self.ext.minpoly.rep
        """
        扩展域原始元素的最小多项式。

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2))
        >>> K.mod
        DMP([1, 0, -2], QQ)
        """

        self.domain = self.dom = dom

        self.ngens = 1
        self.symbols = self.gens = (self.ext,)
        self.unit = self([dom(1), dom(0)])

        self.zero = self.dtype.zero(self.mod.to_list(), dom)
        self.one = self.dtype.one(self.mod.to_list(), dom)

        self._maximal_order = None
        self._discriminant = None
        self._nilradicals_mod_p = {}

    def new(self, element):
        return self.dtype(element, self.mod.to_list(), self.dom)

    def __str__(self):
        return str(self.dom) + '<' + str(self.ext) + '>'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.ext))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, AlgebraicField):
            return self.dtype == other.dtype and self.ext == other.ext
        else:
            return NotImplemented
    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`. """
        # 返回一个代数域，例如 `\mathbb{Q}(\alpha, \ldots)`，使用给定的扩展和别名（如果有）
        return AlgebraicField(self.dom, *((self.ext,) + extension), alias=alias)

    def to_alg_num(self, a):
        """Convert ``a`` of ``dtype`` to an :py:class:`~.AlgebraicNumber`. """
        # 将 ``a`` 转换为 :py:class:`~.AlgebraicNumber` 类型对象
        return self.ext.field_element(a)

    def to_sympy(self, a):
        """Convert ``a`` of ``dtype`` to a SymPy object. """
        # 预先计算一个转换器以便重复使用：
        if not hasattr(self, '_converter'):
            self._converter = _make_converter(self)

        # 使用预先计算的转换器将 ``a`` 转换为 SymPy 对象并返回
        return self._converter(a)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            # 尝试将 SymPy 表达式转换为 ``dtype`` 类型
            return self([self.dom.from_sympy(a)])
        except CoercionFailed:
            pass

        from sympy.polys.numberfields import to_number_field

        try:
            # 尝试将 SymPy 表达式转换为数字域对象，然后再转换为 ``dtype`` 类型并返回
            return self(to_number_field(a, self.ext).native_coeffs())
        except (NotAlgebraic, IsomorphismFailed):
            # 若转换失败，则抛出类型转换失败异常
            raise CoercionFailed(
                "%s is not a valid algebraic number in %s" % (a, self))

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        # 将 Python 的整数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        # 将 Python 的整数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        # 将 Python 的分数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        # 将 Python 的分数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        # 将 GMPY 的大整数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        # 将 GMPY 的有理数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 将 mpmath 的浮点数对象 ``a`` 转换为 ``dtype`` 类型
        return K1(K1.dom.convert(a, K0))

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        # 返回与当前对象关联的环（Ring）对象
        raise DomainError('there is no ring associated with %s' % self)

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        # 如果 ``a`` 是正数，则返回 True
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        # 如果 ``a`` 是负数，则返回 True
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        # 如果 ``a`` 是非正数，则返回 True
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        # 如果 ``a`` 是非负数，则返回 True
        return self.dom.is_nonnegative(a.LC())

    def numer(self, a):
        """Returns numerator of ``a``. """
        # 返回 ``a`` 的分子部分
        return a

    def denom(self, a):
        """Returns denominator of ``a``. """
        # 返回 ``a`` 的分母部分
        return self.one
    def from_AlgebraicField(K1, a, K0):
        """Convert AlgebraicField element 'a' to another AlgebraicField """
        # 使用目标域 K1 的方法将代数域元素 'a' 转换为 sympy 表示后再转换为 K1 的元素
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a GaussianInteger element 'a' to ``dtype``. """
        # 使用目标域 K1 的方法将高斯整数元素 'a' 转换为 sympy 表示后再转换为 K1 的元素
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a GaussianRational element 'a' to ``dtype``. """
        # 使用目标域 K1 的方法将高斯有理数元素 'a' 转换为 sympy 表示后再转换为 K1 的元素
        return K1.from_sympy(K0.to_sympy(a))

    def _do_round_two(self):
        from sympy.polys.numberfields.basis import round_two
        # 调用 sympy 中的 round_two 函数，计算给定域 self 的 ZK 和 dK
        ZK, dK = round_two(self, radicals=self._nilradicals_mod_p)
        # 将计算得到的 ZK（最大秩）设置为 self 的属性
        self._maximal_order = ZK
        # 将计算得到的 dK（判别式）设置为 self 的属性
        self._discriminant = dK

    def maximal_order(self):
        """
        Compute the maximal order, or ring of integers, of the field.

        Returns
        =======

        :py:class:`~sympy.polys.numberfields.modules.Submodule`.

        See Also
        ========

        integral_basis

        """
        if self._maximal_order is None:
            # 如果最大秩尚未计算，则调用 _do_round_two 方法计算
            self._do_round_two()
        return self._maximal_order

    def integral_basis(self, fmt=None):
        r"""
        Get an integral basis for the field.

        Parameters
        ==========

        fmt : str, None, optional (default=None)
            If ``None``, return a list of :py:class:`~.ANP` instances.
            If ``"sympy"``, convert each element of the list to an
            :py:class:`~.Expr`, using ``self.to_sympy()``.
            If ``"alg"``, convert each element of the list to an
            :py:class:`~.AlgebraicNumber`, using ``self.to_alg_num()``.

        Examples
        ========

        >>> from sympy import QQ, AlgebraicNumber, sqrt
        >>> alpha = AlgebraicNumber(sqrt(5), alias='alpha')
        >>> k = QQ.algebraic_field(alpha)
        >>> B0 = k.integral_basis()
        >>> B1 = k.integral_basis(fmt='sympy')
        >>> B2 = k.integral_basis(fmt='alg')
        >>> print(B0[1])  # doctest: +SKIP
        ANP([mpq(1,2), mpq(1,2)], [mpq(1,1), mpq(0,1), mpq(-5,1)], QQ)
        >>> print(B1[1])
        1/2 + alpha/2
        >>> print(B2[1])
        alpha/2 + 1/2

        In the last two cases we get legible expressions, which print somewhat
        differently because of the different types involved:

        >>> print(type(B1[1]))
        <class 'sympy.core.add.Add'>
        >>> print(type(B2[1]))
        <class 'sympy.core.numbers.AlgebraicNumber'>

        See Also
        ========

        to_sympy
        to_alg_num
        maximal_order
        """
        ZK = self.maximal_order()
        M = ZK.QQ_matrix
        n = M.shape[1]
        B = [self.new(list(reversed(M[:, j].flat()))) for j in range(n)]
        if fmt == 'sympy':
            # 如果 fmt 参数为 'sympy'，则将结果列表中的每个元素转换为 sympy.Expr 类型
            return [self.to_sympy(b) for b in B]
        elif fmt == 'alg':
            # 如果 fmt 参数为 'alg'，则将结果列表中的每个元素转换为 sympy.AlgebraicNumber 类型
            return [self.to_alg_num(b) for b in B]
        # 默认情况下返回基本列表 B
        return B
    # 获取该域的判别式。
    def discriminant(self):
        """Get the discriminant of the field."""
        # 如果尚未计算判别式，则执行第二轮计算
        if self._discriminant is None:
            self._do_round_two()
        # 返回计算得到的判别式
        return self._discriminant

    # 计算在给定有理素数 *p* 上的素理想。
    def primes_above(self, p):
        """Compute the prime ideals lying above a given rational prime *p*."""
        # 导入素数分解函数
        from sympy.polys.numberfields.primes import prime_decomp
        # 获取最大秩的环
        ZK = self.maximal_order()
        # 获取域的判别式
        dK = self.discriminant()
        # 获取模 p 下的零根理想
        rad = self._nilradicals_mod_p.get(p)
        # 调用素数分解函数，返回在 p 上的素理想
        return prime_decomp(p, ZK=ZK, dK=dK, radical=rad)

    # 计算该域的 Galois 闭包的 Galois 群。
    def galois_group(self, by_name=False, max_tries=30, randomize=False):
        """
        Compute the Galois group of the Galois closure of this field.

        Examples
        ========

        If the field is Galois, the order of the group will equal the degree
        of the field:

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> k = QQ.alg_field_from_poly(x**4 + 1)
        >>> G, _ = k.galois_group()
        >>> G.order()
        4

        If the field is not Galois, then its Galois closure is a proper
        extension, and the order of the Galois group will be greater than the
        degree of the field:

        >>> k = QQ.alg_field_from_poly(x**4 - 2)
        >>> G, _ = k.galois_group()
        >>> G.order()
        8

        See Also
        ========

        sympy.polys.numberfields.galoisgroups.galois_group

        """
        # 调用元素的最小多项式的 Galois 群计算函数
        return self.ext.minpoly_of_element().galois_group(
            by_name=by_name, max_tries=max_tries, randomize=randomize)
def _make_converter(K):
    """Construct the converter to convert back to Expr"""
    # 构建转换器，将 K 转换回 Expr 类型

    gen = K.ext.as_expr()
    # 将 K 的生成元转换为 SymPy 表达式

    todom = K.dom.from_sympy
    # 获取 K 的定义域，并使用 SymPy 转换函数

    powers = [S.One, gen]
    # 初始化幂次列表，包括常数和生成元

    for n in range(2, K.mod.degree()):
        powers.append((gen * powers[-1]).expand())
    # 计算生成元的各次幂，并扩展结果

    terms = [dict(t.as_coeff_Mul()[::-1] for t in Add.make_args(p)) for p in powers]
    # 生成包含有理系数和代数表达式的字典列表，用于将 ANP 系数映射到扩展的 SymPy 表达式

    algebraics = set().union(*terms)
    # 收集所有代数表达式并组成集合

    matrix = [[todom(t.get(a, S.Zero)) for t in terms] for a in algebraics]
    # 创建转换矩阵，将 ANP 系数映射为扩展的 SymPy 表达式

    def converter(a):
        """Convert a to Expr using converter"""
        # 使用转换器将 a 转换为 Expr 类型
        ai = a.to_list()[::-1]
        # 将 a 转换为列表并进行反转
        tosympy = K.dom.to_sympy
        # 获取 K 的定义域，并使用 SymPy 转换函数

        coeffs_dom = [sum(mij*aj for mij, aj in zip(mi, ai)) for mi in matrix]
        # 计算转换系数在定义域中的表达式

        coeffs_sympy = [tosympy(c) for c in coeffs_dom]
        # 将定义域中的系数转换为 SymPy 表达式

        res = Add(*(Mul(c, a) for c, a in zip(coeffs_sympy, algebraics)))
        # 使用 SymPy 表达式计算结果

        return res

    return converter
```