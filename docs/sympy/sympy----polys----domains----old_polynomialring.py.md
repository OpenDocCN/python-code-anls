# `D:\src\scipysrc\sympy\sympy\polys\domains\old_polynomialring.py`

```
"""Implementation of :class:`PolynomialRing` class. """

# 导入必要的模块和类
from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
        CoercionFailed, ExactQuotientFailed, NotReversible)
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable

# 定义一个公共的类，继承自 Ring 和 CompositeDomain
@public
class PolynomialRingBase(Ring, CompositeDomain):
    """
    Base class for generalized polynomial rings.

    This base class should be used for uniform access to generalized polynomial
    rings. Subclasses only supply information about the element storage etc.

    Do not instantiate.
    """

    # 设置类属性
    has_assoc_Ring = True
    has_assoc_Field = True

    default_order = "grevlex"

    # 初始化方法，接收一个域 dom 和生成器 gens，以及其他选项 opts
    def __init__(self, dom, *gens, **opts):
        # 如果没有指定生成器，则抛出异常
        if not gens:
            raise GeneratorsNeeded("generators not specified")

        # 计算生成器数量并设置为类属性
        lev = len(gens) - 1
        self.ngens = len(gens)

        # 使用 dtype 的 zero 和 one 方法创建零元素和单位元素
        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)

        # 设置域、生成器和符号（gens）
        self.domain = self.dom = dom
        self.symbols = self.gens = gens
        # 注意：如果通过 CompositeDomain 调用 inject 方法，可能未设置 'order'
        self.order = opts.get('order', monomial_key(self.default_order))

    # 设置域的方法，返回一个新的多项式环，带有指定的域
    def set_domain(self, dom):
        """Return a new polynomial ring with given domain. """
        return self.__class__(dom, *self.gens, order=self.order)

    # 创建新元素的方法
    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1)

    # 创建基本元素的方法
    def _ground_new(self, element):
        return self.one.ground_new(element)

    # 从字典创建元素的方法
    def _from_dict(self, element):
        return DMP.from_dict(element, len(self.gens) - 1, self.dom)

    # 返回类的字符串表示形式
    def __str__(self):
        s_order = str(self.order)
        orderstr = (
            " order=" + s_order) if s_order != self.default_order else ""
        return str(self.dom) + '[' + ','.join(map(str, self.gens)) + orderstr + ']'

    # 返回对象的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom,
                     self.gens, self.order))

    # 比较方法，判断两个多项式环是否相等
    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, PolynomialRingBase) and \
            self.dtype == other.dtype and self.dom == other.dom and \
            self.gens == other.gens and self.order == other.order

    # 从整数创建元素的方法
    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    # 从 Python 整数创建元素的方法（与上面的方法功能相同）
    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))
    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        # 使用 K1 对象的 dom 属性将 Python 的 Fraction 对象 a 转换到 dtype 类型，并用 _ground_new 方法创建新对象返回
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        # 使用 K1 对象的 dom 属性将 Python 的 Fraction 对象 a 转换到 dtype 类型，并用 _ground_new 方法创建新对象返回
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        # 使用 K1 对象的 dom 属性将 GMPY 的 mpz 对象 a 转换到 dtype 类型，并用 _ground_new 方法创建新对象返回
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        # 使用 K1 对象的 dom 属性将 GMPY 的 mpq 对象 a 转换到 dtype 类型，并用 _ground_new 方法创建新对象返回
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 使用 K1 对象的 dom 属性将 mpmath 的 mpf 对象 a 转换到 dtype 类型，并用 _ground_new 方法创建新对象返回
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert a ``ANP`` object to ``dtype``. """
        # 如果 K1 的 dom 等于 K0，则直接返回 K1 的新对象，否则返回 a
        if K1.dom == K0:
            return K1._ground_new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``PolyElement`` object to ``dtype``. """
        # 如果 K1 的生成元等于 K0 的符号，且 K1 的 dom 等于 K0 的 dom，则创建新的 K1 对象，否则进行转换并创建新对象返回
        if K1.gens == K0.symbols:
            if K1.dom == K0.dom:
                return K1(dict(a))  # 设置正确的环
            else:
                convert_dom = lambda c: K1.dom.convert_from(c, K0.dom)
                return K1._from_dict({m: convert_dom(c) for m, c in a.items()})
        else:
            # 重新排序字典，并根据需要进行类型转换后创建新对象返回
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.symbols, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1._from_dict(dict(zip(monoms, coeffs)))

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        # 如果 K1 的生成元等于 K0 的生成元，则将 a 转换到 K1 的类型并创建新对象返回，否则重新排序字典并根据需要进行类型转换后创建新对象返回
        if K1.gens == K0.gens:
            if K1.dom != K0.dom:
                a = a.convert(K1.dom)
            return K1(a.to_list())
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1(dict(zip(monoms, coeffs)))

    def get_field(self):
        """Returns a field associated with ``self``. """
        # 返回与 self 相关联的分数域对象
        return FractionField(self.dom, *self.gens)

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        # 抛出 NotImplementedError，不支持嵌套域
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        # 抛出 NotImplementedError，不支持嵌套域
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        # 尝试计算 self.one 与 a 的商，并返回结果
        try:
            return self.exquo(self.one, a)
        except (ExactQuotientFailed, ZeroDivisionError):
            # 如果出错，则抛出 NotReversible 异常
            raise NotReversible('%s is not a unit' % a)

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        # 返回 a 与 b 的扩展欧几里得算法结果
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        # 返回 a 与 b 的最大公约数
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        # 返回 a 与 b 的最小公倍数
        return a.lcm(b)
    def factorial(self, a):
        """Returns factorial of ``a``. """
        # 调用 dom 对象的 factorial 方法计算 a 的阶乘，返回结果
        return self.dtype(self.dom.factorial(a))

    def _vector_to_sdm(self, v, order):
        """
        For internal use by the modules class.

        Convert an iterable of elements of this ring into a sparse distributed
        module element.
        """
        # 抛出 NotImplementedError 异常，表示该方法未实现
        raise NotImplementedError

    def _sdm_to_dics(self, s, n):
        """Helper for _sdm_to_vector."""
        # 从 distributedmodules 模块导入 sdm_to_dict 函数
        from sympy.polys.distributedmodules import sdm_to_dict
        # 使用 sdm_to_dict 将 sdm s 转换成字典形式的 dic
        dic = sdm_to_dict(s)
        # 初始化一个长度为 n 的空列表 res
        res = [{} for _ in range(n)]
        # 遍历字典 dic 的键值对，将其分配到 res 中的子字典中
        for k, v in dic.items():
            res[k[0]][k[1:]] = v
        # 返回结果列表 res
        return res

    def _sdm_to_vector(self, s, n):
        """
        For internal use by the modules class.

        Convert a sparse distributed module into a list of length ``n``.

        Examples
        ========

        >>> from sympy import QQ, ilex
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> L = [((1, 1, 1), QQ(1)), ((0, 1, 0), QQ(1)), ((0, 0, 1), QQ(2))]
        >>> R._sdm_to_vector(L, 2)
        [DMF([[1], [2, 0]], [[1]], QQ), DMF([[1, 0], []], [[1]], QQ)]
        """
        # 将 s 转换成字典列表 dics
        dics = self._sdm_to_dics(s, n)
        # 返回由 dics 中的每个字典元素构成的 self(x) 列表
        # NOTE this works for global and local rings!
        return [self(x) for x in dics]

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        # 返回一个自由模，该模具有给定的 rank 和当前对象 self
        return FreeModulePolyRing(self, rank)
def _vector_to_sdm_helper(v, order):
    """Helper method for common code in Global and Local poly rings."""
    from sympy.polys.distributedmodules import sdm_from_dict
    d = {}
    for i, e in enumerate(v):
        for key, value in e.to_dict().items():
            d[(i,) + key] = value
    return sdm_from_dict(d, order)

@public
class GlobalPolynomialRing(PolynomialRingBase):
    """A true polynomial ring, with objects DMP. """

    is_PolynomialRing = is_Poly = True
    dtype = DMP

    def new(self, element):
        """Create a new element in the polynomial ring based on the type of input."""
        if isinstance(element, dict):
            return DMP.from_dict(element, len(self.gens) - 1, self.dom)
        elif element in self.dom:
            return self._ground_new(self.dom.convert(element))
        else:
            return self.dtype(element, self.dom, len(self.gens) - 1)

    def from_FractionField(K1, a, K0):
        """
        Convert a ``DMF`` object to ``DMP``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP, DMF
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.abc import x

        >>> f = DMF(([ZZ(1), ZZ(1)], [ZZ(1)]), ZZ)
        >>> K = ZZ.old_frac_field(x)

        >>> F = ZZ.old_poly_ring(x).from_FractionField(f, K)

        >>> F == DMP([ZZ(1), ZZ(1)], ZZ)
        True
        >>> type(F)  # doctest: +SKIP
        <class 'sympy.polys.polyclasses.DMP_Python'>
        """
        if a.denom().is_one:
            return K1.from_GlobalPolynomialRing(a.numer(), K0)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object using its dictionary representation."""
        return basic_from_dict(a.to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype`` using a dictionary representation."""
        try:
            rep, _ = dict_from_basic(a, gens=self.gens)
        except PolynomialError:
            raise CoercionFailed("Cannot convert %s to type %s" % (a, self))

        for k, v in rep.items():
            rep[k] = self.dom.from_sympy(v)

        return DMP.from_dict(rep, self.ngens - 1, self.dom)

    def is_positive(self, a):
        """Returns True if the leading coefficient of ``a`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if the leading coefficient of ``a`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if the leading coefficient of ``a`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if the leading coefficient of ``a`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def _vector_to_sdm(self, v, order):
        """
        Convert a vector of polynomials to a sparse distributed module (SDM).

        Examples
        ========

        >>> from sympy import lex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y)
        >>> f = R.convert(x + 2*y)
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], lex)
        [((1, 1, 1), 1), ((0, 1, 0), 1), ((0, 0, 1), 2)]
        """
        return _vector_to_sdm_helper(v, order)
class GeneralizedPolynomialRing(PolynomialRingBase):
    """A generalized polynomial ring, with objects DMF. """

    dtype = DMF  # 设置该类的数据类型为 DMF

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        # 从输入 ``a`` 构造出 ``self`` 域中的元素
        res = self.dtype(a, self.dom, len(self.gens) - 1)

        # 确保 res 确实属于我们的环
        if res.denom().terms(order=self.order)[0][0] != (0,)*len(self.gens):
            from sympy.printing.str import sstr
            raise CoercionFailed("denominator %s not allowed in %s"
                                 % (sstr(res), self))
        return res

    def __contains__(self, a):
        try:
            a = self.convert(a)
        except CoercionFailed:
            return False
        # 检查元素 ``a`` 的分母是否为 (0,)*len(self.gens)
        return a.denom().terms(order=self.order)[0][0] == (0,)*len(self.gens)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        # 将 ``a`` 转换为 SymPy 对象
        return (basic_from_dict(a.numer().to_sympy_dict(), *self.gens) /
                basic_from_dict(a.denom().to_sympy_dict(), *self.gens))

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        # 将 SymPy 的表达式转换为 ``dtype``
        p, q = a.as_numer_denom()

        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)

        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)

        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)

        return self((num, den)).cancel()

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``. """
        # 计算 ``a`` 和 ``b`` 的精确商
        r = a / b

        try:
            r = self.new((r.num, r.den))
        except CoercionFailed:
            raise ExactQuotientFailed(a, b, self)

        return r

    def from_FractionField(K1, a, K0):
        dmf = K1.get_field().from_FractionField(a, K0)
        return K1((dmf.num, dmf.den))

    def _vector_to_sdm(self, v, order):
        """
        Turn an iterable into a sparse distributed module.

        Note that the vector is multiplied by a unit first to make all entries
        polynomials.

        Examples
        ========

        >>> from sympy import ilex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> f = R.convert((x + 2*y) / (1 + x))
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], ilex)
        [((0, 0, 1), 2), ((0, 1, 0), 1), ((1, 1, 1), 1), ((1,
          2, 1), 1)]
        """
        # 注意这个实现方式相当低效...
        u = self.one.numer()
        for x in v:
            u *= x.denom()
        return _vector_to_sdm_helper([x.numer()*u/x.denom() for x in v], order)
    # 从参数中获取选项中的“order”，如果不存在则使用默认的多项式环顺序
    order = opts.get("order", GeneralizedPolynomialRing.default_order)
    # 如果“order”是可迭代的，则构建成多重顺序，并使用生成器列表作为参数
    if iterable(order):
        order = build_product_order(order, gens)
    # 将顺序转换为单调键（monomial_key），确保其为有效的单调顺序表示
    order = monomial_key(order)
    # 更新选项字典中的“order”值为处理后的顺序
    opts['order'] = order

    # 如果顺序是全局顺序，则返回全局多项式环对象
    if order.is_global:
        return GlobalPolynomialRing(dom, *gens, **opts)
    else:
        # 否则返回通用多项式环对象
        return GeneralizedPolynomialRing(dom, *gens, **opts)
```