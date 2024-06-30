# `D:\src\scipysrc\sympy\sympy\polys\domains\fractionfield.py`

```
"""Implementation of :class:`FractionField` class. """

# 导入所需模块和类
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.field import Field
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

@public
# 定义分数域类，继承自Field和CompositeDomain类
class FractionField(Field, CompositeDomain):
    """A class for representing multivariate rational function fields. """

    # 类属性标记为分数域
    is_FractionField = is_Frac = True

    # 指示是否关联环和域
    has_assoc_Ring = True
    has_assoc_Field = True

    # 初始化方法，接受一个域或者分式域对象，符号列表以及排序方式
    def __init__(self, domain_or_field, symbols=None, order=None):
        from sympy.polys.fields import FracField

        # 如果传入的是FracField类型的对象且符号列表和排序方式都是None，则直接使用该对象
        if isinstance(domain_or_field, FracField) and symbols is None and order is None:
            field = domain_or_field
        else:
            # 否则创建一个新的FracField对象
            field = FracField(symbols, domain_or_field, order)

        # 将创建或传入的域对象保存到self.field属性中
        self.field = field
        # 将域对象的dtype属性赋值给self.dtype
        self.dtype = field.dtype

        # 将域对象的生成器列表、生成器个数、符号列表和域保存到相应的属性中
        self.gens = field.gens
        self.ngens = field.ngens
        self.symbols = field.symbols
        self.domain = field.domain

        # TODO: remove this
        # 将self.domain保存到self.dom属性中（这一行有待移除的标记，暂时保留）
        self.dom = self.domain

    # 创建一个新的分数域元素，使用域对象的field_new方法
    def new(self, element):
        return self.field.field_new(element)

    # 返回域对象的零元素
    @property
    def zero(self):
        return self.field.zero

    # 返回域对象的单位元素
    @property
    def one(self):
        return self.field.one

    # 返回域对象的排序方式
    @property
    def order(self):
        return self.field.order

    # 返回分数域对象的字符串表示形式
    def __str__(self):
        return str(self.domain) + '(' + ','.join(map(str, self.symbols)) + ')'

    # 返回分数域对象的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype.field, self.domain, self.symbols))

    # 判断两个分数域对象是否相等的方法
    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FractionField) and \
            (self.dtype.field, self.domain, self.symbols) ==\
            (other.dtype.field, other.domain, other.symbols)

    # 将分数域对象的元素转换为SymPy对象
    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    # 将SymPy的表达式转换为分数域对象的元素
    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.field.from_expr(a)

    # 将Python的整数对象转换为分数域对象的元素
    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    # 将Python的整数对象转换为分数域对象的元素（使用Python的纯Python实现）
    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    # 将Python的分数对象转换为分数域对象的元素
    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        dom = K1.domain
        conv = dom.convert_from
        if dom.is_ZZ:
            return K1(conv(K0.numer(a), K0)) / K1(conv(K0.denom(a), K0))
        else:
            return K1(conv(a, K0))

    # 将Python的分数对象转换为分数域对象的元素（使用Python的纯Python实现）
    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    # 将GMPY的mpz对象转换为分数域对象的元素
    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))
    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        # 使用 K1 的域来将 GMPY 的 mpq 对象 a 转换成 dtype 类型
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        # 使用 K1 的域来将 GaussianRational 对象 a 转换成 dtype 类型
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianInteger`` object to ``dtype``. """
        # 使用 K1 的域来将 GaussianInteger 对象 a 转换成 dtype 类型
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 使用 K1 的域来将 mpmath 的 mpf 对象 a 转换成 dtype 类型
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 使用 K1 的域来将 mpmath 的 mpf 对象 a 转换成 dtype 类型
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        # 如果 K1 的域不等于 K0，则将代数数 a 从 K1.domain 转换到 K0
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        # 如果成功转换了 a，则返回 K1.new(a)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        # 如果多项式 a 是常数项，则返回 K1.convert_from(a.coeff(1), K0.domain)
        if a.is_ground:
            return K1.convert_from(a.coeff(1), K0.domain)
        # 否则尝试使用 K1.field.ring 设置多项式 a 的环，并返回 K1.new(a.set_ring(K1.field.ring))
        try:
            return K1.new(a.set_ring(K1.field.ring))
        # 处理异常情况，如果无法转换，则返回 None
        except (CoercionFailed, GeneratorsError):
            # XXX: 如果 K1=ZZ(x,y) 和 K0=QQ[x,y]，且多项式 a 在 K0 中具有非整数系数
            # K1.new 可以处理这种情况，但是在 K0.domain 是代数域时，K1.new 就不起作用了
            try:
                return K1.new(a)
            except (CoercionFailed, GeneratorsError):
                return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        # 尝试将有理函数 a 设置为 K1.field 的字段，并返回结果
        try:
            return a.set_field(K1.field)
        # 处理异常情况，无法转换则返回 None
        except (CoercionFailed, GeneratorsError):
            return None

    def get_ring(self):
        """Returns a field associated with ``self``. """
        # 返回与 self 相关联的域
        return self.field.to_ring().to_domain()

    def is_positive(self, a):
        """Returns True if ``LC(a)`` is positive. """
        # 如果 a 的最高系数（LC(a)）为正数，则返回 True
        return self.domain.is_positive(a.numer.LC)

    def is_negative(self, a):
        """Returns True if ``LC(a)`` is negative. """
        # 如果 a 的最高系数（LC(a)）为负数，则返回 True
        return self.domain.is_negative(a.numer.LC)

    def is_nonpositive(self, a):
        """Returns True if ``LC(a)`` is non-positive. """
        # 如果 a 的最高系数（LC(a)）为非正数，则返回 True
        return self.domain.is_nonpositive(a.numer.LC)

    def is_nonnegative(self, a):
        """Returns True if ``LC(a)`` is non-negative. """
        # 如果 a 的最高系数（LC(a)）为非负数，则返回 True
        return self.domain.is_nonnegative(a.numer.LC)

    def numer(self, a):
        """Returns numerator of ``a``. """
        # 返回 a 的分子
        return a.numer

    def denom(self, a):
        """Returns denominator of ``a``. """
        # 返回 a 的分母
        return a.denom

    def factorial(self, a):
        """Returns factorial of ``a``. """
        # 返回 a 的阶乘
        return self.dtype(self.domain.factorial(a))
```