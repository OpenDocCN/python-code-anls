# `D:\src\scipysrc\sympy\sympy\polys\domains\polynomialring.py`

```
"""
Implementation of :class:`PolynomialRing` class.
"""

# 导入所需模块和类
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

# 定义 PolynomialRing 类，继承自 Ring 和 CompositeDomain
@public
class PolynomialRing(Ring, CompositeDomain):
    """A class for representing multivariate polynomial rings."""

    # 类属性声明
    is_PolynomialRing = is_Poly = True
    has_assoc_Ring  = True
    has_assoc_Field = True

    # 初始化方法
    def __init__(self, domain_or_ring, symbols=None, order=None):
        # 导入 PolyRing 类
        from sympy.polys.rings import PolyRing

        # 根据输入参数选择合适的 ring 对象
        if isinstance(domain_or_ring, PolyRing) and symbols is None and order is None:
            ring = domain_or_ring
        else:
            ring = PolyRing(symbols, domain_or_ring, order)

        # 设置实例变量
        self.ring = ring
        self.dtype = ring.dtype
        self.gens = ring.gens
        self.ngens = ring.ngens
        self.symbols = ring.symbols
        self.domain = ring.domain

        # 若 symbols 非空，检查是否是唯一变量的精确域，若是则设定为主理想整环
        if symbols:
            if ring.domain.is_Field and ring.domain.is_Exact and len(symbols)==1:
                self.is_PID = True

        # TODO: remove this
        # 设置一个别名 dom，指向 domain 实例

    # 方法定义
    def new(self, element):
        # 返回 ring 对象的 ring_new 方法
        return self.ring.ring_new(element)

    @property
    def zero(self):
        # 返回 ring 对象的 zero 属性
        return self.ring.zero

    @property
    def one(self):
        # 返回 ring 对象的 one 属性
        return self.ring.one

    @property
    def order(self):
        # 返回 ring 对象的 order 属性
        return self.ring.order

    def __str__(self):
        # 返回多项式环的字符串表示
        return str(self.domain) + '[' + ','.join(map(str, self.symbols)) + ']'

    def __hash__(self):
        # 返回当前对象的哈希值
        return hash((self.__class__.__name__, self.dtype.ring, self.domain, self.symbols))

    def __eq__(self, other):
        """Returns `True` if two domains are equivalent."""
        # 比较两个多项式环对象的相等性
        return isinstance(other, PolynomialRing) and \
            (self.dtype.ring, self.domain, self.symbols) == \
            (other.dtype.ring, other.domain, other.symbols)

    def is_unit(self, a):
        """Returns ``True`` if ``a`` is a unit of ``self``"""
        # 判断给定元素是否为当前多项式环的单位元素
        if not a.is_ground:
            return False
        K = self.domain
        return K.is_unit(K.convert_from(a, self))

    def canonical_unit(self, a):
        # 获取给定元素的规范单位元素
        u = self.domain.canonical_unit(a.LC)
        return self.ring.ground_new(u)

    def to_sympy(self, a):
        """Convert `a` to a SymPy object."""
        # 将多项式环元素转换为 SymPy 对象
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to `dtype`."""
        # 将 SymPy 表达式转换为当前多项式环的元素
        return self.ring.from_expr(a)

    @classmethod
    def from_ZZ(K1, a, K0):
        """Convert a Python `int` object to `dtype`."""
        # 将 Python 的整数对象转换为当前多项式环的元素
        return K1(K1.domain.convert(a, K0))

    @classmethod
    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`."""
        # 将 Python 的整数对象转换为当前多项式环的元素
        return K1(K1.domain.convert(a, K0))

    @classmethod
    def from_QQ(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`."""
        # 将 Python 的分数对象转换为当前多项式环的元素
        return K1(K1.domain.convert(a, K0))
    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a `GaussianInteger` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a `GaussianRational` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        # 使用 `K1` 的域将 `a` 从 `K0` 转换为 `K1` 的数据类型
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        # 如果 `K1` 的域不等于 `K0`，将 `a` 转换到 `K0`，然后返回新的 `K1` 实例
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        # 尝试将多项式 `a` 设置为 `K1` 的环
        try:
            return a.set_ring(K1.ring)
        except (CoercionFailed, GeneratorsError):
            return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        # 如果 `K1` 的域与 `K0` 相同，将 `a` 转换为 `K1` 的环中的列表形式
        if K1.domain == K0:
            return K1.ring.from_list([a])

        # 否则，计算有理函数 `a` 的分子和分母，并尝试转换
        q, r = K0.numer(a).div(K0.denom(a))

        if r.is_zero:
            return K1.from_PolynomialRing(q, K0.field.ring.to_domain())
        else:
            return None

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert from old poly ring to ``dtype``. """
        # 如果 `K1` 的符号与 `K0` 的生成器相同
        if K1.symbols == K0.gens:
            ad = a.to_dict()
            # 如果 `K1` 的域与 `K0` 的域不同，将字典中的每个元素转换为 `K1` 的域
            if K1.domain != K0.domain:
                ad = {m: K1.domain.convert(c) for m, c in ad.items()}
            return K1(ad)
        # 如果 `a` 是常数项且 `K0` 的域与 `K1` 相同，直接转换
        elif a.is_ground and K0.domain == K1:
            return K1.convert_from(a.to_list()[0], K0.domain)

    def get_field(self):
        """Returns a field associated with `self`. """
        # 返回与 `self` 相关联的域
        return self.ring.to_field().to_domain()

    def is_positive(self, a):
        """Returns True if `LC(a)` is positive. """
        # 如果 `a` 的首项系数 `LC(a)` 是正数，则返回 True
        return self.domain.is_positive(a.LC)

    def is_negative(self, a):
        """Returns True if `LC(a)` is negative. """
        # 如果 `a` 的首项系数 `LC(a)` 是负数，则返回 True
        return self.domain.is_negative(a.LC)

    def is_nonpositive(self, a):
        """Returns True if `LC(a)` is non-positive. """
        # 如果 `a` 的首项系数 `LC(a)` 是非正数，则返回 True
        return self.domain.is_nonpositive(a.LC)

    def is_nonnegative(self, a):
        """Returns True if `LC(a)` is non-negative. """
        # 如果 `a` 的首项系数 `LC(a)` 是非负数，则返回 True
        return self.domain.is_nonnegative(a.LC)

    def gcdex(self, a, b):
        """Extended GCD of `a` and `b`. """
        # 返回 `a` 和 `b` 的扩展最大公约数
        return a.gcdex(b)
    # 计算给定两个数的最大公约数（GCD）
    def gcd(self, a, b):
        """Returns GCD of `a` and `b`. """
        # 调用对象 `a` 的 gcd 方法来计算最大公约数
        return a.gcd(b)

    # 计算给定两个数的最小公倍数（LCM）
    def lcm(self, a, b):
        """Returns LCM of `a` and `b`. """
        # 调用对象 `a` 的 lcm 方法来计算最小公倍数
        return a.lcm(b)

    # 计算给定数的阶乘
    def factorial(self, a):
        """Returns factorial of `a`. """
        # 调用对象的 domain 属性的 factorial 方法来计算 `a` 的阶乘，
        # 然后将结果转换为当前对象的数据类型（dtype）
        return self.dtype(self.domain.factorial(a))
```