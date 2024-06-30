# `D:\src\scipysrc\sympy\sympy\polys\domains\gaussiandomains.py`

```
"""Domains of Gaussian type."""

# 导入复数域相关的库
from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
# 导入整数环 ZZ 和有理数域 QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
# 导入代数域 AlgebraicField、通用域 Domain、域元素 DomainElement、域 Field、环 Ring
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring

# 定义高斯类型域的元素类
class GaussianElement(DomainElement):
    """Base class for elements of Gaussian type domains."""
    base: Domain  # 元素类的基类域
    _parent: Domain  # 元素类所属的具体域对象

    __slots__ = ('x', 'y')  # 仅允许有 'x' 和 'y' 两个实例属性

    def __new__(cls, x, y=0):
        conv = cls.base.convert  # 获取转换函数
        return cls.new(conv(x), conv(y))  # 调用 new 方法创建新的元素对象

    @classmethod
    def new(cls, x, y):
        """Create a new GaussianElement of the same domain."""
        obj = super().__new__(cls)  # 调用父类的 __new__ 方法创建新对象
        obj.x = x  # 设置实部属性
        obj.y = y  # 设置虚部属性
        return obj

    def parent(self):
        """The domain that this is an element of (ZZ_I or QQ_I)"""
        return self._parent  # 返回元素所属的域对象

    def __hash__(self):
        return hash((self.x, self.y))  # 根据实部和虚部的哈希值进行元素的哈希计算

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y  # 比较两个高斯元素的实部和虚部是否相等
        else:
            return NotImplemented  # 若比较对象不是同类元素，则返回 Not Implemented

    def __lt__(self, other):
        if not isinstance(other, GaussianElement):
            return NotImplemented  # 若比较对象不是高斯元素，则返回 Not Implemented
        return [self.y, self.x] < [other.y, other.x]  # 按照虚部和实部的顺序进行小于比较

    def __pos__(self):
        return self  # 正号操作返回自身

    def __neg__(self):
        return self.new(-self.x, -self.y)  # 负号操作返回新的相反元素

    def __repr__(self):
        return "%s(%s, %s)" % (self._parent.rep, self.x, self.y)  # 返回元素的字符串表示形式

    def __str__(self):
        return str(self._parent.to_sympy(self))  # 返回元素的字符串表示形式，基于转换到 SymPy 对象

    @classmethod
    def _get_xy(cls, other):
        if not isinstance(other, cls):
            try:
                other = cls._parent.convert(other)  # 尝试将其他对象转换为当前域的元素
            except CoercionFailed:
                return None, None  # 转换失败则返回 None
        return other.x, other.y  # 返回其他对象的实部和虚部

    def __add__(self, other):
        x, y = self._get_xy(other)  # 获取其他对象的实部和虚部
        if x is not None:
            return self.new(self.x + x, self.y + y)  # 返回新的高斯元素对象，执行加法操作
        else:
            return NotImplemented  # 若无法进行加法操作则返回 Not Implemented

    __radd__ = __add__  # 右加法运算符与左加法运算符相同

    def __sub__(self, other):
        x, y = self._get_xy(other)  # 获取其他对象的实部和虚部
        if x is not None:
            return self.new(self.x - x, self.y - y)  # 返回新的高斯元素对象，执行减法操作
        else:
            return NotImplemented  # 若无法进行减法操作则返回 Not Implemented

    def __rsub__(self, other):
        x, y = self._get_xy(other)  # 获取其他对象的实部和虚部
        if x is not None:
            return self.new(x - self.x, y - self.y)  # 返回新的高斯元素对象，执行反向减法操作
        else:
            return NotImplemented  # 若无法进行减法操作则返回 Not Implemented

    def __mul__(self, other):
        x, y = self._get_xy(other)  # 获取其他对象的实部和虚部
        if x is not None:
            return self.new(self.x*x - self.y*y, self.x*y + self.y*x)  # 返回新的高斯元素对象，执行乘法操作
        else:
            return NotImplemented  # 若无法进行乘法操作则返回 Not Implemented

    __rmul__ = __mul__  # 右乘法运算符与左乘法运算符相同
    # 定义自定义类中的指数运算符重载方法
    def __pow__(self, exp):
        # 如果指数为0，返回单位元素
        if exp == 0:
            return self.new(1, 0)
        # 如果指数为负数，转换为其倒数，并修改指数为正数
        if exp < 0:
            self, exp = 1/self, -exp
        # 如果指数为1，直接返回自身
        if exp == 1:
            return self
        # 初始化幂次方和乘积结果
        pow2 = self
        prod = self if exp % 2 else self._parent.one
        # 将指数除以2，反复平方和乘积，直至指数为0
        exp //= 2
        while exp:
            pow2 *= pow2
            if exp % 2:
                prod *= pow2
            exp //= 2
        # 返回最终乘积结果
        return prod

    # 定义自定义类中的布尔值转换方法
    def __bool__(self):
        # 返回 x 或 y 的布尔值结果
        return bool(self.x) or bool(self.y)

    # 定义自定义类中的象限判定方法
    def quadrant(self):
        """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
        # 根据点的坐标位置判断所在象限，返回对应的象限编号（0-3）
        if self.y > 0:
            return 0 if self.x > 0 else 1
        elif self.y < 0:
            return 2 if self.x < 0 else 3
        else:
            return 0 if self.x >= 0 else 2

    # 定义自定义类中的右除法运算符重载方法
    def __rdivmod__(self, other):
        try:
            # 尝试将其他对象转换为当前类的类型
            other = self._parent.convert(other)
        except CoercionFailed:
            # 转换失败时返回未实现
            return NotImplemented
        else:
            # 调用其他对象的 divmod 方法，并传入当前对象作为参数
            return other.__divmod__(self)

    # 定义自定义类中的右真除法运算符重载方法
    def __rtruediv__(self, other):
        try:
            # 尝试将其他对象转换为 QQ_I 类型
            other = QQ_I.convert(other)
        except CoercionFailed:
            # 转换失败时返回未实现
            return NotImplemented
        else:
            # 调用其他对象的 truediv 方法，并传入当前对象作为参数
            return other.__truediv__(self)

    # 定义自定义类中的整数除法运算符重载方法
    def __floordiv__(self, other):
        # 调用当前对象的 divmod 方法，返回商，若未实现则返回未实现
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[0]

    # 定义自定义类中的右整数除法运算符重载方法
    def __rfloordiv__(self, other):
        # 调用当前对象的右除法方法，返回商，若未实现则返回未实现
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[0]

    # 定义自定义类中的取模运算符重载方法
    def __mod__(self, other):
        # 调用当前对象的 divmod 方法，返回余数，若未实现则返回未实现
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[1]

    # 定义自定义类中的右取模运算符重载方法
    def __rmod__(self, other):
        # 调用当前对象的右除法方法，返回余数，若未实现则返回未实现
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[1]
class GaussianInteger(GaussianElement):
    """Gaussian integer: domain element for :ref:`ZZ_I`

        >>> from sympy import ZZ_I
        >>> z = ZZ_I(2, 3)
        >>> z
        (2 + 3*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianInteger'>
    """
    base = ZZ  # 设置基础环为整数环 ZZ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        return QQ_I.convert(self)/other  # 转换为 Gaussian rational 并返回除法结果

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError('divmod({}, 0)'.format(self))  # 如果除数为零，则抛出 ZeroDivisionError 异常
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented  # 如果无法转换为合适类型，则返回 NotImplemented

        # multiply self and other by x - I*y
        # self/other == (a + I*b)/c
        a, b = self.x*x + self.y*y, -self.x*y + self.y*x
        c = x*x + y*y

        # find integers qx and qy such that
        # |a - qx*c| <= c/2 and |b - qy*c| <= c/2
        qx = (2*a + c) // (2*c)  # 计算 qx，使得 |a - qx*c| <= c/2
        qy = (2*b + c) // (2*c)  # 计算 qy，使得 |b - qy*c| <= c/2

        q = GaussianInteger(qx, qy)  # 创建新的 GaussianInteger 对象 q
        # |self/other - q| < 1 since
        # |a/c - qx|**2 + |b/c - qy|**2 <= 1/4 + 1/4 < 1

        return q, self - q*other  # 返回商 q 和余数 self - q*other


class GaussianRational(GaussianElement):
    """Gaussian rational: domain element for :ref:`QQ_I`

        >>> from sympy import QQ_I, QQ
        >>> z = QQ_I(QQ(2, 3), QQ(4, 5))
        >>> z
        (2/3 + 4/5*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianRational'>
    """
    base = QQ  # 设置基础环为有理数环 QQ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        if not other:
            raise ZeroDivisionError('{} / 0'.format(self))  # 如果除数为零，则抛出 ZeroDivisionError 异常
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented  # 如果无法转换为合适类型，则返回 NotImplemented
        c = x*x + y*y

        return GaussianRational((self.x*x + self.y*y)/c,
                                (-self.x*y + self.y*x)/c)  # 返回除法结果作为新的 GaussianRational 对象

    def __divmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented  # 如果转换失败，则返回 NotImplemented
        if not other:
            raise ZeroDivisionError('{} % 0'.format(self))  # 如果除数为零，则抛出 ZeroDivisionError 异常
        else:
            return self/other, QQ_I.zero  # 返回除法结果和零作为余数


class GaussianDomain():
    """Base class for Gaussian domains."""
    dom = None  # type: Domain

    is_Numerical = True  # 数值类型标志为 True
    is_Exact = True  # 精确类型标志为 True

    has_assoc_Ring = True  # 具有关联环标志为 True
    has_assoc_Field = True  # 具有关联域标志为 True

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        conv = self.dom.to_sympy  # 获取转换函数
        return conv(a.x) + I*conv(a.y)  # 返回 SymPy 对象

    def from_sympy(self, a):
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)  # 将 SymPy 对象的实部转换为当前域中的元素
        if not b:
            return self.new(x, 0)  # 如果虚部为零，则返回 (x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)  # 将 SymPy 对象的虚部转换为当前域中的元素
        if b is I:
            return self.new(x, y)  # 如果虚部是虚数单位，则返回 (x, y)
        else:
            raise CoercionFailed("{} is not Gaussian".format(a))  # 否则抛出 CoercionFailed 异常
    def inject(self, *gens):
        """将生成器注入到当前域中。"""
        # 调用 poly_ring 方法，将生成器作为参数传递
        return self.poly_ring(*gens)

    def canonical_unit(self, d):
        """返回规范单位元。"""
        # 使用 d 的象限值来获取对应的单位元，负号表示取倒数
        unit = self.units[-d.quadrant()]
        return unit

    def is_negative(self, element):
        """对于任何 GaussianElement 返回 False。"""
        # 总是返回 False，因为这个函数对 GaussianElement 没有具体实现
        return False

    def is_positive(self, element):
        """对于任何 GaussianElement 返回 False。"""
        # 总是返回 False，因为这个函数对 GaussianElement 没有具体实现
        return False

    def is_nonnegative(self, element):
        """对于任何 GaussianElement 返回 False。"""
        # 总是返回 False，因为这个函数对 GaussianElement 没有具体实现
        return False

    def is_nonpositive(self, element):
        """对于任何 GaussianElement 返回 False。"""
        # 总是返回 False，因为这个函数对 GaussianElement 没有具体实现
        return False

    def from_ZZ_gmpy(K1, a, K0):
        """将 GMPY 的 mpz 转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_ZZ(K1, a, K0):
        """将 ZZ_python 元素转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_ZZ_python(K1, a, K0):
        """将 ZZ_python 元素转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_QQ(K1, a, K0):
        """将 GMPY 的 mpq 转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_QQ_gmpy(K1, a, K0):
        """将 GMPY 的 mpq 转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_QQ_python(K1, a, K0):
        """将 QQ_python 元素转换为 self.dtype 类型。"""
        # 使用 K1 的构造函数将 a 转换为 self.dtype 类型
        return K1(a)

    def from_AlgebraicField(K1, a, K0):
        """将 ZZ<I> 或 QQ<I> 中的元素转换为 self.dtype 类型。"""
        # 检查 K0 的扩展参数是否为 I
        if K0.ext.args[0] == I:
            # 将 a 转换为 sympy 格式，再用 K1 的 from_sympy 方法转换为 self.dtype 类型
            return K1.from_sympy(K0.to_sympy(a))
# 定义一个类 GaussianIntegerRing，继承自 GaussianDomain 和 Ring
class GaussianIntegerRing(GaussianDomain, Ring):
    r"""Ring of Gaussian integers ``ZZ_I``

    The :ref:`ZZ_I` domain represents the `Gaussian integers`_ `\mathbb{Z}[i]`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of integers and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`ZZ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I)
    >>> p
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> p.domain
    ZZ_I

    The :ref:`ZZ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian integers.

    >>> from sympy import factor
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, domain='ZZ_I')
    (x - I)*(x + I)

    The corresponding `field of fractions`_ is the domain of the Gaussian
    rationals :ref:`QQ_I`. Conversely :ref:`ZZ_I` is the `ring of integers`_
    of :ref:`QQ_I`.

    >>> from sympy import ZZ_I, QQ_I
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`ZZ_I` can be used as a constructor.

    >>> ZZ_I(3, 4)
    (3 + 4*I)
    >>> ZZ_I(5)
    (5 + 0*I)

    The domain elements of :ref:`ZZ_I` are instances of
    :py:class:`~.GaussianInteger` which support the rings operations
    ``+,-,*,**``.

    >>> z1 = ZZ_I(5, 1)
    >>> z2 = ZZ_I(2, 3)
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 3*I)
    >>> z1 + z2
    (7 + 4*I)
    >>> z1 * z2
    (7 + 17*I)
    >>> z1 ** 2
    (24 + 10*I)

    Both floor (``//``) and modulo (``%``) division work with
    :py:class:`~.GaussianInteger` (see the :py:meth:`~.Domain.div` method).

    >>> z3, z4 = ZZ_I(5), ZZ_I(1, 3)
    >>> z3 // z4  # floor division
    (1 + -1*I)
    >>> z3 % z4   # modulo division (remainder)
    (1 + -2*I)
    >>> (z3//z4)*z4 + z3%z4 == z3
    True

    True division (``/``) in :ref:`ZZ_I` gives an element of :ref:`QQ_I`. The
    :py:meth:`~.Domain.exquo` method can be used to divide in :ref:`ZZ_I` when
    exact division is possible.

    >>> z1 / z2
    (1 + -1*I)
    >>> ZZ_I.exquo(z1, z2)
    (1 + -1*I)
    >>> z3 / z4
    (1/2 + -3/2*I)
    >>> ZZ_I.exquo(z3, z4)
    Traceback (most recent call last):
        ...
    ExactQuotientFailed: (1 + 3*I) does not divide (5 + 0*I) in ZZ_I

    The :py:meth:`~.Domain.gcd` method can be used to compute the `gcd`_ of any
    two elements.

    >>> ZZ_I.gcd(ZZ_I(10), ZZ_I(2))
    (2 + 0*I)
    >>> ZZ_I.gcd(ZZ_I(5), ZZ_I(2, 1))
    (2 + 1*I)

    .. _Gaussian integers: https://en.wikipedia.org/wiki/Gaussian_integer
    .. _gcd: https://en.wikipedia.org/wiki/Greatest_common_divisor

    """

    # 指定该类的域为 ZZ
    dom = ZZ
    # 指定类的数据类型为 GaussianInteger
    dtype = GaussianInteger
    # 定义零元素为 (0 + 0*I)
    zero = dtype(ZZ(0), ZZ(0))
    # 定义单位元素 1 为 (1 + 0*I)
    one = dtype(ZZ(1), ZZ(0))
    # 定义虚数单位 i 为 (0 + 1*I)
    imag_unit = dtype(ZZ(0), ZZ(1))
    # 定义单位元素集合，包括 1, i, -1, -i
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    # 类的表示字符串为 'ZZ_I'
    rep = 'ZZ_I'

    # 表示这个类是 Gaussian 环
    is_GaussianRing = True
    is_ZZ_I = True
    # 设置一个布尔变量 is_ZZ_I，表示当前类是 ZZ_I 类型

    def __init__(self):  # override Domain.__init__
        """For constructing ZZ_I."""
        # 构造函数，用于初始化 ZZ_I 类

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # 判断当前对象是否与另一个对象相等，如果是 GaussianIntegerRing 类型则返回 True
        if isinstance(other, GaussianIntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        # 计算当前对象的哈希值，固定返回 'ZZ_I' 的哈希值
        return hash('ZZ_I')

    @property
    def has_CharacteristicZero(self):
        # 返回属性值为 True，表示当前对象的特征为零
        return True

    def characteristic(self):
        # 返回当前对象的特征值，固定为 0
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        # 返回与当前对象相关联的环对象自身
        return self

    def get_field(self):
        """Returns a field associated with ``self``. """
        # 返回与当前对象相关联的域对象 QQ_I
        return QQ_I

    def normalize(self, d, *args):
        """Return first quadrant element associated with ``d``.

        Also multiply the other arguments by the same power of i.
        """
        # 根据给定的参数 d，返回其在第一象限的标准化元素，并对其他参数乘以相同的 i 的幂次
        unit = self.canonical_unit(d)
        d *= unit
        args = tuple(a*unit for a in args)
        return (d,) + args if args else d

    def gcd(self, a, b):
        """Greatest common divisor of a and b over ZZ_I."""
        # 计算 a 和 b 在 ZZ_I 上的最大公约数
        while b:
            a, b = b, a % b
        return self.normalize(a)

    def lcm(self, a, b):
        """Least common multiple of a and b over ZZ_I."""
        # 计算 a 和 b 在 ZZ_I 上的最小公倍数
        return (a * b) // self.gcd(a, b)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to ZZ_I."""
        # 将一个 ZZ_I 元素转换为另一个 ZZ_I 元素
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to ZZ_I."""
        # 将一个 QQ_I 元素转换为 ZZ_I 元素
        return K1.new(ZZ.convert(a.x), ZZ.convert(a.y))
# 将 GaussianIntegerRing 的实例赋给 ZZ_I，表示 GaussianInteger 的父类是 GaussianIntegerRing
ZZ_I = GaussianInteger._parent = GaussianIntegerRing()

# 定义 GaussianRationalField 类，它继承自 GaussianDomain 和 Field
class GaussianRationalField(GaussianDomain, Field):
    r"""Field of Gaussian rationals ``QQ_I``

    The :ref:`QQ_I` domain represents the `Gaussian rationals`_ `\mathbb{Q}(i)`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of rationals and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`QQ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I/2)
    >>> p
    Poly(x**2 + I/2, x, domain='QQ_I')
    >>> p.domain
    QQ_I

    The polys option ``gaussian=True`` can be used to specify that the domain
    should be :ref:`QQ_I` even if the coefficients do not contain ``I`` or are
    all integers.

    >>> Poly(x**2)
    Poly(x**2, x, domain='ZZ')
    >>> Poly(x**2 + I)
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> Poly(x**2/2)
    Poly(1/2*x**2, x, domain='QQ')
    >>> Poly(x**2, gaussian=True)
    Poly(x**2, x, domain='QQ_I')
    >>> Poly(x**2 + I, gaussian=True)
    Poly(x**2 + I, x, domain='QQ_I')
    >>> Poly(x**2/2, gaussian=True)
    Poly(1/2*x**2, x, domain='QQ_I')

    The :ref:`QQ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian rationals.

    >>> from sympy import factor, QQ_I
    >>> factor(x**2/4 + 1)
    (x**2 + 4)/4
    >>> factor(x**2/4 + 1, domain='QQ_I')
    (x - 2*I)*(x + 2*I)/4
    >>> factor(x**2/4 + 1, domain=QQ_I)
    (x - 2*I)*(x + 2*I)/4

    It is also possible to specify the :ref:`QQ_I` domain explicitly with
    polys functions like :py:func:`~.apart`.

    >>> from sympy import apart
    >>> apart(1/(1 + x**2))
    1/(x**2 + 1)
    >>> apart(1/(1 + x**2), domain=QQ_I)
    I/(2*(x + I)) - I/(2*(x - I))

    The corresponding `ring of integers`_ is the domain of the Gaussian
    integers :ref:`ZZ_I`. Conversely :ref:`QQ_I` is the `field of fractions`_
    of :ref:`ZZ_I`.

    >>> from sympy import ZZ_I, QQ_I, QQ
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`QQ_I` can be used as a constructor.

    >>> QQ_I(3, 4)
    (3 + 4*I)
    >>> QQ_I(5)
    (5 + 0*I)
    >>> QQ_I(QQ(2, 3), QQ(4, 5))
    (2/3 + 4/5*I)

    The domain elements of :ref:`QQ_I` are instances of
    :py:class:`~.GaussianRational` which support the field operations
    ``+,-,*,**,/``.

    >>> z1 = QQ_I(5, 1)
    >>> z2 = QQ_I(2, QQ(1, 2))
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 1/2*I)
    >>> z1 + z2
    (7 + 3/2*I)
    >>> z1 * z2
    (19/2 + 9/2*I)
    >>> z2 ** 2
    (15/4 + 2*I)

    True division (``/``) in :ref:`QQ_I` gives an element of :ref:`QQ_I` and
    is always exact.

    >>> z1 / z2
    (42/17 + -2/17*I)
    >>> QQ_I.exquo(z1, z2)
    (42/17 + -2/17*I)
    >>> z1 == (z1/z2)*z2
    True

    Both floor (``//``) and modulo (``%``) division can be used with
    dom = QQ
    dtype = GaussianRational
    zero = dtype(QQ(0), QQ(0))
    one = dtype(QQ(1), QQ(0))
    imag_unit = dtype(QQ(0), QQ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'QQ_I'

    is_GaussianField = True
    is_QQ_I = True

    def __init__(self):  # override Domain.__init__
        """For constructing QQ_I."""
        # 初始化 QQ_I 域的构造函数

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # 如果两个域相等则返回 True
        if isinstance(other, GaussianRationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        # 计算 self 的哈希码
        return hash('QQ_I')

    @property
    def has_CharacteristicZero(self):
        return True
        # 返回 True 表示 QQ_I 域特征为零

    def characteristic(self):
        return 0
        # 返回 QQ_I 域的特征值为零

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        # 返回与当前域相关联的环
        return ZZ_I

    def get_field(self):
        """Returns a field associated with ``self``. """
        # 返回与当前域相关联的字段
        return self

    def as_AlgebraicField(self):
        """Get equivalent domain as an ``AlgebraicField``. """
        # 返回等价的代数域对象
        return AlgebraicField(self.dom, I)

    def numer(self, a):
        """Get the numerator of ``a``."""
        # 获取 a 的分子部分
        ZZ_I = self.get_ring()
        return ZZ_I.convert(a * self.denom(a))

    def denom(self, a):
        """Get the denominator of ``a``."""
        # 获取 a 的分母部分
        ZZ = self.dom.get_ring()
        QQ = self.dom
        ZZ_I = self.get_ring()
        denom_ZZ = ZZ.lcm(QQ.denom(a.x), QQ.denom(a.y))
        return ZZ_I(denom_ZZ, ZZ.zero)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to QQ_I."""
        # 将 ZZ_I 元素转换为 QQ_I 元素
        return K1.new(a.x, a.y)

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to QQ_I."""
        # 将 QQ_I 元素转换为 QQ_I 元素
        return a

    def from_ComplexField(K1, a, K0):
        """Convert a ComplexField element to QQ_I."""
        # 将 ComplexField 元素转换为 QQ_I 元素
        return K1.new(QQ.convert(a.real), QQ.convert(a.imag))
# 将 GaussianRationalField() 的返回值赋给 QQ_I 和 GaussianRational._parent
QQ_I = GaussianRational._parent = GaussianRationalField()
```