# `D:\src\scipysrc\sympy\sympy\polys\domains\quotientring.py`

```
# 导入从 sympy.polys.agca.modules 中导入 FreeModuleQuotientRing 模块
# 从 sympy.polys.domains.ring 中导入 Ring 类
# 从 sympy.polys.polyerrors 中导入 NotReversible 和 CoercionFailed 异常
# 从 sympy.utilities 中导入 public 函数
from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public

# TODO
# - successive quotients (when quotient ideals are implemented)
# - poly rings over quotients?
# - division by non-units in integral domains?

@public
# 定义 QuotientRing 类，继承自 Ring 类
class QuotientRing(Ring):
    """
    Class representing (commutative) quotient rings.

    You should not usually instantiate this by hand, instead use the constructor
    from the base ring in the construction.

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> I = QQ.old_poly_ring(x).ideal(x**3 + 1)
    >>> QQ.old_poly_ring(x).quotient_ring(I)
    QQ[x]/<x**3 + 1>

    Shorter versions are possible:

    >>> QQ.old_poly_ring(x)/I
    """
    QQ[x]/<x**3 + 1>

    >>> QQ.old_poly_ring(x)/[x**3 + 1]
    QQ[x]/<x**3 + 1>

    Attributes:

    - ring - the base ring
    - base_ideal - the ideal used to form the quotient
    """

    # 设置关联环和关联域的标志
    has_assoc_Ring = True
    has_assoc_Field = False
    # 定义元素类型为 QuotientRingElement
    dtype = QuotientRingElement

    def __init__(self, ring, ideal):
        # 检查理想 ideal 是否属于环 ring
        if not ideal.ring == ring:
            raise ValueError('Ideal must belong to %s, got %s' % (ring, ideal))
        # 初始化环和基础理想
        self.ring = ring
        self.base_ideal = ideal
        # 初始化零元素和单位元素
        self.zero = self(self.ring.zero)
        self.one = self(self.ring.one)

    def __str__(self):
        # 返回环和基础理想的字符串表示形式
        return str(self.ring) + "/" + str(self.base_ideal)

    def __hash__(self):
        # 返回对象的哈希值
        return hash((self.__class__.__name__, self.dtype, self.ring, self.base_ideal))

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        # 如果 a 不是环的元素，则转换为环的元素
        if not isinstance(a, self.ring.dtype):
            a = self.ring(a)
        # TODO optionally disable reduction?
        # 构造并返回新的 QuotientRingElement 对象，将 a 进行基础理想的约化
        return self.dtype(self, self.base_ideal.reduce_element(a))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # 检查两个 QuotientRing 对象是否相等
        return isinstance(other, QuotientRing) and \
            self.ring == other.ring and self.base_ideal == other.base_ideal

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        # 使用环 K1 中的 convert 方法将 Python 整数 a 转换为 dtype 类型
        return K1(K1.ring.convert(a, K0))

    from_ZZ_python = from_ZZ
    from_QQ_python = from_ZZ_python
    from_ZZ_gmpy = from_ZZ_python
    from_QQ_gmpy = from_ZZ_python
    from_RealField = from_ZZ_python
    from_GlobalPolynomialRing = from_ZZ_python
    from_FractionField = from_ZZ_python

    def from_sympy(self, a):
        # 使用环 self.ring 中的 from_sympy 方法将 sympy 对象 a 转换为当前 QuotientRing 的元素
        return self(self.ring.from_sympy(a))

    def to_sympy(self, a):
        # 使用环 self.ring 中的 to_sympy 方法将当前 QuotientRing 的元素 a 转换为 sympy 对象
        return self.ring.to_sympy(a.data)

    def from_QuotientRing(self, a, K0):
        # 如果 K0 与当前 QuotientRing 相同，则直接返回 a
        if K0 == self:
            return a

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        # 不允许嵌套域，因此抛出 NotImplementedError
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        # 不允许嵌套域，因此抛出 NotImplementedError
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        """
        Compute a**(-1), if possible.
        """
        # 计算元素 a 的逆，如果可能的话
        I = self.ring.ideal(a.data) + self.base_ideal
        try:
            return self(I.in_terms_of_generators(1)[0])
        except ValueError:  # 1 not in I
            # 如果 1 不在理想 I 中，则抛出 NotReversible 异常
            raise NotReversible('%s not a unit in %r' % (a, self))

    def is_zero(self, a):
        # 检查元素 a 是否为零元素
        return self.base_ideal.contains(a.data)

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        (QQ[x]/<x**2 + 1>)**2
        """
        # 返回一个在当前 QuotientRing 上的秩为 rank 的自由模
        return FreeModuleQuotientRing(self, rank)
```