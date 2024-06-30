# `D:\src\scipysrc\sympy\sympy\polys\domains\rationalfield.py`

```
"""Implementation of :class:`RationalField` class. """

# 导入MPQ类，用于实现有理数域的功能
from sympy.external.gmpy import MPQ

# 导入SymPyRational, is_square, sqrtrem等函数和类
from sympy.polys.domains.groundtypes import SymPyRational, is_square, sqrtrem

# 导入CharacteristicZero类，用于处理特征为零的域
from sympy.polys.domains.characteristiczero import CharacteristicZero

# 导入Field类，作为有理数域的基类
from sympy.polys.domains.field import Field

# 导入SimpleDomain类，作为简单域的基类
from sympy.polys.domains.simpledomain import SimpleDomain

# 导入CoercionFailed异常类，用于类型转换失败时的异常处理
from sympy.polys.polyerrors import CoercionFailed

# 导入public装饰器，用于指示类的公共可见性
from sympy.utilities import public

@public
class RationalField(Field, CharacteristicZero, SimpleDomain):
    r"""Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """

    # 表示符号名称为'QQ'
    rep = 'QQ'
    # 别名也为'QQ'
    alias = 'QQ'

    # 表示当前类为有理数域
    is_RationalField = is_QQ = True
    # 表示该域属于数值类型
    is_Numerical = True

    # 表示有相关的环
    has_assoc_Ring = True
    # 表示有相关的域
    has_assoc_Field = True

    # 指定数据类型为MPQ，即通过gmpy库实现的有理数类型
    dtype = MPQ
    # 表示零元素为0
    zero = dtype(0)
    # 表示单位元素为1
    one = dtype(1)
    # 记录单位元素的类型
    tp = type(one)

    def __init__(self):
        # 构造函数为空，不执行额外操作
        pass

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # 如果other是RationalField类的实例，则相等
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Returns hash code of ``self``. """
        # 返回对象的哈希值，固定为'QQ'的哈希值
        return hash('QQ')

    def get_ring(self):
        """Returns ring associated with ``self``. """
        # 返回与当前域相关联的整数环ZZ
        from sympy.polys.domains import ZZ
        return ZZ

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        # 将当前域中的有理数a转换为SymPy的有理数对象SymPyRational
        return SymPyRational(int(a.numerator), int(a.denominator))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        # 将SymPy的整数对象a转换为当前域中的dtype类型的有理数
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            # 若无法转换，则抛出类型转换失败的异常
            raise CoercionFailed("expected `Rational` object, got %s" % a)
    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import QQ, sqrt
        >>> QQ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        from sympy.polys.domains import AlgebraicField
        # 返回一个代数域扩展对象，使用给定的生成元和可能的别名
        return AlgebraicField(self, *extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`QQ`.

        See :py:meth:`~.Domain.convert`
        """
        if a.is_ground:
            # 如果 a 是常数项，则转换为 K0.dom 中的元素
            return K1.convert(a.LC(), K0.dom)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            # 如果 GaussianElement 的虚部为 0，则返回实部作为 MPQ 类型的对象
            return MPQ(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        # 将 mpmath 中的浮点数对象转换为有理数的 MPQ 对象
        return MPQ(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        # 返回 a 和 b 的精确商，即 MPQ(a) / MPQ(b)
        return MPQ(a) / MPQ(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        # 返回 a 和 b 的商，即 MPQ(a) / MPQ(b)
        return MPQ(a) / MPQ(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        # 返回 a 对 b 的余数，这里表示为 self.zero，可能为该类的某个属性

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        # 返回 a 和 b 的商和余数的元组，即 (MPQ(a) / MPQ(b), self.zero)
        return MPQ(a) / MPQ(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        # 返回 a 的分子部分
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        # 返回 a 的分母部分
        return a.denominator
    # 如果给定的有理数 `a` 是一个完全平方数，则返回 `True`
    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        A rational number is a square if and only if there exists
        a rational number ``b`` such that ``b * b == a``.
        """
        # 调用 `is_square` 函数来检查分子和分母是否都是完全平方数
        return is_square(a.numerator) and is_square(a.denominator)

    # 如果给定的有理数 `a` 是一个完全平方数，则返回 `a` 的非负平方根
    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        # 如果分子是负数，则返回 `None`，因为分母总是正数
        if a.numerator < 0:  # denominator is always positive
            return None
        # 对分子取平方根并返回余数
        p_sqrt, p_rem = sqrtrem(a.numerator)
        # 如果余数不为零，返回 `None`
        if p_rem != 0:
            return None
        # 对分母取平方根并返回余数
        q_sqrt, q_rem = sqrtrem(a.denominator)
        # 如果余数不为零，返回 `None`
        if q_rem != 0:
            return None
        # 返回一个新的有理数对象，其值为分子的非负平方根和分母的非负平方根
        return MPQ(p_sqrt, q_sqrt)
# 创建有理数域 QQ 的对象，用于处理有理数运算
QQ = RationalField()
```