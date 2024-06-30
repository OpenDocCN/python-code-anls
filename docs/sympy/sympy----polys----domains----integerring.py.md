# `D:\src\scipysrc\sympy\sympy\polys\domains\integerring.py`

```
"""Implementation of :class:`IntegerRing` class. """

# 导入必要的模块和类
from sympy.external.gmpy import MPZ, GROUND_TYPES

from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    SymPyInteger,
    factorial,
    gcdex, gcd, lcm, sqrt, is_square, sqrtrem,
)

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

import math

# 声明 IntegerRing 类并公开
@public
class IntegerRing(Ring, CharacteristicZero, SimpleDomain):
    r"""The domain ``ZZ`` representing the integers `\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """

    # 类属性定义
    rep = 'ZZ'        # 表示字符串 'ZZ'
    alias = 'ZZ'      # 别名 'ZZ'
    dtype = MPZ       # 数据类型为 MPZ（从 gmpy 导入的整数类型）
    zero = dtype(0)   # 表示整数 0
    one = dtype(1)    # 表示整数 1
    tp = type(one)    # 类型为整数的类型

    # 类属性定义
    is_IntegerRing = is_ZZ = True   # 表示是整数环
    is_Numerical = True             # 表示是数值类型
    is_PID = True                   # 表示是唯一分解整环

    has_assoc_Ring = True           # 表示有关联的环
    has_assoc_Field = True          # 表示有关联的域

    def __init__(self):
        """Allow instantiation of this domain. """
        pass

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, IntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute a hash value for this domain. """
        return hash('ZZ')

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def get_field(self):
        r"""Return the associated field of fractions :ref:`QQ`

        Returns
        =======

        :ref:`QQ`:
            The associated field of fractions :ref:`QQ`, a
            :py:class:`~.Domain` representing the rational numbers
            `\mathbb{Q}`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.get_field()
        QQ
        """
        # 导入 QQ 类并返回它
        from sympy.polys.domains import QQ
        return QQ
    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`.
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

        >>> from sympy import ZZ, sqrt
        >>> ZZ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        # 调用对象的 get_field 方法获取域，然后调用其 algebraic_field 方法返回代数域
        return self.get_field().algebraic_field(*extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`ZZ`.

        See :py:meth:`~.Domain.convert`.
        """
        # 如果 a 是常数项（is_ground），则将其最高次数系数（LC()）转换为 K0.dom 所属的类型（K1）
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def log(self, a, b):
        r"""Logarithm of *a* to the base *b*.

        Parameters
        ==========

        a: number
        b: number

        Returns
        =======

        $\\lfloor\log(a, b)\\rfloor$:
            Floor of the logarithm of *a* to the base *b*

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.log(ZZ(8), ZZ(2))
        3
        >>> ZZ.log(ZZ(9), ZZ(2))
        3

        Notes
        =====

        This function uses ``math.log`` which is based on ``float`` so it will
        fail for large integer arguments.
        """
        # 返回以整数类型（dtype(int)）表示的 log(a, b) 的下取整值
        return self.dtype(int(math.log(int(a), b)))

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        # 将 K0 类型的 ModularInteger(a) 转换为 GMPY 类型的 MPZ
        return MPZ(K0.to_int(a))

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        # 将 K0 类型的 ModularInteger(a) 转换为 GMPY 类型的 MPZ
        return MPZ(K0.to_int(a))

    def from_ZZ(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        # 将 Python 的整数类型 a 转换为 GMPY 类型的 MPZ
        return MPZ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        # 将 Python 的整数类型 a 转换为 GMPY 类型的 MPZ
        return MPZ(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        # 如果 a 是整数（分母为 1），则将其分子转换为 GMPY 类型的 MPZ
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        # 如果 a 是整数（分母为 1），则将其分子转换为 GMPY 类型的 MPZ
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        # 将 K0 类型的 ModularInteger(a) 转换为 GMPY 类型的 MPZ
        return MPZ(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        # 直接返回 GMPY 类型的 MPZ a
        return a
    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        # 如果 a 的分母为 1，则直接返回其分子作为整数结果
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        # 使用 K0.to_rational(a) 方法将 mpmath 的 mpf 转换为分数 p/q
        p, q = K0.to_rational(a)

        # 如果 q 等于 1，根据不同情况返回不同类型的 GMPY 的 mpz 对象
        if q == 1:
            # XXX: 如果 MPZ 是 flint.fmpz，并且 p 是 gmpy2.mpz，则需要通过 int 转换，
            # 因为 fmpz 和 mpz 无法相互识别。
            return MPZ(int(p))

    def from_GaussianIntegerRing(K1, a, K0):
        # 如果 a 的虚部为 0，则返回其实部
        if a.y == 0:
            return a.x

    def from_EX(K1, a, K0):
        """Convert ``Expression`` to GMPY's ``mpz``. """
        # 如果 a 是整数类型，则调用 K1.from_sympy(a) 进行转换
        if a.is_Integer:
            return K1.from_sympy(a)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        # 调用 gcdex(a, b) 函数计算扩展的最大公约数，返回结果根据 GROUND_TYPES 的值进行条件判断
        h, s, t = gcdex(a, b)
        # XXX: 这个条件逻辑应该在其他地方处理。
        if GROUND_TYPES == 'gmpy':
            return s, t, h
        else:
            return h, s, t

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        # 调用 gcd(a, b) 函数计算最大公约数，返回结果
        return gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        # 调用 lcm(a, b) 函数计算最小公倍数，返回结果
        return lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        # 调用 sqrt(a) 函数计算给定数的平方根，返回结果
        return sqrt(a)

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        An integer is a square if and only if there exists an integer
        ``b`` such that ``b * b == a``.
        """
        # 调用 is_square(a) 函数判断给定的数是否为完全平方数，返回布尔值
        return is_square(a)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        # 如果 a 小于 0，则返回 None
        if a < 0:
            return None
        # 调用 sqrtrem(a) 函数计算 a 的平方根及余数，如果余数不为 0，则返回 None，否则返回平方根
        root, rem = sqrtrem(a)
        if rem != 0:
            return None
        return root

    def factorial(self, a):
        """Compute factorial of ``a``. """
        # 调用 factorial(a) 函数计算给定数的阶乘，返回结果
        return factorial(a)
# 导入 SageMath 中的整数环 IntegerRing
ZZ = IntegerRing()
```