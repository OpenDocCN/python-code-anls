# `D:\src\scipysrc\sympy\sympy\polys\domains\finitefield.py`

```
"""Implementation of :class:`FiniteField` class. """

# 从 sympy.external.gmpy 导入 GROUND_TYPES 常量
from sympy.external.gmpy import GROUND_TYPES
# 从 sympy.utilities.decorator 导入 doctest_depends_on 装饰器
from sympy.utilities.decorator import doctest_depends_on

# 从 sympy.core.numbers 导入 int_valued 函数
from sympy.core.numbers import int_valued
# 从 sympy.polys.domains.field 导入 Field 类
from sympy.polys.domains.field import Field

# 从 sympy.polys.domains.modularinteger 导入 ModularIntegerFactory 类
from sympy.polys.domains.modularinteger import ModularIntegerFactory
# 从 sympy.polys.domains.simpledomain 导入 SimpleDomain 类
from sympy.polys.domains.simpledomain import SimpleDomain
# 从 sympy.polys.galoistools 导入 gf_zassenhaus 和 gf_irred_p_rabin 函数
from sympy.polys.galoistools import gf_zassenhaus, gf_irred_p_rabin
# 从 sympy.polys.polyerrors 导入 CoercionFailed 异常类
from sympy.polys.polyerrors import CoercionFailed
# 从 sympy.utilities 导入 public 函数
from sympy.utilities import public
# 从 sympy.polys.domains.groundtypes 导入 SymPyInteger 类
from sympy.polys.domains.groundtypes import SymPyInteger

# 如果 GROUND_TYPES 等于 'flint'，跳过 'FiniteField' 相关的 doctest 测试
if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['FiniteField']

# 如果 GROUND_TYPES 等于 'flint'，导入 flint 模块并检查版本以确定使用哪种整数类型
if GROUND_TYPES == 'flint':
    import flint
    # 如果 python-flint 版本低于 0.5.0，提示不支持的版本
    _major, _minor, *_ = flint.__version__.split('.')
    if (int(_major), int(_minor)) < (0, 5):
        flint = None
else:
    flint = None


# 定义一个函数 _modular_int_factory，根据 flint 的可用性返回对应的整数工厂函数或 Python 实现的 ModularIntegerFactory 对象
def _modular_int_factory(mod, dom, symmetric, self):

    # 如果 flint 可用，使用 flint 实现的整数工厂函数
    if flint is not None:
        try:
            # 尝试将 mod 转换为 dom 类型
            mod = dom.convert(mod)
        except CoercionFailed:
            raise ValueError('modulus must be an integer, got %s' % mod)

        # 如果 mod 超出了 flint 支持的范围，使用 fmpz_mod
        try:
            flint.nmod(0, mod)
        except OverflowError:
            # 使用 fmpz_mod 上下文
            ctx = flint.fmpz_mod_ctx(mod)
        else:
            # 使用 nmod 函数
            ctx = lambda x: flint.nmod(x, mod)

        return ctx

    # 如果 flint 不可用，使用 Python 实现的 ModularIntegerFactory 对象
    return ModularIntegerFactory(mod, dom, symmetric, self)


# 使用 public 装饰器声明 FiniteField 类为公共可访问的类，并依赖于 'python' 和 'gmpy' 模块的 doctest 测试
@public
@doctest_depends_on(modules=['python', 'gmpy'])
class FiniteField(Field, SimpleDomain):
    r"""Finite field of prime order :ref:`GF(p)`

    A :ref:`GF(p)` domain represents a `finite field`_ `\mathbb{F}_p` of prime
    order as :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression with integer
    coefficients will have the domain :ref:`ZZ`. However, if the ``modulus=p``
    option is given then the domain will be a finite field instead.

    >>> from sympy import Poly, Symbol
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + 1)
    >>> p
    Poly(x**2 + 1, x, domain='ZZ')
    >>> p.domain
    ZZ
    >>> p2 = Poly(x**2 + 1, modulus=2)
    >>> p2
    Poly(x**2 + 1, x, modulus=2)
    >>> p2.domain
    GF(2)

    It is possible to factorise a polynomial over :ref:`GF(p)` using the
    modulus argument to :py:func:`~.factor` or by specifying the domain
    explicitly. The domain can also be given as a string.

    >>> from sympy import factor, GF
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, modulus=2)
    (x + 1)**2
    >>> factor(x**2 + 1, domain=GF(2))
    (x + 1)**2
    >>> factor(x**2 + 1, domain='GF(2)')
    (x + 1)**2

    It is also possible to use :ref:`GF(p)` with the :py:func:`~.cancel`
    class FiniteField:
        rep = 'FF'  # 用于表示有限域的简称
        alias = 'FF'  # 有限域的别名
    
        is_FiniteField = is_FF = True  # 表示这是一个有限域
        is_Numerical = True  # 表示这是一个数值类型
    
        has_assoc_Ring = False  # 没有关联的环
        has_assoc_Field = True  # 有关联的域
    
        dom = None  # 域为None
        mod = None  # 模数为None
    
        def __init__(self, mod, symmetric=True):
            from sympy.polys.domains import ZZ
            self.dom = ZZ  # 设置域为整数环
    
            if mod <= 0:
                raise ValueError('modulus must be a positive integer, got %s' % mod)
    
            # 根据给定的模数和域创建模整数的数据类型
            self.dtype = _modular_int_factory(mod, self.dom, symmetric, self)
            self.zero = self.dtype(0)  # 设置零元素
            self.one = self.dtype(1)  # 设置单位元素
            self.mod = mod  # 设置模数
            self.sym = symmetric  # 设置对称性
            self._tp = type(self.zero)  # 设置类型属性
    
        @property
        def tp(self):
            return self._tp  # 返回类型属性
    
        def __str__(self):
            return 'GF(%s)' % self.mod  # 返回有限域的字符串表示形式
    
        def __hash__(self):
            return hash((self.__class__.__name__, self.dtype, self.mod, self.dom))  # 返回对象的哈希值
    
        def __eq__(self, other):
            """Returns ``True`` if two domains are equivalent. """
            return isinstance(other, FiniteField) and \
                self.mod == other.mod and self.dom == other.dom
    
        def characteristic(self):
            """Return the characteristic of this domain. """
            return self.mod  # 返回域的特征数
    
        def get_field(self):
            """Returns a field associated with ``self``. """
            return self  # 返回与当前有限域相关联的域对象
    
        def to_sympy(self, a):
            """Convert ``a`` to a SymPy object. """
            return SymPyInteger(self.to_int(a))  # 将有限域元素转换为SymPy对象
    
        def from_sympy(self, a):
            """Convert SymPy's Integer to SymPy's ``Integer``. """
            if a.is_Integer:
                return self.dtype(self.dom.dtype(int(a)))  # 将SymPy整数转换为当前有限域的元素
            elif int_valued(a):
                return self.dtype(self.dom.dtype(int(a)))  # 将整数值转换为当前有限域的元素
            else:
                raise CoercionFailed("expected an integer, got %s" % a)  # 抛出类型转换失败异常
    def to_int(self, a):
        """Convert ``val`` to a Python ``int`` object. """
        # 将输入的值转换为整数类型
        aval = int(a)
        # 如果设置了符号并且aval大于self.mod除以2，将aval减去self.mod
        if self.sym and aval > self.mod // 2:
            aval -= self.mod
        # 返回转换后的整数值aval
        return aval

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        # 判断a是否为正数，使用bool函数将结果转换为布尔值
        return bool(a)

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        # 判断a是否为非负数，始终返回True
        return True

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        # 判断a是否为负数，始终返回False
        return False

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        # 判断a是否为非正数，返回not a的结果，即a为False时返回True，a为True时返回False
        return not a

    def from_FF(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        # 将ModularInteger(int)类型的a转换为dtype类型
        return K1.dtype(K1.dom.from_ZZ(int(a), K0.dom))

    def from_FF_python(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        # 将ModularInteger(int)类型的a转换为dtype类型（Python实现）
        return K1.dtype(K1.dom.from_ZZ_python(int(a), K0.dom))

    def from_ZZ(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        # 将Python的整数a转换为dtype类型
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_ZZ_python(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        # 将Python的整数a转换为dtype类型（Python实现）
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_QQ(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        # 将Python的分数a转换为dtype类型，如果a的分母为1
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_QQ_python(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        # 将Python的分数a转换为dtype类型（Python实现），如果a的分母为1
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_FF_gmpy(K1, a, K0=None):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        # 将ModularInteger(mpz)类型的a转换为dtype类型
        return K1.dtype(K1.dom.from_ZZ_gmpy(a.val, K0.dom))

    def from_ZZ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpz`` to ``dtype``. """
        # 将GMPY的整数类型mpz a转换为dtype类型
        return K1.dtype(K1.dom.from_ZZ_gmpy(a, K0))

    def from_QQ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpq`` to ``dtype``. """
        # 将GMPY的分数类型mpq a转换为dtype类型，如果a的分母为1
        if a.denominator == 1:
            return K1.from_ZZ_gmpy(a.numerator)

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to ``dtype``. """
        # 将mpmath的浮点数类型mpf a转换为dtype类型
        p, q = K0.to_rational(a)

        # 如果q等于1，返回p对应的dtype类型
        if q == 1:
            return K1.dtype(K1.dom.dtype(p))

    def is_square(self, a):
        """Returns True if ``a`` is a quadratic residue modulo p. """
        # 判断a是否为模p下的二次剩余，即判断x**2-a是否为不可约多项式
        poly = [int(x) for x in [self.one, self.zero, -a]]
        return not gf_irred_p_rabin(poly, self.mod, self.dom)
    def exsqrt(self, a):
        """Calculate the square root modulo p of ``a`` if it is a quadratic residue.

        Explanation
        ===========
        This method computes the modular square root of ``a`` modulo ``self.mod``, which is assumed to be a prime number.
        It returns the smallest non-negative integer ``x`` such that ``x**2 ≡ a (mod self.mod)``.

        Parameters
        ----------
        a : int
            The integer whose square root modulo ``self.mod`` is to be computed.

        Returns
        -------
        int or None
            The modular square root of ``a``, or None if no square root exists.

        Notes
        -----
        - If ``self.mod`` is 2 or ``a`` is 0, the function directly returns ``a`` since these cases have trivial solutions.
        - For other values of ``self.mod``, it factors the polynomial x**2 - a using a square-free factorization routine.
          It then checks each factor to find a suitable square root modulo ``self.mod``.
          The square root returned is always less than or equal to ``self.mod // 2``.

        """
        # x**2-a is not square-free if a=0 or the field is characteristic 2
        if self.mod == 2 or a == 0:
            # Return directly if a is 0 or mod is 2
            return a
        # Otherwise, use square-free factorization routine to factorize x**2-a
        poly = [int(x) for x in [self.one, self.zero, -a]]
        for factor in gf_zassenhaus(poly, self.mod, self.dom):
            # Check each factor to find a suitable square root
            if len(factor) == 2 and factor[1] <= self.mod // 2:
                return self.dtype(factor[1])
        # Return None if no square root modulo self.mod is found
        return None
# 将 `FiniteField` 赋值给 `FF` 和 `GF` 两个变量
FF = GF = FiniteField
```