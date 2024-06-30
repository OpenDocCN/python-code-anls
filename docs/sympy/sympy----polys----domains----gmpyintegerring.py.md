# `D:\src\scipysrc\sympy\sympy\polys\domains\gmpyintegerring.py`

```
"""Implementation of :class:`GMPYIntegerRing` class. """

# 导入所需模块和函数
from sympy.polys.domains.groundtypes import (
    GMPYInteger, SymPyInteger,  # 导入 GMPYInteger 和 SymPyInteger 类型
    factorial as gmpy_factorial,  # 导入 factorial 函数并重命名为 gmpy_factorial
    gmpy_gcdex, gmpy_gcd, gmpy_lcm, gmpy_sqrt,  # 导入 GMPY 中的一些数学函数
)
from sympy.core.numbers import int_valued  # 导入 int_valued 函数
from sympy.polys.domains.integerring import IntegerRing  # 导入 IntegerRing 类
from sympy.polys.polyerrors import CoercionFailed  # 导入 CoercionFailed 异常类
from sympy.utilities import public  # 导入 public 装饰器

@public
class GMPYIntegerRing(IntegerRing):
    """Integer ring based on GMPY's ``mpz`` type.

    This will be the implementation of :ref:`ZZ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpz``.
    """

    dtype = GMPYInteger  # 设置数据类型为 GMPYInteger
    zero = dtype(0)  # 创建值为 0 的 GMPYInteger 对象
    one = dtype(1)  # 创建值为 1 的 GMPYInteger 对象
    tp = type(one)  # 获取 one 对象的类型
    alias = 'ZZ_gmpy'  # 设置别名为 'ZZ_gmpy'

    def __init__(self):
        """Allow instantiation of this domain. """

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))  # 将 a 转换为 SymPy 中的 Integer 对象

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return GMPYInteger(a.p)  # 如果 a 是整数，则使用其部分 p 创建 GMPYInteger 对象
        elif int_valued(a):
            return GMPYInteger(int(a))  # 如果 a 可以转换为整数，则创建相应的 GMPYInteger 对象
        else:
            raise CoercionFailed("expected an integer, got %s" % a)  # 抛出类型转换异常

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return K0.to_int(a)  # 将 ModularInteger(int) 转换为 GMPY 的 mpz 类型

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return GMPYInteger(a)  # 将 Python 的 int 转换为 GMPY 的 mpz 类型

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)  # 如果 a 是分数且分母为 1，则将分子转换为 GMPY 的 mpz 类型

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)  # 如果 a 是分数且分母为 1，则将分子转换为 GMPY 的 mpz 类型

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return K0.to_int(a)  # 将 ModularInteger(mpz) 转换为 GMPY 的 mpz 类型

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        return a  # 直接返回 GMPY 的 mpz 对象

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return a.numerator  # 如果 a 是 GMPY 的 mpq 类型且分母为 1，则返回其分子

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        p, q = K0.to_rational(a)  # 将 mpmath 的 mpf 类型 a 转换为有理数 p/q

        if q == 1:
            return GMPYInteger(p)  # 如果 q 为 1，则返回 p 的 GMPYInteger 类型

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x  # 如果 a 是高斯整数且虚部为 0，则返回其实部

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gmpy_gcdex(a, b)  # 计算 a 和 b 的扩展最大公约数
        return s, t, h  # 返回扩展最大公约数的结果

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gmpy_gcd(a, b)  # 计算 a 和 b 的最大公约数

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return gmpy_lcm(a, b)  # 计算 a 和 b 的最小公倍数

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return gmpy_sqrt(a)  # 计算 a 的平方根
    # 定义一个方法 factorial，用于计算参数 a 的阶乘
    def factorial(self, a):
        """Compute factorial of ``a``. """
        # 调用外部函数 gmpy_factorial 来计算阶乘，并返回结果
        return gmpy_factorial(a)
```