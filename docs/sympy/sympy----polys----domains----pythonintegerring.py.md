# `D:\src\scipysrc\sympy\sympy\polys\domains\pythonintegerring.py`

```
"""Implementation of :class:`PythonIntegerRing` class. """

# 从 sympy.core.numbers 模块导入 int_valued 函数
from sympy.core.numbers import int_valued
# 从 sympy.polys.domains.groundtypes 模块导入需要的符号和函数
from sympy.polys.domains.groundtypes import (
    PythonInteger, SymPyInteger, sqrt as python_sqrt,
    factorial as python_factorial, python_gcdex, python_gcd, python_lcm,
)
# 从 sympy.polys.domains.integerring 模块导入 IntegerRing 类
from sympy.polys.domains.integerring import IntegerRing
# 从 sympy.polys.polyerrors 模块导入 CoercionFailed 异常
from sympy.polys.polyerrors import CoercionFailed
# 从 sympy.utilities 模块导入 public 函数
from sympy.utilities import public

# 使用 @public 装饰器声明 PythonIntegerRing 类为公共类
@public
# 定义 PythonIntegerRing 类，继承自 IntegerRing 类
class PythonIntegerRing(IntegerRing):
    """Integer ring based on Python's ``int`` type.

    This will be used as :ref:`ZZ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of the standard Python ``int`` type.
    """

    # 设置 dtype 属性为 PythonInteger，表示该整数环的元素类型为 PythonInteger
    dtype = PythonInteger
    # 定义 zero 属性为 dtype(0)，表示整数环中的零元素
    zero = dtype(0)
    # 定义 one 属性为 dtype(1)，表示整数环中的单位元素
    one = dtype(1)
    # 设置 alias 属性为 'ZZ_python'，表示该整数环的别名
    alias = 'ZZ_python'

    # 定义构造函数 __init__，允许实例化这个整数环
    def __init__(self):
        """Allow instantiation of this domain. """

    # 定义方法 to_sympy，将整数转换为 SymPy 对象
    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(a)

    # 定义方法 from_sympy，将 SymPy 的 Integer 转换为 dtype 类型
    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        # 如果 a 是整数类型
        if a.is_Integer:
            return PythonInteger(a.p)
        # 如果 a 可以转换为整数
        elif int_valued(a):
            return PythonInteger(int(a))
        # 否则抛出 CoercionFailed 异常
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    # 定义方法 from_FF_python，将 ModularInteger(int) 转换为 Python 的 int 类型
    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to Python's ``int``. """
        return K0.to_int(a)

    # 定义方法 from_ZZ_python，直接返回 Python 的 int 类型
    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to Python's ``int``. """
        return a

    # 定义方法 from_QQ，将 Python 的 Fraction 转换为 Python 的 int 类型
    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        # 如果 a 的分母为 1，则返回其分子
        if a.denominator == 1:
            return a.numerator

    # 定义方法 from_QQ_python，将 Python 的 Fraction 转换为 Python 的 int 类型
    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        # 如果 a 的分母为 1，则返回其分子
        if a.denominator == 1:
            return a.numerator

    # 定义方法 from_FF_gmpy，将 ModularInteger(mpz) 转换为 Python 的 int 类型
    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to Python's ``int``. """
        return PythonInteger(K0.to_int(a))

    # 定义方法 from_ZZ_gmpy，将 GMPY 的 mpz 转换为 Python 的 int 类型
    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to Python's ``int``. """
        return PythonInteger(a)

    # 定义方法 from_QQ_gmpy，将 GMPY 的 mpq 转换为 Python 的 int 类型
    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpq`` to Python's ``int``. """
        # 如果 a 的分母为 1，则返回其分子
        if a.denom() == 1:
            return PythonInteger(a.numer())

    # 定义方法 from_RealField，将 mpmath 的 mpf 转换为 Python 的 int 类型
    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to Python's ``int``. """
        # 将 mpmath 的 mpf 转换为有理数 p/q
        p, q = K0.to_rational(a)
        # 如果 q 等于 1，则返回整数 p
        if q == 1:
            return PythonInteger(p)

    # 定义方法 gcdex，计算 a 和 b 的扩展欧几里得算法结果
    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        return python_gcdex(a, b)

    # 定义方法 gcd，计算 a 和 b 的最大公约数
    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return python_gcd(a, b)

    # 定义方法 lcm，计算 a 和 b 的最小公倍数
    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return python_lcm(a, b)

    # 定义方法 sqrt，计算 a 的平方根
    def sqrt(self, a):
        """Compute square root of ``a``. """
        return python_sqrt(a)

    # 定义方法 factorial，计算 a 的阶乘
    def factorial(self, a):
        """Compute factorial of ``a``. """
        return python_factorial(a)
```