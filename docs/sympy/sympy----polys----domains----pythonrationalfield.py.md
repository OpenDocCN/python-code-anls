# `D:\src\scipysrc\sympy\sympy\polys\domains\pythonrationalfield.py`

```
"""Implementation of :class:`PythonRationalField` class. """

# 导入必要的模块和类
from sympy.polys.domains.groundtypes import PythonInteger, PythonRational, SymPyRational
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

# 声明一个公共类 PythonRationalField，继承自 RationalField
@public
class PythonRationalField(RationalField):
    """Rational field based on :ref:`MPQ`.

    This will be used as :ref:`QQ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of :ref:`MPQ`.
    """

    # 类变量 dtype 指定为 PythonRational 类型，表示该有理数域的数据类型
    dtype = PythonRational
    # 类变量 zero 表示该有理数域的零元素
    zero = dtype(0)
    # 类变量 one 表示该有理数域的单位元素
    one = dtype(1)
    # 别名为 'QQ_python'，表示该有理数域的别名
    alias = 'QQ_python'

    # 初始化方法，无需额外操作，因此 pass
    def __init__(self):
        pass

    # 获取与该有理数域相关联的整数环的方法
    def get_ring(self):
        """Returns ring associated with ``self``. """
        # 导入 PythonIntegerRing 类并返回其实例
        from sympy.polys.domains import PythonIntegerRing
        return PythonIntegerRing()

    # 将该有理数域的元素转换为 SymPy 对象的方法
    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        # 使用 SymPyRational 类创建 SymPy 的有理数对象
        return SymPyRational(a.numerator, a.denominator)

    # 将 SymPy 的有理数对象转换为该有理数域的元素的方法
    def from_sympy(self, a):
        """Convert SymPy's Rational to `dtype`. """
        # 如果 a 是 SymPy 的有理数对象，则转换为 PythonRational 对象
        if a.is_Rational:
            return PythonRational(a.p, a.q)
        # 如果 a 是 SymPy 的浮点数对象，则先转换为有理数再转换为 PythonRational 对象
        elif a.is_Float:
            from sympy.polys.domains import RR
            p, q = RR.to_rational(a)
            return PythonRational(int(p), int(q))
        else:
            # 抛出类型转换异常
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    # 从 Python 的整数对象转换为该有理数域的元素的方法
    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        # 直接使用 PythonRational 类创建对象
        return PythonRational(a)

    # 从 Python 的分数对象转换为该有理数域的元素的方法
    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        # 直接返回传入的分数对象 a
        return a

    # 从 GMPY 的大整数对象转换为该有理数域的元素的方法
    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        # 使用 PythonInteger 将 GMPY 的大整数对象 a 转换为 PythonRational 对象
        return PythonRational(PythonInteger(a))

    # 从 GMPY 的有理数对象转换为该有理数域的元素的方法
    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        # 使用 PythonInteger 将 GMPY 的有理数对象 a 转换为 PythonRational 对象
        return PythonRational(PythonInteger(a.numer()),
                              PythonInteger(a.denom()))

    # 从 mpmath 的浮点数对象转换为该有理数域的元素的方法
    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        # 使用 K0.to_rational 将 mpmath 的浮点数对象 a 转换为 PythonRational 对象
        p, q = K0.to_rational(a)
        return PythonRational(int(p), int(q))

    # 返回给定有理数元素的分子的方法
    def numer(self, a):
        """Returns numerator of `a`. """
        return a.numerator

    # 返回给定有理数元素的分母的方法
    def denom(self, a):
        """Returns denominator of `a`. """
        return a.denominator
```