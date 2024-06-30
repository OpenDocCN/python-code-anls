# `D:\src\scipysrc\sympy\sympy\polys\domains\complexfield.py`

```
"""Implementation of :class:`ComplexField` class. """

# 导入必要的模块和类
from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float, I
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.gaussiandomains import QQ_I
from sympy.polys.domains.mpelements import MPContext
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import DomainError, CoercionFailed
from sympy.utilities import public

# 将 ComplexField 声明为公共类
@public
class ComplexField(Field, CharacteristicZero, SimpleDomain):
    """Complex numbers up to the given precision. """

    # 类级别属性：表示复数域的标识符
    rep = 'CC'

    # 类级别属性：指示该类是 ComplexField 类的实例
    is_ComplexField = is_CC = True

    # 类级别属性：指示这不是一个精确数域，而是一个数值数域
    is_Exact = False
    is_Numerical = True

    # 类级别属性：指示复数域没有相关的环结构，但有相关的域结构
    has_assoc_Ring = False
    has_assoc_Field = True

    # 类级别属性：默认精度为 53
    _default_precision = 53

    # 检查是否具有默认精度
    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    # 获取当前精度属性
    @property
    def precision(self):
        return self._context.prec

    # 获取当前十进制精度属性
    @property
    def dps(self):
        return self._context.dps

    # 获取当前容差属性
    @property
    def tolerance(self):
        return self._context.tolerance

    # 初始化 ComplexField 类
    def __init__(self, prec=_default_precision, dps=None, tol=None):
        # 创建 MPContext 上下文对象
        context = MPContext(prec, dps, tol, False)
        context._parent = self
        self._context = context

        # 设置数据类型为 mpc
        self._dtype = context.mpc
        # 设置零值和单位值
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

    # 获取数据类型的别名
    @property
    def tp(self):
        # XXX: 域处理 tp 作为 dtype 的别名。这里我们需要两个分离的概念：
        # dtype 是一个可调用的函数，用于创建/转换实例。
        # 我们使用 tp 和 isinstance 来检查对象是否已经是域的实例。
        return self._dtype

    # dtype 方法用于处理输入参数，以保证其符合 mpmath 库的要求
    def dtype(self, x, y=0):
        # XXX: 这是必需的，因为 mpmath 不认识 fmpz。
        # 如果将转换例程添加到 mpmath 中，可能可以删除这部分代码。
        if isinstance(x, SYMPY_INTS):
            x = int(x)
        if isinstance(y, SYMPY_INTS):
            y = int(y)
        return self._dtype(x, y)

    # 判断两个 ComplexField 类是否相等
    def __eq__(self, other):
        return (isinstance(other, ComplexField)
                and self.precision == other.precision
                and self.tolerance == other.tolerance)

    # 返回 ComplexField 类的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision, self.tolerance))

    # 将 element 转换为 SymPy 数字
    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element.real, self.dps) + I * Float(element.imag, self.dps)

    # 将 SymPy 数字转换为 ComplexField 类的数据类型
    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)
        real, imag = number.as_real_imag()

        if real.is_Number and imag.is_Number:
            return self.dtype(real, imag)
        else:
            raise CoercionFailed("expected complex number, got %s" % expr)
    def from_ZZ(self, element, base):
        return self.dtype(element)
    # 从整数转换为当前域中的元素类型

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))
    # 使用 GMPY 将整数转换为当前域中的元素类型

    def from_ZZ_python(self, element, base):
        return self.dtype(element)
    # 从 Python 整数转换为当前域中的元素类型

    def from_QQ(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)
    # 从有理数转换为当前域中的元素类型，通过整数除法计算

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / element.denominator
    # 从 Python 的有理数转换为当前域中的元素类型，通过 Python 的浮点数除法计算

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)
    # 使用 GMPY 将有理数转换为当前域中的元素类型，通过整数除法计算

    def from_GaussianIntegerRing(self, element, base):
        return self.dtype(int(element.x), int(element.y))
    # 从高斯整数环中的元素转换为当前域中的元素类型

    def from_GaussianRationalField(self, element, base):
        x = element.x
        y = element.y
        return (self.dtype(int(x.numerator)) / int(x.denominator) +
                self.dtype(0, int(y.numerator)) / int(y.denominator))
    # 从高斯有理数域中的元素转换为当前域中的元素类型，通过整数和浮点数除法计算

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))
    # 从代数域中的元素转换为当前域中的元素类型，使用 SymPy 进行转换和评估

    def from_RealField(self, element, base):
        return self.dtype(element)
    # 从实数域中的元素转换为当前域中的元素类型

    def from_ComplexField(self, element, base):
        if self == base:
            return element
        else:
            return self.dtype(element)
    # 从复数域中的元素转换为当前域中的元素类型，如果当前域与给定域相同则直接返回元素

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError("there is no ring associated with %s" % self)
    # 获取与当前域相关联的环

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        return QQ_I
    # 获取与当前域关联的精确域，这里返回的是 QQ_I

    def is_negative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False
    # 判断给定元素是否为负数，对于任何复数元素都返回 False

    def is_positive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False
    # 判断给定元素是否为正数，对于任何复数元素都返回 False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False
    # 判断给定元素是否为非负数，对于任何复数元素都返回 False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False
    # 判断给定元素是否为非正数，对于任何复数元素都返回 False

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one
    # 返回 a 和 b 的最大公约数

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b
    # 返回 a 和 b 的最小公倍数

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)
    # 检查 a 和 b 是否几乎相等，使用 _context 中的方法进行检查

    def is_square(self, a):
        """Returns ``True``. Every complex number has a complex square root."""
        return True
    # 判断给定元素是否为平方数，对于任何复数都返回 True

    def exsqrt(self, a):
        r"""Returns the principal complex square root of ``a``.

        Explanation
        ===========
        The argument of the principal square root is always within
        $(-\frac{\pi}{2}, \frac{\pi}{2}]$. The square root may be
        slightly inaccurate due to floating point rounding error.
        """
        return a ** 0.5
    # 返回给定元素的主复数平方根，根据主平方根的定义进行计算
# 创建一个复数域的对象实例，并将其赋值给变量 CC
CC = ComplexField()
```