# `D:\src\scipysrc\sympy\sympy\external\pythonmpq.py`

```
# 导入操作符模块，用于后续的运算操作
import operator
# 导入求最大公约数的函数
from math import gcd
# 导入精确小数运算的Decimal类
from decimal import Decimal
# 导入Python标准库中的分数类Fraction
from fractions import Fraction
# 导入sys模块，用于获取系统相关信息
import sys
# 导入类型提示模块，将Tuple重命名为tTuple，用于类型提示
from typing import Tuple as tTuple, Type

# 用于计算哈希值的模数，从sys模块的哈希信息中获取
_PyHASH_MODULUS = sys.hash_info.modulus
# 用于计算哈希值的无穷大，从sys模块的哈希信息中获取
_PyHASH_INF = sys.hash_info.inf

class PythonMPQ:
    """PythonMPQ是一个纯Python实现的有理数类，兼容gmpy2的mpq。

    与fractions.Fraction相比，PythonMPQ略快。

    PythonMPQ应被视为不可变对象，尽管没有做任何防止变异的努力（因为那可能会减慢计算速度）。
    """
    # 定义PythonMPQ类的__slots__属性，限制实例动态添加属性
    __slots__ = ('numerator', 'denominator')
    def __new__(cls, numerator, denominator=None):
        """Construct PythonMPQ with gcd computation and checks"""
        if denominator is not None:
            # 如果给定分母，要求分子和分母必须是整数且分母不能为零
            if isinstance(numerator, int) and isinstance(denominator, int):
                # 计算最大公约数，这是一个耗时操作
                divisor = gcd(numerator, denominator)
                # 简化分数，使其最简形式
                numerator //= divisor
                denominator //= divisor
                # 调用内部方法创建对象并进行检查
                return cls._new_check(numerator, denominator)
        else:
            # 如果未给定分母，根据参数类型进行初始化
            # 如果 numerator 是整数，分母默认为 1
            if isinstance(numerator, int):
                return cls._new(numerator, 1)
            # 如果 numerator 是 PythonMPQ 对象，则从其属性创建新对象
            elif isinstance(numerator, PythonMPQ):
                return cls._new(numerator.numerator, numerator.denominator)

            # 如果 numerator 是 Decimal、float 或者 str 类型，则转换为 Fraction
            if isinstance(numerator, (Decimal, float, str)):
                numerator = Fraction(numerator)
            # 如果 numerator 是 Fraction 类型，则创建新对象
            if isinstance(numerator, Fraction):
                return cls._new(numerator.numerator, numerator.denominator)
        # 如果参数类型不符合要求，抛出类型错误异常
        raise TypeError("PythonMPQ() requires numeric or string argument")

    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
        # 构造 PythonMPQ 对象，并检查分母是否为零，同时规范化正负号
        if not denominator:
            raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
        elif denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return cls._new(numerator, denominator)

    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        # 高效构造 PythonMPQ 对象，不进行额外的检查
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __int__(self):
        """Convert to int (truncates towards zero)"""
        # 将 PythonMPQ 对象转换为整数（朝向零截断）
        p, q = self.numerator, self.denominator
        if p < 0:
            return -(-p//q)
        return p//q

    def __float__(self):
        """Convert to float (approximately)"""
        # 将 PythonMPQ 对象转换为浮点数（近似值）
        return self.numerator / self.denominator

    def __bool__(self):
        """True/False if nonzero/zero"""
        # 如果分子非零，返回 True；否则返回 False
        return bool(self.numerator)
    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
        # 检查是否与另一个 PythonMPQ 对象相等，如果是则比较分子和分母是否相同
        if isinstance(other, PythonMPQ):
            return (self.numerator == other.numerator
                and self.denominator == other.denominator)
        # 如果是兼容的类型之一，则将其转换为 PythonMPQ 对象再进行比较
        elif isinstance(other, self._compatible_types):
            return self.__eq__(PythonMPQ(other))
        else:
            return NotImplemented

    def __hash__(self):
        """hash - same as mpq/Fraction"""
        # 尝试计算哈希值，使用分母的模反元素进行计算
        try:
            dinv = pow(self.denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            hash_ = _PyHASH_INF
        else:
            # 计算哈希值
            hash_ = hash(hash(abs(self.numerator)) * dinv)
        # 根据分子的正负决定最终的哈希值
        result = hash_ if self.numerator >= 0 else -hash_
        # 如果结果为 -1，则返回 -2，否则返回计算出的结果
        return -2 if result == -1 else result

    def __reduce__(self):
        """Deconstruct for pickling"""
        # 返回用于 pickle 的对象类型和其状态数据（分子和分母）
        return type(self), (self.numerator, self.denominator)

    def __str__(self):
        """Convert to string"""
        # 将对象转换为字符串表示形式
        if self.denominator != 1:
            return f"{self.numerator}/{self.denominator}"
        else:
            return f"{self.numerator}"

    def __repr__(self):
        """Convert to string"""
        # 返回对象的字符串表示形式，通常用于调试和显示
        return f"MPQ({self.numerator},{self.denominator})"

    def _cmp(self, other, op):
        """Helper for lt/le/gt/ge"""
        # 用于比较运算符（<, <=, >, >=）的辅助函数
        if not isinstance(other, self._compatible_types):
            return NotImplemented
        # 计算两个对象的乘积，并使用给定的操作符进行比较
        lhs = self.numerator * other.denominator
        rhs = other.numerator * self.denominator
        return op(lhs, rhs)

    def __lt__(self, other):
        """self < other"""
        # 实现小于操作符的比较
        return self._cmp(other, operator.lt)

    def __le__(self, other):
        """self <= other"""
        # 实现小于等于操作符的比较
        return self._cmp(other, operator.le)

    def __gt__(self, other):
        """self > other"""
        # 实现大于操作符的比较
        return self._cmp(other, operator.gt)

    def __ge__(self, other):
        """self >= other"""
        # 实现大于等于操作符的比较
        return self._cmp(other, operator.ge)

    def __abs__(self):
        """abs(q)"""
        # 返回该对象的绝对值
        return self._new(abs(self.numerator), self.denominator)

    def __pos__(self):
        """+q"""
        # 返回对象本身，正号操作
        return self

    def __neg__(self):
        """-q"""
        # 返回对象的负值
        return self._new(-self.numerator, self.denominator)
    def __add__(self, other):
        """q1 + q2"""
        # 如果 other 是 PythonMPQ 类型的对象
        if isinstance(other, PythonMPQ):
            # 获取当前对象和其他对象的分子和分母
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            # 计算当前对象和其他对象的分数和的最大公约数
            g = gcd(aq, bq)
            # 如果最大公约数为1，则直接计算分数和
            if g == 1:
                p = ap * bq + aq * bp
                q = bq * aq
            # 如果最大公约数不为1，则先进行分数的通分，然后再计算和
            else:
                q1, q2 = aq // g, bq // g
                p, q = ap * q2 + bp * q1, q1 * q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)

        # 如果 other 是整数类型
        elif isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
        else:
            # 如果 other 既不是 PythonMPQ 类型的对象也不是整数类型，则返回 NotImplemented
            return NotImplemented

        # 返回计算结果的新对象
        return self._new(p, q)

    def __radd__(self, other):
        """z1 + q2"""
        # 如果 other 是整数类型
        if isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
            return self._new(p, q)
        else:
            # 如果 other 不是整数类型，则返回 NotImplemented
            return NotImplemented

    def __sub__(self, other):
        """q1 - q2"""
        # 如果 other 是 PythonMPQ 类型的对象
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap * bq - aq * bp
                q = bq * aq
            else:
                q1, q2 = aq // g, bq // g
                p, q = ap * q2 - bp * q1, q1 * q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)
        # 如果 other 是整数类型
        elif isinstance(other, int):
            p = self.numerator - self.denominator * other
            q = self.denominator
        else:
            # 如果 other 既不是 PythonMPQ 类型的对象也不是整数类型，则返回 NotImplemented
            return NotImplemented

        # 返回计算结果的新对象
        return self._new(p, q)

    def __rsub__(self, other):
        """z1 - q2"""
        # 如果 other 是整数类型
        if isinstance(other, int):
            p = self.denominator * other - self.numerator
            q = self.denominator
            return self._new(p, q)
        else:
            # 如果 other 不是整数类型，则返回 NotImplemented
            return NotImplemented

    def __mul__(self, other):
        """q1 * q2"""
        # 如果 other 是 PythonMPQ 类型的对象
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            # 计算分数乘法的分子和分母
            x1 = gcd(ap, bq)
            x2 = gcd(bp, aq)
            p, q = ((ap // x1) * (bp // x2), (aq // x2) * (bq // x1))
        # 如果 other 是整数类型
        elif isinstance(other, int):
            x = gcd(other, self.denominator)
            p = self.numerator * (other // x)
            q = self.denominator // x
        else:
            # 如果 other 既不是 PythonMPQ 类型的对象也不是整数类型，则返回 NotImplemented
            return NotImplemented

        # 返回计算结果的新对象
        return self._new(p, q)
    def __rmul__(self, other):
        """z1 * q2"""
        # 如果 other 是整数类型
        if isinstance(other, int):
            # 计算自身分母与 other 的最大公约数
            x = gcd(self.denominator, other)
            # 计算乘法结果的分子
            p = self.numerator*(other//x)
            # 计算乘法结果的分母
            q = self.denominator//x
            # 返回一个新的分数对象，表示乘法结果
            return self._new(p, q)
        else:
            # 如果 other 不是整数类型，返回 Not Implemented
            return NotImplemented

    def __pow__(self, exp):
        """q ** z"""
        # 获取自身分子和分母
        p, q = self.numerator, self.denominator

        # 如果指数 exp 是负数
        if exp < 0:
            # 调换分子和分母，并将 exp 变为正数
            p, q, exp = q, p, -exp

        # 返回一个新的分数对象，表示乘幂结果
        return self._new_check(p**exp, q**exp)

    def __truediv__(self, other):
        """q1 / q2"""
        # 如果 other 是 PythonMPQ 类型的对象
        if isinstance(other, PythonMPQ):
            # 获取自身和 other 的分子和分母
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            # 计算分子的最大公约数
            x1 = gcd(ap, bp)
            # 计算分母的最大公约数
            x2 = gcd(bq, aq)
            # 计算除法结果的分子和分母
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))
        elif isinstance(other, int):
            # 如果 other 是整数类型
            x = gcd(other, self.numerator)
            # 计算除法结果的分子
            p = self.numerator//x
            # 计算除法结果的分母
            q = self.denominator*(other//x)
        else:
            # 如果 other 不是 PythonMPQ 类型或整数类型，返回 Not Implemented
            return NotImplemented

        # 返回一个新的分数对象，表示除法结果
        return self._new_check(p, q)

    def __rtruediv__(self, other):
        """z / q"""
        # 如果 other 是整数类型
        if isinstance(other, int):
            # 计算自身分子与 other 的最大公约数
            x = gcd(self.numerator, other)
            # 计算除法结果的分子
            p = self.denominator*(other//x)
            # 计算除法结果的分母
            q = self.numerator//x
            # 返回一个新的分数对象，表示除法结果
            return self._new_check(p, q)
        else:
            # 如果 other 不是整数类型，返回 Not Implemented
            return NotImplemented

    _compatible_types: tTuple[Type, ...] = ()
# 定义 PythonMPQ 将与之进行操作和比较的类型列表，包括 ==、+ 等操作。
# 在此处定义是为了能够将 PythonMPQ 自身包含在列表中。
PythonMPQ._compatible_types = (PythonMPQ, int, Decimal, Fraction)
```