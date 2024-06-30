# `D:\src\scipysrc\sympy\sympy\calculus\accumulationbounds.py`

```
# 导入必要的 SymPy 模块中的类和函数
from sympy.core import Add, Mul, Pow, S
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import _sympifyit, oo, zoo
from sympy.core.relational import is_le, is_lt, is_ge, is_gt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet

# 定义一个新的类 AccumulationBounds，它继承自 Expr 类
class AccumulationBounds(Expr):
    r"""An accumulation bounds.

    # 注意 AccumulationBounds 有一个别名: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that `a \le b`.

    `\left\langle a, b\right\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\left\langle -\infty, b\right\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\left\langle a, \infty \right\rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\left\langle -\infty, \infty \right\rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    ``oo`` and ``-oo`` are added to the second and third definition respectively,
    since if either ``-oo`` or ``oo`` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, ``1/AccumBounds(0, 1) = AccumBounds(1, oo)``, and the limit at
    `0` is not one-sided. As `x` tends to `0-`, then `1/x \rightarrow -\infty`, so `-\infty`
    should be interpreted as belonging to ``AccumBounds(1, oo)`` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of `\cos(x) + 1` are non-negative.
    (``AccumBounds(-1, 1) + 1 = AccumBounds(0, 2)``)

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X \times Y = \{ x \times y \mid x \in X \cap y \in Y\}`

    When an AccumBounds is raised to a negative power, if 0 is contained
    between the bounds then an infinite range is returned, otherwise if an
    endpoint is 0 then a semi-infinite range with consistent sign will be returned.

    AccumBounds in expressions behave a lot like Intervals but the
    semantics are not necessarily the same. Division (or exponentiation
    to a negative integer power) could be handled with *intervals* by
    # AccumBounds 类用于表示带界的数学表达式的区间
    # 定义了上下界之间的结果并集，不过在 AccumBounds 中并未实现
    # 此外，假定边界是彼此独立的；如果在表达式中多次使用相同的边界，则结果可能不是表达式的最大值或最小值（见下文）
    # 最后，当一个边界是 1 时，对无穷大的指数幂运算得到无穷大，而不是 1 或 NaN。
    
    Examples
    ========
    
    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x
    
    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    AccumBounds(1, 3)
    
    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    AccumBounds(-2, 1)
    
    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    AccumBounds(-3, 3)
    
    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    AccumBounds(3, 10)
    
    指数运算的 AccumulationBounds 定义如下：
    
    如果 0 不属于 `X` 或 `n > 0`，则
    
    `X^n = \{ x^n \mid x \in X\}`
    
    >>> AccumBounds(1, 4)**(S(1)/2)
    AccumBounds(1, 2)
    
    否则，会得到无穷或半无穷的结果：
    
    >>> 1/AccumBounds(-1, 1)
    AccumBounds(-oo, oo)
    >>> 1/AccumBounds(0, 2)
    AccumBounds(1/2, oo)
    >>> 1/AccumBounds(-oo, 0)
    AccumBounds(-oo, 0)
    
    边界为 1 时，总会生成所有非负数：
    
    >>> AccumBounds(1, 2)**oo
    AccumBounds(0, oo)
    >>> AccumBounds(0, 1)**oo
    AccumBounds(0, oo)
    
    如果指数本身是 AccumulationBounds 或者不是整数，则除非基数是正数，否则会返回未评估的结果：
    
    >>> AccumBounds(2, 3)**AccumBounds(-1, 2)
    AccumBounds(1/3, 9)
    >>> AccumBounds(-2, 3)**AccumBounds(-1, 2)
    AccumBounds(-2, 3)**AccumBounds(-1, 2)
    
    >>> AccumBounds(-2, -1)**(S(1)/2)
    sqrt(AccumBounds(-2, -1))
    
    注意：`\left\langle a, b\right\rangle^2` 不同于 `\left\langle a, b\right\rangle \times \left\langle a, b\right\rangle`
    
    >>> AccumBounds(-1, 1)**2
    AccumBounds(0, 1)
    
    >>> AccumBounds(1, 3) < 4
    True
    
    >>> AccumBounds(1, 3) < -1
    False
    
    一些基本函数也可以接受 AccumulationBounds 作为输入。
    对于一些实数 AccumulationBounds `\left\langle a, b \right\rangle`，函数 `f` 的定义为 `f(\left\langle a, b\right\rangle) = \{ f(x) \mid a \le x \le b \}`
    
    >>> sin(AccumBounds(pi/6, pi/3))
    AccumBounds(1/2, sqrt(3)/2)
    
    >>> exp(AccumBounds(0, 1))
    AccumBounds(1, E)
    
    >>> log(AccumBounds(1, E))
    AccumBounds(0, 1)
    
    表达式中的某些符号可以被替换为 AccumulationBounds 对象。但是，对于该表达式并不一定会评估 AccumulationBounds。
    
    同一表达式在替换的形式不同时可以计算为不同的值，因为每个 AccumulationBounds 的实例被视为独立的。例如：
    # 计算表达式 (x**2 + 2*x + 1) 在 x 取值区间 AccumBounds(-1, 1) 下的结果
    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    AccumBounds(-1, 4)

    # 计算表达式 ((x + 1)**2) 在 x 取值区间 AccumBounds(-1, 1) 下的结果
    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    AccumBounds(0, 4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] https://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """

    # 设置扩展实数的标志为真
    is_extended_real = True
    # 设置数字标志为假
    is_number = False

    def __new__(cls, min, max):
        # 将 min 和 max 转换为符号表达式
        min = _sympify(min)
        max = _sympify(max)

        # 只允许创建实数区间（使用带有 'is_extended_real=True' 的符号）
        if not min.is_extended_real or not max.is_extended_real:
            raise ValueError("Only real AccumulationBounds are supported")

        # 如果最大值和最小值相等，则返回最大值
        if max == min:
            return max

        # 确保创建的 AccumBounds 对象是有效的
        if max.is_number and min.is_number:
            bad = max.is_comparable and min.is_comparable and max < min
        else:
            bad = (max - min).is_extended_negative
        if bad:
            raise ValueError(
                "Lower limit should be smaller than upper limit")

        # 调用基类的构造方法创建对象
        return Basic.__new__(cls, min, max)

    # 设置操作优先级
    _op_priority = 11.0

    def _eval_is_real(self):
        # 检查最小值和最大值是否都为实数
        if self.min.is_real and self.max.is_real:
            return True

    @property
    def min(self):
        """
        返回 AccumulationBounds 对象可能取得的最小值。

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).min
        1

        """
        return self.args[0]

    @property
    def max(self):
        """
        返回 AccumulationBounds 对象可能取得的最大值。

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).max
        3

        """
        return self.args[1]

    @property
    def delta(self):
        """
        返回 AccumulationBounds 对象可能取得的最大值与最小值之间的差值。

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).delta
        2

        """
        return self.max - self.min

    @property
    def mid(self):
        """
        返回 AccumulationBounds 对象可能取得的最大值与最小值之间的中值。

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).mid
        2

        """
        return (self.min + self.max) / 2

    @_sympifyit('other', NotImplemented)
    def _eval_power(self, other):
        # 实现指数运算的内部方法
        return self.__pow__(other)

    @_sympifyit('other', NotImplemented)
    # 定义运算符重载方法 __add__，处理对象与表达式相加的情况
    def __add__(self, other):
        # 如果 other 是 Expr 类的实例
        if isinstance(other, Expr):
            # 如果 other 是 AccumBounds 类的实例
            if isinstance(other, AccumBounds):
                # 返回一个新的 AccumBounds 对象，其 min 和 max 分别为两个对象的对应属性相加
                return AccumBounds(
                    Add(self.min, other.min),
                    Add(self.max, other.max))
            # 如果 other 是正无穷或负无穷，并且 self 的范围是全负无穷到全正无穷
            if other is S.Infinity and self.min is S.NegativeInfinity or \
                    other is S.NegativeInfinity and self.max is S.Infinity:
                # 返回一个新的 AccumBounds 对象，范围为负无穷到正无穷
                return AccumBounds(-oo, oo)
            # 如果 other 是扩展实数
            elif other.is_extended_real:
                # 如果 self 的范围是全负无穷到全正无穷
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    # 返回一个新的 AccumBounds 对象，范围为负无穷到正无穷
                    return AccumBounds(-oo, oo)
                # 如果 self 的最小值是负无穷
                elif self.min is S.NegativeInfinity:
                    # 返回一个新的 AccumBounds 对象，其最小值为负无穷，最大值为 self 的最大值加上 other
                    return AccumBounds(-oo, self.max + other)
                # 如果 self 的最大值是正无穷
                elif self.max is S.Infinity:
                    # 返回一个新的 AccumBounds 对象，其最小值为 self 的最小值减去 other，最大值为正无穷
                    return AccumBounds(self.min + other, oo)
                else:
                    # 返回一个新的 AccumBounds 对象，其范围为 self 的 min 和 max 分别加上 other
                    return AccumBounds(Add(self.min, other), Add(self.max, other))
            # 返回对象自身与 other 相加的结果，不进行求值
            return Add(self, other, evaluate=False)
        # 如果 other 不是 Expr 类的实例，返回未实现
        return NotImplemented

    # 将右加法操作定义为与左加法相同，即委托给 __add__ 方法处理
    __radd__ = __add__

    # 定义取负操作符重载方法 __neg__
    def __neg__(self):
        # 返回一个新的 AccumBounds 对象，其范围为原对象的最大值和最小值分别取负
        return AccumBounds(-self.max, -self.min)

    # 定义减法操作符重载方法 __sub__
    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        # 如果 other 是 Expr 类的实例
        if isinstance(other, Expr):
            # 如果 other 是 AccumBounds 类的实例
            if isinstance(other, AccumBounds):
                # 返回一个新的 AccumBounds 对象，其 min 和 max 分别为两个对象的对应属性相减
                return AccumBounds(
                    Add(self.min, -other.max),
                    Add(self.max, -other.min))
            # 如果 other 是负无穷或正无穷，并且 self 的范围是全负无穷到全正无穷
            if other is S.NegativeInfinity and self.min is S.NegativeInfinity or \
                    other is S.Infinity and self.max is S.Infinity:
                # 返回一个新的 AccumBounds 对象，范围为负无穷到正无穷
                return AccumBounds(-oo, oo)
            # 如果 other 是扩展实数
            elif other.is_extended_real:
                # 如果 self 的范围是全负无穷到全正无穷
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    # 返回一个新的 AccumBounds 对象，范围为负无穷到正无穷
                    return AccumBounds(-oo, oo)
                # 如果 self 的最小值是负无穷
                elif self.min is S.NegativeInfinity:
                    # 返回一个新的 AccumBounds 对象，其最小值为负无穷，最大值为 self 的最大值减去 other
                    return AccumBounds(-oo, self.max - other)
                # 如果 self 的最大值是正无穷
                elif self.max is S.Infinity:
                    # 返回一个新的 AccumBounds 对象，其最小值为 self 的最小值减去 other，最大值为正无穷
                    return AccumBounds(self.min - other, oo)
                else:
                    # 返回一个新的 AccumBounds 对象，其 min 和 max 分别为 self 的对应属性减去 other
                    return AccumBounds(
                        Add(self.min, -other),
                        Add(self.max, -other))
            # 返回对象自身与 -other 相加的结果，不进行求值
            return Add(self, -other, evaluate=False)
        # 如果 other 不是 Expr 类的实例，返回未实现
        return NotImplemented

    # 定义右减法操作符重载方法 __rsub__
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        # 返回对象自身取负后与 other 相加的结果
        return self.__neg__() + other

    # 以下未完成的部分，暂时无法提供注释
    @_sympifyit('other', NotImplemented)
    # 定义乘法运算符的重载方法，用于处理自定义类的乘法操作
    def __mul__(self, other):
        # 如果自身的参数范围是负无穷到正无穷，则返回自身对象
        if self.args == (-oo, oo):
            return self
        # 如果 other 是 Expr 类的实例
        if isinstance(other, Expr):
            # 如果 other 是 AccumBounds 的实例
            if isinstance(other, AccumBounds):
                # 如果 other 的参数范围是负无穷到正无穷，则返回 other 对象
                if other.args == (-oo, oo):
                    return other
                # 创建一个空集合 v
                v = set()
                # 遍历自身的参数范围
                for a in self.args:
                    # 计算 other 乘以当前参数 a 的结果 vi
                    vi = other * a
                    # 将 vi 的参数加入集合 v，如果 vi.args 为空，则加入 vi 本身
                    v.update(vi.args or (vi,))
                # 返回一个新的 AccumBounds 对象，参数是集合 v 中的最小和最大值
                return AccumBounds(Min(*v), Max(*v))
            # 如果 other 是正无穷
            if other is S.Infinity:
                # 如果自身的最小值是零，则返回区间 [0, 正无穷)
                if self.min.is_zero:
                    return AccumBounds(0, oo)
                # 如果自身的最大值是零，则返回区间 (-无穷, 0]
                if self.max.is_zero:
                    return AccumBounds(-oo, 0)
            # 如果 other 是负无穷
            if other is S.NegativeInfinity:
                # 如果自身的最小值是零，则返回区间 (-无穷, 0]
                if self.min.is_zero:
                    return AccumBounds(-oo, 0)
                # 如果自身的最大值是零，则返回区间 [0, 正无穷)
                if self.max.is_zero:
                    return AccumBounds(0, oo)
            # 如果 other 是实数且不是无穷大
            if other.is_extended_real:
                # 如果 other 是零
                if other.is_zero:
                    # 如果自身的最大值是正无穷，则返回区间 [0, 正无穷)
                    if self.max is S.Infinity:
                        return AccumBounds(0, oo)
                    # 如果自身的最小值是负无穷，则返回区间 (-无穷, 0]
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(-oo, 0)
                    # 否则返回零
                    return S.Zero
                # 如果 other 是正数
                if other.is_extended_positive:
                    # 返回一个新的 AccumBounds 对象，其范围是自身最小值乘以 other 和最大值乘以 other
                    return AccumBounds(
                        Mul(self.min, other),
                        Mul(self.max, other))
                # 如果 other 是负数
                elif other.is_extended_negative:
                    # 返回一个新的 AccumBounds 对象，其范围是自身最大值乘以 other 和最小值乘以 other
                    return AccumBounds(
                        Mul(self.max, other),
                        Mul(self.min, other))
            # 如果 other 是 Order 类的实例
            if isinstance(other, Order):
                # 返回 other 对象
                return other
            # 返回一个新的 Mul 对象，表示自身乘以 other，不进行求值
            return Mul(self, other, evaluate=False)
        # 如果 other 不能处理乘法操作，则返回 NotImplemented
        return NotImplemented

    # 定义反向乘法运算符，与 __mul__ 方法相同
    __rmul__ = __mul__

    # 装饰器，确保方法的参数 other 被 sympify 后不是 NotImplemented
    @_sympifyit('other', NotImplemented)
    @_sympifyit('other', NotImplemented)
    # 实现反向真除法操作符，处理表达式对象与其他对象的真除法操作
    def __rtruediv__(self, other):
        # 检查 other 是否为 Expr 类型的对象
        if isinstance(other, Expr):
            # 检查 other 是否为扩展实数
            if other.is_extended_real:
                # 若 other 为零，则返回零
                if other.is_zero:
                    return S.Zero
                # 检查 self 的最小值是否非正且最大值是否非负
                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative):
                    # 如果 self 的最小值为零
                    if self.min.is_zero:
                        # 如果 other 是正数，则返回一个范围对象，包括负无穷到 other/self.max
                        if other.is_extended_positive:
                            return AccumBounds(Mul(other, 1 / self.max), oo)
                        # 如果 other 是负数，则返回一个范围对象，包括负无穷到 other/self.max
                        if other.is_extended_negative:
                            return AccumBounds(-oo, Mul(other, 1 / self.max))
                    # 如果 self 的最大值为零
                    if self.max.is_zero:
                        # 如果 other 是正数，则返回一个范围对象，包括负无穷到 other/self.min
                        if other.is_extended_positive:
                            return AccumBounds(-oo, Mul(other, 1 / self.min))
                        # 如果 other 是负数，则返回一个范围对象，包括 other/self.min 到正无穷
                        if other.is_extended_negative:
                            return AccumBounds(Mul(other, 1 / self.min), oo)
                    # 若 self 的最小值和最大值不同时为零，则返回一个范围对象，从负无穷到正无穷
                    return AccumBounds(-oo, oo)
                else:
                    # 返回一个范围对象，包括 other/self.min 和 other/self.max 的最小值和最大值
                    return AccumBounds(Min(other / self.min, other / self.max),
                                       Max(other / self.min, other / self.max))
            # 若 other 不是扩展实数，则返回 other 与 self 的真除法结果
            return Mul(other, 1 / self, evaluate=False)
        else:
            # 若 other 不是 Expr 类型，则返回 NotImplemented
            return NotImplemented

    # 定义反向乘幂操作符，处理表达式对象与其他对象的乘幂操作
    @_sympifyit('other', NotImplemented)
    @_sympifyit('other', NotImplemented)
    def __rpow__(self, other):
        # 检查 other 是否为实数且非负，且 self 的范围差大于零
        if other.is_real and other.is_extended_nonnegative and (
                self.max - self.min).is_extended_positive:
            # 如果 other 是 1，则返回 1
            if other is S.One:
                return S.One
            # 如果 other 是正数
            if other.is_extended_positive:
                # 计算 a 和 b 分别为 other 的 self.args 元素的乘幂
                a, b = [other**i for i in self.args]
                # 如果 a 不是最小的，则交换 a 和 b
                if min(a, b) != a:
                    a, b = b, a
                # 返回用 a 和 b 构建的新表达式对象
                return self.func(a, b)
            # 如果 other 是零
            if other.is_zero:
                # 如果 self 的最小值是零，则返回一个新的表达式对象 0 到 1
                if self.min.is_zero:
                    return self.func(0, 1)
                # 如果 self 的最小值是正数，则返回零
                if self.min.is_extended_positive:
                    return S.Zero

        # 若不符合上述条件，则返回 other 与 self 的乘幂结果
        return Pow(other, self, evaluate=False)

    # 定义取绝对值操作符，返回表达式对象的绝对值
    def __abs__(self):
        # 如果 self 的最大值为负数，则返回 self 的负值
        if self.max.is_extended_negative:
            return self.__neg__()
        # 如果 self 的最小值为负数，则返回一个范围对象，包括零到 abs(self.min) 和 self.max 的最大值
        elif self.min.is_extended_negative:
            return AccumBounds(S.Zero, Max(abs(self.min), self.max))
        else:
            # 如果 self 的最小值和最大值都非负，则返回 self
            return self
    def __contains__(self, other):
        """
        Returns ``True`` if other is contained in self, where other
        belongs to extended real numbers, ``False`` if not contained,
        otherwise TypeError is raised.

        Examples
        ========

        >>> from sympy import AccumBounds, oo
        >>> 1 in AccumBounds(-1, 3)
        True

        -oo and oo go together as limits (in AccumulationBounds).

        >>> -oo in AccumBounds(1, oo)
        True

        >>> oo in AccumBounds(-oo, 0)
        True

        """
        # 将输入的other符号化处理
        other = _sympify(other)

        # 如果other是正无穷大或负无穷大
        if other in (S.Infinity, S.NegativeInfinity):
            # 如果self的最小值是负无穷大或者self的最大值是正无穷大，返回True，否则返回False
            if self.min is S.NegativeInfinity or self.max is S.Infinity:
                return True
            return False

        # 计算self的最小值 <= other <= self的最大值，并返回结果
        rv = And(self.min <= other, self.max >= other)
        # 如果rv不是True或False，抛出TypeError异常
        if rv not in (True, False):
            raise TypeError("input failed to evaluate")
        return rv

    def intersection(self, other):
        """
        Returns the intersection of 'self' and 'other'.
        Here other can be an instance of :py:class:`~.FiniteSet` or AccumulationBounds.

        Parameters
        ==========

        other : AccumulationBounds
            Another AccumulationBounds object with which the intersection
            has to be computed.

        Returns
        =======

        AccumulationBounds
            Intersection of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import AccumBounds, FiniteSet
        >>> AccumBounds(1, 3).intersection(AccumBounds(2, 4))
        AccumBounds(2, 3)

        >>> AccumBounds(1, 3).intersection(AccumBounds(4, 6))
        EmptySet

        >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
        {1, 2}

        """
        # 如果other不是AccumulationBounds或FiniteSet对象，抛出TypeError异常
        if not isinstance(other, (AccumBounds, FiniteSet)):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        # 如果other是FiniteSet对象
        if isinstance(other, FiniteSet):
            fin_set = S.EmptySet
            # 遍历other中的元素，将属于self的元素加入到fin_set中
            for i in other:
                if i in self:
                    fin_set = fin_set + FiniteSet(i)
            return fin_set

        # 如果self的最大值小于other的最小值，或者self的最小值大于other的最大值，返回空集
        if self.max < other.min or self.min > other.max:
            return S.EmptySet

        # 如果self的最小值 <= other的最小值
        if self.min <= other.min:
            # 如果self的最大值 <= other的最大值，返回AccumBounds(other的最小值, self的最大值)
            if self.max <= other.max:
                return AccumBounds(other.min, self.max)
            # 如果self的最大值 > other的最大值，返回other
            if self.max > other.max:
                return other

        # 如果other的最小值 <= self的最小值
        if other.min <= self.min:
            # 如果other的最大值 < self的最大值，返回AccumBounds(self的最小值, other的最大值)
            if other.max < self.max:
                return AccumBounds(self.min, other.max)
            # 如果other的最大值 > self的最大值，返回self
            if other.max > self.max:
                return self
    def union(self, other):
        # TODO : Devise a better method for Union of AccumBounds
        # this method is not actually correct and
        # can be made better
        
        # 检查输入参数是否为 AccumBounds 类型，如果不是则抛出类型错误
        if not isinstance(other, AccumBounds):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        # 如果 self 的最小值小于等于 other 的最小值，并且 self 的最大值大于等于 other 的最小值
        if self.min <= other.min and self.max >= other.min:
            # 返回一个新的 AccumBounds 对象，其最小值为 self 和 other 最小值中的较小值，最大值为 self 和 other 最大值中的较大值
            return AccumBounds(self.min, Max(self.max, other.max))

        # 如果 other 的最小值小于等于 self 的最小值，并且 other 的最大值大于等于 self 的最小值
        if other.min <= self.min and other.max >= self.min:
            # 返回一个新的 AccumBounds 对象，其最小值为 self 和 other 最小值中的较小值，最大值为 self 和 other 最大值中的较大值
            return AccumBounds(other.min, Max(self.max, other.max))
@dispatch(AccumulationBounds, AccumulationBounds)  # type:ignore
def _eval_is_le(lhs, rhs):  # noqa:F811
    # 如果 lhs 的最大值小于等于 rhs 的最小值，则返回 True
    if is_le(lhs.max, rhs.min):
        return True
    # 如果 lhs 的最小值大于 rhs 的最大值，则返回 False
    if is_gt(lhs.min, rhs.max):
        return False


@dispatch(AccumulationBounds, Basic)  # type:ignore  # noqa:F811
def _eval_is_le(lhs, rhs):  # noqa: F811
    """
    如果 lhs 的值范围大于 rhs 的值范围，则返回 True。
    这里 rhs 可以是 AccumulationBounds 对象或扩展实数值。
    如果 rhs 满足相同的性质，则返回 False，否则返回未求值的 Relational 对象。
    """
    if not rhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(rhs), rhs))
    elif rhs.is_comparable:
        # 如果 lhs 的最大值小于等于 rhs，则返回 True
        if is_le(lhs.max, rhs):
            return True
        # 如果 lhs 的最小值大于 rhs，则返回 False
        if is_gt(lhs.min, rhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs):  # noqa:F811
    # 如果 lhs 的最小值大于等于 rhs 的最大值，则返回 True
    if is_ge(lhs.min, rhs.max):
        return True
    # 如果 lhs 的最大值小于 rhs 的最小值，则返回 False
    if is_lt(lhs.max, rhs.min):
        return False


@dispatch(AccumulationBounds, Expr)  # type:ignore
def _eval_is_ge(lhs, rhs):  # noqa: F811
    """
    如果 lhs 的值范围小于 rhs 的值范围，则返回 True。
    这里 rhs 可以是 AccumulationBounds 对象或扩展实数值。
    如果 rhs 满足相同的性质，则返回 False，否则返回未求值的 Relational 对象。
    """
    if not rhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(rhs), rhs))
    elif rhs.is_comparable:
        # 如果 lhs 的最小值大于等于 rhs，则返回 True
        if is_ge(lhs.min, rhs):
            return True
        # 如果 lhs 的最大值小于 rhs，则返回 False
        if is_lt(lhs.max, rhs):
            return False


@dispatch(Expr, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs):  # noqa:F811
    # 如果 rhs 的最大值小于等于 lhs，则返回 True
    if is_le(rhs.max, lhs):
        return True
    # 如果 rhs 的最小值大于 lhs，则返回 False
    if is_gt(rhs.min, lhs):
        return False
# 设置 AccumBounds 别名为 AccumulationBounds
AccumBounds = AccumulationBounds
```