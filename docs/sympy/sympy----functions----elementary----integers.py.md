# `D:\src\scipysrc\sympy\sympy\functions\elementary\integers.py`

```
from typing import Tuple as tTuple  # 导入类型提示模块中的 Tuple 别名为 tTuple

from sympy.core.basic import Basic  # 导入 SymPy 核心模块中的 Basic 类
from sympy.core.expr import Expr  # 导入 SymPy 核心模块中的 Expr 类

from sympy.core import Add, S  # 导入 SymPy 核心模块中的 Add 和 S 对象
from sympy.core.evalf import get_integer_part, PrecisionExhausted  # 导入 SymPy 核心模块中的数值计算相关函数和异常
from sympy.core.function import Function  # 导入 SymPy 核心模块中的 Function 类
from sympy.core.logic import fuzzy_or  # 导入 SymPy 核心模块中的 fuzzy_or 函数
from sympy.core.numbers import Integer, int_valued  # 导入 SymPy 核心模块中的 Integer 类和 int_valued 函数
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq  # 导入 SymPy 核心模块中的关系运算相关类和函数
from sympy.core.sympify import _sympify  # 导入 SymPy 核心模块中的 _sympify 函数
from sympy.functions.elementary.complexes import im, re  # 导入 SymPy 基础函数库中的复数函数
from sympy.multipledispatch import dispatch  # 导入 SymPy 多分派模块中的 dispatch 函数

###############################################################################
######################### FLOOR and CEILING FUNCTIONS #########################
###############################################################################


class RoundFunction(Function):
    """Abstract base class for rounding functions."""

    args: tTuple[Expr]  # 类属性，定义为元组，包含 Expr 类型的参数

    @classmethod
    def eval(cls, arg):
        v = cls._eval_number(arg)  # 调用类方法 _eval_number 处理参数 arg 的数值计算
        if v is not None:
            return v

        if arg.is_integer or arg.is_finite is False:  # 检查 arg 是否为整数或非有限数
            return arg
        if arg.is_imaginary or (S.ImaginaryUnit*arg).is_real:  # 检查 arg 是否为虚数或虚数单位与 arg 的乘积为实数
            i = im(arg)  # 获取 arg 的虚部
            if not i.has(S.ImaginaryUnit):
                return cls(i)*S.ImaginaryUnit  # 如果虚部不含虚数单位，则返回虚部乘以虚数单位
            return cls(arg, evaluate=False)  # 否则返回 arg 的实例，不进行评估

        # Integral, numerical, symbolic part
        ipart = npart = spart = S.Zero  # 定义三个变量，分别初始化为零

        # Extract integral (or complex integral) terms
        intof = lambda x: int(x) if int_valued(x) else (
            x if x.is_integer else None)  # 定义函数 intof，用于提取整数或复数整数部分

        for t in Add.make_args(arg):  # 遍历 arg 的各项
            if t.is_imaginary and (i := intof(im(t))) is not None:  # 如果 t 是虚数且其虚部可转换为整数
                ipart += i*S.ImaginaryUnit  # 将其加到 ipart 上，并乘以虚数单位
            elif (i := intof(t)) is not None:  # 如果 t 可转换为整数
                ipart += i  # 将其加到 ipart 上
            elif t.is_number:  # 如果 t 是数值
                npart += t  # 将其加到 npart 上
            else:  # 否则将 t 加到 spart 上
                spart += t

        if not (npart or spart):  # 如果 npart 和 spart 都为空
            return ipart  # 返回 ipart

        # Evaluate npart numerically if independent of spart
        if npart and (
            not spart or
            npart.is_real and (spart.is_imaginary or (S.ImaginaryUnit*spart).is_real) or
                npart.is_imaginary and spart.is_real):
            try:
                r, i = get_integer_part(
                    npart, cls._dir, {}, return_ints=True)  # 尝试计算 npart 的整数部分
                ipart += Integer(r) + Integer(i)*S.ImaginaryUnit  # 将整数部分加到 ipart 上
                npart = S.Zero  # 将 npart 置零
            except (PrecisionExhausted, NotImplementedError):
                pass

        spart += npart  # 将 npart 加到 spart 上
        if not spart:
            return ipart  # 如果 spart 为空，返回 ipart
        elif spart.is_imaginary or (S.ImaginaryUnit*spart).is_real:
            return ipart + cls(im(spart), evaluate=False)*S.ImaginaryUnit  # 如果 spart 是虚数或虚数单位乘以 spart 是实数，返回 ipart 加上虚数部分
        elif isinstance(spart, (floor, ceiling)):
            return ipart + spart  # 如果 spart 是 floor 或 ceiling 类型的实例，返回 ipart 加上 spart
        else:
            return ipart + cls(spart, evaluate=False)  # 否则返回 ipart 加上 spart 的实例，不进行评估

    @classmethod
    def _eval_number(cls, arg):
        raise NotImplementedError()  # 抽象方法，子类需实现具体逻辑
    # 返回对象的第一个参数的有限性评估结果
    def _eval_is_finite(self):
        return self.args[0].is_finite
    
    # 返回对象的第一个参数的实数性评估结果
    def _eval_is_real(self):
        return self.args[0].is_real
    
    # 返回对象的第一个参数的整数性评估结果
    def _eval_is_integer(self):
        return self.args[0].is_real
# floor 类继承自 RoundFunction 类，实现了取整函数 floor，将参数取整为不大于其自身的最大整数。
class floor(RoundFunction):
    """
    Floor is a univariate function which returns the largest integer
    value not greater than its argument. This implementation
    generalizes floor to complex numbers by taking the floor of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import floor, E, I, S, Float, Rational
    >>> floor(17)
    17
    >>> floor(Rational(23, 10))
    2
    >>> floor(2*E)
    5
    >>> floor(-Float(0.567))
    -1
    >>> floor(-I/2)
    -I
    >>> floor(S(5)/2 + 5*I/2)
    2 + 2*I

    See Also
    ========

    sympy.functions.elementary.integers.ceiling

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/FloorFunction.html

    """
    # 私有变量 _dir，表示方向为负
    _dir = -1

    # 类方法 _eval_number 用于计算数字参数的 floor 值
    @classmethod
    def _eval_number(cls, arg):
        # 如果参数是数值类型，则返回其 floor 值
        if arg.is_Number:
            return arg.floor()
        # 如果参数是 floor 或 ceiling 函数的实例，则返回参数本身
        elif any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        # 如果参数是数值符号，则返回其与整数的近似区间的左端点
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[0]

    # 方法 _eval_as_leading_term 用于计算参数在 x 趋于零时的主导项的 floor 值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入必要的类和函数
        from sympy.calculus.accumulationbounds import AccumBounds
        # 获取参数表达式
        arg = self.args[0]
        # 计算参数在 x=0 处的值
        arg0 = arg.subs(x, 0)
        # 计算 floor(arg0)
        r = self.subs(x, 0)
        # 如果 arg0 是 NaN 或 AccumBounds 类型，则计算其在 x 趋于零时的极限，并取其 floor 值
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        # 如果 arg0 是有限值
        if arg0.is_finite:
            # 如果 arg0 等于 r，则根据参数在 x 方向上的变化情况调整返回值
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r - 1
                elif ndir.is_positive:
                    return r
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        # 否则，计算参数在 x 趋于零时的主导项
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    # 方法 _eval_nseries 用于计算参数的 n 级数展开，并返回其 floor 值
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 获取参数表达式
        arg = self.args[0]
        # 计算参数在 x=0 处的值
        arg0 = arg.subs(x, 0)
        # 计算 floor(arg0)
        r = self.subs(x, 0)
        # 如果 arg0 是 NaN，则计算其在 x 趋于零时的极限，并取其 floor 值
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        # 如果 arg0 是无穷大，则计算参数的 n 级数展开并返回
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(-1, 0)
            return s + o
        # 如果 arg0 等于 r，则根据参数在 x 方向上的变化情况调整返回值
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r - 1
            elif ndir.is_positive:
                return r
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    # 方法 _eval_is_negative 用于判断参数是否为负数
    def _eval_is_negative(self):
        return self.args[0].is_negative

    # 方法 _eval_is_nonnegative 用于判断参数是否为非负数
    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative
    # 将当前表达式重写为以 ceiling 函数为基础的形式
    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return -ceiling(-arg)

    # 将当前表达式重写为以 frac 函数为基础的形式
    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg - frac(arg)

    # 小于等于运算符重载方法
    def __le__(self, other):
        # 将 other 转换为 SymPy 的符号对象
        other = S(other)
        # 如果 self 的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数
            if other.is_integer:
                return self.args[0] < other + 1  # 返回是否小于 other 加 1
            # 如果 other 是数字且是实数
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)  # 返回是否小于 other 的上限
        # 如果 self 的第一个参数与 other 相等且 other 是实数
        if self.args[0] == other and other.is_real:
            return S.true  # 返回真值对象
        # 如果 other 是正无穷大且 self 是有限的
        if other is S.Infinity and self.is_finite:
            return S.true  # 返回真值对象

        return Le(self, other, evaluate=False)  # 返回未求值的 Le 对象

    # 大于等于运算符重载方法
    def __ge__(self, other):
        # 将 other 转换为 SymPy 的符号对象
        other = S(other)
        # 如果 self 的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数
            if other.is_integer:
                return self.args[0] >= other  # 返回是否大于等于 other
            # 如果 other 是数字且是实数
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)  # 返回是否大于等于 other 的上限
        # 如果 self 的第一个参数与 other 相等且 other 是实数
        if self.args[0] == other and other.is_real:
            return S.false  # 返回假值对象
        # 如果 other 是负无穷大且 self 是有限的
        if other is S.NegativeInfinity and self.is_finite:
            return S.true  # 返回真值对象

        return Ge(self, other, evaluate=False)  # 返回未求值的 Ge 对象

    # 大于运算符重载方法
    def __gt__(self, other):
        # 将 other 转换为 SymPy 的符号对象
        other = S(other)
        # 如果 self 的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数
            if other.is_integer:
                return self.args[0] >= other + 1  # 返回是否大于等于 other 加 1
            # 如果 other 是数字且是实数
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)  # 返回是否大于等于 other 的上限
        # 如果 self 的第一个参数与 other 相等且 other 是实数
        if self.args[0] == other and other.is_real:
            return S.false  # 返回假值对象
        # 如果 other 是负无穷大且 self 是有限的
        if other is S.NegativeInfinity and self.is_finite:
            return S.true  # 返回真值对象

        return Gt(self, other, evaluate=False)  # 返回未求值的 Gt 对象

    # 小于运算符重载方法
    def __lt__(self, other):
        # 将 other 转换为 SymPy 的符号对象
        other = S(other)
        # 如果 self 的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数
            if other.is_integer:
                return self.args[0] < other  # 返回是否小于 other
            # 如果 other 是数字且是实数
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)  # 返回是否小于 other 的上限
        # 如果 self 的第一个参数与 other 相等且 other 是实数
        if self.args[0] == other and other.is_real:
            return S.false  # 返回假值对象
        # 如果 other 是正无穷大且 self 是有限的
        if other is S.Infinity and self.is_finite:
            return S.true  # 返回真值对象

        return Lt(self, other, evaluate=False)  # 返回未求值的 Lt 对象
@dispatch(floor, Expr)
# 定义 _eval_is_eq 函数的分派版本，用于处理 floor 和 Expr 类型的参数
def _eval_is_eq(lhs, rhs): # noqa:F811
   return is_eq(lhs.rewrite(ceiling), rhs) or \
        is_eq(lhs.rewrite(frac),rhs)


class ceiling(RoundFunction):
    """
    Ceiling is a univariate function which returns the smallest integer
    value not less than its argument. This implementation
    generalizes ceiling to complex numbers by taking the ceiling of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import ceiling, E, I, S, Float, Rational
    >>> ceiling(17)
    17
    >>> ceiling(Rational(23, 10))
    3
    >>> ceiling(2*E)
    6
    >>> ceiling(-Float(0.567))
    0
    >>> ceiling(I/2)
    I
    >>> ceiling(S(5)/2 + 5*I/2)
    3 + 3*I

    See Also
    ========

    sympy.functions.elementary.integers.floor

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/CeilingFunction.html

    """
    _dir = 1

    @classmethod
    # 定义类方法 _eval_number，用于处理 ceiling 对象的数值求值
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.ceiling()
        elif any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[1]

    # 定义 _eval_as_leading_term 方法，用于计算在特定条件下的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r
                elif ndir.is_positive:
                    return r + 1
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 获取表达式的第一个参数
        arg = self.args[0]
        # 计算在 x=0 处的极限值
        arg0 = arg.subs(x, 0)
        # 计算表达式在 x=0 处的数值
        r = self.subs(x, 0)
        # 如果 arg0 是 NaN，则计算 x=0 处的负方向极限，并向上取整
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        # 如果 arg0 是无穷大，则计算 arg 的 n 阶泰勒展开，并返回展开式加上一个阶数对象
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1)
            return s + o
        # 如果 arg0 等于 r，则根据 arg 在 x 方向上的导数值确定返回值
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r
            elif ndir.is_positive:
                return r + 1
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        # 返回参数的负值的下取整的负值
        return -floor(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        # 返回参数与其负值的分数部分之和
        return arg + frac(-arg)

    def _eval_is_positive(self):
        # 判断表达式的第一个参数是否为正数
        return self.args[0].is_positive

    def _eval_is_nonpositive(self):
        # 判断表达式的第一个参数是否为非正数
        return self.args[0].is_nonpositive

    def __lt__(self, other):
        # 小于运算符的重载方法
        other = S(other)
        # 如果表达式的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数，则比较表达式的第一个参数是否小于等于 other-1
            if other.is_integer:
                return self.args[0] <= other - 1
            # 如果 other 是实数且为数值，则比较表达式的第一个参数是否小于等于 other 的下取整
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        # 如果表达式的第一个参数等于 other 且 other 是实数，则返回 False
        if self.args[0] == other and other.is_real:
            return S.false
        # 如果 other 是正无穷并且表达式有限，则返回 True
        if other is S.Infinity and self.is_finite:
            return S.true

        # 否则返回 Lt 对象，不进行求值
        return Lt(self, other, evaluate=False)

    def __gt__(self, other):
        # 大于运算符的重载方法
        other = S(other)
        # 如果表达式的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数，则比较表达式的第一个参数是否大于 other
            if other.is_integer:
                return self.args[0] > other
            # 如果 other 是实数且为数值，则比较表达式的第一个参数是否大于 other 的下取整
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        # 如果表达式的第一个参数等于 other 且 other 是实数，则返回 False
        if self.args[0] == other and other.is_real:
            return S.false
        # 如果 other 是负无穷并且表达式有限，则返回 True
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        # 否则返回 Gt 对象，不进行求值
        return Gt(self, other, evaluate=False)

    def __ge__(self, other):
        # 大于等于运算符的重载方法
        other = S(other)
        # 如果表达式的第一个参数是实数
        if self.args[0].is_real:
            # 如果 other 是整数，则比较表达式的第一个参数是否大于等于 other-1
            if other.is_integer:
                return self.args[0] > other - 1
            # 如果 other 是实数且为数值，则比较表达式的第一个参数是否大于等于 other 的下取整
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        # 如果表达式的第一个参数等于 other 且 other 是实数，则返回 True
        if self.args[0] == other and other.is_real:
            return S.true
        # 如果 other 是负无穷并且表达式有限，则返回 True
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        # 否则返回 Ge 对象，不进行求值
        return Ge(self, other, evaluate=False)
    # 定义小于等于操作符的特殊方法，用于对象与另一个对象的比较
    def __le__(self, other):
        # 将参数转换为 SymPy 对象 S，确保其他操作数是合适的类型
        other = S(other)
        # 如果自身的第一个参数是实数
        if self.args[0].is_real:
            # 如果其他对象是整数，则直接比较自身第一个参数和其他对象
            if other.is_integer:
                return self.args[0] <= other
            # 如果其他对象是数值并且是实数，则比较自身第一个参数和其向下取整的值
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        # 如果自身的第一个参数与其他对象相等且其他对象是实数，则返回逻辑假值
        if self.args[0] == other and other.is_real:
            return S.false
        # 如果其他对象是正无穷大且自身是有限的，则返回逻辑真值
        if other is S.Infinity and self.is_finite:
            return S.true

        # 如果以上条件都不满足，则返回 Le 对象的结果，但不进行求值
        return Le(self, other, evaluate=False)
# 定义一个特殊的函数 _eval_is_eq，用于比较两个表达式是否相等
@dispatch(ceiling, Basic)  # type:ignore
# 标注函数装饰器，指定参数类型为 ceiling 和 Basic，忽略类型检查
def _eval_is_eq(lhs, rhs): # noqa:F811
    # 判断 lhs 重写后是否与 rhs 相等，或者 rhs 是否与原始的 lhs 相等
    return is_eq(lhs.rewrite(floor), rhs) or is_eq(lhs.rewrite(frac),rhs)


class frac(Function):
    r"""Represents the fractional part of x

    For real numbers it is defined [1]_ as

    .. math::
        x - \left\lfloor{x}\right\rfloor

    Examples
    ========

    >>> from sympy import Symbol, frac, Rational, floor, I
    >>> frac(Rational(4, 3))
    1/3
    >>> frac(-Rational(4, 3))
    2/3

    returns zero for integer arguments

    >>> n = Symbol('n', integer=True)
    >>> frac(n)
    0

    rewrite as floor

    >>> x = Symbol('x')
    >>> frac(x).rewrite(floor)
    x - floor(x)

    for complex arguments

    >>> r = Symbol('r', real=True)
    >>> t = Symbol('t', real=True)
    >>> frac(t + I*r)
    I*frac(r) + frac(t)

    See Also
    ========

    sympy.functions.elementary.integers.floor
    sympy.functions.elementary.integers.ceiling

    References
    ===========

    .. [1] https://en.wikipedia.org/wiki/Fractional_part
    .. [2] https://mathworld.wolfram.com/FractionalPart.html

    """
    @classmethod
    def eval(cls, arg):
        # 导入累积边界的计算模块
        from sympy.calculus.accumulationbounds import AccumBounds

        # 定义内部函数 _eval，用于计算不同类型参数的 frac 函数值
        def _eval(arg):
            # 如果参数为正无穷或负无穷，则返回累积边界 0 到 1
            if arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(0, 1)
            # 如果参数为整数，则返回 0
            if arg.is_integer:
                return S.Zero
            # 如果参数为数值类型
            if arg.is_number:
                # 如果参数为 NaN，则返回 NaN
                if arg is S.NaN:
                    return S.NaN
                # 如果参数为复数无穷，则返回 NaN
                elif arg is S.ComplexInfinity:
                    return S.NaN
                else:
                    # 否则返回 arg - floor(arg)
                    return arg - floor(arg)
            # 其他情况返回 frac 对象
            return cls(arg, evaluate=False)

        # 初始化实部和虚部为零
        real, imag = S.Zero, S.Zero
        # 对参数 arg 中的每一项进行处理
        for t in Add.make_args(arg):
            # 对复数参数进行两次检查
            # 详见 issue-7649 获取更多详情
            if t.is_imaginary or (S.ImaginaryUnit*t).is_real:
                # 提取 t 的虚部并进行累加
                i = im(t)
                if not i.has(S.ImaginaryUnit):
                    imag += i
                else:
                    real += t
            else:
                # 对实部进行累加
                real += t

        # 计算实部和虚部的 frac 值
        real = _eval(real)
        imag = _eval(imag)
        # 返回实部加上虚部乘以虚数单位的结果
        return real + S.ImaginaryUnit*imag

    # 将 frac 函数重写为 floor 函数的形式
    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    # 将 frac 函数重写为 ceiling 函数的形式
    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    # 判断 frac 函数是否有限
    def _eval_is_finite(self):
        return True

    # 判断 frac 函数是否是实数
    def _eval_is_real(self):
        return self.args[0].is_extended_real

    # 判断 frac 函数是否是虚数
    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    # 判断 frac 函数是否是整数
    def _eval_is_integer(self):
        return self.args[0].is_integer

    # 判断 frac 函数是否为零
    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    # 判断 frac 函数是否为负数
    def _eval_is_negative(self):
        return False
    # 定义 "__ge__" 方法，用于实现大于或等于运算符（>=）
    def __ge__(self, other):
        # 如果 self 是扩展实数类型
        if self.is_extended_real:
            # 将 other 转换为符号表达式
            other = _sympify(other)
            # 检查 other 是否小于等于 0
            # 如果是，返回真值常量 S.true
            if other.is_extended_nonpositive:
                return S.true
            # 否则，调用内部方法 _value_one_or_more 来进一步判断
            res = self._value_one_or_more(other)
            # 如果 res 不为 None，则返回 not(res)
            if res is not None:
                return not(res)
        # 如果条件不满足，则返回符号表达式 Ge(self, other, evaluate=False)
        return Ge(self, other, evaluate=False)

    # 定义 "__gt__" 方法，用于实现大于运算符（>)
    def __gt__(self, other):
        # 如果 self 是扩展实数类型
        if self.is_extended_real:
            # 将 other 转换为符号表达式
            other = _sympify(other)
            # 调用内部方法 _value_one_or_more 来判断 other 是否大于等于 1
            res = self._value_one_or_more(other)
            # 如果 res 不为 None，则返回 not(res)
            if res is not None:
                return not(res)
            # 检查 other 是否小于 0
            # 如果是，返回真值常量 S.true
            if other.is_extended_negative:
                return S.true
        # 如果条件不满足，则返回符号表达式 Gt(self, other, evaluate=False)
        return Gt(self, other, evaluate=False)

    # 定义 "__le__" 方法，用于实现小于或等于运算符（<=）
    def __le__(self, other):
        # 如果 self 是扩展实数类型
        if self.is_extended_real:
            # 将 other 转换为符号表达式
            other = _sympify(other)
            # 检查 other 是否小于 0
            # 如果是，返回真值常量 S.false
            if other.is_extended_negative:
                return S.false
            # 否则，调用内部方法 _value_one_or_more 来进一步判断
            res = self._value_one_or_more(other)
            # 如果 res 不为 None，则返回 res
            if res is not None:
                return res
        # 如果条件不满足，则返回符号表达式 Le(self, other, evaluate=False)
        return Le(self, other, evaluate=False)

    # 定义 "__lt__" 方法，用于实现小于运算符（<）
    def __lt__(self, other):
        # 如果 self 是扩展实数类型
        if self.is_extended_real:
            # 将 other 转换为符号表达式
            other = _sympify(other)
            # 检查 other 是否小于等于 0
            # 如果是，返回真值常量 S.false
            if other.is_extended_nonpositive:
                return S.false
            # 否则，调用内部方法 _value_one_or_more 来进一步判断
            res = self._value_one_or_more(other)
            # 如果 res 不为 None，则返回 res
            if res is not None:
                return res
        # 如果条件不满足，则返回符号表达式 Lt(self, other, evaluate=False)
        return Lt(self, other, evaluate=False)

    # 定义内部方法 "_value_one_or_more"，用于判断 other 是否大于等于 1
    def _value_one_or_more(self, other):
        # 如果 other 是扩展实数类型
        if other.is_extended_real:
            # 如果 other 是数值类型
            if other.is_number:
                # 判断 other 是否大于等于 1，并且不是关系型表达式的实例
                res = other >= 1
                if res and not isinstance(res, Relational):
                    return S.true
            # 如果 other 是整数且为正数
            if other.is_integer and other.is_positive:
                return S.true

    # 定义内部方法 "_eval_as_leading_term"，用于计算作为主导项时的值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入需要的模块
        from sympy.calculus.accumulationbounds import AccumBounds
        # 获取参数列表中的第一个参数
        arg = self.args[0]
        # 计算参数在 x = 0 处的值
        arg0 = arg.subs(x, 0)
        # 计算表达式在 x = 0 处的值
        r = self.subs(x, 0)

        # 如果 arg0 是有限数值
        if arg0.is_finite:
            # 如果 r 是零
            if r.is_zero:
                # 计算参数在 x 方向上的导数方向
                ndir = arg.dir(x, cdir=cdir)
                # 如果导数方向是负数，返回常数 S.One
                if ndir.is_negative:
                    return S.One
                # 否则，返回 (arg - arg0) 在作为主导项时的值
                return (arg - arg0).as_leading_term(x, logx=logx, cdir=cdir)
            else:
                # 如果 r 不是零，则返回 r
                return r
        # 如果 arg0 是无限数值
        elif arg0 in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
            # 返回 AccumBounds(0, 1)
            return AccumBounds(0, 1)
        # 其他情况下，返回 arg 在作为主导项时的值
        return arg.as_leading_term(x, logx=logx, cdir=cdir)
    # 定义一个方法用于计算 nseries，即计算幂级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 导入 Order 类，用于处理级数展开的高阶项
        from sympy.series.order import Order
        # 获取函数表达式的第一个参数
        arg = self.args[0]
        # 计算在 x=0 处的 arg 的值
        arg0 = arg.subs(x, 0)
        # 计算在 x=0 处的函数表达式的值
        r = self.subs(x, 0)

        # 如果 arg0 是无穷大
        if arg0.is_infinite:
            # 导入 AccumBounds 类，用于处理积分界限
            from sympy.calculus.accumulationbounds import AccumBounds
            # 如果 n 小于等于 0，则返回 x=0 处的一阶 Order
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1) + Order(x**n, (x, 0))
            return o
        else:
            # 否则，计算 arg 减去 arg0 后的 nseries
            res = (arg - arg0)._eval_nseries(x, n, logx=logx, cdir=cdir)
            # 如果 r 是零
            if r.is_zero:
                # 确定 x=0 处的方向
                ndir = arg.dir(x, cdir=cdir)
                # 根据方向添加 S.One 或者 S.Zero 到结果中
                res += S.One if ndir.is_negative else S.Zero
            else:
                # 否则，将 r 添加到结果中
                res += r
            return res
@dispatch(frac, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # 检查是否满足 lhs.rewrite(floor) == rhs 或 lhs.rewrite(ceiling) == rhs
    if (lhs.rewrite(floor) == rhs) or \
        (lhs.rewrite(ceiling) == rhs):
        # 如果满足上述条件之一，返回 True
        return True
    
    # 检查 rhs 是否为负数
    if rhs.is_extended_negative:
        # 如果 rhs 是负数，返回 False
        return False
    
    # 检查 lhs 是否大于等于 1，使用 lhs._value_one_or_more(rhs) 进行检查
    res = lhs._value_one_or_more(rhs)
    if res is not None:
        # 如果 lhs 大于等于 1，返回 False
        return False
```