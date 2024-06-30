# `D:\src\scipysrc\sympy\sympy\codegen\cfunctions.py`

```
"""
This module contains SymPy functions matching corresponding to special math functions in the
C standard library (since C99, also available in C++11).

The functions defined in this module allow the user to express functions such as ``expm1``
as a SymPy function for symbolic manipulation.

"""
# 导入 SymPy 中所需的模块和类
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt


# 定义函数 _expm1(x)，计算 exp(x) - 1
def _expm1(x):
    return exp(x) - S.One


# 定义类 expm1，表示指数函数减一
class expm1(Function):
    """
    Represents the exponential function minus one.

    Explanation
    ===========

    The benefit of using ``expm1(x)`` over ``exp(x) - 1``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import expm1
    >>> '%.0e' % expm1(1e-99).evalf()
    '1e-99'
    >>> from math import exp
    >>> exp(1e-99) - 1
    0.0
    >>> expm1(x).diff(x)
    exp(x)

    See Also
    ========

    log1p
    """
    nargs = 1

    # 计算该函数的第一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return exp(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    # 展开函数，返回 exp(x) - 1
    def _eval_expand_func(self, **hints):
        return _expm1(*self.args)

    # 重写为 exp 函数的表达式，返回 exp(arg) - 1
    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return exp(arg) - S.One

    # 同 _eval_rewrite_as_exp，用于处理
    _eval_rewrite_as_tractable = _eval_rewrite_as_exp

    # 对参数进行求值，返回 exp(arg) - 1 的结果
    @classmethod
    def eval(cls, arg):
        exp_arg = exp.eval(arg)
        if exp_arg is not None:
            return exp_arg - S.One

    # 判断参数是否为实数
    def _eval_is_real(self):
        return self.args[0].is_real

    # 判断参数是否有限
    def _eval_is_finite(self):
        return self.args[0].is_finite


# 定义函数 _log1p(x)，计算 log(x + 1)
def _log1p(x):
    return log(x + S.One)


# 定义类 log1p，表示自然对数加一
class log1p(Function):
    """
    Represents the natural logarithm of a number plus one.

    Explanation
    ===========

    The benefit of using ``log1p(x)`` over ``log(x + 1)``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log1p
    >>> from sympy import expand_log
    >>> '%.0e' % expand_log(log1p(1e-99)).evalf()
    '1e-99'
    >>> from math import log
    >>> log(1 + 1e-99)
    0.0
    >>> log1p(x).diff(x)
    1/(x + 1)

    See Also
    ========

    expm1
    """
    nargs = 1

    # 计算该函数的第一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(self.args[0] + S.One)
        else:
            raise ArgumentIndexError(self, argindex)
    # 根据提示求解扩展函数，返回以 log1p 函数作用于 self.args 的结果
    def _eval_expand_func(self, **hints):
        return _log1p(*self.args)

    # 重写为 log 函数，返回以 log1p 函数作用于 arg 的结果
    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log1p(arg)

    # 将 _eval_rewrite_as_log 方法作为 _eval_rewrite_as_tractable 的别名
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    # 类方法，评估参数 arg
    @classmethod
    def eval(cls, arg):
        # 如果 arg 是有理数，返回 log(arg + 1)
        if arg.is_Rational:
            return log(arg + S.One)
        # 如果 arg 不是浮点数，不安全添加 1 到浮点数，返回 log.eval(arg + 1)
        elif not arg.is_Float:
            return log.eval(arg + S.One)
        # 如果 arg 是数字，返回 log(有理数化的 arg + 1)
        elif arg.is_number:
            return log(Rational(arg) + S.One)

    # 判断实例是否为实数
    def _eval_is_real(self):
        return (self.args[0] + S.One).is_nonnegative

    # 判断实例是否有限
    def _eval_is_finite(self):
        # 如果 self.args[0] + 1 等于零，则返回 False
        if (self.args[0] + S.One).is_zero:
            return False
        # 否则，返回 self.args[0] 是否为有限数
        return self.args[0].is_finite

    # 判断实例是否为正数
    def _eval_is_positive(self):
        return self.args[0].is_positive

    # 判断实例是否为零
    def _eval_is_zero(self):
        return self.args[0].is_zero

    # 判断实例是否为非负数
    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative
# 定义符号常量 S(2)，表示数字 2
_Two = S(2)

# 定义函数 _exp2，计算 2 的 x 次方
def _exp2(x):
    return Pow(_Two, x)

# 定义类 exp2，表示以 2 为底的指数函数
class exp2(Function):
    """
    表示以 2 为底的指数函数。

    Explanation
    ===========

    使用 ``exp2(x)`` 而不是 ``2**x`` 的好处在于，在有限精度算术下，后者效率不高。

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import exp2
    >>> exp2(2).evalf() == 4.0
    True
    >>> exp2(x).diff(x)
    log(2)*exp2(x)

    See Also
    ========

    log2
    """
    nargs = 1

    # 计算函数的一阶导数
    def fdiff(self, argindex=1):
        """
        返回该函数的一阶导数。
        """
        if argindex == 1:
            return self*log(_Two)
        else:
            raise ArgumentIndexError(self, argindex)

    # 将函数重写为 Pow 形式
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _exp2(arg)

    # 在可扩展函数的情况下展开函数
    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow

    # 类方法，对参数进行求值
    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            return _exp2(arg)


# 定义函数 _log2，计算以 2 为底的对数
def _log2(x):
    return log(x)/log(_Two)

# 定义类 log2，表示以 2 为底的对数函数
class log2(Function):
    """
    表示以 2 为底的对数函数。

    Explanation
    ===========

    使用 ``log2(x)`` 而不是 ``log(x)/log(2)`` 的好处在于，在有限精度算术下，后者效率不高。

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log2
    >>> log2(4).evalf() == 2.0
    True
    >>> log2(x).diff(x)
    1/(x*log(2))

    See Also
    ========

    exp2
    log10
    """
    nargs = 1

    # 计算函数的一阶导数
    def fdiff(self, argindex=1):
        """
        返回该函数的一阶导数。
        """
        if argindex == 1:
            return S.One/(log(_Two)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    # 类方法，对参数进行求值
    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Two)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Two:
            return arg.exp

    # 在 evalf 方法中对函数进行重写求值
    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    # 在可扩展函数的情况下展开函数
    def _eval_expand_func(self, **hints):
        return _log2(*self.args)

    # 将函数重写为 log 形式
    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log2(arg)

    # 在可重写为 tractable 形式时重写函数
    _eval_rewrite_as_tractable = _eval_rewrite_as_log


# 定义函数 _fma，实现 "fused multiply add" 操作
def _fma(x, y, z):
    return x*y + z

# 定义类 fma，表示 "fused multiply add" 操作
class fma(Function):
    """
    表示 "fused multiply add" 操作。

    Explanation
    ===========

    使用 ``fma(x, y, z)`` 而不是 ``x*y + z`` 的好处在于，在有限精度算术下，前者在某些 CPU 上支持特殊指令。

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.cfunctions import fma
    >>> fma(x, y, z).diff(x)
    y

    """
    nargs = 3
    # 计算函数的一阶导数
    def fdiff(self, argindex=1):
        # 如果参数索引是 1 或 2，则返回对应的函数参数
        if argindex in (1, 2):
            return self.args[2 - argindex]
        # 如果参数索引是 3，则返回单位元素 S.One
        elif argindex == 3:
            return S.One
        # 如果参数索引不在支持的范围内，抛出参数索引错误
        else:
            raise ArgumentIndexError(self, argindex)

    # 将函数展开成简化形式
    def _eval_expand_func(self, **hints):
        # 调用内部函数 _fma，将当前对象的参数作为参数传递
        return _fma(*self.args)

    # 将函数重写为易处理的形式
    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        # 调用内部函数 _fma，将指定参数作为参数传递
        return _fma(arg)
# 设置一个变量 _Ten，表示数字 10
_Ten = S(10)


def _log10(x):
    # 计算以 10 为底的对数
    return log(x)/log(_Ten)


class log10(Function):
    """
    表示以 10 为底的对数函数。

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log10
    >>> log10(100).evalf() == 2.0
    True
    >>> log10(x).diff(x)
    1/(x*log(10))

    See Also
    ========

    log2
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        返回该函数的一阶导数。
        """
        if argindex == 1:
            return S.One/(log(_Ten)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            # 如果参数是数值，则计算以 10 为底的对数
            result = log.eval(arg, base=_Ten)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Ten:
            # 如果参数是以 10 为底的幂，则直接返回指数部分
            return arg.exp

    def _eval_expand_func(self, **hints):
        # 对函数进行展开处理
        return _log10(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        # 重写为普通对数的形式
        return _log10(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log


def _Sqrt(x):
    # 返回 x 的平方根
    return Pow(x, S.Half)


class Sqrt(Function):  # 'sqrt' already defined in sympy.functions.elementary.miscellaneous
    """
    表示平方根函数。

    Explanation
    ===========

    使用 ``Sqrt(x)`` 而不是 ``sqrt(x)`` 的原因是，后者在内部表示为 ``Pow(x, S.Half)``，
    这在进行代码生成时可能不是预期的行为。

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Sqrt
    >>> Sqrt(x)
    Sqrt(x)
    >>> Sqrt(x).diff(x)
    1/(2*sqrt(x))

    See Also
    ========

    Cbrt
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        返回该函数的一阶导数。
        """
        if argindex == 1:
            # 计算平方根函数的导数
            return Pow(self.args[0], Rational(-1, 2))/_Two
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        # 对函数进行展开处理
        return _Sqrt(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        # 重写为幂函数的形式
        return _Sqrt(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


def _Cbrt(x):
    # 返回 x 的立方根
    return Pow(x, Rational(1, 3))


class Cbrt(Function):  # 'cbrt' already defined in sympy.functions.elementary.miscellaneous
    """
    表示立方根函数。

    Explanation
    ===========

    使用 ``Cbrt(x)`` 而不是 ``cbrt(x)`` 的原因是，后者在内部表示为 ``Pow(x, Rational(1, 3))``，
    这在进行代码生成时可能不是预期的行为。

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Cbrt
    >>> Cbrt(x)
    Cbrt(x)
    >>> Cbrt(x).diff(x)
    1/(3*x**(2/3))

    See Also
    ========

    Sqrt
    """
    nargs = 1
    # 定义一个方法用于计算函数的一阶导数，接受一个参数 argindex 表示导数的变量索引，默认为 1
    def fdiff(self, argindex=1):
        # 如果 argindex 等于 1，返回函数参数列表中第一个参数的负三分之二次方再除以 3 的幂
        if argindex == 1:
            return Pow(self.args[0], Rational(-_Two/3))/3
        else:
            # 如果 argindex 不等于 1，抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    # 定义一个私有方法，用于展开函数，返回其参数的立方根
    def _eval_expand_func(self, **hints):
        return _Cbrt(*self.args)

    # 定义一个方法，将函数重写为 Pow 对象形式，返回参数的立方根
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _Cbrt(arg)

    # 将函数重写为可处理的形式，等同于 _eval_rewrite_as_Pow 方法
    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow
# 定义一个函数来计算两个数的直角三角形斜边长度
def _hypot(x, y):
    return sqrt(Pow(x, 2) + Pow(y, 2))

# hypot 类，表示直角三角形斜边长度函数
class hypot(Function):
    """
    Represents the hypotenuse function.

    Explanation
    ===========

    The hypotenuse function is provided by e.g. the math library
    in the C99 standard, hence one may want to represent the function
    symbolically when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.codegen.cfunctions import hypot
    >>> hypot(3, 4).evalf() == 5.0
    True
    >>> hypot(x, y)
    hypot(x, y)
    >>> hypot(x, y).diff(x)
    x/hypot(x, y)

    """
    nargs = 2

    # 返回函数的第一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return 2*self.args[argindex-1]/(_Two*self.func(*self.args))
        else:
            raise ArgumentIndexError(self, argindex)

    # 将函数展开为基本的计算形式
    def _eval_expand_func(self, **hints):
        return _hypot(*self.args)

    # 将函数重写为 Pow 函数的形式
    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _hypot(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


# isnan 类，表示检查一个数是否为 NaN（Not a Number）
class isnan(Function):
    nargs = 1
```