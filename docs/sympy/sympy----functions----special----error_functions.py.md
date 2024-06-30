# `D:\src\scipysrc\sympy\sympy\functions\special\error_functions.py`

```
# 导入必要的模块和函数
""" This module contains various functions that are special cases
    of incomplete gamma functions. It should probably be renamed. """
from sympy.core import EulerGamma # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, expand_mul
from sympy.core.numbers import I, pi, Rational
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg

# TODO series expansions
# TODO see the "Note:" in Ei

# Helper function
# 定义一个辅助函数，用于处理实数到实数作为实部和虚部的情况
def real_to_real_as_real_imag(self, deep=True, **hints):
    if self.args[0].is_extended_real:
        if deep:
            hints['complex'] = False
            return (self.expand(deep, **hints), S.Zero)
        else:
            return (self, S.Zero)
    if deep:
        x, y = self.args[0].expand(deep, **hints).as_real_imag()
    else:
        x, y = self.args[0].as_real_imag()
    re = (self.func(x + I*y) + self.func(x - I*y))/2
    im = (self.func(x + I*y) - self.func(x - I*y))/(2*I)
    return (re, im)


###############################################################################
################################ ERROR FUNCTION ###############################
###############################################################################

# 定义误差函数 erf
class erf(Function):
    r"""
    The Gauss error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \mathrm{d}t.

    Examples
    ========

    >>> from sympy import I, oo, erf
    >>> from sympy.abc import z

    Several special values are known:

    >>> erf(0)
    0
    >>> erf(oo)
    1
    >>> erf(-oo)
    -1
    >>> erf(I*oo)
    oo*I
    >>> erf(-I*oo)
    -oo*I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erf(-z)
    -erf(z)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf(z))
    erf(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erf(z), z)
    2*exp(-z**2)/sqrt(pi)

    We can numerically evaluate the error function to arbitrary precision
    on the whole complex plane:

    >>> erf(4).evalf(30)
    # 浮点数值，表示一个非常接近于1的浮点数

    >>> erf(-4*I).evalf(30)
    # 计算复数-4i的误差函数，并返回其精确到30位小数的值

    -1296959.73071763923152794095062*I
    # 返回的结果是一个虚部为-1296959.73071763923152794095062的复数

    See Also
    ========

    erfc: 补余误差函数。
    erfi: 虚部误差函数。
    erf2: 双参数误差函数。
    erfinv: 逆误差函数。
    erfcinv: 补余误差函数的逆。
    erf2inv: 双参数误差函数的逆。

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erf.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erf
    # 提供了误差函数相关的参考链接

    """

    unbranched = True
    # 设定 unbranched 变量为 True，表示该函数是无分支的

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(-self.args[0]**2)/sqrt(pi)
            # 返回该函数对第一个参数的偏导数

        else:
            raise ArgumentIndexError(self, argindex)
            # 如果参数索引不是1，抛出参数索引错误异常

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        返回该函数的反函数。
        """
        return erfinv
        # 返回误差函数的逆函数对象

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero
            # 处理特殊情况下的函数值

        if isinstance(arg, erfinv):
             return arg.args[0]
             # 如果参数是误差函数的逆函数对象，返回其第一个参数值

        if isinstance(arg, erfcinv):
            return S.One - arg.args[0]
            # 如果参数是补余误差函数的逆函数对象，返回1减去其第一个参数值

        if arg.is_zero:
            return S.Zero
            # 如果参数为零，返回零

        # Only happens with unevaluated erf2inv
        if isinstance(arg, erf2inv) and arg.args[0].is_zero:
            return arg.args[1]
            # 如果参数是未评估的双参数误差函数的逆函数对象，且其第一个参数是零，返回其第二个参数值

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return arg
            # 尝试从参数中提取因子I，如果参数是正无穷或负无穷，返回参数本身

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return -cls(-arg)
            # 尝试从参数中提取因子-1，返回相反数

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
            # 如果n小于0或n为偶数，返回零

        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
                # 使用泰勒级数计算该函数的n阶项

            else:
                return 2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))
                # 使用泰勒级数计算该函数的n阶项

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())
        # 返回该函数对参数的共轭函数

    def _eval_is_real(self):
        return self.args[0].is_extended_real
        # 返回参数是否为扩展实数

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True
        else:
            return self.args[0].is_extended_real
        # 返回参数是否为有限实数或扩展实数

    def _eval_is_zero(self):
        return self.args[0].is_zero
        # 返回参数是否为零

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))
        # 重写该函数为上伽玛函数的形式，返回重写后的表达式
    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        # 定义参数 arg 为 (1 - i)*z/sqrt(pi)
        arg = (S.One - I)*z/sqrt(pi)
        # 返回 (1 + i)*(fresnelc(arg) - i*fresnels(arg)) 的值作为重写的表达式
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        # 定义参数 arg 为 (1 - i)*z/sqrt(pi)
        arg = (S.One - I)*z/sqrt(pi)
        # 返回 (1 + i)*(fresnelc(arg) - i*fresnels(arg)) 的值作为重写的表达式
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        # 返回 z/sqrt(pi) * meijerg([1/2], [], [0], [-1/2], z**2) 作为重写的表达式
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 返回 2*z/sqrt(pi) * hyper([1/2], [3/2], -z**2) 作为重写的表达式
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # 返回 sqrt(z**2)/z - z*expint(1/2, z**2)/sqrt(pi) 作为重写的表达式
        return sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        # 导入符号极限计算函数
        from sympy.series.limits import limit
        # 如果 limitvar 被指定
        if limitvar:
            # 计算 z 关于 limitvar 在正无穷方向的极限
            lim = limit(z, limitvar, S.Infinity)
            # 如果极限是负无穷
            if lim is S.NegativeInfinity:
                # 返回 1 - _erfs(-z)*exp(-z**2) 作为重写的表达式
                return S.NegativeOne + _erfs(-z)*exp(-z**2)
        # 如果不满足上述条件，返回 1 - _erfs(z)*exp(-z**2) 作为重写的表达式
        return S.One - _erfs(z)*exp(-z**2)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        # 返回 1 - erfc(z) 作为重写的表达式
        return S.One - erfc(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        # 返回 -i*erfi(i*z) 作为重写的表达式
        return -I*erfi(I*z)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取表达式的主导项
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        # 计算在 x = 0 处的值
        arg0 = arg.subs(x, 0)

        # 如果在表达式中 x 是自由符号并且在 x = 0 处为零
        if x in arg.free_symbols and arg0.is_zero:
            # 返回 2*arg/sqrt(pi) 作为主导项的重写表达式
            return 2*arg/sqrt(pi)
        else:
            # 否则返回 self.func(arg0)
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        # 导入阶级别的 Order 对象
        from sympy.series.order import Order
        # 获取参数中的点
        point = args0[0]

        # 如果点是正无穷或负无穷
        if point in [S.Infinity, S.NegativeInfinity]:
            z = self.args[0]

            try:
                # 尝试获取 z 在 x 方向的主导项
                _, ex = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            # 将指数取反，因为在 x -> 1/x 时使用的是 aseries
            ex = -ex
            # 如果指数是正数
            if ex.is_positive:
                # 计算新的阶数
                newn = ceiling(n/ex)
                # 构建级数项列表 s
                s = [S.NegativeOne**k * factorial2(2*k - 1) / (z**(2*k + 1) * 2**k)
                     for k in range(newn)] + [Order(1/z**newn, x)]
                # 返回 1 - (exp(-z**2)/sqrt(pi)) * Add(*s) 作为级数展开的重写表达式
                return S.One - (exp(-z**2)/sqrt(pi)) * Add(*s)

        # 如果不满足上述条件，调用父类的 _eval_aseries 方法
        return super(erf, self)._eval_aseries(n, args0, x, logx)

    # 将实部到实部和虚部的转换
    as_real_imag = real_to_real_as_real_imag
class erfc(Function):
    r"""
    Complementary Error Function.

    Explanation
    ===========

    The function is defined as:

    .. math ::
        \mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfc
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfc(0)
    1
    >>> erfc(oo)
    0
    >>> erfc(-oo)
    2
    >>> erfc(I*oo)
    -oo*I
    >>> erfc(-I*oo)
    oo*I

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erfc(z))
    erfc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfc(z), z)
    -2*exp(-z**2)/sqrt(pi)

    It also follows

    >>> erfc(-z)
    2 - erfc(z)

    We can numerically evaluate the complementary error function to arbitrary
    precision on the whole complex plane:

    >>> erfc(4).evalf(30)
    0.0000000154172579002800188521596734869

    >>> erfc(4*I).evalf(30)
    1.0 - 1296959.73071763923152794095062*I

    See Also
    ========

    erf: Gaussian error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erfc.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erfc

    """

    unbranched = True  # 设置属性 `unbranched` 为 True，表示此函数是无分支的

    def fdiff(self, argindex=1):
        if argindex == 1:
            # 返回对自变量的偏导数，对于 erfc(x)，返回 -2*exp(-x**2)/sqrt(pi)
            return -2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfcinv  # 返回 erfcinv 函数作为此函数的逆函数

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg.is_zero:
                return S.One

        if isinstance(arg, erfinv):
            return S.One - arg.args[0]

        if isinstance(arg, erfcinv):
            return arg.args[0]

        if arg.is_zero:
            return S.One

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return -arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return 2 - cls(-arg)
    def taylor_term(n, x, *previous_terms):
        # 如果 n 等于 0，返回 1
        if n == 0:
            return S.One
        # 如果 n 小于 0 或者 n 是偶数，返回 0
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            # 将 x 转换为符号表达式
            x = sympify(x)
            # 计算 k，向下取整((n - 1)/2)
            k = floor((n - 1)/S(2))
            # 如果 previous_terms 中元素数大于 2，根据公式返回泰勒级数的当前项
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                # 否则，根据公式返回泰勒级数的当前项
                return -2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        # 返回对象的共轭值
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        # 返回对象的实部是否为扩展实数
        return self.args[0].is_extended_real

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        # 将对象重写为 tractable 形式，使用 erf 函数进行深度重写
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        # 将对象重写为与 erf 函数等价的形式
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        # 将对象重写为与 erfi 函数等价的形式
        return S.One + I*erfi(I*z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        # 将对象重写为与 fresnelc 和 fresnels 函数等价的形式
        arg = (S.One - I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        # 将对象重写为与 fresnelc 和 fresnels 函数等价的形式
        arg = (S.One-I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        # 将对象重写为与 meijerg 函数等价的形式
        return S.One - z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 将对象重写为与 hyper 函数等价的形式
        return S.One - 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        # 将对象重写为与 uppergamma 函数等价的形式
        from sympy.functions.special.gamma_functions import uppergamma
        return S.One - sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # 将对象重写为与 expint 函数等价的形式
        return S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        # 展开对象，使用 erf 函数进行重写
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回对象的主导项，如果参数是无穷大，则限制 x 接近 0
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        # 如果 arg0 是无穷大，根据方向进行极限计算
        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        # 如果 arg0 是零，则返回 1
        if arg0.is_zero:
            return S.One
        else:
            # 否则返回对象的函数值
            return self.func(arg0)

    as_real_imag = real_to_real_as_real_imag

    def _eval_aseries(self, n, args0, x, logx):
        # 返回对象的渐近级数展开
        return S.One - erf(*self.args)._eval_aseries(n, args0, x, logx)
class erfi(Function):
    r"""
    Imaginary error function.

    Explanation
    ===========

    The function erfi is defined as:

    .. math ::
        \mathrm{erfi}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfi
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfi(0)
    0
    >>> erfi(oo)
    oo
    >>> erfi(-oo)
    -oo
    >>> erfi(I*oo)
    I
    >>> erfi(-I*oo)
    -I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erfi(-z)
    -erfi(z)

    >>> from sympy import conjugate
    >>> conjugate(erfi(z))
    erfi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfi(z), z)
    2*exp(z**2)/sqrt(pi)

    We can numerically evaluate the imaginary error function to arbitrary
    precision on the whole complex plane:

    >>> erfi(2).evalf(30)
    18.5648024145755525987042919132

    >>> erfi(-2*I).evalf(30)
    -0.995322265018952734162069256367*I

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://mathworld.wolfram.com/Erfi.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/Erfi

    """

    unbranched = True

    def fdiff(self, argindex=1):
        # 检查参数索引是否为1，如果是则返回函数对参数的偏导数
        if argindex == 1:
            return 2*exp(self.args[0]**2)/sqrt(pi)
        else:
            # 抛出异常，指示参数索引不正确
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, z):
        # 如果参数 z 是一个数值
        if z.is_Number:
            # 处理特殊值
            if z is S.NaN:
                return S.NaN
            elif z.is_zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity

        # 如果参数 z 是零
        if z.is_zero:
            return S.Zero

        # 尝试提取 -1 的因子
        if z.could_extract_minus_sign():
            return -cls(-z)

        # 尝试提取 I 的因子
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I*nz.args[0]
            if isinstance(nz, erfcinv):
                return I*(S.One - nz.args[0])
            # 只有在未评估的 erf2inv 情况下才会发生
            if isinstance(nz, erf2inv) and nz.args[0].is_zero:
                return I*nz.args[1]

    @staticmethod
    @cacheit
    # 计算泰勒级数的单个项
    def taylor_term(n, x, *previous_terms):
        # 如果 n 小于 0 或者 n 是偶数，则返回零
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            # 将 x 转换为 sympy 符号
            x = sympify(x)
            # 计算 k 的值，其中 k 是 (n-1)/2 的整数部分
            k = floor((n - 1)/S(2))
            # 如果传入的前几项数量大于2，则返回基于前两项的递推公式计算的当前项
            if len(previous_terms) > 2:
                return previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                # 否则返回 x 的 n 次方的泰勒级数表达式的结果
                return 2 * x**n/(n*factorial(k)*sqrt(pi))

    # 返回共轭复数
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 判断是否是扩展实数
    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    # 判断是否是零
    def _eval_is_zero(self):
        return self.args[0].is_zero

    # 重写为更易处理形式
    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    # 重写为 erf 的形式
    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I*erf(I*z)

    # 重写为 erfc 的形式
    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I*erfc(I*z) - I

    # 重写为 Fresnel S 函数的形式
    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    # 重写为 Fresnel C 函数的形式
    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    # 重写为 Meijer G 函数的形式
    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)

    # 重写为超函Hypergeometric1F1函数的形式
    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], z**2)

    # 重写为上gamma函数的形式
    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(pi) - S.One)

    # 重写为指数积分函数的形式
    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(pi)

    # 展开函数
    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    # 将实部和虚部函数转换为实部
    as_real_imag = real_to_real_as_real_imag

    # 求导时,将变量看成领导数
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获得主要的arg值
        sowie geht ou on a der ze tea H
class erf2(Function):
    r"""
    Two-argument error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf2}(x, y) = \frac{2}{\sqrt{\pi}} \int_x^y e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import oo, erf2
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2(0, 0)
    0
    >>> erf2(x, x)
    0
    >>> erf2(x, oo)
    1 - erf(x)
    >>> erf2(x, -oo)
    -erf(x) - 1
    >>> erf2(oo, y)
    erf(y) - 1
    >>> erf2(-oo, y)
    erf(y) + 1

    In general one can pull out factors of -1:

    >>> erf2(-x, -y)
    -erf2(x, y)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf2(x, y))
    erf2(conjugate(x), conjugate(y))

    Differentiation with respect to $x$, $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2(x, y), x)
    -2*exp(-x**2)/sqrt(pi)
    >>> diff(erf2(x, y), y)
    2*exp(-y**2)/sqrt(pi)

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/Erf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            # Return the partial derivative of erf2 with respect to x
            return -2*exp(-x**2)/sqrt(pi)
        elif argindex == 2:
            # Return the partial derivative of erf2 with respect to y
            return 2*exp(-y**2)/sqrt(pi)
        else:
            # Raise an error if the argument index is not 1 or 2
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        chk = (S.Infinity, S.NegativeInfinity, S.Zero)
        if x is S.NaN or y is S.NaN:
            # Return NaN if either x or y is NaN
            return S.NaN
        elif x == y:
            # Return 0 if x equals y
            return S.Zero
        elif x in chk or y in chk:
            # Return the value of erf(y) - erf(x) for special cases of x and y
            return erf(y) - erf(x)

        if isinstance(y, erf2inv) and y.args[0] == x:
            # Return y.args[1] if y is an instance of erf2inv and its first argument matches x
            return y.args[1]

        if x.is_zero or y.is_zero or x.is_extended_real and x.is_infinite or \
                y.is_extended_real and y.is_infinite:
            # Return the value of erf(y) - erf(x) for other specific cases of x and y
            return erf(y) - erf(x)

        # Try to pull out -1 factor
        sign_x = x.could_extract_minus_sign()
        sign_y = y.could_extract_minus_sign()
        if (sign_x and sign_y):
            # Return -erf2(x, y) if both x and y can have a minus sign extracted
            return -cls(-x, -y)
        elif (sign_x or sign_y):
            # Return erf(y) - erf(x) if only one of x or y can have a minus sign extracted
            return erf(y) - erf(x)

    def _eval_conjugate(self):
        # Return the conjugate of erf2(x, y)
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_is_extended_real(self):
        # Check if both arguments x and y are extended real numbers
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_rewrite_as_erf(self, x, y, **kwargs):
        # Rewrite erf2(x, y) in terms of erf
        return erf(y) - erf(x)

    def _eval_rewrite_as_erfc(self, x, y, **kwargs):
        # Rewrite erf2(x, y) in terms of erfc
        return erfc(x) - erfc(y)

    def _eval_rewrite_as_erfi(self, x, y, **kwargs):
        # Rewrite erf2(x, y) in terms of erfi
        return I*(erfi(I*x)-erfi(I*y))
    # 将表达式重写为以 Fresnel S 函数表示
    def _eval_rewrite_as_fresnels(self, x, y, **kwargs):
        return erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)

    # 将表达式重写为以 Fresnel C 函数表示
    def _eval_rewrite_as_fresnelc(self, x, y, **kwargs):
        return erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)

    # 将表达式重写为以 Meijer G 函数表示
    def _eval_rewrite_as_meijerg(self, x, y, **kwargs):
        return erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)

    # 将表达式重写为以超几何函数表示
    def _eval_rewrite_as_hyper(self, x, y, **kwargs):
        return erf(y).rewrite(hyper) - erf(x).rewrite(hyper)

    # 将表达式重写为以上半伽玛函数表示
    def _eval_rewrite_as_uppergamma(self, x, y, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (sqrt(y**2)/y*(S.One - uppergamma(S.Half, y**2)/sqrt(pi)) -
            sqrt(x**2)/x*(S.One - uppergamma(S.Half, x**2)/sqrt(pi)))

    # 将表达式重写为以指数积分函数表示
    def _eval_rewrite_as_expint(self, x, y, **kwargs):
        return erf(y).rewrite(expint) - erf(x).rewrite(expint)

    # 扩展函数，以误差函数重写
    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    # 判断表达式是否为零
    def _eval_is_zero(self):
        return is_eq(*self.args)
# 定义一个名为 erfinv 的类，表示误差函数的逆函数
class erfinv(Function):
    r"""
    Inverse Error Function. The erfinv function is defined as:

    .. math ::
        \mathrm{erf}(x) = y \quad \Rightarrow \quad \mathrm{erfinv}(y) = x

    Examples
    ========

    >>> from sympy import erfinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfinv(0)
    0
    >>> erfinv(1)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfinv(x), x)
    sqrt(pi)*exp(erfinv(x)**2)/2

    We can numerically evaluate the inverse error function to arbitrary
    precision on [-1, 1]:

    >>> erfinv(0.2).evalf(30)
    0.179143454621291692285822705344

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErf/

    """

    # 定义 fdiff 方法，用于计算偏导数
    def fdiff(self, argindex=1):
        # 如果参数索引为 1
        if argindex == 1:
            # 返回 sqrt(pi)*exp(erfinv(x)**2)*S.Half
            return sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else:
            # 否则引发 ArgumentIndexError 异常
            raise ArgumentIndexError(self, argindex)

    # 定义 inverse 方法，返回这个函数的逆函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erf

    # 类方法，用于评估函数的特定值 z
    @classmethod
    def eval(cls, z):
        # 如果 z 是 NaN，返回 NaN
        if z is S.NaN:
            return S.NaN
        # 如果 z 是 -1，返回负无穷
        elif z is S.NegativeOne:
            return S.NegativeInfinity
        # 如果 z 是 0，返回 0
        elif z.is_zero:
            return S.Zero
        # 如果 z 是 1，返回正无穷
        elif z is S.One:
            return S.Infinity

        # 如果 z 是 erf 类型且其参数是扩展实数
        if isinstance(z, erf) and z.args[0].is_extended_real:
            # 返回 z 的参数
            return z.args[0]

        # 如果 z 是 0，返回 0
        if z.is_zero:
            return S.Zero

        # 尝试提取 -1 的因子
        nz = z.extract_multiplicatively(-1)
        # 如果成功提取且 nz 是 erf 类型且其参数是扩展实数
        if nz is not None and (isinstance(nz, erf) and (nz.args[0]).is_extended_real):
            # 返回 -nz 的参数
            return -nz.args[0]

    # 重写方法，将函数重写为 erfcinv(1-z)
    def _eval_rewrite_as_erfcinv(self, z, **kwargs):
       return erfcinv(1-z)

    # 判断函数是否为零的方法
    def _eval_is_zero(self):
        return self.args[0].is_zero


# 定义一个名为 erfcinv 的类，表示互补误差函数的逆函数
class erfcinv(Function):
    r"""
    Inverse Complementary Error Function. The erfcinv function is defined as:

    .. math ::
        \mathrm{erfc}(x) = y \quad \Rightarrow \quad \mathrm{erfcinv}(y) = x

    Examples
    ========

    >>> from sympy import erfcinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfcinv(1)
    0
    >>> erfcinv(0)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfcinv(x), x)
    -sqrt(pi)*exp(erfcinv(x)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.

    """
    # erf2inv: 反向的二参数误差函数。

    # References
    # ==========

    # [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    # [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/
    
    """
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/

    """

    # fdiff 方法用于计算函数的偏导数
    def fdiff(self, argindex=1):
        # 如果参数索引是 1，则返回相应的偏导数表达式
        if argindex == 1:
            return -sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else:
            # 否则抛出参数索引错误异常
            raise ArgumentIndexError(self, argindex)

    # inverse 方法用于返回该函数的反函数
    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        # 这里应该返回 erfc 函数，但代码中未给出具体实现

    # eval 类方法用于评估函数在给定点 z 处的值
    @classmethod
    def eval(cls, z):
        # 如果 z 是 NaN，则返回 NaN
        if z is S.NaN:
            return S.NaN
        # 如果 z 是 0，则返回正无穷
        elif z.is_zero:
            return S.Infinity
        # 如果 z 是 1，则返回 0
        elif z is S.One:
            return S.Zero
        # 如果 z 等于 2，则返回负无穷
        elif z == 2:
            return S.NegativeInfinity

        # 如果 z 是 0，则返回正无穷（此处可能存在逻辑错误，应进一步验证）

    # _eval_rewrite_as_erfinv 方法用于将函数重写为 erfinv 函数的形式
    def _eval_rewrite_as_erfinv(self, z, **kwargs):
        return erfinv(1-z)

    # _eval_is_zero 方法用于评估函数是否在其参数减去 1 后为零
    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    # _eval_is_infinite 方法用于评估函数的参数是否为零，从而使函数结果为无穷
    def _eval_is_infinite(self):
        return self.args[0].is_zero
class erf2inv(Function):
    r"""
    Two-argument Inverse error function. The erf2inv function is defined as:

    .. math ::
        \mathrm{erf2}(x, w) = y \quad \Rightarrow \quad \mathrm{erf2inv}(x, y) = w

    Examples
    ========

    >>> from sympy import erf2inv, oo
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2inv(0, 0)
    0
    >>> erf2inv(1, 0)
    1
    >>> erf2inv(0, 1)
    oo
    >>> erf2inv(0, y)
    erfinv(y)
    >>> erf2inv(oo, y)
    erfcinv(-y)

    Differentiation with respect to $x$ and $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2inv(x, y), x)
    exp(-x**2 + erf2inv(x, y)**2)
    >>> diff(erf2inv(x, y), y)
    sqrt(pi)*exp(erf2inv(x, y)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse complementary error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/InverseErf2/

    """

    # 定义类方法 fdiff，用于对 erf2inv 函数进行偏导数计算
    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return exp(self.func(x,y)**2-x**2)
        elif argindex == 2:
            return sqrt(pi)*S.Half*exp(self.func(x,y)**2)
        else:
            raise ArgumentIndexError(self, argindex)

    # 定义类方法 eval，用于计算 erf2inv 函数的特定输入情况下的返回值
    @classmethod
    def eval(cls, x, y):
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x.is_zero and y.is_zero:
            return S.Zero
        elif x.is_zero and y is S.One:
            return S.Infinity
        elif x is S.One and y.is_zero:
            return S.One
        elif x.is_zero:
            return erfinv(y)
        elif x is S.Infinity:
            return erfcinv(-y)
        elif y.is_zero:
            return x
        elif y is S.Infinity:
            return erfinv(x)

        if x.is_zero:
            if y.is_zero:
                return S.Zero
            else:
                return erfinv(y)
        if y.is_zero:
            return x

    # 定义私有方法 _eval_is_zero，用于检查 erf2inv 函数是否在特定情况下返回零
    def _eval_is_zero(self):
        x, y = self.args
        if x.is_zero and y.is_zero:
            return True

###############################################################################
#################### EXPONENTIAL INTEGRALS ####################################
###############################################################################

class Ei(Function):
    r"""
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Ei}(x) = \sum_{n=1}^\infty \frac{x^n}{n\, n!}
                                     + \log(x) + \gamma,

    where $\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\mathbb{C} \setminus (-\infty, 0]$.
    @classmethod
    # 类方法，用于计算指定参数 z 的指数积分 Ei(z)
    def eval(cls, z):
        # 如果 z 是零，返回负无穷
        if z.is_zero:
            return S.NegativeInfinity
        # 如果 z 是正无穷，返回正无穷
        elif z is S.Infinity:
            return S.Infinity
        # 如果 z 是负无穷，返回零
        elif z is S.NegativeInfinity:
            return S.Zero

        # 如果 z 是零，返回负无穷（此处重复了，可能是错误，因为已在前面处理过）
        if z.is_zero:
            return S.NegativeInfinity

        # 提取 z 的分支因子和主值
        nz, n = z.extract_branch_factor()
        # 如果存在分支因子 n，则返回 Ei(nz) + 2*I*pi*n
        if n:
            return Ei(nz) + 2*I*pi*n

    # 对象的偏导数函数
    def fdiff(self, argindex=1):
        # 获取第一个参数的主值
        arg = unpolarify(self.args[0])
        # 如果 argindex 等于 1，返回 exp(arg)/arg
        if argindex == 1:
            return exp(arg)/arg
        # 否则抛出参数索引错误
        else:
            raise ArgumentIndexError(self, argindex)

    # 用于数值求解的内部函数
    def _eval_evalf(self, prec):
        # 如果 self.args[0] / polar_lift(-1) 是正数
        if (self.args[0]/polar_lift(-1)).is_positive:
            # 返回函数的数值求解 + (I*pi) 的数值求解
            return Function._eval_evalf(self, prec) + (I*pi)._eval_evalf(prec)
        # 否则只返回函数的数值求解
        return Function._eval_evalf(self, prec)

    # 将函数重写为上不完全伽马函数的形式
    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        # 这段代码目前不实用，因为 uppergamma 立即转换为 expint
        return -uppergamma(0, polar_lift(-1)*z) - I*pi

    # 将函数重写为指数积分的形式
    def _eval_rewrite_as_expint(self, z, **kwargs):
        # 返回 -expint(1, polar_lift(-1)*z) - I*pi
        return -expint(1, polar_lift(-1)*z) - I*pi
    def _eval_rewrite_as_li(self, z, **kwargs):
        # 如果 z 是对数函数 log 的实例，则返回参数的指数函数的对数积分 li
        if isinstance(z, log):
            return li(z.args[0])
        # 否则，根据 Euler 常数和虚部的范围，返回参数的指数函数的对数积分 li
        # 对于 -pi < imag(z) <= pi 的情况
        # 实际上只有：
        #  Ei(z) = li(exp(z))
        return li(exp(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        # 如果 z 是负数，则返回参数的正弦积分 Shi 和余弦积分 Chi 之差再减去虚数单位乘以 pi
        if z.is_negative:
            return Shi(z) + Chi(z) - I*pi
        # 否则，返回参数的正弦积分 Shi 和余弦积分 Chi 的和
        else:
            return Shi(z) + Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        # 返回参数的指数函数 exp(z) 乘以特殊函数 _eis(z) 的结果
        return exp(z) * _eis(z)

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        # 导入积分模块
        from sympy.integrals.integrals import Integral
        # 创建一个虚拟符号 t，确保它在当前环境中是唯一的
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        # 返回指数函数 S.Exp1**t 除以 t 的积分，积分上下限为从负无穷到 z
        return Integral(S.Exp1**t/t, (t, S.NegativeInfinity, z))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入 re 函数
        from sympy import re
        # 计算参数的极限 x0
        x0 = self.args[0].limit(x, 0)
        # 计算参数的主导项 arg
        arg = self.args[0].as_leading_term(x, cdir=cdir)
        # 计算参数的方向 cdir
        cdir = arg.dir(x, cdir)
        # 如果 x0 是零
        if x0.is_zero:
            # 将参数 arg 分解为系数和指数
            c, e = arg.as_coeff_exponent(x)
            # 如果 logx 是空，则设置为 log(x)
            logx = log(x) if logx is None else logx
            # 返回对数项 log(c) + e*logx 加上 EulerGamma 减去 I*pi（如果 re(cdir) 是负数）
            return log(c) + e*logx + EulerGamma - (
                I*pi if re(cdir).is_negative else S.Zero)
        # 否则，调用超类的 _eval_as_leading_term 方法
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 计算参数的极限 x0
        x0 = self.args[0].limit(x, 0)
        # 如果 x0 是零
        if x0.is_zero:
            # 将参数重新表达为正弦积分的形式，并返回其在 x 点的 n 级数展开
            f = self._eval_rewrite_as_Si(*self.args)
            return f._eval_nseries(x, n, logx)
        # 否则，调用超类的 _eval_nseries 方法
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        # 导入级数的 Order 类
        from sympy.series.order import Order
        # 取参数的第一个点
        point = args0[0]

        # 如果 point 是无穷大
        if point is S.Infinity:
            # 取参数的第一个元素
            z = self.args[0]
            # 计算级数的前 n 项
            s = [factorial(k) / (z)**k for k in range(n)] + \
                    [Order(1/z**n, x)]
            # 返回指数函数 exp(z) 除以 z 乘以所有项的和
            return (exp(z)/z) * Add(*s)

        # 否则，调用超类 Ei 的 _eval_aseries 方法
        return super(Ei, self)._eval_aseries(n, args0, x, logx)
class expint(Function):
    r"""
    Generalized exponential integral.

    Explanation
    ===========

    This function is defined as

    .. math:: \operatorname{E}_\nu(z) = z^{\nu - 1} \Gamma(1 - \nu, z),

    where $\Gamma(1 - \nu, z)$ is the upper incomplete gamma function
    (``uppergamma``).

    Hence for $z$ with positive real part we have

    .. math:: \operatorname{E}_\nu(z)
              =   \int_1^\infty \frac{e^{-zt}}{t^\nu} \mathrm{d}t,

    which explains the name.

    The representation as an incomplete gamma function provides an analytic
    continuation for $\operatorname{E}_\nu(z)$. If $\nu$ is a
    non-positive integer, the exponential integral is thus an unbranched
    function of $z$, otherwise there is a branch point at the origin.
    Refer to the incomplete gamma function documentation for details of the
    branching behavior.

    Examples
    ========

    >>> from sympy import expint, S
    >>> from sympy.abc import nu, z

    Differentiation is supported. Differentiation with respect to $z$ further
    explains the name: for integral orders, the exponential integral is an
    iterated integral of the exponential function.

    >>> expint(nu, z).diff(z)
    -expint(nu - 1, z)

    Differentiation with respect to $\nu$ has no classical expression:

    >>> expint(nu, z).diff(nu)
    -z**(nu - 1)*meijerg(((), (1, 1)), ((0, 0, 1 - nu), ()), z)

    At non-postive integer orders, the exponential integral reduces to the
    exponential function:

    >>> expint(0, z)
    exp(-z)/z
    >>> expint(-1, z)
    exp(-z)/z + exp(-z)/z**2

    At half-integers it reduces to error functions:

    >>> expint(S(1)/2, z)
    sqrt(pi)*erfc(sqrt(z))/sqrt(z)

    At positive integer orders it can be rewritten in terms of exponentials
    and ``expint(1, z)``. Use ``expand_func()`` to do this:

    >>> from sympy import expand_func
    >>> expand_func(expint(5, z))
    z**4*expint(1, z)/24 + (-z**3 + z**2 - 2*z + 6)*exp(-z)/24

    The generalised exponential integral is essentially equivalent to the
    incomplete gamma function:

    >>> from sympy import uppergamma
    >>> expint(nu, z).rewrite(uppergamma)
    z**(nu - 1)*uppergamma(1 - nu, z)

    As such it is branched at the origin:

    >>> from sympy import exp_polar, pi, I
    >>> expint(4, z*exp_polar(2*pi*I))
    I*pi*z**3/3 + expint(4, z)
    >>> expint(nu, z*exp_polar(2*pi*I))
    z**(nu - 1)*(exp(2*I*pi*nu) - 1)*gamma(1 - nu) + expint(nu, z)

    See Also
    ========

    Ei: Another related function called exponential integral.
    E1: The classical case, returns expint(1, z).
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma

    References
    ==========

    .. [1] https://dlmf.nist.gov/8.19
    .. [2] https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/

    """
    """
    Evaluate the Meijer G-function \( G_{1, 2}^{2, 2} \) for given parameters \( \nu \) and \( z \).

    This class method computes the Meijer G-function using various special function identities and series expansions.

    Args:
        nu: Parameter \( \nu \) of the Meijer G-function.
        z: Parameter \( z \) of the Meijer G-function.

    Returns:
        The evaluated Meijer G-function \( G_{1, 2}^{2, 2}(\nu, z) \).

    See Also:
        [1] https://en.wikipedia.org/wiki/Meijer_G-function
        [2] https://dlmf.nist.gov/8.19.E7

    """

    @classmethod
    def eval(cls, nu, z):
        # Import necessary functions from SymPy gamma functions module
        from sympy.functions.special.gamma_functions import (gamma, uppergamma)
        
        # Simplify nu using unpolarify function
        nu2 = unpolarify(nu)
        
        # If nu has changed after unpolarify, recursively call eval with nu2
        if nu != nu2:
            return expint(nu2, z)
        
        # Check conditions for special cases and return corresponding evaluations
        if nu.is_Integer and nu <= 0 or (not nu.is_Integer and (2*nu).is_Integer):
            return unpolarify(expand_mul(z**(nu - 1)*uppergamma(1 - nu, z)))
        
        # Extract branching information for z
        z, n = z.extract_branch_factor()
        
        # Return None if no branch factor exists
        if n is S.Zero:
            return
        
        # Handle cases based on whether nu is an integer or not
        if nu.is_integer:
            if not nu > 0:
                return
            # Evaluate the Meijer G-function using specific formula
            return expint(nu, z) \
                - 2*pi*I*n*S.NegativeOne**(nu - 1)/factorial(nu - 1)*unpolarify(z)**(nu - 1)
        else:
            # Evaluate using the general formula involving exponential and gamma functions
            return (exp(2*I*pi*nu*n) - 1)*z**(nu - 1)*gamma(1 - nu) + expint(nu, z)

    # Differentiate the Meijer G-function with respect to its arguments
    def fdiff(self, argindex):
        nu, z = self.args
        if argindex == 1:
            return -z**(nu - 1)*meijerg([], [1, 1], [0, 0, 1 - nu], [], z)
        elif argindex == 2:
            return -expint(nu - 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    # Rewrite the Meijer G-function in terms of upper incomplete gamma function
    def _eval_rewrite_as_uppergamma(self, nu, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return z**(nu - 1)*uppergamma(1 - nu, z)

    # Rewrite the Meijer G-function in terms of exponential integral function
    def _eval_rewrite_as_Ei(self, nu, z, **kwargs):
        if nu == 1:
            return -Ei(z*exp_polar(-I*pi)) - I*pi
        elif nu.is_Integer and nu > 1:
            # Evaluate using DLMF 8.19.7 formula
            x = -unpolarify(z)
            return x**(nu - 1)/factorial(nu - 1)*E1(z).rewrite(Ei) + \
                exp(x)/factorial(nu - 1) * \
                Add(*[factorial(nu - k - 2)*x**k for k in range(nu - 1)])
        else:
            return self

    # Expand the Meijer G-function using its representations
    def _eval_expand_func(self, **hints):
        return self.rewrite(Ei).rewrite(expint, **hints)

    # Rewrite the Meijer G-function in terms of sine and cosine integral functions
    def _eval_rewrite_as_Si(self, nu, z, **kwargs):
        if nu != 1:
            return self
        return Shi(z) - Chi(z)
    
    # Additional rewrite functions for other special functions
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    # Compute the n-series expansion of the Meijer G-function
    def _eval_nseries(self, x, n, logx, cdir=0):
        # Check if nu is independent of x
        if not self.args[0].has(x):
            nu = self.args[0]
            # Handle specific cases for nu = 1 and integer nu > 1
            if nu == 1:
                f = self._eval_rewrite_as_Si(*self.args)
                return f._eval_nseries(x, n, logx)
            elif nu.is_Integer and nu > 1:
                f = self._eval_rewrite_as_Ei(*self.args)
                return f._eval_nseries(x, n, logx)
        # Use default n-series expansion method if nu depends on x
        return super()._eval_nseries(x, n, logx)
    # 定义一个方法用于计算级数表达式，接受参数 n、args0、x、logx
    def _eval_aseries(self, n, args0, x, logx):
        # 导入 Order 对象用于表示级数展开的余项
        from sympy.series.order import Order
        # 获取参数列表中的第二个元素作为展开点
        point = args0[1]
        # 获取类的第一个参数作为 nu
        nu = self.args[0]

        # 如果展开点是正无穷
        if point is S.Infinity:
            # 获取类的第二个参数作为 z
            z = self.args[1]
            # 构建级数表达式的列表 s，包括 RisingFactorial(nu, k) / z**k 的项和一个 1/z**n 的 Order 项
            s = [S.NegativeOne**k * RisingFactorial(nu, k) / z**k for k in range(n)] + [Order(1/z**n, x)]
            # 返回级数表达式的计算结果
            return (exp(-z)/z) * Add(*s)

        # 如果展开点不是正无穷，则调用父类的 _eval_aseries 方法进行处理
        return super(expint, self)._eval_aseries(n, args0, x, logx)

    # 定义一个方法用于重写为积分形式，接受任意数量的位置参数和关键字参数
    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        # 导入 Integral 对象用于表示积分
        from sympy.integrals.integrals import Integral
        # 获取类的第一个参数作为 n，第二个参数作为 x
        n, x = self.args
        # 创建一个虚拟符号 t 作为积分变量，确保其名字是唯一的
        t = Dummy(uniquely_named_symbol('t', args).name)
        # 返回积分表达式，积分变量是 t，积分范围是从 1 到正无穷，被积函数是 t**-n * exp(-t*x)
        return Integral(t**-n * exp(-t*x), (t, 1, S.Infinity))
def E1(z):
    """
    Classical case of the generalized exponential integral.

    Explanation
    ===========

    This is equivalent to ``expint(1, z)``.

    Examples
    ========

    >>> from sympy import E1
    >>> E1(0)
    expint(1, 0)

    >>> E1(5)
    expint(1, 5)

    See Also
    ========

    Ei: Exponential integral.
    expint: Generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    """
    # 返回对应参数 z 的广义指数积分 expint(1, z)
    return expint(1, z)


class li(Function):
    r"""
    The classical logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{li}(x) = \int_0^x \frac{1}{\log(t)} \mathrm{d}t \,.

    Examples
    ========

    >>> from sympy import I, oo, li
    >>> from sympy.abc import z

    Several special values are known:

    >>> li(0)
    0
    >>> li(1)
    -oo
    >>> li(oo)
    oo

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(li(z), z)
    1/log(z)

    Defining the ``li`` function via an integral:
    >>> from sympy import integrate
    >>> integrate(li(z))
    z*li(z) - Ei(2*log(z))

    >>> integrate(li(z),z)
    z*li(z) - Ei(2*log(z))


    The logarithmic integral can also be defined in terms of ``Ei``:

    >>> from sympy import Ei
    >>> li(z).rewrite(Ei)
    Ei(log(z))
    >>> diff(li(z).rewrite(Ei), z)
    1/log(z)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> li(2).evalf(30)
    1.04516378011749278484458888919

    >>> li(2*I).evalf(30)
    1.0652795784357498247001125598 + 3.08346052231061726610939702133*I

    We can even compute Soldner's constant by the help of mpmath:

    >>> from mpmath import findroot
    >>> findroot(li, 2)
    1.45136923488338

    Further transformations include rewriting ``li`` in terms of
    the trigonometric integrals ``Si``, ``Ci``, ``Shi`` and ``Chi``:

    >>> from sympy import Si, Ci, Shi, Chi
    >>> li(z).rewrite(Si)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Ci)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Shi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))
    >>> li(z).rewrite(Chi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))

    See Also
    ========

    Li: Offset logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral

    """
    """
    @classmethod
    def eval(cls, z):
        如果 z 是零，返回 S.Zero
        如果 z 是 S.One，返回 S.NegativeInfinity
        如果 z 是 S.Infinity，返回 S.Infinity
        如果 z 是零，返回 S.Zero

    def fdiff(self, argindex=1):
        获取参数列表中的第一个参数
        如果 argindex 等于 1，返回 S.One 除以参数的自然对数
        否则，引发 ArgumentIndexError 异常，指明参数和索引

    def _eval_conjugate(self):
        获取参数列表中的第一个参数
        如果参数不在复数平面的负半轴上，返回其共轭值

    def _eval_rewrite_as_Li(self, z, **kwargs):
        返回 Li(z) + li(2) 的表达式

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        返回 Ei(log(z)) 的表达式

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        导入上不完全伽玛函数 uppergamma
        返回 (-uppergamma(0, -log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) - log(-log(z))) 的表达式

    def _eval_rewrite_as_Si(self, z, **kwargs):
        返回 (Ci(I*log(z)) - I*Si(I*log(z)) -
                S.Half*(log(S.One/log(z)) - log(log(z))) - log(I*log(z))) 的表达式
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si

    def _eval_rewrite_as_Shi(self, z, **kwargs):
        返回 (Chi(log(z)) - Shi(log(z)) - S.Half*(log(S.One/log(z)) - log(log(z)))) 的表达式
    _eval_rewrite_as_Chi = _eval_rewrite_as_Shi

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        返回 (log(z)*hyper((1, 1), (2, 2), log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) + EulerGamma) 的表达式

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        返回 (-log(-log(z)) - S.Half*(log(S.One/log(z)) - log(log(z)))
                - meijerg(((), (1,)), ((0, 0), ()), -log(z))) 的表达式

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        返回 z * _eis(log(z)) 的表达式

    def _eval_nseries(self, x, n, logx, cdir=0):
        获取参数列表中的第一个参数
        构造一个级数 s，包含 (log(z))^k / (k * k!)，其中 k 从 1 到 n-1
        返回 EulerGamma + log(log(z)) + Add(*s) 的表达式

    def _eval_is_zero(self):
        获取参数列表中的第一个参数
        如果参数为零，返回 True
    """
class Li(Function):
    r"""
    The offset logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Li}(x) = \operatorname{li}(x) - \operatorname{li}(2)

    Examples
    ========

    >>> from sympy import Li
    >>> from sympy.abc import z

    The following special value is known:

    >>> Li(2)
    0

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(Li(z), z)
    1/log(z)

    The shifted logarithmic integral can be written in terms of $li(z)$:

    >>> from sympy import li
    >>> Li(z).rewrite(li)
    li(z) - li(2)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> Li(2).evalf(30)
    0

    >>> Li(4).evalf(30)
    1.92242131492155809316615998938

    See Also
    ========

    li: Logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6

    """


    @classmethod
    # 定义 eval 方法，用于计算特定输入 z 的值
    def eval(cls, z):
        # 若 z 是正无穷，则返回正无穷
        if z is S.Infinity:
            return S.Infinity
        # 若 z 等于 2，则返回 0
        elif z == S(2):
            return S.Zero

    # 定义 fdiff 方法，用于对对象进行偏导数计算
    def fdiff(self, argindex=1):
        # 获取函数参数
        arg = self.args[0]
        # 如果 argindex 等于 1，返回关于参数的对数导数
        if argindex == 1:
            return S.One / log(arg)
        # 否则抛出参数索引错误
        else:
            raise ArgumentIndexError(self, argindex)

    # 定义 _eval_evalf 方法，用于计算对象的数值近似
    def _eval_evalf(self, prec):
        # 将对象重写为 li(z) 形式，并计算数值近似
        return self.rewrite(li).evalf(prec)

    # 定义 _eval_rewrite_as_li 方法，将对象重写为关于 li(z) 的表达式
    def _eval_rewrite_as_li(self, z, **kwargs):
        return li(z) - li(2)

    # 定义 _eval_rewrite_as_tractable 方法，将对象重写为“可处理”的形式
    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        # 将对象重写为 li(z) 形式，并深度重写为“可处理”的形式
        return self.rewrite(li).rewrite("tractable", deep=True)

    # 定义 _eval_nseries 方法，用于计算对象的数值级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 将对象重写为 li(z) 形式，并计算数值级数展开
        f = self._eval_rewrite_as_li(*self.args)
        return f._eval_nseries(x, n, logx)

###############################################################################
#################### TRIGONOMETRIC INTEGRALS ##################################
###############################################################################

class TrigonometricIntegral(Function):
    """ Base class for trigonometric integrals. """


    @classmethod
    def eval(cls, z):
        # 如果 z 是零，则返回预先计算的值
        if z is S.Zero:
            return cls._atzero
        # 如果 z 是正无穷，则返回在正无穷处的值
        elif z is S.Infinity:
            return cls._atinf()
        # 如果 z 是负无穷，则返回在负无穷处的值
        elif z is S.NegativeInfinity:
            return cls._atneginf()

        # 如果 z 是零，则返回预先计算的值
        if z.is_zero:
            return cls._atzero

        # 尝试从 z 中提取与极坐标 I 相关的非零因子
        nz = z.extract_multiplicatively(polar_lift(I))
        # 如果提取成功且在角度函数为零时
        if nz is None and cls._trigfunc(0) == 0:
            # 尝试从 z 中提取与虚数单位 I 相关的非零因子
            nz = z.extract_multiplicatively(I)
        # 如果提取成功，则返回使用 I 因子的结果
        if nz is not None:
            return cls._Ifactor(nz, 1)
        # 尝试从 z 中提取与极坐标 -I 相关的非零因子
        nz = z.extract_multiplicatively(polar_lift(-I))
        # 如果提取成功，则返回使用 -I 因子的结果
        if nz is not None:
            return cls._Ifactor(nz, -1)

        # 尝试从 z 中提取与极坐标 -1 相关的非零因子
        nz = z.extract_multiplicatively(polar_lift(-1))
        # 如果提取成功且在角度函数为零时
        if nz is None and cls._trigfunc(0) == 0:
            # 尝试从 z 中提取与 -1 相关的非零因子
            nz = z.extract_multiplicatively(-1)
        # 如果提取成功，则返回使用 -1 因子的结果
        if nz is not None:
            return cls._minusfactor(nz)

        # 尝试从 z 中提取分支因子
        nz, n = z.extract_branch_factor()
        # 如果没有分支因子或分支因子和 z 相等，则返回 None
        if n == 0 and nz == z:
            return
        # 否则，返回表达式计算的结果
        return 2*pi*I*n*cls._trigfunc(0) + cls(nz)

    def fdiff(self, argindex=1):
        # 获取函数的第一个参数
        arg = unpolarify(self.args[0])
        # 如果参数索引为 1，则返回函数对参数的导数
        if argindex == 1:
            return self._trigfunc(arg)/arg
        # 如果参数索引不是 1，则抛出参数索引错误异常
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        # 返回使用指数积分 Ei 重写的函数表达式
        return self._eval_rewrite_as_expint(z).rewrite(Ei)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        # 导入上伽玛函数并返回使用它重写的函数表达式
        from sympy.functions.special.gamma_functions import uppergamma
        return self._eval_rewrite_as_expint(z).rewrite(uppergamma)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 注意：这段代码效率较低
        # 如果 self 的第一个参数在 x=0 处不为零
        if self.args[0].subs(x, 0) != 0:
            # 返回父类中的 n 阶级数展开结果
            return super()._eval_nseries(x, n, logx)
        # 否则，计算基础级数展开
        baseseries = self._trigfunc(x)._eval_nseries(x, n, logx)
        # 如果在 x=0 处 trigfunc(0) 不为零，则减去 1
        if self._trigfunc(0) != 0:
            baseseries -= 1
        # 将幂运算替换为 t**n/n
        baseseries = baseseries.replace(Pow, lambda t, n: t**n/n, simultaneous=False)
        # 如果在 x=0 处 trigfunc(0) 不为零，则加上 EulerGamma 和 log(x)
        if self._trigfunc(0) != 0:
            baseseries += EulerGamma + log(x)
        # 将 x 替换为 self 的第一个参数，并计算其 n 阶级数展开结果
        return baseseries.subs(x, self.args[0])._eval_nseries(x, n, logx)
class Si(TrigonometricIntegral):
    r"""
    Sine integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Si}(z) = \int_0^z \frac{\sin{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Si
    >>> from sympy.abc import z

    The sine integral is an antiderivative of $sin(z)/z$:

    >>> Si(z).diff(z)
    sin(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Si(z*exp_polar(2*I*pi))
    Si(z)

    Sine integral behaves much like ordinary sine under multiplication by ``I``:

    >>> Si(I*z)
    I*Shi(z)
    >>> Si(-z)
    -Si(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Si(z).rewrite(expint)
    -I*(-expint(1, z*exp_polar(-I*pi/2))/2 +
         expint(1, z*exp_polar(I*pi/2))/2) + pi/2

    It can be rewritten in the form of sinc function (by definition):

    >>> from sympy import sinc
    >>> Si(z).rewrite(sinc)
    Integral(sinc(_t), (_t, 0, z))

    See Also
    ========

    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    sinc: unnormalized sinc function
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sin  # 三角函数为正弦函数
    _atzero = S.Zero  # 在零点处的特定值为零

    @classmethod
    def _atinf(cls):
        # 在无穷远处的特定值为 pi/2
        return pi*S.Half

    @classmethod
    def _atneginf(cls):
        # 在负无穷远处的特定值为 -pi/2
        return -pi*S.Half

    @classmethod
    def _minusfactor(cls, z):
        # 返回 Si(-z) 的负值
        return -Si(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        # 返回 I * Shi(z) * sign
        return I*Shi(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # 将 Sine integral 重写为指数积分的形式
        # XXX should we polarify z? (这里的 XXX 是一个标记，表示可能需要进一步考虑极坐标化 z)
        return pi/2 + (E1(polar_lift(I)*z) - E1(polar_lift(-I)*z))/2/I

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        # 将 Sine integral 重写为积分形式，积分中为 sinc 函数
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sinc(t), (t, 0, z))

    _eval_rewrite_as_sinc =  _eval_rewrite_as_Integral

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 返回 Sine integral 的主导项
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            # 如果主导项在零点处为 NaN，则取极限
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            # 如果主导项在零点处为零，则返回主导项
            return arg
        elif not arg0.is_infinite:
            # 如果主导项在零点处不是无穷大，则返回函数应用在 arg0 上的结果
            return self.func(arg0)
        else:
            # 否则返回自身
            return self
    # 定义 _eval_aseries 方法，用于计算级数展开
    def _eval_aseries(self, n, args0, x, logx):
        # 导入 Order 类
        from sympy.series.order import Order
        # 获取展开点
        point = args0[0]

        # 如果展开点是无穷远（oo）
        if point is S.Infinity:
            # 获取函数的自变量
            z = self.args[0]
            # 计算偶数项的级数展开系数
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            # 计算奇数项的级数展开系数
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            # 返回级数展开结果
            return pi/2 - cos(z)*Add(*p) - sin(z)*Add(*q)

        # 对于其他所有展开点，调用父类的 _eval_aseries 方法处理
        return super(Si, self)._eval_aseries(n, args0, x, logx)

    # 定义 _eval_is_zero 方法，用于检查对象是否为零
    def _eval_is_zero(self):
        # 获取对象的自变量
        z = self.args[0]
        # 如果自变量 z 是零，则返回 True
        if z.is_zero:
            return True
class Ci(TrigonometricIntegral):
    r"""
    Cosine integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Ci}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cos{t} - 1}{t} \mathrm{d}t
           = -\int_x^\infty \frac{\cos{t}}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Ci}(z) =
        -\frac{\operatorname{E}_1\left(e^{i\pi/2} z\right)
               + \operatorname{E}_1\left(e^{-i \pi/2} z\right)}{2}

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for $z \in \mathbb{C}$ with $\Re(z) > 0$.
    By lifting to the principal branch, we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of $\cos(z)/z$:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary $\cos$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cos  # 使用余弦函数作为三角函数
    _atzero = S.ComplexInfinity  # 在零点处的值为复无穷大

    @classmethod
    def _atinf(cls):
        return S.Zero  # 在无穷远处的值为零

    @classmethod
    def _atneginf(cls):
        return I*pi  # 在负无穷远处的值为虚数单位乘以π

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I*pi  # 返回Ci(z)加上虚数单位乘以π

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I*pi/2*sign  # 返回Chi(z)加上虚数单位乘以π/2再乘以符号sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -(E1(polar_lift(I)*z) + E1(polar_lift(-I)*z))/2  # 使用指数积分重写为两个指数积分的负值除以2

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return S.EulerGamma + log(z) - Integral((1-cos(t))/t, (t, 0, z))  # 使用积分表达式重写为欧拉-伽玛常数加上ln(z)减去积分((1-cos(t))/t)关于t从0到z的积分
    # 计算表达式的主导项在给定变量 x 的值，可选参数 logx 为对数值，cdir 表示方向
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取表达式中第一个参数的主导项在 x 处的值
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        # 计算主导项在 x=0 处的值
        arg0 = arg.subs(x, 0)

        # 如果主导项在 x=0 处为 NaN，则通过取极限计算其值
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        # 如果主导项在 x=0 处为零
        if arg0.is_zero:
            # 将主导项表达为系数和指数形式
            c, e = arg.as_coeff_exponent(x)
            # 如果 logx 未提供，则使用默认的 log(x)
            logx = log(x) if logx is None else logx
            # 返回表达式的对数值、指数值及 EulerGamma 的和
            return log(c) + e*logx + EulerGamma
        # 如果主导项在 x=0 处为有限值
        elif arg0.is_finite:
            # 返回以该值为参数的函数值
            return self.func(arg0)
        # 如果主导项在 x=0 处为非有限值
        else:
            # 返回自身对象
            return self

    # 对象的级数展开至 n 阶，参数 args0 表示展开点，x 为自变量，logx 为对数值
    def _eval_aseries(self, n, args0, x, logx):
        # 导入级数展开所需的 Order 类
        from sympy.series.order import Order
        # 展开点为参数中的第一个值
        point = args0[0]

        # 如果展开点为正无穷或负无穷
        if point in (S.Infinity, S.NegativeInfinity):
            # 原始表达式中的参数
            z = self.args[0]
            # 计算级数展开的正奇数项
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            # 计算级数展开的负奇数项
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            # 计算级数展开的结果
            result = sin(z)*(Add(*p)) - cos(z)*(Add(*q))

            # 如果展开点为负无穷，添加虚数单位乘以 π
            if point is S.NegativeInfinity:
                result += I*pi
            # 返回计算结果
            return result

        # 如果展开点为有限值，则调用父类的级数展开方法
        return super(Ci, self)._eval_aseries(n, args0, x, logx)
class Shi(TrigonometricIntegral):
    r"""
    Sinh integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Shi}(z) = \int_0^z \frac{\sinh{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Shi
    >>> from sympy.abc import z

    The Sinh integral is a primitive of $\sinh(z)/z$:

    >>> Shi(z).diff(z)
    sinh(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Shi(z*exp_polar(2*I*pi))
    Shi(z)

    The $\sinh$ integral behaves much like ordinary $\sinh$ under
    multiplication by $i$:

    >>> Shi(I*z)
    I*Si(z)
    >>> Shi(-z)
    -Shi(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Shi(z).rewrite(expint)
    expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sinh  # 设置双曲正弦函数作为私有属性_trigfunc
    _atzero = S.Zero  # 在零点处的特定值设为零

    @classmethod
    def _atinf(cls):
        # 返回正无穷作为类方法的结果
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        # 返回负无穷作为类方法的结果
        return S.NegativeInfinity

    @classmethod
    def _minusfactor(cls, z):
        # 返回-z的双曲正弦积分作为类方法的结果
        return -Shi(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        # 返回I乘以Si(z)乘以符号的结果作为类方法的结果
        return I*Si(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # 将双曲正弦积分重写为指数积分的表达式
        # XXX should we polarify z? 是否应该对z进行极坐标化？
        return (E1(z) - E1(exp_polar(I*pi)*z))/2 - I*pi/2

    def _eval_is_zero(self):
        # 判断函数是否在参数为零时为零
        z = self.args[0]
        if z.is_zero:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 计算函数的主导项
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self


class Chi(TrigonometricIntegral):
    r"""
    Cosh integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Chi}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cosh{t} - 1}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Chi}(z) = \operatorname{Ci}\left(e^{i \pi/2}z\right)
                         - i\frac{\pi}{2},

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The $\cosh$ integral is a primitive of $\cosh(z)/z$:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The $\cosh$ integral behaves somewhat like ordinary $\cosh$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    # Define the hyperbolic cosine function as the trigonometric function for this integral
    _trigfunc = cosh
    # The value of the $\cosh$ integral at zero is complex infinity
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        # The value of the $\cosh$ integral at positive infinity is infinity
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        # The value of the $\cosh$ integral at negative infinity is infinity
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        # Expression for the $\cosh$ integral with an imaginary component
        return Chi(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        # Expression for the $\cosh$ integral with a specified sign and imaginary component
        return Ci(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # Rewrite the $\cosh$ integral in terms of exponential integrals
        return -I*pi/2 - (E1(z) + E1(exp_polar(I*pi)*z))/2

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # Evaluate the leading term of the $\cosh$ integral
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            # Return the leading term involving logarithm and Euler's gamma
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            # Return the function evaluated at the finite leading term
            return self.func(arg0)
        else:
            # Return the function as it is
            return self
###############################################################################
#################### FRESNEL INTEGRALS ########################################
###############################################################################

# 定义 Fresnel 积分函数的基类
class FresnelIntegral(Function):
    """ Base class for the Fresnel integrals."""

    unbranched = True  # 设置默认为非分支解析

    @classmethod
    def eval(cls, z):
        # 对于正无穷的情况返回半
        if z is S.Infinity:
            return S.Half

        # 对于零的情况返回零
        if z.is_zero:
            return S.Zero

        # 尝试提取出 -1 和 I 的因子
        prefact = S.One  # 初始化前置因子
        newarg = z  # 初始化新参数为 z
        changed = False  # 标记是否有改变过参数

        # 提取出 -1 的因子
        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact  # 前置因子变为 -1
            newarg = nz  # 更新参数为提取出 -1 的结果
            changed = True

        # 提取出 I 的因子
        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign*I*prefact  # 更新前置因子
            newarg = nz  # 更新参数为提取出 I 的结果
            changed = True

        # 如果参数有改变，返回前置因子乘以类的实例化对象
        if changed:
            return prefact*cls(newarg)

    # 对于 z 的偏导数函数
    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half*pi*self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    # 判断参数是否是扩展实数
    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    # 判断是否为有限数
    _eval_is_finite = _eval_is_extended_real

    # 判断参数是否为零
    def _eval_is_zero(self):
        return self.args[0].is_zero

    # 返回共轭的函数求值
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 实部到实部的映射
    as_real_imag = real_to_real_as_real_imag


# 定义 Fresnel S 积分类
class fresnels(FresnelIntegral):
    r"""
    Fresnel integral S.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{S}(z) = \int_0^z \sin{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnels
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnels(0)
    0
    >>> fresnels(oo)
    1/2
    >>> fresnels(-oo)
    -1/2
    >>> fresnels(I*oo)
    -I/2
    >>> fresnels(-I*oo)
    I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnels(-z)
    -fresnels(z)
    >>> fresnels(I*z)
    -I*fresnels(z)

    The Fresnel S integral obeys the mirror symmetry
    $\overline{S(z)} = S(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnels(z))
    fresnels(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnels(z), z)
    sin(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, sin, expand_func
    >>> integrate(sin(pi*z**2/2), z)
    3*fresnels(z)*gamma(3/4)/(4*gamma(7/4))
    >>> expand_func(integrate(sin(pi*z**2/2), z))
    fresnels(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision

    """

    # Fresnel S 积分的解释
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    _trigfunc = sin
    # 定义_trigfunc为正弦函数

    _sign = -S.One
    # 定义_sign为-1

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 定义静态方法taylor_term，计算Fresnel积分的泰勒展开项
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                # 如果有前期项，使用递归关系式计算泰勒展开的下一项
                return (-pi**2*x**4*(4*n - 1)/(8*n*(2*n + 1)*(4*n + 3))) * p
            else:
                # 如果是第一项，使用公式计算第一项的值
                return x**3 * (-x**4)**n * (S(2)**(-2*n - 1)*pi**(2*n + 1)) / ((4*n + 3)*factorial(2*n + 1))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        # 重写为误差函数形式的Fresnel积分
        return (S.One + I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 重写为超几何函数形式的Fresnel积分
        return pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        # 重写为Meijer G 函数形式的Fresnel积分
        return (pi*z**Rational(9, 4) / (sqrt(2)*(z**2)**Rational(3, 4)*(-z)**Rational(3, 4))
                * meijerg([], [1], [Rational(3, 4)], [Rational(1, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        # 重写为积分形式的Fresnel积分
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sin(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 计算Fresnel积分的主导项
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return pi*arg**3/6
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)
    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # 如果展开点是无穷大或负无穷大
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # S(x) = S1(x*sqrt(pi/2)) 的展开，参考文献[5] 第 1-8 页
            # 由于只处理实无穷大，sin和cos是O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(0, n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            # 对p和q进行标准化处理
            p = [-sqrt(2/pi)*t for t in p]
            q = [-sqrt(2/pi)*t for t in q]
            
            # 根据展开点确定符号
            s = 1 if point is S.Infinity else -1

            # 在oo处的展开是1/2加上z的奇次幂
            # 要得到-oo处的展开，将z替换为-z并翻转符号
            # 结果是-1/2加上与之前相同的z的奇次幂
            return s*S.Half + (sin(z**2)*Add(*p) + cos(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # 其他所有点不予处理，调用父类的_eval_aseries方法处理
        return super()._eval_aseries(n, args0, x, logx)
class fresnelc(FresnelIntegral):
    r"""
    Fresnel integral C.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{C}(z) = \int_0^z \cos{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnelc
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnelc(0)
    0
    >>> fresnelc(oo)
    1/2
    >>> fresnelc(-oo)
    -1/2
    >>> fresnelc(I*oo)
    I/2
    >>> fresnelc(-I*oo)
    -I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnelc(-z)
    -fresnelc(z)
    >>> fresnelc(I*z)
    I*fresnelc(z)

    The Fresnel C integral obeys the mirror symmetry
    $\overline{C(z)} = C(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnelc(z))
    fresnelc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnelc(z), z)
    cos(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, cos, expand_func
    >>> integrate(cos(pi*z**2/2), z)
    fresnelc(z)*gamma(1/4)/(4*gamma(5/4))
    >>> expand_func(integrate(cos(pi*z**2/2), z))
    fresnelc(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnelc(2).evalf(30)
    0.488253406075340754500223503357

    >>> fresnelc(-2*I).evalf(30)
    -0.488253406075340754500223503357*I

    See Also
    ========

    fresnels: Fresnel sine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelC
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """

    # 定义三角函数为余弦函数
    _trigfunc = cos
    # 符号设为1
    _sign = S.One

    @staticmethod
    @cacheit
    # 计算泰勒展开的项
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                # 计算下一个泰勒展开项
                return (-pi**2*x**4*(4*n - 3)/(8*n*(2*n - 1)*(4*n + 1))) * p
            else:
                # 计算第一个泰勒展开项
                return x * (-x**4)**n * (S(2)**(-2*n)*pi**(2*n)) / ((4*n + 1)*factorial(2*n))

    # 以误差函数形式重写
    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    # 以超几何函数形式重写
    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    # 以梅兴函数形式重写
    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(3, 4) / (sqrt(2)*root(z**2, 4)*root(-z, 4))
                * meijerg([], [1], [Rational(1, 4)], [Rational(3, 4), 0], -pi**2*z**4/16))
    # 将表达式重写为一个积分对象，其中 t 是一个虚拟变量，用于表示 z
    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        # 创建一个唯一命名的虚拟符号 t，并将其用作积分变量
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        # 返回余弦函数的积分对象，积分范围为 t 从 0 到 z
        return Integral(cos(pi*t**2/2), (t, 0, z))

    # 计算作为主导项的表达式，在 x 接近某个点时的值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        # 获取参数表达式的主导项，对 x 求导，同时考虑对数项和方向 cdir
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        # 将主导项在 x=0 处的值求出
        arg0 = arg.subs(x, 0)

        # 处理不同情况下的返回值
        if arg0 is S.ComplexInfinity:
            # 如果主导项在 x=0 处是复无穷，则根据方向 cdir 取极限值
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            # 如果主导项在 x=0 处为零，则直接返回主导项
            return arg
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            # 如果主导项在 x=0 处是正无穷或负无穷，则返回其符号乘以 1/2 加上 Order 对象
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            # 否则，返回调用原函数的结果
            return self.func(arg0)

    # 计算级数展开的高阶项，考虑在不同点处的展开情况
    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # 展开点为无穷时的情况处理
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # 根据参考文献[5]第1-8页的公式展开 C(x) = C1(x*sqrt(pi/2))
            # 由于只处理实无穷大，sin 和 cos 的阶数为 O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            # 对 p 和 q 中的每个元素乘以 -sqrt(2/pi) 和 sqrt(2/pi) 分别
            p = [-sqrt(2/pi)*t for t in p]
            q = [ sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # 在无穷大处的展开结果为 1/2 加上 z 的奇数幂的表达式
            # 要获得负无穷大处的展开，将 z 替换为 -z 并翻转符号
            # 结果为 -1/2 加上与前述相同的 z 的奇数幂
            return s*S.Half + (cos(z**2)*Add(*p) + sin(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # 对于其他点，不做处理，直接调用超类的 _eval_aseries 方法
        return super()._eval_aseries(n, args0, x, logx)
###############################################################################
#################### HELPER FUNCTIONS #########################################
###############################################################################


class _erfs(Function):
    """
    Helper function to make the $\\mathrm{erf}(z)$ function
    tractable for the Gruntz algorithm.

    """

    @classmethod
    def eval(cls, arg):
        # 如果参数是零，则返回1
        if arg.is_zero:
            return S.One

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # 在无穷远处展开
        if point is S.Infinity:
            z = self.args[0]
            # 计算阶乘和幂级数
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # 先添加阶乘再进行幂级数展开是非常低效的
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # 在I*无穷远处展开
        t = point.extract_multiplicatively(I)
        if t is S.Infinity:
            z = self.args[0]
            # TODO: 确认该级数是否正确
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # 先添加阶乘再进行幂级数展开是非常低效的
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # 其他情况未处理
        return super()._eval_aseries(n, args0, x, logx)

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            # 计算函数的导数
            return -2/sqrt(pi) + 2*z*_erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        # 将函数重写为不易处理的形式
        return (S.One - erf(z))*exp(z**2)


class _eis(Function):
    """
    Helper function to make the $\\mathrm{Ei}(z)$ and $\\mathrm{li}(z)$
    functions tractable for the Gruntz algorithm.

    """


    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        # 如果参数不是无穷大，则调用父类的展开方法
        if args0[0] != S.Infinity:
            return super(_erfs, self)._eval_aseries(n, args0, x, logx)

        z = self.args[0]
        # 计算阶乘和幂级数
        l = [factorial(k) * (1/z)**(k + 1) for k in range(n)]
        o = Order(1/z**(n + 1), x)
        # 先添加阶乘再进行幂级数展开是非常低效的
        return (Add(*l))._eval_nseries(x, n, logx) + o


    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            # 计算函数的导数
            return S.One / z - _eis(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        # 将函数重写为不易处理的形式
        return exp(-z)*Ei(z)
    # 计算作为主导项的表达式，对给定变量 x 进行评估
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 计算函数中第一个参数关于变量 x 在 x 趋近于 0 时的极限
        x0 = self.args[0].limit(x, 0)
        # 如果极限为零，将函数重新表达为不可解形式，并再次计算主导项
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # 否则，调用父类方法计算主导项
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    # 计算 n 级数（无穷级数的前 n 项）的表达式
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 计算函数中第一个参数关于变量 x 在 x 趋近于 0 时的极限
        x0 = self.args[0].limit(x, 0)
        # 如果极限为零，将函数重新表达为不可解形式，并重新计算 n 级数
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        # 否则，调用父类方法计算 n 级数
        return super()._eval_nseries(x, n, logx)
```