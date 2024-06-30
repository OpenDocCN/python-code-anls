# `D:\src\scipysrc\sympy\sympy\functions\special\elliptic_integrals.py`

```
""" Elliptic Integrals. """

from sympy.core import S, pi, I, Rational  # 导入必要的符号和常数
from sympy.core.function import Function, ArgumentIndexError  # 导入函数类和异常类
from sympy.core.symbol import Dummy, uniquely_named_symbol  # 导入符号类
from sympy.functions.elementary.complexes import sign  # 导入复数函数
from sympy.functions.elementary.hyperbolic import atanh  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import sin, tan  # 导入三角函数
from sympy.functions.special.gamma_functions import gamma  # 导入 Gamma 函数
from sympy.functions.special.hyper import hyper, meijerg  # 导入超函和梅耶格函数

class elliptic_k(Function):
    r"""
    The complete elliptic integral of the first kind, defined by

    .. math:: K(m) = F\left(\tfrac{\pi}{2}\middle| m\right)

    where $F\left(z\middle| m\right)$ is the Legendre incomplete
    elliptic integral of the first kind.

    Explanation
    ===========

    The function $K(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_k, I
    >>> from sympy.abc import m
    >>> elliptic_k(0)
    pi/2
    >>> elliptic_k(1.0 + I)
    1.50923695405127 + 0.625146415202697*I
    >>> elliptic_k(m).series(n=3)
    pi/2 + pi*m/8 + 9*pi*m**2/128 + O(m**3)

    See Also
    ========

    elliptic_f

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticK

    """

    @classmethod
    def eval(cls, m):
        # 如果 m 为零，则返回 pi/2
        if m.is_zero:
            return pi*S.Half
        # 如果 m 为 1/2，则返回 8*pi**(3/2) / (gamma(-1/4)**2)
        elif m is S.Half:
            return 8*pi**Rational(3, 2)/gamma(Rational(-1, 4))**2
        # 如果 m 为 1，则返回复无穷
        elif m is S.One:
            return S.ComplexInfinity
        # 如果 m 为 -1，则返回 gamma(1/4)**2 / (4*sqrt(2*pi))
        elif m is S.NegativeOne:
            return gamma(Rational(1, 4))**2/(4*sqrt(2*pi))
        # 如果 m 为无穷大、负无穷大、虚无穷大或虚负无穷大，则返回 0
        elif m in (S.Infinity, S.NegativeInfinity, I*S.Infinity,
                   I*S.NegativeInfinity, S.ComplexInfinity):
            return S.Zero

    def fdiff(self, argindex=1):
        # 计算 K(m) 的导数，m 为第一个参数
        m = self.args[0]
        return (elliptic_e(m) - (1 - m)*elliptic_k(m))/(2*m*(1 - m))

    def _eval_conjugate(self):
        # 返回 K(m) 的共轭复数，如果 m 不是实数或 m-1 不是正数
        m = self.args[0]
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        # 使用超函展开重写并计算级数展开
        return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))

    def _eval_rewrite_as_hyper(self, m, **kwargs):
        # 重写为超函形式
        return pi*S.Half*hyper((S.Half, S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, m, **kwargs):
        # 重写为梅耶格函数形式
        return meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -m)/2

    def _eval_is_zero(self):
        # 如果 m 为无穷大，则 K(m) 为零
        m = self.args[0]
        if m.is_infinite:
            return True
    # 定义一个方法 `_eval_rewrite_as_Integral`，用于将当前对象重写为一个积分形式
    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        # 导入 Integral 类，用于创建积分表达式
        from sympy.integrals.integrals import Integral
        # 创建一个唯一命名的虚拟符号 t，其名字基于传入的参数列表 args
        t = Dummy(uniquely_named_symbol('t', args).name)
        # 获取当前对象的第一个参数，假设为 m
        m = self.args[0]
        # 返回一个积分对象，积分表达式为 1 / sqrt(1 - m*sin(t)**2)，积分变量为 t，积分范围为 [0, pi/2)
        return Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, pi/2))
class elliptic_f(Function):
    r"""
    The Legendre incomplete elliptic integral of the first
    kind, defined by

    .. math:: F\left(z\middle| m\right) =
              \int_0^z \frac{dt}{\sqrt{1 - m \sin^2 t}}

    Explanation
    ===========

    This function reduces to a complete elliptic integral of
    the first kind, $K(m)$, when $z = \pi/2$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_f, I
    >>> from sympy.abc import z, m
    >>> elliptic_f(z, m).series(z)
    z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)
    >>> elliptic_f(3.0 + I/2, 1.0 + I)
    2.909449841483 + 1.74720545502474*I

    See Also
    ========

    elliptic_k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticF

    """

    @classmethod
    # 类方法，用于计算不完全椭圆积分的一类：F(z|m)
    def eval(cls, z, m):
        # 如果 z 为零，则结果为零
        if z.is_zero:
            return S.Zero
        # 如果 m 为零，则结果为 z
        if m.is_zero:
            return z
        # 计算参数 k，当 k 为整数时返回 k * elliptic_k(m)
        k = 2*z/pi
        if k.is_integer:
            return k*elliptic_k(m)
        # 当 m 为无穷大或负无穷时，返回零
        elif m in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        # 当 z 可能为负数时，返回 -elliptic_f(-z, m)
        elif z.could_extract_minus_sign():
            return -elliptic_f(-z, m)

    # 椭圆积分的一类在参数 m 上的偏导数
    def fdiff(self, argindex=1):
        z, m = self.args
        # 计算 sqrt(1 - m*sin(z)**2)
        fm = sqrt(1 - m*sin(z)**2)
        # 当 argindex 为 1 时，返回 1/fm
        if argindex == 1:
            return 1/fm
        # 当 argindex 为 2 时，返回复杂的表达式
        elif argindex == 2:
            return (elliptic_e(z, m)/(2*m*(1 - m)) - elliptic_f(z, m)/(2*m) -
                    sin(2*z)/(4*(1 - m)*fm))
        # 抛出参数索引错误
        raise ArgumentIndexError(self, argindex)

    # 对象的共轭值计算方法
    def _eval_conjugate(self):
        z, m = self.args
        # 如果 m 是实数且 m-1 大于零，则返回对象的共轭值
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(z.conjugate(), m.conjugate())

    # 将对象重写为积分形式的方法
    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        # 创建虚拟变量 t
        t = Dummy(uniquely_named_symbol('t', args).name)
        z, m = self.args[0], self.args[1]
        # 返回积分表达式 Integral(1/(sqrt(1 - m*sin(t)**2)), (t, 0, z))
        return Integral(1/(sqrt(1 - m*sin(t)**2)), (t, 0, z))

    # 判断对象是否为零的方法
    def _eval_is_zero(self):
        z, m = self.args
        # 如果 z 是零，则返回 True
        if z.is_zero:
            return True
        # 如果 m 是实数且为无穷大，则返回 True
        if m.is_extended_real and m.is_infinite:
            return True
    """
    定义了椭圆积分函数 elliptic_e 的求值方法和导数计算方法。
    """

    @classmethod
    def eval(cls, m, z=None):
        """
        类方法，用于计算 elliptic_e 函数的值。

        Parameters
        ----------
        m : sympy.Expr
            参数 m，可以是表达式。
        z : sympy.Expr, optional
            参数 z，可以是表达式。默认为 None。

        Returns
        -------
        sympy.Expr
            返回 elliptic_e 函数的值。

        Notes
        -----
        - 当 z 不为 None 时，根据不同的条件返回不同的计算结果。
        - 当 z 为 None 时，根据 m 的值返回相应的计算结果。
        """
        if z is not None:
            z, m = m, z
            k = 2*z/pi
            if m.is_zero:
                return z
            if z.is_zero:
                return S.Zero
            elif k.is_integer:
                return k*elliptic_e(m)
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.ComplexInfinity
            elif z.could_extract_minus_sign():
                return -elliptic_e(-z, m)
        else:
            if m.is_zero:
                return pi/2
            elif m is S.One:
                return S.One
            elif m is S.Infinity:
                return I*S.Infinity
            elif m is S.NegativeInfinity:
                return S.Infinity
            elif m is S.ComplexInfinity:
                return S.ComplexInfinity

    def fdiff(self, argindex=1):
        """
        求解 elliptic_e 函数的偏导数。

        Parameters
        ----------
        argindex : int, optional
            参数索引，默认为 1。

        Returns
        -------
        sympy.Expr
            返回 elliptic_e 函数的偏导数。

        Raises
        ------
        ArgumentIndexError
            如果参数索引不符合条件，抛出该异常。
        """
        if len(self.args) == 2:
            z, m = self.args
            if argindex == 1:
                return sqrt(1 - m*sin(z)**2)
            elif argindex == 2:
                return (elliptic_e(z, m) - elliptic_f(z, m))/(2*m)
        else:
            m = self.args[0]
            if argindex == 1:
                return (elliptic_e(m) - elliptic_k(m))/(2*m)
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        """
        计算 elliptic_e 函数的共轭。

        Returns
        -------
        sympy.Expr
            返回 elliptic_e 函数的共轭结果。
        """
        if len(self.args) == 2:
            z, m = self.args
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(z.conjugate(), m.conjugate())
        else:
            m = self.args[0]
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        计算 elliptic_e 函数的 n 阶级数展开。

        Parameters
        ----------
        x : sympy.Expr
            展开的变量。
        n : int
            展开的阶数。
        logx : bool
            是否包含对数。
        cdir : int, optional
            展开的方向，默认为 0。

        Returns
        -------
        sympy.Expr
            返回 elliptic_e 函数的 n 阶级数展开结果。
        """
        from sympy.simplify import hyperexpand
        if len(self.args) == 1:
            return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))
        return super()._eval_nseries(x, n=n, logx=logx)
    # 将当前对象重写为超几何函数形式
    def _eval_rewrite_as_hyper(self, *args, **kwargs):
        # 检查参数个数是否为1
        if len(args) == 1:
            # 获取参数中的第一个值
            m = args[0]
            # 返回重写后的表达式，使用超几何函数
            return (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), m)

    # 将当前对象重写为梅耶尔函数形式
    def _eval_rewrite_as_meijerg(self, *args, **kwargs):
        # 检查参数个数是否为1
        if len(args) == 1:
            # 获取参数中的第一个值
            m = args[0]
            # 返回重写后的表达式，使用梅耶尔函数
            return -meijerg(((S.Half, Rational(3, 2)), []), \
                            ((S.Zero,), (S.Zero,)), -m)/4

    # 将当前对象重写为积分形式
    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        # 导入积分相关的模块
        from sympy.integrals.integrals import Integral
        # 如果参数列表长度为1，则设置 z 和 m 的值
        z, m = (pi/2, self.args[0]) if len(self.args) == 1 else self.args
        # 创建一个唯一命名的符号 t
        t = Dummy(uniquely_named_symbol('t', args).name)
        # 返回重写后的表达式，使用积分形式
        return Integral(sqrt(1 - m*sin(t)**2), (t, 0, z))
# 定义 elliptic_pi 类，继承自 Function 类
class elliptic_pi(Function):
    r"""
    Called with three arguments $n$, $z$ and $m$, evaluates the
    Legendre incomplete elliptic integral of the third kind, defined by

    .. math:: \Pi\left(n; z\middle| m\right) = \int_0^z \frac{dt}
              {\left(1 - n \sin^2 t\right) \sqrt{1 - m \sin^2 t}}

    Called with two arguments $n$ and $m$, evaluates the complete
    elliptic integral of the third kind:

    .. math:: \Pi\left(n\middle| m\right) =
              \Pi\left(n; \tfrac{\pi}{2}\middle| m\right)

    Explanation
    ===========

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_pi, I
    >>> from sympy.abc import z, n, m
    >>> elliptic_pi(n, z, m).series(z, n=4)
    z + z**3*(m/6 + n/3) + O(z**4)
    >>> elliptic_pi(0.5 + I, 1.0 - I, 1.2)
    2.50232379629182 - 0.760939574180767*I
    >>> elliptic_pi(0, 0)
    pi/2
    >>> elliptic_pi(1.0 - I/3, 2.0 + I)
    3.29136443417283 + 0.32555634906645*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticPi3
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticPi

    """

    @classmethod
    # 定义一个类方法 eval，用于计算椭圆函数的值
    def eval(cls, n, m, z=None):
        # 如果 z 不为 None，则交换 z 和 m 的值
        if z is not None:
            z, m = m, z
            # 如果 n 是零，则返回 elliptic_f(z, m) 的计算结果
            if n.is_zero:
                return elliptic_f(z, m)
            # 如果 n 是 1，则返回以下表达式的计算结果
            elif n is S.One:
                return (elliptic_f(z, m) +
                        (sqrt(1 - m*sin(z)**2)*tan(z) -
                         elliptic_e(z, m))/(1 - m))
            # 计算 k = 2*z/pi，如果 k 是整数，则返回 k*elliptic_pi(n, m)
            k = 2*z/pi
            if k.is_integer:
                return k*elliptic_pi(n, m)
            # 如果 m 是零，则返回 atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1) 的计算结果
            elif m.is_zero:
                return atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1)
            # 如果 n 等于 m，则返回以下表达式的计算结果
            elif n == m:
                return (elliptic_f(z, n) - elliptic_pi(1, z, n) +
                        tan(z)/sqrt(1 - n*sin(z)**2))
            # 如果 n 或者 m 是正无穷或负无穷，则返回 S.Zero
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            # 如果 z 可以提取出负号，则返回 -elliptic_pi(n, -z, m) 的计算结果
            elif z.could_extract_minus_sign():
                return -elliptic_pi(n, -z, m)
            # 如果 n 是零，则返回 elliptic_f(z, m) 的计算结果
            if n.is_zero:
                return elliptic_f(z, m)
            # 如果 m 是实扩展数并且为无穷，或者 n 是实扩展数并且为无穷，则返回 S.Zero
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero
        else:
            # 如果 z 为 None，则执行以下操作
            # 如果 n 是零，则返回 elliptic_k(m) 的计算结果
            if n.is_zero:
                return elliptic_k(m)
            # 如果 n 是 1，则返回 S.ComplexInfinity
            elif n is S.One:
                return S.ComplexInfinity
            # 如果 m 是零，则返回 pi/(2*sqrt(1 - n)) 的计算结果
            elif m.is_zero:
                return pi/(2*sqrt(1 - n))
            # 如果 m 是 1，则返回 S.NegativeInfinity/sign(n - 1) 的计算结果
            elif m == S.One:
                return S.NegativeInfinity/sign(n - 1)
            # 如果 n 等于 m，则返回 elliptic_e(n)/(1 - n) 的计算结果
            elif n == m:
                return elliptic_e(n)/(1 - n)
            # 如果 n 或者 m 是正无穷或负无穷，则返回 S.Zero
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            # 如果 n 是零，则返回 elliptic_k(m) 的计算结果
            if n.is_zero:
                return elliptic_k(m)
            # 如果 m 是实扩展数并且为无穷，或者 n 是实扩展数并且为无穷，则返回 S.Zero
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero

    # 定义一个私有方法 _eval_conjugate，用于计算对象的共轭
    def _eval_conjugate(self):
        # 如果参数个数为 3，则执行以下操作
        if len(self.args) == 3:
            n, z, m = self.args
            # 如果 n 和 m 都不是实数或者 (n - 1) 不是正数，或者 m 不是实数或者 (m - 1) 不是正数，则返回对 n、z 和 m 求共轭后的结果
            if (n.is_real and (n - 1).is_positive) is False and \
               (m.is_real and (m - 1).is_positive) is False:
                return self.func(n.conjugate(), z.conjugate(), m.conjugate())
        else:
            # 如果参数个数不为 3，则执行以下操作
            n, m = self.args
            # 返回对 n 和 m 求共轭后的结果
            return self.func(n.conjugate(), m.conjugate())
    def fdiff(self, argindex=1):
        # 检查参数列表长度是否为3
        if len(self.args) == 3:
            # 解包参数
            n, z, m = self.args
            # 计算中间变量 fm 和 fn
            fm, fn = sqrt(1 - m*sin(z)**2), 1 - n*sin(z)**2
            # 根据 argindex 返回不同的表达式计算结果
            if argindex == 1:
                # 返回第一种情况下的计算结果
                return (elliptic_e(z, m) + (m - n)*elliptic_f(z, m)/n +
                        (n**2 - m)*elliptic_pi(n, z, m)/n -
                        n*fm*sin(2*z)/(2*fn))/(2*(m - n)*(n - 1))
            elif argindex == 2:
                # 返回第二种情况下的计算结果
                return 1/(fm*fn)
            elif argindex == 3:
                # 返回第三种情况下的计算结果
                return (elliptic_e(z, m)/(m - 1) +
                        elliptic_pi(n, z, m) -
                        m*sin(2*z)/(2*(m - 1)*fm))/(2*(n - m))
        else:
            # 如果参数列表长度不为3，重新解包参数
            n, m = self.args
            # 根据 argindex 返回不同的表达式计算结果
            if argindex == 1:
                # 返回第一种情况下的计算结果
                return (elliptic_e(m) + (m - n)*elliptic_k(m)/n +
                        (n**2 - m)*elliptic_pi(n, m)/n)/(2*(m - n)*(n - 1))
            elif argindex == 2:
                # 返回第二种情况下的计算结果
                return (elliptic_e(m)/(m - 1) + elliptic_pi(n, m))/(2*(n - m))
        # 如果未匹配到任何情况，抛出参数索引错误
        raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        # 导入积分模块
        from sympy.integrals.integrals import Integral
        # 检查参数列表长度是否为2
        if len(self.args) == 2:
            # 解包参数
            n, m, z = self.args[0], self.args[1], pi/2
        else:
            # 解包参数
            n, z, m = self.args
        # 创建一个唯一命名的虚拟符号 t
        t = Dummy(uniquely_named_symbol('t', args).name)
        # 返回积分表达式
        return Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, z))
```