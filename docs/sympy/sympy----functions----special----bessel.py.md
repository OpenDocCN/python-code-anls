# `D:\src\scipysrc\sympy\sympy\functions\special\bessel.py`

```
# 导入 functools 模块中的 wraps 装饰器
from functools import wraps

# 从 sympy.core 模块导入 S 符号
from sympy.core import S
# 从 sympy.core.add 模块导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.cache 模块导入 cacheit 装饰器
from sympy.core.cache import cacheit
# 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.expr import Expr
# 从 sympy.core.function 模块导入 Function 类、ArgumentIndexError 异常、_mexpand 函数
from sympy.core.function import Function, ArgumentIndexError, _mexpand
# 从 sympy.core.logic 模块导入 fuzzy_or、fuzzy_not 函数
from sympy.core.logic import fuzzy_or, fuzzy_not
# 从 sympy.core.numbers 模块导入 Rational 分数、pi 圆周率、I 虚数单位
from sympy.core.numbers import Rational, pi, I
# 从 sympy.core.power 模块导入 Pow 类
from sympy.core.power import Pow
# 从 sympy.core.symbol 模块导入 Dummy 符号、uniquely_named_symbol 函数、Wild 通配符
from sympy.core.symbol import Dummy, uniquely_named_symbol, Wild
# 从 sympy.core.sympify 模块导入 sympify 函数
from sympy.core.sympify import sympify
# 从 sympy.functions.combinatorial.factorials 模块导入 factorial 阶乘函数
from sympy.functions.combinatorial.factorials import factorial
# 从 sympy.functions.elementary.trigonometric 模块导入 sin、cos、csc、cot 函数
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
# 从 sympy.functions.elementary.integers 模块导入 ceiling 函数
from sympy.functions.elementary.integers import ceiling
# 从 sympy.functions.elementary.exponential 模块导入 exp、log 函数
from sympy.functions.elementary.exponential import exp, log
# 从 sympy.functions.elementary.miscellaneous 模块导入 cbrt、sqrt、root 函数
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
# 从 sympy.functions.elementary.complexes 模块导入 Abs、re、im、polar_lift、unpolarify 函数
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
# 从 sympy.functions.special.gamma_functions 模块导入 gamma、digamma、uppergamma 函数
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
# 从 sympy.functions.special.hyper 模块导入 hyper 函数
from sympy.functions.special.hyper import hyper
# 从 sympy.polys.orthopolys 模块导入 spherical_bessel_fn 函数
from sympy.polys.orthopolys import spherical_bessel_fn

# 从 mpmath 模块导入 mp、workprec 函数

# TODO
# o Scorer functions G1 and G2
# o Asymptotic expansions
#   These are possible, e.g. for fixed order, but since the bessel type
#   functions are oscillatory they are not actually tractable at
#   infinity, so this is not particularly useful right now.
# o Nicer series expansions.
# o More rewriting.
# o Add solvers to ode.py (or rather add solvers for the hypergeometric equation).

# 定义 BesselBase 类，继承自 Function 类
class BesselBase(Function):
    """
    Abstract base class for Bessel-type functions.

    This class is meant to reduce code duplication.
    All Bessel-type functions can 1) be differentiated, with the derivatives
    expressed in terms of similar functions, and 2) be rewritten in terms
    of other Bessel-type functions.

    Here, Bessel-type functions are assumed to have one complex parameter.

    To use this base class, define class attributes ``_a`` and ``_b`` such that
    ``2*F_n' = -_a*F_{n+1} + b*F_{n-1}``.
    """

    @property
    def order(self):
        """ The order of the Bessel-type function. """
        return self.args[0]

    @property
    def argument(self):
        """ The argument of the Bessel-type function. """
        return self.args[1]

    @classmethod
    def eval(cls, nu, z):
        # 此方法用于计算 Bessel 类型函数的值，但此处未实现具体内容，返回 None
        return

    def fdiff(self, argindex=2):
        # 如果参数索引不是 2，则抛出 ArgumentIndexError 异常
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        # 返回 Bessel 类型函数的导数表达式
        return (self._b/2 * self.__class__(self.order - 1, self.argument) -
                self._a/2 * self.__class__(self.order + 1, self.argument))

    def _eval_conjugate(self):
        # 获取函数的参数
        z = self.argument
        # 如果参数 z 不是扩展负数，则返回共轭函数的实例
        if z.is_extended_negative is False:
            return self.__class__(self.order.conjugate(), z.conjugate())
    # 判断是否为亚纯函数，即在x处是否为可解析的
    def _eval_is_meromorphic(self, x, a):
        # 提取阶数和参数
        nu, z = self.order, self.argument
        
        # 如果阶数包含变量x，则函数不是亚纯函数
        if nu.has(x):
            return False
        
        # 调用参数z的_eval_is_meromorphic方法判断是否在x处可解析
        if not z._eval_is_meromorphic(x, a):
            return None
        
        # 在点a处用z替换x，得到z0
        z0 = z.subs(x, a)
        
        # 如果阶数为整数
        if nu.is_integer:
            # 对于特定的函数类型或非零阶数，判断z0是否无穷大
            if isinstance(self, (besselj, besseli, hn1, hn2, jn, yn)) or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        
        # 否则判断z0是否既不是零又不是无穷大
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))

    # 对函数进行展开的方法，根据阶数nu和参数z
    def _eval_expand_func(self, **hints):
        nu, z, f = self.order, self.argument, self.__class__
        
        # 如果阶数是实数
        if nu.is_real:
            # 如果阶数-1是正数，返回展开式
            if (nu - 1).is_positive:
                return (-self._a*self._b*f(nu - 2, z)._eval_expand_func() +
                        2*self._a*(nu - 1)*f(nu - 1, z)._eval_expand_func()/z)
            # 如果阶数+1是负数，返回展开式
            elif (nu + 1).is_negative:
                return (2*self._b*(nu + 1)*f(nu + 1, z)._eval_expand_func()/z -
                        self._a*self._b*f(nu + 2, z)._eval_expand_func())
        
        # 对于其他情况，返回函数本身
        return self

    # 简化函数表达式的方法，使用sympy中的besselsimp函数
    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import besselsimp
        return besselsimp(self)
class besselj(BesselBase):
    r"""
    Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $J$ function of order $\nu$ is defined to be the function
    satisfying Bessel's differential equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu^2) w = 0,

    with Laurent expansion

    .. math ::
        J_\nu(z) = z^\nu \left(\frac{1}{\Gamma(\nu + 1) 2^\nu} + O(z^2) \right),

    if $\nu$ is not a negative integer. If $\nu=-n \in \mathbb{Z}_{<0}$
    *is* a negative integer, then the definition is

    .. math ::
        J_{-n}(z) = (-1)^n J_n(z).

    Examples
    ========

    Create a Bessel function object:

    >>> from sympy import besselj, jn
    >>> from sympy.abc import z, n
    >>> b = besselj(n, z)

    Differentiate it:

    >>> b.diff(z)
    besselj(n - 1, z)/2 - besselj(n + 1, z)/2

    Rewrite in terms of spherical Bessel functions:

    >>> b.rewrite(jn)
    sqrt(2)*sqrt(z)*jn(n - 1/2, z)/sqrt(pi)

    Access the parameter and argument:

    >>> b.order
    n
    >>> b.argument
    z

    See Also
    ========

    bessely, besseli, besselk

    References
    ==========

    .. [1] Abramowitz, Milton; Stegun, Irene A., eds. (1965), "Chapter 9",
           Handbook of Mathematical Functions with Formulas, Graphs, and
           Mathematical Tables
    .. [2] Luke, Y. L. (1969), The Special Functions and Their
           Approximations, Volume 1
    .. [3] https://en.wikipedia.org/wiki/Bessel_function
    .. [4] https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/

    """

    # 默认类变量，用于表示常数 1
    _a = S.One
    # 默认类变量，用于表示常数 1
    _b = S.One

    @classmethod
    # 类方法，用于计算 Bessel 函数的值
    def eval(cls, nu, z):
        # 当 z 为零时的情况处理
        if z.is_zero:
            # 当 nu 为零时，返回常数 1
            if nu.is_zero:
                return S.One
            # 当 nu 为整数且非零时，或者 nu 的实部为正时，返回常数 0
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            # 当 nu 的实部为负且 nu 不是整数时，返回复无穷
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            # 当 nu 是虚数时，返回 NaN
            elif nu.is_imaginary:
                return S.NaN
        # 当 z 为无穷大或负无穷大时，返回常数 0
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero

        # 当 z 可以提取负号时，利用 Bessel 函数的性质进行计算
        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besselj(nu, -z)
        # 当 nu 是整数时的特殊情况处理
        if nu.is_integer:
            # 当 nu 可以提取负号时，利用 Bessel 函数的对称性进行计算
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*besselj(-nu, z)
            # 当 z 可以通过乘以虚数单位 I 来转化时，返回相应的函数值
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(nu)*besseli(nu, newz)

        # 分支处理
        # 当 nu 是整数时，进行解极化处理
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besselj(nu, newz)
        else:
            # 提取分支因子并进行计算
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besselj(nu, newz)
        # 解极化 nu 并计算对应的 Bessel 函数值
        nnu = unpolarify(nu)
        if nu != nnu:
            return besselj(nnu, z)
    # 重写为贝塞尔函数 besseli 的表达式
    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        # 返回 exp(I*pi*nu/2) * besseli(nu, polar_lift(-I)*z) 的结果
        return exp(I*pi*nu/2)*besseli(nu, polar_lift(-I)*z)

    # 重写为贝塞尔函数 bessely 的表达式
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # 如果 nu 不是整数，则返回 csc(pi*nu)*bessely(-nu, z) - cot(pi*nu)*bessely(nu, z) 的结果
        if nu.is_integer is False:
            return csc(pi*nu)*bessely(-nu, z) - cot(pi*nu)*bessely(nu, z)

    # 重写为贝塞尔函数 jn 的表达式
    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        # 返回 sqrt(2*z/pi)*jn(nu - S.Half, self.argument) 的结果
        return sqrt(2*z/pi)*jn(nu - S.Half, self.argument)

    # 计算作为主导项的表达式
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取参数 nu 和 z
        nu, z = self.args
        try:
            # 获取 z 相对于 x 的主导项
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        # 获取 arg 的系数和指数
        c, e = arg.as_coeff_exponent(x)

        # 如果指数 e 是正数
        if e.is_positive:
            # 返回 arg**nu/(2**nu*gamma(nu + 1)) 的结果
            return arg**nu/(2**nu*gamma(nu + 1))
        # 如果指数 e 是负数
        elif e.is_negative:
            # 如果 cdir 是 0，则将其设置为 1
            cdir = 1 if cdir == 0 else cdir
            # 计算符号
            sign = c*cdir**e
            # 如果符号不为负数，则返回 sqrt(2)*cos(z - pi*(2*nu + 1)/4)/sqrt(pi*z) 的结果
            if not sign.is_negative:
                # 参考 Abramowitz 和 Stegun 1965 年第 364 页，关于贝塞尔函数 besselj 的渐近逼近的更多信息
                return sqrt(2)*cos(z - pi*(2*nu + 1)/4)/sqrt(pi*z)
            # 否则返回 self
            return self

        # 调用父类的 _eval_as_leading_term 方法
        return super(besselj, self)._eval_as_leading_term(x, logx, cdir)

    # 判断是否为扩展实数
    def _eval_is_extended_real(self):
        # 获取参数 nu 和 z
        nu, z = self.args
        # 如果 nu 是整数且 z 是扩展实数，则返回 True
        if nu.is_integer and z.is_extended_real:
            return True

    # 计算 nseries 展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 参考 https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/06/01/04/01/01/0003/
        # 获取参数 nu 和 z
        nu, z = self.args

        # 在幂小于 1 的情况下，需要单独计算项数以避免错误的 n 值反复调用 _eval_nseries
        try:
            # 获取 z 相对于 x 的主导项和指数
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        # 如果指数 exp 是正数
        if exp.is_positive:
            # 计算新的 n 值
            newn = ceiling(n/exp)
            # 计算 Order 对象 o
            o = Order(x**n, x)
            # 计算 r，并移除 Order
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            # 如果 r 是零，则返回 Order 对象 o
            if r is S.Zero:
                return o
            # 计算 t
            t = (_mexpand(r**2) + o).removeO()

            # 计算 term
            term = r**nu/gamma(nu + 1)
            s = [term]
            # 循环计算表达式的和
            for k in range(1, (newn + 1)//2):
                term *= -t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            # 返回表达式的总和加上 Order 对象 o
            return Add(*s) + o

        # 调用父类的 _eval_nseries 方法
        return super(besselj, self)._eval_nseries(x, n, logx, cdir)
class bessely(BesselBase):
    r"""
    Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $Y$ function of order $\nu$ is defined as

    .. math ::
        Y_\nu(z) = \lim_{\mu \to \nu} \frac{J_\mu(z) \cos(\pi \mu)
                                            - J_{-\mu}(z)}{\sin(\pi \mu)},

    where $J_\mu(z)$ is the Bessel function of the first kind.

    It is a solution to Bessel's equation, and linearly independent from
    $J_\nu$.

    Examples
    ========

    >>> from sympy import bessely, yn
    >>> from sympy.abc import z, n
    >>> b = bessely(n, z)
    >>> b.diff(z)
    bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    >>> b.rewrite(yn)
    sqrt(2)*sqrt(z)*yn(n - 1/2, z)/sqrt(pi)

    See Also
    ========

    besselj, besseli, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/

    """

    # 初始化类属性 _a 和 _b，均设置为 1
    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        # 若 z 为零
        if z.is_zero:
            # 若 nu 也为零，返回负无穷
            if nu.is_zero:
                return S.NegativeInfinity
            # 若 nu 不为零且其实部非零，则返回复无穷
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            # 若 nu 的实部为零，则返回非数值
            elif re(nu).is_zero:
                return S.NaN
        # 若 z 为无穷大或负无穷大，返回零
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        # 若 z 为虚无大，则根据 nu 的值返回对应的无穷大
        if z == I*S.Infinity:
            return exp(I*pi*(nu + 1)/2) * S.Infinity
        # 若 z 为虚无小，则根据 nu 的值返回对应的无穷大
        if z == I*S.NegativeInfinity:
            return exp(-I*pi*(nu + 1)/2) * S.Infinity

        # 若 nu 为整数
        if nu.is_integer:
            # 若 nu 可提取负号，则返回负一的 nu 次方乘以 bessely(-nu, z)
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*bessely(-nu, z)

    # 将当前对象重新表示为 besselj 的形式
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*(cos(pi*nu)*besselj(nu, z) - besselj(-nu, z))

    # 将当前对象重新表示为 besseli 的形式
    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(besseli)

    # 将当前对象重新表示为 yn 的形式
    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return sqrt(2*z/pi) * yn(nu - S.Half, self.argument)
    # 对象方法，用于评估当前对象作为主导项时的行为，给定参数 x，对数 logx，以及方向 cdir，默认为 0
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 解构获取参数 nu 和 z
        nu, z = self.args
        try:
            # 调用 z 对象的 as_leading_term 方法，返回近似主导项
            arg = z.as_leading_term(x)
        except NotImplementedError:
            # 如果 z 不支持 as_leading_term 方法，则返回当前对象自身
            return self
        # 获取参数 arg 的系数 c 和指数 e
        c, e = arg.as_coeff_exponent(x)

        # 如果指数 e 是正数
        if e.is_positive:
            # 计算第一个项 ((2/pi)*log(z/2)*besselj(nu, z))
            term_one = ((2/pi)*log(z/2)*besselj(nu, z))
            # 计算第二个项 -(z/2)**(-nu)*factorial(nu - 1)/pi，如果 nu 是正数的话，否则为零
            term_two = -(z/2)**(-nu)*factorial(nu - 1)/pi if (nu).is_positive else S.Zero
            # 计算第三个项 -(z/2)**nu/(pi*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            term_three = -(z/2)**nu/(pi*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            # 将三个项相加，并将结果作为主导项返回，考虑对数 logx
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        # 如果指数 e 是负数
        elif e.is_negative:
            # 根据 cdir 的值确定方向
            cdir = 1 if cdir == 0 else cdir
            # 计算符号 sign
            sign = c*cdir**e
            if not sign.is_negative:
                # 返回 bessely 函数的渐近近似值，参考 Abramowitz 和 Stegun 1965 年第 364 页
                return sqrt(2)*(-sin(pi*nu/2 - z + pi/4) + 3*cos(pi*nu/2 - z + pi/4)/(8*z))*sqrt(1/z)/sqrt(pi)
            # 如果 sign 是负数，返回当前对象自身
            return self

        # 如果以上条件都不满足，则调用超类的 _eval_as_leading_term 方法
        return super(bessely, self)._eval_as_leading_term(x, logx, cdir)

    # 对象方法，用于评估当前对象是否是扩展实数
    def _eval_is_extended_real(self):
        # 解构获取参数 nu 和 z
        nu, z = self.args
        # 如果 nu 是整数且 z 是正数，则返回 True
        if nu.is_integer and z.is_positive:
            return True
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 参考 Wolfram 函数文档 https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/0008/
        # 获取 bessely 函数的 nseries 展开更多信息。

        from sympy.series.order import Order
        nu, z = self.args

        # 如果指数 exp 是正数且 nu 是整数，则计算新的展开阶数 newn
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            newn = ceiling(n/exp)
            bn = besselj(nu, z)

            # 计算 a 部分的 nseries 展开
            a = ((2/pi)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()

            # 如果 r 是零，则直接返回 Order(x**n, x)
            if r is S.Zero:
                return o

            # 计算 t 部分
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                # 计算 b 部分
                term = r**(-nu)*factorial(nu - 1)/pi
                b.append(term)
                for k in range(1, nu):
                    denom = (nu - k)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            # 计算 c 部分
            p = r**nu/(pi*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= -t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)

            # 返回最终结果，包括 a, b, c 的和以及可能的 Order 项
            return a - Add(*b) - Add(*c)  # Order term comes from a

        # 如果不符合以上条件，则调用父类的 _eval_nseries 方法
        return super(bessely, self)._eval_nseries(x, n, logx, cdir)
class besseli(BesselBase):
    r"""
    Modified Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $I$ function is a solution to the modified Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 + \nu^2)^2 w = 0.

    It can be defined as

    .. math ::
        I_\nu(z) = i^{-\nu} J_\nu(iz),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    Examples
    ========

    >>> from sympy import besseli
    >>> from sympy.abc import z, n
    >>> besseli(n, z).diff(z)
    besseli(n - 1, z)/2 + besseli(n + 1, z)/2

    See Also
    ========

    besselj, bessely, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/

    """

    # Class variables _a and _b initialized to specific SymPy constants
    _a = -S.One
    _b = S.One

    # Class method eval to compute the value of the Bessel function
    @classmethod
    def eval(cls, nu, z):
        # Handle special cases and return specific SymPy constants or expressions
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        # Handle z being infinite or imaginary infinite
        if im(z) in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z is S.Infinity:
            return S.Infinity
        if z is S.NegativeInfinity:
            return (-1)**nu*S.Infinity

        # Handle cases where z could have a minus sign extracted
        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besseli(nu, -z)
        
        # Handle integer nu cases
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(-nu)*besselj(nu, -newz)

        # Branch handling for non-integer nu
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besseli(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besseli(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besseli(nnu, z)

    # Method to rewrite the function in terms of Bessel J function
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return exp(-I*pi*nu/2)*besselj(nu, polar_lift(I)*z)

    # Method to rewrite the function in terms of Bessel Y function
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    # Method to rewrite the function in terms of J_n function
    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return self._eval_rewrite_as_besselj(*self.args).rewrite(jn)

    # Method to check if the function is extended real
    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取函数参数 nu 和 z
        nu, z = self.args
        
        # 尝试获取 z 的主导项，如果不支持，则返回当前对象
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        
        # 获取参数 arg 的系数和指数
        c, e = arg.as_coeff_exponent(x)

        # 如果指数 e 是正数
        if e.is_positive:
            # 返回主导项的计算结果
            return arg**nu/(2**nu*gamma(nu + 1))
        # 如果指数 e 是负数
        elif e.is_negative:
            # 根据参数 cdir 的值，设置符号
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            
            # 如果符号不为负数，则返回指定的渐近近似结果
            if not sign.is_negative:
                # 参考 Abramowitz and Stegun 1965, p. 377，获取 Bessel 函数的渐近近似信息
                return exp(z)/sqrt(2*pi*z)
            # 否则返回当前对象
            return self
        
        # 如果以上条件均不满足，则调用父类方法继续处理
        return super(besseli, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 参考文献：https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/06/01/04/01/01/0003/
        # 获取函数参数 nu 和 z
        nu, z = self.args

        # 当指数 exp 是正数时，计算新的 n 值
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self
        
        if exp.is_positive:
            newn = ceiling(n/exp)
            # 定义阶数 o
            o = Order(x**n, x)
            # 计算 z/2 的 nseries 展开并去除高阶项
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            # 如果 r 是零，则返回阶数 o
            if r is S.Zero:
                return o
            # 计算 t 的值，并去除高阶项
            t = (_mexpand(r**2) + o).removeO()

            # 计算主项 term
            term = r**nu/gamma(nu + 1)
            s = [term]
            # 循环计算系列展开的每一项
            for k in range(1, (newn + 1)//2):
                term *= t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            # 返回计算结果的总和加上阶数 o
            return Add(*s) + o

        # 如果指数 exp 不是正数，则调用父类方法继续处理
        return super(besseli, self)._eval_nseries(x, n, logx, cdir)
class`
class besselk(BesselBase):
    # 文档字符串，描述了修正贝塞尔函数第二类K函数的定义和性质
    r"""
    Modified Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $K$ function of order $\nu$ is defined as

    .. math ::
        K_\nu(z) = \lim_{\mu \to \nu} \frac{\pi}{2}
                   \frac{I_{-\mu}(z) -I_\mu(z)}{\sin(\pi \mu)},

    where $I_\mu(z)$ is the modified Bessel function of the first kind.

    It is a solution of the modified Bessel equation, and linearly independent
    from $Y_\nu$.

    Examples
    ========

    >>> from sympy import besselk
    >>> from sympy.abc import z, n
    >>> besselk(n, z).diff(z)
    -besselk(n - 1, z)/2 - besselk(n + 1, z)/2

    See Also
    ========

    besselj, besseli, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/

    """

    _a = S.One  # 类变量_a，设定为1，表示K函数的常数系数
    _b = -S.One  # 类变量_b，设定为-1，表示K函数的常数系数

    @classmethod
    def eval(cls, nu, z):
        # 类方法 eval，计算给定参数的K函数值
        if z.is_zero:
            # 当z为零时，处理不同nu值的特殊情况
            if nu.is_zero:
                return S.Infinity  # 当nu为0时，K函数值为无穷大
            elif re(nu).is_zero is False:
                return S.ComplexInfinity  # 当nu不是实数时，K函数值为复无穷
            elif re(nu).is_zero:
                return S.NaN  # 当nu为复数且其实部为0时，返回NaN
        if z in (S.Infinity, I*S.Infinity, I*S.NegativeInfinity):
            return S.Zero  # 当z为无穷大时，K函数值为0

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besselk(-nu, z)  # 当nu为整数且可以提取负号时，返回负整数次的K函数

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        # 重写为Besseli函数的形式
        if nu.is_integer is False:
            return pi*csc(pi*nu)*(besseli(-nu, z) - besseli(nu, z))/2  # 返回表达式转换为Besseli函数的形式

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # 重写为Besselj函数的形式
        ai = self._eval_rewrite_as_besseli(*self.args)  # 调用_besseli函数进行计算
        if ai:
            return ai.rewrite(besselj)  # 如果ai不为空，返回表达式转换为Besselj函数的形式

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # 重写为Bessely函数的形式
        aj = self._eval_rewrite_as_besselj(*self.args)  # 调用_besselj函数进行计算
        if aj:
            return aj.rewrite(bessely)  # 如果aj不为空，返回表达式转换为Bessely函数的形式

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        # 重写为Yn函数的形式
        ay = self._eval_rewrite_as_bessely(*self.args)  # 调用_bessely函数进行计算
        if ay:
            return ay.rewrite(yn)  # 如果ay不为空，返回表达式转换为Yn函数的形式

    def _eval_is_extended_real(self):
        # 判断K函数是否为扩展实数
        nu, z = self.args  # 获取参数nu和z
        if nu.is_integer and z.is_positive:
            return True  # 如果nu为整数且z为正数，返回True
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 获取参数 nu 和 z
        nu, z = self.args
        try:
            # 获取 z 的主导项
            arg = z.as_leading_term(x)
        except NotImplementedError:
            # 如果无法计算主导项，则返回自身
            return self
        # 获取主导项的系数和指数
        _, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            # 如果指数为正
            # 计算第一个项
            term_one = ((-1)**(nu -1)*log(z/2)*besseli(nu, z))
            # 计算第二个项
            term_two = (z/2)**(-nu)*factorial(nu - 1)/2 if (nu).is_positive else S.Zero
            # 计算第三个项
            term_three = (-1)**nu*(z/2)**nu/(2*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            # 将三个项加起来，并取其主导项
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            # 如果指数为负
            # 返回 Bessel 函数的渐近近似
            return sqrt(pi)*exp(-z)/sqrt(2*z)

        # 如果以上条件都不满足，则调用父类的方法计算
        return super(besselk, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # 参考 Wolfram 函数网站上关于 BesselK 函数 nseries 展开的信息
        from sympy.series.order import Order
        nu, z = self.args

        # 对于小于 1 的幂次，需要单独计算项数以避免错误的 n 值反复调用 _eval_nseries
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            # 如果指数为正且 nu 是整数
            newn = ceiling(n/exp)
            bn = besseli(nu, z)
            a = ((-1)**(nu - 1)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                # 计算 b 部分的系数
                term = r**(-nu)*factorial(nu - 1)/2
                b.append(term)
                for k in range(1, nu):
                    denom = (k - nu)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            p = r**nu*(-1)**nu/(2*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            # 返回所有计算出的项的和，包括 Order 项来自 a 部分
            return a + Add(*b) + Add(*c) # Order term comes from a

        # 如果以上条件都不满足，则调用父类的方法计算
        return super(besselk, self)._eval_nseries(x, n, logx, cdir)
class hankel1(BesselBase):
    r"""
    Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(1)} = J_\nu(z) + iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation.

    Examples
    ========

    >>> from sympy import hankel1
    >>> from sympy.abc import z, n
    >>> hankel1(n, z).diff(z)
    hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2

    See Also
    ========

    hankel2, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH1/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        # 如果 z 不是扩展负数，则返回 hankel2 函数的共轭
        if z.is_extended_negative is False:
            return hankel2(self.order.conjugate(), z.conjugate())


class hankel2(BesselBase):
    r"""
    Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(2)} = J_\nu(z) - iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation, and linearly independent from
    $H_\nu^{(1)}$.

    Examples
    ========

    >>> from sympy import hankel2
    >>> from sympy.abc import z, n
    >>> hankel2(n, z).diff(z)
    hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2

    See Also
    ========

    hankel1, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH2/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        # 如果 z 不是扩展负数，则返回 hankel1 函数的共轭
        if z.is_extended_negative is False:
            return hankel1(self.order.conjugate(), z.conjugate())


def assume_integer_order(fn):
    @wraps(fn)
    def g(self, nu, z):
        # 如果 nu 是整数，则调用传入函数 fn 处理 nu 和 z，并返回结果
        if nu.is_integer:
            return fn(self, nu, z)
    return g


class SphericalBesselBase(BesselBase):
    """
    Base class for spherical Bessel functions.

    These are thin wrappers around ordinary Bessel functions,
    since spherical Bessel functions differ from the ordinary
    ones just by a slight change in order.

    To use this class, define the ``_eval_evalf()`` and ``_expand()`` methods.

    """

    def _expand(self, **hints):
        """ Expand self into a polynomial. Nu is guaranteed to be Integer. """
        # 抛出未实现错误，要求在子类中实现扩展方法
        raise NotImplementedError('expansion')

    def _eval_expand_func(self, **hints):
        # 如果 order 是整数，则调用 _expand 方法展开对象
        if self.order.is_Integer:
            return self._expand(**hints)
        return self

    def fdiff(self, argindex=2):
        # 如果 argindex 不等于 2，则抛出参数索引错误
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        # 返回导数结果，根据公式计算
        return self.__class__(self.order - 1, self.argument) - \
            self * (self.order + 1)/self.argument
    # 返回球贝塞尔函数的特定表达式
    return (spherical_bessel_fn(n, z) * sin(z) +
            S.NegativeOne**(n + 1) * spherical_bessel_fn(-n - 1, z) * cos(z))
def _yn(n, z):
    # 计算球贝塞尔函数的第二类函数 Y_n(z)，使用公式 (-1)**(n + 1) * _jn(-n - 1, z)
    return (S.NegativeOne**(n + 1) * spherical_bessel_fn(-n - 1, z)*sin(z) -
            spherical_bessel_fn(n, z)*cos(z))


class jn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the first kind.

    Explanation
    ===========

    This function is a solution to the spherical Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
          + 2z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu(\nu + 1)) w = 0.

    It can be defined as

    .. math ::
        j_\nu(z) = \sqrt{\frac{\pi}{2z}} J_{\nu + \frac{1}{2}}(z),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    The spherical Bessel functions of integral order are
    calculated using the formula:

    .. math:: j_n(z) = f_n(z) \sin{z} + (-1)^{n+1} f_{-n-1}(z) \cos{z},

    where the coefficients $f_n(z)$ are available as
    :func:`sympy.polys.orthopolys.spherical_bessel_fn`.

    Examples
    ========

    >>> from sympy import Symbol, jn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(jn(0, z)))
    sin(z)/z
    >>> expand_func(jn(1, z)) == sin(z)/z**2 - cos(z)/z
    True
    >>> expand_func(jn(3, z))
    (-6/z**2 + 15/z**4)*sin(z) + (1/z - 15/z**3)*cos(z)
    >>> jn(nu, z).rewrite(besselj)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(nu + 1/2, z)/2
    >>> jn(nu, z).rewrite(bessely)
    (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-nu - 1/2, z)/2
    >>> jn(2, 5.2+0.3j).evalf(20)
    0.099419756723640344491 - 0.054525080242173562897*I

    See Also
    ========

    besselj, bessely, besselk, yn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @classmethod
    def eval(cls, nu, z):
        # 计算球贝塞尔函数 J_n(z) 的特定情况下的值
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif nu.is_integer:
                if nu.is_positive:
                    return S.Zero
                else:
                    return S.ComplexInfinity
        if z in (S.NegativeInfinity, S.Infinity):
            return S.Zero

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # 使用 Bessel 函数 J_{nu + 1/2}(z) 的重写形式计算球贝塞尔函数 J_n(z)
        return sqrt(pi/(2*z)) * besselj(nu + S.Half, z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # 使用 Bessel 函数 Y_{-nu - 1/2}(z) 的重写形式计算球贝塞尔函数 J_n(z)
        return S.NegativeOne**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        # 使用 Y_n(z) 的定义重写球贝塞尔函数 J_n(z)
        return S.NegativeOne**(nu) * yn(-nu - 1, z)

    def _expand(self, **hints):
        # 展开球贝塞尔函数 J_n(z)
        return _jn(self.order, self.argument)

    def _eval_evalf(self, prec):
        # 对整数阶球贝塞尔函数 J_n(z) 进行数值计算
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class yn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the second kind.

    Explanation
    ===========

    This function is another solution to the spherical Bessel equation, and
    linearly independent from $j_n$. It can be defined as
    # 定义修饰符函数，假设整数阶的修饰符
    @assume_integer_order
    # 用 Bessel 函数 J 的形式重写函数
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return S.NegativeOne**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)

    # 定义修饰符函数，假设整数阶的修饰符
    @assume_integer_order
    # 用 Bessel 函数 Y 的形式重写函数
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * bessely(nu + S.Half, z)

    # 用 Bessel 函数 J_n 的形式重写函数
    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu + 1) * jn(-nu - 1, z)

    # 对函数进行展开处理
    def _expand(self, **hints):
        return _yn(self.order, self.argument)

    # 对函数进行数值计算处理
    def _eval_evalf(self, prec):
        # 如果阶数是整数，则用 Bessel 函数 Y 重写函数后再进行数值计算
        if self.order.is_Integer:
            return self.rewrite(bessely)._eval_evalf(prec)
class SphericalHankelBase(SphericalBesselBase):
    
    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # Rewrite the spherical Hankel function of the first kind as a combination of Bessel functions.
        # Using the relation h_nu^(1)(z) = sqrt(pi/(2*z)) * [besselj(nu + 1/2, z) + i * hankel_kind_sign * besselj(-nu - 1/2, z)]
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(besselj(nu + S.Half, z) +
                               hks*I*S.NegativeOne**(nu+1)*besselj(-nu - S.Half, z))

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # Rewrite the spherical Hankel function of the first kind as a combination of Bessel functions.
        # Using the relation h_nu^(1)(z) = sqrt(pi/(2*z)) * [(-1)^nu * bessely(-nu - 1/2, z) + i * hankel_kind_sign * bessely(nu + 1/2, z)]
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(S.NegativeOne**nu*bessely(-nu - S.Half, z) +
                               hks*I*bessely(nu + S.Half, z))

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        # Rewrite the spherical Hankel function of the first kind as a combination of Bessel functions.
        # Using the relation h_nu^(1)(z) = jn(nu, z).rewrite(yn) + i * hankel_kind_sign * yn(nu, z)
        hks = self._hankel_kind_sign
        return jn(nu, z).rewrite(yn) + hks*I*yn(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        # Rewrite the spherical Hankel function of the first kind as a combination of Bessel functions.
        # Using the relation h_nu^(1)(z) = jn(nu, z) + i * hankel_kind_sign * yn(nu, z).rewrite(jn)
        hks = self._hankel_kind_sign
        return jn(nu, z) + hks*I*yn(nu, z).rewrite(jn)

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            # Expand the expression for the spherical Hankel function of the first kind when the order is an integer.
            return self._expand(**hints)
        else:
            nu = self.order
            z = self.argument
            hks = self._hankel_kind_sign
            # Return the expression using Bessel functions when the order is not an integer.
            return jn(nu, z) + hks*I*yn(nu, z)

    def _expand(self, **hints):
        n = self.order
        z = self.argument
        hks = self._hankel_kind_sign

        # Return the fully expanded version of the spherical Hankel function of the first kind.
        # h_n^(1)(z) = fn(n, z) * sin(z) + (-1)^(n + 1) * fn(-n - 1, z) * cos(z)
        #            + hks * i * (-1)^(n + 1) * (fn(-n - 1, z) * hk * i * sin(z) + (-1)^(-n) * fn(n, z) * i * cos(z))
        return (_jn(n, z) + hks*I*_yn(n, z)).expand()

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            # Evaluate the spherical Hankel function of the first kind numerically if the order is an integer.
            return self.rewrite(besselj)._eval_evalf(prec)


class hn1(SphericalHankelBase):
    r"""
    Spherical Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(1)(z) = j_\nu(z) + i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(1)$ is calculated using the formula:

    .. math:: h_n^(1)(z) = j_{n}(z) + i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn1, hankel1, expand_func, yn, jn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn1(nu, z)))
    jn(nu, z) + I*yn(nu, z)
    >>> print(expand_func(hn1(0, z)))
    sin(z)/z - I*cos(z)/z
    >>> print(expand_func(hn1(1, z)))
    -I*sin(z)/z - cos(z)/z + sin(z)/z**2 - I*cos(z)/z**2
    >>> hn1(nu, z).rewrite(jn)
    (-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    # 返回第一类Hankel函数的表达式，其中nu为整数，z为复数参数

    >>> hn1(nu, z).rewrite(yn)
    # 用第二类Bessel函数的形式重写第一类Hankel函数

    (-1)**nu*yn(-nu - 1, z) + I*yn(nu, z)
    # 返回重写后的第一类Hankel函数的表达式，用第二类Bessel函数yn表示

    >>> hn1(nu, z).rewrite(hankel1)
    # 用第一类Hankel函数的形式重写第一类Hankel函数

    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel1(nu, z)/2
    # 返回重写后的第一类Hankel函数的表达式，用第一类Hankel函数hankel1表示

    See Also
    ========

    hn2, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = S.One
    # 设置Hankel函数系数的符号，默认为1

    @assume_integer_order
    def _eval_rewrite_as_hankel1(self, nu, z, **kwargs):
        # 假设nu为整数，使用第一类Hankel函数的形式重写当前函数
        return sqrt(pi/(2*z))*hankel1(nu, z)
class hn2(SphericalHankelBase):
    r"""
    Spherical Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(2)(z) = j_\nu(z) - i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(2)$ is calculated using the formula:

    .. math:: h_n^(2)(z) = j_{n} - i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn2, hankel2, expand_func, jn, yn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn2(nu, z)))
    jn(nu, z) - I*yn(nu, z)
    >>> print(expand_func(hn2(0, z)))
    sin(z)/z + I*cos(z)/z
    >>> print(expand_func(hn2(1, z)))
    I*sin(z)/z - cos(z)/z + sin(z)/z**2 + I*cos(z)/z**2
    >>> hn2(nu, z).rewrite(hankel2)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel2(nu, z)/2
    >>> hn2(nu, z).rewrite(jn)
    -(-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn2(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) - I*yn(nu, z)

    See Also
    ========

    hn1, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = -S.One  # 指定符号常量，用于计算第二类球谐汉克尔函数

    @assume_integer_order
    def _eval_rewrite_as_hankel2(self, nu, z, **kwargs):
        """
        Rewrite the function as a form involving the second kind Hankel function.

        Parameters
        ==========

        nu : integer
            Order of the spherical Hankel function.

        z : symbol
            Symbolic variable.

        Returns
        =======

        expression
            Rewritten expression using the second kind Hankel function.
        """
        return sqrt(pi/(2*z))*hankel2(nu, z)


def jn_zeros(n, k, method="sympy", dps=15):
    """
    Zeros of the spherical Bessel function of the first kind.

    Explanation
    ===========

    This returns an array of zeros of $jn$ up to the $k$-th zero.

    * method = "sympy": uses `mpmath.besseljzero
      <https://mpmath.org/doc/current/functions/bessel.html#mpmath.besseljzero>`_
    * method = "scipy": uses the
      `SciPy's sph_jn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html>`_
      and
      `newton <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html>`_
      to find all
      roots, which is faster than computing the zeros using a general
      numerical solver, but it requires SciPy and only works with low
      precision floating point numbers. (The function used with
      method="sympy" is a recent addition to mpmath; before that a general
      solver was used.)

    Examples
    ========

    >>> from sympy import jn_zeros
    >>> jn_zeros(2, 4, dps=5)
    [5.7635, 9.095, 12.323, 15.515]

    See Also
    ========

    jn, yn, besselj, besselk, bessely

    Parameters
    ==========

    n : integer
        Order of Bessel function

    k : integer
        Number of zeros to return
    """

    from math import pi as math_pi  # 导入数学常数 pi

    if method == "sympy":
        from mpmath import besseljzero
        from mpmath.libmp.libmpf import dps_to_prec
        prec = dps_to_prec(dps)  # 根据给定的精度转换为 mpmath 中的精度
        return [Expr._from_mpmath(besseljzero(S(n + 0.5)._to_mpmath(prec),
                                              int(l)), prec)
                for l in range(1, k + 1)]
    # 如果使用 "scipy" 方法，则导入必要的函数和模块
    elif method == "scipy":
        from scipy.optimize import newton
        try:
            # 尝试导入 scipy 的 spherical_jn 函数，用于计算球面贝塞尔函数
            from scipy.special import spherical_jn
            # 定义函数 f，使用 spherical_jn 计算球面贝塞尔函数的值
            f = lambda x: spherical_jn(n, x)
        except ImportError:
            # 如果 ImportError，可能是旧版本 scipy，使用 sph_jn 函数
            from scipy.special import sph_jn
            # 定义函数 f，使用 sph_jn 计算球面贝塞尔函数的值，取返回的结果的最后一个元素
            f = lambda x: sph_jn(n, x)[0][-1]
    else:
        # 如果 method 不是 "scipy"，则抛出未实现错误
        raise NotImplementedError("Unknown method.")

    # 定义求解器函数 solver，根据给定的函数 f 和初始位置 x 求解方程
    def solver(f, x):
        if method == "scipy":
            # 如果 method 是 "scipy"，使用 newton 函数进行数值求解
            root = newton(f, x)
        else:
            # 如果 method 不是 "scipy"，抛出未实现错误
            raise NotImplementedError("Unknown method.")
        return root

    # 需要近似第一个根的位置：
    root = n + math_pi
    # 精确确定第一个根的位置：
    root = solver(f, root)
    roots = [root]
    for i in range(k - 1):
        # 使用上一个根加上 π 估计下一个根的位置：
        root = solver(f, root + math_pi)
        roots.append(root)
    return roots
class AiryBase(Function):
    """
    Abstract base class for Airy functions.

    This class is meant to reduce code duplication.

    """

    def _eval_conjugate(self):
        # 返回调用函数的共轭值
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        # 返回参数是否是扩展实数
        return self.args[0].is_extended_real

    def as_real_imag(self, deep=True, **hints):
        # 将自身表示为实部和虚部
        z = self.args[0]
        zc = z.conjugate()
        f = self.func
        u = (f(z)+f(zc))/2  # 实部
        v = I*(f(zc)-f(z))/2  # 虚部
        return u, v

    def _eval_expand_complex(self, deep=True, **hints):
        # 扩展复数运算，返回实部加上虚部乘虚数单位
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I


class airyai(AiryBase):
    r"""
    The Airy function $\operatorname{Ai}$ of the first kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Ai}(z) := \frac{1}{\pi}
        \int_0^\infty \cos\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyai
    >>> from sympy.abc import z

    >>> airyai(z)
    airyai(z)

    Several special values are known:

    >>> airyai(0)
    3**(1/3)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airyai(oo)
    0
    >>> airyai(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyai(z))
    airyai(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyai(z), z)
    airyaiprime(z)
    >>> diff(airyai(z), z, 2)
    z*airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyai(z), z, 0, 3)
    3**(5/6)*gamma(1/3)/(6*pi) - 3**(1/6)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyai(-2).evalf(50)
    0.22740742820168557599192443603787379946077222541710

    Rewrite $\operatorname{Ai}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyai(z).rewrite(hyper)
    -3**(2/3)*z*hyper((), (4/3,), z**3/9)/(3*gamma(1/3)) + 3**(1/3)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    # 定义一个静态方法，计算给定参数的表达式值
    def eval(cls, arg):
        # 检查参数是否是一个数值对象
        if arg.is_Number:
            # 如果参数是 NaN，则返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 如果参数是正无穷或负无穷，返回零
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            # 如果参数是零，返回特定计算结果
            elif arg.is_zero:
                return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))
        # 如果参数是零，返回特定计算结果（针对不属于 is_Number 的情况）
        if arg.is_zero:
            return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))

    # 定义一个方法，返回关于对象第一个参数的导数
    def fdiff(self, argindex=1):
        # 如果参数索引是 1，返回某个函数的导数
        if argindex == 1:
            return airyaiprime(self.args[0])
        else:
            # 如果参数索引不是 1，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 定义一个静态方法，计算泰勒级数的某一项
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # 如果 n 小于 0，返回零
        if n < 0:
            return S.Zero
        else:
            # 将 x 转换为符号表达式
            x = sympify(x)
            # 如果有多于一个先前项，使用递归公式计算泰勒级数的当前项
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return ((cbrt(3)*x)**(-n)*(cbrt(3)*x)**(n + 1)*sin(pi*(n*Rational(2, 3) + Rational(4, 3)))*factorial(n) *
                        gamma(n/3 + Rational(2, 3))/(sin(pi*(n*Rational(2, 3) + Rational(2, 3)))*factorial(n + 1)*gamma(n/3 + Rational(1, 3))) * p)
            else:
                # 使用公式计算泰勒级数的当前项
                return (S.One/(3**Rational(2, 3)*pi) * gamma((n+S.One)/S(3)) * sin(Rational(2, 3)*pi*(n+S.One)) /
                        factorial(n) * (cbrt(3)*x)**n)

    # 定义一个方法，将对象的求值重写为 Bessel 函数的表达式
    def _eval_rewrite_as_besselj(self, z, **kwargs):
        # 定义常量值
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        # 如果 z 的实部为负，返回 Bessel 函数的计算结果
        if re(z).is_negative:
            return ot*sqrt(-z) * (besselj(-ot, tt*a) + besselj(ot, tt*a))

    # 定义一个方法，将对象的求值重写为 Bessel 函数的表达式
    def _eval_rewrite_as_besseli(self, z, **kwargs):
        # 定义常量值
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        # 如果 z 的实部为正，返回 Bessel 函数的计算结果
        if re(z).is_positive:
            return ot*sqrt(z) * (besseli(-ot, tt*a) - besseli(ot, tt*a))
        else:
            # 如果 z 的实部不为正，返回 Bessel 函数的计算结果
            return ot*(Pow(a, ot)*besseli(-ot, tt*a) - z*Pow(a, -ot)*besseli(ot, tt*a))

    # 定义一个方法，将对象的求值重写为超几何函数的表达式
    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 定义常量值
        pf1 = S.One / (3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = z / (root(3, 3)*gamma(Rational(1, 3)))
        # 返回超几何函数的计算结果
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) - pf2 * hyper([], [Rational(4, 3)], z**3/9)
    # 定义一个用于评估和扩展函数的私有方法，接受关键字参数 hints
    def _eval_expand_func(self, **hints):
        # 获取函数参数列表中的第一个参数
        arg = self.args[0]
        # 获取参数中的自由符号集合
        symbs = arg.free_symbols

        # 如果自由符号集合中只有一个符号
        if len(symbs) == 1:
            # 弹出这个符号，将其赋值给变量 z
            z = symbs.pop()
            # 创建不包含 z 的通配符对象
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            # 尝试匹配参数 arg，查找形式为 c*(d*z**n)**m 的模式
            M = arg.match(c*(d*z**n)**m)
            # 如果匹配成功
            if M is not None:
                # 从匹配结果中提取 m 的值
                m = M[m]
                # 根据给定的文档 03.05.16.0001.01 进行变换
                # 参考链接：https://functions.wolfram.com/Bessel-TypeFunctions/AiryAi/16/01/01/0001/
                # 如果 3*m 是整数
                if (3*m).is_integer:
                    # 从匹配结果中提取 c、d、n 的值
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    # 计算变换后的表达式 pf
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    # 计算新的参数值 newarg
                    newarg = c * d**m * z**(m*n)
                    # 返回结果，根据公式计算 Airy 函数的线性组合
                    return S.Half * ((pf + S.One)*airyai(newarg) - (pf - S.One)/sqrt(3)*airybi(newarg))
class airybi(AiryBase):
    r"""
    The Airy function $\operatorname{Bi}$ of the second kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Bi}(z) := \frac{1}{\pi}
                 \int_0^\infty
                   \exp\left(-\frac{t^3}{3} + z t\right)
                   + \sin\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybi
    >>> from sympy.abc import z

    >>> airybi(z)
    airybi(z)

    Several special values are known:

    >>> airybi(0)
    3**(5/6)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airybi(oo)
    oo
    >>> airybi(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybi(z))
    airybi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybi(z), z)
    airybiprime(z)
    >>> diff(airybi(z), z, 2)
    z*airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybi(z), z, 0, 3)
    3**(1/3)*gamma(1/3)/(2*pi) + 3**(2/3)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybi(-2).evalf(50)
    -0.41230258795639848808323405461146104203453483447240

    Rewrite $\operatorname{Bi}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybi(z).rewrite(hyper)
    3**(1/6)*z*hyper((), (4/3,), z**3/9)/gamma(1/3) + 3**(5/6)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    # 设置 nargs 为 1，表示该函数接收一个参数
    nargs = 1
    # 设置 unbranched 为 True，表示该函数在复平面上是单值函数

    unbranched = True

    # 类方法，用于评估函数在给定参数处的值
    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数值
        if arg.is_Number:
            # 处理特殊的数值情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                # 当参数为零时返回的特殊值
                return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

        # 如果参数是零，则返回特定的值
        if arg.is_zero:
            return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

    # 实例方法，计算该函数的偏导数
    def fdiff(self, argindex=1):
        # 当指定的参数索引为 1 时，返回一阶导数函数 airybiprime(z)
        if argindex == 1:
            return airybiprime(self.args[0])
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)
    @staticmethod
    @cacheit
    # 静态方法装饰器，将方法标记为静态方法，并且应用缓存装饰器
    def taylor_term(n, x, *previous_terms):
        # 计算泰勒项函数，返回第n项的值，基于给定的x和前面的若干项
        if n < 0:
            # 如果n小于0，则返回零
            return S.Zero
        else:
            x = sympify(x)
            # 符号化x，确保x是符号化对象
            if len(previous_terms) > 1:
                # 如果previous_terms的长度大于1
                p = previous_terms[-1]
                # 取出previous_terms的最后一项作为p
                return (cbrt(3)*x * Abs(sin(Rational(2, 3)*pi*(n + S.One))) * factorial((n - S.One)/S(3)) /
                        ((n + S.One) * Abs(cos(Rational(2, 3)*pi*(n + S.Half))) * factorial((n - 2)/S(3))) * p)
                # 返回根据前一项计算得到的泰勒项值
            else:
                # 否则，如果previous_terms长度不大于1
                return (S.One/(root(3, 6)*pi) * gamma((n + S.One)/S(3)) * Abs(sin(Rational(2, 3)*pi*(n + S.One))) /
                        factorial(n) * (cbrt(3)*x)**n)
                # 返回计算得到的泰勒项值

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        # 重写函数，表示为贝塞尔函数J形式，给定参数z和其他可选参数
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        # 计算a为-z的3/2次方
        if re(z).is_negative:
            # 如果z的实部为负数
            return sqrt(-z/3) * (besselj(-ot, tt*a) - besselj(ot, tt*a))
            # 返回计算得到的表达式

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        # 重写函数，表示为贝塞尔函数I形式，给定参数z和其他可选参数
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        # 计算a为z的3/2次方
        if re(z).is_positive:
            # 如果z的实部为正数
            return sqrt(z)/sqrt(3) * (besseli(-ot, tt*a) + besseli(ot, tt*a))
            # 返回计算得到的表达式
        else:
            b = Pow(a, ot)
            c = Pow(a, -ot)
            # 计算b为a的ot次方，c为a的-ot次方
            return sqrt(ot)*(b*besseli(-ot, tt*a) + z*c*besseli(ot, tt*a))
            # 返回计算得到的表达式

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 重写函数，表示为超几何函数形式，给定参数z和其他可选参数
        pf1 = S.One / (root(3, 6)*gamma(Rational(2, 3)))
        pf2 = z*root(3, 6) / gamma(Rational(1, 3))
        # 计算pf1和pf2的值
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) + pf2 * hyper([], [Rational(4, 3)], z**3/9)
        # 返回计算得到的表达式

    def _eval_expand_func(self, **hints):
        # 重写函数，扩展为特定形式，基于给定的提示参数
        arg = self.args[0]
        symbs = arg.free_symbols
        # 取出函数的第一个参数，查找自由符号

        if len(symbs) == 1:
            # 如果自由符号的数量为1
            z = symbs.pop()
            # 弹出自由符号集合的一个元素作为z
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            # 定义Wild匹配模式，排除z以外的变量
            M = arg.match(c*(d*z**n)**m)
            # 对参数arg进行匹配
            if M is not None:
                # 如果匹配成功
                m = M[m]
                # 取出匹配结果中的m值
                # The transformation is given by 03.06.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBi/16/01/01/0001/
                # 给出的转换对应于03.06.16.0001.01
                if (3*m).is_integer:
                    # 如果3*m是整数
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    # 取出匹配结果中的c、d、n值
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    # 计算pf的值
                    newarg = c * d**m * z**(m*n)
                    # 计算newarg的值
                    return S.Half * (sqrt(3)*(S.One - pf)*airyai(newarg) + (S.One + pf)*airybi(newarg))
                    # 返回计算得到的表达式
class airyaiprime(AiryBase):
    r"""
    The derivative $\operatorname{Ai}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Ai}^\prime(z) := \frac{\mathrm{d} \operatorname{Ai}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyaiprime
    >>> from sympy.abc import z

    >>> airyaiprime(z)
    airyaiprime(z)

    Several special values are known:

    >>> airyaiprime(0)
    -3**(2/3)/(3*gamma(1/3))
    >>> from sympy import oo
    >>> airyaiprime(oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyaiprime(z))
    airyaiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyaiprime(z), z)
    z*airyai(z)
    >>> diff(airyaiprime(z), z, 2)
    z*airyaiprime(z) + airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyaiprime(z), z, 0, 3)
    -3**(2/3)/(3*gamma(1/3)) + 3**(1/3)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyaiprime(-2).evalf(50)
    0.61825902074169104140626429133247528291577794512415

    Rewrite $\operatorname{Ai}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyaiprime(z).rewrite(hyper)
    3**(1/3)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) - 3**(2/3)*hyper((), (1/3,), z**3/9)/(3*gamma(1/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    # 类属性，指定该函数的参数个数
    nargs = 1
    # 是否为无分支函数的标志
    unbranched = True

    @classmethod
    def eval(cls, arg):
        # 如果参数是一个数值
        if arg.is_Number:
            # 处理特殊情况
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero

        # 如果参数是零
        if arg.is_zero:
            # 返回特定值
            return S.NegativeOne / (3**Rational(1, 3) * gamma(Rational(1, 3)))

    # 求导函数，用于计算关于参数的偏导数
    def fdiff(self, argindex=1):
        # 当参数索引为1时
        if argindex == 1:
            # 返回函数关于参数的导数乘以函数自身在该参数值处的值
            return self.args[0]*airyai(self.args[0])
        else:
            # 报错：参数索引超出范围
            raise ArgumentIndexError(self, argindex)

    # 计算函数在指定精度下的数值近似
    def _eval_evalf(self, prec):
        # 将参数转换为 mpmath 的表示形式
        z = self.args[0]._to_mpmath(prec)
        # 设置工作精度
        with workprec(prec):
            # 调用 mpmath 中的函数计算 Airy 函数的导数
            res = mp.airyai(z, derivative=1)
        # 返回数值近似的表达式
        return Expr._from_mpmath(res, prec)
    # 将当前对象重写为 Bessel J 函数的表达式
    def _eval_rewrite_as_besselj(self, z, **kwargs):
        # 设置常量 tt 为 2/3
        tt = Rational(2, 3)
        # 计算 -z 的 3/2 次幂
        a = Pow(-z, Rational(3, 2))
        # 如果 z 的实部为负数，则返回重新表达的结果
        if re(z).is_negative:
            return z/3 * (besselj(-tt, tt*a) - besselj(tt, tt*a))

    # 将当前对象重写为 Bessel I 函数的表达式
    def _eval_rewrite_as_besseli(self, z, **kwargs):
        # 设置常量 ot 为 1/3，tt 为 2/3
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        # 计算 tt * z 的 3/2 次幂
        a = tt * Pow(z, Rational(3, 2))
        # 如果 z 的实部为正数，则返回重新表达的结果
        if re(z).is_positive:
            return z/3 * (besseli(tt, a) - besseli(-tt, a))
        else:
            # 如果 z 的实部不为正数，则重新计算一些值，并返回另一种表达式
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return ot * (z**2*c*besseli(tt, tt*a) - b*besseli(-ot, tt*a))

    # 将当前对象重写为超几何函数的表达式
    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 计算 pf1 和 pf2 的值，分别表示超几何函数的两部分
        pf1 = z**2 / (2*3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = 1 / (root(3, 3)*gamma(Rational(1, 3)))
        # 返回超几何函数的重新表达式结果
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) - pf2 * hyper([], [Rational(1, 3)], z**3/9)

    # 将当前对象展开为函数的表达式
    def _eval_expand_func(self, **hints):
        # 获取参数列表中的第一个参数
        arg = self.args[0]
        # 获取参数中的自由符号集合
        symbs = arg.free_symbols

        # 如果自由符号集合中只有一个符号
        if len(symbs) == 1:
            # 弹出集合中的唯一符号作为 z
            z = symbs.pop()
            # 设置 Wildcards
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            # 尝试匹配参数 arg
            M = arg.match(c*(d*z**n)**m)
            # 如果匹配成功
            if M is not None:
                # 获取匹配结果中的 m、c、d、n 值
                m = M[m]
                c = M[c]
                d = M[d]
                n = M[n]
                # 计算 pf 的值
                pf = (d**m * z**(n*m)) / (d * z**n)**m
                # 计算新的参数值
                newarg = c * d**m * z**(n*m)
                # 返回重新表达的函数表达式
                return S.Half * ((pf + S.One)*airyaiprime(newarg) + (pf - S.One)/sqrt(3)*airybiprime(newarg))
class airybiprime(AiryBase):
    r"""
    The derivative $\operatorname{Bi}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Bi}^\prime(z) := \frac{\mathrm{d} \operatorname{Bi}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybiprime
    >>> from sympy.abc import z

    >>> airybiprime(z)
    airybiprime(z)

    Several special values are known:

    >>> airybiprime(0)
    3**(1/6)/gamma(1/3)
    >>> from sympy import oo
    >>> airybiprime(oo)
    oo
    >>> airybiprime(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybiprime(z))
    airybiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybiprime(z), z)
    z*airybi(z)
    >>> diff(airybiprime(z), z, 2)
    z*airybiprime(z) + airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybiprime(z), z, 0, 3)
    3**(1/6)/gamma(1/3) + 3**(5/6)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybiprime(-2).evalf(50)
    0.27879516692116952268509756941098324140300059345163

    Rewrite $\operatorname{Bi}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybiprime(z).rewrite(hyper)
    3**(5/6)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) + 3**(1/6)*hyper((), (1/3,), z**3/9)/gamma(1/3)

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    # 类属性：指定此函数的参数个数为1
    nargs = 1
    # 类属性：函数是无分支的
    unbranched = True

    # 类方法：评估函数值
    @classmethod
    def eval(cls, arg):
        # 检查参数是否是数值
        if arg.is_Number:
            # 如果参数是 NaN，返回 NaN
            if arg is S.NaN:
                return S.NaN
            # 如果参数是正无穷，返回正无穷
            elif arg is S.Infinity:
                return S.Infinity
            # 如果参数是负无穷，返回零
            elif arg is S.NegativeInfinity:
                return S.Zero
            # 如果参数是零，返回常数表达式
            elif arg.is_zero:
                return 3**Rational(1, 6) / gamma(Rational(1, 3))

        # 如果参数是零，返回常数表达式
        if arg.is_zero:
            return 3**Rational(1, 6) / gamma(Rational(1, 3))

    # 实例方法：计算偏导数
    def fdiff(self, argindex=1):
        # 检查参数索引是否为1
        if argindex == 1:
            # 返回计算结果
            return self.args[0]*airybi(self.args[0])
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)
    # 定义函数 _eval_evalf，用于对表达式进行求值
    def _eval_evalf(self, prec):
        # 将第一个参数转换为精确度为 prec 的多精度数
        z = self.args[0]._to_mpmath(prec)
        # 设置工作精度为 prec，在此精度下计算 Airy 函数的导数
        with workprec(prec):
            res = mp.airybi(z, derivative=1)
        # 将计算结果转换为表达式对象，并返回
        return Expr._from_mpmath(res, prec)

    # 重新定义函数为 Bessel 函数形式
    def _eval_rewrite_as_besselj(self, z, **kwargs):
        # 计算系数 tt
        tt = Rational(2, 3)
        # 计算参数 a
        a = tt * Pow(-z, Rational(3, 2))
        # 如果 z 的实部为负数，返回相应的表达式
        if re(z).is_negative:
            return -z/sqrt(3) * (besselj(-tt, a) + besselj(tt, a))

    # 重新定义函数为 Bessel 函数形式
    def _eval_rewrite_as_besseli(self, z, **kwargs):
        # 计算系数 ot 和 tt
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        # 计算参数 a
        a = tt * Pow(z, Rational(3, 2))
        # 如果 z 的实部为正数，返回相应的表达式
        if re(z).is_positive:
            return z/sqrt(3) * (besseli(-tt, a) + besseli(tt, a))
        else:
            # 重新计算参数 a, b, c
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            # 返回包含 Bessel 函数的表达式
            return sqrt(ot) * (b*besseli(-tt, tt*a) + z**2*c*besseli(tt, tt*a))

    # 重新定义函数为超几何函数形式
    def _eval_rewrite_as_hyper(self, z, **kwargs):
        # 计算两个部分的前缀系数
        pf1 = z**2 / (2*root(3, 6)*gamma(Rational(2, 3)))
        pf2 = root(3, 6) / gamma(Rational(1, 3))
        # 返回用超几何函数表达的结果
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) + pf2 * hyper([], [Rational(1, 3)], z**3/9)

    # 扩展函数以进行功能展开
    def _eval_expand_func(self, **hints):
        # 获取函数的参数
        arg = self.args[0]
        symbs = arg.free_symbols

        # 如果参数只包含一个符号
        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            # 匹配参数中的模式 M
            M = arg.match(c*(d*z**n)**m)
            # 如果匹配成功
            if M is not None:
                m = M[m]
                # 根据给定的公式进行变换，注意公式可能存在错误
                # 参考：https://functions.wolfram.com/Bessel-TypeFunctions/AiryBiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    # 计算转换后的新参数和返回结果
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * (sqrt(3)*(pf - S.One)*airyaiprime(newarg) + (pf + S.One)*airybiprime(newarg))
class marcumq(Function):
    r"""
    The Marcum Q-function.

    Explanation
    ===========

    The Marcum Q-function is defined by the meromorphic continuation of

    .. math::
        Q_m(a, b) = a^{- m + 1} \int_{b}^{\infty} x^{m} e^{- \frac{a^{2}}{2} - \frac{x^{2}}{2}} I_{m - 1}\left(a x\right)\, dx

    Examples
    ========

    >>> from sympy import marcumq
    >>> from sympy.abc import m, a, b
    >>> marcumq(m, a, b)
    marcumq(m, a, b)

    Special values:

    >>> marcumq(m, 0, b)
    uppergamma(m, b**2/2)/gamma(m)
    >>> marcumq(0, 0, 0)
    0
    >>> marcumq(0, a, 0)
    1 - exp(-a**2/2)
    >>> marcumq(1, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2
    >>> marcumq(2, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    Differentiation with respect to $a$ and $b$ is supported:

    >>> from sympy import diff
    >>> diff(marcumq(m, a, b), a)
    a*(-marcumq(m, a, b) + marcumq(m + 1, a, b))
    >>> diff(marcumq(m, a, b), b)
    -a**(1 - m)*b**m*exp(-a**2/2 - b**2/2)*besseli(m - 1, a*b)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Marcum_Q-function
    .. [2] https://mathworld.wolfram.com/MarcumQ-Function.html

    """

    @classmethod
    def eval(cls, m, a, b):
        # 处理特殊情况：a = 0
        if a is S.Zero:
            # 当 m = 0 且 b = 0 时返回 0，否则返回上不完全伽玛函数与伽玛函数的比值
            if m is S.Zero and b is S.Zero:
                return S.Zero
            return uppergamma(m, b**2 * S.Half) / gamma(m)

        # 处理特殊情况：m = 0 且 b = 0
        if m is S.Zero and b is S.Zero:
            return 1 - 1 / exp(a**2 * S.Half)

        # 处理特殊情况：a = b
        if a == b:
            # 当 m = 1 时返回特定表达式，当 m = 2 时返回特定表达式
            if m is S.One:
                return (1 + exp(-a**2) * besseli(0, a**2))*S.Half
            if m == 2:
                return S.Half + S.Half * exp(-a**2) * besseli(0, a**2) + exp(-a**2) * besseli(1, a**2)

        # 处理特殊情况：a = 0
        if a.is_zero:
            # 当 m = 0 且 b = 0 时返回 0，否则返回上不完全伽玛函数与伽玛函数的比值
            if m.is_zero and b.is_zero:
                return S.Zero
            return uppergamma(m, b**2*S.Half) / gamma(m)

        # 处理特殊情况：m = 0 且 b = 0
        if m.is_zero and b.is_zero:
            return 1 - 1 / exp(a**2*S.Half)

    def fdiff(self, argindex=2):
        # 返回关于参数索引的偏导数表达式
        m, a, b = self.args
        if argindex == 2:
            return a * (-marcumq(m, a, b) + marcumq(1+m, a, b))
        elif argindex == 3:
            return (-b**m / a**(m-1)) * exp(-(a**2 + b**2)/2) * besseli(m-1, a*b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, m, a, b, **kwargs):
        # 重写为积分形式
        from sympy.integrals.integrals import Integral
        x = kwargs.get('x', Dummy(uniquely_named_symbol('x').name))
        return a ** (1 - m) * \
               Integral(x**m * exp(-(x**2 + a**2)/2) * besseli(m-1, a*x), [x, b, S.Infinity])

    def _eval_rewrite_as_Sum(self, m, a, b, **kwargs):
        # 重写为求和形式
        from sympy.concrete.summations import Sum
        k = kwargs.get('k', Dummy('k'))
        return exp(-(a**2 + b**2) / 2) * Sum((a/b)**k * besseli(k, a*b), [k, 1-m, S.Infinity])
    # 当参数 a 等于 b 时执行以下操作
    def _eval_rewrite_as_besseli(self, m, a, b, **kwargs):
        if a == b:
            # 当 m 等于 1 时，返回以下表达式的结果
            if m == 1:
                return (1 + exp(-a**2) * besseli(0, a**2)) / 2
            # 当 m 是整数且大于等于 2 时，执行以下操作
            if m.is_Integer and m >= 2:
                # 计算从 1 到 m-1 的贝塞尔函数 besseli(i, a**2) 的和
                s = sum(besseli(i, a**2) for i in range(1, m))
                # 返回以下表达式的结果
                return S.Half + exp(-a**2) * besseli(0, a**2) / 2 + exp(-a**2) * s

    # 检查对象是否为零
    def _eval_is_zero(self):
        # 检查所有参数是否都为零
        if all(arg.is_zero for arg in self.args):
            return True
```