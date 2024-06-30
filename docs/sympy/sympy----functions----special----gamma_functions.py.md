# `D:\src\scipysrc\sympy\sympy\functions\special\gamma_functions.py`

```
from math import prod  # 导入 math 模块中的 prod 函数，用于计算序列的乘积

from sympy.core import Add, S, Dummy, expand_func  # 导入 sympy.core 模块中的 Add, S, Dummy, expand_func 等类和函数
from sympy.core.expr import Expr  # 导入 sympy.core.expr 模块中的 Expr 类
from sympy.core.function import Function, ArgumentIndexError, PoleError  # 导入 sympy.core.function 模块中的 Function, ArgumentIndexError, PoleError 等类和异常
from sympy.core.logic import fuzzy_and, fuzzy_not  # 导入 sympy.core.logic 模块中的 fuzzy_and 和 fuzzy_not 函数
from sympy.core.numbers import Rational, pi, oo, I  # 导入 sympy.core.numbers 模块中的 Rational, pi, oo, I 等常量
from sympy.core.power import Pow  # 导入 sympy.core.power 模块中的 Pow 类
from sympy.functions.special.zeta_functions import zeta  # 导入 sympy.functions.special.zeta_functions 模块中的 zeta 函数
from sympy.functions.special.error_functions import erf, erfc, Ei  # 导入 sympy.functions.special.error_functions 模块中的 erf, erfc, Ei 函数
from sympy.functions.elementary.complexes import re, unpolarify  # 导入 sympy.functions.elementary.complexes 模块中的 re, unpolarify 函数
from sympy.functions.elementary.exponential import exp, log  # 导入 sympy.functions.elementary.exponential 模块中的 exp, log 函数
from sympy.functions.elementary.integers import ceiling, floor  # 导入 sympy.functions.elementary.integers 模块中的 ceiling, floor 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sympy.functions.elementary.miscellaneous 模块中的 sqrt 函数
from sympy.functions.elementary.trigonometric import sin, cos, cot  # 导入 sympy.functions.elementary.trigonometric 模块中的 sin, cos, cot 函数
from sympy.functions.combinatorial.numbers import bernoulli, harmonic  # 导入 sympy.functions.combinatorial.numbers 模块中的 bernoulli, harmonic 函数
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial  # 导入 sympy.functions.combinatorial.factorials 模块中的 factorial, rf, RisingFactorial 函数
from sympy.utilities.misc import as_int  # 导入 sympy.utilities.misc 模块中的 as_int 函数

from mpmath import mp, workprec  # 导入 mpmath 模块中的 mp, workprec 函数
from mpmath.libmp.libmpf import prec_to_dps  # 导入 mpmath.libmp.libmpf 模块中的 prec_to_dps 函数

def intlike(n):
    try:
        as_int(n, strict=False)  # 尝试将参数 n 转换为整数，允许非严格模式
        return True  # 如果成功转换为整数，则返回 True
    except ValueError:
        return False  # 如果出现 ValueError 异常，则返回 False

###############################################################################
############################ COMPLETE GAMMA FUNCTION ##########################
###############################################################################

class gamma(Function):
    r"""
    The gamma function

    .. math::
        \Gamma(x) := \int^{\infty}_{0} t^{x-1} e^{-t} \mathrm{d}t.

    Explanation
    ===========

    The ``gamma`` function implements the function which passes through the
    values of the factorial function (i.e., $\Gamma(n) = (n - 1)!$ when n is
    an integer). More generally, $\Gamma(z)$ is defined in the whole complex
    plane except at the negative integers where there are simple poles.

    Examples
    ========

    >>> from sympy import S, I, pi, gamma
    >>> from sympy.abc import x

    Several special values are known:

    >>> gamma(1)
    1
    >>> gamma(4)
    6
    >>> gamma(S(3)/2)
    sqrt(pi)/2

    The ``gamma`` function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(gamma(x))
    gamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(gamma(x), x)
    gamma(x)*polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(gamma(x), x, 0, 3)
    1/x - EulerGamma + x*(EulerGamma**2/2 + pi**2/12) + x**2*(-EulerGamma*pi**2/12 - zeta(3)/3 - EulerGamma**3/6) + O(x**3)

    We can numerically evaluate the ``gamma`` function to arbitrary precision
    on the whole complex plane:

    >>> gamma(pi).evalf(40)
    2.288037795340032417959588909060233922890
    >>> gamma(1+I).evalf(20)
    0.49801566811835604271 - 0.15494982830181068512*I

    See Also
    ========

    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/GammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma/


    unbranched = True
    _singularities = (S.ComplexInfinity,)

    # 定义函数的导数
    def fdiff(self, argindex=1):
        if argindex == 1:
            # 返回函数乘以多重对数函数的导数
            return self.func(self.args[0])*polygamma(0, self.args[0])
        else:
            # 如果参数索引不是 1，抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 类方法，用于计算函数的特定输入值的值
    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is oo:
                return oo
            elif intlike(arg):
                if arg.is_positive:
                    # 如果参数是正数，返回阶乘
                    return factorial(arg - 1)
                else:
                    # 如果参数不是正数，返回复数无穷大
                    return S.ComplexInfinity
            elif arg.is_Rational:
                if arg.q == 2:
                    n = abs(arg.p) // arg.q

                    if arg.is_positive:
                        k, coeff = n, S.One
                    else:
                        n = k = n + 1

                        if n & 1 == 0:
                            coeff = S.One
                        else:
                            coeff = S.NegativeOne

                    coeff *= prod(range(3, 2*k, 2))

                    if arg.is_positive:
                        return coeff*sqrt(pi) / 2**n
                    else:
                        return 2**n*sqrt(pi) / coeff

    # 用于展开函数的私有方法
    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        if arg.is_Rational:
            if abs(arg.p) > arg.q:
                x = Dummy('x')
                n = arg.p // arg.q
                p = arg.p - n*arg.q
                return self.func(x + n)._eval_expand_func().subs(x, Rational(p, arg.q))

        if arg.is_Add:
            coeff, tail = arg.as_coeff_add()
            if coeff and coeff.q != 1:
                intpart = floor(coeff)
                tail = (coeff - intpart,) + tail
                coeff = intpart
            tail = arg._new_rawargs(*tail, reeval=False)
            return self.func(tail)*RisingFactorial(tail, coeff)

        return self.func(*self.args)

    # 用于计算共轭的私有方法
    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    # 检查函数是否为实数的私有方法
    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonpositive and x.is_integer:
            return False
        if intlike(x) and x <= 0:
            return False
        if x.is_positive or x.is_noninteger:
            return True

    # 检查函数是否为正数的私有方法
    def _eval_is_positive(self):
        x = self.args[0]
        if x.is_positive:
            return True
        elif x.is_noninteger:
            return floor(x).is_even
    # 返回 z 的对数伽马函数的指数函数
    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(loggamma(z))
    
    # 返回 z 的阶乘的值减一
    def _eval_rewrite_as_factorial(self, z, **kwargs):
        return factorial(z - 1)
    
    # 对函数进行渐近级数展开以便于处理
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 获取参数中的第一个表达式的 x=0 时的极限
        x0 = self.args[0].limit(x, 0)
        # 如果 x0 不是整数或者大于 0，则调用父类的 nseries 方法
        if not (x0.is_Integer and x0 <= 0):
            return super()._eval_nseries(x, n, logx)
        # 计算新的变量 t，并返回 nseries 结果
        t = self.args[0] - x0
        return (self.func(t + 1)/rf(self.args[0], -x0 + 1))._eval_nseries(x, n, logx)
    
    # 返回作为主导项的表达式的估算值
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        # 获取在 x=0 处的表达式值
        x0 = arg.subs(x, 0)
    
        # 如果 x0 是负整数，则返回对应的表达式值的估算结果
        if x0.is_integer and x0.is_nonpositive:
            n = -x0
            res = S.NegativeOne**n/self.func(n + 1)
            return res/(arg + n).as_leading_term(x)
        # 如果 x0 不是无穷大，则返回在 x0 处的函数值
        elif not x0.is_infinite:
            return self.func(x0)
        # 如果 x0 是无穷大，则引发极点错误
        raise PoleError()
    # 定义 lowergamma 类，继承自 sympy 的 Function 类
    class lowergamma(Function):
        """
        The lower incomplete gamma function.

        Explanation
        ===========

        It can be defined as the meromorphic continuation of

        .. math::
            \gamma(s, x) := \int_0^x t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \Gamma(s, x).

        This can be shown to be the same as

        .. math::
            \gamma(s, x) = \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

        where ${}_1F_1$ is the (confluent) hypergeometric function.

        Examples
        ========

        >>> from sympy import lowergamma, S
        >>> from sympy.abc import s, x
        >>> lowergamma(s, x)
        lowergamma(s, x)
        >>> lowergamma(3, x)
        -2*(x**2/2 + x + 1)*exp(-x) + 2
        >>> lowergamma(-S(1)/2, x)
        -2*sqrt(pi)*erf(sqrt(x)) - 2*exp(-x)/sqrt(x)

        See Also
        ========

        gamma: Gamma function.
        uppergamma: Upper incomplete gamma function.
        polygamma: Polygamma function.
        loggamma: Log Gamma function.
        digamma: Digamma function.
        trigamma: Trigamma function.
        sympy.functions.special.beta_functions.beta: Euler Beta function.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Lower_incomplete_gamma_function
        .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
               Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
               and Mathematical Tables
        .. [3] https://dlmf.nist.gov/8
        .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
        .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/

        """

        # 定义 fdiff 方法，计算 lowergamma 对其第 argindex 个参数的偏导数
        def fdiff(self, argindex=2):
            # 导入 meijerg 函数
            from sympy.functions.special.hyper import meijerg
            # 如果 argindex 为 2
            if argindex == 2:
                # 提取 self.args 中的参数 a 和 z
                a, z = self.args
                # 返回计算结果 exp(-unpolarify(z)) * z**(a - 1)
                return exp(-unpolarify(z)) * z**(a - 1)
            # 如果 argindex 为 1
            elif argindex == 1:
                # 提取 self.args 中的参数 a 和 z
                a, z = self.args
                # 返回计算结果 gamma(a) * digamma(a) - log(z) * uppergamma(a, z) - meijerg([], [1, 1], [0, 0, a], [], z)
                return gamma(a) * digamma(a) - log(z) * uppergamma(a, z) \
                    - meijerg([], [1, 1], [0, 0, a], [], z)
            # 如果 argindex 不是 1 或 2，则抛出 ArgumentIndexError 异常
            else:
                raise ArgumentIndexError(self, argindex)

        @classmethod
    # 定义一个类方法 eval，用于计算特殊的 gamma 函数 lowergamma(a, x) 的值
    def eval(cls, a, x):
        # 如果 x 为零，则直接返回零
        if x is S.Zero:
            return S.Zero
        # 提取 x 的分支信息
        nx, n = x.extract_branch_factor()
        # 如果 a 是整数且为正数
        if a.is_integer and a.is_positive:
            # 去除 x 的极性
            nx = unpolarify(x)
            # 如果去除极性后的 x 不同于原始 x，则返回 lowergamma(a, nx)
            if nx != x:
                return lowergamma(a, nx)
        # 如果 a 是整数且为非正数
        elif a.is_integer and a.is_nonpositive:
            # 如果分支因子 n 不为零，则返回特定表达式
            if n != 0:
                return 2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + lowergamma(a, nx)
        # 如果 n 不为零
        elif n != 0:
            return exp(2*pi*I*n*a)*lowergamma(a, nx)

        # 处理特殊值
        if a.is_Number:
            # 如果 a 等于 1，则返回特定表达式
            if a is S.One:
                return S.One - exp(-x)
            # 如果 a 等于 1/2，则返回特定表达式
            elif a is S.Half:
                return sqrt(pi)*erf(sqrt(x))
            # 如果 a 是整数或者 2a 是整数
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                # 如果 b 是正数
                if b.is_positive:
                    # 如果 a 是整数，则返回特定表达式
                    if a.is_integer:
                        return factorial(b) - exp(-x) * factorial(b) * Add(*[x ** k / factorial(k) for k in range(a)])
                    # 如果 a 不是整数，则返回特定表达式
                    else:
                        return gamma(a)*(lowergamma(S.Half, x)/sqrt(pi) - exp(-x)*Add(*[x**(k - S.Half)/gamma(S.Half + k) for k in range(1, a + S.Half)]))

                # 如果 a 不是整数
                if not a.is_Integer:
                    return S.NegativeOne**(S.Half - a)*pi*erf(sqrt(x))/gamma(1 - a) + exp(-x)*Add(*[x**(k + a - 1)*gamma(a)/gamma(a + k) for k in range(1, Rational(3, 2) - a)])

        # 如果 x 是零，则返回零
        if x.is_zero:
            return S.Zero

    # 定义一个私有方法 _eval_evalf，用于在给定精度下对表达式求值
    def _eval_evalf(self, prec):
        # 如果所有参数都是数值
        if all(x.is_number for x in self.args):
            # 将第一个参数转换为指定精度的 mpmath 数字
            a = self.args[0]._to_mpmath(prec)
            # 将第二个参数转换为指定精度的 mpmath 数字
            z = self.args[1]._to_mpmath(prec)
            # 在指定精度下计算 gammainc(a, 0, z)
            with workprec(prec):
                res = mp.gammainc(a, 0, z)
            # 返回计算结果
            return Expr._from_mpmath(res, prec)
        else:
            # 如果参数不全为数值，则返回自身
            return self

    # 定义一个私有方法 _eval_conjugate，用于求解共轭值
    def _eval_conjugate(self):
        # 获取第二个参数 x
        x = self.args[1]
        # 如果 x 不是零也不是负无穷
        if x not in (S.Zero, S.NegativeInfinity):
            # 返回表达式的共轭值
            return self.func(self.args[0].conjugate(), x.conjugate())
    # 判断当前对象是否是亚黎可默尔滕函数的形式
    def _eval_is_meromorphic(self, x, a):
        # 根据 https://en.wikipedia.org/wiki/Incomplete_gamma_function#Holomorphic_extension，
        # lowergamma(s, z) = z**s*gamma(s)*gammastar(s, z)，其中gammastar(s, z)对于所有s和z都是全纯的。
        # 因此 lowergamma 的奇点是 z = 0（分支点）和 s 的非正整数值（gamma(s)的极点）。
        s, z = self.args
        # 判断参数是否均为亚黎可默尔滕函数形式
        args_merom = fuzzy_and([z._eval_is_meromorphic(x, a),
            s._eval_is_meromorphic(x, a)])
        if not args_merom:
            return args_merom
        # 在参数为整数时，检查 z 的值是否为有限
        z0 = z.subs(x, a)
        if s.is_integer:
            return fuzzy_and([s.is_positive, z0.is_finite])
        # 对 s 进行替换，检查 s 和 z 的值是否均为有限，且 z 不为零
        s0 = s.subs(x, a)
        return fuzzy_and([s0.is_finite, z0.is_finite, fuzzy_not(z0.is_zero)])

    # 将当前对象展开为幂级数
    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import O
        s, z = self.args
        # 如果第一个参数是无穷大且 z 不包含 x，则计算系数和求和表达式
        if args0[0] is oo and not z.has(x):
            coeff = z**s*exp(-z)
            sum_expr = sum(z**k/rf(s, k + 1) for k in range(n - 1))
            o = O(z**s*s**(-n))
            return coeff*sum_expr + o
        # 否则调用父类的同名方法进行处理
        return super()._eval_aseries(n, args0, x, logx)

    # 将当前对象重写为上完全伽马函数的形式
    def _eval_rewrite_as_uppergamma(self, s, x, **kwargs):
        return gamma(s) - uppergamma(s, x)

    # 将当前对象重写为指数积分函数的形式
    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        # 如果 s 是整数且非正，则保持当前对象不变
        if s.is_integer and s.is_nonpositive:
            return self
        # 否则先重写为上完全伽马函数，再重写为指数积分函数
        return self.rewrite(uppergamma).rewrite(expint)

    # 判断当前对象是否为零
    def _eval_is_zero(self):
        x = self.args[1]
        # 检查第二个参数是否为零
        if x.is_zero:
            return True
# 定义上不完全伽玛函数的类
class uppergamma(Function):
    r"""
    The upper incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \Gamma(s, x) := \int_x^\infty t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \gamma(s, x).

    where $\gamma(s, x)$ is the lower incomplete gamma function,
    :class:`lowergamma`. This can be shown to be the same as

    .. math::
        \Gamma(s, x) = \Gamma(s) - \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    The upper incomplete gamma function is also essentially equivalent to the
    generalized exponential integral:

    .. math::
        \operatorname{E}_{n}(x) = \int_{1}^{\infty}{\frac{e^{-xt}}{t^n} \, dt} = x^{n-1}\Gamma(1-n,x).

    Examples
    ========

    >>> from sympy import uppergamma, S
    >>> from sympy.abc import s, x
    >>> uppergamma(s, x)
    uppergamma(s, x)
    >>> uppergamma(3, x)
    2*(x**2/2 + x + 1)*exp(-x)
    >>> uppergamma(-S(1)/2, x)
    -2*sqrt(pi)*erfc(sqrt(x)) + 2*exp(-x)/sqrt(x)
    >>> uppergamma(-2, x)
    expint(3, x)/x**2

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Upper_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/
    .. [6] https://en.wikipedia.org/wiki/Exponential_integral#Relation_with_other_functions

    """


    # 定义对函数进行偏导数的方法
    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            # 返回关于第二个参数的偏导数
            return -exp(-unpolarify(z))*z**(a - 1)
        elif argindex == 1:
            a, z = self.args
            # 返回关于第一个参数的偏导数
            return uppergamma(a, z)*log(z) + meijerg([], [1, 1], [0, 0, a], [], z)
        else:
            raise ArgumentIndexError(self, argindex)

    # 定义对函数进行数值估计的方法
    def _eval_evalf(self, prec):
        if all(x.is_number for x in self.args):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                # 使用mpmath库计算上不完全伽玛函数的值
                res = mp.gammainc(a, z, mp.inf)
            return Expr._from_mpmath(res, prec)
        return self

    @classmethod
    # 定义一个类方法 eval，用于计算上不完全伽玛函数
    def eval(cls, a, z):
        # 从 sympy 库中导入指数积分函数 expint
        from sympy.functions.special.error_functions import expint
        
        # 如果 z 是一个数值
        if z.is_Number:
            # 如果 z 是 NaN，则返回 NaN
            if z is S.NaN:
                return S.NaN
            # 如果 z 是正无穷，则返回零
            elif z is oo:
                return S.Zero
            # 如果 z 是零
            elif z.is_zero:
                # 如果 a 的实部是正的，则返回伽玛函数 gamma(a)
                if re(a).is_positive:
                    return gamma(a)

        # 在这里提取分支信息，用于 lowergamma 函数的相关内容
        nx, n = z.extract_branch_factor()
        
        # 如果 a 是整数且为正数
        if a.is_integer and a.is_positive:
            # 对 z 进行去极化操作
            nx = unpolarify(z)
            # 如果 z 和 nx 不相等，则返回 uppergamma(a, nx)
            if z != nx:
                return uppergamma(a, nx)
        
        # 如果 a 是整数且为非正数
        elif a.is_integer and a.is_nonpositive:
            # 如果 n 不等于 0
            if n != 0:
                return -2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + uppergamma(a, nx)
        
        # 如果 n 不等于 0
        elif n != 0:
            return gamma(a)*(1 - exp(2*pi*I*n*a)) + exp(2*pi*I*n*a)*uppergamma(a, nx)

        # 特殊值处理
        if a.is_Number:
            # 如果 a 是零且 z 是正数，则返回 -Ei(-z)
            if a is S.Zero and z.is_positive:
                return -Ei(-z)
            # 如果 a 是 1，则返回 exp(-z)
            elif a is S.One:
                return exp(-z)
            # 如果 a 是 1/2，则返回 sqrt(pi)*erfc(sqrt(z))
            elif a is S.Half:
                return sqrt(pi)*erfc(sqrt(z))
            # 如果 a 是整数或者 2*a 是整数
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                # 如果 b 是正数
                if b.is_positive:
                    # 如果 a 是整数
                    if a.is_integer:
                        return exp(-z) * factorial(b) * Add(*[z**k / factorial(k)
                                                              for k in range(a)])
                    # 否则返回复杂的表达式
                    else:
                        return (gamma(a) * erfc(sqrt(z)) +
                                S.NegativeOne**(a - S(3)/2) * exp(-z) * sqrt(z)
                                * Add(*[gamma(-S.Half - k) * (-z)**k / gamma(1-a)
                                        for k in range(a - S.Half)]))
                # 如果 b 是整数
                elif b.is_Integer:
                    return expint(-b, z)*unpolarify(z)**(b + 1)

                # 如果 a 不是整数
                if not a.is_Integer:
                    return (S.NegativeOne**(S.Half - a) * pi*erfc(sqrt(z))/gamma(1-a)
                            - z**a * exp(-z) * Add(*[z**k * gamma(a) / gamma(a+k+1)
                                                     for k in range(S.Half - a)]))

        # 如果 a 是零且 z 是正数，则返回 -Ei(-z)
        if a.is_zero and z.is_positive:
            return -Ei(-z)

        # 如果 z 是零且 a 的实部是正的，则返回 gamma(a)
        if z.is_zero and re(a).is_positive:
            return gamma(a)

    # 定义一个私有方法 _eval_conjugate，用于计算函数的共轭
    def _eval_conjugate(self):
        # 获取函数的第二个参数 z
        z = self.args[1]
        # 如果 z 不是零或负无穷，则返回函数在参数共轭后的结果
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), z.conjugate())

    # 定义一个方法 _eval_is_meromorphic，用于判断函数是否是亚纯函数
    def _eval_is_meromorphic(self, x, a):
        # 调用 lowergamma 的 _eval_is_meromorphic 方法
        return lowergamma._eval_is_meromorphic(self, x, a)

    # 定义一个方法 _eval_rewrite_as_lowergamma，用于将函数重写为 lowergamma 函数的表达式
    def _eval_rewrite_as_lowergamma(self, s, x, **kwargs):
        return gamma(s) - lowergamma(s, x)

    # 定义一个方法 _eval_rewrite_as_tractable，用于将函数重写为 tractable 函数的表达式
    def _eval_rewrite_as_tractable(self, s, x, **kwargs):
        return exp(loggamma(s)) - lowergamma(s, x)

    # 定义一个方法 _eval_rewrite_as_expint，用于将函数重写为 expint 函数的表达式
    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        # 从 sympy.functions.special.error_functions 中导入 expint 函数
        from sympy.functions.special.error_functions import expint
        return expint(1 - s, x)*x**s
# 定义一个名为 polygamma 的类，继承自 Function 类
class polygamma(Function):
    r"""
    The function ``polygamma(n, z)`` returns ``log(gamma(z)).diff(n + 1)``.
    
    Explanation
    ===========
    
    It is a meromorphic function on $\mathbb{C}$ and defined as the $(n+1)$-th
    derivative of the logarithm of the gamma function:
    
    .. math::
        \psi^{(n)} (z) := \frac{\mathrm{d}^{n+1}}{\mathrm{d} z^{n+1}} \log\Gamma(z).
    
    For `n` not a nonnegative integer the generalization by Espinosa and Moll [5]_
    is used:
    
    .. math:: \psi(s,z) = \frac{\zeta'(s+1, z) + (\gamma + \psi(-s)) \zeta(s+1, z)}
        {\Gamma(-s)}
    
    Examples
    ========
    
    Several special values are known:
    
    >>> from sympy import S, polygamma
    >>> polygamma(0, 1)
    -EulerGamma
    >>> polygamma(0, 1/S(2))
    -2*log(2) - EulerGamma
    >>> polygamma(0, 1/S(3))
    -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))
    >>> polygamma(0, 1/S(4))
    -pi/2 - log(4) - log(2) - EulerGamma
    >>> polygamma(0, 2)
    1 - EulerGamma
    >>> polygamma(0, 23)
    19093197/5173168 - EulerGamma
    
    >>> from sympy import oo, I
    >>> polygamma(0, oo)
    oo
    >>> polygamma(0, -oo)
    oo
    >>> polygamma(0, I*oo)
    oo
    >>> polygamma(0, -I*oo)
    oo
    
    Differentiation with respect to $x$ is supported:
    
    >>> from sympy import Symbol, diff
    >>> x = Symbol("x")
    >>> diff(polygamma(0, x), x)
    polygamma(1, x)
    >>> diff(polygamma(0, x), x, 2)
    polygamma(2, x)
    >>> diff(polygamma(0, x), x, 3)
    polygamma(3, x)
    >>> diff(polygamma(1, x), x)
    polygamma(2, x)
    >>> diff(polygamma(1, x), x, 2)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x, 2)
    polygamma(4, x)
    
    >>> n = Symbol("n")
    >>> diff(polygamma(n, x), x)
    polygamma(n + 1, x)
    >>> diff(polygamma(n, x), x, 2)
    polygamma(n + 2, x)
    
    We can rewrite ``polygamma`` functions in terms of harmonic numbers:
    
    >>> from sympy import harmonic
    >>> polygamma(0, x).rewrite(harmonic)
    harmonic(x - 1) - EulerGamma
    >>> polygamma(2, x).rewrite(harmonic)
    2*harmonic(x - 1, 3) - 2*zeta(3)
    >>> ni = Symbol("n", integer=True)
    >>> polygamma(ni, x).rewrite(harmonic)
    (-1)**(n + 1)*(-harmonic(x - 1, n + 1) + zeta(n + 1))*factorial(n)
    
    See Also
    ========
    
    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/Polygamma_function
    .. [2] https://mathworld.wolfram.com/PolygammaFunction.html
    
    """
    """
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma/
    .. [4] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/
    .. [5] O. Espinosa and V. Moll, "A generalized polygamma function",
           *Integral Transforms and Special Functions* (2004), 101-115.

    """

    @classmethod
    # 类方法，用于计算广义多级gamma函数的值
    def eval(cls, n, z):
        # 若 n 或 z 是 NaN，则返回 NaN
        if n is S.NaN or z is S.NaN:
            return S.NaN
        # 若 z 为无穷大，则根据 n 的值返回无穷大或零
        elif z is oo:
            return oo if n.is_zero else S.Zero
        # 若 z 是负整数，则返回复无穷大
        elif z.is_Integer and z.is_nonpositive:
            return S.ComplexInfinity
        # 若 n 是 -1，则返回 loggamma(z) - log(2*pi) / 2
        elif n is S.NegativeOne:
            return loggamma(z) - log(2*pi) / 2
        # 若 n 是零
        elif n.is_zero:
            # 若 z 是负无穷或虚部为无穷的复数，则返回无穷大
            if z is -oo or z.extract_multiplicatively(I) in (oo, -oo):
                return oo
            # 若 z 是整数，则返回调和级数的值减去欧拉常数
            elif z.is_Integer:
                return harmonic(z-1) - S.EulerGamma
            # 若 z 是有理数
            elif z.is_Rational:
                # TODO: 当 n == 1 时，也可以处理一些有理数 z
                p, q = z.as_numer_denom()
                # 只在分母较小的情况下展开，以避免生成过长的表达式
                if q <= 6:
                    return expand_func(polygamma(S.Zero, z, evaluate=False))
        # 若 n 是整数且非负
        elif n.is_integer and n.is_nonnegative:
            # 去极化 z
            nz = unpolarify(z)
            # 若 z 不等于 nz，则返回 polygamma(n, nz) 的值
            if z != nz:
                return polygamma(n, nz)
            # 若 z 是整数
            if z.is_Integer:
                return S.NegativeOne**(n+1) * factorial(n) * zeta(n+1, z)
            # 若 z 是半整数
            elif z is S.Half:
                return S.NegativeOne**(n+1) * factorial(n) * (2**(n+1)-1) * zeta(n+1)

    # 判断函数是否返回实数
    def _eval_is_real(self):
        if self.args[0].is_positive and self.args[1].is_positive:
            return True

    # 判断函数是否返回复数
    def _eval_is_complex(self):
        z = self.args[1]
        # 判断 z 是否为负整数
        is_negative_integer = fuzzy_and([z.is_negative, z.is_integer])
        return fuzzy_and([z.is_complex, fuzzy_not(is_negative_integer)])

    # 判断函数是否返回正数
    def _eval_is_positive(self):
        n, z = self.args
        if n.is_positive:
            if n.is_odd and z.is_real:
                return True
            if n.is_even and z.is_positive:
                return False

    # 判断函数是否返回负数
    def _eval_is_negative(self):
        n, z = self.args
        if n.is_positive:
            if n.is_even and z.is_positive:
                return True
            if n.is_odd and z.is_real:
                return False
    # 评估和扩展函数，接受关键字参数
    def _eval_expand_func(self, **hints):
        # 提取参数列表中的 n 和 z
        n, z = self.args

        # 检查 n 是否为非负整数
        if n.is_Integer and n.is_nonnegative:
            # 如果 z 是加法表达式
            if z.is_Add:
                # 提取第一个参数作为系数
                coeff = z.args[0]
                # 如果系数是整数
                if coeff.is_Integer:
                    # 计算指数 e
                    e = -(n + 1)
                    # 如果系数大于0
                    if coeff > 0:
                        # 构建尾部项
                        tail = Add(*[Pow(z - i, e) for i in range(1, int(coeff) + 1)])
                    else:
                        # 系数小于等于0时的尾部项
                        tail = -Add(*[Pow(z + i, e) for i in range(int(-coeff))])
                    # 返回结果：polygamma 函数和尾部项的加和
                    return polygamma(n, z - coeff) + S.NegativeOne**n * factorial(n) * tail

            # 如果 z 是乘法表达式
            elif z.is_Mul:
                # 将 z 分解为系数和剩余部分
                coeff, z = z.as_two_terms()
                # 如果系数是正整数
                if coeff.is_Integer and coeff.is_positive:
                    # 构建尾部项列表
                    tail = [polygamma(n, z + Rational(i, coeff)) for i in range(int(coeff))]
                    # 如果 n 等于0
                    if n == 0:
                        # 返回结果：尾部项加和除以系数，再加上系数的对数
                        return Add(*tail) / coeff + log(coeff)
                    else:
                        # 返回结果：尾部项加和除以系数的 n+1 次幂
                        return Add(*tail) / coeff**(n + 1)
                # 将 z 乘以系数
                z *= coeff

        # 如果 n 等于0并且 z 是有理数
        if n == 0 and z.is_Rational:
            p, q = z.as_numer_denom()

            # 参考文献：针对有理数参数的 polygamma 函数值，J. Choi, 2007
            # 计算第一部分
            part_1 = -S.EulerGamma - pi * cot(p * pi / q) / 2 - log(q) + Add(
                *[cos(2 * k * pi * p / q) * log(2 * sin(k * pi / q)) for k in range(1, q)])

            # 如果 z 大于0
            if z > 0:
                # 计算整数部分和小数部分
                n = floor(z)
                z0 = z - n
                # 返回结果：第一部分加上小数部分的倒数加和
                return part_1 + Add(*[1 / (z0 + k) for k in range(n)])
            # 如果 z 小于0
            elif z < 0:
                # 计算整数部分和小数部分
                n = floor(1 - z)
                z0 = z + n
                # 返回结果：第一部分减去小数部分的倒数加和
                return part_1 - Add(*[1 / (z0 - 1 - k) for k in range(n)])

        # 如果 n 等于 -1
        if n == -1:
            # 返回结果：对数 gamma 函数减去对数 2*pi 的一半
            return loggamma(z) - log(2*pi) / 2

        # 如果 n 不是整数或者不是非负数
        if n.is_integer is False or n.is_nonnegative is False:
            # 定义虚拟变量 s
            s = Dummy("s")
            # 计算 zeta 函数关于 s 的导数，并在 s=n+1 处求值
            dzt = zeta(s, z).diff(s).subs(s, n+1)
            # 返回结果：导数值加上欧拉常数和 digamma 函数的乘积，再除以 gamma 函数的绝对值
            return (dzt + (S.EulerGamma + digamma(-n)) * zeta(n+1, z)) / gamma(-n)

        # 默认情况下返回 polygamma 函数的值
        return polygamma(n, z)

    # 将函数重写为 zeta 函数的形式
    def _eval_rewrite_as_zeta(self, n, z, **kwargs):
        # 如果 n 是整数且为正数
        if n.is_integer and n.is_positive:
            # 返回结果：(-1)^(n+1) * n! * zeta(n+1, z)
            return S.NegativeOne**(n + 1) * factorial(n) * zeta(n + 1, z)

    # 将函数重写为 harmonic 函数的形式
    def _eval_rewrite_as_harmonic(self, n, z, **kwargs):
        # 如果 n 是整数
        if n.is_integer:
            # 如果 n 等于 0
            if n.is_zero:
                # 返回结果：harmonic(z - 1) - EulerGamma
                return harmonic(z - 1) - S.EulerGamma
            else:
                # 返回结果：(-1)^(n+1) * n! * (zeta(n+1) - harmonic(z-1, n+1))
                return S.NegativeOne**(n + 1) * factorial(n) * (zeta(n + 1) - harmonic(z - 1, n + 1))

    # 将函数重写为主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入必要的模块和类
        from sympy.series.order import Order
        # 提取参数并计算其在 x 上的主导项
        n, z = [a.as_leading_term(x) for a in self.args]
        o = Order(z, x)
        # 如果 n 等于0并且 o 包含 1/x
        if n == 0 and o.contains(1/x):
            # 如果 logx 为空，设置为 log(x)
            logx = log(x) if logx is None else logx
            # 返回结果：o 的次数乘以 logx
            return o.getn() * logx
        else:
            # 返回结果：调用函数自身并传入参数
            return self.func(n, z)
    # 定义一个方法 fdiff，计算多项式伽玛函数的导数
    def fdiff(self, argindex=2):
        # 如果参数索引为 2
        if argindex == 2:
            # 解构参数 args 的前两个元素为 n 和 z
            n, z = self.args[:2]
            # 返回 polygamma(n + 1, z) 的计算结果
            return polygamma(n + 1, z)
        else:
            # 抛出参数索引错误异常，传递 self 和 argindex
            raise ArgumentIndexError(self, argindex)

    # 定义一个方法 _eval_aseries，用于计算级数展开
    def _eval_aseries(self, n, args0, x, logx):
        # 导入必要的 Order 类
        from sympy.series.order import Order
        # 如果 args0 的第二个元素不是无穷大或者 self 的第一个参数不是整数且不是非负数
        if args0[1] != oo or not (self.args[0].is_Integer and self.args[0].is_nonnegative):
            # 调用父类的 _eval_aseries 方法处理
            return super()._eval_aseries(n, args0, x, logx)
        # 获取 self 的第二个参数作为 z
        z = self.args[1]
        # 获取 self 的第一个参数作为 N
        N = self.args[0]

        # 如果 N 等于 0
        if N == 0:
            # digamma 函数的级数展开
            # 参考 Abramowitz & Stegun, p. 259, 6.3.18
            r = log(z) - 1/(2*z)
            o = None
            # 如果 n 小于 2，设置 o 为 Order(1/z, x)
            if n < 2:
                o = Order(1/z, x)
            else:
                # 否则计算更高阶的展开
                m = ceiling((n + 1)//2)
                l = [bernoulli(2*k) / (2*k*z**(2*k)) for k in range(1, m)]
                r -= Add(*l)
                o = Order(1/z**n, x)
            # 返回级数展开的结果 r 和阶数 o 的和
            return r._eval_nseries(x, n, logx) + o
        else:
            # 多项式伽玛函数的级数展开
            # 参考 Abramowitz & Stegun, p. 260, 6.4.10
            # 故意返回比 O(x**n) 高阶的项
            fac = gamma(N)
            e0 = fac + N*fac/(2*z)
            m = ceiling((n + 1)//2)
            for k in range(1, m):
                fac = fac*(2*k + N - 1)*(2*k + N - 2) / ((2*k)*(2*k - 1))
                e0 += bernoulli(2*k)*fac/z**(2*k)
            o = Order(1/z**(2*m), x)
            if n == 0:
                o = Order(1/z, x)
            elif n == 1:
                o = Order(1/z**2, x)
            # 计算级数展开的结果 r，并返回
            r = e0._eval_nseries(z, n, logx) + o
            return (-1 * (-1/z)**N * r)._eval_nseries(x, n, logx)

    # 定义一个方法 _eval_evalf，用于数值求解
    def _eval_evalf(self, prec):
        # 如果 self 的所有参数都是数值类型
        if not all(i.is_number for i in self.args):
            return
        # 将 self 的第一个参数转换为高精度数值 s
        s = self.args[0]._to_mpmath(prec+12)
        # 将 self 的第二个参数转换为高精度数值 z
        z = self.args[1]._to_mpmath(prec+12)
        # 如果 z 是整数且小于等于 0，则返回无穷大
        if mp.isint(z) and z <= 0:
            return S.ComplexInfinity
        # 设置工作精度为 prec+12
        with workprec(prec+12):
            # 如果 s 是整数且大于等于 0
            if mp.isint(s) and s >= 0:
                # 计算 polygamma(s, z) 的数值结果
                res = mp.polygamma(s, z)
            else:
                # 否则使用 zeta 函数进行计算
                zt = mp.zeta(s+1, z)
                dzt = mp.zeta(s+1, z, 1)
                res = (dzt + (mp.euler + mp.digamma(-s)) * zt) * mp.rgamma(-s)
        # 从 mpmath 结果中构建并返回 SymPy 表达式
        return Expr._from_mpmath(res, prec)
class loggamma(Function):
    r"""
    The ``loggamma`` function implements the logarithm of the
    gamma function (i.e., $\log\Gamma(x)$).

    Examples
    ========

    Several special values are known. For numerical integral
    arguments we have:

    >>> from sympy import loggamma
    >>> loggamma(-2)
    oo
    >>> loggamma(0)
    oo
    >>> loggamma(1)
    0
    >>> loggamma(2)
    0
    >>> loggamma(3)
    log(2)

    And for symbolic values:

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> loggamma(n)
    log(gamma(n))
    >>> loggamma(-n)
    oo

    For half-integral values:

    >>> from sympy import S
    >>> loggamma(S(5)/2)
    log(3*sqrt(pi)/4)
    >>> loggamma(n/2)
    log(2**(1 - n)*sqrt(pi)*gamma(n)/gamma(n/2 + 1/2))

    And general rational arguments:

    >>> from sympy import expand_func
    >>> L = loggamma(S(16)/3)
    >>> expand_func(L).doit()
    -5*log(3) + loggamma(1/3) + log(4) + log(7) + log(10) + log(13)
    >>> L = loggamma(S(19)/4)
    >>> expand_func(L).doit()
    -4*log(4) + loggamma(3/4) + log(3) + log(7) + log(11) + log(15)
    >>> L = loggamma(S(23)/7)
    >>> expand_func(L).doit()
    -3*log(7) + log(2) + loggamma(2/7) + log(9) + log(16)

    The ``loggamma`` function has the following limits towards infinity:

    >>> from sympy import oo
    >>> loggamma(oo)
    oo
    >>> loggamma(-oo)
    zoo

    The ``loggamma`` function obeys the mirror symmetry
    if $x \in \mathbb{C} \setminus \{-\infty, 0\}$:

    >>> from sympy.abc import x
    >>> from sympy import conjugate
    >>> conjugate(loggamma(x))
    loggamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(loggamma(x), x)
    polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(loggamma(x), x, 0, 4).cancel()
    -log(x) - EulerGamma*x + pi**2*x**2/12 - x**3*zeta(3)/3 + O(x**4)

    We can numerically evaluate the ``loggamma`` function
    to arbitrary precision on the whole complex plane:

    >>> from sympy import I
    >>> loggamma(5).evalf(30)
    3.17805383034794561964694160130
    >>> loggamma(I).evalf(20)
    -0.65092319930185633889 - 1.8724366472624298171*I

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/LogGammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/LogGamma/

    """

    @classmethod


注释：

# 定义一个名为 loggamma 的类，表示对数 Gamma 函数，即 $\log\Gamma(x)$
class loggamma(Function):
    r"""
    The ``loggamma`` function implements the logarithm of the
    gamma function (i.e., $\log\Gamma(x)$).
    实现对数 Gamma 函数的计算，即 $\log\Gamma(x)$。

    Examples
    ========

    Several special values are known. For numerical integral
    arguments we have:
    有多个已知的特殊值。对于数值积分参数，我们有：

    >>> from sympy import loggamma
    >>> loggamma(-2)
    oo
    >>> loggamma(0)
    oo
    >>> loggamma(1)
    0
    >>> loggamma(2)
    0
    >>> loggamma(3)
    log(2)
    几个特殊值的例子。对于数值积分参数，我们有：

    And for symbolic values:
    对于符号值：

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> loggamma(n)
    log(gamma(n))
    >>> loggamma(-n)
    oo
    对于符号值：

    For half-integral values:
    对于半整数值：

    >>> from sympy import S
    >>> loggamma(S(5)/2)
    log(3*sqrt(pi)/4)
    >>> loggamma(n/2)
    log(2**(1 - n)*sqrt(pi)*gamma(n)/gamma(n/2 + 1/2))
    对于半整数值：

    And general rational arguments:
    对于一般的有理数参数：

    >>> from sympy import expand_func
    >>> L = loggamma(S(16)/3)
    >>> expand_func(L).doit()
    -5*log(3) + loggamma(1/3) + log(4) + log(7) + log(10) + log(13)
    >>> L = loggamma(S(19)/4)
    >>> expand_func(L).doit()
    -4*log(4) + loggamma(3/4) + log(3) + log(7) + log(11) + log(15)
    >>> L = loggamma(S(23)/7)
    >>> expand_func(L).doit()
    -3*log(7) + log(2) + loggamma(2/7) + log(9) + log(16)
    对于一般的有理数参数：

    The ``loggamma`` function has the following limits towards infinity:
    ``loggamma`` 函数在无穷远处的极限如下：

    >>> from sympy import oo
    >>> loggamma(oo)
    oo
    >>> loggamma(-oo)
    zoo
    函数在无穷远处的极限如下：

    The ``loggamma`` function obeys the mirror symmetry
    ``loggamma`` 函数遵循镜像对称性
    if $x \in \mathbb{C} \setminus \{-\infty, 0\}$:
    如果 $x \in \mathbb{C} \setminus \{-\infty, 0\}$：

    >>> from sympy.abc import x
    >>> from sympy import conjugate
    >>> conjugate(loggamma(x))
    loggamma(conjugate(x))
    对于 $x \in \mathbb{C} \setminus \{-\infty, 0\}$，函数满足镜像对称性：

    Differentiation with respect to $x$ is supported:
    支持对 $x$ 的微分：

    >>> from sympy import diff
    >>> diff(loggamma(x), x)
    polygamma(0, x)
    对 $x$ 的微分是支持的：

    Series expansion is also supported:
    同样支持级数展开：

    >>> from sympy import series
    >>> series(loggamma(x), x, 0, 4).cancel()
    -log(x) - EulerGamma*x + pi**2*x**2/12 - x**3*zeta(3)/3 + O(x**4)
    同样支持级数展开：

    We can numerically evaluate the ``loggamma`` function
    to arbitrary precision on the whole complex plane:
    我们可以在整个复平面上对 ``loggamma`` 函数进行任意精度的数值计算：

    >>> from sympy import I
    >>> loggamma(5).evalf(30)
    3.17805383034794561964694160130
    >>> loggamma(I).evalf(20)
    -0.65092319930185633889 - 1.8724366472624298171*I
    可以对整个复平面上的 ``loggamma`` 函数进行任意精度的数值计算：

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    digamma:
    def eval(cls, z):
        # 如果 z 是整数
        if z.is_integer:
            # 如果 z 是非正数
            if z.is_nonpositive:
                return oo  # 返回正无穷大
            elif z.is_positive:
                return log(gamma(z))  # 返回 z 的阶乘的自然对数
        # 如果 z 是有理数
        elif z.is_rational:
            p, q = z.as_numer_denom()
            # 半整数值：
            if p.is_positive and q == 2:
                # 返回半整数值 p/2 的对数表达式
                return log(sqrt(pi) * 2**(1 - p) * gamma(p) / gamma((p + 1)*S.Half))

        # 如果 z 是正无穷大
        if z is oo:
            return oo  # 返回正无穷大
        # 如果 z 的绝对值是正无穷大
        elif abs(z) is oo:
            return S.ComplexInfinity  # 返回复数无穷大
        # 如果 z 是 NaN
        if z is S.NaN:
            return S.NaN  # 返回 NaN

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        z = self.args[0]

        # 如果 z 是有理数
        if z.is_Rational:
            p, q = z.as_numer_denom()
            # 一般的有理数参数 (u + p/q)
            # 将 z 拆分为 n + p/q，其中 p < q
            n = p // q
            p = p - n*q
            if p.is_positive and q.is_positive and p < q:
                k = Dummy("k")
                # 如果 n 是正数
                if n.is_positive:
                    # 返回对数伽玛函数的展开表达式
                    return loggamma(p / q) - n*log(q) + Sum(log((k - 1)*q + p), (k, 1, n))
                # 如果 n 是负数
                elif n.is_negative:
                    # 返回对数伽玛函数的展开表达式
                    return loggamma(p / q) - n*log(q) + pi*I*n - Sum(log(k*q - p), (k, 1, -n))
                # 如果 n 是零
                elif n.is_zero:
                    # 返回对数伽玛函数的展开表达式
                    return loggamma(p / q)

        return self

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        x0 = self.args[0].limit(x, 0)
        # 如果 x0 是零
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            # 返回转化为不可解形式后的 nseries 展开
            return f._eval_nseries(x, n, logx)
        # 否则调用父类的 _eval_nseries 方法
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        # 如果 args0[0] 不是正无穷大
        if args0[0] != oo:
            # 调用父类的 _eval_aseries 方法
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[0]
        r = log(z)*(z - S.Half) - z + log(2*pi)/2
        l = [bernoulli(2*k) / (2*k*(2*k - 1)*z**(2*k - 1)) for k in range(1, n)]
        o = None
        if n == 0:
            o = Order(1, x)
        else:
            o = Order(1/z**n, x)
        # 非常低效地首先添加顺序，然后进行 nseries
        return (r + Add(*l))._eval_nseries(x, n, logx) + o

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        # 返回对数伽玛函数的重写为不可解形式的结果
        return log(gamma(z))

    def _eval_is_real(self):
        z = self.args[0]
        # 如果 z 是正数
        if z.is_positive:
            return True  # 返回 True
        # 如果 z 是非正数
        elif z.is_nonpositive:
            return False  # 返回 False

    def _eval_conjugate(self):
        z = self.args[0]
        # 如果 z 不是零和负无穷大
        if z not in (S.Zero, S.NegativeInfinity):
            # 返回 z 的共轭
            return self.func(z.conjugate())

    def fdiff(self, argindex=1):
        # 如果 argindex 等于 1
        if argindex == 1:
            # 返回 z 的多项式 gamma 函数的一阶导数
            return polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)
class digamma(Function):
    r"""
    The ``digamma`` function is the first derivative of the ``loggamma``
    function

    .. math::
        \psi(x) := \frac{\mathrm{d}}{\mathrm{d} z} \log\Gamma(z)
                = \frac{\Gamma'(z)}{\Gamma(z) }.

    In this case, ``digamma(z) = polygamma(0, z)``.

    Examples
    ========

    >>> from sympy import digamma
    >>> digamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> digamma(z)
    polygamma(0, z)

    To retain ``digamma`` as it is:

    >>> digamma(0, evaluate=False)
    digamma(0)
    >>> digamma(z, evaluate=False)
    digamma(z)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Digamma_function
    .. [2] https://mathworld.wolfram.com/DigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    
    # Override _eval_evalf method to evaluate the digamma function numerically
    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(0, z).evalf(n=nprec)

    # Define the first derivative of the digamma function
    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(0, z).fdiff()

    # Check if the digamma function evaluated at z is real
    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(0, z).is_real

    # Check if the digamma function evaluated at z is positive
    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(0, z).is_positive

    # Check if the digamma function evaluated at z is negative
    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(0, z).is_negative

    # Evaluate the asymptotic expansion of the digamma function
    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.Zero,] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    # Class method to evaluate the digamma function
    @classmethod
    def eval(cls, z):
        return polygamma(0, z)

    # Expand the digamma function using its definition
    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(0, z).expand(func=True)

    # Rewrite the digamma function as a harmonic function
    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return harmonic(z - 1) - S.EulerGamma

    # Rewrite the digamma function using the polygamma function
    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(0, z)

    # Compute the leading term of the digamma function in its asymptotic expansion
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        z = self.args[0]
        return polygamma(0, z).as_leading_term(x)



class trigamma(Function):
    r"""
    The ``trigamma`` function is the second derivative of the ``loggamma``
    function

    .. math::
        \psi^{(1)}(z) := \frac{\mathrm{d}^{2}}{\mathrm{d} z^{2}} \log\Gamma(z).

    In this case, ``trigamma(z) = polygamma(1, z)``.

    Examples
    ========

    >>> from sympy import trigamma
    >>> trigamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> trigamma(z)
    polygamma(1, z)

    To retain ``trigamma`` as it is:

    >>> trigamma(0, evaluate=False)
    trigamma(0)

    """
    # 调用 trigamma 函数计算 trigamma(z)，但不对结果进行数值评估
    trigamma(z, evaluate=False)

    # 返回 trigamma(z) 的表达式
    trigamma(z)


    # 参见
    # ========

    # gamma: Gamma 函数。
    # lowergamma: 下不完全 Gamma 函数。
    # uppergamma: 上不完全 Gamma 函数。
    # polygamma: Polygamma 函数。
    # loggamma: 对数 Gamma 函数。
    # digamma: Digamma 函数。
    # sympy.functions.special.beta_functions.beta: Euler Beta 函数。

    # 引用
    # ==========

    # .. [1] https://en.wikipedia.org/wiki/Trigamma_function
    # .. [2] https://mathworld.wolfram.com/TrigammaFunction.html
    # .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    定义 Trigamma 函数的类，继承于 Function 类。

    """

    # 计算 Trigamma 函数在给定精度下的数值估计
    def _eval_evalf(self, prec):
        # 提取参数 z
        z = self.args[0]
        # 将精度转换为小数位数
        nprec = prec_to_dps(prec)
        # 计算 z 处的一阶多对数函数值并以指定精度评估
        return polygamma(1, z).evalf(n=nprec)

    # 返回 Trigamma 函数关于其参数的一阶导数
    def fdiff(self, argindex=1):
        # 提取参数 z
        z = self.args[0]
        # 计算 z 处的一阶多对数函数的一阶导数
        return polygamma(1, z).fdiff()

    # 检查 Trigamma 函数是否是实数
    def _eval_is_real(self):
        # 提取参数 z
        z = self.args[0]
        # 检查 z 处的一阶多对数函数是否是实数
        return polygamma(1, z).is_real

    # 检查 Trigamma 函数是否是正数
    def _eval_is_positive(self):
        # 提取参数 z
        z = self.args[0]
        # 检查 z 处的一阶多对数函数是否是正数
        return polygamma(1, z).is_positive

    # 检查 Trigamma 函数是否是负数
    def _eval_is_negative(self):
        # 提取参数 z
        z = self.args[0]
        # 检查 z 处的一阶多对数函数是否是负数
        return polygamma(1, z).is_negative

    # 将 Trigamma 函数展开为其一阶多对数函数的级数表示
    def _eval_aseries(self, n, args0, x, logx):
        # 重写为 polygamma 函数的级数表示
        as_polygamma = self.rewrite(polygamma)
        # 将参数列表 args0 前置添加单位元素
        args0 = [S.One,] + args0
        # 返回 polygamma 函数的级数表示
        return as_polygamma._eval_aseries(n, args0, x, logx)

    # 计算 Trigamma 函数的标准求值
    @classmethod
    def eval(cls, z):
        # 返回 z 处的一阶多对数函数
        return polygamma(1, z)

    # 将 Trigamma 函数展开为其一阶多对数函数的扩展形式
    def _eval_expand_func(self, **hints):
        # 提取参数 z
        z = self.args[0]
        # 返回 z 处的一阶多对数函数的扩展形式
        return polygamma(1, z).expand(func=True)

    # 将 Trigamma 函数重写为 Zeta 函数的形式
    def _eval_rewrite_as_zeta(self, z, **kwargs):
        # 返回 z 处的 Zeta 函数的二阶形式
        return zeta(2, z)

    # 将 Trigamma 函数重写为 polygamma 函数的形式
    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        # 返回 z 处的一阶多对数函数
        return polygamma(1, z)

    # 将 Trigamma 函数重写为 Harmonic 函数的形式
    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        # 返回 z 减一处的二阶 Harmonic 函数减去 pi^2 / 6 的结果
        return -harmonic(z - 1, 2) + pi**2 / 6

    # 返回 Trigamma 函数的主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 提取参数 z
        z = self.args[0]
        # 返回 z 处的一阶多对数函数的主导项
        return polygamma(1, z).as_leading_term(x)
# 多变量伽马函数的实现，继承自 sympy 的 Function 类
class multigamma(Function):
    """
    多变量伽马函数是伽马函数的一般化形式

    .. math::
        \Gamma_p(z) = \pi^{p(p-1)/4}\prod_{k=1}^p \Gamma[z + (1 - k)/2].

    在特殊情况下，``multigamma(x, 1) = gamma(x)``.

    Examples
    ========

    >>> from sympy import S, multigamma
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> p = Symbol('p', positive=True, integer=True)

    >>> multigamma(x, p)
    pi**(p*(p - 1)/4)*Product(gamma(-_k/2 + x + 1/2), (_k, 1, p))

    几个已知的特殊值：

    >>> multigamma(1, 1)
    1
    >>> multigamma(4, 1)
    6
    >>> multigamma(S(3)/2, 1)
    sqrt(pi)/2

    将 ``multigamma`` 表达为 ``gamma`` 函数的形式：

    >>> multigamma(x, 1)
    gamma(x)

    >>> multigamma(x, 2)
    sqrt(pi)*gamma(x)*gamma(x - 1/2)

    >>> multigamma(x, 3)
    pi**(3/2)*gamma(x)*gamma(x - 1)*gamma(x - 1/2)

    Parameters
    ==========

    p : 多变量伽马函数的阶数或维数

    See Also
    ========

    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
    sympy.functions.special.beta_functions.beta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_gamma_function

    """
    
    unbranched = True  # 标识符，表示此函数没有分支

    def fdiff(self, argindex=2):
        # 计算偏导数，使用 SymPy 的 Sum 类进行求和
        from sympy.concrete.summations import Sum
        if argindex == 2:
            x, p = self.args
            k = Dummy("k")
            # 返回偏导数结果
            return self.func(x, p)*Sum(polygamma(0, x + (1 - k)/2), (k, 1, p))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, p):
        # 计算函数的值，使用 SymPy 的 Product 类进行乘积计算
        from sympy.concrete.products import Product
        if p.is_positive is False or p.is_integer is False:
            raise ValueError('Order parameter p must be positive integer.')
        k = Dummy("k")
        # 返回计算结果
        return (pi**(p*(p - 1)/4)*Product(gamma(x + (1 - k)/2),
                                          (k, 1, p))).doit()

    def _eval_conjugate(self):
        # 返回函数的共轭，用于复变换
        x, p = self.args
        return self.func(x.conjugate(), p)

    def _eval_is_real(self):
        # 判断函数是否为实数函数
        x, p = self.args
        y = 2*x
        if y.is_integer and (y <= (p - 1)) is True:
            return False
        if intlike(y) and (y <= (p - 1)):
            return False
        if y > (p - 1) or y.is_noninteger:
            return True
```