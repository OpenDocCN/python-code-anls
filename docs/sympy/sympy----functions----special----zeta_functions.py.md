# `D:\src\scipysrc\sympy\sympy\functions\special\zeta_functions.py`

```
""" Riemann zeta and related function. """

# 导入需要的模块和类
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly

###############################################################################
###################### LERCH TRANSCENDENT #####################################
###############################################################################

# 定义 Lerch 超越函数的类
class lerchphi(Function):
    r"""
    Lerch transcendent (Lerch phi function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$, $|z| < 1$ and $s \in \mathbb{C}$, the
    Lerch transcendent is defined as

    .. math :: \Phi(z, s, a) = \sum_{n=0}^\infty \frac{z^n}{(n + a)^s},

    where the standard branch of the argument is used for $n + a$,
    and by analytic continuation for other values of the parameters.

    A commonly used related function is the Lerch zeta function, defined by

    .. math:: L(q, s, a) = \Phi(e^{2\pi i q}, s, a).

    **Analytic Continuation and Branching Behavior**

    It can be shown that

    .. math:: \Phi(z, s, a) = z\Phi(z, s, a+1) + a^{-s}.

    This provides the analytic continuation to $\operatorname{Re}(a) \le 0$.

    Assume now $\operatorname{Re}(a) > 0$. The integral representation

    .. math:: \Phi_0(z, s, a) = \int_0^\infty \frac{t^{s-1} e^{-at}}{1 - ze^{-t}}
                                \frac{\mathrm{d}t}{\Gamma(s)}

    provides an analytic continuation to $\mathbb{C} - [1, \infty)$.
    Finally, for $x \in (1, \infty)$ we find

    .. math:: \lim_{\epsilon \to 0^+} \Phi_0(x + i\epsilon, s, a)
             -\lim_{\epsilon \to 0^+} \Phi_0(x - i\epsilon, s, a)
             = \frac{2\pi i \log^{s-1}{x}}{x^a \Gamma(s)},

    using the standard branch for both $\log{x}$ and
    $\log{\log{x}}$ (a branch of $\log{\log{x}}$ is needed to
    evaluate $\log{x}^{s-1}$).
    This concludes the analytic continuation. The Lerch transcendent is thus
    branched at $z \in \{0, 1, \infty\}$ and
    $a \in \mathbb{Z}_{\le 0}$. For fixed $z, a$ outside these
    branch points, it is an entire function of $s$.

    Examples
    ========

    The Lerch transcendent is a fairly general function, for this reason it does
    not automatically evaluate to simpler functions. Use ``expand_func()`` to
    achieve this.
    If $z=1$, the Lerch transcendent reduces to the Hurwitz zeta function:

    >>> from sympy import lerchphi, expand_func
    >>> from sympy.abc import z, s, a
    >>> expand_func(lerchphi(1, s, a))
    zeta(s, a)

    More generally, if $z$ is a root of unity, the Lerch transcendent
    reduces to a sum of Hurwitz zeta functions:

    >>> expand_func(lerchphi(-1, s, a))
    zeta(s, a/2)/2**s - zeta(s, a/2 + 1/2)/2**s

    If $a=1$, the Lerch transcendent reduces to the polylogarithm:

    >>> expand_func(lerchphi(z, s, 1))
    polylog(s, z)/z

    More generally, if $a$ is rational, the Lerch transcendent reduces
    to a sum of polylogarithms:

    >>> from sympy import S
    >>> expand_func(lerchphi(z, s, S(1)/2))
    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))
    >>> expand_func(lerchphi(z, s, S(3)/2))
    -2**s/z + 2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                          polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))/z

    The derivatives with respect to $z$ and $a$ can be computed in
    closed form:

    >>> lerchphi(z, s, a).diff(z)
    (-a*lerchphi(z, s, a) + lerchphi(z, s - 1, a))/z
    >>> lerchphi(z, s, a).diff(a)
    -s*lerchphi(z, s + 1, a)

    See Also
    ========

    polylog, zeta

    References
    ==========

    .. [1] Bateman, H.; Erdelyi, A. (1953), Higher Transcendental Functions,
           Vol. I, New York: McGraw-Hill. Section 1.11.
    .. [2] https://dlmf.nist.gov/25.14
    .. [3] https://en.wikipedia.org/wiki/Lerch_transcendent
    # 对象方法，用于评估和扩展函数
    def _eval_expand_func(self, **hints):
        # 解构参数
        z, s, a = self.args
        # 如果 z 等于 1，则返回 zeta 函数的结果
        if z == 1:
            return zeta(s, a)
        # 如果 s 是整数且小于等于 0
        if s.is_Integer and s <= 0:
            # 创建虚拟变量 t
            t = Dummy('t')
            # 构建多项式 p
            p = Poly((t + a)**(-s), t)
            # 初始化起始值
            start = 1/(1 - t)
            res = S.Zero
            # 反向遍历多项式的所有系数
            for c in reversed(p.all_coeffs()):
                res += c*start
                start = t*start.diff(t)
            return res.subs(t, z)

        # 如果 a 是有理数
        if a.is_Rational:
            # 函数的注释信息参见论文中的第18节
            add = S.Zero
            mul = S.One
            # 将 a 缩减到区间 (0, 1]
            if a > 1:
                n = floor(a)
                if n == a:
                    n -= 1
                a -= n
                mul = z**(-n)
                add = Add(*[-z**(k - n)/(a + k)**s for k in range(n)])
            elif a <= 0:
                n = floor(-a) + 1
                a += n
                mul = z**n
                add = Add(*[z**(n - 1 - k)/(a - k - 1)**s for k in range(n)])

            m, n = S([a.p, a.q])
            # 计算指数函数的极坐标
            zet = exp_polar(2*pi*I/n)
            # 计算 z 的根
            root = z**(1/n)
            # 解极化 zet
            up_zet = unpolarify(zet)
            addargs = []
            # 遍历范围内的值
            for k in range(n):
                p = polylog(s, zet**k*root)
                if isinstance(p, polylog):
                    p = p._eval_expand_func(**hints)
                addargs.append(p/(up_zet**k*root)**m)
            return add + mul*n**(s - 1)*Add(*addargs)

        # 如果 z 是指数且 (z.args[0]/(pi*I)).is_Rational 或者 z 在 [-1, I, -I] 中
        if isinstance(z, exp) and (z.args[0]/(pi*I)).is_Rational or z in [-1, I, -I]:
            # 参见引用的地方？
            if z == -1:
                p, q = S([1, 2])
            elif z == I:
                p, q = S([1, 4])
            elif z == -I:
                p, q = S([-1, 4])
            else:
                arg = z.args[0]/(2*pi*I)
                p, q = S([arg.p, arg.q])
            return Add(*[exp(2*pi*I*k*p/q)/q**s*zeta(s, (k + a)/q)
                         for k in range(q)])

        # 默认情况下返回 lerchphi 函数的结果
        return lerchphi(z, s, a)
    
    # 对象方法，用于计算关于自变量的偏导数
    def fdiff(self, argindex=1):
        # 解构参数
        z, s, a = self.args
        # 如果参数索引为 3
        if argindex == 3:
            return -s*lerchphi(z, s + 1, a)
        # 如果参数索引为 1
        elif argindex == 1:
            return (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError

    # 对象方法，辅助重写函数以匹配目标
    def _eval_rewrite_helper(self, target):
        # 调用 _eval_expand_func 方法
        res = self._eval_expand_func()
        # 如果结果包含目标，则返回结果
        if res.has(target):
            return res
        else:
            return self

    # 对象方法，重写为 zeta 函数的表达形式
    def _eval_rewrite_as_zeta(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(zeta)
    # 定义一个方法 `_eval_rewrite_as_polylog`，接受参数 `z`, `s`, `a` 和其他关键字参数 `kwargs`
    def _eval_rewrite_as_polylog(self, z, s, a, **kwargs):
        # 调用内部方法 `_eval_rewrite_helper`，传入参数 `polylog`，并返回其结果
        return self._eval_rewrite_helper(polylog)
# 定义一个名为 polylog 的类，继承自 Function 类
class polylog(Function):
    r"""
    Polylogarithm function.

    Explanation
    ===========

    For $|z| < 1$ and $s \in \mathbb{C}$, the polylogarithm is
    defined by

    .. math:: \operatorname{Li}_s(z) = \sum_{n=1}^\infty \frac{z^n}{n^s},

    where the standard branch of the argument is used for $n$. It admits
    an analytic continuation which is branched at $z=1$ (notably not on the
    sheet of initial definition), $z=0$ and $z=\infty$.

    The name polylogarithm comes from the fact that for $s=1$, the
    polylogarithm is related to the ordinary logarithm (see examples), and that

    .. math:: \operatorname{Li}_{s+1}(z) =
                    \int_0^z \frac{\operatorname{Li}_s(t)}{t} \mathrm{d}t.

    The polylogarithm is a special case of the Lerch transcendent:

    .. math:: \operatorname{Li}_{s}(z) = z \Phi(z, s, 1).

    Examples
    ========

    For $z \in \{0, 1, -1\}$, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If $s$ is a negative integer, $0$ or $1$, the polylogarithm can be
    expressed using elementary functions. This can be done using
    ``expand_func()``:

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(1 - z)
    >>> expand_func(polylog(0, z))
    z/(1 - z)

    The derivative with respect to $z$ can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)

    See Also
    ========

    zeta, lerchphi

    """

    @classmethod
    # 定义一个类方法 eval，用于计算给定参数 s 和 z 的特殊函数值
    def eval(cls, s, z):
        # 如果 z 是数值类型
        if z.is_number:
            # 如果 z 是 1，则返回 zeta(s) 的计算结果
            if z is S.One:
                return zeta(s)
            # 如果 z 是 -1，则返回 -dirichlet_eta(s) 的计算结果
            elif z is S.NegativeOne:
                return -dirichlet_eta(s)
            # 如果 z 是 0，则返回 0
            elif z is S.Zero:
                return S.Zero
            # 如果 s 等于 2
            elif s == 2:
                # 获取 dilogtable 表
                dilogtable = _dilogtable()
                # 如果 z 在 dilogtable 中，则返回对应的值
                if z in dilogtable:
                    return dilogtable[z]

        # 如果 z 是 0，则返回 0
        if z.is_zero:
            return S.Zero

        # 尝试判断 z 是否为 1，以避免在具有奇点的表达式中替换
        zone = z.equals(S.One)

        # 如果 z 等于 1，则返回 zeta(s) 的计算结果
        if zone:
            return zeta(s)
        # 如果 z 不等于 1
        elif zone is False:
            # 当 s 为 0 或 -1 时，使用显式公式进行计算
            if s is S.Zero:
                return z/(1 - z)
            elif s is S.NegativeOne:
                return z/(1 - z)**2
            # 如果 s 是 0，则返回 z/(1 - z)
            if s.is_zero:
                return z/(1 - z)

        # polylog 函数在复平面上分支，但不是在单位圆上
        if z.has(exp_polar, polar_lift) and (zone or (Abs(z) <= S.One) == True):
            # 返回类方法 cls 的计算结果，使用 unpolarify 函数处理 z
            return cls(s, unpolarify(z))

    # 定义一个方法 fdiff，用于计算参数 z 的 polylog(s - 1, z) / z
    def fdiff(self, argindex=1):
        # 获取参数 s 和 z
        s, z = self.args
        # 如果 argindex 等于 2，则返回 polylog(s - 1, z) / z 的计算结果
        if argindex == 2:
            return polylog(s - 1, z)/z
        # 如果 argindex 不等于 2，则引发 ArgumentIndexError 异常

    # 定义一个内部方法 _eval_rewrite_as_lerchphi，使用 lerchphi 函数重写为 z * lerchphi(z, s, 1)
    def _eval_rewrite_as_lerchphi(self, s, z, **kwargs):
        return z * lerchphi(z, s, 1)

    # 定义一个内部方法 _eval_expand_func，根据 hints 参数扩展 polylog(s, z) 函数
    def _eval_expand_func(self, **hints):
        # 获取参数 s 和 z
        s, z = self.args
        # 如果 s 等于 1，则返回 -log(1 - z)
        if s == 1:
            return -log(1 - z)
        # 如果 s 是整数且小于等于 0
        if s.is_Integer and s <= 0:
            # 使用 Dummy 变量 u
            u = Dummy('u')
            # 初始化 start 为 u / (1 - u)
            start = u / (1 - u)
            # 迭代 -s 次数
            for _ in range(-s):
                # 更新 start 为 u * start.diff(u)
                start = u * start.diff(u)
            # 返回展开乘法后的表达式，并将 u 替换为 z
            return expand_mul(start).subs(u, z)
        # 否则返回 polylog(s, z) 的计算结果
        return polylog(s, z)

    # 定义一个内部方法 _eval_is_zero，用于判断参数 z 是否为 0
    def _eval_is_zero(self):
        # 获取参数 z
        z = self.args[1]
        # 如果 z 是 0，则返回 True
        if z.is_zero:
            return True
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 导入必要的模块和类
        from sympy.series.order import Order
        # 获取当前对象的参数 nu 和 z
        nu, z = self.args

        # 计算 z 在 x = 0 处的值
        z0 = z.subs(x, 0)
        # 如果 z0 是 NaN，则计算 z 在 x = 0 时的极限
        if z0 is S.NaN:
            z0 = z.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        # 如果 z0 是零
        if z0.is_zero:
            # 对于幂小于 1 的情况，需要单独计算项数以避免使用错误的 n 重复调用 _eval_nseries

            # 尝试获取 z 的主导项和指数
            try:
                _, exp = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self  # 如果获取失败，则返回当前对象

            # 如果指数是正数
            if exp.is_positive:
                # 计算新的项数 newn
                newn = ceiling(n/exp)
                # 创建 Order 对象 o，表示高阶项
                o = Order(x**n, x)
                # 调用 z 的 _eval_nseries 方法，移除高阶项后的结果
                r = z._eval_nseries(x, n, logx, cdir).removeO()
                # 如果移除高阶项后的结果是零，则返回高阶项 o
                if r is S.Zero:
                    return o

                # 初始化首项 term
                term = r
                # 初始化序列 s，包含首项
                s = [term]
                # 生成序列 s 的其余项，直到达到 newn 项
                for k in range(2, newn):
                    term *= r
                    s.append(term/k**nu)
                # 返回项数 newn 的多项式级数和高阶项 o 的和
                return Add(*s) + o

        # 如果 z0 不为零或者不适用上述情况，则调用父类的 _eval_nseries 方法处理
        return super(polylog, self)._eval_nseries(x, n, logx, cdir)
###############################################################################
###################### HURWITZ GENERALIZED ZETA FUNCTION ######################
###############################################################################

# 定义一个名为 `zeta` 的类，继承自 `Function` 类
class zeta(Function):
    r"""
    Hurwitz zeta function (or Riemann zeta function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$ and $\operatorname{Re}(s) > 1$, this
    function is defined as

    .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},

    where the standard choice of argument for $n + a$ is used. For fixed
    $a$ not a nonpositive integer the Hurwitz zeta function admits a
    meromorphic continuation to all of $\mathbb{C}$; it is an unbranched
    function with a simple pole at $s = 1$.

    The Hurwitz zeta function is a special case of the Lerch transcendent:

    .. math:: \zeta(s, a) = \Phi(1, s, a).

    This formula defines an analytic continuation for all possible values of
    $s$ and $a$ (also $\operatorname{Re}(a) < 0$), see the documentation of
    :class:`lerchphi` for a description of the branching behavior.

    If no value is passed for $a$ a default value of $a = 1$ is assumed,
    yielding the Riemann zeta function.

    Examples
    ========

    For $a = 1$ the Hurwitz zeta function reduces to the famous Riemann
    zeta function:

    .. math:: \zeta(s, 1) = \zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}.

    >>> from sympy import zeta
    >>> from sympy.abc import s
    >>> zeta(s, 1)
    zeta(s)
    >>> zeta(s)
    zeta(s)

    The Riemann zeta function can also be expressed using the Dirichlet eta
    function:

    >>> from sympy import dirichlet_eta
    >>> zeta(s).rewrite(dirichlet_eta)
    dirichlet_eta(s)/(1 - 2**(1 - s))

    The Riemann zeta function at nonnegative even and negative integer
    values is related to the Bernoulli numbers and polynomials:

    >>> zeta(2)
    pi**2/6
    >>> zeta(4)
    pi**4/90
    >>> zeta(0)
    -1/2
    >>> zeta(-1)
    -1/12
    >>> zeta(-4)
    0

    The specific formulae are:

    .. math:: \zeta(2n) = -\frac{(2\pi i)^{2n} B_{2n}}{2(2n)!}
    .. math:: \zeta(-n,a) = -\frac{B_{n+1}(a)}{n+1}

    No closed-form expressions are known at positive odd integers, but
    numerical evaluation is possible:

    >>> zeta(3).n()
    1.20205690315959

    The derivative of $\zeta(s, a)$ with respect to $a$ can be computed:

    >>> from sympy.abc import a
    >>> zeta(s, a).diff(a)
    -s*zeta(s + 1, a)

    However the derivative with respect to $s$ has no useful closed form
    expression:

    >>> zeta(s, a).diff(s)
    Derivative(zeta(s, a), s)

    The Hurwitz zeta function can be expressed in terms of the Lerch
    transcendent, :class:`~.lerchphi`:

    >>> from sympy import lerchphi
    >>> zeta(s, a).rewrite(lerchphi)
    lerchphi(1, s, a)

    See Also
    ========

    dirichlet_eta, lerchphi, polylog

    References
    ==========

    .. [1] https://dlmf.nist.gov/25.11
    # 引用维基百科上的Hurwitz zeta函数的说明 [2]
    """
    类方法eval，用于计算Hurwitz zeta函数的特定值
    s: 第一个参数，指定zeta函数的参数
    a: 可选参数，默认为None，指定zeta函数的第二个参数，默认为1
    """
    @classmethod
    def eval(cls, s, a=None):
        # 如果a为1，返回zeta函数的单参数形式
        if a is S.One:
            return cls(s)
        # 如果s或a为NaN，返回NaN
        elif s is S.NaN or a is S.NaN:
            return S.NaN
        # 如果s为1，返回ComplexInfinity（复数无穷大）
        elif s is S.One:
            return S.ComplexInfinity
        # 如果s为Infinity，返回1
        elif s is S.Infinity:
            return S.One
        # 如果a为Infinity，返回0
        elif a is S.Infinity:
            return S.Zero

        # 检查s是否为整数
        sint = s.is_Integer
        # 如果a为None，设为1
        if a is None:
            a = S.One
        # 如果s为整数且非正，计算对应的bernoulli数值
        if sint and s.is_nonpositive:
            return bernoulli(1-s, a) / (s-1)
        # 如果a为1且s为偶数整数，返回特定的数学表达式
        elif a is S.One:
            if sint and s.is_even:
                return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        # 如果s为整数且a为正整数，返回zeta函数减去调和数的结果
        elif sint and a.is_Integer and a.is_positive:
            return cls(s) - harmonic(a-1, s)
        # 如果a为整数且非正，并且s不是整数或者s非正，则返回NaN
        elif a.is_Integer and a.is_nonpositive and \
                (s.is_integer is False or s.is_nonpositive is False):
            return S.NaN

    # 将zeta函数重写为bernoulli函数的表达式
    def _eval_rewrite_as_bernoulli(self, s, a=1, **kwargs):
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        return bernoulli(1-s, a) / (s-1)

    # 将zeta函数重写为dirichlet_eta函数的表达式
    def _eval_rewrite_as_dirichlet_eta(self, s, a=1, **kwargs):
        if a != 1:
            return self
        s = self.args[0]
        return dirichlet_eta(s)/(1 - 2**(1 - s))

    # 将zeta函数重写为lerchphi函数的表达式
    def _eval_rewrite_as_lerchphi(self, s, a=1, **kwargs):
        return lerchphi(1, s, a)

    # 检查zeta函数是否有限
    def _eval_is_finite(self):
        arg_is_one = (self.args[0] - 1).is_zero
        if arg_is_one is not None:
            return not arg_is_one

    # 展开zeta函数的函数表达式
    def _eval_expand_func(self, **hints):
        s = self.args[0]
        a = self.args[1] if len(self.args) > 1 else S.One
        # 如果a为整数
        if a.is_integer:
            # 如果a为正整数，返回zeta函数减去调和数的结果
            if a.is_positive:
                return zeta(s) - harmonic(a-1, s)
            # 如果a为非正整数，并且s不是整数或者s非正，则返回NaN
            if a.is_nonpositive and (s.is_integer is False or
                    s.is_nonpositive is False):
                return S.NaN
        return self

    # zeta函数的偏导数
    def fdiff(self, argindex=1):
        # 如果参数长度为2
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (1,)
        # 如果argindex为2，返回特定的偏导数表达式
        if argindex == 2:
            return -s*zeta(s + 1, a)
        else:
            raise ArgumentIndexError

    # 将zeta函数表达为其主导项
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 如果参数长度为2
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (S.One,)

        try:
            c, e = a.leadterm(x)
        except NotImplementedError:
            return self

        # 如果e为负数且s不为正数，抛出NotImplementedError
        if e.is_negative and not s.is_positive:
            raise NotImplementedError

        return super(zeta, self)._eval_as_leading_term(x, logx, cdir)
class dirichlet_eta(Function):
    r"""
    Dirichlet eta function.

    Explanation
    ===========

    For $\operatorname{Re}(s) > 0$ and $0 < x \le 1$, this function is defined as

    .. math:: \eta(s, a) = \sum_{n=0}^\infty \frac{(-1)^n}{(n+a)^s}.

    It admits a unique analytic continuation to all of $\mathbb{C}$ for any
    fixed $a$ not a nonpositive integer. It is an entire, unbranched function.

    It can be expressed using the Hurwitz zeta function as

    .. math:: \eta(s, a) = \zeta(s,a) - 2^{1-s} \zeta\left(s, \frac{a+1}{2}\right)

    and using the generalized Genocchi function as

    .. math:: \eta(s, a) = \frac{G(1-s, a)}{2(s-1)}.

    In both cases the limiting value of $\log2 - \psi(a) + \psi\left(\frac{a+1}{2}\right)$
    is used when $s = 1$.

    Examples
    ========

    >>> from sympy import dirichlet_eta, zeta
    >>> from sympy.abc import s
    >>> dirichlet_eta(s).rewrite(zeta)
    Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True))

    See Also
    ========

    zeta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_eta_function
    .. [2] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    # 类方法用于求解 Dirichlet eta 函数的值，支持指定参数 a
    def eval(cls, s, a=None):
        # 当 a 为 1 时，返回 Dirichlet eta 函数的值
        if a is S.One:
            return cls(s)
        # 当 a 为 None 时，根据 s 的值返回不同的结果
        if a is None:
            if s == 1:
                return log(2)
            z = zeta(s)
            if not z.has(zeta):
                return (1 - 2**(1-s)) * z
            return
        # 当 s 等于 1 时，使用特定的公式计算 Dirichlet eta 函数的值
        elif s == 1:
            from sympy.functions.special.gamma_functions import digamma
            return log(2) - digamma(a) + digamma((a+1)/2)
        # 使用 Hurwitz zeta 函数和一般化的 Genocchi 函数来计算 Dirichlet eta 函数的值
        z1 = zeta(s, a)
        z2 = zeta(s, (a+1)/2)
        if not z1.has(zeta) and not z2.has(zeta):
            return z1 - 2**(1-s) * z2

    # 使用 Hurwitz zeta 函数重写 Dirichlet eta 函数
    def _eval_rewrite_as_zeta(self, s, a=1, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        if a == 1:
            return Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1-s)) * zeta(s), True))
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (zeta(s, a) - 2**(1-s) * zeta(s, (a+1)/2), True))

    # 使用一般化的 Genocchi 函数重写 Dirichlet eta 函数
    def _eval_rewrite_as_genocchi(self, s, a=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (genocchi(1-s, a) / (2 * (s-1)), True))

    # 对 Dirichlet eta 函数进行数值计算，精度由参数 prec 指定
    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(zeta)._eval_evalf(prec)


class riemann_xi(Function):
    r"""
    Riemann Xi function.

    Examples
    ========

    The Riemann Xi function is closely related to the Riemann zeta function.
    The zeros of Riemann Xi function are precisely the non-trivial zeros
    of the zeta function.

    >>> from sympy import riemann_xi, zeta
    >>> from sympy.abc import s
    >>> riemann_xi(s).rewrite(zeta)
    s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))


# 计算 Riemann Xi 函数的表达式，根据给定的 s 值计算其值



References
==========


# 参考资料，提供了关于 Riemann Xi 函数的相关信息



.. [1] https://en.wikipedia.org/wiki/Riemann_Xi_function


# 参考链接指向维基百科上关于 Riemann Xi 函数的页面



"""


# 以下为类方法的定义



@classmethod
def eval(cls, s):


# 类方法定义，用于计算 Riemann Xi 函数的值，接受一个参数 s



from sympy.functions.special.gamma_functions import gamma
z = zeta(s)
if s in (S.Zero, S.One):
    return S.Half


# 导入 gamma 函数，计算 zeta 函数在 s 处的值
# 如果 s 是 0 或 1，则返回 1/2



if not isinstance(z, zeta):
    return s*(s - 1)*gamma(s/2)*z/(2*pi**(s/2))


# 如果 z 不是 zeta 类型的对象，则计算 Riemann Xi 函数的表达式



def _eval_rewrite_as_zeta(self, s, **kwargs):


# 定义一个方法，将 Riemann Xi 函数重写为 zeta 函数的表达式



from sympy.functions.special.gamma_functions import gamma
return s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))


# 返回 Riemann Xi 函数的 zeta 函数表达式重写结果
class stieltjes(Function):
    r"""
    Represents Stieltjes constants, $\gamma_{k}$ that occur in
    Laurent Series expansion of the Riemann zeta function.

    Examples
    ========

    >>> from sympy import stieltjes
    >>> from sympy.abc import n, m
    >>> stieltjes(n)
    stieltjes(n)

    The zero'th stieltjes constant:

    >>> stieltjes(0)
    EulerGamma
    >>> stieltjes(0, 1)
    EulerGamma

    For generalized stieltjes constants:

    >>> stieltjes(n, m)
    stieltjes(n, m)

    Constants are only defined for integers >= 0:

    >>> stieltjes(-1)
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stieltjes_constants

    """

    @classmethod
    def eval(cls, n, a=None):
        # 检查参数 a 是否为符号表达式
        if a is not None:
            # 将 a 转换为符号表达式
            a = sympify(a)
            # 如果 a 是 NaN，则返回 NaN
            if a is S.NaN:
                return S.NaN
            # 如果 a 是整数且非正数，返回复数无穷大
            if a.is_Integer and a.is_nonpositive:
                return S.ComplexInfinity

        # 检查参数 n 是否为数值
        if n.is_Number:
            # 如果 n 是 NaN，则返回 NaN
            if n is S.NaN:
                return S.NaN
            # 如果 n 小于 0，则返回复数无穷大
            elif n < 0:
                return S.ComplexInfinity
            # 如果 n 不是整数，则返回复数无穷大
            elif not n.is_Integer:
                return S.ComplexInfinity
            # 如果 n 是零且 a 为 None 或 1，则返回欧拉常数 EulerGamma
            elif n is S.Zero and a in [None, 1]:
                return S.EulerGamma

        # 如果 n 是扩展负数，则返回复数无穷大
        if n.is_extended_negative:
            return S.ComplexInfinity

        # 如果 n 是零且 a 为 None 或 1，则返回欧拉常数 EulerGamma
        if n.is_zero and a in [None, 1]:
            return S.EulerGamma

        # 如果 n 不是整数，则返回复数无穷大
        if n.is_integer == False:
            return S.ComplexInfinity


@cacheit
def _dilogtable():
    # 返回一个预定义的对数二函数表
    return {
        S.Half: pi**2/12 - log(2)**2/2,
        Integer(2) : pi**2/4 - I*pi*log(2),
        -(sqrt(5) - 1)/2 : -pi**2/15 + log((sqrt(5)-1)/2)**2/2,
        -(sqrt(5) + 1)/2 : -pi**2/10 - log((sqrt(5)+1)/2)**2,
        (3 - sqrt(5))/2 : pi**2/15 - log((sqrt(5)-1)/2)**2,
        (sqrt(5) - 1)/2 : pi**2/10 - log((sqrt(5)-1)/2)**2,
        I : I*S.Catalan - pi**2/48,
        -I : -I*S.Catalan - pi**2/48,
        1 - I : pi**2/16 - I*S.Catalan - pi*I/4*log(2),
        1 + I : pi**2/16 + I*S.Catalan + pi*I/4*log(2),
        (1 - I)/2 : -log(2)**2/8 + pi*I*log(2)/8 + 5*pi**2/96 - I*S.Catalan
    }
```