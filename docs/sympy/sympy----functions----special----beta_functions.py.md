# `D:\src\scipysrc\sympy\sympy\functions\special\beta_functions.py`

```
# 从 sympy.core 模块导入 S
# 从 sympy.core.function 模块导入 Function 和 ArgumentIndexError
# 从 sympy.core.symbol 模块导入 Dummy 和 uniquely_named_symbol
# 从 sympy.functions.special.gamma_functions 模块导入 gamma 和 digamma
# 从 sympy.functions.combinatorial.numbers 模块导入 catalan
# 从 sympy.functions.elementary.complexes 模块导入 conjugate

# 定义函数 betainc_mpmath_fix，修复 mpmath #569 和 SymPy #20569 的问题
def betainc_mpmath_fix(a, b, x1, x2, reg=0):
    # 从 mpmath 模块导入 betainc 和 mpf
    from mpmath import betainc, mpf
    # 如果 x1 等于 x2，则返回 mpf(0)
    if x1 == x2:
        return mpf(0)
    else:
        # 否则调用 betainc 函数计算 Beta 函数的不完全积分
        return betainc(a, b, x1, x2, reg)

###############################################################################
############################ COMPLETE BETA  FUNCTION ##########################
###############################################################################

# 定义 beta 类，继承自 Function
class beta(Function):
    r"""
    Beta 积分被 Legendre 称为一类欧拉积分的积分：

    .. math::
        \mathrm{B}(x,y)  \int^{1}_{0} t^{x-1} (1-t)^{y-1} \mathrm{d}t.

    Explanation
    ===========

    Beta 函数或欧拉第一类积分与 gamma 函数密切相关。Beta 函数经常用于概率论
    和数理统计。它满足以下性质：

    .. math::
        \mathrm{B}(a,1) = \frac{1}{a} \\
        \mathrm{B}(a,b) = \mathrm{B}(b,a)  \\
        \mathrm{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}

    因此对于整数值的 $a$ 和 $b$：

    .. math::
        \mathrm{B} = \frac{(a-1)! (b-1)!}{(a+b-1)!}

    当 `x = y` 时的 Beta 函数的特殊情况是中心 Beta 函数。它满足以下性质：

    .. math::
        \mathrm{B}(x) = 2^{1 - 2x}\mathrm{B}(x, \frac{1}{2})
        \mathrm{B}(x) = 2^{1 - 2x} cos(\pi x) \mathrm{B}(\frac{1}{2} - x, x)
        \mathrm{B}(x) = \int_{0}^{1} \frac{t^x}{(1 + t)^{2x}} dt
        \mathrm{B}(x) = \frac{2}{x} \prod_{n = 1}^{\infty} \frac{n(n + 2x)}{(n + x)^2}

    Examples
    ========

    >>> from sympy import I, pi
    >>> from sympy.abc import x, y

    Beta 函数遵循镜像对称性：

    >>> from sympy import beta, conjugate
    >>> conjugate(beta(x, y))
    beta(conjugate(x), conjugate(y))

    支持对 $x$ 和 $y$ 的微分：

    >>> from sympy import beta, diff
    >>> diff(beta(x, y), x)
    (polygamma(0, x) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x, y), y)
    (polygamma(0, y) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x), x)
    2*(polygamma(0, x) - polygamma(0, 2*x))*beta(x, x)

    我们可以对任意复数 $x$ 和 $y$ 数值评估 Beta 函数至任意精度：

    >>> from sympy import beta
    >>> beta(pi).evalf(40)
    0.02671848900111377452242355235388489324562

    >>> beta(1 + I).evalf(20)
    -0.2112723729365330143 - 0.7655283165378005676*I

    See Also
    ========

    gamma: Gamma 函数.
    uppergamma: 上不完全 Gamma 函数.
    """
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function
    .. [2] https://mathworld.wolfram.com/BetaFunction.html
    .. [3] https://dlmf.nist.gov/5.12

    """
    unbranched = True  # 设置变量 unbranched 为 True，表示未分支

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            # 对 x 求导数
            return beta(x, y)*(digamma(x) - digamma(x + y))
        elif argindex == 2:
            # 对 y 求导数
            return beta(x, y)*(digamma(y) - digamma(x + y))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y=None):
        if y is None:
            # 当 y 为空时，返回 beta 函数的特殊情况 beta(x, x)
            return beta(x, x)
        if x.is_Number and y.is_Number:
            # 当 x 和 y 都是数值时，求解 beta 函数的值并返回
            return beta(x, y, evaluate=False).doit()

    def doit(self, **hints):
        x = xold = self.args[0]
        # 处理未评估的单参数 beta 函数
        single_argument = len(self.args) == 1
        y = yold = self.args[0] if single_argument else self.args[1]
        if hints.get('deep', True):
            # 如果 hints 中设置了 deep=True，则深度求解 x 和 y
            x = x.doit(**hints)
            y = y.doit(**hints)
        if y.is_zero or x.is_zero:
            # 如果 y 或 x 是零，则返回复无穷
            return S.ComplexInfinity
        if y is S.One:
            # 当 y 是 1 时，返回 1/x
            return 1/x
        if x is S.One:
            # 当 x 是 1 时，返回 1/y
            return 1/y
        if y == x + 1:
            # 当 y = x + 1 时，返回 1/(x*y*catalan(x))
            return 1/(x*y*catalan(x))
        s = x + y
        if (s.is_integer and s.is_negative and x.is_integer is False and
            y.is_integer is False):
            # 当 s 是整数且为负数，且 x 和 y 都不是整数时，返回 0
            return S.Zero
        if x == xold and y == yold and not single_argument:
            # 如果 x 和 y 与旧值相同且不是单参数函数，则返回自身
            return self
        # 返回 beta 函数的计算结果
        return beta(x, y)

    def _eval_expand_func(self, **hints):
        x, y = self.args
        # 展开为 gamma 函数的形式
        return gamma(x)*gamma(y) / gamma(x + y)

    def _eval_is_real(self):
        # 判断是否为实数
        return self.args[0].is_real and self.args[1].is_real

    def _eval_conjugate(self):
        # 对参数取共轭
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_rewrite_as_gamma(self, x, y, piecewise=True, **kwargs):
        # 重写为 gamma 函数的形式，调用 _eval_expand_func 方法
        return self._eval_expand_func(**kwargs)

    def _eval_rewrite_as_Integral(self, x, y, **kwargs):
        # 重写为积分形式
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [x, y]).name)
        return Integral(t**(x - 1)*(1 - t)**(y - 1), (t, 0, 1))
###############################################################################
########################## INCOMPLETE BETA FUNCTION ###########################
###############################################################################

# 定义一个类 `betainc`，表示不完全贝塔函数
class betainc(Function):
    r"""
    不完全贝塔函数被定义为

    .. math::
        \mathrm{B}_{(x_1, x_2)}(a, b) = \int_{x_1}^{x_2} t^{a - 1} (1 - t)^{b - 1} dt

    不完全贝塔函数是一般化不完全贝塔函数的特例：

    .. math:: \mathrm{B}_z (a, b) = \mathrm{B}_{(0, z)}(a, b)

    不完全贝塔函数满足以下关系：

    .. math:: \mathrm{B}_z (a, b) = (-1)^a \mathrm{B}_{\frac{z}{z - 1}} (a, 1 - a - b)

    贝塔函数是不完全贝塔函数的特例：

    .. math:: \mathrm{B}(a, b) = \mathrm{B}_{1}(a, b)

    示例
    ========

    >>> from sympy import betainc, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    一般化不完全贝塔函数的调用方式如下：

    >>> betainc(a, b, x1, x2)
    betainc(a, b, x1, x2)

    可以通过以下方式获取不完全贝塔函数的特例：

    >>> betainc(a, b, 0, x)
    betainc(a, b, 0, x)

    不完全贝塔函数具有镜像对称性：

    >>> conjugate(betainc(a, b, x1, x2))
    betainc(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    我们可以对任意复数 a, b, x1 和 x2 对不完全贝塔函数进行数值计算：

    >>> from sympy import betainc, I
    >>> betainc(2, 3, 4, 5).evalf(10)
    56.08333333
    >>> betainc(0.75, 1 - 4*I, 0, 2 + 3*I).evalf(25)
    0.2241657956955709603655887 + 0.3619619242700451992411724*I

    可以利用广义超几何函数重写一般化不完全贝塔函数：

    >>> from sympy import hyper
    >>> betainc(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/a

    参见
    ========

    beta: 贝塔函数
    hyper: 广义超几何函数

    参考文献
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    # 计算函数在特定参数上对第几个参数的偏导数
    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # 对 x1 求偏导数
            return -(1 - x1)**(b - 1)*x1**(a - 1)
        elif argindex == 4:
            # 对 x2 求偏导数
            return (1 - x2)**(b - 1)*x2**(a - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    # 使用 mpmath 库计算函数值
    def _eval_mpmath(self):
        return betainc_mpmath_fix, self.args

    # 判断函数是否为实数
    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True
    # 返回函数的共轭值
    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))
    
    # 将对象重写为积分形式
    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        # 创建一个虚拟符号，用于积分变量，并确保其名称唯一
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        # 返回 t**(a - 1)*(1 - t)**(b - 1) 在 (t, x1, x2) 区间上的积分对象
        return Integral(t**(a - 1)*(1 - t)**(b - 1), (t, x1, x2))
    
    # 将对象重写为超几何函数的形式
    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        # 返回超几何函数的计算结果
        return (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a
# 定义一个类 `betainc_regularized`，继承自 `Function` 类
class betainc_regularized(Function):
    r"""
    The Generalized Regularized Incomplete Beta function is given by

    .. math::
        \mathrm{I}_{(x_1, x_2)}(a, b) = \frac{\mathrm{B}_{(x_1, x_2)}(a, b)}{\mathrm{B}(a, b)}

    The Regularized Incomplete Beta function is a special case
    of the Generalized Regularized Incomplete Beta function :

    .. math:: \mathrm{I}_z (a, b) = \mathrm{I}_{(0, z)}(a, b)

    The Regularized Incomplete Beta function is the cumulative distribution
    function of the beta distribution.

    Examples
    ========

    >>> from sympy import betainc_regularized, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Regularized Incomplete Beta
    function is given by:

    >>> betainc_regularized(a, b, x1, x2)
    betainc_regularized(a, b, x1, x2)

    The Regularized Incomplete Beta function
    can be obtained as follows:

    >>> betainc_regularized(a, b, 0, x)
    betainc_regularized(a, b, 0, x)

    The Regularized Incomplete Beta function
    obeys the mirror symmetry:

    >>> conjugate(betainc_regularized(a, b, x1, x2))
    betainc_regularized(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Regularized Incomplete Beta function
    to arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc_regularized, pi, E
    >>> betainc_regularized(1, 2, 0, 0.25).evalf(10)
    0.4375000000
    >>> betainc_regularized(pi, E, 0, 1).evalf(5)
    1.00000

    The Generalized Regularized Incomplete Beta function can be
    expressed in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc_regularized(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/(a*beta(a, b))

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4  # 设置类变量 nargs 为 4，表示此函数需要接收四个参数
    unbranched = True  # 设置类变量 unbranched 为 True，表明此函数是无分支的

    def __new__(cls, a, b, x1, x2):
        return Function.__new__(cls, a, b, x1, x2)  # 调用父类 Function 的构造函数，创建新的实例

    def _eval_mpmath(self):
        return betainc_mpmath_fix, (*self.args, S(1))  # 返回一个修复后的 mpmath 版本的 betainc 函数的结果
    `
        # 计算关于参数索引的偏导数
        def fdiff(self, argindex):
            # 将参数解包为 a, b, x1, x2
            a, b, x1, x2 = self.args
            if argindex == 3:
                # 对 x1 求偏导数
                return -(1 - x1)**(b - 1)*x1**(a - 1) / beta(a, b)
            elif argindex == 4:
                # 对 x2 求偏导数
                return (1 - x2)**(b - 1)*x2**(a - 1) / beta(a, b)
            else:
                # 若参数索引不在范围内，引发参数索引错误
                raise ArgumentIndexError(self, argindex)
    
        # 检查是否所有参数都是实数
        def _eval_is_real(self):
            if all(arg.is_real for arg in self.args):
                return True
    
        # 求共轭复数
        def _eval_conjugate(self):
            # 将函数应用于所有参数的共轭
            return self.func(*map(conjugate, self.args))
    
        # 重写为积分形式
        def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
            from sympy.integrals.integrals import Integral
            # 创建虚拟变量 t，并生成被积函数
            t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
            integrand = t**(a - 1)*(1 - t)**(b - 1)
            # 构造积分表达式
            expr = Integral(integrand, (t, x1, x2))
            return expr / Integral(integrand, (t, 0, 1))
    
        # 重写为超几何函数形式
        def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
            from sympy.functions.special.hyper import hyper
            # 构造超几何函数形式的表达式
            expr = (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a
            return expr / beta(a, b)
```