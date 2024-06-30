# `D:\src\scipysrc\sympy\sympy\stats\crv_types.py`

```
# 导入对数学函数和特殊函数的支持模块
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum  # 导入求和函数
from sympy.core.basic import Basic  # 导入基本数学对象
from sympy.core.function import Lambda  # 导入 Lambda 函数
from sympy.core.numbers import (I, Rational, pi)  # 导入复数、有理数和π
from sympy.core.relational import (Eq, Ne)  # 导入相等和不等式关系
from sympy.core.singleton import S  # 导入单例对象
from sympy.core.symbol import Dummy  # 导入符号对象
from sympy.core.sympify import sympify  # 导入表达式转换函数
from sympy.functions.combinatorial.factorials import (binomial, factorial)  # 导入组合数和阶乘
from sympy.functions.elementary.complexes import (Abs, sign)  # 导入绝对值和符号函数
from sympy.functions.elementary.exponential import log  # 导入对数函数
from sympy.functions.elementary.hyperbolic import sinh  # 导入双曲正弦函数
from sympy.functions.elementary.integers import floor  # 导入向下取整函数
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min  # 导入平方根、最大值和最小值函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import asin  # 导入反正弦函数
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)  # 导入误差函数和指数积分函数
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)  # 导入 Gamma 函数和其变体
from sympy.functions.special.zeta_functions import zeta  # 导入 Riemann zeta 函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.integrals.integrals import integrate  # 导入积分函数
from sympy.logic.boolalg import And  # 导入布尔代数与运算函数
from sympy.sets.sets import Interval  # 导入区间对象
from sympy.matrices import MatrixBase  # 导入矩阵基类
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution  # 导入连续概率空间和分布函数
from sympy.stats.rv import _value_check, is_random  # 导入随机变量值检查和随机性检查函数

oo = S.Infinity  # 设置无穷大对象

# 定义公开的连续随机变量类别
__all__ = ['ContinuousRV',
           'Arcsin',
           'Benini',
           'Beta',
           'BetaNoncentral',
           'BetaPrime',
           'BoundedPareto',
           'Cauchy',
           'Chi',
           'ChiNoncentral',
           'ChiSquared',
           'Dagum',
           'Davis',
           'Erlang',
           'ExGaussian',
           'Exponential',
           'ExponentialPower',
           'FDistribution',
           'FisherZ',
           'Frechet',
           'Gamma',
           'GammaInverse',
           'Gompertz',
           'Gumbel',
           'Kumaraswamy',
           'Laplace',
           'Levy',
           'LogCauchy',
           'Logistic',
           'LogLogistic',
           'LogitNormal',
           'LogNormal',
           'Lomax',
           'Maxwell',
           'Moyal',
           'Nakagami',
           'Normal',
           'GaussianInverse',  # 修正：原注释遗漏此项
           'Pareto',
           'PowerFunction',
           'QuadraticU',
           'RaisedCosine',
           'Rayleigh',
           'Reciprocal',
           'StudentT',
           'ShiftedGompertz',
           'Trapezoidal',
           'Triangular',
           'Uniform',
           'UniformSum',
           'VonMises',
           'Wald',
           'Weibull',
           'WignerSemicircle',
           ]


@is_random.register(MatrixBase)
def _(x):
    # 注册矩阵基类为随机变量
    """
    Registers MatrixBase objects as random variables.

    Parameters
    ----------
    x : MatrixBase
        The matrix object to be checked.

    Returns
    -------
    bool
        True if x is a random variable, False otherwise.
    """
    # 返回检查结果
    return True
    # 返回一个布尔值，指示在列表 x 中是否存在满足 is_random 函数条件的任意元素
    return any(is_random(i) for i in x)
def rv(symbol, cls, args, **kwargs):
    # 将 args 列表中的每个元素都应用 sympify 函数，转换成 SymPy 对象
    args = list(map(sympify, args))
    # 使用 cls 类创建一个分布对象，参数为 args 中的元素
    dist = cls(*args)
    # 如果 kwargs 中有 'check' 键且其值为 True，则检查分布对象是否有效
    if kwargs.pop('check', True):
        dist.check(*args)
    # 创建一个 SingleContinuousPSpace 对象，表示单一连续随机变量的概率空间
    pspace = SingleContinuousPSpace(symbol, dist)
    # 如果 args 中有任何一个元素是随机变量，则创建复合概率空间对象
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    # 返回随机变量的值
    return pspace.value


class ContinuousDistributionHandmade(SingleContinuousDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=Interval(-oo, oo)):
        # 使用给定的 pdf 和 set 创建一个新的 ContinuousDistributionHandmade 对象
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        # 返回该分布对象的定义域
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        # 创建一个虚拟符号 x
        x = Dummy('x')
        # 计算 pdf 在 set 区间上的积分值
        val = integrate(pdf(x), (x, set))
        # 检查 pdf 是否在给定的区间上积分为 1，否则引发异常
        _value_check(Eq(val, 1) != S.false, "The pdf on the given set is incorrect.")


def ContinuousRV(symbol, density, set=Interval(-oo, oo), **kwargs):
    """
    Create a Continuous Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set/Interval
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.


    Returns
    =======

    RandomSymbol

    Many common continuous random variable types are already implemented.
    This function should be necessary only very rarely.


    Examples
    ========

    >>> from sympy import Symbol, sqrt, exp, pi
    >>> from sympy.stats import ContinuousRV, P, E

    >>> x = Symbol("x")

    >>> pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)) # Normal distribution
    >>> X = ContinuousRV(x, pdf)

    >>> E(X)
    0
    >>> P(X>0)
    1/2
    """
    # 将 density 定义为在 set 区间上的 Piecewise 函数
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    # 将 pdf 转换为 Lambda 函数，即用 symbol 表示的密度函数
    pdf = Lambda(symbol, pdf)
    # 将 kwargs 中的 'check' 键值设为 False（默认为 False），与 rv 函数的默认相反
    kwargs['check'] = kwargs.pop('check', False)
    # 调用 rv 函数创建随机变量，并返回其值
    return rv(symbol.name, ContinuousDistributionHandmade, (pdf, set), **kwargs)

########################################
# Continuous Probability Distributions #
########################################

#-------------------------------------------------------------------------------
# Arcsin distribution ----------------------------------------------------------

class ArcsinDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        # 返回该分布的定义域为区间 [a, b]
        return Interval(self.a, self.b)

    def pdf(self, x):
        a, b = self.a, self.b
        # 返回 arcsin 分布在 x 处的概率密度函数值
        return 1/(pi*sqrt((x - a)*(b - x)))
    # 定义累积分布函数（CDF），接受一个参数 x
    def _cdf(self, x):
        # 从对象的属性中获取区间的上下限 a 和 b
        a, b = self.a, self.b
        # 返回分段函数 Piecewise 对象，根据输入的 x 返回不同的值：
        # 如果 x 小于 a，则返回 0
        # 如果 x 在 a 到 b 之间，则返回 2 * arcsin(sqrt((x - a) / (b - a))) / pi
        # 如果 x 大于等于 b，则返回 1
        return Piecewise(
            (S.Zero, x < a),
            (2*asin(sqrt((x - a)/(b - a)))/pi, x <= b),
            (S.One, True))
def Arcsin(name, a=0, b=1):
    r"""
    Create a Continuous Random Variable with an arcsin distribution.

    The density of the arcsin distribution is given by

    .. math::
        f(x) := \frac{1}{\pi\sqrt{(x-a)(b-x)}}

    with :math:`x \in (a,b)`. It must hold that :math:`-\infty < a < b < \infty`.

    Parameters
    ==========

    a : Real number, the left interval boundary
    b : Real number, the right interval boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Arcsin, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = Arcsin("x", a, b)

    >>> density(X)(z)
    1/(pi*sqrt((-a + z)*(b - z)))

    >>> cdf(X)(z)
    Piecewise((0, a > z),
            (2*asin(sqrt((-a + z)/(-a + b)))/pi, b >= z),
            (1, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Arcsine_distribution

    """

    # 返回一个随机变量，使用ArcsinDistribution作为分布，参数为(a, b)
    return rv(name, ArcsinDistribution, (a, b))

#-------------------------------------------------------------------------------
# Benini distribution ----------------------------------------------------------

class BeniniDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'sigma')

    @staticmethod
    def check(alpha, beta, sigma):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(sigma > 0, "Scale parameter Sigma must be positive.")

    @property
    def set(self):
        # 返回一个区间对象，表示分布的支持区间为(sigma, oo)
        return Interval(self.sigma, oo)

    def pdf(self, x):
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        # 计算Benini分布的概率密度函数值
        return (exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2)
               *(alpha/x + 2*beta*log(x/sigma)/x))

    def _moment_generating_function(self, t):
        # 抛出未实现错误，因为Benini分布没有存在的矩生成函数
        raise NotImplementedError('The moment generating function of the '
                                  'Benini distribution does not exist.')

def Benini(name, alpha, beta, sigma):
    r"""
    Create a Continuous Random Variable with a Benini distribution.

    The density of the Benini distribution is given by

    .. math::
        f(x) := e^{-\alpha\log{\frac{x}{\sigma}}
                -\beta\log^2\left[{\frac{x}{\sigma}}\right]}
                \left(\frac{\alpha}{x}+\frac{2\beta\log{\frac{x}{\sigma}}}{x}\right)

    This is a heavy-tailed distribution and is also known as the log-Rayleigh
    distribution.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape
    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Benini, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    # 定义一个符号 sigma，其值为正数
    sigma = Symbol("sigma", positive=True)
    # 定义一个符号 z
    z = Symbol("z")

    # 创建一个 Benini 分布的随机变量 X，命名为 "x"，参数为 alpha, beta, sigma
    X = Benini("x", alpha, beta, sigma)

    # 计算 X 的密度函数在 z 处的取值，并打印结果（不使用 Unicode）
    D = density(X)(z)
    pprint(D, use_unicode=False)
    # 打印的结果是 Benini 分布的密度函数表达式

    # 计算 X 的累积分布函数在 z 处的取值
    cdf(X)(z)
    # 返回一个分段函数，如果 sigma <= z，则返回 1 - exp(-alpha*log(z/sigma) - beta*log(z/sigma)**2)，否则返回 0

    """

    返回一个符合 Benini 分布的随机变量，命名为 name，参数为 alpha, beta, sigma
    """
    return rv(name, BeniniDistribution, (alpha, beta, sigma))
#-------------------------------------------------------------------------------
# Beta distribution ------------------------------------------------------------

class BetaDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')  # 定义 Beta 分布的参数名

    set = Interval(0, 1)  # 设置分布的定义域为 [0, 1]

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")  # 检查参数 alpha 是否为正数
        _value_check(beta > 0, "Shape parameter Beta must be positive.")    # 检查参数 beta 是否为正数

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1) * (1 - x)**(beta - 1) / beta_fn(alpha, beta)  # Beta 分布的概率密度函数

    def _characteristic_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), I*t)  # Beta 分布的特征函数

    def _moment_generating_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), t)  # Beta 分布的矩生成函数

def Beta(name, alpha, beta):
    r"""
    Create a Continuous Random Variable with a Beta distribution.

    The density of the Beta distribution is given by

    .. math::
        f(x) := \frac{x^{\alpha-1}(1-x)^{\beta-1}} {\mathrm{B}(\alpha,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Beta, density, E, variance
    >>> from sympy import Symbol, simplify, pprint, factor

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Beta("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     alpha - 1        beta - 1
    z         *(1 - z)
    --------------------------
          B(alpha, beta)

    >>> simplify(E(X))
    alpha/(alpha + beta)

    >>> factor(simplify(variance(X)))
    alpha*beta/((alpha + beta)**2*(alpha + beta + 1))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    .. [2] https://mathworld.wolfram.com/BetaDistribution.html

    """
    return rv(name, BetaDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Noncentral Beta distribution ------------------------------------------------------------

class BetaNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'lamda')  # 定义非中心 Beta 分布的参数名

    set = Interval(0, 1)  # 设置分布的定义域为 [0, 1]

    @staticmethod
    def check(alpha, beta, lamda):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")  # 检查参数 alpha 是否为正数
        _value_check(beta > 0, "Shape parameter Beta must be positive.")    # 检查参数 beta 是否为正数
        _value_check(lamda >= 0, "Noncentrality parameter Lambda must be positive")  # 检查参数 lamda 是否为非负数

    def pdf(self, x):
        alpha, beta, lamda = self.alpha, self.beta, self.lamda
        k = Dummy("k")
        return Sum(exp(-lamda / 2) * (lamda / 2)**k * x**(alpha + k - 1) *(
            1 - x)**(beta - 1) / (factorial(k) * beta_fn(alpha + k, beta)), (k, 0, oo))  # 非中心 Beta 分布的概率密度函数
def BetaNoncentral(name, alpha, beta, lamda):
    r"""
    Create a Continuous Random Variable with a Type I Noncentral Beta distribution.

    The density of the Noncentral Beta distribution is given by

    .. math::
        f(x) := \sum_{k=0}^\infty e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}
                \frac{x^{\alpha+k-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha+k,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
        Shape parameter of the distribution.
    beta : Real number, `\beta > 0`, a shape
        Shape parameter of the distribution.
    lamda : Real number, `\lambda \geq 0`, noncentrality parameter
        Noncentrality parameter of the distribution.

    Returns
    =======

    RandomSymbol
        A symbolic representation of a random variable.

    Examples
    ========

    >>> from sympy.stats import BetaNoncentral, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> lamda = Symbol("lamda", nonnegative=True)
    >>> z = Symbol("z")

    >>> X = BetaNoncentral("x", alpha, beta, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      oo
    _____
    \    `
     \                                              -lamda
      \                          k                  -------
       \    k + alpha - 1 /lamda\         beta - 1     2
        )  z             *|-----| *(1 - z)        *e
       /                  \  2  /
      /    ------------------------------------------------
     /                  B(k + alpha, beta)*k!
    /____,
    k = 0

    Compute cdf with specific 'x', 'alpha', 'beta' and 'lamda' values as follows:

    >>> cdf(BetaNoncentral("x", 1, 1, 1), evaluate=False)(2).doit()
    2*exp(1/2)

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general x, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_beta_distribution
    .. [2] https://reference.wolfram.com/language/ref/NoncentralBetaDistribution.html

    """

    return rv(name, BetaNoncentralDistribution, (alpha, beta, lamda))
    alpha : 实数，`\alpha > 0`，表示形状参数 alpha
    beta : 实数，`\beta > 0`，表示形状参数 beta

    Returns
    =======
    RandomSymbol  返回一个随机符号对象

    Examples
    ========
    
    >>> from sympy.stats import BetaPrime, density  导入 BetaPrime 分布和密度函数
    >>> from sympy import Symbol, pprint  导入符号和 pretty print 函数

    >>> alpha = Symbol("alpha", positive=True)  创建一个命名为 alpha 的正数符号
    >>> beta = Symbol("beta", positive=True)  创建一个命名为 beta 的正数符号
    >>> z = Symbol("z")  创建一个命名为 z 的符号

    >>> X = BetaPrime("x", alpha, beta)  创建一个 BetaPrime 随机变量 X，使用 alpha 和 beta 作为参数

    >>> D = density(X)(z)  计算 X 在 z 处的密度函数值
    >>> pprint(D, use_unicode=False)  使用非 Unicode 字符打印密度函数表达式
     alpha - 1        -alpha - beta
    z         *(z + 1)
    -------------------------------
             B(alpha, beta)  分母是 Beta 函数 B(alpha, beta)

    References
    ==========
    参考资料

    .. [1] https://en.wikipedia.org/wiki/Beta_prime_distribution  Beta' 分布的 Wikipedia 页面
    .. [2] https://mathworld.wolfram.com/BetaPrimeDistribution.html  Beta' 分布的 MathWorld 页面

    """

    return rv(name, BetaPrimeDistribution, (alpha, beta))  返回使用给定参数 alpha 和 beta 创建的 BetaPrimeDistribution 随机变量对象
#-------------------------------------------------------------------------------
# Bounded Pareto Distribution --------------------------------------------------

class BoundedParetoDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'left', 'right')

    @property
    def set(self):
        return Interval(self.left, self.right)

    @staticmethod
    def check(alpha, left, right):
        _value_check (alpha.is_positive, "Shape must be positive.")  # 检查形状参数是否为正数
        _value_check (left.is_positive, "Left value should be positive.")  # 检查左边界是否为正数
        _value_check (right > left, "Right should be greater than left.")  # 检查右边界是否大于左边界

    def pdf(self, x):
        alpha, left, right = self.alpha, self.left, self.right
        num = alpha * (left**alpha) * x**(- alpha -1)  # 计算概率密度函数的分子部分
        den = 1 - (left/right)**alpha  # 计算概率密度函数的分母部分
        return num/den  # 返回概率密度函数值

def BoundedPareto(name, alpha, left, right):
    r"""
    Create a continuous random variable with a Bounded Pareto distribution.

    The density of the Bounded Pareto distribution is given by

    .. math::
        f(x) := \frac{\alpha L^{\alpha}x^{-\alpha-1}}{1-(\frac{L}{H})^{\alpha}}

    Parameters
    ==========

    alpha : Real Number, `\alpha > 0`
        Shape parameter
    left : Real Number, `left > 0`
        Location parameter
    right : Real Number, `right > left`
        Location parameter

    Examples
    ========

    >>> from sympy.stats import BoundedPareto, density, cdf, E
    >>> from sympy import symbols
    >>> L, H = symbols('L, H', positive=True)
    >>> X = BoundedPareto('X', 2, L, H)
    >>> x = symbols('x')
    >>> density(X)(x)
    2*L**2/(x**3*(1 - L**2/H**2))
    >>> cdf(X)(x)
    Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))
    >>> E(X).simplify()
    2*H*L/(H + L)

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution

    """
    return rv (name, BoundedParetoDistribution, (alpha, left, right))

# ------------------------------------------------------------------------------
# Cauchy distribution ----------------------------------------------------------

class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        _value_check(gamma > 0, "Scale parameter Gamma must be positive.")  # 检查尺度参数是否为正数
        _value_check(x0.is_real, "Location parameter must be real.")  # 检查位置参数是否为实数

    def pdf(self, x):
        return 1/(pi*self.gamma*(1 + ((x - self.x0)/self.gamma)**2))  # 计算概率密度函数

    def _cdf(self, x):
        x0, gamma = self.x0, self.gamma
        return (1/pi)*atan((x - x0)/gamma) + S.Half  # 计算累积分布函数

    def _characteristic_function(self, t):
        return exp(self.x0 * I * t -  self.gamma * Abs(t))  # 计算特征函数

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Cauchy distribution does not exist.")  # 抛出未实现的异常信息
    # 定义一个私有方法 _quantile，用于计算分位数
    def _quantile(self, p):
        # 返回计算得到的分位数值，使用了类中的 x0 和 gamma 属性以及数学函数 tan 和常数 pi
        return self.x0 + self.gamma * tan(pi * (p - S.Half))
def Cauchy(name, x0, gamma):
    r"""
    Create a continuous random variable with a Cauchy distribution.

    The density of the Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi \gamma [1 + {(\frac{x-x_0}{\gamma})}^2]}

    Parameters
    ==========

    x0 : Real number, the location
    gamma : Real number, `\gamma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Cauchy, density
    >>> from sympy import Symbol

    >>> x0 = Symbol("x0")
    >>> gamma = Symbol("gamma", positive=True)
    >>> z = Symbol("z")

    >>> X = Cauchy("x", x0, gamma)

    >>> density(X)(z)
    1/(pi*gamma*(1 + (-x0 + z)**2/gamma**2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cauchy_distribution
    .. [2] https://mathworld.wolfram.com/CauchyDistribution.html

    """

    # 返回一个符号随机变量，使用 Cauchy 分布
    return rv(name, CauchyDistribution, (x0, gamma))

#-------------------------------------------------------------------------------
# Chi distribution -------------------------------------------------------------


class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        # 返回 Chi 分布的概率密度函数
        return 2**(1 - self.k/2)*x**(self.k - 1)*exp(-x**2/2)/gamma(self.k/2)

    def _characteristic_function(self, t):
        k = self.k

        # 返回 Chi 分布的特征函数
        part_1 = hyper((k/2,), (S.Half,), -t**2/2)
        part_2 = I*t*sqrt(2)*gamma((k+1)/2)/gamma(k/2)
        part_3 = hyper(((k+1)/2,), (Rational(3, 2),), -t**2/2)
        return part_1 + part_2*part_3

    def _moment_generating_function(self, t):
        k = self.k

        # 返回 Chi 分布的矩生成函数
        part_1 = hyper((k / 2,), (S.Half,), t ** 2 / 2)
        part_2 = t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (S(3) / 2,), t ** 2 / 2)
        return part_1 + part_2 * part_3

def Chi(name, k):
    r"""
    Create a continuous random variable with a Chi distribution.

    The density of the Chi distribution is given by

    .. math::
        f(x) := \frac{2^{1-k/2}x^{k-1}e^{-x^2/2}}{\Gamma(k/2)}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer, The number of degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Chi, density, E
    >>> from sympy import Symbol, simplify

    >>> k = Symbol("k", integer=True)
    >>> z = Symbol("z")

    >>> X = Chi("x", k)

    >>> density(X)(z)
    2**(1 - k/2)*z**(k - 1)*exp(-z**2/2)/gamma(k/2)

    >>> simplify(E(X))
    sqrt(2)*gamma(k/2 + 1/2)/gamma(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_distribution
    .. [2] https://mathworld.wolfram.com/ChiDistribution.html

    """
    # 调用函数 `rv`，传递参数 `name`, `ChiDistribution`, `(k,)`，并返回结果
    return rv(name, ChiDistribution, (k,))
#-------------------------------------------------------------------------------
# Non-central Chi distribution -------------------------------------------------

class ChiNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'l')  # 参数名称包括自由度 k 和偏移参数 l

    @staticmethod
    def check(k, l):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")  # 检查自由度 k 是否为正数
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")  # 检查自由度 k 是否为整数
        _value_check(l > 0, "Shift parameter Lambda must be positive.")  # 检查偏移参数 l 是否为正数

    set = Interval(0, oo)  # 定义变量范围为大于等于 0 的实数集

    def pdf(self, x):
        k, l = self.k, self.l
        return exp(-(x**2+l**2)/2)*x**k*l / (l*x)**(k/2) * besseli(k/2-1, l*x)
        # 返回非中心 Chi 分布的概率密度函数表达式

def ChiNoncentral(name, k, l):
    r"""
    Create a continuous random variable with a non-central Chi distribution.

    Explanation
    ===========

    The density of the non-central Chi distribution is given by

    .. math::
        f(x) := \frac{e^{-(x^2+\lambda^2)/2} x^k\lambda}
                {(\lambda x)^{k/2}} I_{k/2-1}(\lambda x)

    with `x \geq 0`. Here, `I_\nu (x)` is the
    :ref:`modified Bessel function of the first kind <besseli>`.

    Parameters
    ==========

    k : A positive Integer, $k > 0$
        The number of degrees of freedom.
    lambda : Real number, `\lambda > 0`
        Shift parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiNoncentral, density
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True)
    >>> l = Symbol("l")
    >>> z = Symbol("z")

    >>> X = ChiNoncentral("x", k, l)

    >>> density(X)(z)
    l*z**k*exp(-l**2/2 - z**2/2)*besseli(k/2 - 1, l*z)/(l*z)**(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_chi_distribution
    """

    return rv(name, ChiNoncentralDistribution, (k, l))

#-------------------------------------------------------------------------------
# Chi squared distribution -----------------------------------------------------

class ChiSquaredDistribution(SingleContinuousDistribution):
    _argnames = ('k',)  # 参数名称为自由度 k

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")  # 检查自由度 k 是否为正数
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")  # 检查自由度 k 是否为整数

    set = Interval(0, oo)  # 定义变量范围为大于等于 0 的实数集

    def pdf(self, x):
        k = self.k
        return 1/(2**(k/2)*gamma(k/2))*x**(k/2 - 1)*exp(-x/2)
        # 返回卡方分布的概率密度函数表达式

    def _cdf(self, x):
        k = self.k
        return Piecewise(
                (S.One/gamma(k/2)*lowergamma(k/2, x/2), x >= 0),
                (0, True)
        )
        # 返回卡方分布的累积分布函数表达式

    def _characteristic_function(self, t):
        return (1 - 2*I*t)**(-self.k/2)
        # 返回卡方分布的特征函数表达式

    def  _moment_generating_function(self, t):
        return (1 - 2*t)**(-self.k/2)
        # 返回卡方分布的矩生成函数表达式

def ChiSquared(name, k):
    r"""
    Create a continuous random variable with a Chi-squared distribution.

    Explanation
    ===========
    The density of the Chi-squared distribution is given by

    .. math::
        f(x) := \frac{1}{2^{\frac{k}{2}}\Gamma\left(\frac{k}{2}\right)}
                x^{\frac{k}{2}-1} e^{-\frac{x}{2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer
        The number of degrees of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiSquared, density, E, variance, moment
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True, positive=True)
    >>> z = Symbol("z")

    >>> X = ChiSquared("x", k)

    >>> density(X)(z)
    z**(k/2 - 1)*exp(-z/2)/(2**(k/2)*gamma(k/2))

    >>> E(X)
    k

    >>> variance(X)
    2*k

    >>> moment(X, 3)
    k**3 + 6*k**2 + 8*k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_squared_distribution
    .. [2] https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    """

    # 返回一个随机变量对象，使用 Chi-squared 分布，根据给定的名称和自由度参数 k
    return rv(name, ChiSquaredDistribution, (k, ))
#-------------------------------------------------------------------------------
# Dagum distribution -----------------------------------------------------------

class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')  # 参数名称元组包含分布的参数名

    set = Interval(0, oo)  # 设置分布的定义域为 (0, ∞)

    @staticmethod
    def check(p, a, b):
        _value_check(p > 0, "Shape parameter p must be positive.")  # 检查参数 p 是否为正数
        _value_check(a > 0, "Shape parameter a must be positive.")  # 检查参数 a 是否为正数
        _value_check(b > 0, "Scale parameter b must be positive.")  # 检查参数 b 是否为正数

    def pdf(self, x):
        p, a, b = self.p, self.a, self.b
        return a*p/x*((x/b)**(a*p)/(((x/b)**a + 1)**(p + 1)))  # 返回概率密度函数值

    def _cdf(self, x):
        p, a, b = self.p, self.a, self.b
        return Piecewise(((S.One + (S(x)/b)**-a)**-p, x>=0),  # 返回累积分布函数的分段定义
                    (S.Zero, True))

def Dagum(name, p, a, b):
    r"""
    Create a continuous random variable with a Dagum distribution.

    Explanation
    ===========

    The density of the Dagum distribution is given by

    .. math::
        f(x) := \frac{a p}{x} \left( \frac{\left(\tfrac{x}{b}\right)^{a p}}
                {\left(\left(\tfrac{x}{b}\right)^a + 1 \right)^{p+1}} \right)

    with :math:`x > 0`.

    Parameters
    ==========

    p : Real number
        `p > 0`, a shape.
    a : Real number
        `a > 0`, a shape.
    b : Real number
        `b > 0`, a scale.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Dagum, density, cdf
    >>> from sympy import Symbol

    >>> p = Symbol("p", positive=True)
    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Dagum("x", p, a, b)

    >>> density(X)(z)
    a*p*(z/b)**(a*p)*((z/b)**a + 1)**(-p - 1)/z

    >>> cdf(X)(z)
    Piecewise(((1 + (z/b)**(-a))**(-p), z >= 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dagum_distribution

    """

    return rv(name, DagumDistribution, (p, a, b))

#-------------------------------------------------------------------------------
# Davis distribution -----------------------------------------------------------

class DavisDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'n', 'mu')  # 参数名称元组包含分布的参数名

    set = Interval(0, oo)  # 设置分布的定义域为 (0, ∞)

    @staticmethod
    def check(b, n, mu):
        _value_check(b > 0, "Scale parameter b must be positive.")  # 检查参数 b 是否为正数
        _value_check(n > 1, "Shape parameter n must be above 1.")  # 检查参数 n 是否大于 1
        _value_check(mu > 0, "Location parameter mu must be positive.")  # 检查参数 mu 是否为正数

    def pdf(self, x):
        b, n, mu = self.b, self.n, self.mu
        dividend = b**n*(x - mu)**(-1-n)  # 计算概率密度函数的分子
        divisor = (exp(b/(x-mu))-1)*(gamma(n)*zeta(n))  # 计算概率密度函数的分母
        return dividend/divisor  # 返回概率密度函数值


def Davis(name, b, n, mu):
    r""" Create a continuous random variable with Davis distribution.

    Explanation
    ===========

    The density of Davis distribution is given by
    # 返回一个随机变量对象，表示具有 Davis 分布的随机变量
    return rv(name, DavisDistribution, (b, n, mu))
#-------------------------------------------------------------------------------
# Erlang distribution ----------------------------------------------------------

# 定义一个 Erlang 分布的随机变量
def Erlang(name, k, l):
    r"""
    Create a continuous random variable with an Erlang distribution.

    Explanation
    ===========

    The density of the Erlang distribution is given by

    .. math::
        f(x) := \frac{\lambda^k x^{k-1} e^{-\lambda x}}{(k-1)!}

    with :math:`x \in [0,\infty]`.

    Parameters
    ==========

    k : Positive integer
    l : Real number, `\lambda > 0`, the rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Erlang, density, cdf, E, variance
    >>> from sympy import Symbol, simplify, pprint

    >>> k = Symbol("k", integer=True, positive=True)
    >>> l = Symbol("l", positive=True)
    >>> z = Symbol("z")

    >>> X = Erlang("x", k, l)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     k  k - 1  -l*z
    l *z     *e
    ---------------
        Gamma(k)

    >>> C = cdf(X)(z)
    >>> pprint(C, use_unicode=False)
    /lowergamma(k, l*z)
    |------------------  for z > 0
    <     Gamma(k)
    |
    \        0           otherwise


    >>> E(X)
    k/l

    >>> simplify(variance(X))
    k/l**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Erlang_distribution
    .. [2] https://mathworld.wolfram.com/ErlangDistribution.html

    """

    # 使用 RandomSymbol 类创建一个 Erlang 分布的随机变量，参数为 GammaDistribution 和 (k, S.One/l)
    return rv(name, GammaDistribution, (k, S.One/l))

# -------------------------------------------------------------------------------
# ExGaussian distribution -----------------------------------------------------

# 定义一个 ExGaussian 分布的类，继承自 SingleContinuousDistribution
class ExGaussianDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std', 'rate')

    # 设定取值范围为全体实数
    set = Interval(-oo, oo)

    # 静态方法，检查 ExGaussian 分布的参数是否符合要求
    @staticmethod
    def check(mean, std, rate):
        _value_check(
            std > 0, "Standard deviation of ExGaussian must be positive.")
        _value_check(rate > 0, "Rate of ExGaussian must be positive.")

    # 概率密度函数 pdf 的实现
    def pdf(self, x):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = rate/2
        term2 = exp(rate * (2 * mean + rate * std**2 - 2*x)/2)
        term3 = erfc((mean + rate*std**2 - x)/(sqrt(2)*std))
        return term1*term2*term3

    # 累积分布函数 _cdf 的实现
    def _cdf(self, x):
        from sympy.stats import cdf
        mean, std, rate = self.mean, self.std, self.rate
        u = rate*(x - mean)
        v = rate*std
        GaussianCDF1 = cdf(Normal('x', 0, v))(u)
        GaussianCDF2 = cdf(Normal('x', v**2, v))(u)

        return GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))

    # 特征函数 _characteristic_function 的实现
    def _characteristic_function(self, t):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = (1 - I*t/rate)**(-1)
        term2 = exp(I*mean*t - std**2*t**2/2)
        return term1 * term2
    # 定义一个方法用于计算矩生成函数，参数是 t
    def _moment_generating_function(self, t):
        # 获取对象的均值、标准差和速率
        mean, std, rate = self.mean, self.std, self.rate
        # 计算生成函数的第一个术语，表示为 (1 - t/rate)^(-1)
        term1 = (1 - t/rate)**(-1)
        # 计算生成函数的第二个术语，表示为 exp(mean*t + std**2*t**2/2)
        term2 = exp(mean*t + std**2*t**2/2)
        # 返回两个术语的乘积作为矩生成函数的结果
        return term1 * term2
#-------------------------------------------------------------------------------
# Exponential distribution -----------------------------------------------------

class ExponentialDistribution(SingleContinuousDistribution):
    # 定义参数列表
    _argnames = ('rate',)

    # 定义取值范围
    set  = Interval(0, oo)

    @staticmethod
    # 静态方法，检查速率是否为正数
    def check(rate):
        _value_check(rate > 0, "Rate must be positive.")

    # 概率密度函数定义
    def pdf(self, x):
        return self.rate * exp(-self.rate*x)

    # 累积分布函数定义
    def _cdf(self, x):
        return Piecewise(
                (S.One - exp(-self.rate*x), x >= 0),
                (0, True),
        )


这段代码定义了指数分布（Exponential distribution）的概率密度函数（pdf）和累积分布函数（_cdf），并且包含了静态方法用于检查速率（rate）是否为正数，确保分布定义的合法性。
    # 计算特征函数，输入参数 t，返回结果为函数的值
    def _characteristic_function(self, t):
        rate = self.rate  # 从对象中获取速率参数
        # 返回速率除以（速率减去虚数单位乘以 t）的结果
        return rate / (rate - I*t)
    
    # 计算动量生成函数，输入参数 t，返回结果为函数的值
    def _moment_generating_function(self, t):
        rate = self.rate  # 从对象中获取速率参数
        # 返回速率除以（速率减去 t）的结果
        return rate / (rate - t)
    
    # 计算分位数，输入参数 p，返回结果为分位数的值
    def _quantile(self, p):
        # 返回负对数（1-p）除以对象中的速率参数
        return -log(1-p)/self.rate
def Exponential(name, rate):
    r"""
    Create a continuous random variable with an Exponential distribution.

    Explanation
    ===========

    The density of the exponential distribution is given by

    .. math::
        f(x) := \lambda \exp(-\lambda x)

    with $x > 0$. Note that the expected value is `1/\lambda`.

    Parameters
    ==========

    rate : A positive Real number, `\lambda > 0`, the rate (or inverse scale/inverse mean)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Exponential, density, cdf, E
    >>> from sympy.stats import variance, std, skewness, quantile
    >>> from sympy import Symbol

    >>> l = Symbol("lambda", positive=True)
    >>> z = Symbol("z")
    >>> p = Symbol("p")
    >>> X = Exponential("x", l)

    >>> density(X)(z)
    lambda*exp(-lambda*z)

    >>> cdf(X)(z)
    Piecewise((1 - exp(-lambda*z), z >= 0), (0, True))

    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> E(X)
    1/lambda

    >>> variance(X)
    lambda**(-2)

    >>> skewness(X)
    2

    >>> X = Exponential('x', 10)

    >>> density(X)(z)
    10*exp(-10*z)

    >>> E(X)
    1/10

    >>> std(X)
    1/10

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution
    .. [2] https://mathworld.wolfram.com/ExponentialDistribution.html

    """

    # 使用随机变量工厂函数rv创建指定参数的指数分布的随机变量
    return rv(name, ExponentialDistribution, (rate, ))


# -------------------------------------------------------------------------------
# Exponential Power distribution -----------------------------------------------------

class ExponentialPowerDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'alpha', 'beta')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, alpha, beta):
        _value_check(alpha > 0, "Scale parameter alpha must be positive.")
        _value_check(beta > 0, "Shape parameter beta must be positive.")

    def pdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        # 计算指数幂分布的概率密度函数
        num = beta*exp(-(Abs(x - mu)/alpha)**beta)
        den = 2*alpha*gamma(1/beta)
        return num/den

    def _cdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        # 计算指数幂分布的累积分布函数
        num = lowergamma(1/beta, (Abs(x - mu) / alpha)**beta)
        den = 2*gamma(1/beta)
        return sign(x - mu)*num/den + S.Half


def ExponentialPower(name, mu, alpha, beta):
    r"""
    Create a Continuous Random Variable with Exponential Power distribution.
    This distribution is known also as Generalized Normal
    distribution version 1.

    Explanation
    ===========

    The density of the Exponential Power distribution is given by

    .. math::
        f(x) := \frac{\beta}{2\alpha\Gamma(\frac{1}{\beta})}
            e^{{-(\frac{|x - \mu|}{\alpha})^{\beta}}}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number
        A location.
    alpha : Real number,`\alpha > 0`
        A  scale.

    """
    beta : Real number, `\beta > 0`
        A shape.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ExponentialPower, density, cdf
    >>> from sympy import Symbol, pprint
    >>> z = Symbol("z")  # 定义符号变量 z
    >>> mu = Symbol("mu")  # 定义符号变量 mu
    >>> alpha = Symbol("alpha", positive=True)  # 定义符号变量 alpha，并指定其为正数
    >>> beta = Symbol("beta", positive=True)  # 定义符号变量 beta，并指定其为正数
    >>> X = ExponentialPower("x", mu, alpha, beta)  # 创建指数幂分布的随机变量 X
    >>> pprint(density(X)(z), use_unicode=False)  # 打印 X 的密度函数在 z 处的值
                     beta
           /|mu - z|\
          -|--------|
           \ alpha  /
    beta*e
    ---------------------
                  / 1  \
     2*alpha*Gamma|----|
                  \beta/
    >>> cdf(X)(z)  # 计算 X 的累积分布函数在 z 处的值
    1/2 + lowergamma(1/beta, (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/(2*gamma(1/beta))

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/ExponentialPowerDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    """
    return rv(name, ExponentialPowerDistribution, (mu, alpha, beta))


注释：
- `beta : Real number, `\beta > 0``: beta 是一个实数，且必须为正数，用于描述分布的形状参数。
- `Returns ======= RandomSymbol`: 函数返回一个 RandomSymbol 对象。
- `Examples ========`: 下面是一些函数使用的例子。
- `z = Symbol("z")`: 定义符号变量 z。
- `mu = Symbol("mu")`: 定义符号变量 mu。
- `alpha = Symbol("alpha", positive=True)`: 定义符号变量 alpha，并指定其为正数。
- `beta = Symbol("beta", positive=True)`: 定义符号变量 beta，并指定其为正数。
- `X = ExponentialPower("x", mu, alpha, beta)`: 创建一个名为 x 的指数幂分布随机变量 X。
- `pprint(density(X)(z), use_unicode=False)`: 打印 X 的密度函数在 z 处的值。
- `cdf(X)(z)`: 计算 X 的累积分布函数在 z 处的值。
#-------------------------------------------------------------------------------
# F distribution ---------------------------------------------------------------

# 定义一个用于表示 F 分布的类，继承自 SingleContinuousDistribution
class FDistributionDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')  # 定义参数名称

    set = Interval(0, oo)  # 设置定义域为 (0, ∞)

    @staticmethod
    def check(d1, d2):
        # 静态方法：检查参数 d1 和 d2 是否符合要求
        _value_check((d1 > 0, d1.is_integer),
            "Degrees of freedom d1 must be positive integer.")
        _value_check((d2 > 0, d2.is_integer),
            "Degrees of freedom d2 must be positive integer.")

    def pdf(self, x):
        # 概率密度函数定义，接收变量 x
        d1, d2 = self.d1, self.d2
        return (sqrt((d1*x)**d1*d2**d2 / (d1*x+d2)**(d1+d2))
               / (x * beta_fn(d1/2, d2/2)))  # 返回 F 分布的概率密度值

    def _moment_generating_function(self, t):
        # 未实现的方法，因为 F 分布没有矩生成函数
        raise NotImplementedError('The moment generating function for the '
                                  'F-distribution does not exist.')

# 创建 F 分布的随机变量
def FDistribution(name, d1, d2):
    r"""
    Create a continuous random variable with a F distribution.

    Explanation
    ===========

    The density of the F distribution is given by

    .. math::
        f(x) := \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}
                {(d_1 x + d_2)^{d_1 + d_2}}}}
                {x \mathrm{B} \left(\frac{d_1}{2}, \frac{d_2}{2}\right)}

    with :math:`x > 0`.

    Parameters
    ==========

    d1 : `d_1 > 0`, where `d_1` is the degrees of freedom (`n_1 - 1`)
    d2 : `d_2 > 0`, where `d_2` is the degrees of freedom (`n_2 - 1`)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FDistribution, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FDistribution("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      d2
      --    ______________________________
      2    /       d1            -d1 - d2
    d2  *\/  (d1*z)  *(d1*z + d2)
    --------------------------------------
                    /d1  d2\
                 z*B|--, --|
                    \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://mathworld.wolfram.com/F-Distribution.html

    """

    return rv(name, FDistributionDistribution, (d1, d2))

#-------------------------------------------------------------------------------
# Fisher Z distribution --------------------------------------------------------

# 定义一个用于表示 Fisher Z 分布的类，继承自 SingleContinuousDistribution
class FisherZDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')  # 定义参数名称

    set = Interval(-oo, oo)  # 设置定义域为 (-∞, ∞)

    @staticmethod
    def check(d1, d2):
        # 静态方法：检查参数 d1 和 d2 是否符合要求
        _value_check(d1 > 0, "Degree of freedom d1 must be positive.")
        _value_check(d2 > 0, "Degree of freedom d2 must be positive.")

    def pdf(self, x):
        # 概率密度函数定义，接收变量 x
        d1, d2 = self.d1, self.d2
        return (2*d1**(d1/2)*d2**(d2/2) / beta_fn(d1/2, d2/2) *
               exp(d1*x) / (d1*exp(2*x)+d2)**((d1+d2)/2))  # 返回 Fisher Z 分布的概率密度值

# 创建 Fisher Z 分布的随机变量
def FisherZ(name, d1, d2):
    """
    Create a Continuous Random Variable with a Fisher's Z distribution.

    Explanation
    ===========
    
    The density of the Fisher's Z distribution is given by
    
    .. math::
        f(x) := \frac{2d_1^{d_1/2} d_2^{d_2/2}} {\mathrm{B}(d_1/2, d_2/2)}
                \frac{e^{d_1z}}{\left(d_1e^{2z}+d_2\right)^{\left(d_1+d_2\right)/2}}

    .. TODO - What is the difference between these degrees of freedom?

    Parameters
    ==========

    d1 : `d_1 > 0`
        Degree of freedom.
    d2 : `d_2 > 0`
        Degree of freedom.

    Returns
    =======

    RandomSymbol
        A random symbol representing the Fisher's Z distribution.

    Examples
    ========

    >>> from sympy.stats import FisherZ, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FisherZ("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                d1   d2
        d1   d2               - -- - --
        --   --                 2    2
        2    2  /    2*z     \           d1*z
    2*d1  *d2  *\d1*e    + d2/         *e
    -----------------------------------------
                     /d1  d2\
                    B|--, --|
                     \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_z-distribution
    .. [2] https://mathworld.wolfram.com/Fishersz-Distribution.html

    """

    # Create a random variable (RV) with the given name and Fisher's Z distribution
    return rv(name, FisherZDistribution, (d1, d2))
#-------------------------------------------------------------------------------
# Frechet distribution ---------------------------------------------------------

class FrechetDistribution(SingleContinuousDistribution):
    _argnames = ('a', 's', 'm')  # 定义参数名元组，包括形状参数a、尺度参数s和最小值参数m

    set = Interval(0, oo)  # 定义定义域为(0, ∞)的区间对象

    @staticmethod
    def check(a, s, m):
        _value_check(a > 0, "Shape parameter alpha must be positive.")  # 检查形状参数a必须为正数
        _value_check(s > 0, "Scale parameter s must be positive.")  # 检查尺度参数s必须为正数

    def __new__(cls, a, s=1, m=0):
        a, s, m = list(map(sympify, (a, s, m)))  # 将输入参数a, s, m转换为符号对象
        return Basic.__new__(cls, a, s, m)

    def pdf(self, x):
        a, s, m = self.a, self.s, self.m
        return a/s * ((x-m)/s)**(-1-a) * exp(-((x-m)/s)**(-a))
        # 计算概率密度函数：a/s * ((x-m)/s)^(-1-a) * exp(-((x-m)/s)^(-a))

    def _cdf(self, x):
        a, s, m = self.a, self.s, self.m
        return Piecewise((exp(-((x-m)/s)**(-a)), x >= m),
                        (S.Zero, True))
        # 计算累积分布函数：Piecewise((exp(-((x-m)/s)**(-a)), x >= m), (S.Zero, True))

def Frechet(name, a, s=1, m=0):
    r"""
    Create a continuous random variable with a Frechet distribution.

    Explanation
    ===========

    The density of the Frechet distribution is given by

    .. math::
        f(x) := \frac{\alpha}{s} \left(\frac{x-m}{s}\right)^{-1-\alpha}
                 e^{-(\frac{x-m}{s})^{-\alpha}}

    with :math:`x \geq m`.

    Parameters
    ==========

    a : Real number, :math:`a \in \left(0, \infty\right)` the shape
    s : Real number, :math:`s \in \left(0, \infty\right)` the scale
    m : Real number, :math:`m \in \left(-\infty, \infty\right)` the minimum

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Frechet, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", positive=True)
    >>> s = Symbol("s", positive=True)
    >>> m = Symbol("m", real=True)
    >>> z = Symbol("z")

    >>> X = Frechet("x", a, s, m)

    >>> density(X)(z)
    a*((-m + z)/s)**(-a - 1)*exp(-1/((-m + z)/s)**a)/s

    >>> cdf(X)(z)
    Piecewise((exp(-1/((-m + z)/s)**a), m <= z), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution

    """

    return rv(name, FrechetDistribution, (a, s, m))

#-------------------------------------------------------------------------------
# Gamma distribution -----------------------------------------------------------

class GammaDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'theta')  # 定义参数名元组，包括形状参数k和尺度参数theta

    set = Interval(0, oo)  # 定义定义域为(0, ∞)的区间对象

    @staticmethod
    def check(k, theta):
        _value_check(k > 0, "k must be positive")  # 检查形状参数k必须为正数
        _value_check(theta > 0, "Theta must be positive")  # 检查尺度参数theta必须为正数

    def pdf(self, x):
        k, theta = self.k, self.theta
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)
        # 计算概率密度函数：x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def _cdf(self, x):
        k, theta = self.k, self.theta
        return Piecewise(
                    (lowergamma(k, S(x)/theta)/gamma(k), x > 0),
                    (S.Zero, True))
        # 计算累积分布函数：Piecewise((lowergamma(k, S(x)/theta)/gamma(k), x > 0), (S.Zero, True))

    def _characteristic_function(self, t):
        return (1 - self.theta*I*t)**(-self.k)
        # 计算特征函数：(1 - self.theta*I*t)**(-self.k)
    # 定义一个私有方法 _moment_generating_function，计算负二项分布的矩生成函数在给定 t 值处的值
    def _moment_generating_function(self, t):
        # 返回负二项分布的矩生成函数表达式的值
        return (1 - self.theta * t) ** (-self.k)
# 定义 Gamma 分布的随机变量
def Gamma(name, k, theta):
    r"""
    Create a continuous random variable with a Gamma distribution.

    Explanation
    ===========

    The density of the Gamma distribution is given by

    .. math::
        f(x) := \frac{1}{\Gamma(k) \theta^k} x^{k - 1} e^{-\frac{x}{\theta}}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    k : Real number, `k > 0`, a shape
    theta : Real number, `\theta > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gamma, density, cdf, E, variance
    >>> from sympy import Symbol, pprint, simplify

    >>> k = Symbol("k", positive=True)
    >>> theta = Symbol("theta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gamma("x", k, theta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                      -z
                    -----
         -k  k - 1  theta
    theta  *z     *e
    ---------------------
           Gamma(k)

    >>> C = cdf(X, meijerg=True)(z)
    >>> pprint(C, use_unicode=False)
    /            /     z  \
    |k*lowergamma|k, -----|
    |            \   theta/
    <----------------------  for z >= 0
    |     Gamma(k + 1)
    |
    \          0             otherwise

    >>> E(X)
    k*theta

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
           2
    k*theta


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution
    .. [2] https://mathworld.wolfram.com/GammaDistribution.html

    """

    # 返回一个随机变量，其分布为 Gamma 分布
    return rv(name, GammaDistribution, (k, theta))

#-------------------------------------------------------------------------------
# Inverse Gamma distribution ---------------------------------------------------

# 定义 Gamma 逆分布的类
class GammaInverseDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "alpha must be positive")
        _value_check(b > 0, "beta must be positive")

    # 定义 Gamma 逆分布的概率密度函数
    def pdf(self, x):
        a, b = self.a, self.b
        return b**a/gamma(a) * x**(-a-1) * exp(-b/x)

    # 定义 Gamma 逆分布的累积分布函数
    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise((uppergamma(a,b/x)/gamma(a), x > 0),
                        (S.Zero, True))

    # 定义 Gamma 逆分布的特征函数
    def _characteristic_function(self, t):
        a, b = self.a, self.b
        return 2 * (-I*b*t)**(a/2) * besselk(a, sqrt(-4*I*b*t)) / gamma(a)

    # 定义 Gamma 逆分布的矩生成函数
    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'gamma inverse distribution does not exist.')

# 创建 Gamma 逆分布的随机变量
def GammaInverse(name, a, b):
    r"""
    Create a continuous random variable with an inverse Gamma distribution.

    Explanation
    ===========

    The density of the inverse Gamma distribution is given by

    .. math::
        f(x) := \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha - 1}
                \exp\left(\frac{-\beta}{x}\right)

    with :math:`x > 0`.
    Parameters
    ==========

    a : Real number, `a > 0`, a shape parameter of the GammaInverseDistribution
    b : Real number, `b > 0`, a scale parameter of the GammaInverseDistribution

    Returns
    =======

    rv(name, GammaInverseDistribution, (a, b)): RandomVariable
        Returns a random variable (rv) named `name` following the Gamma-inverse distribution
        with parameters `a` (shape) and `b` (scale).

    Examples
    ========

    >>> from sympy.stats import GammaInverse, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = GammaInverse("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                -b
                ---
     a  -a - 1   z
    b *z      *e
    ---------------
       Gamma(a)

    >>> cdf(X)(z)
    Piecewise((uppergamma(a, b/z)/gamma(a), z > 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    """
    
    # 返回一个符号随机变量，其服从参数为 (a, b) 的 GammaInverseDistribution 分布
    return rv(name, GammaInverseDistribution, (a, b))
#-------------------------------------------------------------------------------
# Gumbel distribution (Maximum and Minimum) --------------------------------------------------------

# 定义 GumbelDistribution 类，继承自 SingleContinuousDistribution
class GumbelDistribution(SingleContinuousDistribution):
    _argnames = ('beta', 'mu', 'minimum')

    # 设置定义域为全体实数
    set = Interval(-oo, oo)

    @staticmethod
    def check(beta, mu, minimum):
        # 检查参数 beta 必须为正数
        _value_check(beta > 0, "Scale parameter beta must be positive.")

    # 计算概率密度函数
    def pdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        # 计算最大 Gumbel 分布的概率密度
        f_max = (1/beta)*exp(-z - exp(-z))
        # 计算最小 Gumbel 分布的概率密度
        f_min = (1/beta)*exp(z - exp(z))
        # 返回根据是否为最小分布选择的概率密度
        return Piecewise((f_min, self.minimum), (f_max, not self.minimum))

    # 计算累积分布函数
    def _cdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        # 计算最大 Gumbel 分布的累积分布函数
        F_max = exp(-exp(-z))
        # 计算最小 Gumbel 分布的累积分布函数
        F_min = 1 - exp(-exp(z))
        # 返回根据是否为最小分布选择的累积分布函数
        return Piecewise((F_min, self.minimum), (F_max, not self.minimum))

    # 计算特征函数
    def _characteristic_function(self, t):
        cf_max = gamma(1 - I*self.beta*t) * exp(I*self.mu*t)
        cf_min = gamma(1 + I*self.beta*t) * exp(I*self.mu*t)
        # 返回根据是否为最小分布选择的特征函数
        return Piecewise((cf_min, self.minimum), (cf_max, not self.minimum))

    # 计算矩生成函数
    def _moment_generating_function(self, t):
        mgf_max = gamma(1 - self.beta*t) * exp(self.mu*t)
        mgf_min = gamma(1 + self.beta*t) * exp(self.mu*t)
        # 返回根据是否为最小分布选择的矩生成函数
        return Piecewise((mgf_min, self.minimum), (mgf_max, not self.minimum))

# 创建 Gumbel 分布的随机变量
def Gumbel(name, beta, mu, minimum=False):
    r"""
    Create a Continuous Random Variable with Gumbel distribution.

    Explanation
    ===========

    The density of the Gumbel distribution is given by

    For Maximum

    .. math::
        f(x) := \dfrac{1}{\beta} \exp \left( -\dfrac{x-\mu}{\beta}
                - \exp \left( -\dfrac{x - \mu}{\beta} \right) \right)

    with :math:`x \in [ - \infty, \infty ]`.

    For Minimum

    .. math::
        f(x) := \frac{e^{- e^{\frac{- \mu + x}{\beta}} + \frac{- \mu + x}{\beta}}}{\beta}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number, `\mu`, a location
    beta : Real number, `\beta > 0`, a scale
    minimum : Boolean, by default ``False``, set to ``True`` for enabling minimum distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gumbel, density, cdf
    >>> from sympy import Symbol
    >>> x = Symbol("x")
    >>> mu = Symbol("mu")
    >>> beta = Symbol("beta", positive=True)
    >>> X = Gumbel("x", beta, mu)
    >>> density(X)(x)
    exp(-exp(-(-mu + x)/beta) - (-mu + x)/beta)/beta
    >>> cdf(X)(x)
    exp(-exp(-(-mu + x)/beta))

    References
    ==========

    .. [1] https://mathworld.wolfram.com/GumbelDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Gumbel_distribution
    .. [3] https://web.archive.org/web/20200628222206/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_max.html
    """
    根据给定的参数创建一个概率分布对象，并返回该对象。

    .. [4] https://web.archive.org/web/20200628222212/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_min.html
    参考文档中描述的 Gumbel 最小值分布参数解释

    """
    # 使用给定的参数创建一个随机变量对象，类型为 GumbelDistribution，参数为 (beta, mu, minimum)
    return rv(name, GumbelDistribution, (beta, mu, minimum))
#-------------------------------------------------------------------------------
# Gompertz distribution --------------------------------------------------------

class GompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    # Define the domain of the distribution
    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        # Check that b is positive
        _value_check(b > 0, "b must be positive")
        # Check that eta is positive
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        # Extract parameters b and eta
        eta, b = self.eta, self.b
        # Compute the probability density function (pdf)
        return b * eta * exp(b * x) * exp(eta) * exp(-eta * exp(b * x))

    def _cdf(self, x):
        # Extract parameters b and eta
        eta, b = self.eta, self.b
        # Compute the cumulative distribution function (cdf)
        return 1 - exp(eta) * exp(-eta * exp(b * x))

    def _moment_generating_function(self, t):
        # Extract parameters b and eta
        eta, b = self.eta, self.b
        # Compute the moment generating function
        return eta * exp(eta) * expint(t / b, eta)

def Gompertz(name, b, eta):
    r"""
    Create a Continuous Random Variable with Gompertz distribution.

    Explanation
    ===========

    The density of the Gompertz distribution is given by

    .. math::
        f(x) := b \eta e^{b x} e^{\eta} \exp \left(-\eta e^{bx} \right)

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gompertz("x", b, eta)

    >>> density(X)(z)
    b*eta*exp(eta)*exp(b*z)*exp(-eta*exp(b*z))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gompertz_distribution

    """
    return rv(name, GompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# Kumaraswamy distribution -----------------------------------------------------

class KumaraswamyDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    # Define the domain of the distribution
    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        # Check that a is positive
        _value_check(a > 0, "a must be positive")
        # Check that b is positive
        _value_check(b > 0, "b must be positive")

    def pdf(self, x):
        # Extract parameters a and b
        a, b = self.a, self.b
        # Compute the probability density function (pdf)
        return a * b * x**(a - 1) * (1 - x**a)**(b - 1)

    def _cdf(self, x):
        # Extract parameters a and b
        a, b = self.a, self.b
        # Compute the cumulative distribution function (cdf)
        return Piecewise(
            (S.Zero, x < S.Zero),
            (1 - (1 - x**a)**b, x <= S.One),
            (S.One, True))

def Kumaraswamy(name, a, b):
    r"""
    Create a Continuous Random Variable with a Kumaraswamy distribution.

    Explanation
    ===========

    The density of the Kumaraswamy distribution is given by

    .. math::
        f(x) := a b x^{a-1} (1-x^a)^{b-1}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Kumaraswamy, density, cdf
    >>> from sympy import Symbol, pprint  # 导入符号计算模块中的符号和漂亮打印函数

    >>> a = Symbol("a", positive=True)  # 定义一个名为a的正数符号变量
    >>> b = Symbol("b", positive=True)  # 定义一个名为b的正数符号变量
    >>> z = Symbol("z")  # 定义一个名为z的一般符号变量

    >>> X = Kumaraswamy("x", a, b)  # 创建一个Kumaraswamy分布对象X，带有参数a和b

    >>> D = density(X)(z)  # 计算分布X的密度函数在z处的值，赋给变量D
    >>> pprint(D, use_unicode=False)  # 使用非Unicode字符打印D的值
                       b - 1
         a - 1 /     a\
    a*b*z     *\1 - z /
    
    >>> cdf(X)(z)  # 计算分布X的累积分布函数在z处的值

    Piecewise((0, z < 0), (1 - (1 - z**a)**b, z <= 1), (1, True))
    # 返回一个分段函数：当z小于0时为0，当z在0到1之间时为1 - (1 - z**a)**b，否则为1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kumaraswamy_distribution
    # 参考资料链接到Kumaraswamy分布的维基百科页面

    """

    return rv(name, KumaraswamyDistribution, (a, b))
    # 返回一个具有给定名称、Kumaraswamy分布类型和参数(a, b)的随机变量对象
#-------------------------------------------------------------------------------
# Levy distribution ---------------------------------------------------------

class LevyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'c')  # 定义分布参数名称为 'mu' 和 'c'

    @property
    def set(self):
        return Interval(self.mu, oo)  # 返回定义域为从 self.mu 到正无穷的区间

    @staticmethod
    def check(mu, c):
        _value_check(c > 0, "Scale parameter c must be positive.")  # 检查 c 是否大于 0
        _value_check(mu.is_real, "Location parameter mu should be real")  # 检查 mu 是否为实数

    def pdf(self, x):
        mu, c = self.mu, self.c
        return exp(-c*Abs(x - mu)) / (2*c)  # 返回概率密度函数的值

    def _cdf(self, x):
        mu, c = self.mu, self.c
        return Piecewise(
                    (1 - exp(-c*(x - mu)), x > mu),
                    (0, True)
                        )  # 返回累积分布函数的值

    def _characteristic_function(self, t):
        return exp(I*t*self.mu - self.c*Abs(t))  # 返回特征函数的值

    def _moment_generating_function(self, t):
        return exp(self.mu*t + self.c*t**2 / 2)  # 返回矩生成函数的值
    # 定义检查函数，用于验证参数是否满足条件
    def check(mu, c):
        # 检查参数 c 是否大于 0，如果不是则抛出异常
        _value_check(c > 0, "c (scale parameter) must be positive")
        # 检查参数 mu 是否为实数，如果不是则抛出异常
        _value_check(mu.is_real, "mu (location parameter) must be real")

    # 定义概率密度函数 (PDF)，计算给定参数下的概率密度值
    def pdf(self, x):
        # 获取分布的参数 mu 和 c
        mu, c = self.mu, self.c
        # 计算并返回概率密度函数的值
        return sqrt(c/(2*pi)) * exp(-c/(2*(x - mu))) / ((x - mu)**(S.One + S.Half))

    # 定义累积分布函数 (CDF)，计算给定参数下的累积分布值
    def _cdf(self, x):
        # 获取分布的参数 mu 和 c
        mu, c = self.mu, self.c
        # 计算并返回累积分布函数的值，使用误差函数的补函数计算
        return erfc(sqrt(c/(2*(x - mu))))

    # 定义特征函数，计算给定参数下的特征函数值
    def _characteristic_function(self, t):
        # 获取分布的参数 mu 和 c
        mu, c = self.mu, self.c
        # 计算并返回特征函数的值，使用复数指数函数计算
        return exp(I * mu * t - sqrt(-2 * I * c * t))

    # 定义矩生成函数，对于Levy分布来说，该函数不存在，抛出未实现异常
    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of Levy distribution does not exist.')
#-------------------------------------------------------------------------------
# Levy distribution --------------------------------------------------------------
def Levy(name, mu, c):
    r"""
    Create a continuous random variable with a Levy distribution.

    The density of the Levy distribution is given by

    .. math::
        f(x) := \sqrt(\frac{c}{2 \pi}) \frac{\exp -\frac{c}{2 (x - \mu)}}{(x - \mu)^{3/2}}

    Parameters
    ==========

    mu : Real number
        The location parameter.
    c : Real number, `c > 0`
        A scale parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Levy, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> c = Symbol("c", positive=True)
    >>> z = Symbol("z")

    >>> X = Levy("x", mu, c)

    >>> density(X)(z)
    sqrt(2)*sqrt(c)*exp(-c/(-2*mu + 2*z))/(2*sqrt(pi)*(-mu + z)**(3/2))

    >>> cdf(X)(z)
    erfc(sqrt(c)*sqrt(1/(-2*mu + 2*z)))

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    .. [2] https://mathworld.wolfram.com/LevyDistribution.html
    """

    # Return a random variable using LevyDistribution with parameters (mu, c)
    return rv(name, LevyDistribution, (mu, c))

#-------------------------------------------------------------------------------
# Log-Cauchy distribution --------------------------------------------------------
class LogCauchyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    set = Interval.open(0, oo)

    @staticmethod
    def check(mu, sigma):
        _value_check((sigma > 0) != False, "Scale parameter Gamma must be positive.")
        _value_check(mu.is_real != False, "Location parameter must be real.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        # Probability density function of Log-Cauchy distribution
        return 1/(x*pi)*(sigma/((log(x) - mu)**2 + sigma**2))

    def _cdf(self, x):
        mu, sigma = self.mu, self.sigma
        # Cumulative distribution function of Log-Cauchy distribution
        return (1/pi)*atan((log(x) - mu)/sigma) + S.Half

    def _characteristic_function(self, t):
        # Characteristic function is not defined for Log-Cauchy distribution
        raise NotImplementedError("The characteristic function for the "
                                  "Log-Cauchy distribution does not exist.")

    def _moment_generating_function(self, t):
        # Moment generating function is not defined for Log-Cauchy distribution
        raise NotImplementedError("The moment generating function for the "
                                  "Log-Cauchy distribution does not exist.")

def LogCauchy(name, mu, sigma):
    r"""
    Create a continuous random variable with a Log-Cauchy distribution.
    The density of the Log-Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi x} \frac{\sigma}{(log(x)-\mu^2) + \sigma^2}

    Parameters
    ==========

    mu : Real number, the location

    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogCauchy, density, cdf
    >>> from sympy import Symbol, S

    >>> mu = 2
    >>> sigma = S.One / 5
    >>> z = Symbol("z")

    >>> X = LogCauchy("x", mu, sigma)

    >>> density(X)(z)
    1/(5*pi*z*((log(z) - 2)**2 + 1/25))

    >>> cdf(X)(z)
    atan(5*log(z) - 10)/pi + 1/2

    References
    ==========
    # 创建一个概率分布对象，根据对数柯西分布生成一个随机变量
    return rv(name, LogCauchyDistribution, (mu, sigma))
#-------------------------------------------------------------------------------
# Log-logistic distribution --------------------------------------------------------

# 定义 Log-logistic 分布的类，继承自 SingleContinuousDistribution
class LogLogisticDistribution(SingleContinuousDistribution):
    # 参数名称
    _argnames = ('alpha', 'beta')

    # 分布的定义域为半开区间 (0, oo)
    set = Interval(0, oo)

    # 静态方法：检查参数 alpha 和 beta 是否符合要求
    @staticmethod
    def check(alpha, beta):
        # 检查 alpha 必须大于 0
        _value_check(alpha > 0, "Scale parameter Alpha must be positive.")
        # 检查 beta 必须大于 0
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    # 概率密度函数
    def pdf(self, x):
        # 提取参数 alpha 和 beta
        a, b = self.alpha, self.beta
        # 返回 Log-logistic 分布的概率密度函数表达式
        return ((b/a)*(x/a)**(b - 1))/(1 + (x/a)**b)**2

    # 累积分布函数
    def _cdf(self, x):
        # 提取参数 alpha 和 beta
        a, b = self.alpha, self.beta
        # 返回 Log-logistic 分布的累积分布函数表达式
        return 1/(1 + (x/a)**(-b))

    # 分位数函数
    def _quantile(self, p):
        # 提取参数 alpha 和 beta
        a, b = self.alpha, self.beta
        # 返回 Log-logistic 分布的分位数函数表达式
        return a*((p/(1 - p))**(1/b))

    # 期望值函数
    def expectation(self, expr, var, **kwargs):
        # 提取参数 alpha 和 beta
        a, b = self.args
        # 返回期望值的表达式，考虑了不同参数范围下的分段定义
        return Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))

# 创建 Log-logistic 分布的随机变量
def LogLogistic(name, alpha, beta):
    r"""
    Create a continuous random variable with a log-logistic distribution.

    Explanation
    ===========

    The density of the log-logistic distribution is given by

    .. math::
        f(x) := \frac{(b/a)(x/a)^{b-1}} {1 + (x/a)^b)^2}

    Parameters
    ==========

    alpha : Real number, scale parameter
    beta : Real number, `beta > 0`, shape parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogLogistic, density, cdf
    >>> from sympy import Symbol

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = LogLogistic("x", alpha, beta)

    >>> density(X)(z)
    ((beta/alpha)*(z/alpha)**(beta - 1))/(1 + (z/alpha)**beta)**2

    >>> cdf(X)(z)
    1/(1 + (z/alpha)**(-beta))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-logistic_distribution
    .. [2] https://mathworld.wolfram.com/LogLogisticDistribution.html

    """

    return rv(name, LogLogisticDistribution, (alpha, beta))
    # 创建一个服从对数逻辑分布的连续随机变量。
    # 当 beta > 1 时，该分布是单峰的。

    # 对数逻辑分布的概率密度函数如下：
    #
    # f(x) := ((beta / alpha) * (x / alpha)^(beta - 1)) / (1 + (x / alpha)^beta)^2

    # 参数说明：
    # alpha: 实数，alpha > 0，表示尺度参数和分布的中位数
    # beta: 实数，beta > 0，表示形状参数

    # 返回一个随机变量对象 RandomSymbol

    # 示例：
    # >>> from sympy.stats import LogLogistic, density, cdf, quantile
    # >>> from sympy import Symbol, pprint
    # >>> alpha = Symbol("alpha", positive=True)
    # >>> beta = Symbol("beta", positive=True)
    # >>> p = Symbol("p")
    # >>> z = Symbol("z", positive=True)
    # >>> X = LogLogistic("x", alpha, beta)
    # >>> D = density(X)(z)
    # >>> pprint(D, use_unicode=False)
    # beta - 1
    # /  z  \
    # beta*|-----|
    # \alpha/
    # ------------------------
    # 2
    # /       beta    \
    # |/  z  \        |
    # alpha*||-----|     + 1|
    # \\alpha/        /
    #
    # >>> cdf(X)(z)
    # 1/(1 + (z/alpha)**(-beta))
    #
    # >>> quantile(X)(p)
    # alpha*(p/(1 - p))**(1/beta)

    # 参考文献：
    # [1] https://en.wikipedia.org/wiki/Log-logistic_distribution

    # 返回一个随机变量对象，命名为 name，服从对数逻辑分布，参数为 (alpha, beta)
    return rv(name, LogLogisticDistribution, (alpha, beta))
#-------------------------------------------------------------------------------
#Logit-Normal distribution------------------------------------------------------

class LogitNormalDistribution(SingleContinuousDistribution):
    # Logit-Normal 分布的定义，继承自 SingleContinuousDistribution
    _argnames = ('mu', 's')
    # 参数名称：mu 是均值，s 是尺度参数
    set = Interval.open(0, 1)
    # 分布的定义域为开区间 (0, 1)

    @staticmethod
    def check(mu, s):
        # 静态方法：检查参数 mu 和 s 的合法性
        _value_check((s ** 2).is_real is not False and s ** 2 > 0, "Squared scale parameter s must be positive.")
        _value_check(mu.is_real is not False, "Location parameter must be real")

    def _logit(self, x):
        # 计算 logit 函数的实现，用于转换 x 到 logit(x)
        return log(x / (1 - x))

    def pdf(self, x):
        # 概率密度函数：计算 logit-normal 分布在 x 处的概率密度值
        mu, s = self.mu, self.s
        return exp(-(self._logit(x) - mu)**2/(2*s**2))*(S.One/sqrt(2*pi*(s**2)))*(1/(x*(1 - x)))

    def _cdf(self, x):
        # 累积分布函数的实现：计算 logit-normal 分布在 x 处的累积分布值
        mu, s = self.mu, self.s
        return (S.One/2)*(1 + erf((self._logit(x) - mu)/(sqrt(2*s**2))))


def LogitNormal(name, mu, s):
    r"""
    Create a continuous random variable with a Logit-Normal distribution.

    创建一个具有 logit-normal 分布的连续随机变量。

    The density of the logistic distribution is given by

    logistic 分布的概率密度函数为：

    .. math::
        f(x) := \frac{1}{s \sqrt{2 \pi}} \frac{1}{x(1 - x)} e^{- \frac{(logit(x)  - \mu)^2}{s^2}}
        where logit(x) = \log(\frac{x}{1 - x})
        
    Parameters
    ==========

    mu : Real number, the location (mean)
    mu：实数，位置参数（均值）
    
    s : Real number, `s > 0`, a scale
    s：实数，`s > 0`，尺度参数

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogitNormal, density, cdf
    >>> from sympy import Symbol,pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")
    >>> X = LogitNormal("x",mu,s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                              2
            /         /  z  \\
           -|-mu + log|-----||
            \         \1 - z//
           ---------------------
                       2
      ___           2*s
    \/ 2 *e
    ----------------------------
            ____
        2*\/ pi *s*z*(1 - z)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + log(z/(1 - z)))**2/(2*s**2))/(2*sqrt(pi)*s*z*(1 - z))

    >>> cdf(X)(z)
    erf(sqrt(2)*(-mu + log(z/(1 - z)))/(2*s))/2 + 1/2


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logit-normal_distribution

    """

    return rv(name, LogitNormalDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Log Normal distribution ------------------------------------------------------


class LogNormalDistribution(SingleContinuousDistribution):
    # Log-Normal 分布的定义，继承自 SingleContinuousDistribution
    _argnames = ('mean', 'std')
    # 参数名称：mean 是均值，std 是标准差

    set = Interval(0, oo)
    # 分布的定义域为 (0, ∞)

    @staticmethod
    def check(mean, std):
        # 静态方法：检查参数 mean 和 std 的合法性
        _value_check(std > 0, "Parameter std must be positive.")

    def pdf(self, x):
        # 概率密度函数：计算 log-normal 分布在 x 处的概率密度值
        mean, std = self.mean, self.std
        return exp(-(log(x) - mean)**2 / (2*std**2)) / (x*sqrt(2*pi)*std)
    # 定义累积分布函数 (Cumulative Distribution Function, CDF)，计算随机变量 x 的累积概率
    def _cdf(self, x):
        # 从对象属性中获取均值和标准差
        mean, std = self.mean, self.std
        # 返回分段函数 Piecewise 对象，其中：
        # - 当 x > 0 时，计算正态分布的累积分布函数（CDF），以求得 x 的概率
        # - 当 x <= 0 时，返回零
        return Piecewise(
                (S.Half + S.Half*erf((log(x) - mean)/sqrt(2)/std), x > 0),
                (S.Zero, True)
        )
    
    # 定义矩生成函数 (Moment Generating Function, MGF)，抛出 NotImplementedError 异常
    def _moment_generating_function(self, t):
        raise NotImplementedError('Moment generating function of the log-normal distribution is not defined.')
    
    
    这段代码中，第一个函数 `_cdf` 是计算对数正态分布的累积分布函数，基于给定的均值和标准差。第二个函数 `_moment_generating_function` 则简单地抛出一个异常，因为对数正态分布并没有定义矩生成函数。
def LogNormal(name, mean, std):
    r"""
    Create a continuous random variable with a log-normal distribution.

    Explanation
    ===========

    The density of the log-normal distribution is given by

    .. math::
        f(x) := \frac{1}{x\sqrt{2\pi\sigma^2}}
                e^{-\frac{\left(\ln x-\mu\right)^2}{2\sigma^2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    name : str
        Name of the random variable.
    mean : Real number
        The mean (mu) of the underlying normal distribution (log-scale).
    std : Real number
        The standard deviation (sigma) of the underlying normal distribution.

    Returns
    =======

    RandomSymbol
        A symbolic representation of the log-normal random variable.

    Examples
    ========

    >>> from sympy.stats import LogNormal, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = LogNormal("x", mu, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                          2
           -(-mu + log(z))
           -----------------
                      2
      ___      2*sigma
    \/ 2 *e
    ------------------------
            ____
        2*\/ pi *sigma*z


    >>> X = LogNormal('x', 0, 1) # Mean 0, standard deviation 1

    >>> density(X)(z)
    sqrt(2)*exp(-log(z)**2/2)/(2*sqrt(pi)*z)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lognormal
    .. [2] https://mathworld.wolfram.com/LogNormalDistribution.html

    """

    # 返回一个随机变量，使用 LogNormalDistribution 分布
    return rv(name, LogNormalDistribution, (mean, std))

#-------------------------------------------------------------------------------
# Lomax Distribution -----------------------------------------------------------

class LomaxDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'lamda',)
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, lamda):
        _value_check(alpha.is_real, "Shape parameter should be real.")
        _value_check(lamda.is_real, "Scale parameter should be real.")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(lamda.is_positive, "Scale parameter should be positive.")

    def pdf(self, x):
        lamba, alpha = self.lamda, self.alpha
        # 返回 Lomax 分布的概率密度函数
        return (alpha/lamba) * (S.One + x/lamba)**(-alpha-1)

def Lomax(name, alpha, lamda):
    r"""
    Create a continuous random variable with a Lomax distribution.

    Explanation
    ===========

    The density of the Lomax distribution is given by

    .. math::
        f(x) := \frac{\alpha}{\lambda}\left[1+\frac{x}{\lambda}\right]^{-(\alpha+1)}

    Parameters
    ==========

    name : str
        Name of the random variable.
    alpha : Real Number, `\alpha > 0`
        Shape parameter
    lamda : Real Number, `\lambda > 0`
        Scale parameter

    Examples
    ========

    >>> from sympy.stats import Lomax, density, cdf, E
    >>> from sympy import symbols
    >>> a, l = symbols('a, l', positive=True)
    >>> X = Lomax('X', a, l)
    >>> x = symbols('x')
    >>> density(X)(x)
    a*(1 + x/l)**(-a - 1)/l
    >>> cdf(X)(x)
    Piecewise((1 - 1/(1 + x/l)**a, x >= 0), (0, True))
    >>> a = 2

    """

    # 返回一个随机变量，使用 LomaxDistribution 分布
    return rv(name, LomaxDistribution, (alpha, lamda))
    # 创建一个 Lomax 分布的随机变量对象 X，参数为 a (alpha) 和 l (lambda)
    >>> X = Lomax('X', a, l)
    # 计算随机变量 X 的期望值
    >>> E(X)
    # 返回 Lomax 分布的 lambda 参数作为期望值
    l

    Returns
    =======

    # 返回一个随机符号对象 (RandomSymbol)，可能代表随机变量或表达式中的随机元素

    References
    ==========

    # 引用 Lomax 分布的信息，详细描述在维基百科中可以找到
    .. [1] https://en.wikipedia.org/wiki/Lomax_distribution

    """
    # 使用给定的名称、Lomax 分布类型和参数元组 (alpha, lamda) 创建一个随机变量对象，并返回
    return rv(name, LomaxDistribution, (alpha, lamda))
#-------------------------------------------------------------------------------
# Moyal Distribution -----------------------------------------------------------

class MoyalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    @staticmethod
    def check(mu, sigma):
        # 检查位置参数必须是实数
        _value_check(mu.is_real, "Location parameter must be real.")
        # 检查尺度参数必须是实数且大于零
        _value_check(sigma.is_real and sigma > 0, "Scale parameter must be real\
        and positive.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        # 计算概率密度函数
        num = exp(-(exp(-(x - mu)/sigma) + (x - mu)/(sigma))/2)
        den = (sqrt(2*pi) * sigma)
        return num/den

    def _characteristic_function(self, t):
        mu, sigma = self.mu, self.sigma
        # 计算特征函数
        term1 = exp(I*t*mu)
        term2 = (2**(-I*sigma*t) * gamma(Rational(1, 2) - I*t*sigma))
        return (term1 * term2)/sqrt(pi)

    def _moment_generating_function(self, t):
        mu, sigma = self.mu, self.sigma
        # 计算动量生成函数
        term1 = exp(t*mu)
        term2 = (2**(-1*sigma*t) * gamma(Rational(1, 2) - t*sigma))
        return (term1 * term2)/sqrt(pi)

def Moyal(name, mu, sigma):
    r"""
    Create a continuous random variable with a Moyal distribution.

    Explanation
    ===========

    The density of the Moyal distribution is given by

    .. math::
        f(x) := \frac{\exp\left(-\frac{\exp\left(-\frac{x - \mu}{\sigma}\right) + \frac{x - \mu}{\sigma}}{2}\right)}{\sqrt{2 \pi} \sigma}

    Parameters
    ==========

    mu : Real number, location parameter
    sigma : Real number, scale parameter, `sigma > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Moyal, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Moyal("x", mu, sigma)

    >>> density(X)(z)
    exp(-(exp(-(z - mu)/sigma) + (z - mu)/sigma)/2)/(sqrt(2)*sqrt(pi)*sigma)

    >>> E(X)
    mu + sigma*sqrt(2)*EulerGamma

    >>> simplify(variance(X))
    sigma**2*(2 - pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moyal_distribution
    .. [2] https://mathworld.wolfram.com/MoyalDistribution.html

    """

    return rv(name, MoyalDistribution, (mu, sigma))
    # 返回一个随机变量对象，使用慕亚尔分布进行参数化
    return rv(name, MoyalDistribution, (mu, sigma))
#-------------------------------------------------------------------------------
# Nakagami distribution --------------------------------------------------------

class NakagamiDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'omega')

    # 定义分布参数 mu 和 omega 的取值范围为大于等于0的实数区间
    set = Interval(0, oo)

    @staticmethod
    def check(mu, omega):
        # 检查参数 mu 必须大于等于1/2
        _value_check(mu >= S.Half, "Shape parameter mu must be greater than equal to 1/2.")
        # 检查参数 omega 必须为正数
        _value_check(omega > 0, "Spread parameter omega must be positive.")

    def pdf(self, x):
        # 获取分布的参数 mu 和 omega
        mu, omega = self.mu, self.omega
        # 返回 Nakagami 分布的概率密度函数
        return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)

    def _cdf(self, x):
        # 获取分布的参数 mu 和 omega
        mu, omega = self.mu, self.omega
        # 返回 Nakagami 分布的累积分布函数
        return Piecewise(
                    (lowergamma(mu, (mu/omega)*x**2)/gamma(mu), x > 0),
                    (S.Zero, True))

def Nakagami(name, mu, omega):
    r"""
    Create a continuous random variable with a Nakagami distribution.

    Explanation
    ===========

    The density of the Nakagami distribution is given by

    .. math::
        f(x) := \frac{2\mu^\mu}{\Gamma(\mu)\omega^\mu} x^{2\mu-1}
                \exp\left(-\frac{\mu}{\omega}x^2 \right)

    with :math:`x > 0`.

    Parameters
    ==========

    mu : Real number, `\mu \geq \frac{1}{2}`, a shape
    omega : Real number, `\omega > 0`, the spread

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Nakagami, density, E, variance, cdf
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> omega = Symbol("omega", positive=True)
    >>> z = Symbol("z")

    >>> X = Nakagami("x", mu, omega)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                    2
                               -mu*z
                               -------
        mu      -mu  2*mu - 1  omega
    2*mu  *omega   *z        *e
    ----------------------------------
                Gamma(mu)

    >>> simplify(E(X))
    sqrt(mu)*sqrt(omega)*gamma(mu + 1/2)/gamma(mu + 1)

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
                        2
             omega*Gamma (mu + 1/2)
    omega - -----------------------
            Gamma(mu)*Gamma(mu + 1)

    >>> cdf(X)(z)
    Piecewise((lowergamma(mu, mu*z**2/omega)/gamma(mu), z > 0),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Nakagami_distribution

    """

    # 返回一个随机变量，其服从 Nakagami 分布
    return rv(name, NakagamiDistribution, (mu, omega))

#-------------------------------------------------------------------------------
# Normal distribution ----------------------------------------------------------

class NormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    @staticmethod
    def check(mean, std):
        # 检查参数 std 必须为正数
        _value_check(std > 0, "Standard deviation must be positive")
    # 计算正态分布的概率密度函数（PDF）
    def pdf(self, x):
        return exp(-(x - self.mean)**2 / (2*self.std**2)) / (sqrt(2*pi)*self.std)

    # 计算正态分布的累积分布函数（CDF）
    def _cdf(self, x):
        # 提取对象的均值和标准差
        mean, std = self.mean, self.std
        return erf(sqrt(2)*(-mean + x)/(2*std))/2 + S.Half

    # 计算正态分布的特征函数
    def _characteristic_function(self, t):
        # 提取对象的均值和标准差
        mean, std = self.mean, self.std
        return exp(I*mean*t - std**2*t**2/2)

    # 计算正态分布的矩生成函数
    def _moment_generating_function(self, t):
        # 提取对象的均值和标准差
        mean, std = self.mean, self.std
        return exp(mean*t + std**2*t**2/2)

    # 计算正态分布的分位数函数（逆 CDF）
    def _quantile(self, p):
        # 提取对象的均值和标准差
        mean, std = self.mean, self.std
        return mean + std*sqrt(2)*erfinv(2*p - 1)
# 定义一个函数 Normal，用于创建正态分布的连续随机变量
def Normal(name, mean, std):
    r"""
    Create a continuous random variable with a Normal distribution.

    Explanation
    ===========

    The density of the Normal distribution is given by

    .. math::
        f(x) := \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{(x-\mu)^2}{2\sigma^2} }

    Parameters
    ==========

    mu : Real number or a list representing the mean or the mean vector
    sigma : Real number or a positive definite square matrix,
         :math:`\sigma^2 > 0`, the variance

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Normal, density, E, std, cdf, skewness, quantile, marginal_distribution
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu")
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")
    >>> y = Symbol("y")
    >>> p = Symbol("p")
    >>> X = Normal("x", mu, sigma)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + z)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)

    >>> C = simplify(cdf(X))(z) # it needs a little more help...
    >>> pprint(C, use_unicode=False)
       /  ___          \
       |\/ 2 *(-mu + z)|
    erf|---------------|
       \    2*sigma    /   1
    -------------------- + -
             2             2

    >>> quantile(X)(p)
    mu + sqrt(2)*sigma*erfinv(2*p - 1)

    >>> simplify(skewness(X))
    0

    >>> X = Normal("x", 0, 1) # Mean 0, standard deviation 1
    >>> density(X)(z)
    sqrt(2)*exp(-z**2/2)/(2*sqrt(pi))

    >>> E(2*X + 1)
    1

    >>> simplify(std(2*X + 1))
    2

    >>> m = Normal('X', [1, 2], [[2, 1], [1, 2]])
    >>> pprint(density(m)(y, z), use_unicode=False)
              2          2
             y    y*z   z
           - -- + --- - -- + z - 1
      ___    3     3    3
    \/ 3 *e
    ------------------------------
                 6*pi

    >>> marginal_distribution(m, m[0])(1)
     1/(2*sqrt(pi))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] https://mathworld.wolfram.com/NormalDistributionFunction.html

    """

    # 检查均值和标准差是否是列表或矩阵，如果是则返回多元正态分布对象
    if isinstance(mean, list) or getattr(mean, 'is_Matrix', False) and\
        isinstance(std, list) or getattr(std, 'is_Matrix', False):
        from sympy.stats.joint_rv_types import MultivariateNormal
        return MultivariateNormal(name, mean, std)
    # 否则返回单变量正态分布对象
    return rv(name, NormalDistribution, (mean, std))


#-------------------------------------------------------------------------------
# Inverse Gaussian distribution ----------------------------------------------------------

# 定义一个类 GaussianInverseDistribution，表示逆高斯分布的单变量连续分布
class GaussianInverseDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'shape')

    @property
    def set(self):
        return Interval(0, oo)

    @staticmethod
    # 静态方法 check，用于检查逆高斯分布的参数合法性
    def check(mean, shape):
        _value_check(shape > 0, "Shape parameter must be positive")
        _value_check(mean > 0, "Mean must be positive")
    # 计算概率密度函数 (PDF) 的值，对给定的 x 进行计算
    def pdf(self, x):
        mu, s = self.mean, self.shape
        # 计算正态分布的概率密度函数值
        return exp(-s*(x - mu)**2 / (2*x*mu**2)) * sqrt(s/(2*pi*x**3))

    # 计算累积分布函数 (CDF) 的值，对给定的 x 进行计算
    def _cdf(self, x):
        # 导入符号计算库中的累积分布函数
        from sympy.stats import cdf
        mu, s = self.mean, self.shape
        # 计算标准正态分布的累积分布函数值
        stdNormalcdf = cdf(Normal('x', 0, 1))

        # 第一项计算
        first_term = stdNormalcdf(sqrt(s/x) * ((x/mu) - S.One))
        # 第二项计算
        second_term = exp(2*s/mu) * stdNormalcdf(-sqrt(s/x)*(x/mu + S.One))

        # 返回累积分布函数的值
        return  first_term + second_term

    # 计算特征函数的值，对给定的 t 进行计算
    def _characteristic_function(self, t):
        mu, s = self.mean, self.shape
        # 计算特征函数的表达式
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*I*t)/s)))

    # 计算动差生成函数的值，对给定的 t 进行计算
    def _moment_generating_function(self, t):
        mu, s = self.mean, self.shape
        # 计算动差生成函数的表达式
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*t)/s)))
# 定义一个函数 GaussianInverse，用于创建具有逆高斯分布的连续随机变量
def GaussianInverse(name, mean, shape):
    r"""
    Create a continuous random variable with an Inverse Gaussian distribution.
    Inverse Gaussian distribution is also known as Wald distribution.

    Explanation
    ===========

    The density of the Inverse Gaussian distribution is given by

    .. math::
        f(x) := \sqrt{\frac{\lambda}{2\pi x^3}} e^{-\frac{\lambda(x-\mu)^2}{2x\mu^2}}

    Parameters
    ==========

    mu :
        Positive number representing the mean.
    lambda :
        Positive number representing the shape parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GaussianInverse, density, E, std, skewness
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> lamda = Symbol("lambda", positive=True)
    >>> z = Symbol("z", positive=True)
    >>> X = GaussianInverse("x", mu, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                       2
                      -lambda*(-mu + z)
                      -------------------
                                2
      ___   ________        2*mu *z
    \/ 2 *\/ lambda *e
    -------------------------------------
                    ____  3/2
                2*\/ pi *z

    >>> E(X)
    mu

    >>> std(X).expand()
    mu**(3/2)/sqrt(lambda)

    >>> skewness(X).expand()
    3*sqrt(mu)/sqrt(lambda)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    .. [2] https://mathworld.wolfram.com/InverseGaussianDistribution.html

    """

    # 调用 rv 函数，创建具有逆高斯分布的随机变量，并返回该随机变量
    return rv(name, GaussianInverseDistribution, (mean, shape))

# 将 GaussianInverse 赋值给 Wald，使其成为逆高斯分布的别名
Wald = GaussianInverse

#-------------------------------------------------------------------------------
# Pareto distribution ----------------------------------------------------------

# 定义 ParetoDistribution 类，表示帕累托分布的单变量连续分布
class ParetoDistribution(SingleContinuousDistribution):
    _argnames = ('xm', 'alpha')

    @property
    def set(self):
        return Interval(self.xm, oo)

    @staticmethod
    def check(xm, alpha):
        _value_check(xm > 0, "Xm must be positive")
        _value_check(alpha > 0, "Alpha must be positive")

    # 定义帕累托分布的概率密度函数
    def pdf(self, x):
        xm, alpha = self.xm, self.alpha
        return alpha * xm**alpha / x**(alpha + 1)

    # 定义帕累托分布的累积分布函数
    def _cdf(self, x):
        xm, alpha = self.xm, self.alpha
        return Piecewise(
                (S.One - xm**alpha/x**alpha, x>=xm),
                (0, True),
        )

    # 定义帕累托分布的矩母函数
    def _moment_generating_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-xm*t)**alpha * uppergamma(-alpha, -xm*t)

    # 定义帕累托分布的特征函数
    def _characteristic_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-I * xm * t) ** alpha * uppergamma(-alpha, -I * xm * t)


# 定义 Pareto 函数，用于创建具有帕累托分布的连续随机变量
def Pareto(name, xm, alpha):
    r"""
    Create a continuous random variable with the Pareto distribution.

    Explanation
    ===========

    The density of the Pareto distribution is given by
    # 返回一个符号变量，表示服从帕累托分布的随机变量
    return rv(name, ParetoDistribution, (xm, alpha))
#-------------------------------------------------------------------------------
# PowerFunction distribution ---------------------------------------------------

class PowerFunctionDistribution(SingleContinuousDistribution):
    _argnames=('alpha','a','b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(alpha, a, b):
        _value_check(a.is_real, "Continuous Boundary parameter should be real.")
        _value_check(b.is_real, "Continuous Boundary parameter should be real.")
        _value_check(a < b, " 'a' the left Boundary must be smaller than 'b' the right Boundary." )
        _value_check(alpha.is_positive, "Continuous Shape parameter should be positive.")

    def pdf(self, x):
        alpha, a, b = self.alpha, self.a, self.b
        num = alpha*(x - a)**(alpha - 1)
        den = (b - a)**alpha
        return num/den

def PowerFunction(name, alpha, a, b):
    r"""
    Creates a continuous random variable with a Power Function Distribution.

    Explanation
    ===========

    The density of PowerFunction distribution is given by

    .. math::
        f(x) := \frac{{\alpha}(x - a)^{\alpha - 1}}{(b - a)^{\alpha}}

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    alpha : Positive number, `0 < \alpha`, the shape parameter
    a : Real number, :math:`-\infty < a`, the left boundary
    b : Real number, :math:`a < b < \infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import PowerFunction, density, cdf, E, variance
    >>> from sympy import Symbol
    >>> alpha = Symbol("alpha", positive=True)
    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = PowerFunction("X", 2, a, b)

    >>> density(X)(z)
    (-2*a + 2*z)/(-a + b)**2

    >>> cdf(X)(z)
    Piecewise((a**2/(a**2 - 2*a*b + b**2) - 2*a*z/(a**2 - 2*a*b + b**2) +
    z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    >>> alpha = 2
    >>> a = 0
    >>> b = 1
    >>> Y = PowerFunction("Y", alpha, a, b)

    >>> E(Y)
    2/3

    >>> variance(Y)
    1/18

    References
    ==========

    .. [1] https://web.archive.org/web/20200204081320/http://www.mathwave.com/help/easyfit/html/analyses/distributions/power_func.html

    """
    return rv(name, PowerFunctionDistribution, (alpha, a, b))

#-------------------------------------------------------------------------------
# QuadraticU distribution ------------------------------------------------------

class QuadraticUDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(b > a, "Parameter b must be in range (%s, oo)."%(a))


注释：
- `PowerFunctionDistribution`: 定义了一个幂函数分布类，继承自`SingleContinuousDistribution`。
- `PowerFunctionDistribution.check`: 静态方法，检查参数的合法性，确保边界参数和形状参数的正确性。
- `PowerFunctionDistribution.pdf`: 定义了幂函数分布的概率密度函数。
- `PowerFunction`: 创建一个具有幂函数分布的连续随机变量，返回随机变量对象。
- `QuadraticUDistribution`: 定义了一个二次U分布类，继承自`SingleContinuousDistribution`。
- `QuadraticUDistribution.check`: 静态方法，检查参数的合法性，确保参数`b`大于参数`a`。
    # 定义一个函数，计算概率密度函数 (PDF) 在给定点 x 处的取值
    def pdf(self, x):
        # 从对象的属性中获取区间端点 a 和 b
        a, b = self.a, self.b
        # 计算参数 alpha，用于标准化 PDF
        alpha = 12 / (b-a)**3
        # 计算参数 beta，用于调整 PDF 的位置
        beta = (a+b) / 2
        # 返回一个分段函数 Piecewise，根据 x 的值不同返回不同的表达式
        return Piecewise(
                  (alpha * (x-beta)**2, And(a<=x, x<=b)),
                  (S.Zero, True))

    # 定义一个函数，计算矩生成函数 (MGF) 在给定点 t 处的取值
    def _moment_generating_function(self, t):
        # 从对象的属性中获取区间端点 a 和 b
        a, b = self.a, self.b
        # 根据区间端点 a 和 b 计算矩生成函数的表达式
        return -3 * (exp(a*t) * (4  + (a**2 + 2*a*(-2 + b) + b**2) * t) \
        - exp(b*t) * (4 + (-4*b + (a + b)**2) * t)) / ((a-b)**3 * t**2)

    # 定义一个函数，计算特征函数在给定点 t 处的取值
    def _characteristic_function(self, t):
        # 从对象的属性中获取区间端点 a 和 b
        a, b = self.a, self.b
        # 计算特征函数的表达式
        return -3*I*(exp(I*a*t*exp(I*b*t)) * (4*I - (-4*b + (a+b)**2)*t)) \
                / ((a-b)**3 * t**2)


这段代码定义了三个函数，分别用于计算统计分布中的概率密度函数 (PDF)、矩生成函数 (MGF) 和特征函数的值。每个函数都使用了对象的属性 `self.a` 和 `self.b`，这些属性很可能是定义统计分布区间的端点。
def QuadraticU(name, a, b):
    r"""
    Create a Continuous Random Variable with a U-quadratic distribution.

    Explanation
    ===========

    The density of the U-quadratic distribution is given by

    .. math::
        f(x) := \alpha (x-\beta)^2

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    a : Real number
        Lower bound of the distribution range.
    b : Real number, :math:`a < b`
        Upper bound of the distribution range.

    Returns
    =======

    RandomSymbol
        A symbolic representation of the random variable.

    Examples
    ========

    >>> from sympy.stats import QuadraticU, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = QuadraticU("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                2
    |   /  a   b    \
    |12*|- - - - + z|
    |   \  2   2    /
    <-----------------  for And(b >= z, a <= z)
    |            3
    |    (-a + b)
    |
    \        0                 otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/U-quadratic_distribution

    """

    # Return a random variable with the specified name and U-quadratic distribution
    return rv(name, QuadraticUDistribution, (a, b))

#-------------------------------------------------------------------------------
# RaisedCosine distribution ----------------------------------------------------

class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "s must be positive")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return Piecewise(
                ((1+cos(pi*(x-mu)/s)) / (2*s), And(mu-s<=x, x<=mu+s)),
                (S.Zero, True))

    def _characteristic_function(self, t):
        mu, s = self.mu, self.s
        return Piecewise((exp(-I*pi*mu/s)/2, Eq(t, -pi/s)),
                         (exp(I*pi*mu/s)/2, Eq(t, pi/s)),
                         (pi**2*sin(s*t)*exp(I*mu*t) / (s*t*(pi**2 - s**2*t**2)), True))

    def _moment_generating_function(self, t):
        mu, s = self.mu, self.s
        return pi**2 * sinh(s*t) * exp(mu*t) /  (s*t*(pi**2 + s**2*t**2))

def RaisedCosine(name, mu, s):
    r"""
    Create a Continuous Random Variable with a raised cosine distribution.

    Explanation
    ===========

    The density of the raised cosine distribution is given by

    .. math::
        f(x) := \frac{1}{2s}\left(1+\cos\left(\frac{x-\mu}{s}\pi\right)\right)

    with :math:`x \in [\mu-s,\mu+s]`.

    Parameters
    ==========

    mu : Real number
        Mean or central point of the distribution.
    s : Real number, `s > 0`
        Scale parameter controlling the width of the distribution.

    Returns
    =======

    RandomSymbol
        A symbolic representation of the random variable.

    Examples
    ========

    >>> from sympy.stats import RaisedCosine, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = RaisedCosine("x", mu, s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /   /pi*(-mu + z)\
    |cos|------------| + 1
    |   \     s      /
    <---------------------  for And(z >= mu - s, z <= mu + s)
    |         2*s
    |
    \          0                        otherwise

计算Raised Cosine分布的概率密度函数（PDF）在给定参数条件下的值。


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Raised_cosine_distribution

提供Raised Cosine分布的参考资料链接。


    """

    return rv(name, RaisedCosineDistribution, (mu, s))

返回一个具有指定名称、Raised Cosine分布类型和参数元组（mu, s）的随机变量。
#-------------------------------------------------------------------------------
# Rayleigh distribution --------------------------------------------------------

class RayleighDistribution(SingleContinuousDistribution):
    _argnames = ('sigma',)  # 定义参数名元组，只有一个参数 sigma

    set = Interval(0, oo)  # 定义取值范围为 (0, +∞)

    @staticmethod
    def check(sigma):
        _value_check(sigma > 0, "Scale parameter sigma must be positive.")
        # 检查参数 sigma 是否为正数，否则抛出异常

    def pdf(self, x):
        sigma = self.sigma  # 获取参数 sigma
        return x/sigma**2 * exp(-x**2/(2*sigma**2))
        # 返回 Rayleigh 分布的概率密度函数值

    def _cdf(self, x):
        sigma = self.sigma  # 获取参数 sigma
        return 1 - exp(-(x**2/(2*sigma**2)))
        # 返回 Rayleigh 分布的累积分布函数值

    def _characteristic_function(self, t):
        sigma = self.sigma  # 获取参数 sigma
        return 1 - sigma*t*exp(-sigma**2*t**2/2) * sqrt(pi/2) * (erfi(sigma*t/sqrt(2)) - I)
        # 返回 Rayleigh 分布的特征函数值

    def _moment_generating_function(self, t):
        sigma = self.sigma  # 获取参数 sigma
        return 1 + sigma*t*exp(sigma**2*t**2/2) * sqrt(pi/2) * (erf(sigma*t/sqrt(2)) + 1)
        # 返回 Rayleigh 分布的矩生成函数值


def Rayleigh(name, sigma):
    r"""
    Create a continuous random variable with a Rayleigh distribution.

    Explanation
    ===========

    The density of the Rayleigh distribution is given by

    .. math ::
        f(x) := \frac{x}{\sigma^2} e^{-x^2/2\sigma^2}

    with :math:`x > 0`.

    Parameters
    ==========

    sigma : Real number, `\sigma > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Rayleigh, density, E, variance
    >>> from sympy import Symbol

    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Rayleigh("x", sigma)

    >>> density(X)(z)
    z*exp(-z**2/(2*sigma**2))/sigma**2

    >>> E(X)
    sqrt(2)*sqrt(pi)*sigma/2

    >>> variance(X)
    -pi*sigma**2/2 + 2*sigma**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rayleigh_distribution
    .. [2] https://mathworld.wolfram.com/RayleighDistribution.html

    """

    return rv(name, RayleighDistribution, (sigma, ))

#-------------------------------------------------------------------------------
# Reciprocal distribution --------------------------------------------------------

class ReciprocalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')  # 定义参数名元组，包含参数 a 和 b

    @property
    def set(self):
        return Interval(self.a, self.b)
        # 返回取值范围为 [a, b]

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "Parameter > 0. a = %s"%a)
        # 检查参数 a 是否为正数，否则抛出异常
        _value_check((a < b),
        "Parameter b must be in range (%s, +oo]. b = %s"%(a, b))
        # 检查参数 b 是否大于 a，否则抛出异常

    def pdf(self, x):
        a, b = self.a, self.b  # 获取参数 a 和 b
        return 1/(x*(log(b) - log(a)))
        # 返回 Reciprocal 分布的概率密度函数值


def Reciprocal(name, a, b):
    r"""Creates a continuous random variable with a reciprocal distribution.


    Parameters
    ==========

    a : Real number, :math:`0 < a`
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Reciprocal, density, cdf
    >>> from sympy import symbols
    >>> a, b, x = symbols('a, b, x', positive=True)
    # 创建一个名为 R 的逆分布对象，参数为 a 和 b
    >>> R = Reciprocal('R', a, b)

    # 计算逆分布 R 的密度函数，并返回其表达式
    >>> density(R)(x)
    1/(x*(-log(a) + log(b)))
    
    # 计算逆分布 R 的累积分布函数，并返回其表达式
    >>> cdf(R)(x)
    Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))

    # 参考资料
    # =========
    # [1] https://en.wikipedia.org/wiki/Reciprocal_distribution

    """
    # 使用给定的名称 name 和逆分布 ReciprocalDistribution 的类来创建一个随机变量对象，并返回该对象
    return rv(name, ReciprocalDistribution, (a, b))
#-------------------------------------------------------------------------------
# Shifted Gompertz distribution ------------------------------------------------

class ShiftedGompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')  # 参数名称列表

    set = Interval(0, oo)  # 定义参数取值范围为大于0的实数集合

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")  # 检查参数b必须为正数
        _value_check(eta > 0, "eta must be positive")  # 检查参数eta必须为正数

    def pdf(self, x):
        b, eta = self.b, self.eta  # 获取参数b和eta
        return b*exp(-b*x)*exp(-eta*exp(-b*x))*(1+eta*(1-exp(-b*x)))  # 返回Shifted Gompertz分布的概率密度函数值

def ShiftedGompertz(name, b, eta):
    r"""
    Create a continuous random variable with a Shifted Gompertz distribution.

    Explanation
    ===========

    The density of the Shifted Gompertz distribution is given by

    .. math::
        f(x) := b e^{-b x} e^{-\eta \exp(-b x)} \left[1 + \eta(1 - e^(-bx)) \right]

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import ShiftedGompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> x = Symbol("x")

    >>> X = ShiftedGompertz("x", b, eta)

    >>> density(X)(x)
    b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution

    """
    return rv(name, ShiftedGompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# StudentT distribution --------------------------------------------------------

class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)  # 参数名称列表

    set = Interval(-oo, oo)  # 定义参数取值范围为实数集合

    @staticmethod
    def check(nu):
        _value_check(nu > 0, "Degrees of freedom nu must be positive.")  # 检查参数nu必须为正数

    def pdf(self, x):
        nu = self.nu  # 获取参数nu
        return 1/(sqrt(nu)*beta_fn(S.Half, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)  # 返回Student's t分布的概率密度函数值

    def _cdf(self, x):
        nu = self.nu  # 获取参数nu
        return S.Half + x*gamma((nu+1)/2)*hyper((S.Half, (nu+1)/2),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi*nu)*gamma(nu/2))  # 返回Student's t分布的累积分布函数值

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')  # 学生t分布的矩生成函数未定义

def StudentT(name, nu):
    r"""
    Create a continuous random variable with a student's t distribution.

    Explanation
    ===========

    The density of the student's t distribution is given by

    .. math::
        f(x) := \frac{\Gamma \left(\frac{\nu+1}{2} \right)}
                {\sqrt{\nu\pi}\Gamma \left(\frac{\nu}{2} \right)}
                \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    Parameters
    ==========

    nu : Real number, `\nu > 0`, the degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import StudentT, density
    >>> from sympy import Symbol

    >>> nu = Symbol("nu", positive=True)
    >>> x = Symbol("x")

    >>> X = StudentT("x", nu)

    >>> density(X)(x)
    gamma((nu + 1)/2)/(sqrt(pi*nu)*gamma(nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)

    References
    ==========

    """
    return rv(name, StudentTDistribution, (nu,))
    RandomSymbol


    # 定义一个未定义的符号或变量，这里可能是一个文档示例的错误或占位符
    RandomSymbol


    Examples
    ========

    >>> from sympy.stats import StudentT, density, cdf
    >>> from sympy import Symbol, pprint

    >>> nu = Symbol("nu", positive=True)
    >>> z = Symbol("z")

    >>> X = StudentT("x", nu)


    # 创建一个学生 t 分布的随机变量 X，命名为 "x"，参数为 nu
    X = StudentT("x", nu)


    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
               nu   1
             - -- - -
               2    2
     /     2\
     |    z |
     |1 + --|
     \    nu/
    -----------------
      ____  /     nu\
    \/ nu *B|1/2, --|
            \     2 /


    # 计算 X 在 z 处的概率密度函数值并打印
    D = density(X)(z)
    pprint(D, use_unicode=False)


    >>> cdf(X)(z)
    1/2 + z*gamma(nu/2 + 1/2)*hyper((1/2, nu/2 + 1/2), (3/2,),
                                -z**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))


    # 计算 X 在 z 处的累积分布函数值
    cdf(X)(z)


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Student_t-distribution
    .. [2] https://mathworld.wolfram.com/Studentst-Distribution.html


    # 返回一个随机变量对象，基于学生 t 分布，参数为 nu
    return rv(name, StudentTDistribution, (nu, ))
#-------------------------------------------------------------------------------
# Trapezoidal distribution ------------------------------------------------------

# 定义一个名为TrapezoidalDistribution的类，继承自SingleContinuousDistribution类
class TrapezoidalDistribution(SingleContinuousDistribution):
    # 定义属性_argnames为('a', 'b', 'c', 'd')
    _argnames = ('a', 'b', 'c', 'd')

    # 返回区间[a, d]的Interval对象
    @property
    def set(self):
        return Interval(self.a, self.d)

    # 静态方法，用于检查参数a, b, c, d是否符合定义域要求
    @staticmethod
    def check(a, b, c, d):
        # 检查a < d，否则抛出异常
        _value_check(a < d, "Lower bound parameter a < %s. a = %s"%(d, a))
        # 检查a <= b < c，否则抛出异常
        _value_check((a <= b, b < c),
                     "Level start parameter b must be in range [%s, %s). b = %s"%(a, c, b))
        # 检查b < c <= d，否则抛出异常
        _value_check((b < c, c <= d),
                     "Level end parameter c must be in range (%s, %s]. c = %s"%(b, d, c))
        # 检查d >= c，否则抛出异常
        _value_check(d >= c, "Upper bound parameter d > %s. d = %s"%(c, d))

    # 计算梯形分布的概率密度函数
    def pdf(self, x):
        a, b, c, d = self.a, self.b, self.c, self.d
        return Piecewise(
            (2*(x-a) / ((b-a)*(d+c-a-b)), And(a <= x, x < b)),
            (2 / (d+c-a-b), And(b <= x, x < c)),
            (2*(d-x) / ((d-c)*(d+c-a-b)), And(c <= x, x <= d)),
            (S.Zero, True))

# 创建一个名为Trapezoidal的函数，用于生成梯形分布的随机变量
def Trapezoidal(name, a, b, c, d):
    r"""
    Create a continuous random variable with a trapezoidal distribution.

    Explanation
    ===========

    The density of the trapezoidal distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(d+c-a-b)} & \mathrm{for\ } a \le x < b, \\
                  \frac{2}{d+c-a-b} & \mathrm{for\ } b \le x < c, \\
                  \frac{2(d-x)}{(d-c)(d+c-a-b)} & \mathrm{for\ } c \le x < d, \\
                  0 & \mathrm{for\ } d < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a < d`
    b : Real number, :math:`a \le b < c`
    c : Real number, :math:`b < c \le d`
    d : Real number

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Trapezoidal, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> d = Symbol("d")
    >>> z = Symbol("z")

    >>> X = Trapezoidal("x", a,b,c,d)

    >>> pprint(density(X)(z), use_unicode=False)
    /        -2*a + 2*z
    |-------------------------  for And(a <= z, b > z)
    |(-a + b)*(-a - b + c + d)
    |
    |           2
    |     --------------        for And(b <= z, c > z)
    <     -a - b + c + d
    |
    |        2*d - 2*z
    |-------------------------  for And(d >= z, c <= z)
    |(-c + d)*(-a - b + c + d)
    |
    \            0                     otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trapezoidal_distribution

    """
    # 返回一个随机变量对象，使用rv函数生成，指定分布类型为TrapezoidalDistribution，参数为(a, b, c, d)
    return rv(name, TrapezoidalDistribution, (a, b, c, d))

#-------------------------------------------------------------------------------
# Triangular distribution ------------------------------------------------------
    # 定义私有变量 _argnames，包含元组 ('a', 'b', 'c')
    _argnames = ('a', 'b', 'c')

    @property
    # 定义属性方法 set，返回一个 Interval 对象，其范围由 self.a 和 self.b 决定
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    # 定义静态方法 check，用于验证参数 a、b、c 的合法性
    def check(a, b, c):
        # 检查参数 b 是否大于 a，否则抛出异常
        _value_check(b > a, "Parameter b > %s. b = %s"%(a, b))
        # 检查参数 c 是否在区间 [a, b] 内，否则抛出异常
        _value_check((a <= c, c <= b),
        "Parameter c must be in range [%s, %s]. c = %s"%(a, b, c))

    def pdf(self, x):
        # 提取对象的属性 a、b、c 到局部变量 a、b、c
        a, b, c = self.a, self.b, self.c
        # 返回一个 Piecewise 对象，根据 x 的值返回相应的概率密度函数值
        return Piecewise(
            (2*(x - a)/((b - a)*(c - a)), And(a <= x, x < c)),
            (2/(b - a), Eq(x, c)),
            (2*(b - x)/((b - a)*(b - c)), And(c < x, x <= b)),
            (S.Zero, True))

    def _characteristic_function(self, t):
        # 提取对象的属性 a、b、c 到局部变量 a、b、c
        a, b, c = self.a, self.b, self.c
        # 计算特征函数的表达式并返回结果
        return -2 *((b-c) * exp(I*a*t) - (b-a) * exp(I*c*t) + (c-a) * exp(I*b*t)) / ((b-a)*(c-a)*(b-c)*t**2)

    def _moment_generating_function(self, t):
        # 提取对象的属性 a、b、c 到局部变量 a、b、c
        a, b, c = self.a, self.b, self.c
        # 计算动差生成函数的表达式并返回结果
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
        (b - a) * (c - a) * (b - c) * t ** 2)
# 创建一个具有三角分布的连续随机变量
def Triangular(name, a, b, c):
    r"""
    Create a continuous random variable with a triangular distribution.

    Explanation
    ===========

    The density of the triangular distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(c-a)} & \mathrm{for\ } a \le x < c, \\
                  \frac{2}{b-a} & \mathrm{for\ } x = c, \\
                  \frac{2(b-x)}{(b-a)(b-c)} & \mathrm{for\ } c < x \le b, \\
                  0 & \mathrm{for\ } b < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a \in \left(-\infty, \infty\right)`
    b : Real number, :math:`a < b`
    c : Real number, :math:`a \leq c \leq b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Triangular, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> z = Symbol("z")

    >>> X = Triangular("x", a,b,c)

    >>> pprint(density(X)(z), use_unicode=False)
    /    -2*a + 2*z
    |-----------------  for And(a <= z, c > z)
    |(-a + b)*(-a + c)
    |
    |       2
    |     ------              for c = z
    <     -a + b
    |
    |   2*b - 2*z
    |----------------   for And(b >= z, c < z)
    |(-a + b)*(b - c)
    |
    \        0                otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_distribution
    .. [2] https://mathworld.wolfram.com/TriangularDistribution.html

    """

    # 调用rv函数，创建具有三角分布参数的随机变量
    return rv(name, TriangularDistribution, (a, b, c))

#-------------------------------------------------------------------------------
# Uniform distribution ---------------------------------------------------------

# 定义一个UniformDistribution类，继承自SingleContinuousDistribution类
class UniformDistribution(SingleContinuousDistribution):
    _argnames = ('left', 'right')

    # 返回分布的区间
    @property
    def set(self):
        return Interval(self.left, self.right)

    # 检查参数是否合理
    @staticmethod
    def check(left, right):
        _value_check(left < right, "Lower limit should be less than Upper limit.")

    # 概率密度函数pdf
    def pdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.One/(right - left), And(left <= x, x <= right)),  # 区间内返回概率密度函数值，否则为0
            (S.Zero, True)  # 区间外返回0
        )

    # 累积分布函数cdf
    def _cdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.Zero, x < left),  # 小于左边界返回0
            ((x - left)/(right - left), x <= right),  # 区间内返回累积概率密度函数值
            (S.One, True)  # 大于右边界返回1
        )

    # 特征函数characteristic function
    def _characteristic_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(I*t*right) - exp(I*t*left)) / (I*t*(right - left)), Ne(t, 0)),  # t不等于0时的特征函数
                         (S.One, True))  # t等于0时的特征函数为1

    # 势函数moment generating function
    def _moment_generating_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(t*right) - exp(t*left)) / (t * (right - left)), Ne(t, 0)),  # t不等于0时的势函数
                         (S.One, True))  # t等于0时的势函数为1
    # 定义一个方法 `expectation`，接受参数 `expr` 表示表达式，`var` 表示变量，以及任意其他关键字参数 `kwargs`
        def expectation(self, expr, var, **kwargs):
            # 设置关键字参数 `evaluate` 为 True，用于指示期望值计算时要进行评估
            kwargs['evaluate'] = True
            # 调用 `SingleContinuousDistribution` 类的 `expectation` 方法，计算表达式 `expr` 关于变量 `var` 的期望值
            result = SingleContinuousDistribution.expectation(self, expr, var, **kwargs)
            # 使用最大值和最小值的替换规则来优化计算结果
            result = result.subs({Max(self.left, self.right): self.right,
                                  Min(self.left, self.right): self.left})
            # 返回计算得到的期望值结果
            return result
def UniformSum(name, n):
    r"""
    Create a continuous random variable with an Irwin-Hall distribution.

    Explanation
    ===========

    The probability distribution function depends on a single parameter
    $n$ which is an integer.

    The density of the Irwin-Hall distribution is given by

    .. math ::
        f(x) := \frac{1}{(n-1)!}\sum_{k=0}^{\left\lfloor x\right\rfloor}(-1)^k
                \binom{n}{k}(x-k)^{n-1}

    Parameters
    ==========

    n : A positive integer, `n > 0`

    Returns
    =======
    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Uniform, density, cdf, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", negative=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = UniformSum("x", n)

    >>> density(X)(z)
    Piecewise((1/factorial(n - 1)*Sum((-1)**k*binomial(n, k)*(z - k)**(n - 1),
               (k, 0, floor(z))), True)

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (1/factorial(n)*Sum((-1)**k*binomial(n, k)*(z - k)**(n),
               (k, 0, floor(z))), z <= n), (1, True))

    >>> E(X)
    n/2

    >>> simplify(variance(X))
    n/12

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution

    """
    return rv(name, UniformSumDistribution, (n,))


注释：
    # 返回一个符号变量，代表离散均匀分布的和分布
    return rv(name, UniformSumDistribution, (n, ))
#-------------------------------------------------------------------------------
# VonMises distribution --------------------------------------------------------

class VonMisesDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'k')  # 参数列表包含 mu 和 k

    set = Interval(0, 2*pi)  # 设置定义域为 [0, 2*pi]

    @staticmethod
    def check(mu, k):
        _value_check(k > 0, "k must be positive")  # 检查 k 是否为正数

    def pdf(self, x):
        mu, k = self.mu, self.k
        return exp(k*cos(x-mu)) / (2*pi*besseli(0, k))  # 计算 von Mises 分布的概率密度函数

def VonMises(name, mu, k):
    r"""
    Create a Continuous Random Variable with a von Mises distribution.

    Explanation
    ===========

    The density of the von Mises distribution is given by

    .. math::
        f(x) := \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    with :math:`x \in [0,2\pi]`.

    Parameters
    ==========

    mu : Real number
        Measure of location.
    k : Real number
        Measure of concentration.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import VonMises, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = VonMises("x", mu, k)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
         k*cos(mu - z)
        e
    ------------------
    2*pi*besseli(0, k)


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Von_Mises_distribution
    .. [2] https://mathworld.wolfram.com/vonMisesDistribution.html

    """

    return rv(name, VonMisesDistribution, (mu, k))

#-------------------------------------------------------------------------------
# Weibull distribution ---------------------------------------------------------

class WeibullDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')  # 参数列表包含 alpha 和 beta

    set = Interval(0, oo)  # 设置定义域为 (0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Alpha must be positive")  # 检查 alpha 是否为正数
        _value_check(beta > 0, "Beta must be positive")    # 检查 beta 是否为正数

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return beta * (x/alpha)**(beta - 1) * exp(-(x/alpha)**beta) / alpha  # 计算 Weibull 分布的概率密度函数

def Weibull(name, alpha, beta):
    r"""
    Create a continuous random variable with a Weibull distribution.

    Explanation
    ===========

    The density of the Weibull distribution is given by

    .. math::
        f(x) := \begin{cases}
                  \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
                  e^{-(x/\lambda)^{k}} & x\geq0\\
                  0 & x<0
                \end{cases}

    Parameters
    ==========

    lambda : Real number, $\lambda > 0$, a scale
    k : Real number, $k > 0$, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Weibull, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> l = Symbol("lambda", positive=True)
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    ...

    """

    return rv(name, WeibullDistribution, (alpha, beta))
    # 创建一个 Weibull 分布的随机变量 X，参数为 "x", l, k
    >>> X = Weibull("x", l, k)
    
    # 计算 Weibull 分布 X 的概率密度函数在给定值 z 处的值
    # 公式： k*(z/lambda)**(k - 1)*exp(-(z/lambda)**k)/lambda
    >>> density(X)(z)
    
    # 简化 Weibull 分布 X 的期望值
    # 公式： lambda*gamma(1 + 1/k)
    >>> simplify(E(X))
    
    # 简化 Weibull 分布 X 的方差
    # 公式： lambda**2*(-gamma(1 + 1/k)**2 + gamma(1 + 2/k))
    >>> simplify(variance(X))
    
    # 参考资料
    # [1] https://en.wikipedia.org/wiki/Weibull_distribution
    # [2] https://mathworld.wolfram.com/WeibullDistribution.html
    
    """
    返回一个随机变量对象 rv(name, WeibullDistribution, (alpha, beta))，其中 alpha = "x"，beta = Weibull 分布的参数 (l, k)
    """
    return rv(name, WeibullDistribution, (alpha, beta))
#-------------------------------------------------------------------------------
# Wigner semicircle distribution -----------------------------------------------

# 定义 WignerSemicircleDistribution 类，继承自 SingleContinuousDistribution
class WignerSemicircleDistribution(SingleContinuousDistribution):
    _argnames = ('R',)

    # 返回定义域为 [-R, R] 的区间对象
    @property
    def set(self):
        return Interval(-self.R, self.R)

    # 检查半径 R 是否为正数
    @staticmethod
    def check(R):
        _value_check(R > 0, "Radius R must be positive.")

    # 计算概率密度函数，给定公式为 Wigner semicircle distribution 的密度函数
    def pdf(self, x):
        R = self.R
        return 2/(pi*R**2)*sqrt(R**2 - x**2)

    # 计算特征函数，使用分段函数表示
    def _characteristic_function(self, t):
        return Piecewise((2 * besselj(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

    # 计算生成函数，使用分段函数表示
    def _moment_generating_function(self, t):
        return Piecewise((2 * besseli(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

# 创建 Wigner semicircle 分布的随机变量
def WignerSemicircle(name, R):
    r"""
    Create a continuous random variable with a Wigner semicircle distribution.

    Explanation
    ===========

    The density of the Wigner semicircle distribution is given by

    .. math::
        f(x) := \frac2{\pi R^2}\,\sqrt{R^2-x^2}

    with :math:`x \in [-R,R]`.

    Parameters
    ==========

    R : Real number, `R > 0`, the radius

    Returns
    =======

    A RandomSymbol.

    Examples
    ========

    >>> from sympy.stats import WignerSemicircle, density, E
    >>> from sympy import Symbol

    >>> R = Symbol("R", positive=True)
    >>> z = Symbol("z")

    >>> X = WignerSemicircle("x", R)

    >>> density(X)(z)
    2*sqrt(R**2 - z**2)/(pi*R**2)

    >>> E(X)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    .. [2] https://mathworld.wolfram.com/WignersSemicircleLaw.html

    """

    return rv(name, WignerSemicircleDistribution, (R,))
```