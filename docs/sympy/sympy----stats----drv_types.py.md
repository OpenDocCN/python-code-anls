# `D:\src\scipysrc\sympy\sympy\stats\drv_types.py`

```
# 导入所需模块和函数

from sympy.concrete.summations import Sum  # 导入 Sum 函数，用于表示求和表达式
from sympy.core.basic import Basic  # 导入 Basic 类，作为所有 SymPy 核心类的基类
from sympy.core.function import Lambda  # 导入 Lambda 函数，用于表示匿名函数
from sympy.core.numbers import I  # 导入虚数单位 I
from sympy.core.relational import Eq  # 导入 Eq 函数，用于表示等式
from sympy.core.singleton import S  # 导入 S 单例，表示 SymPy 的单例对象
from sympy.core.symbol import Dummy  # 导入 Dummy 符号，用于生成临时符号
from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将输入转换为 SymPy 对象
from sympy.functions.combinatorial.factorials import (binomial, factorial)  # 导入组合数学函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.integers import floor  # 导入 floor 函数，向下取整
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数，计算平方根
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 函数，表示分段函数
from sympy.functions.special.bessel import besseli  # 导入贝塞尔函数
from sympy.functions.special.beta_functions import beta  # 导入贝塞尔函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.functions.special.zeta_functions import (polylog, zeta)  # 导入多重对数和黎曼 zeta 函数
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace  # 导入离散随机变量相关类
from sympy.stats.rv import _value_check, is_random  # 导入随机变量检查函数和随机性判断函数

# 定义 __all__ 列表，用于模块的导出控制
__all__ = ['FlorySchulz',
           'Geometric',
           'Hermite',
           'Logarithmic',
           'NegativeBinomial',
           'Poisson',
           'Skellam',
           'YuleSimon',
           'Zeta'
           ]


def rv(symbol, cls, *args, **kwargs):
    """
    Create a random variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents the name of the random variable.
    cls : type
        Class representing the distribution.
    *args : tuple
        Arguments for the distribution class.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    =======

    pspace.value : value
        Value of the random variable.
    """

    args = list(map(sympify, args))  # 将所有参数转换为 SymPy 对象
    dist = cls(*args)  # 使用参数创建指定类型的分布对象
    if kwargs.pop('check', True):  # 如果指定需要检查
        dist.check(*args)  # 调用分布对象的检查方法
    pspace = SingleDiscretePSpace(symbol, dist)  # 创建离散随机变量的概率空间对象
    if any(is_random(arg) for arg in args):  # 如果任何一个参数是随机变量
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))  # 创建复合随机变量的概率空间对象
    return pspace.value  # 返回随机变量的值


class DiscreteDistributionHandmade(SingleDiscreteDistribution):
    """
    A custom class for discrete distributions inheriting from
    SingleDiscreteDistribution.

    Attributes
    ==========

    _argnames : tuple
        Names of arguments for the distribution.
    """

    _argnames = ('pdf',)

    def __new__(cls, pdf, set=S.Integers):
        """
        Create a new instance of DiscreteDistributionHandmade.

        Parameters
        ==========

        pdf : callable
            Probability density function.
        set : set
            Set where the pdf is valid, default is set of integers.

        Returns
        =======

        Basic
            New instance of DiscreteDistributionHandmade.
        """
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        """
        Return the set where the pdf is valid.

        Returns
        =======

        set
            Set where the pdf is valid.
        """
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        """
        Check whether the given pdf integrates to 1 over the given set.

        Parameters
        ==========

        pdf : callable
            Probability density function.
        set : set
            Set where the pdf is valid.

        Raises
        ======

        ValueError
            If the pdf is incorrect on the given set.
        """
        x = Dummy('x')  # 创建虚拟符号 x
        val = Sum(pdf(x), (x, set._inf, set._sup)).doit()  # 计算 pdf 在指定集合上的积分
        _value_check(Eq(val, 1) != S.false, "The pdf is incorrect on the given set.")


def DiscreteRV(symbol, density, set=S.Integers, **kwargs):
    """
    Create a Discrete Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents the name of the random variable.
    density : Expression containing symbol
        Represents the probability density function.
    set : set
        Represents the region where the pdf is valid, by default is set of integers.
    check : bool
        If True, it will check whether the given density integrates to 1 over the given set.
        If False, it will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import DiscreteRV, P, E
    >>> from sympy import Rational, Symbol
    >>> x = Symbol('x')
    >>> n = 10
    >>> density = Rational(1, 10)
    >>> X = DiscreteRV(x, density, set=set(range(n)))
    >>> E(X)
    9/2
    >>> P(X > 3)
    3/5

    Returns
    =======

    """
    RandomSymbol

    """
    # 将 `set` 转换为 sympy 符号
    set = sympify(set)
    # 根据符号和密度创建分段函数 pdf
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    # 将 pdf 封装为 Lambda 函数
    pdf = Lambda(symbol, pdf)
    # 设置 kwargs 字典中 'check' 键的默认值为 False，如果已存在则不变
    kwargs['check'] = kwargs.pop('check', False)
    # 使用 rv 函数创建随机变量，使用 DiscreteDistributionHandmade 分布和指定的 pdf、set 参数
    # 其余参数由 kwargs 提供，默认情况下 `check` 为 False
    return rv(symbol.name, DiscreteDistributionHandmade, pdf, set, **kwargs)
    """
#-------------------------------------------------------------------------------
# Flory-Schulz distribution ------------------------------------------------------------

class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        # 检查参数 a 是否在 (0, 1) 范围内
        _value_check((0 < a, a < 1), "a must be between 0 and 1")

    def pdf(self, k):
        a = self.a
        # 计算 Flory-Schulz 分布的概率质量函数
        return (a**2 * k * (1 - a)**(k - 1))

    def _characteristic_function(self, t):
        a = self.a
        # 计算特征函数
        return a**2*exp(I*t)/((1 + (a - 1)*exp(I*t))**2)

    def _moment_generating_function(self, t):
        a = self.a
        # 计算矩生成函数
        return a**2*exp(t)/((1 + (a - 1)*exp(t))**2)


def FlorySchulz(name, a):
    r"""
    Create a discrete random variable with a FlorySchulz distribution.

    The density of the FlorySchulz distribution is given by

    .. math::
        f(k) := (a^2) k (1 - a)^{k-1}

    Parameters
    ==========

    a : A real number between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, E, variance, FlorySchulz
    >>> from sympy import Symbol, S

    >>> a = S.One / 5
    >>> z = Symbol("z")

    >>> X = FlorySchulz("x", a)

    >>> density(X)(z)
    (5/4)**(1 - z)*z/25

    >>> E(X)
    9

    >>> variance(X)
    40

    References
    ==========

    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution
    """
    # 使用给定的参数 a 创建一个 FlorySchulzDistribution 实例
    return rv(name, FlorySchulzDistribution, a)


#-------------------------------------------------------------------------------
# Geometric distribution ------------------------------------------------------------

class GeometricDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        # 检查参数 p 是否在 (0, 1] 范围内
        _value_check((0 < p, p <= 1), "p must be between 0 and 1")

    def pdf(self, k):
        # 计算 Geometric 分布的概率质量函数
        return (1 - self.p)**(k - 1) * self.p

    def _characteristic_function(self, t):
        p = self.p
        # 计算特征函数
        return p * exp(I*t) / (1 - (1 - p)*exp(I*t))

    def _moment_generating_function(self, t):
        p = self.p
        # 计算矩生成函数
        return p * exp(t) / (1 - (1 - p) * exp(t))


def Geometric(name, p):
    r"""
    Create a discrete random variable with a Geometric distribution.

    Explanation
    ===========

    The density of the Geometric distribution is given by

    .. math::
        f(k) := p (1 - p)^{k - 1}

    Parameters
    ==========

    p : A probability between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Geometric, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Geometric("x", p)

    >>> density(X)(z)
    (5/4)**(1 - z)/5

    >>> E(X)
    5

    >>> variance(X)
    20

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution
    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html
    """
    # 使用给定的参数 p 创建一个 GeometricDistribution 实例
    return rv(name, GeometricDistribution, p)
    """
    返回一个从几何分布中随机抽取的样本。

    Parameters:
    name: str
        样本的名称或标识符。
    GeometricDistribution: Distribution
        表示几何分布的类或对象。
    p: float
        成功的概率，范围在 0 和 1 之间。

    Returns:
    rv: object
        表示从几何分布中抽取的随机变量对象。
    """
#-------------------------------------------------------------------------------
# Hermite distribution ---------------------------------------------------------

# 定义 HermiteDistribution 类，继承自 SingleDiscreteDistribution 类
class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')  # 参数名为 a1 和 a2
    set = S.Naturals0  # 定义可取值集合为非负整数集合 Naturals0

    @staticmethod
    def check(a1, a2):
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')  # 检查 a1 是否为非负数，否则抛出异常
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')  # 检查 a2 是否为非负数，否则抛出异常

    # 定义概率密度函数 pdf
    def pdf(self, k):
        a1, a2 = self.a1, self.a2
        term1 = exp(-(a1 + a2))  # 计算指数项的值
        j = Dummy("j", integer=True)  # 创建一个整数符号 j
        num = a1**(k - 2*j) * a2**j  # 计算分子的值
        den = factorial(k - 2*j) * factorial(j)  # 计算分母的阶乘
        return term1 * Sum(num/den, (j, 0, k//2)).doit()  # 返回计算结果

    # 定义矩生成函数 _moment_generating_function
    def _moment_generating_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(t) - 1)  # 计算第一个指数项
        term2 = a2 * (exp(2*t) - 1)  # 计算第二个指数项
        return exp(term1 + term2)  # 返回矩生成函数的值

    # 定义特征函数 _characteristic_function
    def _characteristic_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(I*t) - 1)  # 计算第一个指数项
        term2 = a2 * (exp(2*I*t) - 1)  # 计算第二个指数项
        return exp(term1 + term2)  # 返回特征函数的值

# 创建 Hermite 分布的随机变量
def Hermite(name, a1, a2):
    r"""
    Create a discrete random variable with a Hermite distribution.

    Explanation
    ===========

    The density of the Hermite distribution is given by

    .. math::
        f(x):= e^{-a_1 -a_2}\sum_{j=0}^{\left \lfloor x/2 \right \rfloor}
                    \frac{a_{1}^{x-2j}a_{2}^{j}}{(x-2j)!j!}

    Parameters
    ==========

    a1 : A Positive number greater than equal to 0.
    a2 : A Positive number greater than equal to 0.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Hermite, density, E, variance
    >>> from sympy import Symbol

    >>> a1 = Symbol("a1", positive=True)
    >>> a2 = Symbol("a2", positive=True)
    >>> x = Symbol("x")

    >>> H = Hermite("H", a1=5, a2=4)

    >>> density(H)(2)
    33*exp(-9)/2

    >>> E(H)
    13

    >>> variance(H)
    21

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_distribution

    """
    # 调用 rv 函数创建 Hermite 分布的随机变量
    return rv(name, HermiteDistribution, a1, a2)


#-------------------------------------------------------------------------------
# Logarithmic distribution ------------------------------------------------------------

# 定义 LogarithmicDistribution 类，继承自 SingleDiscreteDistribution 类
class LogarithmicDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)  # 参数名为 p
    set = S.Naturals  # 定义可取值集合为正整数集合 Naturals

    @staticmethod
    def check(p):
        _value_check((p > 0, p < 1), "p should be between 0 and 1")  # 检查 p 是否在 (0, 1) 区间内，否则抛出异常

    # 定义概率密度函数 pdf
    def pdf(self, k):
        p = self.p
        return (-1) * p**k / (k * log(1 - p))  # 返回计算结果

    # 定义特征函数 _characteristic_function
    def _characteristic_function(self, t):
        p = self.p
        return log(1 - p * exp(I*t)) / log(1 - p)  # 返回计算结果

    # 定义矩生成函数 _moment_generating_function
    def _moment_generating_function(self, t):
        p = self.p
        return log(1 - p * exp(t)) / log(1 - p)  # 返回计算结果


# 创建 Logarithmic 分布的随机变量
def Logarithmic(name, p):
    r"""
    Create a discrete random variable with a Logarithmic distribution.

    Explanation
    ===========

    The density of the Logarithmic distribution is given by

    .. math::
        f(x) := -\frac{p^x}{x \log(1 - p)}

    Parameters
    ==========

    p : A number between 0 and 1 (exclusive).

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density
    >>> from sympy import Symbol

    >>> p = Symbol("p", positive=True, less_than=1)
    >>> x = Symbol("x", integer=True, positive=True)

    >>> L = Logarithmic("L", p=0.3)

    >>> density(L)(2)
    -3*log(1 - 0.3)/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution

    """
    # 调用 rv 函数创建 Logarithmic 分布的随机变量
    return rv(name, LogarithmicDistribution, p)
    """
    The density of the Logarithmic distribution is given by

    .. math::
        f(k) := \frac{-p^k}{k \ln{(1 - p)}}

    Parameters
    ==========

    p : A value between 0 and 1, probability parameter of the Logarithmic distribution.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5  # Define probability parameter p as 1/5
    >>> z = Symbol("z")  # Create a symbolic variable z

    >>> X = Logarithmic("x", p)  # Define a Logarithmic random variable X with parameter p

    >>> density(X)(z)  # Compute the density function of X at z
    -1/(5**z*z*log(4/5))

    >>> E(X)  # Compute the expected value (mean) of X
    -1/(-4*log(5) + 8*log(2))

    >>> variance(X)  # Compute the variance of X
    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution
    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html

    """
    return rv(name, LogarithmicDistribution, p)


Here is the annotated code block with explanations for each line:


    """
    The density of the Logarithmic distribution is given by

    .. math::
        f(k) := \frac{-p^k}{k \ln{(1 - p)}}

    Parameters
    ==========

    p : A value between 0 and 1, probability parameter of the Logarithmic distribution.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5  # Define probability parameter p as 1/5
    >>> z = Symbol("z")  # Create a symbolic variable z

    >>> X = Logarithmic("x", p)  # Define a Logarithmic random variable X with parameter p

    >>> density(X)(z)  # Compute the density function of X at z
    -1/(5**z*z*log(4/5))

    >>> E(X)  # Compute the expected value (mean) of X
    -1/(-4*log(5) + 8*log(2))

    >>> variance(X)  # Compute the variance of X
    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution
    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html

    """
    return rv(name, LogarithmicDistribution, p)


This block contains detailed explanations for the definition, parameters, examples, and references related to the Logarithmic distribution. Each line is annotated to provide clarity on its purpose and functionality within the context of using the SymPy library for statistical computations involving the Logarithmic distribution.
#-------------------------------------------------------------------------------
# Negative binomial distribution ------------------------------------------------------------

class NegativeBinomialDistribution(SingleDiscreteDistribution):
    _argnames = ('r', 'p')
    set = S.Naturals0

    @staticmethod
    def check(r, p):
        _value_check(r > 0, 'r should be positive')  # 检查 r 必须为正数
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')  # 检查 p 必须在 0 和 1 之间

    def pdf(self, k):
        r = self.r  # 获取负二项分布的参数 r
        p = self.p  # 获取负二项分布的参数 p

        return binomial(k + r - 1, k) * (1 - p)**r * p**k  # 返回负二项分布的概率质量函数值

    def _characteristic_function(self, t):
        r = self.r  # 获取负二项分布的参数 r
        p = self.p  # 获取负二项分布的参数 p

        return ((1 - p) / (1 - p * exp(I*t)))**r  # 返回负二项分布的特征函数值

    def _moment_generating_function(self, t):
        r = self.r  # 获取负二项分布的参数 r
        p = self.p  # 获取负二项分布的参数 p

        return ((1 - p) / (1 - p * exp(t)))**r  # 返回负二项分布的矩生成函数值

def NegativeBinomial(name, r, p):
    r"""
    Create a discrete random variable with a Negative Binomial distribution.

    Explanation
    ===========

    The density of the Negative Binomial distribution is given by

    .. math::
        f(k) := \binom{k + r - 1}{k} (1 - p)^r p^k

    Parameters
    ==========

    r : A positive value
    p : A value between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import NegativeBinomial, density, E, variance
    >>> from sympy import Symbol, S

    >>> r = 5
    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = NegativeBinomial("x", r, p)

    >>> density(X)(z)
    1024*binomial(z + 4, z)/(3125*5**z)

    >>> E(X)
    5/4

    >>> variance(X)
    25/16

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    return rv(name, NegativeBinomialDistribution, r, p)


#-------------------------------------------------------------------------------
# Poisson distribution ------------------------------------------------------------

class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)

    set = S.Naturals0

    @staticmethod
    def check(lamda):
        _value_check(lamda > 0, "Lambda must be positive")  # 检查 lambda 必须为正数

    def pdf(self, k):
        return self.lamda**k / factorial(k) * exp(-self.lamda)  # 返回泊松分布的概率质量函数值

    def _characteristic_function(self, t):
        return exp(self.lamda * (exp(I*t) - 1))  # 返回泊松分布的特征函数值

    def _moment_generating_function(self, t):
        return exp(self.lamda * (exp(t) - 1))  # 返回泊松分布的矩生成函数值


def Poisson(name, lamda):
    r"""
    Create a discrete random variable with a Poisson distribution.

    Explanation
    ===========

    The density of the Poisson distribution is given by

    .. math::
        f(k) := \frac{\lambda^{k} e^{- \lambda}}{k!}

    Parameters
    ==========

    lamda : Positive number, a rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Poisson, density, E, variance
    >>> from sympy import Symbol, S

    >>> l = 3
    >>> z = Symbol("z")

    >>> Y = Poisson("y", l)

    >>> density(Y)(z)
    3**z*exp(-3)/factorial(z)

    >>> E(Y)
    3

    >>> variance(Y)
    3

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution

    """
    # 导入 sympy 库中的 Symbol 和 simplify 函数
    >>> from sympy import Symbol, simplify

    # 创建一个名为 rate 的符号变量 lambda，并指定其为正数
    >>> rate = Symbol("lambda", positive=True)
    
    # 创建一个名为 z 的符号变量
    >>> z = Symbol("z")
    
    # 创建一个名为 X 的泊松分布随机变量，其参数为 rate
    >>> X = Poisson("x", rate)
    
    # 计算 X 的概率密度函数在 z 处的值
    >>> density(X)(z)
    lambda**z*exp(-lambda)/factorial(z)
    
    # 计算 X 的期望值
    >>> E(X)
    lambda
    
    # 简化 X 的方差并返回结果
    >>> simplify(variance(X))
    lambda
    
    # 返回一个随机变量对象，其名称为 name，分布类型为 PoissonDistribution，参数为 lamda
    """
    return rv(name, PoissonDistribution, lamda)
    ```
# -----------------------------------------------------------------------------
# Skellam distribution --------------------------------------------------------

# 定义 Skellam 分布类，继承自 SingleDiscreteDistribution
class SkellamDistribution(SingleDiscreteDistribution):
    _argnames = ('mu1', 'mu2')
    set = S.Integers

    @staticmethod
    def check(mu1, mu2):
        _value_check(mu1 >= 0, 'Parameter mu1 must be >= 0')  # 检查参数 mu1 是否大于等于 0
        _value_check(mu2 >= 0, 'Parameter mu2 must be >= 0')  # 检查参数 mu2 是否大于等于 0

    # 计算 Skellam 分布的概率质量函数
    def pdf(self, k):
        (mu1, mu2) = (self.mu1, self.mu2)
        term1 = exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)  # 第一个系数 term1
        term2 = besseli(k, 2 * sqrt(mu1 * mu2))  # 第二个系数 term2
        return term1 * term2  # 返回概率质量函数的值

    # 未实现累积分布函数的方法，抛出 NotImplementedError
    def _cdf(self, x):
        raise NotImplementedError("Skellam doesn't have closed form for the CDF.")

    # 计算 Skellam 分布的特征函数
    def _characteristic_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(I * t) + mu2 * exp(-I * t))  # 返回特征函数的值

    # 计算 Skellam 分布的矩生成函数
    def _moment_generating_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(t) + mu2 * exp(-t))  # 返回矩生成函数的值


# 创建 Skellam 分布的随机变量
def Skellam(name, mu1, mu2):
    r"""
    Create a discrete random variable with a Skellam distribution.

    Explanation
    ===========

    The Skellam is the distribution of the difference N1 - N2
    of two statistically independent random variables N1 and N2
    each Poisson-distributed with respective expected values mu1 and mu2.

    The density of the Skellam distribution is given by

    .. math::
        f(k) := e^{-(\mu_1+\mu_2)}(\frac{\mu_1}{\mu_2})^{k/2}I_k(2\sqrt{\mu_1\mu_2})

    Parameters
    ==========

    mu1 : A non-negative value
    mu2 : A non-negative value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Skellam, density, E, variance
    >>> from sympy import Symbol, pprint

    >>> z = Symbol("z", integer=True)
    >>> mu1 = Symbol("mu1", positive=True)
    >>> mu2 = Symbol("mu2", positive=True)
    >>> X = Skellam("x", mu1, mu2)

    >>> pprint(density(X)(z), use_unicode=False)
         z
         -
         2
    /mu1\   -mu1 - mu2        /       _____   _____\
    |---| *e          *besseli\z, 2*\/ mu1 *\/ mu2 /
    \mu2/
    >>> E(X)
    mu1 - mu2
    >>> variance(X).expand()
    mu1 + mu2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Skellam_distribution

    """
    return rv(name, SkellamDistribution, mu1, mu2)


#-------------------------------------------------------------------------------
# Yule-Simon distribution ------------------------------------------------------------

# 定义 Yule-Simon 分布类，继承自 SingleDiscreteDistribution
class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        _value_check(rho > 0, 'rho should be positive')  # 检查参数 rho 是否大于 0

    # 计算 Yule-Simon 分布的概率质量函数
    def pdf(self, k):
        rho = self.rho
        return rho * beta(k, rho + 1)

    # 计算 Yule-Simon 分布的累积分布函数
    def _cdf(self, x):
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))
    # 定义私有方法 _characteristic_function，计算特征函数
    def _characteristic_function(self, t):
        # 从对象属性中获取 rho 值
        rho = self.rho
        # 计算特征函数并返回，使用超几何函数 hyper() 和复指数函数 exp(I*t)
        return rho * hyper((1, 1), (rho + 2,), exp(I*t)) * exp(I*t) / (rho + 1)

    # 定义私有方法 _moment_generating_function，计算矩生成函数
    def _moment_generating_function(self, t):
        # 从对象属性中获取 rho 值
        rho = self.rho
        # 计算矩生成函数并返回，使用超几何函数 hyper() 和指数函数 exp(t)
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)
# 定义一个函数 YuleSimon，用于创建一个服从 Yule-Simon 分布的离散随机变量
def YuleSimon(name, rho):
    r"""
    Create a discrete random variable with a Yule-Simon distribution.

    Explanation
    ===========

    The density of the Yule-Simon distribution is given by

    .. math::
        f(k) := \rho B(k, \rho + 1)

    Parameters
    ==========

    rho : A positive value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import YuleSimon, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> p = 5
    >>> z = Symbol("z")

    >>> X = YuleSimon("x", p)

    >>> density(X)(z)
    5*beta(z, 6)

    >>> simplify(E(X))
    5/4

    >>> simplify(variance(X))
    25/48

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution

    """
    # 调用 rv 函数创建一个随机变量，基于 Yule-Simon 分布的参数 rho
    return rv(name, YuleSimonDistribution, rho)


#-------------------------------------------------------------------------------
# Zeta distribution ------------------------------------------------------------

# 定义 ZetaDistribution 类，表示 Zeta 分布的单一离散分布
class ZetaDistribution(SingleDiscreteDistribution):
    _argnames = ('s',)  # 定义参数名称元组，仅包含 s

    # 设置集合为自然数集 S.Naturals
    set = S.Naturals

    @staticmethod
    # 静态方法：检查参数 s 是否大于 1，否则抛出异常
    def check(s):
        _value_check(s > 1, 's should be greater than 1')

    # 概率密度函数 pdf，接受参数 k
    def pdf(self, k):
        s = self.s
        # 返回 Zeta 分布的概率密度函数值
        return 1 / (k**s * zeta(s))

    # 特征函数 _characteristic_function，接受参数 t
    def _characteristic_function(self, t):
        # 返回 Zeta 分布的特征函数
        return polylog(self.s, exp(I*t)) / zeta(self.s)

    # 指数生成函数 _moment_generating_function，接受参数 t
    def _moment_generating_function(self, t):
        # 返回 Zeta 分布的指数生成函数
        return polylog(self.s, exp(t)) / zeta(self.s)


# 定义函数 Zeta，用于创建一个服从 Zeta 分布的离散随机变量
def Zeta(name, s):
    r"""
    Create a discrete random variable with a Zeta distribution.

    Explanation
    ===========

    The density of the Zeta distribution is given by

    .. math::
        f(k) := \frac{1}{k^s \zeta{(s)}}

    Parameters
    ==========

    s : A value greater than 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Zeta, density, E, variance
    >>> from sympy import Symbol

    >>> s = 5
    >>> z = Symbol("z")

    >>> X = Zeta("x", s)

    >>> density(X)(z)
    1/(z**5*zeta(5))

    >>> E(X)
    pi**4/(90*zeta(5))

    >>> variance(X)
    -pi**8/(8100*zeta(5)**2) + zeta(3)/zeta(5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zeta_distribution

    """
    # 调用 rv 函数创建一个随机变量，基于 Zeta 分布的参数 s
    return rv(name, ZetaDistribution, s)
```