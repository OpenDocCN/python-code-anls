# `D:\src\scipysrc\sympy\sympy\stats\rv_interface.py`

```
# 从 sympy.sets 模块导入 FiniteSet 类
# 从 sympy.core.numbers 模块导入 Rational 类
# 从 sympy.core.relational 模块导入 Eq 类
# 从 sympy.core.symbol 模块导入 Dummy 类
# 从 sympy.functions.combinatorial.factorials 模块导入 FallingFactorial 函数
# 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
# 从 sympy.functions.elementary.piecewise 模块导入 piecewise_fold 函数
# 从 sympy.integrals.integrals 模块导入 Integral 类
# 从 sympy.solvers.solveset 模块导入 solveset 函数
# 从当前包的 rv 模块导入 probability, expectation, density, where, given, pspace, cdf, PSpace,
# characteristic_function, sample, sample_iter, random_symbols, independent, dependent,
# sampling_density, moment_generating_function, quantile, is_random, sample_stochastic_process 函数和类
from sympy.sets import FiniteSet
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.integrals.integrals import Integral
from sympy.solvers.solveset import solveset
from .rv import (probability, expectation, density, where, given, pspace, cdf, PSpace,
                 characteristic_function, sample, sample_iter, random_symbols, independent, dependent,
                 sampling_density, moment_generating_function, quantile, is_random,
                 sample_stochastic_process)

# 将下列符号添加到模块的公共接口中
__all__ = ['P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf',
        'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
        'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'median',
        'independent', 'random_symbols', 'correlation', 'factorial_moment',
        'moment', 'cmoment', 'sampling_density', 'moment_generating_function',
        'smoment', 'quantile', 'sample_stochastic_process']

# 定义函数 moment，计算随机表达式 X 的第 n 阶矩关于 c 的值
def moment(X, n, c=0, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth moment of a random expression about c.

    .. math::
        moment(X, c, n) = E((X-c)^{n})

    Default value of c is 0.

    Examples
    ========

    >>> from sympy.stats import Die, moment, E
    >>> X = Die('X', 6)
    >>> moment(X, 1, 6)
    -5/2
    >>> moment(X, 2)
    91/6
    >>> moment(X, 1) == E(X)
    True
    """
    # 导入 Moment 类并计算期望值
    from sympy.stats.symbolic_probability import Moment
    if evaluate:
        return Moment(X, n, c, condition).doit()
    return Moment(X, n, c, condition).rewrite(Integral)

# 定义函数 variance，计算随机表达式 X 的方差
def variance(X, condition=None, **kwargs):
    """
    Variance of a random expression.

    .. math::
        variance(X) = E((X-E(X))^{2})

    Examples
    ========

    >>> from sympy.stats import Die, Bernoulli, variance
    >>> from sympy import simplify, Symbol

    >>> X = Die('X', 6)
    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> variance(2*X)
    35/3

    >>> simplify(variance(B))
    p*(1 - p)
    """
    # 如果 X 是随机表达式且其概率空间是 PSpace()，则计算方差
    if is_random(X) and pspace(X) == PSpace():
        from sympy.stats.symbolic_probability import Variance
        return Variance(X, condition)

    # 否则，使用 cmoment 函数计算第二阶中心矩
    return cmoment(X, 2, condition, **kwargs)

# 定义函数 standard_deviation，计算随机表达式 X 的标准差
def standard_deviation(X, condition=None, **kwargs):
    r"""
    Standard Deviation of a random expression

    .. math::
        std(X) = \sqrt(E((X-E(X))^{2}))

    Examples
    ========

    >>> from sympy.stats import Bernoulli, std
    >>> from sympy import Symbol, simplify

    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> simplify(std(B))
    sqrt(p*(1 - p))
    """
    return sqrt(variance(X, condition, **kwargs))

# 将 standard_deviation 函数绑定到 std 变量上
std = standard_deviation
# 计算概率分布的熵值。

# Parameters
# ==========
# expr : 要计算熵的随机表达式
# condition : 可选，对随机表达式的条件限制
# kwargs : 其他参数，例如对数的基数

# Returns
# =======
# result : 表达式的熵值，一个常数

# Examples
# ========
# >>> from sympy.stats import Normal, Die, entropy
# >>> X = Normal('X', 0, 1)
# >>> entropy(X)
# log(2)/2 + 1/2 + log(pi)/2
# >>> D = Die('D', 4)
# >>> entropy(D)
# log(4)

# References
# ==========
# .. [1] https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
# .. [2] https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf
# .. [3] https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf
def entropy(expr, condition=None, **kwargs):
    pdf = density(expr, condition, **kwargs)  # 获取概率密度函数
    base = kwargs.get('b', exp(1))  # 获取对数的基数，默认为自然对数的底数 e
    if isinstance(pdf, dict):
        return sum(-prob*log(prob, base) for prob in pdf.values())  # 计算熵值
    return expectation(-log(pdf(expr), base))  # 计算期望值


# 计算两个随机表达式的协方差。

# Parameters
# ==========
# X : 第一个随机表达式
# Y : 第二个随机表达式
# condition : 可选，对随机表达式的条件限制
# kwargs : 其他参数

# Returns
# =======
# result : X 和 Y 的协方差，一个常数

# Examples
# ========
# >>> from sympy.stats import Exponential, covariance
# >>> from sympy import Symbol
# >>> rate = Symbol('lambda', positive=True, real=True)
# >>> X = Exponential('X', rate)
# >>> Y = Exponential('Y', rate)
# >>> covariance(X, X)
# lambda**(-2)
# >>> covariance(X, Y)
# 0
# >>> covariance(X, Y + rate*X)
# 1/lambda

def covariance(X, Y, condition=None, **kwargs):
    if (is_random(X) and pspace(X) == PSpace()) or (is_random(Y) and pspace(Y) == PSpace()):
        from sympy.stats.symbolic_probability import Covariance
        return Covariance(X, Y, condition)  # 使用符号概率模块计算协方差

    return expectation(
        (X - expectation(X, condition, **kwargs)) *
        (Y - expectation(Y, condition, **kwargs)),  # 计算期望值
        condition, **kwargs)


# 计算两个随机表达式的相关性系数，也称为相关系数或皮尔逊相关系数。

# Parameters
# ==========
# X : 第一个随机表达式
# Y : 第二个随机表达式
# condition : 可选，对随机表达式的条件限制
# kwargs : 其他参数

# Returns
# =======
# result : X 和 Y 的相关性系数，一个常数

# Examples
# ========
# >>> from sympy.stats import Exponential, correlation
# >>> from sympy import Symbol
# >>> rate = Symbol('lambda', positive=True, real=True)
# >>> X = Exponential('X', rate)
# >>> Y = Exponential('Y', rate)
# >>> correlation(X, X)
# 1
# >>> correlation(X, Y)
# 0
# >>> correlation(X, Y + rate*X)
# 1/sqrt(1 + lambda**(-2))

def correlation(X, Y, condition=None, **kwargs):
    # 标准化的期望，表达两个变量一起升降的期望
    return expectation(
        (X - expectation(X, condition, **kwargs)) *
        (Y - expectation(Y, condition, **kwargs)) /
        (std(X, condition, **kwargs) * std(Y, condition, **kwargs)),  # 计算标准差
        condition, **kwargs)
    # 返回 X 和 Y 的条件协方差除以它们在给定条件下的标准差乘积的结果
    return covariance(X, Y, condition, **kwargs)/(std(X, condition, **kwargs)
     * std(Y, condition, **kwargs))
# 计算随机表达式关于其均值的第n个中心矩
def cmoment(X, n, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth central moment of a random expression about its mean.

    .. math::
        cmoment(X, n) = E((X - E(X))^{n})

    Examples
    ========

    >>> from sympy.stats import Die, cmoment, variance
    >>> X = Die('X', 6)
    >>> cmoment(X, 3)
    0
    >>> cmoment(X, 2)
    35/12
    >>> cmoment(X, 2) == variance(X)
    True
    """
    from sympy.stats.symbolic_probability import CentralMoment
    # 如果 evaluate 为 True，直接计算中心矩的值
    if evaluate:
        return CentralMoment(X, n, condition).doit()
    # 否则使用积分形式重写中心矩的计算
    return CentralMoment(X, n, condition).rewrite(Integral)


# 计算随机表达式的第n个标准化矩
def smoment(X, n, condition=None, **kwargs):
    r"""
    Return the nth Standardized moment of a random expression.

    .. math::
        smoment(X, n) = E(((X - \mu)/\sigma_X)^{n})

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, smoment
    >>> from sympy import Symbol
    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> smoment(Y, 4)
    9
    >>> smoment(Y, 4) == smoment(3*Y, 4)
    True
    >>> smoment(Y, 3) == skewness(Y)
    True
    """
    # 计算随机表达式的标准差
    sigma = std(X, condition, **kwargs)
    # 返回标准化矩的计算结果
    return (1/sigma)**n*cmoment(X, n, condition, **kwargs)


# 计算随机表达式的偏度
def skewness(X, condition=None, **kwargs):
    r"""
    Measure of the asymmetry of the probability distribution.

    Explanation
    ===========

    Positive skew indicates that most of the values lie to the right of
    the mean.

    .. math::
        skewness(X) = E(((X - E(X))/\sigma_X)^{3})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. skewness(X, X>0) is skewness of X given X > 0

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> skewness(X)
    0
    >>> skewness(X, X > 0) # find skewness given X > 0
    (-sqrt(2)/sqrt(pi) + 4*sqrt(2)/pi**(3/2))/(1 - 2/pi)**(3/2)

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> skewness(Y)
    2
    """
    # 返回随机表达式的第三个标准化矩，即偏度
    return smoment(X, 3, condition=condition, **kwargs)


# 计算随机表达式的峰度
def kurtosis(X, condition=None, **kwargs):
    r"""
    Characterizes the tails/outliers of a probability distribution.

    Explanation
    ===========

    Kurtosis of any univariate normal distribution is 3. Kurtosis less than
    3 means that the distribution produces fewer and less extreme outliers
    than the normal distribution.

    .. math::
        kurtosis(X) = E(((X - E(X))/\sigma_X)^{4})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. kurtosis(X, X>0) is kurtosis of X given X > 0

    Examples
    ========

    >>> from sympy.stats import kurtosis, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> kurtosis(X)
    3
    >>> kurtosis(X, X > 0) # find kurtosis given X > 0
    """
    # 返回随机表达式的第四个标准化矩，即峰度
    return smoment(X, 4, condition=condition, **kwargs)
    (-4/pi - 12/pi**2 + 3)/(1 - 2/pi)**2
    # 计算数学表达式 (-4/pi - 12/pi**2 + 3)/(1 - 2/pi)**2 的值

    >>> rate = Symbol('lamda', positive=True, real=True)
    # 创建一个符号变量 rate，其名称为 'lamda'，要求为正实数

    >>> Y = Exponential('Y', rate)
    # 创建一个指数分布的随机变量 Y，其参数为 rate

    >>> kurtosis(Y)
    # 计算随机变量 Y 的峰度（kurtosis），返回结果为 9

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kurtosis
    .. [2] https://mathworld.wolfram.com/Kurtosis.html
    """
    # 返回随机变量 X 的第四阶中心距，条件为 condition，以及可能的其他关键字参数
    return smoment(X, 4, condition=condition, **kwargs)
# 计算三个随机变量的共偏度（co-skewness）
def coskewness(X, Y, Z, condition=None, **kwargs):
    r"""
    Calculates the co-skewness of three random variables.

    Explanation
    ===========
    
    共偏度是三个随机变量的统计量，用于描述它们共同偏离其期望值的程度。

    Parameters
    ==========

    X, Y, Z: 随机变量，用于计算共偏度。

    condition : Expr containing RandomSymbols
        一个条件表达式，可选。

    Returns
    =======

    返回三个随机变量的共偏度值。

    Examples
    ========

    >>> from sympy.stats import coskewness, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> Y = Normal('Y', 0, 1)
    >>> Z = Normal('Z', 0, 1)
    >>> coskewness(X, Y, Z)
    0
    >>> coskewness(X, X, X)
    1

    References
    ==========

    参考文献待补充。
    
    """
    # 调用统计模块中的 co-skewness 函数计算共偏度
    return coskewness(X, Y, Z, condition=condition, **kwargs)
    # 计算三个随机变量 X, Y, Z 的协偏斜度（coskewness）：
    # coskewness 定义为：
    # coskewness(X,Y,Z) = E[(X-E[X]) * (Y-E[Y]) * (Z-E[Z])] / (σ_X * σ_Y * σ_Z)
    # 其中 E[] 表示期望，σ 表示标准差

    num = expectation((X - expectation(X, condition, **kwargs)) \
         * (Y - expectation(Y, condition, **kwargs)) \
         * (Z - expectation(Z, condition, **kwargs)), condition, **kwargs)
    # 计算协偏斜度的分子，使用随机变量 X, Y, Z 的条件期望

    den = std(X, condition, **kwargs) * std(Y, condition, **kwargs) \
         * std(Z, condition, **kwargs)
    # 计算协偏斜度的分母，使用随机变量 X, Y, Z 的条件标准差

    return num/den
    # 返回三个随机变量 X, Y, Z 的协偏斜度
# 将变量P定义为概率（Probability）
P = probability

# 将变量E定义为期望（Expectation）
E = expectation

# 将变量H定义为熵（Entropy）
H = entropy
```