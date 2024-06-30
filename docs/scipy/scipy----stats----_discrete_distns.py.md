# `D:\src\scipysrc\scipy\scipy\stats\_discrete_distns.py`

```
#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#

# 导入 functools 模块的 partial 函数
from functools import partial

# 导入 scipy 库中的 special 模块
from scipy import special
# 从 scipy.special 模块中导入指定的函数
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
# 从 scipy._lib._util 模块中导入指定的函数
from scipy._lib._util import _lazywhere, rng_integers
# 从 scipy.interpolate 模块中导入 interp1d 函数
from scipy.interpolate import interp1d

# 导入 numpy 库并将其命名为 np
import numpy as np

# 导入 _distn_infrastructure 模块中的指定内容
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
                                    _vectorize_rvs_over_shapes,
                                    _ShapeInfo, _isintegral)
# 导入 _biasedurn 模块中的指定内容
from ._biasedurn import (_PyFishersNCHypergeometric,
                         _PyWalleniusNCHypergeometric,
                         _PyStochasticLib3)
# 导入 scipy.special._ufuncs 模块并将其命名为 scu
import scipy.special._ufuncs as scu


# 定义一个名为 binom_gen 的类，继承自 rv_discrete 类
class binom_gen(rv_discrete):
    r"""A binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `binom` is:

    .. math::

       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`0 \leq p \leq 1`

    `binom` takes :math:`n` and :math:`p` as shape parameters,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, nbinom, nhypergeom

    """

    # 返回一个描述参数形状信息的列表
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("p", False, (0, 1), (True, True))]

    # 随机变量抽样方法，使用给定的参数生成服从二项分布的随机变量
    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.binomial(n, p, size)

    # 参数检查方法，检查参数 n 和 p 是否符合二项分布的要求
    def _argcheck(self, n, p):
        return (n >= 0) & _isintegral(n) & (p >= 0) & (p <= 1)

    # 获取支持范围的方法，返回二项分布的支持范围
    def _get_support(self, n, p):
        return self.a, n

    # 对数概率质量函数，计算二项分布的对数概率质量函数
    def _logpmf(self, x, n, p):
        k = floor(x)
        combiln = (gamln(n+1) - (gamln(k+1) + gamln(n-k+1)))
        return combiln + special.xlogy(k, p) + special.xlog1py(n-k, -p)

    # 概率质量函数，计算二项分布的概率质量函数
    def _pmf(self, x, n, p):
        # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
        return scu._binom_pmf(x, n, p)

    # 累积分布函数，计算二项分布的累积分布函数
    def _cdf(self, x, n, p):
        k = floor(x)
        return scu._binom_cdf(k, n, p)

    # 生存函数，计算二项分布的生存函数
    def _sf(self, x, n, p):
        k = floor(x)
        return scu._binom_sf(k, n, p)

    # 逆累积分布函数，计算二项分布的逆累积分布函数
    def _isf(self, x, n, p):
        return scu._binom_isf(x, n, p)

    # 百分位点函数，计算二项分布的百分位点函数
    def _ppf(self, q, n, p):
        return scu._binom_ppf(q, n, p)

    # 统计量计算方法，计算二项分布的均值、方差及其它统计量
    def _stats(self, n, p, moments='mv'):
        mu = n * p
        var = mu - n * np.square(p)
        g1, g2 = None, None
        if 's' in moments:
            pq = p - np.square(p)
            npq_sqrt = np.sqrt(n * pq)
            t1 = np.reciprocal(npq_sqrt)
            t2 = (2.0 * p) / npq_sqrt
            g1 = t1 - t2
        if 'k' in moments:
            pq = p - np.square(p)
            npq = n * pq
            t1 = np.reciprocal(npq)
            t2 = 6.0/n
            g2 = t1 - t2
        return mu, var, g1, g2
    # 计算熵的私有方法，接受参数 n 和 p
    def _entropy(self, n, p):
        # 生成一个从 0 到 n 的整数数组，包括 n
        k = np.r_[0:n + 1]
        # 使用 _pmf 方法计算离散随机变量的概率质量函数，并得到概率值数组
        vals = self._pmf(k, n, p)
        # 计算概率分布 vals 的熵，并沿着第一个轴（行）求和，得到总熵
        return np.sum(entr(vals), axis=0)
# 创建一个名为 binom 的随机变量生成器对象，使用默认参数 'binom'
binom = binom_gen(name='binom')


class bernoulli_gen(binom_gen):
    r"""A Bernoulli discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                           p    &\text{if } k = 1\end{cases}

    for :math:`k` in :math:`\{0, 1\}`, :math:`0 \leq p \leq 1`

    `bernoulli` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """
    
    # 定义一个方法，返回 `_ShapeInfo` 对象列表，描述参数形状信息
    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    # 生成服从 Bernoulli 分布的随机变量
    def _rvs(self, p, size=None, random_state=None):
        return binom_gen._rvs(self, 1, p, size=size, random_state=random_state)

    # 检查参数是否符合 Bernoulli 分布的条件，返回布尔值
    def _argcheck(self, p):
        return (p >= 0) & (p <= 1)

    # 获取分布的支持范围，重写了 binom_gen._get_support 方法
    def _get_support(self, p):
        return self.a, self.b

    # 计算对数概率质量函数，调用 binom_gen._logpmf 方法
    def _logpmf(self, x, p):
        return binom._logpmf(x, 1, p)

    # 计算概率质量函数，调用 binom_gen._pmf 方法
    def _pmf(self, x, p):
        # bernoulli.pmf(k) = 1-p  if k = 0
        #                  = p    if k = 1
        return binom._pmf(x, 1, p)

    # 计算累积分布函数，调用 binom_gen._cdf 方法
    def _cdf(self, x, p):
        return binom._cdf(x, 1, p)

    # 计算生存函数 (1 - CDF)，调用 binom_gen._sf 方法
    def _sf(self, x, p):
        return binom._sf(x, 1, p)

    # 计算逆累积分布函数，调用 binom_gen._isf 方法
    def _isf(self, x, p):
        return binom._isf(x, 1, p)

    # 计算累积分布函数的逆函数 (CDF 的逆函数)，调用 binom_gen._ppf 方法
    def _ppf(self, q, p):
        return binom._ppf(q, 1, p)

    # 计算分布的统计信息，调用 binom_gen._stats 方法
    def _stats(self, p):
        return binom._stats(1, p)

    # 计算分布的熵，使用 entr 函数计算信息熵
    def _entropy(self, p):
        return entr(p) + entr(1-p)


# 创建一个名为 bernoulli 的 Bernoulli 分布的随机变量生成器对象，使用参数 b=1 和名称 'bernoulli'
bernoulli = bernoulli_gen(b=1, name='bernoulli')


class betabinom_gen(rv_discrete):
    r"""A beta-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-binomial distribution is a binomial distribution with a
    probability of success `p` that follows a beta distribution.

    The probability mass function for `betabinom` is:

    .. math::

       f(k) = \binom{n}{k} \frac{B(k + a, n - k + b)}{B(a, b)}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.4.0

    See Also
    --------
    beta, binom

    %(example)s

    """
    
    # 定义一个方法，返回 `_ShapeInfo` 对象列表，描述参数形状信息
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("a", False, (0, np.inf), (False, False)),
                _ShapeInfo("b", False, (0, np.inf), (False, False))]

    # 生成服从 beta-binomial 分布的随机变量
    def _rvs(self, n, a, b, size=None, random_state=None):
        # 使用随机数生成器生成 beta 分布的参数 p
        p = random_state.beta(a, b, size)
        # 使用生成的 p 参数生成二项分布的随机变量
        return random_state.binomial(n, p, size)

    # 获取分布的支持范围，返回 (0, n)
    def _get_support(self, n, a, b):
        return 0, n
    # 检查参数是否合法，返回布尔值
    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    # 计算负二项分布的对数概率质量函数（log PMF）
    def _logpmf(self, x, n, a, b):
        # 取下界作为整数部分
        k = floor(x)
        # 计算组合数的对数
        combiln = -log(n + 1) - betaln(n - k + 1, k + 1)
        # 返回负二项分布的对数概率质量函数值
        return combiln + betaln(k + a, n - k + b) - betaln(a, b)

    # 计算负二项分布的概率质量函数（PMF）
    def _pmf(self, x, n, a, b):
        # 返回负二项分布的概率质量函数值的指数
        return exp(self._logpmf(x, n, a, b))

    # 计算负二项分布的期望值和方差，以及可能的偏斜度和峰度
    def _stats(self, n, a, b, moments='mv'):
        # 计算期望值
        e_p = a / (a + b)
        # 计算另一个期望值
        e_q = 1 - e_p
        # 计算均值
        mu = n * e_p
        # 计算方差
        var = n * (a + b + n) * e_p * e_q / (a + b + 1)
        g1, g2 = None, None
        if 's' in moments:
            # 计算偏斜度
            g1 = 1.0 / sqrt(var)
            g1 *= (a + b + 2 * n) * (b - a)
            g1 /= (a + b + 2) * (a + b)
        if 'k' in moments:
            # 计算峰度
            g2 = (a + b).astype(e_p.dtype)
            g2 *= (a + b - 1 + 6 * n)
            g2 += 3 * a * b * (n - 2)
            g2 += 6 * n ** 2
            g2 -= 3 * e_p * b * n * (6 - n)
            g2 -= 18 * e_p * e_q * n ** 2
            g2 *= (a + b) ** 2 * (1 + a + b)
            g2 /= (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n))
            g2 -= 3
        # 返回均值、方差、偏斜度和峰度
        return mu, var, g1, g2
# 创建一个名为 `betabinom` 的负二项分布生成器实例
betabinom = betabinom_gen(name='betabinom')

# 定义一个自定义的负二项分布生成器类 `nbinom_gen`，继承自 `rv_discrete`
class nbinom_gen(rv_discrete):
    r"""A negative binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is:

    .. math::

       f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k

    for :math:`k \ge 0`, :math:`0 < p \leq 1`

    `nbinom` takes :math:`n` and :math:`p` as shape parameters where :math:`n`
    is the number of successes, :math:`p` is the probability of a single
    success, and :math:`1-p` is the probability of a single failure.

    Another common parameterization of the negative binomial distribution is
    in terms of the mean number of failures :math:`\mu` to achieve :math:`n`
    successes. The mean :math:`\mu` is related to the probability of success
    as

    .. math::

       p = \frac{n}{n + \mu}

    The number of successes :math:`n` may also be specified in terms of a
    "dispersion", "heterogeneity", or "aggregation" parameter :math:`\alpha`,
    which relates the mean :math:`\mu` to the variance :math:`\sigma^2`,
    e.g. :math:`\sigma^2 = \mu + \alpha \mu^2`. Regardless of the convention
    used for :math:`\alpha`,

    .. math::

       p &= \frac{\mu}{\sigma^2} \\
       n &= \frac{\mu^2}{\sigma^2 - \mu}

    %(after_notes)s

    %(example)s

    See Also
    --------
    hypergeom, binom, nhypergeom

    """

    # 返回一个包含参数 `n` 和 `p` 的列表，这些参数用于定义负二项分布的形状信息
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("p", False, (0, 1), (True, True))]

    # 返回负二项分布随机变量的随机变量样本
    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.negative_binomial(n, p, size)

    # 检查参数 `n` 和 `p` 是否有效，满足负二项分布的条件
    def _argcheck(self, n, p):
        return (n > 0) & (p > 0) & (p <= 1)

    # 返回负二项分布的概率质量函数（PMF）对于给定的参数 `x`, `n`, `p`
    def _pmf(self, x, n, p):
        # nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
        return scu._nbinom_pmf(x, n, p)

    # 返回负二项分布的对数概率质量函数（log PMF）对于给定的参数 `x`, `n`, `p`
    def _logpmf(self, x, n, p):
        coeff = gamln(n+x) - gamln(x+1) - gamln(n)
        return coeff + n*log(p) + special.xlog1py(x, -p)

    # 返回负二项分布的累积分布函数（CDF）对于给定的参数 `x`, `n`, `p`
    def _cdf(self, x, n, p):
        k = floor(x)
        return scu._nbinom_cdf(k, n, p)

    # 返回负二项分布的对数累积分布函数（log CDF）对于给定的参数 `x`, `n`, `p`
    def _logcdf(self, x, n, p):
        k = floor(x)
        k, n, p = np.broadcast_arrays(k, n, p)
        cdf = self._cdf(k, n, p)
        cond = cdf > 0.5
        def f1(k, n, p):
            return np.log1p(-special.betainc(k + 1, n, 1 - p))

        # 在原地进行计算
        logcdf = cdf
        with np.errstate(divide='ignore'):
            logcdf[cond] = f1(k[cond], n[cond], p[cond])
            logcdf[~cond] = np.log(cdf[~cond])
        return logcdf

    # 返回负二项分布的生存函数（Survival Function，SF）对于给定的参数 `x`, `n`, `p`
    def _sf(self, x, n, p):
        k = floor(x)
        return scu._nbinom_sf(k, n, p)

    # 返回负二项分布的逆生存函数（Inverse Survival Function，ISF）对于给定的参数 `x`, `n`, `p`
    def _isf(self, x, n, p):
        with np.errstate(over='ignore'):  # see gh-17432
            return scu._nbinom_isf(x, n, p)
    # 定义一个方法 `_ppf`，用于计算负二项分布的分位数
    def _ppf(self, q, n, p):
        # 设置 numpy 的错误状态，忽略溢出错误，参见 GitHub issue #17432
        with np.errstate(over='ignore'):  
            # 调用 scipy.stats 中的负二项分布分位数计算函数
            return scu._nbinom_ppf(q, n, p)

    # 定义一个方法 `_stats`，用于计算负二项分布的统计量
    def _stats(self, n, p):
        # 返回负二项分布的均值、方差、偏度和峰度（超额峰度）
        return (
            scu._nbinom_mean(n, p),  # 计算负二项分布的均值
            scu._nbinom_variance(n, p),  # 计算负二项分布的方差
            scu._nbinom_skewness(n, p),  # 计算负二项分布的偏度
            scu._nbinom_kurtosis_excess(n, p),  # 计算负二项分布的峰度（超额峰度）
        )
# 导入 nbinom_gen 函数并创建 betanbinom_gen 类，继承自 rv_discrete 类
nbinom = nbinom_gen(name='nbinom')

# 定义一个名为 betanbinom_gen 的新类，表示 beta-negative-binomial 分布
class betanbinom_gen(rv_discrete):
    r"""A beta-negative-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-negative-binomial distribution is a negative binomial
    distribution with a probability of success `p` that follows a
    beta distribution.

    The probability mass function for `betanbinom` is:

    .. math::

       f(k) = \binom{n + k - 1}{k} \frac{B(a + n, b + k)}{B(a, b)}

    for :math:`k \ge 0`, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betanbinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.12.0

    See Also
    --------
    betabinom : Beta binomial distribution

    %(example)s

    """
    
    # 定义 _shape_info 方法，返回参数的形状信息列表
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("a", False, (0, np.inf), (False, False)),
                _ShapeInfo("b", False, (0, np.inf), (False, False))]

    # 定义 _rvs 方法，生成随机变量，返回 beta-negative-binomial 分布的样本
    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)  # 生成 beta 分布的概率参数 p
        return random_state.negative_binomial(n, p, size)  # 生成负二项分布的随机变量

    # 定义 _argcheck 方法，检查参数是否有效
    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    # 定义 _logpmf 方法，计算对数概率质量函数
    def _logpmf(self, x, n, a, b):
        k = floor(x)  # 取 x 的下限整数
        combiln = -np.log(n + k) - betaln(n, k + 1)  # 计算组合数的对数和 beta 函数的对数
        return combiln + betaln(a + n, b + k) - betaln(a, b)  # 返回对数概率质量函数的值

    # 定义 _pmf 方法，计算概率质量函数
    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))  # 返回概率质量函数的值
    # 定义一个方法 `_stats`，计算贝塔负二项分布的统计量
    def _stats(self, n, a, b, moments='mv'):
        # 参考来源：Wolfram Alpha 输入
        # BetaNegativeBinomialDistribution[a, b, n]

        # 定义计算均值的函数
        def mean(n, a, b):
            return n * b / (a - 1.)

        # 根据条件计算均值 mu
        mu = _lazywhere(a > 1, (n, a, b), f=mean, fillvalue=np.inf)

        # 定义计算方差的函数
        def var(n, a, b):
            return (n * b * (n + a - 1.) * (a + b - 1.)
                    / ((a - 2.) * (a - 1.)**2.))

        # 根据条件计算方差 var
        var = _lazywhere(a > 2, (n, a, b), f=var, fillvalue=np.inf)

        # 初始化 g1 和 g2
        g1, g2 = None, None

        # 定义计算偏度的函数
        def skew(n, a, b):
            return ((2 * n + a - 1.) * (2 * b + a - 1.)
                    / (a - 3.) / sqrt(n * b * (n + a - 1.) * (b + a - 1.)
                    / (a - 2.)))

        # 如果需要计算偏度 g1，则根据条件进行计算
        if 's' in moments:
            g1 = _lazywhere(a > 3, (n, a, b), f=skew, fillvalue=np.inf)

        # 定义计算峰度的函数
        def kurtosis(n, a, b):
            term = (a - 2.)
            term_2 = ((a - 1.)**2. * (a**2. + a * (6 * b - 1.)
                      + 6. * (b - 1.) * b)
                      + 3. * n**2. * ((a + 5.) * b**2. + (a + 5.)
                      * (a - 1.) * b + 2. * (a - 1.)**2)
                      + 3 * (a - 1.) * n
                      * ((a + 5.) * b**2. + (a + 5.) * (a - 1.) * b
                      + 2. * (a - 1.)**2.))
            denominator = ((a - 4.) * (a - 3.) * b * n
                           * (a + b - 1.) * (a + n - 1.))
            # Wolfram Alpha 使用 Pearson 峰度，因此减去 3 得到 scipy 的 Fisher 峰度
            return term * term_2 / denominator - 3.

        # 如果需要计算峰度 g2，则根据条件进行计算
        if 'k' in moments:
            g2 = _lazywhere(a > 4, (n, a, b), f=kurtosis, fillvalue=np.inf)

        # 返回均值、方差、偏度和峰度
        return mu, var, g1, g2
betanbinom = betanbinom_gen(name='betanbinom')

# 定义一个几何分布的随机变量类，继承自离散随机变量的基类 rv_discrete
class geom_gen(rv_discrete):
    r"""A geometric discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

    for :math:`k \ge 1`, :math:`0 < p \leq 1`

    `geom` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    See Also
    --------
    planck

    %(example)s

    """

    # 返回形状参数信息
    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    # 生成随机变量
    def _rvs(self, p, size=None, random_state=None):
        return random_state.geometric(p, size=size)

    # 参数检查函数，返回参数是否有效
    def _argcheck(self, p):
        return (p <= 1) & (p > 0)

    # 概率质量函数
    def _pmf(self, k, p):
        return np.power(1-p, k-1) * p

    # 对数概率质量函数
    def _logpmf(self, k, p):
        return special.xlog1py(k - 1, -p) + log(p)

    # 累积分布函数
    def _cdf(self, x, p):
        k = floor(x)
        return -expm1(log1p(-p)*k)

    # 生存函数
    def _sf(self, x, p):
        return np.exp(self._logsf(x, p))

    # 对数生存函数
    def _logsf(self, x, p):
        k = floor(x)
        return k*log1p(-p)

    # 百分位点函数
    def _ppf(self, q, p):
        vals = ceil(log1p(-q) / log1p(-p))
        temp = self._cdf(vals-1, p)
        return np.where((temp >= q) & (vals > 0), vals-1, vals)

    # 统计特征函数
    def _stats(self, p):
        mu = 1.0/p
        qr = 1.0-p
        var = qr / p / p
        g1 = (2.0-p) / sqrt(qr)
        g2 = np.polyval([1, -6, 6], p)/(1.0-p)
        return mu, var, g1, g2

    # 熵函数
    def _entropy(self, p):
        return -np.log(p) - np.log1p(-p) * (1.0-p) / p


# 创建一个几何分布的随机变量对象
geom = geom_gen(a=1, name='geom', longname="A geometric")

# 定义一个超几何分布的随机变量类，继承自离散随机变量的基类 rv_discrete
class hypergeom_gen(rv_discrete):
    r"""A hypergeometric discrete random variable.

    The hypergeometric distribution models drawing objects from a bin.
    `M` is the total number of objects, `n` is total number of Type I objects.
    The random variate represents the number of Type I objects in `N` drawn
    without replacement from the total population.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
    universally accepted.  See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: p(k, M, n, N) = \frac{\binom{n}{k} \binom{M - n}{N - k}}
                                   {\binom{M}{N}}

    for :math:`k \in [\max(0, N - M + n), \min(n, N)]`, where the binomial
    coefficients are defined as,

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import hypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if
    we want to know the probability of finding a given number of dogs if we
    # 定义一个私有方法 `_shape_info`，返回一个包含三个 `_ShapeInfo` 对象的列表，
    # 每个对象表示一个参数：M, n, N，这些参数都是必需的，且取值范围是 [0, 无穷)
    # M 和 N 的类型是整数，而 n 可以是整数也可以是非整数
    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("N", True, (0, np.inf), (True, False))]

    # 定义一个私有方法 `_rvs`，根据给定的参数 M, n, N 和 size，在给定的随机状态下生成超几何分布的随机数
    def _rvs(self, M, n, N, size=None, random_state=None):
        return random_state.hypergeometric(n, M-n, N, size=size)

    # 定义一个私有方法 `_get_support`，返回超几何分布的支持区间下界和上界
    def _get_support(self, M, n, N):
        return np.maximum(N-(M-n), 0), np.minimum(n, N)

    # 定义一个私有方法 `_argcheck`，检查参数 M, n, N 是否符合超几何分布的参数要求
    # 要求 M 大于 0，n 和 N 大于等于 0，n 和 N 小于等于 M，同时这些参数都是整数类型
    def _argcheck(self, M, n, N):
        cond = (M > 0) & (n >= 0) & (N >= 0)
        cond &= (n <= M) & (N <= M)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(N)
        return cond

    # 定义一个私有方法 `_logpmf`，计算超几何分布在给定参数 M, n, N 和 k 下的对数概率质量函数值
    def _logpmf(self, k, M, n, N):
        tot, good = M, n
        bad = tot - good
        # 计算超几何分布的对数概率质量函数值
        result = (betaln(good+1, 1) + betaln(bad+1, 1) + betaln(tot-N+1, N+1) -
                  betaln(k+1, good-k+1) - betaln(N-k+1, bad-N+k+1) -
                  betaln(tot+1, 1))
        return result

    # 定义一个私有方法 `_pmf`，使用 SciPy 的 `_hypergeom_pmf` 函数计算超几何分布的概率质量函数值
    def _pmf(self, k, M, n, N):
        return scu._hypergeom_pmf(k, n, N, M)

    # 定义一个私有方法 `_cdf`，使用 SciPy 的 `_hypergeom_cdf` 函数计算超几何分布的累积分布函数值
    def _cdf(self, k, M, n, N):
        return scu._hypergeom_cdf(k, n, N, M)

    # 定义一个私有方法 `_stats`，计算超几何分布的统计特性：均值、方差、偏度和峰度
    def _stats(self, M, n, N):
        M, n, N = 1. * M, 1. * n, 1. * N
        m = M - n

        # 计算超几何分布的偏度和峰度
        g2 = M * (M + 1) - 6. * N * (M - N) - 6. * n * m
        g2 *= (M - 1) * M * M
        g2 += 6. * n * N * (M - N) * m * (5. * M - 6)
        g2 /= n * N * (M - N) * m * (M - 2.) * (M - 3.)
        return (
            scu._hypergeom_mean(n, N, M),
            scu._hypergeom_variance(n, N, M),
            scu._hypergeom_skewness(n, N, M),
            g2,
        )

    # 定义一个私有方法 `_entropy`，计算超几何分布的信息熵
    def _entropy(self, M, n, N):
        k = np.r_[N - (M - n):min(n, N) + 1]
        vals = self.pmf(k, M, n, N)
        return np.sum(entr(vals), axis=0)

    # 定义一个私有方法 `_sf`，使用 SciPy 的 `_hypergeom_sf` 函数计算超几何分布的生存函数值
    def _sf(self, k, M, n, N):
        return scu._hypergeom_sf(k, n, N, M)
    # 计算对数生存函数（logsf），接受四个参数：k, M, n, N
    def _logsf(self, k, M, n, N):
        # 初始化结果列表
        res = []
        # 使用 np.broadcast_arrays 广播输入参数，使其具有相同的形状
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            # 判断是否满足优化条件：如果计算 log(1-cdf) 可以少计算一些项
            if (quant + 0.5) * (tot + 0.5) < (good - 0.5) * (draw - 0.5):
                # 将 -exp(self.logcdf(quant, tot, good, draw)) 的对数值加入结果列表
                res.append(log1p(-exp(self.logcdf(quant, tot, good, draw))))
            else:
                # 如果不满足优化条件，则使用 logsumexp 对概率质量函数进行积分
                k2 = np.arange(quant + 1, draw + 1)
                # 将 logsumexp(self._logpmf(k2, tot, good, draw)) 的结果加入结果列表
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        # 将结果列表转换为 numpy 数组并返回
        return np.asarray(res)

    # 计算对数累积分布函数（logcdf），接受四个参数：k, M, n, N
    def _logcdf(self, k, M, n, N):
        # 初始化结果列表
        res = []
        # 使用 np.broadcast_arrays 广播输入参数，使其具有相同的形状
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            # 判断是否满足优化条件：如果计算 log(1-sf) 可以少计算一些项
            if (quant + 0.5) * (tot + 0.5) > (good - 0.5) * (draw - 0.5):
                # 将 -exp(self.logsf(quant, tot, good, draw)) 的对数值加入结果列表
                res.append(log1p(-exp(self.logsf(quant, tot, good, draw))))
            else:
                # 如果不满足优化条件，则使用 logsumexp 对概率质量函数进行积分
                k2 = np.arange(0, quant + 1)
                # 将 logsumexp(self._logpmf(k2, tot, good, draw)) 的结果加入结果列表
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        # 将结果列表转换为 numpy 数组并返回
        return np.asarray(res)
# 使用 `hypergeom_gen` 函数创建一个负超几何分布的生成器
hypergeom = hypergeom_gen(name='hypergeom')

# 定义一个负超几何分布的随机变量类，继承自 `rv_discrete`
class nhypergeom_gen(rv_discrete):
    r"""A negative hypergeometric discrete random variable.

    Consider a box containing :math:`M` balls:, :math:`n` red and
    :math:`M-n` blue. We randomly sample balls from the box, one
    at a time and *without* replacement, until we have picked :math:`r`
    blue balls. `nhypergeom` is the distribution of the number of
    red balls :math:`k` we have picked.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `r`) are not
    universally accepted. See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: f(k; M, n, r) = \frac{{{k+r-1}\choose{k}}{{M-r-k}\choose{n-k}}}
                                   {{M \choose n}}

    for :math:`k \in [0, n]`, :math:`n \in [0, M]`, :math:`r \in [0, M-n]`,
    and the binomial coefficient is:

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    It is equivalent to observing :math:`k` successes in :math:`k+r-1`
    samples with :math:`k+r`'th sample being a failure. The former
    can be modelled as a hypergeometric distribution. The probability
    of the latter is simply the number of failures remaining
    :math:`M-n-(r-1)` divided by the size of the remaining population
    :math:`M-(k+r-1)`. This relationship can be shown as:

    .. math:: NHG(k;M,n,r) = HG(k;M,n,k+r-1)\frac{(M-n-(r-1))}{(M-(k+r-1))}

    where :math:`NHG` is probability mass function (PMF) of the
    negative hypergeometric distribution and :math:`HG` is the
    PMF of the hypergeometric distribution.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import nhypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.
    Then if we want to know the probability of finding a given number
    of dogs (successes) in a sample with exactly 12 animals that
    aren't dogs (failures), we can initialize a frozen distribution
    and plot the probability mass function:

    >>> M, n, r = [20, 7, 12]
    >>> rv = nhypergeom(M, n, r)
    >>> x = np.arange(0, n+2)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group with given 12 failures')
    >>> ax.set_ylabel('nhypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `nhypergeom`
    methods directly.  To for example obtain the probability mass
    function, use:

    >>> prb = nhypergeom.pmf(x, M, n, r)

    And to generate random numbers:

    >>> R = nhypergeom.rvs(M, n, r, size=10)

    To verify the relationship between `hypergeom` and `nhypergeom`, use:

    >>> from scipy.stats import hypergeom, nhypergeom
    >>> M, n, r = 45, 13, 8
    >>> k = 6
    >>> nhypergeom.pmf(k, M, n, r)
    0.06180776620271643
    >>> hypergeom.pmf(k, M, n, k+r-1) * (M - n - (r-1)) / (M - (k+r-1))
    0.06180776620271644

    See Also
    --------
    hypergeom, binom, nbinom

    References
    ----------
    .. [1] Negative Hypergeometric Distribution on Wikipedia
           https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution

    .. [2] Negative Hypergeometric Distribution from
           http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Negativehypergeometric.pdf

    """

    # 定义一个内部方法，返回参数的描述信息列表
    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("r", True, (0, np.inf), (True, False))]

    # 定义一个内部方法，返回支持参数的范围
    def _get_support(self, M, n, r):
        return 0, n

    # 定义一个内部方法，检查参数是否符合分布的条件
    def _argcheck(self, M, n, r):
        cond = (n >= 0) & (n <= M) & (r >= 0) & (r <= M-n)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(r)
        return cond

    # 定义一个内部方法，生成符合负超几何分布的随机变量
    def _rvs(self, M, n, r, size=None, random_state=None):

        # 内部装饰函数，用于生成随机变量
        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, r, size, random_state):
            # 根据逆累积分布函数生成随机变量
            a, b = self.support(M, n, r)
            ks = np.arange(a, b+1)
            cdf = self.cdf(ks, M, n, r)
            ppf = interp1d(cdf, ks, kind='next', fill_value='extrapolate')
            rvs = ppf(random_state.uniform(size=size)).astype(int)
            if size is None:
                return rvs.item()
            return rvs

        return _rvs1(M, n, r, size=size, random_state=random_state)

    # 定义一个内部方法，计算对数概率质量函数
    def _logpmf(self, k, M, n, r):
        cond = ((r == 0) & (k == 0))
        # 根据条件选择计算方式，避免数值不稳定性
        result = _lazywhere(~cond, (k, M, n, r),
                            lambda k, M, n, r:
                                (-betaln(k+1, r) + betaln(k+r, 1) -
                                 betaln(n-k+1, M-r-n+1) + betaln(M-r-k+1, 1) +
                                 betaln(n+1, M-n+1) - betaln(M+1, 1)),
                            fillvalue=0.0)
        return result

    # 定义一个内部方法，计算概率质量函数
    def _pmf(self, k, M, n, r):
        # 通过计算对数概率质量函数的指数得到概率质量函数
        return exp(self._logpmf(k, M, n, r))

    # 定义一个内部方法，计算分布的统计特性
    def _stats(self, M, n, r):
        # 提升数据类型至至少 float
        M, n, r = 1.*M, 1.*n, 1.*r
        # 计算均值和方差
        mu = r*n / (M-n+1)
        var = r*(M+1)*n / ((M-n+1)*(M-n+2)) * (1 - r / (M-n+1))

        # 偏度和峰度在数学上难以处理，返回 `None` 参考 [2]_
        g1, g2 = None, None
        return mu, var, g1, g2
# 创建名为 nhypergeom 的离散随机变量生成器
nhypergeom = nhypergeom_gen(name='nhypergeom')


# FIXME: Fails _cdfvec
class logser_gen(rv_discrete):
    r"""A Logarithmic (Log-Series, Series) discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `logser` is:

    .. math::

        f(k) = - \frac{p^k}{k \log(1-p)}

    for :math:`k \ge 1`, :math:`0 < p < 1`

    `logser` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """

    # 定义 Logarithmic 分布的形状信息方法
    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    # 生成 Logarithmic 分布的随机变量方法
    def _rvs(self, p, size=None, random_state=None):
        # 当 p>0.5 时，k=1 的情况看起来不对
        # 尝试使用通用方法效果更差，根本没有 k=1
        return random_state.logseries(p, size=size)

    # 检查参数 p 的有效性方法
    def _argcheck(self, p):
        return (p > 0) & (p < 1)

    # Logarithmic 分布的概率质量函数方法
    def _pmf(self, k, p):
        # logser.pmf(k) = - p**k / (k*log(1-p))
        return -np.power(p, k) * 1.0 / k / special.log1p(-p)

    # Logarithmic 分布的统计特性方法
    def _stats(self, p):
        r = special.log1p(-p)
        mu = p / (p - 1.0) / r
        mu2p = -p / r / (p - 1.0)**2
        var = mu2p - mu*mu
        mu3p = -p / r * (1.0+p) / (1.0 - p)**3
        mu3 = mu3p - 3*mu*mu2p + 2*mu**3
        g1 = mu3 / np.power(var, 1.5)

        mu4p = -p / r * (
            1.0 / (p-1)**2 - 6*p / (p - 1)**3 + 6*p*p / (p-1)**4)
        mu4 = mu4p - 4*mu3p*mu + 6*mu2p*mu*mu - 3*mu**4
        g2 = mu4 / var**2 - 3.0
        return mu, var, g1, g2


# 创建名为 logser 的 Logarithmic 分布的离散随机变量生成器
logser = logser_gen(a=1, name='logser', longname='A logarithmic')


class poisson_gen(rv_discrete):
    r"""A Poisson discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    `poisson` takes :math:`\mu \geq 0` as shape parameter.
    When :math:`\mu = 0`, the ``pmf`` method
    returns ``1.0`` at quantile :math:`k = 0`.

    %(after_notes)s

    %(example)s

    """

    # 定义 Poisson 分布的形状信息方法
    def _shape_info(self):
        return [_ShapeInfo("mu", False, (0, np.inf), (True, False))]

    # 覆盖 rv_discrete._argcheck 方法以允许 mu=0
    def _argcheck(self, mu):
        return mu >= 0

    # 生成 Poisson 分布的随机变量方法
    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    # Poisson 分布的对数概率质量函数方法
    def _logpmf(self, k, mu):
        Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
        return Pk

    # Poisson 分布的概率质量函数方法
    def _pmf(self, k, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return exp(self._logpmf(k, mu))

    # Poisson 分布的累积分布函数方法
    def _cdf(self, x, mu):
        k = floor(x)
        return special.pdtr(k, mu)

    # Poisson 分布的生存函数方法
    def _sf(self, x, mu):
        k = floor(x)
        return special.pdtrc(k, mu)

    # Poisson 分布的分位数函数方法
    def _ppf(self, q, mu):
        vals = ceil(special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)
    # 定义一个方法 `_stats`，接受参数 `mu`
    def _stats(self, mu):
        # 变量 `var` 被赋值为 `mu`，这里可能是一个误解或者需要进一步解释的地方
        var = mu
        # 将 `mu` 转换为 NumPy 数组 `tmp`
        tmp = np.asarray(mu)
        # 创建一个布尔数组 `mu_nonzero`，其元素是 `tmp` 中大于 0 的元素的布尔值
        mu_nonzero = tmp > 0
        # 使用 `_lazywhere` 函数，根据 `mu_nonzero` 的条件，传递 `tmp` 给 lambda 函数 `sqrt(1.0/x)`，返回结果赋值给 `g1`
        g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: sqrt(1.0/x), np.inf)
        # 使用 `_lazywhere` 函数，根据 `mu_nonzero` 的条件，传递 `tmp` 给 lambda 函数 `1.0/x`，返回结果赋值给 `g2`
        g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0/x, np.inf)
        # 返回包含 `mu`, `var`, `g1`, `g2` 的元组
        return mu, var, g1, g2
# 创建一个名为 poisson 的泊松分布随机变量生成器，设置其名称和长名称
poisson = poisson_gen(name="poisson", longname='A Poisson')


class planck_gen(rv_discrete):
    r"""A Planck discrete exponential random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `planck` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k)

    for :math:`k \ge 0` and :math:`\lambda > 0`.

    `planck` takes :math:`\lambda` as shape parameter. The Planck distribution
    can be written as a geometric distribution (`geom`) with
    :math:`p = 1 - \exp(-\lambda)` shifted by ``loc = -1``.

    %(after_notes)s

    See Also
    --------
    geom

    %(example)s

    """
    
    # 定义私有方法 _shape_info，返回 lambda 参数的形状信息
    def _shape_info(self):
        return [_ShapeInfo("lambda", False, (0, np.inf), (False, False))]

    # 定义私有方法 _argcheck，检查 lambda 参数是否大于 0
    def _argcheck(self, lambda_):
        return lambda_ > 0

    # 定义私有方法 _pmf，计算概率质量函数，对应 Planck 分布的概率质量函数
    def _pmf(self, k, lambda_):
        return -expm1(-lambda_)*exp(-lambda_*k)

    # 定义私有方法 _cdf，计算累积分布函数，对应 Planck 分布的累积分布函数
    def _cdf(self, x, lambda_):
        k = floor(x)
        return -expm1(-lambda_*(k+1))

    # 定义私有方法 _sf，计算生存函数，对应 Planck 分布的生存函数
    def _sf(self, x, lambda_):
        return exp(self._logsf(x, lambda_))

    # 定义私有方法 _logsf，计算生存函数的对数，对应 Planck 分布的生存函数的对数
    def _logsf(self, x, lambda_):
        k = floor(x)
        return -lambda_*(k+1)

    # 定义私有方法 _ppf，计算百分位点函数，对应 Planck 分布的百分位点函数
    def _ppf(self, q, lambda_):
        vals = ceil(-1.0/lambda_ * log1p(-q)-1)
        vals1 = (vals-1).clip(*(self._get_support(lambda_)))
        temp = self._cdf(vals1, lambda_)
        return np.where(temp >= q, vals1, vals)

    # 定义私有方法 _rvs，生成随机变量样本，利用与几何分布的关系进行采样
    def _rvs(self, lambda_, size=None, random_state=None):
        p = -expm1(-lambda_)
        return random_state.geometric(p, size=size) - 1.0

    # 定义私有方法 _stats，计算统计量，对应 Planck 分布的均值、方差、偏度和峰度
    def _stats(self, lambda_):
        mu = 1/expm1(lambda_)
        var = exp(-lambda_)/(expm1(-lambda_))**2
        g1 = 2*cosh(lambda_/2.0)
        g2 = 4+2*cosh(lambda_)
        return mu, var, g1, g2

    # 定义私有方法 _entropy，计算熵，对应 Planck 分布的熵
    def _entropy(self, lambda_):
        C = -expm1(-lambda_)
        return lambda_*exp(-lambda_)/C - log(C)


# 创建一个名为 planck 的 Planck 分布随机变量生成器，设置其名称和长名称
planck = planck_gen(a=0, name='planck', longname='A discrete exponential ')


class boltzmann_gen(rv_discrete):
    r"""A Boltzmann (Truncated Discrete Exponential) random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `boltzmann` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k) / (1-\exp(-\lambda N))

    for :math:`k = 0,..., N-1`.

    `boltzmann` takes :math:`\lambda > 0` and :math:`N > 0` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    
    # 定义私有方法 _shape_info，返回 lambda_ 和 N 参数的形状信息
    def _shape_info(self):
        return [_ShapeInfo("lambda_", False, (0, np.inf), (False, False)),
                _ShapeInfo("N", True, (0, np.inf), (False, False))]

    # 定义私有方法 _argcheck，检查 lambda_ 和 N 参数是否合法
    def _argcheck(self, lambda_, N):
        return (lambda_ > 0) & (N > 0) & _isintegral(N)

    # 定义私有方法 _get_support，返回 Boltzmann 分布的支持范围
    def _get_support(self, lambda_, N):
        return self.a, N - 1

    # 定义私有方法 _pmf，计算概率质量函数，对应 Boltzmann 分布的概率质量函数
    def _pmf(self, k, lambda_, N):
        fact = (1-exp(-lambda_))/(1-exp(-lambda_*N))
        return fact*exp(-lambda_*k)
    # 定义累积分布函数（Cumulative Distribution Function, CDF），计算指数分布的累积概率
    def _cdf(self, x, lambda_, N):
        # 取下整数部分作为参数 k
        k = floor(x)
        # 计算并返回累积分布函数的值
        return (1-exp(-lambda_*(k+1)))/(1-exp(-lambda_*N))

    # 定义反函数（Percent Point Function, PPF），计算指数分布的分位数
    def _ppf(self, q, lambda_, N):
        # 根据累积概率 q 计算新的调整过的 qnew
        qnew = q*(1-exp(-lambda_*N))
        # 计算分位数的初步估计值 vals
        vals = ceil(-1.0/lambda_ * log(1-qnew)-1)
        # 将 vals-1 限制在大于等于 0 的范围内，并转换为浮点数数组
        vals1 = (vals-1).clip(0.0, np.inf)
        # 计算基于 vals1 的累积分布函数的值 temp
        temp = self._cdf(vals1, lambda_, N)
        # 根据条件选择最终的分位数值，并返回结果
        return np.where(temp >= q, vals1, vals)

    # 定义统计量函数（Statistics Function），计算指数分布的均值、方差、偏度和峰度
    def _stats(self, lambda_, N):
        # 计算指数分布的参数 z 和 zN
        z = exp(-lambda_)
        zN = exp(-lambda_*N)
        # 计算均值 mu
        mu = z/(1.0-z)-N*zN/(1-zN)
        # 计算方差 var
        var = z/(1.0-z)**2 - N*N*zN/(1-zN)**2
        # 计算偏度 g1
        trm = (1-zN)/(1-z)
        trm2 = (z*trm**2 - N*N*zN)
        g1 = z*(1+z)*trm**3 - N**3*zN*(1+zN)
        g1 = g1 / trm2**(1.5)
        # 计算峰度 g2
        g2 = z*(1+4*z+z*z)*trm**4 - N**4 * zN*(1+4*zN+zN*zN)
        g2 = g2 / trm2 / trm2
        # 返回计算结果：均值、方差、偏度和峰度
        return mu, var, g1, g2
# 定义一个名为 `boltzmann` 的变量，调用 `boltzmann_gen` 函数生成一个名为 'boltzmann' 的离散截断指数分布的随机变量
boltzmann = boltzmann_gen(name='boltzmann', a=0,
                          longname='A truncated discrete exponential ')

# 定义一个名为 `randint_gen` 的类，继承自 `rv_discrete`
class randint_gen(rv_discrete):
    r"""A uniform discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `randint` is:

    .. math::

        f(k) = \frac{1}{\texttt{high} - \texttt{low}}

    for :math:`k \in \{\texttt{low}, \dots, \texttt{high} - 1\}`.

    `randint` takes :math:`\texttt{low}` and :math:`\texttt{high}` as shape
    parameters.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import randint
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> low, high = 7, 31
    >>> mean, var, skew, kurt = randint.stats(low, high, moments='mvsk')

    Display the probability mass function (``pmf``):

    >>> x = np.arange(low - 5, high + 5)
    >>> ax.plot(x, randint.pmf(x, low, high), 'bo', ms=8, label='randint pmf')
    >>> ax.vlines(x, 0, randint.pmf(x, low, high), colors='b', lw=5, alpha=0.5)

    Alternatively, the distribution object can be called (as a function) to
    fix the shape and location. This returns a "frozen" RV object holding the
    given parameters fixed.

    Freeze the distribution and display the frozen ``pmf``:

    >>> rv = randint(low, high)
    >>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-',
    ...           lw=1, label='frozen pmf')
    >>> ax.legend(loc='lower center')
    >>> plt.show()

    Check the relationship between the cumulative distribution function
    (``cdf``) and its inverse, the percent point function (``ppf``):

    >>> q = np.arange(low, high)
    >>> p = randint.cdf(q, low, high)
    >>> np.allclose(q, randint.ppf(p, low, high))
    True

    Generate random numbers:

    >>> r = randint.rvs(low, high, size=1000)

    """

    # 返回一个描述参数的列表，包括 'low' 和 'high'
    def _shape_info(self):
        return [_ShapeInfo("low", True, (-np.inf, np.inf), (False, False)),
                _ShapeInfo("high", True, (-np.inf, np.inf), (False, False))]

    # 检查参数 `low` 和 `high` 是否满足条件，均为整数且 `high` 大于 `low`
    def _argcheck(self, low, high):
        return (high > low) & _isintegral(low) & _isintegral(high)

    # 返回支持的范围，即从 `low` 到 `high-1`
    def _get_support(self, low, high):
        return low, high-1

    # 概率质量函数（pmf），返回每个 `k` 的概率密度值
    def _pmf(self, k, low, high):
        # randint.pmf(k) = 1./(high - low)
        p = np.ones_like(k) / (high - low)
        return np.where((k >= low) & (k < high), p, 0.)

    # 累积分布函数（cdf），返回小于等于 `x` 的概率
    def _cdf(self, x, low, high):
        k = floor(x)
        return (k - low + 1.) / (high - low)

    # 百分位点函数（ppf），返回小于等于给定概率 `q` 的值
    def _ppf(self, q, low, high):
        vals = ceil(q * (high - low) + low) - 1
        vals1 = (vals - 1).clip(low, high)
        temp = self._cdf(vals1, low, high)
        return np.where(temp >= q, vals1, vals)
    # 计算给定范围内的统计值：均值、方差、偏度和峰度
    def _stats(self, low, high):
        # 将low和high转换为NumPy数组
        m2, m1 = np.asarray(high), np.asarray(low)
        # 计算均值mu
        mu = (m2 + m1 - 1.0) / 2
        # 计算范围d
        d = m2 - m1
        # 计算方差var
        var = (d*d - 1) / 12.0
        # 初始化偏度g1为0
        g1 = 0.0
        # 计算峰度g2
        g2 = -6.0/5.0 * (d*d + 1.0) / (d*d - 1.0)
        # 返回计算结果：均值mu、方差var、偏度g1、峰度g2
        return mu, var, g1, g2
    
    # 生成随机整数数组
    def _rvs(self, low, high, size=None, random_state=None):
        """An array of *size* random integers >= ``low`` and < ``high``."""
        # 如果low和high都是标量，则无需向量化
        if np.asarray(low).size == 1 and np.asarray(high).size == 1:
            return rng_integers(random_state, low, high, size=size)
        
        # 如果指定了size，则需处理广播形状问题
        if size is not None:
            # NumPy的RandomState.randint()函数不会广播其参数。
            # 使用broadcast_to()将low和high的形状扩展到size。
            low = np.broadcast_to(low, size)
            high = np.broadcast_to(high, size)
        
        # 使用numpy.vectorize处理随机整数生成，部分应用rng_integers函数
        randint = np.vectorize(partial(rng_integers, random_state),
                               otypes=[np.dtype(int)])
        # 返回随机整数数组
        return randint(low, high)
    
    # 计算给定范围的熵
    def _entropy(self, low, high):
        # 使用对数函数计算范围的熵
        return log(high - low)
# 导入 randint_gen 函数并用指定参数调用，返回一个离散均匀分布的随机整数生成器
randint = randint_gen(name='randint', longname='A discrete uniform '
                      '(random integer)')


# FIXME: problems sampling.
# 定义一个 Zipf (Zeta) 离散随机变量生成器的类，继承自 rv_discrete 类
class zipf_gen(rv_discrete):
    r"""A Zipf (Zeta) discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipfian

    Notes
    -----
    The probability mass function for `zipf` is:

    .. math::

        f(k, a) = \frac{1}{\zeta(a) k^a}

    for :math:`k \ge 1`, :math:`a > 1`.

    `zipf` takes :math:`a > 1` as shape parameter. :math:`\zeta` is the
    Riemann zeta function (`scipy.special.zeta`)

    The Zipf distribution is also known as the zeta distribution, which is
    a special case of the Zipfian distribution (`zipfian`).

    %(after_notes)s

    References
    ----------
    .. [1] "Zeta Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Zeta_distribution

    %(example)s

    Confirm that `zipf` is the large `n` limit of `zipfian`.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipf.pmf(k, a), zipfian.pmf(k, a, n=10000000))
    True

    """

    # 定义用于确定分布参数形状的方法
    def _shape_info(self):
        return [_ShapeInfo("a", False, (1, np.inf), (False, False))]

    # 定义生成随机变量的方法
    def _rvs(self, a, size=None, random_state=None):
        return random_state.zipf(a, size=size)

    # 定义参数检查方法，确保参数 a 大于 1
    def _argcheck(self, a):
        return a > 1

    # 定义概率质量函数方法，计算给定参数下的概率质量
    def _pmf(self, k, a):
        k = k.astype(np.float64)
        # zipf.pmf(k, a) = 1/(zeta(a) * k**a)
        Pk = 1.0 / special.zeta(a, 1) * k**-a
        return Pk

    # 定义原点矩方法，计算原点矩
    def _munp(self, n, a):
        return _lazywhere(
            a > n + 1, (a, n),
            lambda a, n: special.zeta(a - n, 1) / special.zeta(a, 1),
            np.inf)


# 创建一个名为 zipf 的 Zipf (Zeta) 分布对象实例，参数 a=1
zipf = zipf_gen(a=1, name='zipf', longname='A Zipf')


# 定义一个函数，计算广义调和数，对于 a > 1 的情况
def _gen_harmonic_gt1(n, a):
    """Generalized harmonic number, a > 1"""
    # See https://en.wikipedia.org/wiki/Harmonic_number; search for "hurwitz"
    return zeta(a, 1) - zeta(a, n+1)


# 定义一个函数，计算广义调和数，对于 a <= 1 的情况
def _gen_harmonic_leq1(n, a):
    """Generalized harmonic number, a <= 1"""
    if not np.size(n):
        return n
    n_max = np.max(n)  # loop starts at maximum of all n
    out = np.zeros_like(a, dtype=float)
    # add terms of harmonic series; starting from smallest to avoid roundoff
    for i in np.arange(n_max, 0, -1, dtype=float):
        mask = i <= n  # don't add terms after nth
        out[mask] += 1/i**a[mask]
    return out


# 定义一个函数，根据参数计算广义调和数
def _gen_harmonic(n, a):
    """Generalized harmonic number"""
    n, a = np.broadcast_arrays(n, a)
    return _lazywhere(a > 1, (n, a),
                      f=_gen_harmonic_gt1, f2=_gen_harmonic_leq1)


# 定义一个 Zipfian 离散随机变量生成器的类，继承自 rv_discrete 类
class zipfian_gen(rv_discrete):
    r"""A Zipfian discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipf

    Notes
    -----
    The probability mass function for `zipfian` is:

    .. math::

        f(k, a, n) = \frac{1}{H_{n,a} k^a}

    for :math:`k \in \{1, 2, \dots, n-1, n\}`, :math:`a \ge 0`,
    :math:`n \in \{1, 2, 3, \dots\}`.
    `zipfian` takes :math:`a` and :math:`n` as shape parameters.
    :math:`H_{n,a}` is the :math:`n`:sup:`th` generalized harmonic
    number of order :math:`a`.

    The Zipfian distribution reduces to the Zipf (zeta) distribution as
    :math:`n \rightarrow \infty`.

    %(after_notes)s

    References
    ----------
    .. [1] "Zipf's Law", Wikipedia, https://en.wikipedia.org/wiki/Zipf's_law
    .. [2] Larry Leemis, "Zipf Distribution", Univariate Distribution
           Relationships. http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf

    %(example)s

    Confirm that `zipfian` reduces to `zipf` for large `n`, ``a > 1``.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipfian.pmf(k, a=3.5, n=10000000), zipf.pmf(k, a=3.5))
    True

    """


    # 返回参数描述 `a` 和 `n` 的形状信息，包括是否必须为整数和取值范围
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (False, False))]

    # 参数检查函数，确保 `a` 大于等于 0，`n` 大于 0 且为整数
    def _argcheck(self, a, n):
        # 需要使用 np.asarray 转换，因为某些情况下 moment 函数等不会自动转换
        return (a >= 0) & (n > 0) & (n == np.asarray(n, dtype=int))

    # 返回分布的支持范围，从1到`n`
    def _get_support(self, a, n):
        return 1, n

    # 概率质量函数 (PMF)，返回给定参数下的概率质量值
    def _pmf(self, k, a, n):
        k = k.astype(np.float64)
        return 1.0 / _gen_harmonic(n, a) * k**-a

    # 累积分布函数 (CDF)，返回给定参数下的累积分布值
    def _cdf(self, k, a, n):
        return _gen_harmonic(k, a) / _gen_harmonic(n, a)

    # 生存函数 (SF)，返回给定参数下的生存函数值
    def _sf(self, k, a, n):
        k = k + 1  # 以匹配 SciPy 的惯例
        # 参考 http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        return ((k**a*(_gen_harmonic(n, a) - _gen_harmonic(k, a)) + 1)
                / (k**a*_gen_harmonic(n, a)))

    # 统计量函数，返回给定参数下的统计特性，如均值、方差等
    def _stats(self, a, n):
        # 参考 http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        Hna = _gen_harmonic(n, a)
        Hna1 = _gen_harmonic(n, a-1)
        Hna2 = _gen_harmonic(n, a-2)
        Hna3 = _gen_harmonic(n, a-3)
        Hna4 = _gen_harmonic(n, a-4)
        mu1 = Hna1/Hna
        mu2n = (Hna2*Hna - Hna1**2)
        mu2d = Hna**2
        mu2 = mu2n / mu2d
        g1 = (Hna3/Hna - 3*Hna1*Hna2/Hna**2 + 2*Hna1**3/Hna**3)/mu2**(3/2)
        g2 = (Hna**3*Hna4 - 4*Hna**2*Hna1*Hna3 + 6*Hna*Hna1**2*Hna2
              - 3*Hna1**4) / mu2n**2
        g2 -= 3
        return mu1, mu2, g1, g2
# 使用 zipfian_gen 函数生成一个名称为 zipfian 的对象实例，并传入参数 a=1, name='zipfian', longname='A Zipfian'
zipfian = zipfian_gen(a=1, name='zipfian', longname='A Zipfian')

# 定义一个名为 dlaplace_gen 的类，继承自 rv_discrete 类，表示一个离散的拉普拉斯分布随机变量
class dlaplace_gen(rv_discrete):
    r"""A  Laplacian discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `dlaplace` is:

    .. math::

        f(k) = \tanh(a/2) \exp(-a |k|)

    for integers :math:`k` and :math:`a > 0`.

    `dlaplace` takes :math:`a` as shape parameter.

    %(after_notes)s

    %(example)s

    """

    # 定义一个方法 _shape_info，返回一个列表，描述了参数 'a' 的信息
    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    # 定义一个方法 _pmf，表示概率质量函数，计算离散拉普拉斯分布的概率密度函数
    def _pmf(self, k, a):
        # dlaplace.pmf(k) = tanh(a/2) * exp(-a*abs(k))
        return tanh(a/2.0) * exp(-a * abs(k))

    # 定义一个方法 _cdf，表示累积分布函数，计算离散拉普拉斯分布的累积分布函数
    def _cdf(self, x, a):
        k = floor(x)

        # 内部定义一个函数 f(k, a)，计算离散拉普拉斯分布的累积分布函数的一部分
        def f(k, a):
            return 1.0 - exp(-a * k) / (exp(a) + 1)

        # 内部定义一个函数 f2(k, a)，计算离散拉普拉斯分布的累积分布函数的另一部分
        def f2(k, a):
            return exp(a * (k + 1)) / (exp(a) + 1)

        return _lazywhere(k >= 0, (k, a), f=f, f2=f2)

    # 定义一个方法 _ppf，表示反函数，计算给定概率值对应的分位点
    def _ppf(self, q, a):
        const = 1 + exp(a)
        vals = ceil(np.where(q < 1.0 / (1 + exp(-a)),
                             log(q*const) / a - 1,
                             -log((1-q) * const) / a))
        vals1 = vals - 1
        return np.where(self._cdf(vals1, a) >= q, vals1, vals)

    # 定义一个方法 _stats，计算离散拉普拉斯分布的统计特性，返回均值、方差等
    def _stats(self, a):
        ea = exp(a)
        mu2 = 2.*ea/(ea-1.)**2
        mu4 = 2.*ea*(ea**2+10.*ea+1.) / (ea-1.)**4
        return 0., mu2, 0., mu4/mu2**2 - 3.

    # 定义一个方法 _entropy，计算离散拉普拉斯分布的信息熵
    def _entropy(self, a):
        return a / sinh(a) - log(tanh(a/2.0))

    # 定义一个方法 _rvs，表示随机变量生成函数，返回符合离散拉普拉斯分布的随机变量
    def _rvs(self, a, size=None, random_state=None):
        # 离散拉普拉斯分布等价于两个独立几何分布之差的分布
        # 利用几何分布生成离散拉普拉斯分布的随机变量
        probOfSuccess = -np.expm1(-np.asarray(a))
        x = random_state.geometric(probOfSuccess, size=size)
        y = random_state.geometric(probOfSuccess, size=size)
        return x - y

# 创建一个名为 dlaplace 的实例对象，表示一个离散拉普拉斯分布的随机变量，参数 a 初始化为负无穷
dlaplace = dlaplace_gen(a=-np.inf,
                        name='dlaplace', longname='A discrete Laplacian')

# 定义一个类 skellam_gen，表示一个 Skellam 分布的离散随机变量
class skellam_gen(rv_discrete):
    r"""A  Skellam discrete random variable.

    %(before_notes)s

    Notes
    -----
    Probability distribution of the difference of two correlated or
    uncorrelated Poisson random variables.

    Let :math:`k_1` and :math:`k_2` be two Poisson-distributed r.v. with
    expected values :math:`\lambda_1` and :math:`\lambda_2`. Then,
    :math:`k_1 - k_2` follows a Skellam distribution with parameters
    """
    Skellam分布是描述两个独立的泊松分布随机变量差值的概率分布。本类定义了Skellam分布的各种方法。
    
    Parameters :math:`\mu_1` 和 :math:`\mu_2` 是分布的形状参数，必须严格为正数。

    For details see: https://en.wikipedia.org/wiki/Skellam_distribution

    `skellam` 接受 :math:`\mu_1` 和 :math:`\mu_2` 作为形状参数。

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        # 返回一个包含参数信息的列表，描述了参数的名称、是否为可选、取值范围及其边界情况
        return [_ShapeInfo("mu1", False, (0, np.inf), (False, False)),
                _ShapeInfo("mu2", False, (0, np.inf), (False, False))]

    def _rvs(self, mu1, mu2, size=None, random_state=None):
        # 生成Skellam分布的随机变量
        n = size
        return (random_state.poisson(mu1, n) -
                random_state.poisson(mu2, n))

    def _pmf(self, x, mu1, mu2):
        # 计算Skellam分布的概率质量函数
        with np.errstate(over='ignore'):  # 处理数值溢出问题，参见gh-17432
            px = np.where(x < 0,
                          scu._ncx2_pdf(2*mu2, 2*(1-x), 2*mu1)*2,
                          scu._ncx2_pdf(2*mu1, 2*(1+x), 2*mu2)*2)
            # ncx2.pdf() 对极低概率返回NaN
        return px

    def _cdf(self, x, mu1, mu2):
        # 计算Skellam分布的累积分布函数
        x = floor(x)
        with np.errstate(over='ignore'):  # 处理数值溢出问题，参见gh-17432
            px = np.where(x < 0,
                          scu._ncx2_cdf(2*mu2, -2*x, 2*mu1),
                          1 - scu._ncx2_cdf(2*mu1, 2*(x+1), 2*mu2))
        return px

    def _stats(self, mu1, mu2):
        # 计算Skellam分布的均值、方差、偏度和峰度
        mean = mu1 - mu2
        var = mu1 + mu2
        g1 = mean / sqrt((var)**3)
        g2 = 1 / var
        return mean, var, g1, g2
# 创建一个 Skellam 分布的生成器，设定参数 a 为负无穷，名称为 "skellam"，长名称为 "A Skellam"
skellam = skellam_gen(a=-np.inf, name="skellam", longname='A Skellam')

# 定义一个 Yule-Simon 离散随机变量的类 yulesimon_gen，继承自 rv_discrete
class yulesimon_gen(rv_discrete):
    r"""A Yule-Simon discrete random variable.

    %(before_notes)s

    Notes
    -----

    The probability mass function for the `yulesimon` is:

    .. math::

        f(k) =  \alpha B(k, \alpha+1)

    for :math:`k=1,2,3,...`, where :math:`\alpha>0`.
    Here :math:`B` refers to the `scipy.special.beta` function.

    The sampling of random variates is based on pg 553, Section 6.3 of [1]_.
    Our notation maps to the referenced logic via :math:`\alpha=a-1`.

    For details see the wikipedia entry [2]_.

    References
    ----------
    .. [1] Devroye, Luc. "Non-uniform Random Variate Generation",
         (1986) Springer, New York.

    .. [2] https://en.wikipedia.org/wiki/Yule-Simon_distribution

    %(after_notes)s

    %(example)s

    """
    
    # 定义返回参数形状信息的方法，此处返回一个描述 alpha 的 _ShapeInfo 对象的列表
    def _shape_info(self):
        return [_ShapeInfo("alpha", False, (0, np.inf), (False, False))]

    # 定义生成随机变量的方法，根据参数 alpha 生成指定大小的 Yule-Simon 随机变量
    def _rvs(self, alpha, size=None, random_state=None):
        E1 = random_state.standard_exponential(size)
        E2 = random_state.standard_exponential(size)
        ans = ceil(-E1 / log1p(-exp(-E2 / alpha)))
        return ans

    # 定义概率质量函数的方法，返回 Yule-Simon 分布的概率质量函数值
    def _pmf(self, x, alpha):
        return alpha * special.beta(x, alpha + 1)

    # 定义参数检查方法，返回参数 alpha 是否符合条件的布尔值
    def _argcheck(self, alpha):
        return (alpha > 0)

    # 定义对数概率质量函数的方法，返回 Yule-Simon 分布的对数概率质量函数值
    def _logpmf(self, x, alpha):
        return log(alpha) + special.betaln(x, alpha + 1)

    # 定义累积分布函数的方法，返回 Yule-Simon 分布的累积分布函数值
    def _cdf(self, x, alpha):
        return 1 - x * special.beta(x, alpha + 1)

    # 定义生存函数的方法，返回 Yule-Simon 分布的生存函数值
    def _sf(self, x, alpha):
        return x * special.beta(x, alpha + 1)

    # 定义对数生存函数的方法，返回 Yule-Simon 分布的对数生存函数值
    def _logsf(self, x, alpha):
        return log(x) + special.betaln(x, alpha + 1)

    # 定义统计特征的方法，返回 Yule-Simon 分布的均值、方差及偏度、峰度
    def _stats(self, alpha):
        mu = np.where(alpha <= 1, np.inf, alpha / (alpha - 1))
        mu2 = np.where(alpha > 2,
                       alpha**2 / ((alpha - 2.0) * (alpha - 1)**2),
                       np.inf)
        mu2 = np.where(alpha <= 1, np.nan, mu2)
        g1 = np.where(alpha > 3,
                      sqrt(alpha - 2) * (alpha + 1)**2 / (alpha * (alpha - 3)),
                      np.inf)
        g1 = np.where(alpha <= 2, np.nan, g1)
        g2 = np.where(alpha > 4,
                      alpha + 3 + ((11 * alpha**3 - 49 * alpha - 22) /
                                   (alpha * (alpha - 4) * (alpha - 3))),
                      np.inf)
        g2 = np.where(alpha <= 2, np.nan, g2)
        return mu, mu2, g1, g2

# 创建一个 Yule-Simon 分布的生成器，名称为 'yulesimon'，参数 a 设定为 1
yulesimon = yulesimon_gen(name='yulesimon', a=1)

# 定义一个非中心超几何离散随机变量的类 _nchypergeom_gen，用于 nchypergeom_fisher_gen 和 nchypergeom_wallenius_gen 的子类
class _nchypergeom_gen(rv_discrete):
    r"""A noncentral hypergeometric discrete random variable.

    For subclassing by nchypergeom_fisher_gen and nchypergeom_wallenius_gen.

    """

    rvs_name = None  # 随机变量名称初始化为 None
    dist = None  # 分布初始化为 None
    # 返回一个包含 _ShapeInfo 对象的列表，描述了参数 M, n, N, odds 的形状信息
    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("N", True, (0, np.inf), (True, False)),
                _ShapeInfo("odds", False, (0, np.inf), (False, False))]

    # 根据给定的 M, n, N, odds 计算并返回支持值的范围 x_min 和 x_max
    def _get_support(self, M, n, N, odds):
        N, m1, n = M, n, N  # 使用 Wikipedia 的符号约定
        m2 = N - m1
        x_min = np.maximum(0, n - m2)
        x_max = np.minimum(n, m1)
        return x_min, x_max

    # 检查输入的 M, n, N, odds 是否满足条件，返回布尔数组
    def _argcheck(self, M, n, N, odds):
        M, n = np.asarray(M), np.asarray(n),
        N, odds = np.asarray(N), np.asarray(odds)
        cond1 = (M.astype(int) == M) & (M >= 0)
        cond2 = (n.astype(int) == n) & (n >= 0)
        cond3 = (N.astype(int) == N) & (N >= 0)
        cond4 = odds > 0
        cond5 = N <= M
        cond6 = n <= M
        return cond1 & cond2 & cond3 & cond4 & cond5 & cond6

    # 生成服从指定分布的随机变量，返回随机变量数组
    def _rvs(self, M, n, N, odds, size=None, random_state=None):

        # 定义内部函数 _rvs1 用于生成随机变量
        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, N, odds, size, random_state):
            length = np.prod(size)
            urn = _PyStochasticLib3()
            rv_gen = getattr(urn, self.rvs_name)
            rvs = rv_gen(N, n, M, odds, length, random_state)
            rvs = rvs.reshape(size)
            return rvs

        return _rvs1(M, n, N, odds, size=size, random_state=random_state)

    # 计算离散分布的概率质量函数值，返回数组
    def _pmf(self, x, M, n, N, odds):

        # 广播输入参数，确保它们具有相同的形状
        x, M, n, N, odds = np.broadcast_arrays(x, M, n, N, odds)
        if x.size == 0:  # 处理空输入的情况
            return np.empty_like(x)

        # 定义内部函数 _pmf1，使用向量化计算概率质量函数值
        @np.vectorize
        def _pmf1(x, M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.probability(x)

        return _pmf1(x, M, n, N, odds)

    # 计算分布的统计量，返回包含均值和方差的元组
    def _stats(self, M, n, N, odds, moments):

        # 定义内部函数 _moments1，使用向量化计算分布的统计量
        @np.vectorize
        def _moments1(M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.moments()

        # 根据 moments 参数选择返回的统计量
        m, v = (_moments1(M, n, N, odds) if ("m" in moments or "v" in moments)
                else (None, None))
        s, k = None, None
        return m, v, s, k
class nchypergeom_fisher_gen(_nchypergeom_gen):
    r"""A Fisher's noncentral hypergeometric discrete random variable.

    Fisher's noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    take a handful of objects from the bin at once and find out afterwards
    that we took `N` objects.

    %(before_notes)s

    See Also
    --------
    nchypergeom_wallenius, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; M, n, N, \omega) =
        \frac{\binom{n}{x}\binom{M - n}{N-x}\omega^x}{P_0},

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        P_0 = \sum_{y=x_l}^{x_u} \binom{n}{y}\binom{M - n}{N-y}\omega^y,

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_fisher` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Fisher's noncentral hypergeometric distribution is distinct
    from Wallenius' noncentral hypergeometric distribution, which models
    drawing a pre-determined `N` objects from a bin one by one.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Fisher's noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution

    %(example)s

    """

    rvs_name = "rvs_fisher"
    dist = _PyFishersNCHypergeometric
    """
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    draw a pre-determined `N` objects from a bin one by one.

    %(before_notes)s

    See Also
    --------
    nchypergeom_fisher, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; N, n, M) = \binom{n}{x} \binom{M - n}{N-x}
        \int_0^1 \left(1-t^{\omega/D}\right)^x\left(1-t^{1/D}\right)^{N-x} dt

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        D = \omega(n - x) + ((M - n)-(N-x)),

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_wallenius` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Wallenius' noncentral hypergeometric distribution is distinct
    from Fisher's noncentral hypergeometric distribution, which models
    take a handful of objects from the bin at once, finding out afterwards
    that `N` objects were taken.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Wallenius' noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Wallenius'_noncentral_hypergeometric_distribution

    %(example)s

    """

    # 设置随机变量名称为 "rvs_wallenius"
    rvs_name = "rvs_wallenius"
    # 使用 PyWalleniusNCHypergeometric 进行计算
    dist = _PyWalleniusNCHypergeometric
# 使用指定参数创建一个名为 'nchypergeom_wallenius' 的非中心超几何分布生成器
nchypergeom_wallenius = nchypergeom_wallenius_gen(
    name='nchypergeom_wallenius',
    longname="A Wallenius' noncentral hypergeometric")

# 复制全局作用域中的变量，并将其转换为键值对列表
pairs = list(globals().copy().items())
# 调用函数 get_distribution_names，获取分布名称和生成器名称
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_discrete)

# 将所有分布名称和生成器名称添加到 '__all__' 列表中，用于模块的导入
__all__ = _distn_names + _distn_gen_names
```