# `D:\src\scipysrc\scipy\scipy\stats\_morestats.py`

```
from __future__ import annotations
# 导入未来版本的 annotations 特性，用于支持类型提示中的类型自引用

import math
# 导入数学库，提供数学函数和常量

import warnings
# 导入警告处理模块，用于管理警告信息的显示

from collections import namedtuple
# 导入 namedtuple 类，用于创建命名元组对象

import numpy as np
# 导入 NumPy 库，并使用 np 作为别名

from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, sqrt, array,
                   pi, exp, ravel, count_nonzero)
# 从 NumPy 中导入多个函数和常量

from scipy import optimize, special, interpolate, stats
# 导入 SciPy 库的多个模块

from scipy._lib._bunch import _make_tuple_bunch
# 导入 SciPy 内部库中的 _make_tuple_bunch 函数

from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
# 导入 SciPy 内部库中的多个辅助函数

from scipy._lib._array_api import (array_namespace, xp_minimum, size as xp_size,
                                   xp_moveaxis_to_end)
# 导入 SciPy 内部库中的数组 API 相关函数和变量

from ._ansari_swilk_statistics import gscale, swilk
# 导入 Ansari-Swilk 统计模块中的 gscale 和 swilk 函数

from . import _stats_py, _wilcoxon
# 导入当前包中的 _stats_py 和 _wilcoxon 模块

from ._fit import FitResult
# 导入拟合结果类 FitResult

from ._stats_py import (find_repeats, _get_pvalue, SignificanceResult,
                        _SimpleNormal, _SimpleChi2)
# 从当前包中的 _stats_py 模块导入多个函数和类

from .contingency import chi2_contingency
# 导入列联表卡方检验函数

from . import distributions
# 导入当前包中的分布模块

from ._distn_infrastructure import rv_generic
# 导入分布基础设施模块中的随机变量类

from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
# 导入处理轴上 NaN 值的策略工厂函数和数组广播函数

__all__ = ['mvsdist',
           'bayes_mvs', 'kstat', 'kstatvar', 'probplot', 'ppcc_max', 'ppcc_plot',
           'boxcox_llf', 'boxcox', 'boxcox_normmax', 'boxcox_normplot',
           'shapiro', 'anderson', 'ansari', 'bartlett', 'levene',
           'fligner', 'mood', 'wilcoxon', 'median_test',
           'circmean', 'circvar', 'circstd', 'anderson_ksamp',
           'yeojohnson_llf', 'yeojohnson', 'yeojohnson_normmax',
           'yeojohnson_normplot', 'directional_stats',
           'false_discovery_control'
           ]
# 定义模块的公开接口列表

Mean = namedtuple('Mean', ('statistic', 'minmax'))
# 定义命名元组 Mean，用于存储统计量和范围信息

Variance = namedtuple('Variance', ('statistic', 'minmax'))
# 定义命名元组 Variance，用于存储方差和范围信息

Std_dev = namedtuple('Std_dev', ('statistic', 'minmax'))
# 定义命名元组 Std_dev，用于存储标准差和范围信息


def bayes_mvs(data, alpha=0.90):
    r"""
    Bayesian confidence intervals for the mean, var, and std.

    Parameters
    ----------
    data : array_like
        Input data, if multi-dimensional it is flattened to 1-D by `bayes_mvs`.
        Requires 2 or more data points.
    alpha : float, optional
        Probability that the returned confidence interval contains
        the true parameter.

    Returns
    -------
    mean_cntr, var_cntr, std_cntr : tuple
        The three results are for the mean, variance and standard deviation,
        respectively.  Each result is a tuple of the form::

            (center, (lower, upper))

        with ``center`` the mean of the conditional pdf of the value given the
        data, and ``(lower, upper)`` a confidence interval, centered on the
        median, containing the estimate to a probability ``alpha``.

    See Also
    --------
    mvsdist

    Notes
    -----
    Each tuple of mean, variance, and standard deviation estimates represent
    the (center, (lower, upper)) with center the mean of the conditional pdf
    of the value given the data and (lower, upper) is a confidence interval
    centered on the median, containing the estimate to a probability
    ``alpha``.
    """
    # Bayesian 方法计算给定数据的均值、方差和标准差的置信区间
    Converts data to 1-D and assumes all data has the same mean and variance.
    Uses Jeffrey's prior for variance and std.

    Equivalent to ``tuple((x.mean(), x.interval(alpha)) for x in mvsdist(dat))``

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    First a basic example to demonstrate the outputs:

    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.bayes_mvs(data)
    >>> mean
    Mean(statistic=9.0, minmax=(7.103650222612533, 10.896349777387467))
    >>> var
    Variance(statistic=10.0, minmax=(3.176724206, 24.45910382))
    >>> std
    Std_dev(statistic=2.9724954732045084,
            minmax=(1.7823367265645143, 4.945614605014631))

    Now we generate some normally distributed random data, and get estimates of
    mean and standard deviation with 95% confidence intervals for those
    estimates:

    >>> n_samples = 100000
    >>> data = stats.norm.rvs(size=n_samples)
    >>> res_mean, res_var, res_std = stats.bayes_mvs(data, alpha=0.95)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.hist(data, bins=100, density=True, label='Histogram of data')
    >>> ax.vlines(res_mean.statistic, 0, 0.5, colors='r', label='Estimated mean')
    >>> ax.axvspan(res_mean.minmax[0],res_mean.minmax[1], facecolor='r',
    ...            alpha=0.2, label=r'Estimated mean (95% limits)')
    >>> ax.vlines(res_std.statistic, 0, 0.5, colors='g', label='Estimated scale')
    >>> ax.axvspan(res_std.minmax[0],res_std.minmax[1], facecolor='g', alpha=0.2,
    ...            label=r'Estimated scale (95% limits)')

    >>> ax.legend(fontsize=10)
    >>> ax.set_xlim([-4, 4])
    >>> ax.set_ylim([0, 0.5])
    >>> plt.show()

    """
    # 使用 mvsdist 函数计算给定数据的均值、方差和标准差的 Bayesian 估计
    m, v, s = mvsdist(data)
    # 检查 alpha 值是否在合理范围内 (0 < alpha < 1)，若不是则引发 ValueError
    if alpha >= 1 or alpha <= 0:
        raise ValueError(f"0 < alpha < 1 is required, but {alpha=} was given.")

    # 构建 Mean、Variance 和 Std_dev 对象，分别表示均值、方差和标准差的 Bayesian 估计
    m_res = Mean(m.mean(), m.interval(alpha))
    v_res = Variance(v.mean(), v.interval(alpha))
    s_res = Std_dev(s.mean(), s.interval(alpha))

    # 返回均值、方差和标准差的 Bayesian 估计结果
    return m_res, v_res, s_res
# 定义函数 mvsdist，用于计算给定数据的平均值、方差和标准差的“冻结”分布对象
def mvsdist(data):
    # 将输入数据展平为一维数组
    x = ravel(data)
    # 获取数据数组长度
    n = len(x)
    # 如果数据点少于2个，抛出数值错误异常
    if n < 2:
        raise ValueError("Need at least 2 data-points.")
    
    # 计算数据的平均值
    xbar = x.mean()
    # 计算数据的方差
    C = x.var()
    
    # 根据数据点数选择不同的分布类型
    if n > 1000:  # 对于大数据量，使用高斯分布的近似
        # 使用正态分布表示平均值
        mdist = distributions.norm(loc=xbar, scale=math.sqrt(C / n))
        # 使用正态分布表示标准差
        sdist = distributions.norm(loc=math.sqrt(C), scale=math.sqrt(C / (2. * n)))
        # 使用正态分布表示方差
        vdist = distributions.norm(loc=C, scale=math.sqrt(2.0 / n) * C)
    else:
        # 对于较小的数据点数，使用 t 分布、gengamma 分布和 invgamma 分布
        nm1 = n - 1
        fac = n * C / 2.
        val = nm1 / 2.
        # 使用 t 分布表示平均值
        mdist = distributions.t(nm1, loc=xbar, scale=math.sqrt(C / nm1))
        # 使用 gengamma 分布表示标准差
        sdist = distributions.gengamma(val, -2, scale=math.sqrt(fac))
        # 使用 invgamma 分布表示方差
        vdist = distributions.invgamma(val, scale=fac)
    
    # 返回计算得到的分布对象，分别表示平均值、方差和标准差
    return mdist, vdist, sdist


# 创建一个装饰器函数 kstat，用于处理输入数据的 NaN 策略，并返回 k-statistic 中的第 n 个值
@_axis_nan_policy_factory(
    # 将输入数据直接返回
    lambda x: x,
    # 将结果转换为元组形式返回
    result_to_tuple=lambda x: (x,),
    # 输出一个结果
    n_outputs=1,
    # 默认轴设置为 None
    default_axis=None
)
# 定义 kstat 函数，计算输入数据的第 n 个 k-statistic
def kstat(data, n=2, *, axis=None):
    r"""
    Return the `n` th k-statistic ( ``1<=n<=4`` so far).

    The `n` th k-statistic ``k_n`` is the unique symmetric unbiased estimator of the
    `n` th cumulant :math:`\kappa_n` [1]_ [2]_.

    Parameters
    ----------
    data : array_like
        Input array.
    n : int, {1, 2, 3, 4}, optional
        Default is equal to 2.
    # 将数据转换为指定的数组命名空间（如 numpy 或 cupy）
    xp = array_namespace(data)
    # 将数据转换为指定数组命名空间的数组对象
    data = xp.asarray(data)
    # 检查统计阶数是否在支持的范围内，不在范围内则引发异常
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    # 将 n 转换为整数类型
    n = int(n)
    # 如果 axis 为 None，则将数据展平为一维数组，并设定 axis 为 0
    if axis is None:
        data = xp.reshape(data, (-1,))
        axis = 0

    # 获取数据在指定轴上的维度大小
    N = data.shape[axis]

    # 计算数据的各阶矩（S1 到 Sn），使用数组命名空间对象的求和函数
    S = [None] + [xp.sum(data**k, axis=axis) for k in range(1, n + 1)]

    # 根据不同的统计阶数 n，计算对应的 k-statistic
    if n == 1:
        return S[1] * 1.0/N
    elif n == 2:
        return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
    elif n == 3:
        return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
    elif n == 4:
        return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                 4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                (N*(N-1.0)*(N-2.0)*(N-3.0)))
    else:
        # 如果 n 不在支持的范围内，引发异常
        raise ValueError("Should not be here.")
@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1, default_axis=None
)
def kstatvar(data, n=2, *, axis=None):
    r"""Return an unbiased estimator of the variance of the k-statistic.

    See `kstat` and [1]_ for more details about the k-statistic.

    Parameters
    ----------
    data : array_like
        Input array.
    n : int, {1, 2}, optional
        Default is equal to 2.
    axis : int or None, default: None
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear
        in a corresponding element of the output. If ``None``, the input will
        be raveled before computing the statistic.

    Returns
    -------
    kstatvar : float
        The `n` th k-statistic variance.

    See Also
    --------
    kstat : Returns the n-th k-statistic.
    moment : Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    Unbiased estimators of the variances of the first two k-statistics are given by

    .. math::

        \mathrm{var}(k_1) &= \frac{k_2}{n}, \\
        \mathrm{var}(k_2) &= \frac{2k_2^2n + (n-1)k_4}{n(n - 1)}.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/k-Statistic.html

    """  # noqa: E501

    # 使用 array_namespace 将数据转换为合适的数值计算命名空间
    xp = array_namespace(data)
    # 将 data 转换为 xp 数组
    data = xp.asarray(data)
    # 如果 axis 为 None，则将 data 展平成一维数组，并指定 axis 为 0
    if axis is None:
        data = xp.reshape(data, (-1,))
        axis = 0
    # 获取数据在指定 axis 上的形状信息
    N = data.shape[axis]

    # 根据 n 的值计算对应的 k-statistic 方差估计
    if n == 1:
        # 对于 n=1，返回 kstat(data, n=2, axis=axis, _no_deco=True) 除以 N 的结果
        return kstat(data, n=2, axis=axis, _no_deco=True) * 1.0/N
    elif n == 2:
        # 对于 n=2，计算 k2 和 k4 的值，然后返回 k2 和 k4 计算得到的方差估计结果
        k2 = kstat(data, n=2, axis=axis, _no_deco=True)
        k4 = kstat(data, n=4, axis=axis, _no_deco=True)
        return (2*N*k2**2 + (N-1)*k4) / (N*(N+1))
    else:
        # 如果 n 不是 1 或 2，则抛出异常
        raise ValueError("Only n=1 or n=2 supported.")


def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    Order statistics of the uniform distribution on the unit interval
    are marginally distributed according to beta distributions.
    The expectations of these order statistic are evenly spaced across
    the interval, but the distributions are skewed in a way that
    pushes the medians slightly towards the endpoints of the unit interval:

    >>> import numpy as np
    >>> n = 4
    >>> k = np.arange(1, n+1)
    >>> from scipy.stats import beta
    >>> a = k
    >>> b = n-k+1
    >>> beta.mean(a, b)
    array([0.2, 0.4, 0.6, 0.8])
    >>> beta.median(a, b)
    array([0.15910358, 0.38572757, 0.61427243, 0.84089642])

    The Filliben approximation uses the exact medians of the smallest
    """

    # 返回 n 个样本大小的均匀次序统计中位数的近似值
    v = np.linspace(1/(2*n), 1 - 1/(2*n), n)
    return v
    # 创建一个长度为 n 的空数组 v，数据类型为 float64
    v = np.empty(n, dtype=np.float64)
    
    # 设置 v 数组的最后一个元素，使其为 0.5 的 n 次方根
    v[-1] = 0.5**(1.0 / n)
    
    # 设置 v 数组的第一个元素，使其为 1 减去最后一个元素的值
    v[0] = 1 - v[-1]
    
    # 创建一个从 2 到 n-1 的整数数组 i
    i = np.arange(2, n)
    
    # 使用公式计算 v 数组中除了第一个和最后一个位置外的元素
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    
    # 返回计算得到的数组 v
    return v
# 解析 `dist` 关键字参数的函数，用于处理概率分布对象或者概率分布名称字符串
def _parse_dist_kw(dist, enforce_subclass=True):
    """Parse `dist` keyword.

    Parameters
    ----------
    dist : str or stats.distributions instance.
        Several functions take `dist` as a keyword, hence this utility
        function.
    enforce_subclass : bool, optional
        If True (default), `dist` needs to be a
        `_distn_infrastructure.rv_generic` instance.
        It can sometimes be useful to set this keyword to False, if a function
        wants to accept objects that just look somewhat like such an instance
        (for example, they have a ``ppf`` method).

    """
    # 如果 `dist` 是 `_distn_infrastructure.rv_generic` 的实例，则直接通过
    if isinstance(dist, rv_generic):
        pass
    # 如果 `dist` 是字符串，则尝试获取对应的分布对象
    elif isinstance(dist, str):
        try:
            dist = getattr(distributions, dist)
        except AttributeError as e:
            # 如果获取失败，则抛出异常
            raise ValueError(f"{dist} is not a valid distribution name") from e
    # 如果 enforce_subclass 为 True 且 `dist` 不是 rv_generic 的实例，则抛出异常
    elif enforce_subclass:
        msg = ("`dist` should be a stats.distributions instance or a string "
               "with the name of such a distribution.")
        raise ValueError(msg)

    return dist


# 辅助函数：给统计图添加轴标签和标题
def _add_axis_labels_title(plot, xlabel, ylabel, title):
    """Helper function to add axes labels and a title to stats plots."""
    try:
        # 如果 plot 对象有 `set_title` 方法，则将标题、X轴标签和Y轴标签设置进去
        if hasattr(plot, 'set_title'):
            # Matplotlib Axes 实例或类似对象
            plot.set_title(title)
            plot.set_xlabel(xlabel)
            plot.set_ylabel(ylabel)
        else:
            # 如果是 matplotlib.pyplot 模块，则设置标题、X轴标签和Y轴标签
            plot.title(title)
            plot.xlabel(xlabel)
            plot.ylabel(ylabel)
    except Exception:
        # 如果出现异常，说明不是 Matplotlib 对象或类似对象，忽略标签和标题的添加
        pass


# 概率图函数：生成概率图并可选地展示
def probplot(x, sparams=(), dist='norm', fit=True, plot=None, rvalue=False):
    """
    Calculate quantiles for a probability plot, and optionally show the plot.

    Generates a probability plot of sample data against the quantiles of a
    specified theoretical distribution (the normal distribution by default).
    `probplot` optionally calculates a best-fit line for the data and plots the
    results using Matplotlib or a given plot function.

    Parameters
    ----------
    x : array_like
        Sample/response data from which `probplot` creates the plot.
    sparams : tuple, optional
        Distribution-specific shape parameters (shape parameters plus location
        and scale).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name. The default is 'norm' for a
        normal probability plot.  Objects that look enough like a
        stats.distributions instance (i.e. they have a ``ppf`` method) are also
        accepted.
    fit : bool, optional
        Fit a least-squares regression (best-fit) line to the sample data if
        True (default).
    """
    plot : object, optional
        # 如果提供了此参数，将绘制概率图。
        # 如果提供了此参数且 `fit` 为 True，则还会绘制最小二乘拟合线。
        # `plot` 是一个对象，必须具有 "plot" 和 "text" 方法。
        # 可以使用 `matplotlib.pyplot` 模块或 Matplotlib 的 Axes 对象，
        # 或具有相同方法的自定义对象。
        # 默认为 None，表示不创建任何图形。
    rvalue : bool, optional
        # 如果 `plot` 被提供且 `fit` 为 True，则设置 `rvalue` 为 True
        # 将在图上包含确定系数（coefficient of determination）。
        # 默认为 False。

    Returns
    -------
    (osm, osr) : tuple of ndarrays
        # 返回理论分位数（osm，或顺序统计中值）和排序响应（osr）的元组。
        # `osr` 简单地是输入 `x` 的排序结果。
        # 关于 `osm` 的计算细节，请参见注释部分。
    (slope, intercept, r) : tuple of floats, optional
        # 如果 `probplot` 执行了最小二乘拟合，则返回包含拟合结果的元组。
        # `r` 是确定系数的平方根。
        # 如果 ``fit=False`` 且 ``plot=None``，则不返回此元组。

    Notes
    -----
    # 即使提供了 `plot`，`probplot` 也不会显示或保存图形；
    # 应在调用 `probplot` 后使用 ``plt.show()`` 或 ``plt.savefig('figname.png')``。

    # `probplot` 生成一个概率图，不应与 Q-Q 图或 P-P 图混淆。
    # Statsmodels 提供了更广泛的此类功能，参见 ``statsmodels.api.ProbPlot``。

    # 用于理论分位数（概率图的横轴）的公式是 Filliben 的估计：
    # ::
    #
    #     quantiles = dist.ppf(val), for
    #
    #             0.5**(1/n),                  for i = n
    #       val = (i - 0.3175) / (n + 0.365),  for i = 2, ..., n-1
    #             1 - 0.5**(1/n),              for i = 1
    #
    # 其中 ``i`` 表示第 i 个排序值，``n`` 是总值的数量。

    Examples
    --------
    # 导入必要的库
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> nsample = 100
    >>> rng = np.random.default_rng()

    # 小自由度的 t 分布示例：
    >>> ax1 = plt.subplot(221)
    >>> x = stats.t.rvs(3, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    # 大自由度的 t 分布示例：
    >>> ax2 = plt.subplot(222)
    >>> x = stats.t.rvs(25, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    # 两个正态分布的混合示例：
    >>> ax3 = plt.subplot(223)
    >>> x = stats.norm.rvs(loc=[0,5], scale=[1,1.5],
    ...                    size=(nsample//2,2), random_state=rng).ravel()
    >>> res = stats.probplot(x, plot=plt)

    # 标准正态分布示例：
    >>> ax4 = plt.subplot(224)
    >>> x = stats.norm.rvs(loc=0, scale=1, size=nsample, random_state=rng)
   `
# 将输入数据转换为 NumPy 数组
x = np.asarray(x)

# 如果输入数组为空，则根据 fit 参数返回对应结果
if x.size == 0:
    if fit:
        # 如果 fit 参数为 True，则返回空数组和对应的统计信息
        return (x, x), (np.nan, np.nan, 0.0)
    else:
        # 如果 fit 参数为 False，则直接返回空数组
        return x, x

# 计算等间隔的次序统计中位数
osm_uniform = _calc_uniform_order_statistic_medians(len(x))

# 解析并设置分布函数，支持子类
dist = _parse_dist_kw(dist, enforce_subclass=False)

# 如果未提供参数 sparams，则设置为空元组
if sparams is None:
    sparams = ()

# 如果 sparams 是标量，则转换为元组
if isscalar(sparams):
    sparams = (sparams,)

# 如果 sparams 不是元组，则转换为元组
if not isinstance(sparams, tuple):
    sparams = tuple(sparams)

# 计算分布的百分点函数，并传入参数 sparams
osm = dist.ppf(osm_uniform, *sparams)

# 对输入数据进行排序
osr = sort(x)

# 如果需要拟合数据
if fit:
    # 执行线性最小二乘拟合
    slope, intercept, r, prob, _ = _stats_py.linregress(osm, osr)

# 如果提供了绘图对象 plot
if plot is not None:
    # 绘制数据点
    plot.plot(osm, osr, 'bo')
    # 如果需要拟合数据，绘制拟合直线
    if fit:
        plot.plot(osm, slope*osm + intercept, 'r-')
    # 添加坐标轴标签和标题
    _add_axis_labels_title(plot, xlabel='Theoretical quantiles',
                           ylabel='Ordered Values',
                           title='Probability Plot')

    # 如果进行了拟合并且需要显示 R^2 值
    if fit and rvalue:
        # 计算文本显示位置
        xmin = amin(osm)
        xmax = amax(osm)
        ymin = amin(x)
        ymax = amax(x)
        posx = xmin + 0.70 * (xmax - xmin)
        posy = ymin + 0.01 * (ymax - ymin)
        # 在图中添加 R^2 值的文本
        plot.text(posx, posy, "$R^2=%1.4f$" % r**2)

# 根据 fit 参数返回对应结果
if fit:
    return (osm, osr), (slope, intercept, r)
else:
    return osm, osr
def ppcc_max(x, brack=(0.0, 1.0), dist='tukeylambda'):
    """Calculate the shape parameter that maximizes the PPCC.

    The probability plot correlation coefficient (PPCC) plot can be used
    to determine the optimal shape parameter for a one-parameter family
    of distributions. ``ppcc_max`` returns the shape parameter that would
    maximize the probability plot correlation coefficient for the given
    data to a one-parameter family of distributions.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : tuple, optional
        Triple (a,b,c) where (a<b<c). If bracket consists of two numbers (a, c)
        then they are assumed to be a starting interval for a downhill bracket
        search (see `scipy.optimize.brent`).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.

    Returns
    -------
    shape_value : float
        The shape parameter at which the probability plot correlation
        coefficient reaches its max value.

    See Also
    --------
    ppcc_plot, probplot, boxcox

    Notes
    -----
    The brack keyword serves as a starting point which is useful in corner
    cases. One can use a plot to obtain a rough visual estimate of the location
    for the maximum to start the search near it.

    References
    ----------
    .. [1] J.J. Filliben, "The Probability Plot Correlation Coefficient Test
           for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.
    .. [2] Engineering Statistics Handbook, NIST/SEMATEC,
           https://www.itl.nist.gov/div898/handbook/eda/section3/ppccplot.htm

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5:

    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Generate the PPCC plot for this data with the Weibull distribution.

    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax)

    We calculate the value where the shape should reach its maximum and a
    red line is drawn there. The line should coincide with the highest
    point in the PPCC graph.

    >>> cmax = stats.ppcc_max(x, brack=(c/2, 2*c), dist='weibull_min')
    >>> ax.axvline(cmax, color='r')
    >>> plt.show()

    """
    # Parse the distribution parameter to ensure it is in a valid format
    dist = _parse_dist_kw(dist)
    # Calculate the order statistic medians for a uniform distribution
    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    # Sort the input data array
    osr = sort(x)

    # this function computes the x-axis values of the probability plot
    #  and computes a linear regression (including the correlation)
    #  and returns 1-r so that a minimization function maximizes the
    #  correlation
    # 定义一个临时函数 `tempfunc`，接受形状 `shape`、均值 `mi`、`yvals` 和函数 `func` 作为参数
    def tempfunc(shape, mi, yvals, func):
        # 使用 `func` 函数生成 `xvals`
        xvals = func(mi, shape)
        # 使用 `_stats_py.pearsonr` 计算 `xvals` 和 `yvals` 的皮尔逊相关系数 `r` 和对应的概率 `prob`
        r, prob = _stats_py.pearsonr(xvals, yvals)
        # 返回 1 减去皮尔逊相关系数 `r` 的值
        return 1 - r

    # 使用 `optimize.brent` 函数优化 `tempfunc` 函数，设置 `brack` 参数为 `brack`，传入额外参数 `osm_uniform`、`osr` 和 `dist.ppf`
    return optimize.brent(tempfunc, brack=brack,
                          args=(osm_uniform, osr, dist.ppf))
# 定义函数 ppcc_plot，用于计算和可选地绘制概率图相关系数（PPCC）图

"""Calculate and optionally plot probability plot correlation coefficient.

The probability plot correlation coefficient (PPCC) plot can be used to
determine the optimal shape parameter for a one-parameter family of
distributions.  It cannot be used for distributions without shape
parameters (like the normal distribution) or with multiple shape parameters.

By default a Tukey-Lambda distribution (`stats.tukeylambda`) is used. A
Tukey-Lambda PPCC plot interpolates from long-tailed to short-tailed
distributions via an approximately normal one, and is therefore
particularly useful in practice.
"""

# 导入必要的库和模块
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def ppcc_plot(x, a, b, dist='tukeylambda', plot=None, N=80):
    """Calculate and optionally plot probability plot correlation coefficient.

    The probability plot correlation coefficient (PPCC) plot can be used to
    determine the optimal shape parameter for a one-parameter family of
    distributions.  It cannot be used for distributions without shape
    parameters (like the normal distribution) or with multiple shape parameters.

    By default a Tukey-Lambda distribution (`stats.tukeylambda`) is used. A
    Tukey-Lambda PPCC plot interpolates from long-tailed to short-tailed
    distributions via an approximately normal one, and is therefore
    particularly useful in practice.

    Parameters
    ----------
    x : array_like
        Input array.
    a, b : scalar
        Lower and upper bounds of the shape parameter to use.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.
    plot : object, optional
        If given, plots PPCC against the shape parameter.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `a` to `b`).

    Returns
    -------
    svals : ndarray
        The shape values for which `ppcc` was calculated.
    ppcc : ndarray
        The calculated probability plot correlation coefficient values.

    See Also
    --------
    ppcc_max, probplot, boxcox_normplot, tukeylambda

    References
    ----------
    J.J. Filliben, "The Probability Plot Correlation Coefficient Test for
    Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5, and plot the histogram of the data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Take a look at the histogram of the data.

    >>> fig1, ax = plt.subplots(figsize=(9, 4))
    >>> ax.hist(x, bins=50)
    >>> ax.set_title('Histogram of x')
    >>> plt.show()

    Now we explore this data with a PPCC plot as well as the related
    probability plot and Box-Cox normplot.  A red line is drawn where we
    expect the PPCC value to be maximal (at the shape parameter ``c``
    used above):

    >>> fig2 = plt.figure(figsize=(12, 4))
    >>> ax1 = fig2.add_subplot(1, 3, 1)
    >>> ax2 = fig2.add_subplot(1, 3, 2)
    >>> ax3 = fig2.add_subplot(1, 3, 3)
"""
    >>> res = stats.probplot(x, plot=ax1)
    >>> res = stats.boxcox_normplot(x, -4, 4, plot=ax2)
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax3)
    >>> ax3.axvline(c, color='r')
    >>> plt.show()

    """
    # 如果上限 `b` 小于或等于下限 `a`，则抛出值错误异常
    if b <= a:
        raise ValueError("`b` has to be larger than `a`.")

    # 在区间 [a, b] 内生成均匀分布的 `N` 个样本值
    svals = np.linspace(a, b, num=N)
    # 创建一个与 `svals` 大小相同的空数组 `ppcc`
    ppcc = np.empty_like(svals)
    # 对于 `svals` 中的每个索引 `k` 和相应的值 `sval`
    for k, sval in enumerate(svals):
        # 使用 `probplot` 函数计算 `x` 和 `sval` 之间的概率图及其拟合信息
        _, r2 = probplot(x, sval, dist=dist, fit=True)
        # 将拟合信息的最后一个值存储到 `ppcc` 数组中的相应位置 `k`
        ppcc[k] = r2[-1]

    # 如果提供了 `plot` 参数
    if plot is not None:
        # 在 `plot` 上绘制 `svals` 和 `ppcc` 的散点图
        plot.plot(svals, ppcc, 'x')
        # 调用辅助函数 `_add_axis_labels_title` 添加轴标签和标题
        _add_axis_labels_title(plot, xlabel='Shape Values',
                               ylabel='Prob Plot Corr. Coef.',
                               title=f'({dist}) PPCC Plot')

    # 返回 `svals` 和 `ppcc` 数组
    return svals, ppcc
# 计算给定对数值数组的均值的对数
def _log_mean(logx):
    return special.logsumexp(logx, axis=0) - np.log(len(logx))

# 计算给定对数值数组的方差的对数
def _log_var(logx):
    # 计算对数均值
    logmean = _log_mean(logx)
    # 创建一个复数数组，用于计算修正值
    pij = np.full_like(logx, np.pi * 1j, dtype=np.complex128)
    # 计算修正的对数均值
    logxmu = special.logsumexp([logx, logmean + pij], axis=0)
    # 计算方差的对数
    return np.real(special.logsumexp(2 * logxmu, axis=0)) - np.log(len(logx))

# Box-Cox 变换的对数似然函数
def boxcox_llf(lmb, data):
    r"""The boxcox log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Box-Cox 变换的参数。详细信息请参见 `boxcox`。
    data : array_like
        计算 Box-Cox 对数似然函数的数据。如果 `data` 是多维的，似然函数沿第一个轴计算。

    Returns
    -------
    llf : float or ndarray
        给定 `lmb`，`data` 的 Box-Cox 对数似然函数。对于一维 `data` 返回浮点数，否则返回数组。

    See Also
    --------
    boxcox, probplot, boxcox_normplot, boxcox_normmax

    Notes
    -----
    Box-Cox 对数似然函数在这里定义为

    .. math::

        llf = (\lambda - 1) \sum_i(\log(x_i)) -
              N/2 \log(\sum_i (y_i - \bar{y})^2 / N),

    其中 ``y`` 是经过 Box-Cox 变换的输入数据 ``x``。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    生成一些随机变量并计算它们的 Box-Cox 对数似然值，针对一系列 ``lmbda`` 值：

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, loc=10, size=1000, random_state=rng)
    >>> lmbdas = np.linspace(-2, 10)
    >>> llf = np.zeros(lmbdas.shape, dtype=float)
    >>> for ii, lmbda in enumerate(lmbdas):
    ...     llf[ii] = stats.boxcox_llf(lmbda, x)

    使用 `boxcox` 找到最佳的 `lmbda` 值：

    >>> x_most_normal, lmbda_optimal = stats.boxcox(x)

    绘制对数似然作为 `lmbda` 的函数。添加最佳 `lmbda` 作为水平线以检查是否真的是最优：

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(lmbdas, llf, 'b.-')
    >>> ax.axhline(stats.boxcox_llf(lmbda_optimal, x), color='r')
    >>> ax.set_xlabel('lmbda parameter')
    >>> ax.set_ylabel('Box-Cox log-likelihood')

    现在添加一些概率图，显示在对数似然最大化的地方，经过 `boxcox` 变换的数据看起来最接近正态分布：

    >>> locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    >>> for lmbda, loc in zip([-1, lmbda_optimal, 9], locs):
    ...     xt = stats.boxcox(x, lmbda=lmbda)
    ...     (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
    ...     ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
    ...     ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
    data = np.asarray(data)
    N = data.shape[0]
    if N == 0:
        return np.nan

将输入的数据转换为NumPy数组，并获取数据的行数N。如果N为0，则返回NaN（非数字），表示输入数据为空。


    logdata = np.log(data)

对输入数据进行对数变换，即取数据的自然对数。


    # Compute the variance of the transformed data.
    if lmb == 0:
        logvar = np.log(np.var(logdata, axis=0))
    else:
        # Transform without the constant offset 1/lmb.  The offset does
        # not affect the variance, and the subtraction of the offset can
        # lead to loss of precision.
        # Division by lmb can be factored out to enhance numerical stability.
        logx = lmb * logdata
        logvar = _log_var(logx) - 2 * np.log(abs(lmb))

计算转换后数据的方差。如果lmb（lambda值）为0，则计算对数转换后数据的方差的对数。否则，根据给定的lambda值对对数转换后的数据进行调整。这里的_log_var函数用于计算logx的方差的对数，减去2乘以lambda的绝对值的对数，以提高数值稳定性。


    return (lmb - 1) * np.sum(logdata, axis=0) - N/2 * logvar

根据计算得到的方差logvar，返回计算结果。结果是一个数值，表示根据给定的lambda值对数据进行变换后的计算结果。
# 计算 Box-Cox 变换的置信区间上下界
def _boxcox_conf_interval(x, lmax, alpha):
    # 计算临界值 fac，作为目标函数值的阈值
    fac = 0.5 * distributions.chi2.ppf(1 - alpha, 1)
    # 计算目标值 target，用于 rootfunc 函数
    target = boxcox_llf(lmax, x) - fac

    # 定义用于寻找根的函数 rootfunc
    def rootfunc(lmbda, data, target):
        return boxcox_llf(lmbda, data) - target

    # 寻找使得 rootfunc 函数为零的正数端点
    newlm = lmax + 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm += 0.1
        N += 1

    # 若循环超过500次仍未找到端点，则抛出异常
    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    # 使用 Brent 方法寻找正端点的具体值 lmplus
    lmplus = optimize.brentq(rootfunc, lmax, newlm, args=(x, target))

    # 寻找使得 rootfunc 函数为零的负数端点
    newlm = lmax - 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm -= 0.1
        N += 1

    # 若循环超过500次仍未找到端点，则抛出异常
    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    # 使用 Brent 方法寻找负端点的具体值 lmminus
    lmminus = optimize.brentq(rootfunc, newlm, lmax, args=(x, target))
    # 返回找到的置信区间的下界 lmminus 和上界 lmplus
    return lmminus, lmplus
    # 将输入数据转换为 NumPy 数组
    x = np.asarray(x)

    # 如果 lmbda 参数不为 None，则进行单一的 Box-Cox 变换
    if lmbda is not None:  # single transformation
        return special.boxcox(x, lmbda)

    # 如果数据不是一维的，则抛出数值错误异常
    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")

    # 如果数据为空，则直接返回空数组
    if x.size == 0:
        return x

    # 如果数据中所有元素都相同，则抛出数值错误异常
    if np.all(x == x[0]):
        raise ValueError("Data must not be constant.")

    # 如果数据中存在非正数元素，则抛出数值错误异常
    if np.any(x <= 0):
        raise ValueError("Data must be positive.")

    # 如果 lmbda=None，则找到最大化对数似然函数的 lmbda 值
    lmax = boxcox_normmax(x, method='mle', optimizer=optimizer)

    # 进行 Box-Cox 变换
    y = boxcox(x, lmax)

    # 如果 alpha 为 None，则返回变换后的数据和找到的 lmbda 值
    if alpha is None:
        return y, lmax
    else:
        # 如果 alpha 不为 None，则计算置信区间
        interval = _boxcox_conf_interval(x, lmax, alpha)
        return y, lmax, interval
# 定义一个函数 `_boxcox_inv_lmbda`，用于计算 Box-Cox 变换的参数 lambda 的逆运算
def _boxcox_inv_lmbda(x, y):
    # 计算 Box-Cox 变换的 lambda 参数，给定 x 和 y
    num = special.lambertw(-(x ** (-1 / y)) * np.log(x) / y, k=-1)
    # 返回实数部分的 lambda 值
    return np.real(-num / np.log(x) - 1 / y)


# 定义一个类 `_BigFloat`，用于表示一个大浮点数对象
class _BigFloat:
    # 返回字符串 "BIG_FLOAT"，用于对象的字符串表示
    def __repr__(self):
        return "BIG_FLOAT"


# 定义函数 `boxcox_normmax`，用于计算输入数据的最佳 Box-Cox 变换参数
def boxcox_normmax(
    x, brack=None, method='pearsonr', optimizer=None, *, ymax=_BigFloat()
):
    """Compute optimal Box-Cox transform parameter for input data.

    Parameters
    ----------
    x : array_like
        输入数组。所有条目必须为正的、有限的、实数。
    brack : 2-tuple, optional, default (-2.0, 2.0)
         默认的 `optimize.brent` 求解器的向下斜率搜索的起始区间。
         请注意，这在大多数情况下并不关键；最终结果允许超出此区间。
         如果传递了 `optimizer`，则 `brack` 必须为 None。
    method : str, optional
        确定最佳变换参数（`boxcox` `lmbda` 参数）的方法。选项包括：

        'pearsonr'  (默认)
            最大化 `y = boxcox(x)` 与期望值 `y` 之间的 Pearson 相关系数，
            如果 `x` 是正态分布的话。

        'mle'
            最大化对数似然 `boxcox_llf`。这是 `boxcox` 中使用的方法。

        'all'
            使用所有可用的优化方法，并返回所有结果。
            有助于比较不同的方法。
    optimizer : callable, optional
        `optimizer` 是一个可调用对象，接受一个参数：

        fun : callable
            要最小化的目标函数。`fun` 接受一个参数，即 Box-Cox 变换参数 `lmbda`，
            并返回在提供的参数处的函数值（例如，负对数似然）。
            `optimizer` 的任务是找到最小化 `fun` 的 `lmbda` 的值。

        并返回一个对象，例如 `scipy.optimize.OptimizeResult` 的实例，
        其中包含 `x` 属性中的最优 `lmbda` 值。

        有关更多信息，请参见下面的示例或 `scipy.optimize.minimize_scalar` 的文档。
    ymax : float, optional
        无约束的最佳变换参数可能导致 Box-Cox 变换后的数据具有极端的幅度，
        甚至会溢出。此参数约束 MLE 优化，以使变换后的 `x` 的幅度不超过 `ymax`。
        默认值是输入数据类型的最大值。如果设置为无穷大，
        `boxcox_normmax` 将返回无约束的最佳 lambda。
        当 `method='pearsonr'` 时忽略此参数。

    Returns
    -------
    maxlog : float or ndarray
        找到的最佳变换参数。对于 `method='all'`，返回数组而不是标量。

    See Also
    --------
    """
    x = np.asarray(x)
    # 将输入数据转换为 NumPy 数组

    if not np.all(np.isfinite(x) & (x >= 0)):
        # 检查 x 是否包含全部是有限的正实数，如果不是则抛出异常
        message = ("The `x` argument of `boxcox_normmax` must contain "
                   "only positive, finite, real numbers.")
        raise ValueError(message)

    end_msg = "exceed specified `ymax`."
    if isinstance(ymax, _BigFloat):
        # 如果 ymax 是 _BigFloat 类型，调整 ymax 的值以避免溢出
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # 10000 是一个安全因子，因为 `special.boxcox` 会过早溢出
        ymax = np.finfo(dtype).max / 10000
        end_msg = f"overflow in {dtype}."
    elif ymax <= 0:
        # 如果 ymax 小于等于 0，则抛出异常
        raise ValueError("`ymax` must be strictly positive")

    # 如果没有指定 optimizer，则使用默认的 'brent' 优化器
    if optimizer is None:

        # 如果未指定 brack 参数，默认设置为 (-2.0, 2.0)
        if brack is None:
            brack = (-2.0, 2.0)

        # 定义内部函数 _optimizer，使用 optimize.brent 进行优化
        def _optimizer(func, args):
            return optimize.brent(func, args=args, brack=brack)

    # 否则检查 optimizer 是否有效
    else:
        # 如果optimizer不是可调用对象，则抛出数值错误
        if not callable(optimizer):
            raise ValueError("`optimizer` must be a callable")

        # 如果brack不是None，则抛出数值错误，要求当`optimizer`参数存在时，`brack`必须为None
        if brack is not None:
            raise ValueError("`brack` must be None if `optimizer` is given")

        # `optimizer`应返回一个`OptimizeResult`对象，这里获取优化问题的解决方案。
        def _optimizer(func, args):
            # 封装函数，以便能够接受单个参数x
            def func_wrapped(x):
                return func(x, *args)
            # 调用优化器，获取最优解
            return getattr(optimizer(func_wrapped), 'x', None)

    def _pearsonr(x):
        # 计算长度为len(x)的均匀分布的顺序统计中位数
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        # 计算正态分布的分位数
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            # 此函数计算概率图的x轴值，并计算线性回归（包括相关性），返回`1 - r`以便最小化函数能够最大化相关性。
            y = boxcox(samps, lmbda)
            yvals = np.sort(y)
            # 计算Pearson相关系数及其显著性水平
            r, prob = _stats_py.pearsonr(xvals, yvals)
            return 1 - r

        # 调用优化器，使用_eval_pearsonr函数进行优化
        return _optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):
        def _eval_mle(lmb, data):
            # 要最小化的函数
            return -boxcox_llf(lmb, data)

        # 调用优化器，使用_eval_mle函数进行优化
        return _optimizer(_eval_mle, args=(x,))

    def _all(x):
        # 创建一个长度为2的数组，其中存储了两个函数的结果
        maxlog = np.empty(2, dtype=float)
        # 计算并存储Pearson相关系数函数的结果
        maxlog[0] = _pearsonr(x)
        # 计算并存储最大似然估计函数的结果
        maxlog[1] = _mle(x)
        return maxlog

    # 创建方法字典，将方法名映射到相应的函数
    methods = {'pearsonr': _pearsonr,
               'mle': _mle,
               'all': _all}
    # 如果指定的方法不在方法字典的键中，则抛出数值错误
    if method not in methods.keys():
        raise ValueError(f"Method {method} not recognized.")

    # 根据指定的方法名获取相应的函数
    optimfunc = methods[method]

    # 调用相应方法函数，并获取结果
    res = optimfunc(x)

    # 如果结果为None，则说明optimizer函数的返回值中不包含所需的`lmbda`属性，抛出数值错误
    if res is None:
        message = ("The `optimizer` argument of `boxcox_normmax` must return "
                   "an object containing the optimal `lmbda` in attribute `x`.")
        raise ValueError(message)
    # 如果 ymax 不是正无穷，则调整最终的 lambda 值
    elif not np.isinf(ymax):
        # 获取数组 x 的最大值和最小值
        xmax, xmin = np.max(x), np.min(x)
        
        # 根据最小值和最大值的情况确定 x_treme 的取值
        if xmin >= 1:
            x_treme = xmax
        elif xmax <= 1:
            x_treme = xmin
        else:
            # 当 xmin < 1 < xmax 时，比较两个 boxcox 变换后的值来确定 indicator
            indicator = special.boxcox(xmax, res) > abs(special.boxcox(xmin, res))
            if isinstance(res, np.ndarray):
                indicator = indicator[1]  # 选择与 'mle' 对应的值
            # 根据 indicator 选择 xmax 或 xmin 作为 x_treme 的值
            x_treme = xmax if indicator else xmin

        # 判断是否需要进行 lambda 约束以避免超过 ymax
        mask = abs(special.boxcox(x_treme, res)) > ymax
        if np.any(mask):
            # 发出警告信息，说明返回的 lambda 值是为了确保转换后的数据最大值或最小值不超过指定的 ymax
            message = (
                f"The optimal lambda is {res}, but the returned lambda is the "
                f"constrained optimum to ensure that the maximum or the minimum "
                f"of the transformed data does not " + end_msg
            )
            warnings.warn(message, stacklevel=2)

            # 返回约束后的 lambda 值，以确保转换不会导致溢出或超过指定的 ymax
            constrained_res = _boxcox_inv_lmbda(x_treme, ymax * np.sign(x_treme - 1))

            # 如果 res 是数组，则将约束后的 lambda 值赋给相应的位置
            if isinstance(res, np.ndarray):
                res[mask] = constrained_res
            else:
                res = constrained_res

    # 返回最终的 lambda 值（可能被约束过）
    return res
# 计算 Box-Cox 或 Yeo-Johnson 正态性图的参数，并可选地显示它。

# 如果方法为 'boxcox'，设置标题为 'Box-Cox Normality Plot'，并指定变换函数为 boxcox
# 如果方法不为 'boxcox'，设置标题为 'Yeo-Johnson Normality Plot'，并指定变换函数为 yeojohnson
def _normplot(method, x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox or Yeo-Johnson normality plot,
    optionally show it.

    See `boxcox_normplot` or `yeojohnson_normplot` for details.
    """

    # 将输入 x 转换为 NumPy 数组
    x = np.asarray(x)
    # 如果 x 为空数组，直接返回空数组
    if x.size == 0:
        return x

    # 检查 lb 是否大于 la，如果不是则引发 ValueError
    if lb <= la:
        raise ValueError("`lb` has to be larger than `la`.")

    # 如果方法为 'boxcox' 并且 x 中有小于等于 0 的值，引发 ValueError
    if method == 'boxcox' and np.any(x <= 0):
        raise ValueError("Data must be positive.")

    # 在区间 [la, lb] 内生成 N 个等间距的 lambda 值
    lmbdas = np.linspace(la, lb, num=N)
    # 初始化一个数组 ppcc，用于存储每个 lambda 值对应的概率图相关系数的平方根
    ppcc = lmbdas * 0.0
    # 遍历每个 lambda 值及其索引
    for i, val in enumerate(lmbdas):
        # 对 x 应用指定的变换函数（boxcox 或 yeojohnson），得到变换后的数据 z
        z = transform_func(x, lmbda=val)
        # 使用 probplot 计算 z 对应正态分布的概率图，并获取相关系数
        _, (_, _, r) = probplot(z, dist='norm', fit=True)
        # 将相关系数的平方根存储到 ppcc 数组中对应的位置
        ppcc[i] = r

    # 如果 plot 不为 None，则在 plot 上绘制 lambda 值与概率图相关系数的关系
    if plot is not None:
        plot.plot(lmbdas, ppcc, 'x')
        # 调用 _add_axis_labels_title 函数为绘图添加轴标签和标题
        _add_axis_labels_title(plot, xlabel='$\\lambda$',
                               ylabel='Prob Plot Corr. Coef.',
                               title=title)

    # 返回 lambda 值数组 lmbdas 和概率图相关系数数组 ppcc
    return lmbdas, ppcc
    Generate some non-normally distributed data, and create a Box-Cox plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    # 从对数伽马分布中生成500个数据点，并做平移使均值为5，存储在变量x中

    >>> fig = plt.figure()
    # 创建一个新的图形对象

    >>> ax = fig.add_subplot(111)
    # 在图形对象上添加一个子图，使用1x1网格的第一个子图，存储在变量ax中

    >>> prob = stats.boxcox_normplot(x, -20, 20, plot=ax)
    # 使用Box-Cox规范图绘制非正态分布数据x的正态性检验图，指定lambda范围为-20到20，并将结果绘制在ax子图上，返回概率对象prob

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.boxcox(x)
    # 对数据x进行Box-Cox变换，返回变换后的数据和最佳lambda值maxlog

    >>> ax.axvline(maxlog, color='r')
    # 在ax子图中绘制一条垂直线，表示最佳lambda值maxlog，线的颜色为红色

    >>> plt.show()
    # 显示绘制的图形

    """
    return _normplot('boxcox', x, la, lb, plot, N)
    # 调用_normplot函数进行Box-Cox变换的正态性检验，返回其结果
# 返回通过 Yeo-Johnson 功率变换转换后的数据集
def yeojohnson(x, lmbda=None):
    r"""Return a dataset transformed by a Yeo-Johnson power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Should be 1-dimensional.
    lmbda : float, optional
        If ``lmbda`` is ``None``, find the lambda that maximizes the
        log-likelihood function and return it as the second output argument.
        Otherwise the transformation is done for the given value.

    Returns
    -------
    yeojohnson: ndarray
        Yeo-Johnson power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.

    See Also
    --------
    probplot, yeojohnson_normplot, yeojohnson_normmax, yeojohnson_llf, boxcox

    Notes
    -----
    The Yeo-Johnson transform is given by::

        y = ((x + 1)**lmbda - 1) / lmbda,                for x >= 0, lmbda != 0
            log(x + 1),                                  for x >= 0, lmbda = 0
            -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda),  for x < 0, lmbda != 2
            -log(-x + 1),                                for x < 0, lmbda = 2

    Unlike `boxcox`, `yeojohnson` does not require the input data to be
    positive.

    .. versionadded:: 1.2.0

    References
    ----------
    I. Yeo and R.A. Johnson, "A New Family of Power Transformations to
    Improve Normality or Symmetry", Biometrika 87.4 (2000):

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `yeojohnson` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, lmbda = stats.yeojohnson(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Yeo-Johnson transformation')

    >>> plt.show()

    """
    # 将输入数组转换为 ndarray 类型
    x = np.asarray(x)
    # 如果数组大小为 0，直接返回空数组
    if x.size == 0:
        return x

    # 如果输入数组的数据类型是复数，抛出异常
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError('Yeo-Johnson transformation is not defined for '
                         'complex numbers.')

    # 如果输入数组的数据类型是整数，将其转换为 float64 类型
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)

    # 如果 lmbda 不为 None，则使用给定的 lmbda 进行 Yeo-Johnson 变换
    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)

    # 如果 lmbda=None，找到最大化对数似然函数的 lmbda 值
    lmax = yeojohnson_normmax(x)
    # 使用找到的 lmax 对输入数组进行 Yeo-Johnson 变换
    y = _yeojohnson_transform(x, lmax)

    return y, lmax


# 返回通过给定参数 lmbda 进行 Yeo-Johnson 功率变换后的数据集
def _yeojohnson_transform(x, lmbda):
    """Returns `x` transformed by the Yeo-Johnson power transform with given
    parameter `lmbda`.
    """
    # 根据输入数组 x 的数据类型确定输出数组的数据类型，如果 x 的数据类型是浮点数则保持不变，否则使用 np.float64 类型
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    # 创建一个和 x 具有相同形状和数据类型的全零数组
    out = np.zeros_like(x, dtype=dtype)
    # 创建一个布尔掩码数组，标记 x 中大于等于 0 的元素位置
    pos = x >= 0  # binary mask

    # 当 x 中的元素大于等于 0 时
    if abs(lmbda) < np.spacing(1.):
        # 对于满足条件的元素，计算 np.log1p(x[pos]) 并赋值给 out[pos]
        out[pos] = np.log1p(x[pos])
    else:  # 当 lmbda 不等于 0 时
        # 更稳定的计算方式：((x + 1) ** lmbda - 1) / lmbda
        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda

    # 当 x 中的元素小于 0 时
    if abs(lmbda - 2) > np.spacing(1.):
        # 对于不满足条件的元素，计算 -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda) 并赋值给 out[~pos]
        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
    else:  # 当 lmbda 等于 2 时
        # 对于不满足条件的元素，计算 -np.log1p(-x[~pos]) 并赋值给 out[~pos]
        out[~pos] = -np.log1p(-x[~pos])

    # 返回计算结果数组 out
    return out
    # 将输入数据转换为 NumPy 数组
    data = np.asarray(data)
    # 获取数据的样本数
    n_samples = data.shape[0]

    # 如果样本数为 0，则返回 NaN
    if n_samples == 0:
        return np.nan

    # 对数据进行 Yeo-Johnson 变换
    trans = _yeojohnson_transform(data, lmb)
    # 计算变换后数据的方差
    trans_var = trans.var(axis=0)
    # 创建一个与 trans_var 大小相同的空数组
    loglike = np.empty_like(trans_var)
    # 检查是否存在方差过小的情况，避免 np.log 报 RuntimeWarning
    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny

    # 对于方差过小的情况，将对应的 loglike 设为无穷大
    loglike[tiny_variance] = np.inf

    # 对于方差正常的情况，计算对数似然值：
    # 1. 计算对数似然的负半部分
    loglike[~tiny_variance] = (
        -n_samples / 2 * np.log(trans_var[~tiny_variance]))

    # 2. 添加正半部分的对数似然，这部分由 lambda 参数 (lmb - 1) 控制
    #    使用 np.sign(data) * np.log1p(np.abs(data)) 计算每列数据的对数
    #    然后对每列求和
    loglike[~tiny_variance] += (
        (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(axis=0))

    # 返回计算好的对数似然值数组
    return loglike
# 定义一个函数，用于计算 Yeo-Johnson 变换的最优参数
def yeojohnson_normmax(x, brack=None):
    """Compute optimal Yeo-Johnson transform parameter.

    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : 2-tuple, optional
        The starting interval for a downhill bracket search with
        `optimize.brent`. Note that this is in most cases not critical; the
        final result is allowed to be outside this bracket. If None,
        `optimize.fminbound` is used with bounds that avoid overflow.

    Returns
    -------
    maxlog : float
        The optimal transform parameter found.

    See Also
    --------
    yeojohnson, yeojohnson_llf, yeojohnson_normplot

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some data and determine optimal ``lmbda``

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> lmax = stats.yeojohnson_normmax(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax, color='r')

    >>> plt.show()

    """
    # 定义一个内部函数，用于计算给定数据和变换参数下的负对数似然函数值
    def _neg_llf(lmbda, data):
        # 计算 Yeo-Johnson 变换下的对数似然函数值
        llf = yeojohnson_llf(lmbda, data)
        # 将由于转换空间中方差很小而导致的似然函数值为 inf 的情况标记为 -inf，以便排除这些情况
        llf[np.isinf(llf)] = -np.inf
        # 返回负对数似然函数值
        return -llf
    # 设置numpy的错误状态，忽略无效值的警告
    with np.errstate(invalid='ignore'):
        # 检查数组x中的所有值是否有非有限值，若有则抛出值错误异常
        if not np.all(np.isfinite(x)):
            raise ValueError('Yeo-Johnson 输入必须是有限值。')
        # 如果数组x中所有值都为0，则返回1.0
        if np.all(x == 0):
            return 1.0
        # 如果brack参数不为空，则使用optimize.brent函数进行优化
        if brack is not None:
            return optimize.brent(_neg_llf, brack=brack, args=(x,))
        # 将输入转换为numpy数组形式
        x = np.asarray(x)
        # 确定数组元素的数据类型，如果是浮点类型则使用当前类型，否则使用np.float64
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # 计算可安全变换的最大观察值的20倍的对数加1
        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
        # 计算浮点数精度范围的一半，用于安全计算变换数据的方差
        log_eps = np.log(np.finfo(dtype).eps)
        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
        # 根据Yeo-Johnson变换在最小和最大浮点数指数上的逆向近似计算边界
        # 这取决于预期观察到的最大数据。详细内容见引用[1]
        lb = log_tiny_float / log1p_max_x
        ub = log_max_float / log1p_max_x
        # 如果所有数据都为负数，则转换边界值
        if np.all(x < 0):
            lb, ub = 2 - ub, 2 - lb
        # 如果有些数据为负数，则选择转换边界值的最大和最小值
        elif np.any(x < 0):
            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
        # 设置与optimize.brent函数相匹配的容差值
        tol_brent = 1.48e-08
        # 使用optimize.fminbound函数寻找_neg_llf的最小值
        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)
# 定义一个函数 yeojohnson_normplot，用于计算 Yeo-Johnson 正态性检验的参数，并可选地展示其结果。
# 此函数允许生成一个 Yeo-Johnson 正态性检验图，展示最适合的变换参数，使分布接近正态分布。

def yeojohnson_normplot(x, la, lb, plot=None, N=80):
    """Compute parameters for a Yeo-Johnson normality plot, optionally show it.

    A Yeo-Johnson normality plot shows graphically what the best
    transformation parameter is to use in `yeojohnson` to obtain a
    distribution that is close to normal.

    Parameters
    ----------
    x : array_like
        Input array.
    la, lb : scalar
        The lower and upper bounds for the ``lmbda`` values to pass to
        `yeojohnson` for Yeo-Johnson transformations. These are also the
        limits of the horizontal axis of the plot if that is generated.
    plot : object, optional
        If given, plots the quantiles and least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `la` to `lb`).

    Returns
    -------
    lmbdas : ndarray
        The ``lmbda`` values for which a Yeo-Johnson transform was done.
    ppcc : ndarray
        Probability Plot Correlelation Coefficient, as obtained from `probplot`
        when fitting the Box-Cox transformed input `x` against a normal
        distribution.

    See Also
    --------
    probplot, yeojohnson, yeojohnson_normmax, yeojohnson_llf, ppcc_max

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by
    `boxcox_normplot`; ``plt.show()`` or ``plt.savefig('figname.png')``
    should be used after calling `probplot`.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some non-normally distributed data, and create a Yeo-Johnson plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -20, 20, plot=ax)

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.yeojohnson(x)
    >>> ax.axvline(maxlog, color='r')

    >>> plt.show()

    """
    return _normplot('yeojohnson', x, la, lb, plot, N)


# 使用 namedtuple 定义一个名为 ShapiroResult 的命名元组，包含两个字段 'statistic' 和 'pvalue'
ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))


# 使用装饰器 @_axis_nan_policy_factory 对 shapiro 函数进行修饰，处理 NaN 值的策略
# 该函数执行 Shapiro-Wilk 正态性检验，用于检验数据是否来自正态分布。
@_axis_nan_policy_factory(ShapiroResult, n_samples=1, too_small=2, default_axis=None)
def shapiro(x):
    r"""Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data. Must contain at least three observations.

    Returns
    -------
    statistic : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.

    See Also
    --------
    anderson : The Anderson-Darling test for normality
    kstest : The Kolmogorov-Smirnov test for goodness of fit.

    Notes
    -----
    The algorithm used is described in [4]_ but censoring parameters as
    described are not implemented. For N > 5000 the W test statistic is
    accurate, but the p-value may not be.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
           :doi:`10.18434/M32189`
    .. [2] Shapiro, S. S. & Wilk, M.B, "An analysis of variance test for
           normality (complete samples)", Biometrika, 1965, Vol. 52,
           pp. 591-611, :doi:`10.2307/2333709`
    .. [3] Razali, N. M. & Wah, Y. B., "Power comparisons of Shapiro-Wilk,
           Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests", Journal
           of Statistical Modeling and Analytics, 2011, Vol. 2, pp. 21-33.
    .. [4] Royston P., "Remark AS R94: A Remark on Algorithm AS 181: The
           W-test for Normality", 1995, Applied Statistics, Vol. 44,
           :doi:`10.2307/2986146`
    .. [5] Phipson B., and Smyth, G. K., "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn", Statistical Applications in Genetics and Molecular Biology,
           2010, Vol.9, :doi:`10.2202/1544-6115.1585`
    .. [6] Panagiotakos, D. B., "The value of p-value in biomedical
           research", The Open Cardiovascular Medicine Journal, 2008, Vol.2,
           pp. 97-99, :doi:`10.2174/1874192400802010097`

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [2]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])

    The normality test of [1]_ and [2]_ begins by computing a statistic based
    on the relationship between the observations and the expected order
    statistics of a normal distribution.

    >>> from scipy import stats
    >>> res = stats.shapiro(x)
    >>> res.statistic
    0.7888147830963135

    The value of this statistic tends to be high (close to 1) for samples drawn
    from a normal distribution.

    The test is performed by comparing the observed value of the statistic
    against the null distribution: the distribution of statistic values formed
    under the null hypothesis that the weights were drawn from a normal
    distribution. For this normality test, the null distribution is not easy to
    calculate exactly, so it is usually approximated by Monte Carlo methods,
    that is, drawing many samples of the same size as ``x`` from a normal
    distribution and computing the values of the statistic for each.

    >>> def statistic(x):
    ...     # Get only the `shapiro` statistic; ignore its p-value
    ...     return stats.shapiro(x).statistic
    >>> ref = stats.monte_carlo_test(x, stats.norm.rvs, statistic,
    ...                              alternative='less')
    # 进行蒙特卡洛模拟检验，使用正态分布生成随机变量作为对比，返回检验结果的引用

    >>> import matplotlib.pyplot as plt
    # 导入 matplotlib 的 pyplot 模块，并简写为 plt

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    # 创建一个新的图形窗口和子图，指定尺寸为 8x5

    >>> bins = np.linspace(0.65, 1, 50)
    # 生成一个包含等间距数据的数组，范围从 0.65 到 1，总共有 50 个数据点

    >>> def plot(ax):  # we'll reuse this
    ...     ax.hist(ref.null_distribution, density=True, bins=bins)
    ...     ax.set_title("Shapiro-Wilk Test Null Distribution \n"
    ...                  "(Monte Carlo Approximation, 11 Observations)")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    # 定义一个函数 plot，用于绘制直方图和设置图表标题、坐标轴标签

    >>> plot(ax)
    # 调用 plot 函数，传入之前创建的子图 ax，并绘制直方图

    >>> plt.show()
    # 显示所有创建的图形窗口

    The comparison is quantified by the p-value: the proportion of values in
    the null distribution less than or equal to the observed value of the
    statistic.

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    # 创建一个新的图形窗口和子图，指定尺寸为 8x5

    >>> plot(ax)
    # 调用 plot 函数，传入新创建的子图 ax，并绘制直方图

    >>> annotation = (f'p-value={res.pvalue:.6f}\n(highlighted area)')
    # 创建一个注释文本，包含 p-value 值并指示高亮区域

    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    # 创建一个包含箭头样式和其他属性的字典 props

    >>> _ = ax.annotate(annotation, (0.75, 0.1), (0.68, 0.7), arrowprops=props)
    # 在子图 ax 上添加注释，指定注释文本、起始点、终点和箭头样式

    >>> i_extreme = np.where(bins <= res.statistic)[0]
    # 找到直方图中小于或等于统计量 res.statistic 的所有 bin 的索引

    >>> for i in i_extreme:
    ...     ax.patches[i].set_color('C1')
    # 将符合条件的直方图 bin 着色为 'C1'（橙色）

    >>> plt.xlim(0.65, 0.9)
    # 设置 x 轴的显示范围为 0.65 到 0.9

    >>> plt.ylim(0, 4)
    # 设置 y 轴的显示范围为 0 到 4

    >>> plt.show
    # 显示所有创建的图形窗口（此处应为 plt.show()，表示显示图形）

    >>> res.pvalue
    0.006703833118081093
    # 输出检验结果的 p-value 值

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence *for* the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [5]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    """
    x = np.ravel(x).astype(np.float64)
    # 将 x 展平为一维数组并转换为 np.float64 类型

    N = len(x)
    # 计算数组 x 的长度

    if N < 3:
        raise ValueError("Data must be at least length 3.")
    # 如果数组长度小于 3，则抛出 ValueError 异常

    a = zeros(N//2, dtype=np.float64)
    # 创建一个长度为 N//2 的零数组，数据类型为 np.float64

    init = 0
    # 初始化变量 init 为 0

    y = sort(x)
    # 对数组 x 进行排序，赋值给 y

    y -= x[N//2]  # subtract the median (or a nearby value); see gh-15777
    # 将 y 减去中位数（或附近的值），以减少数据集的偏斜性

    w, pw, ifault = swilk(y, a, init)
    # 使用 Shapiro-Wilk 测试函数 swilk 对数据集 y 进行正态性检验，并返回统计量 w、p-value pw 和 ifault

    if ifault not in [0, 2]:
        warnings.warn("scipy.stats.shapiro: Input data has range zero. The"
                      " results may not be accurate.", stacklevel=2)
    # 如果 ifault 不在 [0, 2] 中，发出警告：输入数据范围为零，结果可能不准确

    if N > 5000:
        warnings.warn("scipy.stats.shapiro: For N > 5000, computed p-value "
                      f"may not be accurate. Current N is {N}.",
                      stacklevel=2)
    # 如果数据集长度 N 大于 5000，发出警告：计算得到的 p-value 可能不准确，当前 N 为 {N}

    # `w` and `pw` are always Python floats, which are double precision.
    # We want to ensure that they are NumPy floats, so until dtypes are
    # 返回一个包含 np.float64 类型的 w 和 pw 的 ShapiroResult 对象
    # 这种方式比使用 np.array([w, pw]) 更快速。
    return ShapiroResult(np.float64(w), np.float64(pw))
# Values from Stephens, M A, "EDF Statistics for Goodness of Fit and
#             Some Comparisons", Journal of the American Statistical
#             Association, Vol. 69, Issue 347, Sept. 1974, pp 730-737
_Avals_norm = array([0.576, 0.656, 0.787, 0.918, 1.092])
# Values from Stephens, M A, "Goodness of Fit for the Extreme Value Distribution",
#             Biometrika, Vol. 64, Issue 3, Dec. 1977, pp 583-588.
_Avals_expon = array([0.922, 1.078, 1.341, 1.606, 1.957])
# Values from Stephens, M A, "Tests of Fit for the Logistic Distribution Based
#             on the Empirical Distribution Function.", Biometrika,
#             Vol. 66, Issue 3, Dec. 1979, pp 591-595.
_Avals_logistic = array([0.426, 0.563, 0.660, 0.769, 0.906, 1.010])
# Values from Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
#             Fit for the Three-Parameter Weibull Distribution"
#             Journal of the Royal Statistical Society.Series B(Methodological)
#             Vol. 56, No. 3 (1994), pp. 491-500, table 1. Keys are c*100
_Avals_weibull = [[0.292, 0.395, 0.467, 0.522, 0.617, 0.711, 0.836, 0.931],
                  [0.295, 0.399, 0.471, 0.527, 0.623, 0.719, 0.845, 0.941],
                  [0.298, 0.403, 0.476, 0.534, 0.631, 0.728, 0.856, 0.954],
                  [0.301, 0.408, 0.483, 0.541, 0.640, 0.738, 0.869, 0.969],
                  [0.305, 0.414, 0.490, 0.549, 0.650, 0.751, 0.885, 0.986],
                  [0.309, 0.421, 0.498, 0.559, 0.662, 0.765, 0.902, 1.007],
                  [0.314, 0.429, 0.508, 0.570, 0.676, 0.782, 0.923, 1.030],
                  [0.320, 0.438, 0.519, 0.583, 0.692, 0.802, 0.947, 1.057],
                  [0.327, 0.448, 0.532, 0.598, 0.711, 0.824, 0.974, 1.089],
                  [0.334, 0.469, 0.547, 0.615, 0.732, 0.850, 1.006, 1.125],
                  [0.342, 0.472, 0.563, 0.636, 0.757, 0.879, 1.043, 1.167]]
_Avals_weibull = np.array(_Avals_weibull)
_cvals_weibull = np.linspace(0, 0.5, 11)

_get_As_weibull = interpolate.interp1d(_cvals_weibull, _Avals_weibull.T,
                                       kind='linear', bounds_error=False,
                                       fill_value=_Avals_weibull[-1])

def _weibull_fit_check(params, x):
    # Refine the fit returned by `weibull_min.fit` to ensure that the first
    # order necessary conditions are satisfied. If not, raise an error.
    # Here, use `m` for the shape parameter to be consistent with [7]
    # and avoid confusion with `c` as defined in [7].
    n = len(x)
    m, u, s = params

    def dnllf_dm(m, u):
        # Partial w.r.t. shape w/ optimal scale. See [7] Equation 5.
        xu = x-u
        return (1/m - (xu**m*np.log(xu)).sum()/(xu**m).sum()
                + np.log(xu).sum()/n)

    def dnllf_du(m, u):
        # Partial w.r.t. loc w/ optimal scale. See [7] Equation 6.
        xu = x-u
        return (m-1)/m*(xu**-1).sum() - n*(xu**(m-1)).sum()/(xu**m).sum()
    def get_scale(m, u):
        # 计算尺度参数
        # 计算尺度参数对应的和函数
        return ((x-u)**m/n).sum()**(1/m)

    def dnllf(params):
        # 对负对数似然函数的参数求偏导，即最大似然估计的一阶必要条件
        return [dnllf_dm(*params), dnllf_du(*params)]

    suggestion = ("最大似然估计在三参数威布尔分布中较为困难。考虑使用自定义的拟合优度检验，"
                  "可以使用 `scipy.stats.monte_carlo_test` 进行。")

    if np.allclose(u, np.min(x)) or m < 1:
        # 当估计结果的位置参数等于数据的最小值，或者形状参数小于1时，
        # 根据 [7] 中提供的临界值似乎无法控制第一类错误率。因此报错。
        message = ("最大似然估计收敛到一个解，其中位置参数等于数据的最小值，"
                   "形状参数小于2，或者两者兼有。[7] 中的临界值表格不包括此类情况。"
                   " " + suggestion)
        raise ValueError(message)

    try:
        # 优化最大似然估计的解，验证一阶必要条件是否满足。
        # 如果满足，[7] 中提供的临界值似乎是可靠的。
        with np.errstate(over='raise', invalid='raise'):
            res = optimize.root(dnllf, params[:-1])

        message = ("最大似然估计的一阶条件解未能找到："
                   f"{res.message}. `anderson` 无法继续。 " + suggestion)
        if not res.success:
            raise ValueError(message)

    except (FloatingPointError, ValueError) as e:
        message = ("在拟合威布尔分布到数据时发生错误，因此 `anderson` 无法继续。"
                   " " + suggestion)
        raise ValueError(message) from e

    m, u = res.x
    s = get_scale(m, u)
    return m, u, s
# 创建名为 AndersonResult 的元组扩展对象，包含 statistic、critical_values 和 significance_level 三个属性，以及 fit_result 一个属性
AndersonResult = _make_tuple_bunch('AndersonResult',
                                   ['statistic', 'critical_values',
                                    'significance_level'], ['fit_result'])

# 定义 Anderson-Darling 测试函数，用于检验数据是否来自特定分布
def anderson(x, dist='norm'):
    """Anderson-Darling test for data coming from a particular distribution.

    The Anderson-Darling test tests the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.
    For the Anderson-Darling test, the critical values depend on
    which distribution is being tested against.  This function works
    for normal, exponential, logistic, weibull_min, or Gumbel (Extreme Value
    Type I) distributions.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1', 'weibull_min'}, optional
        The type of distribution to test against.  The default is 'norm'.
        The names 'extreme1', 'gumbel_l' and 'gumbel' are synonyms for the
        same distribution.

    Returns
    -------
    result : AndersonResult
        An object with the following attributes:

        statistic : float
            The Anderson-Darling test statistic.
        critical_values : list
            The critical values for this distribution.
        significance_level : list
            The significance levels for the corresponding critical values
            in percents.  The function returns critical values for a
            differing set of significance levels depending on the
            distribution that is being tested against.
        fit_result : `~scipy.stats._result_classes.FitResult`
            An object containing the results of fitting the distribution to
            the data.

    See Also
    --------
    kstest : The Kolmogorov-Smirnov test for goodness-of-fit.

    Notes
    -----
    Critical values provided are for the following significance levels:

    normal/exponential
        15%, 10%, 5%, 2.5%, 1%
    logistic
        25%, 10%, 5%, 2.5%, 1%, 0.5%
    gumbel_l / gumbel_r
        25%, 10%, 5%, 2.5%, 1%
    weibull_min
        50%, 25%, 15%, 10%, 5%, 2.5%, 1%, 0.5%

    If the returned statistic is larger than these critical values then
    for the corresponding significance level, the null hypothesis that
    the data come from the chosen distribution can be rejected.
    The returned statistic is referred to as 'A2' in the references.

    For `weibull_min`, maximum likelihood estimation is known to be
    challenging. If the test returns successfully, then the first order
    conditions for a maximum likehood estimate have been verified and
    the critical values correspond relatively well to the significance levels,
    provided that the sample is sufficiently large (>10 observations [7]).
    However, for some data - especially data with no left tail - `anderson`
    is likely to result in an error message. In this case, consider
    """
    """Performing a custom goodness of fit test using
    `scipy.stats.monte_carlo_test`.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and
           Some Comparisons, Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [3] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
           Statistics with Unknown Parameters, Annals of Statistics, Vol. 4,
           pp. 357-369.
    .. [4] Stephens, M. A. (1977). Goodness of Fit for the Extreme Value
           Distribution, Biometrika, Vol. 64, pp. 583-588.
    .. [5] Stephens, M. A. (1977). Goodness of Fit with Special Reference
           to Tests for Exponentiality , Technical Report No. 262,
           Department of Statistics, Stanford University, Stanford, CA.
    .. [6] Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution
           Based on the Empirical Distribution Function, Biometrika, Vol. 66,
           pp. 591-595.
    .. [7] Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
           Fit for the Three-Parameter Weibull Distribution"
           Journal of the Royal Statistical Society.Series B(Methodological)
           Vol. 56, No. 3 (1994), pp. 491-500, Table 0.

    Examples
    --------
    Test the null hypothesis that a random sample was drawn from a normal
    distribution (with unspecified mean and standard deviation).

    >>> import numpy as np
    >>> from scipy.stats import anderson
    >>> rng = np.random.default_rng()
    >>> data = rng.random(size=35)
    >>> res = anderson(data)
    >>> res.statistic
    0.8398018749744764
    >>> res.critical_values
    array([0.527, 0.6  , 0.719, 0.839, 0.998])
    >>> res.significance_level
    array([15. , 10. ,  5. ,  2.5,  1. ])

    The value of the statistic (barely) exceeds the critical value associated
    with a significance level of 2.5%, so the null hypothesis may be rejected
    at a significance level of 2.5%, but not at a significance level of 1%.

    """ # numpy/numpydoc#87  # noqa: E501
    # 将分布名称转换为小写
    dist = dist.lower()
    # 如果分布名称是 'extreme1' 或 'gumbel'，则转换为 'gumbel_l'
    if dist in {'extreme1', 'gumbel'}:
        dist = 'gumbel_l'
    # 允许的分布名称集合
    dists = {'norm', 'expon', 'gumbel_l',
             'gumbel_r', 'logistic', 'weibull_min'}

    # 如果指定的分布不在允许的集合中，抛出 ValueError 异常
    if dist not in dists:
        raise ValueError(f"Invalid distribution; dist must be in {dists}.")
    # 对观测数据进行排序
    y = sort(x)
    # 计算观测数据的均值
    xbar = np.mean(x, axis=0)
    # 观测数据的样本大小
    N = len(y)
    # 如果分布是正态分布
    if dist == 'norm':
        # 计算观测数据的标准差
        s = np.std(x, ddof=1, axis=0)
        # 计算标准化的观测数据
        w = (y - xbar) / s
        # 保存拟合参数
        fit_params = xbar, s
        # 计算正态分布的对数累积分布函数和对数生存函数
        logcdf = distributions.norm.logcdf(w)
        logsf = distributions.norm.logsf(w)
        # 签名水平数组
        sig = array([15, 10, 5, 2.5, 1])
        # 计算临界值数组
        critical = around(_Avals_norm / (1.0 + 4.0/N - 25.0/N/N), 3)
    # 如果分布为指数分布
    elif dist == 'expon':
        # 计算参数 w
        w = y / xbar
        # 设置拟合参数
        fit_params = 0, xbar
        # 计算指数分布的对数累积分布函数和对数生存函数
        logcdf = distributions.expon.logcdf(w)
        logsf = distributions.expon.logsf(w)
        # 设置临界值的数组
        sig = array([15, 10, 5, 2.5, 1])
        # 计算指数分布下的临界值
        critical = around(_Avals_expon / (1.0 + 0.6/N), 3)
    
    # 如果分布为 logistic 分布
    elif dist == 'logistic':
        # 定义根函数
        def rootfunc(ab, xj, N):
            a, b = ab
            tmp = (xj - a) / b
            tmp2 = exp(tmp)
            val = [np.sum(1.0/(1+tmp2), axis=0) - 0.5*N,
                   np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
            return array(val)
        
        # 初始解
        sol0 = array([xbar, np.std(x, ddof=1, axis=0)])
        # 使用 fsolve 求解根函数的根
        sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
        # 计算参数 w
        w = (y - sol[0]) / sol[1]
        # 设置拟合参数
        fit_params = sol
        # 计算 logistic 分布的对数累积分布函数和对数生存函数
        logcdf = distributions.logistic.logcdf(w)
        logsf = distributions.logistic.logsf(w)
        # 设置临界值的数组
        sig = array([25, 10, 5, 2.5, 1, 0.5])
        # 计算 logistic 分布下的临界值
        critical = around(_Avals_logistic / (1.0 + 0.25/N), 3)
    
    # 如果分布为 gumbel_r 分布
    elif dist == 'gumbel_r':
        # 使用 gumbel_r 分布的 fit 函数拟合参数 xbar 和 s
        xbar, s = distributions.gumbel_r.fit(x)
        # 计算参数 w
        w = (y - xbar) / s
        # 设置拟合参数
        fit_params = xbar, s
        # 计算 gumbel_r 分布的对数累积分布函数和对数生存函数
        logcdf = distributions.gumbel_r.logcdf(w)
        logsf = distributions.gumbel_r.logsf(w)
        # 设置临界值的数组
        sig = array([25, 10, 5, 2.5, 1])
        # 计算 gumbel_r 分布下的临界值
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)
    
    # 如果分布为 gumbel_l 分布
    elif dist == 'gumbel_l':
        # 使用 gumbel_l 分布的 fit 函数拟合参数 xbar 和 s
        xbar, s = distributions.gumbel_l.fit(x)
        # 计算参数 w
        w = (y - xbar) / s
        # 设置拟合参数
        fit_params = xbar, s
        # 计算 gumbel_l 分布的对数累积分布函数和对数生存函数
        logcdf = distributions.gumbel_l.logcdf(w)
        logsf = distributions.gumbel_l.logsf(w)
        # 设置临界值的数组
        sig = array([25, 10, 5, 2.5, 1])
        # 计算 gumbel_l 分布下的临界值
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)
    
    # 如果分布为 weibull_min 分布
    elif dist == 'weibull_min':
        # 提示消息，针对样本数量少于 10 的情况
        message = ("Critical values of the test statistic are given for the "
                   "asymptotic distribution. These may not be accurate for "
                   "samples with fewer than 10 observations. Consider using "
                   "`scipy.stats.monte_carlo_test`.")
        if N < 10:
            # 发出警告
            warnings.warn(message, stacklevel=2)
        
        # 使用 weibull_min 分布的 fit 函数拟合参数 m, loc, scale
        m, loc, scale = distributions.weibull_min.fit(y)
        # 进行参数检查和调整
        m, loc, scale = _weibull_fit_check((m, loc, scale), y)
        # 设置拟合参数
        fit_params = m, loc, scale
        # 计算 weibull_min 分布的对数累积分布函数和对数生存函数
        logcdf = stats.weibull_min(*fit_params).logcdf(y)
        logsf = stats.weibull_min(*fit_params).logsf(y)
        # 计算参数 c，与文献 [7] 中使用的术语一致
        c = 1 / m  # m and c are as used in [7]
        # 设置临界值的数组
        sig = array([0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995])
        # 根据参数 c 计算 weibull_min 分布下的临界值
        critical = _get_As_weibull(c)
        # 对临界值进行四舍五入处理，保留三位小数
        critical = np.round(critical + 0.0005, decimals=3)

    # 计算 Anderson-Darling 统计量 A2
    i = arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)

    # 设置消息，指示 `anderson` 成功拟合了数据的分布
    message = '`anderson` successfully fit the distribution to the data.'
    # 创建一个优化结果对象，设置成功标志为True，并传入消息作为参数
    res = optimize.OptimizeResult(success=True, message=message)
    # 将fit_params转换为NumPy数组，并赋值给res对象的x属性
    res.x = np.array(fit_params)
    # 创建一个拟合结果对象FitResult，使用getattr动态获取distributions中的dist对象作为分布类型
    # 参数为y，设定discrete为False，同时传入优化结果res作为参数
    fit_result = FitResult(getattr(distributions, dist), y,
                           discrete=False, res=res)
    # 返回AndersonResult对象，包含A2，critical，sig作为参数，并附带拟合结果fit_result
    return AndersonResult(A2, critical, sig, fit_result=fit_result)
# 定义了一个函数 _anderson_ksamp_midrank，用于计算 Scholz 和 Stephens 论文中的 A2akN 统计量
def _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 7 of Scholz and Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2aKN : float
        The A2aKN statistics of Scholz and Stephens 1987.

    """
    # 初始化 A2akN 统计量为 0
    A2akN = 0.
    # 找到 Zstar 在 Z 中左边界的索引
    Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    # 如果总体观测值数等于唯一观测值数，lj 设为 1
    if N == Zstar.size:
        lj = 1.
    else:
        # 否则，计算右边界索引减去左边界索引得到 lj
        lj = Z.searchsorted(Zstar, 'right') - Z_ssorted_left
    # 计算 Bj 值
    Bj = Z_ssorted_left + lj / 2.
    # 遍历每个样本
    for i in arange(0, k):
        # 对样本 i 进行排序
        s = np.sort(samples[i])
        # 找到样本排序后在 Zstar 中的右边界索引
        s_ssorted_right = s.searchsorted(Zstar, side='right')
        # 计算 Mij
        Mij = s_ssorted_right.astype(float)
        # 计算 fij
        fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
        Mij -= fij / 2.
        # 计算内部部分
        inner = lj / float(N) * (N*Mij - Bj*n[i])**2 / (Bj*(N - Bj) - N*lj/4.)
        # 将内部部分加到 A2akN 统计量中
        A2akN += inner.sum() / n[i]
    # 最终计算 A2akN 统计量并乘以 (N - 1) / N
    A2akN *= (N - 1.) / N
    # 返回计算结果 A2akN
    return A2akN


# 定义了一个函数 _anderson_ksamp_right，用于计算 Scholz 和 Stephens 论文中的 A2kN 统计量
def _anderson_ksamp_right(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 6 of Scholz & Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2KN : float
        The A2KN statistics of Scholz and Stephens 1987.

    """
    # 初始化 A2kN 统计量为 0
    A2kN = 0.
    # 计算 lj
    lj = Z.searchsorted(Zstar[:-1], 'right') - Z.searchsorted(Zstar[:-1], 'left')
    # 计算 Bj
    Bj = lj.cumsum()
    # 遍历每个样本
    for i in arange(0, k):
        # 对样本 i 进行排序
        s = np.sort(samples[i])
        # 找到样本排序后在 Zstar[:-1] 中的右边界索引
        Mij = s.searchsorted(Zstar[:-1], side='right')
        # 计算内部部分
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / (Bj * (N - Bj))
        # 将内部部分加到 A2kN 统计量中
        A2kN += inner.sum() / n[i]
    # 返回计算结果 A2kN
    return A2kN


# 定义了一个函数 anderson_ksamp，用于进行 k-样本 Anderson-Darling 测试
def anderson_ksamp(samples, midrank=True, *, method=None):
    """The Anderson-Darling test for k-samples.

    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.
    method : PermutationMethod, optional
        Defines the method used to compute the p-value. If `method` is an
        instance of `PermutationMethod`, the p-value is computed using
        `scipy.stats.permutation_test` with the provided configuration options
        and other appropriate settings. Otherwise, the p-value is interpolated
        from tabulated values.

    Returns
    -------
    res : Anderson_ksampResult
        An object containing attributes:

        statistic : float
            Normalized k-sample Anderson-Darling test statistic.
        critical_values : array
            The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
            0.5%, 0.1%.
        pvalue : float
            The approximate p-value of the test. If `method` is not
            provided, the value is floored / capped at 0.1% / 25%.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.

    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test

    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.

    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> res = stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(loc=0.5, size=30)])
    >>> res.statistic, res.pvalue
    (1.974403288713695, 0.04991293614572478)
    >>> res.critical_values
    # 定义一个包含七个浮点数的数组
    array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546])

    # 当返回的检验值大于5%临界值（1.961）时，可以拒绝两个随机样本来自同一分布的原假设，但不能在2.5%水平上。插值提供了一个近似的p值为4.99%。
    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    p-value of 4.99%.

    # 定义一个包含三个随机数数组的列表，并计算它们的Anderson-Darling K-Sample检验
    >>> samples = [rng.normal(size=50), rng.normal(size=30),
    ...            rng.normal(size=20)]
    >>> res = stats.anderson_ksamp(samples)
    >>> res.statistic, res.pvalue
    (-0.29103725200789504, 0.25)
    >>> res.critical_values
    # 返回的检验统计量和p值
    array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
      4.07210043, 5.56419101])

    # 对于来自相同分布的三个样本，无法拒绝原假设。报告的p值（25%）已经被截断，可能不太准确（因为它对应的值为0.449，而统计量为-0.291）。
    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The reported p-value (25%) has been capped and
    may not be very accurate (since it corresponds to the value 0.449
    whereas the statistic is -0.291).

    # 在p值被截断或样本大小较小时，可能更准确的方法是使用置换检验。
    In such cases where the p-value is capped or when sample sizes are
    small, a permutation test may be more accurate.

    # 定义一个PermutationMethod对象，设置重抽样次数为9999，随机状态为rng，并计算Anderson-Darling K-Sample检验的p值。
    >>> method = stats.PermutationMethod(n_resamples=9999, random_state=rng)
    >>> res = stats.anderson_ksamp(samples, method=method)
    >>> res.pvalue
    0.5254

    """
    # 计算样本数量k
    k = len(samples)
    # 如果k小于2，抛出数值错误
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    # 将每个样本转换为numpy数组并存储在列表中
    samples = list(map(np.asarray, samples))
    # 对样本中所有元素进行排序，并且将它们展平成一个一维数组
    Z = np.sort(np.hstack(samples))
    # 获取展平后唯一值并排序的数组
    N = Z.size
    Zstar = np.unique(Z)
    # 如果唯一值小于2，抛出数值错误
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    # 计算每个样本的大小并存储在数组中
    n = np.array([sample.size for sample in samples])
    # 如果任何样本大小为0，抛出数值错误
    if np.any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    # 如果使用midrank方法，则选择_midrank函数，否则选择_right函数
    if midrank:
        A2kN_fun = _anderson_ksamp_midrank
    else:
        A2kN_fun = _anderson_ksamp_right
    # 调用选定的函数计算A2kN值
    A2kN = A2kN_fun(samples, Z, Zstar, k, n, N)

    # 定义一个统计函数，返回选定函数的结果
    def statistic(*samples):
        return A2kN_fun(samples, Z, Zstar, k, n, N)

    # 如果提供了method参数，则使用置换检验来计算p值
    if method is not None:
        res = stats.permutation_test(samples, statistic, **method._asdict(),
                                     alternative='greater')

    # 计算H的值，表示权重的和
    H = (1. / n).sum()
    # 计算hs_cs数组，表示累积和
    hs_cs = (1. / arange(N - 1, 1, -1)).cumsum()
    # 计算h的值，表示hs_cs的最后一个元素加1
    h = hs_cs[-1] + 1
    # 计算g的值，表示hs_cs除以arange(2, N)的和
    g = (hs_cs / arange(2, N)).sum()

    # 计算a、b、c、d的值，表示系数
    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    # 计算sigmasq的值，表示方差
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    # 计算m的值，表示k-1
    m = k - 1
    # 计算A2的值，表示标准化后的检验统计量
    A2 = (A2kN - m) / math.sqrt(sigmasq)

    # b_i值是Scholz和Stephens 1987年表2中的插值系数
    # 定义b0和b1数组，分别表示插值系数
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    # 定义常数数组 b2，表示特定系数
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    # 计算临界值 critical，根据给定的 b0、b1、m 参数进行计算
    critical = b0 + b1 / math.sqrt(m) + b2 / m

    # 定义显著性水平数组 sig，包含常见的显著性水平值
    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    # 判断 A2 是否小于临界值的最小值，并且 method 为 None
    if A2 < critical.min() and method is None:
        # 设置 p 为 sig 中的最大值
        p = sig.max()
        # 构建警告消息，指出 p 值被截断，建议指定 `method` 参数以获得更准确的结果
        msg = (f"p-value capped: true value larger than {p}. Consider "
               "specifying `method` "
               "(e.g. `method=stats.PermutationMethod()`.)")
        # 发出警告，显示警告消息并指定调用栈级别为 2
        warnings.warn(msg, stacklevel=2)
    # 判断 A2 是否大于临界值的最大值，并且 method 为 None
    elif A2 > critical.max() and method is None:
        # 设置 p 为 sig 中的最小值
        p = sig.min()
        # 构建警告消息，指出 p 值被设置为最小显著性水平，建议指定 `method` 参数以获得更准确的结果
        msg = (f"p-value floored: true value smaller than {p}. Consider "
               "specifying `method` "
               "(e.g. `method=stats.PermutationMethod()`.)")
        # 发出警告，显示警告消息并指定调用栈级别为 2
        warnings.warn(msg, stacklevel=2)
    # 如果 method 为 None，且不符合上述两个条件
    elif method is None:
        # 使用临界值和对数显著性水平进行二次多项式拟合
        pf = np.polyfit(critical, log(sig), 2)
        # 计算 A2 对应的 p 值
        p = math.exp(np.polyval(pf, A2))
    else:
        # 如果 method 不为 None，则使用 res 对象的 pvalue 作为 p 值
        p = res.pvalue if method is not None else p

    # 创建 Anderson_ksampResult 对象，包含 A2 值、临界值数组和计算得到的 p 值
    res = Anderson_ksampResult(A2, critical, p)
    # 设置结果对象的 significance_level 属性为 p 值，以确保向后兼容性
    res.significance_level = p
    # 返回构建好的结果对象
    return res
# 创建一个命名元组 `AnsariResult`，包含两个字段：`statistic` 和 `pvalue`
AnsariResult = namedtuple('AnsariResult', ('statistic', 'pvalue'))

class _ABW:
    """Ansari-Bradley W 统计量在零假设下的分布。"""
    # TODO: 计算考虑到并列情况的精确分布
    # 可以避免对超过频率一半的内容求和，
    # 但最初似乎不值得增加额外的复杂性

    def __init__(self):
        """最小化初始化器。"""
        self.m = None
        self.n = None
        self.astart = None
        self.total = None
        self.freqs = None

    def _recalc(self, n, m):
        """在必要时重新计算精确分布。"""
        if n != self.n or m != self.m:
            self.n, self.m = n, m
            # 当 m + n 是奇数时，分布不对称
            # n 是 x 的长度，m 是 y 的长度，尺度比率被定义为 x/y
            astart, a1, _ = gscale(n, m)
            self.astart = astart  # 统计量的最小值
            # 零假设下测试统计量的精确分布
            # 表达为频率/计数/整数，以保持精度。
            # 存储为浮点数以避免求和溢出。
            self.freqs = a1.astype(np.float64)
            self.total = self.freqs.sum()  # 可以从 m 和 n 计算得到
            # 概率质量为 self.freqs / self.total;

    def pmf(self, k, n, m):
        """概率质量函数。"""
        self._recalc(n, m)
        # 这里的约定是，k = 12.5 时的 PMF 与 k = 12 时相同，
        # -> 在并列情况下使用 `floor`。
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind] / self.total

    def cdf(self, k, n, m):
        """累积分布函数。"""
        self._recalc(n, m)
        # 考虑并列的空假设派生的分布是近似的。
        # 向下取整以避免第一类错误。
        ind = np.ceil(k - self.astart).astype(int)
        return self.freqs[:ind+1].sum() / self.total

    def sf(self, k, n, m):
        """生存函数。"""
        self._recalc(n, m)
        # 考虑并列的空假设派生的分布是近似的。
        # 向下取整以避免第一类错误。
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind:].sum() / self.total


# 为了更快速地重复调用带有 method='exact' 的 Ansari 测试，维护状态
_abw_state = _ABW()


@_axis_nan_policy_factory(AnsariResult, n_samples=2)
def ansari(x, y, alternative='two-sided'):
    """执行 Ansari-Bradley 测试，检验两个样本数据的尺度参数是否相等。

    Ansari-Bradley 测试 ([1]_, [2]_) 是一种非参数测试，
    用于检验从中抽取两个样本的分布尺度参数的相等性。
    零假设表明，分布在 `x` 下的尺度与 `y` 下的尺度的比率为 1。

    Parameters
    ----------
    x, y : array_like
        样本数据的数组。
    alternative : {'two-sided', 'less', 'greater'}, optional
        # 参数：alternative，可选项为 {'two-sided', 'less', 'greater'}，默认为 'two-sided'
        # 指定备择假设类型。可选值如下：

        # * 'two-sided': 表示比例尺的比率不等于1。
        # * 'less': 表示比例尺的比率小于1。
        # * 'greater': 表示比例尺的比率大于1。

        .. versionadded:: 1.7.0
        # 版本说明：新增于版本 1.7.0

    Returns
    -------
    statistic : float
        # 返回值：统计量，为浮点数
        # Ansari-Bradley 检验的统计量。
    pvalue : float
        # 返回值：p 值，为浮点数
        # 假设检验的 p 值。

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
        # 参见：fligner，用于 k 个方差相等性的非参数检验
    mood : A non-parametric test for the equality of two scale parameters
        # 参见：mood，用于两个尺度参数相等性的非参数检验

    Notes
    -----
    The p-value given is exact when the sample sizes are both less than
    55 and there are no ties, otherwise a normal approximation for the
    p-value is used.
        # 注意事项：
        # 当样本大小均小于55且无并列值时，给出的 p 值是精确的，否则使用正态近似的 p 值。

    References
    ----------
    .. [1] Ansari, A. R. and Bradley, R. A. (1960) Rank-sum tests for
           dispersions, Annals of Mathematical Statistics, 31, 1174-1189.
        # 参考文献：
        # .. [1] Ansari, A. R. 和 Bradley, R. A. (1960) Rank-sum tests for
        #        dispersions, Annals of Mathematical Statistics, 31, 1174-1189.

    .. [2] Sprent, Peter and N.C. Smeeton.  Applied nonparametric
           statistical methods.  3rd ed. Chapman and Hall/CRC. 2001.
           Section 5.8.2.
        # .. [2] Sprent, Peter 和 N.C. Smeeton.  应用非参数统计方法。
        #        第3版。Chapman and Hall/CRC. 2001。第5.8.2节。

    .. [3] Nathaniel E. Helwig "Nonparametric Dispersion and Equality
           Tests" at http://users.stat.umn.edu/~helwig/notes/npde-Notes.pdf
        # .. [3] Nathaniel E. Helwig “非参数分散和相等性检验”
        #        网址：http://users.stat.umn.edu/~helwig/notes/npde-Notes.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ansari
    >>> rng = np.random.default_rng()

    For these examples, we'll create three random data sets.  The first
    two, with sizes 35 and 25, are drawn from a normal distribution with
    mean 0 and standard deviation 2.  The third data set has size 25 and
    is drawn from a normal distribution with standard deviation 1.25.

    >>> x1 = rng.normal(loc=0, scale=2, size=35)
    >>> x2 = rng.normal(loc=0, scale=2, size=25)
    >>> x3 = rng.normal(loc=0, scale=1.25, size=25)

    First we apply `ansari` to `x1` and `x2`.  These samples are drawn
    from the same distribution, so we expect the Ansari-Bradley test
    should not lead us to conclude that the scales of the distributions
    are different.

    >>> ansari(x1, x2)
    AnsariResult(statistic=541.0, pvalue=0.9762532927399098)

    With a p-value close to 1, we cannot conclude that there is a
    significant difference in the scales (as expected).

    Now apply the test to `x1` and `x3`:

    >>> ansari(x1, x3)
    AnsariResult(statistic=425.0, pvalue=0.0003087020407974518)

    The probability of observing such an extreme value of the statistic
    under the null hypothesis of equal scales is only 0.03087%. We take this
    as evidence against the null hypothesis in favor of the alternative:
    the scales of the distributions from which the samples were drawn
    are not equal.

    We can use the `alternative` parameter to perform a one-tailed test.
    In the above example, the scale of `x1` is greater than `x3` and so
        # 我们可以使用 `alternative` 参数执行单侧检验。
        # 在上面的例子中，`x1` 的尺度大于 `x3`，因此
    the ratio of scales of `x1` and `x3` is greater than 1. This means
    that the p-value when ``alternative='greater'`` should be near 0 and
    hence we should be able to reject the null hypothesis:

    >>> ansari(x1, x3, alternative='greater')
    AnsariResult(statistic=425.0, pvalue=0.0001543510203987259)

    As we can see, the p-value is indeed quite low. Use of
    ``alternative='less'`` should thus yield a large p-value:

    >>> ansari(x1, x3, alternative='less')
    AnsariResult(statistic=425.0, pvalue=0.9998643258449039)

    """
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("'alternative' must be 'two-sided',"
                         " 'greater', or 'less'.")
    # 将输入的x和y转换为数组格式
    x, y = asarray(x), asarray(y)
    # 计算x和y的长度
    n = len(x)
    m = len(y)
    # 如果y的长度小于1，抛出异常
    if m < 1:
        raise ValueError("Not enough other observations.")
    # 如果x的长度小于1，抛出异常
    if n < 1:
        raise ValueError("Not enough test observations.")

    # 计算总样本数N
    N = m + n
    # 合并x和y为一个新的数组xy
    xy = r_[x, y]  # combine
    # 计算合并后数组的秩
    rank = _stats_py.rankdata(xy)
    # 计算对称秩
    symrank = amin(array((rank, N - rank + 1)), 0)
    # 计算AB统计量
    AB = np.sum(symrank[:n], axis=0)
    # 找出数组xy中唯一值
    uxy = unique(xy)
    # 检查是否有重复值
    repeats = (len(uxy) != len(xy))
    # 检查是否可以使用精确统计量
    exact = ((m < 55) and (n < 55) and not repeats)
    # 如果有重复值并且样本量小于55，给出警告
    if repeats and (m < 55 or n < 55):
        warnings.warn("Ties preclude use of exact statistic.", stacklevel=2)
    # 如果可以使用精确统计量
    if exact:
        if alternative == 'two-sided':
            # 计算双侧检验的p值
            pval = 2.0 * np.minimum(_abw_state.cdf(AB, n, m),
                                    _abw_state.sf(AB, n, m))
        elif alternative == 'greater':
            # 当比例大时，AB统计量较小，这与通常的计算相反
            pval = _abw_state.cdf(AB, n, m)
        else:
            # 计算alternative='less'时的p值
            pval = _abw_state.sf(AB, n, m)
        # 返回Ansari检验的结果
        return AnsariResult(AB, min(1.0, pval))

    # 否则使用正态近似计算
    if N % 2:  # N为奇数时
        mnAB = n * (N+1.0)**2 / 4.0 / N
        varAB = n * m * (N+1.0) * (3+N**2) / (48.0 * N**2)
    else:  # N为偶数时
        mnAB = n * (N+2.0) / 4.0
        varAB = m * n * (N+2) * (N-2.0) / 48 / (N-1.0)
    # 如果有重复值，调整方差估计
    if repeats:
        # 计算调整后的方差估计
        fac = np.sum(symrank**2, axis=0)
        if N % 2:  # N为奇数时
            varAB = m * n * (16*N*fac - (N+1)**4) / (16.0 * N**2 * (N-1))
        else:  # N为偶数时
            varAB = m * n * (16*fac - N*(N+2)**2) / (16.0 * N * (N-1))

    # AB较小表明x样本的离散程度较大
    # AB较大表明y样本的离散程度较大
    # 这与我们定义比例尺的方式相反，参见 [1]_
    z = (mnAB - AB) / sqrt(varAB)
    # 调用_get_pvalue函数计算p值
    pvalue = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    # 返回Ansari检验的结果
    return AnsariResult(AB[()], pvalue[()])
# 导入命名元组工具，用于创建 BartlettResult 结果对象，包含 statistic 和 pvalue 字段
BartlettResult = namedtuple('BartlettResult', ('statistic', 'pvalue'))

# 使用装饰器工厂函数来封装 bartlett 函数，使其具备处理轴上 NaN 值的能力，并返回 BartlettResult 对象
@_axis_nan_policy_factory(BartlettResult, n_samples=None)
def bartlett(*samples, axis=0):
    r"""Perform Bartlett's test for equal variances.

    Bartlett's test tests the null hypothesis that all input samples
    are from populations with equal variances.  For samples
    from significantly non-normal populations, Levene's test
    `levene` is more robust.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        arrays of sample data.  Only 1d arrays are accepted, they may have
        different lengths.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    levene : A robust parametric test for equality of k variances

    Notes
    -----
    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power
    ([3]_).

    References
    ----------
    .. [1]  https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm
    .. [2]  Snedecor, George W. and Cochran, William G. (1989), Statistical
              Methods, Eighth Edition, Iowa State University Press.
    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
    .. [4] Bartlett, M. S. (1937). Properties of Sufficiency and Statistical
           Tests. Proceedings of the Royal Society of London. Series A,
           Mathematical and Physical Sciences, Vol. 160, No.901, pp. 268-282.
    .. [5] C.I. BLISS (1952), The Statistics of Bioassay: With Special
           Reference to the Vitamins, pp 499-503,
           :doi:`10.1016/C2013-0-12584-6`.
    .. [6] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [7] Ludbrook, J., & Dudley, H. (1998). Why permutation tests are
           superior to t and F tests in biomedical research. The American
           Statistician, 52(2), 127-132.

    Examples
    --------
    In [5]_, the influence of vitamin C on the tooth growth of guinea pigs
    was investigated. In a control study, 60 subjects were divided into
    small dose, medium dose, and large dose groups that received
    daily doses of 0.5, 1.0 and 2.0 mg of vitamin C, respectively.
    After 42 days, the tooth growth was measured.
    """
    # 函数体内的具体实现逻辑会在实际代码中补充
    # 定义三个数组，记录三组牙齿生长的微米测量数据
    >>> import numpy as np
    >>> small_dose = np.array([
    ...     4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7,
    ...     15.2, 21.5, 17.6, 9.7, 14.5, 10, 8.2, 9.4, 16.5, 9.7
    ... ])
    >>> medium_dose = np.array([
    ...     16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5,
    ...     19.7, 23.3, 23.6, 26.4, 20, 25.2, 25.8, 21.2, 14.5, 27.3
    ... ])
    >>> large_dose = np.array([
    ...     23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5,
    ...     25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23
    ... ])

    # 使用 Bartlett 统计量检验对三组样本的方差是否存在差异敏感
    >>> from scipy import stats
    >>> res = stats.bartlett(small_dose, medium_dose, large_dose)
    >>> res.statistic
    0.6654670663030519

    # 当方差差异很大时，统计量的值通常较高
    # 我们可以通过将观察到的统计量的值与零假设下的统计量分布进行比较，来测试三组之间的方差是否相等
    # 在此测试中，零假设下的分布遵循卡方分布，如下所示
    >>> import matplotlib.pyplot as plt
    >>> k = 3  # 样本数量
    >>> dist = stats.chi2(df=k-1)
    >>> val = np.linspace(0, 5, 100)
    >>> pdf = dist.pdf(val)
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> def plot(ax):  # 我们将重复使用此函数
    ...     ax.plot(val, pdf, color='C0')
    ...     ax.set_title("Bartlett Test Null Distribution")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    ...     ax.set_xlim(0, 5)
    ...     ax.set_ylim(0, 1)
    >>> plot(ax)
    >>> plt.show()

    # 比较由 p 值量化，即空假设分布中大于等于观察统计量值的值的比例
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> pvalue = dist.sf(res.statistic)
    >>> annotation = (f'p-value={pvalue:.3f}\n(shaded area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (1.5, 0.22), (2.25, 0.3), arrowprops=props)
    >>> i = val >= res.statistic
    >>> ax.fill_between(val[i], y1=0, y2=pdf[i], color='C0')
    >>> plt.show()

    # 统计量的 p 值，如果 p 值较小，即从具有相同方差的分布中抽样数据以产生统计量极端值的概率较低，
    # 则可以将其作为反驳零假设，支持备择假设的证据：三组的方差不相等。注意：
    >>> res.pvalue
    0.71696121509966
    >>> def statistic(*samples):
    ...     return stats.bartlett(*samples).statistic
    定义一个函数 statistic，接受多个样本作为参数，返回 Bartlett 检验的统计量
    
    >>> ref = stats.permutation_test(
    ...     (small_dose, medium_dose, large_dose), statistic,
    ...     permutation_type='independent', alternative='greater'
    ... )
    进行独立分组的置换检验，检验三个样本 (small_dose, medium_dose, large_dose) 是否来自同一总体，检验统计量为上面定义的 statistic 函数，备择假设为大于关系
    
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    创建一个大小为 8x5 的图表
    
    >>> plot(ax)
    调用 plot 函数绘制图表
    
    >>> bins = np.linspace(0, 5, 25)
    生成一个从 0 到 5 的等间距分布的数组，共 25 个元素
    
    >>> ax.hist(
    ...     ref.null_distribution, bins=bins, density=True, facecolor="C1"
    ... )
    在 ax 图表上绘制 ref 的空置分布直方图，使用指定的 bins 和颜色"C1"，进行密度归一化
    
    >>> ax.legend(['aymptotic approximation\n(many observations)',
    ...            'randomized null distribution'])
    在 ax 图表上创建图例，包含两个标签
    
    >>> plot(ax)
    再次调用 plot 函数绘制图表
    
    >>> plt.show()
    显示图表
    
    >>> ref.pvalue  # randomized test p-value
    输出 ref 对象的随机化测试 p 值
    
    0.5387  # may vary
    具体的 p 值，可能会有所不同
    
    Note that there is significant disagreement between the p-value calculated
    here and the asymptotic approximation returned by `bartlett` above.
    在这里计算的 p 值与上面由 `bartlett` 返回的渐近近似存在显著差异。
    The statistical inferences that can be drawn rigorously from a permutation
    test are limited; nonetheless, they may be the preferred approach in many
    circumstances [7]_.
    从置换检验中严格推断出的统计推断是有限的；尽管如此，在许多情况下它们可能是首选方法。
    
    Following is another generic example where the null hypothesis would be
    rejected.
    接下来是另一个通用示例，其中原假设将被拒绝。
    
    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.
    测试列表 `a`, `b` 和 `c` 是否来自具有相等方差的总体。
    
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    定义三个列表 a, b, c 分别包含一组数据
    
    >>> stat, p = stats.bartlett(a, b, c)
    计算列表 a, b, c 的 Bartlett 检验统计量和 p 值
    
    >>> p
    输出计算得到的 p 值
    
    1.1254782518834628e-05
    非常小的 p 值表明这些总体不具有相等的方差。
    
    The very small p-value suggests that the populations do not have equal
    variances.
    非常小的 p 值表明这些总体不具有相等的方差。
    
    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:
    这并不令人惊讶，因为 `b` 的样本方差远大于 `a` 和 `c` 的样本方差。
    # 计算多个样本的 Bartlett 检验统计量和 p 值
    def bartlett_test(*samples, axis=None):
        # 使用传入的样本数据创建适合的数组命名空间
        xp = array_namespace(*samples)
    
        # 获取样本数量 k
        k = len(samples)
        # 如果样本数量小于 2，则抛出数值错误异常
        if k < 2:
            raise ValueError("Must enter at least two input sample vectors.")
    
        # 将样本广播到同一形状
        samples = _broadcast_arrays(samples, axis=axis, xp=xp)
        # 将每个样本的轴移动到末尾
        samples = [xp_moveaxis_to_end(sample, axis, xp=xp) for sample in samples]
    
        # 计算每个样本的最后一个维度的长度 Ni
        Ni = [xp.asarray(sample.shape[-1], dtype=sample.dtype) for sample in samples]
        # 将 Ni 广播到与样本形状的前 n-1 维相同
        Ni = [xp.broadcast_to(N, samples[0].shape[:-1]) for N in Ni]
        # 计算每个样本的方差 ssq
        ssq = [xp.var(sample, correction=1, axis=-1) for sample in samples]
    
        # 对 Ni 和 ssq 进行扩展以进行拼接
        Ni = [arr[xp.newaxis, ...] for arr in Ni]
        ssq = [arr[xp.newaxis, ...] for arr in ssq]
        Ni = xp.concat(Ni, axis=0)
        ssq = xp.concat(ssq, axis=0)
    
        # 计算总样本数 Ntot 和 spsq
        Ntot = xp.sum(Ni, axis=0)
        spsq = xp.sum((Ni - 1) * ssq, axis=0) / (Ntot - k)
    
        # 计算统计量的分子部分 numer 和分母部分 denom
        numer = (Ntot - k) * xp.log(spsq) - xp.sum((Ni - 1) * xp.log(ssq), axis=0)
        denom = 1 + 1 / (3 * (k - 1)) * ((xp.sum(1 / (Ni - 1), axis=0)) - 1 / (Ntot - k))
    
        # 计算 Bartlett 检验的统计量 T
        T = numer / denom
    
        # 使用自定义函数 _SimpleChi2 计算自由度为 k-1 的卡方分布
        chi2 = _SimpleChi2(xp.asarray(k - 1))
        # 调用 _get_pvalue 函数计算 p 值
        pvalue = _get_pvalue(T, chi2, alternative='greater', symmetric=False, xp=xp)
    
        # 如果 T 是标量，则转换为 Python 浮点数，否则保持原样
        T = T[()] if T.ndim == 0 else T
        # 如果 pvalue 是标量，则转换为 Python 浮点数，否则保持原样
        pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    
        # 返回 Bartlett 检验结果对象 BartlettResult(T, pvalue)
        return BartlettResult(T, pvalue)
# 定义一个命名元组 LeveneResult，包含统计量 statistic 和 p 值 pvalue
LeveneResult = namedtuple('LeveneResult', ('statistic', 'pvalue'))

# 装饰器函数 @_axis_nan_policy_factory，用于处理轴向 NaN 值的策略，返回 LeveneResult 类型的结果
@_axis_nan_policy_factory(LeveneResult, n_samples=None)
# 定义 Levene 测试函数，用于检验多个样本是否具有相等的方差
def levene(*samples, center='median', proportiontocut=0.05):
    r"""Perform Levene test for equal variances.

    The Levene test tests the null hypothesis that all input samples
    are from populations with equal variances.  Levene's test is an
    alternative to Bartlett's test `bartlett` in the case where
    there are significant deviations from normality.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample data, possibly with different lengths. Only one-dimensional
        samples are accepted.
    center : {'mean', 'median', 'trimmed'}, optional
        Which function of the data to use in the test.  The default
        is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    bartlett : A parametric test for equality of k variances in normal samples

    Notes
    -----
    Three variations of Levene's test are possible.  The possibilities
    and their recommended usages are:

      * 'median' : Recommended for skewed (non-normal) distributions>
      * 'mean' : Recommended for symmetric, moderate-tailed distributions.
      * 'trimmed' : Recommended for heavy-tailed distributions.

    The test version using the mean was proposed in the original article
    of Levene ([2]_) while the median and trimmed mean have been studied by
    Brown and Forsythe ([3]_), sometimes also referred to as Brown-Forsythe
    test.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
    .. [2] Levene, H. (1960). In Contributions to Probability and Statistics:
           Essays in Honor of Harold Hotelling, I. Olkin et al. eds.,
           Stanford University Press, pp. 278-292.
    .. [3] Brown, M. B. and Forsythe, A. B. (1974), Journal of the American
           Statistical Association, 69, 364-367
    .. [4] C.I. BLISS (1952), The Statistics of Bioassay: With Special
           Reference to the Vitamins, pp 499-503,
           :doi:`10.1016/C2013-0-12584-6`.
    .. [5] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [6] Ludbrook, J., & Dudley, H. (1998). Why permutation tests are
           superior to t and F tests in biomedical research. The American
           Statistician, 52(2), 127-132.

    Examples
    --------
    # 导入 NumPy 库，用于处理数组数据
    import numpy as np
    
    # 创建小剂量维生素 C 组的牙齿生长数据数组
    small_dose = np.array([
        4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7,
        15.2, 21.5, 17.6, 9.7, 14.5, 10, 8.2, 9.4, 16.5, 9.7
    ])
    
    # 创建中剂量维生素 C 组的牙齿生长数据数组
    medium_dose = np.array([
        16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5,
        19.7, 23.3, 23.6, 26.4, 20, 25.2, 25.8, 21.2, 14.5, 27.3
    ])
    
    # 创建大剂量维生素 C 组的牙齿生长数据数组
    large_dose = np.array([
        23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5,
        25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23
    ])
    
    # 导入 SciPy 库中的统计模块
    from scipy import stats
    
    # 进行 Levene 检验，用于检测三组样本方差是否有显著差异
    res = stats.levene(small_dose, medium_dose, large_dose)
    
    # 输出 Levene 检验的统计量
    res.statistic
    
    # 导入 Matplotlib 库，用于绘图
    import matplotlib.pyplot as plt
    
    # 设定 F 分布的自由度参数
    k, n = 3, 60   # 样本数和总观测数
    dist = stats.f(dfn=k-1, dfd=n-k)
    
    # 创建值范围
    val = np.linspace(0, 5, 100)
    
    # 计算 F 分布的概率密度函数
    pdf = dist.pdf(val)
    
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 定义绘图函数，用于绘制 F 分布曲线
    def plot(ax):
        ax.plot(val, pdf, color='C0')
        ax.set_title("Levene Test Null Distribution")
        ax.set_xlabel("statistic")
        ax.set_ylabel("probability density")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
    
    # 调用绘图函数绘制 F 分布曲线
    plot(ax)
    
    # 显示绘制的图表
    plt.show()
    
    # 获取 Levene 检验的 p 值
    res.pvalue
    
    # 创建新的图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 再次调用绘图函数绘制 F 分布曲线
    plot(ax)
    
    # 计算 p 值对应的 shaded area 并进行标注
    pvalue = dist.sf(res.statistic)
    annotation = (f'p-value={pvalue:.3f}\n(shaded area)')
    props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    
    # 添加注释箭头
    _ = ax.annotate(annotation, (1.5, 0.22), (2.25, 0.3), arrowprops=props)
    
    # 标记 F 分布曲线上超过统计量的区域
    i = val >= res.statistic
    ax.fill_between(val[i], y1=0, y2=pdf[i], color='C0')
    
    # 显示绘制的图表
    plt.show()
    
    # 输出 Levene 检验的 p 值
    res.pvalue
    If the p-value is "small" - that is, if there is a low probability of
    sampling data from distributions with identical variances that produces
    such an extreme value of the statistic - this may be taken as evidence
    against the null hypothesis in favor of the alternative: the variances of
    the groups are not equal. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [5]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).
    - Small p-values are not evidence for a *large* effect; rather, they can
      only provide evidence for a "significant" effect, meaning that they are
      unlikely to have occurred under the null hypothesis.

    Note that the F distribution provides an asymptotic approximation of the
    null distribution.
    For small samples, it may be more appropriate to perform a permutation
    test: Under the null hypothesis that all three samples were drawn from
    the same population, each of the measurements is equally likely to have
    been observed in any of the three samples. Therefore, we can form a
    randomized null distribution by calculating the statistic under many
    randomly-generated partitionings of the observations into the three
    samples.

Explanation:


    >>> def statistic(*samples):
    ...     return stats.levene(*samples).statistic

Explanation:


    >>> ref = stats.permutation_test(
    ...     (small_dose, medium_dose, large_dose), statistic,
    ...     permutation_type='independent', alternative='greater'
    ... )

Explanation:


    >>> fig, ax = plt.subplots(figsize=(8, 5))

Explanation:


    >>> plot(ax)

Explanation:


    >>> bins = np.linspace(0, 5, 25)

Explanation:


    >>> ax.hist(
    ...     ref.null_distribution, bins=bins, density=True, facecolor="C1"
    ... )

Explanation:


    >>> ax.legend(['aymptotic approximation\n(many observations)',
    ...            'randomized null distribution'])

Explanation:


    >>> plot(ax)

Explanation:


    >>> plt.show()

Explanation:


    >>> ref.pvalue  # randomized test p-value
    0.4559  # may vary

Explanation:


    Note that there is significant disagreement between the p-value calculated
    here and the asymptotic approximation returned by `levene` above.
    The statistical inferences that can be drawn rigorously from a permutation
    test are limited; nonetheless, they may be the preferred approach in many
    circumstances [6]_.

    Following is another generic example where the null hypothesis would be
    rejected.

Explanation:


    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.

Explanation:


    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]

Explanation:


    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]

Explanation:


    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]

Explanation:
    >>> stat, p = stats.levene(a, b, c)
    >>> p
    0.002431505967249681
    
    The small p-value suggests that the populations do not have equal variances.
    
    This is not surprising, given that the sample variance of `b` is much larger than that of `a` and `c`:
    
    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]
    
    """
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")
    
    k = len(samples)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")
    
    Ni = np.empty(k)
    Yci = np.empty(k, 'd')
    
    if center == 'median':
    
        def func(x):
            return np.median(x, axis=0)
    
    elif center == 'mean':
    
        def func(x):
            return np.mean(x, axis=0)
    
    else:  # center == 'trimmed'
        samples = tuple(_stats_py.trimboth(np.sort(sample), proportiontocut)
                        for sample in samples)
    
        def func(x):
            return np.mean(x, axis=0)
    
    for j in range(k):
        Ni[j] = len(samples[j])
        Yci[j] = func(samples[j])
    Ntot = np.sum(Ni, axis=0)
    
    # compute Zij's
    Zij = [None] * k
    for i in range(k):
        Zij[i] = abs(asarray(samples[i]) - Yci[i])
    
    # compute Zbari
    Zbari = np.empty(k, 'd')
    Zbar = 0.0
    for i in range(k):
        Zbari[i] = np.mean(Zij[i], axis=0)
        Zbar += Zbari[i] * Ni[i]
    
    Zbar /= Ntot
    numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)
    
    # compute denom_variance
    dvar = 0.0
    for i in range(k):
        dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)
    
    denom = (k - 1.0) * dvar
    
    W = numer / denom
    pval = distributions.f.sf(W, k-1, Ntot-k)  # 1 - cdf
    return LeveneResult(W, pval)
# 定义一个函数 `_apply_func`，用于对输入的列表 `x` 根据列表 `g` 中的索引将其分成不同的组，并对每组应用给定的函数 `func`
def _apply_func(x, g, func):
    # g 是索引列表，用于将 x 分成不同的组
    g = unique(r_[0, g, len(x)])  # 将 g 转换为唯一值列表，并在开头和结尾添加索引 0 和 len(x)
    # 对每个组应用函数 func，并将结果存储在列表 output 中
    output = [func(x[g[k]:g[k+1]]) for k in range(len(g) - 1)]

    return asarray(output)


# 定义一个命名元组 FlignerResult，用于存储 Fligner 检验的结果，包括统计量和 p 值
FlignerResult = namedtuple('FlignerResult', ('statistic', 'pvalue'))


# 应用装饰器，为 fligner 函数添加特定的轴的 NaN 策略，返回的结果类型为 FlignerResult，样本数量未指定
@_axis_nan_policy_factory(FlignerResult, n_samples=None)
def fligner(*samples, center='median', proportiontocut=0.05):
    r"""Perform Fligner-Killeen test for equality of variance.

    Fligner's test tests the null hypothesis that all input samples
    are from populations with equal variances.  Fligner-Killeen's test is
    distribution free when populations are identical [2]_.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        Arrays of sample data.  Need not be the same length.
    center : {'mean', 'median', 'trimmed'}, optional
        Keyword argument controlling which function of the data is used in
        computing the test statistic.  The default is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the hypothesis test.

    See Also
    --------
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A robust parametric test for equality of k variances

    Notes
    -----
    As with Levene's test there are three variants of Fligner's test that
    differ by the measure of central tendency used in the test.  See `levene`
    for more information.

    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power
    [3]_.

    References
    ----------
    .. [1] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
           https://cecas.clemson.edu/~cspark/cv/paper/qif/draftqif2.pdf
    .. [2] Fligner, M.A. and Killeen, T.J. (1976). Distribution-free two-sample
           tests for scale. 'Journal of the American Statistical Association.'
           71(353), 210-213.
    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
    """
    # 函数实现 Fligner-Killeen 方差相等性检验
    pass
    # 导入需要的库：numpy
    import numpy as np
    
    # 创建包含小剂量维生素C组的牙齿生长数据的数组
    small_dose = np.array([
        4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7,
        15.2, 21.5, 17.6, 9.7, 14.5, 10, 8.2, 9.4, 16.5, 9.7
    ])
    
    # 创建包含中等剂量维生素C组的牙齿生长数据的数组
    medium_dose = np.array([
        16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5,
        19.7, 23.3, 23.6, 26.4, 20, 25.2, 25.8, 21.2, 14.5, 27.3
    ])
    
    # 创建包含大剂量维生素C组的牙齿生长数据的数组
    large_dose = np.array([
        23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5,
        25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23
    ])
    
    # 导入需要的库：scipy.stats
    from scipy import stats
    
    # 使用Fligner-Killeen检验比较三组数据的方差是否相等
    res = stats.fligner(small_dose, medium_dose, large_dose)
    
    # 输出Fligner-Killeen检验的统计量（检验量）
    res.statistic
    
    # 对于Fligner-Killeen检验，当组间方差差异较大时，统计量通常较高
    
    # 导入需要的库：matplotlib.pyplot
    import matplotlib.pyplot as plt
    
    # 定义自定义函数plot用于绘制图形，这里重复使用
    fig, ax = plt.subplots(figsize=(8, 5))
    def plot(ax):
        # 绘制卡方分布的概率密度函数图像
        ax.plot(val, pdf, color='C0')
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> pvalue = dist.sf(res.statistic)
    >>> annotation = (f'p-value={pvalue:.4f}\n(shaded area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (1.5, 0.22), (2.25, 0.3), arrowprops=props)


    # 创建一个大小为 8x5 的图形窗口，并在其中绘制图表
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    # 调用 plot 函数，将图表绘制在 ax（Axes 对象）上
    >>> plot(ax)
    # 计算并获取分布函数 dist 的 p-value（小于或等于统计量的空分布中的值的比例）
    >>> pvalue = dist.sf(res.statistic)
    # 创建注释文本，包括 p-value 值和附加信息，准备在图中特定位置添加注释
    >>> annotation = (f'p-value={pvalue:.4f}\n(shaded area)')
    # 设置注释箭头的属性，如颜色、宽度和箭头头部尺寸
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    # 在图中的指定位置添加注释，并使用指定的箭头属性
    >>> _ = ax.annotate(annotation, (1.5, 0.22), (2.25, 0.3), arrowprops=props)

    >>> i = val >= res.statistic
    >>> ax.fill_between(val[i], y1=0, y2=pdf[i], color='C0')
    >>> plt.show()


    # 根据比较结果创建布尔数组 i，用于标记 val 中大于等于统计量 res.statistic 的部分
    >>> i = val >= res.statistic
    # 在图中填充 val 和 pdf 中 i 为 True 的区域，使用蓝色填充
    >>> ax.fill_between(val[i], y1=0, y2=pdf[i], color='C0')
    # 显示图表
    >>> plt.show()

    >>> res.pvalue
    0.49960016501182125


    # 打印并获取统计结果对象 res 的 p-value 值
    >>> res.pvalue
    0.49960016501182125

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from distributions with identical variances that produces
    such an extreme value of the statistic, this may be taken as evidence
    against the null hypothesis in favor of the alternative: the variances of
    the groups are not equal. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence for the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [6]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).
    - Small p-values are not evidence for a *large* effect; rather, they can
      only provide evidence for a "significant" effect, meaning that they are
      unlikely to have occurred under the null hypothesis.

    Note that the chi-square distribution provides an asymptotic approximation
    of the null distribution.
    For small samples, it may be more appropriate to perform a
    permutation test: Under the null hypothesis that all three samples were
    drawn from the same population, each of the measurements is equally likely
    to have been observed in any of the three samples. Therefore, we can form
    a randomized null distribution by calculating the statistic under many
    randomly-generated partitionings of the observations into the three
    samples.


    # 如果 p-value 较小，即从具有相同方差的分布中随机抽样产生出现如此极端统计量值的概率很低，
    # 这可以作为反对零假设的证据，支持备择假设：各组的方差不相等。注意：

    - 相反的情况并不成立，即该检验不能用来支持零假设的证据。
    - 将被视为“小”的值的阈值应在数据分析之前做出选择 [6]_，考虑到误差率（错误地拒绝零假设）
      和误差否定（未能拒绝错误的零假设）的风险。
    - 小的 p-value 不是大效应的证据；而是只能提供“显著”效应的证据，即它们不太可能发生在零假设下。

    注意，卡方分布提供了零分布的渐近近似。
    对于小样本，执行置换检验可能更合适：在零假设下，所有三个样本都是从同一总体抽取的，
    每个测量在任何三个样本中观察到的可能性是相等的。因此，我们可以通过计算在观测值随机分
    配到这三个样本中的多个分区中的统计量来形成随机化的零分布。


    >>> def statistic(*samples):
    ...     return stats.fligner(*samples).statistic
    >>> ref = stats.permutation_test(
    ...     (small_dose, medium_dose, large_dose), statistic,
    ...     permutation_type='independent', alternative='greater'
    ... )
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> bins = np.linspace(0, 8, 25)
    >>> ax.hist(
    ...     ref.null_distribution, bins=bins, density=True, facecolor="C1"
    ... )
    >>> ax.legend(['aymptotic approximation\n(many observations)',


    # 定义一个统计函数 statistic，用于计算多个样本的 Fligner-Killeen检验统计量
    >>> def statistic(*samples):
    ...     return stats.fligner(*samples).statistic
    # 执行置换检验，使用 statistic 函数对 (small_dose, medium_dose, large_dose) 进行检验，
    # permutation_type 设为 'independent'，alternative 设为 'greater'
    >>> ref = stats.permutation_test(
    ...     (small_dose, medium_dose, large_dose), statistic,
    ...     permutation_type='independent', alternative='greater'
    ... )
    # 创建一个大小为 8x5 的图形窗口，并在其中绘制图表
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    # 调用 plot 函数，将图表绘制在 ax（Axes 对象）上
    >>> plot(ax)
    # 创建 bins 数组，均匀分布从 0 到 8，共 25 个点
    >>> bins = np.linspace(0, 8, 25)
    # 绘制 ref.null_distribution 的直方图，设置颜色为 "C1"，密度为 True
    >>> ax.hist(
    ...     ref.null_distribution, bins=bins, density=True, facecolor="C1"
    ... )
    # 添加图例，说明含义为“渐近近似\n（多个观察）”
    >>> ax.legend(['aymptotic approximation\n(many observations)',
    """
    以下是一个例子，展示如何使用 permutation test 来检验方差相等的假设。

    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.

    检验列表 `a`, `b` 和 `c` 是否来自方差相等的总体。

    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    定义统计量 `stat` 和 p 值 `p`，使用 `fligner` 函数进行检验
    >>> stat, p = stats.fligner(a, b, c)
    获取计算得到的 p 值
    >>> p
    0.00450826080004775

    The small p-value suggests that the populations do not have equal
    variances.
    
    较小的 p 值表明这些总体的方差不相等。

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    这并不奇怪，因为 `b` 的样本方差远大于 `a` 和 `c`：

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    """

    # 检查 center 参数是否有效，必须是 'mean', 'median' 或 'trimmed'
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

    # 获取样本的数量 k
    k = len(samples)
    # 至少需要两个输入样本向量，否则抛出 ValueError
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    # 处理空输入的情况，如果有样本的大小为 0，则返回 NaN
    for sample in samples:
        if sample.size == 0:
            NaN = _get_nan(*samples)
            return FlignerResult(NaN, NaN)

    # 根据 center 参数选择不同的统计函数
    if center == 'median':

        # 定义一个函数，计算样本向量的中位数
        def func(x):
            return np.median(x, axis=0)

    elif center == 'mean':

        # 定义一个函数，计算样本向量的均值
        def func(x):
            return np.mean(x, axis=0)

    else:  # center == 'trimmed'

        # 对每个样本向量应用 trimboth 函数，计算均值
        samples = tuple(_stats_py.trimboth(sample, proportiontocut)
                        for sample in samples)

        def func(x):
            return np.mean(x, axis=0)

    # 计算每个样本向量的长度 Ni 和对应的中心化后的样本 Yci
    Ni = np.asarray([len(samples[j]) for j in range(k)])
    Yci = np.asarray([func(samples[j]) for j in range(k)])
    Ntot = np.sum(Ni, axis=0)

    # 计算 Zij
    Zij = [np.abs(np.asarray(samples[i]) - Yci[i]) for i in range(k)]
    allZij = []
    g = [0]

    # 将 Zij 扁平化为一个列表，并记录每个样本的索引位置
    for i in range(k):
        allZij.extend(list(Zij[i]))
        g.append(len(allZij))

    # 对 Zij 排序并计算秩次
    ranks = _stats_py.rankdata(allZij)
    # 根据秩次计算样本的标准正态分布值
    sample = distributions.norm.ppf(ranks / (2*(Ntot + 1.0)) + 0.5)

    # 计算 Aibar
    Aibar = _apply_func(sample, g, np.sum) / Ni
    # 计算样本的均值 anbar
    anbar = np.mean(sample, axis=0)
    # 计算样本的方差 varsq
    varsq = np.var(sample, axis=0, ddof=1)
    # 计算统计量
    statistic = np.sum(Ni * (np.asarray(Aibar) - anbar)**2.0, axis=0) / varsq
    # 使用 Fligner 法计算 p 值
    chi2 = _SimpleChi2(k-1)
    pval = _get_pvalue(statistic, chi2, alternative='greater', symmetric=False, xp=np)
    # 返回 FlignerResult 对象，包含计算得到的统计量和 p 值
    return FlignerResult(statistic, pval)
# 使用装饰器 `_axis_nan_policy_factory` 包装函数 `_mood_inner_lc`，该装饰器提供了处理 NaN（Not a Number）的策略
@_axis_nan_policy_factory(lambda x1: (x1,), n_samples=4, n_outputs=1)
# 函数 `_mood_inner_lc` 接受参数 `xy`, `x`, `diffs`, `sorted_xy`, `n`, `m`, `N` 并返回一个浮点数
def _mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N) -> float:
    # 从汇总样本中获取唯一值及其频率。
    # "a_j, + b_j, = t_j, for j = 1, ... k" 其中 `k` 是唯一类别的数量，
    # "[t]he number of values associated with the x's and y's in
    # the jth class will be denoted by a_j, and b_j respectively."
    # (Mielke, 312)
    # 重用先前计算的排序数组和 `diff` 数组以获取唯一值和计数。在 `diffs` 前面添加一个非零值，表示第一个元素不匹配之前的元素。
    diffs_prep = np.concatenate(([1], diffs))
    # 唯一元素是在排序数组中元素之间有差异的地方
    uniques = sorted_xy[diffs_prep != 0]
    # 每个元素的计数是每组连续差异的大小，其中差异为零。用 1 替换非零差异，然后使用累积和来计算索引。
    t = np.bincount(np.cumsum(np.asarray(diffs_prep != 0, dtype=int)))[1:]
    k = len(uniques)
    js = np.arange(1, k + 1, dtype=int)
    # 论文中提到的 `b` 数组在计算 `t` 之外没有使用，因此我们不需要单独计算它。在这里我们计算 `a`。
    # 简单地说，`a[j]` 是在 `x` 中等于 `uniques[j]` 的值的数量。
    sorted_xyx = np.sort(np.concatenate((xy, x)))
    diffs = np.diff(sorted_xyx)
    diffs_prep = np.concatenate(([1], diffs))
    diff_is_zero = np.asarray(diffs_prep != 0, dtype=int)
    xyx_counts = np.bincount(np.cumsum(diff_is_zero))[1:]
    a = xyx_counts - t
    # "Define .. a_0 = b_0 = t_0 = S_0 = 0" (Mielke 312) 所以我们将 `a` 和 `t` 数组向右移动 1 位，允许第一个元素为 0 以适应此索引。
    t = np.concatenate(([0], t))
    a = np.concatenate(([0], a))
    # `S` 是从 `t` 构建的，因此不需要添加前导零。
    S = np.cumsum(t)
    # 定义 `S` 的副本，前面加一个零以供以后使用，以避免需要索引。
    S_i_m1 = np.concatenate(([0], S[:-1]))

    # Psi，由第 313 页第 6 个未编号的方程定义（Mielke）。
    # 注意，在论文中存在一个错误，其中分母 `2` 被平方，而应该是整个方程。
    def psi(indicator):
        return (indicator - (N + 1)/2)**2

    # 定义用于计算 phi 的求和范围，如第 312 页底部未编号方程中的求和。
    s_lower = S[js - 1] + 1
    s_upper = S[js] + 1
    phi_J = [np.arange(s_lower[idx], s_upper[idx]) for idx in range(k)]

    # 对上述数组中的每个范围，确定每个元素的 psi(I) 的总和。将所有总和除以 `t`。遵循第 312 页最后一个未编号的方程。
    # 计算每个 phi_J 中每个 I_j 的求和，然后除以 t[js]。结果存储在 phis 中。
    phis = [np.sum(psi(I_j)) for I_j in phi_J] / t[js]

    # 计算统计量 T，根据第一页上的第一个未编号方程式，T 等于 phis 乘以 a[js]。
    # phis 已经按照 js 的顺序排列，因此我们也用 js 索引到 a。
    T = sum(phis * a[js])

    # 计算近似统计量 E_0_T。
    E_0_T = n * (N * N - 1) / 12

    # 计算方差 varM。
    # varM 是一个复杂的表达式，包括 m、n、N、S 和 S_i_m1 的函数。
    varM = (m * n * (N + 1.0) * (N ** 2 - 4) / 180 -
            m * n / (180 * N * (N - 1)) * np.sum(
                t * (t**2 - 1) * (t**2 - 4 + (15 * (N - S - S_i_m1) ** 2))
            ))

    # 返回一个包含统计量 T 标准化值的元组。
    return ((T - E_0_T) / np.sqrt(varM),)
# 定义一个函数 `_mood_too_small`，用于检查样本数是否过小，判断条件为总样本数小于3
def _mood_too_small(samples, kwargs, axis=-1):
    # 将输入的样本数据分别赋值给变量 x 和 y
    x, y = samples
    # 计算 x 在指定轴上的长度
    n = x.shape[axis]
    # 计算 y 在指定轴上的长度
    m = y.shape[axis]
    # 计算总样本数 N，为 x 和 y 在指定轴上的长度之和
    N = m + n
    # 返回判断结果，即总样本数是否小于3
    return N < 3


# 使用装饰器 `_axis_nan_policy_factory`，对函数 `mood` 进行装饰，设定一些参数并定义 `too_small` 条件为 `_mood_too_small`
@_axis_nan_policy_factory(SignificanceResult, n_samples=2, too_small=_mood_too_small)
# 定义函数 `mood`，执行 Mood's 测试以检验两个样本数据的等比例参数
def mood(x, y, axis=0, alternative="two-sided"):
    """Perform Mood's test for equal scale parameters.

    Mood's two-sample test for scale parameters is a non-parametric
    test for the null hypothesis that two samples are drawn from the
    same distribution with the same scale parameter.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data. There must be at least three observations
        total.
    axis : int, optional
        The axis along which the samples are tested.  `x` and `y` can be of
        different length along `axis`.
        If `axis` is None, `x` and `y` are flattened and the test is done on
        all values in the flattened arrays.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the scales of the distributions underlying `x` and `y`
          are different.
        * 'less': the scale of the distribution underlying `x` is less than
          the scale of the distribution underlying `y`.
        * 'greater': the scale of the distribution underlying `x` is greater
          than the scale of the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : scalar or ndarray
            The z-score for the hypothesis test.  For 1-D inputs a scalar is
            returned.
        pvalue : scalar ndarray
            The p-value for the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    ansari : A non-parametric test for the equality of 2 variances
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A parametric test for equality of k variances

    Notes
    -----
    The data are assumed to be drawn from probability distributions ``f(x)``
    and ``f(x/s) / s`` respectively, for some probability density function f.
    The null hypothesis is that ``s == 1``.

    For multi-dimensional arrays, if the inputs are of shapes
    ``(n0, n1, n2, n3)``  and ``(n0, m1, n2, n3)``, then if ``axis=1``, the
    resulting z and p values will have shape ``(n0, n2, n3)``.  Note that
    ``n1`` and ``m1`` don't have to be equal, but the other dimensions do.

    References
    ----------
    [1] Mielke, Paul W. "Note on Some Squared Rank Tests with Existing Ties."
        Technometrics, vol. 9, no. 2, 1967, pp. 312-14. JSTOR,
        https://doi.org/10.2307/1266427. Accessed 18 May 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    # 将输入的 x 和 y 转换为浮点数的 NumPy 数组
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # 如果 axis 小于 0，则将其转换为 x 的维度加上 axis 的值
    if axis < 0:
        axis = x.ndim + axis
    
    # 确定结果数组的形状
    res_shape = tuple([x.shape[ax] for ax in range(len(x.shape)) if ax != axis])
    
    # 检查 y 的形状在除了 axis 外的所有轴上与 x 的形状是否相匹配
    if not (res_shape == tuple([y.shape[ax] for ax in range(len(y.shape)) if ax != axis])):
        raise ValueError("Dimensions of x and y on all axes except `axis` should match")
    
    # 计算 x 和 y 在给定轴上的样本数
    n = x.shape[axis]
    m = y.shape[axis]
    N = m + n
    
    # 如果样本数量不足 3，则引发 ValueError
    if N < 3:
        raise ValueError("Not enough observations.")
    
    # 将 x 和 y 沿指定轴连接成一个数组 xy
    xy = np.concatenate((x, y), axis=axis)
    
    # 检查数组中是否存在重复值
    sorted_xy = np.sort(xy, axis=axis)
    diffs = np.diff(sorted_xy, axis=axis)
    if 0 in diffs:
        # 如果存在重复值，则调用 _mood_inner_lc 函数处理
        z = np.asarray(_mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N,
                                      axis=axis))
    else:
        # 如果不存在重复值，则执行以下操作
        if axis != 0:
            xy = np.moveaxis(xy, axis, 0)
    
        # 将 xy 展平为二维数组，并计算秩
        xy = xy.reshape(xy.shape[0], -1)
        all_ranks = np.empty_like(xy)
        for j in range(xy.shape[1]):
            all_ranks[:, j] = _stats_py.rankdata(xy[:, j])
    
        Ri = all_ranks[:n]
        M = np.sum((Ri - (N + 1.0) / 2) ** 2, axis=0)
    
        # 计算近似统计量
        mnM = n * (N * N - 1.0) / 12
        varM = m * n * (N + 1.0) * (N + 2) * (N - 2) / 180
        z = (M - mnM) / sqrt(varM)
    
    # 根据 z 值和给定的备择假设 alternative 计算 p 值
    pval = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    
    # 如果结果形状为 ()，将 z 和 pval 转换为标量
    if res_shape == ():
        z = z[0]
        pval = pval[0]
    else:
        # 否则将 z 和 pval 的形状调整为 res_shape
        z.shape = res_shape
        pval.shape = res_shape
    
    # 返回一个 SignificanceResult 对象，包含 z 值和 p 值
    return SignificanceResult(z[()], pval[()])
# 创建一个命名为 WilcoxonResult 的命名元组，包含 statistic 和 pvalue 两个字段
WilcoxonResult = _make_tuple_bunch('WilcoxonResult', ['statistic', 'pvalue'])

# 解包 WilcoxonResult 对象的结果，如果有 zstatistic 属性则返回 statistic、pvalue 和 zstatistic，否则返回 statistic 和 pvalue
def wilcoxon_result_unpacker(res):
    if hasattr(res, 'zstatistic'):
        return res.statistic, res.pvalue, res.zstatistic
    else:
        return res.statistic, res.pvalue

# 创建一个 WilcoxonResult 对象，包含 statistic 和 pvalue 字段，如果提供了 zstatistic 则也包含 zstatistic 字段
def wilcoxon_result_object(statistic, pvalue, zstatistic=None):
    res = WilcoxonResult(statistic, pvalue)
    if zstatistic is not None:
        res.zstatistic = zstatistic
    return res

# 根据参数 kwds 返回 Wilcoxon 检验的输出数量，method 默认为 'auto'，如果为 'approx' 则返回 3，否则返回 2
def wilcoxon_outputs(kwds):
    method = kwds.get('method', 'auto')
    if method == 'approx':
        return 3
    return 2

# 修饰器函数，重命名参数 "mode" 为 "method"，同时配置参数校准和结果转换函数等
@_rename_parameter("mode", "method")
@_axis_nan_policy_factory(
    wilcoxon_result_object, paired=True,
    n_samples=lambda kwds: 2 if kwds.get('y', None) is not None else 1,
    result_to_tuple=wilcoxon_result_unpacker, n_outputs=wilcoxon_outputs,
)
# 计算 Wilcoxon 符号秩检验的函数，根据参数 x 和 y 计算检验结果
def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", method='auto', *, axis=0):
    """Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences ``x - y`` is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    y : array_like, optional
        Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.

        .. warning::
            When `y` is provided, `wilcoxon` calculates the test statistic
            based on the ranks of the absolute values of ``d = x - y``.
            Roundoff error in the subtraction can result in elements of ``d``
            being assigned different ranks even when they would be tied with
            exact arithmetic. Rather than passing `x` and `y` separately,
            consider computing the difference ``x - y``, rounding as needed to
            ensure that only truly unique elements are numerically distinct,
            and passing the result as `x`, leaving `y` at the default (None).
    zero_method : {"wilcox", "pratt", "zsplit"}, optional
        处理相等数值对观测值的不同惯例（"zero-differences"或"zeros"）。

        * "wilcox": 忽略所有零差异（默认）；参见 [4]_。
        * "pratt": 在排名过程中包括零差异，但是删除零的排名（更保守）；参见 [3]_。
          在这种情况下，正态近似会像 [5]_ 中调整。
        * "zsplit": 在排名过程中包括零差异，并将零的排名分为正数和负数部分。

    correction : bool, optional
        如果为True，在使用正态近似计算z统计量时，通过将Wilcoxon排名统计量向均值调整0.5来应用连续性校正。默认为False。

    alternative : {"two-sided", "greater", "less"}, optional
        定义备择假设。默认为'two-sided'。
        在以下情况下，让``d``表示成对样本之间的差异：如果提供了``x``和``y``，则``d = x - y``，否则``d = x``。

        * 'two-sided': ``d``的分布不对称于零。
        * 'less': ``d``的分布随机小于对称于零的分布。
        * 'greater': ``d``的分布随机大于对称于零的分布。

    method : {"auto", "exact", "approx"} or `PermutationMethod` instance, optional
        计算p值的方法，请参阅备注。默认为"auto"。

    axis : int or None, default: 0
        如果是int，沿着计算统计量的输入轴（例如行）。
        输入的每个轴切片的统计量将显示在输出的相应元素中。如果为``None``，则在计算统计量之前将展平输入。

    Returns
    -------
    具有以下属性的对象。

    statistic : array_like
        如果`alternative`是"two-sided"，则差异大于或小于零的排名和，以较小者为准。
        否则，差异大于零的排名和。

    pvalue : array_like
        依赖于`alternative`和`method`的检验的p值。

    zstatistic : array_like
        当``method = 'approx'``时，这是标准化的z统计量::

            z = (T - mn - d) / se

        其中``T``如上所定义为`statistic`，``mn``是零假设下分布的均值，``d``是连续性校正，``se``是标准误差。
        当``method != 'approx'``时，此属性不可用。

    See Also
    --------
    kruskal, mannwhitneyu

    Notes
    -----
    在以下情况下，让``d``表示成对样本之间的差异：
    samples: ``d = x - y`` if both ``x`` and ``y`` are provided, or ``d = x``
    otherwise. Assume that all elements of ``d`` are independent and
    identically distributed observations, and all are distinct and nonzero.
    # 如果提供了``x``和``y``，则计算``d = x - y``；否则，``d = x``。假设``d``的所有元素是独立同分布的观测值，且全部不同且非零。

    - When ``len(d)`` is sufficiently large, the null distribution of the
      normalized test statistic (`zstatistic` above) is approximately normal,
      and ``method = 'approx'`` can be used to compute the p-value.
    # 当``d``的长度足够大时，归一化检验统计量（上述的`zstatistic`）的零分布近似服从正态分布，此时可以使用``method = 'approx'``计算 p 值。

    - When ``len(d)`` is small, the normal approximation may not be accurate,
      and ``method='exact'`` is preferred (at the cost of additional
      execution time).
    # 当``d``的长度较小时，正态近似可能不准确，推荐使用``method='exact'``（虽然会增加计算时间）。

    - The default, ``method='auto'``, selects between the two: when
      ``len(d) <= 50`` and there are no zeros, the exact method is used;
      otherwise, the approximate method is used.
    # 默认情况下，``method='auto'``会自动选择方法：当``len(d) <= 50``且没有零元素时，使用精确方法；否则使用近似方法。

    The presence of "ties" (i.e. not all elements of ``d`` are unique) or
    "zeros" (i.e. elements of ``d`` are zero) changes the null distribution
    of the test statistic, and ``method='exact'`` no longer calculates
    the exact p-value. If ``method='approx'``, the z-statistic is adjusted
    for more accurate comparison against the standard normal, but still,
    for finite sample sizes, the standard normal is only an approximation of
    the true null distribution of the z-statistic. For such situations, the
    `method` parameter also accepts instances `PermutationMethod`. In this
    case, the p-value is computed using `permutation_test` with the provided
    configuration options and other appropriate settings.
    # 如果存在“ties”（即``d``中不是所有元素都是唯一的）或“zeros”（即``d``中存在零元素），将改变检验统计量的零分布，此时``method='exact'``不再计算精确的 p 值。如果使用``method='approx'``，则会调整 z 统计量以更准确地与标准正态分布比较，但对于有限的样本大小，标准正态分布仍然只是 z 统计量真实零分布的近似。对于这种情况，`method`参数还接受`PermutationMethod`的实例。在这种情况下，将使用提供的配置选项和其他适当设置使用`permutation_test`计算 p 值。

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`
    # 引用和参考文献列表，提供了关于此检验的背景信息和相关文献。

    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:

    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

    Cross-fertilized plants appear to be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:

    >>> from scipy.stats import wilcoxon
    >>> res = wilcoxon(d)
    >>> res.statistic, res.pvalue
    (24.0, 0.041259765625)

    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    # 给出了一个示例，说明如何使用 `wilcoxon` 函数进行检验，并解释了如何根据得到的结果拒绝或接受原假设。
    # 返回 Wilcoxon 检验的结果，比较两个样本或一组差异值是否存在显著差异
    return _wilcoxon._wilcoxon_nd(x, y, zero_method, correction, alternative,
                                  method, axis)
# 创建一个命名元组 MedianTestResult，包含 statistic、pvalue、median 和 table 这些字段，初始值为空列表
MedianTestResult = _make_tuple_bunch(
    'MedianTestResult',
    ['statistic', 'pvalue', 'median', 'table'], []
)

# 定义 median_test 函数，执行 Mood's 中位数检验
def median_test(*samples, ties='below', correction=True, lambda_=1,
                nan_policy='propagate'):
    """Perform a Mood's median test.

    Test that two or more samples come from populations with the same median.

    Let ``n = len(samples)`` be the number of samples.  The "grand median" of
    all the data is computed, and a contingency table is formed by
    classifying the values in each sample as being above or below the grand
    median.  The contingency table, along with `correction` and `lambda_`,
    are passed to `scipy.stats.chi2_contingency` to compute the test statistic
    and p-value.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The set of samples.  There must be at least two samples.
        Each sample must be a one-dimensional sequence containing at least
        one value.  The samples are not required to have the same length.
    ties : str, optional
        Determines how values equal to the grand median are classified in
        the contingency table.  The string must be one of::

            "below":
                Values equal to the grand median are counted as "below".
            "above":
                Values equal to the grand median are counted as "above".
            "ignore":
                Values equal to the grand median are not counted.

        The default is "below".
    correction : bool, optional
        If True, *and* there are just two samples, apply Yates' correction
        for continuity when computing the test statistic associated with
        the contingency table.  Default is True.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family to be used instead.  See
        `power_divergence` for details.
        Default is 1 (Pearson's chi-squared statistic).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    res : MedianTestResult
        An object containing attributes:

        statistic : float
            The test statistic.  The statistic that is returned is determined
            by `lambda_`.  The default is Pearson's chi-squared statistic.
        pvalue : float
            The p-value of the test.
        median : float
            The grand median.
        table : ndarray
            The contingency table.  The shape of the table is (2, n), where
            n is the number of samples.  The first row holds the counts of the
            values above the grand median, and the second row holds the counts
            of the values below the grand median.  The table allows further
            analysis with, for example, `scipy.stats.chi2_contingency`, or with
            `scipy.stats.fisher_exact` if there are two samples, without having
            to recompute the table.  If ``nan_policy`` is "propagate" and there
            are nans in the input, the return value for ``table`` is ``None``.

    See Also
    --------
    kruskal : Compute the Kruskal-Wallis H-test for independent samples.
    mannwhitneyu : Computes the Mann-Whitney rank test on samples x and y.

    Notes
    -----
    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Mood, A. M., Introduction to the Theory of Statistics. McGraw-Hill
        (1950), pp. 394-399.
    .. [2] Zar, J. H., Biostatistical Analysis, 5th ed. Prentice Hall (2010).
        See Sections 8.12 and 10.15.

    Examples
    --------
    A biologist runs an experiment in which there are three groups of plants.
    Group 1 has 16 plants, group 2 has 15 plants, and group 3 has 17 plants.
    Each plant produces a number of seeds.  The seed counts for each group
    are::

        Group 1: 10 14 14 18 20 22 24 25 31 31 32 39 43 43 48 49
        Group 2: 28 30 31 33 34 35 36 40 44 55 57 61 91 92 99
        Group 3:  0  3  9 22 23 25 25 33 34 34 40 45 46 48 62 67 84

    The following code applies Mood's median test to these samples.

    >>> g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    >>> g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    >>> g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    >>> from scipy.stats import median_test
    >>> res = median_test(g1, g2, g3)

    The median is

    >>> res.median
    34.0

    and the contingency table is

    >>> res.table
    array([[ 5, 10,  7],
           [11,  5, 10]])

    `p` is too large to conclude that the medians are not the same:

    >>> res.pvalue
    0.12609082774093244

    The "G-test" can be performed by passing ``lambda_="log-likelihood"`` to
    `median_test`.

    >>> res = median_test(g1, g2, g3, lambda_="log-likelihood")
    >>> res.pvalue
    0.12224779737117837

    The median occurs several times in the data, so we'll get a different
    result if, for example, ``ties="above"`` is used:

    >>> res = median_test(g1, g2, g3, ties="above")
    # 输出结果中的 p-value
    >>> res.pvalue
    0.063873276069553273

    # 输出结果中的表格数据，展示了两个样本的数据分布情况
    >>> res.table
    array([[ 5, 11,  9],
           [11,  4,  8]])

    # 这个示例演示了如果数据集不大，并且存在与中位数相等的值，那么 p-value 可能对 'ties' 的选择敏感。

    """
    # 检查样本数量是否小于2，如果是则抛出错误
    if len(samples) < 2:
        raise ValueError('median_test requires two or more samples.')

    # 允许的 'ties' 选项
    ties_options = ['below', 'above', 'ignore']
    # 如果提供的 'ties' 不在允许的选项中，则抛出错误
    if ties not in ties_options:
        raise ValueError(f"invalid 'ties' option '{ties}'; 'ties' must be one "
                         f"of: {str(ties_options)[1:-1]}")

    # 将样本转换为 numpy 数组
    data = [np.asarray(sample) for sample in samples]

    # 验证参数的大小和形状
    for k, d in enumerate(data):
        # 如果样本为空，则抛出错误
        if d.size == 0:
            raise ValueError("Sample %d is empty. All samples must "
                             "contain at least one value." % (k + 1))
        # 如果样本维度不为1，则抛出错误
        if d.ndim != 1:
            raise ValueError("Sample %d has %d dimensions.  All "
                             "samples must be one-dimensional sequences." %
                             (k + 1, d.ndim))

    # 合并所有样本数据
    cdata = np.concatenate(data)
    # 检查数据中是否包含 NaN，并根据 nan_policy 处理
    contains_nan, nan_policy = _contains_nan(cdata, nan_policy)
    if contains_nan and nan_policy == 'propagate':
        # 如果包含 NaN 并且 nan_policy 是 'propagate'，则返回 NaN 的结果
        return MedianTestResult(np.nan, np.nan, np.nan, None)

    # 计算总体的中位数
    if contains_nan:
        grand_median = np.median(cdata[~np.isnan(cdata)])
    else:
        grand_median = np.median(cdata)
    # 当 numpy 的最小版本为 1.9.0 时，上述 if/else 语句可以用下面一行替代：
    #     grand_median = np.nanmedian(cdata)

    # 创建列联表（contingency table）
    table = np.zeros((2, len(data)), dtype=np.int64)
    for k, sample in enumerate(data):
        # 去除样本中的 NaN 值
        sample = sample[~np.isnan(sample)]

        # 计算大于和小于总体中位数的样本数量
        nabove = np.count_nonzero(sample > grand_median)
        nbelow = np.count_nonzero(sample < grand_median)
        nequal = sample.size - (nabove + nbelow)
        table[0, k] += nabove
        table[1, k] += nbelow
        # 根据 'ties' 的选项调整表格数据
        if ties == "below":
            table[1, k] += nequal
        elif ties == "above":
            table[0, k] += nequal

    # 检查表格的每行或每列是否全为零
    rowsums = table.sum(axis=1)
    if rowsums[0] == 0:
        # 如果第一行所有值都在总体中位数以下，则抛出错误
        raise ValueError(f"All values are below the grand median ({grand_median}).")
    if rowsums[1] == 0:
        # 如果第二行所有值都在总体中位数以上，则抛出错误
        raise ValueError(f"All values are above the grand median ({grand_median}).")
    # 如果 `ties` 参数为 "ignore"，则需要检查是否所有样本中的值都等于总体中位数。
    # 这种情况下会导致 `table` 中某些列全部为零。在这里我们检查这种情况。
    zero_cols = np.nonzero((table == 0).all(axis=0))[0]
    # 如果存在全部为零的列，则抛出异常
    if len(zero_cols) > 0:
        msg = ("All values in sample %d are equal to the grand "
               "median (%r), so they are ignored, resulting in an "
               "empty sample." % (zero_cols[0] + 1, grand_median))
        raise ValueError(msg)

    # 计算卡方检验的统计量、p 值、自由度和期望频数
    stat, p, dof, expected = chi2_contingency(table, lambda_=lambda_,
                                              correction=correction)
    # 返回 MedianTestResult 对象，包括统计量、p 值、总体中位数和频数表
    return MedianTestResult(stat, p, grand_median, table)
def _circfuncs_common(samples, period, xp=None):
    # 如果 xp 为 None，则使用 array_namespace 函数将 samples 转换为数组
    xp = array_namespace(samples) if xp is None else xp

    # 如果 samples 的数据类型为整数类型，则将其转换为默认的浮点数类型
    if xp.isdtype(samples.dtype, 'integral'):
        dtype = xp.asarray(1.).dtype  # 获取默认的浮点数类型
        samples = xp.asarray(samples, dtype=dtype)

    # 将 samples 缩放为位于 0 到 2π 之间的弧度，并计算其正弦和余弦值
    scaled_samples = samples * ((2.0 * pi) / period)
    sin_samp = xp.sin(scaled_samples)
    cos_samp = xp.cos(scaled_samples)

    # 返回 samples、sin_samp 和 cos_samp 三个值
    return samples, sin_samp, cos_samp


@_axis_nan_policy_factory(
    # 返回一个函数，该函数将输入直接返回，输出一个值，无默认轴
    lambda x: x, n_outputs=1, default_axis=None,
    # 将结果转换为元组
    result_to_tuple=lambda x: (x,)
)
def circmean(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    r"""Compute the circular mean of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their *circular mean* is defined by ([1]_, Eq. 2.2.4)

    .. math::

       \mathrm{Arg} \left( \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right)

    where :math:`i` is the imaginary unit and :math:`\mathop{\mathrm{Arg}} z`
    gives the principal value of the argument of complex number :math:`z`,
    restricted to the range :math:`[0,2\pi]` by default.  :math:`z` in the
    above expression is known as the `mean resultant vector`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.

    Returns
    -------
    circmean : float
        Circular mean, restricted to the range ``[low, high]``.

        If the mean resultant vector is zero, an input-dependent,
        implementation-defined number between ``[low, high]`` is returned.
        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circstd : Circular standard deviation.
    circvar : Circular variance.

    References
    ----------
    .. [1] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    For readability, all angles are printed out in degrees.

    >>> import numpy as np
    >>> from scipy.stats import circmean
    >>> import matplotlib.pyplot as plt
    >>> angles = np.deg2rad(np.array([20, 30, 330]))
    >>> circmean = circmean(angles)
    >>> np.rad2deg(circmean)
    7.294976657784009

    >>> mean = angles.mean()
    >>> np.rad2deg(mean)
    126.66666666666666

    Plot and compare the circular mean against the arithmetic mean.

    >>> plt.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...          np.sin(np.linspace(0, 2*np.pi, 500)),
    ...          c='k')
    >>> plt.scatter(np.cos(angles), np.sin(angles), c='k')
    >>> plt.scatter(np.cos(circmean), np.sin(circmean), c='b',
    # 使用 matplotlib 库绘制散点图，显示给定数据点
    >>> plt.scatter(np.cos(mean), np.sin(mean), c='r', label='mean')
    # 添加图例，标记 'mean' 为红色点的含义
    >>> plt.legend()
    # 设置坐标轴比例相等，确保圆形数据显示正确
    >>> plt.axis('equal')
    # 显示绘制的图形
    >>> plt.show()

    """
    xp = array_namespace(samples)
    # 使用 array_namespace 函数处理 samples，确保兼容非NumPy数组以获得正确的NaN结果
    # 即使在数学上未定义，atan2(0, 0) 在这里返回 0
    if xp_size(samples) == 0:
        # 如果 samples 为空，直接返回 samples 的均值，沿指定轴
        return xp.mean(samples, axis=axis)
    period = high - low
    # 使用 _circfuncs_common 函数处理 samples，计算正弦和余弦值以及其他共同的操作
    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    # 沿指定轴求和正弦值
    sin_sum = xp.sum(sin_samp, axis=axis)
    # 沿指定轴求和余弦值
    cos_sum = xp.sum(cos_samp, axis=axis)
    # 使用 atan2 函数计算正弦和余弦和的反正切值
    res = xp.atan2(sin_sum, cos_sum)

    # 如果 res 是标量，则转换为 Python 对象
    res = res[()] if res.ndim == 0 else res
    # 将结果 res 转换到指定的周期范围内，然后加上低值以得到最终结果
    return (res * (period / (2.0 * pi)) - low) % period + low
# 使用装饰器 @_axis_nan_policy_factory 包装 circvar 函数，对输入参数进行预处理
@_axis_nan_policy_factory(
    # 通过 lambda 函数返回原始输入参数 x
    lambda x: x, n_outputs=1, default_axis=None,
    # 通过 result_to_tuple 函数将结果转换为单元素元组形式
    result_to_tuple=lambda x: (x,)
)
# 定义 circvar 函数，计算角度观测样本的圆形方差
def circvar(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    r"""Compute the circular variance of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their *circular variance* is defined by ([2]_, Eq. 2.3.3)

    .. math::

       1 - \left| \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right|

    where :math:`i` is the imaginary unit and :math:`|z|` gives the length
    of the complex number :math:`z`.  :math:`|z|` in the above expression
    is known as the `mean resultant length`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.

    Returns
    -------
    circvar : float
        Circular variance.  The returned value is in the range ``[0, 1]``,
        where ``0`` indicates no variance and ``1`` indicates large variance.

        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circmean : Circular mean.
    circstd : Circular standard deviation.

    Notes
    -----
    In the limit of small angles, the circular variance is close to
    half the 'linear' variance if measured in radians.

    References
    ----------
    .. [1] Fisher, N.I. *Statistical analysis of circular data*. Cambridge
           University Press, 1993.
    .. [2] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circvar
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circvar_1 = circvar(samples_1)
    >>> circvar_2 = circvar(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    >>> left.set_title(f"circular variance: {np.round(circvar_1, 2)!r}")
    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    >>> right.set_title(f"circular variance: {np.round(circvar_2, 2)!r}")
    >>> plt.show()

    """
    # 将样本转换为适当的数组命名空间
    xp = array_namespace(samples)
    # 计算角度周期
    period = high - low
    # 调用 _circfuncs_common 函数获取圆形函数的样本数据
    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    # 计算正弦样本的均值
    sin_mean = xp.mean(sin_samp, axis=axis)
    # 计算余弦样本的均值
    cos_mean = xp.mean(cos_samp, axis=axis)
    # 计算正弦和余弦均值构成的直角三角形的斜边长度（即 hypotenuse）
    hypotenuse = (sin_mean**2. + cos_mean**2.)**0.5
    # 由于四舍五入误差，hypotenuse 可能略大于 1
    # 使用 xp_minimum 函数确保 R 不大于 1
    R = xp_minimum(xp.asarray(1.), hypotenuse)

    # 计算并返回最终结果 res
    res = 1. - R
    return res
# 使用装饰器工厂函数 @_axis_nan_policy_factory 包装以下函数，该装饰器的参数如下：
#   - lambda x: x 是对输入参数的恒等映射
#   - n_outputs=1 表示函数返回一个输出值
#   - default_axis=None 表示默认的轴参数为 None
#   - result_to_tuple=lambda x: (x,) 将结果转换为单元素元组

def circstd(samples, high=2*pi, low=0, axis=None, nan_policy='propagate', *,
            normalize=False):
    r"""
    Compute the circular standard deviation of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their `circular standard deviation` is defined by
    ([2]_, Eq. 2.3.11)

    .. math::

       \sqrt{ -2 \log \left| \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right| }

    where :math:`i` is the imaginary unit and :math:`|z|` gives the length
    of the complex number :math:`z`.  :math:`|z|` in the above expression
    is known as the `mean resultant length`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.
    normalize : boolean, optional
        If ``False`` (the default), the return value is computed from the
        above formula with the input scaled by ``(2*pi)/(high-low)`` and
        the output scaled (back) by ``(high-low)/(2*pi)``.  If ``True``,
        the output is not scaled and is returned directly.

    Returns
    -------
    circstd : float
        Circular standard deviation, optionally normalized.

        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circmean : Circular mean.
    circvar : Circular variance.

    Notes
    -----
    In the limit of small angles, the circular standard deviation is close
    to the 'linear' standard deviation if ``normalize`` is ``False``.

    References
    ----------
    .. [1] Mardia, K. V. (1972). 2. In *Statistics of Directional Data*
       (pp. 18-24). Academic Press. :doi:`10.1016/C2013-0-07425-7`.
    .. [2] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circstd
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circstd_1 = circstd(samples_1)
    >>> circstd_2 = circstd(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    # 在左侧图表上绘制样本数据的余弦和正弦值散点图，使用黑色标记，每个点大小为15

    >>> left.set_title(f"circular std: {np.round(circstd_1, 2)!r}")
    # 设置左侧图表的标题，显示圆形标准差，保留两位小数

    >>> right.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...            np.sin(np.linspace(0, 2*np.pi, 500)),
    ...            c='k')
    # 在右侧图表上绘制一个圆形，分别使用余弦和正弦函数绘制500个点，颜色为黑色

    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    # 在右侧图表上绘制样本数据的余弦和正弦值散点图，使用黑色标记，每个点大小为15

    >>> right.set_title(f"circular std: {np.round(circstd_2, 2)!r}")
    # 设置右侧图表的标题，显示圆形标准差，保留两位小数

    >>> plt.show()
    # 显示所有绘制的图表

    """
    xp = array_namespace(samples)
    # 使用array_namespace处理样本数据，返回一个xp数组

    period = high - low
    # 计算周期，即高和低之间的差值

    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    # 调用_circfuncs_common函数处理样本数据，返回处理后的样本、正弦样本和余弦样本

    sin_mean = xp.mean(sin_samp, axis=axis)  # [1] (2.2.3)
    # 计算正弦样本的均值，沿着指定轴（axis）

    cos_mean = xp.mean(cos_samp, axis=axis)  # [1] (2.2.3)
    # 计算余弦样本的均值，沿着指定轴（axis）

    hypotenuse = (sin_mean**2. + cos_mean**2.)**0.5
    # 计算直角三角形的斜边长度，即正弦均值的平方加上余弦均值的平方，再开方

    # hypotenuse can go slightly above 1 due to rounding errors
    R = xp_minimum(xp.asarray(1.), hypotenuse)  # [1] (2.2.4)
    # 将计算得到的斜边长度与1进行比较，取较小值，用于后续计算

    res = (-2*xp.log(R))**0.5+0.0  # torch.pow returns -0.0 if R==1
    # 计算最终结果，根据斜边长度R的值进行对数运算，并进行开方，再加上0.0以确保结果为浮点数

    if not normalize:
        res *= (high-low)/(2.*pi)  # [1] (2.3.14) w/ (2.3.7)
        # 如果不进行归一化处理，则将结果乘以(high-low)/(2*pi)，根据特定的公式进行调整

    return res
    # 返回计算得到的最终结果
# 定义一个类 DirectionalStats，用于存储方向统计数据
class DirectionalStats:
    def __init__(self, mean_direction, mean_resultant_length):
        # 初始化方法，接收平均方向和平均结果长度作为参数，并存储在实例属性中
        self.mean_direction = mean_direction
        self.mean_resultant_length = mean_resultant_length

    def __repr__(self):
        # 返回对象的字符串表示，包括平均方向和平均结果长度
        return (f"DirectionalStats(mean_direction={self.mean_direction},"
                f" mean_resultant_length={self.mean_resultant_length})")


def directional_stats(samples, *, axis=0, normalize=True):
    """
    计算方向数据的样本统计信息。

    计算样本向量的方向平均值（也称为平均方向向量）和平均结果长度。

    方向平均值是向量数据的“首选方向”度量。它类似于样本均值，但在数据长度不重要时使用（例如单位向量）。

    平均结果长度是介于0和1之间的值，用于量化方向数据的离散程度：平均结果长度越小，离散程度越大。
    方向方差的多个定义涉及平均结果长度在文献[1]和[2]中有介绍。

    Parameters
    ----------
    samples : array_like
        输入数组。必须至少是二维的，输入的最后一个轴对应向量空间的维度。
        当输入恰好是二维时，这意味着数据的每一行是一个向量观察值。
    axis : int, 默认值: 0
        计算方向平均值的轴。
    normalize: boolean, 默认值: True
        如果为True，则对输入进行归一化，确保每个观测值是单位向量。如果观测值已经是单位向量，考虑将其设置为False以避免不必要的计算。

    Returns
    -------
    res : DirectionalStats
        包含以下属性的对象：

        mean_direction : ndarray
            方向平均值。
        mean_resultant_length : ndarray
            平均结果长度 [1]_。

    See Also
    --------
    circmean: 圆形均值；即2D角度的方向均值
    circvar: 圆形方差；即2D角度的方向方差

    Notes
    -----
    这里使用了文献[1]中的方向平均值的定义。
    假设观测值是单位向量，计算如下。

    .. code-block:: python

        mean = samples.mean(axis=0)
        mean_resultant_length = np.linalg.norm(mean)
        mean_direction = mean / mean_resultant_length

    这个定义适用于*方向*数据（即每个观测的大小不重要的向量数据），但不适用于*轴向*数据
    （即每个观测的大小和*符号*不重要的向量数据）。

    多种涉及平均结果长度 ``R`` 的方向方差定义已被提出，包括 ``1 - R`` [1]_，``1 - R**2``

    """
    # 计算样本的平均值
    mean = samples.mean(axis=axis)
    # 计算平均结果长度
    mean_resultant_length = np.linalg.norm(mean)
    # 计算平均方向
    mean_direction = mean / mean_resultant_length
    # 返回 DirectionalStats 对象，包含计算得到的平均方向和平均结果长度
    return DirectionalStats(mean_direction, mean_resultant_length)
    # 将输入的样本数据转换为 NumPy 数组
    samples = np.asarray(samples)
    
    # 检查样本数据的维度是否至少为二维，如果不是则抛出 ValueError 异常
    if samples.ndim < 2:
        raise ValueError("samples must at least be two-dimensional. "
                         f"Instead samples has shape: {samples.shape!r}")
    
    # 将数据轴移动到指定的位置
    samples = np.moveaxis(samples, axis, 0)
    
    # 如果指定了 normalize=True，则计算样本向量的范数，并进行归一化处理
    if normalize:
        vectornorms = np.linalg.norm(samples, axis=-1, keepdims=True)
        samples = samples / vectornorms
    
    # 计算样本数据的均值向量
    mean = np.mean(samples, axis=0)
    
    # 计算均值向量的平均结果长度（即模长）
    mean_resultant_length = np.linalg.norm(mean, axis=-1, keepdims=True)
    
    # 计算均值方向，即将均值向量归一化为单位向量
    mean_direction = mean / mean_resultant_length
    
    # 返回 DirectionalStats 对象，包括均值方向和均值结果长度（去除多余的维度）
    return DirectionalStats(mean_direction,
                            mean_resultant_length.squeeze(-1)[()])
# 定义函数，用于调整 p 值以控制假发现率（FDR）。
def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adjusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted according to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.
    """
    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.

    Examples
    --------
    We follow the example from [1]_.

        Thrombolysis with recombinant tissue-type plasminogen activator (rt-PA)
        and anisoylated plasminogen streptokinase activator (APSAC) in
        myocardial infarction has been proved to reduce mortality. [4]_
        investigated the effects of a new front-loaded administration of rt-PA
        versus those obtained with a standard regimen of APSAC, in a randomized
        multicentre trial in 421 patients with acute myocardial infarction.

    There were four families of hypotheses tested in the study, the last of
    which was "cardiac and other events after the start of thrombolitic
    treatment". FDR control may be desired in this family of hypotheses
    because it would not be appropriate to conclude that the front-loaded
    treatment is better if it is merely equivalent to the previous treatment.

    The p-values corresponding with the 15 hypotheses in this family were

    >>> ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    ...       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    If the chosen significance level is 0.05, we may be tempted to reject the
    null hypotheses for the tests corresponding with the first nine p-values,
    as the first nine p-values fall below the chosen significance level.
    However, this would ignore the problem of "multiplicity": if we fail to
    correct for the fact that multiple comparisons are being performed, we
    are more likely to incorrectly reject true null hypotheses.

    One approach to the multiplicity problem is to control the family-wise
    error rate (FWER), that is, the rate at which the null hypothesis is
    rejected when it is actually true. A common procedure of this kind is the
    Bonferroni correction [1]_.  We begin by multiplying the p-values by the
    number of hypotheses tested.

    >>> import numpy as np
    >>> np.array(ps) * len(ps)
    array([1.5000e-03, 6.0000e-03, 2.8500e-02, 1.4250e-01, 3.0150e-01,
           4.1700e-01, 4.4700e-01, 5.1600e-01, 6.8850e-01, 4.8600e+00,
           6.3930e+00, 8.5785e+00, 9.7920e+00, 1.1385e+01, 1.5000e+01])

    To control the FWER at 5%, we reject only the hypotheses corresponding
    with adjusted p-values less than 0.05. In this case, only the hypotheses
    corresponding with the first three p-values can be rejected. According to
    [1]_, these three hypotheses concerned "allergic reaction" and "two
    different aspects of bleeding."
    """
        An alternative approach is to control the false discovery rate: the
        expected fraction of rejected null hypotheses that are actually true. The
        advantage of this approach is that it typically affords greater power: an
        increased rate of rejecting the null hypothesis when it is indeed false. To
        control the false discovery rate at 5%, we apply the Benjamini-Hochberg
        p-value adjustment.
    
        >>> from scipy import stats
        >>> stats.false_discovery_control(ps)
        array([0.0015    , 0.003     , 0.0095    , 0.035625  , 0.0603    ,
               0.06385714, 0.06385714, 0.0645    , 0.0765    , 0.486     ,
               0.58118182, 0.714875  , 0.75323077, 0.81321429, 1.        ])
    
        Now, the first *four* adjusted p-values fall below 0.05, so we would reject
        the null hypotheses corresponding with these *four* p-values. Rejection
        of the fourth null hypothesis was particularly important to the original
        study as it led to the conclusion that the new treatment had a
        "substantially lower in-hospital mortality rate."
    """
    
    # Input Validation and Special Cases
    ps = np.asarray(ps)  # Convert `ps` to a NumPy array for manipulation
    
    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    # Check if all values in `ps` are between 0 and 1 inclusive
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")
    
    methods = {'bh', 'by'}
    # Ensure `method` is either 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()  # Convert `method` to lowercase for consistency
    
    if axis is None:
        axis = 0
        ps = ps.ravel()  # Flatten `ps` if `axis` is None for consistent handling
    
    axis = np.asarray(axis)[()]
    # Ensure `axis` is a single integer or None
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")
    
    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]  # Return the original `ps` if it's empty or has a single value
    
    ps = np.moveaxis(ps, axis, -1)  # Move `axis` to the last dimension
    m = ps.shape[-1]  # Get the number of hypotheses
    
    # Main Algorithm
    # Adjust p-values according to the selected method (Benjamini-Hochberg or Benjamini-Yekutieli)
    # This algorithm is akin to methods found in statistical literature and R's p.adjust function.
    
    # Sort p-values and adjust according to the False Discovery Rate (FDR) procedure
    order = np.argsort(ps, axis=-1)  # Indices that would sort `ps`
    ps = np.take_along_axis(ps, order, axis=-1)  # Sort `ps` along the last axis
    
    # Adjust p-values using the Benjamini-Hochberg procedure
    i = np.arange(1, m+1)
    ps *= m / i
    
    # Additional adjustment for the Benjamini-Yekutieli method
    if method == 'by':
        ps *= np.sum(1 / i)
    
    # Ensure FDR control by applying the minimum across the sorted p-values
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)
    
    # Restore the original order of axes and data in `ps`
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)
    
    return np.clip(ps, 0, 1)  # Clip p-values to be between 0 and 1
```