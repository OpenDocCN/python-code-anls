# `D:\src\scipysrc\scipy\scipy\stats\_mstats_extras.py`

```
"""
Additional statistics functions with support for masked arrays.

"""

# Original author (2007): Pierre GF Gerard-Marchant

# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
from numpy import float64, ndarray  # 导入特定的 NumPy 类型

import numpy.ma as ma  # 导入 NumPy 的 masked array 模块
from numpy.ma import MaskedArray  # 导入 MaskedArray 类

from . import _mstats_basic as mstats  # 导入本地模块 _mstats_basic

from scipy.stats.distributions import norm, beta, t, binom  # 从 SciPy 中导入分布函数

def hdquantiles(data, prob=list([.25,.5,.75]), axis=None, var=False,):
    """
    Computes quantile estimates with the Harrell-Davis method.

    The quantile estimates are calculated as a weighted linear combination
    of order statistics.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of probabilities at which to compute the quantiles.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdquantiles : MaskedArray
        A (p,) array of quantiles (if `var` is False), or a (2,p) array of
        quantiles and variances (if `var` is True), where ``p`` is the
        number of quantiles.

    See Also
    --------
    hdquantiles_sd

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import hdquantiles
    >>>
    >>> # Sample data
    >>> data = np.array([1.2, 2.5, 3.7, 4.0, 5.1, 6.3, 7.0, 8.2, 9.4])
    >>>
    >>> # Probabilities at which to compute quantiles
    >>> probabilities = [0.25, 0.5, 0.75]
    >>>
    >>> # Compute Harrell-Davis quantile estimates
    >>> quantile_estimates = hdquantiles(data, prob=probabilities)
    >>>
    >>> # Display the quantile estimates
    >>> for i, quantile in enumerate(probabilities):
    ...     print(f"{int(quantile * 100)}th percentile: {quantile_estimates[i]}")
    25th percentile: 3.1505820231763066 # may vary
    50th percentile: 5.194344084883956
    75th percentile: 7.430626414674935

    """
    # 定义一个名为 _hd_1D 的函数，用于计算一维数组的 HD 分位数，对于无效数据返回 nan
    def _hd_1D(data, prob, var):
        "Computes the HD quantiles for a 1D array. Returns nan for invalid data."
        # 将压缩后的数据转换为 ndarray，并按升序排序，然后压缩为一维数组
        xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
        # 获取数组的长度，不使用 len(data)，以避免处理 numpy 标量的问题
        n = xsorted.size

        # 初始化一个形状为 (2, len(prob)) 的空数组，数据类型为 float64
        hd = np.empty((2, len(prob)), float64)

        # 如果数组长度小于 2，则将 hd 数组所有元素设为 nan
        if n < 2:
            hd.flat = np.nan
            # 如果 var 为真，直接返回 hd 数组
            if var:
                return hd
            # 否则返回 hd 数组的第一行
            return hd[0]

        # 计算等分点向量 v
        v = np.arange(n + 1) / float(n)
        # 为 beta 分布的 cdf 函数取一个别名
        betacdf = beta.cdf

        # 遍历概率数组 prob
        for (i, p) in enumerate(prob):
            # 计算权重向量 w
            _w = betacdf(v, (n + 1) * p, (n + 1) * (1 - p))
            w = _w[1:] - _w[:-1]
            # 计算 hd_mean，即加权平均值
            hd_mean = np.dot(w, xsorted)
            hd[0, i] = hd_mean
            #
            # 计算 hd 的第二行，即加权平方偏差
            hd[1, i] = np.dot(w, (xsorted - hd_mean) ** 2)
            #

        # 处理概率为 0 或 1 的情况
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]

        # 如果 var 为真，则设置第二行对应位置为 nan，并返回 hd 数组
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = np.nan
            return hd

        # 否则返回 hd 数组的第一行
        return hd[0]

    # 初始化 & 检查
    # 将 data 转换为浮点型的 ma 数组，并且不复制数据
    data = ma.array(data, copy=False, dtype=float64)
    # 将 prob 转换为至少一维的 ndarray 数组
    p = np.atleast_1d(np.asarray(prob))

    # 如果 axis 为 None 或者 data 的维度为 1，则在全局或者沿着轴计算 quantiles
    if (axis is None) or (data.ndim == 1):
        result = _hd_1D(data, p, var)
    else:
        # 如果 data 的维度大于 2，则抛出 ValueError
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        # 否则在指定的 axis 上应用 _hd_1D 函数，计算 quantiles
        result = ma.apply_along_axis(_hd_1D, axis, data, p, var)

    # 返回修正无效值后的 result 结果
    return ma.fix_invalid(result, copy=False)
# 定义函数，计算给定数据沿指定轴的Harrell-Davis中位数估计值
def hdmedian(data, axis=-1, var=False):
    # 调用hdquantiles函数计算中位数估计值，默认返回结果去除多余的轴
    result = hdquantiles(data,[0.5], axis=axis, var=var)
    # 压缩结果以去除多余的轴并返回
    return result.squeeze()


# 定义函数，计算给定数据的Harrell-Davis分位数估计的标准误差，通过jackknife方法
def hdquantiles_sd(data, prob=list([.25,.5,.75]), axis=None):
    # 定义内部函数，计算一维数组的标准误差
    def _hdsd_1D(data, prob):
        # 对压缩后的数据进行排序
        xsorted = np.sort(data.compressed())
        n = len(xsorted)

        # 初始化标准误差数组
        hdsd = np.empty(len(prob), float64)
        # 如果数据点少于2个，标准误差设为NaN
        if n < 2:
            hdsd.flat = np.nan

        # 计算权重
        vv = np.arange(n) / float(n-1)
        betacdf = beta.cdf

        # 遍历每个期望概率计算相应的标准误差
        for (i,p) in enumerate(prob):
            _w = betacdf(vv, n*p, n*(1-p))
            w = _w[1:] - _w[:-1]
            # 权重累积和以及相应数据点的累积和（使用jackknife方法）
            mx_ = np.zeros_like(xsorted)
            mx_[1:] = np.cumsum(w * xsorted[:-1])
            mx_[:-1] += np.cumsum(w[::-1] * xsorted[:0:-1])[::-1]
            # 计算标准误差
            hdsd[i] = np.sqrt(mx_.var() * (n - 1))
        return hdsd

    # 初始化和检查数据
    data = ma.array(data, copy=False, dtype=float64)
    p = np.atleast_1d(np.asarray(prob))

    # 根据轴向计算分位数（或全局计算）
    if axis is None:
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)

    # 修正无效值并将结果展平返回
    return ma.fix_invalid(result, copy=False).ravel()


# 定义函数，计算给定数据沿指定轴的修剪均值的选定置信区间
def trimmed_mean_ci(data, limits=(0.2,0.2), inclusive=(True,True),
                    alpha=0.05, axis=None):
    """
    Selected confidence interval of the trimmed mean along the given axis.

    Parameters
    ----------
    data : array_like
        Input data.

    limits : tuple, optional
        Percentile limits for trimming. Default is (0.2, 0.2).

    inclusive : tuple, optional
        Whether the trimming is inclusive or exclusive. Default is (True, True).

    alpha : float, optional
        Significance level for the confidence interval. Default is 0.05.

    axis : int, optional
        Axis along which to compute the trimmed mean.

    Returns
    -------
    ci : ndarray
        Confidence intervals for the trimmed mean.
    """
    # 将输入数据转换为遮盖数组，以便处理遮盖数据的统计计算
    data = ma.array(data, copy=False)
    # 对数据进行修剪，去除指定百分比范围内的数据，并返回修剪后的遮盖数组
    trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
    # 计算修剪后数据的均值，沿指定轴进行计算
    tmean = trimmed.mean(axis)
    # 计算修剪后数据的修剪标准误差
    tstde = mstats.trimmed_stde(data, limits=limits, inclusive=inclusive, axis=axis)
    # 计算修剪后数据的自由度，用于 t 分布的置信区间计算
    df = trimmed.count(axis) - 1
    # 根据置信水平 alpha 和自由度 df 计算 t 分布的双侧置信区间分位点
    tppf = t.ppf(1 - alpha / 2., df)
    # 返回修剪后数据的下限和上限置信区间作为 NumPy 数组
    return np.array((tmean - tppf * tstde, tmean + tppf * tstde))
# 定义函数 `mjci`，计算数据的 Maritz-Jarrett 估计量的标准误差
def mjci(data, prob=[0.25,0.5,0.75], axis=None):
    """
    Returns the Maritz-Jarrett estimators of the standard error of selected
    experimental quantiles of the data.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    """
    # 定义内部函数 `_mjci_1D`，计算一维数据的 Maritz-Jarrett 估计量
    def _mjci_1D(data, p):
        # 压缩掩码数据并排序
        data = np.sort(data.compressed())
        n = data.size
        # 计算分位数的位置
        prob = (np.array(p) * n + 0.5).astype(int)
        betacdf = beta.cdf

        # 初始化存储 Maritz-Jarrett 估计量的数组
        mj = np.empty(len(prob), float64)
        x = np.arange(1,n+1, dtype=float64) / n
        y = x - 1./n
        # 计算每个分位数的 Maritz-Jarrett 估计量
        for (i,m) in enumerate(prob):
            W = betacdf(x,m-1,n-m) - betacdf(y,m-1,n-m)
            C1 = np.dot(W,data)
            C2 = np.dot(W,data**2)
            mj[i] = np.sqrt(C2 - C1**2)
        return mj

    # 转换数据为掩码数组
    data = ma.array(data, copy=False)
    if data.ndim > 2:
        raise ValueError("Array 'data' must be at most two dimensional, "
                         "but got data.ndim = %d" % data.ndim)

    p = np.atleast_1d(np.asarray(prob))
    # 如果未指定 axis，则在全局或者平铺的数组上计算分位数
    if (axis is None):
        return _mjci_1D(data, p)
    else:
        # 在指定的轴上应用 `_mjci_1D` 函数
        return ma.apply_along_axis(_mjci_1D, axis, data, p)


# 定义函数 `mquantiles_cimj`，使用 Maritz-Jarrett 估计器计算数据的选定分位数的 alpha 置信区间
def mquantiles_cimj(data, prob=[0.25,0.50,0.75], alpha=0.05, axis=None):
    """
    Computes the alpha confidence interval for the selected quantiles of the
    data, with Maritz-Jarrett estimators.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles.
        If None, use a flattened array.

    Returns
    -------
    ci_lower : ndarray
        The lower boundaries of the confidence interval.  Of the same length as
        `prob`.
    ci_upper : ndarray
        The upper boundaries of the confidence interval.  Of the same length as
        `prob`.

    """
    # 确保 alpha 在 [0, 1] 之间
    alpha = min(alpha, 1 - alpha)
    # 计算正态分布的 z 值
    z = norm.ppf(1 - alpha/2.)
    # 计算选定分位数的分位数和 Maritz-Jarrett 估计量
    xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    # 返回置信区间的下界和上界
    return (xq - z * smj, xq + z * smj)


# 定义函数 `median_cihs`，使用 Hettmasperger-Sheather 方法计算数据中位数的 alpha 置信区间
def median_cihs(data, alpha=0.05, axis=None):
    """
    Computes the alpha-level confidence interval for the median of the data.

    Uses the Hettmasperger-Sheather method.

    Parameters
    ----------
    data : array_like
        Input data. Masked values are discarded. The input should be 1D only,
        or `axis` should be set to None.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    ```
    median_cihs
        Alpha level confidence interval.

    """
    # 定义内部函数 _cihs_1D，用于计算一维数据的置信区间
    def _cihs_1D(data, alpha):
        # 对数据进行排序并去除压缩空值
        data = np.sort(data.compressed())
        # 获取数据长度
        n = len(data)
        # 确保 alpha 小于等于 0.5
        alpha = min(alpha, 1-alpha)
        # 计算 k 值，使用二项分布的逆累积分布函数
        k = int(binom._ppf(alpha/2., n, 0.5))
        # 计算 gk，是指定概率处的二项分布累积分布函数值
        gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        # 如果 gk 小于 1-alpha，则调整 k 和 gk
        if gk < 1-alpha:
            k -= 1
            gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        # 计算 gkk，是指定概率处的二项分布累积分布函数值
        gkk = binom.cdf(n-k-1,n,0.5) - binom.cdf(k,n,0.5)
        # 计算置信区间
        I = (gk - 1 + alpha)/(gk - gkk)
        lambd = (n-k) * I / float(k + (n-2*k)*I)
        # 计算置信区间的上下限
        lims = (lambd*data[k] + (1-lambd)*data[k-1],
                lambd*data[n-k-1] + (1-lambd)*data[n-k])
        return lims
    
    # 将数据转换为 MaskedArray 对象，如果已经是，则不进行复制
    data = ma.array(data, copy=False)
    
    # 如果没有指定轴，则对整体数据计算置信区间
    if (axis is None):
        result = _cihs_1D(data, alpha)
    else:
        # 如果数据维度大于 2，则抛出错误
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        # 沿指定轴应用 _cihs_1D 函数计算置信区间
        result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)

    # 返回计算结果
    return result
def idealfourths(data, axis=None):
    """
    Returns an estimate of the lower and upper quartiles.

    Uses the ideal fourths algorithm.

    Parameters
    ----------
    data : array_like
        Input array.
    axis : int, optional
        Axis along which the quartiles are estimated. If None, the arrays are
        flattened.

    Returns
    -------
    idealfourths : {list of floats, masked array}
        Returns the two internal values that divide `data` into four parts
        using the ideal fourths algorithm either along the flattened array
        (if `axis` is None) or along `axis` of `data`.

    """
    # 定义内部函数 `_idf`，用于计算理想四分位数
    def _idf(data):
        # 压缩掉数据中的缺失值
        x = data.compressed()
        # 获取数据长度
        n = len(x)
        # 如果数据长度小于 3，返回 NaN 值
        if n < 3:
            return [np.nan,np.nan]
        # 计算理想四分位数的位置
        (j,h) = divmod(n/4. + 5/12.,1)
        j = int(j)
        # 计算下四分位数
        qlo = (1-h)*x[j-1] + h*x[j]
        # 计算上四分位数
        k = n - j
        qup = (1-h)*x[k] + h*x[k-1]
        return [qlo, qup]
    
    # 对输入数据按指定轴进行排序，并转换成 MaskedArray 对象
    data = ma.sort(data, axis=axis).view(MaskedArray)
    # 如果未指定轴，则调用 `_idf` 函数计算理想四分位数
    if (axis is None):
        return _idf(data)
    else:
        # 如果不是第一种情况，则应用 _idf 函数沿指定轴向数组 data
        return ma.apply_along_axis(_idf, axis, data)
def rsh(data, points=None):
    """
    Evaluates Rosenblatt's shifted histogram estimators for each data point.

    Rosenblatt's estimator is a centered finite-difference approximation to the
    derivative of the empirical cumulative distribution function.

    Parameters
    ----------
    data : sequence
        Input data, should be 1-D. Masked values are ignored.
    points : sequence or None, optional
        Sequence of points where to evaluate Rosenblatt shifted histogram.
        If None, use the data.

    """

    # 将输入的数据转换为 MaskedArray，以便处理掩码值
    data = ma.array(data, copy=False)

    # 如果 points 参数为 None，则使用 data 自身作为评估点
    if points is None:
        points = data
    else:
        # 将 points 转换为至少为 1 维的 ndarray
        points = np.atleast_1d(np.asarray(points))

    # 如果输入数据不是 1 维，则抛出异常
    if data.ndim != 1:
        raise AttributeError("The input array should be 1D only !")

    # 计算数据点的数量（忽略掩码值）
    n = data.count()

    # 计算数据的四分位范围
    r = idealfourths(data, axis=None)

    # 计算带宽 h，采用 Rosenblatt 方法
    h = 1.2 * (r[-1]-r[0]) / n**(1./5)

    # 计算在每个评估点上，数据小于等于该点加上 h 的数量
    nhi = (data[:,None] <= points[None,:] + h).sum(0)

    # 计算在每个评估点上，数据小于该点减去 h 的数量
    nlo = (data[:,None] < points[None,:] - h).sum(0)

    # 计算 Rosenblatt shifted histogram 估计值并返回
    return (nhi-nlo) / (2.*n*h)
```