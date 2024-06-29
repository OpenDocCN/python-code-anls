# `.\numpy\numpy\lib\_histograms_impl.py`

```py
"""
Histogram-related functions
"""
# 导入必要的模块和库
import contextlib
import functools
import operator
import warnings

import numpy as np
from numpy._core import overrides

# 限定外部可访问的函数名
__all__ = ['histogram', 'histogramdd', 'histogram_bin_edges']

# 创建一个偏函数，用于处理数组函数调度，指定模块为 numpy
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# 保留内置的 range 函数，因为它是许多函数的关键字参数
_range = range


def _ptp(x):
    """Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    # 计算数组 x 的峰峰值，返回无符号整数结果以避免数据类型无法表示的问题
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(x, range):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    # 忽略未使用的 range 参数
    del range
    # 计算使用平方根法估计的直方图 bin 宽度
    return _ptp(x) / np.sqrt(x.size)


def _hist_bin_sturges(x, range):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    # 忽略未使用的 range 参数
    del range
    # 计算使用 Sturges 方法估计的直方图 bin 宽度
    return _ptp(x) / (np.log2(x.size) + 1.0)


def _hist_bin_rice(x, range):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    # 忽略未使用的 range 参数
    del range
    # 计算使用 Rice 方法估计的直方图 bin 宽度
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    # 忽略未使用的 range 参数
    del range
    # 计算使用 Scott 方法估计的直方图 bin 宽度
    return _ptp(x) / (np.std(x) * x.size ** (1.0 / 3))
    # 计算给定数据的最优箱宽的估计值
    h : An estimate of the optimal bin width for the given data.
    # 删除 range 变量，因为它未被使用
    del range  # unused
    # 根据数据 x 的大小计算最优箱宽的估计值，并返回结果
    return (24.0 * np.pi**0.5 / x.size)**(1.0 / 3.0) * np.std(x)
def _hist_bin_stone(x, range):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule

    This paper by Stone appears to be the origination of this rule.
    https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : (float, float)
        The lower and upper range of the bins.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """

    # 计算输入数据的大小
    n = x.size
    # 计算输入数据的极差
    ptp_x = _ptp(x)
    
    # 如果数据点数量小于等于1或极差为0，返回0
    if n <= 1 or ptp_x == 0:
        return 0

    # 定义函数 jhat，用于计算指定数量的直方图的集成平方误差估计
    def jhat(nbins):
        # 计算每个 bin 的宽度
        hh = ptp_x / nbins
        # 计算每个 bin 的概率密度
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        # 计算集成平方误差估计值
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    # 设置最大的 bin 数量上限
    nbins_upper_bound = max(100, int(np.sqrt(n)))
    # 选择使得 jhat 函数最小化的 bin 数量
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    
    # 如果选择的 bin 数量达到上限，发出警告
    if nbins == nbins_upper_bound:
        warnings.warn("The number of bins estimated may be suboptimal.",
                      RuntimeWarning, stacklevel=3)
    
    # 返回最优的 bin 宽度估计
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    
    # 删除未使用的 range 参数
    del range

    # 如果数据点数量大于2
    if x.size > 2:
        # 计算 Skewness g1
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        
        # 如果标准差大于0
        if sigma > 0.0:
            # 计算 g1 的平均值
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            
            # 返回 Doane's 公式计算的最优 bin 宽度估计
            return _ptp(x) / (1.0 + np.log2(x.size) +
                              np.log2(1.0 + np.absolute(g1) / sg1))
    
    # 若数据点数量不足2个或标准差为0，则返回0
    return 0.0


def _hist_bin_fd(x, range):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.
    """
    
    # 删除未使用的 range 参数
    del range

    # 返回 Freedman-Diaconis 公式计算的最优 bin 宽度估计
    return _ptp(x) / (2.0 * np.subtract(*np.percentile(x, [75, 25])) * np.power(x.size, -1/3))
    # 如果四分位距（IQR）为0，返回0作为箱宽度。
    # 箱宽度与数据大小的立方根成反比（渐近最优）。

    Parameters
    ----------
    x : array_like
        要制作直方图的输入数据，已经修剪到指定范围。不得为空。

    Returns
    -------
    h : 给定数据的最佳箱宽度的估计值。
    """
    # 删除变量 range，因为它未使用
    del range  # unused
    # 计算数据的四分位距（IQR）
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    # 根据数据大小计算并返回最佳箱宽度的估计值
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)
# 使用自由曼-迪亚孔斯和斯特吕格斯估算器的最小宽度来估算直方图的箱宽度，如果自由曼-迪亚孔斯估算器的箱宽度非零，则使用它。
# 如果自由曼-迪亚孔斯估算器得到的箱宽度为0，则使用斯特吕格斯估算器。
# 自由曼-迪亚孔斯估算器通常是最稳健的方法，但其宽度估算对小数据或有限方差的数据来说可能过大。
# 斯特吕格斯估算器在小数据集（<1000）中表现良好，在 R 语言中是默认的方法。此方法提供了良好的即用即得行为。

# 如果方差有限，IQR 可能为0，这会导致自由曼-迪亚孔斯估算器的箱宽度也为0。这不是有效的箱宽度，
# 因此 `np.histogram_bin_edges` 会选择1个箱，这可能不是最优选择。
# 如果 IQR 为0，则任何基于方差的估算器都不太可能有用，因此我们回归到斯特吕格斯估算器，它只使用数据集的大小进行计算。
def _hist_bin_auto(x, range):
    fd_bw = _hist_bin_fd(x, range)  # 使用自由曼-迪亚孔斯估算器计算箱宽度
    sturges_bw = _hist_bin_sturges(x, range)  # 使用斯特吕格斯估算器计算箱宽度
    del range  # 未使用的参数，删除之
    if fd_bw:
        return min(fd_bw, sturges_bw)  # 返回两个估算宽度的较小值
    else:
        # 方差有限，因此返回一个依赖长度的估算宽度
        return sturges_bw

# 模块加载时初始化的私有字典
_hist_bin_selectors = {'stone': _hist_bin_stone,  # 使用 stone 方法的直方图箱宽度估算器
                       'auto': _hist_bin_auto,    # 使用 auto 方法的直方图箱宽度估算器
                       'doane': _hist_bin_doane,  # 使用 doane 方法的直方图箱宽度估算器
                       'fd': _hist_bin_fd,        # 使用 fd 方法的直方图箱宽度估算器
                       'rice': _hist_bin_rice,    # 使用 rice 方法的直方图箱宽度估算器
                       'scott': _hist_bin_scott,  # 使用 scott 方法的直方图箱宽度估算器
                       'sqrt': _hist_bin_sqrt,    # 使用 sqrt 方法的直方图箱宽度估算器
                       'sturges': _hist_bin_sturges}  # 使用 sturges 方法的直方图箱宽度估算器


def _ravel_and_check_weights(a, weights):
    """ 检查 a 和 weights 的形状是否匹配，并将它们展平 """
    a = np.asarray(a)  # 将 a 转换为 NumPy 数组

    # 确保数组是可“减法”的数据类型
    if a.dtype == np.bool:
        warnings.warn("Converting input from {} to {} for compatibility."
                      .format(a.dtype, np.uint8),
                      RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)  # 将布尔类型转换为无符号整数类型

    if weights is not None:
        weights = np.asarray(weights)  # 将 weights 转换为 NumPy 数组
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')  # 抛出异常，如果 weights 的形状与 a 不匹配
        weights = weights.ravel()  # 展平 weights
    a = a.ravel()  # 展平 a
    return a, weights


def _get_outer_edges(a, range):
    """ 确定要使用的外部箱边缘，可以从数据或 range 参数中获取 """
    # 如果指定了范围参数，则进行范围检查和设置首尾边界
    if range is not None:
        # 解构赋值得到范围的首尾边界
        first_edge, last_edge = range
        # 如果首边界大于尾边界，抛出数值错误异常
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        # 如果首尾边界中有任一不是有限数，则抛出数值错误异常
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    # 如果数组 a 是空数组，则处理空数组情况，设定范围为 0 到 1
    elif a.size == 0:
        # 处理空数组情况，设定范围的首尾边界为 0 和 1
        first_edge, last_edge = 0, 1
    else:
        # 否则，自动检测数组 a 的范围，设定首尾边界为数组的最小值和最大值
        first_edge, last_edge = a.min(), a.max()
        # 如果自动检测得到的首尾边界中有任一不是有限数，则抛出数值错误异常
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # 扩展空范围以避免除以零的情况
    if first_edge == last_edge:
        # 如果首尾边界相等，则向首尾边界分别增加 0.5 和减少 0.5
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    # 返回计算得到的首尾边界
    return first_edge, last_edge
    # 解析重载的 bins 参数
    n_equal_bins = None
    bin_edges = None
    # 如果 bins 是字符串，表示使用自动方法确定的 bin 数量
    if isinstance(bins, str):
        bin_name = bins
        # 如果 bin_name 不在 _hist_bin_selectors 中，抛出 ValueError 异常
        if bin_name not in _hist_bin_selectors:
            raise ValueError(
                "{!r} is not a valid estimator for `bins`".format(bin_name))
        # 如果 weights 不为 None，不支持带权数据的自动估计 bin 数量，抛出 TypeError 异常
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        # 获取数据的外侧边界
        first_edge, last_edge = _get_outer_edges(a, range)

        # 如果指定了 range，根据范围截取数据
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        # 如果数据为空数组，设置默认的 bin 数量为 1
        if a.size == 0:
            n_equal_bins = 1
        else:
            # 根据选择器计算 bin 的宽度
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                # 对于整数类型的数组，确保宽度至少为 1
                if np.issubdtype(a.dtype, np.integer) and width < 1:
                    width = 1
                # 计算等宽 bin 的数量
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                # 对于某些估计器，如 FD 当数据的 IQR 为零时，宽度可能为零
                n_equal_bins = 1

    # 如果 bins 是零维数组，将其解释为整数数量的 bin
    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError(
                '`bins` must be an integer, a string, or an array') from e
        # 如果指定的 bin 数量小于 1，抛出 ValueError 异常
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        # 获取数据的外侧边界
        first_edge, last_edge = _get_outer_edges(a, range)

    # 如果 bins 是一维数组，将其解释为 bin 的边缘值
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        # 检查 bin 边缘值是否单调递增
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    # 如果 bins 不符合上述条件，抛出 ValueError 异常
    else:
        raise ValueError('`bins` must be 1d, when an array')

    # 如果成功确定了 bin 数量
    if n_equal_bins is not None:
        # 确定 bin 边缘值的数据类型，以确保类型一致性
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # 计算等间距的 bin 边缘值
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type)
        # 返回计算得到的 bin 边缘值以及相关信息
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        # 如果无法确定 bin 数量，则返回已有的 bin 边缘值及空信息
        return bin_edges, None
# 定义一个函数 `_search_sorted_inclusive`，用于在数组 `a` 中查找值 `v` 的位置，使得最后一个 `v` 的位置在右侧。
# 在直方图的上下文中，这样可以使最后一个箱边界是包含的。
def _search_sorted_inclusive(a, v):
    return np.concatenate((
        # 查找数组 `a` 中比 `v[:-1]` 中每个值大的最小索引，返回索引数组
        a.searchsorted(v[:-1], 'left'),
        # 查找数组 `a` 中比 `v[-1:]` 中每个值大或相等的最小索引，返回索引数组
        a.searchsorted(v[-1:], 'right')
    ))


# 定义一个调度函数 `_histogram_bin_edges_dispatcher`，返回元组 `(a, bins, weights)`
def _histogram_bin_edges_dispatcher(a, bins=None, range=None, weights=None):
    return (a, bins, weights)


# 使用装饰器 `array_function_dispatch` 注册 `histogram_bin_edges` 函数，调度函数为 `_histogram_bin_edges_dispatcher`
@array_function_dispatch(_histogram_bin_edges_dispatcher)
# 定义函数 `histogram_bin_edges`，计算用于 `histogram` 函数的直方图的箱边界
# 函数文档字符串说明了其功能及参数含义
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    r"""
    Function to calculate only the edges of the bins used by the `histogram`
    function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        If `bins` is a string from the list below, `histogram_bin_edges` will
        use the method chosen to calculate the optimal bin width and
        consequently the number of bins (see the Notes section for more detail
        on the estimators) from the data that falls within the requested range.
        While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Minimum bin width between the 'sturges' and 'fd' estimators. 
            Provides good all-around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that takes into account data variability
            and data size.

        'stone'
            Estimator based on leave-one-out cross-validation estimate of
            the integrated squared error. Can be regarded as a generalization
            of Scott's rule.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.
    # range : (float, float), optional
    #     箱线图的箱体范围，即数据分布的上下限。如果未提供，将使用数据的最小和最大值。
    #     超出此范围的值将被忽略。范围的第一个元素必须小于或等于第二个元素。
    #     `range` 也会影响自动计算的箱子数量。虽然箱宽是根据 `range` 内的实际数据计算的，
    #     但箱子的数量将填充整个范围，包括不包含数据的部分。

    # weights : array_like, optional
    #     与 `a` 具有相同形状的权重数组。`a` 中的每个值仅将其关联的权重贡献到箱子计数中（而不是 1）。
    #     目前没有任何箱子估算器使用这个参数，但将来可能会使用。

    # Returns
    # -------
    # bin_edges : array of dtype float
    #     传递给 `histogram` 的边界值数组。

    # See Also
    # --------
    # histogram

    # Notes
    # -----
    # 估算最佳箱子数量的方法在文献中有充分的基础，并受到了 R 提供的直方图可视化选择的启发。
    # 注意，将箱子数量与 :math:`n^{1/3}` 成比例是渐近最优的，这也是大多数估算器中的选择。
    # 这些仅仅是提供箱子数量的良好起始点的插入式方法。在下面的方程中，:math:`h` 是箱宽，:math:`n_h` 是箱子数量。
    # 所有计算箱子计数的估算器都使用数据的 `ptp` 重新调整到箱宽。最终的箱子数量是通过 ``np.round(np.ceil(range / h))`` 获得的。
    # 最终的箱宽通常小于估算器返回的宽度。

    # 'auto'（'sturges' 和 'fd' 估算器的最小箱宽）
    #     一个折中的选择，以获取一个良好的值。对于小数据集，通常会选择 Sturges 的值，
    #     而对于较大数据集，通常会默认使用 FD。避免了对小和大数据集分别过于保守的行为。
    #     切换点通常是 :math:`a.size \approx 1000`。

    # 'fd'（Freedman Diaconis 估算器）
    #     .. math:: h = 2 \frac{IQR}{n^{1/3}}
    #     
    #     箱宽与四分位间距（IQR）成正比，与 :math:`a.size` 的立方根成反比。对于大数据集效果很好，
    #     但对于小数据集可能过于保守。IQR 对异常值非常健壮。

    # 'scott'
    #     .. math:: h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}
    #     
    #     箱宽与数据的标准差成正比，与 ``x.size`` 的立方根成反比。对于大数据集效果很好，
    #     但对于小数据集可能过于保守。标准差对异常值不太健壮。其值在没有异常值时与 Freedman-Diaconis 估算器非常相似。
    # 将数组 `a` 和权重 `weights` 进行展平并检查
    a, weights = _ravel_and_check_weights(a, weights)
    
    # 根据输入的数据 `a`、分箱策略 `bins`、数据范围 `range` 和权重 `weights`，获取分箱的边界
    bin_edges, _ = _get_bin_edges(a, bins, range, weights)
    
    # 返回计算得到的分箱边界 `bin_edges`，用于直方图分析
    return bin_edges
# 定义函数 _histogram_dispatcher，用于直接返回传入的参数 a, bins, weights
def _histogram_dispatcher(
        a, bins=None, range=None, density=None, weights=None):
    return (a, bins, weights)

# 使用装饰器 array_function_dispatch 将 _histogram_dispatcher 与 histogram 函数关联
@array_function_dispatch(_histogram_dispatcher)
# 定义函数 histogram，用于计算数据集的直方图
def histogram(a, bins=10, range=None, density=None, weights=None):
    r"""
    计算数据集的直方图。

    Parameters
    ----------
    a : array_like
        输入数据。直方图计算将在扁平化的数组上进行。
    bins : int or sequence of scalars or str, optional
        如果 bins 是 int，则定义给定范围内的等宽 bins 的数量（默认为 10）。
        如果 bins 是 sequence，则定义一个单调递增的 bin 边缘数组，包括最右边的边缘，允许非均匀的 bin 宽度。

        .. versionadded:: 1.11.0

        如果 bins 是 str，则定义用于计算最佳 bin 宽度的方法，由 histogram_bin_edges 定义。

    range : (float, float), optional
        bins 的下限和上限。如果未提供，则 range 简单地为 (a.min(), a.max())。
        超出范围的值将被忽略。范围的第一个元素必须小于或等于第二个元素。range 也会影响自动 bin 计算。
        虽然根据实际数据计算范围内的最佳 bin 宽度，但 bin 计数将填充整个范围，包括不包含数据的部分。
    weights : array_like, optional
        与 a 相同形状的权重数组。a 中的每个值仅对 bin 计数贡献其关联的权重（而不是 1）。
        如果 density 为 True，则对权重进行归一化，使得范围内密度的积分保持为 1。
        请注意，weights 的 dtype 也将成为返回的累加器（hist）的 dtype，因此必须足够大以容纳累积值。
    density : bool, optional
        如果为 False，则结果将包含每个 bin 中的样本数。如果为 True，则结果是 bin 处概率密度函数的值，
        归一化使得范围内的积分为 1。请注意，直方图值的总和不会等于 1，除非选择单位宽度的 bins；它不是概率质量函数。

    Returns
    -------
    hist : array
        直方图的值。有关可能语义的描述，请参阅 density 和 weights。
        如果给定了 weights，则 hist.dtype 将从 weights 中获取。
    bin_edges : array of dtype float
        返回 bin 边缘的数组（长度为 hist+1）。

    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----
    所有但最后一个（最右侧的）bin 都是半开放的。换句话说，如果 bins 是::

      [1, 2, 3, 4]
    """
    a, weights = _ravel_and_check_weights(a, weights)
    
    # 将输入数组 a 和权重 weights 展平并检查它们的格式
    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
    
    # 根据输入数组 a、分bin规则 bins、范围 range 和权重 weights 获取 bin 边界和 uniform_bins
    # uniform_bins 是一个布尔值，指示是否使用均匀分bin
    
    # 根据权重的数据类型选择直方图数组的数据类型
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype
    
    # 设置块大小，以便在计算直方图时能够迭代处理块，从而最小化内存使用
    BLOCK = 65536
    
    # 快速路径使用 bincount 函数，但仅适用于特定类型的权重
    simple_weights = (
        weights is None or
        np.can_cast(weights.dtype, np.double) or
        np.can_cast(weights.dtype, complex)
    )
    """
    # 如果 uniform_bins 和 simple_weights 都不为 None，则执行以下代码块
    if uniform_bins is not None and simple_weights:
        # 使用快速算法生成等宽的直方图
        # 这里假设了等宽的箱体，基于这个假设转换 a 中的值为箱子的索引是有效的

        # 从 uniform_bins 中解包出第一个边缘、最后一个边缘和等宽箱子的数量
        first_edge, last_edge, n_equal_bins = uniform_bins

        # 初始化一个空的直方图 n，长度为 n_equal_bins，数据类型为 ntype
        n = np.zeros(n_equal_bins, ntype)

        # 预先计算直方图的缩放因子
        norm_numerator = n_equal_bins
        norm_denom = _unsigned_subtract(last_edge, first_edge)

        # 我们在这里迭代块的原因有两个：首先，对于大数组来说，这样做实际上更快（例如对于一个 10^8 大小的数组，快两倍），
        # 其次，在大数组的极限情况下，它会降低内存占用 3 倍。
        for i in _range(0, len(a), BLOCK):
            # 从数组 a 中获取块大小为 BLOCK 的子数组 tmp_a
            tmp_a = a[i:i+BLOCK]

            # 如果 weights 为 None，则 tmp_w 也为 None；否则，获取对应的 weights 子数组 tmp_w
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]

            # 仅保留落在指定范围内的值
            keep = (tmp_a >= first_edge)
            keep &= (tmp_a <= last_edge)
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]

            # 确保类型转换在这里执行，以避免下面出现不可预测的精度错误
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)

            # 计算箱子的索引，对于恰好在 last_edge 上的值，需要减去一个
            f_indices = ((_unsigned_subtract(tmp_a, first_edge) / norm_denom)
                         * norm_numerator)
            indices = f_indices.astype(np.intp)
            indices[indices == n_equal_bins] -= 1

            # 索引计算可能在箱子边缘附近 ±1 个单位内不一致

            # 如果 tmp_a 小于 bin_edges[indices]，则减 1
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1

            # 最后一个箱子包括右边缘，其它箱子不包括
            increment = ((tmp_a >= bin_edges[indices + 1])
                         & (indices != n_equal_bins - 1))
            indices[increment] += 1

            # 使用 bincount 计算直方图
            if ntype.kind == 'c':
                # 如果 ntype 的种类是复数，则分别对实部和虚部进行加权求和
                n.real += np.bincount(indices, weights=tmp_w.real,
                                      minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag,
                                      minlength=n_equal_bins)
            else:
                # 否则，直接对 n 进行加权求和，并转换为 ntype 类型
                n += np.bincount(indices, weights=tmp_w,
                                 minlength=n_equal_bins).astype(ntype)
    # 如果指定了 density 参数，则计算密度而不是直方图
    if density:
        # 计算直方图的累积分布
        cum_n = np.zeros(bin_edges.shape, ntype)
        # 如果未提供权重，则按块排序并计算累积直方图
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                # 对每个块的数据进行排序
                sa = np.sort(a[i:i+BLOCK])
                # 将排序后的数据加入累积直方图
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                # 按块对数据和权重进行排序
                tmp_a = a[i:i+BLOCK]
                tmp_w = weights[i:i+BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                # 计算权重的累积和
                cw = np.concatenate((zero, sw.cumsum()))
                # 找到排序后的数据在分 bin_edges 中的位置并加入累积直方图
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]

        # 计算直方图的值，即累积分布的差值
        n = np.diff(cum_n)

        # 计算直方图密度，并返回密度和 bin_edges
        db = np.array(np.diff(bin_edges), float)
        return n/db/n.sum(), bin_edges

    # 如果没有指定 density 参数，则计算普通直方图
    # 计算直方图的累积分布
    cum_n = np.zeros(bin_edges.shape, ntype)
    # 如果未提供权重，则按块排序并计算累积直方图
    if weights is None:
        for i in _range(0, len(a), BLOCK):
            # 对每个块的数据进行排序
            sa = np.sort(a[i:i+BLOCK])
            # 将排序后的数据加入累积直方图
            cum_n += _search_sorted_inclusive(sa, bin_edges)
    else:
        zero = np.zeros(1, dtype=ntype)
        for i in _range(0, len(a), BLOCK):
            # 按块对数据和权重进行排序
            tmp_a = a[i:i+BLOCK]
            tmp_w = weights[i:i+BLOCK]
            sorting_index = np.argsort(tmp_a)
            sa = tmp_a[sorting_index]
            sw = tmp_w[sorting_index]
            # 计算权重的累积和
            cw = np.concatenate((zero, sw.cumsum()))
            # 找到排序后的数据在分 bin_edges 中的位置并加入累积直方图
            bin_index = _search_sorted_inclusive(sa, bin_edges)
            cum_n += cw[bin_index]

    # 计算直方图的值，即累积分布的差值
    n = np.diff(cum_n)

    # 返回直方图和 bin_edges
    return n, bin_edges
# 定义 _histogramdd_dispatcher 函数，用于根据输入参数的类型分派到合适的处理函数
def _histogramdd_dispatcher(sample, bins=None, range=None, density=None,
                            weights=None):
    # 如果 sample 具有 shape 属性，则返回该样本数据（与 histogramdd 中的条件相同）
    if hasattr(sample, 'shape'):  
        yield sample  # 返回 sample
    else:
        yield from sample  # 否则，逐个返回 sample 中的元素
    # 使用上下文管理器忽略 TypeError 异常
    with contextlib.suppress(TypeError):
        yield from bins  # 逐个返回 bins 中的元素
    yield weights  # 返回 weights


# 通过 array_function_dispatch 装饰器将 _histogramdd_dispatcher 函数与 histogramdd 函数关联起来
@array_function_dispatch(_histogramdd_dispatcher)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : (N, D) array, or (N, D) array_like
        The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.

        The first form should be preferred.

    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if density is True. If density is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See density and weights
        for the different possible semantics.
    edges : tuple of ndarrays
        A tuple of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> r = rng.normal(size=(100,3))
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    try:
        # 尝试获取样本数据的形状信息
        N, D = sample.shape
    except (AttributeError, ValueError):
        # 如果捕获到 AttributeError 或者 ValueError 异常，则执行以下代码块
        # 将 sample 转换为至少是二维数组
        sample = np.atleast_2d(sample).T
        # 获取样本的行数 N 和维度 D

    nbin = np.empty(D, np.intp)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        # 如果 weights 不为 None，则将其转换为 NumPy 数组
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            # 如果 bins 的长度 M 不等于样本的维度 D，则抛出 ValueError 异常
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                'sample x.')
    except TypeError:
        # 如果 bins 是一个整数，则将其扩展为包含 D 个元素的列表
        bins = D*[bins]

    # 标准化 range 参数
    if range is None:
        # 如果 range 为 None，则设置为 D 个 None 组成的元组
        range = (None,) * D
    elif len(range) != D:
        # 如果 range 的长度不等于样本的维度 D，则抛出 ValueError 异常
        raise ValueError('range argument must have one entry per dimension')

    # 创建边缘数组
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            # 如果 bins[i] 是标量
            if bins[i] < 1:
                # 如果 bins[i] 小于 1，则抛出 ValueError 异常
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            # 获取 sample[:,i] 的最小值 smin 和最大值 smax
            smin, smax = _get_outer_edges(sample[:,i], range[i])
            try:
                # 尝试将 bins[i] 转换为整数
                n = operator.index(bins[i])

            except TypeError as e:
                # 如果失败，则抛出 TypeError 异常
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            # 使用 linspace 生成边缘数组 edges[i]
            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            # 如果 bins[i] 是一维数组，则直接赋值给 edges[i]
            edges[i] = np.asarray(bins[i])
            # 检查 edges[i] 是否严格单调递增
            if np.any(edges[i][:-1] > edges[i][1:]):
                # 如果不是单调递增，则抛出 ValueError 异常
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            # 如果 bins[i] 不是标量或一维数组，则抛出 ValueError 异常
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        # 计算 nbin[i]，包括两个边界点
        nbin[i] = len(edges[i]) + 1  # 包括每个边界上的一个点

        # 计算 dedges[i]，edges[i] 中每相邻两点的差值
        dedges[i] = np.diff(edges[i])

    # 计算每个样本落入的箱子编号
    Ncount = tuple(
        # 避免使用 np.digitize 来解决 gh-11022 的问题
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # 使用 digitize 函数，将落在边缘上的值放入右侧的箱中
    # 对于最右侧的箱子，希望等于右边界的值计入最后一个箱子，而不算作异常值。
    for i in _range(D):
        # 找出落在最右侧边界上的点
        on_edge = (sample[:, i] == edges[i][-1])
        # 将这些点向左移动一个箱子
        Ncount[i][on_edge] -= 1

    # 计算在扁平化直方图矩阵中的样本索引
    # 如果数组过大，这会引发错误
    xy = np.ravel_multi_index(Ncount, nbin)

    # 计算 xy 中每个值的重复次数，并将其分配给扁平化的 histmat
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # 转换成正确形状的矩阵
    hist = hist.reshape(nbin)

    # 暂时保留 gh-7845 中观察到的（不良）行为
    hist = hist.astype(float, casting='safe')
    # 去除异常值（每个维度的第一个和最后一个索引）后的核心数据
    core = D*(slice(1, -1),)
    # 根据核心数据重新定义直方图
    hist = hist[core]

    if density:
        # 计算概率密度函数
        s = hist.sum()
        for i in _range(D):
            # 创建形状数组，用于除法操作
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        # 归一化直方图以得到概率密度
        hist /= s

    if (hist.shape != nbin - 2).any():
        # 如果直方图形状与预期形状不符，则抛出运行时错误
        raise RuntimeError(
            "Internal Shape Error")
    # 返回处理后的直方图和边界
    return hist, edges
```