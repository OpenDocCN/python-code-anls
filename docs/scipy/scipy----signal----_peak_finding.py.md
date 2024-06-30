# `D:\src\scipysrc\scipy\scipy\signal\_peak_finding.py`

```
"""
Functions for identifying peaks in signals.
"""
# 导入必要的库
import math
import numpy as np

# 导入从 Scipy 中的信号处理模块中所需的函数和类
from scipy.signal._wavelets import _cwt, _ricker
from scipy.stats import scoreatpercentile

# 导入本地定义的一些辅助函数
from ._peak_finding_utils import (
    _local_maxima_1d,
    _select_by_peak_distance,
    _peak_prominences,
    _peak_widths
)

# 指定可以被外部访问的函数和类
__all__ = ['argrelmin', 'argrelmax', 'argrelextrema', 'peak_prominences',
           'peak_widths', 'find_peaks', 'find_peaks_cwt']


def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See numpy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._peak_finding import _boolrelextrema
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)

    """
    # 检查 order 参数的有效性
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    # 获取数据的长度
    datalen = data.shape[axis]
    # 创建一个包含所有索引的数组
    locs = np.arange(0, datalen)

    # 初始化结果数组，全为 True
    results = np.ones(data.shape, dtype=bool)
    # 取出主要的数据点
    main = data.take(locs, axis=axis, mode=mode)
    # 迭代比较每个点及其相邻的点
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        # 如果结果中没有 True 的值，则返回结果
        if ~results.any():
            return results
    return results


def argrelmin(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative minima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative minima.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).

    """
    # 此函数将在下一行代码之后继续解释
    mode : str, optional
        边界处理模式，指定向量边界的处理方式。
        可选选项为 'wrap'（循环处理）或 'clip'（溢出处理为最后（或第一个）元素）。
        默认为 'clip'。参见 numpy.take。

    Returns
    -------
    extrema : tuple of ndarrays
        整数数组的极小值的索引。``extrema[k]`` 是 `data` 的第 `k` 轴的索引数组。
        即使 `data` 是 1-D 数组，返回值也是一个元组。

    See Also
    --------
    argrelextrema, argrelmax, find_peaks

    Notes
    -----
    此函数使用 `argrelextrema` 并以 np.less 作为比较器。因此，需要在值的两侧使用严格的不等式才能将其视为极小值。
    这意味着平坦的极小值（超过一个样本宽度）不会被检测到。
    对于 1-D `data`，可以通过对负 `data` 调用 `find_peaks` 来检测所有的局部极小值，包括平坦的极小值。

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import argrelmin
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    return argrelextrema(data, np.less, axis, order, mode)
# 计算给定数据的相对最大值的索引
def argrelmax(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated.
        Available options are 'wrap' (wrap around) or 'clip' (treat overflow
        as the same as the last (or first) element).
        Default 'clip'. See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelextrema, argrelmin, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with np.greater as comparator. Therefore,
    it requires a strict inequality on both sides of a value to consider it a
    maximum. This means flat maxima (more than one sample wide) are not detected.
    In case of 1-D `data` `find_peaks` can be used to detect all
    local maxima, including flat ones.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import argrelmax
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    # 调用 `argrelextrema` 函数，使用 np.greater 作为比较器
    return argrelextrema(data, np.greater, axis, order, mode)


# 计算给定数据的相对极值的索引
def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default is 'clip'. See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelmin, argrelmax

    Notes
    -----

    .. versionadded:: 0.11.0

    Examples
    --------
    """
    # 此函数计算给定数据的相对极值，具体实现依赖于传入的比较器函数
    pass
    # 调用函数前导入必要的库和模块
    >>> import numpy as np
    >>> from scipy.signal import argrelextrema
    
    # 创建一个 NumPy 数组 x，包含整数元素
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    
    # 使用 argrelextrema 函数找到 x 数组中大于相邻元素的极大值的索引
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    
    # 创建一个二维 NumPy 数组 y，包含整数元素
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    
    # 使用 argrelextrema 函数找到 y 数组中按行（axis=1）小于相邻元素的极小值的索引
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))
    
    # 调用 _boolrelextrema 函数计算数据的布尔极值
    results = _boolrelextrema(data, comparator,
                              axis, order, mode)
    
    # 返回 results 中非零元素的索引，即布尔极值的位置索引
    return np.nonzero(results)
def _arg_x_as_expected(value):
    """Ensure argument `x` is a 1-D C-contiguous array of dtype('float64').

    Used in `find_peaks`, `peak_prominences` and `peak_widths` to make `x`
    compatible with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('float64').
    """
    # Convert `value` to a NumPy array with specified properties
    value = np.asarray(value, order='C', dtype=np.float64)
    # Check if `value` is 1-dimensional
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value


def _arg_peaks_as_expected(value):
    """Ensure argument `peaks` is a 1-D C-contiguous array of dtype('intp').

    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible
    with the signature of the wrapped Cython functions.

    Returns
    -------
    value : ndarray
        A 1-D C-contiguous array with dtype('intp').
    """
    # Convert `value` to a NumPy array
    value = np.asarray(value)
    # Handle case where `value` is an empty array
    if value.size == 0:
        # Empty arrays default to np.float64 but are valid input
        value = np.array([], dtype=np.intp)
    try:
        # Safely convert `value` to dtype('intp') and ensure C-contiguity
        value = value.astype(np.intp, order='C', casting='safe',
                             subok=False, copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    # Check if `value` is 1-dimensional
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value


def _arg_wlen_as_expected(value):
    """Ensure argument `wlen` is of type `np.intp` and larger than 1.

    Used in `peak_prominences` and `peak_widths`.

    Returns
    -------
    value : np.intp
        The original `value` rounded up to an integer or -1 if `value` was
        None.
    """
    if value is None:
        # Signal that no value was supplied by the user
        value = -1
    elif 1 < value:
        # Round up `value` to the nearest integer if it's a float
        if isinstance(value, float):
            value = math.ceil(value)
        # Convert `value` to `np.intp`
        value = np.intp(value)
    else:
        raise ValueError(f'`wlen` must be larger than 1, was {value}')
    return value


def peak_prominences(x, peaks, wlen=None):
    """
    Calculate the prominence of each peak in a signal.

    The prominence of a peak measures how much a peak stands out from the
    surrounding baseline of the signal and is defined as the vertical distance
    between the peak and its lowest contour line.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    wlen : int, optional
        A window length in samples that optionally limits the evaluated area for
        each peak to a subset of `x`. The peak is always placed in the middle of
        the window therefore the given length is rounded up to the next odd
        integer. This parameter can speed up the calculation (see Notes).

    Returns
    -------
    ```
    
    # The rest of the function `peak_prominences` is not included as per the instructions.
    # If needed, it should be completed similarly with appropriate comments.
    prominences : ndarray
        每个峰值在 `peaks` 中的突出度。
    left_bases, right_bases : ndarray
        每个峰值在 `x` 中左右两侧的基线索引。每对中较高的基线是峰值的最低等高线。
    Raises
    ------
    ValueError
        如果 `peaks` 中的某个值对于 `x` 是无效的索引。
    Warns
    -----
    PeakPropertyWarning
        对于 `peaks` 中指向 `x` 中无效局部最大值的索引，返回的突出度为 0，并引发此警告。如果 `wlen` 小于峰值的台地大小，也会发生此情况。
    Warnings
    --------
    当数据中包含 NaN 时，此函数可能返回意外的结果。为避免此情况，应删除或替换 NaN。
    See Also
    --------
    find_peaks
        基于峰值属性在信号内查找峰值。
    peak_widths
        计算峰值的宽度。
    Notes
    -----
    计算峰值突出度的策略：
    
    1. 从当前峰值向左右延伸水平线，直到该线到达窗口边界（参见 `wlen`）或再次在更高峰值的斜率处与信号相交。忽略与同高度峰值的交点。
    2. 在每一侧找到上述定义的间隔内的信号最小值。这些点是峰值的基线。
    3. 两个基线中较高的一个标记着峰值的最低等高线。突出度可以计算为峰值本身高度与其最低等高线的垂直差异。
    
    对于具有周期行为的大 `x`，搜索峰值基线可能会很慢，因为需要对信号的大块甚至整个信号进行评估以执行第一算法步骤。此评估区域可以通过参数 `wlen` 进行限制，该参数将算法限制在当前峰值周围的窗口内，如果窗口长度相对于 `x` 较短，则可以缩短计算时间。
    然而，如果峰值的真实基线位于此窗口之外，这可能会阻止算法找到真正的全局等高线。而是在受限制的窗口内找到较高的等高线，导致较小的计算突出度。在实践中，这仅对 `x` 中的最高峰集合是相关的。甚至可以有意使用此行为来计算“局部”突出度。
    
    .. versionadded:: 1.1.0
    
    References
    ----------
    .. [1] Wikipedia Article for Topographic Prominence:
       https://en.wikipedia.org/wiki/Topographic_prominence
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import find_peaks, peak_prominences
    >>> import matplotlib.pyplot as plt
    
    创建一个测试信号，其中包含两个叠加的谐波
    
    >>> x = np.linspace(0, 6 * np.pi, 1000)
    >>> x = np.sin(x) + 0.6 * np.sin(2.6 * x)
    Find all peaks and calculate prominences

    >>> peaks, _ = find_peaks(x)  # 寻找所有峰值的索引
    >>> prominences = peak_prominences(x, peaks)[0]  # 计算每个峰值的显著性
    >>> prominences  # 打印计算得到的峰值显著性数组
    array([1.24159486, 0.47840168, 0.28470524, 3.10716793, 0.284603  ,
           0.47822491, 2.48340261, 0.47822491])

    Calculate the height of each peak's contour line and plot the results

    >>> contour_heights = x[peaks] - prominences  # 计算每个峰值对应的轮廓线高度
    >>> plt.plot(x)  # 绘制原始数据曲线
    >>> plt.plot(peaks, x[peaks], "x")  # 标记峰值点
    >>> plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])  # 绘制峰值点的轮廓线
    >>> plt.show()  # 显示绘制的图形

    Let's evaluate a second example that demonstrates several edge cases for
    one peak at index 5.

    >>> x = np.array([0, 1, 0, 3, 1, 3, 0, 4, 0])  # 定义一个新的数组 x
    >>> peaks = np.array([5])  # 设定峰值索引数组为 [5]
    >>> plt.plot(x)  # 绘制新的数组 x 的曲线
    >>> plt.plot(peaks, x[peaks], "x")  # 标记峰值点
    >>> plt.show()  # 显示绘制的图形
    >>> peak_prominences(x, peaks)  # 计算给定峰值的显著性及其左右基线
    (array([3.]), array([2]), array([6]))

    Note how the peak at index 3 of the same height is not considered as a
    border while searching for the left base. Instead, two minima at 0 and 2
    are found in which case the one closer to the evaluated peak is always
    chosen. On the right side, however, the base must be placed at 6 because the
    higher peak represents the right border to the evaluated area.

    >>> peak_prominences(x, peaks, wlen=3.1)  # 使用指定的窗口长度计算峰值显著性
    (array([2.]), array([4]), array([6]))

    Here, we restricted the algorithm to a window from 3 to 7 (the length is 5
    samples because `wlen` was rounded up to the next odd integer). Thus, the
    only two candidates in the evaluated area are the two neighboring samples
    and a smaller prominence is calculated.
# 定义函数 peak_widths，计算信号中每个峰的宽度及其它相关信息
def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """
    Calculate the width of each peak in a signal.

    This function calculates the width of a peak in samples at a relative
    distance to the peak's height and prominence.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    peaks : sequence
        Indices of peaks in `x`.
    rel_height : float, optional
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence. 1.0 calculates the width of the peak at
        its lowest contour line while 0.5 evaluates at half the prominence
        height. Must be at least 0. See notes for further explanation.
    prominence_data : tuple, optional
        A tuple of three arrays matching the output of `peak_prominences` when
        called with the same arguments `x` and `peaks`. This data are calculated
        internally if not provided.
    wlen : int, optional
        A window length in samples passed to `peak_prominences` as an optional
        argument for internal calculation of `prominence_data`. This argument
        is ignored if `prominence_data` is given.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.

    Raises
    ------
    ValueError
        If `prominence_data` is supplied but doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak,
        has the wrong dtype, is not C-contiguous or does not have the same
        shape.

    Warns
    -----
    PeakPropertyWarning
        Raised if any calculated width is 0. This may stem from the supplied
        `prominence_data` or if `rel_height` is set to 0.

    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.

    See Also
    --------
    find_peaks
        Find peaks inside a signal based on peak properties.
    peak_prominences
        Calculate the prominence of peaks.

    Notes
    -----
    The basic algorithm to calculate a peak's width is as follows:

    * Calculate the evaluation height :math:`h_{eval}` with the formula
      :math:`h_{eval} = h_{Peak} - P \\cdot R`, where :math:`h_{Peak}` is the
      height of the peak itself, :math:`P` is the peak's prominence and
      :math:`R` a positive ratio specified with the argument `rel_height`.
    """
    # 将输入参数 x 转换为预期的格式
    x = _arg_x_as_expected(x)
    # 将输入参数 peaks 转换为预期的格式
    peaks = _arg_peaks_as_expected(peaks)
    # 如果 prominence_data 为 None，则计算突出度并使用给定的 wlen（窗口长度）（如果有的话）
    if prominence_data is None:
        wlen = _arg_wlen_as_expected(wlen)
        # 计算信号 x 中峰值 peaks 的突出度数据
        prominence_data = _peak_prominences(x, peaks, wlen)
    # 返回计算得到的峰宽度数据，使用相对高度 rel_height 和计算好的 prominence_data
    return _peak_widths(x, peaks, rel_height, *prominence_data)
# 解析 `find_peaks` 的条件参数

def _unpack_condition_args(interval, x, peaks):
    """
    Parse condition arguments for `find_peaks`.

    Parameters
    ----------
    interval : number or ndarray or sequence
        Either a number or ndarray or a 2-element sequence of the former. The
        first value is always interpreted as `imin` and the second, if supplied,
        as `imax`.
    x : ndarray
        The signal with `peaks`.
    peaks : ndarray
        An array with indices used to reduce `imin` and / or `imax` if those are
        arrays.

    Returns
    -------
    imin, imax : number or ndarray or None
        Minimal and maximal value in `argument`.

    Raises
    ------
    ValueError :
        If interval border is given as array and its size does not match the size
        of `x`.

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    # 如果 `imin` 是 ndarray，则根据 `peaks` 对其进行筛选
    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    
    # 如果 `imax` 是 ndarray，则根据 `peaks` 对其进行筛选
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]

    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):
    """
    Evaluate where the generic property of peaks confirms to an interval.

    Parameters
    ----------
    peak_properties : ndarray
        An array with properties for each peak.
    pmin : None or number or ndarray
        Lower interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.
    pmax : None or number or ndarray
        Upper interval boundary for `peak_properties`. ``None`` is interpreted as
        an open border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peak_properties` confirms to the
        interval.

    See Also
    --------
    find_peaks

    Notes
    -----

    .. versionadded:: 1.1.0
    """
    # 创建一个布尔掩码，初始值为全部 True
    keep = np.ones(peak_properties.size, dtype=bool)
    
    # 如果 `pmin` 不为 None，则筛选出满足下界条件的峰值
    if pmin is not None:
        keep &= (pmin <= peak_properties)
    
    # 如果 `pmax` 不为 None，则筛选出满足上界条件的峰值
    if pmax is not None:
        keep &= (peak_properties <= pmax)
    
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):
    """
    Evaluate which peaks fulfill the threshold condition.

    Parameters
    ----------
    x : ndarray
        A 1-D array which is indexable by `peaks`.
    peaks : ndarray
        Indices of peaks in `x`.
    tmin, tmax : scalar or ndarray or None
         Minimal and / or maximal required thresholds. If supplied as ndarrays
         their size must match `peaks`. ``None`` is interpreted as an open
         border.

    Returns
    -------
    keep : bool
        A boolean mask evaluating to true where `peaks` fulfill the threshold
        condition.
    """
    # 定义两个数组变量，分别存储左右两侧的阈值，类型为 ndarray
    left_thresholds, right_thresholds : ndarray
        Array matching `peak` containing the thresholds of each peak on
        both sides.
    
    Notes
    -----
    
    .. versionadded:: 1.1.0
    """
    # 将左右两侧的阈值堆叠起来，以便于进行最小值和最大值的比较操作：
    # tmin 与每个峰值两侧的较小阈值进行比较，tmax 与较大阈值进行比较
    stacked_thresholds = np.vstack([x[peaks] - x[peaks - 1],
                                    x[peaks] - x[peaks + 1]])
    # 创建一个布尔数组 keep，其长度为 peaks 的大小，初始值为 True
    keep = np.ones(peaks.size, dtype=bool)
    # 如果 tmin 不为 None，则计算最小的堆叠阈值，并更新 keep 数组
    if tmin is not None:
        min_thresholds = np.min(stacked_thresholds, axis=0)
        keep &= (tmin <= min_thresholds)
    # 如果 tmax 不为 None，则计算最大的堆叠阈值，并更新 keep 数组
    if tmax is not None:
        max_thresholds = np.max(stacked_thresholds, axis=0)
        keep &= (max_thresholds <= tmax)
    
    # 返回布尔数组 keep，以及左右两侧堆叠阈值中的第一个和第二个元素
    return keep, stacked_thresholds[0], stacked_thresholds[1]
# 定义一个函数用于寻找信号中的峰值，基于峰值的属性来确定。
def find_peaks(x, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
    """
    Find peaks inside a signal based on peak properties.

    This function takes a 1-D array and finds all local maxima by
    simple comparison of neighboring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    Parameters
    ----------
    x : sequence
        A signal with peaks.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` for a full
        description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.

        .. versionadded:: 1.2.0

    Returns
    -------
    """
    # peaks : ndarray
    # Indices of peaks in `x` that satisfy all given conditions.
    properties : dict
        # A dictionary containing properties of the returned peaks which were
        # calculated as intermediate results during evaluation of the specified
        # conditions:

        # * 'peak_heights'
        # If `height` is given, the height of each peak in `x`.

        # * 'left_thresholds', 'right_thresholds'
        # If `threshold` is given, these keys contain a peak's vertical
        # distance to its neighbouring samples.

        # * 'prominences', 'right_bases', 'left_bases'
        # If `prominence` is given, these keys are accessible. See
        # `peak_prominences` for a description of their content.

        # * 'width_heights', 'left_ips', 'right_ips'
        # If `width` is given, these keys are accessible. See `peak_widths`
        # for a description of their content.

        # * 'plateau_sizes', left_edges', 'right_edges'
        # If `plateau_size` is given, these keys are accessible and contain
        # the indices of a peak's edges (edges are still part of the
        # plateau) and the calculated plateau sizes.

        # .. versionadded:: 1.2.0

        # To calculate and return properties without excluding peaks, provide the
        # open interval ``(None, None)`` as a value to the appropriate argument
        # (excluding `distance`).

    # Warns
    # -----
    # PeakPropertyWarning
    # Raised if a peak's properties have unexpected values (see
    # `peak_prominences` and `peak_widths`).

    # Warnings
    # --------
    # This function may return unexpected results for data containing NaNs. To
    # avoid this, NaNs should either be removed or replaced.

    # See Also
    # --------
    # find_peaks_cwt
    # Find peaks using the wavelet transformation.
    # peak_prominences
    # Directly calculate the prominence of peaks.
    # peak_widths
    # Directly calculate the width of peaks.

    # Notes
    # -----
    # In the context of this function, a peak or local maximum is defined as any
    # sample whose two direct neighbours have a smaller amplitude. For flat peaks
    # (more than one sample of equal amplitude wide) the index of the middle
    # sample is returned (rounded down in case the number of samples is even).
    # For noisy signals the peak locations can be off because the noise might
    # change the position of local maxima. In those cases consider smoothing the
    # signal before searching for peaks or use other peak finding and fitting
    # methods (like `find_peaks_cwt`).

    # Some additional comments on specifying conditions:
    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima
      (see `peak_prominences`).

    .. versionadded:: 1.1.0

    Examples
    --------
    To demonstrate this function's usage we use a signal `x` supplied with
    SciPy (see `scipy.datasets.electrocardiogram`). Let's find all peaks (local
    maxima) in `x` whose amplitude lies above 0.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.datasets import electrocardiogram
    >>> from scipy.signal import find_peaks
    >>> x = electrocardiogram()[2000:4000]
    >>> peaks, _ = find_peaks(x, height=0)
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.plot(np.zeros_like(x), "--", color="gray")
    >>> plt.show()

    We can select peaks below 0 with ``height=(None, 0)`` or use arrays matching
    `x` in size to reflect a changing condition for different parts of the
    signal.

    >>> border = np.sin(np.linspace(0, 3 * np.pi, x.size))
    >>> peaks, _ = find_peaks(x, height=(-border, border))
    >>> plt.plot(x)
    >>> plt.plot(-border, "--", color="gray")
    >>> plt.plot(border, ":", color="gray")
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Another useful condition for periodic signals can be given with the
    `distance` argument. In this case, we can easily select the positions of
    QRS complexes within the electrocardiogram (ECG) by demanding a distance of
    at least 150 samples.

    >>> peaks, _ = find_peaks(x, distance=150)
    >>> np.diff(peaks)
    array([186, 180, 177, 171, 177, 169, 167, 164, 158, 162, 172])
    >>> plt.plot(x)
    >>> plt.plot(peaks, x[peaks], "x")
    >>> plt.show()

    Especially for noisy signals peaks can be easily grouped by their
    """
        prominence (see `peak_prominences`). E.g., we can select all peaks except
        for the mentioned QRS complexes by limiting the allowed prominence to 0.6.
    
        >>> peaks, properties = find_peaks(x, prominence=(None, 0.6))
        >>> properties["prominences"].max()
        0.5049999999999999
        >>> plt.plot(x)
        >>> plt.plot(peaks, x[peaks], "x")
        >>> plt.show()
    
        And, finally, let's examine a different section of the ECG which contains
        beat forms of different shape. To select only the atypical heart beats, we
        combine two conditions: a minimal prominence of 1 and width of at least 20
        samples.
    
        >>> x = electrocardiogram()[17000:18000]
        >>> peaks, properties = find_peaks(x, prominence=1, width=20)
        >>> properties["prominences"], properties["widths"]
        (array([1.495, 2.3  ]), array([36.93773946, 39.32723577]))
        >>> plt.plot(x)
        >>> plt.plot(peaks, x[peaks], "x")
        >>> plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
        ...            ymax = x[peaks], color = "C1")
        >>> plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
        ...            xmax=properties["right_ips"], color = "C1")
        >>> plt.show()
        """
        # _argmaxima1d expects array of dtype 'float64'
        x = _arg_x_as_expected(x)  # 调用函数 _arg_x_as_expected 处理 x，确保其为 'float64' 类型的数组
    
        if distance is not None and distance < 1:
            raise ValueError('`distance` must be greater or equal to 1')  # 如果 distance 不为 None 且小于 1，则引发 ValueError
    
        peaks, left_edges, right_edges = _local_maxima_1d(x)  # 调用函数 _local_maxima_1d 处理 x，返回峰值、左边缘和右边缘
    
        properties = {}  # 初始化空字典用于存储峰值特性
    
        if plateau_size is not None:
            # Evaluate plateau size
            plateau_sizes = right_edges - left_edges + 1  # 计算高原的大小
            pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)  # 解包 plateau_size 条件参数
            keep = _select_by_property(plateau_sizes, pmin, pmax)  # 根据条件选择保留的峰值
            peaks = peaks[keep]  # 更新 peaks 数组
            properties["plateau_sizes"] = plateau_sizes  # 存储高原大小数组
            properties["left_edges"] = left_edges  # 存储左边缘数组
            properties["right_edges"] = right_edges  # 存储右边缘数组
            properties = {key: array[keep] for key, array in properties.items()}  # 更新 properties 字典
    
        if height is not None:
            # Evaluate height condition
            peak_heights = x[peaks]  # 获取峰值的高度
            hmin, hmax = _unpack_condition_args(height, x, peaks)  # 解包 height 条件参数
            keep = _select_by_property(peak_heights, hmin, hmax)  # 根据条件选择保留的峰值
            peaks = peaks[keep]  # 更新 peaks 数组
            properties["peak_heights"] = peak_heights  # 存储峰值高度数组
            properties = {key: array[keep] for key, array in properties.items()}  # 更新 properties 字典
    
        if threshold is not None:
            # Evaluate threshold condition
            tmin, tmax = _unpack_condition_args(threshold, x, peaks)  # 解包 threshold 条件参数
            keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
                x, peaks, tmin, tmax)  # 根据条件选择保留的峰值及其左右阈值
            peaks = peaks[keep]  # 更新 peaks 数组
            properties["left_thresholds"] = left_thresholds  # 存储左阈值数组
            properties["right_thresholds"] = right_thresholds  # 存储右阈值数组
            properties = {key: array[keep] for key, array in properties.items()}  # 更新 properties 字典
    if distance is not None:
        # 如果距离条件不为空，则执行以下操作
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        # 根据峰值间距条件筛选保留的峰值索引
        peaks = peaks[keep]
        # 更新属性字典，只保留符合条件的属性值
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:
        # 如果突出度或宽度条件不为空，则执行以下操作
        wlen = _arg_wlen_as_expected(wlen)
        # 根据期望的长度调整参数 wlen
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences(x, peaks, wlen=wlen)
        ))
        # 更新属性字典，计算并存储突出度、左基线和右基线

    if prominence is not None:
        # 如果突出度条件不为空，则执行以下操作
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        # 解包突出度条件参数
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        # 根据突出度条件筛选保留的峰值索引
        peaks = peaks[keep]
        # 更新属性字典，只保留符合条件的属性值
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:
        # 如果宽度条件不为空，则执行以下操作
        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths(x, peaks, rel_height, properties['prominences'],
                         properties['left_bases'], properties['right_bases'])
        ))
        # 更新属性字典，计算并存储宽度、宽度高度、左侧半峰宽度和右侧半峰宽度
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        # 解包宽度条件参数
        keep = _select_by_property(properties['widths'], wmin, wmax)
        # 根据宽度条件筛选保留的峰值索引
        peaks = peaks[keep]
        # 更新属性字典，只保留符合条件的属性值

    return peaks, properties
    # 返回筛选后的峰值索引和对应的属性字典
# 定义一个函数来识别二维矩阵中的脊线（ridge lines）
def _identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2-D matrix.

    Expect that the width of the wavelet feature increases with increasing row
    number.

    Parameters
    ----------
    matr : 2-D ndarray
        Matrix in which to identify ridge lines.
    max_distances : 1-D sequence
        At each row, a ridge line is only connected
        if the relative max at row[n] is within
        `max_distances`[n] from the relative max at row[n+1].
    gap_thresh : int
        If a relative maximum is not found within `max_distances`,
        there will be a gap. A ridge line is discontinued if
        there are more than `gap_thresh` points without connecting
        a new relative maximum.

    Returns
    -------
    ridge_lines : tuple
        Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the
        ii-th ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none
        found.  Each ridge-line will be sorted by row (increasing), but the
        order of the ridge lines is not specified.

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._peak_finding import _identify_ridge_lines
    >>> rng = np.random.default_rng()
    >>> data = rng.random((5,5))
    >>> max_dist = 3
    >>> max_distances = np.full(20, max_dist)
    >>> ridge_lines = _identify_ridge_lines(data, max_distances, 1)

    Notes
    -----
    This function is intended to be used in conjunction with `cwt`
    as part of `find_peaks_cwt`.

    """
    # 检查 max_distances 数组的长度是否大于或等于 matr 矩阵的行数，若不是则抛出 ValueError
    if len(max_distances) < matr.shape[0]:
        raise ValueError('Max_distances must have at least as many rows '
                         'as matr')

    # 利用 _boolrelextrema 函数找出所有相对极大值所在的列
    all_max_cols = _boolrelextrema(matr, np.greater, axis=1, order=1)
    # 找出具有相对极大值的最高行
    has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
    # 如果没有找到任何具有相对极大值的行，则返回空列表
    if len(has_relmax) == 0:
        return []
    start_row = has_relmax[-1]
    # 初始化脊线列表，每个脊线由一个三元组组成：行、列、缺口数
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.nonzero(all_max_cols[start_row])[0]]
    # 最终的脊线列表
    final_lines = []
    # 创建行和列的范围数组
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    # 遍历每一行数据
    for row in rows:
        # 获取当前行的最大列数
        this_max_cols = cols[all_max_cols[row]]

        # 增加每条脊线的间隔数，
        # 如果适当的话稍后会将其设置为零
        for line in ridge_lines:
            line[2] += 1

        # XXX 这些应该始终是 all_max_cols[row]
        # 但顺序可能不同。确保顺序相同并避免此迭代
        # 取前一次脊线的最后一个列作为数组
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        
        # 查看当前行找到的每个相对最大值
        # 尝试将它们与现有的脊线连接起来
        for ind, col in enumerate(this_max_cols):
            # 如果在最大距离内有前一个脊线可连接，则连接它
            # 否则开始一个新的脊线
            line = None
            if len(prev_ridge_cols) > 0:
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if line is not None:
                # 找到足够接近的点，扩展当前脊线
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                # 创建新的脊线
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        # 移除间隔数过高的脊线
        # XXX 在迭代过程中修改列表。虽然是逆序迭代，但仍有些不雅
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    # 准备输出的行数据
    out_lines = []
    for line in (final_lines + ridge_lines):
        # 按行的顺序进行排序并准备输出格式
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    # 返回输出的行数据
    return out_lines
# 根据指定条件过滤掉不符合要求的岭线。用于寻找相对极大值。

def _filter_ridge_lines(cwt, ridge_lines, window_size=None, min_length=None,
                        min_snr=1, noise_perc=10):
    """
    Filter ridge lines according to prescribed criteria. Intended
    to be used for finding relative maxima.

    Parameters
    ----------
    cwt : 2-D ndarray
        Continuous wavelet transform from which the `ridge_lines` were defined.
    ridge_lines : 1-D sequence
        Each element should contain 2 sequences, the rows and columns
        of the ridge line (respectively).
    window_size : int, optional
        Size of window to use to calculate noise floor.
        Default is ``cwt.shape[1] / 20``.
    min_length : int, optional
        Minimum length a ridge line needs to be acceptable.
        Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
    min_snr : float, optional
        Minimum SNR ratio. Default 1. The signal is the value of
        the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
        noise is the `noise_perc`\\ th percentile of datapoints contained within a
        window of `window_size` around ``cwt[0, loc]``.
    noise_perc : float, optional
        When calculating the noise floor, percentile of data points
        examined below which to consider noise. Calculated using
        scipy.stats.scoreatpercentile.

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`

    """
    num_points = cwt.shape[1]  # 获取 cwt 矩阵的列数（数据点数量）
    if min_length is None:
        min_length = np.ceil(cwt.shape[0] / 4)  # 如果未指定最小长度，设定为 cwt 高度的 1/4
    if window_size is None:
        window_size = np.ceil(num_points / 20)  # 如果未指定窗口大小，设定为数据点数量的 1/20

    window_size = int(window_size)  # 窗口大小转换为整数
    hf_window, odd = divmod(window_size, 2)  # 计算窗口的一半大小和是否为奇数

    # 根据 SNR 进行滤波
    row_one = cwt[0, :]  # 提取 cwt 矩阵的第一行
    noises = np.empty_like(row_one)  # 创建一个与 row_one 相同大小的空数组
    for ind, val in enumerate(row_one):
        window_start = max(ind - hf_window, 0)  # 计算窗口起始位置
        window_end = min(ind + hf_window + odd, num_points)  # 计算窗口结束位置
        noises[ind] = scoreatpercentile(row_one[window_start:window_end],
                                        per=noise_perc)  # 计算窗口内数据的噪声水平的百分位数

    def filt_func(line):
        if len(line[0]) < min_length:  # 如果岭线长度小于最小长度，返回 False
            return False
        snr = abs(cwt[line[0][0], line[1][0]] / noises[line[1][0]])  # 计算 SNR
        if snr < min_snr:  # 如果 SNR 小于最小要求，返回 False
            return False
        return True  # 符合条件，返回 True

    return list(filter(filt_func, ridge_lines))  # 返回符合条件的岭线列表


def find_peaks_cwt(vector, widths, wavelet=None, max_distances=None,
                   gap_thresh=None, min_length=None,
                   min_snr=1, noise_perc=10, window_size=None):
    """
    Find peaks in a 1-D array with wavelet transformation.

    The general approach is to smooth `vector` by convolving it with
    `wavelet(width)` for each width in `widths`. Relative maxima which
    appear at enough length scales, and with sufficiently high SNR, are
    accepted.

    Parameters
    ----------
    vector : ndarray
        1-D array in which to find the peaks.
    widths : float or sequence
        # 宽度参数，可以是单个浮点数或者包含多个浮点数的数组，用于计算连续小波变换（CWT）矩阵。
        # 通常应涵盖感兴趣峰的预期宽度范围。

    wavelet : callable, optional
        # 小波函数，应接受两个参数并返回一个一维数组，用于与 `vector` 进行卷积。
        # 第一个参数确定返回小波数组的点数，第二个参数是小波的尺度（宽度）。
        # 应该是归一化和对称的。默认为 ricker 小波函数。

    max_distances : ndarray, optional
        # 每一行处，仅当相对最大值在 `row[n]` 处与 `row[n+1]` 处的相对最大值之间距离
        # 在 `max_distances[n]` 范围内时，形成的脊线才连接。
        # 默认值为 `widths/4`。

    gap_thresh : float, optional
        # 如果在 `max_distances` 范围内找不到相对最大值，则存在间隙。
        # 如果有超过 `gap_thresh` 点没有连接到新的相对最大值，则脊线中断。
        # 默认值为 widths 数组的第一个值，即 widths[0]。

    min_length : int, optional
        # 脊线的最小长度要求。默认为 `cwt.shape[0] / 4`，即宽度数组长度的四分之一。

    min_snr : float, optional
        # 最小信噪比。默认为 1。信号是最大脊线上的最大 CWT 系数。
        # 噪声是在同一脊线中包含的数据点的 `noise_perc` 百分位数。
    
    noise_perc : float, optional
        # 在计算噪声水平时，数据点的百分位数，低于这个百分位数的数据点被认为是噪声。
        # 使用 `stats.scoreatpercentile` 计算。默认为 10。

    window_size : int, optional
        # 用于计算噪声水平的窗口大小。默认为 `cwt.shape[1] / 20`。

    Returns
    -------
    peaks_indices : ndarray
        # 找到的峰值在 `vector` 中的索引位置。
        # 返回的列表是按顺序排序的。

    See Also
    --------
    find_peaks
        # 基于峰值属性在信号中查找峰值。

    Notes
    -----
    # 此方法设计用于在嘈杂数据中找到尖锐的峰值，但通过适当的参数选择，应该也能很好地适用于不同的峰形状。

    The algorithm is as follows:
     1. Perform a continuous wavelet transform on `vector`, for the supplied
        `widths`. This is a convolution of `vector` with `wavelet(width)` for
        each width in `widths`. See `cwt`.
        # 算法如下：
        # 1. 对 `vector` 执行连续小波变换，使用提供的 `widths`。
        # 这是 `vector` 与 `wavelet(width)` 的卷积，对于 `widths` 中的每个宽度。详见 `cwt`。

     2. Identify "ridge lines" in the cwt matrix. These are relative maxima
        at each row, connected across adjacent rows. See identify_ridge_lines
        # 在 CWT 矩阵中识别 "脊线"。这些是每一行处的相对最大值，在相邻行之间连接起来。详见 identify_ridge_lines 函数。

     3. Filter the ridge_lines using filter_ridge_lines.
        # 使用 filter_ridge_lines 过滤脊线。

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
       :doi:`10.1093/bioinformatics/btl355`
    # 导入信号处理模块中的 find_peaks_cwt 函数
    >>> from scipy import signal
    # 创建一个数组 xs，包含从 0 到 π 的数值，步长为 0.05
    >>> xs = np.arange(0, np.pi, 0.05)
    # 计算 xs 中每个值的正弦值，存储在 data 数组中
    >>> data = np.sin(xs)
    # 使用 find_peaks_cwt 函数查找 data 中的峰值，使用给定的宽度范围
    >>> peakind = signal.find_peaks_cwt(data, np.arange(1,10))
    # 输出找到的峰值的索引、xs 中对应的值和 data 中对应的值
    >>> peakind, xs[peakind], data[peakind]

    """
    # 将 widths 转换为至少是一维数组的形式
    widths = np.atleast_1d(np.asarray(widths))

    # 如果未指定 gap_thresh，则将其设为 widths 中第一个元素的上取整值
    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    # 如果未指定 max_distances，则将其设为 widths 的四分之一
    if max_distances is None:
        max_distances = widths / 4.0
    # 如果未指定 wavelet，则使用默认的 _ricker 波形函数
    if wavelet is None:
        wavelet = _ricker

    # 对输入向量 vector 进行连续小波变换，得到 cwt_dat
    cwt_dat = _cwt(vector, wavelet, widths)
    # 识别 cwt_dat 中的脊线（ridge lines），使用指定的 max_distances 和 gap_thresh
    ridge_lines = _identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
    # 过滤掉不符合条件的脊线，得到符合条件的 ridge_lines
    filtered = _filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                   window_size=window_size, min_snr=min_snr,
                                   noise_perc=noise_perc)
    # 提取所有最大位置的索引，排序后转换为数组 max_locs
    max_locs = np.asarray([x[1][0] for x in filtered])
    max_locs.sort()

    # 返回排序后的最大位置数组 max_locs
    return max_locs
```