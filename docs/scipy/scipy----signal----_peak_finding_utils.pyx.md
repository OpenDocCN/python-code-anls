# `D:\src\scipysrc\scipy\scipy\signal\_peak_finding_utils.pyx`

```
#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
"""
Utility functions for finding peaks in signals.
"""

import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库

cimport numpy as np  # 引入 NumPy 库的 C 接口
from libc.math cimport ceil  # 引入 C 标准库中的 ceil 函数

np.import_array()  # 调用 NumPy 的 import_array 函数

__all__ = ['_local_maxima_1d', '_select_by_peak_distance', '_peak_prominences',
           '_peak_widths']  # 定义模块的公共接口列表

def _local_maxima_1d(const np.float64_t[::1] x not None):
    """
    Find local maxima in a 1D array.

    This function finds all local maxima in a 1D array and returns the indices
    for their edges and midpoints (rounded down for even plateau sizes).

    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.

    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    left_edges : ndarray
        Indices of edges to the left of local maxima in `x`.
    right_edges : ndarray
        Indices of edges to the right of local maxima in `x`.

    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    - A maxima is defined as one or more samples of equal value that are
      surrounded on both sides by at least one smaller sample.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.intp_t[::1] midpoints, left_edges, right_edges  # 定义三个数组，存储中点和边缘索引
        np.intp_t m, i, i_ahead, i_max  # 定义几个整型变量

    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)  # 预分配存储中点索引的数组
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)  # 预分配存储左边缘索引的数组
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)  # 预分配存储右边缘索引的数组
    m = 0  # 有效数据区域末尾的指针

    with nogil:
        i = 1  # 当前样本的指针，第一个不能是最大值
        i_max = x.shape[0] - 1  # 最后一个样本不能是最大值
        while i < i_max:
            # 测试前一个样本是否更小
            if x[i - 1] < x[i]:
                i_ahead = i + 1  # 查找当前样本之后的索引

                # 查找下一个不等于 x[i] 的样本
                while i_ahead < i_max and x[i_ahead] == x[i]:
                    i_ahead += 1

                # 如果下一个不等样本小于 x[i]，则找到最大值
                if x[i_ahead] < x[i]:
                    left_edges[m] = i
                    right_edges[m] = i_ahead - 1
                    midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                    m += 1
                    # 跳过不能是最大值的样本
                    i = i_ahead
            i += 1

    # 仅保留数组的有效内存部分
    midpoints.base.resize(m, refcheck=False)
    left_edges.base.resize(m, refcheck=False)
    right_edges.base.resize(m, refcheck=False)

    return midpoints.base, left_edges.base, right_edges.base
# 定义一个函数，根据峰值之间的最小距离，评估哪些峰值满足条件
def _select_by_peak_distance(const np.intp_t[::1] peaks not None,
                             const np.float64_t[::1] priority not None,
                             np.float64_t distance):
    """
    Evaluate which peaks fulfill the distance condition.

    Parameters
    ----------
    peaks : ndarray
        Indices of peaks in `vector`.
    priority : ndarray
        An array matching `peaks` used to determine priority of each peak. A
        peak with a higher priority value is kept over one with a lower one.
    distance : np.float64
        Minimal distance that peaks must be spaced.

    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.

    Notes
    -----
    Declaring the input arrays as C-contiguous doesn't seem to have performance
    advantages.

    .. versionadded:: 1.1.0
    """
    cdef:
        np.uint8_t[::1] keep  # 用于标记保留哪些峰值的布尔数组
        np.intp_t[::1] priority_to_position  # 根据优先级排序的峰值位置映射数组
        np.intp_t i, j, k, peaks_size, distance_

    peaks_size = peaks.shape[0]  # 峰值数组的长度
    # 将距离向上取整，因为实际的峰值距离只能是自然数
    distance_ = <np.intp_t>ceil(distance)
    keep = np.ones(peaks_size, dtype=np.uint8)  # 准备一个标志位数组，初始全为1表示保留

    # 创建从按`priority`排序的 `peaks` 索引 `i` 到按位置排序的 `peaks` 索引 `j` 的映射。
    # 这允许按 `priority` 顺序迭代 `peaks` 和 `keep`，同时仍然能够步进到相邻的峰值 (`j` + 1 或 `j` - 1)。
    priority_to_position = np.argsort(priority)

    with nogil:
        # 优先级最高的峰值首先 -> 反向迭代顺序（递减）
        for i in range(peaks_size - 1, -1, -1):
            # 将 `i` 转换为 `j`，它指向当前要评估邻近峰值的峰值
            j = priority_to_position[i]
            if keep[j] == 0:
                # 跳过已标记为“不保留”的峰值的评估
                continue

            k = j - 1
            # 标记“较早”的峰值为删除，直到超过最小距离
            while 0 <= k and peaks[j] - peaks[k] < distance_:
                keep[k] = 0
                k -= 1

            k = j + 1
            # 标记“稍后”峰值为删除，直到超过最小距离
            while k < peaks_size and peaks[k] - peaks[j] < distance_:
                keep[k] = 0
                k += 1

    return keep.base.view(dtype=np.bool_)  # 以布尔数组的形式返回
    """
        peaks : ndarray
            `x` 中峰值的索引数组。
        wlen : np.intp
            以样本为单位的窗口长度（参见 `peak_prominences`），将其四舍五入到最近的奇整数。如果小于2，则使用整个信号 `x`。
    
        Returns
        -------
        prominences : ndarray
            每个峰值在 `peaks` 中的突出度。
        left_bases, right_bases : ndarray
            每个峰值在 `x` 中左右基线的索引。
    
        Raises
        ------
        ValueError
            如果 `peaks` 中的某个索引对于 `x` 是无效的。
    
        Warns
        -----
        PeakPropertyWarning
            如果任何峰值的突出度为0。
    
        Notes
        -----
        这是 `peak_prominences` 的内部函数。
    
        .. versionadded:: 1.1.0
        """
        cdef:
            np.float64_t[::1] prominences         # 声明峰值突出度数组
            np.intp_t[::1] left_bases, right_bases  # 声明左右基线索引数组
            np.float64_t left_min, right_min     # 声明左右最小值
            np.intp_t peak_nr, peak, i_min, i_max, i  # 声明循环使用的变量
            np.uint8_t show_warning              # 是否显示警告的标志
    
        show_warning = False  # 初始化不显示警告
        prominences = np.empty(peaks.shape[0], dtype=np.float64)  # 初始化峰值突出度数组
        left_bases = np.empty(peaks.shape[0], dtype=np.intp)      # 初始化左基线索引数组
        right_bases = np.empty(peaks.shape[0], dtype=np.intp)     # 初始化右基线索引数组
    
        with nogil:  # 进入无全局解释器锁定状态
            for peak_nr in range(peaks.shape[0]):  # 遍历所有峰值
                peak = peaks[peak_nr]             # 获取当前峰值的索引
                i_min = 0                         # 最小索引
                i_max = x.shape[0] - 1            # 最大索引
                if not i_min <= peak <= i_max:    # 如果峰值索引超出了信号 `x` 的范围
                    with gil:                      # 重新获取全局解释器锁定状态
                        raise ValueError("peak {} is not a valid index for `x`"
                                         .format(peak))  # 抛出值错误
    
                if 2 <= wlen:  # 如果窗口长度大于等于2
                    # 调整评估峰值周围的窗口（在边界内）；如果 wlen 是偶数，则隐式地将结果窗口长度四舍五入到下一个奇数
                    i_min = max(peak - wlen // 2, i_min)  # 调整左边界
                    i_max = min(peak + wlen // 2, i_max)  # 调整右边界
    
                # 在区间 [i_min, peak] 中找到左基线
                i = left_bases[peak_nr] = peak  # 初始化左基线索引
                left_min = x[peak]              # 左最小值初始化为当前峰值
                while i_min <= i and x[i] <= x[peak]:  # 向左搜索基线直到找到更低的值
                    if x[i] < left_min:
                        left_min = x[i]
                        left_bases[peak_nr] = i
                    i -= 1
    
                # 在区间 [peak, i_max] 中找到右基线
                i = right_bases[peak_nr] = peak  # 初始化右基线索引
                right_min = x[peak]              # 右最小值初始化为当前峰值
                while i <= i_max and x[i] <= x[peak]:  # 向右搜索基线直到找到更低的值
                    if x[i] < right_min:
                        right_min = x[i]
                        right_bases[peak_nr] = i
                    i += 1
    
                # 计算峰值的突出度
                prominences[peak_nr] = x[peak] - max(left_min, right_min)
                if prominences[peak_nr] == 0:  # 如果突出度为0
                    show_warning = True         # 设置警告标志为真
    
        if show_warning:
            warnings.warn("some peaks have a prominence of 0",
                          PeakPropertyWarning, stacklevel=2)  # 发出突出度为0的警告
        # 返回内存视图作为ndarrays
        return prominences.base, left_bases.base, right_bases.base  # 返回峰值突出度、左基线和右基线数组的基础内存视图
# 定义内部函数 `_peak_widths`，用于计算信号中每个峰的宽度及相关信息
def _peak_widths(const np.float64_t[::1] x not None,
                 const np.intp_t[::1] peaks not None,
                 np.float64_t rel_height,
                 const np.float64_t[::1] prominences not None,
                 const np.intp_t[::1] left_bases not None,
                 const np.intp_t[::1] right_bases not None):
    """
    Calculate the width of each each peak in a signal.

    Parameters
    ----------
    x : ndarray
        A signal with peaks.
    peaks : ndarray
        Indices of peaks in `x`.
    rel_height : np.float64
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence (see `peak_widths`).
    prominences : ndarray
        Prominences of each peak in `peaks` as returned by `peak_prominences`.
    left_bases, right_bases : ndarray
        Left and right bases of each peak in `peaks` as returned by
        `peak_prominences`.

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
        If the supplied prominence data doesn't satisfy the condition
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak or
        if `peaks`, `left_bases` and `right_bases` don't share the same shape.
        Or if `rel_height` is not at least 0.

    Warnings
    --------
    PeakPropertyWarning
        If a width of 0 was calculated for any peak.

    Notes
    -----
    This is the inner function to `peak_widths`.

    .. versionadded:: 1.1.0
    """
    # 声明变量
    cdef:
        np.float64_t[::1] widths, width_heights, left_ips, right_ips
        np.float64_t height, left_ip, right_ip
        np.intp_t p, peak, i, i_max, i_min
        np.uint8_t show_warning
    
    # 检查 `rel_height` 参数是否合法
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    
    # 检查传入的数组是否具有相同形状
    if not (peaks.shape[0] == prominences.shape[0] == left_bases.shape[0]
            == right_bases.shape[0]):
        raise ValueError("arrays in `prominence_data` must have the same shape "
                         "as `peaks`")
    
    # 初始化存储结果的数组
    widths = np.empty(peaks.shape[0], dtype=np.float64)
    width_heights = np.empty(peaks.shape[0], dtype=np.float64)
    left_ips = np.empty(peaks.shape[0], dtype=np.float64)
    right_ips = np.empty(peaks.shape[0], dtype=np.float64)
    # 使用 nogil 上下文以释放全局解释器锁 (GIL)，提高循环效率
    with nogil:
        # 遍历所有峰的索引
        for p in range(peaks.shape[0]):
            # 获取当前峰的左右基线索引和峰值索引
            i_min = left_bases[p]
            i_max = right_bases[p]
            peak = peaks[p]
            
            # 验证索引的有效性和顺序
            if not 0 <= i_min <= peak <= i_max < x.shape[0]:
                # 如果索引无效，则抛出值错误异常
                with gil:
                    raise ValueError("prominence data is invalid for peak {}"
                                     .format(peak))
            
            # 计算当前峰的高度
            height = width_heights[p] = x[peak] - prominences[p] * rel_height

            # 寻找左侧交点
            i = peak
            while i_min < i and height < x[i]:
                i -= 1
            left_ip = <np.float64_t>i
            if x[i] < height:
                # 如果真实交点高度在样本之间，则插值计算交点位置
                left_ip += (height - x[i]) / (x[i + 1] - x[i])

            # 寻找右侧交点
            i = peak
            while i < i_max and height < x[i]:
                i += 1
            right_ip = <np.float64_t>i
            if  x[i] < height:
                # 如果真实交点高度在样本之间，则插值计算交点位置
                right_ip -= (height - x[i]) / (x[i - 1] - x[i])

            # 计算峰宽度
            widths[p] = right_ip - left_ip
            if widths[p] == 0:
                # 如果宽度为0，则设置警告标志
                show_warning = True
            # 存储左右交点位置
            left_ips[p] = left_ip
            right_ips[p] = right_ip

    # 如果存在宽度为0的峰，则发出警告
    if show_warning:
        warnings.warn("some peaks have a width of 0",
                      PeakPropertyWarning, stacklevel=2)
    
    # 返回计算结果的基础内存视图
    return widths.base, width_heights.base, left_ips.base, right_ips.base
```