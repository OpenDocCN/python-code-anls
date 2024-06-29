# `D:\src\scipysrc\pandas\pandas\_libs\window\indexers.pyx`

```
# cython: boundscheck=False, wraparound=False, cdivision=True
# 导入NumPy库
import numpy as np

# 从Cython中导入特定的数据类型和数组类型
from numpy cimport (
    int64_t,
    ndarray,
)

# Cython窗口索引器的例程

# 定义一个函数，计算可变窗口的边界
def calculate_variable_window_bounds(
    int64_t num_values,
    int64_t window_size,
    object min_periods,  # 未使用，但为了匹配get_window_bounds的签名而存在
    bint center,
    str closed,
    const int64_t[:] index
):
    """
    根据时间偏移计算滚动窗口的边界。

    参数
    ----------
    num_values : int64
        总值的数量

    window_size : int64
        从偏移计算得到的窗口大小

    min_periods : object
        忽略，存在于兼容性考虑

    center : bint
        是否将滚动窗口置于当前观察值的中心

    closed : str
        窗口的闭合方向字符串

    index : ndarray[int64]
        用于滚动的时间序列索引

    返回
    -------
    (ndarray[int64], ndarray[int64])
    """
    cdef:
        bint left_closed = False  # 左边界是否闭合，默认为False
        bint right_closed = False  # 右边界是否闭合，默认为False
        ndarray[int64_t, ndim=1] start, end  # 起始和结束边界数组
        int64_t start_bound, end_bound, index_growth_sign = 1  # 起始边界，结束边界，索引增长符号
        Py_ssize_t i, j  # Python的大小类型i和j

    if num_values <= 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    # 默认为 'right'
    if closed is None:
        closed = "right"

    # 如果closed为'right'或'both'，则右边界闭合
    if closed in ["right", "both"]:
        right_closed = True

    # 如果closed为'left'或'both'，则左边界闭合
    if closed in ["left", "both"]:
        left_closed = True

    # GH 43997:
    # 如果前向和后向窗口会导致1/2纳秒的分数，需要同时将区间端点设为包含。
    if center and window_size % 2 == 1:
        right_closed = True
        left_closed = True

    # 如果最后一个索引小于第一个索引，则索引增长符号为-1
    if index[num_values - 1] < index[0]:
        index_growth_sign = -1

    # 创建长度为num_values的起始和结束边界数组，填充为-1
    start = np.empty(num_values, dtype="int64")
    start.fill(-1)
    end = np.empty(num_values, dtype="int64")
    end.fill(-1)

    # 第一个窗口的起始为0
    start[0] = 0

    # 右端点为闭合
    if right_closed:
        end[0] = 1
    # 右端点为开放
    else:
        end[0] = 0

    # 如果center为True，计算结束边界
    if center:
        end_bound = index[0] + index_growth_sign * window_size / 2
        for j in range(0, num_values):
            if (index[j] - end_bound) * index_growth_sign < 0:
                end[0] = j + 1
            elif (index[j] - end_bound) * index_growth_sign == 0 and right_closed:
                end[0] = j + 1
            elif (index[j] - end_bound) * index_growth_sign >= 0:
                end[0] = j
                break
    with nogil:
        # 使用 nogil 上下文，表示此处的代码块将不会被GIL（全局解释器锁）限制

        # start 是切片区间的起始点（包含）
        # end 是切片区间的结束点（不包含）
        for i in range(1, num_values):
            if center:
                # 如果是中心窗口，则计算结束边界和开始边界
                end_bound = index[i] + index_growth_sign * window_size / 2
                start_bound = index[i] - index_growth_sign * window_size / 2
            else:
                # 如果不是中心窗口，则直接使用当前索引作为结束边界，计算开始边界
                end_bound = index[i]
                start_bound = index[i] - index_growth_sign * window_size

            # 左端点是闭合的
            if left_closed:
                start_bound -= 1 * index_growth_sign

            # 推进开始边界，直到满足约束条件
            start[i] = i
            for j in range(start[i - 1], i):
                if (index[j] - start_bound) * index_growth_sign > 0:
                    start[i] = j
                    break

            # 对于中心窗口，推进结束边界，直到超出约束条件
            if center:
                for j in range(end[i - 1], num_values + 1):
                    if j == num_values:
                        end[i] = j
                    elif ((index[j] - end_bound) * index_growth_sign == 0 and
                          right_closed):
                        end[i] = j + 1
                    elif (index[j] - end_bound) * index_growth_sign >= 0:
                        end[i] = j
                        break
            # 结束边界是前一个结束点或当前索引
            elif index[end[i - 1]] == end_bound and not right_closed:
                end[i] = end[i - 1] + 1
            elif (index[end[i - 1]] - end_bound) * index_growth_sign <= 0:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # 右端点是开放的
            if not right_closed and not center:
                end[i] -= 1
    # 返回开始点数组和结束点数组
    return start, end
```