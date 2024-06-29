# `D:\src\scipysrc\pandas\pandas\_libs\window\aggregations.pyx`

```
# cython: boundscheck=False, wraparound=False, cdivision=True
# 设置 Cython 编译器指令，禁用边界检查、循环包装和启用 C 除法优化

from libc.math cimport (
    round,
    signbit,
    sqrt,
)
# 从 C 标准库中导入数学函数：round, signbit, sqrt

from libcpp.deque cimport deque
# 从 C++ 标准库中导入 deque 容器

from pandas._libs.algos cimport TiebreakEnumType
# 从 pandas 内部算法模块导入 TiebreakEnumType

import numpy as np
# 导入 NumPy 库，并使用 np 作为别名

cimport numpy as cnp
from numpy cimport (
    float32_t,
    float64_t,
    int64_t,
    ndarray,
)
# 从 NumPy 中导入特定的 C 类型和函数，如 float32_t, float64_t, int64_t, ndarray

cnp.import_array()
# 导入 NumPy 的 C API 数组功能

import cython
# 导入 Cython 模块

from pandas._libs.algos import is_monotonic
# 从 pandas 内部算法模块导入 is_monotonic 函数

cdef extern from "pandas/skiplist.h":
    ctypedef struct node_t:
        node_t **next
        int *width
        double value
        int is_nil
        int levels
        int ref_count

    ctypedef struct skiplist_t:
        node_t *head
        node_t **tmp_chain
        int *tmp_steps
        int size
        int maxlevels

    skiplist_t* skiplist_init(int) nogil
    void skiplist_destroy(skiplist_t*) nogil
    double skiplist_get(skiplist_t*, int, int*) nogil
    int skiplist_insert(skiplist_t*, double) nogil
    int skiplist_remove(skiplist_t*, double) nogil
    int skiplist_rank(skiplist_t*, double) nogil
    int skiplist_min_rank(skiplist_t*, double) nogil
# 从外部 C 头文件 "pandas/skiplist.h" 中声明 skiplist 相关的 C 结构体和函数

cdef:
    float32_t MINfloat32 = -np.inf
    float64_t MINfloat64 = -np.inf

    float32_t MAXfloat32 = np.inf
    float64_t MAXfloat64 = np.inf

    float64_t NaN = <float64_t>np.nan
# 定义一些常量：最小的 float32 和 float64、最大的 float32 和 float64，以及 NaN

cdef bint is_monotonic_increasing_start_end_bounds(
    ndarray[int64_t, ndim=1] start, ndarray[int64_t, ndim=1] end
):
    return is_monotonic(start, False)[0] and is_monotonic(end, False)[0]
# 定义一个 Cython 函数，检查给定的两个一维整数数组是否是单调递增的起始和结束边界

# ----------------------------------------------------------------------
# Rolling sum

cdef float64_t calc_sum(int64_t minp, int64_t nobs, float64_t sum_x,
                        int64_t num_consecutive_same_value, float64_t prev_value
                        ) noexcept nogil:
    cdef:
        float64_t result
    # 计算滚动总和

    if nobs == 0 == minp:
        result = 0
    elif nobs >= minp:
        if num_consecutive_same_value >= nobs:
            result = prev_value * nobs
        else:
            result = sum_x
    else:
        result = NaN

    return result
# 计算滚动总和的 Cython 函数，根据参数计算结果并返回

cdef void add_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                  float64_t *compensation, int64_t *num_consecutive_same_value,
                  float64_t *prev_value) noexcept nogil:
    """ add a value from the sum calc using Kahan summation """
    # 使用 Kahan 累加算法添加计算的值

    cdef:
        float64_t y, t

    # Not NaN
    if val == val:
        nobs[0] = nobs[0] + 1
        y = val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t

        # GH#42064, record num of same values to remove floating point artifacts
        if val == prev_value[0]:
            num_consecutive_same_value[0] += 1
        else:
            # reset to 1 (include current value itself)
            num_consecutive_same_value[0] = 1
        prev_value[0] = val
# 使用 Kahan 累加算法向计算总和中添加值的 Cython 函数

cdef void remove_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                     float64_t *compensation) noexcept nogil:
    """ remove a value from the sum calc using Kahan summation """

    cdef:
        float64_t y, t  # 定义两个 C 语言风格的双精度浮点数变量 y 和 t

    # 如果 val 不是 NaN（非数字）
    if val == val:
        # 减少观测数的计数
        nobs[0] = nobs[0] - 1
        # 计算 Kahan 累加法需要移除的值
        y = - val - compensation[0]
        # 更新累加和 sum_x，并考虑补偿项 y
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
# 定义一个函数 `roll_sum`，接受一些输入参数，并返回一个 `np.ndarray` 类型的数组
def roll_sum(const float64_t[:] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    # 定义一些 C 语言扩展的变量
    cdef:
        Py_ssize_t i, j  # 定义两个 Py_ssize_t 类型的循环索引变量
        float64_t sum_x, compensation_add, compensation_remove, prev_value  # 定义几个浮点数类型的变量
        int64_t s, e, num_consecutive_same_value  # 定义几个整数类型的变量
        int64_t nobs = 0, N = len(start)  # 初始化 nobs 为 0，N 为 start 数组的长度
        ndarray[float64_t] output  # 定义一个浮点数类型的数组 output

    # 调用函数检查 start 和 end 是否是递增的边界
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    # 创建一个空的浮点数数组 output，大小为 N
    output = np.empty(N, dtype=np.float64)

    # 进入无 GIL 环境
    with nogil:
        # 对每个序列进行循环
        for i in range(0, N):
            s = start[i]  # 取出当前序列的起始索引
            e = end[i]    # 取出当前序列的结束索引

            # 如果是第一个序列或者边界不是递增的或者当前序列的起始大于上一个序列的结束
            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:

                # 设置初始值
                prev_value = values[s]  # 将起始值设为当前序列的第一个值
                num_consecutive_same_value = 0  # 连续相同值的数量设为 0
                sum_x = compensation_add = compensation_remove = 0  # 初始化几个补偿和求和的变量
                nobs = 0  # 观测数量设为 0
                # 对当前序列进行循环
                for j in range(s, e):
                    # 调用函数 add_sum，更新 nobs、sum_x、compensation_add 等变量
                    add_sum(values[j], &nobs, &sum_x, &compensation_add,
                            &num_consecutive_same_value, &prev_value)

            else:
                # 计算删除的部分
                for j in range(start[i - 1], s):
                    # 调用函数 remove_sum，更新 nobs、sum_x、compensation_remove 等变量
                    remove_sum(values[j], &nobs, &sum_x, &compensation_remove)

                # 计算添加的部分
                for j in range(end[i - 1], e):
                    # 调用函数 add_sum，更新 nobs、sum_x、compensation_add 等变量
                    add_sum(values[j], &nobs, &sum_x, &compensation_add,
                            &num_consecutive_same_value, &prev_value)

            # 计算当前序列的总和，并将结果存入 output 数组中
            output[i] = calc_sum(
                minp, nobs, sum_x, num_consecutive_same_value, prev_value
            )

            # 如果边界不是递增的，重置 nobs 和 sum_x，使其为初始值
            if not is_monotonic_increasing_bounds:
                nobs = 0
                sum_x = 0.0
                compensation_remove = 0.0

    # 返回计算结果数组 output
    return output


# ----------------------------------------------------------------------
# 滚动均值


# 定义一个 C 语言扩展函数 calc_mean，计算均值
cdef float64_t calc_mean(int64_t minp, Py_ssize_t nobs, Py_ssize_t neg_ct,
                         float64_t sum_x, int64_t num_consecutive_same_value,
                         float64_t prev_value) noexcept nogil:
    cdef:
        float64_t result  # 定义一个浮点数类型的结果变量

    # 如果观测数量大于等于最小观测数，并且观测数量大于 0
    if nobs >= minp and nobs > 0:
        result = sum_x / <float64_t>nobs  # 计算均值
        # 如果连续相同值的数量大于等于观测数量，则结果设为前一个值
        if num_consecutive_same_value >= nobs:
            result = prev_value
        # 如果负数计数为 0 且结果小于 0，则结果设为 0
        elif neg_ct == 0 and result < 0:
            result = 0
        # 如果负数计数等于观测数量且结果大于 0，则结果设为 0
        elif neg_ct == nobs and result > 0:
            result = 0
        else:
            pass  # 否则不做任何处理
    else:
        result = NaN  # 如果观测数量不足或者为 0，则结果设为 NaN
    return result  # 返回计算结果


# 定义一个 C 语言扩展函数 add_mean，使用 Kahan 和算法计算均值
cdef void add_mean(
    float64_t val,
    Py_ssize_t *nobs,
    float64_t *sum_x,
    Py_ssize_t *neg_ct,
    float64_t *compensation,
    int64_t *num_consecutive_same_value,
    float64_t *prev_value
) noexcept nogil:
    """ add a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t  # 定义两个浮点数类型的变量

    # 如果值不是 NaN，则执行以下操作
    # (此处省略了部分代码，因为在注释中不需要重现所有细节)
    # 检查 val 是否等于自身，通常这种比较没有实际意义，可能是代码中的错误或占位符
    if val == val:
        # 增加观察计数器的值
        nobs[0] = nobs[0] + 1
        # 计算修正后的值 y
        y = val - compensation[0]
        # 更新 sum_x[0]，将 y 加入总和
        t = sum_x[0] + y
        # 更新 compensation[0]，修正 sum_x[0] 的变化
        compensation[0] = t - sum_x[0] - y
        # 更新 sum_x[0] 为新的总和
        sum_x[0] = t
        # 如果 val 是负数，增加负数计数器
        if signbit(val):
            neg_ct[0] = neg_ct[0] + 1

        # GH#42064，记录连续相同值的数量，以消除浮点数误差
        if val == prev_value[0]:
            # 如果当前值与前一个值相同，增加连续相同值的计数
            num_consecutive_same_value[0] += 1
        else:
            # 如果当前值与前一个值不同，重置连续相同值的计数为 1（包括当前值本身）
            num_consecutive_same_value[0] = 1
        # 更新 prev_value[0] 为当前值，以备下一次比较使用
        prev_value[0] = val
# 定义一个 C 语言扩展函数，用于从均值计算中移除一个值，使用 Kahan 累加算法
cdef void remove_mean(float64_t val, Py_ssize_t *nobs, float64_t *sum_x,
                      Py_ssize_t *neg_ct, float64_t *compensation) noexcept nogil:
    """ remove a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t

    # 如果值是有效的（不是 NaN）
    if val == val:
        # 观测数减一
        nobs[0] = nobs[0] - 1
        # 计算修正后的值
        y = - val - compensation[0]
        # 更新累加和
        t = sum_x[0] + y
        # 更新补偿项
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
        # 如果值为负数，负数计数减一
        if signbit(val):
            neg_ct[0] = neg_ct[0] - 1


# 计算滚动均值函数
def roll_mean(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        float64_t val, compensation_add, compensation_remove, sum_x, prev_value
        int64_t s, e, num_consecutive_same_value
        Py_ssize_t nobs, i, j, neg_ct, N = len(start)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    # 判断每段区间是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    # 初始化输出数组
    output = np.empty(N, dtype=np.float64)

    # 使用 nogil 上下文进行并行化处理
    with nogil:

        # 遍历每个区间段
        for i in range(0, N):
            s = start[i]
            e = end[i]

            # 如果是第一个区间或者区间段不是单调递增或者当前段的起始大于前一个段的终止
            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:

                # 设置初始值
                compensation_add = compensation_remove = sum_x = 0
                nobs = neg_ct = 0
                prev_value = values[s]
                num_consecutive_same_value = 0
                # 计算当前区间的均值
                for j in range(s, e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add,
                             &num_consecutive_same_value, &prev_value)

            else:

                # 计算需要删除的均值
                for j in range(start[i - 1], s):
                    val = values[j]
                    remove_mean(val, &nobs, &sum_x, &neg_ct, &compensation_remove)

                # 计算需要添加的均值
                for j in range(end[i - 1], e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add,
                             &num_consecutive_same_value, &prev_value)

            # 计算当前区间的平均值
            output[i] = calc_mean(
                minp, nobs, neg_ct, sum_x, num_consecutive_same_value, prev_value
            )

            # 如果区间段不是单调递增，重置计数和累加和等变量
            if not is_monotonic_increasing_bounds:
                nobs = 0
                neg_ct = 0
                sum_x = 0.0
                compensation_remove = 0.0
    return output


# ----------------------------------------------------------------------
# 滚动方差计算函数


cdef float64_t calc_var(
    int64_t minp,
    int ddof,
    float64_t nobs,
    float64_t ssqdm_x,
    int64_t num_consecutive_same_value
) noexcept nogil:
    cdef:
        float64_t result

    # 如果观测数不变，方差保持不变
    # 检查观测值数量是否大于等于最小观测数和自由度减少值
    if (nobs >= minp) and (nobs > ddof):
        # 处理特殊情况和重复相同值的情况
        if nobs == 1 or num_consecutive_same_value >= nobs:
            # 如果只有一个观测值或连续相同值超过观测值总数，则结果为0
            result = 0
        else:
            # 计算结果为平方和除以观测值总数减去自由度减少值
            result = ssqdm_x / (nobs - <float64_t>ddof)
    else:
        # 如果观测值数量不满足条件，则结果为NaN
        result = NaN

    # 返回计算结果
    return result
cdef void add_var(
    float64_t val,
    float64_t *nobs,
    float64_t *mean_x,
    float64_t *ssqdm_x,
    float64_t *compensation,
    int64_t *num_consecutive_same_value,
    float64_t *prev_value,
) noexcept nogil:
    """ add a value from the var calc """
    cdef:
        float64_t delta, prev_mean, y, t

    # GH#21813, if msvc 2017 bug is resolved, we should be OK with != instead of `isnan`
    # 检查 val 是否为 NaN，如果是则直接返回，不进行后续计算
    if val != val:
        return

    # 增加观测次数计数器
    nobs[0] = nobs[0] + 1

    # GH#42064, 记录连续相同值的数量，以移除浮点数计算中的误差
    # 如果当前值与上一个值相同，则增加连续相同值的计数
    if val == prev_value[0]:
        num_consecutive_same_value[0] += 1
    else:
        # 否则重置计数器为1（包括当前值本身）
        num_consecutive_same_value[0] = 1
    prev_value[0] = val

    # 使用 Welford 方法进行在线方差计算，采用 Kahan 求和算法
    # 参考：https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    prev_mean = mean_x[0] - compensation[0]
    y = val - compensation[0]
    t = y - mean_x[0]
    compensation[0] = t + mean_x[0] - y
    delta = t
    if nobs[0]:
        mean_x[0] = mean_x[0] + delta / nobs[0]
    else:
        mean_x[0] = 0
    ssqdm_x[0] = ssqdm_x[0] + (val - prev_mean) * (val - mean_x[0])


cdef void remove_var(
    float64_t val,
    float64_t *nobs,
    float64_t *mean_x,
    float64_t *ssqdm_x,
    float64_t *compensation
) noexcept nogil:
    """ remove a value from the var calc """
    cdef:
        float64_t delta, prev_mean, y, t
    # 如果 val 是 NaN，则直接返回，不进行移除操作
    if val == val:
        nobs[0] = nobs[0] - 1
        if nobs[0]:
            # 使用 Welford 方法进行在线方差计算，采用 Kahan 求和算法
            # 参考：https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            prev_mean = mean_x[0] - compensation[0]
            y = val - compensation[0]
            t = y - mean_x[0]
            compensation[0] = t + mean_x[0] - y
            delta = t
            mean_x[0] = mean_x[0] - delta / nobs[0]
            ssqdm_x[0] = ssqdm_x[0] - (val - prev_mean) * (val - mean_x[0])
        else:
            mean_x[0] = 0
            ssqdm_x[0] = 0


def roll_var(const float64_t[:] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp, int ddof=1) -> np.ndarray:
    """
    Numerically stable implementation using Welford's method.
    """
    cdef:
        float64_t mean_x, ssqdm_x, nobs, compensation_add,
        float64_t compensation_remove, prev_value
        int64_t s, e, num_consecutive_same_value
        Py_ssize_t i, j, N = len(start)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    # 确保 minp 至少为1
    minp = max(minp, 1)
    # 检查起始和结束边界是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    # 初始化输出数组
    output = np.empty(N, dtype=np.float64)
    # 使用 nogil 上下文，可能是指在没有全局解释器锁的情况下执行循环
    with nogil:
        # 遍历从 0 到 N-1 的索引
        for i in range(0, N):
            # 获取当前窗口的起始和结束索引
            s = start[i]
            e = end[i]

            # 在第一个窗口内，观察值只能添加，不能移除
            # 或者当前窗口不是单调递增边界，或者当前窗口起始大于前一个窗口的结束
            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:
                # 获取起始索引处的值作为初始值
                prev_value = values[s]
                # 记录连续相同值的数量
                num_consecutive_same_value = 0

                # 初始化用于计算方差的变量
                mean_x = ssqdm_x = nobs = compensation_add = compensation_remove = 0
                # 遍历当前窗口内的每一个索引
                for j in range(s, e):
                    # 调用函数将当前值添加到方差计算中
                    add_var(values[j], &nobs, &mean_x, &ssqdm_x, &compensation_add,
                            &num_consecutive_same_value, &prev_value)

            else:
                # 第一个窗口之后，观察值可以添加和删除

                # 计算需要删除的观察值
                for j in range(start[i - 1], s):
                    remove_var(values[j], &nobs, &mean_x, &ssqdm_x,
                               &compensation_remove)

                # 计算需要添加的观察值
                for j in range(end[i - 1], e):
                    add_var(values[j], &nobs, &mean_x, &ssqdm_x, &compensation_add,
                            &num_consecutive_same_value, &prev_value)

            # 计算当前窗口内的方差，并将结果存入输出数组中的当前位置
            output[i] = calc_var(minp, ddof, nobs, ssqdm_x, num_consecutive_same_value)

            # 如果不是单调递增边界，重置相关变量
            if not is_monotonic_increasing_bounds:
                nobs = 0.0
                mean_x = 0.0
                ssqdm_x = 0.0
                compensation_remove = 0.0

    # 返回计算结果的输出数组
    return output
# ----------------------------------------------------------------------
# Rolling skewness

# 计算滚动偏度的函数
cdef float64_t calc_skew(int64_t minp, int64_t nobs,
                         float64_t x, float64_t xx, float64_t xxx,
                         int64_t num_consecutive_same_value
                         ) noexcept nogil:
    cdef:
        float64_t result, dnobs
        float64_t A, B, C, R

    if nobs >= minp:
        # 将 nobs 转换为浮点数
        dnobs = <float64_t>nobs
        # 计算均值 A
        A = x / dnobs
        # 计算方差 B
        B = xx / dnobs - A * A
        # 计算偏度 C
        C = xxx / dnobs - A * A * A - 3 * A * B

        # 如果观测次数小于3，则结果为 NaN
        if nobs < 3:
            result = NaN
        # 如果连续相同值的次数大于等于观测次数，则强制结果为 0.0
        elif num_consecutive_same_value >= nobs:
            result = 0.0
        # 处理均匀分布的情况，当方差 B 很小（小于等于1e-14）时，结果为 NaN
        elif B <= 1e-14:
            result = NaN
        else:
            # 计算 R，即方差的平方根
            R = sqrt(B)
            # 计算偏度的值
            result = ((sqrt(dnobs * (dnobs - 1.)) * C) /
                      ((dnobs - 2) * R * R * R))
    else:
        # 如果观测次数小于指定最小观测次数，则结果为 NaN
        result = NaN

    return result


# 向偏度计算中添加一个新值的函数
cdef void add_skew(float64_t val, int64_t *nobs,
                   float64_t *x, float64_t *xx,
                   float64_t *xxx,
                   float64_t *compensation_x,
                   float64_t *compensation_xx,
                   float64_t *compensation_xxx,
                   int64_t *num_consecutive_same_value,
                   float64_t *prev_value,
                   ) noexcept nogil:
    """ add a value from the skew calc """
    cdef:
        float64_t y, t

    # 如果值不是 NaN
    if val == val:
        # 增加观测次数
        nobs[0] = nobs[0] + 1

        # 计算均值 x
        y = val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t

        # 计算方差 xx
        y = val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t

        # 计算偏度 xxx
        y = val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t

        # 记录连续相同值的数量，以消除浮点数误差
        if val == prev_value[0]:
            num_consecutive_same_value[0] += 1
        else:
            # 如果不是连续相同值，则重置为1（包括当前值本身）
            num_consecutive_same_value[0] = 1
        prev_value[0] = val
# 定义一个 Cython 函数，用于从偏度计算中移除一个值
cdef void remove_skew(float64_t val, int64_t *nobs,
                      float64_t *x, float64_t *xx,
                      float64_t *xxx,
                      float64_t *compensation_x,
                      float64_t *compensation_xx,
                      float64_t *compensation_xxx) noexcept nogil:
    """ remove a value from the skew calc """
    cdef:
        float64_t y, t

    # 检查 val 是否为非 NaN 值
    if val == val:
        # 减少观测数目计数器
        nobs[0] = nobs[0] - 1

        # 更新 x 的累计偏差
        y = - val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t

        # 更新 xx 的累计偏差
        y = - val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t

        # 更新 xxx 的累计偏差
        y = - val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t


# 定义一个 Cython 函数，用于计算滚动偏度
def roll_skew(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t val, min_val, mean_val, sum_val = 0
        float64_t compensation_xxx_add, compensation_xxx_remove
        float64_t compensation_xx_add, compensation_xx_remove
        float64_t compensation_x_add, compensation_x_remove
        float64_t x, xx, xxx
        float64_t prev_value
        int64_t nobs = 0, N = len(start), V = len(values), nobs_mean = 0
        int64_t s, e, num_consecutive_same_value
        ndarray[float64_t] output, values_copy
        bint is_monotonic_increasing_bounds

    # 确保 minp 至少为 3
    minp = max(minp, 3)

    # 检查起始和结束位置是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )

    # 创建一个空的输出数组
    output = np.empty(N, dtype=np.float64)

    # 计算 values 数组中的最小值
    min_val = np.nanmin(values)

    # 复制 values 数组
    values_copy = np.copy(values)
    # 使用 nogil 上下文，这表示在此部分代码中禁止全局解释器锁 (GIL)
    with nogil:
        # 对于每个索引 i 在范围 0 到 V 之间进行循环迭代
        for i in range(0, V):
            # 从 values_copy 中获取当前值 val
            val = values_copy[i]
            # 检查 val 是否等于自身（通常用于检测 NaN）
            if val == val:
                # 如果 val 是有效的，增加有效观测值的计数
                nobs_mean += 1
                # 累加有效值的总和
                sum_val += val
        # 计算有效值的平均数
        mean_val = sum_val / nobs_mean
        # 如果最小值与平均值的差小于 -1e5，则执行以下操作，以避免最小值过小导致精度问题
        if min_val - mean_val > -1e5:
            # 对平均值进行四舍五入
            mean_val = round(mean_val)
            # 对于每个索引 i 在范围 0 到 V 之间进行循环迭代
            for i in range(0, V):
                # 将 values_copy[i] 减去平均值，实现调整
                values_copy[i] = values_copy[i] - mean_val

        # 对于每个索引 i 在范围 0 到 N 之间进行循环迭代
        for i in range(0, N):
            # 获取起始和结束索引
            s = start[i]
            e = end[i]

            # 如果是第一个窗口，或者不是单调递增边界，或者当前窗口的起始大于上一个窗口的结束
            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:

                # 获取起始索引处的值作为前一个值
                prev_value = values[s]
                # 记录连续相同值的数量
                num_consecutive_same_value = 0

                # 初始化补偿值为0
                compensation_xxx_add = compensation_xxx_remove = 0
                compensation_xx_add = compensation_xx_remove = 0
                compensation_x_add = compensation_x_remove = 0
                x = xx = xxx = 0
                nobs = 0
                # 对于每个索引 j 在范围 s 到 e 之间进行循环迭代
                for j in range(s, e):
                    # 从 values_copy 中获取当前值 val
                    val = values_copy[j]
                    # 调用 add_skew 函数，更新 nobs、x、xx、xxx 等参数
                    add_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_add,
                             &compensation_xx_add, &compensation_xxx_add,
                             &num_consecutive_same_value, &prev_value)

            else:
                # 在第一个窗口之后，观测值可以同时添加和移除

                # 计算需要移除的观测值
                for j in range(start[i - 1], s):
                    # 从 values_copy 中获取当前值 val
                    val = values_copy[j]
                    # 调用 remove_skew 函数，更新 nobs、x、xx、xxx 等参数
                    remove_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_remove,
                                &compensation_xx_remove, &compensation_xxx_remove)

                # 计算需要添加的观测值
                for j in range(end[i - 1], e):
                    # 从 values_copy 中获取当前值 val
                    val = values_copy[j]
                    # 调用 add_skew 函数，更新 nobs、x、xx、xxx 等参数
                    add_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_add,
                             &compensation_xx_add, &compensation_xxx_add,
                             &num_consecutive_same_value, &prev_value)

            # 计算输出数组中的每个元素的偏度
            output[i] = calc_skew(minp, nobs, x, xx, xxx, num_consecutive_same_value)

            # 如果不是单调递增边界，重置 nobs、x、xx、xxx 为初始值
            if not is_monotonic_increasing_bounds:
                nobs = 0
                x = 0.0
                xx = 0.0
                xxx = 0.0

    # 返回最终计算出的输出数组
    return output
# ----------------------------------------------------------------------
# Rolling kurtosis

# 定义一个函数calc_kurt，计算滚动峰度
cdef float64_t calc_kurt(int64_t minp, int64_t nobs,
                         float64_t x, float64_t xx,
                         float64_t xxx, float64_t xxxx,
                         int64_t num_consecutive_same_value,
                         ) noexcept nogil:
    cdef:
        float64_t result, dnobs
        float64_t A, B, C, D, R, K

    # 如果观测数大于等于最小观测数
    if nobs >= minp:
        # 如果观测数小于4，结果设为NaN
        if nobs < 4:
            result = NaN
        # 若连续相同值的数量等于观测数，强制结果为-3
        elif num_consecutive_same_value >= nobs:
            result = -3.
        else:
            # 将观测数转换为浮点数
            dnobs = <float64_t>nobs
            # 计算均值A、二阶中心距B、三阶中心距C、四阶中心距D
            A = x / dnobs
            R = A * A
            B = xx / dnobs - R
            R = R * A
            C = xxx / dnobs - R - 3 * A * B
            R = R * A
            D = xxxx / dnobs - R - 6 * B * A * A - 4 * C * A

            # 当B小于等于1e-14时，结果设为NaN，避免浮点数问题
            if B <= 1e-14:
                result = NaN
            else:
                # 计算峰度K，并计算最终结果
                K = (dnobs * dnobs - 1.) * D / (B * B) - 3 * ((dnobs - 1.) ** 2)
                result = K / ((dnobs - 2.) * (dnobs - 3.))
    else:
        # 如果观测数小于最小观测数，结果设为NaN
        result = NaN

    return result


# 定义一个void函数add_kurt，添加峰度计算中的值
cdef void add_kurt(float64_t val, int64_t *nobs,
                   float64_t *x, float64_t *xx,
                   float64_t *xxx, float64_t *xxxx,
                   float64_t *compensation_x,
                   float64_t *compensation_xx,
                   float64_t *compensation_xxx,
                   float64_t *compensation_xxxx,
                   int64_t *num_consecutive_same_value,
                   float64_t *prev_value
                   ) noexcept nogil:
    """ add a value from the kurotic calc """
    cdef:
        float64_t y, t

    # 如果值不是NaN
    # 如果 val 等于自身（这行代码的逻辑存在疑问，因为它总是成立），则执行以下操作
    nobs[0] = nobs[0] + 1

    # 计算新值 y，用于累加到 x[0]，并更新 compensation_x[0]
    y = val - compensation_x[0]
    t = x[0] + y
    compensation_x[0] = t - x[0] - y
    x[0] = t

    # 计算新值 y，用于累加到 xx[0]，并更新 compensation_xx[0]
    y = val * val - compensation_xx[0]
    t = xx[0] + y
    compensation_xx[0] = t - xx[0] - y
    xx[0] = t

    # 计算新值 y，用于累加到 xxx[0]，并更新 compensation_xxx[0]
    y = val * val * val - compensation_xxx[0]
    t = xxx[0] + y
    compensation_xxx[0] = t - xxx[0] - y
    xxx[0] = t

    # 计算新值 y，用于累加到 xxxx[0]，并更新 compensation_xxxx[0]
    y = val * val * val * val - compensation_xxxx[0]
    t = xxxx[0] + y
    compensation_xxxx[0] = t - xxxx[0] - y
    xxxx[0] = t

    # GH#42064，记录连续相同值的数量，以去除浮点数的误差
    if val == prev_value[0]:
        num_consecutive_same_value[0] += 1
    else:
        # 如果遇到不同的值，重置连续相同值的计数为1（包括当前值本身）
        num_consecutive_same_value[0] = 1
    prev_value[0] = val
# 定义一个 C 函数，用于从 kurtosis 计算中移除一个数值
cdef void remove_kurt(float64_t val, int64_t *nobs,
                      float64_t *x, float64_t *xx,
                      float64_t *xxx, float64_t *xxxx,
                      float64_t *compensation_x,
                      float64_t *compensation_xx,
                      float64_t *compensation_xxx,
                      float64_t *compensation_xxxx) noexcept nogil:
    """ remove a value from the kurotic calc """
    cdef:
        float64_t y, t

    # 如果 val 不是 NaN
    if val == val:
        # 减少观测值计数
        nobs[0] = nobs[0] - 1

        # 计算补偿并更新 x 的值
        y = - val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t

        # 计算补偿并更新 xx 的值
        y = - val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t

        # 计算补偿并更新 xxx 的值
        y = - val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t

        # 计算补偿并更新 xxxx 的值
        y = - val * val * val * val - compensation_xxxx[0]
        t = xxxx[0] + y
        compensation_xxxx[0] = t - xxxx[0] - y
        xxxx[0] = t


# 定义一个函数，用于计算滚动 kurtosis
def roll_kurt(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t val, mean_val, min_val, sum_val = 0
        float64_t compensation_xxxx_add, compensation_xxxx_remove
        float64_t compensation_xxx_remove, compensation_xxx_add
        float64_t compensation_xx_remove, compensation_xx_add
        float64_t compensation_x_remove, compensation_x_add
        float64_t x, xx, xxx, xxxx
        float64_t prev_value
        int64_t nobs, s, e, num_consecutive_same_value
        int64_t N = len(start), V = len(values), nobs_mean = 0
        ndarray[float64_t] output, values_copy
        bint is_monotonic_increasing_bounds

    # 确保 minp 至少为 4
    minp = max(minp, 4)
    # 检查起始和结束边界是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    # 创建一个空的输出数组
    output = np.empty(N, dtype=np.float64)
    # 复制输入的值数组
    values_copy = np.copy(values)
    # 计算输入值数组中的最小值（排除 NaN）
    min_val = np.nanmin(values)
    # 使用 nogil 上下文，避免全局解释器锁 (GIL) 的影响，提高并行性能
    with nogil:
        # 遍历值的副本，计算有效值的平均数和总和
        for i in range(0, V):
            val = values_copy[i]
            # 检查值是否有效
            if val == val:
                # 增加有效观测值的计数
                nobs_mean += 1
                # 累加有效值的总和
                sum_val += val
        # 计算有效值的平均数
        mean_val = sum_val / nobs_mean
        # 如果最小值与平均值的差小于 -1e4，则修正平均值
        # 这种情况下会对最小值造成不精确性
        if min_val - mean_val > -1e4:
            # 将平均值四舍五入
            mean_val = round(mean_val)
            # 对值的副本进行平均值的修正
            for i in range(0, V):
                values_copy[i] = values_copy[i] - mean_val

        # 遍历窗口的起始和结束位置
        for i in range(0, N):
            s = start[i]
            e = end[i]

            # 如果是第一个窗口，或者不是单调递增边界，或者当前起始位置大于前一个结束位置
            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:
                # 计算第一个窗口内的累计值
                prev_value = values[s]
                num_consecutive_same_value = 0

                compensation_xxxx_add = compensation_xxxx_remove = 0
                compensation_xxx_remove = compensation_xxx_add = 0
                compensation_xx_remove = compensation_xx_add = 0
                compensation_x_remove = compensation_x_add = 0
                x = xx = xxx = xxxx = 0
                nobs = 0
                # 对窗口内的值进行 kurtosis 计算并修正
                for j in range(s, e):
                    add_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                             &compensation_x_add, &compensation_xx_add,
                             &compensation_xxx_add, &compensation_xxxx_add,
                             &num_consecutive_same_value, &prev_value)

            else:
                # 如果不是第一个窗口，可以同时增加和删除观测值
                # 计算需要删除的观测值
                for j in range(start[i - 1], s):
                    remove_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                                &compensation_x_remove, &compensation_xx_remove,
                                &compensation_xxx_remove, &compensation_xxxx_remove)

                # 计算需要添加的观测值
                for j in range(end[i - 1], e):
                    add_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                             &compensation_x_add, &compensation_xx_add,
                             &compensation_xxx_add, &compensation_xxxx_add,
                             &num_consecutive_same_value, &prev_value)

            # 计算当前窗口的 kurtosis 值并存储到输出数组中
            output[i] = calc_kurt(minp, nobs, x, xx, xxx, xxxx,
                                  num_consecutive_same_value)

            # 如果不是单调递增边界，则重置累计值
            if not is_monotonic_increasing_bounds:
                nobs = 0
                x = 0.0
                xx = 0.0
                xxx = 0.0
                xxxx = 0.0

    # 返回最终的输出数组
    return output
# ----------------------------------------------------------------------
# Rolling median, min, max

# 定义了一个 Cython 函数，用于计算滚动中位数、最小值、最大值
def roll_median_c(const float64_t[:] values, ndarray[int64_t] start,
                  ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j  # 定义 C 语言风格的变量
        bint err = False, is_monotonic_increasing_bounds  # 定义布尔类型的变量
        int midpoint, ret = 0  # 定义整数类型的变量
        int64_t nobs = 0, N = len(start), s, e, win  # 定义整数类型的变量
        float64_t val, res  # 定义浮点数类型的变量
        skiplist_t *sl  # 定义 skiplist_t 结构体指针
        ndarray[float64_t] output  # 定义浮点数数组

    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )

    # 使用 Fixed/Variable Indexer 作为 skiplist 操作的替代，因为实际的 skiplist 操作超过了窗口计算成本
    output = np.empty(N, dtype=np.float64)  # 创建一个空的浮点数数组作为输出

    if (end - start).max() == 0:
        output[:] = NaN  # 如果窗口大小为0，则输出数组全部置为 NaN
        return output

    win = (end - start).max()  # 计算窗口大小为起始和结束数组的最大差值
    sl = skiplist_init(<int>win)  # 初始化 skiplist，设置窗口大小为 win
    if sl == NULL:
        raise MemoryError("skiplist_init failed")  # 如果初始化失败则抛出内存错误异常

    with nogil:  # 使用 nogil 上下文进行无全局解锁的并行处理

        for i in range(0, N):  # 遍历起始和结束数组的长度

            s = start[i]  # 获取当前区间的起始值
            e = end[i]    # 获取当前区间的结束值

            if i == 0 or not is_monotonic_increasing_bounds or s >= end[i - 1]:
                # 如果是第一个区间或者不是单调递增的边界或者当前起始值大于上一个区间的结束值

                if i != 0:
                    skiplist_destroy(sl)  # 销毁当前的 skiplist
                    sl = skiplist_init(<int>win)  # 重新初始化 skiplist，设置窗口大小为 win
                    nobs = 0  # 重置观测数为0

                # 设置阶段
                for j in range(s, e):  # 遍历当前区间的每个索引

                    val = values[j]  # 获取当前值
                    if val == val:  # 如果值为有效数字（不是 NaN）

                        nobs += 1  # 增加观测数
                        err = skiplist_insert(sl, val) == -1  # 尝试将值插入 skiplist，记录是否出错
                        if err:
                            break  # 如果插入出错则跳出循环

            else:
                # 计算增加的部分
                for j in range(end[i - 1], e):
                    val = values[j]
                    if val == val:
                        nobs += 1
                        err = skiplist_insert(sl, val) == -1
                        if err:
                            break

                # 计算删除的部分
                for j in range(start[i - 1], s):
                    val = values[j]
                    if val == val:
                        skiplist_remove(sl, val)  # 从 skiplist 中删除值
                        nobs -= 1  # 减少观测数

            if nobs >= minp:  # 如果观测数大于等于最小观测数要求

                midpoint = <int>(nobs / 2)  # 计算中间点位置
                if nobs % 2:
                    res = skiplist_get(sl, midpoint, &ret)  # 获取中位数值
                else:
                    res = (skiplist_get(sl, midpoint, &ret) +
                           skiplist_get(sl, (midpoint - 1), &ret)) / 2  # 获取中位数值（偶数情况）

                if ret == 0:
                    res = NaN  # 如果返回值标志为0，则结果设为 NaN
            else:
                res = NaN  # 如果观测数不足则结果设为 NaN

            output[i] = res  # 将结果存入输出数组中

            if not is_monotonic_increasing_bounds:
                nobs = 0  # 重置观测数为0
                skiplist_destroy(sl)  # 销毁 skiplist
                sl = skiplist_init(<int>win)  # 重新初始化 skiplist，设置窗口大小为 win

    skiplist_destroy(sl)  # 最终销毁 skiplist
    if err:
        raise MemoryError("skiplist_insert failed")  # 如果插入操作出错则抛出内存错误异常

    return output  # 返回计算结果的输出数组

# ----------------------------------------------------------------------
# Moving maximum / minimum code taken from Bottleneck
# Licence at LICENSES/BOTTLENECK_LICENCE

# 定义一个 C 扩展函数，用于初始化移动最大/最小值的计算
# ai: 初始值
# nobs: 观测值个数的指针
# is_max: 是否计算移动最大值
cdef float64_t init_mm(float64_t ai, Py_ssize_t *nobs, bint is_max) noexcept nogil:
    # 如果 ai 是有效的数值
    if ai == ai:
        # 增加观测值个数
        nobs[0] = nobs[0] + 1
    # 如果 ai 无效且需要计算最大值
    elif is_max:
        # 设置 ai 为最小的 float64 值
        ai = MINfloat64
    # 如果 ai 无效且需要计算最小值
    else:
        # 设置 ai 为最大的 float64 值
        ai = MAXfloat64

    return ai

# 定义一个 C 扩展函数，用于移除移动最大/最小值计算中的某个值
# aold: 待移除的值
# nobs: 观测值个数的指针
cdef void remove_mm(float64_t aold, Py_ssize_t *nobs) noexcept nogil:
    """ remove a value from the mm calc """
    # 如果待移除的值是有效的数值
    if aold == aold:
        # 减少观测值个数
        nobs[0] = nobs[0] - 1

# 定义一个 C 扩展函数，用于计算移动最大/最小值
# minp: 最小观测数
# nobs: 当前窗口内的观测值个数
# value: 当前值
cdef float64_t calc_mm(int64_t minp, Py_ssize_t nobs,
                       float64_t value) noexcept nogil:
    cdef:
        float64_t result

    # 如果当前窗口内的观测值个数大于等于最小观测数
    if nobs >= minp:
        # 结果为当前值
        result = value
    else:
        # 否则结果为 NaN
        result = NaN

    return result

# Python 函数，计算一维数组的滚动最大值
# values: np.ndarray，包含浮点数的数组
# start: np.ndarray，滚动窗口的起始索引
# end: np.ndarray，滚动窗口的结束索引
# minp: 如果窗口内的观测值数量低于此值，则输出 NaN
def roll_max(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving max of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    start : np.ndarray[np.int64]
        Start indices for rolling window computation.
    end : np.ndarray[np.int64]
        End indices for rolling window computation.
    minp : int
        Minimum number of observations in window to compute max.

    Returns
    -------
    np.ndarray[np.float64]
        Array of rolling maximum values.
    """
    # 调用内部函数 _roll_min_max，计算滚动最大值
    return _roll_min_max(values, start, end, minp, is_max=1)

# Python 函数，计算一维数组的滚动最小值
# values: np.ndarray，包含浮点数的数组
# start: np.ndarray，滚动窗口的起始索引
# end: np.ndarray，滚动窗口的结束索引
# minp: 如果窗口内的观测值数量低于此值，则输出 NaN
def roll_min(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving min of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    start : np.ndarray[np.int64]
        Start indices for rolling window computation.
    end : np.ndarray[np.int64]
        End indices for rolling window computation.
    minp : int
        Minimum number of observations in window to compute min.

    Returns
    -------
    np.ndarray[np.float64]
        Array of rolling minimum values.
    """
    # 调用内部函数 _roll_min_max，计算滚动最小值
    return _roll_min_max(values, start, end, minp, is_max=0)

# 定义一个 C 扩展函数，用于计算滚动最大/最小值的核心逻辑
# values: np.ndarray，包含浮点数的数组
# starti: np.ndarray，滚动窗口的起始索引
# endi: np.ndarray，滚动窗口的结束索引
# minp: 如果窗口内的观测值数量低于此值，则输出 NaN
# is_max: 是否计算滚动最大值
cdef _roll_min_max(ndarray[float64_t] values,
                   ndarray[int64_t] starti,
                   ndarray[int64_t] endi,
                   int64_t minp,
                   bint is_max):
    cdef:
        float64_t ai
        int64_t curr_win_size, start
        Py_ssize_t i, k, nobs = 0, N = len(starti)
        deque Q[int64_t]  # min/max always the front
        deque W[int64_t]  # track the whole window for nobs compute
        ndarray[float64_t, ndim=1] output

    # 初始化输出数组
    output = np.empty(N, dtype=np.float64)
    # 初始化两个 deque 结构
    Q = deque[int64_t]()
    W = deque[int64_t]()
    with nogil:
        # 使用 nogil 上下文，表示此处代码是在无全局解释器锁 (GIL) 的环境下执行

        # 这段代码使用了修改版的 C++ 代码，参考了以下 Stack Overflow 帖子：
        # https://stackoverflow.com/a/12239580
        # 原始实现未处理可变窗口大小，因此对代码进行了优化以支持此功能

        # 初始化第一个窗口的大小
        curr_win_size = endi[0] - starti[0]
        # GH 32865
        # 将输出索引锚定到值索引，以提供自定义的 BaseIndexer 支持
        for i in range(N):
            # 计算当前窗口的大小
            curr_win_size = endi[i] - starti[i]
            if i == 0:
                start = starti[i]
            else:
                start = endi[i - 1]

            for k in range(start, endi[i]):
                # 使用 values[k] 初始化移动平均 ai
                ai = init_mm(values[k], &nobs, is_max)
                
                # 如果是最大值，且 ai 大于等于 Q 的最后一个元素值，或者 Q 的最后一个元素不等于自身
                # 则弹出 Q 的最后一个元素，直到不满足条件为止
                if is_max:
                    while not Q.empty() and ((ai >= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                else:
                    # 如果是最小值，且 ai 小于等于 Q 的最后一个元素值，或者 Q 的最后一个元素不等于自身
                    # 则弹出 Q 的最后一个元素，直到不满足条件为止
                    while not Q.empty() and ((ai <= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                
                # 将 k 添加到 Q 和 W 中
                Q.push_back(k)
                W.push_back(k)

            # 清除位于当前窗口左侧以及左侧之外的条目
            while not Q.empty() and Q.front() <= starti[i] - 1:
                Q.pop_front()
            while not W.empty() and W.front() <= starti[i] - 1:
                # 移除 W 的最前端元素
                remove_mm(values[W.front()], &nobs)
                W.pop_front()

            # 根据输入值数组中的索引保存输出
            if not Q.empty() and curr_win_size > 0:
                output[i] = calc_mm(minp, nobs, values[Q.front()])
            else:
                # 如果 Q 为空或当前窗口大小小于等于 0，则输出 NaN
                output[i] = NaN

    # 返回计算结果的输出数组
    return output
# 定义一个枚举类型 InterpolationType，包含几种插值方式
cdef enum InterpolationType:
    LINEAR,
    LOWER,
    HIGHER,
    NEAREST,
    MIDPOINT

# 插值类型映射字典，将字符串映射到对应的 InterpolationType 枚举值
interpolation_types = {
    "linear": LINEAR,
    "lower": LOWER,
    "higher": HIGHER,
    "nearest": NEAREST,
    "midpoint": MIDPOINT,
}

# 定义一个函数 roll_quantile，接受一些输入参数并返回一个 NumPy 数组
def roll_quantile(const float64_t[:] values, ndarray[int64_t] start,
                  ndarray[int64_t] end, int64_t minp,
                  float64_t quantile, str interpolation) -> np.ndarray:
    """
    O(N log(window)) implementation using skip list
    """
    # 声明一些 Cython 变量
    cdef:
        Py_ssize_t i, j, s, e, N = len(start), idx
        int ret = 0
        int64_t nobs = 0, win
        float64_t val, idx_with_fraction
        float64_t vlow, vhigh
        skiplist_t *skiplist
        InterpolationType interpolation_type
        ndarray[float64_t] output

    # 检查 quantile 是否在 [0, 1] 范围内，若不是则抛出 ValueError 异常
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError(f"quantile value {quantile} not in [0, 1]")

    # 尝试从 interpolation_types 字典中获取插值类型，若不支持则抛出 ValueError 异常
    try:
        interpolation_type = interpolation_types[interpolation]
    except KeyError:
        raise ValueError(f"Interpolation '{interpolation}' is not supported")

    # 检查起始和结束索引数组是否单调递增，用于后续操作
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )

    # 初始化一个长度为 N 的空 NumPy 数组，用于存储计算结果
    output = np.empty(N, dtype=np.float64)

    # 计算窗口大小，并检查是否为 0，若为 0 则将 output 数组填充为 NaN 后返回
    win = (end - start).max()
    if win == 0:
        output[:] = NaN
        return output

    # 初始化一个 skiplist 结构体，如果初始化失败则抛出 MemoryError 异常
    skiplist = skiplist_init(<int>win)
    if skiplist == NULL:
        raise MemoryError("skiplist_init failed")

    # 销毁 skiplist 对象
    skiplist_destroy(skiplist)

    # 返回计算结果的 NumPy 数组
    return output


# 定义一个 rolling_rank_tiebreakers 字典，将方法字符串映射到对应的 TiebreakEnumType 枚举值
rolling_rank_tiebreakers = {
    "average": TiebreakEnumType.TIEBREAK_AVERAGE,
    "min": TiebreakEnumType.TIEBREAK_MIN,
    "max": TiebreakEnumType.TIEBREAK_MAX,
}

# 定义一个函数 roll_rank，接受一些输入参数并返回一个 NumPy 数组
def roll_rank(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp, bint percentile,
              str method, bint ascending) -> np.ndarray:
    """
    O(N log(window)) implementation using skip list

    derived from roll_quantile
    """
    # 声明一些 Cython 变量
    cdef:
        Py_ssize_t i, j, s, e, N = len(start)
        float64_t rank_min = 0, rank = 0
        int64_t nobs = 0, win
        float64_t val
        skiplist_t *skiplist
        float64_t[::1] output
        TiebreakEnumType rank_type

    # 尝试从 rolling_rank_tiebreakers 字典中获取排名方法类型，若不支持则抛出 ValueError 异常
    try:
        rank_type = rolling_rank_tiebreakers[method]
    except KeyError:
        raise ValueError(f"Method '{method}' is not supported")

    # 检查起始和结束索引数组是否单调递增，用于后续操作
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )

    # 初始化一个长度为 N 的空 NumPy 数组，用于存储计算结果
    output = np.empty(N, dtype=np.float64)

    # 计算窗口大小，并检查是否为 0，若为 0 则将 output 数组填充为 NaN 后返回
    win = (end - start).max()
    if win == 0:
        output[:] = NaN
        return np.asarray(output)

    # 初始化一个 skiplist 结构体，如果初始化失败则抛出 MemoryError 异常
    skiplist = skiplist_init(<int>win)
    if skiplist == NULL:
        raise MemoryError("skiplist_init failed")

    # 销毁 skiplist 对象
    skiplist_destroy(skiplist)

    # 返回计算结果的 NumPy 数组
    return np.asarray(output)
def roll_apply(object obj,
               ndarray[int64_t] start, ndarray[int64_t] end,
               int64_t minp,
               object function, bint raw,
               tuple args, dict kwargs) -> np.ndarray:
    cdef:
        ndarray[float64_t] output, counts  # 定义输出和计数数组，元素类型为 float64
        ndarray[float64_t, cast=True] arr  # 将 obj 转换为浮点类型数组 arr
        Py_ssize_t i, s, e, N = len(start), n = len(obj)  # 初始化循环变量和长度变量

    if n == 0:
        return np.array([], dtype=np.float64)  # 如果 obj 长度为0，返回空的浮点类型数组

    arr = np.asarray(obj)  # 将 obj 转换为 NumPy 数组 arr

    # ndarray input
    if raw and not arr.flags.c_contiguous:
        arr = arr.copy("C")  # 如果 raw 为真且 arr 不是 C 连续的，则复制 arr 成为 C 连续的

    counts = roll_sum(np.isfinite(arr).astype(float), start, end, minp)  # 计算 arr 的有限数元素，计算窗口内的元素个数

    output = np.empty(N, dtype=np.float64)  # 创建长度为 N 的空浮点类型数组 output

    for i in range(N):
        s = start[i]  # 获取起始索引
        e = end[i]    # 获取结束索引

        if counts[i] >= minp:
            if raw:
                output[i] = function(arr[s:e], *args, **kwargs)  # 如果 raw 为真，则将 arr 的切片作为参数传递给 function
            else:
                output[i] = function(obj.iloc[s:e], *args, **kwargs)  # 否则将 obj 的切片作为参数传递给 function
        else:
            output[i] = NaN  # 如果窗口内有效值的数量小于 minp，则将 output[i] 设置为 NaN

    return output  # 返回计算结果数组


# ----------------------------------------------------------------------
# Rolling sum and mean for weighted window


def roll_weighted_sum(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndarray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=0)  # 调用 _roll_weighted_sum_mean 计算加权和


def roll_weighted_mean(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndarray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=1)  # 调用 _roll_weighted_sum_mean 计算加权平均值


cdef float64_t[:] _roll_weighted_sum_mean(const float64_t[:] values,
                                          const float64_t[:] weights,
                                          int minp, bint avg):
    """
    Assume len(weights) << len(values)
    """
    cdef:
        float64_t[:] output, tot_wgt, counts  # 定义输出、总权重和计数数组，元素类型为 float64
        Py_ssize_t in_i, win_i, win_n, in_n  # 初始化循环变量和长度变量
        float64_t val_in, val_win, c, w  # 定义浮点类型变量

    in_n = len(values)  # 获取 values 数组的长度
    win_n = len(weights)  # 获取 weights 数组的长度

    output = np.zeros(in_n, dtype=np.float64)  # 创建长度为 in_n 的全零浮点类型数组 output
    counts = np.zeros(in_n, dtype=np.float64)  # 创建长度为 in_n 的全零浮点类型数组 counts
    if avg:
        tot_wgt = np.zeros(in_n, dtype=np.float64)  # 如果 avg 为真，创建长度为 in_n 的全零浮点类型数组 tot_wgt

    elif minp > in_n:
        minp = in_n + 1  # 如果 minp 大于 in_n，将 minp 设为 in_n + 1

    minp = max(minp, 1)  # 将 minp 设为 minp 和 1 中的较大值
    # 使用 nogil 上下文以释放全局解释器锁（GIL），提高代码运行效率
    with nogil:
        # 如果 avg 不为空（即为真），执行以下代码块
        if avg:
            # 遍历窗口数量范围
            for win_i in range(win_n):
                # 获取当前窗口权重值
                val_win = weights[win_i]
                # 如果权重值为 NaN，则跳过当前循环
                if val_win != val_win:
                    continue

                # 遍历输入数据的范围，考虑窗口偏移量
                for in_i in range(in_n - (win_n - win_i) + 1):
                    # 获取当前输入值
                    val_in = values[in_i]
                    # 如果输入值不为 NaN，则执行以下操作
                    if val_in == val_in:
                        # 将加权后的输入值添加到输出数组中对应位置
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        # 相应位置的计数器增加 1
                        counts[in_i + (win_n - win_i) - 1] += 1
                        # 相应位置的总权重增加当前窗口权重值
                        tot_wgt[in_i + (win_n - win_i) - 1] += val_win

            # 对每个输入位置执行后续操作
            for in_i in range(in_n):
                # 获取当前位置的计数器值
                c = counts[in_i]
                # 如果计数器值小于指定的最小值，则将输出数组对应位置设置为 NaN
                if c < minp:
                    output[in_i] = NaN
                else:
                    # 否则，计算平均值并更新输出数组
                    w = tot_wgt[in_i]
                    if w == 0:
                        output[in_i] = NaN
                    else:
                        output[in_i] /= tot_wgt[in_i]

        # 如果 avg 为空（即为假），执行以下代码块
        else:
            # 遍历窗口数量范围
            for win_i in range(win_n):
                # 获取当前窗口权重值
                val_win = weights[win_i]
                # 如果权重值为 NaN，则跳过当前循环
                if val_win != val_win:
                    continue

                # 遍历输入数据的范围，考虑窗口偏移量
                for in_i in range(in_n - (win_n - win_i) + 1):
                    # 获取当前输入值
                    val_in = values[in_i]

                    # 如果输入值不为 NaN，则执行以下操作
                    if val_in == val_in:
                        # 将加权后的输入值添加到输出数组中对应位置
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        # 相应位置的计数器增加 1
                        counts[in_i + (win_n - win_i) - 1] += 1

            # 对每个输入位置执行后续操作
            for in_i in range(in_n):
                # 获取当前位置的计数器值
                c = counts[in_i]
                # 如果计数器值小于指定的最小值，则将输出数组对应位置设置为 NaN
                if c < minp:
                    output[in_i] = NaN

    # 返回处理后的输出数组
    return output
# ----------------------------------------------------------------------
# Rolling var for weighted window

# 定义函数：计算加权窗口的方差
cdef float64_t calc_weighted_var(float64_t t,
                                 float64_t sum_w,
                                 Py_ssize_t win_n,
                                 unsigned int ddof,
                                 float64_t nobs,
                                 int64_t minp) noexcept nogil:
    """
    Calculate weighted variance for a window using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    t: float64_t
        sum of weighted squared differences
    sum_w: float64_t
        sum of weights
    win_n: Py_ssize_t
        window size
    ddof: unsigned int
        delta degrees of freedom
    nobs: float64_t
        number of observations
    minp: int64_t
        minimum number of observations

    Returns
    -------
    result : float64_t
        weighted variance of the window
    """

    cdef:
        float64_t result

    # 如果观测数大于等于最小观测数并且大于自由度调整值，则计算方差
    if (nobs >= minp) and (nobs > ddof):

        # 特殊情况：当只有一个观测时，方差为0
        if nobs == 1:
            result = 0
        else:
            result = t * win_n / ((win_n - ddof) * sum_w)
            if result < 0:
                result = 0
    else:
        # 如果不满足条件，则返回NaN
        result = NaN

    return result


# 定义函数：更新加权平均、权重总和和加权平方差总和，添加新值和权重对到加权方差计算中
cdef void add_weighted_var(float64_t val,
                           float64_t w,
                           float64_t *t,
                           float64_t *sum_w,
                           float64_t *mean,
                           float64_t *nobs) noexcept nogil:
    """
    Update weighted mean, sum of weights and sum of weighted squared
    differences to include value and weight pair in weighted variance
    calculation using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    val: float64_t
        window values
    w: float64_t
        window weights
    t: float64_t
        sum of weighted squared differences
    sum_w: float64_t
        sum of weights
    mean: float64_t
        weighted mean
    nobs: float64_t
        number of observations
    """

    cdef:
        float64_t temp, q, r

    # 如果值为NaN，则直接返回
    if val != val:
        return

    # 增加观测数
    nobs[0] = nobs[0] + 1

    q = val - mean[0]
    temp = sum_w[0] + w
    r = q * w / temp

    # 更新加权平均、加权平方差总和和权重总和
    mean[0] = mean[0] + r
    t[0] = t[0] + r * sum_w[0] * q
    sum_w[0] = temp


# 定义函数：更新加权平均、权重总和和加权平方差总和，从加权方差计算中移除值和权重对
cdef void remove_weighted_var(float64_t val,
                              float64_t w,
                              float64_t *t,
                              float64_t *sum_w,
                              float64_t *mean,
                              float64_t *nobs) noexcept nogil:
    """
    Update weighted mean, sum of weights and sum of weighted squared
    differences to remove value and weight pair from weighted variance
    calculation using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153
    """
    python
        # Parameters 参数说明部分，以下是各变量的含义和用途：
        # val: float64_t 浮点数，表示窗口中的数值
        # w: float64_t 浮点数，表示窗口中对应数值的权重
        # t: float64_t 浮点数，表示加权平方差的和
        # sum_w: float64_t 浮点数，表示权重的总和
        # mean: float64_t 浮点数，表示加权平均值
        # nobs: float64_t 浮点数，表示观测值的数量
    
        cdef:
            float64_t temp, q, r  # 定义Cython变量，temp, q, r 为 float64_t 类型
    
        if val == val:  # 检查 val 是否为 NaN（val == val 是 NaN 的一种检测方法）
    
            # 减少观测值数量
            nobs[0] = nobs[0] - 1
    
            if nobs[0]:  # 如果观测值数量不为零
    
                # 计算 q 值，表示当前值与加权平均值的差
                q = val - mean[0]
    
                # 更新临时变量 temp，减去当前权重 w
                temp = sum_w[0] - w
    
                # 计算 r 值，表示更新后的加权平均值的变化量
                r = q * w / temp
    
                # 更新加权平均值 mean
                mean[0] = mean[0] - r
    
                # 更新加权平方差和 t
                t[0] = t[0] - r * sum_w[0] * q
    
                # 更新权重总和 sum_w
                sum_w[0] = temp
    
            else:  # 如果观测值数量为零
    
                # 重置加权平方差和 t
                t[0] = 0
    
                # 重置权重总和 sum_w
                sum_w[0] = 0
    
                # 重置加权平均值 mean
                mean[0] = 0
# 定义一个函数用于计算加权滚动方差，采用West的在线算法。
def roll_weighted_var(const float64_t[:] values, const float64_t[:] weights,
                      int64_t minp, unsigned int ddof):
    """
    Calculates weighted rolling variance using West's online algorithm.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    values: float64_t[:]
        values to roll window over
    weights: float64_t[:]
        array of weights whose length is window size
    minp: int64_t
        minimum number of observations to calculate
        variance of a window
    ddof: unsigned int
         the divisor used in variance calculations
         is the window size - ddof

    Returns
    -------
    output: float64_t[:]
        weighted variances of windows
    """

    cdef:
        float64_t t = 0, sum_w = 0, mean = 0, nobs = 0
        float64_t val, pre_val, w, pre_w
        Py_ssize_t i, n, win_n
        float64_t[:] output

    # 获取输入数组的长度
    n = len(values)
    # 获取权重数组的长度
    win_n = len(weights)
    # 初始化输出数组，用于存储加权方差的结果
    output = np.empty(n, dtype=np.float64)

    # 使用 nogil 来避免全局解释器锁，提高性能
    with nogil:

        # 遍历数据数组和权重数组的前 min(win_n, n) 项
        for i in range(min(win_n, n)):
            # 调用函数 add_weighted_var 计算加权方差的增量
            add_weighted_var(values[i], weights[i], &t,
                             &sum_w, &mean, &nobs)
            # 计算当前窗口的加权方差并存储在输出数组中
            output[i] = calc_weighted_var(t, sum_w, win_n,
                                          ddof, nobs, minp)

        # 继续遍历剩余的数据数组和权重数组的项
        for i in range(win_n, n):
            val = values[i]
            pre_val = values[i - win_n]

            w = weights[i % win_n]
            pre_w = weights[(i - win_n) % win_n]

            # 检查当前值和前一个值是否为有效数值
            if val == val:
                if pre_val == pre_val:
                    # 移除前一个值对加权方差的贡献
                    remove_weighted_var(pre_val, pre_w, &t,
                                        &sum_w, &mean, &nobs)

                # 添加当前值对加权方差的贡献
                add_weighted_var(val, w, &t, &sum_w, &mean, &nobs)

            elif pre_val == pre_val:
                # 当前值无效时，仅移除前一个值的贡献
                remove_weighted_var(pre_val, pre_w, &t,
                                    &sum_w, &mean, &nobs)

            # 计算当前窗口的加权方差并存储在输出数组中
            output[i] = calc_weighted_var(t, sum_w, win_n,
                                          ddof, nobs, minp)

    # 返回最终的加权方差结果数组
    return output


# ----------------------------------------------------------------------
# Exponentially weighted moving
@cython.cpow(True)
# 定义一个函数用于计算指数加权移动平均或总和，根据给定的参数
def ewm(const float64_t[:] vals, const int64_t[:] start, const int64_t[:] end,
        int minp, float64_t com, bint adjust, bint ignore_na,
        const float64_t[:] deltas=None, bint normalize=True) -> np.ndarray:
    """
    Compute exponentially-weighted moving average or sum using center-of-mass.

    Parameters
    ----------
    vals : ndarray (float64 type)
        数值数组，用于计算加权移动平均或总和
    start: ndarray (int64 type)
        起始点数组，用于定义移动窗口的起始位置
    end: ndarray (int64 type)
        结束点数组，用于定义移动窗口的结束位置
    minp : int
        最小观测数，用于计算窗口的加权移动平均或总和
    com : float64
        加权因子，用于指数加权平均的中心位置
    adjust : bool
        是否调整加权平均的偏差
    ignore_na : bool
        是否忽略缺失值
    deltas : ndarray (float64 type), optional
        时间间隔数组，如果为None，则假设数据点间隔相等
    normalize : bool, optional
        如果为True，则计算加权平均，如果为False，则计算加权总和

    Returns
    -------
    np.ndarray[float64_t]
        计算结果数组，包含加权移动平均或总和的结果
    """
    # 定义多个 C 样式变量和常量，包括索引 i, j, 循环起止 s, e, 数据长度 nobs, 窗口大小 win_size, 以及数组长度 N 和 M
    cdef:
        Py_ssize_t i, j, s, e, nobs, win_size, N = len(vals), M = len(start)
        const float64_t[:] sub_vals  # 声明一个常量浮点型数组 sub_vals
        const float64_t[:] sub_deltas=None  # 声明一个常量浮点型数组 sub_deltas，默认为 None
        ndarray[float64_t] sub_output, output = np.empty(N, dtype=np.float64)  # 创建浮点型数组 sub_output 和 output，长度为 N
        float64_t alpha, old_wt_factor, new_wt, weighted, old_wt, cur  # 声明多个浮点型变量
        bint is_observation, use_deltas  # 声明布尔型变量 is_observation 和 use_deltas

    # 如果数据长度 N 为 0，则直接返回空的 output 数组
    if N == 0:
        return output

    # 检查是否使用 deltas，即检查 deltas 是否为 None
    use_deltas = deltas is not None

    # 计算 alpha 和 old_wt_factor 的值
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha

    # 如果 adjust 为 True，则 new_wt 为 1；否则 new_wt 为 alpha
    new_wt = 1. if adjust else alpha

    # 遍历 start 数组的索引
    for j in range(M):
        s = start[j]  # 获取起始位置 s
        e = end[j]    # 获取结束位置 e
        sub_vals = vals[s:e]  # 获取 vals 数组中 s 到 e 之间的子数组 sub_vals

        # 如果使用 deltas，则获取 deltas 数组中 s 到 e-1 之间的子数组 sub_deltas
        if use_deltas:
            sub_deltas = deltas[s:e - 1]

        win_size = len(sub_vals)  # 计算子数组 sub_vals 的长度，即窗口大小
        sub_output = np.empty(win_size, dtype=np.float64)  # 创建与子数组 sub_vals 同样大小的空浮点型数组 sub_output

        weighted = sub_vals[0]  # 初始化加权值为子数组 sub_vals 的第一个元素
        is_observation = weighted == weighted  # 检查第一个元素是否为有效观测值
        nobs = int(is_observation)  # 如果是观测值，则 nobs 初始化为 1；否则为 0
        sub_output[0] = weighted if nobs >= minp else NaN  # 根据观测值数量决定是否将第一个元素赋值给 sub_output[0]，不足 minp 则赋值 NaN
        old_wt = 1.  # 初始化旧权重为 1

        # 使用 nogil 语句块以解除全局锁定
        with nogil:
            # 遍历子数组 sub_vals 中的元素，从第二个元素开始
            for i in range(1, win_size):
                cur = sub_vals[i]  # 获取当前元素值
                is_observation = cur == cur  # 检查当前元素是否为有效观测值
                nobs += is_observation  # 增加观测值计数器

                # 如果加权值为有效值
                if weighted == weighted:
                    # 如果当前元素为观测值或者不忽略缺失值
                    if is_observation or not ignore_na:
                        # 如果需要归一化处理
                        if normalize:
                            # 如果使用 deltas，则根据 sub_deltas[i-1] 更新旧权重值
                            if use_deltas:
                                old_wt *= old_wt_factor ** sub_deltas[i - 1]
                            else:
                                old_wt *= old_wt_factor
                        else:
                            weighted = old_wt_factor * weighted  # 更新加权值

                        # 如果当前元素为观测值
                        if is_observation:
                            # 如果需要归一化处理
                            if normalize:
                                # 避免在常数序列上的数值误差
                                if weighted != cur:
                                    weighted = old_wt * weighted + new_wt * cur
                                    weighted /= (old_wt + new_wt)
                                if adjust:
                                    old_wt += new_wt
                                else:
                                    old_wt = 1.
                            else:
                                weighted += cur  # 累加当前元素值到加权值上
                elif is_observation:
                    weighted = cur  # 如果加权值不为有效值且当前元素为观测值，则将当前元素作为加权值

                # 根据观测值数量决定是否将加权值赋值给 sub_output[i]，不足 minp 则赋值 NaN
                sub_output[i] = weighted if nobs >= minp else NaN

        output[s:e] = sub_output  # 将子数组 sub_output 复制回 output 中相应的位置

    return output  # 返回处理完成的 output 数组
# 定义一个函数，计算使用指数加权移动方差（EWMA）来估算方差，使用质心方法。
def ewmcov(const float64_t[:] input_x, const int64_t[:] start, const int64_t[:] end,
           int minp, const float64_t[:] input_y, float64_t com, bint adjust,
           bint ignore_na, bint bias) -> np.ndarray:
    """
    Compute exponentially-weighted moving variance using center-of-mass.

    Parameters
    ----------
    input_x : ndarray (float64 type)
        输入的 x 数组，数据类型为 float64
    start: ndarray (int64 type)
        每个观察窗口的起始索引，数据类型为 int64
    end: ndarray (int64 type)
        每个观察窗口的结束索引，数据类型为 int64
    minp : int
        最小观察期限
    input_y : ndarray (float64 type)
        输入的 y 数组，数据类型为 float64
    com : float64
        质心参数，控制加权的速度
    adjust : bool
        是否调整加权权重
    ignore_na : bool
        是否忽略缺失值
    bias : bool
        是否使用偏差校正

    Returns
    -------
    np.ndarray[float64_t]
        返回一个 float64 类型的 NumPy 数组，存储计算得到的方差值
    """

    cdef:
        Py_ssize_t i, j, s, e, win_size, nobs
        Py_ssize_t N = len(input_x), M = len(input_y), L = len(start)
        float64_t alpha, old_wt_factor, new_wt, mean_x, mean_y, cov
        float64_t sum_wt, sum_wt2, old_wt, cur_x, cur_y, old_mean_x, old_mean_y
        float64_t numerator, denominator
        const float64_t[:] sub_x_vals, sub_y_vals
        ndarray[float64_t] sub_out, output = np.empty(N, dtype=np.float64)
        bint is_observation

    # 检查输入的 x 和 y 数组长度是否相同
    if M != N:
        raise ValueError(f"arrays are of different lengths ({N} and {M})")

    # 如果输入的数组长度为 0，则直接返回空的 output 数组
    if N == 0:
        return output

    # 计算 EWMA 相关的权重参数
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha

    # 返回计算得到的方差数组
    return output
```