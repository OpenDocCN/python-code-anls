# `D:\src\scipysrc\pandas\pandas\core\_numba\kernels\mean_.py`

```
"""
Numba 1D mean kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from pandas.core._numba.kernels.shared import is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum

if TYPE_CHECKING:
    from pandas._typing import npt


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_mean(
    val: float,
    nobs: int,
    sum_x: float,
    neg_ct: int,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, float, int, float, int, float]:
    """
    计算加权均值并更新相关统计量

    Args:
        val: 当前值
        nobs: 观测次数
        sum_x: 总和
        neg_ct: 负值计数
        compensation: 补偿项
        num_consecutive_same_value: 连续相同值计数
        prev_value: 上一个值

    Returns:
        更新后的观测次数、总和、负值计数、补偿项、连续相同值计数、上一个值
    """
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct += 1

        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    return nobs, sum_x, neg_ct, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_mean(
    val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float
) -> tuple[int, float, int, float]:
    """
    计算去除当前值后的均值并更新相关统计量

    Args:
        val: 当前值
        nobs: 观测次数
        sum_x: 总和
        neg_ct: 负值计数
        compensation: 补偿项

    Returns:
        更新后的观测次数、总和、负值计数、补偿项
    """
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct -= 1
    return nobs, sum_x, neg_ct, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_mean(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    """
    计算滑动均值

    Args:
        values: 输入值数组
        result_dtype: 结果数据类型
        start: 起始位置数组
        end: 结束位置数组
        min_periods: 最小周期数

    Returns:
        包含滑动均值的数组和有效值数列表
    """
    N = len(start)
    nobs = 0
    sum_x = 0.0
    neg_ct = 0
    compensation_add = 0.0
    compensation_remove = 0.0

    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    output = np.empty(N, dtype=result_dtype)
    # 遍历范围为 N 的索引
    for i in range(N):
        # 获取当前段的起始和结束索引
        s = start[i]
        e = end[i]
        
        # 如果是第一个段或不要求单调递增边界
        if i == 0 or not is_monotonic_increasing_bounds:
            # 初始化前一个值和连续相同值的计数器
            prev_value = values[s]
            num_consecutive_same_value = 0

            # 遍历当前段的所有值
            for j in range(s, e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # 忽略 pyright 的参数类型检查报告
                )
        else:
            # 如果需要单调递增边界，处理从前一个段到当前段的值
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, neg_ct, compensation_remove = remove_mean(
                    val, nobs, sum_x, neg_ct, compensation_remove
                )

            # 处理当前段的值
            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # 忽略 pyright 的参数类型检查报告
                )

        # 如果观测值大于等于最小期数且大于零
        if nobs >= min_periods and nobs > 0:
            # 计算平均值
            result = sum_x / nobs
            # 如果连续相同值大于等于观测数目，返回前一个值
            if num_consecutive_same_value >= nobs:
                result = prev_value
            # 如果负值计数为零且结果小于零，返回零
            elif neg_ct == 0 and result < 0:
                result = 0
            # 如果负值计数等于观测数目且结果大于零，返回零
            elif neg_ct == nobs and result > 0:
                result = 0
        else:
            # 否则，结果为 NaN
            result = np.nan

        # 将结果放入输出数组中
        output[i] = result

        # 如果不要求单调递增边界，重置相关变量
        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0.0
            neg_ct = 0
            compensation_remove = 0.0

    # na_position 是一个空列表，因为 float64 已经可以容纳 NaN
    # 使用列表推导生成空的 na_pos 列表，因为 numba 无法自动推断 na_pos 是空的整数列表
    na_pos = [0 for i in range(0)]
    # 返回输出数组和空的 na_pos 列表
    return output, na_pos
# 使用 numba.jit 装饰器对函数进行加速优化，禁止 Python 对象使用，允许并行执行
@numba.jit(nopython=True, nogil=True, parallel=False)
# 定义函数 grouped_mean，接受以下参数，并返回一个元组
def grouped_mean(
    values: np.ndarray,               # 输入数据数组
    result_dtype: np.dtype,           # 结果数据类型
    labels: npt.NDArray[np.intp],     # 标签数组
    ngroups: int,                     # 组数
    min_periods: int,                 # 最小周期数
) -> tuple[np.ndarray, list[int]]:   # 返回类型为包含 numpy 数组和整数列表的元组

    # 调用 grouped_kahan_sum 函数计算得到多个返回值
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups
    )

    # 后处理阶段，替换不满足最小周期数的求和结果
    for lab in range(ngroups):         # 遍历每个组
        nobs = nobs_arr[lab]           # 获取当前组的观测数
        num_consecutive_same_value = consecutive_counts[lab]  # 当前组连续相同值的数量
        prev_value = prev_vals[lab]    # 当前组的前一个值
        sum_x = output[lab]            # 当前组的总和
        if nobs >= min_periods:        # 如果观测数大于等于最小周期数
            if num_consecutive_same_value >= nobs:  # 如果连续相同值的数量大于等于观测数
                result = prev_value * nobs  # 结果为前一个值乘以观测数
            else:
                result = sum_x            # 否则结果为总和
        else:
            result = np.nan              # 如果观测数小于最小周期数，结果为 NaN
        result /= nobs                   # 结果除以观测数
        output[lab] = result             # 更新输出数组中的当前组结果

    # na_position 是一个空列表，因为 float64 已经可以容纳 NaN
    # 使用列表推导式初始化 na_pos，因为 numba 不能自动推断 na_pos 是一个空的 int 列表
    na_pos = [0 for i in range(0)]

    # 返回计算得到的输出数组和空的整数列表 na_pos
    return output, na_pos
```