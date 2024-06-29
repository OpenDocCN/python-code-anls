# `D:\src\scipysrc\pandas\pandas\core\_numba\kernels\sum_.py`

```
"""
Numba 1D sum kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# 导入必要的库和模块
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numba
from numba.extending import register_jitable
import numpy as np

# 如果是类型检查阶段，导入特定类型
if TYPE_CHECKING:
    from pandas._typing import npt

# 从pandas核心模块导入函数
from pandas.core._numba.kernels.shared import is_monotonic_increasing


# jit装饰器定义了一个使用Numba优化的函数，无Python对象，无全局解锁，不并行执行
@numba.jit(nopython=True, nogil=True, parallel=False)
def add_sum(
    val: Any,
    nobs: int,
    sum_x: Any,
    compensation: Any,
    num_consecutive_same_value: int,
    prev_value: Any,
) -> tuple[int, Any, Any, int, Any]:
    # 如果值不是NaN
    if not np.isnan(val):
        # 观测数加一
        nobs += 1
        # 计算新的累加和
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t

        # 如果当前值等于前一个值，则连续相同值个数加一，否则重置为1
        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    # 返回更新后的变量值
    return nobs, sum_x, compensation, num_consecutive_same_value, prev_value


# jit装饰器定义了一个使用Numba优化的函数，无Python对象，无全局解锁，不并行执行
@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(
    val: Any, nobs: int, sum_x: Any, compensation: Any
) -> tuple[int, Any, Any]:
    # 如果值不是NaN
    if not np.isnan(val):
        # 观测数减一
        nobs -= 1
        # 计算新的累加和
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
    # 返回更新后的变量值
    return nobs, sum_x, compensation


# jit装饰器定义了一个使用Numba优化的函数，无Python对象，无全局解锁，不并行执行
@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    # 确定数据类型
    dtype = values.dtype

    # 如果数据类型是整数，缺失值设为0
    na_val: object = np.nan
    if dtype.kind == "i":
        na_val = 0

    # 获取数据长度
    N = len(start)
    # 初始化变量
    nobs = 0
    sum_x = 0
    compensation_add = 0
    compensation_remove = 0
    na_pos = []

    # 判断起始和结束数组是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    # 创建空的输出数组，使用指定的结果数据类型
    output = np.empty(N, dtype=result_dtype)
    # 遍历从 0 到 N-1 的整数序列，每个 i 对应一个区间范围
    for i in range(N):
        # 获取当前区间的起始索引和结束索引
        s = start[i]
        e = end[i]
        
        # 如果当前是第一个区间或者不要求单调递增边界条件
        if i == 0 or not is_monotonic_increasing_bounds:
            # 初始化前一个值为当前区间起始位置的值
            prev_value = values[s]
            # 记录连续相同值的个数
            num_consecutive_same_value = 0

            # 遍历当前区间的所有元素
            for j in range(s, e):
                # 获取当前位置的值
                val = values[j]
                # 调用 add_sum 函数，更新统计信息并返回更新后的值
                (
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            # 如果需要单调递增边界条件，则处理前一个区间和当前区间之间的元素
            for j in range(start[i - 1], s):
                # 获取前一个区间末尾到当前区间起始之间的值
                val = values[j]
                # 调用 remove_sum 函数，更新统计信息并返回更新后的值
                nobs, sum_x, compensation_remove = remove_sum(
                    val, nobs, sum_x, compensation_remove
                )

            # 处理前一个区间末尾到当前区间末尾之间的元素
            for j in range(end[i - 1], e):
                # 获取前一个区间末尾到当前区间末尾之间的值
                val = values[j]
                # 调用 add_sum 函数，更新统计信息并返回更新后的值
                (
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        # 根据统计信息计算当前区间的结果值
        if nobs == 0 == min_periods:
            result: object = 0
        elif nobs >= min_periods:
            # 如果连续相同值的个数大于等于观测数，结果为前一个值乘以观测数
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                # 否则结果为求和值
                result = sum_x
        else:
            # 如果观测数小于最小观测数，结果为缺失值
            result = na_val
            # 如果数据类型为整数，将当前区间索引添加到缺失值位置列表
            if dtype.kind == "i":
                na_pos.append(i)

        # 将计算得到的结果值存储到输出数组中对应的位置
        output[i] = result

        # 如果不需要单调递增边界条件，重置统计信息
        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0
            compensation_remove = 0

    # 返回计算结果数组和缺失值位置列表
    return output, na_pos
# Mypy/pyright 不喜欢装饰器未标记类型
@register_jitable  # type: ignore[misc]
# 定义一个函数 grouped_kahan_sum，用于对输入的数组进行分组的 Kahan 和求和计算
def grouped_kahan_sum(
    values: np.ndarray,  # 输入的值数组
    result_dtype: np.dtype,  # 输出结果的数据类型
    labels: npt.NDArray[np.intp],  # 标签数组，用于分组
    ngroups: int,  # 分组数目
) -> tuple[  # 返回一个包含多个 numpy 数组的元组
    np.ndarray, npt.NDArray[np.int64], np.ndarray, npt.NDArray[np.int64], np.ndarray
]:
    N = len(labels)  # 获取标签数组的长度

    # 初始化用于存储结果的数组
    nobs_arr = np.zeros(ngroups, dtype=np.int64)  # 存储每个分组的观测数
    comp_arr = np.zeros(ngroups, dtype=values.dtype)  # 存储每个分组的补偿值
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)  # 存储每个分组中连续相同值的计数
    prev_vals = np.zeros(ngroups, dtype=values.dtype)  # 存储每个分组中前一个值
    output = np.zeros(ngroups, dtype=result_dtype)  # 存储每个分组的最终结果

    # 遍历标签数组进行计算
    for i in range(N):
        lab = labels[i]  # 获取当前索引对应的标签值
        val = values[i]  # 获取当前索引对应的数值

        if lab < 0:  # 如果标签值小于0，则跳过当前循环
            continue

        # 从数组中获取当前分组的各个状态值
        sum_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]

        # 调用 add_sum 函数更新分组状态值
        (
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        ) = add_sum(
            val,
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        )

        # 更新数组中当前分组的状态值
        output[lab] = sum_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs

    # 返回计算结果的数组和状态数组
    return output, nobs_arr, comp_arr, consecutive_counts, prev_vals


# 使用 numba 库进行加速的装饰器函数定义
@numba.jit(nopython=True, nogil=True, parallel=False)
# 定义一个函数 grouped_sum，用于计算分组的求和，并根据 min_periods 进行后处理
def grouped_sum(
    values: np.ndarray,  # 输入的值数组
    result_dtype: np.dtype,  # 输出结果的数据类型
    labels: npt.NDArray[np.intp],  # 标签数组，用于分组
    ngroups: int,  # 分组数目
    min_periods: int,  # 最小观测期数
) -> tuple[np.ndarray, list[int]]:  # 返回一个包含结果数组和无效位置列表的元组
    na_pos = []  # 初始化一个空列表，用于存储无效位置

    # 调用 grouped_kahan_sum 函数计算初始的分组和状态
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups
    )

    # 后处理阶段，替换不满足 min_periods 的求和结果
    for lab in range(ngroups):
        nobs = nobs_arr[lab]  # 获取当前分组的观测数
        num_consecutive_same_value = consecutive_counts[lab]  # 获取当前分组连续相同值的计数
        prev_value = prev_vals[lab]  # 获取当前分组的前一个值
        sum_x = output[lab]  # 获取当前分组的求和结果

        # 如果观测数达到 min_periods
        if nobs >= min_periods:
            # 如果连续相同值的计数达到观测数
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs  # 结果为前一个值乘以观测数
            else:
                result = sum_x  # 否则结果为当前的求和结果
        else:
            result = sum_x  # 如果观测数不足 min_periods，则结果为当前的求和结果（稍后将被替换为 NaN）

            # 将当前分组的位置添加到无效位置列表中
            na_pos.append(lab)

        # 更新输出数组中当前分组的结果
        output[lab] = result

    # 返回最终处理后的结果数组和无效位置列表
    return output, na_pos
```