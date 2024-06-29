# `D:\src\scipysrc\pandas\pandas\core\_numba\kernels\var_.py`

```
"""
Numba 1D var kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# 从未来导入类型检查相关模块
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 numba 库
import numba
# 导入 numpy 库并使用别名 np
import numpy as np

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从 pandas._typing 模块导入 npt 类型
    from pandas._typing import npt

# 从 pandas.core._numba.kernels.shared 模块导入 is_monotonic_increasing 函数
from pandas.core._numba.kernels.shared import is_monotonic_increasing


# 使用 numba.jit 装饰器定义函数 add_var
@numba.jit(nopython=True, nogil=True, parallel=False)
def add_var(
    val: float,
    nobs: int,
    mean_x: float,
    ssqdm_x: float,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, float, float, float, int, float]:
    # 如果 val 不是 NaN
    if not np.isnan(val):
        # 如果当前值 val 等于上一个值 prev_value
        if val == prev_value:
            # 连续相同值计数加一
            num_consecutive_same_value += 1
        else:
            # 否则重置连续相同值计数为一
            num_consecutive_same_value = 1
        # 更新 prev_value 为当前值 val
        prev_value = val

        # 增加观测次数 nobs
        nobs += 1
        # 计算上一个均值 prev_mean
        prev_mean = mean_x - compensation
        # 计算 y 和 t
        y = val - compensation
        t = y - mean_x
        # 更新补偿 compensation
        compensation = t + mean_x - y
        # 计算 delta
        delta = t
        # 如果观测次数 nobs 不为零
        if nobs:
            # 更新均值 mean_x
            mean_x += delta / nobs
        else:
            # 否则将均值 mean_x 置为零
            mean_x = 0
        # 更新方差 ssqdm_x
        ssqdm_x += (val - prev_mean) * (val - mean_x)
    # 返回更新后的变量作为元组
    return nobs, mean_x, ssqdm_x, compensation, num_consecutive_same_value, prev_value


# 使用 numba.jit 装饰器定义函数 remove_var
@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_var(
    val: float, nobs: int, mean_x: float, ssqdm_x: float, compensation: float
) -> tuple[int, float, float, float]:
    # 如果 val 不是 NaN
    if not np.isnan(val):
        # 减少观测次数 nobs
        nobs -= 1
        # 如果观测次数 nobs 不为零
        if nobs:
            # 计算上一个均值 prev_mean
            prev_mean = mean_x - compensation
            # 计算 y 和 t
            y = val - compensation
            t = y - mean_x
            # 更新补偿 compensation
            compensation = t + mean_x - y
            # 计算 delta
            delta = t
            # 更新均值 mean_x 和方差 ssqdm_x
            mean_x -= delta / nobs
            ssqdm_x -= (val - prev_mean) * (val - mean_x)
        else:
            # 否则将均值 mean_x 和方差 ssqdm_x 置为零
            mean_x = 0
            ssqdm_x = 0
    # 返回更新后的变量作为元组
    return nobs, mean_x, ssqdm_x, compensation


# 使用 numba.jit 装饰器定义函数 sliding_var
@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_var(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    ddof: int = 1,
) -> tuple[np.ndarray, list[int]]:
    # 获取数组长度 N
    N = len(start)
    # 初始化变量
    nobs = 0
    mean_x = 0.0
    ssqdm_x = 0.0
    compensation_add = 0.0
    compensation_remove = 0.0

    # 最小观测期数不得小于 1
    min_periods = max(min_periods, 1)
    # 检查起始和结束数组是否单调递增
    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    # 创建一个与输入长度相同的空数组 output
    output = np.empty(N, dtype=result_dtype)
    # 对于每一个区间i，从start[i]到end[i]遍历
    for i in range(N):
        # 获取当前区间的起始和结束索引
        s = start[i]
        e = end[i]
        # 如果是第一个区间或者不要求单调递增边界
        if i == 0 or not is_monotonic_increasing_bounds:
            # 初始化前一个值为当前区间起始处的值
            prev_value = values[s]
            # 初始化连续相同值的计数器为0
            num_consecutive_same_value = 0

            # 遍历当前区间内的值
            for j in range(s, e):
                # 获取当前值
                val = values[j]
                # 调用函数add_var，更新统计信息
                (
                    nobs,              # 观测数目
                    mean_x,            # 均值
                    ssqdm_x,           # 平方和
                    compensation_add,  # 补偿增加
                    num_consecutive_same_value,  # 连续相同值的计数
                    prev_value,        # 前一个值
                ) = add_var(
                    val,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            # 如果要求单调递增边界，则先移除前一个区间的值
            for j in range(start[i - 1], s):
                val = values[j]
                # 调用函数remove_var，更新统计信息
                nobs, mean_x, ssqdm_x, compensation_remove = remove_var(
                    val, nobs, mean_x, ssqdm_x, compensation_remove
                )

            # 添加当前区间的值
            for j in range(end[i - 1], e):
                val = values[j]
                # 调用函数add_var，更新统计信息
                (
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_var(
                    val,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        # 如果观测数目达到要求的最小阈值并且大于自由度调整数目
        if nobs >= min_periods and nobs > ddof:
            # 如果只有一个观测或者所有观测值都相同
            if nobs == 1 or num_consecutive_same_value >= nobs:
                result = 0.0  # 结果为0
            else:
                result = ssqdm_x / (nobs - ddof)  # 计算方差
        else:
            result = np.nan  # 否则结果为NaN

        output[i] = result  # 将计算结果存入输出数组

        # 如果不要求单调递增边界，则重置统计变量
        if not is_monotonic_increasing_bounds:
            nobs = 0
            mean_x = 0.0
            ssqdm_x = 0.0
            compensation_remove = 0.0

    # na_position 是空列表，因为float64类型已经可以存储NaN值
    # 使用列表推导式创建空的na_pos列表，因为numba不能自动推断出na_pos是空的整数列表
    na_pos = [0 for i in range(0)]
    return output, na_pos
# 使用 numba.jit 进行装饰器，以提高函数性能，禁用 Python 对象、释放全局解释锁，不支持并行
@numba.jit(nopython=True, nogil=True, parallel=False)
# 计算分组变量的函数
def grouped_var(
    values: np.ndarray,                              # 输入值数组
    result_dtype: np.dtype,                          # 结果数据类型
    labels: npt.NDArray[np.intp],                    # 标签数组
    ngroups: int,                                    # 分组数量
    min_periods: int,                                # 最小周期数
    ddof: int = 1,                                   # 自由度差值
) -> tuple[np.ndarray, list[int]]:                   # 返回值为元组（结果数组，空列表）

    N = len(labels)                                  # 标签数组的长度

    nobs_arr = np.zeros(ngroups, dtype=np.int64)     # 初始化观测次数数组
    comp_arr = np.zeros(ngroups, dtype=values.dtype) # 初始化补偿数组
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)  # 初始化连续计数数组
    prev_vals = np.zeros(ngroups, dtype=values.dtype)         # 初始化前一个值数组
    output = np.zeros(ngroups, dtype=result_dtype)            # 初始化输出数组
    means = np.zeros(ngroups, dtype=result_dtype)             # 初始化均值数组

    for i in range(N):                               # 遍历标签数组
        lab = labels[i]                              # 获取当前标签
        val = values[i]                              # 获取当前值

        if lab < 0:                                  # 如果标签小于0，跳过当前循环
            continue

        mean_x = means[lab]                          # 获取当前分组的均值
        ssqdm_x = output[lab]                        # 获取当前分组的平方差
        nobs = nobs_arr[lab]                         # 获取当前分组的观测次数
        compensation_add = comp_arr[lab]             # 获取当前分组的补偿值
        num_consecutive_same_value = consecutive_counts[lab]  # 获取当前分组的连续相同值计数
        prev_value = prev_vals[lab]                  # 获取当前分组的前一个值

        # 调用 add_var 函数，更新当前分组的统计量
        (
            nobs,
            mean_x,
            ssqdm_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        ) = add_var(
            val,
            nobs,
            mean_x,
            ssqdm_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        )

        # 更新当前分组的统计量数组
        output[lab] = ssqdm_x
        means[lab] = mean_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs

    # 后处理，替换不满足最小周期数的变量
    for lab in range(ngroups):                       # 遍历所有分组
        nobs = nobs_arr[lab]                         # 获取当前分组的观测次数
        num_consecutive_same_value = consecutive_counts[lab]  # 获取当前分组的连续相同值计数
        ssqdm_x = output[lab]                        # 获取当前分组的平方差

        # 如果观测次数大于等于最小周期数且大于自由度差值
        if nobs >= min_periods and nobs > ddof:
            # 如果观测次数为1或连续相同值计数大于等于观测次数
            if nobs == 1 or num_consecutive_same_value >= nobs:
                result = 0.0                         # 结果为0.0
            else:
                result = ssqdm_x / (nobs - ddof)     # 计算结果为平方差除以自由度差值
        else:
            result = np.nan                          # 否则结果为NaN

        output[lab] = result                         # 更新输出数组中的结果值

    # 第二次传递以获取标准差
    # na_position 是一个空列表，因为 float64 已经能够容纳NaN
    # 使用列表推导，因为 numba 无法自动推断 na_pos 是空的整数列表
    na_pos = [0 for i in range(0)]                   # 初始化空的列表 na_pos
    return output, na_pos                            # 返回输出数组和空列表 na_pos
```