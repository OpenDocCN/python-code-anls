# `D:\src\scipysrc\pandas\pandas\core\_numba\kernels\min_max_.py`

```
"""
Numba 1D min/max kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""

# 导入必要的模块和类型声明
from typing import TYPE_CHECKING
import numba
import numpy as np

# 如果是类型检查，导入额外的类型声明
if TYPE_CHECKING:
    from pandas._typing import npt

# 定义一个使用 Numba 编译的函数，用于计算滑动窗口的最小值和最大值
@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    # 获取滑动窗口的数量
    N = len(start)
    # 初始化观测数量和输出数组
    nobs = 0
    output = np.empty(N, dtype=result_dtype)
    na_pos = []
    # 使用列表 Q 和 W 来模拟 deque，存储当前窗口的索引
    Q: list = []
    W: list = []

    # 遍历每一个窗口
    for i in range(N):
        # 计算当前窗口的大小
        curr_win_size = end[i] - start[i]
        # 确定起始索引 st
        if i == 0:
            st = start[i]
        else:
            st = end[i - 1]

        # 遍历当前窗口内的每一个元素
        for k in range(st, end[i]):
            ai = values[k]
            # 如果元素不是 NaN，则增加观测数量
            if not np.isnan(ai):
                nobs += 1
            # 处理最大值或最小值的情况
            elif is_max:
                ai = -np.inf
            else:
                ai = np.inf

            # 如果是最大值，移除队列 Q 中所有不符合条件的元素
            if is_max:
                while Q and ((ai >= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            # 如果是最小值，移除队列 Q 中所有不符合条件的元素
            else:
                while Q and ((ai <= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            
            # 将当前索引 k 添加到队列 Q 和 W 中
            Q.append(k)
            W.append(k)

        # 移除队列 Q 和 W 中在当前窗口左侧超出范围的元素
        while Q and Q[0] <= start[i] - 1:
            Q.pop(0)
        while W and W[0] <= start[i] - 1:
            if not np.isnan(values[W[0]]):
                nobs -= 1
            W.pop(0)

        # 根据 Q 中的索引保存输出结果
        if Q and curr_win_size > 0 and nobs >= min_periods:
            output[i] = values[Q[0]]
        else:
            # 如果值的数据类型不是整数，则输出 NaN
            if values.dtype.kind != "i":
                output[i] = np.nan
            else:
                na_pos.append(i)

    # 返回最终的输出数组和 NaN 位置的列表
    return output, na_pos


# 定义一个使用 Numba 编译的函数，用于分组计算最小值和最大值
@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    # 获取分组的数量
    N = len(labels)
    # 初始化每个分组的观测数量、NaN 位置列表和输出数组
    nobs = np.zeros(ngroups, dtype=np.int64)
    na_pos = []
    output = np.empty(ngroups, dtype=result_dtype)
    # 遍历范围为 N 的循环，依次处理每个索引 i
    for i in range(N):
        # 获取当前索引 i 对应的标签
        lab = labels[i]
        # 获取当前索引 i 对应的数值
        val = values[i]
        
        # 如果标签 lab 小于 0，则跳过当前循环，继续下一个索引
        if lab < 0:
            continue
        
        # 检查数值数组 values 的数据类型是否为整数，或者当前值 val 不是 NaN
        if values.dtype.kind == "i" or not np.isnan(val):
            # 如果符合条件，增加标签 lab 对应的观察次数
            nobs[lab] += 1
        else:
            # 如果当前值 val 是 NaN，则跳过当前循环，继续下一个索引
            continue
        
        # 如果 nobs[lab] 等于 1，表示该组的第一个元素，将输出数组中对应位置设置为当前值 val
        if nobs[lab] == 1:
            output[lab] = val
            continue
        
        # 如果需要计算最大值
        if is_max:
            # 如果当前值 val 大于输出数组中对应位置的值，则更新输出数组中的值为当前值 val
            if val > output[lab]:
                output[lab] = val
        else:
            # 如果当前值 val 小于输出数组中对应位置的值，则更新输出数组中的值为当前值 val
            if val < output[lab]:
                output[lab] = val

    # 设置不满足最小周期 min_periods 的标签为 np.nan
    for lab, count in enumerate(nobs):
        # 如果标签对应的观察次数 count 小于 min_periods，则将该标签加入 na_pos 列表
        if count < min_periods:
            na_pos.append(lab)

    # 返回计算得到的输出数组和不满足条件的标签列表 na_pos
    return output, na_pos
```