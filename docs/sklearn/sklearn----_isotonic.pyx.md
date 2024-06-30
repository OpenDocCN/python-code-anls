# `D:\src\scipysrc\scikit-learn\sklearn\_isotonic.pyx`

```
# Author: Nelle Varoquaux, Andrew Tulloch, Antony Lee

# 使用池相邻违规算法（PAVA），在每一步中寻找最长的递减子序列来汇集数据。

import numpy as np
from cython cimport floating

# 定义一个函数，执行原地连续保序回归
def _inplace_contiguous_isotonic_regression(floating[::1] y, floating[::1] w):
    cdef:
        Py_ssize_t n = y.shape[0], i, k
        floating prev_y, sum_wy, sum_w
        Py_ssize_t[::1] target = np.arange(n, dtype=np.intp)

    # target 描述了一个块的列表。任何时候，如果 [i..j]（包括）是一个活跃块，
    # 则 target[i] := j，target[j] := i。

    # 对于“活跃”索引（块的起始点）：
    # w[i] := sum{w_orig[j], j=[i..target[i]]}
    # y[i] := sum{y_orig[j]*w_orig[j], j=[i..target[i]]} / w[i]

    with nogil:
        i = 0
        while i < n:
            k = target[i] + 1
            if k == n:
                break
            if y[i] < y[k]:
                i = k
                continue
            sum_wy = w[i] * y[i]
            sum_w = w[i]
            while True:
                # 我们处于一个递减子序列内。
                prev_y = y[k]
                sum_wy += w[k] * y[k]
                sum_w += w[k]
                k = target[k] + 1
                if k == n or prev_y < y[k]:
                    # 非单例递减子序列已完成，更新第一个条目。
                    y[i] = sum_wy / sum_w
                    w[i] = sum_w
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        # 如果可以回溯，则回溯。这使算法单遍并确保 O(n) 复杂度。
                        i = target[i - 1]
                    # 否则，从相同点重新开始。
                    break
        # 重建解决方案。
        i = 0
        while i < n:
            k = target[i] + 1
            y[i + 1 : k] = y[i]
            i = k


# 定义一个函数，处理重复的 X 值，计算平均 y 值并且去除重复项。
def _make_unique(const floating[::1] X,
                 const floating[::1] y,
                 const floating[::1] sample_weights):
    """Average targets for duplicate X, drop duplicates.

    将重复的 X 值聚合为单个 X 值，其中目标 y 是个体目标的（加权）平均值。

    假设 X 是有序的，所以所有重复项都紧随其后。
    """
    unique_values = len(np.unique(X))

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef floating[::1] y_out = np.empty(unique_values, dtype=dtype)
    cdef floating[::1] x_out = np.empty_like(y_out)
    cdef floating[::1] weights_out = np.empty_like(y_out)

    cdef floating current_x = X[0]
    cdef floating current_y = 0
    cdef floating current_weight = 0
    cdef int i = 0
    cdef int j
    cdef floating x
    cdef int n_samples = len(X)
    cdef floating eps = np.finfo(dtype).resolution
    # 遍历样本数量范围
    for j in range(n_samples):
        # 获取当前样本值
        x = X[j]
        # 如果当前样本值减去当前 x 值大于等于 eps
        if x - current_x >= eps:
            # 将当前 x 值存入输出数组
            x_out[i] = current_x
            # 将当前权重存入输出数组
            weights_out[i] = current_weight
            # 计算并存储当前 y 值的加权平均
            y_out[i] = current_y / current_weight
            # 更新索引 i
            i += 1
            # 更新当前 x 值为当前样本值
            current_x = x
            # 更新当前权重为当前样本权重
            current_weight = sample_weights[j]
            # 更新当前 y 值为当前样本 y 乘以权重
            current_y = y[j] * sample_weights[j]
        else:
            # 如果样本值未达到 eps 要求，则累加样本权重
            current_weight += sample_weights[j]
            # 累加当前样本 y 乘以权重
            current_y += y[j] * sample_weights[j]

    # 存储最后一个 x 值到输出数组
    x_out[i] = current_x
    # 存储最后一个权重到输出数组
    weights_out[i] = current_weight
    # 计算并存储最后一个 y 值的加权平均到输出数组
    y_out[i] = current_y / current_weight
    # 返回三个输出数组的 numpy 数组形式
    return(
        np.asarray(x_out[:i+1]),
        np.asarray(y_out[:i+1]),
        np.asarray(weights_out[:i+1]),
    )
```