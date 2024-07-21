# `.\pytorch\torch\ao\quantization\experimental\apot_utils.py`

```py
r"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""

import math

r"""Converts floating point input into APoT number
    based on quantization levels
"""
def float_to_apot(x, levels, indices, alpha):
    # 如果 x 小于 -alpha，则返回 -alpha
    if x < -alpha:
        return -alpha
    # 如果 x 大于 alpha，则返回 alpha
    elif x > alpha:
        return alpha

    levels_lst = list(levels)
    indices_lst = list(indices)

    min_delta = math.inf
    best_idx = 0

    # 遍历 quantization levels 和对应的 indices
    for level, idx in zip(levels_lst, indices_lst):
        # 计算当前 level 和 x 之间的差值
        cur_delta = abs(level - x)
        # 如果当前差值比之前的最小差值小，则更新最小差值和对应的 index
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx

    # 返回最匹配的 index
    return best_idx

r"""Converts floating point input into
    reduced precision floating point value
    based on quantization levels
"""
def quant_dequant_util(x, levels, indices):
    levels_lst = list(levels)
    indices_lst = list(indices)

    min_delta = math.inf
    best_fp = 0.0

    # 遍历 quantization levels 和对应的 indices
    for level, idx in zip(levels_lst, indices_lst):
        # 计算当前 level 和 x 之间的差值
        cur_delta = abs(level - x)
        # 如果当前差值比之前的最小差值小，则更新最小差值和对应的 floating point 值
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_fp = level

    # 返回最匹配的 floating point 值
    return best_fp

r"""Converts APoT input into floating point number
based on quantization levels
"""
def apot_to_float(x_apot, levels, indices):
    # 找到 x_apot 在 indices 中的索引，并返回对应的 quantization level
    idx = list(indices).index(x_apot)
    return levels[idx]
```