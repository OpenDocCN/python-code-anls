# `D:\src\scipysrc\pandas\pandas\core\window\common.py`

```
"""Common utility functions for rolling operations"""

# 导入必要的库和模块
from __future__ import annotations  # 使得类型提示可以引用自身的类型

from collections import defaultdict  # 导入 defaultdict 类
from typing import cast  # 导入 cast 类型转换函数

import numpy as np  # 导入 NumPy 库

from pandas.core.dtypes.generic import (  # 从 pandas 中导入数据类型相关模块
    ABCDataFrame,  # 导入 ABCDataFrame 类
    ABCSeries,  # 导入 ABCSeries 类
)

from pandas.core.indexes.api import MultiIndex  # 从 pandas 中导入 MultiIndex 类


def flex_binary_moment(arg1, arg2, f, pairwise: bool = False):
    # 如果 arg1 和 arg2 都是 ABCSeries 类型的实例
    if isinstance(arg1, ABCSeries) and isinstance(arg2, ABCSeries):
        # 调用 prep_binary 函数，准备参数 X 和 Y
        X, Y = prep_binary(arg1, arg2)
        # 调用函数 f 计算结果并返回
        return f(X, Y)
    else:
        # 如果 arg1 和 arg2 类型不匹配，递归调用函数，交换参数位置
        return flex_binary_moment(arg2, arg1, f)


def zsqrt(x):
    # 忽略 NumPy 的警告
    with np.errstate(all="ignore"):
        # 对 x 中的每个元素计算平方根
        result = np.sqrt(x)
        # 创建一个标记 x 中负值的布尔掩码
        mask = x < 0

    # 如果 x 是 ABCDataFrame 类型的实例
    if isinstance(x, ABCDataFrame):
        # 如果掩码中有任何 True 值
        if mask._values.any():
            # 将 result 中对应掩码为 True 的元素设为 0
            result[mask] = 0
    else:
        # 如果掩码中有任何 True 值
        if mask.any():
            # 将 result 中对应掩码为 True 的元素设为 0
            result[mask] = 0

    # 返回处理后的结果
    return result


def prep_binary(arg1, arg2):
    # 将 arg1 的值复制给 X，同时保持索引一致性
    X = arg1 + 0 * arg2
    # 将 arg2 的值复制给 Y，同时保持索引一致性
    Y = arg2 + 0 * arg1

    # 返回准备好的参数 X 和 Y
    return X, Y
```