# `.\numpy\numpy\typing\tests\data\pass\literal.py`

```
# 导入未来的注释支持，以便在类型提示中使用类型的字符串表示
from __future__ import annotations

# 导入必要的类型提示
from typing import Any
from functools import partial
from collections.abc import Callable

# 导入 pytest 和 numpy 库
import pytest
import numpy as np

# 创建一个不可变的 NumPy 数组 AR，标记为不可写
AR = np.array(0)
AR.setflags(write=False)

# 定义几个不可变的集合，用于限制函数的可选参数
KACF = frozenset({None, "K", "A", "C", "F"})
ACF = frozenset({None, "A", "C", "F"})
CF = frozenset({None, "C", "F"})

# order_list 列表包含了多个元组，每个元组包含一个集合和一个可调用对象的部分应用
order_list: list[tuple[frozenset[str | None], Callable[..., Any]]] = [
    (KACF, partial(np.ndarray, 1)),  # 使用 np.ndarray 创建一个一维数组的部分应用
    (KACF, AR.tobytes),              # 将数组 AR 转换为字节串的方法
    (KACF, partial(AR.astype, int)), # 将数组 AR 转换为整数类型的部分应用
    (KACF, AR.copy),                 # 复制数组 AR 的方法
    (ACF, partial(AR.reshape, 1)),   # 将数组 AR 重塑为一维数组的部分应用
    (KACF, AR.flatten),              # 将数组 AR 展平为一维数组的方法
    (KACF, AR.ravel),                # 将数组 AR 拉直为一维数组的方法
    (KACF, partial(np.array, 1)),    # 使用 np.array 创建一个一维数组的部分应用
    (CF, partial(np.zeros, 1)),      # 使用 np.zeros 创建一个全零数组的部分应用
    (CF, partial(np.ones, 1)),       # 使用 np.ones 创建一个全一数组的部分应用
    (CF, partial(np.empty, 1)),      # 使用 np.empty 创建一个空数组的部分应用
    (CF, partial(np.full, 1, 1)),    # 使用 np.full 创建一个填充值为 1 的数组的部分应用
    (KACF, partial(np.zeros_like, AR)),     # 使用 np.zeros_like 根据 AR 创建全零数组的部分应用
    (KACF, partial(np.ones_like, AR)),      # 使用 np.ones_like 根据 AR 创建全一数组的部分应用
    (KACF, partial(np.empty_like, AR)),     # 使用 np.empty_like 根据 AR 创建空数组的部分应用
    (KACF, partial(np.full_like, AR, 1)),   # 使用 np.full_like 根据 AR 创建填充值为 1 的数组的部分应用
    (KACF, partial(np.add, 1, 1)),          # 使用 np.add 创建加法函数的部分应用，即 np.ufunc.__call__
    (ACF, partial(np.reshape, AR, 1)),      # 使用 np.reshape 根据 AR 创建一维数组的部分应用
    (KACF, partial(np.ravel, AR)),          # 使用 np.ravel 将 AR 拉直为一维数组的部分应用
    (KACF, partial(np.asarray, 1)),         # 使用 np.asarray 创建一个数组的部分应用
    (KACF, partial(np.asanyarray, 1)),      # 使用 np.asanyarray 创建一个可作为数组的对象的部分应用
]

# 遍历 order_list 中的每个元组
for order_set, func in order_list:
    # 对于 order_set 中的每个 order，调用 func 函数
    for order in order_set:
        func(order=order)

    # 对于不在 order_set 中的每个 order，验证调用 func 会引发 ValueError 异常
    invalid_orders = KACF - order_set
    for order in invalid_orders:
        with pytest.raises(ValueError):
            func(order=order)
```