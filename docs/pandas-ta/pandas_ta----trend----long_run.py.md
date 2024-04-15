# `.\pandas-ta\pandas_ta\trend\long_run.py`

```
# -*- coding: utf-8 -*-
# 导入必要的模块和函数
from .decreasing import decreasing  # 从当前目录下的 decreasing 模块导入 decreasing 函数
from .increasing import increasing  # 从当前目录下的 increasing 模块导入 increasing 函数
from pandas_ta.utils import get_offset, verify_series  # 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数


def long_run(fast, slow, length=None, offset=None, **kwargs):
    """Indicator: Long Run"""
    # Validate Arguments
    # 将 length 转换为整数，如果 length 存在且大于 0；否则默认为 2
    length = int(length) if length and length > 0 else 2
    # 验证 fast 和 slow 是否为有效的序列，并将其长度限制为 length
    fast = verify_series(fast, length)
    slow = verify_series(slow, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 fast 或 slow 为空，则返回空值
    if fast is None or slow is None: return

    # Calculate Result
    # 计算可能的底部或底部的条件，即 fast 增长而 slow 减小
    pb = increasing(fast, length) & decreasing(slow, length)
    # 计算 fast 和 slow 同时增长的条件
    bi = increasing(fast, length) & increasing(slow, length)
    # 计算长期趋势的条件，可能的底部或底部，以及 fast 和 slow 同时增长的情况
    long_run = pb | bi

    # Offset
    # 如果 offset 不为 0，则对长期趋势进行偏移
    if offset != 0:
        long_run = long_run.shift(offset)

    # Handle fills
    # 处理填充值
    if "fillna" in kwargs:
        long_run.fillna(kwargs["fillna"], inplace=True)
    # 使用指定的填充方法填充缺失值
    if "fill_method" in kwargs:
        long_run.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    # 设置长期趋势指标的名称
    long_run.name = f"LR_{length}"
    # 设置长期趋势指标的类别为 "trend"
    long_run.category = "trend"

    return long_run
```