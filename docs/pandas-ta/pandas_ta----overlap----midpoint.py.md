# `.\pandas-ta\pandas_ta\overlap\midpoint.py`

```py
# 设置文件编码为 UTF-8
# -*- coding: utf-8 -*-

# 从 pandas_ta 包中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 midpoint，用于计算 Midpoint 指标
def midpoint(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Midpoint"""
    # 验证参数
    # 如果 length 不为空且大于 0，则转换为整数，否则设为默认值 2
    length = int(length) if length and length > 0 else 2
    # 如果 kwargs 中包含 "min_periods" 键并且值不为空，则转换为整数，否则设为与 length 相同的值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 是否为有效序列，长度至少为 length 或 min_periods
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)
    # 判断是否使用 talib 库，默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    # 如果导入了 talib 并且 mode_tal 为 True
    if Imports["talib"] and mode_tal:
        # 从 talib 库中导入 MIDPOINT 函数，计算 Midpoint 指标
        from talib import MIDPOINT
        midpoint = MIDPOINT(close, length)
    else:
        # 使用 rolling 函数计算最低价和最高价的移动窗口，然后计算 Midpoint
        lowest = close.rolling(length, min_periods=min_periods).min()
        highest = close.rolling(length, min_periods=min_periods).max()
        midpoint = 0.5 * (lowest + highest)

    # 偏移结果
    if offset != 0:
        midpoint = midpoint.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        midpoint.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        midpoint.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    midpoint.name = f"MIDPOINT_{length}"
    midpoint.category = "overlap"

    # 返回 Midpoint 指标
    return midpoint
```