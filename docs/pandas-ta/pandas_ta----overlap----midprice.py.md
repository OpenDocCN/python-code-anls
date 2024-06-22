# `.\pandas-ta\pandas_ta\overlap\midprice.py`

```py
# 设置文件编码为 UTF-8
# -*- coding: utf-8 -*-

# 从 pandas_ta 库导入 Imports 类
from pandas_ta import Imports
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def midprice(high, low, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Midprice"""
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数；否则，默认为 2
    length = int(length) if length and length > 0 else 2
    # 如果 kwargs 中存在 "min_periods"，则将其转换为整数；否则，默认为 length
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 计算有效期长度
    _length = max(length, min_periods)
    # 验证 high 和 low 系列
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    # 获取偏移量
    offset = get_offset(offset)
    # 判断是否启用 talib 模式
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 high 或 low 为 None，则返回
    if high is None or low is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 使用 talib 计算 MIDPRICE 指标
        from talib import MIDPRICE
        midprice = MIDPRICE(high, low, length)
    else:
        # 计算最低低点和最高高点的滚动窗口
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        highest_high = high.rolling(length, min_periods=min_periods).max()
        # 计算中间价
        midprice = 0.5 * (lowest_low + highest_high)

    # 偏移结果
    if offset != 0:
        midprice = midprice.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        midprice.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        midprice.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    midprice.name = f"MIDPRICE_{length}"
    midprice.category = "overlap"

    # 返回中间价指标结果
    return midprice
```