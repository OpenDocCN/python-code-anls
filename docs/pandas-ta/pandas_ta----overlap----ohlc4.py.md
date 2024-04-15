# `.\pandas-ta\pandas_ta\overlap\ohlc4.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义一个名为 ohlc4 的函数，用于计算 OHLC4 指标
def ohlc4(open_, high, low, close, offset=None, **kwargs):
    """Indicator: OHLC4"""
    # 验证输入参数，确保它们都是 pandas Series 类型
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    # 获取位移值
    offset = get_offset(offset)

    # 计算 OHLC4 指标的值，使用开盘价、最高价、最低价和收盘价的均值
    ohlc4 = 0.25 * (open_ + high + low + close)

    # 如果存在位移值，则将 OHLC4 指标的值向前位移相应数量的周期
    if offset != 0:
        ohlc4 = ohlc4.shift(offset)

    # 设置 OHLC4 指标的名称和类别
    ohlc4.name = "OHLC4"
    ohlc4.category = "overlap"

    # 返回计算得到的 OHLC4 指标
    return ohlc4
```