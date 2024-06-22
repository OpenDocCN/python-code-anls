# `.\pandas-ta\pandas_ta\overlap\hlc3.py`

```py
# -*- coding: utf-8 -*-
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 hlc3，计算 HLC3 指标
def hlc3(high, low, close, talib=None, offset=None, **kwargs):
    """Indicator: HLC3"""
    # 验证参数
   # 验证 high、low、close 是否为 Series 类型
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
   # 获取偏移量
    offset = get_offset(offset)
   # 判断是否使用 talib 库，默认为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 计算结果
   # 如果导入了 talib 库并且 mode_tal 为 True
    if Imports["talib"] and mode_tal:
       # 从 talib 库中导入 TYPPRICE 函数，计算 HLC3
        from talib import TYPPRICE
        hlc3 = TYPPRICE(high, low, close)
    else:
       # 否则，使用普通方法计算 HLC3
        hlc3 = (high + low + close) / 3.0

    # 偏移
   # 如果偏移量不为 0，则对结果进行偏移
    if offset != 0:
        hlc3 = hlc3.shift(offset)

    # 名称和类别
   # 设置结果的名称为 "HLC3"，类别为 "overlap"
    hlc3.name = "HLC3"
    hlc3.category = "overlap"

    # 返回计算结果
    return hlc3
```