# `.\pandas-ta\pandas_ta\volume\obv.py`

```
# -*- coding: utf-8 -*-
# 导入所需的库
from pandas_ta import Imports
from pandas_ta.utils import get_offset, signed_series, verify_series

# 定义 On Balance Volume (OBV) 指标函数
def obv(close, volume, talib=None, offset=None, **kwargs):
    """Indicator: On Balance Volume (OBV)"""
    # 验证参数
    # 确保 'close' 和 'volume' 是有效的 Series 对象
    close = verify_series(close)
    volume = verify_series(volume)
    # 获取偏移量
    offset = get_offset(offset)
    # 确定是否使用 TA Lib
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果 TA Lib 可用且 talib 参数为 True，则使用 TA Lib 中的 OBV 函数
        from talib import OBV
        obv = OBV(close, volume)
    else:
        # 否则，计算 signed_volume 并累积求和得到 OBV
        signed_volume = signed_series(close, initial=1) * volume
        obv = signed_volume.cumsum()

    # 偏移结果
    if offset != 0:
        obv = obv.shift(offset)

    # 处理填充
    # 如果 'fillna' 参数存在，则使用指定值填充缺失值
    if "fillna" in kwargs:
        obv.fillna(kwargs["fillna"], inplace=True)
    # 如果 'fill_method' 参数存在，则使用指定的填充方法填充缺失值
    if "fill_method" in kwargs:
        obv.fillna(method=kwargs["fill_method"], inplace=True)

    # 为结果添加名称和分类信息
    obv.name = f"OBV"
    obv.category = "volume"

    return obv


# 设置函数文档字符串
obv.__doc__ = \
"""On Balance Volume (OBV)

On Balance Volume is a cumulative indicator to measure buying and selling
pressure.

Sources:
    https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/on-balance-volume-obv/
    https://www.motivewave.com/studies/on_balance_volume.htm

Calculation:
    signed_volume = signed_series(close, initial=1) * volume
    obv = signed_volume.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```