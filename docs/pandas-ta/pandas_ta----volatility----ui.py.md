# `.\pandas-ta\pandas_ta\volatility\ui.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库导入 sqrt 函数并重命名为 npsqrt
from numpy import sqrt as npsqrt
# 从 pandas_ta.overlap 模块导入 sma 函数
from pandas_ta.overlap import sma
# 从 pandas_ta.utils 模块导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数 ui，计算 Ulcer Index（UI）
def ui(close, length=None, scalar=None, offset=None, **kwargs):
    """Indicator: Ulcer Index (UI)"""
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数，否则默认为 14
    length = int(length) if length and length > 0 else 14
    # 如果 scalar 存在且大于 0，则转换为浮点数，否则默认为 100
    scalar = float(scalar) if scalar and scalar > 0 else 100
    # 验证 close 是否为有效的时间序列，并指定长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    # 计算最近 length 个周期内的最高收盘价
    highest_close = close.rolling(length).max()
    # 计算下行波动性
    downside = scalar * (close - highest_close)
    downside /= highest_close
    d2 = downside * downside

    # 获取 everget 参数，默认为 False
    everget = kwargs.pop("everget", False)
    # 如果 everget 为 True，则使用 SMA 而不是 SUM 进行计算
    if everget:
        # 使用 SMA 对 d2 进行计算，然后应用 npsqrt 函数
        ui = (sma(d2, length) / length).apply(npsqrt)
    else:
        # 使用 SUM 对 d2 进行计算，然后应用 npsqrt 函数
        ui = (d2.rolling(length).sum() / length).apply(npsqrt)

    # 偏移结果
    if offset != 0:
        ui = ui.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 使用指定值填充空值
        ui.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 使用指定的填充方法填充空值
        ui.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    ui.name = f"UI{'' if not everget else 'e'}_{length}"
    ui.category = "volatility"

    # 返回结果
    return ui


# 设置 ui 函数的文档字符串
ui.__doc__ = \
"""Ulcer Index (UI)

The Ulcer Index by Peter Martin measures the downside volatility with the use of
the Quadratic Mean, which has the effect of emphasising large drawdowns.

Sources:
    https://library.tradingtechnologies.com/trade/chrt-ti-ulcer-index.html
    https://en.wikipedia.org/wiki/Ulcer_index
    http://www.tangotools.com/ui/ui.htm

Calculation:
    Default Inputs:
        length=14, scalar=100
    HC = Highest Close
    SMA = Simple Moving Average

    HCN = HC(close, length)
    DOWNSIDE = scalar * (close - HCN) / HCN
    if kwargs["everget"]:
        UI = SQRT(SMA(DOWNSIDE^2, length) / length)
    else:
        UI = SQRT(SUM(DOWNSIDE^2, length) / length)

Args:
    high (pd.Series): Series of 'high's
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 14
    scalar (float): A positive float to scale the bands. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
    everget (value, optional): TradingView's Evergets SMA instead of SUM
        calculation. Default: False

Returns:
    pd.Series: New feature
"""
```