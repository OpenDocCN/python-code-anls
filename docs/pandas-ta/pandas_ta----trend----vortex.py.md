# `.\pandas-ta\pandas_ta\trend\vortex.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.volatility 模块导入 true_range 函数
from pandas_ta.volatility import true_range
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义函数 vortex，用于计算 Vortex 指标
def vortex(high, low, close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Vortex"""
    # 验证参数
    length = length if length and length > 0 else 14
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    # 验证并处理输入的 high、low、close 数据
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    drift = get_drift(drift)  # 获取 drift 参数
    offset = get_offset(offset)  # 获取 offset 参数

    # 若输入数据中有空值，则返回空
    if high is None or low is None or close is None: return

    # 计算结果
    tr = true_range(high=high, low=low, close=close)  # 计算真实范围
    tr_sum = tr.rolling(length, min_periods=min_periods).sum()  # 对真实范围进行滚动求和

    # 计算正向运动价格动量 (VMP) 和反向运动价格动量 (VMM)
    vmp = (high - low.shift(drift)).abs()
    vmm = (low - high.shift(drift)).abs()

    # 计算正向和反向运动指标 (VIP 和 VIM)
    vip = vmp.rolling(length, min_periods=min_periods).sum() / tr_sum
    vim = vmm.rolling(length, min_periods=min_periods).sum() / tr_sum

    # 偏移结果
    if offset != 0:
        vip = vip.shift(offset)
        vim = vim.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        vip.fillna(kwargs["fillna"], inplace=True)
        vim.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vip.fillna(method=kwargs["fill_method"], inplace=True)
        vim.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名并分类化指标
    vip.name = f"VTXP_{length}"
    vim.name = f"VTXM_{length}"
    vip.category = vim.category = "trend"

    # 准备返回的 DataFrame
    data = {vip.name: vip, vim.name: vim}
    vtxdf = DataFrame(data)
    vtxdf.name = f"VTX_{length}"
    vtxdf.category = "trend"

    return vtxdf


# 设置函数文档字符串
vortex.__doc__ = \
"""Vortex

Two oscillators that capture positive and negative trend movement.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

Calculation:
    Default Inputs:
        length=14, drift=1
    TR = True Range
    SMA = Simple Moving Average
    tr = TR(high, low, close)
    tr_sum = tr.rolling(length).sum()

    vmp = (high - low.shift(drift)).abs()
    vmn = (low - high.shift(drift)).abs()

    VIP = vmp.rolling(length).sum() / tr_sum
    VIM = vmn.rolling(length).sum() / tr_sum

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): ROC 1 period. Default: 14
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: vip and vim columns
"""
```