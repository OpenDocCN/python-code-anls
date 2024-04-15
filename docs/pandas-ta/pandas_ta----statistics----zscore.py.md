# `.\pandas-ta\pandas_ta\statistics\zscore.py`

```py
# -*- coding: utf-8 -*- 
# 从pandas_ta.overlap模块中导入sma函数
from pandas_ta.overlap import sma
# 从本地的stdev模块中导入stdev函数
from .stdev import stdev
# 从pandas_ta.utils模块中导入get_offset和verify_series函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数zscore，用于计算Z分数指标
def zscore(close, length=None, std=None, offset=None, **kwargs):
    """Indicator: Z Score"""
    # 验证参数
    # 将length转换为整数，如果length存在且大于1，则取其值，否则默认为30
    length = int(length) if length and length > 1 else 30
    # 将std转换为浮点数，如果std存在且大于1，则取其值，否则默认为1
    std = float(std) if std and std > 1 else 1
    # 验证close是否为有效的Series，长度为length
    close = verify_series(close, length)
    # 获取offset值
    offset = get_offset(offset)

    # 如果close为空，则返回空
    if close is None: return

    # 计算结果
    # 将std乘以stdev函数计算的标准差值
    std *= stdev(close=close, length=length, **kwargs)
    # 计算均值，使用sma函数计算移动平均值
    mean = sma(close=close, length=length, **kwargs)
    # 计算Z分数
    zscore = (close - mean) / std

    # 调整偏移量
    if offset != 0:
        zscore = zscore.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        zscore.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        zscore.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    zscore.name = f"ZS_{length}"
    zscore.category = "statistics"

    return zscore


# 设置zscore函数的文档字符串
zscore.__doc__ = \
"""Rolling Z Score

Sources:

Calculation:
    Default Inputs:
        length=30, std=1
    SMA = Simple Moving Average
    STDEV = Standard Deviation
    std = std * STDEV(close, length)
    mean = SMA(close, length)
    ZSCORE = (close - mean) / std

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    std (float): It's period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```