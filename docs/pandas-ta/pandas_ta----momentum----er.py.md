# `.\pandas-ta\pandas_ta\momentum\er.py`

```
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 和 concat 函数
from pandas import DataFrame, concat
# 从 pandas_ta.utils 模块中导入 get_drift, get_offset, verify_series, signals 函数
from pandas_ta.utils import get_drift, get_offset, verify_series, signals

# 定义效率比率（ER）指标函数
def er(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Efficiency Ratio (ER)"""
    # 验证参数
    # 将长度参数转换为整数，并确保大于零
    length = int(length) if length and length > 0 else 10
    # 验证 close 参数，并且使用长度参数进行验证
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 获取漂移
    drift = get_drift(drift)

    # 如果 close 为 None，则返回 None
    if close is None: return

    # 计算结果
    # 计算价格变化的绝对值
    abs_diff = close.diff(length).abs()
    # 计算价格波动的绝对值
    abs_volatility = close.diff(drift).abs()

    # 计算效率比率
    er = abs_diff
    er /= abs_volatility.rolling(window=length).sum()

    # 偏移
    if offset != 0:
        er = er.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        er.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        er.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    er.name = f"ER_{length}"
    er.category = "momentum"

    # 处理信号指标
    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        # 将效率比率和信号指标拼接成一个 DataFrame
        signalsdf = concat(
            [
                DataFrame({er.name: er}),
                signals(
                    indicator=er,
                    xa=kwargs.pop("xa", 80),
                    xb=kwargs.pop("xb", 20),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return er

# 设置函数文档字符串
er.__doc__ = \
"""Efficiency Ratio (ER)

The Efficiency Ratio was invented by Perry J. Kaufman and presented in his book "New Trading Systems and Methods". It is designed to account for market noise or volatility.

It is calculated by dividing the net change in price movement over N periods by the sum of the absolute net changes over the same N periods.

Sources:
    https://help.tc2000.com/m/69404/l/749623-kaufman-efficiency-ratio

Calculation:
    Default Inputs:
        length=10
    ABS = Absolute Value
    EMA = Exponential Moving Average

    abs_diff = ABS(close.diff(length))
    volatility = ABS(close.diff(1))
    ER = abs_diff / SUM(volatility, length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```