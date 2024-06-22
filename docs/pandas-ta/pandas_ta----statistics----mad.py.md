# `.\pandas-ta\pandas_ta\statistics\mad.py`

```py
# -*- coding: utf-8 -*-
# 从 numpy 中导入 fabs 函数并重命名为 npfabs
from numpy import fabs as npfabs
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数：均值绝对偏差
def mad(close, length=None, offset=None, **kwargs):
    """Indicator: Mean Absolute Deviation"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 30
    length = int(length) if length and length > 0 else 30
    # 如果 kwargs 中存在 "min_periods"，则将其转换为整数，否则使用 length 作为默认值
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    # 验证 close 序列，保证长度至少为 length 和 min_periods 中的较大值
    close = verify_series(close, max(length, min_periods))
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回空值
    if close is None: return

    # 计算结果
    def mad_(series):
        """Mean Absolute Deviation"""
        # 计算序列与其均值的绝对差值的均值
        return npfabs(series - series.mean()).mean()

    # 使用 rolling 函数计算滚动均值绝对偏差
    mad = close.rolling(length, min_periods=min_periods).apply(mad_, raw=True)

    # 偏移
    if offset != 0:
        mad = mad.shift(offset)

    # 处理填充
    # 如果 kwargs 中存在 "fillna"，则使用该值填充空值
    if "fillna" in kwargs:
        mad.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中存在 "fill_method"，则使用指定的填充方法
    if "fill_method" in kwargs:
        mad.fillna(method=kwargs["fill_method"], inplace=True)

    # 设定指标的名称和类别
    mad.name = f"MAD_{length}"
    mad.category = "statistics"

    return mad


# 设置函数文档字符串
mad.__doc__ = \
"""Rolling Mean Absolute Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    mad = close.rolling(length).mad()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```