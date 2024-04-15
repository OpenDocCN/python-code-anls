# `.\pandas-ta\examples\ni.py`

```py
# -*- coding: utf-8 -*-
# 从pandas_ta.overlap导入simple moving average函数
from pandas_ta.overlap import sma
# 从pandas_ta.utils导入获取偏移量的函数和验证序列的函数
from pandas_ta.utils import get_offset, verify_series

# 标准定义你的自定义指标函数（包括文档）
def ni(close, length=None, centered=False, offset=None, **kwargs):
    """
    Example indicator ni
    """
    # 验证参数
    length = int(length) if length and length > 0 else 20
    close = verify_series(close, length)
    offset = get_offset(offset)

    # 如果close为空，则返回空值
    if close is None: return

    # 计算结果
    t = int(0.5 * length) + 1
    # 计算简单移动平均线
    ma = sma(close, length)

    ni = close - ma.shift(t)
    # 如果设置了centered，则将ni进行居中调整
    if centered:
        ni = (close.shift(t) - ma).shift(-t)

    # 偏移
    if offset != 0:
        ni = ni.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        ni.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ni.fillna(method=kwargs["fill_method"], inplace=True)

    # 给新的特征命名和分类
    ni.name = f"ni_{length}"
    ni.category = "trend"

    return ni

# 设置自定义指标函数的文档字符串
ni.__doc__ = \
"""Example indicator (NI)

Is an indicator provided solely as an example

Sources:
    https://github.com/twopirllc/pandas-ta/issues/264

Calculation:
    Default Inputs:
        length=20, centered=False
    SMA = Simple Moving Average
    t = int(0.5 * length) + 1

    ni = close.shift(t) - SMA(close, length)
    if centered:
        ni = ni.shift(-t)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    centered (bool): Shift the ni back by int(0.5 * length) + 1. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""

# 定义匹配类方法
def ni_method(self, length=None, offset=None, **kwargs):
    # 从self中获取'close'列
    close = self._get_column(kwargs.pop("close", "close"))
    # 调用ni函数计算指标结果
    result = ni(close=close, length=length, offset=offset, **kwargs)
    # 对结果进行后处理
    return self._post_process(result, **kwargs)
```