# `.\pandas-ta\pandas_ta\performance\log_return.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 中导入 log 函数并重命名为 nplog
from numpy import log as nplog
# 从 pandas_ta.utils 中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def log_return(close, length=None, cumulative=None, offset=None, **kwargs):
    """Indicator: Log Return"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为 1
    length = int(length) if length and length > 0 else 1
    # 如果 cumulative 存在且为真，则设为 True，否则设为 False
    cumulative = bool(cumulative) if cumulative is not None and cumulative else False
    # 验证 close 是否为有效的 Series，并设置长度
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    if cumulative:
        # 计算累积对数收益率
        log_return = nplog(close / close.iloc[0])
    else:
        # 计算对数收益率
        log_return = nplog(close / close.shift(length))

    # 偏移结果
    if offset != 0:
        log_return = log_return.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        log_return.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        log_return.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    log_return.name = f"{'CUM' if cumulative else ''}LOGRET_{length}"
    log_return.category = "performance"

    return log_return


# 设置 log_return 函数的文档字符串
log_return.__doc__ = \
"""Log Return

Calculates the logarithmic return of a Series.
See also: help(df.ta.log_return) for additional **kwargs a valid 'df'.

Sources:
    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

Calculation:
    Default Inputs:
        length=1, cumulative=False
    LOGRET = log( close.diff(periods=length) )
    CUMLOGRET = LOGRET.cumsum() if cumulative

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    cumulative (bool): If True, returns the cumulative returns. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```