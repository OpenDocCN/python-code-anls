# `.\pandas-ta\pandas_ta\performance\drawdown.py`

```py
# 导入所需模块
# -*- coding: utf-8 -*-
# 从 numpy 模块中导入 log 函数并重命名为 nplog
from numpy import log as nplog
# 从 numpy 模块中导入 seterr 函数
from numpy import seterr
# 从 pandas 模块中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 drawdown，用于计算资产或投资组合的回撤情况
def drawdown(close, offset=None, **kwargs) -> DataFrame:
    """Indicator: Drawdown (DD)"""
    # 验证参数合法性，确保 close 是一个 Series 对象
    close = verify_series(close)
    # 获取偏移量
    offset = get_offset(offset)

    # 计算结果
    # 计算历史最高收盘价
    max_close = close.cummax()
    # 计算回撤
    dd = max_close - close
    # 计算回撤百分比
    dd_pct = 1 - (close / max_close)

    # 临时忽略 numpy 的警告
    _np_err = seterr()
    seterr(divide="ignore", invalid="ignore")
    # 计算回撤的对数
    dd_log = nplog(max_close) - nplog(close)
    # 恢复 numpy 的警告设置
    seterr(divide=_np_err["divide"], invalid=_np_err["invalid"])

    # 调整偏移量
    if offset != 0:
        dd = dd.shift(offset)
        dd_pct = dd_pct.shift(offset)
        dd_log = dd_log.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        dd.fillna(kwargs["fillna"], inplace=True)
        dd_pct.fillna(kwargs["fillna"], inplace=True)
        dd_log.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        dd.fillna(method=kwargs["fill_method"], inplace=True)
        dd_pct.fillna(method=kwargs["fill_method"], inplace=True)
        dd_log.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置列名和分类
    dd.name = "DD"
    dd_pct.name = f"{dd.name}_PCT"
    dd_log.name = f"{dd.name}_LOG"
    dd.category = dd_pct.category = dd_log.category = "performance"

    # 准备返回的 DataFrame
    data = {dd.name: dd, dd_pct.name: dd_pct, dd_log.name: dd_log}
    df = DataFrame(data)
    df.name = dd.name
    df.category = dd.category

    return df

# 设置函数文档字符串
drawdown.__doc__ = \
"""Drawdown (DD)

Drawdown is a peak-to-trough decline during a specific period for an investment,
trading account, or fund. It is usually quoted as the percentage between the
peak and the subsequent trough.

Sources:
    https://www.investopedia.com/terms/d/drawdown.asp

Calculation:
    PEAKDD = close.cummax()
    DD = PEAKDD - close
    DD% = 1 - (close / PEAKDD)
    DDlog = log(PEAKDD / close)

Args:
    close (pd.Series): Series of 'close's.
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: drawdown, drawdown percent, drawdown log columns
"""
```