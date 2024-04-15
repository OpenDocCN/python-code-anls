# `.\pandas-ta\pandas_ta\momentum\dm.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.utils 模块中导入 get_offset, verify_series, get_drift, zero 函数
from pandas_ta.utils import get_offset, verify_series, get_drift, zero

# 定义函数 dm，计算 Directional Movement 指标
def dm(high, low, length=None, mamode=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: DM"""
    # 验证参数
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high)
    low = verify_series(low)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 high 或 low 为空，则返回空
    if high is None or low is None:
        return

    # 如果导入了 talib 并且 mode_tal 为 True，则使用 talib 库中的 PLUS_DM 和 MINUS_DM 函数
    if Imports["talib"] and mode_tal:
        from talib import MINUS_DM, PLUS_DM
        pos = PLUS_DM(high, low, length)
        neg = MINUS_DM(high, low, length)
    else:
        # 计算 up 和 dn
        up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos_ = ((up > dn) & (up > 0)) * up
        neg_ = ((dn > up) & (dn > 0)) * dn

        pos_ = pos_.apply(zero)
        neg_ = neg_.apply(zero)

        # 计算 pos 和 neg
        pos = ma(mamode, pos_, length=length)
        neg = ma(mamode, neg_, length=length)

    # 偏移
    if offset != 0:
        pos = pos.shift(offset)
        neg = neg.shift(offset)

    _params = f"_{length}"
    data = {
        f"DMP{_params}": pos,
        f"DMN{_params}": neg,
    }

    # 创建 DataFrame 对象
    dmdf = DataFrame(data)
    # 设置 DataFrame 的名称和类别
    dmdf.name = f"DM{_params}"
    dmdf.category = "trend"

    return dmdf

# 设置 dm 函数的文档字符串
dm.__doc__ = \
"""Directional Movement (DM)

The Directional Movement was developed by J. Welles Wilder in 1978 attempts to
determine which direction the price of an asset is moving. It compares prior
highs and lows to yield to two series +DM and -DM.

Sources:
    https://www.tradingview.com/pine-script-reference/#fun_dmi
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=24&Name=Directional_Movement_Index

Calculation:
    Default Inputs:
        length=14, mamode="rma", drift=1
            up = high - high.shift(drift)
        dn = low.shift(drift) - low

        pos_ = ((up > dn) & (up > 0)) * up
        neg_ = ((dn > up) & (dn > 0)) * dn

        pos_ = pos_.apply(zero)
        neg_ = neg_.apply(zero)

        # Not the same values as TA Lib's -+DM
        pos = ma(mamode, pos_, length=length)
        neg = ma(mamode, neg_, length=length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    mamode (str): See ```help(ta.ma)```py.  Default: 'rma'
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Returns:
    pd.DataFrame: DMP (+DM) and DMN (-DM) columns.
"""
```