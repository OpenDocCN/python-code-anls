# `.\pandas-ta\pandas_ta\volume\mfi.py`

```py
# -*- coding: utf-8 -*-

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.overlap 模块中导入 hlc3 函数
from pandas_ta.overlap import hlc3
# 从 pandas_ta.utils 模块中导入 get_drift, get_offset, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series

# 定义 Money Flow Index (MFI) 函数
def mfi(high, low, close, volume, length=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Money Flow Index (MFI)"""

    # 验证参数
    length = int(length) if length and length > 0 else 14
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果任何一个输入序列为空，则返回空值
    if high is None or low is None or close is None or volume is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果 TA Lib 已安装且 talib 参数为 True，则使用 TA Lib 实现的 MFI 函数计算结果
        from talib import MFI
        mfi = MFI(high, low, close, volume, length)
    else:
        # 否则，使用自定义方法计算 MFI 指标

        # 计算典型价格和原始资金流
        typical_price = hlc3(high=high, low=low, close=close)
        raw_money_flow = typical_price * volume

        # 创建包含不同情况的数据框
        tdf = DataFrame({"diff": 0, "rmf": raw_money_flow, "+mf": 0, "-mf": 0})

        # 根据典型价格的变化情况更新数据框中的不同列
        tdf.loc[(typical_price.diff(drift) > 0), "diff"] = 1
        tdf.loc[tdf["diff"] == 1, "+mf"] = raw_money_flow
        tdf.loc[(typical_price.diff(drift) < 0), "diff"] = -1
        tdf.loc[tdf["diff"] == -1, "-mf"] = raw_money_flow

        # 计算正和负资金流的滚动和
        psum = tdf["+mf"].rolling(length).sum()
        nsum = tdf["-mf"].rolling(length).sum()

        # 计算资金流比率和 MFI 指标
        tdf["mr"] = psum / nsum
        mfi = 100 * psum / (psum + nsum)
        tdf["mfi"] = mfi

    # 偏移
    if offset != 0:
        mfi = mfi.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        mfi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mfi.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    mfi.name = f"MFI_{length}"
    mfi.category = "volume"

    return mfi

# 设置 MFI 函数的文档字符串
mfi.__doc__ = \
"""Money Flow Index (MFI)

Money Flow Index is an oscillator indicator that is used to measure buying and
selling pressure by utilizing both price and volume.

Sources:
    https://www.tradingview.com/wiki/Money_Flow_(MFI)

Calculation:
    Default Inputs:
        length=14, drift=1
    tp = typical_price = hlc3 = (high + low + close) / 3
    rmf = raw_money_flow = tp * volume

    pmf = pos_money_flow = SUM(rmf, length) if tp.diff(drift) > 0 else 0
    nmf = neg_money_flow = SUM(rmf, length) if tp.diff(drift) < 0 else 0

    MFR = money_flow_ratio = pmf / nmf
    MFI = money_flow_index = 100 * pmf / (pmf + nmf)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The sum period. Default: 14
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    drift (int): The difference period. Default: 1
"""
    offset (int): How many periods to offset the result. Default: 0

# 定义函数参数 offset，表示结果偏移的周期数，默认值为 0。
# 定义函数参数
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)  # 填充缺失值的方法和值
    fill_method (value, optional): Type of fill method  # 填充方法的类型

# 返回一个新的 pandas Series 对象，表示生成的新特征
Returns:
    pd.Series: New feature generated.  # 返回新生成的特征，类型为 pandas 的 Series 对象
```