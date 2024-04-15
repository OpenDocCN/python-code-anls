# `.\pandas-ta\pandas_ta\trend\psar.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 nan 作为 npNaN
from numpy import nan as npNaN
# 导入 DataFrame 和 Series 类
from pandas import DataFrame, Series
# 导入 get_offset, verify_series, zero 函数
from pandas_ta.utils import get_offset, verify_series, zero

# 定义 PSAR 指标函数，参数包括 high, low, close, af0, af, max_af, offset
def psar(high, low, close=None, af0=None, af=None, max_af=None, offset=None, **kwargs):
    """Indicator: Parabolic Stop and Reverse (PSAR)"""
    # 验证参数
    high = verify_series(high)
    low = verify_series(low)
    af = float(af) if af and af > 0 else 0.02
    af0 = float(af0) if af0 and af0 > 0 else af
    max_af = float(max_af) if max_af and max_af > 0 else 0.2
    offset = get_offset(offset)

    # 定义 _falling 函数，用于返回最后一个 -DM 值
    def _falling(high, low, drift:int=1):
        """Returns the last -DM value"""
        # 不要与 ta.falling() 混淆
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
        return _dmn > 0

    # 如果第一个 NaN -DM 为正，则为下降趋势
    falling = _falling(high.iloc[:2], low.iloc[:2])
    if falling:
        sar = high.iloc[0]
        ep = low.iloc[0]
    else:
        sar = low.iloc[0]
        ep = high.iloc[0]

    # 如果存在 close 参数，则使用 close 的值
    if close is not None:
        close = verify_series(close)
        sar = close.iloc[0]

    # 初始化 long, short, reversal, _af
    long = Series(npNaN, index=high.index)
    short = long.copy()
    reversal = Series(0, index=high.index)
    _af = long.copy()
    _af.iloc[0:2] = af0

    # 计算结果
    m = high.shape[0]
    for row in range(1, m):
        high_ = high.iloc[row]
        low_ = low.iloc[row]

        if falling:
            _sar = sar + af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                af = min(af + af0, max_af)

            _sar = max(high.iloc[row - 1], high.iloc[row - 2], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                af = min(af + af0, max_af)

            _sar = min(low.iloc[row - 1], low.iloc[row - 2], _sar)

        if reverse:
            _sar = ep
            af = af0
            falling = not falling
            ep = low_ if falling else high_

        sar = _sar

        if falling:
            short.iloc[row] = sar
        else:
            long.iloc[row] = sar

        _af.iloc[row] = af
        reversal.iloc[row] = int(reverse)

    # 偏移
    if offset != 0:
        _af = _af.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)
        reversal = reversal.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        _af.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
        reversal.fillna(kwargs["fillna"], inplace=True)
    # 检查参数中是否包含'fill_method'关键字
    if "fill_method" in kwargs:
        # 使用指定的填充方法填充数据，直接修改原数据，无需返回值
        _af.fillna(method=kwargs["fill_method"], inplace=True)
        long.fillna(method=kwargs["fill_method"], inplace=True)
        short.fillna(method=kwargs["fill_method"], inplace=True)
        reversal.fillna(method=kwargs["fill_method"], inplace=True)

    # 准备要返回的 DataFrame
    # 根据参数构建用于命名列的字符串
    _params = f"_{af0}_{max_af}"
    # 构建包含 PSAR 数据的字典
    data = {
        f"PSARl{_params}": long,
        f"PSARs{_params}": short,
        f"PSARaf{_params}": _af,
        f"PSARr{_params}": reversal,
    }
    # 从字典创建 DataFrame
    psardf = DataFrame(data)
    # 设置 DataFrame 的名称
    psardf.name = f"PSAR{_params}"
    # 设置 DataFrame 及其列的类别属性为 'trend'
    psardf.category = long.category = short.category = "trend"

    # 返回 PSAR 数据的 DataFrame
    return psardf
# 设置 psar 函数的文档字符串，用于描述 Parabolic Stop and Reverse (PSAR) 指标的作用、计算方式和参数说明
psar.__doc__ = \
"""Parabolic Stop and Reverse (psar)

Parabolic Stop and Reverse (PSAR) was developed by J. Wells Wilder, that is used
to determine trend direction and it's potential reversals in price. PSAR uses a
trailing stop and reverse method called "SAR," or stop and reverse, to identify
possible entries and exits. It is also known as SAR.

PSAR indicator typically appears on a chart as a series of dots, either above or
below an asset's price, depending on the direction the price is moving. A dot is
placed below the price when it is trending upward, and above the price when it
is trending downward.

Sources:
    https://www.tradingview.com/pine-script-reference/#fun_sar
    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=66&Name=Parabolic

Calculation:
    Default Inputs:
        af0=0.02, af=0.02, max_af=0.2

    See Source links

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series, optional): Series of 'close's. Optional
    af0 (float): Initial Acceleration Factor. Default: 0.02
    af (float): Acceleration Factor. Default: 0.02
    max_af (float): Maximum Acceleration Factor. Default: 0.2
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: long, short, af, and reversal columns.
"""
```