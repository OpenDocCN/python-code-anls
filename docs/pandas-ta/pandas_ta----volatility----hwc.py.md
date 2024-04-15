# `.\pandas-ta\pandas_ta\volatility\hwc.py`

```
# -*- coding: utf-8 -*-
# 导入 numpy 库中的 sqrt 函数并重命名为 npSqrt
from numpy import sqrt as npSqrt
# 导入 pandas 库中的 DataFrame 和 Series 类
from pandas import DataFrame, Series
# 导入 pandas_ta 库中的 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义 Holt-Winter Channel 指标函数，接受 close 数据和一系列参数
def hwc(close, na=None, nb=None, nc=None, nd=None, scalar=None, channel_eval=None, offset=None, **kwargs):
    """Indicator: Holt-Winter Channel"""
    # 验证参数
   na = float(na) if na and na > 0 else 0.2
    nb = float(nb) if nb and nb > 0 else 0.1
    nc = float(nc) if nc and nc > 0 else 0.1
    nd = float(nd) if nd and nd > 0 else 0.1
    scalar = float(scalar) if scalar and scalar > 0 else 1
    channel_eval = bool(channel_eval) if channel_eval and channel_eval else False
    close = verify_series(close)
    offset = get_offset(offset)

    # 计算结果
    last_a = last_v = last_var = 0
    last_f = last_price = last_result = close[0]
    lower, result, upper = [], [], []
    chan_pct_width, chan_width = [], []

    m = close.size
    # 遍历 close 数据
    for i in range(m):
        # 计算 F、V、A
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result.append((F + V + 0.5 * A))

        # 计算方差和标准差
        var = (1.0 - nd) * last_var + nd * (last_price - last_result) * (last_price - last_result)
        stddev = npSqrt(last_var)
        upper.append(result[i] + scalar * stddev)
        lower.append(result[i] - scalar * stddev)

        if channel_eval:
            # 计算通道宽度和价格位置
            chan_width.append(upper[i] - lower[i])
            chan_pct_width.append((close[i] - lower[i]) / (upper[i] - lower[i]))

        # 更新数值
        last_price = close[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = result[i]

    # 聚合结果
    hwc = Series(result, index=close.index)
    hwc_upper = Series(upper, index=close.index)
    hwc_lower = Series(lower, index=close.index)
    if channel_eval:
        hwc_width = Series(chan_width, index=close.index)
        hwc_pctwidth = Series(chan_pct_width, index=close.index)

    # 偏移数据
    if offset != 0:
        hwc = hwc.shift(offset)
        hwc_upper = hwc_upper.shift(offset)
        hwc_lower = hwc_lower.shift(offset)
        if channel_eval:
            hwc_width = hwc_width.shift(offset)
            hwc_pctwidth = hwc_pctwidth.shift(offset)

    # 处理��充值
    if "fillna" in kwargs:
        hwc.fillna(kwargs["fillna"], inplace=True)
        hwc_upper.fillna(kwargs["fillna"], inplace=True)
        hwc_lower.fillna(kwargs["fillna"], inplace=True)
        if channel_eval:
            hwc_width.fillna(kwargs["fillna"], inplace=True)
            hwc_pctwidth.fillna(kwargs["fillna"], inplace=True)
    # 检查是否在参数 kwargs 中包含了 "fill_method" 键
    if "fill_method" in kwargs:
        # 对 hwc DataFrame 中的缺失值进行填充，使用 kwargs 中指定的填充方法，inplace 参数设为 True 表示就地修改
        hwc.fillna(method=kwargs["fill_method"], inplace=True)
        # 对 hwc_upper DataFrame 中的缺失值进行填充，使用 kwargs 中指定的填充方法，inplace 参数设为 True 表示就地修改
        hwc_upper.fillna(method=kwargs["fill_method"], inplace=True)
        # 对 hwc_lower DataFrame 中的缺失值进行填充，使用 kwargs 中指定的填充方法，inplace 参数设为 True 表示就地修改
        hwc_lower.fillna(method=kwargs["fill_method"], inplace=True)
        # 如果需要进行通道评估
        if channel_eval:
            # 对 hwc_width DataFrame 中的缺失值进行填充，使用 kwargs 中指定的填充方法，inplace 参数设为 True 表示就地修改
            hwc_width.fillna(method=kwargs["fill_method"], inplace=True)
            # 对 hwc_pctwidth DataFrame 中的缺失值进行填充，使用 kwargs 中指定的填充方法，inplace 参数设为 True 表示就地修改
            hwc_pctwidth.fillna(method=kwargs["fill_method"], inplace=True)

    # 给 DataFrame 添加名称和分类
    hwc.name = "HWM"
    hwc_upper.name = "HWU"
    hwc_lower.name = "HWL"
    hwc.category = hwc_upper.category = hwc_lower.category = "volatility"
    # 如果需要进行通道评估
    if channel_eval:
        hwc_width.name = "HWW"
        hwc_pctwidth.name = "HWPCT"

    # 准备要返回的 DataFrame
    if channel_eval:
        # 构建一个包含各列的字典
        data = {hwc.name: hwc, hwc_upper.name: hwc_upper, hwc_lower.name: hwc_lower,
                hwc_width.name: hwc_width, hwc_pctwidth.name: hwc_pctwidth}
        # 使用字典构建 DataFrame
        df = DataFrame(data)
        # 设置 DataFrame 的名称
        df.name = "HWC"
        # 设置 DataFrame 的分类
        df.category = hwc.category
    else:
        # 构建一个包含各列的字典
        data = {hwc.name: hwc, hwc_upper.name: hwc_upper, hwc_lower.name: hwc_lower}
        # 使用字典构建 DataFrame
        df = DataFrame(data)
        # 设置 DataFrame 的名称
        df.name = "HWC"
        # 设置 DataFrame 的分类
        df.category = hwc.category

    # 返回构建好的 DataFrame
    return df
# 将 hwc 对象的文档字符串设置为指定的内容，用于描述 Holt-Winter 通道指标 HWC（Holt-Winters Channel）
hwc.__doc__ = \
"""HWC (Holt-Winter Channel)

Channel indicator HWC (Holt-Winters Channel) based on HWMA - a three-parameter
moving average calculated by the method of Holt-Winters.

This version has been implemented for Pandas TA by rengel8 based on a
publication for MetaTrader 5 extended by width and percentage price position
against width of channel.

Sources:
    https://www.mql5.com/en/code/20857

Calculation:
    HWMA[i] = F[i] + V[i] + 0.5 * A[i]
    where..
    F[i] = (1-na) * (F[i-1] + V[i-1] + 0.5 * A[i-1]) + na * Price[i]
    V[i] = (1-nb) * (V[i-1] + A[i-1]) + nb * (F[i] - F[i-1])
    A[i] = (1-nc) * A[i-1] + nc * (V[i] - V[i-1])

    Top = HWMA + Multiplier * StDt
    Bottom = HWMA - Multiplier * StDt
    where..
    StDt[i] = Sqrt(Var[i-1])
    Var[i] = (1-d) * Var[i-1] + nD * (Price[i-1] - HWMA[i-1]) * (Price[i-1] - HWMA[i-1])

Args:
    na - parameter of the equation that describes a smoothed series (from 0 to 1)
    nb - parameter of the equation to assess the trend (from 0 to 1)
    nc - parameter of the equation to assess seasonality (from 0 to 1)
    nd - parameter of the channel equation (from 0 to 1)
    scaler - multiplier for the width of the channel calculated
    channel_eval - boolean to return width and percentage price position against price
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
Returns:
    pd.DataFrame: HWM (Mid), HWU (Upper), HWL (Lower) columns.
"""
```