# `.\pandas-ta\pandas_ta\overlap\ema.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 导入 nan 并重命名为 npNaN
from numpy import nan as npNaN
# 从 pandas_ta 导入 Imports 模块
from pandas_ta import Imports
# 从 pandas_ta.utils 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def ema(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Exponential Moving Average (EMA)"""
    # 验证参数
    # 将 length 转换为整数，如果 length 存在且大于 0，否则默认为 10
    length = int(length) if length and length > 0 else 10
    # 从 kwargs 中弹出 "adjust" 键的值，默认为 False
    adjust = kwargs.pop("adjust", False)
    # 从 kwargs 中弹出 "sma" 键的值，默认为 True
    sma = kwargs.pop("sma", True)
    # 验证 close 数据，并使用 length 进行验证
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 talib 存在且为布尔类型，则将 mode_tal 设置为 talib，否则设置为 True
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 close 为 None，则返回
    if close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 如果 Imports 中的 "talib" 为 True 且 mode_tal 为 True，则使用 talib 库中的 EMA 函数
        from talib import EMA
        ema = EMA(close, length)
    else:
        # 否则执行以下操作
        if sma:
            # 如果 sma 为 True，则执行以下操作
            close = close.copy()
            # 计算前 length 个 close 的均值作为初始值
            sma_nth = close[0:length].mean()
            # 将 close 的前 length-1 个值设为 NaN
            close[:length - 1] = npNaN
            # 将 close 的第 length-1 个值设为初始均值
            close.iloc[length - 1] = sma_nth
        # 使用指数加权移动平均计算 EMA
        ema = close.ewm(span=length, adjust=adjust).mean()

    # 偏移结果
    if offset != 0:
        ema = ema.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        ema.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ema.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    # 设置 ema 的名称为 "EMA_length"，类别为 "overlap"
    ema.name = f"EMA_{length}"
    ema.category = "overlap"

    return ema


# 重新定义 ema 函数的文档字符串
ema.__doc__ = \
"""Exponential Moving Average (EMA)

指数移动平均是对比简单移动平均（SMA）更具响应性的移动平均。其权重由 alpha 决定，与其长度成正比。有几种不同的计算 EMA 的方法。一种方法仅使用标准的 EMA 定义，另一种方法使用 SMA 生成其余计算的初始值。

来源：
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

计算：
    默认参数：
        length=10, adjust=False, sma=True
    如果 sma 为 True：
        sma_nth = close[0:length].sum() / length
        close[:length - 1] = np.NaN
        close.iloc[length - 1] = sma_nth
    EMA = close.ewm(span=length, adjust=adjust).mean()

参数：
    close (pd.Series): 'close' 数据的序列
    length (int): 周期。默认为 10
    talib (bool): 如果安装了 TA Lib 并且 talib 为 True，则返回 TA Lib 版本。默认为 True
    offset (int): 结果的偏移周期数。默认为 0

可选参数：
    adjust (bool, optional): 默认为 False
    sma (bool, optional): 如果为 True，则使用 SMA 作为初始值。默认为 True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): 填充方法的类型

返回：
    pd.Series: 生成的新特征。
"""
```