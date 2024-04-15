# `.\pandas-ta\pandas_ta\momentum\rsx.py`

```
# 设置文件编码为 UTF-8
# -*- coding: utf-8 -*-

# 导入 numpy 库中的 nan 常量，并将其重命名为 npNaN
from numpy import nan as npNaN

# 从 pandas 库中导入 concat、DataFrame 和 Series 类
from pandas import concat, DataFrame, Series

# 从 pandas_ta.utils 模块中导入 get_drift、get_offset、verify_series 和 signals 函数
from pandas_ta.utils import get_drift, get_offset, verify_series, signals

# 定义 RSX 指标函数，计算相对强度 Xtra（灵感来自于 Jurik RSX）
def rsx(close, length=None, drift=None, offset=None, **kwargs):
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数，否则设为默认值 14
    length = int(length) if length and length > 0 else 14
    # 验证收盘价序列，确保其长度为 length
    close = verify_series(close, length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果收盘价序列为空，则返回空
    if close is None: return

    # 初始化变量
    vC, v1C = 0, 0
    v4, v8, v10, v14, v18, v20 = 0, 0, 0, 0, 0, 0
    f0, f8, f10, f18, f20, f28, f30, f38 = 0, 0, 0, 0, 0, 0, 0, 0
    f40, f48, f50, f58, f60, f68, f70, f78 = 0, 0, 0, 0, 0, 0, 0, 0
    f80, f88, f90 = 0, 0, 0

    # 计算结果
    m = close.size
    result = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        if f90 == 0:
            f90 = 1.0
            f0 = 0.0
            if length - 1.0 >= 5:
                f88 = length - 1.0
            else:
                f88 = 5.0
            f8 = 100.0 * close.iloc[i]
            f18 = 3.0 / (length + 2.0)
            f20 = 1.0 - f18
        else:
            if f88 <= f90:
                f90 = f88 + 1
            else:
                f90 = f90 + 1
            f10 = f8
            f8 = 100 * close.iloc[i]
            v8 = f8 - f10
            f28 = f20 * f28 + f18 * v8
            f30 = f18 * f28 + f20 * f30
            vC = 1.5 * f28 - 0.5 * f30
            f38 = f20 * f38 + f18 * vC
            f40 = f18 * f38 + f20 * f40
            v10 = 1.5 * f38 - 0.5 * f40
            f48 = f20 * f48 + f18 * v10
            f50 = f18 * f48 + f20 * f50
            v14 = 1.5 * f48 - 0.5 * f50
            f58 = f20 * f58 + f18 * abs(v8)
            f60 = f18 * f58 + f20 * f60
            v18 = 1.5 * f58 - 0.5 * f60
            f68 = f20 * f68 + f18 * v18
            f70 = f18 * f68 + f20 * f70
            v1C = 1.5 * f68 - 0.5 * f70
            f78 = f20 * f78 + f18 * v1C
            f80 = f18 * f78 + f20 * f80
            v20 = 1.5 * f78 - 0.5 * f80

            if f88 >= f90 and f8 != f10:
                f0 = 1.0
            if f88 == f90 and f0 == 0.0:
                f90 = 0.0

        if f88 < f90 and v20 > 0.0000000001:
            v4 = (v14 / v20 + 1.0) * 50.0
            if v4 > 100.0:
                v4 = 100.0
            if v4 < 0.0:
                v4 = 0.0
        else:
            v4 = 50.0
        result.append(v4)

    # 将结果转换为 Series，索引与收盘价相同
    rsx = Series(result, index=close.index)

    # 如果偏移不为 0，则将结果向前偏移指定数量的周期
    if offset != 0:
        rsx = rsx.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        rsx.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsx.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    rsx.name = f"RSX_{length}"
    rsx.category = "momentum"

    # 推导出信号指示器
    signal_indicators = kwargs.pop("signal_indicators", False)
    # 如果信号指标存在
    if signal_indicators:
        # 将RSX指标转换为DataFrame格式，并与信号指标合并
        signalsdf = concat(
            [
                DataFrame({rsx.name: rsx}),
                # 调用signals函数生成信号指标
                signals(
                    indicator=rsx,
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
            axis=1
        )
        # 返回合并后的DataFrame
        return signalsdf
    else:
        # 如果信号指标不存在，直接返回RSX指标
        return rsx
# 设置 rsx 函数的文档字符串，说明了该函数的作用、来源、计算方法、参数和返回值
rsx.__doc__ = \
"""Relative Strength Xtra (rsx)

The Relative Strength Xtra is based on the popular RSI indicator and inspired
by the work Jurik Research. The code implemented is based on published code
found at 'prorealcode.com'. This enhanced version of the rsi reduces noise and
provides a clearer, only slightly delayed insight on momentum and velocity of
price movements.

Sources:
    http://www.jurikres.com/catalog1/ms_rsx.htm
    https://www.prorealcode.com/prorealtime-indicators/jurik-rsx/

Calculation:
    Refer to the sources above for information as well as code example.

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```