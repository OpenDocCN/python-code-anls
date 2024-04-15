# `.\pandas-ta\pandas_ta\volume\efi.py`

```
# -*- coding: utf-8 -*-
# 从 pandas_ta.overlap 模块导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.utils 模块导入 get_drift、get_offset、verify_series 函数
from pandas_ta.utils import get_drift, get_offset, verify_series


# 定义 EFI 函数，计算 Elder's Force Index (EFI)
def efi(close, volume, length=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Elder's Force Index (EFI)"""
    # 验证参数
    # 将 length 转换为整数，如果未提供或小于等于 0，则设为默认值 13
    length = int(length) if length and length > 0 else 13
    # 如果未提供 mamode 或不是字符串，则设为默认值 'ema'
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 验证 close 和 volume 是否为有效序列，长度为 length
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    # 获取漂移和偏移量
    drift = get_drift(drift)
    offset = get_offset(offset)

    # 如果 close 或 volume 为空，则返回空
    if close is None or volume is None: return

    # 计算结果
    # 计算价格和成交量的差值，再乘以漂移量
    pv_diff = close.diff(drift) * volume
    # 计算 EFI，使用指定的移动平均模式和长度
    efi = ma(mamode, pv_diff, length=length)

    # 偏移
    # 如果偏移量不为零，则对 EFI 进行偏移
    if offset != 0:
        efi = efi.shift(offset)

    # 处理填充
    # 如果 kwargs 中包含 'fillna'，则用指定值填充 EFI 的缺失值
    if "fillna" in kwargs:
        efi.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 'fill_method'，则使用指定的填充方法填充 EFI 的缺失值
    if "fill_method" in kwargs:
        efi.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    # 将 EFI 的名称设为字符串模板 "EFI_{length}"
    efi.name = f"EFI_{length}"
    # 将 EFI 的分类设为 "volume"
    efi.category = "volume"

    # 返回 EFI
    return efi


# 设置 EFI 函数的文档字符串
efi.__doc__ = \
"""Elder's Force Index (EFI)

Elder's Force Index measures the power behind a price movement using price
and volume as well as potential reversals and price corrections.

Sources:
    https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
    https://www.motivewave.com/studies/elders_force_index.htm

Calculation:
    Default Inputs:
        length=20, drift=1, mamode=None
    EMA = Exponential Moving Average
    SMA = Simple Moving Average

    pv_diff = close.diff(drift) * volume
    if mamode == 'sma':
        EFI = SMA(pv_diff, length)
    else:
        EFI = EMA(pv_diff, length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period. Default: 13
    drift (int): The diff period. Default: 1
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```