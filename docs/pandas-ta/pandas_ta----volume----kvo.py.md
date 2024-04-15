# `.\pandas-ta\pandas_ta\volume\kvo.py`

```
# 设置文件编码为 UTF-8
# 导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.overlap 模块导入 hlc3 和 ma 函数
from pandas_ta.overlap import hlc3, ma
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, signed_series, verify_series 函数
from pandas_ta.utils import get_drift, get_offset, signed_series, verify_series


# 定义 Klinger Volume Oscillator (KVO) 指标函数
def kvo(high, low, close, volume, fast=None, slow=None, signal=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Klinger Volume Oscillator (KVO)"""
    # 验证参数
    # 如果 fast 存在且大于 0，则将其转换为整数；否则默认为 34
    fast = int(fast) if fast and fast > 0 else 34
    # 如果 slow 存在且大于 0，则将其转换为整数；否则默认为 55
    slow = int(slow) if slow and slow > 0 else 55
    # 如果 signal 存在且大于 0，则将其转换为整数；否则默认为 13
    signal = int(signal) if signal and signal > 0 else 13
    # 如果 mamode 存在且是字符串类型，则将其转换为小写；否则默认为 'ema'
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "ema"
    # 计算参数中最大的长度
    _length = max(fast, slow, signal)
    # 验证输入序列的长度
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果输入的 high、low、close、volume 有任何一个为 None，则返回 None
    if high is None or low is None or close is None or volume is None: return

    # 计算结果
    # 计算带符号的成交量
    signed_volume = volume * signed_series(hlc3(high, low, close), 1)
    # 从第一个有效索引开始取值
    sv = signed_volume.loc[signed_volume.first_valid_index():,]
    # 计算 KVO 指标
    kvo = ma(mamode, sv, length=fast) - ma(mamode, sv, length=slow)
    # 计算 KVO 的信号线
    kvo_signal = ma(mamode, kvo.loc[kvo.first_valid_index():,], length=signal)

    # 调整偏移
    if offset != 0:
        kvo = kvo.shift(offset)
        kvo_signal = kvo_signal.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        kvo.fillna(kwargs["fillna"], inplace=True)
        kvo_signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        kvo.fillna(method=kwargs["fill_method"], inplace=True)
        kvo_signal.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标的名称和分类
    _props = f"_{fast}_{slow}_{signal}"
    kvo.name = f"KVO{_props}"
    kvo_signal.name = f"KVOs{_props}"
    kvo.category = kvo_signal.category = "volume"

    # 准备返回的 DataFrame
    data = {kvo.name: kvo, kvo_signal.name: kvo_signal}
    df = DataFrame(data)
    df.name = f"KVO{_props}"
    df.category = kvo.category

    return df


# 设置函数文档字符串
kvo.__doc__ = \
"""Klinger Volume Oscillator (KVO)

This indicator was developed by Stephen J. Klinger. It is designed to predict
price reversals in a market by comparing volume to price.

Sources:
    https://www.investopedia.com/terms/k/klingeroscillator.asp
    https://www.daytrading.com/klinger-volume-oscillator

Calculation:
    Default Inputs:
        fast=34, slow=55, signal=13, drift=1
    EMA = Exponential Moving Average

    SV = volume * signed_series(HLC3, 1)
    KVO = EMA(SV, fast) - EMA(SV, slow)
    Signal = EMA(KVO, signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    fast (int): The fast period. Default: 34
    long (int): The long period. Default: 55
    length_sig (int): The signal period. Default: 13
    mamode (str): See ```help(ta.ma)```. Default: 'ema'

"""
    offset (int): How many periods to offset the result. Default: 0

# 偏移量（int）：结果要偏移的周期数。默认值为0。
# 参数说明部分，描述函数的参数和返回值
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
Returns:
    pd.DataFrame: KVO and Signal columns.
```