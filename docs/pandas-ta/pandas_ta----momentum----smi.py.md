# `.\pandas-ta\pandas_ta\momentum\smi.py`

```py
# -*- coding: utf-8 -*-                             # 指定编码格式为 UTF-8

# 导入必要的模块和函数
from pandas import DataFrame                       
from .tsi import tsi                                 # 从当前目录下的 tsi 模块中导入 tsi 函数
from pandas_ta.overlap import ema                    # 从 pandas_ta 库中的 overlap 模块导入 ema 函数
from pandas_ta.utils import get_offset, verify_series  # 从 pandas_ta 库中的 utils 模块导入 get_offset 和 verify_series 函数


def smi(close, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs):
    """Indicator: SMI Ergodic Indicator (SMIIO)"""   # 定义函数 smi，计算 SMI 指标
    # 验证参数
    fast = int(fast) if fast and fast > 0 else 5     # 如果 fast 存在且大于 0，则将其转换为整数类型，否则设为默认值 5
    slow = int(slow) if slow and slow > 0 else 20    # 如果 slow 存在且大于 0，则将其转换为整数类型，否则设为默认值 20
    signal = int(signal) if signal and signal > 0 else 5  # 如果 signal 存在且大于 0，则将其转换为整数类型，否则设为默认值 5
    if slow < fast:                                  # 如果 slow 小于 fast，则交换两者的值
        fast, slow = slow, fast
    scalar = float(scalar) if scalar else 1          # 如果 scalar 存在，则转换为浮点数，否则设为默认值 1
    close = verify_series(close, max(fast, slow, signal))  # 验证 close 数据，保证长度足够
    offset = get_offset(offset)                      # 获取偏移量

    if close is None: return                        # 如果 close 为空，则返回

    # 计算结果
    tsi_df = tsi(close, fast=fast, slow=slow, signal=signal, scalar=scalar)  # 调用 tsi 函数计算 TSI 指标
    smi = tsi_df.iloc[:, 0]                         # 获取 TSI 列
    signalma = tsi_df.iloc[:, 1]                    # 获取信号线列
    osc = smi - signalma                            # 计算 SMI 指标的震荡值

    # 偏移
    if offset != 0:                                 # 如果偏移量不为 0
        smi = smi.shift(offset)                     # 对 SMI 进行偏移
        signalma = signalma.shift(offset)           # 对信号线进行偏移
        osc = osc.shift(offset)                     # 对震荡值进行偏移

    # 填充缺失值
    if "fillna" in kwargs:                          # 如果参数中包含 "fillna"
        smi.fillna(kwargs["fillna"], inplace=True)  # 使用指定的填充值填充缺失值
        signalma.fillna(kwargs["fillna"], inplace=True)
        osc.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:                     # 如果参数中包含 "fill_method"
        smi.fillna(method=kwargs["fill_method"], inplace=True)  # 使用指定的填充方法填充缺失值
        signalma.fillna(method=kwargs["fill_method"], inplace=True)
        osc.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    _scalar = f"_{scalar}" if scalar != 1 else ""   # 如果 scalar 不等于 1，则添加到名称中
    _props = f"_{fast}_{slow}_{signal}{_scalar}"   # 构建指标属性字符串
    smi.name = f"SMI{_props}"                       # 设置 SMI 指标名称
    signalma.name = f"SMIs{_props}"                 # 设置信号线指标名称
    osc.name = f"SMIo{_props}"                      # 设置震荡值指标名称
    smi.category = signalma.category = osc.category = "momentum"  # 设置指标分类为动量型

    # 准备要返回的 DataFrame
    data = {smi.name: smi, signalma.name: signalma, osc.name: osc}  # 构建包含指标数据的字典
    df = DataFrame(data)                            # 构建 DataFrame
    df.name = f"SMI{_props}"                        # 设置 DataFrame 名称
    df.category = smi.category                      # 设置 DataFrame 分类与指标相同

    return df                                       # 返回 DataFrame


smi.__doc__ = \
"""SMI Ergodic Indicator (SMI)

The SMI Ergodic Indicator is the same as the True Strength Index (TSI) developed
by William Blau, except the SMI includes a signal line. The SMI uses double
moving averages of price minus previous price over 2 time frames. The signal
line, which is an EMA of the SMI, is plotted to help trigger trading signals.
The trend is bullish when crossing above zero and bearish when crossing below
zero. This implementation includes both the SMI Ergodic Indicator and SMI
Ergodic Oscillator.

Sources:
    https://www.motivewave.com/studies/smi_ergodic_indicator.htm
    https://www.tradingview.com/script/Xh5Q0une-SMI-Ergodic-Oscillator/
    https://www.tradingview.com/script/cwrgy4fw-SMIIO/

Calculation:
    Default Inputs:
        fast=5, slow=20, signal=5
    TSI = True Strength Index
    EMA = Exponential Moving Average

    ERG = TSI(close, fast, slow)
    Signal = EMA(ERG, signal)
    OSC = ERG - Signal

Args:
    close (pd.Series): Series of 'close's

"""                                                 # 设置 smi 函数的文档字符串
    # 快速线的周期，默认为5
    fast (int): The short period. Default: 5
    # 慢速线的周期，默认为20
    slow (int): The long period. Default: 20
    # 信号线的周期，默认为5
    signal (int): The signal period. Default: 5
    # 放大倍数，默认为1
    scalar (float): How much to magnify. Default: 1
    # 结果的偏移周期数，默认为0
    offset (int): How many periods to offset the result. Default: 0
# 参数说明部分，描述函数的参数和返回值
Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
# 返回值说明部分，描述返回的数据类型和列名
Returns:
    pd.DataFrame: smi, signal, oscillator columns.
```