# `.\pandas-ta\pandas_ta\overlap\jma.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库中导入 average 函数并重命名为 npAverage
# 从 numpy 库中导入 nan 函数并重命名为 npNaN
# 从 numpy 库中导入 log 函数并重命名为 npLog
# 从 numpy 库中导入 power 函数并重命名为 npPower
# 从 numpy 库中导入 sqrt 函数并重命名为 npSqrt
# 从 numpy 库中导入 zeros_like 函数并重命名为 npZeroslike
# 从 pandas 库中导入 Series 类
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数

def jma(close, length=None, phase=None, offset=None, **kwargs):
    """Indicator: Jurik Moving Average (JMA)"""
    # 验证参数
    _length = int(length) if length and length > 0 else 7
    phase = float(phase) if phase and phase != 0 else 0
    close = verify_series(close, _length)
    offset = get_offset(offset)
    if close is None: return

    # 定义基本变量
    jma = npZeroslike(close)
    volty = npZeroslike(close)
    v_sum = npZeroslike(close)

    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = close[0]

    # 静态变量
    sum_length = 10
    length = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((npLog(npSqrt(length)) / npLog(2.0)) + 2.0, 0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * npSqrt(length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)

    m = close.shape[0]
    for i in range(1, m):
        price = close[i]

        # 价格波动性
        del1 = price - uBand
        del2 = price - lBand
        volty[i] = max(abs(del1),abs(del2)) if abs(del1)!=abs(del2) else 0

        # 相对价格波动性因子
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[max(i - sum_length, 0)]) / sum_length
        avg_volty = npAverage(v_sum[max(i - 65, 0):i + 1])
        d_volty = 0 if avg_volty ==0 else volty[i] / avg_volty
        r_volty = max(1.0, min(npPower(length1, 1 / pow1), d_volty))

        # Jurik 波动性带
        pow2 = npPower(r_volty, pow1)
        kv = npPower(bet, npSqrt(pow2))
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Jurik 动态因子
        power = npPower(r_volty, pow1)
        alpha = npPower(beta, power)

        # 第一阶段 - 通过自适应 EMA 进行初步平滑
        ma1 = ((1 - alpha) * price) + (alpha * ma1)

        # 第二阶段 - 通过 Kalman 滤波器进行一次额外的初步平滑
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0

        # 第三阶段 - 通过独特的 Jurik 自适应滤波器进行最终平滑
        det1 = ((ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha)) + (alpha * alpha * det1)
        jma[i] = jma[i-1] + det1

    # 移除初始回看数据并转换为 pandas Series
    jma[0:_length - 1] = npNaN
    jma = Series(jma, index=close.index)

    # 偏移
    if offset != 0:
        jma = jma.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        jma.fillna(kwargs["fillna"], inplace=True)
    # 检查是否存在 "fill_method" 参数在传入的关键字参数 kwargs 中
    if "fill_method" in kwargs:
        # 如果存在，使用指定的填充方法对 DataFrame 进行填充，并在原地修改
        jma.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置 JMA 对象的名称为格式化后的字符串，包含长度和相位信息
    jma.name = f"JMA_{_length}_{phase}"
    # 设置 JMA 对象的类别为 "overlap"
    jma.category = "overlap"

    # 返回填充后的 JMA 对象
    return jma
# 将 jma 的文档字符串设置为 JMA 指标的说明文档
jma.__doc__ = \
"""Jurik Moving Average Average (JMA)

Mark Jurik's Moving Average (JMA) attempts to eliminate noise to see the "true"
underlying activity. It has extremely low lag, is very smooth and is responsive
to market gaps.

Sources:
    https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    https://www.prorealcode.com/prorealtime-indicators/jurik-volatility-bands/

Calculation:
    Default Inputs:
        length=7, phase=0

Args:
    close (pd.Series): Series of 'close's
    length (int): Period of calculation. Default: 7
    phase (float): How heavy/light the average is [-100, 100]. Default: 0
    offset (int): How many lengths to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```