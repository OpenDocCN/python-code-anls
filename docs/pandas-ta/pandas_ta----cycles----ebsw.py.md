# `.\pandas-ta\pandas_ta\cycles\ebsw.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 模块导入 cos 函数并命名为 npCos
from numpy import cos as npCos
# 从 numpy 模块导入 exp 函数并命名为 npExp
from numpy import exp as npExp
# 从 numpy 模块导入 nan 常量并命名为 npNaN
from numpy import nan as npNaN
# 从 numpy 模块导入 pi 常量并命名为 npPi
from numpy import pi as npPi
# 从 numpy 模块导入 sin 函数并命名为 npSin
from numpy import sin as npSin
# 从 numpy 模块导入 sqrt 函数并命名为 npSqrt
from numpy import sqrt as npSqrt
# 从 pandas 模块中导入 Series 类
from pandas import Series
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义函数 ebsw，用于计算 Even Better SineWave (EBSW) 指标
def ebsw(close, length=None, bars=None, offset=None, **kwargs):
    """Indicator: Even Better SineWave (EBSW)"""
    # 校验参数
    # 如果 length 存在且大于38，则转换为整数，否则默认为40
    length = int(length) if length and length > 38 else 40
    # 如果 bars 存在且大于0，则转换为整数，否则默认为10
    bars = int(bars) if bars and bars > 0 else 10
    # 验证 close 是否为有效 Series 对象
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回 None
    if close is None: return

    # 初始化变量
    alpha1 = HP = 0  # alpha 和 HighPass
    a1 = b1 = c1 = c2 = c3 = 0
    Filt = Pwr = Wave = 0

    lastClose = lastHP = 0
    FilterHist = [0, 0]  # 过滤器历史记录

    # 计算结果
    m = close.size
    result = [npNaN for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        # 使用周期为 length 的高通滤波器过滤短于 Duration 输入的周期成分
        alpha1 = (1 - npSin(360 / length)) / npCos(360 / length)
        HP = 0.5 * (1 + alpha1) * (close[i] - lastClose) + alpha1 * lastHP

        # 使用超级平滑滤波器平滑数据（方程 3-3）
        a1 = npExp(-npSqrt(2) * npPi / bars)
        b1 = 2 * a1 * npCos(npSqrt(2) * 180 / bars)
        c2 = b1
        c3 = -1 * a1 * a1
        c1 = 1 - c2 - c3
        Filt = c1 * (HP + lastHP) / 2 + c2 * FilterHist[1] + c3 * FilterHist[0]

        # 计算波动和功率的3根均线
        Wave = (Filt + FilterHist[1] + FilterHist[0]) / 3
        Pwr = (Filt * Filt + FilterHist[1] * FilterHist[1] + FilterHist[0] * FilterHist[0]) / 3

        # 将平均波动归一化到平均功率的平方根
        Wave = Wave / npSqrt(Pwr)

        # 更新存储和结果
        FilterHist.append(Filt)  # 添加新的 Filt 值
        FilterHist.pop(0)  # 移除列表中的第一个元素（最早的）-> 更新/修剪
        lastHP = HP
        lastClose = close[i]
        result.append(Wave)

    # 创建结果 Series 对象
    ebsw = Series(result, index=close.index)

    # 如果有偏移量，则进行偏移
    if offset != 0:
        ebsw = ebsw.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        ebsw.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ebsw.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    ebsw.name = f"EBSW_{length}_{bars}"
    ebsw.category = "cycles"

    return ebsw


# 设置函数 ebsw 的文档字符串
ebsw.__doc__ = \
"""Even Better SineWave (EBSW) *beta*

This indicator measures market cycles and uses a low pass filter to remove noise.
Its output is bound signal between -1 and 1 and the maximum length of a detected
trend is limited by its length input.

Written by rengel8 for Pandas TA based on a publication at 'prorealcode.com' and
a book by J.F.Ehlers.
"""
# 这个实现在逻辑上似乎有限制。最好实现与prorealcode中的版本完全相同，并比较行为。

来源：
    https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
    J.F.Ehlers的《Cycle Analytics for Traders》，2014

计算：
    参考'sources'或实现

参数：
    close (pd.Series): 'close'的系列
    length (int): 最大周期/趋势周期。值在40-48之间的效果如预期，最小值为39。默认值：40。
    bars (int): 低通滤波的周期。默认值：10
    drift (int): 差异周期。默认值：1
    offset (int): 结果的偏移周期数。默认值：0

关键字参数：
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): 填充方法的类型

返回：
    pd.Series: 生成的新特征。
```