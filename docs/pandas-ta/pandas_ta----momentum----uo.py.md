# `.\pandas-ta\pandas_ta\momentum\uo.py`

```
# -*- coding: utf-8 -*-

# 导入 DataFrame 类
from pandas import DataFrame
# 导入 Imports 类
from pandas_ta import Imports
# 导入 get_drift 和 get_offset 函数
from pandas_ta.utils import get_drift, get_offset, verify_series


# 定义 Ultimate Oscillator（UO）指标函数
def uo(high, low, close, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: Ultimate Oscillator (UO)"""
    # 验证参数
    fast = int(fast) if fast and fast > 0 else 7
    fast_w = float(fast_w) if fast_w and fast_w > 0 else 4.0
    medium = int(medium) if medium and medium > 0 else 14
    medium_w = float(medium_w) if medium_w and medium_w > 0 else 2.0
    slow = int(slow) if slow and slow > 0 else 28
    slow_w = float(slow_w) if slow_w and slow_w > 0 else 1.0
    _length = max(fast, medium, slow)
    # 验证 high、low、close 序列长度
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    # 获取漂移和偏移量
    drift = get_drift(drift)
    offset = get_offset(offset)
    # 设置是否使用 talib 模式
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # 如果 high、low、close 有任何一个为 None，则返回空
    if high is None or low is None or close is None: return

    # 计算结果
    if Imports["talib"] and mode_tal:
        # 使用 talib 计算 UO 指标
        from talib import ULTOSC
        uo = ULTOSC(high, low, close, fast, medium, slow)
    else:
        # 否则，使用自定义计算方法
        tdf = DataFrame({
            "high": high,
            "low": low,
            f"close_{drift}": close.shift(drift)
        })
        # 获取最大最小值
        max_h_or_pc = tdf.loc[:, ["high", f"close_{drift}"]].max(axis=1)
        min_l_or_pc = tdf.loc[:, ["low", f"close_{drift}"]].min(axis=1)
        del tdf

        # 计算 buying pressure（bp）和 true range（tr）
        bp = close - min_l_or_pc
        tr = max_h_or_pc - min_l_or_pc

        # 计算 fast、medium、slow 平均值
        fast_avg = bp.rolling(fast).sum() / tr.rolling(fast).sum()
        medium_avg = bp.rolling(medium).sum() / tr.rolling(medium).sum()
        slow_avg = bp.rolling(slow).sum() / tr.rolling(slow).sum()

        # 计算总权重和加权平均值
        total_weight = fast_w + medium_w + slow_w
        weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
        uo = 100 * weights / total_weight

    # 考虑偏移量
    if offset != 0:
        uo = uo.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        uo.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        uo.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和分类
    uo.name = f"UO_{fast}_{medium}_{slow}"
    uo.category = "momentum"

    # 返回 Ultimate Oscillator（UO）指标
    return uo


# 设置 Ultimate Oscillator（UO）指标的文档字符串
uo.__doc__ = \
"""Ultimate Oscillator (UO)

The Ultimate Oscillator is a momentum indicator over three different
periods. It attempts to correct false divergence trading signals.

Sources:
    https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)

Calculation:
    Default Inputs:
        fast=7, medium=14, slow=28,
        fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1
    min_low_or_pc  = close.shift(drift).combine(low, min)
    max_high_or_pc = close.shift(drift).combine(high, max)

    bp = buying pressure = close - min_low_or_pc
    tr = true range = max_high_or_pc - min_low_or_pc

    fast_avg = SUM(bp, fast) / SUM(tr, fast)
"""
    # 计算中等速度的平均值，中等速度报告的总和除以中等速度报告的数量
    medium_avg = SUM(bp, medium) / SUM(tr, medium)
    
    # 计算慢速度的平均值，慢速度报告的总和除以慢速度报告的数量
    slow_avg = SUM(bp, slow) / SUM(tr, slow)
    
    # 计算所有速度权重的总和
    total_weight = fast_w + medium_w + slow_w
    
    # 计算加权平均值，每个速度报告的权重乘以其对应的平均值，然后相加
    weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
    
    # 计算不确定性指标（UO），即权重总和乘以100除以所有权重的总和
    UO = 100 * weights / total_weight
# 参数说明：
# high (pd.Series): 'high' 数据序列
# low (pd.Series): 'low' 数据序列
# close (pd.Series): 'close' 数据序列
# fast (int): 快速 %K 周期。默认值：7
# medium (int): 慢速 %K 周期。默认值：14
# slow (int): 慢速 %D 周期。默认值：28
# fast_w (float): 快速 %K 周期。默认值：4.0
# medium_w (float): 慢速 %K 周期。默认值：2.0
# slow_w (float): 慢速 %D 周期。默认值：1.0
# talib (bool): 如果安装了 TA Lib 并且 talib 为 True，则返回 TA Lib 版本。默认值：True
# drift (int): 差异周期。默认值：1
# offset (int): 结果的偏移周期数。默认值：0

# 可选参数：
# fillna (value, optional): pd.DataFrame.fillna(value) 的填充值
# fill_method (value, optional): 填充方法的类型

# 返回值：
# pd.Series: 生成的新特征序列
```