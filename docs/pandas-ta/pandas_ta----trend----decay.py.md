# `.\pandas-ta\pandas_ta\trend\decay.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 导入 exp 函数并重命名为 npExp
from numpy import exp as npExp
# 从 pandas 导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.utils 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


def decay(close, kind=None, length=None, mode=None, offset=None, **kwargs):
    """Indicator: Decay"""
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数，否则设置为默认值 5
    length = int(length) if length and length > 0 else 5
    # 如果 mode 是字符串，则转换为小写，否则设置为默认值 "linear"
    mode = mode.lower() if isinstance(mode, str) else "linear"
    # 验证 close 是否是有效的 Series，长度为 length
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    # 默认模式为线性模式
    _mode = "L"
    # 如果 mode 是 "exp" 或 kind 是 "exponential"，则使用指数模式
    if mode == "exp" or kind == "exponential":
        _mode = "EXP"
        # 计算差异，利用指数函数 exp(-length)
        diff = close.shift(1) - npExp(-length)
    else:  # 默认为 "linear"
        # 计算差异，利用线性函数 (1 / length)
        diff = close.shift(1) - (1 / length)
    # 将第一个元素设置为 close 的第一个值
    diff[0] = close[0]
    # 创建 DataFrame，包含 close、diff 和 0 列
    tdf = DataFrame({"close": close, "diff": diff, "0": 0})
    # 计算最大值
    ld = tdf.max(axis=1)

    # 偏移结果
    if offset != 0:
        ld = ld.shift(offset)

    # 处理填充
    # 如果 kwargs 中包含 "fillna"，则使用指定值填充缺失值
    if "fillna" in kwargs:
        ld.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 "fill_method"，则使用指定的填充方法
    if "fill_method" in kwargs:
        ld.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    # 设置 Series 的名称为模式和长度的组合
    ld.name = f"{_mode}DECAY_{length}"
    # 设置 Series 的分类为 "trend"
    ld.category = "trend"

    return ld


# 设置 decay 函数的文档字符串
decay.__doc__ = \
"""Decay

Creates a decay moving forward from prior signals like crosses. The default is
"linear". Exponential is optional as "exponential" or "exp".

Sources:
    https://tulipindicators.org/decay

Calculation:
    Default Inputs:
        length=5, mode=None

    if mode == "exponential" or mode == "exp":
        max(close, close[-1] - exp(-length), 0)
    else:
        max(close, close[-1] - (1 / length), 0)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 1
    mode (str): If 'exp' then "exponential" decay. Default: 'linear'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```