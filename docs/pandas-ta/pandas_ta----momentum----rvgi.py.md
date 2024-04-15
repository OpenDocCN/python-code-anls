# `.\pandas-ta\pandas_ta\momentum\rvgi.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.overlap 模块中导入 swma 函数
from pandas_ta.overlap import swma
# 从 pandas_ta.utils 模块中导入 get_offset、non_zero_range、verify_series 函数
from pandas_ta.utils import get_offset, non_zero_range, verify_series


# 定义 RVGI 函数，计算相对活力指数
def rvgi(open_, high, low, close, length=None, swma_length=None, offset=None, **kwargs):
    """Indicator: Relative Vigor Index (RVGI)"""
    # 验证参数
    # 计算 high 和 low 之间的范围，不为零
    high_low_range = non_zero_range(high, low)
    # 计算 close 和 open 之间的范围，不为零
    close_open_range = non_zero_range(close, open_)
    # 将 length 转换为整数，如果 length 不存在或小于等于 0，则默认为 14
    length = int(length) if length and length > 0 else 14
    # 将 swma_length 转换为整数，如果 swma_length 不存在或小于等于 0，则默认为 4
    swma_length = int(swma_length) if swma_length and swma_length > 0 else 4
    # 计算最大长度，取 length 和 swma_length 中的最大值
    _length = max(length, swma_length)
    # 验证 open_、high、low、close，使其长度为 _length
    open_ = verify_series(open_, _length)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    # 获取偏移量
    offset = get_offset(offset)

    # 如果 open_、high、low、close 中有任何一个为 None，则返回
    if open_ is None or high is None or low is None or close is None: return

    # 计算结果
    # 计算分子，为 close_open_range 的 swma，长度为 swma_length 的滚动和
    numerator = swma(close_open_range, length=swma_length).rolling(length).sum()
    # 计算分母，为 high_low_range 的 swma，长度为 swma_length 的滚动和
    denominator = swma(high_low_range, length=swma_length).rolling(length).sum()

    # 计算 RVGI，为分子除以分母
    rvgi = numerator / denominator
    # 计算信号线，为 RVGI 的 swma，长度为 swma_length
    signal = swma(rvgi, length=swma_length)

    # 偏移结果
    if offset != 0:
        rvgi = rvgi.shift(offset)
        signal = signal.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        rvgi.fillna(kwargs["fillna"], inplace=True)
        signal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rvgi.fillna(method=kwargs["fill_method"], inplace=True)
        signal.fillna(method=kwargs["fill_method"], inplace=True)

    # 名称和类别
    # 设置 RVGI 的名称为 "RVGI_length_swma_length"，设置 signal 的名称为 "RVGIs_length_swma_length"
    rvgi.name = f"RVGI_{length}_{swma_length}"
    signal.name = f"RVGIs_{length}_{swma_length}"
    # 设置 RVGI 和 signal 的类别为 "momentum"
    rvgi.category = signal.category = "momentum"

    # 准备返回的 DataFrame
    # 创建 DataFrame，包含 RVGI 和 signal，列名为其名称
    df = DataFrame({rvgi.name: rvgi, signal.name: signal})
    # 设置 DataFrame 的名称为 "RVGI_length_swma_length"
    df.name = f"RVGI_{length}_{swma_length}"
    # 设置 DataFrame 的类别为 RVGI 的类别
    df.category = rvgi.category

    return df


# 设置 RVGI 函数的文档字符串
rvgi.__doc__ = \
"""Relative Vigor Index (RVGI)

The Relative Vigor Index attempts to measure the strength of a trend relative to
its closing price to its trading range.  It is based on the belief that it tends
to close higher than they open in uptrends or close lower than they open in
downtrends.

Sources:
    https://www.investopedia.com/terms/r/relative_vigor_index.asp

Calculation:
    Default Inputs:
        length=14, swma_length=4
    SWMA = Symmetrically Weighted Moving Average
    numerator = SUM(SWMA(close - open, swma_length), length)
    denominator = SUM(SWMA(high - low, swma_length), length)
    RVGI = numerator / denominator

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    swma_length (int): It's period. Default: 4
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
"""
    # fill_method 是一个可选参数，用于指定填充方法的类型
# 返回一个 Pandas Series 对象，表示生成的新特征
Returns:
    pd.Series: New feature generated.
```