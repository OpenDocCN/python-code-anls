# `.\pandas-ta\pandas_ta\momentum\brar.py`

```py
# 设置文件编码为 UTF-8
# 导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta.utils 中导入函数 get_drift, get_offset, non_zero_range, verify_series
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

# 定义函数 brar，计算 BRAR 指标
def brar(open_, high, low, close, length=None, scalar=None, drift=None, offset=None, **kwargs):
    """Indicator: BRAR (BRAR)"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则默认为 26
    length = int(length) if length and length > 0 else 26
    # 如果 scalar 存在，则将其转换为浮点数，否则默认为 100
    scalar = float(scalar) if scalar else 100
    # 计算 high 与 open 的非零范围
    high_open_range = non_zero_range(high, open_)
    # 计算 open 与 low 的非零范围
    open_low_range = non_zero_range(open_, low)
    # 验证输入的数据列，并截取长度为 length
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    # 获取漂移值
    drift = get_drift(drift)
    # 获取偏移值
    offset = get_offset(offset)

    # 如果任何输入数据为空，则返回空
    if open_ is None or high is None or low is None or close is None: return

    # 计算结果
    # 计算 high_close_yesterday，即 high 与 close 的差值
    hcy = non_zero_range(high, close.shift(drift))
    # 计算 close_yesterday_low，即 close 与 low 的差值
    cyl = non_zero_range(close.shift(drift), low)
    # 将负值替换为零
    hcy[hcy < 0] = 0
    cyl[cyl < 0] = 0

    # 计算 AR 和 BR 指标
    # AR = scalar * HO 的长度为 length 的滚动和 / OL 的长度为 length 的滚动和
    ar = scalar * high_open_range.rolling(length).sum()
    ar /= open_low_range.rolling(length).sum()
    # BR = scalar * HCY 的长度为 length 的滚动和 / CYL 的长度为 length 的滚动和
    br = scalar * hcy.rolling(length).sum()
    br /= cyl.rolling(length).sum()

    # 偏移
    if offset != 0:
        ar = ar.shift(offset)
        br = ar.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        ar.fillna(kwargs["fillna"], inplace=True)
        br.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ar.fillna(method=kwargs["fill_method"], inplace=True)
        br.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置指标名称和类别
    _props = f"_{length}"
    ar.name = f"AR{_props}"
    br.name = f"BR{_props}"
    ar.category = br.category = "momentum"

    # 准备返回的 DataFrame
    brardf = DataFrame({ar.name: ar, br.name: br})
    brardf.name = f"BRAR{_props}"
    brardf.category = "momentum"

    return brardf

# 设置 brar 函数的文档字符串
brar.__doc__ = \
"""BRAR (BRAR)

BR and AR

Sources:
    No internet resources on definitive definition.
    Request by Github user homily, issue #46

Calculation:
    Default Inputs:
        length=26, scalar=100
    SUM = Sum

    HO_Diff = high - open
    OL_Diff = open - low
    HCY = high - close[-1]
    CYL = close[-1] - low
    HCY[HCY < 0] = 0
    CYL[CYL < 0] = 0
    AR = scalar * SUM(HO, length) / SUM(OL, length)
    BR = scalar * SUM(HCY, length) / SUM(CYL, length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The period. Default: 26
    scalar (float): How much to magnify. Default: 100
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ar, br columns.
"""
```