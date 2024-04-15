# `.\pandas-ta\pandas_ta\momentum\inertia.py`

```
# 设置文件编码为 utf-8
# 从 pandas_ta.overlap 模块导入 linreg 函数
# 从 pandas_ta.volatility 模块导入 rvi 函数
# 从 pandas_ta.utils 模块导入 get_drift, get_offset, verify_series 函数
def inertia(close=None, high=None, low=None, length=None, rvi_length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Inertia (INERTIA)"""
    # 验证参数
   # 如果 length 存在且大于 0，则将其转换为整数，否则设为 20
    length = int(length) if length and length > 0 else 20
   # 如果 rvi_length 存在且大于 0，则将其转换为整数，否则设为 14
    rvi_length = int(rvi_length) if rvi_length and rvi_length > 0 else 14
   # 如果 scalar 存在且大于 0，则将其转换为浮点数，否则设为 100
    scalar = float(scalar) if scalar and scalar > 0 else 100
   # 如果 refined 为 None，则设为 False
    refined = False if refined is None else True
   # 如果 thirds 为 None，则设为 False
    thirds = False if thirds is None else True
   # 如果 mamode 不是字符串，则设为 "ema"
    mamode = mamode if isinstance(mamode, str) else "ema"
   # 选择 length 和 rvi_length 中的最大值
    _length = max(length, rvi_length)
   # 验证 close 序列
    close = verify_series(close, _length)
   # 获取 drift 值
    drift = get_drift(drift)
   # 获取 offset 值
    offset = get_offset(offset)

    # 如果 close 为 None，则返回
    if close is None: return

    # 如果 refined 或 thirds 为 True
    if refined or thirds:
        # 验证 high 和 low 序列
        high = verify_series(high, _length)
        low = verify_series(low, _length)
        # 如果 high 或 low 为 None，则返回
        if high is None or low is None: return

    # 计算结果
    if refined:
        # 使用 'r' 模式计算 rvi
        _mode, rvi_ = "r", rvi(close, high=high, low=low, length=rvi_length, scalar=scalar, refined=refined, mamode=mamode)
    elif thirds:
        # 使用 't' 模式计算 rvi
        _mode, rvi_ = "t", rvi(close, high=high, low=low, length=rvi_length, scalar=scalar, thirds=thirds, mamode=mamode)
    else:
        # 计算 rvi
        _mode, rvi_ = "",  rvi(close, length=rvi_length, scalar=scalar, mamode=mamode)

    # 计算惯性
    inertia = linreg(rvi_, length=length)

    # 偏移
    if offset != 0:
        inertia = inertia.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        inertia.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        inertia.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和类别
    _props = f"_{length}_{rvi_length}"
    inertia.name = f"INERTIA{_mode}{_props}"
    inertia.category = "momentum"

    return inertia

# 设置 inertia 函数的文档字符串
inertia.__doc__ = \
"""Inertia (INERTIA)

Inertia was developed by Donald Dorsey and was introduced his article
in September, 1995. It is the Relative Vigor Index smoothed by the Least
Squares Moving Average. Postive Inertia when values are greater than 50,
Negative Inertia otherwise.

Sources:
    https://www.investopedia.com/terms/r/relative_vigor_index.asp

Calculation:
    Default Inputs:
        length=14, ma_length=20
    LSQRMA = Least Squares Moving Average

    INERTIA = LSQRMA(RVI(length), ma_length)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 20
    rvi_length (int): RVI period. Default: 14
    refined (bool): Use 'refined' calculation. Default: False
    thirds (bool): Use 'thirds' calculation. Default: False
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    drift (int): The difference period. Default: 1

"""
    offset (int): How many periods to offset the result. Default: 0

这是一个函数参数的说明。`offset` 是一个整数类型的参数，表示结果要偏移的周期数，默认为0。
# 接收关键字参数
Kwargs:
    # fillna 参数用于填充缺失值，可选参数，传入的 value 将用于填充 DataFrame 中的缺失值
    fillna (value, optional): pd.DataFrame.fillna(value)
    # fill_method 参数指定填充的方法，可选参数
    fill_method (value, optional): Type of fill method

# 返回一个 Pandas Series 对象
Returns:
    # 返回一个由该函数生成的新特征
    pd.Series: New feature generated.
```