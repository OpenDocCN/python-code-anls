# `.\pandas-ta\pandas_ta\utils\_core.py`

```
# 设置文件编码为 utf-8
# 导入 re 模块并重命名为 re_
# 从 pathlib 模块中导入 Path 类
# 从 sys 模块中导入 float_info 并重命名为 sflt
# 从 numpy 模块中导入 argmax 和 argmin 函数
# 从 pandas 模块中导入 DataFrame 和 Series 类
# 从 pandas 模块中导入 is_datetime64_any_dtype 函数
# 从 pandas_ta 模块中导入 Imports

def _camelCase2Title(x: str):
    """将驼峰命名转换为标题格式"""
    return re_.sub("([a-z])([A-Z])","\g<1> \g<2>", x).title()

def category_files(category: str) -> list:
    """返回类别目录中所有文件名的帮助函数"""
    files = [
        x.stem
        for x in list(Path(f"pandas_ta/{category}/").glob("*.py"))
        if x.stem != "__init__"
    ]
    return files

def get_drift(x: int) -> int:
    """如果不为零，则返回一个整数，否则默认为一"""
    return int(x) if isinstance(x, int) and x != 0 else 1

def get_offset(x: int) -> int:
    """返回一个整数，否则默认为零"""
    return int(x) if isinstance(x, int) else 0

def is_datetime_ordered(df: DataFrame or Series) -> bool:
    """如果索引是日期时间且有序，则返回 True"""
    index_is_datetime = is_datetime64_any_dtype(df.index)
    try:
        ordered = df.index[0] < df.index[-1]
    except RuntimeWarning:
        pass
    finally:
        return True if index_is_datetime and ordered else False

def is_percent(x: int or float) -> bool:
    """检查是否为百分比"""
    if isinstance(x, (int, float)):
        return x is not None and x >= 0 and x <= 100
    return False

def non_zero_range(high: Series, low: Series) -> Series:
    """返回两个序列的差异，并对任何零值添加 epsilon。在加密数据中常见情况是 'high' = 'low'。"""
    diff = high - low
    if diff.eq(0).any().any():
        diff += sflt.epsilon
    return diff

def recent_maximum_index(x):
    """返回最近最大值的索引"""
    return int(argmax(x[::-1]))

def recent_minimum_index(x):
    """返回最近最小值的索引"""
    return int(argmin(x[::-1]))

def signed_series(series: Series, initial: int = None) -> Series:
    """返回带有或不带有初始值的有符号序列"""
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign

def tal_ma(name: str) -> int:
    """返回 TA Lib 的 MA 类型的枚举值"""
    # 检查是否导入了 talib 模块，并且 name 是否是字符串类型，并且长度大于1
    if Imports["talib"] and isinstance(name, str) and len(name) > 1:
        # 如果满足条件，从 talib 模块导入 MA_Type
        from talib import MA_Type
        # 将 name 转换为小写
        name = name.lower()
        # 根据 name 的不同取值返回对应的 MA_Type 枚举值
        if   name == "sma":   return MA_Type.SMA   # 0
        elif name == "ema":   return MA_Type.EMA   # 1
        elif name == "wma":   return MA_Type.WMA   # 2
        elif name == "dema":  return MA_Type.DEMA  # 3
        elif name == "tema":  return MA_Type.TEMA  # 4
        elif name == "trima": return MA_Type.TRIMA # 5
        elif name == "kama":  return MA_Type.KAMA  # 6
        elif name == "mama":  return MA_Type.MAMA  # 7
        elif name == "t3":    return MA_Type.T3    # 8
    # 如果不满足条件，返回默认值 0，代表 SMA
    return 0 # Default: SMA -> 0
# 定义一个函数，计算给定 Series 的无符号差值
def unsigned_differences(series: Series, amount: int = None, **kwargs) -> Series:
    """Unsigned Differences
    返回两个 Series，一个是原始 Series 的无符号正差值，另一个是无符号负差值。
    正差值 Series 仅包含增加值，负差值 Series 仅包含减少值。

    默认示例：
    series   = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5, 3]) 返回
    positive  = Series([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    negative = Series([0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    """
    # 如果未提供 amount 参数，则默认为 1
    amount = int(amount) if amount is not None else 1
    # 计算 Series 的负差值
    negative = series.diff(amount)
    # 将 NaN 值填充为 0
    negative.fillna(0, inplace=True)
    # 复制负差值 Series 以备后用
    positive = negative.copy()

    # 将正差值 Series 中小于等于 0 的值设置为 0
    positive[positive <= 0] = 0
    # 将正差值 Series 中大于 0 的值设置为 1
    positive[positive > 0] = 1

    # 将负差值 Series 中大于等于 0 的值设置为 0
    negative[negative >= 0] = 0
    # 将负差值 Series 中小于 0 的值设置为 1
    negative[negative < 0] = 1

    # 如果 kwargs 中包含 asint 参数且值为 True，则将 Series 转换为整数类型
    if kwargs.pop("asint", False):
        positive = positive.astype(int)
        negative = negative.astype(int)

    # 返回正差值 Series 和负差值 Series
    return positive, negative


# 定义一个函数，验证给定的 Series 是否满足指示器的最小长度要求
def verify_series(series: Series, min_length: int = None) -> Series:
    """If a Pandas Series and it meets the min_length of the indicator return it."""
    # 判断是否指定了最小长度，并且最小长度是整数类型
    has_length = min_length is not None and isinstance(min_length, int)
    # 如果给定的 series 不为空且是 Pandas Series 类型
    if series is not None and isinstance(series, Series):
        # 如果指定了最小长度，并且 series 的大小小于最小长度，则返回 None，否则返回 series
        return None if has_length and series.size < min_length else series
```