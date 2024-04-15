# `.\pandas-ta\pandas_ta\utils\_signals.py`

```
# -*- coding: utf-8 -*-
# 导入 DataFrame 和 Series 类
from pandas import DataFrame, Series

# 导入自定义函数
from ._core import get_offset, verify_series
from ._math import zero

# 定义函数 _above_below，用于比较两个 Series 对象的大小关系
def _above_below(series_a: Series, series_b: Series, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    # 确保 series_a 和 series_b 是 Series 对象
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    # 将 series_a 和 series_b 中的零值替换为 NaN
    series_a.apply(zero)
    series_b.apply(zero)

    # 计算结果
    if above:
        current = series_a >= series_b
    else:
        current = series_a <= series_b

    if asint:
        current = current.astype(int)

    # 偏移
    if offset != 0:
        current = current.shift(offset)

    # 设置名称和类别
    current.name = f"{series_a.name}_{'A' if above else 'B'}_{series_b.name}"
    current.category = "utility"

    return current

# 定义函数 above，用于比较两个 Series 对象的大小关系，series_a 大于等于 series_b
def above(series_a: Series, series_b: Series, asint: bool = True, offset: int = None, **kwargs):
    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)

# 定义函数 above_value，用于比较 Series 对象和给定值的大小关系，series_a 大于等于 value
def above_value(series_a: Series, value: float, asint: bool = True, offset: int = None, **kwargs):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return _above_below(series_a, series_b, above=True, asint=asint, offset=offset, **kwargs)

# 定义函数 below，用于比较两个 Series 对象的大小关系，series_a 小于等于 series_b
def below(series_a: Series, series_b: Series, asint: bool = True, offset: int = None, **kwargs):
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)

# 定义函数 below_value，用于比较 Series 对象和给定值的大小关系，series_a 小于等于 value
def below_value(series_a: Series, value: float, asint: bool = True, offset: int = None, **kwargs):
    if not isinstance(value, (int, float, complex)):
        print("[X] value is not a number")
        return
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))
    return _above_below(series_a, series_b, above=False, asint=asint, offset=offset, **kwargs)

# 定义函数 cross_value，用于判断 Series 对象和给定值是否交叉，above 为 True 表示交叉在上方
def cross_value(series_a: Series, value: float, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    series_b = Series(value, index=series_a.index, name=f"{value}".replace(".", "_"))

    return cross(series_a, series_b, above, asint, offset, **kwargs)

# 定义函数 cross，用于判断两个 Series 对象是否交叉，above 为 True 表示交叉在上方
def cross(series_a: Series, series_b: Series, above: bool = True, asint: bool = True, offset: int = None, **kwargs):
    series_a = verify_series(series_a)
    series_b = verify_series(series_b)
    offset = get_offset(offset)

    series_a.apply(zero)
    series_b.apply(zero)

    # 计算结果
    current = series_a > series_b  # current is above
    previous = series_a.shift(1) < series_b.shift(1)  # previous is below
    # above if both are true, below if both are false
    cross = current & previous if above else ~current & ~previous

    if asint:
        cross = cross.astype(int)

    # 偏移
    if offset != 0:
        cross = cross.shift(offset)

    # 设置名称和类别
    # 设置交叉系列的名称，根据条件选择不同的后缀
    cross.name = f"{series_a.name}_{'XA' if above else 'XB'}_{series_b.name}"
    # 设置交叉系列的类别为"utility"
    cross.category = "utility"

    # 返回交叉系列对象
    return cross
# 根据给定的指标、阈值和参数，生成包含交叉信号的数据框
def signals(indicator, xa, xb, cross_values, xserie, xserie_a, xserie_b, cross_series, offset) -> DataFrame:
    # 创建一个空的数据框
    df = DataFrame()
    
    # 如果 xa 不为空且为整数或浮点数类型
    if xa is not None and isinstance(xa, (int, float)):
        # 如果需要计算交叉值
        if cross_values:
            # 计算指标在阈值 xa 以上交叉的起始点
            crossed_above_start = cross_value(indicator, xa, above=True, offset=offset)
            # 计算指标在阈值 xa 以上交叉的结束点
            crossed_above_end = cross_value(indicator, xa, above=False, offset=offset)
            # 将交叉信号起始点和结束点添加到数据框中
            df[crossed_above_start.name] = crossed_above_start
            df[crossed_above_end.name] = crossed_above_end
        else:
            # 计算指标在阈值 xa 以上的信号
            crossed_above = above_value(indicator, xa, offset=offset)
            # 将信号添加到数据框中
            df[crossed_above.name] = crossed_above

    # 如果 xb 不为空且为整数或浮点数类型
    if xb is not None and isinstance(xb, (int, float)):
        # 如果需要计算交叉值
        if cross_values:
            # 计算指标在阈值 xb 以下交叉的起始点
            crossed_below_start = cross_value(indicator, xb, above=True, offset=offset)
            # 计算指标在阈值 xb 以下交叉的结束点
            crossed_below_end = cross_value(indicator, xb, above=False, offset=offset)
            # 将交叉信号起始点和结束点添加到数据框中
            df[crossed_below_start.name] = crossed_below_start
            df[crossed_below_end.name] = crossed_below_end
        else:
            # 计算指标在阈值 xb 以下的信号
            crossed_below = below_value(indicator, xb, offset=offset)
            # 将信号添加到数据框中
            df[crossed_below.name] = crossed_below

    # 如果 xserie_a 为空，则使用默认值 xserie
    if xserie_a is None:
        xserie_a = xserie
    # 如果 xserie_b 为空，则使用默认值 xserie
    if xserie_b is None:
        xserie_b = xserie

    # 如果 xserie_a 不为空且为有效的数据序列
    if xserie_a is not None and verify_series(xserie_a):
        # 如果需要计算交叉序列
        if cross_series:
            # 计算指标与 xserie_a 交叉的起始点
            cross_serie_above = cross(indicator, xserie_a, above=True, offset=offset)
        else:
            # 计算指标在 xserie_a 以上的信号
            cross_serie_above = above(indicator, xserie_a, offset=offset)
        
        # 将信号添加到数据框中
        df[cross_serie_above.name] = cross_serie_above

    # 如果 xserie_b 不为空且为有效的数据序列
    if xserie_b is not None and verify_series(xserie_b):
        # 如果需要计算交叉序列
        if cross_series:
            # 计算指标与 xserie_b 交叉的起始点
            cross_serie_below = cross(indicator, xserie_b, above=False, offset=offset)
        else:
            # 计算指标在 xserie_b 以下的信号
            cross_serie_below = below(indicator, xserie_b, offset=offset)
        
        # 将信号添加到数据框中
        df[cross_serie_below.name] = cross_serie_below

    # 返回生成的数据框
    return df
```