# `.\pandas-ta\pandas_ta\momentum\td_seq.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库中导入 where 函数并重命名为 npWhere
from numpy import where as npWhere
# 从 pandas 库中导入 DataFrame 和 Series 类
from pandas import DataFrame, Series
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 td_seq，用于计算 Tom Demark Sequential（TD_SEQ）指标
def td_seq(close, asint=None, offset=None, **kwargs):
    """Indicator: Tom Demark Sequential (TD_SEQ)"""
    # 验证参数 close 是否为有效的 Series 对象
    close = verify_series(close)
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 asint 不为布尔值，则设置为 False
    asint = asint if isinstance(asint, bool) else False
    # 获取参数中的 show_all，如果不存在则设置默认值为 True
    show_all = kwargs.setdefault("show_all", True)

    # 定义函数 true_sequence_count，用于计算连续的真值序列数量
    def true_sequence_count(series: Series):
        # 找到最后一个为 False 的索引
        index = series.where(series == False).last_valid_index()

        if index is None:
            # 如果索引为空，则返回序列的总数
            return series.count()
        else:
            # 否则，返回索引之后的序列数量
            s = series[series.index > index]
            return s.count()

    # 定义函数 calc_td，用于计算 TD_SEQ
    def calc_td(series: Series, direction: str, show_all: bool):
        # 计算 TD_SEQ 的布尔值
        td_bool = series.diff(4) > 0 if direction=="up" else series.diff(4) < 0
        # 根据布尔值计算 TD_SEQ 数值
        td_num = npWhere(
            td_bool, td_bool.rolling(13, min_periods=0).apply(true_sequence_count), 0
        )
        td_num = Series(td_num)

        if show_all:
            # 如果 show_all 为 True，则保留所有 TD_SEQ 值
            td_num = td_num.mask(td_num == 0)
        else:
            # 否则，只保留在 6 到 9 之间的 TD_SEQ 值
            td_num = td_num.mask(~td_num.between(6,9))

        return td_num

    # 计算上升序列的 TD_SEQ
    up_seq = calc_td(close, "up", show_all)
    # 计算下降序列的 TD_SEQ
    down_seq = calc_td(close, "down", show_all)

    # 如果需要将结果转换为整数
    if asint:
        if up_seq.hasnans and down_seq.hasnans:
            # 填充缺失值为 0
            up_seq.fillna(0, inplace=True)
            down_seq.fillna(0, inplace=True)
        # 转换结果为整数类型
        up_seq = up_seq.astype(int)
        down_seq = down_seq.astype(int)

    # 如果偏移量不为 0
    if offset != 0:
        # 对结果进行偏移
        up_seq = up_seq.shift(offset)
        down_seq = down_seq.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        up_seq.fillna(kwargs["fillna"], inplace=True)
        down_seq.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        up_seq.fillna(method=kwargs["fill_method"], inplace=True)
        down_seq.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置上升序列和下降序列的名称和分类
    up_seq.name = f"TD_SEQ_UPa" if show_all else f"TD_SEQ_UP"
    down_seq.name = f"TD_SEQ_DNa" if show_all else f"TD_SEQ_DN"
    up_seq.category = down_seq.category = "momentum"

    # 准备要返回的 DataFrame
    df = DataFrame({up_seq.name: up_seq, down_seq.name: down_seq})
    df.name = "TD_SEQ"
    df.category = up_seq.category

    return df

# 设置函数文档字符串
td_seq.__doc__ = \
"""TD Sequential (TD_SEQ)

Tom DeMark's Sequential indicator attempts to identify a price point where an
uptrend or a downtrend exhausts itself and reverses.

Sources:
    https://tradetrekker.wordpress.com/tdsequential/

Calculation:
    Compare current close price with 4 days ago price, up to 13 days. For the
    consecutive ascending or descending price sequence, display 6th to 9th day
    value.

Args:
    close (pd.Series): Series of 'close's
    asint (bool): If True, fillnas with 0 and change type to int. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:

"""
    # 定义函数参数show_all，用于控制展示范围，默认为True，即展示1到13；如果设置为False，仅展示6到9。
    show_all (bool): Show 1 - 13. If set to False, show 6 - 9. Default: True
    # 定义函数参数fillna，用于填充缺失值，参数value为填充的数值，默认为空。
    fillna (value, optional): pd.DataFrame.fillna(value)
# 返回类型说明：返回的是一个 Pandas DataFrame，其中包含了生成的新特征。
```