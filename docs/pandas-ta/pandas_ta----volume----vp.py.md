# `.\pandas-ta\pandas_ta\volume\vp.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库中导入 array_split 函数
from numpy import array_split
# 从 numpy 库中导入 mean 函数
from numpy import mean
# 从 pandas 库中导入 cut、concat、DataFrame 函数
from pandas import cut, concat, DataFrame
# 从 pandas_ta.utils 模块中导入 signed_series、verify_series 函数
from pandas_ta.utils import signed_series, verify_series


def vp(close, volume, width=None, **kwargs):
    """Indicator: Volume Profile (VP)"""
    # 验证参数
    # 如果宽度参数存在且大于 0，则将其转换为整数类型，否则设为默认值 10
    width = int(width) if width and width > 0 else 10
    # 验证 close 数据序列，确保长度为 width
    close = verify_series(close, width)
    # 验证 volume 数据序列，确保长度为 width
    volume = verify_series(volume, width)
    # 从 kwargs 中弹出 sort_close 参数，默认值为 False
    sort_close = kwargs.pop("sort_close", False)

    # 如果 close 或 volume 为空，则返回 None
    if close is None or volume is None: return

    # 设置
    # 生成符号价格序列，即 close 的涨跌情况，用于后续计算正负成交量
    signed_price = signed_series(close, 1)
    # 正成交量为 volume 与符号价格序列大于 0 的乘积
    pos_volume = volume * signed_price[signed_price > 0]
    pos_volume.name = volume.name
    # 负成交量为 volume 与符号价格序列小于 0 的乘积，乘以 -1
    neg_volume = -volume * signed_price[signed_price < 0]
    neg_volume.name = volume.name
    # 合并 close、正成交量、负成交量 到一个 DataFrame 中
    vp = concat([close, pos_volume, neg_volume], axis=1)

    close_col = f"{vp.columns[0]}"
    high_price_col = f"high_{close_col}"
    low_price_col = f"low_{close_col}"
    mean_price_col = f"mean_{close_col}"

    volume_col = f"{vp.columns[1]}"
    pos_volume_col = f"pos_{volume_col}"
    neg_volume_col = f"neg_{volume_col}"
    total_volume_col = f"total_{volume_col}"
    vp.columns = [close_col, pos_volume_col, neg_volume_col]

    # sort_close: 在将数据切分为范围之前，是否根据收盘价进行排序。默认值为 False
    # 如果为 False，则根据日期索引或时间顺序排序，而不是根据价格
    if sort_close:
        # 将 mean_price_col 列设置为 close_col 列的值
        vp[mean_price_col] = vp[close_col]
        # 按照 close_col 列的值进行分组，并计算各范围内的平均价、正成交量、负成交量
        vpdf = vp.groupby(cut(vp[close_col], width, include_lowest=True, precision=2)).agg({
            mean_price_col: mean,
            pos_volume_col: sum,
            neg_volume_col: sum,
        })
        # 从索引中获取范围的最低价格和最高价格
        vpdf[low_price_col] = [x.left for x in vpdf.index]
        vpdf[high_price_col] = [x.right for x in vpdf.index]
        # 重置索引并重新排列列的顺序
        vpdf = vpdf.reset_index(drop=True)
        vpdf = vpdf[[low_price_col, mean_price_col, high_price_col, pos_volume_col, neg_volume_col]]
    else:
        # 将 vp DataFrame 切分为若干子 DataFrame，每个子 DataFrame 包含近似相等数量的数据点
        vp_ranges = array_split(vp, width)
        # 遍历每个子 DataFrame，计算范围内的最低价、平均价、最高价、正成交量、负成交量，并生成生成器对象
        result = ({
            low_price_col: r[close_col].min(),
            mean_price_col: r[close_col].mean(),
            high_price_col: r[close_col].max(),
            pos_volume_col: r[pos_volume_col].sum(),
            neg_volume_col: r[neg_volume_col].sum(),
        } for r in vp_ranges)
        # 将生成器对象转换为 DataFrame
        vpdf = DataFrame(result)
    # 计算总成交量，并添加到 DataFrame 中
    vpdf[total_volume_col] = vpdf[pos_volume_col] + vpdf[neg_volume_col]

    # 处理填充值
    # 如果 kwargs 中包含 fillna 参数，则使用该参数填充 NaN 值
    if "fillna" in kwargs:
        vpdf.fillna(kwargs["fillna"], inplace=True)
    # 如果 kwargs 中包含 fill_method 参数，则使用该参数填充 NaN 值
    if "fill_method" in kwargs:
        vpdf.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    vpdf.name = f"VP_{width}"
    vpdf.category = "volume"

    # 返回结果 DataFrame
    return vpdf


# 将函数文档字符串设为指定内容
vp.__doc__ = \
"""Volume Profile (VP)

Calculates the Volume Profile by slicing price into ranges.
Note: Value Area is not calculated.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:volume_by_price
    https://www.tradingview.com/wiki/Volume_Profile
    http://www.ranchodinero.com/volume-tpo-essentials/
"""
    # 访问指定网址以获取相关信息，这是一个网页链接
    https://www.tradingtechnologies.com/blog/2013/05/15/volume-at-price/
# 计算函数
Calculation:
    # 默认输入参数：宽度为10
    Default Inputs:
        width=10

    # 将 'close'、'pos_volume'、'neg_volume' 三个 Series 按列合并成一个 DataFrame
    vp = pd.concat([close, pos_volume, neg_volume], axis=1)
    # 如果需要按 'close' 排序
    if sort_close:
        # 将 'close' 列按照指定宽度切割成不同范围的区间
        vp_ranges = cut(vp[close_col], width)
        # 对于每个区间，计算以下统计量：左边界、平均 'close'、右边界、'pos_volume'、'neg_volume'，结果为一个字典
        result = ({range_left, mean_close, range_right, pos_volume, neg_volume} foreach range in vp_ranges
    # 如果不需要按 'close' 排序
    else:
        # 将 DataFrame 按照指定宽度等分成不同的区间
        vp_ranges = np.array_split(vp, width)
        # 对于每个区间，计算以下统计量：最低 'close'、平均 'close'、最高 'close'、'pos_volume'、'neg_volume'，结果为一个字典
        result = ({low_close, mean_close, high_close, pos_volume, neg_volume} foreach range in vp_ranges
    # 将结果字典转换成 DataFrame
    vpdf = pd.DataFrame(result)
    # 计算总交易量并添加到 DataFrame 中
    vpdf['total_volume'] = vpdf['pos_volume'] + vpdf['neg_volume']

# 参数说明
Args:
    # 'close' 的 Series 数据
    close (pd.Series): Series of 'close's
    # 'volume' 的 Series 数据
    volume (pd.Series): Series of 'volume's
    # 将价格分布到的区间数，默认为10
    width (int): How many ranges to distrubute price into. Default: 10

# 可选参数说明
Kwargs:
    # 对于缺失值的填充值，默认为 pd.DataFrame.fillna(value)
    fillna (value, optional): pd.DataFrame.fillna(value)
    # 填充方法的类型，默认为 None
    fill_method (value, optional): Type of fill method
    # 是否在切割成区间之前按 'close' 进行排序，默认为 False
    sort_close (value, optional): Whether to sort by close before splitting
        into ranges. Default: False

# 返回结果
Returns:
    # 生成的新特征的 DataFrame
    pd.DataFrame: New feature generated.
```