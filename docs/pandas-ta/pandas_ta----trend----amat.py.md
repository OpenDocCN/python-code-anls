# `.\pandas-ta\pandas_ta\trend\amat.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从当前目录下的 long_run 模块中导入 long_run 函数
from .long_run import long_run
# 从当前目录下的 short_run 模块中导入 short_run 函数
from .short_run import short_run
# 从 pandas_ta 包中的 overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta 包中的 utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义函数 amat，用于计算 Archer Moving Averages Trends (AMAT) 指标
def amat(close=None, fast=None, slow=None, lookback=None, mamode=None, offset=None, **kwargs):
    """Indicator: Archer Moving Averages Trends (AMAT)"""
    # 验证参数的有效性，如果未提供则使用默认值
    fast = int(fast) if fast and fast > 0 else 8
    slow = int(slow) if slow and slow > 0 else 21
    lookback = int(lookback) if lookback and lookback > 0 else 2
    # 将 mamode 转换为小写字符串，如果未提供则使用默认值 "ema"
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    # 验证 close 参数，确保长度足够用于计算指标
    close = verify_series(close, max(fast, slow, lookback))
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 kwargs 中包含 "length" 键，则移除它
    if "length" in kwargs: kwargs.pop("length")

    # 如果未提供 close 参数，则返回空值
    if close is None: return

    # 计算快速移动平均线和慢速移动平均线
    fast_ma = ma(mamode, close, length=fast, **kwargs)
    slow_ma = ma(mamode, close, length=slow, **kwargs)

    # 计算长期和短期运行趋势
    mas_long = long_run(fast_ma, slow_ma, length=lookback)
    mas_short = short_run(fast_ma, slow_ma, length=lookback)

    # 对结果进行偏移处理
    if offset != 0:
        mas_long = mas_long.shift(offset)
        mas_short = mas_short.shift(offset)

    # 如果 kwargs 中包含 "fillna" 键，则使用指定的值填充缺失值
    if "fillna" in kwargs:
        mas_long.fillna(kwargs["fillna"], inplace=True)
        mas_short.fillna(kwargs["fillna"], inplace=True)

    # 如果 kwargs 中包含 "fill_method" 键，则使用指定的填充方法填充缺失值
    if "fill_method" in kwargs:
        mas_long.fillna(method=kwargs["fill_method"], inplace=True)
        mas_short.fillna(method=kwargs["fill_method"], inplace=True)

    # 准备要返回的 DataFrame
    amatdf = DataFrame({
        f"AMAT{mamode[0]}_LR_{fast}_{slow}_{lookback}": mas_long,
        f"AMAT{mamode[0]}_SR_{fast}_{slow}_{lookback}": mas_short
    })

    # 设置 DataFrame 的名称和类别
    amatdf.name = f"AMAT{mamode[0]}_{fast}_{slow}_{lookback}"
    amatdf.category = "trend"

    # 返回结果 DataFrame
    return amatdf
```