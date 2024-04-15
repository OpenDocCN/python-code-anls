# `.\pandas-ta\pandas_ta\volume\pvr.py`

```
# -*- coding: utf-8 -*-
# 从pandas_ta.utils模块中导入verify_series函数
from pandas_ta.utils import verify_series
# 从numpy中导入别名为npNaN的nan值
from numpy import nan as npNaN
# 从pandas模块中导入Series类
from pandas import Series

# 定义函数pvr，计算价格成交量等级指标
def pvr(close, volume):
    """ Indicator: Price Volume Rank"""
    # 验证参数
    close = verify_series(close)
    volume = verify_series(volume)

    # 计算结果
    # 计算收盘价的差分并填充NaN值为0
    close_diff = close.diff().fillna(0)
    # 计算成交量的差分并填充NaN值为0
    volume_diff = volume.diff().fillna(0)
    # 创建与close索引相同的Series对象，并填充NaN值为npNaN
    pvr_ = Series(npNaN, index=close.index)
    # 根据条件设置pvr_中的值
    pvr_.loc[(close_diff >= 0) & (volume_diff >= 0)] = 1
    pvr_.loc[(close_diff >= 0) & (volume_diff < 0)]  = 2
    pvr_.loc[(close_diff < 0) & (volume_diff >= 0)]  = 3
    pvr_.loc[(close_diff < 0) & (volume_diff < 0)]   = 4

    # 设置名称和分类
    pvr_.name = f"PVR"
    pvr_.category = "volume"

    return pvr_

# 设置函数pvr的文档字符串
pvr.__doc__ = \
"""Price Volume Rank

The Price Volume Rank was developed by Anthony J. Macek and is described in his
article in the June, 1994 issue of Technical Analysis of Stocks & Commodities
Magazine. It was developed as a simple indicator that could be calculated even
without a computer. The basic interpretation is to buy when the PV Rank is below
2.5 and sell when it is above 2.5.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=PVrank.htm

Calculation:
    return 1 if 'close change' >= 0 and 'volume change' >= 0
    return 2 if 'close change' >= 0 and 'volume change' < 0
    return 3 if 'close change' < 0 and 'volume change' >= 0
    return 4 if 'close change' < 0 and 'volume change' < 0

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's

Returns:
    pd.Series: New feature generated.
"""
```