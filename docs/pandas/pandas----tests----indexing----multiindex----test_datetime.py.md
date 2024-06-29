# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_datetime.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从 pandas 库中导入多个类和函数
from pandas import (
    DataFrame,    # 数据帧类，用于表示二维数据
    Index,        # 索引类，用于标记数据帧的行或列
    MultiIndex,   # 多级索引类，用于表示具有多级行或列标签的数据帧
    Period,       # 时期类，用于表示时间段
    Series,       # 系列类，用于表示一维数据结构
    period_range, # 函数，用于生成时期范围
    to_datetime,  # 函数，用于将输入转换为 datetime 类型
)
# 导入 pandas 的测试工具模块，并使用 tm 别名
import pandas._testing as tm


# 定义测试函数 test_multiindex_period_datetime
def test_multiindex_period_datetime():
    # GH4861, 使用 datetime 作为多级索引的时期会引发异常

    # 创建第一个索引
    idx1 = Index(["a", "a", "a", "b", "b"])
    # 根据日期频率生成时期范围作为第二个索引
    idx2 = period_range("2012-01", periods=len(idx1), freq="M")
    # 创建随机数据系列，其索引为 idx1 和 idx2
    s = Series(np.random.default_rng(2).standard_normal(len(idx1)), [idx1, idx2])

    # 尝试使用 Period 作为索引
    expected = s.iloc[0]
    result = s.loc["a", Period("2012-01")]
    assert result == expected

    # 尝试使用 datetime 作为索引
    result = s.loc["a", datetime(2012, 1, 1)]
    assert result == expected


# 定义测试函数 test_multiindex_datetime_columns
def test_multiindex_datetime_columns():
    # GH35015, 使用 datetime 作为列索引会引发异常

    # 创建多级索引，列名为日期时间
    mi = MultiIndex.from_tuples(
        [(to_datetime("02/29/2020"), to_datetime("03/01/2020"))], names=["a", "b"]
    )

    # 创建空的数据帧，列索引为 mi
    df = DataFrame([], columns=mi)

    # 期望的数据帧，列索引为特定的日期时间
    expected_df = DataFrame(
        [],
        columns=MultiIndex.from_arrays(
            [[to_datetime("02/29/2020")], [to_datetime("03/01/2020")]], names=["a", "b"]
        ),
    )

    # 使用测试工具函数验证 df 和 expected_df 是否相等
    tm.assert_frame_equal(df, expected_df)
```