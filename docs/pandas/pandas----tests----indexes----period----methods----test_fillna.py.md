# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_fillna.py`

```
from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm

class TestFillNA:
    def test_fillna_period(self):
        # GH#11343
        # 创建一个周期索引对象 idx，包含三个元素，其中一个为 NaT
        idx = PeriodIndex(["2011-01-01 09:00", NaT, "2011-01-01 11:00"], freq="h")

        # 期望的周期索引对象 exp，填充了缺失的时间点
        exp = PeriodIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], freq="h"
        )
        # 使用 Period("2011-01-01 10:00", freq="h") 来填充缺失值，生成结果对象 result
        result = idx.fillna(Period("2011-01-01 10:00", freq="h"))
        # 断言结果对象与期望对象相等
        tm.assert_index_equal(result, exp)

        # 期望的索引对象 exp，填充了缺失的时间点
        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                "x",  # 填充了 NaT 为 "x"
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        # 使用 "x" 来填充缺失值，生成结果对象 result
        result = idx.fillna("x")
        # 断言结果对象与期望对象相等
        tm.assert_index_equal(result, exp)

        # 期望的索引对象 exp，填充了缺失的时间点
        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                Period("2011-01-01", freq="D"),  # 填充了 NaT 为 Period("2011-01-01", freq="D")
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        # 使用 Period("2011-01-01", freq="D") 来填充缺失值，生成结果对象 result
        result = idx.fillna(Period("2011-01-01", freq="D"))
        # 断言结果对象与期望对象相等
        tm.assert_index_equal(result, exp)
```