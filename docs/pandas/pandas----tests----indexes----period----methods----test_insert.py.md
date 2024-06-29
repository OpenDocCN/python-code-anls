# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_insert.py`

```
import numpy as np
import pytest

from pandas import (
    NaT,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm

class TestInsert:
    @pytest.mark.parametrize("na", [np.nan, NaT, None])
    def test_insert(self, na):
        # GH#18295 (test missing)
        # 定义期望的周期索引对象，包含指定的周期字符串和 NaT（Not a Time）占位符
        expected = PeriodIndex(["2017Q1", NaT, "2017Q2", "2017Q3", "2017Q4"], freq="Q")
        # 在生成的周期范围中在第一个位置插入给定的 na 值（可能是 NaN、NaT 或 None）
        result = period_range("2017Q1", periods=4, freq="Q").insert(1, na)
        # 断言插入操作后的结果与期望结果相等
        tm.assert_index_equal(result, expected)
```