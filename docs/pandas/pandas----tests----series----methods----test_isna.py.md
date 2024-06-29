# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_isna.py`

```
"""
We also test Series.notna in this file.
"""

import numpy as np  # 导入 NumPy 库

from pandas import (  # 从 Pandas 库导入以下对象
    Period,  # 时间段对象
    Series,  # 序列对象
)
import pandas._testing as tm  # 导入 Pandas 测试模块


class TestIsna:
    def test_isna_period_dtype(self):
        # GH#13737
        # 创建一个包含 Period 对象的 Series
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        expected = Series([False, True])  # 预期的结果 Series

        result = ser.isna()  # 调用 isna() 方法检测缺失值
        tm.assert_series_equal(result, expected)  # 断言结果与预期相等

        result = ser.notna()  # 调用 notna() 方法检测非缺失值
        tm.assert_series_equal(result, ~expected)  # 断言结果与预期的相反值相等

    def test_isna(self):
        # 创建一个包含不同类型数据的 Series，包括 NaN
        ser = Series([0, 5.4, 3, np.nan, -0.001])
        expected = Series([False, False, False, True, False])  # 预期的结果 Series
        tm.assert_series_equal(ser.isna(), expected)  # 断言结果与预期相等
        tm.assert_series_equal(ser.notna(), ~expected)  # 断言结果与预期的相反值相等

        # 创建一个包含字符串和 NaN 的 Series
        ser = Series(["hi", "", np.nan])
        expected = Series([False, False, True])  # 预期的结果 Series
        tm.assert_series_equal(ser.isna(), expected)  # 断言结果与预期相等
        tm.assert_series_equal(ser.notna(), ~expected)  # 断言结果与预期的相反值相等
```