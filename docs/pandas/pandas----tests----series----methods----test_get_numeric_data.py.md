# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_get_numeric_data.py`

```
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestGetNumericData:
    def test_get_numeric_data_preserve_dtype(self):
        # 创建一个 Series 对象，包含整数数据
        obj = Series([1, 2, 3])
        # 调用 Series 的方法获取其中的数值数据
        result = obj._get_numeric_data()
        # 断言获取的数值数据与原始对象相等
        tm.assert_series_equal(result, obj)

        # 返回的结果是一个浅拷贝对象
        result.iloc[0] = 0
        # 断言原始对象的第一个元素未被改变
        assert obj.iloc[0] == 1

        # 创建一个包含整数、字符串和浮点数的 Series 对象
        obj = Series([1, "2", 3.0])
        # 调用 Series 的方法获取其中的数值数据
        result = obj._get_numeric_data()
        # 期望的结果是一个空的 Series 对象，数据类型为 object
        expected = Series([], dtype=object)
        # 断言获取的数值数据与期望结果相等
        tm.assert_series_equal(result, expected)

        # 创建一个包含布尔值的 Series 对象
        obj = Series([True, False, True])
        # 调用 Series 的方法获取其中的数值数据
        result = obj._get_numeric_data()
        # 断言获取的数值数据与原始对象相等
        tm.assert_series_equal(result, obj)

        # 创建一个包含日期范围的 Series 对象
        obj = Series(date_range("20130101", periods=3))
        # 调用 Series 的方法获取其中的数值数据
        result = obj._get_numeric_data()
        # 期望的结果是一个空的 Series 对象，数据类型为 datetime64[ns]
        expected = Series([], dtype="M8[ns]")
        # 断言获取的数值数据与期望结果相等
        tm.assert_series_equal(result, expected)
```