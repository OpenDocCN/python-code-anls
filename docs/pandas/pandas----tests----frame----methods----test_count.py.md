# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_count.py`

```
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

class TestDataFrameCount:
    def test_count(self):
        # 创建一个空的DataFrame对象
        frame = DataFrame()
        # 按行计算DataFrame中非NA值的数量，返回一个Series对象
        ct1 = frame.count(1)
        assert isinstance(ct1, Series)

        # 按列计算DataFrame中非NA值的数量，返回一个Series对象
        ct2 = frame.count(0)
        assert isinstance(ct2, Series)

        # 测试特定情况 GH#423
        # 创建一个具有指定索引范围的DataFrame对象
        df = DataFrame(index=range(10))
        # 按行计算DataFrame中非NA值的数量，返回一个Series对象
        result = df.count(1)
        # 创建一个期望的Series对象，所有值为0，索引与df相同
        expected = Series(0, index=df.index)
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

        # 创建一个具有指定列范围的DataFrame对象
        df = DataFrame(columns=range(10))
        # 按列计算DataFrame中非NA值的数量，返回一个Series对象
        result = df.count(0)
        # 创建一个期望的Series对象，所有值为0，索引与df的列名相同
        expected = Series(0, index=df.columns)
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

        # 创建一个空的DataFrame对象
        df = DataFrame()
        # 计算DataFrame中非NA值的数量，返回一个Series对象，数据类型为int64
        result = df.count()
        # 创建一个期望的Series对象，数据类型为int64
        expected = Series(dtype="int64")
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

    def test_count_objects(self, float_string_frame):
        # 从另一个DataFrame对象创建一个新的DataFrame对象
        dm = DataFrame(float_string_frame._series)
        df = DataFrame(float_string_frame._series)

        # 检查两个DataFrame对象整体的非NA值数量是否相等
        tm.assert_series_equal(dm.count(), df.count())
        # 按行计算两个DataFrame对象中非NA值的数量，检查结果是否相等
        tm.assert_series_equal(dm.count(1), df.count(1))
```