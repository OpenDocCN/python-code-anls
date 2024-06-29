# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_delitem.py`

```
import pytest

from pandas import (
    Index,
    Series,
    date_range,
)
import pandas._testing as tm

class TestSeriesDelItem:
    def test_delitem(self):
        # GH#5542
        # should delete the item inplace
        # 创建一个包含整数 0 到 4 的 Series 对象
        s = Series(range(5))
        # 删除索引为 0 的元素
        del s[0]

        # 创建预期的 Series 对象，从 1 到 4，索引也是从 1 到 4
        expected = Series(range(1, 5), index=range(1, 5))
        # 断言 s 和 expected 是否相等
        tm.assert_series_equal(s, expected)

        # 删除索引为 1 的元素
        del s[1]
        # 创建新的预期 Series 对象，从 2 到 4，索引从 2 到 4
        expected = Series(range(2, 5), index=range(2, 5))
        # 断言 s 和 expected 是否相等
        tm.assert_series_equal(s, expected)

        # 只剩下一个元素，删除，添加，再删除
        # 创建只含有一个元素 1 的 Series 对象
        s = Series(1)
        # 删除索引为 0 的元素
        del s[0]
        # 断言 s 是否等于一个空的 Series 对象，数据类型为 int64
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="int64")))
        # 添加元素 1 到索引为 0 的位置
        s[0] = 1
        # 断言 s 是否等于包含元素 1 的 Series 对象
        tm.assert_series_equal(s, Series(1))
        # 再次删除索引为 0 的元素
        del s[0]
        # 断言 s 是否等于一个空的 Series 对象，数据类型为 int64
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype="int64")))

    def test_delitem_object_index(self, using_infer_string):
        # Index(dtype=object)
        # 根据使用的数据类型字符串选择 dtype，如果使用推断字符串则为 "string[pyarrow_numpy]"，否则为 object
        dtype = "string[pyarrow_numpy]" if using_infer_string else object
        # 创建一个具有一个元素 1 和索引为 ["a"] 的 Series 对象
        s = Series(1, index=Index(["a"], dtype=dtype))
        # 删除索引为 "a" 的元素
        del s["a"]
        # 断言 s 是否等于一个空的 Series 对象，数据类型为 int64
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype=dtype)))
        # 添加元素 1 到索引为 "a" 的位置
        s["a"] = 1
        # 断言 s 是否等于包含元素 1 和索引为 ["a"] 的 Series 对象
        tm.assert_series_equal(s, Series(1, index=Index(["a"], dtype=dtype)))
        # 再次删除索引为 "a" 的元素
        del s["a"]
        # 断言 s 是否等于一个空的 Series 对象，数据类型为 int64
        tm.assert_series_equal(s, Series(dtype="int64", index=Index([], dtype=dtype)))

    def test_delitem_missing_key(self):
        # empty
        # 创建一个数据类型为 object 的空 Series 对象
        s = Series(dtype=object)

        # 使用 pytest 断言，删除索引为 0 的元素时是否抛出 KeyError 异常，异常消息匹配 "^0$"
        with pytest.raises(KeyError, match=r"^0$"):
            del s[0]

    def test_delitem_extension_dtype(self):
        # GH#40386
        # DatetimeTZDtype
        # 创建一个包含时区为 "US/Pacific" 的日期时间范围的 Series 对象
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)

        # 创建预期的 Series 对象，包含索引为 0 和 2 的元素
        expected = ser[[0, 2]]
        # 删除索引为 1 的元素
        del ser[1]
        # 断言 ser 的数据类型是否等于 dti 的数据类型
        assert ser.dtype == dti.dtype
        # 断言 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)

        # PeriodDtype
        # 将时区为 None 的日期时间转换为日期周期
        pi = dti.tz_localize(None).to_period("D")
        ser = Series(pi)

        # 创建预期的 Series 对象，包含索引为 0 和 1 的元素
        expected = ser[:2]
        # 删除索引为 2 的元素
        del ser[2]
        # 断言 ser 的数据类型是否等于 pi 的数据类型
        assert ser.dtype == pi.dtype
        # 断言 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)
```