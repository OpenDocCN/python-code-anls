# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_to_frame.py`

```
from pandas import (
    DataFrame,
    Index,
    date_range,
)
import pandas._testing as tm


class TestToFrame:
    def test_to_frame_datetime_tz(self):
        # 创建一个日期范围，从"2019-01-01"到"2019-01-30"，每日频率为一天，时区设为UTC
        idx = date_range(start="2019-01-01", end="2019-01-30", freq="D", tz="UTC")
        # 将日期时间索引转换为DataFrame
        result = idx.to_frame()
        # 创建一个期望的DataFrame，使用idx作为数据和索引
        expected = DataFrame(idx, index=idx)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

    def test_to_frame_respects_none_name(self):
        # GH#44212 如果我们显式地传递name=None，那么应该尊重这个设置，
        # 不应该被改变为0
        # GH-45448 这是首先弃用的，未来会有改动
        # 创建一个日期范围，从"2019-01-01"到"2019-01-30"，每日频率为一天，时区设为UTC
        idx = date_range(start="2019-01-01", end="2019-01-30", freq="D", tz="UTC")
        # 将日期时间索引转换为DataFrame，指定name=None
        result = idx.to_frame(name=None)
        # 创建一个期望的索引，其中包含一个值为None的对象类型索引
        exp_idx = Index([None], dtype=object)
        # 断言期望的索引与结果DataFrame的列相等
        tm.assert_index_equal(exp_idx, result.columns)

        # 将日期时间索引重命名为"foo"，然后转换为DataFrame，指定name=None
        result = idx.rename("foo").to_frame(name=None)
        # 再次创建一个期望的索引，其中包含一个值为None的对象类型索引
        exp_idx = Index([None], dtype=object)
        # 再次断言期望的索引与结果DataFrame的列相等
        tm.assert_index_equal(exp_idx, result.columns)
```