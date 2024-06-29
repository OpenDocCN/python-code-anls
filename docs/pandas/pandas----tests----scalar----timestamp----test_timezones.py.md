# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_timezones.py`

```
"""
Tests for Timestamp timezone-related methods
"""

from datetime import datetime  # 导入 datetime 模块中的 datetime 类

from pandas._libs.tslibs import timezones  # 导入 pandas 库中的 timezones 模块

from pandas import Timestamp  # 导入 pandas 库中的 Timestamp 类


class TestTimestampTZOperations:
    # ------------------------------------------------------------------

    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # GH21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)  # 使用 tz_naive_fixture 获取时区信息

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)  # 创建带有时区信息的 Timestamp 对象
        _datetime = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)  # 创建带有时区信息的 datetime 对象

        result = stamp.timetz()  # 获取 Timestamp 对象的时间部分
        expected = _datetime.timetz()  # 获取 datetime 对象的时间部分

        assert result == expected  # 断言 Timestamp 对象的时间部分与 datetime 对象的时间部分相等
```