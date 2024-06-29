# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_to_pydatetime.py`

```
from datetime import (
    datetime,        # 导入 datetime 模块中的 datetime 类
    timedelta,       # 导入 datetime 模块中的 timedelta 类
)

import pytest         # 导入 pytest 测试框架

from pandas._libs.tslibs.timezones import dateutil_gettz as gettz   # 从 pandas 库中导入 dateutil_gettz 函数
import pandas.util._test_decorators as td    # 从 pandas 库中导入 _test_decorators 模块，重命名为 td

from pandas import Timestamp   # 从 pandas 库中导入 Timestamp 类
import pandas._testing as tm   # 从 pandas 库中导入 _testing 模块，重命名为 tm


class TestTimestampToPyDatetime:
    def test_to_pydatetime_fold(self):
        # GH#45087
        tzstr = "dateutil/usr/share/zoneinfo/America/Chicago"   # 定义时区字符串
        ts = Timestamp(year=2013, month=11, day=3, hour=1, minute=0, fold=1, tz=tzstr)   # 创建带时区信息的 Timestamp 对象
        dt = ts.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
        assert dt.fold == 1   # 断言转换后的 datetime 对象的 fold 属性为 1

    def test_to_pydatetime_nonzero_nano(self):
        ts = Timestamp("2011-01-01 9:00:00.123456789")   # 创建带纳秒的 Timestamp 对象

        # Warn the user of data loss (nanoseconds).
        msg = "Discarding nonzero nanoseconds in conversion"   # 提示用户在转换中丢失非零纳秒部分的警告信息
        with tm.assert_produces_warning(UserWarning, match=msg):   # 使用 _testing 模块中的 assert_produces_warning 断言产生 UserWarning 并匹配指定的警告信息
            expected = datetime(2011, 1, 1, 9, 0, 0, 123456)   # 创建预期的 datetime 对象
            result = ts.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
            assert result == expected   # 断言转换后的结果与预期相等

    def test_timestamp_to_datetime(self):
        stamp = Timestamp("20090415", tz="US/Eastern")   # 创建带时区信息的 Timestamp 对象
        dtval = stamp.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
        assert stamp == dtval   # 断言 Timestamp 对象与转换后的 datetime 对象相等
        assert stamp.tzinfo == dtval.tzinfo   # 断言 Timestamp 对象和转换后的 datetime 对象的时区信息相等

    def test_timestamp_to_pydatetime_dateutil(self):
        stamp = Timestamp("20090415", tz="dateutil/US/Eastern")   # 创建带 dateutil 时区信息的 Timestamp 对象
        dtval = stamp.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
        assert stamp == dtval   # 断言 Timestamp 对象与转换后的 datetime 对象相等
        assert stamp.tzinfo == dtval.tzinfo   # 断言 Timestamp 对象和转换后的 datetime 对象的时区信息相等

    def test_timestamp_to_pydatetime_explicit_pytz(self):
        pytz = pytest.importorskip("pytz")   # 导入并检查 pytz 库，如果导入失败则跳过测试
        stamp = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))   # 创建带 pytz 时区信息的 Timestamp 对象
        dtval = stamp.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
        assert stamp == dtval   # 断言 Timestamp 对象与转换后的 datetime 对象相等
        assert stamp.tzinfo == dtval.tzinfo   # 断言 Timestamp 对象和转换后的 datetime 对象的时区信息相等

    @td.skip_if_windows   # 使用 _test_decorators 模块中的 skip_if_windows 装饰器，如果在 Windows 下跳过测试
    def test_timestamp_to_pydatetime_explicit_dateutil(self):
        stamp = Timestamp("20090415", tz=gettz("US/Eastern"))   # 创建带 dateutil 时区信息的 Timestamp 对象
        dtval = stamp.to_pydatetime()   # 将 Timestamp 对象转换为 Python 原生的 datetime 对象
        assert stamp == dtval   # 断言 Timestamp 对象与转换后的 datetime 对象相等
        assert stamp.tzinfo == dtval.tzinfo   # 断言 Timestamp 对象和转换后的 datetime 对象的时区信息相等

    def test_to_pydatetime_bijective(self):
        # Ensure that converting to datetime and back only loses precision
        # by going from nanoseconds to microseconds.
        exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning   # 如果 Timestamp 最大值的纳秒部分为 0，则不期望警告
        with tm.assert_produces_warning(exp_warning):   # 使用 _testing 模块中的 assert_produces_warning 断言是否产生期望的警告
            pydt_max = Timestamp.max.to_pydatetime()   # 将 Timestamp 最大值转换为 Python 原生的 datetime 对象

        assert (
            Timestamp(pydt_max).as_unit("ns")._value / 1000
            == Timestamp.max._value / 1000
        )   # 断言转换回 Timestamp 对象后，时间单位为纳秒且值相等

        exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning   # 如果 Timestamp 最小值的纳秒部分为 0，则不期望警告
        with tm.assert_produces_warning(exp_warning):   # 使用 _testing 模块中的 assert_produces_warning 断言是否产生期望的警告
            pydt_min = Timestamp.min.to_pydatetime()   # 将 Timestamp 最小值转换为 Python 原生的 datetime 对象

        # The next assertion can be enabled once GH#39221 is merged
        #  assert pydt_min < Timestamp.min  # this is bc nanos are dropped
        tdus = timedelta(microseconds=1)   # 创建微秒级别的 timedelta 对象
        assert pydt_min + tdus > Timestamp.min   # 断言经过 timedelta 增加后的时间大于 Timestamp 最小值

        assert (
            Timestamp(pydt_min + tdus).as_unit("ns")._value / 1000
            == Timestamp.min._value / 1000
        )   # 断言转换回 Timestamp 对象后，时间单位为纳秒且值相等
```