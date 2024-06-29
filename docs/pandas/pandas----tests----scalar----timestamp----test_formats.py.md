# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_formats.py`

```
# 导入 datetime 模块中的 datetime 和 timezone 类
# 导入 pprint 模块用于漂亮打印
from datetime import (
    datetime,
    timezone,
)
import pprint

# 导入 dateutil.tz 模块
import dateutil.tz
# 导入 pytest 模块
import pytest

# 从 pandas.compat 模块导入 WASM（WebAssembly）对象
from pandas.compat import WASM

# 从 pandas 模块导入 Timestamp 类
from pandas import Timestamp

# 创建一个不带纳秒的 Timestamp 对象
ts_no_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
# 创建一个带纳秒的 Timestamp 对象，年份为 1
ts_no_ns_year1 = Timestamp(
    year=1,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
# 创建一个带纳秒的 Timestamp 对象
ts_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
)
# 创建一个带纳秒和时区信息的 Timestamp 对象
ts_ns_tz = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
    tz="UTC",
)
# 创建一个没有微秒的 Timestamp 对象，但带纳秒
ts_no_us = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=0,
    nanosecond=123,
)

# 使用 pytest.mark.parametrize 注册多个测试用例
@pytest.mark.parametrize(
    "ts, timespec, expected_iso",
    [
        # 测试 Timestamp 对象的 ISO 8601 格式化输出，使用不同的 timespec
        (ts_no_ns, "auto", "2019-05-18T15:17:08.132263"),
        (ts_no_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_no_ns, "nanoseconds", "2019-05-18T15:17:08.132263000"),
        (ts_no_ns_year1, "seconds", "0001-05-18T15:17:08"),
        (ts_no_ns_year1, "nanoseconds", "0001-05-18T15:17:08.132263000"),
        (ts_ns, "auto", "2019-05-18T15:17:08.132263123"),
        (ts_ns, "hours", "2019-05-18T15"),
        (ts_ns, "minutes", "2019-05-18T15:17"),
        (ts_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_ns, "milliseconds", "2019-05-18T15:17:08.132"),
        (ts_ns, "microseconds", "2019-05-18T15:17:08.132263"),
        (ts_ns, "nanoseconds", "2019-05-18T15:17:08.132263123"),
        (ts_ns_tz, "auto", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_ns_tz, "hours", "2019-05-18T15+00:00"),
        (ts_ns_tz, "minutes", "2019-05-18T15:17+00:00"),
        (ts_ns_tz, "seconds", "2019-05-18T15:17:08+00:00"),
        (ts_ns_tz, "milliseconds", "2019-05-18T15:17:08.132+00:00"),
        (ts_ns_tz, "microseconds", "2019-05-18T15:17:08.132263+00:00"),
        (ts_ns_tz, "nanoseconds", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_no_us, "auto", "2019-05-18T15:17:08.000000123"),
    ],
)
# 定义测试函数 test_isoformat，验证 Timestamp 对象的 ISO 格式输出是否符合预期
def test_isoformat(ts, timespec, expected_iso):
    assert ts.isoformat(timespec=timespec) == expected_iso


# 定义一个测试类 TestTimestampRendering，用于测试 Timestamp 的渲染
class TestTimestampRendering:
    # 使用 pytest.mark.parametrize 注册多个测试参数
    @pytest.mark.parametrize(
        "tz", ["UTC", "Asia/Tokyo", "US/Eastern", "dateutil/America/Los_Angeles"]
    )
    @pytest.mark.parametrize("freq", ["D", "M", "S", "N"])
    @pytest.mark.parametrize(
        "date", ["2014-03-07", "2014-01-01 09:00", "2014-01-01 00:00:00.000000001"]
    )
    # 使用 pytest.mark.skipif 装饰器，如果 WASM 为真，则跳过测试
    @pytest.mark.skipif(WASM, reason="tzset is not available on WASM")
    def test_repr(self, date, freq, tz):
        # 避免与时区名称匹配
        freq_repr = f"'{freq}'"
        # 如果时区以 "dateutil" 开头，去掉其中的 "dateutil"
        if tz.startswith("dateutil"):
            tz_repr = tz.replace("dateutil", "")
        else:
            tz_repr = tz

        # 创建一个只包含日期的 Timestamp 对象
        date_only = Timestamp(date)
        # 断言日期字符串出现在 date_only 的字符串表示中
        assert date in repr(date_only)
        # 断言时区字符串不出现在 date_only 的字符串表示中
        assert tz_repr not in repr(date_only)
        # 断言频率字符串不出现在 date_only 的字符串表示中
        assert freq_repr not in repr(date_only)
        # 断言 date_only 与其字符串表示的 eval 结果相等
        assert date_only == eval(repr(date_only))

        # 创建一个带有日期和时区的 Timestamp 对象
        date_tz = Timestamp(date, tz=tz)
        # 断言日期字符串出现在 date_tz 的字符串表示中
        assert date in repr(date_tz)
        # 断言时区字符串出现在 date_tz 的字符串表示中
        assert tz_repr in repr(date_tz)
        # 断言频率字符串不出现在 date_tz 的字符串表示中
        assert freq_repr not in repr(date_tz)
        # 断言 date_tz 与其字符串表示的 eval 结果相等
        assert date_tz == eval(repr(date_tz))

    def test_repr_utcoffset(self):
        # 这可能导致填充 tz 字段，但在日期字符串中包含这些信息是多余的
        date_with_utc_offset = Timestamp("2014-03-13 00:00:00-0400")
        # 断言特定日期字符串出现在 date_with_utc_offset 的字符串表示中
        assert "2014-03-13 00:00:00-0400" in repr(date_with_utc_offset)
        # 断言字符串 "tzoffset" 不出现在 date_with_utc_offset 的字符串表示中
        assert "tzoffset" not in repr(date_with_utc_offset)
        # 断言字符串 "UTC-04:00" 出现在 date_with_utc_offset 的字符串表示中
        assert "UTC-04:00" in repr(date_with_utc_offset)
        # 将 date_with_utc_offset 的字符串表示赋给 expr
        expr = repr(date_with_utc_offset)
        # 断言 date_with_utc_offset 与 expr 的 eval 结果相等
        assert date_with_utc_offset == eval(expr)

    def test_timestamp_repr_pre1900(self):
        # 用于测试1900年之前的时间戳
        stamp = Timestamp("1850-01-01", tz="US/Eastern")
        # 获取时间戳的字符串表示

        iso8601 = "1850-01-01 01:23:45.012345"
        # 创建带有 ISO 8601 格式的时间戳，并获取其字符串表示
        stamp = Timestamp(iso8601, tz="US/Eastern")
        # 获取时间戳的字符串表示，并将结果赋给 result
        result = repr(stamp)
        # 断言 iso8601 出现在 result 中
        assert iso8601 in result

    def test_pprint(self):
        # GH#12622 测试Pretty Print的输出
        nested_obj = {"foo": 1, "bar": [{"w": {"a": Timestamp("2011-01-01")}}] * 10}
        # 使用 pprint 格式化嵌套对象，限定行宽为 50
        result = pprint.pformat(nested_obj, width=50)
        expected = r"""{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],
 'foo': 1}"""
        # 断言结果与预期输出相等
        assert result == expected

    def test_to_timestamp_repr_is_code(self):
        zs = [
            Timestamp("99-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="America/Los_Angeles"),
            Timestamp("2001-04-17 00:00:00", tz=None),
        ]
        # 遍历时间戳列表 zs
        for z in zs:
            # 断言 eval(repr(z)) 等于 z
            assert eval(repr(z)) == z
    # 测试函数，用于验证 Timestamp 类的字符串表示是否与 datetime 类兼容（不带时区的情况）
    def test_repr_matches_pydatetime_no_tz(self):
        # 创建一个 datetime 对象，表示日期为2013年1月2日
        dt_date = datetime(2013, 1, 2)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_date) == str(Timestamp(dt_date))

        # 创建一个 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒
        dt_datetime = datetime(2013, 1, 2, 12, 1, 3)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        # 创建一个 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒，微秒为45
        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

        # 创建一个 Timestamp 对象，表示纳秒为200
        ts_nanos_only = Timestamp(200)
        # 断言 Timestamp 对象的字符串表示与指定的字符串相同
        assert str(ts_nanos_only) == "1970-01-01 00:00:00.000000200"

        # 创建一个 Timestamp 对象，表示纳秒为1200
        ts_nanos_micros = Timestamp(1200)
        # 断言 Timestamp 对象的字符串表示与指定的字符串相同
        assert str(ts_nanos_micros) == "1970-01-01 00:00:00.000001200"

    # 测试函数，用于验证 Timestamp 类的字符串表示是否与 datetime 类兼容（使用标准库时区）
    def test_repr_matches_pydatetime_tz_stdlib(self):
        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时区为UTC
        dt_date = datetime(2013, 1, 2, tzinfo=timezone.utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_date) == str(Timestamp(dt_date))

        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒，时区为UTC
        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=timezone.utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒，微秒为45，时区为UTC
        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=timezone.utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

    # 测试函数，用于验证 Timestamp 类的字符串表示是否与 datetime 类兼容（使用 dateutil 库时区）
    def test_repr_matches_pydatetime_tz_dateutil(self):
        # 创建一个 dateutil 库中的 UTC 时区对象
        utc = dateutil.tz.tzutc()

        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时区为UTC
        dt_date = datetime(2013, 1, 2, tzinfo=utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_date) == str(Timestamp(dt_date))

        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒，时区为UTC
        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        # 创建一个带有时区信息的 datetime 对象，表示日期为2013年1月2日，时间为12点1分3秒，微秒为45，时区为UTC
        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=utc)
        # 断言 Timestamp 对象的字符串表示与 datetime 对象的相同
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))
```