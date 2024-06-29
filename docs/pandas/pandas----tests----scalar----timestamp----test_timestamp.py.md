# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\test_timestamp.py`

```
# 导入标准库和第三方库
import calendar
from datetime import (
    datetime,
    timedelta,
    timezone,
)
import locale
import time
import unicodedata
import zoneinfo

# 导入第三方库中的特定模块和函数
from dateutil.tz import (
    tzlocal,
    tzutc,
)
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

# 导入 pandas 内部模块和类
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
    dateutil_gettz as gettz,
    get_timezone,
    maybe_get_tz,
    tz_compare,
)
from pandas.compat import IS64

# 导入 pandas 主要对象和类
from pandas import (
    NaT,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm

# 导入 pandas 时间序列相关模块
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# 定义测试类 TestTimestampProperties
class TestTimestampProperties:
    # 定义测试方法 test_properties_business
    def test_properties_business(self):
        # 获取工作日频率的偏移量
        freq = to_offset("B")

        # 创建 Timestamp 对象 ts
        ts = Timestamp("2017-10-01")
        # 断言 ts 的星期几是 6（星期日）
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        # 断言 ts 是否是月初（不是工作日）
        assert ts.is_month_start
        # 断言频率是否将 ts 所在日期视为月初
        assert not freq.is_month_start(ts)
        # 断言在 ts 后一天，频率将其视为月初
        assert freq.is_month_start(ts + Timedelta(days=1))
        # 断言 ts 不是季度初
        assert not freq.is_quarter_start(ts)
        # 断言在 ts 后一天，频率将其视为季度初
        assert freq.is_quarter_start(ts + Timedelta(days=1))

        # 创建另一个 Timestamp 对象 ts
        ts = Timestamp("2017-09-30")
        # 断言 ts 的星期几是 5（星期六）
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        # 断言 ts 是否是月末
        assert ts.is_month_end
        # 断言频率是否将 ts 所在日期视为月末
        assert not freq.is_month_end(ts)
        # 断言在 ts 前一天，频率将其视为月末
        assert freq.is_month_end(ts - Timedelta(days=1))
        # 断言 ts 是季度末
        assert ts.is_quarter_end
        # 断言频率是否将 ts 所在日期视为季度末
        assert not freq.is_quarter_end(ts)
        # 断言在 ts 前一天，频率将其视为季度末
        assert freq.is_quarter_end(ts - Timedelta(days=1))

    # 参数化测试方法，测试各个时间戳字段
    @pytest.mark.parametrize(
        "attr, expected",
        [
            ["year", 2014],
            ["month", 12],
            ["day", 31],
            ["hour", 23],
            ["minute", 59],
            ["second", 0],
            ["microsecond", 0],
            ["nanosecond", 0],
            ["dayofweek", 2],
            ["day_of_week", 2],
            ["quarter", 4],
            ["dayofyear", 365],
            ["day_of_year", 365],
            ["week", 1],
            ["daysinmonth", 31],
        ],
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_fields(self, attr, expected, tz):
        # 创建带时区信息的 Timestamp 对象 ts
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        # 获取属性值
        result = getattr(ts, attr)
        # 断言结果为整数类型
        assert isinstance(result, int)
        # 断言结果与期望值相符
        assert result == expected

    # 参数化测试方法，测试 millisecond 属性抛出异常情况
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_millisecond_raises(self, tz):
        # 创建带时区信息的 Timestamp 对象 ts
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        # 断言访问 millisecond 属性时抛出 AttributeError 异常
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond

    # 参数化测试方法，测试 start 参数的各个取值
    @pytest.mark.parametrize(
        "start", ["is_month_start", "is_quarter_start", "is_year_start"]
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    # 定义一个测试方法，用于检查指定时刻是否符合给定的起始条件
    def test_is_start(self, start, tz):
        # 创建一个时间戳对象，表示"2014-01-01 00:00:00"，使用指定的时区
        ts = Timestamp("2014-01-01 00:00:00", tz=tz)
        # 断言该时间戳对象是否具有指定的起始特征（由参数 `start` 指定）
        assert getattr(ts, start)

    # 使用参数化测试标记，测试指定时间戳是否符合给定的结束条件
    @pytest.mark.parametrize("end", ["is_month_end", "is_year_end", "is_quarter_end"])
    # 参数化测试标记，测试不同的时区对结束条件的影响
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_end(self, end, tz):
        # 创建一个时间戳对象，表示"2014-12-31 23:59:59"，使用指定的时区
        ts = Timestamp("2014-12-31 23:59:59", tz=tz)
        # 断言该时间戳对象是否具有指定的结束特征（由参数 `end` 指定）
        assert getattr(ts, end)

    # GH 12806
    @pytest.mark.parametrize("tz", [None, "EST"])
    # 错误：不支持的操作类型 +（"List[None]" 和 "List[str]"）
    @pytest.mark.parametrize(
        "time_locale",
        [None] + tm.get_locales(),  # type: ignore[operator]
    )
    def test_names(self, tz, time_locale):
        # GH 17354
        # 测试 .day_name() 和 .month_name()
        data = Timestamp("2017-08-28 23:00:00", tz=tz)
        if time_locale is None:
            expected_day = "Monday"
            expected_month = "August"
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                # 使用当前区域设置获取星期和月份的名称
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()

        # 获取时间戳对象的星期名称和月份名称
        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)

        # 解决 https://github.com/pandas-dev/pandas/issues/22342 的问题
        # 不同的规范化方式
        expected_day = unicodedata.normalize("NFD", expected_day)
        expected_month = unicodedata.normalize("NFD", expected_month)

        result_day = unicodedata.normalize("NFD", result_day)
        result_month = unicodedata.normalize("NFD", result_month)

        # 断言计算得到的星期名称和月份名称与预期相符
        assert result_day == expected_day
        assert result_month == expected_month

        # 测试 NaT（Not a Time）情况
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    # 测试是否为闰年的方法
    def test_is_leap_year(self, tz_naive_fixture):
        tz = tz_naive_fixture
        if not IS64 and tz == tzlocal():
            # https://github.com/dateutil/dateutil/issues/197
            # 在32位平台上使用tzlocal()会导致内部溢出错误
            pytest.skip(
                "tzlocal() on a 32 bit platform causes internal overflow errors"
            )
        # GH 13727
        # 创建一个时间戳对象，表示"2000-01-01 00:00:00"，使用指定的时区
        dt = Timestamp("2000-01-01 00:00:00", tz=tz)
        # 断言该时间戳对象是否为闰年
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)

        # 创建一个时间戳对象，表示"1999-01-01 00:00:00"，使用指定的时区
        dt = Timestamp("1999-01-01 00:00:00", tz=tz)
        # 断言该时间戳对象是否不是闰年
        assert not dt.is_leap_year

        # 创建一个时间戳对象，表示"2004-01-01 00:00:00"，使用指定的时区
        dt = Timestamp("2004-01-01 00:00:00", tz=tz)
        # 断言该时间戳对象是否为闰年
        assert dt.is_leap_year

        # 创建一个时间戳对象，表示"2100-01-01 00:00:00"，使用指定的时区
        dt = Timestamp("2100-01-01 00:00:00", tz=tz)
        # 断言该时间戳对象是否不是闰年
        assert not dt.is_leap_year
    def test_woy_boundary(self):
        # 确保年边界处的周数计算是正确的

        # 测试日期为2013年12月31日
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1  # ISO标准
        assert result == expected

        # 测试日期为2008年12月28日
        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52  # ISO标准
        assert result == expected

        # 测试日期为2009年12月31日
        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53  # ISO标准
        assert result == expected

        # 测试日期为2010年1月1日
        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53  # ISO标准
        assert result == expected

        # 测试日期为2010年1月3日
        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53  # ISO标准
        assert result == expected

        # 测试多个日期并使用numpy数组验证周数计算是否正确
        result = np.array(
            [
                Timestamp(datetime(*args)).week
                for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]
            ]
        )
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self):
        # GH#21336, GH#21365
        # 设置一个日期时间对象dt
        dt = Timestamp("2100-01-01 00:00:00.000000000")
        assert dt.resolution == Timedelta(nanoseconds=1)

        # 检查时间戳类上的resolution属性，应与标准库datetime行为一致
        assert Timestamp.resolution == Timedelta(nanoseconds=1)

        # 检查不同时间单位下的分辨率
        assert dt.as_unit("us").resolution == Timedelta(microseconds=1)
        assert dt.as_unit("ms").resolution == Timedelta(milliseconds=1)
        assert dt.as_unit("s").resolution == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "date_string, expected",
        [
            ("0000-2-29", 1),
            ("0000-3-1", 2),
            ("1582-10-14", 3),
            ("-0040-1-1", 4),
            ("2023-06-18", 6),
        ],
    )
    def test_dow_historic(self, date_string, expected):
        # GH 53738
        # 使用给定的日期字符串创建时间戳对象
        ts = Timestamp(date_string)
        # 获取日期的星期几
        dow = ts.weekday()
        assert dow == expected

    @given(
        ts=st.datetimes(),
        sign=st.sampled_from(["-", ""]),
    )
    def test_dow_parametric(self, ts, sign):
        # GH 53738
        # 构建日期时间字符串
        ts = (
            f"{sign}{str(ts.year).zfill(4)}"
            f"-{str(ts.month).zfill(2)}"
            f"-{str(ts.day).zfill(2)}"
        )
        # 获取日期的星期几
        result = Timestamp(ts).weekday()
        # 计算期望值
        expected = (
            (np.datetime64(ts) - np.datetime64("1970-01-01")).astype("int64") - 4
        ) % 7
        assert result == expected
    @pytest.mark.parametrize("tz", [None, zoneinfo.ZoneInfo("US/Pacific")])
    # 使用 pytest 的 parametrize 装饰器为 test_disallow_setting_tz 方法提供两种参数化测试：tz 为 None 或 "US/Pacific" 时的测试
    def test_disallow_setting_tz(self, tz):
        # GH#3746
        # 创建一个 Timestamp 对象，表示时间戳为 "2010"
        ts = Timestamp("2010")
        # 准备一个错误信息字符串
        msg = "Cannot directly set timezone"
        # 使用 pytest 的 raises 断言检测是否会抛出 AttributeError 异常，并匹配错误信息 msg
        with pytest.raises(AttributeError, match=msg):
            # 尝试为 ts 对象设置 tz 属性为 tz
            ts.tz = tz

    def test_default_to_stdlib_utc(self):
        # 准备一个警告信息字符串
        msg = "Timestamp.utcnow is deprecated"
        # 使用 tm.assert_produces_warning 上下文管理器检测是否会产生 FutureWarning 警告，并匹配警告信息 msg
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 调用 Timestamp.utcnow() 方法，并验证其 tz 属性是否为 timezone.utc
            assert Timestamp.utcnow().tz is timezone.utc
        # 调用 Timestamp.now("UTC") 方法，并验证其 tz 属性是否为 timezone.utc
        assert Timestamp.now("UTC").tz is timezone.utc
        # 创建一个带有时区为 "UTC" 的 Timestamp 对象，并验证其 tz 属性是否为 timezone.utc
        assert Timestamp("2016-01-01", tz="UTC").tz is timezone.utc

    def test_tz(self):
        # 准备一个时间字符串
        tstr = "2014-02-01 09:00"
        # 创建一个 Timestamp 对象，表示时间戳为 tstr
        ts = Timestamp(tstr)
        # 将 ts 对象本地化到 "Asia/Tokyo" 时区，并验证本地化后的小时数是否为 9
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        # 验证 local 对象与指定时区的 Timestamp 对象是否相等
        assert local == Timestamp(tstr, tz="Asia/Tokyo")
        # 将 local 对象转换为 "US/Eastern" 时区，并验证转换后的时间是否符合预期
        conv = local.tz_convert("US/Eastern")
        assert conv == Timestamp("2014-01-31 19:00", tz="US/Eastern")
        assert conv.hour == 19

        # 保留纳秒部分，创建一个带有纳秒的 Timestamp 对象
        ts = Timestamp(tstr) + offsets.Nano(5)
        # 将带有纳秒的 ts 对象本地化到 "Asia/Tokyo" 时区，并验证本地化后的小时数及纳秒部分是否符合预期
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local.nanosecond == 5
        # 将 local 对象转换为 "US/Eastern" 时区，并验证转换后的小时数及纳秒部分是否符合预期
        conv = local.tz_convert("US/Eastern")
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self):
        # 创建一个带有 "Z" 时区设计符号的 Timestamp 对象，并验证其时区信息是否为 timezone.utc
        assert get_timezone(Timestamp("2014-11-02 01:00Z").tzinfo) is timezone.utc

    def test_asm8(self):
        # 准备一个时间戳的列表
        ns = [Timestamp.min._value, Timestamp.max._value, 1000]

        # 遍历时间戳列表
        for n in ns:
            # 验证 Timestamp(n) 创建的时间戳的 asm8 属性与 numpy 创建的同等时间戳的 asm8 属性是否相等
            assert (
                Timestamp(n).asm8.view("i8") == np.datetime64(n, "ns").view("i8") == n
            )

        # 验证 "nat" 表示的时间戳的 asm8 属性与 numpy 创建的同等时间戳的 asm8 属性是否相等
        assert Timestamp("nat").asm8.view("i8") == np.datetime64("nat", "ns").view("i8")
    def test_class_ops(self):
        # 定义比较函数，用于比较两个时间戳对象的值是否相等
        def compare(x, y):
            assert int((Timestamp(x)._value - Timestamp(y)._value) / 1e9) == 0

        # 比较当前时间戳和当前日期时间对象的值是否相等
        compare(Timestamp.now(), datetime.now())
        # 比较当前 UTC 时间戳和当前 UTC 日期时间对象的值是否相等
        compare(Timestamp.now("UTC"), datetime.now(timezone.utc))
        # 比较当前 UTC 时间戳和当前 UTC 日期时间对象的值是否相等，使用 tzutc()
        compare(Timestamp.now("UTC"), datetime.now(tzutc()))
        
        # 检查 Timestamp.utcnow() 方法是否会产生警告
        msg = "Timestamp.utcnow is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            compare(Timestamp.utcnow(), datetime.now(timezone.utc))
        
        # 比较当前时间戳和当前日期时间对象的值是否相等
        compare(Timestamp.today(), datetime.today())
        
        # 获取当前时间的时间戳值（UTC）
        current_time = calendar.timegm(datetime.now().utctimetuple())
        
        # 检查 Timestamp.utcfromtimestamp() 方法是否会产生警告
        msg = "Timestamp.utcfromtimestamp is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ts_utc = Timestamp.utcfromtimestamp(current_time)
        # 断言时间戳的秒级时间是否与当前时间一致
        assert ts_utc.timestamp() == current_time
        
        # 比较从时间戳创建的时间对象和从时间戳创建的日期时间对象的值是否相等
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )
        
        # 使用 tz 关键字参数，比较从时间戳创建的 UTC 时间对象和从时间戳创建的 UTC 日期时间对象的值是否相等
        compare(
            Timestamp.fromtimestamp(current_time, "UTC"),
            datetime.fromtimestamp(current_time, timezone.utc),
        )
        
        # 使用 tz 关键字参数，比较从时间戳创建的 UTC 时间对象和从时间戳创建的 UTC 日期时间对象的值是否相等
        compare(
            Timestamp.fromtimestamp(current_time, tz="UTC"),
            datetime.fromtimestamp(current_time, timezone.utc),
        )

        # 获取当前 UTC 时间，并分别获取日期和时间部分
        date_component = datetime.now(timezone.utc)
        time_component = (date_component + timedelta(minutes=10)).time()
        # 比较将日期和时间组合成时间戳对象和日期时间对象的值是否相等
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )
    def test_roundtrip(self):
        # 测试值转换为字符串再转回的过程
        # 进一步测试访问器功能
        # 创建基准时间戳，精确到纳秒
        base = Timestamp("20140101 00:00:00").as_unit("ns")

        # 测试增加5毫秒后的时间戳
        result = Timestamp(base._value + Timedelta("5ms")._value)
        assert result == Timestamp(f"{base}.005000")
        assert result.microsecond == 5000

        # 测试增加5微秒后的时间戳
        result = Timestamp(base._value + Timedelta("5us")._value)
        assert result == Timestamp(f"{base}.000005")
        assert result.microsecond == 5

        # 测试增加5纳秒后的时间戳
        result = Timestamp(base._value + Timedelta("5ns")._value)
        assert result == Timestamp(f"{base}.000000005")
        assert result.nanosecond == 5
        assert result.microsecond == 0

        # 测试增加6毫秒5微秒后的时间戳
        result = Timestamp(base._value + Timedelta("6ms 5us")._value)
        assert result == Timestamp(f"{base}.006005")
        assert result.microsecond == 5 + 6 * 1000

        # 测试增加200毫秒5微秒后的时间戳
        result = Timestamp(base._value + Timedelta("200ms 5us")._value)
        assert result == Timestamp(f"{base}.200005")
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self):
        # 创建包含日期时间对象的字典
        d = {datetime(2011, 1, 1): 5}
        # 创建时间戳对象并使用日期时间对象作为键进行测试
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize(
        "timezone, year, month, day, hour",
        [["America/Chicago", 2013, 11, 3, 1], ["America/Santiago", 2021, 4, 3, 23]],
    )
    def test_hash_timestamp_with_fold(self, timezone, year, month, day, hour):
        # 见gh-33931
        # 获取指定时区的时区对象
        test_timezone = gettz(timezone)
        # 创建两个时间戳对象，分别表示时区转换点1和点2
        transition_1 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=0,
            tzinfo=test_timezone,
        )
        transition_2 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=1,
            tzinfo=test_timezone,
        )
        # 断言两个具有不同折叠标志的时间戳对象的哈希值相等
        assert hash(transition_1) == hash(transition_2)
class TestTimestampNsOperations:
    def test_nanosecond_string_parsing(self):
        # 创建 Timestamp 对象，解析指定的时间字符串
        ts = Timestamp("2013-05-01 07:15:45.123456789")
        # GH 7878
        # 预期的字符串表示
        expected_repr = "2013-05-01 07:15:45.123456789"
        # 预期的时间戳值（纳秒）
        expected_value = 1_367_392_545_123_456_789
        # 断言 Timestamp 对象的内部值与预期值相等
        assert ts._value == expected_value
        # 断言预期的字符串表示在 Timestamp 对象的字符串表示中
        assert expected_repr in repr(ts)

        # 创建带时区信息的 Timestamp 对象
        ts = Timestamp("2013-05-01 07:15:45.123456789+09:00", tz="Asia/Tokyo")
        # 断言 Timestamp 对象的内部值考虑时区修正后与预期值相等
        assert ts._value == expected_value - 9 * 3600 * 1_000_000_000
        # 断言预期的字符串表示在 Timestamp 对象的字符串表示中
        assert expected_repr in repr(ts)

        # 创建 UTC 时区的 Timestamp 对象
        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="UTC")
        # 断言 Timestamp 对象的内部值与预期值相等
        assert ts._value == expected_value
        # 断言预期的字符串表示在 Timestamp 对象的字符串表示中
        assert expected_repr in repr(ts)

        # 创建 US/Eastern 时区的 Timestamp 对象
        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="US/Eastern")
        # 断言 Timestamp 对象的内部值考虑时区修正后与预期值相等
        assert ts._value == expected_value + 4 * 3600 * 1_000_000_000
        # 断言预期的字符串表示在 Timestamp 对象的字符串表示中
        assert expected_repr in repr(ts)

        # GH 10041
        # 创建不带时区信息的 Timestamp 对象（从紧凑格式字符串）
        ts = Timestamp("20130501T071545.123456789")
        # 断言 Timestamp 对象的内部值与预期值相等
        assert ts._value == expected_value
        # 断言预期的字符串表示在 Timestamp 对象的字符串表示中
        assert expected_repr in repr(ts)

    def test_nanosecond_timestamp(self):
        # GH 7610
        # 预期的时间戳值（纳秒）
        expected = 1_293_840_000_000_000_005
        # 创建 Timestamp 对象，并添加纳秒级偏移
        t = Timestamp("2011-01-01") + offsets.Nano(5)
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 5
        assert t.nanosecond == 5

        # 创建新的 Timestamp 对象，作为现有 Timestamp 对象的副本
        t = Timestamp(t)
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 5
        assert t.nanosecond == 5

        # 创建指定时间的 Timestamp 对象
        t = Timestamp("2011-01-01 00:00:00.000000005")
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 5
        assert t.nanosecond == 5

        # 预期的时间戳值（纳秒）
        expected = 1_293_840_000_000_000_010
        # 在现有 Timestamp 对象上添加更多的纳秒级偏移
        t = t + offsets.Nano(5)
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 10
        assert t.nanosecond == 10

        # 创建新的 Timestamp 对象，作为现有 Timestamp 对象的副本
        t = Timestamp(t)
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 10
        assert t.nanosecond == 10

        # 创建指定时间的 Timestamp 对象
        t = Timestamp("2011-01-01 00:00:00.000000010")
        # 断言 Timestamp 对象的字符串表示符合预期
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        # 断言 Timestamp 对象的内部值与预期值相等
        assert t._value == expected
        # 断言 Timestamp 对象的纳秒属性为 10


class TestTimestampConversion:
    def test_conversion(self):
        # GH#9255
        # 创建 Timestamp 对象，并将其单位转换为纳秒
        ts = Timestamp("2000-01-01").as_unit("ns")

        # 将 Timestamp 对象转换为 Python 的 datetime 对象
        result = ts.to_pydatetime()
        expected = datetime(2000, 1, 1)
        # 断言转换结果与预期的 datetime 对象相等
        assert result == expected
        # 断言转换结果与预期的对象类型相同
        assert type(result) == type(expected)

        # 将 Timestamp 对象转换为 NumPy 的 datetime64 对象
        result = ts.to_datetime64()
        expected = np.datetime64(ts._value, "ns")
        # 断言转换结果与预期的 datetime64 对象相等
        assert result == expected
        # 断言转换结果与预期的对象类型相同
        assert type(result) == type(expected)
        # 断言转换结果的数据类型与预期的数据类型相同
        assert result.dtype == expected.dtype
    def test_to_period_tz_warning(self):
        # 测试时区信息丢失时是否发出警告
        ts = Timestamp("2009-04-15 16:17:18", tz="US/Eastern")
        # 使用断言确保在时区信息丢失时会发出警告
        with tm.assert_produces_warning(UserWarning, match="drop timezone information"):
            ts.to_period("D")

    def test_to_numpy_alias(self):
        # GH 24653: 为标量引入 .to_numpy() 别名
        ts = Timestamp(datetime.now())
        # 断言 .to_numpy() 方法等同于 .to_datetime64()
        assert ts.to_datetime64() == ts.to_numpy()

        # GH#44460
        # 检查 "dtype" 和 "copy" 参数是否被忽略，应引发 ValueError 异常
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy("M8[s]")
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy(copy=True)
    # 定义 TestNonNano 类，用于测试非纳秒精度的时间戳功能
    class TestNonNano:
        # 定义 reso 作为 pytest 的参数化 fixture，用于提供时间分辨率参数
        @pytest.fixture(params=["s", "ms", "us"])
        def reso(self, request):
            return request.param

        # 定义 dt64 fixture，根据给定的分辨率参数创建 np.datetime64 对象
        def dt64(self, reso):
            # 创建并返回 np.datetime64 对象，日期设为 "2016-01-01"
            return np.datetime64("2016-01-01", reso)

        # 定义 ts fixture，通过 Timestamp._from_dt64 方法创建时间戳对象
        def ts(self, dt64):
            return Timestamp._from_dt64(dt64)

        # 定义 ts_tz fixture，使用给定的时区信息创建带时区的时间戳对象
        def ts_tz(self, ts, tz_aware_fixture):
            # 从给定的时间戳对象 ts 中提取数值和分辨率，以及可能的时区信息创建时间戳对象
            tz = maybe_get_tz(tz_aware_fixture)
            return Timestamp._from_value_and_reso(ts._value, ts._creso, tz)

        # 测试非纳秒精度时间戳的构造方法
        def test_non_nano_construction(self, dt64, ts, reso):
            # 断言时间戳对象的数值等于 np.datetime64 对象的整型视图
            assert ts._value == dt64.view("i8")

            # 根据不同的时间分辨率参数进行断言，验证时间戳的精度单位
            if reso == "s":
                assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
            elif reso == "ms":
                assert ts._creso == NpyDatetimeUnit.NPY_FR_ms.value
            elif reso == "us":
                assert ts._creso == NpyDatetimeUnit.NPY_FR_us.value

        # 测试非纳秒精度时间戳的日期字段
        def test_non_nano_fields(self, dt64, ts):
            # 创建另一个 Timestamp 对象 alt 作为对比对象
            alt = Timestamp(dt64)

            # 断言时间戳对象的年、月、日字段与 alt 相同
            assert ts.year == alt.year
            assert ts.month == alt.month
            assert ts.day == alt.day

            # 断言时间戳对象的时、分、秒、微秒字段为 0
            assert ts.hour == ts.minute == ts.second == ts.microsecond == 0
            # 断言时间戳对象的纳秒字段为 0
            assert ts.nanosecond == 0

            # 断言时间戳对象的儒略日与 alt 相同
            assert ts.to_julian_date() == alt.to_julian_date()
            # 断言时间戳对象的星期几与 alt 相同
            assert ts.weekday() == alt.weekday()
            # 断言时间戳对象的 ISO 星期几与 alt 相同
            assert ts.isoweekday() == alt.isoweekday()

        # 测试时间戳对象的年/季度/月起始字段
        def test_start_end_fields(self, ts):
            # 断言时间戳对象在年/季度/月开始位置
            assert ts.is_year_start
            assert ts.is_quarter_start
            assert ts.is_month_start
            # 断言时间戳对象不在年/月结束位置
            assert not ts.is_year_end
            assert not ts.is_month_end
            assert not ts.is_month_end

            # 2016-01-01 是星期五，因此根据此频率断言时间戳对象在年/季度/月开始位置
            assert ts.is_year_start
            assert ts.is_quarter_start
            assert ts.is_month_start
            # 断言时间戳对象不在年/月结束位置
            assert not ts.is_year_end
            assert not ts.is_month_end
            assert not ts.is_month_end

        # 测试时间戳对象的星期几名称
        def test_day_name(self, dt64, ts):
            # 创建另一个 Timestamp 对象 alt 作为对比对象
            alt = Timestamp(dt64)
            # 断言时间戳对象的星期几名称与 alt 相同
            assert ts.day_name() == alt.day_name()

        # 测试时间戳对象的月份名称
        def test_month_name(self, dt64, ts):
            # 创建另一个 Timestamp 对象 alt 作为对比对象
            alt = Timestamp(dt64)
            # 断言时间戳对象的月份名称与 alt 相同
            assert ts.month_name() == alt.month_name()

        # 测试时间戳对象的时区转换功能
        def test_tz_convert(self, ts):
            # 将时间戳对象转换为使用 UTC 时区
            ts = Timestamp._from_value_and_reso(ts._value, ts._creso, timezone.utc)

            # 创建 US/Pacific 时区对象
            tz = zoneinfo.ZoneInfo("US/Pacific")
            # 执行时间戳对象的时区转换
            result = ts.tz_convert(tz)

            # 断言结果是 Timestamp 对象
            assert isinstance(result, Timestamp)
            # 断言结果的精度单位与原始时间戳对象相同
            assert result._creso == ts._creso
            # 使用 tz_compare 函数断言结果的时区与预期时区 tz 相同

        # 测试时间戳对象的字符串表示形式
        def test_repr(self, dt64, ts):
            # 创建另一个 Timestamp 对象 alt 作为对比对象
            alt = Timestamp(dt64)

            # 断言时间戳对象的字符串表示与 alt 相同
            assert str(ts) == str(alt)
            # 断言时间戳对象的 repr 表示与 alt 相同
            assert repr(ts) == repr(alt)
    def test_comparison(self, dt64, ts):
        # 创建一个新的 Timestamp 对象
        alt = Timestamp(dt64)

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的相等性
        assert ts == dt64
        assert dt64 == ts
        assert ts == alt
        assert alt == ts

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的不等性
        assert not ts != dt64
        assert not dt64 != ts
        assert not ts != alt
        assert not alt != ts

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的大小关系
        assert not ts < dt64
        assert not dt64 < ts
        assert not ts < alt
        assert not alt < ts

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的大小关系
        assert not ts > dt64
        assert not dt64 > ts
        assert not ts > alt
        assert not alt > ts

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的大于等于关系
        assert ts >= dt64
        assert dt64 >= ts
        assert ts >= alt
        assert alt >= ts

        # 下面一系列断言用于比较 Timestamp 对象 ts 和 dt64 的小于等于关系
        assert ts <= dt64
        assert dt64 <= ts
        assert ts <= alt
        assert alt <= ts

    def test_cmp_cross_reso(self):
        # numpy 由于静默溢出而出错
        dt64 = np.datetime64(9223372800, "s")  # 无法适应 M8[ns]
        ts = Timestamp._from_dt64(dt64)

        # 减去 3600*24 后得到一个可以适应纳秒实现边界的 datetime64
        other = Timestamp(dt64 - 3600 * 24).as_unit("ns")
        assert other < ts
        assert other.asm8 > ts.asm8  # <- numpy 在这里出错
        assert ts > other
        assert ts.asm8 < other.asm8  # <- numpy 在这里出错
        assert not other == ts
        assert ts != other

    @pytest.mark.xfail(reason="Dispatches to np.datetime64 which is wrong")
    def test_cmp_cross_reso_reversed_dt64(self):
        dt64 = np.datetime64(106752, "D")  # 无法适应 M8[ns]
        ts = Timestamp._from_dt64(dt64)
        other = Timestamp(dt64 - 1)

        assert other.asm8 < ts

    def test_pickle(self, ts, tz_aware_fixture):
        # 获取时区
        tz = tz_aware_fixture
        tz = maybe_get_tz(tz)
        # 从值和分辨率创建 Timestamp 对象
        ts = Timestamp._from_value_and_reso(ts._value, ts._creso, tz)
        # 进行 pickle 的来回测试
        rt = tm.round_trip_pickle(ts)
        assert rt._creso == ts._creso
        assert rt == ts

    def test_normalize(self, dt64, ts):
        # 创建一个新的 Timestamp 对象
        alt = Timestamp(dt64)
        # 调用 normalize 方法
        result = ts.normalize()
        assert result._creso == ts._creso
        assert result == alt.normalize()

    def test_asm8(self, dt64, ts):
        # 调用 asm8 属性
        rt = ts.asm8
        assert rt == dt64
        assert rt.dtype == dt64.dtype

    def test_to_numpy(self, dt64, ts):
        # 调用 to_numpy 方法
        res = ts.to_numpy()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_to_datetime64(self, dt64, ts):
        # 调用 to_datetime64 方法
        res = ts.to_datetime64()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_timestamp(self, dt64, ts):
        # 创建一个新的 Timestamp 对象
        alt = Timestamp(dt64)
        # 调用 timestamp 方法进行比较
        assert ts.timestamp() == alt.timestamp()

    def test_to_period(self, dt64, ts):
        # 创建一个新的 Timestamp 对象
        alt = Timestamp(dt64)
        # 调用 to_period 方法
        assert ts.to_period("D") == alt.to_period("D")

    @pytest.mark.parametrize(
        "td", [timedelta(days=4), Timedelta(days=4), np.timedelta64(4, "D")]
    )
    # 测试方法：测试时间戳对象的加减操作，不考虑纳秒级精度
    def test_addsub_timedeltalike_non_nano(self, dt64, ts, td):
        # 计算预期的分辨率，选择最大的时间戳分辨率
        exp_reso = max(ts._creso, Timedelta(td)._creso)

        # 执行时间戳减去时间增量操作
        result = ts - td
        # 计算预期结果
        expected = Timestamp(dt64) - td
        # 断言结果类型为时间戳
        assert isinstance(result, Timestamp)
        # 断言结果分辨率与预期一致
        assert result._creso == exp_reso
        # 断言结果与预期值相等
        assert result == expected

        # 执行时间戳加上时间增量操作
        result = ts + td
        # 计算预期结果
        expected = Timestamp(dt64) + td
        # 断言结果类型为时间戳
        assert isinstance(result, Timestamp)
        # 断言结果分辨率与预期一致
        assert result._creso == exp_reso
        # 断言结果与预期值相等
        assert result == expected

        # 执行时间增量加上时间戳操作
        result = td + ts
        # 计算预期结果
        expected = td + Timestamp(dt64)
        # 断言结果类型为时间戳
        assert isinstance(result, Timestamp)
        # 断言结果分辨率与预期一致
        assert result._creso == exp_reso
        # 断言结果与预期值相等
        assert result == expected

    # 测试方法：测试时间戳对象与时间偏移量的加减操作
    def test_addsub_offset(self, ts_tz):
        # 创建一个年末偏移量对象，单位为年，非Tick精度
        off = offsets.YearEnd(1)
        # 执行时间戳加上偏移量操作
        result = ts_tz + off

        # 断言结果类型为时间戳
        assert isinstance(result, Timestamp)
        # 断言结果分辨率与原时间戳一致
        assert result._creso == ts_tz._creso
        # 如果原时间戳为12月31日，则结果年份应为原年份加一年，否则为原年份
        if ts_tz.month == 12 and ts_tz.day == 31:
            assert result.year == ts_tz.year + 1
        else:
            assert result.year == ts_tz.year
        # 结果日期应为12月31日
        assert result.day == 31
        # 结果月份应为12月
        assert result.month == 12
        # 比较时区是否与原时间戳相同
        assert tz_compare(result.tz, ts_tz.tz)

        # 执行时间戳减去偏移量操作
        result = ts_tz - off

        # 断言结果类型为时间戳
        assert isinstance(result, Timestamp)
        # 断言结果分辨率与原时间戳一致
        assert result._creso == ts_tz._creso
        # 结果年份应为原年份减一年
        assert result.year == ts_tz.year - 1
        # 结果日期应为12月31日
        assert result.day == 31
        # 结果月份应为12月
        assert result.month == 12
        # 比较时区是否与原时间戳相同
        assert tz_compare(result.tz, ts_tz.tz)
    def test_sub_datetimelike_mismatched_reso(self, ts_tz):
        # 定义一个测试方法，用于测试不匹配分辨率的日期时间对象
        ts = ts_tz  # 将传入的日期时间对象赋给变量 ts

        # 选择一个 `other` 单位，使其与 ts_tz 的单位不匹配；
        # 这种构造确保我们能测试 other._creso < ts._creso 和 other._creso > ts._creso 的情况
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._creso]
        other = ts.as_unit(unit)  # 使用选定的单位构造 other 对象
        assert other._creso != ts._creso  # 断言确保 other 的分辨率不等于 ts 的分辨率

        result = ts - other
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result._value == 0  # 断言 Timedelta 对象的值为 0
        assert result._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率

        result = other - ts
        assert isinstance(result, Timedelta)  # 断言结果是 Timedelta 类型
        assert result._value == 0  # 断言 Timedelta 对象的值为 0
        assert result._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率

        if ts._creso < other._creso:
            # 当分辨率丢失时的情况
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) - other2  # 期望的结果

            res = ts - other2
            assert res == exp  # 断言结果等于期望的结果
            assert res._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率

            res = other2 - ts
            assert res == -exp  # 断言结果等于期望的结果的相反数
            assert res._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 - other.as_unit(ts2.unit)  # 期望的结果

            res = ts2 - other
            assert res == exp  # 断言结果等于期望的结果
            assert res._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率

            res = other - ts2
            assert res == -exp  # 断言结果等于期望的结果的相反数
            assert res._creso == max(ts._creso, other._creso)  # 断言 Timedelta 对象的分辨率为 ts 和 other 的最大分辨率
    # 测试函数：检查在时间单位不匹配的情况下，时间戳与时间增量的加法运算是否正确
    def test_sub_timedeltalike_mismatched_reso(self, ts_tz):
        # 使用传入的时间戳进行测试
        ts = ts_tz

        # 选择一个时间单位 `other`，使其与 `ts_tz` 的单位不匹配；
        # 这种构造确保我们既有 `other._creso < ts._creso` 的情况，
        # 也有 `other._creso > ts._creso` 的情况
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._creso]
        other = Timedelta(0).as_unit(unit)
        assert other._creso != ts._creso

        # 测试 `ts` 与 `other` 的加法运算
        result = ts + other
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)

        # 测试 `other` 与 `ts` 的加法运算
        result = other + ts
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)

        if ts._creso < other._creso:
            # 当精度损失的情况
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) + other2

            # 测试加法运算后的结果与预期结果是否一致
            res = ts + other2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

            # 测试另一种顺序的加法运算后的结果与预期结果是否一致
            res = other2 + ts
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
        else:
            # 当精度不损失的情况
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 + other.as_unit(ts2.unit)

            # 测试加法运算后的结果与预期结果是否一致
            res = ts2 + other
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

            # 测试另一种顺序的加法运算后的结果与预期结果是否一致
            res = other + ts2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

    # 测试函数：确保加法操作不会降低时间精度
    def test_addition_doesnt_downcast_reso(self):
        # 创建一个时间戳 `ts`，其微秒数为 999999
        ts = Timestamp(year=2022, month=1, day=1, microsecond=999999).as_unit("us")
        # 创建一个微秒级的时间增量 `td`
        td = Timedelta(microseconds=1).as_unit("us")

        # 执行加法操作
        res = ts + td

        # 断言结果的时间精度与 `ts` 的时间精度相同
        assert res._creso == ts._creso

    # 测试函数：检查时间戳与 `np.timedelta64` 的加法操作在时间单位不匹配时的行为
    def test_sub_timedelta64_mismatched_reso(self, ts_tz):
        # 使用传入的时间戳进行测试
        ts = ts_tz

        # 执行时间戳与 `np.timedelta64` 加法操作
        res = ts + np.timedelta64(1, "ns")
        # 期望的结果
        exp = ts.as_unit("ns") + np.timedelta64(1, "ns")

        # 断言加法操作的结果与期望结果一致，并且时间精度为纳秒级
        assert exp == res
        assert exp._creso == NpyDatetimeUnit.NPY_FR_ns.value

    # 测试函数：验证时间戳的最小值属性
    def test_min(self, ts):
        # 断言时间戳的最小值小于等于时间戳本身，并且时间精度与时间戳本身相同
        assert ts.min <= ts
        assert ts.min._creso == ts._creso
        assert ts.min._value == NaT._value + 1

    # 测试函数：验证时间戳的最大值属性
    def test_max(self, ts):
        # 断言时间戳的最大值大于等于时间戳本身，并且时间精度与时间戳本身相同
        assert ts.max >= ts
        assert ts.max._creso == ts._creso
        assert ts.max._value == np.iinfo(np.int64).max

    # 测试函数：验证时间戳的分辨率属性
    def test_resolution(self, ts):
        # 期望的分辨率，从值和时间精度创建
        expected = Timedelta._from_value_and_reso(1, ts._creso)

        # 获取实际的分辨率
        result = ts.resolution

        # 断言实际的分辨率与期望的分辨率一致，并且时间精度相同
        assert result == expected
        assert result._creso == expected._creso
    def test_out_of_ns_bounds(self):
        # 定义一个测试方法来验证时间戳超出纳秒范围的行为
        # 在这个问题的 GitHub issue 中进行了讨论：https://github.com/pandas-dev/pandas/issues/51060
        
        # 创建一个时间戳对象，传入一个超出纳秒范围的时间戳值（-52700112000秒）
        result = Timestamp(-52700112000, unit="s")
        
        # 断言，验证生成的时间戳对象是否等于预期的日期时间对象 "0300-01-01"
        assert result == Timestamp("0300-01-01")
        
        # 断言，验证时间戳对象转换为 NumPy 的 datetime64 类型是否等于预期的日期时间 "0300-01-01T00:00:00"
        assert result.to_numpy() == np.datetime64("0300-01-01T00:00:00", "s")
# 测试 Timestamp 类的 min、max 和 resolution 方法
def test_timestamp_class_min_max_resolution():
    # 当通过类访问时（而不是实例），默认精度为纳秒
    assert Timestamp.min == Timestamp(NaT._value + 1)
    # 确认 Timestamp.min 的时间分辨率为纳秒
    assert Timestamp.min._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.max == Timestamp(np.iinfo(np.int64).max)
    # 确认 Timestamp.max 的时间分辨率为纳秒
    assert Timestamp.max._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.resolution == Timedelta(1)
    # 确认 Timestamp.resolution 的时间分辨率为纳秒
    assert Timestamp.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value


# 测试通过指定格式的日期字符串创建 Timestamp 的行为
def test_delimited_date():
    # 使用指定的日期字符串创建 Timestamp 对象，应无警告产生
    with tm.assert_produces_warning(None):
        result = Timestamp("13-01-2000")
    expected = Timestamp(2000, 1, 13)
    # 确认创建的 Timestamp 对象与预期的 Timestamp 对象相等
    assert result == expected


# 测试在设置时区为 UTC 的情况下，Timestamp 的 utctimetuple 方法
def test_utctimetuple():
    # 创建设置时区为 UTC 的 Timestamp 对象
    ts = Timestamp("2000-01-01", tz="UTC")
    # 调用 utctimetuple 方法获取结果
    result = ts.utctimetuple()
    # 预期的 struct_time 对象表示的时间
    expected = time.struct_time((2000, 1, 1, 0, 0, 0, 5, 1, 0))
    # 确认结果与预期相等
    assert result == expected


# 测试在负日期情况下对 Timestamp 的一些方法的行为
def test_negative_dates():
    # 创建一个负日期的 Timestamp 对象
    ts = Timestamp("-2000-01-01")
    # 预期的错误消息模式
    msg = (
        " not yet supported on Timestamps which are outside the range of "
        "Python's standard library. For now, please call the components you need "
        r"\(such as `.year` and `.month`\) and construct your string from there.$"
    )
    func = "^strftime"
    # 验证调用 strftime 方法会引发 NotImplementedError 异常，匹配特定错误消息
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.strftime("%Y")

    msg = (
        " not yet supported on Timestamps which "
        "are outside the range of Python's standard library. "
    )
    func = "^date"
    # 验证调用 date 方法会引发 NotImplementedError 异常，匹配特定错误消息
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.date()
    func = "^isocalendar"
    # 验证调用 isocalendar 方法会引发 NotImplementedError 异常，匹配特定错误消息
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.isocalendar()
    func = "^timetuple"
    # 验证调用 timetuple 方法会引发 NotImplementedError 异常，匹配特定错误消息
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.timetuple()
    func = "^toordinal"
    # 验证调用 toordinal 方法会引发 NotImplementedError 异常，匹配特定错误消息
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.toordinal()
```