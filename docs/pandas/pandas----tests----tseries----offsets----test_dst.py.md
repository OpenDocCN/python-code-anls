# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_dst.py`

```
"""
Tests for DateOffset additions over Daylight Savings Time
"""

# 从 datetime 模块导入 timedelta 类
from datetime import timedelta

# 导入 pytest 模块
import pytest

# 从 pandas._libs.tslibs 中导入 Timestamp 和各种时间偏移类
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CustomBusinessDay,
    DateOffset,
    Day,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    YearBegin,
    YearEnd,
)

# 从 pandas 模块导入 DatetimeIndex
from pandas import DatetimeIndex

# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm

# 导入 pytest 的 importorskip 函数并重命名为 pytz，用于导入 pytz 模块
pytz = pytest.importorskip("pytz")


def get_utc_offset_hours(ts):
    # 接收一个 Timestamp 对象，计算其 UTC 偏移的总小时数
    o = ts.utcoffset()
    return (o.days * 24 * 3600 + o.seconds) / 3600.0


class TestDST:
    # DST 转换前的一个微秒
    ts_pre_fallback = "2013-11-03 01:59:59.999999"
    # 春季前进时钟前的一个微秒
    ts_pre_springfwd = "2013-03-10 01:59:59.999999"

    # 测试基本名称和 dateutil 时区
    timezone_utc_offsets = {
        pytz.timezone("US/Eastern"): {
            "utc_offset_daylight": -4,
            "utc_offset_standard": -5,
        },
        "dateutil/US/Pacific": {"utc_offset_daylight": -7, "utc_offset_standard": -8},
    }

    # 单数形式的有效日期偏移量列表
    valid_date_offsets_singular = [
        "weekday",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    ]

    # 复数形式的有效日期偏移量列表
    valid_date_offsets_plural = [
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
    ]

    def _test_all_offsets(self, n, performance_warning, **kwds):
        # 根据 n 的值选择有效的日期偏移量列表
        valid_offsets = (
            self.valid_date_offsets_plural
            if n > 1
            else self.valid_date_offsets_singular
        )

        # 遍历有效的日期偏移量列表，对每个偏移量调用 _test_offset 方法进行测试
        for name in valid_offsets:
            self._test_offset(
                offset_name=name,
                offset_n=n,
                performance_warning=performance_warning,
                **kwds,
            )

    def _test_offset(
        self, offset_name, offset_n, tstart, expected_utc_offset, performance_warning
    ):
        # 在这里实现具体的日期偏移测试逻辑，此处省略具体代码
        pass  # 占位符，实际代码中需要填充具体的测试逻辑
        offset = DateOffset(**{offset_name: offset_n})
        # 使用给定的偏移名称和值创建 DateOffset 对象

        if (
            offset_name in ["hour", "minute", "second", "microsecond"]
            and offset_n == 1
            and tstart
            == Timestamp(
                "2013-11-03 01:59:59.999999-0500", tz=pytz.timezone("US/Eastern")
            )
        ):
            # 如果偏移名称是小时、分钟、秒或微秒，且偏移值为1，并且起始时间 tstart 是一个特定的时间戳
            # 此添加将导致一个模糊的墙上时间
            err_msg = {
                "hour": "2013-11-03 01:59:59.999999",
                "minute": "2013-11-03 01:01:59.999999",
                "second": "2013-11-03 01:59:01.999999",
                "microsecond": "2013-11-03 01:59:59.000001",
            }[offset_name]
            # 使用 pytest 检测是否会抛出 AmbiguousTimeError 异常，并匹配特定的错误消息
            with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
                tstart + offset
            # 同时，在这里检查我们是否在一个矢量化路径中获得相同的行为
            dti = DatetimeIndex([tstart])
            warn_msg = "Non-vectorized DateOffset"
            with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
                with tm.assert_produces_warning(performance_warning, match=warn_msg):
                    dti + offset
            return

        t = tstart + offset
        # 计算新的时间戳 t，加上偏移量 offset

        if expected_utc_offset is not None:
            assert get_utc_offset_hours(t) == expected_utc_offset
            # 断言 t 的 UTC 偏移小时数与预期的偏移量是否相等

        if offset_name == "weeks":
            # 断言日期应该匹配
            assert t.date() == timedelta(days=7 * offset.kwds["weeks"]) + tstart.date()
            # 预期同一周的日期、小时、分钟、秒等应该一致
            assert (
                t.dayofweek == tstart.dayofweek
                and t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name == "days":
            # 断言日期应该匹配
            assert timedelta(offset.kwds["days"]) + tstart.date() == t.date()
            # 预期同一天的小时、分钟、秒等应该一致
            assert (
                t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name in self.valid_date_offsets_singular:
            # 预期在 tstart 和 t 之间的单一偏移值匹配
            datepart_offset = getattr(
                t, offset_name if offset_name != "weekday" else "dayofweek"
            )
            assert datepart_offset == offset.kwds[offset_name]
        else:
            # 偏移量应该与在 UTC 时区执行时相同
            assert t == (tstart.tz_convert("UTC") + offset).tz_convert(
                pytz.timezone("US/Pacific")
            )
            # 使用 pytest 断言 t 等于将 tstart 转换为 UTC 时区后加上偏移量 offset，
            # 再转换为 US/Pacific 时区的结果
    # 测试多个时区在从标准时间到夏令时调整的情况下的偏移量变化
    def test_springforward_plural(self, performance_warning):
        # 遍历每个时区及其对应的UTC偏移量字典
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            # 获取夏令时调整前的UTC偏移量
            hrs_pre = utc_offsets["utc_offset_standard"]
            # 获取夏令时调整后的UTC偏移量
            hrs_post = utc_offsets["utc_offset_daylight"]
            # 调用测试函数，验证多种偏移量情况
            self._test_all_offsets(
                n=3,
                performance_warning=performance_warning,
                # 创建起始时间戳，考虑夏令时调整前的UTC偏移量和时区
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                # 期望的UTC偏移量设为夏令时调整后的偏移量
                expected_utc_offset=hrs_post,
            )

    # 测试单个时区在退回标准时间的情况
    def test_fallback_singular(self, performance_warning):
        # 对于单一偏移量情况，新的时间戳可能处于不同的UTC偏移量下，故不指定期望的UTC偏移量
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            # 获取标准时间的UTC偏移量
            hrs_pre = utc_offsets["utc_offset_standard"]
            # 调用测试函数，验证单种偏移量情况
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                # 创建起始时间戳，考虑标准时间退回前的UTC偏移量和时区
                tstart=self._make_timestamp(self.ts_pre_fallback, hrs_pre, tz),
                # 期望的UTC偏移量未指定
                expected_utc_offset=None,
            )

    # 测试单个时区在夏令时调整前的情况
    def test_springforward_singular(self, performance_warning):
        # 遍历每个时区及其对应的UTC偏移量字典
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            # 获取标准时间的UTC偏移量
            hrs_pre = utc_offsets["utc_offset_standard"]
            # 调用测试函数，验证单种偏移量情况
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                # 创建起始时间戳，考虑夏令时调整前的UTC偏移量和时区
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                # 期望的UTC偏移量未指定
                expected_utc_offset=None,
            )

    # 不同偏移量类的测试数据及其对应的测试时间戳
    offset_classes = {
        MonthBegin: ["11/2/2012", "12/1/2012"],
        MonthEnd: ["11/2/2012", "11/30/2012"],
        BMonthBegin: ["11/2/2012", "12/3/2012"],
        BMonthEnd: ["11/2/2012", "11/30/2012"],
        CBMonthBegin: ["11/2/2012", "12/3/2012"],
        CBMonthEnd: ["11/2/2012", "11/30/2012"],
        SemiMonthBegin: ["11/2/2012", "11/15/2012"],
        SemiMonthEnd: ["11/2/2012", "11/15/2012"],
        Week: ["11/2/2012", "11/9/2012"],
        YearBegin: ["11/2/2012", "1/1/2013"],
        YearEnd: ["11/2/2012", "12/31/2012"],
        BYearBegin: ["11/2/2012", "1/1/2013"],
        BYearEnd: ["11/2/2012", "12/31/2012"],
        QuarterBegin: ["11/2/2012", "12/1/2012"],
        QuarterEnd: ["11/2/2012", "12/31/2012"],
        BQuarterBegin: ["11/2/2012", "12/3/2012"],
        BQuarterEnd: ["11/2/2012", "12/31/2012"],
        Day: ["11/4/2012", "11/4/2012 23:00"],
    }.items()

    # 使用pytest的参数化装饰器，遍历测试不同的偏移量类
    @pytest.mark.parametrize("tup", offset_classes)
    def test_all_offset_classes(self, tup):
        # 分别获取偏移量类和对应的测试时间值
        offset, test_values = tup

        # 创建第一个时间戳，考虑偏移量类和时区为"US/Eastern"
        first = Timestamp(test_values[0], tz="US/Eastern") + offset()
        # 创建第二个时间戳，时区同样为"US/Eastern"
        second = Timestamp(test_values[1], tz="US/Eastern")
        # 断言两个时间戳相等
        assert first == second
# 使用 pytest.mark.parametrize 装饰器为 test_nontick_offset_with_ambiguous_time_error 函数提供多组参数化测试数据
@pytest.mark.parametrize(
    "original_dt, target_dt, offset, tz",
    [
        (
            Timestamp("2021-10-01 01:15"),  # 设置原始时间戳为 "2021-10-01 01:15"
            Timestamp("2021-10-31 01:15"),  # 设置目标时间戳为 "2021-10-31 01:15"
            MonthEnd(1),  # 使用 MonthEnd(1) 表示一个月的结束
            "Europe/London",  # 使用 "Europe/London" 时区
        ),
        (
            Timestamp("2010-12-05 02:59"),  # 设置原始时间戳为 "2010-12-05 02:59"
            Timestamp("2010-10-31 02:59"),  # 设置目标时间戳为 "2010-10-31 02:59"
            SemiMonthEnd(-3),  # 使用 SemiMonthEnd(-3) 表示半月末的向前3个单位
            "Europe/Paris",  # 使用 "Europe/Paris" 时区
        ),
        (
            Timestamp("2021-10-31 01:20"),  # 设置原始时间戳为 "2021-10-31 01:20"
            Timestamp("2021-11-07 01:20"),  # 设置目标时间戳为 "2021-11-07 01:20"
            CustomBusinessDay(2, weekmask="Sun Mon"),  # 使用 CustomBusinessDay(2, weekmask="Sun Mon") 表示自定义的工作日偏移
            "US/Eastern",  # 使用 "US/Eastern" 时区
        ),
        (
            Timestamp("2020-04-03 01:30"),  # 设置原始时间戳为 "2020-04-03 01:30"
            Timestamp("2020-11-01 01:30"),  # 设置目标时间戳为 "2020-11-01 01:30"
            YearBegin(1, month=11),  # 使用 YearBegin(1, month=11) 表示一年的开始，11月份
            "America/Chicago",  # 使用 "America/Chicago" 时区
        ),
    ],
)
# 定义测试函数 test_nontick_offset_with_ambiguous_time_error，测试非Tick偏移在目标时间为夏令时模糊时抛出 AmbiguousTimeError 异常
def test_nontick_offset_with_ambiguous_time_error(original_dt, target_dt, offset, tz):
    # 使用原始时间戳进行时区本地化，时区为给定的 tz 参数
    localized_dt = original_dt.tz_localize(pytz.timezone(tz))

    # 构造异常消息，指出无法从目标时间戳 target_dt 推断夏令时时间，建议使用 'ambiguous' 参数
    msg = f"Cannot infer dst time from {target_dt}, try using the 'ambiguous' argument"
    
    # 使用 pytest 的 pytest.raises 断言捕获 pytz.AmbiguousTimeError 异常，并验证异常消息与预期的 msg 匹配
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        localized_dt + offset
```