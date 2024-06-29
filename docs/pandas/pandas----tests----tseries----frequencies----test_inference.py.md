# `D:\src\scipysrc\pandas\pandas\tests\tseries\frequencies\test_inference.py`

```
# 从datetime模块中导入datetime和timedelta类
from datetime import (
    datetime,
    timedelta,
)

# 导入numpy库，并将其重命名为np
import numpy as np

# 导入pytest库
import pytest

# 从pandas._libs.tslibs.ccalendar中导入DAYS和MONTHS常量
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTHS,
)

# 从pandas._libs.tslibs.offsets中导入_get_offset函数
from pandas._libs.tslibs.offsets import _get_offset

# 从pandas._libs.tslibs.period中导入INVALID_FREQ_ERR_MSG常量
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

# 从pandas.compat中导入is_platform_windows函数
from pandas.compat import is_platform_windows

# 从pandas库中导入多个类和函数
from pandas import (
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)

# 从pandas.core.arrays中导入DatetimeArray和TimedeltaArray类
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)

# 从pandas.core.tools.datetimes中导入to_datetime函数
from pandas.core.tools.datetimes import to_datetime

# 从pandas.tseries中导入frequencies和offsets模块
from pandas.tseries import (
    frequencies,
    offsets,
)


# 使用pytest的fixture装饰器，参数化base_delta_code_pair函数，返回一个参数元组
@pytest.fixture(
    params=[
        (timedelta(1), "D"),
        (timedelta(hours=1), "h"),
        (timedelta(minutes=1), "min"),
        (timedelta(seconds=1), "s"),
        (np.timedelta64(1, "ns"), "ns"),
        (timedelta(microseconds=1), "us"),
        (timedelta(microseconds=1000), "ms"),
    ]
)
def base_delta_code_pair(request):
    return request.param


# 定义频率(freq)列表，包含各种可能的频率字符串
freqs = (
    [f"QE-{month}" for month in MONTHS]  # 每个月的季度末
    + [f"{annual}-{month}" for annual in ["YE", "BYE"] for month in MONTHS]  # 每年月末和年末
    + ["ME", "BME", "BMS"]  # 月末、工作日每月末、工作日每月开始
    + [f"WOM-{count}{day}" for count in range(1, 5) for day in DAYS]  # 每月第几周的工作日
    + [f"W-{day}" for day in DAYS]  # 每周的工作日
)


# 使用pytest的参数化装饰器，将freqs列表中的频率字符串(freq)和期数(periods)参数化
@pytest.mark.parametrize("freq", freqs)
@pytest.mark.parametrize("periods", [5, 7])
def test_infer_freq_range(periods, freq):
    freq = freq.upper()  # 将频率字符串转换为大写

    gen = date_range("1/1/2000", periods=periods, freq=freq)  # 生成日期范围
    index = DatetimeIndex(gen.values)  # 创建DatetimeIndex对象

    if not freq.startswith("QE-"):  # 如果频率字符串不以"QE-"开头
        assert frequencies.infer_freq(index) == gen.freqstr  # 断言推断出的频率与生成的频率字符串相等
    else:
        # 对于季度末的特定断言
        inf_freq = frequencies.infer_freq(index)
        is_dec_range = inf_freq == "QE-DEC" and gen.freqstr in (
            "QE",
            "QE-DEC",
            "QE-SEP",
            "QE-JUN",
            "QE-MAR",
        )
        is_nov_range = inf_freq == "QE-NOV" and gen.freqstr in (
            "QE-NOV",
            "QE-AUG",
            "QE-MAY",
            "QE-FEB",
        )
        is_oct_range = inf_freq == "QE-OCT" and gen.freqstr in (
            "QE-OCT",
            "QE-JUL",
            "QE-APR",
            "QE-JAN",
        )
        assert is_dec_range or is_nov_range or is_oct_range


# 测试函数：如果使用了PeriodIndex，则引发TypeError异常
def test_raise_if_period_index():
    index = period_range(start="1/1/1990", periods=20, freq="M")  # 创建PeriodIndex对象
    msg = "Check the `freq` attribute instead of using infer_freq"  # 错误信息

    with pytest.raises(TypeError, match=msg):  # 断言引发TypeError异常，并匹配错误信息
        frequencies.infer_freq(index)


# 测试函数：如果日期数量少于3，则引发ValueError异常
def test_raise_if_too_few():
    index = DatetimeIndex(["12/31/1998", "1/3/1999"])  # 创建DatetimeIndex对象
    msg = "Need at least 3 dates to infer frequency"  # 错误信息

    with pytest.raises(ValueError, match=msg):  # 断言引发ValueError异常，并匹配错误信息
        frequencies.infer_freq(index)


# 测试函数：检查工作日的推断频率是否正确
def test_business_daily():
    index = DatetimeIndex(["01/01/1999", "1/4/1999", "1/5/1999"])  # 创建DatetimeIndex对象
    assert frequencies.infer_freq(index) == "B"  # 断言推断出的频率为工作日


# 测试函数：检查是否不会错误地推断为工作日频率
def test_business_daily_look_alike():
    # see gh-16624
    #
    # Do not infer "B when "weekend" (2-day gap) in wrong place.
    # 创建一个 DatetimeIndex 对象，包含三个日期字符串作为索引： "12/31/1998", "1/3/1999", "1/4/1999"
    index = DatetimeIndex(["12/31/1998", "1/3/1999", "1/4/1999"])
    # 使用 frequencies 模块推断 index 对象的频率，如果无法推断则返回 None
    assert frequencies.infer_freq(index) is None
def test_day_corner():
    # 创建一个DatetimeIndex对象，包含三个日期字符串
    index = DatetimeIndex(["1/1/2000", "1/2/2000", "1/3/2000"])
    # 断言推断索引频率为日（"D"）
    assert frequencies.infer_freq(index) == "D"


def test_non_datetime_index():
    # 将日期字符串列表转换为日期时间索引
    dates = to_datetime(["1/1/2000", "1/2/2000", "1/3/2000"])
    # 断言推断索引频率为日（"D"）
    assert frequencies.infer_freq(dates) == "D"


def test_fifth_week_of_month_infer():
    # 见gh-9425
    #
    # 只尝试推断 WOM-4 的频率
    index = DatetimeIndex(["2014-03-31", "2014-06-30", "2015-03-30"])
    # 断言无法推断索引的频率
    assert frequencies.infer_freq(index) is None


def test_week_of_month_fake():
    # 所有这些日期都是同一周的同一天，相隔4或5周
    index = DatetimeIndex(["2013-08-27", "2013-10-01", "2013-10-29", "2013-11-26"])
    # 断言推断索引的频率不是 "WOM-4TUE"
    assert frequencies.infer_freq(index) != "WOM-4TUE"


def test_fifth_week_of_month():
    # 见gh-9425
    #
    # 仅支持到 WOM-4 的频率
    msg = (
        "Of the four parameters: start, end, periods, "
        "and freq, exactly three must be specified"
    )

    # 使用pytest的raises函数断言在创建日期范围时会引发值错误，并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        date_range("2014-01-01", freq="WOM-5MON")


def test_monthly_ambiguous():
    # 创建一个DatetimeIndex对象，包含三个日期字符串
    rng = DatetimeIndex(["1/31/2000", "2/29/2000", "3/31/2000"])
    # 断言推断索引频率为每月末尾（"ME"）
    assert rng.inferred_freq == "ME"


def test_annual_ambiguous():
    # 创建一个DatetimeIndex对象，包含三个日期字符串
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    # 断言推断索引频率为每年一月份（"YE-JAN"）
    assert rng.inferred_freq == "YE-JAN"


@pytest.mark.parametrize("count", range(1, 5))
def test_infer_freq_delta(base_delta_code_pair, count):
    # 获取当前时间戳对象
    b = Timestamp(datetime.now())
    base_delta, code = base_delta_code_pair

    # 创建一个DatetimeIndex对象，包含基础增量乘以计数后的日期列表
    inc = base_delta * count
    index = DatetimeIndex([b + inc * j for j in range(3)])

    # 期望的频率字符串，根据计数和代码
    exp_freq = f"{count:d}{code}" if count > 1 else code
    # 断言推断索引的频率与期望一致
    assert frequencies.infer_freq(index) == exp_freq


@pytest.mark.parametrize(
    "constructor",
    [
        lambda now, delta: DatetimeIndex(
            [now + delta * 7] + [now + delta * j for j in range(3)]
        ),
        lambda now, delta: DatetimeIndex(
            [now + delta * j for j in range(3)] + [now + delta * 7]
        ),
    ],
)
def test_infer_freq_custom(base_delta_code_pair, constructor):
    # 获取当前时间戳对象
    b = Timestamp(datetime.now())
    base_delta, _ = base_delta_code_pair

    # 使用参数化的构造函数创建DatetimeIndex对象
    index = constructor(b, base_delta)
    # 断言无法推断索引的频率
    assert frequencies.infer_freq(index) is None


@pytest.mark.parametrize(
    "expected,dates",
    list(
        {
            "YS-JAN": ["2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01"],
            "QE-OCT": ["2009-01-31", "2009-04-30", "2009-07-31", "2009-10-31"],
            "ME": ["2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"],
            "W-SAT": ["2010-12-25", "2011-01-01", "2011-01-08", "2011-01-15"],
            "D": ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"],
            "h": [
                "2011-12-31 22:00",
                "2011-12-31 23:00",
                "2012-01-01 00:00",
                "2012-01-01 01:00",
            ],
        }.items()
    ),
)
def test_infer_freq_tz(tz_naive_fixture, expected, dates, unit):
    # 参数化测试，不需要在此处添加注释
    pass
    # 使用特定的时区固定 fixture（见 GitHub issue #7310 和 GitHub issue #55609）
    tz = tz_naive_fixture
    # 使用给定的日期列表创建 DatetimeIndex 对象，并将其作为指定单位处理
    idx = DatetimeIndex(dates, tz=tz).as_unit(unit)
    # 断言检查索引的推断频率是否与预期相符
    assert idx.inferred_freq == expected
def test_infer_freq_tz_series(tz_naive_fixture):
    # infer_freq should work with both tz-naive and tz-aware series. See gh-52456
    # 使用 tz-naive 和 tz-aware 系列测试 infer_freq 方法的功能。参见 gh-52456
    tz = tz_naive_fixture
    # 创建一个时间索引，从 "2021-01-01" 到 "2021-01-04"，带有时区 tz
    idx = date_range("2021-01-01", "2021-01-04", tz=tz)
    # 将时间索引转换为系列，并重置索引
    series = idx.to_series().reset_index(drop=True)
    # 推断系列的频率
    inferred_freq = frequencies.infer_freq(series)
    # 断言推断的频率为 "D"（天）
    assert inferred_freq == "D"


@pytest.mark.parametrize(
    "date_pair",
    [
        ["2013-11-02", "2013-11-5"],   # Fall DST
        ["2014-03-08", "2014-03-11"],  # Spring DST
        ["2014-01-01", "2014-01-03"],  # Regular Time
    ],
)
@pytest.mark.parametrize(
    "freq",
    ["h", "3h", "10min", "3601s", "3600001ms", "3600000001us", "3600000000001ns"],
)
def test_infer_freq_tz_transition(tz_naive_fixture, date_pair, freq):
    # see gh-8772
    # 使用 tz-naive 系列和不同频率的日期对进行推断频率测试。参见 gh-8772
    tz = tz_naive_fixture
    # 创建时间索引，从 date_pair[0] 到 date_pair[1]，使用给定的频率 freq 和时区 tz
    idx = date_range(date_pair[0], date_pair[1], freq=freq, tz=tz)
    # 断言索引的推断频率与 freq 相同
    assert idx.inferred_freq == freq


def test_infer_freq_tz_transition_custom():
    # 测试自定义频率的时区转换情况
    index = date_range("2013-11-03", periods=5, freq="3h").tz_localize(
        "America/Chicago"
    )
    # 断言推断频率为 None
    assert index.inferred_freq is None


@pytest.mark.parametrize(
    "data,expected",
    [
        # 每天按小时频率，时间格式为 "h"
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
            ],
            "h",
        ),
        # 每个工作日按小时频率，时间格式为 "bh"
        (
            [
                "2014-07-01 09:00",
                "2014-07-01 10:00",
                "2014-07-01 11:00",
                "2014-07-01 12:00",
                "2014-07-01 13:00",
                "2014-07-01 14:00",
                "2014-07-01 15:00",
                "2014-07-01 16:00",
                "2014-07-02 09:00",
                "2014-07-02 10:00",
                "2014-07-02 11:00",
            ],
            "bh",
        ),
        # 每个工作日按小时频率，时间格式为 "bh"
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
            ],
            "bh",
        ),
        # 每个工作日按小时频率，时间格式为 "bh"
        (
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
                "2014-07-07 09:00",
                "2014-07-07 10:00",
                "2014-07-07 11:00",
                "2014-07-07 12:00",
                "2014-07-07 13:00",
                "2014-07-07 14:00",
                "2014-07-07 15:00",
                "2014-07-07 16:00",
                "2014-07-08 09:00",
                "2014-07-08 10:00",
                "2014-07-08 11:00",
                "2014-07-08 12:00",
                "2014-07-08 13:00",
                "2014-07-08 14:00",
                "2014-07-08 15:00",
                "2014-07-08 16:00",
            ],
            "bh",
        ),
    ],
def test_infer_freq_business_hour(data, expected):
    # 为了解决问题 gh-7905
    # 将数据转换为DatetimeIndex对象
    idx = DatetimeIndex(data)
    # 断言推断出的频率与预期值相等
    assert idx.inferred_freq == expected


def test_not_monotonic():
    # 创建一个DatetimeIndex对象，包含日期字符串
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    # 将索引倒序排列
    rng = rng[::-1]

    # 断言推断出的频率为 "-1YE-JAN"
    assert rng.inferred_freq == "-1YE-JAN"


def test_non_datetime_index2():
    # 创建一个DatetimeIndex对象，包含日期字符串
    rng = DatetimeIndex(["1/31/2000", "1/31/2001", "1/31/2002"])
    # 转换为Python的datetime对象数组
    vals = rng.to_pydatetime()

    # 推断日期序列的频率
    result = frequencies.infer_freq(vals)
    # 断言推断出的频率与原始索引的推断频率相等
    assert result == rng.inferred_freq


@pytest.mark.parametrize(
    "idx",
    [
        Index(np.arange(5), dtype=np.int64),
        Index(np.arange(5), dtype=np.float64),
        period_range("2020-01-01", periods=5),
        RangeIndex(5),
    ],
)
def test_invalid_index_types(idx):
    # 为了解决问题 gh-48439
    # 构造错误消息
    msg = "|".join(
        [
            "无法从不可转换的类型推断频率",
            "应检查`freq`属性，而不是使用infer_freq",
        ]
    )

    # 使用断言检查是否抛出TypeError异常，并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(idx)


@pytest.mark.skipif(is_platform_windows(), reason="see gh-10822: Windows issue")
def test_invalid_index_types_unicode():
    # 为了解决问题 gh-10822
    # 错误消息指出转换Unicode字符串到datetime时的异常情况
    msg = "Unknown datetime string format"

    # 使用断言检查是否抛出ValueError异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(Index(["ZqgszYBfuL"]))


def test_string_datetime_like_compat():
    # 为了解决问题 gh-6463
    # 数据包含日期格式的字符串
    data = ["2004-01", "2004-02", "2004-03", "2004-04"]

    # 推断数据的频率
    expected = frequencies.infer_freq(data)
    # 推断索引的频率
    result = frequencies.infer_freq(Index(data))

    # 断言推断出的频率与预期值相等
    assert result == expected


def test_series():
    # 为了解决问题 gh-6407
    # 创建一个包含日期范围的Series对象
    s = Series(date_range("20130101", "20130110"))
    # 推断Series对象的频率
    inferred = frequencies.infer_freq(s)
    # 断言推断出的频率为 "D"
    assert inferred == "D"


@pytest.mark.parametrize("end", [10, 10.0])
def test_series_invalid_type(end):
    # 为了解决问题 gh-6407
    # 构造错误消息
    msg = "无法从Series上的不可转换的dtype推断频率"
    # 创建一个Series对象
    s = Series(np.arange(end))

    # 使用断言检查是否抛出TypeError异常，并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


def test_series_inconvertible_string(using_infer_string):
    # 为了解决问题 gh-6407
    if using_infer_string:
        # 构造错误消息
        msg = "无法从"
        # 使用断言检查是否抛出TypeError异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))
    else:
        # 错误消息指出转换字符串到datetime时的异常情况
        msg = "Unknown datetime string format"
        # 使用断言检查是否抛出ValueError异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            frequencies.infer_freq(Series(["foo", "bar"]))


@pytest.mark.parametrize("freq", [None, "ms"])
def test_series_period_index(freq):
    # 为了解决问题 gh-6407
    # 构造错误消息
    msg = "无法从Series上的不可转换的dtype推断频率"
    # 创建一个PeriodIndex的Series对象
    s = Series(period_range("2013", periods=10, freq=freq))

    # 使用断言检查是否抛出TypeError异常，并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)


@pytest.mark.parametrize("freq", ["ME", "ms", "s"])
def test_series_datetime_index(freq):
    #```python
    # 为了解决问题 gh-6407
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # 创建一个时间序列（Series），起始日期为"20130101"，包含10个时间点，频率由变量 freq 决定
    s = Series(date_range("20130101", periods=10, freq=freq))
    
    # 推断时间序列 s 的频率
    inferred = frequencies.infer_freq(s)
    
    # 断言推断出的频率与变量 freq 相等，如果不相等则会引发 AssertionError
    assert inferred == freq
# 使用 pytest.mark.parametrize 装饰器为 test_legacy_offset_warnings 函数参数化测试用例，每次运行测试时将会依次使用不同的 offset_func 和 freq 组合
@pytest.mark.parametrize(
    "offset_func",
    [
        _get_offset,  # 使用 _get_offset 函数作为 offset_func 参数的一个值
        lambda freq: date_range("2011-01-01", periods=5, freq=freq),  # 使用 lambda 函数生成一个基于不同 freq 参数的日期范围
    ],
)
@pytest.mark.parametrize(
    "freq",
    [
        "WEEKDAY",  # 工作日频率
        "EOM",  # 月末频率
        "W@MON",  # 每周星期一
        "W@TUE",  # 每周星期二
        "W@WED",  # 每周星期三
        "W@THU",  # 每周星期四
        "W@FRI",  # 每周星期五
        "W@SAT",  # 每周星期六
        "W@SUN",  # 每周星期日
        "QE@JAN",  # 每季度一月
        "QE@FEB",  # 每季度二月
        "QE@MAR",  # 每季度三月
        "YE@JAN",  # 每年一月
        "YE@FEB",  # 每年二月
        "YE@MAR",  # 每年三月
        "YE@APR",  # 每年四月
        "YE@MAY",  # 每年五月
        "YE@JUN",  # 每年六月
        "YE@JUL",  # 每年七月
        "YE@AUG",  # 每年八月
        "YE@SEP",  # 每年九月
        "YE@OCT",  # 每年十月
        "YE@NOV",  # 每年十一月
        "YE@DEC",  # 每年十二月
        "WOM@1MON",  # 每月第一个星期一
        "WOM@2MON",  # 每月第二个星期一
        "WOM@3MON",  # 每月第三个星期一
        "WOM@4MON",  # 每月第四个星期一
        "WOM@1TUE",  # 每月第一个星期二
        "WOM@2TUE",  # 每月第二个星期二
        "WOM@3TUE",  # 每月第三个星期二
        "WOM@4TUE",  # 每月第四个星期二
        "WOM@1WED",  # 每月第一个星期三
        "WOM@2WED",  # 每月第二个星期三
        "WOM@3WED",  # 每月第三个星期三
        "WOM@4WED",  # 每月第四个星期三
        "WOM@1THU",  # 每月第一个星期四
        "WOM@2THU",  # 每月第二个星期四
        "WOM@3THU",  # 每月第三个星期四
        "WOM@4THU",  # 每月第四个星期四
        "WOM@1FRI",  # 每月第一个星期五
        "WOM@2FRI",  # 每月第二个星期五
        "WOM@3FRI",  # 每月第三个星期五
        "WOM@4FRI",  # 每月第四个星期五
    ],
)
def test_legacy_offset_warnings(offset_func, freq):
    # 在 pytest 中使用 pytest.raises 来检测是否抛出 ValueError 异常，并验证异常消息是否与 INVALID_FREQ_ERR_MSG 匹配
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        offset_func(freq)


# 测试 _get_offset 函数对 "ms" 和 "MS" 参数的返回值
def test_ms_vs_capital_ms():
    # 获取 "ms" 参数的返回值
    left = _get_offset("ms")
    # 获取 "MS" 参数的返回值
    right = _get_offset("MS")

    # 断言 "ms" 参数返回的结果应为 Milli 对象
    assert left == offsets.Milli()
    # 断言 "MS" 参数返回的结果应为 MonthBegin 对象
    assert right == offsets.MonthBegin()


# 测试 infer_freq 函数对非纳秒级别数组的推断频率
def test_infer_freq_non_nano():
    # 创建一个 numpy 数组，数据类型为 int64，然后转换为 "M8[s]" 类型
    arr = np.arange(10).astype(np.int64).view("M8[s]")
    # 使用 DatetimeArray._simple_new 方法创建一个 DatetimeArray 对象
    dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    # 调用 frequencies.infer_freq 函数推断频率
    res = frequencies.infer_freq(dta)
    # 断言推断的频率应为 "s"
    assert res == "s"

    # 将 arr 转换为 "m8[ms]" 类型
    arr2 = arr.view("m8[ms]")
    # 使用 TimedeltaArray._simple_new 方法创建一个 TimedeltaArray 对象
    tda = TimedeltaArray._simple_new(arr2, dtype=arr2.dtype)
    # 再次调用 frequencies.infer_freq 函数推断频率
    res2 = frequencies.infer_freq(tda)
    # 断言推断的频率应为 "ms"
    assert res2 == "ms"


# 测试 infer_freq 函数对非纳秒级别、带时区的时间序列的推断频率
def test_infer_freq_non_nano_tzaware(tz_aware_fixture):
    # 获取带时区信息的 fixture
    tz = tz_aware_fixture

    # 创建一个带时区信息的日期范围对象
    dti = date_range("2016-01-01", periods=365, freq="B", tz=tz)
    # 将日期范围对象转换为秒单位的时间数据
    dta = dti._data.as_unit("s")

    # 调用 frequencies.infer_freq 函数推断频率
    res = frequencies.infer_freq(dta)
    # 断言推断的频率应为 "B"
    assert res == "B"
```