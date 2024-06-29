# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_date_range.py`

```
"""
test date_range, bdate_range construction from the convenience range functions
"""

# 导入所需的库和模块
from datetime import (
    datetime,    # 导入datetime类
    time,        # 导入time类
    timedelta,   # 导入timedelta类
)
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest       # 导入pytest测试框架
import pytz         # 导入pytz时区库

from pandas._libs.tslibs import timezones  # 导入时间相关的时区模块
from pandas._libs.tslibs.offsets import (
    BDay,         # 导入工作日偏移量
    CDay,         # 导入自定义工作日偏移量
    DateOffset,   # 导入日期偏移量
    MonthEnd,     # 导入月末偏移量
    prefix_mapping,  # 导入偏移量前缀映射
)
from pandas.errors import OutOfBoundsDatetime  # 导入超出日期时间范围的异常类
import pandas.util._test_decorators as td  # 导入Pandas测试装饰器

import pandas as pd  # 导入Pandas库
from pandas import (
    DataFrame,         # 导入DataFrame类
    DatetimeIndex,     # 导入DatetimeIndex类
    Series,            # 导入Series类
    Timedelta,         # 导入Timedelta类
    Timestamp,         # 导入Timestamp类
    bdate_range,       # 导入工作日范围生成函数
    date_range,        # 导入日期范围生成函数
    offsets,           # 导入时间偏移量模块
)
import pandas._testing as tm  # 导入Pandas测试模块
from pandas.core.arrays.datetimes import _generate_range as generate_range  # 导入生成日期时间范围函数
from pandas.tests.indexes.datetimes.test_timezones import (
    FixedOffset,       # 导入固定时区偏移量类
    fixed_off_no_name, # 导入无名称的固定时区偏移量类
)

from pandas.tseries.holiday import USFederalHolidayCalendar  # 导入美国联邦假期日历

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)  # 设置起始和结束日期常量


def _get_expected_range(
    begin_to_match,
    end_to_match,
    both_range,
    inclusive_endpoints,
):
    """Helper to get expected range from a both inclusive range"""
    # 辅助函数：根据给定的两端范围和包含结束点的方式，获取预期范围

    left_match = begin_to_match == both_range[0]
    right_match = end_to_match == both_range[-1]

    if inclusive_endpoints == "left" and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == "right" and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == "neither" and left_match and right_match:
        expected_range = both_range[1:-1]
    elif inclusive_endpoints == "neither" and right_match:
        expected_range = both_range[:-1]
    elif inclusive_endpoints == "neither" and left_match:
        expected_range = both_range[1:]
    elif inclusive_endpoints == "both":
        expected_range = both_range[:]
    else:
        expected_range = both_range[:]

    return expected_range


class TestTimestampEquivDateRange:
    # Older tests in TestTimeSeries constructed their `stamp` objects
    # using `date_range` instead of the `Timestamp` constructor.
    # TestTimestampEquivDateRange checks that these are equivalent in the
    # pertinent cases.

    def test_date_range_timestamp_equiv(self):
        # 测试date_range和Timestamp生成的时间戳对象是否等价
        rng = date_range("20090415", "20090519", tz="US/Eastern")  # 生成日期范围对象，指定时区为美东时间
        stamp = rng[0]  # 获取日期范围的第一个时间戳对象

        ts = Timestamp("20090415", tz="US/Eastern")  # 创建指定时区的时间戳对象
        assert ts == stamp  # 断言两个时间戳对象相等

    def test_date_range_timestamp_equiv_dateutil(self):
        # 测试date_range和Timestamp生成的时间戳对象是否等价（使用dateutil时区）
        rng = date_range("20090415", "20090519", tz="dateutil/US/Eastern")  # 生成日期范围对象，使用dateutil库指定美东时间
        stamp = rng[0]  # 获取日期范围的第一个时间戳对象

        ts = Timestamp("20090415", tz="dateutil/US/Eastern")  # 创建指定dateutil时区的时间戳对象
        assert ts == stamp  # 断言两个时间戳对象相等

    def test_date_range_timestamp_equiv_explicit_pytz(self):
        # 测试date_range和Timestamp生成的时间戳对象是否等价（使用显式的pytz时区）
        pytz = pytest.importorskip("pytz")  # 导入并检查pytz库是否可用
        rng = date_range("20090415", "20090519", tz=pytz.timezone("US/Eastern"))  # 生成日期范围对象，使用pytz库指定美东时间
        stamp = rng[0]  # 获取日期范围的第一个时间戳对象

        ts = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))  # 创建指定pytz时区的时间戳对象
        assert ts == stamp  # 断言两个时间戳对象相等

    @td.skip_if_windows
    # 定义一个测试函数，验证日期范围和时间戳等效性，使用显式的 dateutil 函数获取时区信息
    def test_date_range_timestamp_equiv_explicit_dateutil(self):
        # 导入 dateutil_gettz 函数来获取时区信息
        from pandas._libs.tslibs.timezones import dateutil_gettz as gettz

        # 创建一个日期范围，从 "20090415" 到 "20090519"，使用 "US/Eastern" 时区
        rng = date_range("20090415", "20090519", tz=gettz("US/Eastern"))
        # 获取日期范围中的第一个时间戳
        stamp = rng[0]

        # 创建一个 Timestamp 对象，表示 "20090415"，使用 "US/Eastern" 时区
        ts = Timestamp("20090415", tz=gettz("US/Eastern"))
        # 断言 Timestamp 对象与日期范围的第一个时间戳相等
        assert ts == stamp

    # 定义一个测试函数，验证日期范围和时间戳等效性，从 datetime 实例创建 Timestamp 对象
    def test_date_range_timestamp_equiv_from_datetime_instance(self):
        # 创建一个 datetime 实例，表示日期 "2014-03-04"
        datetime_instance = datetime(2014, 3, 4)
        # 通过 date_range 创建一个频率为 "D" 的日期范围，取第一个时间戳
        timestamp_instance = date_range(datetime_instance, periods=1, freq="D")[0]

        # 创建一个 Timestamp 对象，使用给定的 datetime 实例
        ts = Timestamp(datetime_instance)
        # 断言 Timestamp 对象与从日期范围获取的时间戳实例相等
        assert ts == timestamp_instance

    # 定义一个测试函数，验证日期范围和时间戳等效性，保持频率不变
    def test_date_range_timestamp_equiv_preserve_frequency(self):
        # 通过 date_range 创建一个频率为 "D" 的日期范围，取第一个时间戳
        timestamp_instance = date_range("2014-03-05", periods=1, freq="D")[0]
        # 创建一个 Timestamp 对象，表示 "2014-03-05"
        ts = Timestamp("2014-03-05")

        # 断言从日期范围获取的时间戳实例与 Timestamp 对象相等
        assert timestamp_instance == ts
class TestDateRanges:
    # 测试日期范围的名称设置功能
    def test_date_range_name(self):
        # 调用 date_range 函数生成一个只包含一个日期的索引对象，指定了起始日期、周期数、频率和名称
        idx = date_range(start="2000-01-01", periods=1, freq="YE", name="TEST")
        # 断言索引对象的名称是否与预期相符
        assert idx.name == "TEST"

    # 测试日期范围中无效周期数的处理
    def test_date_range_invalid_periods(self):
        # 准备错误消息字符串
        msg = "periods must be an integer, got foo"
        # 使用 pytest 断言抛出 TypeError 异常，并检查异常消息是否符合预期
        with pytest.raises(TypeError, match=msg):
            date_range(start="1/1/2000", periods="foo", freq="D")

    # 测试日期范围中小数周期数的处理
    def test_date_range_fractional_period(self):
        # 准备错误消息字符串
        msg = "periods must be an integer"
        # 使用 pytest 断言抛出 TypeError 异常，并检查异常消息是否符合预期
        with pytest.raises(TypeError, match=msg):
            date_range("1/1/2000", periods=10.5)

    # 测试日期范围中无效频率字符串的处理
    @pytest.mark.parametrize("freq", ["2M", "1m", "2SM", "2BQ", "1bq", "2BY"])
    def test_date_range_frequency_M_SM_BQ_BY_raises(self, freq):
        # 准备错误消息字符串
        msg = f"Invalid frequency: {freq}"
        # 使用 pytest 断言抛出 ValueError 异常，并检查异常消息是否符合预期
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", periods=4, freq=freq)

    # 测试日期范围中元组频率的处理
    def test_date_range_tuple_freq_raises(self):
        # 准备错误消息字符串
        msg = "pass as a string instead"
        # 使用 pytest 断言抛出 TypeError 异常，并检查异常消息是否符合预期
        with pytest.raises(TypeError, match=msg):
            date_range(end=datetime(2000, 1, 1), freq=("D", 5), periods=20)

    # 测试日期范围中边界条件的处理
    @pytest.mark.parametrize("freq", ["ns", "us", "ms", "min", "s", "h", "D"])
    def test_date_range_edges(self, freq):
        # 创建时间增量对象和时间戳对象
        td = Timedelta(f"1{freq}")
        ts = Timestamp("1970-01-01")

        # 测试正常情况下日期范围的生成
        idx = date_range(
            start=ts + td,
            end=ts + 4 * td,
            freq=freq,
        )
        # 创建预期的日期时间索引对象
        exp = DatetimeIndex(
            [ts + n * td for n in range(1, 5)],
            dtype="M8[ns]",
            freq=freq,
        )
        # 使用 pandas 测试工具断言生成的索引对象是否与预期相符
        tm.assert_index_equal(idx, exp)

        # 测试起始日期晚于结束日期的情况
        idx = date_range(
            start=ts + 4 * td,
            end=ts + td,
            freq=freq,
        )
        # 创建预期的空日期时间索引对象
        exp = DatetimeIndex([], dtype="M8[ns]", freq=freq)
        # 使用 pandas 测试工具断言生成的索引对象是否为空
        tm.assert_index_equal(idx, exp)

        # 测试起始日期与结束日期相同的情况
        idx = date_range(
            start=ts + td,
            end=ts + td,
            freq=freq,
        )
        # 创建预期包含一个日期时间的索引对象
        exp = DatetimeIndex([ts + td], dtype="M8[ns]", freq=freq)
        # 使用 pandas 测试工具断言生成的索引对象是否与预期相符
        tm.assert_index_equal(idx, exp)

    # 测试日期范围中接近实现边界的处理
    def test_date_range_near_implementation_bound(self):
        # 准备时间增量对象
        freq = Timedelta(1)
        # 使用 pytest 断言抛出 OutOfBoundsDatetime 异常，并检查异常消息是否符合预期
        with pytest.raises(OutOfBoundsDatetime, match="Cannot generate range with"):
            date_range(end=Timestamp.min, periods=2, freq=freq)

    # 测试日期范围中 NaT 值的处理
    def test_date_range_nat(self):
        # 准备错误消息字符串
        msg = "Neither `start` nor `end` can be NaT"
        # 使用 pytest 断言抛出 ValueError 异常，并检查异常消息是否符合预期
        with pytest.raises(ValueError, match=msg):
            date_range(start="2016-01-01", end=pd.NaT, freq="D")
        with pytest.raises(ValueError, match=msg):
            date_range(start=pd.NaT, end="2016-01-01", freq="D")
    def test_date_range_multiplication_overflow(self):
        # GH#24255
        # 检查计算 `addend = periods * stride` 是否会导致溢出的情况
        with tm.assert_produces_warning(None):
            # 不应该看到溢出的 RuntimeWarning
            dti = date_range(start="1677-09-22", periods=213503, freq="D")

        assert dti[0] == Timestamp("1677-09-22")
        assert len(dti) == 213503

        msg = "Cannot generate range with"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 测试当 periods 很大时是否会抛出 OutOfBoundsDatetime 异常
            date_range("1969-05-04", periods=200000000, freq="30000D")

    def test_date_range_unsigned_overflow_handling(self):
        # GH#24255
        # 处理 `addend = periods * stride` 超出 int64 但未超出 uint64 的情况
        dti = date_range(start="1677-09-22", end="2262-04-11", freq="D")

        dti2 = date_range(start=dti[0], periods=len(dti), freq="D")
        assert dti2.equals(dti)

        dti3 = date_range(end=dti[-1], periods=len(dti), freq="D")
        assert dti3.equals(dti)

    def test_date_range_int64_overflow_non_recoverable(self):
        # GH#24255
        # 处理 start 晚于 1970-01-01，导致 int64 溢出但未导致 uint64 溢出的情况
        msg = "Cannot generate range with"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 测试当 periods 很大时是否会抛出 OutOfBoundsDatetime 异常
            date_range(start="1970-02-01", periods=106752 * 24, freq="h")

        # 处理 end 早于 1970-01-01，导致 int64 溢出但未导致 uint64 溢出的情况
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 测试当 periods 很大时是否会抛出 OutOfBoundsDatetime 异常
            date_range(end="1969-11-14", periods=106752 * 24, freq="h")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "s_ts, e_ts", [("2262-02-23", "1969-11-14"), ("1970-02-01", "1677-10-22")]
    )
    def test_date_range_int64_overflow_stride_endpoint_different_signs(
        self, s_ts, e_ts
    ):
        # 处理 stride * periods 溢出 int64，且 stride/endpoint 符号不同的情况
        start = Timestamp(s_ts)
        end = Timestamp(e_ts)

        expected = date_range(start=start, end=end, freq="-1h")
        assert expected[0] == start
        assert expected[-1] == end

        dti = date_range(end=end, periods=len(expected), freq="-1h")
        tm.assert_index_equal(dti, expected)

    def test_date_range_out_of_bounds(self):
        # GH#14187
        msg = "Cannot generate range"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 测试当 periods 很大时是否会抛出 OutOfBoundsDatetime 异常
            date_range("2016-01-01", periods=100000, freq="D")
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # 测试当 periods 很大时是否会抛出 OutOfBoundsDatetime 异常
            date_range(end="1763-10-12", periods=100000, freq="D")

    def test_date_range_gen_error(self):
        # 测试生成特定范围内的日期序列
        rng = date_range("1/1/2000 00:00", "1/1/2000 00:18", freq="5min")
        assert len(rng) == 4
    def test_date_range_normalize(self):
        # 获取当前日期时间
        snap = datetime.today()
        # 定义生成周期数
        n = 50

        # 调用date_range函数生成日期范围，每隔2天生成一个日期，不进行日期归一化
        rng = date_range(snap, periods=n, normalize=False, freq="2D")

        # 定义时间偏移量为2天
        offset = timedelta(2)
        # 生成期望的日期时间索引，从snap开始，每个索引增加offset的时间间隔
        expected = DatetimeIndex(
            [snap + i * offset for i in range(n)], dtype="M8[ns]", freq=offset
        )

        # 使用测试工具比较生成的日期范围和期望的日期范围是否相等
        tm.assert_index_equal(rng, expected)

        # 生成日期范围，从指定的时间开始，不进行日期归一化，按工作日频率生成
        rng = date_range("1/1/2000 08:15", periods=n, normalize=False, freq="B")
        # 定义预期的时间
        the_time = time(8, 15)
        # 遍历生成的日期范围，确保每个日期的时间部分与预期的时间相等
        for val in rng:
            assert val.time() == the_time

    def test_date_range_ambiguous_arguments(self):
        # #2538
        # 定义开始时间和结束时间
        start = datetime(2011, 1, 1, 5, 3, 40)
        end = datetime(2011, 1, 1, 8, 9, 40)

        # 定义错误消息，指明参数不明确的错误情况
        msg = (
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )
        # 使用pytest断言期望引发值错误，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            date_range(start, end, periods=10, freq="s")

    def test_date_range_convenience_periods(self, unit):
        # GH 20808
        # 测试日期范围生成的便利性，指定起始日期、结束日期、生成的周期数和单位
        result = date_range("2018-04-24", "2018-04-27", periods=3, unit=unit)
        # 定义期望的日期时间索引
        expected = DatetimeIndex(
            ["2018-04-24 00:00:00", "2018-04-25 12:00:00", "2018-04-27 00:00:00"],
            dtype=f"M8[{unit}]",
            freq=None,
        )

        # 使用测试工具比较生成的日期范围和期望的日期范围是否相等
        tm.assert_index_equal(result, expected)

        # 测试如果在生成的日期范围内时区变更为夏令时，确保间隔保持线性
        result = date_range(
            "2018-04-01 01:00:00",
            "2018-04-01 04:00:00",
            tz="Australia/Sydney",
            periods=3,
            unit=unit,
        )
        expected = DatetimeIndex(
            [
                Timestamp("2018-04-01 01:00:00+1100", tz="Australia/Sydney"),
                Timestamp("2018-04-01 02:00:00+1000", tz="Australia/Sydney"),
                Timestamp("2018-04-01 04:00:00+1000", tz="Australia/Sydney"),
            ]
        ).as_unit(unit)
        # 使用测试工具比较生成的日期范围和期望的日期范围是否相等
        tm.assert_index_equal(result, expected)

    def test_date_range_index_comparison(self):
        # 生成日期范围，指定时区为US/Eastern
        rng = date_range("2011-01-01", periods=3, tz="US/Eastern")
        # 将日期范围转换为DataFrame
        df = Series(rng).to_frame()
        # 将日期范围转换为数组，并转置
        arr = np.array([rng.to_list()]).T
        arr2 = np.array([rng]).T

        # 使用pytest断言期望引发值错误，指明无法将日期范围转换为Series
        with pytest.raises(ValueError, match="Unable to coerce to Series"):
            rng == df

        # 使用pytest断言期望引发值错误，指明无法将DataFrame转换为日期范围
        with pytest.raises(ValueError, match="Unable to coerce to Series"):
            df == rng

        # 定义期望的DataFrame，所有值为True
        expected = DataFrame([True, True, True])

        # 比较DataFrame和数组之间的相等性，并使用测试工具检查结果是否与期望一致
        results = df == arr2
        tm.assert_frame_equal(results, expected)

        # 定义期望的Series，所有值为True
        expected = Series([True, True, True], name=0)

        # 比较DataFrame的第一列和数组的第一列的相等性，并使用测试工具检查结果是否与期望一致
        results = df[0] == arr2[:, 0]
        tm.assert_series_equal(results, expected)

        # 定义期望的二维数组，对角线为True，其它位置为False
        expected = np.array(
            [[True, False, False], [False, True, False], [False, False, True]]
        )
        # 比较日期范围和数组之间的相等性，并使用测试工具检查结果是否与期望一致
        results = rng == arr
        tm.assert_numpy_array_equal(results, expected)
    @pytest.mark.parametrize(
        "start,end,result_tz",
        [  # 使用 pytest 的 parametrize 装饰器，为测试方法提供多组参数进行测试
            ["20180101", "20180103", "US/Eastern"],  # 测试日期范围字符串，指定时区为 US/Eastern
            [datetime(2018, 1, 1), datetime(2018, 1, 3), "US/Eastern"],  # 测试 datetime 对象日期范围，指定时区为 US/Eastern
            [Timestamp("20180101"), Timestamp("20180103"), "US/Eastern"],  # 测试 pandas Timestamp 对象日期范围，指定时区为 US/Eastern
            [
                Timestamp("20180101", tz="US/Eastern"),
                Timestamp("20180103", tz="US/Eastern"),
                "US/Eastern",
            ],  # 测试带时区的 pandas Timestamp 对象日期范围，指定时区为 US/Eastern
            [
                Timestamp("20180101", tz="US/Eastern"),
                Timestamp("20180103", tz="US/Eastern"),
                None,
            ],  # 测试带时区的 pandas Timestamp 对象日期范围，不指定时区
        ],
    )
    def test_date_range_linspacing_tz(self, start, end, result_tz):
        # GH 20983
        # 调用 date_range 函数计算指定时区下的日期范围
        result = date_range(start, end, periods=3, tz=result_tz)
        # 期望的日期范围，从 20180101 开始，每天一个时间步长，指定时区为 US/Eastern
        expected = date_range("20180101", periods=3, freq="D", tz="US/Eastern")
        # 使用 assert_index_equal 断言实际计算结果与期望结果一致
        tm.assert_index_equal(result, expected)

    def test_date_range_timedelta(self):
        start = "2020-01-01"
        end = "2020-01-11"
        # 使用不同的频率单位进行日期范围计算
        rng1 = date_range(start, end, freq="3D")  # 每 3 天一个时间步长
        rng2 = date_range(start, end, freq=timedelta(days=3))  # 使用 timedelta 指定每 3 天一个时间步长
        # 使用 assert_index_equal 断言两个日期范围对象相等
        tm.assert_index_equal(rng1, rng2)

    def test_range_misspecified(self):
        # GH #1095
        # 错误消息提示，要求在 start、end、periods 和 freq 中恰好指定三个参数
        msg = (
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证异常消息与预期相符
        with pytest.raises(ValueError, match=msg):
            date_range(start="1/1/2000")

        with pytest.raises(ValueError, match=msg):
            date_range(end="1/1/2000")

        with pytest.raises(ValueError, match=msg):
            date_range(periods=10)

        with pytest.raises(ValueError, match=msg):
            date_range(start="1/1/2000", freq="h")

        with pytest.raises(ValueError, match=msg):
            date_range(end="1/1/2000", freq="h")

        with pytest.raises(ValueError, match=msg):
            date_range(periods=10, freq="h")

        with pytest.raises(ValueError, match=msg):
            date_range()

    def test_compat_replace(self):
        # https://github.com/statsmodels/statsmodels/issues/3349
        # 测试 replace 方法接受整数或长整数作为参数
        result = date_range(Timestamp("1960-04-01 00:00:00"), periods=76, freq="QS-JAN")
        # 断言生成的日期范围长度为 76
        assert len(result) == 76

    def test_catch_infinite_loop(self):
        offset = offsets.DateOffset(minute=5)
        # 测试是否能捕获并报告无限循环的错误
        msg = "Offset <DateOffset: minute=5> did not increment date"
        with pytest.raises(ValueError, match=msg):
            date_range(datetime(2011, 11, 11), datetime(2011, 11, 12), freq=offset)
    def test_construct_over_dst(self, unit):
        # GH 20854
        # 创建具有模糊时间属性的预定夏令时时间戳
        pre_dst = Timestamp("2010-11-07 01:00:00").tz_localize(
            "US/Pacific", ambiguous=True
        )
        # 创建不含模糊时间属性的太平洋标准时间时间戳
        pst_dst = Timestamp("2010-11-07 01:00:00").tz_localize(
            "US/Pacific", ambiguous=False
        )
        # 期望的日期时间索引列表
        expect_data = [
            Timestamp("2010-11-07 00:00:00", tz="US/Pacific"),
            pre_dst,
            pst_dst,
        ]
        # 根据期望数据创建预期的日期时间索引对象
        expected = DatetimeIndex(expect_data, freq="h").as_unit(unit)
        # 调用函数进行日期范围生成，与预期结果进行比较
        result = date_range(
            start="2010-11-7", periods=3, freq="h", tz="US/Pacific", unit=unit
        )
        tm.assert_index_equal(result, expected)

    def test_construct_with_different_start_end_string_format(self, unit):
        # GH 12064
        # 使用不同的开始和结束时间字符串格式生成日期范围
        result = date_range(
            "2013-01-01 00:00:00+09:00",
            "2013/01/01 02:00:00+09:00",
            freq="h",
            unit=unit,
        )
        # 预期的日期时间索引对象
        expected = DatetimeIndex(
            [
                Timestamp("2013-01-01 00:00:00+09:00"),
                Timestamp("2013-01-01 01:00:00+09:00"),
                Timestamp("2013-01-01 02:00:00+09:00"),
            ],
            freq="h",
        ).as_unit(unit)
        # 比较生成的结果与预期结果
        tm.assert_index_equal(result, expected)

    def test_error_with_zero_monthends(self):
        # 检查使用零月末频率时是否会引发值错误异常
        msg = r"Offset <0 \* MonthEnds> did not increment date"
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", "1/1/2001", freq=MonthEnd(0))

    def test_range_bug(self, unit):
        # GH #770
        # 定义月份偏移量对象
        offset = DateOffset(months=3)
        # 生成日期范围，检查已知的 bug 问题
        result = date_range("2011-1-1", "2012-1-31", freq=offset, unit=unit)

        start = datetime(2011, 1, 1)
        # 生成预期的日期时间索引对象
        expected = DatetimeIndex(
            [start + i * offset for i in range(5)], dtype=f"M8[{unit}]", freq=offset
        )
        # 比较生成的结果与预期结果
        tm.assert_index_equal(result, expected)

    def test_range_tz_pytz(self):
        # 查看 GH #2906
        # 导入 pytest-pytz 并进行必要的检查
        pytz = pytest.importorskip("pytz")
        tz = pytz.timezone("US/Eastern")
        start = tz.localize(datetime(2011, 1, 1))
        end = tz.localize(datetime(2011, 1, 3))

        # 使用不同参数生成日期范围，并检查结果
        dr = date_range(start=start, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(end=end, periods=3)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

        dr = date_range(start=start, end=end)
        assert dr.tz.zone == tz.zone
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize(
        "start, end",
        [
            [datetime(2014, 3, 6), datetime(2014, 3, 12)],
            [datetime(2013, 11, 1), datetime(2013, 11, 6)],
        ],
    )
    def test_range_tz_dst_straddle_pytz(self, start, end):
        # 导入 pytest 模块，如果导入失败则跳过当前测试
        pytz = pytest.importorskip("pytz")
        # 设置时区为 "US/Eastern"
        tz = pytz.timezone("US/Eastern")
        # 创建带时区信息的起始和结束时间戳对象
        start = Timestamp(start, tz=tz)
        end = Timestamp(end, tz=tz)
        # 生成日期范围，频率为每日 ("D")
        dr = date_range(start, end, freq="D")
        # 断言日期范围的第一个值与起始时间戳相等
        assert dr[0] == start
        # 断言日期范围的最后一个值与结束时间戳相等
        assert dr[-1] == end
        # 断言日期范围内所有时间的小时部分为 0
        assert np.all(dr.hour == 0)

        # 在带时区信息的日期范围内生成日期范围，频率为每日 ("D")
        dr = date_range(start, end, freq="D", tz=tz)
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

        # 在去除时区信息的起始和结束时间戳对象生成日期范围，频率为每日 ("D")，时区为 "US/Eastern"
        dr = date_range(
            start.replace(tzinfo=None),
            end.replace(tzinfo=None),
            freq="D",
            tz=tz,
        )
        assert dr[0] == start
        assert dr[-1] == end
        assert np.all(dr.hour == 0)

    def test_range_tz_dateutil(self):
        # 查看 GitHub 上的 issue-2906

        # 使用 maybe_get_tz 函数修复 dateutil 下的时区文件名
        from pandas._libs.tslibs.timezones import maybe_get_tz

        tz = lambda x: maybe_get_tz("dateutil/" + x)

        # 创建带 dateutil 时区信息的起始和结束日期时间对象
        start = datetime(2011, 1, 1, tzinfo=tz("US/Eastern"))
        end = datetime(2011, 1, 3, tzinfo=tz("US/Eastern"))

        # 生成 3 个日期时间索引，从 start 开始
        dr = date_range(start=start, periods=3)
        # 断言日期范围的时区与 "US/Eastern" 相同
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

        # 生成 3 个日期时间索引，从 end 结束
        dr = date_range(end=end, periods=3)
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

        # 生成从 start 到 end 的日期范围
        dr = date_range(start=start, end=end)
        assert dr.tz == tz("US/Eastern")
        assert dr[0] == start
        assert dr[2] == end

    @pytest.mark.parametrize("freq", ["1D", "3D", "2ME", "7W", "3h", "YE"])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_range_closed(self, freq, tz, inclusive_endpoints_fixture):
        # GitHub issue GH#12409, GH#12684

        # 设置开始和结束时间戳，带有时区信息
        begin = Timestamp("2011/1/1", tz=tz)
        end = Timestamp("2014/1/1", tz=tz)

        # 生成闭区间的日期范围，指定频率和包含方式
        result_range = date_range(
            begin, end, inclusive=inclusive_endpoints_fixture, freq=freq
        )
        # 生成包含两端的日期范围
        both_range = date_range(begin, end, inclusive="both", freq=freq)
        # 获取预期的日期范围
        expected_range = _get_expected_range(
            begin, end, both_range, inclusive_endpoints_fixture
        )

        # 断言生成的日期范围与预期的日期范围相等
        tm.assert_index_equal(expected_range, result_range)

    @pytest.mark.parametrize("freq", ["1D", "3D", "2ME", "7W", "3h", "YE"])
    def test_range_with_tz_closed_with_tz_aware_start_end(
        self, freq, inclusive_endpoints_fixture
    ):
        begin = Timestamp("2011/1/1")
        end = Timestamp("2014/1/1")
        begintz = Timestamp("2011/1/1", tz="US/Eastern")
        endtz = Timestamp("2014/1/1", tz="US/Eastern")

        # 使用给定的起始日期、频率和时区创建一个日期范围，包括结束点，使用"US/Eastern"时区
        result_range = date_range(
            begin,
            end,
            inclusive=inclusive_endpoints_fixture,
            freq=freq,
            tz="US/Eastern",
        )
        # 使用给定的起始日期、频率和"both"选项创建一个日期范围，使用"US/Eastern"时区
        both_range = date_range(
            begin, end, inclusive="both", freq=freq, tz="US/Eastern"
        )
        # 获取预期的日期范围，包括不同的结束点设置和时区
        expected_range = _get_expected_range(
            begintz,
            endtz,
            both_range,
            inclusive_endpoints_fixture,
        )

        # 断言预期的日期范围和实际生成的日期范围相等
        tm.assert_index_equal(expected_range, result_range)

    def test_range_closed_boundary(self, inclusive_endpoints_fixture):
        # GH#11804
        # 创建右闭边界的日期范围，使用给定的频率和包含结束点的设置
        right_boundary = date_range(
            "2015-09-12",
            "2015-12-01",
            freq="QS-MAR",
            inclusive=inclusive_endpoints_fixture,
        )
        # 创建左闭边界的日期范围，使用给定的频率和包含结束点的设置
        left_boundary = date_range(
            "2015-09-01",
            "2015-09-12",
            freq="QS-MAR",
            inclusive=inclusive_endpoints_fixture,
        )
        # 创建两侧都闭合的日期范围，使用给定的频率和包含结束点的设置
        both_boundary = date_range(
            "2015-09-01",
            "2015-12-01",
            freq="QS-MAR",
            inclusive=inclusive_endpoints_fixture,
        )
        # 创建两侧都不闭合的日期范围，使用给定的频率和包含结束点的设置
        neither_boundary = date_range(
            "2015-09-11",
            "2015-09-12",
            freq="QS-MAR",
            inclusive=inclusive_endpoints_fixture,
        )

        # 设置预期的右闭边界日期范围为两侧都闭合的日期范围
        expected_right = both_boundary
        # 设置预期的左闭边界日期范围为两侧都闭合的日期范围
        expected_left = both_boundary
        # 设置预期的两侧都闭合的日期范围为两侧都闭合的日期范围
        expected_both = both_boundary

        # 根据包含结束点的设置调整预期的边界日期范围
        if inclusive_endpoints_fixture == "right":
            expected_left = both_boundary[1:]
        elif inclusive_endpoints_fixture == "left":
            expected_right = both_boundary[:-1]
        elif inclusive_endpoints_fixture == "both":
            expected_right = both_boundary[1:]
            expected_left = both_boundary[:-1]

        # 设置预期的两侧都不闭合的日期范围为两侧都闭合的日期范围的中间部分
        expected_neither = both_boundary[1:-1]

        # 断言各个边界日期范围与预期的日期范围相等
        tm.assert_index_equal(right_boundary, expected_right)
        tm.assert_index_equal(left_boundary, expected_left)
        tm.assert_index_equal(both_boundary, expected_both)
        tm.assert_index_equal(neither_boundary, expected_neither)
    def test_date_range_years_only(self, tz_naive_fixture):
        # 使用给定的时区fixture来初始化时区对象
        tz = tz_naive_fixture
        # 创建一个按月末频率的日期范围，从"2014"到"2015"年，包含时区信息
        rng1 = date_range("2014", "2015", freq="ME", tz=tz)
        # 期望的日期范围是从"2014-01-31"到"2014-12-31"，按月末频率，包含时区信息
        expected1 = date_range("2014-01-31", "2014-12-31", freq="ME", tz=tz)
        # 断言生成的日期范围和期望的日期范围相等
        tm.assert_index_equal(rng1, expected1)

        # 创建一个按月初频率的日期范围，从"2014"到"2015"年，包含时区信息
        rng2 = date_range("2014", "2015", freq="MS", tz=tz)
        # 期望的日期范围是从"2014-01-01"到"2015-01-01"，按月初频率，包含时区信息
        expected2 = date_range("2014-01-01", "2015-01-01", freq="MS", tz=tz)
        # 断言生成的日期范围和期望的日期范围相等
        tm.assert_index_equal(rng2, expected2)

        # 创建一个按年末频率的日期范围，从"2014"到"2020"年，包含时区信息
        rng3 = date_range("2014", "2020", freq="YE", tz=tz)
        # 期望的日期范围是从"2014-12-31"到"2019-12-31"，按年末频率，包含时区信息
        expected3 = date_range("2014-12-31", "2019-12-31", freq="YE", tz=tz)
        # 断言生成的日期范围和期望的日期范围相等
        tm.assert_index_equal(rng3, expected3)

        # 创建一个按年初频率的日期范围，从"2014"到"2020"年，包含时区信息
        rng4 = date_range("2014", "2020", freq="YS", tz=tz)
        # 期望的日期范围是从"2014-01-01"到"2020-01-01"，按年初频率，包含时区信息
        expected4 = date_range("2014-01-01", "2020-01-01", freq="YS", tz=tz)
        # 断言生成的日期范围和期望的日期范围相等
        tm.assert_index_equal(rng4, expected4)

    def test_freq_divides_end_in_nanos(self):
        # 创建一个时间范围，从"2005-01-12 10:00"到"2005-01-12 16:00"，频率为"345min"
        result_1 = date_range("2005-01-12 10:00", "2005-01-12 16:00", freq="345min")
        # 创建另一个时间范围，从"2005-01-13 10:00"到"2005-01-13 16:00"，频率为"345min"
        result_2 = date_range("2005-01-13 10:00", "2005-01-13 16:00", freq="345min")
        # 期望的时间索引列表，包括"2005-01-12 10:00:00"和"2005-01-12 15:45:00"
        expected_1 = DatetimeIndex(
            ["2005-01-12 10:00:00", "2005-01-12 15:45:00"],
            dtype="datetime64[ns]",
            freq="345min",
            tz=None,
        )
        # 期望的时间索引列表，包括"2005-01-13 10:00:00"和"2005-01-13 15:45:00"
        expected_2 = DatetimeIndex(
            ["2005-01-13 10:00:00", "2005-01-13 15:45:00"],
            dtype="datetime64[ns]",
            freq="345min",
            tz=None,
        )
        # 断言生成的时间索引和期望的时间索引相等
        tm.assert_index_equal(result_1, expected_1)
        tm.assert_index_equal(result_2, expected_2)

    def test_cached_range_bug(self):
        # 创建一个时间范围，从"2010-09-01 05:00:00"开始，持续50个时间点，每6小时一个时间点
        rng = date_range("2010-09-01 05:00:00", periods=50, freq=DateOffset(hours=6))
        # 断言时间范围长度为50
        assert len(rng) == 50
        # 断言时间范围的第一个时间点为"2010-09-01 05:00:00"
        assert rng[0] == datetime(2010, 9, 1, 5)

    def test_timezone_comparison_bug(self):
        # 创建一个带时区信息的时间戳对象，起始时间为"20130220 10:00"，时区为"US/Eastern"
        start = Timestamp("20130220 10:00", tz="US/Eastern")
        # 创建一个按照指定起始时间生成的时间范围，包含两个时间点，时区为"US/Eastern"
        result = date_range(start, periods=2, tz="US/Eastern")
        # 断言生成的时间范围长度为2
        assert len(result) == 2

    def test_timezone_comparison_assert(self):
        # 创建一个带时区信息的时间戳对象，起始时间为"20130220 10:00"，时区为"US/Eastern"
        start = Timestamp("20130220 10:00", tz="US/Eastern")
        # 期望引发断言错误，因为生成的时间范围时区不等于指定的时区"Europe/Berlin"
        msg = "Inferred time zone not equal to passed time zone"
        with pytest.raises(AssertionError, match=msg):
            date_range(start, periods=2, tz="Europe/Berlin")

    def test_negative_non_tick_frequency_descending_dates(self, tz_aware_fixture):
        # 使用给定的时区fixture来初始化时区对象
        tz = tz_aware_fixture
        # 创建一个时间范围，从"2011-06-01"到"2011-01-01"，频率为"-1MS"，带有时区信息
        result = date_range(start="2011-06-01", end="2011-01-01", freq="-1MS", tz=tz)
        # 期望的时间范围是从"2011-06-01"到"2011-01-01"，频率为"1MS"，带有时区信息，并倒序排列
        expected = date_range(end="2011-06-01", start="2011-01-01", freq="1MS", tz=tz)[
            ::-1
        ]
        # 断言生成的时间索引和期望的时间索引相等
        tm.assert_index_equal(result, expected)
    def test_range_where_start_equal_end(self, inclusive_endpoints_fixture):
        # 测试情况：起始日期等于结束日期的情况
        # GH 43394
        # 设置起始日期和结束日期为 "2021-09-02"
        start = "2021-09-02"
        end = "2021-09-02"
        # 调用 date_range 函数生成日期范围，根据 inclusive_endpoints_fixture 参数确定是否包含端点
        result = date_range(
            start=start, end=end, freq="D", inclusive=inclusive_endpoints_fixture
        )

        # 生成包含两个端点的日期范围
        both_range = date_range(start=start, end=end, freq="D", inclusive="both")
        # 根据 inclusive_endpoints_fixture 参数确定预期的结果
        if inclusive_endpoints_fixture == "neither":
            expected = both_range[1:-1]  # 排除两端点的结果
        elif inclusive_endpoints_fixture in ("left", "right", "both"):
            expected = both_range[:]  # 包含所有端点的结果

        # 使用 assert_index_equal 断言实际结果与预期结果相等
        tm.assert_index_equal(result, expected)

    def test_freq_dateoffset_with_relateivedelta_nanos(self):
        # 测试情况：使用带有纳秒单位的 DateOffset
        # GH 46877
        # 创建一个 DateOffset 对象，包含 10 小时，57 天和 3 纳秒
        freq = DateOffset(hours=10, days=57, nanoseconds=3)
        # 使用 date_range 函数生成日期范围，指定结束日期、生成的周期数、频率和名称
        result = date_range(end="1970-01-01 00:00:00", periods=10, freq=freq, name="a")
        # 设置预期的 DatetimeIndex 结果
        expected = DatetimeIndex(
            [
                "1968-08-02T05:59:59.999999973",
                "1968-09-28T15:59:59.999999976",
                "1968-11-25T01:59:59.999999979",
                "1969-01-21T11:59:59.999999982",
                "1969-03-19T21:59:59.999999985",
                "1969-05-16T07:59:59.999999988",
                "1969-07-12T17:59:59.999999991",
                "1969-09-08T03:59:59.999999994",
                "1969-11-04T13:59:59.999999997",
                "1970-01-01T00:00:00.000000000",
            ],
            name="a",
        )
        # 使用 assert_index_equal 断言实际结果与预期结果相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", ["2T", "2L", "1l", "1U", "2N", "2n"])
    def test_frequency_H_T_S_L_U_N_raises(self, freq):
        # 测试情况：传入无效的频率参数，期望抛出 ValueError 异常
        msg = f"Invalid frequency: {freq}"
        # 使用 pytest.raises 断言调用 date_range 函数时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", periods=2, freq=freq)

    @pytest.mark.parametrize(
        "freq_depr", ["m", "bm", "CBM", "SM", "BQ", "q-feb", "y-may", "Y-MAY"]
    )
    def test_frequency_raises(self, freq_depr):
        # 测试情况：传入已废弃的频率参数，期望抛出 ValueError 异常
        msg = f"Invalid frequency: {freq_depr}"
        # 使用 pytest.raises 断言调用 date_range 函数时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", periods=2, freq=freq_depr)

    def test_date_range_bday(self):
        # 测试情况：使用工作日频率生成日期范围
        sdate = datetime(1999, 12, 25)
        # 使用 date_range 函数生成工作日频率的日期范围，指定起始日期、生成的周期数
        idx = date_range(start=sdate, freq="1B", periods=20)
        # 使用断言检查生成的日期范围长度是否为 20
        assert len(idx) == 20
        # 使用断言检查生成的日期范围的第一个日期是否符合预期（起始日期加 0 个工作日）
        assert idx[0] == sdate + 0 * offsets.BDay()
        # 使用断言检查生成的日期范围的频率是否为工作日频率 "B"
        assert idx.freq == "B"

    @pytest.mark.parametrize("freq", ["200A", "2A-MAY"])
    def test_frequency_A_raises(self, freq):
        # 测试情况：传入无效的年度频率参数，期望抛出 ValueError 异常
        freq_msg = re.split("[0-9]*", freq, maxsplit=1)[1]
        msg = f"Invalid frequency: {freq_msg}"
        # 使用 pytest.raises 断言调用 date_range 函数时抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            date_range("1/1/2000", periods=2, freq=freq)
class TestDateRangeTZ:
    """Tests for date_range with timezones"""

    def test_hongkong_tz_convert(self):
        # GH#1673 smoke test
        # 创建一个日期范围对象，测试香港时区的转换
        dr = date_range("2012-01-01", "2012-01-10", freq="D", tz="Hongkong")

        # 验证对象是否正常工作
        dr.hour

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_span_dst_transition(self, tzstr):
        # GH#1778

        # 从标准时间到夏令时的转换
        # 创建一个日期范围对象，跨越夏令时转换期间的星期五
        dr = date_range("03/06/2012 00:00", periods=200, freq="W-FRI", tz="US/Eastern")

        assert (dr.hour == 0).all()

        # 创建一个日期范围对象，测试指定的时区字符串参数
        dr = date_range("2012-11-02", periods=10, tz=tzstr)
        result = dr.hour
        expected = pd.Index([0] * 10, dtype="int32")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_timezone_str_argument(self, tzstr):
        # 使用给定的时区字符串获取时区对象
        tz = timezones.maybe_get_tz(tzstr)
        # 创建一个日期范围对象，与预期的时区对象进行比较
        result = date_range("1/1/2000", periods=10, tz=tzstr)
        expected = date_range("1/1/2000", periods=10, tz=tz)

        tm.assert_index_equal(result, expected)

    def test_date_range_with_fixed_tz(self):
        # 创建一个具有固定时区偏移的日期范围对象
        off = FixedOffset(420, "+07:00")
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz

        # 使用固定时区偏移创建另一个日期范围对象，并进行索引比较
        rng2 = date_range(start, periods=len(rng), tz=off)
        tm.assert_index_equal(rng, rng2)

        # 创建一个日期范围对象，直接使用包含时区信息的字符串
        rng3 = date_range("3/11/2012 05:00:00+07:00", "6/11/2012 05:00:00+07:00")
        assert (rng.values == rng3.values).all()

    def test_date_range_with_fixedoffset_noname(self):
        # 使用无名称的固定偏移量创建日期范围对象
        off = fixed_off_no_name
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz

        idx = pd.Index([start, end])
        assert off == idx.tz

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_date_range_with_tz(self, tzstr):
        # 使用指定的时区字符串创建时间戳对象
        stamp = Timestamp("3/11/2012 05:00", tz=tzstr)
        assert stamp.hour == 5

        # 创建一个包含时区信息的日期范围对象，并验证时间戳匹配
        rng = date_range("3/11/2012 04:00", periods=10, freq="h", tz=tzstr)

        assert stamp == rng[1]

    @pytest.mark.parametrize("tz", ["Europe/London", "dateutil/Europe/London"])
    def test_date_range_ambiguous_endpoint(self, tz):
        # 使用一个模糊的终点创建日期范围对象
        # GH#11626

        # 断言在构造时出现模糊时间错误
        with pytest.raises(pytz.AmbiguousTimeError, match="Cannot infer dst time"):
            date_range(
                "2013-10-26 23:00", "2013-10-27 01:00", tz="Europe/London", freq="h"
            )

        # 创建一个包含时区信息的日期范围对象，验证时间戳匹配预期结果
        times = date_range(
            "2013-10-26 23:00", "2013-10-27 01:00", freq="h", tz=tz, ambiguous="infer"
        )
        assert times[0] == Timestamp("2013-10-26 23:00", tz=tz)
        assert times[-1] == Timestamp("2013-10-27 01:00:00+0000", tz=tz)
    @pytest.mark.parametrize(
        "tz, option, expected",
        [
            ["US/Pacific", "shift_forward", "2019-03-10 03:00"],
            ["dateutil/US/Pacific", "shift_forward", "2019-03-10 03:00"],
            ["US/Pacific", "shift_backward", "2019-03-10 01:00"],
            ["dateutil/US/Pacific", "shift_backward", "2019-03-10 01:00"],
            ["US/Pacific", timedelta(hours=1), "2019-03-10 03:00"],
        ],
    )
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化，对时区、选项和预期结果进行参数化测试
    def test_date_range_nonexistent_endpoint(self, tz, option, expected):
        # 测试日期范围函数在处理不存在的时间点时的行为

        # 使用 pytest 的断言来验证函数在抛出特定异常时的行为
        with pytest.raises(pytz.NonExistentTimeError, match="2019-03-10 02:00:00"):
            date_range(
                "2019-03-10 00:00", "2019-03-10 02:00", tz="US/Pacific", freq="h"
            )

        # 调用 date_range 函数生成指定时区下的时间序列
        times = date_range(
            "2019-03-10 00:00", "2019-03-10 02:00", freq="h", tz=tz, nonexistent=option
        )
        # 使用断言验证生成的时间序列最后一个时间是否符合预期
        assert times[-1] == Timestamp(expected, tz=tz)
class TestGenRangeGeneration:
    @pytest.mark.parametrize(
        "freqstr,offset",
        [
            ("B", BDay()),  # 参数化测试，freqstr为'B'，offset为工作日的偏移量对象
            ("C", CDay()),  # 参数化测试，freqstr为'C'，offset为自定义日的偏移量对象
        ],
    )
    def test_generate(self, freqstr, offset):
        # 调用generate_range函数生成日期范围rng1和rng2，确保两者相等
        rng1 = list(generate_range(START, END, periods=None, offset=offset, unit="ns"))
        rng2 = list(generate_range(START, END, periods=None, offset=freqstr, unit="ns"))
        assert rng1 == rng2

    def test_1(self):
        # 测试生成从指定日期开始的日期范围，预期结果是包含指定日期的两天日期列表
        rng = list(
            generate_range(
                start=datetime(2009, 3, 25),
                end=None,
                periods=2,
                offset=BDay(),
                unit="ns",
            )
        )
        expected = [datetime(2009, 3, 25), datetime(2009, 3, 26)]
        assert rng == expected

    def test_2(self):
        # 测试生成从开始日期到结束日期的日期范围，预期结果是包含指定日期的三天日期列表
        rng = list(
            generate_range(
                start=datetime(2008, 1, 1),
                end=datetime(2008, 1, 3),
                periods=None,
                offset=BDay(),
                unit="ns",
            )
        )
        expected = [datetime(2008, 1, 1), datetime(2008, 1, 2), datetime(2008, 1, 3)]
        assert rng == expected

    def test_3(self):
        # 测试生成从开始日期到结束日期的日期范围，预期结果是一个空列表，因为开始日期和结束日期相同
        rng = list(
            generate_range(
                start=datetime(2008, 1, 5),
                end=datetime(2008, 1, 6),
                periods=None,
                offset=BDay(),
                unit="ns",
            )
        )
        expected = []
        assert rng == expected

    def test_precision_finer_than_offset(self):
        # GH#9907 测试特定情况下的日期范围生成，验证生成的日期索引与预期的索引是否相等
        result1 = date_range(
            start="2015-04-15 00:00:03", end="2016-04-22 00:00:00", freq="QE"
        )
        result2 = date_range(
            start="2015-04-15 00:00:03", end="2015-06-22 00:00:04", freq="W"
        )
        expected1_list = [
            "2015-06-30 00:00:03",
            "2015-09-30 00:00:03",
            "2015-12-31 00:00:03",
            "2016-03-31 00:00:03",
        ]
        expected2_list = [
            "2015-04-19 00:00:03",
            "2015-04-26 00:00:03",
            "2015-05-03 00:00:03",
            "2015-05-10 00:00:03",
            "2015-05-17 00:00:03",
            "2015-05-24 00:00:03",
            "2015-05-31 00:00:03",
            "2015-06-07 00:00:03",
            "2015-06-14 00:00:03",
            "2015-06-21 00:00:03",
        ]
        expected1 = DatetimeIndex(
            expected1_list, dtype="datetime64[ns]", freq="QE-DEC", tz=None
        )
        expected2 = DatetimeIndex(
            expected2_list, dtype="datetime64[ns]", freq="W-SUN", tz=None
        )
        tm.assert_index_equal(result1, expected1)  # 验证生成的日期范围索引与预期的索引是否相等
        tm.assert_index_equal(result2, expected2)  # 验证生成的日期范围索引与预期的索引是否相等

    dt1, dt2 = "2017-01-01", "2017-01-01"
    tz1, tz2 = "US/Eastern", "Europe/London"
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用于多组参数化测试
        "start,end",  # 参数化测试的参数名
        [  # 参数化测试的参数列表
            (Timestamp(dt1, tz=tz1), Timestamp(dt2)),  # 第一组参数：起始时间和结束时间（结束时间无时区）
            (Timestamp(dt1), Timestamp(dt2, tz=tz2)),  # 第二组参数：起始时间（无时区）和结束时间（有时区）
            (Timestamp(dt1, tz=tz1), Timestamp(dt2, tz=tz2)),  # 第三组参数：起始时间和结束时间均有时区
            (Timestamp(dt1, tz=tz2), Timestamp(dt2, tz=tz1)),  # 第四组参数：起始时间和结束时间时区不同
        ],
    )
    def test_mismatching_tz_raises_err(self, start, end):
        # issue 18488  # 用例编号或问题描述
        msg = "Start and end cannot both be tz-aware with different timezones"  # 异常信息
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的断言检查是否抛出指定异常和匹配的消息
            date_range(start, end)  # 调用被测试的函数或方法，并期待抛出异常
        with pytest.raises(TypeError, match=msg):  # 再次调用，以确保所有情况都被覆盖
            date_range(start, end, freq=BDay())  # 传入额外参数进行测试
class TestBusinessDateRange:
    # 测试BusinessDateRange类的各种功能

    def test_constructor(self):
        # 测试构造函数的不同参数组合
        bdate_range(START, END, freq=BDay())
        bdate_range(START, periods=20, freq=BDay())
        bdate_range(end=START, periods=20, freq=BDay())

        msg = "periods must be an integer, got B"
        # 检查传入非整数类型时是否抛出TypeError异常
        with pytest.raises(TypeError, match=msg):
            date_range("2011-1-1", "2012-1-1", "B")

        with pytest.raises(TypeError, match=msg):
            bdate_range("2011-1-1", "2012-1-1", "B")

        msg = "freq must be specified for bdate_range; use date_range instead"
        # 检查未指定freq参数时是否抛出TypeError异常
        with pytest.raises(TypeError, match=msg):
            bdate_range(START, END, periods=10, freq=None)

    def test_misc(self):
        # 测试一些杂项功能
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20)
        firstDate = end - 19 * BDay()

        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_date_parse_failure(self):
        # 测试日期解析失败的情况
        badly_formed_date = "2007/100/1"

        msg = "Unknown datetime string format, unable to parse: 2007/100/1"
        # 检查解析失败时是否抛出ValueError异常
        with pytest.raises(ValueError, match=msg):
            Timestamp(badly_formed_date)

        with pytest.raises(ValueError, match=msg):
            bdate_range(start=badly_formed_date, periods=10)

        with pytest.raises(ValueError, match=msg):
            bdate_range(end=badly_formed_date, periods=10)

        with pytest.raises(ValueError, match=msg):
            bdate_range(badly_formed_date, badly_formed_date)

    def test_daterange_bug_456(self):
        # 测试日期范围中的bug修复情况
        # GH #456
        rng1 = bdate_range("12/5/2011", "12/5/2011")
        rng2 = bdate_range("12/2/2011", "12/5/2011")
        assert rng2._data.freq == BDay()

        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    def test_bdays_and_open_boundaries(self, inclusive_endpoints_fixture):
        # 测试工作日和开放边界情况
        # GH 6673
        start = "2018-07-21"  # Saturday
        end = "2018-07-29"  # Sunday
        result = date_range(start, end, freq="B", inclusive=inclusive_endpoints_fixture)

        bday_start = "2018-07-23"  # Monday
        bday_end = "2018-07-27"  # Friday
        expected = date_range(bday_start, bday_end, freq="D")
        tm.assert_index_equal(result, expected)
        # 注意：我们不希望这里的频率匹配

    def test_bday_near_overflow(self):
        # 测试接近溢出的工作日情况
        # GH#24252 avoid doing unnecessary addition that _would_ overflow
        start = Timestamp.max.floor("D").to_pydatetime()
        rng = date_range(start, end=None, periods=1, freq="B")
        expected = DatetimeIndex([start], freq="B").as_unit("ns")
        tm.assert_index_equal(rng, expected)

    def test_bday_overflow_error(self):
        # 测试工作日溢出错误情况
        # GH#24252 check that we get OutOfBoundsDatetime and not OverflowError
        msg = "Out of bounds nanosecond timestamp"
        start = Timestamp.max.floor("D").to_pydatetime()
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            date_range(start, periods=2, freq="B")
class TestCustomDateRange:
    # 测试自定义日期范围的类

    def test_constructor(self):
        # 测试构造函数
        bdate_range(START, END, freq=CDay())
        # 调用bdate_range函数，传入起始日期START，结束日期END，以及频率CDay()

        bdate_range(START, periods=20, freq=CDay())
        # 再次调用bdate_range函数，传入起始日期START，期数为20，频率CDay()

        bdate_range(end=START, periods=20, freq=CDay())
        # 第三次调用bdate_range函数，传入结束日期START，期数为20，频率CDay()

        msg = "periods must be an integer, got C"
        # 定义错误消息字符串

        with pytest.raises(TypeError, match=msg):
            # 使用pytest断言捕获TypeError异常，并验证异常消息符合msg
            date_range("2011-1-1", "2012-1-1", "C")
            # 调用date_range函数，传入"2011-1-1"和"2012-1-1"作为起始和结束日期，"C"作为期数参数

        with pytest.raises(TypeError, match=msg):
            # 再次使用pytest断言捕获TypeError异常，并验证异常消息符合msg
            bdate_range("2011-1-1", "2012-1-1", "C")
            # 再次调用bdate_range函数，传入"2011-1-1"和"2012-1-1"作为起始和结束日期，"C"作为期数参数

    def test_misc(self):
        # 测试其他功能
        end = datetime(2009, 5, 13)
        # 定义日期时间对象end，值为2009年5月13日

        dr = bdate_range(end=end, periods=20, freq="C")
        # 调用bdate_range函数，传入结束日期end，期数为20，频率为"C"，并将结果赋给dr

        firstDate = end - 19 * CDay()
        # 计算第一个日期，即end减去19个CDay的时间段

        assert len(dr) == 20
        # 使用断言验证dr的长度为20
        assert dr[0] == firstDate
        # 使用断言验证dr的第一个元素等于firstDate
        assert dr[-1] == end
        # 使用断言验证dr的最后一个元素等于end

    def test_daterange_bug_456(self):
        # 测试日期范围bug #456
        rng1 = bdate_range("12/5/2011", "12/5/2011", freq="C")
        # 调用bdate_range函数，传入起始日期"12/5/2011"和结束日期"12/5/2011"，频率为"C"，并将结果赋给rng1

        rng2 = bdate_range("12/2/2011", "12/5/2011", freq="C")
        # 再次调用bdate_range函数，传入起始日期"12/2/2011"和结束日期"12/5/2011"，频率为"C"，并将结果赋给rng2

        assert rng2._data.freq == CDay()
        # 使用断言验证rng2的数据频率等于CDay()

        result = rng1.union(rng2)
        # 调用rng1的union方法，传入rng2作为参数，将结果赋给result

        assert isinstance(result, DatetimeIndex)
        # 使用断言验证result是DatetimeIndex类型的对象

    def test_cdaterange(self, unit):
        # 测试定制日期范围
        result = bdate_range("2013-05-01", periods=3, freq="C", unit=unit)
        # 调用bdate_range函数，传入起始日期"2013-05-01"，期数为3，频率为"C"，单元为unit，并将结果赋给result

        expected = DatetimeIndex(
            ["2013-05-01", "2013-05-02", "2013-05-03"], dtype=f"M8[{unit}]", freq="C"
        )
        # 创建预期的DatetimeIndex对象expected，包含日期列表，dtype以及频率信息

        tm.assert_index_equal(result, expected)
        # 使用tm模块的断言函数验证result和expected相等
        assert result.freq == expected.freq
        # 使用断言验证result的频率等于expected的频率

    def test_cdaterange_weekmask(self, unit):
        # 测试带有周掩码的定制日期范围
        result = bdate_range(
            "2013-05-01", periods=3, freq="C", weekmask="Sun Mon Tue Wed Thu", unit=unit
        )
        # 调用bdate_range函数，传入起始日期"2013-05-01"，期数为3，频率为"C"，周掩码为"Sun Mon Tue Wed Thu"，单元为unit，并将结果赋给result

        expected = DatetimeIndex(
            ["2013-05-01", "2013-05-02", "2013-05-05"],
            dtype=f"M8[{unit}]",
            freq=result.freq,
        )
        # 创建预期的DatetimeIndex对象expected，包含日期列表，dtype以及频率信息

        tm.assert_index_equal(result, expected)
        # 使用tm模块的断言函数验证result和expected相等
        assert result.freq == expected.freq
        # 使用断言验证result的频率等于expected的频率

        # raise with non-custom freq
        # 使用非定制频率引发异常
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        # 定义错误消息字符串

        with pytest.raises(ValueError, match=msg):
            # 使用pytest断言捕获ValueError异常，并验证异常消息符合msg
            bdate_range("2013-05-01", periods=3, weekmask="Sun Mon Tue Wed Thu")

    def test_cdaterange_holidays(self, unit):
        # 测试带有节假日的定制日期范围
        result = bdate_range(
            "2013-05-01", periods=3, freq="C", holidays=["2013-05-01"], unit=unit
        )
        # 调用bdate_range函数，传入起始日期"2013-05-01"，期数为3，频率为"C"，节假日为["2013-05-01"]，单元为unit，并将结果赋给result

        expected = DatetimeIndex(
            ["2013-05-02", "2013-05-03", "2013-05-06"],
            dtype=f"M8[{unit}]",
            freq=result.freq,
        )
        # 创建预期的DatetimeIndex对象expected，包含日期列表，dtype以及频率信息

        tm.assert_index_equal(result, expected)
        # 使用tm模块的断言函数验证result和expected相等
        assert result.freq == expected.freq
        # 使用断言验证result的频率等于expected的频率

        # raise with non-custom freq
        # 使用非定制频率引发异常
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        # 定义错误消息字符串

        with pytest.raises(ValueError, match=msg):
            # 使用pytest断言捕获ValueError异常，并验证异常消息符合msg
            bdate_range("2013-05-01", periods=3, holidays=["2013-05-01"])
    # 测试自定义工作日范围生成函数，带有自定义频率、工作日掩码和假期列表
    def test_cdaterange_weekmask_and_holidays(self, unit):
        # 调用bdate_range函数生成自定义工作日范围
        result = bdate_range(
            "2013-05-01",
            periods=3,
            freq="C",  # 使用自定义频率'C'生成工作日范围
            weekmask="Sun Mon Tue Wed Thu",  # 指定工作日掩码
            holidays=["2013-05-01"],  # 指定假期列表
            unit=unit,
        )
        # 生成预期的日期时间索引
        expected = DatetimeIndex(
            ["2013-05-02", "2013-05-05", "2013-05-06"],
            dtype=f"M8[{unit}]",  # 设置索引的数据类型
            freq=result.freq,  # 设置索引的频率与生成结果一致
        )
        # 使用测试工具比较生成结果和预期结果的索引是否相等
        tm.assert_index_equal(result, expected)
        # 断言生成结果的频率与预期结果的频率相等
        assert result.freq == expected.freq

    # 测试在不指定自定义频率的情况下，工作日掩码和假期列表的有效性
    def test_cdaterange_holidays_weekmask_requires_freqstr(self):
        # 期望引发值错误，因为在传递工作日掩码和假期时需要指定自定义频率字符串
        msg = (
            "a custom frequency string is required when holidays or "
            "weekmask are passed, got frequency B"
        )
        with pytest.raises(ValueError, match=msg):
            bdate_range(
                "2013-05-01",
                periods=3,
                weekmask="Sun Mon Tue Wed Thu",  # 指定工作日掩码
                holidays=["2013-05-01"],  # 指定假期列表
            )

    # 使用pytest的参数化功能，测试所有自定义频率的情况
    @pytest.mark.parametrize(
        "freq", [freq for freq in prefix_mapping if freq.startswith("C")]
    )
    def test_all_custom_freq(self, freq):
        # 测试在指定工作日掩码和假期列表的情况下不会引发错误
        bdate_range(
            START, END, freq=freq, weekmask="Mon Wed Fri", holidays=["2009-03-14"]
        )

        # 测试在指定无效频率字符串时会引发值错误
        bad_freq = freq + "FOO"
        msg = f"invalid custom frequency string: {bad_freq}"
        with pytest.raises(ValueError, match=msg):
            bdate_range(START, END, freq=bad_freq)

    # 使用pytest的参数化功能，测试带毫秒精度的时间范围
    @pytest.mark.parametrize(
        "start_end",
        [
            ("2018-01-01T00:00:01.000Z", "2018-01-03T00:00:01.000Z"),
            ("2018-01-01T00:00:00.010Z", "2018-01-03T00:00:00.010Z"),
            ("2001-01-01T00:00:00.010Z", "2001-01-03T00:00:00.010Z"),
        ],
    )
    def test_range_with_millisecond_resolution(self, start_end):
        # GH49441: 测试带毫秒精度时间范围的问题
        start, end = start_end
        # 生成带有指定毫秒精度的日期范围
        result = date_range(start=start, end=end, periods=2, inclusive="left")
        # 生成预期的日期时间索引
        expected = DatetimeIndex([start], dtype="M8[ns, UTC]")  # 设置预期的数据类型
        # 使用测试工具比较生成结果和预期结果的索引是否相等
        tm.assert_index_equal(result, expected)

    # 使用pytest的参数化功能，测试带时区和自定义工作日的时间范围
    @pytest.mark.parametrize(
        "start,period,expected",
        [
            ("2022-07-23 00:00:00+02:00", 1, ["2022-07-25 00:00:00+02:00"]),
            ("2022-07-22 00:00:00+02:00", 1, ["2022-07-22 00:00:00+02:00"]),
            (
                "2022-07-22 00:00:00+02:00",
                2,
                ["2022-07-22 00:00:00+02:00", "2022-07-25 00:00:00+02:00"],
            ),
        ],
    )
    def test_range_with_timezone_and_custombusinessday(self, start, period, expected):
        # GH49441: 测试带时区和自定义工作日频率的时间范围
        result = date_range(start=start, periods=period, freq="C")  # 生成自定义工作日范围
        expected = DatetimeIndex(expected).as_unit("ns")  # 设置预期结果的单位为纳秒
        # 使用测试工具比较生成结果和预期结果的索引是否相等
        tm.assert_index_equal(result, expected)
class TestDateRangeNonNano:
    def test_date_range_reso_validation(self):
        msg = "'unit' must be one of 's', 'ms', 'us', 'ns'"
        # 断言当指定的 unit 不在合法范围内时会抛出 ValueError 异常，异常信息为 msg
        with pytest.raises(ValueError, match=msg):
            date_range("2016-01-01", "2016-03-04", periods=3, unit="h")

    def test_date_range_freq_higher_than_reso(self):
        # 当 freq 高于 reso 时会出现问题
        msg = "Use a lower freq or a higher unit instead"
        with pytest.raises(ValueError, match=msg):
            # 使用频率 "ns" 但单位却是 "ms"，会抛出 ValueError 异常
            date_range("2016-01-01", "2016-01-02", freq="ns", unit="ms")

    def test_date_range_freq_matches_reso(self):
        # GH#49106 匹配 reso 是可以的
        # 创建毫秒级别频率的日期范围，并进行索引比较
        dti = date_range("2016-01-01", "2016-01-01 00:00:01", freq="ms", unit="ms")
        rng = np.arange(1_451_606_400_000, 1_451_606_401_001, dtype=np.int64)
        expected = DatetimeIndex(rng.view("M8[ms]"), freq="ms")
        tm.assert_index_equal(dti, expected)

        # 创建微秒级别频率的日期范围，并进行索引比较
        dti = date_range("2016-01-01", "2016-01-01 00:00:01", freq="us", unit="us")
        rng = np.arange(1_451_606_400_000_000, 1_451_606_401_000_001, dtype=np.int64)
        expected = DatetimeIndex(rng.view("M8[us]"), freq="us")
        tm.assert_index_equal(dti, expected)

        # 创建纳秒级别频率的日期范围，并进行索引比较
        dti = date_range("2016-01-01", "2016-01-01 00:00:00.001", freq="ns", unit="ns")
        rng = np.arange(
            1_451_606_400_000_000_000, 1_451_606_400_001_000_001, dtype=np.int64
        )
        expected = DatetimeIndex(rng.view("M8[ns]"), freq="ns")
        tm.assert_index_equal(dti, expected)

    def test_date_range_freq_lower_than_endpoints(self):
        start = Timestamp("2022-10-19 11:50:44.719781")
        end = Timestamp("2022-10-19 11:50:47.066458")

        # 开始和结束时间无法损失精度地转换为 "s" 单位，因此会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            date_range(start, end, periods=3, unit="s")

        # 但可以损失精度地转换为 "us" 单位
        dti = date_range(start, end, periods=2, unit="us")
        rng = np.array(
            [start.as_unit("us")._value, end.as_unit("us")._value], dtype=np.int64
        )
        expected = DatetimeIndex(rng.view("M8[us]"))
        tm.assert_index_equal(dti, expected)

    def test_date_range_non_nano(self):
        start = np.datetime64("1066-10-14")  # Battle of Hastings
        end = np.datetime64("2305-07-13")  # Jean-Luc Picard's birthday

        # 创建秒级别频率的日期范围，并进行类型和频率的断言
        dti = date_range(start, end, freq="D", unit="s")
        assert dti.freq == "D"
        assert dti.dtype == "M8[s]"

        # 生成预期的秒级别日期范围数组，并进行断言比较
        exp = np.arange(
            start.astype("M8[s]").view("i8"),
            (end + 1).astype("M8[s]").view("i8"),
            24 * 3600,
        ).view("M8[s]")

        tm.assert_numpy_array_equal(dti.to_numpy(), exp)


class TestDateRangeNonTickFreq:
    # Tests revolving around less-common (non-Tick) `freq` keywords.
    def test_date_range_custom_business_month_begin(self, unit):
        # 创建一个USFederalHolidayCalendar对象，用于处理美国联邦假日
        hcal = USFederalHolidayCalendar()
        # 创建一个CBMonthBegin对象，表示自定义的商业月初频率，使用上述节假日日历
        freq = offsets.CBMonthBegin(calendar=hcal)
        # 使用指定的频率生成日期范围，从"20120101"到"20130101"，返回日期时间索引对象
        dti = date_range(start="20120101", end="20130101", freq=freq, unit=unit)
        # 断言所有生成的日期时间都符合指定的频率要求
        assert all(freq.is_on_offset(x) for x in dti)

        # 预期的日期时间索引，表示自定义商业月初的日期范围
        expected = DatetimeIndex(
            [
                "2012-01-03",
                "2012-02-01",
                "2012-03-01",
                "2012-04-02",
                "2012-05-01",
                "2012-06-01",
                "2012-07-02",
                "2012-08-01",
                "2012-09-04",
                "2012-10-01",
                "2012-11-01",
                "2012-12-03",
            ],
            dtype=f"M8[{unit}]",
            freq=freq,
        )
        # 使用tm.assert_index_equal方法断言生成的日期时间索引与预期的一致
        tm.assert_index_equal(dti, expected)

    def test_date_range_custom_business_month_end(self, unit):
        # 创建一个USFederalHolidayCalendar对象，用于处理美国联邦假日
        hcal = USFederalHolidayCalendar()
        # 创建一个CBMonthEnd对象，表示自定义的商业月末频率，使用上述节假日日历
        freq = offsets.CBMonthEnd(calendar=hcal)
        # 使用指定的频率生成日期范围，从"20120101"到"20130101"，返回日期时间索引对象
        dti = date_range(start="20120101", end="20130101", freq=freq, unit=unit)
        # 断言所有生成的日期时间都符合指定的频率要求
        assert all(freq.is_on_offset(x) for x in dti)

        # 预期的日期时间索引，表示自定义商业月末的日期范围
        expected = DatetimeIndex(
            [
                "2012-01-31",
                "2012-02-29",
                "2012-03-30",
                "2012-04-30",
                "2012-05-31",
                "2012-06-29",
                "2012-07-31",
                "2012-08-31",
                "2012-09-28",
                "2012-10-31",
                "2012-11-30",
                "2012-12-31",
            ],
            dtype=f"M8[{unit}]",
            freq=freq,
        )
        # 使用tm.assert_index_equal方法断言生成的日期时间索引与预期的一致
        tm.assert_index_equal(dti, expected)

    def test_date_range_with_custom_holidays(self, unit):
        # 创建一个CustomBusinessHour对象，表示自定义的工作小时频率，从"15:00"开始，指定节假日["2020-11-26"]
        freq = offsets.CustomBusinessHour(start="15:00", holidays=["2020-11-26"])
        # 使用指定的频率生成日期范围，从"2020-11-25 15:00"开始，生成4个时间点，返回日期时间索引对象
        result = date_range(start="2020-11-25 15:00", periods=4, freq=freq, unit=unit)
        # 预期的日期时间索引，表示自定义工作小时的日期范围
        expected = DatetimeIndex(
            [
                "2020-11-25 15:00:00",
                "2020-11-25 16:00:00",
                "2020-11-27 15:00:00",
                "2020-11-27 16:00:00",
            ],
            dtype=f"M8[{unit}]",
            freq=freq,
        )
        # 使用tm.assert_index_equal方法断言生成的日期时间索引与预期的一致
        tm.assert_index_equal(result, expected)
    # 定义测试方法，用于验证日期范围内的工作时间索引是否正确生成
    def test_date_range_businesshour(self, unit):
        # 创建日期时间索引对象，包含指定的日期时间点，使用给定的时间单位
        idx = DatetimeIndex(
            [
                "2014-07-04 09:00",
                "2014-07-04 10:00",
                "2014-07-04 11:00",
                "2014-07-04 12:00",
                "2014-07-04 13:00",
                "2014-07-04 14:00",
                "2014-07-04 15:00",
                "2014-07-04 16:00",
            ],
            dtype=f"M8[{unit}]",
            freq="bh",
        )
        # 使用 date_range 函数生成日期范围对象，确保时间单位和频率与索引对象相同
        rng = date_range("2014-07-04 09:00", "2014-07-04 16:00", freq="bh", unit=unit)
        # 断言生成的索引对象与预期的索引对象相等
        tm.assert_index_equal(idx, rng)

        # 重新定义日期时间索引对象，包含另一组指定的日期时间点，使用给定的时间单位
        idx = DatetimeIndex(
            ["2014-07-04 16:00", "2014-07-07 09:00"], dtype=f"M8[{unit}]", freq="bh"
        )
        # 使用 date_range 函数生成日期范围对象，确保时间单位和频率与索引对象相同
        rng = date_range("2014-07-04 16:00", "2014-07-07 09:00", freq="bh", unit=unit)
        # 断言生成的索引对象与预期的索引对象相等
        tm.assert_index_equal(idx, rng)

        # 重新定义日期时间索引对象，包含更大范围的日期时间点，使用给定的时间单位
        idx = DatetimeIndex(
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
            dtype=f"M8[{unit}]",
            freq="bh",
        )
        # 使用 date_range 函数生成日期范围对象，确保时间单位和频率与索引对象相同
        rng = date_range("2014-07-04 09:00", "2014-07-08 16:00", freq="bh", unit=unit)
        # 断言生成的索引对象与预期的索引对象相等
        tm.assert_index_equal(idx, rng)
    # 测试日期范围函数，生成商业小时频率的日期索引
    def test_date_range_business_hour2(self, unit):
        # 生成从 "2014-07-04 15:00" 到 "2014-07-08 10:00" 的商业小时频率日期索引
        idx1 = date_range(
            start="2014-07-04 15:00", end="2014-07-08 10:00", freq="bh", unit=unit
        )
        # 从 "2014-07-04 15:00" 开始，生成包含 12 个商业小时频率的日期索引
        idx2 = date_range(start="2014-07-04 15:00", periods=12, freq="bh", unit=unit)
        # 从 "2014-07-08 10:00" 结束，生成包含 12 个商业小时频率的日期索引
        idx3 = date_range(end="2014-07-08 10:00", periods=12, freq="bh", unit=unit)
        # 预期的日期时间索引
        expected = DatetimeIndex(
            [
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
            ],
            dtype=f"M8[{unit}]",
            freq="bh",
        )
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(idx1, expected)
        tm.assert_index_equal(idx2, expected)
        tm.assert_index_equal(idx3, expected)

        # 生成从 "2014-07-04 15:45" 到 "2014-07-08 10:45" 的商业小时频率日期索引
        idx4 = date_range(
            start="2014-07-04 15:45", end="2014-07-08 10:45", freq="bh", unit=unit
        )
        # 从 "2014-07-04 15:45" 开始，生成包含 12 个商业小时频率的日期索引
        idx5 = date_range(start="2014-07-04 15:45", periods=12, freq="bh", unit=unit)
        # 从 "2014-07-08 10:45" 结束，生成包含 12 个商业小时频率的日期索引
        idx6 = date_range(end="2014-07-08 10:45", periods=12, freq="bh", unit=unit)

        # 预期的日期时间索引，加上 45 分钟后转换为给定单位
        expected2 = expected + Timedelta(minutes=45).as_unit(unit)
        expected2.freq = "bh"
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(idx4, expected2)
        tm.assert_index_equal(idx5, expected2)
        tm.assert_index_equal(idx6, expected2)

    # 测试日期范围函数，对商业小时频率的短期范围进行验证
    def test_date_range_business_hour_short(self, unit):
        # GH#49835
        # 从 "2014-07-01 10:00" 开始，生成一个商业小时频率的日期索引，仅包含一个时间点
        idx4 = date_range(start="2014-07-01 10:00", freq="bh", periods=1, unit=unit)
        # 预期的日期时间索引
        expected4 = DatetimeIndex(["2014-07-01 10:00"], dtype=f"M8[{unit}]", freq="bh")
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(idx4, expected4)

    # 测试日期范围函数，生成以年开始的日期索引
    def test_date_range_year_start(self, unit):
        # see GH#9313
        # 从 "1/1/2013" 到 "7/1/2017" 生成年初频率的日期索引
        rng = date_range("1/1/2013", "7/1/2017", freq="YS", unit=unit)
        # 预期的日期时间索引
        exp = DatetimeIndex(
            ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"],
            dtype=f"M8[{unit}]",
            freq="YS",
        )
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(rng, exp)

    # 测试日期范围函数，生成以年结束的日期索引
    def test_date_range_year_end(self, unit):
        # see GH#9313
        # 从 "1/1/2013" 到 "7/1/2017" 生成年末频率的日期索引
        rng = date_range("1/1/2013", "7/1/2017", freq="YE", unit=unit)
        # 预期的日期时间索引
        exp = DatetimeIndex(
            ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-31"],
            dtype=f"M8[{unit}]",
            freq="YE",
        )
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(rng, exp)

    # 测试日期范围函数，生成负频率的以年结束的日期索引
    def test_date_range_negative_freq_year_end(self, unit):
        # GH#11018
        # 从 "2011-12-31" 开始，生成包含 3 个负二年结束频率的日期索引
        rng = date_range("2011-12-31", freq="-2YE", periods=3, unit=unit)
        # 预期的日期时间索引
        exp = DatetimeIndex(
            ["2011-12-31", "2009-12-31", "2007-12-31"], dtype=f"M8[{unit}]", freq="-2YE"
        )
        # 验证生成的索引是否等于预期的索引
        tm.assert_index_equal(rng, exp)
        # 断言生成的索引频率是否等于预期的频率
        assert rng.freq == "-2YE"
    def test_date_range_business_year_end_year(self, unit):
        # 根据GH#9313，测试生成指定频率（BYE）的日期范围
        rng = date_range("1/1/2013", "7/1/2017", freq="BYE", unit=unit)
        # 期望的日期索引，按照指定的单位和频率生成
        exp = DatetimeIndex(
            ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-30"],
            dtype=f"M8[{unit}]",
            freq="BYE",
        )
        # 使用测试断言库检查生成的日期范围是否符合期望
        tm.assert_index_equal(rng, exp)

    def test_date_range_bms(self, unit):
        # GH#1645，测试生成指定频率（BMS）的日期范围
        result = date_range("1/1/2000", periods=10, freq="BMS", unit=unit)

        # 期望的日期索引，按照指定的单位和频率生成
        expected = DatetimeIndex(
            [
                "2000-01-03",
                "2000-02-01",
                "2000-03-01",
                "2000-04-03",
                "2000-05-01",
                "2000-06-01",
                "2000-07-03",
                "2000-08-01",
                "2000-09-01",
                "2000-10-02",
            ],
            dtype=f"M8[{unit}]",
            freq="BMS",
        )
        # 使用测试断言库检查生成的日期范围是否符合期望
        tm.assert_index_equal(result, expected)

    def test_date_range_semi_month_begin(self, unit):
        dates = [
            datetime(2007, 12, 15),
            datetime(2008, 1, 1),
            datetime(2008, 1, 15),
            datetime(2008, 2, 1),
            datetime(2008, 2, 15),
            datetime(2008, 3, 1),
            datetime(2008, 3, 15),
            datetime(2008, 4, 1),
            datetime(2008, 4, 15),
            datetime(2008, 5, 1),
            datetime(2008, 5, 15),
            datetime(2008, 6, 1),
            datetime(2008, 6, 15),
            datetime(2008, 7, 1),
            datetime(2008, 7, 15),
            datetime(2008, 8, 1),
            datetime(2008, 8, 15),
            datetime(2008, 9, 1),
            datetime(2008, 9, 15),
            datetime(2008, 10, 1),
            datetime(2008, 10, 15),
            datetime(2008, 11, 1),
            datetime(2008, 11, 15),
            datetime(2008, 12, 1),
            datetime(2008, 12, 15),
        ]
        # 确保使用DatetimeIndex生成的日期范围与直接指定的日期列表结果一致
        result = date_range(start=dates[0], end=dates[-1], freq="SMS", unit=unit)
        # 期望的日期索引，按照指定的单位和频率生成
        exp = DatetimeIndex(dates, dtype=f"M8[{unit}]", freq="SMS")
        # 使用测试断言库检查生成的日期范围是否符合期望
        tm.assert_index_equal(result, exp)
    def test_date_range_semi_month_end(self, unit):
        # 定义一组日期列表，包括每月中旬和月末的日期
        dates = [
            datetime(2007, 12, 31),
            datetime(2008, 1, 15),
            datetime(2008, 1, 31),
            datetime(2008, 2, 15),
            datetime(2008, 2, 29),
            datetime(2008, 3, 15),
            datetime(2008, 3, 31),
            datetime(2008, 4, 15),
            datetime(2008, 4, 30),
            datetime(2008, 5, 15),
            datetime(2008, 5, 31),
            datetime(2008, 6, 15),
            datetime(2008, 6, 30),
            datetime(2008, 7, 15),
            datetime(2008, 7, 31),
            datetime(2008, 8, 15),
            datetime(2008, 8, 31),
            datetime(2008, 9, 15),
            datetime(2008, 9, 30),
            datetime(2008, 10, 15),
            datetime(2008, 10, 31),
            datetime(2008, 11, 15),
            datetime(2008, 11, 30),
            datetime(2008, 12, 15),
            datetime(2008, 12, 31),
        ]
        # 确保通过 DatetimeIndex 生成的日期范围与预期结果一致
        result = date_range(start=dates[0], end=dates[-1], freq="SME", unit=unit)
        exp = DatetimeIndex(dates, dtype=f"M8[{unit}]", freq="SME")
        tm.assert_index_equal(result, exp)

    def test_date_range_week_of_month(self, unit):
        # GH#20517
        # 注意：这里的起始日期并非按照该频率的 offset 来确定
        result = date_range(start="20110101", periods=1, freq="WOM-1MON", unit=unit)
        expected = DatetimeIndex(["2011-01-03"], dtype=f"M8[{unit}]", freq="WOM-1MON")
        tm.assert_index_equal(result, expected)

        result2 = date_range(start="20110101", periods=2, freq="WOM-1MON", unit=unit)
        expected2 = DatetimeIndex(
            ["2011-01-03", "2011-02-07"], dtype=f"M8[{unit}]", freq="WOM-1MON"
        )
        tm.assert_index_equal(result2, expected2)

    def test_date_range_week_of_month2(self, unit):
        # GH#5115, GH#5348
        # 根据指定的日期范围和频率生成日期序列，例如每月第一个星期六
        result = date_range("2013-1-1", periods=4, freq="WOM-1SAT", unit=unit)
        expected = DatetimeIndex(
            ["2013-01-05", "2013-02-02", "2013-03-02", "2013-04-06"],
            dtype=f"M8[{unit}]",
            freq="WOM-1SAT",
        )
        tm.assert_index_equal(result, expected)

    def test_date_range_negative_freq_month_end(self, unit):
        # GH#11018
        # 根据负数频率生成日期范围，例如每两个月的倒数第二天
        rng = date_range("2011-01-31", freq="-2ME", periods=3, unit=unit)
        exp = DatetimeIndex(
            ["2011-01-31", "2010-11-30", "2010-09-30"], dtype=f"M8[{unit}]", freq="-2ME"
        )
        tm.assert_index_equal(rng, exp)
        # 确保生成的日期范围具有正确的频率
        assert rng.freq == "-2ME"
    def test_date_range_fy5253(self, unit):
        # 创建一个 FY5253 频率的时间偏移对象，指定从一月开始，星期三，最接近的变体
        freq = offsets.FY5253(startingMonth=1, weekday=3, variation="nearest")
        # 使用指定的频率生成日期范围，从 "2013-01-01" 开始，生成两个日期
        dti = date_range(
            start="2013-01-01",
            periods=2,
            freq=freq,
            unit=unit,
        )
        # 期望的日期时间索引，包含 ["2013-01-31", "2014-01-30"] 两个日期，数据类型为指定的单元类型
        expected = DatetimeIndex(
            ["2013-01-31", "2014-01-30"], dtype=f"M8[{unit}]", freq=freq
        )

        # 断言生成的日期时间索引与期望的索引相等
        tm.assert_index_equal(dti, expected)

    @pytest.mark.parametrize(
        "freqstr,offset",
        [
            ("QS", offsets.QuarterBegin(startingMonth=1)),
            ("BQE", offsets.BQuarterEnd(startingMonth=12)),
            ("W-SUN", offsets.Week(weekday=6)),
        ],
    )
    def test_date_range_freqstr_matches_offset(self, freqstr, offset):
        # 指定开始日期为 1999 年 12 月 25 日
        sdate = datetime(1999, 12, 25)
        # 指定结束日期为 2000 年 1 月 1 日
        edate = datetime(2000, 1, 1)

        # 使用频率字符串生成日期范围 idx1
        idx1 = date_range(start=sdate, end=edate, freq=freqstr)
        # 使用偏移对象生成日期范围 idx2
        idx2 = date_range(start=sdate, end=edate, freq=offset)
        # 断言两个日期范围的长度相等
        assert len(idx1) == len(idx2)
        # 断言两个日期范围的频率相同
        assert idx1.freq == idx2.freq

    def test_date_range_partial_day_year_end(self, unit):
        # GH#56134
        # 从 "2021-12-31 00:00:01" 到 "2023-10-31 00:00:00" 生成每年的最后一天的日期范围
        rng = date_range(
            start="2021-12-31 00:00:01",
            end="2023-10-31 00:00:00",
            freq="YE",
            unit=unit,
        )
        # 期望的日期时间索引，包含 ["2021-12-31 00:00:01", "2022-12-31 00:00:01"] 两个日期，数据类型为指定的单元类型，频率为每年末
        exp = DatetimeIndex(
            ["2021-12-31 00:00:01", "2022-12-31 00:00:01"],
            dtype=f"M8[{unit}]",
            freq="YE",
        )
        # 断言生成的日期时间索引与期望的索引相等
        tm.assert_index_equal(rng, exp)

    def test_date_range_negative_freq_year_end_inbounds(self, unit):
        # GH#56147
        # 从 "2023-10-31 00:00:00" 到 "2021-10-31 00:00:00" 生成每年末的日期范围，频率为每年末向前一年
        rng = date_range(
            start="2023-10-31 00:00:00",
            end="2021-10-31 00:00:00",
            freq="-1YE",
            unit=unit,
        )
        # 期望的日期时间索引，包含 ["2022-12-31 00:00:00", "2021-12-31 00:00:00"] 两个日期，数据类型为指定的单元类型，频率为每年末向前一年
        exp = DatetimeIndex(
            ["2022-12-31 00:00:00", "2021-12-31 00:00:00"],
            dtype=f"M8[{unit}]",
            freq="-1YE",
        )
        # 断言生成的日期时间索引与期望的索引相等
        tm.assert_index_equal(rng, exp)
```