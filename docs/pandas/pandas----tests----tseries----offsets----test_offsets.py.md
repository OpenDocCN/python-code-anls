# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_offsets.py`

```
"""
Tests of pandas.tseries.offsets
"""

# 导入必要的模块和库
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

from pandas._libs.tslibs import (  # 导入 pandas 时间序列相关的 C 模块
    NaT,
    Timedelta,
    Timestamp,
    conversion,
    timezones,
)
import pandas._libs.tslibs.offsets as liboffsets  # 导入时间偏移量相关的 C 模块
from pandas._libs.tslibs.offsets import (  # 导入时间偏移量相关的 C 模块中的特定类和函数
    _get_offset,
    _offset_map,
    to_offset,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG  # 导入无效频率错误信息

from pandas import (  # 导入 pandas 主要模块和类
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试工具模块
from pandas.tests.tseries.offsets.common import WeekDay  # 导入自定义的工作日类 WeekDay

from pandas.tseries import offsets  # 导入 pandas 时间序列的偏移量模块
from pandas.tseries.offsets import (  # 导入 pandas 时间序列的具体偏移量类
    FY5253,
    BDay,
    BMonthEnd,
    BusinessHour,
    CustomBusinessDay,
    CustomBusinessHour,
    CustomBusinessMonthBegin,
    CustomBusinessMonthEnd,
    DateOffset,
    Easter,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    Nano,
    Tick,
    Week,
    WeekOfMonth,
)

_ARITHMETIC_DATE_OFFSET = [  # 定义日期偏移量的列表
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
]


def _create_offset(klass, value=1, normalize=False):
    """
    Create an instance of offset class with optional parameters.
    """
    # 根据传入的偏移量类创建一个实例
    if klass is FY5253:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is FY5253Quarter:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            qtr_with_extra_week=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass


@pytest.fixture(  # 定义 Pytest 的 fixture：month_classes
    params=[
        getattr(offsets, o)
        for o in offsets.__all__
        if issubclass(getattr(offsets, o), liboffsets.MonthOffset)
        and o != "MonthOffset"
    ]
)
def month_classes(request):
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(  # 定义 Pytest 的 fixture：offset_types
    params=[
        getattr(offsets, o) for o in offsets.__all__ if o not in ("Tick", "BaseOffset")
    ]
)
def offset_types(request):
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture  # 定义 Pytest 的 fixture：dt
def dt():
    """
    Fixture returning a Timestamp object for January 2, 2008.
    """
    return Timestamp(datetime(2008, 1, 2))


@pytest.fixture  # 定义 Pytest 的 fixture：expecteds
def expecteds():
    """
    Fixture for expected values used in tests.
    """
    # executed value created by _create_offset
    # are applied to 2011/01/01 09:00 (Saturday)
    # used for .apply and .rollforward
    # 返回一个字典，包含不同时间戳对象，用于日期和时间的操作
    return {
        # 创建一个时间戳对象，表示"2011-01-02 09:00:00"
        "Day": Timestamp("2011-01-02 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-02 09:00:00"
        "DateOffset": Timestamp("2011-01-02 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 09:00:00"
        "BusinessDay": Timestamp("2011-01-03 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 09:00:00"
        "CustomBusinessDay": Timestamp("2011-01-03 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-31 09:00:00"
        "CustomBusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 09:00:00"
        "CustomBusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        # 创建一个时间戳对象，表示"2011-02-01 09:00:00"
        "MonthBegin": Timestamp("2011-02-01 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 09:00:00"
        "BusinessMonthBegin": Timestamp("2011-01-03 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-31 09:00:00"
        "MonthEnd": Timestamp("2011-01-31 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-15 09:00:00"
        "SemiMonthEnd": Timestamp("2011-01-15 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-15 09:00:00"
        "SemiMonthBegin": Timestamp("2011-01-15 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-31 09:00:00"
        "BusinessMonthEnd": Timestamp("2011-01-31 09:00:00"),
        # 创建一个时间戳对象，表示"2012-01-01 09:00:00"
        "YearBegin": Timestamp("2012-01-01 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 09:00:00"
        "BYearBegin": Timestamp("2011-01-03 09:00:00"),
        # 创建一个时间戳对象，表示"2011-12-31 09:00:00"
        "YearEnd": Timestamp("2011-12-31 09:00:00"),
        # 创建一个时间戳对象，表示"2011-12-30 09:00:00"
        "BYearEnd": Timestamp("2011-12-30 09:00:00"),
        # 创建一个时间戳对象，表示"2011-03-01 09:00:00"
        "QuarterBegin": Timestamp("2011-03-01 09:00:00"),
        # 创建一个时间戳对象，表示"2011-03-01 09:00:00"
        "BQuarterBegin": Timestamp("2011-03-01 09:00:00"),
        # 创建一个时间戳对象，表示"2011-03-31 09:00:00"
        "QuarterEnd": Timestamp("2011-03-31 09:00:00"),
        # 创建一个时间戳对象，表示"2011-03-31 09:00:00"
        "BQuarterEnd": Timestamp("2011-03-31 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 10:00:00"
        "BusinessHour": Timestamp("2011-01-03 10:00:00"),
        # 创建一个时间戳对象，表示"2011-01-03 10:00:00"
        "CustomBusinessHour": Timestamp("2011-01-03 10:00:00"),
        # 创建一个时间戳对象，表示"2011-01-08 09:00:00"
        "WeekOfMonth": Timestamp("2011-01-08 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-29 09:00:00"
        "LastWeekOfMonth": Timestamp("2011-01-29 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-25 09:00:00"
        "FY5253Quarter": Timestamp("2011-01-25 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-25 09:00:00"
        "FY5253": Timestamp("2011-01-25 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-08 09:00:00"
        "Week": Timestamp("2011-01-08 09:00:00"),
        # 创建一个时间戳对象，表示"2011-04-24 09:00:00"
        "Easter": Timestamp("2011-04-24 09:00:00"),
        # 创建一个时间戳对象，表示"2011-01-01 10:00:00"
        "Hour": Timestamp("2011-01-01 10:00:00"),
        # 创建一个时间戳对象，表示"2011-01-01 09:01:00"
        "Minute": Timestamp("2011-01-01 09:01:00"),
        # 创建一个时间戳对象，表示"2011-01-01 09:00:01"
        "Second": Timestamp("2011-01-01 09:00:01"),
        # 创建一个时间戳对象，表示"2011-01-01 09:00:00.001000"
        "Milli": Timestamp("2011-01-01 09:00:00.001000"),
        # 创建一个时间戳对象，表示"2011-01-01 09:00:00.000001"
        "Micro": Timestamp("2011-01-01 09:00:00.000001"),
        # 创建一个时间戳对象，表示"2011-01-01T09:00:00.000000001"
        "Nano": Timestamp("2011-01-01T09:00:00.000000001"),
    }
`
class TestCommon:
    # 测试不可变性
    def test_immutable(self, offset_types):
        # GH#21341 检查 __setattr__ 是否引发异常
        offset = _create_offset(offset_types)
        msg = "objects is not writable|DateOffset objects are immutable"
        # 检查是否抛出 AttributeError 异常，并匹配指定的错误信息
        with pytest.raises(AttributeError, match=msg):
            offset.normalize = True
        with pytest.raises(AttributeError, match=msg):
            offset.n = 91

    # 测试返回类型
    def test_return_type(self, offset_types):
        offset = _create_offset(offset_types)

        # 确保返回 Timestamp 对象
        result = Timestamp("20080101") + offset
        assert isinstance(result, Timestamp)

        # 确保返回 NaT
        assert NaT + offset is NaT
        assert offset + NaT is NaT

        assert NaT - offset is NaT
        assert (-offset)._apply(NaT) is NaT

    # 测试 offset.n 属性
    def test_offset_n(self, offset_types):
        offset = _create_offset(offset_types)
        assert offset.n == 1

        neg_offset = offset * -1
        assert neg_offset.n == -1

        mul_offset = offset * 3
        assert mul_offset.n == 3

    # 测试 offset._validate_n 对于 timedelta64 类型参数的处理
    def test_offset_timedelta64_arg(self, offset_types):
        off = _create_offset(offset_types)

        td64 = np.timedelta64(4567, "s")
        # 检查是否抛出 TypeError 异常，并匹配指定的错误信息
        with pytest.raises(TypeError, match="argument must be an integer"):
            type(off)(n=td64, **off.kwds)

    # 测试 offset 与 ndarray 相乘的结果
    def test_offset_mul_ndarray(self, offset_types):
        off = _create_offset(offset_types)

        expected = np.array([[off, off * 2], [off * 3, off * 4]])

        # 检查与 ndarray 相乘后的结果是否与预期一致
        result = np.array([[1, 2], [3, 4]]) * off
        tm.assert_numpy_array_equal(result, expected)

        result = off * np.array([[1, 2], [3, 4]])
        tm.assert_numpy_array_equal(result, expected)

    # 测试 offset.freqstr 属性
    def test_offset_freqstr(self, offset_types):
        offset = _create_offset(offset_types)

        freqstr = offset.freqstr
        # 确保 freqstr 的值符合预期范围内的一种
        if freqstr not in ("<Easter>", "<DateOffset: days=1>", "LWOM-SAT"):
            code = _get_offset(freqstr)
            assert offset.rule_code == code
    def _check_offsetfunc_works(self, offset, funcname, dt, expected, normalize=False):
        # 如果 normalize=True 并且 offset 是 Tick 的子类，则不允许，见 GH#21427
        if normalize and issubclass(offset, Tick):
            return

        # 根据 offset 创建一个偏移对象 offset_s
        offset_s = _create_offset(offset, normalize=normalize)
        
        # 获取偏移对象 offset_s 的方法 funcname
        func = getattr(offset_s, funcname)

        # 测试 func 对 dt 的调用结果
        result = func(dt)
        assert isinstance(result, Timestamp)  # 确保结果是 Timestamp 类型
        assert result == expected  # 确保结果等于预期值 expected

        # 将 dt 转换为 Timestamp 对象再调用 func 进行测试
        result = func(Timestamp(dt))
        assert isinstance(result, Timestamp)
        assert result == expected

        # 查看 GH-14101
        ts = Timestamp(dt) + Nano(5)
        # 测试纳秒是否被保留
        with tm.assert_produces_warning(None):
            result = func(ts)

        assert isinstance(result, Timestamp)
        if normalize is False:
            assert result == expected + Nano(5)  # 如果 normalize=False，确保纳秒偏移也被保留
        else:
            assert result == expected

        if isinstance(dt, np.datetime64):
            # 当输入是 np.datetime64 时，测试时区设定
            return

        # 遍历不同的时区进行测试
        for tz in [
            None,
            "UTC",
            "Asia/Tokyo",
            "US/Eastern",
            "dateutil/Asia/Tokyo",
            "dateutil/US/Pacific",
        ]:
            # 获取在 tz 时区下的预期本地化时间
            expected_localize = expected.tz_localize(tz)
            tz_obj = timezones.maybe_get_tz(tz)
            # 将 dt 本地化到 tz_obj 时区下
            dt_tz = conversion.localize_pydatetime(dt, tz_obj)

            # 测试 func 对 dt_tz 的调用结果
            result = func(dt_tz)
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            # 将 Timestamp 对象 dt 本地化到 tz 时区下再调用 func 进行测试
            result = func(Timestamp(dt, tz=tz))
            assert isinstance(result, Timestamp)
            assert result == expected_localize

            # 查看 GH-14101
            ts = Timestamp(dt, tz=tz) + Nano(5)
            # 测试纳秒是否被保留
            with tm.assert_produces_warning(None):
                result = func(ts)
            assert isinstance(result, Timestamp)
            if normalize is False:
                assert result == expected_localize + Nano(5)  # 如果 normalize=False，确保纳秒偏移也被保留
            else:
                assert result == expected_localize

    def test_apply(self, offset_types, expecteds):
        # 设置一个标准日期时间 sdt 和 np.datetime64 对象 ndt
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        # 获取预期的结果 expected 和标准化后的 expected_norm
        expected = expecteds[offset_types.__name__]
        expected_norm = Timestamp(expected.date())

        # 遍历 sdt 和 ndt 进行测试
        for dt in [sdt, ndt]:
            # 测试 _check_offsetfunc_works 方法的正常调用
            self._check_offsetfunc_works(offset_types, "_apply", dt, expected)

            # 测试 _check_offsetfunc_works 方法的 normalize=True 的调用
            self._check_offsetfunc_works(
                offset_types, "_apply", dt, expected_norm, normalize=True
            )
    # 测试函数：验证在给定偏移类型和期望值的情况下，rollforward 方法的行为是否正确
    def test_rollforward(self, offset_types, expecteds):
        # 复制期望结果，以避免修改原始数据
        expecteds = expecteds.copy()

        # 如果目标在偏移量上，结果不会改变
        no_changes = [
            "Day",
            "MonthBegin",
            "SemiMonthBegin",
            "YearBegin",
            "Week",
            "Hour",
            "Minute",
            "Second",
            "Milli",
            "Micro",
            "Nano",
            "DateOffset",
        ]
        # 将这些偏移类型的期望值设置为固定的时间戳
        for n in no_changes:
            expecteds[n] = Timestamp("2011/01/01 09:00")

        # 特定偏移类型的期望值设置为固定的时间戳
        expecteds["BusinessHour"] = Timestamp("2011-01-03 09:00:00")
        expecteds["CustomBusinessHour"] = Timestamp("2011-01-03 09:00:00")

        # 当 normalize=True 时，期望值会发生变化
        norm_expected = expecteds.copy()
        # 将所有键对应的值都规范化为日期部分
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())

        # 初始化规范化后的期望结果
        normalized = {
            "Day": Timestamp("2011-01-02 00:00:00"),
            "DateOffset": Timestamp("2011-01-02 00:00:00"),
            "MonthBegin": Timestamp("2011-02-01 00:00:00"),
            "SemiMonthBegin": Timestamp("2011-01-15 00:00:00"),
            "YearBegin": Timestamp("2012-01-01 00:00:00"),
            "Week": Timestamp("2011-01-08 00:00:00"),
            "Hour": Timestamp("2011-01-01 00:00:00"),
            "Minute": Timestamp("2011-01-01 00:00:00"),
            "Second": Timestamp("2011-01-01 00:00:00"),
            "Milli": Timestamp("2011-01-01 00:00:00"),
            "Micro": Timestamp("2011-01-01 00:00:00"),
        }
        # 更新规范化后的期望结果到规范化期望字典中
        norm_expected.update(normalized)

        # 初始化两种日期时间类型
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64("2011-01-01 09:00")

        # 针对两种日期时间类型进行测试
        for dt in [sdt, ndt]:
            # 获取对应偏移类型的预期值，并检查 rollforward 方法的输出是否符合预期
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, "rollforward", dt, expected)
            # 获取规范化后对应偏移类型的预期值，并检查 rollforward 方法的输出是否符合预期，此时 normalize=True
            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(
                offset_types, "rollforward", dt, expected, normalize=True
            )

    # 测试函数：验证在给定偏移类型和期望值的情况下，is_on_offset 方法的行为是否正确
    def test_is_on_offset(self, offset_types, expecteds):
        # 获取期望值对应的日期时间对象
        dt = expecteds[offset_types.__name__]
        # 创建对应偏移类型的偏移对象
        offset_s = _create_offset(offset_types)
        # 断言给定的日期时间对象是否在偏移对象的偏移位置上
        assert offset_s.is_on_offset(dt)

        # 当 normalize=True 时，is_on_offset 方法检查时间是否为 00:00:00
        if issubclass(offset_types, Tick):
            # 对于 Tick 的子类，不允许使用 normalize=True，见 GH#21427
            return
        # 创建具有规范化功能的对应偏移类型的偏移对象
        offset_n = _create_offset(offset_types, normalize=True)
        # 断言给定的日期时间对象是否不在规范化后的偏移对象的偏移位置上
        assert not offset_n.is_on_offset(dt)

        # 对于 BusinessHour 和 CustomBusinessHour 偏移类型，默认情况下（非规范化状态），规范化后的时间不能在工作时间范围内
        if offset_types in (BusinessHour, CustomBusinessHour):
            return
        # 创建一个日期对象，以给定日期时间对象的年、月、日初始化
        date = datetime(dt.year, dt.month, dt.day)
        # 断言规范化后的偏移对象是否在给定日期对象的偏移位置上
        assert offset_n.is_on_offset(date)
    # 定义一个测试方法，用于测试时间偏移量的加法操作
    def test_add(self, offset_types, tz_naive_fixture, expecteds):
        # 获取时区修正后的时区对象
        tz = tz_naive_fixture
        # 创建一个指定日期时间的 datetime 对象
        dt = datetime(2011, 1, 1, 9, 0)

        # 创建偏移量对象
        offset_s = _create_offset(offset_types)
        # 获取预期结果
        expected = expecteds[offset_types.__name__]

        # 对 datetime 对象进行偏移量加法操作，生成结果 datetime 对象
        result_dt = dt + offset_s
        # 对 Timestamp 对象进行偏移量加法操作，生成结果 Timestamp 对象
        result_ts = Timestamp(dt) + offset_s
        # 遍历结果列表，确保每个结果是 Timestamp 类型，并且与预期结果相等
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        # 获取时区本地化后的预期结果
        expected_localize = expected.tz_localize(tz)
        # 对带时区的 Timestamp 对象进行偏移量加法操作，生成结果 Timestamp 对象
        result = Timestamp(dt, tz=tz) + offset_s
        # 确保结果是 Timestamp 类型，并且与时区本地化后的预期结果相等
        assert isinstance(result, Timestamp)
        assert result == expected_localize

        # 对于 Tick 的子类，不支持 normalize=True 的情况，直接返回
        if issubclass(offset_types, Tick):
            return

        # 创建带 normalize=True 参数的偏移量对象
        offset_s = _create_offset(offset_types, normalize=True)
        # 获取仅包含日期部分的预期结果
        expected = Timestamp(expected.date())

        # 对 datetime 对象进行偏移量加法操作，生成结果 datetime 对象
        result_dt = dt + offset_s
        # 对 Timestamp 对象进行偏移量加法操作，生成结果 Timestamp 对象
        result_ts = Timestamp(dt) + offset_s
        # 遍历结果列表，确保每个结果是 Timestamp 类型，并且与日期部分预期结果相等
        for result in [result_dt, result_ts]:
            assert isinstance(result, Timestamp)
            assert result == expected

        # 获取时区本地化后的预期结果
        expected_localize = expected.tz_localize(tz)
        # 对带时区的 Timestamp 对象进行偏移量加法操作，生成结果 Timestamp 对象
        result = Timestamp(dt, tz=tz) + offset_s
        # 确保结果是 Timestamp 类型，并且与时区本地化后的日期部分预期结果相等
        assert isinstance(result, Timestamp)
        assert result == expected_localize
    ):
        # GH#12724, GH#30336
        # 创建偏移量对象
        offset_s = _create_offset(offset_types)

        # 创建空的 DatetimeIndex 对象，使用 tz_naive_fixture 作为时区
        dti = DatetimeIndex([], tz=tz_naive_fixture).as_unit("ns")

        # 检查 offset_s 是否不属于一些特定的偏移量类型，如果是，则不优化 apply_index
        if not isinstance(
            offset_s,
            (
                Easter,
                WeekOfMonth,
                LastWeekOfMonth,
                CustomBusinessDay,
                BusinessHour,
                CustomBusinessHour,
                CustomBusinessMonthBegin,
                CustomBusinessMonthEnd,
                FY5253,
                FY5253Quarter,
            ),
        ):
            performance_warning = False  # 不需要性能警告

        # stacklevel 检查很慢，我们只在某些情况下进行检查
        check_stacklevel = tz_naive_fixture is None

        # 断言操作产生警告
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = dti + offset_s
        tm.assert_index_equal(result, dti)  # 断言结果与原始 DatetimeIndex 相等

        # 断言操作产生警告
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = offset_s + dti
        tm.assert_index_equal(result, dti)  # 断言结果与原始 DatetimeIndex 相等

        # 获取 DatetimeIndex 对象的内部数据
        dta = dti._data

        # 断言操作产生警告
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = dta + offset_s
        tm.assert_equal(result, dta)  # 断言结果与原始数据相等

        # 断言操作产生警告
        with tm.assert_produces_warning(
            performance_warning, check_stacklevel=check_stacklevel
        ):
            result = offset_s + dta
        tm.assert_equal(result, dta)  # 断言结果与原始数据相等
    def test_add_dt64_ndarray_non_nano(self, offset_types, unit):
        # 检查非纳秒精度情况下的结果是否与纳秒精度一致
        # 创建偏移量对象
        off = _create_offset(offset_types)

        # 创建一个日期范围，从 "2016-01-01" 开始，包含35天，频率为每天 ("D")，单位为给定的单位
        dti = date_range("2016-01-01", periods=35, freq="D", unit=unit)

        # 对日期时间索引应用偏移量，并设置频率为 None
        result = (dti + off)._with_freq(None)

        # 期望的单位初始化为给定的单位
        exp_unit = unit
        # 如果偏移量是 Tick 类型，并且其分辨率高于 dti 数据的分辨率
        if isinstance(off, Tick) and off._creso > dti._data._creso:
            # 将期望的单位设置为 Timedelta(off) 的单位
            exp_unit = Timedelta(off).unit
        
        # 创建一个预期的日期时间索引，其中每个元素都加上偏移量 off
        expected = DatetimeIndex([x + off for x in dti]).as_unit(exp_unit)

        # 断言结果索引与预期索引相等
        tm.assert_index_equal(result, expected)
class TestDateOffset:
    # 设置方法，用于每个测试方法执行前清空偏移映射表
    def setup_method(self):
        _offset_map.clear()

    # 测试 DateOffset 类的 repr 方法
    def test_repr(self):
        # 返回 DateOffset 对象的字符串表示形式（不进行断言）
        repr(DateOffset())
        repr(DateOffset(2))
        repr(2 * DateOffset())
        repr(2 * DateOffset(months=2))

    # 测试 DateOffset 类的乘法运算符重载
    def test_mul(self):
        # 断言两种形式的乘法运算结果相等
        assert DateOffset(2) == 2 * DateOffset(1)
        assert DateOffset(2) == DateOffset(1) * 2

    # 使用 pytest 参数化装饰器，测试 DateOffset 类的构造函数
    @pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
    def test_constructor(self, kwd, request):
        # 如果关键字参数是 "millisecond"，则标记为预期失败的测试用例
        if kwd == "millisecond":
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason="Constructing DateOffset object with `millisecond` is not "
                    "yet supported.",
                )
            )
        # 使用指定关键字参数创建 DateOffset 对象
        offset = DateOffset(**{kwd: 2})
        # 断言对象的关键字参数与预期相符
        assert offset.kwds == {kwd: 2}
        assert getattr(offset, kwd) == 2

    # 测试 DateOffset 类的默认构造函数
    def test_default_constructor(self, dt):
        # 断言日期对象加上 DateOffset(2) 后的结果符合预期
        assert (dt + DateOffset(2)) == datetime(2008, 1, 4)

    # 测试 DateOffset 类的复制方法
    def test_copy(self):
        # 断言不同方式创建的 DateOffset 对象复制后相等
        assert DateOffset(months=2).copy() == DateOffset(months=2)
        assert DateOffset(milliseconds=1).copy() == DateOffset(milliseconds=1)

    # 使用 pytest 参数化装饰器，测试 DateOffset 类的加法运算
    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2009-01-02",
                "2008-02-02",
                "2008-01-09",
                "2008-01-03",
                "2008-01-02 01:00:00",
                "2008-01-02 00:01:00",
                "2008-01-02 00:00:01",
                "2008-01-02 00:00:00.001000000",
                "2008-01-02 00:00:00.000001000",
            ],
        ),
    )
    def test_add(self, arithmatic_offset_type, expected, dt):
        # 断言日期对象加上指定的 DateOffset 对象后的结果符合预期
        assert DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    # 使用 pytest 参数化装饰器，测试 DateOffset 类的减法运算
    @pytest.mark.parametrize(
        "arithmatic_offset_type, expected",
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                "2007-01-02",
                "2007-12-02",
                "2007-12-26",
                "2008-01-01",
                "2008-01-01 23:00:00",
                "2008-01-01 23:59:00",
                "2008-01-01 23:59:59",
                "2008-01-01 23:59:59.999000000",
                "2008-01-01 23:59:59.999999000",
            ],
        ),
    )
    def test_sub(self, arithmatic_offset_type, expected, dt):
        # 断言日期对象减去指定的 DateOffset 对象后的结果符合预期
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)
        # 断言试图从 DateOffset 对象减去日期对象会引发 TypeError 异常
        with pytest.raises(TypeError, match="Cannot subtract datetime from offset"):
            DateOffset(**{arithmatic_offset_type: 1}) - dt
    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        # 使用 zip 函数将 _ARITHMETIC_DATE_OFFSET、range(1, 10) 和日期字符串列表一一配对
        zip(
            _ARITHMETIC_DATE_OFFSET,  # 使用 _ARITHMETIC_DATE_OFFSET 的元素作为第一个参数
            range(1, 10),  # 使用范围从 1 到 9 的整数作为第二个参数
            [  # 使用给定的日期字符串列表作为第三个参数
                "2009-01-02",
                "2008-03-02",
                "2008-01-23",
                "2008-01-06",
                "2008-01-02 05:00:00",
                "2008-01-02 00:06:00",
                "2008-01-02 00:00:07",
                "2008-01-02 00:00:00.008000000",
                "2008-01-02 00:00:00.000009000",
            ],
        ),
    )
    # 定义测试方法 test_mul_add，使用参数化测试验证日期偏移乘法和加法的计算结果
    def test_mul_add(self, arithmatic_offset_type, n, expected, dt):
        # 断言日期偏移乘法和加法的结果等于给定的时间戳
        assert DateOffset(**{arithmatic_offset_type: 1}) * n + dt == Timestamp(expected)
        assert n * DateOffset(**{arithmatic_offset_type: 1}) + dt == Timestamp(expected)
        assert dt + DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt + n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    @pytest.mark.parametrize(
        "arithmatic_offset_type, n, expected",
        # 使用 zip 函数将 _ARITHMETIC_DATE_OFFSET、range(1, 10) 和日期字符串列表一一配对
        zip(
            _ARITHMETIC_DATE_OFFSET,  # 使用 _ARITHMETIC_DATE_OFFSET 的元素作为第一个参数
            range(1, 10),  # 使用范围从 1 到 9 的整数作为第二个参数
            [  # 使用给定的日期字符串列表作为第三个参数
                "2007-01-02",
                "2007-11-02",
                "2007-12-12",
                "2007-12-29",
                "2008-01-01 19:00:00",
                "2008-01-01 23:54:00",
                "2008-01-01 23:59:53",
                "2008-01-01 23:59:59.992000000",
                "2008-01-01 23:59:59.999991000",
            ],
        ),
    )
    # 定义测试方法 test_mul_sub，使用参数化测试验证日期偏移乘法和减法的计算结果
    def test_mul_sub(self, arithmatic_offset_type, n, expected, dt):
        # 断言日期偏移乘法和减法的结果等于给定的时间戳
        assert dt - DateOffset(**{arithmatic_offset_type: 1}) * n == Timestamp(expected)
        assert dt - n * DateOffset(**{arithmatic_offset_type: 1}) == Timestamp(expected)

    # 定义测试方法 test_leap_year，验证闰年计算
    def test_leap_year(self):
        d = datetime(2008, 1, 31)
        # 断言在闰年中，添加一个月后的日期等于预期的日期
        assert (d + DateOffset(months=1)) == datetime(2008, 2, 29)

    # 定义测试方法 test_eq，验证日期偏移对象的相等性
    def test_eq(self):
        offset1 = DateOffset(days=1)
        offset2 = DateOffset(days=365)

        # 断言两个不同的日期偏移对象不相等
        assert offset1 != offset2

        # 断言不同参数构造的日期偏移对象不相等
        assert DateOffset(milliseconds=3) != DateOffset(milliseconds=7)

    @pytest.mark.parametrize(
        "offset_kwargs, expected_arg",
        # 使用参数化测试验证各种日期偏移参数与预期时间戳的关系
        [
            ({"microseconds": 1, "milliseconds": 1}, "2022-01-01 00:00:00.001001"),
            ({"seconds": 1, "milliseconds": 1}, "2022-01-01 00:00:01.001"),
            ({"minutes": 1, "milliseconds": 1}, "2022-01-01 00:01:00.001"),
            ({"hours": 1, "milliseconds": 1}, "2022-01-01 01:00:00.001"),
            ({"days": 1, "milliseconds": 1}, "2022-01-02 00:00:00.001"),
            ({"weeks": 1, "milliseconds": 1}, "2022-01-08 00:00:00.001"),
            ({"months": 1, "milliseconds": 1}, "2022-02-01 00:00:00.001"),
            ({"years": 1, "milliseconds": 1}, "2023-01-01 00:00:00.001"),
        ],
    )
    # 定义一个测试方法，用于测试时间偏移量与时间戳相加的情况
    def test_milliseconds_combination(self, offset_kwargs, expected_arg):
        # 创建一个时间偏移对象，使用传入的偏移量参数
        offset = DateOffset(**offset_kwargs)
        # 创建一个时间戳对象，设定为"2022-01-01"
        ts = Timestamp("2022-01-01")
        # 将时间戳对象与时间偏移对象相加，得到结果时间戳
        result = ts + offset
        # 创建一个预期的时间戳对象，使用预期的时间参数
        expected = Timestamp(expected_arg)

        # 断言结果时间戳与预期时间戳相等
        assert result == expected

    # 定义一个测试方法，用于测试无效的时间偏移量参数的情况
    def test_offset_invalid_arguments(self):
        # 定义错误消息的正则表达式模式，用于匹配异常消息
        msg = "^Invalid argument/s or bad combination of arguments"
        # 使用 pytest 的上下文管理器，检查是否会引发 ValueError 异常，并验证错误消息
        with pytest.raises(ValueError, match=msg):
            # 创建一个时间偏移对象，传入无效的偏移量参数（皮秒为单位）
            DateOffset(picoseconds=1)
class TestOffsetNames:
    # 测试类 TestOffsetNames，用于测试偏移名称的功能
    def test_get_offset_name(self):
        # 测试获取偏移名称的方法

        # 断言工作日偏移的频率字符串为 "B"
        assert BDay().freqstr == "B"
        # 断言每两个工作日偏移的频率字符串为 "2B"
        assert BDay(2).freqstr == "2B"
        # 断言月末工作日偏移的频率字符串为 "BME"
        assert BMonthEnd().freqstr == "BME"
        # 断言每周一的偏移频率字符串为 "W-MON"
        assert Week(weekday=0).freqstr == "W-MON"
        # 断言每周二的偏移频率字符串为 "W-TUE"
        assert Week(weekday=1).freqstr == "W-TUE"
        # 断言每周三的偏移频率字符串为 "W-WED"
        assert Week(weekday=2).freqstr == "W-WED"
        # 断言每周四的偏移频率字符串为 "W-THU"
        assert Week(weekday=3).freqstr == "W-THU"
        # 断言每周五的偏移频率字符串为 "W-FRI"
        assert Week(weekday=4).freqstr == "W-FRI"

        # 断言每月最后一个周日的偏移频率字符串为 "LWOM-SUN"
        assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"


def test_get_offset():
    # 测试获取偏移的函数 _get_offset()

    # 测试传入无效的频率字符串 "gibberish" 是否会抛出 ValueError 异常，并匹配错误消息 INVALID_FREQ_ERR_MSG
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("gibberish")
    # 测试传入无效的频率字符串 "QS-JAN-B" 是否会抛出 ValueError 异常，并匹配错误消息 INVALID_FREQ_ERR_MSG
    with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
        _get_offset("QS-JAN-B")

    # 定义测试数据对，包括有效的频率字符串和对应的 Offset 对象
    pairs = [
        ("B", BDay()),
        ("b", BDay()),
        ("bme", BMonthEnd()),
        ("Bme", BMonthEnd()),
        ("W-MON", Week(weekday=0)),
        ("W-TUE", Week(weekday=1)),
        ("W-WED", Week(weekday=2)),
        ("W-THU", Week(weekday=3)),
        ("W-FRI", Week(weekday=4)),
    ]

    # 遍历测试数据对
    for name, expected in pairs:
        # 调用 _get_offset() 函数获取对应的 Offset 对象
        offset = _get_offset(name)
        # 断言获取的 Offset 对象与预期的 Offset 对象相等
        assert (
            offset == expected
        ), f"Expected {name!r} to yield {expected!r} (actual: {offset!r})"


def test_get_offset_legacy():
    # 测试获取偏移的函数 _get_offset() 的旧版别名功能

    # 定义测试数据对，包括旧版别名和预期的 Offset 对象
    pairs = [("w@Sat", Week(weekday=5))]
    # 遍历测试数据对
    for name, expected in pairs:
        # 测试传入旧版别名是否会抛出 ValueError 异常，并匹配错误消息 INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            _get_offset(name)


class TestOffsetAliases:
    # 测试类 TestOffsetAliases，用于测试偏移别名的功能
    def setup_method(self):
        # 设置每个测试方法的前置操作，清空偏移别名映射表 _offset_map
        _offset_map.clear()

    def test_alias_equality(self):
        # 测试偏移别名的相等性

        # 遍历偏移别名映射表 _offset_map 的每个键值对
        for k, v in _offset_map.items():
            # 如果值为 None，则跳过
            if v is None:
                continue
            # 断言每个偏移别名与其对应的复制别名相等
            assert k == v.copy()

    def test_rule_code(self):
        # 测试偏移规则码的生成

        # 定义测试用的后缀列表
        lst = ["ME", "MS", "BME", "BMS", "D", "B", "h", "min", "s", "ms", "us"]
        # 遍历后缀列表
        for k in lst:
            # 断言偏移别名 k 的规则码与 _get_offset(k).rule_code 相等
            assert k == _get_offset(k).rule_code
            # 断言偏移别名 k 已经缓存到偏移别名映射表 _offset_map 中
            assert k in _offset_map
            # 断言偏移别名 k 乘以 3 后的规则码与 _get_offset(k) * 3 的规则码相等
            assert k == (_get_offset(k) * 3).rule_code

        # 定义基本偏移别名和后缀列表
        suffix_lst = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        base = "W"
        # 遍历后缀列表
        for v in suffix_lst:
            # 组合偏移别名 alias
            alias = "-".join([base, v])
            # 断言偏移别名 alias 的规则码与 _get_offset(alias).rule_code 相等
            assert alias == _get_offset(alias).rule_code
            # 断言偏移别名 alias 乘以 5 后的规则码与 _get_offset(alias) * 5 的规则码相等
            assert alias == (_get_offset(alias) * 5).rule_code

        # 定义基本偏移别名列表和月份后缀列表
        suffix_lst = [
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        ]
        base_lst = ["YE", "YS", "BYE", "BYS", "QE", "QS", "BQE", "BQS"]
        # 遍历基本偏移别名列表
        for base in base_lst:
            # 遍历月份后缀列表
            for v in suffix_lst:
                # 组合偏移别名 alias
                alias = "-".join([base, v])
                # 断言偏移别名 alias 的规则码与 _get_offset(alias).rule_code 相等
                assert alias == _get_offset(alias).rule_code
                # 断言偏移别名 alias 乘以 5 后的规则码与 _get_offset(alias) * 5 的规则码相等
                assert alias == (_get_offset(alias) * 5).rule_code


def test_freq_offsets():
    # 测试频率偏移

    # 创建一个工作日偏移对象 off，偏移 1 天，偏移量为 timedelta(0, 1800) 秒（30 分钟）
    off = BDay(1, offset=timedelta(0, 1800))
    # 断言工作日偏移对象 off 的频率字符串为 "B+30Min"
    assert off.freqstr == "B+30Min"
    # 创建一个工作日频率对象，偏移量为1个工作日，时间偏移为负30分钟（即向前半小时）
    off = BDay(1, offset=timedelta(0, -1800))
    # 断言，验证工作日频率对象的字符串表示是否为 "B-30Min"
    assert off.freqstr == "B-30Min"
class TestReprNames:
    def test_str_for_named_is_name(self):
        # 定义月份前缀组合列表
        month_prefixes = ["YE", "YS", "BYE", "BYS", "QE", "BQE", "BQS", "QS"]
        # 创建所有可能的名称组合
        names = [
            prefix + "-" + month
            for prefix in month_prefixes
            for month in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ]
        # 定义星期几列表
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        # 添加额外的名称：W-加星期几的格式
        names += ["W-" + day for day in days]
        # 添加额外的名称：WOM-加星期数加星期几的格式
        names += ["WOM-" + week + day for week in ("1", "2", "3", "4") for day in days]
        
        # 清空偏移映射表
        _offset_map.clear()
        # 遍历所有名称，获取对应的偏移量，并断言偏移量的频率字符串与名称相同
        for name in names:
            offset = _get_offset(name)
            assert offset.freqstr == name


# ---------------------------------------------------------------------


def test_valid_default_arguments(offset_types):
    # 检查调用构造函数而不传递任何关键字参数时是否产生有效的偏移量
    cls = offset_types
    cls()


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_month_attributes(kwd, month_classes):
    # GH#18226
    cls = month_classes
    # 检查不能创建带有非预期关键字参数的对象，如MonthEnd(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_month_offset_name(month_classes):
    # GH#33757 off.name with n != 1 should not raise AttributeError
    # 创建月份类的实例，并断言当n != 1时，obj2的名称不应引发AttributeError
    obj = month_classes(1)
    obj2 = month_classes(2)
    assert obj2.name == obj.name


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd, request):
    # 如果关键字参数为"millisecond"，标记测试为预期失败，因为尚不支持使用`millisecond`构造DateOffset对象
    if kwd == "millisecond":
        request.applymarker(
            pytest.mark.xfail(
                raises=NotImplementedError,
                reason="Constructing DateOffset object with `millisecond` is not "
                "yet supported.",
            )
        )
    # 检查所有在liboffsets._relativedelta_kwds中指定的参数是否有效
    DateOffset(**{kwd: 1})


@pytest.mark.parametrize("kwd", sorted(liboffsets._relativedelta_kwds))
def test_valid_tick_attributes(kwd, tick_classes):
    # GH#18226
    cls = tick_classes
    # 检查不能创建带有非预期关键字参数的对象，如Hour(weeks=3)
    msg = rf"__init__\(\) got an unexpected keyword argument '{kwd}'"
    with pytest.raises(TypeError, match=msg):
        cls(**{kwd: 3})


def test_validate_n_error():
    # 断言传递给DateOffset构造函数的n参数必须是整数，否则引发TypeError异常
    with pytest.raises(TypeError, match="argument must be an integer"):
        DateOffset(n="Doh!")

    # 断言传递给MonthBegin构造函数的n参数必须是timedelta对象，否则引发TypeError异常
    with pytest.raises(TypeError, match="argument must be an integer"):
        MonthBegin(n=timedelta(1))
    # 使用 pytest 框架的 `raises` 上下文管理器来测试异常情况
    with pytest.raises(TypeError, match="argument must be an integer"):
        # 调用 BDay 类，传入一个 NumPy 数组作为参数 n，期望引发 TypeError 异常，并且异常消息需要匹配指定的字符串
        BDay(n=np.array([1, 2], dtype=np.int64))
def test_require_integers(offset_types):
    # 设置测试用例中的类别别名为 cls
    cls = offset_types
    # 使用 pytest 来检查传入非整数时是否会引发 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match="argument must be an integer"):
        cls(n=1.5)


def test_tick_normalize_raises(tick_classes):
    # 检查尝试使用 normalize=True 创建 Tick 对象是否会引发异常
    # GH#21427
    cls = tick_classes
    msg = "Tick offset with `normalize=True` are not allowed."
    with pytest.raises(ValueError, match=msg):
        cls(n=3, normalize=True)


@pytest.mark.parametrize(
    "offset_kwargs, expected_arg",
    [
        # 使用不同的偏移关键字参数和预期结果来测试 DateOffset 的加法和减法
        ({"nanoseconds": 1}, "1970-01-01 00:00:00.000000001"),
        ({"nanoseconds": 5}, "1970-01-01 00:00:00.000000005"),
        ({"nanoseconds": -1}, "1969-12-31 23:59:59.999999999"),
        ({"microseconds": 1}, "1970-01-01 00:00:00.000001"),
        ({"microseconds": -1}, "1969-12-31 23:59:59.999999"),
        ({"seconds": 1}, "1970-01-01 00:00:01"),
        ({"seconds": -1}, "1969-12-31 23:59:59"),
        ({"minutes": 1}, "1970-01-01 00:01:00"),
        ({"minutes": -1}, "1969-12-31 23:59:00"),
        ({"hours": 1}, "1970-01-01 01:00:00"),
        ({"hours": -1}, "1969-12-31 23:00:00"),
        ({"days": 1}, "1970-01-02 00:00:00"),
        ({"days": -1}, "1969-12-31 00:00:00"),
        ({"weeks": 1}, "1970-01-08 00:00:00"),
        ({"weeks": -1}, "1969-12-25 00:00:00"),
        ({"months": 1}, "1970-02-01 00:00:00"),
        ({"months": -1}, "1969-12-01 00:00:00"),
        ({"years": 1}, "1971-01-01 00:00:00"),
        ({"years": -1}, "1969-01-01 00:00:00"),
    ],
)
def test_dateoffset_add_sub(offset_kwargs, expected_arg):
    # 使用给定的偏移关键字参数创建 DateOffset 对象
    offset = DateOffset(**offset_kwargs)
    # 创建一个起始时间戳
    ts = Timestamp(0)
    # 执行时间戳加上偏移量的操作，并验证结果是否与预期相符
    result = ts + offset
    expected = Timestamp(expected_arg)
    assert result == expected
    # 执行时间戳减去偏移量的操作，并验证结果是否恢复到起始时间戳
    result -= offset
    assert result == ts
    # 执行偏移量加上时间戳的操作，并验证结果是否与预期相符
    result = offset + ts
    assert result == expected


def test_dateoffset_add_sub_timestamp_with_nano():
    # 使用分钟和纳秒单位的偏移量创建 DateOffset 对象
    offset = DateOffset(minutes=2, nanoseconds=9)
    # 创建一个起始时间戳
    ts = Timestamp(4)
    # 执行时间戳加上偏移量的操作，并验证结果是否与预期相符
    result = ts + offset
    expected = Timestamp("1970-01-01 00:02:00.000000013")
    assert result == expected
    # 执行时间戳减去偏移量的操作，并验证结果是否恢复到起始时间戳
    result -= offset
    assert result == ts
    # 执行偏移量加上时间戳的操作，并验证结果是否与预期相符
    result = offset + ts
    assert result == expected

    # 使用小时、分钟和纳秒单位的偏移量创建 DateOffset 对象
    offset2 = DateOffset(minutes=2, nanoseconds=9, hour=1)
    assert offset2._use_relativedelta  # 验证是否使用了相对时间差对象
    with tm.assert_produces_warning(None):
        # 确保没有关于丢弃非零纳秒的警告信息产生
        result2 = ts + offset2
    expected2 = Timestamp("1970-01-01 01:02:00.000000013")
    assert result2 == expected2


@pytest.mark.parametrize(
    "attribute",
    [
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ],
)
def test_dateoffset_immutable(attribute):
    # 使用指定属性创建不可变的 DateOffset 对象
    offset = DateOffset(**{attribute: 0})
    msg = "DateOffset objects are immutable"
    # 使用 pytest 验证设置属性时是否会引发 AttributeError 异常，并匹配特定错误消息
    with pytest.raises(AttributeError, match=msg):
        setattr(offset, attribute, 5)


def test_dateoffset_misc():
    # 使用月份和天数单位创建 DateOffset 对象，并访问其频率字符串属性
    oset = offsets.DateOffset(months=2, days=4)
    oset.freqstr
    # 使用断言确保 DateOffset 对象的月份偏移量不等于 2
    assert not offsets.DateOffset(months=2) == 2
# 使用 pytest 的 mark.parametrize 装饰器为 test_construct_int_arg_no_kwargs_assumed_days 函数定义多个参数化测试用例
@pytest.mark.parametrize("n", [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n):
    # GH 45890, 45643: GitHub 上的 issue 编号
    # 根据参数 n 创建 DateOffset 对象
    offset = DateOffset(n)
    # 断言 _offset 属性是否为 timedelta(1)
    assert offset._offset == timedelta(1)
    # 创建 Timestamp 对象，并加上偏移量 offset
    result = Timestamp(2022, 1, 2) + offset
    # 创建预期的 Timestamp 对象
    expected = Timestamp(2022, 1, 2 + n)
    # 断言结果与预期相等
    assert result == expected


# 使用 pytest 的 mark.parametrize 装饰器为 test_dateoffset_add_sub_timestamp_series_with_nano 函数定义多个参数化测试用例
@pytest.mark.parametrize(
    "offset, expected",
    [
        (
            DateOffset(minutes=7, nanoseconds=18),
            Timestamp("2022-01-01 00:07:00.000000018"),
        ),
        (DateOffset(nanoseconds=3), Timestamp("2022-01-01 00:00:00.000000003")),
    ],
)
def test_dateoffset_add_sub_timestamp_series_with_nano(offset, expected):
    # GH 47856: GitHub 上的 issue 编号
    # 创建起始时间的 Timestamp 对象
    start_time = Timestamp("2022-01-01")
    # 复制起始时间
    teststamp = start_time
    # 创建包含起始时间的 Series 对象
    testseries = Series([start_time])
    # 将 offset 加到 Series 中的每个元素
    testseries = testseries + offset
    # 断言第一个元素是否等于预期的 Timestamp
    assert testseries[0] == expected
    # 将 offset 从 Series 中的每个元素减去
    testseries -= offset
    # 断言第一个元素是否等于原始的起始时间戳
    assert testseries[0] == teststamp
    # 将 offset 加到 Series 中的每个元素
    testseries = offset + testseries
    # 断言第一个元素是否等于预期的 Timestamp
    assert testseries[0] == expected


# 使用 pytest 的 mark.parametrize 装饰器为 test_offset_multiplication 函数定义多个参数化测试用例
@pytest.mark.parametrize(
    "n_months, scaling_factor, start_timestamp, expected_timestamp",
    [
        (1, 2, "2020-01-30", "2020-03-30"),
        (2, 1, "2020-01-30", "2020-03-30"),
        (1, 0, "2020-01-30", "2020-01-30"),
        (2, 0, "2020-01-30", "2020-01-30"),
        (1, -1, "2020-01-30", "2019-12-30"),
        (2, -1, "2020-01-30", "2019-11-30"),
    ],
)
def test_offset_multiplication(
    n_months, scaling_factor, start_timestamp, expected_timestamp
):
    # GH 47953: GitHub 上的 issue 编号
    # 创建 DateOffset 对象 mo1
    mo1 = DateOffset(months=n_months)

    # 创建起始时间的 Timestamp 对象和包含其的 Series 对象
    startscalar = Timestamp(start_timestamp)
    startarray = Series([startscalar])

    # 计算结果的 Timestamp 对象和 Series 对象
    resultscalar = startscalar + (mo1 * scaling_factor)
    resultarray = startarray + (mo1 * scaling_factor)

    # 创建预期的 Timestamp 对象和 Series 对象
    expectedscalar = Timestamp(expected_timestamp)
    expectedarray = Series([expectedscalar])

    # 断言标量结果是否等于预期的 Timestamp
    assert resultscalar == expectedscalar
    # 断言 Series 结果是否等于预期的 Series
    tm.assert_series_equal(resultarray, expectedarray)


# 定义 test_dateoffset_operations_on_dataframes 函数
def test_dateoffset_operations_on_dataframes(performance_warning):
    # GH 47953: GitHub 上的 issue 编号
    # 创建包含起始时间和 DateOffset 对象的 DataFrame
    df = DataFrame({"T": [Timestamp("2019-04-30")], "D": [DateOffset(months=1)]})
    # 计算 DataFrame 中的日期偏移量运算结果
    frameresult1 = df["T"] + 26 * df["D"]
    
    # 创建另一个包含多个起始时间和 DateOffset 对象的 DataFrame
    df2 = DataFrame(
        {
            "T": [Timestamp("2019-04-30"), Timestamp("2019-04-30")],
            "D": [DateOffset(months=1), DateOffset(months=1)],
        }
    )
    
    # 创建预期的日期 Timestamp
    expecteddate = Timestamp("2021-06-30")
    
    # 使用 assert_produces_warning 上下文管理器捕获性能警告
    with tm.assert_produces_warning(performance_warning):
        frameresult2 = df2["T"] + 26 * df2["D"]

    # 断言第一个结果是否等于预期的日期
    assert frameresult1[0] == expecteddate
    # 断言第二个结果是否等于预期的日期
    assert frameresult2[0] == expecteddate


# 定义 test_is_yqm_start_end 函数
def test_is_yqm_start_end():
    # 创建多个 Offset 对象
    freq_m = to_offset("ME")
    bm = to_offset("BME")
    qfeb = to_offset("QE-FEB")
    qsfeb = to_offset("QS-FEB")
    bq = to_offset("BQE")
    bqs_apr = to_offset("BQS-APR")
    as_nov = to_offset("YS-NOV")
    tests = [
        (freq_m.is_month_start(Timestamp("2013-06-01")), 1),  # 检查是否为月初，预期值为1
        (bm.is_month_start(Timestamp("2013-06-01")), 0),    # 检查是否为月初，预期值为0
        (freq_m.is_month_start(Timestamp("2013-06-03")), 0),  # 检查是否为月初，预期值为0
        (bm.is_month_start(Timestamp("2013-06-03")), 1),    # 检查是否为月初，预期值为1
        (qfeb.is_month_end(Timestamp("2013-02-28")), 1),    # 检查是否为月末，预期值为1
        (qfeb.is_quarter_end(Timestamp("2013-02-28")), 1),  # 检查是否为季末，预期值为1
        (qfeb.is_year_end(Timestamp("2013-02-28")), 1),     # 检查是否为年末，预期值为1
        (qfeb.is_month_start(Timestamp("2013-03-01")), 1),  # 检查是否为月初，预期值为1
        (qfeb.is_quarter_start(Timestamp("2013-03-01")), 1),  # 检查是否为季初，预期值为1
        (qfeb.is_year_start(Timestamp("2013-03-01")), 1),   # 检查是否为年初，预期值为1
        (qsfeb.is_month_end(Timestamp("2013-03-31")), 1),   # 检查是否为月末，预期值为1
        (qsfeb.is_quarter_end(Timestamp("2013-03-31")), 0), # 检查是否为季末，预期值为0
        (qsfeb.is_year_end(Timestamp("2013-03-31")), 0),    # 检查是否为年末，预期值为0
        (qsfeb.is_month_start(Timestamp("2013-02-01")), 1), # 检查是否为月初，预期值为1
        (qsfeb.is_quarter_start(Timestamp("2013-02-01")), 1),  # 检查是否为季初，预期值为1
        (qsfeb.is_year_start(Timestamp("2013-02-01")), 1),   # 检查是否为年初，预期值为1
        (bq.is_month_end(Timestamp("2013-06-30")), 0),      # 检查是否为月末，预期值为0
        (bq.is_quarter_end(Timestamp("2013-06-30")), 0),    # 检查是否为季末，预期值为0
        (bq.is_year_end(Timestamp("2013-06-30")), 0),       # 检查是否为年末，预期值为0
        (bq.is_month_end(Timestamp("2013-06-28")), 1),      # 检查是否为月末，预期值为1
        (bq.is_quarter_end(Timestamp("2013-06-28")), 1),    # 检查是否为季末，预期值为1
        (bq.is_year_end(Timestamp("2013-06-28")), 0),       # 检查是否为年末，预期值为0
        (bqs_apr.is_month_end(Timestamp("2013-06-30")), 0), # 检查是否为月末，预期值为0
        (bqs_apr.is_quarter_end(Timestamp("2013-06-30")), 0),  # 检查是否为季末，预期值为0
        (bqs_apr.is_year_end(Timestamp("2013-06-30")), 0),   # 检查是否为年末，预期值为0
        (bqs_apr.is_month_end(Timestamp("2013-06-28")), 1),  # 检查是否为月末，预期值为1
        (bqs_apr.is_quarter_end(Timestamp("2013-06-28")), 1),  # 检查是否为季末，预期值为1
        (bqs_apr.is_year_end(Timestamp("2013-03-29")), 1),   # 检查是否为年末，预期值为1
        (as_nov.is_year_start(Timestamp("2013-11-01")), 1),  # 检查是否为年初，预期值为1
        (as_nov.is_year_end(Timestamp("2013-10-31")), 1),    # 检查是否为年末，预期值为1
        (Timestamp("2012-02-01").days_in_month, 29),         # 检查2月有多少天，预期值为29
        (Timestamp("2013-02-01").days_in_month, 28),         # 检查2月有多少天，预期值为28
    ]

    for ts, value in tests:
        assert ts == value  # 断言每个测试结果与预期值一致
```