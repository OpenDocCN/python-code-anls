# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_common.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 dateutil.tz.tzlocal 模块中的 tzlocal 类
from dateutil.tz.tz import tzlocal

# 导入 pytest 测试框架
import pytest

# 从 pandas._libs.tslibs 中导入 OutOfBoundsDatetime 和 Timestamp 类
from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
)

# 从 pandas.compat 模块中导入 IS64、WASM、is_platform_windows 函数
from pandas.compat import (
    IS64,
    WASM,
    is_platform_windows,
)

# 从 pandas.tseries.offsets 模块中导入多个日期偏移类
from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessHour,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CDay,
    CustomBusinessHour,
    DateOffset,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    MonthEnd,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)

# 定义一个函数 _get_offset，根据给定的偏移类实例化相应的偏移对象
def _get_offset(klass, value=1, normalize=False):
    # 根据偏移类的不同进行实例化
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
        # 默认情况下，使用给定的 value 和 normalize 参数实例化偏移对象
        klass = klass(value, normalize=normalize)
    return klass

# 定义一个 pytest fixture，用于参数化测试
@pytest.fixture(
    params=[
        BDay,
        BusinessHour,
        BMonthEnd,
        BMonthBegin,
        BQuarterEnd,
        BQuarterBegin,
        BYearEnd,
        BYearBegin,
        CDay,
        CustomBusinessHour,
        CBMonthEnd,
        CBMonthBegin,
        MonthEnd,
        MonthBegin,
        SemiMonthBegin,
        SemiMonthEnd,
        QuarterEnd,
        LastWeekOfMonth,
        WeekOfMonth,
        Week,
        YearBegin,
        YearEnd,
        FY5253,
        FY5253Quarter,
        DateOffset,
    ]
)
# 定义一个带参数的 fixture 函数 _offset，返回每个参数化的偏移类
def _offset(request):
    return request.param

# 用 pytest.mark.skipif 装饰器标记的测试函数，条件是在 WASM 环境下跳过测试
@pytest.mark.skipif(WASM, reason="OverflowError received on WASM")
# 定义测试函数 test_apply_out_of_range，接受 request、tz_naive_fixture、_offset 参数
def test_apply_out_of_range(request, tz_naive_fixture, _offset):
    tz = tz_naive_fixture

    # 尝试创建一个超出范围的时间戳；如果创建失败则跳过这个偏移的测试
    # 详细错误处理略
    try:
        # 如果偏移量是 BusinessHour 或 CustomBusinessHour 类型
        if _offset in (BusinessHour, CustomBusinessHour):
            # 使用值 100000 在 BusinessHour 中由于夏令时差异而失败
            offset = _get_offset(_offset, value=100000)
        else:
            # 使用值 10000 获取偏移量
            offset = _get_offset(_offset, value=10000)

        # 将偏移量应用到 "20080101" 时间戳上，生成新的时间戳对象 result
        result = Timestamp("20080101") + offset
        # 断言 result 是 datetime 对象
        assert isinstance(result, datetime)
        # 断言 result 不带时区信息
        assert result.tzinfo is None

        # 创建带有时区信息的 Timestamp 对象 t
        t = Timestamp("20080101", tz=tz)
        # 将偏移量应用到 t 上，生成新的时间戳对象 result
        result = t + offset
        # 断言 result 是 datetime 对象
        assert isinstance(result, datetime)
        # 如果 tz 不为 None，则断言 t 有时区信息
        if tz is not None:
            assert t.tzinfo is not None

        # 处理特定条件下的测试标记应用
        if (
            isinstance(tz, tzlocal)
            and ((not IS64) or WASM)
            and _offset is not DateOffset
        ):
            # 在非 64 位机器上，如果出现 OutOfBoundsDatetime 错误
            # 则在下一个测试之前会跳出 try 块
            request.applymarker(
                pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
            )
        elif (
            isinstance(tz, tzlocal)
            and is_platform_windows()
            and _offset in (QuarterEnd, BQuarterBegin, BQuarterEnd)
        ):
            # 在 Windows 平台上，在特定情况下应用测试标记
            request.applymarker(
                pytest.mark.xfail(reason="After GH#49737 t.tzinfo is None on CI")
            )
        # 断言 t 和 result 的时区信息相同
        assert str(t.tzinfo) == str(result.tzinfo)

    except OutOfBoundsDatetime:
        # 捕获 OutOfBoundsDatetime 异常，不做处理
        pass
    except (ValueError, KeyError):
        # 捕获 ValueError 或 KeyError 异常，通常是由于创建无效偏移量而忽略
        pass
# 比较两个 _offset 对象是否相等，根源是 GH#456：未实现 __ne__ 方法
def test_offsets_compare_equal(_offset):
    # 创建两个 _offset 实例
    offset1 = _offset()
    offset2 = _offset()
    # 断言 offset1 不等于 offset2
    assert not offset1 != offset2
    # 断言 offset1 等于 offset2
    assert offset1 == offset2


# 使用参数化测试，对日期与偏移量进行减法操作的测试
@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],  # 工作日偏移量，向前推两个工作日
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],  # 业务小时偏移量，向前推3个小时
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(  # 自定义工作小时偏移量，包含特定假期
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],  # 半月末偏移量，向前推两个半月
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],  # 半月初偏移量，向前推两个半月
        [Timestamp(2008, 1, 2), Week(2)],  # 整周偏移量，向前推两周
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],  # 每月第几周偏移量，向前推两个月的第二周
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],  # 每月最后一周偏移量，向前推两个月的最后一周
    ],
)
def test_rsub(date, offset2):
    # 断言日期减去偏移量的结果等于负偏移量应用在日期上的结果
    assert date - offset2 == (-offset2)._apply(date)


# 使用参数化测试，对日期与偏移量进行加法操作的测试
@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],  # 工作日偏移量，向后推两个工作日
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],  # 业务小时偏移量，向后推3个小时
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(  # 自定义工作小时偏移量，包含特定假期
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],  # 半月末偏移量，向后推两个半月
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],  # 半月初偏移量，向后推两个半月
        [Timestamp(2008, 1, 2), Week(2)],  # 整周偏移量，向后推两周
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],  # 每月第几周偏移量，向后推两个月的第二周
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],  # 每月最后一周偏移量，向后推两个月的最后一周
    ],
)
def test_radd(date, offset2):
    # 断言日期加上偏移量的结果等于偏移量加上日期的结果
    assert date + offset2 == offset2 + date


# 使用参数化测试，对日期与偏移量进行减法操作异常情况的测试
@pytest.mark.parametrize(
    "date, offset_box, offset2",
    [
        [Timestamp(2008, 1, 1), BDay, BDay(2)],  # 工作日偏移量的类型，向前推两个工作日
        [Timestamp(2008, 1, 2), SemiMonthEnd, SemiMonthEnd(2)],  # 半月末偏移量的类型，向前推两个半月
        [Timestamp(2008, 1, 2), SemiMonthBegin, SemiMonthBegin(2)],  # 半月初偏移量的类型，向前推两个半月
        [Timestamp(2008, 1, 2), Week, Week(2)],  # 整周偏移量的类型，向前推两周
        [Timestamp(2008, 1, 2), WeekOfMonth, WeekOfMonth(2)],  # 每月第几周偏移量的类型，向前推两个月的第二周
        [Timestamp(2008, 1, 2), LastWeekOfMonth, LastWeekOfMonth(2)],  # 每月最后一周偏移量的类型，向前推两个月的最后一周
    ],
)
def test_sub(date, offset_box, offset2):
    off = offset2
    msg = "Cannot subtract datetime from offset"
    # 断言使用未定义 __sub__ 方法的偏移量与日期相减会引发 TypeError 异常，且异常消息匹配特定信息
    with pytest.raises(TypeError, match=msg):
        off - date

    # 断言偏移量乘以2再减去偏移量的结果等于偏移量本身
    assert 2 * off - off == off
    # 断言日期减去偏移量的结果等于日期加上负偏移量的结果
    assert date - offset2 == date + offset_box(-2)
    # 断言日期减去偏移量的结果等于日期减去偏移量乘以2再减去偏移量的结果
    assert date - offset2 == date - (2 * off - off)


# 使用参数化测试，对偏移量与偏移量乘法的测试
@pytest.mark.parametrize(
    "offset_box, offset1",
    [
        [BDay, BDay()],  # 工作日偏移量的类型与实例
        [LastWeekOfMonth, LastWeekOfMonth()],  # 每月最后一周偏移量的类型与实例
        [WeekOfMonth, WeekOfMonth()],  # 每月第几周偏移量的类型与实例
        [Week, Week()],  # 整周偏移量的类型与实例
        [SemiMonthBegin, SemiMonthBegin()],  # 半月初偏移量的类型与实例
        [SemiMonthEnd, SemiMonthEnd()],  # 半月末偏移量的类型与实例
        [
            CustomBusinessHour,
            CustomBusinessHour(weekmask="Tue Wed Thu Fri"),  # 自定义工作小时偏移量的类型与实例
        ],
        [BusinessHour, BusinessHour()],  # 业务小时偏移量的类型与实例
    ],
)
def test_Mult1(offset_box, offset1):
    dt = Timestamp(2008, 1, 2)
    # 断言日期加上10倍的偏移量等于日期加上偏移量的10倍
    assert dt + 10 * offset1 == dt + offset_box(10)
    # 断言日期加上5倍的偏移量等于日期加上偏移量的5倍
    assert dt + 5 * offset1 == dt + offset_box(5)


# 对比字符串的比较功能
def test_compare_str(_offset):
    # GH#23524
    # 这个测试用例的注释暂缺
    # 使用 _get_offset 函数获取 _offset 的偏移量
    off = _get_offset(_offset)

    # 断言 off 不等于 "infer"
    assert not off == "infer"
    
    # 断言 off 不等于 "foo"
    assert off != "foo"
    
    # 注意：不等式运算仅对 Tick 子类实现；相关的测试位于 test_ticks 中
```