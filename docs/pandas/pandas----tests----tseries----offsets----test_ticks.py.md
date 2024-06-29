# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_ticks.py`

```
# 导入所需模块和库
from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    example,
    given,
)
import numpy as np
import pytest

# 导入 pandas 相关模块和函数
from pandas._libs.tslibs.offsets import delta_to_tick
from pandas import (
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas._testing._hypothesis import INT_NEG_999_TO_POS_999
from pandas.tests.tseries.offsets.common import assert_offset_equal

# 导入 pandas 时间序列偏移相关模块
from pandas.tseries import offsets
from pandas.tseries.offsets import (
    Hour,
    Micro,
    Milli,
    Minute,
    Nano,
    Second,
)

# ---------------------------------------------------------------------
# Test Helpers

# 定义 Tick 类的列表
tick_classes = [Hour, Minute, Second, Milli, Micro, Nano]

# ---------------------------------------------------------------------


def test_apply_ticks():
    # 测试 Hour 类的加法
    result = offsets.Hour(3) + offsets.Hour(4)
    exp = offsets.Hour(7)
    assert result == exp


def test_delta_to_tick():
    # 测试 delta_to_tick 函数
    delta = timedelta(3)
    tick = delta_to_tick(delta)
    assert tick == offsets.Day(3)

    td = Timedelta(nanoseconds=5)
    tick = delta_to_tick(td)
    assert tick == Nano(5)


@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@example(n=800, m=300)
@example(n=1000, m=5)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_add_sub(cls, n, m):
    # 测试 Tick 类的加减法
    # 对于所有 Tick 子类和整数 n, m，应有 tick(n) + tick(m) == tick(n+m)
    # tick(n) - tick(m) == tick(n-m)
    left = cls(n)
    right = cls(m)

    expected = cls(n + m)
    assert left + right == expected

    expected = cls(n - m)
    assert left - right == expected


@pytest.mark.arm_slow
@pytest.mark.parametrize("cls", tick_classes)
@example(n=2, m=3)
@given(n=INT_NEG_999_TO_POS_999, m=INT_NEG_999_TO_POS_999)
def test_tick_equality(cls, n, m):
    # 测试 Tick 类的相等性和不相等性
    assume(m != n)
    left = cls(n)
    right = cls(m)
    assert left != right

    right = cls(n)
    assert left == right
    assert not left != right

    if n != 0:
        assert cls(n) != cls(-n)


# ---------------------------------------------------------------------


def test_Hour():
    # 测试 Hour 类的功能
    assert_offset_equal(Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 1))
    assert_offset_equal(Hour(-1), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Hour(), datetime(2010, 1, 1), datetime(2010, 1, 1, 2))
    assert_offset_equal(-1 * Hour(), datetime(2010, 1, 1, 1), datetime(2010, 1, 1))

    assert Hour(3) + Hour(2) == Hour(5)
    assert Hour(3) - Hour(2) == Hour()

    assert Hour(4) != Hour(1)


def test_Minute():
    # 测试 Minute 类的功能
    assert_offset_equal(Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1))
    assert_offset_equal(Minute(-1), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Minute(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 2))
    # 确保通过计算后的偏移量等于给定的时间间隔，这里是将当前时间向前调整1分钟，并进行断言验证
    assert_offset_equal(-1 * Minute(), datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))
    
    # 断言：两个分钟对象的加法结果应为5分钟
    assert Minute(3) + Minute(2) == Minute(5)
    
    # 断言：两个分钟对象的减法结果应为0分钟（即空的分钟对象）
    assert Minute(3) - Minute(2) == Minute()
    
    # 断言：两个不同的分钟对象不相等
    assert Minute(5) != Minute()
def test_Second():
    # 测试 Second 类的功能
    assert_offset_equal(Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 1))
    # 测试带有负秒数的 Second 类
    assert_offset_equal(Second(-1), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1))
    # 测试两倍秒数的 Second 类
    assert_offset_equal(
        2 * Second(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 2)
    )
    # 测试负秒数的 Second 类
    assert_offset_equal(
        -1 * Second(), datetime(2010, 1, 1, 0, 0, 1), datetime(2010, 1, 1)
    )

    # 测试 Second 类的加法操作
    assert Second(3) + Second(2) == Second(5)
    # 测试 Second 类的减法操作
    assert Second(3) - Second(2) == Second()


def test_Millisecond():
    # 测试 Milli 类的功能
    assert_offset_equal(
        Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1000)
    )
    # 测试带有负毫秒数的 Milli 类
    assert_offset_equal(
        Milli(-1), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )
    # 测试指定毫秒数的 Milli 类
    assert_offset_equal(
        Milli(2), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    # 测试两倍毫秒数的 Milli 类
    assert_offset_equal(
        2 * Milli(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2000)
    )
    # 测试负毫秒数的 Milli 类
    assert_offset_equal(
        -1 * Milli(), datetime(2010, 1, 1, 0, 0, 0, 1000), datetime(2010, 1, 1)
    )

    # 测试 Milli 类的加法操作
    assert Milli(3) + Milli(2) == Milli(5)
    # 测试 Milli 类的减法操作
    assert Milli(3) - Milli(2) == Milli()


def test_MillisecondTimestampArithmetic():
    # 测试 Milli 类在时间戳算术中的表现
    assert_offset_equal(
        Milli(), Timestamp("2010-01-01"), Timestamp("2010-01-01 00:00:00.001")
    )
    assert_offset_equal(
        Milli(-1), Timestamp("2010-01-01 00:00:00.001"), Timestamp("2010-01-01")
    )


def test_Microsecond():
    # 测试 Micro 类的功能
    assert_offset_equal(Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 1))
    # 测试带有负微秒数的 Micro 类
    assert_offset_equal(
        Micro(-1), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    # 测试指定微秒数的 Micro 类
    assert_offset_equal(
        2 * Micro(), datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 0, 0, 2)
    )
    # 测试两倍微秒数的 Micro 类
    assert_offset_equal(
        -1 * Micro(), datetime(2010, 1, 1, 0, 0, 0, 1), datetime(2010, 1, 1)
    )

    # 测试 Micro 类的加法操作
    assert Micro(3) + Micro(2) == Micro(5)
    # 测试 Micro 类的减法操作
    assert Micro(3) - Micro(2) == Micro()


def test_NanosecondGeneric():
    # 测试 NanosecondGeneric 类的功能
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert timestamp.nanosecond == 0

    # 测试 NanosecondGeneric 类的加法操作
    result = timestamp + Nano(10)
    assert result.nanosecond == 10

    # 测试 NanosecondGeneric 类的加法操作（反向）
    reverse_result = Nano(10) + timestamp
    assert reverse_result.nanosecond == 10


def test_Nanosecond():
    # 测试 Nanosecond 类的功能
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert_offset_equal(Nano(), timestamp, timestamp + np.timedelta64(1, "ns"))
    assert_offset_equal(Nano(-1), timestamp + np.timedelta64(1, "ns"), timestamp)
    assert_offset_equal(2 * Nano(), timestamp, timestamp + np.timedelta64(2, "ns"))
    assert_offset_equal(-1 * Nano(), timestamp + np.timedelta64(1, "ns"), timestamp)

    # 测试 Nanosecond 类的加法操作
    assert Nano(3) + Nano(2) == Nano(5)
    # 测试 Nanosecond 类的减法操作
    assert Nano(3) - Nano(2) == Nano()

    # GH9284 测试 Nanosecond 类的特定操作
    assert Nano(1) + Nano(10) == Nano(11)
    assert Nano(5) + Micro(1) == Nano(1005)
    assert Micro(5) + Nano(1) == Nano(5001)


@pytest.mark.parametrize(
    "kls, expected",
    # 定义一个包含多个元组的列表，每个元组包含一个时间单位和对应的时间增量
    [
        # 小时单位和增加5小时的时间增量
        (Hour, Timedelta(hours=5)),
        # 分钟单位和增加2小时3分钟的时间增量
        (Minute, Timedelta(hours=2, minutes=3)),
        # 秒单位和增加2小时3秒的时间增量
        (Second, Timedelta(hours=2, seconds=3)),
        # 毫秒单位和增加2小时3毫秒的时间增量
        (Milli, Timedelta(hours=2, milliseconds=3)),
        # 微秒单位和增加2小时3微秒的时间增量
        (Micro, Timedelta(hours=2, microseconds=3)),
        # 纳秒单位和增加2小时3纳秒的时间增量
        (Nano, Timedelta(hours=2, nanoseconds=3)),
    ],
# 定义测试函数，用于测试 Tick 类型的加法操作
def test_tick_addition(kls, expected):
    # 创建一个偏移量为 3 的 Tick 对象
    offset = kls(3)
    # 创建一个时间增量对象，表示 2 小时
    td = Timedelta(hours=2)

    # 对于列表中的每个对象，执行加法操作并进行断言
    for other in [td, td.to_pytimedelta(), td.to_timedelta64()]:
        # 加法操作：偏移量 + 其他对象
        result = offset + other
        assert isinstance(result, Timedelta)  # 结果应为 Timedelta 类型
        assert result == expected  # 结果应等于预期值

        # 加法操作：其他对象 + 偏移量
        result = other + offset
        assert isinstance(result, Timedelta)  # 结果应为 Timedelta 类型
        assert result == expected  # 结果应等于预期值


# 使用 pytest.mark.parametrize 注入不同的 Tick 类型进行测试
@pytest.mark.parametrize("cls", tick_classes)
def test_tick_division(cls):
    # 创建一个 Tick 对象，初始值为 10
    off = cls(10)

    # 确保除法操作能正常工作
    assert off / cls(5) == 2  # 整数除法
    assert off / 2 == cls(5)  # 整数除法
    assert off / 2.0 == cls(5)  # 浮点数除法

    # 确保 Tick 对象与 Pandas 时间增量对象的除法运算正确
    assert off / off._as_pd_timedelta == 1
    assert off / off._as_pd_timedelta.to_timedelta64() == 1

    # 对于不同 Tick 类型，进行额外的除法操作验证
    assert off / Nano(1) == off._as_pd_timedelta / Nano(1)._as_pd_timedelta

    # 当 Tick 类型不是 Nano 时，验证结果可能是更小的 Tick 类型
    if cls is not Nano:
        result = off / 1000
        assert isinstance(result, offsets.Tick)  # 结果应为 Tick 类型
        assert not isinstance(result, cls)  # 结果不应与原 Tick 类型相同
        assert result._as_pd_timedelta == off._as_pd_timedelta / 1000

    # 当 Tick 类型的纳秒增量小于 1 秒时，结果可能是更大的 Tick 类型
    if cls._nanos_inc < Timedelta(seconds=1)._value:
        result = off / 0.001
        assert isinstance(result, offsets.Tick)  # 结果应为 Tick 类型
        assert not isinstance(result, cls)  # 结果不应与原 Tick 类型相同
        assert result._as_pd_timedelta == off._as_pd_timedelta / 0.001


# 测试 Tick 类型乘以浮点数的情况
def test_tick_mul_float():
    off = Micro(2)

    # 保持 Tick 类型不变的情况
    result = off * 1.5
    expected = Micro(3)
    assert result == expected
    assert isinstance(result, Micro)

    # 提升到下一个 Tick 类型的情况
    result = off * 1.25
    expected = Nano(2500)
    assert result == expected
    assert isinstance(result, Nano)


# 使用 pytest.mark.parametrize 注入不同的 Tick 类型进行测试
@pytest.mark.parametrize("cls", tick_classes)
def test_tick_rdiv(cls):
    # 创建一个 Tick 对象，初始值为 10
    off = cls(10)
    delta = off._as_pd_timedelta
    td64 = delta.to_timedelta64()
    instance__type = ".".join([cls.__module__, cls.__name__])
    msg = (
        "unsupported operand type\\(s\\) for \\/: 'int'|'float' and "
        f"'{instance__type}'"
    )

    # 使用 pytest.raises 断言捕获预期的 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        2 / off
    with pytest.raises(TypeError, match=msg):
        2.0 / off

    # 确保除法运算结果正确
    assert (td64 * 2.5) / off == 2.5

    # 对于非 Nano 类型的 Tick，执行额外的除法操作验证
    if cls is not Nano:
        assert (delta.to_pytimedelta() * 2) / off == 2

    # 使用 NumPy 数组进行除法运算，验证结果是否符合预期
    result = np.array([2 * td64, td64]) / off
    expected = np.array([2.0, 1.0])
    tm.assert_numpy_array_equal(result, expected)


# 使用 pytest.mark.parametrize 注入不同的 Tick 类型进行测试
@pytest.mark.parametrize("cls1", tick_classes)
@pytest.mark.parametrize("cls2", tick_classes)
def test_tick_zero(cls1, cls2):
    # 验证 Tick 类型的零值相等性
    assert cls1(0) == cls2(0)
    assert cls1(0) + cls2(0) == cls1(0)

    # 对于非 Nano 类型的 Tick，验证非零加法操作
    if cls1 is not Nano:
        assert cls1(2) + cls2(0) == cls1(2)

    # 对于 Nano 类型的 Tick，验证特定加法操作
    if cls1 is Nano:
        assert cls1(2) + Nano(0) == cls1(2)


# 使用 pytest.mark.parametrize 注入不同的 Tick 类型进行测试
@pytest.mark.parametrize("cls", tick_classes)
def test_tick_equalities(cls):
    # 验证 Tick 类型的等值性
    assert cls() == cls(1)


# 使用 pytest.mark.parametrize 注入不同的 Tick 类型进行测试
@pytest.mark.parametrize("cls", tick_classes)
def test_compare_ticks(cls):
    # 创建 Tick 对象，数值分别为 3 和 4
    three = cls(3)
    four = cls(4)
    # 断言：对象 three 应小于类 cls 创建的对象 4
    assert three < cls(4)
    # 断言：类 cls 创建的对象 3 应小于对象 four
    assert cls(3) < four
    # 断言：对象 four 应大于类 cls 创建的对象 3
    assert four > cls(3)
    # 断言：类 cls 创建的对象 4 应大于对象 three
    assert cls(4) > three
    # 断言：类 cls 创建的对象 3 应等于另一个类 cls 创建的对象 3
    assert cls(3) == cls(3)
    # 断言：类 cls 创建的对象 3 应不等于类 cls 创建的对象 4
    assert cls(3) != cls(4)
# 使用 pytest.mark.parametrize 装饰器为 test_compare_ticks_to_strs 函数参数化，参数为 tick_classes 列表中的每个类(cls)。
@pytest.mark.parametrize("cls", tick_classes)
# 定义测试函数 test_compare_ticks_to_strs，测试 ticks 类和字符串之间的比较行为。
def test_compare_ticks_to_strs(cls):
    # 创建一个 ticks 类的实例 off，参数为 19
    off = cls(19)

    # 这些测试应该适用于任何字符串，但我们特别关注 "infer"，因为在日期/时间差数组/索引构造函数中进行该比较很方便。
    # 检查 off 不等于 "infer"
    assert not off == "infer"
    # 检查 "foo" 不等于 off
    assert not "foo" == off

    # 构建实例类型的字符串表示，例如 "module_name.class_name"
    instance_type = ".".join([cls.__module__, cls.__name__])
    # 构建错误消息字符串，用于 TypeError 异常的匹配
    msg = (
        "'<'|'<='|'>'|'>=' not supported between instances of "
        f"'str' and '{instance_type}'|'{instance_type}' and 'str'"
    )

    # 遍历 [("infer", off), (off, "infer")]
    for left, right in [("infer", off), (off, "infer")]:
        # 使用 pytest.raises 检查是否抛出 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right


# 使用 pytest.mark.parametrize 装饰器为 test_compare_ticks_to_timedeltalike 函数参数化，参数为 tick_classes 列表中的每个类(cls)。
@pytest.mark.parametrize("cls", tick_classes)
# 定义测试函数 test_compare_ticks_to_timedeltalike，测试 ticks 类和时间差类似对象之间的比较行为。
def test_compare_ticks_to_timedeltalike(cls):
    # 创建一个 ticks 类的实例 off，参数为 19
    off = cls(19)

    # 获取 off 的 Pandas 时间差对象
    td = off._as_pd_timedelta

    # 创建一个包含不同时间差对象的列表 others
    others = [td, td.to_timedelta64()]
    # 如果 ticks 类不是 Nano，则添加其 Python 时间差对象到 others 列表
    if cls is not Nano:
        others.append(td.to_pytimedelta())

    # 遍历 others 列表中的每个时间差对象 other
    for other in others:
        # 检查 off 等于 other
        assert off == other
        # 检查 off 不等于 other
        assert not off != other
        # 检查 off 不小于 other
        assert not off < other
        # 检查 off 不大于 other
        assert not off > other
        # 检查 off 小于等于 other
        assert off <= other
        # 检查 off 大于等于 other
        assert off >= other
```