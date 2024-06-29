# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_business_hour.py`

```
"""
Tests for offsets.BusinessHour
"""

# 导入未来版本的注释语法，以便在类中使用类型提示
from __future__ import annotations

# 导入 datetime 模块中的 datetime 和 time 类
from datetime import (
    datetime,
    time as dt_time,
)

# 导入 pytest 模块，用于单元测试
import pytest

# 导入 pandas 库中的时间相关类
from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)

# 导入 pandas 库中的偏移量相关类
from pandas._libs.tslibs.offsets import (
    BDay,
    BusinessHour,
    Nano,
)

# 导入 pandas 库中的一些常用函数和类
from pandas import (
    DatetimeIndex,
    _testing as tm,
    date_range,
)

# 导入自定义的断言函数
from pandas.tests.tseries.offsets.common import assert_offset_equal

# 定义一个测试用的 datetime 对象
@pytest.fixture
def dt():
    return datetime(2014, 7, 1, 10, 00)

# 定义一个 BusinessHour 类型的偏移量对象
@pytest.fixture
def _offset():
    return BusinessHour

# 定义多个不同配置参数的 BusinessHour 偏移量对象
@pytest.fixture
def offset1():
    return BusinessHour()

@pytest.fixture
def offset2():
    return BusinessHour(n=3)

@pytest.fixture
def offset3():
    return BusinessHour(n=-1)

@pytest.fixture
def offset4():
    return BusinessHour(n=-4)

@pytest.fixture
def offset5():
    return BusinessHour(start=dt_time(11, 0), end=dt_time(14, 30))

@pytest.fixture
def offset6():
    return BusinessHour(start="20:00", end="05:00")

@pytest.fixture
def offset7():
    return BusinessHour(n=-2, start=dt_time(21, 30), end=dt_time(6, 30))

@pytest.fixture
def offset8():
    return BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"])

@pytest.fixture
def offset9():
    return BusinessHour(n=3, start=["09:00", "22:00"], end=["13:00", "03:00"])

@pytest.fixture
def offset10():
    return BusinessHour(n=-1, start=["23:00", "13:00"], end=["02:00", "17:00"])

# 定义一个测试类 TestBusinessHour
class TestBusinessHour:
    # 参数化测试构造函数的错误情况
    @pytest.mark.parametrize(
        "start,end,match",
        [
            (
                dt_time(11, 0, 5),
                "17:00",
                "time data must be specified only with hour and minute",
            ),
            ("AAA", "17:00", "time data must match '%H:%M' format"),
            ("14:00:05", "17:00", "time data must match '%H:%M' format"),
            ([], "17:00", "Must include at least 1 start time"),
            ("09:00", [], "Must include at least 1 end time"),
            (
                ["09:00", "11:00"],
                "17:00",
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["10:00"],
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["12:00", "20:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
            (
                ["12:00", "20:00"],
                ["09:00", "11:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
        ],
    )
    # 定义测试构造函数的错误情况方法
    def test_constructor_errors(self, start, end, match):
        # 使用 pytest 的断言检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=match):
            BusinessHour(start=start, end=end)
    # 测试不同的偏移量是否相等，使用偏移量工厂函数 `_offset` 来获取偏移量实例
    def test_different_normalize_equals(self, _offset):
        # GH#21404 修改了 __eq__ 方法，在 `normalize` 不匹配时返回 False
        offset = _offset()
        offset2 = _offset(normalize=True)
        # 断言偏移量不相等
        assert offset != offset2

    # 测试对象的字符串表示形式是否符合预期
    def test_repr(
        self,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        # 断言偏移量对象的字符串表示是否正确
        assert repr(offset1) == "<BusinessHour: bh=09:00-17:00>"
        assert repr(offset2) == "<3 * BusinessHours: bh=09:00-17:00>"
        assert repr(offset3) == "<-1 * BusinessHour: bh=09:00-17:00>"
        assert repr(offset4) == "<-4 * BusinessHours: bh=09:00-17:00>"

        assert repr(offset5) == "<BusinessHour: bh=11:00-14:30>"
        assert repr(offset6) == "<BusinessHour: bh=20:00-05:00>"
        assert repr(offset7) == "<-2 * BusinessHours: bh=21:30-06:30>"
        assert repr(offset8) == "<BusinessHour: bh=09:00-12:00,13:00-17:00>"
        assert repr(offset9) == "<3 * BusinessHours: bh=09:00-13:00,22:00-03:00>"
        assert repr(offset10) == "<-1 * BusinessHour: bh=13:00-17:00,23:00-02:00>"

    # 测试偏移量对象与日期时间的加法是否正常工作
    def test_with_offset(self, dt):
        expected = Timestamp("2014-07-01 13:00")

        assert dt + BusinessHour() * 3 == expected
        assert dt + BusinessHour(n=3) == expected

    # 使用参数化测试来验证偏移量对象的相等性
    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_eq_attribute(self, offset_name, request):
        offset = request.getfixturevalue(offset_name)
        # 断言同一偏移量对象与自身相等
        assert offset == offset

    # 使用参数化测试来验证偏移量对象的相等性
    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(start="09:00"), BusinessHour()),
            (
                BusinessHour(start=["23:00", "13:00"], end=["12:00", "17:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_eq(self, offset1, offset2):
        # 断言两个偏移量对象是否相等
        assert offset1 == offset2

    # 使用参数化测试来验证偏移量对象的不相等性
    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(), BusinessHour(-1)),
            (BusinessHour(start="09:00"), BusinessHour(start="09:01")),
            (
                BusinessHour(start="09:00", end="17:00"),
                BusinessHour(start="17:00", end="09:01"),
            ),
            (
                BusinessHour(start=["13:00", "23:00"], end=["18:00", "07:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_neq(self, offset1, offset2):
        # 断言两个偏移量对象是否不相等
        assert offset1 != offset2

    # 使用参数化测试来验证偏移量对象的哈希值
    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_hash(self, offset_name, request):
        offset = request.getfixturevalue(offset_name)
        # 断言偏移量对象的哈希值相等
        assert offset == offset
    def test_add_datetime(
        self,
        dt,
        offset1,
        offset2,
        offset3,
        offset4,
        offset8,
        offset9,
        offset10,
    ):
        # 断言加法操作后的结果是否符合预期的日期时间对象
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 13)
        assert offset3 + dt == datetime(2014, 6, 30, 17)
        assert offset4 + dt == datetime(2014, 6, 30, 14)
        assert offset8 + dt == datetime(2014, 7, 1, 11)
        assert offset9 + dt == datetime(2014, 7, 1, 22)
        assert offset10 + dt == datetime(2014, 7, 1, 1)

    def test_sub(self, dt, offset2, _offset):
        # 初始化 off 变量为 offset2
        off = offset2
        # 设置错误消息
        msg = "Cannot subtract datetime from offset"
        # 使用 pytest 来断言是否会抛出 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            off - dt
        # 断言乘法操作的正确性
        assert 2 * off - off == off

        # 断言减法操作后的结果是否符合预期的日期时间对象
        assert dt - offset2 == dt + _offset(-3)

    def test_multiply_by_zero(self, dt, offset1, offset2):
        # 断言乘以 0 后的日期时间对象是否保持不变
        assert dt - 0 * offset1 == dt
        assert dt + 0 * offset1 == dt
        assert dt - 0 * offset2 == dt
        assert dt + 0 * offset2 == dt

    def testRollback1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        # 断言使用 rollback 方法后的结果是否符合预期的日期时间对象
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt
        assert offset3.rollback(dt) == dt
        assert offset4.rollback(dt) == dt
        assert offset5.rollback(dt) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(dt) == datetime(2014, 7, 1, 5, 0)
        assert offset7.rollback(dt) == datetime(2014, 7, 1, 6, 30)
        assert offset8.rollback(dt) == dt
        assert offset9.rollback(dt) == dt
        assert offset10.rollback(dt) == datetime(2014, 7, 1, 2)

        # 初始化一个特定日期时间对象
        datet = datetime(2014, 7, 1, 0)
        # 断言使用 rollback 方法后的结果是否符合预期的日期时间对象
        assert offset1.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset2.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset3.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset4.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset5.rollback(datet) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(datet) == datet
        assert offset7.rollback(datet) == datet
        assert offset8.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset9.rollback(datet) == datet
        assert offset10.rollback(datet) == datet

        # 断言使用 rollback 方法后的结果是否符合预期的日期时间对象
        assert _offset(5).rollback(dt) == dt

    def testRollback2(self, _offset):
        # 断言使用 rollback 方法后的结果是否符合预期的日期时间对象
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
    ):
        # 断言不同的时间偏移对应的 rollforward 方法能够正确推进到指定的日期时间
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt
        assert offset3.rollforward(dt) == dt
        assert offset4.rollforward(dt) == dt
        assert offset5.rollforward(dt) == datetime(2014, 7, 1, 11, 0)
        assert offset6.rollforward(dt) == datetime(2014, 7, 1, 20, 0)
        assert offset7.rollforward(dt) == datetime(2014, 7, 1, 21, 30)
        assert offset8.rollforward(dt) == dt
        assert offset9.rollforward(dt) == dt
        assert offset10.rollforward(dt) == datetime(2014, 7, 1, 13)

        # 针对另一个特定的日期时间进行类似的断言测试
        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset3.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset4.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset5.rollforward(datet) == datetime(2014, 7, 1, 11)
        assert offset6.rollforward(datet) == datet
        assert offset7.rollforward(datet) == datet
        assert offset8.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset9.rollforward(datet) == datet
        assert offset10.rollforward(datet) == datet

        # 断言特定时间偏移的 rollforward 方法在给定日期时间上的正确行为
        assert _offset(5).rollforward(dt) == dt

    # 测试方法：验证给定的偏移量是否能正确将日期时间推进到预期的值
    def testRollforward2(self, _offset):
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    # 测试方法：验证 BusinessHour 类的 rollback 方法的行为是否正确
    def test_roll_date_object(self):
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        # 验证 BusinessHour 类的 rollforward 方法的行为是否正确
        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    # 测试用例：各种情况下 BusinessHour 类的 normalize 参数对 normalize 方法的影响
    normalize_cases = []
    normalize_cases.append(
        (
            BusinessHour(normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        )
    )
    # 将测试用例添加到 normalize_cases 列表中，每个测试用例包含一个 BusinessHour 实例和一组日期时间映射
    normalize_cases.append(
        (
            BusinessHour(-1, normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 30),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        )
    )

    # 将测试用例添加到 normalize_cases 列表中，每个测试用例包含一个 BusinessHour 实例和一组日期时间映射
    normalize_cases.append(
        (
            BusinessHour(1, normalize=True, start="17:00", end="04:00"),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 2),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        )
    )

    # 使用 pytest.mark.parametrize 将 normalize_cases 中的测试用例参数化，以便作为多个单独测试的输入
    @pytest.mark.parametrize("case", normalize_cases)
    # 定义测试函数 test_normalize，接受一个名为 case 的参数
    def test_normalize(self, case):
        # 从 case 中解包出 BusinessHour 实例和日期时间映射
        offset, cases = case
        # 遍历日期时间映射中的每个条目
        for dt, expected in cases.items():
            # 断言 BusinessHour 实例对日期时间进行处理后的结果与期望的结果一致
            assert offset._apply(dt) == expected

    # 将测试用例添加到 on_offset_cases 列表中，每个测试用例包含一个 BusinessHour 实例和一组日期时间是否在工作时间内的映射
    on_offset_cases.append(
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )

    # 将测试用例添加到 on_offset_cases 列表中，每个测试用例包含一个 BusinessHour 实例和一组日期时间是否在工作时间内的映射
    on_offset_cases.append(
        (
            BusinessHour(start="10:00", end="15:00"),
            {
                datetime(2014, 7, 1, 9): False,
                datetime(2014, 7, 1, 10): True,
                datetime(2014, 7, 1, 15): True,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )
    # 将测试用例添加到 on_offset_cases 列表中，每个元素包含 BusinessHour 对象和对应的测试数据字典
    on_offset_cases.append(
        (
            BusinessHour(start="19:00", end="05:00"),  # 创建 BusinessHour 对象，表示工作时间从晚上7点到次日凌晨5点
            {
                datetime(2014, 7, 1, 9, 0): False,    # 测试数据：9点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 10, 0): False,   # 测试数据：10点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 15): False,      # 测试数据：15点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 15, 1): False,   # 测试数据：15点1分，不在工作时间内，期望返回 False
                datetime(2014, 7, 5, 12, 0): False,   # 测试数据：7月5日12点，不在工作时间内，期望返回 False
                datetime(2014, 7, 6, 12, 0): False,   # 测试数据：7月6日12点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 19, 0): True,    # 测试数据：7月1日19点，处于工作时间内，期望返回 True
                datetime(2014, 7, 2, 0, 0): True,     # 测试数据：7月2日0点，处于工作时间内，期望返回 True
                datetime(2014, 7, 4, 23): True,       # 测试数据：7月4日23点，处于工作时间内，期望返回 True
                datetime(2014, 7, 5, 1): True,        # 测试数据：7月5日1点，处于工作时间内，期望返回 True
                datetime(2014, 7, 5, 5, 0): True,     # 测试数据：7月5日5点，处于工作时间内，期望返回 True
                datetime(2014, 7, 6, 23, 0): False,   # 测试数据：7月6日23点，不在工作时间内，期望返回 False
                datetime(2014, 7, 7, 3, 0): False,    # 测试数据：7月7日3点，不在工作时间内，期望返回 False
            },
        )
    )

    # 将测试用例添加到 on_offset_cases 列表中，每个元素包含 BusinessHour 对象和对应的测试数据字典
    on_offset_cases.append(
        (
            BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"]),  # 创建 BusinessHour 对象，表示工作时间从9点到12点，以及13点到17点
            {
                datetime(2014, 7, 1, 9): True,         # 测试数据：7月1日9点，处于工作时间内，期望返回 True
                datetime(2014, 7, 1, 8, 59): False,    # 测试数据：7月1日8点59分，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 8): False,        # 测试数据：7月1日8点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 17): True,        # 测试数据：7月1日17点，处于工作时间内，期望返回 True
                datetime(2014, 7, 1, 17, 1): False,    # 测试数据：7月1日17点1分，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 18): False,       # 测试数据：7月1日18点，不在工作时间内，期望返回 False
                datetime(2014, 7, 5, 9): False,        # 测试数据：7月5日9点，不在工作时间内，期望返回 False
                datetime(2014, 7, 6, 12): False,       # 测试数据：7月6日12点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 12, 30): False,   # 测试数据：7月1日12点30分，不在工作时间内，期望返回 False
            },
        )
    )

    # 将测试用例添加到 on_offset_cases 列表中，每个元素包含 BusinessHour 对象和对应的测试数据字典
    on_offset_cases.append(
        (
            BusinessHour(start=["19:00", "23:00"], end=["21:00", "05:00"]),  # 创建 BusinessHour 对象，表示工作时间从晚上7点到9点，以及晚上11点到次日凌晨5点
            {
                datetime(2014, 7, 1, 9, 0): False,    # 测试数据：9点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 10, 0): False,   # 测试数据：10点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 15): False,      # 测试数据：15点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 15, 1): False,   # 测试数据：15点1分，不在工作时间内，期望返回 False
                datetime(2014, 7, 5, 12, 0): False,   # 测试数据：7月5日12点，不在工作时间内，期望返回 False
                datetime(2014, 7, 6, 12, 0): False,   # 测试数据：7月6日12点，不在工作时间内，期望返回 False
                datetime(2014, 7, 1, 19, 0): True,    # 测试数据：7月1日19点，处于工作时间内，期望返回 True
                datetime(2014, 7, 2, 0, 0): True,     # 测试数据：7月2日0点，处于工作时间内，期望返回 True
                datetime(2014, 7, 4, 23): True,       # 测试数据：7月4日23点，处于工作时间内，期望返回 True
                datetime(2014, 7, 5, 1): True,        # 测试数据：7月5日1点，处于工作时间内，期望返回 True
                datetime(2014, 7, 5, 5, 0): True,     # 测试数据：7月5日5点，处于工作时间内，期望返回 True
                datetime(2014, 7, 6, 23, 0): False,   # 测试数据：7月6日23点，不在工作时间内，期望返回 False
                datetime(2014, 7, 7, 3, 0): False,    # 测试数据：7月7日3点，不在工作时间内，期望返回 False
                datetime(2014, 7, 4, 22): False,      # 测试数据：7月4日22点，不在工作时间内，期望返回 False
            },
        )
    )

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, cases = case
        for dt, expected in cases.items():
            assert offset.is_on_offset(dt) == expected
    # 定义一个测试函数，用于测试在大量数据情况下应用偏移量
    def test_apply_large_n(self, case):
        # 解包测试用例元组，包括偏移量和测试数据字典
        offset, cases = case
        # 遍历测试数据字典，依次比较每个基准时间和预期结果是否相等
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 定义测试函数，测试在纳秒精度下的时间操作
    def test_apply_nanoseconds(self):
        # 定义多组测试用例，每组包含一个 BusinessHour 对象和一个包含时间戳对的字典
        tests = [
            (
                BusinessHour(),
                {
                    # 对基准时间添加 5 纳秒，期望结果是在后续一个小时的时间戳
                    Timestamp("2014-07-04 15:00") + Nano(5): Timestamp(
                        "2014-07-04 16:00"
                    )
                    + Nano(5),
                    # 对基准时间添加 5 纳秒，期望结果是在后续一个工作日的时间戳
                    Timestamp("2014-07-04 16:00") + Nano(5): Timestamp(
                        "2014-07-07 09:00"
                    )
                    + Nano(5),
                    # 对基准时间减去 5 纳秒，期望结果是在后续一个小时的时间戳
                    Timestamp("2014-07-04 16:00") - Nano(5): Timestamp(
                        "2014-07-04 17:00"
                    )
                    - Nano(5),
                },
            ),
            (
                BusinessHour(-1),
                {
                    # 对基准时间添加 5 纳秒，期望结果是在前一小时的时间戳
                    Timestamp("2014-07-04 15:00") + Nano(5): Timestamp(
                        "2014-07-04 14:00"
                    )
                    + Nano(5),
                    # 对基准时间添加 5 纳秒，期望结果是在前一工作日的时间戳
                    Timestamp("2014-07-04 10:00") + Nano(5): Timestamp(
                        "2014-07-04 09:00"
                    )
                    + Nano(5),
                    # 对基准时间减去 5 纳秒，期望结果是在前一工作日的时间戳
                    Timestamp("2014-07-04 10:00") - Nano(5): Timestamp(
                        "2014-07-03 17:00"
                    )
                    - Nano(5),
                },
            ),
        ]

        # 遍历测试用例列表
        for offset, cases in tests:
            # 遍历每组测试用例中的时间戳对，依次比较偏移量后的结果是否符合预期
            for base, expected in cases.items():
                assert_offset_equal(offset, base, expected)

    # 使用 pytest 的参数化标记，对不同的时间单位进行测试
    @pytest.mark.parametrize("td_unit", ["s", "ms", "us", "ns"])
    def test_bday_ignores_timedeltas(self, unit, td_unit):
        # GH#55608
        # 生成一个日期范围，频率为每12小时，使用给定的单位
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit=unit)
        # 创建一个时间增量对象，单位为 td_unit
        td = Timedelta(3, unit="h").as_unit(td_unit)
        # 创建一个工作日偏移对象，偏移量为 td
        off = BDay(offset=td)
        # 将日期范围 idx 加上工作日偏移 off
        t1 = idx + off

        # 获取 td.unit 和 idx.unit 中的最精细单位
        exp_unit = tm.get_finest_unit(td.unit, idx.unit)

        # 生成预期的日期时间索引对象，频率为 None
        expected = DatetimeIndex(
            [
                "2010-02-02 03:00:00",
                "2010-02-02 15:00:00",
                "2010-02-03 03:00:00",
                "2010-02-03 15:00:00",
                "2010-02-04 03:00:00",
                "2010-02-04 15:00:00",
                "2010-02-05 03:00:00",
                "2010-02-05 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-09 03:00:00",
                "2010-02-09 15:00:00",
                "2010-02-10 03:00:00",
                "2010-02-10 15:00:00",
                "2010-02-11 03:00:00",
            ],
            freq=None,
        ).as_unit(exp_unit)
        # 比较 t1 和 expected 是否相等
        tm.assert_index_equal(t1, expected)

        # TODO(GH#55564): as_unit will be unnecessary
        # 逐点比较，使用工作日偏移 off
        pointwise = DatetimeIndex([x + off for x in idx]).as_unit(exp_unit)
        # 比较 pointwise 和 expected 是否相等
        tm.assert_index_equal(pointwise, expected)

    def test_add_bday_offset_nanos(self):
        # GH#55608
        # 生成一个日期范围，频率为每12小时，单位为纳秒
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit="ns")
        # 创建一个工作日偏移对象，偏移量为 Timedelta(3, unit="ns")
        off = BDay(offset=Timedelta(3, unit="ns"))

        # 将日期范围 idx 加上工作日偏移 off
        result = idx + off
        # 生成预期的日期时间索引对象，每个元素加上工作日偏移 off
        expected = DatetimeIndex([x + off for x in idx])
        # 比较 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
# 定义测试类 TestOpeningTimes，用于测试开放时间的计算
class TestOpeningTimes:
    # 使用 pytest 的 parametrize 装饰器，传入开放时间案例参数化列表
    @pytest.mark.parametrize("case", opening_time_cases)
    # 定义测试方法 test_opening_time，参数为 case
    def test_opening_time(self, case):
        # 解构 case 元组，_offsets 为偏移量列表，cases 为日期时间和预期结果的字典
        _offsets, cases = case
        # 遍历 _offsets 列表中的偏移量
        for offset in _offsets:
            # 遍历 cases 字典中的日期时间 dt 及其对应的预期结果 (exp_next, exp_prev)
            for dt, (exp_next, exp_prev) in cases.items():
                # 断言计算的下一个开放时间与预期的下一个开放时间相等
                assert offset._next_opening_time(dt) == exp_next
                # 断言计算的上一个开放时间与预期的上一个开放时间相等
                assert offset._prev_opening_time(dt) == exp_prev
```