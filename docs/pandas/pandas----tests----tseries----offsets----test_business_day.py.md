# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_business_day.py`

```
"""
Tests for offsets.BDay
"""

# 导入必要的库和模块
from __future__ import annotations  # 允许使用类型提示

from datetime import (  # 导入 datetime 模块中的多个类和函数
    date,
    datetime,
    timedelta,
)

import numpy as np  # 导入 NumPy 库，并使用别名 np
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs.offsets import (  # 导入 pandas 中的时间偏移量类
    ApplyTypeError,
    BDay,
    BMonthEnd,
)

from pandas import (  # 导入 pandas 库中的多个类和模块
    DatetimeIndex,
    Timedelta,
    _testing as tm,
)

from pandas.tests.tseries.offsets.common import (  # 导入 pandas 时间偏移量的测试工具
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries import offsets  # 导入 pandas 时间序列的偏移量

# 定义一个 pytest fixture，返回特定日期时间对象
@pytest.fixture
def dt():
    return datetime(2008, 1, 1)

# 定义一个 pytest fixture，返回 BDay 类的实例
@pytest.fixture
def _offset():
    return BDay

# 定义一个 pytest fixture，返回 BDay 类的实例化对象
@pytest.fixture
def offset(_offset):
    return _offset()

# 定义一个 pytest fixture，返回带参数的 BDay 类的实例化对象
@pytest.fixture
def offset2(_offset):
    return _offset(2)

# 测试类 TestBusinessDay
class TestBusinessDay:
    # 测试方法 test_different_normalize_equals
    def test_different_normalize_equals(self, _offset, offset2):
        # GH#21404 改变 __eq__ 方法，当 `normalize` 不匹配时返回 False
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    # 测试方法 test_repr
    def test_repr(self, offset, offset2):
        assert repr(offset) == "<BusinessDay>"
        assert repr(offset2) == "<2 * BusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    # 测试方法 test_with_offset
    def test_with_offset(self, dt, offset):
        offset = offset + timedelta(hours=2)

        assert (dt + offset) == datetime(2008, 1, 2, 2)

    # 使用 pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(hours=2),
            Timedelta(hours=2).to_pytimedelta(),
            Timedelta(hours=2).to_timedelta64(),
        ],
        ids=lambda x: type(x),
    )
    # 测试方法 test_with_offset_index
    def test_with_offset_index(self, td, dt, offset):
        dti = DatetimeIndex([dt])
        expected = DatetimeIndex([datetime(2008, 1, 2, 2)])

        result = dti + (td + offset)
        tm.assert_index_equal(result, expected)

        result = dti + (offset + td)
        tm.assert_index_equal(result, expected)

    # 测试方法 test_eq
    def test_eq(self, offset2):
        assert offset2 == offset2

    # 测试方法 test_hash
    def test_hash(self, offset2):
        assert hash(offset2) == hash(offset2)

    # 测试方法 test_add_datetime
    def test_add_datetime(self, dt, offset2):
        assert offset2 + dt == datetime(2008, 1, 3)
        assert offset2 + np.datetime64("2008-01-01 00:00:00") == datetime(2008, 1, 3)

    # 测试方法 testRollback1
    def testRollback1(self, dt, _offset):
        assert _offset(10).rollback(dt) == dt

    # 测试方法 testRollback2
    def testRollback2(self, _offset):
        assert _offset(10).rollback(datetime(2008, 1, 5)) == datetime(2008, 1, 4)

    # 测试方法 testRollforward1
    def testRollforward1(self, dt, _offset):
        assert _offset(10).rollforward(dt) == dt

    # 测试方法 testRollforward2
    def testRollforward2(self, _offset):
        assert _offset(10).rollforward(datetime(2008, 1, 5)) == datetime(2008, 1, 7)
    # 定义测试方法，用于测试日期偏移对象的回滚和前滚功能
    def test_roll_date_object(self, offset):
        # 创建一个日期对象，表示2012年9月15日
        dt = date(2012, 9, 15)

        # 测试偏移对象的回滚方法，预期结果是2012年9月14日
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 14)

        # 测试偏移对象的前滚方法，预期结果是2012年9月17日
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 17)

        # 创建一个新的偏移对象（Day对象），测试其回滚方法，预期结果是不改变日期，仍为2012年9月15日
        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        # 测试新偏移对象的前滚方法，预期结果同样是不改变日期，仍为2012年9月15日
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    # 使用pytest的参数化装饰器，为方法test_is_on_offset提供不同的输入和预期输出进行多次测试
    @pytest.mark.parametrize(
        "dt, expected",
        [
            (datetime(2008, 1, 1), True),    # 测试在偏移日期上的情况，预期结果为True
            (datetime(2008, 1, 5), False),   # 测试不在偏移日期上的情况，预期结果为False
        ],
    )
    def test_is_on_offset(self, offset, dt, expected):
        # 调用辅助函数assert_is_on_offset，验证偏移日期是否符合预期
        assert_is_on_offset(offset, dt, expected)

    # 定义应用案例列表，每个案例包含一个偏移量和一个包含日期映射的字典
    apply_cases: list[tuple[int, dict[datetime, datetime]]] = [
        (
            1,  # 偏移量为1
            {
                datetime(2008, 1, 1): datetime(2008, 1, 2),    # 预期2008年1月1日偏移一天后为2008年1月2日
                datetime(2008, 1, 4): datetime(2008, 1, 7),    # 预期2008年1月4日偏移一天后为2008年1月7日
                datetime(2008, 1, 5): datetime(2008, 1, 7),    # 预期2008年1月5日偏移一天后为2008年1月7日
                datetime(2008, 1, 6): datetime(2008, 1, 7),    # 预期2008年1月6日偏移一天后为2008年1月7日
                datetime(2008, 1, 7): datetime(2008, 1, 8),    # 预期2008年1月7日偏移一天后为2008年1月8日
            },
        ),
        (
            2,  # 偏移量为2
            {
                datetime(2008, 1, 1): datetime(2008, 1, 3),    # 预期2008年1月1日偏移两天后为2008年1月3日
                datetime(2008, 1, 4): datetime(2008, 1, 8),    # 预期2008年1月4日偏移两天后为2008年1月8日
                datetime(2008, 1, 5): datetime(2008, 1, 8),    # 预期2008年1月5日偏移两天后为2008年1月8日
                datetime(2008, 1, 6): datetime(2008, 1, 8),    # 预期2008年1月6日偏移两天后为2008年1月8日
                datetime(2008, 1, 7): datetime(2008, 1, 9),    # 预期2008年1月7日偏移两天后为2008年1月9日
            },
        ),
        (
            -1,  # 偏移量为-1
            {
                datetime(2008, 1, 1): datetime(2007, 12, 31),  # 预期2008年1月1日回滚一天后为2007年12月31日
                datetime(2008, 1, 4): datetime(2008, 1, 3),    # 预期2008年1月4日回滚一天后为2008年1月3日
                datetime(2008, 1, 5): datetime(2008, 1, 4),    # 预期2008年1月5日回滚一天后为2008年1月4日
                datetime(2008, 1, 6): datetime(2008, 1, 4),    # 预期2008年1月6日回滚一天后为2008年1月4日
                datetime(2008, 1, 7): datetime(2008, 1, 4),    # 预期2008年1月7日回滚一天后为2008年1月4日
                datetime(2008, 1, 8): datetime(2008, 1, 7),    # 预期2008年1月8日回滚一天后为2008年1月7日
            },
        ),
        (
            -2,  # 偏移量为-2
            {
                datetime(2008, 1, 1): datetime(2007, 12, 28),  # 预期2008年1月1日回滚两天后为2007年12月28日
                datetime(2008, 1, 4): datetime(2008, 1, 2),    # 预期2008年1月4日回滚两天后为2008年1月2日
                datetime(2008, 1, 5): datetime(2008, 1, 3),    # 预期2008年1月5日回滚两天后为2008年1月3日
                datetime(2008, 1, 6): datetime(2008, 1, 3),    # 预期2008年1月6日回滚两天后为2008年1月3日
                datetime(2008, 1, 7): datetime(2008, 1, 3),    # 预期2008年1月7日回滚两天后为2008年1月3日
                datetime(2008, 1, 8): datetime(2008, 1, 4),    # 预期2008年1月8日回滚两天后为2008年1月4日
                datetime(2008, 1, 9): datetime(2008, 1, 7),    # 预期2008年1月9日回滚两天后为2008年1月7日
            },
        ),
        (
            0,  # 偏移量为0
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),    # 预期2008年1月1日偏移零天后仍为2008年1月1日
                datetime(2008, 1, 4): datetime(2008, 1, 4),    # 预期2008年1月4日偏移零天后仍为2008年1月4日
                datetime(2008, 1, 5): datetime(2008, 1, 7),    # 预期2008年1月5日偏移零天后为2008年1月7日
                datetime(2008, 1, 6): datetime(2008, 1, 7),    # 预期2008年1月6日偏移零天后为2008年1月7日
                datetime(2008, 1, 7): datetime(2008, 1, 7),    # 预期
    # 测试函数，用于对给定的偏移量函数进行测试
    def test_apply(self, case, _offset):
        # 解包测试用例数据，n为测试编号，cases为测试用例字典
        n, cases = case
        # 计算偏移量
        offset = _offset(n)
        # 遍历每个基准时间和期望结果，验证偏移量函数的准确性
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    
    # 测试函数，针对大数值进行偏移量测试
    def test_apply_large_n(self, _offset):
        # 创建一个日期时间对象
        dt = datetime(2012, 10, 23)
    
        # 测试偏移10天后的日期
        result = dt + _offset(10)
        assert result == datetime(2012, 11, 6)
    
        # 测试偏移100天后再偏移100天前是否回到原日期
        result = dt + _offset(100) - _offset(100)
        assert result == dt
    
        # 使用偏移量乘以6，计算日期时间对象之前的日期
        off = _offset() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 12, 23)
        assert rs == xp
    
        # 计算指定日期之后的日期时间对象
        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2011, 12, 26)
        assert rs == xp
    
        # 再次使用偏移量乘以10，计算特定日期之后的日期
        off = _offset() * 10
        rs = datetime(2014, 1, 5) + off  # see #5890
        xp = datetime(2014, 1, 17)
        assert rs == xp
    
    # 测试边界情况下的应用
    def test_apply_corner(self, _offset):
        # 如果偏移量函数为工作日偏移量，设置错误消息
        if _offset is BDay:
            msg = "Only know how to combine business day with datetime or timedelta"
        else:
            # 如果偏移量函数为交易日偏移量，设置错误消息
            msg = (
                "Only know how to combine trading day "
                "with datetime, datetime64 or timedelta"
            )
        # 使用pytest检查是否引发了预期的类型错误异常，异常消息需匹配msg
        with pytest.raises(ApplyTypeError, match=msg):
            _offset()._apply(BMonthEnd())
```