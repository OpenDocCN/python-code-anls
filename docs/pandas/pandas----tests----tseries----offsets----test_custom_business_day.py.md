# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_custom_business_day.py`

```
"""
Tests for offsets.CustomBusinessDay / CDay
"""

# 导入必要的库
from datetime import (
    datetime,
    timedelta,
)

import numpy as np  # 导入NumPy库
import pytest  # 导入Pytest库

# 导入需要测试的类和函数
from pandas._libs.tslibs.offsets import CDay

from pandas import (
    _testing as tm,
    read_pickle,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.holiday import USFederalHolidayCalendar


@pytest.fixture
def offset():
    return CDay()


@pytest.fixture
def offset2():
    return CDay(2)


class TestCustomBusinessDay:
    def test_repr(self, offset, offset2):
        # 测试CustomBusinessDay和定制偏移量的字符串表示
        assert repr(offset) == "<CustomBusinessDay>"
        assert repr(offset2) == "<2 * CustomBusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    def test_holidays(self):
        # 测试带有假期的CustomBusinessDay偏移量
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        tday = CDay(holidays=holidays)
        for year in range(2012, 2015):
            dt = datetime(year, 4, 30)
            xp = datetime(year, 5, 2)
            rs = dt + tday
            assert rs == xp

    def test_weekmask(self):
        # 测试带有自定义周掩码的CustomBusinessDay偏移量
        weekmask_saudi = "Sat Sun Mon Tue Wed"  # 星期四和星期五是周末
        weekmask_uae = "1111001"  # 星期五和星期六是周末
        weekmask_egypt = [1, 1, 1, 1, 0, 0, 1]  # 星期五和星期六是周末
        bday_saudi = CDay(weekmask=weekmask_saudi)
        bday_uae = CDay(weekmask=weekmask_uae)
        bday_egypt = CDay(weekmask=weekmask_egypt)
        dt = datetime(2013, 5, 1)
        xp_saudi = datetime(2013, 5, 4)
        xp_uae = datetime(2013, 5, 2)
        xp_egypt = datetime(2013, 5, 2)
        assert xp_saudi == dt + bday_saudi
        assert xp_uae == dt + bday_uae
        assert xp_egypt == dt + bday_egypt
        xp2 = datetime(2013, 5, 5)
        assert xp2 == dt + 2 * bday_saudi
        assert xp2 == dt + 2 * bday_uae
        assert xp2 == dt + 2 * bday_egypt

    def test_weekmask_and_holidays(self):
        # 测试带有自定义周掩码和假期的CustomBusinessDay偏移量
        weekmask_egypt = "Sun Mon Tue Wed Thu"  # 星期五和星期六是周末
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        bday_egypt = CDay(holidays=holidays, weekmask=weekmask_egypt)
        dt = datetime(2013, 4, 30)
        xp_egypt = datetime(2013, 5, 5)
        assert xp_egypt == dt + 2 * bday_egypt

    @pytest.mark.filterwarnings("ignore:Non:pandas.errors.PerformanceWarning")
    def test_calendar(self):
        # 测试使用美国联邦假日日历的CustomBusinessDay偏移量
        calendar = USFederalHolidayCalendar()
        dt = datetime(2014, 1, 17)
        assert_offset_equal(CDay(calendar=calendar), dt, datetime(2014, 1, 21))

    def test_roundtrip_pickle(self, offset, offset2):
        # 测试对象的Pickling往返
        def _check_roundtrip(obj):
            unpickled = tm.round_trip_pickle(obj)
            assert unpickled == obj

        _check_roundtrip(offset)
        _check_roundtrip(offset2)
        _check_roundtrip(offset * 2)
    # 定义一个测试方法，用于测试向后兼容性至版本0.14.1的数据文件
    def test_pickle_compat_0_14_1(self, datapath):
        # 创建一个包含4个日期时间对象的列表，每个对象表示2013年1月1日
        hdays = [datetime(2013, 1, 1) for ele in range(4)]
        # 构建文件路径，指向版本0.14.1的pickle数据文件
        pth = datapath("tseries", "offsets", "data", "cday-0.14.1.pickle")
        # 使用read_pickle函数读取pickle文件，返回其数据对象
        cday0_14_1 = read_pickle(pth)
        # 创建CDay对象，使用指定的节假日列表hdays进行初始化
        cday = CDay(holidays=hdays)
        # 断言当前创建的CDay对象与读取的cday0_14_1对象相等
        assert cday == cday0_14_1
```