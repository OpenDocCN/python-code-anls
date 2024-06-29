# `D:\src\scipysrc\pandas\pandas\tests\scalar\period\test_asfreq.py`

```
# 导入pytest模块，用于测试和断言
import pytest

# 导入pandas错误信息
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime

# 导入pandas中的一些重要对象和函数
from pandas import (
    Period,          # 日期周期对象
    Timestamp,       # 时间戳对象
    offsets,         # 时间偏移量
)

# 导入pandas测试工具
import pandas._testing as tm

# 定义全局变量，表示工作日频率的警告信息
bday_msg = "Period with BDay freq is deprecated"

# 定义频率转换测试类
class TestFreqConversion:
    """Test frequency conversion of date objects"""

    # 标记此测试方法忽略特定警告信息
    @pytest.mark.filterwarnings("ignore:Period with BDay:FutureWarning")
    # 参数化测试，测试各种日期频率
    @pytest.mark.parametrize("freq", ["Y", "Q", "M", "W", "B", "D"])
    def test_asfreq_near_zero(self, freq):
        # 创建一个指定频率的Period对象，日期为0001-01-01
        per = Period("0001-01-01", freq=freq)
        # 获取日期周期的年份、小时和天数，并组成元组
        tup1 = (per.year, per.hour, per.day)

        # 计算前一个日期周期
        prev = per - 1
        # 断言前一个日期周期的ordinal属性等于当前周期的ordinal属性减一
        assert prev.ordinal == per.ordinal - 1
        # 获取前一个日期周期的年份、月份和天数，并组成元组
        tup2 = (prev.year, prev.month, prev.day)
        # 断言前一个日期元组小于当前日期元组
        assert tup2 < tup1

    # 测试按周频率的近零处理
    def test_asfreq_near_zero_weekly(self):
        # 创建两个日期周期对象，分别为0001-01-07和0001-01-01
        per1 = Period("0001-01-01", "D") + 6
        per2 = Period("0001-01-01", "D") - 6
        # 将日期周期转换为周频率的日期周期
        week1 = per1.asfreq("W")
        week2 = per2.asfreq("W")
        # 断言两个周频率的日期周期不相等
        assert week1 != week2
        # 断言第一个周频率的日期周期按日频率（工作日结束）大于等于per1
        assert week1.asfreq("D", "E") >= per1
        # 断言第二个周频率的日期周期按日频率（工作日开始）小于等于per2
        assert week2.asfreq("D", "S") <= per2

    # 测试to_timestamp方法超出边界情况
    def test_to_timestamp_out_of_bounds(self):
        # 创建一个工作日频率的Period对象，日期为0001-01-01
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("0001-01-01", freq="B")
        # 设置异常消息
        msg = "Out of bounds nanosecond timestamp"
        # 断言调用per对象的to_timestamp方法抛出OutOfBoundsDatetime异常，并匹配异常消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=bday_msg):
                per.to_timestamp()

    # 测试频率转换的边界情况
    def test_asfreq_corner(self):
        # 创建一个年频率的Period对象，年份为2007
        val = Period(freq="Y", year=2007)
        # 将该Period对象转换为5分钟频率的Period对象
        result1 = val.asfreq("5min")
        # 将该Period对象转换为分钟频率的Period对象
        result2 = val.asfreq("min")
        # 创建预期的Period对象，表示2007年12月31日23时59分
        expected = Period("2007-12-31 23:59", freq="min")
        # 断言result1的ordinal属性与预期的ordinal属性相等
        assert result1.ordinal == expected.ordinal
        # 断言result1的频率字符串为"5min"
        assert result1.freqstr == "5min"
        # 断言result2的ordinal属性与预期的ordinal属性相等
        assert result2.ordinal == expected.ordinal
        # 断言result2的频率字符串为"min"
        assert result2.freqstr == "min"
    def test_conv_monthly(self):
        # 定义测试函数 test_conv_monthly，用于测试频率转换，从月度频率开始

        # 创建月度周期对象，起始日期为2007年1月
        ival_M = Period(freq="M", year=2007, month=1)
        # 创建月度周期对象，截止日期为2007年12月
        ival_M_end_of_year = Period(freq="M", year=2007, month=12)
        # 创建月度周期对象，截止日期为2007年第1季度末
        ival_M_end_of_quarter = Period(freq="M", year=2007, month=3)
        # 创建年度周期对象，对应的月度周期起始日期为2007年
        ival_M_to_A = Period(freq="Y", year=2007)
        # 创建季度周期对象，对应的月度周期为2007年第1季度
        ival_M_to_Q = Period(freq="Q", year=2007, quarter=1)
        # 创建周周期对象，对应的月度周期起始日期为2007年1月1日
        ival_M_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        # 创建周周期对象，对应的月度周期截止日期为2007年1月31日
        ival_M_to_W_end = Period(freq="W", year=2007, month=1, day=31)
        
        # 使用 FutureWarning 断言检测，匹配 bday_msg 提示信息
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建工作日周期对象，对应的月度周期起始日期为2007年1月1日
            ival_M_to_B_start = Period(freq="B", year=2007, month=1, day=1)
            # 创建工作日周期对象，对应的月度周期截止日期为2007年1月31日
            ival_M_to_B_end = Period(freq="B", year=2007, month=1, day=31)
        
        # 创建日周期对象，对应的月度周期起始日期为2007年1月1日
        ival_M_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        # 创建日周期对象，对应的月度周期截止日期为2007年1月31日
        ival_M_to_D_end = Period(freq="D", year=2007, month=1, day=31)
        
        # 创建小时周期对象，对应的月度周期起始日期为2007年1月1日0时
        ival_M_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        # 创建小时周期对象，对应的月度周期截止日期为2007年1月31日23时
        ival_M_to_H_end = Period(freq="h", year=2007, month=1, day=31, hour=23)
        
        # 创建分钟周期对象，对应的月度周期起始日期为2007年1月1日0时0分
        ival_M_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        # 创建分钟周期对象，对应的月度周期截止日期为2007年1月31日23时59分
        ival_M_to_T_end = Period(
            freq="Min", year=2007, month=1, day=31, hour=23, minute=59
        )
        
        # 创建秒周期对象，对应的月度周期起始日期为2007年1月1日0时0分0秒
        ival_M_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        # 创建秒周期对象，对应的月度周期截止日期为2007年1月31日23时59分59秒
        ival_M_to_S_end = Period(
            freq="s", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )

        # 断言语句，验证月度周期按年度频率转换后与预期的年度周期对象相同
        assert ival_M.asfreq("Y") == ival_M_to_A
        # 断言语句，验证月度周期按年度频率转换后与预期的年度周期对象相同
        assert ival_M_end_of_year.asfreq("Y") == ival_M_to_A
        # 断言语句，验证月度周期按季度频率转换后与预期的季度周期对象相同
        assert ival_M.asfreq("Q") == ival_M_to_Q
        # 断言语句，验证月度周期按季度频率转换后与预期的季度周期对象相同
        assert ival_M_end_of_quarter.asfreq("Q") == ival_M_to_Q

        # 断言语句，验证月度周期按周频率转换后与预期的周周期起始对象相同
        assert ival_M.asfreq("W", "s") == ival_M_to_W_start
        # 断言语句，验证月度周期按周频率转换后与预期的周周期截止对象相同
        assert ival_M.asfreq("W", "E") == ival_M_to_W_end
        
        # 使用 FutureWarning 断言检测，匹配 bday_msg 提示信息
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 断言语句，验证月度周期按工作日频率转换后与预期的工作日周期起始对象相同
            assert ival_M.asfreq("B", "s") == ival_M_to_B_start
            # 断言语句，验证月度周期按工作日频率转换后与预期的工作日周期截止对象相同
            assert ival_M.asfreq("B", "E") == ival_M_to_B_end
        
        # 断言语句，验证月度周期按日频率转换后与预期的日周期起始对象相同
        assert ival_M.asfreq("D", "s") == ival_M_to_D_start
        # 断言语句，验证月度周期按日频率转换后与预期的日周期截止对象相同
        assert ival_M.asfreq("D", "E") == ival_M_to_D_end
        # 断言语句，验证月度周期按小时频率转换后与预期的小时周期起始对象相同
        assert ival_M.asfreq("h", "s") == ival_M_to_H_start
        # 断言语句，验证月度周期按小时频率转换后与预期的小时周期截止对象相同
        assert ival_M.asfreq("h", "E") == ival_M_to_H_end
        # 断言语句，验证月度周期按分钟频率转换后与预期的分钟周期起始对象相同
        assert ival_M.asfreq("Min", "s") == ival_M_to_T_start
        # 断言语句，验证月度周期按分钟频率转换后与预期的分钟周期截止对象相同
        assert ival_M.asfreq("Min", "E") == ival_M_to_T_end
        # 断言语句，验证月度周期按秒频率转换后与预期的秒周期起始对象相同
        assert ival_M.asfreq("s", "s") == ival_M_to_S_start
        # 断言语句，验证月度周期按秒频率转换后与预期的秒周期截止对象相同
        assert ival_M.asfreq("s", "E") == ival_M_to_S_end

        # 断言语句，验证月度周期按月度频率转换后与原始的月度周期对象相同
        assert ival_M.asfreq("M") == ival_M
    # 定义一个测试方法，用于测试从每周频率转换的情况
    def test_conv_weekly_legacy(self):
        # 设置无效频率错误消息
        msg = INVALID_FREQ_ERR_MSG
        # 测试在创建周周期时，使用无效的频率“WK”是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=1)

        # 测试在创建以周为单位，以星期六结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-SAT", year=2007, month=1, day=6)
        # 测试在创建以周为单位，以星期五结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-FRI", year=2007, month=1, day=5)
        # 测试在创建以周为单位，以星期四结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-THU", year=2007, month=1, day=4)
        # 测试在创建以周为单位，以星期三结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-WED", year=2007, month=1, day=3)
        # 测试在创建以周为单位，以星期二结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-TUE", year=2007, month=1, day=2)
        # 测试在创建以周为单位，以星期一结束的周期时，是否会引发值错误异常
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-MON", year=2007, month=1, day=1)


这段代码是一个测试方法，用于验证在不同的周频率设置下，是否会正确地抛出值错误异常。每个 `with pytest.raises` 语句测试了不同的周结束日设置，例如以星期六、星期五等结束。
    # 定义测试函数 `test_conv_business`，用于测试业务频率转换

    # 进入一个上下文，测试将来会产生警告，警告信息需要匹配 bday_msg
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        # 创建一个 Business 频率的 Period 对象，表示从 2007 年 1 月 1 日开始的 Business 频率时间段
        ival_B = Period(freq="B", year=2007, month=1, day=1)
        # 创建一个表示 2007 年 12 月 31 日结束的 Business 频率时间段的 Period 对象
        ival_B_end_of_year = Period(freq="B", year=2007, month=12, day=31)
        # 创建一个表示 2007 年 3 月 30 日结束的 Business 频率时间段的 Period 对象
        ival_B_end_of_quarter = Period(freq="B", year=2007, month=3, day=30)
        # 创建一个表示 2007 年 1 月 31 日结束的 Business 频率时间段的 Period 对象
        ival_B_end_of_month = Period(freq="B", year=2007, month=1, day=31)
        # 创建一个表示 2007 年 1 月 5 日结束的 Business 频率时间段的 Period 对象
        ival_B_end_of_week = Period(freq="B", year=2007, month=1, day=5)

    # 创建不同频率转换的 Period 对象
    ival_B_to_A = Period(freq="Y", year=2007)
    ival_B_to_Q = Period(freq="Q", year=2007, quarter=1)
    ival_B_to_M = Period(freq="M", year=2007, month=1)
    ival_B_to_W = Period(freq="W", year=2007, month=1, day=7)
    ival_B_to_D = Period(freq="D", year=2007, month=1, day=1)
    ival_B_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
    ival_B_to_H_end = Period(freq="h", year=2007, month=1, day=1, hour=23)
    ival_B_to_T_start = Period(
        freq="Min", year=2007, month=1, day=1, hour=0, minute=0
    )
    ival_B_to_T_end = Period(
        freq="Min", year=2007, month=1, day=1, hour=23, minute=59
    )
    ival_B_to_S_start = Period(
        freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
    )
    ival_B_to_S_end = Period(
        freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
    )

    # 断言不同频率转换的结果是否符合预期
    assert ival_B.asfreq("Y") == ival_B_to_A
    assert ival_B_end_of_year.asfreq("Y") == ival_B_to_A
    assert ival_B.asfreq("Q") == ival_B_to_Q
    assert ival_B_end_of_quarter.asfreq("Q") == ival_B_to_Q
    assert ival_B.asfreq("M") == ival_B_to_M
    assert ival_B_end_of_month.asfreq("M") == ival_B_to_M
    assert ival_B.asfreq("W") == ival_B_to_W
    assert ival_B_end_of_week.asfreq("W") == ival_B_to_W

    assert ival_B.asfreq("D") == ival_B_to_D

    assert ival_B.asfreq("h", "s") == ival_B_to_H_start
    assert ival_B.asfreq("h", "E") == ival_B_to_H_end
    assert ival_B.asfreq("Min", "s") == ival_B_to_T_start
    assert ival_B.asfreq("Min", "E") == ival_B_to_T_end
    assert ival_B.asfreq("s", "s") == ival_B_to_S_start
    assert ival_B.asfreq("s", "E") == ival_B_to_S_end

    # 再次进入一个上下文，测试将来会产生警告，警告信息需要匹配 bday_msg
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        # 断言 Business 频率转换为自身的结果是否符合预期
        assert ival_B.asfreq("B") == ival_B
    def test_conv_hourly(self):
        # 定义测试函数，用于测试从小时频率转换

        ival_H = Period(freq="h", year=2007, month=1, day=1, hour=0)
        # 创建一个小时频率的时间段，从2007年1月1日00:00开始

        ival_H_end_of_year = Period(freq="h", year=2007, month=12, day=31, hour=23)
        # 创建一个小时频率的时间段，表示2007年12月31日23:00结束

        ival_H_end_of_quarter = Period(freq="h", year=2007, month=3, day=31, hour=23)
        # 创建一个小时频率的时间段，表示2007年3月31日23:00结束

        ival_H_end_of_month = Period(freq="h", year=2007, month=1, day=31, hour=23)
        # 创建一个小时频率的时间段，表示2007年1月31日23:00结束

        ival_H_end_of_week = Period(freq="h", year=2007, month=1, day=7, hour=23)
        # 创建一个小时频率的时间段，表示2007年1月7日23:00结束

        ival_H_end_of_day = Period(freq="h", year=2007, month=1, day=1, hour=23)
        # 创建一个小时频率的时间段，表示2007年1月1日23:00结束

        ival_H_end_of_bus = Period(freq="h", year=2007, month=1, day=1, hour=23)
        # 创建一个小时频率的时间段，表示2007年1月1日23:00结束

        ival_H_to_A = Period(freq="Y", year=2007)
        # 以年为频率将小时频率时间段转换为年

        ival_H_to_Q = Period(freq="Q", year=2007, quarter=1)
        # 以季度为频率将小时频率时间段转换为季度

        ival_H_to_M = Period(freq="M", year=2007, month=1)
        # 以月为频率将小时频率时间段转换为月

        ival_H_to_W = Period(freq="W", year=2007, month=1, day=7)
        # 以周为频率将小时频率时间段转换为周

        ival_H_to_D = Period(freq="D", year=2007, month=1, day=1)
        # 以天为频率将小时频率时间段转换为天

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_H_to_B = Period(freq="B", year=2007, month=1, day=1)
        # 以工作日为频率将小时频率时间段转换为工作日，并产生FutureWarning警告，匹配bday_msg

        ival_H_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        # 以分钟为频率将小时频率时间段转换为分钟，并表示从2007年1月1日00:00开始

        ival_H_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=59
        )
        # 以分钟为频率将小时频率时间段转换为分钟，并表示到2007年1月1日00:59结束

        ival_H_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        # 以秒为频率将小时频率时间段转换为秒，并表示从2007年1月1日00:00:00开始

        ival_H_to_S_end = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )
        # 以秒为频率将小时频率时间段转换为秒，并表示到2007年1月1日00:59:59结束

        assert ival_H.asfreq("Y") == ival_H_to_A
        # 断言将小时频率时间段转换为年的结果是否与预期相同

        assert ival_H_end_of_year.asfreq("Y") == ival_H_to_A
        # 断言将小时频率时间段末尾转换为年的结果是否与预期相同

        assert ival_H.asfreq("Q") == ival_H_to_Q
        # 断言将小时频率时间段转换为季度的结果是否与预期相同

        assert ival_H_end_of_quarter.asfreq("Q") == ival_H_to_Q
        # 断言将小时频率时间段末尾转换为季度的结果是否与预期相同

        assert ival_H.asfreq("M") == ival_H_to_M
        # 断言将小时频率时间段转换为月的结果是否与预期相同

        assert ival_H_end_of_month.asfreq("M") == ival_H_to_M
        # 断言将小时频率时间段末尾转换为月的结果是否与预期相同

        assert ival_H.asfreq("W") == ival_H_to_W
        # 断言将小时频率时间段转换为周的结果是否与预期相同

        assert ival_H_end_of_week.asfreq("W") == ival_H_to_W
        # 断言将小时频率时间段末尾转换为周的结果是否与预期相同

        assert ival_H.asfreq("D") == ival_H_to_D
        # 断言将小时频率时间段转换为天的结果是否与预期相同

        assert ival_H_end_of_day.asfreq("D") == ival_H_to_D
        # 断言将小时频率时间段末尾转换为天的结果是否与预期相同

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_H.asfreq("B") == ival_H_to_B
            # 断言将小时频率时间段转换为工作日的结果是否与预期相同，并产生FutureWarning警告，匹配bday_msg

            assert ival_H_end_of_bus.asfreq("B") == ival_H_to_B
            # 断言将小时频率时间段末尾转换为工作日的结果是否与预期相同，并产生FutureWarning警告，匹配bday_msg

        assert ival_H.asfreq("Min", "s") == ival_H_to_T_start
        # 断言将小时频率时间段转换为分钟的结果是否与预期相同，使用起始时间戳标签"s"

        assert ival_H.asfreq("Min", "E") == ival_H_to_T_end
        # 断言将小时频率时间段转换为分钟的结果是否与预期相同，使用结束时间戳标签"E"

        assert ival_H.asfreq("s", "s") == ival_H_to_S_start
        # 断言将小时频率时间段转换为秒的结果是否与预期相同，使用起始时间戳标签"s"

        assert ival_H.asfreq("s", "E") == ival_H_to_S_end
        # 断言将小时频率时间段转换为秒的结果是否与预期相同，使用结束时间戳标签"E"

        assert ival_H.asfreq("h") == ival_H
        # 断言将小时频率时间段转换为小时的结果是否与预期相同
    # 定义测试函数 test_conv_minutely，用于测试频率转换，这里是从分钟频率开始测试

    # 创建 Period 对象，表示从 2007 年 1 月 1 日 00:00 开始的分钟频率
    ival_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
    # 创建 Period 对象，表示 2007 年 12 月 31 日 23:59 结束的分钟频率
    ival_T_end_of_year = Period(
        freq="Min", year=2007, month=12, day=31, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 3 月 31 日 23:59 结束的分钟频率
    ival_T_end_of_quarter = Period(
        freq="Min", year=2007, month=3, day=31, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 1 月 31 日 23:59 结束的分钟频率
    ival_T_end_of_month = Period(
        freq="Min", year=2007, month=1, day=31, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 1 月 7 日 23:59 结束的分钟频率
    ival_T_end_of_week = Period(
        freq="Min", year=2007, month=1, day=7, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 1 月 1 日 23:59 结束的分钟频率
    ival_T_end_of_day = Period(
        freq="Min", year=2007, month=1, day=1, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 1 月 1 日 23:59 结束的分钟频率
    ival_T_end_of_bus = Period(
        freq="Min", year=2007, month=1, day=1, hour=23, minute=59
    )
    # 创建 Period 对象，表示 2007 年 1 月 1 日 00:59 结束的分钟频率
    ival_T_end_of_hour = Period(
        freq="Min", year=2007, month=1, day=1, hour=0, minute=59
    )

    # 创建 Period 对象，表示从 2007 年开始的年度频率
    ival_T_to_A = Period(freq="Y", year=2007)
    # 创建 Period 对象，表示 2007 年第一季度的频率
    ival_T_to_Q = Period(freq="Q", year=2007, quarter=1)
    # 创建 Period 对象，表示 2007 年 1 月的月度频率
    ival_T_to_M = Period(freq="M", year=2007, month=1)
    # 创建 Period 对象，表示 2007 年 1 月 7 日的周度频率
    ival_T_to_W = Period(freq="W", year=2007, month=1, day=7)
    # 创建 Period 对象，表示 2007 年 1 月 1 日的日度频率
    ival_T_to_D = Period(freq="D", year=2007, month=1, day=1)

    # 使用 with 语句检查是否产生 FutureWarning，匹配 bday_msg 提示信息
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        # 创建 Period 对象，表示 2007 年 1 月 1 日的工作日频率
        ival_T_to_B = Period(freq="B", year=2007, month=1, day=1)
    # 创建 Period 对象，表示 2007 年 1 月 1 日 00:00 的小时频率
    ival_T_to_H = Period(freq="h", year=2007, month=1, day=1, hour=0)

    # 创建 Period 对象，表示从 2007 年 1 月 1 日 00:00:00 开始的秒频率
    ival_T_to_S_start = Period(
        freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
    )
    # 创建 Period 对象，表示从 2007 年 1 月 1 日 00:00:59 结束的秒频率
    ival_T_to_S_end = Period(
        freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=59
    )

    # 断言语句，检查 Period 对象的频率转换结果是否符合预期
    assert ival_T.asfreq("Y") == ival_T_to_A
    assert ival_T_end_of_year.asfreq("Y") == ival_T_to_A
    assert ival_T.asfreq("Q") == ival_T_to_Q
    assert ival_T_end_of_quarter.asfreq("Q") == ival_T_to_Q
    assert ival_T.asfreq("M") == ival_T_to_M
    assert ival_T_end_of_month.asfreq("M") == ival_T_to_M
    assert ival_T.asfreq("W") == ival_T_to_W
    assert ival_T_end_of_week.asfreq("W") == ival_T_to_W
    assert ival_T.asfreq("D") == ival_T_to_D
    assert ival_T_end_of_day.asfreq("D") == ival_T_to_D
    # 使用 with 语句检查是否产生 FutureWarning，匹配 bday_msg 提示信息
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_T.asfreq("B") == ival_T_to_B
        assert ival_T_end_of_bus.asfreq("B") == ival_T_to_B
    assert ival_T.asfreq("h") == ival_T_to_H
    assert ival_T_end_of_hour.asfreq("h") == ival_T_to_H

    # 断言语句，检查分钟频率转换到秒频率的结果是否符合预期
    assert ival_T.asfreq("s", "s") == ival_T_to_S_start
    assert ival_T.asfreq("s", "E") == ival_T_to_S_end

    # 断言语句，检查分钟频率转换到自身的结果是否符合预期
    assert ival_T.asfreq("Min") == ival_T
    def test_conv_secondly(self):
        # 定义频率转换测试函数：从秒频率开始

        # 创建一个秒级时间周期对象，起始时间为2007年1月1日 00:00:00
        ival_S = Period(freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0)
        # 创建一个秒级时间周期对象，结束时间为2007年12月31日 23:59:59
        ival_S_end_of_year = Period(
            freq="s", year=2007, month=12, day=31, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年3月31日 23:59:59（季度末）
        ival_S_end_of_quarter = Period(
            freq="s", year=2007, month=3, day=31, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月31日 23:59:59（月末）
        ival_S_end_of_month = Period(
            freq="s", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月7日 23:59:59（周末）
        ival_S_end_of_week = Period(
            freq="s", year=2007, month=1, day=7, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月1日 23:59:59（日末）
        ival_S_end_of_day = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月1日 23:59:59（工作日末）
        ival_S_end_of_bus = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月1日 00:59:59（小时末）
        ival_S_end_of_hour = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )
        # 创建一个秒级时间周期对象，结束时间为2007年1月1日 00:00:59（分钟末）
        ival_S_end_of_minute = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=59
        )

        # 创建秒级时间周期对象到年度周期对象的转换对象
        ival_S_to_A = Period(freq="Y", year=2007)
        # 创建秒级时间周期对象到季度周期对象的转换对象
        ival_S_to_Q = Period(freq="Q", year=2007, quarter=1)
        # 创建秒级时间周期对象到月度周期对象的转换对象
        ival_S_to_M = Period(freq="M", year=2007, month=1)
        # 创建秒级时间周期对象到周周期对象的转换对象
        ival_S_to_W = Period(freq="W", year=2007, month=1, day=7)
        # 创建秒级时间周期对象到日周期对象的转换对象
        ival_S_to_D = Period(freq="D", year=2007, month=1, day=1)

        # 断言触发未来警告，并匹配工作日消息
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 创建秒级时间周期对象到工作日周期对象的转换对象
            ival_S_to_B = Period(freq="B", year=2007, month=1, day=1)
        
        # 创建秒级时间周期对象到小时周期对象的转换对象
        ival_S_to_H = Period(freq="h", year=2007, month=1, day=1, hour=0)
        # 创建秒级时间周期对象到分钟周期对象的转换对象
        ival_S_to_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)

        # 断言秒级时间周期对象到年度周期对象的转换是否正确
        assert ival_S.asfreq("Y") == ival_S_to_A
        # 断言秒级时间周期对象到年度周期对象的转换是否正确（年末）
        assert ival_S_end_of_year.asfreq("Y") == ival_S_to_A
        # 断言秒级时间周期对象到季度周期对象的转换是否正确
        assert ival_S.asfreq("Q") == ival_S_to_Q
        # 断言秒级时间周期对象到季度周期对象的转换是否正确（季度末）
        assert ival_S_end_of_quarter.asfreq("Q") == ival_S_to_Q
        # 断言秒级时间周期对象到月度周期对象的转换是否正确
        assert ival_S.asfreq("M") == ival_S_to_M
        # 断言秒级时间周期对象到月度周期对象的转换是否正确（月末）
        assert ival_S_end_of_month.asfreq("M") == ival_S_to_M
        # 断言秒级时间周期对象到周周期对象的转换是否正确
        assert ival_S.asfreq("W") == ival_S_to_W
        # 断言秒级时间周期对象到周周期对象的转换是否正确（周末）
        assert ival_S_end_of_week.asfreq("W") == ival_S_to_W
        # 断言秒级时间周期对象到日周期对象的转换是否正确
        assert ival_S.asfreq("D") == ival_S_to_D
        # 断言秒级时间周期对象到日周期对象的转换是否正确（日末）
        assert ival_S_end_of_day.asfreq("D") == ival_S_to_D
        
        # 断言触发未来警告，并匹配工作日消息
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            # 断言秒级时间周期对象到工作日周期对象的转换是否正确
            assert ival_S.asfreq("B") == ival_S_to_B
            # 断言秒级时间周期对象到工作日周期对象的转换是否正确（工作日末）
            assert ival_S_end_of_bus.asfreq("B") == ival_S_to_B
        
        # 断言秒级时间周期对象到小时周期对象的转换是否正确
        assert ival_S.asfreq("h") == ival_S_to_H
        # 断言秒级时间周期对象到小时周期对象的转换是否正确（小时末）
        assert ival_S_end_of_hour.asfreq("h") == ival_S_to_H
        # 断言秒级时间周期对象到分钟周期对象的转换是否正确
        assert ival_S.asfreq("Min") == ival_S_to_T
        # 断言秒级时间周期对象到分钟周期对象的转换是否正确（分钟末）
        assert ival_S_end_of_minute.asfreq("Min") == ival_S_to_T

        # 断言秒级时间周期对象到秒级时间周期对象的转换是否正确
        assert ival_S.asfreq("s") == ival_S
    def test_conv_microsecond(self):
        # 定义测试函数，验证微秒级别频率的时间周期转换功能
        # GH#31475 避免浮点错误导致开始时间落在周期开始之前
        per = Period("2020-01-30 15:57:27.576166", freq="us")
        # 断言周期的序数属性与预期值相等
        assert per.ordinal == 1580399847576166

        # 获取周期的开始时间
        start = per.start_time
        # 预期的开始时间为指定的时间戳
        expected = Timestamp("2020-01-30 15:57:27.576166")
        assert start == expected
        # 断言开始时间的内部数值等于周期序数乘以1000（微秒转换为纳秒）
        assert start._value == per.ordinal * 1000

        # 创建另一个微秒级别的周期，指定一个超出范围的日期
        per2 = Period("2300-01-01", "us")
        msg = "2300-01-01"
        # 使用 pytest 断言，验证超出范围的日期引发 OutOfBoundsDatetime 异常，异常信息匹配预期消息
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.start_time
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.end_time
    def test_asfreq_mult(self):
        # 定义一个年度频率的周期对象，指定年份为2007年
        p = Period(freq="Y", year=2007)
        
        # 将常规频率转换为多倍频率的测试
        for freq in ["3Y", offsets.YearEnd(3)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象
            result = p.asfreq(freq)
            # 预期的结果是一个频率为"3Y"的周期对象，年份为"2007"
            expected = Period("2007", freq="3Y")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq
        
        # 将常规频率转换为多倍频率的测试，指定how="S"
        for freq in ["3Y", offsets.YearEnd(3)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象，同时指定转换规则为"start"
            result = p.asfreq(freq, how="S")
            # 预期的结果是一个频率为"3Y"的周期对象，年份为"2007"
            expected = Period("2007", freq="3Y")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq

        # 将多倍频率转换为常规频率的测试
        p = Period(freq="3Y", year=2007)
        # ordinal属性会改变，因为默认转换规则为"end"
        for freq in ["Y", offsets.YearEnd()]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象
            result = p.asfreq(freq)
            # 预期的结果是一个频率为"Y"的周期对象，年份为"2009"
            expected = Period("2009", freq="Y")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq
        
        # 将多倍频率转换为常规频率的测试，指定how="s"
        for freq in ["Y", offsets.YearEnd()]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象，同时指定转换规则为"start"
            result = p.asfreq(freq, how="s")
            # 预期的结果是一个频率为"Y"的周期对象，年份为"2007"
            expected = Period("2007", freq="Y")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq

        # 以年度频率对象p为例，进行月度频率为"2M"的测试
        p = Period(freq="Y", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象
            result = p.asfreq(freq)
            # 预期的结果是一个频率为"2M"的周期对象，年份为"2007"，月份为"12"
            expected = Period("2007-12", freq="2M")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq
        
        for freq in ["2M", offsets.MonthEnd(2)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象，同时指定转换规则为"start"
            result = p.asfreq(freq, how="s")
            # 预期的结果是一个频率为"2M"的周期对象，年份为"2007"，月份为"01"
            expected = Period("2007-01", freq="2M")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq

        # 以多倍年度频率对象p为例，进行月度频率为"2M"的测试
        p = Period(freq="3Y", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象
            result = p.asfreq(freq)
            # 预期的结果是一个频率为"2M"的周期对象，年份为"2009"，月份为"12"
            expected = Period("2009-12", freq="2M")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq
        
        for freq in ["2M", offsets.MonthEnd(2)]:
            # 使用asfreq方法将周期对象p转换为指定频率freq的周期对象，同时指定转换规则为"start"
            result = p.asfreq(freq, how="s")
            # 预期的结果是一个频率为"2M"的周期对象，年份为"2007"，月份为"01"
            expected = Period("2007-01", freq="2M")

            # 断言结果与预期相等
            assert result == expected
            # 断言结果的ordinal属性与预期的ordinal属性相等
            assert result.ordinal == expected.ordinal
            # 断言结果的频率与预期的频率相等
            assert result.freq == expected.freq
    def test_asfreq_combined(self):
        # normal freq to combined freq
        # 创建一个频率为每小时的 Period 对象
        p = Period("2007", freq="h")

        # ordinal will not change
        # 预期结果为每25小时的 Period 对象，ordinal 应该不变
        expected = Period("2007", freq="25h")
        for freq, how in zip(["1D1h", "1h1D"], ["E", "S"]):
            # 将 p 转换为指定频率和处理方式的 Period 对象
            result = p.asfreq(freq, how=how)
            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        # combined freq to normal freq
        # 创建两个以不同顺序定义的 Period 对象
        p1 = Period(freq="1D1h", year=2007)
        p2 = Period(freq="1h1D", year=2007)

        # ordinal will change because how=E is the default
        # 将 combined frequency 的 Period 对象转换为每小时频率的 Period 对象
        result1 = p1.asfreq("h")
        result2 = p2.asfreq("h")
        expected = Period("2007-01-02", freq="h")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

        # ordinal will not change
        # 将 combined frequency 的 Period 对象按指定方式转换为每小时频率的 Period 对象
        result1 = p1.asfreq("h", how="S")
        result2 = p2.asfreq("h", how="S")
        expected = Period("2007-01-01", freq="h")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

    def test_asfreq_MS(self):
        # 创建一个初始的 Period 对象，表示年度
        initial = Period("2013")

        # 将年度频率的 Period 对象转换为每月频率的 Period 对象，处理方式为 S
        assert initial.asfreq(freq="M", how="S") == Period("2013-01", "M")

        msg = INVALID_FREQ_ERR_MSG
        # 测试无效频率转换的错误提示消息
        with pytest.raises(ValueError, match=msg):
            initial.asfreq(freq="MS", how="S")

        # 测试创建无效频率的 Period 对象的错误提示消息
        with pytest.raises(ValueError, match=msg):
            Period("2013-01", "MS")
```