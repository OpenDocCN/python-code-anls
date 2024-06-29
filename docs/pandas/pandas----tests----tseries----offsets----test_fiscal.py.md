# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_fiscal.py`

```
"""
Tests for Fiscal Year and Fiscal Quarter offset classes
"""

# 导入所需模块和类
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
    WeekDay,
    assert_is_on_offset,
    assert_offset_equal,
)
from pandas.tseries.offsets import (
    FY5253,
    FY5253Quarter,
)


# 创建一个返回最后一个月份的财年季度偏移对象的函数
def makeFY5253LastOfMonthQuarter(*args, **kwds):
    return FY5253Quarter(*args, variation="last", **kwds)


# 创建一个返回最接近结束月份的财年季度偏移对象的函数
def makeFY5253NearestEndMonthQuarter(*args, **kwds):
    return FY5253Quarter(*args, variation="nearest", **kwds)


# 创建一个返回最接近结束月份的财年偏移对象的函数
def makeFY5253NearestEndMonth(*args, **kwds):
    return FY5253(*args, variation="nearest", **kwds)


# 创建一个返回最后一个月份的财年偏移对象的函数
def makeFY5253LastOfMonth(*args, **kwds):
    return FY5253(*args, variation="last", **kwds)


# 定义一个测试函数，用于测试获取偏移名称的功能
def test_get_offset_name():
    # 断言返回的偏移对象的频率字符串等于指定的值
    assert (
        makeFY5253LastOfMonthQuarter(
            weekday=1, startingMonth=3, qtr_with_extra_week=4
        ).freqstr
        == "REQ-L-MAR-TUE-4"
    )
    # 断言返回的偏移对象的频率字符串等于指定的值
    assert (
        makeFY5253NearestEndMonthQuarter(
            weekday=1, startingMonth=3, qtr_with_extra_week=3
        ).freqstr
        == "REQ-N-MAR-TUE-3"
    )


# 定义一个测试类，用于测试最后一个月份的财年偏移对象
class TestFY5253LastOfMonth:
    # 创建一个特定设置的最后一个月份的财年偏移对象
    offset_lom_sat_aug = makeFY5253LastOfMonth(1, startingMonth=8, weekday=WeekDay.SAT)
    # 创建另一个特定设置的最后一个月份的财年偏移对象
    offset_lom_sat_sep = makeFY5253LastOfMonth(1, startingMonth=9, weekday=WeekDay.SAT)
    on_offset_cases = [
        # 定义一个包含测试用例的列表，每个元素是一个元组，包含偏移量函数、日期时间和预期布尔值
        # 来自维基百科的数据，用于测试每年8月最后一个星期六的情况
        (offset_lom_sat_aug, datetime(2006, 8, 26), True),
        (offset_lom_sat_aug, datetime(2007, 8, 25), True),
        (offset_lom_sat_aug, datetime(2008, 8, 30), True),
        (offset_lom_sat_aug, datetime(2009, 8, 29), True),
        (offset_lom_sat_aug, datetime(2010, 8, 28), True),
        (offset_lom_sat_aug, datetime(2011, 8, 27), True),
        (offset_lom_sat_aug, datetime(2012, 8, 25), True),
        (offset_lom_sat_aug, datetime(2013, 8, 31), True),
        (offset_lom_sat_aug, datetime(2014, 8, 30), True),
        (offset_lom_sat_aug, datetime(2015, 8, 29), True),
        (offset_lom_sat_aug, datetime(2016, 8, 27), True),
        (offset_lom_sat_aug, datetime(2017, 8, 26), True),
        (offset_lom_sat_aug, datetime(2018, 8, 25), True),
        (offset_lom_sat_aug, datetime(2019, 8, 31), True),
        # 继续添加其他年份8月最后一个星期六的测试数据，期望结果为True
        (offset_lom_sat_aug, datetime(2006, 8, 27), False),
        (offset_lom_sat_aug, datetime(2007, 8, 28), False),
        (offset_lom_sat_aug, datetime(2008, 8, 31), False),
        (offset_lom_sat_aug, datetime(2009, 8, 30), False),
        (offset_lom_sat_aug, datetime(2010, 8, 29), False),
        (offset_lom_sat_aug, datetime(2011, 8, 28), False),
        (offset_lom_sat_aug, datetime(2006, 8, 25), False),
        (offset_lom_sat_aug, datetime(2007, 8, 24), False),
        (offset_lom_sat_aug, datetime(2008, 8, 29), False),
        (offset_lom_sat_aug, datetime(2009, 8, 28), False),
        (offset_lom_sat_aug, datetime(2010, 8, 27), False),
        (offset_lom_sat_aug, datetime(2011, 8, 26), False),
        (offset_lom_sat_aug, datetime(2019, 8, 30), False),
        # 从GMCR获取的数据，用于测试每年9月最后一个星期六的情况
        # 参考链接为GMCR，用于验证每年9月最后一个星期六的偏移量
        (offset_lom_sat_sep, datetime(2010, 9, 25), True),
        (offset_lom_sat_sep, datetime(2011, 9, 24), True),
        (offset_lom_sat_sep, datetime(2012, 9, 29), True),
    ]

    # 使用pytest的参数化测试标记，迭代测试用例列表中的每个测试案例
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 分别取出测试案例的偏移量函数、日期时间和预期布尔值
        offset, dt, expected = case
        # 调用被测试的函数assert_is_on_offset，验证实际结果与预期结果是否一致
        assert_is_on_offset(offset, dt, expected)
    # 定义一个测试方法 test_apply，用于测试 makeFY5253LastOfMonth 函数的输出是否符合预期
    def test_apply(self):
        # 调用 makeFY5253LastOfMonth 函数，生成从八月开始每月最后一个周六的偏移日期
        offset_lom_aug_sat = makeFY5253LastOfMonth(startingMonth=8, weekday=WeekDay.SAT)
        # 调用 makeFY5253LastOfMonth 函数，生成从八月开始每月第一个周六的偏移日期
        offset_lom_aug_sat_1 = makeFY5253LastOfMonth(
            n=1, startingMonth=8, weekday=WeekDay.SAT
        )

        # 预期的日期序列，包含了从2006年到2016年八月份最后一个周六的日期
        date_seq_lom_aug_sat = [
            datetime(2006, 8, 26),
            datetime(2007, 8, 25),
            datetime(2008, 8, 30),
            datetime(2009, 8, 29),
            datetime(2010, 8, 28),
            datetime(2011, 8, 27),
            datetime(2012, 8, 25),
            datetime(2013, 8, 31),
            datetime(2014, 8, 30),
            datetime(2015, 8, 29),
            datetime(2016, 8, 27),
        ]

        # 定义测试用例列表，每个元素包含一个偏移值和预期的日期序列
        tests = [
            (offset_lom_aug_sat, date_seq_lom_aug_sat),
            (offset_lom_aug_sat_1, date_seq_lom_aug_sat),
            # 测试与偏移值相加后的日期是否与预期相符
            (offset_lom_aug_sat, [datetime(2006, 8, 25)] + date_seq_lom_aug_sat),
            # 测试与偏移值相加后的日期是否与预期相符（偏移值为1）
            (offset_lom_aug_sat_1, [datetime(2006, 8, 27)] + date_seq_lom_aug_sat[1:]),
            # 测试逆序日期序列与偏移值相加后是否与预期相符
            (
                makeFY5253LastOfMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT),
                list(reversed(date_seq_lom_aug_sat)),
            ),
        ]
        # 遍历测试用例列表，对每个测试用例执行断言检查
        for test in tests:
            offset, data = test
            current = data[0]
            for datum in data[1:]:
                current = current + offset
                # 断言当前计算出的日期与预期的日期是否相等
                assert current == datum
class TestFY5253NearestEndMonth:
    # 测试类，用于测试 FY5253NearestEndMonth 类的功能

    def test_get_year_end(self):
        # 测试获取年末日期的方法

        # 断言：以8月开始，每周六作为结束日期的情况下，2013年1月1日的年末日期应为2013年8月31日
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.SAT
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 31)
        
        # 断言：以8月开始，每周日作为结束日期的情况下，2013年1月1日的年末日期应为2013年9月1日
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.SUN
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 9, 1)
        
        # 断言：以8月开始，每周五作为结束日期的情况下，2013年1月1日的年末日期应为2013年8月30日
        assert makeFY5253NearestEndMonth(
            startingMonth=8, weekday=WeekDay.FRI
        ).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 30)

        # 创建一个以12月开始，每周二作为结束日期的 FY5253 对象，采用最近的变化规则
        offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")
        
        # 断言：以2012年1月1日为基准日期，该对象的年末日期应为2013年1月1日
        assert offset_n.get_year_end(datetime(2012, 1, 1)) == datetime(2013, 1, 1)
        
        # 断言：以2012年1月10日为基准日期，该对象的年末日期应为2013年1月1日
        assert offset_n.get_year_end(datetime(2012, 1, 10)) == datetime(2013, 1, 1)

        # 断言：以2013年1月1日为基准日期，该对象的年末日期应为2013年12月31日
        assert offset_n.get_year_end(datetime(2013, 1, 1)) == datetime(2013, 12, 31)
        
        # 断言：以2013年1月2日为基准日期，该对象的年末日期应为2013年12月31日
        assert offset_n.get_year_end(datetime(2013, 1, 2)) == datetime(2013, 12, 31)
        
        # 断言：以2013年1月3日为基准日期，该对象的年末日期应为2013年12月31日
        assert offset_n.get_year_end(datetime(2013, 1, 3)) == datetime(2013, 12, 31)
        
        # 断言：以2013年1月10日为基准日期，该对象的年末日期应为2013年12月31日
        assert offset_n.get_year_end(datetime(2013, 1, 10)) == datetime(2013, 12, 31)

        # 创建一个以12月开始，每周六作为结束日期的 FY5253 对象，第1周为第1周，采用最近的变化规则
        JNJ = FY5253(n=1, startingMonth=12, weekday=6, variation="nearest")
        
        # 断言：以2006年1月1日为基准日期，该对象的年末日期应为2006年12月31日
        assert JNJ.get_year_end(datetime(2006, 1, 1)) == datetime(2006, 12, 31)

    # 创建一个以8月开始，每周六作为结束日期的 FY5253NearestEndMonth 对象
    offset_lom_aug_sat = makeFY5253NearestEndMonth(
        1, startingMonth=8, weekday=WeekDay.SAT
    )
    
    # 创建一个以8月开始，每周四作为结束日期的 FY5253NearestEndMonth 对象
    offset_lom_aug_thu = makeFY5253NearestEndMonth(
        1, startingMonth=8, weekday=WeekDay.THU
    )
    
    # 创建一个以12月开始，每周二作为结束日期的 FY5253 对象，采用最近的变化规则
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")
    # 定义一个包含测试用例的列表，用于测试 "is_on_offset" 函数
    on_offset_cases = [
        # 以下是从维基百科中获取的测试案例，测试日期与预期结果
        # 详见：https://en.wikipedia.org/wiki/4%E2%80%934%E2%80%935_calendar
        # 2006年9月2日   2006年9月2日
        # 2007年9月1日   2007年9月1日
        # 2008年8月30日   2008年8月30日    (闰年)
        # 2009年8月29日   2009年8月29日
        # 2010年8月28日   2010年8月28日
        # 2011年9月3日   2011年9月3日
        # 2012年9月1日   2012年9月1日    (闰年)
        # 2013年8月31日   2013年8月31日
        # 2014年8月30日   2014年8月30日
        # 2015年8月29日   2015年8月29日
        # 2016年9月3日   2016年9月3日    (闰年)
        # 2017年9月2日   2017年9月2日
        # 2018年9月1日   2018年9月1日
        # 2019年8月31日   2019年8月31日
        (offset_lom_aug_sat, datetime(2006, 9, 2), True),
        (offset_lom_aug_sat, datetime(2007, 9, 1), True),
        (offset_lom_aug_sat, datetime(2008, 8, 30), True),
        (offset_lom_aug_sat, datetime(2009, 8, 29), True),
        (offset_lom_aug_sat, datetime(2010, 8, 28), True),
        (offset_lom_aug_sat, datetime(2011, 9, 3), True),
        (offset_lom_aug_sat, datetime(2016, 9, 3), True),
        (offset_lom_aug_sat, datetime(2017, 9, 2), True),
        (offset_lom_aug_sat, datetime(2018, 9, 1), True),
        (offset_lom_aug_sat, datetime(2019, 8, 31), True),
        # 下面的日期不符合预期结果，返回 False
        (offset_lom_aug_sat, datetime(2006, 8, 27), False),
        (offset_lom_aug_sat, datetime(2007, 8, 28), False),
        (offset_lom_aug_sat, datetime(2008, 8, 31), False),
        (offset_lom_aug_sat, datetime(2009, 8, 30), False),
        (offset_lom_aug_sat, datetime(2010, 8, 29), False),
        (offset_lom_aug_sat, datetime(2011, 8, 28), False),
        (offset_lom_aug_sat, datetime(2006, 8, 25), False),
        (offset_lom_aug_sat, datetime(2007, 8, 24), False),
        (offset_lom_aug_sat, datetime(2008, 8, 29), False),
        (offset_lom_aug_sat, datetime(2009, 8, 28), False),
        (offset_lom_aug_sat, datetime(2010, 8, 27), False),
        (offset_lom_aug_sat, datetime(2011, 8, 26), False),
        (offset_lom_aug_sat, datetime(2019, 8, 30), False),
        # 从 Micron 公司获取的测试案例，详见：
        # http://google.brand.edgar-online.com/?sym=MU&formtypeID=7
        # 测试 "offset_lom_aug_thu" 函数
        (offset_lom_aug_thu, datetime(2012, 8, 30), True),
        (offset_lom_aug_thu, datetime(2011, 9, 1), True),
        # 其他测试案例
        (offset_n, datetime(2012, 12, 31), False),
        (offset_n, datetime(2013, 1, 1), True),
        (offset_n, datetime(2013, 1, 2), False),
    ]

    # 使用 pytest 的 parametrize 装饰器，对每个测试案例进行参数化测试
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 解包测试案例元组，包括偏移量、日期时间、预期结果
        offset, dt, expected = case
        # 断言调用 is_on_offset 函数的结果与预期结果一致
        assert_is_on_offset(offset, dt, expected)
    # 定义一个测试方法 `test_apply`，用于测试某些日期计算的准确性
    def test_apply(self):
        # 定义一组参考日期序列 `date_seq_nem_8_sat`
        date_seq_nem_8_sat = [
            datetime(2006, 9, 2),
            datetime(2007, 9, 1),
            datetime(2008, 8, 30),
            datetime(2009, 8, 29),
            datetime(2010, 8, 28),
            datetime(2011, 9, 3),
        ]

        # 另一个参考日期序列 `JNJ`
        JNJ = [
            datetime(2005, 1, 2),
            datetime(2006, 1, 1),
            datetime(2006, 12, 31),
            datetime(2007, 12, 30),
            datetime(2008, 12, 28),
            datetime(2010, 1, 3),
            datetime(2011, 1, 2),
            datetime(2012, 1, 1),
            datetime(2012, 12, 30),
        ]

        # 定义一个特定的日期计算对象 `DEC_SAT`
        DEC_SAT = FY5253(n=-1, startingMonth=12, weekday=5, variation="nearest")

        # 定义一组测试案例列表 `tests`
        tests = [
            (
                makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT),
                date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT),
                date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT),
                [datetime(2006, 9, 1)] + date_seq_nem_8_sat,
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT),
                [datetime(2006, 9, 3)] + date_seq_nem_8_sat[1:],
            ),
            (
                makeFY5253NearestEndMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT),
                list(reversed(date_seq_nem_8_sat)),
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                JNJ,
            ),
            (
                makeFY5253NearestEndMonth(n=-1, startingMonth=12, weekday=WeekDay.SUN),
                list(reversed(JNJ)),
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                [datetime(2005, 1, 2), datetime(2006, 1, 1)],
            ),
            (
                makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN),
                [datetime(2006, 1, 2), datetime(2006, 12, 31)],
            ),
            (DEC_SAT, [datetime(2013, 1, 15), datetime(2012, 12, 29)]),
        ]
        
        # 遍历每个测试案例
        for test in tests:
            # 解包测试案例中的偏移量和日期数据
            offset, data = test
            # 取出当前日期数据中的第一个日期
            current = data[0]
            # 遍历日期数据中的每个日期，进行日期偏移计算
            for datum in data[1:]:
                current = current + offset  # 计算当前日期加上偏移量后的结果
                # 断言当前计算出的日期与预期的日期相等
                assert current == datum
class TestFY5253LastOfMonthQuarter:
    # 定义测试类 TestFY5253LastOfMonthQuarter

    def test_equality(self):
        # 定义测试方法 test_equality，用于测试相等性和不相等性

        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) == makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        # 断言两次调用 makeFY5253LastOfMonthQuarter 返回的结果相等

        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) != makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SUN, qtr_with_extra_week=4
        )
        # 断言两次调用 makeFY5253LastOfMonthQuarter 返回的结果不相等

        assert makeFY5253LastOfMonthQuarter(
            startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4
        ) != makeFY5253LastOfMonthQuarter(
            startingMonth=2, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        # 断言两次调用 makeFY5253LastOfMonthQuarter 返回的结果不相等
    # 定义一个测试方法 test_offset，用于测试 makeFY5253LastOfMonthQuarter 函数的偏移量计算
    def test_offset(self):
        # 调用 makeFY5253LastOfMonthQuarter 函数，计算指定条件下的偏移量 offset
        offset = makeFY5253LastOfMonthQuarter(
            1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        # 再次调用函数，计算不同参数下的偏移量 offset2 和 offset4
        offset2 = makeFY5253LastOfMonthQuarter(
            2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        offset4 = makeFY5253LastOfMonthQuarter(
            4, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        # 计算负数参数下的偏移量 offset_neg1 和 offset_neg2
        offset_neg1 = makeFY5253LastOfMonthQuarter(
            -1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )
        offset_neg2 = makeFY5253LastOfMonthQuarter(
            -2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        # 定义一个日期列表 GMCR，包含多个 datetime 对象
        GMCR = [
            datetime(2010, 3, 27),
            datetime(2010, 6, 26),
            datetime(2010, 9, 25),
            datetime(2010, 12, 25),
            datetime(2011, 3, 26),
            datetime(2011, 6, 25),
            datetime(2011, 9, 24),
            datetime(2011, 12, 24),
            datetime(2012, 3, 24),
            datetime(2012, 6, 23),
            datetime(2012, 9, 29),
            datetime(2012, 12, 29),
            datetime(2013, 3, 30),
            datetime(2013, 6, 29),
        ]

        # 断言偏移量 offset 的计算结果与预期结果相等
        assert_offset_equal(offset, base=GMCR[0], expected=GMCR[1])
        # 断言偏移量 offset 的计算结果与基准日期减一天后的日期相等
        assert_offset_equal(
            offset, base=GMCR[0] + relativedelta(days=-1), expected=GMCR[0]
        )
        # 断言偏移量 offset 的计算结果与预期结果相等
        assert_offset_equal(offset, base=GMCR[1], expected=GMCR[2])

        # 断言偏移量 offset2 的计算结果与预期结果相等
        assert_offset_equal(offset2, base=GMCR[0], expected=GMCR[2])
        # 断言偏移量 offset4 的计算结果与预期结果相等
        assert_offset_equal(offset4, base=GMCR[0], expected=GMCR[4])

        # 断言偏移量 offset_neg1 的计算结果与预期结果相等
        assert_offset_equal(offset_neg1, base=GMCR[-1], expected=GMCR[-2])
        # 断言偏移量 offset_neg1 的计算结果与基准日期加一天后的日期相等
        assert_offset_equal(
            offset_neg1, base=GMCR[-1] + relativedelta(days=+1), expected=GMCR[-1]
        )
        # 断言偏移量 offset_neg2 的计算结果与预期结果相等
        assert_offset_equal(offset_neg2, base=GMCR[-1], expected=GMCR[-3])

        # 从第一个基准日期 GMCR[0] 减去一天开始，依次迭代日期列表 GMCR，断言偏移量 offset 的计算结果与预期结果相等
        date = GMCR[0] + relativedelta(days=-1)
        for expected in GMCR:
            assert_offset_equal(offset, date, expected)
            date = date + offset

        # 从最后一个基准日期 GMCR[-1] 加上一天开始，倒序迭代日期列表 GMCR，断言偏移量 offset_neg1 的计算结果与预期结果相等
        date = GMCR[-1] + relativedelta(days=+1)
        for expected in reversed(GMCR):
            assert_offset_equal(offset_neg1, date, expected)
            date = date + offset_neg1

    # 定义一个变量 lomq_aug_sat_4，调用 makeFY5253LastOfMonthQuarter 函数计算特定条件下的偏移量
    lomq_aug_sat_4 = makeFY5253LastOfMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )
    # 定义一个变量 lomq_sep_sat_4，调用 makeFY5253LastOfMonthQuarter 函数计算特定条件下的偏移量
    lomq_sep_sat_4 = makeFY5253LastOfMonthQuarter(
        1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )
    # 定义包含测试用例的列表，每个元组包含一个偏移量、日期时间对象和预期布尔值
    on_offset_cases = [
        # From Wikipedia
        (lomq_aug_sat_4, datetime(2006, 8, 26), True),    # 测试八月的第四个星期六，2006年8月26日是否为偏移日期
        (lomq_aug_sat_4, datetime(2007, 8, 25), True),    # 同上，2007年8月25日是否为偏移日期
        (lomq_aug_sat_4, datetime(2008, 8, 30), True),    # 同上，2008年8月30日是否为偏移日期
        (lomq_aug_sat_4, datetime(2009, 8, 29), True),    # 同上，2009年8月29日是否为偏移日期
        (lomq_aug_sat_4, datetime(2010, 8, 28), True),    # 同上，2010年8月28日是否为偏移日期
        (lomq_aug_sat_4, datetime(2011, 8, 27), True),    # 同上，2011年8月27日是否为偏移日期
        (lomq_aug_sat_4, datetime(2019, 8, 31), True),    # 同上，2019年8月31日是否为偏移日期
        (lomq_aug_sat_4, datetime(2006, 8, 27), False),   # 测试八月的第四个星期六，2006年8月27日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2007, 8, 28), False),   # 同上，2007年8月28日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2008, 8, 31), False),   # 同上，2008年8月31日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2009, 8, 30), False),   # 同上，2009年8月30日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2010, 8, 29), False),   # 同上，2010年8月29日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2011, 8, 28), False),   # 同上，2011年8月28日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2006, 8, 25), False),   # 同上，2006年8月25日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2007, 8, 24), False),   # 同上，2007年8月24日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2008, 8, 29), False),   # 同上，2008年8月29日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2009, 8, 28), False),   # 同上，2009年8月28日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2010, 8, 27), False),   # 同上，2010年8月27日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2011, 8, 26), False),   # 同上，2011年8月26日是否为偏移日期（预期不是）
        (lomq_aug_sat_4, datetime(2019, 8, 30), False),   # 同上，2019年8月30日是否为偏移日期（预期不是）
        # From GMCR
        (lomq_sep_sat_4, datetime(2010, 9, 25), True),    # 测试九月的第四个星期六，2010年9月25日是否为偏移日期
        (lomq_sep_sat_4, datetime(2011, 9, 24), True),    # 同上，2011年9月24日是否为偏移日期
        (lomq_sep_sat_4, datetime(2012, 9, 29), True),    # 同上，2012年9月29日是否为偏移日期
        (lomq_sep_sat_4, datetime(2013, 6, 29), True),    # 同上，2013年6月29日是否为偏移日期
        (lomq_sep_sat_4, datetime(2012, 6, 23), True),    # 同上，2012年6月23日是否为偏移日期
        (lomq_sep_sat_4, datetime(2012, 6, 30), False),   # 同上，2012年6月30日是否为偏移日期（预期不是）
        (lomq_sep_sat_4, datetime(2013, 3, 30), True),    # 同上，2013年3月30日是否为偏移日期
        (lomq_sep_sat_4, datetime(2012, 3, 24), True),    # 同上，2012年3月24日是否为偏移日期
        (lomq_sep_sat_4, datetime(2012, 12, 29), True),   # 同上，2012年12月29日是否为偏移日期
        (lomq_sep_sat_4, datetime(2011, 12, 24), True),   # 同上，2011年12月24日是否为偏移日期
        # INTC (extra week in Q1)
        # See: http://www.intc.com/releasedetail.cfm?ReleaseID=542844
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2011, 4, 2),
            True,
        ),  # 测试特定配置下，2011年4月2日是否为偏移日期
        # see: http://google.brand.edgar-online.com/?sym=INTC&formtypeID=7
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2012, 12, 29),
            True,
        ),  # 同上，2012年12月29日是否为偏移日期
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2011, 12, 31),
            True,
        ),  # 同上，2011年12月31日是否为偏移日期
        (
            makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ),
            datetime(2010, 12, 25),
            True,
        ),  # 同上，2010年12月25日是否为偏移日期
    ]

    # 使用 pytest 的 parametrize 装饰器将测试函数参数化，参数为每个测试用例
    @pytest.mark.parametrize("case", on_offset_cases)
    # 定义测试函数，每次从测试用例中获取偏移量、日期时间和预期结果进行测试
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        # 断言偏移量函数的返回值是否等于预期结果
        assert_is_on_offset(offset, dt, expected)
    # 定义测试方法，用于验证年份是否有额外周
    def test_year_has_extra_week(self):
        # 验证长第一季度结束时的情况，返回是否有额外周
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2011, 4, 2))

        # 验证长第一季度开始时的情况，返回是否有额外周
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2010, 12, 26))

        # 验证长第一季度前一年结束时的情况，返回是否有额外周
        assert not makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2010, 12, 25))

        # 对1994到2011年中不包含2011、2005、2000、1994的年份进行验证，返回是否有额外周
        for year in [
            x for x in range(1994, 2011 + 1) if x not in [2011, 2005, 2000, 1994]
        ]:
            assert not makeFY5253LastOfMonthQuarter(
                1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
            ).year_has_extra_week(datetime(year, 4, 2))

        # 验证其他有长第一季度的年份，返回是否有额外周
        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2005, 4, 2))

        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(2000, 4, 2))

        assert makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        ).year_has_extra_week(datetime(1994, 4, 2))

    # 定义测试方法，用于验证获取周数
    def test_get_weeks(self):
        # 创建以每年12月第一个星期六为开始的长第一季度对象
        sat_dec_1 = makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1
        )
        # 创建以每年12月第四个星期六为开始的长第一季度对象
        sat_dec_4 = makeFY5253LastOfMonthQuarter(
            1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=4
        )

        # 验证给定日期在长第一季度的情况下，返回各周的周数列表
        assert sat_dec_1.get_weeks(datetime(2011, 4, 2)) == [14, 13, 13, 13]
        assert sat_dec_4.get_weeks(datetime(2011, 4, 2)) == [13, 13, 13, 14]
        assert sat_dec_1.get_weeks(datetime(2010, 12, 25)) == [13, 13, 13, 13]
class TestFY5253NearestEndMonthQuarter:
    # 创建一个偏移量为1的FY5253NearestEndMonthQuarter对象，从8月开始，每个周六，四分之四的额外周
    offset_nem_sat_aug_4 = makeFY5253NearestEndMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.SAT, qtr_with_extra_week=4
    )
    # 创建一个偏移量为1的FY5253NearestEndMonthQuarter对象，从8月开始，每个周四，四分之四的额外周
    offset_nem_thu_aug_4 = makeFY5253NearestEndMonthQuarter(
        1, startingMonth=8, weekday=WeekDay.THU, qtr_with_extra_week=4
    )
    # 创建一个FY5253对象，从12月开始，每个周二，变化模式为最接近
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation="nearest")

    # 定义测试用例列表
    on_offset_cases = [
        # From Wikipedia
        (offset_nem_sat_aug_4, datetime(2006, 9, 2), True),
        (offset_nem_sat_aug_4, datetime(2007, 9, 1), True),
        (offset_nem_sat_aug_4, datetime(2008, 8, 30), True),
        (offset_nem_sat_aug_4, datetime(2009, 8, 29), True),
        (offset_nem_sat_aug_4, datetime(2010, 8, 28), True),
        (offset_nem_sat_aug_4, datetime(2011, 9, 3), True),
        (offset_nem_sat_aug_4, datetime(2016, 9, 3), True),
        (offset_nem_sat_aug_4, datetime(2017, 9, 2), True),
        (offset_nem_sat_aug_4, datetime(2018, 9, 1), True),
        (offset_nem_sat_aug_4, datetime(2019, 8, 31), True),
        (offset_nem_sat_aug_4, datetime(2006, 8, 27), False),
        (offset_nem_sat_aug_4, datetime(2007, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2008, 8, 31), False),
        (offset_nem_sat_aug_4, datetime(2009, 8, 30), False),
        (offset_nem_sat_aug_4, datetime(2010, 8, 29), False),
        (offset_nem_sat_aug_4, datetime(2011, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2006, 8, 25), False),
        (offset_nem_sat_aug_4, datetime(2007, 8, 24), False),
        (offset_nem_sat_aug_4, datetime(2008, 8, 29), False),
        (offset_nem_sat_aug_4, datetime(2009, 8, 28), False),
        (offset_nem_sat_aug_4, datetime(2010, 8, 27), False),
        (offset_nem_sat_aug_4, datetime(2011, 8, 26), False),
        (offset_nem_sat_aug_4, datetime(2019, 8, 30), False),
        # From Micron, see:
        # http://google.brand.edgar-online.com/?sym=MU&formtypeID=7
        (offset_nem_thu_aug_4, datetime(2012, 8, 30), True),
        (offset_nem_thu_aug_4, datetime(2011, 9, 1), True),
        # See: http://google.brand.edgar-online.com/?sym=MU&formtypeID=13
        (offset_nem_thu_aug_4, datetime(2013, 5, 30), True),
        (offset_nem_thu_aug_4, datetime(2013, 2, 28), True),
        (offset_nem_thu_aug_4, datetime(2012, 11, 29), True),
        (offset_nem_thu_aug_4, datetime(2012, 5, 31), True),
        (offset_nem_thu_aug_4, datetime(2007, 3, 1), True),
        (offset_nem_thu_aug_4, datetime(1994, 3, 3), True),
        (offset_n, datetime(2012, 12, 31), False),
        (offset_n, datetime(2013, 1, 1), True),
        (offset_n, datetime(2013, 1, 2), False),
    ]

    # 参数化测试用例
    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 从测试用例中获取偏移量、日期和预期值
        offset, dt, expected = case
        # 断言是否符合预期
        assert_is_on_offset(offset, dt, expected)
    # 定义一个测试方法，用于测试计算偏移量的功能
    def test_offset(self):
        # 使用特定参数调用函数 `makeFY5253NearestEndMonthQuarter`，返回偏移量 `offset`
        offset = makeFY5253NearestEndMonthQuarter(
            1, startingMonth=8, weekday=WeekDay.THU, qtr_with_extra_week=4
        )

        # 定义一个包含多个日期的列表 `MU`
        MU = [
            datetime(2012, 5, 31),
            datetime(2012, 8, 30),
            datetime(2012, 11, 29),
            datetime(2013, 2, 28),
            datetime(2013, 5, 30),
        ]

        # 从 `MU` 中获取第一个日期，向前偏移一天，作为初始日期 `date`
        date = MU[0] + relativedelta(days=-1)
        
        # 遍历日期列表 `MU`，对每个日期 `expected` 进行偏移量测试
        for expected in MU:
            # 断言 `date` 经过 `offset` 偏移后与预期 `expected` 相等
            assert_offset_equal(offset, date, expected)
            # 更新 `date`，加上偏移量 `offset`
            date = date + offset

        # 对特定日期对进行偏移量测试，断言结果与预期相等
        assert_offset_equal(offset, datetime(2012, 5, 31), datetime(2012, 8, 30))
        assert_offset_equal(offset, datetime(2012, 5, 30), datetime(2012, 5, 31))

        # 使用另一种偏移量 `offset2` 进行测试，初始化偏移量对象 `FY5253Quarter`
        offset2 = FY5253Quarter(
            weekday=5, startingMonth=12, variation="last", qtr_with_extra_week=4
        )

        # 断言特定日期对经过 `offset2` 偏移后与预期相等
        assert_offset_equal(offset2, datetime(2013, 1, 15), datetime(2013, 3, 30))
# 测试函数，用于测试“FY5253”类的不同情况
def test_bunched_yearends():
    # 创建一个FY5253对象，定义财政年度的起始月份为12月，变体为“nearest”
    fy = FY5253(n=1, weekday=5, startingMonth=12, variation="nearest")
    # 创建一个时间戳对象，表示2004年1月1日
    dt = Timestamp("2004-01-01")
    # 断言回滚方法的返回结果为2002年12月28日
    assert fy.rollback(dt) == Timestamp("2002-12-28")
    # 断言负FY5253对象应用于时间戳的结果为2002年12月28日
    assert (-fy)._apply(dt) == Timestamp("2002-12-28")
    # 断言时间戳减去FY5253对象的结果为2002年12月28日
    assert dt - fy == Timestamp("2002-12-28")

    # 断言滚动方法的返回结果为2004年1月3日
    assert fy.rollforward(dt) == Timestamp("2004-01-03")
    # 断言FY5253对象应用于时间戳的结果为2004年1月3日
    assert fy._apply(dt) == Timestamp("2004-01-03")
    # 断言时间戳加上FY5253对象的结果为2004年1月3日
    assert fy + dt == Timestamp("2004-01-03")
    # 断言时间戳加上FY5253对象的结果为2004年1月3日
    assert dt + fy == Timestamp("2004-01-03")

    # 使用上一个时间戳，即2003年12月31日
    dt = Timestamp("2003-12-31")
    # 断言回滚方法的返回结果为2002年12月28日
    assert fy.rollback(dt) == Timestamp("2002-12-28")
    # 断言负FY5253对象应用于时间戳的结果为2002年12月28日
    assert (-fy)._apply(dt) == Timestamp("2002-12-28")
    # 断言时间戳减去FY5253对象的结果为2002年12月28日
    assert dt - fy == Timestamp("2002-12-28")


# 测试函数，用于测试“FY5253”类中处理最后一个偏移的情况
def test_fy5253_last_onoffset():
    # 创建一个FY5253对象，定义财政年度的起始月份为5月，变体为“last”，工作日为0（星期一）
    offset = FY5253(n=-5, startingMonth=5, variation="last", weekday=0)
    # 创建一个时间戳对象，表示1984年5月28日06:29:43.955911354 +0200时区为“Europe/San_Marino”
    ts = Timestamp("1984-05-28 06:29:43.955911354+0200", tz="Europe/San_Marino")
    # 快速检查时间戳是否在偏移位置上
    fast = offset.is_on_offset(ts)
    # 慢速检查：时间戳加上偏移后再减去偏移是否等于原时间戳
    slow = (ts + offset) - offset == ts
    # 断言快速检查结果与慢速检查结果相等
    assert fast == slow


# 测试函数，用于测试“FY5253”类中处理最近一个偏移的情况
def test_fy5253_nearest_onoffset():
    # 创建一个FY5253对象，定义财政年度的起始月份为7月，变体为“nearest”，工作日为2（星期三）
    offset = FY5253(n=3, startingMonth=7, variation="nearest", weekday=2)
    # 创建一个时间戳对象，表示2032年7月28日00:12:59.035729419 +0000时区为“Africa/Dakar”
    ts = Timestamp("2032-07-28 00:12:59.035729419+0000", tz="Africa/Dakar")
    # 快速检查时间戳是否在偏移位置上
    fast = offset.is_on_offset(ts)
    # 慢速检查：时间戳加上偏移后再减去偏移是否等于原时间戳
    slow = (ts + offset) - offset == ts
    # 断言快速检查结果与慢速检查结果相等
    assert fast == slow


# 测试函数，用于测试“FY5253Quarter”类中最近偏移的情况
def test_fy5253qtr_onoffset_nearest():
    # 创建一个时间戳对象，表示1985年9月2日23:57:46.232550356 -0300时区为“Atlantic/Bermuda”
    ts = Timestamp("1985-09-02 23:57:46.232550356-0300", tz="Atlantic/Bermuda")
    # 创建一个FY5253Quarter对象，定义财政季度为3，带有额外周数，起始月份为2，变体为“nearest”，工作日为0（星期一）
    offset = FY5253Quarter(n=3, qtr_with_extra_week=1, startingMonth=2, variation="nearest", weekday=0)
    # 快速检查时间戳是否在偏移位置上
    fast = offset.is_on_offset(ts)
    # 慢速检查：时间戳加上偏移后再减去偏移是否等于原时间戳
    slow = (ts + offset) - offset == ts
    # 断言快速检查结果与慢速检查结果相等
    assert fast == slow


# 测试函数，用于测试“FY5253Quarter”类中最后偏移的情况
def test_fy5253qtr_onoffset_last():
    # 创建一个FY5253Quarter对象，定义财政季度为-2，带有额外周数，起始月份为7，变体为“last”，工作日为2（星期三）
    offset = FY5253Quarter(n=-2, qtr_with_extra_week=1, startingMonth=7, variation="last", weekday=2)
    # 创建一个时间戳对象，表示2011年1月26日19:03:40.331096129 +0200时区为“Africa/Windhoek”
    ts = Timestamp("2011-01-26 19:03:40.331096129+0200", tz="Africa/Windhoek")
    # 慢速检查：时间戳加上偏移后再减去偏移是否等于原时间戳
    slow = (ts + offset) - offset == ts
    # 快速检查时间戳是否在偏移位置上
    fast = offset.is_on_offset(ts)
    # 断言快速检查结果与慢速检查结果相等
    assert fast == slow
```