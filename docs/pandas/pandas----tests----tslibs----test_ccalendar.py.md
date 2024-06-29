# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_ccalendar.py`

```
# 从 datetime 模块导入 date 和 datetime 类
# 从 hypothesis 模块导入 given 函数
# 导入 numpy 库，并将其命名为 np
# 导入 pytest 模块
# 从 pandas._libs.tslibs 中导入 ccalendar 模块
# 从 pandas._testing._hypothesis 中导入 DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，测试函数 test_get_day_of_year_numeric
# 参数为 date_tuple 和 expected，分别代表日期元组和预期结果
@pytest.mark.parametrize(
    "date_tuple,expected",
    [
        ((2001, 3, 1), 60),           # 测试2001年3月1日的结果是否为60
        ((2004, 3, 1), 61),           # 测试2004年3月1日的结果是否为61
        ((1907, 12, 31), 365),        # 测试1907年12月31日的结果是否为365，年末，非闰年
        ((2004, 12, 31), 366),        # 测试2004年12月31日的结果是否为366，年末，闰年
    ],
)
def test_get_day_of_year_numeric(date_tuple, expected):
    # 断言调用 ccalendar 模块的 get_day_of_year 函数，检查返回结果是否等于预期值
    assert ccalendar.get_day_of_year(*date_tuple) == expected

# 定义测试函数 test_get_day_of_year_dt
def test_get_day_of_year_dt():
    # 生成一个随机日期 dt
    dt = datetime.fromordinal(1 + np.random.default_rng(2).integers(365 * 4000))
    # 调用 ccalendar 模块的 get_day_of_year 函数计算 dt 的年份、月份和日期的一年中的天数
    result = ccalendar.get_day_of_year(dt.year, dt.month, dt.day)
    
    # 计算预期值，即从该年的第一天到 dt 的天数加一
    expected = (dt - dt.replace(month=1, day=1)).days + 1
    # 断言 result 是否等于 expected
    assert result == expected

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，测试函数 test_dt_correct_iso_8601_year_week_and_day
# 参数为 input_date_tuple 和 expected_iso_tuple，分别代表输入日期元组和预期的 ISO 8601 年份、周数和日期元组
@pytest.mark.parametrize(
    "input_date_tuple, expected_iso_tuple",
    [
        [(2020, 1, 1), (2020, 1, 3)],      # 测试2020年1月1日的 ISO 8601 结果是否为 (2020, 1, 3)
        [(2019, 12, 31), (2020, 1, 2)],   # 测试2019年12月31日的 ISO 8601 结果是否为 (2020, 1, 2)
        [(2019, 12, 30), (2020, 1, 1)],   # 测试2019年12月30日的 ISO 8601 结果是否为 (2020, 1, 1)
        [(2009, 12, 31), (2009, 53, 4)],  # 测试2009年12月31日的 ISO 8601 结果是否为 (2009, 53, 4)
        [(2010, 1, 1), (2009, 53, 5)],    # 测试2010年1月1日的 ISO 8601 结果是否为 (2009, 53, 5)
        [(2010, 1, 3), (2009, 53, 7)],    # 测试2010年1月3日的 ISO 8601 结果是否为 (2009, 53, 7)
        [(2010, 1, 4), (2010, 1, 1)],     # 测试2010年1月4日的 ISO 8601 结果是否为 (2010, 1, 1)
        [(2006, 1, 1), (2005, 52, 7)],    # 测试2006年1月1日的 ISO 8601 结果是否为 (2005, 52, 7)
        [(2005, 12, 31), (2005, 52, 6)],  # 测试2005年12月31日的 ISO 8601 结果是否为 (2005, 52, 6)
        [(2008, 12, 28), (2008, 52, 7)],  # 测试2008年12月28日的 ISO 8601 结果是否为 (2008, 52, 7)
        [(2008, 12, 29), (2009, 1, 1)],   # 测试2008年12月29日的 ISO 8601 结果是否为 (2009, 1, 1)
    ],
)
def test_dt_correct_iso_8601_year_week_and_day(input_date_tuple, expected_iso_tuple):
    # 调用 ccalendar 模块的 get_iso_calendar 函数计算 ISO 8601 年份、周数和日期
    result = ccalendar.get_iso_calendar(*input_date_tuple)
    # 获取日期的 ISO 8601 结果
    expected_from_date_isocalendar = date(*input_date_tuple).isocalendar()
    # 断言 result 是否等于 expected_from_date_isocalendar
    assert result == expected_from_date_isocalendar
    # 再次断言 result 是否等于 expected_iso_tuple
    assert result == expected_iso_tuple

# 使用 given 装饰器定义属性的参数化测试函数 test_isocalendar
# 参数为 DTETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ，即位于 pandas 时间戳范围内的无时区日期时间
@given(DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ)
def test_isocalendar(dt):
    # 计算预期 ISO 8601 结果
    expected = dt.isocalendar()
    # 调用 ccalendar 模块的 get_iso_calendar 函数计算 ISO 8601 年份、周数和日期
    result = ccalendar.get_iso_calendar(dt.year, dt.month, dt.day)
    # 断言 result 是否等于 expected
    assert result == expected
```