# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_isocalendar.py`

```
# 导入必要的库和模块：DataFrame, DatetimeIndex, date_range
from pandas import (
    DataFrame,
    DatetimeIndex,
    date_range,
)
# 导入 pandas 内部测试模块
import pandas._testing as tm


# 测试函数：验证带有时区信息的 DatetimeIndex 和其 TimeStamp 元素的 isocalendar() 方法在新年附近返回正确的 weekofyear 访问器
def test_isocalendar_returns_correct_values_close_to_new_year_with_tz():
    # 定义日期字符串列表
    dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
    # 创建带有时区信息的 DatetimeIndex 对象
    dates = DatetimeIndex(dates, tz="Europe/Brussels")
    # 对 DatetimeIndex 对象调用 isocalendar() 方法
    result = dates.isocalendar()
    # 预期的 DataFrame 结果，包含年份、周数和天数
    expected_data_frame = DataFrame(
        [[2013, 52, 7], [2014, 1, 1], [2014, 1, 2]],
        columns=["year", "week", "day"],
        index=dates,
        dtype="UInt32",
    )
    # 使用测试模块中的 assert_frame_equal 方法验证结果与预期是否一致
    tm.assert_frame_equal(result, expected_data_frame)


# 测试函数：验证 DatetimeIndex 和其 TimeStamp 元素的 isocalendar() 方法返回正确的字段
def test_dti_timestamp_isocalendar_fields():
    # 创建一个日期范围为 10 天的 DatetimeIndex 对象
    idx = date_range("2020-01-01", periods=10)
    # 取最后一个时间戳的 isocalendar() 方法的结果作为预期值
    expected = tuple(idx.isocalendar().iloc[-1].to_list())
    # 对最后一个时间戳调用 isocalendar() 方法得到实际结果
    result = idx[-1].isocalendar()
    # 使用 assert 语句验证实际结果与预期结果是否相等
    assert result == expected
```