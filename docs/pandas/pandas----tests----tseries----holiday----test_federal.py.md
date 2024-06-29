# `D:\src\scipysrc\pandas\pandas\tests\tseries\holiday\test_federal.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

from pandas import DatetimeIndex  # 从 pandas 库中导入 DatetimeIndex 类
import pandas._testing as tm  # 导入 pandas 库中的测试模块

from pandas.tseries.holiday import (  # 从 pandas 库中的时间序列假期模块中导入以下类和函数
    AbstractHolidayCalendar,  # 抽象假日日历类
    USFederalHolidayCalendar,  # 美国联邦假日日历类
    USMartinLutherKingJr,  # 美国马丁·路德·金日规则类
    USMemorialDay,  # 美国阵亡将士纪念日规则类
)


def test_no_mlk_before_1986():
    # see gh-10278
    # 定义一个继承 AbstractHolidayCalendar 的马丁·路德·金日日历类，规则包含马丁·路德·金日
    class MLKCalendar(AbstractHolidayCalendar):
        rules = [USMartinLutherKingJr]

    # 生成从 1984 年到 1988 年之间的马丁·路德·金日假日列表，并转换为 Python datetime 对象列表
    holidays = MLKCalendar().holidays(start="1984", end="1988").to_pydatetime().tolist()

    # 测试确保在 1986 年之前没有错误地观察到马丁·路德·金日
    assert holidays == [datetime(1986, 1, 20, 0, 0), datetime(1987, 1, 19, 0, 0)]


def test_memorial_day():
    # 定义一个继承 AbstractHolidayCalendar 的阵亡将士纪念日日历类，规则包含阵亡将士纪念日
    class MemorialDay(AbstractHolidayCalendar):
        rules = [USMemorialDay]

    # 生成从 1971 年到 1980 年之间的阵亡将士纪念日假日列表，并转换为 Python datetime 对象列表
    holidays = MemorialDay().holidays(start="1971", end="1980").to_pydatetime().tolist()

    # 修正 5 月 31 日的错误，并手动对比维基百科
    assert holidays == [
        datetime(1971, 5, 31, 0, 0),
        datetime(1972, 5, 29, 0, 0),
        datetime(1973, 5, 28, 0, 0),
        datetime(1974, 5, 27, 0, 0),
        datetime(1975, 5, 26, 0, 0),
        datetime(1976, 5, 31, 0, 0),
        datetime(1977, 5, 30, 0, 0),
        datetime(1978, 5, 29, 0, 0),
        datetime(1979, 5, 28, 0, 0),
    ]


def test_federal_holiday_inconsistent_returntype():
    # GH 49075 test case
    # 实例化两个美国联邦假日日历对象，以排除 _cache 影响
    cal1 = USFederalHolidayCalendar()
    cal2 = USFederalHolidayCalendar()

    # 检查 2018 年 8 月和 2019 年 8 月之间的节假日，与 GH49075 提交的预期结果一致
    results_2018 = cal1.holidays(start=datetime(2018, 8, 1), end=datetime(2018, 8, 31))
    results_2019 = cal2.holidays(start=datetime(2019, 8, 1), end=datetime(2019, 8, 31))
    expected_results = DatetimeIndex([], dtype="datetime64[ns]", freq=None)

    # 断言检查预期结果与实际结果是否一致
    tm.assert_index_equal(results_2018, expected_results)
    tm.assert_index_equal(results_2019, expected_results)
```