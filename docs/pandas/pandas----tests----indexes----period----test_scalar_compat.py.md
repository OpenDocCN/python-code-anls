# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_scalar_compat.py`

```
# 导入 pytest 库，用于编写和运行测试
import pytest

# 从 pandas 库中导入需要使用的模块
from pandas import (
    Timedelta,
    date_range,
    period_range,
)

# 导入 pandas 内部的测试模块
import pandas._testing as tm

# 定义一个测试类 TestPeriodIndexOps，用于测试 PeriodIndex 的操作
class TestPeriodIndexOps:
    
    # 定义测试方法 test_start_time，测试 PeriodIndex 的起始时间
    def test_start_time(self):
        # 创建一个频率为每月的 PeriodIndex，从 '2016-01-01' 到 '2016-05-31'
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        # 创建预期的日期索引，频率为每月第一天
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="MS")
        # 断言 index 的起始时间与预期的日期索引相等
        tm.assert_index_equal(index.start_time, expected_index)

    # 定义测试方法 test_end_time，测试 PeriodIndex 的结束时间
    def test_end_time(self):
        # 创建一个频率为每月的 PeriodIndex，从 '2016-01-01' 到 '2016-05-31'
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        # 创建预期的日期索引，频率为每月最后一天
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="ME")
        # 将预期的日期索引调整为每月最后一天的下一天，并减去一纳秒
        expected_index += Timedelta(1, "D") - Timedelta(1, "ns")
        # 断言 index 的结束时间与预期的日期索引相等
        tm.assert_index_equal(index.end_time, expected_index)

    # 定义测试方法 test_end_time_business_friday，测试工作日频率下的结束时间
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_end_time_business_friday(self):
        # 创建一个频率为工作日（每周五）的 PeriodIndex，从 '1990-01-05' 开始，包含 1 个周期
        pi = period_range("1990-01-05", freq="B", periods=1)
        # 获取工作日频率的结束时间
        result = pi.end_time
        # 创建预期的日期索引，频率为每日，包含 1 个周期，并调整为工作日频率的结束时间
        dti = date_range("1990-01-05", freq="D", periods=1)._with_freq(None)
        expected = dti + Timedelta(days=1, nanoseconds=-1)
        # 断言 result 与预期的日期索引相等
        tm.assert_index_equal(result, expected)
```