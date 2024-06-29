# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_arithmetic.py`

```
# 引入 pytest 测试框架
import pytest

# 从 pandas 库中导入所需的类和函数
from pandas import (
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
    timedelta_range,
)

# 导入 pandas 内部用于测试的模块
import pandas._testing as tm

# 定义测试类 TestDatetimeIndexArithmetic
class TestDatetimeIndexArithmetic:
    
    # 测试加法操作保留频率信息
    def test_add_timedelta_preserves_freq(self):
        # 时区设定为 Canada/Eastern
        tz = "Canada/Eastern"
        # 创建一个日期范围对象 dti，从 '2019-03-26' 到 '2020-10-17'，频率为每天 ('D')
        dti = date_range(
            start=Timestamp("2019-03-26 00:00:00-0400", tz=tz),
            end=Timestamp("2020-10-17 00:00:00-0400", tz=tz),
            freq="D",
        )
        # 对 dti 加上一天的 Timedelta
        result = dti + Timedelta(days=1)
        # 断言结果的频率与 dti 的频率相同
        assert result.freq == dti.freq

    # 测试减法操作保留频率信息
    def test_sub_datetime_preserves_freq(self, tz_naive_fixture):
        # 创建一个时区感知的日期范围对象 dti，从 '2016-01-01' 开始，包含 12 个周期，使用给定的时区 tz_naive_fixture
        dti = date_range("2016-01-01", periods=12, tz=tz_naive_fixture)
        # 对 dti 减去第一个元素（Timestamp），得到时间差索引 res
        res = dti - dti[0]
        # 期望的时间差范围对象 expected，从 '0 Days' 到 '11 Days'
        expected = timedelta_range("0 Days", "11 Days")
        # 断言索引 res 与期望的时间差范围对象 expected 相等
        tm.assert_index_equal(res, expected)
        # 断言结果的频率与期望的频率相同
        assert res.freq == expected.freq

    # 标记为预期失败的测试用例，因为继承的频率信息不正确
    @pytest.mark.xfail(
        reason="The inherited freq is incorrect bc dti.freq is incorrect "
        "https://github.com/pandas-dev/pandas/pull/48818/files#r982793461"
    )
    def test_sub_datetime_preserves_freq_across_dst(self):
        # 创建一个具有时区信息的 Timestamp 对象 ts，时区为 'US/Pacific'，日期为 '2016-03-11'
        ts = Timestamp("2016-03-11", tz="US/Pacific")
        # 创建一个日期范围对象 dti，从 ts 开始，包含 4 个周期
        dti = date_range(ts, periods=4)
        # 对 dti 减去第一个元素（Timestamp），得到时间差索引 res
        res = dti - dti[0]
        # 期望的时间差索引 expected，包含了几个 Timedelta 对象
        expected = TimedeltaIndex(
            [
                Timedelta(days=0),
                Timedelta(days=1),
                Timedelta(days=2),
                Timedelta(days=2, hours=23),
            ]
        )
        # 断言索引 res 与期望的时间差索引 expected 相等
        tm.assert_index_equal(res, expected)
        # 断言结果的频率与期望的频率相同
        assert res.freq == expected.freq
```