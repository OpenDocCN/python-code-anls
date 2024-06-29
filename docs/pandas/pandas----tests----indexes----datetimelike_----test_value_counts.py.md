# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_value_counts.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 导入 Pandas 库中的以下模块：
    DatetimeIndex,  # 日期时间索引
    NaT,  # 表示缺失日期时间的常量
    PeriodIndex,  # 时期索引
    Series,  # 序列数据结构
    TimedeltaIndex,  # 时间增量索引
    date_range,  # 创建日期范围
    period_range,  # 创建时期范围
    timedelta_range,  # 创建时间增量范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestValueCounts:
    # GH#7735

    def test_value_counts_unique_datetimeindex(self, tz_naive_fixture):
        tz = tz_naive_fixture  # 从测试夹具中获取时区信息
        orig = date_range("2011-01-01 09:00", freq="h", periods=10, tz=tz)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_timedeltaindex(self):
        orig = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_periodindex(self):
        orig = period_range("2011-01-01 09:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def _check_value_counts_with_repeats(self, orig):
        # 创建重复值索引，第'n'个元素重复出现n+1次
        idx = type(orig)(  # 使用原始索引类型构建索引对象
            np.repeat(orig._values, range(1, len(orig) + 1)), dtype=orig.dtype
        )

        exp_idx = orig[::-1]  # 将原始索引逆序排列
        if not isinstance(exp_idx, PeriodIndex):
            exp_idx = exp_idx._with_freq(None)  # 如果不是时期索引，则移除频率信息

        expected = Series(range(10, 0, -1), index=exp_idx, dtype="int64", name="count")  # 期望的结果序列，反向索引

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)  # 断言序列的值计数结果与期望一致

        tm.assert_index_equal(idx.unique(), orig)  # 断言索引的唯一值与原始索引一致

    def test_value_counts_unique_datetimeindex2(self, tz_naive_fixture):
        tz = tz_naive_fixture  # 从测试夹具中获取时区信息
        idx = DatetimeIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,  # 表示缺失的日期时间
            ],
            tz=tz,  # 指定时区信息
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_timedeltaindex2(self):
        idx = TimedeltaIndex(
            [
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 08:00:00",
                "1 days 08:00:00",
                NaT,  # 表示缺失的时间增量
            ]
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_periodindex2(self):
        idx = PeriodIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,  # 表示缺失的时期
            ],
            freq="h",  # 指定时期的频率
        )
        self._check_value_counts_dropna(idx)
    # 定义一个方法用于检查值计数的情况，并在不包含 NaN 值时执行操作
    def _check_value_counts_dropna(self, idx):
        # 从 idx 中选择索引为 2 和 3 的元素，创建一个新的索引对象
        exp_idx = idx[[2, 3]]
        # 创建一个预期的 Series 对象，用于比较值计数结果
        expected = Series([3, 2], index=exp_idx, name="count")

        # 对于 idx 和其转换为 Series 后的对象，分别比较其值计数是否与预期相同
        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)

        # 更新 exp_idx，包含索引为 2、3 和最后一个元素的索引
        exp_idx = idx[[2, 3, -1]]
        # 创建一个包含 NaN 值的预期 Series 对象，用于比较值计数结果
        expected = Series([3, 2, 1], index=exp_idx, name="count")

        # 对于 idx 和其转换为 Series 后的对象，分别比较其值计数（包含 NaN 值）是否与预期相同
        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(dropna=False), expected)

        # 检查 idx 的唯一值是否与更新后的 exp_idx 相等
        tm.assert_index_equal(idx.unique(), exp_idx)
```