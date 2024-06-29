# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_drop_duplicates.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下对象：
    PeriodIndex,  # 时间周期索引对象
    Series,  # 数据序列对象
    date_range,  # 生成日期范围的函数
    period_range,  # 生成时间周期范围的函数
    timedelta_range,  # 生成时间间隔范围的函数
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块

class DropDuplicates:
    def test_drop_duplicates_metadata(self, idx):
        # GH#10115
        # 调用 drop_duplicates 方法去除重复项
        result = idx.drop_duplicates()
        # 使用 assert_index_equal 断言确保 idx 和 result 相等
        tm.assert_index_equal(idx, result)
        # 断言 idx 和 result 的频率相同
        assert idx.freq == result.freq

        # 创建一个包含重复索引的新索引 idx_dup
        idx_dup = idx.append(idx)
        # 再次调用 drop_duplicates 方法去除重复项
        result = idx_dup.drop_duplicates()

        expected = idx
        if not isinstance(idx, PeriodIndex):
            # 对于非 PeriodIndex，重置频率为 None
            assert idx_dup.freq is None
            assert result.freq is None
            # 重置 expected 的频率为 None
            expected = idx._with_freq(None)
        else:
            # 对于 PeriodIndex，断言结果的频率与期望的相同
            assert result.freq == expected.freq

        # 使用 assert_index_equal 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "keep, expected, index",
        [
            (
                "first",
                np.concatenate(([False] * 10, [True] * 5)),
                np.arange(0, 10, dtype=np.int64),
            ),
            (
                "last",
                np.concatenate(([True] * 5, [False] * 10)),
                np.arange(5, 15, dtype=np.int64),
            ),
            (
                False,
                np.concatenate(([True] * 5, [False] * 5, [True] * 5)),
                np.arange(5, 10, dtype=np.int64),
            ),
        ],
    )
    def test_drop_duplicates(self, keep, expected, index, idx):
        # 检查 Index/Series 的兼容性
        idx = idx.append(idx[:5])

        # 使用 assert_numpy_array_equal 断言 idx.duplicated(keep=keep) 和 expected 相等
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected)
        # 创建一个不包含重复项的期望结果
        expected = idx[~expected]

        # 调用 drop_duplicates 方法去除重复项
        result = idx.drop_duplicates(keep=keep)
        # 使用 assert_index_equal 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)

        # 将 idx 转换为 Series，并调用 drop_duplicates 方法去除重复项
        result = Series(idx).drop_duplicates(keep=keep)
        # 创建一个期望的 Series 对象
        expected = Series(expected, index=index)
        # 使用 assert_series_equal 断言 result 和 expected 相等
        tm.assert_series_equal(result, expected)


class TestDropDuplicatesPeriodIndex(DropDuplicates):
    @pytest.fixture(params=["D", "3D", "h", "2h", "min", "2min", "s", "3s"])
    def freq(self, request):
        # 使用 pytest.fixture 注入频率参数
        return request.param

    @pytest.fixture
    def idx(self, freq):
        # 使用 period_range 创建一个 PeriodIndex 索引对象
        return period_range("2011-01-01", periods=10, freq=freq, name="idx")


class TestDropDuplicatesDatetimeIndex(DropDuplicates):
    @pytest.fixture
    def idx(self, freq_sample):
        # 使用 date_range 创建一个 DatetimeIndex 索引对象
        return date_range("2011-01-01", freq=freq_sample, periods=10, name="idx")


class TestDropDuplicatesTimedeltaIndex(DropDuplicates):
    @pytest.fixture
    def idx(self, freq_sample):
        # 使用 timedelta_range 创建一个 TimedeltaIndex 索引对象
        return timedelta_range("1 day", periods=10, freq=freq_sample, name="idx")
```