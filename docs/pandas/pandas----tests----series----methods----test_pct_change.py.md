# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_pct_change.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下模块：
    Series,  # Series 数据结构，用于处理一维数组
    date_range,  # 用于生成时间序列
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestSeriesPctChange:
    def test_pct_change(self, datetime_series):
        rs = datetime_series.pct_change()  # 计算时间序列的变化率
        tm.assert_series_equal(rs, datetime_series / datetime_series.shift(1) - 1)

        rs = datetime_series.pct_change(2)  # 计算时间序列每两个元素之间的变化率
        filled = datetime_series.ffill()  # 使用前向填充缺失值
        tm.assert_series_equal(rs, filled / filled.shift(2) - 1)

        rs = datetime_series.pct_change(freq="5D")  # 根据5天的频率计算时间序列的变化率
        filled = datetime_series.ffill()  # 使用前向填充缺失值
        tm.assert_series_equal(
            rs, (filled / filled.shift(freq="5D") - 1).reindex_like(filled)
        )

    def test_pct_change_with_duplicate_axis(self):
        # GH#28664
        common_idx = date_range("2019-11-14", periods=5, freq="D")  # 创建日期范围
        result = Series(range(5), common_idx).pct_change(freq="B")  # 计算百分比变化

        # the reason that the expected should be like this is documented at PR 28681
        expected = Series([np.nan, np.inf, np.nan, np.nan, 3.0], common_idx)  # 预期的结果序列

        tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等

    def test_pct_change_shift_over_nas(self):
        s = Series([1.0, 1.5, np.nan, 2.5, 3.0])  # 创建包含 NaN 的 Series
        chg = s.pct_change()  # 计算序列的变化率
        expected = Series([np.nan, 0.5, np.nan, np.nan, 0.2])  # 预期的结果序列
        tm.assert_series_equal(chg, expected)  # 断言结果序列与预期序列相等

    @pytest.mark.parametrize("freq, periods", [("5B", 5), ("3B", 3), ("14B", 14)])
    def test_pct_change_periods_freq(self, freq, periods, datetime_series):
        # GH#7292
        rs_freq = datetime_series.pct_change(freq=freq)  # 根据频率计算时间序列的变化率
        rs_periods = datetime_series.pct_change(periods)  # 根据指定期数计算时间序列的变化率
        tm.assert_series_equal(rs_freq, rs_periods)  # 断言两种计算方式得到的结果序列相等

        empty_ts = Series(index=datetime_series.index, dtype=object)
        rs_freq = empty_ts.pct_change(freq=freq)  # 空 Series 根据频率计算变化率
        rs_periods = empty_ts.pct_change(periods)  # 空 Series 根据指定期数计算变化率
        tm.assert_series_equal(rs_freq, rs_periods)  # 断言两种计算方式得到的结果序列相等


def test_pct_change_with_duplicated_indices():
    # GH30463
    s = Series([np.nan, 1, 2, 3, 9, 18], index=["a", "b"] * 3)  # 创建包含重复索引的 Series
    result = s.pct_change()  # 计算序列的变化率
    expected = Series([np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], index=["a", "b"] * 3)  # 预期的结果序列
    tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等


def test_pct_change_no_warning_na_beginning():
    # GH#54981
    ser = Series([None, None, 1, 2, 3])  # 创建包含 NaN 的 Series
    result = ser.pct_change()  # 计算序列的变化率
    expected = Series([np.nan, np.nan, np.nan, 1, 0.5])  # 预期的结果序列
    tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等


def test_pct_change_empty():
    # GH 57056
    ser = Series([], dtype="float64")  # 创建空的 float64 类型的 Series
    expected = ser.copy()  # 复制空 Series 作为预期结果
    result = ser.pct_change(periods=0)  # 计算空 Series 的变化率（期数为0）
    tm.assert_series_equal(expected, result)  # 断言结果序列与预期序列相等
```