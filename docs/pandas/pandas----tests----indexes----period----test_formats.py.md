# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_formats.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库并从中导入特定模块
import pandas as pd
from pandas import (
    PeriodIndex,
    Series,
)
# 导入 pandas 内部测试模块
import pandas._testing as tm


# 定义测试函数，用于测试 PeriodIndex 的 CSV 导出功能
def test_get_values_for_csv():
    # 创建 PeriodIndex 对象，包含三个日期字符串，频率为每日
    index = PeriodIndex(["2017-01-01", "2017-01-02", "2017-01-03"], freq="D")

    # 第一个测试，不带参数调用 _get_values_for_csv 方法
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv()
    tm.assert_numpy_array_equal(result, expected)

    # 第二个测试，设置 na_rep 参数为 "pandas"，检查处理 NaN 值的情况
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # 第三个测试，设置 date_format 参数为 "%m-%Y-%d"，检查日期格式化是否有效
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype=object)
    result = index._get_values_for_csv(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # 第四个测试，设置其中一个值为 NaT（缺失日期），检查 na_rep 参数为 "NaT" 的处理
    index = PeriodIndex(["2017-01-01", pd.NaT, "2017-01-03"], freq="D")
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv(na_rep="NaT")
    tm.assert_numpy_array_equal(result, expected)

    # 第五个测试，设置 na_rep 参数为 "pandas"，检查处理其他类型的 NaN 值
    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)


# 定义测试类 TestPeriodIndexRendering，用于测试 PeriodIndex 对象的字符串表示方法
class TestPeriodIndexRendering:
    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    # 定义一个测试方法，用于验证 PeriodIndex 对象的字符串表示形式是否正确
    def test_representation(self, method):
        # 创建空的 PeriodIndex 对象，频率为每日
        idx1 = PeriodIndex([], freq="D")
        # 创建包含一个日期的 PeriodIndex 对象，频率为每日
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        # 创建包含两个日期的 PeriodIndex 对象，频率为每日
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        # 创建包含三个日期的 PeriodIndex 对象，频率为每日
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        # 创建包含三个年份的 PeriodIndex 对象，频率为每年（12月结束）
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        # 创建包含三个日期时间的 PeriodIndex 对象，频率为每小时
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")
        # 创建包含一个季度的 PeriodIndex 对象，从2013年第一季度开始，频率为每季度（12月结束）
        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        # 创建包含两个季度的 PeriodIndex 对象，从2013年第一季度开始，频率为每季度（12月结束）
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        # 创建包含三个季度的 PeriodIndex 对象，从2013年第一季度开始，频率为每季度（12月结束）
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")
        # 创建包含两个日期的 PeriodIndex 对象，频率为每3天
        idx10 = PeriodIndex(["2011-01-01", "2011-02-01"], freq="3D")

        # 期望的字符串表示形式，用于后续验证
        exp1 = "PeriodIndex([], dtype='period[D]')"
        exp2 = "PeriodIndex(['2011-01-01'], dtype='period[D]')"
        exp3 = "PeriodIndex(['2011-01-01', '2011-01-02'], dtype='period[D]')"
        exp4 = (
            "PeriodIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='period[D]')"
        )
        exp5 = "PeriodIndex(['2011', '2012', '2013'], dtype='period[Y-DEC]')"
        exp6 = (
            "PeriodIndex(['2011-01-01 09:00', '2012-02-01 10:00', 'NaT'], "
            "dtype='period[h]')"
        )
        exp7 = "PeriodIndex(['2013Q1'], dtype='period[Q-DEC]')"
        exp8 = "PeriodIndex(['2013Q1', '2013Q2'], dtype='period[Q-DEC]')"
        exp9 = "PeriodIndex(['2013Q1', '2013Q2', '2013Q3'], dtype='period[Q-DEC]')"
        exp10 = "PeriodIndex(['2011-01-01', '2011-02-01'], dtype='period[3D]')"

        # 遍历每个 PeriodIndex 对象及其对应的期望字符串表示形式，验证结果是否符合期望
        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10],
        ):
            # 调用传入的方法获取实际的字符串表示形式
            result = getattr(idx, method)()
            # 使用断言验证实际结果与期望结果是否一致
            assert result == expected

    # TODO: These are Series.__repr__ tests
    # 定义一个测试方法，用于验证 Series 对象的字符串表示形式是否正确
    def test_representation_to_series(self):
        # 创建不同情况下的 PeriodIndex 对象
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")
        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        # 期望的 Series 对象的字符串表示形式
        exp1 = """Series([], dtype: period[D])"""
        exp2 = """0    2011-01-01
dtype: period[D]"""
    def test_summary(self):
        # GH#9116
        # 创建不同的 PeriodIndex 对象，以便进行测试
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")

        # 创建指定频率的 PeriodIndex 对象
        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        # 期望的输出结果
        exp1 = """PeriodIndex: 0 entries
Freq: D"""

        exp2 = """PeriodIndex: 1 entries, 2011-01-01 to 2011-01-01
Freq: D"""

        exp3 = """PeriodIndex: 2 entries, 2011-01-01 to 2011-01-02
Freq: D"""

        exp4 = """PeriodIndex: 3 entries, 2011-01-01 to 2011-01-03
Freq: D"""

        exp5 = """PeriodIndex: 3 entries, 2011 to 2013
Freq: Y-DEC"""

        exp6 = """PeriodIndex: 3 entries, 2011-01-01 09:00 to NaT
Freq: h"""

        exp7 = """PeriodIndex: 1 entries, 2013Q1 to 2013Q1
Freq: Q-DEC"""

        exp8 = """PeriodIndex: 2 entries, 2013Q1 to 2013Q2
Freq: Q-DEC"""

        exp9 = """PeriodIndex: 3 entries, 2013Q1 to 2013Q3
Freq: Q-DEC"""

        # 遍历每个期望输出和对应的 PeriodIndex 对象，进行断言比较
        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9],
        ):
            result = idx._summary()
            assert result == expected
```