# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_between.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下几个模块和函数
    Series,  # pandas 的 Series 数据结构，用于处理一维数据
    bdate_range,  # 生成工作日日期范围
    date_range,  # 生成日期范围
    period_range,  # 生成时期范围
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestBetween:  # 定义一个测试类 TestBetween

    def test_between(self):  # 定义测试方法 test_between
        series = Series(date_range("1/1/2000", periods=10))  # 创建一个日期范围的 Series 对象
        left, right = series[[2, 7]]  # 获取索引为 2 和 7 的两个元素

        result = series.between(left, right)  # 调用 Series 的 between 方法，获取指定范围内的数据
        expected = (series >= left) & (series <= right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

    def test_between_datetime_object_dtype(self):  # 定义测试方法 test_between_datetime_object_dtype
        ser = Series(bdate_range("1/1/2000", periods=20), dtype=object)  # 创建包含工作日日期范围的 Series 对象，数据类型为 object
        ser[::2] = np.nan  # 将偶数位置的值设置为 NaN

        result = ser[ser.between(ser[3], ser[17])]  # 使用 between 方法获取指定范围内的数据
        expected = ser[3:18].dropna()  # 期望的结果，去掉 NaN 值的范围内数据
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

        result = ser[ser.between(ser[3], ser[17], inclusive="neither")]  # 使用 inclusive 参数获取指定范围内的数据，不包括边界
        expected = ser[5:16].dropna()  # 期望的结果，去掉 NaN 值的范围内数据
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

    def test_between_period_values(self):  # 定义测试方法 test_between_period_values
        ser = Series(period_range("2000-01-01", periods=10, freq="D"))  # 创建一个时期范围的 Series 对象
        left, right = ser[[2, 7]]  # 获取索引为 2 和 7 的两个元素

        result = ser.between(left, right)  # 调用 Series 的 between 方法，获取指定范围内的数据
        expected = (ser >= left) & (ser <= right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

    def test_between_inclusive_string(self):  # 定义测试方法 test_between_inclusive_string
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))  # 创建一个日期范围的 Series 对象
        left, right = series[[2, 7]]  # 获取索引为 2 和 7 的两个元素

        result = series.between(left, right, inclusive="both")  # 使用 inclusive 参数获取指定范围内的数据，包括边界
        expected = (series >= left) & (series <= right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

        result = series.between(left, right, inclusive="left")  # 使用 inclusive 参数获取指定范围内的数据，包括左边界
        expected = (series >= left) & (series < right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

        result = series.between(left, right, inclusive="right")  # 使用 inclusive 参数获取指定范围内的数据，包括右边界
        expected = (series > left) & (series <= right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

        result = series.between(left, right, inclusive="neither")  # 使用 inclusive 参数获取指定范围内的数据，不包括边界
        expected = (series > left) & (series < right)  # 期望的结果，返回范围内的布尔值 Series
        tm.assert_series_equal(result, expected)  # 断言两个 Series 相等

    @pytest.mark.parametrize("inclusive", ["yes", True, False])  # 使用 pytest.mark.parametrize 进行参数化测试
    def test_between_error_args(self, inclusive):  # 定义测试方法 test_between_error_args
        # GH 40628
        series = Series(date_range("1/1/2000", periods=10))  # 创建一个日期范围的 Series 对象
        left, right = series[[2, 7]]  # 获取索引为 2 和 7 的两个元素

        value_error_msg = (
            "Inclusive has to be either string of 'both',"
            "'left', 'right', or 'neither'."
        )

        series = Series(date_range("1/1/2000", periods=10))
        with pytest.raises(ValueError, match=value_error_msg):  # 断言抛出 ValueError 异常，匹配异常消息
            series.between(left, right, inclusive=inclusive)  # 调用 Series 的 between 方法进行测试
```