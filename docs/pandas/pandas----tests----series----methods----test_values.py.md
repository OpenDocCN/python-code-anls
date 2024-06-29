# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_values.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库，并用 np 别名表示
import pytest  # 导入 pytest 测试框架

# 从 pandas 库中导入以下模块：
from pandas import (
    IntervalIndex,  # 导入 IntervalIndex 类
    Series,  # 导入 Series 类
    period_range,  # 导入 period_range 函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 pandas._testing

# 定义测试类 TestValues
class TestValues:
    # 使用 pytest.mark.parametrize 装饰器进行参数化测试
    @pytest.mark.parametrize(
        "data",
        [
            period_range("2000", periods=4),  # 创建一个包含 2000 年起 4 个周期的日期范围
            IntervalIndex.from_breaks([1, 2, 3, 4]),  # 创建一个间隔索引对象，从给定的断点列表创建
        ],
    )
    # 定义测试方法 test_values_object_extension_dtypes，参数化输入 data
    def test_values_object_extension_dtypes(self, data):
        # 调用 Series 构造函数，将 data 转换为 Series 对象，并获取其 values 属性
        result = Series(data).values
        # 将 data 转换为对象类型的 NumPy 数组，并赋值给 expected
        expected = np.array(data.astype(object))
        # 使用 pandas._testing 模块中的 assert_numpy_array_equal 函数，断言 result 和 expected 相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法 test_values，接收 datetime_series 参数
    def test_values(self, datetime_series):
        # 使用 pandas._testing 模块中的 assert_almost_equal 函数，
        # 检查 datetime_series 的 values 属性与 list(datetime_series) 的值是否接近，但不检查数据类型
        tm.assert_almost_equal(
            datetime_series.values, list(datetime_series), check_dtype=False
        )
```