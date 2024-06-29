# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_categorical.py`

```
# 导入 numpy 库，用于数值计算
import numpy as np

# 从 pandas 库中导入以下两个类
from pandas import (
    Categorical,  # 用于处理分类数据的类
    Series,       # 用于表示一维数据的类
)

# 导入 pandas 内部测试模块
import pandas._testing as tm


# 定义一个测试类 TestCategoricalComparisons，用于比较分类数据的测试
class TestCategoricalComparisons:
    
    # 定义测试方法 test_categorical_nan_equality，测试分类数据与 NaN 的相等性
    def test_categorical_nan_equality(self):
        # 创建一个包含分类数据和 NaN 的 Series 对象
        cat = Series(Categorical(["a", "b", "c", np.nan]))
        # 创建预期的结果 Series 对象，判断分类数据与自身的相等性
        expected = Series([True, True, True, False])
        # 执行比较操作，判断分类数据与自身的相等性
        result = cat == cat
        # 使用测试模块中的方法验证结果是否与预期相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法 test_categorical_tuple_equality，测试分类数据与元组的相等性
    def test_categorical_tuple_equality(self):
        # 创建一个包含元组数据的 Series 对象
        ser = Series([(0, 0), (0, 1), (0, 0), (1, 0), (1, 1)])
        # 创建预期的结果 Series 对象，判断元组数据与 (0, 0) 的相等性
        expected = Series([True, False, True, False, False])
        # 执行比较操作，判断元组数据与 (0, 0) 的相等性
        result = ser == (0, 0)
        # 使用测试模块中的方法验证结果是否与预期相等
        tm.assert_series_equal(result, expected)

        # 将 Series 对象转换为分类数据后，再次测试其与 (0, 0) 的相等性
        result = ser.astype("category") == (0, 0)
        # 使用测试模块中的方法验证结果是否与预期相等
        tm.assert_series_equal(result, expected)
```