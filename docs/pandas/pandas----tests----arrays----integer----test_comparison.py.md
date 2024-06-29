# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_comparison.py`

```
import pytest  # 导入 pytest 模块

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
import pandas._testing as tm  # 导入 pandas 内部测试工具模块
from pandas.tests.arrays.masked_shared import (  # 从指定路径导入以下类
    ComparisonOps,  # 比较操作类
    NumericOps,  # 数值操作类
)


class TestComparisonOps(NumericOps, ComparisonOps):
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1, 0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_to_int(self, dtype, comparison_op):
        # GH 28930
        # 根据传入的比较操作函数获取其名称
        op_name = f"__{comparison_op.__name__}__"
        # 创建包含整数和缺失值的 Series 对象 s1
        s1 = pd.Series([1, None, 3], dtype=dtype)
        # 创建包含浮点数的 Series 对象 s2
        s2 = pd.Series([1, None, 3], dtype="float")

        # 获取 s1 中指定比较操作函数的方法
        method = getattr(s1, op_name)
        # 对 s1 应用指定比较操作函数并获取结果
        result = method(2)

        # 获取 s2 中指定比较操作函数的方法
        method = getattr(s2, op_name)
        # 对 s2 应用指定比较操作函数并获取期望结果，并将其转换为布尔型
        expected = method(2).astype("boolean")
        # 将 s2 中缺失值位置设为 pd.NA
        expected[s2.isna()] = pd.NA

        # 使用测试工具模块 tm 来断言 Series 对象 result 与 expected 相等
        tm.assert_series_equal(result, expected)


def test_equals():
    # GH-30652
    # equals 方法通常在 /tests/extension/base/methods 中进行测试，但这里特别测试
    # 当两个相同类别但不同数据类型的数组不相等时
    a1 = pd.array([1, 2, None], dtype="Int64")
    a2 = pd.array([1, 2, None], dtype="Int32")
    # 断言 a1 不等于 a2
    assert a1.equals(a2) is False
```