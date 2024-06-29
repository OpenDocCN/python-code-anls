# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_comparison.py`

```
import numpy as np
import pytest  # 导入 pytest 模块

import pandas as pd  # 导入 pandas 库，并简称为 pd
import pandas._testing as tm  # 导入 pandas 内部测试模块
from pandas.core.arrays import FloatingArray  # 从 pandas 核心数组中导入 FloatingArray 类
from pandas.tests.arrays.masked_shared import (  # 从 pandas 测试数组的共享掩码模块中导入以下类
    ComparisonOps,
    NumericOps,
)


class TestComparisonOps(NumericOps, ComparisonOps):  # 定义一个测试类，继承 NumericOps 和 ComparisonOps
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1.0, 0.0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        # 使用 pytest 的参数化标记定义测试函数参数，测试标量操作
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_with_integerarray(self, comparison_op):
        # 测试使用整数数组进行比较操作
        op = comparison_op  # 为比较操作赋值
        a = pd.array([0, 1, None] * 3, dtype="Int64")  # 创建整数类型数组 a
        b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype="Float64")  # 创建浮点类型数组 b
        other = b.astype("Int64")  # 将数组 b 转换为整数类型，并赋值给 other
        expected = op(a, other)  # 使用 op 操作符比较 a 和 other 的预期结果
        result = op(a, b)  # 使用 op 操作符比较 a 和 b 的结果
        tm.assert_extension_array_equal(result, expected)  # 使用测试模块验证结果的一致性
        expected = op(other, a)  # 使用 op 操作符比较 other 和 a 的预期结果
        result = op(b, a)  # 使用 op 操作符比较 b 和 a 的结果
        tm.assert_extension_array_equal(result, expected)  # 使用测试模块验证结果的一致性


def test_equals():
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    # 测试 equals 方法的功能，验证不同 dtype 的相同类别数组不相等
    a1 = pd.array([1, 2, None], dtype="Float64")  # 创建 Float64 类型数组 a1
    a2 = pd.array([1, 2, None], dtype="Float32")  # 创建 Float32 类型数组 a2
    assert a1.equals(a2) is False  # 断言 a1 和 a2 不相等


def test_equals_nan_vs_na():
    # GH#44382
    # 测试 equals 方法的功能，验证 FloatingArray 对象的相等性
    mask = np.zeros(3, dtype=bool)  # 创建一个长度为 3 的布尔类型全零数组 mask
    data = np.array([1.0, np.nan, 3.0], dtype=np.float64)  # 创建一个浮点类型数组 data

    left = FloatingArray(data, mask)  # 创建一个 FloatingArray 对象 left
    assert left.equals(left)  # 断言 left 等于自身
    tm.assert_extension_array_equal(left, left)  # 使用测试模块验证 left 和自身的一致性

    assert left.equals(left.copy())  # 断言 left 等于其复制对象
    assert left.equals(FloatingArray(data.copy(), mask.copy()))  # 断言 left 等于另一个相同数据和掩码的对象

    mask2 = np.array([False, True, False], dtype=bool)  # 创建另一个布尔类型数组 mask2
    data2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 创建另一个浮点类型数组 data2
    right = FloatingArray(data2, mask2)  # 创建一个新的 FloatingArray 对象 right
    assert right.equals(right)  # 断言 right 等于自身
    tm.assert_extension_array_equal(right, right)  # 使用测试模块验证 right 和自身的一致性

    assert not left.equals(right)  # 断言 left 不等于 right

    # 当 mask[1] = True 时，唯一的差异是 data[1]，但对于 equals 方法来说这不应该有影响
    mask[1] = True
    assert left.equals(right)  # 断言 left 等于 right
```