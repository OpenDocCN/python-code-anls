# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_comparison.py`

```
# 导入 NumPy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 Pandas 库，并使用 pd 别名
import pandas as pd
# 导入 Pandas 内部测试模块
import pandas._testing as tm
# 从 Pandas 的 arrays 模块导入 BooleanArray 类
from pandas.arrays import BooleanArray
# 导入 Pandas 测试模块中的 masked_shared 模块下的 ComparisonOps 类
from pandas.tests.arrays.masked_shared import ComparisonOps


@pytest.fixture
def data():
    """Fixture returning boolean array with valid and missing data"""
    # 返回包含有效和缺失数据的布尔数组
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def dtype():
    """Fixture returning BooleanDtype"""
    # 返回 BooleanDtype 类型对象作为 fixture
    return pd.BooleanDtype()


class TestComparisonOps(ComparisonOps):
    def test_compare_scalar(self, data, comparison_op):
        # 调用基类方法 _compare_other，比较数据与标量值
        self._compare_other(data, comparison_op, True)

    def test_compare_array(self, data, comparison_op):
        # 准备不同类型的数组作为比较对象，调用基类方法 _compare_other 进行比较
        other = pd.array([True] * len(data), dtype="boolean")
        self._compare_other(data, comparison_op, other)
        other = np.array([True] * len(data))
        self._compare_other(data, comparison_op, other)
        other = pd.Series([True] * len(data))
        self._compare_other(data, comparison_op, other)

    @pytest.mark.parametrize("other", [True, False, pd.NA])
    def test_scalar(self, other, comparison_op, dtype):
        # 调用基类 ComparisonOps 的 test_scalar 方法，测试标量值的比较操作
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_array(self, comparison_op):
        # 准备两个布尔数组 a 和 b 进行比较操作，验证结果与预期的 ExtensionArray 相等
        op = comparison_op
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")

        result = op(a, b)

        # 计算操作后的值和掩码，并创建预期的 ExtensionArray
        values = op(a._data, b._data)
        mask = a._mask | b._mask
        expected = BooleanArray(values, mask)
        
        # 使用 Pandas 测试模块中的 assert_extension_array_equal 方法验证结果
        tm.assert_extension_array_equal(result, expected)

        # 确保没有在原地改变任何东西
        result[0] = None
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )
```