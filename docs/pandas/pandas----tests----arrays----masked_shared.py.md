# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked_shared.py`

```
"""
Tests shared by MaskedArray subclasses.
"""

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据处理
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.extension.base import BaseOpsUtil  # 从 Pandas 测试中导入基础操作工具类


class ComparisonOps(BaseOpsUtil):
    def _compare_other(self, data, op, other):
        # array
        result = pd.Series(op(data, other))  # 对 data 执行操作 op，并封装成 Pandas Series
        expected = pd.Series(op(data._data, other), dtype="boolean")  # 对 data._data 执行操作 op，并指定数据类型为布尔型

        # 填充 NaN 位置
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)  # 使用测试模块验证结果的一致性

        # series
        ser = pd.Series(data)  # 将 data 转换为 Pandas Series
        result = op(ser, other)  # 对 Pandas Series 执行操作 op

        # 在此设置可空数据类型，以避免在设置为 pd.NA 时的类型转换
        expected = op(pd.Series(data._data), other).astype("boolean")

        # 填充 NaN 位置
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)  # 使用测试模块验证结果的一致性

    # subclass will override to parametrize 'other'
    def test_scalar(self, other, comparison_op, dtype):
        op = comparison_op
        left = pd.array([1, 0, None], dtype=dtype)

        result = op(left, other)

        if other is pd.NA:
            expected = pd.array([None, None, None], dtype="boolean")
        else:
            values = op(left._data, other)
            expected = pd.arrays.BooleanArray(values, left._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([1, 0, None], dtype=dtype))


class NumericOps:
    # Shared by IntegerArray and FloatingArray, not BooleanArray

    def test_searchsorted_nan(self, dtype):
        # The base class casts to object dtype, for which searchsorted returns
        #  0 from the left and 10 from the right.
        arr = pd.array(range(10), dtype=dtype)

        assert arr.searchsorted(np.nan, side="left") == 10
        assert arr.searchsorted(np.nan, side="right") == 10

    def test_no_shared_mask(self, data):
        result = data + 1
        assert not tm.shares_memory(result, data)

    def test_array(self, comparison_op, dtype):
        op = comparison_op

        left = pd.array([0, 1, 2, None, None, None], dtype=dtype)
        right = pd.array([0, 1, None, 0, 1, None], dtype=dtype)

        result = op(left, right)
        values = op(left._data, right._data)
        mask = left._mask | right._mask

        expected = pd.arrays.BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(
            left, pd.array([0, 1, 2, None, None, None], dtype=dtype)
        )
        tm.assert_extension_array_equal(
            right, pd.array([0, 1, None, 0, 1, None], dtype=dtype)
        )
    def test_compare_with_booleanarray(self, comparison_op, dtype):
        # 将传入的比较操作符保存到变量 op 中
        op = comparison_op

        # 创建一个布尔类型的 Pandas 数组 left，包含 True、False 和 None，重复三次
        left = pd.array([True, False, None] * 3, dtype="boolean")
        
        # 创建一个包含整数和 None 的 Pandas 数组 right，重复三次，数据类型由参数 dtype 指定
        right = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype=dtype)
        
        # 创建一个布尔类型的 Pandas 数组 other，包含 False、True 和 None，重复三次
        other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype="boolean")

        # 计算 left 与 other 的期望结果
        expected = op(left, other)
        
        # 计算 left 与 right 的结果
        result = op(left, right)
        
        # 断言计算结果与期望结果相等
        tm.assert_extension_array_equal(result, expected)

        # 对换操作数顺序进行比较
        expected = op(other, left)
        result = op(right, left)
        
        # 断言计算结果与期望结果相等
        tm.assert_extension_array_equal(result, expected)

    def test_compare_to_string(self, dtype):
        # GH#28930: 测试与字符串比较
        ser = pd.Series([1, None], dtype=dtype)
        
        # 执行与字符串 "a" 的比较
        result = ser == "a"
        
        # 期望结果是一个包含 False 和 pd.NA 的布尔类型的 Pandas Series
        expected = pd.Series([False, pd.NA], dtype="boolean")
        
        # 断言计算结果与期望结果相等
        tm.assert_series_equal(result, expected)

    def test_ufunc_with_out(self, dtype):
        # 使用 ufunc 操作并指定输出参数

        # 创建一个 Pandas 数组 arr
        arr = pd.array([1, 2, 3], dtype=dtype)
        
        # 创建另一个 Pandas 数组 arr2，包含整数和 pd.NA，数据类型由参数 dtype 指定
        arr2 = pd.array([1, 2, pd.NA], dtype=dtype)

        # 创建一个布尔掩码 mask，对 arr 中的每个元素执行等于比较操作
        mask = arr == arr
        
        # 创建另一个布尔掩码 mask2，对 arr2 中的每个元素执行等于比较操作
        mask2 = arr2 == arr2

        # 创建一个全为 False 的布尔类型的 NumPy 数组 result
        result = np.zeros(3, dtype=bool)
        
        # 将 mask 的值逻辑或运算到 result 中
        result |= mask
        
        # 断言 result 的类型为 ndarray
        assert isinstance(result, np.ndarray)
        
        # 断言 result 中所有元素为 True
        assert result.all()

        # 创建一个全为 False 的布尔类型的 NumPy 数组 result
        result = np.zeros(3, dtype=bool)
        
        # 尝试将 mask2 的值逻辑或运算到 result 中，这会引发 ValueError
        msg = "Specify an appropriate 'na_value' for this dtype"
        with pytest.raises(ValueError, match=msg):
            result |= mask2

        # 使用 ufunc 进行加法操作，将 arr 和 arr2 相加
        res = np.add(arr, arr2)
        
        # 期望结果是一个包含整数和 pd.NA 的 Pandas 数组
        expected = pd.array([2, 4, pd.NA], dtype=dtype)
        
        # 断言计算结果与期望结果相等
        tm.assert_extension_array_equal(res, expected)

        # 使用 ufunc 进行加法操作，将 arr 和 arr2 相加，并指定输出参数 out=arr
        res = np.add(arr, arr2, out=arr)
        
        # 断言计算结果与期望结果相等
        assert res is arr
        
        # 断言 arr 的内容与期望结果相等
        tm.assert_extension_array_equal(res, expected)
        tm.assert_extension_array_equal(arr, expected)

    def test_mul_td64_array(self, dtype):
        # GH#45622: 测试 Timedelta64 类型数组的乘法

        # 创建一个 Pandas 数组 arr，包含整数和 pd.NA，数据类型由参数 dtype 指定
        arr = pd.array([1, 2, pd.NA], dtype=dtype)
        
        # 创建一个 NumPy int64 类型的数组 other，转换为 Timedelta64 类型
        other = np.arange(3, dtype=np.int64).view("m8[ns]")

        # 执行 arr 与 other 的乘法操作
        result = arr * other
        
        # 期望结果是一个包含 Timedelta64 类型的 Pandas 数组
        expected = pd.array([pd.Timedelta(0), pd.Timedelta(2), pd.NaT])
        
        # 断言计算结果与期望结果相等
        tm.assert_extension_array_equal(result, expected)
```