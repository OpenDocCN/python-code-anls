# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_function.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

import pandas as pd  # 导入Pandas库，用于数据处理和分析
import pandas._testing as tm  # 导入Pandas测试模块，用于测试辅助函数


@pytest.mark.parametrize(
    "ufunc", [np.add, np.logical_or, np.logical_and, np.logical_xor]
)
def test_ufuncs_binary(ufunc):
    # 测试二元通用函数（ufunc）对Pandas的BooleanArray的操作

    # 创建一个包含True、False和None的BooleanArray
    a = pd.array([True, False, None], dtype="boolean")
    
    # 对BooleanArray使用ufunc，得到结果result
    result = ufunc(a, a)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(a._data, a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 将BooleanArray转为Series
    s = pd.Series(a)
    
    # 对Series使用ufunc，得到结果result
    result = ufunc(s, a)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.Series(ufunc(a._data, a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_series_equal(result, expected)

    # 创建一个普通的NumPy数组
    arr = np.array([True, True, False])
    
    # 对BooleanArray和NumPy数组使用ufunc，得到结果result
    result = ufunc(a, arr)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(a._data, arr), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 对NumPy数组和BooleanArray使用ufunc，得到结果result
    result = ufunc(arr, a)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(arr, a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 对BooleanArray和标量True使用ufunc，得到结果result
    result = ufunc(a, True)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(a._data, True), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 对标量True和BooleanArray使用ufunc，得到结果result
    result = ufunc(True, a)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(True, a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 测试未处理的操作类型
    msg = r"operand type\(s\) all returned NotImplemented from __array_ufunc__"
    with pytest.raises(TypeError, match=msg):
        # 断言使用ufunc对BooleanArray和字符串"test"执行操作会引发TypeError异常
        ufunc(a, "test")


@pytest.mark.parametrize("ufunc", [np.logical_not])
def test_ufuncs_unary(ufunc):
    # 测试一元通用函数（ufunc）对Pandas的BooleanArray的操作
    
    # 创建一个包含True、False和None的BooleanArray
    a = pd.array([True, False, None], dtype="boolean")
    
    # 对BooleanArray使用ufunc，得到结果result
    result = ufunc(a)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.array(ufunc(a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_extension_array_equal(result, expected)

    # 将BooleanArray转为Series
    ser = pd.Series(a)
    
    # 对Series使用ufunc，得到结果result
    result = ufunc(ser)
    
    # 使用ufunc直接在其数据上进行操作，得到期望的结果expected
    expected = pd.Series(ufunc(a._data), dtype="boolean")
    
    # 将操作结果中对应mask值的位置设为NaN
    expected[a._mask] = np.nan
    
    # 断言result与expected相等
    tm.assert_series_equal(result, expected)


def test_ufunc_numeric():
    # 测试NumPy的sqrt函数对Pandas的BooleanArray的操作
    
    # 创建一个包含True、False和None的BooleanArray
    arr = pd.array([True, False, None], dtype="boolean")

    # 对BooleanArray使用sqrt函数，得到结果res
    res = np.sqrt(arr)

    # 期望的结果是将BooleanArray中的True和False分别开方得到1和0，而None保持不变
    expected = pd.array([1, 0, None], dtype="Float32")
    
    # 断言res与expected相等
    tm.assert_extension_array_equal(res, expected)


@pytest.mark.parametrize("values", [[True, False], [True, None]])
def test_ufunc_reduce_raises(values):
    # 测试ufunc的reduce方法在包含NA值的BooleanArray上的行为
    
    # 创建一个包含True、False和NA的BooleanArray
    arr = pd.array(values, dtype="boolean")

    # 对BooleanArray使用reduce方法，得到结果res
    res = np.add.reduce(arr)
    
    # 如果最后一个元素是NA，则期望的结果是NA，否则是对数据求和
    if arr[-1] is pd.NA:
        expected = pd.NA
    else:
        expected = arr._data.sum()
    
    # 断言res与expected的近似相等
    tm.assert_almost_equal(res, expected)


def test_value_counts_na():
    # 测试BooleanArray的value_counts方法在处理NA值时的行为
    
    # 创建一个包含True、False和NA的BooleanArray
    arr = pd.array([True, False, pd.NA], dtype="boolean")
    
    # 调用value_counts方法，得到结果result
    result = arr.value_counts(dropna=False)
    
    # 期望的结果是统计各个值的出现次数，包括NA
    expected = pd.Series([1, 1, 1], index=arr, dtype="Int64", name="count")
    # 断言确保期望值的索引类型与数组的类型相同
    assert expected.index.dtype == arr.dtype
    
    # 使用 Pandas 的 value_counts 方法计算数组中每个值的频数，忽略缺失值
    result = arr.value_counts(dropna=True)
    
    # 创建期望的 Pandas Series 对象，包括数值、索引、数据类型和名称
    expected = pd.Series([1, 1], index=arr[:-1], dtype="Int64", name="count")
    
    # 再次断言确保期望值的索引类型与数组的类型相同
    assert expected.index.dtype == arr.dtype
    
    # 断言结果和期望值的 Pandas Series 相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试带有归一化参数的值计数功能
def test_value_counts_with_normalize():
    # 创建一个布尔类型的 Pandas Series，包括 True、False 和缺失值 pd.NA
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    # 对该 Series 进行值计数，并进行归一化处理
    result = ser.value_counts(normalize=True)
    # 创建预期结果，期望得到一个名为 proportion 的 Series，其值为计数结果的一半
    expected = pd.Series([1, 1], index=ser[:-1], dtype="Float64", name="proportion") / 2
    # 断言预期结果的索引类型为布尔型
    assert expected.index.dtype == "boolean"
    # 使用测试框架中的函数验证计算结果是否与预期一致
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试差分操作
def test_diff():
    # 创建一个包含布尔值和缺失值的 Pandas array
    a = pd.array(
        [True, True, False, False, True, None, True, None, False], dtype="boolean"
    )
    # 对数组执行差分操作，计算相邻元素的差异
    result = pd.core.algorithms.diff(a, 1)
    # 创建预期结果，其中包含了对应的差分操作结果
    expected = pd.array(
        [None, False, True, False, True, None, None, None, None], dtype="boolean"
    )
    # 使用测试框架中的函数验证计算结果是否与预期一致
    tm.assert_extension_array_equal(result, expected)

    # 将上面的 array 转换为 Series
    ser = pd.Series(a)
    # 对 Series 执行差分操作
    result = ser.diff()
    # 创建预期结果，其中包含了对应的差分操作结果
    expected = pd.Series(expected)
    # 使用测试框架中的函数验证计算结果是否与预期一致
    tm.assert_series_equal(result, expected)
```