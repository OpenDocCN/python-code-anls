# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_astype.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库及其测试模块
import pandas as pd
import pandas._testing as tm

# 定义测试函数 test_astype
def test_astype():
    # 创建包含缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.1, 0.2, None], dtype="Float64")

    # 测试转换为 "int64" 类型时是否会引发 ValueError 异常，且匹配特定的错误信息
    with pytest.raises(ValueError, match="cannot convert NA to integer"):
        arr.astype("int64")

    # 测试转换为 "bool" 类型时是否会引发 ValueError 异常，且匹配特定的错误信息
    with pytest.raises(ValueError, match="cannot convert float NaN to bool"):
        arr.astype("bool")

    # 将数组 arr 转换为 "float64" 类型，并将结果与期望的 numpy 数组进行比较
    result = arr.astype("float64")
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    # 创建没有缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.0, 1.0, 0.5], dtype="Float64")

    # 将数组 arr 转换为 "int64" 类型，并将结果与期望的 numpy 数组进行比较
    result = arr.astype("int64")
    expected = np.array([0, 1, 0], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    # 将数组 arr 转换为 "bool" 类型，并将结果与期望的 numpy 数组进行比较
    result = arr.astype("bool")
    expected = np.array([False, True, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_to_floating_array():
    # 创建包含缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    # 将数组 arr 转换为 "Float64" 类型，并验证结果与原始数组相等
    result = arr.astype("Float64")
    tm.assert_extension_array_equal(result, arr)

    # 将数组 arr 转换为 pd.Float64Dtype() 类型，并验证结果与原始数组相等
    result = arr.astype(pd.Float64Dtype())
    tm.assert_extension_array_equal(result, arr)

    # 将数组 arr 转换为 "Float32" 类型，并将结果与期望的 Pandas 数组进行比较
    result = arr.astype("Float32")
    expected = pd.array([0.0, 1.0, None], dtype="Float32")
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_boolean_array():
    # 创建包含缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    # 将数组 arr 转换为 "boolean" 类型，并将结果与期望的 Pandas 数组进行比较
    result = arr.astype("boolean")
    expected = pd.array([False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)

    # 将数组 arr 转换为 pd.BooleanDtype() 类型，并将结果与期望的 Pandas 数组进行比较
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_integer_array():
    # 创建包含缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.0, 1.5, None], dtype="Float64")

    # 将数组 arr 转换为 "Int64" 类型，并将结果与期望的 Pandas 数组进行比较
    result = arr.astype("Int64")
    expected = pd.array([0, 1, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_str():
    # 创建包含缺失值的 Pandas 数组 a，数据类型为 "Float64"
    a = pd.array([0.1, 0.2, None], dtype="Float64")
    
    # 将数组 a 转换为 "str" 类型，并将结果与期望的 numpy 数组进行比较
    expected = np.array(["0.1", "0.2", "<NA>"], dtype="U32")
    tm.assert_numpy_array_equal(a.astype(str), expected)
    tm.assert_numpy_array_equal(a.astype("str"), expected)


def test_astype_copy():
    # 创建包含缺失值的 Pandas 数组 arr，数据类型为 "Float64"
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    # 使用 copy=True 将数组 arr 转换为 "Float64" 类型，确保数据和掩码都是实际的副本
    result = arr.astype("Float64", copy=True)
    assert result is not arr
    assert not tm.shares_memory(result, arr)
    result[0] = 10
    tm.assert_extension_array_equal(arr, orig)
    result[0] = pd.NA
    tm.assert_extension_array_equal(arr, orig)

    # 使用 copy=False
    result = arr.astype("Float64", copy=False)
    assert result is arr
    assert np.shares_memory(result._data, arr._data)
    assert np.shares_memory(result._mask, arr._mask)
    result[0] = 10
    assert arr[0] == 10
    result[0] = pd.NA
    assert arr[0] is pd.NA
    # 将 arr 数组中的数据转换为 "Float32" 类型，即使使用 copy=False 也需要复制
    # 需要确保掩码（mask）也被复制
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    # 在不复制的情况下将 arr 转换为 "Float32" 类型
    result = arr.astype("Float32", copy=False)
    # 断言 result 和 arr 不共享内存
    assert not tm.shares_memory(result, arr)
    
    # 修改 result 的第一个元素为 10
    result[0] = 10
    # 断言修改后的 arr 与原始 arr 相等
    tm.assert_extension_array_equal(arr, orig)
    
    # 将 result 的第一个元素修改为 pd.NA
    result[0] = pd.NA
    # 再次断言修改后的 arr 与原始 arr 相等
    tm.assert_extension_array_equal(arr, orig)
# 将数组中的元素类型转换为 object 类型的测试函数
def test_astype_object(dtype):
    # 创建一个 Pandas 数组，包含一个浮点数和一个缺失值 pd.NA
    arr = pd.array([1.0, pd.NA], dtype=dtype)

    # 将数组元素类型转换为 object
    result = arr.astype(object)
    # 预期的转换结果，将原始数组中的浮点数和 pd.NA 转换为 object 类型的 numpy 数组
    expected = np.array([1.0, pd.NA], dtype=object)
    # 使用 Pandas 测试工具检查两个 numpy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
    
    # 检查确切的元素类型
    assert isinstance(result[0], float)
    assert result[1] is pd.NA


def test_Float64_conversion():
    # GH#40729
    # 创建一个包含字符串的 Pandas Series，类型为 object
    testseries = pd.Series(["1", "2", "3", "4"], dtype="object")
    # 将 Series 中的元素转换为 Float64 类型
    result = testseries.astype(pd.Float64Dtype())

    # 预期的结果，将字符串转换为浮点数，并保持 Float64 类型
    expected = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=pd.Float64Dtype())

    # 使用 Pandas 测试工具检查两个 Series 是否相等
    tm.assert_series_equal(result, expected)
```