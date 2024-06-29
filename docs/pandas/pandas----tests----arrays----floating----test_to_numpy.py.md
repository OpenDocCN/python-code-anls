# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_to_numpy.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray

@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box):
    con = pd.Series if box else pd.array

    # 创建包含浮点数的 Pandas Series 或 Pandas array
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    # 转换为 NumPy 数组
    result = arr.to_numpy()
    # 创建期望的 NumPy 数组
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    # 使用测试工具检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 创建包含浮点数和缺失值的 Pandas Series 或 Pandas array
    arr = con([0.1, 0.2, None], dtype="Float64")
    # 转换为 NumPy 数组
    result = arr.to_numpy()
    # 创建期望的 NumPy 数组，将缺失值转换为 NaN
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    # 使用测试工具检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_float(box):
    con = pd.Series if box else pd.array

    # 创建不包含缺失值的 Pandas Series 或 Pandas array，可以转换为 float 类型
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    # 转换为指定的 NumPy 数组类型 float64
    result = arr.to_numpy(dtype="float64")
    # 创建期望的 NumPy 数组
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    # 使用测试工具检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 创建包含缺失值的 Pandas Series 或 Pandas array，转换为 float 类型的 NumPy 数组
    arr = con([0.1, 0.2, None], dtype="Float64")
    # 转换为指定的 NumPy 数组类型 float64
    result = arr.to_numpy(dtype="float64")
    # 创建期望的 NumPy 数组，将缺失值转换为 NaN
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    # 使用测试工具检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 使用自定义的缺失值（na_value）转换为 float 类型的 NumPy 数组
    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    # 创建期望的 NumPy 数组，将缺失值转换为 NaN
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    # 使用测试工具检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_int(box):
    con = pd.Series if box else pd.array

    # 创建不包含缺失值的 Pandas Series 或 Pandas array，可以转换为 int 类型
    arr = con([1.0, 2.0, 3.0], dtype="Float64")
    # 转换为指定的 NumPy 数组类型 int64
    result = arr.to_numpy(dtype="int64")
    #
def test_to_numpy_na_value_with_nan():
    # 创建包含 NaN 和 NA 的浮点数数组 -> 仅用 `na_value` 填充 NA
    arr = FloatingArray(np.array([0.0, np.nan, 0.0]), np.array([False, False, True]))
    # 将数组转换为 NumPy 数组，指定数据类型为 float64，NA 值用 -1 替代
    result = arr.to_numpy(dtype="float64", na_value=-1)
    # 期望的 NumPy 数组，包含 [0.0, NaN, -1.0]，数据类型为 float64
    expected = np.array([0.0, np.nan, -1.0], dtype="float64")
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_dtype(box, dtype):
    # 根据 box 参数选择创建 Series 或者 array 对象
    con = pd.Series if box else pd.array
    # 创建包含 [0.0, 1.0] 的 Series 或 array 对象，数据类型为 Float64
    arr = con([0.0, 1.0], dtype="Float64")

    # 将对象转换为 NumPy 数组，指定数据类型为 dtype
    result = arr.to_numpy(dtype=dtype)
    # 期望的 NumPy 数组，包含 [0, 1]，数据类型为 dtype
    expected = np.array([0, 1], dtype=dtype)
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_na_raises(box, dtype):
    # 根据 box 参数选择创建 Series 或者 array 对象
    con = pd.Series if box else pd.array
    # 创建包含 [0.0, 1.0, None] 的 Series 或 array 对象，数据类型为 Float64
    arr = con([0.0, 1.0, None], dtype="Float64")
    # 断言转换为 NumPy 数组时，遇到 NA 值会引发 ValueError 异常
    with pytest.raises(ValueError, match=dtype):
        arr.to_numpy(dtype=dtype)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_string(box):
    # 根据 box 参数选择创建 Series 或者 array 对象
    con = pd.Series if box else pd.array
    # 创建包含 [0.0, 1.0, None] 的 Series 或 array 对象，数据类型为 Float64
    arr = con([0.0, 1.0, None], dtype="Float64")

    # 将对象转换为 NumPy 数组，指定数据类型为 str
    result = arr.to_numpy(dtype="str")
    # 期望的 NumPy 数组，包含 [0.0, 1.0, pd.NA]，数据类型为与平台相关的 Unicode 字符串
    expected = np.array([0.0, 1.0, pd.NA], dtype=f"{tm.ENDIAN}U32")
    # 断言两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_copy():
    # 如果没有缺失值，to_numpy 可以是零拷贝的
    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    # 将数组转换为 NumPy 数组，数据类型为 float64
    result = arr.to_numpy(dtype="float64")
    # 修改结果数组的第一个元素
    result[0] = 10
    # 断言扩展数组是否相等
    tm.assert_extension_array_equal(arr, pd.array([10, 0.2, 0.3], dtype="Float64"))

    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    # 将数组转换为 NumPy 数组，数据类型为 float64，强制进行拷贝
    result = arr.to_numpy(dtype="float64", copy=True)
    # 修改结果数组的第一个元素
    result[0] = 10
    # 断言扩展数组是否相等
    tm.assert_extension_array_equal(arr, pd.array([0.1, 0.2, 0.3], dtype="Float64"))
```