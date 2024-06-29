# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_dtypes.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及其相关模块
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import (
    Int8Dtype,
    UInt32Dtype,
)


def test_dtypes(dtype):
    # 对自动数据类型构造进行基本测试

    # 如果 dtype 是有符号整数类型
    if dtype.is_signed_integer:
        assert np.dtype(dtype.type).kind == "i"
    else:
        assert np.dtype(dtype.type).kind == "u"
    # 确保 dtype 的名称不为 None
    assert dtype.name is not None


@pytest.mark.parametrize("op", ["sum", "min", "max", "prod"])
def test_preserve_dtypes(op):
    # 对能够保持数据类型的操作进行测试

    # 创建包含不同类型数据的 DataFrame
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([1, None, 3], dtype="Int64"),
        }
    )

    # 对操作 op 进行测试
    result = getattr(df.C, op)()

    # 根据操作类型检查结果的数据类型
    if op in {"sum", "prod", "min", "max"}:
        assert isinstance(result, np.int64)
    else:
        assert isinstance(result, int)

    # 对 groupby 操作进行测试
    result = getattr(df.groupby("A"), op)()

    # 预期的 DataFrame 结果
    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([1, 3], dtype="Int64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    # 使用测试工具函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


def test_astype_nansafe():
    # 见 gh-22343
    # 创建包含 NaN 值的 Int8 类型的数组
    arr = pd.array([np.nan, 1, 2], dtype="Int8")
    msg = "cannot convert NA to integer"

    # 确保在转换时会抛出 ValueError 异常，且异常信息匹配预期
    with pytest.raises(ValueError, match=msg):
        arr.astype("uint32")


def test_construct_index(all_data, dropna):
    # 确保不会强制转换为不同的 Index 数据类型或非 Index 类型

    # 只保留前 10 行数据用于测试
    all_data = all_data[:10]

    # 根据 dropna 参数决定是否删除 NaN 值
    if dropna:
        other = np.array(all_data[~all_data.isna()])
    else:
        other = all_data

    # 构造索引对象
    result = pd.Index(pd.array(other, dtype=all_data.dtype))
    expected = pd.Index(other, dtype=all_data.dtype)
    # 确保结果索引的数据类型与预期相同，不会强制转换为 object 类型
    assert all_data.dtype == expected.dtype

    # 使用测试工具函数检查结果与预期是否相等
    tm.assert_index_equal(result, expected)


def test_astype_index(all_data, dropna):
    # 将 int/uint 类型转换为 Index 类型

    # 只保留前 10 行数据用于测试
    all_data = all_data[:10]

    # 根据 dropna 参数决定是否删除 NaN 值
    if dropna:
        other = all_data[~all_data.isna()]
    else:
        other = all_data

    # 获取数据的 dtype
    dtype = all_data.dtype

    # 创建 Index 对象
    idx = pd.Index(np.array(other))
    # 确保 idx 是 ABCIndex 类型的实例
    assert isinstance(idx, ABCIndex)

    # 进行类型转换
    result = idx.astype(dtype)
    expected = idx.astype(object).astype(dtype)
    # 使用测试工具函数检查结果与预期是否相等
    tm.assert_index_equal(result, expected)


def test_astype(all_data):
    # 只保留前 10 行数据用于测试
    all_data = all_data[:10]

    # 获取不包含 NaN 的整数数据和包含 NaN 的混合数据
    ints = all_data[~all_data.isna()]
    mixed = all_data
    # 创建 Int8Dtype 类型的对象
    dtype = Int8Dtype()

    # 强制转换为相同类型 - 整数
    s = pd.Series(ints)
    result = s.astype(all_data.dtype)
    expected = pd.Series(ints)
    # 使用测试工具函数检查结果与预期是否相等
    tm.assert_series_equal(result, expected)

    # 强制转换为相同其他类型 - 整数
    s = pd.Series(ints)
    result = s.astype(dtype)
    expected = pd.Series(ints, dtype=dtype)
    # 使用测试工具函数检查结果与预期是否相等
    tm.assert_series_equal(result, expected)

    # 强制转换为相同 numpy_dtype 类型 - 整数
    s = pd.Series(ints)
    result = s.astype(all_data.dtype.numpy_dtype)
    # 创建一个预期的 Series 对象，其数据类型是 all_data 的 NumPy 数据类型
    expected = pd.Series(ints._data.astype(all_data.dtype.numpy_dtype))
    # 断言 result 和 expected 的 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 强制转换为相同类型 - 混合类型
    s = pd.Series(mixed)
    # 将 Series 对象的数据类型强制转换为 all_data 的数据类型
    result = s.astype(all_data.dtype)
    # 创建一个预期的 Series 对象，其保持混合类型不变
    expected = pd.Series(mixed)
    # 断言 result 和 expected 的 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 强制转换为相同类型 - 使用给定的 dtype
    s = pd.Series(mixed)
    # 将 Series 对象的数据类型强制转换为指定的 dtype
    result = s.astype(dtype)
    # 创建一个预期的 Series 对象，其保持混合类型，但数据类型为指定的 dtype
    expected = pd.Series(mixed, dtype=dtype)
    # 断言 result 和 expected 的 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 强制转换为相同的 numpy 数据类型 - 混合类型
    s = pd.Series(mixed)
    # 检查是否无法将 NA（缺失值）转换为整数，预期抛出 ValueError，匹配特定消息
    msg = "cannot convert NA to integer"
    with pytest.raises(ValueError, match=msg):
        s.astype(all_data.dtype.numpy_dtype)

    # 强制转换为对象类型
    s = pd.Series(mixed)
    # 将 Series 对象的数据类型强制转换为 "object" 类型
    result = s.astype("object")
    # 创建一个预期的 Series 对象，其将数据以对象类型存储
    expected = pd.Series(np.asarray(mixed, dtype=object))
    # 断言 result 和 expected 的 Series 对象相等
    tm.assert_series_equal(result, expected)
# 定义测试函数 test_astype_copy，用于测试 Pandas 中数组的类型转换功能
def test_astype_copy():
    # 创建一个包含整数和空值的 Pandas 数组 arr，指定数据类型为 Int64
    arr = pd.array([1, 2, 3, None], dtype="Int64")
    # 创建原始的 Pandas 数组 orig，数据类型也为 Int64
    orig = pd.array([1, 2, 3, None], dtype="Int64")

    # 使用 copy=True 进行类型转换，确保数据和掩码都是实际的副本
    result = arr.astype("Int64", copy=True)
    assert result is not arr  # 确保结果不是原始数组的引用
    assert not tm.shares_memory(result, arr)  # 确保结果不与原始数组共享内存
    result[0] = 10  # 修改结果数组的第一个元素
    tm.assert_extension_array_equal(arr, orig)  # 断言原始数组未被修改
    result[0] = pd.NA  # 将结果数组的第一个元素设置为 NA
    tm.assert_extension_array_equal(arr, orig)  # 断言原始数组未被修改

    # 使用 copy=False 进行类型转换
    result = arr.astype("Int64", copy=False)
    assert result is arr  # 确保结果是原始数组的引用
    assert np.shares_memory(result._data, arr._data)  # 确保数据部分共享内存
    assert np.shares_memory(result._mask, arr._mask)  # 确保掩码部分共享内存
    result[0] = 10  # 修改结果数组的第一个元素
    assert arr[0] == 10  # 断言原始数组的第一个元素也被修改
    result[0] = pd.NA  # 将结果数组的第一个元素设置为 NA
    assert arr[0] is pd.NA  # 断言原始数组的第一个元素为 NA

    # 进行到不同数据类型的类型转换（astype to different dtype），即使使用 copy=False 也总是需要复制
    # 我们需要确保掩码也实际被复制
    arr = pd.array([1, 2, 3, None], dtype="Int64")
    orig = pd.array([1, 2, 3, None], dtype="Int64")

    # 使用 copy=False 将数组转换为 Int32
    result = arr.astype("Int32", copy=False)
    assert not tm.shares_memory(result, arr)  # 确保结果不与原始数组共享内存
    result[0] = 10  # 修改结果数组的第一个元素
    tm.assert_extension_array_equal(arr, orig)  # 断言原始数组未被修改
    result[0] = pd.NA  # 将结果数组的第一个元素设置为 NA
    tm.assert_extension_array_equal(arr, orig)  # 断言原始数组未被修改


# 定义测试函数 test_astype_to_larger_numpy，用于测试 Pandas 中将数组转换为更大的 NumPy 类型的功能
def test_astype_to_larger_numpy():
    # 创建一个包含整数的 Pandas 数组 a，数据类型为 Int32
    a = pd.array([1, 2], dtype="Int32")
    # 将数组转换为 int64 类型
    result = a.astype("int64")
    expected = np.array([1, 2], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)  # 断言转换结果与预期一致

    # 创建一个包含无符号整数的 Pandas 数组 a，数据类型为 UInt32
    a = pd.array([1, 2], dtype="UInt32")
    # 将数组转换为 uint64 类型
    result = a.astype("uint64")
    expected = np.array([1, 2], dtype="uint64")
    tm.assert_numpy_array_equal(result, expected)  # 断言转换结果与预期一致


# 使用参数化测试框架测试具体的类型转换操作
@pytest.mark.parametrize("dtype", [Int8Dtype(), "Int8", UInt32Dtype(), "UInt32"])
def test_astype_specific_casting(dtype):
    # 创建一个包含整数的 Pandas Series s，数据类型为 Int64
    s = pd.Series([1, 2, 3], dtype="Int64")
    # 将 Series 转换为指定的数据类型 dtype
    result = s.astype(dtype)
    expected = pd.Series([1, 2, 3], dtype=dtype)
    tm.assert_series_equal(result, expected)  # 断言转换结果与预期一致

    # 创建一个包含整数和空值的 Pandas Series s，数据类型为 Int64
    s = pd.Series([1, 2, 3, None], dtype="Int64")
    # 将 Series 转换为指定的数据类型 dtype
    result = s.astype(dtype)
    expected = pd.Series([1, 2, 3, None], dtype=dtype)
    tm.assert_series_equal(result, expected)  # 断言转换结果与预期一致


# 定义测试函数 test_astype_floating，测试 Pandas 中数组的浮点类型转换功能
def test_astype_floating():
    # 创建一个包含整数和空值的 Pandas 数组 arr，数据类型为 Int64
    arr = pd.array([1, 2, None], dtype="Int64")
    # 将数组转换为 Float64 类型
    result = arr.astype("Float64")
    expected = pd.array([1.0, 2.0, None], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)  # 断言转换结果与预期一致


# 定义测试函数 test_astype_dt64，测试 Pandas 中将数组转换为 datetime64 类型的功能
def test_astype_dt64():
    # 创建一个包含整数和 NA 值的 Pandas 数组 arr
    arr = pd.array([1, 2, 3, pd.NA]) * 10**9
    # 将数组转换为 datetime64[ns] 类型
    result = arr.astype("datetime64[ns]")
    # 生成预期的结果数组 expected
    expected = np.array([1, 2, 3, "NaT"], dtype="M8[s]").astype("M8[ns]")
    tm.assert_numpy_array_equal(result, expected)  # 断言转换结果与预期一致


# 定义测试函数 test_construct_cast_invalid，测试在构造和转换过程中遇到不支持的数据类型时的异常处理
def test_construct_cast_invalid(dtype):
    msg = "cannot safely"
    
    # 测试当数组 arr 包含不支持的数据类型时抛出异常
    arr = [1.2, 2.3, 3.7]
    with pytest.raises(TypeError, match=msg):
        pd.array(arr, dtype=dtype)

    # 测试当使用 Series 构造函数时，如果包含不支持的数据类型会抛出异常
    with pytest.raises(TypeError, match=msg):
        pd.Series(arr).astype(dtype)

    # 测试当数组 arr 包含 NaN 值时，如果包含不支持的数据类型会抛出异常
    arr = [1.2, 2.3, 3.7, np.nan]
    with pytest.raises(TypeError, match=msg):
        pd.array(arr, dtype=dtype)

    # 测试当使用 Series 构造函数时，如果包含 NaN 值和不支持的数据类型会抛出异常
    with pytest.raises(TypeError, match=msg):
        pd.Series(arr).astype(dtype)
# 使用 pytest 的装饰器标记测试参数化函数，in_series 可以为 True 或 False
@pytest.mark.parametrize("in_series", [True, False])
def test_to_numpy_na_nan(in_series):
    # 创建包含整数和空值的 Pandas 数组 a，使用 Int64 类型
    a = pd.array([0, 1, None], dtype="Int64")
    if in_series:
        # 如果 in_series 为 True，则将 a 转换为 Pandas Series
        a = pd.Series(a)

    # 将 Pandas 数组转换为 NumPy 数组，设置空值为 np.nan，类型为 float64
    result = a.to_numpy(dtype="float64", na_value=np.nan)
    # 期望得到的 NumPy 数组，包含浮点数和 np.nan
    expected = np.array([0.0, 1.0, np.nan], dtype="float64")
    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)

    # 将 Pandas 数组转换为 NumPy 数组，设置空值为 -1，类型为 int64
    result = a.to_numpy(dtype="int64", na_value=-1)
    # 期望得到的 NumPy 数组，包含整数和 -1
    expected = np.array([0, 1, -1], dtype="int64")
    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)

    # 将 Pandas 数组转换为 NumPy 数组，设置空值为 False，类型为 bool
    result = a.to_numpy(dtype="bool", na_value=False)
    # 期望得到的 NumPy 数组，包含布尔值和 False
    expected = np.array([False, True, False], dtype="bool")
    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)


# 使用 pytest 的装饰器标记测试参数化函数，in_series 可以为 True 或 False；dtype 可以为 int32, int64, bool
@pytest.mark.parametrize("in_series", [True, False])
@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_to_numpy_dtype(dtype, in_series):
    # 创建包含整数的 Pandas 数组 a，使用 Int64 类型
    a = pd.array([0, 1], dtype="Int64")
    if in_series:
        # 如果 in_series 为 True，则将 a 转换为 Pandas Series
        a = pd.Series(a)

    # 将 Pandas 数组转换为 NumPy 数组，指定目标 dtype
    result = a.to_numpy(dtype=dtype)
    # 根据指定的 dtype 创建期望的 NumPy 数组
    expected = np.array([0, 1], dtype=dtype)
    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(result, expected)


# 使用 pytest 的装饰器标记测试参数化函数，dtype 可以为 int64 或 bool
@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_to_numpy_na_raises(dtype):
    # 创建包含整数和空值的 Pandas 数组 a，使用 Int64 类型
    a = pd.array([0, 1, None], dtype="Int64")
    # 使用 pytest 的上下文管理器检查是否抛出 ValueError 异常，异常信息需要与 dtype 匹配
    with pytest.raises(ValueError, match=dtype):
        a.to_numpy(dtype=dtype)


# 测试将 Pandas 数组转换为字符串类型的 NumPy 数组
def test_astype_str():
    # 创建包含整数和空值的 Pandas 数组 a，使用 Int64 类型
    a = pd.array([1, 2, None], dtype="Int64")
    # 根据预期的结果创建一个字符串类型的 NumPy 数组
    expected = np.array(["1", "2", "<NA>"], dtype=f"{tm.ENDIAN}U21")

    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(a.astype(str), expected)
    # 断言两个 NumPy 数组相等
    tm.assert_numpy_array_equal(a.astype("str"), expected)


# 测试将 Pandas 数组转换为布尔类型的 Pandas 数组
def test_astype_boolean():
    # 创建包含整数和空值的 Pandas 数组 a，使用 Int64 类型
    a = pd.array([1, 0, -1, 2, None], dtype="Int64")
    # 将 Pandas 数组转换为布尔类型的 Pandas 数组
    result = a.astype("boolean")
    # 根据预期的结果创建一个布尔类型的 Pandas 数组
    expected = pd.array([True, False, True, True, None], dtype="boolean")
    # 断言两个扩展数组相等
    tm.assert_extension_array_equal(result, expected)
```