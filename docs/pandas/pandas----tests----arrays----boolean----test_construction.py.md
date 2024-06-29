# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_construction.py`

```
# 导入所需的库
import numpy as np  # 导入NumPy库，用于处理数组和数值计算
import pytest  # 导入Pytest库，用于编写和运行测试用例

import pandas as pd  # 导入Pandas库，用于数据处理和分析
import pandas._testing as tm  # 导入Pandas内部测试模块，用于测试框架
from pandas.arrays import BooleanArray  # 导入Pandas的BooleanArray数组类型
from pandas.core.arrays.boolean import coerce_to_array  # 导入Pandas的数组处理函数


def test_boolean_array_constructor():
    # 创建包含布尔值的NumPy数组
    values = np.array([True, False, True, False], dtype="bool")
    # 创建布尔掩码数组
    mask = np.array([False, False, False, True], dtype="bool")

    # 使用BooleanArray构造函数创建扩展数组
    result = BooleanArray(values, mask)
    # 创建预期的Pandas数组对象
    expected = pd.array([True, False, True, None], dtype="boolean")
    # 使用Pandas测试模块验证扩展数组与预期值相等
    tm.assert_extension_array_equal(result, expected)

    # 测试当值不是布尔NumPy数组时，是否会引发TypeError异常
    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.tolist(), mask)

    # 测试当掩码不是布尔NumPy数组时，是否会引发TypeError异常
    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, mask.tolist())

    # 测试当值不是布尔NumPy数组时，是否会引发TypeError异常
    with pytest.raises(TypeError, match="values should be boolean numpy array"):
        BooleanArray(values.astype(int), mask)

    # 测试当掩码为None时，是否会引发TypeError异常
    with pytest.raises(TypeError, match="mask should be boolean numpy array"):
        BooleanArray(values, None)

    # 测试当值数组和掩码数组的形状不匹配时，是否会引发ValueError异常
    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values.reshape(1, -1), mask)

    # 测试当值数组和掩码数组的形状不匹配时，是否会引发ValueError异常
    with pytest.raises(ValueError, match="values.shape must match mask.shape"):
        BooleanArray(values, mask.reshape(1, -1))


def test_boolean_array_constructor_copy():
    # 创建包含布尔值的NumPy数组
    values = np.array([True, False, True, False], dtype="bool")
    # 创建布尔掩码数组
    mask = np.array([False, False, False, True], dtype="bool")

    # 使用BooleanArray构造函数创建扩展数组
    result = BooleanArray(values, mask)
    # 验证结果数组的数据是否与初始值数组相同
    assert result._data is values
    # 验证结果数组的掩码是否与初始掩码数组相同
    assert result._mask is mask

    # 使用BooleanArray构造函数创建扩展数组（复制数据）
    result = BooleanArray(values, mask, copy=True)
    # 验证结果数组的数据是否与初始值数组不同
    assert result._data is not values
    # 验证结果数组的掩码是否与初始掩码数组不同
    assert result._mask is not mask


def test_to_boolean_array():
    # 创建预期的BooleanArray对象
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, False])
    )

    # 创建Pandas数组对象
    result = pd.array([True, False, True], dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)
    
    # 创建Pandas数组对象
    result = pd.array(np.array([True, False, True]), dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)
    
    # 创建Pandas数组对象（包含对象类型数组）
    result = pd.array(np.array([True, False, True], dtype=object), dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)

    # 测试包含缺失值的情况
    expected = BooleanArray(
        np.array([True, False, True]), np.array([False, False, True])
    )

    # 创建Pandas数组对象（包含缺失值）
    result = pd.array([True, False, None], dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)
    
    # 创建Pandas数组对象（包含对象类型数组和缺失值）
    result = pd.array(np.array([True, False, None], dtype=object), dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_all_none():
    # 创建预期的BooleanArray对象（全部为None）
    expected = BooleanArray(np.array([True, True, True]), np.array([True, True, True]))

    # 创建Pandas数组对象（全部为None）
    result = pd.array([None, None, None], dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)
    
    # 创建Pandas数组对象（包含对象类型数组，全部为None）
    result = pd.array(np.array([None, None, None], dtype=object), dtype="boolean")
    # 使用Pandas测试模块验证结果与预期值相等
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        # 第一个元组：输入列表包含 True, False, None, np.nan, pd.NA，预期输出列表是对应元素的映射
        ([True, False, None, np.nan, pd.NA], [True, False, None, None, None]),
        
        # 第二个元组：输入列表包含 True, np.nan，预期输出列表是对应元素的映射
        ([True, np.nan], [True, None]),
        
        # 第三个元组：输入列表包含 True, pd.NA，预期输出列表是对应元素的映射
        ([True, pd.NA], [True, None]),
        
        # 第四个元组：输入列表包含 np.nan, np.nan，预期输出列表是对应元素的映射
        ([np.nan, np.nan], [None, None]),
        
        # 第五个元组：输入是由 np.nan 组成的 numpy 数组，预期输出是对应元素的映射
        (np.array([np.nan, np.nan], dtype=float), [None, None]),
    ],
def test_to_boolean_array_missing_indicators(a, b):
    # 使用 pd.array 创建布尔类型的数组 result，使用参数 a 和指定的数据类型 "boolean"
    result = pd.array(a, dtype="boolean")
    # 使用 pd.array 创建布尔类型的数组 expected，使用参数 b 和指定的数据类型 "boolean"
    expected = pd.array(b, dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["foo", "bar"],  # 字符串数组
        ["1", "2"],  # 字符串数组
        # "foo",  # 被注释掉的单个字符串值
        [1, 2],  # 整数数组
        [1.0, 2.0],  # 浮点数数组
        pd.date_range("20130101", periods=2),  # 时间序列
        np.array(["foo"]),  # NumPy 字符串数组
        np.array([1, 2]),  # NumPy 整数数组
        np.array([1.0, 2.0]),  # NumPy 浮点数数组
        [np.nan, {"a": 1}],  # 包含 NaN 和字典元素的混合数组
    ],
)
def test_to_boolean_array_error(values):
    # 测试转换现有数组为 BooleanArray 时的错误情况
    msg = "Need to pass bool-like value"  # 错误信息
    # 使用 pytest 断言检测 TypeError 异常，并匹配错误信息 msg
    with pytest.raises(TypeError, match=msg):
        pd.array(values, dtype="boolean")


def test_to_boolean_array_from_integer_array():
    # 使用 pd.array 从整数数组创建布尔类型数组 result
    result = pd.array(np.array([1, 0, 1, 0]), dtype="boolean")
    # 预期的布尔类型数组 expected
    expected = pd.array([True, False, True, False], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)

    # 包含缺失值的情况
    # 使用 pd.array 从整数数组创建布尔类型数组 result
    result = pd.array(np.array([1, 0, 1, None]), dtype="boolean")
    # 预期的布尔类型数组 expected，包含 None（缺失值）
    expected = pd.array([True, False, True, None], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_float_array():
    # 使用 pd.array 从浮点数数组创建布尔类型数组 result
    result = pd.array(np.array([1.0, 0.0, 1.0, 0.0]), dtype="boolean")
    # 预期的布尔类型数组 expected
    expected = pd.array([True, False, True, False], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)

    # 包含缺失值的情况
    # 使用 pd.array 从浮点数数组创建布尔类型数组 result
    result = pd.array(np.array([1.0, 0.0, 1.0, np.nan]), dtype="boolean")
    # 预期的布尔类型数组 expected，包含 None（缺失值）
    expected = pd.array([True, False, True, None], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_integer_like():
    # 使用 pd.array 从整数类似数组创建布尔类型数组 result
    result = pd.array([1, 0, 1, 0], dtype="boolean")
    # 预期的布尔类型数组 expected
    expected = pd.array([True, False, True, False], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)

    # 包含缺失值的情况
    # 使用 pd.array 从整数类似数组创建布尔类型数组 result
    result = pd.array([1, 0, 1, None], dtype="boolean")
    # 预期的布尔类型数组 expected，包含 None（缺失值）
    expected = pd.array([True, False, True, None], dtype="boolean")
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)


def test_coerce_to_array():
    # TODO this is currently not public API
    # 创建布尔类型的值数组 values 和掩码数组 mask
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")
    # 调用 coerce_to_array 将 values 和 mask 转换为 BooleanArray
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    # 预期的 BooleanArray
    expected = BooleanArray(values, mask)
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)
    # 断言 result 的数据和 values 相同
    assert result._data is values
    # 断言 result 的掩码和 mask 相同
    assert result._mask is mask
    # 调用 coerce_to_array 将 values 和 mask 转换为 BooleanArray，使用 copy=True
    result = BooleanArray(*coerce_to_array(values, mask=mask, copy=True))
    # 预期的 BooleanArray
    expected = BooleanArray(values, mask)
    # 断言 result 和 expected 数组相等
    tm.assert_extension_array_equal(result, expected)
    # 断言 result 的数据和 values 不同
    assert result._data is not values
    # 断言 result 的掩码和 mask 不同
    assert result._mask is not mask

    # 混合包含 values 和 mask 的缺失值
    values = [True, False, None, False]
    mask = np.array([False, False, False, True], dtype="bool")
    # 调用 coerce_to_array 将 values 和 mask 转换为 BooleanArray
    result = BooleanArray(*coerce_to_array(values, mask=mask))
    # 创建预期的布尔数组对象，使用两个 NumPy 数组作为输入
    expected = BooleanArray(
        np.array([True, False, True, True]), np.array([False, False, True, True])
    )
    # 断言结果与预期的扩展数组相等
    tm.assert_extension_array_equal(result, expected)
    
    # 使用 coerce_to_array 将输入值转换为布尔数组对象
    result = BooleanArray(*coerce_to_array(np.array(values, dtype=object), mask=mask))
    # 再次断言结果与预期的扩展数组相等
    tm.assert_extension_array_equal(result, expected)
    
    # 使用 coerce_to_array 将输入值和掩码转换为布尔数组对象
    result = BooleanArray(*coerce_to_array(values, mask=mask.tolist()))
    # 再次断言结果与预期的扩展数组相等
    tm.assert_extension_array_equal(result, expected)

    # 对于错误的维度，应该抛出错误
    values = np.array([True, False, True, False], dtype="bool")
    mask = np.array([False, False, False, True], dtype="bool")

    # 传递 2D 值是可以的，只要没有掩码
    coerce_to_array(values.reshape(1, -1))

    # 使用 pytest 断言应该抛出 ValueError，并检查错误消息
    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values.reshape(1, -1), mask=mask)

    # 使用 pytest 断言应该抛出 ValueError，并检查错误消息
    with pytest.raises(ValueError, match="values.shape and mask.shape must match"):
        coerce_to_array(values, mask=mask.reshape(1, -1))
def test_coerce_to_array_from_boolean_array():
    # 创建包含布尔值的 NumPy 数组
    values = np.array([True, False, True, False], dtype="bool")
    # 创建布尔掩码数组
    mask = np.array([False, False, False, True], dtype="bool")
    # 使用 BooleanArray 类创建布尔数组对象
    arr = BooleanArray(values, mask)
    # 调用 coerce_to_array 函数，将 BooleanArray 转换为结果数组
    result = BooleanArray(*coerce_to_array(arr))
    # 断言扩展数组相等
    tm.assert_extension_array_equal(result, arr)
    # 不复制数据，确保结果数据和原始数据相同
    assert result._data is arr._data
    assert result._mask is arr._mask

    # 复制数据
    result = BooleanArray(*coerce_to_array(arr), copy=True)
    # 再次断言扩展数组相等
    tm.assert_extension_array_equal(result, arr)
    # 确保数据已复制
    assert result._data is not arr._data
    assert result._mask is not arr._mask

    # 测试传递掩码会引发 ValueError 异常
    with pytest.raises(ValueError, match="cannot pass mask for BooleanArray input"):
        coerce_to_array(arr, mask=mask)


def test_coerce_to_numpy_array():
    # 包含缺失值，转换为对象 dtype 的 NumPy 数组
    arr = pd.array([True, False, None], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, pd.NA], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    # 不包含缺失值，转换为对象 dtype 的 NumPy 数组
    arr = pd.array([True, False, True], dtype="boolean")
    result = np.array(arr)
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    # 强制转换为 bool dtype 的 NumPy 数组
    result = np.array(arr, dtype="bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)
    # 包含缺失值将引发错误
    arr = pd.array([True, False, None], dtype="boolean")
    msg = (
        "cannot convert to 'bool'-dtype NumPy array with missing values. "
        "Specify an appropriate 'na_value' for this dtype."
    )
    with pytest.raises(ValueError, match=msg):
        np.array(arr, dtype="bool")


def test_to_boolean_array_from_strings():
    # 从字符串序列创建 BooleanArray 对象
    result = BooleanArray._from_sequence_of_strings(
        np.array(["True", "False", "1", "1.0", "0", "0.0", np.nan], dtype=object),
        dtype=pd.BooleanDtype(),
    )
    expected = BooleanArray(
        np.array([True, False, True, True, False, False, False]),
        np.array([False, False, False, False, False, False, True]),
    )

    tm.assert_extension_array_equal(result, expected)


def test_to_boolean_array_from_strings_invalid_string():
    # 测试从无效字符串序列创建 BooleanArray 对象会引发 ValueError 异常
    with pytest.raises(ValueError, match="cannot be cast"):
        BooleanArray._from_sequence_of_strings(["donkey"], dtype=pd.BooleanDtype())


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box):
    con = pd.Series if box else pd.array
    # 默认情况下（包含或不包含缺失值），转换为对象 dtype 的 NumPy 数组
    arr = con([True, False, True], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([True, False, None], dtype="boolean")
    result = arr.to_numpy()
    expected = np.array([True, False, pd.NA], dtype="object")
    # 使用 pandas.testing 模块的 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 创建一个包含布尔值、None 和缺失值 NA 的 pandas 数组 arr，指定数据类型为 "boolean"
    arr = con([True, False, None], dtype="boolean")
    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "str"
    result = arr.to_numpy(dtype="str")
    # 创建一个预期的 numpy 数组，包含 True、False 和缺失值 NA，数据类型由 tm.ENDIAN 决定，长度为 5
    expected = np.array([True, False, pd.NA], dtype=f"{tm.ENDIAN}U5")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 如果数组中没有缺失值，则可以将其转换为布尔值，否则会引发异常
    arr = con([True, False, True], dtype="boolean")
    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "bool"
    result = arr.to_numpy(dtype="bool")
    # 创建一个预期的 numpy 数组，包含 True、False 和 True，数据类型为 "bool"
    expected = np.array([True, False, True], dtype="bool")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 创建一个包含布尔值、None 和缺失值 NA 的 pandas 数组 arr，指定数据类型为 "boolean"
    arr = con([True, False, None], dtype="boolean")
    # 使用 pytest 的 assertRaises 方法检查是否会引发 ValueError 异常，异常消息中包含 "cannot convert to 'bool'-dtype"
    with pytest.raises(ValueError, match="cannot convert to 'bool'-dtype"):
        # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "bool"
        result = arr.to_numpy(dtype="bool")

    # 创建一个包含布尔值、None 和缺失值 NA 的 pandas 数组 arr，指定数据类型为 "boolean"
    arr = con([True, False, None], dtype="boolean")
    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "object"，缺失值 NA 被映射为 None
    result = arr.to_numpy(dtype=object, na_value=None)
    # 创建一个预期的 numpy 数组，包含 True、False 和 None，数据类型为 "object"
    expected = np.array([True, False, None], dtype="object")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "bool"，缺失值 NA 被映射为 False
    result = arr.to_numpy(dtype=bool, na_value=False)
    # 创建一个预期的 numpy 数组，包含 True、False 和 False，数据类型为 "bool"
    expected = np.array([True, False, False], dtype="bool")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "int64"，缺失值 NA 被映射为 -99
    result = arr.to_numpy(dtype="int64", na_value=-99)
    # 创建一个预期的 numpy 数组，包含 1、0 和 -99，数据类型为 "int64"
    expected = np.array([1, 0, -99], dtype="int64")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 将 pandas 数组 arr 转换为 numpy 数组，数据类型转换为 "float64"，缺失值 NA 被映射为 np.nan
    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    # 创建一个预期的 numpy 数组，包含 1、0 和 np.nan，数据类型为 "float64"
    expected = np.array([1, 0, np.nan], dtype="float64")
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 尝试将 pandas 数组 arr 转换为 "int64" 类型的 numpy 数组，但未指定缺失值映射时会引发 ValueError 异常
    with pytest.raises(ValueError, match="cannot convert to 'int64'-dtype"):
        arr.to_numpy(dtype="int64")
def test_to_numpy_copy():
    # 定义一个测试函数，验证 pd.array 对象的 to_numpy 方法

    # 创建一个包含布尔值的 pandas array 对象
    arr = pd.array([True, False, True], dtype="boolean")
    
    # 调用 to_numpy 方法转换为 numpy 数组，如果没有缺失值，则可能是零拷贝操作
    result = arr.to_numpy(dtype=bool)
    
    # 修改结果数组的第一个元素为 False
    result[0] = False
    
    # 使用测试工具 tm.assert_extension_array_equal 检查修改后的 arr 是否与预期结果相同
    tm.assert_extension_array_equal(
        arr, pd.array([False, False, True], dtype="boolean")
    )

    # 重新创建 arr 对象，再次调用 to_numpy 方法，这次指定拷贝操作
    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.to_numpy(dtype=bool, copy=True)
    
    # 修改结果数组的第一个元素为 False
    result[0] = False
    
    # 使用测试工具 tm.assert_extension_array_equal 检查修改后的 arr 是否与预期结果相同
    tm.assert_extension_array_equal(
        arr, pd.array([True, False, True], dtype="boolean")
    )
```