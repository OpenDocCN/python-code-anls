# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_construction.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
import pandas._testing as tm  # 导入 Pandas 内部测试工具
from pandas.core.arrays import FloatingArray  # 导入 Pandas 扩展数组
from pandas.core.arrays.floating import (  # 导入 Pandas 浮点数相关模块
    Float32Dtype,
    Float64Dtype,
)


def test_uses_pandas_na():
    a = pd.array([1, None], dtype=Float64Dtype())  # 创建一个 Pandas 扩展数组，包含浮点数和缺失值
    assert a[1] is pd.NA  # 断言第二个元素是 Pandas 中的 NA


def test_floating_array_constructor():
    values = np.array([1, 2, 3, 4], dtype="float64")  # 创建一个 NumPy 浮点数数组
    mask = np.array([False, False, False, True], dtype="bool")  # 创建一个布尔类型的掩码数组

    result = FloatingArray(values, mask)  # 使用给定的数据和掩码创建 Pandas 扩展数组
    expected = pd.array([1, 2, 3, np.nan], dtype="Float64")  # 创建预期的 Pandas 扩展数组
    tm.assert_extension_array_equal(result, expected)  # 断言结果与预期相等
    tm.assert_numpy_array_equal(result._data, values)  # 断言数据部分与给定的值数组相等
    tm.assert_numpy_array_equal(result._mask, mask)  # 断言掩码部分与给定的掩码数组相等

    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):  # 断言在使用不正确类型的数据时会抛出 TypeError 异常
        FloatingArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):  # 断言在使用不正确类型的数据时会抛出 TypeError 异常
        FloatingArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):  # 断言在使用不正确类型的数据时会抛出 TypeError 异常
        FloatingArray(values.astype(int), mask)

    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):  # 断言在缺少必需的 'mask' 参数时会抛出 TypeError 异常
        FloatingArray(values)


def test_floating_array_disallows_float16():
    # GH#44715
    arr = np.array([1, 2], dtype=np.float16)  # 创建一个 NumPy float16 类型数组

    msg = "FloatingArray does not support np.float16 dtype"
    with pytest.raises(TypeError, match=msg):  # 断言当使用 float16 类型数组时会抛出 TypeError 异常
        FloatingArray(arr, mask)


def test_floating_array_disallows_Float16_dtype(request):
    # GH#44715
    with pytest.raises(TypeError, match="data type 'Float16' not understood"):  # 断言当使用不支持的 'Float16' 数据类型时会抛出 TypeError 异常
        pd.array([1.0, 2.0], dtype="Float16")


def test_floating_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype="float64")  # 创建一个 NumPy 浮点数数组
    mask = np.array([False, False, False, True], dtype="bool")  # 创建一个布尔类型的掩码数组

    result = FloatingArray(values, mask)  # 使用给定的数据和掩码创建 Pandas 扩展数组
    assert result._data is values  # 断言数据部分是原始的数据数组
    assert result._mask is mask  # 断言掩码部分是原始的掩码数组

    result = FloatingArray(values, mask, copy=True)  # 使用给定的数据和掩码创建 Pandas 扩展数组，强制复制数据
    assert result._data is not values  # 断言数据部分不再是原始的数据数组
    assert result._mask is not mask  # 断言掩码部分不再是原始的掩码数组


def test_to_array():
    result = pd.array([0.1, 0.2, 0.3, 0.4])  # 创建一个 Pandas 扩展数组，包含浮点数
    expected = pd.array([0.1, 0.2, 0.3, 0.4], dtype="Float64")  # 创建预期的 Pandas 扩展数组
    tm.assert_extension_array_equal(result, expected)  # 断言结果与预期相等


@pytest.mark.parametrize(  # 使用 pytest 参数化测试
    "a, b",
    [
        ([1, None], [1, pd.NA]),  # 测试包含 None 的情况
        ([None], [pd.NA]),  # 测试全为 None 的情况
        ([None, np.nan], [pd.NA, pd.NA]),  # 测试同时包含 None 和 np.nan 的情况
        ([1, np.nan], [1, pd.NA]),  # 测试包含 np.nan 的情况
        ([np.nan], [pd.NA]),  # 测试仅包含 np.nan 的情况
    ],
)
def test_to_array_none_is_nan(a, b):
    result = pd.array(a, dtype="Float64")  # 创建一个 Pandas 扩展数组，指定数据类型为浮点数
    expected = pd.array(b, dtype="Float64")  # 创建预期的 Pandas 扩展数组
    tm.assert_extension_array_equal(result, expected)  # 断言结果与预期相等


def test_to_array_mixed_integer_float():
    result = pd.array([1, 2.0])  # 创建一个 Pandas 扩展数组，包含混合的整数和浮点数
    expected = pd.array([1.0, 2.0], dtype="Float64")  # 创建预期的 Pandas 扩展数组
    tm.assert_extension_array_equal(result, expected)  # 断言结果与预期相等

    result = pd.array([1, None, 2.0])  # 创建一个 Pandas 扩展数组，包含混合的整数、缺失值和浮点数
    expected = pd.array([1.0, None, 2.0], dtype="Float64")  # 创建预期的 Pandas 扩展数组
    # 使用测试框架中的断言函数 tm.assert_extension_array_equal 对比 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器标记多组输入值进行参数化测试
@pytest.mark.parametrize(
    "values",  # 参数名为 values
    [
        ["foo", "bar"],  # 列表中的两个字符串元素
        "foo",  # 单个字符串
        1,  # 整数
        1.0,  # 浮点数
        pd.date_range("20130101", periods=2),  # Pandas 日期范围
        np.array(["foo"]),  # NumPy 数组，包含一个字符串元素
        [[1, 2], [3, 4]],  # 嵌套列表
        [np.nan, {"a": 1}],  # 包含 NaN 和字典的列表
        # GH#44514 用于在检查 ndim 之前静默交换所有 NaN 情况
        np.array([pd.NA] * 6, dtype=object).reshape(3, 2),  # 包含 NaN 的 NumPy 对象数组
    ],
)
def test_to_array_error(values):
    # 测试将现有数组转换为 FloatingArray 时的错误处理
    msg = "|".join(
        [
            "cannot be converted to FloatingDtype",  # 无法转换为 FloatingDtype
            "values must be a 1D list-like",  # values 必须是一维类列表
            "Cannot pass scalar",  # 无法传递标量
            r"float\(\) argument must be a string or a (real )?number, not 'dict'",  # float() 参数必须是字符串或（实）数值，而不是 'dict'
            "could not convert string to float: 'foo'",  # 无法将字符串 'foo' 转换为浮点数
            r"could not convert string to float: np\.str_\('foo'\)",  # 无法将字符串 np.str_('foo') 转换为浮点数
        ]
    )
    # 使用 pytest 来断言引发 TypeError 或 ValueError 异常，并匹配预期的错误消息
    with pytest.raises((TypeError, ValueError), match=msg):
        pd.array(values, dtype="Float64")


@pytest.mark.parametrize(
    "values",  # 参数名为 values
    [["1", "2", None], ["1.5", "2", None]]  # 包含两个子列表的列表
)
def test_construct_from_float_strings(values):
    # 参见 test_to_integer_array_str
    # 构造预期的 FloatingArray 数组，将第一个元素转换为浮点数
    expected = pd.array([float(values[0]), 2, None], dtype="Float64")

    # 使用 pd.array 构造结果数组，指定 dtype 为 "Float64"
    res = pd.array(values, dtype="Float64")
    # 使用 assert_extension_array_equal 方法断言两个 ExtensionArray 数组是否相等
    tm.assert_extension_array_equal(res, expected)

    # 使用 FloatingArray._from_sequence 方法构造结果数组
    res = FloatingArray._from_sequence(values)
    # 再次使用 assert_extension_array_equal 方法断言两个 ExtensionArray 数组是否相等
    tm.assert_extension_array_equal(res, expected)


def test_to_array_inferred_dtype():
    # 如果 values 具有 dtype -> 尊重它
    result = pd.array(np.array([1, 2], dtype="float32"))
    # 断言结果数组的 dtype 是 Float32Dtype 类型
    assert result.dtype == Float32Dtype()

    # 如果 values 没有 dtype -> 始终使用 float64
    result = pd.array([1.0, 2.0])
    # 断言结果数组的 dtype 是 Float64Dtype 类型
    assert result.dtype == Float64Dtype()


def test_to_array_dtype_keyword():
    # 使用指定的 dtype 构造数组
    result = pd.array([1, 2], dtype="Float32")
    # 断言结果数组的 dtype 是 Float32Dtype 类型
    assert result.dtype == Float32Dtype()

    # 如果 values 具有 dtype -> 覆盖它
    result = pd.array(np.array([1, 2], dtype="float32"), dtype="Float64")
    # 断言结果数组的 dtype 是 Float64Dtype 类型
    assert result.dtype == Float64Dtype()


def test_to_array_integer():
    # 使用指定的 dtype 构造数组，并预期的浮点数
    result = pd.array([1, 2], dtype="Float64")
    expected = pd.array([1.0, 2.0], dtype="Float64")
    # 使用 assert_extension_array_equal 方法断言两个 ExtensionArray 数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 对于整数 dtype，不会保留 itemsize
    # TODO 在一般情况下我们可以指定 "floating" 吗？
    result = pd.array(np.array([1, 2], dtype="int32"), dtype="Float64")
    # 断言结果数组的 dtype 是 Float64Dtype 类型
    assert result.dtype == Float64Dtype()


@pytest.mark.parametrize(
    "bool_values, values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Float64Dtype(), Float64Dtype()),
        ([False, True], [0, 1], "Float64", Float64Dtype()),
        ([False, True, np.nan], [0, 1, np.nan], Float64Dtype(), Float64Dtype()),
    ],
)
def test_to_array_bool(bool_values, values, target_dtype, expected_dtype):
    # 使用指定的 dtype 构造布尔数组
    result = pd.array(bool_values, dtype=target_dtype)
    # 断言结果数组的 dtype 是预期的 expected_dtype
    assert result.dtype == expected_dtype
    # 使用 pd.array 构造期望的数组，指定 dtype
    expected = pd.array(values, dtype=target_dtype)
    # 使用 assert_extension_array_equal 方法断言两个 ExtensionArray 数组是否相等
    tm.assert_extension_array_equal(result, expected)
# 基于给定数据的数据类型构造一个 Series 对象
def test_series_from_float(data):
    # 获取数据的 dtype
    dtype = data.dtype

    # 从浮点数构造 Series 对象
    expected = pd.Series(data)
    # 使用 to_numpy 方法将数据转换为 NumPy 数组，处理 NaN 值并指定数据类型为 "float"
    result = pd.Series(data.to_numpy(na_value=np.nan, dtype="float"), dtype=str(dtype))
    # 使用 assert_series_equal 函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 从列表构造 Series 对象
    expected = pd.Series(data)
    # 将数据转换为列表并构造 Series 对象，指定数据类型为数据原始的字符串形式
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    # 使用 assert_series_equal 函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```