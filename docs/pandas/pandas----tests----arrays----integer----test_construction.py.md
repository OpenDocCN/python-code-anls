# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_construction.py`

```
# 导入 numpy 库，命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库，命名为 pd
import pandas as pd
# 导入 pandas 内部测试模块
import pandas._testing as tm
# 导入 pandas 的类型检查工具，判断是否为整数
from pandas.api.types import is_integer
# 导入 pandas 的整数数组相关模块
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int32Dtype,
    Int64Dtype,
)

# 定义 pytest 的 fixture，返回一个参数化的 IntegerArray，从给定序列中生成
@pytest.fixture(params=[pd.array, IntegerArray._from_sequence])
def constructor(request):
    """Fixture returning parametrized IntegerArray from given sequence.

    Used to test dtype conversions.
    """
    return request.param


# 测试函数，验证 IntegerArray 中使用 pd.NA 的情况
def test_uses_pandas_na():
    # 创建一个包含整数和 NA 值的 IntegerArray 对象
    a = pd.array([1, None], dtype=Int64Dtype())
    # 断言第二个元素是 pd.NA
    assert a[1] is pd.NA


# 测试函数，测试从浮点数创建 IntegerArray 对象
def test_from_dtype_from_float(data):
    # 从参数中获取数据的数据类型
    dtype = data.dtype

    # 从浮点数创建 Series 对象
    expected = pd.Series(data)
    # 使用 to_numpy 方法将数据转换为浮点数，构造一个新的 Series 对象
    result = pd.Series(data.to_numpy(na_value=np.nan, dtype="float"), dtype=str(dtype))
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 从整数或列表创建 Series 对象
    expected = pd.Series(data)
    # 将数据转换为列表，构造一个新的 Series 对象
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 从整数或数组创建 Series 对象
    expected = pd.Series(data).dropna().reset_index(drop=True)
    # 将数据转换为指定数据类型的数组，构造一个新的 Series 对象
    dropped = np.array(data.dropna()).astype(np.dtype(dtype.type))
    result = pd.Series(dropped, dtype=str(dtype))
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)


# 测试函数，测试类型转换
def test_conversions(data_missing):
    # 将数据转换为对象类型的 Series 对象
    df = pd.DataFrame({"A": data_missing})
    result = df["A"].astype("object")
    expected = pd.Series(np.array([pd.NA, 1], dtype=object), name="A")
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)

    # 将 Series 对象转换为对象类型的 ndarray
    result = df["A"].astype("object").values
    expected = np.array([pd.NA, 1], dtype=object)
    # 断言两个 ndarray 对象相等
    tm.assert_numpy_array_equal(result, expected)

    # 逐一比较结果和期望值
    for r, e in zip(result, expected):
        if pd.isnull(r):
            assert pd.isnull(e)
        elif is_integer(r):
            assert r == e
            assert is_integer(e)
        else:
            assert r == e
            assert type(r) == type(e)


# 测试函数，测试 IntegerArray 的构造函数
def test_integer_array_constructor():
    values = np.array([1, 2, 3, 4], dtype="int64")
    mask = np.array([False, False, False, True], dtype="bool")

    # 使用给定的值和掩码创建 IntegerArray 对象
    result = IntegerArray(values, mask)
    expected = pd.array([1, 2, 3, np.nan], dtype="Int64")
    # 断言两个扩展数组对象相等
    tm.assert_extension_array_equal(result, expected)

    # 预期的错误消息，匹配正则表达式，检查是否抛出 TypeError 异常
    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.astype(float), mask)
    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values)


# 测试函数，测试 IntegerArray 的构造函数（复制版本）
def test_integer_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype="int64")
    # 创建一个布尔类型的 NumPy 数组，表示掩码（mask），长度为4，只有最后一个元素为True，其余为False
    mask = np.array([False, False, False, True], dtype="bool")
    
    # 使用给定的数值数组（values）和上面创建的掩码（mask），初始化一个 IntegerArray 对象
    result = IntegerArray(values, mask)
    
    # 断言：确保 IntegerArray 对象的 _data 属性与传入的数值数组 values 是同一个对象
    assert result._data is values
    
    # 断言：确保 IntegerArray 对象的 _mask 属性与传入的掩码 mask 是同一个对象
    assert result._mask is mask
    
    # 使用给定的数值数组（values）、掩码（mask）和 copy=True 参数，初始化另一个 IntegerArray 对象
    result = IntegerArray(values, mask, copy=True)
    
    # 断言：确保设置了 copy=True 后，IntegerArray 对象的 _data 属性与传入的数值数组 values 不是同一个对象
    assert result._data is not values
    
    # 断言：确保设置了 copy=True 后，IntegerArray 对象的 _mask 属性与传入的掩码 mask 不是同一个对象
    assert result._mask is not mask
@pytest.mark.parametrize(
    "a, b",
    [  # 参数化测试用例，a 和 b 分别是输入和预期输出
        ([1, None], [1, np.nan]),  # 测试输入包含整数和 None，预期输出包含整数和 NaN
        ([None], [np.nan]),  # 测试输入为 None，预期输出为 NaN
        ([None, np.nan], [np.nan, np.nan]),  # 测试输入包含 None 和 NaN，预期输出都是 NaN
        ([np.nan, np.nan], [np.nan, np.nan]),  # 测试输入全是 NaN，预期输出也全是 NaN
    ],
)
def test_to_integer_array_none_is_nan(a, b):
    result = pd.array(a, dtype="Int64")  # 使用 pd.array 将输入数组 a 转换为整数数组
    expected = pd.array(b, dtype="Int64")  # 使用 pd.array 将预期输出数组 b 转换为整数数组
    tm.assert_extension_array_equal(result, expected)  # 断言 result 和 expected 数组相等


@pytest.mark.parametrize(
    "values",
    [  # 参数化测试用例，values 是输入值
        ["foo", "bar"],  # 测试字符串数组
        "foo",  # 测试单个字符串
        1,  # 测试整数
        1.0,  # 测试浮点数
        pd.date_range("20130101", periods=2),  # 测试日期范围
        np.array(["foo"]),  # 测试 NumPy 字符串数组
        [[1, 2], [3, 4]],  # 测试嵌套列表
        [np.nan, {"a": 1}],  # 测试包含 NaN 和字典的数组
    ],
)
def test_to_integer_array_error(values):
    # 在将现有数组转换为 IntegerArrays 时可能出现的错误
    msg = "|".join(
        [
            r"cannot be converted to IntegerDtype",  # 不能转换为整数类型
            r"invalid literal for int\(\) with base 10:",  # 无效的整数文本
            r"values must be a 1D list-like",  # 值必须是一维列表
            r"Cannot pass scalar",  # 不能传递标量
            r"int\(\) argument must be a string",  # int() 参数必须是字符串
        ]
    )
    with pytest.raises((ValueError, TypeError), match=msg):  # 使用 pytest 断言捕获 ValueError 或 TypeError 异常，并匹配预期错误消息
        pd.array(values, dtype="Int64")  # 尝试使用 pd.array 将 values 转换为整数数组

    with pytest.raises((ValueError, TypeError), match=msg):  # 同上，测试 IntegerArray._from_sequence 函数
        IntegerArray._from_sequence(values)


def test_to_integer_array_inferred_dtype(constructor):
    # 如果 values 已经有 dtype，则尊重它
    result = constructor(np.array([1, 2], dtype="int8"))  # 使用 constructor 函数构造整数数组
    assert result.dtype == Int8Dtype()  # 断言结果的 dtype 是 Int8Dtype
    result = constructor(np.array([1, 2], dtype="int32"))  # 使用 constructor 函数构造整数数组
    assert result.dtype == Int32Dtype()  # 断言结果的 dtype 是 Int32Dtype

    # 如果 values 没有 dtype，则始终使用 int64
    result = constructor([1, 2])  # 使用 constructor 函数构造整数数组
    assert result.dtype == Int64Dtype()  # 断言结果的 dtype 是 Int64Dtype()


def test_to_integer_array_dtype_keyword(constructor):
    result = constructor([1, 2], dtype="Int8")  # 使用 constructor 函数构造整数数组，指定 dtype 为 Int8
    assert result.dtype == Int8Dtype()  # 断言结果的 dtype 是 Int8Dtype

    # 如果 values 已经有 dtype，则覆盖它
    result = constructor(np.array([1, 2], dtype="int8"), dtype="Int32")  # 使用 constructor 函数构造整数数组，指定 dtype 为 Int32
    assert result.dtype == Int32Dtype()  # 断言结果的 dtype 是 Int32Dtype()


def test_to_integer_array_float():
    result = IntegerArray._from_sequence([1.0, 2.0], dtype="Int64")  # 使用 IntegerArray._from_sequence 函数构造整数数组，输入是浮点数数组
    expected = pd.array([1, 2], dtype="Int64")  # 使用 pd.array 构造预期输出的整数数组
    tm.assert_extension_array_equal(result, expected)  # 断言 result 和 expected 数组相等

    with pytest.raises(TypeError, match="cannot safely cast non-equivalent"):  # 测试无法安全转换的错误情况
        IntegerArray._from_sequence([1.5, 2.0], dtype="Int64")

    # 对于浮点数 dtype，itemsize 不保留
    result = IntegerArray._from_sequence(
        np.array([1.0, 2.0], dtype="float32"), dtype="Int64"
    )  # 使用 IntegerArray._from_sequence 函数构造整数数组，输入是浮点数数组，指定 dtype 为 Int64
    assert result.dtype == Int64Dtype()  # 断言结果的 dtype 是 Int64Dtype()


def test_to_integer_array_str():
    result = IntegerArray._from_sequence(["1", "2", None], dtype="Int64")  # 使用 IntegerArray._from_sequence 函数构造整数数组，输入是字符串数组
    expected = pd.array([1, 2, np.nan], dtype="Int64")  # 使用 pd.array 构造预期输出的整数数组
    tm.assert_extension_array_equal(result, expected)  # 断言 result 和 expected 数组相等

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):  # 测试无效整数文本的错误情况
        IntegerArray._from_sequence(["1", "2", ""], dtype="Int64")

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):  # 同上，测试无效整数文本的错误情况
        IntegerArray._from_sequence(["1", "2", "not a number"], dtype="Int64")


这里是完整的注释代码块，按照要求为每一行代码进行了详细的注释说明。
    ):
        # 使用 IntegerArray 类的 _from_sequence 方法，从字符串数组 ["1.5", "2.0"] 创建一个整数数组
        IntegerArray._from_sequence(["1.5", "2.0"], dtype="Int64")
# 使用 pytest 的 mark.parametrize 装饰器定义了一个参数化测试函数，测试函数名为 test_to_integer_array_bool。
# 参数化测试用例包括四组输入：
# - bool_values: 布尔值列表
# - int_values: 整数值列表
# - target_dtype: 目标数据类型，可以是 Int64Dtype 对象或字符串 "Int64"
# - expected_dtype: 预期的结果数据类型，也是 Int64Dtype 对象

@pytest.mark.parametrize(
    "bool_values, int_values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Int64Dtype(), Int64Dtype()),  # 测试整数数组的布尔转换
        ([False, True], [0, 1], "Int64", Int64Dtype()),       # 测试整数数组的布尔转换（使用字符串指定类型）
        ([False, True, np.nan], [0, 1, np.nan], Int64Dtype(), Int64Dtype()),  # 测试带 NaN 的整数数组的布尔转换
    ],
)

# 定义测试函数 test_to_integer_array_bool，用于测试将布尔数组转换为整数数组的行为
def test_to_integer_array_bool(
    constructor, bool_values, int_values, target_dtype, expected_dtype
):
    # 调用被测试的构造函数 constructor，将 bool_values 转换为整数数组，指定目标数据类型为 target_dtype
    result = constructor(bool_values, dtype=target_dtype)
    # 断言结果的数据类型是否符合预期
    assert result.dtype == expected_dtype
    # 使用 pandas 的 array 函数创建预期结果，将 int_values 转换为目标数据类型
    expected = pd.array(int_values, dtype=target_dtype)
    # 使用 pandas 的测试工具 tm 来断言扩展数组 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 使用 pytest 的 mark.parametrize 装饰器定义另一个参数化测试函数，测试函数名为 test_to_integer_array。
# 参数化测试用例包括三组输入：
# - values: 数组值，类型为 int64 的 numpy 数组
# - to_dtype: 目标数据类型，可以为 None 或字符串类型
# - result_dtype: 预期结果数据类型，是一个数据类型对象（如 Int64Dtype）

@pytest.mark.parametrize(
    "values, to_dtype, result_dtype",
    [
        (np.array([1], dtype="int64"), None, Int64Dtype),   # 测试将 int64 数组转换为整数数组
        (np.array([1, np.nan]), None, Int64Dtype),          # 测试将包含 NaN 的 int64 数组转换为整数数组
        (np.array([1, np.nan]), "int8", Int8Dtype),         # 测试将 int64 数组转换为 int8 类型的整数数组
    ],
)

# 定义测试函数 test_to_integer_array，用于测试将数组转换为整数数组的行为
def test_to_integer_array(values, to_dtype, result_dtype):
    # 调用 IntegerArray 的 _from_sequence 方法，将 values 序列转换为整数数组，指定目标数据类型为 to_dtype
    result = IntegerArray._from_sequence(values, dtype=to_dtype)
    # 断言结果的数据类型是否符合预期
    assert result.dtype == result_dtype()
    # 使用 pandas 的 array 函数创建预期结果，将 values 转换为 result_dtype 指定的数据类型
    expected = pd.array(values, dtype=result_dtype())
    # 使用 pandas 的测试工具 tm 来断言扩展数组 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_integer_array_from_boolean，测试从布尔数组创建整数数组的行为
def test_integer_array_from_boolean():
    # GH31104：测试用例标识符
    # 创建预期结果，将包含 True 和 False 的布尔数组转换为 Int64 类型的整数数组
    expected = pd.array(np.array([True, False]), dtype="Int64")
    # 调用 pandas 的 array 函数，将包含 True 和 False 的布尔数组转换为 Int64 类型的整数数组
    result = pd.array(np.array([True, False], dtype=object), dtype="Int64")
    # 使用 pandas 的测试工具 tm 来断言扩展数组 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)
```