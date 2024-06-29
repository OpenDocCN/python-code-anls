# `D:\src\scipysrc\pandas\pandas\tests\arrays\string_\test_string.py`

```
"""
This module tests the functionality of StringArray and ArrowStringArray.
Tests for the str accessors are in pandas/tests/strings/test_string_array.py
"""

import operator  # 导入操作符模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas.compat.pyarrow import pa_version_under12p0  # 导入兼容性函数

from pandas.core.dtypes.common import is_dtype_equal  # 导入数据类型比较函数

import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入 Pandas 测试工具模块
from pandas.core.arrays.string_arrow import (  # 导入字符串数组相关模块
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)


def na_val(dtype):
    """根据 dtype 返回相应的缺失值标识"""
    if dtype.storage == "pyarrow_numpy":
        return np.nan
    else:
        return pd.NA


@pytest.fixture
def dtype(string_storage):
    """根据参数化的 'string_storage' 返回 StringDtype 的 fixture"""
    return pd.StringDtype(storage=string_storage)


@pytest.fixture
def cls(dtype):
    """根据参数化的 'dtype' 返回数组类型的 fixture"""
    return dtype.construct_array_type()


def test_repr(dtype):
    """测试 DataFrame 和 Series 的字符串表示形式是否正确"""
    df = pd.DataFrame({"A": pd.array(["a", pd.NA, "b"], dtype=dtype)})

    # 检查 DataFrame 的字符串表示形式是否符合预期
    if dtype.storage == "pyarrow_numpy":
        expected = "     A\n0    a\n1  NaN\n2    b"
    else:
        expected = "      A\n0     a\n1  <NA>\n2     b"
    assert repr(df) == expected

    # 检查 Series 的字符串表示形式是否符合预期
    if dtype.storage == "pyarrow_numpy":
        expected = "0      a\n1    NaN\n2      b\nName: A, dtype: string"
    else:
        expected = "0       a\n1    <NA>\n2       b\nName: A, dtype: string"
    assert repr(df.A) == expected

    # 检查不同存储类型下的 ExtensionArray 的字符串表示形式是否符合预期
    if dtype.storage == "pyarrow":
        arr_name = "ArrowStringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    elif dtype.storage == "pyarrow_numpy":
        arr_name = "ArrowStringArrayNumpySemantics"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: string"
    else:
        arr_name = "StringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    assert repr(df.A.array) == expected


def test_none_to_nan(cls, dtype):
    """测试将 None 转换为对应的缺失值标识"""
    a = cls._from_sequence(["a", None, "b"], dtype=dtype)
    assert a[1] is not None
    assert a[1] is na_val(a.dtype)


def test_setitem_validates(cls, dtype):
    """测试设置数组元素时的类型验证"""
    arr = cls._from_sequence(["a", "b"], dtype=dtype)

    # 检查在 StringArray 中设置非字符串值时的异常消息
    if cls is pd.arrays.StringArray:
        msg = "Cannot set non-string value '10' into a StringArray."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        arr[0] = 10

    # 检查在 StringArray 中使用非字符串数组时的异常消息
    if cls is pd.arrays.StringArray:
        msg = "Must provide strings."
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        arr[:] = np.array([1, 2])


def test_setitem_with_scalar_string(dtype):
    """测试使用标量字符串设置数组元素"""
    # is_float_dtype 函数可能将一些字符串（如 'd'）视为浮点数，可能会引起问题。
    arr = pd.array(["a", "c"], dtype=dtype)
    arr[0] = "d"
    expected = pd.array(["d", "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_with_array_with_missing(dtype):
    # 创建一个 Pandas ExtensionArray 对象 `arr`，包含字符串 "a", "b", "c"，使用指定的数据类型 `dtype`
    arr = pd.array(["a", "b", "c"], dtype=dtype)
    
    # 创建一个 NumPy 数组 `value`，包含元素 "A" 和 None
    value = np.array(["A", None])
    
    # 备份 `value` 数组的原始副本，以便后续比较
    value_orig = value.copy()
    
    # 修改 `arr` 对象中索引为 0 和 1 的位置，用 `value` 数组的值进行设置
    arr[[0, 1]] = value
    
    # 创建一个预期的 Pandas ExtensionArray `expected`，包含元素 "A", pd.NA, "c"，使用指定的数据类型 `dtype`
    expected = pd.array(["A", pd.NA, "c"], dtype=dtype)
    
    # 使用测试框架中的函数检查 `arr` 和 `expected` 是否相等
    tm.assert_extension_array_equal(arr, expected)
    
    # 使用测试框架中的函数检查 `value` 和 `value_orig` 是否相等
    tm.assert_numpy_array_equal(value, value_orig)
# 定义一个函数用于测试数据类型转换的往返过程
def test_astype_roundtrip(dtype):
    # 创建一个包含日期范围的时间序列，并将第一个元素设置为None
    ser = pd.Series(pd.date_range("2000", periods=12))
    ser[0] = None

    # 将时间序列转换为指定的数据类型
    casted = ser.astype(dtype)
    # 断言转换后的数据类型与期望的数据类型相等
    assert is_dtype_equal(casted.dtype, dtype)

    # 将转换后的时间序列再次转换为datetime64[ns]类型，并断言与原始序列相等
    result = casted.astype("datetime64[ns]")
    tm.assert_series_equal(result, ser)

    # GH#38509：对timedelta64类型进行相同的操作
    # 计算时间序列与其最后一个元素的差值
    ser2 = ser - ser.iloc[-1]
    # 将差值序列转换为指定的数据类型
    casted2 = ser2.astype(dtype)
    # 断言转换后的数据类型与期望的数据类型相等
    assert is_dtype_equal(casted2.dtype, dtype)

    # 将转换后的差值序列再次转换为与原始差值序列相同的数据类型，并断言相等
    result2 = casted2.astype(ser2.dtype)
    tm.assert_series_equal(result2, ser2)


# 定义一个函数用于测试序列相加的不同方式
def test_add(dtype):
    # 创建两个包含字符串和None值的序列a和b，指定数据类型为dtype
    a = pd.Series(["a", "b", "c", None, None], dtype=dtype)
    b = pd.Series(["x", "y", None, "z", None], dtype=dtype)

    # 执行序列的加法操作，并断言结果与期望相等
    result = a + b
    expected = pd.Series(["ax", "by", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    # 使用add方法执行序列的加法操作，并断言结果与期望相等
    result = a.add(b)
    tm.assert_series_equal(result, expected)

    # 使用radd方法执行序列的右加操作，并断言结果与期望相等
    result = a.radd(b)
    expected = pd.Series(["xa", "yb", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    # 使用fill_value参数执行序列的加法操作，并断言结果与期望相等
    result = a.add(b, fill_value="-")
    expected = pd.Series(["ax", "by", "c-", "-z", None], dtype=dtype)
    tm.assert_series_equal(result, expected)


# 定义一个函数用于测试二维序列与标量的加法操作
def test_add_2d(dtype, request, arrow_string_storage):
    # 如果数据类型的存储类型在arrow_string_storage中，则标记为xfail，并设定原因
    if dtype.storage in arrow_string_storage:
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.applymarker(mark)

    # 创建一个包含字符串的一维序列a和一个包含字符串的二维数组b，指定数据类型为dtype
    a = pd.array(["a", "b", "c"], dtype=dtype)
    b = np.array([["a", "b", "c"]], dtype=object)
    
    # 使用assert语句断言加法操作会引发ValueError异常，匹配异常信息"3 != 1"
    with pytest.raises(ValueError, match="3 != 1"):
        a + b

    # 将序列a转换为Series对象s，并使用assert语句断言加法操作会引发ValueError异常，匹配异常信息"3 != 1"
    s = pd.Series(a)
    with pytest.raises(ValueError, match="3 != 1"):
        s + b


# 定义一个函数用于测试序列与列表的加法操作
def test_add_sequence(dtype):
    # 创建一个包含字符串和None值的序列a，以及一个包含字符串和None值的列表other
    a = pd.array(["a", "b", None, None], dtype=dtype)
    other = ["x", None, "y", None]

    # 执行序列与列表的加法操作，并断言结果与期望相等
    result = a + other
    expected = pd.array(["ax", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行列表与序列的加法操作，并断言结果与期望相等
    result = other + a
    expected = pd.array(["xa", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


# 定义一个函数用于测试序列与标量的乘法操作
def test_mul(dtype):
    # 创建一个包含字符串和None值的序列a，以及对其进行乘法操作后的期望结果expected
    a = pd.array(["a", "b", None], dtype=dtype)
    result = a * 2
    expected = pd.array(["aa", "bb", None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行标量与序列的乘法操作，并断言结果与期望相等
    result = 2 * a
    tm.assert_extension_array_equal(result, expected)


# 标记为xfail的测试函数，用于测试序列与DataFrame对象的加法操作
@pytest.mark.xfail(reason="GH-28527")
def test_add_strings(dtype):
    # 创建一个包含字符串的序列arr和一个包含字符串的DataFrame对象df
    arr = pd.array(["a", "b", "c", "d"], dtype=dtype)
    df = pd.DataFrame([["t", "y", "v", "w"]], dtype=object)
    # 断言序列与DataFrame对象的加法操作未实现
    assert arr.__add__(df) is NotImplemented

    # 执行序列与DataFrame对象的加法操作，并断言结果与期望相等
    result = arr + df
    expected = pd.DataFrame([["at", "by", "cv", "dw"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    # 执行DataFrame对象与序列的加法操作，并断言结果与期望相等
    result = df + arr
    expected = pd.DataFrame([["ta", "yb", "vc", "wd"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


# 标记为xfail的测试函数，用于测试序列与包含NaN值的数组的加法操作
@pytest.mark.xfail(reason="GH-28527")
def test_add_frame(dtype):
    # 创建一个包含字符串和NaN值的序列arr
    arr = pd.array(["a", "b", np.nan, np.nan], dtype=dtype)
    # 创建一个包含单行数据的 Pandas 数据帧，包括字符串和 NaN 值
    df = pd.DataFrame([["x", np.nan, "y", np.nan]])
    
    # 断言检查尝试对数组 `arr` 执行 `__add__` 操作是否返回 `NotImplemented`
    assert arr.__add__(df) is NotImplemented
    
    # 将数组 `arr` 与数据帧 `df` 执行加法运算，并将结果存储在 `result` 变量中
    result = arr + df
    
    # 期望的结果数据帧 `expected`，包含根据 dtype 转换的预期值
    expected = pd.DataFrame([["ax", np.nan, np.nan, np.nan]]).astype(dtype)
    
    # 使用测试工具比较 `result` 和 `expected` 数据帧是否相等
    tm.assert_frame_equal(result, expected)
    
    # 将数据帧 `df` 与数组 `arr` 执行加法运算，并将结果存储在 `result` 变量中
    result = df + arr
    
    # 期望的结果数据帧 `expected`，包含根据 dtype 转换的预期值
    expected = pd.DataFrame([["xa", np.nan, np.nan, np.nan]]).astype(dtype)
    
    # 使用测试工具比较 `result` 和 `expected` 数据帧是否相等
    tm.assert_frame_equal(result, expected)
# 测试比较方法，比较数组中的每个元素与标量的关系
def test_comparison_methods_scalar(comparison_op, dtype):
    # 获取比较运算符的名称
    op_name = f"__{comparison_op.__name__}__"
    # 创建包含字符串和空值的 Pandas 数组对象
    a = pd.array(["a", None, "c"], dtype=dtype)
    # 设置比较的标量值为字符串 "a"
    other = "a"
    # 使用 getattr 动态调用 Pandas 数组对象的比较方法
    result = getattr(a, op_name)(other)
    
    # 根据数据类型的存储方式进行不同的预期结果比较
    if dtype.storage == "pyarrow_numpy":
        # 如果存储方式是 pyarrow_numpy，生成预期的 NumPy 数组
        expected = np.array([getattr(item, op_name)(other) for item in a])
        # 对于不等于操作，调整预期结果的第二个元素为 True
        if comparison_op == operator.ne:
            expected[1] = True
        else:
            expected[1] = False
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_numpy_array_equal(result, expected.astype(np.bool_))
    else:
        # 对于其他存储方式，设置预期的数据类型
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        # 生成预期的 Pandas 数组对象，类型为 object
        expected = np.array([getattr(item, op_name)(other) for item in a], dtype=object)
        expected = pd.array(expected, dtype=expected_dtype)
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_extension_array_equal(result, expected)


# 测试比较方法，使用 Pandas NA 值作为标量与数组元素比较
def test_comparison_methods_scalar_pd_na(comparison_op, dtype):
    # 获取比较运算符的名称
    op_name = f"__{comparison_op.__name__}__"
    # 创建包含字符串和空值的 Pandas 数组对象
    a = pd.array(["a", None, "c"], dtype=dtype)
    # 使用 getattr 动态调用 Pandas 数组对象的比较方法，比较 Pandas 的 NA 值
    result = getattr(a, op_name)(pd.NA)

    # 根据数据类型的存储方式生成预期结果
    if dtype.storage == "pyarrow_numpy":
        # 对于 pyarrow_numpy 存储方式，生成预期的 NumPy 数组
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_numpy_array_equal(result, expected)
    else:
        # 对于其他存储方式，设置预期的数据类型
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        # 生成预期的 Pandas 数组对象，全为 None 值
        expected = pd.array([None, None, None], dtype=expected_dtype)
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_extension_array_equal(result, expected)


# 测试比较方法，使用非字符串的标量与数组元素比较
def test_comparison_methods_scalar_not_string(comparison_op, dtype):
    # 获取比较运算符的名称
    op_name = f"__{comparison_op.__name__}__"
    # 创建包含字符串和空值的 Pandas 数组对象
    a = pd.array(["a", None, "c"], dtype=dtype)
    # 设置非字符串的标量值为整数 42
    other = 42

    # 如果比较运算符不是等于或不等于，则应触发 TypeError 异常
    if op_name not in ["__eq__", "__ne__"]:
        with pytest.raises(TypeError, match="Invalid comparison|not supported between"):
            getattr(a, op_name)(other)
        return

    # 否则，正常比较并获取结果
    result = getattr(a, op_name)(other)

    # 根据数据类型的存储方式生成预期结果
    if dtype.storage == "pyarrow_numpy":
        # 使用字典映射获取预期的 NumPy 数组
        expected_data = {
            "__eq__": [False, False, False],
            "__ne__": [True, True, True],
        }[op_name]
        expected = np.array(expected_data)
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_numpy_array_equal(result, expected)
    else:
        # 使用字典映射获取预期的 Pandas 数组和数据类型
        expected_data = {"__eq__": [False, None, False], "__ne__": [True, None, True]}[op_name]
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array(expected_data, dtype=expected_dtype)
        # 使用 Pandas 的测试工具函数检查结果是否符合预期
        tm.assert_extension_array_equal(result, expected)


# 测试比较方法，比较数组与数组的元素逐个比较
def test_comparison_methods_array(comparison_op, dtype):
    # 获取比较运算符的名称
    op_name = f"__{comparison_op.__name__}__"
    # 创建包含字符串和空值的 Pandas 数组对象
    a = pd.array(["a", None, "c"], dtype=dtype)
    # 设置作为比较数组的另一个数组
    other = [None, None, "c"]
    # 使用 getattr 动态调用 Pandas 数组对象的比较方法
    result = getattr(a, op_name)(other)
    # 如果数据类型的存储方式是 "pyarrow_numpy"
    if dtype.storage == "pyarrow_numpy":
        # 如果比较运算符是 !=
        if operator.ne == comparison_op:
            # 设置预期结果为一个 numpy 数组，包含 [True, True, False]
            expected = np.array([True, True, False])
        else:
            # 否则设置预期结果为一个 numpy 数组，包含 [False, False, False]
            expected = np.array([False, False, False])
            # 在预期结果的最后一个元素上调用指定操作符和数组的最后一个元素
            expected[-1] = getattr(other[-1], op_name)(a[-1])
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

        # 调用对象 a 的指定操作符方法，传入 pd.NA
        result = getattr(a, op_name)(pd.NA)
        # 如果比较运算符是 !=
        if operator.ne == comparison_op:
            # 设置预期结果为一个 numpy 数组，包含 [True, True, True]
            expected = np.array([True, True, True])
        else:
            # 否则设置预期结果为一个 numpy 数组，包含 [False, False, False]
            expected = np.array([False, False, False])
        # 断言 numpy 数组的相等性
        tm.assert_numpy_array_equal(result, expected)

    else:
        # 确定预期数据类型字符串，根据 dtype.storage 的值
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        # 创建一个长度为 a 的长度，元素全为 None 的 numpy 数组
        expected = np.full(len(a), fill_value=None, dtype="object")
        # 在预期结果的最后一个元素上调用指定操作符和数组的最后一个元素
        expected[-1] = getattr(other[-1], op_name)(a[-1])
        # 创建一个 pandas 的延展数组，使用指定的预期数据类型
        expected = pd.array(expected, dtype=expected_dtype)
        # 断言延展数组的相等性
        tm.assert_extension_array_equal(result, expected)

        # 调用对象 a 的指定操作符方法，传入 pd.NA
        result = getattr(a, op_name)(pd.NA)
        # 创建一个 pandas 的延展数组，包含 [None, None, None]，使用指定的预期数据类型
        expected = pd.array([None, None, None], dtype=expected_dtype)
        # 断言延展数组的相等性
        tm.assert_extension_array_equal(result, expected)
# 测试构造函数是否会引发异常
def test_constructor_raises(cls):
    # 如果 cls 是 pd.arrays.StringArray，则设置特定错误消息
    if cls is pd.arrays.StringArray:
        msg = "StringArray requires a sequence of strings or pandas.NA"
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"

    # 断言传入 np.array(["a", "b"], dtype="S1") 会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", "b"], dtype="S1"))

    # 断言传入空数组会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        cls(np.array([]))

    if cls is pd.arrays.StringArray:
        # 对于字符串数组，np.nan 和 None 不会引发异常，因为它们被视为有效的 NA 值
        cls(np.array(["a", np.nan], dtype=object))
        cls(np.array(["a", None], dtype=object))
    else:
        # 断言传入包含 np.nan 的数组会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", np.nan], dtype=object))
        # 断言传入包含 None 的数组会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", None], dtype=object))

    # 断言传入包含 pd.NaT 的数组会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", pd.NaT], dtype=object))

    # 断言传入包含 np.datetime64("NaT", "ns") 的数组会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.datetime64("NaT", "ns")], dtype=object))

    # 断言传入包含 np.timedelta64("NaT", "ns") 的数组会引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.timedelta64("NaT", "ns")], dtype=object))


# 测试构造函数处理 NaN 类型值
@pytest.mark.parametrize("na", [np.nan, np.float64("nan"), float("nan"), None, pd.NA])
def test_constructor_nan_like(na):
    # 预期的 StringArray 对象，初始化时传入包含 NaN 或 pd.NA 的数组
    expected = pd.arrays.StringArray(np.array(["a", pd.NA]))
    # 断言实际生成的 StringArray 对象与预期对象相等
    tm.assert_extension_array_equal(
        pd.arrays.StringArray(np.array(["a", na], dtype="object")), expected
    )


# 测试从序列创建对象且不改变原始数组
@pytest.mark.parametrize("copy", [True, False])
def test_from_sequence_no_mutate(copy, cls, dtype):
    nan_arr = np.array(["a", np.nan], dtype=object)
    expected_input = nan_arr.copy()
    na_arr = np.array(["a", pd.NA], dtype=object)

    # 调用 _from_sequence 方法创建对象
    result = cls._from_sequence(nan_arr, dtype=dtype, copy=copy)

    if cls in (ArrowStringArray, ArrowStringArrayNumpySemantics):
        import pyarrow as pa

        # 使用 pyarrow 创建 ArrowStringArray 对象的预期结果
        expected = cls(pa.array(na_arr, type=pa.string(), from_pandas=True))
    else:
        # 创建普通的 ArrowStringArray 对象的预期结果
        expected = cls(na_arr)

    # 断言实际生成的对象与预期对象相等
    tm.assert_extension_array_equal(result, expected)
    # 断言输入的原始数组与预期输入数组相等
    tm.assert_numpy_array_equal(nan_arr, expected_input)


# 测试转换为整型的操作
def test_astype_int(dtype):
    # 创建包含字符串的数组
    arr = pd.array(["1", "2", "3"], dtype=dtype)
    # 将数组转换为 int64 类型
    result = arr.astype("int64")
    # 预期的 int64 类型的数组
    expected = np.array([1, 2, 3], dtype="int64")
    # 断言实际生成的数组与预期数组相等
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)
    # 根据 dtype 的存储类型设置错误类型和消息
    if dtype.storage == "pyarrow_numpy":
        err = ValueError
        msg = "cannot convert float NaN to integer"
    else:
        err = TypeError
        msg = (
            r"int\(\) argument must be a string, a bytes-like "
            r"object or a( real)? number"
        )
    # 断言转换为 int64 类型会引发错误异常，并匹配预期的错误消息
    with pytest.raises(err, match=msg):
        arr.astype("int64")


# 测试转换为可空整型的操作
def test_astype_nullable_int(dtype):
    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)

    # 将数组转换为 Int64 类型
    result = arr.astype("Int64")
    # 预期的 Int64 类型的数组
    expected = pd.array([1, pd.NA, 3], dtype="Int64")
    # 断言实际生成的数组与预期数组相等
    tm.assert_extension_array_equal(result, expected)
def test_astype_float(dtype, any_float_dtype):
    # 创建一个包含字符串和pd.NA的Series对象
    ser = pd.Series(["1.1", pd.NA, "3.3"], dtype=dtype)
    # 将Series对象中的数据类型转换为指定的浮点数数据类型
    result = ser.astype(any_float_dtype)
    # 创建一个期望的Series对象，其中包含转换后的浮点数值和NaN
    expected = pd.Series([1.1, np.nan, 3.3], dtype=any_float_dtype)
    # 使用测试框架检查转换后的结果与期望值是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce(skipna, dtype):
    # 创建一个包含字符串的Series对象
    arr = pd.Series(["a", "b", "c"], dtype=dtype)
    # 调用sum方法对Series对象进行求和操作
    result = arr.sum(skipna=skipna)
    # 使用断言检查求和结果是否符合预期（这里预期结果为字符串"abc"）
    assert result == "abc"


@pytest.mark.xfail(reason="Not implemented StringArray.sum")
def test_reduce_missing(skipna, dtype):
    # 创建一个包含None和字符串的Series对象
    arr = pd.Series([None, "a", None, "b", "c", None], dtype=dtype)
    # 调用sum方法对Series对象进行求和操作
    result = arr.sum(skipna=skipna)
    # 根据skipna的值进行条件判断
    if skipna:
        # 使用断言检查求和结果是否符合预期（这里预期结果为字符串"abc"）
        assert result == "abc"
    else:
        # 使用断言检查结果是否为NaN
        assert pd.isna(result)


@pytest.mark.parametrize("method", ["min", "max"])
def test_min_max(method, skipna, dtype):
    # 创建一个包含字符串和None的Series对象
    arr = pd.Series(["a", "b", "c", None], dtype=dtype)
    # 根据参数method调用对应的方法（min或max）对Series对象进行操作
    result = getattr(arr, method)(skipna=skipna)
    # 根据skipna的值进行条件判断
    if skipna:
        # 根据method的值确定期望的结果
        expected = "a" if method == "min" else "c"
        # 使用断言检查结果是否符合预期
        assert result == expected
    else:
        # 如果skipna为False，使用na_val函数来确定期望的结果
        assert result is na_val(arr.dtype)


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request, arrow_string_storage):
    # 根据条件进行测试标记，处理特定情况下的异常
    if dtype.storage in arrow_string_storage and box is pd.array:
        if box is pd.array:
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            reason = "'ArrowStringArray' object has no attribute 'max'"
        # 应用测试标记
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.applymarker(mark)

    # 根据参数box创建特定类型的数据结构（Series或array）
    arr = box(["a", "b", "c", None], dtype=dtype)
    # 根据method参数调用对应的numpy方法（np.min或np.max）对数据结构进行操作
    result = getattr(np, method)(arr)
    # 根据method的值确定期望的结果
    expected = "a" if method == "min" else "c"
    # 使用断言检查结果是否符合预期
    assert result == expected


def test_fillna_args(dtype, arrow_string_storage):
    # GH 37987

    # 创建一个包含字符串和pd.NA的扩展数组
    arr = pd.array(["a", pd.NA], dtype=dtype)

    # 使用指定值填充缺失值
    res = arr.fillna(value="b")
    expected = pd.array(["a", "b"], dtype=dtype)
    # 使用测试框架检查填充后的结果与期望值是否相等
    tm.assert_extension_array_equal(res, expected)

    # 使用指定值填充缺失值
    res = arr.fillna(value=np.str_("b"))
    # 使用测试框架检查填充后的结果与期望值是否相等
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    # 根据数据类型的存储方式选择性地引发异常
    if dtype.storage in arrow_string_storage:
        msg = "Invalid value '1' for dtype string"
    else:
        msg = "Cannot set non-string value '1' into a StringArray."
    # 使用断言检查特定操作是否引发了预期的异常
    with pytest.raises(TypeError, match=msg):
        arr.fillna(value=1)


def test_arrow_array(dtype):
    # protocol added in 0.15.0
    # 确保pyarrow库可用
    pa = pytest.importorskip("pyarrow")
    import pyarrow.compute as pc

    # 创建一个包含字符串的扩展数组
    data = pd.array(["a", "b", "c"], dtype=dtype)
    # 将pandas扩展数组转换为pyarrow数组
    arr = pa.array(data)
    # 创建一个期望的pyarrow数组
    expected = pa.array(list(data), type=pa.large_string(), from_pandas=True)
    # 根据数据类型的存储方式和pyarrow版本选择性地进行处理
    if dtype.storage in ("pyarrow", "pyarrow_numpy") and pa_version_under12p0:
        expected = pa.chunked_array(expected)
    if dtype.storage == "python":
        expected = pc.cast(expected, pa.string())
    # 使用断言检查数组 `arr` 是否与期望的数组 `expected` 相等
    assert arr.equals(expected)
@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
# 定义测试函数，标记为忽略特定警告
def test_arrow_roundtrip(dtype, string_storage2, request, using_infer_string):
    # 根据需要导入pyarrow模块，如果版本低于1.0.0则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 如果使用infer_string且string_storage2不是"pyarrow_numpy"，标记为xfail（预期失败）
    if using_infer_string and string_storage2 != "pyarrow_numpy":
        request.applymarker(
            pytest.mark.xfail(
                reason="infer_string takes precedence over string storage"
            )
        )

    # 创建包含字符串和None值的Pandas数组
    data = pd.array(["a", "b", None], dtype=dtype)
    # 根据数组创建DataFrame
    df = pd.DataFrame({"a": data})
    # 将DataFrame转换为pyarrow表格
    table = pa.table(df)
    # 根据dtype的存储类型，断言字段"a"的类型是"string"或"large_string"
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    # 在指定的上下文中，将pyarrow表格转换为Pandas DataFrame
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    # 断言结果DataFrame的"a"列的dtype是pd.StringDtype类型
    assert isinstance(result["a"].dtype, pd.StringDtype)
    # 根据指定的string_storage2类型，将原始DataFrame转换为预期的DataFrame
    expected = df.astype(f"string[{string_storage2}]")
    # 使用TestManger（tm）检查两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)
    # 确保缺失值被NA表示，而不是np.nan或None
    assert result.loc[2, "a"] is na_val(result["a"].dtype)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
# 定义测试函数，标记为忽略特定警告
def test_arrow_load_from_zero_chunks(
    dtype, string_storage2, request, using_infer_string
):
    # GH-41040，加载零个数据块的pyarrow表格
    pa = pytest.importorskip("pyarrow")

    # 如果使用infer_string且string_storage2不是"pyarrow_numpy"，标记为xfail（预期失败）
    if using_infer_string and string_storage2 != "pyarrow_numpy":
        request.applymarker(
            pytest.mark.xfail(
                reason="infer_string takes precedence over string storage"
            )
        )

    # 创建空的Pandas数组
    data = pd.array([], dtype=dtype)
    # 根据数组创建DataFrame
    df = pd.DataFrame({"a": data})
    # 将DataFrame转换为pyarrow表格
    table = pa.table(df)
    # 根据dtype的存储类型，断言字段"a"的类型是"string"或"large_string"
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    # 实例化一个没有任何数据块的相同表格
    table = pa.table([pa.chunked_array([], type=pa.string())], schema=table.schema)
    # 在指定的上下文中，将pyarrow表格转换为Pandas DataFrame
    with pd.option_context("string_storage", string_storage2):
        result = table.to_pandas()
    # 断言结果DataFrame的"a"列的dtype是pd.StringDtype类型
    assert isinstance(result["a"].dtype, pd.StringDtype)
    # 根据指定的string_storage2类型，将原始DataFrame转换为预期的DataFrame
    expected = df.astype(f"string[{string_storage2}]")
    # 使用TestManger（tm）检查两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_value_counts_na(dtype):
    # 根据dtype的存储类型，确定预期的dtype
    if getattr(dtype, "storage", "") == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    elif getattr(dtype, "storage", "") == "pyarrow_numpy":
        exp_dtype = "int64"
    else:
        exp_dtype = "Int64"
    # 创建包含字符串和NA值的Pandas数组
    arr = pd.array(["a", "b", "a", pd.NA], dtype=dtype)
    # 计算数组中每个值的出现次数，包括NA值
    result = arr.value_counts(dropna=False)
    # 创建预期的Series，包括NA值，以及相应的dtype和名称
    expected = pd.Series([2, 1, 1], index=arr[[0, 1, 3]], dtype=exp_dtype, name="count")
    # 使用TestManger（tm）检查两个Series是否相等
    tm.assert_series_equal(result, expected)

    # 计算数组中每个值的出现次数，不包括NA值
    result = arr.value_counts(dropna=True)
    # 创建预期的Series，不包括NA值，以及相应的dtype和名称
    expected = pd.Series([2, 1], index=arr[:2], dtype=exp_dtype, name="count")
    # 使用TestManger（tm）检查两个Series是否相等
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize(dtype):
    # 根据dtype的存储类型，确定预期的dtype
    if getattr(dtype, "storage", "") == "pyarrow":
        exp_dtype = "double[pyarrow]"
    # 如果 dtype 对象具有 "storage" 属性且其值为 "pyarrow_numpy"
    elif getattr(dtype, "storage", "") == "pyarrow_numpy":
        # 期望的数据类型为 numpy 的 float64
        exp_dtype = np.float64
    # 否则，假设的数据类型为 "Float64"
    else:
        exp_dtype = "Float64"

    # 创建一个 Pandas Series 对象，包含字符串和 pd.NA（缺失值），指定数据类型为 dtype
    ser = pd.Series(["a", "b", "a", pd.NA], dtype=dtype)

    # 统计 Series 中每个元素的出现次数，并计算其占比
    result = ser.value_counts(normalize=True)

    # 创建一个 Pandas Series 对象，包含两个数值，索引为 ser 的前两个元素，
    # 数据类型为 exp_dtype，名称为 "proportion"，然后每个值除以 3
    expected = pd.Series([2, 1], index=ser[:2], dtype=exp_dtype, name="proportion") / 3

    # 使用测试框架检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试 value_counts 方法，sort 参数为 False 时的情况
def test_value_counts_sort_false(dtype):
    # 根据 dtype 的 storage 属性值判断 exp_dtype 的取值
    if getattr(dtype, "storage", "") == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    elif getattr(dtype, "storage", "") == "pyarrow_numpy":
        exp_dtype = "int64"
    else:
        exp_dtype = "Int64"
    # 创建一个包含字符串数据的 Series 对象
    ser = pd.Series(["a", "b", "c", "b"], dtype=dtype)
    # 调用 value_counts 方法，sort 参数为 False
    result = ser.value_counts(sort=False)
    # 创建一个期望的 Series 对象
    expected = pd.Series([1, 2, 1], index=ser[:3], dtype=exp_dtype, name="count")
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试 memory_usage 方法
def test_memory_usage(dtype, arrow_string_storage):
    # GH 33963

    # 如果 dtype 的 storage 属性值在 arrow_string_storage 中，则跳过测试
    if dtype.storage in arrow_string_storage:
        pytest.skip(f"not applicable for {dtype.storage}")

    # 创建一个包含字符串数据的 Series 对象
    series = pd.Series(["a", "b", "c"], dtype=dtype)

    # 断言 Series 对象的 nbytes 大于 0，且小于 memory_usage() 的返回值，且小于 memory_usage(deep=True) 的返回值
    assert 0 < series.nbytes <= series.memory_usage() < series.memory_usage(deep=True)


# 使用参数化装饰器定义一个测试函数，用于测试 astype 方法从 float 类型转换为指定类型的情况
@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    # https://github.com/pandas-dev/pandas/issues/36451
    # 创建一个包含浮点数数据的 Series 对象
    ser = pd.Series([0.1], dtype=float_dtype)
    # 调用 astype 方法进行类型转换
    result = ser.astype(dtype)
    # 创建一个期望的 Series 对象
    expected = pd.Series(["0.1"], dtype=dtype)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试 to_numpy 方法返回默认值的情况
def test_to_numpy_returns_pdna_default(dtype):
    # 创建一个包含字符串数据的数组对象
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    # 调用 np.array 方法将数组对象转换为 numpy 数组
    result = np.array(arr)
    # 创建一个期望的 numpy 数组
    expected = np.array(["a", na_val(dtype), "b"], dtype=object)
    # 断言两个 numpy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


# 定义一个测试函数，用于测试 to_numpy 方法返回指定 NA 值的情况
def test_to_numpy_na_value(dtype, nulls_fixture):
    # 获取 nulls_fixture 作为 NA 值
    na_value = nulls_fixture
    # 创建一个包含字符串数据的数组对象
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    # 调用 to_numpy 方法将数组对象转换为 numpy 数组，指定 NA 值
    result = arr.to_numpy(na_value=na_value)
    # 创建一个期望的 numpy 数组
    expected = np.array(["a", na_value, "b"], dtype=object)
    # 断言两个 numpy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


# 定义一个测试函数，用于测试 isin 方法的情况
def test_isin(dtype, fixed_now_ts):
    # 创建一个包含字符串数据的 Series 对象
    s = pd.Series(["a", "b", None], dtype=dtype)

    # 测试 isin 方法是否能正确判断元素是否在指定列表中
    result = s.isin(["a", "c"])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.NA])
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)

    result = s.isin([])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", fixed_now_ts])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试设置标量值时的情况
def test_setitem_scalar_with_mask_validation(dtype):
    # https://github.com/pandas-dev/pandas/issues/47628
    # 通过布尔掩码设置 None 值，应该在底层数组中产生 pd.NA 值
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    mask = np.array([False, True, False])

    ser[mask] = None
    assert ser.array[1] is na_val(ser.dtype)

    # 对于其他非字符串类型，应该引发错误
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    if type(ser.array) is pd.arrays.StringArray:
        msg = "Cannot set non-string value"
    else:
        msg = "Scalar must be NA or str"
    with pytest.raises(TypeError, match=msg):
        ser[mask] = 1
# 使用给定的数据类型创建一个包含字符串数组的列表
def test_from_numpy_str(dtype):
    # 定义一个包含字符串 "a", "b", "c" 的列表
    vals = ["a", "b", "c"]
    # 使用 numpy 创建一个字符串类型的数组
    arr = np.array(vals, dtype=np.str_)
    # 使用 pandas 创建一个扩展数组，数据类型由参数 dtype 指定
    result = pd.array(arr, dtype=dtype)
    # 用预期结果创建一个 pandas 扩展数组
    expected = pd.array(vals, dtype=dtype)
    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)

# 将 pandas 扩展数组转换为普通 Python 列表，并与预期的列表进行比较
def test_tolist(dtype):
    # 定义一个包含字符串 "a", "b", "c" 的列表
    vals = ["a", "b", "c"]
    # 使用 pandas 创建一个扩展数组，数据类型由参数 dtype 指定
    arr = pd.array(vals, dtype=dtype)
    # 将 pandas 扩展数组转换为普通 Python 列表
    result = arr.tolist()
    # 定义预期的普通 Python 列表
    expected = vals
    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_equal(result, expected)
```