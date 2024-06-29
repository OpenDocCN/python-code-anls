# `D:\src\scipysrc\pandas\pandas\tests\arrays\numpy_\test_numpy.py`

```
"""
Additional tests for NumpyExtensionArray that aren't covered by
the interface tests.
"""

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray


@pytest.fixture(
    params=[
        np.array(["a", "b"], dtype=object),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=int),
        np.array([0, 1 + 2j], dtype=complex),
        np.array([True, False], dtype=bool),
        np.array([0, 1], dtype="datetime64[ns]"),
        np.array([0, 1], dtype="timedelta64[ns]"),
    ]
)
def any_numpy_array(request):
    """
    Parametrized fixture for NumPy arrays with different dtypes.

    This excludes string and bytes.
    """
    return request.param


# ----------------------------------------------------------------------------
# NumpyEADtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", True),
        ("uint", True),
        ("float", True),
        ("complex", True),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_numeric(dtype, expected):
    """
    Test case for checking if dtype is numeric.

    Uses NumpyEADtype to instantiate the dtype and asserts the _is_numeric property.
    """
    dtype = NumpyEADtype(dtype)
    assert dtype._is_numeric is expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", False),
        ("uint", False),
        ("float", False),
        ("complex", False),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_boolean(dtype, expected):
    """
    Test case for checking if dtype is boolean.

    Uses NumpyEADtype to instantiate the dtype and asserts the _is_boolean property.
    """
    dtype = NumpyEADtype(dtype)
    assert dtype._is_boolean is expected


def test_repr():
    """
    Test case for verifying the __repr__ method of NumpyEADtype.

    Instantiates NumpyEADtype with a specific numpy dtype and checks the repr output.
    """
    dtype = NumpyEADtype(np.dtype("int64"))
    assert repr(dtype) == "NumpyEADtype('int64')"


def test_constructor_from_string():
    """
    Test case for constructing NumpyEADtype from a string.

    Constructs NumpyEADtype using construct_from_string method and compares with expected result.
    """
    result = NumpyEADtype.construct_from_string("int64")
    expected = NumpyEADtype(np.dtype("int64"))
    assert result == expected


def test_dtype_idempotent(any_numpy_dtype):
    """
    Test case for verifying idempotency of NumpyEADtype constructor.

    Constructs NumpyEADtype with any numpy dtype and checks if the constructor is idempotent.
    """
    dtype = NumpyEADtype(any_numpy_dtype)

    result = NumpyEADtype(dtype)
    assert result == dtype


# ----------------------------------------------------------------------------
# Construction


def test_constructor_no_coercion():
    """
    Test case for ensuring no coercion in NumpyExtensionArray constructor.

    Raises ValueError if coercion is attempted in the constructor.
    """
    with pytest.raises(ValueError, match="NumPy array"):
        NumpyExtensionArray([1, 2, 3])


def test_series_constructor_with_copy():
    """
    Test case for constructing Series with NumpyExtensionArray with copy=True.

    Verifies that Series does not share memory with the original ndarray.
    """
    ndarray = np.array([1, 2, 3])
    ser = pd.Series(NumpyExtensionArray(ndarray), copy=True)

    assert ser.values is not ndarray


def test_series_constructor_with_astype():
    """
    Test case for constructing Series with NumpyExtensionArray and dtype conversion.

    Checks if Series values match the expected values after dtype conversion.
    """
    ndarray = np.array([1, 2, 3])
    result = pd.Series(NumpyExtensionArray(ndarray), dtype="float64")
    expected = pd.Series([1.0, 2.0, 3.0], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_from_sequence_dtype():
    """
    Test case for creating NumpyExtensionArray from a sequence with dtype.

    Uses np.array with dtype specified and does not perform any additional checks.
    """
    arr = np.array([1, 2, 3], dtype="int64")
    # 创建一个 NumpyExtensionArray 对象，使用给定的数组 `arr` 和数据类型 `uint64`
    result = NumpyExtensionArray._from_sequence(arr, dtype="uint64")
    
    # 创建一个预期的 NumpyExtensionArray 对象，使用给定的 NumPy 数组 [1, 2, 3] 和数据类型 `uint64`
    expected = NumpyExtensionArray(np.array([1, 2, 3], dtype="uint64"))
    
    # 使用测试框架中的函数 `assert_extension_array_equal` 检查 `result` 和 `expected` 是否相等
    tm.assert_extension_array_equal(result, expected)
# 测试构造函数，使用给定数组创建 NumpyExtensionArray 对象，确保对象是复制的
def test_constructor_copy():
    arr = np.array([0, 1])
    result = NumpyExtensionArray(arr, copy=True)

    # 断言确保 result 和 arr 不共享内存
    assert not tm.shares_memory(result, arr)


# 测试构造函数，使用任意的 NumPy 数组创建 NumpyExtensionArray 对象，并验证 dtype 是否匹配
def test_constructor_with_data(any_numpy_array):
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr)
    
    # 断言验证 NumpyExtensionArray 对象的 dtype 是否与原始数组的 dtype 相同
    assert arr.dtype.numpy_dtype == nparr.dtype


# ----------------------------------------------------------------------------
# 转换操作


# 测试将 NumpyExtensionArray 转换为原始的 NumPy 数组
def test_to_numpy():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    result = arr.to_numpy()
    
    # 断言验证转换结果与原始数组相同
    assert result is arr._ndarray

    # 测试带有复制选项的转换
    result = arr.to_numpy(copy=True)
    assert result is not arr._ndarray

    # 测试指定 dtype 的转换
    result = arr.to_numpy(dtype="f8")
    expected = np.array([1, 2, 3], dtype="f8")
    tm.assert_numpy_array_equal(result, expected)


# ----------------------------------------------------------------------------
# 赋值操作


# 测试修改 Series 的 NumpyExtensionArray 中的元素
def test_setitem_series():
    ser = pd.Series([1, 2, 3])
    ser.array[0] = 10
    expected = pd.Series([10, 2, 3])
    
    # 断言验证修改后的 Series 是否符合预期
    tm.assert_series_equal(ser, expected)


# 测试修改 NumpyExtensionArray 中的元素
def test_setitem(any_numpy_array):
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr, copy=True)

    # 修改 NumpyExtensionArray 的第一个元素为第二个元素的值
    arr[0] = arr[1]
    nparr[0] = nparr[1]

    # 断言验证修改后的 NumpyExtensionArray 是否与原始 NumPy 数组相同
    tm.assert_numpy_array_equal(arr.to_numpy(), nparr)


# ----------------------------------------------------------------------------
# 归约操作


# 测试不支持的归约操作是否会引发 TypeError 异常
def test_bad_reduce_raises():
    arr = np.array([1, 2, 3], dtype="int64")
    arr = NumpyExtensionArray(arr)
    msg = "cannot perform not_a_method with type int"
    
    # 使用 pytest 断言检查是否引发了预期的 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        arr._reduce(msg)


# 测试归约操作的关键字参数是否合法
def test_validate_reduction_keyword_args():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    msg = "the 'keepdims' parameter is not supported .*all"
    
    # 使用 pytest 断言检查是否引发了预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        arr.all(keepdims=True)


# 测试 np.maximum.reduce 在嵌套元组的情况下的行为
def test_np_max_nested_tuples():
    vals = [
        (("j", "k"), ("l", "m")),
        (("l", "m"), ("o", "p")),
        (("o", "p"), ("j", "k")),
    ]
    ser = pd.Series(vals)
    arr = ser.array

    # 断言验证最大值归约操作的正确性
    assert arr.max() is arr[2]
    assert ser.max() is arr[2]

    # 使用 np.maximum.reduce 进行归约操作，并验证结果是否符合预期
    result = np.maximum.reduce(arr)
    assert result == arr[2]

    # 使用 np.maximum.reduce 对 Series 进行归约操作，并验证结果是否符合预期
    result = np.maximum.reduce(ser)
    assert result == arr[2]


# 测试二维数组的归约操作
def test_np_reduce_2d():
    raw = np.arange(12).reshape(4, 3)
    arr = NumpyExtensionArray(raw)

    # 在 axis=0 上进行最大值归约操作，并验证结果是否符合预期
    res = np.maximum.reduce(arr, axis=0)
    tm.assert_extension_array_equal(res, arr[-1])

    # 使用 NumpyExtensionArray 对象的 max 方法进行归约操作，并验证结果是否符合预期
    alt = arr.max(axis=0)
    tm.assert_extension_array_equal(alt, arr[-1])


# ----------------------------------------------------------------------------
# 运算操作


# 使用 pytest 参数化装饰器测试一元通用函数的行为
@pytest.mark.parametrize("ufunc", [np.abs, np.negative, np.positive])
def test_ufunc_unary(ufunc):
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))
    result = ufunc(arr)
    expected = NumpyExtensionArray(ufunc(arr._ndarray))
    
    # 断言验证一元通用函数的应用结果是否符合预期
    tm.assert_extension_array_equal(result, expected)

    # 使用 'out' 关键字参数测试一元通用函数的另一种用法
    # 创建一个 NumpyExtensionArray 对象，用给定的 numpy 数组 [-9.0, -9.0, -9.0] 初始化
    out = NumpyExtensionArray(np.array([-9.0, -9.0, -9.0]))
    # 调用 ufunc 函数，对输入的 arr 应用该函数，并将输出指定为之前创建的 out 对象
    ufunc(arr, out=out)
    # 使用 tm.assert_extension_array_equal 函数来断言 out 对象与预期结果 expected 相等
    tm.assert_extension_array_equal(out, expected)
def test_ufunc():
    # 创建 NumpyExtensionArray 对象，传入包含浮点数的 NumPy 数组
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))

    # 对数组进行 np.divmod 和 np.add 操作
    r1, r2 = np.divmod(arr, np.add(arr, 2))

    # 从 NumpyExtensionArray 中提取 _ndarray 属性，再进行相同的 np.divmod 和 np.add 操作
    e1, e2 = np.divmod(arr._ndarray, np.add(arr._ndarray, 2))

    # 将结果转换为 NumpyExtensionArray 对象
    e1 = NumpyExtensionArray(e1)
    e2 = NumpyExtensionArray(e2)

    # 使用测试框架的函数验证 r1 和 e1，r2 和 e2 是否相等
    tm.assert_extension_array_equal(r1, e1)
    tm.assert_extension_array_equal(r2, e2)


def test_basic_binop():
    # 基本的测试用例，验证 NumpyExtensionArray 对象的加法操作
    x = NumpyExtensionArray(np.array([1, 2, 3]))
    result = x + x
    expected = NumpyExtensionArray(np.array([2, 4, 6]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, object])
def test_setitem_object_typecode(dtype):
    # 使用不同的数据类型创建 NumpyExtensionArray 对象，并修改第一个元素
    arr = NumpyExtensionArray(np.array(["a", "b", "c"], dtype=dtype))
    arr[0] = "t"
    expected = NumpyExtensionArray(np.array(["t", "b", "c"], dtype=dtype))

    # 验证修改后的数组是否符合预期
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_no_coercion():
    # 验证不同类型的赋值操作是否会引发 ValueError 异常
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="int"):
        arr[0] = "a"

    # 验证赋值浮点数是否会强制转换为整数类型
    arr[0] = 2.5
    assert isinstance(arr[0], (int, np.integer)), type(arr[0])


def test_setitem_preserves_views():
    # 验证赋值操作对视图的影响
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    view1 = arr.view()
    view2 = arr[:]
    view3 = np.asarray(arr)

    # 修改数组的第一个元素，并验证视图的响应
    arr[0] = 9
    assert view1[0] == 9
    assert view2[0] == 9
    assert view3[0] == 9

    # 修改数组的最后一个元素，并验证视图的响应
    arr[-1] = 2.5
    view1[-1] = 5
    assert arr[-1] == 5


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_quantile_empty(dtype):
    # 验证空数组的分位数计算，应返回 np.nan 而不是 -1
    arr = NumpyExtensionArray(np.array([], dtype=dtype))
    idx = pd.Index([0.0, 0.5])

    # 使用线性插值计算分位数，并与预期结果进行比较
    result = arr._quantile(idx, interpolation="linear")
    expected = NumpyExtensionArray(np.array([np.nan, np.nan]))
    tm.assert_extension_array_equal(result, expected)


def test_factorize_unsigned():
    # 验证无符号整数类型的 factorize 方法调用
    arr = np.array([1, 2, 3], dtype=np.uint64)
    obj = NumpyExtensionArray(arr)

    # 调用 factorize 方法并与 pandas 的 factorize 进行比较
    res_codes, res_unique = obj.factorize()
    exp_codes, exp_unique = pd.factorize(arr)

    tm.assert_numpy_array_equal(res_codes, exp_codes)
    tm.assert_extension_array_equal(res_unique, NumpyExtensionArray(exp_unique))
```