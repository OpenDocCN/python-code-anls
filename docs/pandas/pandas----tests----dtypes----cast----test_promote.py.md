# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_promote.py`

```
"""
These test the method maybe_promote from core/dtypes/cast.py
"""

# 导入必要的模块和库
import datetime  # 导入日期时间模块
from decimal import Decimal  # 导入 Decimal 类

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import NaT  # 导入 NaT 对象

# 导入 pandas 相关模块和函数
from pandas.core.dtypes.cast import maybe_promote  # 导入 maybe_promote 函数
from pandas.core.dtypes.common import is_scalar  # 导入 is_scalar 函数
from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 导入 DatetimeTZDtype 类型
from pandas.core.dtypes.missing import isna  # 导入 isna 函数

import pandas as pd  # 导入 pandas 库


def _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar=None):
    """
    Auxiliary function to unify testing of scalar/array promotion.

    Parameters
    ----------
    dtype : dtype
        The value to pass on as the first argument to maybe_promote.
    fill_value : scalar
        The value to pass on as the second argument to maybe_promote as
        a scalar.
    expected_dtype : dtype
        The expected dtype returned by maybe_promote (by design this is the
        same regardless of whether fill_value was passed as a scalar or in an
        array!).
    exp_val_for_scalar : scalar
        The expected value for the (potentially upcast) fill_value returned by
        maybe_promote.
    """
    assert is_scalar(fill_value)  # 断言 fill_value 是标量

    # 调用 maybe_promote 函数，将 fill_value 作为标量传递，期望返回填充值和数据类型
    result_dtype, result_fill_value = maybe_promote(dtype, fill_value)
    expected_fill_value = exp_val_for_scalar

    # 断言返回的数据类型与期望的数据类型相同
    assert result_dtype == expected_dtype
    _assert_match(result_fill_value, expected_fill_value)


def _assert_match(result_fill_value, expected_fill_value):
    # GH#23982/25425 require the same type in addition to equality/NA-ness
    res_type = type(result_fill_value)
    ex_type = type(expected_fill_value)

    if hasattr(result_fill_value, "dtype"):
        # Compare types in a way that is robust to platform-specific
        #  idiosyncrasies where e.g. sometimes we get "ulonglong" as an alias
        #  for "uint64" or "intc" as an alias for "int32"
        assert result_fill_value.dtype.kind == expected_fill_value.dtype.kind
        assert result_fill_value.dtype.itemsize == expected_fill_value.dtype.itemsize
    else:
        # On some builds, type comparison fails, e.g. np.int32 != np.int32
        assert res_type == ex_type or res_type.__name__ == ex_type.__name__

    match_value = result_fill_value == expected_fill_value
    if match_value is pd.NA:
        match_value = False

    # Note: type check above ensures that we have the _same_ NA value
    # for missing values, None == None (which is checked
    # through match_value above), but np.nan != np.nan and pd.NaT != pd.NaT
    match_missing = isna(result_fill_value) and isna(expected_fill_value)

    assert match_value or match_missing


@pytest.mark.parametrize(
    "dtype, fill_value, expected_dtype",
    ],
)
def test_maybe_promote_int_with_int(dtype, fill_value, expected_dtype):
    dtype = np.dtype(dtype)  # 将 dtype 转换为 NumPy 的数据类型
    `
        # 将期望的数据类型转换为 NumPy 的数据类型对象
        expected_dtype = np.dtype(expected_dtype)
    
        # 将填充值转换为期望的数据类型，并取出其标量值
        exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]
    
        # 调用 _check_promote 函数，进行数据类型和填充值的检查
        _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)
def test_maybe_promote_int_with_float(any_int_numpy_dtype, float_numpy_dtype):
    # 根据输入的任意整数类型和浮点数类型创建相应的数据类型对象
    dtype = np.dtype(any_int_numpy_dtype)
    fill_dtype = np.dtype(float_numpy_dtype)

    # 创建一个包含单个元素 "1" 的数组，并将其转换为指定的填充数据类型
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 预期的数据类型将始终是 np.float64，因为填充整数值使用浮点数时会自动提升为 float64
    expected_dtype = np.float64
    # exp_val_for_scalar 是填充值 fill_value 的 np.float64 类型的标量表示
    exp_val_for_scalar = np.float64(fill_value)

    # 调用内部函数 _check_promote，检查类型提升是否符合预期
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_float_with_int(float_numpy_dtype, any_int_numpy_dtype):
    # 根据输入的浮点数类型和任意整数类型创建相应的数据类型对象
    dtype = np.dtype(float_numpy_dtype)
    fill_dtype = np.dtype(any_int_numpy_dtype)

    # 创建一个包含单个元素 "1" 的数组，并将其转换为指定的填充数据类型
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 填充浮点数值时，数据类型预期将保持不变，因为 np.finfo('float32').max > np.iinfo('uint64').max
    expected_dtype = dtype
    # exp_val_for_scalar 是填充值 fill_value 的与 expected_dtype 相对应的标量表示
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    # 调用内部函数 _check_promote，检查类型提升是否符合预期
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "dtype, fill_value, expected_dtype",
    [
        # float 填充 float
        ("float32", 1, "float32"),
        ("float32", float(np.finfo("float32").max) * 1.1, "float64"),
        ("float64", 1, "float64"),
        ("float64", float(np.finfo("float32").max) * 1.1, "float64"),
        # complex 填充 float
        ("complex64", 1, "complex64"),
        ("complex64", float(np.finfo("float32").max) * 1.1, "complex128"),
        ("complex128", 1, "complex128"),
        ("complex128", float(np.finfo("float32").max) * 1.1, "complex128"),
        # float 填充 complex
        ("float32", 1 + 1j, "complex64"),
        ("float32", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        ("float64", 1 + 1j, "complex128"),
        ("float64", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        # complex 填充 complex
        ("complex64", 1 + 1j, "complex64"),
        ("complex64", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
        ("complex128", 1 + 1j, "complex128"),
        ("complex128", float(np.finfo("float32").max) * (1.1 + 1j), "complex128"),
    ],
)
def test_maybe_promote_float_with_float(dtype, fill_value, expected_dtype):
    # 根据输入的数据类型创建相应的 dtype 对象
    dtype = np.dtype(dtype)
    expected_dtype = np.dtype(expected_dtype)

    # exp_val_for_scalar 是填充值 fill_value 的与 expected_dtype 相对应的标量表示
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    # 调用内部函数 _check_promote，检查类型提升是否符合预期
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_bool_with_any(any_numpy_dtype):
    # 创建布尔类型的 dtype 对象
    dtype = np.dtype(bool)
    fill_dtype = np.dtype(any_numpy_dtype)

    # 创建一个包含单个元素 "1" 的数组，并将其转换为指定的填充数据类型
    fill_value = np.array([1], dtype=fill_dtype)[0]
    # 根据填充数据类型是否为布尔类型，选择相应的期望数据类型
    expected_dtype = np.dtype(object) if fill_dtype != bool else fill_dtype
    # 设置用于标量的期望值
    exp_val_for_scalar = fill_value
    
    # 调用检查函数，验证数据类型的一致性和推广
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)
def test_maybe_promote_any_with_bool(any_numpy_dtype):
    # 将输入参数转换为NumPy的dtype对象
    dtype = np.dtype(any_numpy_dtype)
    # 设定填充值为布尔类型True
    fill_value = True

    # 如果dtype不是布尔类型，则期望的dtype为对象类型，否则为布尔类型本身
    expected_dtype = np.dtype(object) if dtype != bool else dtype
    # 创建一个包含填充值的NumPy数组，类型为expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    # 调用内部函数_check_promote，验证推断类型的结果
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_bytes_with_any(bytes_dtype, any_numpy_dtype):
    # 将输入参数转换为NumPy的dtype对象
    dtype = np.dtype(bytes_dtype)
    # 将输入参数转换为NumPy的dtype对象
    fill_dtype = np.dtype(any_numpy_dtype)

    # 创建一个NumPy数组，用给定的dtype将数字1转换为相应的数据类型
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 我们从不在内部使用bytes类型，始终将其推广为对象类型
    expected_dtype = np.dtype(np.object_)
    exp_val_for_scalar = fill_value

    # 调用内部函数_check_promote，验证推断类型的结果
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_bytes(any_numpy_dtype):
    # 将输入参数转换为NumPy的dtype对象
    dtype = np.dtype(any_numpy_dtype)

    # 创建一个具有给定dtype的NumPy数组
    fill_value = b"abc"

    # 我们从不在内部使用bytes类型，始终将其推广为对象类型
    expected_dtype = np.dtype(np.object_)
    # 创建一个包含填充值的NumPy数组，类型为expected_dtype
    exp_val_for_scalar = np.array([fill_value], dtype=expected_dtype)[0]

    # 调用内部函数_check_promote，验证推断类型的结果
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_datetime64_with_any(datetime64_dtype, any_numpy_dtype):
    # 将输入参数转换为NumPy的dtype对象
    dtype = np.dtype(datetime64_dtype)
    # 将输入参数转换为NumPy的dtype对象
    fill_dtype = np.dtype(any_numpy_dtype)

    # 创建一个NumPy数组，用给定的dtype将数字1转换为相应的数据类型
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 如果填充值的dtype的kind为"M"（datetime64），则期望的dtype为dtype本身
    if fill_dtype.kind == "M":
        expected_dtype = dtype
        # 对于datetime64类型，标量值将被转换为pd.Timestamp.to_datetime64
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        # 否则期望的dtype为对象类型
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    # 调用内部函数_check_promote，验证推断类型的结果
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp("now"),
        np.datetime64("now"),
        datetime.datetime.now(),
        datetime.date.today(),
    ],
    ids=["pd.Timestamp", "np.datetime64", "datetime.datetime", "datetime.date"],
)
def test_maybe_promote_any_with_datetime64(any_numpy_dtype, fill_value):
    # 将输入参数转换为NumPy的dtype对象
    dtype = np.dtype(any_numpy_dtype)

    # 如果dtype的kind为"M"（datetime64），则期望的dtype为dtype本身
    if dtype.kind == "M":
        expected_dtype = dtype
        # 对于datetime64类型，标量值将被转换为pd.Timestamp.to_datetime64
        exp_val_for_scalar = pd.Timestamp(fill_value).to_datetime64()
    else:
        # 否则期望的dtype为对象类型
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value
    # 如果 fill_value 的类型是 datetime.date，并且 dtype 的类型是 "M"（日期时间类型）
    if type(fill_value) is datetime.date and dtype.kind == "M":
        # 将日期转换为 dt64 已被弃用，在 2.0 版本中将强制转换为 object 类型
        expected_dtype = np.dtype(object)
        exp_val_for_scalar = fill_value

    # 调用 _check_promote 函数，检查并推断数据类型的提升
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)
@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp(2023, 1, 1),  # 使用pd.Timestamp创建一个时间戳作为填充值
        np.datetime64("2023-01-01"),  # 使用np.datetime64创建一个日期时间作为填充值
        datetime.datetime(2023, 1, 1),  # 使用datetime.datetime创建一个日期时间作为填充值
        datetime.date(2023, 1, 1),  # 使用datetime.date创建一个日期作为填充值
    ],
    ids=["pd.Timestamp", "np.datetime64", "datetime.datetime", "datetime.date"],  # 对参数化测试用例进行标识
)
def test_maybe_promote_any_numpy_dtype_with_datetimetz(
    any_numpy_dtype, tz_aware_fixture, fill_value
):
    dtype = np.dtype(any_numpy_dtype)  # 获取给定numpy数据类型的dtype对象
    fill_dtype = DatetimeTZDtype(tz=tz_aware_fixture)  # 创建一个具有时区信息的DatetimeTZDtype对象作为填充dtype

    fill_value = pd.Series([fill_value], dtype=fill_dtype)[0]  # 将填充值转换为填充dtype的Series并获取第一个值

    # 填充任何numpy数据类型与datetimetz时将其强制转换为object类型
    expected_dtype = np.dtype(object)  # 预期的结果dtype为object类型
    exp_val_for_scalar = fill_value  # 标量值的预期填充值

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_timedelta64_with_any(timedelta64_dtype, any_numpy_dtype):
    dtype = np.dtype(timedelta64_dtype)  # 获取给定timedelta64数据类型的dtype对象
    fill_dtype = np.dtype(any_numpy_dtype)  # 获取任意给定numpy数据类型的dtype对象

    # 创建给定dtype的数组；将"1"强制转换为正确的dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 填充timedelta64数据类型与任何其他类型时将其强制转换为object类型
    if fill_dtype.kind == "m":  # 如果填充dtype的kind为"m"，表示是timedelta64类型
        expected_dtype = dtype  # 预期结果dtype与原dtype相同
        # 对于timedelta数据类型，标量值将被转换为pd.Timedelta.value
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)  # 预期结果dtype为object类型
        exp_val_for_scalar = fill_value  # 标量值的预期填充值

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


@pytest.mark.parametrize(
    "fill_value",
    [pd.Timedelta(days=1), np.timedelta64(24, "h"), datetime.timedelta(1)],
    ids=["pd.Timedelta", "np.timedelta64", "datetime.timedelta"],  # 对参数化测试用例进行标识
)
def test_maybe_promote_any_with_timedelta64(any_numpy_dtype, fill_value):
    dtype = np.dtype(any_numpy_dtype)  # 获取给定numpy数据类型的dtype对象

    # 填充任何类型与timedelta64时将其强制转换为object类型
    if dtype.kind == "m":  # 如果填充dtype的kind为"m"，表示是timedelta64类型
        expected_dtype = dtype  # 预期结果dtype与原dtype相同
        # 对于timedelta数据类型，标量值将被转换为pd.Timedelta.value
        exp_val_for_scalar = pd.Timedelta(fill_value).to_timedelta64()
    else:
        expected_dtype = np.dtype(object)  # 预期结果dtype为object类型
        exp_val_for_scalar = fill_value  # 标量值的预期填充值

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_string_with_any(string_dtype, any_numpy_dtype):
    dtype = np.dtype(string_dtype)  # 获取给定字符串数据类型的dtype对象
    fill_dtype = np.dtype(any_numpy_dtype)  # 获取任意给定numpy数据类型的dtype对象

    # 创建给定dtype的数组；将"1"强制转换为正确的dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 填充字符串数据类型与任何其他类型时将其强制转换为object类型
    expected_dtype = np.dtype(object)  # 预期结果dtype为object类型
    exp_val_for_scalar = fill_value  # 标量值的预期填充值

    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_string(any_numpy_dtype):
    dtype = np.dtype(any_numpy_dtype)  # 获取给定numpy数据类型的dtype对象

    # 创建给定dtype的数组
    fill_value = "abc"  # 填充值为字符串"abc"

    # 填充任何类型与字符串时将其强制转换为object类型
    expected_dtype = np.dtype(object)  # 预期结果dtype为object类型
    exp_val_for_scalar = fill_value  # 标量值的预期填充值
    # 调用名为 _check_promote 的函数，并传入参数 dtype, fill_value, expected_dtype, exp_val_for_scalar
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)
def test_maybe_promote_object_with_any(object_dtype, any_numpy_dtype):
    # 将输入的 object_dtype 转换为 numpy 的数据类型对象
    dtype = np.dtype(object_dtype)
    # 将输入的 any_numpy_dtype 转换为 numpy 的数据类型对象
    fill_dtype = np.dtype(any_numpy_dtype)

    # 创建一个数组，以给定的 fill_dtype 类型存储值 1，并将其转换为正确的 dtype
    fill_value = np.array([1], dtype=fill_dtype)[0]

    # 填充对象为任何值时，保持对象的数据类型不变
    expected_dtype = np.dtype(object)
    # 用 fill_value 初始化标量表达式的预期值
    exp_val_for_scalar = fill_value

    # 调用 _check_promote 函数，验证数据类型的提升和填充值的预期行为
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_with_object(any_numpy_dtype):
    # 将输入的 any_numpy_dtype 转换为 numpy 的数据类型对象
    dtype = np.dtype(any_numpy_dtype)

    # 创建一个对象数据类型的数组，使用标量值填充 (例如通过 dtypes.common.is_scalar)，无法转换为 int/float 等
    fill_value = pd.DateOffset(1)

    # 填充对象为任何值时，保持对象的数据类型不变
    expected_dtype = np.dtype(object)
    # 用 fill_value 初始化标量表达式的预期值
    exp_val_for_scalar = fill_value

    # 调用 _check_promote 函数，验证数据类型的提升和填充值的预期行为
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def test_maybe_promote_any_numpy_dtype_with_na(any_numpy_dtype, nulls_fixture):
    # 获取填充值 nulls_fixture
    fill_value = nulls_fixture
    # 将输入的 any_numpy_dtype 转换为 numpy 的数据类型对象
    dtype = np.dtype(any_numpy_dtype)

    if isinstance(fill_value, Decimal):
        # 如果填充值为 Decimal 类型
        # 注意：此处的行为可能会更改，当前依赖于 is_valid_na_for_dtype
        if dtype.kind in "iufc":
            if dtype.kind in "iu":
                # 对于整数类型，将其转换为 np.float64
                expected_dtype = np.dtype(np.float64)
            else:
                # 对于其他数值类型，保持不变
                expected_dtype = dtype
            # 对于标量表达式，使用 np.nan 作为预期值
            exp_val_for_scalar = np.nan
        else:
            # 对于其他类型，保持对象数据类型不变
            expected_dtype = np.dtype(object)
            # 用 fill_value 初始化标量表达式的预期值
            exp_val_for_scalar = fill_value
    elif dtype.kind in "iu" and fill_value is not NaT:
        # 如果数据类型是整数，且填充值不是 NaT
        # 整数与其他缺失值（如 np.nan / None）将转换为浮点数
        expected_dtype = np.float64
        # 对于标量表达式，使用 np.nan 作为预期值
        exp_val_for_scalar = np.nan
    elif dtype == object and fill_value is NaT:
        # 如果插入对象不转换值
        # 但 * 会 * 将 None 转换为 np.nan
        expected_dtype = np.dtype(object)
        # 用 fill_value 初始化标量表达式的预期值
        exp_val_for_scalar = fill_value
    elif dtype.kind in "mM":
        # datetime / timedelta 将所有缺失值转换为 dtyped-NaT
        expected_dtype = dtype
        # 对于标量表达式，使用 dtype 类型的 "NaT" 值作为预期值
        exp_val_for_scalar = dtype.type("NaT", "ns")
    elif fill_value is NaT:
        # NaT 将除 datetime/timedelta 外的所有内容向上转换为对象
        expected_dtype = np.dtype(object)
        # 对于标量表达式，使用 NaT 作为预期值
        exp_val_for_scalar = NaT
    elif dtype.kind in "fc":
        # 浮点数 / 复数 + 缺失值（！= NaT）保持不变
        expected_dtype = dtype
        # 对于标量表达式，使用 np.nan 作为预期值
        exp_val_for_scalar = np.nan
    else:
        # 所有其他情况转换为对象，并使用 np.nan 作为缺失值
        expected_dtype = np.dtype(object)
        if fill_value is pd.NA:
            exp_val_for_scalar = pd.NA
        else:
            exp_val_for_scalar = np.nan

    # 调用 _check_promote 函数，验证数据类型的提升和填充值的预期行为
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)
```