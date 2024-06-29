# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_dtype.py`

```
import re  # 导入正则表达式模块
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 Pandas 库
from pandas import SparseDtype  # 从 Pandas 中导入 SparseDtype 类


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用来定义参数化测试
    "dtype, fill_value",  # 参数列表
    [  # 参数化的测试数据列表
        ("int", 0),  # 整型 dtype 和填充值 0
        ("float", np.nan),  # 浮点型 dtype 和填充值 NaN
        ("bool", False),  # 布尔型 dtype 和填充值 False
        ("object", np.nan),  # 对象型 dtype 和填充值 NaN
        ("datetime64[ns]", np.datetime64("NaT", "ns")),  # 日期时间型 dtype 和填充值 NaT
        ("timedelta64[ns]", np.timedelta64("NaT", "ns")),  # 时间间隔型 dtype 和填充值 NaT
    ],
)
def test_inferred_dtype(dtype, fill_value):
    sparse_dtype = SparseDtype(dtype)  # 创建 SparseDtype 对象
    result = sparse_dtype.fill_value  # 获取填充值
    if pd.isna(fill_value):  # 如果填充值是 NaN
        assert pd.isna(result) and type(result) == type(fill_value)  # 断言结果是 NaN 且类型与填充值相同
    else:
        assert result == fill_value  # 否则断言结果等于填充值


def test_from_sparse_dtype():
    dtype = SparseDtype("float", 0)  # 创建 float 类型的 SparseDtype 对象
    result = SparseDtype(dtype)  # 创建新的 SparseDtype 对象
    assert result.fill_value == 0  # 断言填充值为 0


def test_from_sparse_dtype_fill_value():
    dtype = SparseDtype("int", 1)  # 创建 int 类型的 SparseDtype 对象
    result = SparseDtype(dtype, fill_value=2)  # 使用新的填充值创建 SparseDtype 对象
    expected = SparseDtype("int", 2)  # 期望的 SparseDtype 对象
    assert result == expected  # 断言结果与期望相等


@pytest.mark.parametrize(  # 参数化测试
    "dtype, fill_value",  # 参数列表
    [  # 参数化的测试数据列表
        ("int", None),  # 整型 dtype 和空填充值
        ("float", None),  # 浮点型 dtype 和空填充值
        ("bool", None),  # 布尔型 dtype 和空填充值
        ("object", None),  # 对象型 dtype 和空填充值
        ("datetime64[ns]", None),  # 日期时间型 dtype 和空填充值
        ("timedelta64[ns]", None),  # 时间间隔型 dtype 和空填充值
        ("int", np.nan),  # 整型 dtype 和填充值 NaN
        ("float", 0),  # 浮点型 dtype 和填充值 0
    ],
)
def test_equal(dtype, fill_value):
    a = SparseDtype(dtype, fill_value)  # 创建 SparseDtype 对象 a
    b = SparseDtype(dtype, fill_value)  # 创建 SparseDtype 对象 b
    assert a == b  # 断言 a 等于 b
    assert b == a  # 断言 b 等于 a


def test_nans_equal():
    a = SparseDtype(float, float("nan"))  # 创建浮点型 SparseDtype 对象 a，使用 NaN 作为填充值
    b = SparseDtype(float, np.nan)  # 创建浮点型 SparseDtype 对象 b，使用 NaN 作为填充值
    assert a == b  # 断言 a 等于 b
    assert b == a  # 断言 b 等于 a


def test_nans_not_equal():
    # GH 54770
    a = SparseDtype(float, 0)  # 创建浮点型 SparseDtype 对象 a，使用 0 作为填充值
    b = SparseDtype(float, pd.NA)  # 创建浮点型 SparseDtype 对象 b，使用 Pandas 的 NA 作为填充值
    assert a != b  # 断言 a 不等于 b
    assert b != a  # 断言 b 不等于 a


with warnings.catch_warnings():  # 捕获警告
    msg = "Allowing arbitrary scalar fill_value in SparseDtype is deprecated"  # 警告信息
    warnings.filterwarnings("ignore", msg, category=FutureWarning)  # 忽略特定类型的 FutureWarning 警告

    tups = [  # 元组列表
        (SparseDtype("float64"), SparseDtype("float32")),  # 创建两个不同的 SparseDtype 对象
        (SparseDtype("float64"), SparseDtype("float64", 0)),  # 创建两个不同的 SparseDtype 对象
        (SparseDtype("float64"), SparseDtype("datetime64[ns]", np.nan)),  # 创建两个不同的 SparseDtype 对象
        (SparseDtype("float64"), np.dtype("float64")),  # 创建一个 SparseDtype 对象和一个 NumPy dtype 对象
    ]


@pytest.mark.parametrize(  # 参数化测试
    "a, b",  # 参数列表
    tups,  # 参数化的测试数据列表
)
def test_not_equal(a, b):
    assert a != b  # 断言 a 不等于 b


def test_construct_from_string_raises():
    with pytest.raises(  # 断言会抛出异常
        TypeError, match="Cannot construct a 'SparseDtype' from 'not a dtype'"  # 异常类型和匹配的异常信息
    ):
        SparseDtype.construct_from_string("not a dtype")  # 调用构造函数，传入非 dtype 字符串


@pytest.mark.parametrize(  # 参数化测试
    "dtype, expected",  # 参数列表
    [  # 参数化的测试数据列表
        (int, True),  # 整型和预期的结果 True
        (float, True),  # 浮点型和预期的结果 True
        (bool, True),  # 布尔型和预期的结果 True
        (object, False),  # 对象型和预期的结果 False
        (str, False),  # 字符串型和预期的结果 False
    ],
)
def test_is_numeric(dtype, expected):
    assert SparseDtype(dtype)._is_numeric is expected  # 断言 SparseDtype 对象的 _is_numeric 属性与预期结果相等


def test_str_uses_object():
    result = SparseDtype(str).subtype  # 创建字符串类型的 SparseDtype 对象，获取其 subtype 属性
    assert result == np.dtype("object")  # 断言 subtype 属性等于 NumPy 的对象类型


@pytest.mark.parametrize(  # 参数化测试
    "string, expected",  # 参数列表
    expected,  # 参数化的测试数据列表（在下一行继续）
)
    [
        # 定义稀疏数据类型，使用 float64 的底层数据类型
        ("Sparse[float64]", SparseDtype(np.dtype("float64"))),
        # 定义稀疏数据类型，使用 float32 的底层数据类型
        ("Sparse[float32]", SparseDtype(np.dtype("float32"))),
        # 定义稀疏数据类型，使用 int 的底层数据类型
        ("Sparse[int]", SparseDtype(np.dtype("int"))),
        # 定义稀疏数据类型，使用 str 的底层数据类型
        ("Sparse[str]", SparseDtype(np.dtype("str"))),
        # 定义稀疏数据类型，使用 datetime64[ns] 的底层数据类型
        ("Sparse[datetime64[ns]]", SparseDtype(np.dtype("datetime64[ns]"))),
        # 定义稀疏数据类型，使用 float 的底层数据类型，缺失值为 NaN
        ("Sparse", SparseDtype(np.dtype("float"), np.nan)),
    ],
)
def test_construct_from_string(string, expected):
    result = SparseDtype.construct_from_string(string)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (SparseDtype(float, 0.0), SparseDtype(np.dtype("float"), 0.0), True),
        (SparseDtype(int, 0), SparseDtype(int, 0), True),
        (SparseDtype(float, float("nan")), SparseDtype(float, np.nan), True),
        (SparseDtype(float, 0), SparseDtype(float, np.nan), False),
        (SparseDtype(int, 0.0), SparseDtype(float, 0.0), False),
    ],
)
def test_hash_equal(a, b, expected):
    result = a == b
    assert result is expected

    result = hash(a) == hash(b)
    assert result is expected


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[int]", "int"),
        ("Sparse[int, 0]", "int"),
        ("Sparse[int64]", "int64"),
        ("Sparse[int64, 0]", "int64"),
        ("Sparse[datetime64[ns], 0]", "datetime64[ns]"),
    ],
)
def test_parse_subtype(string, expected):
    # 解析给定字符串以获取稀疏数据类型的子类型
    subtype, _ = SparseDtype._parse_subtype(string)
    assert subtype == expected


@pytest.mark.parametrize(
    "string", ["Sparse[int, 1]", "Sparse[float, 0.0]", "Sparse[bool, True]"]
)
def test_construct_from_string_fill_value_raises(string):
    # 确保从字符串构造稀疏数据类型时，填充值引发类型错误异常
    with pytest.raises(TypeError, match="fill_value in the string is not"):
        SparseDtype.construct_from_string(string)


@pytest.mark.parametrize(
    "original, dtype, expected",
    [
        (SparseDtype(int, 0), float, SparseDtype(float, 0.0)),
        (SparseDtype(int, 1), float, SparseDtype(float, 1.0)),
        (SparseDtype(int, 1), str, SparseDtype(object, "1")),
        (SparseDtype(float, 1.5), int, SparseDtype(int, 1)),
    ],
)
def test_update_dtype(original, dtype, expected):
    # 更新稀疏数据类型的基础类型，并验证更新后的期望结果
    result = original.update_dtype(dtype)
    assert result == expected


@pytest.mark.parametrize(
    "original, dtype, expected_error_msg",
    [
        (
            SparseDtype(float, np.nan),
            int,
            re.escape("Cannot convert non-finite values (NA or inf) to integer"),
        ),
        (
            SparseDtype(str, "abc"),
            int,
            r"invalid literal for int\(\) with base 10: ('abc'|np\.str_\('abc'\))",
        ),
    ],
)
def test_update_dtype_raises(original, dtype, expected_error_msg):
    # 确保更新稀疏数据类型基础类型时，对特定情况引发值错误异常
    with pytest.raises(ValueError, match=expected_error_msg):
        original.update_dtype(dtype)


def test_repr():
    # 测试稀疏数据类型的字符串表示是否正确
    result = str(SparseDtype("int64", fill_value=0))
    expected = "Sparse[int64, 0]"
    assert result == expected

    result = str(SparseDtype(object, fill_value="0"))
    expected = "Sparse[object, '0']"
    assert result == expected


def test_sparse_dtype_subtype_must_be_numpy_dtype():
    # 确保稀疏数据类型的子类型必须是numpy数据类型
    msg = "SparseDtype subtype must be a numpy dtype"
    with pytest.raises(TypeError, match=msg):
        SparseDtype("category", fill_value="c")
```