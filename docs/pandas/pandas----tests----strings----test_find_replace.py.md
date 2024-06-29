# `D:\src\scipysrc\pandas\pandas\tests\strings\test_find_replace.py`

```
# 导入必要的库
from datetime import datetime  # 导入datetime模块中的datetime类
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库，重命名为np
import pytest  # 导入pytest测试框架

import pandas.util._test_decorators as td  # 导入pandas的测试装饰器模块

import pandas as pd  # 导入pandas库，重命名为pd
from pandas import (  # 从pandas库导入Series类和_testing模块
    Series,
    _testing as tm,
)
from pandas.tests.strings import (  # 从pandas测试模块中导入相关函数
    _convert_na_value,
    object_pyarrow_numpy,
)

# --------------------------------------------------------------------------------------
# str.contains
# --------------------------------------------------------------------------------------

# 定义一个函数，判断dtype是否为"string[pyarrow]"或"string[pyarrow_numpy]"
def using_pyarrow(dtype):
    return dtype in ("string[pyarrow]", "string[pyarrow_numpy]")

# 定义测试函数，测试Series的str.contains方法
def test_contains(any_string_dtype):
    # 创建包含特定值的NumPy数组
    values = np.array(
        ["foo", np.nan, "fooommm__foo", "mmm_", "foommm[_]+bar"], dtype=np.object_
    )
    values = Series(values, dtype=any_string_dtype)  # 使用any_string_dtype创建Series对象
    pat = "mmm[_]+"  # 定义正则表达式模式

    result = values.str.contains(pat)  # 调用str.contains方法进行匹配
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    expected = Series(
        np.array([False, np.nan, True, True, False], dtype=np.object_),
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)  # 使用测试框架中的assert_series_equal函数比较结果

    result = values.str.contains(pat, regex=False)  # 使用regex=False进行非正则表达式匹配
    expected = Series(
        np.array([False, np.nan, False, False, True], dtype=np.object_),
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)  # 比较结果

    # 创建另一个包含特定值的Series对象
    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=object),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)  # 使用str.contains方法进行匹配
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)  # 比较结果

    # 使用不区分大小写的正则表达式进行匹配
    values = Series(
        np.array(["Foo", "xYz", "fOOomMm__fOo", "MMM_"], dtype=object),
        dtype=any_string_dtype,
    )

    result = values.str.contains("FOO|mmm", case=False)  # 使用case=False进行不区分大小写匹配
    expected = Series(np.array([True, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)  # 比较结果

    # 使用不区分大小写且不使用正则表达式进行匹配
    result = values.str.contains("foo", regex=False, case=False)
    expected = Series(np.array([True, False, True, False]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)  # 比较结果

    # 创建包含特定值的Series对象，包括Unicode字符
    values = Series(
        np.array(["foo", np.nan, "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    pat = "mmm[_]+"

    result = values.str.contains(pat)  # 使用str.contains方法进行匹配
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    expected = Series(
        np.array([False, np.nan, True, True], dtype=np.object_), dtype=expected_dtype
    )
    tm.assert_series_equal(result, expected)  # 比较结果

    result = values.str.contains(pat, na=False)  # 使用na=False参数进行匹配
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    # 使用 pandas.testing 模块中的 assert_series_equal 函数比较 result 和 expected 序列是否相等
    tm.assert_series_equal(result, expected)

    # 创建一个 pandas Series 对象 values，其中包含特定的字符串数组，使用指定的数据类型 any_string_dtype
    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    # 对 values 中的每个字符串应用 str.contains 方法，检查是否包含指定的正则表达式模式 pat
    result = values.str.contains(pat)
    # 创建预期的结果 Series 对象 expected，其中包含布尔数组，表示每个字符串是否匹配正则表达式
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    # 使用 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
def test_contains_object_mixed():
    # 创建一个包含多种类型对象的 Series 对象
    mixed = Series(
        np.array(
            ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
            dtype=object,
        )
    )
    # 对 Series 中的字符串进行模式匹配，返回布尔类型的结果
    result = mixed.str.contains("o")
    # 创建预期结果的 Series 对象，包含布尔值和缺失值
    expected = Series(
        np.array(
            [False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan],
            dtype=np.object_,
        )
    )
    # 使用测试框架验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_contains_na_kwarg_for_object_category():
    # gh 22158

    # 对类别类型的 Series 进行字符串包含操作，考虑缺失值
    values = Series(["a", "b", "c", "a", np.nan], dtype="category")
    result = values.str.contains("a", na=True)
    # 创建预期结果的 Series 对象，包含布尔值
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)

    # 对对象类型的 Series 进行字符串包含操作，考虑缺失值
    values = Series(["a", "b", "c", "a", np.nan])
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "na, expected",
    [
        (None, pd.NA),
        (True, True),
        (False, False),
        (0, False),
        (3, True),
        (np.nan, pd.NA),
    ],
)
@pytest.mark.parametrize("regex", [True, False])
def test_contains_na_kwarg_for_nullable_string_dtype(
    nullable_string_dtype, na, expected, regex
):
    # https://github.com/pandas-dev/pandas/pull/41025#issuecomment-824062416

    # 对可空字符串类型的 Series 进行字符串包含操作，考虑缺失值和正则表达式
    values = Series(["a", "b", "c", "a", np.nan], dtype=nullable_string_dtype)
    result = values.str.contains("a", na=na, regex=regex)
    # 创建预期结果的 Series 对象，包含布尔值，并指定数据类型为布尔型
    expected = Series([True, False, False, True, expected], dtype="boolean")
    tm.assert_series_equal(result, expected)


def test_contains_moar(any_string_dtype):
    # PR #1179
    # 创建一个包含任意类型字符串的 Series 对象
    s = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = s.str.contains("a")
    # 根据输入的字符串类型决定预期结果的数据类型
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期结果的 Series 对象，包含布尔值
    expected = Series(
        [False, False, False, True, True, False, np.nan, False, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("a", case=False)
    expected = Series(
        [True, False, False, True, True, False, np.nan, True, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("Aa")
    expected = Series(
        [False, False, False, True, False, False, np.nan, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba")
    # 这行代码未完，应继续在实际代码中查看
    # 创建一个 Series 对象，包含布尔值和可能的 NaN 值，用于测试结果
    expected = Series(
        [False, False, False, True, False, False, np.nan, False, False, False],
        dtype=expected_dtype,  # 指定 Series 的数据类型
    )
    
    # 使用测试数据集进行字符串包含操作，并将结果存储在 result 中
    result = s.str.contains("ba", case=False)
    
    # 创建一个 Series 对象，包含预期的布尔值和可能的 NaN 值，用于与结果比较
    expected = Series(
        [False, False, False, True, True, False, np.nan, True, False, False],
        dtype=expected_dtype,  # 指定 Series 的数据类型
    )
    
    # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
# 定义测试函数，用于检查 Series 对象中是否包含 NaN 值
def test_contains_nan(any_string_dtype):
    # 创建包含三个 NaN 值的 Series 对象，指定数据类型为 any_string_dtype
    s = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)

    # 检查 Series 对象中每个字符串是否包含子字符串 "foo"，忽略 NaN 值
    result = s.str.contains("foo", na=False)
    # 根据条件判断，确定期望的数据类型
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期的 Series 对象，期望结果全为 False
    expected = Series([False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # 再次检查字符串是否包含 "foo"，但这次考虑 NaN 值
    result = s.str.contains("foo", na=True)
    # 创建预期的 Series 对象，期望结果全为 True
    expected = Series([True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # 检查字符串是否包含 "foo"，并使用字符串 "foo" 作为 NaN 值的替代
    result = s.str.contains("foo", na="foo")
    # 根据 any_string_dtype 的值确定期望的 Series 对象
    if any_string_dtype == "object":
        expected = Series(["foo", "foo", "foo"], dtype=np.object_)
    elif any_string_dtype == "string[pyarrow_numpy]":
        expected = Series([True, True, True], dtype=np.bool_)
    else:
        expected = Series([True, True, True], dtype="boolean")
    tm.assert_series_equal(result, expected)

    # 最后一次检查字符串是否包含 "foo"，不指定 na 参数
    result = s.str.contains("foo")
    # 根据条件确定期望的数据类型
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期的 Series 对象，期望结果全为 NaN
    expected = Series([np.nan, np.nan, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.startswith
# --------------------------------------------------------------------------------------


# 参数化测试函数，测试 Series 对象的 startswith 方法
@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_startswith(pat, dtype, null_value, na):
    # 创建包含特定值的 Series 对象，指定数据类型为 dtype
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    # 检查 Series 对象中的每个字符串是否以 pat 开头
    result = values.str.startswith(pat)
    # 创建预期的 Series 对象，期望结果与实际结果匹配
    exp = Series([False, np.nan, True, False, False, np.nan, True])
    # 根据特定条件填充预期的 Series 对象
    if dtype == "object" and null_value is pd.NA:
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        exp[exp.isna()] = None
    tm.assert_series_equal(result, exp)

    # 再次检查字符串是否以 pat 开头，考虑 na 参数
    result = values.str.startswith(pat, na=na)
    # 创建预期的 Series 对象，期望结果与实际结果匹配
    exp = Series([False, na, True, False, False, na, True])
    tm.assert_series_equal(result, exp)

    # 混合数据类型测试
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=np.object_,
    )
    # 对混合数据类型创建 Series 对象，并检查每个字符串是否以 "f" 开头
    rs = Series(mixed).str.startswith("f")
    # 创建预期的 Series 对象，期望结果与实际结果匹配
    xp = Series([False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)


# 参数化测试函数，测试可为空的字符串数据类型的 startswith 方法
@pytest.mark.parametrize("na", [None, True, False])
def test_startswith_nullable_string_dtype(nullable_string_dtype, na):
    # 创建包含特定值的 Series 对象，指定数据类型为 nullable_string_dtype
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=nullable_string_dtype,
    )
    # 检查 Series 对象中的每个字符串是否以 "foo" 开头，考虑 na 参数
    result = values.str.startswith("foo", na=na)
    # 创建一个布尔Series，用于测试是否与期望结果相等
    exp = Series(
        [False, na, True, False, False, na, True, False, False], dtype="boolean"
    )
    # 使用测试函数tm.assert_series_equal()比较结果和期望值，确保它们相等
    tm.assert_series_equal(result, exp)
    
    # 使用Series的str.startswith()方法检查每个字符串是否以"rege."开头，na参数指定缺失值的处理方式
    result = values.str.startswith("rege.", na=na)
    # 创建一个布尔Series，用于测试是否与期望结果相等
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype="boolean"
    )
    # 使用测试函数tm.assert_series_equal()比较结果和期望值，确保它们相等
    tm.assert_series_equal(result, exp)
# --------------------------------------------------------------------------------------
# str.endswith
# --------------------------------------------------------------------------------------

# 参数化测试：pat参数为"foo"或("foo", "baz")，dtype参数为"object"或"category"，null_value参数为None、np.nan或pd.NA，na参数为True或False
@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_endswith(pat, dtype, null_value, na):
    # 创建Series对象values，包含特定数据和dtype类型
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    # 调用str.endswith方法，根据pat参数返回布尔类型的Series对象result
    result = values.str.endswith(pat)
    # 创建预期结果Series对象exp
    exp = Series([False, np.nan, False, False, True, np.nan, True])
    if dtype == "object" and null_value is pd.NA:
        # 处理特定情况：填充NaN值为pd.NA
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        # 处理特定情况：将exp中的NaN值设置为None
        exp[exp.isna()] = None
    # 断言result与exp相等
    tm.assert_series_equal(result, exp)

    # 调用str.endswith方法，根据pat和na参数返回布尔类型的Series对象result
    result = values.str.endswith(pat, na=na)
    # 创建预期结果Series对象exp
    exp = Series([False, na, False, False, True, na, True])
    # 断言result与exp相等
    tm.assert_series_equal(result, exp)

    # mixed情况：创建包含不同类型数据的numpy数组mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    # 创建Series对象rs，调用str.endswith方法，根据"f"返回布尔类型的Series对象
    rs = Series(mixed).str.endswith("f")
    # 创建预期结果Series对象xp
    xp = Series([False, np.nan, False, np.nan, np.nan, False, None, np.nan, np.nan])
    # 断言rs与xp相等
    tm.assert_series_equal(rs, xp)


# 参数化测试：na参数为None、True或False
@pytest.mark.parametrize("na", [None, True, False])
def test_endswith_nullable_string_dtype(nullable_string_dtype, na):
    # 创建Series对象values，包含特定数据和nullable_string_dtype类型
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=nullable_string_dtype,
    )
    # 调用str.endswith方法，根据"foo"和na参数返回布尔类型的Series对象result
    result = values.str.endswith("foo", na=na)
    # 创建预期结果Series对象exp
    exp = Series(
        [False, na, False, False, True, na, True, False, False], dtype="boolean"
    )
    # 断言result与exp相等
    tm.assert_series_equal(result, exp)

    # 调用str.endswith方法，根据"rege."和na参数返回布尔类型的Series对象result
    result = values.str.endswith("rege.", na=na)
    # 创建预期结果Series对象exp
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype="boolean"
    )
    # 断言result与exp相等
    tm.assert_series_equal(result, exp)


# --------------------------------------------------------------------------------------
# str.replace
# --------------------------------------------------------------------------------------

def test_replace_dict_invalid(any_string_dtype):
    # GH 51914
    # 创建Series对象series，包含特定数据和名称
    series = Series(data=["A", "B_junk", "C_gunk"], name="my_messy_col")
    msg = "repl cannot be used when pat is a dictionary"

    # 使用pytest.raises断言抛出异常(ValueError)，匹配消息msg
    with pytest.raises(ValueError, match=msg):
        # 调用str.replace方法，pat参数为字典{"A": "a", "B": "b"}，repl参数为"A"
        series.str.replace(pat={"A": "a", "B": "b"}, repl="A")


def test_replace_dict(any_string_dtype):
    # GH 51914
    # 创建Series对象series，包含特定数据和名称
    series = Series(data=["A", "B", "C"], name="my_messy_col")
    # 调用str.replace方法，pat参数为字典{"A": "a", "B": "b"}，不指定repl参数
    new_series = series.str.replace(pat={"A": "a", "B": "b"})
    # 创建预期结果Series对象expected
    expected = Series(data=["a", "b", "C"], name="my_messy_col")
    # 断言new_series与expected相等
    tm.assert_series_equal(new_series, expected)


def test_replace(any_string_dtype):
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)
    # 使用 Pandas Series 的 str.replace 方法，替换其中匹配 "BAD[_]*" 正则表达式的字符串为空字符串
    result = ser.str.replace("BAD[_]*", "", regex=True)
    # 创建一个期望的 Series 对象，包含预期的字符串值和 NaN 值，使用指定的数据类型 any_string_dtype
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    # 使用 Pandas 的测试工具 tm.assert_series_equal 检查 result 和 expected 两个 Series 是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于替换字符串中的指定内容并进行单元测试
def test_replace_max_replacements(any_string_dtype):
    # 创建一个包含字符串和 NaN 值的 Series 对象
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # 期望的结果是替换第一个匹配到的 "BAD[_]*" 为 "" 的字符串
    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    # 调用 str.replace 方法进行替换操作
    result = ser.str.replace("BAD[_]*", "", n=1, regex=True)
    # 使用测试框架检查结果是否与期望相同
    tm.assert_series_equal(result, expected)

    # 另一个测试案例，期望的结果是替换第一个匹配到的 "BAD" 为 "" 的字符串（非正则方式）
    expected = Series(["foo__barBAD", np.nan], dtype=any_string_dtype)
    result = ser.str.replace("BAD", "", n=1, regex=False)
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于处理包含不同类型对象的 Series
def test_replace_mixed_object():
    # 创建一个包含不同类型对象的 Series
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    # 使用 str.replace 方法替换所有匹配到的 "BAD[_]*" 为 ""
    result = Series(ser).str.replace("BAD[_]*", "", regex=True)
    # 期望的结果是将所有匹配到的 "BAD[_]*" 替换为 ""
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于处理包含 Unicode 字符的 Series
def test_replace_unicode(any_string_dtype, performance_warning):
    # 创建一个包含 Unicode 字符的 Series
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    # 期望的结果是在逗号后添加一个空格
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    # 使用 str.replace 方法进行替换操作，使用正则表达式进行匹配
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace(r"(?<=\w),(?=\w)", ", ", flags=re.UNICODE, regex=True)
    tm.assert_series_equal(result, expected)


# 使用参数化测试装饰器，测试不同的替换参数 repl 和 data
@pytest.mark.parametrize("repl", [None, 3, {"a": "b"}])
@pytest.mark.parametrize("data", [["a", "b", None], ["a", "b", "c", "ad"]])
def test_replace_wrong_repl_type_raises(any_string_dtype, index_or_series, repl, data):
    # 创建一个索引或 Series，其中包含指定数据类型的数据
    obj = index_or_series(data, dtype=any_string_dtype)
    # 期望抛出 TypeError 异常，提示 repl 参数必须是字符串或可调用对象
    msg = "repl must be a string or callable"
    with pytest.raises(TypeError, match=msg):
        # 使用 str.replace 方法尝试替换 "a" 为 repl 参数
        obj.str.replace("a", repl)


# 定义一个测试函数，测试使用可调用对象进行替换操作
def test_replace_callable(any_string_dtype, performance_warning):
    # 创建一个包含字符串和 NaN 值的 Series 对象
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # 定义一个可调用对象，用于替换匹配的字符串
    repl = lambda m: m.group(0).swapcase()
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        # 使用 str.replace 方法进行替换操作，使用正则表达式匹配
        result = ser.str.replace("[a-z][A-Z]{2}", repl, n=2, regex=True)
    # 期望的结果是将匹配到的字符反转大小写
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


# 使用参数化测试装饰器，测试不同参数 repl 引发的异常情况
@pytest.mark.parametrize(
    "repl", [lambda: None, lambda m, x: None, lambda m, x, y=None: None]
)
def test_replace_callable_raises(any_string_dtype, performance_warning, repl):
    # 创建一个包含字符串和 NaN 值的 Series 对象
    values = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # 期望抛出 TypeError 异常，提示可调用对象的参数个数不正确
    msg = (
        r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
        r"(?(3)required )positional arguments?"
    )
    if not using_pyarrow(any_string_dtype):
        performance_warning = False
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(performance_warning):
            # 使用 str.replace 方法尝试替换 "a" 为 repl 参数
            values.str.replace("a", repl, regex=True)
# 测试替换操作，使用正则表达式的命名分组
def test_replace_callable_named_groups(any_string_dtype, performance_warning):
    # 创建一个包含字符串和 NaN 值的序列
    ser = Series(["Foo Bar Baz", np.nan], dtype=any_string_dtype)
    # 定义正则表达式模式，包含命名分组
    pat = r"(?P<first>\w+) (?P<middle>\w+) (?P<last>\w+)"
    # 定义替换函数，根据命名分组返回修改后的字符串
    repl = lambda m: m.group("middle").swapcase()
    # 使用正则表达式替换每个匹配的模式，生成新的序列
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace(pat, repl, regex=True)
    # 期望的结果序列
    expected = Series(["bAR", np.nan], dtype=any_string_dtype)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 测试使用预编译的正则表达式进行替换
def test_replace_compiled_regex(any_string_dtype, performance_warning):
    # 创建一个包含字符串和 NaN 值的序列
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # 使用预编译的正则表达式模式
    pat = re.compile(r"BAD_*")
    # 使用预编译的正则表达式进行替换操作，删除匹配的模式
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace(pat, "", regex=True)
    # 期望的结果序列
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)

    # 限制替换次数为1次
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace(pat, "", n=1, regex=True)
    # 期望的结果序列
    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 测试使用预编译的正则表达式替换字符串序列中的模式
def test_replace_compiled_regex_mixed_object():
    # 定义预编译的正则表达式模式
    pat = re.compile(r"BAD_*")
    # 创建一个混合类型的序列
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    # 使用预编译的正则表达式替换字符串序列中的匹配模式
    result = Series(ser).str.replace(pat, "", regex=True)
    # 期望的结果序列，替换匹配模式后的新序列
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 测试使用预编译的正则表达式处理Unicode字符串的替换操作
def test_replace_compiled_regex_unicode(any_string_dtype, performance_warning):
    # 创建一个包含 Unicode 字符串的序列
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    # 期望的结果序列，替换操作后的新序列
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    # 定义带有 Unicode 标志的正则表达式模式
    pat = re.compile(r"(?<=\w),(?=\w)", flags=re.UNICODE)
    # 使用预编译的正则表达式替换每个匹配的模式
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace(pat, ", ", regex=True)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 测试使用预编译的正则表达式时的异常情况
def test_replace_compiled_regex_raises(any_string_dtype):
    # 创建一个包含字符串和 NaN 值的序列
    ser = Series(["fooBAD__barBAD__bad", np.nan], dtype=any_string_dtype)
    # 定义预编译的正则表达式模式
    pat = re.compile(r"BAD_*")

    # 试图在使用预编译的正则表达式时设置不支持的选项会引发 ValueError 异常
    msg = "case and flags cannot be set when pat is a compiled regex"

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", flags=re.IGNORECASE, regex=True)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=False, regex=True)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=True, regex=True)
    # 创建一个包含字符串和 NaN 值的 Pandas Series 对象，使用指定的数据类型 `any_string_dtype`
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)
    
    # 定义一个 lambda 函数 `repl`，用于对匹配到的字符串进行大小写转换
    repl = lambda m: m.group(0).swapcase()
    
    # 编译正则表达式模式 `pat`，匹配小写字母后跟两个大写字母的字符串序列
    pat = re.compile("[a-z][A-Z]{2}")
    
    # 使用 `maybe_produces_warning` 上下文管理器，设置可能的性能警告和使用 `any_string_dtype` 的 `pyarrow` 数据类型
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        # 对 Series `ser` 中的字符串应用替换操作，匹配 `pat` 的部分使用 `repl` 进行替换，最多替换 2 次
        result = ser.str.replace(pat, repl, n=2, regex=True)
    
    # 创建预期的 Pandas Series `expected`，包含转换后的字符串和 NaN 值，数据类型为 `any_string_dtype`
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    
    # 使用测试工具 `assert_series_equal` 检查 `result` 和 `expected` 是否相等
    tm.assert_series_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_replace_literal 提供多组参数组合
@pytest.mark.parametrize("regex,expected_val", [(True, "bao"), (False, "foo")])
def test_replace_literal(regex, expected_val, any_string_dtype):
    # GH16808 literal replace (regex=False vs regex=True)
    # 创建 Series 对象，包含字符串 "f.o", "foo", 和 np.nan，使用指定的数据类型 any_string_dtype
    ser = Series(["f.o", "foo", np.nan], dtype=any_string_dtype)
    # 创建期望的 Series 对象，使用指定的数据类型 any_string_dtype
    expected = Series(["bao", expected_val, np.nan], dtype=any_string_dtype)
    # 调用 Series 的 str.replace 方法，根据 regex 参数是否为 True，进行字符串替换操作
    result = ser.str.replace("f.", "ba", regex=regex)
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected，确保它们相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_replace_literal_callable_raises，测试当 regex=False 时，使用可调用对象作为替换对象是否会抛出 ValueError 异常
def test_replace_literal_callable_raises(any_string_dtype):
    # 创建空的 Series 对象，使用指定的数据类型 any_string_dtype
    ser = Series([], dtype=any_string_dtype)
    # 定义一个 lambda 函数 repl，用于替换操作
    repl = lambda m: m.group(0).swapcase()

    # 定义异常消息
    msg = "Cannot use a callable replacement when regex=False"
    # 使用 pytest 的 raises 函数验证调用 str.replace 方法时是否会抛出 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        ser.str.replace("abc", repl, regex=False)


# 定义测试函数 test_replace_literal_compiled_raises，测试当 regex=False 时，使用预编译的正则表达式作为替换模式是否会抛出 ValueError 异常
def test_replace_literal_compiled_raises(any_string_dtype):
    # 创建空的 Series 对象，使用指定的数据类型 any_string_dtype
    ser = Series([], dtype=any_string_dtype)
    # 编译正则表达式，用于替换操作
    pat = re.compile("[a-z][A-Z]{2}")

    # 定义异常消息
    msg = "Cannot use a compiled regex as replacement pattern with regex=False"
    # 使用 pytest 的 raises 函数验证调用 str.replace 方法时是否会抛出 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", regex=False)


# 定义测试函数 test_replace_moar，测试不同的字符串替换操作，包括简单替换和正则表达式替换
def test_replace_moar(any_string_dtype, performance_warning):
    # PR #1179
    # 创建 Series 对象，包含多个字符串和 NaN 值，使用指定的数据类型 any_string_dtype
    ser = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    # 进行简单的字符串替换操作，将 "A" 替换为 "YYY"
    result = ser.str.replace("A", "YYY")
    # 期望的替换结果
    expected = Series(
        ["YYY", "B", "C", "YYYaba", "Baca", "", np.nan, "CYYYBYYY", "dog", "cat"],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected，确保它们相等
    tm.assert_series_equal(result, expected)

    # 进行带有警告的字符串替换操作，将 "A" 替换为 "YYY"，同时忽略大小写
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace("A", "YYY", case=False)
    # 期望的替换结果
    expected = Series(
        [
            "YYY",
            "B",
            "C",
            "YYYYYYbYYY",
            "BYYYcYYY",
            "",
            np.nan,
            "CYYYBYYY",
            "dog",
            "cYYYt",
        ],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected，确保它们相等
    tm.assert_series_equal(result, expected)

    # 进行带有警告的正则表达式替换操作，将匹配 "^\.a|dog" 的字符串替换为 "XX-XX "，忽略大小写
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace("^.a|dog", "XX-XX ", case=False, regex=True)
    # 期望的替换结果
    expected = Series(
        [
            "A",
            "B",
            "C",
            "XX-XX ba",
            "XX-XX ca",
            "",
            np.nan,
            "XX-XX BA",
            "XX-XX ",
            "XX-XX t",
        ],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected，确保它们相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_replace_not_case_sensitive_not_regex，测试在非大小写敏感且非正则表达式模式下的字符串替换操作
def test_replace_not_case_sensitive_not_regex(any_string_dtype, performance_warning):
    # https://github.com/pandas-dev/pandas/issues/41602
    # 创建 Series 对象，包含字符串 "A.", "a.", "Ab", "ab" 和 np.nan，使用指定的数据类型 any_string_dtype
    ser = Series(["A.", "a.", "Ab", "ab", np.nan], dtype=any_string_dtype)

    # 进行不区分大小写的简单字符串替换操作，将 "a" 替换为 "c"
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.replace("a", "c", case=False, regex=False)
    # 期望的替换结果
    expected = Series(["c.", "c.", "Ab", "cb", np.nan], dtype=any_string_dtype)
    # 使用测试工具来比较结果和期望值的序列是否相等
    tm.assert_series_equal(result, expected)
    
    # 通过可能产生警告的上下文管理器来执行字符串替换操作，并将结果赋给result变量
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        # 对字符串序列进行替换操作，将"a."替换为"c."，不区分大小写，不使用正则表达式
        result = ser.str.replace("a.", "c.", case=False, regex=False)
    
    # 期望的结果序列，包含了替换后的预期值以及可能的缺失值(np.nan)，数据类型为任意字符串类型
    expected = Series(["c.", "c.", "Ab", "ab", np.nan], dtype=any_string_dtype)
    
    # 使用测试工具来再次比较结果和更新后的期望值的序列是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试字符串替换操作，接受一个字符串数据类型参数 `any_string_dtype`
def test_replace_regex(any_string_dtype):
    # 创建一个包含特定字符串的 Series 对象 `s`，包括 "a", "b", "ac", NaN, ""
    s = Series(["a", "b", "ac", np.nan, ""], dtype=any_string_dtype)
    # 使用正则表达式替换操作，将单个字符的字符串（符合 "^.$" 正则表达式）替换为 "a"
    result = s.str.replace("^.$", "a", regex=True)
    # 预期的结果 Series `expected`，与 `result` 进行比较
    expected = Series(["a", "a", "ac", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


# 使用 pytest 参数化装饰器，定义另一个测试函数，用于测试字符串替换操作，接受 `regex` 和 `any_string_dtype` 两个参数
@pytest.mark.parametrize("regex", [True, False])
def test_replace_regex_single_character(regex, any_string_dtype):
    # 创建一个包含特定字符串的 Series 对象 `s`，包括 "a.b", ".", "b", NaN, ""
    s = Series(["a.b", ".", "b", np.nan, ""], dtype=any_string_dtype)

    # 根据参数 `regex` 决定是否使用正则表达式替换操作
    result = s.str.replace(".", "a", regex=regex)
    # 根据 `regex` 参数不同，设定不同的预期结果 Series `expected`，与 `result` 进行比较
    if regex:
        expected = Series(["aaa", "a", "a", np.nan, ""], dtype=any_string_dtype)
    else:
        expected = Series(["aab", "a", "b", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.match
# --------------------------------------------------------------------------------------


# 定义一个测试函数，用于测试字符串匹配操作，接受一个字符串数据类型参数 `any_string_dtype`
def test_match(any_string_dtype):
    # 根据 `any_string_dtype` 的类型确定预期的数据类型 `expected_dtype`
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"

    # 创建一个包含特定字符串的 Series 对象 `values`
    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    # 进行字符串匹配操作，根据给定的正则表达式 ".*(BAD[_]+).*(BAD)"，返回匹配结果的 Series `result`
    result = values.str.match(".*(BAD[_]+).*(BAD)")
    # 预期的结果 Series `expected`，与 `result` 进行比较
    expected = Series([True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # 继续测试，更新 Series `values`，并进行匹配操作，比较多个不同的预期结果
    values = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match(".*BAD[_]+.*BAD")
    expected = Series([True, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("BAD[_]+.*BAD")
    expected = Series([False, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        ["fooBAD__barBAD", "^BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match("^BAD[_]+.*BAD")
    expected = Series([False, False, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("\\^BAD[_]+.*BAD")
    expected = Series([False, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，测试混合对象类型的字符串匹配操作
def test_match_mixed_object():
    # 创建一个包含多种类型数据的 Series `mixed`
    mixed = Series(
        [
            "aBAD_BAD",
            np.nan,
            "BAD_b_BAD",
            True,
            datetime.today(),
            "foo",
            None,
            1,
            2.0,
        ]
    )
    # 进行字符串匹配操作，使用正则表达式 ".*(BAD[_]+).*(BAD)"，返回匹配结果的 Series `result`
    result = Series(mixed).str.match(".*(BAD[_]+).*(BAD)")
    # 预期的结果 Series `expected`，与 `result` 进行比较
    expected = Series([True, np.nan, True, np.nan, np.nan, False, None, np.nan, np.nan])
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，测试 `match` 方法中 `na` 参数的影响
def test_match_na_kwarg(any_string_dtype):
    # GH #6609
    # 创建一个包含字符串和 NaN 值的 Series 对象 s，使用指定的数据类型 any_string_dtype
    s = Series(["a", "b", np.nan], dtype=any_string_dtype)
    
    # 使用 Series 对象 s 的 str.match 方法进行字符串匹配，返回匹配结果的 Series 对象 result，
    # na=False 表示对 NaN 值不进行匹配，将其视作 False
    result = s.str.match("a", na=False)
    
    # 根据条件判断确定期望的数据类型 expected_dtype，如果 any_string_dtype 在 object_pyarrow_numpy 中，则为 np.bool_，否则为 "boolean"
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    
    # 创建期望的结果 Series 对象 expected，包含与匹配结果相对应的布尔值序列
    expected = Series([True, False, False], dtype=expected_dtype)
    
    # 使用测试框架中的函数 tm.assert_series_equal 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)
    
    # 再次使用 Series 对象 s 的 str.match 方法进行字符串匹配，返回匹配结果的 Series 对象 result，
    # 此次未指定 na 参数，即使用默认值 True，表示对 NaN 值进行匹配，返回结果为 np.nan
    result = s.str.match("a")
    
    # 根据条件判断确定期望的数据类型 expected_dtype，如果 any_string_dtype 在 object_pyarrow_numpy 中，则为 "object"，否则为 "boolean"
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    
    # 创建期望的结果 Series 对象 expected，包含与匹配结果相对应的值序列，包括 True、False 和 np.nan
    expected = Series([True, False, np.nan], dtype=expected_dtype)
    
    # 使用测试框架中的函数 tm.assert_series_equal 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试 Series 对象在指定条件下的字符串匹配行为
def test_match_case_kwarg(any_string_dtype):
    # 创建包含字符串的 Series 对象
    values = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    # 对 Series 中的字符串进行不区分大小写的匹配，返回匹配结果的 Series
    result = values.str.match("ab", case=False)
    # 根据条件确定期望的数据类型
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建期望的结果 Series
    expected = Series([True, True, True, True], dtype=expected_dtype)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.fullmatch
# --------------------------------------------------------------------------------------


def test_fullmatch(any_string_dtype):
    # GH 32806
    # 创建包含字符串的 Series 对象，包括 NaN 值
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    # 对 Series 中的每个字符串进行全匹配检查，返回匹配结果的 Series
    result = ser.str.fullmatch(".*BAD[_]+.*BAD")
    # 根据条件确定期望的数据类型
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建期望的结果 Series
    expected = Series([True, False, np.nan, False], dtype=expected_dtype)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_fullmatch_dollar_literal(any_string_dtype):
    # GH 56652
    # 创建包含特定字符串的 Series 对象，包括 NaN 值
    ser = Series(["foo", "foo$foo", np.nan, "foo$"], dtype=any_string_dtype)
    # 对 Series 中的每个字符串进行全匹配检查，返回匹配结果的 Series
    result = ser.str.fullmatch("foo\\$")
    # 根据条件确定期望的数据类型
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建期望的结果 Series
    expected = Series([False, False, np.nan, True], dtype=expected_dtype)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_fullmatch_na_kwarg(any_string_dtype):
    # 创建包含字符串的 Series 对象，包括 NaN 值
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    # 对 Series 中的每个字符串进行全匹配检查，返回匹配结果的 Series
    result = ser.str.fullmatch(".*BAD[_]+.*BAD", na=False)
    # 根据条件确定期望的数据类型
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建期望的结果 Series
    expected = Series([True, False, False, False], dtype=expected_dtype)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_fullmatch_case_kwarg(any_string_dtype, performance_warning):
    # 创建包含字符串的 Series 对象
    ser = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    # 根据条件确定期望的数据类型
    expected_dtype = np.bool_ if any_string_dtype in object_pyarrow_numpy else "boolean"

    # 创建期望的结果 Series
    expected = Series([True, False, False, False], dtype=expected_dtype)
    # 对 Series 中的每个字符串进行大小写敏感的全匹配检查，返回匹配结果的 Series
    result = ser.str.fullmatch("ab", case=True)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 创建期望的结果 Series
    expected = Series([True, True, False, False], dtype=expected_dtype)
    # 对 Series 中的每个字符串进行大小写不敏感的全匹配检查，返回匹配结果的 Series
    result = ser.str.fullmatch("ab", case=False)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 对 Series 中的每个字符串进行忽略大小写的全匹配检查，返回匹配结果的 Series，并可能产生警告
    with tm.maybe_produces_warning(
        performance_warning, using_pyarrow(any_string_dtype)
    ):
        result = ser.str.fullmatch("ab", flags=re.IGNORECASE)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.findall
# --------------------------------------------------------------------------------------


def test_findall(any_string_dtype):
    # 创建包含字符串的 Series 对象，包括 NaN 值
    ser = Series(["fooBAD__barBAD", np.nan, "foo", "BAD"], dtype=any_string_dtype)
    # 对 Series 中的每个字符串进行查找所有匹配的子串，返回匹配结果的 Series
    result = ser.str.findall("BAD[_]*")
    # 创建一个 Series 对象，包含多个列表作为元素，其中包括字符串列表和 NaN 值的组合
    expected = Series([["BAD__", "BAD"], np.nan, [], ["BAD"]])
    
    # 调用函数 _convert_na_value 处理 Series 对象 ser 和预期的结果 expected，将其中的缺失值转换为统一的表示
    expected = _convert_na_value(ser, expected)
    
    # 使用测试工具包中的 assert_series_equal 函数比较变量 result 和 expected 是否相等，用于测试结果的正确性
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试字符串序列中的 find 和 rfind 方法
def test_findall_mixed_object():
    # 创建一个包含不同类型数据的 Series 对象
    ser = Series(
        [
            "fooBAD__barBAD",
            np.nan,
            "foo",
            True,
            datetime.today(),
            "BAD",
            None,
            1,
            2.0,
        ]
    )

    # 使用 str.findall 方法查找每个字符串中的所有匹配项，返回结果存储在 result 中
    result = ser.str.findall("BAD[_]*")
    # 创建一个预期的 Series 对象，包含每个字符串中的预期结果
    expected = Series(
        [
            ["BAD__", "BAD"],
            np.nan,
            [],
            np.nan,
            np.nan,
            ["BAD"],
            None,
            np.nan,
            np.nan,
        ]
    )
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.find
# --------------------------------------------------------------------------------------

# 定义测试函数 test_find，用于测试 Series 对象的 str.find 和 str.rfind 方法
def test_find(any_string_dtype):
    # 创建一个包含字符串的 Series 对象，类型为 any_string_dtype
    ser = Series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF", "XXXX"], dtype=any_string_dtype
    )
    # 根据数据类型选择预期的结果类型
    expected_dtype = np.int64 if any_string_dtype in object_pyarrow_numpy else "Int64"

    # 测试 str.find 方法，查找字符串中子串 "EF" 的位置
    result = ser.str.find("EF")
    # 创建预期的结果 Series 对象，包含每个字符串中子串 "EF" 的位置
    expected = Series([4, 3, 1, 0, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.find("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    # 测试 str.rfind 方法，查找字符串中子串 "EF" 最后出现的位置
    result = ser.str.rfind("EF")
    # 创建预期的结果 Series 对象，包含每个字符串中子串 "EF" 最后出现的位置
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.rfind("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    # 测试 str.find 方法，查找字符串中子串 "EF" 的位置，指定起始搜索位置
    result = ser.str.find("EF", 3)
    # 创建预期的结果 Series 对象，包含每个字符串中从指定位置开始的子串 "EF" 的位置
    expected = Series([4, 3, 7, 4, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.find("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    # 测试 str.rfind 方法，查找字符串中子串 "EF" 最后出现的位置，指定起始搜索位置
    result = ser.str.rfind("EF", 3)
    # 创建预期的结果 Series 对象，包含每个字符串中从指定位置开始的子串 "EF" 最后出现的位置
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.rfind("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    # 测试 str.find 方法，查找字符串中子串 "EF" 的位置，指定起始和结束搜索位置
    result = ser.str.find("EF", 3, 6)
    # 创建预期的结果 Series 对象，包含每个字符串中从指定位置开始到结束位置的子串 "EF" 的位置
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.find("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    # 测试 str.rfind 方法，查找字符串中子串 "EF" 最后出现的位置，指定起始和结束搜索位置
    result = ser.str.rfind("EF", 3, 6)
    # 创建预期的结果 Series 对象，包含每个字符串中从指定位置开始到结束位置的子串 "EF" 最后出现的位置
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 使用 numpy 数组比较工具，检查 result 转换为 numpy 数组后是否与预期的一致
    expected = np.array([v.rfind("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)


# 定义一个测试函数，用于测试当输入参数不合规时是否会引发异常
def test_find_bad_arg_raises(any_string_dtype):
    # 创建一个空的 Series 对象，类型为 any_string_dtype
    ser = Series([], dtype=any_string_dtype)
    # 使用 pytest 的断言 `raises` 来测试是否会抛出指定类型的异常（TypeError），并且异常消息中包含特定的字符串匹配（"expected a string object, not int"）
    with pytest.raises(TypeError, match="expected a string object, not int"):
        # 调用 ser 对象的 str 属性的 find 方法，传入整数参数 0
        ser.str.find(0)
    
    # 使用 pytest 的断言 `raises` 来测试是否会抛出指定类型的异常（TypeError），并且异常消息中包含特定的字符串匹配（"expected a string object, not int"）
    with pytest.raises(TypeError, match="expected a string object, not int"):
        # 调用 ser 对象的 str 属性的 rfind 方法，传入整数参数 0
        ser.str.rfind(0)
# 定义一个测试函数，用于测试包含 NaN 值的 Series 对象中的字符串查找功能
def test_find_nan(any_string_dtype):
    # 创建一个 Series 对象，包含字符串和 NaN 值，指定数据类型为 any_string_dtype
    ser = Series(
        ["ABCDEFG", np.nan, "DEFGHIJEF", np.nan, "XXXX"], dtype=any_string_dtype
    )
    # 根据条件判断设置期望的数据类型
    expected_dtype = np.float64 if any_string_dtype in object_pyarrow_numpy else "Int64"

    # 使用 Series 的 str.find 方法查找子字符串 "EF" 的位置
    result = ser.str.find("EF")
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, 1, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 Series 的 str.rfind 方法从右侧查找子字符串 "EF" 的位置
    result = ser.str.rfind("EF")
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 Series 的 str.find 方法从指定位置开始查找子字符串 "EF" 的位置
    result = ser.str.find("EF", 3)
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 Series 的 str.rfind 方法从指定位置开始从右侧查找子字符串 "EF" 的位置
    result = ser.str.rfind("EF", 3)
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 Series 的 str.find 方法从指定范围内查找子字符串 "EF" 的位置
    result = ser.str.find("EF", 3, 6)
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 Series 的 str.rfind 方法从指定范围内从右侧查找子字符串 "EF" 的位置
    result = ser.str.rfind("EF", 3, 6)
    # 创建期望的结果 Series 对象
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.translate
# --------------------------------------------------------------------------------------


# 使用参数化测试装饰器对 str.translate 方法进行参数化测试
@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
def test_translate(index_or_series, any_string_dtype, infer_string):
    # 创建一个 Index 或 Series 对象，包含字符串，指定数据类型为 any_string_dtype
    obj = index_or_series(
        ["abcdefg", "abcc", "cdddfg", "cdefggg"], dtype=any_string_dtype
    )
    # 使用 str.maketrans 方法创建字符转换表
    table = str.maketrans("abc", "cde")
    # 使用 Series 的 str.translate 方法应用字符转换表
    result = obj.str.translate(table)
    # 创建期望的结果 Index 或 Series 对象
    expected = index_or_series(
        ["cdedefg", "cdee", "edddfg", "edefggg"], dtype=any_string_dtype
    )
    # 断言结果与期望一致
    tm.assert_equal(result, expected)


# 定义一个测试函数，测试处理包含非字符串值的 Series 对象的 str.translate 方法
def test_translate_mixed_object():
    # 创建一个包含非字符串值的 Series 对象
    s = Series(["a", "b", "c", 1.2])
    # 使用 str.maketrans 方法创建字符转换表
    table = str.maketrans("abc", "cde")
    # 创建期望的结果 Series 对象
    expected = Series(["c", "d", "e", np.nan], dtype=object)
    # 使用 Series 的 str.translate 方法应用字符转换表
    result = s.str.translate(table)
    # 断言结果与期望一致
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------


# 定义一个测试函数，测试处理包含特定模式的 Series 对象的 str.extract 和 str.match 方法
def test_flags_kwarg(any_string_dtype, performance_warning):
    # 创建一个包含特定数据的 Series 对象
    data = {
        "Dave": "dave@google.com",
        "Steve": "steve@gmail.com",
        "Rob": "rob@gmail.com",
        "Wes": np.nan,
    }
    data = Series(data, dtype=any_string_dtype)

    # 定义一个匹配电子邮件模式的正则表达式模式
    pat = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"

    # 根据数据类型是否使用 pyarrow 来选择使用 str.extract 方法
    use_pyarrow = using_pyarrow(any_string_dtype)

    # 使用 Series 的 str.extract 方法根据正则表达式模式提取数据
    result = data.str.extract(pat, flags=re.IGNORECASE, expand=True)
    # 断言结果的第一行与期望一致
    assert result.iloc[0].tolist() == ["dave", "google", "com"]

    # 使用 Series 的 str.match 方法根据正则表达式模式匹配数据
    with tm.maybe_produces_warning(performance_warning, use_pyarrow):
        result = data.str.match(pat, flags=re.IGNORECASE)
    # 断言结果的第一行为真
    assert result.iloc[0]
    # 使用 tm.maybe_produces_warning 函数处理可能产生的性能警告，根据参数 performance_warning 和 use_pyarrow 判断是否需要警告
    with tm.maybe_produces_warning(performance_warning, use_pyarrow):
        # 使用 data.str.fullmatch 方法对数据进行全匹配，返回匹配结果
        result = data.str.fullmatch(pat, flags=re.IGNORECASE)
    # 断言结果的第一个元素为真
    assert result.iloc[0]

    # 使用 data.str.findall 方法查找数据中所有匹配的子串，忽略大小写
    result = data.str.findall(pat, flags=re.IGNORECASE)
    # 断言结果的第一个元素的第一个匹配结果为 ("dave", "google", "com")
    assert result.iloc[0][0] == ("dave", "google", "com")

    # 使用 data.str.count 方法计算数据中匹配的次数，忽略大小写
    result = data.str.count(pat, flags=re.IGNORECASE)
    # 断言结果的第一个元素为 1
    assert result.iloc[0] == 1

    # 设置警告信息文本
    msg = "has match groups"
    # 使用 tm.assert_produces_warning 函数检查是否会产生指定警告类型的警告，根据参数 raise_on_extra_warnings 决定是否抛出额外的警告
    with tm.assert_produces_warning(
        UserWarning, match=msg, raise_on_extra_warnings=not use_pyarrow
    ):
        # 使用 data.str.contains 方法检查数据中是否包含指定模式的子串，忽略大小写
        result = data.str.contains(pat, flags=re.IGNORECASE)
    # 断言结果的第一个元素为真
    assert result.iloc[0]
```