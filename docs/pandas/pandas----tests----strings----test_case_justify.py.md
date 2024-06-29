# `D:\src\scipysrc\pandas\pandas\tests\strings\test_case_justify.py`

```
# 导入所需模块和函数
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import operator  # 导入 operator 模块

import numpy as np  # 导入 numpy 库，并使用 np 别名
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入 Series 和 _testing 模块
    Series,
    _testing as tm,
)


# 定义测试函数 test_title，测试 Series 的 str.title() 方法
def test_title(any_string_dtype):
    # 创建 Series 对象 s，包含字符串数据和 NaN 值，使用给定的数据类型 any_string_dtype
    s = Series(["FOO", "BAR", np.nan, "Blah", "blurg"], dtype=any_string_dtype)
    # 调用 Series 的 str.title() 方法，将每个字符串首字母大写
    result = s.str.title()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(["Foo", "Bar", np.nan, "Blah", "Blurg"], dtype=any_string_dtype)
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_title_mixed_object，测试处理包含不同类型数据的 Series 的 str.title() 方法
def test_title_mixed_object():
    # 创建 Series 对象 s，包含不同类型的数据
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "blah", None, 1, 2.0])
    # 调用 Series 的 str.title() 方法，将每个字符串首字母大写
    result = s.str.title()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(
        ["Foo", np.nan, "Bar", np.nan, np.nan, "Blah", None, np.nan, np.nan],
        dtype=object,
    )
    # 使用测试模块 tm 的 assert_almost_equal 函数比较 result 和 expected 是否近似相等
    tm.assert_almost_equal(result, expected)


# 定义测试函数 test_lower_upper，测试 Series 的 str.upper() 和 str.lower() 方法
def test_lower_upper(any_string_dtype):
    # 创建 Series 对象 s，包含字符串数据和 NaN 值，使用给定的数据类型 any_string_dtype
    s = Series(["om", np.nan, "nom", "nom"], dtype=any_string_dtype)

    # 调用 Series 的 str.upper() 方法，将所有字符串转换为大写
    result = s.str.upper()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(["OM", np.nan, "NOM", "NOM"], dtype=any_string_dtype)
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 调用 Series 的 str.lower() 方法，将所有字符串转换为小写
    result = result.str.lower()
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和原始 Series 对象 s 是否相等
    tm.assert_series_equal(result, s)


# 定义测试函数 test_lower_upper_mixed_object，测试处理包含不同类型数据的 Series 的 str.upper() 和 str.lower() 方法
def test_lower_upper_mixed_object():
    # 创建 Series 对象 s，包含不同类型的数据
    s = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])

    # 调用 Series 的 str.upper() 方法，将所有字符串转换为大写
    result = s.str.upper()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(
        ["A", np.nan, "B", np.nan, np.nan, "FOO", None, np.nan, np.nan], dtype=object
    )
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 调用 Series 的 str.lower() 方法，将所有字符串转换为小写
    result = s.str.lower()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(
        ["a", np.nan, "b", np.nan, np.nan, "foo", None, np.nan, np.nan], dtype=object
    )
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数 test_capitalize
@pytest.mark.parametrize(
    "data, expected",
    [
        (["FOO", "BAR", np.nan, "Blah", "blurg"], ["Foo", "Bar", np.nan, "Blah", "Blurg"]),
        (["a", "b", "c"], ["A", "B", "C"]),
        (["a b", "a bc. de"], ["A b", "A bc. de"]),
    ],
)
# 定义测试函数 test_capitalize，测试处理包含不同类型数据的 Series 的 str.capitalize() 方法
def test_capitalize(data, expected, any_string_dtype):
    # 创建 Series 对象 s，包含给定的数据列表 data，使用给定的数据类型 any_string_dtype
    s = Series(data, dtype=any_string_dtype)
    # 调用 Series 的 str.capitalize() 方法，将每个字符串的首字母大写，其余小写
    result = s.str.capitalize()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(expected, dtype=any_string_dtype)
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_capitalize_mixed_object，测试处理包含不同类型数据的 Series 的 str.capitalize() 方法
def test_capitalize_mixed_object():
    # 创建 Series 对象 s，包含不同类型的数据
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "blah", None, 1, 2.0])
    # 调用 Series 的 str.capitalize() 方法，将每个字符串的首字母大写，其余小写
    result = s.str.capitalize()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(
        ["Foo", np.nan, "Bar", np.nan, np.nan, "Blah", None, np.nan, np.nan],
        dtype=object,
    )
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_swapcase，测试 Series 的 str.swapcase() 方法
def test_swapcase(any_string_dtype):
    # 创建 Series 对象 s，包含字符串数据和 NaN 值，使用给定的数据类型 any_string_dtype
    s = Series(["FOO", "BAR", np.nan, "Blah", "blurg"], dtype=any_string_dtype)
    # 调用 Series 的 str.swapcase() 方法，将字符串中的大写转换为小写，小写转换为大写
    result = s.str.swapcase()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(["foo", "bar", np.nan, "bLAH", "BLURG"], dtype=any_string_dtype)
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_swapcase_mixed_object，测试处理包含不同类型数据的 Series 的 str.swapcase() 方法
def test_swapcase_mixed_object():
    # 创建 Series 对象 s，包含不同类型的数据
    s = Series(["FOO", np.nan, "bar", True, datetime.today(), "Blah", None, 1, 2.0])
    # 调用 Series 的 str.swapcase() 方法，将字符串中的大写转换为小写，小写转换为大写
    result = s.str.swapcase()
    # 创建期望的 Series 对象 expected，包含预期的结果数据
    expected = Series(
        ["foo", np.nan, "BAR", np.nan, np.nan, "bLAH", None, np.nan, np.nan],
        dtype=object,
    )
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
    # 使用测试框架中的函数来断言两个序列是否相等
    tm.assert_series_equal(result, expected)
def test_casefold():
    # GH25405
    # 设置预期的结果 Series
    expected = Series(["ss", np.nan, "case", "ssd"])
    # 创建 Series 对象 s，包含特定字符串和 NaN 值
    s = Series(["ß", np.nan, "case", "ßd"])
    # 调用字符串方法 casefold()，返回处理后的结果 Series
    result = s.str.casefold()

    # 使用测试框架检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


def test_casemethods(any_string_dtype):
    # 创建包含字符串的列表 values
    values = ["aaa", "bbb", "CCC", "Dddd", "eEEE"]
    # 创建 Series 对象 s，使用指定的数据类型 any_string_dtype
    s = Series(values, dtype=any_string_dtype)
    
    # 使用测试框架验证各种字符串方法的结果是否正确
    assert s.str.lower().tolist() == [v.lower() for v in values]
    assert s.str.upper().tolist() == [v.upper() for v in values]
    assert s.str.title().tolist() == [v.title() for v in values]
    assert s.str.capitalize().tolist() == [v.capitalize() for v in values]
    assert s.str.swapcase().tolist() == [v.swapcase() for v in values]


def test_pad(any_string_dtype):
    # 创建包含字符串和 NaN 的 Series 对象 s
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    # 测试字符串左填充操作
    result = s.str.pad(5, side="left")
    expected = Series(
        ["    a", "    b", np.nan, "    c", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串右填充操作
    result = s.str.pad(5, side="right")
    expected = Series(
        ["a    ", "b    ", np.nan, "c    ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串两侧填充操作
    result = s.str.pad(5, side="both")
    expected = Series(
        ["  a  ", "  b  ", np.nan, "  c  ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_pad_mixed_object():
    # 创建包含多种类型数据的 Series 对象 s
    s = Series(["a", np.nan, "b", True, datetime.today(), "ee", None, 1, 2.0])

    # 测试字符串左填充操作
    result = s.str.pad(5, side="left")
    expected = Series(
        ["    a", np.nan, "    b", np.nan, np.nan, "   ee", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串右填充操作
    result = s.str.pad(5, side="right")
    expected = Series(
        ["a    ", np.nan, "b    ", np.nan, np.nan, "ee   ", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串两侧填充操作
    result = s.str.pad(5, side="both")
    expected = Series(
        ["  a  ", np.nan, "  b  ", np.nan, np.nan, "  ee ", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_pad_fillchar(any_string_dtype):
    # 创建包含字符串和 NaN 的 Series 对象 s
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    # 测试使用指定填充字符进行左填充操作
    result = s.str.pad(5, side="left", fillchar="X")
    expected = Series(
        ["XXXXa", "XXXXb", np.nan, "XXXXc", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试使用指定填充字符进行右填充操作
    result = s.str.pad(5, side="right", fillchar="X")
    expected = Series(
        ["aXXXX", "bXXXX", np.nan, "cXXXX", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试使用指定填充字符进行两侧填充操作
    result = s.str.pad(5, side="both", fillchar="X")
    expected = Series(
        ["XXaXX", "XXbXX", np.nan, "XXcXX", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    # 创建一个 Pandas Series 对象，包含字符串和 NaN 值，使用指定的数据类型 any_string_dtype
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)
    
    # 定义错误消息，用于验证 pytest 引发的异常类型和匹配的错误消息
    msg = "fillchar must be a character, not str"
    
    # 使用 pytest 的上下文管理器，检查是否会引发 TypeError 异常，并验证错误消息匹配
    with pytest.raises(TypeError, match=msg):
        # 对 Series 中的字符串元素进行填充操作，指定填充长度为 5，填充字符为 "XY"
        s.str.pad(5, fillchar="XY")
    
    # 重新定义错误消息，用于验证另一种填充字符类型的异常
    msg = "fillchar must be a character, not int"
    
    # 同样使用 pytest 的上下文管理器，检查是否会引发 TypeError 异常，并验证错误消息匹配
    with pytest.raises(TypeError, match=msg):
        # 对 Series 中的字符串元素进行填充操作，指定填充长度为 5，填充字符为 5
        s.str.pad(5, fillchar=5)
# 使用 pytest 的参数化装饰器来定义多个测试用例，测试字符串的填充方法
@pytest.mark.parametrize("method_name", ["center", "ljust", "rjust", "zfill", "pad"])
def test_pad_width_bad_arg_raises(method_name, any_string_dtype):
    # 见 issue gh-13598
    # 创建包含不同字符串类型的 Series 对象
    s = Series(["1", "22", "a", "bb"], dtype=any_string_dtype)
    # 使用 operator.methodcaller 创建指定字符串填充方法的操作对象
    op = operator.methodcaller(method_name, "f")

    # 准备错误消息，用于测试异常情况
    msg = "width must be of integer type, not str"
    # 断言调用操作对象时会触发 TypeError 异常，并匹配指定的错误消息
    with pytest.raises(TypeError, match=msg):
        op(s.str)


# 测试字符串的居中、左对齐、右对齐方法
def test_center_ljust_rjust(any_string_dtype):
    # 创建包含不同字符串和 NaN 值的 Series 对象
    s = Series(["a", "b", np.nan, "c", np.nan, "eeeeee"], dtype=any_string_dtype)

    # 测试字符串的居中方法
    result = s.str.center(5)
    expected = Series(
        ["  a  ", "  b  ", np.nan, "  c  ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串的左对齐方法
    result = s.str.ljust(5)
    expected = Series(
        ["a    ", "b    ", np.nan, "c    ", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串的右对齐方法
    result = s.str.rjust(5)
    expected = Series(
        ["    a", "    b", np.nan, "    c", np.nan, "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


# 测试包含不同类型对象的 Series 对象的居中、左对齐、右对齐方法
def test_center_ljust_rjust_mixed_object():
    s = Series(["a", np.nan, "b", True, datetime.today(), "c", "eee", None, 1, 2.0])

    # 测试字符串的居中方法
    result = s.str.center(5)
    expected = Series(
        [
            "  a  ",
            np.nan,
            "  b  ",
            np.nan,
            np.nan,
            "  c  ",
            " eee ",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串的左对齐方法
    result = s.str.ljust(5)
    expected = Series(
        [
            "a    ",
            np.nan,
            "b    ",
            np.nan,
            np.nan,
            "c    ",
            "eee  ",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)

    # 测试字符串的右对齐方法
    result = s.str.rjust(5)
    expected = Series(
        [
            "    a",
            np.nan,
            "    b",
            np.nan,
            np.nan,
            "    c",
            "  eee",
            None,
            np.nan,
            np.nan,
        ],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


# 测试指定填充字符的居中方法
def test_center_ljust_rjust_fillchar(any_string_dtype):
    # 如果使用的是特定的字符串类型，则跳过测试
    if any_string_dtype == "string[pyarrow_numpy]":
        pytest.skip(
            "Arrow logic is different, "
            "see https://github.com/pandas-dev/pandas/pull/54533/files#r1299808126",
        )
    # 创建包含不同长度字符串的 Series 对象
    s = Series(["a", "bb", "cccc", "ddddd", "eeeeee"], dtype=any_string_dtype)

    # 测试居中方法，并指定填充字符为 'X'
    result = s.str.center(5, fillchar="X")
    expected = Series(
        ["XXaXX", "XXbbX", "Xcccc", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)
    # 预期结果与 numpy 数组转换为字符串并用 'X' 填充的结果进行比较
    expected = np.array([v.center(5, "X") for v in np.array(s)], dtype=np.object_)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)


这段代码是关于使用 pytest 来测试 Pandas 库中字符串填充方法的单元测试。
    # 对 Series 中的字符串元素进行左对齐填充操作，填充字符为 'X'，长度为 5
    result = s.str.ljust(5, fillchar="X")
    # 创建预期结果 Series，包含左对齐填充后的字符串
    expected = Series(
        ["aXXXX", "bbXXX", "ccccX", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    # 使用测试框架检查结果 Series 是否与预期结果相等
    tm.assert_series_equal(result, expected)
    # 创建预期结果 numpy 数组，包含每个字符串元素左对齐填充后的结果
    expected = np.array([v.ljust(5, "X") for v in np.array(s)], dtype=np.object_)
    # 使用测试框架检查 numpy 数组的结果是否与预期相等
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)

    # 对 Series 中的字符串元素进行右对齐填充操作，填充字符为 'X'，长度为 5
    result = s.str.rjust(5, fillchar="X")
    # 创建预期结果 Series，包含右对齐填充后的字符串
    expected = Series(
        ["XXXXa", "XXXbb", "Xcccc", "ddddd", "eeeeee"], dtype=any_string_dtype
    )
    # 使用测试框架检查结果 Series 是否与预期结果相等
    tm.assert_series_equal(result, expected)
    # 创建预期结果 numpy 数组，包含每个字符串元素右对齐填充后的结果
    expected = np.array([v.rjust(5, "X") for v in np.array(s)], dtype=np.object_)
    # 使用测试框架检查 numpy 数组的结果是否与预期相等
    tm.assert_numpy_array_equal(np.array(result, dtype=np.object_), expected)
def test_wrap(any_string_dtype):
    # 创建一个包含不同字符串的 Series 对象，用于测试文本包装函数
    # 测试值包括：两个单词少于宽度，两个单词等于宽度，两个单词大于宽度，
    # 一个单词少于宽度，一个单词等于宽度，一个单词大于宽度，多个带有尾随空格的标记等于宽度
    s = Series(
        [
            "hello world",
            "hello world!",
            "hello world!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdefa",
            "ab ab ab ab ",
            "ab ab ab ab a",
            "\t",
        ],
        dtype=any_string_dtype,
    )

    # 预期的结果值
    expected = Series(
        [
            "hello world",
            "hello world!",
            "hello\nworld!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdef\na",
            "ab ab ab ab",
            "ab ab ab ab\na",
            "",
        ],
        dtype=any_string_dtype,
    )
    # 调用 Series 对象 s 的 str 属性，并对其内容进行换行处理，每行最多包含 12 个字符，
    # 同时确保长单词可以被打断以适应换行。
    result = s.str.wrap(12, break_long_words=True)
    
    # 使用测试工具 tm 来比较结果 Series 对象 result 和期望的结果 expected 是否相等。
    tm.assert_series_equal(result, expected)
def test_wrap_unicode(any_string_dtype):
    # 定义一个测试函数，用于测试字符串包装功能，传入字符串数据类型参数 any_string_dtype
    # 创建一个 Series 对象 s，包含以下元素：带前后空格（非 Unicode），NaN，以及包含非 ASCII Unicode 字符的字符串
    s = Series(
        ["  pre  ", np.nan, "\xac\u20ac\U00008000 abadcafe"], dtype=any_string_dtype
    )
    # 创建一个期望的 Series 对象 expected，预期结果是字符串被包装后的形式，
    # 第一行字符串中预期结果移除了末尾空格，第二行是 NaN，第三行中的特殊 Unicode 字符应被正确包装
    expected = Series(
        ["  pre", np.nan, "\xac\u20ac\U00008000 ab\nadcafe"], dtype=any_string_dtype
    )
    # 调用 Series 对象 s 的 str.wrap 方法，以 6 个字符的宽度对字符串进行包装处理，返回结果赋给 result
    result = s.str.wrap(6)
    # 使用测试工具包中的 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```