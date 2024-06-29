# `D:\src\scipysrc\pandas\pandas\tests\strings\test_extract.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库，用于编写测试
import pytest

# 从 pandas 库中导入特定模块和类
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


# 定义一个测试函数，测试在错误类型下是否会抛出 TypeError 异常
def test_extract_expand_kwarg_wrong_type_raises(any_string_dtype):
    # 创建一个包含多种类型数据的 Series 对象
    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match="expand must be True or False"):
        # 调用 Series 的 str.extract 方法，尝试提取特定模式的字符串
        values.str.extract(".*(BAD[_]+).*(BAD)", expand=None)


# 定义一个测试函数，测试在正常情况下是否正确提取字符串
def test_extract_expand_kwarg(any_string_dtype):
    # 创建一个包含多种类型数据的 Series 对象
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    # 创建一个期望的 DataFrame 对象，用于验证提取操作的结果
    expected = DataFrame(["BAD__", np.nan, np.nan], dtype=any_string_dtype)

    # 第一次调用 str.extract 方法，不展开结果
    result = s.str.extract(".*(BAD[_]+).*")
    tm.assert_frame_equal(result, expected)

    # 第二次调用 str.extract 方法，展开结果
    result = s.str.extract(".*(BAD[_]+).*", expand=True)
    tm.assert_frame_equal(result, expected)

    # 创建另一个期望的 DataFrame 对象，用于验证提取操作的结果
    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    # 第三次调用 str.extract 方法，不展开结果，使用多个捕获组
    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在混合对象类型下是否正确提取字符串
def test_extract_expand_False_mixed_object():
    # 创建一个包含多种对象类型数据的 Series 对象
    ser = Series(
        ["aBAD_BAD", np.nan, "BAD_b_BAD", True, datetime.today(), "foo", None, 1, 2.0]
    )

    # 第一次调用 str.extract 方法，不展开结果，使用两个捕获组
    result = ser.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    er = [np.nan, np.nan]  # 空行
    # 创建一个期望的 DataFrame 对象，用于验证提取操作的结果
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )
    tm.assert_frame_equal(result, expected)

    # 第二次调用 str.extract 方法，不展开结果，使用单个捕获组
    result = ser.str.extract(".*(BAD[_]+).*BAD", expand=False)
    # 创建一个期望的 Series 对象，用于验证提取操作的结果
    expected = Series(
        ["BAD_", np.nan, "BAD_", np.nan, np.nan, np.nan, None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，测试在索引对象下是否会抛出异常
def test_extract_expand_index_raises():
    # 创建一个 Index 对象
    idx = Index(["A1", "A2", "A3", "A4", "B5"])
    msg = "only one regex group is supported with Index"
    # 使用 pytest 检查是否会抛出 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match=msg):
        # 调用 Index 的 str.extract 方法，尝试在索引对象上提取特定模式的字符串
        idx.str.extract("([AB])([123])", expand=False)


# 定义一个测试函数，测试在不包含捕获组的情况下是否会抛出异常
def test_extract_expand_no_capture_groups_raises(index_or_series, any_string_dtype):
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)
    msg = "pattern contains no capture groups"

    # 没有捕获组的情况
    with pytest.raises(ValueError, match=msg):
        # 调用 Series/Index 的 str.extract 方法，尝试在对象上提取特定模式的字符串
        s_or_idx.str.extract("[ABC][123]", expand=False)

    # 只包含非捕获组的情况
    with pytest.raises(ValueError, match=msg):
        # 调用 Series/Index 的 str.extract 方法，尝试在对象上提取特定模式的字符串
        s_or_idx.str.extract("(?:[AB]).*", expand=False)


# 定义一个测试函数，测试在包含单个捕获组的情况下是否正确提取字符串
def test_extract_expand_single_capture_group(index_or_series, any_string_dtype):
    # 创建一个 Series/Index 对象
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)
    # 调用 Series/Index 的 str.extract 方法，使用命名捕获组提取特定模式的字符串
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=False)
    # 创建一个期望的 Series 对象，包含两个值为 "A" 的元素，指定名称为 "uno"，数据类型为任意字符串类型
    expected = index_or_series(["A", "A"], name="uno", dtype=any_string_dtype)
    # 如果 index_or_series 的类型为 Series
    if index_or_series == Series:
        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
    # 如果 index_or_series 的类型不是 Series
    else:
        # 使用测试工具函数检查 result 和 expected 的索引是否相等
        tm.assert_index_equal(result, expected)
def test_extract_expand_capture_groups(any_string_dtype):
    # 创建一个包含字符串的 Series，指定数据类型为 any_string_dtype
    s = Series(["A1", "B2", "C3"], dtype=any_string_dtype)
    
    # 使用正则表达式提取单个分组，没有匹配项
    result = s.str.extract("(_)", expand=False)
    # 创建一个预期的 Series，所有值为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    # 断言两个 Series 相等
    tm.assert_series_equal(result, expected)

    # 使用正则表达式提取两个分组，没有匹配项
    result = s.str.extract("(_)(_)", expand=False)
    # 创建一个预期的 DataFrame，所有值为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 使用正则表达式提取单个分组，有匹配项
    result = s.str.extract("([AB])[123]", expand=False)
    # 创建一个预期的 Series，包含提取到的匹配组结果，未匹配的位置为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    # 断言两个 Series 相等
    tm.assert_series_equal(result, expected)

    # 使用正则表达式提取两个分组，有匹配项
    result = s.str.extract("([AB])([123])", expand=False)
    # 创建一个预期的 DataFrame，包含提取到的匹配组结果，未匹配的位置为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 使用正则表达式提取一个命名组
    result = s.str.extract("(?P<letter>[AB])", expand=False)
    # 创建一个预期的 Series，包含提取到的命名组结果，未匹配的位置为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = Series(["A", "B", np.nan], name="letter", dtype=any_string_dtype)
    # 断言两个 Series 相等
    tm.assert_series_equal(result, expected)

    # 使用正则表达式提取两个命名组
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=False)
    # 创建一个预期的 DataFrame，包含提取到的命名组结果，未匹配的位置为 NaN，列名为指定的命名组名字，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 混合使用命名组和非命名组
    result = s.str.extract("([AB])(?P<number>[123])", expand=False)
    # 创建一个预期的 DataFrame，包含提取到的组结果，未匹配的位置为 NaN，列名为混合的组名和默认数字索引，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 使用一个普通组和一个非捕获组
    result = s.str.extract("([AB])(?:[123])", expand=False)
    # 创建一个预期的 Series，包含提取到的普通组结果，未匹配的位置为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    # 断言两个 Series 相等
    tm.assert_series_equal(result, expected)

    # 使用一个普通组和一个非捕获组，提取三个字符的输入
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    result = s.str.extract("([AB])([123])(?:[123])", expand=False)
    # 创建一个预期的 DataFrame，包含提取到的组结果，未匹配的位置为 NaN，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 使用一个可选组后跟一个普通组
    s = Series(["A1", "B2", "3"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=False)
    # 创建一个预期的 DataFrame，包含提取到的组结果，未匹配的位置为 NaN，列名为指定的命名组名字，数据类型与输入的 any_string_dtype 一致
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    # 断言两个 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 使用一个普通组后跟一个可选组
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=False)
    # 此处省略了最后一个测试用例的注释部分
    # 创建预期的 DataFrame 对象，包含三行数据，每行包括一个字母和一个数字（可能是字符串或 NaN）
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    # 使用测试工具（例如 pytest 的 tm.assert_frame_equal）比较 result 和 expected，确保它们相等
    tm.assert_frame_equal(result, expected)
def test_extract_expand_capture_groups_index(index, any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/6348
    # 提供了一个链接到该测试函数的 GitHub 问题页面的注释

    # 定义测试数据
    data = ["A1", "B2", "C"]

    # 如果传入的索引长度为0，则跳过测试
    if len(index) == 0:
        pytest.skip("Test requires len(index) > 0")

    # 当传入的索引长度小于数据长度时，通过重复索引使其长度至少与数据长度相同
    while len(index) < len(data):
        index = index.repeat(2)

    # 保证索引长度与数据长度相同
    index = index[: len(data)]

    # 创建一个 Series 对象，将数据与索引关联起来，并指定数据类型
    ser = Series(data, index=index, dtype=any_string_dtype)

    # 使用正则表达式从每个元素中提取单个数字，不扩展结果
    result = ser.str.extract(r"(\d)", expand=False)

    # 期望的结果是一个 Series 对象，其中每个元素是对应位置的数字或 NaN
    expected = Series(["1", "2", np.nan], index=index, dtype=any_string_dtype)

    # 断言提取结果与期望结果相等
    tm.assert_series_equal(result, expected)

    # 使用正则表达式从每个元素中提取字母和数字，不扩展结果
    result = ser.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=False)

    # 期望的结果是一个 DataFrame 对象，列名为 'letter' 和 'number'，索引与 ser 相同
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )

    # 断言提取结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_extract_single_series_name_is_preserved(any_string_dtype):
    # 创建一个带有指定名称的 Series 对象，并指定数据类型
    s = Series(["a3", "b3", "c2"], name="bob", dtype=any_string_dtype)

    # 使用正则表达式从每个元素中提取字母，并不扩展结果
    result = s.str.extract(r"(?P<sue>[a-z])", expand=False)

    # 期望的结果是一个 Series 对象，名称为 'sue'，包含每个元素提取的字母
    expected = Series(["a", "b", "c"], name="sue", dtype=any_string_dtype)

    # 断言提取结果与期望结果相等
    tm.assert_series_equal(result, expected)


def test_extract_expand_True(any_string_dtype):
    # 包含类似于 test_match 的测试，以及其他一些测试

    # 创建一个 Series 对象，包含指定的字符串数据，并指定数据类型
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)

    # 使用正则表达式从每个元素中提取以 'BAD__' 开头和以 'BAD' 结尾的字符串，并扩展结果为 DataFrame
    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=True)

    # 期望的结果是一个 DataFrame 对象，包含每个元素提取的结果或 NaN
    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )

    # 断言提取结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_mixed_object():
    # 创建一个包含混合类型数据的 Series 对象

    # 定义一个空的行
    er = [np.nan, np.nan]

    # 创建一个包含各种类型数据的 Series 对象
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

    # 使用正则表达式从每个元素中提取以 'BAD_' 开头和以 'BAD' 结尾的字符串，并扩展结果为 DataFrame
    result = mixed.str.extract(".*(BAD[_]+).*(BAD)", expand=True)

    # 期望的结果是一个 DataFrame 对象，包含每个元素提取的结果或空行
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )

    # 断言提取结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_single_capture_group_raises(
    index_or_series, any_string_dtype
):
    # 这些测试适用于 Series 和 Index

    # 创建一个 Series 或 Index 对象，包含指定的字符串数据，并指定数据类型
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)

    # 定义一个错误消息
    msg = "pattern contains no capture groups"

    # 使用正则表达式尝试从每个元素中提取匹配模式，期望引发 ValueError 异常，并包含指定的错误消息
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("[ABC][123]", expand=True)

    # 使用正则表达式尝试从每个元素中提取匹配模式，期望引发 ValueError 异常，并包含指定的错误消息
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("(?:[AB]).*", expand=True)


def test_extract_expand_True_single_capture_group(index_or_series, any_string_dtype):
    # 单个组会正确重命名 Series 或 Index

    # 创建一个 Series 或 Index 对象，包含指定的字符串数据，并指定数据类型
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)

    # 使用正则表达式从每个元素中提取以 'A' 开头的字符，并扩展结果为 DataFrame
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=True)
    # 创建一个预期的 DataFrame 对象，其中包含一个名为 "uno" 的列，列中包含两个字符串 "A" 和 "A"，数据类型为自定义的 any_string_dtype
    expected = DataFrame({"uno": ["A", "A"]}, dtype=any_string_dtype)
    
    # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("name", [None, "series_name"])
# 使用 pytest 的参数化装饰器，定义了一个测试函数，参数 name 分别取 None 和 "series_name"
def test_extract_series(name, any_string_dtype):
    # extract should give the same result whether or not the series has a name.
    # 创建一个 Series 对象 s，包含三个元素 "A1", "B2", "C3"，可选地指定名称为 name，数据类型为 any_string_dtype
    s = Series(["A1", "B2", "C3"], name=name, dtype=any_string_dtype)

    # one group, no matches
    # 对 s 应用正则表达式 "(_)" 进行提取，期望返回一个全是 NaN 的 DataFrame
    result = s.str.extract("(_)", expand=True)
    expected = DataFrame([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, no matches
    # 对 s 应用正则表达式 "(_)(_)" 进行提取，期望返回一个全是 NaN 的 DataFrame
    result = s.str.extract("(_)(_)", expand=True)
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one group, some matches
    # 对 s 应用正则表达式 "([AB])[123]" 进行提取，期望返回一个包含匹配结果的 DataFrame
    result = s.str.extract("([AB])[123]", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, some matches
    # 对 s 应用正则表达式 "([AB])([123])" 进行提取，期望返回一个包含匹配结果的 DataFrame
    result = s.str.extract("([AB])([123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named group
    # 对 s 应用正则表达式 "(?P<letter>[AB])" 进行提取，期望返回一个包含命名组 "letter" 的 DataFrame
    result = s.str.extract("(?P<letter>[AB])", expand=True)
    expected = DataFrame({"letter": ["A", "B", np.nan]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two named groups
    # 对 s 应用正则表达式 "(?P<letter>[AB])(?P<number>[123])" 进行提取，期望返回一个包含命名组 "letter" 和 "number" 的 DataFrame
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # mix named and unnamed groups
    # 对 s 应用正则表达式 "([AB])(?P<number>[123])" 进行提取，期望返回一个包含混合命名组和未命名组的 DataFrame
    result = s.str.extract("([AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group, one non-capturing group
    # 对 s 应用正则表达式 "([AB])(?:[123])" 进行提取，期望返回一个包含非捕获组的 DataFrame
    result = s.str.extract("([AB])(?:[123])", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


def test_extract_optional_groups(any_string_dtype):
    # two normal groups, one non-capturing group
    # 创建一个 Series 对象 s，包含三个元素 "A11", "B22", "C33"，数据类型为 any_string_dtype
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    # 对 s 应用正则表达式 "([AB])([123])(?:[123])" 进行提取，期望返回一个包含匹配结果的 DataFrame
    result = s.str.extract("([AB])([123])(?:[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one optional group followed by one normal group
    # 创建一个 Series 对象 s，包含三个元素 "A1", "B2", "3"，数据类型为 any_string_dtype
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group followed by one optional group
    # 对 s 应用正则表达式 "(?P<letter>[AB])?(?P<number>[123])" 进行提取，期望返回一个包含可选组的 DataFrame
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)
    # 创建一个 Pandas Series 对象 `s`，包含字符串元素 ["A1", "B2", "C"]，使用自定义的字符串类型 `any_string_dtype`
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    
    # 对 Series `s` 中的每个字符串应用正则表达式 "(?P<letter>[ABC])(?P<number>[123])?"，提取字母和数字，并扩展为 DataFrame
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=True)
    
    # 创建一个预期的 DataFrame `expected`，包含指定的数据 [["A", "1"], ["B", "2"], ["C", np.nan]]，
    # 列名分别为 "letter" 和 "number"，并且使用自定义的字符串类型 `any_string_dtype`
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    
    # 使用 Pandas 的 `tm.assert_frame_equal` 函数比较变量 `result` 和 `expected` 的内容是否相等
    tm.assert_frame_equal(result, expected)
def test_extract_dataframe_capture_groups_index(index, any_string_dtype):
    # GH6348
    # GH6348 issue: The test case is related to GitHub issue 6348.

    # Data for testing, a list of strings
    data = ["A1", "B2", "C"]

    # Check if the provided index has enough values for the data; if not, skip the test
    if len(index) < len(data):
        pytest.skip(f"Index needs more than {len(data)} values")

    # Trim the index to match the length of the data
    index = index[: len(data)]

    # Create a pandas Series with the data and specified index
    s = Series(data, index=index, dtype=any_string_dtype)

    # Apply regex pattern to extract single digits (\d) from each element of the Series
    result = s.str.extract(r"(\d)", expand=True)

    # Expected DataFrame with extracted digits
    expected = DataFrame(["1", "2", np.nan], index=index, dtype=any_string_dtype)

    # Assert that the result matches the expected DataFrame
    tm.assert_frame_equal(result, expected)

    # Apply regex pattern to extract a letter (\D) and an optional digit (\d)
    result = s.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=True)

    # Expected DataFrame with extracted letter and number (if present)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )

    # Assert that the result matches the expected DataFrame
    tm.assert_frame_equal(result, expected)


def test_extract_single_group_returns_frame(any_string_dtype):
    # GH11386
    # GH11386 issue: Ensure that extract always returns a DataFrame, even for a single group.

    # Create a pandas Series with strings
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)

    # Apply regex pattern to extract a single lowercase letter [a-z]
    result = s.str.extract(r"(?P<letter>[a-z])", expand=True)

    # Expected DataFrame with extracted letters
    expected = DataFrame({"letter": ["a", "b", "c"]}, dtype=any_string_dtype)

    # Assert that the result matches the expected DataFrame
    tm.assert_frame_equal(result, expected)


def test_extractall(any_string_dtype):
    # Test case for extractall method

    # Sample data containing email-like strings
    data = [
        "dave@google.com",
        "tdhock5@gmail.com",
        "maudelaperriere@gmail.com",
        "rob@gmail.com some text steve@gmail.com",
        "a@b.com some text c@d.com and e@f.com",
        np.nan,
        "",
    ]

    # Expected tuples after extraction of user, domain, and top-level domain (tld)
    expected_tuples = [
        ("dave", "google", "com"),
        ("tdhock5", "gmail", "com"),
        ("maudelaperriere", "gmail", "com"),
        ("rob", "gmail", "com"),
        ("steve", "gmail", "com"),
        ("a", "b", "com"),
        ("c", "d", "com"),
        ("e", "f", "com"),
    ]

    # Regular expression pattern for extracting email components
    pat = r"""
    (?P<user>[a-z0-9]+)
    @
    (?P<domain>[a-z]+)
    \.
    (?P<tld>[a-z]{2,4})
    """

    # Expected column names for the resulting DataFrame
    expected_columns = ["user", "domain", "tld"]

    # Create a pandas Series with the sample data
    s = Series(data, dtype=any_string_dtype)

    # Define expected index for the resulting DataFrame using MultiIndex
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)],
        names=(None, "match"),
    )

    # Expected DataFrame after applying extractall with the regex pattern
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )

    # Apply extractall to the Series with the defined pattern and flags
    result = s.str.extractall(pat, flags=re.VERBOSE)

    # Assert that the result matches the expected DataFrame
    tm.assert_frame_equal(result, expected)

    # Example of MultiIndex construction based on the input Series index
    mi = MultiIndex.from_tuples(
        [
            ("single", "Dave"),
            ("single", "Toby"),
            ("single", "Maude"),
            ("multiple", "robAndSteve"),
            ("multiple", "abcdef"),
            ("none", "missing"),
            ("none", "empty"),
        ]
    )
    # 使用给定的数据和索引创建一个 Series 对象，指定数据类型为 any_string_dtype
    s = Series(data, index=mi, dtype=any_string_dtype)
    # 创建预期的 MultiIndex 对象，包含特定的元组作为索引，其中的 "match" 是级别名称
    expected_index = MultiIndex.from_tuples(
        [
            ("single", "Dave", 0),
            ("single", "Toby", 0),
            ("single", "Maude", 0),
            ("multiple", "robAndSteve", 0),
            ("multiple", "robAndSteve", 1),
            ("multiple", "abcdef", 0),
            ("multiple", "abcdef", 1),
            ("multiple", "abcdef", 2),
        ],
        names=(None, None, "match"),  # 设置 MultiIndex 的级别名称
    )
    # 创建预期的 DataFrame 对象，使用给定的元组、索引和列名称，数据类型为 any_string_dtype
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    # 对 Series 中的字符串进行正则表达式提取，使用给定的模式和标志 re.VERBOSE
    result = s.str.extractall(pat, flags=re.VERBOSE)
    # 使用测试工具进行预期结果和实际结果的比较
    tm.assert_frame_equal(result, expected)

    # 使用相同的数据和索引创建另一个 Series 对象 s
    s = Series(data, index=mi, dtype=any_string_dtype)
    # 设置新的索引级别名称为 ("matches", "description")
    s.index.names = ("matches", "description")
    # 更新预期的索引级别名称，包括额外的 "match" 级别
    expected_index.names = ("matches", "description", "match")
    # 创建另一个预期的 DataFrame 对象，使用给定的元组、索引和列名称，数据类型为 any_string_dtype
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    # 再次对 Series 中的字符串进行正则表达式提取，使用给定的模式和标志 re.VERBOSE
    result = s.str.extractall(pat, flags=re.VERBOSE)
    # 使用测试工具再次比较预期结果和实际结果
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "pat,expected_names",
    [
        # 定义参数化测试的参数，正则表达式模式和期望的列名列表
        ("(?P<letter>[AB])?(?P<number>[123])", ["letter", "number"]),
        # 只有两个组中的一个有名称
        ("([AB])?(?P<number>[123])", [0, "number"]),
    ],
)
def test_extractall_column_names(pat, expected_names, any_string_dtype):
    # 创建一个 Series 对象，包含字符串数据和指定的数据类型
    s = Series(["", "A1", "32"], dtype=any_string_dtype)

    # 使用给定的正则表达式模式对 Series 进行 extractall 操作，返回结果 DataFrame
    result = s.str.extractall(pat)

    # 创建期望的 DataFrame 对象，包含预期的数据和索引
    expected = DataFrame(
        [("A", "1"), (np.nan, "3"), (np.nan, "2")],
        index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, "match")),
        columns=expected_names,
        dtype=any_string_dtype,
    )
    # 使用测试工具 tm.assert_frame_equal 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group(any_string_dtype):
    # 创建一个 Series 对象，包含字符串数据和指定的数据类型
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    
    # 创建预期的索引对象
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
    )

    # 使用指定的正则表达式模式对 Series 进行 extractall 操作，返回结果 DataFrame
    result = s.str.extractall(r"(?P<letter>[a-z])")
    
    # 创建期望的 DataFrame 对象，包含预期的数据和索引
    expected = DataFrame(
        {"letter": ["a", "b", "d", "c"]}, index=expected_index, dtype=any_string_dtype
    )
    # 使用测试工具 tm.assert_frame_equal 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)

    # 使用指定的正则表达式模式对 Series 进行 extractall 操作，返回结果 DataFrame
    result = s.str.extractall(r"([a-z])")
    
    # 创建期望的 DataFrame 对象，包含预期的数据和索引
    expected = DataFrame(
        ["a", "b", "d", "c"], index=expected_index, dtype=any_string_dtype
    )
    # 使用测试工具 tm.assert_frame_equal 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group_with_quantifier(any_string_dtype):
    # GH#13382
    # 使用指定的正则表达式模式对 Series 进行 extractall 操作，返回结果 DataFrame
    s = Series(["ab3", "abc3", "d4cd2"], name="series_name", dtype=any_string_dtype)
    result = s.str.extractall(r"([a-z]+)")
    
    # 创建期望的 DataFrame 对象，包含预期的数据和索引
    expected = DataFrame(
        ["ab", "abc", "d", "cd"],
        index=MultiIndex.from_tuples(
            [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
        ),
        dtype=any_string_dtype,
    )
    # 使用测试工具 tm.assert_frame_equal 检查结果与期望是否一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, names",
    [
        ([], (None,)),
        ([], ("i1",)),
        ([], (None, "i2")),
        ([], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None,)),
        (["a3", "b3", "d4c2"], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None, "i2")),
    ],
)
def test_extractall_no_matches(data, names, any_string_dtype):
    # GH19075 extractall with no matches should return a valid MultiIndex
    n = len(data)
    
    # 根据参数 names 创建适当的索引对象
    if len(names) == 1:
        index = Index(range(n), name=names[0])
    else:
        tuples = (tuple([i] * (n - 1)) for i in range(n))
        index = MultiIndex.from_tuples(tuples, names=names)
    
    # 创建包含指定数据和索引的 Series 对象
    s = Series(data, name="series_name", index=index, dtype=any_string_dtype)
    
    # 创建空的预期索引对象
    expected_index = MultiIndex.from_tuples([], names=(names + ("match",)))
    
    # 使用指定的正则表达式模式对 Series 进行 extractall 操作，返回结果 DataFrame
    result = s.str.extractall("(z)")
    # 创建一个空的 DataFrame 对象，具有指定的列和索引结构，使用指定的数据类型
    expected = DataFrame(columns=range(1), index=expected_index, dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期的 DataFrame 是否相等，包括列类型的检查
    tm.assert_frame_equal(result, expected, check_column_type=True)
    
    # 提取所有匹配 "(z)(z)" 的子串
    result = s.str.extractall("(z)(z)")
    # 创建一个空的 DataFrame 对象，具有指定的列和索引结构，使用指定的数据类型
    expected = DataFrame(columns=range(2), index=expected_index, dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期的 DataFrame 是否相等，包括列类型的检查
    tm.assert_frame_equal(result, expected, check_column_type=True)
    
    # 提取所有匹配 "(?P<first>z)" 的子串，其中 "(?P<first>z)" 是一个带有命名组 "first" 的正则表达式模式
    result = s.str.extractall("(?P<first>z)")
    # 创建一个 DataFrame 对象，具有指定的列和索引结构，其中列名为 "first"，使用指定的数据类型
    expected = DataFrame(columns=["first"], index=expected_index, dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
    
    # 提取所有匹配 "(?P<first>z)(?P<second>z)" 的子串，其中包含两个命名组 "first" 和 "second"
    result = s.str.extractall("(?P<first>z)(?P<second>z)")
    # 创建一个 DataFrame 对象，具有指定的列和索引结构，其中列名为 "first" 和 "second"，使用指定的数据类型
    expected = DataFrame(columns=["first", "second"], index=expected_index, dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
    
    # 提取所有匹配 "(z)(?P<second>z)" 的子串，其中包含一个未命名组和一个命名组 "second"
    result = s.str.extractall("(z)(?P<second>z)")
    # 创建一个 DataFrame 对象，具有指定的列和索引结构，其中一列未命名为 0，另一列命名为 "second"，使用指定的数据类型
    expected = DataFrame(columns=[0, "second"], index=expected_index, dtype=any_string_dtype)
    # 使用 pytest 的 assert_frame_equal 函数比较结果 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
def test_extractall_stringindex(any_string_dtype):
    # 创建一个 Series 对象，包含三个字符串，指定名称和数据类型
    s = Series(["a1a2", "b1", "c1"], name="xxx", dtype=any_string_dtype)
    
    # 使用正则表达式从每个字符串中提取匹配项，并创建一个多层次索引的 DataFrame
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    
    # 创建预期的 DataFrame 对象，包含提取的数字作为列，设置索引的多层次结构
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=[None, "match"]),
        dtype=any_string_dtype,
    )
    
    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)

    # 如果数据类型是 "object"，则测试不同的索引情况
    if any_string_dtype == "object":
        for idx in [
            Index(["a1a2", "b1", "c1"], dtype=object),
            Index(["a1a2", "b1", "c1"], name="xxx", dtype=object),
        ]:
            # 对不同的索引应用提取所有匹配项的操作，并检查结果与预期是否一致
            result = idx.str.extractall(r"[ab](?P<digit>\d)")
            tm.assert_frame_equal(result, expected)

    # 创建另一个 Series 对象，指定名称、索引和数据类型
    s = Series(
        ["a1a2", "b1", "c1"],
        name="s_name",
        index=Index(["XX", "yy", "zz"], name="idx_name"),
        dtype=any_string_dtype,
    )
    
    # 使用正则表达式从每个字符串中提取匹配项，并创建一个多层次索引的 DataFrame
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    
    # 创建预期的 DataFrame 对象，包含提取的数字作为列，设置索引的多层次结构
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples(
            [("XX", 0), ("XX", 1), ("yy", 0)], names=["idx_name", "match"]
        ),
        dtype=any_string_dtype,
    )
    
    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)


def test_extractall_no_capture_groups_raises(any_string_dtype):
    # 不合理的使用 extractall 函数，因为正则表达式没有捕获组
    # （它返回一个每个捕获组为一列的 DataFrame）
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    
    # 使用 pytest 框架断言应该引发 ValueError 异常，匹配错误信息为 "no capture groups"
    with pytest.raises(ValueError, match="no capture groups"):
        s.str.extractall(r"[a-z]")


def test_extract_index_one_two_groups():
    # 创建一个 Series 对象，包含三个字符串，并指定索引和名称
    s = Series(["a3", "b3", "d4c2"], index=["A3", "B3", "D4"], name="series_name")
    
    # 使用正则表达式从索引中提取第一个大写字母，并扩展为 DataFrame
    r = s.index.str.extract(r"([A-Z])", expand=True)
    
    # 创建预期的 DataFrame 对象，包含提取的大写字母作为列
    e = DataFrame(["A", "B", "D"])
    
    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(r, e)

    # 使用正则表达式从索引中提取大写字母和数字，并扩展为 DataFrame
    r = s.index.str.extract(r"(?P<letter>[A-Z])(?P<digit>[0-9])", expand=True)
    
    # 创建预期的 DataFrame 对象，包含提取的大写字母和数字作为列
    e_list = [("A", "3"), ("B", "3"), ("D", "4")]
    e = DataFrame(e_list, columns=["letter", "digit"])
    
    # 使用测试框架检查结果和预期是否相等
    tm.assert_frame_equal(r, e)


def test_extractall_same_as_extract(any_string_dtype):
    # 创建一个 Series 对象，包含三个字符串，并指定名称和数据类型
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)

    # 定义两个不同的正则表达式模式
    pattern_two_noname = r"([a-z])([0-9])"
    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    
    # 使用第一个模式提取结果，扩展为 DataFrame
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    
    # 使用第一个模式提取所有匹配项的结果，并创建具有多层次索引的 DataFrame
    has_multi_index = s.str.extractall(pattern_two_noname)
    
    # 从具有多层次索引的 DataFrame 中选择第一层索引为 0 的数据，形成单层索引的 DataFrame
    no_multi_index = has_multi_index.xs(0, level="match")
    
    # 使用测试框架检查提取结果和单层索引的 DataFrame 是否相等
    tm.assert_frame_equal(extract_two_noname, no_multi_index)

    # 使用第二个模式提取结果，扩展为 DataFrame
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    
    # 使用第二个模式提取所有匹配项的结果，并创建具有多层次索引的 DataFrame
    has_multi_index = s.str.extractall(pattern_two_named)
    # 使用 `has_multi_index` 数据框中的第一层索引为0的切片，赋给 `no_multi_index`
    no_multi_index = has_multi_index.xs(0, level="match")
    # 断言提取出的两个命名组与 `no_multi_index` 数据框相等
    tm.assert_frame_equal(extract_two_named, no_multi_index)

    # 定义一个命名捕获组的正则表达式模式，命名为 `group_name`
    pattern_one_named = r"(?P<group_name>[a-z])"
    # 对字符串列 `s` 应用命名捕获组的正则表达式，扩展为数据框
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    # 使用命名捕获组的正则表达式对 `s` 中所有匹配进行提取，并存储为 `has_multi_index` 数据框
    has_multi_index = s.str.extractall(pattern_one_named)
    # 使用 `has_multi_index` 数据框中的第一层索引为0的切片，赋给 `no_multi_index`
    no_multi_index = has_multi_index.xs(0, level="match")
    # 断言提取出的一个命名组与 `no_multi_index` 数据框相等
    tm.assert_frame_equal(extract_one_named, no_multi_index)

    # 定义一个非命名捕获组的正则表达式模式，提取字符范围 `[a-z]`
    pattern_one_noname = r"([a-z])"
    # 对字符串列 `s` 应用非命名捕获组的正则表达式，扩展为数据框
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    # 使用非命名捕获组的正则表达式对 `s` 中所有匹配进行提取，并存储为 `has_multi_index` 数据框
    has_multi_index = s.str.extractall(pattern_one_noname)
    # 使用 `has_multi_index` 数据框中的第一层索引为0的切片，赋给 `no_multi_index`
    no_multi_index = has_multi_index.xs(0, level="match")
    # 断言提取出的一个非命名组与 `no_multi_index` 数据框相等
    tm.assert_frame_equal(extract_one_noname, no_multi_index)
# 定义一个测试函数，用于验证 extractall 方法与 extract 方法在处理具有 MultiIndex 的 Series 时的一致性
def test_extractall_same_as_extract_subject_index(any_string_dtype):
    # 创建一个 MultiIndex，指定两个级别的名称
    mi = MultiIndex.from_tuples(
        [("A", "first"), ("B", "second"), ("C", "third")],
        names=("capital", "ordinal"),
    )
    # 创建一个带有 MultiIndex 的 Series 对象，其中每个元素是字符串，数据类型由参数 any_string_dtype 指定
    s = Series(["a3", "b3", "c2"], index=mi, name="series_name", dtype=any_string_dtype)

    # 定义一个匹配两个分组但未命名的正则表达式模式
    pattern_two_noname = r"([a-z])([0-9])"
    # 使用 extract 方法提取符合 pattern_two_noname 模式的内容，返回一个 DataFrame
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    # 使用 extractall 方法提取所有符合 pattern_two_noname 模式的内容，返回一个 MultiIndex DataFrame
    has_match_index = s.str.extractall(pattern_two_noname)
    # 从 extractall 返回的 MultiIndex DataFrame 中提取第一层索引为 0 的数据，形成一个 DataFrame
    no_match_index = has_match_index.xs(0, level="match")
    # 使用 assert_frame_equal 函数比较 extract_two_noname 和 no_match_index，确保它们相等
    tm.assert_frame_equal(extract_two_noname, no_match_index)

    # 定义一个匹配两个分组且命名的正则表达式模式
    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    # 使用 extract 方法提取符合 pattern_two_named 模式的内容，返回一个 DataFrame
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    # 使用 extractall 方法提取所有符合 pattern_two_named 模式的内容，返回一个 MultiIndex DataFrame
    has_match_index = s.str.extractall(pattern_two_named)
    # 从 extractall 返回的 MultiIndex DataFrame 中提取第一层索引为 0 的数据，形成一个 DataFrame
    no_match_index = has_match_index.xs(0, level="match")
    # 使用 assert_frame_equal 函数比较 extract_two_named 和 no_match_index，确保它们相等
    tm.assert_frame_equal(extract_two_named, no_match_index)

    # 定义一个匹配一个分组且命名的正则表达式模式
    pattern_one_named = r"(?P<group_name>[a-z])"
    # 使用 extract 方法提取符合 pattern_one_named 模式的内容，返回一个 DataFrame
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    # 使用 extractall 方法提取所有符合 pattern_one_named 模式的内容，返回一个 MultiIndex DataFrame
    has_match_index = s.str.extractall(pattern_one_named)
    # 从 extractall 返回的 MultiIndex DataFrame 中提取第一层索引为 0 的数据，形成一个 DataFrame
    no_match_index = has_match_index.xs(0, level="match")
    # 使用 assert_frame_equal 函数比较 extract_one_named 和 no_match_index，确保它们相等
    tm.assert_frame_equal(extract_one_named, no_match_index)

    # 定义一个匹配一个分组但未命名的正则表达式模式
    pattern_one_noname = r"([a-z])"
    # 使用 extract 方法提取符合 pattern_one_noname 模式的内容，返回一个 DataFrame
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    # 使用 extractall 方法提取所有符合 pattern_one_noname 模式的内容，返回一个 MultiIndex DataFrame
    has_match_index = s.str.extractall(pattern_one_noname)
    # 从 extractall 返回的 MultiIndex DataFrame 中提取第一层索引为 0 的数据，形成一个 DataFrame
    no_match_index = has_match_index.xs(0, level="match")
    # 使用 assert_frame_equal 函数比较 extract_one_noname 和 no_match_index，确保它们相等
    tm.assert_frame_equal(extract_one_noname, no_match_index)


# 定义一个测试函数，用于验证 extractall 方法在保留特定数据类型的 Series 列时的正确性
def test_extractall_preserves_dtype():
    # 导入 pytest 和 pyarrow 库，如果 pyarrow 未安装则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 创建一个字符串类型为 pyarrow 的 Series 对象，并使用 extractall 方法提取 "(ab)" 模式的内容
    result = Series(["abc", "ab"], dtype=ArrowDtype(pa.string())).str.extractall("(ab)")
    # 断言提取结果的第一列数据类型为 "string[pyarrow]"
    assert result.dtypes[0] == "string[pyarrow]"
```