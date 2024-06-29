# `D:\src\scipysrc\pandas\pandas\tests\strings\test_strings.py`

```
# 从 datetime 模块中导入 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库进行单元测试
import pytest

# 从 pandas 库中导入 DataFrame, Index, MultiIndex, Series 类
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
# 导入 pandas 内部测试模块
import pandas._testing as tm
# 从 pandas.core.strings.accessor 模块导入 StringMethods 类
from pandas.core.strings.accessor import StringMethods
# 从 pandas.tests.strings 模块导入 object_pyarrow_numpy 对象
from pandas.tests.strings import object_pyarrow_numpy


# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同的 pattern 值
@pytest.mark.parametrize("pattern", [0, True, Series(["foo", "bar"])])
def test_startswith_endswith_non_str_patterns(pattern):
    # GH3485
    # 创建 Series 对象
    ser = Series(["foo", "bar"])
    # 构建错误消息，指出期望的参数类型
    msg = f"expected a string or tuple, not {type(pattern).__name__}"
    # 使用 pytest 断言异常，期望抛出 TypeError 异常并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)


def test_iter_raises():
    # GH 54173
    # 创建 Series 对象
    ser = Series(["foo", "bar"])
    # 使用 pytest 断言异常，期望抛出 TypeError 异常并匹配特定错误消息
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)


# 定义测试函数 test_count，测试 Series 对象的 str.count 方法
def test_count(any_string_dtype):
    # 创建 Series 对象，指定数据类型为 any_string_dtype
    ser = Series(["foo", "foofoo", np.nan, "foooofooofommmfoo"], dtype=any_string_dtype)
    # 调用 Series 的 str.count 方法，统计匹配模式 "f[o]+" 的次数
    result = ser.str.count("f[o]+")
    # 根据数据类型 any_string_dtype 的不同，选择期望的数据类型
    expected_dtype = np.float64 if any_string_dtype in object_pyarrow_numpy else "Int64"
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_count_mixed_object，测试混合类型数据的 str.count 方法
def test_count_mixed_object():
    # 创建包含不同类型数据的 Series 对象
    ser = Series(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    # 调用 Series 的 str.count 方法，统计匹配模式 "a" 的次数
    result = ser.str.count("a")
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_repeat，测试 Series 对象的 str.repeat 方法
def test_repeat(any_string_dtype):
    # 创建 Series 对象，指定数据类型为 any_string_dtype
    ser = Series(["a", "b", np.nan, "c", np.nan, "d"], dtype=any_string_dtype)

    # 调用 Series 的 str.repeat 方法，重复每个元素 3 次
    result = ser.str.repeat(3)
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series(
        ["aaa", "bbb", np.nan, "ccc", np.nan, "ddd"], dtype=any_string_dtype
    )
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)

    # 调用 Series 的 str.repeat 方法，根据指定列表重复每个元素
    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series(
        ["a", "bb", np.nan, "cccc", np.nan, "dddddd"], dtype=any_string_dtype
    )
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_repeat_mixed_object，测试混合类型数据的 str.repeat 方法
def test_repeat_mixed_object():
    # 创建包含不同类型数据的 Series 对象
    ser = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])
    # 调用 Series 的 str.repeat 方法，重复每个元素 3 次
    result = ser.str.repeat(3)
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series(
        ["aaa", np.nan, "bbb", np.nan, np.nan, "foofoofoo", None, np.nan, np.nan],
        dtype=object,
    )
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试不同的 arg 和 repeat 值
@pytest.mark.parametrize("arg, repeat", [[None, 4], ["b", None]])
def test_repeat_with_null(any_string_dtype, arg, repeat):
    # GH: 31632
    # 创建 Series 对象，指定数据类型为 any_string_dtype
    ser = Series(["a", arg], dtype=any_string_dtype)
    # 调用 Series 的 str.repeat 方法，根据指定列表重复每个元素
    result = ser.str.repeat([3, repeat])
    # 创建期望的 Series 对象，与计算结果进行比较
    expected = Series(["aaa", None], dtype=any_string_dtype)
    # 使用 pandas 提供的测试工具进行结果比较
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_empty_str_methods，测试空 Series 对象的 str 方法
def test_empty_str_methods(any_string_dtype):
    # 创建空的 Series 对象，指定数据类型为 any_string_dtype
    empty_str = empty = Series(dtype=any_string_dtype)
    if any_string_dtype in object_pyarrow_numpy:
        # 如果 any_string_dtype 存在于 object_pyarrow_numpy 中
        empty_int = Series(dtype="int64")
        # 创建一个空的整数 Series
        empty_bool = Series(dtype=bool)
        # 创建一个空的布尔 Series
    else:
        # 否则
        empty_int = Series(dtype="Int64")
        # 创建一个空的可空整数 Series
        empty_bool = Series(dtype="boolean")
        # 创建一个空的布尔 Series
    empty_object = Series(dtype=object)
    # 创建一个空的对象 Series
    empty_bytes = Series(dtype=object)
    # 创建一个空的字节 Series
    empty_df = DataFrame()
    # 创建一个空的 DataFrame

    # GH7241
    # 在空 Series 上执行字符串方法

    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    # 断言空 Series 经过 str.cat 方法后与 empty_str 相等
    assert "" == empty.str.cat()
    # 断言空 Series 调用 str.cat() 方法后返回空字符串
    tm.assert_series_equal(empty_str, empty.str.title())
    # 断言空 Series 经过 str.title 方法后与 empty_str 相等
    tm.assert_series_equal(empty_int, empty.str.count("a"))
    # 断言空 Series 经过 str.count 方法后与 empty_int 相等
    tm.assert_series_equal(empty_bool, empty.str.contains("a"))
    # 断言空 Series 经过 str.contains 方法后与 empty_bool 相等
    tm.assert_series_equal(empty_bool, empty.str.startswith("a"))
    # 断言空 Series 经过 str.startswith 方法后与 empty_bool 相等
    tm.assert_series_equal(empty_bool, empty.str.endswith("a"))
    # 断言空 Series 经过 str.endswith 方法后与 empty_bool 相等
    tm.assert_series_equal(empty_str, empty.str.lower())
    # 断言空 Series 经过 str.lower 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.upper())
    # 断言空 Series 经过 str.upper 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.replace("a", "b"))
    # 断言空 Series 经过 str.replace 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    # 断言空 Series 经过 str.repeat 方法后与 empty_str 相等
    tm.assert_series_equal(empty_bool, empty.str.match("^a"))
    # 断言空 Series 经过 str.match 方法后与 empty_bool 相等
    tm.assert_frame_equal(
        DataFrame(columns=[0], dtype=any_string_dtype),
        empty.str.extract("()", expand=True),
    )
    # 断言空 Series 经过 str.extract 方法后返回的 DataFrame 结果与指定的 DataFrame 相等
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=True),
    )
    # 断言空 Series 经过 str.extract 方法后返回的 DataFrame 结果与指定的 DataFrame 相等
    tm.assert_series_equal(empty_str, empty.str.extract("()", expand=False))
    # 断言空 Series 经过 str.extract 方法后与 empty_str 相等
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=False),
    )
    # 断言空 Series 经过 str.extract 方法后返回的 DataFrame 结果与指定的 DataFrame 相等
    tm.assert_frame_equal(empty_df.set_axis([], axis=1), empty.str.get_dummies())
    # 断言空 Series 经过 str.get_dummies 方法后返回的 DataFrame 结果与空 DataFrame 相等
    tm.assert_series_equal(empty_str, empty_str.str.join(""))
    # 断言空 Series 经过 str.join 方法后与 empty_str 相等
    tm.assert_series_equal(empty_int, empty.str.len())
    # 断言空 Series 经过 str.len 方法后与 empty_int 相等
    tm.assert_series_equal(empty_object, empty_str.str.findall("a"))
    # 断言空 Series 经过 str.findall 方法后与 empty_object 相等
    tm.assert_series_equal(empty_int, empty.str.find("a"))
    # 断言空 Series 经过 str.find 方法后与 empty_int 相等
    tm.assert_series_equal(empty_int, empty.str.rfind("a"))
    # 断言空 Series 经过 str.rfind 方法后与 empty_int 相等
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    # 断言空 Series 经过 str.pad 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.center(42))
    # 断言空 Series 经过 str.center 方法后与 empty_str 相等
    tm.assert_series_equal(empty_object, empty.str.split("a"))
    # 断言空 Series 经过 str.split 方法后与 empty_object 相等
    tm.assert_series_equal(empty_object, empty.str.rsplit("a"))
    # 断言空 Series 经过 str.rsplit 方法后与 empty_object 相等
    tm.assert_series_equal(empty_object, empty.str.partition("a", expand=False))
    # 断言空 Series 经过 str.partition 方法后与 empty_object 相等
    tm.assert_frame_equal(empty_df, empty.str.partition("a"))
    # 断言空 Series 经过 str.partition 方法后返回的 DataFrame 结果与空 DataFrame 相等
    tm.assert_series_equal(empty_object, empty.str.rpartition("a", expand=False))
    # 断言空 Series 经过 str.rpartition 方法后与 empty_object 相等
    tm.assert_frame_equal(empty_df, empty.str.rpartition("a"))
    # 断言空 Series 经过 str.rpartition 方法后返回的 DataFrame 结果与空 DataFrame 相等
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    # 断言空 Series 经过 str.slice 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    # 断言空 Series 经过 str.slice 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.strip())
    # 断言空 Series 经过 str.strip 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    # 断言空 Series 经过 str.lstrip 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    # 断言空 Series 经过 str.rstrip 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    # 断言空 Series 经过 str.wrap 方法后与 empty_str 相等
    tm.assert_series_equal(empty_str, empty.str.get(0))
    # 断言空 Series 经过 str.get 方法后与 empty_str 相等
    tm.assert_series_equal(empty_object, empty_bytes.str.decode("ascii"))
    # 断言空 Series 经过 str.decode 方法后与 empty_object 相等
    # 使用 pandas 测试框架中的 assert_series_equal 函数，比较两个 Series 是否相等
    tm.assert_series_equal(empty_bytes, empty.str.encode("ascii"))
    
    # 使用 pandas 测试框架中的 assert_series_equal 函数，验证返回的 Series 是否始终为布尔值类型
    # GH 29624 是该问题的 GitHub Issue 编号
    tm.assert_series_equal(empty_bool, empty.str.isalnum())
    tm.assert_series_equal(empty_bool, empty.str.isalpha())
    tm.assert_series_equal(empty_bool, empty.str.isdigit())
    tm.assert_series_equal(empty_bool, empty.str.isspace())
    tm.assert_series_equal(empty_bool, empty.str.islower())
    tm.assert_series_equal(empty_bool, empty.str.isupper())
    tm.assert_series_equal(empty_bool, empty.str.istitle())
    tm.assert_series_equal(empty_bool, empty.str.isnumeric())
    tm.assert_series_equal(empty_bool, empty.str.isdecimal())
    
    # 使用 pandas 测试框架中的 assert_series_equal 函数，比较两个 Series 是否相等
    tm.assert_series_equal(empty_str, empty.str.capitalize())
    tm.assert_series_equal(empty_str, empty.str.swapcase())
    tm.assert_series_equal(empty_str, empty.str.normalize("NFC"))
    
    # 使用 str 类的 maketrans 方法创建转换表，将 'a' 替换为 'b'
    table = str.maketrans("a", "b")
    # 使用 pandas 测试框架中的 assert_series_equal 函数，比较两个 Series 是否相等
    tm.assert_series_equal(empty_str, empty.str.translate(table))
# 使用 pytest 模块的 mark.parametrize 装饰器，定义参数化测试用例
@pytest.mark.parametrize(
    "method, expected",
    [
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        (
            "isdigit",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isnumeric",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isspace",
            [False, False, False, False, False, False, False, False, False, True],
        ),
        (
            "islower",
            [False, True, False, False, False, False, False, False, False, False],
        ),
        (
            "isupper",
            [True, False, False, False, True, False, True, False, False, False],
        ),
        (
            "istitle",
            [True, False, True, False, True, False, False, False, False, False],
        ),
    ],
)
def test_ismethods(method, expected, any_string_dtype):
    # 创建包含不同类型字符串的 Series 对象
    ser = Series(
        ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "], dtype=any_string_dtype
    )
    # 根据数据类型选择预期的结果类型
    expected_dtype = "bool" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期结果的 Series 对象
    expected = Series(expected, dtype=expected_dtype)
    # 调用 Series 对象的字符串方法，并比较结果是否符合预期
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # 使用标准库方法获取预期结果，并与测试结果进行比较
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method, expected, any_string_dtype):
    # 创建包含不同 Unicode 字符串的 Series 对象
    ser = Series(
        ["A", "3", "¼", "★", "፸", "３", "four"],  # noqa: RUF001
        dtype=any_string_dtype,
    )
    # 根据数据类型选择预期的结果类型
    expected_dtype = "bool" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期结果的 Series 对象
    expected = Series(expected, dtype=expected_dtype)
    # 调用 Series 对象的字符串方法，并比较结果是否符合预期
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # 使用标准库方法获取预期结果，并与测试结果进行比较
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method, expected, any_string_dtype):
    # 创建包含 NaN 和 Unicode 字符串的 Series 对象
    values = ["A", np.nan, "¼", "★", np.nan, "３", "four"]  # noqa: RUF001
    ser = Series(values, dtype=any_string_dtype)
    # 根据数据类型选择预期的结果类型
    expected_dtype = "object" if any_string_dtype in object_pyarrow_numpy else "boolean"
    # 创建预期结果的 Series 对象
    expected = Series(expected, dtype=expected_dtype)
    # 使用 getattr 函数获取 ser.str 对象中指定方法（method）的结果
    result = getattr(ser.str, method)()
    # 使用 tm.assert_series_equal 函数比较 result 和 expected 两个序列是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试字符串序列的分割和连接操作的往返性
def test_spilt_join_roundtrip(any_string_dtype):
    # 创建一个包含字符串和 NaN 值的序列，指定数据类型为 any_string_dtype
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    # 对序列中的每个字符串进行以 "_" 分割后再以 "_" 连接的操作
    result = ser.str.split("_").str.join("_")
    # 期望的结果是将序列转换为 object 类型
    expected = ser.astype(object)
    # 使用测试工具库 tm 来比较结果和期望的序列是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试混合对象类型的字符串序列的分割和连接操作的往返性
def test_spilt_join_roundtrip_mixed_object():
    # 创建一个包含多种类型数据的序列，包括字符串、NaN、布尔、日期等
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    # 对序列中的每个字符串进行以 "_" 分割后再以 "_" 连接的操作
    result = ser.str.split("_").str.join("_")
    # 期望的结果是一个 object 类型的序列，其中非字符串部分用 NaN 填充
    expected = Series(
        ["a_b", np.nan, "asdf_cas_asdf", np.nan, np.nan, "foo", None, np.nan, np.nan],
        dtype=object,
    )
    # 使用测试工具库 tm 来比较结果和期望的序列是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试字符串序列中每个字符串的长度计算
def test_len(any_string_dtype):
    # 创建一个包含不同长度字符串和 NaN 的序列，指定数据类型为 any_string_dtype
    ser = Series(
        ["foo", "fooo", "fooooo", np.nan, "fooooooo", "foo\n", "あ"],
        dtype=any_string_dtype,
    )
    # 计算序列中每个字符串的长度
    result = ser.str.len()
    # 根据数据类型 any_string_dtype 确定期望结果的数据类型
    expected_dtype = "float64" if any_string_dtype in object_pyarrow_numpy else "Int64"
    # 期望的结果是一个包含每个字符串长度的 Series
    expected = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    # 使用测试工具库 tm 来比较结果和期望的序列是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试混合对象类型的字符串序列中每个字符串的长度计算
def test_len_mixed():
    # 创建一个包含多种类型数据的序列，包括字符串、NaN、布尔、日期等
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    # 计算序列中每个字符串的长度
    result = ser.str.len()
    # 期望的结果是一个包含每个字符串长度的 Series，非字符串部分用 NaN 填充
    expected = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    # 使用测试工具库 tm 来比较结果和期望的序列是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化测试功能，定义一个测试函数，测试字符串序列中查找子串的位置方法
@pytest.mark.parametrize(
    "method,sub,start,end,expected",
    [
        ("index", "EF", None, None, [4, 3, 1, 0]),
        ("rindex", "EF", None, None, [4, 5, 7, 4]),
        ("index", "EF", 3, None, [4, 3, 7, 4]),
        ("rindex", "EF", 3, None, [4, 5, 7, 4]),
        ("index", "E", 4, 8, [4, 5, 7, 4]),
        ("rindex", "E", 0, 5, [4, 3, 1, 4]),
    ],
)
def test_index(method, sub, start, end, index_or_series, any_string_dtype, expected):
    # 创建一个包含多个字符串的序列，指定数据类型为 any_string_dtype
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    # 根据数据类型 any_string_dtype 确定期望结果的数据类型
    expected_dtype = np.int64 if any_string_dtype in object_pyarrow_numpy else "Int64"
    # 创建期望的结果序列，其类型与 obj 的元素类型一致
    expected = index_or_series(expected, dtype=expected_dtype)

    # 调用对象的字符串方法 method（如 index 或 rindex）查找子串的位置
    result = getattr(obj.str, method)(sub, start, end)

    # 如果 index_or_series 是 Series 类型，则使用 tm.assert_series_equal 比较结果和期望的序列是否相等
    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        # 否则，使用 tm.assert_index_equal 比较结果和期望的索引是否相等
        tm.assert_index_equal(result, expected)

    # 使用标准库的相应方法，获取每个元素的子串位置并与结果比较
    expected = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected


# 定义一个测试函数，测试字符串序列中查找指定子串失败时是否抛出 ValueError 异常
def test_index_not_found_raises(index_or_series, any_string_dtype):
    # 创建一个空的字符串序列，指定数据类型为 any_string_dtype
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    # 测试查找一个不存在的子串是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match="substring not found"):
        obj.str.index("DE")


# 使用 pytest 的参数化测试功能，定义一个测试函数，测试在非字符串对象上调用查找子串方法是否会抛出 TypeError 异常
@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_wrong_type_raises(index_or_series, any_string_dtype, method):
    # 创建一个空的字符串序列，指定数据类型为 any_string_dtype
    obj = index_or_series([], dtype=any_string_dtype)
    msg = "expected a string object, not int"

    # 测试在空序列上调用查找子串方法是否会抛出 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)
    "method, exp",
    # 定义一个包含方法名和预期结果的列表，每个元素是一个包含两个元素的子列表
    [
        # 第一个子列表，方法名为 'index'，预期结果是列表 [1, 1, 0]
        ["index", [1, 1, 0]],
        # 第二个子列表，方法名为 'rindex'，预期结果是列表 [3, 1, 2]
        ["rindex", [3, 1, 2]],
    ],
def test_index_missing(any_string_dtype, method, exp):
    # 创建一个包含字符串和 NaN 值的 Series，指定字符串类型为 any_string_dtype
    ser = Series(["abcb", "ab", "bcbe", np.nan], dtype=any_string_dtype)
    # 根据条件选择期望的数据类型
    expected_dtype = np.float64 if any_string_dtype in object_pyarrow_numpy else "Int64"

    # 调用 Series 对象的 str 属性下的指定方法，返回结果 Series 对象
    result = getattr(ser.str, method)("b")
    # 创建期望的结果 Series 对象
    expected = Series(exp + [np.nan], dtype=expected_dtype)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_pipe_failures(any_string_dtype):
    # #2119
    # 创建一个包含字符串的 Series，指定字符串类型为 any_string_dtype
    ser = Series(["A|B|C"], dtype=any_string_dtype)

    # 调用 Series 对象的 str.split 方法，按 "|" 分割字符串并返回结果 Series 对象
    result = ser.str.split("|")
    # 创建期望的结果 Series 对象
    expected = Series([["A", "B", "C"]], dtype=object)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 调用 Series 对象的 str.replace 方法，将 "|" 替换为 " "，返回结果 Series 对象
    result = ser.str.replace("|", " ", regex=False)
    # 创建期望的结果 Series 对象
    expected = Series(["A B C"], dtype=any_string_dtype)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", "bar", np.nan, "baz"]),
        (0, 3, -1, ["", "", np.nan, ""]),
        (None, None, -1, ["owtoofaa", "owtrabaa", np.nan, "xuqzabaa"]),
        (3, 10, 2, ["oto", "ato", np.nan, "aqx"]),
        (3, 0, -1, ["ofa", "aba", np.nan, "aba"]),
    ],
)
def test_slice(start, stop, step, expected, any_string_dtype):
    # 创建一个包含字符串和 NaN 值的 Series，指定字符串类型为 any_string_dtype
    ser = Series(["aafootwo", "aabartwo", np.nan, "aabazqux"], dtype=any_string_dtype)
    # 调用 Series 对象的 str.slice 方法，进行切片操作，返回结果 Series 对象
    result = ser.str.slice(start, stop, step)
    # 创建期望的结果 Series 对象
    expected = Series(expected, dtype=any_string_dtype)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", np.nan, "bar", np.nan, np.nan, None, np.nan, np.nan]),
        (4, 1, -1, ["oof", np.nan, "rab", np.nan, np.nan, None, np.nan, np.nan]),
    ],
)
def test_slice_mixed_object(start, stop, step, expected):
    # 创建一个包含不同类型数据的 Series
    ser = Series(["aafootwo", np.nan, "aabartwo", True, datetime.today(), None, 1, 2.0])
    # 调用 Series 对象的 str.slice 方法，进行切片操作，返回结果 Series 对象
    result = ser.str.slice(start, stop, step)
    # 创建期望的结果 Series 对象
    expected = Series(expected, dtype=object)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,stop,repl,expected",
    [
        (2, 3, None, ["shrt", "a it longer", "evnlongerthanthat", "", np.nan]),
        (2, 3, "z", ["shzrt", "a zit longer", "evznlongerthanthat", "z", np.nan]),
        (2, 2, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (2, 1, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (-1, None, "z", ["shorz", "a bit longez", "evenlongerthanthaz", "z", np.nan]),
        (None, -2, "z", ["zrt", "zer", "zat", "z", np.nan]),
        (6, 8, "z", ["shortz", "a bit znger", "evenlozerthanthat", "z", np.nan]),
        (-10, 3, "z", ["zrt", "a zit longer", "evenlongzerthanthat", "z", np.nan]),
    ],
)
def test_slice_replace(start, stop, repl, expected, any_string_dtype):
    # 创建一个包含字符串和 NaN 值的 Series，指定字符串类型为 any_string_dtype
    ser = Series(
        ["short", "a bit longer", "evenlongerthanthat", "", np.nan],
        dtype=any_string_dtype,
    )
    # 创建期望的结果 Series 对象
    expected = Series(expected, dtype=any_string_dtype)
    # 调用 Series 对象的 str.slice_replace 方法，进行切片替换操作，返回结果 Series 对象
    result = ser.str.slice_replace(start, stop, repl)
    # 使用测试框架中的 assert_series_equal 函数比较 result 和 expected 两个对象的内容是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", "bb", np.nan, "cc"]],
        ["lstrip", ["aa   ", "bb \n", np.nan, "cc  "]],
        ["rstrip", ["  aa", " bb", np.nan, "cc"]],
    ],
)
def test_strip_lstrip_rstrip(any_string_dtype, method, exp):
    # 创建一个包含测试参数的参数化测试函数，用于测试字符串处理方法 strip, lstrip, rstrip
    ser = Series(["  aa   ", " bb \n", np.nan, "cc  "], dtype=any_string_dtype)

    # 调用字符串序列的对应方法（strip, lstrip, rstrip）进行处理
    result = getattr(ser.str, method)()
    # 创建预期的处理结果的 Series 对象
    expected = Series(exp, dtype=any_string_dtype)
    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", np.nan, "bb"]],
        ["lstrip", ["aa  ", np.nan, "bb \t\n"]],
        ["rstrip", ["  aa", np.nan, " bb"]],
    ],
)
def test_strip_lstrip_rstrip_mixed_object(method, exp):
    # 创建一个包含测试参数的参数化测试函数，用于测试混合对象的字符串处理方法 strip, lstrip, rstrip
    ser = Series(["  aa  ", np.nan, " bb \t\n", True, datetime.today(), None, 1, 2.0])

    # 调用字符串序列的对应方法（strip, lstrip, rstrip）进行处理
    result = getattr(ser.str, method)()
    # 创建预期的处理结果的 Series 对象
    expected = Series(exp + [np.nan, np.nan, None, np.nan, np.nan], dtype=object)
    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["ABC", " BNSD", "LDFJH "]],
        ["lstrip", ["ABCxx", " BNSD", "LDFJH xx"]],
        ["rstrip", ["xxABC", "xx BNSD", "LDFJH "]],
    ],
)
def test_strip_lstrip_rstrip_args(any_string_dtype, method, exp):
    # 创建一个包含测试参数的参数化测试函数，用于测试带参数的字符串处理方法 strip, lstrip, rstrip
    ser = Series(["xxABCxx", "xx BNSD", "LDFJH xx"], dtype=any_string_dtype)

    # 调用字符串序列的对应方法（strip, lstrip, rstrip）进行处理，传入额外的参数 "x"
    result = getattr(ser.str, method)("x")
    # 创建预期的处理结果的 Series 对象
    expected = Series(exp, dtype=any_string_dtype)
    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "prefix, expected", [("a", ["b", " b c", "bc"]), ("ab", ["", "a b c", "bc"])]
)
def test_removeprefix(any_string_dtype, prefix, expected):
    # 创建一个包含测试参数的参数化测试函数，用于测试 removeprefix 方法
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    # 调用字符串序列的 removeprefix 方法进行处理
    result = ser.str.removeprefix(prefix)
    # 创建预期的处理结果的 Series 对象
    ser_expected = Series(expected, dtype=any_string_dtype)
    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_series_equal(result, ser_expected)


@pytest.mark.parametrize(
    "suffix, expected", [("c", ["ab", "a b ", "b"]), ("bc", ["ab", "a b c", ""])]
)
def test_removesuffix(any_string_dtype, suffix, expected):
    # 创建一个包含测试参数的参数化测试函数，用于测试 removesuffix 方法
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    # 调用字符串序列的 removesuffix 方法进行处理
    result = ser.str.removesuffix(suffix)
    # 创建预期的处理结果的 Series 对象
    ser_expected = Series(expected, dtype=any_string_dtype)
    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_series_equal(result, ser_expected)


def test_string_slice_get_syntax(any_string_dtype):
    # 创建一个测试函数，测试字符串序列的切片操作
    ser = Series(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", np.nan, "CYYYBYYY", "dog", "cYYYt"],
        dtype=any_string_dtype,
    )

    # 测试获取单个字符的切片操作
    result = ser.str[0]
    expected = ser.str.get(0)
    tm.assert_series_equal(result, expected)

    # 测试获取前三个字符的切片操作
    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)

    # 测试获取倒序从第三个字符开始的切片操作
    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds_nested():
    # 创建一个测试函数，测试嵌套序列的字符串切片操作
    ser = Series([(1, 2), (1,), (3, 4, 5)])
    # 测试获取第二个元素的切片操作
    result = ser.str[1]
    expected = Series([2, np.nan, 4])
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试字符串切片是否超出边界
def test_string_slice_out_of_bounds(any_string_dtype):
    # 创建一个字符串序列对象，包含三个元素："foo", "b", "ba"，数据类型由参数指定
    ser = Series(["foo", "b", "ba"], dtype=any_string_dtype)
    # 对字符串序列进行字符串切片操作，获取每个字符串的第二个字符
    result = ser.str[1]
    # 期望结果是包含三个元素："o", np.nan, "a" 的字符串序列，数据类型与参数指定相同
    expected = Series(["o", np.nan, "a"], dtype=any_string_dtype)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试字符串编码和解码操作
def test_encode_decode(any_string_dtype):
    # 创建一个字符串序列对象，包含三个元素："a", "b", "a\xe4"，数据类型由参数指定，然后对每个字符串进行 UTF-8 编码
    ser = Series(["a", "b", "a\xe4"], dtype=any_string_dtype).str.encode("utf-8")
    # 对编码后的字符串序列进行 UTF-8 解码
    result = ser.str.decode("utf-8")
    # 使用 map 函数对原始序列进行 UTF-8 解码，并将结果转换为 object 类型的序列
    expected = ser.map(lambda x: x.decode("utf-8")).astype(object)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试字符串编码时的错误处理
def test_encode_errors_kwarg(any_string_dtype):
    # 创建一个字符串序列对象，包含三个元素："a", "b", "a\x9d"，数据类型由参数指定
    ser = Series(["a", "b", "a\x9d"], dtype=any_string_dtype)

    # 定义一个错误信息，用于断言是否会引发 UnicodeEncodeError 异常
    msg = (
        r"'charmap' codec can't encode character '\\x9d' in position 1: "
        "character maps to <undefined>"
    )
    # 使用 pytest 的异常断言，验证是否会引发指定异常并包含特定错误信息
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode("cp1252")

    # 对字符串序列进行 cp1252 编码，使用 'ignore' 参数忽略错误字符
    result = ser.str.encode("cp1252", "ignore")
    # 使用 map 函数对原始序列进行 cp1252 编码，并使用 'ignore' 参数
    expected = ser.map(lambda x: x.encode("cp1252", "ignore"))
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试字符串解码时的错误处理
def test_decode_errors_kwarg():
    # 创建一个字节序列索引对象，包含三个元素：b"a", b"b", b"a\x9d"
    ser = Series([b"a", b"b", b"a\x9d"])

    # 定义一个错误信息，用于断言是否会引发 UnicodeDecodeError 异常
    msg = (
        "'charmap' codec can't decode byte 0x9d in position 1: "
        "character maps to <undefined>"
    )
    # 使用 pytest 的异常断言，验证是否会引发指定异常并包含特定错误信息
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode("cp1252")

    # 对字节序列索引对象进行 cp1252 解码，使用 'ignore' 参数忽略错误字节
    result = ser.str.decode("cp1252", "ignore")
    # 使用 map 函数对原始序列进行 cp1252 解码，并使用 'ignore' 参数
    expected = ser.map(lambda x: x.decode("cp1252", "ignore")).astype(object)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化装饰器，定义一个测试函数，用于测试字符串规范化操作
@pytest.mark.parametrize(
    "form, expected",
    [
        ("NFKC", ["ABC", "ABC", "123", np.nan, "アイエ"]),
        ("NFC", ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]),  # noqa: RUF001
    ],
)
def test_normalize(form, expected, any_string_dtype):
    # 创建一个字符串序列对象，包含五个元素：["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]
    # 指定索引和数据类型由参数提供
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    # 创建期望的字符串序列对象，内容为 expected 列表，索引和数据类型与 ser 相同
    expected = Series(expected, index=["a", "b", "c", "d", "e"], dtype=any_string_dtype)
    # 对字符串序列进行指定规范化（NFKC 或 NFC）
    result = ser.str.normalize(form)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试当传递无效规范化形式参数时是否引发 ValueError
def test_normalize_bad_arg_raises(any_string_dtype):
    # 创建一个字符串序列对象，包含五个元素：["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]
    # 指定索引和数据类型由参数提供
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    # 使用 pytest 的异常断言，验证是否会引发 ValueError 异常并包含特定错误信息
    with pytest.raises(ValueError, match="invalid normalization form"):
        ser.str.normalize("xxx")


# 定义一个测试函数，用于测试索引对象的字符串规范化操作
def test_normalize_index():
    # 创建一个索引对象，包含三个元素："ＡＢＣ", "１２３", "ｱｲｴ"
    idx = Index(["ＡＢＣ", "１２３", "ｱｲｴ"])  # noqa: RUF001
    # 创建期望的索引对象，内容为 ["ABC", "123", "アイエ"]
    expected = Index(["ABC", "123", "アイエ"])
    # 对索引对象进行 NFKC 规范化
    result = idx.str.normalize("NFKC")
    # 断言结果索引与期望索引相等
    tm.assert_index_equal(result, expected)


# 使用 pytest 的参数化装饰器，定义一个测试函数，用于测试索引或系列对象的字符串访问器是否正确工作
@pytest.mark.parametrize(
    "values,inferred_type",
    [
        (["a", "b"], "string"),
        (["a", "b", 1], "mixed-integer"),
        (["a", "b", 1.3], "mixed"),
        (["a", "b", 1.3, 1], "mixed-integer"),
        (["aa", datetime(2011, 1, 1)], "mixed"),
    ],
)
def test_index_str_accessor_visibility(values, inferred_type, index_or_series):
    # 调用传入的 index_or_series 函数，创建一个对象
    obj = index_or_series(values)
    # 如果 index_or_series 是 Index 类型，则断言 obj 的推断类型等于 inferred_type
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    # 断言 obj 的 str 属性是 StringMethods 类的实例
    assert isinstance(obj.str, StringMethods)
# 使用 pytest 的 parametrize 装饰器来多次运行同一个测试函数，每次使用不同的参数
@pytest.mark.parametrize(
    "values,inferred_type",
    [
        ([1, np.nan], "floating"),  # 测试非字符串值引发 AttributeError 异常
        ([datetime(2011, 1, 1)], "datetime64"),  # 测试 datetime64 类型
        ([timedelta(1)], "timedelta64"),  # 测试 timedelta64 类型
    ],
)
def test_index_str_accessor_non_string_values_raises(
    values, inferred_type, index_or_series
):
    # 根据传入的参数创建对象
    obj = index_or_series(values)
    # 如果 index_or_series 是 Index 类型，则断言 inferred_type 属性与预期类型相同
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    # 准备错误消息
    msg = "Can only use .str accessor with string values"
    # 使用 pytest.raises 断言特定的异常和消息会被抛出
    with pytest.raises(AttributeError, match=msg):
        obj.str


def test_index_str_accessor_multiindex_raises():
    # 创建 MultiIndex 对象，含有混合的数据类型
    idx = MultiIndex.from_tuples([("a", "b"), ("a", "b")])
    # 断言 MultiIndex 对象的 inferred_type 属性为 "mixed"
    assert idx.inferred_type == "mixed"

    # 准备错误消息
    msg = "Can only use .str accessor with Index, not MultiIndex"
    # 使用 pytest.raises 断言特定的异常和消息会被抛出
    with pytest.raises(AttributeError, match=msg):
        idx.str


def test_str_accessor_no_new_attributes(any_string_dtype):
    # 创建 Series 对象，其元素为字符列表，数据类型由参数决定
    ser = Series(list("aabbcde"), dtype=any_string_dtype)
    # 准备错误消息
    with pytest.raises(AttributeError, match="You cannot add any new attribute"):
        # 尝试为 str 属性添加新的属性
        ser.str.xlabel = "a"


def test_cat_on_bytes_raises():
    # 创建 Series 对象，其元素为字节数组
    lhs = Series(np.array(list("abc"), "S1").astype(object))
    rhs = Series(np.array(list("def"), "S1").astype(object))
    # 准备错误消息
    msg = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    # 使用 pytest.raises 断言特定的异常和消息会被抛出
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)


def test_str_accessor_in_apply_func():
    # 创建 DataFrame 对象，包含两列字符数据
    df = DataFrame(zip("abc", "def"))
    expected = Series(["A/D", "B/E", "C/F"])
    # 使用 apply 函数，对每行数据应用 lambda 函数并连接结果
    result = df.apply(lambda f: "/".join(f.str.upper()), axis=1)
    # 断言结果 Series 与预期 Series 相等
    tm.assert_series_equal(result, expected)


def test_zfill():
    # 创建 Series 对象，包含多种数据类型的元素
    value = Series(["-1", "1", "1000", 10, np.nan])
    expected = Series(["-01", "001", "1000", np.nan, np.nan], dtype=object)
    # 测试 str.zfill 方法，预期填充后的结果与预期 Series 相等
    tm.assert_series_equal(value.str.zfill(3), expected)

    value = Series(["-2", "+5"])
    expected = Series(["-0002", "+0005"])
    # 测试 str.zfill 方法，预期填充后的结果与预期 Series 相等
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_zfill_with_non_integer_argument():
    value = Series(["-2", "+5"])
    wid = "a"
    # 准备错误消息
    msg = f"width must be of integer type, not {type(wid).__name__}"
    # 使用 pytest.raises 断言特定的异常和消息会被抛出
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)


def test_zfill_with_leading_sign():
    value = Series(["-cat", "-1", "+dog"])
    expected = Series(["-0cat", "-0001", "+0dog"])
    # 测试 str.zfill 方法，预期填充后的结果与预期 Series 相等
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_get_with_dict_label():
    # 创建包含字典的 Series 对象
    s = Series(
        [
            {"name": "Hello", "value": "World"},
            {"name": "Goodbye", "value": "Planet"},
            {"value": "Sea"},
        ]
    )
    # 使用 str.get 方法获取字典中指定键的值组成的 Series
    result = s.str.get("name")
    expected = Series(["Hello", "Goodbye", None], dtype=object)
    # 断言结果 Series 与预期 Series 相等
    tm.assert_series_equal(result, expected)
    # 再次使用 str.get 方法，获取字典中另一键的值组成的 Series
    result = s.str.get("value")
    # 创建一个预期的 Series 对象，包含字符串 "World", "Planet", "Sea"，数据类型为 object
    expected = Series(["World", "Planet", "Sea"], dtype=object)
    # 使用测试框架中的 assert_series_equal 函数比较 result 和 expected 两个 Series 对象
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试 Series 对象的字符串解码功能
def test_series_str_decode():
    # GH 22613，指示这是解决的GitHub问题编号
    # 创建一个 Series 对象，包含两个字节字符串 b"x" 和 b"y"，然后对其进行 UTF-8 编码解析，错误处理方式为严格模式
    result = Series([b"x", b"y"]).str.decode(encoding="UTF-8", errors="strict")
    # 创建一个期望的 Series 对象，包含两个字符串 "x" 和 "y"，数据类型为 object
    expected = Series(["x", "y"], dtype="object")
    # 使用测试框架中的 assert_series_equal 函数比较实际结果和期望结果的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```